#include "gl_loader.h"
#include "gpu_export_iosurface.h"
#include "fd_socket.h"
#include "../../plugin/Source/SharedGpuSurfaceProtocol.h"
#include "gpu_backend/frame_renderer_metal.h"
#include "renderer.h"
#include "media.h"
#include "visual_plan_executor.h"
#include "../src/canonical_block_c_frame.h"
#include "support/hdr_image_readback_checks.h"
#include "support/hlg_material_oracle.h"
#include "support/sdf_display_oracle.h"
#include "../src/hdr_movie_readback.h"

#include <GLFW/glfw3.h>
#include <CoreFoundation/CoreFoundation.h>
#include <IOSurface/IOSurface.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <cstring>
#include <unistd.h>
#include <utility>
#include <vector>

namespace
{

class ForeignDeviceHdrFrame final : public arbitgpu::NativeFixtureSceneFrame
{
public:
    explicit ForeignDeviceHdrFrame(std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> source)
        : source_(std::move(source)) {}
    const std::string& backend() const noexcept override { return source_->backend(); }
    std::uint32_t width() const noexcept override { return source_->width(); }
    std::uint32_t height() const noexcept override { return source_->height(); }
    std::uintptr_t colorImageHandle() const noexcept override { return source_->colorImageHandle(); }
    std::uintptr_t colorTextureViewHandle() const noexcept override { return source_->colorTextureViewHandle(); }
    arbitgpu::NativeTextureViewDescriptor colorTextureDescriptor() const noexcept override
    {
        auto descriptor = source_->colorTextureDescriptor();
        ++descriptor.deviceOrContextIdentity;
        return descriptor;
    }
private:
    std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> source_;
};

// Primary-standard oracle, independent of ColorTransformGpuMath and shaders.
std::array<double, 3> rec2020Nits(const std::array<float, 3>& rgb)
{
    constexpr double matrix[3][3] {{0.6274040, 0.3292820, 0.0433136},
        {0.0690970, 0.9195400, 0.0113612}, {0.0163916, 0.0880132, 0.8955950}};
    std::array<double, 3> result {};
    for (int c = 0; c < 3; ++c)
        for (int i = 0; i < 3; ++i) result[c] += 203 * matrix[c][i] * rgb[i];
    return result;
}

double pqNits(double value)
{
    const double p = std::pow(value, 32.0 / 2523.0);
    return 10000 * std::pow(std::max(p - 3424.0 / 4096.0, 0.0)
        / (2413.0 / 128.0 - 2392.0 / 128.0 * p), 16384.0 / 2610.0);
}

std::array<double, 3> hlgNits(const float* rgb)
{
    std::array<double, 3> scene {};
    for (int i = 0; i < 3; ++i)
        scene[i] = rgb[i] <= 0.5 ? rgb[i] * rgb[i] / 3.0
            : (std::exp((rgb[i] - 0.55991073) / 0.17883277) + 0.28466892) / 12;
    const double luminance = 0.2627 * scene[0] + 0.6780 * scene[1] + 0.0593 * scene[2];
    for (auto& value : scene) value *= 1000 * std::pow(luminance, 0.2);
    return scene;
}

bool hdrMoviePixelGuards(std::string& error)
{
    std::vector<float> input {4.92578125f, 4.92578125f, 4.92578125f, 1};
    std::vector<float> output;
    float storedCode = 0.751953125f;
    bool transformed = false;
    const auto read = [&](auto& pixels, auto&) { pixels = input; return true; };
    const auto transform = [&](const auto&, auto& pixels, auto&) {
        transformed = true; pixels = {storedCode, storedCode, storedCode, 1}; return true;
    };
    const auto pq = [&] { return videorender::hdrmovie::readPq(1000, {1, 1}, read, transform, output, error); };
    if (!pq() || std::abs(pqNits(output[0]) - 999.93359375) > 2)
    { error = "PQ binary16 guard rejected a representable in-range near-peak pixel"; return false; }
    storedCode = 0.75244140625f;
    if (pq() || !output.empty())
    { error = "PQ guard admitted the next binary16 code above its justified bound"; return false; }
    storedCode = 0.751953125f;
    for (const float invalid : {-0.01f, 4.9296875f,
        std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()})
    {
        input[0] = invalid; transformed = false;
        if (pq() || transformed || !output.empty()
            || videorender::hdrmovie::readHlg(1000, read, output, error) || !output.empty())
        { error = "HDR movie guard admitted negative/nonfinite/over-peak source pixels"; return false; }
    }
    input = {1, 1, 1, 0.5f}; transformed = false;
    if (pq() || transformed || !output.empty()
        || videorender::hdrmovie::readHlg(1000, read, output, error) || !output.empty())
    { error = "HDR movie guard admitted nonopaque source pixels"; return false; }
    error.clear(); return true;
}

bool malformedStreamFdCount(std::string& error)
{
    FdSocketServer server;
    std::string path;
    if (!server.listen(path, error)) return false;
    const int client = ::socket(AF_UNIX, SOCK_STREAM, 0);
    sockaddr_un address {};
    address.sun_family = AF_UNIX;
    if (client < 0 || path.size() >= sizeof(address.sun_path))
    { if (client >= 0) ::close(client); error = "Stream framing fixture could not create client"; return false; }
    std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
    if (::connect(client, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0 || !server.poll())
    { ::close(client); error = "Stream framing fixture could not connect"; return false; }
    gpusurf::MsgHeader malformed;
    malformed.type = static_cast<uint32_t>(gpusurf::MsgType::LinearIOSurfaceSelect);
    malformed.fdCount = 1; // This stream cannot carry ancillary handles.
    const bool sent = ::send(client, &malformed, sizeof(malformed), 0) == static_cast<ssize_t>(sizeof(malformed));
    FdSocketServer::Received message;
    const bool rejected = server.recvMsg(message) == FdSocketServer::RecvResult::Disconnected && !server.hasClient();
    ::close(client);
    if (!sent || !rejected) error = "Stream accepted a forged declared fd count";
    return sent && rejected;
}

bool retainedGeometryLinearFrames(int width, int height, std::string& error)
{
    using namespace videowire::geometry;
    PortContract contract;
    contract.carrier = CarrierKind::geometry3D;
    contract.maxVertices = 4; contract.maxIndices = 6;
    auto grid = lowerGrid(9501, 9501, 1, 2, 2, 1, contract, error);
    if (!grid) return false;
    const auto admitted = admitValue(*grid, contract, error);
    if (!admitted) return false;
    videowire::CompiledVisualLayerPlan plan;
    plan.clipId = 9501; plan.structuralRevision = 1;
    plan.nodeKinds = {"geometry.core.runtime"}; plan.nodeIds = {9501};
    plan.ports = {{9501, 1, 1, "out", "frame", "image", "rgba8", "sRGB"}};
    plan.operations = {{9501, "geometry.core.runtime", "native-gpu",
        encodeLoweredPlanText(lowerRuntimePlan(contract, *admitted))}};
    const videowire::VisualPlanEvaluationContext context {9501, false,
        visualtemporalsampling::EvaluationMode::preview, 9502, 9503};
    videorender::LayerDesc first, cached;
    if (!videowire::admitGeometryCoreOperations(plan, videohelper::geometry::PlanUse::preview,
            &context, width, height, first, error, 0, true)
        || !videowire::admitGeometryCoreOperations(plan, videohelper::geometry::PlanUse::preview,
            &context, width, height, cached, error, 0, true)) return false;
    if (!first.nativeLinearImage || !cached.nativeLinearImage || !first.nativeTextureOwner
        || first.nativeTextureOwner != cached.nativeTextureOwner
        || first.nativeLinearImage != cached.nativeLinearImage
        || !arbitgpu::isLinearSceneColor(cached.nativeTextureDescriptor)
        || first.nativeLinearImage->colorTextureDescriptor().rendererGeneration
            != cached.nativeTextureDescriptor.rendererGeneration)
    { error = "New/cached Geometry Core publication lost its exact retained linear frame"; return false; }
    auto sdr = first;
    ++plan.clipId;
    if (!videowire::admitGeometryCoreOperations(plan, videohelper::geometry::PlanUse::preview,
            &context, width, height, sdr, error, 0, false) || sdr.nativeLinearImage
        || arbitgpu::isLinearSceneColor(sdr.nativeTextureDescriptor))
    { error = "SDR Geometry Core publication kept a stale linear frame"; return false; }
    return true;
}

bool linearSceneTransport(videorender::MetalFrameRenderer& renderer,
                          const videorender::LayerDesc& layer, int width, int height,
                          const std::filesystem::path& consumer, std::string& error)
{
    gpuexp::IOSurfaceExporter exporter;
    gpuexp::ExportedBuffer buffer;
    if (!exporter.initialize(nullptr, error)
        || !exporter.allocate(buffer, width, height, error, gpuexp::IOSurfaceFormat::LinearRgba16Float))
        return false;
    struct Cleanup { gpuexp::IOSurfaceExporter& exporter; gpuexp::ExportedBuffer& buffer;
        ~Cleanup() { exporter.destroy(buffer); } } cleanup {exporter, buffer};
    const auto draw = [&](const videorender::LayerDesc& source, void* target, bool linear = true) {
        return renderer.renderCompositeToIOSurface(nullptr, target, width, height, &source, 1, nullptr, 0, 0, linear);
    };
    if (!draw(layer, buffer.surface)) { error = renderer.lastError(); return false; }
    const auto surface = static_cast<IOSurfaceRef>(buffer.surface);
    if (IOSurfaceGetPixelFormat(surface) != gpusurf::kIOSurfaceRGBAHalf
        || IOSurfaceGetBytesPerElement(surface) != 8
        || IOSurfaceGetBytesPerRow(surface) != buffer.rowBytes
        || IOSurfaceGetAllocSize(surface) != buffer.allocationBytes)
    { error = "Float IOSurface allocator changed its declared storage"; return false; }
    // The importer is a separate process and decodes IEEE binary16 itself.
    // Its completion handler returns the generation-tagged slot only after GPU readback.
    int releasePipe[2];
    if (::pipe(releasePipe) != 0) { error = "Float consumer release pipe failed"; return false; }
    const auto id = std::to_string(IOSurfaceGetID(surface));
    const auto w = std::to_string(width), h = std::to_string(height);
    const auto registry = std::to_string(renderer.metalDeviceRegistryId());
    const auto stride = std::to_string(buffer.rowBytes), allocation = std::to_string(buffer.allocationBytes);
    const auto releaseFd = std::to_string(releasePipe[1]);
    const auto pid = ::fork();
    if (pid == 0)
    {
        ::close(releasePipe[0]);
        ::execl(consumer.c_str(), consumer.c_str(), id.c_str(), w.c_str(), h.c_str(), "linear",
            registry.c_str(), stride.c_str(), allocation.c_str(), releaseFd.c_str(), "17", nullptr);
        ::_exit(127);
    }
    ::close(releasePipe[1]);
    int status = 0;
    const bool exited = pid > 0 && ::waitpid(pid, &status, 0) == pid;
    gpusurf::LinearIOSurfaceFrameReleasePayload release;
    const auto received = ::read(releasePipe[0], &release, sizeof(release));
    ::close(releasePipe[0]);
    if (!exited || !WIFEXITED(status) || WEXITSTATUS(status) != 0
        || received != static_cast<ssize_t>(sizeof(release)) || !gpusurf::validLinearRelease(release, 17, 3))
    { error = "Independent float IOSurface consumer or GPU completion release failed"; return false; }
    auto bad = layer;
    ++bad.nativeTextureDescriptor.rendererGeneration;
    if (draw(bad, buffer.surface)) { error = "Float transport admitted stale frame generation"; return false; }
    bad = layer; ++bad.nativeTextureDescriptor.deviceOrContextIdentity;
    if (draw(bad, buffer.surface)) { error = "Float transport admitted foreign native device"; return false; }
    bad = layer; bad.nativeTextureOwner.reset();
    if (draw(bad, buffer.surface)) { error = "Float transport admitted missing native owner"; return false; }
    bad = layer; bad.nativeLinearImage.reset();
    if (draw(bad, buffer.surface)) { error = "Float transport admitted unretained source"; return false; }
    bad = layer; bad.opacity = 0.5f;
    if (draw(bad, buffer.surface)) { error = "Float transport admitted nonopaque layer"; return false; }
    bad = layer; bad.graphColorTransformActive = true;
    if (draw(bad, buffer.surface)) { error = "Float transport admitted undeclared linear transform"; return false; }
    if (draw(layer, buffer.surface, false)) { error = "v2 output reinterpreted a float IOSurface"; return false; }
    renderer.setHdrImageCapture(false);
    if (draw(layer, buffer.surface)) { error = "Float output admitted unselected capture mode"; return false; }
    renderer.setHdrImageCapture(true);
    if (!draw(layer, buffer.surface)) { error = renderer.lastError(); return false; }
    renderer.clearDirectOutputs();
    return true;
}

bool metalHdrMovie(videorender::MetalFrameRenderer& renderer, IOSurfaceRef surface,
                   int width, int height, const std::filesystem::path& consumer, std::string& error)
{
    using namespace HarmonicMIDI::grid;
    if (!hdrMoviePixelGuards(error) || !malformedStreamFdCount(error)
        || !retainedGeometryLinearFrames(width, height, error)) return false;
    constexpr std::array<std::array<float, 3>, 4> colors {{{1, 1, 1},
        {4.92578125f, 4.92578125f, 4.92578125f}, {2, 0.25f, 1}, {0.01f, 0.04f, 0.125f}}};
    auto scene = std::make_shared<Visual3DScene>(*linearscenechecks::scene());
    scene->objectCount = scene->materialCount = 4;
    scene->vertexCount = 16; scene->indexCount = 24;
    scene->textureCount = scene->textureTexelCount = 0; scene->textureTexels.clear();
    for (int q = 0; q < 4; ++q)
    {
        const float left = q % 2 == 0 ? -2 : 0, right = left + 2;
        const float top = q / 2 == 0 ? 2 : 0, bottom = top - 2;
        const std::array<std::array<float, 3>, 4> positions {{{left, top, 0}, {right, top, 0},
            {left, bottom, 0}, {right, bottom, 0}}};
        for (int v = 0; v < 4; ++v)
            scene->vertices[q * 4 + v] = {{positions[v][0], positions[v][1], 0}, {0, 0, 1}, {}, {}};
        const std::array<std::uint32_t, 6> indices {0, 2, 1, 1, 2, 3};
        for (int i = 0; i < 6; ++i) scene->indices[q * 6 + i] = indices[i];
        scene->objects[q] = {};
        scene->objects[q].id = SceneObjectId {static_cast<std::uint32_t>(q + 10)};
        scene->objects[q].material = SceneMaterialId {static_cast<std::uint32_t>(q + 20)};
        scene->objects[q].firstVertex = q * 4; scene->objects[q].vertexCount = 4;
        scene->objects[q].firstIndex = q * 6; scene->objects[q].indexCount = 6;
        scene->materials[q] = {};
        scene->materials[q].id = scene->objects[q].material;
        scene->materials[q].baseColor = {};
        scene->materials[q].emissive = {colors[q][0], colors[q][1], colors[q][2]};
        scene->materials[q].doubleSided = true;
    }
    auto& backend = arbitgpu::nativeFixtureSceneBackend();
    const auto prepared = backend.prepare(scene, nullptr);
    arbitgpu::NativeFixtureSceneRuntimeInputs inputs; inputs.linearColor = true;
    const auto rendered = prepared.prepared
        ? backend.render(scene, prepared.resources, width, height, inputs) : arbitgpu::NativeFixtureSceneSubmission {};
    if (!rendered.rendered || !rendered.frame)
    { error = "Asymmetric native HDR scene failed: " + prepared.error + rendered.error; return false; }
    videorender::LayerDesc layer;
    layer.nativeTextureDescriptor = rendered.frame->colorTextureDescriptor();
    layer.nativeTextureBackend = rendered.frame->backend();
    layer.nativeTextureView = rendered.frame->colorTextureViewHandle();
    layer.nativeTextureOwner = rendered.frame; layer.nativeLinearImage = rendered.frame;
    layer.texWidth = width; layer.texHeight = height;
    const auto draw = [&](const auto* layers, int count, const auto* overlays, int overlayCount) {
        return renderer.renderCompositeToIOSurface(nullptr, surface, width, height,
            layers, count, overlays, overlayCount);
    };
    const auto rejectedReads = [&] {
        std::vector<float> pixels;
        return !renderer.readLastCompositePq(1000, pixels, error) && pixels.empty()
            && !renderer.readLastCompositeHlg(1000, pixels, error) && pixels.empty();
    };
    renderer.setHdrImageCapture(true);
    if (!linearSceneTransport(renderer, layer, width, height, consumer, error)) return false;
    if (!rejectedReads()) { error = "Metal movie accepted stale capture"; return false; }
    if (!draw(&layer, 1, static_cast<const videorender::ImageLayerDesc*>(nullptr), 0)) return false;
    std::vector<float> linear, pq, hlg, repeat;
    if (!renderer.readLastCompositeFloat(linear, error)
        || !renderer.readLastCompositePq(1000, pq, error)
        || !renderer.readLastCompositeHlg(1000, hlg, error)
        || !renderer.readLastCompositeFloat(repeat, error) || repeat != linear
        || pq.size() != linear.size() || hlg.size() != linear.size()) return false;
    for (int q = 0; q < 4; ++q)
    {
        const int x = (q % 2 == 0 ? 1 : 3) * width / 4;
        const int y = (q / 2 == 0 ? 1 : 3) * height / 4;
        const auto index = (static_cast<std::size_t>(y) * width + x) * 4;
        const auto expected = rec2020Nits(colors[q]);
        const auto hlgActual = hlgNits(hlg.data() + index);
        for (int c = 0; c < 3; ++c)
            if (std::abs(linear[index + c] - colors[q][c]) > 0.003
                || std::abs(pqNits(pq[index + c]) - expected[c]) > std::max(1.0, expected[c] * 0.004)
                || std::abs(hlgActual[c] - expected[c]) > std::max(0.1, expected[c] * 0.002))
            { error = "Metal HDR movie changed native scene primaries, magnitude or top-first orientation"; return false; }
        if (pq[index + 3] != 1 || hlg[index + 3] != 1) return false;
    }
    // The 999.93359375-nit scene patch must survive PQ's upward binary16 rounding.
    const auto peakIndex = (static_cast<std::size_t>(height / 4) * width + 3 * width / 4) * 4;
    if (pq[peakIndex] != 0.751953125f)
    { error = "Metal PQ near-peak binary16 storage differs"; return false; }
    for (const float peak : {399.0f, 10001.0f, std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::infinity(), 500.0f})
        if (renderer.readLastCompositePq(peak, repeat, error) || !repeat.empty()
            || renderer.readLastCompositeHlg(peak, repeat, error) || !repeat.empty())
        { error = "Metal HDR accepted an invalid peak or out-of-range native pixel"; return false; }
    const auto rejectLayer = [&](const videorender::LayerDesc& bad) {
        return !draw(&bad, 1, static_cast<const videorender::ImageLayerDesc*>(nullptr), 0) && rejectedReads();
    };
    auto bad = layer; ++bad.nativeTextureDescriptor.rendererGeneration;
    if (!rejectLayer(bad)) { error = "Metal HDR accepted stale generation"; return false; }
    bad = layer; ++bad.nativeTextureDescriptor.deviceOrContextIdentity;
    if (!rejectLayer(bad)) { error = "Metal HDR accepted foreign device"; return false; }
    bad = layer; bad.nativeTextureOwner.reset();
    if (!rejectLayer(bad)) { error = "Metal HDR accepted missing owner"; return false; }
    bad = layer; bad.nativeLinearImage.reset();
    if (!rejectLayer(bad)) { error = "Metal HDR accepted unretained linear source"; return false; }
    bad = layer; bad.nativeTextureDescriptor.transfer = colortransform::TransferFunction::Unspecified;
    const std::array<videorender::LayerDesc, 2> mixed {layer, bad};
    if (draw(mixed.data(), 2, static_cast<const videorender::ImageLayerDesc*>(nullptr), 0) || !rejectedReads()) return false;
    for (int guard = 0; guard < 4; ++guard)
    {
        bad = layer;
        if (guard == 0) bad.blendMode = 1;
        if (guard == 1) bad.transitionType = 1;
        if (guard == 2) bad.graphColorTransformActive = true;
        if (guard == 3) bad.depthFog = true;
        if (!rejectLayer(bad)) { error = "Metal HDR admitted undeclared effect/transition"; return false; }
    }
    videorender::EffectSlotState effect; effect.enabled = true;
    bad = layer; bad.effects = &effect; bad.effectCount = 1;
    if (!rejectLayer(bad)) { error = "Metal HDR admitted an undeclared SDR effect"; return false; }
    const videorender::ImageLayerDesc overlay {};
    if (draw(&layer, 1, &overlay, 1) || !rejectedReads()) return false;
    renderer.setBackgroundColor(0, 0, 0, 0);
    bad = layer; bad.opacity = 0.5f;
    if (!draw(&bad, 1, static_cast<const videorender::ImageLayerDesc*>(nullptr), 0) || !rejectedReads())
    { error = "Metal HDR accepted nonopaque final pixels"; return false; }
    renderer.setBackgroundColor(0, 0, 0, 1);
    if (!draw(&layer, 1, static_cast<const videorender::ImageLayerDesc*>(nullptr), 0)) return false;
    if (renderer.renderCompositeToIOSurface(nullptr, nullptr, width, height, &layer, 1, nullptr, 0)
        || !rejectedReads())
    { error = "Metal HDR kept admission after an invalid direct render"; return false; }
    if (!draw(&layer, 1, static_cast<const videorender::ImageLayerDesc*>(nullptr), 0)) return false;
    renderer.clearDirectOutputs();
    if (!rejectedReads()) { error = "Metal HDR kept admission after direct-output cleanup"; return false; }
    if (!draw(&layer, 1, static_cast<const videorender::ImageLayerDesc*>(nullptr), 0)) return false;
    renderer.setOutputSize(nullptr, width + 2, height + 2);
    if (!rejectedReads()) { error = "Metal HDR kept admission after extent change"; return false; }
    renderer.setOutputSize(nullptr, width, height);
    renderer.setHdrImageCapture(false);
    if (!rejectedReads()) return false;
    error.clear(); return true;
}

std::array<uint8_t, 4> readCenter (arbitgl::GlFuncs& gl, unsigned texture,
                                   int width, int height)
{
    unsigned fbo = 0;
    gl.GenFramebuffers (1, &fbo);
    gl.BindFramebuffer (GL_FRAMEBUFFER, fbo);
    gl.FramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_2D, texture, 0);
    std::array<uint8_t, 4> pixel = {};
    glReadPixels (width / 2, height / 2, 1, 1,
                  GL_RGBA, GL_UNSIGNED_BYTE, pixel.data());
    gl.BindFramebuffer (GL_FRAMEBUFFER, 0);
    gl.DeleteFramebuffers (1, &fbo);
    return pixel;
}

std::array<uint8_t, 4> readPixel (arbitgl::GlFuncs& gl, unsigned texture,
                                  int x, int y)
{
    unsigned fbo = 0;
    gl.GenFramebuffers (1, &fbo);
    gl.BindFramebuffer (GL_FRAMEBUFFER, fbo);
    gl.FramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_2D, texture, 0);
    std::array<uint8_t, 4> pixel = {};
    glReadPixels (x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel.data());
    gl.BindFramebuffer (GL_FRAMEBUFFER, 0);
    gl.DeleteFramebuffers (1, &fbo);
    return pixel;
}

bool sdfDisplay(videorender::MetalFrameRenderer& renderer, arbitgl::GlFuncs& gl,
                IOSurfaceRef surface, std::string& error)
{
    using namespace videohelper::sdf;
    NativeSdfRenderer native(arbitgpu::nativeSdfExecutionBackend());
    renderer.setHdrImageCapture(false);
    for (const std::uint8_t output : {std::uint8_t{0}, std::uint8_t{1}})
        for (const auto use : {NativeSdfRenderUse::Preview, NativeSdfRenderUse::Export})
        {
            const auto plan = sdfdisplayoracle::plan(output);
            videorender::LayerDesc layer;
            std::vector<NativeSdfRenderedFrame> owners;
            if (prepareVisualSdfLayer({plan}, plan.clipId, 64, 36, use, native, layer, owners, error)
                    != VisualSdfPreparation::rendered || owners.size() != 1) return false;
            auto source = owners.front().nativeFrame;
            std::vector<float> rawBefore, rawAfter;
            if (!source->readColorFloatPixels(rawBefore)) return false;
            owners.clear();
            const auto texture = renderer.renderComposite(&gl, &layer, 1, nullptr, 0);
            if (!texture || !sdfdisplayoracle::verify([&](int x, int y) {
                    return readPixel(gl, texture, x, y);
                }, output == 1, error)) return false;
            // Ordinary video export receives the same SDR IOSurface, without
            // enabling HDR capture or relabelling the linear source texture.
            if (!renderer.renderCompositeToIOSurface(&gl, surface, 64, 36, &layer, 1, nullptr, 0))
            { error = renderer.lastError(); return false; }
            IOSurfaceLock(surface, kIOSurfaceLockReadOnly, nullptr);
            const auto* bytes = static_cast<const std::uint8_t*>(IOSurfaceGetBaseAddress(surface));
            const auto stride = IOSurfaceGetBytesPerRow(surface);
            const bool exportOk = bytes && sdfdisplayoracle::verify([&](int x, int y) {
                const auto* pixel = bytes + y * stride + x * 4;
                return std::array<std::uint8_t,4>{pixel[2], pixel[1], pixel[0], pixel[3]};
            }, output == 1, error);
            IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, nullptr);
            if (!exportOk) return false;
            renderer.setHdrImageCapture(true);
            std::vector<float> hdr;
            const bool captured = renderer.renderComposite(&gl, &layer, 1, nullptr, 0) != 0
                && renderer.readLastCompositeFloat(hdr, error);
            renderer.setHdrImageCapture(false);
            if (!captured || hdr.size() != rawBefore.size()) return false;
            for (std::size_t i = 0; i < hdr.size(); ++i)
                if (std::abs(hdr[i] - rawBefore[i]) > 0.002f)
                { error = "Metal SDF HDR capture used the display copy"; return false; }
            if (!source->readColorFloatPixels(rawAfter) || rawAfter != rawBefore
                || !arbitgpu::isLinearSceneColor(layer.nativeTextureDescriptor))
            { error = "Metal SDF display mutated its half-float source"; return false; }
            auto stale = layer;
            ++stale.nativeTextureDescriptor.rendererGeneration;
            if (renderer.renderComposite(&gl, &stale, 1, nullptr, 0) != 0)
            { error = "Metal SDF display accepted a stale source generation"; return false; }
        }
    const auto fogPlan = sdfdisplayoracle::plan(0, true);
    std::array<std::array<uint8_t, 4>, 2> fogPreview {};
    for (const auto use : {NativeSdfRenderUse::Preview, NativeSdfRenderUse::Export})
    {
        videorender::LayerDesc layer;
        std::vector<NativeSdfRenderedFrame> owners;
        if (prepareVisualSdfLayer({fogPlan}, fogPlan.clipId, 64, 36, use,
                native, layer, owners, error) != VisualSdfPreparation::rendered
            || owners.size() != 1 || !layer.depthFog || layer.depthEffect != 1
            || layer.nativeDepthTextureDescriptor.format != arbitgpu::NativeTexturePixelFormat::R32Float)
        { error = "Metal SDF Depth Fog lost its typed raw attachment"; return false; }
        std::vector<float> depth;
        if (!owners.front().nativeFrame->readDepthFloatPixels(depth)
            || depth.size() != 64u * 36u
            || std::abs(depth[12u * 64u + 32u] - 0.327399042269f) > 0.002f
            || depth[24u * 64u + 32u] != 1.0f)
        { error = "Metal SDF Depth Fog received wrong analytic depth"; return false; }
        owners.clear();
        const auto texture = renderer.renderComposite(&gl, &layer, 1, nullptr, 0);
        if (!texture) { error = renderer.lastError(); return false; }
        const auto upper = readPixel(gl, texture, 32, 12);
        const auto lower = readPixel(gl, texture, 32, 24);
        if (upper[0] <= 56 || lower[0] <= 46 || upper[0] >= lower[0] || lower[3] != 255)
        { error = "Metal SDF Depth Fog did not apply distinct hit and miss fog"; return false; }
        if (!renderer.renderCompositeToIOSurface(&gl, surface, 64, 36, &layer, 1, nullptr, 0))
        { error = renderer.lastError(); return false; }
        IOSurfaceLock(surface, kIOSurfaceLockReadOnly, nullptr);
        const auto* bytes = static_cast<const std::uint8_t*>(IOSurfaceGetBaseAddress(surface));
        const auto stride = IOSurfaceGetBytesPerRow(surface);
        const auto at = [&](int y) {
            const auto* p = bytes + static_cast<std::size_t>(y) * stride + 32 * 4;
            return std::array<uint8_t,4>{p[2], p[1], p[0], p[3]};
        };
        const bool exportSame = bytes != nullptr && at(12) == upper && at(24) == lower;
        IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, nullptr);
        if (!exportSame) { error = "Metal SDF Depth Fog preview and export disagree"; return false; }
        auto malformed = layer;
        ++malformed.nativeDepthTextureDescriptor.deviceOrContextIdentity;
        if (renderer.renderComposite(&gl, &malformed, 1, nullptr, 0) != 0)
        { error = "Metal SDF Depth Fog accepted a foreign Depth owner"; return false; }
        const std::array<std::array<uint8_t, 4>, 2> pixels {upper, lower};
        if (use == NativeSdfRenderUse::Preview) fogPreview = pixels;
        else if (pixels != fogPreview)
        { error = "Metal SDF Depth Fog changes with render use"; return false; }
    }
    return true;
}

bool isRed (const std::array<uint8_t, 4>& pixel)
{
    return pixel[0] > 220 && pixel[1] < 20 && pixel[2] < 20 && pixel[3] > 220;
}

bool isBlue (const std::array<uint8_t, 4>& pixel)
{
    return pixel[0] < 20 && pixel[1] < 20 && pixel[2] > 220 && pixel[3] > 220;
}

bool isCyan (const std::array<uint8_t, 4>& pixel)
{
    return pixel[0] < 20 && pixel[1] > 220 && pixel[2] > 220 && pixel[3] > 220;
}

bool isGreen (const std::array<uint8_t, 4>& pixel)
{
    return pixel[0] < 20 && pixel[1] > 220 && pixel[2] < 20 && pixel[3] > 220;
}

bool isMagenta (const std::array<uint8_t, 4>& pixel)
{
    return pixel[0] > 220 && pixel[1] < 20 && pixel[2] > 220 && pixel[3] > 220;
}

bool isYellow (const std::array<uint8_t, 4>& pixel)
{
    return pixel[0] > 220 && pixel[1] > 220 && pixel[2] < 20 && pixel[3] > 220;
}

void setSurfaceInt (CFMutableDictionaryRef properties, CFStringRef key, int value)
{
    CFNumberRef number = CFNumberCreate (
        kCFAllocatorDefault, kCFNumberIntType, &value);
    CFDictionarySetValue (properties, key, number);
    CFRelease (number);
}

videowire::CompiledVisualLayerPlan typedMattePlan()
{
    videowire::CompiledVisualLayerPlan plan;
    plan.clipId = 501;
    plan.producerValidated = true;
    plan.nodeKinds = { "video.source", "visual.matte.asset", "visual.matte.refine",
                       "visual.matte.apply", "video.out" };
    plan.nodeIds = { 11, 41, 42, 43, 12 };
    plan.edges = { { 11, 0, 43, 0 }, { 41, 0, 42, 0 },
                   { 42, 1, 43, 1 }, { 43, 2, 12, 0 } };
    plan.operations = {
        { 11, "video.source", "source-decode", "" },
        { 41, "visual.matte.asset", "source-decode",
          "<MatteAssetBinding matteAssetId=\"metal-smoke-matte\" state=\"available\" "
          "cacheKey=\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\" "
          "contentReceipt=\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\" "
          "framePrefix=\"subject-\" frameExtension=\".rgba\" firstFrame=\"42\" "
          "frameDigits=\"3\" backend=\"rgba-cpu-decode-native-gpu-upload\"/>" },
        { 42, "visual.matte.refine", "native-gpu",
          "<NodeParams invert=\"1\" black=\"0.2\" white=\"0.8\" "
          "erodeDilate=\"1\" feather=\"1\" choke=\"0.1\"/>" },
        { 43, "visual.matte.apply", "native-gpu", "" },
        { 12, "video.out", "native-gpu", "" }
    };
    return plan;
}


} // namespace

int main (int argc, char** argv)
{
    constexpr int width = 64;
    constexpr int height = 32;
    if (glfwInit() != GLFW_TRUE)
    {
        std::cerr << "GLFW initialization failed\n";
        return 1;
    }
    glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint (GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    glfwWindowHint (GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow (width, height,
                                           "Arbit Metal frame compositor smoke",
                                           nullptr, nullptr);
    if (window == nullptr)
    {
        std::cerr << "OpenGL 4.1 context creation failed\n";
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent (window);

    arbitgl::GlFuncs gl;
    std::string error;
    if (! arbitgl::loadGlFunctions (gl, error))
    {
        std::cerr << "OpenGL loader failed: " << error << '\n';
        glfwDestroyWindow (window);
        glfwTerminate();
        return 1;
    }

    videorender::MetalFrameRenderer renderer;
    if (! renderer.initialize (&gl, width, height, error))
    {
        std::cerr << "Metal compositor initialization failed: " << error << '\n';
        glfwDestroyWindow (window);
        glfwTerminate();
        return 1;
    }
    renderer.setCanvas (width, height);
    renderer.setPresentSize (width, height);
    renderer.setBackgroundColor (0, 0, 0, 1);

    {
        videorender::MetalFrameRenderer sdfRenderer;
        // Keep the viewport compositor alive while another owner initializes,
        // as a frame probe does in the application.
        if (!sdfRenderer.initialize(&gl, 64, 36, error))
        {
            std::cerr << "Concurrent Metal compositor initialization failed: " << error << '\n';
            renderer.shutdown(&gl);
            glfwDestroyWindow(window);
            glfwTerminate();
            return 1;
        }
        auto properties = CFDictionaryCreateMutable(kCFAllocatorDefault, 0,
            &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
        setSurfaceInt(properties, kIOSurfaceWidth, 64);
        setSurfaceInt(properties, kIOSurfaceHeight, 36);
        setSurfaceInt(properties, kIOSurfaceBytesPerElement, 4);
        setSurfaceInt(properties, kIOSurfacePixelFormat, static_cast<int32_t>('BGRA'));
        const auto surface = IOSurfaceCreate(properties);
        CFRelease(properties);
        const bool passed = surface && sdfDisplay(sdfRenderer, gl, surface, error);
        sdfRenderer.shutdown(&gl);
        if (surface) CFRelease(surface);
        if (!passed) { std::cerr << "SDF Metal SDR composition: " << error << '\n'; return 1; }
    }

    if (!hdrimagechecks::readback(renderer,
            [&] { return renderer.renderComposite(&gl, nullptr, 0, nullptr, 0) != 0; }, error)
        || !hdrimagechecks::sceneComposite(renderer, [&](const auto* layers, int count)
            { return renderer.renderComposite(&gl, layers, count, nullptr, 0) != 0; }, error, width, height))
    {
        std::cerr << error << '\n';
        renderer.shutdown(&gl); glfwDestroyWindow(window); glfwTerminate(); return 1;
    }

    std::vector<uint8_t> red (static_cast<size_t> (width) * height * 4, 255);
    for (size_t i = 0; i < red.size(); i += 4)
    {
        red[i + 1] = 0;
        red[i + 2] = 0;
    }
    constexpr unsigned sourceHandle = 42;
    renderer.uploadRgba (sourceHandle, red.data(), width, height, width * 4);
    videorender::LayerDesc layer;
    layer.texture = sourceHandle;
    layer.texWidth = width;
    layer.texHeight = height;

    std::array<uint8_t, 4> pixel = {};
    for (int frame = 0; frame < 12; ++frame)
    {
        const unsigned output = renderer.renderComposite (&gl, &layer, 1, nullptr, 0);
        if (output == 0)
        {
            std::cerr << "Metal compositor rejected a supported frame: "
                      << renderer.lastError() << '\n';
            renderer.shutdown (&gl);
            glfwDestroyWindow (window);
            glfwTerminate();
            return 1;
        }
        pixel = readCenter (gl, output, width, height);
    }

    if (! isRed (pixel))
    {
        std::cerr << "Metal compositor did not preserve the uploaded red frame\n";
        renderer.deleteTexture (sourceHandle);
        renderer.shutdown (&gl);
        glfwDestroyWindow (window);
        glfwTerminate();
        return 1;
    }

    // Physical Apple Silicon oracle: this uses the production R16 owner and
    // typed layer boundary, then exercises deletion/stale-handle rejection.
    constexpr unsigned depthHandle = 43;
    std::vector<uint16_t> depth (static_cast<size_t> (width) * height, 65535);
    if (! renderer.uploadR16 (depthHandle, depth.data(), width, height))
    {
        std::cerr << "Metal R16 depth upload failed: " << renderer.lastError() << '\n';
        return 1;
    }
    layer.depthFog = true;
    layer.depthTexture = depthHandle;
    layer.depthWidth = width;
    layer.depthHeight = height;
    layer.fogNear = 0.0f;
    layer.fogFar = 1.0f;
    layer.fogDensity = 1.0f;
    layer.fogRed = 0.0f;
    layer.fogGreen = 0.0f;
    layer.fogBlue = 1.0f;
    layer.fogAlpha = 1.0f;
    unsigned fogOutput = renderer.renderComposite (&gl, &layer, 1, nullptr, 0);
    pixel = readCenter (gl, fogOutput, width, height);
    const float fogAmount = 1.0f - std::exp (-1.0f);
    const int expectedRed = static_cast<int> (std::lround (255.0f * (1.0f - fogAmount)));
    const int expectedBlue = static_cast<int> (std::lround (255.0f * fogAmount));
    if (fogOutput == 0 || std::abs (static_cast<int> (pixel[0]) - expectedRed) > 3
        || pixel[1] > 3 || std::abs (static_cast<int> (pixel[2]) - expectedBlue) > 3)
    {
        std::cerr << "Metal Depth Fog production pixel oracle failed\n";
        return 1;
    }
    for (int mode = 2; mode <= 4; ++mode)
    {
        layer.depthFog = false;
        layer.depthEffect = mode;
        layer.depthParam0 = mode == 2 ? 4.0f : (mode == 3 ? 0.1f : 1.0f);
        layer.depthParam1 = mode == 2 ? 0.0f : (mode == 3 ? 0.0f : 0.5f);
        layer.depthParam2 = mode == 2 ? 1.0f : 0.5f;
        layer.depthColorRed = layer.depthColorGreen = layer.depthColorBlue = 1.0f;
        if (renderer.renderComposite (&gl, &layer, 1, nullptr, 0) == 0)
        {
            std::cerr << "Metal depth primitive mode " << mode << " was rejected: "
                      << renderer.lastError() << '\n';
            return 1;
        }
    }
    renderer.deleteTexture (depthHandle);
    if (renderer.renderComposite (&gl, &layer, 1, nullptr, 0) != 0)
    {
        std::cerr << "Metal Depth Fog accepted a deleted/stale depth handle\n";
        return 1;
    }
    layer.depthFog = false;
    layer.depthEffect = 0;
    layer.depthTexture = 0;
    layer.depthWidth = layer.depthHeight = 0;

    std::vector<uint8_t> dimGray (static_cast<size_t> (width) * height * 4, 255);
    for (size_t i = 0; i < dimGray.size(); i += 4)
        dimGray[i] = dimGray[i + 1] = dimGray[i + 2] = 64;
    renderer.uploadRgba (sourceHandle, dimGray.data(), width, height, width * 4);
    renderer.setPostFx (0.0f, 1.0f, 0.0f, 0, 2.0f);
    unsigned postOutput = renderer.renderComposite (&gl, &layer, 1, nullptr, 0);
    pixel = readCenter (gl, postOutput, width, height);
    if (pixel[0] < 120 || pixel[0] > 136
        || pixel[1] < 120 || pixel[1] > 136 || pixel[2] < 120 || pixel[2] > 136)
    {
        std::cerr << "Metal HDR exposure did not double dim gray\n";
        return 1;
    }

    std::vector<uint8_t> bloomSource (static_cast<size_t> (width) * height * 4, 255);
    for (size_t i = 0; i < bloomSource.size(); i += 4)
        bloomSource[i] = bloomSource[i + 1] = bloomSource[i + 2] = 0;
    for (int y = 0; y < height; ++y)
        bloomSource[(static_cast<size_t> (y) * width + width / 2) * 4] = 255;
    renderer.uploadRgba (sourceHandle, bloomSource.data(), width, height, width * 4);
    renderer.setPostFx (1.0f, 0.1f, 5.0f, 0, 1.0f);
    postOutput = renderer.renderComposite (&gl, &layer, 1, nullptr, 0);
    pixel = readPixel (gl, postOutput, width / 2 + 3, height / 2);
    if (pixel[0] <= 5)
    {
        std::cerr << "Metal HDR bloom did not spread the bright line\n";
        return 1;
    }
    renderer.setPostFx (0.0f, 1.0f, 0.0f, 0, 1.0f);
    renderer.uploadRgba (sourceHandle, red.data(), width, height, width * 4);
    unsigned resetOutput = renderer.renderComposite (&gl, &layer, 1, nullptr, 0);
    pixel = readCenter (gl, resetOutput, width, height);
    if (! isRed (pixel))
    {
        std::cerr << "Metal source/post reset did not restore red; got "
                  << static_cast<int> (pixel[0]) << ','
                  << static_cast<int> (pixel[1]) << ','
                  << static_cast<int> (pixel[2]) << ','
                  << static_cast<int> (pixel[3]) << '\n';
        return 1;
    }

    videorender::EffectSlotState invert;
    invert.type = static_cast<int> (videofx::EffectType::Invert);
    invert.enabled = true;
    invert.params[0] = 1.0f;
    layer.effects = &invert;
    layer.effectCount = 1;
    for (int frame = 0; frame < 12 && ! isCyan (pixel); ++frame)
    {
        const unsigned output = renderer.renderComposite (&gl, &layer, 1, nullptr, 0);
        pixel = readCenter (gl, output, width, height);
    }
    if (! isCyan (pixel))
    {
        std::cerr << "Metal color effect suite did not invert red to cyan; got "
                  << static_cast<int> (pixel[0]) << ','
                  << static_cast<int> (pixel[1]) << ','
                  << static_cast<int> (pixel[2]) << ','
                  << static_cast<int> (pixel[3]) << '\n';
        renderer.deleteTexture (sourceHandle);
        renderer.shutdown (&gl);
        glfwDestroyWindow (window);
        glfwTerminate();
        return 1;
    }
    layer.effects = nullptr;
    layer.effectCount = 0;

    videorender::LayerDesc adjustment;
    adjustment.isAdjustment = true;
    adjustment.clipId = 99;
    adjustment.opacity = 1.0f;
    adjustment.effects = &invert;
    adjustment.effectCount = 1;
    const std::array<videorender::LayerDesc, 2> adjustedLayers = {
        layer, adjustment };
    unsigned adjustedOutput = renderer.renderComposite (
        &gl, adjustedLayers.data(), static_cast<int> (adjustedLayers.size()), nullptr, 0);
    pixel = readCenter (gl, adjustedOutput, width, height);
    if (! isCyan (pixel))
    {
        std::cerr << "Metal adjustment layer did not invert the composite beneath it\n";
        return 1;
    }

    constexpr unsigned ownedOverlayHandle = 46;
    std::vector<uint8_t> ownedBlue (static_cast<size_t> (width) * height * 4, 255);
    for (size_t i = 0; i < ownedBlue.size(); i += 4)
        ownedBlue[i] = ownedBlue[i + 1] = 0;
    renderer.uploadRgba (ownedOverlayHandle, ownedBlue.data(), width, height, width * 4);
    videorender::ImageLayerDesc ownedOverlay;
    ownedOverlay.texture = ownedOverlayHandle;
    ownedOverlay.width = width;
    ownedOverlay.height = height;
    ownedOverlay.ownerClipId = adjustment.clipId;
    unsigned overlayOnlyOutput = renderer.renderComposite (
        &gl, &layer, 1, &ownedOverlay, 1);
    pixel = readCenter (gl, overlayOnlyOutput, width, height);
    if (! isBlue (pixel))
    {
        std::cerr << "Metal owned overlay did not composite before adjustment; got "
                  << static_cast<int> (pixel[0]) << ','
                  << static_cast<int> (pixel[1]) << ','
                  << static_cast<int> (pixel[2]) << ','
                  << static_cast<int> (pixel[3]) << '\n';
        return 1;
    }
    for (int frame = 0; frame < 4 && ! isYellow (pixel); ++frame)
    {
        adjustedOutput = renderer.renderComposite (
            &gl, adjustedLayers.data(), static_cast<int> (adjustedLayers.size()),
            &ownedOverlay, 1);
        pixel = readCenter (gl, adjustedOutput, width, height);
    }
    if (! isYellow (pixel))
    {
        std::cerr << "Metal adjustment did not process its owned blue overlay; got "
                  << static_cast<int> (pixel[0]) << ','
                  << static_cast<int> (pixel[1]) << ','
                  << static_cast<int> (pixel[2]) << ','
                  << static_cast<int> (pixel[3]) << '\n';
        return 1;
    }
    renderer.deleteTexture (ownedOverlayHandle);

    constexpr unsigned lutHandle = 45;
    std::array<float, 24> magentaLut = {};
    for (size_t i = 0; i < magentaLut.size(); i += 3)
    {
        magentaLut[i] = 1.0f;
        magentaLut[i + 2] = 1.0f;
    }
    renderer.uploadLut3D (lutHandle, magentaLut.data(), 2);
    layer.lutTexture = lutHandle;
    layer.lutSize = 2;
    const unsigned lutOutput = renderer.renderComposite (&gl, &layer, 1, nullptr, 0);
    pixel = readCenter (gl, lutOutput, width, height);
    if (! isMagenta (pixel))
    {
        std::cerr << "Metal 3D LUT did not map the source to magenta; got "
                  << static_cast<int> (pixel[0]) << ','
                  << static_cast<int> (pixel[1]) << ','
                  << static_cast<int> (pixel[2]) << ','
                  << static_cast<int> (pixel[3]) << '\n';
        return 1;
    }
    layer.lutTexture = 0;
    layer.lutSize = 0;
    renderer.deleteTexture (lutHandle);

    std::vector<uint8_t> edge (static_cast<size_t> (width) * height * 4, 255);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
        {
            const size_t offset = (static_cast<size_t> (y) * width + x) * 4;
            const uint8_t value = x < width / 2 ? 255 : 0;
            edge[offset] = value;
            edge[offset + 1] = 0;
            edge[offset + 2] = 0;
        }
    renderer.uploadRgba (sourceHandle, edge.data(), width, height, width * 4);
    videorender::EffectSlotState blur;
    blur.type = static_cast<int> (videofx::EffectType::Blur);
    blur.enabled = true;
    blur.params[0] = 4.0f;
    layer.effects = &blur;
    layer.effectCount = 1;
    unsigned filteredOutput = renderer.renderComposite (&gl, &layer, 1, nullptr, 0);
    pixel = readPixel (gl, filteredOutput, width / 2, height / 2);
    if (pixel[0] <= 20 || pixel[0] >= 220)
    {
        std::cerr << "Metal Gaussian blur did not soften a hard edge; red="
                  << static_cast<int> (pixel[0]) << '\n';
        return 1;
    }

    std::vector<uint8_t> ridge (static_cast<size_t> (width) * height * 4, 255);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
        {
            const size_t offset = (static_cast<size_t> (y) * width + x) * 4;
            const uint8_t value = x == width / 2 ? 192 : 128;
            ridge[offset] = ridge[offset + 1] = ridge[offset + 2] = value;
        }
    renderer.uploadRgba (sourceHandle, ridge.data(), width, height, width * 4);
    videorender::EffectSlotState sharpen;
    sharpen.type = static_cast<int> (videofx::EffectType::Sharpen);
    sharpen.enabled = true;
    sharpen.params[0] = 1.0f;
    layer.effects = &sharpen;
    filteredOutput = renderer.renderComposite (&gl, &layer, 1, nullptr, 0);
    pixel = readPixel (gl, filteredOutput, width / 2, height / 2);
    if (pixel[0] <= 200)
    {
        std::cerr << "Metal unsharp mask did not increase ridge contrast; red="
                  << static_cast<int> (pixel[0]) << '\n';
        return 1;
    }

    std::vector<uint8_t> asymmetric (static_cast<size_t> (width) * height * 4, 255);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
        {
            const size_t offset = (static_cast<size_t> (y) * width + x) * 4;
            asymmetric[offset] = x < width / 2 ? 255 : 0;
            asymmetric[offset + 1] = x < width / 2 ? 0 : 255;
            asymmetric[offset + 2] = 0;
        }
    renderer.uploadRgba (sourceHandle, asymmetric.data(), width, height, width * 4);
    videorender::EffectSlotState geometryEffect;
    geometryEffect.type = static_cast<int> (videofx::EffectType::Mirror);
    geometryEffect.enabled = true;
    geometryEffect.params[0] = 0.0f;
    layer.effects = &geometryEffect;
    filteredOutput = renderer.renderComposite (&gl, &layer, 1, nullptr, 0);
    pixel = readPixel (gl, filteredOutput, width * 3 / 4, height / 2);
    if (! isRed (pixel))
    {
        std::cerr << "Metal mirror effect did not reflect the left half\n";
        return 1;
    }

    const std::array<videofx::EffectType, 8> statelessGeometry = {
        videofx::EffectType::Kaleidoscope, videofx::EffectType::Mirror,
        videofx::EffectType::Tile, videofx::EffectType::Warp,
        videofx::EffectType::Displace, videofx::EffectType::PolarSwirl,
        videofx::EffectType::DisplaceRgb, videofx::EffectType::Pixelate };
    for (const auto type : statelessGeometry)
    {
        geometryEffect = {};
        geometryEffect.type = static_cast<int> (type);
        geometryEffect.enabled = true;
        const auto* definition = videofx::effectDefFor (geometryEffect.type);
        for (int param = 0; definition != nullptr && param < definition->paramCount; ++param)
            geometryEffect.params[param] = definition->params[param].defaultValue;
        filteredOutput = renderer.renderComposite (&gl, &layer, 1, nullptr, 0);
        if (filteredOutput == 0)
        {
            std::cerr << "Metal compositor rejected stateless geometry effect "
                      << geometryEffect.type << '\n';
            return 1;
        }
    }

    renderer.uploadRgba (sourceHandle, red.data(), width, height, width * 4);
    videorender::EffectSlotState feedback;
    feedback.type = static_cast<int> (videofx::EffectType::FeedbackTrail);
    feedback.enabled = true;
    feedback.params[0] = 0.9f;
    feedback.params[1] = 1.0f;
    feedback.params[2] = 0.0f;
    layer.clipId = 7;
    layer.effects = &feedback;
    layer.effectCount = 1;
    renderer.renderComposite (&gl, &layer, 1, nullptr, 0);
    std::vector<uint8_t> black (static_cast<size_t> (width) * height * 4, 255);
    for (size_t i = 0; i < black.size(); i += 4)
        black[i] = black[i + 1] = black[i + 2] = 0;
    renderer.uploadRgba (sourceHandle, black.data(), width, height, width * 4);
    filteredOutput = renderer.renderComposite (&gl, &layer, 1, nullptr, 0);
    pixel = readCenter (gl, filteredOutput, width, height);
    if (pixel[0] < 200 || pixel[1] > 20 || pixel[2] > 20)
    {
        std::cerr << "Metal feedback trail did not retain the previous red frame\n";
        return 1;
    }
    renderer.uploadRgba (sourceHandle, red.data(), width, height, width * 4);
    layer.clipId = 0;
    layer.effects = nullptr;
    layer.effectCount = 0;
    renderer.renderComposite (&gl, &layer, 1, nullptr, 0);

    constexpr unsigned incomingHandle = 44;
    std::vector<uint8_t> green (static_cast<size_t> (width) * height * 4, 255);
    for (size_t i = 0; i < green.size(); i += 4)
    {
        green[i] = 0;
        green[i + 2] = 0;
    }
    renderer.uploadRgba (incomingHandle, green.data(), width, height, width * 4);
    const videorender::LayerDesc outgoing = layer;
    videorender::LayerDesc incoming = layer;
    incoming.texture = incomingHandle;
    incoming.transitionType = 2;
    incoming.fromLayer = &outgoing;
    incoming.transitionProgress = 0.0f;
    unsigned transitionOutput = renderer.renderComposite (&gl, &incoming, 1, nullptr, 0);
    pixel = readCenter (gl, transitionOutput, width, height);
    if (! isRed (pixel))
    {
        std::cerr << "Metal dissolve did not begin on the outgoing frame\n";
        return 1;
    }
    incoming.transitionProgress = 1.0f;
    transitionOutput = renderer.renderComposite (&gl, &incoming, 1, nullptr, 0);
    pixel = readCenter (gl, transitionOutput, width, height);
    if (! isGreen (pixel))
    {
        std::cerr << "Metal dissolve did not end on the incoming frame\n";
        return 1;
    }
    renderer.deleteTexture (incomingHandle);

    constexpr int overlayWidth = 16;
    constexpr int overlayHeight = 16;
    constexpr unsigned overlayHandle = 43;
    std::vector<uint8_t> blue (
        static_cast<size_t> (overlayWidth) * overlayHeight * 4, 255);
    for (size_t i = 0; i < blue.size(); i += 4)
    {
        blue[i] = 0;
        blue[i + 1] = 0;
    }
    renderer.uploadRgba (overlayHandle, blue.data(), overlayWidth, overlayHeight,
                         overlayWidth * 4);
    videorender::ImageLayerDesc overlay;
    overlay.texture = overlayHandle;
    overlay.width = overlayWidth;
    overlay.height = overlayHeight;
    for (int frame = 0; frame < 8; ++frame)
    {
        const unsigned output = renderer.renderComposite (&gl, &layer, 1, &overlay, 1);
        if (output == 0)
        {
            std::cerr << "Metal compositor rejected a supported overlay\n";
            renderer.deleteTexture (overlayHandle);
            renderer.deleteTexture (sourceHandle);
            renderer.shutdown (&gl);
            glfwDestroyWindow (window);
            glfwTerminate();
            return 1;
        }
        pixel = readCenter (gl, output, width, height);
    }

    renderer.uploadRgba (sourceHandle, red.data(), width, height, width * 4);
    renderer.renderComposite (&gl, &layer, 1, nullptr, 0);
    CFMutableDictionaryRef properties = CFDictionaryCreateMutable (
        kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks,
        &kCFTypeDictionaryValueCallBacks);
    setSurfaceInt (properties, kIOSurfaceWidth, width);
    setSurfaceInt (properties, kIOSurfaceHeight, height);
    setSurfaceInt (properties, kIOSurfaceBytesPerElement, 4);
    setSurfaceInt (properties, kIOSurfacePixelFormat, static_cast<int32_t> ('BGRA'));
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    CFDictionarySetValue (properties, kIOSurfaceIsGlobal, kCFBooleanTrue);
#pragma clang diagnostic pop
    IOSurfaceRef directSurface = IOSurfaceCreate (properties);
    CFRelease (properties);
    if (directSurface == nullptr
        || ! renderer.renderCompositeToIOSurface (
            &gl, directSurface, width, height, &layer, 1, nullptr, 0))
    {
        std::cerr << "Metal compositor did not render directly to IOSurface\n";
        return 1;
    }
    IOSurfaceLock (directSurface, kIOSurfaceLockReadOnly, nullptr);
    const auto* bytes = static_cast<const uint8_t*> (IOSurfaceGetBaseAddress (directSurface));
    const size_t rowBytes = IOSurfaceGetBytesPerRow (directSurface);
    const size_t center = static_cast<size_t> (height / 2) * rowBytes
                        + static_cast<size_t> (width / 2) * 4;
    const std::array<uint8_t, 4> directPixel = {
        bytes[center + 2], bytes[center + 1], bytes[center], bytes[center + 3] };
    IOSurfaceUnlock (directSurface, kIOSurfaceLockReadOnly, nullptr);
    if (! isRed (directPixel))
    {
        std::cerr << "Direct Metal IOSurface output did not contain the red frame\n";
        return 1;
    }

    const auto consumer = std::filesystem::path (argc > 0 ? argv[0] : "")
                            .parent_path() / "arbit-metal-iosurface-consumer";
    std::ostringstream consumerCommand;
    consumerCommand << std::quoted (consumer.string()) << ' '
                    << IOSurfaceGetID (directSurface) << ' ' << width << ' ' << height;
    const int consumerStatus = std::system (consumerCommand.str().c_str());
    if (consumerStatus == -1 || ! WIFEXITED (consumerStatus)
        || WEXITSTATUS (consumerStatus) != 0)
    {
        std::cerr << "Cross-process Metal consumer could not see the rendered IOSurface\n";
        return 1;
    }
    renderer.clearDirectOutputs();
    CFRelease (directSurface);

    renderer.deleteTexture (overlayHandle);
    renderer.deleteTexture (sourceHandle);
    renderer.shutdown (&gl);

    // The production shared-viewport mode must work without a GL context or
    // GL-backed source IDs. Exercise opaque Metal uploads, GPU frame blending,
    // and direct IOSurface output as one strict path.
    videorender::MetalFrameRenderer directRenderer;
    if (! directRenderer.initialize (nullptr, width, height, error, true))
    {
        std::cerr << "Metal-only compositor initialization failed: " << error << '\n';
        return 1;
    }
    directRenderer.setBackgroundColor (0, 0, 0, 1);
    constexpr unsigned redHandle = 0x80000001u;
    constexpr unsigned blueHandle = 0x80000002u;
    constexpr unsigned mixHandle = 0x80000003u;
    std::vector<uint8_t> fullBlue (static_cast<size_t> (width) * height * 4, 255);
    for (size_t i = 0; i < fullBlue.size(); i += 4)
        fullBlue[i] = fullBlue[i + 1] = 0;
    directRenderer.uploadRgba (redHandle, red.data(), width, height, width * 4);
    directRenderer.uploadRgba (blueHandle, fullBlue.data(), width, height, width * 4);
    directRenderer.setFrameBlend (mixHandle, redHandle, blueHandle,
                                  width, height, 0.5f);
    videorender::LayerDesc mixedLayer;
    mixedLayer.texture = mixHandle;
    mixedLayer.texWidth = width;
    mixedLayer.texHeight = height;

    properties = CFDictionaryCreateMutable (
        kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks,
        &kCFTypeDictionaryValueCallBacks);
    setSurfaceInt (properties, kIOSurfaceWidth, width);
    setSurfaceInt (properties, kIOSurfaceHeight, height);
    setSurfaceInt (properties, kIOSurfaceBytesPerElement, 4);
    setSurfaceInt (properties, kIOSurfacePixelFormat, static_cast<int32_t> ('BGRA'));
    IOSurfaceRef metalOnlySurface = IOSurfaceCreate (properties);
    CFRelease (properties);
    if (metalOnlySurface == nullptr)
    {
        std::cerr << "Metal-only compositor could not allocate its IOSurface\n";
        return 1;
    }

    if (!metalHdrMovie(directRenderer, metalOnlySurface, width, height,
        std::filesystem::path(argv[0]).parent_path() / "arbit-metal-iosurface-consumer", error))
    { std::cerr << "Metal PQ/HLG movie capture: " << error << '\n'; return 1; }

    auto nativeFrame = directRenderer.leaseRgbaTexture (blueHandle, width, height, error);
    if (! nativeFrame)
    {
        std::cerr << "Metal-only compositor could not retain its native source: "
                  << error << '\n';
        return 1;
    }
    videorender::LayerDesc nativeLayer;
    nativeLayer.nativeTextureBackend = nativeFrame->backend();
    nativeLayer.nativeTextureView = nativeFrame->colorTextureViewHandle();
    nativeLayer.nativeTextureDescriptor = nativeFrame->colorTextureDescriptor();
    nativeLayer.nativeTextureOwner = nativeFrame;
    nativeLayer.texWidth = width;
    nativeLayer.texHeight = height;
    if (nativeLayer.texture != 0 || ! directRenderer.renderCompositeToIOSurface (
            nullptr, metalOnlySurface, width, height, &nativeLayer, 1, nullptr, 0))
    {
        std::cerr << "Metal-only compositor rejected its textureless native layer: "
                  << directRenderer.lastError() << '\n';
        return 1;
    }
    IOSurfaceLock (metalOnlySurface, kIOSurfaceLockReadOnly, nullptr);
    const auto* nativeBytes = static_cast<const uint8_t*> (
        IOSurfaceGetBaseAddress (metalOnlySurface));
    const size_t nativeCenter = static_cast<size_t> (height / 2)
        * IOSurfaceGetBytesPerRow (metalOnlySurface) + static_cast<size_t> (width / 2) * 4;
    const bool nativeVisible = nativeBytes != nullptr && isBlue ({
        nativeBytes[nativeCenter + 2], nativeBytes[nativeCenter + 1],
        nativeBytes[nativeCenter], nativeBytes[nativeCenter + 3] });
    IOSurfaceUnlock (metalOnlySurface, kIOSurfaceLockReadOnly, nullptr);
    if (! nativeVisible)
    {
        std::cerr << "Metal-only compositor skipped its textureless native layer\n";
        return 1;
    }
    auto wrongDevice = nativeLayer;
    ++wrongDevice.nativeTextureDescriptor.deviceOrContextIdentity;
    if (directRenderer.renderCompositeToIOSurface (
            nullptr, metalOnlySurface, width, height, &wrongDevice, 1, nullptr, 0))
    {
        std::cerr << "Metal-only compositor accepted a native view from another device\n";
        return 1;
    }
    wrongDevice.nativeTextureOwner.reset();
    nativeLayer.nativeTextureOwner.reset();
    nativeFrame.reset();
    if (directRenderer.renderCompositeToIOSurface (
            nullptr, metalOnlySurface, width, height, &nativeLayer, 1, nullptr, 0))
    {
        std::cerr << "Metal-only compositor accepted a released native view\n";
        return 1;
    }

    DecodedHdrFrame hdr;
    hdr.width = hdr.height = 2;
    hdr.strideFloats = 8;
    hdr.color = {colortransform::ColorSpace::Rec2020, colortransform::TransferFunction::PQ};
    hdr.matrix = AVCOL_SPC_BT2020_NCL;
    hdr.range = AVCOL_RANGE_MPEG;
    hdr.rgba = {0.123456f, 0.25f, 0.5f, 1.0f, 0.75f, 0.1f, 0.2f, 1.0f,
                0.3f, 0.4f, 0.5f, 1.0f, 0.6f, 0.7f, 0.8f, 1.0f};
    auto hdrLease = directRenderer.leaseDecodedHdrFrame(hdr, error);
    std::vector<float> hdrPixels;
    if (!hdrLease || !directRenderer.validDecodedHdrFrameTexture(hdrLease)
        || arbitgpu::validMaterialFrameTexture(hdrLease)
        || !hdrLease->readColorFloatPixels(hdrPixels)
        || hdrPixels.size() != hdr.rgba.size()
        || !std::equal(hdrPixels.begin(), hdrPixels.end(), hdr.rgba.begin(),
            [](float actual, float expected) { return std::abs(actual - expected) < 0.000001f; }))
    {
        std::cerr << "Metal PQ float lease lost code values, row order, or ownership: " << error << '\n';
        return 1;
    }
    const auto originalHdr = hdr;
    hdr.rgba = {0.5f,0.5f,0.5f,1, 0.75f,0.75f,0.75f,1, 0.5f,0,0,1, 0,0,0,1};
    auto linearPq = directRenderer.leaseDecodedPqFrameForMaterial(hdr,error);
    auto exportedPq = directRenderer.leaseDecodedPqFrameForMaterial(hdr,error);
    std::vector<float> linearPqPixels, exportedPqPixels;
    const std::vector<float> linearOracle {0.45441236f,0.45441236f,0.45441236f,1,
        4.84422589f,4.84422589f,4.84422589f,1, 0.75454763f,-0.05659728f,-0.00824793f,1, 0,0,0,1};
    if (!linearPq || !exportedPq || !arbitgpu::validMaterialFrameTexture(linearPq)
        || !arbitgpu::materialFrameIsLinear(linearPq->colorTextureDescriptor())
        || !linearPq->readColorFloatPixels(linearPqPixels)
        || !exportedPq->readColorFloatPixels(exportedPqPixels)
        || linearPqPixels != exportedPqPixels || linearPqPixels.size() != linearOracle.size()
        || !std::equal(linearPqPixels.begin(),linearPqPixels.end(),linearOracle.begin(),
            [](float actual,float expected) { return std::abs(actual-expected) < 0.008f; })
        || linearPqPixels[5] <= 1 || linearPqPixels[9] >= 0 || linearPqPixels[10] >= 0)
    { std::cerr << "Metal PQ material conversion lost luminance, signed gamut, orientation or preview/export identity: " << error << '\n'; return 1; }
    hdr.rgba.assign(16,1.0f);
    auto advancedPq = directRenderer.leaseDecodedPqFrameForMaterial(hdr,error);
    if (!advancedPq || !linearPq->readColorFloatPixels(exportedPqPixels) || exportedPqPixels != linearPqPixels)
    { std::cerr << "Metal PQ material owner changed after decoder advancement\n"; return 1; }
    for (int fault = 0; fault < 6; ++fault) {
        auto invalid = originalHdr;
        if (fault == 0) invalid.color.transfer = colortransform::TransferFunction::HLG;
        if (fault == 1) invalid.color.primaries = colortransform::ColorSpace::SRGB;
        if (fault == 2) invalid.rgba[3] = 0.5f;
        if (fault == 3) invalid.rgba[0] = 1.01f;
        if (fault == 4) invalid.strideFloats = 9;
        if (fault == 5) invalid.range = AVCOL_RANGE_UNSPECIFIED;
        if (directRenderer.leaseDecodedPqFrameForMaterial(invalid,error))
        { std::cerr << "Metal PQ material lease accepted malformed color, alpha, code values or rows\n"; return 1; }
    }
    hdr = originalHdr;
    hdr.color.transfer = colortransform::TransferFunction::HLG;
    auto hlgLease = directRenderer.leaseDecodedHdrFrame(hdr, error);
    if (directRenderer.leaseDecodedHdrFrameForMaterial(hdr,{},error))
    { std::cerr << "Metal HLG material guessed an absent source peak\n"; return 1; }
    const auto retainedHlg = hdr;
    std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> linearHlg;
    hdr.rgba.assign(videohelper::tests::hlg::encoded.begin(),videohelper::tests::hlg::encoded.end());
    for (const auto& sample : videohelper::tests::hlg::cases)
    {
        const videosourcecolor::Policy policy {sample.peak};
        auto previewHlg = directRenderer.leaseDecodedHdrFrameForMaterial(hdr,policy,error);
        auto exportHlg = directRenderer.leaseDecodedHdrFrameForMaterial(hdr,policy,error);
        if (sample.peak == 1000) linearHlg = previewHlg;
        std::vector<float> before, exported;
        if (!previewHlg || !exportHlg || !arbitgpu::validMaterialFrameTexture(previewHlg)
            || !arbitgpu::materialFrameIsLinear(previewHlg->colorTextureDescriptor())
            || previewHlg->colorTextureDescriptor().rowOrder != arbitgpu::NativeTextureRowOrder::TopFirst
            || !previewHlg->readColorFloatPixels(before) || !exportHlg->readColorFloatPixels(exported)
            || before != exported || before.size() != sample.linear.size())
        { std::cerr << "Metal HLG material lost owned float storage or preview/export identity\n"; return 1; }
        for (std::size_t i = 0; i < before.size(); ++i)
            if (std::abs(before[i]-sample.linear[i]) >= 0.002f*std::max(1.0f,std::abs(sample.linear[i])))
            { std::cerr << "Metal HLG native pixels differ from independent BT.2100 oracle\n"; return 1; }
        auto advanced = hdr; advanced.rgba.assign(16,1.0f);
        if (!directRenderer.leaseDecodedHdrFrameForMaterial(advanced,policy,error)
            || !previewHlg->readColorFloatPixels(exported) || before != exported)
        { std::cerr << "Metal HLG material changed after decoder advancement\n"; return 1; }
    }
    for (int fault = 0; fault < 4; ++fault) {
        auto invalid = retainedHlg;
        if (fault == 0) invalid.rgba[0] = 1.01f;
        if (fault == 1) invalid.rgba[0] = -0.01f;
        if (fault == 2) invalid.rgba[3] = 0.5f;
        if (fault == 3) invalid.range = AVCOL_RANGE_UNSPECIFIED;
        if (directRenderer.leaseDecodedHdrFrameForMaterial(invalid,videosourcecolor::Policy{1000.0},error))
        { std::cerr << "Metal HLG admitted malformed decoded facts or unverified source headroom\n"; return 1; }
    }
    hdr = retainedHlg;
    hdrPixels.clear();
    if (!hlgLease || !directRenderer.validDecodedHdrFrameTexture(hlgLease)
        || hlgLease->colorTextureDescriptor().transfer != colortransform::TransferFunction::HLG
        || !hlgLease->readColorFloatPixels(hdrPixels) || hdrPixels != hdr.rgba)
    {
        std::cerr << "Metal HLG float lease lost transfer or exact pixels: " << error << '\n';
        return 1;
    }
    hdr.rgba = {0.5f,0.5f,0.5f,1.0f, 0.5f,0.5f,0.5f,1.0f,
                0.5f,0.5f,0.5f,1.0f, 0.5f,0.5f,0.5f,1.0f};
    for (const auto transfer : {colortransform::TransferFunction::PQ,
                                colortransform::TransferFunction::HLG})
    {
        hdr.color.transfer = transfer;
        auto display = directRenderer.leaseDecodedHdrFrameForDisplay(hdr,error);
        auto materialLayer = nativeLayer;
        if (!display || !arbitgpu::validMaterialFrameTexture(display)
            || !arbitgpu::materialFrameIsSrgb(display))
        { std::cerr << "Metal HDR material display lease failed: " << error << '\n'; return 1; }
        materialLayer.nativeTextureBackend = display->backend();
        materialLayer.nativeTextureView = display->colorTextureViewHandle();
        materialLayer.nativeTextureDescriptor = display->colorTextureDescriptor();
        materialLayer.nativeTextureOwner = display;
        materialLayer.texWidth = materialLayer.texHeight = 2;
        if (!directRenderer.renderCompositeToIOSurface(
                nullptr,metalOnlySurface,width,height,&materialLayer,1,nullptr,0))
        { std::cerr << "Metal HDR material preview failed: " << directRenderer.lastError() << '\n'; return 1; }
        IOSurfaceLock(metalOnlySurface,kIOSurfaceLockReadOnly,nullptr);
        const auto* bytes = static_cast<const std::uint8_t*>(IOSurfaceGetBaseAddress(metalOnlySurface));
        const auto stride = IOSurfaceGetBytesPerRow(metalOnlySurface);
        const auto* center = bytes ? bytes + (height / 2) * stride + (width / 2) * 4 : nullptr;
        const int expected = transfer == colortransform::TransferFunction::PQ ? 184 : 107;
        const bool matches = center && std::abs(int(center[0])-expected) <= 4
            && std::abs(int(center[1])-expected) <= 4
            && std::abs(int(center[2])-expected) <= 4 && center[3] >= 250;
        IOSurfaceUnlock(metalOnlySurface,kIOSurfaceLockReadOnly,nullptr);
        if (!matches)
        { std::cerr << "Metal PQ/HLG material preview missed expected sRGB gray\n"; return 1; }
    }
    auto foreignDevice = std::make_shared<ForeignDeviceHdrFrame>(hdrLease);
    auto byteLease = directRenderer.leaseRgbaTexture(blueHandle, width, height, error);
    if (directRenderer.validDecodedHdrFrameTexture(foreignDevice)
        || !byteLease || directRenderer.validDecodedHdrFrameTexture(byteLease))
    {
        std::cerr << "Metal HDR validator accepted a foreign device or byte owner\n";
        return 1;
    }
    foreignDevice.reset();
    hdr.strideFloats = 9;
    if (directRenderer.leaseDecodedHdrFrame(hdr, error))
    { std::cerr << "Metal HDR lease accepted non-tight rows\n"; return 1; }
    hdr.strideFloats = 8;
    hdr.rgba.pop_back();
    if (directRenderer.leaseDecodedHdrFrame(hdr, error))
    { std::cerr << "Metal HDR lease accepted truncated pixels\n"; return 1; }
    hdr.rgba.push_back(1.0f);
    hdr.matrix = AVCOL_SPC_BT709;
    if (directRenderer.leaseDecodedHdrFrame(hdr, error))
    { std::cerr << "Metal HDR lease accepted a wrong matrix\n"; return 1; }
    hdr.matrix = AVCOL_SPC_BT2020_NCL;
    hdr.color.transfer = colortransform::TransferFunction::SRGB;
    if (directRenderer.leaseDecodedHdrFrame(hdr, error))
    { std::cerr << "Metal HDR lease accepted SDR transfer\n"; return 1; }
    hdr.color.transfer = colortransform::TransferFunction::PQ;
    hdr.rgba[0] = std::numeric_limits<float>::quiet_NaN();
    if (directRenderer.leaseDecodedHdrFrame(hdr, error))
    { std::cerr << "Metal HDR lease accepted a nonfinite code value\n"; return 1; }
    hdr.rgba[0] = 0.123456f;
    hdr.width = 4097;
    if (directRenderer.leaseDecodedHdrFrame(hdr, error))
    { std::cerr << "Metal HDR lease accepted an oversized extent\n"; return 1; }

    // Decoded RGBA rows and the GUI's imported IOSurface both start at image-top.
    // Sample each quadrant of the actual direct target; a solid center sample
    // cannot detect a vertical flip at the final Metal pass.
    constexpr unsigned quadrantHandle = 0x80000004u;
    std::vector<uint8_t> quadrants (static_cast<size_t> (width) * height * 4, 255);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
        {
            const size_t offset = (static_cast<size_t> (y) * width + x) * 4;
            quadrants[offset] = (x < width / 2) == (y < height / 2) ? 255 : 0;
            quadrants[offset + 1] = x >= width / 2 ? 255 : 0;
            quadrants[offset + 2] = y >= height / 2 && x < width / 2 ? 255 : 0;
        }
    directRenderer.uploadRgba (quadrantHandle, quadrants.data(), width, height, width * 4);
    videorender::LayerDesc quadrantLayer;
    quadrantLayer.texture = quadrantHandle;
    quadrantLayer.texWidth = width;
    quadrantLayer.texHeight = height;
    for (const bool canvas : { false, true })
    {
        directRenderer.setCanvas (canvas ? width : 0, canvas ? height : 0);
        if (! directRenderer.renderCompositeToIOSurface (
                nullptr, metalOnlySurface, width, height, &quadrantLayer, 1, nullptr, 0))
        {
            std::cerr << "Metal-only quadrant composition failed: "
                      << directRenderer.lastError() << '\n';
            return 1;
        }
        IOSurfaceLock (metalOnlySurface, kIOSurfaceLockReadOnly, nullptr);
        const auto* quadrantBytes = static_cast<const uint8_t*> (
            IOSurfaceGetBaseAddress (metalOnlySurface));
        const size_t quadrantStride = IOSurfaceGetBytesPerRow (metalOnlySurface);
        const auto sample = [&] (int x, int y)
        {
            const auto* bgra = quadrantBytes + static_cast<size_t> (y) * quadrantStride
                                             + static_cast<size_t> (x) * 4;
            return std::array<uint8_t, 4> { bgra[2], bgra[1], bgra[0], bgra[3] };
        };
        const bool oriented = quadrantBytes != nullptr
            && isRed (sample (width / 4, height / 4))
            && isGreen (sample (3 * width / 4, height / 4))
            && isBlue (sample (width / 4, 3 * height / 4))
            && isYellow (sample (3 * width / 4, 3 * height / 4));
        IOSurfaceUnlock (metalOnlySurface, kIOSurfaceLockReadOnly, nullptr);
        if (! oriented)
        {
            std::cerr << "Metal-only direct IOSurface quadrants lost top-first order"
                      << (canvas ? " with canvas\n" : " without canvas\n");
            return 1;
        }
    }
    directRenderer.setCanvas (0, 0);
    directRenderer.deleteTexture (quadrantHandle);

    if (! directRenderer.renderCompositeToIOSurface (
            nullptr, metalOnlySurface, width, height, &mixedLayer, 1, nullptr, 0))
    {
        std::cerr << "Metal-only compositor did not render its frame blend\n";
        return 1;
    }

    // Execute the real typed-plan lowering seam, then bind both uploaded RGBA
    // resources through the strict production Metal compositor. The matte has
    // two uniform regions so samples away from its edge remain deterministic
    // while still exercising bounded morphology, feather, levels, choke, and
    // invert in the production shader.
    constexpr unsigned matteHandle = 0x80000005u;
    std::vector<uint8_t> matteRgba (static_cast<size_t> (width) * height * 4, 255);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
        {
            const size_t offset = (static_cast<size_t> (y) * width + x) * 4;
            matteRgba[offset] = x < width / 2 ? 64 : 192;
            matteRgba[offset + 1] = 17;
            matteRgba[offset + 2] = 239;
        }
    directRenderer.uploadRgba (matteHandle, matteRgba.data(), width, height, width * 4);

    auto mattePlan = typedMattePlan();
    videorender::LayerDesc matteLayer;
    matteLayer.clipId = mattePlan.clipId;
    matteLayer.texture = redHandle;
    matteLayer.texWidth = width;
    matteLayer.texHeight = height;
    matteLayer.matteTexture = matteHandle;
    matteLayer.matteWidth = width;
    matteLayer.matteHeight = height;
    std::string matteError;
    if (! videowire::executeVisualLayerPlan (
            { mattePlan }, mattePlan.clipId, matteLayer, matteError, videohelper::geometry::PlanUse::preview)
        || ! matteLayer.matteApply || ! matteLayer.matteInvert
        || matteLayer.matteBlack != 0.2f || matteLayer.matteWhite != 0.8f
        || matteLayer.matteErodeDilate != 1.0f || matteLayer.matteFeather != 1.0f
        || matteLayer.matteChoke != 0.1f)
    {
        std::cerr << "Typed matte plan did not lower bounded refinement parameters: "
                  << matteError << '\n';
        return 1;
    }
    if (! directRenderer.renderCompositeToIOSurface (
            nullptr, metalOnlySurface, width, height, &matteLayer, 1, nullptr, 0))
    {
        std::cerr << "Strict Metal rejected the typed matte frame: "
                  << directRenderer.lastError() << '\n';
        return 1;
    }

    IOSurfaceLock (metalOnlySurface, kIOSurfaceLockReadOnly, nullptr);
    const auto* matteBytes = static_cast<const uint8_t*> (
        IOSurfaceGetBaseAddress (metalOnlySurface));
    const size_t matteRowBytes = IOSurfaceGetBytesPerRow (metalOnlySurface);
    const auto mattePixelAt = [&](int x)
    {
        const size_t offset = static_cast<size_t> (height / 2) * matteRowBytes
                            + static_cast<size_t> (x) * 4;
        return std::array<uint8_t, 4> { matteBytes[offset + 2], matteBytes[offset + 1],
                                        matteBytes[offset], matteBytes[offset + 3] };
    };
    const auto matteLeft = mattePixelAt (width / 8);
    const auto matteRight = mattePixelAt (width * 7 / 8);
    IOSurfaceUnlock (metalOnlySurface, kIOSurfaceLockReadOnly, nullptr);
    if (matteLeft[0] < 195 || matteLeft[0] > 220 || matteLeft[1] > 4
        || matteLeft[2] > 4 || matteLeft[3] < 248
        || matteRight[0] > 4 || matteRight[1] > 4 || matteRight[2] > 4
        || matteRight[3] < 248)
    {
        std::cerr << "Typed matte pixels did not reflect the uploaded red channel; left="
                  << static_cast<int> (matteLeft[0]) << ','
                  << static_cast<int> (matteLeft[1]) << ','
                  << static_cast<int> (matteLeft[2]) << ','
                  << static_cast<int> (matteLeft[3]) << " right="
                  << static_cast<int> (matteRight[0]) << ','
                  << static_cast<int> (matteRight[1]) << ','
                  << static_cast<int> (matteRight[2]) << ','
                  << static_cast<int> (matteRight[3]) << '\n';
        return 1;
    }

    auto malformedMattePlan = mattePlan;
    malformedMattePlan.edges[2].toPort = 0;
    if (videowire::executeVisualLayerPlan (
            { malformedMattePlan }, malformedMattePlan.clipId, matteLayer, matteError, videohelper::geometry::PlanUse::preview)
        || matteError != "typed matte graph has unsupported production topology")
    {
        std::cerr << "Typed matte executor accepted malformed topology\n";
        return 1;
    }
    videorender::LayerDesc texturelessMatte = matteLayer;
    texturelessMatte.matteTexture = 0;
    if (videowire::executeVisualLayerPlan (
            { mattePlan }, mattePlan.clipId, texturelessMatte, matteError, videohelper::geometry::PlanUse::preview)
        || matteError != "typed matte GPU texture is unavailable")
    {
        std::cerr << "Typed matte executor accepted a missing texture\n";
        return 1;
    }
    videorender::LayerDesc missingResourceLayer = mixedLayer;
    missingResourceLayer.texture = 0x8fffffffu;
    if (directRenderer.renderCompositeToIOSurface (
            nullptr, metalOnlySurface, width, height,
            &missingResourceLayer, 1, nullptr, 0))
    {
        std::cerr << "Strict Metal accepted a missing GPU resource instead of failing closed\n";
        return 1;
    }
    if (! directRenderer.renderCompositeToIOSurface (
            nullptr, metalOnlySurface, width, height, &mixedLayer, 1, nullptr, 0))
    {
        std::cerr << "Strict Metal did not recover after a rejected frame\n";
        return 1;
    }
    IOSurfaceLock (metalOnlySurface, kIOSurfaceLockReadOnly, nullptr);
    bytes = static_cast<const uint8_t*> (IOSurfaceGetBaseAddress (metalOnlySurface));
    const size_t metalOnlyRowBytes = IOSurfaceGetBytesPerRow (metalOnlySurface);
    const size_t metalOnlyCenter = static_cast<size_t> (height / 2) * metalOnlyRowBytes
                                 + static_cast<size_t> (width / 2) * 4;
    const std::array<uint8_t, 4> mixedPixel = {
        bytes[metalOnlyCenter + 2], bytes[metalOnlyCenter + 1],
        bytes[metalOnlyCenter], bytes[metalOnlyCenter + 3] };
    IOSurfaceUnlock (metalOnlySurface, kIOSurfaceLockReadOnly, nullptr);

    // Strict production-path coverage: every public effect, blend mode,
    // transition, mask shape, and a combined transform/crop frame must render
    // without a GL context. Individual visual assertions above catch shader
    // correctness; this matrix prevents newly added combinations from silently
    // reintroducing an API fallback.
    for (int effectType = 0; effectType < videofx::kEffectTypeCount; ++effectType)
    {
        videorender::EffectSlotState effectState;
        effectState.type = effectType;
        effectState.enabled = true;
        const auto* definition = videofx::effectDefFor (effectType);
        for (int param = 0; definition != nullptr && param < definition->paramCount; ++param)
            effectState.params[param] = definition->params[param].defaultValue;
        mixedLayer.clipId = 1000 + effectType;
        mixedLayer.effects = &effectState;
        mixedLayer.effectCount = 1;
        if (! directRenderer.renderCompositeToIOSurface (
                nullptr, metalOnlySurface, width, height, &mixedLayer, 1, nullptr, 0))
        {
            std::cerr << "Strict Metal rejected effect " << effectType << ": "
                      << directRenderer.lastError() << '\n';
            return 1;
        }
    }
    mixedLayer.effects = nullptr;
    mixedLayer.effectCount = 0;

    videorender::LayerDesc redLayer = mixedLayer;
    redLayer.texture = redHandle;
    videorender::LayerDesc blueLayer = mixedLayer;
    blueLayer.texture = blueHandle;
    blueLayer.opacity = 0.5f;
    std::array<videorender::LayerDesc, 2> blendLayers = { redLayer, blueLayer };
    for (int blendMode = 0; blendMode < static_cast<int> (videofx::BlendMode::Count);
         ++blendMode)
    {
        blendLayers[1].blendMode = blendMode;
        if (! directRenderer.renderCompositeToIOSurface (
                nullptr, metalOnlySurface, width, height,
                blendLayers.data(), static_cast<int> (blendLayers.size()), nullptr, 0))
        {
            std::cerr << "Strict Metal rejected blend mode " << blendMode << '\n';
            return 1;
        }
    }

    videorender::LayerDesc transitionLayer = blueLayer;
    transitionLayer.opacity = 1.0f;
    transitionLayer.fromLayer = &redLayer;
    transitionLayer.transitionProgress = 0.5f;
    for (int blendMode = 0; blendMode < static_cast<int> (videofx::BlendMode::Count);
         ++blendMode)
    {
        transitionLayer.blendMode = blendMode;
        for (int transition = 1;
             transition < static_cast<int> (videofx::TransitionType::Count); ++transition)
        {
            transitionLayer.transitionType = transition;
            if (! directRenderer.renderCompositeToIOSurface (
                    nullptr, metalOnlySurface, width, height,
                    &transitionLayer, 1, nullptr, 0))
            {
                std::cerr << "Strict Metal rejected transition " << transition
                          << " with blend mode " << blendMode << '\n';
                return 1;
            }
        }
    }

    for (int maskType = 1; maskType <= 2; ++maskType)
    {
        videorender::LayerDesc masked = redLayer;
        masked.maskType = maskType;
        masked.maskInvert = maskType == 2;
        masked.maskFeather = 0.15f;
        masked.scale = 0.75f;
        masked.rotationDeg = 17.0f;
        masked.translateX = 0.1f;
        masked.translateY = -0.1f;
        masked.cropLeft = 0.05f;
        masked.cropBottom = 0.1f;
        if (! directRenderer.renderCompositeToIOSurface (
                nullptr, metalOnlySurface, width, height, &masked, 1, nullptr, 0))
        {
            std::cerr << "Strict Metal rejected mask/transform combination "
                      << maskType << '\n';
            return 1;
        }
    }

    constexpr unsigned strictLutHandle = 0x80000004u;
    directRenderer.uploadLut3D (strictLutHandle, magentaLut.data(), 2);
    videorender::LayerDesc lutLayer = redLayer;
    lutLayer.lutTexture = strictLutHandle;
    lutLayer.lutSize = 2;
    if (! directRenderer.renderCompositeToIOSurface (
            nullptr, metalOnlySurface, width, height, &lutLayer, 1, nullptr, 0))
    {
        std::cerr << "Strict Metal rejected a 3D LUT layer\n";
        return 1;
    }
    directRenderer.deleteTexture (strictLutHandle);

    videorender::LayerDesc particleLayer;
    particleLayer.clipId = 77;
    particleLayer.particleSource = true;
    particleLayer.texWidth = width;
    particleLayer.texHeight = height;
    particleLayer.genParams["count"] = 512;
    particleLayer.genParams["size"] = 4.0;
    particleLayer.genParams["force"] = 1.0;
    auto particleScore = std::make_shared<arbitmod::Score>();
    particleScore->notationVersion = particleScore->scoreRevision = 1;
    particleScore->rootFreq = 440.0f;
    arbitmod::Note particleNote;
    particleNote.id = 77;
    particleNote.trackId = 0;
    particleNote.startBeat = 0.0f;
    particleNote.lengthBeats = 4.0f;
    particleNote.durationSeconds = 2.0f;
    particleNote.velocity = 100.0f;
    particleNote.freqHz = 261.625565f;
    particleScore->notes.push_back(particleNote);
    canonicalblockc::FrameKey particleKey;
    particleKey.projectGeneration = particleKey.sourceGeneration = particleKey.helperGeneration = 1;
    particleKey.backendGeneration = particleKey.deviceGeneration = particleKey.scoreGeneration = 1;
    particleKey.beatMapGeneration = particleKey.fpsGeneration = particleKey.loopGeneration = particleKey.seekGeneration = 1;
    particleKey.fps = 60.0;
    canonicalblockc::FrameProducer particleProducer;
    particleLayer.canonicalBlockCFrame = particleProducer.evaluate(particleKey, particleScore, 0.0f);
    particleLayer.shaderClock.playing = true;
    particleLayer.shaderClock.timeDelta = 1.0 / 30.0;
    videorender::EffectSlotState particleBlur;
    particleBlur.type = static_cast<int> (videofx::EffectType::Blur);
    particleBlur.enabled = true;
    particleBlur.params[0] = 2.0f;
    particleLayer.effects = &particleBlur;
    particleLayer.effectCount = 1;
    for (int frame = 0; frame < 4; ++frame)
    {
        particleLayer.shaderClock.frame = frame;
        if (! directRenderer.renderCompositeToIOSurface (
                nullptr, metalOnlySurface, width, height,
                &particleLayer, 1, nullptr, 0))
        {
            std::cerr << "Metal-only compositor rejected a particle layer\n";
            return 1;
        }
    }
    IOSurfaceLock (metalOnlySurface, kIOSurfaceLockReadOnly, nullptr);
    bytes = static_cast<const uint8_t*> (IOSurfaceGetBaseAddress (metalOnlySurface));
    int litParticlePixels = 0;
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
        {
            const size_t pixelOffset = static_cast<size_t> (y) * metalOnlyRowBytes
                                     + static_cast<size_t> (x) * 4;
            if (bytes[pixelOffset] > 4 || bytes[pixelOffset + 1] > 4
                || bytes[pixelOffset + 2] > 4)
                ++litParticlePixels;
        }
    IOSurfaceUnlock (metalOnlySurface, kIOSurfaceLockReadOnly, nullptr);
    const bool particlePassed = litParticlePixels > 0
        && directRenderer.particleBackend() == "metal-compute";

    std::string shaderLog;
    std::vector<videorender::GenParam> shaderParams;
    if (! directRenderer.setClipShader (
            88, "void main(){ fragColor=(uTime>0.5 && uResolution.x>1.0)"
                "?vec4(0.0,1.0,0.0,1.0):vec4(1.0,0.0,0.0,1.0); }",
            shaderLog, shaderParams))
    {
        std::cerr << "Native Metal generator compile failed: " << shaderLog << '\n';
        return 1;
    }
    videorender::LayerDesc shaderLayer;
    shaderLayer.clipId = 88;
    shaderLayer.shaderSource = true;
    shaderLayer.shaderClock.timeSec = 1.0;
    shaderLayer.effects = &invert;
    shaderLayer.effectCount = 1;
    if (! directRenderer.renderCompositeToIOSurface (
            nullptr, metalOnlySurface, width, height, &shaderLayer, 1, nullptr, 0))
    {
        std::cerr << "Strict Metal rejected a generator/effect combination\n";
        return 1;
    }
    shaderLayer.effects = nullptr;
    shaderLayer.effectCount = 0;
    if (! directRenderer.renderCompositeToIOSurface (
            nullptr, metalOnlySurface, width, height, &shaderLayer, 1, nullptr, 0))
    {
        std::cerr << "Metal-only compositor rejected its native generator: "
                  << directRenderer.lastError() << '\n';
        return 1;
    }
    IOSurfaceLock (metalOnlySurface, kIOSurfaceLockReadOnly, nullptr);
    bytes = static_cast<const uint8_t*> (IOSurfaceGetBaseAddress (metalOnlySurface));
    const std::array<uint8_t, 4> shaderPixel = {
        bytes[metalOnlyCenter + 2], bytes[metalOnlyCenter + 1],
        bytes[metalOnlyCenter], bytes[metalOnlyCenter + 3] };
    const size_t shaderInterior = static_cast<size_t> (height / 2) * metalOnlyRowBytes
                                + static_cast<size_t> (width / 8) * 4;
    const std::array<uint8_t, 4> shaderInteriorPixel = {
        bytes[shaderInterior + 2], bytes[shaderInterior + 1],
        bytes[shaderInterior], bytes[shaderInterior + 3] };
    IOSurfaceUnlock (metalOnlySurface, kIOSurfaceLockReadOnly, nullptr);
    directRenderer.clearClipShader (88);
    directRenderer.clearDirectOutputs();
    CFRelease (metalOnlySurface);
    directRenderer.deleteTexture (matteHandle);
    directRenderer.deleteTexture (mixHandle);
    directRenderer.deleteTexture (blueHandle);
    directRenderer.deleteTexture (redHandle);
    directRenderer.shutdown (nullptr);
    if (directRenderer.validDecodedHdrFrameTexture(hdrLease)
        || directRenderer.validDecodedHdrFrameTexture(hlgLease)
        || hdrLease->readColorFloatPixels(hdrPixels)
        || arbitgpu::validMaterialFrameTexture(linearPq)
        || linearPq->readColorFloatPixels(hdrPixels)
        || !linearHlg || arbitgpu::validMaterialFrameTexture(linearHlg)
        || linearHlg->readColorFloatPixels(hdrPixels))
    {
        std::cerr << "Metal HDR lease survived renderer owner release\n";
        return 1;
    }
    hdrLease.reset();
    hlgLease.reset();
    if (mixedPixel[0] < 120 || mixedPixel[0] > 136
        || mixedPixel[1] > 8 || mixedPixel[2] < 120 || mixedPixel[2] > 136
        || mixedPixel[3] < 248)
    {
        std::cerr << "Metal-only frame blend was not half red/half blue\n";
        return 1;
    }
    if (! particlePassed)
    {
        std::cerr << "Metal-only particle layer produced no visible pixels\n";
        return 1;
    }
    if (! isGreen (shaderPixel))
    {
        std::cerr << "Native Metal generator was not rendered into the compositor; got "
                  << static_cast<int> (shaderPixel[0]) << ','
                  << static_cast<int> (shaderPixel[1]) << ','
                  << static_cast<int> (shaderPixel[2]) << ','
                  << static_cast<int> (shaderPixel[3]) << '\n';
        return 1;
    }
    if (! isGreen (shaderInteriorPixel))
    {
        std::cerr << "Native Metal generator did not fill the canvas; interior RGBA="
                  << static_cast<int> (shaderInteriorPixel[0]) << ','
                  << static_cast<int> (shaderInteriorPixel[1]) << ','
                  << static_cast<int> (shaderInteriorPixel[2]) << ','
                  << static_cast<int> (shaderInteriorPixel[3]) << '\n';
        return 1;
    }

    glfwDestroyWindow (window);
    glfwTerminate();

    std::cout << "Metal compositor center RGBA="
              << static_cast<int> (pixel[0]) << ','
              << static_cast<int> (pixel[1]) << ','
              << static_cast<int> (pixel[2]) << ','
              << static_cast<int> (pixel[3]) << '\n';
    if (! isBlue (pixel))
    {
        std::cerr << "Metal compositor did not composite the blue overlay\n";
        return 1;
    }
    std::cout << "Metal frame compositor smoke passed\n";
    return 0;
}
