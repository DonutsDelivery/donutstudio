#include "renderer.h"
#include "native_texture_rows_gl.h"
#include "geometry_core_backend.h"
#include "glb_geometry_core_adapter.h"
#include "sdf_visual_plan_execution.h"
#include "support/native_scene_orientation_fixture.h"
#include "support/sdf_display_oracle.h"
#include "support/typed_scene_pass_product_checks.h"
#include "../../shared/GeometrySurfaceMaterial.h"
#include "../../shared/generated/SurfaceMaterialStarterPrograms.h"
#include "../../shared/HdrImageOutputContract.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <cstring>
#include <iostream>

namespace
{
namespace fixture = videohelper::tests::orientation;
using Frame = std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>;

videorender::LayerDesc layerFor(const Frame& frame)
{
    videorender::LayerDesc layer;
    layer.clipId = 91;
    layer.texture = static_cast<unsigned>(frame->colorTextureViewHandle());
    layer.nativeTextureBackend = frame->backend();
    layer.nativeTextureView = frame->colorTextureViewHandle();
    layer.nativeTextureDescriptor = frame->colorTextureDescriptor();
    layer.nativeTextureOwner = frame;
    layer.rawExportFrame = frame;
    layer.texWidth = frame->width(); layer.texHeight = frame->height();
    layer.depthTexture = static_cast<unsigned>(frame->depthTextureViewHandle());
    layer.nativeDepthTextureBackend = frame->backend();
    layer.nativeDepthTextureView = frame->depthTextureViewHandle();
    layer.nativeDepthTextureDescriptor = frame->depthTextureDescriptor();
    layer.depthWidth = frame->width(); layer.depthHeight = frame->height();
    return layer;
}

std::vector<std::uint8_t> texturePixels(const arbitgl::GlFuncs& gl, unsigned texture)
{
    std::vector<std::uint8_t> image(fixture::width * fixture::height * 4);
    GLint previous = 0, pack = 0;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &previous);
    glGetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING, &pack);
    gl.BindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, texture);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, image.data());
    glBindTexture(GL_TEXTURE_2D, static_cast<unsigned>(previous));
    gl.BindBuffer(GL_PIXEL_PACK_BUFFER, static_cast<unsigned>(pack));
    return image;
}

bool composite(videorender::FrameRenderer& renderer, const arbitgl::GlFuncs& gl,
               const videorender::LayerDesc& layer, std::string& error, bool lit = false)
{
    const auto preview = renderer.renderComposite(&layer, 1);
    if (!preview) { error = renderer.lastError(); return false; }
    const auto previewPixels = texturePixels(gl, preview);
    std::vector<std::uint8_t> exported;
    if (!fixture::upright(previewPixels, error, lit)
        || !renderer.renderToPixels(&layer, 1, exported, error)
        || !fixture::upright(exported, error, lit)) return false;
    if (previewPixels != exported) { error = "Preview and export composite pixels differ"; return false; }
    return true;
}

std::optional<std::string> geometryMaterial(std::string& error)
{
    using namespace surfacematerial;
    surfacematerialbinding::ImportedSceneMaterialRequest request;
    request.version = surfacematerialbinding::kGraphFrameWireVersion;
    request.scene = {1}; request.binding.object = {1};
    request.sceneRevision = request.structuralRevision = request.evaluationRevision = request.programRevision = 1;
    request.binding.targetKind = surfacematerialbinding::BindingTargetKind::ObjectOverride;
    request.binding.surfaceMaterialRevision = 1;
    request.program = surfacematerialstarterfixture::kPrograms[0].program();
    const auto output = [&](OutputSemantic semantic) -> Operation& {
        const auto id = request.program.outputs[static_cast<std::size_t>(semantic)];
        return *std::find_if(request.program.operations.begin(), request.program.operations.end(),
                            [&](const auto& operation) { return operation.id == id; });
    };
    output(OutputSemantic::MaterialId).unsignedLiteral = 1;
    output(OutputSemantic::BaseColor).literal = {1, 1, 1, 0};
    output(OutputSemantic::Metallic).literal = {};
    output(OutputSemantic::Roughness).literal = {1, 0, 0, 0};
    output(OutputSemantic::Transmission).literal = {};
    output(OutputSemantic::Clearcoat).literal = {};
    output(OutputSemantic::Emission).literal = {};
    output(OutputSemantic::Opacity).literal = {1, 0, 0, 0};
    const auto append = [&](OperationKind kind, ValueType type, std::initializer_list<ValueId> inputs,
                            InputSemantic semantic = InputSemantic::Invalid, std::uint16_t parameter = 0) {
        Operation operation;
        operation.id = static_cast<ValueId>(request.program.operations.size() + 1);
        operation.kind = kind; operation.resultType = type; operation.semantic = semantic;
        operation.parameter = parameter; operation.inputCount = static_cast<std::uint8_t>(inputs.size());
        std::copy(inputs.begin(), inputs.end(), operation.inputs.begin());
        request.program.operations.push_back(operation); return operation.id;
    };
    request.program.textureSlotCount = 1;
    const auto uv = append(OperationKind::Input, ValueType::Vec2, {}, InputSemantic::TexCoord0);
    const auto sample = append(OperationKind::TextureSample2D, ValueType::Vec4, {uv});
    const auto r = append(OperationKind::Component, ValueType::Scalar, {sample}, InputSemantic::Invalid, 0);
    const auto g = append(OperationKind::Component, ValueType::Scalar, {sample}, InputSemantic::Invalid, 1);
    const auto b = append(OperationKind::Component, ValueType::Scalar, {sample}, InputSemantic::Invalid, 2);
    const auto rgb = append(OperationKind::ComposeVec3, ValueType::Vec3, {r, g, b});
    request.program.outputs[0] = append(OperationKind::Multiply, ValueType::Vec3, {rgb, request.program.outputs[0]});
    surfacematerialbinding::TextureSlotBinding texture;
    texture.source = surfacematerialbinding::TextureSourceKind::GraphFrame;
    texture.graphFrame = surfacematerialbinding::TextureSlotBinding::GraphFrameEndpoint {91, 0};
    request.binding.textures = {texture};
    const auto admitted = admit(request.program, error);
    if (!admitted) return std::nullopt;
    request.binding.surfaceMaterialDigest = admitted->structuralDigest();
    return geometrysurfacematerial::encode(request);
}

bool geometryCases(videorender::FrameRenderer& renderer, const arbitgl::GlFuncs& gl,
                   arbitgpu::NativeFixtureSceneBackend& backend, const Frame& decoded,
                   const Frame& embedded, std::string& error)
{
    using namespace videowire::geometry;
    using namespace videohelper::geometry;
    videohelper::gltf::GlbAdmissionOptions options;
    options.supportedRequiredExtensions = {"KHR_lights_punctual"};
    const auto document = videohelper::gltf::decodeStaticGlb(
        videoscreen::kGlb.data(), videoscreen::kGlb.size(), options, error);
    if (!document) return false;
    const auto value = videohelper::gltf::adaptGlbMeshToGeometryCore(*document, 0, 9001, 9000, 1, error);
    const auto material = geometryMaterial(error);
    const auto screen = fixture::screen(error);
    if (!value || !material || !screen) return false;
    PortContract contract;
    contract.carrier = CarrierKind::geometry3D;
    contract.maxVertices = 4096; contract.maxIndices = 12288; contract.maxAttributes = kMaximumAttributes;
    contract = withAttributeContract(contract, value->descriptor());
    const auto wire = lowerRuntimePlan(contract, *value, *material);
    NativeGeometryCoreCapabilitySource capabilities(backend);
    GeometryCorePlanRuntime runtime(capabilities);
    const auto preview = runtime.admitPreview({1, 1, 1, 9005, 1, PlanUse::preview}, wire, {}, {}, error);
    const auto exported = runtime.admitExport({1, 1, 1, 9005, 1, PlanUse::exportRender}, wire, {}, {}, error);
    if (!preview || !exported) return false;
    for (const auto& source : {decoded, embedded})
    {
        arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
        inputs.cameraOverride = screen->cameras[0];
        inputs.materialFrameTexture = source;
        const auto a = executeNativeGeometry(backend, *preview, fixture::width, fixture::height, error, false, inputs);
        const auto b = executeNativeGeometry(backend, *exported, fixture::width, fixture::height, error, false, inputs);
        if (!a || !b || !composite(renderer, gl, layerFor(a->frame), error, true)
            || !composite(renderer, gl, layerFor(b->frame), error, true)) return false;
        arbitgpu::NativeRawPassPixels rawA, rawB;
        if (!a->frame->readRawPass(renderpassoutput::Output::Color, rawA, error)
            || !b->frame->readRawPass(renderpassoutput::Output::Color, rawB, error)
            || !fixture::upright(rawA.bytes, error, true) || rawA.bytes != rawB.bytes) return false;
    }
    return true;
}

bool depthAndHdrCases(videorender::FrameRenderer& renderer, arbitgpu::NativeFixtureSceneBackend& backend,
                      std::string& error)
{
    auto scene = fixture::screen(error);
    if (!scene) return false;
    for (std::size_t i = 0; i < scene->vertexCount; ++i)
        scene->vertices[i].position.z = scene->vertices[i].position.y * 0.5f;
    auto resources = backend.prepare(scene, nullptr);
    auto frame = resources.prepared ? backend.render(scene, resources.resources, fixture::width, fixture::height, {})
        : arbitgpu::NativeFixtureSceneSubmission {};
    if (!frame.rendered) { error = resources.error + frame.error; return false; }
    arbitgpu::NativeRawPassPixels color, depth;
    if (!frame.frame->readRawPass(renderpassoutput::Output::Color, color, error)
        || !frame.frame->readRawPass(renderpassoutput::Output::Depth, depth, error)) return false;
    auto layer = layerFor(frame.frame);
    layer.depthFog = true; layer.fogNear = 0; layer.fogFar = 0.03f; layer.fogDensity = 2;
    layer.fogRed = 0.2f; layer.fogGreen = 0.6f; layer.fogBlue = 0.1f; layer.fogAlpha = 0.75f;
    std::vector<std::uint8_t> fogged;
    if (!renderer.renderToPixels(&layer, 1, fogged, error) || fogged.size() != color.bytes.size()
        || depth.bytes.size() != fixture::width * fixture::height * sizeof(float)) return false;
    const std::array<float, 3> fog {layer.fogRed, layer.fogGreen, layer.fogBlue};
    for (std::size_t i = 0; i < fogged.size(); i += 4)
    {
        float d; std::memcpy(&d, depth.bytes.data() + i, sizeof(d));
        const double distance = std::clamp(static_cast<double>(d) / layer.fogFar, 0.0, 1.0);
        const double opacity = layer.fogAlpha * (1.0 - std::exp(-layer.fogDensity * distance));
        for (std::size_t c = 0; c < 3; ++c)
        {
            const auto expected = std::lround(color.bytes[i + c] * (1.0 - opacity) + 255 * fog[c] * opacity);
            if (std::abs(static_cast<long>(fogged[i + c]) - expected) > 3)
            { error = "Tilted scene depth and color rows no longer align in the compositor"; return false; }
        }
    }
    scene = fixture::screen(error);
    if (!scene) return false;
    scene->materials[0].emissive = {2, 3, 4};
    scene->materials[0].emissiveTexture = scene->materials[0].baseColorTexture;
    resources = backend.prepare(scene, nullptr);
    arbitgpu::NativeFixtureSceneRuntimeInputs inputs; inputs.linearColor = true;
    frame = resources.prepared ? backend.render(scene, resources.resources, fixture::width, fixture::height, inputs)
        : arbitgpu::NativeFixtureSceneSubmission {};
    if (!frame.rendered || !frame.frame->readRawPass(renderpassoutput::Output::Color, color, error)) return false;
    layer = layerFor(frame.frame);
    renderer.setHdrImageCapture(true);
    std::vector<float> hdr;
    const bool captured = renderer.renderComposite(&layer, 1) != 0 && renderer.readLastCompositeFloat(hdr, error);
    renderer.setHdrImageCapture(false);
    if (!captured || hdr.size() * sizeof(std::uint16_t) != color.bytes.size()) return false;
    for (std::size_t i = 0; i < hdr.size(); ++i)
    {
        std::uint16_t half; std::memcpy(&half, color.bytes.data() + i * 2, 2);
        if (std::abs(hdr[i] - hdrimage::halfToFloat(half)) > 0.006f)
        { error = "Native row normalization changed linear HDR values or their raster positions"; return false; }
    }
    arbitgpu::NativeRawPassPixels retained;
    return frame.frame->readRawPass(renderpassoutput::Output::Color, retained, error) && retained.bytes == color.bytes;
}

bool sdfPublicationCases(videorender::FrameRenderer& renderer, const arbitgl::GlFuncs& gl,
                         std::string& error)
{
    using namespace videohelper::sdf;
    const auto reject = [&](const char* message) { error = message; return false; };
    const auto sourceFloats = [&](unsigned texture)
    {
        std::vector<float> pixels(fixture::width * fixture::height * 4);
        GLint previous = 0, pack = 0;
        glGetIntegerv(GL_TEXTURE_BINDING_2D, &previous);
        glGetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING, &pack);
        gl.BindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        glBindTexture(GL_TEXTURE_2D, texture);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, pixels.data());
        glBindTexture(GL_TEXTURE_2D, static_cast<unsigned>(previous));
        gl.BindBuffer(GL_PIXEL_PACK_BUFFER, static_cast<unsigned>(pack));
        return pixels;
    };
    NativeSdfRenderer sdfRenderer(arbitgpu::nativeSdfExecutionBackend());
    renderer.setHdrImageCapture(false);
    for (const std::uint8_t output : {std::uint8_t{0}, std::uint8_t{1}})
    {
        const auto plan = sdfdisplayoracle::plan(output);
        std::vector<std::uint8_t> previewPixels;
        for (const auto use : {NativeSdfRenderUse::Preview, NativeSdfRenderUse::Export})
        {
            videorender::LayerDesc layer;
            layer.clipId = plan.clipId;
            std::vector<NativeSdfRenderedFrame> owners;
            if (prepareVisualSdfLayer({plan}, plan.clipId, fixture::width, fixture::height,
                    use, sdfRenderer, layer, owners, error) != VisualSdfPreparation::rendered)
                return false;
            if (owners.size() != 1 || layer.nativeTextureOwner != owners.front().nativeFrame
                || !arbitgpu::isLinearSceneColor(layer.nativeTextureDescriptor)
                || layer.nativeTextureDescriptor.rowOrder != arbitgpu::NativeTextureRowOrder::BottomFirst
                || layer.nativeTextureDescriptor.deviceOrContextIdentity
                    != reinterpret_cast<std::uintptr_t>(glfwGetCurrentContext()))
                return reject("Prepared SDF layer lost its native descriptor or compositor lease");
            std::weak_ptr<const arbitgpu::NativeSdfSceneFrame> retained = owners.front().nativeFrame;
            owners.clear();
            if (retained.expired() || !glIsTexture(layer.texture))
                return reject("SDF attachment expired before its compositor layer was consumed");
            const auto rawBeforeDisplay = sourceFloats(layer.texture);
            const auto texture = renderer.renderComposite(&layer, 1);
            if (!texture) { error = renderer.lastError(); return false; }
            const auto pixels = texturePixels(gl, texture);
            if (!sdfdisplayoracle::verify([&](int x, int y) {
                    const auto at = (y * fixture::width + x) * 4;
                    return std::array<std::uint8_t, 4> {pixels[at], pixels[at + 1], pixels[at + 2], pixels[at + 3]};
                }, output == 1, error)) return false;
            const auto canonical = videorender::copyTopFirstOpenGlTexture(gl, layer.nativeTextureDescriptor, error);
            if (!canonical) return false;
            const auto canonicalDescriptor = canonical->colorTextureDescriptor();
            if (!arbitgpu::isLinearSceneColor(canonicalDescriptor)
                || canonicalDescriptor.rowOrder != arbitgpu::NativeTextureRowOrder::TopFirst
                || canonicalDescriptor.deviceOrContextIdentity != layer.nativeTextureDescriptor.deviceOrContextIdentity
                || canonicalDescriptor.width != fixture::width || canonicalDescriptor.height != fixture::height
                || canonicalDescriptor.textureViewHandle == layer.nativeTextureView
                || layer.nativeTextureDescriptor.rowOrder != arbitgpu::NativeTextureRowOrder::BottomFirst)
                return reject("SDF row normalization lost its canonical top-first descriptor or changed the native origin");
            const auto canonicalPixels = texturePixels(gl, static_cast<unsigned>(canonicalDescriptor.textureViewHandle));
            std::vector<std::uint8_t> exported;
            if (!renderer.renderToPixels(&layer, 1, exported, error) || pixels != exported)
                return reject("Prepared SDF preview and export compositor pixels disagree");
            if (renderer.hdrImageCaptureEnabled()) return reject("SDF video export must exercise ordinary SDR composition");
            renderer.setHdrImageCapture(true);
            std::vector<float> hdr;
            const bool haveHdr = renderer.renderComposite(&layer, 1) != 0 && renderer.readLastCompositeFloat(hdr, error);
            renderer.setHdrImageCapture(false);
            if (!haveHdr || hdr.size() != fixture::width * fixture::height * 4u) return false;
            if (output == 1)
            {
                const auto upper = (12 * fixture::width + 32) * 4;
                if (std::abs(hdr[upper] - 0.327399042269f) > 0.002f)
                    return reject("SDF HDR capture consumed the sRGB display copy instead of the original linear frame");
            }
            if (sourceFloats(layer.texture) != rawBeforeDisplay
                || !arbitgpu::isLinearSceneColor(layer.nativeTextureDescriptor))
                return reject("SDF display or HDR composition changed the authored half-float source");
            if (use == NativeSdfRenderUse::Preview) previewPixels = pixels;
            else if (pixels != previewPixels)
                return reject("SDF preview and export preparation changed the composed image");
            // The positive-Y sphere is above the image centre. Its depth is
            // nearer than the white miss below it, independent of display transfer.
            if (output == 1)
            {
                const auto upper = (fixture::height / 3 * fixture::width + fixture::width / 2) * 4;
                const auto lower = (2 * fixture::height / 3 * fixture::width + fixture::width / 2) * 4;
                if (pixels[upper] + 32u >= pixels[lower])
                    return reject("Composed SDF depth lost visible geometry or flipped its native rows");
                if (canonicalPixels[upper] + 32u >= canonicalPixels[lower])
                    return reject("SDF top-first descriptor disagrees with the canonical pixel origin");
            }
            for (int fault = 0; fault < 5; ++fault)
            {
                auto malformed = layer;
                if (fault == 0) malformed.nativeTextureDescriptor = {};
                if (fault == 1) malformed.nativeTextureDescriptor.deviceOrContextIdentity ^= 1u;
                if (fault == 2) ++malformed.nativeTextureDescriptor.rendererGeneration;
                if (fault == 3) malformed.nativeTextureDescriptor.format = arbitgpu::NativeTexturePixelFormat::Rgba8Unorm;
                if (fault == 4) malformed.nativeTextureOwner.reset();
                if (renderer.renderComposite(&malformed, 1) != 0 || renderer.lastError().empty())
                    return reject("SDF composition accepted missing, foreign, stale or forged native metadata");
            }
        }
    }
    return true;
}

bool run(const arbitgl::GlFuncs& gl, std::string& error)
{
    auto expected = fixture::pixels(), inverted = expected;
    for (int y = 0; y < fixture::height; ++y)
        std::copy_n(expected.data() + y * fixture::width * 4, fixture::width * 4,
                    inverted.data() + (fixture::height - 1 - y) * fixture::width * 4);
    std::string rejected;
    if (fixture::upright(inverted, rejected)) { error = "Orientation oracle admitted an upside-down fixture"; return false; }
    videorender::FrameRenderer renderer;
    if (!renderer.initialize(&gl, fixture::width, fixture::height, error)) return false;
    auto& backend = arbitgpu::nativeFixtureSceneBackend();
    const auto scene = fixture::screen(error);
    if (!scene) return false;
    const auto texture = renderer.uploadRgba(expected.data(), fixture::width, fixture::height, fixture::width * 4, 0);
    const auto decoded = renderer.leaseDecodedFrame(texture, fixture::width, fixture::height, {}, error);
    renderer.deleteTexture(texture);
    if (!decoded || decoded->colorTextureDescriptor().rowOrder != arbitgpu::NativeTextureRowOrder::TopFirst
        || texturePixels(gl, static_cast<unsigned>(decoded->colorTextureViewHandle())) != expected
        || !composite(renderer, gl, layerFor(decoded), error)) return false;
    const auto resources = backend.prepare(scene, nullptr);
    const auto embedded = resources.prepared ? backend.render(scene, resources.resources, fixture::width, fixture::height, {})
        : arbitgpu::NativeFixtureSceneSubmission {};
    if (!embedded.rendered) { error = resources.error + embedded.error; return false; }
    if (!composite(renderer, gl, layerFor(embedded.frame), error)) return false;
    const auto material = fixture::frameMaterial(*scene, "opengl");
    const auto frameResources = backend.prepare(scene, material);
    if (!frameResources.prepared) { error = frameResources.error; return false; }
    for (const auto& source : {decoded, embedded.frame})
    {
        arbitgpu::NativeFixtureSceneRuntimeInputs inputs; inputs.materialFrameTexture = source;
        const auto drawn = backend.render(scene, frameResources.resources, fixture::width, fixture::height, inputs);
        if (!drawn.rendered || !composite(renderer, gl, layerFor(drawn.frame), error)) return false;
        arbitgpu::NativeRawPassPixels raw;
        if (!drawn.frame->readRawPass(renderpassoutput::Output::Color, raw, error)
            || !fixture::upright(raw.bytes, error)) return false;
    }
    auto a = layerFor(decoded), b = layerFor(embedded.frame);
    // Dissolve preserves every oracle pixel when both images are identical.
    b.fromLayer = &a; b.transitionType = static_cast<int>(videofx::TransitionType::Dissolve);
    b.transitionProgress = 0.35f;
    if (!composite(renderer, gl, b, error)) return false;
    b.fromLayer = nullptr; b.transitionType = static_cast<int>(videofx::TransitionType::None);
    a.fromLayer = &b; a.transitionType = static_cast<int>(videofx::TransitionType::Dissolve);
    a.transitionProgress = 0.65f;
    if (!composite(renderer, gl, a, error)) return false;
    std::vector<std::uint8_t> async;
    bool havePrevious = false;
    if (!renderer.renderToPixelsAsync(&b, 1, async, havePrevious, error)
        || !renderer.renderToPixelsAsync(&b, 1, async, havePrevious, error)
        || !havePrevious || !fixture::upright(async, error)
        || !renderer.drainAsyncReadback(async) || !fixture::upright(async, error)) return false;
    if (!geometryCases(renderer, gl, backend, decoded, embedded.frame, error)
        || !depthAndHdrCases(renderer, backend, error)
        || !sdfPublicationCases(renderer, gl, error)) return false;
    // Neither publication nor native Frame sampling may mutate retained inputs.
    arbitgpu::NativeRawPassPixels retained;
    return embedded.frame->readRawPass(renderpassoutput::Output::Color, retained, error)
        && fixture::upright(retained.bytes, error)
        && texturePixels(gl, static_cast<unsigned>(decoded->colorTextureViewHandle())) == expected
        && glGetError() == GL_NO_ERROR;
}
} // namespace

int main(int argc, char** argv)
{
    if (argc != 1 && (argc != 3 || std::string(argv[1]) != "--typed-scene-passes"))
    { std::cerr << "Usage: arbit-native-scene-orientation-gl-tests [--typed-scene-passes fixtures.json]\n"; return 1; }
    if (!glfwInit()) { std::cerr << "SKIP: no native GL display\n"; return 77; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    auto* window = glfwCreateWindow(64, 36, "Native scene orientation", nullptr, nullptr);
    if (!window) { glfwTerminate(); std::cerr << "SKIP: no native GL 4.3 context\n"; return 77; }
    glfwMakeContextCurrent(window);
    arbitgl::GlFuncs gl;
    std::string error;
    const bool passed = arbitgl::loadGlFunctions(gl, error)
        && (argc == 3 ? typedscenepasstests::run(argv[2], gl, error) : run(gl, error));
    if (!passed) std::cerr << "FAIL: native scene orientation: " << error << '\n';
    else std::cout << (argc == 3 ? "PASS: typed scene graph -> native preparation -> SDR compositor preview/export\n"
                                : "PASS: native scene/Frame/Geometry Core/SDF preview and export orientation\n");
    arbitgpu::invalidateNativeSdfExecutionContext(reinterpret_cast<std::uintptr_t>(window));
    glfwDestroyWindow(window); glfwTerminate();
    return passed ? 0 : 1;
}
