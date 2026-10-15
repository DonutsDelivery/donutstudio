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
    if (!captured || hdr.size() * sizeof(std::uint16_t) != color.bytes.size()) return false;
    for (std::size_t i = 0; i < hdr.size(); ++i)
    {
        std::uint16_t half; std::memcpy(&half, color.bytes.data() + i * 2, 2);
        if (std::abs(hdr[i] - hdrimage::halfToFloat(half)) > 0.006f)
        { error = "Native row normalization changed linear HDR values or their raster positions"; return false; }
    }
    // Outside the white glyphs, the blue quadrant has ambient 2 * diffuse 0.5
    // plus blue emission 4, so its authored linear pixel is exactly (0, 0, 5).
    const std::size_t blueProbe = (3 * fixture::height / 4 * fixture::width + fixture::width / 2 - 2) * 4;
    if (hdr[blueProbe] != 0.0f || hdr[blueProbe + 1] != 0.0f
        || hdr[blueProbe + 2] != 5.0f || hdr[blueProbe + 3] != 1.0f)
    { error = "HDR blue quadrant differs from independent authored diffuse and emission light"; return false; }
    std::vector<float> pq;
    if (!renderer.readLastCompositePq(10000.0f, pq, error) || pq.size() != hdr.size()) return false;
    const auto encodePq = [](double linear) {
        const double m1 = 2610.0 / 16384.0, m2 = 2523.0 / 32.0;
        const double c1 = 3424.0 / 4096.0, c2 = 2413.0 / 128.0;
        const double c3 = 2392.0 / 128.0;
        const double power = std::pow(linear * 203.0 / 10000.0, m1);
        return std::pow((c1 + c2 * power) / (1.0 + c3 * power), m2);
    };
    constexpr double srgbTo2020[3][3] = {
        {0.6274040, 0.3292820, 0.0433136},
        {0.0690970, 0.9195400, 0.0113612},
        {0.0163916, 0.0880132, 0.8955950}
    };
    bool haveHighlight = false;
    for (std::size_t i = 0; i < hdr.size(); i += 4)
    {
        haveHighlight |= hdr[i] > 2.0f || hdr[i + 1] > 2.0f || hdr[i + 2] > 2.0f;
        for (int c = 0; c < 3; ++c)
        {
            const double rec2020 = srgbTo2020[c][0] * hdr[i]
                + srgbTo2020[c][1] * hdr[i + 1] + srgbTo2020[c][2] * hdr[i + 2];
            if (std::abs(pq[i + c] - encodePq(rec2020)) > 0.003)
            { error = "PQ final composite differs from independent Rec.2020 and ST 2084 pixels"; return false; }
        }
        if (pq[i + 3] != 1.0f)
        { error = "PQ final composite lost opaque alpha"; return false; }
    }
    if (!haveHighlight || renderer.readLastCompositePq(400.0f, pq, error)
        || error.find("declared luminance range") == std::string::npos)
    { error = "PQ movie accepted Scene3D pixels above the requested peak"; return false; }
    // Independent forward HLG EOTF oracle, rather than the production inverse.
    // ITU-R BT.2100-3 Table 5, reference black and surround, and Note 5f gamma.
    const auto decodeHlg = [](const float* encoded, double peak) {
        std::array<double, 3> sceneLight {};
        constexpr double a = 0.17883277, b = 0.28466892, c = 0.55991073;
        for (int channel = 0; channel < 3; ++channel)
            sceneLight[channel] = encoded[channel] <= 0.5
                ? encoded[channel] * encoded[channel] / 3.0
                : (std::exp((encoded[channel] - c) / a) + b) / 12.0;
        const double gamma = peak <= 2000.0 ? 1.2 + 0.42 * std::log10(peak / 1000.0)
            : 1.2 * std::pow(1.111, std::log2(peak / 1000.0));
        const double luminance = 0.2627 * sceneLight[0] + 0.6780 * sceneLight[1]
            + 0.0593 * sceneLight[2];
        for (auto& component : sceneLight)
            component *= peak * std::pow(luminance, gamma - 1.0);
        return sceneLight;
    };
    std::vector<float> hlg;
    // At 2000 nits, inverse OOTF of the authored blue pixel requires scene
    // blue 1.025569224. Its display components fit the peak, but the normalized
    // HLG signal cannot represent it. At 4000/10000 the scene blue is below one.
    if (renderer.readLastCompositeHlg(2000.0f, hlg, error) || !hlg.empty()
        || error.find("normalized scene-light range") == std::string::npos)
    { error = "HLG movie accepted the unrepresentable authored blue quadrant at 2000 nits"; return false; }
    for (const float peak : {4000.0f, 10000.0f})
    {
        if (!renderer.readLastCompositeHlg(peak, hlg, error) || hlg.size() != hdr.size()) return false;
        for (std::size_t i = 0; i < hdr.size(); i += 4)
        {
            const auto decoded = decodeHlg(hlg.data() + i, peak);
            for (int channel = 0; channel < 3; ++channel)
            {
                const double nits = 203.0 * (srgbTo2020[channel][0] * hdr[i]
                    + srgbTo2020[channel][1] * hdr[i + 1] + srgbTo2020[channel][2] * hdr[i + 2]);
                if (std::abs(decoded[channel] - nits) > std::max(0.02, nits * 0.0001))
                { error = "HLG final composite differs from independent Rec.2020 and BT.2100 display pixels"; return false; }
            }
            if (hlg[i + 3] != 1.0f)
            { error = "HLG final composite lost opaque alpha"; return false; }
        }
    }
    if (renderer.readLastCompositeHlg(400.0f, hlg, error) || !hlg.empty()
        || error.find("declared luminance range") == std::string::npos)
    { error = "HLG movie accepted Scene3D pixels above the requested peak"; return false; }
    // Keep the source frame unchanged, including after failed conversion.
    arbitgpu::NativeRawPassPixels retained;
    if (!frame.frame->readRawPass(renderpassoutput::Output::Color, retained, error)
        || retained.bytes != color.bytes) return false;
    float previousNeutralPq = -1.0f;
    for (const auto emission : {std::array<float, 3> {0, 0, 0}, {0.015625f, 0.015625f, 0.015625f},
                               {1, 1, 1}, {4, 4, 4}, {4.92578125f, 4.92578125f, 4.92578125f},
                               {4.9296875f, 4.9296875f, 4.9296875f}, {48, 0, 0}})
    {
        scene = fixture::screen(error);
        if (!scene) return false;
        scene->ambientColor = {};
        scene->materials[0].baseColor = {};
        scene->materials[0].emissive = {emission[0], emission[1], emission[2]};
        scene->materials[0].emissiveTexture = {};
        resources = backend.prepare(scene, nullptr);
        frame = resources.prepared ? backend.render(scene, resources.resources, fixture::width, fixture::height, inputs)
            : arbitgpu::NativeFixtureSceneSubmission {};
        if (!frame.rendered) { error = resources.error + frame.error; return false; }
        layer = layerFor(frame.frame);
        if (!renderer.renderComposite(&layer, 1) || !renderer.readLastCompositeFloat(hdr, error)) return false;
        if (emission[0] == 4.9296875f)
        {
            if (renderer.readLastCompositePq(1000, pq, error) || !pq.empty()
                || renderer.readLastCompositeHlg(1000, hlg, error) || !hlg.empty())
            { error = "HDR admitted the next over-peak linear binary16 pixel"; return false; }
            continue;
        }
        if (emission[0] == 4.92578125f)
        {
            const auto center = (fixture::height / 2 * fixture::width + fixture::width / 2) * 4;
            if (hdr.size() != static_cast<std::size_t>(fixture::width * fixture::height * 4))
            { error = "PQ boundary source HDR extent differs from the authored frame"; return false; }
            for (int channel = 0; channel < 3; ++channel)
                if (hdr[center + channel] != emission[channel])
                {
                    error = "PQ boundary source HDR channel " + std::to_string(channel)
                        + " is " + std::to_string(hdr[center + channel])
                        + "; expected authored binary16 " + std::to_string(emission[channel]);
                    return false;
                }
            if (hdr[center + 3] != 1.0f)
            { error = "PQ boundary source HDR alpha differs from authored opaque alpha"; return false; }
            if (!renderer.readLastCompositePq(1000, pq, error)) return false;
            if (pq.size() != hdr.size())
            {
                error = "PQ boundary output size is " + std::to_string(pq.size())
                    + "; expected " + std::to_string(hdr.size());
                return false;
            }
            // Independent ST.2084 gives .7518198570936413 at the exact
            // 999.93359375-nit source. RGBA16F storage can truncate or round
            // to nearest. Only its two enclosing binary16 values qualify;
            // the next source half above 1000 nits is still rejected above.
            const auto ideal = encodePq(emission[0]);
            const auto lower = static_cast<float>(std::floor(ideal * 2048.0) / 2048.0);
            const auto upper = static_cast<float>(std::ceil(ideal * 2048.0) / 2048.0);
            if (lower != 0.75146484375f || upper != 0.751953125f)
            { error = "Independent PQ boundary storage endpoints changed"; return false; }
            for (int channel = 0; channel < 3; ++channel)
                if (pq[center + channel] != lower && pq[center + channel] != upper)
                {
                    error = "PQ boundary center channel " + std::to_string(channel)
                        + " code is " + std::to_string(pq[center + channel])
                        + "; expected an enclosing binary16 code for 999.93359375 nits";
                    return false;
                }
            if (pq[center + 3] != 1.0f)
            { error = "PQ boundary output lost opaque alpha"; return false; }
        }
        if (emission[0] == 48)
        {
            // Components fit the declared display peak, but saturated red would
            // need a scene-light component above unity after the inverse OOTF.
            if (renderer.readLastCompositeHlg(10000, hlg, error) || !hlg.empty()
                || error.find("normalized scene-light range") == std::string::npos)
            { error = "HLG movie clipped an unrepresentable saturated pixel"; return false; }
            continue;
        }
        if (emission[0] != 4.92578125f && !renderer.readLastCompositePq(1000, pq, error)) return false;
        if (pq.size() != static_cast<std::size_t>(fixture::width * fixture::height * 4))
        { error = "Neutral PQ output extent differs from the authored frame"; return false; }
        const auto pqCenter = (fixture::height / 2 * fixture::width + fixture::width / 2) * 4;
        const auto idealPq = encodePq(emission[0]);
        int pqExponent = 0; std::frexp(idealPq, &pqExponent);
        const auto pqStep = std::ldexp(1.0, std::max(pqExponent - 11, -24));
        const auto lowerPq = static_cast<float>(std::floor(idealPq / pqStep) * pqStep);
        const auto upperPq = static_cast<float>(std::ceil(idealPq / pqStep) * pqStep);
        for (int channel = 0; channel < 3; ++channel)
            if (pq[pqCenter + channel] != lowerPq && pq[pqCenter + channel] != upperPq)
            { error = "Neutral PQ output differs from the independent ST.2084 storage endpoints at linear "
                      + std::to_string(emission[0]); return false; }
        if (pq[pqCenter] <= previousNeutralPq || pq[pqCenter + 3] != 1.0f)
        { error = "Neutral PQ output lost monotonic luminance or opaque alpha"; return false; }
        previousNeutralPq = pq[pqCenter];
        const float peak = emission[0] == 4 ? 812.0f : 1000.0f;
        if (!renderer.readLastCompositeHlg(peak, hlg, error) || hlg.size() != hdr.size()) return false;
        const std::size_t center = (fixture::height / 2 * fixture::width + fixture::width / 2) * 4;
        const auto decoded = decodeHlg(hlg.data() + center, peak);
        for (int channel = 0; channel < 3; ++channel)
            if (std::abs(decoded[channel] - 203.0 * emission[channel]) > 0.02)
            { error = "HLG black, low-light, reference white or peak display value differs"; return false; }
        if ((emission[0] == 0 && hlg[center] != 0.0f)
            || (emission[0] == 1 && std::abs(hlg[center] - 0.75f) > 0.001f)
            || (emission[0] == 4 && std::abs(hlg[center] - 1.0f) > 0.00001f))
        { error = "HLG reference signal values differ"; return false; }
        if (emission[0] == 1)
        {
            if (!renderer.readLastCompositeHlg(400, hlg, error)) return false;
            const auto lowPeakWhite = decodeHlg(hlg.data() + center, 400);
            for (const auto component : lowPeakWhite)
                if (std::abs(component - 203.0) > 0.02)
                { error = "HLG minimum-peak output changed reference white luminance"; return false; }
        }
    }
    for (const auto peak : {399.0f, 10001.0f})
        if (renderer.readLastCompositeHlg(peak, hlg, error) || !hlg.empty())
        { error = "HLG readback accepted an invalid declared peak"; return false; }
    renderer.setHdrImageCapture(false);
    if (renderer.readLastCompositeHlg(1000, hlg, error) || !hlg.empty())
    { error = "HLG readback accepted a compositor without HDR capture"; return false; }
    return true;
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
    const auto fogPlan = sdfdisplayoracle::plan(0, true);
    std::vector<std::uint8_t> fogPreview;
    for (const auto use : {NativeSdfRenderUse::Preview, NativeSdfRenderUse::Export})
    {
        videorender::LayerDesc layer;
        std::vector<NativeSdfRenderedFrame> owners;
        if (prepareVisualSdfLayer({fogPlan}, fogPlan.clipId, fixture::width, fixture::height,
                use, sdfRenderer, layer, owners, error) != VisualSdfPreparation::rendered
            || owners.size() != 1 || !layer.depthFog || layer.depthEffect != 1
            || layer.nativeDepthTextureDescriptor.format != arbitgpu::NativeTexturePixelFormat::R32Float)
            return reject("SDF Depth Fog lost its typed raw attachment");
        std::vector<float> depth;
        if (!owners.front().nativeFrame->readDepthFloatPixels(depth)
            || depth.size() != static_cast<std::size_t>(fixture::width) * fixture::height
            || std::abs(depth[12 * fixture::width + 32] - 0.327399042269f) > 0.002f
            || depth[24 * fixture::width + 32] != 1.0f)
            return reject("SDF Depth Fog received values other than analytic hit and miss depth");
        owners.clear();
        const auto texture = renderer.renderComposite(&layer, 1);
        if (!texture) { error = renderer.lastError(); return false; }
        const auto pixels = texturePixels(gl, texture);
        const auto upper = (12 * fixture::width + 32) * 4;
        const auto lower = (24 * fixture::width + 32) * 4;
        if (pixels[upper] <= 56 || pixels[lower] <= 46
            || pixels[upper] >= pixels[lower] || pixels[lower + 3] != 255)
            return reject("SDF Depth Fog preview did not apply distinct hit and miss fog");
        std::vector<std::uint8_t> exported;
        if (!renderer.renderToPixels(&layer, 1, exported, error) || pixels != exported)
            return reject("SDF Depth Fog preview and export pixels disagree");
        auto malformed = layer;
        ++malformed.nativeDepthTextureDescriptor.rendererGeneration;
        if (renderer.renderComposite(&malformed, 1) != 0 || renderer.lastError().empty())
            return reject("SDF Depth Fog accepted a stale Depth owner");
        if (use == NativeSdfRenderUse::Preview) fogPreview = pixels;
        else if (fogPreview != pixels) return reject("SDF Depth Fog changes with render use");
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
