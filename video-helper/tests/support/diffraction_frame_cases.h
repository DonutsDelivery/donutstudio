#pragma once

#include "material_frame_binding_cases.h"
#include "material_field_fixture.h"
#include "../../src/fixture_scene_renderer.h"
#include "../../../shared/VisualImportedSceneRenderOperationContract.h"

namespace videohelper::tests
{
// Uses real native images. Scheduler/source-time validation is covered by the
// shared prerequisite tests; decoder freeze/retime is covered by lease cases.
template <typename Backend, typename ReadPixels>
bool diffractionFrameCases(Backend& backend,
    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& card,
    visualimportedscenerender::Request request,
    materialprogram::BackendTarget target, ReadPixels readPixels)
{
    using namespace arbitgpu;
    using namespace HarmonicMIDI::grid;
    bool ok = true;
    const auto check = [&](bool value, const char* message)
    { if (!value) { std::cerr << "FAIL: " << message << '\n'; ok = false; } return value; };
    std::string error;
    auto& binding = *request.diffractionMaterial;
    binding.version = diffractionmaterialbinding::kGraphFrameWireVersion;
    binding.graphFrame = surfacematerialbinding::TextureSlotBinding::GraphFrameEndpoint {91, 0};
    visualimportedscenerender::Request decoded;
    if (!check(visualimportedscenerender::decodeCanonical(
            visualimportedscenerender::encode(request), decoded, error),
            "card imagery request round trips through the ordinary scene transport")) return false;
    const auto admitted = videorender::fixture3d::admitDiffractionMaterialBinding(
        card, *decoded.diffractionMaterial, target, error);
    if (!check(admitted != nullptr, "physical diffraction admits the existing Frame binding")) return false;
    auto replacedEndpoint = *decoded.diffractionMaterial;
    ++replacedEndpoint.graphFrame->node;
    const auto replacedBinding = videorender::fixture3d::admitDiffractionMaterialBinding(
        card, replacedEndpoint, target, error);
    check(replacedBinding && replacedBinding->bindingDigest() != admitted->bindingDigest(),
          "another authored imagery source cannot alias the current material binding");

    const auto source = [&](bool pattern, float opacity)
    {
        auto value = fixture3d::makeScene();
        value.vertexCount = 4; value.indexCount = 6;
        value.vertices[0] = {{-2, 2, 0}, {0, 0, 1}, {}, {0, 0}};
        value.vertices[1] = {{ 2, 2, 0}, {0, 0, 1}, {}, {1, 0}};
        value.vertices[2] = {{-2,-2, 0}, {0, 0, 1}, {}, {0, 1}};
        value.vertices[3] = {{ 2,-2, 0}, {0, 0, 1}, {}, {1, 1}};
        const std::array<std::uint32_t, 6> indices {0, 2, 1, 1, 2, 3};
        std::copy(indices.begin(), indices.end(), value.indices.begin());
        value.objects[0].transform = {};
        value.objects[0].vertexCount = 4; value.objects[0].indexCount = 6;
        value.cameras[0].transform = {}; value.cameras[0].transform.translation.z = 2;
        value.lightCount = 0; value.ambientColor = {1, 1, 1};
        value.materials[0].baseColor = {1, 1, 1};
        value.materials[0].metallic = 0; value.materials[0].roughness = 0;
        value.materials[0].opacity = opacity; value.materials[0].doubleSided = true;
        value.materials[0].baseColorTexture = pattern ? fixture3d::kCubeTextureId : SceneTextureId {};
        value.textureTexels[0] = {255, 0, 0, 255};
        value.textureTexels[1] = {0, 255, 0, 255};
        value.textureTexels[2] = {0, 0, 255, 255};
        value.textureTexels[3] = {128, 128, 128, 255};
        value.textures[0].minFilter = value.textures[0].magFilter = 9728;
        auto scene = std::make_shared<const Visual3DScene>(value);
        auto prepared = backend.prepare(scene, nullptr);
        auto result = backend.render(scene, prepared.resources, 64, 64, {});
        check(result.rendered, "card imagery fixture produces a real immutable native image");
        return result.frame;
    };
    auto white = source(false, 1);
    auto pattern = source(true, 1);
    auto halfAlpha = source(false, 0.5f);
    if (!white || !pattern || !halfAlpha) return false;
    videorender::fixture3d::FixtureSceneRenderer renderer(backend);
    NativeFixtureSceneRuntimeInputs runtime;
    runtime.cameraOverride = request.camera; runtime.lightOverride = request.light;
    NativeFixtureSceneStats lastStats;
    const auto draw = [&](std::shared_ptr<const NativeFixtureSceneFrame> input, bool exporting)
    {
        runtime.materialFrameTexture = std::move(input);
        videorender::fixture3d::RenderedFrame result;
        const bool rendered = exporting
            ? renderer.renderExport(card, admitted, runtime, {96, 96}, "native-gpu", result, error)
            : renderer.renderPreview(card, admitted, runtime, {96, 96}, "native-gpu", result, error);
        lastStats = result.stats;
        return rendered ? result.nativeFrame : std::shared_ptr<const NativeFixtureSceneFrame> {};
    };
    check(!draw({}, false) && !draw({}, true), "card imagery cannot reuse imported pixels for a missing lease");
    for (int fault = 0; fault < 3; ++fault)
    {
        auto descriptor = pattern->colorTextureDescriptor();
        if (fault == 0) ++descriptor.deviceOrContextIdentity;
        if (fault == 1) --descriptor.width;
        if (fault == 2) descriptor.rendererGeneration = 0;
        auto invalid = std::make_shared<MaterialFrameDescriptorOverride>(pattern, descriptor);
        check(!draw(invalid, false) && !draw(invalid, true),
              "card imagery rejects stale generation, wrong extent and foreign device in preview/export");
    }
    auto whiteResult = draw(white, false);
    auto printed = draw(pattern, false);
    auto exported = draw(pattern, true);
    auto translucent = draw(halfAlpha, false);
    if (!check(whiteResult && printed && exported && translucent,
               "physical diffraction and imagery render together")) return false;
    const auto printedPixels = readPixels(printed);
    check(printedPixels.size() == 96u * 96u * 4u
          && printedPixels == readPixels(exported) && printedPixels != readPixels(whiteResult),
          "asymmetric card imagery changes pixels and preview/export agree");
    {
        runtime.diffractionParameters = {{"rmsHeightNanometres", 30}, {"period00Nanometres", 1100},
                                         {"angle11Degrees", 30}, {"mask10", 1}};
        auto animated = draw(pattern, false);
        check(animated && lastStats.reusedStaticResources && lastStats.staticUploadCount == 0,
              "animated foil uses frame-local admitted uniforms without reuploading geometry or compiling pipelines");
        auto animatedExport = draw(pattern, true);
        check(animated && animatedExport && readPixels(animated) != printedPixels
                  && readPixels(animated) == readPixels(animatedExport),
              "modulated physical foil and imagery share OpenGL/Metal preview/export pixels");
        const auto authoredDigest = admitted->bindingDigest();
        for (const auto& invalid : std::vector<diffractionmaterialbinding::RuntimeParameters> {
                {{"profile", 2}}, {{"lattice", 1}}, {{"coating", 1}}, {{"grooveFieldMode", 2}},
                {{"period00Nanometres", 0}}, {{"rmsSlope", 0}}, {{"maximumEvaluations", 1}},
                {{"emissionGain", 2}}, {{"rmsHeightNanometres", std::numeric_limits<double>::quiet_NaN()}}})
        {
            runtime.diffractionParameters = invalid;
            check(!draw(pattern, false) && !draw(pattern, true),
                  "unsupported material mode, work budget, emission or invalid physical modulation fails closed");
        }
        runtime.diffractionParameters.clear();
        auto returned = draw(pattern, false);
        check(returned && readPixels(returned) == printedPixels && admitted->bindingDigest() == authoredDigest,
              "seeking back after invalid and valid modulation restores the immutable card material");
    }
    const auto originalLight = runtime.lightOverride;
    {
        const auto field = materialFieldFixture(72);
        videowire::geometry::RuntimeFieldEvaluation clock;
        clock.timelineSeconds = 1;
        check(materialfield::evaluate(field, clock, binding, runtime.diffractionParameters, error),
              "canonical Geometry Field evaluates before the physical draw");
        auto fieldPreview = draw(pattern, false);
        auto fieldExport = draw(pattern, true);
        check(fieldPreview && fieldExport && readPixels(fieldPreview) != printedPixels
                  && readPixels(fieldPreview) == readPixels(fieldExport),
              "OpenGL/Metal field-driven physical response changes printed pixels with preview/export parity");
        runtime.diffractionParameters.clear();
        clock.timelineSeconds = 0;
        check(materialfield::evaluate(field, clock, binding, runtime.diffractionParameters, error),
              "canonical Geometry Field reevaluates on backward seek");
        auto fieldSeek = draw(pattern, false);
        check(fieldSeek && readPixels(fieldSeek) == printedPixels, "seeking the Field back restores physical card pixels");
        runtime.diffractionParameters.clear();
    }
    if (runtime.lightOverride)
    {
        runtime.lightOverride->intensity = 0;
        auto darkKey = draw(pattern, false);
        auto darkExport = draw(pattern, true);
        check(darkKey && darkExport && readPixels(darkKey) != printedPixels
                  && readPixels(darkKey) == readPixels(darkExport),
              "physical key lighting still changes the image-tinted foil in preview/export");
        runtime.lightOverride = originalLight;
    }
    auto soughtBack = draw(pattern, false);
    check(soughtBack && printedPixels == readPixels(soughtBack),
          "source replacement then returning to the retained image repeats card pixels");
    const auto alphaPixels = readPixels(translucent);
    const auto alphaExport = draw(halfAlpha, true);
    check(alphaExport && alphaPixels == readPixels(alphaExport),
          "source alpha has the same diffraction composition in preview and export");
    constexpr std::size_t centerAlpha = (48u * 96u + 48u) * 4u + 3u;
    check(alphaPixels.size() == 96u * 96u * 4u
              && alphaPixels[centerAlpha] >= 126 && alphaPixels[centerAlpha] <= 129,
          "card source alpha survives all diffraction lighting passes exactly once");
    std::weak_ptr<const NativeFixtureSceneFrame> lifetime = pattern;
    pattern.reset(); runtime.materialFrameTexture.reset();
    check(!lifetime.expired(), "card draw outputs retain their input Frame lease");
    printed.reset(); exported.reset(); soughtBack.reset();
    check(lifetime.expired(), "card imagery releases after its final consuming output");
    return ok;
}
} // namespace videohelper::tests
