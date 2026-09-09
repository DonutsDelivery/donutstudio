#include "VisualAnimationDeformationEvaluation.h"
#include "gpu_backend/backend.h"
#include "native_animation_deformation_renderer.h"
#include "support/fixture_scene.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
using namespace visualdeformation;
namespace animation = visualanimation;
using videorender::animation3d::NativeAnimationDeformationRenderer;
using videorender::animation3d::RenderedDeformationFrame;

int failures = 0;

void check (bool condition, const char* message)
{
    if (condition)
        return;
    ++failures;
    std::fprintf (stderr, "FAIL: %s\n", message);
}

bool near (float actual, float expected, float tolerance = 1.0e-5f)
{
    return std::abs (actual - expected) <= tolerance;
}

std::array<float, 16> identityMatrix()
{
    return { 1.0f, 0.0f, 0.0f, 0.0f,
             0.0f, 1.0f, 0.0f, 0.0f,
             0.0f, 0.0f, 1.0f, 0.0f,
             0.0f, 0.0f, 0.0f, 1.0f };
}

struct Fixture
{
    HarmonicMIDI::grid::Visual3DScene sceneValue = videohelper::fixture3d::makeScene();
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> scene;
    std::array<JointView, 1> joints { JointView { JointId { 301 }, {} } };
    std::array<float, 16> inverseBind = identityMatrix();
    std::array<std::uint32_t, 24 * 4> jointIndices {};
    std::array<float, 24 * 4> jointWeights {};
    std::array<float, 24 * 3> morphPositions {};
    std::array<float, 24 * 3> morphNormals {};
    JointWeightSetView weightSet;
    MorphTargetView morphTarget;
    SkinView skinView;
    MeshView meshView;
    AssetView assetView;
    std::shared_ptr<const DeformationAsset> asset;

    std::array<double, 2> keyTimes { 0.0, 1.0 };
    std::array<float, 6> translationValues { 0.0f, 0.0f, 0.0f,
                                             2.0f, 0.0f, 0.0f };
    std::array<float, 2> morphValues { 0.0f, 0.5f };
    std::array<animation::TrackView, 2> tracks;
    animation::ClipView clipView;
    std::shared_ptr<const animation::Clip> clip;

    std::array<JointAnimationBindingView, 1> jointBindings;
    std::array<MorphTargetId, 1> morphTargets { MorphTargetId { 601 } };
    std::array<MorphAnimationBindingView, 1> morphBindings;
    AnimationDeformationBindingView bindings;
    std::shared_ptr<const AnimationDeformationSnapshot> snapshot;
    std::shared_ptr<const arbitgpu::NativeDeformationScene> source;

    Fixture()
    {
        sceneValue.materials[0].baseColorTexture = {};
        sceneValue.textureCount = 0;
        sceneValue.textureTexelCount = 0;
        sceneValue.objects[0].transform = {};
        scene = std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (sceneValue);

        for (std::size_t vertex = 0; vertex < 24; ++vertex)
        {
            jointWeights[vertex * 4] = 1.0f;
            morphPositions[vertex * 3] = vertex == 0 ? 4.0f : 0.0f;
            morphNormals[vertex * 3 + 2] = vertex == 0 ? 1.0f : 0.0f;
        }
        weightSet = { jointIndices.data(), jointIndices.size(),
                      jointWeights.data(), jointWeights.size() };
        morphTarget = { MorphTargetId { 601 }, morphPositions.data(),
                        morphPositions.size(), morphNormals.data(),
                        morphNormals.size(), nullptr, 0 };
        skinView = { SkinId { 401 }, joints.data(), joints.size(),
                     inverseBind.data(), inverseBind.size() };
        meshView = { MeshId { 501 }, 24, skinView.id,
                     &weightSet, 1, &morphTarget, 1 };
        assetView = { &skinView, 1, &meshView, 1 };
        std::string error;
        asset = DeformationAsset::create (assetView, {}, error);
        check (asset != nullptr && error.empty(), "fixture deformation asset is admitted");

        tracks = {
            animation::TrackView { animation::TrackId { 11 }, animation::TargetId { 1001 },
                animation::Channel::Translation, animation::Interpolation::Linear,
                keyTimes.data(), translationValues.data(), keyTimes.size(),
                translationValues.size(), 0 },
            animation::TrackView { animation::TrackId { 12 }, animation::TargetId { 1002 },
                animation::Channel::MorphWeights, animation::Interpolation::Linear,
                keyTimes.data(), morphValues.data(), keyTimes.size(), morphValues.size(), 1 }
        };
        clipView = { animation::ClipId { 701 }, 1.0, tracks.data(), tracks.size() };
        clip = animation::Clip::create (clipView, {}, error);
        check (clip != nullptr && error.empty(), "fixture animation clip is admitted");

        jointBindings = { JointAnimationBindingView {
            animation::TargetId { 1001 }, skinView.id, joints[0].id } };
        morphBindings = { MorphAnimationBindingView {
            animation::TargetId { 1002 }, meshView.id,
            morphTargets.data(), morphTargets.size() } };
        bindings = { jointBindings.data(), jointBindings.size(),
                     morphBindings.data(), morphBindings.size() };
        if (asset && clip)
        {
            const AnimationDeformationRequest request {
                RationalFrameTime { 15, 30, 1 }, 77, animation::Playback::Clamp,
                std::nullopt
            };
            snapshot = evaluateAnimationDeformation (*asset, *clip, bindings,
                                                      request, {}, error);
        }
        check (snapshot != nullptr && error.empty(), "fixture deformation snapshot evaluates");

        auto mutableSource = std::make_shared<arbitgpu::NativeDeformationScene>();
        mutableSource->sourceStableId = 801;
        mutableSource->deformationStableId = 802;
        mutableSource->structuralRevision = 77;
        mutableSource->clip = animation::ClipId { 701 };
        mutableSource->mesh = meshView.id;
        mutableSource->object = videohelper::fixture3d::kCubeObjectId;
        mutableSource->scene = scene;
        mutableSource->deformation = asset;
        mutableSource->jointBaseTransforms = {
            arbitgpu::NativeDeformationJointBaseTransform {
                skinView.id, joints[0].id,
                { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 1.0f },
                { 1.0f, 1.0f, 1.0f } }
        };
        mutableSource->morphBaseWeights = { 0.125f };
        source = std::move (mutableSource);
    }
};

class FakeResources final : public arbitgpu::NativeDeformationResources
{
public:
    const std::string& backend() const noexcept override { return backend_; }
private:
    std::string backend_ = "opengl";
};

class FakeFrame final : public arbitgpu::NativeFixtureSceneFrame
{
public:
    FakeFrame (std::uint32_t width, std::uint32_t height) : width_ (width), height_ (height) {}
    const std::string& backend() const noexcept override { return backend_; }
    std::uint32_t width() const noexcept override { return width_; }
    std::uint32_t height() const noexcept override { return height_; }
    std::uintptr_t colorImageHandle() const noexcept override { return 101; }
    std::uintptr_t colorTextureViewHandle() const noexcept override { return 102; }
    std::uintptr_t depthImageHandle() const noexcept override { return 103; }
    std::uintptr_t depthTextureViewHandle() const noexcept override { return 104; }
private:
    std::string backend_ = "opengl";
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
};

class FakeBackend final : public arbitgpu::NativeDeformationBackend
{
public:
    arbitgpu::BackendInfo info() const override
    {
        return { true, true, "opengl", "strict-test-device", {} };
    }

    arbitgpu::NativeDeformationPreparation prepare (
        const std::shared_ptr<const arbitgpu::NativeDeformationScene>& source,
        const std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram>& materialProgram) override
    {
        ++prepareCalls;
        preparedSource = source;
        preparedMaterialProgram = materialProgram;
        arbitgpu::NativeDeformationPreparation result;
        result.prepared = true;
        result.resources = std::make_shared<const FakeResources>();
        result.stats.staticVertexBytes = source->scene->vertexCount
            * sizeof (HarmonicMIDI::grid::SceneVertex);
        result.stats.staticDeformationBytes = 1234;
        return result;
    }

    arbitgpu::NativeDeformationSubmission render (
        const std::shared_ptr<const arbitgpu::NativeDeformationScene>& source,
        const std::shared_ptr<const AnimationDeformationSnapshot>& snapshot,
        const std::shared_ptr<const arbitgpu::NativeDeformationResources>& resources,
        std::uint32_t width,
        std::uint32_t height,
        arbitgpu::NativeDeformationRuntimeInputs runtimeInputs) override
    {
        ++renderCalls;
        renderedSource = source;
        renderedSnapshot = snapshot;
        renderedResources = resources;
        lastRuntimeInputs = runtimeInputs;
        arbitgpu::NativeDeformationSubmission result;
        if (! arbitgpu::prepareNativeDeformationFrame (*source, *snapshot,
                                                        lastFrameData, result.error))
            return result;
        result.rendered = true;
        result.frame = std::make_shared<const FakeFrame> (width, height);
        result.stats.drawCount = 1;
        result.stats.dispatchCount = 1;
        result.stats.dynamicUniformBytes = sizeof (arbitgpu::NativeDeformationFrameData);
        result.stats.sourceStableId = source->sourceStableId;
        result.stats.deformationStableId = source->deformationStableId;
        result.stats.clipId = snapshot->clip().value;
        result.stats.meshId = source->mesh.value;
        result.stats.skinId = source->deformation->findMesh (source->mesh)->skin().value;
        result.stats.revision = snapshot->revision();
        result.stats.time = snapshot->time();
        return result;
    }

    int prepareCalls = 0;
    int renderCalls = 0;
    arbitgpu::NativeDeformationFrameData lastFrameData;
    std::shared_ptr<const arbitgpu::NativeDeformationScene> preparedSource;
    std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram>
        preparedMaterialProgram;
    std::shared_ptr<const arbitgpu::NativeDeformationScene> renderedSource;
    std::shared_ptr<const AnimationDeformationSnapshot> renderedSnapshot;
    std::shared_ptr<const arbitgpu::NativeDeformationResources> renderedResources;
    arbitgpu::NativeDeformationRuntimeInputs lastRuntimeInputs;
};

void testFramePreparation()
{
    Fixture fixture;
    if (! fixture.source || ! fixture.snapshot)
        return;
    arbitgpu::NativeDeformationFrameData output;
    output.vertexCount = 999;
    std::string error;
    check (arbitgpu::prepareNativeDeformationFrame (*fixture.source, *fixture.snapshot,
                                                    output, error)
           && error.empty(),
           "exact immutable source and snapshot prepare native frame data");
    check (output.vertexCount == 24 && output.jointCount == 1
           && output.morphTargetCount == 1,
           "prepared counts exactly match admitted immutable input");
    check (near (output.jointPalette[12], 1.0f)
           && near (output.jointPalette[13], 0.0f)
           && near (output.jointPalette[14], 0.0f),
           "sampled joint transform becomes the native joint palette");
    check (near (output.morphWeights[0], 0.25f),
           "sampled morph weight replaces the imported base weight");

    AnimationDeformationRequest weightedRequest;
    weightedRequest.time = { 15, 30, 1 };
    weightedRequest.revision = 77;
    weightedRequest.combinationMode = animation::CombinationMode::WeightedBlend;
    weightedRequest.combinationWeight = 0.5;
    const auto weightedSnapshot = evaluateAnimationDeformation (
        *fixture.asset, *fixture.clip, fixture.bindings, weightedRequest, {}, error);
    arbitgpu::NativeDeformationFrameData weightedOutput;
    check (weightedSnapshot
           && arbitgpu::prepareNativeDeformationFrame (
               *fixture.source, *weightedSnapshot, weightedOutput, error)
           && near (weightedOutput.jointPalette[12], 0.5f)
           && near (weightedOutput.morphWeights[0], 0.1875f),
           "weighted blend deterministically mixes sampled joints and morphs with the imported base");

    AnimationDeformationRequest combinedRequest;
    combinedRequest.time = { 15, 30, 1 };
    combinedRequest.revision = 77;
    // Playback control resolves reverse before deformation evaluation. Keep the
    // exact presentation frame while sampling the reversed point in the trim.
    combinedRequest.requestedTimeSeconds = 0.75;
    combinedRequest.sampleRangeStartSeconds = 0.25;
    combinedRequest.sampleRangeEndSeconds = 0.75;
    combinedRequest.combinationMode = animation::CombinationMode::WeightedBlend;
    combinedRequest.combinationWeight = 0.5;
    const auto combinedSnapshot = evaluateAnimationDeformation(
        *fixture.asset, *fixture.clip, fixture.bindings, combinedRequest, {}, error);
    arbitgpu::NativeDeformationFrameData combinedOutput;
    check (combinedSnapshot
           && arbitgpu::prepareNativeDeformationFrame(
               *fixture.source, *combinedSnapshot, combinedOutput, error)
           && combinedSnapshot->time() == RationalFrameTime { 15, 30, 1 }
           && near(combinedOutput.jointPalette[12], 0.75f)
           && near(combinedOutput.morphWeights[0], 0.25f),
           "trimmed reverse playback, weighted mixing, skin, and morph execute in one immutable frame");

    weightedRequest.combinationMode = animation::CombinationMode::Add;
    const auto additiveSnapshot = evaluateAnimationDeformation (
        *fixture.asset, *fixture.clip, fixture.bindings, weightedRequest, {}, error);
    arbitgpu::NativeDeformationFrameData additiveOutput;
    check (additiveSnapshot
           && arbitgpu::prepareNativeDeformationFrame (
               *fixture.source, *additiveSnapshot, additiveOutput, error)
           && near (additiveOutput.jointPalette[12], 0.5f)
           && near (additiveOutput.morphWeights[0], 0.25f),
           "add mode applies weighted sampled deltas to the imported base pose");

    weightedRequest.combinationMode = animation::CombinationMode::Multiply;
    const auto multiplySnapshot = evaluateAnimationDeformation (
        *fixture.asset, *fixture.clip, fixture.bindings, weightedRequest, {}, error);
    arbitgpu::NativeDeformationFrameData multiplyOutput;
    check (multiplySnapshot
           && arbitgpu::prepareNativeDeformationFrame (
               *fixture.source, *multiplySnapshot, multiplyOutput, error)
           && near (multiplyOutput.jointPalette[12], 0.0f)
           && near (multiplyOutput.morphWeights[0], 0.078125f),
           "multiply mode scales the imported base pose by weighted sampled values");

    weightedRequest.combinationMode =
        animation::CombinationMode::AddAfterImportedAnimation;
    const auto postAddSnapshot = evaluateAnimationDeformation (
        *fixture.asset, *fixture.clip, fixture.bindings, weightedRequest, {}, error);
    arbitgpu::NativeDeformationFrameData postAddOutput;
    check (postAddSnapshot
           && arbitgpu::prepareNativeDeformationFrame (
               *fixture.source, *postAddSnapshot, postAddOutput, error)
           && near (postAddOutput.morphWeights[0], 0.25f),
           "add-after mode preserves a distinct admitted rotation order and additive morph path");

    weightedRequest.combinationMode = animation::CombinationMode::WeightedBlend;
    weightedRequest.combinationWeight = 1.5;
    check (! evaluateAnimationDeformation (
               *fixture.asset, *fixture.clip, fixture.bindings,
               weightedRequest, {}, error)
           && error == "animation deformation combination mode or weight is invalid",
           "weighted blend rejects an out-of-bounds weight before native execution");

    const auto unchanged = output;
    auto incompatible = *fixture.source;
    incompatible.structuralRevision = 78;
    check (! arbitgpu::prepareNativeDeformationFrame (incompatible, *fixture.snapshot,
                                                      output, error)
           && ! error.empty() && output.vertexCount == unchanged.vertexCount
           && output.jointPalette == unchanged.jointPalette,
           "identity rejection leaves the caller output unchanged");

    incompatible = *fixture.source;
    incompatible.morphBaseWeights = { 0.0f, 1.0f };
    check (! arbitgpu::prepareNativeDeformationFrame (incompatible, *fixture.snapshot,
                                                      output, error)
           && error.find ("base morph weights") != std::string::npos,
           "base morph weights require an exact bounded target range");

    incompatible = *fixture.source;
    incompatible.jointBaseTransforms[0].rotation[3] = 2.0f;
    check (! arbitgpu::prepareNativeDeformationFrame (incompatible, *fixture.snapshot,
                                                      output, error)
           && error.find ("not normalized") != std::string::npos,
           "imported base rotations are finite and normalized before native upload");
}

void testSharedPreviewExportRoute()
{
    Fixture fixture;
    if (! fixture.source || ! fixture.snapshot)
        return;
    FakeBackend backend;
    NativeAnimationDeformationRenderer renderer (backend);
    RenderedDeformationFrame preview;
    RenderedDeformationFrame exportFrame;
    std::string error;
    check (renderer.renderPreview (fixture.source, fixture.snapshot, 320, 180,
                                   videorender::animation3d::kNativeGpuCapability,
                                   preview, error),
           "preview renders through strict native backend");
    check (renderer.renderExport (fixture.source, fixture.snapshot, 320, 180,
                                  videorender::animation3d::kNativeGpuCapability,
                                  exportFrame, error),
           "export renders through the same strict native backend route");
    check (backend.prepareCalls == 1 && backend.renderCalls == 2,
           "preview and export share one exact-source static GPU preparation");
    check (preview.use == videorender::animation3d::RenderUse::Preview
           && exportFrame.use == videorender::animation3d::RenderUse::Export
           && preview.source == fixture.source && exportFrame.source == fixture.source
           && preview.deformation == fixture.snapshot
           && exportFrame.deformation == fixture.snapshot,
           "published frames retain immutable source and evaluation ownership");
    check (preview.stats.staticUploadCount == 1
           && ! preview.stats.reusedStaticResources
           && exportFrame.stats.staticUploadCount == 0
           && exportFrame.stats.reusedStaticResources,
           "cache telemetry distinguishes initial upload from exact-owner reuse");
    check (preview.stats.dispatchCount == 1 && preview.stats.drawCount == 1
           && exportFrame.stats.dispatchCount == 1 && exportFrame.stats.drawCount == 1
           && preview.nativeFrame && exportFrame.nativeFrame,
           "each route returns a native GPU dispatch and draw frame without CPU pixels");
    check (near (backend.lastFrameData.jointPalette[12], 1.0f)
           && near (backend.lastFrameData.morphWeights[0], 0.25f),
           "backend receives the same prepared deformation values for preview and export");

    auto material = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>();
    material->backend = arbitgpu::NativeFixtureMaterialBackend::OpenGl;
    material->object = fixture.source->object;
    material->bindingDigest = "deformed-material-binding";
    material->programIdentity = "deformed-material-program";
    arbitgpu::NativeDeformationRuntimeInputs previewInputs {};
    previewInputs.timeSeconds = 0.5f;
    arbitgpu::NativeDeformationRuntimeInputs exportInputs {};
    exportInputs.timeSeconds = 0.75f;
    check (renderer.renderPreview (
               fixture.source, fixture.snapshot, material, previewInputs, 320, 180,
               videorender::animation3d::kNativeGpuCapability, preview, error),
           "preview admits one immutable material program with deformation");
    check (renderer.renderExport (
               fixture.source, fixture.snapshot, material, exportInputs, 320, 180,
               videorender::animation3d::kNativeGpuCapability, exportFrame, error),
           "export shares the same deformed material resources");
    check (backend.prepareCalls == 2 && backend.renderCalls == 4
           && backend.preparedMaterialProgram == material
           && near (backend.lastRuntimeInputs.timeSeconds, 0.75f),
           "material identity invalidates static preparation once and time stays per draw");

    RenderedDeformationFrame unchanged = preview;
    check (! renderer.renderPreview (fixture.source, fixture.snapshot, 320, 180,
                                     "cpu", preview, error)
           && error.find ("without CPU fallback") != std::string::npos
           && preview.nativeFrame == unchanged.nativeFrame,
           "non-native capability is rejected without replacing prior output");
    check (! renderer.renderExport (fixture.source, fixture.snapshot, 0, 180,
                                    videorender::animation3d::kNativeGpuCapability,
                                    exportFrame, error)
           && error.find ("dimensions") != std::string::npos,
           "preview/export shared route enforces strict bounded dimensions");
}

void testUnavailableBackendHasNoFallback()
{
    Fixture fixture;
    if (! fixture.source || ! fixture.snapshot)
        return;
    auto& backend = arbitgpu::nativeDeformationBackend();
    const auto info = backend.info();
    check (! info.available && ! info.compute,
           "non-Metal focused test backend advertises no deformation execution");
    const auto prepared = backend.prepare (fixture.source);
    check (! prepared.prepared && ! prepared.resources && ! prepared.error.empty(),
           "unavailable production backend fails closed instead of CPU-preparing deformation");
    NativeAnimationDeformationRenderer renderer (backend);
    RenderedDeformationFrame output;
    std::string error;
    check (! renderer.renderPreview (fixture.source, fixture.snapshot, 64, 64,
                                     videorender::animation3d::kNativeGpuCapability,
                                     output, error)
           && ! output.nativeFrame && ! error.empty(),
           "renderer exposes backend unavailability without CPU render fallback");
}
} // namespace

int main()
{
    static_assert (std::is_same_v<decltype (
        std::declval<const arbitgpu::NativeDeformationScene&>().scene),
        std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>>,
        "published source scene storage must have immutable pointee ownership");
    static_assert (std::is_same_v<decltype (
        std::declval<const arbitgpu::NativeDeformationScene&>().deformation),
        std::shared_ptr<const DeformationAsset>>,
        "published skin and morph storage must have immutable pointee ownership");

    testFramePreparation();
    testSharedPreviewExportRoute();
    testUnavailableBackendHasNoFallback();

    if (failures != 0)
        std::fprintf (stderr, "%d native animation deformation test(s) failed\n", failures);
    return failures == 0 ? 0 : 1;
}
