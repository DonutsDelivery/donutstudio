#include "VisualAnimationDeformationEvaluation.h"
#include "gpu_backend/backend.h"
#include "native_animation_deformation_renderer.h"
#include "support/fixture_scene.h"
#include "gltf_glb.h"
#include "support/native_deformation_composition_fixture.h"

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
        sceneValue.textureTexels.clear();
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
    check(output.compositionPost.control()==std::array<float,4>{1,0,0,0}
        && output.compositionPost.point()==identityMatrix() && output.compositionPost.normal()==identityMatrix(),
        "disabled legacy preparation retains the canonical identity post block");
    check (output.vertexCount == 24 && output.jointCount == 1
           && output.morphTargetCount == 1,
           "prepared counts exactly match admitted immutable input");
    check (near (output.jointPalette[12], 1.0f)
           && near (output.jointPalette[13], 0.0f)
           && near (output.jointPalette[14], 0.0f),
           "sampled joint transform becomes the native joint palette");
    check (near (output.morphWeights[0], 0.25f),
           "sampled morph weight replaces the imported base weight");

    auto matrixSource = *fixture.source;
    auto matrix = identityMatrix();
    matrix[13] = 0.75f;
    matrixSource.jointBaseTransforms[0].matrix = matrix;
    arbitgpu::NativeDeformationFrameData matrixOutput;
    check(arbitgpu::prepareNativeDeformationFrame(matrixSource, *fixture.snapshot, matrixOutput, error)
          && near(matrixOutput.jointPalette[12], 1.0f)
          && near(matrixOutput.jointPalette[13], 0.75f),
          "retained matrix ancestors compose with sampled joint transforms");

    auto batchSource = *fixture.source;
    auto batchMember = std::make_shared<arbitgpu::NativeDeformationScene>(*fixture.source);
    batchMember->batchMember = true;
    batchSource.draws = {batchMember};
    check(arbitgpu::prepareNativeDeformationFrame(batchSource, *fixture.snapshot, matrixOutput, error),
          "scene batches admit a topology-stable owned mesh range");
    batchSource.draws.push_back(batchMember);
    check(!arbitgpu::prepareNativeDeformationFrame(batchSource, *fixture.snapshot, matrixOutput, error)
          && error == "native deformation scene batch overlaps mesh topology",
          "scene batches reject overlapping compute output ranges before submission");

    AnimationDeformationRequest poseRequest;
    poseRequest.time = { 15, 30, 1 };
    poseRequest.revision = 77;
    poseRequest.pose.meshStableId = 501;
    poseRequest.pose.boneStableId = 301;
    poseRequest.pose.boneEnabled = true;
    poseRequest.pose.translation = { 0.5, 0.25, 0.0 };
    poseRequest.pose.rotationDegrees[2] = 90.0;
    poseRequest.pose.scale[0] = 2.0;
    poseRequest.pose.morphEnabled = true;
    poseRequest.pose.morphTargetStableId = 1;
    poseRequest.pose.morphWeight = 0.8;
    const auto poseSnapshot = evaluateAnimationDeformation(
        *fixture.asset, *fixture.clip, fixture.bindings, poseRequest, {}, error);
    arbitgpu::NativeDeformationFrameData poseOutput;
    check(poseSnapshot && arbitgpu::prepareNativeDeformationFrame(
              *fixture.source, *poseSnapshot, poseOutput, error)
          && near(poseOutput.jointPalette[12], 1.5f)
          && near(poseOutput.jointPalette[13], 0.25f)
          && near(poseOutput.jointPalette[0], 0.0f)
          && near(poseOutput.jointPalette[1], 2.0f)
          && near(poseOutput.jointPalette[4], -1.0f)
          && near(poseOutput.morphWeights[0], 0.8f),
          "selected bone TRS adjusts the baked pose and only the selected morph weight is overridden");
    for (const auto missingBone : {false, true})
    {
        auto invalidPose = poseRequest;
        if (missingBone) invalidPose.pose.boneStableId = 302;
        else invalidPose.pose.morphTargetIndex = 1;
        const auto invalidSnapshot = evaluateAnimationDeformation(
            *fixture.asset, *fixture.clip, fixture.bindings, invalidPose, {}, error);
        poseOutput.vertexCount = 999;
        check(invalidSnapshot && !arbitgpu::prepareNativeDeformationFrame(
                  *fixture.source, *invalidSnapshot, poseOutput, error)
              && poseOutput.vertexCount == 999,
              "invalid selected bone or morph identity leaves native frame output unchanged");
    }
    {
        std::array<MorphTargetView, 2> targets {fixture.morphTarget, fixture.morphTarget};
        targets[1].id = {602};
        auto mesh = fixture.meshView;
        mesh.morphTargets = targets.data();
        mesh.morphTargetCount = targets.size();
        const auto asset = DeformationAsset::create({&fixture.skinView, 1, &mesh, 1}, {}, error);
        const std::array<float, 4> values {0.0f, 0.5f, 0.5f, 0.75f};
        auto tracks = fixture.tracks;
        tracks[1].values = values.data();
        tracks[1].valueCount = values.size();
        tracks[1].morphWeightCount = 2;
        const auto clip = animation::Clip::create({fixture.clip->id(), 1.0, tracks.data(), tracks.size()}, {}, error);
        auto source = *fixture.source;
        source.deformation = asset;
        source.morphBaseWeights = {0.125f, 0.25f};
        const std::array<MorphTargetId, 2> targetIds {{{601}, {602}}};
        const MorphAnimationBindingView morph {
            fixture.morphBindings[0].animationTarget, mesh.id, targetIds.data(), targetIds.size()};
        const AnimationDeformationBindingView bindings {fixture.jointBindings.data(), 1, &morph, 1};
        auto request = poseRequest;
        request.combinationWeight = 0.25;
        request.pose.morphTargetStableId = 2;
        request.pose.morphTargetIndex = 1;
        const auto snapshot = asset && clip
            ? evaluateAnimationDeformation(*asset, *clip, bindings, request, {}, error) : nullptr;
        check(snapshot && arbitgpu::prepareNativeDeformationFrame(source, *snapshot, poseOutput, error)
              && near(poseOutput.morphWeights[0], 0.0625f)
              && near(poseOutput.morphWeights[1], 0.8f),
              "independent named morph override preserves the other target's animated playback gain");
    }

    // Per-key morph weight deltas add to the evaluated weight of each key under the selected
    // morph gate. The sum clamps to [-16, 16], zero deltas change nothing, and a key the mesh
    // lacks is refused with the output left unchanged.
    {
        std::array<MorphTargetView, 2> targets {fixture.morphTarget, fixture.morphTarget};
        targets[1].id = {602};
        auto mesh = fixture.meshView;
        mesh.morphTargets = targets.data();
        mesh.morphTargetCount = targets.size();
        const auto asset = DeformationAsset::create({&fixture.skinView, 1, &mesh, 1}, {}, error);
        const std::array<float, 4> values {0.0f, 0.5f, 0.5f, 0.75f};
        auto tracks = fixture.tracks;
        tracks[1].values = values.data();
        tracks[1].valueCount = values.size();
        tracks[1].morphWeightCount = 2;
        const auto clip = animation::Clip::create({fixture.clip->id(), 1.0, tracks.data(), tracks.size()}, {}, error);
        auto source = *fixture.source;
        source.deformation = asset;
        source.morphBaseWeights = {0.125f, 0.25f};
        const std::array<MorphTargetId, 2> targetIds {{{601}, {602}}};
        const MorphAnimationBindingView morph {
            fixture.morphBindings[0].animationTarget, mesh.id, targetIds.data(), targetIds.size()};
        const AnimationDeformationBindingView bindings {fixture.jointBindings.data(), 1, &morph, 1};
        auto request = poseRequest;
        request.combinationWeight = 0.25;
        request.pose.morphTargetStableId = 2;
        request.pose.morphTargetIndex = 1;
        const auto prepare = [&] (const AnimationDeformationRequest& candidate,
                                  arbitgpu::NativeDeformationFrameData& output)
        {
            const auto snapshot = asset && clip
                ? evaluateAnimationDeformation(*asset, *clip, bindings, candidate, {}, error) : nullptr;
            return snapshot && arbitgpu::prepareNativeDeformationFrame(source, *snapshot, output, error);
        };

        arbitgpu::NativeDeformationFrameData baseline;
        check(prepare(request, baseline)
              && near(baseline.morphWeights[0], 0.0625f) && near(baseline.morphWeights[1], 0.8f),
              "morph weight delta baseline keeps the gained sampled key and the selected key");
        auto keyed = request;
        keyed.pose.morphWeightDeltas = { { 0u, 0.5 }, { 1u, -0.3 } };
        arbitgpu::NativeDeformationFrameData keyedOutput;
        check(prepare(keyed, keyedOutput)
              && near(keyedOutput.morphWeights[0], 0.5625f) && near(keyedOutput.morphWeights[1], 0.5f),
              "morph weight deltas add to the evaluated weights of two keys, the selected one included");
        const auto keyedSnapshot = asset && clip
            ? evaluateAnimationDeformation(*asset, *clip, bindings, keyed, {}, error) : nullptr;
        check(keyedSnapshot && keyedSnapshot->pose().morphWeightDeltas == keyed.pose.morphWeightDeltas,
              "the evaluated snapshot carries the morph weight deltas it was requested with");

        auto capped = request;
        capped.pose.morphWeightDeltas = { { 0u, 16.0 } };
        arbitgpu::NativeDeformationFrameData cappedOutput;
        check(prepare(capped, cappedOutput) && near(cappedOutput.morphWeights[0], 16.0f)
              && near(cappedOutput.morphWeights[1], 0.8f),
              "a morph weight delta clamps the sum to the upper bound of 16");
        auto floored = request;
        floored.pose.morphWeight = -10.0;
        floored.pose.morphWeightDeltas = { { 1u, -10.0 } };
        arbitgpu::NativeDeformationFrameData flooredOutput;
        check(prepare(floored, flooredOutput) && near(flooredOutput.morphWeights[1], -16.0f),
              "a morph weight delta clamps the sum to the lower bound of -16");

        auto zeroed = request;
        zeroed.pose.morphWeightDeltas = { { 0u, 0.0 }, { 1u, 0.0 } };
        arbitgpu::NativeDeformationFrameData zeroedOutput;
        check(prepare(zeroed, zeroedOutput)
              && near(zeroedOutput.morphWeights[0], baseline.morphWeights[0])
              && near(zeroedOutput.morphWeights[1], baseline.morphWeights[1]),
              "zero morph weight deltas leave every key at its baseline weight");

        auto ungated = request;
        ungated.pose.morphEnabled = false;
        arbitgpu::NativeDeformationFrameData ungatedBaseline;
        check(prepare(ungated, ungatedBaseline), "morph-disabled baseline prepares a native frame");
        ungated.pose.morphWeightDeltas = { { 0u, 0.5 }, { 1u, -0.3 } };
        arbitgpu::NativeDeformationFrameData ungatedOutput;
        check(prepare(ungated, ungatedOutput)
              && near(ungatedOutput.morphWeights[0], ungatedBaseline.morphWeights[0])
              && near(ungatedOutput.morphWeights[1], ungatedBaseline.morphWeights[1]),
              "morph weight deltas are ignored while the selected morph gate is closed");

        auto missing = request;
        missing.pose.morphWeightDeltas = { { 2u, 0.5 } };
        arbitgpu::NativeDeformationFrameData missingOutput;
        missingOutput.vertexCount = 999;
        error.clear();
        check(!prepare(missing, missingOutput) && error.find("out of range") != std::string::npos
              && missingOutput.vertexCount == 999,
              "a morph weight delta naming a key the mesh lacks is refused and leaves output unchanged");
    }

    // Runtime bone TRS reaches the native palette only when the bone is enabled.
    {
        AnimationDeformationRequest baseline = poseRequest;
        baseline.pose.boneEnabled = false;
        baseline.pose.translation = {};
        baseline.pose.rotationDegrees = {};
        baseline.pose.scale = { 1.0, 1.0, 1.0 };
        const auto palette = [&] (const AnimationDeformationRequest& request,
                                  arbitgpu::NativeDeformationFrameData& output)
        {
            const auto snapshot = evaluateAnimationDeformation(
                *fixture.asset, *fixture.clip, fixture.bindings, request, {}, error);
            return snapshot && arbitgpu::prepareNativeDeformationFrame(
                *fixture.source, *snapshot, output, error);
        };
        arbitgpu::NativeDeformationFrameData baselineOutput;
        check(palette(baseline, baselineOutput),
              "selected bone gating baseline prepares a native frame");
        const auto gateAxis = [&] (auto&& modify, const char* offMessage, const char* onMessage)
        {
            AnimationDeformationRequest off = baseline;
            modify(off);
            AnimationDeformationRequest on = off;
            on.pose.boneEnabled = true;
            arbitgpu::NativeDeformationFrameData offOutput;
            arbitgpu::NativeDeformationFrameData onOutput;
            const bool prepared = palette(off, offOutput) && palette(on, onOutput);
            check(prepared, "selected bone gating axis snapshots prepare native frames");
            check(prepared && offOutput.jointPalette == baselineOutput.jointPalette
                  && offOutput.morphWeights == baselineOutput.morphWeights, offMessage);
            check(prepared && onOutput.jointPalette != baselineOutput.jointPalette, onMessage);
        };
        gateAxis([] (AnimationDeformationRequest& request) { request.pose.rotationDegrees[1] = 30.0; },
                 "selectedBoneEnabled=0 leaves native output equal to the baseline for boneRotateY",
                 "selectedBoneEnabled=1 changes native output for boneRotateY");
        gateAxis([] (AnimationDeformationRequest& request) { request.pose.translation[0] = 2.5; },
                 "selectedBoneEnabled=0 leaves native output equal to the baseline for boneTranslateX",
                 "selectedBoneEnabled=1 changes native output for boneTranslateX");
        gateAxis([] (AnimationDeformationRequest& request) { request.pose.scale[2] = 1.5; },
                 "selectedBoneEnabled=0 leaves native output equal to the baseline for boneScaleZ",
                 "selectedBoneEnabled=1 changes native output for boneScaleZ");
    }

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

using CompositionFixture = videohelper::fixture3d::NativeCompositionFixture;

void testCompositionCapabilities()
{
    {
        CompositionFixture fixture(true);
        auto original=std::make_shared<arbitgpu::NativeDeformationScene>(*fixture.source);
        original->structuralRevision=31;original->evaluationRevision=35;
        fixture.source=original;fixture.owner=fixture.admit(fixture.composed,fixture.mappings);
        const auto snapshot=fixture.snapshot(15);
        const auto frame=arbitgpu::NativeDeformationCompositionFixture::frame(fixture.owner,snapshot,fixture.error);
        auto batch=fixture.batch();
        arbitgpu::NativeDeformationFrameData output;
        check(fixture.owner && fixture.owner->revision()==31 && fixture.owner->evaluationRevision()==35
            && frame && frame->evaluationRevision()==35 && snapshot->revision()==31,
            "literal structural 31 and parameter epoch 35 remain separate from rational time");
        check(frame && arbitgpu::prepareNativeDeformationFrame(*batch,*snapshot,frame,output,fixture.error),
            "typed composition accepts the actual source epoch");
        auto stale=std::make_shared<arbitgpu::NativeDeformationScene>(*batch);stale->evaluationRevision=34;
        const auto incumbent=output;
        check(frame && !arbitgpu::prepareNativeDeformationFrame(*stale,*snapshot,frame,output,fixture.error)
            && output.jointPalette==incumbent.jointPalette && output.compositionPost.point()==incumbent.compositionPost.point(),
            "source epoch substitution refuses without changing prepared output");
        original=std::make_shared<arbitgpu::NativeDeformationScene>(*fixture.source);original->evaluationRevision=30;
        fixture.source=original;
        check(!fixture.admit(fixture.composed,fixture.mappings),"composition refuses parameter epoch below structural revision");
        original->evaluationRevision=0;
        check(!fixture.admit(fixture.composed,fixture.mappings),"composition refuses a fabricated default parameter epoch");
        arbitgpu::NativeSceneMotionOwner oldOwner {{801,802,1,31,11,12,42,0},35};
        auto newOwner=oldOwner;newOwner.evaluationRevision=36;
        check(!(oldOwner==newOwner),"Motion ownership includes the distinct parameter epoch");
    }
    for (bool importedParent:{false,true})
    {
        CompositionFixture fixture(importedParent);
        check(bool(fixture.owner),"typed composition admits both root-parent directions with original assets");
        if (!fixture.owner) continue;
        const auto current=fixture.snapshot(15),previous=fixture.snapshot(14);
        const auto frame=arbitgpu::NativeDeformationCompositionFixture::frame(fixture.owner,current,fixture.error);
        const auto prior=arbitgpu::NativeDeformationCompositionFixture::frame(fixture.owner,previous,fixture.error);
        const auto batch=fixture.batch();
        arbitgpu::NativeDeformationFrameData value;
        arbitgpu::NativeSceneMotionSample::ImportedNodeWorldMatrices worlds;
        check(frame && prior && arbitgpu::prepareNativeDeformationFrame(*batch,*current,frame,value,fixture.error,&worlds),
            "all disjoint members share the full admitted scene and raw imported Motion worlds");
        check(worlds.size()==1 && worlds[0].nodeStableId==1 && near(worlds[0].world[12],1),
            "composition does not insert procedural worlds in the GLB Motion table");
        check(!arbitgpu::prepareNativeDeformationFrame(*batch,*current,value,fixture.error),
            "legacy entry refuses a composed owner without its exact typed frame");
        for (const auto& draw:batch->draws)
        {
            check(arbitgpu::prepareNativeDeformationFrame(*draw,*current,frame,value,fixture.error),
                "typed member prepares original-coordinate palette before its post transform");
            if (draw->animationNodeStableId==1)
                check(near(value.jointPalette[12],1) && near(value.compositionPost.point()[12],0),
                    "skin-backed own-node animation is applied once under an external parent");
            else
            {
                const auto currentPoint=value.compositionPost.point();
                check(arbitgpu::prepareNativeDeformationFrame(*draw,*previous,prior,value,fixture.error)
                    && (importedParent ? currentPoint!=value.compositionPost.point():currentPoint==value.compositionPost.point()),
                    "preceding procedural descendant frame retains its own exact sampled parent transform");
            }
            const auto unchanged=value;
            check(!arbitgpu::prepareNativeDeformationFrame(*draw,*previous,frame,value,fixture.error)
                && value.jointPalette==unchanged.jointPalette && value.compositionPost.point()==unchanged.compositionPost.point(),
                "typed snapshot time and exact pointer mismatch leaves frame output unchanged");
        }
        check(!arbitgpu::NativeDeformationCompositionFixture::frame(fixture.owner,current,fixture.error,13,12)
            && !arbitgpu::NativeDeformationCompositionFixture::frame(fixture.owner,current,fixture.error,11,13),
            "composition frames reject stale project and helper generations");
        auto foreign=fixture.admit(fixture.composed,fixture.mappings);
        const auto foreignFrame=arbitgpu::NativeDeformationCompositionFixture::frame(foreign,current,fixture.error);
        check(!arbitgpu::prepareNativeDeformationFrame(*batch,*current,foreignFrame,value,fixture.error),
            "an equivalent foreign owner does not replace exact admitted ownership");
        auto stale=std::make_shared<arbitgpu::NativeDeformationScene>(*batch->draws[0]);stale->structuralRevision++;
        check(!arbitgpu::prepareNativeDeformationFrame(*stale,*current,frame,value,fixture.error),
            "a stale native member revision does not reuse a current typed frame");
        arbitgpu::NativeDeformationRuntimeInputs inputs; inputs.compositionFrame=frame; inputs.objectNodeStableId=1;
        check(arbitgpu::runtimeTargetsObject(inputs,fixture.composed->objects[0])
            && arbitgpu::runtimeTargetsObject(inputs,fixture.composed->objects[1])==importedParent,
            "selected imported runtime controls resolve descendants through typed membership");
        arbitgpu::NativeSceneMotionHistory history;
        inputs.compositionFrame=prior;
        check(history.commit(14,30,1,64,64,inputs,previous),"composition Motion commits its exact preceding sample");
        inputs.compositionFrame=frame;
        check(bool(history.predecessor(15,30,1,64,64,{},inputs))
            && !history.predecessor(17,30,1,64,64,{},inputs),"composition Motion preserves rational adjacency resets");
        const auto saved=history.current;
        check(!history.commit(15,30,1,64,64,inputs,previous) && history.current->compositionFrame==saved->compositionFrame,
            "refused typed history commits retain the prior published sample");
        for (int failure=0; failure<12; ++failure)
        {
            auto scene=std::make_shared<HarmonicMIDI::grid::Visual3DScene>(*fixture.composed);
            auto mappings=fixture.mappings;
            if (failure==0) scene->objects[1].firstVertex=0;
            if (failure==1) scene->objects[1].firstIndex=0;
            if (failure==2) {scene->objects[0].parent={900};scene->objects[1].parent={1};}
            if (failure==3) scene->objects[0].transform.scale.x=0;
            if (failure==4) scene->objects[0].transform.translation.x=std::numeric_limits<float>::infinity();
            if (failure==5) scene->vertices[0].position.x+=1;
            if (failure==6) mappings[1].importedNodeStableId=2;
            if (failure==7) scene->objectCount=65;
            if (failure==8) scene->vertexCount=4097;
            if (failure==9) scene->indexCount=12289;
            if (failure==10) mappings[1].composedObject={1};
            if (failure==11) scene->materials[0].roughness=.8f;
            check(!fixture.admit(scene,mappings),"owner admission refuses overlap, cycle, singular, nonfinite, rest edits, identity and caps");
        }
    }
    for (bool parent:{false,true})
    {
        CompositionFixture morph(parent,false,true);
        const auto snapshot=morph.snapshot(15);
        const auto frame=arbitgpu::NativeDeformationCompositionFixture::frame(morph.owner,snapshot,morph.error);
        const auto batch=morph.batch();
        arbitgpu::NativeDeformationFrameData output;
        check(frame && arbitgpu::prepareNativeDeformationFrame(*batch->draws[0],*snapshot,frame,output,morph.error)
            && output.jointCount==0 && output.morphTargetCount==1 && near(output.compositionPost.point()[12],0),
            "unskinned morph uses only a proven static original node world under either root parent direction");
    }
    CompositionFixture missing(true,true,true);
    const auto missingSnapshot=missing.snapshot(30);
    check(missing.owner && missingSnapshot
        && !arbitgpu::NativeDeformationCompositionFixture::frame(missing.owner,missingSnapshot,missing.error),
        "animated unskinned ancestor without canonical sampled world coverage refuses rest fallback");
    if (missingSnapshot)
    {
        animation::SampleRequest request;request.timeSeconds=missingSnapshot->requestedTimeSeconds();
        const auto sampled=animation::sample(*missing.clip,request,missing.error);
        const auto sceneSample=sampled?std::make_shared<const animation::Sample>(*sampled):nullptr;
        const auto frame=arbitgpu::NativeDeformationCompositionFixture::frame(missing.owner,missingSnapshot,missing.error,11,12,sceneSample);
        const auto batch=missing.batch();arbitgpu::NativeDeformationFrameData output;
        check(frame && arbitgpu::prepareNativeDeformationFrame(*batch->draws[0],*missingSnapshot,frame,output,missing.error)
            && output.jointCount==0 && near(output.compositionPost.point()[0],-2),
            "authenticated scene channels supply an unskinned node world without a forged matrix table");
    }
    CompositionFixture posed(true);
    for (bool weighted:{false,true})
    {
        const auto snapshot=posed.snapshot(15,true,weighted);
        const auto frame=arbitgpu::NativeDeformationCompositionFixture::frame(posed.owner,snapshot,posed.error);
        const auto batch=posed.batch();
        arbitgpu::NativeDeformationFrameData output;
        check(frame && arbitgpu::prepareNativeDeformationFrame(*batch->draws[1],*snapshot,frame,output,posed.error)
            && near(output.compositionPost.point()[12],weighted?1.0f:1.5f),
            "procedural descendants retain canonical combination weight and selected bone pose in their imported parent world");
    }
    CompositionFixture scaled(true,true);
    const auto snapshot=scaled.snapshot(30);
    const auto frame=arbitgpu::NativeDeformationCompositionFixture::frame(scaled.owner,snapshot,scaled.error);
    const auto batch=scaled.batch();
    arbitgpu::NativeDeformationFrameData value;
    check(frame && arbitgpu::prepareNativeDeformationFrame(*batch->draws[1],*snapshot,frame,value,scaled.error)
        && near(value.compositionPost.point()[0],-2) && near(value.compositionPost.point()[5],3)
        && near(value.compositionPost.normal()[0],-.5f) && near(value.compositionPost.normal()[5],1.0f/3.0f)
        && value.compositionPost.normal()[15]==-1,
        "nonuniform reflected descendant post retains inverse-transpose normals and tangent handedness");
    for (bool parent:{false,true})
    for (const auto kind:{videohelper::fixture3d::CompositionSkinCase::IndependentRotation,
        videohelper::fixture3d::CompositionSkinCase::IndependentReflection})
    {
        CompositionFixture independent(parent,false,false,kind);
        check(independent.owner && independent.document && independent.animationDocument
            && independent.payload->bytes().size()>4
            && independent.payload->key().contentSha256==videohelper::sha256Bytes(
                independent.payload->bytes().data(),independent.payload->bytes().size())
            && independent.animationDocument->deformation==independent.source->deformation,
            "composition fixture retains exact original admitted GLB bytes and decode-backed documents");
        if (!independent.owner) continue;
        const auto sample=independent.snapshot(30);
        const auto current=arbitgpu::NativeDeformationCompositionFixture::frame(independent.owner,sample,independent.error);
        const auto draws=independent.batch();arbitgpu::NativeDeformationFrameData data;
        const bool ready=current && sample && arbitgpu::prepareNativeDeformationFrame(
            *draws->draws[0],*sample,current,data,independent.error);
        check(ready && data.compositionPost.point()==identityMatrix(),
            "independent skin keeps identity composition post instead of repeating its joint animation");
        if (!ready) continue;
        const auto reference=videohelper::fixture3d::compositionReference(independent,*draws,*sample,current,independent.error);
        check(bool(reference),"literal rotated/reflected skin reference prepares both parent directions");
        if (!reference) continue;
        const auto& v=reference->vertices[0];
        if (kind==videohelper::fixture3d::CompositionSkinCase::IndependentRotation)
            check(near(data.jointPalette[0],0) && near(data.jointPalette[1],1)
                && near(v.position.x,1) && near(v.position.y,-1) && near(v.normal.z,1)
                && near(v.tangent[0],0) && near(v.tangent[1],1) && v.tangent[3]==1,
                "literal ninety-degree skin rotation carries the normal-map tangent into +Y");
        else
            check(near(data.jointPalette[0],-1) && near(v.position.x,1) && near(v.position.y,-1)
                && near(v.normal.z,1) && near(v.tangent[0],-1) && v.tangent[3]==-1,
                "literal skin reflection flips tangent X and handedness with identity post");
    }
    for (bool parent:{false,true})
    {
        CompositionFixture nonuniform(parent,false,false,videohelper::fixture3d::CompositionSkinCase::IndependentNonuniform);
        const auto sample=nonuniform.snapshot(30);
        const auto frame=arbitgpu::NativeDeformationCompositionFixture::frame(nonuniform.owner,sample,nonuniform.error);
        check(nonuniform.owner && sample && frame,"original slanted-face GLB admits a nonuniform independent skin");
        if (!frame || !sample) continue;
        const auto draws=nonuniform.batch();arbitgpu::NativeDeformationFrameData data;
        const auto reference=videohelper::fixture3d::compositionReference(nonuniform,*draws,*sample,frame,nonuniform.error);
        const float inverseSqrt5=1.0f/std::sqrt(5.0f);
        check(reference && arbitgpu::prepareNativeDeformationFrame(*draws->draws[0],*sample,frame,data,nonuniform.error)
            && near(data.jointPalette[0],2) && near(data.jointPalette[5],1)
            && data.compositionPost.point()==identityMatrix(),
            "nonuniform skin deformation is applied before identity composition post");
        if (!reference) continue;
        const auto& vertex=reference->vertices[12];
        check(near(vertex.normal.x,inverseSqrt5) && near(vertex.normal.y,2*inverseSqrt5)
            && near(vertex.tangent[0],2*inverseSqrt5) && near(vertex.tangent[1],-inverseSqrt5)
            && near(vertex.normal.x*vertex.tangent[0]+vertex.normal.y*vertex.tangent[1],0)
            && vertex.tangent[3]==1,
            "literal diag(2,1,1) skin keeps the slanted normal and normal-map tangent perpendicular");
    }
    for (const auto kind:{videohelper::fixture3d::CompositionSkinCase::SingularBlend,
        videohelper::fixture3d::CompositionSkinCase::NonfiniteInverse})
    {
        CompositionFixture singular(true,false,false,kind);
        const auto singularSample=singular.snapshot(30);
        const auto singularFrame=arbitgpu::NativeDeformationCompositionFixture::frame(singular.owner,singularSample,singular.error);
        check(singular.owner && singularFrame && singularSample,
            "undefined-skin fixtures retain two individually invertible authenticated joints");
        if (singularFrame && singularSample)
        {
            const auto draws=singular.batch();arbitgpu::NativeDeformationFrameData data;
            const auto saved=data;
            check(!arbitgpu::prepareNativeDeformationFrame(*draws,*singularSample,singularFrame,data,singular.error)
                && data.jointPalette==saved.jointPalette && data.compositionPost.point()==saved.compositionPost.point(),
                "typed admission refuses singular or nonfinite weighted skin inverse before publishing frame output");
            check(arbitgpu::prepareNativeDeformationFrame(*singular.source,*singularSample,data,singular.error)
                && data.compositionPost.control()==std::array<float,4>{1,0,0,0},
                "undefined blended skin retains the disabled legacy entry's behavior");
        }
    }
    check(!std::is_default_constructible_v<arbitgpu::NativeDeformationCompositionOwner>
        && !std::is_default_constructible_v<arbitgpu::NativeDeformationCompositionFrame>
        && !std::is_copy_constructible_v<arbitgpu::NativeDeformationCompositionOwner>
        && !std::is_copy_constructible_v<arbitgpu::NativeDeformationCompositionFrame>,
        "production composition capabilities have no public constructor");
}

using CompositionMapping = arbitgpu::NativeDeformationCompositionMapping;
using CompositionScene = HarmonicMIDI::grid::Visual3DScene;

arbitgpu::NativeDeformationCompositionLeaf animatedCompositionLeaf(const CompositionFixture& fixture,
    std::shared_ptr<const arbitgpu::NativeDeformationScene> source = {},
    std::shared_ptr<const videohelper::gltf::GlbStaticMeshDocument> document = {})
{
    arbitgpu::NativeDeformationCompositionLeaf leaf;
    leaf.payload=fixture.payload; leaf.document=document?document:fixture.document;
    leaf.animation=fixture.animationDocument; leaf.selectedScene=0; leaf.selectedMesh=1;
    leaf.imported=source?source:fixture.source;
    if (leaf.imported) leaf.scene=leaf.imported->scene;
    return leaf;
}

arbitgpu::NativeDeformationCompositionLeaf staticCompositionLeaf(const CompositionFixture& fixture)
{
    arbitgpu::NativeDeformationCompositionLeaf leaf;
    leaf.payload=fixture.payload; leaf.document=fixture.document; leaf.selectedScene=0;
    if (fixture.source) leaf.scene=fixture.source->scene;
    return leaf;
}

arbitgpu::NativeDeformationCompositionLeafSample compositionLeafSample(
    std::shared_ptr<const AnimationDeformationSnapshot> snapshot)
{
    arbitgpu::NativeDeformationCompositionLeafSample sample;
    sample.deformation=std::move(snapshot);
    return sample;
}

// Appends object zero of a leaf rest scene as a new composed object. The cube
// material matches composed material zero by content.
void appendCompositionLeafObject(CompositionScene& composed,const CompositionScene& leaf,
    HarmonicMIDI::grid::SceneObjectId id,HarmonicMIDI::grid::SceneObjectId parent)
{
    auto object=leaf.objects[0];
    const auto firstVertex=object.firstVertex, firstIndex=object.firstIndex;
    object.id=id; object.parent=parent; object.material=composed.materials[0].id;
    object.firstVertex=static_cast<std::uint32_t>(composed.vertexCount);
    object.firstIndex=static_cast<std::uint32_t>(composed.indexCount);
    for (std::uint32_t v=0; v<object.vertexCount; ++v) composed.vertices[composed.vertexCount+v]=leaf.vertices[firstVertex+v];
    for (std::uint32_t j=0; j<object.indexCount; ++j) composed.indices[composed.indexCount+j]=leaf.indices[firstIndex+j];
    composed.vertexCount+=object.vertexCount; composed.indexCount+=object.indexCount;
    composed.objects[composed.objectCount++]=object;
}

std::shared_ptr<arbitgpu::NativeDeformationScene> compositionLeafBatch(const CompositionFixture& identity,
    const std::shared_ptr<const arbitgpu::NativeDeformationCompositionOwner>& owner)
{
    auto result=std::make_shared<arbitgpu::NativeDeformationScene>(*identity.source);
    if (!owner) return result;
    result->scene=owner->scene(); result->compositionOwner=owner; result->batchMember=false; result->draws.clear();
    for (std::size_t i=0; i<owner->memberCount(); ++i)
    {
        auto member=std::make_shared<arbitgpu::NativeDeformationScene>(*owner->originalMember(i));
        member->object=owner->memberObject(i); member->scene=owner->scene(); member->compositionOwner=owner;
        member->batchMember=true; result->draws.push_back(std::move(member));
    }
    return result;
}

std::size_t compositionMemberIndex(const arbitgpu::NativeDeformationCompositionOwner& owner,std::uint32_t object)
{
    for (std::size_t i=0; i<owner.memberCount(); ++i) if (owner.memberObject(i).value==object) return i;
    return owner.memberCount();
}

std::shared_ptr<const arbitgpu::NativeDeformationScene> compositionMemberDraw(const arbitgpu::NativeDeformationScene& batch,
    std::uint32_t object)
{
    for (const auto& draw:batch.draws) if (draw->object.value==object) return draw;
    return {};
}

void testMultiLeafComposition()
{
    using Fixture=arbitgpu::NativeDeformationCompositionFixture;
    const visualdeformation::RationalFrameTime frameTime {15,30,1};
    // Leaf 0 animates node translation (x = 2t); leaf 1 animates node scale
    // from (1,1,1) to (-2,3,4). Leaf 1 carries its own source identities.
    CompositionFixture translated(true), scaled(true,true);
    check(translated.owner && scaled.owner,"multi-leaf fixtures admit their single-leaf owners");
    if (!translated.owner || !scaled.owner) return;
    auto scaledSource=std::make_shared<arbitgpu::NativeDeformationScene>(*scaled.source);
    scaledSource->sourceStableId=803; scaledSource->deformationStableId=804;
    auto twoLeafScene=std::make_shared<CompositionScene>(*translated.composed);
    appendCompositionLeafObject(*twoLeafScene,*scaled.source->scene,{2001},{});
    const std::vector<CompositionMapping> twoLeafMappings {
        {CompositionMapping::Kind::Imported,{1},{1},1},
        {CompositionMapping::Kind::Procedural,{65537},{900},0},
        {CompositionMapping::Kind::Imported,{1},{2001},2049}};
    const auto twoLeafOwner=[&](std::shared_ptr<const CompositionScene> scene,const std::vector<CompositionMapping>& mappings,
        std::shared_ptr<const arbitgpu::NativeDeformationScene> second) {
        std::string error;
        return Fixture::leafOwner({animatedCompositionLeaf(translated),animatedCompositionLeaf(scaled,std::move(second))},
            translated.procedural,std::move(scene),mappings,error);
    };
    const auto owner=twoLeafOwner(twoLeafScene,twoLeafMappings,scaledSource);
    check(owner && owner->leafCount()==2 && owner->memberCount()==3 && owner->importedNodeCount()==2
        && !owner->staticLeaf(0) && !owner->staticLeaf(1),
        "two animated leaves admit one owner with their own payloads, documents and native draws");
    if (!owner) return;
    const auto translatedIndex=compositionMemberIndex(*owner,1), scaledIndex=compositionMemberIndex(*owner,2001),
        proceduralIndex=compositionMemberIndex(*owner,900);
    check(translatedIndex<3 && scaledIndex<3 && proceduralIndex<3
        && owner->memberLeaf(translatedIndex)==0 && owner->memberLeaf(scaledIndex)==1 && owner->memberLeaf(proceduralIndex)==0
        && !owner->memberRigid(translatedIndex) && !owner->memberRigid(scaledIndex) && owner->memberRigid(proceduralIndex)
        && owner->originalMember(scaledIndex)->sourceStableId==803,
        "multi-leaf members keep their leaf, rigid kind and leaf source identity");
    check(owner->targetsObject(2049,{2001}) && !owner->targetsObject(1,{2001})
        && owner->targetsObject(1,{1}) && !owner->targetsObject(2049,{1})
        && owner->targetsObject(1,{900}) && !owner->targetsObject(2049,{900})
        && !owner->targetsObject(2050,{2001}) && owner->targetsObject(0,{2001}),
        "object targeting matches leaf 1 node 1 only through its stride identity");
    check(!owner->targetsObject(4097,{2001}) && !owner->targetsObject(4097,{1}),
        "a query naming a leaf the owner does not have targets no object");
    if (translatedIndex>=3 || scaledIndex>=3 || proceduralIndex>=3) return;
    {
        // One presentation frame; each leaf samples its own clip time.
        const auto first=translated.snapshot(15,false,false,.25), second=scaled.snapshot(15,false,false,1.0);
        std::string error;
        const auto frame=Fixture::leafFrame(owner,{compositionLeafSample(first),compositionLeafSample(second)},frameTime,error);
        check(frame && first && second && &frame->memberSnapshot(translatedIndex)==first.get()
            && &frame->memberSnapshot(scaledIndex)==second.get() && &frame->memberSnapshot(proceduralIndex)==first.get()
            && frame->memberSnapshot(translatedIndex).sampleTimeSeconds()==.25
            && frame->memberSnapshot(scaledIndex).sampleTimeSeconds()==1.0,
            "each animated leaf keeps its own clip and clip time within one presentation frame");
        if (!frame) return;
        const auto batch=compositionLeafBatch(translated,owner);
        arbitgpu::NativeDeformationFrameData value;
        arbitgpu::NativeSceneMotionSample::ImportedNodeWorldMatrices worlds;
        check(arbitgpu::prepareNativeDeformationFrame(*batch,*first,frame,value,error,&worlds)
            && worlds.size()==2 && worlds[0].nodeStableId==1 && near(worlds[0].world[12],.5f)
            && worlds[1].nodeStableId==2049 && near(worlds[1].world[0],-2) && near(worlds[1].world[5],3)
            && near(worlds[1].world[10],4) && near(worlds[1].world[12],0),
            "the multi-leaf world table addresses leaf 1 through the 2048 stride");
        const auto translatedDraw=compositionMemberDraw(*batch,1), scaledDraw=compositionMemberDraw(*batch,2001),
            proceduralDraw=compositionMemberDraw(*batch,900);
        check(translatedDraw && arbitgpu::prepareNativeDeformationFrame(*translatedDraw,*first,frame,value,error)
            && value.jointCount==1 && near(value.jointPalette[12],.5f) && near(value.jointPalette[0],1)
            && value.compositionPost.point()==identityMatrix(),
            "leaf 0 evaluates its translation clip at its own clip time 0.25");
        check(scaledDraw && arbitgpu::prepareNativeDeformationFrame(*scaledDraw,*first,frame,value,error)
            && value.jointCount==1 && near(value.jointPalette[0],-2) && near(value.jointPalette[5],3)
            && near(value.jointPalette[10],4) && near(value.jointPalette[12],0)
            && value.compositionPost.point()==identityMatrix() && value.compositionPost.control()[2]==1,
            "leaf 1 evaluates its scale clip at its own clip time 1.0");
        check(proceduralDraw && arbitgpu::prepareNativeDeformationFrame(*proceduralDraw,*first,frame,value,error)
            && near(value.jointPalette[12],0) && near(value.compositionPost.point()[12],.5f)
            && value.compositionPost.control()[2]==2,
            "a procedural child follows its imported parent leaf at that leaf's clip time");
        const auto unchanged=value;
        check(scaledDraw && !arbitgpu::prepareNativeDeformationFrame(*scaledDraw,*second,frame,value,error)
            && value.jointPalette==unchanged.jointPalette,
            "a non-identity leaf snapshot does not authenticate the frame");
        check(arbitgpu::validNativeCompositionRetention(frame,first) && !arbitgpu::validNativeCompositionRetention(frame,second),
            "frame retention is keyed to the identity leaf snapshot");
        {
            // Both GLBs number their one clip 1, so only the payload digest
            // tells the two samples apart.
            std::string swapError;
            const auto swapped=Fixture::leafFrame(owner,{compositionLeafSample(second),compositionLeafSample(first)},
                frameTime,swapError);
            check(first && second && first->clip()==second->clip()
                && first->sourceContentSha256()!=second->sourceContentSha256()
                && !swapped && swapError=="native composition leaf snapshot is not from its leaf's exact payload",
                ("two leaves sharing a clip ID refuse swapped samples by payload digest: "+swapError).c_str());
            AnimationDeformationRequest unattested;
            unattested.time=frameTime; unattested.revision=scaled.source->structuralRevision;
            unattested.allowNodeTransformAndMorph=true; unattested.requestedTimeSeconds=1.0;
            const AnimationDeformationBindingView view {scaled.bindings.data(),scaled.bindings.size(),
                scaled.morphBindings.data(),scaled.morphBindings.size(),scaled.sceneTargets.data(),scaled.sceneTargets.size()};
            const auto bare=evaluateAnimationDeformation(*scaled.source->deformation,*scaled.clip,view,unattested,{},swapError);
            std::string bareError;
            check(bare && !Fixture::leafFrame(owner,{compositionLeafSample(first),compositionLeafSample(bare)},frameTime,bareError)
                && bareError=="native composition leaf snapshot is not from its leaf's exact payload",
                ("a multi-leaf frame refuses a snapshot without a payload digest: "+bareError).c_str());
        }
        check(!Fixture::leafFrame(owner,{compositionLeafSample(first)},frameTime,error)
            && !Fixture::leafFrame(owner,{compositionLeafSample(first),compositionLeafSample({})},frameTime,error)
            && !Fixture::leafFrame(owner,{compositionLeafSample(first),compositionLeafSample(scaled.snapshot(14,false,false,1.0))},frameTime,error)
            && !Fixture::frame(owner,first,error),
            "a multi-leaf frame needs one same-time sample per leaf through the leaf entry");
    }
    {
        // Weight and pose come from each leaf's own snapshot and apply only
        // to that leaf's members and their procedural descendants.
        for (bool weightedFirst:{true,false})
        {
            const auto first=translated.snapshot(15,!weightedFirst,weightedFirst),
                second=scaled.snapshot(15,weightedFirst,!weightedFirst,1.0);
            std::string error;
            const auto frame=Fixture::leafFrame(owner,{compositionLeafSample(first),compositionLeafSample(second)},frameTime,error);
            const auto batch=compositionLeafBatch(translated,owner);
            const auto translatedDraw=compositionMemberDraw(*batch,1), scaledDraw=compositionMemberDraw(*batch,2001),
                proceduralDraw=compositionMemberDraw(*batch,900);
            arbitgpu::NativeDeformationFrameData a, b, p;
            const bool ready=frame && translatedDraw && scaledDraw && proceduralDraw
                && arbitgpu::prepareNativeDeformationFrame(*translatedDraw,*first,frame,a,error)
                && arbitgpu::prepareNativeDeformationFrame(*scaledDraw,*first,frame,b,error)
                && arbitgpu::prepareNativeDeformationFrame(*proceduralDraw,*first,frame,p,error);
            check(ready,"per-leaf weighted and posed snapshots prepare every member");
            if (!ready) continue;
            if (weightedFirst)
                check(near(a.jointPalette[12],.5f) && near(p.compositionPost.point()[12],.5f)
                    && near(b.jointPalette[0],-2) && near(b.jointPalette[5],3) && near(b.jointPalette[12],.5f),
                    "leaf 0 weight 0.5 halves its translation; leaf 1 keeps full scale and its own bone pose");
            else
                check(near(a.jointPalette[12],1.5f) && near(p.compositionPost.point()[12],1.5f)
                    && near(b.jointPalette[0],-.5f) && near(b.jointPalette[5],2) && near(b.jointPalette[12],0),
                    "leaf 0 bone pose adds 0.5; leaf 1 weight 0.5 blends only its own scale");
        }
    }
    {
        // A static leaf becomes a rigid member, placed under an animated leaf.
        CompositionFixture still(true);
        auto scene=std::make_shared<CompositionScene>(*translated.composed);
        appendCompositionLeafObject(*scene,*still.source->scene,{3001},{1});
        const std::vector<CompositionMapping> mappings {
            {CompositionMapping::Kind::Imported,{1},{1},2049},
            {CompositionMapping::Kind::Procedural,{65537},{900},0},
            {CompositionMapping::Kind::Imported,{1},{3001},1}};
        std::string error;
        const auto mixed=Fixture::leafOwner({staticCompositionLeaf(still),animatedCompositionLeaf(translated)},
            translated.procedural,scene,mappings,error);
        const auto rigidIndex=mixed?compositionMemberIndex(*mixed,3001):0;
        check(mixed && mixed->leafCount()==2 && mixed->memberCount()==3 && mixed->staticLeaf(0) && !mixed->staticLeaf(1)
            && rigidIndex<3 && mixed->memberRigid(rigidIndex) && mixed->memberLeaf(rigidIndex)==0
            && mixed->originalMember(rigidIndex)->scene==still.source->scene
            && mixed->originalMember(rigidIndex)->jointBaseTransforms.size()==1
            && mixed->originalMember(rigidIndex)->clip==translated.source->clip,
            "a static leaf before the identity leaf becomes a rigid member with the identity clip");
        if (mixed && rigidIndex<3)
        {
            const auto current=translated.snapshot(15);
            const auto frame=Fixture::leafFrame(mixed,{arbitgpu::NativeDeformationCompositionLeafSample{},
                compositionLeafSample(current)},frameTime,error);
            const auto batch=compositionLeafBatch(translated,mixed);
            const auto rigidDraw=compositionMemberDraw(*batch,3001), animatedDraw=compositionMemberDraw(*batch,1);
            arbitgpu::NativeDeformationFrameData rigid, animated;
            arbitgpu::NativeSceneMotionSample::ImportedNodeWorldMatrices worlds;
            check(frame && rigidDraw && animatedDraw && &frame->memberSnapshot(rigidIndex)==current.get()
                && arbitgpu::prepareNativeDeformationFrame(*rigidDraw,*current,frame,rigid,error)
                && arbitgpu::prepareNativeDeformationFrame(*animatedDraw,*current,frame,animated,error)
                && arbitgpu::prepareNativeDeformationFrame(*batch,*current,frame,animated,error,&worlds)
                && rigid.jointCount==1 && near(rigid.jointPalette[0],1) && near(rigid.jointPalette[12],0)
                && near(rigid.compositionPost.point()[12],1) && rigid.compositionPost.control()[2]==2
                && worlds.size()==2 && worlds[0].nodeStableId==1 && worlds[0].world==identityMatrix()
                && worlds[1].nodeStableId==2049 && near(worlds[1].world[12],1),
                "the rigid static member keeps an identity palette and follows its animated parent leaf");
            check(!Fixture::leafFrame(mixed,{compositionLeafSample(current),compositionLeafSample(current)},frameTime,error),
                "a static leaf refuses a deformation sample");
        }
        check(!Fixture::leafOwner({staticCompositionLeaf(still),staticCompositionLeaf(translated)},
                translated.procedural,scene,mappings,error),
            "a multi-leaf owner needs at least one animated leaf");
    }
    {
        // Each leaf keeps its own node bound; leaf k addresses its nodes
        // through the k * 2048 stride, so several leaves stay below 2048.
        const auto widened=[&](std::size_t count) {
            auto document=std::make_shared<videohelper::gltf::GlbStaticMeshDocument>(*translated.document);
            videohelper::gltf::GlbNodeRecord extra; extra.name="Bound node";
            document->nodes.resize(count,extra);
            return std::shared_ptr<const videohelper::gltf::GlbStaticMeshDocument>(std::move(document));
        };
        std::string error;
        check(Fixture::owner(translated.payload,widened(4096),translated.animationDocument,translated.source,
                translated.procedural,translated.composed,translated.mappings,error)
            && !Fixture::owner(translated.payload,widened(4097),translated.animationDocument,translated.source,
                translated.procedural,translated.composed,translated.mappings,error),
            "a single leaf keeps its 4096-node bound");
        const auto bounded=[&](std::size_t count) {
            std::string refusal;
            return Fixture::leafOwner({animatedCompositionLeaf(translated,{},widened(count)),
                animatedCompositionLeaf(scaled,scaledSource)},translated.procedural,twoLeafScene,twoLeafMappings,refusal);
        };
        const auto wide=bounded(2047);
        check(wide && wide->importedNodeCount()==2048 && !bounded(2048),
            "each of several leaves stays below the 2048 leaf stride");
        if (wide)
        {
            const auto first=translated.snapshot(15), second=scaled.snapshot(15);
            const auto frame=Fixture::leafFrame(wide,{compositionLeafSample(first),compositionLeafSample(second)},frameTime,error);
            const auto batch=compositionLeafBatch(translated,wide);
            arbitgpu::NativeDeformationFrameData value;
            arbitgpu::NativeSceneMotionSample::ImportedNodeWorldMatrices worlds;
            const bool ready=frame && arbitgpu::prepareNativeDeformationFrame(*batch,*first,frame,value,error,&worlds);
            check(ready && worlds.size()==2048 && worlds[2046].nodeStableId==2047 && worlds[2047].nodeStableId==2049,
                "a 2047-node leaf publishes IDs 1 to 2047 and leaf 1 starts at 2049");
            if (ready)
            {
                const auto table=std::make_shared<const arbitgpu::NativeSceneMotionSample::ImportedNodeWorldMatrices>(worlds);
                auto foreign=worlds; foreign.back().nodeStableId=2050;
                auto missing=worlds; missing.back().nodeStableId=4097;
                check(arbitgpu::validNativeCompositionWorldTable(frame,table)
                    && !arbitgpu::validNativeCompositionWorldTable(frame,
                        std::make_shared<const arbitgpu::NativeSceneMotionSample::ImportedNodeWorldMatrices>(foreign))
                    && !arbitgpu::validNativeCompositionWorldTable(frame,
                        std::make_shared<const arbitgpu::NativeSceneMotionSample::ImportedNodeWorldMatrices>(missing)),
                    "the world table accepts each leaf's own node range and refuses IDs past it");
            }
        }
        auto staleSource=std::make_shared<arbitgpu::NativeDeformationScene>(*scaledSource);
        staleSource->structuralRevision=76;
        auto rawMappings=twoLeafMappings; rawMappings[2].importedNodeStableId=1;
        auto missingMappings=twoLeafMappings; missingMappings[2].importedNodeStableId=4097;
        check(!twoLeafOwner(twoLeafScene,twoLeafMappings,staleSource) && !twoLeafOwner(twoLeafScene,rawMappings,scaledSource)
            && !twoLeafOwner(twoLeafScene,missingMappings,scaledSource),
            "multi-leaf owners refuse a split revision, a raw leaf-1 node and a missing leaf");
    }
    {
        // N == 1 keeps the released owner and frame; a second identical leaf
        // reproduces its member values exactly.
        CompositionFixture single(true);
        const auto current=single.snapshot(15);
        std::string error;
        const auto frame=Fixture::frame(single.owner,current,error);
        const auto batch=single.batch();
        arbitgpu::NativeDeformationFrameData imported, procedural;
        arbitgpu::NativeSceneMotionSample::ImportedNodeWorldMatrices worlds;
        check(single.owner && frame && single.owner->leafCount()==1 && single.owner->importedNodeCount()==1
            && !single.owner->staticLeaf(0) && single.owner->memberLeaf(0)==0 && single.owner->memberLeaf(1)==0
            && !single.owner->memberRigid(0) && single.owner->memberRigid(1)
            && &frame->memberSnapshot(0)==current.get() && &frame->memberSnapshot(1)==current.get()
            && arbitgpu::prepareNativeDeformationFrame(*batch,*current,frame,imported,error,&worlds)
            && worlds.size()==1 && worlds[0].nodeStableId==1 && near(worlds[0].world[12],1)
            && arbitgpu::prepareNativeDeformationFrame(*batch->draws[0],*current,frame,imported,error)
            && arbitgpu::prepareNativeDeformationFrame(*batch->draws[1],*current,frame,procedural,error)
            && near(imported.jointPalette[12],1) && near(procedural.compositionPost.point()[12],1),
            "a single-leaf owner keeps raw node IDs, one snapshot and its released values");
        {
            // The leaf lists are vector headers the released owner and frame
            // did not have. A single leaf charges exactly the released schedule.
            using Owner=arbitgpu::NativeDeformationCompositionOwner;
            using Frame=arbitgpu::NativeDeformationCompositionFrame;
            const std::size_t released=sizeof(Owner)-sizeof(std::vector<int>), n=single.mappings.size();
            const std::size_t expected=2*released+4096*sizeof(bool)+12288*sizeof(bool)
                +n*(sizeof(arbitgpu::NativeDeformationScene)+sizeof(visualdeformation::DeformationAsset)
                    +sizeof(visualdeformation::Skin)+sizeof(visualdeformation::Joint)+sizeof(visualdeformation::Mesh)
                    +sizeof(visualdeformation::JointWeightSet)+sizeof(arbitgpu::NativeDeformationJointBaseTransform)
                    +sizeof(std::vector<std::uint32_t>)+sizeof(std::vector<float>))
                +2*4096*4*(sizeof(std::uint32_t)+sizeof(float))+2*n*sizeof(arbitgpu::NativeDeformationFrameData);
            check(single.owner && single.owner->logicalBytes()==expected
                && Owner::singleLeafBytes()==released && Owner::leafListBytes(1)==0 && Owner::leafListBytes(2)>0
                && Frame::singleLeafBytes()==sizeof(Frame)-sizeof(std::vector<int>) && Frame::leafListBytes(1)==0,
                "a single-leaf owner charges the released schedule without leaf list storage");
        }
        check(!Fixture::leafOwner({animatedCompositionLeaf(single)},single.procedural,single.composed,single.mappings,error)
            && !Fixture::leafFrame(single.owner,{compositionLeafSample(current)},frameTime,error),
            "the multi-leaf entries refuse a single leaf");
        auto scene=std::make_shared<CompositionScene>(*single.composed);
        appendCompositionLeafObject(*scene,*single.source->scene,{2001},{});
        const std::vector<CompositionMapping> mappings {
            {CompositionMapping::Kind::Imported,{1},{1},1},
            {CompositionMapping::Kind::Procedural,{65537},{900},0},
            {CompositionMapping::Kind::Imported,{1},{2001},2049}};
        const auto twin=Fixture::leafOwner({animatedCompositionLeaf(single),animatedCompositionLeaf(single)},
            single.procedural,scene,mappings,error);
        const auto twinFrame=Fixture::leafFrame(twin,{compositionLeafSample(current),compositionLeafSample(current)},frameTime,error);
        const auto twinBatch=compositionLeafBatch(single,twin);
        const auto first=compositionMemberDraw(*twinBatch,1), second=compositionMemberDraw(*twinBatch,2001),
            child=compositionMemberDraw(*twinBatch,900);
        arbitgpu::NativeDeformationFrameData a, b, p;
        check(twinFrame && first && second && child
            && arbitgpu::prepareNativeDeformationFrame(*first,*current,twinFrame,a,error)
            && arbitgpu::prepareNativeDeformationFrame(*second,*current,twinFrame,b,error)
            && arbitgpu::prepareNativeDeformationFrame(*child,*current,twinFrame,p,error)
            && a.jointPalette==imported.jointPalette && b.jointPalette==imported.jointPalette
            && a.compositionPost.point()==imported.compositionPost.point()
            && p.compositionPost.point()==procedural.compositionPost.point(),
            "a duplicated leaf reproduces the single-leaf palette and posts");
    }
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
    testCompositionCapabilities();
    testMultiLeafComposition();
    testSharedPreviewExportRoute();
    testUnavailableBackendHasNoFallback();

    if (failures != 0)
        std::fprintf (stderr, "%d native animation deformation test(s) failed\n", failures);
    return failures == 0 ? 0 : 1;
}
