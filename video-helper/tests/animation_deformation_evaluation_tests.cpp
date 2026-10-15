#include "VisualAnimationDeformationEvaluation.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
using namespace visualdeformation;
namespace animation = visualanimation;

int failures = 0;

void check (bool condition, const char* message)
{
    if (condition)
        return;
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", message);
}

bool near (float actual, float expected, float tolerance = 1.0e-5f)
{
    return std::abs(actual - expected) <= tolerance;
}

using Matrix = std::array<float, 16>;
using Position = std::array<float, 3>;

Matrix identityMatrix()
{
    return { 1.0f, 0.0f, 0.0f, 0.0f,
             0.0f, 1.0f, 0.0f, 0.0f,
             0.0f, 0.0f, 1.0f, 0.0f,
             0.0f, 0.0f, 0.0f, 1.0f };
}

Matrix multiply (const Matrix& left, const Matrix& right)
{
    Matrix result {};
    for (std::size_t column = 0; column < 4; ++column)
        for (std::size_t row = 0; row < 4; ++row)
            for (std::size_t inner = 0; inner < 4; ++inner)
                result[column * 4 + row]
                    += left[inner * 4 + row] * right[column * 4 + inner];
    return result;
}

Matrix compose (const JointTransformEvaluation& transform)
{
    const auto& q = transform.rotation();
    const auto& s = transform.scale();
    const auto& t = transform.translation();
    const auto x = q[0];
    const auto y = q[1];
    const auto z = q[2];
    const auto w = q[3];
    return {
        (1.0f - 2.0f * (y * y + z * z)) * s[0],
        (2.0f * (x * y + z * w)) * s[0],
        (2.0f * (x * z - y * w)) * s[0], 0.0f,
        (2.0f * (x * y - z * w)) * s[1],
        (1.0f - 2.0f * (x * x + z * z)) * s[1],
        (2.0f * (y * z + x * w)) * s[1], 0.0f,
        (2.0f * (x * z + y * w)) * s[2],
        (2.0f * (y * z - x * w)) * s[2],
        (1.0f - 2.0f * (x * x + y * y)) * s[2], 0.0f,
        t[0], t[1], t[2], 1.0f
    };
}

Position transformPoint (const Matrix& matrix, const Position& point)
{
    return {
        matrix[0] * point[0] + matrix[4] * point[1] + matrix[8] * point[2] + matrix[12],
        matrix[1] * point[0] + matrix[5] * point[1] + matrix[9] * point[2] + matrix[13],
        matrix[2] * point[0] + matrix[6] * point[1] + matrix[10] * point[2] + matrix[14]
    };
}

struct DeformationFixture
{
    std::array<JointView, 2> joints {
        JointView { JointId { 101 }, {} },
        JointView { JointId { 102 }, JointId { 101 } }
    };
    std::vector<float> inverseBindValues;
    std::array<std::uint32_t, 8> jointIndices { 0, 1, 0, 0, 0, 0, 0, 0 };
    std::array<float, 8> jointWeights { 0.5f, 0.5f, 0.0f, 0.0f,
                                       1.0f, 0.0f, 0.0f, 0.0f };
    std::array<float, 6> targetAPositions { 2.0f, 0.0f, 0.0f,
                                            0.0f, 2.0f, 0.0f };
    std::array<float, 6> targetBPositions { 0.0f, 0.0f, 2.0f,
                                            0.0f, 0.0f, 0.0f };
    JointWeightSetView weightSet;
    std::array<MorphTargetView, 2> targets;
    SkinView skin;
    MeshView mesh;
    AssetView view;

    DeformationFixture()
    {
        const auto identity = identityMatrix();
        inverseBindValues.insert(inverseBindValues.end(), identity.begin(), identity.end());
        inverseBindValues.insert(inverseBindValues.end(), identity.begin(), identity.end());
        weightSet = { jointIndices.data(), jointIndices.size(),
                      jointWeights.data(), jointWeights.size() };
        targets = {
            MorphTargetView { MorphTargetId { 301 }, targetAPositions.data(),
                              targetAPositions.size(), nullptr, 0, nullptr, 0 },
            MorphTargetView { MorphTargetId { 302 }, targetBPositions.data(),
                              targetBPositions.size(), nullptr, 0, nullptr, 0 }
        };
        skin = { SkinId { 201 }, joints.data(), joints.size(),
                 inverseBindValues.data(), inverseBindValues.size() };
        mesh = { MeshId { 401 }, 2, skin.id, &weightSet, 1, targets.data(), targets.size() };
        view = { &skin, 1, &mesh, 1 };
    }
};

struct AnimationFixture
{
    std::array<double, 2> times { 0.0, 1.0 };
    std::array<float, 6> rootTranslation { 0.0f, 0.0f, 0.0f,
                                           4.0f, 0.0f, 0.0f };
    std::array<float, 8> rootRotation { 0.0f, 0.0f, 0.0f, 1.0f,
                                        0.0f, 0.0f, 0.0f, 1.0f };
    std::array<float, 6> childTranslation { 0.0f, 0.0f, 0.0f,
                                            0.0f, 6.0f, 0.0f };
    std::array<float, 4> morphWeights { 0.0f, 0.0f, 0.5f, 1.0f };
    std::array<animation::TrackView, 4> tracks;
    animation::ClipView view;

    AnimationFixture()
    {
        tracks = {
            animation::TrackView { animation::TrackId { 11 }, animation::TargetId { 1001 },
                animation::Channel::Translation, animation::Interpolation::Linear,
                times.data(), rootTranslation.data(), times.size(), rootTranslation.size(), 0 },
            animation::TrackView { animation::TrackId { 12 }, animation::TargetId { 1001 },
                animation::Channel::Rotation, animation::Interpolation::Linear,
                times.data(), rootRotation.data(), times.size(), rootRotation.size(), 0 },
            animation::TrackView { animation::TrackId { 13 }, animation::TargetId { 1002 },
                animation::Channel::Translation, animation::Interpolation::Linear,
                times.data(), childTranslation.data(), times.size(), childTranslation.size(), 0 },
            animation::TrackView { animation::TrackId { 14 }, animation::TargetId { 2001 },
                animation::Channel::MorphWeights, animation::Interpolation::Linear,
                times.data(), morphWeights.data(), times.size(), morphWeights.size(), 2 }
        };
        view = { animation::ClipId { 501 }, 1.0, tracks.data(), tracks.size() };
    }
};

bool snapshotsEqual (const AnimationDeformationSnapshot& left,
                     const AnimationDeformationSnapshot& right)
{
    if (left.clip() != right.clip() || left.revision() != right.revision()
        || ! (left.time() == right.time())
        || left.requestedTimeSeconds() != right.requestedTimeSeconds()
        || left.sampleTimeSeconds() != right.sampleTimeSeconds()
        || left.playback() != right.playback()
        || left.jointTransforms().size() != right.jointTransforms().size()
        || left.morphWeights().size() != right.morphWeights().size())
        return false;
    for (std::size_t index = 0; index < left.jointTransforms().size(); ++index)
    {
        const auto& a = left.jointTransforms()[index];
        const auto& b = right.jointTransforms()[index];
        if (a.animationTarget() != b.animationTarget() || a.skin() != b.skin()
            || a.joint() != b.joint() || a.translationTrack() != b.translationTrack()
            || a.rotationTrack() != b.rotationTrack() || a.scaleTrack() != b.scaleTrack()
            || a.translation() != b.translation() || a.rotation() != b.rotation()
            || a.scale() != b.scale())
            return false;
    }
    for (std::size_t index = 0; index < left.morphWeights().size(); ++index)
    {
        const auto& a = left.morphWeights()[index];
        const auto& b = right.morphWeights()[index];
        if (a.animationTarget() != b.animationTarget() || a.track() != b.track()
            || a.mesh() != b.mesh() || a.targets() != b.targets() || a.values() != b.values())
            return false;
    }
    return true;
}

// Test oracle only. Production evaluation stops at immutable TRS and morph
// weights so a renderer cannot silently substitute CPU skinning for a GPU path.
std::vector<Position> referenceDeform (const DeformationAsset& asset,
                                       const AnimationDeformationSnapshot& snapshot,
                                       const std::vector<Position>& basePositions)
{
    const auto& skin = asset.skins().front();
    const auto& mesh = asset.meshes().front();
    std::vector<Matrix> world(skin.joints().size());
    for (std::size_t jointIndex = 0; jointIndex < skin.joints().size(); ++jointIndex)
    {
        const auto jointId = skin.joints()[jointIndex].id();
        const auto sampled = std::find_if(snapshot.jointTransforms().begin(),
                                          snapshot.jointTransforms().end(),
                                          [jointId] (const auto& value)
                                          { return value.joint() == jointId; });
        const auto local = compose(*sampled);
        const auto parentId = skin.joints()[jointIndex].parent();
        if (! parentId.isValid())
            world[jointIndex] = local;
        else
        {
            const auto parent = std::find_if(skin.joints().begin(), skin.joints().end(),
                                             [parentId] (const auto& value)
                                             { return value.id() == parentId; });
            world[jointIndex] = multiply(world[static_cast<std::size_t>(
                                           parent - skin.joints().begin())], local);
        }
    }

    std::vector<Matrix> palettes;
    palettes.reserve(world.size());
    for (std::size_t joint = 0; joint < world.size(); ++joint)
    {
        Matrix inverseBind {};
        std::copy_n(skin.inverseBindMatrixValues().begin()
                        + static_cast<std::ptrdiff_t>(joint * 16),
                    16, inverseBind.begin());
        palettes.push_back(multiply(world[joint], inverseBind));
    }

    auto positions = basePositions;
    const auto& morph = snapshot.morphWeights().front();
    for (std::size_t target = 0; target < mesh.morphTargets().size(); ++target)
        for (std::size_t vertex = 0; vertex < positions.size(); ++vertex)
            for (std::size_t axis = 0; axis < 3; ++axis)
                positions[vertex][axis] += morph.values()[target]
                    * mesh.morphTargets()[target].positionDeltas()[vertex * 3 + axis];

    std::vector<Position> result(positions.size(), { 0.0f, 0.0f, 0.0f });
    for (std::size_t vertex = 0; vertex < positions.size(); ++vertex)
        for (const auto& set : mesh.jointWeightSets())
            for (std::size_t component = 0; component < 4; ++component)
            {
                const auto value = vertex * 4 + component;
                const auto transformed = transformPoint(palettes[set.jointIndices()[value]],
                                                        positions[vertex]);
                for (std::size_t axis = 0; axis < 3; ++axis)
                    result[vertex][axis] += set.weights()[value] * transformed[axis];
            }
    return result;
}
} // namespace

int main()
{
    static_assert(std::is_const_v<std::remove_reference_t<decltype(
        std::declval<const AnimationDeformationSnapshot&>().jointTransforms())>>,
        "evaluated joint transforms must be immutable");
    static_assert(std::is_const_v<std::remove_reference_t<decltype(
        std::declval<const AnimationDeformationSnapshot&>().morphWeights())>>,
        "evaluated morph weights must be immutable");

    std::string error;
    DeformationFixture deformationSource;
    const auto asset = DeformationAsset::create(deformationSource.view, {}, error);
    check(asset != nullptr && error.empty(), "deformation fixture is admitted");
    AnimationFixture animationSource;
    const auto clip = animation::Clip::create(animationSource.view, {}, error);
    check(clip != nullptr && error.empty(), "animation fixture is admitted");
    if (! asset || ! clip)
        return 1;

    std::array<JointAnimationBindingView, 2> joints {
        JointAnimationBindingView { animation::TargetId { 1002 }, SkinId { 201 }, JointId { 102 } },
        JointAnimationBindingView { animation::TargetId { 1001 }, SkinId { 201 }, JointId { 101 } }
    };
    std::array<MorphTargetId, 2> morphTargets { MorphTargetId { 301 }, MorphTargetId { 302 } };
    std::array<MorphAnimationBindingView, 1> morphs {
        MorphAnimationBindingView { animation::TargetId { 2001 }, MeshId { 401 },
                                    morphTargets.data(), morphTargets.size() }
    };
    const AnimationDeformationBindingView bindings {
        joints.data(), joints.size(), morphs.data(), morphs.size()
    };
    const AnimationDeformationRequest request {
        RationalFrameTime { 15, 30, 1 }, 77, animation::Playback::Clamp,
        std::nullopt
    };

    const auto preview = evaluateAnimationDeformation(*asset, *clip, bindings,
                                                       request, {}, error);
    const auto exportEvaluation = evaluateAnimationDeformation(*asset, *clip, bindings,
                                                                request, {}, error);
    check(preview != nullptr && exportEvaluation != nullptr && error.empty(),
          "valid animation and deformation bindings evaluate");
    check(preview && exportEvaluation && snapshotsEqual(*preview, *exportEvaluation),
          "preview and export receive identical snapshots for identical frame and revision inputs");
    check(preview && preview->time() == RationalFrameTime { 15, 30, 1 }
          && preview->revision() == 77 && preview->requestedTimeSeconds() == 0.5
          && preview->sampleTimeSeconds() == 0.5,
          "snapshot retains exact rational frame identity and sampled clip time");
    check(preview && preview->jointTransforms().size() == 2
          && preview->jointTransforms()[0].animationTarget() == animation::TargetId { 1001 }
          && preview->jointTransforms()[1].animationTarget() == animation::TargetId { 1002 },
          "joint evaluations use canonical stable-target order independent of binding order");
    check(preview && preview->jointTransforms()[0].translationTrack() == animation::TrackId { 11 }
          && preview->jointTransforms()[0].rotationTrack() == animation::TrackId { 12 }
          && preview->jointTransforms()[0].hasTranslation()
          && preview->jointTransforms()[0].hasRotation()
          && ! preview->jointTransforms()[0].hasScale()
          && near(preview->jointTransforms()[0].translation()[0], 2.0f)
          && preview->jointTransforms()[0].scale() == std::array<float, 3> { 1.0f, 1.0f, 1.0f },
          "sampled transform channels retain source tracks and mark absent channels");
    check(preview && preview->morphWeights().size() == 1
          && preview->morphWeights()[0].track() == animation::TrackId { 14 }
          && preview->morphWeights()[0].targets() == std::vector<MorphTargetId>(
              morphTargets.begin(), morphTargets.end())
          && near(preview->morphWeights()[0].values()[0], 0.25f)
          && near(preview->morphWeights()[0].values()[1], 0.5f),
          "morph samples retain stable target order and sampled weights");

    auto endRequest = request;
    endRequest.time.frame = 30;
    const auto forward = evaluateAnimationDeformation(*asset, *clip, bindings,
                                                       endRequest, {}, error);
    const auto soughtBack = evaluateAnimationDeformation(*asset, *clip, bindings,
                                                          request, {}, error);
    const auto equalTime = evaluateAnimationDeformation(*asset, *clip, bindings,
                                                         request, {}, error);
    check(forward && forward->sampleTimeSeconds() == 1.0
          && near(forward->jointTransforms()[0].translation()[0], 4.0f),
          "forward evaluation reaches the requested offline frame");
    check(preview && soughtBack && equalTime
          && snapshotsEqual(*preview, *soughtBack)
          && snapshotsEqual(*soughtBack, *equalTime),
          "backward seek and equal-time reevaluation reproduce the exact snapshot");

    auto loopRequest = request;
    loopRequest.time.frame = 45;
    loopRequest.playback = animation::Playback::Loop;
    const auto looped = evaluateAnimationDeformation(*asset, *clip, bindings,
                                                      loopRequest, {}, error);
    check(looped && looped->time() == RationalFrameTime { 45, 30, 1 }
          && looped->requestedTimeSeconds() == 1.5
          && looped->sampleTimeSeconds() == 0.5
          && near(looped->jointTransforms()[0].translation()[0], 2.0f)
          && near(looped->morphWeights()[0].values()[1], 0.5f),
          "loop evaluation retains exact frame identity and wraps sampled clip time");

    std::array<AnimationDeformationRequest, 3> offlineRequests {
        AnimationDeformationRequest { RationalFrameTime { 0, 30, 1 }, 77,
                                      animation::Playback::Clamp, std::nullopt },
        request,
        endRequest
    };
    std::array<std::shared_ptr<const AnimationDeformationSnapshot>, 3> offlineForward;
    for (std::size_t index = 0; index < offlineRequests.size(); ++index)
        offlineForward[index] = evaluateAnimationDeformation(
            *asset, *clip, bindings, offlineRequests[index], {}, error);
    bool offlineOrderIndependent = true;
    for (std::size_t reverse = offlineRequests.size(); reverse-- > 0;)
    {
        const auto repeated = evaluateAnimationDeformation(
            *asset, *clip, bindings, offlineRequests[reverse], {}, error);
        offlineOrderIndependent = offlineOrderIndependent && repeated
            && offlineForward[reverse]
            && snapshotsEqual(*offlineForward[reverse], *repeated);
    }
    check(offlineOrderIndependent,
          "offline frame evaluation is deterministic and independent of request order");

    if (preview)
    {
        const auto deformed = referenceDeform(*asset, *preview,
                                              { Position { 0.0f, 0.0f, 0.0f },
                                                Position { 1.0f, 0.0f, 0.0f } });
        check(deformed.size() == 2
              && near(deformed[0][0], 2.5f) && near(deformed[0][1], 1.5f)
              && near(deformed[0][2], 1.0f)
              && near(deformed[1][0], 3.0f) && near(deformed[1][1], 0.5f)
              && near(deformed[1][2], 0.0f),
              "test-only CPU oracle applies evaluated morphs before joint palettes");
    }

    morphTargets[0] = MorphTargetId { 999 };
    joints[0].joint = JointId { 999 };
    check(preview && preview->jointTransforms()[1].joint() == JointId { 102 }
          && preview->morphWeights()[0].targets()[0] == MorphTargetId { 301 },
          "evaluation snapshot owns immutable binding and morph identities");
    morphTargets[0] = MorphTargetId { 301 };
    joints[0].joint = JointId { 102 };

    {
        auto invalid = joints;
        invalid[0].animationTarget = {};
        const AnimationDeformationBindingView bad { invalid.data(), invalid.size(),
                                                    morphs.data(), morphs.size() };
        check(! evaluateAnimationDeformation(*asset, *clip, bad, request, {}, error)
              && error == "joint animation binding has a missing stable identity",
              "missing stable binding identities are rejected");
    }
    {
        auto duplicate = joints;
        duplicate[0].animationTarget = duplicate[1].animationTarget;
        const AnimationDeformationBindingView bad { duplicate.data(), duplicate.size(),
                                                    morphs.data(), morphs.size() };
        check(! evaluateAnimationDeformation(*asset, *clip, bad, request, {}, error)
              && error == "animation deformation bindings duplicate an animation target identity",
              "duplicate animation target identities are rejected");
    }
    {
        auto duplicate = joints;
        duplicate[0].joint = duplicate[1].joint;
        const AnimationDeformationBindingView bad { duplicate.data(), duplicate.size(),
                                                    morphs.data(), morphs.size() };
        check(! evaluateAnimationDeformation(*asset, *clip, bad, request, {}, error)
              && error == "animation deformation bindings duplicate a joint target",
              "two animation targets cannot own one deformation joint");
    }
    {
        auto mismatched = joints;
        mismatched[0].joint = JointId { 999 };
        const AnimationDeformationBindingView bad { mismatched.data(), mismatched.size(),
                                                    morphs.data(), morphs.size() };
        check(! evaluateAnimationDeformation(*asset, *clip, bad, request, {}, error)
              && error == "joint animation binding does not match the deformation asset",
              "joint bindings must resolve within the named skin");
    }
    {
        auto wrongOrder = morphTargets;
        std::swap(wrongOrder[0], wrongOrder[1]);
        auto mismatched = morphs;
        mismatched[0].targets = wrongOrder.data();
        const AnimationDeformationBindingView bad { joints.data(), joints.size(),
                                                    mismatched.data(), mismatched.size() };
        check(! evaluateAnimationDeformation(*asset, *clip, bad, request, {}, error)
              && error == "morph animation binding target order does not match the mesh",
              "morph bindings cannot drift sampled values onto another target order");
    }
    {
        std::array<float, 2> oneMorphWeight { 0.0f, 0.5f };
        auto mismatchedAnimation = animationSource;
        mismatchedAnimation.tracks[3].values = oneMorphWeight.data();
        mismatchedAnimation.tracks[3].valueCount = oneMorphWeight.size();
        mismatchedAnimation.tracks[3].morphWeightCount = 1;
        mismatchedAnimation.view.tracks = mismatchedAnimation.tracks.data();
        const auto mismatchedClip = animation::Clip::create(
            mismatchedAnimation.view, {}, error);
        check(mismatchedClip != nullptr,
              "animation clip admits independently of deformation morph cardinality");
        check(mismatchedClip
              && ! evaluateAnimationDeformation(*asset, *mismatchedClip, bindings,
                                                request, {}, error)
              && error == "sampled morph target and value cardinalities do not match the mesh",
              "binding validation rejects morph cardinality drift before evaluation");
    }
    {
        const AnimationDeformationBindingView missing { joints.data(), 1,
                                                        morphs.data(), morphs.size() };
        check(! evaluateAnimationDeformation(*asset, *clip, missing, request, {}, error)
              && error == "sampled animation target has no deformation binding",
              "every sampled animation target requires an explicit deformation binding");
    }
    {
        AnimationDeformationLimits limited;
        limited.maxSampledTracks = 3;
        check(! evaluateAnimationDeformation(*asset, *clip, bindings, request, limited, error)
              && error == "animation deformation sample capacity exceeded",
              "sample track budgets reject before sampling allocations");
        limited = {};
        limited.maxTotalScalarValues = 21;
        check(! evaluateAnimationDeformation(*asset, *clip, bindings, request, limited, error)
              && error == "animation deformation morph sample capacity exceeded",
              "total evaluated scalar budgets include joint defaults and morph weights");
    }
    {
        auto missingRevision = request;
        missingRevision.revision = 0;
        check(! evaluateAnimationDeformation(*asset, *clip, bindings,
                                             missingRevision, {}, error)
              && error == "animation deformation revision is missing",
              "snapshot evaluation requires an explicit nonzero revision");
        auto missingRate = request;
        missingRate.time.rateDenominator = 0;
        check(! evaluateAnimationDeformation(*asset, *clip, bindings,
                                             missingRate, {}, error)
              && error == "animation deformation frame rate is missing",
              "snapshot evaluation requires an explicit rational frame rate");
    }
    {
        auto excessiveAnimation = animationSource;
        excessiveAnimation.morphWeights = { 17.0f, 0.0f, 17.0f, 0.0f };
        excessiveAnimation.tracks[3].values = excessiveAnimation.morphWeights.data();
        excessiveAnimation.view.tracks = excessiveAnimation.tracks.data();
        const auto excessiveClip = animation::Clip::create(excessiveAnimation.view, {}, error);
        check(excessiveClip != nullptr, "finite animation values admit before target-specific bounds");
        check(excessiveClip
              && ! evaluateAnimationDeformation(*asset, *excessiveClip, bindings,
                                                request, {}, error)
              && error == "sampled morph weight is nonfinite or out of range",
              "deformation evaluation enforces the hard sampled morph bound");
    }
    {
        auto nonfiniteAnimation = animationSource;
        nonfiniteAnimation.rootTranslation[0] = std::numeric_limits<float>::infinity();
        nonfiniteAnimation.tracks[0].values = nonfiniteAnimation.rootTranslation.data();
        nonfiniteAnimation.view.tracks = nonfiniteAnimation.tracks.data();
        check(! animation::Clip::create(nonfiniteAnimation.view, {}, error)
              && error == "animation key value is nonfinite",
              "nonfinite animation values fail before an evaluation snapshot exists");
    }

    std::fprintf(stderr, failures == 0
        ? "Animation-to-deformation evaluation checks passed\n"
        : "%d animation-to-deformation evaluation checks failed\n", failures);
    return failures == 0 ? 0 : 1;
}
