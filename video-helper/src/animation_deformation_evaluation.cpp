#include "../../shared/VisualAnimationDeformationEvaluation.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <utility>

namespace visualdeformation
{
namespace
{
AnimationDeformationLimits boundedLimits (const AnimationDeformationLimits& requested) noexcept
{
    const AnimationDeformationLimits hard;
    return {
        std::min(requested.maxSampledTracks, hard.maxSampledTracks),
        std::min(requested.maxJointTransforms, hard.maxJointTransforms),
        std::min(requested.maxMorphMeshes, hard.maxMorphMeshes),
        std::min(requested.maxMorphWeights, hard.maxMorphWeights),
        std::min(requested.maxTotalScalarValues, hard.maxTotalScalarValues)
    };
}

bool addWithin (std::size_t value, std::size_t& total, std::size_t limit) noexcept
{
    if (value > limit || total > limit - value)
        return false;
    total += value;
    return true;
}

bool addProductWithin (std::size_t left, std::size_t right,
                       std::size_t& total, std::size_t limit) noexcept
{
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
        return false;
    return addWithin(left * right, total, limit);
}

const Joint* findJoint (const Skin& skin, JointId id) noexcept
{
    const auto found = std::find_if(skin.joints().begin(), skin.joints().end(),
                                    [id] (const Joint& joint) { return joint.id() == id; });
    return found == skin.joints().end() ? nullptr : &*found;
}

bool targetLess (const JointAnimationBindingView& left,
                 const JointAnimationBindingView& right) noexcept
{
    return left.animationTarget < right.animationTarget;
}

bool targetLess (const MorphAnimationBindingView& left,
                 const MorphAnimationBindingView& right) noexcept
{
    return left.animationTarget < right.animationTarget;
}

bool allFinite (const std::vector<float>& values) noexcept
{
    return std::all_of(values.begin(), values.end(),
                       [] (float value) { return std::isfinite(value); });
}

std::size_t expectedWidth (visualanimation::Channel channel) noexcept
{
    switch (channel)
    {
        case visualanimation::Channel::Translation:
        case visualanimation::Channel::Scale:
            return 3;
        case visualanimation::Channel::Rotation:
            return 4;
        case visualanimation::Channel::MorphWeights:
            return 0;
    }
    return 0;
}
} // namespace

class AnimationDeformationEvaluationBuilder
{
public:
    static std::shared_ptr<const AnimationDeformationSnapshot> build (
        const DeformationAsset& asset,
        const visualanimation::Clip& clip,
        const AnimationDeformationBindingView& bindings,
        const AnimationDeformationRequest& request,
        const AnimationDeformationLimits& requestedLimits,
        std::string& error)
    {
        error.clear();
        if (request.time.rateNumerator == 0 || request.time.rateDenominator == 0)
        {
            error = "animation deformation frame rate is missing";
            return {};
        }
        if (request.revision == 0)
        {
            error = "animation deformation revision is missing";
            return {};
        }
        if ((request.combinationMode != visualanimation::CombinationMode::Replace
             && request.combinationMode != visualanimation::CombinationMode::WeightedBlend
             && request.combinationMode != visualanimation::CombinationMode::Add
             && request.combinationMode != visualanimation::CombinationMode::Multiply
             && request.combinationMode
                 != visualanimation::CombinationMode::AddAfterImportedAnimation)
            || ! std::isfinite(request.combinationWeight)
            || request.combinationWeight < 0.0 || request.combinationWeight > 1.0)
        {
            error = "animation deformation combination mode or weight is invalid";
            return {};
        }

        const auto limits = boundedLimits(requestedLimits);
        if (clip.tracks().size() > limits.maxSampledTracks
            || bindings.jointCount > limits.maxJointTransforms
            || bindings.morphCount > limits.maxMorphMeshes)
        {
            error = "animation deformation sample capacity exceeded";
            return {};
        }
        if ((bindings.jointCount != 0 && bindings.joints == nullptr)
            || (bindings.morphCount != 0 && bindings.morphs == nullptr))
        {
            error = "animation deformation binding storage is missing";
            return {};
        }
        if (bindings.jointCount == 0 && bindings.morphCount == 0)
        {
            error = "animation deformation bindings are missing";
            return {};
        }

        try
        {
            std::size_t totalScalarValues = 0;
            std::size_t totalMorphWeights = 0;
            if (! addProductWithin(bindings.jointCount, 10, totalScalarValues,
                                   limits.maxTotalScalarValues))
            {
                error = "animation deformation scalar sample capacity exceeded";
                return {};
            }

            std::vector<JointAnimationBindingView> jointBindings;
            if (bindings.jointCount != 0)
                jointBindings.assign(bindings.joints, bindings.joints + bindings.jointCount);
            std::sort(jointBindings.begin(), jointBindings.end(),
                      [] (const auto& left, const auto& right)
                      { return targetLess(left, right); });

            std::vector<std::pair<std::uint64_t, std::uint64_t>> boundJoints;
            boundJoints.reserve(jointBindings.size());
            for (std::size_t index = 0; index < jointBindings.size(); ++index)
            {
                const auto& binding = jointBindings[index];
                if (! binding.animationTarget.isValid() || ! binding.skin.isValid()
                    || ! binding.joint.isValid())
                {
                    error = "joint animation binding has a missing stable identity";
                    return {};
                }
                if (index != 0
                    && binding.animationTarget == jointBindings[index - 1].animationTarget)
                {
                    error = "animation deformation bindings duplicate an animation target identity";
                    return {};
                }
                const auto* skin = asset.findSkin(binding.skin);
                if (skin == nullptr || findJoint(*skin, binding.joint) == nullptr)
                {
                    error = "joint animation binding does not match the deformation asset";
                    return {};
                }
                boundJoints.emplace_back(binding.skin.value, binding.joint.value);
            }
            std::sort(boundJoints.begin(), boundJoints.end());
            if (std::adjacent_find(boundJoints.begin(), boundJoints.end()) != boundJoints.end())
            {
                error = "animation deformation bindings duplicate a joint target";
                return {};
            }

            std::vector<MorphAnimationBindingView> morphBindings;
            if (bindings.morphCount != 0)
                morphBindings.assign(bindings.morphs, bindings.morphs + bindings.morphCount);
            std::sort(morphBindings.begin(), morphBindings.end(),
                      [] (const auto& left, const auto& right)
                      { return targetLess(left, right); });

            std::vector<std::uint64_t> boundMeshes;
            boundMeshes.reserve(morphBindings.size());
            for (std::size_t index = 0; index < morphBindings.size(); ++index)
            {
                const auto& binding = morphBindings[index];
                if (! binding.animationTarget.isValid() || ! binding.mesh.isValid())
                {
                    error = "morph animation binding has a missing stable identity";
                    return {};
                }
                if (index != 0
                    && binding.animationTarget == morphBindings[index - 1].animationTarget)
                {
                    error = "animation deformation bindings duplicate an animation target identity";
                    return {};
                }
                if (std::binary_search(jointBindings.begin(), jointBindings.end(),
                                       JointAnimationBindingView { binding.animationTarget, {}, {} },
                                       [] (const auto& left, const auto& right)
                                       { return targetLess(left, right); }))
                {
                    error = "animation deformation bindings duplicate an animation target identity";
                    return {};
                }

                const auto* mesh = asset.findMesh(binding.mesh);
                if (mesh == nullptr || mesh->morphTargets().empty()
                    || binding.targetCount != mesh->morphTargets().size())
                {
                    error = "morph animation binding does not match the deformation mesh";
                    return {};
                }
                if (binding.targetCount != 0 && binding.targets == nullptr)
                {
                    error = "morph animation binding target storage is missing";
                    return {};
                }
                if (! addWithin(binding.targetCount, totalMorphWeights,
                                limits.maxMorphWeights)
                    || ! addWithin(binding.targetCount, totalScalarValues,
                                   limits.maxTotalScalarValues))
                {
                    error = "animation deformation morph sample capacity exceeded";
                    return {};
                }

                std::vector<std::uint64_t> targetIds;
                targetIds.reserve(binding.targetCount);
                for (std::size_t target = 0; target < binding.targetCount; ++target)
                {
                    if (! binding.targets[target].isValid())
                    {
                        error = "morph animation binding has a missing target identity";
                        return {};
                    }
                    targetIds.push_back(binding.targets[target].value);
                }
                std::sort(targetIds.begin(), targetIds.end());
                if (std::adjacent_find(targetIds.begin(), targetIds.end()) != targetIds.end())
                {
                    error = "morph animation binding duplicates a target identity";
                    return {};
                }
                for (std::size_t target = 0; target < binding.targetCount; ++target)
                    if (binding.targets[target] != mesh->morphTargets()[target].id())
                    {
                        error = "morph animation binding target order does not match the mesh";
                        return {};
                    }
                boundMeshes.push_back(binding.mesh.value);
            }
            std::sort(boundMeshes.begin(), boundMeshes.end());
            if (std::adjacent_find(boundMeshes.begin(), boundMeshes.end()) != boundMeshes.end())
            {
                error = "animation deformation bindings duplicate a morph mesh target";
                return {};
            }

            for (const auto& track : clip.tracks())
            {
                const auto jointFound = std::lower_bound(
                    jointBindings.begin(), jointBindings.end(), track.target(),
                    [] (const JointAnimationBindingView& value,
                        visualanimation::TargetId target)
                    { return value.animationTarget < target; });
                const auto morphFound = std::lower_bound(
                    morphBindings.begin(), morphBindings.end(), track.target(),
                    [] (const MorphAnimationBindingView& value,
                        visualanimation::TargetId target)
                    { return value.animationTarget < target; });
                const auto isJoint = jointFound != jointBindings.end()
                                  && jointFound->animationTarget == track.target();
                const auto isMorph = morphFound != morphBindings.end()
                                  && morphFound->animationTarget == track.target();

                if (! isJoint && ! isMorph)
                {
                    error = "sampled animation target has no deformation binding";
                    return {};
                }
                if (track.channel() == visualanimation::Channel::MorphWeights)
                {
                    if (! isMorph)
                    {
                        error = "sampled morph channel does not match its deformation target";
                        return {};
                    }
                    if (track.valueWidth() != morphFound->targetCount)
                    {
                        error = "sampled morph target and value cardinalities do not match the mesh";
                        return {};
                    }
                }
                else
                {
                    if (! isJoint || isMorph)
                    {
                        error = "sampled transform channel does not match its deformation target";
                        return {};
                    }
                    if (track.valueWidth() != expectedWidth(track.channel()))
                    {
                        error = "sampled transform channel has an invalid value width";
                        return {};
                    }
                }
            }

            const auto requestedTimeSeconds = request.requestedTimeSeconds.value_or(
                static_cast<double>(request.time.frame)
                * static_cast<double>(request.time.rateDenominator)
                / static_cast<double>(request.time.rateNumerator));
            if (! std::isfinite(requestedTimeSeconds))
            {
                error = "animation deformation frame time is out of range";
                return {};
            }

            std::string sampleError;
            const auto sampled = visualanimation::sample(
                clip, { requestedTimeSeconds, request.playback,
                        request.sampleRangeStartSeconds,
                        request.sampleRangeEndSeconds }, sampleError);
            if (! sampled)
            {
                error = sampleError;
                return {};
            }

            std::vector<std::uint64_t> sampledTrackIds;
            sampledTrackIds.reserve(sampled->tracks().size());
            for (const auto& track : sampled->tracks())
            {
                if (! track.track().isValid() || ! track.target().isValid())
                {
                    error = "sampled animation channel has a missing stable identity";
                    return {};
                }
                if (! allFinite(track.values()))
                {
                    error = "sampled animation channel contains a nonfinite value";
                    return {};
                }
                sampledTrackIds.push_back(track.track().value);
            }
            std::sort(sampledTrackIds.begin(), sampledTrackIds.end());
            if (std::adjacent_find(sampledTrackIds.begin(), sampledTrackIds.end())
                != sampledTrackIds.end())
            {
                error = "sampled animation channels duplicate a track identity";
                return {};
            }

            std::shared_ptr<AnimationDeformationSnapshot> result(
                new AnimationDeformationSnapshot());
            result->clip_ = clip.id();
            result->revision_ = request.revision;
            result->time_ = request.time;
            result->requestedTimeSeconds_ = requestedTimeSeconds;
            result->sampleTimeSeconds_ = sampled->sampleTimeSeconds();
            result->playback_ = request.playback;
            result->combinationMode_ = request.combinationMode;
            result->combinationWeight_ = request.combinationWeight;
            result->jointTransforms_.reserve(jointBindings.size());
            for (const auto& binding : jointBindings)
            {
                JointTransformEvaluation output;
                output.animationTarget_ = binding.animationTarget;
                output.skin_ = binding.skin;
                output.joint_ = binding.joint;
                result->jointTransforms_.push_back(output);
            }
            result->morphWeights_.resize(morphBindings.size());
            for (std::size_t index = 0; index < morphBindings.size(); ++index)
                result->morphWeights_[index].animationTarget_
                    = morphBindings[index].animationTarget;

            for (const auto& track : sampled->tracks())
            {
                const auto jointFound = std::lower_bound(
                    result->jointTransforms_.begin(), result->jointTransforms_.end(),
                    track.target(), [] (const JointTransformEvaluation& value,
                                        visualanimation::TargetId target)
                    { return value.animationTarget() < target; });
                const auto morphFound = std::lower_bound(
                    morphBindings.begin(), morphBindings.end(), track.target(),
                    [] (const MorphAnimationBindingView& value,
                        visualanimation::TargetId target)
                    { return value.animationTarget < target; });
                const auto isJoint = jointFound != result->jointTransforms_.end()
                                  && jointFound->animationTarget() == track.target();
                const auto isMorph = morphFound != morphBindings.end()
                                  && morphFound->animationTarget == track.target();

                if (! isJoint && ! isMorph)
                {
                    error = "sampled animation target has no deformation binding";
                    return {};
                }
                if (track.channel() == visualanimation::Channel::MorphWeights)
                {
                    if (! isMorph)
                    {
                        error = "sampled morph channel does not match its deformation target";
                        return {};
                    }
                    const auto index = static_cast<std::size_t>(morphFound - morphBindings.begin());
                    if (result->morphWeights_[index].sample_ != nullptr)
                    {
                        error = "sampled animation target duplicates a morph channel";
                        return {};
                    }
                    const MorphWeightSampleView sampleView {
                        morphFound->mesh, morphFound->targets, morphFound->targetCount,
                        track.values().data(), track.values().size()
                    };
                    std::string morphError;
                    auto morphSample = MorphWeightSample::create(asset, sampleView, {}, morphError);
                    if (! morphSample)
                    {
                        error = morphError;
                        return {};
                    }
                    result->morphWeights_[index].track_ = track.track();
                    result->morphWeights_[index].sample_ = std::move(morphSample);
                    continue;
                }
                if (! isJoint || isMorph)
                {
                    error = "sampled transform channel does not match its deformation target";
                    return {};
                }
                if (track.values().size() != expectedWidth(track.channel()))
                {
                    error = "sampled transform channel has an invalid value width";
                    return {};
                }

                auto& output = *jointFound;
                switch (track.channel())
                {
                    case visualanimation::Channel::Translation:
                        if (output.translationTrack_.isValid())
                        {
                            error = "sampled animation target duplicates a translation channel";
                            return {};
                        }
                        output.translationTrack_ = track.track();
                        std::copy(track.values().begin(), track.values().end(),
                                  output.translation_.begin());
                        break;
                    case visualanimation::Channel::Rotation:
                        if (output.rotationTrack_.isValid())
                        {
                            error = "sampled animation target duplicates a rotation channel";
                            return {};
                        }
                        output.rotationTrack_ = track.track();
                        std::copy(track.values().begin(), track.values().end(),
                                  output.rotation_.begin());
                        break;
                    case visualanimation::Channel::Scale:
                        if (output.scaleTrack_.isValid())
                        {
                            error = "sampled animation target duplicates a scale channel";
                            return {};
                        }
                        output.scaleTrack_ = track.track();
                        std::copy(track.values().begin(), track.values().end(),
                                  output.scale_.begin());
                        break;
                    case visualanimation::Channel::MorphWeights:
                        break;
                }
            }

            for (const auto& output : result->jointTransforms_)
                if (! output.translationTrack().isValid()
                    && ! output.rotationTrack().isValid()
                    && ! output.scaleTrack().isValid())
                {
                    error = "joint animation binding has no sampled channel";
                    return {};
                }
            for (const auto& output : result->morphWeights_)
                if (output.sample_ == nullptr)
                {
                    error = "morph animation binding has no sampled channel";
                    return {};
                }
            return result;
        }
        catch (const std::bad_alloc&)
        {
            error = "animation deformation evaluation allocation failed within admitted limits";
            return {};
        }
    }
};

std::shared_ptr<const AnimationDeformationSnapshot> evaluateAnimationDeformation (
    const DeformationAsset& asset,
    const visualanimation::Clip& clip,
    const AnimationDeformationBindingView& bindings,
    const AnimationDeformationRequest& request,
    const AnimationDeformationLimits& limits,
    std::string& error)
{
    return AnimationDeformationEvaluationBuilder::build(
        asset, clip, bindings, request, limits, error);
}
} // namespace visualdeformation
