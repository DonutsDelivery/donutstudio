#pragma once

#include "VisualAnimationClip.h"
#include "VisualAnimationPoseControls.h"
#include "VisualDeformationContract.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace visualdeformation
{
// Exact frame identity retained by every evaluation. The requested time in
// seconds is frame * rateDenominator / rateNumerator.
struct RationalFrameTime
{
    std::int64_t frame = 0;
    std::uint32_t rateNumerator = 0;
    std::uint32_t rateDenominator = 0;

    constexpr bool operator== (const RationalFrameTime& other) const noexcept
    {
        return frame == other.frame
            && rateNumerator == other.rateNumerator
            && rateDenominator == other.rateDenominator;
    }
};

struct AnimationDeformationLimits
{
    std::size_t maxSampledTracks = 4096;
    std::size_t maxJointTransforms = 4096;
    std::size_t maxMorphMeshes = 4096;
    std::size_t maxMorphWeights = 8192;
    std::size_t maxTotalScalarValues = 65536;
};

struct JointAnimationBindingView
{
    visualanimation::TargetId animationTarget;
    SkinId skin;
    JointId joint;
};

struct MorphAnimationBindingView
{
    visualanimation::TargetId animationTarget;
    MeshId mesh;
    const MorphTargetId* targets = nullptr;
    std::size_t targetCount = 0;
};

struct AnimationDeformationBindingView
{
    const JointAnimationBindingView* joints = nullptr;
    std::size_t jointCount = 0;
    const MorphAnimationBindingView* morphs = nullptr;
    std::size_t morphCount = 0;
    const visualanimation::TargetId* sceneTransforms = nullptr;
    std::size_t sceneTransformCount = 0;
};

struct AnimationDeformationRequest
{
    RationalFrameTime time;
    std::uint64_t revision = 0;
    visualanimation::Playback playback = visualanimation::Playback::Clamp;
    // When present, sample this already-resolved playback time while retaining
    // the exact presentation-frame identity above. Playback-control consumers
    // use this to preserve beat/time remapping without rounding it to a frame.
    std::optional<double> requestedTimeSeconds;
    double sampleRangeStartSeconds = 0.0;
    double sampleRangeEndSeconds = 0.0;
    visualanimation::CombinationMode combinationMode =
        visualanimation::CombinationMode::Replace;
    double combinationWeight = 1.0;
    visualanimation::PoseControls pose {};
    bool allowNodeTransformAndMorph = false;
    // An immutable mixed sample must retain this clip's exact track topology.
    std::shared_ptr<const visualanimation::Sample> mixedSample {};
    // SHA-256 of the exact payload whose deformation asset is sampled. All
    // zero means unattested. A multi-leaf composition refuses that, because
    // clip, skin and joint IDs repeat across GLBs.
    std::array<std::uint8_t, 32> sourceContentSha256 {};
};

// Parses a lowercase 64-digit hex SHA-256 into its 32 bytes.
inline bool sha256Digest(std::string_view hex, std::array<std::uint8_t, 32>& digest) noexcept
{
    if (hex.size() != 64) return false;
    const auto nibble = [](char value) -> int
    {
        if (value >= '0' && value <= '9') return value - '0';
        if (value >= 'a' && value <= 'f') return value - 'a' + 10;
        return -1;
    };
    for (std::size_t index = 0; index < digest.size(); ++index)
    {
        const auto high = nibble(hex[index * 2]), low = nibble(hex[index * 2 + 1]);
        if (high < 0 || low < 0) return false;
        digest[index] = static_cast<std::uint8_t>(high * 16 + low);
    }
    return true;
}

class JointTransformEvaluation
{
public:
    visualanimation::TargetId animationTarget() const noexcept { return animationTarget_; }
    SkinId skin() const noexcept { return skin_; }
    JointId joint() const noexcept { return joint_; }
    visualanimation::TrackId translationTrack() const noexcept { return translationTrack_; }
    visualanimation::TrackId rotationTrack() const noexcept { return rotationTrack_; }
    visualanimation::TrackId scaleTrack() const noexcept { return scaleTrack_; }
    bool hasTranslation() const noexcept { return translationTrack_.isValid(); }
    bool hasRotation() const noexcept { return rotationTrack_.isValid(); }
    bool hasScale() const noexcept { return scaleTrack_.isValid(); }
    const std::array<float, 3>& translation() const noexcept { return translation_; }
    const std::array<float, 4>& rotation() const noexcept { return rotation_; }
    const std::array<float, 3>& scale() const noexcept { return scale_; }

private:
    friend class AnimationDeformationEvaluationBuilder;

    visualanimation::TargetId animationTarget_;
    SkinId skin_;
    JointId joint_;
    visualanimation::TrackId translationTrack_;
    visualanimation::TrackId rotationTrack_;
    visualanimation::TrackId scaleTrack_;
    std::array<float, 3> translation_ { 0.0f, 0.0f, 0.0f };
    std::array<float, 4> rotation_ { 0.0f, 0.0f, 0.0f, 1.0f };
    std::array<float, 3> scale_ { 1.0f, 1.0f, 1.0f };
};

class MorphWeightEvaluation
{
public:
    visualanimation::TargetId animationTarget() const noexcept { return animationTarget_; }
    visualanimation::TrackId track() const noexcept { return track_; }
    MeshId mesh() const noexcept { return sample_->mesh(); }
    const std::vector<MorphTargetId>& targets() const noexcept { return sample_->targets(); }
    const std::vector<float>& values() const noexcept { return sample_->values(); }

private:
    friend class AnimationDeformationEvaluationBuilder;

    visualanimation::TargetId animationTarget_;
    visualanimation::TrackId track_;
    std::shared_ptr<const MorphWeightSample> sample_;
};

// Immutable backend input only. It contains sampled joint channels and morph
// weights. A false has*() flag means the channel is absent, so its neutral
// storage value must not replace an imported base-pose component. This contract
// does not compute skin palettes, deform vertices, or provide a CPU fallback.
class AnimationDeformationSnapshot
{
public:
    visualanimation::ClipId clip() const noexcept { return clip_; }
    std::uint64_t revision() const noexcept { return revision_; }
    const RationalFrameTime& time() const noexcept { return time_; }
    double requestedTimeSeconds() const noexcept { return requestedTimeSeconds_; }
    double sampleTimeSeconds() const noexcept { return sampleTimeSeconds_; }
    visualanimation::Playback playback() const noexcept { return playback_; }
    visualanimation::CombinationMode combinationMode() const noexcept
    {
        return combinationMode_;
    }
    double combinationWeight() const noexcept { return combinationWeight_; }
    const visualanimation::PoseControls& pose() const noexcept { return pose_; }
    const std::vector<JointTransformEvaluation>& jointTransforms() const noexcept
    {
        return jointTransforms_;
    }
    const std::vector<MorphWeightEvaluation>& morphWeights() const noexcept
    {
        return morphWeights_;
    }
    const std::array<std::uint8_t, 32>& sourceContentSha256() const noexcept
    {
        return sourceContentSha256_;
    }

private:
    friend class AnimationDeformationEvaluationBuilder;

    visualanimation::ClipId clip_;
    std::uint64_t revision_ = 0;
    RationalFrameTime time_;
    double requestedTimeSeconds_ = 0.0;
    double sampleTimeSeconds_ = 0.0;
    visualanimation::Playback playback_ = visualanimation::Playback::Clamp;
    visualanimation::CombinationMode combinationMode_ =
        visualanimation::CombinationMode::Replace;
    double combinationWeight_ = 1.0;
    visualanimation::PoseControls pose_;
    std::vector<JointTransformEvaluation> jointTransforms_;
    std::vector<MorphWeightEvaluation> morphWeights_;
    std::array<std::uint8_t, 32> sourceContentSha256_ {};
};

std::shared_ptr<const AnimationDeformationSnapshot> evaluateAnimationDeformation (
    const DeformationAsset& asset,
    const visualanimation::Clip& clip,
    const AnimationDeformationBindingView& bindings,
    const AnimationDeformationRequest& request,
    const AnimationDeformationLimits& limits,
    std::string& error);
} // namespace visualdeformation
