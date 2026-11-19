#pragma once

#include "VisualImportedAnimationDeformationRequest.h"
#include "VisualModelAssetContract.h"
#include "VisualModelAssetPayload.h"

#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>

namespace visualanimationoperation
{
inline constexpr std::string_view kCompiledOperationHeader =
    "visual.imported-animation.deformation.v1";
inline constexpr std::string_view kCompiledOperationHeaderV2 =
    "visual.imported-animation.deformation.v2";
inline constexpr std::string_view kCompiledOperationHeaderV3 =
    "visual.imported-animation.deformation.v3";
inline constexpr std::string_view kCompiledOperationHeaderV4 =
    "visual.imported-animation.deformation.v4";
inline constexpr std::string_view kCompiledOperationHeaderV5 =
    "visual.imported-animation.deformation.v5";
inline constexpr std::string_view kCompiledOperationHeaderV6 =
    "visual.imported-animation.deformation.v6";
inline constexpr std::string_view kCompiledOperationHeaderV7 =
    "visual.imported-animation.deformation.v7";
inline constexpr std::string_view kCompiledOperationHeaderV8 =
    "visual.imported-animation.deformation.v8";
// v9 is v8 plus the selected mesh's per-key morph weight deltas after the crossfade line.
inline constexpr std::string_view kCompiledOperationHeaderV9 =
    "visual.imported-animation.deformation.v9";
inline constexpr std::string_view kSourceNodeKind =
    "visual.3d.imported-animation";
inline constexpr std::string_view kDeformationNodeKind =
    "visual.deformation.imported";
inline constexpr std::string_view kSourceBackendCapability = "source-decode";
inline constexpr std::string_view kDeformationBackendCapability = "native-gpu";
// The payload holds as many morph deltas as the pose carries; this bound only
// keeps a size_t length from overflowing.
inline constexpr std::size_t kMaximumEncodedBytes = std::size_t { 1 } << 62;
inline constexpr std::size_t kMaximumClipNameBytes = 256;

inline bool valid(const visualanimationimport::Request& request) noexcept
{
    const auto fitsSigned = [](std::uint64_t value)
    {
        return value <= static_cast<std::uint64_t>(
            std::numeric_limits<std::int64_t>::max());
    };
    const auto finite = [](double value) { return std::isfinite(value); };
    if (request.sourceStableId == 0 || request.deformationStableId == 0
        || ! fitsSigned(request.sourceStableId)
        || ! fitsSigned(request.deformationStableId)
        || request.schedule != std::array<visualanimationimport::StableId, 2> {
               request.sourceStableId, request.deformationStableId }
        || ! visualmodelasset::validStableAssetId(request.asset.id)
        || request.asset.version == 0 || ! fitsSigned(request.asset.version)
        || ! visualmodelasset::validSha256(request.asset.contentSha256)
        || request.asset.sourceMediaType != "model/gltf-binary"
        || request.asset.sourceByteSize == 0
        || request.asset.sourceByteSize > visualmodelassetpayload::kMaxPayloadBytes
        || (request.animationClipStableId != 0
            && ! fitsSigned(request.animationClipStableId))
        || ! fitsSigned(request.crossfadeClipStableId)
        || (request.crossfadeClipStableId == 0 ? !request.crossfadeClipName.empty()
            : request.crossfadeClipName.size() > kMaximumClipNameBytes)
        || ! finite(request.crossfadeProgress) || request.crossfadeProgress < 0.0
        || request.crossfadeProgress > 1.0
        || (request.crossfadeClipStableId == 0 && request.crossfadeProgress != 0.0)
        || (request.meshStableId != 0 && ! fitsSigned(request.meshStableId))
        || (request.clipName.empty() ? request.animationClipStableId == 0
                                     : request.clipName.size() > kMaximumClipNameBytes)
        || (request.combinationMode != visualanimation::CombinationMode::Replace
            && request.combinationMode != visualanimation::CombinationMode::WeightedBlend
            && request.combinationMode != visualanimation::CombinationMode::Add
            && request.combinationMode != visualanimation::CombinationMode::Multiply
            && request.combinationMode
                != visualanimation::CombinationMode::AddAfterImportedAnimation))
        return false;
    if (request.sceneIndex
        && *request.sceneIndex > static_cast<std::size_t>(
               std::numeric_limits<std::int64_t>::max()))
        return false;
    for (const auto* name : { &request.clipName, &request.crossfadeClipName })
        for (const auto character : *name)
            if (static_cast<unsigned char>(character) < 0x20u || character == 0x7f)
                return false;

    const auto& playback = request.playback;
    if ((playback.timeSource != visualanimation::TimeSource::Timeline
         && playback.timeSource != visualanimation::TimeSource::BeatSync)
        || (playback.playback != visualanimation::Playback::Clamp
            && playback.playback != visualanimation::Playback::Loop)
        || ! finite(playback.timelineSeconds)
        || ! finite(playback.clipStartTimelineSeconds)
        || ! finite(playback.timelineBeat) || ! finite(playback.clipStartBeat)
        || ! finite(playback.beatsPerLoop) || playback.beatsPerLoop <= 0.0
        || (playback.timeSource == visualanimation::TimeSource::BeatSync
            && (std::abs(playback.timelineBeat) > visualanimation::PlaybackControlLimits{}.maxAbsoluteBeat
                || std::abs(playback.clipStartBeat) > visualanimation::PlaybackControlLimits{}.maxAbsoluteBeat
                || playback.beatsPerLoop > visualanimation::PlaybackControlLimits{}.maxBeatsPerLoop))
        || ! finite(playback.speed) || playback.speed < -64.0 || playback.speed > 64.0
        || ! finite(playback.offsetSeconds)
        || ! finite(playback.trimStartSeconds) || playback.trimStartSeconds < 0.0
        || playback.trimStartSeconds > 86400.0
        || ! finite(playback.trimEndSeconds) || playback.trimEndSeconds < 0.0
        || playback.trimEndSeconds > 86400.0
        || (playback.trimEndSeconds != 0.0
            && playback.trimEndSeconds <= playback.trimStartSeconds)
        || ! finite(playback.weight) || playback.weight < 0.0 || playback.weight > 1.0)
        return false;
    // Per-key deltas index the selected mesh, so they bind to it like the morph selector.
    const bool bindsMesh = request.pose.boneEnabled || request.pose.morphEnabled
        || !request.pose.morphWeightDeltas.empty();
    return visualanimation::validPoseControls(request.pose)
        && (!bindsMesh
            || request.pose.meshStableId == request.meshStableId
            || (request.meshStableId == 0 && request.pose.nodeStableId != 0));
}

inline std::string encode(const visualanimationimport::Request& request)
{
    if (! valid(request)) return {};
    std::ostringstream encoded;
    const bool writesCombination =
        request.combinationMode != visualanimation::CombinationMode::Replace;
    const bool writesTrim = request.playback.trimStartSeconds != 0.0
        || request.playback.trimEndSeconds != 0.0;
    const bool writesPose = request.pose.boneEnabled || request.pose.morphEnabled;
    const auto header = !request.pose.morphWeightDeltas.empty() ? kCompiledOperationHeaderV9
        : request.crossfadeClipStableId != 0 ? kCompiledOperationHeaderV8
        : request.pose.nodeStableId != 0 ? kCompiledOperationHeaderV7
        : writesPose ? kCompiledOperationHeaderV6
        : writesTrim ? kCompiledOperationHeaderV5
        : request.meshStableId != 0 ? kCompiledOperationHeaderV4
        : writesCombination ? kCompiledOperationHeaderV3
        : request.animationClipStableId != 0 ? kCompiledOperationHeaderV2
                                              : kCompiledOperationHeader;
    encoded << header << '\n'
            << request.sourceStableId << ' ' << request.deformationStableId << '\n'
            << std::quoted(request.asset.id) << '\n'
            << request.asset.version << '\n'
            << std::quoted(request.asset.contentSha256) << '\n'
            << std::quoted(request.asset.sourceMediaType) << '\n'
            << request.asset.sourceByteSize << '\n';
    if (request.sceneIndex)
        encoded << 1 << ' ' << *request.sceneIndex << '\n';
    else
        encoded << 0 << ' ' << 0 << '\n';
    if (header == kCompiledOperationHeaderV2 || header == kCompiledOperationHeaderV3
        || header == kCompiledOperationHeaderV4 || header == kCompiledOperationHeaderV5
        || header == kCompiledOperationHeaderV6 || header == kCompiledOperationHeaderV7
        || header == kCompiledOperationHeaderV8 || header == kCompiledOperationHeaderV9)
        encoded << request.animationClipStableId << '\n';
    if (header == kCompiledOperationHeaderV4 || header == kCompiledOperationHeaderV5
        || header == kCompiledOperationHeaderV6 || header == kCompiledOperationHeaderV7
        || header == kCompiledOperationHeaderV8 || header == kCompiledOperationHeaderV9)
        encoded << request.meshStableId << '\n';
    encoded << std::quoted(request.clipName) << '\n';
    if (header == kCompiledOperationHeaderV3 || header == kCompiledOperationHeaderV4
        || header == kCompiledOperationHeaderV5 || header == kCompiledOperationHeaderV6
        || header == kCompiledOperationHeaderV7 || header == kCompiledOperationHeaderV8 || header == kCompiledOperationHeaderV9)
        encoded << static_cast<int>(request.combinationMode) << '\n';
    encoded << static_cast<int>(request.playback.timeSource) << ' '
            << static_cast<int>(request.playback.playback) << '\n'
            << std::setprecision(std::numeric_limits<double>::max_digits10)
            << request.playback.timelineSeconds << ' '
            << request.playback.clipStartTimelineSeconds << ' '
            << request.playback.timelineBeat << ' '
            << request.playback.clipStartBeat << ' '
            << request.playback.beatsPerLoop << ' '
            << request.playback.speed << ' '
            << request.playback.offsetSeconds << ' '
            << request.playback.weight << '\n';
    if (header == kCompiledOperationHeaderV5 || header == kCompiledOperationHeaderV6
        || header == kCompiledOperationHeaderV7 || header == kCompiledOperationHeaderV8 || header == kCompiledOperationHeaderV9)
        encoded << request.playback.trimStartSeconds << ' '
                << request.playback.trimEndSeconds << '\n';
    if (header == kCompiledOperationHeaderV6 || header == kCompiledOperationHeaderV7
        || header == kCompiledOperationHeaderV8 || header == kCompiledOperationHeaderV9)
    {
        const auto& pose = request.pose;
        encoded << pose.meshStableId << ' ' << pose.boneStableId << ' '
                << pose.morphTargetStableId << ' ' << pose.morphTargetIndex << ' ' << pose.boneEnabled << ' '
                << pose.morphEnabled << ' ' << pose.morphWeight << '\n';
        for (const auto& values : { pose.translation, pose.rotationDegrees, pose.scale })
            encoded << values[0] << ' ' << values[1] << ' ' << values[2] << '\n';
        if (header == kCompiledOperationHeaderV7 || header == kCompiledOperationHeaderV8 || header == kCompiledOperationHeaderV9) encoded << pose.nodeStableId << '\n';
    }
    if (header == kCompiledOperationHeaderV8 || header == kCompiledOperationHeaderV9)
        encoded << request.crossfadeClipStableId << ' ' << std::quoted(request.crossfadeClipName)
                << ' ' << request.crossfadeProgress << '\n';
    if (header == kCompiledOperationHeaderV9)
    {
        encoded << request.pose.morphWeightDeltas.size();
        for (const auto& [index, weight] : request.pose.morphWeightDeltas)
            encoded << ' ' << index << ' ' << weight;
        encoded << '\n';
    }
    auto result = encoded.str();
    return result.size() <= kMaximumEncodedBytes ? result : std::string {};
}

inline bool decode(std::string_view bytes,
                   visualanimationimport::Request& destination,
                   std::string& error)
{
    error.clear();
    if (bytes.empty() || bytes.size() > kMaximumEncodedBytes)
    {
        error = "imported animation operation payload is malformed or out of bounds";
        return false;
    }
    std::istringstream encoded { std::string(bytes) };
    std::string header;
    if (! std::getline(encoded, header)
        || (header != kCompiledOperationHeader && header != kCompiledOperationHeaderV2
            && header != kCompiledOperationHeaderV3
            && header != kCompiledOperationHeaderV4
            && header != kCompiledOperationHeaderV5
            && header != kCompiledOperationHeaderV6 && header != kCompiledOperationHeaderV7
            && header != kCompiledOperationHeaderV8 && header != kCompiledOperationHeaderV9))
    {
        error = "imported animation operation payload is malformed or out of bounds";
        return false;
    }

    visualanimationimport::Request candidate;
    int hasScene = 0;
    std::size_t sceneIndex = 0;
    int timeSource = -1;
    int playback = -1;
    int combinationMode = static_cast<int>(visualanimation::CombinationMode::Replace);
    if (! (encoded >> candidate.sourceStableId >> candidate.deformationStableId
           >> std::quoted(candidate.asset.id)
           >> candidate.asset.version
           >> std::quoted(candidate.asset.contentSha256)
           >> std::quoted(candidate.asset.sourceMediaType)
           >> candidate.asset.sourceByteSize
           >> hasScene >> sceneIndex)
        || ((header == kCompiledOperationHeaderV2
             || header == kCompiledOperationHeaderV3
             || header == kCompiledOperationHeaderV4
             || header == kCompiledOperationHeaderV5 || header == kCompiledOperationHeaderV6
             || header == kCompiledOperationHeaderV7 || header == kCompiledOperationHeaderV8 || header == kCompiledOperationHeaderV9)
            && ! (encoded >> candidate.animationClipStableId))
        || ((header == kCompiledOperationHeaderV4
             || header == kCompiledOperationHeaderV5 || header == kCompiledOperationHeaderV6
             || header == kCompiledOperationHeaderV7 || header == kCompiledOperationHeaderV8 || header == kCompiledOperationHeaderV9)
            && ! (encoded >> candidate.meshStableId))
        || ! (encoded >> std::quoted(candidate.clipName)
              )
        || ((header == kCompiledOperationHeaderV3
             || header == kCompiledOperationHeaderV4
             || header == kCompiledOperationHeaderV5 || header == kCompiledOperationHeaderV6
             || header == kCompiledOperationHeaderV7 || header == kCompiledOperationHeaderV8 || header == kCompiledOperationHeaderV9)
            && ! (encoded >> combinationMode))
        || ! (encoded >> timeSource >> playback
              >> candidate.playback.timelineSeconds
              >> candidate.playback.clipStartTimelineSeconds
              >> candidate.playback.timelineBeat
              >> candidate.playback.clipStartBeat
              >> candidate.playback.beatsPerLoop
              >> candidate.playback.speed
              >> candidate.playback.offsetSeconds
              >> candidate.playback.weight)
        || ((header == kCompiledOperationHeaderV5 || header == kCompiledOperationHeaderV6
             || header == kCompiledOperationHeaderV7 || header == kCompiledOperationHeaderV8 || header == kCompiledOperationHeaderV9)
            && ! (encoded >> candidate.playback.trimStartSeconds
                          >> candidate.playback.trimEndSeconds)))
    {
        error = "imported animation operation payload is malformed or out of bounds";
        return false;
    }
    if (header == kCompiledOperationHeaderV6 || header == kCompiledOperationHeaderV7
        || header == kCompiledOperationHeaderV8 || header == kCompiledOperationHeaderV9)
    {
        auto& pose = candidate.pose;
        int boneEnabled = 0, morphEnabled = 0;
        if (!(encoded >> pose.meshStableId >> pose.boneStableId >> pose.morphTargetStableId >> pose.morphTargetIndex
              >> boneEnabled >> morphEnabled >> pose.morphWeight
              >> pose.translation[0] >> pose.translation[1] >> pose.translation[2]
              >> pose.rotationDegrees[0] >> pose.rotationDegrees[1] >> pose.rotationDegrees[2]
              >> pose.scale[0] >> pose.scale[1] >> pose.scale[2])
            || (boneEnabled != 0 && boneEnabled != 1)
            || (morphEnabled != 0 && morphEnabled != 1))
        {
            error = "imported animation selected pose controls are malformed";
            return false;
        }
        pose.boneEnabled = boneEnabled != 0;
        pose.morphEnabled = morphEnabled != 0;
        if ((header == kCompiledOperationHeaderV7 || header == kCompiledOperationHeaderV8 || header == kCompiledOperationHeaderV9)
            && !(encoded >> pose.nodeStableId))
        { error = "imported animation pose node identity is malformed"; return false; }
    }
    if ((header == kCompiledOperationHeaderV8 || header == kCompiledOperationHeaderV9)
        && (!(encoded >> candidate.crossfadeClipStableId >> std::quoted(candidate.crossfadeClipName)
                     >> candidate.crossfadeProgress)
            || (header == kCompiledOperationHeaderV8 && candidate.crossfadeClipStableId == 0)))
    { error = "imported animation crossfade selection is malformed"; return false; }
    if (header == kCompiledOperationHeaderV9)
    {
        std::size_t overrideCount = 0;
        if (!(encoded >> overrideCount)
            || overrideCount > visualanimation::kMaximumMorphWeightDeltas
            || overrideCount > bytes.size())
        {
            error = "imported animation selected morph weight deltas are malformed";
            return false;
        }
        candidate.pose.morphWeightDeltas.reserve(overrideCount);
        for (std::size_t entry = 0; entry < overrideCount; ++entry)
        {
            std::uint32_t targetIndex = 0;
            double weight = 0.0;
            if (!(encoded >> targetIndex >> weight))
            {
                error = "imported animation selected morph weight deltas are malformed";
                return false;
            }
            candidate.pose.morphWeightDeltas.emplace_back(targetIndex, weight);
        }
    }
    encoded >> std::ws;
    if (! encoded.eof() || (hasScene != 0 && hasScene != 1)
        || timeSource < static_cast<int>(visualanimation::TimeSource::Timeline)
        || timeSource > static_cast<int>(visualanimation::TimeSource::BeatSync)
        || combinationMode < static_cast<int>(visualanimation::CombinationMode::Replace)
        || combinationMode
            > static_cast<int>(
                visualanimation::CombinationMode::AddAfterImportedAnimation)
        || playback < static_cast<int>(visualanimation::Playback::Clamp)
        || playback > static_cast<int>(visualanimation::Playback::Loop))
    {
        error = "imported animation operation payload is malformed or out of bounds";
        return false;
    }

    if (hasScene != 0) candidate.sceneIndex = sceneIndex;
    candidate.schedule = { candidate.sourceStableId, candidate.deformationStableId };
    candidate.combinationMode =
        static_cast<visualanimation::CombinationMode>(combinationMode);
    candidate.playback.timeSource = static_cast<visualanimation::TimeSource>(timeSource);
    candidate.playback.playback = static_cast<visualanimation::Playback>(playback);
    if (! valid(candidate))
    {
        error = "imported animation operation payload is malformed or out of bounds";
        return false;
    }
    destination = std::move(candidate);
    return true;
}
} // namespace visualanimationoperation
