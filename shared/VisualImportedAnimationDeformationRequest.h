#pragma once

#include "VisualAnimationPlaybackControl.h"
#include "VisualAnimationPoseControls.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace visualanimationimport
{
using StableId = std::uint64_t;

// Backend-neutral identity for immutable caller-owned model bytes. This is not
// a path or an authority to acquire content: consumers must receive the bytes
// separately and verify both size and fingerprint before decoding them.
struct ExactContentAssetKey final
{
    std::string id;
    std::uint64_t version = 0;
    std::string contentSha256;
    std::string sourceMediaType;
    std::uint64_t sourceByteSize = 0;
};

// Exact request lowered by the graph producer and consumed by the helper. It
// names existing graph identities and controls only; it grants no file access.
struct Request final
{
    StableId sourceStableId = 0;
    StableId deformationStableId = 0;
    std::array<StableId, 2> schedule {};
    ExactContentAssetKey asset;
    std::optional<std::size_t> sceneIndex;
    StableId meshStableId = 0;
    StableId animationClipStableId = 0;
    std::string clipName;
    StableId crossfadeClipStableId = 0;
    std::string crossfadeClipName;
    double crossfadeProgress = 0.0;
    visualanimation::CombinationMode combinationMode =
        visualanimation::CombinationMode::Replace;
    visualanimation::PlaybackControlRequest playback;
    visualanimation::PoseControls pose;
};

inline bool sameAsset (const ExactContentAssetKey& left,
                       const ExactContentAssetKey& right) noexcept
{
    return left.id == right.id
        && left.version == right.version
        && left.contentSha256 == right.contentSha256
        && left.sourceMediaType == right.sourceMediaType
        && left.sourceByteSize == right.sourceByteSize;
}
} // namespace visualanimationimport
