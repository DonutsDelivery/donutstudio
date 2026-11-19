#pragma once

#include "Visual3DScene.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace HarmonicMIDI::grid::visualimportedsceneidentity
{
inline constexpr std::uint64_t kObjectPrimitiveStride = 65536;
inline constexpr std::size_t kNodeTransformPrimitiveIndex
    = static_cast<std::size_t> (kObjectPrimitiveStride - 1);

inline std::optional<Scene3DId> sceneId (std::size_t selectedScene) noexcept
{
    if (selectedScene >= std::numeric_limits<std::uint32_t>::max())
        return std::nullopt;
    return Scene3DId { static_cast<std::uint32_t> (selectedScene + 1) };
}

inline std::optional<SceneObjectId> objectId (std::size_t nodeIndex,
                                              std::size_t primitiveIndex) noexcept
{
    if (primitiveIndex >= kObjectPrimitiveStride)
        return std::nullopt;
    constexpr auto maximumIdentity = std::numeric_limits<std::uint32_t>::max();
    if (nodeIndex > (maximumIdentity - 1) / kObjectPrimitiveStride)
        return std::nullopt;
    const auto identity = static_cast<std::uint64_t> (nodeIndex)
                        * kObjectPrimitiveStride
                        + static_cast<std::uint64_t> (primitiveIndex) + 1;
    if (identity > maximumIdentity)
        return std::nullopt;
    return SceneObjectId { static_cast<std::uint32_t> (identity) };
}

inline std::optional<SceneObjectId> nodeTransformObjectId (std::size_t nodeIndex) noexcept
{
    return objectId (nodeIndex, kNodeTransformPrimitiveIndex);
}
} // namespace HarmonicMIDI::grid::visualimportedsceneidentity
