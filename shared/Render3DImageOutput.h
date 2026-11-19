#pragma once

#include "RenderPassOutputContract.h"
#include "RenderPassCompositeContract.h"
#include <array>
#include <cstdint>

namespace render3dimage
{
// Persisted Render 3D export selections use the existing output enum bit positions.
inline constexpr std::uint32_t kRawExportMask = 255u;
inline constexpr std::array<const char*, 8> kRawExportParameters {
    "exportImage", "exportDepth", "exportNormal", "exportMotion", "exportEmission",
    "exportMask", "exportMaterialId", "exportObjectId" };
// A graph node's stable id is its value plus one, so zero stays unset. Raw
// passes key their frames by this id, so every producer must use this rule.
inline constexpr std::uint64_t renderStableIdForNodeValue(std::uint64_t nodeValue) noexcept
{
    return nodeValue + 1u;
}
inline constexpr bool exports(std::uint32_t mask, renderpassoutput::Output output) noexcept
{
    return static_cast<unsigned>(output) < 8u
        && (mask & (1u << static_cast<unsigned>(output))) != 0;
}
// These are display images on the existing Image port, not raw typed AOVs.
// Normal maps [-1, 1] to [0, 1]; depth uses the camera's near/far range.
// Motion maps signed pixel displacement / image extent around neutral gray.
// IDs use a stable palette. The palette does not encode a recoverable ID.
inline constexpr bool supported(renderpassoutput::Output output) noexcept
{
    return output == renderpassoutput::Output::Color
        || output == renderpassoutput::Output::Depth
        || output == renderpassoutput::Output::Normal
        || output == renderpassoutput::Output::Motion
        || output == renderpassoutput::Output::Emission
        || output == renderpassoutput::Output::Mask
        || output == renderpassoutput::Output::MaterialId
        || output == renderpassoutput::Output::ObjectId;
}

inline std::array<float, 3> identityColor(std::uint32_t id) noexcept
{
    auto hash = id * 0x9e3779b9u;
    hash ^= hash >> 16u;
    return { static_cast<float>(32u + (hash & 191u)) / 255.0f,
             static_cast<float>(32u + ((hash >> 8u) & 191u)) / 255.0f,
             static_cast<float>(32u + ((hash >> 16u) & 191u)) / 255.0f };
}
} // namespace render3dimage
