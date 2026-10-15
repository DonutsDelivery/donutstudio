#pragma once

#include "../../../shared/Visual3DScene.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace arbitgpu::fixturetexture
{
using HarmonicMIDI::grid::SceneTexelRgba8;
using HarmonicMIDI::grid::Visual3DScene;
static_assert (sizeof (SceneTexelRgba8) == 4, "fixture mip accounting requires RGBA8 texels");

inline constexpr std::size_t kMaximumSourceTexels = Visual3DScene::kMaxTextureTexels;
inline constexpr std::size_t kMaximumMipLevels = 16;
// Retained RGBA8 texel payload, including every mip, role copy and fallback.
// Native driver bookkeeping and alignment are not observable through this seam.
inline constexpr std::size_t kMaximumTextureBytes = 16u * 1024u * 1024u;

struct Minification
{
    bool linear = false;
    bool mipmapped = false;
    bool linearMipmap = false;
};

inline bool minification (std::uint32_t filter, Minification& result) noexcept
{
    result = {};
    switch (filter)
    {
        case 9728: result = { false, false, false }; return true;
        case 9729: result = { true,  false, false }; return true;
        case 9984: result = { false, true,  false }; return true;
        case 9985: result = { true,  true,  false }; return true;
        case 9986: result = { false, true,  true  }; return true;
        case 9987: result = { true,  true,  true  }; return true;
        default: return false;
    }
}

struct MipLevel
{
    std::uint32_t width = 0, height = 0;
    std::size_t offsetBytes = 0, bytes = 0;
};

struct MipLayout
{
    std::array<MipLevel, kMaximumMipLevels> levels {};
    std::size_t levelCount = 0, bytes = 0;
};

inline bool mipLayout (std::uint32_t width, std::uint32_t height,
                       std::uint32_t filter, std::uint32_t maximumExtent,
                       MipLayout& result) noexcept
{
    result = {};
    Minification sampling;
    if (! minification (filter, sampling) || width == 0 || height == 0
        || width > maximumExtent || height > maximumExtent
        || width > static_cast<std::uint32_t> (std::numeric_limits<int>::max())
        || height > static_cast<std::uint32_t> (std::numeric_limits<int>::max())
        || static_cast<std::uint64_t> (width) * height > kMaximumSourceTexels)
        return false;

    MipLayout candidate;
    for (;;)
    {
        if (candidate.levelCount == candidate.levels.size()) return false;
        const auto bytes = static_cast<std::size_t> (width) * height * sizeof (SceneTexelRgba8);
        if (bytes > kMaximumTextureBytes - candidate.bytes) return false;
        candidate.levels[candidate.levelCount++] = { width, height, candidate.bytes, bytes };
        candidate.bytes += bytes;
        if (! sampling.mipmapped || (width == 1 && height == 1)) break;
        width = std::max (1u, width / 2);
        height = std::max (1u, height / 2);
    }
    result = candidate;
    return true;
}

inline bool accountBytes (const MipLayout& layout, std::size_t copies,
                          std::size_t budget, std::size_t& total) noexcept
{
    if (layout.levelCount == 0 || layout.levelCount > kMaximumMipLevels
        || layout.bytes == 0 || copies == 0 || copies > 2
        || budget == 0 || budget > kMaximumTextureBytes || total > budget
        || layout.bytes > (budget - total) / copies)
        return false;
    total += layout.bytes * copies;
    return true;
}

using SceneMipLayouts = std::array<MipLayout, Visual3DScene::kMaxTextures>;

// Run before any fixture GPU allocation. Aliased image ranges still consume a
// separate upload per texture record and are charged for each retained copy.
inline bool sceneMipLayouts (const Visual3DScene& scene, std::size_t copies,
                             std::uint32_t maximumExtent, std::size_t budget,
                             std::size_t& textureBytes, SceneMipLayouts& result) noexcept
{
    result = {};
    if (scene.textureCount > scene.textures.size()
        || scene.textureTexelCount > scene.textureTexels.size()
        || scene.textureTexelCount > kMaximumSourceTexels
        || maximumExtent == 0 || copies == 0 || copies > 2
        || budget == 0 || budget > kMaximumTextureBytes || textureBytes > budget)
        return false;
    SceneMipLayouts candidate;
    auto total = textureBytes;
    const auto validWrap = [] (std::uint32_t value)
    {
        return value == 10497 || value == 33071 || value == 33648;
    };
    for (std::size_t index = 0; index < scene.textureCount; ++index)
    {
        const auto& source = scene.textures[index];
        if (! mipLayout (source.width, source.height, source.minFilter,
                         maximumExtent, candidate[index])
            || (source.magFilter != 9728 && source.magFilter != 9729)
            || ! validWrap (source.wrapS) || ! validWrap (source.wrapT)
            || source.firstTexel > scene.textureTexelCount
            || candidate[index].levels[0].bytes / sizeof (SceneTexelRgba8)
                > scene.textureTexelCount - source.firstTexel
            || ! accountBytes (candidate[index], copies, budget, total))
            return false;
    }
    result = candidate;
    textureBytes = total;
    return true;
}

// Immutable Sokol images need every level at creation. Keep only one temporary
// chain while uploading. RGB color roles filter in linear light; alpha and data
// roles remain linear values. Level zero is copied without changing any byte.
inline bool mipTexels (const SceneTexelRgba8* source, std::size_t sourceCount,
                       const MipLayout& layout, bool srgb,
                       std::vector<SceneTexelRgba8>& result)
{
    MipLayout checked;
    if (source == nullptr || ! mipLayout (layout.levels[0].width, layout.levels[0].height,
            layout.levelCount > 1 ? 9987 : 9728,
            static_cast<std::uint32_t> (std::numeric_limits<int>::max()), checked)
        || layout.levelCount != checked.levelCount || layout.bytes != checked.bytes
        || sourceCount < checked.levels[0].bytes / sizeof (SceneTexelRgba8))
        return false;
    for (std::size_t level = 0; level < checked.levelCount; ++level)
        if (layout.levels[level].width != checked.levels[level].width
            || layout.levels[level].height != checked.levels[level].height
            || layout.levels[level].offsetBytes != checked.levels[level].offsetBytes
            || layout.levels[level].bytes != checked.levels[level].bytes)
            return false;

    std::vector<SceneTexelRgba8> texels;
    try { texels.resize (checked.bytes / sizeof (SceneTexelRgba8)); }
    catch (const std::bad_alloc&) { return false; }
    catch (const std::length_error&) { return false; }
    std::copy_n (source, checked.levels[0].bytes / sizeof (SceneTexelRgba8), texels.data());
    std::array<double, 256> decoded {};
    for (std::size_t value = 0; value < decoded.size(); ++value)
    {
        const double normalized = static_cast<double> (value) / 255.0;
        decoded[value] = ! srgb ? normalized : (normalized <= 0.04045
            ? normalized / 12.92 : std::pow ((normalized + 0.055) / 1.055, 2.4));
    }
    const auto encode = [srgb] (double value, bool color)
    {
        if (srgb && color)
            value = value <= 0.0031308 ? value * 12.92
                : 1.055 * std::pow (value, 1.0 / 2.4) - 0.055;
        return static_cast<std::uint8_t> (std::clamp (value * 255.0 + 0.5, 0.0, 255.0));
    };
    for (std::size_t level = 1; level < checked.levelCount; ++level)
    {
        const auto& previous = checked.levels[level - 1];
        const auto& current = checked.levels[level];
        const auto* input = texels.data() + previous.offsetBytes / sizeof (SceneTexelRgba8);
        auto* output = texels.data() + current.offsetBytes / sizeof (SceneTexelRgba8);
        for (std::uint32_t y = 0; y < current.height; ++y)
        for (std::uint32_t x = 0; x < current.width; ++x)
        {
            // Area weights include the last row/column of odd-sized images.
            const double x0 = static_cast<double> (x) * previous.width / current.width;
            const double x1 = static_cast<double> (x + 1) * previous.width / current.width;
            const double y0 = static_cast<double> (y) * previous.height / current.height;
            const double y1 = static_cast<double> (y + 1) * previous.height / current.height;
            std::array<double, 4> sum {};
            const auto endX = std::min (previous.width, static_cast<std::uint32_t> (std::ceil (x1)));
            const auto endY = std::min (previous.height, static_cast<std::uint32_t> (std::ceil (y1)));
            for (auto sy = static_cast<std::uint32_t> (y0); sy < endY; ++sy)
            for (auto sx = static_cast<std::uint32_t> (x0); sx < endX; ++sx)
            {
                const auto weight = (std::min (x1, sx + 1.0) - std::max (x0, double (sx)))
                    * (std::min (y1, sy + 1.0) - std::max (y0, double (sy)));
                const auto& texel = input[static_cast<std::size_t> (sy) * previous.width + sx];
                sum[0] += decoded[texel.red] * weight;
                sum[1] += decoded[texel.green] * weight;
                sum[2] += decoded[texel.blue] * weight;
                sum[3] += (texel.alpha / 255.0) * weight;
            }
            const auto area = (x1 - x0) * (y1 - y0);
            output[static_cast<std::size_t> (y) * current.width + x] = {
                encode (sum[0] / area, true), encode (sum[1] / area, true),
                encode (sum[2] / area, true), encode (sum[3] / area, false)
            };
        }
    }
    result = std::move (texels);
    return true;
}
} // namespace arbitgpu::fixturetexture
