#pragma once

// Realtime's diffraction transport at reduced resolution. A grating or receiver pixel's
// gather over the other side's table rows, and the grating's sky through its orders,
// cost about 1.5 us a pixel on the RTX 4070 once the bands filled the GPU: 3.2 s for
// the orb at 1920x1080. A first pass draws the gratings and receivers into a target of
// at most kGatherPixels and keeps only that term; the full-resolution pass adds the
// term back from the texels of the same object around each pixel whose depth and
// normal match, and gathers itself where none does, as at a silhouette. Direct light,
// shading and edges stay at full resolution.

#include "fixture_diffraction_bands.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace arbitgpu::realtimediffractiongather
{
inline constexpr std::uint64_t kGatherPixels = 640ull * 360ull;

inline std::uint32_t extent (std::uint32_t size, std::uint32_t scale) noexcept
{
    return scale == 0 ? size : (size + scale - 1) / scale;
}

// The integer scale that brings width x height to at most kGatherPixels; 1 keeps the
// gather at full resolution.
inline std::uint32_t scale (std::uint32_t width, std::uint32_t height) noexcept
{
    const double ratio = double (width) * double (height) / double (kGatherPixels);
    if (! (ratio > 1.0)) return 1;
    auto s = static_cast<std::uint32_t> (std::ceil (std::sqrt (ratio)));
    while (std::uint64_t (extent (width, s)) * extent (height, s) > kGatherPixels) ++s;
    return s;
}

// The gather pass's bands: the target's rows, split by the queries a pixel gathers over
// the target's width. Bands of the full frame's rows left the 640-pixel target's bands
// too small to fill the GPU.
inline arbitgpu::NativeFixtureDiffractionBands gatherPassBands (
    arbitgpu::NativeFixtureDiffractionBands unsplit, std::uint64_t pixelQueries, std::uint32_t scale) noexcept
{
    unsplit.width = extent (unsplit.width, scale);
    unsplit.height = extent (unsplit.height, scale);
    unsplit.bandHeight = std::max (1u, std::min (unsplit.bandHeight, unsplit.height));
    unsplit.bandCount = (unsplit.height + unsplit.bandHeight - 1) / unsplit.bandHeight;
    arbitgpu::nativeFixtureDiffractionSplitBandsByQueries (unsplit, pixelQueries);
    return unsplit;
}

// The full pass's bands: there only a pixel with no matching gather texel gathers, so
// a pixel's queries are spread over the scale's square.
inline arbitgpu::NativeFixtureDiffractionBands fullPassBands (
    arbitgpu::NativeFixtureDiffractionBands unsplit, std::uint64_t pixelQueries, std::uint32_t scale) noexcept
{
    const std::uint64_t square = std::uint64_t (std::max (1u, scale)) * std::max (1u, scale);
    arbitgpu::nativeFixtureDiffractionSplitBandsByQueries (unsplit, std::max<std::uint64_t> (1, pixelQueries / square));
    return unsplit;
}

// ARBIT_GL_REALTIME_GATHER_SCALE=N sets the scale for a test run (1 is full resolution).
inline std::uint32_t frameScale (std::uint32_t width, std::uint32_t height) noexcept
{
    if (const char* value = std::getenv ("ARBIT_GL_REALTIME_GATHER_SCALE"))
    {
        const long forced = std::strtol (value, nullptr, 10);
        if (forced >= 1 && forced <= 64) return static_cast<std::uint32_t> (forced);
    }
    return scale (width, height);
}
} // namespace arbitgpu::realtimediffractiongather
