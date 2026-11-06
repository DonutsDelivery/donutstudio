#pragma once

// The Render node's Render Mode on the Path Trace and Realtime engines.
//
// Accurate, the default, draws every preview and export with the node's full
// settings, so a settled Accurate preview is the export frame. Fast trades
// quality for speed in the viewport and in export alike:
//   Path Trace: at most kFastPreviewBounces bounces and kFastPreviewVolumeBounces
//     volume bounces. Previews play a quarter of the playback samples per frame
//     and denoise only at power-of-two sample counts and once settled. Export
//     renders the preview sample count, so a settled Fast preview is the Fast
//     export frame's sample count.
//   Realtime: no screen-space reflection pass, the costliest of its phases.
// The viewport header names the mode and the bounce limit in force.

#include <algorithm>
#include <cstdint>

namespace videorender::fixture3d
{
inline constexpr std::uint32_t kFastPreviewBounces = 4;
inline constexpr std::uint32_t kFastPreviewVolumeBounces = 1;
inline constexpr std::uint32_t kFastPlaybackSampleDivisor = 4;

struct PathTracePreviewCost
{
    std::uint32_t maxBounces = 0;
    std::uint32_t maxVolumeBounces = 0;
    std::uint32_t playbackSamples = 1;
};

// The bounce limits and playback budget a draw traces with. fast is true for
// every Fast draw, preview or export; export ignores the playback budget.
inline PathTracePreviewCost pathTracePreviewCost (bool fast, std::uint32_t maxBounces, std::uint32_t maxVolumeBounces,
                                                  std::uint32_t playbackSamples) noexcept
{
    PathTracePreviewCost cost { maxBounces, maxVolumeBounces, std::max (1u, playbackSamples) };
    if (! fast)
        return cost;
    cost.maxBounces = std::min (maxBounces, kFastPreviewBounces);
    cost.maxVolumeBounces = std::min (maxVolumeBounces, kFastPreviewVolumeBounces);
    cost.playbackSamples = std::max (1u, playbackSamples / kFastPlaybackSampleDivisor);
    return cost;
}

// The samples per pixel an export renders: the node's export samples in
// Accurate, the preview samples in Fast (never more than export asks for).
inline std::uint32_t exportSamples (bool fast, std::uint32_t nodeExportSamples, std::uint32_t previewSamples) noexcept
{
    return std::max (1u, fast ? std::min (nodeExportSamples, previewSamples) : nodeExportSamples);
}

// Whether a draw denoises its new image. A Fast Preview that is still
// converging denoises at 1, 2, 4, 8... samples per pixel and shows that image
// in between; every other draw denoises each new image.
inline bool denoiseThisImage (bool fast, bool converging, std::uint32_t samplesPerPixel) noexcept
{
    if (! fast || ! converging)
        return true;
    return samplesPerPixel != 0 && (samplesPerPixel & (samplesPerPixel - 1)) == 0;
}

// Whether a Realtime draw runs its screen-space reflection pass.
inline bool realtimeReflections (bool fast) noexcept { return ! fast; }
} // namespace videorender::fixture3d
