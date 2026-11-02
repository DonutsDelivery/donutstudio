#pragma once

#include "backend.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace arbitgpu
{
// Row bands for one native diffraction frame.
//
// The lobe caps in backend.h and an authored spatial foil budget were sized
// per draw at 1080p. Charged against a whole 3840x2160 frame, they refused the
// Holographic Trading Card at the default 4K canvas although 1080p rendered.
// Each band is one scissored draw per lighting path whose rows fit both the
// per-draw lobe cap and the authored foil budget, so every band carries an
// exact shared diffractivefoil schedule. The frame stays bounded three ways:
//  - at most kMaximumNativeFixtureDiffractionBands bands;
//  - total lobe work within the frame caps below, four times the per-draw caps
//    because 3840x2160 is four times 1920x1080, so a frame admitted at 1080p is
//    admitted at 4K;
//  - total foil slots within diffractivefoil::kMaximumEvaluations.
// The card at 4K needs four 540-row bands, 6.57e9 lobe evaluations of the
// 9.0e9 frame cap and 24.9 M of the 67.1 M foil slots.
// A frame that fits one band keeps the single-draw admission unchanged.
inline constexpr std::uint32_t kMaximumNativeFixtureDiffractionBands = 8;
inline constexpr std::uint64_t kNativeFixtureDiffractionFrameScale = 4;
inline constexpr std::uint64_t kMaximumNativeFixtureDiffractionFrame1DLobeEvaluations =
    kNativeFixtureDiffractionFrameScale * kMaximumNativeFixtureDiffraction1DLobeEvaluations;
inline constexpr std::uint64_t kMaximumNativeFixtureDiffractionFrame2DLobeEvaluations =
    kNativeFixtureDiffractionFrameScale * kMaximumNativeFixtureDiffraction2DLobeEvaluations;

struct NativeFixtureDiffractionBands final
{
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t bandHeight = 0;
    std::uint32_t bandCount = 0;
    // Foil schedule of a full band. Zero maximumEvaluations means no foil.
    std::uint32_t pathCount = 0;
    std::uint32_t maximumEvaluations = 0;
    // Pixel slots of the whole frame across every band and lighting path.
    std::uint32_t requiredEvaluations = 0;
};

// First row of a band in the backend's own raster coordinates. GL counts rows
// from the bottom (glScissor, gl_FragCoord) and Metal from the top; each
// backend uses its own convention for both the scissor and the shader index.
inline std::uint32_t nativeFixtureDiffractionBandOrigin (
    const NativeFixtureDiffractionBands& bands, std::uint32_t band) noexcept
{
    return band * bands.bandHeight;
}

inline std::uint32_t nativeFixtureDiffractionBandRows (
    const NativeFixtureDiffractionBands& bands, std::uint32_t band) noexcept
{
    const auto origin = nativeFixtureDiffractionBandOrigin (bands, band);
    return origin >= bands.height ? 0u : std::min (bands.bandHeight, bands.height - origin);
}

// Exclusive upper bound for the unchanged shader index
// path * (width * rows) + y * width + x, where y is the frame row. Shifting the
// band's foil budget by origin * width makes the comparison equal to a local
// index, path * (width * rows) + (y - origin) * width + x, against that budget.
inline std::uint32_t nativeFixtureDiffractionBandEvaluationLimit (
    const NativeFixtureDiffractionBands& bands, std::uint32_t band) noexcept
{
    return bands.maximumEvaluations
        + nativeFixtureDiffractionBandOrigin (bands, band) * bands.width;
}

// A long submission trips the driver's context switch timeout, so receiver
// transport splits its draw by the queries each row can run. A row is the
// smallest band. This changes how the frame is submitted, never what it renders.
inline constexpr std::uint64_t kNativeDiffractionQueriesPerSubmission = 16'777'216;
inline void nativeFixtureDiffractionSplitBandsByQueries (
    NativeFixtureDiffractionBands& bands, std::uint64_t queriesPerPixel) noexcept
{
    if (bands.width == 0 || bands.height == 0 || bands.bandHeight == 0 || queriesPerPixel == 0)
        return;
    const auto pixels = kNativeDiffractionQueriesPerSubmission / queriesPerPixel;
    const auto rows = std::max<std::uint64_t> (1, pixels / bands.width);
    bands.bandHeight = static_cast<std::uint32_t> (std::min<std::uint64_t> (bands.bandHeight, rows));
    bands.bandCount = (bands.height + bands.bandHeight - 1) / bands.bandHeight;
}

// checkLobes is false for authored spectral-receiver (version 3) lighting.
// That transport admits its own sample/query/lobe frame work in buildFrame.
inline bool nativeFixtureDiffractionBandSchedule (
    const NativeFixtureSurfaceMaterialProgram& material,
    std::uint32_t width, std::uint32_t height, bool checkLobes,
    NativeFixtureDiffractionBands& bands) noexcept
{
    bands = {};
    if (! nativeFixtureDimensionsWithinBounds (width, height))
        return false;
    if (material.kind != NativeFixtureMaterialKind::DiffractionReflective)
    {
        if (checkLobes && ! nativeFixtureDiffractionWorkWithinBudget (material, width, height))
            return false;
        bands = { width, height, height, 1u, 0u, 0u, 0u };
        return true;
    }
    if (material.diffractionPathCount == 0
        || material.diffractionPathCount > diffractionmaterial::kMaximumLightingPaths)
        return false;

    std::uint64_t rows = height;
    if (checkLobes)
    {
        // The estimate is pixels times a per-material constant, so one row
        // gives the exact per-row cost.
        const auto rowLobes = nativeFixtureDiffractionLobeEvaluations (material, width, 1);
        if (rowLobes == std::numeric_limits<std::uint64_t>::max())
            return false;
        const bool crossed = material.diffractionPaths[0].material.secondaryGeometry.w
            == static_cast<float> (diffractionmaterial::GratingLattice::CrossedTwoDimensional);
        const auto drawCap = crossed ? kMaximumNativeFixtureDiffraction2DLobeEvaluations
                                     : kMaximumNativeFixtureDiffraction1DLobeEvaluations;
        const auto frameCap = crossed ? kMaximumNativeFixtureDiffractionFrame2DLobeEvaluations
                                      : kMaximumNativeFixtureDiffractionFrame1DLobeEvaluations;
        if (rowLobes != 0)
        {
            if (rowLobes > drawCap || rowLobes > frameCap / height)
                return false;
            rows = std::min<std::uint64_t> (rows, drawCap / rowLobes);
        }
    }

    const auto foilBudget = material.diffractionFoilMaximumEvaluations;
    const std::uint32_t paths = material.diffractionPathCount;
    std::uint64_t frameSlots = 0;
    if (foilBudget != 0)
    {
        if (foilBudget > diffractivefoil::kMaximumEvaluations)
            return false;
        const auto rowSlots = static_cast<std::uint64_t> (width) * paths;
        frameSlots = rowSlots * height;
        if (rowSlots > foilBudget || frameSlots > diffractivefoil::kMaximumEvaluations)
            return false;
        rows = std::min<std::uint64_t> (rows, foilBudget / rowSlots);
    }
    if (rows == 0)
        return false;
    const auto bandCount = (static_cast<std::uint64_t> (height) + rows - 1) / rows;
    if (bandCount > kMaximumNativeFixtureDiffractionBands)
        return false;

    NativeFixtureDiffractionBands candidate { width, height,
        static_cast<std::uint32_t> (rows), static_cast<std::uint32_t> (bandCount),
        foilBudget != 0 ? paths : 0u, foilBudget, static_cast<std::uint32_t> (frameSlots) };
    if (foilBudget != 0)
    {
        // Every band, including a shorter last band, is an exact shared schedule.
        for (std::uint32_t band = 0; band < candidate.bandCount; ++band)
        {
            diffractivefoil::EvaluationSchedule schedule;
            if (! diffractivefoil::makeEvaluationSchedule (foilBudget, width,
                    nativeFixtureDiffractionBandRows (candidate, band), paths, schedule))
                return false;
        }
    }
    bands = candidate;
    return true;
}
} // namespace arbitgpu
