#pragma once

// Equirectangular HDR environment image for the CPU path tracer, sampled in
// proportion to its power.
//
// Lookup follows the raster backends' environmentImageRadiance
// (gpu_backend/fixture_environment_image.h, shader lines 116-129, and the
// uniforms at lines 102-108), so the tracer and the raster read the same texel
// for the same direction:
// - longitude = atan2(d.x, d.z), or 0 when d is within 1e-6 of the Y axis;
// - u = fract((longitude - orientation) / 2pi + 0.5), v = acos(d.y) / pi;
// - texel (i, j) is centred at ((i + 0.5) / W, (j + 0.5) / H), row 0 is the
//   zenith (+Y), and u = 0.5 looks along +Z before rotation;
// - bilinear filtering wraps columns and clamps rows at texel centres;
// - radiance = filtered texel * tint, tint = color * intensity *
//   2^exposureStops, formed in float as the raster uniform is.
// A positive orientation turns the image about +Y from +Z toward +X. The
// decoded pixels are the helper's top-first rows
// (shared/VisualEnvironmentImageAssetContract.h, lines 52-58), the same
// buffer the raster uploads as its texture.
//
// Sampling. Cell (i, j) is the footprint of texel (i, j) in (u, v). Its
// weight is the luminance of the mean filtered radiance over the cell times
// the cell's solid angle. A bilinear reconstruction averaged over one texel
// footprint is a separable [1/8, 3/4, 1/8] blend of that texel and its
// neighbours, so every cell where the filtered radiance is non-zero has a
// non-zero weight. A marginal CDF over rows picks a row, the row's
// conditional CDF picks a column, and the direction is uniform in solid angle
// inside the cell (uniform longitude, uniform cos(polar)). The pdf is
// constant over a cell, P(row) P(column | row) / cell solid angle, so it has
// no 1/sin(polar) pole.
//
// Behavioural reference (studied, not copied): Blender Cycles
// intern/cycles/kernel/light/background.h (marginal and conditional CDF
// sampling of the background map and its pdf). Cycles samples uniformly in
// (u, v) inside a cell and divides its pdf by sin(polar); this map weights
// cells by their exact solid angle and samples uniformly in solid angle.

#include "vec_math.h"
#include "../../../shared/Visual3DScene.h"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace videohelper::pathtrace
{

// Linear RGB texels, row-major, row 0 at the zenith.
using EnvironmentTexels = std::vector<std::array<float, 3>>;

// Decoded pixels of one environment image and the asset they decode. texels
// may share ownership with the helper's resident store (see
// environment_resident.h), so holding this keeps that resident alive.
struct EnvironmentPixels
{
    visualenvironmentimage::Identity asset;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::shared_ptr<const EnvironmentTexels> texels;

    bool valid() const noexcept;
};

// Copies width * height RGB triples (row 0 at the zenith) into new storage.
// Returns empty pixels when rgb is null or a dimension is zero.
EnvironmentPixels copyEnvironmentPixels (std::uint32_t width, std::uint32_t height, const float* rgb,
                                         visualenvironmentimage::Identity asset = {});

// The raster's tint and orientation for an environment image record
// (fixture_environment_image.h uniforms()).
Vec3 environmentImageTint (const HarmonicMIDI::grid::SceneLightRecord& record) noexcept;
float environmentImageOrientationRadians (const HarmonicMIDI::grid::SceneLightRecord& record) noexcept;

class EnvironmentMap
{
public:
    // Returns false and leaves the map inactive when the pixels are missing,
    // the tint or orientation is not finite, the tint is negative, a texel is
    // negative or not finite, or the tinted image is black. Keeps a reference
    // to pixels.texels; nothing is copied.
    bool build (const EnvironmentPixels& pixels, Vec3 tint, float orientationRadians);
    void clear() noexcept;

    bool active() const noexcept { return texels_ != nullptr; }
    std::uint32_t width() const noexcept { return width_; }
    std::uint32_t height() const noexcept { return height_; }
    Vec3 tint() const noexcept { return tint_; }
    float orientationRadians() const noexcept { return orientation_; }

    // Raster texture coordinates of a direction and the direction of (u, v).
    // Both take the orientation into account.
    Vec2 directionToUv (Vec3 direction) const noexcept;
    Vec3 uvToDirection (Vec2 uv) const noexcept;

    // Filtered, tinted radiance arriving from a direction. 0 when inactive.
    Vec3 radiance (Vec3 direction) const noexcept;
    // Solid-angle density of sample() for a direction.
    float pdf (Vec3 direction) const noexcept;
    // A unit direction drawn in proportion to the cell weights. pdfOut is the
    // density of the chosen cell, 0 (with a zero direction) on failure.
    Vec3 sample (Vec2 u, float& pdfOut) const noexcept;

    // Integral of the tinted radiance over the sphere, from the cell means.
    Vec3 integratedRadiance() const noexcept { return integrated_; }
    // Largest untinted texel value per channel.
    Vec3 maximumTexel() const noexcept { return maximumTexel_; }
    // Solid angle of one cell of row `row`.
    float cellSolidAngle (std::uint32_t row) const noexcept;

private:
    Vec3 texel (std::uint32_t x, std::uint32_t y) const noexcept;
    std::uint32_t rowOf (Vec3 unitDirection) const noexcept;
    std::uint32_t columnOf (float u) const noexcept;

    std::shared_ptr<const EnvironmentTexels> texels_;
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    Vec3 tint_ {};
    float orientation_ = 0.0f;
    std::vector<float> marginal_;      // height + 1 entries, 0 to 1
    std::vector<float> conditional_;   // height rows of width + 1 entries, 0 to 1
    std::vector<float> cellSolidAngle_;
    std::vector<double> rowCos_;       // cos(pi * row / height), height + 1 entries
    Vec3 integrated_ {};
    Vec3 maximumTexel_ {};
};

} // namespace videohelper::pathtrace
