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
#include "devices/device_interface.h"
#if ! defined(PATHTRACE_DEVICE_METAL)
#include "../../../shared/Visual3DScene.h"
#endif

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace videohelper::pathtrace
{

// Host only: the Metal kernel (device_metal.metal) leaves this out.
#if ! defined(PATHTRACE_DEVICE_METAL)
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
#endif

// The read side of a built map: plain data over borrowed buffers, so the same
// lookup and sampling code runs on a device once the buffers are uploaded. An
// EnvironmentMap owns the buffers its view reads.
class EnvironmentMapView
{
public:
    PT_SHARED_FN bool active() const noexcept { return texels_.data != nullptr; }
    PT_SHARED_FN std::uint32_t width() const noexcept { return width_; }
    PT_SHARED_FN std::uint32_t height() const noexcept { return height_; }
    PT_SHARED_FN Vec3 tint() const noexcept { return tint_; }
    PT_SHARED_FN float orientationRadians() const noexcept { return orientation_; }

    // Raster texture coordinates of a direction and the direction of (u, v).
    // Both take the orientation into account.
    PT_SHARED_ENTRY Vec2 directionToUv (Vec3 direction) const noexcept;
    PT_SHARED_ENTRY Vec3 uvToDirection (Vec2 uv) const noexcept;

    // Filtered, tinted radiance arriving from a direction. 0 when inactive.
    PT_SHARED_ENTRY Vec3 radiance (Vec3 direction) const noexcept;
    // Solid-angle density of sample() for a direction.
    PT_SHARED_ENTRY float pdf (Vec3 direction) const noexcept;
    // A unit direction drawn in proportion to the cell weights. pdfOut is the
    // density of the chosen cell, 0 (with a zero direction) on failure.
    PT_SHARED_ENTRY Vec3 sample (Vec2 u, float& pdfOut) const noexcept;

    // Integral of the tinted radiance over the sphere, from the cell means.
    PT_SHARED_FN Vec3 integratedRadiance() const noexcept { return integrated_; }
    // Largest untinted texel value per channel.
    PT_SHARED_FN Vec3 maximumTexel() const noexcept { return maximumTexel_; }
    // Solid angle of one cell of row `row`.
    PT_SHARED_ENTRY float cellSolidAngle (std::uint32_t row) const noexcept;

#if ! defined(__CUDACC__)
    // This view with each buffer replaced by what upload returns for it.
    template <typename Upload>
    EnvironmentMapView deviceCopy (Upload&& upload) const
    {
        EnvironmentMapView copy = *this;
        copy.texels_ = upload (texels_);
        copy.marginal_ = upload (marginal_);
        copy.conditional_ = upload (conditional_);
        copy.cellSolidAngle_ = upload (cellSolidAngle_);
        copy.rowCos_ = upload (rowCos_);
        return copy;
    }
#endif

protected:
    device::BufferView<std::array<float, 3>> texels_;
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    Vec3 tint_ {};
    float orientation_ = 0.0f;
    device::BufferView<float> marginal_;      // height + 1 entries, 0 to 1
    device::BufferView<float> conditional_;   // height rows of width + 1 entries, 0 to 1
    device::BufferView<float> cellSolidAngle_;
    device::BufferView<double> rowCos_;       // cos(pi * row / height), height + 1 entries
    Vec3 integrated_ {};
    Vec3 maximumTexel_ {};

private:
    PT_SHARED_ENTRY Vec3 texel (std::uint32_t x, std::uint32_t y) const noexcept;
    PT_SHARED_ENTRY std::uint32_t rowOf (Vec3 unitDirection) const noexcept;
    PT_SHARED_ENTRY std::uint32_t columnOf (float u) const noexcept;
};

// Host only: the Metal kernel (device_metal.metal) leaves this out.
#if ! defined(PATHTRACE_DEVICE_METAL)
class EnvironmentMap : public EnvironmentMapView
{
public:
    EnvironmentMap() = default;
    EnvironmentMap (const EnvironmentMap& other);
    EnvironmentMap (EnvironmentMap&& other) noexcept;
    EnvironmentMap& operator= (const EnvironmentMap& other);
    EnvironmentMap& operator= (EnvironmentMap&& other) noexcept;

    // Returns false and leaves the map inactive when the pixels are missing,
    // the tint or orientation is not finite, the tint is negative, a texel is
    // negative or not finite, or the tinted image is black. Keeps a reference
    // to pixels.texels; nothing is copied.
    bool build (const EnvironmentPixels& pixels, Vec3 tint, float orientationRadians);
    void clear() noexcept;

private:
    // Points the view at the buffers below. Runs after every change to them.
    void refreshView() noexcept;

    std::shared_ptr<const EnvironmentTexels> ownedTexels_;
    std::vector<float> ownedMarginal_;
    std::vector<float> ownedConditional_;
    std::vector<float> ownedCellSolidAngle_;
    std::vector<double> ownedRowCos_;
};
#endif

} // namespace videohelper::pathtrace
