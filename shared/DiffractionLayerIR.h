#pragma once

#include "DiffractiveFoilIR.h"
#include "../video-helper/src/gpu_backend/stam_diffraction.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

// Diffraction layer: an optional companion of any Surface Material program.
// The layer adds Stam grating orders (and an optional clear coat, thin film and
// glitter) on top of the material's own BSDF. The math is the one in
// video-helper/src/gpu_backend/stam_diffraction.h; this header only holds the
// authored values, their bounds and the presets.
//
// Back-compat: a program without the companion renders exactly as before. Its
// structural digest, wire bytes and request version are unchanged. Only a
// program that carries the companion uses request wire version 5.
namespace surfacematerial
{
inline constexpr std::uint32_t kDiffractionLayerWireVersion = 1;

// Groove modes. The values are the stamdiffraction::kGroove* constants. Map is
// the Stam FIELD mode: the across-groove direction comes from a tangent-space
// direction instead of a fixed angle.
enum class DiffractionGrooveMode : std::uint32_t
{
    Linear = 0,
    Radial = 1,
    Crossed = 2,
    Map = 3,
    Cells = 4
};

// Base modes. Metallized is an opaque reflector under the grating (aluminium).
// Hri is a transparent high-index coating over the material.
enum class DiffractionBaseMode : std::uint32_t
{
    Metallized = 0,
    Hri = 1
};

enum class DiffractionLayerPreset : std::uint32_t
{
    Cd = 0,
    Dvd = 1,
    HolographicFoil = 2,
    Card = 3,
    OilSlick = 4
};
inline constexpr std::size_t kDiffractionLayerPresetCount = 5;

static_assert(static_cast<int>(DiffractionGrooveMode::Linear) == arbit::stamdiffraction::kGrooveLinear
                  && static_cast<int>(DiffractionGrooveMode::Radial) == arbit::stamdiffraction::kGrooveRadial
                  && static_cast<int>(DiffractionGrooveMode::Crossed) == arbit::stamdiffraction::kGrooveCrossed
                  && static_cast<int>(DiffractionGrooveMode::Map) == arbit::stamdiffraction::kGrooveField
                  && static_cast<int>(DiffractionGrooveMode::Cells) == arbit::stamdiffraction::kGrooveCells
                  && static_cast<int>(DiffractionBaseMode::Metallized) == arbit::stamdiffraction::kBaseOpaque
                  && static_cast<int>(DiffractionBaseMode::Hri) == arbit::stamdiffraction::kBaseHri,
              "Diffraction layer modes are the Stam layer constants");

// Bounds. Every field is finite and inside these ranges after admission.
inline constexpr std::uint32_t kDiffractionLayerMaximumOrder = 8;
inline constexpr float kDiffractionLayerMinimumPeriodNm = 100.0f;
inline constexpr float kDiffractionLayerMaximumPeriodNm = 100000.0f;
inline constexpr float kDiffractionLayerMaximumJitter = 0.95f;
inline constexpr float kDiffractionLayerMaximumAngle = 64.0f; // radians, about ten turns
inline constexpr float kDiffractionLayerMinimumCellSize = 1.0e-4f;
inline constexpr float kDiffractionLayerMaximumCellSize = 1.0e6f;
inline constexpr float kDiffractionLayerMaximumStrength = 16.0f;
inline constexpr float kDiffractionLayerMaximumIor = 4.0f;
inline constexpr float kDiffractionLayerMaximumFilmNm = 5000.0f;
inline constexpr float kDiffractionLayerMaximumTilt = 1.5707963f;
inline constexpr float kDiffractionLayerMinimumSharpness = 1.0e-3f;
inline constexpr float kDiffractionLayerMaximumUvScale = 1.0e6f;
inline constexpr float kDiffractionLayerMaximumCentre = 1000.0f;

// Field order after the three integers matches stamdiffraction::LayerParams.
// The last three fields place the layer on the surface: surfaceCoord is
// uv * uvScale (cells and glitter), and radial grooves circle the UV point
// (radialCentreU, radialCentreV), so the centre follows the mesh's UV layout
// and is the same for every instance.
struct DiffractionLayerDescription
{
    std::uint32_t version = kDiffractionLayerWireVersion;
    std::uint32_t grooveMode = static_cast<std::uint32_t>(DiffractionGrooveMode::Linear);
    std::uint32_t baseMode = static_cast<std::uint32_t>(DiffractionBaseMode::Metallized);
    std::uint32_t maxOrder = 3;
    float periodNm = 1000.0f;
    float secondPeriodNm = 1000.0f;
    float periodJitter = 0.0f;
    float grooveAngle = 0.0f; // radians
    float crossAngle = 0.0f;  // radians
    float grooveCellSize = 0.5f;
    float roughness = 0.03f;
    float zeroOrderSmear = 0.0f;
    float strength = 1.0f;
    float mask = 1.0f;
    float hriIor = 2.3f;
    float coatIor = 0.0f;
    float coatRoughness = 0.02f;
    float thinFilmThicknessNm = 0.0f;
    float thinFilmIor = 1.38f;
    float glitterDensity = 0.0f;
    float glitterCellSize = 0.1f;
    float glitterMaxTilt = 0.35f;
    float glitterSharpness = 0.05f;
    float uvScale = 100.0f;
    float radialCentreU = 0.5f;
    float radialCentreV = 0.5f;
};

// Visits every float field in wire order as visit(name, field). Digest, wire
// encoding, equality and the GPU packing (stam_diffraction_layer.h) all walk
// this one list. The names are the stamdiffraction::LayerParams and GLSL
// StamLayerParams field names, except the three placement fields at the end.
template <class Description, class Visitor>
inline void forEachDiffractionLayerFloat(Description& d, Visitor&& visit)
{
    visit("periodNm", d.periodNm);
    visit("secondPeriodNm", d.secondPeriodNm);
    visit("periodJitter", d.periodJitter);
    visit("grooveAngle", d.grooveAngle);
    visit("crossAngle", d.crossAngle);
    visit("grooveCellSize", d.grooveCellSize);
    visit("roughness", d.roughness);
    visit("zeroOrderSmear", d.zeroOrderSmear);
    visit("strength", d.strength);
    visit("mask", d.mask);
    visit("hriIor", d.hriIor);
    visit("coatIor", d.coatIor);
    visit("coatRoughness", d.coatRoughness);
    visit("thinFilmThicknessNm", d.thinFilmThicknessNm);
    visit("thinFilmIor", d.thinFilmIor);
    visit("glitterDensity", d.glitterDensity);
    visit("glitterCellSize", d.glitterCellSize);
    visit("glitterMaxTilt", d.glitterMaxTilt);
    visit("glitterSharpness", d.glitterSharpness);
    visit("uvScale", d.uvScale);
    visit("radialCentreU", d.radialCentreU);
    visit("radialCentreV", d.radialCentreV);
}
inline constexpr std::size_t kDiffractionLayerFloatCount = 22;
// The trailing float fields that place the layer and are not Stam parameters.
inline constexpr std::size_t kDiffractionLayerPlacementCount = 3;

inline bool validDiffractionLayer(const DiffractionLayerDescription& d) noexcept
{
    const auto within = [](float value, float low, float high) {
        return std::isfinite(value) && value >= low && value <= high;
    };
    bool finite = true;
    forEachDiffractionLayerFloat(d, [&finite](const char*, float value) { finite = finite && std::isfinite(value); });
    return finite && d.version == kDiffractionLayerWireVersion
        && d.grooveMode <= static_cast<std::uint32_t>(DiffractionGrooveMode::Cells)
        && d.baseMode <= static_cast<std::uint32_t>(DiffractionBaseMode::Hri)
        && d.maxOrder <= kDiffractionLayerMaximumOrder
        && within(d.periodNm, kDiffractionLayerMinimumPeriodNm, kDiffractionLayerMaximumPeriodNm)
        && within(d.secondPeriodNm, kDiffractionLayerMinimumPeriodNm, kDiffractionLayerMaximumPeriodNm)
        && within(d.periodJitter, 0.0f, kDiffractionLayerMaximumJitter)
        && within(d.grooveAngle, -kDiffractionLayerMaximumAngle, kDiffractionLayerMaximumAngle)
        && within(d.crossAngle, -kDiffractionLayerMaximumAngle, kDiffractionLayerMaximumAngle)
        && within(d.grooveCellSize, kDiffractionLayerMinimumCellSize, kDiffractionLayerMaximumCellSize)
        && within(d.roughness, 0.0f, 1.0f)
        && within(d.zeroOrderSmear, 0.0f, 1.0f)
        && within(d.strength, 0.0f, kDiffractionLayerMaximumStrength)
        && within(d.mask, 0.0f, 1.0f)
        && within(d.hriIor, 1.0f, kDiffractionLayerMaximumIor)
        && within(d.coatIor, 0.0f, kDiffractionLayerMaximumIor)
        && within(d.coatRoughness, 0.0f, 1.0f)
        && within(d.thinFilmThicknessNm, 0.0f, kDiffractionLayerMaximumFilmNm)
        && within(d.thinFilmIor, 1.0f, kDiffractionLayerMaximumIor)
        && within(d.glitterDensity, 0.0f, 1.0f)
        && within(d.glitterCellSize, kDiffractionLayerMinimumCellSize, kDiffractionLayerMaximumCellSize)
        && within(d.glitterMaxTilt, 0.0f, kDiffractionLayerMaximumTilt)
        && within(d.glitterSharpness, kDiffractionLayerMinimumSharpness, 1.0f)
        && d.uvScale > 0.0f && d.uvScale <= kDiffractionLayerMaximumUvScale
        && within(d.radialCentreU, -kDiffractionLayerMaximumCentre, kDiffractionLayerMaximumCentre)
        && within(d.radialCentreV, -kDiffractionLayerMaximumCentre, kDiffractionLayerMaximumCentre);
}

inline bool sameDiffractionLayer(const DiffractionLayerDescription& a, const DiffractionLayerDescription& b) noexcept
{
    if (a.version != b.version || a.grooveMode != b.grooveMode || a.baseMode != b.baseMode
        || a.maxOrder != b.maxOrder)
        return false;
    std::array<float, kDiffractionLayerFloatCount> left {}, right {};
    std::size_t i = 0;
    forEachDiffractionLayerFloat(a, [&](const char*, float value) { left[i++] = value; });
    i = 0;
    forEachDiffractionLayerFloat(b, [&](const char*, float value) { right[i++] = value; });
    // Bitwise, so -0 and +0 differ the same way the digest and wire bytes do.
    for (std::size_t k = 0; k < left.size(); ++k)
    {
        std::uint32_t x = 0, y = 0;
        static_assert(sizeof(float) == sizeof(std::uint32_t), "binary32 floats");
        std::memcpy(&x, &left[k], sizeof(x));
        std::memcpy(&y, &right[k], sizeof(y));
        if (x != y) return false;
    }
    return true;
}

// The Stam twin's parameters for this layer.
inline arbit::stamdiffraction::LayerParams toStamLayerParams(const DiffractionLayerDescription& d) noexcept
{
    arbit::stamdiffraction::LayerParams p;
    p.mode = static_cast<int>(d.grooveMode);
    p.baseMode = static_cast<int>(d.baseMode);
    p.periodNm = d.periodNm;
    p.secondPeriodNm = d.secondPeriodNm;
    p.periodJitter = d.periodJitter;
    p.grooveAngle = d.grooveAngle;
    p.crossAngle = d.crossAngle;
    p.grooveCellSize = d.grooveCellSize;
    p.maxOrder = static_cast<int>(d.maxOrder);
    p.roughness = d.roughness;
    p.zeroOrderSmear = d.zeroOrderSmear;
    p.strength = d.strength;
    p.mask = d.mask;
    p.hriIor = d.hriIor;
    p.coatIor = d.coatIor;
    p.coatRoughness = d.coatRoughness;
    p.thinFilmThicknessNm = d.thinFilmThicknessNm;
    p.thinFilmIor = d.thinFilmIor;
    p.glitterDensity = d.glitterDensity;
    p.glitterCellSize = d.glitterCellSize;
    p.glitterMaxTilt = d.glitterMaxTilt;
    p.glitterSharpness = d.glitterSharpness;
    return p;
}

// Layer values from Stam parameters. Placement fields keep their defaults.
inline DiffractionLayerDescription fromStamLayerParams(const arbit::stamdiffraction::LayerParams& p) noexcept
{
    DiffractionLayerDescription d;
    d.grooveMode = static_cast<std::uint32_t>(p.mode);
    d.baseMode = static_cast<std::uint32_t>(p.baseMode);
    d.maxOrder = static_cast<std::uint32_t>(p.maxOrder);
    d.periodNm = static_cast<float>(p.periodNm);
    d.secondPeriodNm = static_cast<float>(p.secondPeriodNm);
    d.periodJitter = static_cast<float>(p.periodJitter);
    d.grooveAngle = static_cast<float>(p.grooveAngle);
    d.crossAngle = static_cast<float>(p.crossAngle);
    d.grooveCellSize = static_cast<float>(p.grooveCellSize);
    d.roughness = static_cast<float>(p.roughness);
    d.zeroOrderSmear = static_cast<float>(p.zeroOrderSmear);
    d.strength = static_cast<float>(p.strength);
    d.mask = static_cast<float>(p.mask);
    d.hriIor = static_cast<float>(p.hriIor);
    d.coatIor = static_cast<float>(p.coatIor);
    d.coatRoughness = static_cast<float>(p.coatRoughness);
    d.thinFilmThicknessNm = static_cast<float>(p.thinFilmThicknessNm);
    d.thinFilmIor = static_cast<float>(p.thinFilmIor);
    d.glitterDensity = static_cast<float>(p.glitterDensity);
    d.glitterCellSize = static_cast<float>(p.glitterCellSize);
    d.glitterMaxTilt = static_cast<float>(p.glitterMaxTilt);
    d.glitterSharpness = static_cast<float>(p.glitterSharpness);
    return d;
}

inline constexpr std::string_view token(DiffractionLayerPreset preset) noexcept
{
    switch (preset)
    {
        case DiffractionLayerPreset::Cd: return "cd";
        case DiffractionLayerPreset::Dvd: return "dvd";
        case DiffractionLayerPreset::HolographicFoil: return "holographic-foil";
        case DiffractionLayerPreset::Card: return "card";
        case DiffractionLayerPreset::OilSlick: return "oil-slick";
    }
    return {};
}

// Presets are the Stam presets, so the GLSL stam*Preset constants, the twin
// and the material layer agree. CD and DVD set uvScale to 120, so surfaceCoord
// is in millimetres on a disc whose UV square spans the 120 mm disc.
inline DiffractionLayerDescription diffractionLayerPreset(DiffractionLayerPreset preset) noexcept
{
    namespace stam = arbit::stamdiffraction;
    switch (preset)
    {
        case DiffractionLayerPreset::Cd:
        {
            auto d = fromStamLayerParams(stam::kCdPreset);
            d.uvScale = 120.0f;
            return d;
        }
        case DiffractionLayerPreset::Dvd:
        {
            auto d = fromStamLayerParams(stam::kDvdPreset);
            d.uvScale = 120.0f;
            return d;
        }
        case DiffractionLayerPreset::HolographicFoil:
            return fromStamLayerParams(stam::kHoloFoilMetallizedPreset);
        case DiffractionLayerPreset::Card:
            return fromStamLayerParams(stam::kCardPreset);
        case DiffractionLayerPreset::OilSlick:
            return fromStamLayerParams(stam::kOilSlickPreset);
    }
    return {};
}

// The native Diffractive Foil material as a holographic-foil layer. The foil
// stores a 2x2 groove lattice over UV. The layer keeps the holographic-foil
// preset and takes the lattice's mean period, mean groove direction and mean
// coverage; the corner-to-corner period spread becomes the per-cell jitter.
inline DiffractionLayerDescription holographicFoilLayerFromFoil(const diffractivefoil::Description& foil) noexcept
{
    auto d = diffractionLayerPreset(DiffractionLayerPreset::HolographicFoil);
    double period = 0.0, low = 0.0, high = 0.0, coverage = 0.0, du = 0.0, dv = 0.0;
    bool first = true;
    for (const auto& corner : foil.grooveField)
    {
        const double spacing = corner.grooveSpacingNanometres;
        period += spacing;
        low = first ? spacing : std::min(low, spacing);
        high = first ? spacing : std::max(high, spacing);
        first = false;
        coverage += corner.diffractionCoverage;
        const double u = corner.reciprocalDirectionUv[0], v = corner.reciprocalDirectionUv[1];
        const double length = std::sqrt(u * u + v * v);
        if (std::isfinite(length) && length > 0.0)
        {
            du += u / length;
            dv += v / length;
        }
    }
    const double count = static_cast<double>(foil.grooveField.size());
    period /= count;
    coverage /= count;
    if (std::isfinite(period))
        d.periodNm = static_cast<float>(std::clamp(period,
            static_cast<double>(kDiffractionLayerMinimumPeriodNm), static_cast<double>(kDiffractionLayerMaximumPeriodNm)));
    d.secondPeriodNm = d.periodNm;
    const double spread = period > 0.0 ? 0.5 * (high - low) / period : 0.0;
    d.periodJitter = std::isfinite(spread)
        ? static_cast<float>(std::clamp(spread, 0.0, static_cast<double>(kDiffractionLayerMaximumJitter))) : 0.0f;
    d.grooveAngle = du * du + dv * dv > 0.0 ? static_cast<float>(std::atan2(dv, du)) : 0.0f;
    d.mask = std::isfinite(coverage) ? static_cast<float>(std::clamp(coverage, 0.0, 1.0)) : 1.0f;
    return d;
}
} // namespace surfacematerial
