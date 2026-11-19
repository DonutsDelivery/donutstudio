#pragma once

#include "DiffractionMaterialIR.h"
#include "DiffractionLightingPlan.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <type_traits>

namespace diffractionmaterial
{
// Backend-neutral 16-byte lanes shared by the strict OpenGL and Metal physical
// diffraction shaders.
struct alignas(16) GpuFloat4 final
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 0.0f;
};

struct alignas(16) PhysicalDiffractionGpuParameters final
{
    GpuFloat4 geometry;       // reciprocal xy, spacing nm, depth nm
    GpuFloat4 secondaryGeometry; // reciprocal xy, spacing nm, GratingLattice
    GpuFloat4 microstructure; // duty, substrate n/k, first order
    GpuFloat4 control;        // last order, profile, third groove family, coating model
    GpuFloat4 incident;
    GpuFloat4 coating;        // thickness nm, coating n/k, reserved
    GpuFloat4 roughness;      // RMS height nm, RMS slope, streak slope, streak spacing uv
    GpuFloat4 grooveField;    // mode, origin uv, orientation degrees/unit
    GpuFloat4 grooveVariation; // axis uv, primary/secondary spacing delta nm/unit
    std::array<GpuFloat4, kMaximumSpectralSamples> spectral;
    std::array<GpuFloat4, kMaximumSpectralSamples> spectralZ;
};

struct alignas(16) PhysicalDiffractionGpuLightingPath final
{
    PhysicalDiffractionGpuParameters material;
    GpuFloat4 incidentDirectionAndIntensity;
    std::array<std::uint32_t, 4> kindBounceAndReserved {};
};

inline PhysicalDiffractionGpuLightingPath makeGpuLightingPath(
    PhysicalDiffractionGpuParameters material,
    const SpectralLightingPath& source) noexcept
{
    PhysicalDiffractionGpuLightingPath result;
    result.material = material;
    float intensity = 0.0f;
    for (const auto radiance : source.incident.radiance)
        intensity = std::max(intensity, radiance);
    result.incidentDirectionAndIntensity = {
        source.incident.direction[0], source.incident.direction[1],
        source.incident.direction[2], intensity };
    result.kindBounceAndReserved[0] = static_cast<std::uint32_t>(source.kind);
    result.kindBounceAndReserved[1] = source.bounceDepth;
    if (intensity > 0.0f)
        for (auto& spectral : result.material.spectral)
            spectral.y /= intensity;
    return result;
}

static_assert(static_cast<unsigned>(GrooveProfile::BinaryRectangular) == 1u
              && static_cast<unsigned>(GrooveProfile::Sinusoidal) == 2u
              && static_cast<unsigned>(GrooveProfile::BlazedSawtooth) == 3u,
              "native shader profile tokens are part of the GPU ABI");
static_assert(static_cast<unsigned>(GratingLattice::OneDimensional) == 1u
              && static_cast<unsigned>(GratingLattice::CrossedTwoDimensional) == 2u,
              "native shader lattice tokens are part of the GPU ABI");
static_assert(std::is_standard_layout_v<GpuFloat4>
              && std::is_trivially_copyable_v<GpuFloat4>
              && sizeof(GpuFloat4) == 16 && alignof(GpuFloat4) == 16);
static_assert(std::is_standard_layout_v<PhysicalDiffractionGpuParameters>
              && std::is_trivially_copyable_v<PhysicalDiffractionGpuParameters>);
static_assert(offsetof(PhysicalDiffractionGpuParameters, geometry) == 0);
static_assert(offsetof(PhysicalDiffractionGpuParameters, secondaryGeometry) == 16);
static_assert(offsetof(PhysicalDiffractionGpuParameters, microstructure) == 32);
static_assert(offsetof(PhysicalDiffractionGpuParameters, control) == 48);
static_assert(offsetof(PhysicalDiffractionGpuParameters, incident) == 64);
static_assert(offsetof(PhysicalDiffractionGpuParameters, coating) == 80);
static_assert(offsetof(PhysicalDiffractionGpuParameters, roughness) == 96);
static_assert(offsetof(PhysicalDiffractionGpuParameters, grooveField) == 112);
static_assert(offsetof(PhysicalDiffractionGpuParameters, grooveVariation) == 128);
static_assert(offsetof(PhysicalDiffractionGpuParameters, spectral) == 144);
static_assert(offsetof(PhysicalDiffractionGpuParameters, spectralZ) == 272);
static_assert(sizeof(PhysicalDiffractionGpuParameters) == 400);
static_assert(std::is_standard_layout_v<PhysicalDiffractionGpuLightingPath>
              && std::is_trivially_copyable_v<PhysicalDiffractionGpuLightingPath>);
static_assert(offsetof(PhysicalDiffractionGpuLightingPath, material) == 0);
static_assert(offsetof(PhysicalDiffractionGpuLightingPath,
                       incidentDirectionAndIntensity) == 400);
static_assert(offsetof(PhysicalDiffractionGpuLightingPath,
                       kindBounceAndReserved) == 416);
static_assert(sizeof(PhysicalDiffractionGpuLightingPath) == 432);
} // namespace diffractionmaterial
