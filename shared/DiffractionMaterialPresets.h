#pragma once

#include "DiffractionMaterialIR.h"

namespace diffractionmaterial
{
inline void setCanonicalVisibleSpectrum(SpectralSampling& spectrum)
{
    spectrum.integrationModel = SpectralIntegrationModel::Cie1931Xyz;
    spectrum.wavelengthCount = static_cast<std::uint8_t>(kMaximumSpectralSamples);
    spectrum.wavelengthsNanometres = kProductionWavelengthsNanometres;
    spectrum.firstOrder = 1;
    spectrum.lastOrder = 4;
}

// Aluminium-like binary reflective grating used by the strict native checkpoint.
// Optical constants are documented wavelength-invariant scalar approximations.
inline Description makeAluminiumBinaryGratingPreset()
{
    Description preset;
    preset.geometry.directionUv = { 1.0f, 0.0f };
    preset.geometry.grooveSpacingNanometres = 1000.0f;
    preset.microstructure = {
        GrooveProfile::BinaryRectangular, 125.0f, 0.50f, 0.0f
    };
    preset.substrate = { 0.17f, 3.10f };
    preset.coating = { CoatingModel::Uncoated, 0.0f, {} };
    preset.roughness = { 0.0f, 0.08f };
    setCanonicalVisibleSpectrum(preset.spectrum);
    return preset;
}

inline Description makeSinusoidalGratingPreset()
{
    auto preset = makeAluminiumBinaryGratingPreset();
    preset.microstructure = {
        GrooveProfile::Sinusoidal, 180.0f, 0.50f, 0.0f
    };
    preset.roughness = { 6.0f, 0.04f };
    return preset;
}

inline Description makeBlazedGratingPreset()
{
    auto preset = makeAluminiumBinaryGratingPreset();
    preset.geometry.grooveSpacingNanometres = 1000.0f;
    // depth/(duty*period) = 0.25, so the physical blaze angle is atan(0.25).
    preset.microstructure = {
        GrooveProfile::BlazedSawtooth, 250.0f, 1.0f, 14.036243f
    };
    preset.coating = {
        CoatingModel::IncoherentDielectric, 1000.0f, { 1.50f, 0.001f }
    };
    preset.roughness = { 2.0f, 0.02f };
    return preset;
}

inline Description makeCrossedTwoDimensionalGratingPreset() noexcept
{
    auto result = makeAluminiumBinaryGratingPreset();
    result.geometry.lattice = GratingLattice::CrossedTwoDimensional;
    result.geometry.secondaryDirectionUv = { 0.0f, 1.0f };
    result.geometry.secondaryGrooveSpacingNanometres = 1400.0f;
    return result;
}

inline Description makeHexagonalGratingPreset() noexcept
{
    auto result = makeCrossedTwoDimensionalGratingPreset();
    result.geometry.secondaryDirectionUv = { 0.5f, 0.8660254f };
    result.geometry.secondaryGrooveSpacingNanometres = result.geometry.grooveSpacingNanometres;
    result.geometry.thirdGrooveFamily = true;
    result.spectrum.lastOrder = 1;
    return result;
}

inline Description makeRealtimeCrossedTwoDimensionalGratingPreset() noexcept
{
    auto result = makeCrossedTwoDimensionalGratingPreset();
    // Keep the bounded native 2D tier viable at 1080p. The full physical
    // preset remains available for smaller frames and offline evaluation.
    result.spectrum.lastOrder = 1;
    return result;
}
} // namespace diffractionmaterial
