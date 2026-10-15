#pragma once

#include "diffraction_material_admission.h"
#include "../../shared/DiffractionMaterialGpuLayout.h"

#include <array>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <string_view>

namespace diffractionmaterial
{
namespace physicalcheckpoint
{
inline constexpr float kDirectionTolerance = 1.0e-4f;
inline constexpr float kMaximumSpectralRadiance = 16.0f;
inline constexpr float kMaximumSinusoidalPhaseArgument = 4.0f;
inline constexpr auto kWavelengths = kProductionWavelengthsNanometres;

// CIE 1931 2-degree standard observer values at the exact checkpoint samples.
inline constexpr std::array<std::array<float, 3>, kMaximumSpectralSamples> kCie1931 {{
    { 0.001368f, 0.000039f, 0.006450f },
    { 0.134380f, 0.004000f, 0.645600f },
    { 0.290800f, 0.060000f, 1.669200f },
    { 0.004900f, 0.323000f, 0.272000f },
    { 0.290400f, 0.954000f, 0.020300f },
    { 0.916300f, 0.870000f, 0.001650f },
    { 0.447900f, 0.175000f, 0.000020f },
    { 0.011359f, 0.004102f, 0.000000f }
}};

inline constexpr std::array<float, kMaximumSpectralSamples> kQuadratureWeights {
    0.0625f, 0.125f, 0.125f, 0.125f, 0.125f, 0.15625f, 0.1875f, 0.09375f
};

inline bool validate(const AdmittedDiffractionMaterialIR& material,
                     const SpectralIncidentLight& light,
                     std::string_view backend,
                     std::string& error)
{
    const auto prefix = std::string("physical diffraction ") + std::string(backend);
    const auto& description = material.description();
    if (description.spectrum.wavelengthCount != kMaximumSpectralSamples)
    {
        error = prefix + " checkpoint requires eight spectral samples";
        return false;
    }
    for (std::size_t index = 0; index < kWavelengths.size(); ++index)
    {
        if (description.spectrum.wavelengthsNanometres[index] != kWavelengths[index])
        {
            error = prefix + " checkpoint requires the canonical wavelength layout";
            return false;
        }
    }
    if (description.microstructure.profile != GrooveProfile::BinaryRectangular
        && description.microstructure.profile != GrooveProfile::Sinusoidal
        && description.microstructure.profile != GrooveProfile::BlazedSawtooth)
    {
        error = prefix + " checkpoint does not implement this groove profile";
        return false;
    }
    if (description.microstructure.profile == GrooveProfile::Sinusoidal)
    {
        constexpr float twoPi = 6.28318530717958647692f;
        const float maximumArgument = twoPi
            * description.microstructure.grooveDepthNanometres
            / kWavelengths.front();
        if (maximumArgument > kMaximumSinusoidalPhaseArgument)
        {
            error = prefix
                + " sinusoidal profile exceeds the bounded phase argument";
            return false;
        }
    }
    if (description.coating.model != CoatingModel::Uncoated
        && description.coating.model != CoatingModel::IncoherentDielectric)
    {
        error = prefix + " checkpoint does not implement this coating model";
        return false;
    }
    const auto lengthSquared = light.direction[0] * light.direction[0]
        + light.direction[1] * light.direction[1]
        + light.direction[2] * light.direction[2];
    if (!std::isfinite(lengthSquared)
        || std::abs(lengthSquared - 1.0f) > kDirectionTolerance
        || light.direction[2] <= 1.0e-6f)
    {
        error = prefix + " incident direction is invalid";
        return false;
    }
    for (const auto radiance : light.radiance)
    {
        if (!std::isfinite(radiance) || radiance < 0.0f
            || radiance > kMaximumSpectralRadiance)
        {
            error = prefix + " incident spectrum is invalid";
            return false;
        }
    }
    return true;
}

inline PhysicalDiffractionGpuParameters makeGpuParameters(
    const AdmittedDiffractionMaterialIR& material,
    const SpectralIncidentLight& light) noexcept
{
    const auto& description = material.description();
    PhysicalDiffractionGpuParameters parameters;
    parameters.geometry = {
        description.geometry.directionUv[0],
        description.geometry.directionUv[1],
        description.geometry.grooveSpacingNanometres,
        description.microstructure.grooveDepthNanometres
    };
    parameters.secondaryGeometry = {
        description.geometry.secondaryDirectionUv[0],
        description.geometry.secondaryDirectionUv[1],
        description.geometry.secondaryGrooveSpacingNanometres,
        static_cast<float>(description.geometry.lattice)
    };
    parameters.microstructure = {
        description.microstructure.dutyCycle,
        description.substrate.refractiveIndex,
        description.substrate.extinctionCoefficient,
        static_cast<float>(description.spectrum.firstOrder)
    };
    parameters.control = {
        static_cast<float>(description.spectrum.lastOrder),
        static_cast<float>(description.microstructure.profile), 0.0f,
        static_cast<float>(description.coating.model)
    };
    parameters.incident = {
        light.direction[0], light.direction[1], light.direction[2], 0.0f
    };
    parameters.coating = {
        description.coating.thicknessNanometres,
        description.coating.opticalConstants.refractiveIndex,
        description.coating.opticalConstants.extinctionCoefficient, 0.0f
    };
    parameters.roughness = {
        description.roughness.rmsHeightNanometres,
        description.roughness.rmsSlope, 0.0f, 0.0f
    };
    parameters.grooveField = {
        static_cast<float>(description.grooveField.mode),
        description.grooveField.originUv[0],
        description.grooveField.originUv[1],
        description.grooveField.orientationDegreesPerUnit
    };
    parameters.grooveVariation = {
        description.grooveField.axisUv[0],
        description.grooveField.axisUv[1],
        description.grooveField.grooveSpacingDeltaNanometresPerUnit,
        description.grooveField.secondarySpacingDeltaNanometresPerUnit
    };
    for (std::size_t index = 0; index < kMaximumSpectralSamples; ++index)
    {
        parameters.spectral[index] = {
            kWavelengths[index], light.radiance[index],
            kCie1931[index][0], kCie1931[index][1]
        };
        parameters.spectralZ[index] = {
            kCie1931[index][2], kQuadratureWeights[index], 0.0f, 0.0f
        };
    }
    return parameters;
}

inline std::size_t expectedSignedOrderEvaluations(
    const AdmittedDiffractionMaterialIR& material) noexcept
{
    const auto& spectrum = material.description().spectrum;
    const auto signedNonzero = 2u
        * (static_cast<std::size_t>(spectrum.lastOrder)
           - static_cast<std::size_t>(spectrum.firstOrder) + 1u);
    const auto orderPairs = material.description().geometry.lattice
            == GratingLattice::CrossedTwoDimensional
        ? (signedNonzero + 1u) * (signedNonzero + 1u) - 1u
        : signedNonzero;
    return static_cast<std::size_t>(spectrum.wavelengthCount) * orderPairs;
}

// Validate the exact four-target shader readback before any native frame is
// published. This is validation only; it is not a CPU rendering path.
inline bool validateGpuReadback(
    const std::array<float, 4>& linearSrgb,
    const std::array<float, 4>& energy,
    const std::array<float, 4>& orders,
    const std::array<float, 4>& directionMoment,
    std::size_t expectedSignedOrderEvaluations) noexcept
{
    const auto finite = [](const std::array<float, 4>& values)
    {
        return std::all_of(values.begin(), values.end(),
                           [](float value) { return std::isfinite(value); });
    };
    if (!finite(linearSrgb) || !finite(energy) || !finite(orders)
        || !finite(directionMoment) || linearSrgb[3] != 1.0f
        || expectedSignedOrderEvaluations == 0
        || expectedSignedOrderEvaluations
            > kMaximumSpectralSamples
                * ((2u * kMaximumDiffractionOrder + 1u)
                   * (2u * kMaximumDiffractionOrder + 1u) - 1u))
    {
        return false;
    }
    if (std::any_of(linearSrgb.begin(), linearSrgb.begin() + 3,
                    [](float value) { return value < 0.0f; })
        || std::any_of(energy.begin(), energy.end(),
                       [](float value) { return value < 0.0f; })
        || std::any_of(orders.begin(), orders.end(),
                       [](float value) { return value < 0.0f; })
        || directionMoment[3] < 0.0f)
    {
        return false;
    }

    const float incident = energy[0];
    const float reflected = energy[1];
    const float zero = energy[2];
    const float higher = energy[3];
    const float unresolved = orders[0];
    const float absorbed = orders[1];
    const float propagating = orders[2];
    const float rejected = orders[3];
    const float tolerance = 2.0e-4f * std::max(1.0f, incident);
    const float countTolerance = 0.01f;
    const auto integerCount = [countTolerance](float value)
    {
        return std::abs(value - std::round(value)) <= countTolerance;
    };
    const float expected = static_cast<float>(expectedSignedOrderEvaluations);
    const float momentSquared = directionMoment[0] * directionMoment[0]
        + directionMoment[1] * directionMoment[1]
        + directionMoment[2] * directionMoment[2];
    const float resolved = zero + higher;
    return std::abs(incident - reflected - absorbed) <= tolerance
        && std::abs(reflected - resolved - unresolved) <= tolerance
        && std::abs(directionMoment[3] - resolved) <= tolerance
        && integerCount(propagating) && integerCount(rejected)
        && propagating <= expected + countTolerance
        && rejected <= expected + countTolerance
        && std::abs(propagating + rejected - expected) <= countTolerance
        && std::sqrt(momentSquared) <= resolved + tolerance;
}
} // namespace physicalcheckpoint
} // namespace diffractionmaterial
