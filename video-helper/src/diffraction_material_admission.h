#pragma once

#include "../../shared/DiffractionMaterialIR.h"
#include "sha256.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <utility>

namespace diffractionmaterial
{
class AdmittedDiffractionMaterialIR;

std::optional<AdmittedDiffractionMaterialIR> admit(
    const Description& source,
    std::string& error,
    const AdmissionLimits& limits = {});

// Structural admission never grants rendering. A backend must separately own and
// initialize an exact physical executor. The independent scalar oracle is tests-only.
inline constexpr bool kAuthorizesNativeGpuExecution = false;
inline constexpr bool kAllowsCpuProductionRenderingFallback = false;

enum class NativeBackend : std::uint8_t
{
    Invalid = 0,
    OpenGl = 1,
    Metal = 2
};

enum class NativeExecutionUse : std::uint8_t
{
    Invalid = 0,
    Preview = 1,
    Export = 2
};

class AdmittedDiffractionMaterialIR final
{
public:
    AdmittedDiffractionMaterialIR(const AdmittedDiffractionMaterialIR&) = default;
    AdmittedDiffractionMaterialIR(AdmittedDiffractionMaterialIR&&) = default;
    AdmittedDiffractionMaterialIR& operator=(const AdmittedDiffractionMaterialIR&) = delete;
    AdmittedDiffractionMaterialIR& operator=(AdmittedDiffractionMaterialIR&&) = delete;

    const Description& description() const noexcept { return description_; }
    const std::string& structuralDigest() const noexcept { return digest_; }

    static constexpr bool authorizesNativeGpuExecution = false;
    static constexpr bool allowsCpuProductionRenderingFallback = false;

private:
    friend std::optional<AdmittedDiffractionMaterialIR> admit(
        const Description&, std::string&, const AdmissionLimits&);

    AdmittedDiffractionMaterialIR(Description description, std::string digest)
        : description_(std::move(description)), digest_(std::move(digest))
    {
    }

    const Description description_;
    const std::string digest_;
};

// Readiness is backend-local and fail-closed. Material-specific support is checked
// independently before command submission by the strict execution checkpoint.
inline bool requireNativeExecution(NativeExecutionUse use,
                                   NativeBackend backend,
                                   bool openGlPhysicalExecutorReady,
                                   bool metalPhysicalExecutorReady,
                                   std::string& error)
{
    if (use != NativeExecutionUse::Preview && use != NativeExecutionUse::Export)
    {
        error = "physical diffraction execution use is invalid";
        return false;
    }
    switch (backend)
    {
        case NativeBackend::OpenGl:
            if (openGlPhysicalExecutorReady)
            {
                error.clear();
                return true;
            }
            error = "physical diffraction native OpenGL backend is unavailable";
            return false;
        case NativeBackend::Metal:
            if (metalPhysicalExecutorReady)
            {
                error.clear();
                return true;
            }
            error = "physical diffraction native Metal backend is unavailable";
            return false;
        case NativeBackend::Invalid:
            error = "physical diffraction native backend is invalid";
            return false;
    }
    error = "physical diffraction native backend is invalid";
    return false;
}

namespace detail
{
inline bool finite(float value) noexcept
{
    return std::isfinite(value);
}

inline bool bounded(float value, float minimum, float maximum) noexcept
{
    return finite(value) && value >= minimum && value <= maximum;
}

inline bool validateLimits(const AdmissionLimits& limits, std::string& error)
{
    if (limits.spectralSamples != kMaximumSpectralSamples
        || limits.diffractionOrder == 0
        || limits.diffractionOrder > kMaximumDiffractionOrder)
    {
        error = "diffraction material admission limits do not match the bounded production tier";
        return false;
    }
    return true;
}

inline bool validateGeometry(const GratingGeometry& geometry, std::string& error)
{
    const auto lengthSquared = geometry.directionUv[0] * geometry.directionUv[0]
        + geometry.directionUv[1] * geometry.directionUv[1];
    if (!finite(lengthSquared) || std::abs(lengthSquared - 1.0f) > 0.0001f)
    {
        error = "diffraction material grating direction must be unit length";
        return false;
    }
    if (!bounded(geometry.grooveSpacingNanometres,
                 kMinimumGrooveSpacingNanometres,
                 kMaximumGrooveSpacingNanometres))
    {
        error = "diffraction material groove period is invalid";
        return false;
    }
    if (geometry.lattice == GratingLattice::OneDimensional)
    {
        if (geometry.secondaryDirectionUv[0] != 0.0f
            || geometry.secondaryDirectionUv[1] != 0.0f
            || geometry.secondaryGrooveSpacingNanometres != 0.0f)
        {
            error = "one-dimensional diffraction material carries secondary lattice data";
            return false;
        }
        return true;
    }
    if (geometry.lattice != GratingLattice::CrossedTwoDimensional)
    {
        error = "diffraction material grating lattice is unsupported";
        return false;
    }
    const auto secondaryLengthSquared
        = geometry.secondaryDirectionUv[0] * geometry.secondaryDirectionUv[0]
        + geometry.secondaryDirectionUv[1] * geometry.secondaryDirectionUv[1];
    const auto directionDot
        = geometry.directionUv[0] * geometry.secondaryDirectionUv[0]
        + geometry.directionUv[1] * geometry.secondaryDirectionUv[1];
    if (!finite(secondaryLengthSquared)
        || std::abs(secondaryLengthSquared - 1.0f) > 0.0001f
        || !finite(directionDot) || std::abs(directionDot) > 0.0001f
        || !bounded(geometry.secondaryGrooveSpacingNanometres,
                    kMinimumGrooveSpacingNanometres,
                    kMaximumGrooveSpacingNanometres))
    {
        error = "crossed diffraction material requires an orthonormal secondary lattice and valid period";
        return false;
    }
    return true;
}

inline bool validateGrooveField(const GratingGeometry& geometry,
                                const GrooveField& field,
                                std::string& error)
{
    const auto canonicalConstant = field.originUv == std::array<float, 2> {}
        && field.axisUv == std::array<float, 2> {}
        && field.grooveSpacingDeltaNanometresPerUnit == 0.0f
        && field.secondarySpacingDeltaNanometresPerUnit == 0.0f
        && field.orientationDegreesPerUnit == 0.0f;
    if (field.mode == GrooveFieldMode::Constant)
    {
        if (!canonicalConstant)
        {
            error = "constant diffraction groove field carries analytic field data";
            return false;
        }
        return true;
    }
    if (field.mode != GrooveFieldMode::Linear
        && field.mode != GrooveFieldMode::Radial)
    {
        error = "diffraction groove field mode is unsupported";
        return false;
    }
    if (!bounded(field.originUv[0], 0.0f, 1.0f)
        || !bounded(field.originUv[1], 0.0f, 1.0f)
        || !bounded(field.grooveSpacingDeltaNanometresPerUnit,
                    -kMaximumGrooveSpacingNanometres,
                    kMaximumGrooveSpacingNanometres)
        || !bounded(field.secondarySpacingDeltaNanometresPerUnit,
                    -kMaximumGrooveSpacingNanometres,
                    kMaximumGrooveSpacingNanometres)
        || !bounded(field.orientationDegreesPerUnit,
                    -kMaximumFieldOrientationDegreesPerUnit,
                    kMaximumFieldOrientationDegreesPerUnit))
    {
        error = "diffraction groove field parameters are invalid";
        return false;
    }
    if (geometry.lattice == GratingLattice::OneDimensional
        && field.secondarySpacingDeltaNanometresPerUnit != 0.0f)
    {
        error = "one-dimensional diffraction groove field carries secondary spacing variation";
        return false;
    }
    if (field.mode == GrooveFieldMode::Linear)
    {
        const auto axisLengthSquared = field.axisUv[0] * field.axisUv[0]
            + field.axisUv[1] * field.axisUv[1];
        if (!finite(axisLengthSquared)
            || std::abs(axisLengthSquared - 1.0f) > 0.0001f)
        {
            error = "linear diffraction groove field axis must be unit length";
            return false;
        }
    }
    else if (field.axisUv != std::array<float, 2> {})
    {
        error = "radial diffraction groove field carries a linear axis";
        return false;
    }

    float minimumCoordinate = 0.0f;
    float maximumCoordinate = 0.0f;
    const std::array<std::array<float, 2>, 4> corners {{
        {{ 0.0f, 0.0f }}, {{ 1.0f, 0.0f }},
        {{ 0.0f, 1.0f }}, {{ 1.0f, 1.0f }}
    }};
    bool first = true;
    for (const auto& corner : corners)
    {
        const auto dx = corner[0] - field.originUv[0];
        const auto dy = corner[1] - field.originUv[1];
        const auto coordinate = field.mode == GrooveFieldMode::Linear
            ? dx * field.axisUv[0] + dy * field.axisUv[1]
            : std::sqrt(dx * dx + dy * dy);
        if (first || coordinate < minimumCoordinate) minimumCoordinate = coordinate;
        if (first || coordinate > maximumCoordinate) maximumCoordinate = coordinate;
        first = false;
    }
    const auto periodIsBounded = [minimumCoordinate, maximumCoordinate](
        float base, float delta)
    {
        return bounded(base + delta * minimumCoordinate,
                       kMinimumGrooveSpacingNanometres,
                       kMaximumGrooveSpacingNanometres)
            && bounded(base + delta * maximumCoordinate,
                       kMinimumGrooveSpacingNanometres,
                       kMaximumGrooveSpacingNanometres);
    };
    if (!periodIsBounded(geometry.grooveSpacingNanometres,
                         field.grooveSpacingDeltaNanometresPerUnit)
        || (geometry.lattice == GratingLattice::CrossedTwoDimensional
            && !periodIsBounded(geometry.secondaryGrooveSpacingNanometres,
                                field.secondarySpacingDeltaNanometresPerUnit)))
    {
        error = "diffraction groove field produces an out-of-bounds local period";
        return false;
    }
    return true;
}

inline bool validateMicrostructure(const GratingGeometry& geometry,
                                   const GratingMicrostructure& microstructure,
                                   std::string& error)
{
    if (microstructure.profile != GrooveProfile::BinaryRectangular
        && microstructure.profile != GrooveProfile::Sinusoidal
        && microstructure.profile != GrooveProfile::BlazedSawtooth)
    {
        error = "diffraction material groove profile is unsupported";
        return false;
    }
    if (!bounded(microstructure.grooveDepthNanometres,
                 kMinimumGrooveDepthNanometres,
                 kMaximumGrooveDepthNanometres)
        || !bounded(microstructure.dutyCycle, 0.01f, 1.0f)
        || !finite(microstructure.blazeAngleDegrees))
    {
        error = "diffraction material physical groove microstructure is invalid";
        return false;
    }

    if (microstructure.profile == GrooveProfile::Sinusoidal
        && std::abs(microstructure.dutyCycle - 0.5f) > 1.0e-6f)
    {
        error = "diffraction material sinusoidal profile requires canonical duty cycle";
        return false;
    }
    if (microstructure.profile != GrooveProfile::BlazedSawtooth)
    {
        if (microstructure.blazeAngleDegrees != 0.0f)
        {
            error = "diffraction material non-blazed profile cannot carry a blaze angle";
            return false;
        }
        return true;
    }

    constexpr double radiansToDegrees = 57.2957795130823208768;
    const auto expected = std::atan(
        static_cast<double>(microstructure.grooveDepthNanometres)
        / (static_cast<double>(microstructure.dutyCycle)
           * geometry.grooveSpacingNanometres)) * radiansToDegrees;
    if (!(microstructure.blazeAngleDegrees > 0.0f)
        || microstructure.blazeAngleDegrees >= 89.0f
        || std::abs(static_cast<double>(microstructure.blazeAngleDegrees) - expected)
            > 1.0e-3)
    {
        error = "diffraction material blaze angle is inconsistent with depth, duty, and period";
        return false;
    }
    return true;
}

inline bool validateOpticalConstants(const OpticalConstants& constants) noexcept
{
    return bounded(constants.refractiveIndex,
                   kMinimumSubstrateRefractiveIndex,
                   kMaximumSubstrateRefractiveIndex)
        && bounded(constants.extinctionCoefficient, 0.0f,
                   kMaximumExtinctionCoefficient);
}

inline bool validateCoating(const SurfaceCoating& coating, std::string& error)
{
    if (coating.model == CoatingModel::Uncoated)
    {
        if (coating.thicknessNanometres != 0.0f
            || coating.opticalConstants.refractiveIndex != 0.0f
            || coating.opticalConstants.extinctionCoefficient != 0.0f)
        {
            error = "diffraction material uncoated surface carries coating data";
            return false;
        }
        return true;
    }
    if (coating.model != CoatingModel::IncoherentDielectric)
    {
        error = "diffraction material coating model is unsupported";
        return false;
    }
    if (!bounded(coating.thicknessNanometres, 0.1f,
                 kMaximumCoatingThicknessNanometres)
        || !validateOpticalConstants(coating.opticalConstants))
    {
        error = "diffraction material physical coating is invalid";
        return false;
    }
    return true;
}

inline bool validateRoughness(const SurfaceRoughness& roughness,
                              std::string& error)
{
    if (!bounded(roughness.rmsHeightNanometres, 0.0f,
                 kMaximumRmsHeightNanometres)
        || !bounded(roughness.rmsSlope, 0.0f, kMaximumRmsSlope))
    {
        error = "diffraction material physical roughness is invalid";
        return false;
    }
    return true;
}

inline bool validateSpectrum(const SpectralSampling& spectrum,
                             const AdmissionLimits& limits,
                             std::string& error)
{
    if (spectrum.integrationModel != SpectralIntegrationModel::Cie1931Xyz)
    {
        error = "diffraction material spectral integration model is unsupported";
        return false;
    }
    if (spectrum.wavelengthCount != limits.spectralSamples)
    {
        error = "diffraction material production tier requires exactly eight spectral samples";
        return false;
    }

    float previous = 0.0f;
    for (std::size_t index = 0; index < spectrum.wavelengthsNanometres.size(); ++index)
    {
        const auto wavelength = spectrum.wavelengthsNanometres[index];
        if (index >= spectrum.wavelengthCount)
        {
            if (wavelength != 0.0f)
            {
                error = "diffraction material has data outside declared wavelengths";
                return false;
            }
            continue;
        }
        if (!bounded(wavelength, kMinimumWavelengthNanometres,
                     kMaximumWavelengthNanometres)
            || (index != 0 && wavelength <= previous)
            || wavelength != kProductionWavelengthsNanometres[index])
        {
            error = "diffraction material production tier requires the canonical wavelength schedule";
            return false;
        }
        previous = wavelength;
    }

    if (spectrum.firstOrder != 1
        || spectrum.firstOrder > spectrum.lastOrder
        || spectrum.lastOrder > limits.diffractionOrder)
    {
        error = "diffraction material diffraction order range is invalid";
        return false;
    }
    return true;
}

inline void canonicalize(float& value) noexcept
{
    if (value == 0.0f)
        value = 0.0f;
}

inline void canonicalize(OpticalConstants& constants) noexcept
{
    canonicalize(constants.refractiveIndex);
    canonicalize(constants.extinctionCoefficient);
}

inline void canonicalize(Description& description) noexcept
{
    for (auto& direction : description.geometry.directionUv)
        canonicalize(direction);
    canonicalize(description.geometry.grooveSpacingNanometres);
    for (auto& direction : description.geometry.secondaryDirectionUv)
        canonicalize(direction);
    canonicalize(description.geometry.secondaryGrooveSpacingNanometres);
    for (auto& value : description.grooveField.originUv) canonicalize(value);
    for (auto& value : description.grooveField.axisUv) canonicalize(value);
    canonicalize(description.grooveField.grooveSpacingDeltaNanometresPerUnit);
    canonicalize(description.grooveField.secondarySpacingDeltaNanometresPerUnit);
    canonicalize(description.grooveField.orientationDegreesPerUnit);
    canonicalize(description.microstructure.grooveDepthNanometres);
    canonicalize(description.microstructure.dutyCycle);
    canonicalize(description.microstructure.blazeAngleDegrees);
    canonicalize(description.substrate);
    canonicalize(description.coating.thicknessNanometres);
    canonicalize(description.coating.opticalConstants);
    canonicalize(description.roughness.rmsHeightNanometres);
    canonicalize(description.roughness.rmsSlope);
    for (auto& wavelength : description.spectrum.wavelengthsNanometres)
        canonicalize(wavelength);
}

inline void hashU8(videohelper::Sha256& hash, std::uint8_t value)
{
    hash.update(&value, sizeof(value));
}

inline void hashU32(videohelper::Sha256& hash, std::uint32_t value)
{
    const std::uint8_t bytes[] {
        static_cast<std::uint8_t>(value >> 24u),
        static_cast<std::uint8_t>(value >> 16u),
        static_cast<std::uint8_t>(value >> 8u),
        static_cast<std::uint8_t>(value)
    };
    hash.update(bytes, sizeof(bytes));
}

inline void hashFloat(videohelper::Sha256& hash, float value)
{
    static_assert(sizeof(float) == sizeof(std::uint32_t)
                  && std::numeric_limits<float>::is_iec559,
                  "Diffraction material digest requires IEEE-754 binary32 floats");
    if (value == 0.0f)
        value = 0.0f;
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    hashU32(hash, bits);
}

template <std::size_t Size>
inline void hashFloats(videohelper::Sha256& hash,
                       const std::array<float, Size>& values)
{
    for (const auto value : values)
        hashFloat(hash, value);
}

inline void hashOpticalConstants(videohelper::Sha256& hash,
                                 const OpticalConstants& constants)
{
    hashFloat(hash, constants.refractiveIndex);
    hashFloat(hash, constants.extinctionCoefficient);
}

inline std::string structuralDigest(const Description& description)
{
    static constexpr char domain[] = "DonutStudio/DiffractionMaterialIR/Structure/v5";
    videohelper::Sha256 hash;
    hash.update(domain, sizeof(domain) - 1);
    hashU32(hash, description.version);
    hashFloats(hash, description.geometry.directionUv);
    hashFloat(hash, description.geometry.grooveSpacingNanometres);
    hashU8(hash, static_cast<std::uint8_t>(description.geometry.lattice));
    hashFloats(hash, description.geometry.secondaryDirectionUv);
    hashFloat(hash, description.geometry.secondaryGrooveSpacingNanometres);
    hashU8(hash, static_cast<std::uint8_t>(description.grooveField.mode));
    hashFloats(hash, description.grooveField.originUv);
    hashFloats(hash, description.grooveField.axisUv);
    hashFloat(hash, description.grooveField.grooveSpacingDeltaNanometresPerUnit);
    hashFloat(hash, description.grooveField.secondarySpacingDeltaNanometresPerUnit);
    hashFloat(hash, description.grooveField.orientationDegreesPerUnit);

    hashU8(hash, static_cast<std::uint8_t>(description.microstructure.profile));
    hashFloat(hash, description.microstructure.grooveDepthNanometres);
    hashFloat(hash, description.microstructure.dutyCycle);
    hashFloat(hash, description.microstructure.blazeAngleDegrees);
    hashOpticalConstants(hash, description.substrate);
    hashU8(hash, static_cast<std::uint8_t>(description.coating.model));
    hashFloat(hash, description.coating.thicknessNanometres);
    hashOpticalConstants(hash, description.coating.opticalConstants);
    hashFloat(hash, description.roughness.rmsHeightNanometres);
    hashFloat(hash, description.roughness.rmsSlope);

    hashU8(hash, static_cast<std::uint8_t>(description.spectrum.integrationModel));
    hashU8(hash, description.spectrum.wavelengthCount);
    for (std::size_t index = 0; index < description.spectrum.wavelengthCount; ++index)
        hashFloat(hash, description.spectrum.wavelengthsNanometres[index]);
    hashU8(hash, description.spectrum.firstOrder);
    hashU8(hash, description.spectrum.lastOrder);
    return hash.finishHex();
}
} // namespace detail

inline std::optional<AdmittedDiffractionMaterialIR> admit(
    const Description& source,
    std::string& error,
    const AdmissionLimits& limits)
{
    error.clear();
    if (!detail::validateLimits(limits, error))
        return std::nullopt;
    if (source.version != kWireVersion)
    {
        error = "diffraction material wire version is unsupported";
        return std::nullopt;
    }
    if (!detail::validateGeometry(source.geometry, error)
        || !detail::validateGrooveField(source.geometry, source.grooveField, error)
        || !detail::validateMicrostructure(source.geometry, source.microstructure, error))
    {
        return std::nullopt;
    }
    if (source.microstructure.profile == GrooveProfile::BlazedSawtooth
        && (source.grooveField.grooveSpacingDeltaNanometresPerUnit != 0.0f
            || source.grooveField.secondarySpacingDeltaNanometresPerUnit != 0.0f))
    {
        error = "diffraction material blazed profile requires a spatially constant period";
        return std::nullopt;
    }
    if (source.microstructure.profile == GrooveProfile::BlazedSawtooth
        && source.geometry.lattice == GratingLattice::CrossedTwoDimensional
        && source.geometry.secondaryGrooveSpacingNanometres
            != source.geometry.grooveSpacingNanometres)
    {
        error = "crossed blazed diffraction requires equal primary and secondary periods";
        return std::nullopt;
    }
    if (!detail::validateOpticalConstants(source.substrate))
    {
        error = "diffraction material substrate optical constants are invalid";
        return std::nullopt;
    }
    if (!detail::validateCoating(source.coating, error)
        || !detail::validateRoughness(source.roughness, error)
        || !detail::validateSpectrum(source.spectrum, limits, error))
    {
        return std::nullopt;
    }

    Description candidate = source;
    detail::canonicalize(candidate);
    return AdmittedDiffractionMaterialIR(
        candidate, detail::structuralDigest(candidate));
}

static_assert(!kAuthorizesNativeGpuExecution);
static_assert(!kAllowsCpuProductionRenderingFallback);
static_assert(!AdmittedDiffractionMaterialIR::authorizesNativeGpuExecution);
static_assert(!AdmittedDiffractionMaterialIR::allowsCpuProductionRenderingFallback);
} // namespace diffractionmaterial
