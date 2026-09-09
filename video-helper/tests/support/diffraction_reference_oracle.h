#pragma once

#include "../../src/diffraction_material_admission.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace diffractionmaterial::reference
{
// High-precision test-only scalar-wave oracle. Production preview/export must
// execute an admitted native GPU path; this code is never a CPU fallback renderer.
inline constexpr std::size_t kMaximumOrderEvents =
    kMaximumSpectralSamples
        * (1 + 2 * kMaximumDiffractionOrder)
        * (1 + 2 * kMaximumDiffractionOrder);
inline constexpr double kVisibleRangeNanometres =
    static_cast<double>(kMaximumWavelengthNanometres - kMinimumWavelengthNanometres);
inline constexpr std::size_t kHighSampleReferenceCount = 65;

enum class OrderDisposition : std::uint8_t
{
    Invalid = 0,
    Propagating,
    Evanescent,
    Grazing
};

enum class EvaluationFailure : std::uint8_t
{
    None = 0,
    InvalidIncidentDirection,
    InvalidMaterialUv,
    InvalidSpectrum,
    DegenerateGrating,
    EnergyBoundViolation
};

inline constexpr std::string_view token(OrderDisposition disposition) noexcept
{
    switch (disposition)
    {
        case OrderDisposition::Invalid: return "invalid";
        case OrderDisposition::Propagating: return "propagating";
        case OrderDisposition::Evanescent: return "evanescent";
        case OrderDisposition::Grazing: return "grazing";
    }
    return {};
}

inline constexpr std::string_view token(EvaluationFailure failure) noexcept
{
    switch (failure)
    {
        case EvaluationFailure::None: return "none";
        case EvaluationFailure::InvalidIncidentDirection: return "invalid-incident-direction";
        case EvaluationFailure::InvalidMaterialUv: return "invalid-material-uv";
        case EvaluationFailure::InvalidSpectrum: return "invalid-spectrum";
        case EvaluationFailure::DegenerateGrating: return "degenerate-grating";
        case EvaluationFailure::EnergyBoundViolation: return "energy-bound-violation";
    }
    return {};
}

struct Direction
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct OrderSolution
{
    OrderDisposition disposition = OrderDisposition::Invalid;
    Direction outgoingDirection;
};

struct CieXyz
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct LinearSrgb
{
    double red = 0.0;
    double green = 0.0;
    double blue = 0.0;
};

struct EvaluationInput
{
    // Directions use the local material frame and point away from the surface.
    // The reflective grating equation is wo_t + wi_t = m lambda / period.
    Direction incidentDirection { 0.0, 0.0, 1.0 };
    std::array<double, 2> materialUv { 0.5, 0.5 };
    // Spectral radiance at the admitted wavelengths. The high-sample convergence
    // oracle linearly interpolates this physical spectrum; it does not invent RGB.
    std::array<double, kMaximumSpectralSamples> incidentSpectrum {
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    };
};

struct SpatialParitySample
{
    std::string_view name;
    Description description;
    EvaluationInput input;
};

const std::array<SpatialParitySample, 7>& spatialParitySamples() noexcept;

Description resolveLocalGrooveGeometry(
    const Description& source,
    const std::array<double, 2>& materialUv) noexcept;

struct SpectralSample
{
    double wavelengthNanometres = 0.0;
    double quadratureWeight = 0.0;
    double incidentEnergy = 0.0;
    double substrateReflectedEnergy = 0.0;
    double resolvedReflectedEnergy = 0.0;
    double zeroOrderEnergy = 0.0;
    double higherOrderEnergy = 0.0;
    double unresolvedReflectedEnergy = 0.0;
    double absorbedEnergy = 0.0;
};

struct OrderEvent
{
    std::uint8_t wavelengthIndex = 0;
    std::int8_t order = 0;
    std::int8_t secondaryOrder = 0;
    double wavelengthNanometres = 0.0;
    Direction outgoingDirection;
    // Fraction of incident spectral energy carried by this resolved order.
    double efficiency = 0.0;
    double energy = 0.0;
    double angularStandardDeviationRadians = 0.0;
};

struct IntegratedLedger
{
    double incidentEnergy = 0.0;
    double substrateReflectedEnergy = 0.0;
    double resolvedReflectedEnergy = 0.0;
    double zeroOrderEnergy = 0.0;
    double higherOrderEnergy = 0.0;
    double unresolvedReflectedEnergy = 0.0;
    double absorbedEnergy = 0.0;
    CieXyz resolvedReflectedXyz;
    CieXyz zeroOrderXyz;
    CieXyz higherOrderXyz;
    LinearSrgb outputLinearSrgb;
    LinearSrgb zeroOrderLinearSrgb;
    LinearSrgb higherOrderLinearSrgb;
};

struct EvaluationResult : IntegratedLedger
{
    std::uint8_t spectralSampleCount = 0;
    std::array<SpectralSample, kMaximumSpectralSamples> spectralSamples {};
    std::size_t eventCount = 0;
    std::array<OrderEvent, kMaximumOrderEvents> events {};
    std::size_t evanescentOrderCount = 0;
    std::size_t grazingOrderCount = 0;
};

struct HighSampleReferenceResult : IntegratedLedger
{
    std::size_t spectralSampleCount = kHighSampleReferenceCount;
};

struct DirectionalBsdfResult
{
    CieXyz radianceXyz;
    LinearSrgb outputLinearSrgb;
};

OrderSolution solveOrder(Direction incidentDirection,
                         std::array<double, 2> gratingDirection,
                         double grooveSpacingNanometres,
                         double wavelengthNanometres,
                         int signedOrder) noexcept;

OrderSolution solveCrossedOrder(
    Direction incidentDirection,
    std::array<double, 2> primaryDirection,
    double primarySpacingNanometres,
    int primaryOrder,
    std::array<double, 2> secondaryDirection,
    double secondarySpacingNanometres,
    int secondaryOrder,
    double wavelengthNanometres) noexcept;

// Scalar normal-incidence Fresnel reflectance for n + i*k.
double normalIncidenceSubstrateReflectance(double refractiveIndex,
                                           double extinctionCoefficient) noexcept;

// Coherence-averaged two-interface/absorption model. It contains no optical phase,
// interference fringes, or authored color and is deliberately not a thin-film model.
double incoherentCoatedSubstrateReflectance(
    double wavelengthNanometres,
    OpticalConstants substrate,
    const SurfaceCoating& coating) noexcept;

// Analytic/Fourier-derived intensity coefficient for one scalar phase profile.
// Complete integer orders obey Parseval before evanescent/truncation classification
// when the incident and outgoing cosines match.
double profileOrderEfficiency(GrooveProfile profile,
                              double wavelengthNanometres,
                              double grooveDepthNanometres,
                              double dutyCycle,
                              double incidentCosine,
                              int signedOrder) noexcept;

// Reflective scalar-relief efficiency. The optical path through a relief step
// depends on both the incident and diffracted outgoing angles.
double reflectiveProfileOrderEfficiency(GrooveProfile profile,
                                        double wavelengthNanometres,
                                        double grooveDepthNanometres,
                                        double dutyCycle,
                                        double incidentCosine,
                                        double outgoingCosine,
                                        int signedOrder) noexcept;

// Fraction remaining in coherent discrete orders after RMS-height attenuation.
double coherentRoughnessFraction(double wavelengthNanometres,
                                 double rmsHeightNanometres,
                                 double incidentCosine) noexcept;

double reflectiveCoherentRoughnessFraction(double wavelengthNanometres,
                                           double rmsHeightNanometres,
                                           double incidentCosine,
                                           double outgoingCosine) noexcept;

// CIE 1931 XYZ to D65 linear sRGB at the output boundary only.
LinearSrgb xyzToLinearSrgbD65(CieXyz xyz) noexcept;

double gaussianOrderLobeDensity(Direction centre,
                                Direction outgoing,
                                double angularStandardDeviationRadians) noexcept;

std::optional<DirectionalBsdfResult> evaluateDirectionalBsdf(
    const AdmittedDiffractionMaterialIR& material,
    const EvaluationInput& input,
    Direction outgoingDirection,
    EvaluationFailure& failure) noexcept;

std::optional<EvaluationResult> evaluate(
    const AdmittedDiffractionMaterialIR& material,
    const EvaluationInput& input,
    EvaluationFailure& failure) noexcept;

// Independent 5 nm (65 sample) integration path used only for convergence tests.
// It does not alter production sample bounds or authorize CPU rendering.
std::optional<HighSampleReferenceResult> evaluateHighSampleReference(
    const AdmittedDiffractionMaterialIR& material,
    const EvaluationInput& input,
    EvaluationFailure& failure) noexcept;
} // namespace diffractionmaterial::reference
