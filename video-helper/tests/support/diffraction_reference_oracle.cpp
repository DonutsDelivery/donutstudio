#include "diffraction_reference_oracle.h"
#include "diffraction_cie1931_2deg_5nm.h"
#include "../../../shared/DiffractionMaterialPresets.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>

namespace diffractionmaterial::reference
{
const std::array<SpatialParitySample, 7>& spatialParitySamples() noexcept
{
    static const auto samples = []
    {
        std::array<SpatialParitySample, 7> result {};
        for (auto& sample : result)
        {
            sample.description = makeAluminiumBinaryGratingPreset();
            sample.description.roughness.rmsSlope = 1.0e-4f;
        }

        result[0].name = "constant-boundary-00";
        result[0].input.materialUv = { 0.0, 0.0 };

        result[1].name = "linear-boundary-10";
        result[1].description = makeBlazedGratingPreset();
        result[1].description.grooveField.mode = GrooveFieldMode::Linear;
        result[1].description.grooveField.originUv = { 0.5f, 0.5f };
        result[1].description.grooveField.axisUv = { 1.0f, 0.0f };
        result[1].description.grooveField.orientationDegreesPerUnit = -40.0f;
        result[1].input.materialUv = { 1.0, 0.0 };

        result[2].name = "radial-boundary-01";
        result[2].description = makeBlazedGratingPreset();
        result[2].description.grooveField.mode = GrooveFieldMode::Radial;
        result[2].description.grooveField.originUv = { 0.0f, 0.0f };
        result[2].description.grooveField.orientationDegreesPerUnit = -20.0f;
        result[2].input.materialUv = { 0.0, 1.0 };

        result[3].name = "crossed-boundary-11";
        result[3].description = makeCrossedTwoDimensionalGratingPreset();
        result[3].description.roughness.rmsSlope = 1.0e-4f;
        result[3].input.materialUv = { 1.0, 1.0 };

        result[4].name = "orientation-only-blazed-boundary-00";
        result[4].description = makeBlazedGratingPreset();
        result[4].description.roughness.rmsSlope = 1.0e-4f;
        result[4].description.grooveField.mode = GrooveFieldMode::Linear;
        result[4].description.grooveField.originUv = { 0.5f, 0.5f };
        result[4].description.grooveField.axisUv = { 1.0f, 0.0f };
        result[4].description.grooveField.orientationDegreesPerUnit = 40.0f;
        result[4].input.materialUv = { 0.0, 0.0 };

        result[5].name = "crossed-linear-period-orientation-origin";
        result[5].description = makeCrossedTwoDimensionalGratingPreset();
        result[5].description.grooveField.mode = GrooveFieldMode::Linear;
        result[5].description.grooveField.originUv = { 0.5f, 0.5f };
        result[5].description.grooveField.axisUv = { 0.6f, 0.8f };
        result[5].description.grooveField.grooveSpacingDeltaNanometresPerUnit
            = 120.0f;
        result[5].description.grooveField.secondarySpacingDeltaNanometresPerUnit
            = -80.0f;
        result[5].description.grooveField.orientationDegreesPerUnit = 35.0f;
        result[5].input.materialUv = { 0.5, 0.5 };

        result[6].name = "radial-period-orientation-origin";
        result[6].description.grooveField.mode = GrooveFieldMode::Radial;
        result[6].description.grooveField.originUv = { 0.25f, 0.75f };
        result[6].description.grooveField.grooveSpacingDeltaNanometresPerUnit
            = 180.0f;
        result[6].description.grooveField.orientationDegreesPerUnit = -30.0f;
        result[6].input.materialUv = { 0.25, 0.75 };
        for (auto& sample : result)
        {
            sample.description.spectrum.firstOrder = 1;
            sample.description.spectrum.lastOrder = 1;
            sample.description.roughness.rmsSlope = 1.0e-4f;
        }
        return result;
    }();
    return samples;
}

namespace
{
constexpr double kPi = 3.1415926535897932384626433832795;
constexpr double kDirectionTolerance = 1.0e-9;
constexpr double kGrazingCosine = 1.0e-8;
constexpr double kGrazingCosineSquared = kGrazingCosine * kGrazingCosine;
constexpr double kMaximumSpectralRadiance = 16.0;
constexpr double kEnergyTolerance = 2.0e-10;


struct WavelengthFractions
{
    double reflectance = 0.0;
    double resolved = 0.0;
    double zero = 0.0;
    double higher = 0.0;
};

bool finite(double value) noexcept
{
    return std::isfinite(value);
}

double lengthSquared(Direction direction) noexcept
{
    return direction.x * direction.x + direction.y * direction.y
         + direction.z * direction.z;
}

Direction normalize(Direction direction) noexcept
{
    const auto inverseLength = 1.0 / std::sqrt(lengthSquared(direction));
    direction.x *= inverseLength;
    direction.y *= inverseLength;
    direction.z *= inverseLength;
    return direction;
}

CieEntry cieAt(double wavelengthNanometres) noexcept
{
    const auto clamped = std::clamp(
        wavelengthNanometres,
        static_cast<double>(kMinimumWavelengthNanometres),
        static_cast<double>(kMaximumWavelengthNanometres));
    const auto coordinate = (clamped - kMinimumWavelengthNanometres) / 5.0;
    const auto lower = static_cast<std::size_t>(std::floor(coordinate));
    if (lower >= kCie1931.size() - 1)
        return kCie1931.back();
    const auto fraction = coordinate - static_cast<double>(lower);
    const auto& first = kCie1931[lower];
    const auto& second = kCie1931[lower + 1];
    return {
        first.x + fraction * (second.x - first.x),
        first.y + fraction * (second.y - first.y),
        first.z + fraction * (second.z - first.z)
    };
}

void add(CieXyz& destination, const CieXyz& value) noexcept
{
    destination.x += value.x;
    destination.y += value.y;
    destination.z += value.z;
}

CieXyz colorContribution(double energy, const CieEntry& cie,
                         double referenceWhiteY) noexcept
{
    const auto scale = energy / referenceWhiteY;
    return { cie.x * scale, cie.y * scale, cie.z * scale };
}

double quadratureWeight(const SpectralSampling& spectrum, std::size_t index) noexcept
{
    const auto wavelength = static_cast<double>(spectrum.wavelengthsNanometres[index]);
    const auto left = index == 0
        ? static_cast<double>(kMinimumWavelengthNanometres)
        : 0.5 * (spectrum.wavelengthsNanometres[index - 1] + wavelength);
    const auto right = index + 1 == spectrum.wavelengthCount
        ? static_cast<double>(kMaximumWavelengthNanometres)
        : 0.5 * (wavelength + spectrum.wavelengthsNanometres[index + 1]);
    return (right - left) / kVisibleRangeNanometres;
}

bool validInput(const EvaluationInput& input, std::size_t sampleCount,
                EvaluationFailure& failure) noexcept
{
    const auto& incident = input.incidentDirection;
    if (!finite(input.materialUv[0]) || !finite(input.materialUv[1])
        || input.materialUv[0] < 0.0 || input.materialUv[0] > 1.0
        || input.materialUv[1] < 0.0 || input.materialUv[1] > 1.0)
    {
        failure = EvaluationFailure::InvalidMaterialUv;
        return false;
    }
    if (!finite(incident.x) || !finite(incident.y) || !finite(incident.z)
        || std::abs(lengthSquared(incident) - 1.0) > kDirectionTolerance
        || incident.z <= kGrazingCosine)
    {
        failure = EvaluationFailure::InvalidIncidentDirection;
        return false;
    }
    for (std::size_t index = 0; index < sampleCount; ++index)
    {
        const auto value = input.incidentSpectrum[index];
        if (!finite(value) || value < 0.0 || value > kMaximumSpectralRadiance)
        {
            failure = EvaluationFailure::InvalidSpectrum;
            return false;
        }
    }
    return true;
}

std::complex<double> exponentialIntegral(double frequency,
                                         double begin,
                                         double end) noexcept
{
    if (std::abs(frequency) < 1.0e-12)
        return { end - begin, 0.0 };
    const std::complex<double> imaginary(0.0, 1.0);
    return (std::exp(imaginary * frequency * end)
            - std::exp(imaginary * frequency * begin))
        / (imaginary * frequency);
}

double interfaceReflectance(std::complex<double> first,
                            std::complex<double> second) noexcept
{
    const auto denominator = first + second;
    if (!(std::norm(denominator) > std::numeric_limits<double>::min()))
        return 1.0;
    return std::clamp(std::norm((first - second) / denominator), 0.0, 1.0);
}

double interpolatedIncident(const SpectralSampling& spectrum,
                            const EvaluationInput& input,
                            double wavelength) noexcept
{
    const auto count = static_cast<std::size_t>(spectrum.wavelengthCount);
    if (wavelength <= spectrum.wavelengthsNanometres[0])
        return input.incidentSpectrum[0];
    if (wavelength >= spectrum.wavelengthsNanometres[count - 1])
        return input.incidentSpectrum[count - 1];
    for (std::size_t upper = 1; upper < count; ++upper)
    {
        if (wavelength <= spectrum.wavelengthsNanometres[upper])
        {
            const auto lowerWavelength = spectrum.wavelengthsNanometres[upper - 1];
            const auto width = spectrum.wavelengthsNanometres[upper] - lowerWavelength;
            const auto t = (wavelength - lowerWavelength) / width;
            return input.incidentSpectrum[upper - 1]
                 + t * (input.incidentSpectrum[upper] - input.incidentSpectrum[upper - 1]);
        }
    }
    return input.incidentSpectrum[count - 1];
}

bool wavelengthFractions(const Description& description,
                         Direction incident,
                         double wavelength,
                         WavelengthFractions& fractions,
                         std::size_t* evanescentCount,
                         std::size_t* grazingCount,
                         OrderEvent* events,
                         std::size_t* eventCount,
                         std::uint8_t wavelengthIndex) noexcept
{
    fractions = {};
    fractions.reflectance = incoherentCoatedSubstrateReflectance(
        wavelength, description.substrate, description.coating);
    const auto zeroCoherent = reflectiveCoherentRoughnessFraction(
        wavelength, description.roughness.rmsHeightNanometres,
        incident.z, incident.z);
    if (!finite(fractions.reflectance) || !finite(zeroCoherent))
        return false;

    const auto zeroProfile = reflectiveProfileOrderEfficiency(
        description.microstructure.profile, wavelength,
        description.microstructure.grooveDepthNanometres,
        description.microstructure.dutyCycle, incident.z, incident.z, 0);
    const auto crossedTwoDimensional = description.geometry.lattice
        == GratingLattice::CrossedTwoDimensional;
    fractions.zero = zeroProfile * zeroCoherent
        * (crossedTwoDimensional ? zeroProfile : 1.0);
    fractions.resolved = fractions.zero;

    if (events && eventCount)
    {
        auto& zero = events[(*eventCount)++];
        zero.wavelengthIndex = wavelengthIndex;
        zero.wavelengthNanometres = wavelength;
        zero.outgoingDirection = { -incident.x, -incident.y, incident.z };
        zero.efficiency = fractions.reflectance * fractions.zero;
        zero.angularStandardDeviationRadians = 2.0 * description.roughness.rmsSlope;
    }

    const auto firstOrder = static_cast<int>(description.spectrum.firstOrder);
    const auto lastOrder = static_cast<int>(description.spectrum.lastOrder);
    for (int primaryOrder = -lastOrder; primaryOrder <= lastOrder; ++primaryOrder)
    {
        const auto primaryMagnitude = std::abs(primaryOrder);
        if (primaryOrder != 0 && primaryMagnitude < firstOrder)
            continue;
        const auto secondaryFirst = crossedTwoDimensional ? -lastOrder : 0;
        const auto secondaryLast = crossedTwoDimensional ? lastOrder : 0;
        for (int secondaryOrder = secondaryFirst;
             secondaryOrder <= secondaryLast; ++secondaryOrder)
        {
            const auto secondaryMagnitude = std::abs(secondaryOrder);
            if ((secondaryOrder != 0 && secondaryMagnitude < firstOrder)
                || (primaryOrder == 0 && secondaryOrder == 0))
                continue;

            const auto solution = crossedTwoDimensional
                ? solveCrossedOrder(
                    incident,
                    { description.geometry.directionUv[0],
                      description.geometry.directionUv[1] },
                    description.geometry.grooveSpacingNanometres, primaryOrder,
                    { description.geometry.secondaryDirectionUv[0],
                      description.geometry.secondaryDirectionUv[1] },
                    description.geometry.secondaryGrooveSpacingNanometres,
                    secondaryOrder, wavelength)
                : solveOrder(
                    incident,
                    { description.geometry.directionUv[0],
                      description.geometry.directionUv[1] },
                    description.geometry.grooveSpacingNanometres,
                    wavelength, primaryOrder);
            if (solution.disposition == OrderDisposition::Evanescent)
            {
                if (evanescentCount) ++(*evanescentCount);
                continue;
            }
            if (solution.disposition == OrderDisposition::Grazing)
            {
                if (grazingCount) ++(*grazingCount);
                continue;
            }
            if (solution.disposition != OrderDisposition::Propagating)
                return false;

            const auto outgoingCosine = solution.outgoingDirection.z;
            auto efficiency = reflectiveProfileOrderEfficiency(
                description.microstructure.profile, wavelength,
                description.microstructure.grooveDepthNanometres,
                description.microstructure.dutyCycle, incident.z,
                outgoingCosine, primaryOrder)
                * reflectiveCoherentRoughnessFraction(
                    wavelength, description.roughness.rmsHeightNanometres,
                    incident.z, outgoingCosine);
            if (crossedTwoDimensional)
                efficiency *= reflectiveProfileOrderEfficiency(
                    description.microstructure.profile, wavelength,
                    description.microstructure.grooveDepthNanometres,
                    description.microstructure.dutyCycle, incident.z,
                    outgoingCosine, secondaryOrder);
            // Convert the scalar field coefficient to reflected power flux.
            // The obliquity ratio is applied once for a paired 2D order.
            efficiency *= outgoingCosine / incident.z;
            if (!finite(efficiency) || efficiency < 0.0)
                return false;

            fractions.higher += efficiency;
            fractions.resolved += efficiency;
            if (events && eventCount)
            {
                auto& event = events[(*eventCount)++];
                event.wavelengthIndex = wavelengthIndex;
                event.order = static_cast<std::int8_t>(primaryOrder);
                event.secondaryOrder = static_cast<std::int8_t>(secondaryOrder);
                event.wavelengthNanometres = wavelength;
                event.outgoingDirection = solution.outgoingDirection;
                event.efficiency = fractions.reflectance * efficiency;
                event.angularStandardDeviationRadians =
                    2.0 * description.roughness.rmsSlope;
            }
        }
    }
    return fractions.resolved <= 1.0 + kEnergyTolerance;
}

void addIntegratedSample(IntegratedLedger& ledger,
                         const WavelengthFractions& fractions,
                         double weightedIncident,
                         const CieEntry& cie,
                         double referenceWhiteY) noexcept
{
    const auto reflected = weightedIncident * fractions.reflectance;
    const auto resolved = reflected * fractions.resolved;
    const auto zero = reflected * fractions.zero;
    const auto higher = reflected * fractions.higher;
    const auto unresolved = std::max(0.0, reflected - resolved);
    const auto absorbed = std::max(0.0, weightedIncident - reflected);

    ledger.incidentEnergy += weightedIncident;
    ledger.substrateReflectedEnergy += reflected;
    ledger.resolvedReflectedEnergy += resolved;
    ledger.zeroOrderEnergy += zero;
    ledger.higherOrderEnergy += higher;
    ledger.unresolvedReflectedEnergy += unresolved;
    ledger.absorbedEnergy += absorbed;
    add(ledger.resolvedReflectedXyz,
        colorContribution(resolved, cie, referenceWhiteY));
    add(ledger.zeroOrderXyz, colorContribution(zero, cie, referenceWhiteY));
    add(ledger.higherOrderXyz, colorContribution(higher, cie, referenceWhiteY));
}

void finalizeColors(IntegratedLedger& ledger) noexcept
{
    ledger.outputLinearSrgb = xyzToLinearSrgbD65(ledger.resolvedReflectedXyz);
    ledger.zeroOrderLinearSrgb = xyzToLinearSrgbD65(ledger.zeroOrderXyz);
    ledger.higherOrderLinearSrgb = xyzToLinearSrgbD65(ledger.higherOrderXyz);
}
} // namespace

OrderSolution solveOrder(Direction incidentDirection,
                         std::array<double, 2> gratingDirection,
                         double grooveSpacingNanometres,
                         double wavelengthNanometres,
                         int signedOrder) noexcept
{
    const auto gratingLengthSquared = gratingDirection[0] * gratingDirection[0]
                                    + gratingDirection[1] * gratingDirection[1];
    if (!finite(incidentDirection.x) || !finite(incidentDirection.y)
        || !finite(incidentDirection.z)
        || std::abs(lengthSquared(incidentDirection) - 1.0) > kDirectionTolerance
        || incidentDirection.z <= kGrazingCosine
        || !finite(gratingLengthSquared)
        || std::abs(gratingLengthSquared - 1.0) > kDirectionTolerance
        || !finite(grooveSpacingNanometres) || grooveSpacingNanometres <= 0.0
        || !finite(wavelengthNanometres) || wavelengthNanometres <= 0.0
        || signedOrder < -static_cast<int>(kMaximumDiffractionOrder)
        || signedOrder > static_cast<int>(kMaximumDiffractionOrder))
    {
        return {};
    }

    incidentDirection = normalize(incidentDirection);
    const auto shift = static_cast<double>(signedOrder)
                     * wavelengthNanometres / grooveSpacingNanometres;
    const auto tangentX = -incidentDirection.x + shift * gratingDirection[0];
    const auto tangentY = -incidentDirection.y + shift * gratingDirection[1];
    const auto tangentLengthSquared = tangentX * tangentX + tangentY * tangentY;
    if (tangentLengthSquared > 1.0 + kGrazingCosineSquared)
        return { OrderDisposition::Evanescent, {} };
    const auto normalSquared = 1.0 - tangentLengthSquared;
    if (normalSquared <= kGrazingCosineSquared)
        return { OrderDisposition::Grazing, {} };
    return {
        OrderDisposition::Propagating,
        { tangentX, tangentY, std::sqrt(normalSquared) }
    };
}

OrderSolution solveCrossedOrder(
    Direction incidentDirection,
    std::array<double, 2> primaryDirection,
    double primarySpacingNanometres,
    int primaryOrder,
    std::array<double, 2> secondaryDirection,
    double secondarySpacingNanometres,
    int secondaryOrder,
    double wavelengthNanometres) noexcept
{
    const auto primaryLengthSquared = primaryDirection[0] * primaryDirection[0]
                                    + primaryDirection[1] * primaryDirection[1];
    const auto secondaryLengthSquared = secondaryDirection[0] * secondaryDirection[0]
                                      + secondaryDirection[1] * secondaryDirection[1];
    const auto directionDot = primaryDirection[0] * secondaryDirection[0]
                            + primaryDirection[1] * secondaryDirection[1];
    if (!finite(incidentDirection.x) || !finite(incidentDirection.y)
        || !finite(incidentDirection.z)
        || std::abs(lengthSquared(incidentDirection) - 1.0) > kDirectionTolerance
        || incidentDirection.z <= kGrazingCosine
        || !finite(primaryLengthSquared) || !finite(secondaryLengthSquared)
        || std::abs(primaryLengthSquared - 1.0) > kDirectionTolerance
        || std::abs(secondaryLengthSquared - 1.0) > kDirectionTolerance
        || !finite(directionDot) || std::abs(directionDot) > kDirectionTolerance
        || !finite(primarySpacingNanometres) || primarySpacingNanometres <= 0.0
        || !finite(secondarySpacingNanometres) || secondarySpacingNanometres <= 0.0
        || !finite(wavelengthNanometres) || wavelengthNanometres <= 0.0
        || primaryOrder < -static_cast<int>(kMaximumDiffractionOrder)
        || primaryOrder > static_cast<int>(kMaximumDiffractionOrder)
        || secondaryOrder < -static_cast<int>(kMaximumDiffractionOrder)
        || secondaryOrder > static_cast<int>(kMaximumDiffractionOrder))
    {
        return {};
    }

    incidentDirection = normalize(incidentDirection);
    const auto primaryShift = static_cast<double>(primaryOrder)
                            * wavelengthNanometres / primarySpacingNanometres;
    const auto secondaryShift = static_cast<double>(secondaryOrder)
                              * wavelengthNanometres / secondarySpacingNanometres;
    const auto tangentX = -incidentDirection.x
        + primaryShift * primaryDirection[0]
        + secondaryShift * secondaryDirection[0];
    const auto tangentY = -incidentDirection.y
        + primaryShift * primaryDirection[1]
        + secondaryShift * secondaryDirection[1];
    const auto tangentLengthSquared = tangentX * tangentX + tangentY * tangentY;
    if (tangentLengthSquared > 1.0 + kGrazingCosineSquared)
        return { OrderDisposition::Evanescent, {} };
    const auto normalSquared = 1.0 - tangentLengthSquared;
    if (normalSquared <= kGrazingCosineSquared)
        return { OrderDisposition::Grazing, {} };
    return {
        OrderDisposition::Propagating,
        { tangentX, tangentY, std::sqrt(normalSquared) }
    };
}

double normalIncidenceSubstrateReflectance(double refractiveIndex,
                                           double extinctionCoefficient) noexcept
{
    if (!finite(refractiveIndex) || !finite(extinctionCoefficient)
        || refractiveIndex <= 0.0 || extinctionCoefficient < 0.0)
        return std::numeric_limits<double>::quiet_NaN();
    const auto numerator = (refractiveIndex - 1.0) * (refractiveIndex - 1.0)
                         + extinctionCoefficient * extinctionCoefficient;
    const auto denominator = (refractiveIndex + 1.0) * (refractiveIndex + 1.0)
                           + extinctionCoefficient * extinctionCoefficient;
    return numerator / denominator;
}

double incoherentCoatedSubstrateReflectance(
    double wavelengthNanometres,
    OpticalConstants substrate,
    const SurfaceCoating& coating) noexcept
{
    if (!finite(wavelengthNanometres) || wavelengthNanometres <= 0.0)
        return std::numeric_limits<double>::quiet_NaN();
    if (coating.model == CoatingModel::Uncoated)
        return normalIncidenceSubstrateReflectance(
            substrate.refractiveIndex, substrate.extinctionCoefficient);
    if (coating.model != CoatingModel::IncoherentDielectric)
        return std::numeric_limits<double>::quiet_NaN();

    const std::complex<double> air(1.0, 0.0);
    const std::complex<double> layer(
        coating.opticalConstants.refractiveIndex,
        coating.opticalConstants.extinctionCoefficient);
    const std::complex<double> base(
        substrate.refractiveIndex, substrate.extinctionCoefficient);
    const auto r01 = interfaceReflectance(air, layer);
    const auto r12 = interfaceReflectance(layer, base);
    // Beer-Lambert intensity attenuation for a round trip through the layer.
    const auto roundTrip = std::exp(
        -8.0 * kPi * coating.opticalConstants.extinctionCoefficient
        * coating.thicknessNanometres / wavelengthNanometres);
    const auto denominator = 1.0 - r01 * r12 * roundTrip;
    if (!(denominator > std::numeric_limits<double>::min()))
        return 1.0;
    const auto incoherent = r01
        + (1.0 - r01) * (1.0 - r01) * r12 * roundTrip / denominator;
    return std::clamp(incoherent, 0.0, 1.0);
}

double profileOrderEfficiency(GrooveProfile profile,
                              double wavelengthNanometres,
                              double grooveDepthNanometres,
                              double dutyCycle,
                              double incidentCosine,
                              int signedOrder) noexcept
{
    return reflectiveProfileOrderEfficiency(
        profile, wavelengthNanometres, grooveDepthNanometres, dutyCycle,
        incidentCosine, incidentCosine, signedOrder);
}

double reflectiveProfileOrderEfficiency(GrooveProfile profile,
                                        double wavelengthNanometres,
                                        double grooveDepthNanometres,
                                        double dutyCycle,
                                        double incidentCosine,
                                        double outgoingCosine,
                                        int signedOrder) noexcept
{
    if (!finite(wavelengthNanometres) || wavelengthNanometres <= 0.0
        || !finite(grooveDepthNanometres) || grooveDepthNanometres < 0.0
        || !finite(dutyCycle) || dutyCycle <= 0.0 || dutyCycle > 1.0
        || !finite(incidentCosine) || incidentCosine <= 0.0
        || incidentCosine > 1.0
        || !finite(outgoingCosine) || outgoingCosine <= 0.0
        || outgoingCosine > 1.0)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const auto phaseExcursion = 2.0 * kPi * grooveDepthNanometres
                              * (incidentCosine + outgoingCosine)
                              / wavelengthNanometres;
    if (profile == GrooveProfile::BinaryRectangular)
    {
        if (signedOrder == 0)
        {
            const auto real = 1.0 - dutyCycle
                            + dutyCycle * std::cos(phaseExcursion);
            const auto imaginary = dutyCycle * std::sin(phaseExcursion);
            return real * real + imaginary * imaginary;
        }
        const auto order = static_cast<double>(signedOrder);
        const auto sine = std::sin(kPi * order * dutyCycle);
        const auto amplitudeSquared = sine * sine / (kPi * kPi * order * order);
        return 4.0 * std::sin(0.5 * phaseExcursion)
                   * std::sin(0.5 * phaseExcursion) * amplitudeSquared;
    }
    if (profile == GrooveProfile::Sinusoidal)
    {
#if defined(__APPLE__)
        // libc++ does not provide the C++17 special-math functions on every
        // supported macOS SDK, while Darwin's libSystem exports integer jn.
        const auto amplitude = ::jn(std::abs(signedOrder), 0.5 * phaseExcursion);
#else
        const auto amplitude = std::cyl_bessel_j(
            std::abs(signedOrder), 0.5 * phaseExcursion);
#endif
        return amplitude * amplitude;
    }
    if (profile == GrooveProfile::BlazedSawtooth)
    {
        const auto orderFrequency = -2.0 * kPi * signedOrder;
        const auto ramp = exponentialIntegral(
            phaseExcursion / dutyCycle + orderFrequency, 0.0, dutyCycle);
        const auto land = exponentialIntegral(orderFrequency, dutyCycle, 1.0);
        return std::norm(ramp + land);
    }
    return std::numeric_limits<double>::quiet_NaN();
}

double coherentRoughnessFraction(double wavelengthNanometres,
                                 double rmsHeightNanometres,
                                 double incidentCosine) noexcept
{
    return reflectiveCoherentRoughnessFraction(
        wavelengthNanometres, rmsHeightNanometres,
        incidentCosine, incidentCosine);
}

double reflectiveCoherentRoughnessFraction(double wavelengthNanometres,
                                           double rmsHeightNanometres,
                                           double incidentCosine,
                                           double outgoingCosine) noexcept
{
    if (!finite(wavelengthNanometres) || wavelengthNanometres <= 0.0
        || !finite(rmsHeightNanometres) || rmsHeightNanometres < 0.0
        || !finite(incidentCosine) || incidentCosine <= 0.0
        || incidentCosine > 1.0
        || !finite(outgoingCosine) || outgoingCosine <= 0.0
        || outgoingCosine > 1.0)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const auto argument = 2.0 * kPi * rmsHeightNanometres
                        * (incidentCosine + outgoingCosine)
                        / wavelengthNanometres;
    return std::exp(-argument * argument);
}

LinearSrgb xyzToLinearSrgbD65(CieXyz xyz) noexcept
{
    return {
        std::max(0.0, 3.2406 * xyz.x - 1.5372 * xyz.y - 0.4986 * xyz.z),
        std::max(0.0, -0.9689 * xyz.x + 1.8758 * xyz.y + 0.0415 * xyz.z),
        std::max(0.0, 0.0557 * xyz.x - 0.2040 * xyz.y + 1.0570 * xyz.z)
    };
}

double gaussianOrderLobeDensity(
    Direction centre, Direction outgoing,
    double angularStandardDeviationRadians) noexcept
{
    if (!finite(centre.x) || !finite(centre.y)
        || !finite(outgoing.x) || !finite(outgoing.y)
        || !finite(angularStandardDeviationRadians)
        || angularStandardDeviationRadians <= 0.0)
        return std::numeric_limits<double>::quiet_NaN();
    const auto deltaX = outgoing.x - centre.x;
    const auto deltaY = outgoing.y - centre.y;
    const auto variance = angularStandardDeviationRadians
                        * angularStandardDeviationRadians;
    return std::exp(-(deltaX * deltaX + deltaY * deltaY) / (2.0 * variance))
         / (2.0 * kPi * variance);
}

Description resolveLocalGrooveGeometry(const Description& source,
                                       const std::array<double, 2>& uv) noexcept
{
    auto local = source;
    const auto& field = source.grooveField;
    if (field.mode == GrooveFieldMode::Constant)
        return local;
    const auto dx = uv[0] - static_cast<double>(field.originUv[0]);
    const auto dy = uv[1] - static_cast<double>(field.originUv[1]);
    const auto coordinate = field.mode == GrooveFieldMode::Linear
        ? dx * field.axisUv[0] + dy * field.axisUv[1]
        : std::sqrt(dx * dx + dy * dy);
    local.geometry.grooveSpacingNanometres += static_cast<float>(
        coordinate * field.grooveSpacingDeltaNanometresPerUnit);
    if (local.geometry.lattice == GratingLattice::CrossedTwoDimensional)
        local.geometry.secondaryGrooveSpacingNanometres += static_cast<float>(
            coordinate * field.secondarySpacingDeltaNanometresPerUnit);
    const auto radians = coordinate * field.orientationDegreesPerUnit * kPi / 180.0;
    const auto cosine = static_cast<float>(std::cos(radians));
    const auto sine = static_cast<float>(std::sin(radians));
    const auto rotate = [cosine, sine](std::array<float, 2> value)
    {
        return std::array<float, 2> {
            cosine * value[0] - sine * value[1],
            sine * value[0] + cosine * value[1]
        };
    };
    local.geometry.directionUv = rotate(local.geometry.directionUv);
    if (local.geometry.lattice == GratingLattice::CrossedTwoDimensional)
        local.geometry.secondaryDirectionUv = rotate(
            local.geometry.secondaryDirectionUv);
    return local;
}

std::optional<DirectionalBsdfResult> evaluateDirectionalBsdf(
    const AdmittedDiffractionMaterialIR& material,
    const EvaluationInput& input,
    Direction outgoingDirection,
    EvaluationFailure& failure) noexcept
{
    auto aggregate = evaluate(material, input, failure);
    if (!aggregate)
        return std::nullopt;
    const auto outgoingLengthSquared = lengthSquared(outgoingDirection);
    if (!finite(outgoingLengthSquared)
        || outgoingLengthSquared <= std::numeric_limits<double>::min())
    {
        failure = EvaluationFailure::InvalidIncidentDirection;
        return std::nullopt;
    }
    const auto outgoing = normalize(outgoingDirection);
    if (outgoing.z <= 0.0)
    {
        failure = EvaluationFailure::InvalidIncidentDirection;
        return std::nullopt;
    }

    const auto& description = material.description();
    const auto incident = normalize(input.incidentDirection);
    const auto sampleCount = static_cast<std::size_t>(description.spectrum.wavelengthCount);
    double referenceWhiteY = 0.0;
    for (std::size_t index = 0; index < sampleCount; ++index)
    {
        const auto wavelength = description.spectrum.wavelengthsNanometres[index];
        referenceWhiteY += quadratureWeight(description.spectrum, index)
                         * cieAt(wavelength).y;
    }
    if (!(referenceWhiteY > std::numeric_limits<double>::min()))
    {
        failure = EvaluationFailure::InvalidSpectrum;
        return std::nullopt;
    }

    DirectionalBsdfResult result;
    for (std::size_t eventIndex = 0; eventIndex < aggregate->eventCount; ++eventIndex)
    {
        const auto& event = aggregate->events[eventIndex];
        const auto lobe = gaussianOrderLobeDensity(
            event.outgoingDirection, outgoing,
            event.angularStandardDeviationRadians);
        if (!finite(lobe))
        {
            failure = EvaluationFailure::EnergyBoundViolation;
            return std::nullopt;
        }
        const auto cie = cieAt(event.wavelengthNanometres);
        const auto radiance = event.energy * lobe * incident.z / referenceWhiteY;
        result.radianceXyz.x += radiance * cie.x;
        result.radianceXyz.y += radiance * cie.y;
        result.radianceXyz.z += radiance * cie.z;
    }
    result.outputLinearSrgb = xyzToLinearSrgbD65(result.radianceXyz);
    failure = EvaluationFailure::None;
    return result;
}

std::optional<EvaluationResult> evaluate(
    const AdmittedDiffractionMaterialIR& material,
    const EvaluationInput& input,
    EvaluationFailure& failure) noexcept
{
    failure = EvaluationFailure::None;
    const auto description = resolveLocalGrooveGeometry(
        material.description(), input.materialUv);
    const auto sampleCount = static_cast<std::size_t>(description.spectrum.wavelengthCount);
    if (!validInput(input, sampleCount, failure))
        return std::nullopt;
    const auto incident = normalize(input.incidentDirection);

    EvaluationResult result;
    result.spectralSampleCount = description.spectrum.wavelengthCount;
    std::array<double, kMaximumSpectralSamples> weights {};
    std::array<CieEntry, kMaximumSpectralSamples> cieSamples {};
    double referenceWhiteY = 0.0;
    for (std::size_t index = 0; index < sampleCount; ++index)
    {
        weights[index] = quadratureWeight(description.spectrum, index);
        cieSamples[index] = cieAt(description.spectrum.wavelengthsNanometres[index]);
        referenceWhiteY += weights[index] * cieSamples[index].y;
    }
    if (!(referenceWhiteY > std::numeric_limits<double>::min()))
    {
        failure = EvaluationFailure::InvalidSpectrum;
        return std::nullopt;
    }

    for (std::size_t index = 0; index < sampleCount; ++index)
    {
        const auto wavelength = static_cast<double>(
            description.spectrum.wavelengthsNanometres[index]);
        const auto firstEvent = result.eventCount;
        WavelengthFractions fractions;
        if (!wavelengthFractions(description, incident, wavelength, fractions,
                                 &result.evanescentOrderCount,
                                 &result.grazingOrderCount,
                                 result.events.data(), &result.eventCount,
                                 static_cast<std::uint8_t>(index)))
        {
            failure = EvaluationFailure::EnergyBoundViolation;
            return std::nullopt;
        }

        const auto weightedIncident = input.incidentSpectrum[index] * weights[index];
        const auto reflected = weightedIncident * fractions.reflectance;
        const auto resolved = reflected * fractions.resolved;
        const auto zero = reflected * fractions.zero;
        const auto higher = reflected * fractions.higher;
        auto& sample = result.spectralSamples[index];
        sample.wavelengthNanometres = wavelength;
        sample.quadratureWeight = weights[index];
        sample.incidentEnergy = weightedIncident;
        sample.substrateReflectedEnergy = reflected;
        sample.resolvedReflectedEnergy = resolved;
        sample.zeroOrderEnergy = zero;
        sample.higherOrderEnergy = higher;
        sample.unresolvedReflectedEnergy = std::max(0.0, reflected - resolved);
        sample.absorbedEnergy = std::max(0.0, weightedIncident - reflected);
        for (std::size_t event = firstEvent; event < result.eventCount; ++event)
            result.events[event].energy = weightedIncident * result.events[event].efficiency;
        addIntegratedSample(result, fractions, weightedIncident,
                            cieSamples[index], referenceWhiteY);
    }

    if (std::abs(result.substrateReflectedEnergy
                   - result.resolvedReflectedEnergy
                   - result.unresolvedReflectedEnergy) > kEnergyTolerance
        || std::abs(result.incidentEnergy
                   - result.substrateReflectedEnergy
                   - result.absorbedEnergy) > kEnergyTolerance
        || std::abs(result.resolvedReflectedEnergy
                   - result.zeroOrderEnergy
                   - result.higherOrderEnergy) > kEnergyTolerance)
    {
        failure = EvaluationFailure::EnergyBoundViolation;
        return std::nullopt;
    }
    finalizeColors(result);
    return result;
}

std::optional<HighSampleReferenceResult> evaluateHighSampleReference(
    const AdmittedDiffractionMaterialIR& material,
    const EvaluationInput& input,
    EvaluationFailure& failure) noexcept
{
    failure = EvaluationFailure::None;
    const auto description = resolveLocalGrooveGeometry(
        material.description(), input.materialUv);
    if (!validInput(input, description.spectrum.wavelengthCount, failure))
        return std::nullopt;
    const auto incident = normalize(input.incidentDirection);

    HighSampleReferenceResult result;
    double referenceWhiteY = 0.0;
    for (std::size_t index = 0; index < kHighSampleReferenceCount; ++index)
    {
        const auto endpoint = index == 0 || index + 1 == kHighSampleReferenceCount;
        const auto weight = (endpoint ? 2.5 : 5.0) / kVisibleRangeNanometres;
        referenceWhiteY += weight * kCie1931[index].y;
    }

    for (std::size_t index = 0; index < kHighSampleReferenceCount; ++index)
    {
        const auto wavelength = 380.0 + 5.0 * static_cast<double>(index);
        const auto endpoint = index == 0 || index + 1 == kHighSampleReferenceCount;
        const auto weight = (endpoint ? 2.5 : 5.0) / kVisibleRangeNanometres;
        WavelengthFractions fractions;
        if (!wavelengthFractions(description, incident, wavelength, fractions,
                                 nullptr, nullptr, nullptr, nullptr, 0))
        {
            failure = EvaluationFailure::EnergyBoundViolation;
            return std::nullopt;
        }
        const auto weightedIncident = interpolatedIncident(
            description.spectrum, input, wavelength) * weight;
        addIntegratedSample(result, fractions, weightedIncident,
                            kCie1931[index], referenceWhiteY);
    }

    if (std::abs(result.resolvedReflectedEnergy
                   - result.zeroOrderEnergy
                   - result.higherOrderEnergy) > 1.0e-9
        || std::abs(result.substrateReflectedEnergy
                   - result.resolvedReflectedEnergy
                   - result.unresolvedReflectedEnergy) > 1.0e-9)
    {
        failure = EvaluationFailure::EnergyBoundViolation;
        return std::nullopt;
    }
    finalizeColors(result);
    return result;
}
} // namespace diffractionmaterial::reference
