#include "diffraction_order_bsdf.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>

namespace diffractionmaterial
{
namespace
{
constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double directionTolerance = 1.0e-10;
constexpr double energyTolerance = 1.0e-10;
constexpr double probabilityTolerance = 1.0e-12;
constexpr double distributionConsistencyTolerance = 1.0e-10;

bool finiteVector(const std::array<double, 3>& value) noexcept
{
    return std::all_of(value.begin(), value.end(),
                       [] (double component) { return std::isfinite(component); });
}

double dot(const std::array<double, 3>& first,
           const std::array<double, 3>& second) noexcept
{
    return first[0] * second[0] + first[1] * second[1]
        + first[2] * second[2];
}

std::array<double, 3> cross(const std::array<double, 3>& first,
                            const std::array<double, 3>& second) noexcept
{
    return { first[1] * second[2] - first[2] * second[1],
             first[2] * second[0] - first[0] * second[2],
             first[0] * second[1] - first[1] * second[0] };
}

bool unit(const std::array<double, 3>& value) noexcept
{
    return finiteVector(value) && std::abs(dot(value, value) - 1.0) <= directionTolerance;
}

bool validFrame(const DiffractionLocalFrame& frame) noexcept
{
    if (!unit(frame.tangent) || !unit(frame.bitangent) || !unit(frame.normal)
        || std::abs(dot(frame.tangent, frame.bitangent)) > directionTolerance
        || std::abs(dot(frame.tangent, frame.normal)) > directionTolerance
        || std::abs(dot(frame.bitangent, frame.normal)) > directionTolerance)
        return false;
    const auto handedNormal = cross(frame.tangent, frame.bitangent);
    return dot(handedNormal, frame.normal) >= 1.0 - directionTolerance;
}

std::complex<double> exponentialIntegral(double frequency,
                                         double begin,
                                         double end) noexcept
{
    if (std::abs(frequency) < 1.0e-12)
        return { end - begin, 0.0 };
    return { (std::sin(frequency * end) - std::sin(frequency * begin)) / frequency,
             -(std::cos(frequency * end) - std::cos(frequency * begin)) / frequency };
}

double besselJ(int order, double argument) noexcept
{
    const int magnitude = std::abs(order);
    const double halfArgument = 0.5 * argument;
    double term = 1.0;
    for (int factor = 1; factor <= magnitude; ++factor)
        term *= halfArgument / static_cast<double>(factor);
    double sum = term;
    for (int index = 1; index <= 48; ++index)
    {
        term *= -(halfArgument * halfArgument)
            / (static_cast<double>(index) * static_cast<double>(magnitude + index));
        sum += term;
    }
    if (order < 0 && (magnitude & 1) != 0)
        sum = -sum;
    return sum;
}

double profileEfficiency(GrooveProfile profile,
                         int order,
                         double duty,
                         double phase) noexcept
{
    const std::complex<double> terrace(std::cos(phase), std::sin(phase));
    if (profile == GrooveProfile::BinaryRectangular)
    {
        if (order == 0)
            return std::norm((1.0 - duty) + duty * terrace);
        const double signedOrder = static_cast<double>(order);
        const double aperture = std::sin(pi * signedOrder * duty)
            / (pi * signedOrder);
        return std::norm(aperture * (terrace - 1.0)
                         * std::polar(1.0, -pi * signedOrder * duty));
    }
    if (profile == GrooveProfile::Sinusoidal)
    {
        const double amplitude = besselJ(order, 0.5 * phase);
        return amplitude * amplitude;
    }
    if (profile == GrooveProfile::BlazedSawtooth)
    {
        const double frequency = -2.0 * pi * static_cast<double>(order);
        return std::norm(exponentialIntegral(phase / duty + frequency, 0.0, duty)
                         + exponentialIntegral(frequency, duty, 1.0));
    }
    return -1.0;
}

double interfaceReflectance(std::complex<double> first,
                            std::complex<double> second) noexcept
{
    return std::clamp(std::norm(first - second) / std::norm(first + second), 0.0, 1.0);
}

double reflectance(const Description& description, double wavelength) noexcept
{
    const std::complex<double> air(1.0, 0.0);
    const std::complex<double> substrate(description.substrate.refractiveIndex,
                                         description.substrate.extinctionCoefficient);
    if (description.coating.model == CoatingModel::Uncoated)
        return interfaceReflectance(air, substrate);
    if (description.coating.model != CoatingModel::IncoherentDielectric)
        return -1.0;
    const std::complex<double> layer(description.coating.opticalConstants.refractiveIndex,
                                     description.coating.opticalConstants.extinctionCoefficient);
    const double r01 = interfaceReflectance(air, layer);
    const double r12 = interfaceReflectance(layer, substrate);
    const double roundTrip = std::exp(-8.0 * pi
        * description.coating.opticalConstants.extinctionCoefficient
        * description.coating.thicknessNanometres / wavelength);
    const double denominator = 1.0 - r01 * r12 * roundTrip;
    if (!(denominator > 0.0) || !std::isfinite(denominator))
        return -1.0;
    return std::clamp(r01 + (1.0 - r01) * (1.0 - r01)
        * r12 * roundTrip / denominator, 0.0, 1.0);
}

bool admittedOrder(int order, const SpectralSampling& spectrum) noexcept
{
    const int magnitude = std::abs(order);
    return order == 0 || (magnitude >= spectrum.firstOrder
                          && magnitude <= spectrum.lastOrder);
}
} // namespace

bool evaluateDiffractionOrders(const AdmittedDiffractionMaterialIR& material,
                               double wavelength,
                               const DiffractionLocalFrame& frame,
                               const std::array<double, 3>& incidentDirection,
                               const std::array<double, 2>& materialUv,
                               DiffractionOrderDistribution& result,
                               std::string& error) noexcept
{
    DiffractionOrderDistribution candidate;
    const auto fail = [&] (const char* message)
    {
        error = message;
        return false;
    };
    const auto& description = material.description();
    if (!std::isfinite(wavelength) || wavelength < kMinimumWavelengthNanometres
        || wavelength > kMaximumWavelengthNanometres)
        return fail("diffraction order BSDF wavelength is invalid");
    if (!validFrame(frame) || !unit(incidentDirection))
        return fail("diffraction order BSDF local frame is invalid");
    if (!std::isfinite(materialUv[0]) || !std::isfinite(materialUv[1])
        || materialUv[0] < 0.0 || materialUv[0] > 1.0
        || materialUv[1] < 0.0 || materialUv[1] > 1.0)
        return fail("diffraction order BSDF material coordinate is invalid");
    if (description.geometry.lattice != GratingLattice::OneDimensional
        && description.geometry.lattice != GratingLattice::CrossedTwoDimensional)
        return fail("diffraction order BSDF lattice is unsupported");
    if (description.microstructure.profile != GrooveProfile::BinaryRectangular
        && description.microstructure.profile != GrooveProfile::Sinusoidal
        && description.microstructure.profile != GrooveProfile::BlazedSawtooth)
        return fail("diffraction order BSDF profile is unsupported");
    if (description.microstructure.profile == GrooveProfile::Sinusoidal
        && 2.0 * pi * description.microstructure.grooveDepthNanometres
            / kMinimumWavelengthNanometres > 4.0)
        return fail("diffraction order BSDF sinusoidal phase exceeds the bounded model");

    const std::array<double, 3> incidentLocal {
        dot(incidentDirection, frame.tangent),
        dot(incidentDirection, frame.bitangent),
        dot(incidentDirection, frame.normal)
    };
    if (!(incidentLocal[2] > 1.0e-6))
        return fail("diffraction order BSDF incident direction is grazing or backfacing");

    double coordinate = 0.0;
    if (description.grooveField.mode == GrooveFieldMode::Linear)
    {
        coordinate = (materialUv[0] - description.grooveField.originUv[0])
                * description.grooveField.axisUv[0]
            + (materialUv[1] - description.grooveField.originUv[1])
                * description.grooveField.axisUv[1];
    }
    else if (description.grooveField.mode == GrooveFieldMode::Radial)
    {
        coordinate = std::hypot(materialUv[0] - description.grooveField.originUv[0],
                                materialUv[1] - description.grooveField.originUv[1]);
    }
    else if (description.grooveField.mode != GrooveFieldMode::Constant)
        return fail("diffraction order BSDF groove field is unsupported");

    const double angle = coordinate * description.grooveField.orientationDegreesPerUnit
        * pi / 180.0;
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    const auto rotate = [cosine, sine] (const std::array<float, 2>& direction)
    {
        return std::array<double, 2> {
            cosine * direction[0] - sine * direction[1],
            sine * direction[0] + cosine * direction[1]
        };
    };
    const auto primaryDirection = rotate(description.geometry.directionUv);
    const auto secondaryDirection = rotate(description.geometry.secondaryDirectionUv);
    const double primaryPeriod = description.geometry.grooveSpacingNanometres
        + coordinate * description.grooveField.grooveSpacingDeltaNanometresPerUnit;
    const double secondaryPeriod = description.geometry.secondaryGrooveSpacingNanometres
        + coordinate * description.grooveField.secondarySpacingDeltaNanometresPerUnit;
    const bool crossed = description.geometry.lattice == GratingLattice::CrossedTwoDimensional;
    if (!std::isfinite(primaryPeriod) || primaryPeriod < kMinimumGrooveSpacingNanometres
        || primaryPeriod > kMaximumGrooveSpacingNanometres
        || (crossed && (!std::isfinite(secondaryPeriod)
            || secondaryPeriod < kMinimumGrooveSpacingNanometres
            || secondaryPeriod > kMaximumGrooveSpacingNanometres)))
        return fail("diffraction order BSDF local grating period is invalid");

    const double substratePower = reflectance(description, wavelength);
    if (!std::isfinite(substratePower) || substratePower < 0.0 || substratePower > 1.0)
        return fail("diffraction order BSDF reflectance is invalid");

    const double depth = description.microstructure.grooveDepthNanometres;
    const double duty = description.microstructure.dutyCycle;
    const double roughnessSigma = 2.0 * description.roughness.rmsSlope;
    for (int primary = -static_cast<int>(kMaximumDiffractionOrder);
         primary <= static_cast<int>(kMaximumDiffractionOrder); ++primary)
    {
        if (!admittedOrder(primary, description.spectrum))
            continue;
        const int secondaryBegin = crossed ? -static_cast<int>(kMaximumDiffractionOrder) : 0;
        const int secondaryEnd = crossed ? static_cast<int>(kMaximumDiffractionOrder) : 0;
        for (int secondary = secondaryBegin; secondary <= secondaryEnd; ++secondary)
        {
            if (!admittedOrder(secondary, description.spectrum))
                continue;
            const double tangentX = -incidentLocal[0]
                + primary * wavelength / primaryPeriod * primaryDirection[0]
                + (crossed ? secondary * wavelength / secondaryPeriod
                     * secondaryDirection[0] : 0.0);
            const double tangentY = -incidentLocal[1]
                + primary * wavelength / primaryPeriod * primaryDirection[1]
                + (crossed ? secondary * wavelength / secondaryPeriod
                     * secondaryDirection[1] : 0.0);
            const double tangentSquared = tangentX * tangentX + tangentY * tangentY;
            if (!std::isfinite(tangentSquared) || tangentSquared >= 1.0 - 1.0e-12)
                continue;
            const double outgoingZ = std::sqrt(std::max(0.0, 1.0 - tangentSquared));
            const double phase = 2.0 * pi * depth
                * (incidentLocal[2] + outgoingZ) / wavelength;
            double efficiency = profileEfficiency(description.microstructure.profile,
                                                  primary, duty, phase);
            if (crossed)
                efficiency *= profileEfficiency(description.microstructure.profile,
                                                secondary, duty, phase);
            if (primary != 0 || secondary != 0)
                efficiency *= outgoingZ / incidentLocal[2];
            const double roughnessArgument = 2.0 * pi
                * description.roughness.rmsHeightNanometres
                * (incidentLocal[2] + outgoingZ) / wavelength;
            efficiency *= std::exp(-(roughnessArgument * roughnessArgument));
            const double power = substratePower * efficiency;
            if (!std::isfinite(power) || power < 0.0
                || candidate.orderCount >= candidate.orders.size())
                return fail("diffraction order BSDF produced invalid order power");
            const std::array<double, 3> direction {
                tangentX * frame.tangent[0] + tangentY * frame.bitangent[0]
                    + outgoingZ * frame.normal[0],
                tangentX * frame.tangent[1] + tangentY * frame.bitangent[1]
                    + outgoingZ * frame.normal[1],
                tangentX * frame.tangent[2] + tangentY * frame.bitangent[2]
                    + outgoingZ * frame.normal[2]
            };
            candidate.orders[candidate.orderCount++] = {
                static_cast<std::int8_t>(primary), static_cast<std::int8_t>(secondary),
                direction, power, 0.0, roughnessSigma
            };
            candidate.resolvedReflectedPower += power;
        }
    }
    if (candidate.orderCount == 0 || !(candidate.resolvedReflectedPower > 0.0)
        || !std::isfinite(candidate.resolvedReflectedPower))
        return fail("diffraction order BSDF has no resolved reflected energy");
    if (candidate.resolvedReflectedPower > substratePower * (1.0 + energyTolerance)
        || (substratePower == 0.0 && candidate.resolvedReflectedPower > 0.0))
        return fail("diffraction order BSDF resolved energy exceeds reflection");

    candidate.substrateReflectedPower = substratePower;
    candidate.unresolvedReflectedPower
        = std::max(0.0, substratePower - candidate.resolvedReflectedPower);
    double probabilitySum = 0.0;
    for (std::size_t index = 0; index < candidate.orderCount; ++index)
    {
        candidate.orders[index].selectionProbability
            = candidate.orders[index].reflectedPower / candidate.resolvedReflectedPower;
        probabilitySum += candidate.orders[index].selectionProbability;
    }
    // Canonical distributions use exact support: zero probability means zero
    // power, and positive probability means positive power. Put normalization
    // rounding in the last positive order so the bins cover all of [0, 1).
    std::size_t lastPositive = candidate.orderCount;
    for (std::size_t index = 0; index < candidate.orderCount; ++index)
        if (candidate.orders[index].selectionProbability > 0.0)
            lastPositive = index;
    if (lastPositive == candidate.orderCount)
        return fail("diffraction order BSDF has no sampleable reflected energy");
    candidate.orders[lastPositive].selectionProbability += 1.0 - probabilitySum;
    candidate.orders[lastPositive].reflectedPower
        = candidate.orders[lastPositive].selectionProbability
            * candidate.resolvedReflectedPower;
    if (!std::all_of(candidate.orders.begin(),
                     candidate.orders.begin() + static_cast<std::ptrdiff_t>(candidate.orderCount),
                     [] (const DiffractionOrder& order)
                     {
                         return unit(order.direction)
                             && std::isfinite(order.reflectedPower)
                             && order.reflectedPower >= 0.0
                             && std::isfinite(order.selectionProbability)
                             && order.selectionProbability >= 0.0
                             && ((order.selectionProbability == 0.0)
                                 == (order.reflectedPower == 0.0));
                     }))
        return fail("diffraction order BSDF produced an invalid distribution");

    result = candidate;
    error.clear();
    return true;
}

bool sampleDiffractionOrder(const DiffractionOrderDistribution& distribution,
                            double unitSample,
                            DiffractionOrderSample& result,
                            std::string& error) noexcept
{
    DiffractionOrderSample candidate;
    if (!std::isfinite(unitSample) || unitSample < 0.0 || unitSample >= 1.0)
    {
        error = "diffraction order BSDF sample must be in [0,1)";
        return false;
    }
    if (distribution.orderCount == 0
        || distribution.orderCount > distribution.orders.size()
        || !(distribution.resolvedReflectedPower > 0.0)
        || !std::isfinite(distribution.resolvedReflectedPower))
    {
        error = "diffraction order BSDF distribution is invalid";
        return false;
    }
    double cumulative = 0.0;
    double powerSum = 0.0;
    std::size_t lastPositive = distribution.orderCount;
    for (std::size_t index = 0; index < distribution.orderCount; ++index)
    {
        const auto& order = distribution.orders[index];
        if (!unit(order.direction)
            || !std::isfinite(order.selectionProbability)
            || order.selectionProbability < 0.0
            || !std::isfinite(order.reflectedPower) || order.reflectedPower < 0.0
            || !std::isfinite(order.angularStandardDeviationRadians)
            || order.angularStandardDeviationRadians < 0.0
            || order.selectionProbability > 1.0 + probabilityTolerance
            || cumulative > std::numeric_limits<double>::max()
                - order.selectionProbability
            || powerSum > std::numeric_limits<double>::max() - order.reflectedPower)
        {
            error = "diffraction order BSDF distribution is invalid";
            return false;
        }
        cumulative += order.selectionProbability;
        powerSum += order.reflectedPower;
        if (!std::isfinite(cumulative) || cumulative > 1.0 + probabilityTolerance
            || !std::isfinite(powerSum))
        {
            error = "diffraction order BSDF distribution is invalid";
            return false;
        }
        if ((order.selectionProbability == 0.0) != (order.reflectedPower == 0.0))
        {
            error = "diffraction order BSDF power and probability have different support";
            return false;
        }
        if (order.selectionProbability > 0.0)
            lastPositive = index;
        const double expectedPower
            = order.selectionProbability * distribution.resolvedReflectedPower;
        if (!std::isfinite(expectedPower)
            || std::abs(order.reflectedPower - expectedPower)
                > distributionConsistencyTolerance
                    * distribution.resolvedReflectedPower)
        {
            error = "diffraction order BSDF power and probability are inconsistent";
            return false;
        }
    }
    if (std::abs(cumulative - 1.0) > probabilityTolerance
        || lastPositive == distribution.orderCount)
    {
        error = "diffraction order BSDF probabilities are not normalized";
        return false;
    }
    if (std::abs(powerSum - distribution.resolvedReflectedPower)
        > distributionConsistencyTolerance * distribution.resolvedReflectedPower)
    {
        error = "diffraction order BSDF resolved power is inconsistent";
        return false;
    }

    // Canonicalize tolerated rounding onto the last positive bin. This absorbs
    // either a residual or an overshoot while earlier boundaries stay fixed.
    const double probabilityBeforeLast
        = cumulative - distribution.orders[lastPositive].selectionProbability;
    const double canonicalLastProbability = 1.0 - probabilityBeforeLast;
    if (!(canonicalLastProbability > 0.0))
    {
        error = "diffraction order BSDF probabilities are not normalized";
        return false;
    }

    // Positive-mass order i owns [sum(p[0..i-1]), sum(p[0..i])). Exact
    // boundaries belong to the following positive-mass order; zero-mass bins
    // own no samples. The canonical last positive bin closes at one.
    cumulative = 0.0;
    std::size_t selected = distribution.orderCount;
    for (std::size_t index = 0; index < distribution.orderCount; ++index)
    {
        const double probability = index == lastPositive
            ? canonicalLastProbability : distribution.orders[index].selectionProbability;
        cumulative += probability;
        if (unitSample < cumulative)
        {
            selected = index;
            break;
        }
    }
    if (selected == distribution.orderCount)
    {
        error = "diffraction order BSDF probabilities are not normalized";
        return false;
    }
    const auto& order = distribution.orders[selected];
    const double canonicalProbability = selected == lastPositive
        ? canonicalLastProbability : order.selectionProbability;
    if (!(canonicalProbability > 0.0))
    {
        error = "diffraction order BSDF selected a zero-probability order";
        return false;
    }
    candidate.order = order;
    candidate.order.selectionProbability = canonicalProbability;
    candidate.order.reflectedPower
        = canonicalProbability * distribution.resolvedReflectedPower;
    candidate.orderIndex = selected;
    candidate.pdf = canonicalProbability;
    candidate.throughput = candidate.order.reflectedPower / candidate.pdf;
    if (!std::isfinite(candidate.throughput) || candidate.throughput < 0.0)
    {
        error = "diffraction order BSDF sample throughput is invalid";
        return false;
    }
    result = candidate;
    error.clear();
    return true;
}
} // namespace diffractionmaterial
