#include "diffraction_order_bsdf.h"
#include "DiffractionMaterialPresets.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace
{
constexpr double pi = 3.141592653589793238462643383279502884;
// What a value from the float authority can round by against a double reference.
constexpr double kFloatTolerance = 16.0 * std::numeric_limits<float>::epsilon();
constexpr std::size_t phaseSamples = 65536;
constexpr std::size_t crossedAxisSamples = 1024;
constexpr std::uint32_t samplingTrials = 1048576;
int checks = 0;
int failures = 0;

void check(bool condition, const std::string& message)
{
    ++checks;
    if (!condition)
    {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

template <typename A, typename B>
double dot(const std::array<A, 3>& a, const std::array<B, 3>& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

std::array<float, 3> narrow(const std::array<double, 3>& v)
{
    return { static_cast<float>(v[0]), static_cast<float>(v[1]), static_cast<float>(v[2]) };
}

std::array<double, 3> normalized(std::array<double, 3> v)
{
    const double length = std::sqrt(dot(v, v));
    for (double& component : v)
        component /= length;
    return v;
}

double interfacePower(std::complex<double> first, std::complex<double> second)
{
    return std::clamp(std::norm(first - second) / std::norm(first + second), 0.0, 1.0);
}

double coatingPower(const diffractionmaterial::Description& material, double wavelength)
{
    const std::complex<double> air(1.0, 0.0);
    const std::complex<double> substrate(material.substrate.refractiveIndex,
                                         material.substrate.extinctionCoefficient);
    if (material.coating.model == diffractionmaterial::CoatingModel::Uncoated)
        return interfacePower(air, substrate);
    const std::complex<double> coating(material.coating.opticalConstants.refractiveIndex,
                                       material.coating.opticalConstants.extinctionCoefficient);
    const double first = interfacePower(air, coating);
    const double second = interfacePower(coating, substrate);
    const double attenuation = std::exp(-8.0 * pi
        * material.coating.opticalConstants.extinctionCoefficient
        * material.coating.thicknessNanometres / wavelength);
    return std::clamp(first + (1.0 - first) * (1.0 - first) * second * attenuation
        / (1.0 - first * second * attenuation), 0.0, 1.0);
}

double profilePhase(diffractionmaterial::GrooveProfile profile, double position,
                    double duty, double phase)
{
    if (profile == diffractionmaterial::GrooveProfile::BinaryRectangular)
        return position < duty ? phase : 0.0;
    if (profile == diffractionmaterial::GrooveProfile::Sinusoidal)
        return 0.5 * phase * std::cos(2.0 * pi * position);
    return position < duty ? phase * position / duty : 0.0;
}

std::complex<double> numericalCoefficient(diffractionmaterial::GrooveProfile profile,
                                          int order, double duty, double phase)
{
    std::complex<double> sum {};
    for (std::size_t index = 0; index < phaseSamples; ++index)
    {
        const double position = (static_cast<double>(index) + 0.5)
            / static_cast<double>(phaseSamples);
        sum += std::polar(1.0, profilePhase(profile, position, duty, phase)
            - 2.0 * pi * order * position);
    }
    return sum / static_cast<double>(phaseSamples);
}

// This is a direct midpoint integral over the unit cell. It does not call the
// one-dimensional oracle or form a product of two production-shaped results.
std::complex<double> numericalCoefficient2D(diffractionmaterial::GrooveProfile profile,
                                            int primary, int secondary,
                                            double duty, double phase,
                                            std::size_t axisSamples = crossedAxisSamples)
{
    std::complex<double> sum {};
    for (std::size_t yIndex = 0; yIndex < axisSamples; ++yIndex)
    {
        const double y = (static_cast<double>(yIndex) + 0.5)
            / static_cast<double>(axisSamples);
        for (std::size_t xIndex = 0; xIndex < axisSamples; ++xIndex)
        {
            const double x = (static_cast<double>(xIndex) + 0.5)
                / static_cast<double>(axisSamples);
            const double phaseAtCell = profilePhase(profile, x, duty, phase)
                + profilePhase(profile, y, duty, phase)
                - 2.0 * pi * (primary * x + secondary * y);
            sum += std::polar(1.0, phaseAtCell);
        }
    }
    const double samples = static_cast<double>(axisSamples * axisSamples);
    return sum / samples;
}

struct ReferenceOrder
{
    int primary = 0;
    int secondary = 0;
    std::array<double, 3> direction {};
    double power = 0.0;
    double probability = 0.0;
};

std::vector<ReferenceOrder> referenceDistribution(
    const diffractionmaterial::Description& material, double wavelength,
    const diffractionmaterial::DiffractionLocalFrame& frame,
    const std::array<double, 3>& incidentWorld,
    const std::array<double, 2>& uv)
{
    std::vector<ReferenceOrder> result;
    const bool crossed = material.geometry.lattice
        == diffractionmaterial::GratingLattice::CrossedTwoDimensional;
    const auto admitted = [&] (int order)
    {
        const int magnitude = std::abs(order);
        return order == 0 || (magnitude >= material.spectrum.firstOrder
                              && magnitude <= material.spectrum.lastOrder);
    };
    const std::array<double, 3> incoming {
        dot(incidentWorld, frame.tangent), dot(incidentWorld, frame.bitangent),
        dot(incidentWorld, frame.normal)
    };
    double fieldCoordinate = 0.0;
    if (material.grooveField.mode == diffractionmaterial::GrooveFieldMode::Linear)
        fieldCoordinate = (uv[0] - material.grooveField.originUv[0])
                * material.grooveField.axisUv[0]
            + (uv[1] - material.grooveField.originUv[1])
                * material.grooveField.axisUv[1];
    else if (material.grooveField.mode == diffractionmaterial::GrooveFieldMode::Radial)
        fieldCoordinate = std::hypot(uv[0] - material.grooveField.originUv[0],
                                     uv[1] - material.grooveField.originUv[1]);
    const double rotation = fieldCoordinate
        * material.grooveField.orientationDegreesPerUnit * pi / 180.0;
    const auto rotated = [rotation] (const std::array<float, 2>& axis)
    {
        return std::array<double, 2> {
            std::cos(rotation) * axis[0] - std::sin(rotation) * axis[1],
            std::sin(rotation) * axis[0] + std::cos(rotation) * axis[1]
        };
    };
    const auto reciprocalA = rotated(material.geometry.directionUv);
    const auto reciprocalB = rotated(material.geometry.secondaryDirectionUv);
    const double periodA = material.geometry.grooveSpacingNanometres
        + fieldCoordinate * material.grooveField.grooveSpacingDeltaNanometresPerUnit;
    const double periodB = material.geometry.secondaryGrooveSpacingNanometres
        + fieldCoordinate * material.grooveField.secondarySpacingDeltaNanometresPerUnit;
    const double reflected = coatingPower(material, wavelength);
    double total = 0.0;
    for (int primary = -diffractionmaterial::kMaximumDiffractionOrder;
         primary <= diffractionmaterial::kMaximumDiffractionOrder; ++primary)
    {
        if (!admitted(primary))
            continue;
        const int secondaryBegin = crossed ? -diffractionmaterial::kMaximumDiffractionOrder : 0;
        const int secondaryEnd = crossed ? diffractionmaterial::kMaximumDiffractionOrder : 0;
        for (int secondary = secondaryBegin; secondary <= secondaryEnd; ++secondary)
        {
            if (!admitted(secondary))
                continue;
            const double localX = -incoming[0] + primary * wavelength / periodA * reciprocalA[0]
                + (crossed ? secondary * wavelength / periodB * reciprocalB[0] : 0.0);
            const double localY = -incoming[1] + primary * wavelength / periodA * reciprocalA[1]
                + (crossed ? secondary * wavelength / periodB * reciprocalB[1] : 0.0);
            const double transverse = localX * localX + localY * localY;
            if (transverse >= 1.0 - 1.0e-12)
                continue;
            const double localZ = std::sqrt(1.0 - transverse);
            const double phase = 2.0 * pi * material.microstructure.grooveDepthNanometres
                * (incoming[2] + localZ) / wavelength;
            const auto coefficient = crossed
                ? numericalCoefficient2D(material.microstructure.profile, primary, secondary,
                                         material.microstructure.dutyCycle, phase)
                : numericalCoefficient(material.microstructure.profile, primary,
                                       material.microstructure.dutyCycle, phase);
            double efficiency = std::norm(coefficient);
            if (primary != 0 || secondary != 0)
                efficiency *= localZ / incoming[2];
            const double roughnessPhase = 2.0 * pi * material.roughness.rmsHeightNanometres
                * (incoming[2] + localZ) / wavelength;
            const double power = reflected * efficiency
                * std::exp(-(roughnessPhase * roughnessPhase));
            // Build the world vector component by component from the supplied basis.
            const std::array<double, 3> world {
                frame.tangent[0] * localX + frame.bitangent[0] * localY + frame.normal[0] * localZ,
                frame.tangent[1] * localX + frame.bitangent[1] * localY + frame.normal[1] * localZ,
                frame.tangent[2] * localX + frame.bitangent[2] * localY + frame.normal[2] * localZ
            };
            result.push_back({ primary, secondary, world, power, 0.0 });
            total += power;
        }
    }
    for (auto& order : result)
        order.probability = order.power / total;
    return result;
}

std::uint32_t reverseBits(std::uint32_t value)
{
    value = ((value & 0x55555555u) << 1) | ((value >> 1) & 0x55555555u);
    value = ((value & 0x33333333u) << 2) | ((value >> 2) & 0x33333333u);
    value = ((value & 0x0f0f0f0fu) << 4) | ((value >> 4) & 0x0f0f0f0fu);
    return (value << 24) | ((value & 0xff00u) << 8)
        | ((value >> 8) & 0xff00u) | (value >> 24);
}

double lowDiscrepancySample(std::uint32_t index)
{
    return std::ldexp(static_cast<double>(reverseBits(index)), -32);
}

void runCase(diffractionmaterial::Description description, double wavelength,
             const diffractionmaterial::DiffractionLocalFrame& frame,
             const std::array<double, 3>& incident, const std::array<double, 2>& uv,
             const char* label)
{
    using namespace diffractionmaterial;
    std::string error;
    const auto admitted = admit(description, error);
    check(admitted.has_value(), std::string(label) + " admits");
    if (!admitted)
        return;
    DiffractionOrderDistribution production;
    check(evaluateDiffractionOrders(*admitted, static_cast<float>(wavelength), frame, narrow(incident),
                                    { static_cast<float>(uv[0]), static_cast<float>(uv[1]) },
                                    production, error),
          std::string(label) + " evaluates: " + error);
    if (production.orderCount == 0)
        return;
    const auto reference = referenceDistribution(description, wavelength, frame, incident, uv);
    check(reference.size() == production.orderCount,
          std::string(label) + " propagating order count agrees");
    if (reference.size() != production.orderCount)
        return;

    double probabilitySum = 0.0;
    double referenceResolved = 0.0;
    bool asymmetricPropagation = false;
    for (std::size_t index = 0; index < reference.size(); ++index)
    {
        const auto& expected = reference[index];
        const auto& actual = production.orders[index];
        check(actual.primary == expected.primary && actual.secondary == expected.secondary,
              std::string(label) + " order identity agrees");
        double directionGap = 0.0;
        for (std::size_t a = 0; a < 3; ++a)
            directionGap = std::max(directionGap, std::abs(actual.direction[a] - expected.direction[a]));
        check(directionGap <= kFloatTolerance,
              std::string(label) + " independently built world direction agrees");
        const double powerError = std::abs(actual.reflectedPower - expected.power);
        const double relativeTolerance = description.geometry.lattice
                == GratingLattice::CrossedTwoDimensional ? 2.0e-4 : 5.0e-5;
        check(powerError <= 1.0e-8
                  || powerError / std::max(expected.power, 1.0e-30) <= relativeTolerance,
              std::string(label) + " numerical phase power tolerance");
        probabilitySum += actual.selectionProbability;
        referenceResolved += expected.power;
        asymmetricPropagation |= actual.primary > 0;
    }
    if (std::abs(dot(incident, frame.normal) - 1.0) > 1.0e-4)
    {
        bool asymmetricPair = false;
        for (std::size_t first = 0; first < production.orderCount; ++first)
        {
            for (std::size_t second = first + 1; second < production.orderCount; ++second)
            {
                const auto& a = production.orders[first];
                const auto& b = production.orders[second];
                if (a.primary == -b.primary && a.secondary == -b.secondary
                    && std::abs(a.reflectedPower - b.reflectedPower)
                        > 1.0e-8 * std::max(a.reflectedPower, b.reflectedPower))
                    asymmetricPair = true;
            }
        }
        check(asymmetricPair,
              std::string(label) + " oblique case has asymmetric opposite orders");
    }
    check(asymmetricPropagation || production.orderCount == 1,
          std::string(label) + " has a positive or sole propagating order");
    check(std::abs(probabilitySum - 1.0) <= ordercore::probabilityTolerance,
          std::string(label) + " probabilities normalize");
    check(std::abs(production.resolvedReflectedPower + production.unresolvedReflectedPower
                   - production.substrateReflectedPower)
              <= ordercore::distributionConsistencyTolerance * std::max(1.0, static_cast<double>(production.substrateReflectedPower)),
          std::string(label) + " reflected energy ledger closes");

    std::vector<std::uint32_t> counts(production.orderCount, 0u);
    double energyEstimate = 0.0;
    for (std::uint32_t trial = 0; trial < samplingTrials; ++trial)
    {
        DiffractionOrderSample sample;
        if (!sampleDiffractionOrder(production, lowDiscrepancySample(trial), sample, error))
        {
            check(false, std::string(label) + " sampler accepts full sample-count oracle");
            return;
        }
        ++counts[sample.orderIndex];
        energyEstimate += sample.throughput;
    }
    energyEstimate /= samplingTrials;
    for (std::size_t index = 0; index < reference.size(); ++index)
    {
        if (reference[index].probability < 1.0e-4)
            continue;
        const double observed = static_cast<double>(counts[index]) / samplingTrials;
        check(std::abs(observed - reference[index].probability) <= 2.0e-4,
              std::string(label) + " sampling frequency tolerance");
    }
    check(std::abs(energyEstimate - referenceResolved)
              / std::max(referenceResolved, 1.0e-30) <= 1.0e-3,
          std::string(label) + " sampled energy tolerance");
}

// Each wavelength and view offset the disc can reach, through the two calls the
// integrator makes. A uniform sky returns resolved plus unresolved power, whichever
// order is drawn, so a refused sample shows as that wavelength's loss.
// Offsets past 0.8 are the steep views where the scalar orders sum above the metal's
// reflectance; they must still return exactly that reflectance.
void perWavelengthFurnaceTests()
{
    using namespace diffractionmaterial;
    const DiffractionLocalFrame frame { { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 0.0, 0.0, 1.0 } };
    for (const float depth : { 30.0f, 70.0f, 100.0f })
    {
        auto description = makeAluminiumBinaryGratingPreset();
        description.geometry.directionUv = { 1.0f, 0.0f };
        description.geometry.grooveSpacingNanometres = 1400.0f;
        description.microstructure.dutyCycle = 0.5f;
        description.microstructure.grooveDepthNanometres = depth;
        std::string error;
        const auto admitted = admit(description, error);
        check(admitted.has_value(), "the furnace grating admits: " + error);
        if (!admitted)
            continue;
        int refused = 0, wrongEnergy = 0;
        for (int nanometres = 380; nanometres <= 780; nanometres += 5)
            for (int percent = 0; percent <= 98; percent += 2)
            {
                const double offset = percent / 100.0;
                const auto incident = narrow({ -offset, 0.0, std::sqrt(1.0 - offset * offset) });
                DiffractionOrderDistribution distribution;
                if (!evaluateDiffractionOrders(*admitted, nanometres, frame, incident, { 0.5, 0.5 },
                                               distribution, error))
                {
                    ++refused;
                    continue;
                }
                double returned = distribution.unresolvedReflectedPower;
                for (std::uint32_t trial = 0; trial < 64; ++trial)
                {
                    DiffractionOrderSample sample;
                    if (!sampleDiffractionOrder(distribution, lowDiscrepancySample(trial), sample, error))
                        ++refused;
                    else
                        returned += sample.throughput / 64.0;
                }
                if (std::abs(returned - distribution.substrateReflectedPower) > 1.0e-2)
                    ++wrongEnergy;
            }
        check(refused == 0, "every wavelength and offset evaluates and samples, refused "
                                + std::to_string(refused) + " at depth " + std::to_string(depth));
        check(wrongEnergy == 0, "every wavelength returns the metal's reflectance, wrong at "
                                    + std::to_string(wrongEnergy) + " points at depth " + std::to_string(depth));
    }
}

// The measured table reaches the path's wavelength, and a material without it stays constant.
void measuredAluminiumTests()
{
    using namespace diffractionmaterial;
    const DiffractionLocalFrame frame { { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 0.0, 0.0, 1.0 } };
    auto description = makeAluminiumBinaryGratingPreset();
    description.geometry.directionUv = { 1.0f, 0.0f };
    description.geometry.grooveSpacingNanometres = 1400.0f;
    std::string error;
    const auto constant = admit(description, error);
    check(constant.has_value(), "the constant grating admits: " + error);
    description.substrateData = SubstrateData::AluminiumMcPeak2015;
    const auto measured = admit(description, error);
    check(measured.has_value(), "the measured grating admits: " + error);
    if (!constant || !measured)
        return;
    check(constant->structuralDigest() != measured->structuralDigest(),
          "the table is part of the material's identity");
    const std::array<float, 3> incident { 0.0f, 0.0f, 1.0f };
    const auto power = [&](const auto& material, double nanometres)
    {
        DiffractionOrderDistribution distribution;
        check(evaluateDiffractionOrders(material, nanometres, frame, incident, { 0.5, 0.5 }, distribution, error),
              "the evaluator takes " + std::to_string(nanometres) + " nm: " + error);
        return distribution.substrateReflectedPower;
    };
    // Normal-incidence Fresnel from the source file's own rows at these wavelengths.
    for (const auto& row : std::array<std::array<double, 2>, 4> {
             { { 380.0, 0.9245 }, { 550.0, 0.9157 }, { 700.0, 0.8951 }, { 780.0, 0.8692 } } })
        check(std::abs(power(*measured, row[0]) - row[1]) < 5.0e-4,
              "measured reflectance at " + std::to_string(row[0]) + " nm");
    // 545 lies between table rows; the source's 5 nm row there gives 0.9160.
    check(std::abs(power(*measured, 545.0) - 0.9160) < 5.0e-4, "the table interpolates between rows");
    check(power(*measured, 380.0) > power(*measured, 780.0) + 0.05, "the metal falls toward the near infrared");
    check(power(*constant, 380.0) == power(*constant, 780.0), "constants stay flat across wavelength");
    description.substrateData = static_cast<SubstrateData>(2);
    check(!admit(description, error).has_value(), "unknown substrate data is refused");
}

// Orders in a medium: the grating's step is lambda / (n d), and n sin(theta) is the same as
// in air, so Snell at the top surface returns the air grating's directions.
void orderMediumTests()
{
    using namespace diffractionmaterial;
    const DiffractionLocalFrame frame { { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 0.0, 0.0, 1.0 } };
    auto description = makeAluminiumBinaryGratingPreset();
    description.geometry.directionUv = { 1.0f, 0.0f };
    description.geometry.grooveSpacingNanometres = 1400.0f;
    description.microstructure.grooveDepthNanometres = 70.0f;
    description.substrateData = SubstrateData::AluminiumMcPeak2015;
    description.orderMedium = OrderMedium::SurroundingMedium;
    description.spectrum.lastOrder = 3;
    std::string error;
    const auto admitted = admit(description, error);
    check(admitted.has_value(), "a grating in a medium admits: " + error);
    if (!admitted)
        return;
    const std::array<float, 3> incident { 0.0f, 0.0f, 1.0f };
    constexpr double index = 1.5893;
    DiffractionOrderDistribution air, inside;
    check(evaluateDiffractionOrders(*admitted, 550.0, frame, incident, { 0.5, 0.5 }, air, error), error);
    check(evaluateDiffractionOrders(*admitted, 550.0, frame, incident, { 0.5, 0.5 }, inside, error, index), error);
    const auto orderOf = [](const DiffractionOrderDistribution& distribution, int order) -> const DiffractionOrder*
    {
        for (std::size_t at = 0; at < distribution.orderCount; ++at)
            if (distribution.orders[at].primary == order)
                return &distribution.orders[at];
        return nullptr;
    };
    const auto* airOne = orderOf(air, 1);
    const auto* insideOne = orderOf(inside, 1);
    check(airOne != nullptr && insideOne != nullptr, "order 1 propagates on both sides");
    if (airOne != nullptr && insideOne != nullptr)
    {
        check(std::abs(std::abs(airOne->direction[0]) - 550.0 / 1400.0) < kFloatTolerance, "in air sin(theta) is lambda / d");
        check(std::abs(std::abs(insideOne->direction[0]) * index - 550.0 / 1400.0) < kFloatTolerance,
              "in the medium n sin(theta) is lambda / d, so it leaves at the air grating's angle");
    }
    // Order 3 at 550 nm: sin(theta) 1.18 in air, so absent, and 0.74 inside, so present.
    check(orderOf(air, 3) == nullptr && orderOf(inside, 3) != nullptr, "the medium admits an order air cannot carry");
    // Metal against the medium: |(n - m) / (n + m)|^2 with m = 0.7894 + 5.8519i.
    check(std::abs(inside.substrateReflectedPower - 0.874) < 1.0e-3, "the metal reflects less against plastic");
    // Binary depth 70 at normal incidence: phase 4 pi 70 n / lambda = 2.54 rad,
    // so order 0 is cos^2(1.27) = 0.087 of the metal, against 0.486 in air.
    const auto* airZero = orderOf(air, 0);
    const auto* insideZero = orderOf(inside, 0);
    check(airZero != nullptr && insideZero != nullptr, "order 0 exists on both sides");
    if (airZero != nullptr && insideZero != nullptr)
    {
        check(std::abs(insideZero->reflectedPower / inside.substrateReflectedPower - 0.0873) < 2.0e-3,
              "the groove's optical depth is n times its depth");
        check(std::abs(airZero->reflectedPower / air.substrateReflectedPower - 0.4857) < 2.0e-3,
              "in air the groove's depth is its own");
    }
    check(!evaluateDiffractionOrders(*admitted, 550.0, frame, incident, { 0.5, 0.5 }, inside, error, 0.9),
          "an index below 1 is refused");
    DiffractionOrderDistribution untouched;
    const auto plain = [&] { auto copy = description; copy.orderMedium = OrderMedium::Air; return admit(copy, error); }();
    check(plain.has_value() && plain->structuralDigest() != admitted->structuralDigest(),
          "the medium is part of the material's identity");
    check(plain.has_value() && evaluateDiffractionOrders(*plain, 550.0, frame, incident, { 0.5, 0.5 }, untouched, error)
              && untouched.substrateReflectedPower == air.substrateReflectedPower,
          "with index 1 the evaluator computes what air always did");
    auto unknown = description;
    unknown.orderMedium = static_cast<OrderMedium>(2);
    check(!admit(unknown, error).has_value(), "an unknown order medium is refused");
}

void pressedDiscPitTests()
{
    using namespace diffractionmaterial;
    const DiffractionLocalFrame frame { { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 0.0, 0.0, 1.0 } };
    auto description = makeAluminiumBinaryGratingPreset();
    description.geometry.directionUv = { 1.0f, 0.0f };
    description.geometry.grooveSpacingNanometres = 1600.0f;
    description.microstructure.grooveDepthNanometres = 124.1f;
    description.microstructure.dutyCycle = 0.3125f;
    description.substrateData = SubstrateData::AluminiumMcPeak2015;
    description.orderMedium = OrderMedium::SurroundingMedium;
    description.pitStatistics = PitStatistics::PressedDisc;
    description.spectrum.lastOrder = 3;
    std::string error;
    const auto admitted = admit(description, error);
    check(admitted.has_value(), "a pressed disc track admits: " + error);
    if (!admitted)
        return;
    const std::array<float, 3> incident { 0.0f, 0.0f, 1.0f };
    constexpr double index = 1.5893;
    DiffractionOrderDistribution pits;
    check(evaluateDiffractionOrders(*admitted, 550.0, frame, incident, { 0.5, 0.5 }, pits, error, index), error);
    const auto share = [](const DiffractionOrderDistribution& distribution, int order)
    {
        for (std::size_t at = 0; at < distribution.orderCount; ++at)
            if (distribution.orders[at].primary == order)
                return distribution.orders[at].reflectedPower / distribution.substrateReflectedPower;
        return -1.0f;
    };
    // 550 nm, n 1.5893, depth 124.1: phase 4 pi 124.1 n / 550 = 4.506 rad at order 0, and
    // qW = 0.15625. Orders past 0 add their cosine, in the phase and as the power's factor.
    check(std::abs(share(pits, 0) - 0.6824) < 1.0e-3, "order 0 is |1 + (e^(i phase) - 1) q W|^2");
    check(std::abs(share(pits, 1) - 0.04296) < 1.0e-3, "order 1 follows the pit width's sinc");
    check(std::abs(share(pits, -1) - 0.04296) < 1.0e-3, "order -1 mirrors order 1 at normal incidence");
    check(std::abs(share(pits, 2) - 0.01379) < 1.0e-3, "order 2 is not zero, as a pit is not half the pitch");
    check(std::abs(share(pits, 3) - 0.00027) < 1.0e-3, "order 3 is near the sinc's zero at m W = 1");
    // W q (1 - q) |e^(i phase) - 1|^2 = 0.3125 x 0.25 x 2.41 = 0.188 of the metal forms no
    // order, so at least that much stays out of the resolved power.
    const double unresolved = 1.0 - pits.resolvedReflectedPower / pits.substrateReflectedPower;
    check(std::abs(unresolved - 0.2036) < 2.0e-3 && unresolved > 0.188, "the pits' noise forms no order");
    check(std::abs(pits.pitScatterPower / pits.substrateReflectedPower - 0.1882) < 1.0e-3, "that noise is the pit scatter");
    // The track orders and the pit scatter are one budget, so the lobe's power is not
    // counted twice. The orders past 20000 hold about 5e-6.
    for (const auto [nanometres, n] : { std::pair<double, double> { 450.0, 1.6074 }, { 550.0, index }, { 650.0, 1.5792 } })
    {
        DiffractionOrderDistribution at;
        check(evaluateDiffractionOrders(*admitted, nanometres, frame, incident, { 0.5, 0.5 }, at, error, n), error);
        const float phase = static_cast<float>(4.0 * 3.14159265358979323846 * 124.1 * n / nanometres);
        // W q (1 - q) |e^(i phase) - 1|^2, by Parseval.
        const double sine = std::sin(0.5 * phase);
        const double parseval = 0.3125 * sine * sine;
        double total = parseval;
        for (int order = -20000; order <= 20000; ++order)
            total += ordercore::pressedDiscEfficiency(order, 0.3125f, phase);
        check(std::abs(total - 1.0) < 1.0e-4, "orders plus pit scatter are the metal's power at " + std::to_string(nanometres) + " nm, " + std::to_string(total));
        // Each resolved order's phase uses its own outgoing cosine, which moves its power
        // off the normal-incidence figure, and orders past lastOrder stay unresolved. Where
        // the first outweighs the second, as at 450 nm, the scatter is what is left: 0.00032.
        const double scatter = at.pitScatterPower / at.substrateReflectedPower;
        double gained = 0.0, listed = 0.0;
        for (std::size_t i = 0; i < at.orderCount; ++i)
        {
            const float normal = ordercore::pressedDiscEfficiency(at.orders[i].primary, 0.3125f, phase);
            gained += at.orders[i].reflectedPower / at.substrateReflectedPower - normal;
            listed += normal;
        }
        const double shortfall = std::max(0.0, gained - (total - parseval - listed));
        check(std::abs(scatter - (parseval - shortfall)) < 1.0e-5, "the compiled pit scatter is that term less the orders' gain at " + std::to_string(nanometres) + " nm, " + std::to_string(scatter));
        check(nanometres != 450.0 || std::abs(shortfall - 0.00032) < 1.0e-5, "450 nm gives up the derived 0.00032, " + std::to_string(shortfall));
        check(at.resolvedReflectedPower + at.pitScatterPower <= at.substrateReflectedPower * 1.000001f, "orders and pit scatter stay inside the metal's power");
    }
    // lambda / n / w = 550 / 1.5893 / 500, and lambda / n / (pi 1354).
    check(std::abs(pits.pitAcrossWidth - 0.6921) < 1.0e-3, "across tracks it spreads by the pit's width");
    check(std::abs(pits.pitAlongWidth - 0.08135) < 1.0e-4, "along tracks it spreads by the mean run");
    check(std::abs(std::abs(pits.pitAcrossAxis[0]) - 1.0) < 1.0e-6 && std::abs(pits.pitAlongAxis[1] - pits.pitAcrossAxis[0]) < 1.0e-6,
          "the scatter's axes are the track's axes");
    auto stripes = description;
    stripes.pitStatistics = PitStatistics::None;
    const auto plain = admit(stripes, error);
    DiffractionOrderDistribution continuous;
    check(plain.has_value() && plain->structuralDigest() != admitted->structuralDigest(),
          "pit statistics are part of the material's identity");
    // A continuous groove of the same shape: 1 - 4 W (1 - W) sin^2(phase / 2) = 0.483.
    check(plain.has_value()
              && evaluateDiffractionOrders(*plain, 550.0, frame, incident, { 0.5, 0.5 }, continuous, error, index)
              && std::abs(share(continuous, 0) - 0.483) < 2.0e-3,
          "without pit statistics the groove computes what it always did");
    check(continuous.pitScatterPower == 0.0, "without pit statistics nothing scatters");
    auto crossed = makeCrossedTwoDimensionalGratingPreset();
    crossed.pitStatistics = PitStatistics::PressedDisc;
    check(!admit(crossed, error).has_value(), "pit statistics refuse a crossed lattice");
    auto sinusoidal = description;
    sinusoidal.microstructure.profile = GrooveProfile::Sinusoidal;
    check(!admit(sinusoidal, error).has_value(), "pit statistics refuse a non-binary profile");
    auto unknown = description;
    unknown.pitStatistics = static_cast<PitStatistics>(2);
    check(!admit(unknown, error).has_value(), "unknown pit statistics are refused");
}

void boundaryAndMalformedTests()
{
    using namespace diffractionmaterial;
    DiffractionOrderDistribution valid;
    valid.orderCount = 5;
    valid.resolvedReflectedPower = 10.0;
    const std::array<float, 5> probabilities { 0.2f, 0.0f, 0.3f, 0.5f, 0.0f };
    for (std::size_t index = 0; index < valid.orderCount; ++index)
    {
        valid.orders[index].direction = { 0.0, 0.0, 1.0 };
        valid.orders[index].selectionProbability = probabilities[index];
        valid.orders[index].reflectedPower = 10.0f * probabilities[index];
        valid.orders[index].angularStandardDeviationRadians = 0.01;
    }
    std::string error;
    DiffractionOrderSample sample;
    const auto owns = [&] (float u, std::size_t index, const char* label)
    {
        check(sampleDiffractionOrder(valid, u, sample, error) && sample.orderIndex == index,
              label);
    };
    owns(0.0, 0, "u=0 belongs to the first positive bin");
    owns(std::nextafter(0.2f, 0.0f), 0, "value below first boundary stays in first bin");
    owns(0.2f, 2, "exact first boundary skips zero-probability interior bin");
    owns(std::nextafter(0.2f, 1.0f), 2, "value above first boundary enters next positive bin");
    owns(std::nextafter(0.5f, 0.0f), 2, "value below second boundary stays in middle bin");
    owns(0.5f, 3, "exact second boundary belongs to following bin");
    owns(std::nextafter(0.5f, 1.0f), 3, "value above second boundary stays in following bin");
    owns(std::nextafter(1.0f, 0.0f), 3, "largest unit sample skips trailing zero bin");

    for (const float residual : { -0.5f * ordercore::probabilityTolerance, 0.5f * ordercore::probabilityTolerance })
    {
        auto rounded = valid;
        rounded.orders[3].selectionProbability += residual;
        rounded.orders[3].reflectedPower += residual * rounded.resolvedReflectedPower;
        check(sampleDiffractionOrder(rounded, std::nextafter(1.0f, 0.0f), sample, error)
                  && sample.orderIndex == 3
                  && std::abs(sample.pdf - 0.5f) <= std::numeric_limits<float>::epsilon()
                  && sample.order.selectionProbability == sample.pdf
                  && sample.order.reflectedPower
                      == sample.pdf * rounded.resolvedReflectedPower,
              residual < 0.0f
                  ? "low tolerance edge closes canonically at one"
                  : "high tolerance edge closes canonically at one");
    }

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const float huge = std::numeric_limits<float>::max();
    struct MalformedCase
    {
        const char* label;
        std::function<void(DiffractionOrderDistribution&)> mutate;
    };
    const std::vector<MalformedCase> cases {
        { "NaN trailing probability rejected", [=] (auto& d) { d.orders[4].selectionProbability = nan; } },
        { "infinite trailing power rejected", [=] (auto& d) { d.orders[4].reflectedPower = inf; } },
        { "huge probability rejected without cumulative overflow", [=] (auto& d) { d.orders[3].selectionProbability = huge; } },
        { "low total probability rejected", [] (auto& d) { d.orders[3].selectionProbability -= 0.01; d.orders[3].reflectedPower -= 0.1; } },
        { "high total probability rejected", [] (auto& d) { d.orders[3].selectionProbability += 0.01; d.orders[3].reflectedPower += 0.1; } },
        { "per-order power and PMF mismatch rejected", [] (auto& d) { d.orders[2].reflectedPower += 0.01; } },
        { "tiny positive power in interior zero bin rejected", [] (auto& d) { d.orders[1].reflectedPower = 1.0e-15; d.orders[3].reflectedPower -= 1.0e-15; } },
        { "tiny positive power in trailing zero bin rejected", [] (auto& d) { d.orders[4].reflectedPower = 1.0e-15; d.orders[3].reflectedPower -= 1.0e-15; } },
        { "compensated positive-bin mismatch rejected", [] (auto& d) { d.orders[0].reflectedPower += 1.0e-3f; d.orders[3].reflectedPower -= 1.0e-3f; } },
        { "resolved-power mismatch rejected", [] (auto& d) { d.resolvedReflectedPower = 11.0; } },
        { "NaN direction rejected", [=] (auto& d) { d.orders[4].direction[0] = nan; } },
        { "non-unit trailing direction rejected", [] (auto& d) { d.orders[4].direction = { 0.0, 0.0, 2.0 }; } },
        { "negative angular width rejected", [] (auto& d) { d.orders[4].angularStandardDeviationRadians = -1.0; } },
        { "infinite angular width rejected", [=] (auto& d) { d.orders[4].angularStandardDeviationRadians = inf; } },
        { "negative probability rejected", [] (auto& d) { d.orders[1].selectionProbability = -0.1; } },
        { "negative power rejected", [] (auto& d) { d.orders[1].reflectedPower = -0.1; } }
    };
    for (const auto& malformedCase : cases)
    {
        auto malformed = valid;
        malformedCase.mutate(malformed);
        check(!sampleDiffractionOrder(malformed, 0.0, sample, error), malformedCase.label);
    }

    std::uint64_t state = 0x9e3779b97f4a7c15ULL;
    for (int iteration = 0; iteration < 128; ++iteration)
    {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        auto malformed = valid;
        const std::size_t index = static_cast<std::size_t>(state % valid.orderCount);
        switch ((state >> 8) % 6)
        {
            case 0: malformed.orders[index].selectionProbability = nan; break;
            case 1: malformed.orders[index].reflectedPower = inf; break;
            case 2: malformed.orders[index].angularStandardDeviationRadians = -huge; break;
            case 3: malformed.orders[index].direction[static_cast<std::size_t>(state % 3)] = huge; break;
            case 4: malformed.orders[index].selectionProbability = huge; break;
            default: malformed.orders[index].reflectedPower += 0.125; break;
        }
        check(!sampleDiffractionOrder(malformed, 0.0, sample, error),
              "deterministic malformed-distribution fuzz rejects corruption");
    }
    check(!sampleDiffractionOrder(valid, -0.1, sample, error)
              && !sampleDiffractionOrder(valid, 1.0, sample, error)
              && !sampleDiffractionOrder(valid, inf, sample, error),
          "out-of-range and nonfinite random inputs fail closed");
}

void crossedIntegrationConvergenceTest()
{
    using namespace diffractionmaterial;
    constexpr double duty = 0.3003;
    constexpr double phase = 1.2;
    const auto exactAxisPower = [phase] (int order)
    {
        const double aperture = std::sin(pi * static_cast<double>(order) * duty)
            / (pi * static_cast<double>(order));
        return aperture * aperture * std::norm(std::polar(1.0, phase) - 1.0);
    };
    const double exactPower = exactAxisPower(1) * exactAxisPower(-1);
    std::array<double, 3> errors {};
    const std::array<std::size_t, 3> refinements { 512, 1024, 2048 };
    for (std::size_t index = 0; index < refinements.size(); ++index)
    {
        const double power = std::norm(numericalCoefficient2D(
            GrooveProfile::BinaryRectangular, 1, -1, duty, phase, refinements[index]));
        errors[index] = std::abs(power - exactPower);
    }
    check(std::fmod(duty * 1024.0, 1.0) != 0.0,
          "crossed convergence duty cycle is not aligned to the 1024 grid");
    check(errors[0] > errors[1] && errors[1] > errors[2],
          "crossed midpoint integration converges through 512, 1024, and 2048");
    check(errors[2] / exactPower < 2.0e-4,
          "crossed 2048 midpoint integration proves the production tolerance");
}
void discreteCameraPixelTests()
{
    using namespace diffractionmaterial;
    auto description = makeAluminiumBinaryGratingPreset();
    description.roughness = {}; description.spectrum.firstOrder = description.spectrum.lastOrder = 1;
    const auto patternedDuty = description.microstructure.dutyCycle;
    // A full-duty terrace is exactly planar while retaining an admitted depth.
    description.microstructure.dutyCycle = 1;
    std::string error;
    const auto flatMaterial = admit(description, error);
    check(bool(flatMaterial), "flat discrete reflective IR admits"); if (!flatMaterial) return;
    const DiffractionLocalFrame frame;
    const std::array<float,3> normal{0,0,1};
    const DiffractionCameraIncident irradiance{DiffractionIncidentMeasure::SurfaceSpectralIrradiance, 2};
    const DiffractionCameraPixel whole{frame, pi/2, 1,1,0,0};
    DiffractionCameraPixelIntegral flat;
    check(integrateDiscreteDiffractionCameraPixel(*flatMaterial, 3, frame, normal, irradiance, whole, flat, error),
        "flat delta integrates in finite actual perspective domain");
    // Four times atan(1/sqrt(3)) is the independent spherical rectangle area.
    const double exactAngle = 2*pi/3;
    const double reflection = interfacePower({1,0},
        {description.substrate.refractiveIndex,description.substrate.extinctionCoefficient});
    check(std::abs(flat.solidAngleSteradians-exactAngle)<1.e-12
        &&std::abs(flat.meanSpectralRadiance-2*reflection/exactAngle)<kFloatTolerance
        &&std::abs(flat.pixelReflectedIrradiance-2*reflection)<kFloatTolerance
        &&std::abs(flat.unresolvedReflectedIrradiance)<kFloatTolerance,
        "analytic flat limit has exact solid angle, cosine-normalized radiance and reflection");
    unsigned owners = 0;
    double angleSum = 0, integralSum = 0;
    for (unsigned y=0;y<2;++y) for(unsigned x=0;x<2;++x)
    {
        DiffractionCameraPixel pixel=whole;pixel.width=pixel.height=2;pixel.x=x;pixel.y=y;
        DiffractionCameraPixelIntegral child;
        check(integrateDiscreteDiffractionCameraPixel(*flatMaterial,3,frame,normal,irradiance,pixel,child,error),
            "flat pixel subdivision is finite");
        for(std::size_t order=0;order<child.orderCount;++order)
            if(child.orders[order].order.primary==0&&child.orders[order].inPixel)++owners;
        angleSum+=child.solidAngleSteradians;integralSum+=child.meanSpectralRadiance*child.solidAngleSteradians;
        if(x==1&&y==1) check(child.pixelReflectedIrradiance==flat.pixelReflectedIrradiance,
            "exact shared boundary belongs to top-first half-open pixel 1,1");
    }
    check(owners==1&&std::abs(angleSum-exactAngle)<1.e-12
        &&std::abs(integralSum-flat.meanSpectralRadiance*exactAngle)<1.e-12,
        "pixel partition counts a boundary delta once and preserves its radiance integral");
    DiffractionCameraPixelIntegral tilted;
    const double incidenceCosine=std::sqrt(.96);
    check(integrateDiscreteDiffractionCameraPixel(*flatMaterial,3,frame,{.2,0,incidenceCosine},irradiance,whole,tilted,error)
        &&std::abs(tilted.meanSpectralRadiance*exactAngle-2*reflection/incidenceCosine)<kFloatTolerance,
        "surface irradiance already includes incident cosine and is divided only by exact outgoing cosine");
    description.microstructure.dutyCycle=patternedDuty;
    description.microstructure.grooveDepthNanometres=.9f;
    const auto material=admit(description,error);check(bool(material),"nonzero binary discrete IR admits");if(!material)return;
    DiffractionCameraPixelIntegral coarse;
    check(integrateDiscreteDiffractionCameraPixel(*material,3,frame,normal,irradiance,whole,coarse,error),
        "nonzero physical orders integrate without a Gaussian width");
    const double wavelength=description.spectrum.wavelengthsNanometres[3];
    const double exitCosine=std::sqrt(1-std::pow(wavelength/description.geometry.grooveSpacingNanometres,2));
    const double phase=2*pi*description.microstructure.grooveDepthNanometres*(1+exitCosine)/wavelength;
    const double signedPower=reflection*4*std::pow(std::sin(phase/2),2)
        *std::pow(std::sin(pi*description.microstructure.dutyCycle)/pi,2)*exitCosine;
    check(std::abs(coarse.orders[0].reflectedIrradiance-2*signedPower)<kFloatTolerance
        &&std::abs(coarse.orders[2].reflectedIrradiance-2*signedPower)<kFloatTolerance
        &&std::abs(coarse.orders[0].integratedRadiance-2*signedPower/exitCosine)<kFloatTolerance,
        "analytical binary profile and projected outgoing cosine determine the finite delta coefficient");
    DiffractionOrderDistribution original;
    check(evaluateDiffractionOrders(*material,description.spectrum.wavelengthsNanometres[3],frame,normal,{0,0},original,error),
        "original reflected distribution is retained independently of camera filter");
    double partition = 0, radianceIntegral = 0, solidAngle = 0;
    std::array<unsigned,3> counts{};
    for(unsigned y=0;y<7;++y)for(unsigned x=0;x<7;++x)
    {
        DiffractionCameraPixel pixel=whole;pixel.width=pixel.height=7;pixel.x=x;pixel.y=y;
        DiffractionCameraPixelIntegral child;
        check(integrateDiscreteDiffractionCameraPixel(*material,3,frame,normal,irradiance,pixel,child,error),
            "nonzero-order pixel partition admits");
        partition+=child.pixelReflectedIrradiance;solidAngle+=child.solidAngleSteradians;
        radianceIntegral+=child.meanSpectralRadiance*child.solidAngleSteradians;
        for(std::size_t order=0;order<child.orderCount;++order) {
            if(child.orders[order].inPixel)++counts[order];
            check(child.orders[order].order.direction==original.orders[order].direction
                &&child.orders[order].order.reflectedPower==original.orders[order].reflectedPower
                &&child.orders[order].order.selectionProbability==original.orders[order].selectionProbability,
                "filter preserves exact original support, power and discrete probability mass");
        }
    }
    check(counts==std::array<unsigned,3>{1,1,1}&&coarse.orders[0].reflectedIrradiance>0
        &&coarse.orders[2].reflectedIrradiance>0&&std::abs(partition-coarse.resolvedReflectedIrradiance)<kFloatTolerance
        &&std::abs(partition+coarse.unresolvedReflectedIrradiance+coarse.absorbedIrradiance-irradiance.value)<kFloatTolerance
        &&std::abs(solidAngle-exactAngle)<1.e-12
        &&std::abs(radianceIntegral-coarse.meanSpectralRadiance*exactAngle)<kFloatTolerance,
        "nontrivial signed-order angular partition preserves reflected/absorbed/unresolved energy once");
    DiffractionCameraPixelIntegral doubled, dark, away;
    auto twice=irradiance;twice.value*=2;
    auto zero=irradiance;zero.value=0;
    auto back=whole;back.axes.tangent={-1,0,0};back.axes.normal={0,0,-1};
    check(integrateDiscreteDiffractionCameraPixel(*material,3,frame,normal,twice,whole,doubled,error)
        &&doubled.meanSpectralRadiance==2*coarse.meanSpectralRadiance
        &&doubled.unresolvedReflectedIrradiance==2*coarse.unresolvedReflectedIrradiance
        &&integrateDiscreteDiffractionCameraPixel(*material,3,frame,normal,zero,whole,dark,error)
        &&dark.meanSpectralRadiance==0
        &&integrateDiscreteDiffractionCameraPixel(*material,3,frame,normal,irradiance,back,away,error)
        &&away.meanSpectralRadiance==0&&away.resolvedReflectedIrradiance==coarse.resolvedReflectedIrradiance,
        "linear source additivity, zero input and outside-camera orders preserve physical metadata");
    auto rectangular=whole;rectangular.width=3;rectangular.height=2;
    double rectangleAngle=0;
    for(unsigned y=0;y<2;++y)for(unsigned x=0;x<3;++x) {
        rectangular.x=x;rectangular.y=y;DiffractionCameraPixelIntegral piece;
        check(integrateDiscreteDiffractionCameraPixel(*material,3,frame,normal,irradiance,rectangular,piece,error),
            "actual non-square perspective aspect integrates");
        rectangleAngle+=piece.solidAngleSteradians;
    }
    check(std::abs(rectangleAngle-4*std::atan(1.5/std::sqrt(4.25)))<1.e-12,
        "non-square pixel partition matches independent analytic spherical rectangle area");
    auto rotated=whole;rotated.axes.tangent={0,1,0};rotated.axes.bitangent={-1,0,0};
    DiffractionCameraPixelIntegral turned;
    check(integrateDiscreteDiffractionCameraPixel(*material,3,frame,normal,irradiance,rotated,turned,error)
        &&turned.meanSpectralRadiance==coarse.meanSpectralRadiance,
        "rotating square camera axes changes projection without changing integrated physical power");
    for(int fault=0;fault<14;++fault)
    {
        auto pixel=whole;auto incident=irradiance;auto ray=normal;std::size_t band=3;
        if(fault==0)incident.measure=DiffractionIncidentMeasure::Unspecified;
        if(fault==1)incident.measure=DiffractionIncidentMeasure::SpectralRadiance;
        if(fault==2)incident.measure=DiffractionIncidentMeasure::SpectralFlux;
        if(fault==3)incident.value=-1;
        if(fault==4)incident.value=std::numeric_limits<double>::quiet_NaN();
        if(fault==5)incident.value=std::numeric_limits<double>::infinity();
        if(fault==6)pixel.width=0;
        if(fault==7)pixel.height=4097;
        if(fault==8)pixel.x=1;
        if(fault==9)pixel.verticalFovRadians=pi;
        if(fault==10)pixel.axes.bitangent=pixel.axes.tangent;
        if(fault==11)band=8;
        if(fault==12)ray={1,0,0};
        if(fault==13) { incident.value=std::numeric_limits<double>::max(); pixel.verticalFovRadians=.1; }
        auto sentinel=coarse;sentinel.meanSpectralRadiance=123;
        check(!integrateDiscreteDiffractionCameraPixel(*material,band,frame,ray,incident,pixel,sentinel,error)
            &&sentinel.meanSpectralRadiance==123,
            "invalid units, singular/grazing projection and unrepresentable measures refuse atomically");
    }
    for(int fault=0;fault<4;++fault) {
        auto wrong=description;
        if(fault==0)wrong.roughness.rmsSlope=.08f;
        if(fault==1)wrong.roughness.rmsHeightNanometres=1;
        if(fault==2)wrong.spectrum.lastOrder=2;
        if(fault==3)wrong.microstructure.profile=GrooveProfile::Sinusoidal;
        const auto authority=admit(wrong,error);DiffractionCameraPixelIntegral sentinel=coarse;
        check(authority&&!integrateDiscreteDiffractionCameraPixel(*authority,3,frame,normal,irradiance,whole,sentinel,error),
            "private delta camera authority refuses other physical models without changing their finite-lobe admission");
    }
}
// A second axis at 60 degrees gives first orders along six directions 60 degrees
// apart, each a tangent step of wavelength / period, at normal incidence.
void hexagonalOrderDirectionsTest(const diffractionmaterial::Description& material)
{
    using namespace diffractionmaterial;
    std::string error;
    const auto admitted = admit(material, error);
    check(admitted.has_value(), "sixty degree lattice admits");
    if (!admitted)
        return;
    const double wavelength = 460.0;
    const double step = wavelength / material.geometry.grooveSpacingNanometres;
    const DiffractionLocalFrame frame;
    DiffractionOrderDistribution distribution;
    check(evaluateDiffractionOrders(*admitted, wavelength, frame, { 0.0, 0.0, 1.0 }, { 0.5, 0.5 },
                                    distribution, error),
          "sixty degree lattice evaluates");
    const std::array<std::array<int, 2>, 6> arms {{ { 1, 0 }, { 0, 1 }, { -1, 1 },
                                                     { -1, 0 }, { 0, -1 }, { 1, -1 } }};
    for (std::size_t arm = 0; arm < arms.size(); ++arm)
    {
        const auto found = std::find_if(distribution.orders.begin(),
            distribution.orders.begin() + static_cast<std::ptrdiff_t>(distribution.orderCount),
            [&] (const DiffractionOrder& order)
            { return order.primary == arms[arm][0] && order.secondary == arms[arm][1]; });
        check(found != distribution.orders.begin() + static_cast<std::ptrdiff_t>(distribution.orderCount),
              "sixty degree lattice has each arm's order");
        if (found == distribution.orders.begin() + static_cast<std::ptrdiff_t>(distribution.orderCount))
            continue;
        const double angle = static_cast<double>(arm) * 60.0 * pi / 180.0;
        check(std::abs(found->direction[0] - step * std::cos(angle)) < 1.0e-5
                  && std::abs(found->direction[1] - step * std::sin(angle)) < 1.0e-5,
              "sixty degree lattice arm points along its sixth of the circle");
    }
}

// With all three families on, the first orders carry equal power on all six arms
// and match the Fourier coefficient of the three-family surface, integrated numerically.
void hexagonalOrderPowerTest(diffractionmaterial::GrooveProfile profile, double depth,
                             const char* name)
{
    using namespace diffractionmaterial;
    auto material = makeHexagonalGratingPreset();
    material.roughness = {};
    material.microstructure.profile = profile;
    material.microstructure.grooveDepthNanometres = static_cast<float>(depth);
    material.microstructure.dutyCycle = 0.5f;
    std::string error;
    const auto admitted = admit(material, error);
    check(admitted.has_value(), std::string(name) + " three-family lattice admits: " + error);
    if (!admitted)
        return;
    const double wavelength = 540.0;
    const auto height = [&] (double u, double phase)
    {
        if (profile == GrooveProfile::Sinusoidal)
            return 0.5 * phase * std::cos(2.0 * pi * u);
        return std::abs(u - std::floor(u + 0.5)) < 0.25 ? phase : 0.0;
    };
    DiffractionOrderDistribution distribution;
    check(evaluateDiffractionOrders(*admitted, wavelength, DiffractionLocalFrame {}, { 0.0, 0.0, 1.0 },
                                    { 0.5, 0.5 }, distribution, error),
          std::string(name) + " three-family lattice evaluates");
    double firstArm = -1.0;
    constexpr std::size_t samples = 512;
    for (std::size_t index = 0; index < distribution.orderCount; ++index)
    {
        const auto& order = distribution.orders[index];
        const double z = order.direction[2];
        const double phase = 2.0 * pi * depth * (1.0 + z) / wavelength;
        std::complex<double> sum {};
        for (std::size_t yIndex = 0; yIndex < samples; ++yIndex)
            for (std::size_t xIndex = 0; xIndex < samples; ++xIndex)
            {
                const double x = (static_cast<double>(xIndex) + 0.5) / samples;
                const double y = (static_cast<double>(yIndex) + 0.5) / samples;
                sum += std::polar(1.0, height(x, phase) + height(y, phase) + height(x - y, phase)
                    - 2.0 * pi * (order.primary * x + order.secondary * y));
            }
        sum /= static_cast<double>(samples * samples);
        const double expected = distribution.substrateReflectedPower * std::norm(sum)
            * (order.primary != 0 || order.secondary != 0 ? z : 1.0);
        check(std::abs(order.reflectedPower - expected) < 0.004,
              std::string(name) + " three-family order power matches the surface integral");
        const bool arm = (order.primary == 0 || order.secondary == 0
                          || order.primary == -order.secondary)
            && (order.primary != 0 || order.secondary != 0);
        if (!arm)
            continue;
        if (firstArm < 0.0)
            firstArm = order.reflectedPower;
        check(std::abs(order.reflectedPower - firstArm) < 1.0e-5,
              std::string(name) + " three-family arms carry equal power");
    }
    check(firstArm > 0.0, std::string(name) + " three-family arms carry power");
}
} // namespace

int main()
{
    using namespace diffractionmaterial;
    const DiffractionLocalFrame identity;
    const std::array<float, 3> normal { 0.0f, 0.0f, 1.0f };

    auto binary = makeAluminiumBinaryGratingPreset();
    binary.spectrum.lastOrder = 2;
    binary.roughness = { 3.0f, 0.08f };
    runCase(binary, 540.0, identity, { 0.0, 0.0, 1.0 }, { 0.5, 0.5 }, "binary 1D normal");

    const double c = std::sqrt(0.5);
    const DiffractionLocalFrame rotatedFrame {
        { c, c, 0.0 }, { -0.5, 0.5, c }, { 0.5, -0.5, c }
    };
    const auto obliqueWorld = normalized(std::array<double, 3> {
        0.22 * rotatedFrame.tangent[0] - 0.17 * rotatedFrame.bitangent[0]
            + 0.960989073 * rotatedFrame.normal[0],
        0.22 * rotatedFrame.tangent[1] - 0.17 * rotatedFrame.bitangent[1]
            + 0.960989073 * rotatedFrame.normal[1],
        0.22 * rotatedFrame.tangent[2] - 0.17 * rotatedFrame.bitangent[2]
            + 0.960989073 * rotatedFrame.normal[2]
    });

    auto sinusoidal = makeSinusoidalGratingPreset();
    sinusoidal.spectrum.lastOrder = 2;
    sinusoidal.geometry.directionUv = { 0.6f, 0.8f };
    sinusoidal.grooveField = { GrooveFieldMode::Linear, { 0.2f, 0.3f }, { 0.8f, 0.6f },
                               180.0f, 0.0f, 37.0f };
    runCase(sinusoidal, 580.0, rotatedFrame, obliqueWorld, { 0.83, 0.71 },
            "sinusoidal oblique rotated linear field");

    auto blazed = makeBlazedGratingPreset();
    blazed.spectrum.lastOrder = 2;
    blazed.geometry.directionUv = { -0.8f, 0.6f };
    runCase(blazed, 640.0, rotatedFrame, obliqueWorld, { 0.31, 0.77 },
            "blazed oblique rotated reciprocal axis");

    auto crossed = makeCrossedTwoDimensionalGratingPreset();
    crossed.spectrum.lastOrder = 1;
    crossed.roughness = { 2.0f, 0.03f };
    crossed.geometry.directionUv = { 0.6f, 0.8f };
    crossed.geometry.secondaryDirectionUv = { -0.8f, 0.6f };
    crossed.grooveField = { GrooveFieldMode::Radial, { 0.25f, 0.2f }, {},
                            120.0f, -90.0f, 29.0f };
    runCase(crossed, 460.0, rotatedFrame, obliqueWorld, { 0.72, 0.81 },
            "crossed oblique rotated radial varying periods");

    auto hexagonal = makeCrossedTwoDimensionalGratingPreset();
    hexagonal.spectrum.lastOrder = 1;
    hexagonal.geometry.directionUv = { 1.0f, 0.0f };
    hexagonal.geometry.secondaryDirectionUv = { 0.5f, 0.8660254f };
    hexagonal.geometry.secondaryGrooveSpacingNanometres = hexagonal.geometry.grooveSpacingNanometres;
    runCase(hexagonal, 460.0, rotatedFrame, obliqueWorld, { 0.5, 0.5 },
            "crossed at sixty degrees");
    hexagonalOrderDirectionsTest(hexagonal);
    hexagonalOrderPowerTest(GrooveProfile::BinaryRectangular, 120.0, "binary");
    hexagonalOrderPowerTest(GrooveProfile::Sinusoidal, 120.0, "sinusoidal");

    std::string error;
    const auto admitted = admit(binary, error);
    DiffractionOrderDistribution distribution;
    DiffractionLocalFrame malformedFrame;
    malformedFrame.bitangent = malformedFrame.tangent;
    check(admitted && !evaluateDiffractionOrders(*admitted, 540.0, malformedFrame,
              normal, { 0.5, 0.5 }, distribution, error),
          "malformed frame fails closed before publication");
    check(admitted && !evaluateDiffractionOrders(*admitted, 540.0, identity,
              { 1.0, 0.0, 0.0 }, { 0.5, 0.5 }, distribution, error),
          "grazing incidence fails closed");
    check(admitted && !evaluateDiffractionOrders(*admitted, 379.0, identity,
              normal, { 0.5, 0.5 }, distribution, error),
          "out-of-range wavelength fails closed");
    check(admitted && !evaluateDiffractionOrders(*admitted, 540.0, identity, normal,
              { std::numeric_limits<double>::quiet_NaN(), 0.5 }, distribution, error),
          "nonfinite material coordinate fails closed");

    discreteCameraPixelTests();
    perWavelengthFurnaceTests();
    measuredAluminiumTests();
    orderMediumTests();
    pressedDiscPitTests();
    boundaryAndMalformedTests();
    crossedIntegrationConvergenceTest();
    std::cout << checks << " checks, " << failures << " failures\n";
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
