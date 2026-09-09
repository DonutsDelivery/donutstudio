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

double dot(const std::array<double, 3>& a, const std::array<double, 3>& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
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
    check(evaluateDiffractionOrders(*admitted, wavelength, frame, incident, uv,
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
        const double directionDot = std::clamp(dot(actual.direction, expected.direction), -1.0, 1.0);
        check(std::acos(directionDot) <= 1.0e-6,
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
    check(std::abs(probabilitySum - 1.0) <= 2.0e-12,
          std::string(label) + " probabilities normalize");
    check(std::abs(production.resolvedReflectedPower + production.unresolvedReflectedPower
                   - production.substrateReflectedPower)
              <= 1.0e-10 * std::max(1.0, production.substrateReflectedPower),
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

void boundaryAndMalformedTests()
{
    using namespace diffractionmaterial;
    DiffractionOrderDistribution valid;
    valid.orderCount = 5;
    valid.resolvedReflectedPower = 10.0;
    const std::array<double, 5> probabilities { 0.2, 0.0, 0.3, 0.5, 0.0 };
    for (std::size_t index = 0; index < valid.orderCount; ++index)
    {
        valid.orders[index].direction = { 0.0, 0.0, 1.0 };
        valid.orders[index].selectionProbability = probabilities[index];
        valid.orders[index].reflectedPower = 10.0 * probabilities[index];
        valid.orders[index].angularStandardDeviationRadians = 0.01;
    }
    std::string error;
    DiffractionOrderSample sample;
    const auto owns = [&] (double u, std::size_t index, const char* label)
    {
        check(sampleDiffractionOrder(valid, u, sample, error) && sample.orderIndex == index,
              label);
    };
    owns(0.0, 0, "u=0 belongs to the first positive bin");
    owns(std::nextafter(0.2, 0.0), 0, "value below first boundary stays in first bin");
    owns(0.2, 2, "exact first boundary skips zero-probability interior bin");
    owns(std::nextafter(0.2, 1.0), 2, "value above first boundary enters next positive bin");
    owns(std::nextafter(0.5, 0.0), 2, "value below second boundary stays in middle bin");
    owns(0.5, 3, "exact second boundary belongs to following bin");
    owns(std::nextafter(0.5, 1.0), 3, "value above second boundary stays in following bin");
    owns(std::nextafter(1.0, 0.0), 3, "largest unit sample skips trailing zero bin");

    for (const double residual : { -0.5e-12, 0.5e-12 })
    {
        auto rounded = valid;
        rounded.orders[3].selectionProbability += residual;
        rounded.orders[3].reflectedPower += residual * rounded.resolvedReflectedPower;
        check(sampleDiffractionOrder(rounded, std::nextafter(1.0, 0.0), sample, error)
                  && sample.orderIndex == 3
                  && sample.pdf == 0.5
                  && sample.order.selectionProbability == sample.pdf
                  && sample.order.reflectedPower
                      == sample.pdf * rounded.resolvedReflectedPower,
              residual < 0.0
                  ? "low tolerance edge closes canonically at one"
                  : "high tolerance edge closes canonically at one");
    }

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    const double huge = std::numeric_limits<double>::max();
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
        { "compensated positive-bin mismatch rejected", [] (auto& d) { d.orders[0].reflectedPower += 2.0e-9; d.orders[3].reflectedPower -= 2.0e-9; } },
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
} // namespace

int main()
{
    using namespace diffractionmaterial;
    const DiffractionLocalFrame identity;
    const std::array<double, 3> normal { 0.0, 0.0, 1.0 };

    auto binary = makeAluminiumBinaryGratingPreset();
    binary.spectrum.lastOrder = 2;
    binary.roughness = { 3.0f, 0.08f };
    runCase(binary, 540.0, identity, normal, { 0.5, 0.5 }, "binary 1D normal");

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

    boundaryAndMalformedTests();
    crossedIntegrationConvergenceTest();
    std::cout << checks << " checks, " << failures << " failures\n";
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
