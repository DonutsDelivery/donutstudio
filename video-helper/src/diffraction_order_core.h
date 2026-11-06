#pragma once

// The diffraction authority itself, for host and device. No strings, heap or host
// containers: refusals are a DiffractionOrderProblem, and the material is the flat
// Description. diffraction_order_bsdf.h wraps these for the host and turns a problem
// into its text.

#include "../../shared/DiffractionMaterialIR.h"
#include "../../shared/DiffractionSubstrateData.h"
#include "pathtrace/devices/device_annotations.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace diffractionmaterial
{
inline constexpr std::size_t kMaximumDiffractionBsdfOrders
    = (2u * kMaximumDiffractionOrder + 1u)
        * (2u * kMaximumDiffractionOrder + 1u);

struct DiffractionLocalFrame
{
    std::array<float, 3> tangent { 1.0f, 0.0f, 0.0f };
    std::array<float, 3> bitangent { 0.0f, 1.0f, 0.0f };
    std::array<float, 3> normal { 0.0f, 0.0f, 1.0f };
};

struct DiffractionOrder
{
    std::int8_t primary = 0;
    std::int8_t secondary = 0;
    std::array<float, 3> direction {};
    // Canonical support is exact: reflectedPower and selectionProbability are
    // either both zero or both positive. The sampler assigns normalization
    // rounding, bounded by its validation tolerance, to the last positive bin.
    float reflectedPower = 0.0f;
    float selectionProbability = 0.0f;
    float angularStandardDeviationRadians = 0.0f;
};

struct DiffractionOrderDistribution
{
    std::array<DiffractionOrder, kMaximumDiffractionBsdfOrders> orders {};
    std::size_t orderCount = 0;
    float substrateReflectedPower = 0.0f;
    float resolvedReflectedPower = 0.0f;
    float unresolvedReflectedPower = 0.0f;
    // The part of the unresolved power a pressed disc's pits scatter, 0 for any other
    // material. Across the tracks its tangent offset has density sinc^2(offset / width),
    // along them a Lorentzian of half width pitAlongWidth. The axes are unit surface tangents.
    float pitScatterPower = 0.0f;
    float pitAcrossWidth = 0.0f;
    float pitAlongWidth = 0.0f;
    std::array<float, 3> pitAcrossAxis {};
    std::array<float, 3> pitAlongAxis {};
    // Roughness across the grooves: every lobe also spreads by this standard deviation
    // along one groove family's reciprocal axis, a unit surface tangent, chosen evenly.
    float acrossGrooveDeviation = 0.0f;
    std::size_t grooveFamilyCount = 0;
    std::array<std::array<float, 3>, 3> grooveFamilyAxes {};
    // Relief scatter, shallow non-periodic relief along each groove family's axis. Its tangent
    // offset is uniform to +-reliefReach along the axis and a Gaussian across it. The bound is
    // the power its sampling band takes, and reliefLeavingPower the part that leaves, since
    // the scatter goes as cos wi and only inside the horizon. The rest stays in the mirror.
    float reliefScatterBound = 0.0f;
    float reliefLeavingPower = 0.0f;
    float reliefReach = 0.0f;
    float reliefAcrossDeviation = 0.0f;
};

// Every member but the orders back to its default, in place. Keep it in step with
// the struct above.
PT_SHARED_FN void clearAllButOrders(DiffractionOrderDistribution& distribution) noexcept
{
    distribution.orderCount = 0;
    distribution.substrateReflectedPower = 0.0f;
    distribution.resolvedReflectedPower = 0.0f;
    distribution.unresolvedReflectedPower = 0.0f;
    distribution.pitScatterPower = 0.0f;
    distribution.pitAcrossWidth = 0.0f;
    distribution.pitAlongWidth = 0.0f;
    distribution.pitAcrossAxis = {};
    distribution.pitAlongAxis = {};
    distribution.acrossGrooveDeviation = 0.0f;
    distribution.grooveFamilyCount = 0;
    distribution.grooveFamilyAxes = {};
    distribution.reliefScatterBound = 0.0f;
    distribution.reliefLeavingPower = 0.0f;
    distribution.reliefReach = 0.0f;
    distribution.reliefAcrossDeviation = 0.0f;
}

struct DiffractionOrderSample
{
    DiffractionOrder order;
    std::size_t orderIndex = 0;
    // Probability mass for this discrete order. This is not a solid-angle PDF.
    // A future angular-lobe sampler must report its directional density separately.
    float pdf = 0.0f;
    float throughput = 0.0f;
};

struct LocalGrating
{
    std::array<float, 2> directionUv {};
    float periodNanometres = 0.0f;
};


enum class DiffractionOrderProblem : std::uint8_t
{
    None = 0,
    Wavelength,
    MediumIndex,
    LocalFrame,
    MaterialCoordinate,
    Lattice,
    Profile,
    SinusoidalPhase,
    GrazingIncident,
    GrooveField,
    LocalPeriod,
    Reflectance,
    OrderPower,
    NoResolvedEnergy,
    NoSampleableEnergy,
    InvalidDistribution,
    SampleRange,
    SampleDistribution,
    SampleSupport,
    SampleInconsistent,
    SampleNotNormalized,
    SampleResolvedPower,
    SampleZeroProbability,
    SampleThroughput,
};

namespace ordercore
{
// The arithmetic std::complex does on finite values. std::complex is not
// callable from device code.
struct Complex
{
    float re = 0.0f;
    float im = 0.0f;
};

PT_SHARED_FN Complex operator+ (Complex a, Complex b) noexcept { return { a.re + b.re, a.im + b.im }; }
PT_SHARED_FN Complex operator- (Complex a, Complex b) noexcept { return { a.re - b.re, a.im - b.im }; }
PT_SHARED_FN Complex operator* (Complex a, Complex b) noexcept { return { a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re }; }
PT_SHARED_FN Complex operator* (float a, Complex b) noexcept { return { a * b.re, a * b.im }; }
PT_SHARED_FN Complex operator+ (float a, Complex b) noexcept { return { a + b.re, b.im }; }
PT_SHARED_FN Complex operator- (Complex a, float b) noexcept { return { a.re - b, a.im }; }
PT_SHARED_FN float normSquared (Complex a) noexcept { return a.re * a.re + a.im * a.im; }

constexpr float pi = 3.141592653589793238462643383279502884f;
// Path Trace's hero wavelengths run to 780 nm (pathtrace/spectrum.h kWavelengthMaxNm). Only
// the grating equation and the coating's phase read the wavelength here, and neither has a
// table that stops at the 700 nm of the material's sampled spectrum.
constexpr float maximumEvaluatedWavelength = 780.0f;
// Float's own rounding, stated in its epsilon. A direction is a sum of three frame axes
// scaled by coefficients, and dot(v, v) rounds three squares and two sums, so a true unit
// vector misses 1 by a few epsilon and a frame's errors add. 32 keeps every real frame.
constexpr float directionTolerance = 32.0f * std::numeric_limits<float>::epsilon();
// Up to kMaximumDiffractionBsdfOrders probabilities are summed, each sum rounding by
// epsilon at most, and a reflected power is one of them times the total.
constexpr float probabilityTolerance = static_cast<float>(kMaximumDiffractionBsdfOrders) * std::numeric_limits<float>::epsilon();
constexpr float distributionConsistencyTolerance = probabilityTolerance;

PT_SHARED_FN bool finiteVector(const std::array<float, 3>& value) noexcept
{
    return std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2]);
}

PT_SHARED_FN float dot(const std::array<float, 3>& first,
           const std::array<float, 3>& second) noexcept
{
    return first[0] * second[0] + first[1] * second[1]
        + first[2] * second[2];
}

PT_SHARED_FN std::array<float, 3> cross(const std::array<float, 3>& first,
                            const std::array<float, 3>& second) noexcept
{
    return { first[1] * second[2] - first[2] * second[1],
             first[2] * second[0] - first[0] * second[2],
             first[0] * second[1] - first[1] * second[0] };
}

// Makes a tangent square to a unit normal and unit length. One pass leaves the rounding of
// its subtraction, and normalizing a short remainder magnifies that past directionTolerance,
// so a tangent near the normal would have its frame refused. The second pass takes it out.
PT_SHARED_FN bool orthonormalTangent(const std::array<float, 3>& normal, std::array<float, 3>& tangent) noexcept
{
    for (int pass = 0; pass < 2; ++pass)
    {
        const float along = dot(tangent, normal);
        for (std::size_t i = 0; i < 3; ++i)
            tangent[i] -= along * normal[i];
        const float size = std::sqrt(dot(tangent, tangent));
        if (!(size > 1.0e-6f))
            return false;
        for (float& component : tangent)
            component /= size;
    }
    return true;
}

PT_SHARED_FN bool unit(const std::array<float, 3>& value) noexcept
{
    return finiteVector(value) && std::abs(dot(value, value) - 1.0f) <= directionTolerance;
}

PT_SHARED_FN bool validFrame(const DiffractionLocalFrame& frame) noexcept
{
    if (!unit(frame.tangent) || !unit(frame.bitangent) || !unit(frame.normal)
        || std::abs(dot(frame.tangent, frame.bitangent)) > directionTolerance
        || std::abs(dot(frame.tangent, frame.normal)) > directionTolerance
        || std::abs(dot(frame.bitangent, frame.normal)) > directionTolerance)
        return false;
    const auto handedNormal = cross(frame.tangent, frame.bitangent);
    return dot(handedNormal, frame.normal) >= 1.0f - directionTolerance;
}

PT_SHARED_FN Complex exponentialIntegral(float frequency,
                                         float begin,
                                         float end) noexcept
{
    if (std::abs(frequency) < 1.0e-12f)
        return Complex { end - begin, 0.0f };
    return Complex { (std::sin(frequency * end) - std::sin(frequency * begin)) / frequency,
                     -(std::cos(frequency * end) - std::cos(frequency * begin)) / frequency };
}

PT_SHARED_FN float besselJ(int order, float argument) noexcept
{
    const int magnitude = std::abs(order);
    const float halfArgument = 0.5f * argument;
    float term = 1.0f;
    for (int factor = 1; factor <= magnitude; ++factor)
        term *= halfArgument / static_cast<float>(factor);
    float sum = term;
    for (int index = 1; index <= 48; ++index)
    {
        term *= -(halfArgument * halfArgument)
            / (static_cast<float>(index) * static_cast<float>(magnitude + index));
        sum += term;
        // Terms grow before their peak, where a partial sum can't exceed the last term, and only
        // shrink after it. So 2^-25 of the sum is past the peak, and no later term can round
        // into the float sum. Running on only reaches zeros and subnormals: the orb's OptiX
        // sample took 5.49 s with them and takes 1.59 s without.
        if (std::abs(term) <= 2.98e-8f * std::abs(sum))
            break;
    }
    if (order < 0 && (magnitude & 1) != 0)
        sum = -sum;
    return sum;
}

PT_SHARED_FN Complex profileAmplitude(GrooveProfile profile,
                                      int order,
                                      float duty,
                                      float phase) noexcept
{
    const Complex terrace { std::cos(phase), std::sin(phase) };
    if (profile == GrooveProfile::BinaryRectangular)
    {
        if (order == 0)
            return (1.0f - duty) + duty * terrace;
        // Measured from the groove centre, so the three families of a lattice
        // share a centre of symmetry.
        const float signedOrder = static_cast<float>(order);
        return std::sin(pi * signedOrder * duty) / (pi * signedOrder) * (terrace - 1.0f);
    }
    if (profile == GrooveProfile::Sinusoidal)
    {
        // exp(i b cos x) = sum i^m J_m(b) exp(i m x).
        const int magnitude = std::abs(order);
        const float bessel = besselJ(magnitude, 0.5f * phase);
        const int quarterTurns = magnitude % 4;
        return quarterTurns == 0 ? Complex { bessel, 0.0f }
            : quarterTurns == 1 ? Complex { 0.0f, bessel }
            : quarterTurns == 2 ? Complex { -bessel, 0.0f }
                                : Complex { 0.0f, -bessel };
    }
    if (profile == GrooveProfile::BlazedSawtooth)
    {
        const float frequency = -2.0f * pi * static_cast<float>(order);
        return exponentialIntegral(phase / duty + frequency, 0.0f, duty)
            + exponentialIntegral(frequency, duty, 1.0f);
    }
    return Complex { std::numeric_limits<float>::quiet_NaN(), 0.0f };
}

// A pressed track's mean run: EFM runs are 3T to 11T, and the maxentropic code with those
// limits has P(L) = 1.4558^-L, mean 4.877 T. T is 833 / 3 nm at 1.2 m/s, so 1354 nm.
constexpr float kPressedDiscMeanRunNanometres = 1354.0f;

// A reflective phase object 1 + (e^(i phase) - 1) P(x) s(y): P is a pit stripe of width
// W = stripeFraction per track pitch, s is the random pit or land along the track with
// mean q = 1/2, since EFM coding holds a track DC-free. Across tracks only the mean of s
// is periodic, so a_0 = 1 + (e^(i phase) - 1) q W and a_m = (e^(i phase) - 1) q W sinc(m W).
// The rest, W q (1 - q) |e^(i phase) - 1|^2 by Parseval, is not periodic and forms no
// order. It stays out of the resolved power and leaves with the unresolved reflection.
PT_SHARED_FN float pressedDiscEfficiency(int order, float stripeFraction, float phase) noexcept
{
    const Complex step { std::cos(phase) - 1.0f, std::sin(phase) };
    const float meanStripe = 0.5f * stripeFraction;
    if (order == 0)
        return normSquared(1.0f + meanStripe * step);
    const float argument = pi * order * stripeFraction;
    const float amplitude = meanStripe * std::sin(argument) / argument;
    return normSquared(step) * amplitude * amplitude;
}

PT_SHARED_FN float latticeEfficiency(GrooveProfile profile,
                         int primary,
                         int secondary,
                         bool crossed,
                         bool thirdFamily,
                         float duty,
                         float phase) noexcept
{
    const auto first = profileAmplitude(profile, primary, duty, phase);
    if (!crossed)
        return normSquared(first);
    if (!thirdFamily)
        return normSquared(first) * normSquared(profileAmplitude(profile, secondary, duty, phase));
    // The third family's reciprocal vector is g1/d1 - g2/d2, so its order r
    // lands on (m + r, n - r) and the amplitudes convolve.
    Complex sum {};
    for (int r = -kThirdFamilyConvolutionRadius; r <= kThirdFamilyConvolutionRadius; ++r)
        sum = sum + profileAmplitude(profile, primary - r, duty, phase)
            * profileAmplitude(profile, secondary + r, duty, phase)
            * profileAmplitude(profile, r, duty, phase);
    return normSquared(sum);
}


PT_SHARED_FN float interfaceReflectance(Complex first,
                            Complex second) noexcept
{
    return std::clamp(normSquared(first - second) / normSquared(first + second), 0.0f, 1.0f);
}

PT_SHARED_FN float reflectance(const Description& description, float wavelength, float mediumIndex) noexcept
{
    const Complex air { mediumIndex, 0.0f };
    // aluminiumIndexAt's own lines, in float: it is called per order path on the GPU.
    const float tablePosition = (wavelength - static_cast<float>(kAluminiumTableFirstNanometres))
        / static_cast<float>(kAluminiumTableStepNanometres);
    const auto tableRow = tablePosition >= 40.0f ? std::size_t { 39 } : static_cast<std::size_t>(tablePosition);
    const float tableAlong = tablePosition - static_cast<float>(tableRow);
    constexpr auto table = kAluminiumMcPeak2015;
    const float measuredLowN = static_cast<float>(table[tableRow].n), measuredLowK = static_cast<float>(table[tableRow].k);
    const struct { float n, k; } measured {
        measuredLowN + (static_cast<float>(table[tableRow + 1].n) - measuredLowN) * tableAlong,
        measuredLowK + (static_cast<float>(table[tableRow + 1].k) - measuredLowK) * tableAlong };
    const bool tabulated = description.substrateData == SubstrateData::AluminiumMcPeak2015;
    const Complex substrate {
        tabulated ? measured.n : description.substrate.refractiveIndex,
        tabulated ? measured.k : description.substrate.extinctionCoefficient };
    if (description.coating.model == CoatingModel::Uncoated)
        return interfaceReflectance(air, substrate);
    if (description.coating.model != CoatingModel::IncoherentDielectric)
        return -1.0f;
    const Complex layer { static_cast<float>(description.coating.opticalConstants.refractiveIndex),
                          static_cast<float>(description.coating.opticalConstants.extinctionCoefficient) };
    const float r01 = interfaceReflectance(air, layer);
    const float r12 = interfaceReflectance(layer, substrate);
    const float roundTrip = std::exp(-8.0f * pi
        * description.coating.opticalConstants.extinctionCoefficient
        * description.coating.thicknessNanometres / wavelength);
    const float denominator = 1.0f - r01 * r12 * roundTrip;
    if (!(denominator > 0.0f) || !std::isfinite(denominator))
        return -1.0f;
    return std::clamp(r01 + (1.0f - r01) * (1.0f - r01)
        * r12 * roundTrip / denominator, 0.0f, 1.0f);
}

PT_SHARED_FN bool admittedOrder(int order, const SpectralSampling& spectrum) noexcept
{
    const int magnitude = std::abs(order);
    return order == 0 || (magnitude >= spectrum.firstOrder
                          && magnitude <= spectrum.lastOrder);
}

// A uv direction turned by the angle whose cosine and sine are given.
PT_SHARED_FN std::array<float, 2> rotateUv(const std::array<float, 2>& direction, float cosine, float sine) noexcept
{
    return std::array<float, 2> { cosine * direction[0] - sine * direction[1],
                                  sine * direction[0] + cosine * direction[1] };
}

// Adds the groove family whose direction in the local frame's tangent plane is (x, y),
// unit length; a zero direction adds none.
PT_SHARED_FN void addGrooveFamily(DiffractionOrderDistribution& candidate, const DiffractionLocalFrame& frame,
                                  float x, float y) noexcept
{
    const float size = std::hypot(x, y);
    if (!(size > 0.0f))
        return;
    auto& axis = candidate.grooveFamilyAxes[candidate.grooveFamilyCount++];
    for (std::size_t i = 0; i < 3; ++i)
        axis[i] = (x * frame.tangent[i] + y * frame.bitangent[i]) / size;
}
} // namespace ordercore

// The share of a relief family's bound that leaves: the mean of cos wi over its sampling,
// 0 past the horizon. parallel and perpendicular are the mirror's tangent along and across
// the axis. Along the axis the integral of sqrt(c^2 - x^2) is exact. Across it a three-point
// Gauss-Hermite rule takes the Gaussian, which is narrow against the unit disc.
// The integral of sqrt(c^2 - x^2) dx, from 0 to x.
PT_SHARED_FN float discChordAntiderivative(float x, float c) noexcept
{
    const float ratio = std::min(1.0f, std::max(-1.0f, x / c));
    return 0.5f * (x * std::sqrt(std::max(0.0f, c * c - x * x)) + c * c * std::asin(ratio));
}

PT_SHARED_FN float reliefFamilyLeavingShare(float parallel, float perpendicular, float reach,
                                            float deviation) noexcept
{
    const float nodes[3] = { 0.0f, 1.7320508f * deviation, -1.7320508f * deviation };
    const float weights[3] = { 2.0f / 3.0f, 1.0f / 6.0f, 1.0f / 6.0f };
    float share = 0.0f;
    for (int i = 0; i < 3; ++i)
    {
        const float across = perpendicular + nodes[i];
        const float c2 = 1.0f - across * across;
        if (!(c2 > 0.0f))
            continue;
        const float c = std::sqrt(c2);
        const float low = std::max(-c, parallel - reach);
        const float high = std::min(c, parallel + reach);
        if (high > low)
            share += weights[i] * (discChordAntiderivative(high, c) - discChordAntiderivative(low, c)) / (2.0f * reach);
    }
    return share;
}

// Inverse of the standard normal's distribution (Acklam's rational fit), so one random
// number makes the relief's offset across its axis.
PT_SHARED_FN float reliefNormalQuantile(float u) noexcept
{
    const float p = std::min(1.0f - 1.0e-6f, std::max(1.0e-6f, u));
    if (p < 0.02425f || p > 0.97575f)
    {
        const float q = std::sqrt(-2.0f * std::log(p < 0.5f ? p : 1.0f - p));
        const float tail = (((((-7.784894e-03f * q - 3.2239646e-01f) * q - 2.4007583f) * q - 2.5497325f) * q + 4.3746641f) * q + 2.9381640f)
                           / ((((7.7846957e-03f * q + 3.2246713e-01f) * q + 2.4451342f) * q + 3.7544087f) * q + 1.0f);
        return p < 0.5f ? tail : -tail;
    }
    const float q = p - 0.5f;
    const float r = q * q;
    return (((((-3.9696830e+01f * r + 2.2094610e+02f) * r - 2.7592851e+02f) * r + 1.3835775e+02f) * r - 3.0664798e+01f) * r + 2.5066283f) * q
           / (((((-5.4476099e+01f * r + 1.6158584e+02f) * r - 1.5569898e+02f) * r + 6.6801312e+01f) * r - 1.3280682e+01f) * r + 1.0f);
}

// The BSDF times cos wi toward wi, which is also the bound times the sampling's density
// per solid angle, so sampleReliefScatter's weight is the constant total power. A
// plane-of-tangents density carries to the sphere by cos wi, and the acceptance is another.
// Symmetric in wo and wi: the offset is, and cos i cos s is once this is divided by cos wi.
PT_SHARED_FN float reliefScatterDensity(const DiffractionOrderDistribution& orders, const std::array<float, 3>& normal,
                                        const std::array<float, 3>& wo, const std::array<float, 3>& wi) noexcept
{
    if (!(orders.reliefScatterBound > 0.0f))
        return 0.0f;
    const float cosO = normal[0] * wo[0] + normal[1] * wo[1] + normal[2] * wo[2];
    const float cosI = normal[0] * wi[0] + normal[1] * wi[1] + normal[2] * wi[2];
    if (!(cosI > 0.0f) || !(cosO > 0.0f))
        return 0.0f;
    float offset[3];
    for (int i = 0; i < 3; ++i)
        offset[i] = wi[i] - normal[i] * cosI + wo[i] - normal[i] * cosO;
    const float offset2 = offset[0] * offset[0] + offset[1] * offset[1] + offset[2] * offset[2];
    const float deviation = orders.reliefAcrossDeviation;
    float sum = 0.0f;
    for (std::size_t k = 0; k < orders.grooveFamilyCount; ++k)
    {
        const auto& axis = orders.grooveFamilyAxes[k];
        const float along = offset[0] * axis[0] + offset[1] * axis[1] + offset[2] * axis[2];
        if (std::fabs(along) < orders.reliefReach)
            sum += std::exp(-std::max(0.0f, offset2 - along * along) / (2.0f * deviation * deviation));
    }
    return orders.reliefScatterBound * sum * cosI * cosI
           / (static_cast<float>(orders.grooveFamilyCount) * 2.0f * orders.reliefReach * 2.5066283f * deviation);
}

// One draw from the relief's band. acceptance is uniform and independent of uAlong and
// uAcross. False when the draw stays in the mirror: past the horizon or by the cos wi share.
PT_SHARED_FN bool sampleReliefScatter(const DiffractionOrderDistribution& orders, const std::array<float, 3>& normal,
                                      const std::array<float, 3>& wo, float acceptance, float uAlong, float uAcross,
                                      std::array<float, 3>& wi) noexcept
{
    const float scaled = uAcross * static_cast<float>(orders.grooveFamilyCount);
    const std::size_t family = std::min(static_cast<std::size_t>(scaled), orders.grooveFamilyCount - 1);
    const auto& axis = orders.grooveFamilyAxes[family];
    const std::array<float, 3> side { normal[1] * axis[2] - normal[2] * axis[1],
                                      normal[2] * axis[0] - normal[0] * axis[2],
                                      normal[0] * axis[1] - normal[1] * axis[0] };
    const float along = orders.reliefReach * (2.0f * uAlong - 1.0f);
    const float across = orders.reliefAcrossDeviation * reliefNormalQuantile(scaled - static_cast<float>(family));
    const float cosO = normal[0] * wo[0] + normal[1] * wo[1] + normal[2] * wo[2];
    float tangent[3];
    float tangent2 = 0.0f;
    for (int i = 0; i < 3; ++i)
    {
        tangent[i] = normal[i] * cosO - wo[i] + axis[i] * along + side[i] * across;
        tangent2 += tangent[i] * tangent[i];
    }
    if (!(tangent2 < 1.0f))
        return false;
    const float cosI = std::sqrt(1.0f - tangent2);
    if (!(acceptance < cosI))
        return false;
    for (int i = 0; i < 3; ++i)
        wi[i] = tangent[i] + normal[i] * cosI;
    return true;
}

PT_SHARED_FN DiffractionOrderProblem evaluateDiffractionOrdersCore (const Description& description,
                                                                    float wavelength,
                                                                    const DiffractionLocalFrame& frame,
                                                                    const std::array<float, 3>& incidentDirection,
                                                                    const std::array<float, 2>& materialUv,
                                                                    DiffractionOrderDistribution& result,
                                                                    float mediumIndex,
                                                                    const LocalGrating* local) noexcept
{
    using namespace ordercore;
    // The host keeps result untouched on a refusal. Device code has no stack for a
    // second distribution, and no caller there reads result after a refusal.
    // Assigning {} built an 8 KB temporary on the GPU stack, so the reset leaves the
    // orders as they are: no reader goes past orderCount, and the loop below writes
    // each entry before it counts it.
#if defined(__CUDACC__)
    DiffractionOrderDistribution& candidate = result;
    clearAllButOrders(candidate);
#else
    DiffractionOrderDistribution candidate;
#endif
    if (!std::isfinite(wavelength) || wavelength < kMinimumWavelengthNanometres
        || wavelength > maximumEvaluatedWavelength)
        return DiffractionOrderProblem::Wavelength;
    if (!std::isfinite(mediumIndex) || mediumIndex < 1.0f)
        return DiffractionOrderProblem::MediumIndex;
    // Inside a medium of index n the light's wavelength is lambda / n. That shortens the
    // grating's momentum step and lengthens the groove's optical depth. The directions here
    // are the medium's own, and the top surface applies Snell when the path leaves. With
    // n = 1 this is exactly the wavelength, so a grating in air computes what it always did.
    const float wavelengthInMedium = wavelength / mediumIndex;
    if (!validFrame(frame) || !unit(incidentDirection))
        return DiffractionOrderProblem::LocalFrame;
    if (!std::isfinite(materialUv[0]) || !std::isfinite(materialUv[1])
        || materialUv[0] < 0.0f || materialUv[0] > 1.0f
        || materialUv[1] < 0.0f || materialUv[1] > 1.0f)
        return DiffractionOrderProblem::MaterialCoordinate;
    if (description.geometry.lattice != GratingLattice::OneDimensional
        && description.geometry.lattice != GratingLattice::CrossedTwoDimensional)
        return DiffractionOrderProblem::Lattice;
    if (description.microstructure.profile != GrooveProfile::BinaryRectangular
        && description.microstructure.profile != GrooveProfile::Sinusoidal
        && description.microstructure.profile != GrooveProfile::BlazedSawtooth)
        return DiffractionOrderProblem::Profile;
    if (description.microstructure.profile == GrooveProfile::Sinusoidal
        && 2.0f * pi * description.microstructure.grooveDepthNanometres
            * mediumIndex / kMinimumWavelengthNanometres > 4.0f)
        return DiffractionOrderProblem::SinusoidalPhase;

    const std::array<float, 3> incidentLocal {
        dot(incidentDirection, frame.tangent),
        dot(incidentDirection, frame.bitangent),
        dot(incidentDirection, frame.normal)
    };
    if (!(incidentLocal[2] > 1.0e-6f))
        return DiffractionOrderProblem::GrazingIncident;

    float coordinate = 0.0f;
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
        return DiffractionOrderProblem::GrooveField;

    const float angle = coordinate * description.grooveField.orientationDegreesPerUnit
        * pi / 180.0f;
    const float cosine = std::cos(angle);
    const float sine = std::sin(angle);
    const auto primaryDirection = local != nullptr ? local->directionUv
        : ordercore::rotateUv(description.geometry.directionUv, cosine, sine);
    const auto secondaryDirection = ordercore::rotateUv(description.geometry.secondaryDirectionUv, cosine, sine);
    const float primaryPeriod = local != nullptr ? local->periodNanometres
        : description.geometry.grooveSpacingNanometres
            + coordinate * description.grooveField.grooveSpacingDeltaNanometresPerUnit;
    const float secondaryPeriod = description.geometry.secondaryGrooveSpacingNanometres
        + coordinate * description.grooveField.secondarySpacingDeltaNanometresPerUnit;
    const bool crossed = description.geometry.lattice == GratingLattice::CrossedTwoDimensional;
    if (!std::isfinite(primaryPeriod) || primaryPeriod < kMinimumGrooveSpacingNanometres
        || primaryPeriod > kMaximumGrooveSpacingNanometres
        || (crossed && (!std::isfinite(secondaryPeriod)
            || secondaryPeriod < kMinimumGrooveSpacingNanometres
            || secondaryPeriod > kMaximumGrooveSpacingNanometres)))
        return DiffractionOrderProblem::LocalPeriod;

    const float substratePower = reflectance(description, wavelength, mediumIndex);
    if (!std::isfinite(substratePower) || substratePower < 0.0f || substratePower > 1.0f)
        return DiffractionOrderProblem::Reflectance;

    const float depth = description.microstructure.grooveDepthNanometres;
    const float duty = description.microstructure.dutyCycle;
    const float roughnessSigma = 2.0f * description.roughness.rmsSlope;
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
            const float shiftX = primary * wavelengthInMedium / primaryPeriod * primaryDirection[0]
                + (crossed ? secondary * wavelengthInMedium / secondaryPeriod
                     * secondaryDirection[0] : 0.0f);
            const float shiftY = primary * wavelengthInMedium / primaryPeriod * primaryDirection[1]
                + (crossed ? secondary * wavelengthInMedium / secondaryPeriod
                     * secondaryDirection[1] : 0.0f);
            const float tangentX = -incidentLocal[0] + shiftX;
            const float tangentY = -incidentLocal[1] + shiftY;
            // 1 - tangent^2 written from the incident's own z. Float can't hold the difference
            // from 1 near the horizon: a mirror order within 0.02 degrees of it rounded to
            // tangent^2 = 1 and vanished, which refused the hit or moved its power away.
            const float outgoingZSquared = incidentLocal[2] * incidentLocal[2]
                + shiftX * (2.0f * incidentLocal[0] - shiftX)
                + shiftY * (2.0f * incidentLocal[1] - shiftY);
            if (!std::isfinite(outgoingZSquared) || outgoingZSquared <= 1.0e-12f)
                continue;
            const float outgoingZ = std::sqrt(outgoingZSquared);
            const float phase = 2.0f * pi * depth
                * (incidentLocal[2] + outgoingZ) / wavelengthInMedium;
            float efficiency = description.pitStatistics == PitStatistics::PressedDisc
                ? pressedDiscEfficiency(primary, duty, phase)
                : latticeEfficiency(description.microstructure.profile,
                                    primary, secondary, crossed,
                                    description.geometry.thirdGrooveFamily,
                                    duty, phase);
            if (primary != 0 || secondary != 0)
                efficiency *= outgoingZ / incidentLocal[2];
            const float roughnessArgument = 2.0f * pi
                * description.roughness.rmsHeightNanometres
                * (incidentLocal[2] + outgoingZ) / wavelengthInMedium;
            efficiency *= std::exp(-(roughnessArgument * roughnessArgument));
            const float power = substratePower * efficiency;
            if (!std::isfinite(power) || power < 0.0f
                || candidate.orderCount >= candidate.orders.size())
                return DiffractionOrderProblem::OrderPower;
            const std::array<float, 3> direction {
                tangentX * frame.tangent[0] + tangentY * frame.bitangent[0]
                    + outgoingZ * frame.normal[0],
                tangentX * frame.tangent[1] + tangentY * frame.bitangent[1]
                    + outgoingZ * frame.normal[1],
                tangentX * frame.tangent[2] + tangentY * frame.bitangent[2]
                    + outgoingZ * frame.normal[2]
            };
            candidate.orders[candidate.orderCount++] = {
                static_cast<std::int8_t>(primary), static_cast<std::int8_t>(secondary),
                direction, power, 0.0f, roughnessSigma
            };
            candidate.resolvedReflectedPower += power;
        }
    }
    if (candidate.orderCount == 0 || !(candidate.resolvedReflectedPower > 0.0f)
        || !std::isfinite(candidate.resolvedReflectedPower))
        return DiffractionOrderProblem::NoResolvedEnergy;
    // An order below the rounding of the probability sum cannot own a sample bin. A
    // binary profile's even orders at duty 0.5 are such noise, and as the last order they
    // left the sampler no last bin, which killed the path. Their power leaves with the
    // unresolved reflection, so no energy is lost. Summing n probabilities rounds by at
    // most n epsilon / 2, so a bin above n epsilon survives as the last one. Scaling by
    // the array's capacity instead took real faint orders, 3.4e-5 of the resolved power.
    const float absentPower = static_cast<float>(candidate.orderCount)
        * std::numeric_limits<float>::epsilon() * candidate.resolvedReflectedPower;
    for (std::size_t index = 0; index < candidate.orderCount; ++index)
        if (candidate.orders[index].reflectedPower <= absentPower)
        {
            candidate.resolvedReflectedPower -= candidate.orders[index].reflectedPower;
            candidate.orders[index].reflectedPower = 0.0f;
        }
    // The scalar model's outgoingZ / incidentZ factor is not energy conserving at a
    // steep view: past about 37 degrees from the surface the orders sum above 1, up to
    // 1.12 at 26 degrees. Refusing that ended the path and left the far side of a disc
    // black. The metal reflects what it reflects, so the orders share exactly that.
    if (candidate.resolvedReflectedPower > substratePower)
    {
        const float share = substratePower / candidate.resolvedReflectedPower;
        for (std::size_t index = 0; index < candidate.orderCount; ++index)
            candidate.orders[index].reflectedPower *= share;
        candidate.resolvedReflectedPower = substratePower;
    }

    candidate.substrateReflectedPower = substratePower;
    if (description.roughness.acrossGrooveSlope > 0.0f || description.roughness.reliefRmsHeightNanometres > 0.0f)
    {
        candidate.acrossGrooveDeviation = 2.0f * description.roughness.acrossGrooveSlope;
        ordercore::addGrooveFamily(candidate, frame, primaryDirection[0], primaryDirection[1]);
        if (crossed)
        {
            ordercore::addGrooveFamily(candidate, frame, secondaryDirection[0], secondaryDirection[1]);
            // The third family's reciprocal vector is g1/d1 - g2/d2.
            if (description.geometry.thirdGrooveFamily)
                ordercore::addGrooveFamily(candidate, frame,
                                           primaryDirection[0] / primaryPeriod - secondaryDirection[0] / secondaryPeriod,
                                           primaryDirection[1] / primaryPeriod - secondaryDirection[1] / secondaryPeriod);
        }
    }
    if (description.roughness.reliefRmsHeightNanometres > 0.0f && candidate.grooveFamilyCount > 0)
    {
        // Rayleigh-Rice, Tayabaly et al. 2016 Eqs. 2 to 5. dP = R (4 pi / lambda)^2 cos i cos s
        // PSD df_x df_y, as d(sin s) = lambda df and cos s dw = lambda^2 df_x df_y. The height's
        // two-sided 1D spectrum integrates to its variance, flat over +-1 / spacing.
        const float phase = 4.0f * pi * description.roughness.reliefRmsHeightNanometres / wavelengthInMedium;
        const float families = static_cast<float>(candidate.grooveFamilyCount);
        // First order says the orders lose x and scatter takes x, which is only good for x well
        // under 1, and three families at the blue end aren't. The bounded split is the total
        // integrated scatter's form (Harvey, CREOL notes, Eq. 7): the orders keep exp(-x) and
        // scatter takes the rest. The rest keeps the first-order shape, as no second-order
        // term is modelled, so past small x reciprocity holds only up to (1 - exp(-x)) / x.
        const float loss = phase * phase * incidentLocal[2] * families;
        const float kept = std::exp(-loss);
        const float bound = substratePower * (1.0f - kept);
        const float reach = wavelengthInMedium / description.roughness.reliefMinimumSpacingNanometres;
        float share = 0.0f;
        for (std::size_t k = 0; k < candidate.grooveFamilyCount; ++k)
        {
            const auto& axis = candidate.grooveFamilyAxes[k];
            const float x = dot(axis, frame.tangent);
            const float y = dot(axis, frame.bitangent);
            // The mirror's tangent is minus the incident one.
            share += reliefFamilyLeavingShare(-incidentLocal[0] * x - incidentLocal[1] * y,
                                              incidentLocal[0] * y - incidentLocal[1] * x, reach, roughnessSigma) / families;
        }
        candidate.reliefScatterBound = bound;
        candidate.reliefLeavingPower = bound * share;
        candidate.reliefReach = reach;
        candidate.reliefAcrossDeviation = roughnessSigma;
        // What scatters leaves every coherent order alike.
        for (std::size_t index = 0; index < candidate.orderCount; ++index)
            candidate.orders[index].reflectedPower *= kept;
        candidate.resolvedReflectedPower *= kept;
    }
    candidate.unresolvedReflectedPower
        = std::max(0.0f, substratePower - candidate.resolvedReflectedPower);
    if (description.pitStatistics == PitStatistics::PressedDisc)
    {
        // Tracks are independent, so the pits' noise is a single stripe's spectrum across
        // them, of width lambda / pit width. Along them s has autocorrelation exp(-2 y / run),
        // a Lorentzian of half width 1 / (pi run) in spatial frequency.
        const float mirrorPhase = 2.0f * pi * depth * 2.0f * incidentLocal[2] / wavelengthInMedium;
        const float halfSine = std::sin(0.5f * mirrorPhase);
        candidate.pitScatterPower = std::min(std::max(0.0f, candidate.unresolvedReflectedPower - candidate.reliefScatterBound),
                                             substratePower * duty * 0.25f * 4.0f * halfSine * halfSine);
        candidate.pitAcrossWidth = wavelengthInMedium / (duty * primaryPeriod);
        candidate.pitAlongWidth = wavelengthInMedium / (pi * kPressedDiscMeanRunNanometres);
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            candidate.pitAcrossAxis[axis] = primaryDirection[0] * frame.tangent[axis]
                                          + primaryDirection[1] * frame.bitangent[axis];
            candidate.pitAlongAxis[axis] = primaryDirection[0] * frame.bitangent[axis]
                                         - primaryDirection[1] * frame.tangent[axis];
        }
    }
    float probabilitySum = 0.0f;
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
        if (candidate.orders[index].selectionProbability > 0.0f)
            lastPositive = index;
    if (lastPositive == candidate.orderCount)
        return DiffractionOrderProblem::NoSampleableEnergy;
    candidate.orders[lastPositive].selectionProbability += 1.0f - probabilitySum;
    candidate.orders[lastPositive].reflectedPower
        = candidate.orders[lastPositive].selectionProbability
            * candidate.resolvedReflectedPower;
    for (std::size_t index = 0; index < candidate.orderCount; ++index)
    {
        const DiffractionOrder& order = candidate.orders[index];
        if (!(unit(order.direction)
              && std::isfinite(order.reflectedPower)
              && order.reflectedPower >= 0.0f
              && std::isfinite(order.selectionProbability)
              && order.selectionProbability >= 0.0f
              && ((order.selectionProbability == 0.0f)
                  == (order.reflectedPower == 0.0f))))
            return DiffractionOrderProblem::InvalidDistribution;
    }

#if !defined(__CUDACC__)
    result = candidate;
#endif
    return DiffractionOrderProblem::None;
}

PT_SHARED_FN DiffractionOrderProblem sampleDiffractionOrderCore (const DiffractionOrderDistribution& distribution,
                                                                 float unitSample,
                                                                 DiffractionOrderSample& result) noexcept
{
    using namespace ordercore;
    DiffractionOrderSample candidate;
    if (!std::isfinite(unitSample) || unitSample < 0.0f || unitSample >= 1.0f)
    {
        return DiffractionOrderProblem::SampleRange;
    }
    if (distribution.orderCount == 0
        || distribution.orderCount > distribution.orders.size()
        || !(distribution.resolvedReflectedPower > 0.0f)
        || !std::isfinite(distribution.resolvedReflectedPower))
    {
        return DiffractionOrderProblem::SampleDistribution;
    }
    float cumulative = 0.0f;
    float powerSum = 0.0f;
    std::size_t lastPositive = distribution.orderCount;
    for (std::size_t index = 0; index < distribution.orderCount; ++index)
    {
        const auto& order = distribution.orders[index];
        if (!unit(order.direction)
            || !std::isfinite(order.selectionProbability)
            || order.selectionProbability < 0.0f
            || !std::isfinite(order.reflectedPower) || order.reflectedPower < 0.0f
            || !std::isfinite(order.angularStandardDeviationRadians)
            || order.angularStandardDeviationRadians < 0.0f
            || order.selectionProbability > 1.0f + probabilityTolerance
            || cumulative > std::numeric_limits<float>::max()
                - order.selectionProbability
            || powerSum > std::numeric_limits<float>::max() - order.reflectedPower)
        {
            return DiffractionOrderProblem::SampleDistribution;
        }
        cumulative += order.selectionProbability;
        powerSum += order.reflectedPower;
        if (!std::isfinite(cumulative) || cumulative > 1.0f + probabilityTolerance
            || !std::isfinite(powerSum))
        {
            return DiffractionOrderProblem::SampleDistribution;
        }
        if ((order.selectionProbability == 0.0f) != (order.reflectedPower == 0.0f))
        {
            return DiffractionOrderProblem::SampleSupport;
        }
        if (order.selectionProbability > 0.0f)
            lastPositive = index;
        const float expectedPower
            = order.selectionProbability * distribution.resolvedReflectedPower;
        if (!std::isfinite(expectedPower)
            || std::abs(order.reflectedPower - expectedPower)
                > distributionConsistencyTolerance
                    * distribution.resolvedReflectedPower)
        {
            return DiffractionOrderProblem::SampleInconsistent;
        }
    }
    if (std::abs(cumulative - 1.0f) > probabilityTolerance
        || lastPositive == distribution.orderCount)
    {
        return DiffractionOrderProblem::SampleNotNormalized;
    }
    if (std::abs(powerSum - distribution.resolvedReflectedPower)
        > distributionConsistencyTolerance * distribution.resolvedReflectedPower)
    {
        return DiffractionOrderProblem::SampleResolvedPower;
    }

    // Canonicalize tolerated rounding onto the last positive bin. This absorbs
    // either a residual or an overshoot while earlier boundaries stay fixed.
    const float probabilityBeforeLast
        = cumulative - distribution.orders[lastPositive].selectionProbability;
    const float canonicalLastProbability = 1.0f - probabilityBeforeLast;
    if (!(canonicalLastProbability > 0.0f))
    {
        return DiffractionOrderProblem::SampleNotNormalized;
    }

    // Positive-mass order i owns [sum(p[0..i-1]), sum(p[0..i])). Exact
    // boundaries belong to the following positive-mass order; zero-mass bins
    // own no samples. The canonical last positive bin closes at one, whatever
    // the running sum rounds to: 0.7% of distributions summed to just under the
    // largest sample and refused it.
    cumulative = 0.0f;
    std::size_t selected = distribution.orderCount;
    for (std::size_t index = 0; index < distribution.orderCount; ++index)
    {
        const float probability = index == lastPositive
            ? canonicalLastProbability : distribution.orders[index].selectionProbability;
        cumulative += probability;
        if (unitSample < cumulative || index == lastPositive)
        {
            selected = index;
            break;
        }
    }
    if (selected == distribution.orderCount)
    {
        return DiffractionOrderProblem::SampleNotNormalized;
    }
    const auto& order = distribution.orders[selected];
    const float canonicalProbability = selected == lastPositive
        ? canonicalLastProbability : order.selectionProbability;
    if (!(canonicalProbability > 0.0f))
    {
        return DiffractionOrderProblem::SampleZeroProbability;
    }
    candidate.order = order;
    candidate.order.selectionProbability = canonicalProbability;
    candidate.order.reflectedPower
        = canonicalProbability * distribution.resolvedReflectedPower;
    candidate.orderIndex = selected;
    candidate.pdf = canonicalProbability;
    candidate.throughput = candidate.order.reflectedPower / candidate.pdf;
    if (!std::isfinite(candidate.throughput) || candidate.throughput < 0.0f)
    {
        return DiffractionOrderProblem::SampleThroughput;
    }
    result = candidate;
    return DiffractionOrderProblem::None;
}
} // namespace diffractionmaterial
