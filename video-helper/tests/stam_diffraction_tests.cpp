#include "gpu_backend/stam_diffraction.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

// Tests for the Stam diffraction layer (stam_diffraction.h). Every optical
// test goes through the layer entry points: diffractionLayer,
// diffractionLayerAtWavelength (the same code with a narrowband spectrum) and
// diffractionLayerEnvironment.
//
// Angle convention: light and viewer lie in the plane spanned by N = +z and
// T = +x. theta_i is the signed polar angle of L and theta_o that of V, both
// measured from N toward +T. The mirror is theta_o = -theta_i and order m lies
// where d * (sin(theta_o) + sin(theta_i)) = m * lambda. Positive m is on the +T
// side of the mirror.
//
// Tolerances: scans step 0.02 deg and must find every predicted order within
// 0.2 deg. Identities that hold exactly in real arithmetic (reciprocity,
// thin-film zero thickness, the backward grating equation) are checked to
// 1e-9 relative in double. The GLSL runs in float32 and is only compared
// textually here.

#ifndef STAM_DIFFRACTION_HEADER_PATH
#define STAM_DIFFRACTION_HEADER_PATH ""
#endif

namespace
{
namespace sd = arbit::stamdiffraction;

constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double scanStartDeg = -89.9;
constexpr double scanStepDeg = 0.02;
constexpr int scanCount = 8991; // -89.9 .. 89.9 inclusive
constexpr double maxAngleErrorDeg = 0.2;
constexpr double probeBandwidthNm = 0.5;
constexpr double peakFloor = 1e-4;

const sd::Vec3 kN{0.0, 0.0, 1.0};
const sd::Vec3 kT{1.0, 0.0, 0.0};
const sd::Vec3 kB{0.0, 1.0, 0.0};
const sd::Vec3 kMetalBase{0.91, 0.92, 0.93};
const sd::Vec3 kDielectricBase{0.04, 0.04, 0.04};
const sd::Vec3 kWhite{1.0, 1.0, 1.0};

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

double radians(double degreesValue)
{
    return degreesValue * pi / 180.0;
}

double degrees(double radiansValue)
{
    return radiansValue * 180.0 / pi;
}

double scanAngleDeg(int index)
{
    return scanStartDeg + scanStepDeg * static_cast<double>(index);
}

sd::Vec3 inPlaneDirection(double thetaDeg)
{
    const double theta = radians(thetaDeg);
    return sd::Vec3{std::sin(theta), 0.0, std::cos(theta)};
}

double sum(sd::Vec3 c)
{
    return c.x + c.y + c.z;
}

double maxAbsDiff(sd::Vec3 a, sd::Vec3 b)
{
    return std::max({std::fabs(a.x - b.x), std::fabs(a.y - b.y), std::fabs(a.z - b.z)});
}

sd::Vec3 chromaticity(sd::Vec3 c)
{
    const double total = sum(c);
    return total > 0.0 ? c * (1.0 / total) : sd::Vec3{};
}

bool finite(sd::Vec3 c)
{
    return std::isfinite(c.x) && std::isfinite(c.y) && std::isfinite(c.z);
}

bool relativelyEqual(double a, double b, double tolerance)
{
    return std::fabs(a - b) <= tolerance * std::max({1.0, std::fabs(a), std::fabs(b)});
}

bool sameBits(const sd::LayerResult& a, const sd::LayerResult& b)
{
    const double av[] = {a.diffraction.x, a.diffraction.y, a.diffraction.z, a.coatSpecular.x,
                         a.coatSpecular.y, a.coatSpecular.z, a.specularScale.x,
                         a.specularScale.y, a.specularScale.z, a.diffractedWeight,
                         a.grooveT.x, a.grooveT.y, a.grooveT.z};
    const double bv[] = {b.diffraction.x, b.diffraction.y, b.diffraction.z, b.coatSpecular.x,
                         b.coatSpecular.y, b.coatSpecular.z, b.specularScale.x,
                         b.specularScale.y, b.specularScale.z, b.diffractedWeight,
                         b.grooveT.x, b.grooveT.y, b.grooveT.z};
    return std::memcmp(av, bv, sizeof(av)) == 0;
}

// Rodrigues rotation of v about the unit axis k.
sd::Vec3 rotateAbout(sd::Vec3 v, sd::Vec3 k, double angleDeg)
{
    const double c = std::cos(radians(angleDeg));
    const double s = std::sin(radians(angleDeg));
    return v * c + sd::cross(k, v) * s + k * (sd::dot(k, v) * (1.0 - c));
}

sd::Vec3 spherical(double polarDeg, double azimuthDeg, double distance)
{
    const double p = radians(polarDeg);
    const double a = radians(azimuthDeg);
    return sd::Vec3{distance * std::sin(p) * std::cos(a), distance * std::sin(p) * std::sin(a),
                    distance * std::cos(p)};
}

// Deterministic generator for sampled checks.
struct Lcg
{
    std::uint64_t state = 12345u;

    double next()
    {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<double>(state >> 11) * (1.0 / 9007199254740992.0);
    }

    sd::Vec3 hemisphere(double minZ)
    {
        const double z = minZ + (1.0 - minZ) * next();
        const double phi = 2.0 * pi * next();
        const double r = std::sqrt(std::max(1.0 - z * z, 0.0));
        return sd::Vec3{r * std::cos(phi), r * std::sin(phi), z};
    }
};

// Straight grooves along +y (T = +x), no coat, opaque base.
sd::LayerParams flatParams(double periodNm, int maxOrder, double roughness)
{
    sd::LayerParams P;
    P.mode = sd::kGrooveLinear;
    P.periodNm = periodNm;
    P.secondPeriodNm = periodNm;
    P.maxOrder = maxOrder;
    P.roughness = roughness;
    P.strength = 1.0;
    P.coatIor = 0.0;
    return P;
}

// A point 40 mm from the disc centre on +x: radial grooves give T = +x there.
sd::Surface discSurface()
{
    sd::Surface S;
    S.position = sd::Vec3{40.0, 0.0, 0.0};
    return S;
}

sd::Vec3 coatTransmission(double coatIor, sd::Vec3 L, sd::Vec3 V, sd::Vec3 N)
{
    const double t = (1.0 - sd::dielectricFresnel(sd::dot(N, L), coatIor))
                   * (1.0 - sd::dielectricFresnel(sd::dot(N, V), coatIor));
    return sd::Vec3{t, t, t};
}

// ---------------------------------------------------------------------------
// Geometry factor
// ---------------------------------------------------------------------------
void testGeometryFactor()
{
    // Mirror pair at theta: S = 2 cos(theta) N, every masking term is 1, so the
    // factor is N.L = cos(theta).
    const double angles[] = {0.0, 30.0, 60.0, 85.0};
    for (double theta : angles)
    {
        const double g = sd::geometryFactor(inPlaneDirection(theta), inPlaneDirection(-theta), kN);
        check(relativelyEqual(g, std::cos(radians(theta)), 1e-12),
              "geometry factor of a mirror pair at " + std::to_string(theta) + " deg is cos");
    }
    const double tiny = 1e-6;
    const double side = std::sqrt(1.0 - tiny * tiny);
    const sd::Vec3 grazingPlus{side, 0.0, tiny};
    const sd::Vec3 grazingMinus{-side, 0.0, tiny};
    const sd::Vec3 L30 = inPlaneDirection(30.0);
    const double g1 = sd::geometryFactor(L30, grazingPlus, kN);
    const double g2 = sd::geometryFactor(grazingPlus, grazingMinus, kN);
    const double g3 = sd::geometryFactor(grazingPlus, grazingPlus, kN);
    const double g4 = sd::geometryFactor(grazingMinus, L30, kN);
    std::printf("geometry N.V=1e-6: %.6g, grazing mirror %.6g, grazing retro %.6g, reverse %.6g\n",
                g1, g2, g3, g4);
    for (double g : {g1, g2, g3, g4})
        check(std::isfinite(g) && g >= 0.0, "geometry factor is finite and non-negative at grazing");
    check(g1 <= 4.0 * sd::dot(kN, L30), "geometry factor stays under the cap times N.L");
    check(g3 <= 4.0 * tiny * (1.0 + 1e-9), "grazing retro-reflection is capped");
    check(sd::geometryFactor(sd::Vec3{1.0, 0.0, 0.0}, L30, kN) == 0.0,
          "light exactly at the horizon gives 0");
    check(sd::geometryFactor(L30, sd::Vec3{0.0, 0.0, -1.0}, kN) == 0.0, "viewer below gives 0");
}

// ---------------------------------------------------------------------------
// Spectrum and gates
// ---------------------------------------------------------------------------
void testColourSanityAndGates()
{
    const sd::Vec3 blue = sd::wavelengthToLinearSrgb(450.0);
    const sd::Vec3 green = sd::wavelengthToLinearSrgb(550.0);
    const sd::Vec3 red = sd::wavelengthToLinearSrgb(650.0);
    check(blue.z > blue.x && blue.z > blue.y, "450 nm is blue-dominant");
    check(green.y > green.x && green.y > green.z, "550 nm is green-dominant");
    check(red.x > red.y && red.x > red.z, "650 nm is red-dominant");

    // Inclusive gates: 380 and 780 are inside, anything beyond is black.
    check(sum(sd::wavelengthToLinearSrgb(380.0)) > 0.0, "380 nm is inside the gate");
    check(sum(sd::wavelengthToLinearSrgb(780.0)) > 0.0, "780 nm is inside the gate");
    const double outside[] = {300.0, 379.999, 780.001, 850.0};
    for (double nm : outside)
        check(sum(sd::wavelengthToLinearSrgb(nm)) == 0.0, std::to_string(nm) + " nm is black");

    // Order loop gates with a flat spectrum: < 380 stops the loop, <= 780 is in,
    // and an order above 780 is skipped without stopping the loop.
    const auto one = [](double) { return sd::Vec3{1.0, 1.0, 1.0}; };
    check(sd::orderSum(1.0, 380.0, 1, one).weight > 0.0, "order at exactly 380 nm counts");
    check(sd::orderSum(1.0, 780.0, 1, one).weight > 0.0, "order at exactly 780 nm counts");
    check(sd::orderSum(1.0, 379.5, 1, one).weight == 0.0, "order below 380 nm is dropped");
    check(sd::orderSum(1.0, 780.5, 1, one).weight == 0.0, "order above 780 nm is dropped");
    const sd::OrderSum skip = sd::orderSum(1.0, 1000.0, 2, one);
    check(relativelyEqual(skip.weight, 0.25 / sd::kSpectrumChannelPeak, 1e-12),
          "order 1 at 1000 nm is skipped and order 2 at 500 nm still counts");

    // The channel peak normalises every order colour to at most 1.
    double peak = 0.0;
    for (int i = 0; i <= 40000; ++i)
        peak = std::max(peak, sd::maxComponent(sd::wavelengthToLinearSrgb(380.0 + 0.01 * i)));
    std::printf("spectrum channel peak %.5f (constant %.4f)\n", peak, sd::kSpectrumChannelPeak);
    check(peak <= sd::kSpectrumChannelPeak && peak > sd::kSpectrumChannelPeak - 1e-3,
          "kSpectrumChannelPeak is the rounded-up channel maximum");

    constexpr int steps = 40000;
    const double stepNm = (sd::kMaxVisibleNm - sd::kMinVisibleNm) / steps;
    sd::Vec3 mean{};
    for (int i = 0; i < steps; ++i)
    {
        const double nm = sd::kMinVisibleNm + (static_cast<double>(i) + 0.5) * stepNm;
        mean = mean + sd::wavelengthToLinearSrgb(nm) * (1.0 / steps);
    }
    std::printf("visible mean rgb %.4f %.4f %.4f\n", mean.x, mean.y, mean.z);
    check(maxAbsDiff(mean, kWhite) < 0.02, "mean colour over 380 to 780 nm is white");

    const sd::LayerParams P = flatParams(1000.0, 3, 0.1);
    const sd::Surface S;
    const sd::Vec3 below{0.0, 0.0, -1.0};
    check(sum(sd::diffractionLayer(P, S, below, inPlaneDirection(10.0), kWhite).diffraction) == 0.0,
          "light below the surface is black");
    check(sum(sd::diffractionLayer(P, S, inPlaneDirection(10.0), below, kWhite).diffraction) == 0.0,
          "viewer below the surface is black");
}

// ---------------------------------------------------------------------------
// Grating scans through the layer: flat, CD and DVD
// ---------------------------------------------------------------------------
std::vector<double> scanPeaks(const sd::LayerParams& P, const sd::Surface& S, sd::Vec3 L,
                              double mirrorDeg, int side, double lambdaNm)
{
    std::vector<double> values(static_cast<std::size_t>(scanCount), 0.0);
    for (int i = 0; i < scanCount; ++i)
    {
        const double thetaO = scanAngleDeg(i);
        const bool inWindow = side > 0 ? thetaO > mirrorDeg : thetaO < mirrorDeg;
        if (inWindow)
            values[static_cast<std::size_t>(i)] = sd::diffractionLayerAtWavelength(
                P, S, L, inPlaneDirection(thetaO), lambdaNm, probeBandwidthNm);
    }
    std::vector<double> peaks;
    for (std::size_t i = 1; i + 1 < values.size(); ++i)
        if (values[i] > peakFloor && values[i] >= values[i - 1] && values[i] > values[i + 1])
            peaks.push_back(scanAngleDeg(static_cast<int>(i)));
    return peaks;
}

struct FanEnds
{
    double cd450 = 0.0;
    double cd650 = 0.0;
    double dvd450 = 0.0;
    double dvd650 = 0.0;
};

void testGratingScans()
{
    struct Case
    {
        const char* name;
        sd::LayerParams params;
    };
    const Case cases[] = {{"flat", flatParams(1000.0, 3, 0.05)},
                          {"cd", sd::kCdPreset},
                          {"dvd", sd::kDvdPreset}};
    const double incidentDeg[] = {30.0, 0.0};
    const double wavelengthsNm[] = {450.0, 550.0, 650.0};
    const sd::Surface S = discSurface();
    FanEnds fan;
    double worst = 0.0;
    std::printf("surface theta_i_deg lambda_nm m predicted_deg measured_deg error_deg\n");
    for (const Case& c : cases)
    {
        const double d = c.params.periodNm;
        for (double thetaI : incidentDeg)
        {
            const sd::Vec3 L = inPlaneDirection(thetaI);
            for (double lambdaNm : wavelengthsNm)
            {
                for (int side : {1, -1})
                {
                    const std::vector<double> peaks = scanPeaks(c.params, S, L, -thetaI, side, lambdaNm);
                    int predictedCount = 0;
                    for (int n = 1; n <= c.params.maxOrder; ++n)
                    {
                        const int m = side * n;
                        const double sinO = m * lambdaNm / d - std::sin(radians(thetaI));
                        if (std::fabs(sinO) >= 1.0)
                        {
                            std::printf("%s %.0f %.0f %+d evanescent - -\n", c.name, thetaI,
                                        lambdaNm, m);
                            continue;
                        }
                        ++predictedCount;
                        const double predictedDeg = degrees(std::asin(sinO));
                        double measuredDeg = std::numeric_limits<double>::quiet_NaN();
                        double errorDeg = std::numeric_limits<double>::infinity();
                        for (double p : peaks)
                            if (std::fabs(p - predictedDeg) < errorDeg)
                            {
                                errorDeg = std::fabs(p - predictedDeg);
                                measuredDeg = p;
                            }
                        worst = std::max(worst, errorDeg);
                        std::printf("%s %.0f %.0f %+d %.4f %.4f %.4f\n", c.name, thetaI, lambdaNm,
                                    m, predictedDeg, measuredDeg, errorDeg);
                        check(errorDeg <= maxAngleErrorDeg,
                              std::string(c.name) + ": order " + std::to_string(m) + " at "
                                  + std::to_string(lambdaNm) + " nm within 0.2 deg");
                        if (thetaI == 0.0 && m == 1)
                        {
                            const bool cd = std::strcmp(c.name, "cd") == 0;
                            const bool dvd = std::strcmp(c.name, "dvd") == 0;
                            if (cd && lambdaNm == 450.0)
                                fan.cd450 = measuredDeg;
                            if (cd && lambdaNm == 650.0)
                                fan.cd650 = measuredDeg;
                            if (dvd && lambdaNm == 450.0)
                                fan.dvd450 = measuredDeg;
                            if (dvd && lambdaNm == 650.0)
                                fan.dvd650 = measuredDeg;
                        }
                        if (n == 1)
                            check(std::fabs(sd::firstOrderWavelengthNm(
                                                L, inPlaneDirection(measuredDeg), kT, d)
                                            - lambdaNm)
                                      < 1.0,
                                  "firstOrderWavelengthNm agrees at the measured order 1 peak");
                    }
                    // Orders exist exactly where the grating equation allows them.
                    check(static_cast<int>(peaks.size()) == predictedCount,
                          std::string(c.name) + ": peak count equals the propagating orders at "
                              + std::to_string(lambdaNm) + " nm, theta_i "
                              + std::to_string(thetaI) + ", side " + std::to_string(side));
                }
            }
        }
    }
    const double cdFan = fan.cd650 - fan.cd450;
    const double dvdFan = fan.dvd650 - fan.dvd450;
    const double cdPredicted = degrees(std::asin(650.0 / sd::kCdTrackPitchNm))
                             - degrees(std::asin(450.0 / sd::kCdTrackPitchNm));
    const double dvdPredicted = degrees(std::asin(650.0 / sd::kDvdTrackPitchNm))
                              - degrees(std::asin(450.0 / sd::kDvdTrackPitchNm));
    std::printf("first-order fan 450-650 nm at normal incidence: cd %.3f deg, dvd %.3f deg, "
                "ratio %.4f, predicted %.4f; worst order error %.4f deg\n",
                cdFan, dvdFan, dvdFan / cdFan, dvdPredicted / cdPredicted, worst);
    check(dvdFan > cdFan, "the DVD fan is wider than the CD fan");
    check(relativelyEqual(dvdFan / cdFan, dvdPredicted / cdPredicted, 0.01),
          "DVD to CD fan ratio matches the grating equation within 1 percent");
}

// Order 0: the coat reflection (CD) and the HRI mirror peak at the mirror.
void testOrderZeroAtMirror()
{
    const double incidentDeg[] = {30.0, 0.0, -45.0};
    for (double thetaI : incidentDeg)
    {
        const sd::Vec3 L = inPlaneDirection(thetaI);
        double bestCd = -1.0;
        double bestCdDeg = 0.0;
        double bestHri = -1.0;
        double bestHriDeg = 0.0;
        for (int i = 0; i < scanCount; ++i)
        {
            const double thetaO = scanAngleDeg(i);
            const sd::Vec3 V = inPlaneDirection(thetaO);
            const double cd = sum(sd::diffractionLayer(sd::kCdPreset, discSurface(), L, V, kWhite)
                                      .coatSpecular);
            const double hri = sum(sd::diffractionLayer(sd::kHoloFoilHriPreset, sd::Surface{}, L,
                                                        V, kWhite)
                                       .coatSpecular);
            if (cd > bestCd)
            {
                bestCd = cd;
                bestCdDeg = thetaO;
            }
            if (hri > bestHri)
            {
                bestHri = hri;
                bestHriDeg = thetaO;
            }
        }
        std::printf("order0 theta_i %.1f predicted %.4f coat %.4f hri %.4f\n", thetaI, -thetaI,
                    bestCdDeg, bestHriDeg);
        check(std::fabs(bestCdDeg + thetaI) <= maxAngleErrorDeg,
              "coat reflection peaks at the mirror for theta_i " + std::to_string(thetaI));
        check(std::fabs(bestHriDeg + thetaI) <= maxAngleErrorDeg,
              "HRI mirror peaks at the mirror for theta_i " + std::to_string(thetaI));
        check(bestCd > 0.0 && bestHri > 0.0, "order 0 is lit");
    }
}

// Inside the first-order fan for theta_i = 30 deg, neighbouring 0.02 deg
// samples must not jump in chromaticity or RGB, the fan must sweep many
// colours, and no two neighbours may be identical (no colour bands).
void testContinuityInsideFirstOrderFan()
{
    const double thetaI = 30.0;
    const sd::LayerParams P = flatParams(1000.0, 3, 0.05);
    const sd::Surface S;
    const sd::Vec3 L = inPlaneDirection(thetaI);
    bool havePrevious = false;
    sd::Vec3 previousRgb{};
    sd::Vec3 previousChroma{};
    double maxChromaStep = 0.0;
    double chromaTravel = 0.0;
    double maxRgbStep = 0.0;
    double maxValue = 0.0;
    int identicalNeighbours = 0;
    int samples = 0;
    for (int i = 0; i < scanCount; ++i)
    {
        const double thetaO = scanAngleDeg(i);
        const double firstOrderNm = 1000.0 * (std::sin(radians(thetaI)) + std::sin(radians(thetaO)));
        if (firstOrderNm < 420.0 || firstOrderNm > 680.0)
            continue;
        const sd::Vec3 rgb =
            sd::diffractionLayer(P, S, L, inPlaneDirection(thetaO), kWhite).diffraction;
        check(sum(rgb) > 0.0, "first-order fan is lit at " + std::to_string(thetaO) + " deg");
        if (sum(rgb) <= 0.0)
            continue;
        const sd::Vec3 chroma = chromaticity(rgb);
        maxValue = std::max(maxValue, sd::maxComponent(rgb));
        ++samples;
        if (havePrevious)
        {
            const double chromaStep = maxAbsDiff(chroma, previousChroma);
            const double rgbStep = maxAbsDiff(rgb, previousRgb);
            maxChromaStep = std::max(maxChromaStep, chromaStep);
            chromaTravel += chromaStep;
            maxRgbStep = std::max(maxRgbStep, rgbStep);
            if (rgbStep < 1e-12)
                ++identicalNeighbours;
        }
        havePrevious = true;
        previousRgb = rgb;
        previousChroma = chroma;
    }
    std::printf("continuity samples %d max_chroma_step %.5f chroma_travel %.3f "
                "max_rgb_step/max %.5f identical_neighbours %d\n",
                samples, maxChromaStep, chromaTravel, maxRgbStep / maxValue, identicalNeighbours);
    check(samples > 500, "first-order fan has enough 0.02 deg samples");
    check(maxChromaStep <= 0.02, "chromaticity has no jumps between 0.02 deg neighbours");
    check(chromaTravel >= 0.5, "first-order fan sweeps through many colours");
    check(maxRgbStep <= 0.05 * maxValue, "RGB has no jumps between 0.02 deg neighbours");
    check(identicalNeighbours == 0, "no constant colour bands inside the fan");
}

// ---------------------------------------------------------------------------
// Along-groove aperture and roughness sanitising
// ---------------------------------------------------------------------------
void testApertureLobe()
{
    const double roughness = 0.05;
    const sd::LayerParams P = flatParams(1000.0, 1, roughness);
    const sd::Surface S;
    const sd::Vec3 V = inPlaneDirection(5.0);
    const sd::Vec3 L0 = inPlaneDirection(30.0);
    const sd::LayerResult R0 = sd::diffractionLayer(P, S, L0, V, kWhite);
    const double reference = sum(R0.diffraction) / sd::geometryFactor(L0, V, kN);
    check(reference > 0.0, "reference order is lit");
    // Keep u = dot(T, L + V) fixed and move L along the grooves: w = L.y.
    for (double w : {0.01, 0.03, 0.06, -0.04})
    {
        const sd::Vec3 L{L0.x, w, std::sqrt(1.0 - L0.x * L0.x - w * w)};
        const sd::LayerResult R = sd::diffractionLayer(P, S, L, V, kWhite);
        const double ratio = sum(R.diffraction) / sd::geometryFactor(L, V, kN) / reference;
        const double expected = std::exp(-(w * w) / (2.0 * roughness * roughness));
        std::printf("aperture w %.3f ratio %.9f expected %.9f\n", w, ratio, expected);
        check(relativelyEqual(ratio, expected, 1e-9),
              "w != 0 lobe is exp(-w^2 / 2 sigma^2) at w " + std::to_string(w));
    }

    sd::LayerParams nanParams = sd::kCdPreset;
    nanParams.roughness = std::numeric_limits<double>::quiet_NaN();
    nanParams.coatRoughness = std::numeric_limits<double>::quiet_NaN();
    sd::LayerParams minParams = sd::kCdPreset;
    minParams.roughness = sd::kMinSigma;
    minParams.coatRoughness = sd::kMinSigma;
    const sd::Vec3 L = inPlaneDirection(30.0);
    const sd::Vec3 Vd = inPlaneDirection(-9.0);
    const sd::LayerResult a = sd::diffractionLayer(nanParams, discSurface(), L, Vd, kMetalBase);
    const sd::LayerResult b = sd::diffractionLayer(minParams, discSurface(), L, Vd, kMetalBase);
    check(finite(a.diffraction) && finite(a.coatSpecular) && finite(a.specularScale),
          "NaN roughness gives finite output");
    check(sameBits(a, b), "NaN roughness behaves as the minimum sigma");
    check(sd::sigmaFromRoughness(std::numeric_limits<double>::quiet_NaN()) == sd::kMinSigma,
          "sigmaFromRoughness sanitises NaN");
}

// ---------------------------------------------------------------------------
// Radial spokes and the tilt sweep (CD preset, 129 x 129 image)
// ---------------------------------------------------------------------------
struct DiscView
{
    sd::Vec3 light = spherical(35.0, 210.0, 1000.0); // mm
    sd::Vec3 camera = spherical(12.0, 30.0, 1000.0);
    sd::Vec3 axis{std::cos(radians(120.0)), std::sin(radians(120.0)), 0.0};
};

struct DiscSample
{
    sd::Vec3 rgb{};
    double firstOrderNm = 0.0;
};

DiscSample shadeDisc(const DiscView& view, double x, double y, double tiltDeg)
{
    sd::Surface S;
    S.N = rotateAbout(kN, view.axis, tiltDeg);
    S.position = rotateAbout(sd::Vec3{x, y, 0.0}, view.axis, tiltDeg);
    S.centre = sd::Vec3{};
    const sd::Vec3 L = sd::normalize(view.light - S.position);
    const sd::Vec3 V = sd::normalize(view.camera - S.position);
    const sd::LayerResult R = sd::diffractionLayer(sd::kCdPreset, S, L, V, kMetalBase);
    return DiscSample{R.diffraction,
                      sd::firstOrderWavelengthNm(L, V, R.grooveT, sd::kCdPreset.periodNm)};
}

struct SpokeHistogram
{
    int bins[36] = {};
    int lit = 0;
    int peakFold = 0;
    double concentration = 0.0;
};

SpokeHistogram renderSpokes(const DiscView& view, double tiltDeg)
{
    constexpr int size = 129;
    std::vector<double> xs;
    std::vector<double> ys;
    std::vector<double> lum;
    double maxLum = 0.0;
    for (int j = 0; j < size; ++j)
    {
        for (int i = 0; i < size; ++i)
        {
            const double x = -60.0 + 120.0 * i / (size - 1);
            const double y = -60.0 + 120.0 * j / (size - 1);
            const double r = std::hypot(x, y);
            if (r < sd::kDiscHubRadiusMm || r > sd::kDiscDataOuterRadiusMm)
                continue;
            const double value = sum(shadeDisc(view, x, y, tiltDeg).rgb);
            xs.push_back(x);
            ys.push_back(y);
            lum.push_back(value);
            maxLum = std::max(maxLum, value);
        }
    }
    SpokeHistogram h;
    for (std::size_t k = 0; k < lum.size(); ++k)
    {
        if (lum[k] <= 0.05 * maxLum)
            continue;
        double angle = degrees(std::atan2(ys[k], xs[k]));
        if (angle < 0.0)
            angle += 360.0;
        ++h.bins[std::min(35, static_cast<int>(angle / 10.0))];
        ++h.lit;
    }
    int fold[18] = {};
    for (int b = 0; b < 18; ++b)
        fold[b] = h.bins[b] + h.bins[b + 18];
    for (int b = 1; b < 18; ++b)
        if (fold[b] > fold[h.peakFold])
            h.peakFold = b;
    const int neighbour = std::max(fold[(h.peakFold + 1) % 18], fold[(h.peakFold + 17) % 18]);
    h.concentration = h.lit > 0 ? static_cast<double>(fold[h.peakFold] + neighbour) / h.lit : 0.0;
    return h;
}

void testRadialSpokesAndTilt()
{
    const DiscView view;
    const SpokeHistogram flat = renderSpokes(view, 0.0);
    const SpokeHistogram tilted = renderSpokes(view, 5.0);
    for (const SpokeHistogram* h : {&flat, &tilted})
    {
        std::printf("spokes lit %d peak_fold_deg %d concentration %.3f bins", h->lit,
                    h->peakFold * 10, h->concentration);
        for (int b : h->bins)
            std::printf(" %d", b);
        std::printf("\n");
    }
    check(flat.lit > 100, "the flat disc shows a lit spoke");
    check(flat.concentration >= 0.9, "lit pixels form one line through the centre");
    const int half = flat.bins[flat.peakFold] + flat.bins[(flat.peakFold + 1) % 36]
                   + flat.bins[(flat.peakFold + 35) % 36];
    const int opposite = flat.bins[flat.peakFold + 18] + flat.bins[(flat.peakFold + 19) % 36]
                       + flat.bins[(flat.peakFold + 17) % 36];
    check(half >= flat.lit / 20 && opposite >= flat.lit / 20,
          "both opposite spokes are lit on the flat disc");

    // Tilt +5 deg about an in-plane axis perpendicular to the spoke.
    check(tilted.lit > 50, "the tilted disc still shows a spoke");
    check(tilted.concentration >= 0.9, "the spoke stays radial after the tilt");
    check(std::abs(tilted.peakFold - flat.peakFold) <= 1
              || std::abs(tilted.peakFold - flat.peakFold) == 17,
          "the spoke keeps its direction within 10 deg");
    double minShift = 1e9;
    double minChromaChange = 1e9;
    for (double r : {30.0, 40.0, 50.0})
    {
        const double x = r * std::cos(radians(210.0));
        const double y = r * std::sin(radians(210.0));
        const DiscSample a = shadeDisc(view, x, y, 0.0);
        const DiscSample b = shadeDisc(view, x, y, 5.0);
        std::printf("tilt sweep r %.0f mm lambda1 %.1f -> %.1f nm\n", r, a.firstOrderNm,
                    b.firstOrderNm);
        check(sum(a.rgb) > 0.0 && sum(b.rgb) > 0.0, "spoke samples are lit before and after");
        minShift = std::min(minShift, std::fabs(b.firstOrderNm - a.firstOrderNm));
        if (r >= 40.0)
            minChromaChange = std::min(minChromaChange, maxAbsDiff(chromaticity(a.rgb), chromaticity(b.rgb)));
    }
    check(minShift > 100.0, "the colour sequence along the radius shifts with the tilt");
    check(minChromaChange > 0.5, "the hue along the spoke moves with the tilt");
}

// ---------------------------------------------------------------------------
// Energy split
// ---------------------------------------------------------------------------
void testEnergy()
{
    struct Case
    {
        const char* name;
        sd::LayerParams params;
        sd::Vec3 base;
    };
    sd::LayerParams glitter = sd::kCdPreset;
    glitter.glitterDensity = 0.3;
    const Case cases[] = {{"cd metal", sd::kCdPreset, kMetalBase},
                          {"cd dielectric", sd::kCdPreset, kDielectricBase},
                          {"dvd metal", sd::kDvdPreset, kMetalBase},
                          {"foil metallized", sd::kHoloFoilMetallizedPreset, kMetalBase},
                          {"foil hri", sd::kHoloFoilHriPreset, sd::Vec3{0.6, 0.2, 0.1}},
                          {"cd glitter", glitter, kMetalBase}};
    for (const Case& c : cases)
    {
        Lcg rng;
        double worstUnderTransmission = -1.0;
        double worstTotal = -1.0;
        int lit = 0;
        bool allFinite = true;
        for (int i = 0; i < 4000; ++i)
        {
            const sd::Vec3 L = rng.hemisphere(0.0);
            const sd::Vec3 V = rng.hemisphere(0.0);
            sd::Surface S = discSurface();
            S.surfaceCoord = sd::Vec2{10.0 * rng.next(), 10.0 * rng.next()};
            const sd::LayerResult R = sd::diffractionLayer(c.params, S, L, V, c.base);
            allFinite = allFinite && finite(R.diffraction) && finite(R.coatSpecular)
                     && finite(R.specularScale);
            const double g = sd::geometryFactor(L, V, kN);
            if (g <= 0.0)
                continue;
            const sd::Vec3 T = coatTransmission(c.params.coatIor, L, V, kN);
            const bool hri = c.params.baseMode == sd::kBaseHri;
            const double b[3] = {c.base.x, c.base.y, c.base.z};
            const double d[3] = {R.diffraction.x, R.diffraction.y, R.diffraction.z};
            const double cs[3] = {R.coatSpecular.x, R.coatSpecular.y, R.coatSpecular.z};
            const double ss[3] = {R.specularScale.x, R.specularScale.y, R.specularScale.z};
            const double t[3] = {T.x, T.y, T.z};
            for (int k = 0; k < 3; ++k)
            {
                // Opaque: orders plus the dimmed mirror never exceed what the coat lets through.
                if (!hri)
                    worstUnderTransmission =
                        std::max(worstUnderTransmission, d[k] / g + b[k] * ss[k] - b[k] * t[k]);
                // Everything (coat, HRI mirror, orders, base) stays at or under 1.
                const double baseWeight = hri ? 1.0 : b[k];
                worstTotal = std::max(worstTotal, (cs[k] + d[k]) / g + baseWeight * ss[k] - 1.0);
                check(ss[k] <= t[k] * (1.0 + 1e-12), "specularScale never exceeds transmission");
            }
            if (R.diffractedWeight > 0.01)
                ++lit;
        }
        std::printf("energy %s: worst excess over transmission %.3g, worst total excess %.3g, "
                    "lit samples %d\n",
                    c.name, worstUnderTransmission, worstTotal, lit);
        check(allFinite, std::string(c.name) + ": output is finite");
        check(worstUnderTransmission <= 1e-9, std::string(c.name) + ": orders fit under the coat transmission");
        check(worstTotal <= 1e-9, std::string(c.name) + ": total reflectance stays at or under 1");
        check(lit > 10, std::string(c.name) + ": enough samples light an order");
    }

    // The aluminium mirror dims exactly where an order is lit.
    const sd::Vec3 L = inPlaneDirection(30.0);
    const double orderDeg = degrees(std::asin(550.0 / sd::kCdTrackPitchNm - 0.5));
    const sd::Vec3 Vorder = inPlaneDirection(orderDeg);
    const sd::LayerResult lit = sd::diffractionLayer(sd::kCdPreset, discSurface(), L, Vorder, kMetalBase);
    const sd::Vec3 Tlit = coatTransmission(sd::kPolycarbonateIor, L, Vorder, kN);
    std::printf("mirror at order 1 550 nm: specularScale %.4f transmission %.4f weight %.4f\n",
                lit.specularScale.x, Tlit.x, lit.diffractedWeight);
    check(lit.specularScale.x < 0.8 * Tlit.x, "the mirror dims where order 1 is lit");
    check(relativelyEqual(lit.specularScale.x, Tlit.x * (1.0 - lit.diffractedWeight), 1e-12),
          "opaque: specularScale = transmission * (1 - weight)");
    for (double thetaO : {-30.0, -31.0, -25.0})
    {
        const sd::Vec3 V = inPlaneDirection(thetaO);
        const sd::LayerResult dark = sd::diffractionLayer(sd::kCdPreset, discSurface(), L, V, kMetalBase);
        const sd::Vec3 T = coatTransmission(sd::kPolycarbonateIor, L, V, kN);
        check(dark.diffractedWeight == 0.0 && maxAbsDiff(dark.specularScale, T) <= 1e-12,
              "no order, no dimming at theta_o " + std::to_string(thetaO));
    }
}

// ---------------------------------------------------------------------------
// Environment lighting
// ---------------------------------------------------------------------------
void testEnvironment()
{
    // Backward grating equation is exact.
    int validCount = 0;
    int evanescentCount = 0;
    double worstAcross = 0.0;
    double worstAlong = 0.0;
    for (double periodNm : {740.0, 1600.0})
        for (double polar = 0.0; polar < 89.0; polar += 7.0)
            for (double azimuth = 0.0; azimuth < 360.0; azimuth += 30.0)
                for (double lambdaNm : {400.0, 550.0, 700.0})
                    for (int order : {-3, -2, -1, 1, 2, 3})
                    {
                        const sd::Vec3 V = spherical(polar, azimuth, 1.0);
                        sd::Vec3 L{};
                        bool valid = false;
                        sd::environmentGratingDirection(V, kN, kT, periodNm, lambdaNm, order, L, valid);
                        const double across = order * lambdaNm / periodNm - sd::dot(V, kT);
                        const double along = -sd::dot(V, kB);
                        check(valid == (1.0 - across * across - along * along > 0.0),
                              "valid flags exactly the propagating orders");
                        if (!valid)
                        {
                            ++evanescentCount;
                            continue;
                        }
                        ++validCount;
                        worstAcross = std::max(worstAcross,
                                               std::fabs(periodNm * sd::dot(kT, L + V) - order * lambdaNm));
                        worstAlong = std::max(worstAlong, std::fabs(sd::dot(kB, L + V)));
                        check(std::fabs(sd::dot(L, L) - 1.0) < 1e-12 && sd::dot(kN, L) > 0.0,
                              "returned L is a unit vector above the surface");
                    }
    std::printf("environment directions valid %d evanescent %d worst |d u - m lambda| %.3g nm, "
                "worst |w| %.3g\n",
                validCount, evanescentCount, worstAcross, worstAlong);
    check(validCount > 1000 && evanescentCount > 100, "both propagating and evanescent orders occur");
    check(worstAcross < 1e-9, "returned L satisfies d * u = m * lambda");
    check(worstAlong < 1e-12, "returned L conserves momentum along the grooves");

    // A small disc light (0.5 deg radius) as the environment against the same
    // light as a point source: the order 1 fan must peak at the same angle and
    // carry the same colours.
    const sd::LayerParams P = flatParams(1000.0, 1, 0.05);
    const sd::Surface S;
    const sd::Vec3 Ls = inPlaneDirection(30.0);
    const double cosRadius = std::cos(radians(0.5));
    const auto disc = [&](sd::Vec3 dir) {
        return sd::dot(dir, Ls) > cosRadius ? kWhite : sd::Vec3{};
    };
    constexpr int envSteps = 1799;
    double bestFwd = -1.0;
    double bestEnv = -1.0;
    double bestFwdDeg = 0.0;
    double bestEnvDeg = 0.0;
    double worstChroma = 0.0;
    int compared = 0;
    for (int i = 0; i < envSteps; ++i)
    {
        const double thetaO = -89.9 + 0.1 * i;
        if (thetaO <= -30.0)
            continue;
        const sd::Vec3 V = inPlaneDirection(thetaO);
        const sd::Vec3 fwd = sd::diffractionLayer(P, S, Ls, V, kWhite).diffraction;
        sd::Vec3 env{};
        for (int j = 0; j < 16; ++j)
            env = env + sd::diffractionLayerEnvironment(P, S, V, kWhite, (j + 0.5) / 16.0, disc).diffraction;
        if (sum(fwd) > bestFwd)
        {
            bestFwd = sum(fwd);
            bestFwdDeg = thetaO;
        }
        if (sum(env) > bestEnv)
        {
            bestEnv = sum(env);
            bestEnvDeg = thetaO;
        }
        const double lambdaNm = 1000.0 * (0.5 + std::sin(radians(thetaO)));
        if (lambdaNm >= 420.0 && lambdaNm <= 680.0 && sum(env) > 0.0 && sum(fwd) > 0.0)
        {
            worstChroma = std::max(worstChroma, maxAbsDiff(chromaticity(fwd), chromaticity(env)));
            ++compared;
        }
    }
    std::printf("environment disc light: forward peak %.2f deg, environment peak %.2f deg, "
                "worst chroma difference %.4f over %d angles\n",
                bestFwdDeg, bestEnvDeg, worstChroma, compared);
    check(bestEnv > 0.0, "the environment lights the order 1 fan");
    check(std::fabs(bestFwdDeg - bestEnvDeg) <= 0.5, "environment response peaks where the forward model peaks");
    check(compared > 100 && worstChroma <= 0.08, "environment colours match the forward colours");

    // Uniform white environment: the orders never return more than the weight
    // they take from the zero order.
    const auto uniform = [](sd::Vec3) { return sd::Vec3{1.0, 1.0, 1.0}; };
    sd::LayerParams strong = flatParams(1000.0, 3, 0.05);
    strong.strength = 3.0;
    for (double thetaO : {0.0, 20.0, 50.0, 80.0})
    {
        const sd::LayerResult R =
            sd::diffractionLayerEnvironment(strong, S, inPlaneDirection(thetaO), kWhite, 0.5, uniform);
        check(sd::maxComponent(R.diffraction) <= R.diffractedWeight + 1e-12,
              "uniform environment: orders stay under their weight");
        check(relativelyEqual(R.specularScale.x + R.diffractedWeight, 1.0, 1e-12),
              "uniform environment: mirror plus orders is 1 without a coat");
    }
}

// ---------------------------------------------------------------------------
// Transparent HRI base
// ---------------------------------------------------------------------------
void testHri()
{
    sd::Surface S;
    S.surfaceCoord = sd::Vec2{0.25, 0.25};
    sd::Vec3 T{};
    double periodNm = 0.0;
    sd::layerGrating(sd::kHoloFoilHriPreset, S, T, periodNm);
    const sd::Vec3 L = kN;
    const double sinO = 550.0 / periodNm;
    const sd::Vec3 V = sd::normalize(T * sinO + kN * std::sqrt(1.0 - sinO * sinO));
    const double cosVH = 0.5 * std::sqrt(sd::dot(L + V, L + V));
    const sd::Vec3 transmission = coatTransmission(sd::kHoloFoilHriPreset.coatIor, L, V, kN);

    const sd::LayerResult red = sd::diffractionLayer(sd::kHoloFoilHriPreset, S, L, V, sd::Vec3{1.0, 0.0, 0.0});
    const sd::LayerResult blue = sd::diffractionLayer(sd::kHoloFoilHriPreset, S, L, V, sd::Vec3{0.0, 0.0, 1.0});
    const double R = sd::dielectricFresnel(cosVH, sd::kZincSulfideIor / sd::kHoloFoilHriPreset.coatIor);
    std::printf("hri R %.5f specularScale %.5f transmission %.5f diffraction %.5f %.5f %.5f\n", R,
                red.specularScale.x, transmission.x, red.diffraction.x, red.diffraction.y,
                red.diffraction.z);
    check(maxAbsDiff(red.diffraction, blue.diffraction) == 0.0,
          "HRI orders do not depend on the print underneath");
    check(sum(red.diffraction) > 0.0 && red.diffractedWeight > 0.0, "HRI foil diffracts");
    check(maxAbsDiff(red.specularScale, transmission * (1.0 - R)) <= 1e-12,
          "the print shows through with transmission * (1 - R)");
    check(red.specularScale.x >= 0.9 * transmission.x, "most of the print shows through");

    // Diffraction scales with the index contrast.
    sd::LayerParams lower = sd::kHoloFoilHriPreset;
    lower.hriIor = 1.9;
    const sd::LayerResult low = sd::diffractionLayer(lower, S, L, V, kWhite);
    const double Rlow = sd::dielectricFresnel(cosVH, 1.9 / lower.coatIor);
    check(relativelyEqual(sum(red.diffraction) / sum(low.diffraction), R / Rlow, 1e-9),
          "HRI diffraction scales with the Fresnel contrast");
    sd::LayerParams matched = sd::kHoloFoilHriPreset;
    matched.hriIor = matched.coatIor;
    const sd::LayerResult none = sd::diffractionLayer(matched, S, L, V, kWhite);
    check(sum(none.diffraction) < 1e-20 && maxAbsDiff(none.specularScale, transmission) <= 1e-12,
          "index-matched HRI is invisible");
    const sd::LayerResult metal =
        sd::diffractionLayer(sd::kHoloFoilMetallizedPreset, S, L, V, sd::Vec3{0.9, 0.9, 0.9});
    check(sum(metal.diffraction) > 5.0 * sum(red.diffraction),
          "metallized foil diffracts far more than HRI");
    check(sum(metal.specularScale) < sum(red.specularScale),
          "metallized foil hides more of what is underneath");

    // Card: the foil stack, never the CD stack.
    check(sd::kCardPreset.baseMode == sd::kBaseHri && sd::kCardPreset.mode == sd::kGrooveCells
              && sd::kCardPreset.periodNm == sd::kHoloFoilHriPreset.periodNm
              && sd::kCardPreset.coatIor != sd::kPolycarbonateIor,
          "the card preset is the HRI foil stack");
}

// ---------------------------------------------------------------------------
// Groove helpers and the per-cell map
// ---------------------------------------------------------------------------
void testGrooveHelpersAndCells()
{
    check(maxAbsDiff(sd::linearGrooveNormal(kN, kT, 0.5 * pi), kB) < 1e-12, "linear grooves rotate T");
    check(maxAbsDiff(sd::radialGrooveNormal(sd::Vec3{0.0, 30.0, 0.0}, sd::Vec3{}, kN), kB) < 1e-12,
          "radial T points away from the centre");
    check(maxAbsDiff(sd::fieldGrooveNormal(kN, kT, sd::Vec2{0.0, 2.0}), kB) < 1e-12,
          "field direction is normalised in tangent space");
    check(maxAbsDiff(sd::fieldGrooveNormal(kN, kT, sd::Vec2{}), kT) < 1e-12,
          "zero field direction falls back to the tangent");
    const sd::Vec3 fallback = sd::tangentAxis(kN, kN);
    check(std::fabs(sd::dot(fallback, fallback) - 1.0) < 1e-12 && std::fabs(sd::dot(fallback, kN)) < 1e-12,
          "tangent parallel to N falls back to a unit perpendicular");

    // Crossed: light normal, viewer on the second grating's order 1 at 550 nm.
    sd::LayerParams crossed = flatParams(1000.0, 1, 0.05);
    crossed.mode = sd::kGrooveCrossed;
    crossed.crossAngle = 0.5 * pi;
    const double sinO = 0.55;
    const sd::Vec3 Vsecond{0.0, sinO, std::sqrt(1.0 - sinO * sinO)};
    const sd::Vec3 Vfirst{sinO, 0.0, std::sqrt(1.0 - sinO * sinO)};
    check(sum(sd::diffractionLayer(crossed, sd::Surface{}, kN, Vsecond, kWhite).diffraction) > 0.0,
          "crossed grating lights the second grating's orders");
    check(sum(sd::diffractionLayer(crossed, sd::Surface{}, kN, Vfirst, kWhite).diffraction) > 0.0,
          "crossed grating lights the first grating's orders");
    check(sum(sd::diffractionLayer(flatParams(1000.0, 1, 0.05), sd::Surface{}, kN, Vsecond, kWhite)
                  .diffraction)
              < 1e-12,
          "a single linear grating leaves the second direction dark");

    // Cells: each cell has its own direction and period, constant inside it.
    const sd::LayerParams& P = sd::kHoloFoilMetallizedPreset;
    double minDiffDeg = 1e9;
    double totalDiffDeg = 0.0;
    int pairs = 0;
    double minPeriod = 1e9;
    double maxPeriod = 0.0;
    auto cellAt = [&](int i, int j, double& angleDeg, double& periodNm) {
        sd::Surface S;
        S.surfaceCoord = sd::Vec2{(i + 0.5) * P.grooveCellSize, (j + 0.5) * P.grooveCellSize};
        sd::Vec3 T{};
        sd::layerGrating(P, S, T, periodNm);
        angleDeg = std::fmod(degrees(std::atan2(T.y, T.x)) + 360.0, 180.0);
    };
    for (int i = -4; i < 8; ++i)
    {
        for (int j = -4; j < 8; ++j)
        {
            double a = 0.0;
            double d = 0.0;
            cellAt(i, j, a, d);
            minPeriod = std::min(minPeriod, d);
            maxPeriod = std::max(maxPeriod, d);
            const int next[2][2] = {{1, 0}, {0, 1}};
            for (const auto& step : next)
            {
                if (i + step[0] >= 8 || j + step[1] >= 8)
                    continue;
                double b = 0.0;
                double e = 0.0;
                cellAt(i + step[0], j + step[1], b, e);
                double diff = std::fabs(a - b);
                diff = std::min(diff, 180.0 - diff);
                minDiffDeg = std::min(minDiffDeg, diff);
                totalDiffDeg += diff;
                ++pairs;
            }
        }
    }
    std::printf("cells: neighbour direction difference min %.3f mean %.2f deg, period %.1f to %.1f nm\n",
                minDiffDeg, totalDiffDeg / pairs, minPeriod, maxPeriod);
    check(minDiffDeg > 0.1, "neighbouring cells have different groove directions");
    check(totalDiffDeg / pairs > 30.0, "cell directions are spread out");
    check(minPeriod >= P.periodNm * (1.0 - P.periodJitter) && maxPeriod <= P.periodNm * (1.0 + P.periodJitter)
              && maxPeriod - minPeriod > 200.0,
          "cell periods spread within the jitter");

    sd::Surface a;
    a.surfaceCoord = sd::Vec2{0.1, 0.1};
    sd::Surface b;
    b.surfaceCoord = sd::Vec2{0.4, 0.4};
    sd::Surface c;
    c.surfaceCoord = sd::Vec2{-0.1, 0.1};
    const sd::Vec3 L = inPlaneDirection(20.0);
    const sd::Vec3 V = inPlaneDirection(5.0);
    const sd::LayerResult ra = sd::diffractionLayer(P, a, L, V, kMetalBase);
    const sd::LayerResult rb = sd::diffractionLayer(P, b, L, V, kMetalBase);
    const sd::LayerResult rc = sd::diffractionLayer(P, c, L, V, kMetalBase);
    check(sameBits(ra, rb), "points in one cell shade identically");
    check(maxAbsDiff(ra.grooveT, rc.grooveT) > 1e-6, "the cell left of the origin is its own cell");
    check(sameBits(ra, sd::diffractionLayer(P, a, L, V, kMetalBase)), "cell shading is deterministic");
}

// ---------------------------------------------------------------------------
// Thin film
// ---------------------------------------------------------------------------
void testThinFilm()
{
    const double films[][2] = {{1.38, 1.5}, {1.47, 1.33}, {2.0, 1.5}, {1.2, 1.6}};
    double worst = 0.0;
    for (const auto& f : films)
        for (double c : {1.0, 0.8, 0.5, 0.2, 0.01})
        {
            const sd::Vec3 film = sd::thinFilmReflectance(c, 0.0, f[0], f[1]);
            const double plain = sd::dielectricFresnel(c, f[1]);
            worst = std::max(worst, maxAbsDiff(film, sd::Vec3{plain, plain, plain}));
        }
    std::printf("thin film zero thickness worst difference from Fresnel %.3g\n", worst);
    check(worst <= 1e-12, "a zero-thickness film is plain Fresnel");

    const sd::LayerParams& oil = sd::kOilSlickPreset;
    const auto film = [&](double c, double t) {
        return chromaticity(sd::thinFilmReflectance(c, t, oil.thinFilmIor, oil.coatIor));
    };
    const double angleShift = maxAbsDiff(film(1.0, 400.0), film(0.7, 400.0));
    const double thicknessShift = maxAbsDiff(film(1.0, 300.0), film(1.0, 500.0));
    std::printf("oil film chroma shift: angle %.3f thickness %.3f\n", angleShift, thicknessShift);
    check(angleShift > 0.05, "film colour shifts with angle");
    check(thicknessShift > 0.05, "film colour shifts with thickness");

    const sd::Vec3 L = inPlaneDirection(20.0);
    const sd::Vec3 V = inPlaneDirection(-20.0);
    const sd::LayerResult R = sd::diffractionLayer(oil, sd::Surface{}, L, V, kDielectricBase);
    const sd::Vec3 chroma = chromaticity(R.coatSpecular);
    check(sum(R.diffraction) == 0.0 && R.diffractedWeight == 0.0, "oil slick has no grating");
    check(sd::maxComponent(chroma) - std::min({chroma.x, chroma.y, chroma.z}) > 0.05,
          "oil slick reflection is coloured");
    const sd::Vec3 tIn = sd::thinFilmReflectance(sd::dot(kN, L), oil.thinFilmThicknessNm, oil.thinFilmIor, oil.coatIor);
    const sd::Vec3 tOut = sd::thinFilmReflectance(sd::dot(kN, V), oil.thinFilmThicknessNm, oil.thinFilmIor, oil.coatIor);
    check(maxAbsDiff(R.specularScale, (kWhite - tIn) * (kWhite - tOut)) <= 1e-12,
          "oil slick transmits (1 - R) in and out");
}

// ---------------------------------------------------------------------------
// Absolute order efficiency
// ---------------------------------------------------------------------------
// Order 1 at 550 nm carries 0.2 * strength times its largest channel. The
// largest channel of the 550 nm colour over the channel peak is pinned as a
// literal (0.720730677651081, from the CMF fit and sRGB matrix), not taken from
// the layer's own diffractedWeight.
void testAbsoluteOrderEfficiency()
{
    constexpr double green550OverPeak = 0.720730677651081;
    const sd::Vec3 L = kN;
    const sd::Vec3 V{0.55, 0.0, std::sqrt(1.0 - 0.55 * 0.55)};
    for (double strength : {1.0, 2.5})
    {
        sd::LayerParams P = flatParams(1000.0, 1, 0.05);
        P.strength = strength;
        const sd::LayerResult R = sd::diffractionLayer(P, sd::Surface{}, L, V, kWhite);
        const double perGeometry = sd::maxComponent(R.diffraction) / sd::geometryFactor(L, V, kN);
        const double expected = 0.2 * strength * green550OverPeak;
        std::printf("efficiency strength %.1f: order 1 at 550 nm %.12f expected %.12f\n", strength,
                    perGeometry, expected);
        check(relativelyEqual(perGeometry, expected, 1e-9),
              "order 1 at 550 nm is 0.2 * strength * peak channel");
        check(R.diffraction.y == sd::maxComponent(R.diffraction), "550 nm is carried by green");
        check(relativelyEqual(R.diffractedWeight, expected, 1e-9), "diffractedWeight is the same energy");
        check(relativelyEqual(R.specularScale.x, 1.0 - expected, 1e-9),
              "the mirror loses exactly that energy");
    }
}

// ---------------------------------------------------------------------------
// Input guards
// ---------------------------------------------------------------------------
bool finiteResult(const sd::LayerResult& R)
{
    return finite(R.diffraction) && finite(R.coatSpecular) && finite(R.specularScale)
        && std::isfinite(R.diffractedWeight) && finite(R.grooveT);
}

void testInputGuards()
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();

    // Indices: no 0 / 0, and absolute indices below 1 act as 1.
    for (double ior : {0.0, -1.0, nan, 1e-9})
    {
        check(std::isfinite(sd::dielectricFresnel(0.5, ior)), "dielectricFresnel is finite for a bad index");
        check(finite(sd::thinFilmReflectance(0.5, 300.0, ior, 1.5))
                  && finite(sd::thinFilmReflectance(0.5, 300.0, 1.4, ior)),
              "thin film is finite for a bad index");
    }
    check(maxAbsDiff(sd::thinFilmReflectance(0.6, 300.0, 0.5, 1.5), sd::thinFilmReflectance(0.6, 300.0, 1.0, 1.5)) == 0.0,
          "film index below 1 acts as 1");
    const sd::Vec3 L = inPlaneDirection(30.0);
    const sd::Vec3 V = inPlaneDirection(-9.0);
    sd::LayerParams hriOne = sd::kHoloFoilHriPreset;
    hriOne.hriIor = 1.0;
    for (double ior : {0.0, 0.5, nan})
    {
        sd::LayerParams hriBad = sd::kHoloFoilHriPreset;
        hriBad.hriIor = ior;
        check(sameBits(sd::diffractionLayer(hriBad, sd::Surface{}, L, V, kWhite),
                       sd::diffractionLayer(hriOne, sd::Surface{}, L, V, kWhite)),
              "HRI index below 1 or NaN acts as 1");
    }
    sd::LayerParams bad = sd::kHoloFoilHriPreset;
    bad.coatIor = nan;
    bad.thinFilmIor = 0.0;
    bad.thinFilmThicknessNm = nan;
    bad.grooveCellSize = 0.0;
    bad.glitterDensity = 0.5;
    bad.glitterCellSize = nan;
    sd::LayerParams film = sd::kCdPreset;
    film.thinFilmThicknessNm = 300.0;
    film.thinFilmIor = 0.0;
    sd::Surface distant;
    distant.surfaceCoord = sd::Vec2{1e30, -1e30};
    for (const sd::LayerParams* P : {&bad, &film})
    {
        check(finiteResult(sd::diffractionLayer(*P, distant, L, V, kMetalBase)), "point layer is finite for bad inputs");
        check(finiteResult(sd::diffractionLayerEnvironment(*P, distant, V, kMetalBase, 0.5,
                                                           [](sd::Vec3) { return sd::Vec3{1.0, 1.0, 1.0}; })),
              "environment layer is finite for bad inputs");
    }

    // Cells: coordinates clamp to +-2^30 before the int32 conversion, and cell
    // sizes below kMinCellSize act as kMinCellSize.
    const std::uint32_t upper = static_cast<std::uint32_t>(static_cast<std::int32_t>(1073741824));
    const std::uint32_t lowerLimit = static_cast<std::uint32_t>(static_cast<std::int32_t>(-1073741824));
    const sd::Cell big = sd::cellOf(sd::Vec2{1e30, -1e30}, 0.5);
    const sd::Cell infinite = sd::cellOf(sd::Vec2{inf, -inf}, 0.5);
    const sd::Cell nanCell = sd::cellOf(sd::Vec2{nan, 3.2}, 0.5);
    check(big.x == upper && big.y == lowerLimit, "huge coordinates clamp to +-2^30");
    check(infinite.x == upper && infinite.y == lowerLimit, "infinite coordinates clamp to +-2^30");
    check(nanCell.x == lowerLimit && nanCell.y == 6u, "a NaN coordinate maps to the lower limit");
    const sd::Vec2 coord{0.01234, -0.05678};
    const sd::Cell reference = sd::cellOf(coord, sd::kMinCellSize);
    check(reference.x == 123u && reference.y == static_cast<std::uint32_t>(static_cast<std::int32_t>(-568)),
          "cells at the minimum size index as expected");
    for (double size : {0.0, -1.0, nan, 1e-12})
    {
        const sd::Cell c = sd::cellOf(coord, size);
        check(c.x == reference.x && c.y == reference.y, "a bad cell size acts as kMinCellSize");
    }
}

// Thin film thickness is clamped to what kSpectralSamples resolves. Below the
// limit the 24-sample result matches a 4000-sample reference.
sd::Vec3 denseThinFilm(double cosTheta, double thicknessNm, double nf, double nb, int samples)
{
    const double c0 = cosTheta;
    const double sin0Sq = 1.0 - c0 * c0;
    const double c1 = std::sqrt(1.0 - sin0Sq / (nf * nf));
    const double c2 = std::sqrt(1.0 - sin0Sq / (nb * nb));
    const double rs01 = (c0 - nf * c1) / (c0 + nf * c1);
    const double rp01 = (nf * c0 - c1) / (nf * c0 + c1);
    const double rs12 = (nf * c1 - nb * c2) / (nf * c1 + nb * c2);
    const double rp12 = (nb * c1 - nf * c2) / (nb * c1 + nf * c2);
    const double phase = 4.0 * pi * nf * thicknessNm * c1;
    sd::Vec3 total{};
    sd::Vec3 norm{};
    for (int k = 0; k < samples; ++k)
    {
        const double lambdaNm = 380.0 + (k + 0.5) * 400.0 / samples;
        const double cosDelta = std::cos(phase / lambdaNm);
        const double r = 0.5 * (sd::airy(rs01, rs12, cosDelta) + sd::airy(rp01, rp12, cosDelta));
        const sd::Vec3 c = sd::wavelengthToLinearSrgb(lambdaNm);
        total = total + c * r;
        norm = norm + c;
    }
    return sd::Vec3{total.x / norm.x, total.y / norm.y, total.z / norm.z};
}

void testThinFilmThicknessLimit()
{
    // 380^2 / (4 n (400 / 24)) nm.
    check(relativelyEqual(sd::thinFilmMaxThicknessNm(1.38), 1569.5652173913043, 1e-12),
          "thin film limit for n = 1.38 is 1569.6 nm");
    check(sd::thinFilmMaxThicknessNm(0.5) == sd::thinFilmMaxThicknessNm(1.0), "limit uses an index of at least 1");
    const double films[][2] = {{1.38, 1.5}, {1.47, 1.33}, {2.3, 1.5}};
    double worst = 0.0;
    for (const auto& f : films)
    {
        const double tMax = sd::thinFilmMaxThicknessNm(f[0]);
        for (double c : {1.0, 0.7, 0.4})
        {
            const sd::Vec3 atLimit = sd::thinFilmReflectance(c, tMax, f[0], f[1]);
            check(maxAbsDiff(sd::thinFilmReflectance(c, 2.0 * tMax, f[0], f[1]), atLimit) == 0.0
                      && maxAbsDiff(sd::thinFilmReflectance(c, 1e9, f[0], f[1]), atLimit) == 0.0,
                  "films thicker than the limit shade as the limit");
            check(maxAbsDiff(sd::thinFilmReflectance(c, std::numeric_limits<double>::quiet_NaN(), f[0], f[1]),
                             sd::thinFilmReflectance(c, 0.0, f[0], f[1]))
                      == 0.0,
                  "NaN thickness acts as 0");
            for (double fraction : {0.25, 0.5, 0.75, 1.0})
            {
                const double t = fraction * tMax;
                worst = std::max(worst, maxAbsDiff(sd::thinFilmReflectance(c, t, f[0], f[1]),
                                                   denseThinFilm(c, t, f[0], f[1], 4000)));
            }
        }
    }
    std::printf("thin film up to the thickness limit: worst difference from 4000 samples %.5f\n", worst);
    check(worst <= 0.003, "24 samples resolve every film up to the thickness limit");
}

// ---------------------------------------------------------------------------
// Glitter
// ---------------------------------------------------------------------------
void testGlitter()
{
    sd::LayerParams P = sd::kCdPreset;
    P.glitterDensity = 0.3;
    int flakes = 0;
    int cells = 0;
    double worstTilt = 0.0;
    sd::Vec2 flakeCoord{};
    sd::Vec2 emptyCoord{};
    bool haveFlake = false;
    bool haveEmpty = false;
    for (int i = 0; i < 100; ++i)
        for (int j = 0; j < 100; ++j)
        {
            const sd::Vec2 coord{(i + 0.5) * P.glitterCellSize, (j + 0.5) * P.glitterCellSize};
            sd::Vec3 normal{};
            const double has = sd::glitterFlake(coord, P.glitterCellSize, P.glitterDensity,
                                                P.glitterMaxTilt, kN, kT, normal);
            sd::Vec3 again{};
            const double hasAgain = sd::glitterFlake(coord, P.glitterCellSize, P.glitterDensity,
                                                     P.glitterMaxTilt, kN, kT, again);
            check(has == hasAgain && maxAbsDiff(normal, again) == 0.0, "flakes are deterministic");
            worstTilt = std::max(worstTilt, std::acos(std::min(1.0, sd::dot(normal, kN))));
            ++cells;
            if (has > 0.0)
            {
                ++flakes;
                if (!haveFlake)
                {
                    flakeCoord = coord;
                    haveFlake = true;
                }
            }
            else if (!haveEmpty)
            {
                emptyCoord = coord;
                haveEmpty = true;
            }
        }
    const double density = static_cast<double>(flakes) / cells;
    std::printf("glitter density %.4f (target %.2f), worst tilt %.4f rad\n", density, P.glitterDensity,
                worstTilt);
    check(std::fabs(density - P.glitterDensity) < 0.02, "flake density matches glitterDensity");
    check(worstTilt <= P.glitterMaxTilt + 1e-12, "flake tilt stays within glitterMaxTilt");
    check(haveFlake && haveEmpty, "the grid has flake and empty cells");

    // A flake cell: mirror about the flake normal sparkles at full strength.
    sd::Surface S = discSurface();
    S.surfaceCoord = flakeCoord;
    sd::Vec3 normal{};
    sd::glitterFlake(flakeCoord, P.glitterCellSize, P.glitterDensity, P.glitterMaxTilt, kN, kT, normal);
    const sd::Vec3 V = kN;
    const sd::Vec3 L = sd::reflect(V * -1.0, normal);
    const sd::LayerResult flake = sd::diffractionLayer(P, S, L, V, kMetalBase);
    const sd::Vec3 expected = kMetalBase * coatTransmission(P.coatIor, L, V, kN) * sd::geometryFactor(L, V, kN);
    check(maxAbsDiff(flake.diffraction, expected) <= 1e-12, "a flake mirrors along its own normal");
    check(sum(flake.specularScale) == 0.0, "a flake replaces the base in its cell");
    check(sameBits(flake, sd::diffractionLayer(P, S, L, V, kMetalBase)), "glitter is deterministic");

    S.surfaceCoord = emptyCoord;
    sd::LayerParams noGlitter = P;
    noGlitter.glitterDensity = 0.0;
    check(sameBits(sd::diffractionLayer(P, S, L, V, kMetalBase), sd::diffractionLayer(noGlitter, S, L, V, kMetalBase)),
          "an empty cell shades like the plain grating");
}

// ---------------------------------------------------------------------------
// Disc helpers
// ---------------------------------------------------------------------------
void testDiscHelpers()
{
    const double hub = sd::kDiscHubRadiusMm;
    const double lip = sd::kDiscDataOuterRadiusMm;
    check(sd::annulusMask(hub - 0.1, hub, lip) == 0.0 && sd::annulusMask(hub, hub, lip) == 1.0
              && sd::annulusMask(lip, hub, lip) == 1.0 && sd::annulusMask(lip + 0.1, hub, lip) == 0.0,
          "annulus mask is 1 on [hub, lip] and 0 outside");
    sd::LayerParams masked = sd::kCdPreset;
    masked.mask = sd::annulusMask(10.0, hub, lip);
    const sd::Vec3 L = inPlaneDirection(30.0);
    const sd::Vec3 V = inPlaneDirection(degrees(std::asin(550.0 / sd::kCdTrackPitchNm - 0.5)));
    const sd::LayerResult hubResult = sd::diffractionLayer(masked, discSurface(), L, V, kMetalBase);
    check(sum(hubResult.diffraction) == 0.0
              && maxAbsDiff(hubResult.specularScale, coatTransmission(sd::kPolycarbonateIor, L, V, kN)) <= 1e-12,
          "the hub has no grating, only the coated mirror");

    double lo = 1e9;
    double hi = 0.0;
    for (int i = 0; i <= 3500; ++i)
    {
        const double r = hub + 0.01 * i;
        const double v = sd::trackRipple(r, 0.03, sd::kDiscRippleAmplitude, sd::kDiscRipplePeriodMm);
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }
    check(lo >= 0.03 * (1.0 - sd::kDiscRippleAmplitude) - 1e-12
              && hi <= 0.03 * (1.0 + sd::kDiscRippleAmplitude) + 1e-12 && hi - lo > 0.003,
          "track ripple modulates roughness by the amplitude");
    check(sd::trackRipple(30.0, 0.03, 0.06, 0.0) == 0.03, "zero ripple period leaves roughness alone");

    const sd::Vec3 Lm = inPlaneDirection(30.0);
    const sd::Vec3 Vm = inPlaneDirection(-30.0);
    check(relativelyEqual(sd::anisotropicMirrorLobe(Lm, Vm, kN, kT, 0.03, 0.08), 1.0, 1e-12),
          "zero-order lobe is 1 at the mirror");
    const sd::Vec3 Vacross = inPlaneDirection(-25.0);
    const sd::Vec3 Valong = sd::normalize(Vm + kB * 0.035);
    check(sd::anisotropicMirrorLobe(Lm, Vacross, kN, kT, 0.03, 0.08)
              > 10.0 * sd::anisotropicMirrorLobe(Lm, Vacross, kN, kT, 0.03, 0.0),
          "smear widens the zero order across the grooves");
    check(relativelyEqual(sd::anisotropicMirrorLobe(Lm, Valong, kN, kT, 0.03, 0.08),
                          sd::anisotropicMirrorLobe(Lm, Valong, kN, kT, 0.03, 0.0), 1e-3),
          "smear leaves the zero order along the grooves nearly alone");
}

// ---------------------------------------------------------------------------
// Anti-fake guard
// ---------------------------------------------------------------------------
std::string readFile(const char* path)
{
    std::ifstream in(path, std::ios::binary);
    std::stringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

std::vector<std::string> identifiers(const std::string& text)
{
    std::vector<std::string> out;
    std::size_t i = 0;
    while (i < text.size())
    {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        if (text.compare(i, 2, "//") == 0)
        {
            while (i < text.size() && text[i] != '\n')
                ++i;
        }
        else if (std::isalpha(c) || c == '_')
        {
            const std::size_t start = i;
            while (i < text.size() && (std::isalnum(static_cast<unsigned char>(text[i])) || text[i] == '_'))
                ++i;
            out.push_back(text.substr(start, i - start));
        }
        else if (std::isdigit(c))
        {
            while (i < text.size() && (std::isalnum(static_cast<unsigned char>(text[i])) || text[i] == '.'))
                ++i;
        }
        else
        {
            ++i;
        }
    }
    return out;
}

std::string lower(std::string s)
{
    for (char& ch : s)
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return s;
}

void testGuardLightResponse()
{
    // (a) The colour follows lambda = d * u / m. Order 1 only, narrowband probe
    // swept over the target wavelength: the argmax must be d * |u|.
    const sd::LayerParams P = flatParams(1000.0, 1, 0.05);
    const sd::Surface S;
    const sd::Vec3 pairs[][2] = {{inPlaneDirection(30.0), inPlaneDirection(5.0)},
                                 {sd::normalize(sd::Vec3{0.3, 0.02, 0.9}), sd::normalize(sd::Vec3{0.1, -0.01, 0.95})},
                                 {inPlaneDirection(-20.0), inPlaneDirection(60.0)}};
    for (const auto& pair : pairs)
    {
        const double predicted = 1000.0 * std::fabs(sd::dot(kT, pair[0] + pair[1]));
        double best = -1.0;
        double bestNm = 0.0;
        for (int k = 0; k <= 1600; ++k)
        {
            const double target = 380.0 + 0.25 * k;
            const double v = sd::diffractionLayerAtWavelength(P, S, pair[0], pair[1], target, 1.0);
            if (v > best)
            {
                best = v;
                bestNm = target;
            }
        }
        std::printf("guard lambda: predicted %.3f nm measured %.2f nm\n", predicted, bestNm);
        check(std::fabs(bestNm - predicted) <= 0.25, "response follows lambda = d * u");
    }

    // Moving L across the grooves changes the colour; moving it along them
    // inside the aperture keeps the colour and only dims it.
    const sd::LayerParams full = flatParams(1000.0, 3, 0.05);
    const sd::Vec3 V = inPlaneDirection(5.0);
    const sd::Vec3 L0 = inPlaneDirection(30.0);
    const sd::Vec3 c0 = chromaticity(sd::diffractionLayer(full, S, L0, V, kWhite).diffraction);
    const sd::Vec3 across = rotateAbout(L0, kB, 2.0);
    const sd::Vec3 along{L0.x, 0.03, std::sqrt(1.0 - L0.x * L0.x - 0.03 * 0.03)};
    const sd::Vec3 cAcross = chromaticity(sd::diffractionLayer(full, S, across, V, kWhite).diffraction);
    const sd::Vec3 cAlong = chromaticity(sd::diffractionLayer(full, S, along, V, kWhite).diffraction);
    check(maxAbsDiff(c0, cAcross) > 0.02, "moving the light across the grooves changes the colour");
    check(maxAbsDiff(c0, cAlong) < 1e-12, "moving the light along the grooves keeps the colour");
}

void testGuardNoHiddenState(const std::string& headerText)
{
    // (b) Same inputs, same bits, whatever ran in between.
    const sd::Vec3 L = inPlaneDirection(25.0);
    const sd::Vec3 V = inPlaneDirection(-3.0);
    sd::Surface S = discSurface();
    S.surfaceCoord = sd::Vec2{1.3, 2.7};
    const auto uniform = [](sd::Vec3 dir) { return sd::Vec3{dir.z, 0.5, 1.0}; };
    for (const sd::LayerParams* P : {&sd::kCdPreset, &sd::kHoloFoilHriPreset, &sd::kOilSlickPreset})
    {
        const sd::LayerResult first = sd::diffractionLayer(*P, S, L, V, kMetalBase);
        const sd::LayerResult firstEnv = sd::diffractionLayerEnvironment(*P, S, V, kMetalBase, 0.25, uniform);
        sd::diffractionLayer(sd::kDvdPreset, discSurface(), V, L, kDielectricBase);
        sd::diffractionLayerEnvironment(sd::kHoloFoilMetallizedPreset, S, L, kWhite, 0.75, uniform);
        check(sameBits(first, sd::diffractionLayer(*P, S, L, V, kMetalBase)), "point layer has no hidden state");
        check(sameBits(firstEnv, sd::diffractionLayerEnvironment(*P, S, V, kMetalBase, 0.25, uniform)),
              "environment layer has no hidden state");
    }

    // No time, frame or global inputs in the GLSL or the C++ twin.
    const std::string glsl = std::string(kStamDiffractionGlsl) + kStamDiffractionEnvironmentGlsl;
    check(!headerText.empty(), "the test can read stam_diffraction.h");
    for (const std::string* text : {&glsl, &headerText})
    {
        for (const std::string& id : identifiers(*text))
        {
            const std::string l = lower(id);
            check(l.find("time") == std::string::npos && l.find("frame") == std::string::npos,
                  "no time or frame identifier: " + id);
            check(id != "uniform" && id != "buffer" && id != "texture" && id != "static"
                      && id != "thread_local" && id != "mutable" && id != "volatile" && id != "rand"
                      && id != "srand" && id != "chrono" && id != "clock",
                  "no global or hidden input: " + id);
        }
    }
}

void testGuardReciprocity()
{
    // (c) f(L, V) / N.L == f(V, L) / N.V for every point-light output, and
    // specularScale is symmetric.
    sd::LayerParams crossed = flatParams(1000.0, 3, 0.05);
    crossed.mode = sd::kGrooveCrossed;
    crossed.crossAngle = pi / 3.0;
    crossed.coatIor = 1.5;
    crossed.thinFilmThicknessNm = 350.0;
    sd::LayerParams glitter = sd::kCdPreset;
    glitter.glitterDensity = 0.3;
    const sd::LayerParams* cases[] = {&sd::kCdPreset, &sd::kDvdPreset, &sd::kHoloFoilHriPreset,
                                      &sd::kHoloFoilMetallizedPreset, &crossed, &glitter};
    Lcg rng;
    double worst = 0.0;
    for (const sd::LayerParams* P : cases)
    {
        for (int i = 0; i < 2000; ++i)
        {
            const sd::Vec3 L = rng.hemisphere(0.05);
            const sd::Vec3 V = rng.hemisphere(0.05);
            sd::Surface S = discSurface();
            S.surfaceCoord = sd::Vec2{5.0 * rng.next(), 5.0 * rng.next()};
            const sd::LayerResult a = sd::diffractionLayer(*P, S, L, V, kMetalBase);
            const sd::LayerResult b = sd::diffractionLayer(*P, S, V, L, kMetalBase);
            const double NL = sd::dot(kN, L);
            const double NV = sd::dot(kN, V);
            const sd::Vec3 da = a.diffraction * (1.0 / NL);
            const sd::Vec3 db = b.diffraction * (1.0 / NV);
            const sd::Vec3 ca = a.coatSpecular * (1.0 / NL);
            const sd::Vec3 cb = b.coatSpecular * (1.0 / NV);
            const double scale = std::max({1.0, sd::maxComponent(da), sd::maxComponent(ca)});
            worst = std::max({worst, maxAbsDiff(da, db) / scale, maxAbsDiff(ca, cb) / scale,
                              maxAbsDiff(a.specularScale, b.specularScale)});
        }
    }
    std::printf("reciprocity worst relative difference %.3g\n", worst);
    check(worst <= 1e-9, "the layer is reciprocal");
}

// ---------------------------------------------------------------------------
// GLSL and C++ parity
// ---------------------------------------------------------------------------
// Every numeric literal in GLSL source, in order, with its sign: a '-' before
// a number (after optional spaces) makes it negative, so "a - 0.065 * x"
// reads as -0.065. Identifiers, // comments and suffixes are skipped; hex
// literals parse through strtod.
std::vector<double> signedLiterals(const char* source)
{
    std::vector<double> values;
    const char* begin = source;
    const char* p = source;
    while (*p != '\0')
    {
        const unsigned char c = static_cast<unsigned char>(*p);
        if (p[0] == '/' && p[1] == '/')
        {
            while (*p != '\0' && *p != '\n')
                ++p;
        }
        else if (std::isalpha(c) || c == '_')
        {
            while (std::isalnum(static_cast<unsigned char>(*p)) || *p == '_')
                ++p;
        }
        else if (std::isdigit(c) || (c == '.' && std::isdigit(static_cast<unsigned char>(p[1]))))
        {
            const char* q = p;
            while (q > begin && std::isspace(static_cast<unsigned char>(q[-1])))
                --q;
            const bool negative = q > begin && q[-1] == '-';
            char* end = nullptr;
            const double value = std::strtod(p, &end);
            values.push_back(negative ? -value : value);
            p = end;
        }
        else
        {
            ++p;
        }
    }
    return values;
}

bool sameValue(double a, double b)
{
    return std::fabs(a - b) <= 1e-12 * std::max(1.0, std::fabs(b));
}

void checkLiteralSequence(const char* name, const char* source, const std::vector<double>& expected)
{
    const std::vector<double> literals = signedLiterals(source);
    std::printf("%s: %zu GLSL literals, %zu expected\n", name, literals.size(), expected.size());
    check(literals.size() == expected.size(), std::string(name) + ": literal count matches");
    const std::size_t count = std::min(literals.size(), expected.size());
    for (std::size_t i = 0; i < count; ++i)
        check(sameValue(literals[i], expected[i]),
              std::string(name) + ": literal " + std::to_string(i) + " is "
                  + std::to_string(literals[i]) + ", C++ has " + std::to_string(expected[i]));
}

std::string collapseSpaces(const std::string& text)
{
    std::string out;
    bool space = false;
    for (char ch : text)
    {
        if (std::isspace(static_cast<unsigned char>(ch)))
        {
            space = true;
            continue;
        }
        if (space && !out.empty())
            out.push_back(' ');
        space = false;
        out.push_back(ch);
    }
    return out;
}

std::vector<double> presetFields(const sd::LayerParams& P)
{
    return {static_cast<double>(P.mode), static_cast<double>(P.baseMode), P.periodNm,
            P.secondPeriodNm, P.periodJitter, P.grooveAngle, P.crossAngle, P.grooveCellSize,
            static_cast<double>(P.maxOrder), P.roughness, P.zeroOrderSmear, P.strength, P.mask,
            P.hriIor, P.coatIor, P.coatRoughness, P.thinFilmThicknessNm, P.thinFilmIor,
            P.glitterDensity, P.glitterCellSize, P.glitterMaxTilt, P.glitterSharpness};
}

void checkGlslPreset(const std::string& glsl, const char* glslName, const sd::LayerParams& cpp)
{
    const std::string head = std::string("const StamLayerParams ") + glslName + " = StamLayerParams(";
    const std::size_t start = glsl.find(head);
    check(start != std::string::npos, std::string("GLSL defines ") + glslName);
    if (start == std::string::npos)
        return;
    const std::size_t open = start + head.size();
    const std::size_t close = glsl.find(')', open);
    const std::string args = glsl.substr(open, close - open);
    struct Named
    {
        const char* name;
        double value;
    };
    const Named names[] = {{"STAM_GROOVE_LINEAR", sd::kGrooveLinear},
                           {"STAM_GROOVE_RADIAL", sd::kGrooveRadial},
                           {"STAM_GROOVE_CROSSED", sd::kGrooveCrossed},
                           {"STAM_GROOVE_FIELD", sd::kGrooveField},
                           {"STAM_GROOVE_CELLS", sd::kGrooveCells},
                           {"STAM_BASE_OPAQUE", sd::kBaseOpaque},
                           {"STAM_BASE_HRI", sd::kBaseHri},
                           {"stamCdTrackPitchNm", sd::kCdTrackPitchNm},
                           {"stamDvdTrackPitchNm", sd::kDvdTrackPitchNm},
                           {"stamZincSulfideIor", sd::kZincSulfideIor},
                           {"stamPolycarbonateIor", sd::kPolycarbonateIor}};
    std::vector<double> values;
    std::stringstream stream(args);
    std::string item;
    while (std::getline(stream, item, ','))
    {
        const std::string trimmed = collapseSpaces(item);
        if (!trimmed.empty() && (std::isdigit(static_cast<unsigned char>(trimmed[0])) || trimmed[0] == '-'
                                 || trimmed[0] == '.'))
        {
            values.push_back(std::strtod(trimmed.c_str(), nullptr));
            continue;
        }
        bool found = false;
        for (const Named& n : names)
            if (trimmed == n.name)
            {
                values.push_back(n.value);
                found = true;
            }
        check(found, std::string(glslName) + ": GLSL argument " + trimmed + " has a C++ twin");
        if (!found)
            values.push_back(std::numeric_limits<double>::quiet_NaN());
    }
    const std::vector<double> expected = presetFields(cpp);
    check(values.size() == expected.size(), std::string(glslName) + ": field count matches");
    for (std::size_t i = 0; i < std::min(values.size(), expected.size()); ++i)
        check(sameValue(values[i], expected[i]),
              std::string(glslName) + ": field " + std::to_string(i) + " matches the C++ preset");
}

void testGlslParity(const std::string& headerText)
{
    const std::string glsl = kStamDiffractionGlsl;
    const std::string env = kStamDiffractionEnvironmentGlsl;
    const char* required[] = {"StamLayerResult stamDiffractionLayer(StamLayerParams P, StamSurface S, vec3 L, vec3 V, vec3 baseSpecular)",
                              "dot(T, L + V)", "dot(B, L + V)", "cross(N, T)",
                              "stamWavelengthToLinearSrgb(", "stamGeometryFactor(",
                              "vec3 stamThinFilmReflectance(float cosTheta, float thicknessNm, float filmIor, float baseIor)"};
    for (const char* text : required)
        check(glsl.find(text) != std::string::npos, std::string("GLSL contains ") + text);
    check(env.find("vec3 stamEnvironmentRadiance(vec3 dir);") != std::string::npos,
          "environment GLSL forward-declares stamEnvironmentRadiance");
    check(env.find("void stamEnvironmentGratingDirection(vec3 V, vec3 N, vec3 T, float periodNm, float lambdaNm, int order, out vec3 L, out bool valid)")
              != std::string::npos,
          "environment GLSL has stamEnvironmentGratingDirection");
    for (const std::string* text : {&glsl, &env})
        check(text->find("#version") == std::string::npos && text->find("precision") == std::string::npos,
              "GLSL snippets carry no #version or precision line");

    // Gate operators, character for character, in both copies.
    // C++ gates are searched after the namespace opens, so the GLSL strings
    // earlier in the header cannot satisfy them.
    const std::size_t twin = headerText.find("namespace arbit::stamdiffraction");
    check(twin != std::string::npos, "header has the C++ twin namespace");
    const std::string cpp = collapseSpaces(twin == std::string::npos ? std::string() : headerText.substr(twin));
    const std::string g = collapseSpaces(glsl + "\n" + env);
    const char* gates[][2] = {
        {"if (nm < stamMinNm || nm > stamMaxNm)", "if (nm < kMinVisibleNm || nm > kMaxVisibleNm)"},
        {"if (lambdaNm < stamMinNm) break;", "if (lambdaNm < kMinVisibleNm) break;"},
        {"if (lambdaNm <= stamMaxNm)", "if (lambdaNm <= kMaxVisibleNm)"},
        {"return roughness > stamMinSigma ? roughness : stamMinSigma;", "return roughness > kMinSigma ? roughness : kMinSigma;"},
        {"if (NL <= 0.0 || NV <= 0.0) return 0.0;", "if (NL <= 0.0 || NV <= 0.0) return 0.0;"},
        {"if (len2 > stamEpsilon * stamEpsilon)", "if (len2 > kEpsilon * kEpsilon)"},
        {"if (len2 <= stamEpsilon * stamEpsilon) return X;", "if (len2 <= kEpsilon * kEpsilon) return X;"},
        {"abs(N.x) < 0.9", "std::fabs(N.x) < 0.9"},
        {"(radius >= innerRadius && radius <= outerRadius)", "(radius >= innerRadius && radius <= outerRadius)"},
        {"if (period <= 0.0) return baseRoughness;", "if (period <= 0.0) return baseRoughness;"},
        {"if (P.thinFilmThicknessNm > 0.0)", "if (P.thinFilmThicknessNm > 0.0)"},
        {"P.coatIor > 1.0 ? P.coatIor : 1.0", "P.coatIor > 1.0 ? P.coatIor : 1.0"},
        {"if (P.coatIor > 1.0) {", "if (P.coatIor > 1.0) {"},
        {"stamHashUnit(h0) < density ? 1.0 : 0.0", "hashUnit(h0) < density ? 1.0 : 0.0"},
        {"if (P.glitterDensity > 0.0", "if (P.glitterDensity > 0.0"},
        {"if (orders.w > 1.0) orders /= orders.w;", "if (orders.weight > 1.0) {"},
        {"valid = normalSq > 0.0 && dot(N, V) > 0.0;", "valid = normalSq > 0.0 && dot(N, V) > 0.0;"},
        {"if (NV <= 0.0) return R;", "if (NV <= 0.0) return R;"},
        {"if (geometry <= 0.0) return R;", "if (geometry <= 0.0) return R;"},
        {"return n > 1.0 ? n : 1.0;", "return n > 1.0 ? n : 1.0;"},
        {"float n = ior > stamEpsilon ? ior : stamEpsilon;", "const double n = ior > kEpsilon ? ior : kEpsilon;"},
        {"float size = cellSize > stamMinCellSize ? cellSize : stamMinCellSize;", "const double size = cellSize > kMinCellSize ? cellSize : kMinCellSize;"},
        {"return q > -stamCellIndexLimit ? (q < stamCellIndexLimit ? q : stamCellIndexLimit) : -stamCellIndexLimit;",
         "return q > -kCellIndexLimit ? (q < kCellIndexLimit ? q : kCellIndexLimit) : -kCellIndexLimit;"},
        {"float t = thicknessNm > 0.0 ? thicknessNm : 0.0;", "double t = thicknessNm > 0.0 ? thicknessNm : 0.0;"},
        {"t = t < tMax ? t : tMax;", "t = t < tMax ? t : tMax;"},
        {"return stamMinNm * stamMinNm / (4.0 * stamIorAtLeastOne(filmIor) * stepNm);",
         "return kMinVisibleNm * kMinVisibleNm / (4.0 * iorAtLeastOne(filmIor) * stepNm);"},
    };
    for (const auto& gate : gates)
    {
        check(g.find(gate[0]) != std::string::npos, std::string("GLSL gate: ") + gate[0]);
        check(cpp.find(gate[1]) != std::string::npos, std::string("C++ gate: ") + gate[1]);
    }

    checkGlslPreset(glsl, "stamCdPreset", sd::kCdPreset);
    checkGlslPreset(glsl, "stamDvdPreset", sd::kDvdPreset);
    checkGlslPreset(glsl, "stamHoloFoilMetallizedPreset", sd::kHoloFoilMetallizedPreset);
    checkGlslPreset(glsl, "stamHoloFoilHriPreset", sd::kHoloFoilHriPreset);
    checkGlslPreset(glsl, "stamOilSlickPreset", sd::kOilSlickPreset);
    check(glsl.find("const StamLayerParams stamCardPreset = stamHoloFoilHriPreset;") != std::string::npos,
          "GLSL card preset is the HRI foil");
    const std::vector<double> card = presetFields(sd::kCardPreset);
    const std::vector<double> hri = presetFields(sd::kHoloFoilHriPreset);
    check(card == hri, "C++ card preset is the HRI foil");

    // Every signed literal, in order. Named entries are the C++ constants;
    // plain numbers are structural literals (0, 1, 0.5, loop bounds, hash
    // shifts) whose sign and position are pinned here.
    const std::vector<double> core = {
        static_cast<double>(sd::kGrooveLinear), static_cast<double>(sd::kGrooveRadial),
        static_cast<double>(sd::kGrooveCrossed), static_cast<double>(sd::kGrooveField),
        static_cast<double>(sd::kGrooveCells), static_cast<double>(sd::kBaseOpaque),
        static_cast<double>(sd::kBaseHri), sd::kMinVisibleNm, sd::kMaxVisibleNm, sd::kMinSigma,
        sd::kGeometryMax, sd::kEpsilon, sd::kSpectrumChannelPeak, sd::kFirstOrderEfficiency,
        sd::kCdTrackPitchNm, sd::kDvdTrackPitchNm, sd::kPolycarbonateIor, sd::kZincSulfideIor,
        sd::kDiscHubRadiusMm, sd::kDiscDataOuterRadiusMm, sd::kDiscRippleAmplitude,
        sd::kDiscRipplePeriodMm, static_cast<double>(sd::kSpectralSamples), sd::kMinCellSize,
        sd::kCellIndexLimit, sd::kCdPreset.periodJitter, sd::kCdPreset.grooveAngle,
        sd::kCdPreset.crossAngle, sd::kCdPreset.grooveCellSize,
        static_cast<double>(sd::kCdPreset.maxOrder), sd::kCdPreset.roughness,
        sd::kCdPreset.zeroOrderSmear, sd::kCdPreset.strength, sd::kCdPreset.mask,
        sd::kCdPreset.coatRoughness, sd::kCdPreset.thinFilmThicknessNm, sd::kCdPreset.thinFilmIor,
        sd::kCdPreset.glitterDensity, sd::kCdPreset.glitterCellSize, sd::kCdPreset.glitterMaxTilt,
        sd::kCdPreset.glitterSharpness, sd::kDvdPreset.periodJitter, sd::kDvdPreset.grooveAngle,
        sd::kDvdPreset.crossAngle, sd::kDvdPreset.grooveCellSize,
        static_cast<double>(sd::kDvdPreset.maxOrder), sd::kDvdPreset.roughness,
        sd::kDvdPreset.zeroOrderSmear, sd::kDvdPreset.strength, sd::kDvdPreset.mask,
        sd::kDvdPreset.coatRoughness, sd::kDvdPreset.thinFilmThicknessNm,
        sd::kDvdPreset.thinFilmIor, sd::kDvdPreset.glitterDensity, sd::kDvdPreset.glitterCellSize,
        sd::kDvdPreset.glitterMaxTilt, sd::kDvdPreset.glitterSharpness,
        sd::kHoloFoilMetallizedPreset.periodNm, sd::kHoloFoilMetallizedPreset.secondPeriodNm,
        sd::kHoloFoilMetallizedPreset.periodJitter, sd::kHoloFoilMetallizedPreset.grooveAngle,
        sd::kHoloFoilMetallizedPreset.crossAngle, sd::kHoloFoilMetallizedPreset.grooveCellSize,
        static_cast<double>(sd::kHoloFoilMetallizedPreset.maxOrder),
        sd::kHoloFoilMetallizedPreset.roughness, sd::kHoloFoilMetallizedPreset.zeroOrderSmear,
        sd::kHoloFoilMetallizedPreset.strength, sd::kHoloFoilMetallizedPreset.mask,
        sd::kHoloFoilMetallizedPreset.coatIor, sd::kHoloFoilMetallizedPreset.coatRoughness,
        sd::kHoloFoilMetallizedPreset.thinFilmThicknessNm,
        sd::kHoloFoilMetallizedPreset.thinFilmIor, sd::kHoloFoilMetallizedPreset.glitterDensity,
        sd::kHoloFoilMetallizedPreset.glitterCellSize,
        sd::kHoloFoilMetallizedPreset.glitterMaxTilt,
        sd::kHoloFoilMetallizedPreset.glitterSharpness, sd::kHoloFoilHriPreset.periodNm,
        sd::kHoloFoilHriPreset.secondPeriodNm, sd::kHoloFoilHriPreset.periodJitter,
        sd::kHoloFoilHriPreset.grooveAngle, sd::kHoloFoilHriPreset.crossAngle,
        sd::kHoloFoilHriPreset.grooveCellSize,
        static_cast<double>(sd::kHoloFoilHriPreset.maxOrder), sd::kHoloFoilHriPreset.roughness,
        sd::kHoloFoilHriPreset.zeroOrderSmear, sd::kHoloFoilHriPreset.strength,
        sd::kHoloFoilHriPreset.mask, sd::kHoloFoilHriPreset.coatIor,
        sd::kHoloFoilHriPreset.coatRoughness, sd::kHoloFoilHriPreset.thinFilmThicknessNm,
        sd::kHoloFoilHriPreset.thinFilmIor, sd::kHoloFoilHriPreset.glitterDensity,
        sd::kHoloFoilHriPreset.glitterCellSize, sd::kHoloFoilHriPreset.glitterMaxTilt,
        sd::kHoloFoilHriPreset.glitterSharpness, sd::kOilSlickPreset.periodNm,
        sd::kOilSlickPreset.secondPeriodNm, sd::kOilSlickPreset.periodJitter,
        sd::kOilSlickPreset.grooveAngle, sd::kOilSlickPreset.crossAngle,
        sd::kOilSlickPreset.grooveCellSize, static_cast<double>(sd::kOilSlickPreset.maxOrder),
        sd::kOilSlickPreset.roughness, sd::kOilSlickPreset.zeroOrderSmear,
        sd::kOilSlickPreset.strength, sd::kOilSlickPreset.mask, sd::kOilSlickPreset.coatIor,
        sd::kOilSlickPreset.coatRoughness, sd::kOilSlickPreset.thinFilmThicknessNm,
        sd::kOilSlickPreset.thinFilmIor, sd::kOilSlickPreset.glitterDensity,
        sd::kOilSlickPreset.glitterCellSize, sd::kOilSlickPreset.glitterMaxTilt,
        sd::kOilSlickPreset.glitterSharpness, -0.5, 0.0, sd::kCmfLobes[0][0], sd::kCmfLobes[0][1],
        sd::kCmfLobes[0][2], sd::kCmfLobes[0][3], sd::kCmfLobes[1][0], sd::kCmfLobes[1][1],
        sd::kCmfLobes[1][2], sd::kCmfLobes[1][3], sd::kCmfLobes[2][0], sd::kCmfLobes[2][1],
        sd::kCmfLobes[2][2], sd::kCmfLobes[2][3], sd::kCmfLobes[3][0], sd::kCmfLobes[3][1],
        sd::kCmfLobes[3][2], sd::kCmfLobes[3][3], sd::kCmfLobes[4][0], sd::kCmfLobes[4][1],
        sd::kCmfLobes[4][2], sd::kCmfLobes[4][3], sd::kCmfLobes[5][0], sd::kCmfLobes[5][1],
        sd::kCmfLobes[5][2], sd::kCmfLobes[5][3], sd::kCmfLobes[6][0], sd::kCmfLobes[6][1],
        sd::kCmfLobes[6][2], sd::kCmfLobes[6][3], sd::kXyzToLinearSrgb[0],
        sd::kXyzToLinearSrgb[1], sd::kXyzToLinearSrgb[2], sd::kXyzToLinearSrgb[3],
        sd::kXyzToLinearSrgb[4], sd::kXyzToLinearSrgb[5], sd::kXyzToLinearSrgb[6],
        sd::kXyzToLinearSrgb[7], sd::kXyzToLinearSrgb[8], 0.0, sd::kWhiteBalance[0],
        sd::kWhiteBalance[1], sd::kWhiteBalance[2], 0.0, 0.0, 0.0, 4.0, 0.0, 1.0, 1.0, 2.0,
        static_cast<double>(sd::kHashMulX), static_cast<double>(sd::kHashMulY), 16.0,
        static_cast<double>(sd::kHashMulA), 15.0, static_cast<double>(sd::kHashMulB), 16.0, 8.0,
        1.0, 16777216.0, 0.9, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, sd::kPi, 3.0, 1.0, 2.0, 4.0, -1.0,
        1.0, 0.0, 0.0, 1.0, sd::kTwoPi, -0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.5, 2.0, 1.0, 4.0,
        0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, sd::kFourPi, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 1.0,
        1.0, sd::kTwoPi, 1.0, 2.0, 1.0, 0.0, 1.0, -1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5,
        1.0, 1.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 1.0, 1.0,
    };
    const std::vector<double> environment = {
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, -1.0, 1.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0,
        1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0,
    };
    checkLiteralSequence("core GLSL", kStamDiffractionGlsl, core);
    checkLiteralSequence("environment GLSL", kStamDiffractionEnvironmentGlsl, environment);
}
} // namespace

int main()
{
    const std::string headerText = readFile(STAM_DIFFRACTION_HEADER_PATH);

    testGeometryFactor();
    testColourSanityAndGates();
    testGratingScans();
    testOrderZeroAtMirror();
    testContinuityInsideFirstOrderFan();
    testApertureLobe();
    testRadialSpokesAndTilt();
    testEnergy();
    testEnvironment();
    testHri();
    testGrooveHelpersAndCells();
    testThinFilm();
    testThinFilmThicknessLimit();
    testAbsoluteOrderEfficiency();
    testInputGuards();
    testGlitter();
    testDiscHelpers();
    testGuardLightResponse();
    testGuardNoHiddenState(headerText);
    testGuardReciprocity();
    testGlslParity(headerText);

    std::printf("%d checks, %d failures\n", checks, failures);
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
