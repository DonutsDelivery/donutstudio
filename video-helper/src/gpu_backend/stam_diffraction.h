#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

// Stam-style diffraction grating layer, after GPU Gems 1 chapter 8
// ("Simulating Diffraction", Jos Stam). Any material can carry it: the
// material system calls one layer function and composes its result with the
// material's own BSDF.
//
// The header holds the same math twice:
//   kStamDiffractionGlsl             core GLSL (point lights, presets, helpers)
//   kStamDiffractionEnvironmentGlsl  environment-lit GLSL, appended after the core
//                                    only by shaders that define stamEnvironmentRadiance
//   arbit::stamdiffraction::*        double-precision C++ twin used by the tests
// The GLSL and the C++ must change together. stam_diffraction_tests.cpp compares
// every signed numeric literal in both GLSL strings, in order, with the C++
// constants, and checks the comparison operators of the wavelength and horizon
// gates in both copies. C++ names drop the "stam" prefix (stamDiffractionLayer is
// diffractionLayer, stamCell is cellOf) and use k-prefixed constants (stamMinNm
// is kMinVisibleNm). The C++ twin is plain double math with no GPU types and no
// global state, so a CPU path tracer can call it per sample.
// The GLSL runs in float32 and the twin in double. In highp, wavelengths agree
// within about 0.01 nm and colours within about 1e-4 relative. The twin is the
// reference. On GLSL ES the including shader must declare highp float: periodNm *
// |u| reaches the thousands and mediump loses the nanometres the colours need.
// The snippets have no #version or precision line.
//
// ---------------------------------------------------------------------------
// Conventions
// ---------------------------------------------------------------------------
// All vectors are unit length and in one space.
//   N  surface normal.
//   T  across-groove direction in the surface plane (perpendicular to N). The
//      groove helpers produce it; the layer resolves it from the params.
//   B  = cross(N, T), along the grooves.
//   L  toward the light, V toward the viewer, both pointing away from the surface.
//   u = dot(T, L + V), w = dot(B, L + V). The mirror direction is u = w = 0.
// With light and viewer in the T-N plane and signed polar angles theta_i (of L)
// and theta_o (of V) measured from N toward +T, u = sin(theta_i) + sin(theta_o).
// The mirror is theta_o = -theta_i and order m appears where
// periodNm * (sin(theta_o) + sin(theta_i)) = m * lambda. With the incident ray's
// propagation angle theta_in = -theta_i this is the textbook
// d * (sin(theta_o) - sin(theta_in)) = m * lambda. Order n >= 1 shows
// lambda_n = periodNm * |u| / n, so one loop covers both signs of m.
//
// ---------------------------------------------------------------------------
// Terms
// ---------------------------------------------------------------------------
// Spectrum. CIE 1931 2 degree colour matching functions from the multi-lobe
//   piecewise Gaussian fit of Wyman, Sloan and Shirley, "Simple Analytic
//   Approximations to the CIE XYZ Color Matching Functions", JCGT 2(2), 2013.
//   XYZ goes to linear sRGB through the D65 matrix, negatives clamp to 0, and
//   the result is 0 outside 380 to 780 nm. kWhiteBalance scales each channel so
//   the mean over 380 to 780 nm is (1, 1, 1).
// Groove aperture. Momentum along the grooves is conserved, so each order is
//   multiplied by exp(-w^2 / (2 sigma^2)), sigma = roughness clamped to at
//   least 0.02 (NaN roughness becomes 0.02). Across the grooves the colour
//   follows u directly, so there is no lobe in u.
// Order efficiency. Order n carries strength * kFirstOrderEfficiency / n^2 (the
//   1/n^2 envelope of a binary grating) times the order colour divided by
//   kSpectrumChannelPeak, so no normalised channel exceeds 1. The energy weight
//   of an order is its efficiency times its largest normalised channel. The
//   layer moves exactly that weight out of the zero order.
// Geometry. The chapter prints no geometric term and refers to Stam's 1999
//   paper "Diffraction Shaders". That paper's Kirchhoff factor
//   (1 + L.V)^2 / (N.L N.V (N.(L+V))^2) is multiplied by the Cook-Torrance
//   V-cavity masking min(1, 2 N.H N.V / V.H, 2 N.H N.L / V.H), capped at 4, and
//   multiplied by N.L to give radiance per unit irradiance. With S = L + V the
//   product is written min(|S|^4 / (4 N.L N.V (N.S)^2), |S|^2 / (N.L N.S),
//   |S|^2 / (N.V N.S)), which never forms inf * 0. Everything before the final
//   N.L is symmetric in L and V, so every point-light output except
//   specularScale satisfies f(L, V) / N.L == f(V, L) / N.V, and specularScale is
//   symmetric.
// Coat. A clear dielectric coat (coatIor > 1) reflects
//   Fresnel(V.H) * exp(-|S_tangential|^2 / (2 coatSigma^2)) and transmits
//   (1 - F(N.L)) (1 - F(N.V)) into and out of the stack below. A flat coat does
//   not move the orders: Snell's law keeps the tangential wave vector, so the
//   grating equation in air equals the one inside the coat, and refracting L and
//   V into the coat first would give the same colours. The coat's thickness only
//   offsets the image and is ignored. With thinFilmThicknessNm > 0 the coat's top
//   reflectance and transmission come from a thin film of thinFilmIor over the
//   coat medium.
// Base modes.
//   STAM_BASE_OPAQUE: the grating is embossed in an opaque reflector (aluminium,
//     silver). The orders are tinted by baseSpecular and take their energy from
//     the base's zero order: specularScale = transmission * (1 - weight).
//   STAM_BASE_HRI: the grating is embossed in a transparent high-index coating
//     (ZnS, hriIor about 2.3) over the material. The coating reflects
//     R = Fresnel(V.H, hriIor / surround), where surround is coatIor (or 1 with
//     no coat). R sets the strength of both the orders and the coating's own
//     mirror reflection. The material underneath shows through with
//     specularScale = transmission * (1 - R), so the rainbow floats over the print.
// Groove modes. LINEAR: T is the tangent rotated by grooveAngle. RADIAL: circular
//   grooves around S.centre, so T is radial (CD, DVD). CROSSED: two linear
//   gratings at crossAngle, with periodNm and secondPeriodNm, each carrying half
//   the efficiency. FIELD: T from a tangent-space direction (a groove map).
//   CELLS: kinegram or pixel hologram. Each cell of size grooveCellSize in
//   S.surfaceCoord gets its own groove angle (uniform in [0, pi) added to
//   grooveAngle) and period (periodNm * (1 + periodJitter * (2h - 1))) from an
//   integer hash. A shader may instead override periodNm per pixel from a map.
// Glitter. Optional. Cells of size glitterCellSize in S.surfaceCoord hold a flake
//   with probability glitterDensity, chosen by the same integer hash, so the
//   result is identical on every run and every GPU. A flake is a mirror tilted
//   by up to glitterMaxTilt radians that replaces the grating and the base in its
//   cell (specularScale = 0 there).
// Input guards. coatIor, hriIor and thinFilmIor below 1 (or NaN) act as 1.
//   Cell sizes below 1e-4 act as 1e-4 and cell indices clamp to +-2^30, so the
//   int32 conversion is always defined. Film thickness clamps to
//   stamThinFilmMaxThicknessNm(thinFilmIor), the thickest film the 24 spectral
//   samples resolve without aliasing.
// No hidden state. Nothing reads time, frame counters or uniforms; every input
//   is an argument.
//
// ---------------------------------------------------------------------------
// Integration notes for the renderer slice
// ---------------------------------------------------------------------------
// Call stamDiffractionLayer once per light (or stamDiffractionLayerEnvironment
// once for the HDRI) per shading point, then compose:
//   colour = coatSpecular + diffraction + baseBsdf * specularScale
// where baseBsdf is the material's own response (specular and diffuse lobes).
// The visual target is a CD data side: polycarbonate on top (n = 1.58), Fresnel
// in and out of it, aluminium below. The aluminium's zero order is an
// anisotropic mirror smeared across the tracks: align the base BSDF's
// anisotropy with result.grooveT and use params.zeroOrderSmear
// (stamAnisotropicMirrorLobe is the reference shape). The orders take their
// energy from that mirror through specularScale, so the mirror dims where the
// rainbow is bright. The HDRI path solves the grating equation backwards per
// wavelength and order and looks up the environment there. The hub and outer
// lip carry no grating: pass mask = stamAnnulusMask(radius, hub, dataOuter).
// Track ripple modulates roughness by radius (stamTrackRipple). Thin film
// (label foil, oil) and glitter are per-material options in the same params.
// Studio cards, tonemapping and the transmissive disc body belong to the
// renderer.

inline constexpr const char* kStamDiffractionGlsl = R"GLSL(
const int STAM_GROOVE_LINEAR = 0;
const int STAM_GROOVE_RADIAL = 1;
const int STAM_GROOVE_CROSSED = 2;
const int STAM_GROOVE_FIELD = 3;
const int STAM_GROOVE_CELLS = 4;
const int STAM_BASE_OPAQUE = 0;
const int STAM_BASE_HRI = 1;

const float stamMinNm = 380.0;
const float stamMaxNm = 780.0;
const float stamMinSigma = 0.02;
const float stamGeometryMax = 4.0;
const float stamEpsilon = 1e-6;
const float stamSpectrumChannelPeak = 6.9513;
const float stamFirstOrderEfficiency = 0.2;
const float stamCdTrackPitchNm = 1600.0;
const float stamDvdTrackPitchNm = 740.0;
const float stamPolycarbonateIor = 1.58;
const float stamZincSulfideIor = 2.3;
const float stamDiscHubRadiusMm = 23.0;
const float stamDiscDataOuterRadiusMm = 58.0;
const float stamDiscRippleAmplitude = 0.06;
const float stamDiscRipplePeriodMm = 1.5;
const int stamSpectralSamples = 24;
const float stamMinCellSize = 1e-4;
const float stamCellIndexLimit = 1073741824.0;

struct StamLayerParams {
    int mode;
    int baseMode;
    float periodNm;
    float secondPeriodNm;
    float periodJitter;
    float grooveAngle;
    float crossAngle;
    float grooveCellSize;
    int maxOrder;
    float roughness;
    float zeroOrderSmear;
    float strength;
    float mask;
    float hriIor;
    float coatIor;
    float coatRoughness;
    float thinFilmThicknessNm;
    float thinFilmIor;
    float glitterDensity;
    float glitterCellSize;
    float glitterMaxTilt;
    float glitterSharpness;
};

struct StamSurface {
    vec3 N;
    vec3 tangent;
    vec3 position;
    vec3 centre;
    vec2 fieldDirection;
    vec2 surfaceCoord;
};

struct StamLayerResult {
    vec3 diffraction;
    vec3 coatSpecular;
    vec3 specularScale;
    float diffractedWeight;
    vec3 grooveT;
};

const StamLayerParams stamCdPreset = StamLayerParams(STAM_GROOVE_RADIAL, STAM_BASE_OPAQUE,
    stamCdTrackPitchNm, stamCdTrackPitchNm, 0.0, 0.0, 0.0, 0.5, 3, 0.03, 0.08, 3.0, 1.0,
    stamZincSulfideIor, stamPolycarbonateIor, 0.02, 0.0, 1.38, 0.0, 0.1, 0.35, 0.05);
const StamLayerParams stamDvdPreset = StamLayerParams(STAM_GROOVE_RADIAL, STAM_BASE_OPAQUE,
    stamDvdTrackPitchNm, stamDvdTrackPitchNm, 0.0, 0.0, 0.0, 0.5, 2, 0.03, 0.08, 3.0, 1.0,
    stamZincSulfideIor, stamPolycarbonateIor, 0.02, 0.0, 1.38, 0.0, 0.1, 0.35, 0.05);
const StamLayerParams stamHoloFoilMetallizedPreset = StamLayerParams(STAM_GROOVE_CELLS,
    STAM_BASE_OPAQUE, 1000.0, 1000.0, 0.25, 0.0, 0.0, 0.5, 2, 0.05, 0.0, 3.0, 1.0,
    stamZincSulfideIor, 1.5, 0.02, 0.0, 1.38, 0.0, 0.1, 0.35, 0.05);
const StamLayerParams stamHoloFoilHriPreset = StamLayerParams(STAM_GROOVE_CELLS,
    STAM_BASE_HRI, 1000.0, 1000.0, 0.25, 0.0, 0.0, 0.5, 2, 0.05, 0.0, 3.0, 1.0,
    stamZincSulfideIor, 1.5, 0.02, 0.0, 1.38, 0.0, 0.1, 0.35, 0.05);
const StamLayerParams stamCardPreset = stamHoloFoilHriPreset;
const StamLayerParams stamOilSlickPreset = StamLayerParams(STAM_GROOVE_LINEAR,
    STAM_BASE_OPAQUE, 1000.0, 1000.0, 0.0, 0.0, 0.0, 0.5, 0, 0.03, 0.0, 0.0, 0.0,
    stamZincSulfideIor, 1.33, 0.02, 400.0, 1.47, 0.0, 0.1, 0.35, 0.05);

float stamCmfLobe(float nm, float meanNm, float sigmaBelowNm, float sigmaAboveNm) {
    float t = (nm - meanNm) / (nm < meanNm ? sigmaBelowNm : sigmaAboveNm);
    return exp(-0.5 * t * t);
}

vec3 stamWavelengthToLinearSrgb(float nm) {
    if (nm < stamMinNm || nm > stamMaxNm) return vec3(0.0);
    float x = 1.056 * stamCmfLobe(nm, 599.8, 37.9, 31.0)
            + 0.362 * stamCmfLobe(nm, 442.0, 16.0, 26.7)
            - 0.065 * stamCmfLobe(nm, 501.1, 20.4, 26.2);
    float y = 0.821 * stamCmfLobe(nm, 568.8, 46.9, 40.5)
            + 0.286 * stamCmfLobe(nm, 530.9, 16.3, 31.1);
    float z = 1.217 * stamCmfLobe(nm, 437.0, 11.8, 36.0)
            + 0.681 * stamCmfLobe(nm, 459.0, 26.0, 13.8);
    vec3 rgb = vec3( 3.2404542 * x - 1.5371385 * y - 0.4985314 * z,
                    -0.9692660 * x + 1.8760108 * y + 0.0415560 * z,
                     0.0556434 * x - 0.2040259 * y + 1.0572252 * z);
    return max(rgb, vec3(0.0)) * vec3(2.2704, 3.4666, 3.6590);
}

float stamSigma(float roughness) {
    return roughness > stamMinSigma ? roughness : stamMinSigma;
}

float stamGeometryFactor(vec3 L, vec3 V, vec3 N) {
    float NL = dot(N, L);
    float NV = dot(N, V);
    if (NL <= 0.0 || NV <= 0.0) return 0.0;
    vec3 S = L + V;
    float s2 = dot(S, S);
    float nS = dot(N, S);
    float a = max(NL, stamEpsilon);
    float b = max(NV, stamEpsilon);
    float brdf = min(s2 * s2 / (4.0 * a * b * nS * nS), min(s2 / (a * nS), s2 / (b * nS)));
    return NL * min(brdf, stamGeometryMax);
}

vec4 stamOrderSum(float u, float periodNm, int maxOrder) {
    vec4 sum = vec4(0.0);
    for (int n = 1; n <= maxOrder; ++n) {
        float lambdaNm = periodNm * abs(u) / float(n);
        if (lambdaNm < stamMinNm) break;
        if (lambdaNm <= stamMaxNm) {
            vec3 c = stamWavelengthToLinearSrgb(lambdaNm) / stamSpectrumChannelPeak;
            float eta = 1.0 / float(n * n);
            sum += eta * vec4(c, max(c.r, max(c.g, c.b)));
        }
    }
    return sum;
}

vec4 stamGratingOrders(vec3 L, vec3 V, vec3 N, vec3 T, float periodNm, float sigma, int maxOrder) {
    vec3 B = cross(N, T);
    float u = dot(T, L + V);
    float w = dot(B, L + V);
    float lobe = exp(-(w * w) / (2.0 * sigma * sigma));
    return stamOrderSum(u, periodNm, maxOrder) * lobe;
}

uint stamHash(uvec2 v) {
    uint h = (v.x * 0x8da6b343u) ^ (v.y * 0xd8163841u);
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    return h;
}

float stamHashUnit(uint h) {
    return float(h >> 8) * (1.0 / 16777216.0);
}

float stamCellIndex(float q) {
    return q > -stamCellIndexLimit ? (q < stamCellIndexLimit ? q : stamCellIndexLimit) : -stamCellIndexLimit;
}

uvec2 stamCell(vec2 coord, float cellSize) {
    float size = cellSize > stamMinCellSize ? cellSize : stamMinCellSize;
    vec2 q = floor(coord / size);
    return uvec2(ivec2(stamCellIndex(q.x), stamCellIndex(q.y)));
}

vec3 stamTangentAxis(vec3 N, vec3 tangent) {
    vec3 x = tangent - N * dot(N, tangent);
    float len2 = dot(x, x);
    if (len2 > stamEpsilon * stamEpsilon) return x * inversesqrt(len2);
    vec3 helper = abs(N.x) < 0.9 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
    return normalize(cross(helper, N));
}

vec3 stamLinearGrooveNormal(vec3 N, vec3 tangent, float angle) {
    vec3 X = stamTangentAxis(N, tangent);
    vec3 Y = cross(N, X);
    return cos(angle) * X + sin(angle) * Y;
}

vec3 stamRadialGrooveNormal(vec3 p, vec3 centre, vec3 N) {
    return stamTangentAxis(N, p - centre);
}

vec3 stamFieldGrooveNormal(vec3 N, vec3 tangent, vec2 direction) {
    vec3 X = stamTangentAxis(N, tangent);
    vec3 Y = cross(N, X);
    float len2 = dot(direction, direction);
    if (len2 <= stamEpsilon * stamEpsilon) return X;
    vec2 d = direction * inversesqrt(len2);
    return d.x * X + d.y * Y;
}

void stamLayerGrating(StamLayerParams P, StamSurface S, out vec3 T, out float periodNm) {
    periodNm = P.periodNm;
    if (P.mode == STAM_GROOVE_RADIAL) {
        T = stamRadialGrooveNormal(S.position, S.centre, S.N);
        return;
    }
    if (P.mode == STAM_GROOVE_FIELD) {
        T = stamFieldGrooveNormal(S.N, S.tangent, S.fieldDirection);
        return;
    }
    float angle = P.grooveAngle;
    if (P.mode == STAM_GROOVE_CELLS) {
        uint h = stamHash(stamCell(S.surfaceCoord, P.grooveCellSize));
        angle += 3.141592653589793 * stamHashUnit(stamHash(uvec2(h, 3u)));
        periodNm *= 1.0 + P.periodJitter * (2.0 * stamHashUnit(stamHash(uvec2(h, 4u))) - 1.0);
    }
    T = stamLinearGrooveNormal(S.N, S.tangent, angle);
}

float stamAnnulusMask(float radius, float innerRadius, float outerRadius) {
    return (radius >= innerRadius && radius <= outerRadius) ? 1.0 : 0.0;
}

float stamTrackRipple(float radius, float baseRoughness, float amplitude, float period) {
    if (period <= 0.0) return baseRoughness;
    return baseRoughness * (1.0 + amplitude * sin(6.283185307179586 * radius / period));
}

float stamAnisotropicMirrorLobe(vec3 L, vec3 V, vec3 N, vec3 T, float roughness, float smear) {
    vec3 S = L + V;
    float u = dot(T, S);
    float w = dot(cross(N, T), S);
    float nS = dot(N, S);
    float sigma = stamSigma(roughness);
    float acrossSq = sigma * sigma + nS * nS * smear * smear;
    return exp(-0.5 * (u * u / acrossSq + w * w / (sigma * sigma)));
}

float stamIorAtLeastOne(float n) {
    return n > 1.0 ? n : 1.0;
}

float stamDielectricFresnel(float cosI, float ior) {
    float n = ior > stamEpsilon ? ior : stamEpsilon;
    float c = clamp(cosI, stamEpsilon, 1.0);
    float sinTSq = (1.0 - c * c) / (n * n);
    float cosT = sqrt(max(1.0 - sinTSq, 0.0));
    float rs = (c - n * cosT) / (c + n * cosT);
    float rp = (n * c - cosT) / (n * c + cosT);
    return 0.5 * (rs * rs + rp * rp);
}

float stamAiry(float r01, float r12, float cosDelta) {
    float cross12 = 2.0 * r01 * r12 * cosDelta;
    return (r01 * r01 + r12 * r12 + cross12) / (1.0 + r01 * r01 * r12 * r12 + cross12);
}

float stamThinFilmMaxThicknessNm(float filmIor) {
    float stepNm = (stamMaxNm - stamMinNm) / float(stamSpectralSamples);
    return stamMinNm * stamMinNm / (4.0 * stamIorAtLeastOne(filmIor) * stepNm);
}

vec3 stamThinFilmReflectance(float cosTheta, float thicknessNm, float filmIor, float baseIor) {
    float nf = stamIorAtLeastOne(filmIor);
    float nb = stamIorAtLeastOne(baseIor);
    float t = thicknessNm > 0.0 ? thicknessNm : 0.0;
    float tMax = stamThinFilmMaxThicknessNm(nf);
    t = t < tMax ? t : tMax;
    float c0 = clamp(cosTheta, stamEpsilon, 1.0);
    float sin0Sq = 1.0 - c0 * c0;
    float c1 = sqrt(max(1.0 - sin0Sq / (nf * nf), 0.0));
    float c2 = sqrt(max(1.0 - sin0Sq / (nb * nb), 0.0));
    float rs01 = (c0 - nf * c1) / (c0 + nf * c1);
    float rp01 = (nf * c0 - c1) / (nf * c0 + c1);
    float rs12 = (nf * c1 - nb * c2) / (nf * c1 + nb * c2);
    float rp12 = (nb * c1 - nf * c2) / (nb * c1 + nf * c2);
    float phase = 12.566370614359172 * nf * t * c1;
    float stepNm = (stamMaxNm - stamMinNm) / float(stamSpectralSamples);
    vec3 sum = vec3(0.0);
    vec3 norm = vec3(0.0);
    for (int k = 0; k < stamSpectralSamples; ++k) {
        float lambdaNm = stamMinNm + (float(k) + 0.5) * stepNm;
        float cosDelta = cos(phase / lambdaNm);
        float r = 0.5 * (stamAiry(rs01, rs12, cosDelta) + stamAiry(rp01, rp12, cosDelta));
        vec3 c = stamWavelengthToLinearSrgb(lambdaNm);
        sum += c * r;
        norm += c;
    }
    return sum / max(norm, vec3(stamEpsilon));
}

vec3 stamCoatReflectance(StamLayerParams P, float cosTheta) {
    if (P.thinFilmThicknessNm > 0.0)
        return stamThinFilmReflectance(cosTheta, P.thinFilmThicknessNm, P.thinFilmIor, P.coatIor);
    return vec3(stamDielectricFresnel(cosTheta, stamIorAtLeastOne(P.coatIor)));
}

float stamHriReflectance(StamLayerParams P, float cosTheta) {
    return stamDielectricFresnel(cosTheta, stamIorAtLeastOne(P.hriIor) / (P.coatIor > 1.0 ? P.coatIor : 1.0));
}

float stamGlitterFlake(vec2 surfaceCoord, float cellSize, float density, float maxTilt, vec3 N, vec3 T, out vec3 flakeNormal) {
    uint h0 = stamHash(stamCell(surfaceCoord, cellSize));
    float phi = 6.283185307179586 * stamHashUnit(stamHash(uvec2(h0, 1u)));
    float tilt = maxTilt * sqrt(stamHashUnit(stamHash(uvec2(h0, 2u))));
    vec3 B = cross(N, T);
    flakeNormal = normalize(N * cos(tilt) + (T * cos(phi) + B * sin(phi)) * sin(tilt));
    return stamHashUnit(h0) < density ? 1.0 : 0.0;
}

float stamGlitterSparkle(vec3 L, vec3 V, vec3 flakeNormal, float sharpness) {
    vec3 H = normalize(L + V);
    float s = max(sharpness, stamEpsilon);
    return exp(-(1.0 - clamp(dot(H, flakeNormal), -1.0, 1.0)) / (s * s));
}

StamLayerResult stamDiffractionLayer(StamLayerParams P, StamSurface S, vec3 L, vec3 V, vec3 baseSpecular) {
    StamLayerResult R;
    R.diffraction = vec3(0.0);
    R.coatSpecular = vec3(0.0);
    R.specularScale = vec3(1.0);
    R.diffractedWeight = 0.0;
    float periodNm;
    stamLayerGrating(P, S, R.grooveT, periodNm);
    float geometry = stamGeometryFactor(L, V, S.N);
    if (geometry <= 0.0) return R;
    vec3 H = L + V;
    float hh = dot(H, H);
    float nh = dot(S.N, H);
    float tangentialSq = max(hh - nh * nh, 0.0);
    float cosVH = 0.5 * sqrt(hh);
    vec3 transmission = vec3(1.0);
    if (P.coatIor > 1.0) {
        float coatSigma = stamSigma(P.coatRoughness);
        float coatLobe = exp(-tangentialSq / (2.0 * coatSigma * coatSigma));
        R.coatSpecular = stamCoatReflectance(P, cosVH) * (coatLobe * geometry);
        transmission = (vec3(1.0) - stamCoatReflectance(P, dot(S.N, L)))
                     * (vec3(1.0) - stamCoatReflectance(P, dot(S.N, V)));
    }
    R.specularScale = transmission;
    vec3 flakeNormal;
    if (P.glitterDensity > 0.0
        && stamGlitterFlake(S.surfaceCoord, P.glitterCellSize, P.glitterDensity, P.glitterMaxTilt, S.N, R.grooveT, flakeNormal) > 0.0) {
        R.diffraction = baseSpecular * transmission * (stamGlitterSparkle(L, V, flakeNormal, P.glitterSharpness) * geometry);
        R.specularScale = vec3(0.0);
        return R;
    }
    float sigma = stamSigma(P.roughness);
    vec4 orders = stamGratingOrders(L, V, S.N, R.grooveT, periodNm, sigma, P.maxOrder);
    if (P.mode == STAM_GROOVE_CROSSED) {
        vec3 T2 = stamLinearGrooveNormal(S.N, S.tangent, P.grooveAngle + P.crossAngle);
        orders = 0.5 * (orders + stamGratingOrders(L, V, S.N, T2, P.secondPeriodNm, sigma, P.maxOrder));
    }
    orders *= clamp(P.mask, 0.0, 1.0) * max(P.strength, 0.0) * stamFirstOrderEfficiency;
    if (orders.w > 1.0) orders /= orders.w;
    vec3 reflector = baseSpecular;
    if (P.baseMode == STAM_BASE_HRI) {
        float hri = stamHriReflectance(P, cosVH);
        float mirror = exp(-tangentialSq / (2.0 * sigma * sigma));
        reflector = vec3(hri);
        R.coatSpecular += transmission * (hri * (1.0 - orders.w) * mirror * geometry);
        R.specularScale = transmission * (1.0 - hri);
        R.diffractedWeight = hri * orders.w;
    } else {
        R.specularScale = transmission * (1.0 - orders.w);
        R.diffractedWeight = orders.w;
    }
    R.diffraction = orders.rgb * reflector * transmission * geometry;
    return R;
}
)GLSL";

inline constexpr const char* kStamDiffractionEnvironmentGlsl = R"GLSL(
vec3 stamEnvironmentRadiance(vec3 dir);

void stamEnvironmentGratingDirection(vec3 V, vec3 N, vec3 T, float periodNm, float lambdaNm, int order, out vec3 L, out bool valid) {
    vec3 B = cross(N, T);
    float across = float(order) * lambdaNm / periodNm - dot(V, T);
    float along = -dot(V, B);
    float normalSq = 1.0 - across * across - along * along;
    valid = normalSq > 0.0 && dot(N, V) > 0.0;
    L = across * T + along * B + sqrt(max(normalSq, 0.0)) * N;
}

vec4 stamEnvironmentOrders(vec3 V, vec3 N, vec3 T, float periodNm, int maxOrder, float stratumOffset) {
    float stepNm = (stamMaxNm - stamMinNm) / float(stamSpectralSamples);
    vec4 sum = vec4(0.0);
    for (int k = 0; k < stamSpectralSamples; ++k) {
        float lambdaNm = stamMinNm + (float(k) + stratumOffset) * stepNm;
        vec3 c = stamWavelengthToLinearSrgb(lambdaNm) / stamSpectrumChannelPeak;
        float peak = max(c.r, max(c.g, c.b));
        for (int n = 1; n <= maxOrder; ++n) {
            float eta = 1.0 / float(n * n);
            for (int side = -1; side <= 1; side += 2) {
                vec3 L;
                bool valid;
                stamEnvironmentGratingDirection(V, N, T, periodNm, lambdaNm, side * n, L, valid);
                if (valid) sum += eta * vec4(c * stamEnvironmentRadiance(L), peak);
            }
        }
    }
    return sum / float(stamSpectralSamples);
}

StamLayerResult stamDiffractionLayerEnvironment(StamLayerParams P, StamSurface S, vec3 V, vec3 baseSpecular, float stratumOffset) {
    StamLayerResult R;
    R.diffraction = vec3(0.0);
    R.coatSpecular = vec3(0.0);
    R.specularScale = vec3(1.0);
    R.diffractedWeight = 0.0;
    float periodNm;
    stamLayerGrating(P, S, R.grooveT, periodNm);
    float NV = dot(S.N, V);
    if (NV <= 0.0) return R;
    vec3 mirrorDir = reflect(-V, S.N);
    vec3 transmission = vec3(1.0);
    if (P.coatIor > 1.0) {
        vec3 f = stamCoatReflectance(P, NV);
        R.coatSpecular = f * stamEnvironmentRadiance(mirrorDir);
        transmission = (vec3(1.0) - f) * (vec3(1.0) - f);
    }
    R.specularScale = transmission;
    vec3 flakeNormal;
    if (P.glitterDensity > 0.0
        && stamGlitterFlake(S.surfaceCoord, P.glitterCellSize, P.glitterDensity, P.glitterMaxTilt, S.N, R.grooveT, flakeNormal) > 0.0) {
        R.diffraction = baseSpecular * transmission * stamEnvironmentRadiance(reflect(-V, flakeNormal));
        R.specularScale = vec3(0.0);
        return R;
    }
    vec4 orders = stamEnvironmentOrders(V, S.N, R.grooveT, periodNm, P.maxOrder, stratumOffset);
    if (P.mode == STAM_GROOVE_CROSSED) {
        vec3 T2 = stamLinearGrooveNormal(S.N, S.tangent, P.grooveAngle + P.crossAngle);
        orders = 0.5 * (orders + stamEnvironmentOrders(V, S.N, T2, P.secondPeriodNm, P.maxOrder, stratumOffset));
    }
    orders *= clamp(P.mask, 0.0, 1.0) * max(P.strength, 0.0) * stamFirstOrderEfficiency;
    if (orders.w > 1.0) orders /= orders.w;
    vec3 reflector = baseSpecular;
    if (P.baseMode == STAM_BASE_HRI) {
        float hri = stamHriReflectance(P, NV);
        reflector = vec3(hri);
        R.coatSpecular += transmission * (hri * (1.0 - orders.w)) * stamEnvironmentRadiance(mirrorDir);
        R.specularScale = transmission * (1.0 - hri);
        R.diffractedWeight = hri * orders.w;
    } else {
        R.specularScale = transmission * (1.0 - orders.w);
        R.diffractedWeight = orders.w;
    }
    R.diffraction = orders.rgb * reflector * transmission;
    return R;
}
)GLSL";

namespace arbit::stamdiffraction
{
// ---------------------------------------------------------------------------
// Constants shared with the GLSL. The test checks them against the literals in
// both GLSL strings.
// ---------------------------------------------------------------------------
inline constexpr int kGrooveLinear = 0;
inline constexpr int kGrooveRadial = 1;
inline constexpr int kGrooveCrossed = 2;
inline constexpr int kGrooveField = 3;
inline constexpr int kGrooveCells = 4;
inline constexpr int kBaseOpaque = 0;
inline constexpr int kBaseHri = 1;

inline constexpr double kMinVisibleNm = 380.0;
inline constexpr double kMaxVisibleNm = 780.0;
inline constexpr double kMinSigma = 0.02;
inline constexpr double kGeometryMax = 4.0;
inline constexpr double kEpsilon = 1e-6;
// Largest linear sRGB channel of wavelengthToLinearSrgb over 380 to 780 nm
// (6.95127, blue, at 447.9 nm), rounded up.
inline constexpr double kSpectrumChannelPeak = 6.9513;
inline constexpr double kFirstOrderEfficiency = 0.2;
inline constexpr double kCdTrackPitchNm = 1600.0;
inline constexpr double kDvdTrackPitchNm = 740.0;
inline constexpr double kPolycarbonateIor = 1.58;
inline constexpr double kZincSulfideIor = 2.3;
// 120 mm CD: no grating inside the clamping hub or past the data area.
inline constexpr double kDiscHubRadiusMm = 23.0;
inline constexpr double kDiscDataOuterRadiusMm = 58.0;
inline constexpr double kDiscRippleAmplitude = 0.06;
inline constexpr double kDiscRipplePeriodMm = 1.5;
inline constexpr int kSpectralSamples = 24;
// Smallest cell size for the groove-cell map and glitter, in surfaceCoord units.
inline constexpr double kMinCellSize = 1e-4;
// Cell indices clamp to +-2^30, exact in float32 and inside int32.
inline constexpr double kCellIndexLimit = 1073741824.0;

inline constexpr double kPi = 3.141592653589793;
inline constexpr double kTwoPi = 6.283185307179586;
inline constexpr double kFourPi = 12.566370614359172;

// Wyman, Sloan, Shirley 2013 multi-lobe fit. Each row is
// {amplitude, meanNm, sigmaBelowNm, sigmaAboveNm}. Rows 0 to 2 build x-bar,
// rows 3 and 4 build y-bar, rows 5 and 6 build z-bar.
inline constexpr double kCmfLobes[7][4] = {
    {1.056, 599.8, 37.9, 31.0},
    {0.362, 442.0, 16.0, 26.7},
    {-0.065, 501.1, 20.4, 26.2},
    {0.821, 568.8, 46.9, 40.5},
    {0.286, 530.9, 16.3, 31.1},
    {1.217, 437.0, 11.8, 36.0},
    {0.681, 459.0, 26.0, 13.8},
};

// XYZ to linear sRGB, D65 white, row-major.
inline constexpr double kXyzToLinearSrgb[9] = {
    3.2404542, -1.5371385, -0.4985314,
    -0.9692660, 1.8760108, 0.0415560,
    0.0556434, -0.2040259, 1.0572252,
};

// Per-channel scale so the mean of the clamped spectrum over 380 to 780 nm is
// (1, 1, 1).
inline constexpr double kWhiteBalance[3] = {2.2704, 3.4666, 3.6590};

// Integer hash multipliers (lowbias32-style finaliser).
inline constexpr std::uint32_t kHashMulX = 0x8da6b343u;
inline constexpr std::uint32_t kHashMulY = 0xd8163841u;
inline constexpr std::uint32_t kHashMulA = 0x7feb352du;
inline constexpr std::uint32_t kHashMulB = 0x846ca68bu;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
struct Vec3
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct Vec2
{
    double x = 0.0;
    double y = 0.0;
};

inline Vec3 operator+(Vec3 a, Vec3 b)
{
    return Vec3{a.x + b.x, a.y + b.y, a.z + b.z};
}

inline Vec3 operator-(Vec3 a, Vec3 b)
{
    return Vec3{a.x - b.x, a.y - b.y, a.z - b.z};
}

inline Vec3 operator*(Vec3 a, double s)
{
    return Vec3{a.x * s, a.y * s, a.z * s};
}

inline Vec3 operator*(Vec3 a, Vec3 b)
{
    return Vec3{a.x * b.x, a.y * b.y, a.z * b.z};
}

inline double dot(Vec3 a, Vec3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 cross(Vec3 a, Vec3 b)
{
    return Vec3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

inline Vec3 normalize(Vec3 a)
{
    return a * (1.0 / std::sqrt(dot(a, a)));
}

inline double maxComponent(Vec3 a)
{
    return std::max(a.x, std::max(a.y, a.z));
}

// GLSL reflect(I, N) = I - 2 dot(N, I) N.
inline Vec3 reflect(Vec3 i, Vec3 n)
{
    return i - n * (2.0 * dot(n, i));
}

// Layer parameters. Field order matches the GLSL struct StamLayerParams. Plain
// data: a shader or C++ caller may override any field per pixel, for example
// periodNm from a map or mask from stamAnnulusMask.
struct LayerParams
{
    int mode = kGrooveLinear;         // kGrooveLinear, Radial, Crossed, Field or Cells
    int baseMode = kBaseOpaque;       // kBaseOpaque (metal) or kBaseHri (clear high-index)
    double periodNm = 1000.0;         // groove period d
    double secondPeriodNm = 1000.0;   // crossed: period of the second grating
    double periodJitter = 0.0;        // cells: relative period spread per cell
    double grooveAngle = 0.0;         // T rotated from the tangent, radians
    double crossAngle = 0.0;          // crossed: second grating rotated from the first, radians
    double grooveCellSize = 0.5;      // cells: cell size in surfaceCoord units
    int maxOrder = 3;                 // orders 1..maxOrder on each side
    double roughness = 0.03;          // sigma of the along-groove aperture on w
    double zeroOrderSmear = 0.0;      // base BSDF hint: normal jitter across the grooves, radians
    double strength = 1.0;            // multiplies kFirstOrderEfficiency
    double mask = 1.0;                // 0 removes the grating (hub, lip, unprinted areas)
    double hriIor = kZincSulfideIor;  // HRI base: index of the high-index coating
    double coatIor = 0.0;             // > 1 enables the clear coat
    double coatRoughness = 0.02;      // sigma of the coat reflection lobe
    double thinFilmThicknessNm = 0.0; // > 0 makes the coat top a thin film
    double thinFilmIor = 1.38;
    double glitterDensity = 0.0;      // > 0 enables flakes: fraction of cells with a flake
    double glitterCellSize = 0.1;     // flake cell size in surfaceCoord units
    double glitterMaxTilt = 0.35;     // radians
    double glitterSharpness = 0.05;   // angular width of a flake's sparkle, radians
};

struct Surface
{
    Vec3 N{0.0, 0.0, 1.0};
    Vec3 tangent{1.0, 0.0, 0.0};   // surface or UV tangent; need not be orthogonal to N
    Vec3 position{};               // radial: shading point
    Vec3 centre{};                 // radial: groove centre
    Vec2 fieldDirection{1.0, 0.0}; // field: across-groove direction in tangent space
    Vec2 surfaceCoord{};           // cells and glitter: 2D surface position (mm or UV)
};

struct LayerResult
{
    Vec3 diffraction{};                // add: coloured orders or glitter
    Vec3 coatSpecular{};               // add: coat reflection, plus the HRI's own mirror
    Vec3 specularScale{1.0, 1.0, 1.0}; // multiply the material's own BSDF by this
    double diffractedWeight = 0.0;     // energy fraction sent into the orders
    Vec3 grooveT{};                    // resolved across-groove direction
};

// Presets, identical to the GLSL stam*Preset constants.
// CD: aluminium reflection grating, radial 1.6 um tracks, under 1.2 mm polycarbonate.
inline constexpr LayerParams kCdPreset{
    kGrooveRadial, kBaseOpaque, kCdTrackPitchNm, kCdTrackPitchNm, 0.0, 0.0, 0.0, 0.5, 3, 0.03,
    0.08, 3.0, 1.0, kZincSulfideIor, kPolycarbonateIor, 0.02, 0.0, 1.38, 0.0, 0.1, 0.35, 0.05};
// DVD: same stack with 0.74 um tracks.
inline constexpr LayerParams kDvdPreset{
    kGrooveRadial, kBaseOpaque, kDvdTrackPitchNm, kDvdTrackPitchNm, 0.0, 0.0, 0.0, 0.5, 2, 0.03,
    0.08, 3.0, 1.0, kZincSulfideIor, kPolycarbonateIor, 0.02, 0.0, 1.38, 0.0, 0.1, 0.35, 0.05};
// Metallized hologram foil: grating embossed in lacquer, vacuum-metallized with
// a few nm of aluminium (opaque), per-cell groove map.
inline constexpr LayerParams kHoloFoilMetallizedPreset{
    kGrooveCells, kBaseOpaque, 1000.0, 1000.0, 0.25, 0.0, 0.0, 0.5, 2, 0.05,
    0.0, 3.0, 1.0, kZincSulfideIor, 1.5, 0.02, 0.0, 1.38, 0.0, 0.1, 0.35, 0.05};
// Transparent HRI hologram foil: same embossing, coated with ZnS over the print.
inline constexpr LayerParams kHoloFoilHriPreset{
    kGrooveCells, kBaseHri, 1000.0, 1000.0, 0.25, 0.0, 0.0, 0.5, 2, 0.05,
    0.0, 3.0, 1.0, kZincSulfideIor, 1.5, 0.02, 0.0, 1.38, 0.0, 0.1, 0.35, 0.05};
// Trading card: HRI foil over the artwork; the caller sets mask from the foil map.
inline constexpr LayerParams kCardPreset = kHoloFoilHriPreset;
// Oil slick: thin oil film (n 1.47, 400 nm) on water (n 1.33), no grating.
inline constexpr LayerParams kOilSlickPreset{
    kGrooveLinear, kBaseOpaque, 1000.0, 1000.0, 0.0, 0.0, 0.0, 0.5, 0, 0.03,
    0.0, 0.0, 0.0, kZincSulfideIor, 1.33, 0.02, 400.0, 1.47, 0.0, 0.1, 0.35, 0.05};

// ---------------------------------------------------------------------------
// Spectrum and geometry
// ---------------------------------------------------------------------------
inline double cmfLobe(double nm, double meanNm, double sigmaBelowNm, double sigmaAboveNm)
{
    const double t = (nm - meanNm) / (nm < meanNm ? sigmaBelowNm : sigmaAboveNm);
    return std::exp(-0.5 * t * t);
}

inline Vec3 wavelengthToLinearSrgb(double nm)
{
    if (nm < kMinVisibleNm || nm > kMaxVisibleNm)
        return Vec3{};
    double xyz[3] = {0.0, 0.0, 0.0};
    for (int row = 0; row < 7; ++row)
    {
        const int channel = row < 3 ? 0 : (row < 5 ? 1 : 2);
        const double* lobe = kCmfLobes[row];
        xyz[channel] += lobe[0] * cmfLobe(nm, lobe[1], lobe[2], lobe[3]);
    }
    double rgb[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < 3; ++i)
    {
        const double linear = kXyzToLinearSrgb[3 * i] * xyz[0]
                            + kXyzToLinearSrgb[3 * i + 1] * xyz[1]
                            + kXyzToLinearSrgb[3 * i + 2] * xyz[2];
        rgb[i] = std::max(linear, 0.0) * kWhiteBalance[i];
    }
    return Vec3{rgb[0], rgb[1], rgb[2]};
}

inline double sigmaFromRoughness(double roughness)
{
    return roughness > kMinSigma ? roughness : kMinSigma; // NaN becomes kMinSigma
}

inline double geometryFactor(Vec3 L, Vec3 V, Vec3 N)
{
    const double NL = dot(N, L);
    const double NV = dot(N, V);
    if (NL <= 0.0 || NV <= 0.0)
        return 0.0;
    const Vec3 S = L + V;
    const double s2 = dot(S, S);
    const double nS = dot(N, S);
    const double a = std::max(NL, kEpsilon);
    const double b = std::max(NV, kEpsilon);
    const double brdf = std::min(s2 * s2 / (4.0 * a * b * nS * nS),
                                 std::min(s2 / (a * nS), s2 / (b * nS)));
    return NL * std::min(brdf, kGeometryMax);
}

// ---------------------------------------------------------------------------
// Order loop. One loop for every caller: the colour path passes
// wavelengthToLinearSrgb, the narrowband test probe passes a Gaussian.
// GLSL twin: stamOrderSum (colour path only).
// ---------------------------------------------------------------------------
struct OrderSum
{
    Vec3 rgb{};
    double weight = 0.0;
};

template <class Spectral>
OrderSum orderSum(double u, double periodNm, int maxOrder, Spectral spectral)
{
    OrderSum sum;
    for (int n = 1; n <= maxOrder; ++n)
    {
        const double lambdaNm = periodNm * std::fabs(u) / static_cast<double>(n);
        if (lambdaNm < kMinVisibleNm)
            break;
        if (lambdaNm <= kMaxVisibleNm)
        {
            const Vec3 c = spectral(lambdaNm) * (1.0 / kSpectrumChannelPeak);
            const double eta = 1.0 / static_cast<double>(n * n);
            sum.rgb = sum.rgb + c * eta;
            sum.weight += eta * maxComponent(c);
        }
    }
    return sum;
}

template <class Spectral>
OrderSum gratingOrders(Vec3 L, Vec3 V, Vec3 N, Vec3 T, double periodNm, double sigma, int maxOrder,
                       Spectral spectral)
{
    const Vec3 B = cross(N, T);
    const double u = dot(T, L + V);
    const double w = dot(B, L + V);
    const double lobe = std::exp(-(w * w) / (2.0 * sigma * sigma));
    OrderSum sum = orderSum(u, periodNm, maxOrder, spectral);
    sum.rgb = sum.rgb * lobe;
    sum.weight *= lobe;
    return sum;
}

// ---------------------------------------------------------------------------
// Hash
// ---------------------------------------------------------------------------
inline std::uint32_t hash(std::uint32_t x, std::uint32_t y)
{
    std::uint32_t h = (x * kHashMulX) ^ (y * kHashMulY);
    h ^= h >> 16;
    h *= kHashMulA;
    h ^= h >> 15;
    h *= kHashMulB;
    h ^= h >> 16;
    return h;
}

// Top 24 bits as a value in [0, 1). Exact in float32, so GLSL matches bit for bit.
inline double hashUnit(std::uint32_t h)
{
    return static_cast<double>(h >> 8) * (1.0 / 16777216.0);
}

struct Cell
{
    std::uint32_t x = 0;
    std::uint32_t y = 0;
};

// Cell index clamped to +-kCellIndexLimit. NaN becomes -kCellIndexLimit, so
// the int32 conversion below is always defined.
inline double cellIndex(double q)
{
    return q > -kCellIndexLimit ? (q < kCellIndexLimit ? q : kCellIndexLimit) : -kCellIndexLimit;
}

// Integer cell of coord. cellSize below kMinCellSize (or NaN) uses kMinCellSize.
// GLSL uvec2(ivec2(...)) keeps the two's-complement bit pattern, as does the
// cast here.
inline Cell cellOf(Vec2 coord, double cellSize)
{
    const double size = cellSize > kMinCellSize ? cellSize : kMinCellSize;
    const double qx = cellIndex(std::floor(coord.x / size));
    const double qy = cellIndex(std::floor(coord.y / size));
    return Cell{static_cast<std::uint32_t>(static_cast<std::int32_t>(qx)),
                static_cast<std::uint32_t>(static_cast<std::int32_t>(qy))};
}

// ---------------------------------------------------------------------------
// Groove direction helpers. Each returns T, the unit across-groove direction
// perpendicular to N.
// ---------------------------------------------------------------------------
// tangent projected into the surface plane, with a fixed perpendicular as the
// fallback when tangent is zero or parallel to N.
inline Vec3 tangentAxis(Vec3 N, Vec3 tangent)
{
    const Vec3 x = tangent - N * dot(N, tangent);
    const double len2 = dot(x, x);
    if (len2 > kEpsilon * kEpsilon)
        return x * (1.0 / std::sqrt(len2));
    const Vec3 helper = std::fabs(N.x) < 0.9 ? Vec3{1.0, 0.0, 0.0} : Vec3{0.0, 1.0, 0.0};
    return normalize(cross(helper, N));
}

// Straight grooves: the tangent rotated by angle about N.
inline Vec3 linearGrooveNormal(Vec3 N, Vec3 tangent, double angle)
{
    const Vec3 X = tangentAxis(N, tangent);
    const Vec3 Y = cross(N, X);
    return X * std::cos(angle) + Y * std::sin(angle);
}

// Circular grooves around centre (CD, DVD): T is radial. At the centre the
// grooves have no direction and the tangentAxis fallback is returned.
inline Vec3 radialGrooveNormal(Vec3 p, Vec3 centre, Vec3 N)
{
    return tangentAxis(N, p - centre);
}

// Grooves from a map: direction is the across-groove vector in tangent space
// (x along the tangent, y along cross(N, tangent)). Zero falls back to the tangent.
inline Vec3 fieldGrooveNormal(Vec3 N, Vec3 tangent, Vec2 direction)
{
    const Vec3 X = tangentAxis(N, tangent);
    const Vec3 Y = cross(N, X);
    const double len2 = direction.x * direction.x + direction.y * direction.y;
    if (len2 <= kEpsilon * kEpsilon)
        return X;
    const double inv = 1.0 / std::sqrt(len2);
    return X * (direction.x * inv) + Y * (direction.y * inv);
}

// Resolves T and the period for the layer's groove mode.
inline void layerGrating(const LayerParams& P, const Surface& S, Vec3& T, double& periodNm)
{
    periodNm = P.periodNm;
    if (P.mode == kGrooveRadial)
    {
        T = radialGrooveNormal(S.position, S.centre, S.N);
        return;
    }
    if (P.mode == kGrooveField)
    {
        T = fieldGrooveNormal(S.N, S.tangent, S.fieldDirection);
        return;
    }
    double angle = P.grooveAngle;
    if (P.mode == kGrooveCells)
    {
        const Cell cell = cellOf(S.surfaceCoord, P.grooveCellSize);
        const std::uint32_t h = hash(cell.x, cell.y);
        angle += kPi * hashUnit(hash(h, 3u));
        periodNm *= 1.0 + P.periodJitter * (2.0 * hashUnit(hash(h, 4u)) - 1.0);
    }
    T = linearGrooveNormal(S.N, S.tangent, angle);
}

// 1 between innerRadius and outerRadius inclusive, else 0 (hub and lip).
inline double annulusMask(double radius, double innerRadius, double outerRadius)
{
    return (radius >= innerRadius && radius <= outerRadius) ? 1.0 : 0.0;
}

// Faint periodic roughness ripple across the tracks, by radius.
inline double trackRipple(double radius, double baseRoughness, double amplitude, double period)
{
    if (period <= 0.0)
        return baseRoughness;
    return baseRoughness * (1.0 + amplitude * std::sin(kTwoPi * radius / period));
}

// Reference zero-order shape for the base BSDF: a mirror whose normal is
// jittered by `smear` radians across the grooves, so highlights smear across
// the tracks. Tilting the normal by a small angle a toward T moves u by about
// N.S * a, so the across-groove sigma grows by N.S * smear.
inline double anisotropicMirrorLobe(Vec3 L, Vec3 V, Vec3 N, Vec3 T, double roughness, double smear)
{
    const Vec3 S = L + V;
    const double u = dot(T, S);
    const double w = dot(cross(N, T), S);
    const double nS = dot(N, S);
    const double sigma = sigmaFromRoughness(roughness);
    const double acrossSq = sigma * sigma + nS * nS * smear * smear;
    return std::exp(-0.5 * (u * u / acrossSq + w * w / (sigma * sigma)));
}

// ---------------------------------------------------------------------------
// Coat, HRI and thin film
// ---------------------------------------------------------------------------
// Absolute index parameters (coatIor, hriIor, thinFilmIor) are clamped to at
// least 1 (NaN becomes 1), so no Fresnel or thin-film term can form 0 / 0.
inline double iorAtLeastOne(double n)
{
    return n > 1.0 ? n : 1.0;
}

// Unpolarised Fresnel reflectance into a dielectric of relative index ior.
// ior below kEpsilon (or NaN) uses kEpsilon. A relative index below 1 gives
// total internal reflection (1) past the critical angle.
inline double dielectricFresnel(double cosI, double ior)
{
    const double n = ior > kEpsilon ? ior : kEpsilon;
    const double c = std::clamp(cosI, kEpsilon, 1.0);
    const double sinTSq = (1.0 - c * c) / (n * n);
    const double cosT = std::sqrt(std::max(1.0 - sinTSq, 0.0));
    const double rs = (c - n * cosT) / (c + n * cosT);
    const double rp = (n * c - cosT) / (n * c + cosT);
    return 0.5 * (rs * rs + rp * rp);
}

// Airy reflectance of one film for real amplitude coefficients r01 and r12.
inline double airy(double r01, double r12, double cosDelta)
{
    const double cross12 = 2.0 * r01 * r12 * cosDelta;
    return (r01 * r01 + r12 * r12 + cross12) / (1.0 + r01 * r01 * r12 * r12 + cross12);
}

// Thickest film the kSpectralSamples wavelengths resolve: neighbouring samples
// stay less than half a fringe apart (phase step below pi at 380 nm and normal
// incidence), so the colour does not alias. About 1570 nm for n = 1.38 and
// 940 nm for n = 2.3. Thicker films look nearly white in reality anyway.
inline double thinFilmMaxThicknessNm(double filmIor)
{
    const double stepNm = (kMaxVisibleNm - kMinVisibleNm) / static_cast<double>(kSpectralSamples);
    return kMinVisibleNm * kMinVisibleNm / (4.0 * iorAtLeastOne(filmIor) * stepNm);
}

// Reflectance of a film (filmIor, thicknessNm) on a dielectric base (baseIor)
// seen from air, averaged over s and p and over kSpectralSamples wavelengths,
// each weighted by its colour. Both indices are clamped to at least 1 and the
// thickness to [0, thinFilmMaxThicknessNm(filmIor)] (NaN becomes 0). A constant
// reflectance R gives (R, R, R), and a zero-thickness film gives
// dielectricFresnel(cosTheta, baseIor).
inline Vec3 thinFilmReflectance(double cosTheta, double thicknessNm, double filmIor, double baseIor)
{
    const double nf = iorAtLeastOne(filmIor);
    const double nb = iorAtLeastOne(baseIor);
    double t = thicknessNm > 0.0 ? thicknessNm : 0.0;
    const double tMax = thinFilmMaxThicknessNm(nf);
    t = t < tMax ? t : tMax;
    const double c0 = std::clamp(cosTheta, kEpsilon, 1.0);
    const double sin0Sq = 1.0 - c0 * c0;
    const double c1 = std::sqrt(std::max(1.0 - sin0Sq / (nf * nf), 0.0));
    const double c2 = std::sqrt(std::max(1.0 - sin0Sq / (nb * nb), 0.0));
    const double rs01 = (c0 - nf * c1) / (c0 + nf * c1);
    const double rp01 = (nf * c0 - c1) / (nf * c0 + c1);
    const double rs12 = (nf * c1 - nb * c2) / (nf * c1 + nb * c2);
    const double rp12 = (nb * c1 - nf * c2) / (nb * c1 + nf * c2);
    const double phase = kFourPi * nf * t * c1;
    const double stepNm = (kMaxVisibleNm - kMinVisibleNm) / static_cast<double>(kSpectralSamples);
    Vec3 sum{};
    Vec3 norm{};
    for (int k = 0; k < kSpectralSamples; ++k)
    {
        const double lambdaNm = kMinVisibleNm + (static_cast<double>(k) + 0.5) * stepNm;
        const double cosDelta = std::cos(phase / lambdaNm);
        const double r = 0.5 * (airy(rs01, rs12, cosDelta) + airy(rp01, rp12, cosDelta));
        const Vec3 c = wavelengthToLinearSrgb(lambdaNm);
        sum = sum + c * r;
        norm = norm + c;
    }
    return Vec3{sum.x / std::max(norm.x, kEpsilon), sum.y / std::max(norm.y, kEpsilon),
                sum.z / std::max(norm.z, kEpsilon)};
}

inline Vec3 coatReflectance(const LayerParams& P, double cosTheta)
{
    if (P.thinFilmThicknessNm > 0.0)
        return thinFilmReflectance(cosTheta, P.thinFilmThicknessNm, P.thinFilmIor, P.coatIor);
    const double f = dielectricFresnel(cosTheta, iorAtLeastOne(P.coatIor));
    return Vec3{f, f, f};
}

// Reflectance of the HRI coating against its surround (the coat, or air).
inline double hriReflectance(const LayerParams& P, double cosTheta)
{
    return dielectricFresnel(cosTheta, iorAtLeastOne(P.hriIor) / (P.coatIor > 1.0 ? P.coatIor : 1.0));
}

// ---------------------------------------------------------------------------
// Glitter
// ---------------------------------------------------------------------------
// Returns 1 when the glitter cell holding surfaceCoord has a flake, else 0, and
// the flake normal either way.
inline double glitterFlake(Vec2 surfaceCoord, double cellSize, double density, double maxTilt,
                           Vec3 N, Vec3 T, Vec3& flakeNormal)
{
    const Cell cell = cellOf(surfaceCoord, cellSize);
    const std::uint32_t h0 = hash(cell.x, cell.y);
    const double phi = kTwoPi * hashUnit(hash(h0, 1u));
    const double tilt = maxTilt * std::sqrt(hashUnit(hash(h0, 2u)));
    const Vec3 B = cross(N, T);
    flakeNormal =
        normalize(N * std::cos(tilt) + (T * std::cos(phi) + B * std::sin(phi)) * std::sin(tilt));
    return hashUnit(h0) < density ? 1.0 : 0.0;
}

inline double glitterSparkle(Vec3 L, Vec3 V, Vec3 flakeNormal, double sharpness)
{
    const Vec3 H = normalize(L + V);
    const double s = std::max(sharpness, kEpsilon);
    return std::exp(-(1.0 - std::clamp(dot(H, flakeNormal), -1.0, 1.0)) / (s * s));
}

// ---------------------------------------------------------------------------
// Shared tail of both entry points: efficiency scale, energy clamp and base mode.
// ---------------------------------------------------------------------------
inline void finishOrders(const LayerParams& P, OrderSum orders, Vec3 transmission,
                         Vec3 baseSpecular, double hri, double radianceScale, LayerResult& R,
                         Vec3& hriMirrorWeight)
{
    const double scale =
        std::clamp(P.mask, 0.0, 1.0) * std::max(P.strength, 0.0) * kFirstOrderEfficiency;
    orders.rgb = orders.rgb * scale;
    orders.weight *= scale;
    if (orders.weight > 1.0)
    {
        orders.rgb = orders.rgb * (1.0 / orders.weight);
        orders.weight = 1.0;
    }
    Vec3 reflector = baseSpecular;
    hriMirrorWeight = Vec3{};
    if (P.baseMode == kBaseHri)
    {
        reflector = Vec3{hri, hri, hri};
        hriMirrorWeight = transmission * (hri * (1.0 - orders.weight));
        R.specularScale = transmission * (1.0 - hri);
        R.diffractedWeight = hri * orders.weight;
    }
    else
    {
        R.specularScale = transmission * (1.0 - orders.weight);
        R.diffractedWeight = orders.weight;
    }
    R.diffraction = orders.rgb * reflector * transmission * radianceScale;
}

// ---------------------------------------------------------------------------
// Layer entry points
// ---------------------------------------------------------------------------
// Point light. spectral is the per-wavelength colour: diffractionLayer passes
// wavelengthToLinearSrgb, the narrowband test probe passes a Gaussian.
template <class Spectral>
LayerResult diffractionLayerSpectral(const LayerParams& P, const Surface& S, Vec3 L, Vec3 V,
                                     Vec3 baseSpecular, Spectral spectral)
{
    LayerResult R;
    double periodNm = P.periodNm;
    layerGrating(P, S, R.grooveT, periodNm);
    const double geometry = geometryFactor(L, V, S.N);
    if (geometry <= 0.0)
        return R;
    const Vec3 H = L + V;
    const double hh = dot(H, H);
    const double nh = dot(S.N, H);
    const double tangentialSq = std::max(hh - nh * nh, 0.0);
    const double cosVH = 0.5 * std::sqrt(hh);
    Vec3 transmission{1.0, 1.0, 1.0};
    if (P.coatIor > 1.0)
    {
        const double coatSigma = sigmaFromRoughness(P.coatRoughness);
        const double coatLobe = std::exp(-tangentialSq / (2.0 * coatSigma * coatSigma));
        R.coatSpecular = coatReflectance(P, cosVH) * (coatLobe * geometry);
        transmission = (Vec3{1.0, 1.0, 1.0} - coatReflectance(P, dot(S.N, L)))
                     * (Vec3{1.0, 1.0, 1.0} - coatReflectance(P, dot(S.N, V)));
    }
    R.specularScale = transmission;
    Vec3 flakeNormal{};
    if (P.glitterDensity > 0.0
        && glitterFlake(S.surfaceCoord, P.glitterCellSize, P.glitterDensity, P.glitterMaxTilt, S.N,
                        R.grooveT, flakeNormal)
               > 0.0)
    {
        R.diffraction = baseSpecular * transmission
                      * (glitterSparkle(L, V, flakeNormal, P.glitterSharpness) * geometry);
        R.specularScale = Vec3{};
        return R;
    }
    const double sigma = sigmaFromRoughness(P.roughness);
    OrderSum orders = gratingOrders(L, V, S.N, R.grooveT, periodNm, sigma, P.maxOrder, spectral);
    if (P.mode == kGrooveCrossed)
    {
        const Vec3 T2 = linearGrooveNormal(S.N, S.tangent, P.grooveAngle + P.crossAngle);
        const OrderSum second =
            gratingOrders(L, V, S.N, T2, P.secondPeriodNm, sigma, P.maxOrder, spectral);
        orders.rgb = (orders.rgb + second.rgb) * 0.5;
        orders.weight = 0.5 * (orders.weight + second.weight);
    }
    const double hri = P.baseMode == kBaseHri ? hriReflectance(P, cosVH) : 0.0;
    Vec3 hriMirrorWeight{};
    finishOrders(P, orders, transmission, baseSpecular, hri, geometry, R, hriMirrorWeight);
    const double mirror = std::exp(-tangentialSq / (2.0 * sigma * sigma));
    R.coatSpecular = R.coatSpecular + hriMirrorWeight * (mirror * geometry);
    return R;
}

inline LayerResult diffractionLayer(const LayerParams& P, const Surface& S, Vec3 L, Vec3 V,
                                    Vec3 baseSpecular)
{
    return diffractionLayerSpectral(P, S, L, V, baseSpecular,
                                    [](double nm) { return wavelengthToLinearSrgb(nm); });
}

// Narrowband probe for tests (no GLSL twin). Runs the full point-light layer
// with every order's colour replaced by exp(-(lambda - targetNm)^2 /
// (2 bandwidthNm^2)) in each channel and a white base, and returns the red
// channel of diffraction.
inline double diffractionLayerAtWavelength(const LayerParams& P, const Surface& S, Vec3 L, Vec3 V,
                                           double targetNm, double bandwidthNm)
{
    const LayerResult R = diffractionLayerSpectral(
        P, S, L, V, Vec3{1.0, 1.0, 1.0}, [targetNm, bandwidthNm](double nm) {
            const double t = (nm - targetNm) / bandwidthNm;
            const double g = std::exp(-0.5 * t * t);
            return Vec3{g, g, g};
        });
    return R.diffraction.x;
}

// Wavelength that order 1 sends from L to V: periodNm * |dot(T, L + V)|.
inline double firstOrderWavelengthNm(Vec3 L, Vec3 V, Vec3 T, double periodNm)
{
    return periodNm * std::fabs(dot(T, L + V));
}

// ---------------------------------------------------------------------------
// Environment lighting (GLSL: kStamDiffractionEnvironmentGlsl)
// ---------------------------------------------------------------------------
// Solves the grating equation backwards: the incident direction L that order
// `order` at lambdaNm sends into V. Along the grooves the momentum is
// conserved (L.B = -V.B); across them L.T = order * lambda / d - V.T; the
// normal part fills the unit length. valid is false for an evanescent order or
// a viewer below the surface.
inline void environmentGratingDirection(Vec3 V, Vec3 N, Vec3 T, double periodNm, double lambdaNm,
                                        int order, Vec3& L, bool& valid)
{
    const Vec3 B = cross(N, T);
    const double across = static_cast<double>(order) * lambdaNm / periodNm - dot(V, T);
    const double along = -dot(V, B);
    const double normalSq = 1.0 - across * across - along * along;
    valid = normalSq > 0.0 && dot(N, V) > 0.0;
    L = T * across + B * along + N * std::sqrt(std::max(normalSq, 0.0));
}

// Sums kSpectralSamples stratified wavelengths, lambda_k = 380 + (k + offset) *
// 400 / kSpectralSamples with stratumOffset in [0, 1), over orders
// +-1..+-maxOrder. weight is the energy the orders take from the zero order:
// for a uniform environment of radiance E, rgb never exceeds weight * E in any
// channel.
template <class Radiance>
OrderSum environmentOrders(Vec3 V, Vec3 N, Vec3 T, double periodNm, int maxOrder,
                           double stratumOffset, Radiance radiance)
{
    const double stepNm = (kMaxVisibleNm - kMinVisibleNm) / static_cast<double>(kSpectralSamples);
    OrderSum sum;
    for (int k = 0; k < kSpectralSamples; ++k)
    {
        const double lambdaNm = kMinVisibleNm + (static_cast<double>(k) + stratumOffset) * stepNm;
        const Vec3 c = wavelengthToLinearSrgb(lambdaNm) * (1.0 / kSpectrumChannelPeak);
        const double peak = maxComponent(c);
        for (int n = 1; n <= maxOrder; ++n)
        {
            const double eta = 1.0 / static_cast<double>(n * n);
            for (int side = -1; side <= 1; side += 2)
            {
                Vec3 L{};
                bool valid = false;
                environmentGratingDirection(V, N, T, periodNm, lambdaNm, side * n, L, valid);
                if (valid)
                {
                    sum.rgb = sum.rgb + c * radiance(L) * eta;
                    sum.weight += eta * peak;
                }
            }
        }
    }
    const double inv = 1.0 / static_cast<double>(kSpectralSamples);
    sum.rgb = sum.rgb * inv;
    sum.weight *= inv;
    return sum;
}

// Environment-lit layer. radiance(dir) returns the environment's linear RGB
// radiance arriving from direction dir (GLSL: stamEnvironmentRadiance, defined
// by the including shader). There is no geometry factor: the result is
// radiance from radiance, energy-balanced for a uniform environment. Coat
// transmission uses the zero-order path (N.L = N.V) for every order.
template <class Radiance>
LayerResult diffractionLayerEnvironment(const LayerParams& P, const Surface& S, Vec3 V,
                                        Vec3 baseSpecular, double stratumOffset, Radiance radiance)
{
    LayerResult R;
    double periodNm = P.periodNm;
    layerGrating(P, S, R.grooveT, periodNm);
    const double NV = dot(S.N, V);
    if (NV <= 0.0)
        return R;
    const Vec3 mirrorDir = reflect(V * -1.0, S.N);
    Vec3 transmission{1.0, 1.0, 1.0};
    if (P.coatIor > 1.0)
    {
        const Vec3 f = coatReflectance(P, NV);
        R.coatSpecular = f * radiance(mirrorDir);
        transmission = (Vec3{1.0, 1.0, 1.0} - f) * (Vec3{1.0, 1.0, 1.0} - f);
    }
    R.specularScale = transmission;
    Vec3 flakeNormal{};
    if (P.glitterDensity > 0.0
        && glitterFlake(S.surfaceCoord, P.glitterCellSize, P.glitterDensity, P.glitterMaxTilt, S.N,
                        R.grooveT, flakeNormal)
               > 0.0)
    {
        R.diffraction = baseSpecular * transmission * radiance(reflect(V * -1.0, flakeNormal));
        R.specularScale = Vec3{};
        return R;
    }
    OrderSum orders =
        environmentOrders(V, S.N, R.grooveT, periodNm, P.maxOrder, stratumOffset, radiance);
    if (P.mode == kGrooveCrossed)
    {
        const Vec3 T2 = linearGrooveNormal(S.N, S.tangent, P.grooveAngle + P.crossAngle);
        const OrderSum second = environmentOrders(V, S.N, T2, P.secondPeriodNm, P.maxOrder,
                                                  stratumOffset, radiance);
        orders.rgb = (orders.rgb + second.rgb) * 0.5;
        orders.weight = 0.5 * (orders.weight + second.weight);
    }
    const double hri = P.baseMode == kBaseHri ? hriReflectance(P, NV) : 0.0;
    Vec3 hriMirrorWeight{};
    finishOrders(P, orders, transmission, baseSpecular, hri, 1.0, R, hriMirrorWeight);
    R.coatSpecular = R.coatSpecular + hriMirrorWeight * radiance(mirrorDir);
    return R;
}
} // namespace arbit::stamdiffraction
