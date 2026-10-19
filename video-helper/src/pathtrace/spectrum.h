#pragma once

// Transport channels of the CPU path tracer (Plan 3, PT-D).
//
// Every path carries four channels. In RGB mode they are linear sRGB red,
// green and blue, and the fourth is unused (always 0). In spectral mode they
// are radiance at four wavelengths: a hero wavelength drawn uniformly in
// [380, 780) nm and three companions rotated by 100, 200 and 300 nm, wrapped
// into the same interval (Wilkie et al., "Hero Wavelength Spectral Sampling",
// EGSR 2014).
//
// RGB to spectrum. A linear RGB triple becomes
//     S(lambda) = r * Br(lambda) + g * Bg(lambda) + b * Bb(lambda)
// with three smooth basis spectra tabulated every 5 nm. They were fitted
// offline as a softmax of cubic B-splines, so each is non-negative and the
// three sum to exactly 1 at every wavelength. Consequences:
// - white (1, 1, 1) is the flat spectrum 1, so a furnace stays exact;
// - an RGB reflectance in [0, 1] becomes a spectrum in [0, 1], so no surface
//   reflects more than it receives;
// - it is linear, so emission of any brightness and spatially varying
//   radiance (environment images, skies) upsample with three multiply-adds.
// At start-up the tables are corrected by the inverse of their 3 x 3 colour
// matrix, so converting a basis spectrum back gives its primary and an RGB
// value round-trips up to float rounding, as long as neither non-negativity
// clamp acts: the corrected table entries are clamped at 0, and so is an
// upsampled spectrum (an RGB value with a negative component). Where a clamp
// acts, the round trip is only approximate. The fit was chosen over
// Jakob and Hanika's sigmoid polynomials ("A Low-Dimensional Function Space
// for Efficient Spectral Upsampling", 2019) because those need a 3D table or
// a per-colour solve, and cannot reach exactly 0 or 1.
//
// Spectrum to RGB. Channel values convert through the CIE 1931 2-degree colour
// matching functions and the CIE D65 illuminant (5 nm tables, linearly
// interpolated) and the IEC 61966-2-1 XYZ to linear sRGB matrix. Spectra are
// taken relative to D65: the per-wavelength weights are
// M * xyz(lambda) * D65(lambda), each channel normalized to integrate to 1
// over [380, 780] nm. That is the D65 white balance: a flat spectrum maps to
// exactly (1, 1, 1), so a non-dispersive white scene renders the same as in
// RGB mode.
//
// Behavioural reference (studied, not copied): Blender Cycles
// intern/cycles/kernel/closure/bsdf_microfacet.h (bsdf_glass_ior, Cauchy IOR
// from an Abbe number), intern/cycles/kernel/integrator/path_state.h
// (dispersion_throughput_weight) and intern/cycles/kernel/tables.h (5 nm CIE
// and D65 table layout). The tables here were generated from the CIE data
// distributed with colour-science, not taken from Cycles.

#include "vec_math.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace videohelper::pathtrace
{

constexpr int kSpectrumChannels = 4;
constexpr float kWavelengthMinNm = 380.0f;
constexpr float kWavelengthMaxNm = 780.0f;
constexpr float kWavelengthSpanNm = kWavelengthMaxNm - kWavelengthMinNm;
// Sodium d line, the wavelength the material IOR is given at
// (surfacematerial::kGlassReferenceWavelength).
constexpr float kDLineNm = 587.56f;
// Hydrogen F and C lines, used to turn an Abbe number into a Cauchy B.
constexpr float kFLineNm = 486.13f;
constexpr float kCLineNm = 656.27f;

struct Spectrum
{
    std::array<float, kSpectrumChannels> v {};

    static Spectrum constant (float x) noexcept
    {
        Spectrum s;
        s.v.fill (x);
        return s;
    }

    float operator[] (int i) const noexcept { return v[static_cast<std::size_t> (i)]; }
    float& operator[] (int i) noexcept { return v[static_cast<std::size_t> (i)]; }

    float maxValue() const noexcept
    {
        float m = v[0];
        for (int i = 1; i < kSpectrumChannels; ++i)
            m = m > v[static_cast<std::size_t> (i)] ? m : v[static_cast<std::size_t> (i)];
        return m;
    }
    float average() const noexcept
    {
        float sum = 0.0f;
        for (const float x : v)
            sum += x;
        return sum * (1.0f / static_cast<float> (kSpectrumChannels));
    }
    bool finite() const noexcept
    {
        for (const float x : v)
            if (! std::isfinite (x))
                return false;
        return true;
    }
    bool anyPositive() const noexcept
    {
        for (const float x : v)
            if (x > 0.0f)
                return true;
        return false;
    }
};

inline Spectrum operator+ (const Spectrum& a, const Spectrum& b) noexcept
{
    Spectrum r;
    for (int i = 0; i < kSpectrumChannels; ++i)
        r[i] = a[i] + b[i];
    return r;
}
inline Spectrum operator* (const Spectrum& a, const Spectrum& b) noexcept
{
    Spectrum r;
    for (int i = 0; i < kSpectrumChannels; ++i)
        r[i] = a[i] * b[i];
    return r;
}
inline Spectrum operator* (const Spectrum& a, float s) noexcept
{
    Spectrum r;
    for (int i = 0; i < kSpectrumChannels; ++i)
        r[i] = a[i] * s;
    return r;
}
inline Spectrum operator* (float s, const Spectrum& a) noexcept { return a * s; }

// What the four channels of a path mean.
struct Wavelengths
{
    // Spectral mode: nm[0] is the hero. RGB mode: unused, all at the d line.
    std::array<float, kSpectrumChannels> nm { { kDLineNm, kDLineNm, kDLineNm, kDLineNm } };
    bool spectral = false;
};

Wavelengths rgbChannels() noexcept;
// Hero wavelength 380 + 400 u nm and its three rotated companions.
Wavelengths sampleHeroWavelengths (float u) noexcept;

// RGB mode: (r, g, b, 0). Spectral mode: the basis spectrum at each channel's
// wavelength, never negative.
Spectrum upsampleRgb (Vec3 rgb, const Wavelengths& wavelengths) noexcept;
// Linear sRGB of one path estimate. RGB mode: the first three channels.
// Spectral mode: sum over channels of value * weight(lambda) * 400 / 4, the
// Monte Carlo estimate of the colour integral with uniform wavelengths.
Vec3 channelsToRgb (const Spectrum& value, const Wavelengths& wavelengths) noexcept;

// Basis spectra (Br, Bg, Bb) at one wavelength, after the exact correction.
Vec3 rgbBasisAt (float nm) noexcept;
// Linear sRGB weight per nm at one wavelength. Each channel integrates to 1
// over [380, 780] nm.
Vec3 rgbMatchingWeightAt (float nm) noexcept;

// Cauchy index n(lambda) = A + B / lambda^2, lambda in micrometres, with A
// chosen so that n(d line) = iorD (SurfaceMaterialIR.h wavelengthIor).
float cauchyIor (float iorD, float cauchyB, float nm) noexcept;
// The Cauchy B of an Abbe number V_d = (n_d - 1) / (n_F - n_C).
float cauchyBFromAbbe (float iorD, float abbe) noexcept;

} // namespace videohelper::pathtrace
