// block_b_defs.h — Arbit Media Machine Block B audio-feature constants + DSP (pre-work A3)
//
// Pure, dependency-free C++17: no GL, no JUCE, no FFmpeg, no aubio, no I/O. This
// is the shared definition behind the uniform contract's Block B
// (generative-visuals-research.md §4.2): master-bus RMS / peak / onset and 64
// log-spaced magnitude bands (uRMS, uPeak, uOnset, uOnsetAge, uAudioBands).
//
// Why a shared header (the effect_defs.h precedent): Block B is computed in TWO
// places that MUST agree —
//   - LIVE viewport: Arbit's audio thread fills an AudioFeaturesBlock at a fixed
//     hop over a ring buffer (NOT per host block — host block sizes vary).
//   - EXPORT: the video-helper recomputes the SAME features from the mix WAV at
//     each output frame time.
// If the two paths used different window/hop/band/smoothing constants the live
// preview and the rendered file would drift apart. So every constant and every
// DSP primitive lives here once, compiled into both sides. Onsets specifically
// come from ONE in-house spectral-flux detector (this file) — aubio (GPL,
// helper-only) is reserved for beat detection and must never define uOnset, or
// live and export would trigger from two different detectors.
//
// It carries no GPL code → safe under the CI gpl-hygiene scan. Destined for
// `video-helper/src/block_b_defs.h`.
//
// Load-bearing conventions (each unit-tested in tests/block_b_tests.cpp):
//   - the analyzer emits one feature frame every kHop samples, and a frame's
//     content is a pure function of the absolute sample window
//     [k*kHop - kFftSize, k*kHop) — so feeding the same audio in ANY block
//     partition yields bit-identical frames (live == offline). See block_b_analyzer.h.
//   - the radix-2 FFT here matches a naive O(n^2) DFT to float tolerance.
//   - smoothing is a one-pole filter specified in SECONDS (audio domain, unlike
//     A1's beat-domain smoothing), so it is hop-rate consistent.
//   - silence is DEFINED: zero input → all features zero, onset never fires.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace arbitblockb
{

// ---------------------------------------------------------------------------
// Analysis constants — the single source of truth for both delivery paths.
// ---------------------------------------------------------------------------
inline constexpr int   kFftSize        = 2048;   // power of two (radix-2)
inline constexpr int   kHop            = 512;    // 4x overlap; frame cadence
inline constexpr int   kBandCount      = 64;     // uAudioBands length, R32F
inline constexpr float kBandLoHz       = 30.0f;  // lowest band center
inline constexpr float kBandHiHz       = 18000.0f; // highest band center
inline constexpr float kBandGain       = 9.0f;   // post-FFT magnitude scale (visual)
inline constexpr float kRmsSmoothMs    = 20.0f;  // uRMS one-pole time constant
inline constexpr float kPeakReleaseMs  = 80.0f;  // uPeak release (instant attack)
inline constexpr float kOnsetDecayMs   = 150.0f; // uOnset env e-folding time
inline constexpr float kOnsetMinIntervalMs = 50.0f; // onset debounce
inline constexpr float kFluxAvgMs      = 350.0f; // adaptive-threshold mean/var averaging
inline constexpr float kOnsetThreshK   = 3.5f;   // trigger above mean + K*stddev of flux
inline constexpr float kOnsetThreshFloor   = 0.012f; // absolute floor (silence guard)
inline constexpr float kOnsetAgeMax    = 99.0f;  // initial / clamped onsetAge (seconds)

// Recommended offline pre-roll (samples) the export path should decode-and-discard
// before an export region so its frame-ordered recurrences (RMS/peak/onset/flux
// mean) are warmed to the same steady state the continuous live analyzer is in.
// = one full FFT window + several flux-mean time constants. See block_b_analyzer.h
// "Cross-path agreement" for why this is required, not optional.
inline int warmupSamples (float sampleRate) noexcept
{
    const int settle = static_cast<int> (4.0f * kFluxAvgMs * 0.001f * sampleRate);
    return kFftSize + settle;
}

inline constexpr double kPi = 3.14159265358979323846;

// ---------------------------------------------------------------------------
// Small numeric helpers.
// ---------------------------------------------------------------------------
// NaN-safe: a NaN argument maps to 0 (NaN > 0.0f is false). This matters because
// clamp01 sits on the smoother input path; a NaN slipping through would poison a
// frame-ordered recurrence permanently. Inputs are also scrubbed at ingestion
// (block_b_analyzer.h pushSamples), so this is defense in depth.
inline float clamp01 (float v) noexcept
{
    return v > 0.0f ? (v > 1.0f ? 1.0f : v) : 0.0f;
}

// One-pole smoothing toward target. dt and tau both in SECONDS; BPM- and
// hop-rate-independent (alpha derived from the exponential, not a fixed coef).
// dt<=0 holds prev (paused), tau<=0 snaps to target. Mirrors A1's onePoleBeats
// guard split (the snap-on-pause bug fix) but in the audio time domain.
inline float onePoleSeconds (float prev, float target, float dtSec, float tauSec) noexcept
{
    if (tauSec <= 0.0f) return target;
    if (dtSec  <= 0.0f) return prev;
    const float alpha = 1.0f - std::exp (-dtSec / tauSec);
    return prev + (target - prev) * alpha;
}

// ---------------------------------------------------------------------------
// Hann window. Precomputed once per analyzer; exposed here so the offline path
// uses the identical taper. Symmetric (periodic=false) Hann, matching the
// workbench's analyzeWav taper (0.5 - 0.5*cos(2pi i/(N-1))).
// ---------------------------------------------------------------------------
inline std::vector<float> makeHannWindow (int n)
{
    std::vector<float> w (static_cast<size_t> (n));
    if (n <= 1) { if (n == 1) w[0] = 1.0f; return w; }
    const float denom = static_cast<float> (n - 1);
    for (int i = 0; i < n; ++i)
        w[static_cast<size_t> (i)] = 0.5f - 0.5f * std::cos (2.0f * static_cast<float> (kPi) * i / denom);
    return w;
}

// ---------------------------------------------------------------------------
// Radix-2 iterative Cooley-Tukey FFT (in-place, decimation-in-time).
// re/im are length n (power of two). Forward transform, no normalization.
// Self-contained so neither side depends on JUCE dsp::FFT or any library.
// ---------------------------------------------------------------------------
inline void fftRadix2 (std::vector<float>& re, std::vector<float>& im) noexcept
{
    const size_t n = re.size();
    if (n < 2 || (n & (n - 1)) != 0) return; // require power of two

    // Bit-reversal permutation.
    for (size_t i = 1, j = 0; i < n; ++i)
    {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { std::swap (re[i], re[j]); std::swap (im[i], im[j]); }
    }

    // Butterflies.
    for (size_t len = 2; len <= n; len <<= 1)
    {
        const float ang = -2.0f * static_cast<float> (kPi) / static_cast<float> (len);
        const float wRe = std::cos (ang);
        const float wIm = std::sin (ang);
        for (size_t i = 0; i < n; i += len)
        {
            float curRe = 1.0f, curIm = 0.0f;
            for (size_t k = 0; k < (len >> 1); ++k)
            {
                const size_t a = i + k;
                const size_t b = i + k + (len >> 1);
                const float tRe = re[b] * curRe - im[b] * curIm;
                const float tIm = re[b] * curIm + im[b] * curRe;
                re[b] = re[a] - tRe; im[b] = im[a] - tIm;
                re[a] += tRe;        im[a] += tIm;
                const float nextRe = curRe * wRe - curIm * wIm;
                curIm = curRe * wIm + curIm * wRe;
                curRe = nextRe;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Log-spaced band edges. Returns kBandCount+1 frequency edges (Hz) spanning
// [kBandLoHz, kBandHiHz] geometrically. Band b owns [edges[b], edges[b+1]).
// Center of band b (for documentation / shader band->freq mapping) is the
// geometric mean of its edges.
// ---------------------------------------------------------------------------
inline std::array<float, kBandCount + 1> bandEdgesHz () noexcept
{
    std::array<float, kBandCount + 1> edges{};
    const float ratio = std::log (kBandHiHz / kBandLoHz);
    for (int i = 0; i <= kBandCount; ++i)
        edges[static_cast<size_t> (i)] =
            kBandLoHz * std::exp (ratio * static_cast<float> (i) / static_cast<float> (kBandCount));
    return edges;
}

// ---------------------------------------------------------------------------
// Integrate a magnitude spectrum (mag[0..fftSize/2]) into kBandCount bands.
// Each band averages the FFT-bin magnitudes whose center frequency falls in
// [edges[b], edges[b+1]). A band narrower than one FFT bin (the bottom octave at
// kFftSize=2048 / 44.1 kHz — bins are ~21.5 Hz apart) captures no bins; rather
// than snapping every such band to the same distant "nearest" bin (which would
// pin a whole block of low bands to one identical value and put the per-tone
// argmax in the wrong band), it LINEARLY INTERPOLATES the spectrum at the band's
// geometric-center frequency. That gives each sub-bin band a distinct, monotone,
// in-grid value — no dead bands, no flat plateau. Output scaled by kBandGain and
// clamped to 0..1.
// ---------------------------------------------------------------------------
inline void integrateBands (const std::vector<float>& mag, float sampleRate,
                            std::array<float, kBandCount>& out) noexcept
{
    const auto edges = bandEdgesHz();
    const int nyqBin = static_cast<int> (mag.size()) - 1; // mag has fftSize/2+1 entries
    const float binHz = sampleRate / static_cast<float> (kFftSize);

    for (int b = 0; b < kBandCount; ++b)
    {
        const float fLo = edges[static_cast<size_t> (b)];
        const float fHi = edges[static_cast<size_t> (b) + 1];
        int loBin = static_cast<int> (std::ceil  (fLo / binHz));
        int hiBin = static_cast<int> (std::floor (fHi / binHz));
        if (loBin < 1) loBin = 1;                 // skip DC bin
        if (hiBin > nyqBin) hiBin = nyqBin;

        float sum = 0.0f;
        int count = 0;
        for (int k = loBin; k <= hiBin; ++k) { sum += mag[static_cast<size_t> (k)]; ++count; }

        float bandMag;
        if (count > 0)
        {
            bandMag = sum / static_cast<float> (count);
        }
        else
        {
            // No FFT bin inside this band — interpolate the spectrum at the band's
            // geometric-center frequency so adjacent narrow bands get distinct values.
            const float center = std::sqrt (fLo * fHi);
            float pos = center / binHz;            // fractional bin position
            if (pos < 1.0f) pos = 1.0f;            // never read DC bin 0
            int lo = static_cast<int> (std::floor (pos));
            if (lo > nyqBin - 1) lo = nyqBin - 1;
            if (lo < 1) lo = 1;
            const float frac = pos - static_cast<float> (lo);
            bandMag = mag[static_cast<size_t> (lo)] * (1.0f - frac)
                    + mag[static_cast<size_t> (lo + 1)] * frac;
        }

        out[static_cast<size_t> (b)] = clamp01 (bandMag / static_cast<float> (kFftSize) * kBandGain);
    }
}

} // namespace arbitblockb
