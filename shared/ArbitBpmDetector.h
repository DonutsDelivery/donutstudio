// ArbitBpmDetector — native BPM detection for Arbit sample loops.
//
// Implements a two-stage pipeline:
//   1. Spectral-flux onset detection function (ODF) from a log-magnitude STFT
//   2. Tempo estimation via biased autocorrelation of the ODF, picking the
//      lag whose period best fits the loop as an integer beat count
//
// Algorithm references (papers, not code — this implementation is original):
//   - Bello et al. 2005, "A Tutorial on Onset Detection in Music Signals"
//   - Klapuri et al. 2006, "Analysis of the meter of acoustic musical signals"
//   - Davies & Plumbley 2007, "Context-dependent beat tracking"
//
// Design choices specific to Arbit:
//   - We're detecting tempo for short loops (≤ 60s), not long tracks, so we
//     aggressively snap to integer beat counts using loop duration as a prior.
//   - Optional project-BPM prior biases disambiguation toward the user's
//     current tempo when candidates are close.
//   - Pure header-only so it can be dropped into the offline test harness and
//     the plugin without any build-system wiring.
//
// No external dependencies beyond the C++17 stdlib and a caller-provided FFT.

#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <functional>
#include <numeric>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace arbit::bpm {

// Caller-provided FFT. We avoid baking in a specific library so the same code
// runs under the offline test and inside the plugin. Must compute an in-place
// complex FFT on `size` points (size must be a power of 2).
using FftFn = std::function<void(std::vector<std::complex<float>>& inOut)>;

// Built-in radix-2 Cooley-Tukey FFT — iterative, in-place, no dependencies.
// Good enough for BPM detection (sizes 2048..8192); not as fast as a library
// FFT but avoids a dependency when the plugin doesn't already link one.
inline void simpleFft(std::vector<std::complex<float>>& a) {
    const size_t n = a.size();
    if (n < 2) return;
    // Bit-reversal permutation
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const float ang = -2.0f * float(M_PI) / float(len);
        const std::complex<float> wlen(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < n; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            for (size_t k = 0; k < len / 2; ++k) {
                const auto u = a[i + k];
                const auto v = a[i + k + len / 2] * w;
                a[i + k]           = u + v;
                a[i + k + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

struct DetectorOptions {
    int fftSize = 2048;              // STFT window length in samples
    int hopSize = 512;
    float minBpm = 60.0f;
    float maxBpm = 200.0f;
    // Log-Gaussian tempo prior centered at this BPM. Strongly biases the ACF
    // toward musically plausible tempos and fixes the octave-error problem
    // at high tempos. Set tempoPriorSigmaOctaves = 0 to disable.
    float tempoPriorCenterBpm = 120.0f;
    float tempoPriorSigmaOctaves = 0.7f;
    // Project BPM hint: if > 0, adds a tighter second prior peak at the user's
    // current project tempo for disambiguation. Independent of the loose
    // general-music prior above.
    float projectBpmHint = 0.0f;
    float projectBpmSigmaOctaves = 0.12f;  // narrow — trust the hint strongly
    // Small bonus subtracted from the score penalty when the snapped beat
    // count is a power of two (1, 2, 4, 8, 16, 32). Loops are overwhelmingly
    // 1/2/4/8 bars long, so this is a strong tiebreaker without being a
    // hard constraint — loops that aren't pow-2 still win when their ACF
    // evidence is strong enough.
    float powerOfTwoBeatsBonus = 0.20f;
    // Max number of top ACF peaks to rank before final scoring
    int maxCandidatePeaks = 12;
    // Snap tolerance — measured as how far `beats_in_loop` is from the
    // nearest integer, as a fraction of a beat. Snap is only applied when
    // the ACF says we're already very close to an integer beat count; this
    // avoids snapping non-bar-aligned loops (e.g. a 3.9s 120 BPM file where
    // 7.8 beats would wrongly snap to 8 = 123 BPM). 0.35 was the empirical
    // sweet spot from grid search on the bench corpus.
    float snapMaxIntErrFraction = 0.45f;

    // Disable all loop-duration priors for fair head-to-head comparison
    // against general-purpose tempo trackers (SoundTouch, aubio, QM-DSP)
    // that don't know they're looking at a loop. When true: no integer-beat
    // snap, no intErr penalty in the scoring function, no power-of-two
    // bonus. The detector becomes purely "best ACF peak under tempo prior".
    bool useLoopDurationPrior = true;

    // Weight on the integer-beat error penalty (how much beats-in-loop is
    // away from an integer, 0..0.5). Was hardcoded to 1.0 in earlier phases,
    // which made it dominate the ACF and pulse-train scores (which top out
    // at ~1.0 each). Lowering this lets stronger ACF/pulse evidence survive
    // even when integer beat fit isn't perfect.
    float intErrPenaltyWeight = 0.6f;

    // Harmonic enhancement weights at lag*2 and lag*4 (Percival/Tzanetakis
    // 2014). Paper specifies (1, 1) but empirically (0.5, 0.25) gives better
    // score balance against ACF strength. Tunable for further A/B.
    float harmEnhWeight2 = 0.5f;
    float harmEnhWeight4 = 0.25f;

    // --- Phase D: Percival/Tzanetakis 2014 + Klapuri 2006 pipeline pieces ---
    //
    // Generalized autocorrelation (Tolonen/Karjalainen 2000): IFFT(|X|^c).
    // Percival recommends c<1 for compression (suppressing dominant peaks),
    // but on this percussive bench corpus c slightly less than 2 (magnitude
    // *expansion*, sharpening dominant peaks against noise) wins by 2 points
    // held-out. Probably because biased ACF's natural shape works better
    // with the integer-snap pipeline than compression does.
    bool useGeneralizedAcf = true;
    float acfCompression = 1.8f;

    // Multi-band ODF (Klapuri/Eronen 2006 accent signals): split STFT bins
    // into N log-spaced bands, compute spectral flux per band, normalize each
    // band by its own max so quiet hi-hat-band onsets aren't drowned by loud
    // kick-band onsets, then sum into a single ODF. 4 = Klapuri's 4-band split.
    int numOdfBands = 4;

    // SuperFlux-style maximum filter on the previous frame before differencing
    // (Böck/Widmer DAFx-13). Disabled by default (mu=0). Tested on the bench
    // corpus (mostly drum/percussive loops) at mu=1..5, all costing 1-2 points
    // vs plain spectral flux. The vibrato-suppression benefit doesn't apply on
    // corpora dominated by drum hits — the max filter only blunts legitimate
    // onsets. May still help if our future test set leans more melodic.
    int superFluxMu = 0;

    // Percival/Tzanetakis 2014 harmonic enhancement: add time-stretched copies
    // of the ACF at 2× and 4× to reinforce the true period and suppress
    // subharmonic peaks at 2T, 3T, 4T that ACF naturally favors.
    bool useHarmonicEnhancement = true;

    // Pulse-train cross-correlation evaluation (Percival/Tzanetakis 2014).
    // For each candidate lag, build a synthetic impulse train at that period,
    // sweep phase, and use sum-of-ODF-energy at the best phase (normalized
    // by total ODF energy) as evidence the candidate aligns with onsets.
    // Critical detail: SUM not average — at lag L the train hits all N/L
    // kicks (full energy), at lag 2L it hits N/(2L) kicks (half energy).
    // Average would erase this difference; sum/total preserves it and acts
    // as an octave disambiguator. Tolerance window of ±1 hop on each pulse
    // accommodates parabolic-interpolated lag candidates landing slightly
    // off-grid.
    bool usePulseTrainScoring = true;
    float pulseTrainWeight = 0.5f;

    // Onset-salience masking for long songs (Phase K). When enabled, compute
    // a smoothed ODF envelope (~1s moving average), then zero out hops where
    // the envelope drops below `rhythmicMaskThreshold * max(envelope)`. This
    // suppresses ambient intros/outros/breakdowns so they stop diluting the
    // ACF. Off by default; the SampleAssetPool wrapper enables it for
    // samples longer than 30 seconds.
    bool maskNonRhythmicRegions = false;
    float rhythmicMaskThreshold = 0.25f;   // fraction of envelope max
    float rhythmicMaskWindowSec = 1.0f;    // moving-average window

    // --- Phase H: Learned re-ranker (logistic regression on per-candidate
    // features). Trained on ~4400 labeled loops via tools/bpm-bench/train_reranker.py.
    // Replaces the hand-tuned linear scoring formula with one whose weights
    // were fit to the bench corpus. The features are computed inside the
    // scoring loop and combined with constexpr weights from learned_weights.h.
    bool useLearnedReRanker = true;
};

// Per-candidate features used by the scoring loop. Captured for every
// (peak-lag, multiplier) combination so the bench can dump them as training
// data for the learned re-ranker.
struct Candidate {
    float candBpm = 0.0f;
    int rounded = 0;          // beats-in-loop, rounded
    float here = 0.0f;        // prior-weighted ACF strength at the peak
    float pulseEvidence = 0.0f;
    float intErr = 0.0f;
    float multiplier = 1.0f;
    float logMPenalty = 0.0f; // |log2(m)|
    int isPow2 = 0;
    float distToPriorOctaves = 0.0f;  // log2(candBpm / 120)
    int lag = 0;
    float beatsInLoop = 0.0f;
    float score = 0.0f;       // hand-tuned scoring formula's value
};

struct DetectionResult {
    float bpm = 0.0f;                // 0 = detection failed
    float confidence = 0.0f;         // 0..1, autocorrelation peak strength
    int beatsInLoop = 0;             // snapped beat count (0 if no snap applied)
    std::vector<float> odf;          // onset detection function (debug)
    std::vector<float> acf;          // autocorrelation curve (debug)
    std::vector<Candidate> candidates;  // all scored candidates (for training)
};

namespace detail {

// Hann window, precomputed by caller and reused per STFT frame.
inline std::vector<float> makeHannWindow(int size) {
    std::vector<float> w(static_cast<size_t>(size));
    for (int n = 0; n < size; ++n) {
        w[static_cast<size_t>(n)] =
            0.5f * (1.0f - std::cos(2.0f * float(M_PI) * n / float(size - 1)));
    }
    return w;
}

// Downmix any channel count to mono, simple average.
inline std::vector<float> downmixToMono(const float* interleaved,
                                        size_t frames, int channels) {
    std::vector<float> mono(frames);
    if (channels == 1) {
        std::copy(interleaved, interleaved + frames, mono.begin());
        return mono;
    }
    const float inv = 1.0f / float(channels);
    for (size_t i = 0; i < frames; ++i) {
        float sum = 0.0f;
        for (int c = 0; c < channels; ++c) {
            sum += interleaved[i * size_t(channels) + size_t(c)];
        }
        mono[i] = sum * inv;
    }
    return mono;
}

// Compute the spectral-flux onset detection function.
//
// For each STFT frame, we compare the current log-magnitude spectrum to the
// previous one, half-wave-rectify the differences (only increases count as
// onsets), and sum across bins. The result is one scalar per hop, representing
// "how much spectral energy appeared in this hop relative to the last".
//
// Why log magnitude: onsets in bass/mellow content show up as small absolute
// changes on top of large existing energy. Taking the log compresses the
// dynamic range so those relative changes become visible. This is the main
// reason this ODF outperforms SoundTouch's time-domain envelope tracker on
// sparse/melodic material.
inline std::vector<float> spectralFluxOdf(const std::vector<float>& mono,
                                          int fftSize, int hopSize,
                                          const FftFn& fft,
                                          const std::vector<float>& window,
                                          int superFluxMu = 0,
                                          int numBands = 1,
                                          int sampleRate = 44100) {
    const size_t N = mono.size();
    if (N < size_t(fftSize)) return {};

    const int numBins = fftSize / 2 + 1;
    const int numFrames = int((N - size_t(fftSize)) / size_t(hopSize)) + 1;
    if (numFrames <= 1) return {};

    std::vector<float> odf(size_t(numFrames), 0.0f);
    std::vector<float> prevLogMag(size_t(numBins), 0.0f);
    std::vector<std::complex<float>> frame;
    frame.resize(size_t(fftSize));

    // NOTE: Audit recommended Hz-based band edges (Klapuri 250/500/1000/2000),
    // but on this corpus the original log-spaced bin-index edges performed
    // marginally better. Keeping the bin-space approach. (void)sampleRate to
    // silence the unused-parameter warning.
    (void)sampleRate;
    const int numB = std::max(1, numBands);
    std::vector<int> bandEdges(static_cast<size_t>(numB + 1), 0);
    bandEdges[0] = 1;  // skip DC
    bandEdges[static_cast<size_t>(numB)] = numBins;
    if (numB > 1) {
        const float logMin = std::log(1.0f);
        const float logMax = std::log(float(numBins - 1));
        for (int b = 1; b < numB; ++b) {
            const float t = float(b) / float(numB);
            const int edge = int(std::round(std::exp(logMin + t * (logMax - logMin))));
            bandEdges[static_cast<size_t>(b)] = std::max(bandEdges[static_cast<size_t>(b - 1)] + 1, edge);
        }
    }
    // Per-band flux storage when multi-band is active
    std::vector<std::vector<float>> bandFlux;
    if (numB > 1) bandFlux.assign(static_cast<size_t>(numB),
                                  std::vector<float>(static_cast<size_t>(numFrames), 0.0f));

    // Small epsilon so log(0) doesn't explode. Tuned to roughly -100 dB floor.
    constexpr float kLogFloor = 1e-5f;

    for (int f = 0; f < numFrames; ++f) {
        const size_t start = size_t(f * hopSize);
        for (int n = 0; n < fftSize; ++n) {
            frame[size_t(n)] =
                std::complex<float>(mono[start + size_t(n)] * window[size_t(n)], 0.0f);
        }
        fft(frame);

        // Stash current log-mag before computing flux (so prevLogMag isn't
        // overwritten until after we read it through the max filter).
        std::vector<float> curLogMag(static_cast<size_t>(numBins), 0.0f);
        for (int k = 0; k < numBins; ++k) {
            const float mag = std::abs(frame[size_t(k)]);
            curLogMag[size_t(k)] = std::log(mag + kLogFloor);
        }

        if (numB > 1) {
            // Per-band flux accumulation
            for (int b = 0; b < numB; ++b) {
                const int kLo = bandEdges[size_t(b)];
                const int kHi = bandEdges[size_t(b + 1)];
                float flux = 0.0f;
                for (int k = kLo; k < kHi; ++k) {
                    float prevRef;
                    if (superFluxMu > 0) {
                        prevRef = -1e30f;
                        const int jLo = std::max(0, k - superFluxMu);
                        const int jHi = std::min(numBins - 1, k + superFluxMu);
                        for (int j = jLo; j <= jHi; ++j)
                            if (prevLogMag[size_t(j)] > prevRef) prevRef = prevLogMag[size_t(j)];
                    } else {
                        prevRef = prevLogMag[size_t(k)];
                    }
                    const float diff = curLogMag[size_t(k)] - prevRef;
                    if (diff > 0.0f) flux += diff;
                }
                bandFlux[size_t(b)][size_t(f)] = flux;
            }
        } else {
            float flux = 0.0f;
            // Skip DC and Nyquist bins for consistency with the multi-band
            // path (which starts at bandEdges[0]=1).
            if (superFluxMu > 0) {
                for (int k = 1; k < numBins - 1; ++k) {
                    float prevMax = -1e30f;
                    const int kLo = std::max(1, k - superFluxMu);
                    const int kHi = std::min(numBins - 2, k + superFluxMu);
                    for (int j = kLo; j <= kHi; ++j) {
                        if (prevLogMag[size_t(j)] > prevMax) prevMax = prevLogMag[size_t(j)];
                    }
                    const float diff = curLogMag[size_t(k)] - prevMax;
                    if (diff > 0.0f) flux += diff;
                }
            } else {
                for (int k = 1; k < numBins - 1; ++k) {
                    const float diff = curLogMag[size_t(k)] - prevLogMag[size_t(k)];
                    if (diff > 0.0f) flux += diff;
                }
            }
            odf[size_t(f)] = flux;
        }
        prevLogMag = curLogMag;
    }

    // Multi-band: normalize each band's flux by its own max, then sum into
    // the final ODF. This gives quiet hi-hat-band onsets equal weight to
    // loud kick-band onsets — the Klapuri 2006 idea.
    if (numB > 1) {
        // First-frame per-band flux is meaningless (no predecessor) — zero it.
        for (int b = 0; b < numB; ++b) bandFlux[size_t(b)][0] = 0.0f;
        for (int b = 0; b < numB; ++b) {
            float bMax = 0.0f;
            for (float v : bandFlux[size_t(b)]) if (v > bMax) bMax = v;
            const float inv = (bMax > 0.0f) ? 1.0f / bMax : 0.0f;
            for (int f = 0; f < numFrames; ++f) {
                odf[size_t(f)] += bandFlux[size_t(b)][size_t(f)] * inv;
            }
        }
    }

    // First frame has no predecessor — its "flux" is meaningless. Zero it.
    if (!odf.empty()) odf[0] = 0.0f;

    // NOTE: Bello 2005 recommends a local median DC subtraction, which the
    // back-end audit echoed. Tried at W=0.15s/0.20s/0.30s with both true
    // median and 25th percentile — all variants regressed by 1-2 points.
    // The local floor subtraction removes onset structure that the ACF
    // depends on (ODF "humps" decay over multiple hops, and a local floor
    // eats their tails). Global mean is empirically better on this corpus.
    const float mean =
        std::accumulate(odf.begin(), odf.end(), 0.0f) / float(odf.size());
    for (auto& v : odf) v = std::max(0.0f, v - mean);

    return odf;
}

// Convert ODF lag (hops) to BPM, given hop size and sample rate.
inline float lagToBpm(int lag, int hopSize, int sampleRate) {
    if (lag <= 0) return 0.0f;
    const float secondsPerHop = float(hopSize) / float(sampleRate);
    const float secondsPerBeat = float(lag) * secondsPerHop;
    return 60.0f / secondsPerBeat;
}

// Convert a fractional lag to BPM (used after parabolic interpolation gives
// a sub-bin estimate of the true peak position).
inline float fractionalLagToBpm(float lag, int hopSize, int sampleRate) {
    if (lag <= 0.0f) return 0.0f;
    const float secondsPerHop = float(hopSize) / float(sampleRate);
    return 60.0f / (lag * secondsPerHop);
}

// Parabolic interpolation around a discrete peak. Given three consecutive
// ACF values centered on a local maximum, fits a parabola and returns the
// fractional offset of the true maximum from the center sample. Result is
// in [-0.5, +0.5]. Standard DSP trick — error from quantization drops from
// O(1) to O((1/N)^2) at almost no cost.
inline float parabolicPeakOffset(float yL, float yC, float yR) {
    const float denom = (yL - 2.0f * yC + yR);
    if (std::abs(denom) < 1e-12f) return 0.0f;
    const float offset = 0.5f * (yL - yR) / denom;
    // True parabolic max around three samples cannot lie outside ±0.5; any
    // larger value indicates the center isn't actually a local max or the
    // parabola is degenerate. Smith, "Spectral Audio Signal Processing".
    if (offset < -0.5f || offset > 0.5f) return 0.0f;
    return offset;
}

inline int bpmToLag(float bpm, int hopSize, int sampleRate) {
    if (bpm <= 0.0f) return 0;
    const float secondsPerBeat = 60.0f / bpm;
    return int(std::round(secondsPerBeat * float(sampleRate) / float(hopSize)));
}

// Biased autocorrelation. `acf[lag] = sum_n odf[n]*odf[n+lag]`, normalized,
// for lags in [minLag, maxLag]. Pure correlation — tempo priors are applied
// later in BPM space so the ACF itself stays interpretable for debugging.
inline std::vector<float> autocorrelateOdf(const std::vector<float>& odf,
                                           int minLag, int maxLag) {
    const int N = int(odf.size());
    std::vector<float> acf(size_t(maxLag + 1), 0.0f);
    if (N < maxLag + 2) return acf;

    double energy = 0.0;
    for (int n = 0; n < N; ++n) energy += double(odf[size_t(n)]) * double(odf[size_t(n)]);
    if (energy <= 0.0) return acf;
    const float invEnergy = float(1.0 / energy);

    for (int lag = minLag; lag <= maxLag; ++lag) {
        double sum = 0.0;
        const int limit = N - lag;
        for (int n = 0; n < limit; ++n) {
            sum += double(odf[size_t(n)]) * double(odf[size_t(n + lag)]);
        }
        acf[size_t(lag)] = float(sum) * invEnergy;
    }
    return acf;
}

// Onset-salience masking (Phase K). Smooths the ODF with a moving-average
// window, then zeros out hops where the smoothed envelope drops below
// `threshold * max(envelope)`. Intended for long songs where ambient intros,
// breakdowns, and outros dilute the ACF with non-rhythmic regions.
//
// Returns the fraction of hops that survived (for diagnostics). Modifies
// `odf` in place.
inline float maskByRhythmicSalience(std::vector<float>& odf,
                                    int hopSize, int sampleRate,
                                    float windowSec, float threshold) {
    const int N = int(odf.size());
    if (N < 8 || threshold <= 0.0f) return 1.0f;
    const int W = std::max(3, int(windowSec * double(sampleRate) / double(hopSize)));
    if (W >= N) return 1.0f;

    // Smoothed envelope: symmetric moving average over ±W/2 hops
    const int half = W / 2;
    std::vector<float> env(static_cast<size_t>(N), 0.0f);
    double running = 0.0;
    int count = 0;
    for (int i = 0; i < std::min(half, N); ++i) { running += odf[size_t(i)]; ++count; }
    for (int i = 0; i < N; ++i) {
        const int add = i + half;
        const int rem = i - half - 1;
        if (add < N) { running += odf[size_t(add)]; ++count; }
        if (rem >= 0) { running -= odf[size_t(rem)]; --count; }
        env[size_t(i)] = count > 0 ? float(running / count) : 0.0f;
    }

    // Threshold on max(envelope)
    float envMax = 0.0f;
    for (float v : env) if (v > envMax) envMax = v;
    if (envMax <= 0.0f) return 1.0f;
    const float thresh = threshold * envMax;

    int surviving = 0;
    for (int i = 0; i < N; ++i) {
        if (env[size_t(i)] < thresh) {
            odf[size_t(i)] = 0.0f;
        } else {
            ++surviving;
        }
    }
    return float(surviving) / float(N);
}

// Generalized autocorrelation via FFT (Tolonen/Karjalainen 2000).
// Computes IFFT(|FFT(odf)|^c). With c=2 this reduces to standard biased ACF
// (Wiener-Khinchin). With c<2 the magnitude compression suppresses dominant
// peaks and surfaces weaker periodicities — Percival/Tzanetakis 2014 use
// this front-end. Returns an ACF normalized so acf[0]=1.
//
// Implementation note: zero-pads to next power of 2 ≥ 2N to avoid circular
// wraparound. IFFT is implemented via the conjugate trick — for the real
// nonneg input |X|^c, IFFT(Y)[n] = Re(FFT(Y))[n] / M.
inline std::vector<float> generalizedAcf(const std::vector<float>& odf,
                                         float compression,
                                         const FftFn& fft,
                                         int maxLag) {
    const size_t N = odf.size();
    std::vector<float> acf(size_t(maxLag + 1), 0.0f);
    if (N < 4) return acf;

    size_t M = 1;
    while (M < 2 * N) M <<= 1;
    std::vector<std::complex<float>> buf(M, std::complex<float>(0.0f, 0.0f));
    for (size_t i = 0; i < N; ++i) buf[i] = std::complex<float>(odf[i], 0.0f);

    fft(buf);  // forward
    for (size_t i = 0; i < M; ++i) {
        const float mag = std::abs(buf[i]);
        const float p = std::pow(mag, compression);
        buf[i] = std::complex<float>(p, 0.0f);
    }
    fft(buf);  // forward again — for real nonneg input, IFFT = Re(FFT)/M

    const float invM = 1.0f / float(M);
    const float norm = buf[0].real() * invM;
    if (norm <= 0.0f) return acf;
    const float invNorm = 1.0f / norm;
    const int upper = std::min(maxLag, int(N) - 1);
    for (int lag = 0; lag <= upper; ++lag) {
        acf[size_t(lag)] = buf[size_t(lag)].real() * invM * invNorm;
    }
    return acf;
}

// Percival/Tzanetakis 2014 harmonic enhancement. Adds time-stretched copies
// of the ACF at 2× and 4× to reinforce the true period vs subharmonic peaks
// at 2T, 3T, 4T. Read-modify-write on a snapshot to avoid feedback.
inline void enhanceHarmonics(std::vector<float>& acf, int minLag, int maxLag,
                             float harmEnhWeight2 = 0.5f,
                             float harmEnhWeight4 = 0.25f) {
    const int N = int(acf.size());
    std::vector<float> orig = acf;
    // Attenuated (1, 0.5, 0.25) was empirically tuned. Equal weights (1,1,1)
    // per Percival/Tzanetakis paper destabilize the score balance even with
    // intErr penalty rebalanced — tried at intErrPenaltyWeight=0.6.
    const float w2 = harmEnhWeight2;
    const float w4 = harmEnhWeight4;
    for (int lag = minLag; lag <= maxLag; ++lag) {
        float boost = 0.0f;
        if (lag * 2 < N) boost += w2 * orig[size_t(lag * 2)];
        if (lag * 4 < N) boost += w4 * orig[size_t(lag * 4)];
        acf[size_t(lag)] += boost;
    }
}

// Pulse-train cross-correlation evaluation (Percival/Tzanetakis 2014).
// For a candidate period `lag` (in ODF hops), sweep an impulse train across
// all phases and return the SUM of ODF energy at the best phase, normalized
// by the total ODF energy. Using SUM (not average) is critical for octave
// disambiguation: at lag L the train hits all N/L kicks (total sum), at
// lag 2L it hits N/(2L) kicks (half the sum). The average wouldn't see
// this difference because it divides by count. Using `totalEnergy` as the
// denominator gives a scale-free score in [0..1].
//
// Tolerance window: each pulse is summed over ±tol hops, since the parabolic
// peak of the ACF rarely lands exactly on an integer lag and we want pulses
// near onsets to count even if not exactly aligned.
inline float pulseTrainScore(const std::vector<float>& odf, int lag,
                             float totalEnergy, int tol = 1) {
    const int N = int(odf.size());
    if (lag <= 1 || N < lag * 2 || totalEnergy <= 0.0f) return 0.0f;
    const float invTotal = 1.0f / totalEnergy;
    float best = 0.0f;
    for (int phase = 0; phase < lag; ++phase) {
        float sum = 0.0f;
        for (int p = phase; p < N; p += lag) {
            const int lo = std::max(0, p - tol);
            const int hi = std::min(N - 1, p + tol);
            for (int q = lo; q <= hi; ++q) sum += odf[size_t(q)];
        }
        sum *= invTotal;
        if (sum > best) best = sum;
    }
    return best;
}

// Log-Gaussian weighting in BPM space. Lags corresponding to BPMs far from
// the center (in octaves) get discounted. This is the Klapuri/Ellis fix for
// octave errors — without it, ACF naturally favors longer lags (slower
// tempos) and we get stuck reporting 85 BPM for 170 BPM loops.
inline void applyTempoPrior(std::vector<float>& acf,
                            int minLag, int maxLag,
                            int hopSize, int sampleRate,
                            float centerBpm, float sigmaOctaves,
                            float weight) {
    if (sigmaOctaves <= 0.0f || centerBpm <= 0.0f || weight <= 0.0f) return;
    const float invSigma = 1.0f / sigmaOctaves;
    for (int lag = minLag; lag <= maxLag; ++lag) {
        const float bpm = lagToBpm(lag, hopSize, sampleRate);
        const float z = std::log2(bpm / centerBpm) * invSigma;
        // Interpolate between 1.0 (no weighting) and exp(-0.5 z^2) by `weight`.
        // weight=1.0 → pure Gaussian, weight=0.5 → half pull toward Gaussian.
        const float gaussian = std::exp(-0.5f * z * z);
        const float factor = (1.0f - weight) + weight * gaussian;
        acf[size_t(lag)] *= factor;
    }
}

// Learned re-ranker (MLP) trained by tools/bpm-bench/train_reranker.py.
// Held-out file accuracy below; weights auto-generated; do not edit.
namespace learned_mlp {
// Feature order (compute in detector exactly the same way):
//   [0] here
//   [1] pulseEvidence
//   [2] intErr
//   [3] logMPenalty
//   [4] isPow2
//   [5] distToPriorOctaves
//   [6] score
//   [7] here_x_pulse
//   [8] intErr_sq
//   [9] fast_x_pulse
//   [10] slow_x_pulse
//   [11] log_bpm
//   [12] abs_log_bpm
//   [13] pow2_x_low_err
//   [14] m_one
//   [15] m_half_or_double

static constexpr int kL0_in = 16;
static constexpr int kL0_out = 64;
static constexpr float kL0_W[1024] = {
    +0.023862f, +1.673757f, +0.828211f, +0.982983f, +0.230278f, -1.125746f, -3.255719f, +0.513421f, -1.439991f, +0.855635f, -1.793462f, +1.656083f, +2.155123f, -0.955351f, -1.797045f, -1.203216f, -0.493241f, +1.201979f, -1.208269f, -0.878017f, +0.965789f, -1.577540f, +0.651729f, -2.358546f, -2.392662f, +3.168882f, -1.987891f, +0.175873f, +0.655931f, -0.190477f, +0.647757f, -1.585179f, -1.209547f, +0.150698f, +2.158938f, +0.636776f, -0.288413f, -2.066644f, -0.793460f, -3.084510f, -2.473140f, -3.304854f, -1.088561f, +2.791188f, -1.761238f, -0.976362f, -0.490344f, -2.938017f, -1.026400f, -3.503779f, +0.413980f, +1.580000f, +0.580593f, +0.097898f, -0.254171f, +0.320507f, -2.954921f, -2.407844f, -1.223105f, -0.261998f, -1.601040f, -1.929620f, +0.927521f, +1.660317f,
    -0.764804f, +2.043463f, -2.680932f, +2.746039f, -1.357119f, +5.215820f, +4.512895f, -1.962452f, +0.499686f, +1.195907f, +2.580444f, +2.957190f, +1.398144f, -2.637510f, +1.486835f, +1.279146f, +3.463653f, -0.655159f, +0.365257f, -1.972974f, +0.725607f, -0.429285f, -5.528042f, +2.935254f, +1.500221f, -2.681522f, +1.327400f, +1.875029f, -1.198960f, -5.644644f, +0.690477f, +3.675258f, +1.965620f, +1.245629f, -5.218519f, -1.109047f, -3.186187f, +1.381750f, +1.710766f, +1.962104f, +2.162178f, +2.120395f, +2.766310f, +1.413163f, +0.857385f, -2.227707f, -3.953843f, -0.398931f, +3.689512f, +1.397304f, +3.384047f, +1.310504f, +3.516336f, -2.615341f, +2.486980f, +1.881298f, +2.713064f, +3.443445f, +1.647732f, -0.195020f, +1.099376f, +2.286733f, +0.639842f, -7.618473f,
    -5.039437f, +3.357667f, -1.362102f, -0.080620f, +1.861323f, +4.779366f, +2.286585f, -1.183912f, -0.427076f, +4.877623f, -3.977635f, +6.667588f, +1.234216f, -4.034319f, -4.430700f, -3.504673f, -3.614353f, +0.273681f, +0.450729f, -3.216954f, -1.715636f, +3.060845f, +1.956558f, +0.371989f, -0.988421f, +1.074989f, +4.467435f, +0.168752f, +2.778561f, +1.634302f, +0.607479f, +2.474106f, -4.678947f, -4.593814f, +2.731169f, -1.362449f, -3.307397f, +5.037841f, -1.377497f, -1.192033f, -0.498057f, +3.440945f, +0.001859f, -0.517813f, +0.691351f, -2.273542f, +0.120336f, -0.659454f, -4.077946f, +1.816106f, +1.951860f, +0.076313f, -5.301604f, -1.399262f, -3.071757f, +1.299608f, -2.025138f, -2.196175f, -2.653102f, +1.993700f, -3.453813f, -4.216163f, -4.442636f, +2.133529f,
    +0.425216f, +0.140221f, -0.179202f, -0.015461f, +0.341323f, +0.622202f, +1.007530f, +0.417649f, +0.322134f, -0.649114f, -0.423373f, +0.951178f, +0.676449f, -0.349661f, +0.044093f, -0.012849f, -0.326607f, -0.715360f, -0.570437f, +0.200204f, -0.093078f, -0.192122f, +0.403863f, -1.330635f, -0.183142f, +0.319226f, -0.233348f, +0.844242f, +0.022233f, -0.324785f, -1.057282f, -0.995964f, -0.054605f, -0.232466f, +0.025691f, +0.071403f, +0.421431f, -0.509543f, +0.255649f, +0.507498f, +0.453095f, -0.386624f, -0.577311f, +1.200400f, -0.568723f, -0.762161f, -0.264715f, -0.720606f, +0.219331f, +1.073216f, +0.338996f, -0.057572f, -0.215212f, +0.402972f, +0.057130f, +0.232657f, -0.094333f, +0.303493f, -0.294318f, +0.062788f, +0.296566f, -0.323371f, -0.448015f, -0.166307f,
    +0.200911f, +1.204631f, -0.013701f, -0.622947f, +0.422548f, +1.015437f, -0.177308f, -0.375338f, +1.026032f, +0.850550f, +0.249280f, +1.137540f, -0.244539f, -0.115519f, +0.243492f, -0.109395f, +0.384234f, +0.745232f, -0.629568f, -0.370991f, +0.160633f, +0.571615f, -0.369211f, +0.819134f, +0.395649f, -0.631898f, +0.214473f, -0.839448f, +0.651561f, -0.558838f, -0.334993f, -0.176257f, -0.072024f, -0.051147f, -0.474002f, -0.331290f, +0.026852f, -0.480434f, -0.912223f, -0.121779f, +0.210009f, +0.089102f, -0.744554f, -1.048175f, -0.897412f, -0.392627f, -0.136318f, +0.058263f, +0.264549f, +1.384928f, -0.972756f, -0.357057f, +0.206482f, -0.583433f, -0.336184f, -0.850325f, -1.013463f, +0.304245f, +0.039239f, +0.063275f, -0.218772f, -0.216772f, -0.530769f, -0.928423f,
    +0.393030f, +0.614201f, -0.235915f, -0.064080f, +0.190762f, +1.201388f, +0.275567f, -0.365992f, -0.367390f, -0.867297f, +0.395499f, +0.801448f, -1.100751f, -0.048006f, -0.561835f, -0.694393f, -0.042975f, +0.100538f, -0.637574f, -0.239942f, -0.330426f, -0.109320f, +0.208279f, -0.473415f, -0.240626f, -1.949591f, -0.344464f, -0.102351f, -0.907919f, +0.177478f, -0.272976f, +1.217955f, -0.157881f, -0.810106f, -0.314491f, +0.060472f, -0.589793f, -0.270428f, +0.782044f, -0.483641f, -0.568485f, +0.176600f, -0.066230f, +0.886569f, +0.311515f, -0.337081f, +0.066392f, +0.595260f, -0.324387f, -1.298994f, -1.691911f, -1.413228f, -0.545564f, +0.469767f, -0.745339f, +0.569933f, +0.321387f, +0.007938f, +0.137008f, -0.366828f, -0.407475f, +0.013199f, +0.229107f, +0.043106f,
    +1.222706f, +0.826458f, -0.333766f, -0.783245f, +0.498710f, +0.228043f, +0.721724f, -0.597418f, -0.384137f, +0.445524f, +0.485116f, +0.598745f, +1.065055f, +0.382366f, +1.114883f, +0.881659f, -0.312389f, +1.148846f, -0.342028f, +1.409693f, -0.047400f, +0.434794f, -0.448012f, -1.793442f, -0.457043f, -0.269224f, -0.789290f, +0.725544f, +0.615380f, +0.094720f, +0.138835f, -0.437909f, +0.346175f, -0.340060f, +0.458632f, +0.876525f, +0.892367f, -0.327308f, -1.150049f, -1.134383f, -2.139027f, -1.249299f, +0.265873f, +1.177795f, -1.011989f, -1.670752f, -1.068348f, -0.846194f, +0.849779f, -2.563562f, -1.977872f, +0.481590f, -0.641544f, -0.343525f, +0.961528f, -1.517164f, -1.281802f, +0.892502f, +0.559632f, -0.353703f, +0.943602f, +1.438960f, +1.336884f, +0.643371f,
    -1.252113f, -2.190497f, +4.036036f, +3.570731f, +1.677998f, +7.248771f, -3.846436f, -4.080435f, -0.539498f, +1.108567f, -5.479758f, -1.319438f, -2.083230f, +1.662459f, +4.629079f, +4.144130f, +7.596189f, -0.543476f, +0.114298f, +2.404432f, -0.144359f, +7.478349f, -14.787493f, +5.186118f, -4.485371f, +3.635427f, +2.726073f, +6.183354f, -6.767116f, -7.083824f, +5.983661f, +3.986887f, +3.406915f, -0.219266f, +1.726695f, +1.096421f, +0.808152f, +6.076226f, -1.409764f, -1.308986f, +4.449924f, +2.615921f, +2.257054f, +2.403099f, -0.272498f, -3.483405f, -4.412996f, -2.135012f, +1.492427f, -4.897095f, +1.505512f, +6.495334f, +1.819107f, -6.747105f, -3.558062f, +5.114595f, -3.382508f, -7.206723f, +8.240260f, +5.214655f, +5.085292f, +3.037776f, -1.111075f, -5.303033f,
    +6.716978f, -1.780933f, +0.753112f, -4.773811f, -1.446792f, +0.041438f, +0.190414f, -6.705787f, -0.753462f, -0.744571f, +2.864223f, +1.984722f, +0.137185f, +2.845233f, +1.736238f, +0.901268f, +0.438115f, -1.080473f, +1.598283f, +5.378618f, +7.712722f, -1.937857f, +1.018449f, +1.083152f, -0.646738f, -3.430323f, -5.825133f, -4.313531f, +3.064442f, +2.184330f, -4.784259f, -2.058406f, +2.922088f, -0.211504f, +1.909809f, +3.870825f, +3.637356f, -0.796235f, -4.045467f, -0.308131f, -2.629673f, -4.090432f, +2.875580f, -3.527442f, -8.231290f, -7.000444f, -0.833879f, -1.283759f, +0.990280f, -5.695861f, -5.185588f, +2.358746f, +0.365970f, -4.215415f, +0.491065f, -6.083304f, -0.992000f, -0.178259f, +0.918184f, +1.707008f, +1.952169f, +4.052054f, +7.823060f, +4.362626f,
    -0.835642f, +3.646748f, -0.714297f, -2.065663f, +1.143789f, +4.511027f, +0.952253f, -0.667208f, +0.133307f, -3.206809f, -1.267109f, +0.284874f, -2.763673f, -0.874301f, -0.314573f, +2.227903f, -1.099511f, +1.134669f, -1.194841f, +0.168130f, +2.104059f, -0.656808f, -1.974928f, -2.984112f, -1.692615f, -6.153384f, -0.116030f, -2.567027f, -2.696171f, -1.643833f, -2.362338f, +0.859732f, +1.145665f, +1.065892f, -4.835437f, +1.151363f, +0.310685f, -1.146647f, +2.803312f, -0.428046f, -3.643345f, +1.163926f, +2.573480f, +2.507875f, -0.516567f, -1.349904f, -1.736875f, +1.304510f, +1.594560f, -2.011315f, -2.541578f, +2.317561f, -1.397067f, -0.643374f, +0.763163f, +1.500977f, +0.821032f, +2.186948f, -0.324993f, +0.968446f, +1.494882f, +0.696070f, -1.509941f, -5.343407f,
    -0.522138f, -3.578678f, -0.351487f, -1.469517f, -1.482679f, -3.066793f, -0.763378f, -0.937530f, +6.941451f, -0.150277f, -2.310211f, -2.623228f, +3.497499f, +1.184379f, +0.634256f, +0.631444f, +1.026434f, -2.357327f, +4.439339f, +1.206942f, +0.426067f, +3.775985f, -0.425393f, -1.788077f, -2.257635f, +1.918640f, -0.603510f, -2.324451f, -0.105054f, +0.403347f, -1.639423f, +2.326936f, -0.601082f, +2.264746f, -0.843732f, +2.731932f, -0.924448f, -1.394810f, +3.187702f, +6.922646f, +3.177006f, -3.136944f, -1.161985f, +1.090088f, -0.046738f, +0.709674f, +1.614912f, -2.365724f, +0.676807f, -4.001026f, +1.254091f, -1.372230f, +0.640651f, -3.964591f, -0.416059f, +1.935256f, -3.630407f, -0.214551f, +0.121017f, +1.260081f, +0.482071f, -0.547812f, +0.125953f, +1.645796f,
    -0.127043f, +0.909008f, -0.596597f, +0.496002f, -0.548643f, +1.141128f, +0.495593f, -0.383176f, -0.311897f, -0.691661f, +0.387911f, +0.985174f, -0.227327f, +0.860699f, -0.478042f, +0.043658f, +0.045725f, -0.458557f, -1.072156f, +0.447865f, -0.242448f, +0.285008f, +0.172628f, -0.087658f, -0.450939f, -2.433454f, -0.845973f, +0.714687f, -0.597009f, +0.067952f, -0.427295f, +1.240125f, -0.442541f, -0.110334f, +0.079249f, -0.222696f, -0.126572f, -0.573230f, +0.576147f, -0.561283f, -0.735020f, +0.450900f, -0.144671f, +0.884366f, +0.274058f, -0.082605f, -0.821736f, +0.632165f, +0.137530f, -1.015761f, -1.650913f, -0.595327f, +0.016784f, +0.310064f, +0.133367f, +1.068825f, -0.418514f, +0.316326f, -0.466795f, +0.117139f, -0.497980f, -0.122562f, +0.371835f, +0.042350f,
    -0.491241f, +0.620979f, -0.863762f, +0.998563f, +0.268890f, -0.864077f, -0.766886f, -1.169194f, -0.840874f, +0.210250f, +1.024281f, +1.112067f, +1.755749f, -0.458612f, -0.405636f, -0.112766f, +0.026358f, +1.463180f, -0.498920f, -1.020431f, -0.691594f, +1.976162f, -0.509122f, -0.847797f, -1.009673f, +1.797550f, -2.191348f, +2.181568f, +1.366671f, +0.284986f, +1.462432f, +0.530146f, +0.906958f, -0.338735f, -0.998603f, -1.342592f, -0.413732f, -0.592610f, +1.234652f, +1.212005f, -0.946294f, -1.712308f, +0.521019f, +4.037820f, -2.311128f, +1.981626f, +1.097144f, -0.510275f, -0.259512f, -2.019987f, -2.431803f, -0.025825f, +0.434046f, -1.768348f, -0.245311f, +1.689431f, -1.218700f, +0.604941f, +0.057823f, +0.049016f, +0.520514f, +0.472732f, -0.630470f, -0.221035f,
    -0.539586f, -0.275413f, +0.041956f, -0.070084f, +1.229997f, -0.805098f, +0.323672f, -0.835745f, -1.935310f, +0.831784f, -0.006960f, +1.470047f, -0.805654f, +0.213779f, -0.537233f, +0.603306f, +0.120474f, -0.144286f, -1.469516f, +0.325498f, -0.746107f, +1.163567f, -1.020910f, -1.188857f, -2.128024f, -0.144693f, +0.727823f, -0.937824f, +0.583623f, -0.647965f, -0.575575f, -0.361372f, +0.066846f, +0.128935f, -1.599769f, -0.219214f, +0.468653f, +0.726304f, -1.218734f, -0.820705f, -0.308437f, +0.396907f, -2.049461f, -1.250533f, -0.017350f, -0.303454f, -1.569986f, -1.758993f, +0.021991f, +0.260223f, -3.013208f, -1.920896f, +0.409327f, -1.553607f, +0.121452f, -1.022632f, -1.657779f, -0.536798f, +0.235261f, +1.009705f, +0.555230f, +0.039311f, +0.305990f, -1.064459f,
    +0.084036f, +0.449038f, -0.372998f, -0.441194f, -0.375581f, -0.790373f, -0.908207f, -0.377780f, +0.078808f, +0.632252f, +0.355881f, -0.663949f, -1.471107f, -0.384677f, -0.137359f, +0.038621f, -0.012444f, -0.021951f, +0.710567f, +0.076237f, +0.310815f, +0.575834f, +0.147729f, +0.414405f, +0.554309f, -0.819931f, +1.135833f, -0.496818f, -0.275353f, -0.282016f, -0.164117f, +0.870338f, -0.024887f, -0.065437f, +0.263214f, +0.343308f, +0.629254f, +0.442465f, -0.350073f, -0.181293f, -0.760459f, +0.731902f, -0.261558f, -1.814417f, -0.368853f, +0.554296f, +0.322221f, +0.751904f, +0.075875f, -1.134583f, -0.874793f, +0.743164f, +0.070823f, -0.568203f, +0.103999f, +0.325954f, +0.334198f, +0.461829f, -0.073436f, -0.117545f, +0.301514f, -0.172785f, -0.208254f, +0.622544f,
    +0.033057f, -0.130974f, +0.031631f, +0.220252f, -0.270541f, -0.372827f, +0.782858f, -0.868161f, +0.568365f, +0.042554f, +0.143219f, +0.748921f, -0.234148f, -0.162806f, +0.146832f, +0.160930f, +0.455748f, -1.124881f, -0.220057f, +0.326253f, +0.382145f, +0.652164f, -0.316584f, -0.743509f, +0.677981f, -0.745642f, +0.422187f, -0.668176f, -0.688303f, +0.029482f, -0.283623f, +0.146195f, -0.032214f, +0.454159f, +0.241721f, -0.108742f, +0.494839f, -0.777585f, +0.262387f, +1.256753f, +0.145914f, -0.360622f, +0.325243f, -0.397611f, +0.021877f, +0.654639f, -0.512337f, +0.433989f, -0.087255f, +0.318927f, -1.459533f, +1.077602f, +0.812734f, -1.265949f, +0.444610f, -0.026641f, -0.681197f, -0.104117f, +0.350785f, -0.285272f, +0.144326f, +0.064380f, +0.669203f, +0.671746f,
};
static constexpr float kL0_b[64] = {
    +0.874431f, -1.670096f, +1.118517f, -0.130247f, -0.749660f, -1.128851f, -0.902361f, +2.250003f, -0.312280f, -1.626514f, +0.177533f, -3.328338f, -1.608018f, +1.985423f, +0.123713f, -0.167039f, -0.172501f, +0.093001f, +0.388549f, +0.681891f, -0.437547f, -1.844605f, +1.114194f, +0.957353f, +1.411430f, -0.322082f, +0.208035f, -0.764603f, -0.892220f, +1.196572f, +0.652460f, -0.535762f, +0.022421f, +0.335422f, +0.582156f, +0.208687f, +0.562600f, +0.001088f, +0.236310f, -0.780288f, +1.038244f, +0.953840f, -0.559313f, -1.876520f, +2.700826f, +1.162196f, +0.730869f, +1.490199f, -0.408100f, +1.292041f, +1.593921f, -1.725901f, -0.237625f, +3.501822f, -0.340460f, -0.590418f, +2.251851f, -0.582010f, -0.094950f, -0.754715f, -0.342899f, +0.033772f, +0.598952f, +0.621047f
};

static constexpr int kL1_in = 64;
static constexpr int kL1_out = 1;
static constexpr float kL1_W[64] = {
    +0.477086f,
    -0.432111f,
    +0.204615f,
    -0.175809f,
    -0.094227f,
    -0.330840f,
    -0.344160f,
    -0.426006f,
    -0.380697f,
    -0.541577f,
    +0.517459f,
    -0.548254f,
    -0.546295f,
    +0.455115f,
    +0.479240f,
    +0.368300f,
    +0.342738f,
    -0.348412f,
    -0.234489f,
    +0.343659f,
    +0.209044f,
    -0.452436f,
    +0.384058f,
    -0.423171f,
    -0.336511f,
    -0.825816f,
    -0.491884f,
    -0.234768f,
    -0.519984f,
    +0.359394f,
    -0.336732f,
    -0.571232f,
    +0.426578f,
    +0.607939f,
    +0.411112f,
    +0.169417f,
    +0.140596f,
    -0.326938f,
    -0.311264f,
    -0.484868f,
    -0.319607f,
    -0.577947f,
    +0.365437f,
    -0.404663f,
    -0.530036f,
    -0.400516f,
    +0.320756f,
    -0.590252f,
    +0.472243f,
    -0.333058f,
    -0.356491f,
    +0.398802f,
    +0.350105f,
    -0.348666f,
    +0.333098f,
    -0.206936f,
    -0.287866f,
    +0.375341f,
    +0.230826f,
    -0.076625f,
    +0.240778f,
    +0.523874f,
    +0.356338f,
    +0.541899f,
};
static constexpr float kL1_b[1] = {
    -0.017915f
};

// Max hidden dim across layers: 64
inline float forward(const float* x) {
    float buf[2][64] = {};
    int cur = 0;
    // Layer 0: in=16 out=64
    for (int j = 0; j < 64; ++j) {
        float s = kL0_b[j];
        for (int i = 0; i < 16; ++i)
            s += x[i] * kL0_W[i * 64 + j];
        buf[cur ^ 1][j] = s > 0.0f ? s : 0.0f;  // ReLU
    }
    cur ^= 1;
    // Layer 1: in=64 out=1
    for (int j = 0; j < 1; ++j) {
        float s = kL1_b[j];
        for (int i = 0; i < 64; ++i)
            s += buf[cur][i] * kL1_W[i * 1 + j];
        buf[cur ^ 1][j] = s;  // output (logit)
    }
    return buf[cur ^ 1][0];
}
} // namespace learned_mlp

} // namespace detail

// Main entry point. Caller supplies mono-or-multichannel interleaved audio,
// sample rate, and an FFT callable. Returns the detected BPM plus diagnostics.
inline DetectionResult detectBpm(const float* interleaved,
                                 size_t frames,
                                 int channels,
                                 int sampleRate,
                                 const FftFn& fft,
                                 const DetectorOptions& opts = {}) {
    DetectionResult result;
    if (!interleaved || frames == 0 || channels <= 0 || sampleRate <= 0) {
        return result;
    }

    auto mono = detail::downmixToMono(interleaved, frames, channels);

    // NOTE: Font/Serra 2016 silence trim was tried (5%, 1%, 0.5% of peak,
    // leading-only and bidirectional). All variants regressed by 6-22 points
    // on this bench corpus because the loops are *already* tightly trimmed
    // by their respective sample-pack publishers — any trimming makes durSec
    // shorter than the true musical length and breaks the integer-beat snap.
    // The Font/Serra recipe assumes loop libraries silence-pad, which most
    // commercial packs we tested do not. May be re-enabled if a future test
    // set is dominated by Freesound-style user-uploaded loops.
    const double durSec = double(mono.size()) / double(sampleRate);

    // Compute ODF
    auto window = detail::makeHannWindow(opts.fftSize);
    result.odf = detail::spectralFluxOdf(mono, opts.fftSize, opts.hopSize, fft,
                                         window, opts.superFluxMu, opts.numOdfBands,
                                         sampleRate);
    if (result.odf.size() < 8) return result;

    // Phase K: mask out non-rhythmic regions (intros, breakdowns, outros)
    // so they don't dilute the ACF. Only meaningful for long samples with
    // varying rhythmic activity — loops are uniformly rhythmic so this is
    // a no-op there, but we still only enable when explicitly requested
    // (SampleAssetPool turns it on for samples >30s).
    if (opts.maskNonRhythmicRegions) {
        detail::maskByRhythmicSalience(result.odf, opts.hopSize, sampleRate,
                                       opts.rhythmicMaskWindowSec,
                                       opts.rhythmicMaskThreshold);
    }

    // Tempo range → lag range (lower BPM = longer lag)
    const int minLag = detail::bpmToLag(opts.maxBpm, opts.hopSize, sampleRate);
    int maxLag = detail::bpmToLag(opts.minBpm, opts.hopSize, sampleRate);
    if (minLag < 2) return result;
    // Clamp maxLag to what the ODF actually has room for. Without this,
    // very short loops (< 1 second) silently return 0 because the ACF needs
    // odf.size() > maxLag + 2 hops. This effectively raises the floor on
    // the slowest detectable tempo for short clips, but at least returns
    // *some* answer instead of nothing.
    if (maxLag >= int(result.odf.size()) - 2) {
        maxLag = int(result.odf.size()) - 2;
    }
    if (maxLag <= minLag) return result;

    if (opts.useGeneralizedAcf) {
        result.acf = detail::generalizedAcf(result.odf, opts.acfCompression,
                                            fft, maxLag);
    } else {
        result.acf = detail::autocorrelateOdf(result.odf, minLag, maxLag);
    }

    if (opts.useHarmonicEnhancement) {
        detail::enhanceHarmonics(result.acf, minLag, maxLag,
                                 opts.harmEnhWeight2, opts.harmEnhWeight4);
    }

    // Apply the general music tempo prior (log-Gaussian centered near 120 BPM)
    // as a multiplicative weighting on the ACF. This is the Klapuri/Ellis fix
    // for octave errors: without it the ACF naturally prefers longer lags
    // (slower tempos), so 170 BPM loops detect as 85 BPM. The Gaussian in
    // log-tempo space gently pulls the chosen peak toward musically plausible
    // tempos without fully overriding strong evidence for extreme tempos.
    detail::applyTempoPrior(result.acf, minLag, maxLag,
                            opts.hopSize, sampleRate,
                            opts.tempoPriorCenterBpm,
                            opts.tempoPriorSigmaOctaves,
                            /*weight=*/0.7f);

    // Optional tighter project-BPM prior on top of the general prior.
    if (opts.projectBpmHint > 0.0f) {
        detail::applyTempoPrior(result.acf, minLag, maxLag,
                                opts.hopSize, sampleRate,
                                opts.projectBpmHint,
                                opts.projectBpmSigmaOctaves,
                                /*weight=*/0.5f);
    }

    // Score every local-maximum lag in the valid band. For each candidate
    // we also consider its 0.5×, 2×, 2/3×, and 1.5× harmonics in case the
    // ACF peak landed on a sub/super-multiple of the true beat period.
    // Final score combines:
    //   - prior-weighted ACF peak strength (higher is better)
    //   - closeness of beats-in-loop to an integer (lower is better)
    //   - slight preference for staying at the raw peak (m == 1)
    //   - project BPM prior bonus when hint is present
    float bestScore = -1e9f;
    float bestBpm = 0.0f;
    int bestBeats = 0;
    float bestConfidence = 0.0f;

    // Precompute total ODF energy for pulse-train score normalization
    float odfTotalEnergy = 0.0f;
    for (float v : result.odf) odfTotalEnergy += v;

    for (int lag = minLag + 1; lag < maxLag; ++lag) {
        const float here = result.acf[size_t(lag)];
        if (here <= 0.0f) continue;
        // Require local maximum to kill noise-floor clutter. Use rising-edge
        // tolerance (strict on one side, loose on the other) so plateau peaks
        // aren't dropped.
        const float yL = result.acf[size_t(lag - 1)];
        const float yR = result.acf[size_t(lag + 1)];
        if (here < yL) continue;
        if (here <= yR) continue;

        // Parabolic interpolation around the peak gives sub-lag precision
        // for the raw BPM estimate, which lets the integer-beat snap stay
        // tight without losing 1% imprecision corrections.
        const float fracOffset = detail::parabolicPeakOffset(yL, here, yR);
        const float fracLag = float(lag) + fracOffset;
        const float rawBpm = detail::fractionalLagToBpm(fracLag, opts.hopSize, sampleRate);

        const float multipliers[] = { 1.0f, 0.5f, 2.0f, 2.0f / 3.0f, 1.5f,
                                      1.0f / 3.0f, 3.0f };
        for (float m : multipliers) {
            const float candBpm = rawBpm * m;
            if (candBpm < opts.minBpm || candBpm > opts.maxBpm) continue;

            const double beatsInLoop = durSec * double(candBpm) / 60.0;
            const int rounded = int(std::round(beatsInLoop));
            if (rounded < 1) continue;
            const double intErr = std::abs(beatsInLoop - rounded);

            float penalty = std::abs(std::log2(m)) * 0.05f;
            if (opts.useLoopDurationPrior) {
                penalty += opts.intErrPenaltyWeight * float(intErr);
                if ((rounded & (rounded - 1)) == 0) penalty -= opts.powerOfTwoBeatsBonus;
            }

            if (opts.projectBpmHint > 0.0f) {
                const float rel = std::abs(candBpm - opts.projectBpmHint)
                                / opts.projectBpmHint;
                if (rel < 0.05f) penalty -= 0.10f;
            }

            // Pulse-train cross-correlation evidence: how well does an
            // impulse train at this candidate's period actually align with
            // ODF onsets? Better octave/harmonic resolver than blindly
            // multiplying the raw ACF peak by 0.5×/2×/etc.
            float pulseEvidence = 0.0f;
            if (opts.usePulseTrainScoring) {
                const int candLag = detail::bpmToLag(candBpm, opts.hopSize, sampleRate);
                pulseEvidence = detail::pulseTrainScore(result.odf, candLag, odfTotalEnergy);
            }

            const float handTunedScore = here + opts.pulseTrainWeight * pulseEvidence - penalty;

            float score = handTunedScore;
            if (opts.useLearnedReRanker) {
                // Build feature vector matching tools/bpm-bench/train_reranker.py
                const int isPow2Int = ((rounded > 0) && ((rounded & (rounded - 1)) == 0)) ? 1 : 0;
                const float distToPrior = std::log2(candBpm / 120.0f);
                const float here_x_pulse = here * pulseEvidence;
                const float intErr_sq = float(intErr) * float(intErr);
                const float fast_x_pulse = (candBpm > 130.0f) ? pulseEvidence : 0.0f;
                const float slow_x_pulse = (candBpm < 100.0f) ? pulseEvidence : 0.0f;
                const float log_bpm = std::log2(candBpm / 120.0f);
                const float abs_log_bpm = std::abs(log_bpm);
                const float pow2_x_low_err = float(isPow2Int) * (intErr < 0.1 ? 1.0f : 0.0f);
                const float m_one = (std::abs(m - 1.0f) < 0.01f) ? 1.0f : 0.0f;
                const float m_half_or_double =
                    ((std::abs(m - 0.5f) < 0.01f) || (std::abs(m - 2.0f) < 0.01f)) ? 1.0f : 0.0f;
                const float feat[16] = {
                    here, pulseEvidence, float(intErr), std::abs(std::log2(m)),
                    float(isPow2Int), distToPrior, handTunedScore, here_x_pulse,
                    intErr_sq, fast_x_pulse, slow_x_pulse, log_bpm, abs_log_bpm,
                    pow2_x_low_err, m_one, m_half_or_double
                };
                score = detail::learned_mlp::forward(feat);
            }

            // Stash this candidate's features for the bench / training pipeline
            Candidate cand;
            cand.candBpm = candBpm;
            cand.rounded = rounded;
            cand.here = here;
            cand.pulseEvidence = pulseEvidence;
            cand.intErr = float(intErr);
            cand.multiplier = m;
            cand.logMPenalty = std::abs(std::log2(m));
            cand.isPow2 = ((rounded > 0) && ((rounded & (rounded - 1)) == 0)) ? 1 : 0;
            cand.distToPriorOctaves = std::log2(candBpm / 120.0f);
            cand.lag = lag;
            cand.beatsInLoop = float(beatsInLoop);
            cand.score = score;
            result.candidates.push_back(cand);

            if (score > bestScore) {
                bestScore = score;
                // Snap to integer beats only when the ACF says we're already
                // very close to an integer beat count and the duration prior
                // is enabled. This prevents snapping non-bar-aligned loops
                // the wrong way (3.9s 120 BPM → 123, 4.0s 125 BPM → 120).
                if (opts.useLoopDurationPrior &&
                    intErr <= opts.snapMaxIntErrFraction) {
                    bestBpm = float(rounded) * 60.0f / float(durSec);
                    bestBeats = rounded;
                } else {
                    bestBpm = candBpm;
                    bestBeats = 0;
                }
                bestConfidence = here;
            }
        }
    }

    // (Octave doubling check was tried here but proved to be a wash on the
    // bench — fixed high-tempo cases ~+30 points but broke genuine 70-95 BPM
    // loops by the same amount. Without external context [filename hint or
    // project BPM] there's no reliable way to disambiguate "this loop is
    // 85 BPM" from "this loop is 170 BPM heard at half speed". Left out
    // intentionally; the user-provided projectBpmHint covers this when set.)

    result.bpm = bestBpm;
    result.beatsInLoop = bestBeats;
    result.confidence = bestConfidence;
    return result;
}

// ---------------------------------------------------------------------------
// Onset detection — reusable entry point for warp marker generation.
//
// Runs the same spectral-flux ODF used by detectBpm(), then peak-picks prominent
// onsets across equal source-time regions. Returns their positions in seconds,
// sorted chronologically. The caller can map these to beat-grid positions for
// warp markers.
//
// Parameters:
//   mono        — mono audio buffer
//   sampleRate  — audio sample rate
//   fft         — caller-provided FFT function (same as detectBpm)
//   maxOnsets   — return at most this many onsets (default 16)
//   fftSize     — STFT window size (default 2048)
//   hopSize     — STFT hop size (default 512)
// ---------------------------------------------------------------------------
struct OnsetResult {
    std::vector<float> onsetTimesSeconds;  // sorted chronologically
    std::vector<float> odf;                // raw ODF for debugging
};

namespace detail {

struct OnsetPeak {
    int frame = 0;
    float strength = 0.0f;
};

// Select one strong onset from each equal source-time region before filling any
// unused capacity by strength. This prevents a loud intro from consuming a
// fixed marker budget while later, quieter beats disappear from the map.
inline std::vector<OnsetPeak> selectTimeDistributedPeaks (std::vector<OnsetPeak> peaks,
                                                         int maxPeaks,
                                                         int frameCount) {
    if (maxPeaks <= 0 || peaks.empty()) return {};
    std::sort(peaks.begin(), peaks.end(), [] (const auto& a, const auto& b) {
        return a.frame < b.frame;
    });
    if (int(peaks.size()) <= maxPeaks) return peaks;

    frameCount = std::max(frameCount, peaks.back().frame + 1);
    std::vector<bool> chosen(peaks.size(), false);
    std::vector<OnsetPeak> selected;
    selected.reserve(size_t(maxPeaks));

    for (int bucket = 0; bucket < maxPeaks; ++bucket) {
        const int lo = bucket * frameCount / maxPeaks;
        const int hi = (bucket + 1) * frameCount / maxPeaks;
        size_t best = peaks.size();
        for (size_t i = 0; i < peaks.size(); ++i) {
            if (peaks[i].frame < lo || peaks[i].frame >= hi) continue;
            if (best == peaks.size() || peaks[i].strength > peaks[best].strength)
                best = i;
        }
        if (best != peaks.size()) {
            chosen[best] = true;
            selected.push_back(peaks[best]);
        }
    }

    if (int(selected.size()) < maxPeaks) {
        std::vector<size_t> remaining;
        for (size_t i = 0; i < peaks.size(); ++i)
            if (!chosen[i]) remaining.push_back(i);
        std::sort(remaining.begin(), remaining.end(), [&peaks] (size_t a, size_t b) {
            return peaks[a].strength > peaks[b].strength;
        });
        for (size_t i : remaining) {
            if (int(selected.size()) >= maxPeaks) break;
            selected.push_back(peaks[i]);
        }
    }

    std::sort(selected.begin(), selected.end(), [] (const auto& a, const auto& b) {
        return a.frame < b.frame;
    });
    return selected;
}

} // namespace detail

inline OnsetResult computeOnsets(const std::vector<float>& mono,
                                 int sampleRate,
                                 const FftFn& fft,
                                 int maxOnsets = 16,
                                 int fftSize = 2048,
                                 int hopSize = 512) {
    OnsetResult result;
    if (mono.size() < size_t(fftSize)) return result;

    auto window = detail::makeHannWindow(fftSize);
    result.odf = detail::spectralFluxOdf(mono, fftSize, hopSize, fft, window);

    if (result.odf.size() < 3) return result;

    const float secondsPerHop = float(hopSize) / float(sampleRate);

    // Adaptive threshold: median + 0.5 * stddev of the ODF.
    // This picks out genuinely prominent transients while ignoring low-level
    // spectral wobble. The 0.5 multiplier is conservative — it captures most
    // audible onsets without flooding with false positives.
    std::vector<float> sorted = result.odf;
    std::sort(sorted.begin(), sorted.end());
    const float median = sorted[sorted.size() / 2];

    float sumSq = 0.0f;
    for (float v : result.odf) sumSq += (v - median) * (v - median);
    const float stddev = std::sqrt(sumSq / float(result.odf.size()));
    const float threshold = median + 0.5f * stddev;

    // Peak-pick: local maxima above threshold, with a minimum inter-onset
    // gap of 50ms (~2.2 hops at 512/44100) to avoid double-triggers.
    const int minGapFrames = std::max(1, int(0.05f / secondsPerHop));

    std::vector<detail::OnsetPeak> peaks;

    for (int f = 1; f < int(result.odf.size()) - 1; ++f) {
        float val = result.odf[size_t(f)];
        if (val <= threshold) continue;
        if (val <= result.odf[size_t(f - 1)] || val < result.odf[size_t(f + 1)])
            continue;

        // Enforce minimum gap from previously accepted peak
        if (!peaks.empty() && (f - peaks.back().frame) < minGapFrames)
        {
            // Keep the stronger one
            if (val > peaks.back().strength)
                peaks.back() = { f, val };
            continue;
        }
        peaks.push_back({ f, val });
    }

    // Keep coverage across the source instead of taking the globally loudest N,
    // which can discard every beat after a loud intro or chorus.
    peaks = detail::selectTimeDistributedPeaks(std::move(peaks), maxOnsets,
                                               int(result.odf.size()));

    result.onsetTimesSeconds.reserve(peaks.size());
    for (const auto& p : peaks)
        result.onsetTimesSeconds.push_back(float(p.frame) * secondsPerHop);

    return result;
}

} // namespace arbit::bpm
