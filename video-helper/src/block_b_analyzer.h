// block_b_analyzer.h — Arbit Media Machine streaming Block B analyzer (pre-work A3)
//
// The stateful counterpart to block_b_defs.h. A BlockBAnalyzer is fed mono PCM
// in arbitrary chunks via pushSamples(); it emits one FeatureFrame per kHop
// samples. Counting from the analyzer's last reset, frame k (0-based) is computed
// from the window ENDING at sample (k+1)*kHop, i.e. the kFftSize samples
//   [(k+1)*kHop - kFftSize, (k+1)*kHop)
// (indices before the reset point read as zero — defined left-edge padding).
//
// GUARANTEE 1 — block-partition independence (proven, bit-exact):
//   Feeding the SAME sample stream from the SAME reset point in ANY chunk sizes
//   yields BIT-IDENTICAL frames. Frame emission keys on a running sample counter,
//   not on chunk boundaries, and every per-frame recurrence runs in frame order,
//   so there is nothing for the host block size to perturb. This is what the
//   live==offline crown test asserts with `==` across ten partitions.
//
// GUARANTEE 2 — cross-path agreement (the real two-path scenario) requires a
// small INTEGRATION CONTRACT, because guarantee 1 only covers identical streams
// from identical reset points, and the live audio thread is NOT reset per clip:
//   (a) Hop-phase alignment: the offline export must start its hop grid at the
//       same phase as the live analyzer at the region's first sample. Easiest:
//       the offline path resets at the region boundary and the live path is the
//       reference only after an equivalent reset — OR the export decodes from a
//       region-start-aligned sample so both grids coincide.
//   (b) Warm-up: the offline path must decode-and-DISCARD ~warmupSamples() of real
//       pre-roll audio before the region, so its frame-ordered recurrences
//       (RMS/peak/onset env, flux mean+var) reach the same steady state the
//       continuous live analyzer is already in. A fresh-from-zero offline pass
//       does NOT match a mid-song live analyzer for the first ~kFluxAvgMs and
//       carries a permanently wrong onsetAge — see the cross-path convergence
//       test. With the pre-roll discarded, the two paths converge within the
//       §4.2 "smoothing tolerance", which is all the contract requires.
// A fresh analyzer also suppresses onset triggers until its window is primed
// (samplesSeen_ >= kFftSize) so the zero-padded startup edge cannot fabricate a
// spurious onset on non-silent content.
//
// Per-frame pipeline (all reading only the chronological window + frame-ordered
// recurrences, never the chunk boundaries):
//   ingest: non-finite samples scrubbed to 0 (one NaN must not poison anything)
//   window -> Hann -> radix-2 FFT -> magnitude -> 64 log bands (block_b_defs.h)
//   instantaneous RMS / peak over the window
//   spectral flux = sum of half-wave-rectified band increases vs previous frame
//   onset: trigger when flux > mean + K*stddev + floor (variance-adaptive, so a
//          stationary noise wash does not strobe), debounced, then exp-decay env
//   one-pole smoothing (seconds domain) of RMS; release smoothing of peak
//
// Dependency-free C++17. No GL/JUCE/FFmpeg. Destined for video-helper/src/.

#pragma once

#include "block_b_defs.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace arbitblockb
{

// One analysis frame. The first four scalars + bands are the shipped Block B
// uniforms (§4.2); flux / frameIndex / sampleIndex are diagnostics for tuning
// and tests (not uniforms).
struct FeatureFrame
{
    float rms      = 0.0f;             // uRMS  — smoothed ~20 ms, 0..1
    float peak     = 0.0f;             // uPeak — instant attack / ~80 ms release, 0..1
    float onset    = 0.0f;             // uOnset — 1.0 at onset, exp decay ~150 ms
    float onsetAge = kOnsetAgeMax;     // uOnsetAge — seconds since last onset
    std::array<float, kBandCount> bands{}; // uAudioBands — 64 log bands, 0..1

    float   flux = 0.0f;               // raw spectral flux this frame (diagnostic)
    int64_t frameIndex  = 0;           // 0-based frame counter since reset (diagnostic)
    int64_t sampleIndex = 0;           // window-END sample position = (frameIndex+1)*kHop (diagnostic)
};

class BlockBAnalyzer
{
public:
    explicit BlockBAnalyzer (float sampleRate = 44100.0f)
        : hann_ (makeHannWindow (kFftSize)),
          history_ (static_cast<size_t> (kFftSize), 0.0f)
    {
        setSampleRate (sampleRate);
        reset();
    }

    // Guards degenerate rates: a non-positive sample rate would make hopSeconds_
    // non-finite and binHz=0, poisoning every downstream time constant.
    void setSampleRate (float sr) noexcept
    {
        if (sr <= 0.0f) sr = 44100.0f;
        sampleRate_ = sr;
        hopSeconds_ = static_cast<float> (kHop) / sr;
    }

    float sampleRate() const noexcept { return sampleRate_; }

    // Clear all state. History zeroed so the left edge is defined padding.
    void reset() noexcept
    {
        std::fill (history_.begin(), history_.end(), 0.0f);
        writePos_ = 0;
        pendingSinceFrame_ = 0;
        samplesSeen_ = 0;
        frameCount_ = 0;
        rms_ = 0.0f;
        peak_ = 0.0f;
        onsetEnv_ = 0.0f;
        onsetAge_ = kOnsetAgeMax;
        fluxAvg_ = 0.0f;
        fluxVar_ = 0.0f;
        secsSinceOnset_ = kOnsetAgeMax;
        prevBands_.fill (0.0f);
        haverPrevBands_ = false;
    }

    // Feed mono samples. Emits a frame into `out` for each hop boundary crossed.
    // Frames are appended; `out` is not cleared. Partition-independent: the same
    // total sample sequence yields the same frames no matter how it is chunked.
    void pushSamples (const float* data, int numSamples, std::vector<FeatureFrame>& out)
    {
        for (int i = 0; i < numSamples; ++i)
        {
            // Scrub non-finite inputs to 0 at ingestion. A single NaN reaching a
            // frame-ordered recurrence (rms_/peak_/fluxAvg_) would stick forever
            // (NaN propagates through every later frame); an Inf turns FFT
            // butterflies into NaN (Inf-Inf). One scrub here keeps both the band
            // path and the scalar path finite, and keeps frames reproducible.
            const float s = std::isfinite (data[i]) ? data[i] : 0.0f;
            history_[static_cast<size_t> (writePos_)] = s;
            writePos_ = (writePos_ + 1) % kFftSize;
            ++samplesSeen_;
            ++pendingSinceFrame_;

            if (pendingSinceFrame_ >= kHop)
            {
                pendingSinceFrame_ -= kHop;
                out.push_back (computeFrame());
            }
        }
    }

    // Convenience: offline single-pass over a whole buffer.
    std::vector<FeatureFrame> analyzeOffline (const float* data, int numSamples)
    {
        std::vector<FeatureFrame> frames;
        pushSamples (data, numSamples, frames);
        return frames;
    }

    int64_t framesEmitted() const noexcept { return frameCount_; }

private:
    // Compute one frame from the current history window (the most recent kFftSize
    // samples, in chronological order). Advances all frame-ordered state.
    FeatureFrame computeFrame()
    {
        // --- Copy the window into a contiguous scratch buffer, oldest-first. The
        //     oldest of the last kFftSize samples sits at writePos_ (the buffer is
        //     exactly kFftSize long and full). This read order depends only on the
        //     global sample stream, never on chunking.
        std::vector<float> re (static_cast<size_t> (kFftSize));
        std::vector<float> im (static_cast<size_t> (kFftSize), 0.0f);

        float sumSq = 0.0f;
        float pk = 0.0f;
        for (int n = 0; n < kFftSize; ++n)
        {
            const float s = history_[static_cast<size_t> ((writePos_ + n) % kFftSize)];
            sumSq += s * s;
            const float a = s < 0.0f ? -s : s;
            if (a > pk) pk = a;
            re[static_cast<size_t> (n)] = s * hann_[static_cast<size_t> (n)];
        }

        // --- Spectrum -> magnitude -> 64 log bands.
        fftRadix2 (re, im);
        std::vector<float> mag (static_cast<size_t> (kFftSize / 2 + 1));
        for (int k = 0; k <= kFftSize / 2; ++k)
            mag[static_cast<size_t> (k)] = std::sqrt (re[static_cast<size_t> (k)] * re[static_cast<size_t> (k)]
                                                    + im[static_cast<size_t> (k)] * im[static_cast<size_t> (k)]);

        std::array<float, kBandCount> bands{};
        integrateBands (mag, sampleRate_, bands);

        // --- Spectral flux: half-wave-rectified sum of band increases vs the
        //     previous frame's bands. First frame has no predecessor -> flux 0.
        float flux = 0.0f;
        if (haverPrevBands_)
        {
            for (int b = 0; b < kBandCount; ++b)
            {
                const float d = bands[static_cast<size_t> (b)] - prevBands_[static_cast<size_t> (b)];
                if (d > 0.0f) flux += d;
            }
        }
        prevBands_ = bands;
        haverPrevBands_ = true;

        // --- Onset: variance-adaptive threshold over a running flux mean+stddev,
        //     with debounce. A single-frame spike on STATIONARY broadband content
        //     (cymbal/hat sizzle, applause, distortion wash) has high frame-to-frame
        //     flux variance; a plain mean*factor gate strobes on it. Gating on
        //     mean + K*stddev scales the bar with the content's own noise floor, so
        //     only genuine outliers (real transients) fire.
        secsSinceOnset_ = std::min (secsSinceOnset_ + hopSeconds_, kOnsetAgeMax);
        const float fluxStd = std::sqrt (std::max (0.0f, fluxVar_));
        const float threshold = fluxAvg_ + kOnsetThreshK * fluxStd + kOnsetThreshFloor;
        const bool debounceOk = secsSinceOnset_ >= kOnsetMinIntervalMs * 0.001f;
        // Suppress triggers until the window is fully primed: a fresh analyzer's
        // zero-padded left edge produces a large flux jump as signal first fills
        // the window, which would fabricate an onset on non-silent content that the
        // continuous live analyzer never sees. samplesSeen_ is absolute => still
        // partition-independent.
        const bool primed = samplesSeen_ >= static_cast<int64_t> (kFftSize);
        bool triggered = false;
        if (flux > threshold && debounceOk && primed)
        {
            triggered = true;
            secsSinceOnset_ = 0.0f;
        }
        // Update running mean+variance AFTER thresholding (so the spike that triggers does
        // not raise its own threshold first). Both are frame-ordered one-poles over the
        // same kFluxAvgMs window -> identity-safe. Variance uses the pre-update mean
        // (EWMA-of-squared-deviation), the standard one-pole variance estimator.
        const float dev = flux - fluxAvg_;
        fluxVar_ = onePoleSeconds (fluxVar_, dev * dev, hopSeconds_, kFluxAvgMs * 0.001f);
        fluxAvg_ = onePoleSeconds (fluxAvg_, flux, hopSeconds_, kFluxAvgMs * 0.001f);

        // Onset envelope: snap to 1.0 on trigger, else exp decay (e-folding kOnsetDecayMs).
        if (triggered) onsetEnv_ = 1.0f;
        else           onsetEnv_ *= std::exp (-hopSeconds_ / (kOnsetDecayMs * 0.001f));
        onsetAge_ = secsSinceOnset_;

        // --- RMS / peak smoothing (seconds domain).
        const float instRms = std::sqrt (sumSq / static_cast<float> (kFftSize));
        rms_ = onePoleSeconds (rms_, clamp01 (instRms), hopSeconds_, kRmsSmoothMs * 0.001f);
        const float relCoef = std::exp (-hopSeconds_ / (kPeakReleaseMs * 0.001f));
        peak_ = std::max (clamp01 (pk), peak_ * relCoef); // instant attack, exp release

        FeatureFrame f;
        f.rms = rms_;
        f.peak = peak_;
        f.onset = onsetEnv_;
        f.onsetAge = onsetAge_;
        f.bands = bands;
        f.flux = flux;
        f.frameIndex = frameCount_;
        f.sampleIndex = (frameCount_ + 1) * static_cast<int64_t> (kHop); // window-END sample position
        ++frameCount_;
        return f;
    }

    float sampleRate_ = 44100.0f;
    float hopSeconds_ = static_cast<float> (kHop) / 44100.0f;
    std::vector<float> hann_;

    // Circular history of the last kFftSize samples.
    std::vector<float> history_;
    int     writePos_ = 0;
    int     pendingSinceFrame_ = 0;
    int64_t samplesSeen_ = 0;   // 64-bit: long is 32-bit on LLP64 (Windows)
    int64_t frameCount_ = 0;

    // Frame-ordered feature state.
    float rms_ = 0.0f;
    float peak_ = 0.0f;
    float onsetEnv_ = 0.0f;
    float onsetAge_ = kOnsetAgeMax;
    float fluxAvg_ = 0.0f;
    float fluxVar_ = 0.0f;
    float secsSinceOnset_ = kOnsetAgeMax;
    std::array<float, kBandCount> prevBands_{};
    bool  haverPrevBands_ = false;
};

} // namespace arbitblockb
