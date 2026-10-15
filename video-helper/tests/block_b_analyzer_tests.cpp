// block_b_analyzer_tests.cpp: deterministic checks of the Block B analyzer
// (block_b_analyzer.h, constants in block_b_defs.h). Each check feeds a
// synthetic buffer at 48 kHz and compares the output with a value derived from
// the constants. The derivation sits next to each check.
//
// Shared facts (h = kHop / 48000 s = 0.0106667 s, N = kFftSize = 2048):
//   - Frame k is computed from the window [(k+1)*kHop - N, (k+1)*kHop). Indices
//     before sample 0 read as zero.
//   - One-pole smoothing with time constant tau, against a constant target,
//     multiplies the residual by exp(-h/tau) each frame. So residual(m) =
//     rho * residual(m-1) with rho = exp(-h/tau).
//   - Peak release uses that recurrence with tau = kPeakReleaseMs. The onset
//     envelope uses tau = kOnsetDecayMs, after a trigger sets it to 1.
//   - Symmetric Hann window: the sum of w[n] over n = 0..N-1 is N/2 - 1/2 = 1023.5.
//   - A unit impulse at window index n has flat FFT magnitude w[n] in every
//     bin. Each band therefore reads 9 * w[n] / N. The frame flux, a sum of 64
//     rises, is 64 times that value when the weight rises from zero.
//   - A sine of period 64 samples lands on FFT bin N/64 = 32, which is 750 Hz at
//     48 kHz. Its magnitude is (A/2) * 1023.5 at bin 32 and about (A/2) * 512 at
//     bins 31 and 33.

#include "block_b_analyzer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <vector>

namespace
{
using arbitblockb::BlockBAnalyzer;
using arbitblockb::FeatureFrame;

constexpr float kRate = 48000.0f;
const double kHopSec = static_cast<double> (arbitblockb::kHop) / static_cast<double> (kRate);

int failures = 0;

void check (bool condition, const char* message)
{
    if (! condition)
    {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void near (double actual, double expected, double tolerance, const char* message)
{
    check (std::abs (actual - expected) <= tolerance, message);
}

// exp(-h/tau) with tau in milliseconds: the per-frame residual factor.
double residualFactor (float tauMs)
{
    return std::exp (-kHopSec / (static_cast<double> (tauMs) * 0.001));
}

// Feeds the buffer in 333-sample chunks (not a divisor of kHop) and returns all
// emitted frames. Partition independence is the analyzer's own guarantee.
std::vector<FeatureFrame> run (const std::vector<float>& x)
{
    BlockBAnalyzer analyzer (kRate);
    std::vector<FeatureFrame> frames;
    constexpr size_t chunk = 333;
    for (size_t i = 0; i < x.size(); i += chunk)
    {
        const size_t n = std::min (chunk, x.size() - i);
        analyzer.pushSamples (x.data() + i, static_cast<int> (n), frames);
    }
    return frames;
}

// Sine with a period of 64 samples, indexed from absolute sample `from`. 64
// divides kHop and kFftSize, so windows one hop apart are bit-identical once
// they are full of tone.
std::vector<float> tone (size_t from, size_t count, float amplitude)
{
    std::vector<float> out (count);
    for (size_t i = 0; i < count; ++i)
    {
        const double phase = 2.0 * arbitblockb::kPi * static_cast<double> ((from + i) % 64) / 64.0;
        out[i] = static_cast<float> (amplitude * std::sin (phase));
    }
    return out;
}

std::vector<float> silence (size_t count)
{
    return std::vector<float> (count, 0.0f);
}

// True only if every field is exactly zero. A subnormal value is not equal to
// zero, so this also rules out denormal residue.
bool isSilent (const FeatureFrame& f)
{
    if (f.rms != 0.0f || f.peak != 0.0f || f.onset != 0.0f || f.flux != 0.0f)
        return false;
    if (f.onsetAge != arbitblockb::kOnsetAgeMax)
        return false;
    for (float b : f.bands)
        if (b != 0.0f)
            return false;
    return true;
}

bool finiteAndInRange (const FeatureFrame& f)
{
    bool ok = std::isfinite (f.rms) && std::isfinite (f.peak) && std::isfinite (f.onset)
           && std::isfinite (f.onsetAge) && std::isfinite (f.flux);
    ok = ok && f.rms >= 0.0f && f.rms <= 1.0f && f.peak >= 0.0f && f.peak <= 1.0f;
    for (float b : f.bands)
        ok = ok && std::isfinite (b) && b >= 0.0f && b <= 1.0f;
    return ok;
}
}

int main()
{
    // --- Silence: exact zeros, onsetAge stays at its clamp.
    {
        const auto frames = run (silence (16384));
        bool ok = ! frames.empty();
        for (const auto& f : frames)
            ok = ok && isSilent (f);
        check (ok, "silence: every feature is exactly zero and onsetAge stays at kOnsetAgeMax");
    }

    // --- NaN and Inf guard. Ingestion maps non-finite samples to zero.
    {
        const std::vector<float> nanOnly (16384, std::numeric_limits<float>::quiet_NaN());
        const auto frames = run (nanOnly);
        bool ok = ! frames.empty();
        for (const auto& f : frames)
            ok = ok && isSilent (f);
        check (ok, "all-NaN input produces the same exact silence as zero input");

        auto mixed = tone (0, 16384, 0.5f);
        mixed[3000] = std::numeric_limits<float>::quiet_NaN();
        mixed[5000] = std::numeric_limits<float>::infinity();
        mixed[7000] = -std::numeric_limits<float>::infinity();
        const auto mixedFrames = run (mixed);
        bool finite = ! mixedFrames.empty();
        for (const auto& f : mixedFrames)
            finite = finite && finiteAndInRange (f);
        check (finite, "NaN and Inf samples leave every frame finite and within 0..1");
    }

    // --- RMS. A tone of amplitude A has window RMS A/sqrt(2) once the window is
    //     full (32 whole cycles in 2048 samples). The smoother then obeys
    //     rms(m) - T = rho * (rms(m-1) - T), with rho = exp(-h/kRmsSmoothMs).
    {
        const double T = 0.5 / std::sqrt (2.0);
        const auto frames = run (tone (0, 46080, 0.5f));
        const double rho = residualFactor (arbitblockb::kRmsSmoothMs);
        for (size_t m = 3; m <= 20; ++m)
        {
            const double residual = static_cast<double> (frames[m].rms) - T;
            const double previous = static_cast<double> (frames[m - 1].rms) - T;
            near (residual, rho * previous, 2e-6, "rms residual shrinks by exp(-h/tau_rms) per frame");
        }
        near (frames[89].rms, T, 1e-5, "rms converges to A/sqrt(2) for a steady tone");

        // Step from A1 = 0.5 to A2 = 0.25 at sample 20480. Frame 43 is the first
        // whose window [20480, 22528) lies wholly in the new tone.
        std::vector<float> step = tone (0, 20480, 0.5f);
        const auto second = tone (20480, 25600, 0.25f);
        step.insert (step.end(), second.begin(), second.end());
        const auto stepFrames = run (step);
        const double T2 = 0.25 / std::sqrt (2.0);
        for (size_t m = 44; m <= 60; ++m)
        {
            const double residual = static_cast<double> (stepFrames[m].rms) - T2;
            const double previous = static_cast<double> (stepFrames[m - 1].rms) - T2;
            near (residual, rho * previous, 2e-6, "rms residual after a step shrinks by exp(-h/tau_rms)");
        }
        near (stepFrames[80].rms, T2, 1e-5, "rms settles on the new level after a step");
    }

    // --- Peak. Instant attack, then release. A unit sample at index 0 at 0.8 is
    //     in the window for frames 0..3. From frame 4 the window no longer holds
    //     it, so peak(m) = 0.8 * rho^(m-3) with rho = exp(-h/kPeakReleaseMs).
    //     A 1.0 sample at index 10240 enters at frame 20 and resets the peak.
    {
        auto x = silence (12288);
        x[0] = 0.8f;
        x[10240] = 1.0f;
        const auto frames = run (x);
        const double rho = residualFactor (arbitblockb::kPeakReleaseMs);
        for (size_t m = 0; m <= 3; ++m)
            near (frames[m].peak, 0.8, 1e-6, "peak holds the impulse level while it is in the window");
        for (size_t m = 4; m <= 19; ++m)
            near (frames[m].peak, 0.8 * std::pow (rho, static_cast<double> (m - 3)), 1e-6,
                  "peak releases by exp(-h/tau_peak) per frame");
        near (frames[20].peak, 1.0, 1e-6, "peak attacks instantly to a louder impulse");
    }

    // --- Onset, one transient, with the debounce suppressing a second rise.
    //     A unit impulse at sample 4096 enters frame 8 with Hann weight
    //     w(1536) = 0.49885, giving flux 64 * 9 * 0.49885 / 2048 = 0.1403 and a
    //     trigger (threshold is 0.012 with no history). Frame 9 has weight 1.0,
    //     so its rise is 64 * 9 * (1.0 - 0.49885) / 2048 = 0.1409. The adaptive
    //     threshold there is about 0.101, so frame 9 would fire without the
    //     debounce. Frame 9 is one hop (0.0107 s) after the trigger, which is
    //     below kOnsetMinIntervalMs (50 ms), so it must be suppressed.
    {
        auto x = silence (8192);
        x[4096] = 1.0f;
        const auto frames = run (x);
        size_t triggers = 0;
        for (const auto& f : frames)
            if (f.onset == 1.0f)
                ++triggers;
        check (triggers == 1u && frames[8].onset == 1.0f,
               "one impulse fires exactly one onset, at its entry frame 8");
        check (frames[9].flux > 0.1f,
               "frame 9 carries the second flux rise that the debounce must suppress");
        const double rho = residualFactor (arbitblockb::kOnsetDecayMs);
        for (size_t k = 0; k <= 4; ++k)
            near (frames[8 + k].onsetAge, static_cast<double> (k) * kHopSec, 1e-5,
                  "onsetAge rises by one hop per frame after a trigger");
        for (size_t k = 1; k <= 7; ++k)
            near (frames[8 + k].onset, std::pow (rho, static_cast<double> (k)), 1e-5,
                  "onset envelope decays by exp(-h/tau_onset) per frame");
    }

    // --- Onset re-fires for a transient after the debounce window. The second
    //     impulse at 153600 enters at frame 300, long after the adaptive
    //     threshold has decayed (about 0.97 per frame, from frame 8). Expect
    //     triggers at exactly frames 8 and 300.
    {
        auto x = silence (158720);
        x[4096] = 1.0f;
        x[153600] = 1.0f;
        const auto frames = run (x);
        std::vector<size_t> triggered;
        for (size_t m = 0; m < frames.size(); ++m)
            if (frames[m].onset == 1.0f)
                triggered.push_back (m);
        check (triggered.size() == 2u && triggered[0] == 8u && triggered[1] == 300u,
               "two separated transients fire two onsets, at frames 8 and 300");
    }

    // --- Steady tone: windows one hop apart are identical once full, so flux is
    //     exactly zero from frame 4 on, and no onset fires there.
    {
        const auto frames = run (tone (0, 16384, 0.5f));
        bool ok = true;
        for (size_t m = 4; m < frames.size(); ++m)
            ok = ok && frames[m].onset == 0.0f && frames[m].flux == 0.0f;
        check (ok, "steady tone fires no onset once its windows repeat");
    }

    // --- Band placement. A 750 Hz tone of amplitude 0.5 sits on bin 32. Band b
    //     spans 30 * 600^(b/64) to 30 * 600^((b+1)/64) Hz. Band 32 covers
    //     734.8 to 812.1 Hz, so bins 32..34. Band 31 covers 665.9 to 734.8 Hz,
    //     so bins 29..31. With magnitudes 255.9 at bin 32 and 128 at bins 31 and
    //     33, and each band equal to 9/2048 times the bin mean:
    //       band 32 = 9/2048 * (255.9 + 128 + 0) / 3 = 0.5623
    //       band 31 = 9/2048 * (0 + 0 + 128) / 3 = 0.1875
    //     Bands three or more bins away are leakage only, so they stay well below
    //     0.002. Frame 10 holds a full window of tone.
    {
        const auto frames = run (tone (0, 8192, 0.5f));
        const FeatureFrame& f = frames[10];
        near (f.bands[32], 0.5623, 0.005, "750 Hz tone reads 9/2048 * (255.9+128)/3 in band 32");
        near (f.bands[31], 0.1875, 0.005, "750 Hz tone reads 9/2048 * 128/3 in band 31");
        size_t best = 0;
        for (size_t b = 1; b < f.bands.size(); ++b)
            if (f.bands[b] > f.bands[best])
                best = b;
        check (best == 32u, "the tone's energy peaks in band 32");
        bool farQuiet = true;
        for (size_t b = 0; b < f.bands.size(); ++b)
        {
            const size_t distance = b > 32u ? b - 32u : 32u - b;
            if (distance >= 3u && f.bands[b] >= 0.002f)
                farQuiet = false;
        }
        check (farQuiet, "bands three or more away from 750 Hz stay below 0.002 (leakage only)");
    }

    if (failures == 0)
        std::cout << "block b analyzer: all checks passed\n";
    return failures == 0 ? 0 : 1;
}
