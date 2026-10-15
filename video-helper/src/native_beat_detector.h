#pragma once

#include "../../shared/ArbitBpmDetector.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

// The legacy helper RPC uses the same owned DSP and constant-tempo phase fit
// as NativeBeatAnalysisBackend. The app's Beat This analysis path is unchanged.
// This is a fixed-grid fallback, not an adaptive tracker for tempo changes.
namespace arbitnativebeat
{
struct Result
{
    double bpm = 0.0;
    double confidence = 0.0;
    std::vector<double> beats;
};

inline std::string analyze (const std::vector<float>& mono, int sampleRate,
                            Result& result)
{
    result = {};
    if (sampleRate <= 0 || mono.size() < static_cast<size_t> (sampleRate))
        return "audio too short for beat detection";
    if (! std::all_of (mono.begin(), mono.end(), [] (float value)
        { return std::isfinite (value); }))
        return "beat detection received non-finite audio";

    constexpr int fftSize = 2048, hopSize = 512;
    arbit::bpm::DetectorOptions options;
    options.fftSize = fftSize;
    options.hopSize = hopSize;
    options.minBpm = 40.0f;
    options.maxBpm = 220.0f;
    options.useLoopDurationPrior = false;
    const auto detection = arbit::bpm::detectBpm (
        mono.data(), mono.size(), 1, sampleRate, arbit::bpm::simpleFft, options);
    if (! std::isfinite (detection.bpm) || detection.bpm <= 0.0f
        || ! std::isfinite (detection.confidence))
        return "native beat detector found no stable tempo";

    const double duration = static_cast<double> (mono.size()) / sampleRate;
    const int maxOnsets = static_cast<int> (std::clamp (
        duration * detection.bpm / 30.0, 32.0, 4096.0));
    const auto onsets = arbit::bpm::computeOnsets (
        mono, sampleRate, arbit::bpm::simpleFft, maxOnsets, fftSize, hopSize);
    if (onsets.onsetTimesSeconds.size() < 4)
        return "native beat detector found too few rhythmic events";

    // computeOnsets timestamps the start of each forward-looking FFT window.
    // Use its center for this RPC's source-time grid; this is a window estimate,
    // not sample-accurate transient localization.
    const double onsetOffset = 0.5 * fftSize / sampleRate;
    const double period = 60.0 / detection.bpm;
    double bestPhase = std::fmod (onsets.onsetTimesSeconds.front() + onsetOffset, period);
    double bestScore = -1.0;
    const size_t stride = std::max<size_t> (1, onsets.onsetTimesSeconds.size() / 256);
    for (size_t candidate = 0; candidate < onsets.onsetTimesSeconds.size(); candidate += stride)
    {
        const double phase = std::fmod (onsets.onsetTimesSeconds[candidate] + onsetOffset, period);
        double score = 0.0;
        for (const double onset : onsets.onsetTimesSeconds)
        {
            const double distance = std::abs (std::remainder (onset + onsetOffset - phase, period));
            score += std::exp (-0.5 * std::pow (distance / (0.12 * period), 2.0));
        }
        if (score > bestScore)
        {
            bestScore = score;
            bestPhase = phase;
        }
    }

    result.bpm = detection.bpm;
    result.confidence = std::clamp (static_cast<double> (detection.confidence), 0.0, 1.0);
    for (double beat = bestPhase; beat <= duration; beat += period)
        result.beats.push_back (beat);
    return {};
}
} // namespace arbitnativebeat
