#include "../src/native_beat_detector.h"
#include "../../plugin/Source/ArbitBpmDetector.h"

#include <iostream>
#include <limits>
#include <stdexcept>

namespace
{
void require (bool condition, const char* message)
{
    if (! condition) throw std::runtime_error (message);
}

std::vector<float> clicks (int rate, double bpm, double duration, double phase)
{
    std::vector<float> audio (static_cast<size_t> (duration * rate), 0.0f);
    for (double beat = phase; beat < duration; beat += 60.0 / bpm)
    {
        const auto start = static_cast<size_t> (beat * rate);
        for (int i = 0; i < rate / 50 && start + i < audio.size(); ++i)
            audio[start + i] += static_cast<float> (
                std::exp (-250.0 * i / rate) * std::cos (2.0 * std::acos (-1.0) * 1000.0 * i / rate));
    }
    return audio;
}

void checkGrid (int rate, double bpm, double phase)
{
    const auto audio = clicks (rate, bpm, 12.0, phase);
    arbitnativebeat::Result result;
    const auto error = arbitnativebeat::analyze (audio, rate, result);
    require (error.empty(), error.c_str());
    require (std::abs (result.bpm - bpm) < 2.0, "click-track BPM outside two BPM");
    require (result.confidence >= 0.0 && result.confidence <= 1.0, "invalid confidence");
    require (result.beats.size() >= 8, "too few beats");
    require (std::abs (std::remainder (result.beats.front() - phase, 60.0 / bpm)) < 0.04,
             "click-track phase error exceeds 40 ms");
    double maxTimingError = 0.0;
    for (size_t i = 0; i < result.beats.size(); ++i)
    {
        require (std::isfinite (result.beats[i]) && result.beats[i] >= 0.0
                 && result.beats[i] <= 12.0, "beat outside source");
        const double timingError = std::abs (std::remainder (result.beats[i] - phase, 60.0 / bpm));
        maxTimingError = std::max (maxTimingError, timingError);
        require (timingError < 0.04, "click-track beat timing error exceeds 40 ms");
        if (i > 0)
            require (std::abs (result.beats[i] - result.beats[i - 1] - 60.0 / result.bpm)
                     < 1.0e-8, "constant-tempo contract changed");
    }
    arbitnativebeat::Result repeated;
    require (arbitnativebeat::analyze (audio, rate, repeated).empty(), "repeat failed");
    require (result.beats == repeated.beats && result.bpm == repeated.bpm,
             "analysis is not deterministic");
    std::cout << "PASS click grid rate=" << rate << " bpm=" << bpm
              << " detected=" << result.bpm << " phase=" << result.beats.front()
              << " beats=" << result.beats.size() << " maxTimingErrorMs=" << 1000.0 * maxTimingError << '\n';
}
}

int main()
{
    try
    {
        arbitnativebeat::Result result { 120.0, 0.9, { 1.0 } };
        require (! arbitnativebeat::analyze ({}, 44100, result).empty(), "empty accepted");
        require (result.bpm == 0.0 && result.beats.empty(), "stale result survived");
        require (! arbitnativebeat::analyze (std::vector<float> (100), 44100, result).empty(),
                 "short audio accepted");
        require (! arbitnativebeat::analyze (std::vector<float> (44100), 0, result).empty(),
                 "invalid rate accepted");
        require (! arbitnativebeat::analyze (std::vector<float> (44100 * 3), 44100, result).empty(),
                 "silence accepted");
        auto invalid = clicks (44100, 120.0, 3.0, 0.25);
        invalid[100] = std::numeric_limits<float>::quiet_NaN();
        require (! arbitnativebeat::analyze (invalid, 44100, result).empty(), "NaN accepted");
        invalid[100] = std::numeric_limits<float>::infinity();
        require (! arbitnativebeat::analyze (invalid, 44100, result).empty(), "infinity accepted");
        std::cout << "PASS invalid input, silence and result reset\n";
        checkGrid (44100, 120.0, 0.25);
        checkGrid (48000, 120.0, 0.36);
        checkGrid (44100, 109.5, 0.31);
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
