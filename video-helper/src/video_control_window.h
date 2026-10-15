#pragma once

#include <cmath>
#include <cstdint>

namespace videocontrol
{
// Video reducers run once over each complete 48 kHz frame window. This does
// not describe host audio blocks or per-sample video rendering.
inline constexpr int kSignalSampleRate = 48000;
inline constexpr int kMaxSignalFrames = 28801; // 60 seconds at 480 fps, plus the end frame
inline constexpr int kMaxSignalFrameValues = 32768;
inline constexpr double kMaxSignalDurationSeconds = 60.0;
inline constexpr std::int64_t kMaxSignalSampleWork = 8000000;
inline constexpr int kMaxSignalPlanBytes = 2 * 1024 * 1024; // helper request cap is 16 MiB

struct SignalWindow
{
    int sampleRate = 0;
    double fps = 0.0;
    int frameCount = 0;

    bool present() const noexcept { return sampleRate != 0 || fps != 0.0 || frameCount != 0; }
    bool valid() const noexcept
    {
        return sampleRate == kSignalSampleRate && std::isfinite(fps)
            && fps >= 1.0 && fps <= 480.0 && frameCount > 0
            && frameCount <= kMaxSignalFrames
            && frameCount <= static_cast<int>(std::ceil(kMaxSignalDurationSeconds * fps)) + 1;
    }
    std::int64_t sampleStart(std::int64_t frame) const noexcept
    {
        return static_cast<std::int64_t>(std::floor(
            static_cast<double>(frame) * kSignalSampleRate / fps));
    }
};
} // namespace videocontrol
