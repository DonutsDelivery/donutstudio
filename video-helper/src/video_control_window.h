#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace videocontrol
{
// Video reducers run once over each complete 48 kHz frame window. This does
// not describe host audio blocks or per-sample video rendering.
inline constexpr int kSignalSampleRate = 48000;
// The window covers the whole project. Frame indices are int, so that is the
// only bound on its length; frame values and sample work follow the project.
inline constexpr int kMaxSignalFrames = std::numeric_limits<int>::max();
inline constexpr std::size_t kMaxSignalFrameValues = std::numeric_limits<std::size_t>::max();
inline constexpr std::int64_t kMaxSignalSampleWork = std::numeric_limits<std::int64_t>::max();

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
            && frameCount <= kMaxSignalFrames;
    }
    std::int64_t sampleStart(std::int64_t frame) const noexcept
    {
        return static_cast<std::int64_t>(std::floor(
            static_cast<double>(frame) * kSignalSampleRate / fps));
    }
};
} // namespace videocontrol
