#pragma once

// Median and p95 of frame times, and the rolling window of the last 120 the
// viewport keeps for Realtime (render_speed_stats.h writes them into replies).
// Standard library only.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <vector>

namespace videorender::speedstats
{

// Median and p95 of a set of frame times. p95 is the nearest-rank percentile, so
// it is always one of the measured times.
struct FrameTimeSummary
{
    std::size_t count = 0;
    double median = 0.0;
    double p95 = 0.0;
};

// Non-finite values are skipped. An empty set returns count 0.
inline FrameTimeSummary summarizeFrameTimes (std::vector<double> times)
{
    times.erase (std::remove_if (times.begin(), times.end(), [] (double value) { return ! std::isfinite (value); }),
                 times.end());
    FrameTimeSummary summary;
    summary.count = times.size();
    if (times.empty())
        return summary;
    std::sort (times.begin(), times.end());
    const std::size_t middle = times.size() / 2;
    summary.median = times.size() % 2 == 1 ? times[middle] : (times[middle - 1] + times[middle]) * 0.5;
    const auto rank = static_cast<std::size_t> (std::ceil (0.95 * static_cast<double> (times.size())));
    summary.p95 = times[std::max<std::size_t> (rank, 1) - 1];
    return summary;
}

// The last kCapacity frame times, oldest overwritten first.
class FrameTimeWindow
{
public:
    static constexpr std::size_t kCapacity = 120;

    void push (double milliseconds) noexcept
    {
        if (! std::isfinite (milliseconds))
            return;
        times_[next_] = milliseconds;
        next_ = (next_ + 1) % kCapacity;
        size_ = std::min (size_ + 1, kCapacity);
    }

    void clear() noexcept { next_ = size_ = 0; }
    std::size_t size() const noexcept { return size_; }

    FrameTimeSummary summary() const
    {
        return summarizeFrameTimes (std::vector<double> (times_.begin(), times_.begin() + static_cast<std::ptrdiff_t> (size_)));
    }

private:
    std::array<double, kCapacity> times_ {};
    std::size_t next_ = 0;
    std::size_t size_ = 0;
};

// What a Realtime frame cost. gpuMs is absent with a reason where the backend
// cannot time the GPU; the window fields exist only for the viewport.
struct RealtimeFrameTimes
{
    std::optional<double> gpuMs;
    const char* gpuAbsent = nullptr;
    std::optional<double> cpuMs;
    std::optional<FrameTimeSummary> window;
};

} // namespace videorender::speedstats
