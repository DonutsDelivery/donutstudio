#pragma once

// Tells a kernel compile from a stall. A cache miss makes the kernel compile for over a
// minute, once per kernel text and driver. It runs on its own thread (kernel_build_gate.h),
// and a preview holds its last frame meanwhile. An export blocks on it inside the viewport's
// step or its own: the device marks that wait here, and the viewport subtracts it from the
// step's age. What remains is the step's own work, and it is judged as any other step is.
// So a step stuck anywhere else is still a stall, during a compile or after one.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>

namespace videohelper::pathtrace::kernelcompile
{

struct Reading
{
    std::int64_t activeStartNs = 0; // 0 when no compile is running
    std::int64_t lastStartNs = 0;
    std::int64_t lastEndNs = 0;
};

enum class StepVerdict
{
    Stalled,   // the step's own work passed the report age
    Compiling, // a compile accounts for the step's age and is still running
    Compiled   // a finished compile accounts for it, and the step is back at work
};

// A compile has no time limit: it runs on one CPU core for as long as that machine needs.
// stepEnterNs and nowNs share Reading's clock, std::chrono::steady_clock.
constexpr StepVerdict judgeStep (std::int64_t stepEnterNs, std::int64_t nowNs, const Reading& compile,
                                 double reportMs) noexcept
{
    const auto overlapNs = [stepEnterNs] (std::int64_t start, std::int64_t end) -> std::int64_t
    {
        const std::int64_t from = start > stepEnterNs ? start : stepEnterNs;
        return end > from ? end - from : 0;
    };
    const bool active = compile.activeStartNs != 0;
    const std::int64_t spentNs = (active ? overlapNs (compile.activeStartNs, nowNs) : 0)
                                 + overlapNs (compile.lastStartNs, compile.lastEndNs);
    if (static_cast<double> (nowNs - stepEnterNs - spentNs) * 1e-6 >= reportMs)
        return StepVerdict::Stalled;
    return active ? StepVerdict::Compiling : StepVerdict::Compiled;
}

class State
{
public:
    void begin (std::int64_t nowNs)
    {
        const std::lock_guard<std::mutex> lock (mutex_);
        if (running_++ == 0)
            reading_.activeStartNs = nowNs;
    }
    void end (std::int64_t nowNs)
    {
        const std::lock_guard<std::mutex> lock (mutex_);
        if (--running_ != 0)
            return;
        reading_.lastStartNs = reading_.activeStartNs;
        reading_.lastEndNs = nowNs;
        reading_.activeStartNs = 0;
    }
    Reading read() const
    {
        const std::lock_guard<std::mutex> lock (mutex_);
        return reading_;
    }

private:
    mutable std::mutex mutex_;
    int running_ = 0; // an export and the preview may compile at once
    Reading reading_;
};

inline std::int64_t steadyNowNs() noexcept
{
    return std::chrono::duration_cast<std::chrono::nanoseconds> (
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline State& processState() noexcept
{
    static State state;
    return state;
}

// A preview is holding its frame for the compile. The notice reads this: a compile from
// helper startup that nothing waits for tells the user nothing.
inline std::atomic<bool>& previewWaiting() noexcept
{
    static std::atomic<bool> waiting { false };
    return waiting;
}

// The user cancelled the compile, so a preview holds its frame and says so.
inline std::atomic<bool>& previewStopped() noexcept
{
    static std::atomic<bool> stopped { false };
    return stopped;
}

// An export is blocked on the compile, so Cancel can't be offered.
inline std::atomic<int>& exportsWaiting() noexcept
{
    static std::atomic<int> waiting { 0 };
    return waiting;
}

// Marks a wait on the kernel compile for as long as it lives.
class Scope
{
public:
    Scope() { processState().begin (steadyNowNs()); }
    ~Scope() { processState().end (steadyNowNs()); }
    Scope (const Scope&) = delete;
    Scope& operator= (const Scope&) = delete;
};

} // namespace videohelper::pathtrace::kernelcompile
