#pragma once

// Diagnostic timing for calls that run on the viewport render thread. A Scope
// names the call it is inside. While a sink is installed on the thread, the
// scope also publishes its label into the sink, so the loop-step probe that
// viewport_info reports names the innermost call. On exit the scope restores
// the enclosing label and prints one stderr line when the call took 200 ms or
// more. A sink with a slow record also keeps that label and its duration, so a
// status reader can name the slow call after it has returned. No allocation and
// no locks on the timing path.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>

namespace loopstep {

// The atomics that the viewport's advance sub-step probe reads. Only the thread
// that installed the sink writes them.
struct Sink
{
    std::atomic<const char*>* label = nullptr;
    std::atomic<std::int64_t>* enterNs = nullptr;
    // The last scope that ran for kReportNs or longer, and its duration in
    // milliseconds. Null pointers leave the record off.
    std::atomic<const char*>* slowLabel = nullptr;
    std::atomic<std::int64_t>* slowMs = nullptr;
};

// Stores one slow-call record in the sink, when the sink keeps one. The two
// stores are relaxed, so a reader can briefly pair a new label with an old
// duration. That is acceptable for a diagnostic.
inline void recordSlow (Sink* sink, const char* label, std::int64_t elapsedMs) noexcept
{
    if (sink == nullptr || sink->slowLabel == nullptr || sink->slowMs == nullptr)
        return;
    sink->slowMs->store (elapsedMs, std::memory_order_relaxed);
    sink->slowLabel->store (label, std::memory_order_relaxed);
}

// The sink for the current thread. Null means no sink is installed here, and
// scopes then only time and print.
inline thread_local Sink* currentSink = nullptr;

// Installs a sink for the current thread and restores the previous one on exit.
class SinkInstall final
{
public:
    explicit SinkInstall (Sink* sink) noexcept
        : previous_ (currentSink)
    {
        currentSink = sink;
    }

    ~SinkInstall()
    {
        currentSink = previous_;
    }

    SinkInstall (const SinkInstall&) = delete;
    SinkInstall& operator= (const SinkInstall&) = delete;

private:
    Sink* previous_;
};

// Names the enclosed call. The label must be a string literal or otherwise
// outlive the scope.
class Scope final
{
public:
    explicit Scope (const char* label) noexcept
        : label_ (label),
          sink_ (currentSink),
          start_ (std::chrono::steady_clock::now())
    {
        if (sink_ != nullptr)
        {
            previousLabel_ = sink_->label->load (std::memory_order_relaxed);
            previousEnterNs_ = sink_->enterNs->load (std::memory_order_relaxed);
            sink_->label->store (label_, std::memory_order_relaxed);
            sink_->enterNs->store (toNs (start_), std::memory_order_relaxed);
        }
    }

    ~Scope()
    {
        const auto end = std::chrono::steady_clock::now();
        if (sink_ != nullptr)
        {
            sink_->label->store (previousLabel_, std::memory_order_relaxed);
            sink_->enterNs->store (previousEnterNs_, std::memory_order_relaxed);
        }
        const std::int64_t elapsedNs = toNs (end) - toNs (start_);
        if (elapsedNs >= kReportNs)
        {
            std::fprintf (stderr, "[viewport] step %s took %lld ms\n", label_,
                          static_cast<long long> (elapsedNs / 1000000));
            recordSlow (sink_, label_, elapsedNs / 1000000);
        }
    }

    Scope (const Scope&) = delete;
    Scope& operator= (const Scope&) = delete;

    static constexpr std::int64_t kReportNs = 200000000; // 200 ms

private:

    static std::int64_t toNs (std::chrono::steady_clock::time_point at) noexcept
    {
        return std::chrono::duration_cast<std::chrono::nanoseconds> (
            at.time_since_epoch()).count();
    }

    const char* label_;
    Sink* sink_;
    std::chrono::steady_clock::time_point start_;
    const char* previousLabel_ = nullptr;
    std::int64_t previousEnterNs_ = 0;
};

} // namespace loopstep
