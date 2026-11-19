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
#include <cstdlib>
#include <cstring>

namespace loopstep {

struct StageTotals;

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
    // Null leaves the per-label totals off.
    StageTotals* stages = nullptr;
};

// Each scope's duration summed by label while a sink holds this. Scopes nest,
// so a label's total includes the scopes inside it. The slots are fixed: a label
// past the last one is added to the last slot, renamed "other". Labels are
// string literals, compared by text because one literal may have several copies.
struct StageTotals
{
    static constexpr int kSlots = 48;
    const char* labels[kSlots] {};
    std::int64_t ns[kSlots] {};
    int count = 0;

    void clear() noexcept { count = 0; }

    void add (const char* label, std::int64_t elapsedNs) noexcept
    {
        for (int i = 0; i < count; ++i)
            if (labels[i] == label || std::strcmp (labels[i], label) == 0)
            {
                ns[i] += elapsedNs;
                return;
            }
        if (count == kSlots)
        {
            labels[kSlots - 1] = "other";
            ns[kSlots - 1] += elapsedNs;
            return;
        }
        labels[count] = label;
        ns[count] = elapsedNs;
        ++count;
    }
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

// Adds a duration measured some other way, such as a Path Trace frame phase, to
// the current thread's totals. The label must outlive the totals' reader.
inline void addStage (const char* label, std::int64_t elapsedNs) noexcept
{
    if (currentSink != nullptr && currentSink->stages != nullptr)
        currentSink->stages->add (label, elapsedNs);
}

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
        if (sink_ != nullptr && sink_->stages != nullptr)
            sink_->stages->add (label_, elapsedNs);
        if (elapsedNs >= reportNs())
        {
            std::fprintf (stderr, "[viewport] step %s took %lld ms\n", label_,
                          static_cast<long long> (elapsedNs / 1000000));
            recordSlow (sink_, label_, elapsedNs / 1000000);
        }
    }

    Scope (const Scope&) = delete;
    Scope& operator= (const Scope&) = delete;

    static constexpr std::int64_t kReportNs = 200000000; // 200 ms

    // ARBIT_LOOP_STEP_MS overrides the report threshold for a diagnostic run;
    // the release default stays kReportNs. The value is read once, and the
    // first read names the threshold in the log so a run's receipt says which
    // line set it.
    static std::int64_t reportNs() noexcept
    {
        static const std::int64_t value = []
        {
            std::int64_t result = kReportNs;
            if (const char* text = std::getenv ("ARBIT_LOOP_STEP_MS"))
            {
                const auto ms = std::strtoll (text, nullptr, 10);
                if (ms > 0 && ms <= 60000)
                    result = ms * 1000000;
            }
            if (result != kReportNs)
                std::fprintf (stderr, "[viewport] loop step report=%lld ms\n",
                              static_cast<long long> (result / 1000000));
            return result;
        }();
        return value;
    }

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
