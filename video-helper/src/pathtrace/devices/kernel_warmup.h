#pragma once

// Compiles the Path Trace kernel in the background after an update, so a user who picks
// Path Trace later finds it done. The driver's cache makes a repeat compile take 0.1 s, and a
// miss takes over a minute on one core, once per kernel text and driver (kernel_build_gate.h).
//
// The host says go once its window has painted and the app is idle. The warm-up then waits
// for the machine to be quiet, and builds on the gate's own thread at the lowest priority. It
// holds at its next task while the transport plays or an export or an offline render runs,
// and goes on after. A user who picks Path Trace, or an export that needs the kernel, joins
// the running build through the gate: nothing compiles twice. A warm-up that fails says
// nothing to the user and leaves the gate idle, so the normal path builds and reports as if
// the warm-up had never run. No progress fraction exists: the driver reports none.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "kernel_build_gate.h"
#include "kernel_compile_state.h"

#if defined(__linux__)
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace videohelper::pathtrace::kernelcompile
{

enum class WarmupState { Idle, Running, Paused, Done, Failed };

constexpr const char* warmupStateName (WarmupState state) noexcept
{
    switch (state)
    {
        case WarmupState::Idle: return "idle";
        case WarmupState::Running: return "running";
        case WarmupState::Paused: return "paused";
        case WarmupState::Done: return "done";
        case WarmupState::Failed: return "failed";
    }
    return "idle";
}

// What keeps the warm-up out of the way. The host forwards the transport, as the helper may
// have no viewport to read it from. The helper adds its own exports and offline renders.
struct Activity
{
    std::atomic<bool> transportPlaying { false };
    std::function<bool()> helperBusy; // set once at startup, before any warm-up is armed

    bool busy() const { return transportPlaying.load() || (helperBusy && helperBusy()); }
};

inline Activity& activity() noexcept
{
    static Activity state;
    return state;
}

// The calling thread's priority, lowered for the warm-up. Linux can't give a thread its
// priority back without privilege (RLIMIT_NICE is 0 by default), so a joined build there
// finishes at the low priority; it still gets every spare core. Windows restores it.
class ThreadPriority
{
public:
    // True when the priority was lowered.
    bool lowerCurrent()
    {
#if defined(__linux__)
        thread_ = static_cast<int> (::syscall (SYS_gettid));
        return ::setpriority (PRIO_PROCESS, static_cast<id_t> (thread_), 19) == 0;
#elif defined(_WIN32)
        const bool duplicated = DuplicateHandle (GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &thread_,
                                                 THREAD_SET_INFORMATION, FALSE, 0) != 0;
        return duplicated && SetThreadPriority (thread_, THREAD_PRIORITY_IDLE) != 0;
#else
        return false;
#endif
    }

    // True when the thread is back at normal priority. Any thread may call it.
    bool restore()
    {
#if defined(__linux__)
        return thread_ != 0 && ::setpriority (PRIO_PROCESS, static_cast<id_t> (thread_), 0) == 0;
#elif defined(_WIN32)
        return thread_ != nullptr && SetThreadPriority (thread_, THREAD_PRIORITY_NORMAL) != 0;
#else
        return false;
#endif
    }

private:
#if defined(__linux__)
    int thread_ = 0;
#elif defined(_WIN32)
    HANDLE thread_ = nullptr;
#endif
};

struct WarmupHooks
{
    // The driver's cache already holds this kernel, so a build is the 0.1 s kind.
    std::function<bool()> moduleHashPresent;
    // Why a warm-up can't help on this machine ("" when it can): no OptiX, or the cache is off.
    std::function<std::string()> skipReason;
    // The build, with a yield() it calls at each task and a keep() it calls once built. A
    // false keep() means nobody wants the context: the build releases all it made and returns
    // true, as the warm-up's point was the driver's cache. Runs on the gate's thread.
    std::function<bool (std::string&, const std::function<void()>& yield, const std::function<bool()>& keep)> build;
    // True when the compile should stop, for a Cancel or the process exit.
    std::function<bool()> cancelled;
    // Priority of the build's thread, so tests don't touch the real one.
    std::function<bool()> lowerPriority;
    std::function<bool()> restorePriority;
};

struct WarmupReading
{
    WarmupState state = WarmupState::Idle;
    std::string reason = "not asked"; // why idle, or what holds a pause
    bool armed = false;
    bool moduleHashPresent = false;
    bool joined = false;              // a caller claimed the running build
    bool contextReleased = false;     // done unclaimed: the GPU context and pipeline were freed
    bool busy = false;
    bool lowPriority = false;         // the build thread still runs at the lowered priority
    int elapsedSeconds = 0;           // of a running build, from its start
    std::string error;                // of a failed warm-up, for the log
};

class Warmup
{
public:
    Warmup (BuildGate& gate, WarmupHooks hooks, std::int64_t settleNs = 3'000'000'000LL)
        : gate_ (gate), hooks_ (std::move (hooks)), settleNs_ (settleNs) {}

    Warmup (const Warmup&) = delete;
    Warmup& operator= (const Warmup&) = delete;

    ~Warmup()
    {
        {
            const std::lock_guard<std::mutex> lock (mutex_);
            quit_ = true;
        }
        wake_.notify_all();
        if (ticker_.joinable())
            ticker_.join();
    }

    // The host's go. Safe to repeat: a later go after a failure tries again.
    void arm()
    {
        const std::lock_guard<std::mutex> lock (mutex_);
        skipReason_ = hooks_.skipReason ? hooks_.skipReason() : std::string();
        hashPresent_ = skipReason_.empty() && hooks_.moduleHashPresent && hooks_.moduleHashPresent();
        armed_ = true;
        attempted_ = false;
        idleSinceNs_ = 0;
        if (! ticker_.joinable())
            ticker_ = std::thread ([this] { runTicker(); });
        wake_.notify_all();
    }

    // One look at whether to start. The ticker calls it; nowNs is steadyNowNs()'s clock.
    // True when it started the build.
    bool tick (std::int64_t nowNs)
    {
        {
            const std::lock_guard<std::mutex> lock (mutex_);
            if (! armed_ || attempted_ || hashPresent_ || ! skipReason_.empty())
                return false;
            if (gate_.status() != BuildGate::Status::Idle || gate_.stopped())
                return false;
            if (activity().busy())
            {
                idleSinceNs_ = 0;
                return false;
            }
            if (idleSinceNs_ == 0)
                idleSinceNs_ = nowNs;
            if (nowNs - idleSinceNs_ < settleNs_)
                return false;
            attempted_ = true;
        }
        const auto priority = std::make_shared<ThreadPriority>();
        lowered_ = false;
        gate_.start (
            [this, priority] (std::string& error)
            {
                lowered_ = hooks_.lowerPriority ? hooks_.lowerPriority() : priority->lowerCurrent();
                return hooks_.build (error, [this] { yield(); }, [this] { return gate_.keepWarmUp(); });
            },
            BuildGate::Origin::WarmUp,
            [this, priority]
            {
                // A user is waiting on it now.
                const bool back = hooks_.restorePriority ? hooks_.restorePriority() : priority->restore();
                if (back)
                    lowered_ = false;
                wake_.notify_all();
            });
        return true;
    }

    // Called by the build at each task: holds while the machine is busy with the user's own
    // work, unless a caller is waiting on the build or it was cancelled.
    void yield()
    {
        if (gate_.claimed() || ! activity().busy())
            return;
        paused_ = true;
        while (! gate_.claimed() && activity().busy() && ! (hooks_.cancelled && hooks_.cancelled()))
        {
            std::unique_lock<std::mutex> lock (mutex_);
            wake_.wait_for (lock, std::chrono::milliseconds (100));
        }
        paused_ = false;
    }

    WarmupReading read() const
    {
        WarmupReading reading;
        std::string silent;
        const bool failed = gate_.silentFailure (silent);
        {
            const std::lock_guard<std::mutex> lock (mutex_);
            reading.armed = armed_;
            reading.moduleHashPresent = hashPresent_;
        }
        reading.joined = gate_.claimed();
        reading.contextReleased = gate_.released();
        reading.busy = activity().busy();
        reading.lowPriority = lowered_;
        switch (gate_.status())
        {
            case BuildGate::Status::Ready:
                reading.state = WarmupState::Done;
                reading.reason = "kernel ready";
                break;
            case BuildGate::Status::Running:
                reading.state = paused_ ? WarmupState::Paused : WarmupState::Running;
                reading.reason = paused_ ? "playing, exporting or rendering" : reading.joined ? "joined by the viewport" : "compiling";
                reading.elapsedSeconds = buildElapsedSeconds (runningBuildStartNs().load(), steadyNowNs());
                break;
            default:
                if (failed)
                {
                    reading.state = WarmupState::Failed;
                    reading.reason = "failed, the normal path builds again";
                    reading.error = silent;
                }
                else if (reading.contextReleased)
                {
                    reading.state = WarmupState::Done;
                    reading.reason = "kernel is in the driver cache, GPU context released";
                    reading.moduleHashPresent = hooks_.moduleHashPresent && hooks_.moduleHashPresent();
                }
                else
                {
                    const std::lock_guard<std::mutex> lock (mutex_);
                    if (! skipReason_.empty())
                        reading.reason = skipReason_;
                    else if (hashPresent_)
                        reading.reason = "kernel already in the driver cache";
                    else if (armed_)
                        reading.reason = activity().busy() ? "waiting for playback, export or render to end" : "waiting for a quiet machine";
                }
                break;
        }
        return reading;
    }

private:
    void runTicker()
    {
        std::unique_lock<std::mutex> lock (mutex_);
        while (! quit_)
        {
            wake_.wait_for (lock, std::chrono::milliseconds (250));
            if (quit_)
                break;
            lock.unlock();
            tick (steadyNowNs());
            lock.lock();
        }
    }

    BuildGate& gate_;
    WarmupHooks hooks_;
    const std::int64_t settleNs_;
    mutable std::mutex mutex_;
    std::condition_variable wake_;
    std::thread ticker_;
    bool quit_ = false;
    bool armed_ = false;
    bool attempted_ = false;
    bool hashPresent_ = false;
    std::string skipReason_;
    std::int64_t idleSinceNs_ = 0;
    std::atomic<bool> paused_ { false };
    std::atomic<bool> lowered_ { false };
};

} // namespace videohelper::pathtrace::kernelcompile
