#pragma once

// One kernel build for the process, on its own thread. Whoever asks first starts it, at
// helper startup or on the first frame. Later callers join that build: a preview polls
// status() and holds its last frame, and an export blocks in await(). Two builds of the
// same kernel would each compile it from nothing, as the driver's cache fills only when
// one ends. It lives for the process, as its thread may outlast any one renderer.
//
// A background warm-up (kernel_warmup.h) starts the build on nobody's behalf. The first
// caller of any other kind to start, join or await it claims it: the build is then theirs
// (a failure reaches them, and the warm-up's low priority is lifted), and it is still the
// one build, with the elapsed time counted from the warm-up's start.
//
// A warm-up only fills the driver's cache. When it ends unclaimed, its build releases what
// it made (keepWarmUp() says no) and the gate goes back to idle, so the next real use builds
// again from the cache in about 0.1 s and holds no GPU context until then.

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>
#include <utility>

#include "kernel_compile_state.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <process.h>
#include <windows.h>
#endif

namespace videohelper::pathtrace::kernelcompile
{

#if defined(_WIN32)
// Windows gives std::thread 1 MiB. The OptiX documentation names no stack size for module
// compiles, and the driver's compiler is LLVM, which recurses. This only reserves address
// space: pages commit as the stack reaches them. Linux already gives 8 MiB by default, and
// OptiX doesn't exist on macOS.
inline constexpr unsigned kBuildThreadStackBytes = 64u * 1024u * 1024u;

template <typename Body>
void startBuildThread (Body body)
{
    auto* owned = new std::function<void()> (std::move (body));
    const auto entry = [] (void* argument) -> unsigned
    {
        std::unique_ptr<std::function<void()>> run (static_cast<std::function<void()>*> (argument));
        (*run)();
        return 0;
    };
    const auto handle = _beginthreadex (nullptr, kBuildThreadStackBytes, entry, owned, STACK_SIZE_PARAM_IS_A_RESERVATION, nullptr);
    if (handle == 0)
    {
        delete owned;
        throw std::system_error (std::make_error_code (std::errc::resource_unavailable_try_again));
    }
    CloseHandle (reinterpret_cast<HANDLE> (handle));
}
#else
template <typename Body>
void startBuildThread (Body body)
{
    std::thread (std::move (body)).detach();
}
#endif

class BuildGate
{
public:
    // Stopped: the user cancelled, and nothing builds until they start again or an export asks.
    enum class Status { Idle, Running, Ready, Failed, Stopped };
    // Who asked for the build. A warm-up that fails unclaimed leaves the gate idle, so the
    // next caller builds from nothing and reports its own failure.
    enum class Origin { Caller, WarmUp };

    // Starts the build unless one has run, is running, was stopped, or waits for a caller
    // to read its failure. A failure read in await() or join() lets the next caller try again.
    // onClaim runs once, on the claiming caller's thread, when a warm-up's build is claimed.
    void start (std::function<bool (std::string&)> build, Origin origin = Origin::Caller,
                std::function<void()> onClaim = {})
    {
        std::function<void()> claimed;
        bool launches = false;
        {
            const std::lock_guard<std::mutex> lock (mutex_);
            if (status_ != Status::Idle)
            {
                if (origin == Origin::Caller)
                    claimed = claimLocked();
            }
            else
            {
                markRunning (origin);
                onClaim_ = std::move (onClaim);
                launches = true;
            }
        }
        if (claimed)
            claimed();
        if (launches)
            launch (std::move (build));
    }

    // Starts the build if nothing runs and blocks until it ends, for an export. A stop
    // doesn't reach a build someone is blocked on, and a build still ending from an earlier
    // stop finishes first, so two never run at once. True when it succeeded.
    bool join (const std::function<bool (std::string&)>& build, std::string& error)
    {
        std::unique_lock<std::mutex> lock (mutex_);
        stopRequested_ = false;
        if (status_ == Status::Stopped)
            status_ = Status::Idle;
        ++joined_;
        const auto leave = [this] { --joined_; };
        if (const auto claimed = claimLocked())
        {
            lock.unlock();
            claimed();
            lock.lock();
        }
        for (;;)
        {
            if (status_ == Status::Idle)
            {
                markRunning (Origin::Caller);
                lock.unlock();
                launch (build);
                lock.lock();
            }
            changed_.wait (lock, [this] { return status_ != Status::Running; });
            if (status_ == Status::Ready)
            {
                leave();
                return true;
            }
            if (status_ == Status::Failed)
            {
                error = error_;
                status_ = Status::Idle;
                leave();
                return false;
            }
        }
    }

    // The user's Cancel. True when the caller should now stop the running compile, or
    // nothing is running to stop. False if the build is done or an export is blocked on it.
    bool stop()
    {
        const std::lock_guard<std::mutex> lock (mutex_);
        if (status_ == Status::Ready || joined_ > 0)
            return false;
        stopRequested_ = true;
        if (status_ == Status::Running)
            stopping_ = true;
        else
            status_ = Status::Stopped;
        return true;
    }

    // The user's Start again. The caller then calls start(). If the stopped build is still
    // ending, that start does nothing and the next poll's start begins the new build.
    void resume()
    {
        const std::lock_guard<std::mutex> lock (mutex_);
        stopRequested_ = false;
        if (status_ == Status::Stopped)
            status_ = Status::Idle;
    }

    bool stopped() const
    {
        const std::lock_guard<std::mutex> lock (mutex_);
        return stopRequested_;
    }

    bool hasWaiters() const
    {
        const std::lock_guard<std::mutex> lock (mutex_);
        return joined_ > 0;
    }

    Status status() const
    {
        const std::lock_guard<std::mutex> lock (mutex_);
        return status_;
    }

    double runningForMs() const
    {
        const std::lock_guard<std::mutex> lock (mutex_);
        return status_ == Status::Running
            ? std::chrono::duration<double, std::milli> (std::chrono::steady_clock::now() - started_).count() : 0.0;
    }

    // A warm-up's build that a caller has since joined.
    bool claimed() const
    {
        const std::lock_guard<std::mutex> lock (mutex_);
        return claimed_;
    }

    // The last warm-up failed unclaimed: the gate went back to idle and said nothing. Empty
    // when none did since the last build began.
    // The build calls this when it has finished and holds its result: false means nobody
    // claimed it, so release what it made and return true. From then on a caller can't claim
    // it; the gate goes idle when the build returns, and the caller builds again.
    bool keepWarmUp()
    {
        const std::lock_guard<std::mutex> lock (mutex_);
        if (claimed_)
            return true;
        releasing_ = true;
        return false;
    }

    // The last warm-up ended unclaimed and released what it made.
    bool released() const
    {
        const std::lock_guard<std::mutex> lock (mutex_);
        return released_;
    }

    bool silentFailure (std::string& error) const
    {
        const std::lock_guard<std::mutex> lock (mutex_);
        error = silentError_;
        return silentFailed_;
    }

    // Blocks while the build runs. True when it succeeded.
    bool await (std::string& error)
    {
        std::unique_lock<std::mutex> lock (mutex_);
        if (const auto claimed = claimLocked())
        {
            lock.unlock();
            claimed();
            lock.lock();
        }
        changed_.wait (lock, [this] { return status_ != Status::Running; });
        if (status_ == Status::Ready)
            return true;
        error = error_;
        if (status_ == Status::Failed)
            status_ = Status::Idle;
        return false;
    }

private:
    // The claim hook for the first caller to join a running warm-up, or empty.
    std::function<void()> claimLocked()
    {
        if (status_ != Status::Running || ! warmUp_ || claimed_ || releasing_)
            return {};
        claimed_ = true;
        return std::exchange (onClaim_, {});
    }

    void markRunning (Origin origin)
    {
        status_ = Status::Running;
        stopping_ = false;
        warmUp_ = origin == Origin::WarmUp;
        claimed_ = false;
        releasing_ = false;
        released_ = false;
        silentFailed_ = false;
        silentError_.clear();
        onClaim_ = {};
        started_ = std::chrono::steady_clock::now();
        startedNs_ = steadyNowNs();
        runningBuildStartNs() = startedNs_;
    }

    void launch (std::function<bool (std::string&)> build)
    {
        const auto run = [this, build = std::move (build)]
        {
            std::string error;
            const bool built = build (error);
            const std::lock_guard<std::mutex> lock (mutex_);
            auto mine = startedNs_;
            runningBuildStartNs().compare_exchange_strong (mine, 0);
            // A compile that ends failed after a Cancel, even one taken back since, is that
            // Cancel's doing and no failure to report.
            if (built && releasing_)
            {
                status_ = Status::Idle;
                released_ = true;
                releasing_ = false;
            }
            else if (built)
            {
                status_ = Status::Ready;
                stopRequested_ = false;
            }
            else if (stopRequested_ || stopping_)
                status_ = stopRequested_ ? Status::Stopped : Status::Idle;
            else if (warmUp_ && ! claimed_)
            {
                silentFailed_ = true;
                silentError_ = std::move (error);
                status_ = Status::Idle;
            }
            else
            {
                status_ = Status::Failed;
                error_ = std::move (error);
            }
            changed_.notify_all();
        };
        try
        {
            startBuildThread (run);
        }
        catch (const std::system_error&)
        {
            run();  // no thread to be had: the caller waits, and the build still happens
        }
    }

    mutable std::mutex mutex_;
    std::condition_variable changed_;
    Status status_ = Status::Idle;
    bool stopRequested_ = false;
    bool stopping_ = false;
    int joined_ = 0;
    bool warmUp_ = false;
    bool claimed_ = false;
    bool releasing_ = false;
    bool released_ = false;
    bool silentFailed_ = false;
    std::string silentError_;
    std::function<void()> onClaim_;
    std::string error_;
    std::chrono::steady_clock::time_point started_ {};
    std::int64_t startedNs_ = 0;
};

} // namespace videohelper::pathtrace::kernelcompile
