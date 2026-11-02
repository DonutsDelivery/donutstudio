#pragma once

// One kernel build for the process, on its own thread. Whoever asks first starts it, at
// helper startup or on the first frame. Later callers join that build: a preview polls
// status() and holds its last frame, and an export blocks in await(). Two builds of the
// same kernel would each compile it from nothing, as the driver's cache fills only when
// one ends. It lives for the process, as its thread may outlast any one renderer.

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>

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

    // Starts the build unless one has run, is running, was stopped, or waits for a caller
    // to read its failure. A failure read in await() or join() lets the next caller try again.
    void start (std::function<bool (std::string&)> build)
    {
        {
            const std::lock_guard<std::mutex> lock (mutex_);
            if (status_ != Status::Idle)
                return;
            markRunning();
        }
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
        for (;;)
        {
            if (status_ == Status::Idle)
            {
                markRunning();
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

    // Blocks while the build runs. True when it succeeded.
    bool await (std::string& error)
    {
        std::unique_lock<std::mutex> lock (mutex_);
        changed_.wait (lock, [this] { return status_ != Status::Running; });
        if (status_ == Status::Ready)
            return true;
        error = error_;
        if (status_ == Status::Failed)
            status_ = Status::Idle;
        return false;
    }

private:
    void markRunning()
    {
        status_ = Status::Running;
        stopping_ = false;
        started_ = std::chrono::steady_clock::now();
    }

    void launch (std::function<bool (std::string&)> build)
    {
        const auto run = [this, build = std::move (build)]
        {
            std::string error;
            const bool built = build (error);
            const std::lock_guard<std::mutex> lock (mutex_);
            // A compile that ends failed after a Cancel, even one taken back since, is that
            // Cancel's doing and no failure to report.
            if (built)
            {
                status_ = Status::Ready;
                stopRequested_ = false;
            }
            else if (stopRequested_ || stopping_)
                status_ = stopRequested_ ? Status::Stopped : Status::Idle;
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
    std::string error_;
    std::chrono::steady_clock::time_point started_ {};
};

} // namespace videohelper::pathtrace::kernelcompile
