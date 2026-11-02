// One kernel compile, not two: whoever asks during a build joins it.

#include "pathtrace/devices/kernel_build_gate.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

namespace kc = videohelper::pathtrace::kernelcompile;

namespace
{
int failures = 0;

void check (bool condition, const std::string& message)
{
    if (! condition)
    {
        ++failures;
        std::printf ("FAIL: %s\n", message.c_str());
    }
}
} // namespace

int main()
{
    {
        // Startup begins the build. A preview polls and an export blocks while it runs.
        static kc::BuildGate gate;
        std::atomic<int> builds { 0 };
        std::atomic<bool> release { false };
        const auto build = [&] (std::string&)
        {
            ++builds;
            while (! release)
                std::this_thread::sleep_for (std::chrono::milliseconds (1));
            return true;
        };
        check (gate.status() == kc::BuildGate::Status::Idle, "nothing runs at first");
        gate.start (build);
        check (gate.status() == kc::BuildGate::Status::Running, "the first caller starts the build");
        std::atomic<int> joined { 0 };
        std::vector<std::thread> callers;
        for (int caller = 0; caller < 8; ++caller)
            callers.emplace_back ([&]
            {
                gate.start (build);
                std::string error;
                if (gate.await (error))
                    ++joined;
            });
        for (int poll = 0; poll < 20; ++poll)
        {
            gate.start (build);
            check (gate.status() == kc::BuildGate::Status::Running, "a preview's poll sees the build still running");
        }
        check (joined == 0, "an export waits for the build it joined");
        release = true;
        for (auto& caller : callers)
            caller.join();
        check (builds == 1, "one compile, not two, for startup, previews and exports together");
        check (joined == 8, "every caller gets the one build's result");
        gate.start (build);
        check (builds == 1 && gate.status() == kc::BuildGate::Status::Ready, "a finished build stays finished");
        check (gate.runningForMs() == 0.0, "a finished build has no running time");
    }
    {
        // A failure reaches a caller, and only then does another caller try again.
        static kc::BuildGate gate;
        std::atomic<int> builds { 0 };
        const auto build = [&] (std::string& error)
        {
            error = "no " + std::to_string (++builds);
            return builds > 1;
        };
        gate.start (build);
        std::string error;
        check (! gate.await (error) && error == "no 1", "a failed build names its failure");
        check (gate.status() == kc::BuildGate::Status::Idle, "a read failure leaves the gate open");
        gate.start (build);
        error.clear();
        check (gate.await (error) && builds == 2, "the next caller builds again and can succeed");
    }
    {
        // Cancel, then Start again while the cancelled compile still ends: never two at once.
        static kc::BuildGate gate;
        std::atomic<int> builds { 0 }, running { 0 }, most { 0 };
        std::atomic<bool> cancel { false };
        const auto build = [&] (std::string& error)
        {
            ++builds;
            most = std::max (most.load(), ++running);
            while (! cancel)
                std::this_thread::sleep_for (std::chrono::milliseconds (1));
            --running;
            error = "cancelled";
            return false;
        };
        gate.start (build);
        check (gate.stop(), "Cancel applies to a running build");
        check (gate.stopped(), "a preview now waits by name");
        gate.start (build);
        check (builds <= 1, "a preview's poll starts nothing after Cancel");
        gate.resume();
        gate.start (build);
        std::this_thread::sleep_for (std::chrono::milliseconds (20));
        check (builds == 1 && gate.status() == kc::BuildGate::Status::Running, "Start again joins the compile still ending");
        cancel = true;
        while (gate.status() == kc::BuildGate::Status::Running)
            std::this_thread::sleep_for (std::chrono::milliseconds (1));
        check (gate.status() == kc::BuildGate::Status::Idle, "a compile ended by a Cancel is no failure");
        cancel = false;
        std::atomic<bool> finish { false };
        const auto second = [&] (std::string&)
        {
            ++builds;
            most = std::max (most.load(), ++running);
            while (! finish)
                std::this_thread::sleep_for (std::chrono::milliseconds (1));
            --running;
            return true;
        };
        gate.start (second);
        finish = true;
        std::string error;
        check (gate.await (error) && builds == 2 && most == 1, "the next poll starts one new compile, alone");
        check (! gate.stop(), "Cancel does nothing to a finished build");
    }
    {
        // Cancel with nothing running holds until Start again, and an export overrides it.
        static kc::BuildGate gate;
        std::atomic<int> builds { 0 };
        const auto build = [&] (std::string&) { ++builds; return true; };
        check (gate.stop() && gate.status() == kc::BuildGate::Status::Stopped, "Cancel before any build holds");
        gate.start (build);
        check (builds == 0, "nothing starts while stopped");
        std::string error;
        check (gate.join (build, error) && builds == 1 && ! gate.stopped(), "an export builds anyway");
    }
    {
        // An export blocked on the build can't have it cancelled under it.
        static kc::BuildGate gate;
        std::atomic<bool> finish { false };
        const auto build = [&] (std::string&)
        {
            while (! finish)
                std::this_thread::sleep_for (std::chrono::milliseconds (1));
            return true;
        };
        std::string error;
        std::thread exporter ([&] { gate.join (build, error); });
        while (! gate.hasWaiters())
            std::this_thread::sleep_for (std::chrono::milliseconds (1));
        check (! gate.stop(), "Cancel is refused while an export waits");
        finish = true;
        exporter.join();
        check (gate.status() == kc::BuildGate::Status::Ready, "the export got its build");
    }
    std::printf (failures == 0 ? "PASS\n" : "FAILED\n");
    return failures == 0 ? 0 : 1;
}
