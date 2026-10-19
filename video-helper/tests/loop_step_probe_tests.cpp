// Unit tests for the render-thread loop-step probe. A scope names its call in
// the installed sink, restores the enclosing label on exit, and keeps a record
// of the last call that ran long, so viewport_info can name a stall after the
// call has returned.
#include "../src/loop_step_probe.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace
{
void require (bool condition, const char* message)
{
    if (! condition) throw std::runtime_error (message);
}

bool same (const char* a, const char* b)
{
    return a != nullptr && b != nullptr && std::strcmp (a, b) == 0;
}

struct ProbeState
{
    std::atomic<const char*> label { "advance:imported-scene-exec" };
    std::atomic<std::int64_t> enterNs { 42 };
    std::atomic<const char*> slowLabel { "" };
    std::atomic<std::int64_t> slowMs { 0 };
};

void scopeNamesAndRestoresTheEnclosingStep()
{
    ProbeState state;
    loopstep::Sink sink { &state.label, &state.enterNs, &state.slowLabel, &state.slowMs };
    loopstep::SinkInstall install (&sink);
    {
        loopstep::Scope outer ("advance:imported-scene:preview-render");
        require (same (state.label.load(), "advance:imported-scene:preview-render"),
                 "outer scope did not publish its label");
        {
            loopstep::Scope inner ("metal:fixture-scene-pipeline-create");
            require (same (state.label.load(), "metal:fixture-scene-pipeline-create"),
                     "inner scope did not publish its label");
        }
        require (same (state.label.load(), "advance:imported-scene:preview-render"),
                 "inner scope did not restore the outer label");
    }
    require (same (state.label.load(), "advance:imported-scene-exec"),
             "outer scope did not restore the loop step label");
    require (state.enterNs.load() == 42, "outer scope did not restore the loop step entry time");
    require (same (state.slowLabel.load(), ""), "a fast scope left a slow-call record");
    require (state.slowMs.load() == 0, "a fast scope left a slow-call duration");
}

void slowScopeLeavesARecordAfterItReturns()
{
    ProbeState state;
    loopstep::Sink sink { &state.label, &state.enterNs, &state.slowLabel, &state.slowMs };
    loopstep::SinkInstall install (&sink);
    const auto reportMs = loopstep::Scope::kReportNs / 1000000;
    {
        loopstep::Scope slow ("advance:imported-scene:fixture-prepare");
        std::this_thread::sleep_for (std::chrono::milliseconds (reportMs + 20));
    }
    require (same (state.slowLabel.load(), "advance:imported-scene:fixture-prepare"),
             "a slow scope did not record its label");
    require (state.slowMs.load() >= reportMs, "a slow scope recorded less than its duration");
    {
        loopstep::Scope fast ("advance:imported-scene:fixture-submit");
    }
    require (same (state.slowLabel.load(), "advance:imported-scene:fixture-prepare"),
             "a later fast scope replaced the slow-call record");
}

void sinkWithoutSlowRecordStillNamesTheCall()
{
    std::atomic<const char*> label { "advance" };
    std::atomic<std::int64_t> enterNs { 7 };
    loopstep::Sink sink { &label, &enterNs, nullptr, nullptr };
    loopstep::SinkInstall install (&sink);
    {
        loopstep::Scope scope ("advance:imported-scene:animated-render");
        require (same (label.load(), "advance:imported-scene:animated-render"),
                 "a sink without a slow record did not receive the label");
    }
    loopstep::recordSlow (&sink, "ignored", 9000);
    require (same (label.load(), "advance") && enterNs.load() == 7,
             "a sink without a slow record was not restored");
}

void threadWithoutSinkOnlyTimes()
{
    ProbeState state;
    loopstep::Sink sink { &state.label, &state.enterNs, &state.slowLabel, &state.slowMs };
    loopstep::SinkInstall install (&sink);
    std::thread worker ([]
    {
        loopstep::Scope scope ("advance:imported-scene:export-render");
    });
    worker.join();
    require (same (state.label.load(), "advance:imported-scene-exec"),
             "a scope on a thread without a sink touched another thread's sink");
}
} // namespace

int main()
{
    try
    {
        scopeNamesAndRestoresTheEnclosingStep();
        slowScopeLeavesARecordAfterItReturns();
        sinkWithoutSlowRecordStillNamesTheCall();
        threadWithoutSinkOnlyTimes();
    }
    catch (const std::exception& error)
    {
        std::cerr << "loop step probe tests: FAIL: " << error.what() << "\n";
        return 1;
    }
    std::cout << "loop step probe tests: PASS\n";
    return 0;
}
