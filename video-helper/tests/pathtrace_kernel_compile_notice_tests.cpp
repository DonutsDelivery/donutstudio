// Tells a kernel compile from a stall: the compile notice stands only while a compile
// accounts for a step's age, and the stall report stays for everything else.

#include "pathtrace/devices/kernel_compile_state.h"

#include <cstdio>
#include <string>

namespace
{
namespace kc = videohelper::pathtrace::kernelcompile;

int checks = 0;
int failures = 0;

void check (bool condition, const std::string& message)
{
    ++checks;
    if (! condition)
    {
        ++failures;
        std::printf ("FAIL: %s\n", message.c_str());
    }
}

constexpr std::int64_t second = 1000000000;
constexpr double report = 5000.0;

kc::StepVerdict judge (std::int64_t step, std::int64_t now, kc::Reading compile) { return kc::judgeStep (step * second, now * second, compile, report); }
kc::Reading running (std::int64_t start) { return { start * second, 0, 0 }; }
kc::Reading finished (std::int64_t start, std::int64_t end) { return { 0, start * second, end * second }; }
} // namespace

int main()
{
    check (judge (100, 106, {}) == kc::StepVerdict::Stalled, "a step past the report age with no compile is a stall");
    check (judge (100, 190, running (101)) == kc::StepVerdict::Compiling, "a cold compile inside the step is a compile, not a stall");
    check (judge (100, 190, running (50)) == kc::StepVerdict::Compiling, "a compile that began before the step accounts for all of it");
    check (judge (100, 190, running (120)) == kc::StepVerdict::Stalled, "a step stuck for 20 s before a compile began is still a stall");
    check (judge (100, 190, finished (101, 188)) == kc::StepVerdict::Compiled, "a step back at work after a compile is neither");
    check (judge (100, 200, finished (101, 188)) == kc::StepVerdict::Stalled, "a step stuck 12 s after its compile returned is a stall");
    check (judge (100, 190, finished (10, 20)) == kc::StepVerdict::Stalled, "an earlier step's compile explains nothing");
    check (judge (100, 100 + 7200, running (100)) == kc::StepVerdict::Compiling, "a compile of two hours is still a compile, as a compile has no limit");
    check (judge (100, 101, running (100)) == kc::StepVerdict::Compiling, "a compile is named from its first moment");

    kc::State state;
    check (state.read().activeStartNs == 0, "nothing runs at first");
    state.begin (5);
    state.begin (6);
    check (state.read().activeStartNs == 5, "the first of two compiles starts the span");
    state.end (7);
    check (state.read().activeStartNs == 5, "the span stays while another compile runs");
    state.end (9);
    check (state.read().activeStartNs == 0 && state.read().lastStartNs == 5 && state.read().lastEndNs == 9,
           "the last compile to end closes the span");
    {
        kc::Scope scope;
        check (kc::processState().read().activeStartNs != 0, "a scope marks the compile");
    }
    check (kc::processState().read().activeStartNs == 0 && kc::processState().read().lastEndNs != 0,
           "leaving the scope ends it, on a failed compile's early return too");

    std::printf ("%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
