// The background kernel warm-up: it starts only when the machine is quiet, holds while the
// user plays, exports or renders, is joined by the viewport instead of restarted, and fails
// without a word. A real BuildGate runs a fake build, so no GPU or driver is involved.

#include "pathtrace/devices/kernel_module_stamp.h"
#include "pathtrace/devices/kernel_warmup.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <thread>

namespace kc = videohelper::pathtrace::kernelcompile;
namespace ks = videohelper::pathtrace::kernelstamp;

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

template <typename Condition>
bool waitFor (Condition condition)
{
    for (int i = 0; i < 2000; ++i)
    {
        if (condition())
            return true;
        std::this_thread::sleep_for (std::chrono::milliseconds (2));
    }
    return false;
}

constexpr std::int64_t second = 1'000'000'000LL;

// One fake kernel build: `tasks` task boundaries, each calling yield, then a result.
struct Fake
{
    std::atomic<int> builds { 0 };
    std::atomic<int> tasksDone { 0 };
    std::atomic<int> lowered { 0 };
    std::atomic<int> restored { 0 };
    std::atomic<bool> succeed { true };
    std::atomic<bool> release { true }; // false holds the build before its next task
    std::atomic<bool> contextHeld { false }; // the fake GPU context the build made
    std::atomic<bool> destroying { false };  // the release is in progress
    std::atomic<bool> holdRelease { false }; // true holds the release before it frees the context
    int tasks = 3;

    kc::WarmupHooks hooks (bool hashPresent, std::string skip = {})
    {
        kc::WarmupHooks h;
        h.moduleHashPresent = [hashPresent] { return hashPresent; };
        h.skipReason = [skip] { return skip; };
        h.build = [this] (std::string& error, const std::function<void()>& yield, const std::function<bool()>& keep)
        {
            ++builds;
            contextHeld = true;
            for (int task = 0; task < tasks; ++task)
            {
                yield();
                while (! release)
                    std::this_thread::sleep_for (std::chrono::milliseconds (1));
                ++tasksDone;
            }
            if (! succeed)
            {
                contextHeld = false;
                error = "driver said no";
                return false;
            }
            if (! keep())
            {
                destroying = true;
                while (holdRelease)
                    std::this_thread::sleep_for (std::chrono::milliseconds (1));
                contextHeld = false;
                destroying = false;
            }
            return true;
        };
        h.lowerPriority = [this] { ++lowered; return true; };
        h.restorePriority = [this] { ++restored; return true; };
        return h;
    }
};

bool releasedDone (kc::Warmup& warmup, kc::BuildGate& gate)
{
    return waitFor ([&] { return warmup.read().state == kc::WarmupState::Done && gate.status() == kc::BuildGate::Status::Idle; });
}

void quiet()
{
    kc::activity().transportPlaying = false;
    kc::activity().helperBusy = [] { return false; };
}
} // namespace

int main()
{
    quiet();

    {
        // A kernel already in the driver's cache needs no warm-up, and the state says why.
        static kc::BuildGate gate;
        Fake fake;
        kc::Warmup warmup (gate, fake.hooks (true), 0);
        warmup.arm();
        check (! warmup.tick (10 * second), "no warm-up when the hash is in the cache");
        const auto reading = warmup.read();
        check (reading.state == kc::WarmupState::Idle && reading.moduleHashPresent, "idle, hash present");
        check (fake.builds == 0 && gate.status() == kc::BuildGate::Status::Idle, "nothing built");
    }
    {
        // A machine with no OptiX, or a cache turned off, is not warmed, and says so.
        static kc::BuildGate gate;
        Fake fake;
        kc::Warmup warmup (gate, fake.hooks (false, "the OptiX cache is turned off"), 0);
        warmup.arm();
        check (! warmup.tick (10 * second), "no warm-up when it can't help");
        check (warmup.read().reason == "the OptiX cache is turned off", "the skip reason is shown");
    }
    {
        // Nothing starts before the host's go, nor while the machine is busy, nor until it
        // has been quiet for the settle time.
        static kc::BuildGate gate;
        Fake fake;
        kc::Warmup warmup (gate, fake.hooks (false), 3 * second);
        check (! warmup.tick (100 * second), "no start before the host's go");
        check (warmup.read().state == kc::WarmupState::Idle && ! warmup.read().armed, "idle before the go");
        warmup.arm();
        kc::activity().transportPlaying = true;
        check (! warmup.tick (100 * second) && ! warmup.tick (200 * second), "no start while the transport plays");
        check (warmup.read().state == kc::WarmupState::Idle && warmup.read().busy, "idle and busy while playing");
        kc::activity().transportPlaying = false;
        kc::activity().helperBusy = [] { return true; };
        check (! warmup.tick (300 * second), "no start during an export or an offline render");
        quiet();
        check (! warmup.tick (400 * second), "the quiet time starts counting at the first quiet look");
        check (! warmup.tick (402 * second), "two quiet seconds are not enough");
        kc::activity().transportPlaying = true;
        check (! warmup.tick (404 * second), "a play press resets the quiet time");
        kc::activity().transportPlaying = false;
        check (! warmup.tick (500 * second), "and the count starts again");
        check (warmup.tick (503 * second), "three quiet seconds start the warm-up");
        check (releasedDone (warmup, gate), "the build finishes");
        check (fake.builds == 1 && fake.lowered == 1, "one build, its thread lowered once");
        check (warmup.read().state == kc::WarmupState::Done, "done");
        check (warmup.read().contextReleased && ! fake.contextHeld, "done with no user leaves no GPU context");
        check (gate.status() == kc::BuildGate::Status::Idle, "and the gate is idle for the next real use");
        check (! warmup.tick (600 * second) && fake.builds == 1, "a finished kernel is not built again");
    }
    {
        // Playing holds the build at its next task, and ending the play lets it go on.
        static kc::BuildGate gate;
        Fake fake;
        fake.tasks = 4;
        fake.release = false; // holds the build inside its first task
        kc::Warmup warmup (gate, fake.hooks (false), 0);
        warmup.arm();
        check (warmup.tick (10 * second), "started");
        check (waitFor ([&] { return fake.builds == 1; }), "it works");
        kc::activity().transportPlaying = true;
        fake.release = true; // the first task ends; the next task boundary sees the play
        check (waitFor ([&] { return warmup.read().state == kc::WarmupState::Paused; }), "playing pauses it");
        check (fake.tasksDone == 1, "the task in flight finished, and no further task starts while playing");
        std::this_thread::sleep_for (std::chrono::milliseconds (400));
        check (fake.tasksDone == 1, "it stays held");
        check (gate.status() == kc::BuildGate::Status::Running, "a paused warm-up is still the one running build");
        kc::activity().transportPlaying = false;
        check (releasedDone (warmup, gate), "it resumes and finishes");
        check (fake.tasksDone == 4 && fake.builds == 1, "every task ran once; no restart");
        check (warmup.read().state == kc::WarmupState::Done, "done after the resume");
    }
    {
        // An export running in the helper holds it the same way.
        static kc::BuildGate gate;
        Fake fake;
        fake.release = false;
        kc::Warmup warmup (gate, fake.hooks (false), 0);
        warmup.arm();
        std::atomic<bool> exporting { false };
        kc::activity().helperBusy = [&] { return exporting.load(); };
        warmup.tick (10 * second);
        check (waitFor ([&] { return fake.builds == 1; }), "started");
        exporting = true;
        fake.release = true;
        check (waitFor ([&] { return warmup.read().state == kc::WarmupState::Paused; }), "an export pauses it");
        exporting = false;
        check (releasedDone (warmup, gate), "an export's end releases it");
        quiet();
    }
    {
        // The user picks Path Trace mid warm-up: the viewport joins the running build. One
        // compile, the start time stays the warm-up's, the priority comes back, and a pause
        // does not hold up someone who waits.
        static kc::BuildGate gate;
        Fake fake;
        fake.tasks = 3;
        fake.release = false;
        kc::Warmup warmup (gate, fake.hooks (false), 0);
        warmup.arm();
        warmup.tick (10 * second);
        check (waitFor ([&] { return fake.builds == 1; }), "the warm-up works");
        const auto startedNs = kc::runningBuildStartNs().load();
        check (startedNs != 0, "the build's start time is set");
        kc::activity().transportPlaying = true;
        fake.release = true;
        check (waitFor ([&] { return warmup.read().state == kc::WarmupState::Paused; }), "paused by playing");
        std::atomic<int> viewportBuilds { 0 };
        gate.start ([&] (std::string&) { ++viewportBuilds; return true; });
        check (viewportBuilds == 0, "the viewport's poll starts no second build");
        check (gate.status() == kc::BuildGate::Status::Running, "still the warm-up's build");
        check (kc::runningBuildStartNs().load() == startedNs, "the elapsed time still counts from the warm-up's start");
        check (warmup.read().joined, "the build is claimed");
        check (fake.restored == 1, "the claim lifted the low priority");
        check (waitFor ([&] { return gate.status() == kc::BuildGate::Status::Ready; }),
               "a claimed build ignores the pause and finishes while the transport plays");
        check (viewportBuilds == 0 && fake.builds == 1, "one compile for the warm-up and the viewport together");
        check (warmup.read().state == kc::WarmupState::Done, "done");
        check (! warmup.read().contextReleased && fake.contextHeld, "done after a join keeps the context");
        kc::activity().transportPlaying = false;
    }
    {
        // An export joins the same way, and gets the result.
        static kc::BuildGate gate;
        Fake fake;
        fake.release = false;
        kc::Warmup warmup (gate, fake.hooks (false), 0);
        warmup.arm();
        warmup.tick (10 * second);
        check (waitFor ([&] { return fake.builds == 1; }), "started");
        std::atomic<bool> exported { false };
        std::thread exporter ([&]
        {
            std::string error;
            exported = gate.join ([&] (std::string&) { return false; }, error);
        });
        check (waitFor ([&] { return gate.hasWaiters(); }), "the export waits");
        fake.release = true;
        exporter.join();
        check (exported && fake.builds == 1, "the export got the warm-up's kernel");
    }
    {
        // A warm-up that fails unclaimed says nothing: the gate is idle, and the normal path
        // builds again and reports its own result.
        static kc::BuildGate gate;
        Fake fake;
        fake.succeed = false;
        kc::Warmup warmup (gate, fake.hooks (false), 0);
        warmup.arm();
        warmup.tick (10 * second);
        check (waitFor ([&] { return warmup.read().state == kc::WarmupState::Failed; }), "the warm-up failed");
        check (warmup.read().error == "driver said no", "the failure is kept for the log");
        check (gate.status() == kc::BuildGate::Status::Idle, "the gate is idle, not failed");
        std::string error;
        check (! gate.hasWaiters(), "nobody waits");
        check (! warmup.tick (20 * second), "the warm-up doesn't retry by itself");
        std::atomic<int> userBuilds { 0 };
        gate.start ([&] (std::string&) { ++userBuilds; return true; });
        check (waitFor ([&] { return gate.status() == kc::BuildGate::Status::Ready; }) && userBuilds == 1,
               "the normal path builds as if no warm-up had run");
        check (! gate.await (error) || error.empty(), "and reports no error of the warm-up's");
    }
    {
        // A warm-up the viewport joined fails as the viewport's build: it reaches the caller.
        static kc::BuildGate gate;
        Fake fake;
        fake.succeed = false;
        fake.release = false;
        kc::Warmup warmup (gate, fake.hooks (false), 0);
        warmup.arm();
        warmup.tick (10 * second);
        check (waitFor ([&] { return fake.builds == 1; }), "started");
        gate.start ([] (std::string&) { return true; });
        fake.release = true;
        check (waitFor ([&] { return gate.status() == kc::BuildGate::Status::Failed; }), "a joined failure stays a failure");
        std::string error;
        check (! gate.await (error) && error == "driver said no", "the viewport reads the error");
    }
    {
        // The viewport got there first: no warm-up starts beside its build.
        static kc::BuildGate gate;
        Fake fake;
        kc::Warmup warmup (gate, fake.hooks (false), 0);
        warmup.arm();
        std::atomic<bool> release { false };
        gate.start ([&] (std::string&) { while (! release) std::this_thread::sleep_for (std::chrono::milliseconds (1)); return true; });
        check (! warmup.tick (10 * second) && fake.builds == 0, "no second build beside the viewport's");
        check (warmup.read().state == kc::WarmupState::Running, "the state follows the one running build");
        release = true;
        waitFor ([&] { return gate.status() == kc::BuildGate::Status::Ready; });
    }
    {
        // The user's Cancel stays cancelled: the warm-up does not start the build again.
        static kc::BuildGate gate;
        Fake fake;
        kc::Warmup warmup (gate, fake.hooks (false), 0);
        warmup.arm();
        check (gate.stop(), "Cancel applies to an idle gate");
        check (! warmup.tick (10 * second) && fake.builds == 0, "a cancelled build is not started by the warm-up");
    }
    {
        // A cancel or the process exit frees a held build.
        static kc::BuildGate gate;
        Fake fake;
        fake.release = false;
        std::atomic<bool> cancelled { false };
        auto hooks = fake.hooks (false);
        hooks.cancelled = [&] { return cancelled.load(); };
        kc::Warmup warmup (gate, hooks, 0);
        warmup.arm();
        warmup.tick (10 * second);
        check (waitFor ([&] { return fake.builds == 1; }), "started");
        kc::activity().transportPlaying = true;
        fake.release = true;
        check (waitFor ([&] { return warmup.read().state == kc::WarmupState::Paused; }), "held");
        cancelled = true;
        check (releasedDone (warmup, gate), "a cancel releases the hold");
        kc::activity().transportPlaying = false;
    }
    {
        // After a release, the next real use builds again, and holds the context this time.
        static kc::BuildGate gate;
        Fake fake;
        kc::Warmup warmup (gate, fake.hooks (false), 0);
        warmup.arm();
        warmup.tick (10 * second);
        check (releasedDone (warmup, gate) && ! fake.contextHeld, "released after the warm-up");
        std::atomic<int> userBuilds { 0 };
        gate.start ([&] (std::string&) { ++userBuilds; return true; });
        check (waitFor ([&] { return gate.status() == kc::BuildGate::Status::Ready; }) && userBuilds == 1,
               "a later Path Trace request builds again");
        check (! warmup.tick (20 * second) && fake.builds == 1, "and the warm-up does not run again");
    }
    {
        // A caller that arrives while the context is being freed can't claim it: it waits for
        // the release, then builds for itself.
        static kc::BuildGate gate;
        Fake fake;
        fake.holdRelease = true;
        kc::Warmup warmup (gate, fake.hooks (false), 0);
        warmup.arm();
        warmup.tick (10 * second);
        check (waitFor ([&] { return fake.destroying.load(); }), "the release has begun");
        std::atomic<int> userBuilds { 0 };
        std::atomic<bool> joined { false };
        std::thread exporter ([&]
        {
            std::string error;
            joined = gate.join ([&] (std::string&) { ++userBuilds; return true; }, error);
        });
        std::this_thread::sleep_for (std::chrono::milliseconds (50));
        check (! warmup.read().joined && userBuilds == 0, "the caller did not claim a context being freed");
        fake.holdRelease = false;
        exporter.join();
        check (joined && userBuilds == 1, "it built for itself once the release ended");
        check (gate.status() == kc::BuildGate::Status::Ready, "ready");
    }
    {
        // The stamp that stands in for "the cache holds this kernel".
        const auto root = std::filesystem::temp_directory_path() / "kernel_warmup_stamp_test";
        std::filesystem::remove_all (root);
        std::filesystem::create_directories (root / "cache");
        ks::KeyHasher a, b, c;
        a.add (std::string ("ptx text")); a.addNumber ("driver", 12040);
        b.add (std::string ("ptx text")); b.addNumber ("driver", 12040);
        c.add (std::string ("ptx text")); c.addNumber ("driver", 12050);
        check (a.hex() == b.hex() && a.hex().size() == 32, "the key is stable");
        check (a.hex() != c.hex(), "a new driver is a new key");
        check (! ks::stampPresent (root / "cache", a.hex()), "no stamp in a fresh cache");
        check (ks::writeStamp (root / "cache", a.hex()), "a stamp is written");
        check (ks::stampPresent (root / "cache", a.hex()) && ! ks::stampPresent (root / "cache", c.hex()), "present for its key only");
        check (! ks::writeStamp (root / "missing", a.hex()), "a missing cache directory is not made");
        std::filesystem::remove_all (root / "cache");
        std::filesystem::create_directories (root / "cache");
        check (! ks::stampPresent (root / "cache", a.hex()), "a cleared cache takes the stamp with it");
        const auto env = ks::cacheDirectories ("/priv/optix", "/home/u", nullptr, "u");
        check (env.size() == 1 && env[0] == "/priv/optix", "OPTIX_CACHE_PATH is the only place when set");
        const auto defaults = ks::cacheDirectories (nullptr, "/home/u", nullptr, "u");
        check (defaults.size() == 2 && defaults[0] == "/var/tmp/OptixCache_u", "the default places");
        std::filesystem::remove_all (root);
    }

    if (failures == 0)
        std::printf ("pathtrace kernel warm-up: all checks passed\n");
    return failures == 0 ? 0 : 1;
}
