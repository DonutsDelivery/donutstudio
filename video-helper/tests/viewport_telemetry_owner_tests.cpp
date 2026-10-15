#include "../src/viewport_telemetry_owner.h"
#include "../src/viewport_publish_gate.h"
#include "../src/viewport_rpc_policy.h"

#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{
struct FakeClock
{
    using time_point = uint64_t;
    static std::vector<uint64_t> ticks;
    static size_t cursor;
    static time_point now() { return ticks.at(cursor++); }
    static uint64_t elapsedNs(time_point begin, time_point end) { return end - begin; }
    static void set(std::initializer_list<uint64_t> values) { ticks = values; cursor = 0; }
};
std::vector<uint64_t> FakeClock::ticks;
size_t FakeClock::cursor = 0;
int failures = 0;
void check(bool value, const char* message)
{
    if (! value) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; }
}

// The docked render loop's publish gate, driven the way the loop drives it.
// The shm path publishes one produce late: a readback maps the frame queued by
// the previous produce. The shared path publishes in the produce that composed
// the frame. `version` counts the scene state. A render shows that version once
// `settled` is set, and a placeholder before that, as an unfinished decode would.
// `lastPublished` is the content the consumer received last.
constexpr int kPlaceholder = -2;

struct GateSim
{
    explicit GateSim (bool shmPath) : pipelined (shmPath) {}

    arbitviewport::PublishGate gate;
    arbitviewport::PublishGate::Snapshot in;
    bool pipelined = false;
    bool settled = true;
    bool failNextPublish = false;
    bool queued = false;
    int version = 0;
    int queuedVersion = -1;
    int lastPublished = -1;
    int produces = 0;

    // What a render made now shows.
    int render() const { return settled ? version : kPlaceholder; }

    void change()
    {
        ++version;
        ++in.revision;
    }

    void resize (int width, int height)
    {
        in.width = width;
        in.height = height;
        change();
        if (pipelined)
            queued = false; // a fresh readback returns nothing on its first call
    }

    bool tick()
    {
        const auto now = in;
        if (! gate.shouldProduce (now))
            return false;
        ++produces;
        const bool hadQueued = queued;
        const int mapped = queuedVersion;
        const int shown = render();
        if (pipelined)
        {
            queued = true;
            queuedVersion = shown;
        }
        const bool delivered = (! pipelined || hadQueued) && ! failNextPublish;
        failNextPublish = false;
        if (delivered)
            lastPublished = pipelined ? mapped : shown;
        gate.finish (now, delivered, settled, pipelined);
        return true;
    }
};

int runTicks (GateSim& sim, int count)
{
    int produced = 0;
    for (int i = 0; i < count; ++i)
        produced += sim.tick() ? 1 : 0;
    return produced;
}

void publishGateTests()
{
    {
        GateSim shm (true);
        shm.change();
        runTicks (shm, 4);
        check (shm.lastPublished == shm.version, "shm: first frame reaches the consumer");
        const int before = shm.produces;
        check (runTicks (shm, 200) == 0 && shm.produces == before,
               "shm: stopped unchanged scene produces nothing after the flush");
    }
    {
        GateSim shm (true);
        shm.change();
        runTicks (shm, 4);
        shm.change();
        runTicks (shm, 2);
        check (shm.lastPublished == shm.version, "shm: parameter change reaches the consumer in two produces");
        check (runTicks (shm, 50) == 0, "shm: idle again after a parameter change");
        shm.in.displaySec = 4.25;
        shm.change();
        runTicks (shm, 2);
        check (shm.lastPublished == shm.version, "shm: seek while stopped reaches the consumer");
        check (runTicks (shm, 50) == 0, "shm: idle again after a seek");
        shm.resize (1280, 720);
        runTicks (shm, 4);
        check (shm.lastPublished == shm.version, "shm: resize publishes the new size");
        check (runTicks (shm, 50) == 0, "shm: idle again after a resize");
    }
    {
        GateSim shm (true);
        shm.change();
        runTicks (shm, 4);
        shm.in.playing = true;
        shm.change();
        int played = 0;
        for (int i = 0; i < 10; ++i)
        {
            shm.in.displaySec += 1.0 / 30.0;
            shm.change();
            played += shm.tick() ? 1 : 0;
        }
        check (played == 10, "shm: playback produces every tick");
        check (shm.lastPublished >= shm.version - 1, "shm: playback lags the consumer by one produce");
        shm.in.playing = false;
        shm.change();
        runTicks (shm, 3);
        check (shm.lastPublished == shm.version, "shm: stop settles on the final frame");
        check (runTicks (shm, 50) == 0, "shm: idle after playback stops");
    }
    {
        GateSim shm (true);
        shm.change();
        runTicks (shm, 4);
        shm.in.visible = false;
        shm.in.displaySec = 9.0;
        shm.change();
        check (runTicks (shm, 20) == 0, "shm: hidden viewport never produces");
        shm.in.visible = true;
        shm.change();
        check (runTicks (shm, 1) == 1, "shm: showing the viewport produces at once");
        runTicks (shm, 4);
        check (shm.lastPublished == shm.version, "shm: showing the viewport refreshes the frame");
    }
    {
        GateSim shm (true);
        shm.settled = false; // a live-preview script, scopes, optical flow, or a decode in flight
        shm.change();
        check (runTicks (shm, 6) == 6, "shm: unsettled frame keeps producing every tick");
        shm.settled = true; // the decode lands, and no input changes
        runTicks (shm, 4);
        check (shm.lastPublished == shm.version,
               "shm: a frame that settles without new input reaches the consumer");
        check (runTicks (shm, 50) == 0, "shm: settled frame stops producing");
    }
    {
        GateSim shared (false);
        shared.change();
        check (runTicks (shared, 1) == 1 && shared.lastPublished == shared.version,
               "shared: a change publishes in the produce that composed it");
        check (runTicks (shared, 50) == 0, "shared: idle after one frame");
        shared.change();
        check (runTicks (shared, 1) == 1 && shared.lastPublished == shared.version,
               "shared: each change publishes at once");
        shared.in.interpSuspended = true;
        check (runTicks (shared, 1) == 1, "shared: suspending Speed Warp produces once");
        shared.in.interpSuspended = false;
        check (runTicks (shared, 1) == 1, "shared: resuming Speed Warp produces once");
        check (runTicks (shared, 50) == 0, "shared: idle after Speed Warp resumes");
    }
    {
        GateSim shared (false);
        shared.change();
        shared.failNextPublish = true;
        check (runTicks (shared, 1) == 1 && shared.lastPublished == -1,
               "shared: a dropped frame is not recorded as published");
        check (runTicks (shared, 1) == 1 && shared.lastPublished == shared.version,
               "shared: a dropped frame is produced again on the next tick");
        check (runTicks (shared, 50) == 0, "shared: idle after the retry");
    }
    {
        GateSim shared (false);
        shared.settled = false; // the frame is not final yet
        shared.change();
        check (runTicks (shared, 3) == 3, "shared: unsettled frame keeps producing");
        shared.settled = true; // the decode lands, and no input changes
        check (runTicks (shared, 1) == 1 && shared.lastPublished == shared.version,
               "shared: a frame that settles without new input is published");
        check (runTicks (shared, 50) == 0, "shared: idle once the frame settles");
    }
    {
        GateSim shared (false);
        shared.change();
        runTicks (shared, 1);
        check (runTicks (shared, 50) == 0, "shared: settled frame idles before a consumer connects");
        shared.gate.requestProduce(); // a consumer connects to the settled frame
        check (runTicks (shared, 1) == 1 && shared.lastPublished == shared.version,
               "shared: a connecting consumer gets the settled frame in one produce");
        check (runTicks (shared, 50) == 0, "shared: idle again after the connect produce");
    }
    {
        GateSim shared (false);
        shared.change();
        runTicks (shared, 1);
        shared.in.visible = false;
        shared.gate.requestProduce();
        check (runTicks (shared, 20) == 0, "shared: a connect request while hidden produces nothing");
        shared.in.visible = true;
        shared.change(); // showing the viewport moves the revision
        check (runTicks (shared, 1) == 1 && shared.lastPublished == shared.version,
               "shared: showing the viewport after a hidden connect publishes once");
        check (runTicks (shared, 50) == 0, "shared: idle after showing the viewport");
    }
    {
        GateSim shm (true);
        shm.change();
        runTicks (shm, 4);
        shm.gate.requestProduce();
        check (runTicks (shm, 1) == 1 && shm.lastPublished == shm.version,
               "shm: a requested produce on a settled scene keeps the consumer current");
        check (runTicks (shm, 50) == 0, "shm: idle after a requested produce");
    }
}

// The revision policy the RPC guard applies to each handle() call.
void rpcPolicyTests()
{
    using namespace arbitviewport::rpc;
    check (isExempt ("ping") && ! isSetter ("ping"), "policy: polls are exempt and not setters");
    check (isSetter ("viewport_set_view") && ! isExempt ("viewport_set_view"),
           "policy: a plain setter is neither exempt nor other");
    check (! isSetter ("viewport_set_timeline") && ! isSetter ("graph_set_lut")
           && ! isSetter ("text_set_image"),
           "policy: RPCs that name files or shared memory are not setters");
    check (bumpsItself ("viewport_set_canvas") && bumpsItself ("viewport_set_canvas_background")
           && bumpsItself ("viewport_set_visible") && ! bumpsItself ("viewport_set_view"),
           "policy: Viewport bumps itself only for the canvas pair and visibility");

    const auto exempt = decide ("ping", false, false);
    check (! exempt.bump && ! exempt.remember && ! exempt.forget,
           "policy: exempt calls leave the revision and memo alone");
    const auto exemptFailed = decide ("ping", true, false);
    check (! exemptFailed.bump && ! exemptFailed.forget,
           "policy: a failed exempt call leaves the memo alone");

    const auto firstSetter = decide ("viewport_set_view", false, false);
    check (firstSetter.bump && firstSetter.remember && ! firstSetter.forget,
           "policy: a new setter bumps and is remembered");
    const auto repeatSetter = decide ("viewport_set_view", false, true);
    check (! repeatSetter.bump && ! repeatSetter.remember && ! repeatSetter.forget,
           "policy: an identical repeat of a setter changes nothing");
    const auto failedSetter = decide ("viewport_set_view", true, true);
    check (failedSetter.bump && failedSetter.forget && ! failedSetter.remember,
           "policy: a failed setter bumps and clears the memo");

    const auto canvasFirst = decide ("viewport_set_canvas", false, false);
    check (! canvasFirst.bump && canvasFirst.remember,
           "policy: a new canvas call records the memo and leaves the bump to Viewport");
    const auto canvasFailed = decide ("viewport_set_canvas", true, false);
    check (canvasFailed.bump && canvasFailed.forget, "policy: a failed canvas call bumps");

    const auto lutRepeat = decide ("graph_set_lut", false, true);
    check (lutRepeat.bump && lutRepeat.forget, "policy: a repeated LUT path bumps and clears the memo");
    const auto timelineRepeat = decide ("viewport_set_timeline", false, true);
    check (timelineRepeat.bump && timelineRepeat.forget,
           "policy: a repeated timeline bumps and clears the memo");
    const auto imageRepeat = decide ("text_set_image", false, true);
    check (imageRepeat.bump && imageRepeat.forget, "policy: a repeated shared-memory image bumps");
    const auto requestFrame = decide ("viewport_request_frame", false, false);
    check (requestFrame.bump && requestFrame.forget, "policy: other content RPCs always bump");
}

// The guard object, driven the way handle() drives it. It is a template over the params
// type, so these tests use int and need no JSON.
void revisionGuardTests()
{
    using namespace arbitviewport::rpc;
    using Guard = RevisionGuard<int>;
    RepeatMemo<int> memo;
    int bumps = 0;
    const auto bump = [&bumps] { ++bumps; };
    const std::string none;
    const std::string failure = "handler failed";

    // One handle() call. The guard's destructor runs when the call ends.
    const auto call = [&](const char* method, int params, const std::string& error)
    {
        const Guard guard (memo, method, params, error, bump);
    };

    call ("ping", 1, none);
    check (bumps == 0 && ! memo.valid, "guard: an exempt call moves nothing");

    call ("viewport_set_bounds", 1, none);
    check (bumps == 1 && memo.valid, "guard: a first setter bumps and is remembered");
    call ("viewport_set_bounds", 1, none);
    check (bumps == 1, "guard: an identical setter repeat does not bump");
    call ("viewport_set_bounds", 2, none);
    check (bumps == 2, "guard: a setter with changed params bumps");

    call ("viewport_set_canvas", 5, none);
    check (bumps == 2 && memo.method == "viewport_set_canvas",
           "guard: a canvas call is remembered and Viewport bumps for it");
    call ("viewport_set_canvas", 5, none);
    check (bumps == 2, "guard: an identical canvas repeat does not bump");
    call ("viewport_set_bounds", 2, none);
    check (bumps == 3, "guard: a canvas call breaks the repeat chain");

    call ("viewport_set_bounds", 2, failure);
    check (bumps == 4 && ! memo.valid, "guard: a failed call bumps and clears the memo");
    call ("viewport_set_bounds", 2, none);
    check (bumps == 5, "guard: an identical call after a failure bumps");

    call ("graph_set_lut", 7, none);
    call ("graph_set_lut", 7, none);
    check (bumps == 7 && ! memo.valid, "guard: a path RPC bumps on every call");

    call ("viewport_set_bounds", 3, none);
    check (bumps == 8 && memo.valid, "guard: a setter is remembered before a throw");
    const int thrownParams = 3;
    try
    {
        const Guard guard (memo, "viewport_set_bounds", thrownParams, none, bump);
        throw std::runtime_error ("handler threw");
    }
    catch (const std::runtime_error&)
    {
    }
    check (bumps == 9 && ! memo.valid, "guard: a throwing handler bumps and clears the memo");
}

} // namespace

int main()
{
    videowire::VisualPlanTelemetry telemetry;
    videowire::ViewportTelemetryOwner<FakeClock> owner(telemetry, false);
    check(owner.admitBackend(videowire::VisualBackend::openGL), "OpenGL admission");
    check(owner.admitBackend(videowire::VisualBackend::metal), "repeated admission ignored");

    FakeClock::set({10, 10});
    check(owner.renderComposite([] { return 7u; }, [] (unsigned value) { return value != 0; }) == 7,
          "valid compositor output returned");
    FakeClock::set({20, 20});
    check(owner.present([] { return true; }), "successful presentation returned");

    FakeClock::set({30, 35});
    owner.renderComposite([] { return 0u; }, [] (unsigned value) { return value != 0; });
    FakeClock::set({40, 49});
    check(! owner.present([] { return false; }), "presentation failure returned");

    auto snapshot = telemetry.snapshot();
    check(snapshot.compositor.count == 2 && snapshot.compositor.totalNs == 5,
          "zero compositor duration is observed and counted");
    check(snapshot.presentation.count == 2 && snapshot.presentation.totalNs == 9,
          "zero presentation duration is observed and counted");
    check(snapshot.framesRendered == 1 && snapshot.framesPresented == 1,
          "failures never increment success counters");
    check(snapshot.droppedRenderFailure == 1 && snapshot.droppedTransportFailure == 1,
          "renderer and presentation failures map exact drop reasons");
    check(snapshot.initialBackend == videowire::VisualBackend::openGL
          && snapshot.currentBackend == videowire::VisualBackend::openGL
          && snapshot.fallbackCount == 0,
          "backend identity is exact and repeated admission is not fallback");

    check(owner.transitionBackend(videowire::VisualBackend::vulkan), "actual backend transition");
    snapshot = telemetry.snapshot();
    check(snapshot.currentBackend == videowire::VisualBackend::vulkan && snapshot.fallbackCount == 1,
          "actual transition increments fallback once");

    videowire::VisualPlanTelemetry strictTelemetry;
    videowire::ViewportTelemetryOwner<FakeClock> strict(strictTelemetry, true);
    check(strict.admitBackend(videowire::VisualBackend::metal), "strict Metal admission");
    check(! strict.transitionBackend(videowire::VisualBackend::openGL),
          "strict Metal refuses degraded transition");
    const auto strictSnapshot = strictTelemetry.snapshot();
    check(strictSnapshot.initialBackend == videowire::VisualBackend::metal
          && strictSnapshot.currentBackend == videowire::VisualBackend::metal
          && strictSnapshot.fallbackCount == 0 && strictSnapshot.failedLowerings == 1,
          "strict Metal diagnoses without recording degraded pass");
    check(! snapshot.particles.observed && ! snapshot.generators.observed,
          "unowned generator and particle timings remain unavailable");

    videowire::VisualPlanTelemetry transportTelemetry;
    videowire::ViewportTelemetryOwner<FakeClock> transport(transportTelemetry, false);
    check(transport.readbackCopied(true, 32, 3, 8, 3, 8, 3),
          "successful readback payload is admitted");
    auto transportSnapshot = transportTelemetry.snapshot();
    check(transportSnapshot.readbackFrames == 1
          && transportSnapshot.readbackCopiedBytes == 96
          && transportSnapshot.transportFramesHandedOff == 1
          && transportSnapshot.actualWidth == 8 && transportSnapshot.actualHeight == 3,
          "readback records checked stride times height and actual dimensions");
    check(! transport.readbackCopied(false, 32, 3, 8, 3, 8, 3)
          && ! transport.readbackCopied(true, std::numeric_limits<size_t>::max(), 2,
                                        8, 3, 8, 3),
          "failed and overflowing copies are rejected");
    transportSnapshot = transportTelemetry.snapshot();
    check(transportSnapshot.readbackFrames == 1 && transportSnapshot.readbackCopiedBytes == 96
          && transportSnapshot.droppedTransportFailure == 2,
          "failed copies do not count successful readback");

    check(transport.exportedBufferAllocated(80)
          && transport.exportedBufferAllocated(80), "concrete exported pool allocations");
    transportSnapshot = transportTelemetry.snapshot();
    check(transportSnapshot.zeroCopyAllocationBytes == 160
          && transportSnapshot.retainedFramesCurrent == 2
          && transportSnapshot.retainedFramesPeak == 2
          && transportSnapshot.intermediateImagesCurrent == 2
          && transportSnapshot.retainedBytesCurrent == 160,
          "pool publishes absolute current and peak snapshots");
    check(! transport.zeroCopyHandoff(false, true, true)
          && ! transport.zeroCopyHandoff(true, false, true)
          && ! transport.zeroCopyHandoff(true, true, false),
          "zero-copy stage failures are rejected");
    transport.noFreeExportedBuffer();
    transportSnapshot = transportTelemetry.snapshot();
    check(transportSnapshot.zeroCopyFrames == 0
          && transportSnapshot.droppedBlitFailure == 1
          && transportSnapshot.droppedFenceFailure == 1
          && transportSnapshot.droppedSocketFailure == 1
          && transportSnapshot.droppedNoBuffer == 1,
          "exact zero-copy failure reasons never count success");
    check(transport.zeroCopyHandoff(true, true, true), "successful socket handoff");
    transport.exportedBufferReleased(80);
    transport.exportedBufferPoolClosed();
    transportSnapshot = transportTelemetry.snapshot();
    check(transportSnapshot.zeroCopyFrames == 1
          && transportSnapshot.transportFramesHandedOff == 2
          && transportSnapshot.framesPresented == 0,
          "transport handoff is distinct from window presentation");
    check(transportSnapshot.retainedFramesCurrent == 0
          && transportSnapshot.intermediateImagesCurrent == 0
          && transportSnapshot.retainedBytesCurrent == 0
          && transportSnapshot.retainedFramesPeak == 2,
          "pool teardown returns current resources to zero and preserves peak");

    videowire::VisualPlanTelemetry saturated;
    saturated.seedForSaturationTesting(std::numeric_limits<uint64_t>::max());
    saturated.recordTransportHandoff(videowire::VisualTransportMode::zeroCopy);
    saturated.recordDrop(videowire::VisualDropReason::blitFailure);
    saturated.recordDrop(videowire::VisualDropReason::fenceFailure);
    saturated.recordDrop(videowire::VisualDropReason::socketFailure);
    const auto saturatedSnapshot = saturated.snapshot();
    check(saturatedSnapshot.transportFramesHandedOff == std::numeric_limits<uint64_t>::max()
          && saturatedSnapshot.droppedBlitFailure == std::numeric_limits<uint64_t>::max()
          && saturatedSnapshot.droppedFenceFailure == std::numeric_limits<uint64_t>::max()
          && saturatedSnapshot.droppedSocketFailure == std::numeric_limits<uint64_t>::max(),
          "viewport transport and stage-specific drop counters saturate");

    publishGateTests();
    rpcPolicyTests();
    revisionGuardTests();

    std::printf("viewport telemetry owner: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
