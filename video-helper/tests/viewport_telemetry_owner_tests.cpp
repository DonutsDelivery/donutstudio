#include "../src/layer_failure_attribution.h"
#include "../src/preview_render_extent.h"
#include "../src/viewport_telemetry_owner.h"
#include "../src/viewport_publish_gate.h"
#include "../src/viewport_rpc_policy.h"
#include "../src/visual_plan_telemetry_json.h"

#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
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

// A frame that the plan keeps refusing, driven the way the render loop drives the
// governor: shouldHold before the attempt, noteFailure after a failed one.
void frameFailureGovernorTests()
{
    using Governor = arbitviewport::FrameFailureGovernor;
    Governor::Key frame;
    frame.revision = 7;
    frame.displaySec = 2.0;
    frame.width = 1280;
    frame.height = 720;

    // Tick every 10 ms against one refused frame for ten seconds.
    const auto attempts = [] (Governor& governor, const Governor::Key& key, int64_t fromMs, int64_t toMs)
    {
        int made = 0;
        for (int64_t at = fromMs; at < toMs; at += 10)
        {
            if (governor.shouldHold (key, at))
                continue;
            ++made;
            governor.noteFailure (key, at);
        }
        return made;
    };

    {
        Governor governor;
        check (! governor.shouldHold (frame, 0), "governor: a frame that never failed is not held");
        const int made = attempts (governor, frame, 0, 10000);
        check (made == Governor::kMaxAttempts, "governor: a refused frame gets a few attempts, then none");
        check (governor.exhausted() && governor.shouldHold (frame, 600000),
               "governor: an exhausted frame stays held however long it waits");
    }
    {
        Governor governor;
        check (! governor.shouldHold (frame, 0), "governor: first attempt runs");
        check (! governor.noteFailure (frame, 0), "governor: the first failure leaves attempts");
        check (governor.shouldHold (frame, 50), "governor: held inside the first backoff");
        check (! governor.shouldHold (frame, 100), "governor: retries when the first backoff ends");
        governor.noteFailure (frame, 100);
        check (governor.shouldHold (frame, 300), "governor: the second backoff is longer than the first");
        check (! governor.shouldHold (frame, 500), "governor: retries when the second backoff ends");
        governor.noteFailure (frame, 500);
        check (governor.shouldHold (frame, 1500), "governor: the third backoff is longer again");
        check (! governor.shouldHold (frame, 2100), "governor: retries when the third backoff ends");
        check (governor.noteFailure (frame, 2100), "governor: the last attempt reports the stop");
        check (governor.shouldHold (frame, 2101), "governor: no attempt after the last");
    }
    {
        // Anything that can change the outcome gives the new frame a full set of attempts.
        Governor governor;
        check (attempts (governor, frame, 0, 10000) == Governor::kMaxAttempts, "governor: setup exhausts");

        auto edited = frame;
        edited.revision = 8;
        check (! governor.shouldHold (edited, 20000), "governor: a state RPC releases the hold");
        check (attempts (governor, edited, 20000, 30000) == Governor::kMaxAttempts,
               "governor: the edited frame gets its own attempts");

        auto moved = edited;
        moved.displaySec = 2.5;
        check (! governor.shouldHold (moved, 40000), "governor: a playhead move releases the hold");
        check (attempts (governor, moved, 40000, 50000) == Governor::kMaxAttempts,
               "governor: the new playhead frame gets its own attempts");

        auto resized = moved;
        resized.width = 1920;
        resized.height = 1080;
        check (! governor.shouldHold (resized, 60000), "governor: a resize releases the hold");
    }
    {
        // Playing, the loop keys on the clips under the playhead, not on the frame. A
        // playhead that moves inside one refused clip must not earn fresh attempts.
        const auto clipsKey = [&] (std::initializer_list<int> ids)
        {
            auto key = frame;
            key.displaySec = 0.0;
            key.clipSet = Governor::kEmptyClipSet;
            for (const int id : ids)
                key.clipSet = Governor::foldClip (key.clipSet, id);
            return key;
        };
        Governor governor;
        const auto refused = clipsKey ({ 4 });
        check (attempts (governor, refused, 0, 10000) == Governor::kMaxAttempts,
               "governor: a refused clip set gets a few attempts while playing");
        check (attempts (governor, refused, 10000, 60000) == 0,
               "governor: playing on inside the refused clip set makes no further attempts");

        check (! governor.shouldHold (clipsKey ({}), 70000),
               "governor: leaving the refused clip releases the hold");
        governor.noteSuccess(); // a gap or a good clip renders
        check (clipsKey ({ 4 }) != clipsKey ({ 5 }) && clipsKey ({ 4 }) != clipsKey ({ 4, 5 })
                   && clipsKey ({ 4 }) != clipsKey ({}),
               "governor: the clip set tells different clips under the playhead apart");
        check (attempts (governor, clipsKey ({ 4, 5 }), 70000, 80000) == Governor::kMaxAttempts,
               "governor: a different clip set gets its own attempts");
    }
    {
        // A frame that renders, or a deferral from another cause, clears the count.
        Governor governor;
        governor.noteFailure (frame, 0);
        governor.noteFailure (frame, 100);
        governor.noteSuccess();
        check (governor.failures() == 0 && ! governor.shouldHold (frame, 101),
               "governor: success forgets earlier failures");
    }
}

// Which export layer errors name their clip. The plugin isolates a named clip, so a
// device, encoder or helper error must not carry a clip id.
void layerFailureAttributionTests()
{
    using videoexport::attributeLayerError;
    check (attributeLayerError (7, "Optical Flow export timeline frame is invalid")
               == "clip 7: Optical Flow export timeline frame is invalid",
           "attribution: a plan error names the layer's clip");
    check (attributeLayerError (7, "typed matte asset frame is unavailable")
               == "clip 7: typed matte asset frame is unavailable",
           "attribution: a content error names the layer's clip");
    check (attributeLayerError (7, "native GPU SDF backend does not support operation blur")
               == "clip 7: native GPU SDF backend does not support operation blur",
           "attribution: a plan error that mentions the GPU still names its clip");
    check (attributeLayerError (7, "clip 3 LUT: bad cube") == "clip 3 LUT: bad cube",
           "attribution: an error that already names a clip is kept");
    check (attributeLayerError (7, "cancelled") == "cancelled", "attribution: cancellation is not a clip error");
    check (attributeLayerError (-1, "bad plan") == "bad plan" && attributeLayerError (7, "") == "",
           "attribution: no layer or no error leaves the text alone");
    // Errors about a clip's own shader, material or source name the clip, even when they
    // mention OpenGL, a readback or an upload.
    for (const char* content : { "physical diffraction OpenGL shader compile failed: 0:12 syntax error",
                                 "physical diffraction OpenGL validation readback rejected shader output",
                                 "physical diffraction Metal validation readback rejected shader output",
                                 "Optical Flow OpenGL linear source mapping failed",
                                 "native surface material execution requires an OpenGL or Metal material program" })
        check (attributeLayerError (7, content) == "clip 7: " + std::string (content),
               "attribution: a content error that mentions OpenGL or a readback names its clip");
    for (const char* environment : { "FBO incomplete", "float HDR lease OpenGL context changed",
                                     "OpenGL context lost", "GPU out of memory",
                                     "export encoder rejected the frame",
                                     "production GPU compositor is busy",
                                     "Optical Flow export has no native source worker" })
        check (attributeLayerError (7, environment) == environment,
               "attribution: a device or helper error stays unnamed");
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
    check (isSetter ("viewport_set_full_canvas_scene") && bumpsItself ("viewport_set_full_canvas_scene"),
           "policy: the full canvas request wakes the viewport only on a real change");

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

// The size the live viewport renders an imported 3D scene at (preview_render_extent.h).
void previewExtentTests()
{
    using videohelper::previewextent::Extent;
    using videohelper::previewextent::fitToPresentation;
    using videohelper::previewextent::zoomBucket;

    // A 4K canvas in the docked panel renders at the panel size, canvas aspect kept.
    check(fitToPresentation(3840, 2160, 424, 240) == Extent { 424, 239 },
          "4K canvas in a 424x240 dock fits to 424x239");
    check(fitToPresentation(3840, 2160, 800, 240) == Extent { 427, 240 },
          "a wide dock is limited by its height");
    check(fitToPresentation(3840, 2160, 240, 800) == Extent { 240, 135 },
          "a tall dock is limited by its width");
    check(fitToPresentation(1080, 1920, 424, 240) == Extent { 135, 240 },
          "a portrait canvas fits inside a landscape dock");
    // Never larger than the canvas.
    check(fitToPresentation(640, 360, 1920, 1080) == Extent { 640, 360 },
          "a canvas smaller than the viewport keeps its size");
    check(fitToPresentation(640, 360, 640, 360) == Extent { 640, 360 },
          "a canvas equal to the viewport keeps its size");
    // Unknown sizes fall back to the canvas, an unknown canvas gives no extent.
    check(fitToPresentation(3840, 2160, 0, 0) == Extent { 3840, 2160 },
          "an unknown viewport gives the canvas");
    check(fitToPresentation(3840, 2160, -5, 240) == Extent { 3840, 2160 },
          "a negative viewport side gives the canvas");
    check(! fitToPresentation(0, 2160, 424, 240).valid() && ! fitToPresentation(3840, 0, 424, 240).valid(),
          "a zero canvas gives no extent");
    // The short side never drops below the floor.
    check(fitToPresentation(3840, 2160, 40, 20) == Extent { 114, 64 },
          "a tiny viewport is held at a 64 px short side");
    check(fitToPresentation(2160, 3840, 20, 40) == Extent { 64, 114 },
          "the floor applies to the short side of a portrait canvas");
    check(fitToPresentation(48, 27, 20, 10) == Extent { 48, 27 },
          "a canvas already below the floor is not enlarged");
    // Whole pixels and a stable result for the same inputs.
    const auto a = fitToPresentation(1920, 1080, 777, 431);
    check(a.valid() && a.width <= 777 && a.height <= 431 && a == fitToPresentation(1920, 1080, 777, 431),
          "the fit stays inside the viewport and repeats exactly");
    // Zoom widens the fit, up to the canvas.
    check(fitToPresentation(3840, 2160, 424, 240, 4) == Extent { 1696, 954 },
          "a 4x zoom bucket renders four times the fitted width");
    check(fitToPresentation(3840, 2160, 424, 240, 32) == Extent { 3840, 2160 },
          "a deep zoom is capped at the canvas");
    check(fitToPresentation(3840, 2160, 424, 240, 0) == Extent { 424, 239 },
          "a zero zoom bucket counts as 1");
    check(zoomBucket(0.1) == 1 && zoomBucket(1.0) == 1 && zoomBucket(1.01) == 2
          && zoomBucket(2.0) == 2 && zoomBucket(3.0) == 4 && zoomBucket(32.0) == 32
          && zoomBucket(1000.0) == 32 && zoomBucket(std::numeric_limits<double>::quiet_NaN()) == 1,
          "zoom buckets are powers of two from 1 to 32");

    // A full canvas request gives the canvas at any panel size or zoom.
    using videohelper::previewextent::sceneExtent;
    check(sceneExtent(1920, 1080, 804, 452, 1, true) == Extent { 1920, 1080 },
          "a full canvas request renders 1080p in the 804x452 dock");
    check(sceneExtent(3840, 2160, 424, 240, 1, true) == Extent { 3840, 2160 },
          "a full canvas request renders 4K in a small dock");
    check(sceneExtent(1920, 1080, 0, 0, 1, true) == Extent { 1920, 1080 }
          && sceneExtent(1920, 1080, 3840, 2160, 4, true) == Extent { 1920, 1080 },
          "a full canvas request ignores the framebuffer and the zoom");
    check(! sceneExtent(0, 1080, 804, 452, 1, true).valid(), "a zero canvas still gives no extent");
    // With no request the extent is exactly the panel fit.
    check(sceneExtent(1920, 1080, 804, 452, 1, false) == Extent { 804, 452 },
          "with no request 1080p in the 804x452 dock stays at 804x452");
    {
        const int cases[][4] = { { 3840, 2160, 424, 240 }, { 3840, 2160, 800, 240 }, { 1080, 1920, 424, 240 },
                                 { 640, 360, 1920, 1080 }, { 3840, 2160, 0, 0 }, { 3840, 2160, 40, 20 },
                                 { 48, 27, 20, 10 }, { 0, 2160, 424, 240 }, { 1920, 1080, 777, 431 } };
        bool same = true;
        for (const auto& c : cases)
            for (const std::uint32_t zoom : { 0u, 1u, 2u, 4u, 32u })
                same = same && sceneExtent(c[0], c[1], c[2], c[3], zoom, false)
                                   == fitToPresentation(c[0], c[1], c[2], c[3], zoom);
        check(same, "with no request every panel, canvas and zoom matches fitToPresentation");
    }

    // The telemetry keeps the last completed extent and forgets it on reset.
    videowire::VisualPlanTelemetry telemetry;
    check(telemetry.snapshot().importedScenePreviewWidth == 0
          && telemetry.snapshot().importedScenePreviewHeight == 0,
          "preview extent is unobserved before any render");
    check(telemetry.recordImportedScenePreviewExtent(424, 239), "preview extent is recorded");
    check(telemetry.snapshot().importedScenePreviewWidth == 424
          && telemetry.snapshot().importedScenePreviewHeight == 239,
          "snapshot carries the preview extent");
    telemetry.resetOwner(0, 0);
    check(telemetry.snapshot().importedScenePreviewWidth == 0,
          "owner reset forgets the preview extent");
}

// The latest caustic degradation reason of an imported scene preview: recorded, shown in
// the JSON the HUD reads, cleared by a frame that renders the caustic as planned, and
// forgotten on reset.
void importedSceneCausticDegradationTests()
{
    const std::string reason = "caustic off: exceeds GL limits (deposit receipts need 125.0 MiB, max SSBO 64.0 MiB)";
    videowire::VisualPlanTelemetry telemetry;
    check(telemetry.snapshot().importedSceneCausticDegradation.empty(),
          "no caustic reason before any preview render");
    auto json = videowire::visualTelemetryJson(telemetry.snapshot());
    check(json.contains("importedSceneCausticDegradation")
          && json["importedSceneCausticDegradation"].is_string()
          && json["importedSceneCausticDegradation"].get<std::string>().empty(),
          "the JSON field is an empty string when nothing degraded");

    check(telemetry.recordImportedSceneCausticDegradation(reason), "caustic reason is recorded");
    check(telemetry.snapshot().importedSceneCausticDegradation == reason,
          "snapshot carries the caustic reason");
    json = videowire::visualTelemetryJson(telemetry.snapshot());
    check(json["importedSceneCausticDegradation"].get<std::string>() == reason,
          "the JSON carries the caustic reason the HUD reads");

    const std::string reduced = "caustic reduced to 32 entry subdivisions: deposit receipts need 70.0 MiB, max SSBO 64.0 MiB";
    check(telemetry.recordImportedSceneCausticDegradation(reduced)
          && telemetry.snapshot().importedSceneCausticDegradation == reduced,
          "a later frame replaces the reason");

    check(telemetry.recordImportedSceneCausticDegradation(std::string())
          && telemetry.snapshot().importedSceneCausticDegradation.empty(),
          "a frame that renders the caustic as planned clears the reason");
    json = videowire::visualTelemetryJson(telemetry.snapshot());
    check(json["importedSceneCausticDegradation"].get<std::string>().empty(),
          "the JSON field is empty again after the clear");

    const std::string oversized(videowire::kMaxCausticDegradationBytes * 4, 'x');
    check(telemetry.recordImportedSceneCausticDegradation(oversized)
          && telemetry.snapshot().importedSceneCausticDegradation.size() == videowire::kMaxCausticDegradationBytes,
          "an oversized reason is cut to the bound");

    check(telemetry.recordImportedSceneCausticDegradation(reason), "caustic reason is recorded again");
    telemetry.resetOwner(0, 0);
    check(telemetry.snapshot().importedSceneCausticDegradation.empty(),
          "owner reset forgets the caustic reason");
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
    check(! transport.zeroCopyReleased()
          && transportTelemetry.snapshot().framesPresented == 0,
          "a release with no observed zero-copy handoff presents nothing");
    check(transport.zeroCopyHandoff(true, true, true), "successful socket handoff");
    transportSnapshot = transportTelemetry.snapshot();
    check(transportSnapshot.zeroCopyFrames == 1
          && transportSnapshot.transportFramesHandedOff == 2
          && transportSnapshot.framesPresented == 0,
          "transport handoff alone leaves presented at zero");
    check(transport.zeroCopyReleased(), "consumer release is recorded");
    transport.exportedBufferReleased(80);
    transport.exportedBufferPoolClosed();
    transportSnapshot = transportTelemetry.snapshot();
    check(transportSnapshot.framesPresented == 1
          && transportSnapshot.transportFramesHandedOff == 2,
          "consumer release of a handed-off buffer counts one presented frame");
    check(transportSnapshot.retainedFramesCurrent == 0
          && transportSnapshot.intermediateImagesCurrent == 0
          && transportSnapshot.retainedBytesCurrent == 0
          && transportSnapshot.retainedFramesPeak == 2,
          "pool teardown returns current resources to zero and preserves peak");

    videowire::VisualPlanTelemetry sceneTelemetry;
    check(! sceneTelemetry.snapshot().importedSceneRender.observed,
          "imported scene render is unobserved before any sample");
    check(sceneTelemetry.recordImportedSceneRender(1'000'000)
          && sceneTelemetry.recordImportedSceneRender(2'000'000),
          "imported scene preview render samples are recorded");
    const auto sceneSnapshot = sceneTelemetry.snapshot();
    check(sceneSnapshot.importedSceneRender.observed
          && sceneSnapshot.importedSceneRender.count == 2
          && sceneSnapshot.importedSceneRender.totalNs == 3'000'000
          && sceneSnapshot.importedSceneRender.movingNs > 1'000'000.0
          && sceneSnapshot.graphEvaluations == 0,
          "imported scene render is separate from the graph timer");
    sceneTelemetry.resetOwner(0, 0);
    check(! sceneTelemetry.snapshot().importedSceneRender.observed,
          "owner reset forgets the imported scene render duration");

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
    frameFailureGovernorTests();
    layerFailureAttributionTests();
    rpcPolicyTests();
    revisionGuardTests();
    previewExtentTests();
    importedSceneCausticDegradationTests();

    std::printf("viewport telemetry owner: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
