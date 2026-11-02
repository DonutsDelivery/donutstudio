// What a failed device pass leaves in the host film, and what the Auto fallback
// (src/pathtrace_fixture_engine.cpp) starts from after it. A fake DevicePass
// stands in for OptiX, so no GPU or SDK is needed. The real pass cannot run here;
// pathtrace_optix_device_source_contract.py pins that it writes the film only
// after its last copy from the device, as the clean fake below does.

#include "pathtrace/devices/auto_device_latch.h"
#include "pathtrace/devices/setup_unwind.h"
#include "pathtrace/engine.h"
#include "support/pathtrace_glass_scene.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace
{

namespace pt = videohelper::pathtrace;
namespace grid = HarmonicMIDI::grid;
namespace glass = pathtraceglassscene;

int checks = 0;
int failures = 0;

void check (bool condition, const std::string& message)
{
    ++checks;
    if (! condition)
    {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

// Far above anything the glass scene reaches, so the image shows how much of the
// film came from the fake device.
constexpr float kMarker = 3000.0f;

struct FakeDevice
{
    enum class Failure
    {
        None,
        Clean,            // returns false with nothing on the film, as OptiX does
        AfterWritingFilm  // writes first, the partial readback OptiX must not do
    };

    Failure failure = Failure::None;
    std::uint32_t samplesWritten = 0;  // per pixel; 0 means the whole pass
    bool reportCancelled = false;
    int calls = 0;
    std::vector<std::uint32_t> firstSamples;

    pt::PathTracer::DevicePassFunction function()
    {
        return [this] (pt::PathTracer::DevicePass& pass) {
            ++calls;
            firstSamples.push_back (pass.firstSample);
            if (failure == Failure::Clean)
            {
                pass.error = "fake device: reading the film back failed";
                return false;
            }
            const std::uint32_t samples = samplesWritten != 0 ? samplesWritten : pass.samples;
            for (std::uint32_t y = 0; y < pass.film.height(); ++y)
                for (std::uint32_t x = 0; x < pass.film.width(); ++x)
                    pass.film.addSampleSum (x, y, { kMarker * static_cast<float> (samples), 0.0f, 0.0f }, samples);
            if (failure == Failure::AfterWritingFilm)
            {
                pass.error = "fake device: failed after writing the film";
                return false;
            }
            pass.stats.samplesAdded += static_cast<std::uint64_t> (samples) * pass.film.width() * pass.film.height();
            pass.stats.cancelled = reportCancelled;
            return true;
        };
    }
};

float halfToFloat (std::uint16_t half)
{
    const int exponent = (half >> 10) & 0x1f;
    const float magnitude = exponent == 0 ? static_cast<float> (half & 0x3ff) * std::ldexp (1.0f, -24)
                                          : static_cast<float> ((half & 0x3ff) | 0x400) * std::ldexp (1.0f, exponent - 25);
    return (half & 0x8000) != 0 ? -magnitude : magnitude;
}

float firstPixelRed (const pt::EngineImage& image)
{
    std::uint16_t red = 0;
    std::memcpy (&red, image.colorBytes.data(), sizeof (red));
    return halfToFloat (red);
}

bool near (float value, float expected)
{
    return std::abs (value - expected) < 0.05f * expected;
}

struct Fixture
{
    std::shared_ptr<const grid::Visual3DScene> scene = glass::makeScene();
    pt::EngineSceneSetup setup = glass::makeSetup (scene);
    pt::EngineView view = glass::makeView (*scene, 4, 4);
};

constexpr pt::EngineColor kLinear = pt::EngineColor::LinearRgba16f;

// ---------------------------------------------------------------- the tracer

pt::PinholeCamera tracerCamera()
{
    pt::PinholeCamera camera;
    camera.width = 2;
    camera.height = 2;
    return camera;
}

void tracerKeepsWhatTheFilmHolds()
{
    const pt::RenderScene scene;
    const pt::CancellationToken cancel;
    const pt::PinholeCamera camera = tracerCamera();

    {
        pt::PathTracer tracer (1);
        FakeDevice device;
        tracer.setDevicePass (device.function());
        device.failure = FakeDevice::Failure::Clean;
        const auto first = tracer.renderProgressive (scene, camera, 2, 2, 2, 7, cancel);
        check (first.deviceFailed && ! tracer.deviceFailure().empty(), "a failed pass is reported with its reason");
        check (tracer.accumulator().sampleCount (0, 0) == 0, "a clean failure on the first pass leaves the film empty");

        device.failure = FakeDevice::Failure::None;
        tracer.renderProgressive (scene, camera, 2, 2, 2, 7, cancel);
        device.failure = FakeDevice::Failure::Clean;
        const auto later = tracer.renderProgressive (scene, camera, 2, 2, 2, 7, cancel);
        check (later.deviceFailed && tracer.accumulator().sampleCount (0, 0) == 2,
               "a clean failure adds nothing, and the earlier good pass stays on the film");
    }
    {
        // The tracer has no rollback, so only the device's own order protects the film.
        pt::PathTracer tracer (1);
        FakeDevice device;
        tracer.setDevicePass (device.function());
        device.failure = FakeDevice::Failure::AfterWritingFilm;
        device.samplesWritten = 1;
        const auto stats = tracer.renderProgressive (scene, camera, 2, 2, 2, 7, cancel);
        check (stats.deviceFailed && tracer.accumulator().sampleCount (0, 0) == 1,
               "a device that writes before failing leaves those samples: the tracer does not undo them");
    }
    {
        pt::RenderScene framed;
        framed.frameBindings.push_back ({ 1, 0 });
        pt::PathTracer tracer (1);
        FakeDevice device;
        tracer.setDevicePass (device.function());
        const auto stats = tracer.renderProgressive (framed, camera, 2, 2, 2, 7, cancel);
        check (stats.deviceFailed && device.calls == 0 && tracer.accumulator().sampleCount (0, 0) == 0,
               "a frame-tinted scene fails before the device runs, with nothing on the film");
    }
}

// ---------------------------------------------------------------- the Auto fallback

// The fixture engine's steps when a pass fails under Auto.
bool fallBackToCpu (pt::SceneEngine& engine, const Fixture& f, pt::EngineImage& image, std::string& error)
{
    engine.setDevicePass ({});
    engine.restartPreview();
    image = {};
    error.clear();
    return engine.renderPreview (f.setup, f.view, kLinear, image, error);
}

void fallbackAfterGoodPassesStartsEmpty()
{
    Fixture f;
    pt::SceneEngine engine (1);
    FakeDevice device;
    engine.setDevicePass (device.function());
    pt::EngineImage image;
    std::string error;

    check (engine.renderPreview (f.setup, f.view, kLinear, image, error), "first good pass: " + error);
    check (engine.renderPreview (f.setup, f.view, kLinear, image, error), "second good pass: " + error);
    check (near (firstPixelRed (image), kMarker), "two device passes show the marker");

    device.failure = FakeDevice::Failure::Clean;
    check (! engine.renderPreview (f.setup, f.view, kLinear, image, error) && image.stats.deviceFailed,
           "the third pass fails");
    check (engine.previewSampleCount() == 2, "the failed pass is not counted");

    check (fallBackToCpu (engine, f, image, error), "the CPU takes over: " + error);
    check (image.restarted && image.samplesPerPixel == 1, "the fallback starts a new accumulation");
    check (firstPixelRed (image) < kMarker * 0.01f,
           "none of the device's samples stay under the CPU's, got " + std::to_string (firstPixelRed (image)));
}

void fallbackOnFirstPassStartsAtSampleZero()
{
    Fixture f;
    pt::SceneEngine engine (1);
    FakeDevice failing;
    failing.failure = FakeDevice::Failure::Clean;
    engine.setDevicePass (failing.function());
    pt::EngineImage image;
    std::string error;
    check (! engine.renderPreview (f.setup, f.view, kLinear, image, error), "the first pass fails");

    // A second fake in the CPU's place, to read the index the retry starts from.
    FakeDevice retry;
    engine.setDevicePass (retry.function());
    engine.restartPreview();
    image = {};
    check (engine.renderPreview (f.setup, f.view, kLinear, image, error), "the retry renders: " + error);
    check (image.samplesPerPixel == 1 && near (firstPixelRed (image), kMarker),
           "nothing from the failed pass is on the film");
    check (retry.firstSamples == std::vector<std::uint32_t> { 0 }, "the retry draws from sample 0");
    check (image.restarted, "the retry counts as a restart");
}

void frameTintedSceneLeavesTheDeviceForTheNext()
{
    const pt::CancellationToken cancel;
    const pt::PinholeCamera camera = tracerCamera();
    pt::RenderScene framed;
    framed.frameBindings.push_back ({ 1, 0 });
    const pt::RenderScene plain;

    pt::PathTracer tracer (1);
    FakeDevice device;
    tracer.setDevicePass (device.function());
    tracer.setDeviceSceneFallback (true);
    const auto onCpu = tracer.renderProgressive (framed, camera, 2, 2, 1, 7, cancel);
    check (! onCpu.deviceFailed && onCpu.deviceSkipped != nullptr && device.calls == 0,
           "under Auto a frame-tinted scene runs on the CPU and says why");
    check (tracer.accumulator().sampleCount (0, 0) == 1, "the CPU drew it");
    const auto next = tracer.renderProgressive (plain, camera, 2, 2, 1, 8, cancel);
    check (next.deviceSkipped == nullptr && device.calls == 1, "the next scene is back on the device");
    check (tracer.accumulator().sampleCount (0, 0) == 1, "and starts on its own film");
}

void exportFallbackStartsClean()
{
    Fixture f;
    pt::EngineImage clean;
    std::string error;
    {
        pt::SceneEngine engine (1);
        check (engine.renderExport (f.setup, f.view, kLinear, 2, clean, error), "the CPU export renders: " + error);
    }

    pt::SceneEngine engine (1);
    FakeDevice device;
    device.failure = FakeDevice::Failure::AfterWritingFilm;
    engine.setDevicePass (device.function());
    pt::EngineImage image;
    check (! engine.renderExport (f.setup, f.view, kLinear, 2, image, error) && image.stats.deviceFailed,
           "the device export fails");
    engine.setDevicePass ({});
    image = {};
    error.clear();
    check (engine.renderExport (f.setup, f.view, kLinear, 2, image, error), "the CPU takes the export: " + error);
    check (image.colorBytes == clean.colorBytes, "even a partial write is gone: the retried export matches a CPU export bit for bit");
}

void cancelledPartialPassIsDroppedByTheNextCall()
{
    Fixture f;
    pt::SceneEngine engine (1);
    engine.samplesPerPass = 2;
    FakeDevice device;
    device.samplesWritten = 1;
    device.reportCancelled = true;
    engine.setDevicePass (device.function());
    pt::EngineImage image;
    std::string error;
    check (engine.renderPreview (f.setup, f.view, kLinear, image, error) && image.stats.cancelled,
           "a cancelled pass is not an error: " + error);
    check (engine.previewSampleCount() == 2, "the engine counts the whole cancelled pass, not the one sample it holds");

    FakeDevice next;
    engine.setDevicePass (next.function());
    image = {};
    check (engine.renderPreview (f.setup, f.view, kLinear, image, error), "the next preview renders: " + error);
    check (image.restarted && image.samplesPerPixel == 2 && next.firstSamples == std::vector<std::uint32_t> { 0 },
           "the call after a cancel starts over, so the partial pass is gone");
}

// A stand-in setup: one context, memory in it, and a failure at any step. The real
// setup is static inside the SDK-only class, so this covers the unwind it relies on
// and the source contract pins its use there.
void failedSetupLeavesNothingForTheRetry()
{
    struct Setup
    {
        int contexts = 0;
        int contextsEverCreated = 0;
        int allocations = 0;
        bool published = false;
        std::vector<std::string> releases;

        bool build (pt::device::SetupUnwind& undo, int failAtStep)
        {
            ++contexts;
            ++contextsEverCreated;
            undo.add ([this] { --contexts; releases.push_back ("context"); });
            undo.add ([this] { allocations = 0; releases.push_back ("memory"); });
            if (failAtStep == 1) return false;
            allocations = 3;
            if (failAtStep == 2) return false;
            published = true;
            return true;
        }

        bool ensure (int failAtStep)
        {
            pt::device::SetupUnwind undo;
            const bool built = build (undo, failAtStep);
            if (built)
                undo.commit();
            return built;
        }
    };

    Setup setup;
    check (! setup.ensure (2) && ! setup.published, "a failed setup publishes nothing");
    check (setup.contexts == 0 && setup.allocations == 0, "a failed setup leaves no context and no memory");
    check ((setup.releases == std::vector<std::string> { "memory", "context" }), "memory goes before its context");
    check (! setup.ensure (1) && setup.contexts == 0, "a second failure leaves nothing either");
    check (setup.ensure (0) && setup.published, "the retry finishes");
    check (setup.contexts == 1 && setup.contextsEverCreated == 3 && setup.allocations == 3,
           "the finished setup holds one context and only its own memory");
}

void autoFailureOutlivesEngineRebuilds()
{
    pt::device::AutoDeviceLatch latch;
    int setups = 0;
    // What each new Path Trace engine does on its first draw.
    const auto engineDraws = [&] {
        latch.noteEngine (0, 34, true);
        if (latch.failure() == nullptr)
        {
            ++setups;
            latch.fail (std::make_shared<const std::string> ("setup failed"));
        }
    };
    for (int rebuild = 0; rebuild < 7; ++rebuild)
        engineDraws();
    check (setups == 1, "seven engine rebuilds run one failed setup");

    latch.noteEngine (0, 35, false);
    latch.noteEngine (1, 34, false);
    check (latch.failure() != nullptr, "a node that never left Path Trace releases nothing");

    latch.noteEngine (0, 34, false);
    check (latch.failure() == nullptr, "moving the failed node off Path Trace releases the failure");
    latch.noteEngine (0, 34, false);
    engineDraws();
    check (setups == 2, "the switch back runs one new setup");
    engineDraws();
    check (setups == 2, "a failed retry holds again");
}

} // namespace

// A kernel still compiling holds a preview at no samples. A scene that stays on the CPU
// never asks, and a finished compile starts the film at its first sample.
void previewWaitsForTheDeviceOnlyWhenItWillUseIt()
{
    Fixture f;
    pt::SceneEngine engine (1);
    FakeDevice device;
    bool ready = false;
    int asked = 0;
    engine.setDevicePass (device.function(), [&] { ++asked; return ready; });
    engine.setDeviceSceneFallback (true);
    pt::EngineImage image;
    std::string error;
    check (engine.renderPreview (f.setup, f.view, kLinear, image, error) && image.pending, "a compiling kernel holds the preview: " + error);
    check (engine.previewSampleCount() == 0 && device.calls == 0, "a held preview traces nothing");
    ready = true;
    check (engine.renderPreview (f.setup, f.view, kLinear, image, error) && ! image.pending, "a finished kernel renders: " + error);
    check (image.restarted && image.samplesPerPixel == 1 && device.calls == 1, "the film starts at its first sample");

    auto tinted = f.setup;
    tinted.objectFrames.push_back ({ 0, 0 });
    ready = false;
    asked = 0;
    check (engine.renderPreview (tinted, f.view, kLinear, image, error) && ! image.pending, "a frame-tinted scene doesn't wait: " + error);
    check (asked == 0 && image.stats.deviceSkipped != nullptr, "it goes to the CPU without asking the device");
}

int main()
{
    previewWaitsForTheDeviceOnlyWhenItWillUseIt();
    tracerKeepsWhatTheFilmHolds();
    fallbackAfterGoodPassesStartsEmpty();
    fallbackOnFirstPassStartsAtSampleZero();
    frameTintedSceneLeavesTheDeviceForTheNext();
    exportFallbackStartsClean();
    cancelledPartialPassIsDroppedByTheNextCall();
    failedSetupLeavesNothingForTheRetry();
    autoFailureOutlivesEngineRebuilds();
    std::cout << checks - failures << "/" << checks << " device failure checks passed\n";
    return failures == 0 ? 0 : 1;
}
