// The Raster stand-in for a Path Trace Preview whose device kernel is still building.
//
// CPU cases: PathTraceStandInScope asks the registered probe only for an eligible frame
// and nests, and KernelBuildRetryWatch lets the viewport retry refused frames once when
// the build behind a stand-in ends. A background warm-up's build that the viewport joined
// wakes a settled Preview when it ends.
//
// The stand-in ships on. Turned off (setPathTraceRasterStandIn, or
// ARBIT_PATH_TRACE_RASTER_STAND_IN=0), a Path Trace Preview goes to the path tracer and is
// held while the kernel builds. The cases below check both.
//
// GL cases: the lighthouse pair of dielectrics (a closed glass volume beside one thin
// water interface, as render_mode_preview_parity_tests.cpp builds it) drawn through
// FixtureSceneRenderer. A fake path tracer's kernel build runs on a real BuildGate that
// the test releases. While it runs, a Path Trace Preview has exactly the Color bytes of
// the Raster-selected Preview, with the renderer deciding by itself or inside a scope,
// and an export still goes to the engine. Once the gate completes, the next frame goes to
// the engine. A frame whose inputs were built under a stand-in decision stays Raster's
// even when the build ends before it renders. The GL cases skip with 77 when no hardware
// OpenGL 4.3 context exists.

#include "../src/fixture_scene_renderer.h"
#include "../src/viewport_publish_gate.h"
#include "../src/pathtrace/devices/kernel_build_gate.h"
#include "../src/pathtrace/devices/kernel_compile_state.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#if ARBIT_KERNEL_STAND_IN_GL
#include "../src/gpu_backend/backend.h"
#include "../src/gl_loader.h"
#include "support/glass_caustic_oracle.h"
#include <GLFW/glfw3.h>
#include <GL/gl.h>
#endif

namespace
{
namespace fixture = videorender::fixture3d;
namespace kernelcompile = videohelper::pathtrace::kernelcompile;

int failures = 0, checks = 0;
void check (bool condition, const std::string& text)
{
    ++checks;
    if (! condition)
    {
        ++failures;
        std::fprintf (stderr, "FAIL: %s\n", text.c_str());
    }
}

// ---------------------------------------------------------------- fake kernel build

kernelcompile::BuildGate gate;
std::promise<void> release;
std::shared_future<void> released = release.get_future().share();
std::atomic<int> probeCalls { 0 };

bool build (std::string&)
{
    released.wait();
    return true;
}

// The device's readiness call: it starts the build and says whether it runs.
bool fakeKernelBuilding() noexcept
{
    ++probeCalls;
    gate.start (&build);
    return gate.status() == kernelcompile::BuildGate::Status::Running;
}

bool fixedProbeAnswer = false;
bool fixedProbe() noexcept
{
    ++probeCalls;
    return fixedProbeAnswer;
}

struct FakeEngineState
{
    int renders = 0;
    int primes = 0;
    fixture::RenderUse lastUse = fixture::RenderUse::Preview;
};
FakeEngineState engineState;

class FakeEngine final : public fixture::SceneRenderEngine
{
public:
    bool render (const fixture::SceneEngineDraw& draw, fixture::RenderedFrame& output, std::string&) override
    {
        ++engineState.renders;
        engineState.lastUse = draw.use;
        output = {};
        output.pending = true;
        return true;
    }

    void cancel() noexcept override {}
    bool kernelBuilding() noexcept override { return fakeKernelBuilding(); }
    void primeWhileKernelBuilds (const fixture::SceneEngineDraw&) noexcept override { ++engineState.primes; }
};

std::shared_ptr<fixture::SceneRenderEngine> makeFakeEngine() { return std::make_shared<FakeEngine>(); }

// ---------------------------------------------------------------- CPU cases

void scopeTests()
{
    fixture::setPathTraceKernelBuildingProbe (&fixedProbe);
    probeCalls = 0;
    fixedProbeAnswer = true;
    if (std::getenv ("ARBIT_PATH_TRACE_RASTER_STAND_IN") == nullptr)
    {
        const fixture::PathTraceStandInScope shipped (true);
        check (shipped.standIn() && probeCalls == 1, "the stand-in ships on: an eligible frame stands in while the kernel builds");
        probeCalls = 0;
    }
    fixture::setPathTraceRasterStandIn (false);
    {
        const fixture::PathTraceStandInScope off (true);
        check (! off.standIn() && probeCalls == 0, "with the stand-in off nothing stands in");
    }
    fixture::setPathTraceRasterStandIn (true);
    {
        const fixture::PathTraceStandInScope ineligible (false);
        check (! ineligible.standIn() && probeCalls == 0, "an ineligible frame is no stand-in and asks no device");
    }
    {
        const fixture::PathTraceStandInScope outer (true);
        check (outer.standIn() && probeCalls == 1, "an eligible frame asks the probe once and stands in while it builds");
        fixedProbeAnswer = false;
        {
            const fixture::PathTraceStandInScope inner (true);
            check (! inner.standIn(), "a nested scope takes its own answer");
        }
    }
    fixture::setPathTraceKernelBuildingProbe (nullptr);
    {
        const fixture::PathTraceStandInScope noTracer (true);
        check (! noTracer.standIn(), "without a registered probe nothing stands in");
    }
}

void retryWatchTests()
{
    arbitviewport::KernelBuildRetryWatch watch;
    check (! watch.buildEnded (false, 0, false), "no stand-in and no build asks for nothing");
    check (! watch.buildEnded (true, 100, false), "a stand-in while its build runs waits");
    check (! watch.buildEnded (true, 100, false), "the same build still running waits");
    check (watch.buildEnded (true, 0, false), "the build behind the stand-in ended: retry once");
    check (! watch.buildEnded (true, 0, false), "and only once");

    arbitviewport::KernelBuildRetryWatch cancelled;
    cancelled.buildEnded (true, 200, false);
    check (! cancelled.buildEnded (true, 0, true), "a cancelled build leaves nothing to retry for");
    check (! cancelled.buildEnded (true, 0, true), "nor later");

    arbitviewport::KernelBuildRetryWatch swapped;
    swapped.buildEnded (true, 300, false);
    check (swapped.buildEnded (false, 0, false), "a build that ends after the stand-in was replaced still retries");

    // The governor forgets its failures on that answer, so a frame it stopped retrying runs again.
    arbitviewport::FrameFailureGovernor governor;
    arbitviewport::FrameFailureGovernor::Key key;
    for (int attempt = 0; attempt < arbitviewport::FrameFailureGovernor::kMaxAttempts; ++attempt)
        governor.noteFailure (key, 0);
    check (governor.shouldHold (key, 1000000), "four refused stand-in frames stop the retries");
    governor.noteSuccess();
    check (! governor.shouldHold (key, 1000000), "the build's end lets the same frame run again");
}

// A background warm-up starts the build, the viewport joins it, and the warm-up ends. The
// live cold run of 7 Nov stuck here: the last stand-in frame had settled, so the idle gate
// skipped every tick and nothing asked the device whether the kernel was ready.
void joinedWarmUpTests()
{
    kernelcompile::BuildGate warmGate;
    std::promise<void> finish;
    std::shared_future<void> finished = finish.get_future().share();
    bool claimedHook = false;
    // The warm-up's build: it compiles, then keeps what it made only when someone joined.
    warmGate.start ([&warmGate, finished] (std::string&)
                    {
                        finished.wait();
                        warmGate.keepWarmUp();
                        return true;
                    },
                    kernelcompile::BuildGate::Origin::WarmUp, [&claimedHook] { claimedHook = true; });
    check (warmGate.status() == kernelcompile::BuildGate::Status::Running, "the warm-up runs");

    arbitviewport::PublishGate publish;
    arbitviewport::PublishGate::Snapshot snapshot;
    arbitviewport::FrameFailureGovernor governor;
    arbitviewport::FrameFailureGovernor::Key key;
    arbitviewport::KernelBuildRetryWatch watch;
    // The viewport picks Path Trace: its readiness call joins the running build, as
    // OptixRenderer::pipelineReady starts it as a caller.
    warmGate.start ([] (std::string&) { return false; });
    check (warmGate.claimed() && claimedHook, "the viewport joined the warm-up's build");
    check (publish.shouldProduce (snapshot), "the first Preview frame is produced");
    publish.finish (snapshot, true, true, false);
    check (! publish.shouldProduce (snapshot), "a settled stand-in frame leaves the idle gate quiet");
    for (int attempt = 0; attempt < arbitviewport::FrameFailureGovernor::kMaxAttempts; ++attempt)
        governor.noteFailure (key, 0);
    for (int tick = 0; tick < 3; ++tick)
        check (! arbitviewport::wakeForKernelBuildEnd (watch, governor, publish, true,
                                                        kernelcompile::runningBuildStartNs().load(), false),
               "nothing wakes while the joined build runs");
    check (! publish.shouldProduce (snapshot), "the gate stays quiet during the build");

    finish.set_value();
    for (int wait = 0; wait < 2000 && warmGate.status() == kernelcompile::BuildGate::Status::Running; ++wait)
        std::this_thread::sleep_for (std::chrono::milliseconds (1));
    check (warmGate.status() == kernelcompile::BuildGate::Status::Ready,
           "the joined warm-up ends ready, not released, so the device has a kernel");
    check (kernelcompile::runningBuildStartNs().load() == 0, "no build runs after it");
    check (! publish.shouldProduce (snapshot), "without the wake the idle gate would never produce again");
    // The loop's tick, as it runs before the idle gate. The waiting flag may already read
    // false (the stand-in flag clears with the build), and the wake still comes.
    check (arbitviewport::wakeForKernelBuildEnd (watch, governor, publish, false,
                                                 kernelcompile::runningBuildStartNs().load(), false),
           "the build's end wakes the Preview");
    check (! governor.shouldHold (key, 1000000), "the governor lets the refused frame run again");
    check (publish.shouldProduce (snapshot), "the idle gate produces the swap frame");
    check (! arbitviewport::wakeForKernelBuildEnd (watch, governor, publish, false, 0, false), "and wakes once");
}

#if ARBIT_KERNEL_STAND_IN_GL
// ---------------------------------------------------------------- GL cases

namespace gt = glasscaustictests;
using namespace HarmonicMIDI::grid;

// The ordinary caustic fixture with a thin water plane at z=-1.5 between the glass and
// the receiver: a closed glass volume and one thin water interface, the lighthouse pair.
gt::Fixture lighthouse()
{
    auto f = gt::ordinaryFixture();
    const auto first = f.scene->vertexCount;
    const auto o = f.scene->objectCount;
    videohelper::tests::smoothglassfixture::receiver (*f.scene, -6, { 0, 0, 0 });
    for (std::size_t v = first; v < f.scene->vertexCount; ++v)
        f.scene->vertices[v].position.z = -1.5f;
    auto p = std::make_shared<gt::Program>();
    p->backend = arbitgpu::NativeFixtureMaterialBackend::OpenGl;
    p->object = f.scene->objects[o].id;
    p->bindingDigest = "stand-in-fixture-binding-water";
    p->programIdentity = "stand-in-fixture-program-water";
    p->authoredScene = f.scene->id;
    p->authoredMaterial = f.scene->objects[o].material;
    p->sceneRevision = p->structuralRevision = p->evaluationRevision = p->materialRevision = 1;
    p->parameters.identifiers[0] = p->authoredMaterial.value;
    p->parameters.identifiers[1] = std::uint32_t (surfacematerial::TransportMode::ThinInterface);
    p->parameters.normalOpacity = { 0, 0, 1, 1 };
    p->parameters.baseColorMetallic = { 1, 1, 1, 0 };
    p->parameters.transmissionIorClearcoat = { 1, 1.33f, 0, 0 };
    p->parameters.emissionRoughness = { 0, 0, 0, 0 };
    f.program->objectPrograms.push_back (p);
    return f;
}

struct Scene
{
    std::shared_ptr<const Visual3DScene> snapshot;
    std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram> program;
};

constexpr fixture::RenderDimensions kExtent { 32, 32 };

arbitgpu::NativeFixtureSceneRuntimeInputs inputs()
{
    arbitgpu::NativeFixtureSceneRuntimeInputs runtime;
    runtime.linearColor = true;
    return runtime;
}

// Preview frames until the fixture program has linked and a frame comes back, then its Color.
bool settle (fixture::FixtureSceneRenderer& renderer, const Scene& scene, bool scoped, std::vector<std::uint8_t>& bytes,
             std::string& error)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds (90);
    while (std::chrono::steady_clock::now() < deadline)
    {
        fixture::RenderedFrame frame;
        bool rendered = false;
        if (scoped)
        {
            const fixture::PathTraceStandInScope scope (true);
            rendered = renderer.renderGeometry (scene.snapshot, scene.program, inputs(), kExtent, fixture::RenderUse::Preview,
                                                fixture::kNativeGpuCapability, frame, error);
        }
        else
            rendered = renderer.renderGeometry (scene.snapshot, scene.program, inputs(), kExtent, fixture::RenderUse::Preview,
                                                fixture::kNativeGpuCapability, frame, error);
        if (! rendered)
            return false;
        glFinish();
        if (! frame.pending && frame.nativeFrame)
        {
            arbitgpu::NativeRawPassPixels pixels;
            if (! frame.nativeFrame->readRawPass (renderpassoutput::Output::Color, pixels, error))
                return false;
            bytes = std::move (pixels.bytes);
            return true;
        }
        std::this_thread::sleep_for (std::chrono::milliseconds (20));
    }
    error = "the Preview stayed pending for 90 s";
    return false;
}

std::string differing (const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b)
{
    if (a.size() != b.size())
        return std::to_string (a.size()) + " bytes against " + std::to_string (b.size());
    std::size_t count = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        count += a[i] != b[i] ? 1u : 0u;
    return std::to_string (count) + " of " + std::to_string (a.size()) + " bytes differ";
}

bool waitForGate()
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds (10);
    while (gate.status() == kernelcompile::BuildGate::Status::Running && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for (std::chrono::milliseconds (5));
    return gate.status() == kernelcompile::BuildGate::Status::Ready;
}

void glTests (arbitgpu::NativeFixtureSceneBackend& backend)
{
    fixture::setPathTraceSceneEngineFactory (&makeFakeEngine);
    fixture::setPathTraceKernelBuildingProbe (&fakeKernelBuilding);
    probeCalls = 0;
    const auto built = lighthouse();
    const Scene scene { built.scene, built.program };
    std::string error;

    std::vector<std::uint8_t> raster;
    {
        fixture::FixtureSceneRenderer renderer (backend);
        renderer.selectEngine ({});
        check (settle (renderer, scene, false, raster, error) && ! raster.empty(),
               "the lighthouse pair draws with Raster selected: " + error);
    }
    check (engineState.renders == 0 && probeCalls == 0, "Raster selected asks no path tracer and no device");

    fixture::SceneEngineSelection pathTrace;
    pathTrace.engine = fixture::SceneEngineKind::PathTrace;
    fixture::FixtureSceneRenderer renderer (backend);
    renderer.selectEngine (pathTrace);

    // Turned off: no stand-in, the Preview waits for the kernel under the notice.
    fixture::setPathTraceRasterStandIn (false);
    {
        fixture::RenderedFrame held;
        error.clear();
        check (renderer.renderGeometry (scene.snapshot, scene.program, inputs(), kExtent, fixture::RenderUse::Preview,
                                        fixture::kNativeGpuCapability, held, error)
                   && held.pending && engineState.renders == 1 && ! kernelcompile::previewStandIn().load(),
               "with the stand-in off the Path Trace Preview goes to the path tracer and is held: " + error);
        check (gate.status() == kernelcompile::BuildGate::Status::Running, "and its readiness call started the build");
        engineState = {};
    }
    fixture::setPathTraceRasterStandIn (true);

    std::vector<std::uint8_t> standIn;
    error.clear();
    check (settle (renderer, scene, false, standIn, error), "the Path Trace Preview draws while its kernel builds: " + error);
    check (gate.status() == kernelcompile::BuildGate::Status::Running, "the kernel build still runs");
    check (standIn == raster, "the stand-in has the Raster-selected Color bytes: " + differing (standIn, raster));
    check (engineState.renders == 0 && engineState.primes >= 1,
           "the stand-in draws no Path Trace frame and hands the engine its scene");
    check (fixture::pathTraceStandInOnThisThread() && kernelcompile::previewStandIn().load(),
           "the frame reports itself as the stand-in");

    std::vector<std::uint8_t> scopedStandIn;
    error.clear();
    check (settle (renderer, scene, true, scopedStandIn, error) && scopedStandIn == raster,
           "inside a scope the stand-in has the Raster-selected Color bytes: " + error + differing (scopedStandIn, raster));

    {
        fixture::RenderedFrame exported;
        error.clear();
        check (renderer.renderGeometry (scene.snapshot, scene.program, inputs(), kExtent, fixture::RenderUse::Export,
                                        fixture::kNativeGpuCapability, exported, error)
                   && engineState.renders == 1 && engineState.lastUse == fixture::RenderUse::Export,
               "an export goes to the path tracer and waits for its kernel: " + error);
    }

    // The caller built this frame's inputs for the stand-in, then the build ended.
    {
        const fixture::PathTraceStandInScope decided (true);
        check (decided.standIn(), "the scope decides the stand-in while the build runs");
        release.set_value();
        check (waitForGate(), "the fake build gate completes");
        fixture::RenderedFrame frame;
        error.clear();
        check (renderer.renderGeometry (scene.snapshot, scene.program, inputs(), kExtent, fixture::RenderUse::Preview,
                                        fixture::kNativeGpuCapability, frame, error)
                   && engineState.renders == 1,
               "a frame decided as the stand-in stays Raster's when the build ends before it draws: " + error);
    }

    fixture::RenderedFrame swapped;
    error.clear();
    {
        const fixture::PathTraceStandInScope next (true);
        check (! next.standIn(), "the next frame's scope finds the kernel ready");
        check (renderer.renderGeometry (scene.snapshot, scene.program, inputs(), kExtent, fixture::RenderUse::Preview,
                                        fixture::kNativeGpuCapability, swapped, error)
                   && engineState.renders == 2 && engineState.lastUse == fixture::RenderUse::Preview && swapped.pending,
               "the next frame swaps to Path Trace: " + error);
    }
    check (! fixture::pathTraceStandInOnThisThread() && ! kernelcompile::previewStandIn().load(),
           "the swapped frame no longer reports a stand-in");
    fixture::RenderedFrame unscoped;
    check (renderer.renderGeometry (scene.snapshot, scene.program, inputs(), kExtent, fixture::RenderUse::Preview,
                                    fixture::kNativeGpuCapability, unscoped, error)
               && engineState.renders == 3,
           "a frame outside any scope swaps too");
    fixture::setPathTraceSceneEngineFactory (nullptr);
    fixture::setPathTraceKernelBuildingProbe (nullptr);
}

int runGl()
{
    if (! glfwInit())
    {
        std::fprintf (stderr, "SKIP: GLFW initialization failed\n");
        return 77;
    }
    glfwWindowHint (GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    auto* window = glfwCreateWindow (64, 64, "pathtrace-kernel-stand-in", nullptr, nullptr);
    if (window == nullptr)
    {
        std::fprintf (stderr, "SKIP: hidden OpenGL 4.3 core context creation failed\n");
        glfwTerminate();
        return 77;
    }
    glfwMakeContextCurrent (window);
    int result = 0;
    arbitgl::GlFuncs gl;
    std::string missing;
    const auto* rendererRaw = reinterpret_cast<const char*> (glGetString (GL_RENDERER));
    std::string name = rendererRaw != nullptr ? rendererRaw : "";
    for (auto& c : name)
        if (c >= 'A' && c <= 'Z')
            c = static_cast<char> (c - 'A' + 'a');
    const bool software = name.find ("llvmpipe") != std::string::npos || name.find ("softpipe") != std::string::npos
        || name.find ("lavapipe") != std::string::npos || name.find ("swr") != std::string::npos;
    if (! arbitgl::loadGlFunctions (gl, missing))
    {
        std::fprintf (stderr, "FAIL: OpenGL function loading failed: %s\n", missing.c_str());
        result = 1;
    }
    else if (software && std::getenv ("ARBIT_GL_TESTS_ALLOW_SOFTWARE") == nullptr)
    {
        std::fprintf (stderr, "SKIP: software OpenGL renderer %s\n", name.c_str());
        result = 77;
    }
    else
        glTests (arbitgpu::nativeFixtureSceneBackend());
    glfwDestroyWindow (window);
    glfwTerminate();
    return result;
}
#endif
} // namespace

int main()
{
    scopeTests();
    retryWatchTests();
    joinedWarmUpTests();
#if ARBIT_KERNEL_STAND_IN_GL
    const int gl = runGl();
    if (gl == 77 && failures == 0)
        return 77;
    if (gl != 0 && gl != 77)
        return gl;
#endif
    if (failures != 0)
    {
        std::fprintf (stderr, "%d of %d kernel stand-in checks failed\n", failures, checks);
        return 1;
    }
    std::printf ("%d kernel stand-in checks passed\n", checks);
    return 0;
}
