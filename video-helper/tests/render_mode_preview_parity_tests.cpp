// Tier K S2: an Accurate Preview frame equals the export frame once it settles.
// A Fast Preview keeps the reduced Preview budgets it always drew with.
//
// One source, two executables. The CPU build (no ARBIT_RENDER_MODE_PARITY_GL)
// pins the caustic plan and the budget scope, and compares Path Trace preview
// accumulation with export at the same sample count. The GL build renders the
// ordinary closed-glass caustic and the lighthouse (water beside a closed glass
// volume) once for export and then as Accurate Preview frames until the fenced
// caustic is adopted, and compares the Color bytes. It skips with 77 when no
// hardware OpenGL 4.3 context exists.

#include "../src/gpu_backend/fixture_preview_budget.h"
#include "../src/gpu_backend/fixture_glass_caustics_opengl.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#if ARBIT_RENDER_MODE_PARITY_GL
#include "../src/gpu_backend/backend.h"
#include "../src/gpu_backend/fixture_glass_overdraw_probe.h"
#include "../src/fixture_scene_renderer.h"
#include "../src/gl_loader.h"
#include "support/glass_caustic_oracle.h"
#include "support/authored_rough_caustic_renderer_cases.h"
#include <GLFW/glfw3.h>
#include <GL/gl.h>
#else
#include "pathtrace/engine.h"
#include "support/pathtrace_glass_scene.h"
#endif

namespace
{
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

namespace budget = arbitgpu::previewbudget;

#if ! ARBIT_RENDER_MODE_PARITY_GL
namespace gc = arbitgpu::glasscaustics;
namespace pt = videohelper::pathtrace;
namespace glass = pathtraceglassscene;

// The plan each draw class takes in each mode (tier K S4). The budget in
// scope picks it: outside any scope a thread holds the Preview budgets, which
// is a Fast draw, and under Export budgets - the default path - the export
// plan is today's plan exactly.
void causticPlanTests()
{
    using Use = arbitgpu::OpenGlCausticRenderUse;
    const auto fast = arbitgpu::openGlCausticRenderPlan (Use::Preview, 424, 240);
    const auto fastExport = arbitgpu::openGlCausticRenderPlan (Use::Export, 424, 240);
    check (fast.scheduling.entrySubdivisions == arbitgpu::kPreviewCausticEntrySubdivisions
               && fast.scheduling.version == gc::kScheduleVersion,
           "a Fast Preview keeps the coarser Preview entry schedule");
    check (fastExport.scheduling.version == fast.scheduling.version
               && fastExport.scheduling.entrySubdivisions == fast.scheduling.entrySubdivisions
               && fastExport.hostReceipt == fast.hostReceipt && fastExport.stageFences == fast.stageFences
               && fastExport.bandFences == fast.bandFences && fastExport.reuseSpectrum == fast.reuseSpectrum,
           "a Fast export takes the same Preview plan as a Fast Preview (mode symmetry)");
    for (const auto* plan : { &fast, &fastExport })
        check (! plan->hostReceipt && ! plan->stageFences && ! plan->bandFences && plan->reuseSpectrum,
               "both Fast draws skip the host receipt and the stage and band waits");
    const budget::Scope exportBudgets (budget::Budget::Export);
    const auto exporting = arbitgpu::openGlCausticRenderPlan (Use::Export, 424, 240);
    const auto accurate = arbitgpu::openGlCausticRenderPlan (Use::Preview, 424, 240, true);
    check (exporting.scheduling.entrySubdivisions == gc::kEntrySubdivisions && exporting.hostReceipt
               && exporting.stageFences && exporting.bandFences && ! exporting.reuseSpectrum,
           "the default export keeps the full schedule, the fences and the host receipt");
    check (accurate.scheduling.entrySubdivisions == exporting.scheduling.entrySubdivisions
               && accurate.scheduling.version == exporting.scheduling.version,
           "an Accurate Preview traces the export entry schedule");
    check (! accurate.hostReceipt && ! accurate.stageFences && ! accurate.bandFences && accurate.reuseSpectrum,
           "an Accurate Preview skips the host receipt and the stage and band waits");
    const auto large = arbitgpu::openGlCausticRenderPlan (Use::Preview, 3840, 2160, true);
    check (large.bandFences && ! large.reuseSpectrum && ! large.hostReceipt,
           "an Accurate 4K Preview keeps the Preview band waits and reuse limit");
    check (arbitgpu::openGlCausticRenderPlan (Use::Export, 424, 240, true).scheduling.entrySubdivisions
               == gc::kEntrySubdivisions,
           "the default export plan ignores the mode flag");
}

void budgetScopeTests()
{
    check (! budget::exportBudgets() && budget::currentBudget() == budget::Budget::Preview,
           "a thread starts with the Preview budgets, which a draw outside any scope takes");
    {
        const budget::Scope accurate (budget::Budget::Export);
        check (budget::exportBudgets(), "an Accurate scope selects the export budgets");
        {
            const budget::Scope fast (budget::Budget::Preview);
            check (! budget::exportBudgets(), "a nested Fast scope selects the Preview budgets");
        }
        check (budget::exportBudgets(), "closing a nested scope restores the outer budget");
    }
    check (! budget::exportBudgets(), "closing the scope restores the Preview budgets");
}

// Path Trace: the Accurate Preview after N passes equals renderExport at N.
void pathTraceTests()
{
    const auto scene = glass::makeScene();
    const auto setup = glass::makeSetup (scene);
    const auto view = glass::makeView (*scene, 24, 16);
    constexpr std::uint32_t samples = 8;
    std::string error;

    pt::SceneEngine exporter (2);
    pt::EngineImage exported;
    check (exporter.renderExport (setup, view, pt::EngineColor::LinearRgba16f, samples, exported, error),
           "the glass scene exports: " + error);

    const auto previewAfter = [&] (bool seeded, pt::SceneEngine& engine, pt::EngineImage& image) {
        engine.previewSeededLikeExport = seeded;
        engine.maximumPreviewSamples = samples;
        for (std::uint32_t pass = 0; pass < samples; ++pass)
            if (! engine.renderPreview (setup, view, pt::EngineColor::LinearRgba16f, image, error))
                return false;
        return image.samplesPerPixel == samples;
    };

    pt::SceneEngine accurateEngine (2);
    pt::EngineImage accurate;
    check (previewAfter (true, accurateEngine, accurate), "the Accurate Preview reaches 8 samples: " + error);
    check (accurate.generation != pt::kExportGeneration,
           "the Accurate Preview keeps its own accumulation generation");
    check (! exported.colorBytes.empty() && accurate.colorBytes == exported.colorBytes,
           "the Accurate Preview at 8 samples has the export Color bytes");
    check (accurate.depth == exported.depth, "the Accurate Preview has the export depth");

    // A different thread count changes the pass split, never the sample streams.
    pt::SceneEngine threaded (5);
    pt::EngineImage threadedImage;
    check (previewAfter (true, threaded, threadedImage) && threadedImage.colorBytes == exported.colorBytes,
           "the Accurate Preview matches the export at any thread count");

    pt::SceneEngine fastEngine (2);
    pt::EngineImage fast;
    check (previewAfter (false, fastEngine, fast), "the Fast Preview reaches 8 samples: " + error);
    check (fast.generation == fastEngine.previewGeneration() && fast.generation != pt::kExportGeneration,
           "the Fast Preview draws its own generation's streams, as before tier K");
    check (fast.colorBytes != exported.colorBytes,
           "the Fast Preview streams stay independent of the export streams");

    // Switching the mode restarts the accumulation on the other streams.
    fastEngine.previewSeededLikeExport = true;
    pt::EngineImage switched;
    check (fastEngine.renderPreview (setup, view, pt::EngineColor::LinearRgba16f, switched, error)
               && switched.restarted && switched.samplesPerPixel == 1,
           "a mode change restarts the Preview accumulation");
    bool switchedRendered = true;
    for (std::uint32_t pass = 1; pass < samples && switchedRendered; ++pass)
        switchedRendered = fastEngine.renderPreview (setup, view, pt::EngineColor::LinearRgba16f, switched, error);
    check (switchedRendered && switched.samplesPerPixel == samples, "the switched Preview reaches 8 samples: " + error);
    check (switched.colorBytes == exported.colorBytes,
           "a Preview switched to Accurate converges onto the export bytes");

    // The renderer's sample target: Accurate settles at the export sample count, so
    // the settled Preview is the export frame; Fast keeps previewSamples.
    check (pt::previewSampleTarget (true, 4096, 64) == 64 && pt::previewSampleTarget (false, 4096, 64) == 4096
               && pt::previewSampleTarget (true, 4096, 0) == 1 && pt::previewSampleTarget (false, 0, 64) == 1,
           "Accurate targets exportSamples and Fast targets previewSamples");
    pt::SceneEngine capped (2);
    capped.previewSeededLikeExport = true;
    capped.maximumPreviewSamples = pt::previewSampleTarget (true, 4096, samples);
    pt::EngineImage settledImage;
    bool cappedRendered = true;
    for (std::uint32_t pass = 0; pass < samples && cappedRendered; ++pass)
        cappedRendered = capped.renderPreview (setup, view, pt::EngineColor::LinearRgba16f, settledImage, error);
    pt::EngineImage after;
    check (cappedRendered && ! capped.previewProgressing() && settledImage.colorBytes == exported.colorBytes,
           "an Accurate Preview stops at the export sample count with the export bytes: " + error);
    check (capped.renderPreview (setup, view, pt::EngineColor::LinearRgba16f, after, error) && after.unchanged
               && ! after.traced,
           "a settled Accurate Preview traces no further samples");

    // Export Samples lowered while the Accurate Preview is live: the engine is kept,
    // the target drops below the samples taken, and the accumulation restarts so the
    // settled Preview is the export at the new count.
    constexpr std::uint32_t lowered = 4;
    pt::SceneEngine lowerExporter (2);
    pt::EngineImage loweredExport;
    check (lowerExporter.renderExport (setup, view, pt::EngineColor::LinearRgba16f, lowered, loweredExport, error),
           "the glass scene exports at 4 samples: " + error);
    capped.maximumPreviewSamples = pt::previewSampleTarget (true, 4096, lowered);
    check (capped.previewProgressing(), "a lowered target keeps the idle gate open for the restart");
    pt::EngineImage restarted;
    bool loweredRendered = capped.renderPreview (setup, view, pt::EngineColor::LinearRgba16f, restarted, error);
    check (loweredRendered && restarted.restarted && restarted.samplesPerPixel == 1,
           "a lowered target restarts the Accurate Preview");
    for (std::uint32_t pass = 1; pass < lowered && loweredRendered; ++pass)
        loweredRendered = capped.renderPreview (setup, view, pt::EngineColor::LinearRgba16f, restarted, error);
    check (loweredRendered && restarted.samplesPerPixel == lowered && ! capped.previewProgressing()
               && restarted.colorBytes == loweredExport.colorBytes,
           "the restarted Accurate Preview settles on the export at the lowered count: " + error);
}

// The Accurate refusal latch: the same key answers with the latched error, and a
// change to any keyed input retries.
void refusalLatchTests()
{
    arbitgpu::OpenGlCausticRefusalKey key;
    key.sceneContent = 0x1000;
    key.resources = 0x2000;
    key.context = 0x3000;
    key.rendererGeneration = 4;
    key.source = 1;
    key.glassObject = 2;
    key.glassMaterial = 3;
    key.policy = "policy";
    key.width = 1920;
    key.height = 1080;
    key.exportBudgets = true;
    key.limits.maximumStorageBlockBytes = 64ull << 20;
    key.cameraIdentity = "camera-a";
    key.light.id.value = 1;
    key.light.transform.rotation = { 0, 0.1f, 0, 0.995f };
    arbitgpu::opaquevisibility::Instance instance {};
    instance.world[0] = instance.world[5] = instance.world[10] = instance.world[15] = 1;
    instance.object = 0;
    key.instances = { instance, instance };
    key.instances[1].object = 1;
    const auto reason = arbitgpu::openGlAccurateCausticRefusal (
        "OpenGL caustic admitted storage exceeds native GL limits: deposit receipts need 125.0 MiB, max SSBO 64.0 MiB");
    check (reason.find ("deposit receipts need 125.0 MiB") != std::string::npos
               && reason.find ("switch Render Mode to Fast") != std::string::npos,
           "the Accurate refusal names the limit and tells the user to switch Render Mode to Fast: " + reason);

    arbitgpu::OpenGlCausticRefusalLatch latch;
    check (latch.find (key) == nullptr, "an empty latch holds no refusal");
    latch.latch (key, reason);
    const auto* cached = latch.find (key);
    check (cached != nullptr && *cached == reason, "the same key returns the latched refusal");
    check (latch.find (key) != nullptr, "the latch holds across repeated ticks");

    const auto retries = [&] (const char* what, auto change) {
        latch.latch (key, reason);
        auto other = key;
        change (other);
        check (latch.find (other) == nullptr, std::string ("a changed ") + what + " retries");
        check (latch.find (key) == nullptr, std::string ("a retry after a changed ") + what + " clears the latch");
    };
    retries ("scene content", [] (arbitgpu::OpenGlCausticRefusalKey& k) { k.sceneContent = 0x1001; });
    retries ("prepared resource", [] (arbitgpu::OpenGlCausticRefusalKey& k) { k.resources = 0x2001; });
    retries ("renderer generation", [] (arbitgpu::OpenGlCausticRefusalKey& k) { ++k.rendererGeneration; });
    retries ("width", [] (arbitgpu::OpenGlCausticRefusalKey& k) { k.width = 1280; });
    retries ("height", [] (arbitgpu::OpenGlCausticRefusalKey& k) { k.height = 720; });
    retries ("mode", [] (arbitgpu::OpenGlCausticRefusalKey& k) { k.exportBudgets = false; });
    retries ("schedule", [] (arbitgpu::OpenGlCausticRefusalKey& k) { k.scheduling.entrySubdivisions = 2; });
    retries ("policy", [] (arbitgpu::OpenGlCausticRefusalKey& k) { k.policy = "other"; });
    retries ("GL limit", [] (arbitgpu::OpenGlCausticRefusalKey& k) { k.limits.maximumStorageBlockBytes = 2048ull << 20; });
    retries ("source light", [] (arbitgpu::OpenGlCausticRefusalKey& k) { k.source = 9; });
    retries ("glass object", [] (arbitgpu::OpenGlCausticRefusalKey& k) { k.glassObject = 9; });
    retries ("glass material", [] (arbitgpu::OpenGlCausticRefusalKey& k) { k.glassMaterial = 9; });
    retries ("multiple scatter policy", [] (arbitgpu::OpenGlCausticRefusalKey& k) { k.multipleScatter = 2; });
    retries ("GL context", [] (arbitgpu::OpenGlCausticRefusalKey& k) { k.context = 0x3001; });
    retries ("runtime camera", [] (arbitgpu::OpenGlCausticRefusalKey& k) { k.cameraIdentity = "camera-b"; });
    retries ("runtime light direction", [] (arbitgpu::OpenGlCausticRefusalKey& k) {
        k.light.transform.rotation = { 0, 0.2f, 0, 0.98f };
    });
    retries ("runtime light radiance", [] (arbitgpu::OpenGlCausticRefusalKey& k) { k.light.spectralSource.radiance[0] = 3; });
    retries ("runtime instance transform", [] (arbitgpu::OpenGlCausticRefusalKey& k) { k.instances[1].world[12] = 0.25f; });
    retries ("instance count", [] (arbitgpu::OpenGlCausticRefusalKey& k) { k.instances.pop_back(); });
    // Time is not in the key: equal evaluated inputs at another time keep the latch.
    latch.latch (key, reason);
    auto still = key;
    check (latch.find (still) != nullptr, "an unchanged evaluated frame at another time keeps the latch");

    // The scene component is content-stable, not address-stable: the snapshot address is
    // no longer in the key (verify81's held-key miss named scene every tick while the
    // viewport alternated snapshots), a copied scene keys the same, a moved object keys
    // differently, and a Motion-only previous-frame change keeps the latch.
    {
        const auto scene = std::make_unique<HarmonicMIDI::grid::Visual3DScene>();
        scene->vertexCount = 1;
        scene->objectCount = 1;
        scene->objects[0].transform.translation.x = 1.0f;
        const auto stable = arbitgpu::causticSceneContentFingerprint (*scene);
        const HarmonicMIDI::grid::Visual3DScene copy = *scene;
        check (stable == arbitgpu::causticSceneContentFingerprint (copy),
               "a copied scene snapshot keeps its fingerprint");
        scene->objects[0].transform.translation.x = 2.0f;
        const auto moved = arbitgpu::causticSceneContentFingerprint (*scene);
        check (stable != moved, "a moved object changes the scene fingerprint");
        scene->previousObjectTransformCount = 1;
        scene->previousObjectTransforms[0] = scene->objects[0].transform;
        check (moved == arbitgpu::causticSceneContentFingerprint (*scene),
               "a Motion-only previous-frame change keeps the scene fingerprint");
        // The light records stay out of the fingerprint: the source light is keyed as its
        // evaluated record, and verify83 named the lights array as the only segment the
        // lighthouse starter churns per tick.
        scene->lightCount = 1;
        const auto lit = arbitgpu::causticSceneContentFingerprint (*scene);
        scene->lights[0].intensity = 7.0f;
        check (lit == arbitgpu::causticSceneContentFingerprint (*scene),
               "a changed light record keeps the scene fingerprint");
    }
}

// ARBIT_TEST_GL_CAUSTIC_LIMIT_SCALE: only a number in (0,1) overrides, it scales the size
// limits and nothing else, and a 672x378 frame that fits an RTX 4070 is refused at 0.01.
void limitScaleTests()
{
    using arbitgpu::parseOpenGlCausticLimitScale;
    for (const char* ignored : { (const char*) nullptr, "", "abc", "0", "-0.5", "1", "1.0", "4", "nan", "inf", "0.5x", " " })
        check (! parseOpenGlCausticLimitScale (ignored).has_value(),
               std::string ("limit scale text is ignored: ") + (ignored ? ignored : "(unset)"));
    const auto half = parseOpenGlCausticLimitScale ("0.5");
    check (half && *half == 0.5, "limit scale 0.5 parses");
    const auto small = parseOpenGlCausticLimitScale ("1e-2");
    check (small && *small == 0.01, "limit scale 1e-2 parses");

    arbitgpu::OpenGlCausticDeviceLimits device;
    device.maximumStorageBlockBytes = 2147483647ull;
    device.maximumWorkGroupsX = 2147483647ull;
    device.maximumWorkGroupsY = 65535;
    device.maximumTextureSize = 32768;
    device.maximumInvocations = 1024;
    device.maximumLocalSizeX = 1024;
    device.maximumLocalSizeY = 1024;
    device.storageBindings = 96;
    device.computeStorageBlocks = 16;
    check (arbitgpu::scaleOpenGlCausticDeviceLimits (device, 1.0) == device
               && arbitgpu::scaleOpenGlCausticDeviceLimits (device, 0.0) == device
               && arbitgpu::scaleOpenGlCausticDeviceLimits (device, 2.0) == device,
           "a scale outside (0,1) leaves the limits unchanged");
    const auto scaled = arbitgpu::scaleOpenGlCausticDeviceLimits (device, 0.01);
    check (scaled.maximumStorageBlockBytes == 21474836 && scaled.maximumWorkGroupsX == 21474836
               && scaled.maximumWorkGroupsY == 655 && scaled.maximumTextureSize == 327,
           "scale 0.01 scales storage bytes, work group counts and texture size");
    check (scaled.maximumInvocations == 1024 && scaled.maximumLocalSizeX == 1024 && scaled.maximumLocalSizeY == 1024
               && scaled.storageBindings == 96 && scaled.computeStorageBlocks == 16,
           "scale keeps the capability counts");
    auto tiny = device;
    tiny.maximumTextureSize = 3;
    check (arbitgpu::scaleOpenGlCausticDeviceLimits (tiny, 0.01).maximumTextureSize == 1, "a nonzero limit scales to at least 1");

    arbitgpu::OpenGlCausticStorageDemand demand;
    demand.textureExtent = 672;
    demand.bytes[2] = demand.bytes[6] = 672ull * 378 * 4 * sizeof (float);
    check (arbitgpu::openGlCausticLimitFailure (device, demand).empty(), "the 672x378 frame fits the queried limits");
    const auto failure = arbitgpu::openGlCausticLimitFailure (scaled, demand);
    check (failure == "texture extent 672 exceeds max texture size 327", "scale 0.01 refuses the 672x378 frame: " + failure);

    arbitgpu::OpenGlCausticRefusalKey key, scaledKey;
    key.limits = device;
    scaledKey.limits = scaled;
    arbitgpu::OpenGlCausticRefusalLatch latch;
    latch.latch (scaledKey, arbitgpu::openGlAccurateCausticRefusal (failure));
    check (latch.find (scaledKey) != nullptr, "the scaled limits key the latch");
    check (latch.find (key) == nullptr, "removing the scale retries the latched frame");
}
#else
namespace gc = arbitgpu::glasscaustics;
namespace gt = glasscaustictests;
namespace sg = arbitgpu::smoothglass;
using namespace HarmonicMIDI::grid;

// The ordinary caustic fixture with a thin water plane at z=-1.5 between the
// glass and the receiver, as lighthouse_caustic_volume_tests.cpp builds it.
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
    p->bindingDigest = "parity-fixture-binding-water";
    p->programIdentity = "parity-fixture-program-water";
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

bool readColor (const arbitgpu::NativeFixtureSceneSubmission& submission, std::vector<std::uint8_t>& bytes,
                std::string& error)
{
    arbitgpu::NativeRawPassPixels pixels;
    if (! submission.rendered || ! submission.frame
        || ! submission.frame->readRawPass (renderpassoutput::Output::Color, pixels, error))
        return false;
    bytes = std::move (pixels.bytes);
    return true;
}

// Export once, then Accurate Preview frames until the caustic is adopted and
// resolved. The test waits on the GPU between frames, as later viewport frames
// would find the fence signalled. The renderer itself never waits.
void parity (const char* name, const gt::Fixture& fixture,
             const std::shared_ptr<const arbitgpu::NativeFixtureSceneResources>& resources, bool causticVisible)
{
    using Use = arbitgpu::OpenGlCausticRenderUse;
    constexpr std::uint32_t width = 32, height = 32;
    constexpr int maximumFrames = 12;
    const std::string label = name;
    if (! fixture.policy)
    {
        check (false, label + " fixture admits its branch policy");
        return;
    }
    std::string error;
    arbitgpu::NativeFixtureSceneRuntimeInputs runtime;
    runtime.linearColor = true;

    const auto exported = arbitgpu::renderOpenGlGlassCausticScene (
        fixture.scene, resources, fixture.ordinaryLight, fixture.program->object, fixture.program->authoredMaterial,
        *fixture.policy, arbitgpu::openGlCausticRenderPlan (Use::Export, width, height).scheduling, width, height,
        runtime);
    std::vector<std::uint8_t> exportBytes;
    check (readColor (exported, exportBytes, error), label + " exports its caustic: " + exported.error + error);

    // The base frame without the caustic, to show the caustic changes the pixels.
    // The lighthouse water absorbs the photons that reach it, so only the open
    // receiver is sure to show one.
    if (causticVisible)
    {
        auto baseRuntime = runtime;
        baseRuntime.imageOutput = renderpassoutput::Output::Color;
        const auto base = arbitgpu::nativeFixtureSceneBackend().render (fixture.scene, resources, width, height,
                                                                         baseRuntime);
        std::vector<std::uint8_t> baseBytes;
        check (readColor (base, baseBytes, error) && baseBytes.size() == exportBytes.size()
                   && baseBytes != exportBytes,
               label + " export Color carries a visible caustic: " + base.error + error);
    }

    const budget::Scope accurate (budget::Budget::Export);
    auto previewRuntime = runtime;
    previewRuntime.previewUse = true;
    const auto schedule = arbitgpu::openGlCausticRenderPlan (Use::Preview, width, height, true).scheduling;
    std::vector<std::uint8_t> previewBytes;
    bool settled = false;
    for (int frame = 0; frame < maximumFrames && ! settled; ++frame)
    {
        std::string degradation;
        const auto preview = arbitgpu::renderOpenGlGlassCausticScene (
            fixture.scene, resources, fixture.ordinaryLight, fixture.program->object,
            fixture.program->authoredMaterial, *fixture.policy, schedule, width, height, previewRuntime, nullptr,
            Use::Preview, &degradation);
        if (! preview.rendered)
        {
            check (false, label + " Accurate Preview frame renders: " + preview.error);
            return;
        }
        glFinish();
        settled = frame > 0 && ! preview.causticPending && degradation.empty();
        if (settled)
            check (readColor (preview, previewBytes, error), label + " Accurate Preview Color reads: " + error);
    }
    check (settled, label + " Accurate Preview adopts its caustic within 12 frames");
    check (settled && previewBytes == exportBytes, label + " settled Accurate Preview has the export Color bytes");

    // A Fast Preview passes the coarser schedule outside the scope; the mismatch
    // check shows the scope is what moves the plan.
    const auto fastSchedule = arbitgpu::openGlCausticRenderPlan (Use::Preview, width, height).scheduling;
    const auto refused = arbitgpu::renderOpenGlGlassCausticScene (
        fixture.scene, resources, fixture.ordinaryLight, fixture.program->object, fixture.program->authoredMaterial,
        *fixture.policy, fastSchedule, width, height, previewRuntime, nullptr, Use::Preview);
    check (! refused.rendered
               && refused.error == "OpenGL caustic Preview schedule does not match the Preview render plan",
           label + " an Accurate Preview refuses the Fast schedule");
}

// The caustic route wants an exact authored owner for every object: the glass, and an
// eight-band spectral rho on each diffuse receiver. A glass-only binding leaves the receiver
// without a program, and the route refuses with "exact object/material/revision owner is
// stale or missing". This admits the two-object collection a composed scene publishes.
std::shared_ptr<const videorender::fixture3d::AdmittedSurfaceMaterialBinding> admitCausticCollection (
    const std::shared_ptr<const Visual3DScene>& snapshot, float roughness, std::string& error)
{
    namespace rough = videohelper::tests::authoredroughcaustic;
    geometrysurfacematerial::ObjectPrograms programs;

    geometrysurfacematerial::ObjectProgram glass;
    glass.program.sourceMaterial = 901;
    glass.program.material = rough::request (snapshot, roughness);
    programs.push_back (std::move (glass));

    geometrysurfacematerial::ObjectProgram receiver;
    receiver.program.sourceMaterial = 902;
    auto& material = receiver.program.material;
    material = rough::request (snapshot, 0.0f);
    material.binding.object = snapshot->objects[1].id;
    material.program = videohelper::tests::smoothglassfixture::authoredProgram (
        snapshot->objects[1].material.value, 1.5f, 0.0f);
    material.program.transport = surfacematerial::TransportMode::ThinSurface;
    const auto outputId = [&] (surfacematerial::OutputSemantic semantic) {
        return material.program.outputs[static_cast<std::size_t> (semantic)];
    };
    for (auto& operation : material.program.operations)
    {
        if (operation.id == outputId (surfacematerial::OutputSemantic::Transmission))
            operation.literal = {};
        if (operation.id == outputId (surfacematerial::OutputSemantic::Roughness))
            operation.literal = { 1.0f, 0.0f, 0.0f, 0.0f };
    }
    material.program.spectralReflectance = surfacematerial::SpectralReflectanceDescription {};
    material.program.spectralReflectance->coefficients = { .1f, .2f, .3f, .4f, .5f, .6f, .7f, .8f };
    material.version = surfacematerialbinding::kReflectanceWireVersion;
    const auto admitted = surfacematerial::admit (material.program, error);
    if (! admitted)
        return {};
    material.binding.surfaceMaterialDigest = admitted->structuralDigest();
    programs.push_back (std::move (receiver));

    return videorender::fixture3d::admitSurfaceMaterialCollection (
        snapshot, programs, {}, videohelper::materialprogram::BackendTarget::OpenGl, error);
}

// End to end through FixtureSceneRenderer: the selected mode opens the budget scope.
// An Accurate renderer's settled caustic Preview has the export bytes. A Fast renderer
// traces the coarser Preview schedule, so its settled caustic differs from the export.
void rendererModeTests (arbitgpu::NativeFixtureSceneBackend& backend)
{
    namespace renderer = videorender::fixture3d;
    namespace rough = videohelper::tests::authoredroughcaustic;
    std::string error;
    // Smooth glass on purpose. The export checks every one of its 4096 paths against the 1e-4
    // unresolved tail. With roughness .35 a few percent of those paths fail: each interface
    // spawns up to 16 microfacet exits, the 64 visit cap runs out while the small branches
    // popped first are still being traced, and the large branch left on the stack lands in the
    // tail. The export refuses that, as designed. This test is about the Preview and Export
    // schedules, not roughness, so it keeps the fixture geometry and Li8 and drops the roughness.
    constexpr float glassRoughness = 0.0f;
    auto sceneValue = gt::roughFixture (glassRoughness).scene;
    sceneValue->materials[1].baseColor = { .5f, .5f, .5f };
    const std::shared_ptr<const Visual3DScene> snapshot = sceneValue;
    const auto material = admitCausticCollection (snapshot, glassRoughness, error);
    if (! material)
    {
        check (false, "renderer caustic material admits: " + error);
        return;
    }
    const auto runtime = rough::inputs (*snapshot);
    const renderer::RenderDimensions extent { 32, 32 };
    renderer::FixtureSceneRenderer exporter (backend);
    renderer::RenderedFrame exportFrame;
    std::vector<std::uint8_t> exportBytes;
    arbitgpu::NativeRawPassPixels pixels;
    // Run the render before building the message. Argument order is unspecified, and GCC
    // builds the right-hand argument first, which logged an empty error here.
    const bool exported = exporter.renderExport (snapshot, material, {}, runtime, extent,
                                                 renderer::kNativeGpuCapability, exportFrame, error)
        && exportFrame.nativeFrame
        && exportFrame.nativeFrame->readRawPass (renderpassoutput::Output::Color, pixels, error);
    check (exported, "renderer exports the caustic: " + error);
    exportBytes = pixels.bytes;

    const auto settle = [&] (renderer::SceneEngineMode mode, std::vector<std::uint8_t>& bytes) {
        renderer::FixtureSceneRenderer preview (backend);
        renderer::SceneEngineSelection selection;
        selection.mode = mode;
        preview.selectEngine (selection);
        for (int frame = 0; frame < 12; ++frame)
        {
            renderer::RenderedFrame result;
            if (! preview.renderPreview (snapshot, material, {}, runtime, extent, renderer::kNativeGpuCapability,
                                         result, error))
                return false;
            glFinish();
            if (frame > 0 && result.nativeFrame && ! renderer::previewCausticPendingOnThisThread()
                && result.causticDegradation.empty())
            {
                arbitgpu::NativeRawPassPixels read;
                if (! result.nativeFrame->readRawPass (renderpassoutput::Output::Color, read, error))
                    return false;
                bytes = std::move (read.bytes);
                return true;
            }
        }
        error = "caustic did not settle within 12 frames";
        return false;
    };
    std::vector<std::uint8_t> accurate, fast;
    // Name what went wrong. A settled Preview that simply differs from the export leaves
    // `error` empty, which logged an empty failure message here.
    const auto describe = [&] (bool settled, const std::vector<std::uint8_t>& bytes) -> std::string {
        if (! error.empty())
            return error;
        if (! settled)
            return "the Preview did not settle";
        if (exportBytes.empty())
            return "the export read back no Color";
        if (bytes.size() != exportBytes.size())
            return "the Preview Color has " + std::to_string (bytes.size()) + " bytes and the export has "
                + std::to_string (exportBytes.size());
        std::size_t differing = 0;
        for (std::size_t i = 0; i < bytes.size(); ++i)
            differing += bytes[i] != exportBytes[i] ? 1u : 0u;
        return std::to_string (differing) + " of " + std::to_string (bytes.size())
            + " Preview Color bytes differ from the export";
    };
    error.clear();
    const bool accurateSettled = settle (renderer::SceneEngineMode::Accurate, accurate);
    const std::string accurateWhy = describe (accurateSettled, accurate);
    check (accurateSettled && ! exportBytes.empty() && accurate == exportBytes,
           "an Accurate renderer's settled caustic Preview has the export bytes: " + accurateWhy);
    error.clear();
    const bool fastSettled = settle (renderer::SceneEngineMode::Fast, fast);
    const std::string fastWhy = fastSettled && fast.size() == exportBytes.size() && fast == exportBytes
        ? std::string ("the Fast Preview Color equals the export Color byte for byte")
        : describe (fastSettled, fast);
    check (fastSettled && fast.size() == exportBytes.size() && fast != exportBytes,
           "a Fast renderer's settled caustic Preview keeps the coarser Preview schedule: " + fastWhy);
    check (! arbitgpu::previewbudget::exportBudgets(), "the renderer closes its budget scope after each draw");
}

constexpr const char* kGlassDepthPrepassEnvironment = "ARBIT_GL_GLASS_DEPTH_PREPASS";

void setEnvironment (const char* name, const char* value)
{
#ifdef _WIN32
    _putenv_s (name, value != nullptr ? value : "");
#else
    if (value != nullptr)
        ::setenv (name, value, 1);
    else
        ::unsetenv (name);
#endif
}

// The ordinary closed-glass fixture seen from its default camera at z=3, so the slab
// covers the middle of the frame. The parity fixture above puts the camera behind it.
gt::Fixture visibleClosedGlass()
{
    return gt::ordinaryFixture();
}

// A floor drawn before the glass. It lies in the slab's bottom face plane, z=-0.5, and
// covers x >= 0, so half of the bottom face shares it. The camera looks straight up from
// z=-3, so every fragment of both planes has exactly the same depth. A single GL_LESS draw
// keeps the floor there. The other half of the bottom face shows the glass.
gt::Fixture glassOnCoplanarFloor()
{
    auto f = gt::ordinaryFixture();
    auto& scene = *f.scene;
    const auto first = scene.vertexCount;
    const auto floor = scene.objectCount;
    videohelper::tests::smoothglassfixture::receiver (scene, 0, { .25, .5, .75 });
    for (std::size_t v = first; v < scene.vertexCount; ++v)
        scene.vertices[v].position.z = -.5f;
    scene.materials[scene.materialCount - 1].doubleSided = true;
    auto p = std::make_shared<gt::Program> (*f.program->objectPrograms.front());
    p->object = scene.objects[floor].id;
    p->authoredMaterial = scene.objects[floor].material;
    p->parameters.identifiers[0] = p->authoredMaterial.value;
    p->bindingDigest = "parity-fixture-binding-floor";
    p->programIdentity = "parity-fixture-program-floor";
    f.program->objectPrograms.push_back (p);
    // Opaque objects draw in scene order, so moving the floor first draws it before the glass.
    std::rotate (scene.objects.begin(), scene.objects.begin() + floor, scene.objects.begin() + floor + 1);
    scene.cameras[0].transform.translation = { 0, 0, -3 };
    scene.cameras[0].transform.rotation = { 1, 0, 0, 0 };
    return f;
}

bool readObjectIds (const arbitgpu::NativeFixtureSceneSubmission& submission, std::vector<std::uint32_t>& ids,
                    std::string& error)
{
    arbitgpu::NativeRawPassPixels pixels;
    if (! submission.rendered || ! submission.frame
        || ! submission.frame->readRawPass (renderpassoutput::Output::ObjectId, pixels, error))
        return false;
    ids.assign (pixels.bytes.size() / sizeof (std::uint32_t), 0u);
    if (! ids.empty())
        std::memcpy (ids.data(), pixels.bytes.data(), ids.size() * sizeof (std::uint32_t));
    return true;
}

struct PrepassFrame
{
    std::vector<std::uint8_t> color;
    std::vector<std::uint32_t> ids;
    std::uint64_t prepassDraws = 0;
};

// The walked-glass depth prepass only skips hidden walk work, so each render mode draws
// the same Color and object-id bytes with it on and off. Off, on, off: the two off frames
// must match, which shows the Fast Preview program did not change between them. The
// prepass draw counter shows the prepass ran only when on, and every object in `glass`
// must own pixels, so the frame shows the walked glass.
void glassDepthPrepassParity (const char* name, const gt::Fixture& fixture,
                              const std::shared_ptr<const arbitgpu::NativeFixtureSceneResources>& resources,
                              const std::vector<SceneObjectId>& glass, bool exportFrame)
{
    constexpr std::uint32_t width = 48, height = 48;
    constexpr int warmFrames = 12;
    auto& backend = arbitgpu::nativeFixtureSceneBackend();
    const char* priorValue = std::getenv (kGlassDepthPrepassEnvironment);
    const std::optional<std::string> prior = priorValue != nullptr
        ? std::optional<std::string> (priorValue) : std::nullopt;
    const auto draw = [&] (const arbitgpu::NativeFixtureSceneRuntimeInputs& runtime, bool prepass,
                           PrepassFrame& out, std::string& error) {
        setEnvironment (kGlassDepthPrepassEnvironment, prepass ? "1" : "0");
        const auto before = arbitgpu::glassoverdraw::depthPrepassDraws().load();
        const auto frame = backend.render (fixture.scene, resources, width, height, runtime);
        glFinish();
        out.prepassDraws = arbitgpu::glassoverdraw::depthPrepassDraws().load() - before;
        if (readColor (frame, out.color, error) && readObjectIds (frame, out.ids, error))
            return true;
        error += frame.error;
        return false;
    };
    const auto differing = [] (const auto& left, const auto& right) {
        std::size_t count = left.size() == right.size() ? 0u : std::max (left.size(), right.size());
        for (std::size_t i = 0; i < std::min (left.size(), right.size()); ++i)
            count += left[i] != right[i] ? 1u : 0u;
        return count;
    };
    const auto compare = [&] (const std::string& mode, const arbitgpu::NativeFixtureSceneRuntimeInputs& runtime) {
        const std::string label = std::string (name) + " " + mode;
        std::string error;
        PrepassFrame before, with, after;
        for (int frame = 0; frame < warmFrames; ++frame)
            if (! draw (runtime, true, with, error))
                break;
        const bool drawn = error.empty() && draw (runtime, false, before, error)
            && draw (runtime, true, with, error) && draw (runtime, false, after, error);
        check (drawn, label + " draws its base frame with and without the glass depth prepass: " + error);
        if (! drawn)
            return;
        check (with.prepassDraws > 0, label + " issues glass depth prepass draws when the prepass is on");
        check (before.prepassDraws == 0 && after.prepassDraws == 0,
               label + " issues no glass depth prepass draws when the prepass is off");
        for (const auto object : glass)
        {
            const auto pixels = std::count (with.ids.begin(), with.ids.end(), object.value);
            check (pixels > 0, label + " shows glass object " + std::to_string (object.value) + " in "
                                   + std::to_string (pixels) + " pixels");
        }
        check (! before.color.empty() && before.color == after.color && before.ids == after.ids,
               label + " draws the same bytes twice without the prepass");
        const auto colorDiffering = differing (with.color, before.color);
        check (colorDiffering == 0, label + " Color is byte-identical with the glass depth prepass: "
                                        + std::to_string (colorDiffering) + " of "
                                        + std::to_string (before.color.size()) + " bytes differ");
        const auto idsDiffering = differing (with.ids, before.ids);
        check (idsDiffering == 0, label + " object ids are identical with the glass depth prepass: "
                                      + std::to_string (idsDiffering) + " of "
                                      + std::to_string (before.ids.size()) + " pixels differ");
    };
    arbitgpu::NativeFixtureSceneRuntimeInputs runtime;
    runtime.linearColor = true;
    runtime.imageOutput = renderpassoutput::Output::Color;
    if (exportFrame)
        compare ("export", runtime);
    auto preview = runtime;
    preview.previewUse = true;
    {
        const budget::Scope accurate (budget::Budget::Export);
        compare ("Accurate Preview", preview);
    }
    compare ("Fast Preview", preview);
    setEnvironment (kGlassDepthPrepassEnvironment, prior ? prior->c_str() : nullptr);
}

constexpr const char* kSmoothOnlyGlassProgramEnvironment = "ARBIT_GL_SMOOTH_ONLY_GLASS_PROGRAM";

using GlassResources = std::shared_ptr<const arbitgpu::NativeFixtureSceneResources>;

// A scene whose glass is all roughness 0 draws with the smooth-only program, which leaves
// the rough walks out of its text, when ARBIT_GL_SMOOTH_ONLY_GLASS_PROGRAM=1 is set at
// preparation. Without it the full program keeps them. The two programs must draw the same
// Color and object-id bytes in export, Accurate Preview and Fast Preview. Fast Preview draws a variant that each program builds without
// a wait, so each one first draws Fast frames until a frame uses its variant.
void smoothOnlyProgramParity (const char* name, const gt::Fixture& fixture,
                              const std::function<GlassResources (std::string&)>& prepare, bool exportFrame)
{
    constexpr std::uint32_t width = 48, height = 48;
    const std::string label = name;
    auto& backend = arbitgpu::nativeFixtureSceneBackend();
    const char* priorValue = std::getenv (kSmoothOnlyGlassProgramEnvironment);
    const std::optional<std::string> prior = priorValue != nullptr
        ? std::optional<std::string> (priorValue) : std::nullopt;
    std::string error;
    setEnvironment (kSmoothOnlyGlassProgramEnvironment, "1");
    const auto smoothChoices = sg::smoothOnlyProgramChoices().load();
    const auto smooth = prepare (error);
    check (smooth != nullptr, label + " prepares with the smooth-only glass program: " + error);
    check (sg::smoothOnlyProgramChoices().load() > smoothChoices, label + " chooses the smooth-only glass program");
    setEnvironment (kSmoothOnlyGlassProgramEnvironment, nullptr);
    const auto fullChoices = sg::smoothOnlyProgramChoices().load();
    error.clear();
    const auto full = prepare (error);
    check (full != nullptr, label + " prepares with the full glass program: " + error);
    check (sg::smoothOnlyProgramChoices().load() == fullChoices, label + " default program takes no smooth-only choice");
    setEnvironment (kSmoothOnlyGlassProgramEnvironment, prior ? prior->c_str() : nullptr);
    if (smooth == nullptr || full == nullptr)
        return;
    const auto draw = [&] (const GlassResources& resources, const arbitgpu::NativeFixtureSceneRuntimeInputs& runtime,
                           PrepassFrame& out, std::string& why) {
        const auto frame = backend.render (fixture.scene, resources, width, height, runtime);
        glFinish();
        if (readColor (frame, out.color, why) && readObjectIds (frame, out.ids, why))
            return true;
        why += frame.error;
        return false;
    };
    // Fast Preview only: draws Fast frames until one is drawn with this program's Preview
    // variant, for at most 20 s. A variant build that ends Failed fails the arm.
    const auto awaitVariant = [&] (const GlassResources& resources, const std::string& program,
                                   const arbitgpu::NativeFixtureSceneRuntimeInputs& runtime, const std::string& text) {
        std::string why;
        const auto failures = sg::previewGlassVariantFailures().load();
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds (20);
        while (std::chrono::steady_clock::now() < until)
        {
            const auto frames = sg::previewGlassVariantFrames().load();
            PrepassFrame ignored;
            if (! draw (resources, runtime, ignored, why))
            {
                check (false, text + " draws Fast frames with the " + program + " glass program: " + why);
                return false;
            }
            if (sg::previewGlassVariantFailures().load() != failures)
            {
                check (false, text + " Preview variant of the " + program + " glass program ended Failed");
                return false;
            }
            if (sg::previewGlassVariantFrames().load() > frames)
                return true;
            std::this_thread::sleep_for (std::chrono::milliseconds (20));
        }
        check (false, text + " Preview variant of the " + program + " glass program is neither Ready nor Failed after 20 s");
        return false;
    };
    const auto compare = [&] (const std::string& mode, const arbitgpu::NativeFixtureSceneRuntimeInputs& runtime,
                              bool fast) {
        const std::string text = label + " " + mode;
        if (fast && (! awaitVariant (smooth, "smooth-only", runtime, text) || ! awaitVariant (full, "full", runtime, text)))
            return;
        std::string why;
        PrepassFrame withSmooth, withFull;
        const auto beforeSmooth = sg::previewGlassVariantFrames().load();
        const bool smoothDrawn = draw (smooth, runtime, withSmooth, why);
        const auto beforeFull = sg::previewGlassVariantFrames().load();
        const bool drawn = smoothDrawn && draw (full, runtime, withFull, why);
        const auto afterFull = sg::previewGlassVariantFrames().load();
        check (drawn, text + " draws with the smooth-only and the full glass program: " + why);
        if (! drawn)
            return;
        // Fast frames must both come from the Preview variants. Export and Accurate frames
        // never draw a variant.
        if (fast)
            check (beforeFull > beforeSmooth && afterFull > beforeFull,
                   text + " compares two frames drawn by the Preview variants of both glass programs");
        else
            check (afterFull == beforeSmooth, text + " compares two frames drawn by the prepared glass programs");
        check (! withSmooth.color.empty() && withSmooth.color == withFull.color,
               text + " Color is byte-identical with the smooth-only and the full glass program");
        check (withSmooth.ids == withFull.ids, text + " object ids are identical with both glass programs");
    };
    arbitgpu::NativeFixtureSceneRuntimeInputs runtime;
    runtime.linearColor = true;
    runtime.imageOutput = renderpassoutput::Output::Color;
    if (exportFrame)
        compare ("export", runtime, false);
    auto preview = runtime;
    preview.previewUse = true;
    {
        const budget::Scope accurate (budget::Budget::Export);
        compare ("Accurate Preview", preview, false);
    }
    compare ("Fast Preview", preview, true);
}

void glParityTests (arbitgpu::NativeFixtureSceneBackend& backend)
{
    std::string error;
    {
        auto fixture = gt::ordinaryFixture();
        fixture.scene->cameras[0].transform.translation = { 0, 0, -2 };
        if (! fixture.prepare (error))
        {
            check (false, "ordinary caustic fixture prepares: " + error);
            return;
        }
        auto preparation = backend.prepare (fixture.scene, fixture.program);
        check (preparation.prepared, "ordinary caustic scene prepares: " + preparation.error);
        if (preparation.prepared)
            parity ("closed glass", fixture, preparation.resources, true);
    }
    {
        auto fixture = visibleClosedGlass();
        auto preparation = backend.prepare (fixture.scene, fixture.program);
        check (preparation.prepared, "closed glass seen from z=3 prepares: " + preparation.error);
        if (preparation.prepared)
            glassDepthPrepassParity ("closed glass", fixture, preparation.resources,
                                     { fixture.scene->objects[0].id }, true);
        smoothOnlyProgramParity ("closed glass", fixture, [&] (std::string& why) -> GlassResources {
            auto prepared = backend.prepare (fixture.scene, fixture.program);
            why = prepared.error;
            return prepared.prepared ? prepared.resources : nullptr;
        }, true);
    }
    {
        // A rough glass program keeps the rough walks. The decision count shows the
        // preparation reached the program choice; the smooth-only count shows it kept the full text.
        // The variable is set so the smooth-only choice is on offer and rough roughness still refuses it.
        auto fixture = gt::roughFixture (.05f);
        const char* priorRough = std::getenv (kSmoothOnlyGlassProgramEnvironment);
        const std::optional<std::string> priorRoughValue = priorRough != nullptr
            ? std::optional<std::string> (priorRough) : std::nullopt;
        setEnvironment (kSmoothOnlyGlassProgramEnvironment, "1");
        const auto decisions = sg::glassProgramDecisions().load();
        const auto choices = sg::smoothOnlyProgramChoices().load();
        (void) backend.prepare (fixture.scene, fixture.program);
        setEnvironment (kSmoothOnlyGlassProgramEnvironment, priorRoughValue ? priorRoughValue->c_str() : nullptr);
        check (sg::glassProgramDecisions().load() > decisions, "rough closed glass reaches the glass program choice");
        check (sg::smoothOnlyProgramChoices().load() == choices, "rough closed glass keeps the full glass program");
    }
    {
        // Export refuses a frame with an incomplete glass pixel, and a walk that starts on a
        // face shared with an opaque floor may not complete, so only Preview runs here.
        auto fixture = glassOnCoplanarFloor();
        auto preparation = backend.prepare (fixture.scene, fixture.program);
        check (preparation.prepared, "glass on a coplanar floor prepares: " + preparation.error);
        if (preparation.prepared)
            glassDepthPrepassParity ("glass on a coplanar floor", fixture, preparation.resources,
                                     { fixture.program->object }, false);
    }
    {
        auto fixture = lighthouse();
        const sg::CausticVolumeSelection selection { fixture.scene->objects[0].id, fixture.scene->objects[0].material };
        const std::shared_ptr<const Visual3DScene> scene = fixture.scene;
        auto preparation = arbitgpu::prepareOpenGlGlassCausticVolumeResources (
            backend, scene, fixture.program, selection, arbitgpu::NativeFixturePrepareUse::Blocking);
        check (preparation.prepared, "lighthouse water and glass volume prepare: " + preparation.error);
        if (preparation.prepared)
        {
            parity ("lighthouse", fixture, preparation.resources, false);
            glassDepthPrepassParity ("lighthouse", fixture, preparation.resources,
                                     { fixture.scene->objects[0].id, fixture.program->objectPrograms.back()->object },
                                     true);
            // No smooth-only comparison here. With the rough code cut, the paired Lighthouse
            // program exceeds the NVIDIA fragment instruction limit at link (24 Oct), so only
            // the closed-glass fixture runs smoothOnlyProgramParity.
        }
    }
    rendererModeTests (backend);
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
    auto* window = glfwCreateWindow (64, 64, "render-mode-preview-parity", nullptr, nullptr);
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
    std::string renderer = rendererRaw != nullptr ? rendererRaw : "";
    for (auto& c : renderer)
        if (c >= 'A' && c <= 'Z')
            c = static_cast<char> (c - 'A' + 'a');
    const bool software = renderer.find ("llvmpipe") != std::string::npos
        || renderer.find ("softpipe") != std::string::npos || renderer.find ("lavapipe") != std::string::npos
        || renderer.find ("swr") != std::string::npos;
    if (! arbitgl::loadGlFunctions (gl, missing))
    {
        std::fprintf (stderr, "FAIL: OpenGL function loading failed: %s\n", missing.c_str());
        result = 1;
    }
    else if (software && std::getenv ("ARBIT_GL_TESTS_ALLOW_SOFTWARE") == nullptr)
    {
        std::fprintf (stderr, "SKIP: software OpenGL renderer %s\n", renderer.c_str());
        result = 77;
    }
    else
    {
        glParityTests (arbitgpu::nativeFixtureSceneBackend());
        result = failures == 0 ? 0 : 1;
    }
    glfwDestroyWindow (window);
    glfwTerminate();
    return result;
}
#endif
} // namespace

int main()
{
#if ARBIT_RENDER_MODE_PARITY_GL
    const int result = runGl();
    if (result == 77)
        return 77;
#else
    causticPlanTests();
    budgetScopeTests();
    pathTraceTests();
    refusalLatchTests();
    limitScaleTests();
#endif
    if (failures != 0)
    {
        std::fprintf (stderr, "%d of %d render mode parity checks failed\n", failures, checks);
        return 1;
    }
    std::printf ("PASS: %d render mode parity checks\n", checks);
    return 0;
}
