// Route tests for the Path Trace render engine (Plan 3, PT-C1): the glass
// material mapping and input refusals of the fixture engine adapter, the
// FixtureSceneRenderer engine branch, and the imported scene routes that go
// through the animated execution. A Path Trace request either draws with the
// path tracer or fails with a named refusal. It never draws Raster.

#include "../src/imported_scene_payload_execution.h"
#include "../src/imported_animated_scene_payload_execution.h"
#include "../src/pathtrace_fixture_engine.h"
#include "../src/path_trace_stats_reply.h"
#include "../src/pathtrace_oidn.h"
#include "../src/pathtrace_preview_denoise.h"
#include "../src/diffraction_material_admission.h"
#include "DiffractionMaterialPresets.h"
#include "../src/pathtrace/engine.h"
#include "../src/pathtrace/integrator.h"
#include "../../shared/VisualStarterModelAssets.h"
#include "support/fixture_scene.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace
{

namespace fixture = videorender::fixture3d;
namespace ptfixture = videorender::fixture3d::pathtracefixture;
namespace pt = videohelper::pathtrace;
namespace render = visualimportedscenerender;
using Program = arbitgpu::NativeFixtureSurfaceMaterialProgram;

int checks = 0;
int failures = 0;

void check (bool condition, const char* message)
{
    ++checks;
    if (! condition)
    {
        ++failures;
        std::fprintf (stderr, "FAIL: %s\n", message);
    }
}

void check (bool condition, const std::string& message) { check (condition, message.c_str()); }

bool closeTo (float a, float b) { return std::abs (a - b) <= 1.0e-5f; }

// ---------------------------------------------------------------- fakes

class FakeFrame final : public arbitgpu::NativeFixtureSceneFrame
{
public:
    FakeFrame (std::string backend, std::uint32_t width, std::uint32_t height, std::uintptr_t handle) noexcept
        : backend_ (std::move (backend)), width_ (width), height_ (height), handle_ (handle)
    {
    }

    const std::string& backend() const noexcept override { return backend_; }
    std::uint32_t width() const noexcept override { return width_; }
    std::uint32_t height() const noexcept override { return height_; }
    std::uintptr_t colorImageHandle() const noexcept override { return handle_; }
    std::uintptr_t colorTextureViewHandle() const noexcept override { return handle_ + 1000; }
    std::uintptr_t depthImageHandle() const noexcept override { return handle_ + 2000; }
    std::uintptr_t depthTextureViewHandle() const noexcept override { return handle_ + 3000; }
    arbitgpu::NativeTextureViewDescriptor colorTextureDescriptor() const noexcept override
    {
        return { backend_, arbitgpu::NativeTextureViewKind::Texture2D,
            arbitgpu::NativeTexturePixelFormat::Rgba8Unorm, colorImageHandle(), colorTextureViewHandle(),
            width_, height_, 1, true, 77, 1 };
    }

private:
    std::string backend_;
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    std::uintptr_t handle_ = 0;
};

class FakeResources final : public arbitgpu::NativeFixtureSceneResources
{
public:
    explicit FakeResources (std::string backend) : backend_ (std::move (backend)) {}
    const std::string& backend() const noexcept override { return backend_; }

private:
    std::string backend_;
};

// The raster fixture backend. Any submission here is a Raster draw.
class FakeBackend final : public arbitgpu::NativeFixtureSceneBackend
{
public:
    arbitgpu::BackendInfo info() const override { return backendInfo; }

    arbitgpu::NativeFixtureScenePreparation prepare (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
        const std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram>&,
        arbitgpu::NativeFixturePrepareUse = arbitgpu::NativeFixturePrepareUse::Blocking) override
    {
        ++preparations;
        arbitgpu::NativeFixtureScenePreparation result;
        result.prepared = true;
        result.resources = std::make_shared<FakeResources> (backendInfo.backend);
        result.stats.staticUploadCount = static_cast<std::uint32_t> (scene->objectCount);
        result.stats.vertexBytes = scene->vertexCount * sizeof (scene->vertices[0]);
        result.stats.indexBytes = scene->indexCount * sizeof (scene->indices[0]);
        result.stats.textureBytes = scene->textureTexelCount > 0
            ? scene->textureTexelCount * sizeof (scene->textureTexels[0]) : 4;
        return result;
    }

    arbitgpu::NativeFixtureSceneSubmission render (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
        const std::shared_ptr<const arbitgpu::NativeFixtureSceneResources>&,
        std::uint32_t width,
        std::uint32_t height,
        arbitgpu::NativeFixtureSceneRuntimeInputs) override
    {
        ++submissions;
        arbitgpu::NativeFixtureSceneSubmission result;
        result.rendered = true;
        result.frame = std::make_shared<FakeFrame> (backendInfo.backend, width, height,
                                                    static_cast<std::uintptr_t> (submissions));
        result.stats.drawCount = static_cast<std::uint32_t> (scene->objectCount);
        result.stats.materialBytes = sizeof (scene->materials[0]) * scene->objectCount;
        return result;
    }

    arbitgpu::BackendInfo backendInfo { true, false, "opengl", "fake", {} };
    int submissions = 0;
    int preparations = 0;
};

class FakeDeformationResources final : public arbitgpu::NativeDeformationResources
{
public:
    const std::string& backend() const noexcept override { return backend_; }

private:
    std::string backend_ = "opengl";
};

// The deformation renderer's backend. Any submission here is a Raster draw.
class FakeDeformationBackend final : public arbitgpu::NativeDeformationBackend
{
public:
    arbitgpu::BackendInfo info() const override { return { true, true, "opengl", "strict-test-device", {} }; }

    arbitgpu::NativeDeformationPreparation prepare (
        const std::shared_ptr<const arbitgpu::NativeDeformationScene>&,
        const std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram>&) override
    {
        arbitgpu::NativeDeformationPreparation result;
        result.prepared = true;
        result.resources = std::make_shared<const FakeDeformationResources>();
        return result;
    }

    arbitgpu::NativeDeformationSubmission render (
        const std::shared_ptr<const arbitgpu::NativeDeformationScene>& source,
        const std::shared_ptr<const visualdeformation::AnimationDeformationSnapshot>&,
        const std::shared_ptr<const arbitgpu::NativeDeformationResources>&,
        std::uint32_t width,
        std::uint32_t height,
        arbitgpu::NativeDeformationRuntimeInputs) override
    {
        ++submissions;
        arbitgpu::NativeDeformationSubmission result;
        result.rendered = true;
        result.frame = std::make_shared<FakeFrame> ("opengl", width, height,
                                                    static_cast<std::uintptr_t> (submissions + 100));
        result.stats.drawCount = 1;
        result.stats.meshId = source->mesh.value;
        return result;
    }

    int submissions = 0;
};

// Stands in for the path tracer. It records each draw and returns a frame
// on the draw's backend.
struct FakeEngineState
{
    int factoryCalls = 0;
    int renders = 0;
    int cancels = 0;
    fixture::RenderUse lastUse = fixture::RenderUse::Preview;
    std::uint32_t lastSamples = 0;
    std::uint32_t lastPreviewSamples = 0;
    std::string lastBackend;
    bool lastHadAnimatedLights = false;
    bool lastHadGlassCaustic = false;
    bool lastHadDiffraction = false;
    bool reportStats = false;
    bool lastDenoise = false;
    bool lastTransportPlaying = false;
    std::optional<fixture::PathTraceFrameStats> statsSeenDuringRender;
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> lastSnapshot;
};

FakeEngineState fakeState;

class FakeEngine final : public fixture::SceneRenderEngine
{
public:
    bool render (const fixture::SceneEngineDraw& draw, fixture::RenderedFrame& output, std::string&) override
    {
        ++fakeState.renders;
        fakeState.statsSeenDuringRender = fixture::lastPathTraceStats (fixture::RenderUse::Preview);
        fakeState.lastSnapshot = draw.snapshot;
        fakeState.lastUse = draw.use;
        fakeState.lastSamples = draw.exportSamples;
        fakeState.lastPreviewSamples = draw.previewSamples;
        fakeState.lastBackend = draw.backend;
        fakeState.lastHadAnimatedLights = draw.inputs != nullptr && ! draw.inputs->animatedLights.empty();
        fakeState.lastHadGlassCaustic = draw.inputs != nullptr && draw.inputs->glassCaustic.has_value();
        fakeState.lastHadDiffraction = draw.diffractionMaterial != nullptr;
        lastUse_ = draw.use;
        lastMode_ = draw.mode;
        fakeState.lastDenoise = draw.denoise;
        fakeState.lastTransportPlaying = draw.transportPlaying;
        output = {};
        output.nativeFrame = std::make_shared<FakeFrame> (draw.backend, draw.dimensions.width,
                                                          draw.dimensions.height,
                                                          static_cast<std::uintptr_t> (500 + fakeState.renders));
        return true;
    }

    void cancel() noexcept override { ++fakeState.cancels; }

    bool pathTraceStats (fixture::PathTraceFrameStats& stats) const noexcept override
    {
        if (! fakeState.reportStats)
            return false;
        stats = {};
        stats.use = lastUse_;
        stats.mode = lastMode_;
        stats.samplesAdded = 768;
        stats.samplesPerPixel = 1;
        stats.bounceLimitPaths = 5;
        stats.limitEndedPaths = 5;
        return true;
    }

private:
    fixture::RenderUse lastUse_ = fixture::RenderUse::Preview;
    fixture::SceneEngineMode lastMode_ = fixture::SceneEngineMode::Accurate;
};

std::shared_ptr<fixture::SceneRenderEngine> makeFakeEngine()
{
    ++fakeState.factoryCalls;
    return std::make_shared<FakeEngine>();
}

std::shared_ptr<fixture::SceneRenderEngine> makeNoEngine() { return nullptr; }

fixture::SceneEngineSelection pathTrace (std::uint32_t samples = 64, std::uint32_t previewSamples = 4096)
{
    return { fixture::SceneEngineKind::PathTrace, samples, previewSamples };
}

fixture::SceneEngineSelection raster() { return {}; }

// ---------------------------------------------------------------- glass mapping

Program surfaceProgram (float transmission, float ior, float dispersion, surfacematerial::TransportMode transport)
{
    Program program;
    program.backend = arbitgpu::NativeFixtureMaterialBackend::OpenGl;
    program.kind = arbitgpu::NativeFixtureMaterialKind::SurfacePbr;
    program.parameters.baseColorMetallic = { 0.9f, 0.95f, 1.0f, 0.0f };
    program.parameters.emissionRoughness = { 0.0f, 0.0f, 0.0f, 0.05f };
    program.parameters.normalOpacity = { 0.0f, 0.0f, 1.0f, 1.0f };
    program.parameters.transmissionIorClearcoat = { transmission, ior, 0.0f, dispersion };
    program.parameters.identifiers[1] = static_cast<std::uint32_t> (transport);
    return program;
}

void glassMappingTests()
{
    using surfacematerial::TransportMode;
    std::string error;

    // A closed volume refracts with the admitted transmission and the d-line
    // index, and its Cauchy B makes the tracer disperse around that index.
    auto closed = surfaceProgram (1.0f, 1.52f, 0.004f, TransportMode::ClosedVolume);
    check (arbitgpu::validNativeFixtureSurfaceParameters (closed.parameters), "the closed glass block is admitted");
    for (const float value : { 0.1f, 0.5f })
        check (arbitgpu::validNativeFixtureSurfaceParameters (
                   surfaceProgram (1.0f, 1.52f, value, TransportMode::ClosedVolume).parameters),
               "the backend gate admits a Cauchy B above 0.02");
    // Negative B leaves the normal dispersion domain. This is not a size rule.
    check (! arbitgpu::validNativeFixtureSurfaceParameters (
               surfaceProgram (1.0f, 1.52f, -0.001f, TransportMode::ClosedVolume).parameters),
           "the backend gate refuses a negative Cauchy B");
    pt::Material material;
    check (ptfixture::mapSurfaceProgram (closed, nullptr, 0.0f, material, error) && error.empty(),
           "a closed glass volume maps");
    check (closeTo (material.transmission, 1.0f) && closeTo (material.ior, 1.52f),
           "closed glass takes transmission and the d-line index from the admitted block");
    check (closeTo (material.dispersion, 0.004f) && material.transport == pt::DielectricTransport::ClosedVolume
               && pt::materialDispersive (material),
           "closed glass keeps its Cauchy B and disperses");
    check (closeTo (material.opacity, 1.0f) && material.alphaMode == pt::MaterialAlphaMode::Opaque,
           "closed glass keeps full coverage, as the raster does");
    check (closeTo (material.roughness, 0.05f) && closeTo (material.metallic, 0.0f)
               && closeTo (material.baseColor.x, 0.9f) && closeTo (material.baseColor.z, 1.0f),
           "closed glass keeps the block's base colour, metallic and roughness");

    auto thinInterface = surfaceProgram (0.75f, 1.33f, 0.0f, TransportMode::ThinInterface);
    check (ptfixture::mapSurfaceProgram (thinInterface, nullptr, 0.0f, material, error)
               && closeTo (material.transmission, 0.75f) && closeTo (material.ior, 1.33f)
               && closeTo (material.opacity, 1.0f),
           "a thin interface refracts with its own transmission and index");

    // A thin surface does not refract. Transmission lowers coverage to
    // 1 - t (1 - F0), F0 = ((ior - 1) / (ior + 1))^2, and the surface blends,
    // as in the raster.
    auto thin = surfaceProgram (0.5f, 1.5f, 0.0f, TransportMode::ThinSurface);
    check (ptfixture::mapSurfaceProgram (thin, nullptr, 0.0f, material, error)
               && closeTo (material.transmission, 0.0f) && closeTo (material.ior, 1.5f)
               && closeTo (material.opacity, 1.0f - 0.5f * (1.0f - 0.04f))
               && material.alphaMode == pt::MaterialAlphaMode::Blend,
           "a thin surface becomes partial coverage with blending, not refraction");
    auto solid = surfaceProgram (0.0f, 1.5f, 0.0f, TransportMode::ThinSurface);
    check (ptfixture::mapSurfaceProgram (solid, nullptr, 0.0f, material, error)
               && closeTo (material.opacity, 1.0f) && material.alphaMode == pt::MaterialAlphaMode::Opaque,
           "an opaque thin surface stays opaque");

    auto diffraction = closed;
    diffraction.kind = arbitgpu::NativeFixtureMaterialKind::DiffractionReflective;
    error.clear();
    check (! ptfixture::mapSurfaceProgram (diffraction, nullptr, 0.0f, material, error)
               && error == "Path Trace renders Surface materials only; choose Raster for diffraction materials",
           "a diffraction material is refused by name");
    auto frameTexture = closed;
    frameTexture.baseColorSource = Program::BaseColorSource::GraphFrameSrgbTexture;
    error.clear();
    check (ptfixture::mapSurfaceProgram (frameTexture, nullptr, 0.0f, material, error) && error.empty()
               && closeTo (material.baseColor.y, 0.95f),
           "a frame-texture base colour maps its factor, and the engine applies the frame");
    auto invalid = closed;
    invalid.parameters.transmissionIorClearcoat[1] = 0.5f;
    error.clear();
    check (! ptfixture::mapSurfaceProgram (invalid, nullptr, 0.0f, material, error) && ! error.empty(),
           "an index below 1 is refused");
}

// A rectangle of two triangles at depth z, facing +Z.
void appendFacingQuad (pt::RenderScene& scene, float z, float half, std::uint32_t material)
{
    const pt::Vec3 corners[4] { { -half, -half, z }, { half, -half, z }, { half, half, z }, { -half, half, z } };
    const int triangles[2][3] { { 0, 1, 2 }, { 0, 2, 3 } };
    for (const auto& triangle : triangles)
    {
        pt::TriangleAttributes attributes;
        attributes.materialIndex = material;
        attributes.primitiveIndex = static_cast<std::uint32_t> (scene.geometry.triangles.size());
        attributes.geometricNormal = { 0.0f, 0.0f, 1.0f };
        for (auto& normal : attributes.shadingNormals)
            normal = attributes.geometricNormal;
        for (const int corner : triangle)
        {
            scene.geometry.positions.push_back (corners[corner]);
            scene.geometry.bounds.grow (corners[corner]);
        }
        scene.geometry.triangles.push_back (attributes);
    }
}

// A thin surface over an emissive background gives the raster's picture,
// alpha * surface + (1 - alpha) * background with alpha = 1 - t (1 - F0).
// Nothing lights the surface's front, so its own shade is 0 in both engines
// and every channel shows t (1 - F0) of the background, untinted. The
// background's index of 1.01 leaves it a specular reflectance near 2.5e-5, so
// light bounced back onto the surface's rear stays far below the bound.
void thinSurfaceCoverageRenderTests()
{
    using surfacematerial::TransportMode;
    const float transmission = 0.5f;
    const float ior = 1.5f;
    const auto thin = surfaceProgram (transmission, ior, 0.0f, TransportMode::ThinSurface);
    pt::Material surface;
    std::string error;
    check (ptfixture::mapSurfaceProgram (thin, nullptr, 0.0f, surface, error), "the thin surface maps for rendering");
    pt::Material background;
    background.baseColor = { 0.0f, 0.0f, 0.0f };
    background.emission = { 1.0f, 1.0f, 1.0f };
    background.ior = 1.01f;

    pt::RenderScene scene;
    scene.materials = { surface, background };
    appendFacingQuad (scene, 0.0f, 2.0f, 0);
    appendFacingQuad (scene, -2.0f, 3.0f, 1);
    scene.commit();

    pt::PinholeCamera camera;
    camera.position = { 0.0f, 0.0f, 3.0f };
    camera.tanHalfFov = 0.3f;
    camera.aspect = 1.0f;
    camera.nearPlane = 0.01f;
    camera.width = 16;
    camera.height = 16;
    pt::PathTracer tracer (2);
    tracer.settings.maxDepth = 4;
    const pt::CancellationToken cancel {};
    const std::uint32_t samples = 64;
    const pt::RenderStats stats = tracer.renderFixed (scene, camera, camera.width, camera.height, samples, 1, cancel);
    check (stats.nonFiniteSamples == 0, "the thin surface render is finite");

    double sum[3] { 0.0, 0.0, 0.0 };
    for (std::uint32_t y = 0; y < camera.height; ++y)
        for (std::uint32_t x = 0; x < camera.width; ++x)
        {
            const pt::Rgb p = tracer.accumulator().mean (x, y);
            sum[0] += p.r;
            sum[1] += p.g;
            sum[2] += p.b;
        }
    const double pixels = static_cast<double> (camera.width) * static_cast<double> (camera.height);
    const double ratio = (ior - 1.0) / (ior + 1.0);
    const double expected = transmission * (1.0 - ratio * ratio);
    // Each sample sees the background (1) or the unlit surface (0), so its
    // standard deviation is at most 0.5 and that of the image mean at most
    // 0.5 / sqrt(16 * 16 * 64) = 0.004 even without stratification. 0.02 is
    // five of those. A ThinSheet mapping would show t (1 - 2R / (1 + R))
    // times the base colour instead: 0.415 in red, far outside the bound.
    const double tolerance = 0.02;
    const double sheetRed = transmission * (1.0 - 0.08 / 1.04) * 0.9;
    check (std::fabs (sheetRed - expected) > 3.0 * tolerance, "the coverage test separates the two thin models");
    bool matches = true;
    for (const double channel : sum)
        matches = matches && std::fabs (channel / pixels - expected) <= tolerance;
    check (matches, "a thin surface over a background renders the raster coverage result in every channel");
}

void inputRefusalTests()
{
    const auto refused = [] (const arbitgpu::NativeFixtureSceneRuntimeInputs& inputs, const char* expected) {
        std::string error;
        return ! ptfixture::admitInputs (inputs, error) && error == expected;
    };

    arbitgpu::NativeFixtureSceneRuntimeInputs plain;
    std::string error;
    check (ptfixture::admitInputs (plain, error), "plain Color inputs are admitted");

    // Glass caustics and diffraction render in the tracer (slice 4); the
    // diffraction inputs need the draw's admitted material, checked later.
    auto caustic = plain;
    caustic.glassCaustic = render::GlassCausticSelection {};
    check (ptfixture::admitInputs (caustic, error), "a glass caustic selection is admitted");
    auto source = plain;
    source.diffractionSourceProgram = std::make_shared<const Program>();
    check (ptfixture::admitInputs (source, error), "a diffraction source program is admitted");
    auto evaluated = plain;
    evaluated.diffractionEvaluatedProgram = std::make_shared<const Program>();
    check (ptfixture::admitInputs (evaluated, error), "an evaluated diffraction program is admitted");

    // Image views, data exports and pass composites read the render outputs.
    check (! ptfixture::readsRenderOutputs (plain), "a plain Color draw carries no render outputs");
    auto composite = plain;
    composite.passComposite = renderpasscomposite::Parameters {};
    check (ptfixture::admitInputs (composite, error) && ptfixture::readsRenderOutputs (composite),
           "a render pass composite is admitted and reads the outputs");
    // A program is only valid beside the composite its output step matches.
    auto program = plain;
    renderpasscomposite::Program compositeProgram;
    compositeProgram.count = 1u;
    program.passProgram = compositeProgram;
    program.passComposite = compositeProgram.steps[compositeProgram.output].parameters;
    check (renderpasscomposite::valid (compositeProgram) && ptfixture::admitInputs (program, error)
               && ptfixture::readsRenderOutputs (program),
           "a render pass program is admitted and reads the outputs");

    auto depth = plain;
    depth.imageOutput = renderpassoutput::Output::Depth;
    check (ptfixture::admitInputs (depth, error) && ptfixture::readsRenderOutputs (depth),
           "a Depth View image output is admitted and reads the outputs");
    auto exported = plain;
    exported.rawExportMask = 1u << static_cast<unsigned> (renderpassoutput::Output::ObjectId);
    check (ptfixture::admitInputs (exported, error) && ptfixture::readsRenderOutputs (exported),
           "an Object ID data export is admitted and reads the outputs");
    auto frame = plain;
    frame.materialFrameTexture = std::make_shared<FakeFrame> ("opengl", 4, 4, 9);
    check (ptfixture::admitInputs (frame, error), "a material frame texture is admitted");

    // Note instancing: one instance per note, at the raster's position and scale.
    auto score = std::make_shared<arbitmod::Score>();
    score->scoreRevision = 1;
    for (const int midi : { 60, 67 })
    {
        arbitmod::Note note;
        note.id = -midi; note.midiNote = midi; note.freqHz = 261.625565f;
        note.velocity = 127; note.lengthBeats = 8;
        score->notes.push_back (note);
    }
    canonicalblockc::FrameKey key;
    key.projectGeneration = key.sourceGeneration = key.helperGeneration = 1;
    key.backendGeneration = key.deviceGeneration = key.scoreGeneration = 1;
    key.beatMapGeneration = key.fpsGeneration = key.loopGeneration = key.seekGeneration = 1;
    key.fps = 24;
    canonicalblockc::FrameProducer producer;
    visualnoteinstancing::Mapping mapping;
    mapping.schemaVersion = visualnoteinstancing::kDirectAxesMappingSchemaVersion;
    mapping.x = mapping.y = mapping.z = visualnoteinstancing::MappingAxis::MidiNote;
    mapping.xScale = 2.0f; mapping.yScale = 0.0f; mapping.zScale = 0.0f;
    mapping.meshScale = 0.5f;
    auto notes = plain;
    notes.canonicalBlockCFrame = producer.evaluate (key, score, 0.0f);
    notes.noteInstanceMapping = mapping;
    check (ptfixture::admitInputs (notes, error), "note instancing is admitted");

    auto mesh = std::make_shared<HarmonicMIDI::grid::Visual3DScene> (videohelper::fixture3d::makeScene());
    mesh->objectCount = 1;
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> scene (mesh), instanced;
    check (ptfixture::noteInstancedScene (scene, notes, instanced, error) && instanced != nullptr
               && instanced->instancedObject == instanced->objects[0].id && instanced->objectInstanceCount == 2,
           "one mesh object is instanced once per sounding note");
    if (instanced != nullptr && instanced->objectInstanceCount == 2)
    {
        const auto batch = arbitgpu::prepareNativeNoteInstances (notes);
        for (std::size_t index = 0; index < 2u; ++index)
        {
            const auto& placed = instanced->objectInstances[index];
            check (std::abs (placed.translation.x - batch.transforms[index][0]) < 1.0e-5f
                       && std::abs (placed.translation.x / 2.0f - static_cast<float> (-batch.identities[index])) < 1.0e-3f
                       && placed.translation.y == 0.0f && placed.scale.x == 0.5f,
                   "a note instance sits at the raster's position and mesh scale");
        }
    }
    auto tinted = notes;
    tinted.noteInstanceMapping->appearanceHigh = { 1.0f, 0.0f, 0.0f, 0.0f };
    check (! ptfixture::noteInstancedScene (scene, tinted, instanced, error)
               && error == "Path Trace does not render note appearance yet; choose Raster",
           "note appearance is refused by name");
    auto scoreOnly = plain;
    scoreOnly.canonicalBlockCFrame = notes.canonicalBlockCFrame;
    check (refused (scoreOnly, "Path Trace does not render deformed or composed animation frames yet; choose Raster"),
           "a score frame without a note mapping stays refused by name");
}

// ---------------------------------------------------------------- FixtureSceneRenderer engine branch

void rendererEngineTests()
{
    using fixture::FixtureSceneRenderer;
    using fixture::RenderedFrame;
    using fixture::kNativeGpuCapability;
    const auto scene = std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (videohelper::fixture3d::makeScene());
    std::string error;

    {
        // No registered factory: the draw fails by name and nothing rasterises.
        fixture::setPathTraceSceneEngineFactory (nullptr);
        FakeBackend backend;
        FixtureSceneRenderer renderer (backend);
        renderer.selectEngine (pathTrace());
        RenderedFrame frame;
        check (! renderer.renderPreview (scene, { 32, 24 }, kNativeGpuCapability, frame, error)
                   && error == "Path Trace is not available in this build",
               "Path Trace without an engine factory is refused by name");
        check (! renderer.renderExport (scene, { 32, 24 }, kNativeGpuCapability, frame, error)
                   && error == "Path Trace is not available in this build",
               "a Path Trace export without an engine factory is refused by name");
        check (backend.submissions == 0 && backend.preparations == 0 && frame.nativeFrame == nullptr,
               "a refused Path Trace draw never reaches the raster backend");

        fixture::setPathTraceSceneEngineFactory (&makeNoEngine);
        check (! renderer.renderPreview (scene, { 32, 24 }, kNativeGpuCapability, frame, error)
                   && error == "Path Trace is not available in this build" && backend.submissions == 0,
               "a factory that returns no engine is refused the same way");

        renderer.selectEngine (raster());
        check (renderer.renderPreview (scene, { 32, 24 }, kNativeGpuCapability, frame, error)
                   && backend.submissions == 1,
               "selecting Raster again draws with the raster backend");
    }

    {
        fakeState = {};
        fixture::setPathTraceSceneEngineFactory (&makeFakeEngine);
        FakeBackend backend;
        FixtureSceneRenderer renderer (backend);
        check (renderer.selectedEngine().engine == fixture::SceneEngineKind::Raster, "a new renderer selects Raster");
        renderer.selectEngine (pathTrace (200, 128));
        check (renderer.selectedEngine().engine == fixture::SceneEngineKind::PathTrace
                   && renderer.selectedEngine().exportSamples == 200
                   && renderer.selectedEngine().previewSamples == 128,
               "the selection keeps the engine and both sample counts");
        RenderedFrame preview;
        check (renderer.renderPreview (scene, { 32, 24 }, kNativeGpuCapability, preview, error)
                   && fakeState.renders == 1 && fakeState.lastUse == fixture::RenderUse::Preview
                   && fakeState.lastPreviewSamples == 128
                   && fakeState.lastBackend == "opengl" && backend.submissions == 0,
               "a Path Trace preview draws with the engine and its preview target");
        check (preview.nativeFrame != nullptr && preview.use == fixture::RenderUse::Preview
                   && preview.sceneId == scene->id && preview.dimensions.width == 32 && ! preview.pending,
               "the engine frame carries the raster's use, scene and size");
        RenderedFrame exported;
        check (renderer.renderExport (scene, { 32, 24 }, kNativeGpuCapability, exported, error)
                   && fakeState.renders == 2 && fakeState.lastUse == fixture::RenderUse::Export
                   && fakeState.lastSamples == 200 && backend.submissions == 0,
               "a Path Trace export passes its fixed sample count to the engine");
        check (fakeState.factoryCalls == 1, "the engine is created once and kept across draws");

        arbitgpu::NativeFixtureSceneRuntimeInputs caustic;
        caustic.glassCaustic = render::GlassCausticSelection {};
        RenderedFrame causticFrame;
        check (renderer.renderPreview (scene, {}, {}, caustic, { 32, 24 }, kNativeGpuCapability, causticFrame, error)
                   && fakeState.renders == 3 && fakeState.lastHadGlassCaustic && ! fakeState.lastHadDiffraction
                   && causticFrame.nativeFrame != nullptr && backend.submissions == 0,
               "a glass caustic draw under Path Trace reaches the engine with its selection, not the raster");

        renderer.selectEngine (raster());
        check (fakeState.cancels == 1, "selecting Raster cancels the path tracer");
        check (renderer.renderPreview (scene, { 32, 24 }, kNativeGpuCapability, preview, error)
                   && backend.submissions == 1 && fakeState.renders == 3,
               "a Raster selection draws with the raster backend only");
        renderer.selectEngine (pathTrace());
        check (renderer.renderPreview (scene, { 32, 24 }, kNativeGpuCapability, preview, error)
                   && fakeState.factoryCalls == 2 && fakeState.renders == 4 && backend.submissions == 1,
               "Path Trace after Raster creates a fresh engine");
    }

    {
        // The real engine traces the admitted cube, then this build's stub
        // upload refuses. Metal is refused the same way until PT-C2.
        fixture::setPathTraceSceneEngineFactory (&ptfixture::makeEngine);
        FakeBackend backend;
        backend.backendInfo.backend = "metal";
        FixtureSceneRenderer renderer (backend);
        renderer.selectEngine (pathTrace (2));
        RenderedFrame frame;
        const bool rendered = renderer.renderPreview (scene, { 8, 6 }, kNativeGpuCapability, frame, error);
        check (! rendered && error.rfind ("Path Trace", 0) == 0 && backend.submissions == 0
                   && backend.preparations == 0 && frame.nativeFrame == nullptr,
               "the real engine's refusal on Metal is reported by name, not replaced by Raster");
        check (error == "Path Trace has no metal frame upload in this build",
               "the real engine traces the cube and stops at the stub Metal upload");
        if (error != "Path Trace has no metal frame upload in this build")
            std::fprintf (stderr, "  real engine error: %s\n", error.c_str());
    }
}

// The pathTraceStats object of the composite probe and viewport_info replies.
// Raster replies do not carry the key. The fake engine stands in where a render
// has to finish, because this build's stub upload refuses the real engine.
void pathTraceStatsReplyTests()
{
    using fixture::FixtureSceneRenderer;
    using fixture::RenderedFrame;
    using fixture::RenderUse;
    using fixture::kNativeGpuCapability;
    const auto scene = std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (videohelper::fixture3d::makeScene());
    std::string error;

    const auto replyFor = [] (RenderUse use)
    {
        nlohmann::json reply { { "width", 32 }, { "compositorBackend", "opengl" } };
        fixture::appendPathTraceStats (reply, fixture::lastPathTraceStats (use));
        return reply;
    };
    const auto countersAreNonNegative = [] (const nlohmann::json& stats)
    {
        for (const char* key : { "samplesAdded", "samplesPerPixel", "nonFiniteSamples", "bounceLimitPaths",
                                 "transparentLimitPaths", "limitEndedPaths", "lightPaths", "causticSplats" })
            if (! stats.contains (key) || ! stats[key].is_number_integer() || stats[key].get<std::int64_t>() < 0)
                return false;
        return true;
    };

    {
        // Raster: the reply stays exactly as it was.
        fixture::clearPathTraceExportStats();
        FakeBackend backend;
        FixtureSceneRenderer renderer (backend);
        RenderedFrame preview, exported;
        check (renderer.renderPreview (scene, { 32, 24 }, kNativeGpuCapability, preview, error)
                   && renderer.renderExport (scene, { 32, 24 }, kNativeGpuCapability, exported, error),
               "the Raster renders behind the stats replies succeed");
        const auto probe = replyFor (RenderUse::Export);
        const auto viewport = replyFor (RenderUse::Preview);
        check (! probe.contains ("pathTraceStats") && ! viewport.contains ("pathTraceStats"),
               "a Raster reply has no pathTraceStats key");
        check (probe == nlohmann::json ({ { "width", 32 }, { "compositorBackend", "opengl" } }),
               "a Raster reply keeps its keys and values");
    }

    {
        // Path Trace: both replies carry the counters and the names.
        fakeState = {};
        fakeState.reportStats = true;
        fixture::setPathTraceSceneEngineFactory (&makeFakeEngine);
        fixture::clearPathTraceExportStats();
        FakeBackend backend;
        FixtureSceneRenderer renderer (backend);
        renderer.selectEngine (pathTrace (16));
        RenderedFrame preview, exported;
        check (renderer.renderPreview (scene, { 32, 24 }, kNativeGpuCapability, preview, error)
                   && renderer.renderExport (scene, { 32, 24 }, kNativeGpuCapability, exported, error),
               "the Path Trace renders behind the stats replies succeed");
        const auto probe = replyFor (RenderUse::Export);
        const auto viewport = replyFor (RenderUse::Preview);
        check (probe.contains ("pathTraceStats") && countersAreNonNegative (probe["pathTraceStats"]),
               "a Path Trace probe reply carries non-negative counters");
        check (probe.value ("/pathTraceStats/engine"_json_pointer, std::string {}) == "pathTrace"
                   && probe.value ("/pathTraceStats/mode"_json_pointer, std::string {}) == "accurate"
                   && probe.value ("/pathTraceStats/use"_json_pointer, std::string {}) == "export",
               "a Path Trace probe reply names the engine, the mode and the use");
        check (probe.value ("/pathTraceStats/imageSource"_json_pointer, std::string {}) == "pathTraceFilm",
               "a Path Trace probe reply names the film as the image source");
        check (probe.value ("/pathTraceStats/temporalDenoise"_json_pointer, true) == false,
               "a Path Trace probe reply says its export blends no playback history");
        check (probe.value ("/pathTraceStats/device"_json_pointer, std::string {}) == "CPU"
                   && probe.value ("/pathTraceStats/deviceRequested"_json_pointer, std::string {}) == "Auto",
               "a Path Trace probe reply names the device and the setting");
        check (probe["width"] == 32 && probe["compositorBackend"] == "opengl",
               "pathTraceStats leaves the other reply keys alone");
        check (viewport.contains ("pathTraceStats") && countersAreNonNegative (viewport["pathTraceStats"])
                   && viewport["pathTraceStats"]["use"] == "preview",
               "a viewport_info reply carries the same object for the preview");

        // A second export in the same frame adds to the first. The next probe
        // starts from nothing, so it never reports an earlier job's paths.
        const auto firstBounces = probe["pathTraceStats"]["bounceLimitPaths"].get<std::uint64_t>();
        check (renderer.renderExport (scene, { 32, 24 }, kNativeGpuCapability, exported, error)
                   && replyFor (RenderUse::Export)["pathTraceStats"]["bounceLimitPaths"].get<std::uint64_t>()
                          == 2 * firstBounces,
               "two Path Trace layers in one frame sum their counters");
        fixture::clearPathTraceExportStats();
        check (! replyFor (RenderUse::Export).contains ("pathTraceStats"),
               "a new probe clears the counters of the last one");

        // A selection change retires the viewport's report at once. A probe
        // reply is not touched by a viewport render, and the other way round.
        check (renderer.renderExport (scene, { 32, 24 }, kNativeGpuCapability, exported, error),
               "an export after the clear succeeds");
        renderer.selectEngine (raster());
        check (! replyFor (RenderUse::Preview).contains ("pathTraceStats"),
               "selecting Raster retires the viewport's report");
        check (renderer.renderPreview (scene, { 32, 24 }, kNativeGpuCapability, preview, error)
                   && ! replyFor (RenderUse::Preview).contains ("pathTraceStats")
                   && replyFor (RenderUse::Export).contains ("pathTraceStats"),
               "a Raster viewport frame leaves the probe's report alone");
        fixture::clearPathTraceExportStats();
    }

    {
        // viewport_info answers while a trace runs, so the frame on screen keeps
        // its report until the next frame ends. A frame without one retires it.
        fakeState = {};
        fakeState.reportStats = true;
        fixture::setPathTraceSceneEngineFactory (&makeFakeEngine);
        FakeBackend backend;
        FixtureSceneRenderer renderer (backend);
        renderer.selectEngine (pathTrace (16));
        RenderedFrame preview;
        check (renderer.renderPreview (scene, { 32, 24 }, kNativeGpuCapability, preview, error)
                   && replyFor (RenderUse::Preview).contains ("pathTraceStats"),
               "a traced Preview frame publishes its report");
        fakeState.statsSeenDuringRender.reset();
        check (renderer.renderPreview (scene, { 32, 24 }, kNativeGpuCapability, preview, error)
                   && fakeState.statsSeenDuringRender.has_value()
                   && replyFor (RenderUse::Preview).contains ("pathTraceStats"),
               "the viewport reads the last frame's report while the next trace runs");

        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> noScene;
        check (! renderer.renderPreview (noScene, { 32, 24 }, kNativeGpuCapability, preview, error)
                   && ! replyFor (RenderUse::Preview).contains ("pathTraceStats"),
               "a failed Preview render retires the report");

        check (renderer.renderPreview (scene, { 32, 24 }, kNativeGpuCapability, preview, error)
                   && replyFor (RenderUse::Preview).contains ("pathTraceStats"),
               "the next traced frame reports again");
        fakeState.reportStats = false;
        check (renderer.renderPreview (scene, { 32, 24 }, kNativeGpuCapability, preview, error)
                   && ! replyFor (RenderUse::Preview).contains ("pathTraceStats"),
               "a traced frame without counters does not keep the last frame's report");
    }

    {
        // An engine without counters reports nothing rather than zeros.
        fakeState = {};
        fixture::setPathTraceSceneEngineFactory (&makeFakeEngine);
        fixture::clearPathTraceExportStats();
        FakeBackend backend;
        FixtureSceneRenderer renderer (backend);
        renderer.selectEngine (pathTrace());
        RenderedFrame exported;
        check (renderer.renderExport (scene, { 32, 24 }, kNativeGpuCapability, exported, error)
                   && ! replyFor (RenderUse::Export).contains ("pathTraceStats"),
               "an engine that reports no counters adds no pathTraceStats key");
    }

    {
        // The real tracer. It traces, then the stub upload refuses, so the
        // counters are read from the engine itself.
        fixture::setPathTraceSceneEngineFactory (&ptfixture::makeEngine);
        FakeBackend backend;
        backend.backendInfo.backend = "metal";
        FixtureSceneRenderer renderer (backend);
        auto selection = pathTrace (2, 8);
        selection.mode = fixture::SceneEngineMode::Fast;
        renderer.selectEngine (selection);
        const std::uint64_t pixels = 8 * 6;
        RenderedFrame frame;
        for (int pass = 0; pass < 3; ++pass)
            (void) renderer.renderPreview (scene, { 8, 6 }, kNativeGpuCapability, frame, error);
        auto engine = renderer.releasePathTraceEngine();
        fixture::PathTraceFrameStats preview;
        check (engine != nullptr && engine->pathTraceStats (preview), "the real engine reports counters after a trace");
        check (preview.samplesAdded + preview.nonFiniteSamples == 3 * pixels && preview.samplesPerPixel == 3,
               "the preview counters cover every pass of the accumulation");
        check (preview.use == RenderUse::Preview && preview.mode == fixture::SceneEngineMode::Fast,
               "the real engine reports the use and mode of its draw");
        check (preview.limitEndedPaths >= preview.bounceLimitPaths,
               "limitEndedPaths includes the bounce-limit paths");

        renderer.adoptPathTraceEngine (std::move (engine));
        (void) renderer.renderExport (scene, { 8, 6 }, kNativeGpuCapability, frame, error);
        engine = renderer.releasePathTraceEngine();
        fixture::PathTraceFrameStats exported;
        check (engine != nullptr && engine->pathTraceStats (exported)
                   && exported.samplesAdded + exported.nonFiniteSamples == 2 * pixels
                   && exported.samplesPerPixel == 2 && exported.use == RenderUse::Export,
               "an export reports its own samples, not the preview's");
    }
}

// ---------------------------------------------------------------- denoise

namespace denoise = videorender::fixture3d::pathtracedenoise;

struct CapturedUpload
{
    std::vector<std::uint8_t> color;
    std::vector<float> depth;
    bool linear = false;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<arbitgpu::CpuSceneFrameLayer> frameLayers;
    std::vector<std::vector<std::uint8_t>> layerLight;
    std::vector<std::vector<float>> layerUvCoverage;
};

CapturedUpload capturedUpload;

// The engine's upload seam: keeps the bytes the compositor would receive.
std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> captureUpload (
    const std::string& backend, const arbitgpu::CpuSceneImage& image, std::string&)
{
    capturedUpload.color = *image.color;
    capturedUpload.depth = *image.depth;
    capturedUpload.linear = image.linearColor;
    capturedUpload.width = image.width;
    capturedUpload.height = image.height;
    capturedUpload.frameLayers = image.frameLayers;
    for (const auto& layer : image.frameLayers)
    {
        capturedUpload.layerLight.push_back (*layer.light);
        capturedUpload.layerUvCoverage.push_back (*layer.uvCoverage);
    }
    return std::make_shared<FakeFrame> (backend, image.width, image.height, 900);
}

std::shared_ptr<fixture::SceneRenderEngine> makeCapturingEngine()
{
    return ptfixture::makeEngineWithUpload (&captureUpload);
}

// R2 provenance: what a Path Trace frame hands the compositor is the film the
// integrator traced, byte for byte. The reference comes straight from a second
// SceneEngine given the setup the fixture engine builds.
void filmProvenanceTests()
{
    const auto scene = std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (videohelper::fixture3d::makeScene());
    std::string error;
    arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
    inputs.linearColor = true;

    pt::EngineSceneSetup setup;
    setup.scene = scene;
    setup.emissionGain = inputs.emissionGain;
    for (std::size_t index = 0; index < scene->objectCount; ++index)
        setup.objectWorlds.push_back (pt::objectWorldMatrix (*scene, index));
    for (std::size_t index = 0; index < arbitgpu::effectiveNativeLightCount (*scene, inputs); ++index)
        setup.lights.push_back (arbitgpu::effectiveNativeLightRecord (*scene, inputs, index));
    pt::EngineView view;
    view.camera = arbitgpu::runtimeCameraRecord (
        *HarmonicMIDI::grid::visual3d_detail::findById (scene->cameras, scene->cameraCount, scene->activeCamera),
        inputs.cameraTranslationOffset, inputs.cameraRotationDegrees, inputs.cameraVerticalFovDegrees);
    view.width = 24;
    view.height = 18;
    view.timeSeconds = inputs.timeSeconds;
    pt::SceneEngine tracer;
    pt::EngineImage film;
    check (tracer.renderExport (setup, view, pt::EngineColor::LinearRgba16f, 2, film, error),
           "the tracer renders the reference film");

    capturedUpload = {};
    const auto engine = makeCapturingEngine();
    fixture::SceneEngineDraw draw;
    draw.snapshot = scene;
    draw.inputs = &inputs;
    draw.dimensions = { 24, 18 };
    draw.use = fixture::RenderUse::Export;
    draw.exportSamples = 2;
    draw.backend = "opengl";
    fixture::RenderedFrame frame;
    check (engine->render (draw, frame, error), "the fixture engine renders a Path Trace frame");
    check (frame.nativeFrame != nullptr && frame.nativeFrame->colorImageHandle() == 900,
           "the frame's colour texture is the uploaded film");
    check (capturedUpload.linear && capturedUpload.width == 24 && capturedUpload.height == 18,
           "the upload is the linear film at the frame size");
    check (! film.colorBytes.empty() && capturedUpload.color == film.colorBytes && capturedUpload.depth == film.depth,
           "the uploaded bytes are the integrator's film");
}

// R5: the frame travels to the upload with the traced uv, and the film alone
// never stands in for it. Nothing in this path can read the frame's pixels.
void frameLayerDrawTests()
{
    auto authored = videohelper::fixture3d::makeScene();
    for (std::size_t index = 0; index < authored.vertexCount; ++index)
        authored.vertices[index].uv = { 0.25f, 0.75f };
    const auto scene = std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (std::move (authored));

    auto program = std::make_shared<Program> (surfaceProgram (0.0f, 1.5f, 0.0f, surfacematerial::TransportMode::ThinSurface));
    program->object = scene->objects[0].id;
    program->baseColorSource = Program::BaseColorSource::GraphFrameSrgbTexture;

    arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
    inputs.linearColor = true;
    const auto imagery = std::make_shared<FakeFrame> ("opengl", 4, 4, 77);
    inputs.materialFrameTexture = imagery;

    fixture::SceneEngineDraw draw;
    draw.snapshot = scene;
    draw.material = program;
    draw.inputs = &inputs;
    draw.dimensions = { 24, 18 };
    draw.use = fixture::RenderUse::Export;
    draw.exportSamples = 4;
    draw.backend = "opengl";

    std::string error;
    fixture::RenderedFrame frame;
    capturedUpload = {};
    auto engine = makeCapturingEngine();
    check (engine->render (draw, frame, error), "a frame-tinted object renders a Path Trace frame: " + error);
    check (capturedUpload.frameLayers.size() == 1 && capturedUpload.frameLayers[0].frame == imagery,
           "the upload receives the draw's frame owner");
    const std::size_t pixels = 24u * 18u;
    check (capturedUpload.layerLight.size() == 1 && capturedUpload.layerLight[0].size() == pixels * 8u
               && capturedUpload.layerUvCoverage[0].size() == pixels * 3u,
           "the layer has light and uv for every pixel");
    float coverage = 0.0f;
    bool uvExact = true;
    for (std::size_t index = 0; index < capturedUpload.layerUvCoverage[0].size(); index += 3)
    {
        const float* uv = capturedUpload.layerUvCoverage[0].data() + index;
        coverage += uv[2];
        if (uv[2] > 0.0f && (! closeTo (uv[0], 0.25f) || ! closeTo (uv[1], 0.75f)))
            uvExact = false;
    }
    check (coverage > 0.0f, "camera samples reach the frame-tinted object");
    check (uvExact, "the layer's uv is the surface's uv");

    // Denoising filters the layer's light, and still hands the frame on.
    draw.denoise = true;
    capturedUpload = {};
    check (engine->render (draw, frame, error) && capturedUpload.frameLayers.size() == 1
               && capturedUpload.frameLayers[0].frame == imagery && capturedUpload.layerLight[0].size() == pixels * 8u,
           "a denoised frame-tinted draw keeps its layer: " + error);
    draw.denoise = false;

    // A frame that would be lost stops the draw by name.
    const auto otherObject = std::make_shared<Program> (*program);
    otherObject->object = {};
    draw.material = otherObject;
    error.clear();
    check (! engine->render (draw, frame, error)
               && error == "Path Trace received a frame texture that no scene object samples",
           "a frame no object samples is not dropped silently");
    draw.material = program;
    inputs.materialFrameTexture = nullptr;
    error.clear();
    check (! engine->render (draw, frame, error)
               && error == "Path Trace material names a frame texture the draw does not carry",
           "a frame-tinted material without its frame is not drawn untinted");
}

void denoiseKernelTests()
{
    check (std::string (denoise::denoiserName (denoise::Denoiser::None)) == "none"
               && std::string (denoise::denoiserName (denoise::Denoiser::NativeFilter)) == "native-filter"
               && std::string (denoise::denoiserName (denoise::Denoiser::OptixAi)) == "optix-ai"
               && std::string (denoise::denoiserName (denoise::Denoiser::Oidn)) == "oidn",
           "each denoiser has its telemetry name");

    check (closeTo (denoise::halfToFloat (pt::halfBits (0.25f)), 0.25f)
               && closeTo (denoise::halfToFloat (pt::halfBits (-3.5f)), -3.5f)
               && denoise::halfToFloat (pt::halfBits (0.0f)) == 0.0f
               && closeTo (denoise::halfToFloat (1u), std::ldexp (1.0f, -24))
               && denoise::halfToFloat (0x7bffu) == 65504.0f,
           "half bits read back as the tracer wrote them");

    std::vector<std::uint8_t> output;
    const std::vector<std::uint8_t> flat (5u * 5u * 4u, 120);
    check (denoise::applyNativeFilter (flat, false, 5, 5, output) && output == flat,
           "a flat RGBA8 image is unchanged");
    check (! denoise::applyNativeFilter (flat, false, 4, 5, output), "a byte count that misses the size is refused");

    // One speck on grey moves toward the grey, and its neighbours stay near grey.
    auto speck = flat;
    speck[(2u * 5u + 2u) * 4u] = 150;
    check (denoise::applyNativeFilter (speck, false, 5, 5, output)
               && output[(2u * 5u + 2u) * 4u] < 150 && output[(2u * 5u + 2u) * 4u] > 120
               && output[(2u * 5u + 1u) * 4u] >= 120 && output[(2u * 5u + 1u) * 4u] < 150,
           "the kernel pulls a small speck toward its surroundings");

    // A hard edge is far outside the range sigma, so it keeps both sides.
    std::vector<std::uint8_t> edge (6u * 2u * 4u, 0);
    for (std::uint32_t y = 0; y < 2; ++y)
        for (std::uint32_t x = 3; x < 6; ++x)
            for (std::uint32_t channel = 0; channel < 4; ++channel)
                edge[(y * 6u + x) * 4u + channel] = 255;
    check (denoise::applyNativeFilter (edge, false, 6, 2, output)
               && output[2u * 4u] <= 2 && output[3u * 4u] >= 253,
           "the kernel keeps a hard edge");

    // Half floats go through the same kernel, and a region split changes nothing.
    std::vector<std::uint8_t> halfSpeck (5u * 5u * 8u, 0);
    for (std::size_t index = 0; index < 5u * 5u * 4u; ++index)
    {
        const std::uint16_t bits = pt::halfBits (index == (2u * 5u + 2u) * 4u ? 0.6f : 0.5f);
        std::memcpy (halfSpeck.data() + index * 2, &bits, sizeof (bits));
    }
    std::vector<std::uint8_t> whole;
    check (denoise::applyNativeFilter (halfSpeck, true, 5, 5, whole), "RGBA16F filters");
    std::uint16_t filteredBits = 0;
    std::memcpy (&filteredBits, whole.data() + (2u * 5u + 2u) * 8u, sizeof (filteredBits));
    const float filtered = denoise::halfToFloat (filteredBits);
    check (filtered > 0.5f && filtered < 0.6f, "the kernel pulls a linear speck toward its surroundings");
    std::vector<float> pixels;
    std::vector<std::uint8_t> split (halfSpeck.size());
    check (denoise::loadPixels (halfSpeck, true, 5, 5, pixels), "the linear image loads");
    denoise::filterRegion (pixels.data(), 5, 5, true, 0, 5, 0, 2, split.data());
    denoise::filterRegion (pixels.data(), 5, 5, true, 0, 2, 2, 5, split.data());
    denoise::filterRegion (pixels.data(), 5, 5, true, 2, 5, 2, 5, split.data());
    std::vector<float> filteredPixels;
    check (denoise::loadPixels (split, true, 5, 5, filteredPixels), "the filtered image loads");
    std::vector<float> scales;
    std::vector<float> totals;
    denoise::restoreEnergy (pixels.data(), filteredPixels.data(), 5, 5, scales, totals, split.data(), [] (auto&& region) {
        region (0u, 5u, 0u, 2u);
        region (0u, 2u, 2u, 5u);
        region (2u, 5u, 2u, 5u);
    });
    check (split == whole, "disjoint regions give the same bytes as one");

    const auto linearImage = [] (std::uint32_t width, std::uint32_t height, auto value) {
        std::vector<std::uint8_t> bytes (static_cast<std::size_t> (width) * height * 8u, 0);
        for (std::uint32_t y = 0; y < height; ++y)
            for (std::uint32_t x = 0; x < width; ++x)
                for (std::uint32_t channel = 0; channel < 4; ++channel)
                {
                    const std::uint16_t bits = pt::halfBits (channel == 3 ? 1.0f : value (x, y));
                    std::memcpy (bytes.data() + (static_cast<std::size_t> (y) * width + x) * 8u + channel * 2u, &bits, sizeof (bits));
                }
        return bytes;
    };
    const auto red = [] (const std::vector<std::uint8_t>& bytes, std::uint32_t width, std::uint32_t x, std::uint32_t y) {
        std::uint16_t bits = 0;
        std::memcpy (&bits, bytes.data() + (static_cast<std::size_t> (y) * width + x) * 8u, sizeof (bits));
        return denoise::halfToFloat (bits);
    };
    const auto redTotal = [&] (const std::vector<std::uint8_t>& bytes, std::uint32_t width, std::uint32_t height) {
        double total = 0.0;
        for (std::uint32_t y = 0; y < height; ++y)
            for (std::uint32_t x = 0; x < width; ++x)
                total += red (bytes, width, x, y);
        return total;
    };
    // Linear radiance has no upper end. A path that found a lamp through a delta order reads
    // hundreds on a face near 0.5. Its energy is real, and nothing near it shares its light.
    const auto spark = linearImage (9, 9, [] (std::uint32_t x, std::uint32_t y) { return x == 4 && y == 4 ? 300.0f : 0.5f; });
    // Each filter runs before its check, because a check's message can be built before its condition.
    const bool sparkRan = denoise::applyNativeFilter (spark, true, 9, 9, output);
    check (sparkRan && red (output, 9, 4, 4) >= 300.0f
               && red (output, 9, 3, 4) < 1.0f && red (output, 9, 2, 2) < 1.0f
               && std::abs (redTotal (output, 9, 9) - redTotal (spark, 9, 9)) < 0.01 * redTotal (spark, 9, 9),
           "a lone linear spark stays one pixel and keeps its energy, centre " + std::to_string (red (output, 9, 4, 4)) + " total "
               + std::to_string (redTotal (output, 9, 9)));

    // Two sparks side by side are a cluster: they share.
    const auto pair = linearImage (9, 9, [] (std::uint32_t x, std::uint32_t y) { return (x == 4 || x == 5) && y == 4 ? 300.0f : 0.5f; });
    const bool pairRan = denoise::applyNativeFilter (pair, true, 9, 9, output);
    check (pairRan && red (output, 9, 4, 4) < 300.0f
               && std::abs (redTotal (output, 9, 9) - redTotal (pair, 9, 9)) < 0.01 * redTotal (pair, 9, 9),
           "a cluster of sparks shares and keeps its energy, centre " + std::to_string (red (output, 9, 4, 4)) + " total "
               + std::to_string (redTotal (output, 9, 9)));

    // A channel at the largest half keeps its input and sends nothing, with either sign. Its
    // neighbours stay finite and near their own value, and they keep their alpha.
    for (const float limit : { 70000.0f, -70000.0f })
    {
        const auto overflow = linearImage (9, 9, [limit] (std::uint32_t x, std::uint32_t y) { return x == 4 && y == 4 ? limit : 0.5f; });
        const bool overflowRan = denoise::applyNativeFilter (overflow, true, 9, 9, output);
        check (overflowRan
                   && red (output, 9, 4, 4) == denoise::halfToFloat (pt::halfBits (limit))
                   && std::abs (red (output, 9, 3, 4) - 0.5f) < 0.01f && std::abs (red (output, 9, 2, 2) - 0.5f) < 0.01f,
               "a pixel at the half limit keeps its input and sends nothing, neighbour " + std::to_string (red (output, 9, 3, 4))
                   + " and " + std::to_string (red (output, 9, 2, 2)));
    }

    // Every pixel but one at the limit: that one has nothing to average, so it stays as it is.
    const auto walled = linearImage (5, 5, [] (std::uint32_t x, std::uint32_t y) { return x == 2 && y == 2 ? 0.5f : 70000.0f; });
    const bool walledRan = denoise::applyNativeFilter (walled, true, 5, 5, output);
    check (walledRan && std::abs (red (output, 5, 2, 2) - 0.5f) < 0.01f,
           "a pixel walled in by the half limit keeps its value, got " + std::to_string (red (output, 5, 2, 2)));
    std::uint16_t walledAlpha = 0;
    std::memcpy (&walledAlpha, output.data() + (2u * 5u + 2u) * 8u + 6u, sizeof (walledAlpha));
    check (denoise::halfToFloat (walledAlpha) == 1.0f, "a pixel beside the half limit keeps its alpha");

    // Light that arrives as sparse hits: the 5 x 5 median is 0, and the mean is 1/16.
    const auto sparse = linearImage (64, 64, [] (std::uint32_t x, std::uint32_t y) { return x % 4 == 0 && y % 4 == 0 ? 1.0f : 0.0f; });
    const bool sparseRan = denoise::applyNativeFilter (sparse, true, 64, 64, output);
    check (sparseRan
               && std::abs (redTotal (output, 64, 64) - redTotal (sparse, 64, 64)) < 0.01 * redTotal (sparse, 64, 64),
           "sparse hits keep their energy: " + std::to_string (redTotal (output, 64, 64)) + " of "
               + std::to_string (redTotal (sparse, 64, 64)));

    // Dark noise: the view transform clamps below zero, so a negative pixel would add light.
    const auto noisy = linearImage (64, 64, [] (std::uint32_t x, std::uint32_t y) {
        const std::uint32_t hash = (x * 73856093u) ^ (y * 19349663u);
        return (hash >> 7) % 11u == 0u ? 4.0f : 0.01f;
    });
    check (denoise::applyNativeFilter (noisy, true, 64, 64, output), "dark noise filters");
    float lowest = 0.0f;
    for (std::uint32_t y = 0; y < 64; ++y)
        for (std::uint32_t x = 0; x < 64; ++x)
            lowest = std::min (lowest, red (output, 64, x, y));
    check (lowest >= 0.0f && std::abs (redTotal (output, 64, 64) - redTotal (noisy, 64, 64)) < 0.01 * redTotal (noisy, 64, 64),
           "dark noise stays above zero and keeps its energy, lowest " + std::to_string (lowest) + " total "
               + std::to_string (redTotal (output, 64, 64)) + " of " + std::to_string (redTotal (noisy, 64, 64)));

    // A flat image stays flat up to its corners, at any size down to one pixel.
    for (const std::uint32_t size : { 1u, 2u, 3u, 16u })
    {
        const auto flat = linearImage (size, size, [] (std::uint32_t, std::uint32_t) { return 0.5f; });
        const bool flatRan = denoise::applyNativeFilter (flat, true, size, size, output);
        float furthest = 0.0f;
        for (std::uint32_t y = 0; y < size; ++y)
            for (std::uint32_t x = 0; x < size; ++x)
                furthest = std::max (furthest, std::abs (red (output, size, x, y) - 0.5f));
        check (flatRan && furthest < 0.001f,
               "a flat image of side " + std::to_string (size) + " stays flat, furthest off by " + std::to_string (furthest));
    }

    // A fan's edge is a straight boundary, and must stay one at any brightness.
    const auto linearEdge = linearImage (12, 4, [] (std::uint32_t x, std::uint32_t) { return x < 6 ? 0.05f : 8.0f; });
    const bool linearEdgeRan = denoise::applyNativeFilter (linearEdge, true, 12, 4, output);
    check (linearEdgeRan && red (output, 12, 5, 2) < 0.06f && red (output, 12, 6, 2) > 7.9f,
           "the kernel keeps a bright linear edge, got " + std::to_string (red (output, 12, 5, 2)) + " and " + std::to_string (red (output, 12, 6, 2)));
}

void denoiseEngineTests()
{
    const auto scene = std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (videohelper::fixture3d::makeScene());
    std::string error;
    // These frames trace on the CPU, so Auto asks for OIDN. Without its library
    // Auto falls back to the native filter and says why; the tests pin that case.
    denoise::setOidnLibraryForTests ("/nonexistent/libOpenImageDenoise.so.2");

    // Each sequence runs on a fresh engine. The second draw moves time, which
    // puts the tracer on its fixed-sample playback frame.
    struct Frames
    {
        CapturedUpload first;
        CapturedUpload moved;
        std::string name;
        bool rendered = false;
    };
    const auto run = [&] (bool denoised, bool linear, fixture::RenderUse use) {
        Frames frames;
        const auto engine = makeCapturingEngine();
        arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
        inputs.linearColor = linear;
        fixture::SceneEngineDraw draw;
        draw.snapshot = scene;
        draw.inputs = &inputs;
        draw.dimensions = { 24, 18 };
        draw.use = use;
        draw.exportSamples = 2;
        draw.previewSamples = 2;
        draw.backend = "opengl";
        draw.denoise = denoised;
        fixture::RenderedFrame frame;
        frames.rendered = engine != nullptr && engine->render (draw, frame, error);
        frames.first = capturedUpload;
        inputs.timeSeconds = 0.5f;
        frames.rendered = frames.rendered && engine->render (draw, frame, error);
        frames.moved = capturedUpload;
        frames.name = engine != nullptr ? engine->denoiserRan() : "";
        return frames;
    };
    const auto filteredCopy = [] (const CapturedUpload& image) {
        std::vector<std::uint8_t> output;
        denoise::applyNativeFilter (image.color, image.linear, image.width, image.height, output);
        return output;
    };

    for (const bool linear : { false, true })
    {
        const auto off = run (false, linear, fixture::RenderUse::Preview);
        const auto offAgain = run (false, linear, fixture::RenderUse::Preview);
        const auto on = run (true, linear, fixture::RenderUse::Preview);
        check (off.rendered && offAgain.rendered && on.rendered, "the denoise sequences render");
        check (off.first.color == offAgain.first.color && off.moved.color == offAgain.moved.color,
               "the tracer's frames repeat, so the comparisons below are exact");
        check (off.name == "none", "a frame with the switch off reports no denoiser");
        check (on.name == "native-filter", "a denoised frame names the native filter");
        check (on.first.color != off.first.color, "the switch changes a preview frame");
        check (on.first.color == filteredCopy (off.first),
               "a preview frame with the switch on is the filtered frame with the switch off");
        check (on.moved.color != off.moved.color && on.moved.color == filteredCopy (off.moved),
               "a fixed-sample playback frame takes the same filter");
        check (on.first.depth == off.first.depth, "the filter leaves depth alone");
    }

    const auto exportOff = run (false, false, fixture::RenderUse::Export);
    const auto exportOn = run (true, false, fixture::RenderUse::Export);
    check (exportOff.rendered && exportOn.rendered && exportOn.first.color == filteredCopy (exportOff.first),
           "an export frame takes the same filter, so a settled preview still matches it");

    {
        // A capped preview reuses the last image. The accumulation keeps the
        // unfiltered image, and the switch applies to a frame already resolved.
        const auto engine = makeCapturingEngine();
        arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
        fixture::SceneEngineDraw draw;
        draw.snapshot = scene;
        draw.inputs = &inputs;
        draw.dimensions = { 24, 18 };
        draw.use = fixture::RenderUse::Preview;
        draw.exportSamples = 1;
        draw.previewSamples = 1;
        draw.backend = "opengl";
        fixture::RenderedFrame frame;
        check (engine->render (draw, frame, error), "a capped preview renders");
        const auto resolved = capturedUpload.color;
        check (engine->render (draw, frame, error) && capturedUpload.color == resolved,
               "the capped preview repeats its image");
        draw.denoise = true;
        check (engine->render (draw, frame, error), "the switch turns on at the cap");
        const auto denoisedAtCap = capturedUpload.color;
        check (denoisedAtCap != resolved, "the frame at the cap changes when the switch turns on");
        std::vector<std::uint8_t> expected;
        denoise::applyNativeFilter (resolved, false, 24, 18, expected);
        check (denoisedAtCap == expected, "the frame at the cap is the filtered resolved image");
        check (engine->render (draw, frame, error) && capturedUpload.color == expected,
               "the filter does not stack on repeated capped frames");
        draw.denoise = false;
        check (engine->render (draw, frame, error) && capturedUpload.color == resolved
                   && std::string (engine->denoiserRan()) == "none",
               "turning the switch off restores the unfiltered image and the report");
    }

    {
        // The selection reaches the draw and the renderer reports what ran.
        fakeState = {};
        fixture::setPathTraceSceneEngineFactory (&makeFakeEngine);
        FakeBackend backend;
        fixture::FixtureSceneRenderer renderer (backend);
        auto selection = pathTrace (2, 2);
        renderer.selectEngine (selection);
        fixture::RenderedFrame frame;
        check (renderer.renderPreview (scene, { 8, 6 }, fixture::kNativeGpuCapability, frame, error)
                   && ! fakeState.lastDenoise,
               "the default selection leaves denoise off");
        selection.denoise = true;
        renderer.selectEngine (selection);
        check (renderer.renderPreview (scene, { 8, 6 }, fixture::kNativeGpuCapability, frame, error)
                   && fakeState.lastDenoise && fakeState.factoryCalls == 1,
               "the selection reaches the draw without replacing the engine");
        check (std::string (fixture::pathTraceDenoiserRanOnThisThread()) == "none",
               "the report is the engine's, and this fake engine ran none");

        fixture::setPathTraceSceneEngineFactory (&makeCapturingEngine);
        fixture::FixtureSceneRenderer tracing (backend);
        tracing.selectEngine (selection);
        check (tracing.renderPreview (scene, { 8, 6 }, fixture::kNativeGpuCapability, frame, error)
                   && std::string (fixture::pathTraceDenoiserRanOnThisThread()) == "native-filter",
               "the renderer reports the native filter for a denoised draw");
        const auto traced = fixture::lastPathTraceStats (fixture::RenderUse::Preview);
        check (traced.has_value() && std::string (traced->denoiserRequested) == "auto"
                   && traced->denoiserReason != nullptr
                   && traced->denoiserReason->find ("OIDN did not run: the OIDN library is not installed") != std::string::npos,
               "the native filter is a named fallback with the reason OIDN did not run");
        selection.denoise = false;
        tracing.selectEngine (selection);
        check (tracing.renderPreview (scene, { 8, 6 }, fixture::kNativeGpuCapability, frame, error)
                   && std::string (fixture::pathTraceDenoiserRanOnThisThread()) == "none",
               "the report clears when the switch goes off");
        fixture::setPathTraceSceneEngineFactory (&ptfixture::makeEngine);
    }

    {
        // Raster has no denoise step.
        fixture::setPathTraceSceneEngineFactory (&makeFakeEngine);
        FakeBackend backend;
        fixture::FixtureSceneRenderer renderer (backend);
        auto selection = raster();
        selection.denoise = true;
        renderer.selectEngine (selection);
        fixture::RenderedFrame frame;
        check (renderer.renderPreview (scene, { 8, 6 }, fixture::kNativeGpuCapability, frame, error)
                   && backend.submissions == 1
                   && std::string (fixture::pathTraceDenoiserRanOnThisThread()) == "none",
               "Raster ignores the selection's switch and reports no denoiser");
        fixture::setPathTraceSceneEngineFactory (&ptfixture::makeEngine);
    }
    denoise::setOidnLibraryForTests (nullptr);
}

// Plan 3, Path Trace and Playback: what the viewport does, and says it does, on
// the real engine, and that a settled viewport is the export frame.
void viewportConvergenceTests()
{
    const auto scene = std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (videohelper::fixture3d::makeScene());
    std::string error;
    const auto statsOf = [] (const fixture::SceneRenderEngine& engine) {
        fixture::PathTraceFrameStats stats;
        check (engine.pathTraceStats (stats), "the engine reports the frame it traced");
        return stats;
    };

    arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
    inputs.linearColor = true;
    inputs.emissionGain = 2.0f;
    fixture::SceneEngineDraw draw;
    draw.snapshot = scene;
    draw.inputs = &inputs;
    draw.dimensions = { 24, 18 };
    draw.exportSamples = 3;
    draw.previewSamples = 9;
    draw.backend = "opengl";
    draw.denoise = true;
    fixture::RenderedFrame frame;

    {
        // An Accurate viewport converges to the export count one pass per draw,
        // then settles on the denoised bytes of the export of the same inputs.
        draw.use = fixture::RenderUse::Preview;
        const auto engine = makeCapturingEngine();
        for (std::uint32_t pass = 1; pass <= 3; ++pass)
        {
            check (engine->render (draw, frame, error), "the viewport pass renders");
            const auto stats = statsOf (*engine);
            check (stats.samplesPerPixel == pass && stats.sampleTarget == 3,
                   "the viewport reports its samples and its target");
            check (std::string (stats.renderActivity) == (pass < 3 ? "converging" : "settled"),
                   "the viewport says converging until the target and settled at it");
            check (std::string (stats.denoiser) == "native-filter", "the viewport names the denoiser that ran");
        }
        const auto settled = capturedUpload;
        check (engine->render (draw, frame, error) && statsOf (*engine).samplesPerPixel == 3
                   && capturedUpload.color == settled.color,
               "a settled viewport adds nothing and repeats its image");

        draw.use = fixture::RenderUse::Export;
        const auto exporter = makeCapturingEngine();
        check (exporter->render (draw, frame, error), "the export renders");
        const auto exported = statsOf (*exporter);
        check (std::string (exported.renderActivity) == "export" && exported.sampleTarget == 3
                   && std::string (exported.denoiser) == "native-filter",
               "the export reports its activity, count and denoiser");
        check (capturedUpload.color == settled.color && capturedUpload.depth == settled.depth,
               "the settled denoised viewport is the export, so it strips no scene input");
    }

    {
        // A Fast viewport has its own target beside the export count.
        draw.use = fixture::RenderUse::Preview;
        draw.mode = fixture::SceneEngineMode::Fast;
        const auto engine = makeCapturingEngine();
        check (engine->render (draw, frame, error) && statsOf (*engine).sampleTarget == 9,
               "a Fast viewport targets its preview samples, not the export count");
        draw.mode = fixture::SceneEngineMode::Accurate;
    }

    {
        // While the transport plays, a repeated time stays on the playback budget
        // and keeps its image. Stopping converges from there.
        draw.use = fixture::RenderUse::Preview;
        draw.exportSamples = 16;
        draw.transportPlaying = true;
        const auto engine = makeCapturingEngine();
        check (engine->render (draw, frame, error), "the playing frame renders");
        const auto played = statsOf (*engine);
        check (std::string (played.renderActivity) == "playback-fixed" && played.sampleTarget == 4
                   && played.samplesPerPixel == 4,
               "a playing frame reaches the fixed budget and reports it");
        const auto playedImage = capturedUpload.color;
        check (engine->render (draw, frame, error), "the repeated playing frame renders");
        const auto repeated = statsOf (*engine);
        check (std::string (repeated.renderActivity) == "playback-fixed" && repeated.samplesPerPixel == 4
                   && capturedUpload.color == playedImage,
               "a repeated time while playing traces nothing more and keeps the image");
        draw.transportPlaying = false;
        check (engine->render (draw, frame, error), "the stopped frame renders");
        const auto stopped = statsOf (*engine);
        check (std::string (stopped.renderActivity) == "converging" && stopped.sampleTarget == 16
                   && stopped.samplesPerPixel == 5,
               "stopping keeps converging toward the viewport target");
        // Time moves: the accumulation restarts at the budget.
        inputs.timeSeconds = 0.5f;
        check (engine->render (draw, frame, error) && statsOf (*engine).samplesPerPixel == 4,
               "a moved time restarts the accumulation");
        // The camera moves: the accumulation restarts while stopped.
        check (engine->render (draw, frame, error) && statsOf (*engine).samplesPerPixel == 5,
               "the same time accumulates again");
        inputs.cameraVerticalFovDegrees = 20.0f;
        check (engine->render (draw, frame, error) && statsOf (*engine).samplesPerPixel == 1,
               "a moved camera restarts the accumulation");
    }

    {
        // The viewport's transport state reaches Preview draws and never Export.
        fakeState = {};
        fixture::setPathTraceSceneEngineFactory (&makeFakeEngine);
        FakeBackend backend;
        fixture::FixtureSceneRenderer renderer (backend);
        renderer.selectEngine (pathTrace (2, 2));
        fixture::setPathTraceTransportPlaying (true);
        check (renderer.renderPreview (scene, { 8, 6 }, fixture::kNativeGpuCapability, frame, error)
                   && fakeState.lastTransportPlaying,
               "a Preview draw carries the playing transport");
        check (renderer.renderExport (scene, { 8, 6 }, fixture::kNativeGpuCapability, frame, error)
                   && ! fakeState.lastTransportPlaying,
               "an Export draw never carries it");
        fixture::setPathTraceTransportPlaying (false);
        check (renderer.renderPreview (scene, { 8, 6 }, fixture::kNativeGpuCapability, frame, error)
                   && ! fakeState.lastTransportPlaying,
               "a stopped transport clears it");
        fixture::setPathTraceSceneEngineFactory (&ptfixture::makeEngine);
    }

    {
        // The reply carries what the header reads.
        fixture::PathTraceFrameStats stats;
        stats.renderActivity = "settled";
        stats.sampleTarget = 65535;
        stats.denoiser = "native-filter";
        nlohmann::json reply;
        fixture::appendPathTraceStats (reply, stats);
        check (reply["pathTraceStats"]["renderActivity"] == "settled"
                   && reply["pathTraceStats"]["sampleTarget"] == 65535
                   && reply["pathTraceStats"]["denoiser"] == "native-filter",
               "the stats reply names the activity, the target and the denoiser");
    }
}

// The real engine on caustic and diffraction draws (slice 4). Tracing ends at
// this build's stub Metal upload, so that message proves admission, scene
// setup and the trace all ran.
void causticDrawTests()
{
    const auto scene = std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (videohelper::fixture3d::makeScene());
    constexpr const char* traced = "Path Trace has no metal frame upload in this build";
    const auto draw = [&] (const arbitgpu::NativeFixtureSceneRuntimeInputs& inputs,
                           std::shared_ptr<const diffractionmaterial::AdmittedDiffractionMaterialIR> material,
                           HarmonicMIDI::grid::SceneObjectId object, std::string& error) {
        const auto engine = ptfixture::makeEngine();
        fixture::SceneEngineDraw request;
        request.snapshot = scene;
        request.inputs = &inputs;
        request.dimensions = { 8, 6 };
        request.use = fixture::RenderUse::Export;
        request.exportSamples = 2;
        request.backend = "metal";
        request.diffractionObject = object;
        request.diffractionMaterial = std::move (material);
        fixture::RenderedFrame frame;
        return engine != nullptr && engine->render (request, frame, error);
    };

    std::string error;
    arbitgpu::NativeFixtureSceneRuntimeInputs caustic;
    caustic.glassCaustic = render::GlassCausticSelection {};
    check (! draw (caustic, nullptr, {}, error) && error == traced,
           "a glass caustic draw traces in the real engine without a gate refusal");

    const auto admitted = diffractionmaterial::admit (diffractionmaterial::makeAluminiumBinaryGratingPreset(), error);
    check (admitted.has_value(), "the aluminium grating preset admits");
    if (! admitted)
        return;
    const auto grating = std::make_shared<const diffractionmaterial::AdmittedDiffractionMaterialIR> (*admitted);
    arbitgpu::NativeFixtureSceneRuntimeInputs plain;
    check (! draw (plain, grating, videohelper::fixture3d::kCubeObjectId, error) && error == traced,
           "a diffraction material on its object traces in the real engine without a gate refusal");
    arbitgpu::NativeFixtureSceneRuntimeInputs evaluated;
    evaluated.diffractionEvaluatedProgram = std::make_shared<const Program>();
    check (! draw (evaluated, grating, videohelper::fixture3d::kCubeObjectId, error) && error == traced,
           "runtime diffraction inputs trace with their admitted material");
    check (! draw (evaluated, nullptr, {}, error)
               && error == "Path Trace needs the admitted diffraction material of these diffraction inputs",
           "runtime diffraction inputs without their material are refused by name");
    check (! draw (plain, grating, HarmonicMIDI::grid::SceneObjectId { 999 }, error)
               && error == "Path Trace diffraction material names no scene object",
           "a diffraction material on a missing object is refused by name");

    // A masked foil whose object carries only the foil's diffraction program, as the
    // Holographic Trading Card does: the foil traces as the grating and the mask
    // shows the object's scene material, instead of refusing the frame.
    auto foilProgram = std::make_shared<Program>();
    foilProgram->backend = arbitgpu::NativeFixtureMaterialBackend::Metal;
    foilProgram->kind = arbitgpu::NativeFixtureMaterialKind::DiffractionReflective;
    foilProgram->object = videohelper::fixture3d::kCubeObjectId;
    const auto engine = ptfixture::makeEngine();
    fixture::SceneEngineDraw masked;
    masked.snapshot = scene;
    masked.material = foilProgram;
    masked.inputs = &plain;
    masked.dimensions = { 8, 6 };
    masked.use = fixture::RenderUse::Export;
    masked.exportSamples = 2;
    masked.backend = "metal";
    masked.diffractionObject = videohelper::fixture3d::kCubeObjectId;
    masked.diffractionMaterial = grating;
    masked.diffractionField = std::array<std::array<float, 4>, 4> { { { 1.0f, 0.0f, 720.0f, 0.15f },
                                                                      { 0.0f, 1.0f, 980.0f, 0.45f },
                                                                      { 0.7f, 0.7f, 1260.0f, 0.7f },
                                                                      { 0.7f, -0.7f, 1540.0f, 0.95f } } };
    fixture::RenderedFrame maskedFrame;
    error.clear();
    check (engine != nullptr && ! engine->render (masked, maskedFrame, error) && error == traced,
           "a masked foil on an object with only its diffraction program traces without a refusal");
}

// ---------------------------------------------------------------- imported scene routes

std::uint32_t readU32 (const std::uint8_t* bytes)
{
    return static_cast<std::uint32_t> (bytes[0]) | static_cast<std::uint32_t> (bytes[1]) << 8u
        | static_cast<std::uint32_t> (bytes[2]) << 16u | static_cast<std::uint32_t> (bytes[3]) << 24u;
}

void appendU32 (std::vector<std::uint8_t>& bytes, std::uint32_t value)
{
    for (unsigned shift = 0; shift < 32u; shift += 8u)
        bytes.push_back (static_cast<std::uint8_t> (value >> shift));
}

void appendFloat (std::vector<std::uint8_t>& bytes, float value)
{
    std::uint32_t encoded = 0;
    std::memcpy (&encoded, &value, sizeof (encoded));
    appendU32 (bytes, encoded);
}

std::vector<std::uint8_t> makeGlb (const nlohmann::json& root, std::vector<std::uint8_t> bin)
{
    auto text = root.dump();
    while ((text.size() & 3u) != 0)
        text.push_back (' ');
    while ((bin.size() & 3u) != 0)
        bin.push_back (0);
    std::vector<std::uint8_t> bytes;
    appendU32 (bytes, 0x46546c67u);
    appendU32 (bytes, 2u);
    appendU32 (bytes, static_cast<std::uint32_t> (12u + 8u + text.size() + 8u + bin.size()));
    appendU32 (bytes, static_cast<std::uint32_t> (text.size()));
    appendU32 (bytes, 0x4e4f534au);
    bytes.insert (bytes.end(), text.begin(), text.end());
    appendU32 (bytes, static_cast<std::uint32_t> (bin.size()));
    appendU32 (bytes, 0x004e4942u);
    bytes.insert (bytes.end(), bin.begin(), bin.end());
    return bytes;
}

// The animated starter with its morph channel replaced by a translation track
// on the light node: camera/light animation only. The morph mesh stays bound, no
// clip animates it, and the scene renderer draws it at its authored weights.
std::vector<std::uint8_t> lightOnlyStarter (float restWeight = 0.0f)
{
    const auto& bytes = visualstartermodel::kAnimatedTriangleGlb;
    const auto jsonSize = readU32 (bytes.data() + 12u);
    auto root = nlohmann::json::parse (bytes.begin() + 20u, bytes.begin() + 20u + jsonSize);
    const auto binHeader = 20u + jsonSize;
    const auto binSize = readU32 (bytes.data() + binHeader);
    std::vector<std::uint8_t> bin (bytes.begin() + binHeader + 8u, bytes.begin() + binHeader + 8u + binSize);
    const auto input = root["animations"][0]["samplers"][0]["input"].get<std::size_t>();
    const auto count = root["accessors"][input]["count"].get<std::size_t>();
    const auto offset = bin.size();
    for (std::size_t key = 0; key < count; ++key)
        for (const auto value : { static_cast<float> (key) * 0.25f, 0.0f, 0.0f })
            appendFloat (bin, value);
    const auto view = root["bufferViews"].size();
    root["bufferViews"].push_back ({ { "buffer", 0 }, { "byteOffset", offset }, { "byteLength", count * 12u } });
    const auto output = root["accessors"].size();
    root["accessors"].push_back ({ { "bufferView", view }, { "componentType", 5126 }, { "count", count },
                                   { "type", "VEC3" } });
    auto& animation = root["animations"][0];
    // The decoder refuses an unused sampler, so the morph sampler goes with its channel.
    animation["samplers"] = nlohmann::json::array ({ { { "input", input }, { "output", output }, { "interpolation", "LINEAR" } } });
    animation["channels"] = nlohmann::json::array ({ { { "sampler", 0 }, { "target", { { "node", 1 }, { "path", "translation" } } } } });
    root["meshes"][0]["weights"][0] = restWeight;
    root["buffers"][0]["byteLength"] = bin.size();
    return makeGlb (root, std::move (bin));
}

std::string encodeBase64 (const std::vector<std::uint8_t>& bytes)
{
    static constexpr char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string encoded;
    for (std::size_t offset = 0; offset < bytes.size(); offset += 3u)
    {
        const auto remaining = bytes.size() - offset;
        const auto value = static_cast<std::uint32_t> (bytes[offset]) << 16u
            | (remaining > 1u ? static_cast<std::uint32_t> (bytes[offset + 1u]) << 8u : 0u)
            | (remaining > 2u ? static_cast<std::uint32_t> (bytes[offset + 2u]) : 0u);
        encoded.push_back (alphabet[(value >> 18u) & 63u]);
        encoded.push_back (alphabet[(value >> 12u) & 63u]);
        encoded.push_back (remaining > 1u ? alphabet[(value >> 6u) & 63u] : '=');
        encoded.push_back (remaining > 2u ? alphabet[value & 63u] : '=');
    }
    return encoded;
}

visualanimationimport::ExactContentAssetKey exactKey (const char* id, const std::vector<std::uint8_t>& bytes)
{
    return { id, 1u, videohelper::modelpayload::detail::digestHex (bytes), "model/gltf-binary", bytes.size() };
}

bool publish (videohelper::modelpayload::Store& store, const char* transferId,
              const visualanimationimport::ExactContentAssetKey& key, const std::vector<std::uint8_t>& bytes)
{
    return store.begin (transferId, key) && store.appendBase64 (transferId, 0u, encodeBase64 (bytes))
        && store.commit (transferId);
}

visualanimationimport::Request starterOperation (const visualanimationimport::ExactContentAssetKey& key)
{
    visualanimationimport::Request operation;
    operation.sourceStableId = 1;
    operation.deformationStableId = 2;
    operation.schedule = { 1, 2 };
    operation.asset = key;
    operation.sceneIndex = 0u;
    operation.meshStableId = 1u;
    operation.animationClipStableId = 1u;
    operation.clipName = std::string (visualstartermodel::kClipName);
    return operation;
}

// A constant Surface material bound to one object, with its MaterialId literal.
surfacematerialbinding::ImportedSceneMaterialRequest surfaceMaterialRequest (
    decltype (HarmonicMIDI::grid::Visual3DScene::id) scene, HarmonicMIDI::grid::SceneObjectId object, std::uint32_t materialId)
{
    surfacematerial::ProgramDescription program;
    for (const auto output : surfacematerial::kSurfaceOutputs)
    {
        surfacematerial::Operation operation;
        operation.id = static_cast<surfacematerial::ValueId> (program.operations.size() + 1u);
        operation.resultType = surfacematerial::outputType (output);
        operation.kind = operation.resultType == surfacematerial::ValueType::UInt
            ? surfacematerial::OperationKind::UIntConstant : surfacematerial::OperationKind::FloatConstant;
        if (output == surfacematerial::OutputSemantic::BaseColor) operation.literal = { 0.8f, 0.15f, 0.05f, 0.0f };
        else if (output == surfacematerial::OutputSemantic::Normal) operation.literal = { 0.0f, 0.0f, 1.0f, 0.0f };
        else if (output == surfacematerial::OutputSemantic::Opacity) operation.literal[0] = 1.0f;
        else if (output == surfacematerial::OutputSemantic::Ior) operation.literal[0] = 1.5f;
        else if (output == surfacematerial::OutputSemantic::MaterialId) operation.unsignedLiteral = materialId;
        program.outputs[static_cast<std::size_t> (output)] = operation.id;
        program.operations.push_back (operation);
    }
    std::string admissionError;
    const auto admitted = surfacematerial::admit (program, admissionError);
    check (admitted.has_value(), "the bound mesh material admits: " + admissionError);
    surfacematerialbinding::ImportedSceneMaterialRequest material;
    material.scene = scene;
    material.sceneRevision = material.structuralRevision = material.evaluationRevision = material.programRevision = 1;
    material.program = program;
    material.binding.targetKind = surfacematerialbinding::BindingTargetKind::ObjectOverride;
    material.binding.object = object;
    material.binding.surfaceMaterialDigest = admitted ? admitted->structuralDigest() : std::string (64, '0');
    material.binding.surfaceMaterialRevision = 1;
    return material;
}

void importedRouteTests()
{
    using videohelper::modelpayload::ImportedAnimatedSceneReceipt;
    using videohelper::modelpayload::ImportedAnimatedSceneRequest;
    using videohelper::modelpayload::ImportedAnimatedScenePayloadExecution;
    using videohelper::modelpayload::ImportedSceneExecutionReceipt;
    using videohelper::modelpayload::ImportedScenePayloadExecution;
    using videohelper::modelpayload::ImportedSceneRequest;

    const std::vector<std::uint8_t> starterBytes { visualstartermodel::kAnimatedTriangleGlb.begin(),
                                                   visualstartermodel::kAnimatedTriangleGlb.end() };
    const auto lightOnlyBytes = lightOnlyStarter();
    const auto starterKey = exactKey ("pathtrace-route-starter", starterBytes);
    const auto lightOnlyKey = exactKey ("pathtrace-route-light-only", lightOnlyBytes);
    videohelper::modelpayload::Store payloads;
    check (publish (payloads, "pathtrace-route-starter-transfer", starterKey, starterBytes)
               && publish (payloads, "pathtrace-route-light-only-transfer", lightOnlyKey, lightOnlyBytes),
           "the route fixtures publish their exact bytes");
    auto lightOnlyOperation = starterOperation (lightOnlyKey);
    lightOnlyOperation.meshStableId = 0;

    fakeState = {};
    fixture::setPathTraceSceneEngineFactory (&makeFakeEngine);
    std::string error;

    {
        // The animated execution itself: it forwards the selection to its
        // fixture scene renderer.
        FakeDeformationBackend deformation;
        FakeBackend scene;
        ImportedAnimatedScenePayloadExecution execution (payloads, deformation, scene);
        check (execution.selectedEngine().engine == fixture::SceneEngineKind::Raster,
               "the animated execution starts on Raster");
        execution.selectEngine (pathTrace (48));
        check (execution.selectedEngine().engine == fixture::SceneEngineKind::PathTrace
                   && execution.selectedEngine().exportSamples == 48,
               "the animated execution keeps the Path Trace selection");

        ImportedAnimatedSceneRequest request;
        request.operation = starterOperation (starterKey);
        request.frame = { 0, 24, 1 };
        request.structuralRevision = 1;
        request.width = 96;
        request.height = 64;
        ImportedAnimatedSceneReceipt receipt;
        // The clip's morph weight is 0 at frame 0 and has risen by frame 12.
        const bool boundDrew = execution.executePreview (request, receipt, error);
        check (boundDrew && fakeState.renders == 1 && deformation.submissions == 0 && scene.submissions == 0,
               "a bound animated mesh renders on Path Trace: " + error);
        const auto first = fakeState.lastSnapshot;
        auto later = request;
        later.frame = { 12, 24, 1 };
        later.operation.playback.timelineSeconds = 0.5;
        check (execution.executePreview (later, receipt, error) && fakeState.renders == 2 && first != nullptr
                   && fakeState.lastSnapshot != nullptr
                   && (fakeState.lastSnapshot->vertices[0].position.x != first->vertices[0].position.x
                       || fakeState.lastSnapshot->vertices[0].position.y != first->vertices[0].position.y
                       || fakeState.lastSnapshot->vertices[0].position.z != first->vertices[0].position.z
                       || fakeState.lastSnapshot->vertices[1].position.y != first->vertices[1].position.y
                       || fakeState.lastSnapshot->vertices[2].position.y != first->vertices[2].position.y),
               "the path tracer gets this frame's deformed vertices: " + error);
        const auto laterSnapshot = fakeState.lastSnapshot;

        {
            // A material bound to the mesh was admitted on the source scene, and the frame draws its baked copy.
            const auto material = surfaceMaterialRequest (first->id, first->objects[0].id, first->objects[0].material.value);
            auto withMaterial = later;
            withMaterial.material = material;
            const auto rendersBefore = fakeState.renders;
            check (execution.executePreview (withMaterial, receipt, error) && fakeState.renders == rendersBefore + 1
                       && deformation.submissions == 0 && fakeState.lastSnapshot != nullptr
                       && fakeState.lastSnapshot->vertices[0].position.x == laterSnapshot->vertices[0].position.x,
                   "a bound mesh with a Surface material renders baked on Path Trace: " + error);
        }

        auto lightOnly = request;
        lightOnly.operation = lightOnlyOperation;
        const bool lightOnlyDrew = execution.executePreview (lightOnly, receipt, error);
        check (lightOnlyDrew && receipt.valid()
                   && fakeState.renders == 4 && fakeState.lastUse == fixture::RenderUse::Preview
                   && deformation.submissions == 0 && scene.submissions == 0,
               "camera/light animation draws with the path tracer: " + error);
        check (fakeState.lastHadAnimatedLights, "the path tracer receives the sampled imported lights");
        const bool lightOnlyExported = execution.executeExport (lightOnly, receipt, error);
        check (lightOnlyExported && fakeState.renders == 5
                   && fakeState.lastUse == fixture::RenderUse::Export && fakeState.lastSamples == 48
                   && scene.submissions == 0,
               "an animated Path Trace export uses the selected sample count: " + error);

        {
            // An unanimated morph mesh renders at its authored rest weights.
            const auto restBytes = lightOnlyStarter (1.0f);
            const auto restKey = exactKey ("pathtrace-route-rest-weight", restBytes);
            check (publish (payloads, "pathtrace-route-rest-weight-transfer", restKey, restBytes),
                   "the rest weight fixture publishes its exact bytes");
            auto unmoved = lightOnly;
            check (execution.executePreview (unmoved, receipt, error) && fakeState.lastSnapshot != nullptr,
                   "the weight 0 morph mesh draws with the path tracer");
            const auto base = fakeState.lastSnapshot;
            auto rested = lightOnly;
            rested.operation = starterOperation (restKey);
            rested.operation.meshStableId = 0;
            const auto restDrawn = execution.executePreview (rested, receipt, error);
            check (restDrawn && deformation.submissions == 0, "an unanimated morph mesh renders on Path Trace");
            const auto& starter = visualstartermodel::kAnimatedTriangleGlb;
            // The starter's morph target: accessor 4, bufferView 5, three VEC3s at byte 372 of the BIN.
            const auto* deltaBytes = starter.data() + 20u + readU32 (starter.data() + 12u) + 8u + 372u;
            bool moved = restDrawn && fakeState.lastSnapshot != nullptr
                && fakeState.lastSnapshot->vertexCount == base->vertexCount;
            for (std::size_t vertex = 0; moved && vertex < 3u; ++vertex)
            {
                std::array<float, 3> delta {};
                std::memcpy (delta.data(), deltaBytes + vertex * 12u, 12u);
                const auto& at = fakeState.lastSnapshot->vertices[vertex].position;
                const auto& from = base->vertices[vertex].position;
                moved = std::abs (at.x - from.x - delta[0]) < 1.0e-5f && std::abs (at.y - from.y - delta[1]) < 1.0e-5f
                    && std::abs (at.z - from.z - delta[2]) < 1.0e-5f;
            }
            check (moved, "the path tracer gets the vertices at weight 1");
        }

        {
            // Two animated leaves: each leaf's vertices for this frame reach the path tracer.
            const auto secondKey = exactKey ("pathtrace-route-composed-second", starterBytes);
            check (publish (payloads, "pathtrace-route-composed-second-transfer", secondKey, starterBytes),
                   "the second leaf fixture publishes its exact bytes");
            auto procedural = std::make_shared<HarmonicMIDI::grid::Visual3DScene> (*first);
            const auto shift = [] (auto& id) { if (id.isValid()) id.value += 1000u; };
            for (std::size_t i = 0; i < procedural->textureCount; ++i) shift (procedural->textures[i].id);
            for (std::size_t i = 0; i < procedural->materialCount; ++i)
            {
                auto& material = procedural->materials[i];
                shift (material.id); shift (material.baseColorTexture); shift (material.metallicRoughnessTexture);
                shift (material.normalTexture); shift (material.occlusionTexture); shift (material.emissiveTexture);
            }
            for (std::size_t i = 0; i < procedural->objectCount; ++i) shift (procedural->objects[i].material);
            for (std::size_t i = 0; i < procedural->lightCount; ++i) shift (procedural->lights[i].id);
            for (std::size_t i = 0; i < procedural->cameraCount; ++i) shift (procedural->cameras[i].id);
            shift (procedural->activeCamera);

            auto composed = later;
            composed.evaluationRevision = 1;
            composed.renderClipId = 88;
            composed.projectGeneration = composed.helperGeneration = 1;
            composed.proceduralScene = procedural;
            composed.compositionLeaves.resize (2);
            composed.compositionLeaves[0].operation = later.operation;
            composed.compositionLeaves[1].operation = later.operation;
            composed.compositionLeaves[1].operation.asset = secondKey;
            composed.compositionLeaves[1].operation.sourceStableId += 100;
            composed.compositionLeaves[1].operation.deformationStableId += 100;
            composed.compositionLeaves[1].operation.schedule = { composed.compositionLeaves[1].operation.sourceStableId,
                                                                 composed.compositionLeaves[1].operation.deformationStableId };
            const auto rendersBefore = fakeState.renders;
            const bool composedDrew = execution.executePreview (composed, receipt, error);
            check (composedDrew && fakeState.renders == rendersBefore + 1 && deformation.submissions == 0 && receipt.valid(),
                   "a multi-leaf Scene Compose renders on Path Trace: " + error);
            std::size_t matching = 0;
            if (composedDrew && fakeState.lastSnapshot != nullptr)
                for (std::size_t vertex = 0; vertex < fakeState.lastSnapshot->vertexCount; ++vertex)
                {
                    const auto& at = fakeState.lastSnapshot->vertices[vertex].position;
                    const auto& moved = laterSnapshot->vertices[0].position;
                    matching += at.x == moved.x && at.y == moved.y && at.z == moved.z ? 1u : 0u;
                }
            check (matching == 2u, "both leaves reach the path tracer at this frame's vertices, "
                       + std::to_string (matching) + " matches");
        }

        execution.selectEngine (raster());
        const bool rasterDrew = execution.executePreview (request, receipt, error);
        check (rasterDrew && receipt.valid()
                   && deformation.submissions == 1 && fakeState.renders == 8,
               "Raster draws the bound mesh with the deformation renderer: " + error + ", renders "
                   + std::to_string (fakeState.renders) + ", submissions " + std::to_string (deformation.submissions));
    }

    {
        // The outer route that used to fall back: a Deformation input sends
        // the request through the animated execution.
        FakeDeformationBackend deformation;
        FakeBackend scene;
        ImportedScenePayloadExecution execution (payloads, scene, deformation);
        ImportedSceneRequest request;
        request.asset = starterKey;
        request.sceneIndex = 0;
        request.dimensions = { 96, 64 };
        request.frame = { 12, 24, 1 };
        request.structuralRevision = 1;
        request.evaluationRevision = 1;
        request.deformation = starterOperation (starterKey);
        request.deformation->playback.timelineSeconds = 0.5;
        request.renderEngine = render::RenderEngine::PathTrace;
        request.pathTraceExportSamples = 32;
        ImportedSceneExecutionReceipt receipt;
        check (execution.executePreview (request, receipt, error) && deformation.submissions == 0 && scene.submissions == 0,
               "a deformed imported scene renders on Path Trace: " + error);

        auto lightOnly = request;
        lightOnly.asset = lightOnlyKey;
        lightOnly.deformation = lightOnlyOperation;
        lightOnly.deformation->playback.timelineSeconds = 0.5;
        const int before = fakeState.renders;
        const bool sceneExported = execution.executeExport (lightOnly, receipt, error);
        check (sceneExported && receipt.deformationFrame
                   && fakeState.renders == before + 1 && fakeState.lastUse == fixture::RenderUse::Export
                   && fakeState.lastSamples == 32 && deformation.submissions == 0 && scene.submissions == 0,
               "an animated imported scene carries the node's engine and samples to the path tracer: " + error);

        auto composed = request;
        composed.additionalImportedScenes.push_back ({ std::make_shared<const render::Request>(), std::nullopt });
        const auto rendersBeforeMalformed = fakeState.renders;
        check (! execution.executePreview (composed, receipt, error)
                   && error == "Scene Compose imported request or explicit lighting replacement is invalid"
                   && fakeState.renders == rendersBeforeMalformed && deformation.submissions == 0 && scene.submissions == 0,
               "a malformed imported leaf refuses before any owner changes: " + error);

        auto rasterRequest = request;
        rasterRequest.renderEngine = render::RenderEngine::Raster;
        rasterRequest.pathTraceExportSamples = render::kDefaultPathTraceExportSamples;
        check (execution.executePreview (rasterRequest, receipt, error) && receipt.deformationFrame
                   && deformation.submissions == 1,
               "the same execution draws Raster again when the node selects it");
    }

    {
        // Two static leaves over a procedural scene take the static route on either engine.
        const auto secondBytes = lightOnlyStarter (1.0f);
        const auto secondKey = exactKey ("pathtrace-route-static-leaf", secondBytes);
        check (publish (payloads, "pathtrace-route-static-leaf-transfer", secondKey, secondBytes),
               "the second static leaf publishes its exact bytes");
        auto procedural = std::make_shared<HarmonicMIDI::grid::Visual3DScene> (videohelper::fixture3d::makeScene());
        render::Request first;
        first.sourceStableId = 71; first.renderStableId = 72;
        first.asset = starterKey; first.sceneIndex = 0;
        first.structuralRevision = first.evaluationRevision = 1;
        auto second = first;
        second.sourceStableId = 81; second.asset = secondKey;

        FakeDeformationBackend deformation;
        FakeBackend scene;
        ImportedScenePayloadExecution execution (payloads, scene, deformation);
        ImportedSceneRequest request;
        request.asset = starterKey;
        request.sceneIndex = 0;
        request.dimensions = { 96, 64 };
        request.frame = { 12, 24, 1 };
        request.clipId = 903;
        request.structuralRevision = request.evaluationRevision = 1;
        request.staticPayloadIdentity = "pathtrace-route-static-leaves";
        request.sceneSnapshot = procedural;
        request.importedScene = std::make_shared<const render::Request> (first);
        request.additionalImportedScenes.push_back ({ std::make_shared<const render::Request> (second), std::nullopt });
        request.renderEngine = render::RenderEngine::PathTrace;
        request.pathTraceExportSamples = 16;

        ImportedSceneExecutionReceipt receipt;
        const auto rendersBefore = fakeState.renders;
        const bool traced = execution.executePreview (request, receipt, error);
        if (! traced) std::fprintf (stderr, "two static leaves refused: %s\n", error.c_str());
        check (traced && receipt.valid() && fakeState.renders == rendersBefore + 1
                   && scene.submissions == 0 && deformation.submissions == 0,
               "two static leaves draw on Path Trace");
        const auto drawn = fakeState.lastSnapshot;
        namespace composition = HarmonicMIDI::grid::scenecomposition;
        const auto leafObject = [&] (std::size_t leaf) -> const HarmonicMIDI::grid::SceneObjectRecord* {
            for (std::size_t i = 0; drawn != nullptr && i < drawn->objectCount; ++i)
                if (composition::importedLeafOwnsObjectId (drawn->objects[i].id.value, leaf, 2)) return &drawn->objects[i];
            return nullptr;
        };
        check (leafObject (0) != nullptr, "the drawn scene holds leaf 0's object");
        check (leafObject (1) != nullptr, "the drawn scene holds leaf 1's object");

        auto again = request;
        const auto ledger = execution.staticLedgerWork();
        check (execution.executePreview (again, receipt, error) && fakeState.lastSnapshot == drawn
                   && execution.staticLedgerWork() == ledger,
               "the next frame reuses the joined scene without a second admission");

        auto raster = request;
        raster.renderEngine = render::RenderEngine::Raster;
        raster.pathTraceExportSamples = render::kDefaultPathTraceExportSamples;
        const auto rasterRendersBefore = fakeState.renders;
        check (execution.executePreview (raster, receipt, error) && receipt.valid()
                   && scene.submissions == 1 && fakeState.renders == rasterRendersBefore,
               "two static leaves draw on Raster");

        auto selecting = request;
        auto selectingSecond = second;
        selectingSecond.objectNodeStableId = *composition::encodeImportedLeafId (1, 2, 1);
        selectingSecond.objectMeshStableId = 1;
        selecting.additionalImportedScenes[0].scene = std::make_shared<const render::Request> (selectingSecond);
        const bool selectingRefused = ! execution.executePreview (selecting, receipt, error);
        if (selectingRefused) std::fprintf (stderr, "selecting leaf: %s\n", error.c_str());
        check (selectingRefused && error.find ("static imported leaves") != std::string::npos,
               "a selected object with only static leaves stays a named refusal");

        if (const auto* target = leafObject (1))
        {
            // A material on leaf 1 binds in its leaf range. A wrong MaterialId refuses.
            auto bound = second;
            bound.material = surfaceMaterialRequest (drawn->id, target->id, target->material.value);
            auto withMaterial = request;
            withMaterial.additionalImportedScenes[0].scene = std::make_shared<const render::Request> (bound);
            const bool materialDrew = execution.executePreview (withMaterial, receipt, error);
            if (! materialDrew) std::fprintf (stderr, "leaf 1 material: %s\n", error.c_str());
            check (materialDrew && receipt.valid() && ! receipt.usedLastGoodMaterial,
                   "a Render Material on static leaf 1 joins the composed Surface collection");

            auto wrongId = second;
            wrongId.material = surfaceMaterialRequest (drawn->id, target->id, target->material.value + 1u);
            auto withWrongId = request;
            withWrongId.additionalImportedScenes[0].scene = std::make_shared<const render::Request> (wrongId);
            const bool wrongRefused = ! execution.executePreview (withWrongId, receipt, error);
            if (wrongRefused) std::fprintf (stderr, "wrong leaf 1 material id: %s\n", error.c_str());
            check (wrongRefused && error.find ("Render Material ID does not match") != std::string::npos,
                   "a static leaf 1 material with another object's ID refuses");
        }
    }
}

} // namespace

// holographic_card_pathtrace_tests.cpp: the card's committed request in the real engine.
int holographicCardPathTraceTests();

int main()
{
    glassMappingTests();
    thinSurfaceCoverageRenderTests();
    inputRefusalTests();
    rendererEngineTests();
    pathTraceStatsReplyTests();
    filmProvenanceTests();
    frameLayerDrawTests();
    denoiseKernelTests();
    denoiseEngineTests();
    viewportConvergenceTests();
    causticDrawTests();
    importedRouteTests();
    const int cardFailures = holographicCardPathTraceTests();
    std::printf ("%d checks, %d failures\n", checks, failures);
    return failures == 0 && cardFailures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
