// Route tests for the Path Trace render engine (Plan 3, PT-C1): the glass
// material mapping and input refusals of the fixture engine adapter, the
// FixtureSceneRenderer engine branch, and the imported scene routes that go
// through the animated execution. A Path Trace request either draws with the
// path tracer or fails with a named refusal. It never draws Raster.

#include "../src/imported_scene_payload_execution.h"
#include "../src/imported_animated_scene_payload_execution.h"
#include "../src/pathtrace_fixture_engine.h"
#include "../src/pathtrace/integrator.h"
#include "../../shared/VisualStarterModelAssets.h"
#include "support/fixture_scene.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <memory>
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
    std::string lastBackend;
    bool lastHadAnimatedLights = false;
};

FakeEngineState fakeState;

class FakeEngine final : public fixture::SceneRenderEngine
{
public:
    bool render (const fixture::SceneEngineDraw& draw, fixture::RenderedFrame& output, std::string&) override
    {
        ++fakeState.renders;
        fakeState.lastUse = draw.use;
        fakeState.lastSamples = draw.exportSamples;
        fakeState.lastBackend = draw.backend;
        fakeState.lastHadAnimatedLights = draw.inputs != nullptr && ! draw.inputs->animatedLights.empty();
        output = {};
        output.nativeFrame = std::make_shared<FakeFrame> (draw.backend, draw.dimensions.width,
                                                          draw.dimensions.height,
                                                          static_cast<std::uintptr_t> (500 + fakeState.renders));
        return true;
    }

    void cancel() noexcept override { ++fakeState.cancels; }
};

std::shared_ptr<fixture::SceneRenderEngine> makeFakeEngine()
{
    ++fakeState.factoryCalls;
    return std::make_shared<FakeEngine>();
}

std::shared_ptr<fixture::SceneRenderEngine> makeNoEngine() { return nullptr; }

fixture::SceneEngineSelection pathTrace (std::uint32_t samples = 64)
{
    return { fixture::SceneEngineKind::PathTrace, samples };
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
    check (! ptfixture::mapSurfaceProgram (frameTexture, nullptr, 0.0f, material, error)
               && error == "Path Trace does not sample frame textures in materials yet; choose Raster",
           "a frame-texture base colour is refused by name");
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
    constexpr const char* passes = "Path Trace does not render diffraction or glass caustic passes yet; choose Raster";
    constexpr const char* composites = "Path Trace does not run render pass composites yet; choose Raster";

    arbitgpu::NativeFixtureSceneRuntimeInputs plain;
    std::string error;
    check (ptfixture::admitInputs (plain, error), "plain Color inputs are admitted");

    auto caustic = plain;
    caustic.glassCaustic = render::GlassCausticSelection {};
    check (refused (caustic, passes), "a glass caustic selection is refused by name");
    auto source = plain;
    source.diffractionSourceProgram = std::make_shared<const Program>();
    check (refused (source, passes), "a diffraction source program is refused by name");
    auto evaluated = plain;
    evaluated.diffractionEvaluatedProgram = std::make_shared<const Program>();
    check (refused (evaluated, passes), "an evaluated diffraction program is refused by name");

    auto composite = plain;
    composite.passComposite = renderpasscomposite::Parameters {};
    check (refused (composite, composites), "a render pass composite is refused by name");
    auto program = plain;
    program.passProgram = renderpasscomposite::Program {};
    check (refused (program, composites), "a render pass program is refused by name");

    auto depth = plain;
    depth.imageOutput = renderpassoutput::Output::Depth;
    check (refused (depth, "Path Trace renders the Color image only; choose Raster for other outputs and data exports"),
           "a non-Color image output is refused by name");
    auto frame = plain;
    frame.materialFrameTexture = std::make_shared<FakeFrame> ("opengl", 4, 4, 9);
    check (refused (frame, "Path Trace does not sample frame textures in materials yet; choose Raster"),
           "a material frame texture is refused by name");
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
        renderer.selectEngine (pathTrace (200));
        check (renderer.selectedEngine().engine == fixture::SceneEngineKind::PathTrace
                   && renderer.selectedEngine().exportSamples == 200,
               "the selection keeps the engine and export samples");
        RenderedFrame preview;
        check (renderer.renderPreview (scene, { 32, 24 }, kNativeGpuCapability, preview, error)
                   && fakeState.renders == 1 && fakeState.lastUse == fixture::RenderUse::Preview
                   && fakeState.lastBackend == "opengl" && backend.submissions == 0,
               "a Path Trace preview draws with the engine, not the raster backend");
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
        RenderedFrame refused;
        check (! renderer.renderPreview (scene, {}, {}, caustic, { 32, 24 }, kNativeGpuCapability, refused, error)
                   && error == "Path Trace does not render diffraction or glass caustic passes yet; choose Raster"
                   && fakeState.renders == 2 && backend.submissions == 0,
               "a glass caustic draw under Path Trace is refused before either engine runs");

        renderer.selectEngine (raster());
        check (fakeState.cancels == 1, "selecting Raster cancels the path tracer");
        check (renderer.renderPreview (scene, { 32, 24 }, kNativeGpuCapability, preview, error)
                   && backend.submissions == 1 && fakeState.renders == 2,
               "a Raster selection draws with the raster backend only");
        renderer.selectEngine (pathTrace());
        check (renderer.renderPreview (scene, { 32, 24 }, kNativeGpuCapability, preview, error)
                   && fakeState.factoryCalls == 2 && fakeState.renders == 3 && backend.submissions == 1,
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
// on the light node: camera/light animation only, so no mesh is bound and the
// fixture scene renderer draws it (glb_native_render_seam_tests.cpp keeps the
// same fixture).
std::vector<std::uint8_t> lightOnlyStarter()
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
    const auto sampler = animation["samplers"].size();
    animation["samplers"].push_back ({ { "input", input }, { "output", output }, { "interpolation", "LINEAR" } });
    animation["channels"].push_back ({ { "sampler", sampler }, { "target", { { "node", 1 }, { "path", "translation" } } } });
    animation["channels"].erase (animation["channels"].begin());
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
        check (! execution.executePreview (request, receipt, error)
                   && error == std::string (render::kPathTraceAnimatedMeshUnavailable)
                   && deformation.submissions == 0 && scene.submissions == 0 && fakeState.renders == 0,
               "a bound animated mesh refuses Path Trace by name instead of drawing Raster");

        auto lightOnly = request;
        lightOnly.operation = lightOnlyOperation;
        check (execution.executePreview (lightOnly, receipt, error) && receipt.valid()
                   && fakeState.renders == 1 && fakeState.lastUse == fixture::RenderUse::Preview
                   && deformation.submissions == 0 && scene.submissions == 0,
               "camera/light animation draws with the path tracer");
        check (fakeState.lastHadAnimatedLights, "the path tracer receives the sampled imported lights");
        check (execution.executeExport (lightOnly, receipt, error) && fakeState.renders == 2
                   && fakeState.lastUse == fixture::RenderUse::Export && fakeState.lastSamples == 48
                   && scene.submissions == 0,
               "an animated Path Trace export uses the selected sample count");

        auto composed = request;
        composed.compositionLeaves.resize (2);
        check (! execution.executePreview (composed, receipt, error)
                   && error == std::string (render::kPathTraceMultipleImportedLeavesUnavailable)
                   && deformation.submissions == 0,
               "a multi-leaf Scene Compose refuses Path Trace by name");

        execution.selectEngine (raster());
        check (execution.executePreview (request, receipt, error) && receipt.valid()
                   && deformation.submissions == 1 && fakeState.renders == 2,
               "Raster draws the bound mesh with the deformation renderer");
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
        check (! execution.executePreview (request, receipt, error)
                   && error == std::string (render::kPathTraceAnimatedMeshUnavailable)
                   && deformation.submissions == 0 && scene.submissions == 0,
               "a deformed imported scene refuses Path Trace by name instead of drawing Raster");

        auto lightOnly = request;
        lightOnly.asset = lightOnlyKey;
        lightOnly.deformation = lightOnlyOperation;
        lightOnly.deformation->playback.timelineSeconds = 0.5;
        const int before = fakeState.renders;
        check (execution.executeExport (lightOnly, receipt, error) && receipt.deformationFrame
                   && fakeState.renders == before + 1 && fakeState.lastUse == fixture::RenderUse::Export
                   && fakeState.lastSamples == 32 && deformation.submissions == 0 && scene.submissions == 0,
               "an animated imported scene carries the node's engine and samples to the path tracer");

        auto composed = request;
        composed.additionalImportedScenes.push_back ({ std::make_shared<const render::Request>(), std::nullopt });
        check (! execution.executePreview (composed, receipt, error)
                   && error == std::string (render::kPathTraceMultipleImportedLeavesUnavailable),
               "an imported Scene Compose refuses Path Trace before any owner changes");

        auto rasterRequest = request;
        rasterRequest.renderEngine = render::RenderEngine::Raster;
        rasterRequest.pathTraceExportSamples = render::kDefaultPathTraceExportSamples;
        check (execution.executePreview (rasterRequest, receipt, error) && receipt.deformationFrame
                   && deformation.submissions == 1,
               "the same execution draws Raster again when the node selects it");
    }
}

} // namespace

int main()
{
    glassMappingTests();
    thinSurfaceCoverageRenderTests();
    inputRefusalTests();
    rendererEngineTests();
    importedRouteTests();
    std::printf ("%d checks, %d failures\n", checks, failures);
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
