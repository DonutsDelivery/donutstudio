// Renders Realtime frames through the native Metal fixture backend at the live probe's
// 960x540 and writes each Color as a binary PPM, for side-by-side judging against the
// Linux Realtime frames. Built by hand on the MacBook from the same translation units as
// arbit-metal-fixture-diffraction-tests, with this file in place of its main.
//
//   big    a 1025x1025 vertex ocean grid (over a million vertices) under a lighthouse
//          box: the size the Lighthouse starter's scene has. The fixed-array guard
//          refused it.
//   orb    the Diffraction Orb starter's live view (the Linux receiver test's liveorb),
//          whose Color raw pass readback failed.
//   disc   the orb's view with a flat grooved annulus of 4096 triangles in place of the
//          sphere, for the Grooved Disc starter's tiled frame.
//   card   the Holographic Trading Card starter's GLB and diffraction binding through
//          FixtureSceneRenderer, as the app's export probe draws it.
//
// Usage: metal-realtime-starter-harness <outdir> [case...]. Prints one line per case with
// its result, wall time and a coarse checksum of the image.

#include "gpu_backend/backend.h"
#include "gpu_backend/fixture_realtime_opengl.h"
#include "fixture_scene_renderer.h"
#include "support/diffraction_native_transport_fixture.h"
#include "support/fixture_scene.h"

#import <Foundation/Foundation.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace
{
using namespace HarmonicMIDI::grid;
namespace tt = triangletransporttests;
std::uint32_t kWidth = 960, kHeight = 540;
bool gRealtime = true;

std::vector<float> linearPixels (const arbitgpu::NativeRawPassPixels& pixels)
{
    std::vector<float> result;
    if (pixels.format == arbitgpu::NativeTexturePixelFormat::Rgba16Float)
    {
        result.resize (pixels.bytes.size() / 2);
        for (std::size_t i = 0; i < result.size(); ++i)
        {
            std::uint16_t half = 0;
            std::memcpy (&half, pixels.bytes.data() + 2 * i, 2);
            const auto exponent = (half >> 10) & 31u, fraction = half & 1023u;
            const float value = exponent == 31 ? std::numeric_limits<float>::quiet_NaN()
                : exponent == 0 ? std::ldexp (float (fraction), -24) : std::ldexp (float (1024u + fraction), int (exponent) - 25);
            result[i] = (half & 32768u) ? -value : value;
        }
    }
    else if (pixels.format == arbitgpu::NativeTexturePixelFormat::Rgba32Float)
    {
        result.resize (pixels.bytes.size() / 4);
        std::memcpy (result.data(), pixels.bytes.data(), pixels.bytes.size());
    }
    else if (pixels.format == arbitgpu::NativeTexturePixelFormat::Rgba8Unorm)
    {
        result.resize (pixels.bytes.size());
        for (std::size_t i = 0; i < result.size(); ++i)
        {
            const float c = pixels.bytes[i] / 255.0f;
            result[i] = (i % 4 == 3) ? c : (c <= 0.04045f ? c / 12.92f : std::pow ((c + 0.055f) / 1.055f, 2.4f));
        }
    }
    return result;
}

// Scene-linear RGBA to an sRGB PPM, clamped, as the composite shows it.
bool writePpm (const std::string& path, const std::vector<float>& rgba, std::uint32_t width, std::uint32_t height,
               double& mean, std::size_t& nonfinite)
{
    std::FILE* file = std::fopen (path.c_str(), "wb");
    if (file == nullptr) return false;
    std::fprintf (file, "P6\n%u %u\n255\n", width, height);
    std::vector<unsigned char> row (std::size_t (width) * 3);
    mean = 0.0;
    nonfinite = 0;
    for (std::uint32_t y = 0; y < height; ++y)
    {
        for (std::uint32_t x = 0; x < width; ++x)
            for (unsigned c = 0; c < 3; ++c)
            {
                float v = rgba[(std::size_t (y) * width + x) * 4 + c];
                if (! std::isfinite (v)) { ++nonfinite; v = 0.0f; }
                mean += v;
                v = std::clamp (v, 0.0f, 1.0f);
                v = v <= 0.0031308f ? v * 12.92f : 1.055f * std::pow (v, 1.0f / 2.4f) - 0.055f;
                row[x * 3 + c] = static_cast<unsigned char> (std::lround (v * 255.0f));
            }
        std::fwrite (row.data(), 1, row.size(), file);
    }
    mean /= double (width) * height * 3;
    return std::fclose (file) == 0;
}

bool readColor (const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& frame, std::vector<float>& pixels,
                std::string& error)
{
    arbitgpu::NativeRawPassPixels raw;
    if (frame == nullptr) { error = "no frame"; return false; }
    if (! frame->readRawPass (renderpassoutput::Output::Color, raw, error)) return false;
    pixels = linearPixels (raw);
    if (pixels.size() != std::size_t (kWidth) * kHeight * 4) { error = "unexpected Color format or extent"; return false; }
    return true;
}

// The ocean: a grid of 1025 x 1025 vertices (1,050,625) with low swell, a grey box for the
// lighthouse, a sun and an analytic sky, seen from a camera standing above the water.
std::shared_ptr<Visual3DScene> bigScene()
{
    auto scene = std::make_shared<Visual3DScene> (videohelper::fixture3d::makeScene());
    constexpr std::uint32_t n = 1025;
    Visual3DSceneGrowth growth;
    growth.vertices = std::size_t (n) * n;
    growth.indices = std::size_t (n - 1) * (n - 1) * 6;
    growth.objects = 1;
    growth.materials = 1;
    growth.lights = 1;
    std::string error;
    if (! growVisual3DScene (*scene, growth, error)) { std::printf ("big: grow failed: %s\n", error.c_str()); return nullptr; }
    // The cube becomes the lighthouse's tower.
    scene->objects[0].transform.rotation = { 0, 0, 0, 1 };
    scene->objects[0].transform.translation = { 0.0f, 1.5f, -6.0f };
    scene->objects[0].transform.scale = { 0.6f, 2.5f, 0.6f };
    scene->materials[0].baseColorTexture = {};
    scene->materials[0].baseColor = { 0.85f, 0.85f, 0.85f };
    const auto firstVertex = scene->vertexCount, firstIndex = scene->indexCount;
    for (std::uint32_t j = 0; j < n; ++j)
        for (std::uint32_t i = 0; i < n; ++i)
        {
            const float x = -40.0f + 80.0f * float (i) / float (n - 1), z = -40.0f + 80.0f * float (j) / float (n - 1);
            const float h = 0.15f * std::sin (0.9f * x + 0.3f * z) + 0.08f * std::sin (1.7f * z - 0.6f * x);
            const float dx = 0.15f * 0.9f * std::cos (0.9f * x + 0.3f * z) - 0.08f * 0.6f * std::cos (1.7f * z - 0.6f * x);
            const float dz = 0.15f * 0.3f * std::cos (0.9f * x + 0.3f * z) + 0.08f * 1.7f * std::cos (1.7f * z - 0.6f * x);
            const float length = std::sqrt (dx * dx + 1.0f + dz * dz);
            auto& v = scene->vertices[scene->vertexCount++];
            v.position = { x, h - 0.8f, z };
            v.normal = { -dx / length, 1.0f / length, -dz / length };
            v.uv = { float (i) / float (n - 1), float (j) / float (n - 1) };
            v.color = { 1, 1, 1, 1 };
        }
    for (std::uint32_t j = 0; j + 1 < n; ++j)
        for (std::uint32_t i = 0; i + 1 < n; ++i)
        {
            const std::uint32_t a = j * n + i, b = a + 1, c = a + n, d = c + 1;
            for (const std::uint32_t index : { a, c, b, b, c, d }) scene->indices[scene->indexCount++] = index;
        }
    auto& material = scene->materials[scene->materialCount++];
    material.id = SceneMaterialId { 202 };
    material.baseColor = { 0.05f, 0.18f, 0.28f };
    material.roughness = 0.12f;
    material.metallic = 0.0f;
    auto& ocean = scene->objects[scene->objectCount++];
    ocean.id = SceneObjectId { 102 };
    ocean.material = material.id;
    ocean.firstVertex = static_cast<std::uint32_t> (firstVertex);
    ocean.vertexCount = static_cast<std::uint32_t> (scene->vertexCount - firstVertex);
    ocean.firstIndex = static_cast<std::uint32_t> (firstIndex);
    ocean.indexCount = static_cast<std::uint32_t> (scene->indexCount - firstIndex);
    auto& sky = scene->lights[scene->lightCount++];
    sky.id = SceneLightId { 302 };
    sky.kind = SceneLightKind::Environment;
    sky.sky.version = 1;
    sky.sky.zenith = { 0.25f, 0.45f, 0.85f };
    sky.sky.horizon = { 0.75f, 0.8f, 0.9f };
    sky.sky.sunIntensity = 0.0f;
    auto& camera = scene->cameras[0];
    camera.transform.translation = { 0.0f, 1.2f, 4.0f };
    camera.transform.rotation = { -0.04f, 0.0f, 0.0f, 0.9992f };
    camera.verticalFovRadians = 0.9f;
    camera.farPlane = 200.0f;
    return scene;
}

// --- The Diffraction Orb's live view, from realtime_receiver_transport_gl_tests.cpp ---
diffractionmaterial::Description orbGrating()
{
    auto d = diffractionmaterial::makeAluminiumBinaryGratingPreset();
    d.geometry.directionUv = { 0.0f, 1.0f };
    d.geometry.grooveSpacingNanometres = 1400.0f;
    d.geometry.lattice = diffractionmaterial::GratingLattice::CrossedTwoDimensional;
    d.geometry.secondaryDirectionUv = { -0.8660254f, 0.5f };
    d.geometry.secondaryGrooveSpacingNanometres = 1400.0f;
    d.geometry.thirdGrooveFamily = true;
    d.microstructure.profile = diffractionmaterial::GrooveProfile::Sinusoidal;
    d.microstructure.grooveDepthNanometres = 90.0f;
    d.spectrum.lastOrder = 4;
    d.roughness.rmsSlope = 0.02f;
    return d;
}

// A compact disc's 1D grating, its grooves running round the disc (uv v is the radius).
diffractionmaterial::Description discGrating()
{
    auto d = diffractionmaterial::makeAluminiumBinaryGratingPreset();
    d.geometry.directionUv = { 0.0f, 1.0f };
    d.geometry.grooveSpacingNanometres = 1600.0f;
    d.spectrum.lastOrder = 3;
    d.roughness.rmsSlope = 0.02f;
    return d;
}

// disc: a flat grooved annulus of many triangles filling the frame, in place of the
// orb's sphere, standing in for the Grooved Disc starter's film ring.
tt::Fixture orbFixture (std::string& error, bool disc = false)
{
    auto f = tt::nativeFixture (false, 0, 0);
    f.program.backend = arbitgpu::NativeFixtureMaterialBackend::Metal;
    {
        auto receiver = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram> (*f.program.objectPrograms[0]);
        receiver->backend = f.program.backend;
        f.program.objectPrograms[0] = receiver;
    }
    f.program.diffractionPaths[0].material.roughness.y = .15f;
    f.scene->activeCamera = { 301 };
    f.scene->cameraCount = 1;
    auto& camera0 = f.scene->cameras[0];
    camera0.id = f.scene->activeCamera;
    camera0.verticalFovRadians = 1.4f;
    camera0.nearPlane = .01f;
    camera0.farPlane = 30;
    f.scene->ambientColor = { 0, 0, 0 };
    for (std::size_t o = 0; o < f.scene->objectCount; ++o)
    {
        const auto& object = f.scene->objects[o];
        f.scene->materials[o].doubleSided = true;
        for (std::size_t v = 0; v < object.vertexCount; ++v)
            f.scene->vertices[object.firstVertex + v].normal = { 0, 0, o == 1 ? -1.0f : 1.0f };
    }
    auto& scene = *f.scene;
    {
        const auto& plate = scene.objects[0];
        for (std::size_t v = 0; v < plate.vertexCount; ++v)
        {
            auto& position = scene.vertices[plate.firstVertex + v].position;
            position.x *= 4.0f * 0.25f;
            position.y *= 4.0f * 0.3125f;
        }
    }
    const unsigned segments = disc ? 256 : 64, rings = disc ? 8 : 32;
    constexpr unsigned panels = 4;
    constexpr float radius = 0.25f, pi = 3.14159265358979f;
    const std::array<float, 3> center { 0.0f, 0.0f, 0.25f };
    Visual3DSceneGrowth growth;
    growth.vertices = std::size_t (segments + 1) * (rings + 1) + 4 * panels + 8;
    growth.indices = std::size_t (segments) * (disc ? rings : rings - 1) * 6 + 6 * panels + 12;
    growth.objects = panels + 2;
    growth.materials = panels + 2;
    if (! growVisual3DScene (scene, growth, error)) return f;
    auto& object = scene.objects[scene.objectCount - 1];
    scene.vertexCount = object.firstVertex;
    scene.indexCount = object.firstIndex;
    for (unsigned ring = 0; ring <= rings; ++ring)
        for (unsigned segment = 0; segment <= segments; ++segment)
        {
            const float polar = pi * float (ring) / float (rings), azimuth = 2 * pi * float (segment) / float (segments);
            const float x = std::sin (polar) * std::cos (azimuth), y = std::cos (polar), z = std::sin (polar) * std::sin (azimuth);
            auto& v = scene.vertices[scene.vertexCount++];
            v.position = { center[0] + radius * x, center[1] + radius * y, center[2] + radius * z };
            v.color = { 1, 1, 1, 1 };
            v.normal = { x, y, z };
            if (disc)
            {
                const float r = 0.06f + 0.24f * float (ring) / float (rings);
                v.position = { r * std::cos (azimuth), r * std::sin (azimuth), 0.02f };
                v.normal = { 0, 0, 1 };
            }
            v.uv = { float (segment) / float (segments), float (ring) / float (rings) };
        }
    const auto at = [&] (unsigned ring, unsigned segment) { return ring * (segments + 1) + segment; };
    const auto triangle = [&] (std::uint32_t a, std::uint32_t b, std::uint32_t c)
    {
        scene.indices[scene.indexCount++] = a;
        scene.indices[scene.indexCount++] = b;
        scene.indices[scene.indexCount++] = c;
    };
    for (unsigned ring = 0; ring < rings; ++ring)
        for (unsigned segment = 0; segment < segments; ++segment)
        {
            if (ring != 0 || disc) triangle (at (ring, segment), at (ring, segment + 1), at (ring + 1, segment));
            if (ring + 1 != rings || disc) triangle (at (ring, segment + 1), at (ring + 1, segment + 1), at (ring + 1, segment));
        }
    object.vertexCount = scene.vertexCount - object.firstVertex;
    object.indexCount = scene.indexCount - object.firstIndex;

    using D3 = std::array<double, 3>;
    struct Panel { D3 centre, u, v; SceneVec3 emission; };
    const std::array<Panel, panels> set { {
        { { -0.4, -2.25, 0.8 }, { 0.1125, 0, 0 }, { 0, 0, 0.1125 }, { 40, 40, 40 } },
        { { 0.0, 0.75, 1.5 }, { 0.125, 0, 0 }, { 0, 0.125, 0 }, { 20, 20, 20 } },
        { { -2.0, -1.125, 0.75 }, { 0, 0.75, 0 }, { 0, 0, 0.5 }, { 0.2f, 1.8f, 0.6f } },
        { { 2.0, -1.125, 0.75 }, { 0, 0.75, 0 }, { 0, 0, 0.5 }, { 1.2f, 0.2f, 1.8f } } } };
    const auto addQuad = [&] (std::array<D3, 4> corners, std::uint32_t id, SceneVec3 normal)
    {
        f.addObject (corners, id, id + 100);
        const auto& added = scene.objects[scene.objectCount - 1];
        for (std::size_t v = 0; v < added.vertexCount; ++v)
        {
            auto& vertex = scene.vertices[added.firstVertex + v];
            vertex.normal = normal;
            vertex.uv = { v == 1 || v == 2 ? 1.0f : 0.0f, v >= 2 ? 1.0f : 0.0f };
        }
    };
    for (std::size_t i = 0; i < set.size(); ++i)
    {
        const auto& panel = set[i];
        std::array<D3, 4> corners;
        for (unsigned k = 0; k < 4; ++k)
        {
            const double su = k == 1 || k == 2 ? 1 : -1, sv = k >= 2 ? 1 : -1;
            for (unsigned a = 0; a < 3; ++a) corners[k][a] = panel.centre[a] + su * panel.u[a] + sv * panel.v[a];
        }
        const D3 n { panel.u[1] * panel.v[2] - panel.u[2] * panel.v[1], panel.u[2] * panel.v[0] - panel.u[0] * panel.v[2],
                     panel.u[0] * panel.v[1] - panel.u[1] * panel.v[0] };
        const double length = std::sqrt (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        addQuad (corners, std::uint32_t (104 + i), { float (n[0] / length), float (n[1] / length), float (n[2] / length) });
        scene.materials[scene.materialCount - 1].emissive = panel.emission;
    }
    const auto addPlain = [&] (std::array<D3, 4> corners, std::uint32_t id, SceneVec3 colour, SceneVec3 normal)
    {
        addQuad (corners, id, normal);
        auto& material = scene.materials[scene.materialCount - 1];
        material.baseColor = colour;
        material.doubleSided = true;
    };
    addPlain ({ { { -10, -10, -0.01 }, { 10, -10, -0.01 }, { 10, 10, -0.01 }, { -10, 10, -0.01 } } }, 110,
              { 0.35f, 0.35f, 0.35f }, { 0, 0, 1 });
    addPlain ({ { { -10, 1.25, -2 }, { 10, 1.25, -2 }, { 10, 1.25, 8 }, { -10, 1.25, 8 } } }, 111,
              { 0.02f, 0.02f, 0.02f }, { 0, -1, 0 });
    auto& sky = scene.lights[scene.lightCount++];
    sky.id = SceneLightId { 401 };
    sky.kind = SceneLightKind::Environment;
    sky.sky.version = 1;
    sky.sky.zenith = { 0.03f, 0.03f, 0.04f };
    sky.sky.horizon = { 0.07f, 0.07f, 0.07f };
    sky.sky.sunIntensity = 0.0f;
    auto& camera = scene.cameras[0];
    camera.transform.translation = { 0.0f, -0.8f, 0.375f };
    const float half = 0.5f * (1.5707963f - 0.17453293f);
    camera.transform.rotation = SceneQuaternion { std::sin (half), 0, 0, std::cos (half) };
    camera.verticalFovRadians = 0.90757121f;
    if (disc)
    {
        // Looking down at 45 degrees from 0.85 away.
        const float tilt = 0.5f * 0.78539816f;
        camera.transform.translation = { 0.0f, -0.6f, 0.62f };
        camera.transform.rotation = SceneQuaternion { std::sin (tilt), 0, 0, std::cos (tilt) };
    }
    const double length = std::sqrt (0.36 * 0.36 + 0.93 * 0.93 + 0.07 * 0.07);
    f.input.source.direction = { -0.36 / length, 0.93 / length, 0.07 / length };
    f.input.source.radiance.fill (24.0);

    using namespace diffractionmaterial;
    LightingDescription lighting = f.program.diffractionLightingAdmission->description();
    for (unsigned a = 0; a < 3; ++a) lighting.paths[0].incident.direction[a] = float (f.input.source.direction[a]);
    for (unsigned b = 0; b < 8; ++b) lighting.paths[0].incident.radiance[b] = float (f.input.source.radiance[b]);
    const auto admitted = AdmittedLightingPlan::admit (lighting, error);
    const auto physical = admit (disc ? discGrating() : orbGrating(), error);
    if (! admitted || ! physical) return f;
    f.program.diffractionLightingAdmission = std::make_shared<const AdmittedLightingPlan> (*admitted);
    f.program.diffractionPaths[0] = makeGpuLightingPath (
        physicalcheckpoint::makeGpuParameters (*physical, lighting.paths[0].incident), lighting.paths[0]);
    f.program.diffractionPaths[0].material.incident = { 0, 0, 1, 0 };
    f.program.diffractionPaths[0].kindBounceAndReserved[2] = f.program.diffractionPatternMask;
    if (f.program.diffractionPatternMask)
        std::memcpy (&f.program.diffractionPaths[0].kindBounceAndReserved[3], &f.program.diffractionMaskCoverage, sizeof (float));
    auto receiver = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram> (*f.program.objectPrograms[0]);
    receiver->spectralReflectance = f.input.receiver.reflectance;
    f.program.objectPrograms[0] = std::move (receiver);
    f.program.programIdentity = arbitgpu::nativeFixtureDiffractionProgramIdentity (
        f.program.bindingDigest, *admitted, f.program.diffractionPatternMask, f.program.diffractionMaskCoverage,
        f.program.diffractionProductDigest, false, &f.program);
    return f;
}

struct Outcome
{
    bool ok = false;
    std::string error;
    double seconds = 0.0;
    std::vector<float> pixels;
    bool realtimeApplied = false;
};

Outcome renderBackend (const std::shared_ptr<const Visual3DScene>& scene,
                       const std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram>& program,
                       bool previewUse)
{
    Outcome outcome;
    auto& backend = arbitgpu::nativeFixtureSceneBackend();
    const auto started = std::chrono::steady_clock::now();
    std::optional<arbitgpu::realtimeengine::Scope> scope;
    if (gRealtime) scope.emplace();
    const auto prepared = backend.prepare (scene, program);
    if (! prepared.prepared) { outcome.error = "prepare: " + prepared.error; return outcome; }
    arbitgpu::NativeFixtureSceneRuntimeInputs runtime;
    runtime.linearColor = true;
    runtime.realtimeEngine = gRealtime;
    runtime.previewUse = previewUse;
    const auto rendered = backend.render (scene, prepared.resources, kWidth, kHeight, runtime);
    if (! rendered.rendered || ! rendered.frame) { outcome.error = "render: " + rendered.error; return outcome; }
    outcome.realtimeApplied = rendered.stats.realtimeProgramApplied;
    if (! readColor (rendered.frame, outcome.pixels, outcome.error)) { outcome.error = "readback: " + outcome.error; return outcome; }
    outcome.seconds = std::chrono::duration<double> (std::chrono::steady_clock::now() - started).count();
    outcome.ok = true;
    return outcome;
}

Outcome renderCard()
{
    Outcome outcome;
    const auto started = std::chrono::steady_clock::now();
    const auto card = videohelper::fixture3d::loadHolographicTradingCardScene (outcome.error);
    if (! card) return outcome;
    const auto binding = videorender::fixture3d::admitDiffractionMaterialBinding (
        card->scene, *card->operation.diffractionMaterial, videohelper::materialprogram::BackendTarget::Metal, outcome.error);
    if (! binding) return outcome;
    auto& backend = arbitgpu::nativeFixtureSceneBackend();
    videorender::fixture3d::FixtureSceneRenderer renderer (backend);
    videorender::fixture3d::SceneEngineSelection selection;
    selection.engine = gRealtime ? videorender::fixture3d::SceneEngineKind::Realtime
                                 : videorender::fixture3d::SceneEngineKind::Raster;
    renderer.selectEngine (selection);
    videorender::fixture3d::RenderedFrame frame;
    arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
    inputs.linearColor = true;
    if (! renderer.renderExport (card->scene, binding, inputs, { kWidth, kHeight },
                                 videorender::fixture3d::kNativeGpuCapability, frame, outcome.error))
    { outcome.error = "renderExport: " + outcome.error; return outcome; }
    outcome.realtimeApplied = frame.stats.realtimeProgramApplied;
    if (! readColor (frame.nativeFrame, outcome.pixels, outcome.error)) { outcome.error = "readback: " + outcome.error; return outcome; }
    outcome.seconds = std::chrono::duration<double> (std::chrono::steady_clock::now() - started).count();
    outcome.ok = true;
    return outcome;
}

void report (const std::string& name, const std::string& directory, const Outcome& outcome)
{
    if (! outcome.ok)
    {
        std::printf ("%s: FAIL %s\n", name.c_str(), outcome.error.c_str());
        return;
    }
    double mean = 0.0;
    std::size_t nonfinite = 0;
    const auto path = directory + "/" + name + ".ppm";
    const bool written = writePpm (path, outcome.pixels, kWidth, kHeight, mean, nonfinite);
    std::printf ("%s: OK %.3fs realtime=%d mean=%.5f nonfinite=%zu %s\n", name.c_str(), outcome.seconds,
                 int (outcome.realtimeApplied), mean, nonfinite, written ? path.c_str() : "(write failed)");
}
} // namespace

int main (int argc, char** argv)
{
    @autoreleasepool
    {
        if (argc < 2) { std::printf ("usage: %s <outdir> [big|orb|orb-preview|disc|card]...\n", argv[0]); return 2; }
        const std::string directory = argv[1];
        // HARNESS_SIZE=WxH renders smaller frames; HARNESS_RASTER=1 draws the Raster engine.
        if (const char* size = std::getenv ("HARNESS_SIZE")) std::sscanf (size, "%ux%u", &kWidth, &kHeight);
        gRealtime = std::getenv ("HARNESS_RASTER") == nullptr;
        std::vector<std::string> cases;
        for (int i = 2; i < argc; ++i) cases.emplace_back (argv[i]);
        if (cases.empty()) cases = { "big", "orb", "disc", "card" };
        const auto info = arbitgpu::nativeFixtureSceneBackend().info();
        std::printf ("backend %s available=%d %s\n", info.backend.c_str(), int (info.available), info.error.c_str());
        int failures = 0;
        for (const auto& name : cases)
        {
            Outcome outcome;
            if (name == "big")
            {
                const auto scene = bigScene();
                if (scene) outcome = renderBackend (scene, nullptr, false);
                else outcome.error = "no scene";
            }
            else if (name == "orb" || name == "orb-preview" || name == "disc")
            {
                std::string error;
                auto f = orbFixture (error, name == "disc");
                if (! error.empty()) outcome.error = "orb fixture: " + error;
                else outcome = renderBackend (f.scene,
                    std::make_shared<const arbitgpu::NativeFixtureSurfaceMaterialProgram> (f.program), name == "orb-preview");
            }
            else if (name == "card") outcome = renderCard();
            else outcome.error = "unknown case";
            report (name, directory, outcome);
            if (! outcome.ok) ++failures;
        }
        return failures == 0 ? 0 : 1;
    }
}
