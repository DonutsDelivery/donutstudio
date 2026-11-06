// A GPU check of the authored receiver transport, the pass the diffraction orb's plate
// draws, on a hardware OpenGL context. The triangle-transport fixture holds a
// receiver plate and a grating under one authored spectral source. Each view leaves
// background pixels around the plate. Every case renders the frame through the
// fixture backend, whose whole-output readback refuses any nonfinite accumulation and
// names the pass, then reads the published Color and counts nonfinite values. Raster
// and Realtime both run, so a Realtime-only refusal is told apart from a shared one.
//
// A full-HD receiver frame then reads its GPU time per submission: the plate's receiver
// draw splits into the grating's row bands as the grating does, and no submission
// may near the driver's context switch timeout, which killed the orb's plate draw.
//
// The orb case swaps the grating quad for the Diffraction Orb starter's sphere, 64
// segments and 32 rings, and its grating: a hexagonal lattice of three 1400 nm groove
// families to order 4, sinusoidal and 90 nm deep. The starter's four glowing panels
// and dim sky stand around it. It requires the whole Realtime receiver frame at
// 1920x1080 to finish on the GPU in under a second while the receiver still lights.
// The lattice's 27 profile amplitudes per order, read in every gathered row and every
// sky sub-wavelength, held one row band of the live orb past the driver's timeout.
//
// The live orb case renders the starter's own view at the live frame's 960x540: its
// camera, a floor and back wall, a raking source the wall shades, and no light but the
// panels and the sky. The fixture's orb view lit its plate by the source and missed
// Realtime's black floor and lower ball. It requires the floor, plate, backdrop and both
// halves of the ball lit, and the orders free of hard-edged blocks.
//
// ARBIT_ORB_REPLAY=<dir> redraws a frame the app dumped with ARBIT_GL_DUMP_REALTIME_SCENE
// and compares it with the dumped Color, so the live frame is judged, not a stand-in.
//
// The host half checks the readback's naming: a half-float target keeps no NaN payload
// bits, so the probes also write minus the reason in alpha.

#include "gpu_backend/backend.h"
#include "gpu_backend/fixture_realtime_opengl.h"
#include "gpu_backend/grating_submission_clock.h"
#include "support/diffraction_opengl_transport_fixture.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <GL/gl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// Defined in backend_opengl.cpp.
namespace arbitgpu { std::vector<std::string> realtimeFixtureVariantCompileErrors(); }

namespace
{
namespace nt = arbitgpu::diffractiontransportnative;
namespace tg = triangletransporttests::opengl;
int failures = 0;
void check (bool condition, const std::string& text)
{
    if (condition) return;
    ++failures;
    std::fprintf (stderr, "FAIL: %s\n", text.c_str());
}
float bitsFloat (std::uint32_t bits)
{
    float value = 0.0f;
    std::memcpy (&value, &bits, sizeof (value));
    return value;
}

void namingChecks()
{
    std::string error;
    // A probe's NaN after a half-float store: the payload is gone, the alpha is not.
    std::vector<float> lost (16, 0.0f);
    lost[0] = lost[1] = lost[2] = bitsFloat (0x7fffffffu);
    lost[3] = -12.0f;
    check (! nt::validateWholeOutput (lost, 2, 2, error) && error.find ("receiver output nonfinite: 3") != std::string::npos,
           "a payload lost to a half-float target is named by its alpha");
    lost[3] = -15.0f;
    check (! nt::validateWholeOutput (lost, 2, 2, error) && error.find ("composite output nonfinite: 3") != std::string::npos,
           "the composite probe is named by its alpha");
    // An arithmetic NaN with an ordinary alpha stays unlisted, and a fractional or
    // out-of-range alpha names nothing.
    lost[3] = 1.0f;
    check (! nt::validateWholeOutput (lost, 2, 2, error) && error.find ("unlisted NaN payload: 3") != std::string::npos,
           "an arithmetic NaN with alpha one stays unlisted");
    lost[3] = -2.5f;
    check (! nt::validateWholeOutput (lost, 2, 2, error) && error.find ("unlisted NaN payload: 3") != std::string::npos,
           "a fractional alpha names no reason");
    lost[3] = -40.0f;
    check (! nt::validateWholeOutput (lost, 2, 2, error) && error.find ("unlisted NaN payload: 3") != std::string::npos,
           "an alpha past the listed reasons names no reason");
    // A payload that survived (RGBA32F) keeps its own name over the alpha.
    lost[0] = lost[1] = lost[2] = bitsFloat (0x7fc00000u | (3u << 10u));
    lost[3] = -12.0f;
    check (! nt::validateWholeOutput (lost, 2, 2, error) && error.find ("sample segment query incomplete: 3") != std::string::npos,
           "a surviving payload keeps its own name");
}

// The Diffraction Orb starter's grating (BuiltInVisualModules.h, diffractionOrbStarter)
// as its material node lowers it: groove direction 90 degrees and the second family 60
// degrees on, both 1400 nm, with the third family; maximum order choice 3 is last order
// 4; a sinusoidal profile 90 nm deep; rms slope 0.02.
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

// Replaces the grating quad, the fixture's last object, with the Diffraction Orb's UV
// sphere (64 segments, 32 rings), wound and lit outward, over a plate four times the
// fixture's. The camera looks down on both from outside the sphere, as the starter's
// does; the fixture's receiver camera at z 0.8 sat inside it. The starter's set is half
// size here, as the ball is: its front and rear squares and its green and purple walls
// glow, eight emitter triangles, under a dim analytic sky with no sun. The starter's y
// up is z here.
void makeOrb (triangletransporttests::Fixture& f)
{
    {
        const auto& plate = f.scene->objects[0];
        for (std::size_t v = 0; v < plate.vertexCount; ++v)
        {
            auto& position = f.scene->vertices[plate.firstVertex + v].position;
            position.x *= 4.0f;
            position.y *= 4.0f;
        }
        auto& camera = f.scene->cameras[0];
        camera.transform.translation = { 0.0f, 0.0f, 1.6f };
        camera.transform.rotation = HarmonicMIDI::grid::SceneQuaternion { 0, 0, 0, 1 };
    }
    constexpr unsigned segments = 64, rings = 32;
    constexpr float radius = 0.25f, pi = 3.14159265358979f;
    const std::array<float, 3> center { 0.0f, 0.0f, 1.0f };
    auto& scene = *f.scene;
    HarmonicMIDI::grid::Visual3DSceneGrowth growth;
    constexpr unsigned panels = 4;
    growth.vertices = std::size_t (segments + 1) * (rings + 1) + 4 * panels;
    growth.indices = std::size_t (segments) * (rings - 1) * 6 + 6 * panels;
    std::string growthError;
    check (HarmonicMIDI::grid::growVisual3DScene (scene, growth, growthError), "scene grows for the orb's sphere");
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
            if (ring != 0) triangle (at (ring, segment), at (ring, segment + 1), at (ring + 1, segment));
            if (ring + 1 != rings) triangle (at (ring, segment + 1), at (ring + 1, segment + 1), at (ring + 1, segment));
        }
    object.vertexCount = scene.vertexCount - object.firstVertex;
    object.indexCount = scene.indexCount - object.firstIndex;

    using D3 = std::array<double, 3>;
    struct Panel { D3 centre, u, v; HarmonicMIDI::grid::SceneVec3 emission; };
    const std::array<Panel, panels> set { {
        { { -0.4, -2.25, 1.55 }, { 0.1125, 0, 0 }, { 0, 0, 0.1125 }, { 40, 40, 40 } },
        { { 0.0, 0.75, 2.25 }, { 0.125, 0, 0 }, { 0, 0.125, 0 }, { 20, 20, 20 } },
        { { -2.0, -1.125, 1.5 }, { 0, 0.75, 0 }, { 0, 0, 0.5 }, { 0.2f, 1.8f, 0.6f } },
        { { 2.0, -1.125, 1.5 }, { 0, 0.75, 0 }, { 0, 0, 0.5 }, { 1.2f, 0.2f, 1.8f } } } };
    for (std::size_t i = 0; i < set.size(); ++i)
    {
        const auto& panel = set[i];
        std::array<D3, 4> corners;
        for (unsigned k = 0; k < 4; ++k)
        {
            const double su = k == 1 || k == 2 ? 1 : -1, sv = k >= 2 ? 1 : -1;
            for (unsigned a = 0; a < 3; ++a) corners[k][a] = panel.centre[a] + su * panel.u[a] + sv * panel.v[a];
        }
        f.addObject (corners, std::uint32_t (104 + i), std::uint32_t (204 + i));
        scene.materials[scene.materialCount - 1].emissive = panel.emission;
        const auto& added = scene.objects[scene.objectCount - 1];
        const D3 n { panel.u[1] * panel.v[2] - panel.u[2] * panel.v[1], panel.u[2] * panel.v[0] - panel.u[0] * panel.v[2],
                     panel.u[0] * panel.v[1] - panel.u[1] * panel.v[0] };
        const double length = std::sqrt (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        for (std::size_t v = 0; v < added.vertexCount; ++v)
        {
            auto& vertex = scene.vertices[added.firstVertex + v];
            vertex.normal = { float (n[0] / length), float (n[1] / length), float (n[2] / length) };
            vertex.uv = { v == 1 || v == 2 ? 1.0f : 0.0f, v >= 2 ? 1.0f : 0.0f };
        }
    }
    auto& sky = scene.lights[scene.lightCount++];
    sky.id = HarmonicMIDI::grid::SceneLightId { 401 };
    sky.kind = HarmonicMIDI::grid::SceneLightKind::Environment;
    sky.sky.version = 1;
    sky.sky.zenith = { 0.03f, 0.03f, 0.04f };
    sky.sky.horizon = { 0.07f, 0.07f, 0.07f };
    sky.sky.sunIntensity = 0.0f;
}

// The live starter's view of the orb (BuiltInVisualModules.h, diffractionOrbStarter) at
// half size with z up: the ball rests on a plate of four radii, a grey floor 0.01 below
// it and a near-black back wall stand in the room, the camera sits 0.375 up and 0.8 in
// front looking 10 degrees down with the starter's 52 degree field, and the authored
// source rakes in from behind the ball 4 degrees up, where the back wall shades it. The
// room is lit only by its glowing panels and the dim sky, as the live frame is: the
// fixture's own view keeps a plate lit by the source, which hid a black live floor.
namespace liveorb
{
constexpr float kCentre = 0.25f, kCameraY = -0.8f, kCameraZ = 0.375f, kFov = 0.90757121f;
constexpr float kPitch = 0.17453293f; // 10 degrees down
}
void makeLiveOrb (triangletransporttests::Fixture& f)
{
    makeOrb (f);
    auto& scene = *f.scene;
    {
        const auto& plate = scene.objects[0];
        for (std::size_t v = 0; v < plate.vertexCount; ++v)
        {
            auto& position = scene.vertices[plate.firstVertex + v].position;
            position.x *= 0.25f;
            position.y *= 0.3125f;
        }
    }
    // The sphere and the set come down so the ball rests on the plate.
    for (std::size_t o = 1; o < scene.objectCount; ++o)
    {
        const auto& object = scene.objects[o];
        for (std::size_t v = 0; v < object.vertexCount; ++v) scene.vertices[object.firstVertex + v].position.z -= 0.75f;
    }
    HarmonicMIDI::grid::Visual3DSceneGrowth growth;
    growth.vertices = 8;
    growth.indices = 12;
    growth.objects = 2;
    growth.materials = 2;
    std::string growthError;
    check (HarmonicMIDI::grid::growVisual3DScene (scene, growth, growthError), "scene grows for the live orb's floor and wall");
    using D3 = std::array<double, 3>;
    const auto addPlain =[&] (std::array<D3, 4> corners, std::uint32_t id, HarmonicMIDI::grid::SceneVec3 colour,
                               HarmonicMIDI::grid::SceneVec3 normal)
    {
        f.addObject (corners, id, id + 100);
        auto& material = scene.materials[scene.materialCount - 1];
        material.baseColor = colour;
        material.doubleSided = true;
        const auto& added = scene.objects[scene.objectCount - 1];
        for (std::size_t v = 0; v < added.vertexCount; ++v)
        {
            auto& vertex = scene.vertices[added.firstVertex + v];
            vertex.normal = normal;
            vertex.uv = { v == 1 || v == 2 ? 1.0f : 0.0f, v >= 2 ? 1.0f : 0.0f };
        }
    };
    addPlain ({ { { -10, -10, -0.01 }, { 10, -10, -0.01 }, { 10, 10, -0.01 }, { -10, 10, -0.01 } } }, 110,
              { 0.35f, 0.35f, 0.35f }, { 0, 0, 1 });
    addPlain ({ { { -10, 1.25, -2 }, { 10, 1.25, -2 }, { 10, 1.25, 8 }, { -10, 1.25, 8 } } }, 111,
              { 0.02f, 0.02f, 0.02f }, { 0, -1, 0 });
    auto& camera = scene.cameras[0];
    camera.transform.translation = { 0.0f, liveorb::kCameraY, liveorb::kCameraZ };
    // 80 degrees about x turns the camera's -z to (0, cos 10, -sin 10).
    const float half = 0.5f * (1.5707963f - liveorb::kPitch);
    camera.transform.rotation = HarmonicMIDI::grid::SceneQuaternion { std::sin (half), 0, 0, std::cos (half) };
    camera.verticalFovRadians = liveorb::kFov;
    const double length = std::sqrt (0.36 * 0.36 + 0.93 * 0.93 + 0.07 * 0.07);
    f.input.source.direction = { -0.36 / length, 0.93 / length, 0.07 / length };
    f.input.source.radiance.fill (24.0);
}

// Renders one view through the fixture backend and reads its published Color.
bool render (arbitgpu::NativeFixtureSceneBackend& backend, bool grating, bool realtime,
             std::uint32_t width, std::uint32_t height, std::vector<float>& pixels, std::string& error,
             bool orb = false, bool live = false)
{
    auto f = tg::scene (grating);
    if (live) makeLiveOrb (f);
    else if (orb) makeOrb (f);
    const auto orbMaterial = orbGrating();
    if (! tg::publishPhysics (f, error, orb || live ? &orbMaterial : nullptr)) return false;
    const auto program = std::make_shared<const arbitgpu::NativeFixtureSurfaceMaterialProgram> (f.program);
    std::optional<arbitgpu::realtimeengine::Scope> scope;
    if (realtime) scope.emplace();
    const auto prepared = backend.prepare (f.scene, program);
    if (! prepared.prepared) { error = "prepare: " + prepared.error; return false; }
    arbitgpu::NativeFixtureSceneRuntimeInputs runtime;
    runtime.linearColor = true;
    const auto rendered = backend.render (f.scene, prepared.resources, width, height, runtime);
    if (! rendered.rendered || ! rendered.frame) { error = "render: " + rendered.error; return false; }
    if (realtime && ! rendered.stats.realtimeProgramApplied) { error = "the Realtime program was not applied"; return false; }
    GLint texture = 0;
    glGetIntegerv (GL_TEXTURE_BINDING_2D, &texture);
    glBindTexture (GL_TEXTURE_2D, static_cast<GLuint> (rendered.frame->colorImageHandle()));
    pixels.assign (std::size_t (width) * height * 4, 0.0f);
    glGetTexImage (GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, pixels.data());
    const auto gpuError = glGetError();
    glBindTexture (GL_TEXTURE_2D, static_cast<GLuint> (texture));
    if (gpuError != GL_NO_ERROR) { error = "Color readback failed"; return false; }
    return true;
}

// Every Realtime program variant compiles, with and without the grating: a scene with no
// grating once failed on uniforms only the grating's text declares.
void variantChecks()
{
    const auto errors = arbitgpu::realtimeFixtureVariantCompileErrors();
    for (const auto& error : errors) std::printf ("Realtime variant: %s\n", error.c_str());
    check (errors.empty(), "every Realtime fixture program variant compiles");
}

void frameChecks (arbitgpu::NativeFixtureSceneBackend& backend)
{
    constexpr std::uint32_t width = 160, height = 90;
    for (const bool realtime : { false, true })
        for (const bool grating : { false, true })
        {
            const std::string name = std::string (realtime ? "Realtime" : "Raster") + (grating ? " grating view" : " receiver view");
            std::vector<float> pixels;
            std::string error;
            const bool rendered = render (backend, grating, realtime, width, height, pixels, error);
            if (! rendered) std::printf ("%s: refused: %s\n", name.c_str(), error.c_str());
            check (rendered, name + " renders with a whole-output readback that admits");
            if (! rendered) continue;
            std::size_t nonfinite = 0, background = 0, lit = 0;
            double sum[3] {};
            for (std::size_t p = 0; p < std::size_t (width) * height; ++p)
            {
                bool bad = false;
                for (unsigned c = 0; c < 4; ++c) bad = bad || ! std::isfinite (pixels[p * 4 + c]);
                if (bad) { ++nonfinite; continue; }
                const float r = pixels[p * 4], g = pixels[p * 4 + 1], b = pixels[p * 4 + 2];
                if (r == 0.0f && g == 0.0f && b == 0.0f) ++background;
                else ++lit;
                sum[0] += r; sum[1] += g; sum[2] += b;
            }
            const double count = double (width) * height;
            std::printf ("%s: nonfinite %zu, black %zu, lit %zu, mean rgb %.6g %.6g %.6g\n", name.c_str(), nonfinite,
                         background, lit, sum[0] / count, sum[1] / count, sum[2] / count);
            check (nonfinite == 0, name + " publishes no nonfinite pixel");
            check (lit > 0, name + " lights some pixels");
        }
}

void submissionChecks (arbitgpu::NativeFixtureSceneBackend& backend)
{
    constexpr std::uint32_t width = 1920, height = 1080;
    for (const bool realtime : { false, true })
    {
        const std::string name = std::string (realtime ? "Realtime" : "Raster") + " receiver view 1920x1080";
        std::vector<float> pixels;
        std::string error;
        const bool rendered = render (backend, false, realtime, width, height, pixels, error);
        if (! rendered) std::printf ("%s: refused: %s\n", name.c_str(), error.c_str());
        check (rendered, name + " renders");
        if (! rendered) continue;
        const auto spans = arbitgpu::lastGratingSubmissions();
        const arbitgpu::GratingSubmissionSpan* longest = nullptr;
        double total = 0.0, surfaceBands = 0.0, gratingBands = 0.0;
        for (const auto& span : spans)
        {
            total += span.milliseconds;
            if (span.name.rfind ("surface ", 0) == 0 && span.name.find (" band ") != std::string::npos) ++surfaceBands;
            if (span.name.rfind ("grating band ", 0) == 0) ++gratingBands;
            if (! longest || span.milliseconds > longest->milliseconds) longest = &span;
        }
        std::printf ("%s: %zu submissions over %.1f ms, %.0f grating bands, %.0f surface bands, longest %s %.2f ms\n",
                     name.c_str(), spans.size(), total, gratingBands, surfaceBands,
                     longest ? longest->name.c_str() : "-", longest ? longest->milliseconds : 0.0);
        check (! spans.empty(), name + " times its submissions");
        // Raster's receiver view gathers every row at full resolution, so it must band.
        // Realtime gathers at reduced resolution (realtime_diffraction_gather.h) and bands
        // each pass by its own work, which for this view fits one submission; whenever
        // its grating does band, the receiver surfaces band with it.
        check ((realtime || gratingBands > 1) && surfaceBands >= gratingBands,
               name + " draws its receiver surfaces in the grating's row bands");
        check (longest && longest->milliseconds < 1000.0, name + " keeps every submission under a second");
    }
}

void orbChecks (arbitgpu::NativeFixtureSceneBackend& backend)
{
    constexpr std::uint32_t width = 1920, height = 1080;
    const std::string name = "Realtime orb receiver view 1920x1080";
    std::vector<float> pixels;
    std::string error;
    const bool rendered = render (backend, false, true, width, height, pixels, error, true);
    if (! rendered) std::printf ("%s: refused: %s\n", name.c_str(), error.c_str());
    check (rendered, name + " renders");
    if (! rendered) return;
    const auto spans = arbitgpu::lastGratingSubmissions();
    double total = 0.0, longest = 0.0;
    for (const auto& span : spans) { total += span.milliseconds; longest = std::max (longest, span.milliseconds); }
    std::size_t lit = 0, nonfinite = 0;
    for (std::size_t p = 0; p < std::size_t (width) * height; ++p)
    {
        const float* c = &pixels[p * 4];
        if (! std::isfinite (c[0]) || ! std::isfinite (c[1]) || ! std::isfinite (c[2])) ++nonfinite;
        else if (c[0] > 0.0f || c[1] > 0.0f || c[2] > 0.0f) ++lit;
    }
    std::string longestName = "-";
    for (const auto& span : spans) if (span.milliseconds == longest) longestName = span.name;
    std::printf ("%s: %zu submissions, GPU %.1f ms, longest %s %.2f ms, lit %zu, nonfinite %zu\n",
                 name.c_str(), spans.size(), total, longestName.c_str(), longest, lit, nonfinite);
    check (! spans.empty(), name + " times its submissions");
    check (total < 1000.0, name + " finishes on the GPU in under a second");
    check (lit > 0 && nonfinite == 0, name + " lights its receiver with finite pixels");
}
// Region means of the live view and its hard-edge share (liveorb). A point projects to
// the bottom-up pixel the readback holds.
struct LiveOrbRegions
{
    double floor = 0, plate = 0, backdrop = 0, upperOrb = 0, lowerOrb = 0, hardEdges = 0;
};
LiveOrbRegions liveOrbRegions (const std::vector<float>& pixels, std::uint32_t width, std::uint32_t height)
{
    const double c = std::cos (double (liveorb::kPitch)), s = std::sin (double (liveorb::kPitch));
    const double tanHalf = std::tan (0.5 * double (liveorb::kFov)), aspect = double (width) / double (height);
    const auto project = [&] (double x, double y, double z)
    {
        const double dx = x, dy = y - liveorb::kCameraY, dz = z - liveorb::kCameraZ;
        const double depth = dy * c - dz * s, up = dy * s + dz * c;
        return std::array<double, 2> { (0.5 + 0.5 * dx / (depth * tanHalf * aspect)) * width,
                                       (0.5 + 0.5 * up / (depth * tanHalf)) * height };
    };
    const auto luminance = [&] (std::size_t x, std::size_t y)
    {
        const float* p = &pixels[(y * width + x) * 4];
        return 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
    };
    const auto mean = [&] (double x, double y, double z)
    {
        const auto at = project (x, y, z);
        const int radius = std::max (1, int (height / 108));
        double sum = 0; int count = 0;
        for (int j = -radius; j <= radius; ++j)
            for (int i = -radius; i <= radius; ++i)
            {
                const int px = int (at[0]) + i, py = int (at[1]) + j;
                if (px < 0 || py < 0 || px >= int (width) || py >= int (height)) continue;
                const double value = luminance (std::size_t (px), std::size_t (py));
                if (std::isfinite (value)) { sum += value; ++count; }
            }
        return count ? sum / count : 0.0;
    };
    const double r = 0.25, zc = liveorb::kCentre;
    LiveOrbRegions regions;
    regions.floor = mean (1.2, 1.0, -0.01);
    regions.plate = mean (0.45, 0.2, 0.0);
    regions.backdrop = mean (1.5, 1.25, 0.5);
    regions.upperOrb = mean (0.0, -r * 0.866, zc + r * 0.5);
    regions.lowerOrb = mean (0.0, -r * 0.866, zc - r * 0.5);
    // Inside the ball's disc, the share of horizontal neighbours whose luminance steps by
    // more than a quarter of the brighter one: a mosaic of hard-edged blocks steps at every
    // block border, round soft spots ramp over several pixels.
    const auto centre = project (0.0, 0.0, zc), rim = project (r, 0.0, zc);
    const double disc = 0.8 * std::abs (rim[0] - centre[0]);
    std::size_t pairs = 0, steps = 0;
    for (int y = int (centre[1] - disc); y <= int (centre[1] + disc); ++y)
        for (int x = int (centre[0] - disc); x < int (centre[0] + disc); ++x)
        {
            if (x < 0 || y < 0 || x + 1 >= int (width) || y >= int (height)) continue;
            if (std::hypot (x - centre[0], y - centre[1]) > disc) continue;
            const double a = luminance (std::size_t (x), std::size_t (y)), b = luminance (std::size_t (x + 1), std::size_t (y));
            if (! std::isfinite (a) || ! std::isfinite (b) || std::max (a, b) < 1.0e-3) continue;
            ++pairs;
            if (std::abs (a - b) > 0.25 * std::max (a, b)) ++steps;
        }
    regions.hardEdges = pairs ? double (steps) / double (pairs) : 1.0;
    return regions;
}

// The live starter's view in Realtime at the live frame's 960x540: the floor, the plate,
// the backdrop and both halves of the ball are lit, as Path Trace lights them, and the
// orders show soft spots, not a mosaic of hard-edged blocks.
void liveOrbChecks (arbitgpu::NativeFixtureSceneBackend& backend)
{
    constexpr std::uint32_t width = 960, height = 540;
    const std::string name = "Realtime live orb view 960x540";
    std::vector<float> pixels;
    std::string error;
    const bool rendered = render (backend, false, true, width, height, pixels, error, true, true);
    if (! rendered) std::printf ("%s: refused: %s\n", name.c_str(), error.c_str());
    check (rendered, name + " renders");
    if (! rendered) return;
    const auto regions = liveOrbRegions (pixels, width, height);
    std::printf ("%s: luminance floor %.5g, plate %.5g, backdrop %.5g, upper orb %.5g, lower orb %.5g, "
                 "hard edges %.4f\n", name.c_str(), regions.floor, regions.plate, regions.backdrop, regions.upperOrb,
                 regions.lowerOrb, regions.hardEdges);
    check (regions.floor > 1.0e-3, name + " lights the floor");
    check (regions.plate > 1.0e-3, name + " lights the receiver plate");
    check (regions.backdrop > 0.0, name + " lights the backdrop");
    check (regions.upperOrb > 1.0e-3 && regions.lowerOrb > 1.0e-3, name + " lights both halves of the ball");
    check (regions.hardEdges < 0.08, name + " shows the orders without hard-edged blocks");
}
// ARBIT_ORB_MEASURE=WxH renders only the orb case at that size and prints its GPU time
// by submission kind, for ablation runs (ARBIT_GL_REALTIME_ABLATE) that draw no real frame.
int orbMeasure (const char* size)
{
    unsigned width = 0, height = 0;
    if (std::sscanf (size, "%ux%u", &width, &height) != 2 || width == 0 || height == 0)
    {
        std::fprintf (stderr, "ARBIT_ORB_MEASURE wants WxH, not %s\n", size);
        return 2;
    }
    std::vector<float> pixels;
    std::string error;
    // ARBIT_ORB_LIVE=1 measures the live starter's view (makeLiveOrb) instead.
    const bool live = std::getenv ("ARBIT_ORB_LIVE") != nullptr;
    if (! render (arbitgpu::nativeFixtureSceneBackend(), false, true, width, height, pixels, error, true, live))
    {
        std::printf ("orb measure %ux%u: refused: %s\n", width, height, error.c_str());
        return 1;
    }
    double grating = 0.0, gratingLongest = 0.0, surface = 0.0, surfaceLongest = 0.0, other = 0.0;
    std::size_t gratingCount = 0, surfaceCount = 0;
    for (const auto& span : arbitgpu::lastGratingSubmissions())
    {
        if (span.name.rfind ("grating band ", 0) == 0)
        { grating += span.milliseconds; gratingLongest = std::max (gratingLongest, span.milliseconds); ++gratingCount; }
        else if (span.name.rfind ("surface ", 0) == 0)
        { surface += span.milliseconds; surfaceLongest = std::max (surfaceLongest, span.milliseconds); ++surfaceCount; }
        else other += span.milliseconds;
    }
    std::size_t lit = 0;
    for (std::size_t p = 0; p < std::size_t (width) * height; ++p)
        if (pixels[p * 4] > 0.0f || pixels[p * 4 + 1] > 0.0f || pixels[p * 4 + 2] > 0.0f) ++lit;
    std::printf ("orb measure %ux%u: total %.2f ms, grating %zu bands %.2f ms (longest %.2f), "
                 "surface %zu %.2f ms (longest %.2f), other %.2f ms, lit %zu\n",
                 width, height, grating + surface + other, gratingCount, grating, gratingLongest,
                 surfaceCount, surface, surfaceLongest, other, lit);
    if (live)
    {
        const auto regions = liveOrbRegions (pixels, width, height);
        std::printf ("orb live regions %ux%u: floor %.5g, plate %.5g, backdrop %.5g, upper orb %.5g, lower orb %.5g, "
                     "hard edges %.4f\n", width, height, regions.floor, regions.plate, regions.backdrop,
                     regions.upperOrb, regions.lowerOrb, regions.hardEdges);
    }
    // ARBIT_ORB_DUMP=path writes the frame as "W H\n" then W*H*4 raw floats, for comparing
    // the reduced-resolution gather against full resolution.
    if (const char* dump = std::getenv ("ARBIT_ORB_DUMP"))
    {
        std::FILE* file = std::fopen (dump, "wb");
        if (file == nullptr) { std::fprintf (stderr, "ARBIT_ORB_DUMP: cannot write %s\n", dump); return 1; }
        std::fprintf (file, "%u %u\n", width, height);
        const bool written = std::fwrite (pixels.data(), sizeof (float), pixels.size(), file) == pixels.size();
        if (std::fclose (file) != 0 || ! written) { std::fprintf (stderr, "ARBIT_ORB_DUMP: short write %s\n", dump); return 1; }
    }
    return 0;
}
// ARBIT_ORB_REPLAY=<dir> redraws a frame the app dumped (ARBIT_GL_DUMP_REALTIME_SCENE,
// backend_opengl.cpp) through the same backend call: the dumped geometry, objects,
// materials, camera and lights, and the drawn material programs rebuilt from their
// dumped blocks over the fixture's program, which supplies only the binding digest and
// the fields the dump has no blocks for. It prints how the replay's Color compares with
// the dumped one and, with ARBIT_ORB_DUMP, writes the replay frame.
namespace replay
{
bool readFile (const std::string& path, std::vector<char>& bytes)
{
    std::FILE* file = std::fopen (path.c_str(), "rb");
    if (file == nullptr) return false;
    bytes.clear();
    char buffer[65536];
    for (;;)
    {
        const std::size_t n = std::fread (buffer, 1, sizeof (buffer), file);
        if (n == 0) break;
        bytes.insert (bytes.end(), buffer, buffer + n);
    }
    std::fclose (file);
    return true;
}
template <typename Value>
bool readBlock (const std::string& path, Value& value, std::string& error)
{
    std::vector<char> bytes;
    if (! readFile (path, bytes)) { error = "missing " + path; return false; }
    if (bytes.size() != sizeof (Value))
    { error = path + " holds " + std::to_string (bytes.size()) + " bytes, not " + std::to_string (sizeof (Value)); return false; }
    std::memcpy (static_cast<void*> (&value), bytes.data(), sizeof (Value));
    return true;
}
template <typename Buffer>
bool readRecords (const std::string& path, Buffer& buffer, std::size_t& count, std::string& error)
{
    using Record = typename Buffer::value_type;
    std::vector<char> bytes;
    if (! readFile (path, bytes)) { error = "missing " + path; return false; }
    if (bytes.size() % sizeof (Record) != 0) { error = path + " is not whole records"; return false; }
    count = bytes.size() / sizeof (Record);
    buffer.resize (std::max<std::size_t> (count, 1));
    if (count != 0) std::memcpy (static_cast<void*> (buffer.data()), bytes.data(), bytes.size());
    return true;
}
// The number after a keyword on a dumped line, or false when the keyword is absent.
template <typename... Values>
bool after (const std::string& line, const char* keyword, const char* format, Values*... values)
{
    const auto at = line.find (std::string (" ") + keyword + " ");
    if (at == std::string::npos) return false;
    return std::sscanf (line.c_str() + at + std::strlen (keyword) + 2, format, values...) == int (sizeof... (Values));
}
std::vector<std::string> lines (const std::string& path)
{
    std::vector<char> bytes;
    std::vector<std::string> out;
    if (! readFile (path, bytes)) return out;
    std::string line;
    for (const char c : bytes)
        if (c == '\n') { out.push_back (" " + line); line.clear(); }
        else line += c;
    return out;
}

int run (const std::string& dir)
{
    using namespace HarmonicMIDI::grid;
    std::string error;
    const auto sceneText = lines (dir + "/scene.txt");
    const auto programText = lines (dir + "/program.txt");
    if (sceneText.empty() || programText.empty())
    {
        std::printf ("orb replay: refused: %s has no scene.txt or program.txt (a dump from before the program "
                     "blocks cannot be replayed)\n", dir.c_str());
        return 1;
    }
    unsigned width = 0, height = 0;
    int linearColor = 1;
    float emissionGain = 1.0f;
    after (sceneText[0], "size", "%ux%u", &width, &height);
    after (sceneText[0], "linearColor", "%d", &linearColor);
    after (sceneText[0], "emissionGain", "%f", &emissionGain);
    if (width == 0 || height == 0) { std::printf ("orb replay: refused: no frame size in scene.txt\n"); return 1; }

    auto f = tg::scene (false);
    makeOrb (f);
    const auto orbMaterial = orbGrating();
    if (! tg::publishPhysics (f, error, &orbMaterial)) { std::printf ("orb replay: refused: %s\n", error.c_str()); return 1; }

    auto scene = std::make_shared<Visual3DScene>();
    scene->id = f.scene->id;
    std::size_t lightTotal = 0;
    if (! readRecords (dir + "/scene-vertices.bin", scene->vertices, scene->vertexCount, error)
        || ! readRecords (dir + "/scene-indices.bin", scene->indices, scene->indexCount, error)
        || ! readRecords (dir + "/scene-objects.bin", scene->objects, scene->objectCount, error)
        || ! readRecords (dir + "/scene-materials.bin", scene->materials, scene->materialCount, error)
        || ! readRecords (dir + "/scene-cameras.bin", scene->cameras, scene->cameraCount, error))
    { std::printf ("orb replay: refused: %s\n", error.c_str()); return 1; }
    for (const auto& line : sceneText)
    {
        unsigned camera = 0;
        if (line.rfind (" ambient ", 0) == 0)
        {
            std::sscanf (line.c_str(), " ambient %f %f %f", &scene->ambientColor.x, &scene->ambientColor.y, &scene->ambientColor.z);
            if (after (line, "activeCamera", "%u", &camera)) scene->activeCamera = SceneCameraId { camera };
        }
        unsigned id = 0, kind = 0, shape = 0, visibility = 0, skyVersion = 0;
        int cameraVisible = 1;
        SceneLightRecord l;
        auto& t = l.transform;
        if (line.rfind (" light ", 0) == 0
            && std::sscanf (line.c_str(), " light %u kind %u", &id, &kind) == 2
            && after (line, "translation", "%f %f %f", &t.translation.x, &t.translation.y, &t.translation.z)
            && after (line, "rotation", "%f %f %f %f", &t.rotation.x, &t.rotation.y, &t.rotation.z, &t.rotation.w)
            && after (line, "color", "%f %f %f", &l.color.x, &l.color.y, &l.color.z)
            && after (line, "intensity", "%f", &l.intensity) && after (line, "range", "%f", &l.range)
            && after (line, "cone", "%f %f", &l.innerConeAngle, &l.outerConeAngle)
            && after (line, "radius", "%f", &l.radius) && after (line, "angularRadius", "%f", &l.angularRadius)
            && after (line, "area", "%f %f", &l.areaSizeX, &l.areaSizeY) && after (line, "shape", "%u", &shape)
            && after (line, "spread", "%f", &l.spread) && after (line, "visibility", "%u", &visibility)
            && after (line, "cameraVisible", "%d", &cameraVisible) && after (line, "sky", "%u", &skyVersion))
        {
            l.id = SceneLightId { id };
            l.kind = SceneLightKind (kind);
            l.areaShape = SceneAreaLightShape (shape);
            l.visibility = std::uint8_t (visibility);
            l.environmentCameraVisible = cameraVisible != 0;
            const auto i = std::to_string (lightTotal);
            if (! readBlock (dir + "/light-" + i + "-sky.bin", l.sky, error)
                || ! readBlock (dir + "/light-" + i + "-spectral.bin", l.spectralSource, error)
                || ! readBlock (dir + "/light-" + i + "-background.bin", l.cameraBackground, error))
            { std::printf ("orb replay: refused: %s\n", error.c_str()); return 1; }
            scene->lights.resize (lightTotal + 1);
            scene->lights[lightTotal++] = l;
        }
        if (line.find ("environmentImage version ") != std::string::npos
            && line.find ("environmentImage version 0 ") == std::string::npos)
        { std::printf ("orb replay: refused: a light uses an environment image, which the dump does not carry\n"); return 1; }
    }
    scene->lightCount = lightTotal;

    // The root program and its object programs, rebuilt from their dumped blocks.
    std::vector<std::shared_ptr<arbitgpu::NativeFixtureSurfaceMaterialProgram>> programs;
    std::shared_ptr<const diffractionmaterial::AdmittedLightingPlan> admission;
    for (std::size_t k = 0; k < programText.size(); ++k)
    {
        const auto& line = programText[k];
        if (line.rfind (" program ", 0) != 0) continue;
        std::size_t index = 0;
        unsigned object = 0, kind = 0, source = 0, authored = 0, paths = 0, bounce = 0, mask = 0;
        int reflectance = 0, layer = 0, lighting = 0;
        float coverage = 1.0f;
        if (std::sscanf (line.c_str(), " program %zu object %u kind %u baseColorSource %u authoredMaterial %u",
                         &index, &object, &kind, &source, &authored) != 5
            || ! after (line, "paths", "%u", &paths) || ! after (line, "bounce", "%u", &bounce)
            || ! after (line, "patternMask", "%u", &mask) || ! after (line, "coverage", "%f", &coverage)
            || ! after (line, "spectralReflectance", "%d", &reflectance)
            || ! after (line, "diffractionLayer", "%d", &layer) || ! after (line, "lightingAdmission", "%d", &lighting))
        { std::printf ("orb replay: refused: unreadable program line: %s\n", line.c_str()); return 1; }
        if (source != 0 || layer != 0)
        { std::printf ("orb replay: refused: program %zu uses a base colour texture or Stam layer, which the dump does not carry\n", index); return 1; }
        const bool root = programs.empty();
        auto p = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram> (
            root ? f.program : *f.program.objectPrograms[0]);
        p->objectPrograms.clear();
        p->object = SceneObjectId { object };
        p->kind = arbitgpu::NativeFixtureMaterialKind (kind);
        p->authoredMaterial = SceneMaterialId { authored };
        const auto prefix = dir + "/program-" + std::to_string (index) + "-";
        if (! readBlock (prefix + "parameters.bin", p->parameters, error)
            || ! readBlock (prefix + "paths.bin", p->diffractionPaths, error))
        { std::printf ("orb replay: refused: %s\n", error.c_str()); return 1; }
        p->diffractionPathCount = std::uint8_t (paths);
        p->diffractionMaximumBounceDepth = std::uint8_t (bounce);
        p->diffractionPatternMask = std::uint8_t (mask);
        p->diffractionMaskCoverage = coverage;
        if (mask == 0) p->diffractionProductDigest.clear();
        else if (p->diffractionProductDigest.empty()) p->diffractionProductDigest = "orb replay";
        p->spectralReflectance.reset();
        if (reflectance)
        {
            surfacematerial::SpectralReflectanceDescription rho;
            if (! readBlock (prefix + "reflectance.bin", rho, error)) { std::printf ("orb replay: refused: %s\n", error.c_str()); return 1; }
            p->spectralReflectance = rho;
        }
        if (lighting)
        {
            diffractionmaterial::LightingDescription description;
            if (! readBlock (prefix + "lighting.bin", description, error)) { std::printf ("orb replay: refused: %s\n", error.c_str()); return 1; }
            const auto admitted = diffractionmaterial::AdmittedLightingPlan::admit (description, error);
            if (! admitted) { std::printf ("orb replay: refused: dumped lighting: %s\n", error.c_str()); return 1; }
            p->diffractionLightingAdmission = std::make_shared<const diffractionmaterial::AdmittedLightingPlan> (*admitted);
            if (root) admission = p->diffractionLightingAdmission;
        }
        else if (! root) p->diffractionLightingAdmission.reset();
        programs.push_back (std::move (p));
    }
    if (programs.empty() || admission == nullptr)
    { std::printf ("orb replay: refused: program.txt names no root program with lighting\n"); return 1; }
    auto& root = *programs[0];
    for (std::size_t k = 1; k < programs.size(); ++k) root.objectPrograms.push_back (programs[k]);
    root.programIdentity = arbitgpu::nativeFixtureDiffractionProgramIdentity (root.bindingDigest, *admission,
        root.diffractionPatternMask, root.diffractionMaskCoverage, root.diffractionProductDigest, false, &root);
    const auto program = std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram> (programs[0]);

    auto& backend = arbitgpu::nativeFixtureSceneBackend();
    arbitgpu::realtimeengine::Scope scope;
    const std::shared_ptr<const Visual3DScene> snapshot = scene;
    const auto prepared = backend.prepare (snapshot, program);
    if (! prepared.prepared) { std::printf ("orb replay: refused: prepare: %s\n", prepared.error.c_str()); return 1; }
    arbitgpu::NativeFixtureSceneRuntimeInputs runtime;
    runtime.linearColor = linearColor != 0;
    runtime.emissionGain = emissionGain;
    const auto rendered = backend.render (snapshot, prepared.resources, width, height, runtime);
    if (! rendered.rendered || ! rendered.frame) { std::printf ("orb replay: refused: render: %s\n", rendered.error.c_str()); return 1; }
    GLint texture = 0;
    glGetIntegerv (GL_TEXTURE_BINDING_2D, &texture);
    glBindTexture (GL_TEXTURE_2D, static_cast<GLuint> (rendered.frame->colorImageHandle()));
    std::vector<float> pixels (std::size_t (width) * height * 4, 0.0f);
    glGetTexImage (GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, pixels.data());
    glBindTexture (GL_TEXTURE_2D, static_cast<GLuint> (texture));

    std::vector<char> live;
    const bool haveLive = readFile (dir + "/color.f32", live) && live.size() == pixels.size() * sizeof (float);
    std::vector<float> reference (pixels.size(), 0.0f);
    if (haveLive) std::memcpy (reference.data(), live.data(), live.size());
    // Both images in the same terms: mean RGB, the share of lit pixels, and the mean
    // per-pixel difference over the mean, on the same bottom-up rows.
    const auto stats = [&] (const std::vector<float>& image, double* mean, double& lit)
    {
        lit = 0;
        for (int c = 0; c < 3; ++c) mean[c] = 0;
        for (std::size_t q = 0; q < std::size_t (width) * height; ++q)
        {
            bool on = false;
            for (int c = 0; c < 3; ++c) { mean[c] += image[q * 4 + c]; on = on || image[q * 4 + c] > 0.0f; }
            lit += on;
        }
        for (int c = 0; c < 3; ++c) mean[c] /= double (width) * height;
        lit /= double (width) * height;
    };
    double replayMean[3], liveMean[3], replayLit = 0, liveLit = 0, difference = 0, magnitude = 0;
    stats (pixels, replayMean, replayLit);
    stats (reference, liveMean, liveLit);
    for (std::size_t q = 0; q < pixels.size(); ++q)
        if (q % 4 != 3) { difference += std::abs (pixels[q] - reference[q]); magnitude += std::abs (reference[q]); }
    std::printf ("orb replay %ux%u: programs %zu lights %zu; replay mean %.5g %.5g %.5g lit %.4f; "
                 "dumped mean %.5g %.5g %.5g lit %.4f; relative difference %.4f%s\n", width, height, programs.size(),
                 scene->lightCount, replayMean[0], replayMean[1], replayMean[2], replayLit, liveMean[0], liveMean[1],
                 liveMean[2], liveLit, magnitude > 0 ? difference / magnitude : 0.0, haveLive ? "" : " (no dumped colour)");
    if (const char* dump = std::getenv ("ARBIT_ORB_DUMP"))
    {
        std::FILE* file = std::fopen (dump, "wb");
        if (file == nullptr) return 1;
        std::fprintf (file, "%u %u\n", width, height);
        const bool written = std::fwrite (pixels.data(), sizeof (float), pixels.size(), file) == pixels.size();
        if (std::fclose (file) != 0 || ! written) return 1;
    }
    return 0;
}
} // namespace replay
} // namespace

int main()
{
    namingChecks();
    if (! glfwInit()) { std::fprintf (stderr, "SKIP: GLFW initialization failed\n"); return 77; }
    // An EGL context first, as the check is meant to run off-screen; the platform's
    // own context API is named when EGL is not there.
    glfwWindowHint (GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint (GLFW_CONTEXT_CREATION_API, GLFW_EGL_CONTEXT_API);
    glfwWindowHint (GLFW_CONTEXT_ROBUSTNESS, GLFW_LOSE_CONTEXT_ON_RESET);
    const char* api = "EGL";
    auto* window = glfwCreateWindow (64, 64, "realtime-receiver-transport", nullptr, nullptr);
    if (window == nullptr)
    {
        glfwWindowHint (GLFW_CONTEXT_CREATION_API, GLFW_NATIVE_CONTEXT_API);
        api = "native (EGL context creation failed)";
        window = glfwCreateWindow (64, 64, "realtime-receiver-transport", nullptr, nullptr);
    }
    if (window == nullptr)
    {
        std::fprintf (stderr, "SKIP: hidden OpenGL 3.3 core context creation failed\n");
        glfwTerminate();
        return 77;
    }
    glfwMakeContextCurrent (window);
    std::printf ("context: %s, %s, %s\n", api, reinterpret_cast<const char*> (glGetString (GL_RENDERER)),
                 reinterpret_cast<const char*> (glGetString (GL_VERSION)));
    if (const char* dir = std::getenv ("ARBIT_ORB_REPLAY"))
    {
        const int result = replay::run (dir);
        glfwDestroyWindow (window);
        glfwTerminate();
        return result;
    }
    if (const char* size = std::getenv ("ARBIT_ORB_MEASURE"))
    {
        const int result = orbMeasure (size);
        glfwDestroyWindow (window);
        glfwTerminate();
        return result;
    }
    variantChecks();
    frameChecks (arbitgpu::nativeFixtureSceneBackend());
    submissionChecks (arbitgpu::nativeFixtureSceneBackend());
    orbChecks (arbitgpu::nativeFixtureSceneBackend());
    liveOrbChecks (arbitgpu::nativeFixtureSceneBackend());
    glfwDestroyWindow (window);
    glfwTerminate();
    if (failures != 0)
    {
        std::printf ("realtime receiver transport gl: %d failures\n", failures);
        return 1;
    }
    std::puts ("realtime receiver transport gl: ok");
    return 0;
}
