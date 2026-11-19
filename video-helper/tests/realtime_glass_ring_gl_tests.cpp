// Realtime's thick glass and planar mirror (fixture_realtime_opengl.h, realtime_planar_mirror.h).
//
// The CPU half checks the plane fit the planar mirror uses: a flat quad gives its plane
// facing the viewer, a bent surface or a line gives none, and a point's mirror image
// lies as far behind the plane as the point is in front.
//
// The GPU half draws the Glass Ring Caustics starter (BuiltInVisualModules.h) in
// Realtime: a closed dispersive glass torus, circle radius 0.5 through a 24-sided tube of
// radius 0.1 at IOR 1.5 and dispersion 0.15, standing on a violet floor that sweeps up
// behind it, under the starter's analytic sky and key, from the starter's camera. Its
// Color goes to ARBIT_RING_DUMP ("W H\n" then W*H*4 floats) and its GPU time, the median
// of ARBIT_RING_FRAMES renders (default 9), is printed. ARBIT_RING_SIZE=WxH sets the
// frame (default 674x380).
//
// Build: see /tmp/p234-frames/build/rtgaps/compile.sh with this file as the main unit.

#include "gpu_backend/backend.h"
#include "gpu_backend/fixture_realtime_opengl.h"
#if __has_include("gpu_backend/realtime_planar_mirror.h")
#include "gpu_backend/realtime_planar_mirror.h"
#define RTGAPS_HAVE_PLANAR_MIRROR 1
#endif

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#define GL_GLEXT_PROTOTYPES 1
#include <GL/gl.h>
#include <GL/glext.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace
{
using namespace HarmonicMIDI::grid;
int failures = 0;
void check (bool condition, const std::string& text)
{
    if (condition) return;
    ++failures;
    std::fprintf (stderr, "FAIL: %s\n", text.c_str());
}

#if RTGAPS_HAVE_PLANAR_MIRROR
void planeChecks()
{
    namespace planar = arbitgpu::realtimeplanarmirror;
    // The card's floor: a quad at y -0.0445 seen from above.
    const std::vector<planar::Point> quad { { -2, -0.0445f, -2 }, { 2, -0.0445f, -2 }, { 2, -0.0445f, 2 }, { -2, -0.0445f, 2 } };
    const auto floor = planar::plane (quad, { 0, 1, 3 });
    check (floor && std::abs ((*floor)[1] - 1.0f) < 1e-6f && std::abs ((*floor)[3] - 0.0445f) < 1e-6f,
           "a flat quad gives its plane, facing the viewer above it");
    const auto under = planar::plane (quad, { 0, -1, 3 });
    check (under && std::abs ((*under)[1] + 1.0f) < 1e-6f, "the plane faces a viewer below it");
    auto bent = quad;
    bent[2][1] += 0.01f;
    check (! planar::plane (bent, { 0, 1, 3 }), "a bent quad gives no plane");
    check (! planar::plane ({ { 0, 0, 0 }, { 1, 0, 0 }, { 2, 0, 0 } }, { 0, 1, 0 }), "points on a line give no plane");
    check (! planar::plane ({ { 0, 0, 0 }, { 1, 0, 0 } }, { 0, 1, 0 }), "two points give no plane");
    if (floor)
    {
        const planar::Point p { 0.3f, 0.5f, -0.2f };
        const auto image = planar::reflect (*floor, p);
        check (std::abs (planar::distance (*floor, image) + planar::distance (*floor, p)) < 1e-6f
                   && std::abs (image[0] - p[0]) < 1e-6f && std::abs (image[2] - p[2]) < 1e-6f,
               "the mirror image lies as far behind the plane as the point is in front");
        check (planar::same (*floor, *floor, 1e-6f) && ! planar::same (*floor, *under, 1e-3f),
               "a plane matches itself and not its reverse");
    }
}
#endif

SceneQuaternion aboutX (float degrees)
{
    const float half = 0.5f * degrees * 3.14159265358979f / 180.0f;
    return { std::sin (half), 0.0f, 0.0f, std::cos (half) };
}

struct Ring
{
    std::shared_ptr<Visual3DScene> scene;
    std::shared_ptr<arbitgpu::NativeFixtureSurfaceMaterialProgram> program;
};

// The starter's ring, floor, sky, key and camera.
Ring makeRing()
{
    Ring ring;
    ring.scene = std::make_shared<Visual3DScene>();
    auto& scene = *ring.scene;
    scene.id = { 1 };
    constexpr unsigned points = 96, sides = 24, columns = 64;
    constexpr float radius = 0.5f, tube = 0.1f, sweep = 1.6f, pi = 3.14159265358979f;
    Visual3DSceneGrowth growth;
    growth.vertices = std::size_t (points + 1) * (sides + 1) + 4 + 2 * (columns + 1);
    growth.indices = std::size_t (points) * sides * 6 + 6 + 6 * columns;
    std::string error;
    check (growVisual3DScene (scene, growth, error), "scene grows for the ring: " + error);
    const auto addObject = [&] (std::uint32_t id, std::uint32_t material)
    {
        auto& object = scene.objects[scene.objectCount++];
        object = {};
        object.id = { id };
        object.material = { material };
        object.firstVertex = std::uint32_t (scene.vertexCount);
        object.firstIndex = std::uint32_t (scene.indexCount);
        return scene.objectCount - 1;
    };
    const auto vertex = [&] (SceneVec3 p, SceneVec3 n, SceneVec2 uv)
    {
        auto& v = scene.vertices[scene.vertexCount++];
        v = {};
        v.position = p;
        v.normal = n;
        v.uv = uv;
    };
    const auto close = [&] (std::size_t o)
    {
        auto& object = scene.objects[o];
        object.vertexCount = std::uint32_t (scene.vertexCount - object.firstVertex);
        object.indexCount = std::uint32_t (scene.indexCount - object.firstIndex);
    };
    const auto triangle = [&] (std::size_t o, std::uint32_t a, std::uint32_t b, std::uint32_t c)
    {
        const auto base = scene.objects[o].firstVertex;
        scene.indices[scene.indexCount++] = a - base;
        scene.indices[scene.indexCount++] = b - base;
        scene.indices[scene.indexCount++] = c - base;
    };

    // The ring stands in the XY plane, its lowest point on the floor.
    const auto ringObject = addObject (101, 6);
    scene.objects[ringObject].transform.translation = { 0.0f, 0.6f, 0.0f };
    const auto first = std::uint32_t (scene.vertexCount);
    for (unsigned i = 0; i <= points; ++i)
        for (unsigned j = 0; j <= sides; ++j)
        {
            const float u = 2 * pi * float (i) / points, v = 2 * pi * float (j) / sides;
            const SceneVec3 n { std::cos (v) * std::cos (u), std::cos (v) * std::sin (u), std::sin (v) };
            vertex ({ radius * std::cos (u) + tube * n.x, radius * std::sin (u) + tube * n.y, tube * n.z }, n,
                    { float (i) / points, float (j) / sides });
        }
    const auto at = [&] (unsigned i, unsigned j) { return first + i * (sides + 1) + j; };
    for (unsigned i = 0; i < points; ++i)
        for (unsigned j = 0; j < sides; ++j)
        {
            triangle (ringObject, at (i, j), at (i + 1, j), at (i + 1, j + 1));
            triangle (ringObject, at (i, j), at (i + 1, j + 1), at (i, j + 1));
        }
    close (ringObject);

    // The floor: a plane in front, a quarter-turn sweep of radius 1.6 behind. As in the
    // starter, the sweep is cut from a cylinder, so its normals and front faces point out
    // of the cylinder, away from the camera, and the floor material is single sided.
    const auto floorObject = addObject (102, 7);
    const auto plane = std::uint32_t (scene.vertexCount);
    vertex ({ -4, 0, 4 }, { 0, 1, 0 }, { 0, 0 });
    vertex ({ 4, 0, 4 }, { 0, 1, 0 }, { 1, 0 });
    vertex ({ 4, 0, 0 }, { 0, 1, 0 }, { 1, 1 });
    vertex ({ -4, 0, 0 }, { 0, 1, 0 }, { 0, 1 });
    triangle (floorObject, plane, plane + 1, plane + 2);
    triangle (floorObject, plane, plane + 2, plane + 3);
    const auto arc = std::uint32_t (scene.vertexCount);
    for (unsigned c = 0; c <= columns; ++c)
    {
        const float phi = 0.25f * pi * float (c) / columns;
        const SceneVec3 n { 0, -std::cos (phi), -std::sin (phi) };
        vertex ({ -4, sweep - sweep * std::cos (phi), -sweep * std::sin (phi) }, n, { 0, float (c) / columns });
        vertex ({ 4, sweep - sweep * std::cos (phi), -sweep * std::sin (phi) }, n, { 1, float (c) / columns });
    }
    for (unsigned c = 0; c < columns; ++c)
    {
        const auto a = arc + 2 * c;
        triangle (floorObject, a, a + 3, a + 1);
        triangle (floorObject, a, a + 2, a + 3);
    }
    close (floorObject);

    auto& glass = scene.materials[scene.materialCount++];
    glass = {};
    glass.id = { 6 };
    glass.roughness = 0.0f;
    auto& floor = scene.materials[scene.materialCount++];
    floor = {};
    floor.id = { 7 };
    floor.baseColor = { 0.07f, 0.05f, 0.19f };

    scene.cameraCount = 1;
    auto& camera = scene.cameras[0];
    camera = {};
    camera.id = { 401 };
    camera.transform.translation = { 0.0f, 0.95f, 2.1f };
    camera.transform.rotation = aboutX (-14.6f);
    camera.verticalFovRadians = 52.0f * pi / 180.0f;
    camera.nearPlane = 0.1f;
    camera.farPlane = 100.0f;
    scene.activeCamera = camera.id;

    scene.lightCount = 2;
    auto& key = scene.lights[0];
    key = {};
    key.id = { 301 };
    key.kind = SceneLightKind::Directional;
    key.transform.rotation = aboutX (-120.0f);
    key.color = { 1.0f, 0.9f, 0.75f };
    key.intensity = 3.0f;
    auto& sky = scene.lights[1];
    sky = {};
    sky.id = { 302 };
    sky.kind = SceneLightKind::Environment;
    sky.color = { 0.5f, 0.5f, 0.5f };
    sky.intensity = 1.0f;
    sky.sky.version = 1;
    sky.sky.zenith = { 0.1f, 0.7f, 2.0f };
    sky.sky.horizon = { 2.0f, 0.24f, 0.9f };
    sky.sky.ground = { 0.04f, 1.1f, 1.4f };
    sky.sky.sunColor = { 1, 1, 1 };
    sky.sky.sunAzimuthDegrees = -140.0f;
    sky.sky.sunElevationDegrees = 8.0f;
    sky.sky.sunRadiusDegrees = 30.0f;
    sky.sky.sunIntensity = 3.0f;

    // One program per object, as a lowered surface material gives: the ring closed
    // dispersive glass, the floor a constant diffuse violet.
    const auto program = [&] (std::size_t o)
    {
        auto p = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>();
        p->backend = arbitgpu::NativeFixtureMaterialBackend::OpenGl;
        p->object = scene.objects[o].id;
        p->bindingDigest = "glass-ring-binding-" + std::to_string (o);
        p->programIdentity = "glass-ring-program-" + std::to_string (o);
        p->authoredScene = scene.id;
        p->authoredMaterial = scene.objects[o].material;
        p->sceneRevision = p->structuralRevision = p->evaluationRevision = p->materialRevision = 1;
        p->parameters.identifiers[0] = p->authoredMaterial.value;
        p->parameters.normalOpacity = { 0, 0, 1, 1 };
        return p;
    };
    ring.program = program (ringObject);
    ring.program->parameters.identifiers[1] = std::uint32_t (surfacematerial::TransportMode::ClosedVolume);
    ring.program->parameters.baseColorMetallic = { 1, 1, 1, 0 };
    ring.program->parameters.transmissionIorClearcoat = { 1, 1.5f, 0, 0.15f };
    ring.program->parameters.emissionRoughness = { 0, 0, 0, 0 };
    auto floorProgram = program (floorObject);
    floorProgram->parameters.baseColorMetallic = { 0.07f, 0.05f, 0.19f, 0 };
    floorProgram->parameters.transmissionIorClearcoat = { 0, 1.5f, 0, 0 };
    floorProgram->parameters.emissionRoughness = { 0, 0, 0, 1 };
    ring.program->objectPrograms.push_back (floorProgram);
    return ring;
}

int ringFrame()
{
    unsigned width = 674, height = 380, frames = 9;
    if (const char* size = std::getenv ("ARBIT_RING_SIZE")) std::sscanf (size, "%ux%u", &width, &height);
    if (const char* count = std::getenv ("ARBIT_RING_FRAMES")) frames = std::max (1, std::atoi (count));
    const auto ring = makeRing();
    if (failures != 0) return 1;
    auto& backend = arbitgpu::nativeFixtureSceneBackend();
    arbitgpu::realtimeengine::Scope scope;
    const std::shared_ptr<const Visual3DScene> snapshot = ring.scene;
    const std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram> program = ring.program;
    const auto prepared = backend.prepare (snapshot, program);
    if (! prepared.prepared) { std::printf ("glass ring: refused: prepare: %s\n", prepared.error.c_str()); return 1; }
    arbitgpu::NativeFixtureSceneRuntimeInputs runtime;
    runtime.linearColor = true;
    std::vector<double> times;
    std::vector<float> pixels (std::size_t (width) * height * 4, 0.0f);
    GLuint query = 0;
    glGenQueries (1, &query);
    for (unsigned frame = 0; frame <= frames; ++frame)
    {
        glFinish();
        glBeginQuery (GL_TIME_ELAPSED, query);
        auto rendered = backend.render (snapshot, prepared.resources, width, height, runtime);
        glEndQuery (GL_TIME_ELAPSED);
        if (! rendered.rendered || ! rendered.frame)
        { std::printf ("glass ring: refused: render: %s\n", rendered.error.c_str()); return 1; }
        if (! rendered.stats.realtimeProgramApplied) { std::printf ("glass ring: the Realtime program was not applied\n"); return 1; }
        GLuint64 elapsed = 0;
        glGetQueryObjectui64v (query, GL_QUERY_RESULT, &elapsed);
        // The first frame compiles and uploads; the rest are timed.
        if (frame != 0) times.push_back (double (elapsed) * 1e-6);
        if (frame == frames)
        {
            GLint texture = 0;
            glGetIntegerv (GL_TEXTURE_BINDING_2D, &texture);
            glBindTexture (GL_TEXTURE_2D, GLuint (rendered.frame->colorImageHandle()));
            glGetTexImage (GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, pixels.data());
            glBindTexture (GL_TEXTURE_2D, GLuint (texture));
        }
    }
    glDeleteQueries (1, &query);
    std::sort (times.begin(), times.end());
    double mean[3] {};
    std::size_t finite = 0;
    for (std::size_t q = 0; q < std::size_t (width) * height; ++q)
    {
        bool ok = true;
        for (int c = 0; c < 3; ++c) { ok = ok && std::isfinite (pixels[q * 4 + c]); mean[c] += pixels[q * 4 + c]; }
        finite += ok;
    }
    for (auto& m : mean) m /= double (width) * height;
    check (finite == std::size_t (width) * height, "every ring pixel is finite");
    // Path Trace shows the floor sweep up to 0.39 of the frame from the top, so at mid
    // height left of the ring the pixel is violet floor, not the pink sky behind it.
    const auto side = (std::size_t (height / 2) * width + width / 10) * 4;
    check (pixels[side + 2] > 1.5f * pixels[side], "the floor sweep shows left of the ring at mid height");
    std::printf ("glass ring %ux%u: GPU median %.3f ms (min %.3f, max %.3f, %zu frames); mean %.5g %.5g %.5g\n",
                 width, height, times[times.size() / 2], times.front(), times.back(), times.size(), mean[0], mean[1], mean[2]);
    if (const char* dump = std::getenv ("ARBIT_RING_DUMP"))
    {
        std::FILE* file = std::fopen (dump, "wb");
        if (file == nullptr) return 1;
        std::fprintf (file, "%u %u\n", width, height);
        const bool written = std::fwrite (pixels.data(), sizeof (float), pixels.size(), file) == pixels.size();
        if (std::fclose (file) != 0 || ! written) return 1;
    }
    return 0;
}
} // namespace

int main()
{
#if RTGAPS_HAVE_PLANAR_MIRROR
    planeChecks();
#endif
    if (! glfwInit()) { std::fprintf (stderr, "SKIP: GLFW initialization failed\n"); return 77; }
    glfwWindowHint (GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint (GLFW_CONTEXT_CREATION_API, GLFW_EGL_CONTEXT_API);
    auto* window = glfwCreateWindow (64, 64, "realtime-glass-ring", nullptr, nullptr);
    if (window == nullptr)
    {
        glfwWindowHint (GLFW_CONTEXT_CREATION_API, GLFW_NATIVE_CONTEXT_API);
        window = glfwCreateWindow (64, 64, "realtime-glass-ring", nullptr, nullptr);
    }
    if (window == nullptr)
    {
        std::fprintf (stderr, "SKIP: hidden OpenGL 3.3 core context creation failed\n");
        glfwTerminate();
        return 77;
    }
    glfwMakeContextCurrent (window);
    std::printf ("context: %s, %s\n", reinterpret_cast<const char*> (glGetString (GL_RENDERER)),
                 reinterpret_cast<const char*> (glGetString (GL_VERSION)));
    const int result = ringFrame();
    glfwDestroyWindow (window);
    glfwTerminate();
    if (result != 0 || failures != 0)
    {
        std::printf ("realtime glass ring gl: %d failures\n", failures);
        return 1;
    }
    std::puts ("realtime glass ring gl: ok");
    return 0;
}
