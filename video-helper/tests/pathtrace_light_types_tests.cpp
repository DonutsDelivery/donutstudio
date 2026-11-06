// Path Trace light types: rect, disk and ellipse area lights sample with the
// density their hits report, spread narrows a panel without changing its
// power, a scene Area light keeps a point light's on-axis intensity, ray
// visibility masks camera and shadow rays, and the scene wire carries the new
// fields while a scene without them encodes as before.

#include "pathtrace/integrator.h"
#include "pathtrace/lights.h"
#include "pathtrace/sampler.h"
#include "pathtrace/scene_ingest.h"
#include "SceneAovOperationContract.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace
{

namespace pt = videohelper::pathtrace;
namespace grid = HarmonicMIDI::grid;
using pt::Vec3;

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

std::string describe (double value) { return std::to_string (value); }

bool closeRelative (double a, double b, double tolerance)
{
    return std::fabs (a - b) <= tolerance * std::max (std::fabs (b), 1.0e-9);
}

// Turns local -Z to -Y: a planar light shining down, a camera looking down.
const pt::Quat kDown { -0.70710678f, 0.0f, 0.0f, 0.70710678f };

// Light sampling estimate of the irradiance at p on a surface facing up, and
// the worst disagreement between a sample's pdf and radiance and what
// analyticHit reports for the same direction.
double sampledIrradiance (const pt::LightSet& lights, Vec3 p, int n, double& worstMismatch)
{
    double sum = 0.0;
    worstMismatch = 0.0;
    for (int i = 0; i < n; ++i)
    {
        const float u = (static_cast<float> (i) + 0.5f) / static_cast<float> (n);
        const float radical = pt::sampling::toUnitFloat (pt::sampling::reverseBits (static_cast<std::uint32_t> (i)));
        const pt::Vec2 v { u, (radical * static_cast<float> (n - 1) + 0.5f) / static_cast<float> (n) };
        pt::LightSample s;
        if (! lights.sample (p, 0.5f, v, s) || s.delta)
            continue;
        sum += static_cast<double> (s.radiance.x) * std::max (0.0f, s.wi.y) / static_cast<double> (s.pdf);
        Vec3 radiance;
        float pdf = 0.0f;
        float distance = 0.0f;
        const bool hit = lights.analyticHit (0, p, s.wi, std::numeric_limits<float>::infinity(), radiance, pdf, distance);
        const double mismatch = hit ? std::fabs (static_cast<double> (pdf / s.pdf) - 1.0) + std::fabs (radiance.x - s.radiance.x)
                                    : 1.0;
        worstMismatch = std::max (worstMismatch, mismatch);
    }
    return sum / n;
}

// Brute-force irradiance at p (facing up) from a planar light of radiance 1
// spanning centre +- axisX +- axisY, clipped to the inscribed ellipse when
// round, with the spread attenuation applied.
double referenceIrradiance (Vec3 p, Vec3 centre, Vec3 axisX, Vec3 axisY, bool round, float spread)
{
    const float tanHalf = std::tan (0.5f * spread);
    const float normalization = grid::sceneAreaSpreadNormalization (spread);
    const int n = 500;
    double sum = 0.0;
    const Vec3 normal = pt::normalizeOrZero (pt::cross (axisX, axisY));
    const double cellArea = static_cast<double> (pt::length (axisX) * pt::length (axisY)) * 4.0 / (n * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            const float a = -1.0f + (2.0f * static_cast<float> (i) + 1.0f) / n;
            const float b = -1.0f + (2.0f * static_cast<float> (j) + 1.0f) / n;
            if (round && a * a + b * b > 1.0f)
                continue;
            const Vec3 d = centre + axisX * a + axisY * b - p;
            const Vec3 w = pt::normalizeOrZero (d);
            const float cosAtLight = std::fabs (pt::dot (normal, w));
            sum += std::max (0.0f, w.y) * cosAtLight * pt::areaSpreadAttenuation (tanHalf, normalization, cosAtLight)
                   / pt::dot (d, d) * cellArea;
        }
    return sum;
}

void areaShapeTests()
{
    const pt::IngestedScene empty {};
    const std::vector<pt::Material> noMaterials {};
    const pt::Material fallback {};
    struct Case { pt::AreaLightShape shape; const char* name; float sizeY; };
    for (const Case shape : { Case { pt::AreaLightShape::Rect, "rect", 0.6f }, Case { pt::AreaLightShape::Disk, "disk", 1.0f },
                              Case { pt::AreaLightShape::Ellipse, "ellipse", 0.4f } })
        for (const float spread : { pt::kPi, 1.2f })
            for (const Vec3 p : { Vec3 { 0.0f, 0.0f, 0.0f }, Vec3 { 0.7f, 0.0f, 0.2f } })
            {
                pt::AreaLightRecord area;
                area.shape = shape.shape;
                area.position = { 0.0f, 1.0f, 0.0f };
                area.rotation = kDown;
                area.sizeX = 1.0f;
                area.sizeY = shape.sizeY;
                area.spread = spread;
                const std::vector<pt::AreaLightRecord> areas { area };
                pt::LightSet lights;
                lights.build (nullptr, 0, {}, empty, noMaterials, fallback, nullptr, nullptr, &areas);
                const std::string name = std::string (shape.name) + " spread " + describe (spread) + " at x " + describe (p.x);
                check (lights.analyticCount() == 1 && lights.stats().areaLights == 1, name + ": one area light");
                const Vec3 axisX = pt::rotate (kDown, { 0.5f, 0.0f, 0.0f });
                const Vec3 axisY = pt::rotate (kDown, { 0.0f, 0.5f * shape.sizeY, 0.0f });
                const double expected = referenceIrradiance (p, area.position, axisX, axisY,
                                                             shape.shape != pt::AreaLightShape::Rect, spread);
                double mismatch = 0.0;
                const double e = sampledIrradiance (lights, p, 8192, mismatch);
                check (closeRelative (e, expected, 0.02), name + ": irradiance " + describe (e) + " vs " + describe (expected));
                check (mismatch < 2.0e-3, name + ": a sampled direction hits the light with the same pdf and radiance");
            }
}

void spreadTests()
{
    // Spread keeps a Lambertian panel's power: the integral of the attenuation
    // times cos over the hemisphere stays pi.
    for (const float spread : { 2.5f, 1.0f, 0.3f, 0.05f })
    {
        const float tanHalf = std::tan (0.5f * spread);
        const float normalization = grid::sceneAreaSpreadNormalization (spread);
        const int n = 200000;
        double sum = 0.0;
        for (int i = 0; i < n; ++i)
        {
            const double theta = 0.5 * pt::kPi * (i + 0.5) / n;
            sum += pt::areaSpreadAttenuation (tanHalf, normalization, static_cast<float> (std::cos (theta)))
                   * std::cos (theta) * std::sin (theta);
        }
        const double power = 2.0 * pt::kPi * sum * 0.5 * pt::kPi / n;
        check (closeRelative (power, pt::kPi, 0.01), "spread " + describe (spread) + " keeps power: " + describe (power));
        check (pt::areaSpreadAttenuation (tanHalf, normalization, std::cos (0.5f * spread + 0.01f)) == 0.0f,
               "spread " + describe (spread) + " emits nothing outside its cone");
    }
    check (grid::sceneAreaSpreadNormalization (pt::kPi) == 0.0f
               && pt::areaSpreadAttenuation (0.0f, 0.0f, 0.2f) == 1.0f,
           "a full spread is a plain Lambertian panel");

    // A narrow panel lights the floor under it harder and the side less.
    const pt::IngestedScene empty {};
    const std::vector<pt::Material> noMaterials {};
    const pt::Material fallback {};
    const auto irradiance = [&] (float spread, Vec3 p) {
        pt::AreaLightRecord area;
        area.position = { 0.0f, 2.0f, 0.0f };
        area.rotation = kDown;
        area.sizeX = area.sizeY = 0.5f;
        area.spread = spread;
        const std::vector<pt::AreaLightRecord> areas { area };
        pt::LightSet lights;
        lights.build (nullptr, 0, {}, empty, noMaterials, fallback, nullptr, nullptr, &areas);
        double mismatch = 0.0;
        return sampledIrradiance (lights, p, 4096, mismatch);
    };
    check (irradiance (0.4f, {}) > 3.0 * irradiance (pt::kPi, {}), "a narrow spread brightens the spot under it");
    check (irradiance (0.4f, { 3.0f, 0.0f, 0.0f }) == 0.0 && irradiance (pt::kPi, { 3.0f, 0.0f, 0.0f }) > 0.0,
           "a narrow spread leaves the side dark");
}

grid::SceneLightRecord areaRecord (grid::SceneAreaLightShape shape, float sizeX, float sizeY, float intensity)
{
    grid::SceneLightRecord light;
    light.id = grid::SceneLightId { 7 };
    light.kind = grid::SceneLightKind::Area;
    light.transform.rotation = { kDown.x, kDown.y, kDown.z, kDown.w };
    light.transform.translation = { 0.0f, 4.0f, 0.0f };
    light.intensity = intensity;
    light.areaShape = shape;
    light.areaSizeX = sizeX;
    light.areaSizeY = sizeY;
    return light;
}

void sceneAreaRecordTests()
{
    const pt::IngestedScene empty {};
    const std::vector<pt::Material> noMaterials {};
    const pt::Material fallback {};
    // A small panel of on-axis intensity I gives I / d^2 straight below, as a point light would.
    for (const auto shape : { grid::SceneAreaLightShape::Square, grid::SceneAreaLightShape::Rectangle,
                              grid::SceneAreaLightShape::Disk, grid::SceneAreaLightShape::Ellipse })
    {
        const std::vector<grid::SceneLightRecord> records { areaRecord (shape, 0.04f, 0.02f, 8.0f) };
        pt::LightSet lights;
        lights.build (records.data(), records.size(), {}, empty, noMaterials, fallback);
        check (lights.stats().areaLights == 1, "scene area record " + std::to_string (static_cast<int> (shape)) + " builds a panel");
        double mismatch = 0.0;
        const double e = sampledIrradiance (lights, {}, 2048, mismatch);
        check (closeRelative (e, 8.0 / 16.0, 0.01), "scene area record " + std::to_string (static_cast<int> (shape))
                                                         + " keeps on-axis intensity: " + describe (e));
    }
}

// ------------------------------------------------------------ visibility

pt::Material diffuse (Vec3 colour)
{
    pt::Material m;
    m.baseColor = colour;
    m.roughness = 1.0f;
    return m;
}

void appendQuad (pt::RenderScene& scene, Vec3 centre, Vec3 halfU, Vec3 halfV, std::uint32_t material, std::uint8_t visibility)
{
    const Vec3 c[4] { centre - halfU - halfV, centre + halfU - halfV, centre + halfU + halfV, centre - halfU + halfV };
    const Vec3 n = pt::normalizeOrZero (pt::cross (halfU, halfV));
    for (const auto& tri : { std::array<int, 3> { 0, 1, 2 }, std::array<int, 3> { 0, 2, 3 } })
    {
        pt::TriangleAttributes attributes;
        attributes.materialIndex = material;
        attributes.primitiveIndex = static_cast<std::uint32_t> (scene.geometry.triangles.size());
        attributes.geometricNormal = n;
        attributes.shadingNormals[0] = attributes.shadingNormals[1] = attributes.shadingNormals[2] = n;
        attributes.visibility = visibility;
        for (const int k : tri)
        {
            scene.geometry.positions.push_back (c[k]);
            scene.geometry.bounds.grow (c[k]);
        }
        scene.geometry.triangles.push_back (attributes);
    }
}

pt::PinholeCamera camera (Vec3 position, pt::Quat rotation)
{
    pt::PinholeCamera result;
    result.position = position;
    result.rotation = rotation;
    result.tanHalfFov = 1.0f;
    result.aspect = 1.0f;
    result.nearPlane = 0.01f;
    result.farPlane = 1000.0f;
    result.width = 32;
    result.height = 32;
    return result;
}

// Mean of a few rows through the middle at column x.
double column (const pt::PathTracer& tracer, std::uint32_t x)
{
    double sum = 0.0;
    for (std::uint32_t y = 14; y < 18; ++y)
    {
        const pt::Rgb v = tracer.accumulator().mean (x, y);
        sum += (v.r + v.g + v.b) / 3.0;
    }
    return sum / 4.0;
}

// A black blocker at height 2 over a grey floor, lit by a point light off to
// the side, seen from straight above. Column 16 looks at the blocker, column 9
// at its shadow on the floor, column 23 at lit floor.
std::array<double, 3> blockerColumns (std::uint8_t blockerVisibility, std::uint8_t lightVisibility)
{
    pt::RenderScene scene;
    scene.materials = { diffuse ({ 0.8f, 0.8f, 0.8f }), diffuse ({ 0.0f, 0.0f, 0.0f }) };
    appendQuad (scene, {}, { 4.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, -4.0f }, 0, grid::sceneRayVisibility::All);
    appendQuad (scene, { 0.0f, 2.0f, 0.0f }, { 1.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, -1.0f }, 1, blockerVisibility);
    grid::SceneLightRecord light;
    light.kind = grid::SceneLightKind::Point;
    light.transform.translation = { 3.0f, 4.0f, 0.0f };
    light.intensity = 30.0f;
    light.visibility = lightVisibility;
    scene.lightRecords = { light };
    scene.commit();
    pt::PathTracer tracer (2);
    tracer.settings.maxDepth = 1;
    const pt::CancellationToken cancel {};
    const auto view = camera ({ 0.0f, 6.0f, 0.0f }, kDown);
    tracer.renderFixed (scene, view, view.width, view.height, 8, 1, cancel);
    return { column (tracer, 16), column (tracer, 9), column (tracer, 23) };
}

void visibilityTests()
{
    using namespace grid::sceneRayVisibility;
    const auto all = blockerColumns (All, All);
    check (all[0] < 1.0e-6 && all[1] < 1.0e-6 && all[2] > 0.05, "a visible black blocker shows and casts its shadow");

    const auto noCamera = blockerColumns (static_cast<std::uint8_t> (All & ~Camera), All);
    check (noCamera[0] > 0.05, "a camera-hidden blocker shows the floor behind it: " + describe (noCamera[0]));
    check (noCamera[1] < 1.0e-6, "a camera-hidden blocker still casts its shadow");

    const auto noShadow = blockerColumns (static_cast<std::uint8_t> (All & ~Shadow), All);
    check (noShadow[0] < 1.0e-6, "a shadow-hidden blocker is still seen");
    check (noShadow[1] > 0.05, "a shadow-hidden blocker casts no shadow: " + describe (noShadow[1]));

    const auto lightNoShadow = blockerColumns (All, static_cast<std::uint8_t> (All & ~Shadow));
    check (lightNoShadow[1] > 0.05, "a light without shadows lights the floor behind the blocker");

    const auto lightNoDiffuse = blockerColumns (All, static_cast<std::uint8_t> (All & ~Diffuse));
    check (lightNoDiffuse[2] < 1.0e-6, "a light hidden from diffuse rays leaves a diffuse floor dark");

    // An area light facing the camera shows in the frame unless hidden from camera rays.
    const auto facing = [] (std::uint8_t visibility) {
        pt::RenderScene scene;
        scene.materials = { diffuse ({ 0.0f, 0.0f, 0.0f }) };
        appendQuad (scene, { 0.0f, 0.0f, -10.0f }, { 20.0f, 0.0f, 0.0f }, { 0.0f, 20.0f, 0.0f }, 0, grid::sceneRayVisibility::All);
        auto light = areaRecord (grid::SceneAreaLightShape::Disk, 1.0f, 1.0f, 2.0f);
        light.transform.translation = { 0.0f, 0.0f, -3.0f };
        light.transform.rotation = { 0.0f, 1.0f, 0.0f, 0.0f };
        light.visibility = visibility;
        scene.lightRecords = { light };
        scene.commit();
        pt::PathTracer tracer (2);
        tracer.settings.maxDepth = 1;
        const pt::CancellationToken cancel {};
        const auto view = camera ({}, pt::Quat {});
        tracer.renderFixed (scene, view, view.width, view.height, 4, 1, cancel);
        return column (tracer, 16);
    };
    check (facing (All) > 0.1, "an area light is seen by the camera");
    check (facing (static_cast<std::uint8_t> (All & ~Camera)) == 0.0, "a camera-hidden area light is not");
}

// ------------------------------------------------------------ wire

grid::Visual3DScene wireScene()
{
    grid::Visual3DScene scene;
    scene.id = grid::Scene3DId { 1 };
    scene.vertices[0] = { { -0.5f, -0.5f, 0.0f }, { 0.0f, 0.0f, 1.0f }, { 0.0f, 1.0f } };
    scene.vertices[1] = { { 0.5f, -0.5f, 0.0f }, { 0.0f, 0.0f, 1.0f }, { 1.0f, 1.0f } };
    scene.vertices[2] = { { 0.0f, 0.5f, 0.0f }, { 0.0f, 0.0f, 1.0f }, { 0.5f, 0.0f } };
    scene.indices[0] = 0;
    scene.indices[1] = 1;
    scene.indices[2] = 2;
    scene.vertexCount = scene.indexCount = 3;
    scene.materials[0].id = grid::SceneMaterialId { 2 };
    scene.materialCount = 1;
    auto& object = scene.objects[0];
    object.id = grid::SceneObjectId { 3 };
    object.material = scene.materials[0].id;
    object.vertexCount = object.indexCount = 3;
    scene.objectCount = 1;
    scene.cameras[0].id = grid::SceneCameraId { 4 };
    scene.cameraCount = 1;
    scene.activeCamera = scene.cameras[0].id;
    grid::SceneLightRecord light;
    light.id = grid::SceneLightId { 5 };
    light.kind = grid::SceneLightKind::Point;
    light.intensity = 1.0f;
    scene.lights[0] = light;
    scene.lightCount = 1;
    return scene;
}

void wireTests()
{
    const auto plain = wireScene();
    check (grid::validateVisual3DScene (plain).valid(), "the wire scene is valid");
    const auto plainBytes = sceneaov::detail::encodeScene (plain);

    auto sized = plain;
    sized.lights[0] = areaRecord (grid::SceneAreaLightShape::Ellipse, 2.0f, 0.5f, 3.0f);
    sized.lights[0].spread = 0.7f;
    sized.lights[0].visibility = static_cast<std::uint8_t> (grid::sceneRayVisibility::All & ~grid::sceneRayVisibility::Glossy);
    sized.objects[0].visibility = static_cast<std::uint8_t> (grid::sceneRayVisibility::All & ~grid::sceneRayVisibility::Shadow);
    check (grid::validateVisual3DScene (sized).valid(), "an area light and object visibility validate");
    const auto sizedBytes = sceneaov::detail::encodeScene (sized);
    check (sizedBytes.size() > plainBytes.size(), "the new fields add wire trailers");

    grid::Visual3DScene decoded;
    check (sceneaov::detail::decodeScene (sizedBytes, decoded), "the extended scene decodes");
    check (decoded.lights[0].kind == grid::SceneLightKind::Area
               && grid::sameLightShapeAndVisibility (decoded.lights[0], sized.lights[0])
               && decoded.objects[0].visibility == sized.objects[0].visibility,
           "shape, spread and visibility survive the wire");

    grid::Visual3DScene decodedPlain;
    check (sceneaov::detail::decodeScene (plainBytes, decodedPlain)
               && sceneaov::detail::encodeScene (decodedPlain) == plainBytes
               && decodedPlain.objects[0].visibility == grid::sceneRayVisibility::All
               && ! grid::sceneLightHasShapeOrVisibility (decodedPlain.lights[0]),
           "a scene without the new fields round-trips to the same bytes and defaults");

    auto badShape = sized;
    badShape.lights[0].spread = 0.0f;
    check (! grid::validateVisual3DScene (badShape).valid(), "a zero spread is refused");
}

} // namespace

int main()
{
    areaShapeTests();
    spreadTests();
    sceneAreaRecordTests();
    visibilityTests();
    wireTests();
    std::cout << checks << " checks, " << failures << " failures\n";
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
