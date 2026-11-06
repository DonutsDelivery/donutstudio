// A light's Volume Scatter visibility on the media: a light that clears the bit
// adds no in-scatter to the world fog on the CPU path tracer (and OptiX, which
// runs the same integrator body), and the raster medium marches on OpenGL and
// Metal skip it. volume_light_visibility_source_contract.py pins the backends'
// light tables. Renders run on the CPU at 4 samples.

#include "pathtrace/accumulator.h"
#include "pathtrace/integrator.h"
#include "pathtrace/scene_ingest.h"
#include "../../shared/RasterMedium.h"

#include <cstdint>
#include <iostream>
#include <string>

namespace
{

namespace pt = videohelper::pathtrace;
namespace grid = HarmonicMIDI::grid;

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

grid::SceneLightRecord pointLight (float radius, std::uint8_t visibility)
{
    grid::SceneLightRecord light;
    light.kind = grid::SceneLightKind::Point;
    light.transform.translation = { 0.0f, 2.0f, 0.0f };
    light.intensity = 20.0f;
    light.radius = radius;
    light.visibility = visibility;
    return light;
}

// Mean radiance of a 4 by 4 view through fog lit by one light out of frame,
// with no surfaces, so all light reaches the camera by scattering in the fog.
double fogMean (const grid::SceneLightRecord& light, std::uint64_t& nonFinite)
{
    pt::RenderScene scene;
    scene.materials = { pt::Material {} };
    scene.lightRecords = { light };
    scene.worldVolume.volumeDensity = 0.2f;
    scene.worldVolume.volumeColor = { 1.0f, 1.0f, 1.0f };
    scene.worldVolumeRadius = 6.0f;
    scene.commit();

    pt::PinholeCamera camera;
    camera.position = { 0.0f, 0.0f, 5.0f };
    camera.tanHalfFov = 0.1f;
    camera.aspect = 1.0f;
    camera.nearPlane = 0.01f;
    camera.farPlane = 1000.0f;
    camera.width = 4;
    camera.height = 4;

    pt::PathTracer tracer (4);
    tracer.settings.maxDepth = 4;
    const pt::CancellationToken cancel {};
    nonFinite = tracer.renderFixed (scene, camera, camera.width, camera.height, 4, 1, cancel).nonFiniteSamples;
    double sum = 0.0;
    for (std::uint32_t y = 0; y < camera.height; ++y)
        for (std::uint32_t x = 0; x < camera.width; ++x)
        {
            const pt::Rgb v = tracer.accumulator().mean (x, y);
            sum += (v.r + v.g + v.b) / 3.0;
        }
    return sum / 16.0;
}

void pathTracerFog()
{
    constexpr auto all = grid::sceneRayVisibility::All;
    constexpr auto noVolume = static_cast<std::uint8_t> (all & ~grid::sceneRayVisibility::VolumeScatter);
    constexpr auto noDiffuse = static_cast<std::uint8_t> (all & ~grid::sceneRayVisibility::Diffuse);
    for (const float radius : { 0.0f, 0.3f })
    {
        const std::string light = radius > 0.0f ? "a sphere light" : "a point light";
        std::uint64_t nonFinite = 0;
        const double seen = fogMean (pointLight (radius, all), nonFinite);
        check (nonFinite == 0 && seen > 1.0e-3, "fog scatters " + light + " toward the camera: " + std::to_string (seen));
        const double hidden = fogMean (pointLight (radius, noVolume), nonFinite);
        check (nonFinite == 0 && hidden == 0.0, light + " hidden from volume scatter leaves the fog dark: " + std::to_string (hidden));
        const double diffuseOff = fogMean (pointLight (radius, noDiffuse), nonFinite);
        check (diffuseOff > 1.0e-3, light + " hidden only from diffuse bounces still lights the fog");
    }
}

void rasterMarches()
{
    const auto glsl = rastermedium::glslShader();
    check (glsl.find ("if (!fixtureLightInVolumes(i)) continue;") != std::string::npos,
           "the OpenGL medium march skips a light hidden from volume scatter");
    const auto metal = rastermedium::metalShader();
    check (metal.find ("if (u.lightKinds[i].y != 0u) continue;") != std::string::npos,
           "the Metal medium march skips a light hidden from volume scatter");
}

} // namespace

int main()
{
    pathTracerFog();
    rasterMarches();
    if (failures != 0)
    {
        std::cerr << failures << " of " << checks << " volume light visibility checks failed\n";
        return 1;
    }
    std::cout << checks << " volume light visibility checks passed\n";
    return 0;
}
