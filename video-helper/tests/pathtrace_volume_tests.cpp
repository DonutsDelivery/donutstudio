// Tests for participating media in the path tracer (Plan 3, volumes):
// transmittance against Beer-Lambert, the Henyey-Greenstein phase function,
// free-flight distance sampling, volume emission, a white furnace for a
// medium that only scatters, light samples from inside fog, camera
// visibility, and the volume bounce limit. Also the scene record mapping, the
// Render node's media wire and the Realtime Depth Fog that stands in for scene
// fog, and the OpenGL Raster and Realtime material media. Renders run on the CPU at 16 samples or fewer.

#include "pathtrace/accumulator.h"
#include "pathtrace/bsdf.h"
#include "pathtrace/integrator.h"
#include "pathtrace/media.h"
#include "pathtrace/scene_ingest.h"
#include "pathtrace/spectrum.h"
#include "pathtrace/thread_pool.h"
#include "../../shared/RasterMedium.h"
#include "../../shared/SceneAovOperationContract.h"
#include "../../shared/VisualImportedSceneRenderOperationContract.h"
#include "../../shared/WorldVolumeDepthFog.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
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

bool closeRelative (double a, double b, double relative, double absolute = 0.0)
{
    return std::fabs (a - b) <= relative * std::max (std::fabs (a), std::fabs (b)) + absolute;
}

float u01 (std::mt19937& rng)
{
    return static_cast<float> (static_cast<std::uint32_t> (rng()) >> 8) * (1.0f / 16777216.0f);
}

// ---------------------------------------------------------------- scenes

struct Mesh
{
    std::vector<Vec3> vertices;
    std::vector<std::uint32_t> indices;
};

Mesh box (Vec3 lo, Vec3 hi)
{
    Mesh mesh;
    for (std::uint32_t i = 0; i < 8; ++i)
        mesh.vertices.push_back ({ (i & 1u) != 0 ? hi.x : lo.x, (i & 2u) != 0 ? hi.y : lo.y, (i & 4u) != 0 ? hi.z : lo.z });
    const std::uint32_t faces[6][4] { { 0, 2, 6, 4 }, { 1, 5, 7, 3 }, { 0, 4, 5, 1 },
                                      { 2, 3, 7, 6 }, { 0, 1, 3, 2 }, { 4, 6, 7, 5 } };
    for (const auto& f : faces)
        mesh.indices.insert (mesh.indices.end(), { f[0], f[1], f[2], f[0], f[2], f[3] });
    return mesh;
}

Mesh quad (Vec3 centre, Vec3 halfU, Vec3 halfV)
{
    Mesh mesh;
    mesh.vertices = { centre - halfU - halfV, centre + halfU - halfV, centre + halfU + halfV, centre - halfU + halfV };
    mesh.indices = { 0, 1, 2, 0, 2, 3 };
    return mesh;
}

// Flat triangles; with outwardFrom set every front face points away from it.
void appendMesh (pt::RenderScene& scene, const Mesh& mesh, std::uint32_t material, const Vec3* outwardFrom = nullptr)
{
    for (std::size_t i = 0; i + 2 < mesh.indices.size(); i += 3)
    {
        Vec3 p0 = mesh.vertices[mesh.indices[i]];
        Vec3 p1 = mesh.vertices[mesh.indices[i + 1]];
        Vec3 p2 = mesh.vertices[mesh.indices[i + 2]];
        Vec3 n = pt::normalizeOrZero (pt::cross (p1 - p0, p2 - p0));
        if (outwardFrom != nullptr && pt::dot (n, (p0 + p1 + p2) * (1.0f / 3.0f) - *outwardFrom) < 0.0f)
        {
            std::swap (p1, p2);
            n = -n;
        }
        pt::TriangleAttributes attributes;
        attributes.materialIndex = material;
        attributes.primitiveIndex = static_cast<std::uint32_t> (i / 3);
        attributes.geometricNormal = n;
        attributes.shadingNormals[0] = n;
        attributes.shadingNormals[1] = n;
        attributes.shadingNormals[2] = n;
        attributes.hasShadingNormals = false;
        for (const Vec3 p : { p0, p1, p2 })
        {
            scene.geometry.positions.push_back (p);
            scene.geometry.bounds.grow (p);
        }
        scene.geometry.triangles.push_back (attributes);
    }
}

grid::SceneLightRecord environmentRecord (float radiance)
{
    grid::SceneLightRecord light;
    light.kind = grid::SceneLightKind::Environment;
    light.color = { radiance, radiance, radiance };
    light.intensity = 1.0f;
    return light;
}

grid::SceneLightRecord pointRecord (Vec3 position, float intensity)
{
    grid::SceneLightRecord light;
    light.kind = grid::SceneLightKind::Point;
    light.transform.translation = { position.x, position.y, position.z };
    light.intensity = intensity;
    return light;
}

grid::SceneLightRecord directionalRecord (float irradiance)
{
    // The record's +Z is the direction toward the light: here straight up +Z.
    grid::SceneLightRecord light;
    light.kind = grid::SceneLightKind::Directional;
    light.intensity = irradiance;
    return light;
}

pt::PinholeCamera makeCamera (Vec3 position, std::uint32_t width, std::uint32_t height, float tanHalfFov = 0.3f)
{
    pt::PinholeCamera camera;
    camera.position = position;  // looks down -Z
    camera.tanHalfFov = tanHalfFov;
    camera.aspect = static_cast<float> (width) / static_cast<float> (height);
    camera.nearPlane = 0.01f;
    camera.farPlane = 1000.0f;
    camera.width = width;
    camera.height = height;
    return camera;
}

// A medium with no surface: rays cross its boundary unchanged.
pt::Material volumeOnly (float density, Vec3 colour, Vec3 absorptionColour = {}, Vec3 emission = {})
{
    pt::Material m;
    m.volumeOnly = true;
    m.volumeDensity = density;
    m.volumeColor = colour;
    m.volumeAbsorptionColor = absorptionColour;
    m.volumeEmission = emission;
    return m;
}

struct Image
{
    std::vector<Vec3> pixels;
    pt::RenderStats stats {};

    double mean() const
    {
        double sum = 0.0;
        for (const Vec3 p : pixels)
            sum += (p.x + p.y + p.z) / 3.0;
        return sum / static_cast<double> (std::max<std::size_t> (1, pixels.size()));
    }
};

Image render (const pt::RenderScene& scene, const pt::PinholeCamera& camera, const pt::IntegratorSettings& settings,
              std::uint32_t samples)
{
    pt::PathTracer tracer (4);
    tracer.settings = settings;
    const pt::CancellationToken cancel {};
    Image image;
    image.stats = tracer.renderFixed (scene, camera, camera.width, camera.height, samples, 1, cancel);
    for (std::uint32_t y = 0; y < camera.height; ++y)
        for (std::uint32_t x = 0; x < camera.width; ++x)
        {
            const pt::Rgb v = tracer.accumulator().mean (x, y);
            image.pixels.push_back ({ v.r, v.g, v.b });
        }
    return image;
}

// Transmittance of the field material along a line by midpoint quadrature.
std::array<double, 3> quadratureTransmittance (const pt::MediumCoefficients& m, Vec3 origin, Vec3 dir, float d)
{
    const int steps = 20000;
    std::array<double, 3> optical { 0.0, 0.0, 0.0 };
    const double h = static_cast<double> (d) / steps;
    for (int i = 0; i < steps; ++i)
    {
        const float x = static_cast<float> ((i + 0.5) * h);
        const float f = pt::volumeFieldAt (*m.field, origin + dir * x);
        for (int c = 0; c < 3; ++c)
            optical[static_cast<std::size_t> (c)] += h * (m.absorption[c] + f * (m.volumeAbsorption[c] + m.scattering[c]));
    }
    return { std::exp (-optical[0]), std::exp (-optical[1]), std::exp (-optical[2]) };
}

// ---------------------------------------------------------------- transmittance

void transmittanceTests()
{
    const pt::Wavelengths rgb = pt::rgbChannels();
    pt::Material fog = volumeOnly (2.0f, { 0.8f, 0.5f, 0.2f }, { 0.1f, 0.3f, 0.6f });
    pt::Spectrum absorption;
    pt::Spectrum scattering;
    pt::Spectrum emission;
    pt::volumeCoefficients (fog, rgb, absorption, scattering, emission);
    check (std::fabs (scattering[0] - 1.6f) < 1.0e-5f && std::fabs (scattering[2] - 0.4f) < 1.0e-5f,
           "scattering is density times colour");
    check (std::fabs (absorption[1] - 2.0f * 0.5f * 0.7f) < 1.0e-5f, "absorption is density (1 - colour)(1 - absorption colour)");
    check (scattering[3] == 0.0f && absorption[3] == 0.0f, "RGB transport leaves the fourth channel empty");

    const pt::MediumCoefficients m = pt::mediumCoefficientsOf (&fog, rgb, true);
    pt::VolumeRng rng { 7u };
    const float d = 0.8f;
    const pt::Spectrum t = pt::mediumTransmittance (m, {}, { 0.0f, 0.0f, 1.0f }, d, rng);
    for (int c = 0; c < 3; ++c)
        check (closeRelative (t[c], std::exp (-(absorption[c] + scattering[c]) * d), 1.0e-5),
               "homogeneous transmittance is Beer-Lambert of extinction, channel " + std::to_string (c));
    const pt::MediumCoefficients hidden = pt::mediumCoefficientsOf (&fog, rgb, false);
    check (! hidden.any(), "a volume hidden from the ray leaves nothing for a volume-only material");

    // A world volume fills its ball only.
    pt::MediumCoefficients world = m;
    world.bounded = true;
    world.radius = 1.0f;
    const pt::Spectrum outside = pt::mediumTransmittance (world, { 0.0f, 0.0f, -5.0f }, { 0.0f, 0.0f, 1.0f },
                                                          std::numeric_limits<float>::infinity(), rng);
    check (closeRelative (outside[0], std::exp (-(absorption[0] + scattering[0]) * 2.0f), 1.0e-4),
           "an infinite ray through the world ball crosses its diameter: " + describe (outside[0]));

    // Ratio tracking through a noise field matches quadrature.
    pt::Material cloud = fog;
    cloud.volumeField = pt::VolumeDensityField::Noise;
    cloud.volumeFieldScale = 2.0f;
    cloud.volumeFieldDetail = 3;
    const pt::MediumCoefficients het = pt::mediumCoefficientsOf (&cloud, rgb, true);
    check (het.field == &cloud, "a noise field makes the medium heterogeneous");
    const Vec3 origin { 0.3f, -0.2f, 0.1f };
    const Vec3 dir = pt::normalizeOrZero ({ 0.4f, 0.3f, 1.0f });
    const float length = 1.5f;
    const std::array<double, 3> expected = quadratureTransmittance (het, origin, dir, length);
    std::array<double, 3> sum { 0.0, 0.0, 0.0 };
    const int trials = 40000;
    for (int i = 0; i < trials; ++i)
    {
        pt::VolumeRng stream { pt::VolumeRng::hash (static_cast<std::uint32_t> (i) * 2654435761u) };
        const pt::Spectrum estimate = pt::mediumTransmittance (het, origin, dir, length, stream);
        for (int c = 0; c < 3; ++c)
            sum[static_cast<std::size_t> (c)] += estimate[c];
    }
    for (int c = 0; c < 3; ++c)
    {
        const double measured = sum[static_cast<std::size_t> (c)] / trials;
        check (closeRelative (measured, expected[static_cast<std::size_t> (c)], 0.02, 2.0e-3),
               "ratio tracking matches the quadrature transmittance, channel " + std::to_string (c) + ": "
                   + describe (measured) + " vs " + describe (expected[static_cast<std::size_t> (c)]));
    }
    float lo = 1.0f;
    float hi = 0.0f;
    for (int i = 0; i < 2000; ++i)
    {
        const float f = pt::volumeFieldAt (cloud, { 0.37f * static_cast<float> (i), 0.11f * static_cast<float> (i), -0.23f * static_cast<float> (i) });
        lo = std::min (lo, f);
        hi = std::max (hi, f);
    }
    check (lo >= 0.0f && hi <= 1.0f && hi - lo > 0.3f, "the noise field varies inside [0, 1]");
}

// ---------------------------------------------------------------- phase function

void phaseTests()
{
    for (const float g : { -0.7f, 0.0f, 0.3f, 0.9f })
    {
        // The pdf over the sphere: 2 pi times its integral over cos.
        const int steps = 200000;
        double integral = 0.0;
        for (int i = 0; i < steps; ++i)
        {
            const float c = -1.0f + 2.0f * (static_cast<float> (i) + 0.5f) / static_cast<float> (steps);
            integral += pt::henyeyGreenstein (c, g) * (2.0 / steps);
        }
        integral *= 2.0 * 3.14159265358979323846;
        check (std::fabs (integral - 1.0) < 2.0e-3, "HG pdf integrates to 1 for g = " + describe (g) + ": " + describe (integral));

        std::mt19937 rng (11);
        const Vec3 forward = pt::normalizeOrZero ({ 0.2f, -0.5f, 0.8f });
        double meanCos = 0.0;
        bool pdfMatches = true;
        const int n = 60000;
        for (int i = 0; i < n; ++i)
        {
            float pdf = 0.0f;
            const Vec3 wi = pt::sampleHenyeyGreenstein (forward, g, { u01 (rng), u01 (rng) }, pdf);
            const float c = pt::dot (wi, forward);
            meanCos += c;
            pdfMatches = pdfMatches && std::fabs (pt::length (wi) - 1.0f) < 1.0e-3f
                      && closeRelative (pdf, pt::henyeyGreenstein (c, g), 2.0e-3, 1.0e-5);
        }
        meanCos /= n;
        check (pdfMatches, "HG samples are unit and report their own density, g = " + describe (g));
        check (std::fabs (meanCos - g) < 0.01, "HG mean cosine is g = " + describe (g) + ": " + describe (meanCos));
    }
}

// ---------------------------------------------------------------- distance sampling

void distanceSamplingTests()
{
    const pt::Wavelengths rgb = pt::rgbChannels();
    pt::Spectrum live;
    live[0] = live[1] = live[2] = 1.0f;
    const float d = 1.2f;
    const Vec3 dir { 0.0f, 0.0f, 1.0f };

    // Homogeneous and chromatic: the estimator's expectation is the
    // transmittance to the end plus the scattered share before it.
    pt::Material fog = volumeOnly (1.5f, { 0.9f, 0.5f, 0.1f }, { 0.2f, 0.2f, 0.2f });
    const pt::MediumCoefficients m = pt::mediumCoefficientsOf (&fog, rgb, true);
    const pt::Spectrum sigmaT = m.majorant();
    std::array<double, 3> pass { 0.0, 0.0, 0.0 };
    std::array<double, 3> scatter { 0.0, 0.0, 0.0 };
    const int n = 200000;
    for (int i = 0; i < n; ++i)
    {
        const float u = (static_cast<float> (i) + 0.5f) / static_cast<float> (n);
        const pt::MediumEvent event = pt::sampleMediumEvent (m, {}, dir, d, live, u);
        for (int c = 0; c < 3; ++c)
            (event.scattered ? scatter : pass)[static_cast<std::size_t> (c)] += event.weight[c];
    }
    for (int c = 0; c < 3; ++c)
    {
        const double t = std::exp (-sigmaT[c] * d);
        const double s = m.scattering[c] / sigmaT[c] * (1.0 - t);
        check (closeRelative (pass[static_cast<std::size_t> (c)] / n, t, 0.01),
               "passing weight averages the transmittance, channel " + std::to_string (c));
        check (closeRelative (scatter[static_cast<std::size_t> (c)] / n, s, 0.01),
               "scatter weight averages the scattered share, channel " + std::to_string (c) + ": "
                   + describe (scatter[static_cast<std::size_t> (c)] / n) + " vs " + describe (s));
    }

    // Heterogeneous: spectral tracking passes with the field's transmittance.
    pt::Material cloud = fog;
    cloud.volumeField = pt::VolumeDensityField::Noise;
    cloud.volumeFieldScale = 1.5f;
    const pt::MediumCoefficients het = pt::mediumCoefficientsOf (&cloud, rgb, true);
    const Vec3 origin { 0.1f, 0.7f, -0.4f };
    const std::array<double, 3> expected = quadratureTransmittance (het, origin, dir, d);
    std::array<double, 3> hetPass { 0.0, 0.0, 0.0 };
    int scattered = 0;
    for (int i = 0; i < n; ++i)
    {
        const float u = (static_cast<float> (i) + 0.5f) / static_cast<float> (n);
        const pt::MediumEvent event = pt::sampleMediumEvent (het, origin, dir, d, live, u);
        scattered += event.scattered ? 1 : 0;
        if (! event.scattered)
            for (int c = 0; c < 3; ++c)
                hetPass[static_cast<std::size_t> (c)] += event.weight[c];
    }
    check (scattered > 0, "spectral tracking scatters in a heterogeneous medium");
    for (int c = 0; c < 3; ++c)
        check (closeRelative (hetPass[static_cast<std::size_t> (c)] / n, expected[static_cast<std::size_t> (c)], 0.03, 3.0e-3),
               "spectral tracking passes with the field's transmittance, channel " + std::to_string (c) + ": "
                   + describe (hetPass[static_cast<std::size_t> (c)] / n) + " vs " + describe (expected[static_cast<std::size_t> (c)]));
}

// ---------------------------------------------------------------- emission

void emissionTests()
{
    const pt::Wavelengths rgb = pt::rgbChannels();
    pt::Spectrum live;
    live[0] = live[1] = live[2] = 1.0f;
    // Pure absorber that glows: colour 0 scatters nothing.
    const float density = 1.3f;
    const Vec3 glow { 0.6f, 0.3f, 0.9f };
    pt::Material ember = volumeOnly (density, {}, {}, glow);
    const pt::MediumCoefficients m = pt::mediumCoefficientsOf (&ember, rgb, true);
    const float d = 0.7f;
    const pt::MediumEvent event = pt::sampleMediumEvent (m, {}, { 0.0f, 0.0f, 1.0f }, d, live, 0.5f);
    const double closed = (1.0 - std::exp (-density * d)) / density;
    check (! event.scattered && closeRelative (event.emitted[0], glow.x * closed, 1.0e-4)
               && closeRelative (event.emitted[2], glow.z * closed, 1.0e-4),
           "a segment gathers emission e (1 - exp(-sigma d)) / sigma");

    // A glowing slab seen through: thickness 1, nothing else in the scene.
    const auto slab = [&] (bool visible, bool lit) {
        pt::RenderScene scene;
        scene.materials = { ember };
        const Vec3 centre {};
        appendMesh (scene, box ({ -3.0f, -3.0f, -0.5f }, { 3.0f, 3.0f, 0.5f }), 0, &centre);
        if (lit)
            scene.lightRecords = { environmentRecord (0.25f) };
        scene.commit();
        pt::IntegratorSettings settings;
        settings.volumesVisibleToCamera = visible;
        return render (scene, makeCamera ({ 0.0f, 0.0f, 4.0f }, 8, 8, 0.05f), settings, 4);
    };
    const Image glowing = slab (true, false);
    const double expected = (1.0 - std::exp (-density * 1.0)) / density;
    check (glowing.stats.nonFiniteSamples == 0 && closeRelative (glowing.pixels[27].x, glow.x * expected, 0.02)
               && closeRelative (glowing.pixels[27].z, glow.z * expected, 0.02),
           "a glowing slab shows its emission integral: " + describe (glowing.pixels[27].x) + " vs "
               + describe (glow.x * expected));
    const Image lit = slab (true, true);
    const double behind = 0.25 * std::exp (-density * 1.0);
    check (closeRelative (lit.pixels[27].y, glow.y * expected + behind, 0.02),
           "emission adds to the light the slab lets through: " + describe (lit.pixels[27].y) + " vs "
               + describe (glow.y * expected + behind));
    const Image hidden = slab (false, true);
    check (closeRelative (hidden.pixels[27].x, 0.25, 1.0e-3),
           "a volume hidden from camera rays shows the background unchanged: " + describe (hidden.pixels[27].x));
}

// ---------------------------------------------------------------- white furnace

pt::RenderScene furnaceScene (float density)
{
    pt::RenderScene scene;
    scene.materials = { volumeOnly (density, { 1.0f, 1.0f, 1.0f }) };
    const Vec3 centre {};
    appendMesh (scene, box ({ -1.0f, -1.0f, -1.0f }, { 1.0f, 1.0f, 1.0f }), 0, &centre);
    scene.lightRecords = { environmentRecord (0.5f) };
    scene.commit();
    return scene;
}

void furnaceTests()
{
    const pt::RenderScene scene = furnaceScene (2.0f);
    const pt::PinholeCamera camera = makeCamera ({ 0.0f, 0.0f, 4.0f }, 16, 16, 0.2f);
    for (const pt::SpectralMode mode : { pt::SpectralMode::Off, pt::SpectralMode::On })
    {
        pt::IntegratorSettings settings;
        settings.maxDepth = 256;
        settings.spectral = mode;
        const Image image = render (scene, camera, settings, 16);
        const std::string name = mode == pt::SpectralMode::Off ? "RGB" : "spectral";
        check (image.stats.nonFiniteSamples == 0, name + " furnace: no non-finite samples");
        check (closeRelative (image.mean(), 0.5, 0.04),
               name + " white furnace: a medium that only scatters conserves energy: " + describe (image.mean()));
        check (image.stats.bounceLimitPaths == 0, name + " furnace: no path reached a limit");
    }
}

// ---------------------------------------------------------------- bounce limit

void bounceLimitTests()
{
    const pt::RenderScene scene = furnaceScene (4.0f);
    const pt::PinholeCamera camera = makeCamera ({ 0.0f, 0.0f, 4.0f }, 16, 16, 0.2f);
    pt::IntegratorSettings settings;
    settings.maxDepth = 256;
    const Image full = render (scene, camera, settings, 8);
    settings.maxVolumeBounces = 1;
    const Image limited = render (scene, camera, settings, 8);
    check (limited.stats.samplesAdded == full.stats.samplesAdded && ! limited.stats.invalidRequest,
           "a volume bounce limit never refuses the frame");
    check (limited.stats.bounceLimitPaths > 0 && limited.stats.limitEndedPaths >= limited.stats.bounceLimitPaths,
           "paths ended by the volume bounce limit are counted for the HUD: " + std::to_string (limited.stats.bounceLimitPaths));
    check (limited.mean() < full.mean() - 0.02 && limited.mean() > 0.0,
           "the limit ends paths and their later light is missing: " + describe (limited.mean()) + " vs " + describe (full.mean()));
    settings.maxVolumeBounces = 0;
    const Image single = render (scene, camera, settings, 8);
    check (single.mean() > 0.0 && single.mean() < limited.mean(), "a limit of 0 still shows single scattering");
}

// ---------------------------------------------------------------- light samples in fog

void fogLightTests()
{
    // Fog around a small floor below the view; the camera looks through the fog.
    const auto fog = [] (const grid::SceneLightRecord& light) {
        pt::RenderScene scene;
        scene.materials = { pt::Material {} };
        appendMesh (scene, quad ({ 0.0f, -3.0f, 0.0f }, { 0.5f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.5f }), 0);
        scene.lightRecords = { light };
        scene.worldVolume.volumeDensity = 0.2f;
        scene.worldVolume.volumeColor = { 1.0f, 1.0f, 1.0f };
        scene.worldVolumeRadius = 6.0f;
        scene.commit();
        pt::IntegratorSettings settings;
        settings.maxDepth = 4;
        return render (scene, makeCamera ({ 0.0f, 0.0f, 5.0f }, 4, 4, 0.1f), settings, 4);
    };
    const Image point = fog (pointRecord ({ 0.0f, 2.0f, 0.0f }, 20.0f));
    const Image sun = fog (directionalRecord (3.0f));
    check (point.stats.nonFiniteSamples == 0 && point.mean() > 1.0e-3, "fog scatters a point light toward the camera: " + describe (point.mean()));
    check (sun.stats.nonFiniteSamples == 0 && sun.mean() > 1.0e-3, "fog scatters the sun toward the camera: " + describe (sun.mean()));

    pt::RenderScene clear;
    clear.materials = { pt::Material {} };
    appendMesh (clear, quad ({ 0.0f, -20.0f, 0.0f }, { 0.5f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.5f }), 0);
    clear.lightRecords = { pointRecord ({ 0.0f, 2.0f, 0.0f }, 20.0f) };
    clear.commit();
    check (! clear.hasWorldVolume && ! clear.anyVolume, "a scene without volumes keeps the volume path off");
    const Image dark = render (clear, makeCamera ({ 0.0f, 0.0f, 5.0f }, 4, 4, 0.1f), pt::IntegratorSettings {}, 4);
    check (dark.mean() == 0.0, "without fog the same view stays black");
}

} // namespace

// A scene record's medium reaches the tracer's material, and the scene's fog
// becomes the world volume unless the caller supplies one.
void sceneRecordTests()
{
    grid::SceneMaterialRecord record;
    record.volume.density = 3.0f;
    record.volume.color = { 0.2f, 0.4f, 0.6f };
    record.volume.absorptionColor = { 0.0f, 0.5f, 0.0f };
    record.volume.anisotropy = 0.7f;
    record.volume.emission = { 2.0f, 0.0f, 1.0f };
    record.volume.field = grid::SceneVolumeField::Noise;
    record.volume.fieldDetail = 4;
    record.volume.fieldScale = 0.5f;
    record.volume.fieldOffset = { 1.0f, 2.0f, 3.0f };
    record.volume.volumeOnly = true;
    const pt::Material material = pt::materialFromRecord (record);
    check (material.volumeDensity == 3.0f && pt::materialHasVolume (material), "record density maps");
    check (material.volumeColor.y == 0.4f && material.volumeAbsorptionColor.y == 0.5f, "record colours map");
    check (material.volumeAnisotropy == 0.7f, "record anisotropy maps");
    check (material.volumeEmission.x == 2.0f && material.volumeEmission.z == 1.0f, "record emission maps");
    check (material.volumeField == pt::VolumeDensityField::Noise && material.volumeFieldDetail == 4
           && material.volumeFieldScale == 0.5f && material.volumeFieldOffset.z == 3.0f, "record field maps");
    check (material.volumeOnly, "record volume-only maps");
    check (! pt::materialHasVolume (pt::materialFromRecord (grid::SceneMaterialRecord {})),
           "a released record has no medium");

    grid::Visual3DScene scene;
    scene.id.value = 1;
    scene.worldVolume.density = 0.25f;
    scene.worldVolume.color = { 0.9f, 0.9f, 0.9f };
    scene.worldVolume.radius = 12.0f;
    pt::RenderScene fromScene;
    pt::prepareRenderScene (scene, fromScene, {});
    check (fromScene.worldVolume.volumeDensity == 0.25f && fromScene.worldVolumeRadius == 12.0f,
           "scene fog becomes the world volume");
    pt::PrepareOptions options;
    options.worldVolume.volumeDensity = 0.5f;
    pt::RenderScene explicitVolume;
    pt::prepareRenderScene (scene, explicitVolume, options);
    check (explicitVolume.worldVolume.volumeDensity == 0.5f, "an explicit world volume wins");
    grid::Visual3DScene clear;
    clear.id.value = 1;
    pt::RenderScene noFog;
    pt::prepareRenderScene (clear, noFog, {});
    check (! pt::materialHasVolume (noFog.worldVolume), "a scene without fog has no world volume");

    // Noise fog keeps its field through the scene's commit.
    scene.worldVolume.field = grid::SceneVolumeField::Noise;
    scene.worldVolume.fieldScale = 0.5f;
    pt::RenderScene noiseFog;
    pt::prepareRenderScene (scene, noiseFog, {});
    noiseFog.commit();
    check (noiseFog.hasWorldVolume && noiseFog.worldVolume.volumeField == pt::VolumeDensityField::Noise
           && noiseFog.worldVolume.volumeFieldScale == 0.5f, "noise fog keeps its density field");
}

// V35: the Render node's media settings wrap every other envelope, absent on the default.
// V37: an imported scene's Environment Light fog wraps them all.
void renderWireTests()
{
    namespace render = visualimportedscenerender;
    check (render::pathTraceVolume (render::kPathTraceVolumeUnlimitedBounces, true) == render::kDefaultPathTraceVolume,
           "the default byte lifts the limit and shows media");
    check (! render::pathTraceVolumeBounces (render::kDefaultPathTraceVolume).has_value(), "the top step is unlimited");
    check (render::pathTraceVolumeBounces (render::pathTraceVolume (3, false)) == 3u
               && ! render::pathTraceVolumesVisibleToCamera (render::pathTraceVolume (3, false)),
           "a limit and hidden media pack into one byte");
    check (render::pathTraceVolumeBounces (render::pathTraceVolume (500, true)) == std::nullopt,
           "a limit past the top step lifts it");

    render::Request traced;
    traced.sourceStableId = 71;
    traced.renderStableId = 72;
    traced.asset.id = "model-asset-1";
    traced.asset.version = 3;
    traced.asset.contentSha256 = std::string (64, 'a');
    traced.asset.sourceMediaType = "model/gltf-binary";
    traced.asset.sourceByteSize = 4096;
    traced.sceneIndex = 0;
    traced.renderEngine = render::RenderEngine::PathTrace;
    const auto tracedWire = render::encode (traced);
    check (! tracedWire.empty() && tracedWire.find (render::kCompiledOperationHeaderV35) == std::string::npos,
           "the default media settings keep the wire");
    auto media = traced;
    media.pathTraceVolume = render::pathTraceVolume (2, false);
    const auto mediaWire = render::encode (media);
    check (mediaWire == std::string (render::kCompiledOperationHeaderV35) + "\n"
                            + std::to_string (media.pathTraceVolume) + " " + std::to_string (tracedWire.size())
                            + "\n" + tracedWire,
           "V35 carries the byte and the exact default bytes");
    render::Request decoded;
    check (render::decode (mediaWire, decoded) && decoded.pathTraceVolume == media.pathTraceVolume
               && render::sameRequest (media, decoded) && ! render::sameRequest (traced, decoded),
           "media settings round-trip and are part of request identity");
    auto bounded = media;
    bounded.pathTraceMaxBounces = 9;
    check (render::decode (render::encode (bounded), decoded) && decoded.pathTraceMaxBounces == 9
               && decoded.pathTraceVolume == media.pathTraceVolume,
           "V35 wraps the bounce limit envelope");
    const auto header = std::string (render::kCompiledOperationHeaderV35) + "\n";
    check (! render::decode (header + std::to_string (render::kDefaultPathTraceVolume) + " "
                                 + std::to_string (tracedWire.size()) + "\n" + tracedWire, decoded),
           "the default written out is not canonical");
    auto raster = media;
    raster.renderEngine = render::RenderEngine::Raster;
    check (render::encode (raster).empty(), "Raster may not set media settings");

    // V37: an imported scene's Environment Light fog, for every engine.
    auto fog = traced;
    fog.renderEngine = render::RenderEngine::Realtime;
    const auto clearWire = render::encode (fog);
    check (clearWire.find (render::kCompiledOperationHeaderV37) == std::string::npos, "no fog keeps the wire");
    grid::SceneVolume volume;
    volume.density = 0.2f;
    volume.color = { 0.7f, 0.8f, 0.9f };
    volume.anisotropy = 0.4f;
    volume.radius = 30.0f;
    volume.field = grid::SceneVolumeField::Noise;
    volume.fieldDetail = 5;
    volume.fieldScale = 0.25f;
    fog.worldVolume = volume;
    const auto fogWire = render::encode (fog);
    check (fogWire.rfind (std::string (render::kCompiledOperationHeaderV37) + "\n"
                              + std::to_string (render::detail::floatBits (0.2f)) + " ", 0) == 0
               && fogWire.compare (fogWire.size() - clearWire.size(), clearWire.size(), clearWire) == 0,
           "V37 carries the fog and the exact fogless bytes");
    check (render::decode (fogWire, decoded) && decoded.worldVolume
               && grid::sameSceneVolume (*decoded.worldVolume, volume) && render::sameRequest (fog, decoded),
           "fog round-trips and is part of request identity");
    auto both = fog;
    both.renderEngine = render::RenderEngine::PathTrace;
    both.pathTraceVolume = render::pathTraceVolume (2, false);
    const auto bothWire = render::encode (both);
    check (bothWire.rfind (std::string (render::kCompiledOperationHeaderV37) + "\n", 0) == 0
               && render::decode (bothWire, decoded) && decoded.pathTraceVolume == both.pathTraceVolume
               && decoded.worldVolume,
           "V37 wraps the V35 media envelope");
    check (render::decode (mediaWire, decoded) && ! decoded.worldVolume, "V35 bytes decode with no fog");
    auto off = fog;
    off.worldVolume->density = 0.0f;
    check (render::encode (off).empty(), "fog that is off is never written");
    auto volumeOnly = fog;
    volumeOnly.worldVolume->volumeOnly = true;
    check (render::encode (volumeOnly).empty(), "fog has no volume-only form");
}

// Scene fog becomes Realtime's Depth Fog: 95% at three mean free paths, or the
// true opacity at the far plane when the fog is thinner than that.
void realtimeFogTests()
{
    grid::SceneVolume volume;
    check (! renderpasscomposite::depthFogForWorldVolume (volume, 0.1f, 100.0f).has_value(), "no fog, no pass");
    volume.density = 0.3f;
    volume.color = { 0.2f, 0.4f, 0.9f };
    const auto dense = renderpasscomposite::depthFogForWorldVolume (volume, 0.0f, 100.0f);
    check (dense.has_value() && dense->mode == renderpasscomposite::Mode::DepthFog, "fog maps to Depth Fog");
    check (dense && std::abs (dense->farDepth - 0.1f) < 1e-6f && dense->amount == 0.95f,
           "dense fog reaches 95% at three mean free paths");
    check (dense && dense->fogColor[2] == 0.9f && renderpasscomposite::valid (*dense), "fog keeps its colour");
    volume.density = 0.001f;
    const auto thin = renderpasscomposite::depthFogForWorldVolume (volume, 0.0f, 100.0f);
    check (thin && thin->farDepth == 1.0f && std::abs (thin->amount - (1.0f - std::exp (-0.1f))) < 1e-5f,
           "thin fog holds its true opacity at the far plane");
    check (! renderpasscomposite::depthFogForWorldVolume (volume, 5.0f, 5.0f).has_value(),
           "a degenerate camera span maps to no pass");
    // Noise fog draws at the field's mean density.
    volume.density = 0.6f;
    volume.field = grid::SceneVolumeField::Noise;
    const auto noise = renderpasscomposite::depthFogForWorldVolume (volume, 0.0f, 100.0f);
    check (noise && std::abs (noise->farDepth - 0.1f) < 1e-6f, "noise fog draws at half its peak density");

    // Raster and Realtime draw the render's fog over the scene's, over the active camera's span.
    auto scene = std::make_unique<grid::Visual3DScene>();
    scene->cameras[0].id = { 4 };
    scene->cameras[0].nearPlane = 0.0f;
    scene->cameras[0].farPlane = 100.0f;
    scene->cameraCount = 1;
    scene->activeCamera = { 4 };
    check (! renderpasscomposite::depthFogForScene (*scene, nullptr).has_value(), "a scene without fog draws no pass");
    scene->worldVolume.density = 0.3f;
    scene->worldVolume.color = { 0.7f, 0.1f, 0.2f };
    const auto sceneFog = renderpasscomposite::depthFogForScene (*scene, nullptr);
    check (sceneFog && sceneFog->fogColor[0] == 0.7f && std::abs (sceneFog->farDepth - 0.1f) < 1e-6f,
           "the scene's fog draws as Depth Fog");
    grid::SceneVolume renderFog;
    renderFog.density = 0.001f;
    renderFog.color = { 0.0f, 0.0f, 1.0f };
    const auto overridden = renderpasscomposite::depthFogForScene (*scene, &renderFog);
    check (overridden && overridden->fogColor[2] == 1.0f && overridden->farDepth == 1.0f,
           "the render's own fog wins over the scene's");
    scene->activeCamera = { 5 };
    check (! renderpasscomposite::depthFogForScene (*scene, nullptr).has_value(), "no active camera draws no pass");

    // A material medium is found.
    scene->materialCount = 2;
    check (! renderpasscomposite::sceneHasMaterialMedia (*scene), "plain materials carry no media");
    scene->materials[1].volume.density = 2.0f;
    check (renderpasscomposite::sceneHasMaterialMedia (*scene), "a material medium is found");
    scene->materialCount = 1;
    check (! renderpasscomposite::sceneHasMaterialMedia (*scene), "materials past the count are ignored");
}

// OpenGL Raster and Realtime draw material media with RasterMedium: glass and
// water tint transmitted light by Beer-Lambert, and Volume Materials march
// inside their world bounds with Path Trace's coefficients and noise field.
void rasterMediumTests()
{
    HarmonicMIDI::grid::SceneVolume glass;
    check (rastermedium::modeOf (glass) == rastermedium::Mode::None, "no density is no medium");
    check (rastermedium::pack (glass, {}, {})[0][3] == 0.0f, "no medium packs mode 0");
    glass.density = 2.0f;
    glass.color = { 0.25f, 0.5f, 1.0f };
    glass.absorptionColor = { 0.0f, 0.5f, 0.0f };
    check (rastermedium::modeOf (glass) == rastermedium::Mode::Tint, "a surface medium tints");
    const auto a = rastermedium::absorption (glass), s = rastermedium::scattering (glass);
    check (std::fabs (a[0] - 1.5f) < 1.0e-6f && std::fabs (a[1] - 0.5f) < 1.0e-6f && a[2] == 0.0f,
           "absorption is density (1 - colour)(1 - absorption colour)");
    check (std::fabs (s[0] - 0.5f) < 1.0e-6f && std::fabs (s[2] - 2.0f) < 1.0e-6f, "scattering is density times colour");

    // Path Trace's coefficients match.
    pt::Material twin = volumeOnly (2.0f, { 0.25f, 0.5f, 1.0f }, { 0.0f, 0.5f, 0.0f });
    pt::Spectrum ptAbsorption, ptScattering, ptEmission;
    pt::volumeCoefficients (twin, pt::rgbChannels(), ptAbsorption, ptScattering, ptEmission);
    check (std::fabs (ptAbsorption[0] - a[0]) < 1.0e-5f && std::fabs (ptScattering[1] - s[1]) < 1.0e-5f,
           "Raster media use Path Trace's coefficients");

    const auto u = rastermedium::pack (glass, { -1.0f, -1.0f, -1.0f }, { 1.0f, 1.0f, 1.0f });
    check (u[0][3] == 1.0f && u[4][0] == -1.0f && u[5][2] == 1.0f, "pack carries the mode and the bounds");
    const auto t = rastermedium::tint (u, 0.5f);
    check (std::fabs (t[0] - std::exp (-0.75f)) < 1.0e-6f && t[2] == 1.0f, "tint is Beer-Lambert over the path");
    check (rastermedium::tint (u, -3.0f)[0] == 1.0f, "a negative path tints nothing");

    // The paired walk's partner: row 6 holds the partner dielectric's absorption.
    {
        auto pairScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene>();
        pairScene->objects[0].id = { 3 };
        pairScene->objects[0].material = { 8 };
        pairScene->objectCount = 1;
        pairScene->materials[0].id = { 8 };
        pairScene->materials[0].volume = glass;
        pairScene->materialCount = 1;
        const std::array<float, 4> partner { 1.33f, 0.0f, 3.0f, 0.0f };
        check (rastermedium::partnerId (partner).value == 3u, "the partner id is cameraPartner's two halves");
        check (rastermedium::partnerId ({ 1.0f, 0.0f, 5.0f, 2.0f }).value == 5u + (2u << 16u), "the high half shifts by 16");
        auto paired = u;
        rastermedium::packPartner (paired, *pairScene, rastermedium::partnerId (partner));
        check (std::fabs (paired[6][0] - a[0]) < 1.0e-6f && std::fabs (paired[6][1] - a[1]) < 1.0e-6f && paired[6][2] == 0.0f,
               "row 6 carries the partner's absorption");
        check (paired[0] == u[0] && paired[5] == u[5], "the partner leaves the draw's own medium");
        rastermedium::packPartner (paired, *pairScene, {});
        check (paired[6] == std::array<float, 4> {}, "no partner packs zero");
        pairScene->materials[0].volume.volumeOnly = true;
        rastermedium::packPartner (paired, *pairScene, { 3 });
        check (paired[6] == std::array<float, 4> {}, "a Volume Material partner is not a dielectric medium");
    }

    // The paired walk's water body: the water's footprint across its up axis
    // from its top down to the scene's lowest point, and a miss below the water
    // travels inside it. Without authored normals up is the water's local +Y.
    {
        auto pool = std::make_unique<HarmonicMIDI::grid::Visual3DScene>();
        const HarmonicMIDI::grid::SceneVec3 points[] = { { -2.0f, 0.0f, -2.0f }, { 2.0f, 0.5f, 2.0f },
                                                         { -9.0f, -5.0f, -9.0f }, { 9.0f, -5.0f, 9.0f } };
        for (std::size_t i = 0; i < 4; ++i) pool->vertices[i].position = points[i];
        pool->vertexCount = 4;
        pool->objects[0].id = { 4 };
        pool->objects[0].firstVertex = 0;
        pool->objects[0].vertexCount = 2;
        pool->objects[1].id = { 6 };
        pool->objects[1].firstVertex = 2;
        pool->objects[1].vertexCount = 2;
        pool->objectCount = 2;
        const auto identity = [] (const HarmonicMIDI::grid::SceneObjectRecord&) {
            return std::array<float, 16> { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
        };
        rastermedium::Uniforms body {};
        rastermedium::packWaterBody (body, *pool, { 4 }, identity);
        check (body[7] == std::array<float, 4> { 0.0f, 1.0f, 0.0f, -5.0f }
                   && body[8] == std::array<float, 4> { 1.0f, 0.0f, 0.0f, 0.5f }
                   && body[9] == std::array<float, 4> { -2.0f, 2.0f, -2.0f, 2.0f },
               "the water body is the footprint down to the scene floor");
        const float surface[3] = { 0.0f, 0.5f, 0.0f }, down[3] = { 0.0f, -1.0f, 0.0f };
        check (std::fabs (rastermedium::waterExit (body, surface, down) - 5.5f) < 1.0e-5f,
               "a branch leaving straight down travels the body's depth");
        const float slant[3] = { 0.6f, -0.8f, 0.0f };
        check (std::fabs (rastermedium::waterExit (body, surface, slant) - 2.0f / 0.6f) < 1.0e-5f,
               "a slanted branch leaves through the footprint's side");
        const float above[3] = { 0.0f, 3.0f, 0.0f }, up[3] = { 0.0f, 1.0f, 0.0f };
        check (rastermedium::waterExit (body, above, up) == 0.0f, "a branch that misses the body travels none of it");
        rastermedium::packWaterBody (body, *pool, { 9 }, identity);
        check (body[7] == std::array<float, 4> {} && rastermedium::waterExit (body, surface, down) == 0.0f,
               "no water object packs no body");

        // A drag moves the water: the body follows the transform its draw takes.
        const auto dragged = [] (const HarmonicMIDI::grid::SceneObjectRecord& object) {
            const float x = object.id.value == 4u ? 10.0f : 0.0f;
            return std::array<float, 16> { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, 0, 0, 1 };
        };
        rastermedium::packWaterBody (body, *pool, { 4 }, dragged);
        const float moved[3] = { 10.0f, 0.5f, 0.0f };
        check (std::fabs (rastermedium::waterExit (body, moved, down) - 5.5f) < 1.0e-5f
                   && rastermedium::waterExit (body, surface, down) == 0.0f,
               "the body follows the dragged water");

        // Z-up water: authored +Z normals make +Z the body's up axis.
        auto tilted = std::make_unique<HarmonicMIDI::grid::Visual3DScene>();
        const HarmonicMIDI::grid::SceneVec3 flat[] = { { -2.0f, -2.0f, 0.5f }, { 2.0f, 2.0f, 0.5f },
                                                       { -9.0f, -9.0f, -5.0f }, { 9.0f, 9.0f, -5.0f } };
        for (std::size_t i = 0; i < 4; ++i)
        {
            tilted->vertices[i].position = flat[i];
            tilted->vertices[i].normal = { 0.0f, 0.0f, 1.0f };
        }
        tilted->vertexCount = 4;
        tilted->objects[0] = pool->objects[0];
        tilted->objects[1] = pool->objects[1];
        tilted->objectCount = 2;
        rastermedium::packWaterBody (body, *tilted, { 4 }, identity);
        const float zSurface[3] = { 0.0f, 0.0f, 0.5f }, zDown[3] = { 0.0f, 0.0f, -1.0f }, zSlant[3] = { 0.6f, 0.0f, -0.8f };
        check (body[7] == std::array<float, 4> { 0.0f, 0.0f, 1.0f, -5.0f } && body[8][3] == 0.5f,
               "the water's normals give the body its up axis and depth");
        check (std::fabs (rastermedium::waterExit (body, zSurface, zDown) - 5.5f) < 1.0e-5f
                   && std::fabs (rastermedium::waterExit (body, zSurface, zSlant) - 2.0f / 0.6f) < 1.0e-5f,
               "a branch leaves the Z-up body through its floor or its side");

        // A mirrored water keeps its normals' side.
        const auto mirrored = [] (const HarmonicMIDI::grid::SceneObjectRecord&) {
            return std::array<float, 16> { -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
        };
        rastermedium::packWaterBody (body, *tilted, { 4 }, mirrored);
        check (body[7][2] == 1.0f, "a mirror across the water's side leaves up as up");
        check (rastermedium::glslShader().find ("#define GLASS_WATER_EXIT(origin, direction) mediumWaterExit(origin, direction)")
                       != std::string::npos
                   && rastermedium::metalShader().find ("#define GLASS_WATER_EXIT(origin, direction) mediumWaterExit(u, origin, direction)")
                          != std::string::npos,
               "both programs define the walk's water exit");
    }

    // A ray through the unit box spans 2 units.
    float t0 = 0.0f, t1 = 0.0f;
    const float o[3] = { -3.0f, 0.2f, 0.0f }, d[3] = { 1.0f, 0.0f, 0.0f };
    check (rastermedium::boxSpan (u, o, d, t0, t1) && std::fabs (t0 - 2.0f) < 1.0e-6f && std::fabs (t1 - 4.0f) < 1.0e-6f,
           "the box span is the entry and exit distance");
    const float away[3] = { -1.0f, 0.0f, 0.0f };
    check (! rastermedium::boxSpan (u, o, away, t0, t1), "a ray leaving the box misses it");

    // A homogeneous Volume Material: transmittance is exp(-sigma_t L), and an
    // absorbing emitter with no light gives e (1 - exp(-sigma_t L)) / sigma_t.
    HarmonicMIDI::grid::SceneVolume smoke;
    smoke.density = 1.5f;
    smoke.volumeOnly = true;
    smoke.emission = { 2.0f, 0.0f, 1.0f };
    check (rastermedium::modeOf (smoke) == rastermedium::Mode::Volume, "a Volume Material marches");
    const auto v = rastermedium::pack (smoke, { -1.0f, -1.0f, -1.0f }, { 1.0f, 1.0f, 1.0f });
    const auto none = [] (const float*, int& count) { count = 0; return std::array<rastermedium::Light, 8> {}; };
    const auto march = rastermedium::march (v, o, d, 2.0f, 4.0f, { 0.0f, 0.0f, 0.0f }, none);
    const float expected = std::exp (-1.5f * 2.0f);
    check (std::fabs (march.transmittance[0] - expected) < 1.0e-5f, "homogeneous transmittance is Beer-Lambert");
    check (std::fabs (march.radiance[0] - 2.0f * (1.0f - expected) / 1.5f) < 1.0e-4f
               && std::fabs (march.radiance[2] - (1.0f - expected) / 1.5f) < 1.0e-4f,
           "emission integrates exactly in a homogeneous absorber");

    // A lit white scatterer gains radiance toward the light.
    HarmonicMIDI::grid::SceneVolume cloudVolume = smoke;
    cloudVolume.color = { 1.0f, 1.0f, 1.0f };
    cloudVolume.emission = {};
    const auto lit = rastermedium::pack (cloudVolume, { -1.0f, -1.0f, -1.0f }, { 1.0f, 1.0f, 1.0f });
    const auto sun = [] (const float*, int& count)
    {
        count = 1;
        std::array<rastermedium::Light, 8> lights {};
        lights[0] = { { -1.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f } };
        return lights;
    };
    const auto litMarch = rastermedium::march (lit, o, d, 2.0f, 4.0f, { 0.0f, 0.0f, 0.0f }, sun);
    check (litMarch.radiance[1] > 0.0f && std::isfinite (litMarch.radiance[1]), "scene lights scatter into the view");

    // The noise field is Path Trace's field bit for bit.
    HarmonicMIDI::grid::SceneVolume noisy = smoke;
    noisy.field = HarmonicMIDI::grid::SceneVolumeField::Noise;
    noisy.fieldScale = 2.0f;
    noisy.fieldDetail = 3;
    noisy.fieldOffset = { 0.5f, -1.0f, 0.25f };
    const auto n = rastermedium::pack (noisy, {}, {});
    pt::Material cloud = volumeOnly (1.5f, {});
    cloud.volumeField = pt::VolumeDensityField::Noise;
    cloud.volumeFieldScale = 2.0f;
    cloud.volumeFieldDetail = 3;
    cloud.volumeFieldOffset = { 0.5f, -1.0f, 0.25f };
    bool same = true;
    for (int i = 0; i < 64; ++i)
    {
        const float x = 0.37f * float (i), y = -0.11f * float (i), z = 0.23f * float (i);
        same = same && std::fabs (rastermedium::field (n, x, y, z) - pt::volumeFieldAt (cloud, { x, y, z })) < 1.0e-6f;
    }
    check (same, "the Raster noise field matches Path Trace's");
    check (rastermedium::glslShader().find ("uMedium[10]") != std::string::npos, "the GL program declares the medium");
    // The Metal text carries the same march, reading u.medium, and the glass walks' hook.
    const auto metal = rastermedium::metalShader();
    check (metal.find ("#define GLASS_MEDIUM_TINT(distance) mediumTint(u, distance)") != std::string::npos
               && metal.find ("float4 mediumVolume(constant Uniforms& u") != std::string::npos
               && metal.find ("uMedium") == std::string::npos && metal.find ("vec3") == std::string::npos,
           "the Metal program marches u.medium in MSL");
}

// The scene wire keeps main's trailer numbers: the light shape, object
// visibility and lens trailers (9 to 11) come before the media, sky extras and
// camera background trailers (13 to 16), and a scene with all of them round-trips.
void sceneWireOrderTests()
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
    scene.materials[0].volume.density = 1.5f;
    scene.materialCount = 1;
    auto& object = scene.objects[0];
    object.id = grid::SceneObjectId { 3 };
    object.material = scene.materials[0].id;
    object.vertexCount = object.indexCount = 3;
    object.visibility = static_cast<std::uint8_t> (grid::sceneRayVisibility::All & ~grid::sceneRayVisibility::Shadow);
    scene.objectCount = 1;
    scene.cameras[0].id = grid::SceneCameraId { 4 };
    scene.cameraCount = 1;
    scene.activeCamera = scene.cameras[0].id;
    grid::SceneLightRecord light;
    light.id = grid::SceneLightId { 5 };
    light.kind = grid::SceneLightKind::Point;
    light.intensity = 1.0f;
    light.radius = 0.25f;
    scene.lights[0] = light;
    scene.lightCount = 1;
    scene.worldVolume.density = 0.5f;

    const auto bytes = sceneaov::detail::encodeScene (scene);
    const auto at = [&bytes] (std::uint32_t magic)
    {
        for (std::size_t i = 0; i + 4 <= bytes.size(); ++i)
            if (bytes[i] == (magic & 0xff) && bytes[i + 1] == ((magic >> 8) & 0xff)
                && bytes[i + 2] == ((magic >> 16) & 0xff) && bytes[i + 3] == (magic >> 24))
                return static_cast<long> (i);
        return -1L;
    };
    const long shape = at (sceneaov::detail::kLightShapeExtensionMagic);
    const long visibility = at (sceneaov::detail::kObjectVisibilityExtensionMagic);
    const long material = at (sceneaov::detail::kMaterialVolumeExtensionMagic);
    const long world = at (sceneaov::detail::kWorldVolumeExtensionMagic);
    check (shape > 0 && visibility > shape && material > visibility && world > material,
           "lights' trailers precede the media trailers on the wire");
    grid::Visual3DScene decoded;
    check (sceneaov::detail::decodeScene (bytes, decoded), "a scene with light and media trailers decodes");
    check (grid::sameSceneVolume (decoded.materials[0].volume, scene.materials[0].volume)
               && grid::sameSceneVolume (decoded.worldVolume, scene.worldVolume)
               && decoded.lights[0].radius == 0.25f && decoded.objects[0].visibility == object.visibility,
           "both branches' fields survive the wire");
    check (sceneaov::detail::encodeScene (decoded) == bytes, "the decoded scene encodes to the same bytes");
}

int main()
{
    transmittanceTests();
    phaseTests();
    distanceSamplingTests();
    emissionTests();
    furnaceTests();
    bounceLimitTests();
    fogLightTests();
    sceneRecordTests();
    renderWireTests();
    realtimeFogTests();
    rasterMediumTests();
    sceneWireOrderTests();
    std::cout << checks - failures << " of " << checks << " checks passed\n";
    return failures == 0 ? 0 : 1;
}
