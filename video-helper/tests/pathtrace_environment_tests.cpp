// Tests for HDR environment image lighting in the CPU path tracer (Plan 3,
// PT-B): the raster lookup convention, the power distribution, light-set
// wiring, renders against the constant environment and a quadrature
// reference, MIS consistency, and pixels shared from the helper's resident
// store.

#include "pathtrace/accumulator.h"
#include "pathtrace/bsdf.h"
#include "pathtrace/bvh.h"
#include "pathtrace/environment_map.h"
#include "pathtrace/environment_resident.h"
#include "pathtrace/integrator.h"
#include "pathtrace/lights.h"
#include "pathtrace/scene_ingest.h"
#include "pathtrace/thread_pool.h"
#include "environment_image_payload_transport.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace
{

namespace pt = videohelper::pathtrace;
namespace grid = HarmonicMIDI::grid;
using pt::Vec2;
using pt::Vec3;

constexpr double kPiD = 3.14159265358979323846;

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

std::string str (double value) { return std::to_string (value); }

float u01 (std::mt19937& rng)
{
    return static_cast<float> (static_cast<std::uint32_t> (rng()) >> 8) * (1.0f / 16777216.0f);
}

float meanOf (Vec3 v) { return (v.x + v.y + v.z) / 3.0f; }
double luminance (Vec3 v) { return 0.2126 * v.x + 0.7152 * v.y + 0.0722 * v.z; }

Vec3 uniformSphere (std::mt19937& rng)
{
    const float z = 1.0f - 2.0f * u01 (rng);
    const float r = std::sqrt (std::max (0.0f, 1.0f - z * z));
    const float phi = 2.0f * pt::kPi * u01 (rng);
    return { r * std::cos (phi), r * std::sin (phi), z };
}

bool closeTo (Vec3 a, Vec3 b, float tolerance)
{
    return std::fabs (a.x - b.x) <= tolerance && std::fabs (a.y - b.y) <= tolerance && std::fabs (a.z - b.z) <= tolerance;
}

visualenvironmentimage::Identity testAsset (std::uint64_t version = 1)
{
    return { "pathtrace-environment-test", version, std::string (64, 'a') };
}

// Texels filled by a function of the texel-centre direction at orientation 0
// (+Z at u = 0.5, row 0 at +Y), the convention under test.
pt::EnvironmentPixels makePixels (std::uint32_t width, std::uint32_t height, const std::function<Vec3 (Vec3, std::uint32_t, std::uint32_t)>& texel)
{
    std::vector<float> rgb;
    rgb.reserve (static_cast<std::size_t> (width) * height * 3);
    for (std::uint32_t y = 0; y < height; ++y)
        for (std::uint32_t x = 0; x < width; ++x)
        {
            const double longitude = ((x + 0.5) / width - 0.5) * 2.0 * kPiD;
            const double polar = (y + 0.5) / height * kPiD;
            const Vec3 d { static_cast<float> (std::sin (polar) * std::sin (longitude)), static_cast<float> (std::cos (polar)),
                           static_cast<float> (std::sin (polar) * std::cos (longitude)) };
            const Vec3 value = texel (d, x, y);
            rgb.insert (rgb.end(), { value.x, value.y, value.z });
        }
    return pt::copyEnvironmentPixels (width, height, rgb.data(), testAsset());
}

pt::EnvironmentPixels constantPixels (std::uint32_t width, std::uint32_t height, Vec3 value)
{
    return makePixels (width, height, [value] (Vec3, std::uint32_t, std::uint32_t) { return value; });
}

// A smooth sky: gradient, warm lobe, dark ground.
pt::EnvironmentPixels smoothSky (std::uint32_t width, std::uint32_t height)
{
    const Vec3 lobe = pt::normalizeOrZero ({ 1.0f, 1.0f, 0.5f });
    return makePixels (width, height, [lobe] (Vec3 d, std::uint32_t, std::uint32_t) {
        const float up = std::max (0.0f, d.y);
        const float glow = std::pow (std::max (0.0f, pt::dot (d, lobe)), 8.0f);
        const Vec3 sky = Vec3 { 0.25f, 0.35f, 0.55f } + Vec3 { 0.3f, 0.3f, 0.3f } * up;
        const Vec3 ground { 0.08f, 0.07f, 0.06f };
        return (d.y >= 0.0f ? sky : ground) + Vec3 { 3.0f, 2.6f, 2.0f } * glow;
    });
}

grid::SceneLightRecord imageRecord (const visualenvironmentimage::Identity& asset, float orientationDegrees = 0.0f,
                                    float exposureStops = 0.0f, Vec3 color = { 1.0f, 1.0f, 1.0f }, float intensity = 1.0f)
{
    grid::SceneLightRecord light;
    light.kind = grid::SceneLightKind::Environment;
    light.color = { color.x, color.y, color.z };
    light.intensity = intensity;
    light.environmentImage.version = 1;
    light.environmentImage.asset = asset;
    light.environmentImage.orientationDegrees = orientationDegrees;
    light.environmentImage.exposureStops = exposureStops;
    return light;
}

grid::SceneLightRecord constantRecord (float radiance)
{
    grid::SceneLightRecord light;
    light.kind = grid::SceneLightKind::Environment;
    light.color = { radiance, radiance, radiance };
    light.intensity = 1.0f;
    return light;
}

// Literal C++ transcription of environmentImageRadiance in
// src/gpu_backend/fixture_environment_image.h (shader lines 116-129), kept
// apart from EnvironmentMap so the two are compared, not shared.
Vec3 gpuReference (const pt::EnvironmentPixels& image, Vec3 tint, float orientation, Vec3 direction)
{
    const float len = std::sqrt (pt::dot (direction, direction));
    const Vec3 d = direction * (1.0f / len);
    const float longitude = d.x * d.x + d.z * d.z > 1.0e-12f ? std::atan2 (d.x, d.z) : 0.0f;
    const float wrapped = (longitude - orientation) / 6.283185307179586f + 0.5f;
    const float u = wrapped - std::floor (wrapped);
    const float v = std::acos (std::min (1.0f, std::max (-1.0f, d.y))) / 3.141592653589793f;
    const int w = static_cast<int> (image.width);
    const int h = static_cast<int> (image.height);
    const float px = u * static_cast<float> (w) - 0.5f;
    const float py = v * static_cast<float> (h) - 0.5f;
    const int ax = static_cast<int> (std::floor (px));
    const int ay = static_cast<int> (std::floor (py));
    const float fx = px - std::floor (px);
    const float fy = py - std::floor (py);
    const int x0 = (ax % w + w) % w;
    const int x1 = (x0 + 1) % w;
    const int y0 = std::min (std::max (ay, 0), h - 1);
    const int y1 = std::min (std::max (ay + 1, 0), h - 1);
    const auto read = [&] (int x, int y) {
        const auto& t = (*image.texels)[static_cast<std::size_t> (y * w + x)];
        return Vec3 { t[0], t[1], t[2] };
    };
    const auto mix = [] (Vec3 a, Vec3 b, float t) { return a * (1.0f - t) + b * t; };
    return mix (mix (read (x0, y0), read (x1, y0), fx), mix (read (x0, y1), read (x1, y1), fx), fy) * tint;
}

// ---------------------------------------------------------------- convention

void conventionTests()
{
    std::mt19937 rng (11);
    std::vector<float> rgb (16 * 8 * 3);
    for (float& v : rgb)
        v = 4.0f * u01 (rng);
    const pt::EnvironmentPixels pixels = pt::copyEnvironmentPixels (16, 8, rgb.data(), testAsset());
    check (pixels.valid(), "copied pixels are valid");

    const Vec3 tint { 0.5f, 2.0f, 1.25f };
    for (const float orientation : { 0.0f, 0.7f, -2.5f, 1.5707964f, 6.2831855f })
    {
        pt::EnvironmentMap map;
        check (map.build (pixels, tint, orientation), "a random map builds");
        int mismatches = 0;
        for (int i = 0; i < 4000; ++i)
        {
            Vec3 d = uniformSphere (rng);
            if (i % 10 == 0)
                d = pt::normalizeOrZero ({ d.x * 1.0e-4f, d.y, d.z * 1.0e-4f });  // near the poles
            if (i % 10 == 1)
                d = pt::normalizeOrZero ({ 1.0e-4f * d.x, d.y, -1.0f });         // near the -Z seam
            const Vec3 reference = gpuReference (pixels, tint, orientation, d);
            const Vec3 traced = map.radiance (d);
            if (! closeTo (traced, reference, 1.0e-5f * (1.0f + std::max (reference.x, std::max (reference.y, reference.z)))))
                ++mismatches;
        }
        check (mismatches == 0, "the tracer lookup matches the raster shader at orientation " + str (orientation)
                                    + ", mismatches " + std::to_string (mismatches));
    }

    // Anchors of the convention.
    pt::EnvironmentMap map;
    map.build (pixels, { 1.0f, 1.0f, 1.0f }, 0.0f);
    check (closeTo (map.uvToDirection ({ 0.5f, 0.5f }), { 0.0f, 0.0f, 1.0f }, 1.0e-6f), "u = 0.5 on the equator looks along +Z");
    check (closeTo (map.uvToDirection ({ 0.75f, 0.5f }), { 1.0f, 0.0f, 0.0f }, 1.0e-6f), "u = 0.75 looks along +X");
    check (std::fabs (map.uvToDirection ({ 0.3f, 0.0f }).y - 1.0f) < 1.0e-6f, "v = 0 is the zenith");
    const Vec2 uvX = map.directionToUv ({ 1.0f, 0.0f, 0.0f });
    check (std::fabs (uvX.x - 0.75f) < 1.0e-6f && std::fabs (uvX.y - 0.5f) < 1.0e-6f, "+X maps to (0.75, 0.5)");
    for (std::uint32_t y = 0; y < 8; ++y)
        for (std::uint32_t x = 0; x < 16; ++x)
        {
            const Vec3 centre = map.uvToDirection ({ (x + 0.5f) / 16.0f, (y + 0.5f) / 8.0f });
            const auto& t = (*pixels.texels)[y * 16 + x];
            if (! closeTo (map.radiance (centre), { t[0], t[1], t[2] }, 1.0e-3f))
                check (false, "texel centre (" + std::to_string (x) + ", " + std::to_string (y) + ") reads its own texel");
        }

    // Tint and orientation of a record, as the raster uniforms form them.
    const grid::SceneLightRecord record = imageRecord (testAsset(), 90.0f, -2.0f, { 0.5f, 0.75f, 1.0f }, 2.0f);
    const Vec3 recordTint = pt::environmentImageTint (record);
    check (closeTo (recordTint, { 0.25f, 0.375f, 0.5f }, 1.0e-7f), "tint is color * intensity * 2^exposure");
    check (std::fabs (pt::environmentImageOrientationRadians (record) - 1.5707964f) < 1.0e-6f, "orientation is in radians");

    // Refused inputs.
    pt::EnvironmentMap refused;
    check (! refused.build ({}, { 1.0f, 1.0f, 1.0f }, 0.0f) && ! refused.active(), "missing pixels are refused");
    check (! refused.build (constantPixels (8, 4, {}), { 1.0f, 1.0f, 1.0f }, 0.0f), "a black image is refused");
    check (! refused.build (pixels, { -1.0f, 1.0f, 1.0f }, 0.0f), "a negative tint is refused");
    std::vector<float> bad (8 * 4 * 3, 1.0f);
    bad[7] = std::numeric_limits<float>::quiet_NaN();
    check (! refused.build (pt::copyEnvironmentPixels (8, 4, bad.data()), { 1.0f, 1.0f, 1.0f }, 0.0f),
           "a NaN texel is refused");
}

// ---------------------------------------------------------------- distribution

void distributionTests()
{
    const pt::EnvironmentPixels pixels = smoothSky (64, 32);
    pt::EnvironmentMap map;
    check (map.build (pixels, { 1.0f, 1.0f, 1.0f }, 0.4f), "the smooth sky builds");

    // Exact: the pdf is constant per cell, so cell pdf times cell solid angle
    // sums to one.
    double exact = 0.0;
    for (std::uint32_t y = 0; y < 32; ++y)
        for (std::uint32_t x = 0; x < 64; ++x)
            exact += static_cast<double> (map.pdf (map.uvToDirection ({ (x + 0.5f) / 64.0f, (y + 0.5f) / 32.0f })))
                   * map.cellSolidAngle (y);
    check (std::fabs (exact - 1.0) < 1.0e-4, "cell probabilities sum to one: " + str (exact));

    // Monte Carlo over the sphere.
    std::mt19937 rng (3);
    double integral = 0.0;
    const int n = 400000;
    for (int i = 0; i < n; ++i)
        integral += map.pdf (uniformSphere (rng));
    integral *= 4.0 * kPiD / n;
    check (std::fabs (integral - 1.0) < 0.02, "the pdf integrates to one over the sphere: " + str (integral));

    // sample() reports the density pdf() gives, and directions land where the
    // pdf says they should (8 x 4 coarse bins of 8 x 8 cells).
    const int samples = 200000;
    int mismatched = 0;
    int failed = 0;
    std::vector<double> observed (32, 0.0);
    for (int i = 0; i < samples; ++i)
    {
        float pdf = 0.0f;
        const Vec3 d = map.sample ({ u01 (rng), u01 (rng) }, pdf);
        if (! (pdf > 0.0f) || std::fabs (pt::length (d) - 1.0f) > 1.0e-4f)
        {
            ++failed;
            continue;
        }
        if (std::fabs (pdf - map.pdf (d)) > 1.0e-3f * pdf)
            ++mismatched;
        const Vec2 uv = map.directionToUv (d);
        const int bx = std::min (7, static_cast<int> (uv.x * 8.0f));
        const int by = std::min (3, static_cast<int> (uv.y * 4.0f));
        observed[static_cast<std::size_t> (by * 8 + bx)] += 1.0;
    }
    check (failed == 0, "every sample succeeds with a unit direction, failures " + std::to_string (failed));
    check (mismatched <= samples / 1000, "sample pdf agrees with pdf(direction), mismatches " + std::to_string (mismatched));
    bool binsAgree = true;
    for (int by = 0; by < 4; ++by)
        for (int bx = 0; bx < 8; ++bx)
        {
            double expected = 0.0;
            for (std::uint32_t y = static_cast<std::uint32_t> (by * 8); y < static_cast<std::uint32_t> (by * 8 + 8); ++y)
                for (std::uint32_t x = static_cast<std::uint32_t> (bx * 8); x < static_cast<std::uint32_t> (bx * 8 + 8); ++x)
                    expected += static_cast<double> (map.pdf (map.uvToDirection ({ (x + 0.5f) / 64.0f, (y + 0.5f) / 32.0f })))
                              * map.cellSolidAngle (y);
            const double fraction = observed[static_cast<std::size_t> (by * 8 + bx)] / samples;
            const double sigma = std::sqrt (std::max (expected * (1.0 - expected), 1.0e-12) / samples);
            if (std::fabs (fraction - expected) > 5.0 * sigma + 1.0e-4)
            {
                binsAgree = false;
                std::cerr << "  bin " << bx << ", " << by << ": observed " << fraction << ", expected " << expected << '\n';
            }
        }
    check (binsAgree, "sample frequencies follow the pdf");

    // Integrated radiance: exact for a constant map, close to quadrature for the sky.
    pt::EnvironmentMap flat;
    flat.build (constantPixels (32, 16, { 0.5f, 1.0f, 2.0f }), { 1.0f, 1.0f, 1.0f }, 0.0f);
    const Vec3 flatIntegral = flat.integratedRadiance();
    check (std::fabs (flatIntegral.x - 2.0f * pt::kPi) < 1.0e-4f && std::fabs (flatIntegral.z - 8.0f * pt::kPi) < 1.0e-3f,
           "a constant map integrates to 4 pi times its radiance");
    double quadrature = 0.0;
    const int nu = 512;
    const int nv = 256;
    for (int j = 0; j < nv; ++j)
        for (int i = 0; i < nu; ++i)
        {
            const float z = 1.0f - 2.0f * (j + 0.5f) / nv;
            const float r = std::sqrt (std::max (0.0f, 1.0f - z * z));
            const float phi = 2.0f * pt::kPi * (i + 0.5f) / nu;
            quadrature += meanOf (map.radiance ({ r * std::cos (phi), z, r * std::sin (phi) }));
        }
    quadrature *= 4.0 * kPiD / (nu * nv);
    const double integrated = meanOf (map.integratedRadiance());
    check (std::fabs (integrated - quadrature) < 0.01 * quadrature,
           "integrated radiance " + str (integrated) + " matches quadrature " + str (quadrature));
}

// Fraction of samples in the 3 x 3 cells around a bright texel equals that
// region's share of the luminance integral, measured by fine quadrature of
// the filtered radiance (not from the map's own cell weights).
void brightTexelTests()
{
    const std::uint32_t w = 32;
    const std::uint32_t h = 16;
    const std::uint32_t bx = 5;
    const std::uint32_t by = 6;
    for (const float background : { 0.05f, 0.0f })
    {
        const pt::EnvironmentPixels pixels = makePixels (w, h, [&] (Vec3, std::uint32_t x, std::uint32_t y) {
            const float v = x == bx && y == by ? 20.0f : background;
            return Vec3 { v, v, v };
        });
        pt::EnvironmentMap map;
        check (map.build (pixels, { 1.0f, 1.0f, 1.0f }, 0.0f), "the bright-texel map builds");

        double block = 0.0;
        double total = 0.0;
        const int sub = 8;
        for (std::uint32_t y = 0; y < h; ++y)
            for (std::uint32_t x = 0; x < w; ++x)
            {
                double cell = 0.0;
                const double c0 = std::cos (kPiD * y / h);
                const double c1 = std::cos (kPiD * (y + 1) / h);
                for (int sy = 0; sy < sub; ++sy)
                    for (int sx = 0; sx < sub; ++sx)
                    {
                        // Uniform in solid angle inside the cell.
                        const double cosPolar = c0 + (c1 - c0) * (sy + 0.5) / sub;
                        const double v = std::acos (cosPolar) / kPiD;
                        const double u = (x + (sx + 0.5) / sub) / w;
                        cell += luminance (map.radiance (map.uvToDirection ({ static_cast<float> (u), static_cast<float> (v) })));
                    }
                cell *= map.cellSolidAngle (y) / (sub * sub);
                total += cell;
                if (std::abs (static_cast<int> (x) - static_cast<int> (bx)) <= 1 && std::abs (static_cast<int> (y) - static_cast<int> (by)) <= 1)
                    block += cell;
            }
        const double expected = block / total;

        std::mt19937 rng (17);
        const int samples = 200000;
        int inside = 0;
        for (int i = 0; i < samples; ++i)
        {
            float pdf = 0.0f;
            const Vec3 d = map.sample ({ u01 (rng), u01 (rng) }, pdf);
            const Vec2 uv = map.directionToUv (d);
            const int x = std::min (static_cast<int> (w) - 1, static_cast<int> (uv.x * w));
            const int y = std::min (static_cast<int> (h) - 1, static_cast<int> (uv.y * h));
            if (pdf > 0.0f && std::abs (x - static_cast<int> (bx)) <= 1 && std::abs (y - static_cast<int> (by)) <= 1)
                ++inside;
        }
        const double fraction = static_cast<double> (inside) / samples;
        check (std::fabs (fraction - expected) <= 0.01,
               "bright texel on background " + str (background) + ": sampled share " + str (fraction)
                   + " vs power share " + str (expected));
        if (background == 0.0f)
            check (inside == samples, "with a black background every sample lands next to the bright texel");
    }
}

// ---------------------------------------------------------------- orientation

void orientationTests()
{
    // A bright 2 x 2 patch centred on u = 0.5, v = 0.5: +Z before rotation.
    const pt::EnvironmentPixels pixels = makePixels (32, 16, [] (Vec3, std::uint32_t x, std::uint32_t y) {
        const float v = (x == 15 || x == 16) && (y == 7 || y == 8) ? 10.0f : 0.0f;
        return Vec3 { v, v, v };
    });
    pt::EnvironmentMap still;
    pt::EnvironmentMap turned;
    pt::EnvironmentMap back;
    still.build (pixels, { 1.0f, 1.0f, 1.0f }, 0.0f);
    turned.build (pixels, { 1.0f, 1.0f, 1.0f }, 90.0f * 0.01745329251994329577f);
    back.build (pixels, { 1.0f, 1.0f, 1.0f }, -90.0f * 0.01745329251994329577f);
    const Vec3 px { 1.0f, 0.0f, 0.0f };
    const Vec3 pz { 0.0f, 0.0f, 1.0f };
    const Vec3 nx { -1.0f, 0.0f, 0.0f };
    check (still.radiance (pz).x > 9.0f && still.radiance (px).x == 0.0f, "orientation 0: the patch is at +Z");
    check (turned.radiance (px).x > 9.0f && turned.radiance (pz).x == 0.0f, "orientation +90: the patch moves to +X");
    check (back.radiance (nx).x > 9.0f && back.radiance (pz).x == 0.0f, "orientation -90: the patch moves to -X");

    // A +90 degree turn about +Y takes (x, y, z) to (z, y, -x).
    std::mt19937 rng (23);
    int mismatches = 0;
    for (int i = 0; i < 4000; ++i)
    {
        const Vec3 d = uniformSphere (rng);
        if (! closeTo (turned.radiance ({ d.z, d.y, -d.x }), still.radiance (d), 1.0e-3f))
            ++mismatches;
        if (! closeTo (turned.radiance (d), gpuReference (pixels, { 1.0f, 1.0f, 1.0f }, 90.0f * 0.01745329251994329577f, d), 1.0e-4f))
            ++mismatches;
    }
    check (mismatches == 0, "rotation matches the raster convention, mismatches " + std::to_string (mismatches));

    Vec3 mean {};
    for (int i = 0; i < 2000; ++i)
    {
        float pdf = 0.0f;
        mean = mean + turned.sample ({ u01 (rng), u01 (rng) }, pdf);
    }
    check (pt::dot (pt::normalizeOrZero (mean), px) > 0.95f, "samples of the turned map point toward +X");

    // Through a scene record: orientationDegrees reaches the light set.
    pt::RenderScene scene;
    scene.lightRecords = { imageRecord (testAsset(), 90.0f) };
    scene.environmentImage = pixels;
    scene.commit();
    check (scene.lights.environmentRadiance (px).x > 9.0f && scene.lights.environmentRadiance (pz).x == 0.0f,
           "a record's orientation turns the lit environment");
}

// ---------------------------------------------------------------- light set

void lightSetTests()
{
    const pt::EnvironmentPixels sky = smoothSky (64, 32);

    pt::RenderScene scene;
    scene.lightRecords = { constantRecord (0.3f), imageRecord (testAsset(), 0.0f, 1.0f) };
    scene.ambient = { 0.01f, 0.02f, 0.03f };
    scene.environmentImage = sky;
    scene.commit();
    const pt::LightSetStats& stats = scene.lights.stats();
    check (stats.environmentImages == 1 && stats.ignoredEnvironmentImages == 0, "a usable image lights the scene");
    check (stats.supersededEnvironments == 1 && stats.environment == 0, "the image replaces the constant record");
    pt::EnvironmentMap reference;
    reference.build (sky, { 2.0f, 2.0f, 2.0f }, 0.0f);
    const Vec3 d = pt::normalizeOrZero ({ 0.3f, 0.5f, -0.2f });
    check (closeTo (scene.lights.environmentRadiance (d), reference.radiance (d) + scene.ambient, 1.0e-5f),
           "radiance is the exposed image plus the ambient, without the replaced constant");
    const pt::EnvironmentField& field = scene.lights.environment();
    check (field.image != nullptr && field.imageSampleProbability > 0.05f && field.imageSampleProbability < 1.0f,
           "ambient keeps a uniform share of environment samples");

    // Mixture pdf normalization and sample consistency with the ambient term.
    std::mt19937 rng (29);
    double integral = 0.0;
    for (int i = 0; i < 200000; ++i)
        integral += field.pdf (uniformSphere (rng));
    integral *= 4.0 * kPiD / 200000;
    check (std::fabs (integral - 1.0) < 0.02, "the mixture pdf integrates to one: " + str (integral));
    bool consistent = true;
    for (int i = 0; i < 2000; ++i)
    {
        float pdf = 0.0f;
        const Vec3 s = field.sample ({ u01 (rng), u01 (rng) }, pdf);
        consistent = consistent && pdf > 0.0f && std::fabs (pdf - field.pdf (s)) <= 1.0e-4f * pdf;
    }
    check (consistent, "mixture samples report their pdf");

    // Selection power: a directional light against the image.
    pt::RenderScene mixed;
    grid::SceneLightRecord sun;
    sun.kind = grid::SceneLightKind::Directional;
    sun.color = { 1.0f, 1.0f, 1.0f };
    sun.intensity = 2.0f;
    mixed.lightRecords = { sun, imageRecord (testAsset()) };
    mixed.environmentImage = sky;
    mixed.commit();
    check (mixed.lights.size() == 2, "a directional light and the image");
    pt::EnvironmentMap plain;
    plain.build (sky, { 1.0f, 1.0f, 1.0f }, 0.0f);
    // Directional: average(strength) * disk. Environment: pi * mean radiance * disk.
    const double expectedRatio = (meanOf (plain.integratedRadiance()) / 4.0) / 2.0;
    const double ratio = mixed.lights.selectionProbability (1) / mixed.lights.selectionProbability (0);
    check (std::fabs (ratio - expectedRatio) < 1.0e-3 * expectedRatio,
           "the image's selection weight is its integrated radiance: " + str (ratio) + " vs " + str (expectedRatio));

    // Unusable images light nothing and are counted.
    const auto ignoredCount = [] (const std::vector<grid::SceneLightRecord>& records, const pt::EnvironmentPixels& pixels) {
        pt::RenderScene s;
        s.lightRecords = records;
        s.environmentImage = pixels;
        s.commit();
        return std::make_pair (s.lights.stats().ignoredEnvironmentImages, s.lights.hasEnvironment());
    };
    const auto noPixels = ignoredCount ({ imageRecord (testAsset()) }, {});
    check (noPixels.first == 1 && ! noPixels.second, "an image without pixels is ignored");
    const auto stale = ignoredCount ({ imageRecord (testAsset (2)) }, sky);
    check (stale.first == 1 && ! stale.second, "pixels of another asset version are ignored");
    const auto twice = ignoredCount ({ imageRecord (testAsset()), imageRecord (testAsset()) }, sky);
    check (twice.first == 1 && twice.second, "a second image record is ignored, the first lights");
    const auto hot = ignoredCount ({ imageRecord (testAsset(), 0.0f, 16.0f, { 64.0f, 64.0f, 64.0f }, 1000.0f) }, sky);
    check (hot.first == 1 && ! hot.second, "an exposure past the raster's radiance bound is ignored, as the raster refuses it");
}

// ---------------------------------------------------------------- renders

struct Mesh
{
    std::vector<Vec3> vertices;
    std::vector<std::uint32_t> indices;
};

Mesh uvSphere (std::uint32_t stacks, std::uint32_t slices, float radius)
{
    Mesh mesh;
    mesh.vertices.push_back ({ 0.0f, radius, 0.0f });
    for (std::uint32_t i = 1; i < stacks; ++i)
    {
        const float theta = pt::kPi * static_cast<float> (i) / static_cast<float> (stacks);
        for (std::uint32_t j = 0; j < slices; ++j)
        {
            const float phi = 2.0f * pt::kPi * static_cast<float> (j) / static_cast<float> (slices);
            mesh.vertices.push_back ({ radius * std::sin (theta) * std::cos (phi), radius * std::cos (theta),
                                       radius * std::sin (theta) * std::sin (phi) });
        }
    }
    mesh.vertices.push_back ({ 0.0f, -radius, 0.0f });
    const auto ring = [slices] (std::uint32_t i, std::uint32_t j) { return 1 + (i - 1) * slices + j % slices; };
    const auto south = static_cast<std::uint32_t> (mesh.vertices.size() - 1);
    for (std::uint32_t j = 0; j < slices; ++j)
    {
        mesh.indices.insert (mesh.indices.end(), { 0u, ring (1, j + 1), ring (1, j) });
        for (std::uint32_t i = 1; i + 1 < stacks; ++i)
        {
            mesh.indices.insert (mesh.indices.end(), { ring (i, j), ring (i, j + 1), ring (i + 1, j) });
            mesh.indices.insert (mesh.indices.end(), { ring (i, j + 1), ring (i + 1, j + 1), ring (i + 1, j) });
        }
        mesh.indices.insert (mesh.indices.end(), { ring (stacks - 1, j), ring (stacks - 1, j + 1), south });
    }
    return mesh;
}

// Rectangle centre +- halfU +- halfV, front face along cross(halfU, halfV).
Mesh quad (Vec3 centre, Vec3 halfU, Vec3 halfV)
{
    Mesh mesh;
    mesh.vertices = { centre - halfU - halfV, centre + halfU - halfV, centre + halfU + halfV, centre - halfU + halfV };
    mesh.indices = { 0, 1, 2, 0, 2, 3 };
    return mesh;
}

// Flat-shaded triangles; with outwardFrom set, front faces point away from it.
void appendMesh (pt::RenderScene& scene, const Mesh& mesh, std::uint32_t material, const Vec3* outwardFrom = nullptr)
{
    for (std::size_t i = 0; i + 2 < mesh.indices.size(); i += 3)
    {
        Vec3 p0 = mesh.vertices[mesh.indices[i]];
        Vec3 p1 = mesh.vertices[mesh.indices[i + 1]];
        Vec3 p2 = mesh.vertices[mesh.indices[i + 2]];
        Vec3 n = pt::normalizeOrZero (pt::cross (p1 - p0, p2 - p0));
        if (pt::dot (n, n) == 0.0f)
            continue;
        if (outwardFrom != nullptr && pt::dot (n, (p0 + p1 + p2) * (1.0f / 3.0f) - *outwardFrom) < 0.0f)
        {
            std::swap (p1, p2);
            n = -n;
        }
        pt::TriangleAttributes attributes;
        attributes.materialIndex = material;
        attributes.primitiveIndex = static_cast<std::uint32_t> (i / 3);
        attributes.geometricNormal = n;
        attributes.shadingNormals = { n, n, n };
        attributes.hasShadingNormals = false;
        for (const Vec3 p : { p0, p1, p2 })
        {
            scene.geometry.positions.push_back (p);
            scene.geometry.bounds.grow (p);
        }
        scene.geometry.triangles.push_back (attributes);
    }
}

pt::PinholeCamera makeCamera (Vec3 position, pt::Quat rotation, float tanHalfFov, std::uint32_t width, std::uint32_t height)
{
    pt::PinholeCamera camera;
    camera.position = position;
    camera.rotation = rotation;
    camera.tanHalfFov = tanHalfFov;
    camera.aspect = static_cast<float> (width) / static_cast<float> (height);
    camera.nearPlane = 0.01f;
    camera.farPlane = 1000.0f;
    camera.width = width;
    camera.height = height;
    return camera;
}

Vec3 pixel (const pt::Accumulator& accumulator, std::uint32_t x, std::uint32_t y)
{
    const pt::Rgb v = accumulator.mean (x, y);
    return { v.r, v.g, v.b };
}

double imageMean (const pt::Accumulator& accumulator)
{
    double sum = 0.0;
    for (std::uint32_t y = 0; y < accumulator.height(); ++y)
        for (std::uint32_t x = 0; x < accumulator.width(); ++x)
            sum += meanOf (pixel (accumulator, x, y));
    return sum / (static_cast<double> (accumulator.width()) * accumulator.height());
}

bool hits (const pt::RenderScene& scene, const pt::PinholeCamera& camera, float x, float y, pt::Hit& hit)
{
    return scene.bvh.intersect (camera.generateRay (x, y), hit);
}

// Every corner and the centre of the pixel hit the scene.
bool fullyCovered (const pt::RenderScene& scene, const pt::PinholeCamera& camera, std::uint32_t x, std::uint32_t y)
{
    pt::Hit hit;
    for (const Vec2 o : { Vec2 { 0.0f, 0.0f }, Vec2 { 1.0f, 0.0f }, Vec2 { 0.0f, 1.0f }, Vec2 { 1.0f, 1.0f }, Vec2 { 0.5f, 0.5f } })
        if (! hits (scene, camera, static_cast<float> (x) + o.x, static_cast<float> (y) + o.y, hit))
            return false;
    return true;
}

// A constant image renders the same picture as the PT-A2 constant environment.
void constantImageTests()
{
    const auto render = [] (bool useImage, double& covered, Vec3& background) {
        pt::RenderScene scene;
        pt::Material grey;
        grey.baseColor = { 0.7f, 0.7f, 0.7f };
        scene.materials.push_back (grey);
        const Vec3 centre {};
        appendMesh (scene, uvSphere (24, 48, 1.0f), 0, &centre);
        if (useImage)
        {
            scene.lightRecords = { imageRecord (testAsset()) };
            scene.environmentImage = constantPixels (16, 8, { 0.5f, 0.5f, 0.5f });
        }
        else
            scene.lightRecords = { constantRecord (0.5f) };
        scene.commit();
        const pt::PinholeCamera camera = makeCamera ({ 0.0f, 0.0f, 3.5f }, pt::Quat {}, 0.4f, 24, 24);
        pt::PathTracer tracer (4);
        const pt::CancellationToken cancel {};
        const pt::RenderStats stats = tracer.renderFixed (scene, camera, 24, 24, 64, 1, cancel);
        check (stats.nonFiniteSamples == 0 && ! stats.invalidRequest, "constant comparison renders cleanly");
        check (stats.lights.environmentImages == (useImage ? 1u : 0u), "the expected environment kind is in use");
        double sum = 0.0;
        int count = 0;
        for (std::uint32_t y = 0; y < 24; ++y)
            for (std::uint32_t x = 0; x < 24; ++x)
                if (fullyCovered (scene, camera, x, y))
                {
                    sum += meanOf (pixel (tracer.accumulator(), x, y));
                    ++count;
                }
        covered = count > 0 ? sum / count : 0.0;
        background = pixel (tracer.accumulator(), 0, 0);
    };
    double constantMean = 0.0;
    double imageMeanValue = 0.0;
    Vec3 constantBackground;
    Vec3 imageBackground;
    render (false, constantMean, constantBackground);
    render (true, imageMeanValue, imageBackground);
    check (constantMean > 0.2, "the sphere is lit: " + str (constantMean));
    check (std::fabs (imageMeanValue - constantMean) <= 0.015 * constantMean,
           "a constant image matches the constant environment: " + str (imageMeanValue) + " vs " + str (constantMean));
    check (closeTo (imageBackground, constantBackground, 1.0e-5f) && std::fabs (imageBackground.x - 0.5f) < 1.0e-5f,
           "both backgrounds show radiance 0.5");
}

// White furnace under an HDRI: a convex white sphere has no interreflection,
// so each pixel converges to the hemisphere average of the environment
// weighted by its BSDF, computed here by quadrature.
void hdriFurnaceTests()
{
    pt::RenderScene scene;
    pt::Material white;  // base colour 1, roughness 1, dielectric
    scene.materials.push_back (white);
    const Vec3 centre {};
    appendMesh (scene, uvSphere (32, 64, 1.0f), 0, &centre);
    scene.lightRecords = { imageRecord (testAsset(), 30.0f) };
    scene.environmentImage = smoothSky (64, 32);
    scene.commit();
    check (scene.lights.stats().environmentImages == 1, "the furnace uses the image");

    const std::uint32_t size = 24;
    const pt::PinholeCamera camera = makeCamera ({ 0.0f, 0.0f, 3.5f }, pt::Quat {}, 0.4f, size, size);
    pt::PathTracer tracer (4);
    tracer.settings.maxDepth = 8;
    const pt::CancellationToken cancel {};
    const pt::RenderStats stats = tracer.renderFixed (scene, camera, size, size, 128, 7, cancel);
    check (stats.nonFiniteSamples == 0, "the HDRI furnace renders finite samples");

    // Quadrature directions, equal solid angle each.
    const int nu = 192;
    const int nv = 96;
    std::vector<Vec3> directions;
    std::vector<Vec3> radiance;
    for (int j = 0; j < nv; ++j)
        for (int i = 0; i < nu; ++i)
        {
            const float z = 1.0f - 2.0f * (j + 0.5f) / nv;
            const float r = std::sqrt (std::max (0.0f, 1.0f - z * z));
            const float phi = 2.0f * pt::kPi * (i + 0.5f) / nu;
            const Vec3 d { r * std::cos (phi), z, r * std::sin (phi) };
            directions.push_back (d);
            radiance.push_back (scene.lights.environmentRadiance (d));
        }
    const float dOmega = 4.0f * pt::kPi / static_cast<float> (nu * nv);

    double renderedSum = 0.0;
    double predictedSum = 0.0;
    double worst = 0.0;
    double lowestAlbedo = 2.0;
    double highestAlbedo = 0.0;
    int count = 0;
    for (std::uint32_t y = 0; y < size; ++y)
        for (std::uint32_t x = 0; x < size; ++x)
        {
            if (! fullyCovered (scene, camera, x, y))
                continue;
            const pt::Ray ray = camera.generateRay (x + 0.5f, y + 0.5f);
            pt::Hit hit;
            hits (scene, camera, x + 0.5f, y + 0.5f, hit);
            const Vec3 n = scene.geometry.triangles[hit.triangle].geometricNormal;
            const Vec3 wo = -pt::normalizeOrZero (ray.direction);
            pt::Bsdf bsdf;
            pt::buildMaterialBsdf (white, n, n, wo, true, bsdf);
            Vec3 predicted {};
            double albedo = 0.0;
            for (std::size_t k = 0; k < directions.size(); ++k)
            {
                if (pt::dot (directions[k], n) <= 0.0f)
                    continue;
                const pt::BsdfEval eval = bsdf.eval (wo, directions[k]);
                predicted = predicted + eval.value * radiance[k] * dOmega;
                albedo += meanOf (eval.value) * dOmega;
            }
            lowestAlbedo = std::min (lowestAlbedo, albedo);
            highestAlbedo = std::max (highestAlbedo, albedo);
            const double rendered = meanOf (pixel (tracer.accumulator(), x, y));
            renderedSum += rendered;
            predictedSum += meanOf (predicted);
            worst = std::max (worst, std::fabs (rendered - meanOf (predicted)) / meanOf (predicted));
            ++count;
        }
    check (count > 150, "the sphere fills the frame centre: " + std::to_string (count) + " pixels");
    check (lowestAlbedo > 0.95 && highestAlbedo < 1.02,
           "the white BSDF weights form an average (albedo " + str (lowestAlbedo) + " to " + str (highestAlbedo) + ")");
    check (std::fabs (renderedSum - predictedSum) <= 0.015 * predictedSum,
           "the HDRI furnace converges to the hemisphere average: " + str (renderedSum / count) + " vs "
               + str (predictedSum / count));
    check (worst < 0.12, "every pixel is near its hemisphere average, worst relative error " + str (worst));
}

// MIS, light sampling alone and BSDF sampling alone agree on an HDRI-lit
// diffuse plane.
void misTests()
{
    const pt::EnvironmentPixels patchSky = makePixels (32, 16, [] (Vec3 d, std::uint32_t x, std::uint32_t y) {
        const float base = 0.2f + 0.4f * std::max (0.0f, d.y);
        const bool patch = x >= 20 && x <= 22 && (y == 4 || y == 5);
        return patch ? Vec3 { 15.0f, 14.0f, 12.0f } : Vec3 { base, base, base * 1.2f };
    });
    pt::RenderScene scene;
    pt::Material floor;
    floor.baseColor = { 0.8f, 0.8f, 0.8f };
    scene.materials = { floor };
    appendMesh (scene, quad ({ 0.0f, 0.0f, 0.0f }, { 3.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, -3.0f }), 0);
    scene.lightRecords = { imageRecord (testAsset(), -40.0f) };
    scene.environmentImage = patchSky;
    scene.commit();
    check (scene.lights.stats().environmentImages == 1 && scene.lights.size() == 1, "the plane is lit by the image alone");

    // Looking straight down from 2 units: every pixel sees the plane.
    const pt::PinholeCamera camera = makeCamera ({ 0.0f, 2.0f, 0.0f }, pt::Quat { -0.70710678f, 0.0f, 0.0f, 0.70710678f },
                                                 0.6f, 20, 20);
    const pt::CancellationToken cancel {};
    const pt::LightStrategy strategies[] { pt::LightStrategy::MultipleImportance, pt::LightStrategy::LightOnly,
                                           pt::LightStrategy::BsdfOnly };
    double means[3] {};
    for (int i = 0; i < 3; ++i)
    {
        pt::PathTracer tracer (4);
        tracer.settings.maxDepth = 1;
        tracer.settings.strategy = strategies[i];
        const pt::RenderStats stats = tracer.renderFixed (scene, camera, 20, 20, 256, 9, cancel);
        check (stats.nonFiniteSamples == 0, "the MIS plane renders finite samples");
        means[i] = imageMean (tracer.accumulator());
    }
    check (means[0] > 0.1, "the plane is lit: " + str (means[0]));
    check (std::fabs (means[1] - means[0]) <= 0.03 * means[0],
           "light sampling alone agrees with MIS: " + str (means[1]) + " vs " + str (means[0]));
    check (std::fabs (means[2] - means[0]) <= 0.03 * means[0],
           "BSDF sampling alone agrees with MIS: " + str (means[2]) + " vs " + str (means[0]));
}

// ---------------------------------------------------------------- resident store

// A synthetic 8 x 4 RLE Radiance image.
std::vector<std::uint8_t> radianceBytes()
{
    const std::string header = "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\nPRIMARIES=0.64 0.33 0.30 0.60 0.15 0.06 0.3127 0.3290\n\n-Y 4 +X 8\n";
    std::vector<std::uint8_t> bytes (header.begin(), header.end());
    for (int y = 0; y < 4; ++y)
    {
        bytes.insert (bytes.end(), { 2, 2, 0, 8 });
        for (int c = 0; c < 4; ++c)
        {
            bytes.push_back (8);
            for (int x = 0; x < 8; ++x)
                bytes.push_back (static_cast<std::uint8_t> (c == 0 ? 40 + 20 * x : c == 1 ? 30 + 25 * y : c == 2 ? 200 - 9 * x - 7 * y : 130));
        }
    }
    return bytes;
}

std::string base64 (const std::vector<std::uint8_t>& bytes)
{
    static const char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    for (std::size_t i = 0; i < bytes.size(); i += 3)
    {
        const std::size_t count = std::min<std::size_t> (3, bytes.size() - i);
        unsigned packed = static_cast<unsigned> (bytes[i]) << 16;
        if (count > 1)
            packed |= static_cast<unsigned> (bytes[i + 1]) << 8;
        if (count > 2)
            packed |= bytes[i + 2];
        out += alphabet[(packed >> 18) & 63];
        out += alphabet[(packed >> 12) & 63];
        out += count > 1 ? alphabet[(packed >> 6) & 63] : '=';
        out += count > 2 ? alphabet[packed & 63] : '=';
    }
    return out;
}

void residentStoreTests()
{
    using namespace videohelper;
    const std::vector<std::uint8_t> bytes = radianceBytes();
    const visualenvironmentimage::Identity key { "pathtrace-store-hdr", 1, modelpayload::detail::digestHex (bytes) };
    const visualenvironmentimage::Metadata metadata { 8, 4, visualenvironmentimage::SourcePrimaries::Rec709D65 };
    auto& store = environmentimage::store();
    std::string error;
    std::uint64_t received = 0;
    const bool transferred = store.begin ("pathtrace-store-transfer", key, metadata, bytes.size(), error)
                          && store.append ("pathtrace-store-transfer", 0, base64 (bytes), received, error)
                          && store.commit ("pathtrace-store-transfer", error) != nullptr;
    check (transferred, "the synthetic image reaches the resident store: " + error);
    if (! transferred)
        return;

    grid::SceneLightRecord record = imageRecord (key, 90.0f, -1.0f, { 0.5f, 1.0f, 0.75f }, 2.0f);
    {
        const environmentimage::ResidentPtr resident = store.resolve (key);
        const pt::EnvironmentPixels pixels = pt::residentEnvironmentPixels (record.environmentImage);
        check (pixels.valid() && pixels.width == 8 && pixels.height == 4, "resident pixels resolve with their size");
        check (resident != nullptr && pixels.texels.get() == &resident->image.pixels,
               "the tracer reads the store's decoded pixels, not a copy");
        check (pixels.asset.id == key.id && pixels.asset.version == key.version && pixels.asset.contentSha256 == key.contentSha256,
               "resident pixels carry their asset identity");
    }
    grid::SceneLightRecord unknown = record;
    unknown.environmentImage.asset.version = 2;
    check (! pt::residentEnvironmentPixels (unknown.environmentImage).valid(), "an asset that is not resident resolves empty");

    auto scene = std::make_unique<grid::Visual3DScene>();
    scene->lights[0] = record;
    scene->lightCount = 1;
    pt::PrepareOptions options;
    options.environmentImageSource = [] (const grid::SceneEnvironmentImage& image) { return pt::residentEnvironmentPixels (image); };
    pt::RenderScene prepared;
    pt::prepareRenderScene (*scene, prepared, options);
    check (prepared.lights.stats().environmentImages == 1 && prepared.lights.stats().ignoredEnvironmentImages == 0,
           "prepareRenderScene lights the scene from the resident image");
    std::mt19937 rng (31);
    int mismatches = 0;
    for (int i = 0; i < 2000; ++i)
    {
        const Vec3 d = uniformSphere (rng);
        const Vec3 reference = gpuReference (prepared.environmentImage, pt::environmentImageTint (record),
                                             pt::environmentImageOrientationRadians (record), d);
        if (! closeTo (prepared.lights.environmentRadiance (d), reference, 1.0e-5f * (1.0f + meanOf (reference))))
            ++mismatches;
    }
    check (mismatches == 0, "resident image radiance matches the raster lookup, mismatches " + std::to_string (mismatches));

    // The prepared scene holds a lease: a store reset keeps the resident.
    store.reset();
    check (store.resolve (key) != nullptr, "a held tracer lease keeps the resident through a store reset");
    prepared.environmentImage = {};
    prepared.commit();
    store.reset();
    check (store.resolve (key) == nullptr, "releasing the lease lets the store evict the resident");

    pt::RenderScene withoutSource;
    pt::prepareRenderScene (*scene, withoutSource);
    check (withoutSource.lights.stats().ignoredEnvironmentImages == 1 && ! withoutSource.lights.hasEnvironment(),
           "without a pixel source the image is counted as ignored");
}

} // namespace

int main()
{
    conventionTests();
    distributionTests();
    brightTexelTests();
    orientationTests();
    lightSetTests();
    constantImageTests();
    hdriFurnaceTests();
    misTests();
    residentStoreTests();
    std::cout << checks << " checks, " << failures << " failures\n";
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
