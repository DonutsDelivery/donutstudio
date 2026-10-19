// Tests for the CPU path tracer refraction slice (Plan 3, PT-D): hero
// wavelength spectral transport, Cauchy dispersion, nested dielectrics, thin
// sheets, Beer-Lambert absorption and sized lights with MIS.

#include "pathtrace/accumulator.h"
#include "pathtrace/bsdf.h"
#include "pathtrace/bvh.h"
#include "pathtrace/integrator.h"
#include "pathtrace/lights.h"
#include "pathtrace/media.h"
#include "pathtrace/sampler.h"
#include "pathtrace/scene_ingest.h"
#include "pathtrace/spectrum.h"
#include "pathtrace/thread_pool.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <utility>
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

std::string describe (double value)
{
    return std::to_string (value);
}

constexpr double kDegree = 0.017453292519943295;

float u01 (std::mt19937& rng)
{
    return static_cast<float> (static_cast<std::uint32_t> (rng()) >> 8) * (1.0f / 16777216.0f);
}

float meanOf (Vec3 v) { return (v.x + v.y + v.z) / 3.0f; }

pt::Quat axisAngle (Vec3 axis, float angle)
{
    const Vec3 a = pt::normalizeOrZero (axis);
    const float s = std::sin (0.5f * angle);
    return { a.x * s, a.y * s, a.z * s, std::cos (0.5f * angle) };
}

// Rotation taking +Z to the unit direction d.
pt::Quat fromZTo (Vec3 d)
{
    const Vec3 z { 0.0f, 0.0f, 1.0f };
    const Vec3 axis = pt::cross (z, d);
    const float c = std::min (1.0f, std::max (-1.0f, pt::dot (z, d)));
    if (pt::length (axis) < 1.0e-6f)
        return c > 0.0f ? pt::Quat {} : axisAngle ({ 1.0f, 0.0f, 0.0f }, pt::kPi);
    return axisAngle (axis, std::acos (c));
}

// Looks down -Y; also turns local -Z to -Y (a planar light shining down) and
// local +Z to +Y (a directional light from above).
const pt::Quat kDown { -0.70710678f, 0.0f, 0.0f, 0.70710678f };

// ---------------------------------------------------------------- scenes

struct Mesh
{
    std::vector<Vec3> vertices;
    std::vector<std::uint32_t> indices;
};

Mesh uvSphere (std::uint32_t stacks, std::uint32_t slices, float radius, Vec3 centre = {})
{
    Mesh mesh;
    mesh.vertices.push_back (centre + Vec3 { 0.0f, radius, 0.0f });
    for (std::uint32_t i = 1; i < stacks; ++i)
    {
        const float theta = pt::kPi * static_cast<float> (i) / static_cast<float> (stacks);
        for (std::uint32_t j = 0; j < slices; ++j)
        {
            const float phi = 2.0f * pt::kPi * static_cast<float> (j) / static_cast<float> (slices);
            mesh.vertices.push_back (centre + Vec3 { radius * std::sin (theta) * std::cos (phi), radius * std::cos (theta),
                                                     radius * std::sin (theta) * std::sin (phi) });
        }
    }
    mesh.vertices.push_back (centre + Vec3 { 0.0f, -radius, 0.0f });
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

// Axis-aligned box; use appendMesh with outwardFrom at its centre.
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

// Appends flat-shaded triangles. With outwardFrom set, every front face points
// away from that point.
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

grid::SceneLightRecord directionalRecord (Vec3 toLight, float irradiance)
{
    grid::SceneLightRecord light;
    light.kind = grid::SceneLightKind::Directional;
    const pt::Quat q = fromZTo (pt::normalizeOrZero (toLight));
    light.transform.rotation = { q.x, q.y, q.z, q.w };
    light.intensity = irradiance;
    return light;
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

pt::Material diffuse (Vec3 colour)
{
    pt::Material m;
    m.baseColor = colour;
    m.roughness = 1.0f;
    return m;
}

pt::Material lamp (float radiance)
{
    pt::Material m;
    m.baseColor = { 0.0f, 0.0f, 0.0f };
    m.emission = { radiance, radiance, radiance };
    return m;
}

pt::Material clearGlass (float ior, float dispersion = 0.0f)
{
    pt::Material m;
    m.baseColor = { 1.0f, 1.0f, 1.0f };
    m.transmission = 1.0f;
    m.roughness = 0.0f;
    m.ior = ior;
    m.dispersion = dispersion;
    return m;
}

struct Image
{
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<Vec3> pixels;
    pt::RenderStats stats {};

    Vec3 at (std::uint32_t x, std::uint32_t y) const { return pixels[static_cast<std::size_t> (y) * width + x]; }
};

Image render (const pt::RenderScene& scene, const pt::PinholeCamera& camera, const pt::IntegratorSettings& settings,
              std::uint32_t samples, std::uint64_t generation = 1)
{
    pt::PathTracer tracer (4);
    tracer.settings = settings;
    const pt::CancellationToken cancel {};
    Image image;
    image.width = camera.width;
    image.height = camera.height;
    image.stats = tracer.renderFixed (scene, camera, camera.width, camera.height, samples, generation, cancel);
    for (std::uint32_t y = 0; y < camera.height; ++y)
        for (std::uint32_t x = 0; x < camera.width; ++x)
        {
            const pt::Rgb v = tracer.accumulator().mean (x, y);
            image.pixels.push_back ({ v.r, v.g, v.b });
        }
    return image;
}

std::array<double, 3> channelMeans (const Image& image)
{
    std::array<double, 3> sum { 0.0, 0.0, 0.0 };
    for (const Vec3 p : image.pixels)
    {
        sum[0] += p.x;
        sum[1] += p.y;
        sum[2] += p.z;
    }
    const double n = static_cast<double> (std::max<std::size_t> (1, image.pixels.size()));
    return { sum[0] / n, sum[1] / n, sum[2] / n };
}

// Channel means over pixels whose centre ray hits the scene.
std::array<double, 3> coveredChannelMeans (const pt::RenderScene& scene, const pt::PinholeCamera& camera,
                                           const Image& image, int& covered)
{
    std::array<double, 3> sum { 0.0, 0.0, 0.0 };
    covered = 0;
    for (std::uint32_t y = 0; y < camera.height; ++y)
        for (std::uint32_t x = 0; x < camera.width; ++x)
        {
            pt::Hit hit;
            if (! scene.bvh.intersect (camera.generateRay (static_cast<float> (x) + 0.5f, static_cast<float> (y) + 0.5f), hit))
                continue;
            const Vec3 p = image.at (x, y);
            sum[0] += p.x;
            sum[1] += p.y;
            sum[2] += p.z;
            ++covered;
        }
    const double n = std::max (1, covered);
    return { sum[0] / n, sum[1] / n, sum[2] / n };
}

bool closeRelative (double a, double b, double relative, double absolute = 0.0)
{
    return std::fabs (a - b) <= relative * std::max (std::fabs (a), std::fabs (b)) + absolute;
}

// Refraction of direction d (unit) through a surface with unit normal n on
// the side d comes from, eta the far index over the near index. False on
// total internal reflection.
bool refractDirection (Vec3 d, Vec3 n, double eta, Vec3& out)
{
    const double cosI = -static_cast<double> (pt::dot (d, n));
    const double sin2T = std::max (0.0, 1.0 - cosI * cosI) / (eta * eta);
    if (sin2T >= 1.0)
        return false;
    const double cosT = std::sqrt (1.0 - sin2T);
    const double a = 1.0 / eta;
    const double b = cosI / eta - cosT;
    out = { static_cast<float> (a * d.x + b * n.x), static_cast<float> (a * d.y + b * n.y),
            static_cast<float> (a * d.z + b * n.z) };
    out = pt::normalizeOrZero (out);
    return true;
}

// ---------------------------------------------------------------- spectrum

void spectrumTests()
{
    // A flat white spectrum upsamples to exactly 1 at every wavelength.
    bool flat = true;
    for (int i = 0; i < 400; i += 7)
    {
        const pt::Wavelengths wl = pt::sampleHeroWavelengths (static_cast<float> (i) / 400.0f);
        const pt::Spectrum s = pt::upsampleRgb ({ 1.0f, 1.0f, 1.0f }, wl);
        for (int c = 0; c < pt::kSpectrumChannels; ++c)
            flat = flat && std::fabs (s[c] - 1.0f) < 1.0e-5f;
    }
    check (flat, "white upsamples to the flat spectrum 1");

    // The basis is non-negative and a partition of unity.
    float lowest = 1.0f;
    float worstSum = 0.0f;
    for (int i = 0; i <= 800; ++i)
    {
        const Vec3 b = pt::rgbBasisAt (380.0f + 0.5f * static_cast<float> (i));
        lowest = std::min (lowest, std::min (b.x, std::min (b.y, b.z)));
        worstSum = std::max (worstSum, std::fabs (b.x + b.y + b.z - 1.0f));
    }
    check (lowest >= 0.0f, "basis spectra are never negative, lowest " + describe (lowest));
    check (worstSum < 1.0e-5f, "basis spectra sum to 1, worst error " + describe (worstSum));

    // Round trip: the colour of each basis spectrum is its own primary.
    double colour[3][3] {};
    const int steps = 8000;
    const double step = 400.0 / steps;
    for (int i = 0; i < steps; ++i)
    {
        const float nm = static_cast<float> (380.0 + (i + 0.5) * step);
        const Vec3 b = pt::rgbBasisAt (nm);
        const Vec3 w = pt::rgbMatchingWeightAt (nm);
        const float bv[3] { b.x, b.y, b.z };
        const float wv[3] { w.x, w.y, w.z };
        for (int k = 0; k < 3; ++k)
            for (int j = 0; j < 3; ++j)
                colour[k][j] += static_cast<double> (bv[k]) * static_cast<double> (wv[j]) * step;
    }
    double worstRoundTrip = 0.0;
    for (int k = 0; k < 3; ++k)
        for (int j = 0; j < 3; ++j)
            worstRoundTrip = std::max (worstRoundTrip, std::fabs (colour[k][j] - (k == j ? 1.0 : 0.0)));
    check (worstRoundTrip < 2.0e-3, "RGB to spectrum to RGB is the identity, worst error " + describe (worstRoundTrip));

    // Hero and companions: uniform, 100 nm apart, wrapped into [380, 780).
    {
        bool spaced = true;
        for (int i = 0; i < 1000; ++i)
        {
            const pt::Wavelengths wl = pt::sampleHeroWavelengths ((static_cast<float> (i) + 0.5f) / 1000.0f);
            spaced = spaced && wl.spectral;
            for (int c = 0; c < pt::kSpectrumChannels; ++c)
            {
                const float nm = wl.nm[static_cast<std::size_t> (c)];
                float expected = wl.nm[0] + 100.0f * static_cast<float> (c);
                if (expected >= 780.0f)
                    expected -= 400.0f;
                spaced = spaced && nm >= 380.0f && nm < 780.0f && std::fabs (nm - expected) < 1.0e-3f;
            }
        }
        check (spaced, "the companions are the hero rotated by 100, 200 and 300 nm");
    }

    // The colour estimate of a flat spectrum averages to white.
    {
        Vec3 sum {};
        const int n = 4000;
        for (int i = 0; i < n; ++i)
        {
            const pt::Wavelengths wl = pt::sampleHeroWavelengths ((static_cast<float> (i) + 0.5f) / static_cast<float> (n));
            sum = sum + pt::channelsToRgb (pt::Spectrum::constant (1.0f), wl);
        }
        sum = sum * (1.0f / static_cast<float> (n));
        check (std::fabs (sum.x - 1.0f) < 2.0e-3f && std::fabs (sum.y - 1.0f) < 2.0e-3f && std::fabs (sum.z - 1.0f) < 2.0e-3f,
               "a flat spectrum converts to white (D65 white balance): " + describe (sum.x) + ", " + describe (sum.y)
                   + ", " + describe (sum.z));
    }

    // RGB mode is a pass-through.
    {
        const pt::Wavelengths rgb = pt::rgbChannels();
        const pt::Spectrum s = pt::upsampleRgb ({ 0.2f, 0.4f, 0.6f }, rgb);
        const Vec3 back = pt::channelsToRgb (s, rgb);
        check (! rgb.spectral && s[0] == 0.2f && s[1] == 0.4f && s[2] == 0.6f && s[3] == 0.0f && back.x == 0.2f
                   && back.y == 0.4f && back.z == 0.6f,
               "RGB channels pass through unchanged");
    }

    // Cauchy index and Abbe number.
    {
        check (std::fabs (pt::cauchyIor (1.5f, 0.01f, pt::kDLineNm) - 1.5f) < 1.0e-6f, "the Cauchy index equals ior at the d line");
        check (pt::cauchyIor (1.5f, 0.01f, 400.0f) > pt::cauchyIor (1.5f, 0.01f, 700.0f), "blue bends more than red");
        const float b = pt::cauchyBFromAbbe (1.5168f, 64.17f);
        const float spread = pt::cauchyIor (1.5168f, b, pt::kFLineNm) - pt::cauchyIor (1.5168f, b, pt::kCLineNm);
        check (std::fabs (spread - 0.5168f / 64.17f) < 1.0e-5f, "an Abbe number gives n_F - n_C = (n_d - 1) / V");
        pt::Material abbeGlass = clearGlass (1.5168f);
        abbeGlass.abbe = 64.17f;
        check (pt::materialDispersive (abbeGlass) && std::fabs (pt::materialCauchyB (abbeGlass) - b) < 1.0e-7f,
               "a material with only an Abbe number is dispersive");
        check (! pt::materialDispersive (clearGlass (1.5f)), "glass without B or Abbe is not dispersive");
        check (pt::materialIorAt (abbeGlass, 450.0f, false) == 1.5168f, "RGB mode uses the d-line index");
    }
}

// ---------------------------------------------------------------- prism

// Prism of apex angle 60 degrees with the entry face at 50 degrees to the ray.
constexpr double kApex = 60.0 * kDegree;
const Vec3 kEntryNormal { -0.76604444f, 0.0f, 0.64278761f };  // outward, 50 degrees from +Z
const Vec3 kExitNormal { -0.17364818f, 0.0f, -0.98480775f };  // outward

double expectedDeviation (double n)
{
    const double theta1 = 50.0 * kDegree;
    const double theta2 = std::asin (std::sin (theta1) / n);
    const double theta3 = kApex - theta2;
    const double theta4 = std::asin (n * std::sin (theta3));
    return theta1 + theta4 - kApex;
}

void prismLobeTests()
{
    const pt::Material glass = clearGlass (1.5f, 0.02f);
    const Vec3 d0 { 0.0f, 0.0f, -1.0f };
    double deviation[2] {};
    double expected[2] {};
    const float targets[2] { 400.0f, 700.0f };
    for (int i = 0; i < 2; ++i)
    {
        const pt::Wavelengths wl = pt::sampleHeroWavelengths ((targets[i] - 380.0f) / 400.0f);
        const float nm = wl.nm[0];
        const std::string where = " at " + describe (nm) + " nm";
        check (std::fabs (nm - targets[i]) < 1.0e-2f, "the hero is the requested wavelength" + where);

        pt::MediumStack media;
        const pt::BoundaryCrossing in = pt::classifyBoundary (media, &glass, true);
        check (! in.falseInterface && in.nearSide == nullptr && in.farSide == &glass, "air to glass at the entry face" + where);
        const pt::DielectricSetup entry = pt::dielectricSetupFor (in, wl);
        check (entry.dispersive && entry.eta[0] == pt::materialIorAt (glass, nm, true) && std::fabs (entry.etaD - 1.5f) < 1.0e-6f,
               "the entry setup is dispersive with the d-line ratio 1.5" + where);
        pt::SpectralBsdf first;
        pt::buildMaterialBsdf (glass, kEntryNormal, kEntryNormal, -d0, wl, entry, first);
        pt::SpectralBsdfSample t1;
        const bool refracted = first.sample (-d0, 0.9f, { 0.5f, 0.5f }, t1);
        check (refracted && (t1.flags & pt::kBsdfTransmission) != 0 && (t1.flags & pt::kBsdfHeroOnly) != 0,
               "dispersive refraction keeps the hero only" + where);
        check (t1.weight[0] > 0.0f && t1.weight[1] == 0.0f && t1.weight[2] == 0.0f && t1.weight[3] == 0.0f
                   && t1.channelPdfRatio[1] == 0.0f && t1.channelPdfRatio[2] == 0.0f && t1.channelPdfRatio[3] == 0.0f,
               "companions carry nothing after dispersive refraction" + where);
        pt::applyCrossing (media, &glass, in);

        const Vec3 d1 = t1.wi;
        const pt::BoundaryCrossing out = pt::classifyBoundary (media, &glass, false);
        check (! out.falseInterface && out.nearSide == &glass && out.farSide == nullptr, "glass to air at the exit face" + where);
        const pt::DielectricSetup exit = pt::dielectricSetupFor (out, wl);
        pt::SpectralBsdf second;
        pt::buildMaterialBsdf (glass, -kExitNormal, -kExitNormal, -d1, wl, exit, second);
        pt::SpectralBsdfSample t2;
        const bool left = second.sample (-d1, 0.9f, { 0.5f, 0.5f }, t2);
        check (left && (t2.flags & pt::kBsdfTransmission) != 0, "the ray leaves through the exit face" + where);
        const double c = std::min (1.0, std::max (-1.0, static_cast<double> (pt::dot (d0, t2.wi))));
        deviation[i] = std::acos (c);
        expected[i] = expectedDeviation (static_cast<double> (pt::cauchyIor (1.5f, 0.02f, nm)));
        check (std::fabs (deviation[i] - expected[i]) < 2.0e-4,
               "prism deviation follows Snell's law with the Cauchy index" + where + ": " + describe (deviation[i])
                   + " vs " + describe (expected[i]));
    }
    check (deviation[0] > deviation[1] + 0.05, "400 nm deviates more than 700 nm: " + describe (deviation[0]) + " vs "
                                                  + describe (deviation[1]));
    check (std::fabs ((deviation[0] - deviation[1]) - (expected[0] - expected[1])) < 3.0e-4,
           "the angular spread between 400 and 700 nm is the Cauchy prediction");
}

// The same prism as geometry: a sun seen through it spreads into a spectrum.
void prismImageTest()
{
    const float a = static_cast<float> (-20.0 * kDegree);
    const auto turn = [a] (float x, float z) {
        return Vec3 { std::cos (a) * x + std::sin (a) * z, 0.0f, -std::sin (a) * x + std::cos (a) * z - 4.0f };
    };
    const float side = 6.0f;
    const float half = static_cast<float> (0.5 * kApex);
    const Vec3 v0 = turn (-2.0f, 0.0f);
    const Vec3 v1 = turn (-2.0f + side * std::cos (half), side * std::sin (half));
    const Vec3 v2 = turn (-2.0f + side * std::cos (half), -side * std::sin (half));
    const Vec3 centre = (v0 + v1 + v2) * (1.0f / 3.0f);
    const auto outward = [&centre] (Vec3 p, Vec3 q) {
        const Vec3 e = q - p;
        Vec3 n = pt::normalizeOrZero ({ e.z, 0.0f, -e.x });
        if (pt::dot (n, (p + q) * 0.5f - centre) < 0.0f)
            n = -n;
        return n;
    };
    const Vec3 n1 = outward (v0, v1);
    const Vec3 n2 = outward (v0, v2);
    check (std::fabs (pt::dot (n1, kEntryNormal) - 1.0f) < 1.0e-5f && std::fabs (pt::dot (n2, kExitNormal) - 1.0f) < 1.0e-5f,
           "the prism mesh has the faces of the lobe test");

    const auto exitDirection = [&n1, &n2] (Vec3 d, float nm, Vec3& out) {
        const double n = pt::cauchyIor (1.5f, 0.02f, nm);
        Vec3 inside;
        return refractDirection (d, n1, n, inside) && refractDirection (inside, -n2, 1.0 / n, out);
    };
    Vec3 sun;
    check (exitDirection ({ 0.0f, 0.0f, -1.0f }, 550.0f, sun), "550 nm passes the prism");

    pt::RenderScene scene;
    scene.materials = { clearGlass (1.5f, 0.02f) };
    Mesh prism;
    const float h = 10.0f;
    prism.vertices = { v0 + Vec3 { 0.0f, -h, 0.0f }, v1 + Vec3 { 0.0f, -h, 0.0f }, v2 + Vec3 { 0.0f, -h, 0.0f },
                       v0 + Vec3 { 0.0f, h, 0.0f }, v1 + Vec3 { 0.0f, h, 0.0f }, v2 + Vec3 { 0.0f, h, 0.0f } };
    prism.indices = { 0, 1, 2, 3, 4, 5, 0, 1, 4, 0, 4, 3, 1, 2, 5, 1, 5, 4, 2, 0, 3, 2, 3, 5 };
    appendMesh (scene, prism, 0, &centre);
    scene.lightRecords = { directionalRecord (sun, 1.0f) };
    pt::LightShape disk;
    disk.angularRadius = 0.03f;
    scene.lightShapes = { disk };
    scene.commit();
    check (scene.anyDispersive && scene.lights.analyticCount() == 1, "the prism scene is dispersive with one sun disk");

    const pt::PinholeCamera camera = makeCamera ({}, pt::Quat {}, 0.3f, 64, 64);
    pt::IntegratorSettings settings;
    settings.maxDepth = 8;
    const Image image = render (scene, camera, settings, 64);
    check (image.stats.spectral, "a dispersive scene renders spectrally by default");
    check (image.stats.nonFiniteSamples == 0, "the prism renders finite samples");

    double redWeight = 0.0, redX = 0.0, blueWeight = 0.0, blueX = 0.0;
    for (std::uint32_t y = 0; y < image.height; ++y)
        for (std::uint32_t x = 0; x < image.width; ++x)
        {
            const Vec3 p = image.at (x, y);
            const double px = static_cast<double> (x) + 0.5;
            redWeight += std::max (0.0f, p.x);
            redX += std::max (0.0f, p.x) * px;
            blueWeight += std::max (0.0f, p.z);
            blueX += std::max (0.0f, p.z) * px;
        }
    check (redWeight > 0.0 && blueWeight > 0.0, "the sun is seen through the prism");

    // Where the sun's centre appears at 450 and 650 nm.
    double predicted[2] {};
    const float wavelengths[2] { 450.0f, 650.0f };
    for (int i = 0; i < 2; ++i)
    {
        double best = 1.0e9;
        for (std::uint32_t x = 0; x < camera.width; ++x)
        {
            Vec3 out;
            const pt::Ray ray = camera.generateRay (static_cast<float> (x) + 0.5f, 0.5f * static_cast<float> (camera.height));
            if (! exitDirection (ray.direction, wavelengths[i], out))
                continue;
            const double angle = std::acos (std::min (1.0, static_cast<double> (pt::dot (out, sun))));
            if (angle < best)
            {
                best = angle;
                predicted[i] = static_cast<double> (x) + 0.5;
            }
        }
    }
    const double expectedShift = predicted[0] - predicted[1];
    check (std::fabs (expectedShift) >= 4.0, "the design separates 450 and 650 nm by pixels: " + describe (expectedShift));
    if (redWeight > 0.0 && blueWeight > 0.0)
    {
        const double measuredShift = blueX / blueWeight - redX / redWeight;
        check (measuredShift * expectedShift > 0.0 && std::fabs (measuredShift) >= 0.3 * std::fabs (expectedShift),
               "the prism separates blue from red in the image: shift " + describe (measuredShift) + " px, predicted "
                   + describe (expectedShift));
    }
}

// ---------------------------------------------------------------- furnaces

void furnace (const std::string& name, const pt::Material& material, pt::SpectralMode mode, std::uint32_t samples,
              double low, double high, double channelTolerance, std::array<double, 3>* means = nullptr)
{
    const float environment = 0.5f;
    pt::RenderScene scene;
    scene.materials = { material };
    const Vec3 origin {};
    appendMesh (scene, uvSphere (24, 48, 1.0f), 0, &origin);
    scene.lightRecords = { environmentRecord (environment) };
    scene.commit();
    const pt::PinholeCamera camera = makeCamera ({ 0.0f, 0.0f, 3.5f }, pt::Quat {}, 0.4f, 32, 32);
    pt::IntegratorSettings settings;
    settings.maxDepth = 64;
    settings.spectral = mode;
    const Image image = render (scene, camera, settings, samples);
    check (image.stats.nonFiniteSamples == 0, name + ": no non-finite samples");
    int covered = 0;
    const std::array<double, 3> c = coveredChannelMeans (scene, camera, image, covered);
    const double mean = (c[0] + c[1] + c[2]) / 3.0;
    check (covered > 300, name + ": the sphere covers the image centre");
    check (mean >= environment * low && mean <= environment * high,
           name + ": sphere mean " + describe (mean) + " near the environment " + describe (environment));
    for (int k = 0; k < 3; ++k)
        check (std::fabs (c[static_cast<std::size_t> (k)] - mean) <= channelTolerance * environment,
               name + ": no colour shift, channel " + std::to_string (k) + " = " + describe (c[static_cast<std::size_t> (k)]));
    if (means != nullptr)
        *means = c;
}

// Mean sample weight of a material's BSDF at cos = c on the side entering
// says: its directional albedo, 1 for a lobe that keeps all energy.
double directionalAlbedo (const pt::Material& material, bool entering, float c, std::mt19937& rng)
{
    const Vec3 up { 0.0f, 0.0f, 1.0f };
    const Vec3 wo { std::sqrt (std::max (0.0f, 1.0f - c * c)), 0.0f, c };
    pt::Bsdf bsdf;
    pt::buildMaterialBsdf (material, up, up, wo, entering, bsdf);
    const int n = 20000;
    double sum = 0.0;
    for (int i = 0; i < n; ++i)
    {
        pt::BsdfSample s;
        const float uLobe = u01 (rng);
        const float u1 = u01 (rng);
        const float u2 = u01 (rng);
        if (bsdf.sample (wo, uLobe, { u1, u2 }, s))
            sum += static_cast<double> (meanOf (s.weight));
    }
    return sum / n;
}

void dispersiveFurnaceTests()
{
    furnace ("dispersive glass furnace", clearGlass (1.5f, 0.02f), pt::SpectralMode::Automatic, 64, 0.975, 1.02, 0.05);

    // Rough glass loses energy without any dispersion: single-scattering GGX
    // drops the light that would bounce between microfacets. Measure that
    // loss on the lobe and bound the sphere with it.
    // - Entering, the camera sees the sphere's facets cosine-weighted, so the
    //   first crossing keeps the cosine-weighted albedo outside.
    // - Inside a sphere a chord meets the surface at the same angle at both
    //   ends, the refraction angle of the entry (cos >= 0.745 at index 1.5)
    //   spread a little by the roughness, so inside crossings keep at least
    //   the lowest albedo over cos in [0.6, 1].
    // - Inside, Fresnel reflects about 10%, so a path crosses inside about
    //   1.1 times on average. Allowing three inside crossings gives margin,
    //   and 0.01 covers the render noise.
    pt::Material frostedClear = clearGlass (1.5f);
    frostedClear.roughness = 0.3f;
    std::mt19937 rng (646u);
    double outside = 0.0;
    for (int i = 0; i < 8; ++i)
    {
        const float c = (static_cast<float> (i) + 0.5f) / 8.0f;
        outside += 2.0 * static_cast<double> (c) / 8.0 * directionalAlbedo (frostedClear, true, c, rng);
    }
    double inside = 1.0;
    for (const float c : { 0.6f, 0.7f, 0.8f, 0.9f, 1.0f })
        inside = std::min (inside, directionalAlbedo (frostedClear, false, c, rng));
    check (outside <= 1.005 && inside <= 1.005 && outside > 0.9 && inside > 0.9,
           "the rough glass lobe loses a few percent per crossing: outside albedo " + describe (outside) + ", inside "
               + describe (inside));
    const double low = outside * inside * inside * inside - 0.01;

    // Dispersion moves energy between wavelengths, never in or out, so the
    // dispersive sphere matches the same glass without dispersion.
    std::array<double, 3> reference {};
    std::array<double, 3> dispersive {};
    furnace ("rough glass furnace, spectral", frostedClear, pt::SpectralMode::On, 64, low, 1.02, 0.06, &reference);
    pt::Material frosted = clearGlass (1.5f, 0.02f);
    frosted.roughness = 0.3f;
    furnace ("rough dispersive glass furnace", frosted, pt::SpectralMode::Automatic, 64, low, 1.02, 0.06, &dispersive);
    const double referenceMean = (reference[0] + reference[1] + reference[2]) / 3.0;
    const double dispersiveMean = (dispersive[0] + dispersive[1] + dispersive[2]) / 3.0;
    check (closeRelative (dispersiveMean, referenceMean, 0.02),
           "rough dispersive glass keeps the energy of the same glass without dispersion: " + describe (dispersiveMean)
               + " vs " + describe (referenceMean));
}

// ---------------------------------------------------------------- spectral vs RGB

void spectralMatchesRgbTests()
{
    // White, grey and clear materials over several bounces.
    {
        pt::RenderScene scene;
        pt::Material sphere = diffuse ({ 0.8f, 0.8f, 0.8f });
        sphere.roughness = 0.5f;
        scene.materials = { diffuse ({ 0.7f, 0.7f, 0.7f }), sphere, clearGlass (1.5f) };
        appendMesh (scene, quad ({ 0.0f, 0.0f, 0.0f }, { 4.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, -4.0f }), 0);
        const Vec3 c1 { -0.6f, 0.5f, 0.0f };
        const Vec3 c2 { 0.6f, 0.5f, 0.0f };
        appendMesh (scene, uvSphere (16, 32, 0.5f, c1), 1, &c1);
        appendMesh (scene, uvSphere (16, 32, 0.5f, c2), 2, &c2);
        scene.lightRecords = { environmentRecord (0.6f), pointRecord ({ 1.0f, 3.0f, 1.0f }, 6.0f) };
        scene.commit();
        check (! scene.anyDispersive, "the grey scene has no dispersive material");
        const pt::PinholeCamera camera = makeCamera ({ 0.0f, 1.5f, 3.0f }, axisAngle ({ 1.0f, 0.0f, 0.0f }, -0.3f), 0.5f, 32, 32);
        pt::IntegratorSettings settings;
        settings.maxDepth = 6;
        settings.spectral = pt::SpectralMode::Off;
        const Image rgb = render (scene, camera, settings, 64);
        settings.spectral = pt::SpectralMode::Automatic;
        const Image automatic = render (scene, camera, settings, 64);
        settings.spectral = pt::SpectralMode::On;
        const Image spectral = render (scene, camera, settings, 64);
        check (! automatic.stats.spectral && spectral.stats.spectral, "automatic mode stays RGB without dispersion");
        check (automatic.pixels.size() == rgb.pixels.size()
                   && std::equal (automatic.pixels.begin(), automatic.pixels.end(), rgb.pixels.begin(),
                                  [] (Vec3 p, Vec3 q) { return p.x == q.x && p.y == q.y && p.z == q.z; }),
               "a non-dispersive scene renders exactly as in RGB mode");
        check (rgb.stats.nonFiniteSamples == 0 && spectral.stats.nonFiniteSamples == 0, "grey scene samples are finite");
        const std::array<double, 3> a = channelMeans (rgb);
        const std::array<double, 3> b = channelMeans (spectral);
        for (std::size_t k = 0; k < 3; ++k)
            check (closeRelative (a[k], b[k], 0.015, 1.0e-3),
                   "white multi-bounce: spectral matches RGB, channel " + std::to_string (k) + ": " + describe (b[k])
                       + " vs " + describe (a[k]));
    }

    // A coloured surface at one bounce: the upsampled spectrum converts back.
    {
        pt::RenderScene scene;
        scene.materials = { diffuse ({ 0.8f, 0.3f, 0.1f }) };
        const Vec3 origin {};
        appendMesh (scene, uvSphere (16, 32, 1.0f), 0, &origin);
        scene.lightRecords = { environmentRecord (1.0f) };
        scene.commit();
        const pt::PinholeCamera camera = makeCamera ({ 0.0f, 0.0f, 3.0f }, pt::Quat {}, 0.35f, 24, 24);
        pt::IntegratorSettings settings;
        settings.maxDepth = 1;
        settings.environmentVisibleToCamera = false;
        settings.spectral = pt::SpectralMode::Off;
        const Image rgb = render (scene, camera, settings, 128);
        settings.spectral = pt::SpectralMode::On;
        const Image spectral = render (scene, camera, settings, 128);
        const std::array<double, 3> a = channelMeans (rgb);
        const std::array<double, 3> b = channelMeans (spectral);
        for (std::size_t k = 0; k < 3; ++k)
            check (closeRelative (a[k], b[k], 0.03, 3.0e-3),
                   "coloured single bounce: spectral matches RGB, channel " + std::to_string (k) + ": " + describe (b[k])
                       + " vs " + describe (a[k]));
    }

    // Clear non-dispersive glass in spectral mode keeps its colour.
    std::array<double, 3> rgb {};
    std::array<double, 3> spectral {};
    furnace ("clear glass furnace, RGB", clearGlass (1.5f), pt::SpectralMode::Off, 16, 0.985, 1.0001, 0.001, &rgb);
    furnace ("clear glass furnace, spectral", clearGlass (1.5f), pt::SpectralMode::On, 32, 0.98, 1.01, 0.02, &spectral);
    for (std::size_t k = 0; k < 3; ++k)
        check (closeRelative (rgb[k], spectral[k], 0.02), "clear glass: spectral matches RGB, channel " + std::to_string (k));
}

// ---------------------------------------------------------------- total internal reflection

void tirTests()
{
    const pt::Material glass = clearGlass (1.5f);
    const double critical = std::asin (1.0 / 1.5);
    const Vec3 n { 0.0f, 0.0f, 1.0f };
    const pt::Wavelengths rgb = pt::rgbChannels();

    const auto sweep = [&] (const pt::Material& material, const pt::Wavelengths& wl, double angle, int& reflected,
                            int& transmitted, float& sinOut, bool& finite) {
        const Vec3 wo { static_cast<float> (std::sin (angle)), 0.0f, static_cast<float> (std::cos (angle)) };
        pt::BoundaryCrossing crossing;
        crossing.nearSide = &material;
        const pt::DielectricSetup setup = pt::dielectricSetupFor (crossing, wl);
        pt::SpectralBsdf bsdf;
        pt::buildMaterialBsdf (material, n, n, wo, wl, setup, bsdf);
        reflected = transmitted = 0;
        finite = true;
        for (int i = 0; i < 256; ++i)
        {
            pt::SpectralBsdfSample s;
            if (! bsdf.sample (wo, (static_cast<float> (i) + 0.5f) / 256.0f, { 0.5f, 0.5f }, s))
                continue;
            finite = finite && s.weight.finite() && std::isfinite (s.pdf);
            if ((s.flags & pt::kBsdfTransmission) != 0)
            {
                ++transmitted;
                sinOut = std::sqrt (s.wi.x * s.wi.x + s.wi.y * s.wi.y);
            }
            else
            {
                ++reflected;
                finite = finite && std::fabs (s.weight[0] - 1.0f) < 1.0e-6f;
            }
        }
    };

    int reflected = 0, transmitted = 0;
    float sinOut = 0.0f;
    bool finite = true;
    sweep (glass, rgb, critical + 0.2 * kDegree, reflected, transmitted, sinOut, finite);
    check (transmitted == 0 && reflected == 256 && finite, "past the critical angle every sample reflects with weight 1");
    sweep (glass, rgb, critical - 0.3 * kDegree, reflected, transmitted, sinOut, finite);
    const double expectedSin = 1.5 * std::sin (critical - 0.3 * kDegree);
    check (transmitted > 0 && finite && std::fabs (sinOut - expectedSin) < 1.0e-4,
           "just below the critical angle light grazes out at sin " + describe (sinOut) + " vs " + describe (expectedSin));

    // With dispersion the critical angle depends on the wavelength.
    const pt::Material flint = clearGlass (1.5f, 0.02f);
    const double angle = 40.5 * kDegree;
    const pt::Wavelengths blue = pt::sampleHeroWavelengths ((400.0f - 380.0f) / 400.0f);
    const pt::Wavelengths red = pt::sampleHeroWavelengths ((700.0f - 380.0f) / 400.0f);
    check (std::asin (1.0 / pt::cauchyIor (1.5f, 0.02f, blue.nm[0])) < angle
               && std::asin (1.0 / pt::cauchyIor (1.5f, 0.02f, red.nm[0])) > angle,
           "40.5 degrees lies between the critical angles at 400 and 700 nm");
    sweep (flint, blue, angle, reflected, transmitted, sinOut, finite);
    check (transmitted == 0 && finite, "a 400 nm hero is totally reflected at 40.5 degrees");
    sweep (flint, red, angle, reflected, transmitted, sinOut, finite);
    check (transmitted > 0 && finite, "a 700 nm hero escapes at 40.5 degrees");
}

// ---------------------------------------------------------------- nested dielectrics

void nestedTests()
{
    pt::Material glass = clearGlass (1.5f);
    pt::Material water = clearGlass (1.33f);
    const pt::Wavelengths rgb = pt::rgbChannels();

    // Default priorities: entry order decides.
    {
        pt::MediumStack media;
        const pt::BoundaryCrossing a = pt::classifyBoundary (media, &glass, true);
        check (! a.falseInterface && a.nearSide == nullptr && a.farSide == &glass, "air into glass");
        pt::applyCrossing (media, &glass, a);
        const pt::BoundaryCrossing b = pt::classifyBoundary (media, &water, true);
        const pt::DielectricSetup bs = pt::dielectricSetupFor (b, rgb);
        check (! b.falseInterface && b.nearSide == &glass && b.farSide == &water
                   && std::fabs (bs.eta[0] - 1.33f / 1.5f) < 1.0e-6f && std::fabs (bs.etaD - 1.33f / 1.5f) < 1.0e-6f,
               "glass into water refracts with 1.33 / 1.5");
        pt::applyCrossing (media, &water, b);
        check (media.active() == &water && media.size() == 2, "water owns the inside of the inner sphere");
        const pt::BoundaryCrossing c = pt::classifyBoundary (media, &water, false);
        const pt::DielectricSetup cs = pt::dielectricSetupFor (c, rgb);
        check (! c.falseInterface && c.nearSide == &water && c.farSide == &glass
                   && std::fabs (cs.eta[0] - 1.5f / 1.33f) < 1.0e-5f,
               "water back into glass refracts with 1.5 / 1.33");
        pt::applyCrossing (media, &water, c);
        const pt::BoundaryCrossing d = pt::classifyBoundary (media, &glass, false);
        check (! d.falseInterface && d.nearSide == &glass && d.farSide == nullptr
                   && std::fabs (pt::dielectricSetupFor (d, rgb).eta[0] - 1.0f / 1.5f) < 1.0e-6f,
               "glass back into air");
        pt::applyCrossing (media, &glass, d);
        check (media.empty(), "the stack is empty again");
    }

    // Overlapping media of equal priority: the one entered last owns the overlap.
    {
        pt::Material a = clearGlass (1.4f);
        pt::Material b = clearGlass (1.6f);
        pt::MediumStack media;
        media.push (&a);
        const pt::BoundaryCrossing enterB = pt::classifyBoundary (media, &b, true);
        check (! enterB.falseInterface && enterB.nearSide == &a && enterB.farSide == &b, "entering B inside A is real");
        pt::applyCrossing (media, &b, enterB);
        const pt::BoundaryCrossing leaveA = pt::classifyBoundary (media, &a, false);
        check (leaveA.falseInterface, "leaving A inside B is a false interface");
        pt::applyCrossing (media, &a, leaveA);
        const pt::BoundaryCrossing leaveB = pt::classifyBoundary (media, &b, false);
        check (! leaveB.falseInterface && leaveB.nearSide == &b && leaveB.farSide == nullptr, "leaving B into air is real");
    }

    // Water filling a glass: the glass wall has priority over the water.
    {
        pt::Material wall = glass;
        wall.mediumPriority = 1;
        pt::MediumStack media;
        media.push (&wall);
        const pt::BoundaryCrossing intoWater = pt::classifyBoundary (media, &water, true);
        check (intoWater.falseInterface, "the water surface inside the glass wall is false");
        pt::applyCrossing (media, &water, intoWater);
        const pt::BoundaryCrossing outOfWall = pt::classifyBoundary (media, &wall, false);
        check (! outOfWall.falseInterface && outOfWall.nearSide == &wall && outOfWall.farSide == &water
                   && std::fabs (pt::dielectricSetupFor (outOfWall, rgb).eta[0] - 1.33f / 1.5f) < 1.0e-6f,
               "the inner wall surface is the glass to water interface, 1.33 / 1.5");
    }

    // An exit with nothing on the stack is treated as leaving into the owner.
    {
        pt::MediumStack media;
        const pt::BoundaryCrossing exit = pt::classifyBoundary (media, &glass, false);
        check (! exit.falseInterface && exit.nearSide == &glass && exit.farSide == nullptr, "an unbalanced exit stays finite");
        pt::MediumStack full;
        for (int i = 0; i < pt::MediumStack::kCapacity; ++i)
            check (full.push (&glass), "the stack takes its capacity");
        check (! full.push (&water) && full.size() == pt::MediumStack::kCapacity, "a full stack refuses more media");
    }

    // An index-matched inner sphere is invisible. Its boundary is a real
    // interface of index ratio exactly 1, so the vertex there must hand the
    // path on unchanged: no reflection, the incoming direction (up to float
    // rounding) and throughput 1 in every channel. This checks each vertex
    // the integrator builds, entering and leaving the inner sphere, without
    // comparing noisy images.
    {
        const pt::Material outer = clearGlass (1.5f);
        const pt::Material inner = clearGlass (1.5f);
        const Vec3 n { 0.0f, 0.0f, 1.0f };
        int vertices = 0;
        int changed = 0;
        int reflected = 0;
        double worstDirection = 0.0;
        double worstWeight = 0.0;
        for (const bool spectral : { false, true })
            for (const bool entering : { true, false })
            {
                const pt::Wavelengths wl = spectral ? pt::sampleHeroWavelengths (0.37f) : pt::rgbChannels();
                const int channels = spectral ? pt::kSpectrumChannels : 3;
                pt::MediumStack media;
                media.push (&outer);
                if (! entering)
                    media.push (&inner);
                const pt::BoundaryCrossing crossing = pt::classifyBoundary (media, &inner, entering);
                const pt::DielectricSetup setup = pt::dielectricSetupFor (crossing, wl);
                bool unit = ! crossing.falseInterface && setup.etaD == 1.0f && ! setup.dispersive;
                for (const float e : setup.eta)
                    unit = unit && e == 1.0f;
                check (unit, std::string ("the matched boundary has index ratio 1, ") + (entering ? "entering" : "leaving"));
                for (const float c : { 1.0f, 0.7f, 0.3f, 0.1f })
                    for (int k = 0; k < 16; ++k)
                    {
                        const float sine = std::sqrt (1.0f - c * c);
                        const Vec3 wo { sine * 0.6f, sine * 0.8f, c };
                        pt::SpectralBsdf bsdf;
                        pt::buildMaterialBsdf (inner, n, n, wo, wl, setup, bsdf);
                        pt::SpectralBsdfSample sample;
                        ++vertices;
                        if (! bsdf.sample (wo, (static_cast<float> (k) + 0.5f) / 16.0f, { 0.5f, 0.5f }, sample))
                        {
                            ++changed;
                            continue;
                        }
                        if ((sample.flags & pt::kBsdfTransmission) == 0)
                        {
                            ++reflected;
                            continue;
                        }
                        worstDirection = std::max (worstDirection, static_cast<double> (pt::length (sample.wi + wo)));
                        for (int ch = 0; ch < channels; ++ch)
                            worstWeight = std::max (worstWeight, std::fabs (static_cast<double> (sample.weight[ch]) - 1.0));
                    }
            }
        check (changed == 0 && reflected == 0,
               "every vertex at the matched boundary transmits: " + std::to_string (reflected) + " reflected, "
                   + std::to_string (changed) + " failed of " + std::to_string (vertices));
        check (worstDirection < 1.0e-5, "the path keeps its direction through the matched boundary, worst " + describe (worstDirection));
        check (worstWeight < 1.0e-5, "the path keeps its throughput through the matched boundary, worst " + describe (worstWeight));
    }

    // Rendered: an air bubble inside the glass changes the image.
    const auto renderNested = [] (bool bubble) {
        pt::RenderScene scene;
        scene.materials = { clearGlass (1.5f), clearGlass (1.01f), lamp (1.0f), lamp (0.2f) };
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                appendMesh (scene,
                            quad ({ -1.5f + static_cast<float> (i), -1.5f + static_cast<float> (j), -3.0f },
                                  { 0.5f, 0.0f, 0.0f }, { 0.0f, 0.5f, 0.0f }),
                            (i + j) % 2 == 0 ? 2u : 3u);
        const Vec3 origin {};
        appendMesh (scene, uvSphere (32, 64, 1.0f), 0, &origin);
        if (bubble)
            appendMesh (scene, uvSphere (24, 48, 0.6f), 1, &origin);
        scene.commit();
        const pt::PinholeCamera camera = makeCamera ({ 0.0f, 0.0f, 4.0f }, pt::Quat {}, 0.3f, 48, 48);
        pt::IntegratorSettings settings;
        settings.maxDepth = 16;
        return render (scene, camera, settings, 128);
    };
    const Image solid = renderNested (false);
    const Image bubble = renderNested (true);
    check (solid.stats.nonFiniteSamples == 0 && bubble.stats.nonFiniteSamples == 0, "nested renders are finite");
    double bubbleDifference = 0.0;
    for (std::size_t i = 0; i < solid.pixels.size(); ++i)
        bubbleDifference += std::fabs (static_cast<double> (meanOf (solid.pixels[i]) - meanOf (bubble.pixels[i])));
    bubbleDifference /= static_cast<double> (std::max<std::size_t> (1, solid.pixels.size()));
    check (bubbleDifference > 0.05, "an air bubble inside the glass changes the image: " + describe (bubbleDifference));
}

// ---------------------------------------------------------------- thin sheets

void thinSheetTests()
{
    pt::Material sheet = clearGlass (1.5f);
    sheet.transport = pt::DielectricTransport::ThinSheet;
    check (pt::isThinSheet (sheet) && ! pt::isMediumBoundary (sheet), "a sheet is not a medium boundary");
    check (std::fabs (pt::thinSheetReflectance (1.0f, 1.5f) - 0.08f / 1.04f) < 1.0e-5f, "sheet reflectance is 2R / (1 + R)");
    const pt::Wavelengths rgb = pt::rgbChannels();
    for (const float c : { 1.0f, 0.7f, 0.3f, 0.05f })
    {
        const pt::Spectrum t = pt::straightTransmittance (sheet, c, 1.5f, rgb);
        check (std::fabs (t[0] + pt::thinSheetReflectance (c, 1.5f) - 1.0f) < 1.0e-6f, "a clear sheet reflects or transmits all light");
    }

    // Lobe level: the transmitted direction is exactly the incoming one.
    {
        const Vec3 n = pt::normalizeOrZero ({ 0.3f, 0.2f, 0.9f });
        const Vec3 wo = pt::normalizeOrZero ({ -0.4f, 0.1f, 0.6f });
        pt::DielectricSetup setup;
        setup.eta.fill (1.5f);
        setup.etaD = 1.5f;
        pt::SpectralBsdf bsdf;
        pt::buildMaterialBsdf (sheet, n, n, wo, rgb, setup, bsdf);
        pt::SpectralBsdfSample s;
        const bool ok = bsdf.sample (wo, 0.95f, { 0.3f, 0.7f }, s);
        const Vec3 straight = -wo;
        check (ok && (s.flags & pt::kBsdfStraight) != 0 && s.wi.x == straight.x && s.wi.y == straight.y && s.wi.z == straight.z,
               "sheet transmission continues the ray without bending");
        check (! bsdf.hasNonDelta() && bsdf.eval (wo, straight).pdf == 0.0f, "a clear sheet is purely specular");
    }

    // Rendered: stripes behind a tilted sheet keep their place.
    const auto renderStripes = [&sheet] (bool withSheet) {
        pt::RenderScene scene;
        scene.materials = { lamp (1.0f), sheet };
        for (int k = 0; k < 6; ++k)
            appendMesh (scene, quad ({ -1.1f + 0.4f * static_cast<float> (k), 0.0f, -2.0f }, { 0.1f, 0.0f, 0.0f },
                                     { 0.0f, 2.0f, 0.0f }),
                        0);
        if (withSheet)
        {
            const float angle = static_cast<float> (40.0 * kDegree);
            appendMesh (scene, quad ({}, { 2.0f * std::cos (angle), 0.0f, -2.0f * std::sin (angle) }, { 0.0f, 2.0f, 0.0f }), 1);
        }
        scene.commit();
        const pt::PinholeCamera camera = makeCamera ({ 0.0f, 0.0f, 3.0f }, pt::Quat {}, 0.25f, 48, 48);
        pt::IntegratorSettings settings;
        settings.maxDepth = 8;
        return render (scene, camera, settings, 32);
    };
    const Image bare = renderStripes (false);
    const Image covered = renderStripes (true);
    check (covered.stats.nonFiniteSamples == 0, "the sheet scene renders finite samples");
    const pt::PinholeCamera camera = makeCamera ({ 0.0f, 0.0f, 3.0f }, pt::Quat {}, 0.25f, 48, 48);
    const float angle = static_cast<float> (40.0 * kDegree);
    const Vec3 sheetNormal { std::sin (angle), 0.0f, std::cos (angle) };
    int dark = 0, leaked = 0;
    double expected = 0.0, measured = 0.0;
    for (std::uint32_t y = 0; y < camera.height; ++y)
        for (std::uint32_t x = 0; x < camera.width; ++x)
        {
            const float without = meanOf (bare.at (x, y));
            const float with = meanOf (covered.at (x, y));
            if (without <= 0.0f)
            {
                ++dark;
                if (with > 1.0e-3f)
                    ++leaked;
                continue;
            }
            const Vec3 d = camera.generateRay (static_cast<float> (x) + 0.5f, static_cast<float> (y) + 0.5f).direction;
            const float c = std::fabs (pt::dot (d, sheetNormal));
            expected += static_cast<double> (without) * (1.0 - static_cast<double> (pt::thinSheetReflectance (c, 1.5f)));
            measured += with;
        }
    check (dark > 200 && leaked <= dark / 100, "no light moves into the gaps (no lateral offset): " + std::to_string (leaked)
                                                   + " of " + std::to_string (dark) + " dark pixels lit");
    check (expected > 0.0 && closeRelative (measured, expected, 0.03),
           "stripes seen through the sheet carry its transmittance: " + describe (measured) + " vs " + describe (expected));
}

// ---------------------------------------------------------------- Beer-Lambert

void beerLambertTests()
{
    pt::Material tinted = clearGlass (1.01f);
    tinted.baseColor = { 0.5f, 0.25f, 0.8f };
    tinted.absorptionDistance = 2.0f;
    const pt::Spectrum sigma = pt::mediumAbsorption (tinted, pt::rgbChannels());
    check (std::fabs (sigma[0] - std::log (2.0f) / 2.0f) < 1.0e-6f && std::fabs (sigma[1] - std::log (4.0f) / 2.0f) < 1.0e-6f
               && sigma[3] == 0.0f,
           "absorption is -ln(colour) / distance");
    const pt::Spectrum unbounded = pt::beerLambert (sigma, std::numeric_limits<float>::infinity());
    check (unbounded.finite() && unbounded[0] == 0.0f && unbounded[3] == 1.0f, "an infinite path absorbs fully where sigma > 0 and never makes NaN");
    check (pt::mediumAbsorption (clearGlass (1.5f), pt::rgbChannels()).maxValue() == 0.0f,
           "without a distance there is no volume absorption");

    const auto renderSlab = [] (float thickness, pt::SpectralMode mode) {
        pt::Material slab = clearGlass (1.01f);
        slab.baseColor = { 0.5f, 0.25f, 0.8f };
        slab.absorptionDistance = 1.0f;
        pt::RenderScene scene;
        scene.materials = { slab, lamp (1.0f) };
        const Vec3 centre {};
        appendMesh (scene, box ({ -2.0f, -2.0f, -0.5f * thickness }, { 2.0f, 2.0f, 0.5f * thickness }), 0, &centre);
        appendMesh (scene, quad ({ 0.0f, 0.0f, -3.0f }, { 4.0f, 0.0f, 0.0f }, { 0.0f, 4.0f, 0.0f }), 1);
        scene.commit();
        const pt::PinholeCamera camera = makeCamera ({ 0.0f, 0.0f, 4.0f }, pt::Quat {}, 0.08f, 16, 16);
        pt::IntegratorSettings settings;
        settings.maxDepth = 8;
        settings.spectral = mode;
        return render (scene, camera, settings, 64);
    };
    const Image one = renderSlab (1.0f, pt::SpectralMode::Off);
    const Image two = renderSlab (2.0f, pt::SpectralMode::Off);
    const Image oneSpectral = renderSlab (1.0f, pt::SpectralMode::On);
    check (one.stats.nonFiniteSamples == 0 && two.stats.nonFiniteSamples == 0 && oneSpectral.stats.nonFiniteSamples == 0,
           "slab renders are finite");
    const std::array<double, 3> t1 = channelMeans (one);
    const std::array<double, 3> t2 = channelMeans (two);
    const std::array<double, 3> s1 = channelMeans (oneSpectral);
    const double base[3] { 0.5, 0.25, 0.8 };
    for (std::size_t k = 0; k < 3; ++k)
    {
        check (closeRelative (t1[k], base[k], 0.03), "one unit of medium leaves the base colour, channel " + std::to_string (k)
                                                         + ": " + describe (t1[k]));
        check (closeRelative (t2[k], t1[k] * t1[k], 0.02), "two units leave its square, channel " + std::to_string (k) + ": "
                                                               + describe (t2[k]) + " vs " + describe (t1[k] * t1[k]));
        check (closeRelative (s1[k], base[k], 0.04, 2.0e-3), "spectral absorption over one unit matches, channel "
                                                                 + std::to_string (k) + ": " + describe (s1[k]));
    }
}

// ---------------------------------------------------------------- sized lights

// Light sampling estimate of the irradiance at p on a surface facing up.
double sampledIrradiance (const pt::LightSet& lights, Vec3 p, int n, double& worstPdfMismatch)
{
    double sum = 0.0;
    worstPdfMismatch = 0.0;
    for (int i = 0; i < n; ++i)
    {
        const float u = (static_cast<float> (i) + 0.5f) / static_cast<float> (n);
        // Second coordinate: radical inverse kept off 0 and 1 so no sample
        // lands exactly on a disk rim.
        const float radical = pt::sampling::toUnitFloat (pt::sampling::reverseBits (static_cast<std::uint32_t> (i)));
        const pt::Vec2 v { u, (radical * static_cast<float> (n - 1) + 0.5f) / static_cast<float> (n) };
        pt::LightSample s;
        if (! lights.sample (p, 0.5f, v, s) || s.delta)
            continue;
        const float c = std::max (0.0f, s.wi.y);
        sum += static_cast<double> (s.radiance.x) * c / static_cast<double> (s.pdf);
        Vec3 radiance;
        float pdf = 0.0f;
        float distance = 0.0f;
        const bool hit = lights.analyticHit (0, p, s.wi, std::numeric_limits<float>::infinity(), radiance, pdf, distance);
        const double mismatch = hit ? std::fabs (static_cast<double> (pdf / s.pdf) - 1.0) + std::fabs (radiance.x - s.radiance.x)
                                    : 1.0;
        worstPdfMismatch = std::max (worstPdfMismatch, mismatch);
    }
    return sum / n;
}

// Brute-force irradiance at p (facing up) from a planar light of radiance 1.
double referenceIrradiance (Vec3 p, Vec3 centre, Vec3 axisX, Vec3 axisY, bool disk)
{
    const int n = 600;
    double sum = 0.0;
    const Vec3 normal = pt::normalizeOrZero (pt::cross (axisX, axisY));
    const double cellArea = static_cast<double> (pt::length (axisX) * pt::length (axisY)) * 4.0 / (n * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            const float a = -1.0f + (2.0f * static_cast<float> (i) + 1.0f) / n;
            const float b = -1.0f + (2.0f * static_cast<float> (j) + 1.0f) / n;
            if (disk && a * a + b * b > 1.0f)
                continue;
            const Vec3 q = centre + axisX * a + axisY * b;
            const Vec3 d = q - p;
            const double d2 = pt::dot (d, d);
            const Vec3 w = pt::normalizeOrZero (d);
            sum += std::max (0.0f, w.y) * std::fabs (pt::dot (normal, w)) / d2 * cellArea;
        }
    return sum;
}

void sizedLightUnitTests()
{
    const pt::IngestedScene empty {};
    const std::vector<pt::Material> noMaterials {};
    const pt::Material fallback {};
    double mismatch = 0.0;

    // A sphere light delivers the point light's irradiance.
    for (const Vec3 p : { Vec3 { 0.0f, 0.0f, 0.0f }, Vec3 { 1.0f, 0.0f, 0.0f } })
    {
        const std::vector<grid::SceneLightRecord> records { pointRecord ({ 0.0f, 3.0f, 0.0f }, 9.0f) };
        std::vector<pt::LightShape> shapes (1);
        shapes[0].radius = 0.5f;
        pt::LightSet lights;
        lights.build (records.data(), records.size(), {}, empty, noMaterials, fallback, nullptr, &shapes);
        check (lights.analyticCount() == 1 && lights.stats().sizedLights == 1, "a point with a radius is a sphere light");
        const Vec3 toLight = Vec3 { 0.0f, 3.0f, 0.0f } - p;
        const double d2 = pt::dot (toLight, toLight);
        const double expected = 9.0 * (3.0 / std::sqrt (d2)) / d2;
        const double e = sampledIrradiance (lights, p, 4096, mismatch);
        check (closeRelative (e, expected, 0.005), "sphere light irradiance " + describe (e) + " equals the point light's "
                                                       + describe (expected));
        check (mismatch < 1.0e-3, "a sampled sphere direction hits the sphere with the same pdf and radiance");
    }

    // A sun disk delivers its irradiance at normal incidence.
    {
        const std::vector<grid::SceneLightRecord> records { directionalRecord ({ 0.0f, 1.0f, 0.0f }, 2.0f) };
        std::vector<pt::LightShape> shapes (1);
        shapes[0].angularRadius = 0.2f;
        pt::LightSet lights;
        lights.build (records.data(), records.size(), {}, empty, noMaterials, fallback, nullptr, &shapes);
        const double e = sampledIrradiance (lights, {}, 4096, mismatch);
        check (closeRelative (e, 2.0, 0.005), "sun disk irradiance " + describe (e) + " equals 2");
        check (mismatch < 1.0e-3, "a sampled sun direction hits the disk with the same pdf");
    }

    // Rect and disk lights against a brute-force integral.
    for (const bool disk : { false, true })
        for (const Vec3 p : { Vec3 { 0.0f, 0.0f, 0.0f }, Vec3 { 0.8f, 0.0f, 0.3f }, Vec3 { 3.0f, 0.0f, 0.0f } })
        {
            pt::AreaLightRecord area;
            area.shape = disk ? pt::AreaLightShape::Disk : pt::AreaLightShape::Rect;
            area.position = { 0.0f, 1.0f, 0.0f };
            area.rotation = kDown;
            area.sizeX = 1.0f;
            area.sizeY = 0.6f;
            const std::vector<pt::AreaLightRecord> areas { area };
            pt::LightSet lights;
            lights.build (nullptr, 0, {}, empty, noMaterials, fallback, nullptr, nullptr, &areas);
            const std::string name = std::string (disk ? "disk" : "rect") + " light at x " + describe (p.x);
            check (lights.analyticCount() == 1 && lights.stats().areaLights == 1, name + ": one area light");
            const Vec3 axisX = pt::rotate (kDown, { 0.5f, 0.0f, 0.0f });
            const Vec3 axisY = pt::rotate (kDown, { 0.0f, 0.3f, 0.0f });
            const double expected = referenceIrradiance (p, area.position, axisX, axisY, disk);
            const double e = sampledIrradiance (lights, p, 8192, mismatch);
            check (closeRelative (e, expected, 0.01), name + ": irradiance " + describe (e) + " vs " + describe (expected));
            check (mismatch < 2.0e-3, name + ": a sampled direction hits the light with the same pdf");
            // One-sided: nothing above it.
            pt::LightSample s;
            check (! lights.sample ({ 0.0f, 2.0f, 0.0f }, 0.5f, { 0.3f, 0.6f }, s), name + ": a one-sided light is dark behind");
        }

    // Bad sizes fall back to delta lights.
    {
        const std::vector<grid::SceneLightRecord> records { pointRecord ({ 0.0f, 3.0f, 0.0f }, 9.0f),
                                                            directionalRecord ({ 0.0f, 1.0f, 0.0f }, 1.0f) };
        std::vector<pt::LightShape> shapes (2);
        shapes[0].radius = std::numeric_limits<float>::quiet_NaN();
        shapes[1].angularRadius = -1.0f;
        pt::LightSet lights;
        lights.build (records.data(), records.size(), {}, empty, noMaterials, fallback, nullptr, &shapes);
        check (lights.size() == 2 && lights.analyticCount() == 0, "invalid sizes keep the delta lights");
    }
}

// Shadow of a straight edge: a larger sphere gives a wider penumbra.
int penumbraColumns (float radius)
{
    const auto renderFloor = [radius] (bool occluded) {
        pt::RenderScene scene;
        scene.materials = { diffuse ({ 0.8f, 0.8f, 0.8f }), diffuse ({ 0.0f, 0.0f, 0.0f }) };
        appendMesh (scene, quad ({ 0.0f, 0.0f, 0.0f }, { 4.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, -4.0f }), 0);
        if (occluded)
            appendMesh (scene, quad ({ -3.0f, 2.0f, 0.0f }, { 3.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, -6.0f }), 1);
        scene.lightRecords = { pointRecord ({ 0.0f, 4.0f, 0.0f }, 16.0f) };
        std::vector<pt::LightShape> shapes (1);
        shapes[0].radius = radius;
        scene.lightShapes = shapes;
        scene.commit();
        const pt::PinholeCamera camera = makeCamera ({ 0.0f, 1.5f, 0.0f }, kDown, 1.0f, 64, 64);
        pt::IntegratorSettings settings;
        settings.maxDepth = 1;
        return render (scene, camera, settings, 64);
    };
    const Image lit = renderFloor (false);
    const Image shadowed = renderFloor (true);
    check (lit.stats.nonFiniteSamples == 0 && shadowed.stats.nonFiniteSamples == 0, "penumbra renders are finite");
    int columns = 0;
    int umbra = 0;
    int full = 0;
    for (std::uint32_t x = 0; x < lit.width; ++x)
    {
        double a = 0.0, b = 0.0;
        for (std::uint32_t y = 0; y < lit.height; ++y)
        {
            a += meanOf (lit.at (x, y));
            b += meanOf (shadowed.at (x, y));
        }
        const double ratio = a > 0.0 ? b / a : 0.0;
        if (ratio > 0.1 && ratio < 0.9)
            ++columns;
        else if (ratio <= 0.1)
            ++umbra;
        else
            ++full;
    }
    check (umbra > 10 && full > 10, "the floor shows both shadow and light at radius " + describe (radius));
    return columns;
}

void areaLightTests()
{
    sizedLightUnitTests();

    const int narrow = penumbraColumns (0.05f);
    const int wide = penumbraColumns (0.4f);
    check (narrow <= 5 && wide >= 9 && wide > 2 * narrow,
           "the penumbra widens with the light radius: " + std::to_string (narrow) + " then " + std::to_string (wide)
               + " columns");

    // MIS, light sampling only and BSDF sampling only agree for every shape.
    pt::RenderScene scene;
    scene.materials = { diffuse ({ 0.8f, 0.8f, 0.8f }) };
    appendMesh (scene, quad ({ 0.0f, 0.0f, 0.0f }, { 5.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, -5.0f }), 0);
    scene.lightRecords = { pointRecord ({ -1.5f, 2.5f, 0.0f }, 10.0f), directionalRecord ({ 0.3f, 1.0f, 0.2f }, 0.5f) };
    std::vector<pt::LightShape> shapes (2);
    shapes[0].radius = 0.6f;
    shapes[1].angularRadius = 0.25f;
    scene.lightShapes = shapes;
    pt::AreaLightRecord rect;
    rect.position = { 1.5f, 2.5f, 0.0f };
    rect.rotation = kDown;
    rect.sizeX = 1.2f;
    rect.sizeY = 1.2f;
    rect.intensity = 3.0f;
    pt::AreaLightRecord disk = rect;
    disk.shape = pt::AreaLightShape::Disk;
    disk.position = { 0.0f, 2.5f, 1.5f };
    scene.areaLights = { rect, disk };
    scene.commit();
    check (scene.lights.analyticCount() == 4, "sphere, sun, rect and disk are analytic lights");
    const pt::PinholeCamera camera = makeCamera ({ 0.0f, 6.0f, 0.0f }, kDown, 0.6f, 32, 32);
    const pt::LightStrategy strategies[] { pt::LightStrategy::MultipleImportance, pt::LightStrategy::LightOnly,
                                           pt::LightStrategy::BsdfOnly };
    // Each strategy renders 512 samples per pixel as 8 independent
    // generations of 64. The spread of the 8 image means estimates the
    // standard error of their average, so the agreement bound follows from
    // the measured noise instead of a fixed percentage. BSDF sampling alone
    // hits each light with probability about 0.05, a relative deviation near
    // 2 per sample, which puts the standard error near 0.3% at 512 x 32 x 32
    // samples.
    constexpr int kBatches = 8;
    double means[3] {};
    double standardErrors[3] {};
    for (int i = 0; i < 3; ++i)
    {
        pt::IntegratorSettings settings;
        settings.maxDepth = 1;
        settings.strategy = strategies[i];
        double batchMeans[kBatches] {};
        for (int b = 0; b < kBatches; ++b)
        {
            const Image image = render (scene, camera, settings, 64, 3 + static_cast<std::uint64_t> (b));
            check (image.stats.nonFiniteSamples == 0, "sized light renders are finite");
            const std::array<double, 3> c = channelMeans (image);
            batchMeans[b] = (c[0] + c[1] + c[2]) / 3.0;
            means[i] += batchMeans[b] / kBatches;
        }
        double squares = 0.0;
        for (const double m : batchMeans)
            squares += (m - means[i]) * (m - means[i]);
        standardErrors[i] = std::sqrt (squares / (kBatches - 1) / kBatches);
    }
    check (means[0] > 0.05, "the floor is lit: " + describe (means[0]));
    for (int i = 1; i < 3; ++i)
    {
        const std::string name = i == 1 ? "light sampling only" : "BSDF sampling only";
        const double error = std::sqrt (standardErrors[0] * standardErrors[0] + standardErrors[i] * standardErrors[i]);
        // The comparison must resolve a 3% bias: four standard errors of the
        // difference stay below it.
        check (4.0 * error <= 0.03 * means[0],
               name + " against MIS resolves 3%: standard error " + describe (error) + " of " + describe (means[0]));
        // Five standard errors: with at least 7 degrees of freedom behind the
        // estimate, Student's t puts a larger gap below 1 in 1000. 0.5% more
        // covers the float sums.
        check (std::fabs (means[i] - means[0]) <= 5.0 * error + 0.005 * means[0],
               name + " agrees with MIS: " + describe (means[i]) + " vs " + describe (means[0]) + ", standard error "
                   + describe (error));
    }
}

// ---------------------------------------------------------------- non-finite inputs

void nonFiniteTests()
{
    std::mt19937 rng (20261015u);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    int bad = 0;
    int pdfMismatch = 0;
    int nonDelta = 0;
    const pt::DielectricTransport transports[] { pt::DielectricTransport::ClosedVolume, pt::DielectricTransport::Interface,
                                                 pt::DielectricTransport::ThinSheet };
    for (int i = 0; i < 20000; ++i)
    {
        pt::Material m;
        m.baseColor = { u01 (rng) * 1.2f, u01 (rng), u01 (rng) };
        m.metallic = u01 (rng) < 0.2f ? u01 (rng) : 0.0f;
        m.roughness = u01 (rng) < 0.4f ? 0.0f : u01 (rng);
        m.transmission = u01 (rng) < 0.7f ? 1.0f : u01 (rng);
        m.ior = 0.5f + 3.5f * u01 (rng);
        m.dispersion = u01 (rng) < 0.5f ? 0.03f * u01 (rng) : (u01 (rng) < 0.1f ? nan : 0.0f);
        m.abbe = u01 (rng) < 0.2f ? 60.0f * u01 (rng) : (u01 (rng) < 0.05f ? inf : 0.0f);
        m.transport = transports[i % 3];
        m.absorptionDistance = u01 (rng) < 0.5f ? 2.0f * u01 (rng) : 0.0f;
        const pt::Wavelengths wl = u01 (rng) < 0.8f ? pt::sampleHeroWavelengths (u01 (rng)) : pt::rgbChannels();
        pt::DielectricSetup setup;
        for (float& e : setup.eta)
            e = 0.2f + 3.0f * u01 (rng);
        setup.etaD = u01 (rng) < 0.05f ? 1.0f : 0.3f + 2.5f * u01 (rng);
        setup.dispersive = u01 (rng) < 0.5f;
        if (u01 (rng) < 0.05f)
            setup.eta.fill (1.0f);
        const Vec3 n { 0.0f, 0.0f, 1.0f };
        const float c = std::max (1.0e-4f, u01 (rng));
        const float phi = 2.0f * pt::kPi * u01 (rng);
        const float sine = std::sqrt (std::max (0.0f, 1.0f - c * c));
        const Vec3 wo { sine * std::cos (phi), sine * std::sin (phi), c };
        pt::SpectralBsdf bsdf;
        pt::buildMaterialBsdf (m, n, n, wo, wl, setup, bsdf);
        pt::SpectralBsdfSample s;
        if (! bsdf.sample (wo, u01 (rng), { u01 (rng), u01 (rng) }, s))
            continue;
        if (! s.weight.finite() || ! std::isfinite (s.pdf) || ! (s.pdf > 0.0f) || ! pt::isFinite (s.wi))
            ++bad;
        for (const float r : s.channelPdfRatio)
            if (! std::isfinite (r) || r < 0.0f)
                ++bad;
        if ((s.flags & pt::kBsdfDelta) == 0)
        {
            ++nonDelta;
            const pt::SpectralBsdfEval e = bsdf.eval (wo, s.wi);
            if (! e.value.finite() || ! std::isfinite (e.pdf))
                ++bad;
            else if (std::fabs (e.pdf - s.pdf) > 1.0e-3f * std::max (1.0f, s.pdf))
                ++pdfMismatch;
        }
    }
    check (bad == 0, "random spectral materials sample finite weights and pdfs: " + std::to_string (bad) + " bad");
    check (nonDelta > 1000 && pdfMismatch <= nonDelta / 1000,
           "spectral sample pdf equals eval pdf: " + std::to_string (pdfMismatch) + " of " + std::to_string (nonDelta));

    // A path through hostile media and lights stays finite.
    pt::RenderScene scene;
    pt::Material deep = clearGlass (2.8f, 0.05f);
    deep.absorptionDistance = 1.0e-3f;
    deep.baseColor = { 0.0f, 0.5f, 1.0f };
    pt::Material bubble = clearGlass (1.01f);
    bubble.mediumPriority = 5;
    pt::Material sheet = clearGlass (1.5f, 0.02f);
    sheet.transport = pt::DielectricTransport::ThinSheet;
    sheet.roughness = 0.5f;
    scene.materials = { deep, bubble, sheet, diffuse ({ 0.5f, 0.5f, 0.5f }) };
    const Vec3 origin {};
    appendMesh (scene, uvSphere (12, 24, 1.0f), 0, &origin);
    const Vec3 bubbleCentre { 0.4f, 0.0f, 0.0f };
    appendMesh (scene, uvSphere (12, 24, 0.7f, bubbleCentre), 1, &bubbleCentre);
    appendMesh (scene, quad ({ 0.0f, 0.0f, 1.2f }, { 2.0f, 0.0f, 0.0f }, { 0.0f, 2.0f, 0.0f }), 2);
    appendMesh (scene, quad ({ 0.0f, -1.2f, 0.0f }, { 3.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, -3.0f }), 3);
    scene.lightRecords = { environmentRecord (0.3f), pointRecord ({ 0.0f, 0.0f, 0.0f }, 5.0f),
                           directionalRecord ({ 0.0f, 1.0f, 0.0f }, 2.0f) };
    std::vector<pt::LightShape> shapes (3);
    shapes[1].radius = 0.2f;
    shapes[2].angularRadius = 1.5f;
    scene.lightShapes = shapes;
    pt::AreaLightRecord rect;
    rect.position = { 0.0f, 0.0f, 0.0f };
    rect.twoSided = true;
    rect.sizeX = 0.0f;
    pt::AreaLightRecord disk;
    disk.position = { 0.0f, 2.0f, 0.0f };
    disk.shape = pt::AreaLightShape::Disk;
    disk.rotation = kDown;
    disk.twoSided = true;
    scene.areaLights = { rect, disk };
    scene.commit();
    check (scene.lights.stats().invalidLights == 1, "a zero-sized rect is refused");
    const pt::PinholeCamera camera = makeCamera ({ 0.0f, 0.0f, 3.0f }, pt::Quat {}, 0.5f, 24, 24);
    for (const pt::SpectralMode mode : { pt::SpectralMode::Off, pt::SpectralMode::On })
    {
        pt::IntegratorSettings settings;
        settings.maxDepth = 24;
        settings.spectral = mode;
        const Image image = render (scene, camera, settings, 16);
        check (image.stats.nonFiniteSamples == 0, "the hostile scene renders finite samples");
        bool finite = true;
        for (const Vec3 p : image.pixels)
            finite = finite && pt::isFinite (p);
        check (finite, "the hostile scene image is finite");
    }
}

} // namespace

int main()
{
    spectrumTests();
    prismLobeTests();
    prismImageTest();
    dispersiveFurnaceTests();
    spectralMatchesRgbTests();
    tirTests();
    nestedTests();
    thinSheetTests();
    beerLambertTests();
    areaLightTests();
    nonFiniteTests();
    std::cout << checks << " checks, " << failures << " failures\n";
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
