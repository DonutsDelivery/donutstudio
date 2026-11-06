// Tests for the CPU path tracer refraction slice (Plan 3, PT-D): hero
// wavelength spectral transport, Cauchy dispersion, nested dielectrics, thin
// sheets, Beer-Lambert absorption, sized lights with MIS, and the caustics
// that light tracing carries through glass and gratings.

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
#include "diffraction_material_admission.h"
#include "DiffractionMaterialPresets.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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
    std::vector<Vec3> caustic;  // the light-path share of pixels
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
            const pt::Rgb c = tracer.accumulator().splatMean (x, y);
            image.caustic.push_back ({ c.r, c.g, c.b });
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
            sum = sum + pt::channelsToRgb (pt::Spectrum::filled (1.0f), wl);
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

// ---------------------------------------------------------------- caustics

// Camera at position looking at target.
pt::PinholeCamera lookAt (Vec3 position, Vec3 target, float tanHalfFov, std::uint32_t width, std::uint32_t height)
{
    return makeCamera (position, fromZTo (pt::normalizeOrZero (position - target)), tanHalfFov, width, height);
}

// Triangle cross-section in the xy plane, extruded over z in +-halfDepth.
// Use appendMesh with outwardFrom inside it.
Mesh prism (const std::array<Vec3, 3>& section, float halfDepth)
{
    Mesh mesh;
    for (const float z : { -halfDepth, halfDepth })
        for (const Vec3 p : section)
            mesh.vertices.push_back ({ p.x, p.y, z });
    mesh.indices = { 0, 1, 2, 3, 4, 5 };
    for (std::uint32_t a = 0; a < 3; ++a)
    {
        const std::uint32_t b = (a + 1) % 3;
        mesh.indices.insert (mesh.indices.end(), { a, b, b + 3, a, b + 3, a + 3 });
    }
    return mesh;
}

// Torus around the vertical axis through centre. The winding puts every
// front face outward: cross(d/dv, d/du) is the outward normal.
Mesh torus (Vec3 centre, float major, float minor, std::uint32_t segments, std::uint32_t sides)
{
    Mesh mesh;
    for (std::uint32_t i = 0; i < segments; ++i)
    {
        const float u = 2.0f * pt::kPi * static_cast<float> (i) / static_cast<float> (segments);
        for (std::uint32_t j = 0; j < sides; ++j)
        {
            const float v = 2.0f * pt::kPi * static_cast<float> (j) / static_cast<float> (sides);
            const float ring = major + minor * std::cos (v);
            mesh.vertices.push_back (centre + Vec3 { ring * std::cos (u), minor * std::sin (v), ring * std::sin (u) });
        }
    }
    const auto at = [segments, sides] (std::uint32_t i, std::uint32_t j) { return (i % segments) * sides + j % sides; };
    for (std::uint32_t i = 0; i < segments; ++i)
        for (std::uint32_t j = 0; j < sides; ++j)
        {
            mesh.indices.insert (mesh.indices.end(), { at (i, j), at (i, j + 1), at (i + 1, j) });
            mesh.indices.insert (mesh.indices.end(), { at (i, j + 1), at (i + 1, j + 1), at (i + 1, j) });
        }
    return mesh;
}

// Floor point seen at pixel (x, y): the centre ray's first hit, when that
// is one of the scene's first floorTriangles triangles.
bool floorPoint (const pt::RenderScene& scene, const pt::PinholeCamera& camera, std::uint32_t x, std::uint32_t y,
                 std::uint32_t floorTriangles, Vec3& point)
{
    const pt::Ray ray = camera.generateRay (static_cast<float> (x) + 0.5f, static_cast<float> (y) + 0.5f);
    pt::Hit hit;
    if (! scene.bvh.intersect (ray, hit) || hit.triangle >= floorTriangles)
        return false;
    point = ray.origin + ray.direction * hit.t;
    return true;
}

// Mean grey value of values over the pixels that see the floor inside
// x0 < x < x1, z0 < z < z1.
double floorMean (const pt::RenderScene& scene, const pt::PinholeCamera& camera, const std::vector<Vec3>& values,
                  std::uint32_t floorTriangles, float x0, float x1, float z0, float z1, int& count)
{
    double sum = 0.0;
    count = 0;
    for (std::uint32_t y = 0; y < camera.height; ++y)
        for (std::uint32_t x = 0; x < camera.width; ++x)
        {
            Vec3 p;
            if (! floorPoint (scene, camera, x, y, floorTriangles, p) || ! (p.x > x0 && p.x < x1 && p.z > z0 && p.z < z1))
                continue;
            sum += meanOf (values[static_cast<std::size_t> (y) * camera.width + x]);
            ++count;
        }
    return sum / std::max (1, count);
}

// Light-path share of the whole image.
double causticShare (const Image& image)
{
    double splats = 0.0;
    double total = 0.0;
    for (std::size_t i = 0; i < image.pixels.size(); ++i)
    {
        splats += meanOf (image.caustic[i]);
        total += meanOf (image.pixels[i]);
    }
    return total > 0.0 ? splats / total : 0.0;
}

const char* const kLightTypeNames[] { "directional", "point", "spot", "environment", "triangle", "rect", "disk" };

// One emitter's splats are all recorded under its type.
bool splatsOnlyFrom (const pt::RenderStats& stats, pt::LightType type)
{
    const auto index = static_cast<std::size_t> (type);
    return stats.causticSplats > 0 && stats.causticSplatsByLight[index] == stats.causticSplats;
}

// A floating glass slab under a sun straight overhead. The floor under it is
// lit only through two interfaces, which camera paths cannot connect to a
// delta light, so light tracing carries all of it. Every order of internal
// reflection exits straight down, so the floor receives the slab's
// transmittance (1 - R) / (1 + R) of the open floor's irradiance, with
// R = 0.04 at n = 1.5: 0.923.
void slabCausticTests()
{
    pt::RenderScene scene;
    scene.materials = { diffuse ({ 0.5f, 0.5f, 0.5f }), clearGlass (1.5f) };
    appendMesh (scene, quad ({ 0.0f, 0.0f, 0.0f }, { 2.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, -2.0f }), 0);
    const Vec3 centre { 0.0f, 1.1f, 0.0f };
    appendMesh (scene, box ({ -0.5f, 1.0f, -0.5f }, { 0.5f, 1.2f, 0.5f }), 1, &centre);
    scene.lightRecords = { directionalRecord ({ 0.0f, 1.0f, 0.0f }, 1.0f) };
    scene.commit();
    // Below the slab, so the camera sees the floor under it directly. tanHalfFov is
    // vertical, so this is a horizontal half angle of tan 0.6.
    const pt::PinholeCamera camera = lookAt ({ 0.0f, 0.8f, 2.5f }, { 0.0f, 0.0f, 0.0f }, 0.4f, 48, 32);
    pt::IntegratorSettings settings;
    settings.maxDepth = 8;
    settings.spectral = pt::SpectralMode::Off;
    const Image image = render (scene, camera, settings, 64, 11);
    int underCount = 0;
    int litCount = 0;
    const double under = floorMean (scene, camera, image.pixels, 2, -0.35f, 0.35f, -0.35f, 0.35f, underCount);
    const double underSplats = floorMean (scene, camera, image.caustic, 2, -0.35f, 0.35f, -0.35f, 0.35f, underCount);
    const double lit = floorMean (scene, camera, image.pixels, 2, 0.8f, 1.6f, -0.35f, 0.35f, litCount);
    const double litSplats = floorMean (scene, camera, image.caustic, 2, 0.8f, 1.6f, -0.35f, 0.35f, litCount);
    check (underCount >= 20 && litCount >= 20,
           "the camera sees the floor under and beside the slab: " + std::to_string (underCount) + ", "
               + std::to_string (litCount) + " pixels");
    check (image.stats.nonFiniteSamples == 0 && splatsOnlyFrom (image.stats, pt::LightType::Directional),
           "slab caustic splats are finite and recorded as directional");
    check (closeRelative (lit, 0.5 / pt::kPi, 0.03), "the open floor shows the sun's diffuse radiance: " + describe (lit));
    // About 1800 light paths cross the measured patch, a 2.4% standard
    // error on 0.923. A camera path counting the same light again would put
    // the ratio near 1.9, and a missing caustic near 0.
    check (under > 0.84 * lit && under < 1.0 * lit,
           "the floor under the slab receives its transmittance: " + describe (under / lit) + " of the open floor");
    check (underSplats > 0.95 * under, "light paths carry the light under the slab: " + describe (underSplats) + " of "
                                           + describe (under));
    check (litSplats < 1.0e-3 * lit, "light paths add nothing to the open floor: " + describe (litSplats));

    settings.causticLightPaths = false;
    const Image off = render (scene, camera, settings, 16, 11);
    const double dark = floorMean (scene, camera, off.pixels, 2, -0.35f, 0.35f, -0.35f, 0.35f, underCount);
    // Camera paths still bring what the slab mirrors down from the open floor: floor,
    // slab, floor, sun. A Monte Carlo of that path gives 0.0052 of the open floor with a
    // standard error of 0.0022 over these 34 pixels at 16 samples. Four of those errors
    // above it is 0.014. The missing caustic itself would be 0.92.
    check (off.stats.lightPaths == 0 && off.stats.causticSplats == 0 && dark < 0.015 * lit,
           "without light paths the floor under the slab is dark: " + describe (dark));
}

// A glass ring over a floor, once per emitter type. The ring's shadow
// carries the light it refracts, so the light-path share of the image is
// near the ring's share of the floor's light, about the transmitted 0.85 of
// the intercepted flux: about 10% under the point and disk lights above
// it, 4% under the sun and 3% under the uniform sky, whose flux the ring
// intercepts over half its 4.7 m^2 surface against 29 m^2 of visible floor.
// 0.005 stays six times below the smallest.
void ringCausticTests()
{
    const Vec3 ringCentre { 0.0f, 1.0f, 0.0f };
    const Mesh ring = torus (ringCentre, 0.6f, 0.2f, 48, 16);
    bool outward = true;
    for (std::size_t i = 0; i + 2 < ring.indices.size(); i += 3)
    {
        const Vec3 p0 = ring.vertices[ring.indices[i]];
        const Vec3 c = (p0 + ring.vertices[ring.indices[i + 1]] + ring.vertices[ring.indices[i + 2]]) * (1.0f / 3.0f);
        const Vec3 n = pt::cross (ring.vertices[ring.indices[i + 1]] - p0, ring.vertices[ring.indices[i + 2]] - p0);
        const Vec3 radial = pt::normalizeOrZero (Vec3 { c.x - ringCentre.x, 0.0f, c.z - ringCentre.z });
        const Vec3 tube = ringCentre + radial * 0.6f;
        outward = outward && pt::dot (n, c - tube) > 0.0f;
    }
    check (outward, "the ring's faces point out of its tube");

    pt::RenderScene base;
    base.materials = { diffuse ({ 0.8f, 0.8f, 0.8f }), clearGlass (1.5f) };
    appendMesh (base, quad ({ 0.0f, 0.0f, 0.0f }, { 3.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, -3.0f }), 0);
    appendMesh (base, ring, 1);
    const pt::PinholeCamera camera = makeCamera ({ 0.0f, 3.0f, 0.0f }, kDown, 0.9f, 32, 32);
    pt::IntegratorSettings settings;
    settings.maxDepth = 8;
    settings.spectral = pt::SpectralMode::Off;

    pt::AreaLightRecord disk;
    disk.shape = pt::AreaLightShape::Disk;
    disk.position = { 0.2f, 3.5f, 0.0f };
    disk.rotation = kDown;
    disk.sizeX = 0.6f;
    disk.sizeY = 0.6f;
    disk.intensity = 4.0f;
    for (const pt::LightType type : { pt::LightType::Point, pt::LightType::Directional, pt::LightType::Environment,
                                      pt::LightType::Disk })
    {
        pt::RenderScene scene = base;
        if (type == pt::LightType::Point)
            scene.lightRecords = { pointRecord ({ 0.3f, 3.5f, 0.2f }, 8.0f) };
        else if (type == pt::LightType::Directional)
            scene.lightRecords = { directionalRecord ({ 0.2f, 1.0f, 0.1f }, 1.0f) };
        else if (type == pt::LightType::Environment)
            scene.lightRecords = { environmentRecord (0.5f) };
        else
            scene.areaLights = { disk };
        scene.commit();
        const std::string name = kLightTypeNames[static_cast<std::size_t> (type)];
        check (scene.anyMediumBoundary && scene.lights.size() == 1, name + ": one light over a glass ring");
        const Image image = render (scene, camera, settings, 32, 5);
        const double share = causticShare (image);
        bool finite = true;
        for (const Vec3 p : image.pixels)
            finite = finite && pt::isFinite (p);
        check (image.stats.nonFiniteSamples == 0 && finite, name + ": the ring renders finite pixels");
        check (image.stats.lightPaths == 32u * 32u * 32u, name + ": one light path per camera sample");
        check (splatsOnlyFrom (image.stats, type),
               name + ": the ring's caustic splats are recorded under the " + name + " light: "
                   + std::to_string (image.stats.causticSplats));
        check (share > 0.005 && share < 0.5, name + ": the ring's caustic carries " + describe (share) + " of the image");
    }
}

// Dispersion across two interfaces: a 30 degree wedge under a sun straight
// overhead. Light enters the tilted top and leaves through the flat bottom
// bent toward +x, blue more than red, so the caustic's blue centroid lies
// past its red one on the floor 1 m below. Snell's law at 450 and 650 nm
// predicts the separation. The RGB channels overlap in wavelength, but their
// negative lobes push the centroids apart again, to about the prediction.
void dispersionCausticTests()
{
    const float cauchyB = 0.05f;
    const float alpha = 30.0f * static_cast<float> (kDegree);
    pt::RenderScene scene;
    scene.materials = { diffuse ({ 0.8f, 0.8f, 0.8f }), clearGlass (1.5f, cauchyB) };
    appendMesh (scene, quad ({ 0.0f, 0.0f, 0.0f }, { 2.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, -2.0f }), 0);
    const std::array<Vec3, 3> section { { { -0.5f, 1.0f, 0.0f }, { 0.5f, 1.0f, 0.0f },
                                          { 0.5f, 1.0f + std::tan (alpha), 0.0f } } };
    const Vec3 inside { 0.17f, 1.19f, 0.0f };
    appendMesh (scene, prism (section, 1.0f), 1, &inside);
    scene.lightRecords = { directionalRecord ({ 0.0f, 1.0f, 0.0f }, 1.0f) };
    scene.commit();
    check (scene.anyDispersive, "the wedge is dispersive");

    const auto floorShift = [&] (float nm) {
        const float n = pt::cauchyIor (1.5f, cauchyB, nm);
        Vec3 inGlass;
        Vec3 out;
        if (! refractDirection ({ 0.0f, -1.0f, 0.0f }, { -std::sin (alpha), std::cos (alpha), 0.0f }, n, inGlass)
            || ! refractDirection (inGlass, { 0.0f, 1.0f, 0.0f }, 1.0 / n, out))
            return 0.0;
        return static_cast<double> (out.x) / static_cast<double> (-out.y);
    };
    const double predicted = floorShift (450.0f) - floorShift (650.0f);
    check (predicted > 0.05, "Snell's law spreads 450 and 650 nm by " + describe (predicted) + " m on the floor");

    const pt::PinholeCamera camera = lookAt ({ 0.3f, 0.9f, 2.5f }, { 0.3f, 0.0f, -0.2f }, 0.5f, 64, 40);
    pt::IntegratorSettings settings;
    settings.maxDepth = 8;
    const Image image = render (scene, camera, settings, 32, 13);
    check (image.stats.spectral && image.stats.nonFiniteSamples == 0, "the wedge renders spectrally and finite");
    check (splatsOnlyFrom (image.stats, pt::LightType::Directional), "the wedge's caustic is recorded as directional");
    double red = 0.0, redX = 0.0, blue = 0.0, blueX = 0.0;
    for (std::uint32_t y = 0; y < camera.height; ++y)
        for (std::uint32_t x = 0; x < camera.width; ++x)
        {
            Vec3 p;
            if (! floorPoint (scene, camera, x, y, 2, p) || ! (p.x > -0.6f && p.x < 1.2f && std::fabs (p.z) < 0.9f))
                continue;
            const Vec3 c = image.caustic[static_cast<std::size_t> (y) * camera.width + x];
            red += c.x;
            redX += static_cast<double> (c.x) * p.x;
            blue += c.z;
            blueX += static_cast<double> (c.z) * p.x;
        }
    check (red > 0.0 && blue > 0.0, "the wedge's caustic reaches the floor: " + describe (red) + ", " + describe (blue));
    const double separation = blue > 0.0 && red > 0.0 ? blueX / blue - redX / red : 0.0;
    check (separation > 0.2 * predicted && separation < 1.2 * predicted,
           "the caustic's blue lands past its red by " + describe (separation) + " m of " + describe (predicted));

    // The caster splits white light but adds no colour, so the caustic's total
    // stays white, while its two sides differ. Counting only the positive part
    // of each wavelength's colour makes the red total 1.48 / 1.12 = 1.32 times
    // the green and washes the sides out.
    std::array<double, 3> total { 0.0, 0.0, 0.0 };
    std::array<double, 3> nearSide { 0.0, 0.0, 0.0 };
    std::array<double, 3> farSide { 0.0, 0.0, 0.0 };
    const double middle = 0.5 * (blueX / blue + redX / red);
    for (std::uint32_t y = 0; y < camera.height; ++y)
        for (std::uint32_t x = 0; x < camera.width; ++x)
        {
            Vec3 p;
            if (! floorPoint (scene, camera, x, y, 2, p) || ! (p.x > -0.6f && p.x < 1.2f && std::fabs (p.z) < 0.9f))
                continue;
            const Vec3 c = image.caustic[static_cast<std::size_t> (y) * camera.width + x];
            auto& side = p.x < middle ? nearSide : farSide;
            const std::array<double, 3> channels { c.x, c.y, c.z };
            for (std::size_t k = 0; k < 3; ++k)
            {
                total[k] += channels[k];
                side[k] += channels[k];
            }
        }
    check (closeRelative (total[0], total[1], 0.15) && closeRelative (total[2], total[1], 0.15),
           "the dispersive caustic's total stays white: " + describe (total[0]) + ", " + describe (total[1]) + ", "
               + describe (total[2]));
    const double nearRatio = nearSide[0] / std::max (std::fabs (nearSide[2]), 1.0e-9);
    const double farRatio = farSide[0] / std::max (std::fabs (farSide[2]), 1.0e-9);
    // The 8 cm split on a beam about 1 m wide moves 4% of each band across the
    // middle: about 0.54 / 0.46 = 1.17 and its inverse.
    check (nearRatio > 1.05 && farRatio < 0.95,
           "the caustic is red on its near side and blue on its far side: red over blue "
               + describe (nearRatio) + ", " + describe (farRatio));
}

// Light leaving after total internal reflection: a 45-45-90 prism turns a
// horizontal sun straight down. It enters and leaves at normal incidence and
// reflects totally at the hypotenuse (45 degrees past the 41.8 critical
// angle), so the floor under it receives (1 - R)^2 = 0.9216 of the sun. The
// sun grazes the floor itself, so nothing else lights it.
void tirCausticTests()
{
    pt::RenderScene scene;
    scene.materials = { diffuse ({ 0.8f, 0.8f, 0.8f }), clearGlass (1.5f) };
    appendMesh (scene, quad ({ 0.0f, 0.0f, 0.0f }, { 2.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, -2.0f }), 0);
    const std::array<Vec3, 3> section { { { 0.5f, 1.5f, 0.0f }, { 0.5f, 2.5f, 0.0f }, { -0.5f, 1.5f, 0.0f } } };
    const Vec3 inside { 0.17f, 1.83f, 0.0f };
    appendMesh (scene, prism (section, 1.0f), 1, &inside);
    scene.lightRecords = { directionalRecord ({ 1.0f, 0.0f, 0.0f }, 1.0f) };
    scene.commit();
    const pt::PinholeCamera camera = lookAt ({ 0.0f, 1.2f, 3.0f }, { 0.0f, 0.0f, 0.0f }, 0.5f, 48, 32);
    pt::IntegratorSettings settings;
    settings.maxDepth = 8;
    settings.spectral = pt::SpectralMode::Off;
    const double expected = 0.9216 * 0.8 / pt::kPi;

    // About 3600 light paths cross the measured patch: a 1.7% standard error.
    const Image image = render (scene, camera, settings, 64, 17);
    int count = 0;
    const double patch = floorMean (scene, camera, image.pixels, 2, -0.35f, 0.35f, -0.8f, 0.8f, count);
    check (count >= 20, "the camera sees the floor under the prism: " + std::to_string (count) + " pixels");
    check (image.stats.nonFiniteSamples == 0 && splatsOnlyFrom (image.stats, pt::LightType::Directional),
           "the prism's caustic is finite and recorded as directional");
    check (patch > 0.9 * expected && patch < 1.08 * expected,
           "light leaving after total internal reflection lights the floor: " + describe (patch) + " vs "
               + describe (expected));

    settings.causticLightPaths = false;
    const Image off = render (scene, camera, settings, 16, 17);
    const double dark = floorMean (scene, camera, off.pixels, 2, -0.35f, 0.35f, -0.8f, 0.8f, count);
    check (dark < 1.0e-3 * expected, "without light paths the floor under the prism is dark: " + describe (dark));
}

// A grooved disc: a fan of triangles whose radial first edges give the
// grating its groove frame, bound to the aluminium grating. Under a point
// light its orders reflect onto a wall behind it. The disc intercepts about
// 0.2 sr of the light against about 1.2 sr for the wall, so the reflected
// orders give the wall over 10% of its light; 0.005 is a twentieth of that.
void gratingCausticTests()
{
    std::string error;
    const auto admitted = diffractionmaterial::admit (diffractionmaterial::makeAluminiumBinaryGratingPreset(), error);
    check (admitted.has_value(), "the aluminium binary grating preset admits: " + error);
    if (! admitted)
        return;

    pt::RenderScene scene;
    scene.materials = { diffuse ({ 0.8f, 0.8f, 0.8f }), diffuse ({ 0.2f, 0.2f, 0.2f }) };
    appendMesh (scene, quad ({ 0.0f, 1.5f, -1.5f }, { 2.0f, 0.0f, 0.0f }, { 0.0f, 1.5f, 0.0f }), 0);
    Mesh disc;
    const Vec3 discCentre { 0.0f, 0.05f, 0.0f };
    disc.vertices.push_back (discCentre);
    constexpr std::uint32_t kSegments = 48;
    for (std::uint32_t k = 0; k < kSegments; ++k)
    {
        const float phi = 2.0f * pt::kPi * static_cast<float> (k) / static_cast<float> (kSegments);
        disc.vertices.push_back (discCentre + Vec3 { 0.5f * std::cos (phi), 0.0f, 0.5f * std::sin (phi) });
    }
    // (centre, next, this) faces up.
    for (std::uint32_t k = 0; k < kSegments; ++k)
        disc.indices.insert (disc.indices.end(), { 0u, 1 + (k + 1) % kSegments, 1 + k });
    appendMesh (scene, disc, 1);
    scene.gratingMaterials = { std::make_shared<const diffractionmaterial::AdmittedDiffractionMaterialIR> (*admitted) };
    scene.gratings = { { 1u, scene.gratingMaterials[0]->description() } };
    scene.lightRecords = { pointRecord ({ 0.0f, 1.5f, 1.0f }, 8.0f) };
    scene.commit();
    check (! scene.anyMediumBoundary, "light tracing runs for the grating alone");

    const pt::PinholeCamera camera = lookAt ({ 0.0f, 1.5f, 3.0f }, { 0.0f, 1.5f, -1.5f }, 0.6f, 48, 32);
    pt::IntegratorSettings settings;
    settings.maxDepth = 8;
    const Image image = render (scene, camera, settings, 32, 19);
    const double share = causticShare (image);
    check (image.stats.spectral && image.stats.nonFiniteSamples == 0, "the grooved disc renders spectrally and finite");
    check (splatsOnlyFrom (image.stats, pt::LightType::Point),
           "the grooved disc's orders splat as point light: " + std::to_string (image.stats.causticSplats));
    check (share > 0.005 && share < 0.5, "the grooved disc's orders carry " + describe (share) + " of the image");

    settings.causticLightPaths = false;
    const Image off = render (scene, camera, settings, 8, 19);
    check (off.stats.lightPaths == 0 && causticShare (off) == 0.0, "the setting turns grating light paths off");
}

// A rough grating faces the camera with a small lamp behind the camera, in a black room.
// Sigma = 2 x 0.02 = 0.04 rad, and the lamp subtends 0.15 / 4 = 0.0375 rad on a side.
// Without a light sample an order reaches the lamp by chance, in about 0.0014 / 0.01 of
// its samples (lamp solid angle against 2 pi sigma^2), so a pixel in the arm averages few
// large hits. Noise is the RMS difference of two seeds; its standard error sets the mean's
// tolerance. A light sample needs more than half the noise gone to count; the true gain
// is far larger, so 0.5 does not sit near the result.
void gratingOrderLightSamplingTests()
{
    std::string error;
    auto description = diffractionmaterial::makeAluminiumBinaryGratingPreset();
    description.roughness.rmsSlope = 0.02f;
    const auto admitted = diffractionmaterial::admit (description, error);
    check (admitted.has_value(), "the rough grating admits: " + error);
    if (! admitted)
        return;

    pt::RenderScene scene;
    scene.materials = { diffuse ({ 0.0f, 0.0f, 0.0f }), lamp (200.0f) };
    appendMesh (scene, quad ({ 0.0f, 0.0f, 0.0f }, { 1.5f, 0.0f, 0.0f }, { 0.0f, 1.5f, 0.0f }), 0);
    appendMesh (scene, quad ({ 0.0f, 0.0f, 4.0f }, { 0.0f, 0.075f, 0.0f }, { 0.075f, 0.0f, 0.0f }), 1);
    scene.gratingMaterials = { std::make_shared<const diffractionmaterial::AdmittedDiffractionMaterialIR> (*admitted) };
    scene.gratings = { { 0u, scene.gratingMaterials[0]->description() } };
    scene.commit();
    const pt::PinholeCamera camera = lookAt ({ 0.0f, 0.0f, 2.0f }, { 0.0f, 0.0f, 0.0f }, 0.9f, 48, 48);

    const auto rendered = [&] (pt::LightStrategy strategy, std::uint64_t generation) {
        pt::IntegratorSettings settings;
        settings.maxDepth = 4;
        settings.causticLightPaths = false;
        settings.strategy = strategy;
        return render (scene, camera, settings, 64, generation);
    };
    const auto meanOfImage = [] (const Image& image) {
        double total = 0.0;
        for (const Vec3 pixel : image.pixels)
            total += meanOf (pixel);
        return total / static_cast<double> (image.pixels.size());
    };
    const auto noiseBetween = [] (const Image& a, const Image& b) {
        double total = 0.0;
        for (std::size_t i = 0; i < a.pixels.size(); ++i)
        {
            const double difference = meanOf (a.pixels[i]) - meanOf (b.pixels[i]);
            total += difference * difference;
        }
        return std::sqrt (total / static_cast<double> (a.pixels.size()));
    };
    const Image chance1 = rendered (pt::LightStrategy::BsdfOnly, 31);
    const Image chance2 = rendered (pt::LightStrategy::BsdfOnly, 32);
    const Image sampled1 = rendered (pt::LightStrategy::MultipleImportance, 31);
    const Image sampled2 = rendered (pt::LightStrategy::MultipleImportance, 32);
    const Image lightOnly = rendered (pt::LightStrategy::LightOnly, 31);
    check (sampled1.stats.nonFiniteSamples == 0 && chance1.stats.nonFiniteSamples == 0, "a lit grating stays finite");

    const double chanceNoise = noiseBetween (chance1, chance2);
    const double sampledNoise = noiseBetween (sampled1, sampled2);
    check (meanOfImage (sampled1) > 0.0, "the grating shows the lamp");
    check (sampledNoise < 0.5 * chanceNoise,
           "a light sample through the orders cuts the noise: " + describe (sampledNoise) + " from " + describe (chanceNoise));

    // Means over 2304 pixels: the image's standard error is the per-pixel difference over
    // sqrt (2 x 2304), and 4 of them leaves room for a true fluctuation only.
    const double tolerance = 4.0 * chanceNoise / std::sqrt (2.0 * 2304.0);
    const double chanceMean = 0.5 * (meanOfImage (chance1) + meanOfImage (chance2));
    check (std::abs (meanOfImage (sampled1) - chanceMean) < tolerance,
           "MIS keeps the mean: " + describe (meanOfImage (sampled1)) + " against " + describe (chanceMean));
    check (std::abs (meanOfImage (lightOnly) - chanceMean) < tolerance,
           "light sampling alone keeps the mean: " + describe (meanOfImage (lightOnly)) + " against " + describe (chanceMean));

    // Energy: under a uniform environment of radiance 1 the grating returns its reflected
    // power and no more, by either estimator, and an aluminium surface returns under 1.
    pt::RenderScene furnace;
    furnace.materials = { diffuse ({ 0.0f, 0.0f, 0.0f }) };
    appendMesh (furnace, quad ({ 0.0f, 0.0f, 0.0f }, { 1.5f, 0.0f, 0.0f }, { 0.0f, 1.5f, 0.0f }), 0);
    furnace.gratingMaterials = scene.gratingMaterials;
    furnace.gratings = { { 0u, furnace.gratingMaterials[0]->description() } };
    furnace.lightRecords = { environmentRecord (1.0f) };
    furnace.commit();
    const pt::PinholeCamera inside = lookAt ({ 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 0.0f }, 0.5f, 16, 16);
    pt::IntegratorSettings settings;
    settings.maxDepth = 2;
    settings.causticLightPaths = false;
    settings.environmentVisibleToCamera = false;
    settings.strategy = pt::LightStrategy::BsdfOnly;
    const double scattered = meanOfImage (render (furnace, inside, settings, 256, 33));
    settings.strategy = pt::LightStrategy::MultipleImportance;
    const double weighted = meanOfImage (render (furnace, inside, settings, 256, 33));
    check (scattered > 0.3 && scattered < 1.0 && weighted < 1.0,
           "the grating's furnace stays under 1: " + describe (scattered) + " and " + describe (weighted));
    check (std::abs (weighted - scattered) < 0.01,
           "the furnace is the same with a light sample: " + describe (weighted) + " against " + describe (scattered));
}

// A pressed disc has no roughness, so its orders and mirror are deltas and its pit scatter is
// the only lobe a light sample can serve. A small lamp away from every order lights the scatter
// alone. The same rule as above: same mean as scatter sampling, with more than half the noise gone.
void pitScatterLightSamplingTests()
{
    std::string error;
    auto description = diffractionmaterial::makeAluminiumBinaryGratingPreset();
    description.geometry.grooveSpacingNanometres = 1600.0f;
    description.microstructure.grooveDepthNanometres = 124.1f;
    description.microstructure.dutyCycle = 0.3125f;
    description.pitStatistics = diffractionmaterial::PitStatistics::PressedDisc;
    description.roughness.rmsSlope = 0.0f;
    description.spectrum.lastOrder = 3;
    const auto admitted = diffractionmaterial::admit (description, error);
    check (admitted.has_value(), "the pressed disc admits: " + error);
    if (! admitted)
        return;

    pt::RenderScene scene;
    scene.materials = { diffuse ({ 0.0f, 0.0f, 0.0f }), lamp (200.0f) };
    appendMesh (scene, quad ({ 0.0f, 0.0f, 0.0f }, { 1.5f, 0.0f, 0.0f }, { 0.0f, 1.5f, 0.0f }), 0);
    appendMesh (scene, quad ({ 0.0f, 0.0f, 4.0f }, { 0.0f, 0.075f, 0.0f }, { 0.075f, 0.0f, 0.0f }), 1);
    scene.gratingMaterials = { std::make_shared<const diffractionmaterial::AdmittedDiffractionMaterialIR> (*admitted) };
    scene.gratings = { { 0u, scene.gratingMaterials[0]->description() } };
    scene.commit();
    const pt::PinholeCamera camera = lookAt ({ 0.0f, 0.0f, 2.0f }, { 0.0f, 0.0f, 0.0f }, 0.9f, 48, 48);

    const auto rendered = [&] (pt::LightStrategy strategy, std::uint64_t generation) {
        pt::IntegratorSettings settings;
        settings.maxDepth = 4;
        settings.causticLightPaths = false;
        settings.strategy = strategy;
        return render (scene, camera, settings, 64, generation);
    };
    const auto meanOfImage = [] (const Image& image) {
        double total = 0.0;
        for (const Vec3 pixel : image.pixels)
            total += meanOf (pixel);
        return total / static_cast<double> (image.pixels.size());
    };
    // The light sample covers pit scatter only. At slope 0 the mirror and the orders are
    // deltas that hit the lamp by chance of wavelength, in either strategy. So measure where
    // none of them can reach it: the lamp's offset from the mirror direction, in direction
    // cosines, clears the mirror and stays under the first order's 380 / 1600 nm whatever the
    // groove axis. The margin 0.055 is the lamp's half diagonal at 4 units, 0.027, plus half
    // a pixel, 0.028.
    std::vector<std::size_t> scatterOnly;
    for (std::uint32_t y = 0; y < camera.height; ++y)
        for (std::uint32_t x = 0; x < camera.width; ++x)
        {
            const pt::Ray ray = camera.generateRay (static_cast<float> (x) + 0.5f, static_cast<float> (y) + 0.5f);
            const Vec3 p = ray.origin + ray.direction * (-ray.origin.z / ray.direction.z);
            const float radius = std::sqrt (p.x * p.x + p.y * p.y);
            const float offset = radius * (1.0f / std::sqrt (radius * radius + 4.0f) + 1.0f / std::sqrt (radius * radius + 16.0f));
            if (offset > 0.055f && offset < 380.0f / 1600.0f - 0.055f)
                scatterOnly.push_back (static_cast<std::size_t> (y) * camera.width + x);
        }
    check (scatterOnly.size() >= 20, "pit scatter alone reaches the lamp over enough pixels: " + std::to_string (scatterOnly.size()));
    const auto noiseBetween = [&scatterOnly] (const Image& a, const Image& b) {
        double total = 0.0;
        for (const std::size_t i : scatterOnly)
        {
            const double difference = meanOf (a.pixels[i]) - meanOf (b.pixels[i]);
            total += difference * difference;
        }
        return std::sqrt (total / static_cast<double> (std::max<std::size_t> (scatterOnly.size(), 1)));
    };
    const Image chance1 = rendered (pt::LightStrategy::BsdfOnly, 41);
    const Image chance2 = rendered (pt::LightStrategy::BsdfOnly, 42);
    const Image sampled1 = rendered (pt::LightStrategy::MultipleImportance, 41);
    const Image sampled2 = rendered (pt::LightStrategy::MultipleImportance, 42);
    check (sampled1.stats.nonFiniteSamples == 0 && chance1.stats.nonFiniteSamples == 0, "a lit pressed disc stays finite");

    const double chanceNoise = noiseBetween (chance1, chance2);
    const double sampledNoise = noiseBetween (sampled1, sampled2);
    check (meanOfImage (sampled1) > 0.0, "the pit scatter shows the lamp");
    check (sampledNoise < 0.5 * chanceNoise,
           "a light sample through the pit scatter cuts the noise: " + describe (sampledNoise) + " from " + describe (chanceNoise));
    const double tolerance = 4.0 * chanceNoise / std::sqrt (2.0 * static_cast<double> (std::max<std::size_t> (scatterOnly.size(), 1)));
    const auto meanOfScatter = [&scatterOnly] (const Image& image) {
        double total = 0.0;
        for (const std::size_t i : scatterOnly)
            total += meanOf (image.pixels[i]);
        return total / static_cast<double> (std::max<std::size_t> (scatterOnly.size(), 1));
    };
    const double chanceMean = 0.5 * (meanOfScatter (chance1) + meanOfScatter (chance2));
    check (std::abs (meanOfScatter (sampled1) - chanceMean) < tolerance,
           "MIS keeps the pit scatter's mean: " + describe (meanOfScatter (sampled1)) + " against " + describe (chanceMean));
}

// Roughness across one groove family stretches a lamp's mirror image into a line along
// that family's axis. At slope 0.15 the line's sigma is 0.3 rad against 0.04 across it, so
// its second moment must beat the other axis's by far more than the 3 asked here, and
// light sampling must keep the mean the scatter sampling gets on its own.
void gratingAcrossGrooveRoughnessTests()
{
    std::string error;
    auto description = diffractionmaterial::makeAluminiumBinaryGratingPreset();
    description.roughness.rmsSlope = 0.02f;
    description.roughness.acrossGrooveSlope = 0.15f;
    const auto admitted = diffractionmaterial::admit (description, error);
    check (admitted.has_value(), "the grating with roughness across its grooves admits: " + error);
    if (! admitted)
        return;

    pt::RenderScene scene;
    scene.materials = { diffuse ({ 0.0f, 0.0f, 0.0f }), lamp (200.0f) };
    appendMesh (scene, quad ({ 0.0f, 0.0f, 0.0f }, { 1.5f, 0.0f, 0.0f }, { 0.0f, 1.5f, 0.0f }), 0);
    appendMesh (scene, quad ({ 0.0f, 0.0f, 4.0f }, { 0.0f, 0.075f, 0.0f }, { 0.075f, 0.0f, 0.0f }), 1);
    scene.gratingMaterials = { std::make_shared<const diffractionmaterial::AdmittedDiffractionMaterialIR> (*admitted) };
    scene.gratings = { { 0u, scene.gratingMaterials[0]->description() } };
    scene.commit();
    const pt::PinholeCamera camera = lookAt ({ 0.0f, 0.0f, 2.0f }, { 0.0f, 0.0f, 0.0f }, 0.9f, 48, 48);
    const auto rendered = [&] (pt::LightStrategy strategy, std::uint64_t generation) {
        pt::IntegratorSettings settings;
        settings.maxDepth = 4;
        settings.causticLightPaths = false;
        settings.strategy = strategy;
        return render (scene, camera, settings, 64, generation);
    };
    const Image sampled = rendered (pt::LightStrategy::MultipleImportance, 41);
    const Image chance1 = rendered (pt::LightStrategy::BsdfOnly, 41);
    const Image chance2 = rendered (pt::LightStrategy::BsdfOnly, 42);
    double total = 0.0, momentX = 0.0, momentY = 0.0, noise = 0.0, chanceMean = 0.0;
    for (std::uint32_t y = 0; y < 48; ++y)
        for (std::uint32_t x = 0; x < 48; ++x)
        {
            const double v = meanOf (sampled.at (x, y));
            const double a = meanOf (chance1.at (x, y)), b = meanOf (chance2.at (x, y));
            total += v;
            momentX += v * (x - 23.5) * (x - 23.5);
            momentY += v * (y - 23.5) * (y - 23.5);
            noise += (a - b) * (a - b);
            chanceMean += 0.5 * (a + b);
        }
    check (total > 0.0 && sampled.stats.nonFiniteSamples == 0, "the rough grating shows the lamp, finite");
    const double longer = std::max (momentX, momentY), shorter = std::min (momentX, momentY);
    check (longer > 3.0 * shorter, "the lamp's image is a line: moments " + describe (longer) + " and " + describe (shorter));
    const double tolerance = 4.0 * std::sqrt (noise / 2304.0) / std::sqrt (2.0 * 2304.0);
    check (std::abs (total - chanceMean) / 2304.0 < tolerance,
           "the light sample keeps the line's mean: " + describe (total / 2304.0) + " against " + describe (chanceMean / 2304.0));
}

// ---------------------------------------------------------------- authored normals

// Appends one triangle whose stored vertex normals are the given ones.
void appendShadedTriangle (pt::RenderScene& scene, std::array<Vec3, 3> positions, std::array<Vec3, 3> normals, bool authored,
                           std::uint32_t material = 0)
{
    pt::TriangleAttributes attributes;
    attributes.materialIndex = material;
    attributes.geometricNormal = pt::normalizeOrZero (pt::cross (positions[1] - positions[0], positions[2] - positions[0]));
    attributes.shadingNormals = normals;
    attributes.hasShadingNormals = true;
    attributes.authoredShading = authored;
    for (const Vec3 p : positions)
    {
        scene.geometry.positions.push_back (p);
        scene.geometry.bounds.grow (p);
    }
    scene.geometry.triangles.push_back (attributes);
}

// An eight sided prism along y, lit from 70 degrees off the camera, with radial
// vertex normals: the coarse mesh shaded smooth.
// The same disc under a clear body, as the Grooved Disc starter has it. A shadow ray stops at the
// body's top, so the lamp is reached only through the refracted line planarExitTo solves. One
// flat top: same mean as scattering alone, with more than half the noise gone. Past a second slab
// the line is blocked, and an arched top often has none: both keep to scattering, same mean.
void pitScatterThroughFlatTopTests()
{
    std::string error;
    auto description = diffractionmaterial::makeAluminiumBinaryGratingPreset();
    description.geometry.grooveSpacingNanometres = 1600.0f;
    description.microstructure.grooveDepthNanometres = 124.1f;
    description.microstructure.dutyCycle = 0.3125f;
    description.pitStatistics = diffractionmaterial::PitStatistics::PressedDisc;
    description.roughness.rmsSlope = 0.0f;
    description.spectrum.lastOrder = 3;
    const auto admitted = diffractionmaterial::admit (description, error);
    check (admitted.has_value(), "the pressed disc admits: " + error);
    if (! admitted)
        return;

    enum class Top { flat, secondSlab, arched };
    const auto sceneWith = [&] (Top top) {
        pt::RenderScene scene;
        scene.materials = { diffuse ({ 0.0f, 0.0f, 0.0f }), lamp (200.0f), clearGlass (1.5f) };
        appendMesh (scene, quad ({ 0.0f, 0.0f, 0.0f }, { 1.5f, 0.0f, 0.0f }, { 0.0f, 1.5f, 0.0f }), 0);
        appendMesh (scene, quad ({ 0.0f, 0.0f, 4.0f }, { 0.0f, 0.075f, 0.0f }, { 0.075f, 0.0f, 0.0f }), 1);
        if (top == Top::arched)
        {
            // Flat-shaded strips along an arc, narrower than the body is deep, so the line to a
            // lamp often leaves the strip the straight line met.
            const auto archHeight = [] (float x) { return 0.05f + 0.25f * (1.0f - x * x / 4.0f); };
            for (int strip = -20; strip < 20; ++strip)
            {
                const float x0 = 0.1f * static_cast<float> (strip);
                const float x1 = x0 + 0.1f;
                const Vec3 centre { 0.5f * (x0 + x1), 0.0f, 0.5f * (archHeight (x0) + archHeight (x1)) };
                appendMesh (scene, quad (centre, { 0.05f, 0.0f, 0.5f * (archHeight (x1) - archHeight (x0)) }, { 0.0f, 2.0f, 0.0f }), 2);
            }
        }
        else
            appendMesh (scene, box ({ -2.0f, -2.0f, -0.02f }, { 2.0f, 2.0f, 0.05f }), 2);
        if (top == Top::secondSlab)
            appendMesh (scene, box ({ -2.0f, -2.0f, 0.2f }, { 2.0f, 2.0f, 0.25f }), 2);
        scene.gratingMaterials = { std::make_shared<const diffractionmaterial::AdmittedDiffractionMaterialIR> (*admitted) };
        scene.gratings = { { 0u, scene.gratingMaterials[0]->description() } };
        scene.commit();
        return scene;
    };
    const pt::PinholeCamera camera = lookAt ({ 0.0f, 0.0f, 2.0f }, { 0.0f, 0.0f, 0.0f }, 0.9f, 48, 48);
    std::vector<std::size_t> scatterOnly;
    for (std::uint32_t y = 0; y < camera.height; ++y)
        for (std::uint32_t x = 0; x < camera.width; ++x)
        {
            const pt::Ray ray = camera.generateRay (static_cast<float> (x) + 0.5f, static_cast<float> (y) + 0.5f);
            const Vec3 p = ray.origin + ray.direction * (-ray.origin.z / ray.direction.z);
            const float radius = std::sqrt (p.x * p.x + p.y * p.y);
            const float offset = radius * (1.0f / std::sqrt (radius * radius + 4.0f) + 1.0f / std::sqrt (radius * radius + 16.0f));
            if (offset > 0.055f && offset < 380.0f / 1600.0f - 0.055f)
                scatterOnly.push_back (static_cast<std::size_t> (y) * camera.width + x);
        }
    const auto rendered = [&] (const pt::RenderScene& scene, pt::LightStrategy strategy, std::uint64_t generation) {
        pt::IntegratorSettings settings;
        settings.causticLightPaths = false;
        settings.strategy = strategy;
        return render (scene, camera, settings, 16, generation);
    };
    const auto meanOver = [&scatterOnly] (const Image& image) {
        double total = 0.0;
        for (const std::size_t i : scatterOnly)
            total += meanOf (image.pixels[i]);
        return total / static_cast<double> (std::max<std::size_t> (scatterOnly.size(), 1));
    };
    const auto noiseBetween = [&scatterOnly] (const Image& a, const Image& b) {
        double total = 0.0;
        for (const std::size_t i : scatterOnly)
        {
            const double difference = meanOf (a.pixels[i]) - meanOf (b.pixels[i]);
            total += difference * difference;
        }
        return std::sqrt (total / static_cast<double> (std::max<std::size_t> (scatterOnly.size(), 1)));
    };
    // 16 samples seldom hit so small a lamp by chance: a single pair of frames can show no hit at
    // all, which reads as no noise. So 20 pairs, 16 samples each, and the mean over all 40.
    const Top tops[] = { Top::flat, Top::secondSlab, Top::arched };
    const char* names[] = { "a flat top", "a second slab", "an arched top" };
    for (int t = 0; t < 3; ++t)
    {
        const pt::RenderScene scene = sceneWith (tops[t]);
        const std::string name = names[t];
        double chanceNoise = 0.0, sampledNoise = 0.0;
        std::vector<double> difference;
        bool finite = true;
        for (std::uint64_t pair = 0; pair < 20; ++pair)
        {
            const Image chance1 = rendered (scene, pt::LightStrategy::BsdfOnly, 100 + 2 * pair);
            const Image chance2 = rendered (scene, pt::LightStrategy::BsdfOnly, 101 + 2 * pair);
            const Image sampled1 = rendered (scene, pt::LightStrategy::MultipleImportance, 100 + 2 * pair);
            const Image sampled2 = rendered (scene, pt::LightStrategy::MultipleImportance, 101 + 2 * pair);
            finite = finite && sampled1.stats.nonFiniteSamples == 0 && sampled2.stats.nonFiniteSamples == 0;
            const double chanceBetween = noiseBetween (chance1, chance2);
            const double sampledBetween = noiseBetween (sampled1, sampled2);
            chanceNoise += chanceBetween * chanceBetween / 20.0;
            sampledNoise += sampledBetween * sampledBetween / 20.0;
            difference.push_back (meanOver (sampled1) - meanOver (chance1));
            difference.push_back (meanOver (sampled2) - meanOver (chance2));
        }
        chanceNoise = std::sqrt (chanceNoise);
        sampledNoise = std::sqrt (sampledNoise);
        double shift = 0.0, spread = 0.0;
        for (const double d : difference)
            shift += d / static_cast<double> (difference.size());
        for (const double d : difference)
            spread += (d - shift) * (d - shift) / static_cast<double> (difference.size() - 1);
        const double standardError = std::sqrt (spread / static_cast<double> (difference.size()));
        std::cout << name << ": noise " << chanceNoise << " to " << sampledNoise << ", mean shift " << shift << " +- " << standardError << "\n";
        check (finite, name + " stays finite");
        check (std::abs (shift) <= 4.0 * standardError, name + " keeps the mean: shift " + describe (shift) + " against " + describe (standardError));
        if (tops[t] == Top::flat)
            check (sampledNoise < 0.5 * chanceNoise,
                   "a light sample through the flat top cuts the noise: " + describe (sampledNoise) + " from " + describe (chanceNoise));
    }
}

// A diffuse plate under a clear body, as a pressed disc's hub is, lit only through both faces.
// The light sample must keep the mean of scattering alone.
void lightSampleThroughSlabTests()
{
    const pt::PinholeCamera camera = lookAt ({ 0.0f, 0.0f, 2.0f }, { 0.0f, 0.0f, 0.0f }, 0.9f, 48, 48);
    struct Case { const char* name; pt::Material body; float top; Vec3 lamp; bool blocker; bool cutsNoise; };
    pt::Material tinted = clearGlass (1.5f);
    tinted.baseColor = { 0.6f, 0.8f, 0.9f };
    const Case cases[] = {
        { "a thin body", clearGlass (1.5f), 0.1f, { 0.0f, 0.0f, 4.0f }, false, true },
        { "a thick body", clearGlass (1.5f), 0.8f, { 0.0f, 0.0f, 4.0f }, false, true },
        { "an oblique lamp", clearGlass (1.5f), 0.4f, { 3.0f, 0.0f, 2.0f }, false, true },
        { "a tinted body", tinted, 0.4f, { 0.0f, 0.0f, 4.0f }, false, true },
        { "a dispersive body", clearGlass (1.5f, 0.02f), 0.4f, { 0.0f, 0.0f, 4.0f }, false, true },
        { "a sheet inside the body", clearGlass (1.5f), 0.4f, { 0.0f, 0.0f, 4.0f }, true, false },
    };
    for (const Case& c : cases)
    {
        pt::RenderScene scene;
        scene.materials = { diffuse ({ 0.5f, 0.5f, 0.5f }), lamp (40.0f), c.body };
        appendMesh (scene, quad ({ 0.0f, 0.0f, 0.0f }, { 2.0f, 0.0f, 0.0f }, { 0.0f, 2.0f, 0.0f }), 0);
        appendMesh (scene, quad (c.lamp, { 0.0f, 0.3f, 0.0f }, { 0.3f, 0.0f, 0.0f }), 1);
        const Vec3 bodyCentre { 0.0f, 0.0f, 0.5f * (0.05f + c.top) };
        appendMesh (scene, box ({ -6.0f, -6.0f, 0.05f }, { 6.0f, 6.0f, c.top }), 2, &bodyCentre);
        if (c.blocker)
            appendMesh (scene, quad ({ 0.5f, 0.0f, 0.5f * (0.05f + c.top) }, { 0.5f, 0.0f, 0.0f }, { 0.0f, 2.0f, 0.0f }), 0);
        scene.commit();
        const auto rendered = [&] (pt::LightStrategy strategy, std::uint64_t generation) {
            pt::IntegratorSettings settings;
            settings.causticLightPaths = false;
            settings.strategy = strategy;
            return render (scene, camera, settings, 16, generation);
        };
        const auto meanOver = [] (const Image& image) {
            double total = 0.0;
            for (const auto& pixel : image.pixels)
                total += meanOf (pixel);
            return total / static_cast<double> (image.pixels.size());
        };
        const auto noiseBetween = [] (const Image& a, const Image& b) {
            double total = 0.0;
            for (std::size_t i = 0; i < a.pixels.size(); ++i)
            {
                const double difference = meanOf (a.pixels[i]) - meanOf (b.pixels[i]);
                total += difference * difference;
            }
            return std::sqrt (total / static_cast<double> (a.pixels.size()));
        };
        double chanceNoise = 0.0, sampledNoise = 0.0;
        std::vector<double> difference;
        bool finite = true;
        for (std::uint64_t pair = 0; pair < 20; ++pair)
        {
            const Image chance1 = rendered (pt::LightStrategy::BsdfOnly, 100 + 2 * pair);
            const Image chance2 = rendered (pt::LightStrategy::BsdfOnly, 101 + 2 * pair);
            const Image sampled1 = rendered (pt::LightStrategy::MultipleImportance, 100 + 2 * pair);
            const Image sampled2 = rendered (pt::LightStrategy::MultipleImportance, 101 + 2 * pair);
            finite = finite && sampled1.stats.nonFiniteSamples == 0 && sampled2.stats.nonFiniteSamples == 0;
            const double chanceBetween = noiseBetween (chance1, chance2);
            const double sampledBetween = noiseBetween (sampled1, sampled2);
            chanceNoise += chanceBetween * chanceBetween / 20.0;
            sampledNoise += sampledBetween * sampledBetween / 20.0;
            difference.push_back (meanOver (sampled1) - meanOver (chance1));
            difference.push_back (meanOver (sampled2) - meanOver (chance2));
        }
        chanceNoise = std::sqrt (chanceNoise);
        sampledNoise = std::sqrt (sampledNoise);
        double shift = 0.0, spread = 0.0;
        for (const double d : difference)
            shift += d / static_cast<double> (difference.size());
        for (const double d : difference)
            spread += (d - shift) * (d - shift) / static_cast<double> (difference.size() - 1);
        const double standardError = std::sqrt (spread / static_cast<double> (difference.size()));
        const std::string name = c.name;
        std::cout << name << ": noise " << chanceNoise << " to " << sampledNoise << ", mean shift " << shift << " +- " << standardError << "\n";
        check (finite, name + " stays finite");
        check (std::abs (shift) <= 4.0 * standardError, name + " keeps the mean: shift " + describe (shift) + " against " + describe (standardError));
        if (c.cutsNoise)
            check (sampledNoise < 0.5 * chanceNoise, name + ": the light sample cuts the noise: " + describe (sampledNoise) + " from " + describe (chanceNoise));
    }
}

Image renderCoarsePrism (bool authored)
{
    pt::RenderScene scene;
    scene.materials = { diffuse ({ 0.8f, 0.8f, 0.8f }) };
    const auto radial = [] (std::uint32_t k) {
        const float angle = static_cast<float> (k) * pt::kPi * 0.25f;
        return Vec3 { std::sin (angle), 0.0f, std::cos (angle) };
    };
    const Vec3 up { 0.0f, 4.0f, 0.0f };
    for (std::uint32_t k = 0; k < 8; ++k)
    {
        const Vec3 n0 = radial (k), n1 = radial (k + 1);
        appendShadedTriangle (scene, { n0 - up, n1 - up, n1 + up }, { n0, n1, n1 }, authored);
        appendShadedTriangle (scene, { n0 - up, n1 + up, n0 + up }, { n0, n1, n0 }, authored);
    }
    const float light = 70.0f * pt::kPi / 180.0f;
    scene.lightRecords = { directionalRecord ({ std::sin (light), 0.0f, std::cos (light) }, 1.0f) };
    pt::LightShape disk;
    disk.angularRadius = 0.001f;
    scene.lightShapes = { disk };
    scene.commit();
    pt::IntegratorSettings settings;
    settings.maxDepth = 2;
    // tanHalfFov is vertical. Over the film's aspect of 24 this is 1.5 units either side
    // at distance 20, 0.03125 per pixel, so the prism spans pixels 16 to 80.
    return render (scene, makeCamera ({ 0.0f, 0.0f, 20.0f }, pt::Quat {}, 0.075f / 24.0f, 96, 4), settings, 16);
}

// Measurable form of "no faceted shadow edge": along a centre scanline, away from
// the silhouettes, no two pixels two apart differ by more than 25% of the peak
// radiance. The span keeps a step straddling a pixel in one measure. The facet-normal terminator of the same mesh jumps by cos 70 of peak,
// about 34%, and stops about ten pixels earlier.
void coarseMeshTerminatorTests()
{
    const Image authored = renderCoarsePrism (true);
    const Image facet = renderCoarsePrism (false);
    const auto measure = [] (const Image& image, float& peak, float& jump, int& lit) {
        peak = jump = 0.0f;
        lit = 0;
        for (std::uint32_t x = 20; x < 76; ++x)
        {
            const float value = image.at (x, 2).x;
            peak = std::max (peak, value);
            lit += value > 0.0f ? 1 : 0;
            if (x > 21)
                jump = std::max (jump, std::fabs (value - image.at (x - 2, 2).x));
        }
    };
    float peak = 0.0f, jump = 0.0f, facetPeak = 0.0f, facetJump = 0.0f;
    int lit = 0, facetLit = 0;
    measure (authored, peak, jump, lit);
    measure (facet, facetPeak, facetJump, facetLit);
    check (peak > 0.0f && facetPeak > 0.0f, "the coarse prism is lit");
    check (jump <= 0.25f * peak, "authored normals leave no faceted shadow edge on a coarse mesh");
    check (facetJump > 0.25f * facetPeak, "the facet-normal terminator is the step the measure detects");
    check (lit >= facetLit + 5, "the terminator follows the normals past the facet edge");
    check (authored.stats.nonFiniteSamples == 0, "the coarse prism renders finite samples");
}

// A camera ray refracts by the stored normal: only a sun on the refracted direction
// of the tilted normal is seen through the sheet, and it shows only with that normal.
// A light path has to reach what a camera path shades through a shading normal.
// At a receiver: the slab's floor again, every vertex normal tilted the same way off
// the facet. The open floor is f |toLight.ns| from a camera path, and the patch under
// the slab must stay at the slab's 0.923 of it. Reading the importance through ns gave
// (toCamera.ns / toCamera.ng) / (toLight.ns / toLight.ng), about 2.6 times too much here.
void shadingNormalReceiverSplatTests()
{
    pt::RenderScene scene;
    scene.materials = { diffuse ({ 0.5f, 0.5f, 0.5f }), clearGlass (1.5f) };
    const Vec3 tilted = pt::normalizeOrZero ({ 0.0f, 1.0f, 0.5f });
    const Vec3 a { -2.0f, 0.0f, -2.0f }, b { -2.0f, 0.0f, 2.0f }, c { 2.0f, 0.0f, 2.0f }, d { 2.0f, 0.0f, -2.0f };
    appendShadedTriangle (scene, { a, b, c }, { tilted, tilted, tilted }, false);
    appendShadedTriangle (scene, { a, c, d }, { tilted, tilted, tilted }, false);
    const Vec3 centre { 0.0f, 1.1f, 0.0f };
    appendMesh (scene, box ({ -0.5f, 1.0f, -0.5f }, { 0.5f, 1.2f, 0.5f }), 1, &centre);
    scene.lightRecords = { directionalRecord ({ 0.0f, 1.0f, 0.0f }, 1.0f) };
    scene.commit();
    const pt::PinholeCamera camera = lookAt ({ 0.0f, 0.8f, 2.5f }, { 0.0f, 0.0f, 0.0f }, 0.4f, 48, 32);
    pt::IntegratorSettings settings;
    settings.maxDepth = 8;
    settings.spectral = pt::SpectralMode::Off;
    const Image image = render (scene, camera, settings, 64, 11);
    int count = 0;
    const double under = floorMean (scene, camera, image.pixels, 2, -0.35f, 0.35f, -0.35f, 0.35f, count);
    const double lit = floorMean (scene, camera, image.pixels, 2, 0.8f, 1.6f, -0.35f, 0.35f, count);
    check (closeRelative (lit, 0.5 / pt::kPi * tilted.y, 0.03), "a tilted normal shades the open floor: " + describe (lit));
    check (under > 0.84 * lit && under < 1.0 * lit,
           "a tilted receiver takes the slab's light as a camera path would: " + describe (under / lit) + " of the open floor");
}

// At a delta vertex: a glass floor under a sun straight overhead, flat, with a vertex normal
// tilted 25 degrees, mirrors the sun onto a wall the sun itself runs parallel to. A camera
// path from the wall reads Fresnel * sun off this glass at the 50 degree mirror direction, so
// the wall shows f * F * |out.wall|. Without the factor |out.ng| / |toLight.ng| the wall is
// 1 / 0.643 too bright.
void shadingNormalDeltaSplatTests()
{
    pt::RenderScene scene;
    scene.materials = { diffuse ({ 0.5f, 0.5f, 0.5f }), clearGlass (1.5f) };
    appendMesh (scene, quad ({ 0.0f, 1.7f, 2.0f }, { 0.0f, 2.0f, 0.0f }, { 3.0f, 0.0f, 0.0f }), 0);
    const float tilt = 25.0f * pt::kPi / 180.0f;
    const Vec3 tilted { 0.0f, std::cos (tilt), std::sin (tilt) };
    const Vec3 a { -2.0f, 0.0f, -1.0f }, b { -2.0f, 0.0f, 1.0f }, c { 2.0f, 0.0f, 1.0f }, d { 2.0f, 0.0f, -1.0f };
    appendShadedTriangle (scene, { a, b, c }, { tilted, tilted, tilted }, false, 1);
    appendShadedTriangle (scene, { a, c, d }, { tilted, tilted, tilted }, false, 1);
    scene.lightRecords = { directionalRecord ({ 0.0f, 1.0f, 0.0f }, 1.0f) };
    scene.commit();
    const pt::PinholeCamera camera = lookAt ({ 0.0f, 1.7f, -1.5f }, { 0.0f, 1.7f, 2.0f }, 0.5f, 48, 32);
    pt::IntegratorSettings settings;
    settings.maxDepth = 4;
    settings.spectral = pt::SpectralMode::Off;
    const Image image = render (scene, camera, settings, 128, 11);
    // The mirrored band lands on the wall from y 0.84 to 2.52.
    double sum = 0.0;
    int count = 0;
    for (std::uint32_t y = 0; y < camera.height; ++y)
        for (std::uint32_t x = 0; x < camera.width; ++x)
        {
            Vec3 p;
            if (floorPoint (scene, camera, x, y, 2, p) && p.y > 1.2f && p.y < 2.2f && std::fabs (p.x) < 1.0f)
            {
                sum += meanOf (image.pixels[static_cast<std::size_t> (y) * camera.width + x]);
                ++count;
            }
        }
    const double expected = 0.5 / pt::kPi * pt::fresnelDielectric (std::cos (tilt), 1.5f) * std::sin (2.0f * tilt);
    check (count >= 20, "the camera sees the mirrored band: " + std::to_string (count) + " pixels");
    check (closeRelative (sum / std::max (1, count), expected, 0.2),
           "a delta vertex with a tilted normal splats what a camera path reads: " + describe (sum / std::max (1, count))
               + " against " + describe (expected));
}

void authoredRefractionTests()
{
    const float tilt = 20.0f * pt::kPi / 180.0f;
    const Vec3 tilted { std::sin (tilt), 0.0f, std::cos (tilt) };
    const Vec3 facing { 0.0f, 0.0f, 1.0f };
    Vec3 sun;
    check (refractDirection ({ 0.0f, 0.0f, -1.0f }, tilted, 1.5, sun), "the tilted normal refracts");
    const auto seen = [&sun] (Vec3 normal, bool authored) {
        pt::RenderScene scene;
        scene.materials = { clearGlass (1.5f) };
        const Vec3 a { -2.0f, -2.0f, 0.0f }, b { 2.0f, -2.0f, 0.0f }, c { 2.0f, 2.0f, 0.0f }, d { -2.0f, 2.0f, 0.0f };
        appendShadedTriangle (scene, { a, b, c }, { normal, normal, normal }, authored);
        appendShadedTriangle (scene, { a, c, d }, { normal, normal, normal }, authored);
        scene.lightRecords = { directionalRecord (sun, 1.0f) };
        pt::LightShape disk;
        disk.angularRadius = 0.03f;
        scene.lightShapes = { disk };
        scene.commit();
        pt::IntegratorSettings settings;
        settings.maxDepth = 4;
        const Image image = render (scene, makeCamera ({ 0.0f, 0.0f, 5.0f }, pt::Quat {}, 0.02f, 8, 8), settings, 32);
        return image.at (4, 4).x;
    };
    check (seen (tilted, true) > 0.0f, "refraction follows the authored normal");
    check (seen (facing, true) == 0.0f, "a facing normal does not bend the ray to the sun");
}

// ---------------------------------------------------------------- ring fringes

// Normal of a tube side at angle v around the tube. Flat keeps the side's own
// normal, the Smooth by Angle split the Glass Ring starter had at 7.5 degrees.
float tubeShadingAngle (float v, float sideAngle, bool flat, std::uint32_t side)
{
    return flat ? (static_cast<float> (side) + 0.5f) * sideAngle : v;
}

// The Glass Ring starter's 96 x 24 tube about the y axis, major 0.5, minor 0.1.
// Normals blend along the hoop. Across the tube they blend or stay flat per side.
void appendShadedRing (pt::RenderScene& scene, std::uint32_t material, bool flatSides)
{
    constexpr std::uint32_t segments = 96, sides = 24;
    constexpr float major = 0.5f, minor = 0.1f;
    const float segmentAngle = 2.0f * pt::kPi / static_cast<float> (segments);
    const float sideAngle = 2.0f * pt::kPi / static_cast<float> (sides);
    const auto point = [&] (float u, float v) {
        const float r = major + minor * std::cos (v);
        return Vec3 { r * std::cos (u), minor * std::sin (v), r * std::sin (u) };
    };
    const auto normal = [&] (float u, float v) {
        return Vec3 { std::cos (v) * std::cos (u), std::sin (v), std::cos (v) * std::sin (u) };
    };
    for (std::uint32_t i = 0; i < segments; ++i)
        for (std::uint32_t j = 0; j < sides; ++j)
        {
            const float u0 = static_cast<float> (i) * segmentAngle, u1 = u0 + segmentAngle;
            const float v0 = static_cast<float> (j) * sideAngle, v1 = v0 + sideAngle;
            const float n0 = tubeShadingAngle (v0, sideAngle, flatSides, j);
            const float n1 = tubeShadingAngle (v1, sideAngle, flatSides, j);
            appendShadedTriangle (scene, { point (u0, v0), point (u0, v1), point (u1, v0) },
                                  { normal (u0, n0), normal (u0, n1), normal (u1, n0) }, true, material);
            appendShadedTriangle (scene, { point (u0, v1), point (u1, v1), point (u1, v0) },
                                  { normal (u0, n1), normal (u1, n1), normal (u1, n0) }, true, material);
        }
}

// A straight 24 sided tube of radius 0.1 along x through the origin.
void appendShadedTube (pt::RenderScene& scene, std::uint32_t material, float halfLength, bool flatSides)
{
    constexpr std::uint32_t sides = 24;
    const float sideAngle = 2.0f * pt::kPi / static_cast<float> (sides);
    const auto normal = [] (float v) { return Vec3 { 0.0f, std::cos (v), std::sin (v) }; };
    const Vec3 end { halfLength, 0.0f, 0.0f };
    for (std::uint32_t j = 0; j < sides; ++j)
    {
        const float v0 = static_cast<float> (j) * sideAngle, v1 = v0 + sideAngle;
        const float n0 = tubeShadingAngle (v0, sideAngle, flatSides, j);
        const float n1 = tubeShadingAngle (v1, sideAngle, flatSides, j);
        const Vec3 p0 = normal (v0) * 0.1f, p1 = normal (v1) * 0.1f;
        appendShadedTriangle (scene, { p0 - end, p1 - end, p0 + end }, { normal (n0), normal (n1), normal (n0) }, true, material);
        appendShadedTriangle (scene, { p0 + end, p1 - end, p1 + end }, { normal (n0), normal (n1), normal (n1) }, true, material);
    }
}

// Width in pixels of the widest coherent spectral band on the ring's far arc,
// seen at the given film scale. 346 pixels per unit is the old starter's 1080p
// frame at fov 52 and distance 3.2. The only light is a bright disc 1 below
// the ring, seen only through the glass, so the tube images the disc's round edge
// as a band that follows the hoop.
//
// The metric. Pixels fall into one pixel wide rings around the hoop's centre and
// 12 sectors along the arc. A band is a ring where at least 10 consecutive sectors
// have (red - blue) / (red + green + blue) of one sign and above minimumChroma, with that
// ratio taken over the sector's summed light and only where the mean pixel sum is
// over 0.15, 5% of the disc's white. An isolated chromatic pixel cannot
// pass: it moves one sector's sum by little, and speckle has no shared sign. This
// is the reference's structure: saturated bands that run on along the rim.
int coherentRingBandWidth (float cauchyB, std::uint64_t generation, float pixelsPerUnit = 346.0f,
                           double minimumChroma = 0.2)
{
    constexpr float cameraHeight = 3.2f;
    constexpr std::uint32_t width = 220, height = 120;
    constexpr int sectors = 12, rings = 80;
    pt::RenderScene scene;
    scene.materials = { clearGlass (1.5f, cauchyB), lamp (1.0f) };
    appendShadedRing (scene, 0, false);
    Mesh disc;
    disc.vertices.push_back ({ 0.0f, -1.0f, 0.0f });
    constexpr std::uint32_t discEdges = 256;
    for (std::uint32_t k = 0; k < discEdges; ++k)
    {
        const float a = 2.0f * pt::kPi * static_cast<float> (k) / static_cast<float> (discEdges);
        disc.vertices.push_back ({ 0.2f * std::cos (a), -1.0f, 0.2f * std::sin (a) });
    }
    for (std::uint32_t k = 0; k < discEdges; ++k)
        disc.indices.insert (disc.indices.end(), { 0u, 1 + (k + 1) % discEdges, 1 + k });
    appendMesh (scene, disc, 1);
    scene.commit();

    const float tanHalfFov = 0.5f * static_cast<float> (height) / (pixelsPerUnit * cameraHeight);
    const pt::PinholeCamera camera = makeCamera ({ 0.0f, cameraHeight, -0.5f }, kDown, tanHalfFov, width, height);
    pt::IntegratorSettings settings;
    settings.maxDepth = 8;
    settings.causticLightPaths = false;
    const Image image = render (scene, camera, settings, 32, generation);
    check (image.stats.spectral && image.stats.nonFiniteSamples == 0, "the ring fringe fixture renders spectrally and finite");

    constexpr float arc = 24.0f * pt::kPi / 180.0f;
    std::vector<std::array<double, 3>> sums (static_cast<std::size_t> (sectors * rings), { 0.0, 0.0, 0.0 });
    std::vector<int> counts (sums.size(), 0);
    for (std::uint32_t y = 0; y < height; ++y)
        for (std::uint32_t x = 0; x < width; ++x)
        {
            // Image up is -z, away from the hoop's centre here.
            const float worldX = (static_cast<float> (x) + 0.5f - 0.5f * width) / pixelsPerUnit;
            const float worldZ = -0.5f - (0.5f * height - static_cast<float> (y) - 0.5f) / pixelsPerUnit;
            const float angle = std::atan2 (worldX, -worldZ);
            const float radius = std::sqrt (worldX * worldX + worldZ * worldZ);
            const int ring = static_cast<int> (std::floor ((radius - 0.4f) * pixelsPerUnit));
            const int sector = static_cast<int> (std::floor ((angle + arc) / (2.0f * arc) * sectors));
            if (ring < 0 || ring >= rings || sector < 0 || sector >= sectors)
                continue;
            const Vec3 p = image.at (x, y);
            const auto cell = static_cast<std::size_t> (ring * sectors + sector);
            ++counts[cell];
            auto& sum = sums[cell];
            sum[0] += p.x;
            sum[1] += p.y;
            sum[2] += p.z;
        }

    int band = 0;
    for (int ring = 0; ring < rings; ++ring)
    {
        int longest = 0, run = 0, runSign = 0;
        for (int sector = 0; sector < sectors; ++sector)
        {
            const auto cell = static_cast<std::size_t> (ring * sectors + sector);
            const auto& sum = sums[cell];
            const double total = sum[0] + sum[1] + sum[2];
            const double chroma = total > 0.15 * counts[cell] && total > 0.0 ? (sum[0] - sum[2]) / total : 0.0;
            const int sign = chroma > minimumChroma ? 1 : chroma < -minimumChroma ? -1 : 0;
            run = sign != 0 && sign == runSign ? run + 1 : (sign != 0 ? 1 : 0);
            runSign = sign;
            longest = std::max (longest, run);
        }
        band += longest >= 10 ? 1 : 0;
    }
    return band;
}

// A cross-section model of this fixture with the CIE tables gives 3 rings above
// 0.2 at B = 0.02 and none at 0.004, whose peak ratio is about 0.12. That model
// leaves out the hoop's curve. The camera sits over the far arc, so a sector at 24
// degrees sees the tube's top 0.06 shifted by 0.2 * 0.06 / 3.2, 1.3 pixels, against
// the arc's centre, and 10 sectors must share a ring. So B = 0.02, the starter's value
// until it became 0.15, holds a band about a pixel wide. ringStarterFrameBandTests
// covers the wide band at 0.15.
void ringFringeCoherenceTests()
{
    const int starter = coherentRingBandWidth (0.02f, 31);
    const int previous = coherentRingBandWidth (0.004f, 31);
    check (starter >= 1, "B = 0.02 forms a coherent band " + std::to_string (starter) + " pixels wide");
    check (previous <= 1, "the old Cauchy B forms no band over a pixel wide: " + std::to_string (previous));
}

// The fixture at the starter's own film scales on the 804 x 452 live film, the
// frame that stayed grey: 143 pixels per unit at the old camera distance of 3.2,
// 241 at the authored 1.9. Saturated means a ratio above 0.5. The cross-section
// model with the CIE tables gives 0 such pixels for B = 0.02 at 143, the frame the
// old test passed beside, and 6 for B = 0.15 at 241. The wide margin below is for
// the hoop's curve, which that model leaves out.
void ringStarterFrameBandTests()
{
    const int oldFrame = coherentRingBandWidth (0.02f, 43, 143.0f, 0.5);
    const int newFrame = coherentRingBandWidth (0.15f, 43, 241.0f, 0.5);
    check (oldFrame <= 1, "the old starter's frame has no saturated band: " + std::to_string (oldFrame));
    check (newFrame >= 3, "the authored starter's frame has a saturated band " + std::to_string (newFrame) + " pixels wide");
}

// Dispersion above any old cap reaches the walk at the authored strength. The
// path tracer capped B at 0.05 and n at 3 after the admission stopped doing so.
void strongDispersionReachesEvalTests()
{
    const pt::Material strong = clearGlass (1.5f, 0.15f);
    check (pt::materialCauchyB (strong) == 0.15f, "B = 0.15 renders as authored: " + describe (pt::materialCauchyB (strong)));
    check (std::fabs (pt::materialIorAt (strong, 400.0f, true) - 2.003f) < 1.0e-3f,
           "B = 0.15 gives n = 2.003 at 400 nm: " + describe (pt::materialIorAt (strong, 400.0f, true)));
    const pt::Material extreme = clearGlass (1.5f, 0.6f);
    const float n400 = pt::materialIorAt (extreme, 400.0f, true);
    check (std::fabs (n400 - 3.512f) < 1.0e-2f, "n has no upper cap: " + describe (n400));

    // The eta of the lobe the walk refracts with: a ray at 45 degrees into n = 3.512
    // bends to asin (sin 45 / 3.512) = 11.6 degrees. The old eta cap of 3 gave 13.6.
    pt::Wavelengths wl = pt::sampleHeroWavelengths ((400.0f - pt::kWavelengthMinNm) / pt::kWavelengthSpanNm);
    pt::MediumStack media;
    const pt::BoundaryCrossing in = pt::classifyBoundary (media, &extreme, true);
    const pt::DielectricSetup entry = pt::dielectricSetupFor (in, wl);
    pt::SpectralBsdf bsdf;
    const Vec3 normal { 0.0f, 0.0f, 1.0f };
    const Vec3 incoming { std::sqrt (0.5f), 0.0f, std::sqrt (0.5f) };
    pt::buildMaterialBsdf (extreme, normal, normal, incoming, wl, entry, bsdf);
    pt::SpectralBsdfSample through;
    const bool refracted = bsdf.sample (incoming, 0.9f, { 0.5f, 0.5f }, through);
    check (refracted && (through.flags & pt::kBsdfTransmission) != 0, "the strong glass refracts");
    const float degrees = std::asin (std::sqrt (through.wi.x * through.wi.x + through.wi.y * through.wi.y)) * 180.0f / pt::kPi;
    check (std::fabs (degrees - 11.6f) < 0.3f, "the walk refracts with the uncapped eta: " + describe (degrees));
}

// Blue's and red's RMS spread across a straight tube's caustic on a floor 0.17
// below its axis, the Glass Ring starter's lowest point. The summed spread is
// immune to the per-pixel colour noise of a light path's four wavelengths, as long
// as the sum stays signed: splats are signed, and dropping a pixel's negative part
// would keep that noise.
double causticBlueOverRedSpread (float cauchyB, std::uint64_t generation)
{
    pt::RenderScene scene;
    scene.materials = { diffuse ({ 0.8f, 0.8f, 0.8f }), clearGlass (1.5f, cauchyB) };
    appendMesh (scene, quad ({ 0.0f, -0.17f, 0.0f }, { 0.75f, 0.0f, 0.0f }, { 0.0f, 0.0f, -0.75f }), 0);
    appendShadedTube (scene, 1, 0.75f, false);
    scene.lightRecords = { directionalRecord ({ 0.0f, 1.0f, 0.0f }, 1.0f) };
    scene.commit();
    const pt::PinholeCamera camera = lookAt ({ 0.0f, 0.15f, 2.0f }, { 0.0f, -0.17f, 0.0f }, 0.04f, 128, 96);
    pt::IntegratorSettings settings;
    settings.maxDepth = 8;
    const Image image = render (scene, camera, settings, 128, generation);
    check (image.stats.spectral && image.stats.causticSplats > 0, "the tube's caustic is spectral and reaches the camera");
    double red = 0.0, redZ2 = 0.0, blue = 0.0, blueZ2 = 0.0;
    for (std::uint32_t y = 0; y < camera.height; ++y)
        for (std::uint32_t x = 0; x < camera.width; ++x)
        {
            Vec3 p;
            if (! floorPoint (scene, camera, x, y, 2, p) || std::fabs (p.z) > 0.35f)
                continue;
            const Vec3 c = image.caustic[static_cast<std::size_t> (y) * camera.width + x];
            const double z2 = static_cast<double> (p.z) * p.z;
            red += c.x;
            redZ2 += static_cast<double> (c.x) * z2;
            blue += c.z;
            blueZ2 += static_cast<double> (c.z) * z2;
        }
    check (red > 0.0 && blue > 0.0, "the tube's caustic lands on the floor in red and blue");
    return red > 0.0 && blue > 0.0 ? std::sqrt ((blueZ2 / blue) / (redZ2 / red)) : 0.0;
}

// Signed (red - blue) / sum of the tube's caustic over floor bands 0.02 wide,
// out to 0.1 from its line. Index 0 is the core. Only splats count, so this is
// the colour light paths deposit. Each cell sums many splats of all wavelengths,
// so a neutral deposit would stay near 0 whatever the per-pixel colour noise.
std::vector<double> causticBandChroma (float cauchyB, std::uint64_t generation)
{
    pt::RenderScene scene;
    scene.materials = { diffuse ({ 0.8f, 0.8f, 0.8f }), clearGlass (1.5f, cauchyB) };
    appendMesh (scene, quad ({ 0.0f, -0.17f, 0.0f }, { 0.75f, 0.0f, 0.0f }, { 0.0f, 0.0f, -0.75f }), 0);
    appendShadedTube (scene, 1, 0.75f, false);
    scene.lightRecords = { directionalRecord ({ 0.0f, 1.0f, 0.0f }, 1.0f) };
    scene.commit();
    const pt::PinholeCamera camera = lookAt ({ 0.0f, 0.15f, 2.0f }, { 0.0f, -0.17f, 0.0f }, 0.04f, 128, 96);
    pt::IntegratorSettings settings;
    settings.maxDepth = 8;
    const Image image = render (scene, camera, settings, 128, generation);
    std::vector<std::array<double, 3>> bands (5, { 0.0, 0.0, 0.0 });
    for (std::uint32_t y = 0; y < camera.height; ++y)
        for (std::uint32_t x = 0; x < camera.width; ++x)
        {
            Vec3 p;
            if (! floorPoint (scene, camera, x, y, 2, p))
                continue;
            const auto band = static_cast<std::size_t> (std::fabs (p.z) / 0.02f);
            if (band >= bands.size())
                continue;
            const Vec3 c = image.caustic[static_cast<std::size_t> (y) * camera.width + x];
            bands[band][0] += c.x;
            bands[band][1] += c.y;
            bands[band][2] += c.z;
        }
    std::vector<double> chroma;
    for (const auto& band : bands)
    {
        const double total = band[0] + band[1] + band[2];
        check (total > 0.0, "the tube's caustic reaches every floor band");
        chroma.push_back (total > 0.0 ? (band[0] - band[2]) / total : 0.0);
    }
    return chroma;
}

// Red focuses on this floor and blue crosses over before it, so a coloured
// deposit has a red core and a blue halo. The cross-section model with the CIE
// tables gives +0.25 core and -0.20 to -0.31 halo at B = 0.15, and +0.04 and -0.05
// at 0.02, the white caustic of the frame.
void causticSplatColourTests()
{
    const std::vector<double> strong = causticBandChroma (0.15f, 47);
    const std::vector<double> weak = causticBandChroma (0.02f, 47);
    check (strong[0] > 0.12, "B = 0.15 deposits a red caustic core: " + describe (strong[0]));
    for (std::size_t band = 1; band < strong.size(); ++band)
        check (strong[band] < -0.1, "B = 0.15 deposits a blue caustic halo: " + describe (strong[band]));
    check (std::fabs (weak[0]) < 0.12, "B = 0.02 leaves the caustic core near white: " + describe (weak[0]));
}

// Blue focuses nearer than red, so past the focus it spreads wider. The same
// cross-section model gives 1.09 at B = 0.02 and 1.04 at 0.004, summed with the
// signed colour weights over this floor window.
void ringCausticColourTests()
{
    const double starter = causticBlueOverRedSpread (0.02f, 37);
    const double previous = causticBlueOverRedSpread (0.004f, 37);
    check (starter > 1.06, "the starter's caustic separates blue from red by a spread ratio of " + describe (starter));
    check (previous < 1.06, "the old Cauchy B leaves the caustic near white, a spread ratio of " + describe (previous));
}

// The Glass Ring starter's tube as it stands: the circle in the XY plane, centre
// at y 0.6. appendShadedRing's mesh turned 90 degrees about x, so winding holds.
void appendStandingRing (pt::RenderScene& scene, std::uint32_t material)
{
    constexpr std::uint32_t segments = 96, sides = 24;
    constexpr float major = 0.5f, minor = 0.1f, centreY = 0.6f;
    const float segmentAngle = 2.0f * pt::kPi / static_cast<float> (segments);
    const float sideAngle = 2.0f * pt::kPi / static_cast<float> (sides);
    const auto normal = [] (float u, float v) {
        return Vec3 { std::cos (v) * std::cos (u), std::cos (v) * std::sin (u), -std::sin (v) };
    };
    const auto point = [&] (float u, float v) {
        const float r = major + minor * std::cos (v);
        return Vec3 { r * std::cos (u), centreY + r * std::sin (u), -minor * std::sin (v) };
    };
    for (std::uint32_t i = 0; i < segments; ++i)
        for (std::uint32_t j = 0; j < sides; ++j)
        {
            const float u0 = static_cast<float> (i) * segmentAngle, u1 = u0 + segmentAngle;
            const float v0 = static_cast<float> (j) * sideAngle, v1 = v0 + sideAngle;
            appendShadedTriangle (scene, { point (u0, v0), point (u0, v1), point (u1, v0) },
                                  { normal (u0, v0), normal (u0, v1), normal (u1, v0) }, true, material);
            appendShadedTriangle (scene, { point (u0, v1), point (u1, v1), point (u1, v0) },
                                  { normal (u0, v1), normal (u1, v1), normal (u1, v0) }, true, material);
        }
}

// Authored light of the Glass Ring starter.
struct RingFloorLight
{
    Vec3 toLight;
    float keyIntensity = 0.0f;
    float floorAlbedo = 0.0f;
    float skyRadiance = 0.0f;
};

struct RingFloorResult
{
    double diffuseFace = 0.0;
    double predictedFace = 0.0;
    double renderedFace = 0.0;
    double causticOverFace = 0.0;
    double causticChroma = 0.0;
    double brightestStripChannel = 0.0;
};

double rec709Luminance (Vec3 c)
{
    return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
}

// Mirrors the starter at full scale: ring and B 0.15 glass, floor plate, key colour,
// camera position and fov, hidden sky. The one stand-in is a constant sky: art_studio's
// cosine-weighted upper-hemisphere luminance of 0.86, times 0.05. The starter now binds a
// brighter colour HDR (0.605 * 0.5 = 0.30), which this model doesn't follow. The old light
// keeps the bare 0.86.
// GraphInstrumentTests pins the same authored values on the starter itself.
RingFloorResult renderRingFloor (const RingFloorLight& light, std::uint64_t generation)
{
    const float skyRadiance = light.skyRadiance;
    const Vec3 keyColour { 1.0f, 0.9f, 0.75f };
    const Vec3 cameraPosition { 0.0f, 0.95f, 1.9f };
    pt::RenderScene scene;
    scene.materials = { diffuse ({ light.floorAlbedo, light.floorAlbedo, light.floorAlbedo }), clearGlass (1.5f, 0.15f) };
    appendMesh (scene, quad ({ 0.0f, 0.0f, 0.0f }, { 1.7f, 0.0f, 0.0f }, { 0.0f, 0.0f, -1.3f }), 0);
    appendStandingRing (scene, 1);
    grid::SceneLightRecord key = directionalRecord (light.toLight, light.keyIntensity);
    key.color = { keyColour.x, keyColour.y, keyColour.z };
    scene.lightRecords = { key, environmentRecord (skyRadiance) };
    scene.commit();
    pt::IntegratorSettings settings;
    settings.maxDepth = 8;
    settings.environmentVisibleToCamera = false;

    // Face from open floor beyond the ring's shadow, in the starter's own framing.
    const pt::PinholeCamera wide = lookAt (cameraPosition, { 0.0f, 0.67f, 0.0f }, std::tan (26.0f * pt::kPi / 180.0f), 96, 54);
    const Image wideImage = render (scene, wide, settings, 32, generation);
    check (wideImage.stats.nonFiniteSamples == 0, "the ring floor stays finite");
    std::vector<Vec3> wideLuminance;
    for (const Vec3 c : wideImage.pixels)
    {
        const auto y = static_cast<float> (rec709Luminance (c));
        wideLuminance.push_back ({ y, y, y });
    }
    int openPixels = 0;
    RingFloorResult result;
    result.renderedFace = floorMean (scene, wide, wideLuminance, 2, 0.9f, 1.4f, 0.0f, 0.4f, openPixels);
    check (openPixels >= 20, "the open floor is in frame, pixels " + describe (static_cast<double> (openPixels)));
    const Vec3 toLight = pt::normalizeOrZero (light.toLight);
    result.diffuseFace = light.floorAlbedo * (light.keyIntensity * toLight.y * rec709Luminance (keyColour) / pt::kPi + skyRadiance);
    // The floor is an opaque dielectric, so it also has a specular lobe with F0 0.04 at
    // roughness 1, whatever its albedo. Over the open pixels, through the renderer's own
    // BSDF: the key's value, plus the sky over the lobes' directional albedo.
    double predicted = 0.0;
    int predictedPixels = 0;
    for (std::uint32_t y = 0; y < wide.height; ++y)
        for (std::uint32_t x = 0; x < wide.width; ++x)
        {
            Vec3 p;
            if (! floorPoint (scene, wide, x, y, 2, p) || ! (p.x > 0.9f && p.x < 1.4f && p.z > 0.0f && p.z < 0.4f))
                continue;
            const Vec3 wo = pt::normalizeOrZero (wide.position - p);
            const Vec3 up { 0.0f, 1.0f, 0.0f };
            pt::Bsdf floorBsdf;
            pt::buildMaterialBsdf (scene.materials[0], up, up, wo, true, floorBsdf);
            float scale = 0.0f, bias = 0.0f;
            pt::ggxAlbedoScaleBias (wo.y, 1.0f, scale, bias);
            const float specular = 0.04f * scale + bias;
            const double directional = light.floorAlbedo * (1.0f - specular) + specular;
            predicted += light.keyIntensity * rec709Luminance (keyColour) * meanOf (floorBsdf.eval (wo, toLight).value)
                       + skyRadiance * directional;
            ++predictedPixels;
        }
    result.predictedFace = predicted / std::max (1, predictedPixels);

    // Same camera position with a narrow lens, so the same sight lines resolve the
    // strip. Floor strips 0.02 deep, 0.16 wide, under the ring's foot.
    constexpr float nearZ = -0.15f, stripDepth = 0.02f;
    constexpr std::size_t strips = 22;
    const pt::PinholeCamera narrow = lookAt (cameraPosition, { 0.0f, 0.0f, 0.05f }, 0.07f, 256, 128);
    const Image image = render (scene, narrow, settings, 96, generation + 1);
    std::array<std::array<double, 6>, strips> sums {};
    std::array<int, strips> counts {};
    for (std::uint32_t y = 0; y < narrow.height; ++y)
        for (std::uint32_t x = 0; x < narrow.width; ++x)
        {
            Vec3 p;
            if (! floorPoint (scene, narrow, x, y, 2, p) || std::fabs (p.x) >= 0.08f || p.z < nearZ)
                continue;
            const auto strip = static_cast<std::size_t> ((p.z - nearZ) / stripDepth);
            if (strip >= strips)
                continue;
            const std::size_t index = static_cast<std::size_t> (y) * narrow.width + x;
            sums[strip][0] += rec709Luminance (image.pixels[index]);
            sums[strip][1] += image.caustic[index].x;
            sums[strip][2] += image.caustic[index].z;
            sums[strip][3] += image.pixels[index].x;
            sums[strip][4] += image.pixels[index].y;
            sums[strip][5] += image.pixels[index].z;
            ++counts[strip];
        }
    double brightest = 0.0;
    for (std::size_t strip = 0; strip < strips; ++strip)
    {
        if (counts[strip] < 40 || sums[strip][0] / counts[strip] <= brightest)
            continue;
        brightest = sums[strip][0] / counts[strip];
        result.causticChroma = (sums[strip][1] - sums[strip][2]) / std::max (sums[strip][1] + sums[strip][2], 1.0e-9);
        // The largest channel of the strip's mean. A mean of each pixel's largest channel
        // would count the colour noise of four wavelengths as brightness.
        result.brightestStripChannel = std::max ({ sums[strip][3], sums[strip][4], sums[strip][5] }) / counts[strip];
    }
    result.causticOverFace = brightest / std::max (result.renderedFace, 1.0e-9);
    return result;
}

// The r2-acceptance floor was a uniform dark grey with no caustic in it. Arithmetic:
// old face 0.03 * (3.5 * 0.574 * 0.911 / pi + 0.86) = 0.043, sRGB 59. New
// 0.03 * (10 * 0.866 * 0.911 / pi + 0.043) = 0.077. Both count the albedo only; the
// floor's specular lobe takes the rendered faces to 0.075 and 0.117. The Monte Carlo ring
// model gave the strip 1.07x the face at the old light and 2.31x with chroma +0.43 at the
// -102 key, with the ring 0.07 clear of the floor. With the ring at contact the same kind
// of model gives 1.63x at the old light and 6.5x at the -120 key, so the old light does
// leave a strip now, a quarter of the new one's. Its chroma threshold is still unmeasured.
void ringCausticFloorTests()
{
    const RingFloorResult before = renderRingFloor ({ { -0.346f, 0.574f, 0.742f }, 3.5f, 0.03f, 0.86f }, 53);
    const RingFloorResult after = renderRingFloor ({ { 0.0f, 0.866f, -0.5f }, 10.0f, 0.03f, 0.043f }, 55);
    check (before.diffuseFace < 0.09, "the old light underexposes the floor, face " + describe (before.diffuseFace));
    check (after.diffuseFace > 0.06 && after.diffuseFace < 0.09,
           "the floor's diffuse face lands in 0.06 to 0.09 linear: " + describe (after.diffuseFace));
    check (std::fabs (after.renderedFace - after.predictedFace) < 0.15 * after.predictedFace,
           "the rendered face " + describe (after.renderedFace) + " matches the prediction " + describe (after.predictedFace));
    check (after.brightestStripChannel < 1.0, "the caustic stays under display white: " + describe (after.brightestStripChannel));
    check (before.causticOverFace < 2.0, "the old light leaves only a faint caustic, ratio " + describe (before.causticOverFace));
    check (after.causticOverFace > 1.6, "the caustic strip stands above the floor, ratio " + describe (after.causticOverFace));
    check (after.causticChroma > 0.12, "the caustic strip keeps its red core: " + describe (after.causticChroma));
}

// A sine across 32 rows of a coarse environment, seen through a tube. Rows are
// 5.6 degrees, so a nearest lookup or a flat tube side would show as steps.
pt::EnvironmentPixels bandedEnvironment()
{
    constexpr std::uint32_t width = 16, height = 32;
    std::vector<float> rgb;
    for (std::uint32_t y = 0; y < height; ++y)
        for (std::uint32_t x = 0; x < width; ++x)
        {
            const float value = 0.5f + 0.4f * std::sin (2.0f * pt::kPi * static_cast<float> (y) / 16.0f);
            rgb.insert (rgb.end(), { value, value, value });
        }
    return pt::copyEnvironmentPixels (width, height, rgb.data(), { "ring-refraction-test", 1, std::string (64, 'b') });
}

// Largest second difference of the tube's mean brightness down the image,
// away from its rims, over the brightness range seen there. Each row averages
// many columns, so what is left is the profile across the tube.
double refractedBlockiness (bool flatSides, std::uint64_t generation)
{
    pt::RenderScene scene;
    scene.materials = { clearGlass (1.5f) };
    appendShadedTube (scene, 0, 3.0f, flatSides);
    grid::SceneLightRecord light;
    light.kind = grid::SceneLightKind::Environment;
    light.color = { 1.0f, 1.0f, 1.0f };
    light.intensity = 1.0f;
    light.environmentImage.version = 1;
    scene.environmentImage = bandedEnvironment();
    light.environmentImage.asset = scene.environmentImage.asset;
    scene.lightRecords = { light };
    scene.commit();
    constexpr std::uint32_t width = 96, height = 56;
    constexpr float pixelsPerUnit = 346.0f, distance = 3.2f;
    const float tanHalfFov = 0.5f * static_cast<float> (height) / (pixelsPerUnit * distance);
    const pt::PinholeCamera camera = makeCamera ({ 0.0f, 0.0f, distance }, pt::Quat {}, tanHalfFov, width, height);
    pt::IntegratorSettings settings;
    settings.maxDepth = 8;
    settings.spectral = pt::SpectralMode::Off;
    settings.causticLightPaths = false;
    const Image image = render (scene, camera, settings, 32, generation);
    std::vector<double> profile;
    for (std::uint32_t y = 8; y < height - 8; ++y)
    {
        double sum = 0.0;
        for (std::uint32_t x = 0; x < width; ++x)
            sum += image.at (x, y).y;
        profile.push_back (sum / width);
    }
    const auto [low, high] = std::minmax_element (profile.begin(), profile.end());
    double worst = 0.0;
    for (std::size_t i = 1; i + 1 < profile.size(); ++i)
        worst = std::max (worst, std::fabs (profile[i + 1] - 2.0 * profile[i] + profile[i - 1]));
    return worst / std::max (*high - *low, 1.0e-9);
}

// The lookup itself is bilinear: across one texel of a stripe image at any
// non-integer offset the radiance moves in equal steps, with no jump a nearest
// lookup or a coarse block would make. Then the rendered form: through a
// smooth tube the banded environment stays smooth, and the flat sided tube of
// the same mesh shows the steps that were the starter's rectangular patches.
void refractionSmoothnessTests()
{
    constexpr std::uint32_t width = 16, height = 8;
    std::vector<float> stripes;
    for (std::uint32_t y = 0; y < height; ++y)
        for (std::uint32_t x = 0; x < width; ++x)
        {
            const float value = x % 2 == 0 ? 0.0f : 1.0f;
            stripes.insert (stripes.end(), { value, value, value });
        }
    pt::EnvironmentMap map;
    check (map.build (pt::copyEnvironmentPixels (width, height, stripes.data(), { "ring-stripes-test", 1, std::string (64, 'c') }),
                      { 1.0f, 1.0f, 1.0f }, 0.0f),
           "the stripe environment builds");
    double largestStep = 0.0;
    constexpr int stepsPerTexel = 7;
    float previous = map.radiance (map.uvToDirection ({ 0.25f, 0.53f })).y;
    for (int k = 1; k <= 4 * stepsPerTexel; ++k)
    {
        const float u = 0.25f + static_cast<float> (k) / (static_cast<float> (stepsPerTexel) * width);
        const float value = map.radiance (map.uvToDirection ({ u, 0.53f })).y;
        largestStep = std::max (largestStep, static_cast<double> (std::fabs (value - previous)));
        previous = value;
    }
    check (largestStep < 1.1 / stepsPerTexel, "the environment lookup is bilinear, largest step " + describe (largestStep));

    const double smooth = refractedBlockiness (false, 41);
    const double flat = refractedBlockiness (true, 41);
    check (smooth < 0.05, "a smooth tube refracts the banded environment smoothly: " + describe (smooth));
    check (flat > 0.08, "the flat sided tube shows the steps the measure detects: " + describe (flat));
}

void causticTests()
{
    slabCausticTests();
    ringCausticTests();
    dispersionCausticTests();
    tirCausticTests();
    gratingCausticTests();
    gratingOrderLightSamplingTests();
    gratingAcrossGrooveRoughnessTests();
    pitScatterLightSamplingTests();
    pitScatterThroughFlatTopTests();
    lightSampleThroughSlabTests();
}

} // namespace

// What the R2 glass ring's fringes and caustic depend on: the Cauchy spread,
// the colour of one wavelength, four wavelengths in every dispersive sample,
// signed splats, and a receiver spectrum with no steps in it.
void spectralQualityTests()
{
    // B = 0.004 um^2, the ring's value: n(450) - n(650) = B (1/0.45^2 - 1/0.65^2).
    const double spread = static_cast<double> (pt::cauchyIor (1.5f, 0.004f, 450.0f))
                        - static_cast<double> (pt::cauchyIor (1.5f, 0.004f, 650.0f));
    check (closeRelative (spread, 0.01028, 0.01), "Cauchy B 0.004 spreads the index by " + describe (spread));

    // All four channels on one wavelength: the conversion is 400 w(lambda).
    pt::Wavelengths pure;
    pure.spectral = true;
    pure.nm.fill (515.0f);
    const Vec3 green = pt::channelsToRgb (pt::Spectrum::filled (1.0f), pure);
    const Vec3 expected = pt::rgbMatchingWeightAt (515.0f) * 400.0f;
    check (closeRelative (green.y, expected.y, 1.0e-3) && green.y > 3.0f && green.x < -1.5f,
           "515 nm converts to a saturated green with negative red: " + describe (green.x) + ", " + describe (green.y));

    // A dispersive pane under a white sky. Every direction sees radiance 1, so
    // a sample's colour is only its wavelengths' weights: one wavelength swings
    // red by a standard deviation of 3, four rotated ones by 0.32.
    pt::RenderScene pane;
    pane.materials = { clearGlass (1.5f, 0.02f) };
    const Vec3 paneCentre { 0.0f, 0.0f, 0.0f };
    appendMesh (pane, box ({ -1.0f, -1.0f, -0.1f }, { 1.0f, 1.0f, 0.1f }), 0, &paneCentre);
    pane.lightRecords = { environmentRecord (1.0f) };
    pane.commit();
    pt::IntegratorSettings settings;
    settings.maxDepth = 8;
    pt::Ray ray;
    ray.origin = { 0.05f, 0.03f, 2.0f };
    ray.direction = pt::normalizeOrZero (Vec3 { 0.1f, 0.05f, -1.0f });
    double mean = 0.0;
    double square = 0.0;
    const int samples = 512;
    for (int i = 0; i < samples; ++i)
    {
        const pt::PathSampler sampler (0, static_cast<std::uint32_t> (i), 7, settings.pattern);
        pt::PathCounters counters;
        Vec3 radiance;
        check (pt::tracePath (pane, settings, ray, sampler, counters, radiance), "the pane's path is finite");
        mean += radiance.x;
        square += static_cast<double> (radiance.x) * static_cast<double> (radiance.x);
    }
    mean /= samples;
    const double deviation = std::sqrt (std::max (0.0, square / samples - mean * mean));
    check (closeRelative (mean, 1.0, 0.05), "a white sky through the pane stays white: red " + describe (mean));
    check (deviation < 1.0, "one dispersive sample holds several wavelengths: red deviates by " + describe (deviation));

    // Splats keep their sign and their size. A single wavelength's peak at
    // 1080p, about 10^7, is well inside the range.
    pt::Accumulator film;
    film.resize (1, 1);
    film.addSplat (0, 0, { -1.5f, 2.0f, 1.0e7f });
    film.addSplat (0, 0, { 0.5f, 0.0f, 0.0f });
    film.addLightPath();
    const pt::Rgb splat = film.splatMean (0, 0);
    check (closeRelative (splat.r, -1.0, 1.0e-5) && closeRelative (splat.g, 2.0, 1.0e-5)
               && closeRelative (splat.b, 1.0e7, 1.0e-5),
           "splats sum signed and unclipped: " + describe (splat.r) + ", " + describe (splat.b));

    // Not the source of the R2 contours: the receiver's spectrum is continuous,
    // so its reflectance has no steps a caustic could show. The contours come
    // from the ring's 96 x 24 mesh. Light refracted through facets and C0
    // normals lands with a density that ripples at the mesh spacing, and
    // tracing that mesh with no sampler at all shows the ripple falling about
    // fourfold at 384 x 96.
    double largestStep = 0.0;
    Vec3 previous = pt::rgbBasisAt (380.0f);
    for (int nm = 381; nm <= 780; ++nm)
    {
        const Vec3 b = pt::rgbBasisAt (static_cast<float> (nm));
        largestStep = std::max (largestStep, static_cast<double> (std::max (std::fabs (b.x - previous.x),
                                                  std::max (std::fabs (b.y - previous.y), std::fabs (b.z - previous.z)))));
        previous = b;
    }
    check (largestStep < 0.05, "the receiver spectrum moves at most " + describe (largestStep) + " per nm");
}

int main()
{
    spectrumTests();
    spectralQualityTests();
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
    causticTests();
    coarseMeshTerminatorTests();
    authoredRefractionTests();
    shadingNormalReceiverSplatTests();
    shadingNormalDeltaSplatTests();
    ringFringeCoherenceTests();
    ringStarterFrameBandTests();
    strongDispersionReachesEvalTests();
    ringCausticColourTests();
    causticSplatColourTests();
    ringCausticFloorTests();
    refractionSmoothnessTests();
    std::cout << checks << " checks, " << failures << " failures\n";
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
