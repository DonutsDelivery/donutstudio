// Tests for the CPU path tracer integrator (Plan 3, PT-A2): the sampler, the
// BSDF lobes, light units, MIS consistency, furnace tests and the renderer.

#include "pathtrace/accumulator.h"
#include "pathtrace/bsdf.h"
#include "pathtrace/bvh.h"
#include "pathtrace/integrator.h"
#include "pathtrace/lights.h"
#include "pathtrace/sampler.h"
#include "pathtrace/scene_ingest.h"
#include "pathtrace/thread_pool.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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

std::string describe (float value)
{
    return std::to_string (value);
}

float u01 (std::mt19937& rng)
{
    return static_cast<float> (static_cast<std::uint32_t> (rng()) >> 8) * (1.0f / 16777216.0f);
}

float maxComponent (Vec3 v) { return std::max (v.x, std::max (v.y, v.z)); }
float meanOf (Vec3 v) { return (v.x + v.y + v.z) / 3.0f; }

Vec3 uniformSphere (std::mt19937& rng)
{
    const float z = 1.0f - 2.0f * u01 (rng);
    const float r = std::sqrt (std::max (0.0f, 1.0f - z * z));
    const float phi = 2.0f * pt::kPi * u01 (rng);
    return { r * std::cos (phi), r * std::sin (phi), z };
}

Vec3 upperHemisphere (std::mt19937& rng, float minCos)
{
    for (;;)
    {
        Vec3 v = uniformSphere (rng);
        if (v.z < 0.0f)
            v.z = -v.z;
        if (v.z >= minCos)
            return v;
    }
}

Vec3 directionWithCos (float c)
{
    return { std::sqrt (std::max (0.0f, 1.0f - c * c)), 0.0f, c };
}

bool relativelyClose (float a, float b, float tolerance)
{
    return std::fabs (a - b) <= tolerance * std::max (1.0f, std::max (std::fabs (a), std::fabs (b)));
}

// ---------------------------------------------------------------- sampler

void samplerTests()
{
    {
        const pt::PathSampler a (17, 5, 3);
        const pt::PathSampler b (17, 5, 3);
        check (a.get1D (7) == b.get1D (7) && a.get2D (9).x == b.get2D (9).x && a.get2D (9).y == b.get2D (9).y,
               "the sampler is a pure function of pixel, sample, generation and dimension");
        const pt::PathSampler c (17, 5, 4);
        check (a.get2D (9).x != c.get2D (9).x || a.get2D (9).y != c.get2D (9).y,
               "a new generation gives new sample values");
    }

    const std::uint32_t dimensions[] { 0u, 5u, 37u, 1000u };
    const std::uint32_t pixels[] { 0u, 12345u };
    for (const std::uint32_t pixel : pixels)
        for (const std::uint32_t dimension : dimensions)
        {
            std::vector<int> cells (256, 0);
            std::vector<int> bins (256, 0);
            bool inRange = true;
            for (std::uint32_t s = 0; s < 256; ++s)
            {
                const pt::PathSampler sampler (pixel, s, 42);
                const pt::Vec2 v = sampler.get2D (dimension);
                const float w = sampler.get1D (dimension);
                inRange = inRange && v.x >= 0.0f && v.x < 1.0f && v.y >= 0.0f && v.y < 1.0f && w >= 0.0f && w < 1.0f;
                const int cx = std::min (15, static_cast<int> (v.x * 16.0f));
                const int cy = std::min (15, static_cast<int> (v.y * 16.0f));
                cells[static_cast<std::size_t> (cx * 16 + cy)] += 1;
                bins[static_cast<std::size_t> (std::min (255, static_cast<int> (w * 256.0f)))] += 1;
            }
            const std::string where = " (pixel " + std::to_string (pixel) + ", dimension " + std::to_string (dimension) + ")";
            check (inRange, "sample values lie in [0, 1)" + where);
            check (std::all_of (cells.begin(), cells.end(), [] (int n) { return n == 1; }),
                   "256 2D samples put exactly one point in every 16x16 cell" + where);
            check (std::all_of (bins.begin(), bins.end(), [] (int n) { return n == 1; }),
                   "256 1D samples put exactly one point in every 1/256 bin" + where);
        }

    // Different dimensions are decorrelated.
    {
        const int n = 4096;
        double sa = 0.0, sb = 0.0, sab = 0.0, saa = 0.0, sbb = 0.0;
        for (int s = 0; s < n; ++s)
        {
            const pt::PathSampler sampler (77, static_cast<std::uint32_t> (s), 9);
            const double a = sampler.get1D (3);
            const double b = sampler.get1D (4);
            sa += a; sb += b; sab += a * b; saa += a * a; sbb += b * b;
        }
        const double cov = sab / n - (sa / n) * (sb / n);
        const double r = cov / std::sqrt ((saa / n - (sa / n) * (sa / n)) * (sbb / n - (sb / n) * (sb / n)));
        check (std::fabs (r) < 0.06, "neighbouring dimensions are uncorrelated (r = " + std::to_string (r) + ")");
    }

    // The independent fallback stays in range and is unbiased.
    {
        double sum = 0.0;
        bool inRange = true;
        for (std::uint32_t s = 0; s < 4096; ++s)
        {
            const pt::PathSampler sampler (5, s, 1, pt::SamplePattern::Independent);
            const float v = sampler.get1D (2);
            const pt::Vec2 w = sampler.get2D (2);
            inRange = inRange && v >= 0.0f && v < 1.0f && w.x >= 0.0f && w.x < 1.0f && w.y >= 0.0f && w.y < 1.0f;
            sum += v;
        }
        check (inRange, "independent samples lie in [0, 1)");
        check (std::fabs (sum / 4096.0 - 0.5) < 0.02, "independent samples average one half");
    }
}

// ---------------------------------------------------------------- BSDF lobes

struct LobeCase
{
    std::string name;
    pt::Lobe lobe;
    bool uniformCheck = false;  // the lobe is wide enough for a uniform-sphere estimate
    bool reciprocal = false;    // reflection-only lobe, tested for reciprocity
};

pt::Lobe makeLobe (pt::LobeKind kind, float alpha, Vec3 f0, float eta)
{
    pt::Lobe lobe;
    lobe.kind = kind;
    lobe.weight = { 1.0f, 1.0f, 1.0f };
    lobe.transmissionTint = { 1.0f, 1.0f, 1.0f };
    lobe.f0 = f0;
    lobe.alpha = alpha;
    lobe.eta = eta;
    lobe.selectWeight = 1.0f;
    return lobe;
}

void lobeTests()
{
    std::mt19937 rng (1234);
    const Vec3 one { 1.0f, 1.0f, 1.0f };
    const Vec3 dielectricF0 { 0.04f, 0.04f, 0.04f };
    std::vector<LobeCase> cases;
    cases.push_back ({ "lambert", makeLobe (pt::LobeKind::Lambert, 1.0f, one, 1.5f), true, true });
    cases.push_back ({ "ggx conductor a=0.09", makeLobe (pt::LobeKind::GgxReflection, 0.09f, one, 1.5f), false, true });
    cases.push_back ({ "ggx conductor a=0.25", makeLobe (pt::LobeKind::GgxReflection, 0.25f, one, 1.5f), true, true });
    cases.push_back ({ "ggx conductor a=0.64", makeLobe (pt::LobeKind::GgxReflection, 0.64f, one, 1.5f), true, true });
    cases.push_back ({ "ggx conductor a=1", makeLobe (pt::LobeKind::GgxReflection, 1.0f, one, 1.5f), true, true });
    cases.push_back ({ "ggx dielectric a=0.25", makeLobe (pt::LobeKind::GgxReflection, 0.25f, dielectricF0, 1.5f), true, true });
    cases.push_back ({ "rough glass in a=0.09", makeLobe (pt::LobeKind::RoughDielectric, 0.09f, one, 1.5f), false, false });
    cases.push_back ({ "rough glass in a=0.25", makeLobe (pt::LobeKind::RoughDielectric, 0.25f, one, 1.5f), true, false });
    cases.push_back ({ "rough glass in a=0.64", makeLobe (pt::LobeKind::RoughDielectric, 0.64f, one, 1.5f), true, false });
    cases.push_back ({ "rough glass out a=0.25", makeLobe (pt::LobeKind::RoughDielectric, 0.25f, one, 1.0f / 1.5f), true, false });
    cases.push_back ({ "rough glass out a=0.64", makeLobe (pt::LobeKind::RoughDielectric, 0.64f, one, 1.0f / 1.5f), true, false });

    const float cosines[] { 0.95f, 0.5f, 0.15f };
    for (const LobeCase& item : cases)
        for (const float c : cosines)
        {
            const std::string where = " [" + item.name + ", cos " + describe (c) + "]";
            const Vec3 wo = directionWithCos (c);

            // Importance-sampled energy and sample/eval agreement.
            const int n = 40000;
            double energy[3] { 0.0, 0.0, 0.0 };
            int evaluated = 0;
            int pdfMismatches = 0;
            int weightMismatches = 0;
            for (int i = 0; i < n; ++i)
            {
                pt::BsdfSample s;
                const float choice = u01 (rng);
                const pt::Vec2 u { u01 (rng), u01 (rng) };
                if (! pt::sampleLobe (item.lobe, wo, choice, u, s))
                    continue;
                energy[0] += s.weight.x;
                energy[1] += s.weight.y;
                energy[2] += s.weight.z;
                const pt::BsdfEval e = pt::evalLobe (item.lobe, wo, s.wi);
                ++evaluated;
                if (! relativelyClose (e.pdf, s.pdf, 2.0e-3f))
                    ++pdfMismatches;
                if (! (e.pdf > 0.0f) || ! relativelyClose (e.value.x / e.pdf, s.weight.x, 2.0e-3f))
                    ++weightMismatches;
            }
            const float meanWeight = static_cast<float> (std::max (energy[0], std::max (energy[1], energy[2])) / n);
            check (meanWeight <= 1.005f, "sampled energy is at most 1" + where + " got " + describe (meanWeight));
            check (evaluated > n / 4, "most samples succeed" + where);
            check (pdfMismatches <= evaluated / 1000, "sample pdf equals eval pdf" + where
                                                          + " mismatches " + std::to_string (pdfMismatches));
            check (weightMismatches <= evaluated / 1000, "sample weight equals eval value / pdf" + where
                                                             + " mismatches " + std::to_string (weightMismatches));

            // Uniform-sphere estimate of the integral of f * cos and of the pdf.
            if (item.uniformCheck && c > 0.3f)
            {
                const int m = 400000;
                double integral = 0.0;
                double pdfIntegral = 0.0;
                for (int i = 0; i < m; ++i)
                {
                    const pt::BsdfEval e = pt::evalLobe (item.lobe, wo, uniformSphere (rng));
                    integral += e.value.x;
                    pdfIntegral += e.pdf;
                }
                const double scale = 4.0 * 3.14159265358979323846 / m;
                const float fcos = static_cast<float> (integral * scale);
                const float pdfTotal = static_cast<float> (pdfIntegral * scale);
                check (fcos <= 1.02f, "integral of f cos is at most 1" + where + " got " + describe (fcos));
                check (std::fabs (fcos - meanWeight) <= 0.04f, "uniform estimate matches sampled energy" + where
                                                                   + " got " + describe (fcos) + " vs " + describe (meanWeight));
                check (pdfTotal <= 1.03f, "pdf integrates to at most 1" + where + " got " + describe (pdfTotal));
            }

            // Reciprocity of reflection lobes.
            if (item.reciprocal)
            {
                int violations = 0;
                for (int i = 0; i < 2000; ++i)
                {
                    const Vec3 a = upperHemisphere (rng, 0.05f);
                    const Vec3 b = upperHemisphere (rng, 0.05f);
                    const float fab = pt::evalLobe (item.lobe, a, b).value.x / b.z;
                    const float fba = pt::evalLobe (item.lobe, b, a).value.x / a.z;
                    if (! relativelyClose (fab, fba, 1.0e-3f))
                        ++violations;
                }
                check (violations == 0, "f(a, b) equals f(b, a)" + where);
            }
        }

    // The tabulated GGX albedo agrees with sampling.
    {
        const pt::Lobe lobe = makeLobe (pt::LobeKind::GgxReflection, 0.25f, dielectricF0, 1.5f);
        const Vec3 wo = directionWithCos (0.5f);
        double sum = 0.0;
        const int n = 100000;
        for (int i = 0; i < n; ++i)
        {
            pt::BsdfSample s;
            if (pt::sampleLobe (lobe, wo, 0.0f, { u01 (rng), u01 (rng) }, s))
                sum += s.weight.x;
        }
        float scale = 0.0f;
        float bias = 0.0f;
        pt::ggxAlbedoScaleBias (0.5f, 0.25f, scale, bias);
        const float table = 0.04f * scale + bias;
        check (std::fabs (table - static_cast<float> (sum / n)) < 0.005f,
               "GGX albedo table matches sampling: " + describe (table) + " vs " + describe (static_cast<float> (sum / n)));
    }

    check (std::fabs (pt::fresnelDielectric (1.0f, 1.5f) - 0.04f) < 1.0e-5f, "normal-incidence Fresnel of ior 1.5 is 0.04");
    check (pt::fresnelDielectric (0.1f, 1.0f / 1.5f) == 1.0f, "grazing light inside glass is totally reflected");

    // Smooth dielectric: delta lobes, Fresnel split, Snell's law, unit weights.
    const float etas[] { 1.5f, 1.0f / 1.5f };
    const float smoothCosines[] { 0.9f, 0.5f, 0.1f };
    for (const float eta : etas)
        for (const float c : smoothCosines)
        {
            const std::string where = " [smooth eta " + describe (eta) + ", cos " + describe (c) + "]";
            const pt::Lobe lobe = makeLobe (pt::LobeKind::SmoothDielectric, 1.0f, one, eta);
            const Vec3 wo = directionWithCos (c);
            const int n = 20000;
            int reflected = 0;
            bool unitWeights = true;
            bool snell = true;
            bool deltaFlags = true;
            for (int i = 0; i < n; ++i)
            {
                pt::BsdfSample s;
                if (! pt::sampleLobe (lobe, wo, u01 (rng), { u01 (rng), u01 (rng) }, s))
                {
                    unitWeights = false;
                    continue;
                }
                unitWeights = unitWeights && s.weight.x == 1.0f && s.weight.y == 1.0f && s.weight.z == 1.0f;
                deltaFlags = deltaFlags && (s.flags & pt::kBsdfDelta) != 0 && s.pdf > 0.0f && s.pdf <= 1.0f;
                if (s.wi.z > 0.0f)
                {
                    ++reflected;
                    snell = snell && std::fabs (s.wi.z - c) < 1.0e-5f;
                }
                else
                {
                    const float sinI = std::sqrt (1.0f - c * c);
                    const float sinT = std::sqrt (std::max (0.0f, 1.0f - s.wi.z * s.wi.z));
                    snell = snell && std::fabs (sinT - sinI / eta) < 1.0e-4f;
                }
            }
            const float f = pt::fresnelDielectric (c, eta);
            check (unitWeights, "smooth dielectric samples carry weight 1" + where);
            check (deltaFlags, "smooth dielectric samples are delta" + where);
            check (snell, "smooth dielectric follows mirror reflection and Snell's law" + where);
            check (std::fabs (static_cast<float> (reflected) / n - f) < 0.02f, "reflected share equals Fresnel" + where);
            const pt::BsdfEval e = pt::evalLobe (lobe, wo, { 0.0f, 0.0f, 1.0f });
            check (e.pdf == 0.0f && maxComponent (e.value) == 0.0f, "delta lobes evaluate to zero" + where);
        }
}

// ---------------------------------------------------------------- materials

void materialBsdfTests()
{
    std::mt19937 rng (99);
    struct MaterialCase
    {
        std::string name;
        pt::Material material;
        float minEnergy;
        float maxEnergy;
    };
    std::vector<MaterialCase> cases;
    {
        pt::Material m;
        m.baseColor = { 1.0f, 1.0f, 1.0f };
        m.roughness = 0.5f;
        cases.push_back ({ "white plastic", m, 0.97f, 1.01f });
    }
    {
        pt::Material m;
        m.baseColor = { 0.8f, 0.5f, 0.3f };
        m.roughness = 0.2f;
        cases.push_back ({ "orange plastic", m, 0.0f, 1.01f });
    }
    {
        pt::Material m;
        m.baseColor = { 0.9f, 0.6f, 0.2f };
        m.metallic = 1.0f;
        m.roughness = 0.3f;
        cases.push_back ({ "gold", m, 0.0f, 1.01f });
    }
    {
        pt::Material m;
        m.transmission = 1.0f;
        m.roughness = 0.0f;
        cases.push_back ({ "clear glass", m, 0.999f, 1.001f });
    }
    {
        pt::Material m;
        m.transmission = 1.0f;
        m.roughness = 0.4f;
        cases.push_back ({ "frosted glass", m, 0.0f, 1.01f });
    }
    {
        pt::Material m;
        m.baseColor = { 0.7f, 0.9f, 0.8f };
        m.metallic = 0.5f;
        m.transmission = 0.5f;
        m.roughness = 0.2f;
        cases.push_back ({ "mixed", m, 0.0f, 1.01f });
    }
    {
        pt::Material m;
        m.baseColor = { 4.0f, -1.0f, std::numeric_limits<float>::quiet_NaN() };
        m.metallic = 2.0f;
        m.roughness = -3.0f;
        m.ior = 0.5f;
        cases.push_back ({ "out of range inputs", m, 0.0f, 1.01f });
    }

    const Vec3 up { 0.0f, 0.0f, 1.0f };
    const float cosines[] { 0.9f, 0.4f };
    for (const MaterialCase& item : cases)
        for (const float c : cosines)
        {
            const std::string where = " [" + item.name + ", cos " + describe (c) + "]";
            const Vec3 wo = directionWithCos (c);
            pt::Bsdf bsdf;
            pt::buildMaterialBsdf (item.material, up, up, wo, true, bsdf);
            check (bsdf.lobeCount() > 0, "material has lobes" + where);
            const int n = 40000;
            double energy[3] { 0.0, 0.0, 0.0 };
            int mismatches = 0;
            bool finite = true;
            for (int i = 0; i < n; ++i)
            {
                pt::BsdfSample s;
                if (! bsdf.sample (wo, u01 (rng), { u01 (rng), u01 (rng) }, s))
                    continue;
                finite = finite && pt::isFinite (s.weight) && std::isfinite (s.pdf);
                energy[0] += s.weight.x;
                energy[1] += s.weight.y;
                energy[2] += s.weight.z;
                if ((s.flags & pt::kBsdfDelta) == 0)
                {
                    const pt::BsdfEval e = bsdf.eval (wo, s.wi);
                    if (! relativelyClose (e.pdf, s.pdf, 2.0e-3f))
                        ++mismatches;
                }
            }
            const float top = static_cast<float> (std::max (energy[0], std::max (energy[1], energy[2])) / n);
            check (finite, "material samples are finite" + where);
            check (top <= item.maxEnergy, "material energy at most " + describe (item.maxEnergy) + where + " got " + describe (top));
            check (top >= item.minEnergy, "material energy at least " + describe (item.minEnergy) + where + " got " + describe (top));
            check (mismatches <= n / 1000, "material sample pdf equals eval pdf" + where);
        }
}

// ---------------------------------------------------------------- scenes

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

bool allFinite (const pt::Accumulator& accumulator)
{
    const std::vector<float> image = accumulator.resolveRgba32f();
    return std::all_of (image.begin(), image.end(), [] (float v) { return std::isfinite (v); });
}

// Mean over pixels whose centre ray hits the scene, and their count.
double coveredMean (const pt::RenderScene& scene, const pt::PinholeCamera& camera,
                    const pt::Accumulator& accumulator, int& covered, float& lowest, float& highest)
{
    double sum = 0.0;
    covered = 0;
    lowest = std::numeric_limits<float>::infinity();
    highest = -std::numeric_limits<float>::infinity();
    for (std::uint32_t y = 0; y < camera.height; ++y)
        for (std::uint32_t x = 0; x < camera.width; ++x)
        {
            pt::Hit hit;
            if (! scene.bvh.intersect (camera.generateRay (static_cast<float> (x) + 0.5f, static_cast<float> (y) + 0.5f), hit))
                continue;
            const float v = meanOf (pixel (accumulator, x, y));
            sum += v;
            lowest = std::min (lowest, v);
            highest = std::max (highest, v);
            ++covered;
        }
    return covered > 0 ? sum / covered : 0.0;
}

// ---------------------------------------------------------------- furnace

void furnaceTest (const std::string& name, const pt::Material& material, std::uint32_t maxDepth,
                  std::uint32_t samples, pt::SamplePattern pattern, float lowRatio, float highRatio, float pixelSpread)
{
    const float environment = 0.5f;
    pt::RenderScene scene;
    scene.materials.push_back (material);
    const Vec3 centre {};
    appendMesh (scene, uvSphere (24, 48, 1.0f), 0, &centre);
    scene.lightRecords.push_back (environmentRecord (environment));
    scene.commit();
    check (scene.lights.size() == 1 && scene.lights.hasEnvironment(), name + ": the environment is the only light");

    const pt::PinholeCamera camera = makeCamera ({ 0.0f, 0.0f, 3.5f }, pt::Quat {}, 0.4f, 32, 32);
    pt::PathTracer tracer (4);
    tracer.settings.maxDepth = maxDepth;
    tracer.settings.pattern = pattern;
    const pt::CancellationToken cancel {};
    const pt::RenderStats stats = tracer.renderFixed (scene, camera, 32, 32, samples, 1, cancel);
    check (! stats.invalidRequest && ! stats.cancelled, name + ": render ran");
    check (stats.nonFiniteSamples == 0, name + ": no non-finite samples");
    check (stats.samplesAdded == 32ull * 32ull * samples, name + ": every sample accumulated");
    check (stats.indirectClamp == 0.0f && stats.clampedContributions == 0, name + ": nothing clamped by default");

    int covered = 0;
    float lowest = 0.0f;
    float highest = 0.0f;
    const double mean = coveredMean (scene, camera, tracer.accumulator(), covered, lowest, highest);
    check (covered > 300, name + ": the sphere covers the image centre");
    check (mean >= environment * lowRatio && mean <= environment * highRatio,
           name + ": sphere mean " + std::to_string (mean) + " converges to the environment " + describe (environment));
    check (lowest >= environment * (1.0f - pixelSpread) && highest <= environment * (1.0f + pixelSpread),
           name + ": every sphere pixel near the environment, range " + describe (lowest) + " to " + describe (highest));
    const Vec3 corner = pixel (tracer.accumulator(), 0, 0);
    check (std::fabs (corner.x - environment) < 1.0e-5f, name + ": the background shows the environment radiance");
}

void furnaceTests()
{
    pt::Material white;
    white.baseColor = { 1.0f, 1.0f, 1.0f };
    white.roughness = 1.0f;
    furnaceTest ("white furnace", white, 12, 64, pt::SamplePattern::OwenSobol, 0.985f, 1.015f, 0.3f);
    furnaceTest ("white furnace, independent sampler", white, 12, 64, pt::SamplePattern::Independent, 0.98f, 1.02f, 0.4f);

    pt::Material glass;
    glass.baseColor = { 1.0f, 1.0f, 1.0f };
    glass.transmission = 1.0f;
    glass.roughness = 0.0f;
    glass.ior = 1.5f;
    // Facets near the silhouette can trap a rare path by total internal
    // reflection until maxDepth, so pixels get a wider band than the mean.
    furnaceTest ("smooth glass furnace", glass, 64, 16, pt::SamplePattern::OwenSobol, 0.985f, 1.0001f, 0.2f);

    pt::Material frosted = glass;
    frosted.roughness = 0.3f;
    furnaceTest ("rough glass furnace", frosted, 64, 32, pt::SamplePattern::OwenSobol, 0.8f, 1.01f, 0.6f);
}

// ---------------------------------------------------------------- MIS

void misConsistencyTests()
{
    pt::RenderScene scene;
    pt::Material floor;
    floor.baseColor = { 0.8f, 0.8f, 0.8f };
    floor.roughness = 1.0f;
    pt::Material lamp;
    lamp.baseColor = { 0.0f, 0.0f, 0.0f };
    lamp.emission = { 5.0f, 5.0f, 5.0f };
    scene.materials = { floor, lamp };
    appendMesh (scene, quad ({ 0.0f, 0.0f, 0.0f }, { 1.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, -1.0f }), 0);
    appendMesh (scene, quad ({ 0.0f, 1.0f, 0.0f }, { 0.4f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.4f }), 1);
    scene.commit();
    check (scene.lights.size() == 2 && scene.lights.stats().emissiveTriangles == 2, "the lamp quad becomes two area lights");
    check (! scene.lights.hasEnvironment(), "the MIS scene has no environment");

    const pt::PinholeCamera camera = makeCamera ({ 0.0f, 2.5f, 2.5f }, pt::Quat { -0.38268343f, 0.0f, 0.0f, 0.92387953f },
                                                 0.6f, 20, 20);
    const pt::CancellationToken cancel {};
    const pt::LightStrategy strategies[] { pt::LightStrategy::MultipleImportance, pt::LightStrategy::LightOnly,
                                           pt::LightStrategy::BsdfOnly };
    double means[3] { 0.0, 0.0, 0.0 };
    for (int i = 0; i < 3; ++i)
    {
        pt::PathTracer tracer (4);
        tracer.settings.maxDepth = 1;
        tracer.settings.strategy = strategies[i];
        const pt::RenderStats stats = tracer.renderFixed (scene, camera, 20, 20, 256, 5, cancel);
        check (stats.nonFiniteSamples == 0, "MIS scene renders finite samples");
        means[i] = imageMean (tracer.accumulator());
    }
    check (means[0] > 0.05, "the lit floor is visible: mean " + std::to_string (means[0]));
    check (std::fabs (means[1] - means[0]) <= 0.03 * means[0],
           "light sampling only agrees with MIS: " + std::to_string (means[1]) + " vs " + std::to_string (means[0]));
    check (std::fabs (means[2] - means[0]) <= 0.03 * means[0],
           "BSDF sampling only agrees with MIS: " + std::to_string (means[2]) + " vs " + std::to_string (means[0]));
}

// ---------------------------------------------------------------- light units

// The raster's direct base GGX (backend_opengl.cpp baseSurfaceDirect).
Vec3 rasterBaseSurfaceDirect (Vec3 base, float metallic, float roughness, float dielectric, Vec3 n, Vec3 l, Vec3 v)
{
    const float nl = std::max (0.0f, pt::dot (n, l));
    const float nv = std::max (0.0f, pt::dot (n, v));
    if (nl <= 0.0f || nv <= 0.0f)
        return {};
    const Vec3 h = pt::normalizeOrZero (l + v);
    const float nh = std::max (0.0f, pt::dot (n, h));
    const float vh = std::max (0.0f, pt::dot (v, h));
    const float alpha = std::max (roughness * roughness, 0.0025f);
    const float a2 = alpha * alpha;
    const float denominator = (1.0f - nh * nh) + a2 * nh * nh;
    const float distribution = a2 / (pt::kPi * denominator * denominator);
    const float visibility = 1.0f / ((nl + std::sqrt (a2 + (1.0f - a2) * nl * nl)) * (nv + std::sqrt (a2 + (1.0f - a2) * nv * nv)));
    const float s = std::pow (1.0f - vh, 5.0f);
    Vec3 result;
    for (int c = 0; c < 3; ++c)
    {
        const float f0 = dielectric + (base[c] - dielectric) * metallic;
        const float fresnel = f0 + (1.0f - f0) * s;
        const float diffuse = (1.0f - fresnel) * (1.0f - metallic) * base[c] / pt::kPi;
        result[c] = (distribution * visibility * fresnel + diffuse) * nl;
    }
    return result;
}

void lightUnitTests()
{
    pt::Material floor;
    floor.baseColor = { 0.8f, 0.8f, 0.8f };
    floor.roughness = 1.0f;
    const Vec3 up { 0.0f, 1.0f, 0.0f };
    pt::Bsdf bsdf;
    pt::buildMaterialBsdf (floor, up, up, up, true, bsdf);
    const float tracerResponse = bsdf.eval (up, up).value.x;
    const float rasterResponse = rasterBaseSurfaceDirect (floor.baseColor, 0.0f, 1.0f, 0.04f, up, up, up).x;
    check (std::fabs (tracerResponse - rasterResponse) <= 0.05f * rasterResponse,
           "tracer and raster BRDFs agree at normal incidence: " + describe (tracerResponse) + " vs " + describe (rasterResponse));

    const pt::Quat down { -0.70710678f, 0.0f, 0.0f, 0.70710678f };  // local -Z to world -Y
    struct UnitCase
    {
        std::string name;
        grid::SceneLightRecord record;
        float irradiance;
    };
    std::vector<UnitCase> cases;
    {
        grid::SceneLightRecord light;
        light.kind = grid::SceneLightKind::Point;
        light.transform.translation = { 0.0f, 2.0f, 0.0f };
        light.intensity = 10.0f;
        cases.push_back ({ "point at distance 2", light, 10.0f / 4.0f });
        light.transform.translation = { 0.0f, 0.5f, 0.0f };
        light.intensity = 1.0f;
        cases.push_back ({ "point inside unit distance", light, 1.0f });
        light.transform.translation = { 0.0f, 2.0f, 0.0f };
        light.intensity = 10.0f;
        light.range = 4.0f;
        cases.push_back ({ "point with range", light, 10.0f / 4.0f * 0.25f });
    }
    {
        grid::SceneLightRecord light;
        light.kind = grid::SceneLightKind::Spot;
        light.transform.translation = { 0.0f, 2.0f, 0.0f };
        light.transform.rotation = { down.x, down.y, down.z, down.w };
        light.intensity = 10.0f;
        light.innerConeAngle = 0.3f;
        light.outerConeAngle = 0.5f;
        cases.push_back ({ "spot on axis", light, 10.0f / 4.0f });
    }
    {
        grid::SceneLightRecord light;
        light.kind = grid::SceneLightKind::Directional;
        light.transform.rotation = { down.x, down.y, down.z, down.w };
        light.color = { 1.0f, 1.0f, 1.0f };
        light.intensity = 3.0f;
        cases.push_back ({ "directional from above", light, 3.0f });
    }

    const pt::PinholeCamera camera = makeCamera ({ 0.0f, 5.0f, 0.0f }, down, 0.05f, 33, 33);
    const pt::CancellationToken cancel {};
    for (const UnitCase& item : cases)
    {
        pt::RenderScene scene;
        scene.materials = { floor };
        appendMesh (scene, quad ({ 0.0f, 0.0f, 0.0f }, { 5.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, -5.0f }), 0);
        scene.lightRecords = { item.record };
        scene.commit();
        check (scene.lights.size() == 1, item.name + ": one light");
        pt::PathTracer tracer (2);
        tracer.settings.maxDepth = 1;
        tracer.renderFixed (scene, camera, 33, 33, 16, 3, cancel);
        const float value = pixel (tracer.accumulator(), 16, 16).x;
        const float expected = tracerResponse * item.irradiance;
        check (std::fabs (value - expected) <= 0.01f * expected,
               item.name + ": centre radiance " + describe (value) + " equals BSDF x irradiance " + describe (expected));
        const float raster = rasterResponse * item.irradiance;
        check (std::fabs (value - raster) <= 0.05f * raster,
               item.name + ": centre radiance " + describe (value) + " matches the raster " + describe (raster));
    }

    // A narrow spot leaves the image edge dark.
    {
        grid::SceneLightRecord light;
        light.kind = grid::SceneLightKind::Spot;
        light.transform.translation = { 0.0f, 2.0f, 0.0f };
        light.transform.rotation = { down.x, down.y, down.z, down.w };
        light.intensity = 10.0f;
        light.innerConeAngle = 0.02f;
        light.outerConeAngle = 0.05f;
        pt::RenderScene scene;
        scene.materials = { floor };
        appendMesh (scene, quad ({ 0.0f, 0.0f, 0.0f }, { 5.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, -5.0f }), 0);
        scene.lightRecords = { light };
        scene.commit();
        pt::PathTracer tracer (2);
        tracer.settings.maxDepth = 1;
        tracer.renderFixed (scene, camera, 33, 33, 4, 3, cancel);
        check (pixel (tracer.accumulator(), 16, 16).x > 0.5f, "narrow spot lights its axis");
        check (pixel (tracer.accumulator(), 0, 16).x == 0.0f, "narrow spot leaves points outside its cone dark");
    }
}

// ---------------------------------------------------------------- Cornell box

void addQuadObject (grid::Visual3DScene& scene, std::uint32_t objectId, std::uint32_t materialId,
                    Vec3 centre, Vec3 halfU, Vec3 halfV)
{
    const Vec3 n = pt::normalizeOrZero (pt::cross (halfU, halfV));
    const auto firstVertex = static_cast<std::uint32_t> (scene.vertexCount);
    const Vec3 corners[4] { centre - halfU - halfV, centre + halfU - halfV, centre + halfU + halfV, centre - halfU + halfV };
    for (const Vec3& corner : corners)
    {
        grid::SceneVertex& vertex = scene.vertices[scene.vertexCount++];
        vertex.position = { corner.x, corner.y, corner.z };
        vertex.normal = { n.x, n.y, n.z };
    }
    grid::SceneObjectRecord& object = scene.objects[scene.objectCount++];
    object.id.value = objectId;
    object.material.value = materialId;
    object.firstVertex = firstVertex;
    object.vertexCount = 4;
    object.firstIndex = static_cast<std::uint32_t> (scene.indexCount);
    object.indexCount = 6;
    for (const std::uint32_t index : { 0u, 1u, 2u, 0u, 2u, 3u })
        scene.indices[scene.indexCount++] = index;
}

std::unique_ptr<grid::Visual3DScene> cornellBox()
{
    auto scene = std::make_unique<grid::Visual3DScene>();
    scene->id.value = 1;
    const auto material = [&] (std::uint32_t id, Vec3 base, Vec3 emission) {
        grid::SceneMaterialRecord& record = scene->materials[scene->materialCount++];
        record.id.value = id;
        record.baseColor = { base.x, base.y, base.z };
        record.emissive = { emission.x, emission.y, emission.z };
        record.roughness = 1.0f;
    };
    material (1, { 0.75f, 0.75f, 0.75f }, {});
    material (2, { 0.75f, 0.1f, 0.1f }, {});
    material (3, { 0.1f, 0.75f, 0.1f }, {});
    material (4, { 0.0f, 0.0f, 0.0f }, { 6.0f, 6.0f, 6.0f });

    addQuadObject (*scene, 10, 1, { 0.0f, 0.0f, 0.0f }, { 1.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, -1.0f });   // floor
    addQuadObject (*scene, 11, 1, { 0.0f, 2.0f, 0.0f }, { 1.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 1.0f });    // ceiling
    addQuadObject (*scene, 12, 1, { 0.0f, 1.0f, -1.0f }, { 1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f });   // back
    addQuadObject (*scene, 13, 2, { -1.0f, 1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f });   // left, red
    addQuadObject (*scene, 14, 3, { 1.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f }, { 0.0f, 1.0f, 0.0f });    // right, green
    addQuadObject (*scene, 15, 4, { 0.0f, 1.98f, 0.0f }, { 0.3f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.3f });   // lamp

    grid::SceneLightRecord& point = scene->lights[scene->lightCount++];
    point.id.value = 20;
    point.kind = grid::SceneLightKind::Point;
    point.transform.translation = { 0.0f, 1.5f, 0.5f };
    point.intensity = 0.5f;
    grid::SceneLightRecord& spot = scene->lights[scene->lightCount++];
    spot.id.value = 21;
    spot.kind = grid::SceneLightKind::Spot;
    spot.transform.translation = { 0.5f, 1.9f, 0.5f };
    spot.transform.rotation = { -0.70710678f, 0.0f, 0.0f, 0.70710678f };
    spot.intensity = 1.0f;
    spot.innerConeAngle = 0.3f;
    spot.outerConeAngle = 0.6f;
    grid::SceneLightRecord& sky = scene->lights[scene->lightCount++];
    sky = environmentRecord (0.05f);
    sky.id.value = 22;

    grid::SceneCameraRecord& camera = scene->cameras[scene->cameraCount++];
    camera.id.value = 9;
    camera.transform.translation = { 0.0f, 1.0f, 3.4f };
    camera.verticalFovRadians = 0.9f;
    camera.nearPlane = 0.05f;
    camera.farPlane = 100.0f;
    scene->activeCamera.value = 9;
    return scene;
}

void cornellBoxTests()
{
    const auto box = cornellBox();
    pt::RenderScene scene;
    pt::prepareRenderScene (*box, scene);
    check (scene.geometry.triangleCount() == 12, "the box ingests to 12 triangles");
    check (scene.materials.size() == 4, "the box has four materials");
    check (scene.lights.size() == 5, "lamp triangles, point, spot and environment are lights: "
                                         + std::to_string (scene.lights.size()));
    check (scene.lights.stats().emissiveTriangles == 2 && scene.lights.stats().point == 1
               && scene.lights.stats().spot == 1 && scene.lights.stats().environment == 1,
           "light statistics count each kind");

    pt::PinholeCamera camera;
    check (pt::resolveActiveCamera (*box, 32, 32, camera), "the box camera resolves");
    pt::PathTracer tracer (4);
    const pt::CancellationToken cancel {};
    const pt::RenderStats stats = tracer.renderFixed (scene, camera, 32, 32, 32, 11, cancel);
    check (stats.nonFiniteSamples == 0 && tracer.accumulator().rejectedSamples() == 0, "the box renders no NaN or Inf");
    check (allFinite (tracer.accumulator()), "the box image is finite");
    check (stats.closestRays > 32ull * 32ull * 32ull && stats.shadowRays > 0, "the box traced bounce and shadow rays");
    check (stats.lights.emissiveTriangles == 2, "render stats carry the light statistics");

    bool nonNegative = true;
    for (std::uint32_t y = 0; y < 32; ++y)
        for (std::uint32_t x = 0; x < 32; ++x)
        {
            const Vec3 v = pixel (tracer.accumulator(), x, y);
            nonNegative = nonNegative && v.x >= 0.0f && v.y >= 0.0f && v.z >= 0.0f;
        }
    check (nonNegative, "the box image has no negative radiance");
    const double mean = imageMean (tracer.accumulator());
    check (mean > 0.02 && mean < 3.0, "the box mean radiance is plausible: " + std::to_string (mean));

    Vec3 left {};
    Vec3 right {};
    for (std::uint32_t y = 12; y < 20; ++y)
        for (std::uint32_t x = 3; x < 8; ++x)
        {
            left = left + pixel (tracer.accumulator(), x, y);
            right = right + pixel (tracer.accumulator(), 31 - x, y);
        }
    check (left.x > left.y, "the left wall reads red");
    check (right.y > right.x, "the right wall reads green");

    // The optional indirect clamp is reported, never silent.
    pt::PathTracer clamped (4);
    clamped.settings.indirectClamp = 0.1f;
    const pt::RenderStats clampStats = clamped.renderFixed (scene, camera, 32, 32, 8, 11, cancel);
    check (clampStats.indirectClamp == 0.1f, "render stats report the indirect clamp value");
    check (clampStats.clampedContributions > 0, "render stats count clamped contributions");
}

// ---------------------------------------------------------------- renderer

void determinismTests()
{
    const auto box = cornellBox();
    pt::RenderScene scene;
    pt::prepareRenderScene (*box, scene);
    pt::PinholeCamera camera;
    check (pt::resolveActiveCamera (*box, 24, 16, camera), "the box camera resolves at 24x16");
    const pt::CancellationToken cancel {};

    const auto render = [&] (unsigned threads, std::uint64_t generation) {
        pt::PathTracer tracer (threads);
        tracer.settings.tileSize = 8;
        const pt::RenderStats first = tracer.renderProgressive (scene, camera, 24, 16, 2, generation, cancel);
        const pt::RenderStats second = tracer.renderProgressive (scene, camera, 24, 16, 2, generation, cancel);
        check (first.firstSampleIndex == 0 && second.firstSampleIndex == 2, "progressive passes continue the sample index");
        check (first.tilesCompleted == first.tileCount && ! first.cancelled, "a progressive pass covers every tile");
        check (tracer.accumulator().sampleCount (5, 5) == 4, "two passes of two samples accumulate four");
        return tracer.accumulator().resolveRgba32f();
    };
    const std::vector<float> single = render (1, 7);
    const std::vector<float> multi = render (3, 7);
    const std::vector<float> other = render (3, 8);
    check (single.size() == multi.size()
               && std::memcmp (single.data(), multi.data(), single.size() * sizeof (float)) == 0,
           "the same seed renders the same image on any thread count");
    check (other.size() == single.size()
               && std::memcmp (single.data(), other.data(), single.size() * sizeof (float)) != 0,
           "a new generation renders new noise");

    // A new generation clears the accumulator; a wrong size is refused.
    {
        pt::PathTracer tracer (2);
        tracer.renderProgressive (scene, camera, 24, 16, 2, 1, cancel);
        tracer.renderProgressive (scene, camera, 24, 16, 1, 2, cancel);
        check (tracer.accumulator().generation() == 2 && tracer.accumulator().sampleCount (0, 0) == 1,
               "a new generation restarts accumulation");
        const pt::RenderStats refused = tracer.renderProgressive (scene, camera, 25, 16, 1, 2, cancel);
        check (refused.invalidRequest && refused.samplesAdded == 0, "a size that differs from the camera is refused");
    }

    // A cancelled token stops the pass.
    {
        pt::CancellationToken stop;
        stop.cancel();
        pt::PathTracer tracer (2);
        const pt::RenderStats stats = tracer.renderFixed (scene, camera, 24, 16, 4, 1, stop);
        check (stats.cancelled && stats.samplesAdded == 0, "a cancelled render adds nothing and says so");
    }
}

void nonFiniteTests()
{
    pt::RenderScene scene;
    pt::Material broken;
    broken.emission = { std::numeric_limits<float>::infinity(), 1.0f, 1.0f };
    scene.materials = { broken };
    appendMesh (scene, quad ({ 0.0f, 0.0f, 0.0f }, { 0.5f, 0.0f, 0.0f }, { 0.0f, 0.5f, 0.0f }), 0);
    scene.lightRecords = { environmentRecord (0.2f) };
    scene.commit();
    check (scene.lights.stats().invalidLights == 2, "non-finite emitters are counted, not sampled");

    const pt::PinholeCamera camera = makeCamera ({ 0.0f, 0.0f, 3.0f }, pt::Quat {}, 0.4f, 8, 8);
    pt::PathTracer tracer (2);
    const pt::CancellationToken cancel {};
    const pt::RenderStats stats = tracer.renderFixed (scene, camera, 8, 8, 4, 1, cancel);
    check (stats.nonFiniteSamples > 0, "non-finite samples are counted");
    check (stats.samplesAdded + stats.nonFiniteSamples == 8ull * 8ull * 4ull, "every sample is either added or dropped");
    check (tracer.accumulator().rejectedSamples() == 0, "dropped samples never reach the accumulator");
    check (allFinite (tracer.accumulator()), "the image stays finite");
    check (std::fabs (pixel (tracer.accumulator(), 0, 0).x - 0.2f) < 1.0e-5f, "pixels away from the bad emitter are unaffected");
}

void analyticSkyTests()
{
    grid::SceneLightRecord light;
    light.kind = grid::SceneLightKind::Environment;
    light.sky.version = 1;
    light.sky.sunElevationDegrees = 90.0f;
    light.sky.sunRadiusDegrees = 5.0f;
    pt::RenderScene scene;
    scene.lightRecords = { light };
    scene.commit();
    const pt::EnvironmentField& field = scene.lights.environment();
    check (field.analytic && field.sunSampleProbability > 0.0f, "an analytic sky with a sun samples its cap");
    const Vec3 zenith = scene.lights.environmentRadiance ({ 0.0f, 1.0f, 0.0f });
    const Vec3 expectedZenith { 0.12f + 1.0f * 10.0f, 0.3f + 0.9f * 10.0f, 0.7f + 0.7f * 10.0f };
    check (std::fabs (zenith.x - expectedZenith.x) < 1.0e-3f && std::fabs (zenith.z - expectedZenith.z) < 1.0e-3f,
           "the sun peak adds to the zenith colour");
    const Vec3 horizon = scene.lights.environmentRadiance ({ 1.0f, 0.0f, 0.0f });
    check (std::fabs (horizon.x - 0.6f) < 1.0e-5f, "the horizon colour is the gradient's middle");

    // The mixture pdf integrates to one over the sphere.
    std::mt19937 rng (5);
    double integral = 0.0;
    const int n = 200000;
    for (int i = 0; i < n; ++i)
        integral += field.pdf (uniformSphere (rng));
    integral *= 4.0 * 3.14159265358979323846 / n;
    check (std::fabs (integral - 1.0) < 0.03, "the environment pdf integrates to one: " + std::to_string (integral));

    // Sampled directions report the pdf the evaluator gives them.
    bool consistent = true;
    for (int i = 0; i < 1000; ++i)
    {
        float pdf = 0.0f;
        const Vec3 d = field.sample ({ u01 (rng), u01 (rng) }, pdf);
        consistent = consistent && std::fabs (pt::length (d) - 1.0f) < 1.0e-4f && pdf > 0.0f
                  && std::fabs (pdf - field.pdf (d)) <= 1.0e-4f * pdf;
    }
    check (consistent, "environment samples are unit directions with matching pdf");

    // An environment image without decoded pixels lights nothing, and that is
    // reported. pathtrace_environment_tests.cpp covers images with pixels.
    grid::SceneLightRecord image;
    image.kind = grid::SceneLightKind::Environment;
    image.environmentImage.version = 1;
    pt::RenderScene imageScene;
    imageScene.lightRecords = { image };
    imageScene.commit();
    check (imageScene.lights.stats().ignoredEnvironmentImages == 1 && ! imageScene.lights.hasEnvironment(),
           "an environment image without pixels is counted as ignored");
}

} // namespace

int main()
{
    samplerTests();
    lobeTests();
    materialBsdfTests();
    furnaceTests();
    misConsistencyTests();
    lightUnitTests();
    cornellBoxTests();
    determinismTests();
    nonFiniteTests();
    analyticSkyTests();
    std::cout << checks << " checks, " << failures << " failures\n";
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
