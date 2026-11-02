// Relief scatter on a grating: Rayleigh-Rice from a flat height spectrum (Tayabaly et al.
// 2016, arXiv 1610.00723). The lobe against its own formula and against a brute-force
// integral, then the path tracer: gain in both furnaces, light sampling against none, and
// the light it throws on a plate against the closed form.

#include "pathtrace/accumulator.h"
#include "pathtrace/integrator.h"
#include "pathtrace/scene_ingest.h"
#include "diffraction_material_admission.h"
#include "diffraction_order_bsdf.h"
#include "DiffractionMaterialPresets.h"
#include "VisualImportedSceneRenderOperationContract.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace
{

namespace pt = videohelper::pathtrace;
namespace dm = diffractionmaterial;
namespace grid = HarmonicMIDI::grid;
using pt::Vec3;
using Direction = std::array<double, 3>;

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

// Heights and spacings are the look's fitted values, not a measurement of anything.
dm::Description reliefMaterial (float height, bool threeFamilies)
{
    auto description = dm::makeAluminiumBinaryGratingPreset();
    description.geometry.grooveSpacingNanometres = 1500.0f;
    description.roughness.rmsSlope = 0.01f;
    description.roughness.reliefRmsHeightNanometres = height;
    description.roughness.reliefMinimumSpacingNanometres = 400.0f;
    if (threeFamilies)
    {
        description.geometry.lattice = dm::GratingLattice::CrossedTwoDimensional;
        description.geometry.secondaryDirectionUv = { 0.5f, 0.8660254f };
        description.geometry.secondaryGrooveSpacingNanometres = 1500.0f;
        description.geometry.thirdGrooveFamily = true;
    }
    return description;
}

std::shared_ptr<const dm::AdmittedDiffractionMaterialIR> admitted (const dm::Description& description)
{
    std::string error;
    auto admission = dm::admit (description, error);
    check (admission.has_value(), "the relief material admits: " + error);
    return admission ? std::make_shared<const dm::AdmittedDiffractionMaterialIR> (*admission) : nullptr;
}

using Single = std::array<float, 3>;
const Single kNormal { 0.0f, 0.0f, 1.0f };

Single single (const Direction& d)
{
    return { static_cast<float> (d[0]), static_cast<float> (d[1]), static_cast<float> (d[2]) };
}

dm::DiffractionOrderDistribution evaluated (const dm::AdmittedDiffractionMaterialIR& material, double wavelength, const Direction& wo)
{
    std::string error;
    dm::DiffractionLocalFrame frame;
    dm::DiffractionOrderDistribution orders;
    check (dm::evaluateDiffractionOrders (material, static_cast<float> (wavelength), frame, single (wo), { 0.5f, 0.5f }, orders, error),
           "the orders evaluate: " + error);
    return orders;
}

Direction unit (double x, double y, double z)
{
    const double size = std::sqrt (x * x + y * y + z * z);
    return { x / size, y / size, z / size };
}

double reliefValue (const dm::DiffractionOrderDistribution& orders, const Direction& wo, const Direction& wi)
{
    return dm::reliefScatterDensity (orders, kNormal, single (wo), single (wi));
}

// The relief's own power law, its bookkeeping, and where each wavelength's arm ends.
void energyAndPositionTests()
{
    const auto one = admitted (reliefMaterial (10.0f, false));
    const auto triple = admitted (reliefMaterial (18.0f, true));
    if (one == nullptr || triple == nullptr)
        return;
    const double pi = 3.14159265358979323846;
    const Direction wo = unit (-0.3, 0.1, 1.0);
    for (const double wavelength : { 400.0, 450.0, 550.0, 650.0, 700.0 })
    {
        const auto orders = evaluated (*one, wavelength, wo);
        const double phase = 4.0 * pi * 10.0 / wavelength;
        const double loss = phase * phase * wo[2];
        const double bound = orders.substrateReflectedPower * (1.0 - std::exp (-loss));
        check (std::abs (orders.reliefScatterBound - bound) < 1.0e-5 * bound,
               "the bound at " + describe (wavelength) + " nm is R (1 - exp(-(4 pi sigma / lambda)^2 cos i)): "
                   + describe (orders.reliefScatterBound) + " against " + describe (bound));
        check (std::abs (orders.reliefReach - wavelength / 400.0) < 1.0e-5, "the arm's reach is lambda x fmax at " + describe (wavelength));
        check (orders.reliefLeavingPower > 0.0 && orders.reliefLeavingPower <= orders.reliefScatterBound, "what leaves is within the bound");
        const auto bare = evaluated (*admitted (reliefMaterial (0.0f, false)), wavelength, wo);
        check (std::abs (orders.resolvedReflectedPower + orders.unresolvedReflectedPower - orders.substrateReflectedPower) < 1.0e-6,
               "resolved and unresolved make the metal's reflectance at " + describe (wavelength));
        check (std::abs (orders.resolvedReflectedPower / bare.resolvedReflectedPower
                         - std::exp (-loss)) < 1.0e-5,
               "the orders keep exp(-x) at " + describe (wavelength));
        const auto tripled = evaluated (*triple, wavelength, wo);
        check (tripled.grooveFamilyCount == 3 && tripled.reliefLeavingPower <= tripled.substrateReflectedPower
                   && tripled.unresolvedReflectedPower >= tripled.reliefScatterBound,
               "three families stay inside the metal's reflectance at " + describe (wavelength));
    }
    // The band ends at g = lambda / spacing, so longer wavelengths reach farther.
    const auto density = [&] (double wavelength, double offset)
    {
        const auto orders = evaluated (*one, wavelength, wo);
        const double x = -wo[0] - offset, y = -wo[1];
        const double z = std::sqrt (std::max (0.0, 1.0 - x * x - y * y));
        return reliefValue (orders, wo, Direction { x, y, z });
    };
    check (density (450.0, 0.9 * 450.0 / 400.0) > 0.0 && density (450.0, 1.01 * 450.0 / 400.0) == 0.0, "450 nm stops at g = lambda / spacing");
    check (density (400.0, 0.9) > 0.0 && density (400.0, 1.01) == 0.0 && density (500.0, 1.01) > 0.0,
           "a place 400 nm can't reach is lit by 500 nm");
}

// The first-order BRDF is reciprocal, and the bounded split scales it by (1 - exp(-x)) / x of
// the incident side, so f(wo, wi) / f(wi, wo) is that factor's ratio. The limit is the float
// error, sample by sample. Both ways compute the same offset o from the same inputs, each
// component within e = 8 u by rounding (u = 2^-24). The Gaussian's exponent is |o|^2 / 2 s^2,
// so its absolute error, the value's relative error, is |o| e / s^2. The rest of the value is
// a few products, and 1 - exp(-x) loses u / (1 - exp(-x)): 64 u covers x down to 0.15 here.
void reciprocityTests()
{
    const float height = 10.0f;
    const auto material = admitted (reliefMaterial (height, true));
    if (material == nullptr)
        return;
    const double unitRoundoff = std::ldexp (1.0, -24);
    const double deviation = 2.0 * material->description().roughness.rmsSlope;
    const double phase = 4.0 * 3.14159265358979323846 * height / 550.0;
    const auto factor = [&] (const Direction& incident)
    {
        const double x = 3.0 * phase * phase * incident[2];
        return (1.0 - std::exp (-x)) / x;
    };
    std::mt19937 random (7);
    std::uniform_real_distribution<double> uniform (-0.6, 0.6);
    int compared = 0, needFactor = 0;
    double largestError = 0.0, largestLimit = 0.0, largestShare = 0.0, largestSplit = 0.0;
    for (int i = 0; i < 200000; ++i)
    {
        const Direction wo = unit (uniform (random), uniform (random), 1.0);
        const Direction wi = unit (-wo[0] + 0.1 * uniform (random), -wo[1] + 0.02 * uniform (random), 1.0);
        const double f = reliefValue (evaluated (*material, 550.0, wo), wo, wi) / wi[2];
        const double b = reliefValue (evaluated (*material, 550.0, wi), wi, wo) / wo[2];
        if (! (f > 0.0 && b > 0.0))
            continue;
        ++compared;
        const double scale = factor (wo) / factor (wi);
        const double error = std::abs (f - b * scale) / std::max (f, b * scale);
        const double ox = wi[0] + wo[0], oy = wi[1] + wo[1];
        const double limit = std::sqrt (ox * ox + oy * oy) * 8.0 * unitRoundoff / (deviation * deviation) + 64.0 * unitRoundoff;
        largestError = std::max (largestError, error);
        largestLimit = std::max (largestLimit, limit);
        largestShare = std::max (largestShare, error / limit);
        largestSplit = std::max (largestSplit, std::abs (scale - 1.0));
        if (std::abs (scale - 1.0) > limit)
            ++needFactor;
        if (error > limit)
        {
            check (false, "the bounded BRDF's ratio is its factor's: off by " + describe (error) + ", limit " + describe (limit));
            return;
        }
    }
    check (compared > 100000, "reciprocity compared real values: " + std::to_string (compared));
    // Without the factor these pairs would fail, or this is only the first-order check again.
    check (needFactor > compared / 100, "the limit can see the factor on over 1% of pairs: " + std::to_string (needFactor));
    std::cout << "reciprocity over " << compared << " pairs: largest error " << largestError << " (" << largestError / unitRoundoff
              << " u), largest limit " << largestLimit << ", largest error / limit " << largestShare
              << ", largest split asymmetry " << largestSplit << ", pairs needing the factor " << needFactor << '\n';
}

// The split itself, stated without the renderer's arithmetic: x small gives first order,
// orders and scatter make the metal, and x large spends the orders without passing the metal.
void boundedSplitTests()
{
    const Direction wo = unit (-0.3, 0.1, 1.0);
    const double pi = 3.14159265358979323846;
    for (const float height : { 0.5f, 2.0f, 19.5f, 100.0f })
        for (const bool three : { false, true })
            for (const double wavelength : { 380.0, 540.0, 700.0 })
            {
                const auto material = admitted (reliefMaterial (height, three));
                if (material == nullptr)
                    return;
                const auto orders = evaluated (*material, wavelength, wo);
                const auto bare = evaluated (*admitted (reliefMaterial (0.0f, three)), wavelength, wo);
                const double metal = orders.substrateReflectedPower;
                const double phase = 4.0 * pi * height / wavelength;
                const double x = (three ? 3.0 : 1.0) * phase * phase * wo[2];
                const std::string where = describe (height) + " nm at " + describe (wavelength);
                check (std::abs (orders.reliefScatterBound - metal * (1.0 - std::exp (-x))) < 1.0e-5 * metal, "scatter takes 1 - exp(-x), " + where);
                check (orders.reliefScatterBound <= metal, "scatter stays inside the metal, " + where);
                check (std::abs (orders.resolvedReflectedPower - bare.resolvedReflectedPower * std::exp (-x)) < 1.0e-5 * metal, "the orders keep exp(-x), " + where);
                check (orders.unresolvedReflectedPower >= orders.reliefScatterBound - 1.0e-5 * metal, "the band fits the unresolved power, " + where);
                if (x < 0.01)
                    check (std::abs (orders.reliefScatterBound - metal * x) < 0.01 * metal * x, "small x is first order, " + where);
                else if (x > 0.5)
                    check (orders.reliefScatterBound < 0.9 * metal * x, "large x is under first order, " + where);
            }
}

// The sampler's histogram against the density, cell by cell over the tangent plane; the
// accepted share against the closed form; and the density against a uniform hemisphere that
// knows nothing of the lobe.
void densityTests()
{
    // The bounded form's far end too: the orb's height at 380 nm, x near 1.
    struct Case { bool three; float height; double wavelength; };
    for (const Case c : { Case { false, 10.0f, 550.0 }, Case { true, 10.0f, 550.0 }, Case { true, 19.5f, 380.0 } })
    {
        const bool three = c.three;
        auto description = reliefMaterial (c.height, three);
        description.roughness.rmsSlope = 0.05f;
        const auto material = admitted (description);
        if (material == nullptr)
            return;
        const Direction wo = unit (-0.4, 0.2, 1.0);
        const auto orders = evaluated (*material, c.wavelength, wo);
        std::mt19937_64 random (three ? 11u : 5u);
        std::uniform_real_distribution<float> uniform (0.0f, 1.0f);
        constexpr int cells = 24;
        const double cell = 2.0 / cells;
        std::vector<double> histogram (cells * cells, 0.0);
        const std::uint64_t draws = 8000000;
        std::uint64_t accepted = 0;
        for (std::uint64_t i = 0; i < draws; ++i)
        {
            Single wi {};
            if (! dm::sampleReliefScatter (orders, kNormal, single (wo), uniform (random), uniform (random), uniform (random), wi))
                continue;
            ++accepted;
            const int x = std::min (cells - 1, static_cast<int> ((wi[0] + 1.0) / cell));
            const int y = std::min (cells - 1, static_cast<int> ((wi[1] + 1.0) / cell));
            histogram[static_cast<std::size_t> (y * cells + x)] += 1.0;
        }
        const double share = static_cast<double> (accepted) / static_cast<double> (draws);
        const double closed = orders.reliefLeavingPower / orders.reliefScatterBound;
        check (std::abs (share - closed) < 2.0e-3,
               "the accepted share is the closed form: " + describe (share) + " against " + describe (closed));

        double worst = 0.0;
        for (int y = 0; y < cells; ++y)
            for (int x = 0; x < cells; ++x)
            {
                double expected = 0.0;
                constexpr int fine = 16;
                for (int v = 0; v < fine; ++v)
                    for (int u = 0; u < fine; ++u)
                    {
                        const double px = -1.0 + (x + (u + 0.5) / fine) * cell;
                        const double py = -1.0 + (y + (v + 0.5) / fine) * cell;
                        const double z2 = 1.0 - px * px - py * py;
                        if (! (z2 > 0.0))
                            continue;
                        // Tangent-plane area is cos wi per solid angle.
                        const double z = std::sqrt (z2);
                        expected += reliefValue (orders, wo, Direction { px, py, z }) / z * (cell / fine) * (cell / fine);
                    }
                expected /= orders.reliefScatterBound;
                const double measured = histogram[static_cast<std::size_t> (y * cells + x)] / static_cast<double> (draws);
                if (expected > 2.0e-3)
                    worst = std::max (worst, std::abs (measured - expected) / expected);
            }
        check (worst < 0.03, "the histogram follows the density, worst cell off by " + describe (worst));

        // Importance sampling against none: the same smooth sky seen by the lobe, once by
        // its own sampler and once by directions uniform on the hemisphere.
        const auto sky = [] (const Direction& d) { return 1.0 + d[0] + 0.5 * d[1] * d[1]; };
        double guided = 0.0;
        for (std::uint64_t i = 0; i < 2000000; ++i)
        {
            Single wi {};
            if (dm::sampleReliefScatter (orders, kNormal, single (wo), uniform (random), uniform (random), uniform (random), wi))
                guided += sky (Direction { wi[0], wi[1], wi[2] });
        }
        guided *= orders.reliefScatterBound / 2.0e6;
        double unguided = 0.0;
        double power = 0.0;
        const std::uint64_t blind = 40000000;
        const double twoPi = 6.283185307179586;
        for (std::uint64_t i = 0; i < blind; ++i)
        {
            const double z = 1.0 - static_cast<double> (uniform (random));
            const double r = std::sqrt (std::max (0.0, 1.0 - z * z));
            const double turn = twoPi * static_cast<double> (uniform (random));
            const Direction wi { r * std::cos (turn), r * std::sin (turn), z };
            const double value = reliefValue (orders, wo, wi) * twoPi;
            unguided += value * sky (wi);
            power += value;
        }
        unguided /= static_cast<double> (blind);
        power /= static_cast<double> (blind);
        check (std::abs (guided - unguided) < 0.01 * unguided,
               "the sampled lobe matches an unguided reference: " + describe (guided) + " against " + describe (unguided));
        check (std::abs (power - orders.reliefLeavingPower) < 0.01 * orders.reliefLeavingPower,
               "the density integrates to the power that leaves: " + describe (power) + " against " + describe (orders.reliefLeavingPower));
    }
}

void appendFace (pt::RenderScene& scene, Vec3 centre, Vec3 u, Vec3 v, std::uint32_t material)
{
    const Vec3 normal = pt::normalizeOrZero (pt::cross (u, v));
    const Vec3 tangent = pt::normalizeOrZero (u);
    const Vec3 corners[4] = { centre - u - v, centre + u - v, centre + u + v, centre - u + v };
    const pt::Vec2 uvs[4] = { { 0.0f, 0.0f }, { 1.0f, 0.0f }, { 1.0f, 1.0f }, { 0.0f, 1.0f } };
    const std::uint32_t indices[6] = { 0, 1, 2, 0, 2, 3 };
    for (int t = 0; t < 2; ++t)
    {
        pt::TriangleAttributes attributes;
        attributes.materialIndex = material;
        attributes.primitiveIndex = static_cast<std::uint32_t> (scene.geometry.triangles.size());
        attributes.geometricNormal = normal;
        for (int k = 0; k < 3; ++k)
        {
            const auto slot = static_cast<std::size_t> (k);
            const int corner = static_cast<int> (indices[t * 3 + k]);
            scene.geometry.positions.push_back (corners[corner]);
            scene.geometry.bounds.grow (corners[corner]);
            attributes.shadingNormals[slot] = normal;
            attributes.uvs[slot] = uvs[corner];
            attributes.tangents[slot] = { tangent.x, tangent.y, tangent.z, 1.0f };
        }
        scene.geometry.triangles.push_back (attributes);
    }
}

pt::Material diffuse (float albedo)
{
    pt::Material m;
    m.baseColor = { albedo, albedo, albedo };
    return m;
}

pt::Material lamp (float radiance)
{
    pt::Material m;
    m.baseColor = { 0.0f, 0.0f, 0.0f };
    m.emission = { radiance, radiance, radiance };
    return m;
}

pt::PinholeCamera cameraAt (Vec3 position, float tanHalfFov, std::uint32_t size)
{
    pt::PinholeCamera camera;
    camera.position = position;
    camera.tanHalfFov = tanHalfFov;
    camera.aspect = 1.0f;
    camera.nearPlane = 0.01f;
    camera.farPlane = 1000.0f;
    camera.width = size;
    camera.height = size;
    return camera;
}

double meanRadiance (const pt::RenderScene& scene, const pt::PinholeCamera& camera, const pt::IntegratorSettings& settings,
                     std::uint32_t samples, std::uint64_t generation)
{
    pt::PathTracer tracer (4);
    tracer.settings = settings;
    const pt::CancellationToken cancel {};
    const auto stats = tracer.renderFixed (scene, camera, camera.width, camera.height, samples, generation, cancel);
    check (stats.nonFiniteSamples == 0, "the render's samples are finite");
    double sum = 0.0;
    for (std::uint32_t y = 0; y < camera.height; ++y)
        for (std::uint32_t x = 0; x < camera.width; ++x)
            sum += tracer.accumulator().mean (x, y).g;
    return sum / static_cast<double> (camera.width * camera.height);
}

grid::SceneLightRecord environmentRecord (float radiance)
{
    grid::SceneLightRecord light;
    light.kind = grid::SceneLightKind::Environment;
    light.color = { radiance, radiance, radiance };
    light.intensity = 1.0f;
    return light;
}

// Open: every path leaves to a uniform sky, so a pixel holds the sky times the metal.
// Closed: a sealed box of the same grating, every wall glowing, so a pixel holds E / (1 - R).
// Relief that made or lost light would show in either, and a bounce multiplies it.
void furnaceTests()
{
    for (const float height : { 0.0f, 19.5f, 100.0f })
    {
        const auto material = admitted (reliefMaterial (height, true));
        if (material == nullptr)
            return;
        const double metal = evaluated (*material, pt::kDLineNm, Direction { 0.0, 0.0, 1.0 }).substrateReflectedPower;
        pt::IntegratorSettings settings;
        settings.spectral = pt::SpectralMode::Off;
        settings.causticLightPaths = false;

        pt::RenderScene open;
        open.materials = { diffuse (0.0f) };
        appendFace (open, { 0.0f, 0.0f, 0.0f }, { 4.0f, 0.0f, 0.0f }, { 0.0f, 4.0f, 0.0f }, 0);
        open.gratingMaterials = { material };
        open.gratings = { { 0u, material->description() } };
        open.lightRecords = { environmentRecord (0.5f) };
        open.commit();
        settings.maxDepth = 4;
        const double sky = meanRadiance (open, cameraAt ({ 0.0f, 0.0f, 2.0f }, 0.6f, 4), settings, 16384, 3);
        check (std::abs (sky - 0.5 * metal) < 0.005 * 0.5 * metal,
               "open furnace at relief " + describe (height) + " nm: " + describe (sky) + " against " + describe (0.5 * metal));

        pt::RenderScene closed;
        const float glow = 0.1f;
        closed.materials = { lamp (glow) };
        appendFace (closed, { 0.0f, 0.0f, -1.0f }, { 1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, 0);
        appendFace (closed, { 0.0f, 0.0f, 1.0f }, { 0.0f, 1.0f, 0.0f }, { 1.0f, 0.0f, 0.0f }, 0);
        appendFace (closed, { -1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f }, 0);
        appendFace (closed, { 1.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 1.0f }, { 0.0f, 1.0f, 0.0f }, 0);
        appendFace (closed, { 0.0f, -1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f }, { 1.0f, 0.0f, 0.0f }, 0);
        appendFace (closed, { 0.0f, 1.0f, 0.0f }, { 1.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 1.0f }, 0);
        closed.gratingMaterials = { material };
        closed.gratings = { { 0u, material->description() } };
        closed.commit();
        settings.maxDepth = 256;
        const double box = meanRadiance (closed, cameraAt ({ 0.0f, 0.0f, 0.5f }, 0.5f, 4), settings, 16384, 5);
        const double expected = glow / (1.0 - metal);
        check (std::abs (box - expected) < 0.02 * expected,
               "closed furnace at relief " + describe (height) + " nm: " + describe (box) + " against " + describe (expected));
        std::cout << "furnaces at relief " << height << " nm: open " << sky / (0.5 * metal) << ", closed " << box / expected << " of the exact value\n";
    }
}

// Light sampling and none on the same lamp: the mean can't depend on which found it.
void lightSamplingTests()
{
    const auto material = admitted (reliefMaterial (18.0f, true));
    if (material == nullptr)
        return;
    pt::RenderScene scene;
    scene.materials = { diffuse (0.0f), lamp (200.0f) };
    appendFace (scene, { 0.0f, 0.0f, 0.0f }, { 1.5f, 0.0f, 0.0f }, { 0.0f, 1.5f, 0.0f }, 0);
    appendFace (scene, { 1.0f, 0.3f, 3.0f }, { 0.0f, 0.2f, 0.0f }, { 0.2f, 0.0f, 0.0f }, 1);
    scene.gratingMaterials = { material };
    scene.gratings = { { 0u, material->description() } };
    scene.commit();
    const pt::PinholeCamera camera = cameraAt ({ 0.0f, 0.0f, 2.0f }, 0.5f, 16);
    pt::IntegratorSettings settings;
    settings.maxDepth = 3;
    settings.causticLightPaths = false;
    settings.spectral = pt::SpectralMode::Off;
    settings.strategy = pt::LightStrategy::MultipleImportance;
    const double weighted = meanRadiance (scene, camera, settings, 8192, 41);
    settings.strategy = pt::LightStrategy::BsdfOnly;
    const double blind = meanRadiance (scene, camera, settings, 32768, 42);
    check (weighted > 0.0 && std::abs (weighted - blind) < 0.03 * blind,
           "the light sample keeps the lobe's mean: " + describe (weighted) + " against " + describe (blind));
    std::cout << "lamp seen by the lobe: light sampling " << weighted << ", none " << blind << '\n';
}

// A plate the lamp can't see directly, away from the mirror spot and the orders. All it gets
// is the grating's relief scatter: the lobe's caustic, through the light tracer, once.
void plateTests()
{
    // A narrow lobe, so order 1's Gaussian tail stays off the patch. The relief's height is
    // what's tested here, and the closed form takes the width as it is.
    const auto narrow = [] (float height)
    {
        auto description = reliefMaterial (height, false);
        description.roughness.rmsSlope = 0.004f;
        return description;
    };
    const auto material = admitted (narrow (18.0f));
    const auto smooth = admitted (narrow (0.0f));
    if (material == nullptr || smooth == nullptr)
        return;
    const float albedo = 0.8f;
    const float radiance = 20000.0f;
    const float lampHalf = 0.05f;
    const float gratingHalf = 0.3f;
    const Vec3 lampCentre { -2.0f, 0.0f, -3.0f };
    const Vec3 gratingCentre { 0.0f, 0.0f, 3.0f };
    // Clear of the mirror's image, x up to 1.5, and order 1's, from about 2.2.
    const Vec3 spot { 1.85f, 0.0f, 0.0f };
    const float spotHalf = 0.25f;
    const auto build = [&] (const std::shared_ptr<const dm::AdmittedDiffractionMaterialIR>& bound)
    {
        pt::RenderScene scene;
        scene.materials = { diffuse (0.0f), diffuse (albedo), lamp (radiance) };
        // Facing down, u along +x as the grooves' axis.
        appendFace (scene, gratingCentre, { gratingHalf, 0.0f, 0.0f }, { 0.0f, -gratingHalf, 0.0f }, 0);
        appendFace (scene, { 2.5f, 0.0f, 0.0f }, { 2.0f, 0.0f, 0.0f }, { 0.0f, 2.0f, 0.0f }, 1);
        appendFace (scene, lampCentre, { lampHalf, 0.0f, 0.0f }, { 0.0f, lampHalf, 0.0f }, 2);
        scene.gratingMaterials = { bound };
        scene.gratings = { { 0u, bound->description() } };
        scene.commit();
        return scene;
    };
    const pt::RenderScene scene = build (material);
    const pt::RenderScene bare = build (smooth);

    // The closed form: the lamp's power on each grating element, the lobe's density toward
    // the spot, the inverse square to the plate, and a diffuse plate's albedo / pi.
    const double pi = 3.14159265358979323846;
    double expected = 0.0;
    constexpr int steps = 32;
    constexpr int spots = 8;
    const double step = 2.0 * gratingHalf / steps;
    const double spotStep = 2.0 * spotHalf / spots;
    for (int j = 0; j < steps; ++j)
        for (int i = 0; i < steps; ++i)
        {
            const double x = -gratingHalf + (i + 0.5) * step;
            const double y = -gratingHalf + (j + 0.5) * step;
            const Direction toLamp { lampCentre.x - x, lampCentre.y - y, lampCentre.z - gratingCentre.z };
            const double r2 = toLamp[0] * toLamp[0] + toLamp[1] * toLamp[1] + toLamp[2] * toLamp[2];
            const Direction wo = unit (toLamp[0], toLamp[1], -toLamp[2]);
            const auto orders = evaluated (*material, pt::kDLineNm, wo);
            const double arriving = radiance * (2.0 * lampHalf) * (2.0 * lampHalf) * wo[2] / r2 * wo[2];
            for (int v = 0; v < spots; ++v)
                for (int u = 0; u < spots; ++u)
                {
                    const Direction toSpot { spot.x - spotHalf + (u + 0.5) * spotStep - x,
                                             spot.y - spotHalf + (v + 0.5) * spotStep - y, spot.z - gratingCentre.z };
                    const double d2 = toSpot[0] * toSpot[0] + toSpot[1] * toSpot[1] + toSpot[2] * toSpot[2];
                    const Direction wi = unit (toSpot[0], toSpot[1], -toSpot[2]);
                    expected += arriving * step * step * reliefValue (orders, wo, wi) * wi[2] / d2 / (spots * spots);
                }
        }
    expected *= albedo / pi;

    const pt::PinholeCamera camera = cameraAt ({ spot.x, spot.y, 1.0f }, spotHalf, 16);
    pt::IntegratorSettings settings;
    settings.maxDepth = 4;
    settings.spectral = pt::SpectralMode::Off;
    settings.causticLightPaths = true;
    const double lit = meanRadiance (scene, camera, settings, 65536, 51);
    const double dark = meanRadiance (bare, camera, settings, 16384, 52);
    check (std::abs (lit - dark - expected) < 0.1 * expected,
           "the plate holds the lobe's caustic once: " + describe (lit - dark) + " against " + describe (expected));
    check (dark < 0.05 * expected, "without relief nothing lands there: " + describe (dark));
    std::cout << "plate: with relief " << lit << ", without " << dark << ", closed form " << expected << '\n';
}

// Plumbing: a relief needs a width, and no relief leaves every digest as it was.
void admissionTests()
{
    std::string error;
    auto widthless = reliefMaterial (10.0f, false);
    widthless.roughness.rmsSlope = 0.0f;
    check (! dm::admit (widthless, error).has_value(), "relief with no slope is refused");
    auto tooHigh = reliefMaterial (dm::kMaximumReliefRmsHeightNanometres * 2.0f, false);
    check (! dm::admit (tooHigh, error).has_value(), "relief past its range is refused");

    const auto none = admitted (reliefMaterial (0.0f, false));
    auto spaced = reliefMaterial (0.0f, false);
    spaced.roughness.reliefMinimumSpacingNanometres = 800.0f;
    const auto noneSpaced = admitted (spaced);
    const auto some = admitted (reliefMaterial (10.0f, false));
    if (none == nullptr || noneSpaced == nullptr || some == nullptr)
        return;
    check (none->structuralDigest() == noneSpaced->structuralDigest(), "a spacing with no height changes nothing");
    check (none->structuralDigest() != some->structuralDigest(), "a relief height changes the digest");

    // The wire carries the relief, and leaves a material without it byte for byte as it was.
    namespace wire = visualimportedscenerender::detail;
    const auto encoded = [] (const dm::Description& description)
    {
        diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest request;
        request.material = description;
        std::ostringstream stream;
        wire::encodeDiffractionMaterial (stream, request);
        return stream.str();
    };
    const auto withRelief = reliefMaterial (12.5f, false);
    std::istringstream input (encoded (withRelief));
    diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest decoded;
    // A bare request fails the whole binding's check at the end, after the material is read.
    wire::decodeDiffractionMaterial (input, decoded);
    check (decoded.material.roughness.reliefRmsHeightNanometres == 12.5f
               && decoded.material.roughness.reliefMinimumSpacingNanometres == 400.0f,
           "the wire keeps the relief's two values");
    check (decoded.material.spectrum.lastOrder == withRelief.spectrum.lastOrder
               && decoded.material.spectrum.wavelengthCount == withRelief.spectrum.wavelengthCount,
           "the fields after the relief still line up");
    check (encoded (reliefMaterial (0.0f, false)).find (" r ") == std::string::npos, "a material with no relief writes no token");
}

} // namespace

int main()
{
    energyAndPositionTests();
    reciprocityTests();
    boundedSplitTests();
    densityTests();
    admissionTests();
    furnaceTests();
    lightSamplingTests();
    plateTests();
    std::cout << (checks - failures) << "/" << checks << " relief scatter checks passed\n";
    return failures == 0 ? 0 : 1;
}
