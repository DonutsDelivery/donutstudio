// Tests for the single-source path integrator (Plan 3, Tier V) on the CPU
// device: the per-sample kernel driven through the device interface matches
// PathTracer, the termination contract, and diffraction-grating vertices.

#include "pathtrace/devices/device_cpu.h"
#include "diffraction_order_bsdf.h"
#include "DiffractionMaterialPresets.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace
{

namespace pt = videohelper::pathtrace;
namespace device = videohelper::pathtrace::device;
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

// Flat-shaded quad facing +z (counter-clockwise seen from +z), with an
// authored tangent along +x.
void appendQuad (pt::RenderScene& scene, Vec3 centre, float half, std::uint32_t material)
{
    const Vec3 p[4] = { { centre.x - half, centre.y - half, centre.z }, { centre.x + half, centre.y - half, centre.z },
                        { centre.x + half, centre.y + half, centre.z }, { centre.x - half, centre.y + half, centre.z } };
    const std::uint32_t indices[6] = { 0, 1, 2, 0, 2, 3 };
    for (int t = 0; t < 2; ++t)
    {
        pt::TriangleAttributes attributes;
        attributes.materialIndex = material;
        attributes.primitiveIndex = static_cast<std::uint32_t> (t);
        attributes.geometricNormal = { 0.0f, 0.0f, 1.0f };
        for (int k = 0; k < 3; ++k)
        {
            const Vec3 v = p[indices[t * 3 + k]];
            scene.geometry.positions.push_back (v);
            attributes.shadingNormals[static_cast<std::size_t> (k)] = attributes.geometricNormal;
            attributes.uvs[static_cast<std::size_t> (k)] = { (v.x - centre.x + half) / (2.0f * half),
                                                             (v.y - centre.y + half) / (2.0f * half) };
            attributes.tangents[static_cast<std::size_t> (k)] = { 1.0f, 0.0f, 0.0f, 1.0f };
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

pt::PinholeCamera makeCamera (Vec3 position, float tanHalfFov, std::uint32_t width, std::uint32_t height)
{
    pt::PinholeCamera camera;
    camera.position = position;
    camera.tanHalfFov = tanHalfFov;
    camera.aspect = static_cast<float> (width) / static_cast<float> (height);
    camera.nearPlane = 0.01f;
    camera.farPlane = 1000.0f;
    camera.width = width;
    camera.height = height;
    return camera;
}

// Renders every pixel through device::cpu::renderSample, as a GPU launch
// would: one kernel call per (pixel, sample).
// The film can't move, so the constructor renders in place.
struct DeviceRender
{
    DeviceRender (const device::cpu::SceneView& view, const pt::IntegratorSettings& settings,
                  const pt::PinholeCamera& camera, std::uint32_t samples, std::uint64_t generation)
    {
        film.resize (camera.width, camera.height);
        film.reset (generation);
        pt::PathCounters counters;
        for (std::uint32_t y = 0; y < camera.height; ++y)
            for (std::uint32_t x = 0; x < camera.width; ++x)
                for (std::uint32_t s = 0; s < samples; ++s)
                {
                    if (device::cpu::renderSample (view, settings, camera, x, y, camera.width, s, generation, film,
                                                   counters)
                        == device::SampleOutcome::Added)
                        ++added;
                    else
                        ++nonFinite;
                }
    }

    pt::Accumulator film;
    std::uint64_t added = 0;
    std::uint64_t nonFinite = 0;
};

bool sameImage (const pt::Accumulator& a, const pt::Accumulator& b)
{
    const std::vector<float> left = a.resolveRgba32f();
    const std::vector<float> right = b.resolveRgba32f();
    return left.size() == right.size() && std::memcmp (left.data(), right.data(), left.size() * sizeof (float)) == 0;
}

// ---------------------------------------------------------------- parity

void parityTest (const std::string& name, const pt::Material& material, pt::SpectralMode spectral, bool expectSpectral)
{
    pt::RenderScene scene;
    scene.materials.push_back (material);
    appendQuad (scene, { 0.0f, 0.0f, 0.0f }, 1.0f, 0);
    pt::Material lamp;
    lamp.emission = { 4.0f, 3.0f, 2.0f };
    scene.materials.push_back (lamp);
    appendQuad (scene, { 0.3f, 0.2f, -0.5f }, 0.6f, 1);
    scene.lightRecords.push_back (environmentRecord (0.25f));
    scene.commit();

    pt::IntegratorSettings settings;
    settings.spectral = spectral;
    const pt::PinholeCamera camera = makeCamera ({ 0.1f, -0.2f, 3.0f }, 0.5f, 12, 10);
    const device::cpu::SceneView view = device::cpu::makeSceneView (scene);
    check (device::cpu::usesSpectralTransport (view, settings) == expectSpectral
               && pt::usesSpectralTransport (scene, settings) == expectSpectral,
           name + ": the view and the scene agree on spectral transport");

    pt::PathTracer tracer (1);
    tracer.settings = settings;
    const pt::CancellationToken cancel {};
    const pt::RenderStats stats = tracer.renderFixed (scene, camera, camera.width, camera.height, 8, 5, cancel);
    const DeviceRender render (view, settings, camera, 8, 5);
    check (! stats.invalidRequest && stats.samplesAdded == render.added && stats.nonFiniteSamples == render.nonFinite,
           name + ": the device kernel adds the same samples as PathTracer");
    check (sameImage (tracer.accumulator(), render.film), name + ": the device kernel image equals PathTracer bit for bit");
    check (render.added == 12ull * 10ull * 8ull, name + ": every sample is finite");

    // tracePath (the public API) and the device entry are one function.
    const pt::PathSampler sampler (3, 1, 5, settings.pattern);
    const pt::Ray ray = camera.generateRay (3.25f, 4.5f);
    pt::PathCounters a, b;
    Vec3 left, right;
    const bool okLeft = pt::tracePath (scene, settings, ray, sampler, a, left);
    const bool okRight = device::cpu::tracePath (view, settings, ray, sampler, b, right);
    check (okLeft == okRight && left.x == right.x && left.y == right.y && left.z == right.z
               && a.closestRays == b.closestRays && a.shadowRays == b.shadowRays,
           name + ": tracePath and device::cpu::tracePath return the same estimate");
}

void parityTests()
{
    pt::Material diffuse;
    diffuse.baseColor = { 0.8f, 0.6f, 0.4f };
    diffuse.roughness = 1.0f;
    parityTest ("diffuse, RGB", diffuse, pt::SpectralMode::Automatic, false);
    parityTest ("diffuse, spectral forced", diffuse, pt::SpectralMode::On, true);

    pt::Material glass;
    glass.transmission = 1.0f;
    glass.roughness = 0.0f;
    glass.ior = 1.5f;
    glass.abbe = 30.0f;
    parityTest ("dispersive glass sheet", glass, pt::SpectralMode::Automatic, true);
}

// ---------------------------------------------------------------- termination

void terminationTests()
{
    pt::RenderScene scene;
    pt::Material grey;
    grey.baseColor = { 0.5f, 0.5f, 0.5f };
    grey.roughness = 1.0f;
    scene.materials.push_back (grey);
    pt::Material broken;
    broken.emission = { std::numeric_limits<float>::quiet_NaN(), 1.0f, 1.0f };
    scene.materials.push_back (broken);
    appendQuad (scene, { -1.0f, 0.0f, 0.0f }, 1.0f, 0);  // left half of the image
    appendQuad (scene, { 1.0f, 0.0f, 0.0f }, 1.0f, 1);   // right half: NaN emission
    scene.lightRecords.push_back (environmentRecord (0.5f));
    scene.commit();
    const device::cpu::SceneView view = device::cpu::makeSceneView (scene);
    const pt::PinholeCamera camera = makeCamera ({ 0.0f, 0.0f, 1.0f }, 0.9f, 8, 4);

    // Bounce limit 0: the camera hit ends the path; it keeps what it gathered
    // (nothing on the grey quad) and is still a finite sample.
    pt::IntegratorSettings settings;
    settings.maxDepth = 0;
    const DeviceRender limited (view, settings, camera, 4, 1);
    const pt::Rgb left = limited.film.mean (0, 1);
    check (limited.film.sampleCount (0, 1) == 4 && left.r == 0.0f && left.g == 0.0f && left.b == 0.0f,
           "maxDepth 0 ends a path at the camera hit with its gathered radiance (none here)");

    // Non-finite pixels drop their samples; the rest of the frame renders.
    settings.maxDepth = 4;
    const DeviceRender render (view, settings, camera, 4, 1);
    check (render.nonFinite == 4u * 4u * 4u, "every sample on the NaN emitter is dropped and counted");
    check (render.added == 4u * 4u * 4u, "every sample on the grey quad still renders");
    bool leftLit = true;
    bool rightEmpty = true;
    for (std::uint32_t y = 0; y < 4; ++y)
        for (std::uint32_t x = 0; x < 8; ++x)
        {
            const pt::Rgb v = render.film.mean (x, y);
            if (x < 4)
                leftLit = leftLit && render.film.sampleCount (x, y) == 4 && v.r > 0.0f && std::isfinite (v.r);
            else
                rightEmpty = rightEmpty && render.film.sampleCount (x, y) == 0;
        }
    check (leftLit, "grey pixels next to the NaN emitter are lit by the environment");
    check (rightEmpty, "NaN pixels hold no samples rather than failing the frame");
}

// ---------------------------------------------------------------- gratings

void gratingTests()
{
    std::string error;
    const auto admitted = diffractionmaterial::admit (diffractionmaterial::makeAluminiumBinaryGratingPreset(), error);
    check (admitted.has_value(), "the aluminium binary grating preset admits: " + error);
    if (! admitted)
        return;

    const float environment = 0.5f;
    pt::RenderScene scene;
    scene.materials.push_back (pt::Material {});
    appendQuad (scene, { 0.0f, 0.0f, 0.0f }, 1.0f, 0);
    scene.lightRecords.push_back (environmentRecord (environment));
    scene.commit();
    const std::array<device::GratingBinding, 1> bindings { { { 0u, admitted->description() } } };
    const device::cpu::SceneView view = device::cpu::makeSceneView (
        scene, { bindings.data(), static_cast<std::uint32_t> (bindings.size()) });

    pt::IntegratorSettings settings;
    check (device::cpu::usesSpectralTransport (view, settings) && ! pt::usesSpectralTransport (scene, settings),
           "a bound grating makes Automatic transport spectral");

    // RGB transport evaluates the orders at the d line. Looking straight down, every
    // order and the unresolved remainder escape to the uniform environment, so a pixel
    // holds the environment times the metal's reflectance, no more and no less.
    diffractionmaterial::DiffractionLocalFrame frame;
    diffractionmaterial::DiffractionOrderDistribution orders;
    const bool evaluated = diffractionmaterial::evaluateDiffractionOrders (*admitted, pt::kDLineNm, frame, { 0.0, 0.0, 1.0 },
                                                                          { 0.5, 0.5 }, orders, error);
    check (evaluated && orders.resolvedReflectedPower > 0.0, "the grating reflects at normal incidence: " + error);
    const auto expected = static_cast<float> (environment * orders.substrateReflectedPower);

    settings.spectral = pt::SpectralMode::Off;
    const pt::PinholeCamera camera = makeCamera ({ 0.0f, 0.0f, 3.0f }, 0.002f, 1, 1);
    const DeviceRender render (view, settings, camera, 65536, 7);
    const pt::Rgb mean = render.film.mean (0, 0);
    check (render.added == 65536 && render.nonFinite == 0, "grating samples are finite");
    check (std::fabs (mean.r - expected) <= 1.0e-3f * expected && std::fabs (mean.g - expected) <= 1.0e-3f * expected,
           "grating pixel " + std::to_string (mean.r) + " equals environment x metal " + std::to_string (expected));

    // The check above has to see a gain. 3% in the metal lobe, the unresolved remainder and
    // order 0, adds this much to the pixel. A metal that much stronger stands in for it,
    // with no seam in the kernel, and must land outside the tolerance.
    double lobePower = orders.unresolvedReflectedPower;
    for (std::size_t i = 0; i < orders.orderCount; ++i)
        if (orders.orders[i].primary == 0)
            lobePower += orders.orders[i].reflectedPower;
    const double target = orders.substrateReflectedPower + 0.03 * lobePower;
    auto strongerMetal = diffractionmaterial::makeAluminiumBinaryGratingPreset();
    double low = strongerMetal.substrate.extinctionCoefficient, high = 4.0 * low;
    for (int step = 0; step < 40; ++step)
    {
        const double middle = 0.5 * (low + high);
        strongerMetal.substrate.extinctionCoefficient = static_cast<float> (middle);
        const auto trial = diffractionmaterial::admit (strongerMetal, error);
        diffractionmaterial::DiffractionOrderDistribution trialOrders;
        if (! trial || ! diffractionmaterial::evaluateDiffractionOrders (*trial, pt::kDLineNm, frame, { 0.0, 0.0, 1.0 }, { 0.5, 0.5 }, trialOrders, error))
            break;
        (trialOrders.substrateReflectedPower < target ? low : high) = middle;
    }
    const auto stronger = diffractionmaterial::admit (strongerMetal, error);
    check (stronger.has_value(), "the stronger metal admits: " + error);
    if (stronger)
    {
        const std::array<device::GratingBinding, 1> gainBindings { { { 0u, stronger->description() } } };
        const DeviceRender gained (device::cpu::makeSceneView (scene, { gainBindings.data(), 1u }), settings, camera, 65536, 7);
        const float gainedMean = gained.film.mean (0, 0).r;
        check (std::fabs (gainedMean - expected) > 1.0e-3f * expected,
               "a 3% gain in the metal lobe fails the furnace: " + std::to_string (gainedMean) + " against " + std::to_string (expected));
        check (std::fabs (gainedMean - static_cast<float> (environment * target)) <= 1.0e-3f * expected,
               "the stand-in renders the gain it stands for: " + std::to_string (gainedMean) + " against " + std::to_string (environment * target));
    }

    // Without the binding the same surface renders its Material lobes.
    const DeviceRender plain (device::cpu::makeSceneView (scene), settings, camera, 32, 7);
    check (std::fabs (plain.film.mean (0, 0).r - expected) > 1.0e-3f, "the binding, not the Material, drives the grating");

    // Spectral transport: the hero carries the orders on its own.
    settings.spectral = pt::SpectralMode::Automatic;
    const pt::PinholeCamera wide = makeCamera ({ 0.0f, 0.0f, 3.0f }, 0.2f, 4, 4);
    const DeviceRender spectral (view, settings, wide, 64, 9);
    double sum = 0.0;
    for (std::uint32_t y = 0; y < 4; ++y)
        for (std::uint32_t x = 0; x < 4; ++x)
            sum += spectral.film.mean (x, y).r + spectral.film.mean (x, y).g + spectral.film.mean (x, y).b;
    check (spectral.nonFinite == 0 && spectral.added == 4u * 4u * 64u, "spectral grating samples are finite");
    check (sum > 0.0 && std::isfinite (sum), "spectral grating pixels receive the diffracted environment");

    // A pressed disc in a uniform environment: order 0 with no width, the delta orders and
    // the pit scatter all see it, so a pixel holds the environment times the metal.
    auto pressed = diffractionmaterial::makeAluminiumBinaryGratingPreset();
    pressed.geometry.grooveSpacingNanometres = 1600.0f;
    pressed.microstructure.grooveDepthNanometres = 124.1f;
    pressed.microstructure.dutyCycle = 0.3125f;
    pressed.pitStatistics = diffractionmaterial::PitStatistics::PressedDisc;
    pressed.roughness.rmsSlope = 0.0f;
    pressed.spectrum.lastOrder = 3;
    const auto admittedPressed = diffractionmaterial::admit (pressed, error);
    check (admittedPressed.has_value(), "a pressed disc admits with no roughness: " + error);
    if (admittedPressed)
    {
        const std::array<device::GratingBinding, 1> pressedBindings { { { 0u, admittedPressed->description() } } };
        const device::cpu::SceneView pressedView = device::cpu::makeSceneView (
            scene, { pressedBindings.data(), static_cast<std::uint32_t> (pressedBindings.size()) });
        diffractionmaterial::DiffractionOrderDistribution pressedOrders;
        check (diffractionmaterial::evaluateDiffractionOrders (*admittedPressed, pt::kDLineNm, frame, { 0.0, 0.0, 1.0 },
                                                               { 0.5, 0.5 }, pressedOrders, error)
                   && pressedOrders.pitScatterPower > 0.0,
               "the pits scatter at the d line: " + error);
        settings.spectral = pt::SpectralMode::Off;
        const DeviceRender furnace (pressedView, settings, camera, 4096, 11);
        const auto metal = static_cast<float> (environment * pressedOrders.substrateReflectedPower);
        check (furnace.nonFinite == 0 && std::fabs (furnace.film.mean (0, 0).r - metal) <= 0.01f * metal,
               "pressed disc pixel " + std::to_string (furnace.film.mean (0, 0).r) + " equals environment x metal " + std::to_string (metal));
    }
}

} // namespace

int main()
{
    parityTests();
    terminationTests();
    gratingTests();
    std::cout << (checks - failures) << "/" << checks << " pathtrace device checks passed\n";
    return failures == 0 ? 0 : 1;
}
