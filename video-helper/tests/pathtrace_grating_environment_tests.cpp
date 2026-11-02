// The grooved disc's chain in the path tracer: a bound environment image, the
// grating it lights, and the coloured orders that land on a receiver.

#include "pathtrace/accumulator.h"
#include "pathtrace/bsdf.h"
#include "pathtrace/environment_map.h"
#include "pathtrace/integrator.h"
#include "pathtrace/lights.h"
#include "pathtrace/scene_ingest.h"
#include "diffraction_material_admission.h"
#include "DiffractionMaterialPresets.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
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
        std::cerr << "FAIL: " << message << "\n";
    }
}

pt::Material diffuse (Vec3 colour)
{
    pt::Material m;
    m.baseColor = colour;
    m.roughness = 1.0f;
    return m;
}

void appendTriangle (pt::RenderScene& scene, std::array<Vec3, 3> p, std::array<pt::Vec2, 3> uv, std::uint32_t material)
{
    const Vec3 n = pt::normalizeOrZero (pt::cross (p[1] - p[0], p[2] - p[0]));
    pt::TriangleAttributes attributes;
    attributes.materialIndex = material;
    attributes.primitiveIndex = static_cast<std::uint32_t> (scene.geometry.triangles.size());
    attributes.geometricNormal = n;
    attributes.shadingNormals = { n, n, n };
    attributes.uvs = uv;
    // The default every generated mesh carries.
    attributes.tangents = { { { 1.0f, 0.0f, 0.0f, 1.0f }, { 1.0f, 0.0f, 0.0f, 1.0f }, { 1.0f, 0.0f, 0.0f, 1.0f } } };
    for (const Vec3 v : p)
    {
        scene.geometry.positions.push_back (v);
        scene.geometry.bounds.grow (v);
    }
    scene.geometry.triangles.push_back (attributes);
}

// A rectangle in the plane z = height, facing the disc below it.
void appendReceiver (pt::RenderScene& scene, float x0, float x1, float y0, float y1, float height, std::uint32_t material)
{
    const Vec3 a { x0, y0, height }, b { x1, y0, height }, c { x1, y1, height }, d { x0, y1, height };
    appendTriangle (scene, { a, c, b }, { pt::Vec2 {}, pt::Vec2 {}, pt::Vec2 {} }, material);
    appendTriangle (scene, { a, d, c }, { pt::Vec2 {}, pt::Vec2 {}, pt::Vec2 {} }, material);
}

// A cap in the plane z = 0 facing +Z, with the polar uv of a generated cylinder
// cap: u runs around the hub, v from hub to rim, and the last column steps back
// to u = 0. With the reciprocal axis on v its grooves are concentric rings.
void appendPolarCap (pt::RenderScene& scene, float radius, std::uint32_t segments, std::uint32_t material)
{
    const auto rim = [radius, segments] (std::uint32_t k) {
        const float phi = 2.0f * pt::kPi * static_cast<float> (k) / static_cast<float> (segments);
        return Vec3 { radius * std::cos (phi), radius * std::sin (phi), 0.0f };
    };
    for (std::uint32_t k = 0; k < segments; ++k)
    {
        const std::uint32_t next = (k + 1) % segments;
        const float u0 = static_cast<float> (k) / static_cast<float> (segments);
        const float u1 = static_cast<float> (next) / static_cast<float> (segments);
        appendTriangle (scene, { Vec3 {}, rim (k), rim (next) }, { pt::Vec2 { 0.5f, 0.0f }, pt::Vec2 { u0, 1.0f }, pt::Vec2 { u1, 1.0f } },
                        material);
    }
}

// A 180 x 90 image, black but for a white lobe about a direction, in the
// convention of pathtrace_environment_tests: +Z at u = 0.5, row 0 at +Y.
pt::EnvironmentPixels lobePixels (Vec3 axis, float radiance, Vec3 tint, std::uint64_t version)
{
    constexpr std::uint32_t width = 180, height = 90;
    const float cosRadius = std::cos (4.0f * pt::kPi / 180.0f);
    std::vector<float> rgb;
    rgb.reserve (static_cast<std::size_t> (width) * height * 3);
    for (std::uint32_t y = 0; y < height; ++y)
        for (std::uint32_t x = 0; x < width; ++x)
        {
            const float longitude = ((static_cast<float> (x) + 0.5f) / width - 0.5f) * 2.0f * pt::kPi;
            const float polar = (static_cast<float> (y) + 0.5f) / height * pt::kPi;
            const Vec3 d { std::sin (polar) * std::sin (longitude), std::cos (polar), std::sin (polar) * std::cos (longitude) };
            const float value = pt::dot (d, axis) >= cosRadius ? radiance : 0.0f;
            rgb.insert (rgb.end(), { value * tint.x, value * tint.y, value * tint.z });
        }
    return pt::copyEnvironmentPixels (width, height, rgb.data(), { "pathtrace-grating-environment-test", version, std::string (64, 'a') });
}

grid::SceneLightRecord imageRecord (const pt::EnvironmentPixels& pixels)
{
    grid::SceneLightRecord light;
    light.kind = grid::SceneLightKind::Environment;
    light.color = { 1.0f, 1.0f, 1.0f };
    light.intensity = 1.0f;
    light.environmentImage.version = 1;
    light.environmentImage.asset = pixels.asset;
    return light;
}

constexpr float kReceiverHeight = 2.0f;
constexpr float kOpening = 0.7f;
constexpr float kReach = 2.7f;

// The module's grating on a disc, with no lights yet. With receivers, four face
// it, one per side, leaving an opening that lets the lobe straight overhead reach
// the disc and order 0 back out.
std::unique_ptr<pt::RenderScene> makeDiscGeometry (bool receivers, std::optional<float> rmsSlope = std::nullopt, float streakSlope = 0.0f)
{
    // The module's grating: 1400 nm aluminium with the reciprocal axis on v.
    auto description = diffractionmaterial::makeAluminiumBinaryGratingPreset();
    description.geometry.directionUv = { 0.0f, 1.0f };
    description.geometry.grooveSpacingNanometres = 1400.0f;
    if (rmsSlope.has_value())
        description.roughness.rmsSlope = *rmsSlope;
    description.roughness.streakSlope = streakSlope;
    // Coarse enough that a streak spans pixels of the test's small images: the disc's
    // radius is 0.5 m, so 0.08 of it is 0.04 m, about 3 pixels at 96.
    description.roughness.streakSpacingUv = 0.08f;
    std::string error;
    const auto admitted = diffractionmaterial::admit (description, error);
    check (admitted.has_value(), "the disc's grating admits: " + error);
    if (! admitted)
        return nullptr;

    auto disc = std::make_unique<pt::RenderScene>();
    pt::RenderScene& scene = *disc;
    scene.materials = { diffuse ({ 0.8f, 0.8f, 0.8f }), diffuse ({ 0.2f, 0.2f, 0.2f }) };
    if (receivers)
    {
        appendReceiver (scene, kOpening, kReach, -kOpening, kOpening, kReceiverHeight, 0);
        appendReceiver (scene, -kReach, -kOpening, -kOpening, kOpening, kReceiverHeight, 0);
        appendReceiver (scene, -kOpening, kOpening, kOpening, kReach, kReceiverHeight, 0);
        appendReceiver (scene, -kOpening, kOpening, -kReach, -kOpening, kReceiverHeight, 0);
    }
    appendPolarCap (scene, 0.5f, 64, 1);
    scene.gratingMaterials = { std::make_shared<const diffractionmaterial::AdmittedDiffractionMaterialIR> (*admitted) };
    scene.gratings = { { 1u, scene.gratingMaterials[0]->description() } };
    return disc;
}

std::unique_ptr<pt::RenderScene> makeDiscScene (const pt::EnvironmentPixels& pixels)
{
    auto disc = makeDiscGeometry (true);
    if (disc == nullptr)
        return nullptr;
    disc->lightRecords = { imageRecord (pixels) };
    disc->environmentImage = pixels;
    disc->commit();
    return disc;
}

// Below the disc looking up +Z, so pixel distance from the image centre is
// proportional to distance from the axis in the receiver plane.
pt::PinholeCamera underCamera (std::uint32_t size)
{
    pt::PinholeCamera camera;
    camera.position = { 0.0f, 0.0f, -3.0f };
    camera.rotation = { 0.0f, 1.0f, 0.0f, 0.0f };  // half turn about Y: the view axis, local -Z, becomes +Z
    camera.tanHalfFov = 0.6f;
    camera.width = size;
    camera.height = size;
    return camera;
}

struct Rendered
{
    std::uint32_t size = 0;
    std::vector<Vec3> pixels;
    std::vector<Vec3> caustic;
    pt::RenderStats stats {};
};

Rendered render (const pt::RenderScene& scene, const pt::PinholeCamera& camera, std::uint32_t size, std::uint32_t samples,
                 pt::LightStrategy strategy = pt::LightStrategy::MultipleImportance)
{
    pt::PathTracer tracer (4);
    tracer.settings.maxDepth = 8;
    tracer.settings.strategy = strategy;
    const pt::CancellationToken cancel {};
    Rendered out;
    out.size = size;
    out.stats = tracer.renderFixed (scene, camera, size, size, samples, 23, cancel);
    for (std::uint32_t y = 0; y < size; ++y)
        for (std::uint32_t x = 0; x < size; ++x)
        {
            const pt::Rgb v = tracer.accumulator().mean (x, y);
            out.pixels.push_back ({ v.r, v.g, v.b });
            const pt::Rgb c = tracer.accumulator().splatMean (x, y);
            out.caustic.push_back ({ c.r, c.g, c.b });
        }
    return out;
}

Rendered render (const pt::RenderScene& scene, std::uint32_t size, std::uint32_t samples)
{
    return render (scene, underCamera (size), size, samples);
}

// Splat energy per sector of the image's diagonal cross, one receiver in each.
std::array<double, 4> sideEnergy (const Rendered& image)
{
    std::array<double, 4> sides { 0.0, 0.0, 0.0, 0.0 };
    const double centre = (image.size - 1) * 0.5;
    for (std::uint32_t y = 0; y < image.size; ++y)
        for (std::uint32_t x = 0; x < image.size; ++x)
        {
            const Vec3 c = image.caustic[static_cast<std::size_t> (y) * image.size + x];
            const double energy = c.x + c.y + c.z;
            const double dx = x - centre, dy = y - centre;
            const std::size_t side = std::abs (dx) > std::abs (dy) ? (dx > 0.0 ? 0u : 1u) : (dy > 0.0 ? 2u : 3u);
            sides[side] += energy;
        }
    return sides;
}

// Each channel's mean splat radius in pixels, and the red share (red - blue)
// / (red + blue) inside and outside the midpoint of those radii.
struct Dispersion
{
    double redRadius = 0.0;
    double blueRadius = 0.0;
    double innerBalance = 0.0;
    double outerBalance = 0.0;
};

Dispersion dispersion (const Rendered& image)
{
    const double centre = (image.size - 1) * 0.5;
    double red = 0.0, blue = 0.0, redMoment = 0.0, blueMoment = 0.0;
    for (std::uint32_t y = 0; y < image.size; ++y)
        for (std::uint32_t x = 0; x < image.size; ++x)
        {
            const Vec3 c = image.caustic[static_cast<std::size_t> (y) * image.size + x];
            const double r = std::hypot (x - centre, y - centre);
            red += c.x;
            blue += c.z;
            redMoment += c.x * r;
            blueMoment += c.z * r;
        }
    Dispersion result;
    result.redRadius = redMoment / std::max (red, 1.0e-12);
    result.blueRadius = blueMoment / std::max (blue, 1.0e-12);
    const double split = 0.5 * (result.redRadius + result.blueRadius);
    double innerRed = 0.0, innerBlue = 0.0, outerRed = 0.0, outerBlue = 0.0;
    for (std::uint32_t y = 0; y < image.size; ++y)
        for (std::uint32_t x = 0; x < image.size; ++x)
        {
            const Vec3 c = image.caustic[static_cast<std::size_t> (y) * image.size + x];
            const bool inner = std::hypot (x - centre, y - centre) < split;
            (inner ? innerRed : outerRed) += c.x;
            (inner ? innerBlue : outerBlue) += c.z;
        }
    result.innerBalance = (innerRed - innerBlue) / std::max (innerRed + innerBlue, 1.0e-12);
    result.outerBalance = (outerRed - outerBlue) / std::max (outerRed + outerBlue, 1.0e-12);
    return result;
}

const Vec3 kOverhead { 0.0f, 0.0f, 1.0f };

// Output follows the bound image: moving or tinting the lobe changes the disc's
// pixels. Each changed image is a new asset identity, as a rebind would be.
void boundImageChangesOutputTests()
{
    const pt::EnvironmentPixels white = lobePixels (kOverhead, 400.0f, { 1.0f, 1.0f, 1.0f }, 1);
    const pt::EnvironmentPixels tinted = lobePixels (kOverhead, 400.0f, { 1.0f, 0.1f, 0.1f }, 2);
    const pt::EnvironmentPixels moved = lobePixels (pt::normalizeOrZero ({ 0.6f, 0.0f, 0.8f }), 400.0f, { 1.0f, 1.0f, 1.0f }, 3);
    const auto a = makeDiscScene (white), b = makeDiscScene (tinted), c = makeDiscScene (moved);
    if (a == nullptr || b == nullptr || c == nullptr)
        return;
    check (a->environmentImage.valid() && a->lights.environmentRadiance (kOverhead).x > 100.0f,
           "the bound image is the lit environment");

    const Rendered base = render (*a, 48, 32);
    const auto largestChange = [&base] (const Rendered& other) {
        double largest = 0.0;
        for (std::size_t i = 0; i < base.pixels.size(); ++i)
            largest = std::max ({ largest, static_cast<double> (std::abs (base.pixels[i].x - other.pixels[i].x)),
                                  static_cast<double> (std::abs (base.pixels[i].y - other.pixels[i].y)),
                                  static_cast<double> (std::abs (base.pixels[i].z - other.pixels[i].z)) });
        return largest;
    };
    const double tintChange = largestChange (render (*b, 48, 32));
    const double moveChange = largestChange (render (*c, 48, 32));
    check (tintChange > 0.01, "tinting the image changes the output by " + std::to_string (tintChange));
    check (moveChange > 0.01, "moving the image's lobe changes the output by " + std::to_string (moveChange));
}

// Concentric grooves diffract every way from the hub, so all four receivers get
// light. One uniform tangent across the disc sends every order along one axis
// and leaves two receivers dark.
void concentricOrdersTests()
{
    const auto disc = makeDiscScene (lobePixels (kOverhead, 400.0f, { 1.0f, 1.0f, 1.0f }, 1));
    if (disc == nullptr)
        return;
    const Rendered image = render (*disc, 96, 96);
    check (image.stats.spectral && image.stats.nonFiniteSamples == 0, "the disc renders spectrally and finite");
    check (image.stats.causticSplats > 0
               && image.stats.causticSplatsByLight[static_cast<std::size_t> (pt::LightType::Environment)] == image.stats.causticSplats,
           "the receivers' splats all come from the environment: " + std::to_string (image.stats.causticSplats));
    const std::array<double, 4> sides = sideEnergy (image);
    const double most = *std::max_element (sides.begin(), sides.end());
    const double least = *std::min_element (sides.begin(), sides.end());
    check (most > 0.0 && least > 0.5 * most,
           "orders reach all four receivers, weakest over strongest " + std::to_string (most > 0.0 ? least / most : 0.0));
}

// The environment's white light splits by wavelength. Order 1 leaves at
// sin(theta) = wavelength / 1400 nm, so on the receiver plane red lands further
// from the hub than blue, and the same white lobe gives a redder outer band.
void dispersedDepositTests()
{
    const auto disc = makeDiscScene (lobePixels (kOverhead, 400.0f, { 1.0f, 1.0f, 1.0f }, 1));
    if (disc == nullptr)
        return;
    const Dispersion spread = dispersion (render (*disc, 96, 96));
    check (spread.redRadius > 1.02 * spread.blueRadius,
           "red lands further out than blue, radii " + std::to_string (spread.redRadius) + " and " + std::to_string (spread.blueRadius));
    check (spread.outerBalance > spread.innerBalance + 0.05,
           "the deposit is not grey, red balance inner " + std::to_string (spread.innerBalance) + " outer " + std::to_string (spread.outerBalance));
}

} // namespace

// An authored version 3 plan: one source toward `toLight`, in W / (m^2 nm).
diffractionmaterial::LightingDescription authoredPlan (Vec3 toLight, std::array<float, 8> bands)
{
    diffractionmaterial::LightingDescription plan;
    plan.version = 3;
    plan.pathCount = 1;
    plan.paths[0].kind = diffractionmaterial::LightingPathKind::Direct;
    plan.paths[0].incident.direction = { toLight.x, toLight.y, toLight.z };
    plan.paths[0].incident.radiance = bands;
    plan.sceneReceiverTransport = diffractionmaterial::SceneReceiverTransport::AuthoredSpectralLambertian;
    return plan;
}

constexpr std::array<float, 8> kFlatBands { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
constexpr std::array<float, 8> kRedBands { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f };
// The first order of 1400 nm carries 550 nm to the camera on the axis.
const Vec3 kOrderOneSource = pt::normalizeOrZero ({ 0.393f, 0.0f, 0.92f });

// Above the disc looking down -Z, with world +X to the right of the image.
pt::PinholeCamera overCamera (std::uint32_t size)
{
    pt::PinholeCamera camera;
    camera.position = { 0.0f, 0.0f, 4.0f };
    camera.tanHalfFov = 0.15f;
    camera.width = size;
    camera.height = size;
    return camera;
}

// Mean colour of the hub's horizontal band, at x from 0.2 to 0.4 on one side.
Vec3 bandColour (const Rendered& image, bool positiveX)
{
    const double centre = (image.size - 1) * 0.5;
    const double metresPerPixel = 2.0 * 4.0 * 0.15 / image.size;
    Vec3 sum {};
    int count = 0;
    for (std::uint32_t y = 0; y < image.size; ++y)
        for (std::uint32_t x = 0; x < image.size; ++x)
        {
            const double wx = (x - centre) * metresPerPixel * (positiveX ? 1.0 : -1.0);
            const double wy = (y - centre) * metresPerPixel;
            if (wx < 0.2 || wx > 0.4 || std::abs (wy) > 0.05)
                continue;
            sum = sum + image.pixels[static_cast<std::size_t> (y) * image.size + x];
            ++count;
        }
    return count > 0 ? sum * (1.0f / static_cast<float> (count)) : Vec3 {};
}

double redBalance (Vec3 c)
{
    return (c.x - c.z) / std::max (static_cast<double> (c.x + c.z), 1.0e-12);
}

Rendered renderLitFromAbove (const grid::SceneLightRecord& record, const pt::LightShape* shape, std::uint32_t size, std::uint32_t samples)
{
    auto disc = makeDiscGeometry (false);
    if (disc == nullptr)
        return {};
    disc->lightRecords = { record };
    if (shape != nullptr)
        disc->lightShapes = { *shape };
    disc->commit();
    return render (*disc, overCamera (size), size, samples);
}

// R3: a plain key light of irradiance E W/m^2 reaches the disc's delta orders.
// Sized to the angular radius a, its radiance is E / (pi sin^2 a) W/(m^2 sr):
// 139.5 at E = 1.2. A hub pixel sees it through order 1 with efficiency at least
// 0.05, over at least 15 percent of the visible band, so the floor is 139.5 x
// 0.05 x 0.15, about 1.0 W/(m^2 sr). A delta key light gives exactly 0 here. An
// order cannot exceed its source, so 139.5 also caps the result: this is the
// source's energy and no scale on it.
void keyLightEnergyTests()
{
    constexpr float irradiance = 1.2f;
    grid::SceneLightRecord key;
    pt::LightShape unused;
    check (pt::authoredSpectralSourceLight (authoredPlan (kOrderOneSource, kFlatBands), key, unused), "the source is a light");
    key.intensity = irradiance;

    const double sinRadius = std::sin (static_cast<double> (pt::kGratingSourceAngularRadius));
    const double sourceRadiance = irradiance / (static_cast<double> (pt::kPi) * sinRadius * sinRadius);
    const double energyFloor = sourceRadiance * 0.05 * 0.15;
    check (energyFloor > 1.0 && energyFloor < 1.1, "the energy floor is about 1 W/(m^2 sr)");

    const Rendered image = renderLitFromAbove (key, nullptr, 96, 256);
    check (image.stats.nonFiniteSamples == 0, "the lit disc stays finite");
    const Vec3 right = bandColour (image, true);
    const Vec3 left = bandColour (image, false);
    const auto brightest = [] (Vec3 c) { return static_cast<double> (std::max ({ c.x, c.y, c.z })); };
    check (brightest (right) >= energyFloor, "the key light lights the band right of the hub above the floor, got " + std::to_string (brightest (right)));
    check (brightest (left) >= energyFloor, "the key light lights the band left of the hub above the floor, got " + std::to_string (brightest (left)));
    check (brightest (right) < sourceRadiance && brightest (left) < sourceRadiance, "no order exceeds its source's radiance");
    // The view leans back over the hub, so t(view) + t(source) = 0.393 - 0.075 at x = +0.3,
    // on the source's side: 445 nm toward the camera. At -0.3 it is 0.393 + 0.075: 655 nm.
    check (redBalance (right) < 0.0, "the band runs blue on the side nearer the source, balance " + std::to_string (redBalance (right)));
    check (redBalance (left) > 0.0, "the band runs red on the far side, balance " + std::to_string (redBalance (left)));
}

// R4: the source of an authored plan, from where the plan says, in its colours.
void authoredSourceTests()
{
    grid::SceneLightRecord record;
    pt::LightShape shape;
    auto plan = authoredPlan (kOrderOneSource, kFlatBands);
    plan.version = 2;
    check (! pt::authoredSpectralSourceLight (plan, record, shape), "a version 2 plan's paths are not a source");
    check (! pt::authoredSpectralSourceLight (authoredPlan (kOrderOneSource, {}), record, shape), "a source with no energy is not a light");

    // Direction: mirroring the source mirrors the split around the hub.
    check (pt::authoredSpectralSourceLight (authoredPlan (kOrderOneSource, kFlatBands), record, shape), "the authored source is a light");
    const Rendered towardSource = renderLitFromAbove (record, &shape, 96, 256);
    check (pt::authoredSpectralSourceLight (authoredPlan ({ -kOrderOneSource.x, 0.0f, kOrderOneSource.z }, kFlatBands), record, shape),
           "the mirrored source is a light");
    const Rendered mirrored = renderLitFromAbove (record, &shape, 96, 256);
    check (redBalance (bandColour (towardSource, false)) > 0.0 && redBalance (bandColour (towardSource, true)) < 0.0, "red lies away from the source");
    check (redBalance (bandColour (mirrored, true)) > 0.0 && redBalance (bandColour (mirrored, false)) < 0.0, "red follows the source across the hub");

    // Receiver: overhead, the orders land on the receivers, red farther out.
    const auto deposit = [] (std::array<float, 8> bands) {
        grid::SceneLightRecord source;
        pt::LightShape sourceShape;
        check (pt::authoredSpectralSourceLight (authoredPlan (kOverhead, bands), source, sourceShape), "the overhead source is a light");
        auto disc = makeDiscGeometry (true);
        if (disc == nullptr)
            return Rendered {};
        disc->lightRecords = { source };
        disc->lightShapes = { sourceShape };
        disc->commit();
        return render (*disc, 96, 64);
    };
    const auto total = [] (const Rendered& image) {
        Vec3 sum {};
        for (const Vec3 c : image.caustic)
            sum = sum + c;
        return sum;
    };
    const Rendered flat = deposit (kFlatBands);
    const Vec3 flatEnergy = total (flat);
    check (flatEnergy.x > 0.0f && flatEnergy.y > 0.0f && flatEnergy.z > 0.0f, "the authored source reaches the receivers in every colour");
    const Dispersion split = dispersion (flat);
    check (split.redRadius > split.blueRadius * 1.02, "the grating equation puts red farther out than blue on the receivers");

    // Colour: a source with only its two reddest bands deposits red only, at the
    // energy those bands carry in the flat source.
    const Vec3 redEnergy = total (deposit (kRedBands));
    check (redEnergy.z < 0.1f * redEnergy.x, "a red source deposits no blue");
    check (redEnergy.x > 0.35f * flatEnergy.x && redEnergy.x < 1.3f * flatEnergy.x, "a red band carries its authored energy");
}

// A sphere about the origin with the generated mesh's uv, u around and v from
// pole to pole, smooth normals, on the module's grating: latitude grooves.
void appendOrb (pt::RenderScene& scene, Vec3 centre, float radius, std::uint32_t material)
{
    constexpr std::uint32_t segments = 96, rings = 48;
    const auto normalAt = [] (std::uint32_t k, std::uint32_t r) {
        const float polar = pt::kPi * static_cast<float> (r) / rings;
        const float around = 2.0f * pt::kPi * static_cast<float> (k) / segments;
        return Vec3 { std::sin (polar) * std::cos (around), std::sin (polar) * std::sin (around), std::cos (polar) };
    };
    const auto uvAt = [] (std::uint32_t k, std::uint32_t r) {
        return pt::Vec2 { static_cast<float> (k) / segments, static_cast<float> (r) / rings };
    };
    const auto add = [&] (std::array<std::array<std::uint32_t, 2>, 3> corners) {
        std::array<Vec3, 3> n, p;
        std::array<pt::Vec2, 3> uv;
        for (std::size_t i = 0; i < 3; ++i)
        {
            n[i] = normalAt (corners[i][0], corners[i][1]);
            p[i] = centre + n[i] * radius;
            uv[i] = uvAt (corners[i][0], corners[i][1]);
        }
        appendTriangle (scene, p, uv, material);
        scene.geometry.triangles.back().hasShadingNormals = true;
        scene.geometry.triangles.back().shadingNormals = n;
    };
    for (std::uint32_t r = 0; r < rings; ++r)
        for (std::uint32_t k = 0; k < segments; ++k)
        {
            if (r > 0)
                add ({ { { k, r }, { k, r + 1 }, { k + 1, r } } });
            if (r + 1 < rings)
                add ({ { { k + 1, r }, { k, r + 1 }, { k + 1, r + 1 } } });
        }
}

std::shared_ptr<const diffractionmaterial::AdmittedDiffractionMaterialIR> moduleGrating()
{
    auto description = diffractionmaterial::makeAluminiumBinaryGratingPreset();
    description.geometry.directionUv = { 0.0f, 1.0f };
    description.geometry.grooveSpacingNanometres = 1400.0f;
    std::string error;
    const auto admitted = diffractionmaterial::admit (description, error);
    check (admitted.has_value(), "the module's grating admits: " + error);
    return admitted ? std::make_shared<const diffractionmaterial::AdmittedDiffractionMaterialIR> (*admitted) : nullptr;
}

pt::EnvironmentPixels uniformPixels (float radiance, std::uint64_t version)
{
    constexpr std::uint32_t width = 64, height = 32;
    const std::vector<float> rgb (static_cast<std::size_t> (width) * height * 3, radiance);
    return pt::copyEnvironmentPixels (width, height, rgb.data(), { "pathtrace-grating-environment-test", version, std::string (64, 'a') });
}

double luminance (Vec3 c)
{
    return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
}

// R3 silver: under a uniform sky a metal returns its substrate's reflectance, 0.94
// for aluminium at normal incidence. The orders alone resolve about 0.75 of the sky
// on this groove depth and period, the rest being the ledger's unresolved power, so
// the face's mean over the sky separates a surface with its metal lobe (above 0.85)
// from the orders alone. The face stays at or below 1, so no energy was added, and a
// key light still draws the band across the hub.
void discSilverTests()
{
    constexpr float sky = 0.3f;
    auto disc = makeDiscGeometry (false);
    if (disc == nullptr)
        return;
    const pt::EnvironmentPixels pixels = uniformPixels (sky, 11);
    disc->lightRecords = { imageRecord (pixels) };
    disc->environmentImage = pixels;
    disc->commit();
    const Rendered image = render (*disc, overCamera (96), 96, 256);
    const double centre = (image.size - 1) * 0.5;
    const double metresPerPixel = 2.0 * 4.0 * 0.15 / image.size;
    double sum = 0.0;
    int count = 0, covered = 0;
    for (std::uint32_t y = 0; y < image.size; ++y)
        for (std::uint32_t x = 0; x < image.size; ++x)
        {
            if (std::hypot (x - centre, y - centre) * metresPerPixel > 0.45)
                continue;
            const double value = luminance (image.pixels[static_cast<std::size_t> (y) * image.size + x]);
            sum += value;
            ++count;
            covered += value > 0.5 * sky ? 1 : 0;
        }
    const double reflectance = sum / std::max (count, 1) / sky;
    check (reflectance > 0.85, "the disc's face returns its metal's reflectance of the sky, got " + std::to_string (reflectance));
    check (reflectance <= 1.0, "the disc's face returns no more than the sky, got " + std::to_string (reflectance));
    check (covered > 0.95 * count, "silver covers the face, " + std::to_string (covered) + " of " + std::to_string (count));

    grid::SceneLightRecord key;
    pt::LightShape unused;
    check (pt::authoredSpectralSourceLight (authoredPlan (kOrderOneSource, kFlatBands), key, unused), "the key is a light");
    const Rendered lit = renderLitFromAbove (key, nullptr, 96, 256);
    check (redBalance (bandColour (lit, true)) < 0.0 && redBalance (bandColour (lit, false)) > 0.0,
           "the band still crosses the hub with the metal lobe");
}

// Slice 3: a real slab. The film sits inside a closed polycarbonate box and forms its
// orders there. Under a uniform sky the face returns the top surface's Fresnel 0.052 plus
// what crosses in (0.948) and comes back off metal of R 0.874 against plastic. Orders past
// the critical angle (sin 0.63) reflect inside and meet the metal again, so the sum is below
// 0.052 + 0.948 x 0.874 = 0.88. A second metal bounce for all of it would be 0.78, and the
// lower bound leaves room below that.
// In air the same metal returns 0.916, so the slab is a real loss and no energy is added.
// An authored normal tilted from dP/du must not cost the path: a float normal is unit
// only to about 1e-7, and the frame the authority checks allows 1e-10.
void tiltedNormalFrameTests()
{
    constexpr float sky = 0.3f;
    auto disc = makeDiscGeometry (false);
    if (disc == nullptr)
        return;
    const Vec3 tilted = pt::normalizeOrZero ({ 0.02f, 0.013f, 1.0f });
    for (auto& attributes : disc->geometry.triangles)
    {
        attributes.shadingNormals = { tilted, tilted, tilted };
        attributes.hasShadingNormals = true;
        attributes.authoredShading = true;
    }
    const pt::EnvironmentPixels pixels = uniformPixels (sky, 41);
    disc->lightRecords = { imageRecord (pixels) };
    disc->environmentImage = pixels;
    disc->commit();
    const Rendered image = render (*disc, overCamera (64), 64, 64);
    const double centre = (image.size - 1) * 0.5;
    const double metresPerPixel = 2.0 * 4.0 * 0.15 / image.size;
    double sum = 0.0;
    int count = 0;
    for (std::uint32_t y = 0; y < image.size; ++y)
        for (std::uint32_t x = 0; x < image.size; ++x)
            if (std::hypot (x - centre, y - centre) * metresPerPixel <= 0.45)
            {
                sum += luminance (image.pixels[static_cast<std::size_t> (y) * image.size + x]);
                ++count;
            }
    const double reflectance = sum / std::max (count, 1) / sky;
    check (reflectance > 0.85, "a tilted normal keeps the metal's reflectance of the sky, got " + std::to_string (reflectance));
}

// Brushed relief moves light between neighbouring pixels and adds none: a sky with
// no direction in it reads the metal's reflectance as before, and a small source
// breaks up into streaks that the same disc without relief does not have.
void brushedReliefTests()
{
    constexpr float sky = 0.3f;
    constexpr float slope = 0.05f;
    const auto overDisc = [] (const Rendered& image, auto&& visit) {
        const double centre = (image.size - 1) * 0.5;
        const double metresPerPixel = 2.0 * 4.0 * 0.15 / image.size;
        for (std::uint32_t y = 0; y + 1 < image.size; ++y)
            for (std::uint32_t x = 0; x + 1 < image.size; ++x)
                if (std::hypot (x - centre, y - centre) * metresPerPixel <= 0.45)
                    visit (static_cast<std::size_t> (y) * image.size + x);
    };
    {
        auto disc = makeDiscGeometry (false, std::nullopt, slope);
        if (disc == nullptr)
            return;
        const pt::EnvironmentPixels pixels = uniformPixels (sky, 41);
        disc->lightRecords = { imageRecord (pixels) };
        disc->environmentImage = pixels;
        disc->commit();
        const Rendered image = render (*disc, overCamera (64), 64, 64);
        double sum = 0.0;
        int count = 0;
        overDisc (image, [&] (std::size_t i) { sum += luminance (image.pixels[i]); ++count; });
        const double reflectance = sum / std::max (count, 1) / sky;
        check (reflectance > 0.85 && reflectance < 1.02,
               "brushed relief keeps the metal's reflectance of the sky, got " + std::to_string (reflectance));
    }
    const auto neighbourContrast = [&] (float streak) {
        auto disc = makeDiscGeometry (false, std::nullopt, streak);
        if (disc == nullptr)
            return 0.0;
        const pt::EnvironmentPixels pixels = lobePixels (kOverhead, 400.0f, { 1.0f, 1.0f, 1.0f }, 1);
        disc->lightRecords = { imageRecord (pixels) };
        disc->environmentImage = pixels;
        disc->commit();
        const Rendered image = render (*disc, overCamera (96), 96, 128);
        double difference = 0.0, sum = 0.0;
        overDisc (image, [&] (std::size_t i) {
            const double here = luminance (image.pixels[i]);
            difference += std::abs (here - luminance (image.pixels[i + 1]))
                        + std::abs (here - luminance (image.pixels[i + image.size]));
            sum += here;
        });
        return difference / std::max (sum, 1.0e-9);
    };
    const double smooth = neighbourContrast (0.0f);
    // Three times the slope above, so the streaks stand clear of 128 samples' noise.
    const double brushed = neighbourContrast (3.0f * slope);
    check (brushed > 1.5 * smooth, "brushed relief breaks up a small source's reflection, contrast "
                                       + std::to_string (smooth) + " to " + std::to_string (brushed));
}

// A spatial foil's corners reach the orders, and its mask leaves the object's own
// material showing, as Raster blends them.
void spatialFoilFieldTests()
{
    constexpr float sky = 0.3f;
    const auto meanReflectance = [&] (float coverage) {
        auto disc = makeDiscGeometry (false);
        if (disc == nullptr)
            return 0.0;
        // makeDiscGeometry's slot 1 stands in for the object's own 0.2 grey.
        auto& binding = disc->gratings[0];
        for (auto& corner : binding.field)
            corner = { 0.0f, 1.0f, 1400.0f, coverage };
        binding.masked = true;
        const pt::EnvironmentPixels pixels = uniformPixels (sky, 41);
        disc->lightRecords = { imageRecord (pixels) };
        disc->environmentImage = pixels;
        disc->commit();
        const Rendered image = render (*disc, overCamera (64), 64, 64);
        const double centre = (image.size - 1) * 0.5;
        const double metresPerPixel = 2.0 * 4.0 * 0.15 / image.size;
        double sum = 0.0;
        int count = 0;
        for (std::uint32_t y = 0; y < image.size; ++y)
            for (std::uint32_t x = 0; x < image.size; ++x)
                if (std::hypot (x - centre, y - centre) * metresPerPixel <= 0.45)
                {
                    sum += luminance (image.pixels[static_cast<std::size_t> (y) * image.size + x]);
                    ++count;
                }
        return sum / std::max (count, 1) / sky;
    };
    const double bare = meanReflectance (0.0f);
    const double half = meanReflectance (0.5f);
    const double whole = meanReflectance (1.0f);
    check (bare > 0.15 && bare < 0.25, "a mask of 0 shows the object's own 0.2 grey, got " + std::to_string (bare));
    check (whole > 0.85, "a mask of 1 is all foil, got " + std::to_string (whole));
    check (std::abs (half - 0.5 * (bare + whole)) < 0.03,
           "a mask of one half blends the two, got " + std::to_string (half) + " between " + std::to_string (bare) + " and "
               + std::to_string (whole));

    // Corners carrying another grating render as a material that has that grating.
    const auto lit = [] (pt::RenderScene& disc) {
        const pt::EnvironmentPixels pixels = lobePixels (kOverhead, 400.0f, { 1.0f, 1.0f, 1.0f }, 1);
        disc.lightRecords = { imageRecord (pixels) };
        disc.environmentImage = pixels;
        disc.commit();
        return render (disc, overCamera (48), 48, 32);
    };
    auto corners = makeDiscGeometry (false);
    auto plain = makeDiscGeometry (false);
    auto other = diffractionmaterial::makeAluminiumBinaryGratingPreset();
    other.geometry.directionUv = { 1.0f, 0.0f };
    other.geometry.grooveSpacingNanometres = 900.0f;
    std::string error;
    const auto admitted = diffractionmaterial::admit (other, error);
    check (admitted.has_value(), "the other grating admits: " + error);
    if (corners == nullptr || plain == nullptr || ! admitted)
        return;
    plain->gratingMaterials = { std::make_shared<const diffractionmaterial::AdmittedDiffractionMaterialIR> (*admitted) };
    plain->gratings[0].description = plain->gratingMaterials[0]->description();
    for (auto& corner : corners->gratings[0].field)
        corner = { 1.0f, 0.0f, 900.0f, 1.0f };
    corners->gratings[0].varied = true;
    const Rendered fromCorners = lit (*corners);
    const Rendered fromMaterial = lit (*plain);
    double difference = 0.0, sum = 0.0;
    for (std::size_t i = 0; i < fromCorners.pixels.size(); ++i)
    {
        difference += std::abs (luminance (fromCorners.pixels[i]) - luminance (fromMaterial.pixels[i]));
        sum += luminance (fromMaterial.pixels[i]);
    }
    check (sum > 0.0 && difference < 0.01 * sum, "corners set the orders' direction and period, difference "
                                                     + std::to_string (difference / std::max (sum, 1.0e-9)));
}

void discUnderSlabTests()
{
    auto description = diffractionmaterial::makeAluminiumBinaryGratingPreset();
    description.geometry.directionUv = { 0.0f, 1.0f };
    description.geometry.grooveSpacingNanometres = 1400.0f;
    description.microstructure.grooveDepthNanometres = 70.0f;
    description.substrateData = diffractionmaterial::SubstrateData::AluminiumMcPeak2015;
    description.orderMedium = diffractionmaterial::OrderMedium::SurroundingMedium;
    std::string error;
    const auto admitted = diffractionmaterial::admit (description, error);
    check (admitted.has_value(), "the slab disc's grating admits: " + error);
    if (! admitted)
        return;
    constexpr float sky = 0.3f;
    pt::RenderScene scene;
    pt::Material plastic;
    plastic.transmission = 1.0f;
    plastic.metallic = 0.0f;
    plastic.roughness = 0.0f;
    plastic.ior = 1.5848f;
    plastic.dispersion = 0.011f;
    scene.materials = { diffuse ({ 0.2f, 0.2f, 0.2f }), plastic };
    appendPolarCap (scene, 0.5f, 64, 0);
    // A closed box around the film, normals out.
    const float h = 0.8f, top = 0.1f, bottom = -0.1f;
    const Vec3 a { -h, -h, bottom }, b { h, -h, bottom }, c { h, h, bottom }, d { -h, h, bottom };
    const Vec3 e { -h, -h, top }, f { h, -h, top }, g { h, h, top }, i { -h, h, top };
    const pt::Vec2 o {};
    for (const auto& face : std::array<std::array<Vec3, 4>, 6> { { { e, f, g, i }, { d, c, b, a }, { a, b, f, e },
                                                                    { b, c, g, f }, { c, d, i, g }, { d, a, e, i } } })
    {
        appendTriangle (scene, { face[0], face[1], face[2] }, { o, o, o }, 1);
        appendTriangle (scene, { face[0], face[2], face[3] }, { o, o, o }, 1);
    }
    scene.gratingMaterials = { std::make_shared<const diffractionmaterial::AdmittedDiffractionMaterialIR> (*admitted) };
    scene.gratings = { { 0u, scene.gratingMaterials[0]->description() } };
    const pt::EnvironmentPixels pixels = uniformPixels (sky, 41);
    scene.lightRecords = { imageRecord (pixels) };
    scene.environmentImage = pixels;
    scene.commit();
    const Rendered image = render (scene, overCamera (64), 64, 256);
    const double centre = (image.size - 1) * 0.5;
    const double metresPerPixel = 2.0 * 4.0 * 0.15 / image.size;
    double sum = 0.0;
    int count = 0;
    for (std::uint32_t y = 0; y < image.size; ++y)
        for (std::uint32_t x = 0; x < image.size; ++x)
            if (std::hypot (x - centre, y - centre) * metresPerPixel < 0.4)
            {
                sum += luminance (image.pixels[static_cast<std::size_t> (y) * image.size + x]);
                ++count;
            }
    const double returned = sum / std::max (count, 1) / sky;
    check (returned > 0.72, "the film under plastic returns the sky, got " + std::to_string (returned));
    check (returned < 0.89, "the slab's stack loses to the metal against plastic, got " + std::to_string (returned));
}

// R4 orb shading: a key light of E W/m^2 gives the metal lobe a peak radiance of
// E x share / (2 pi deviation^2), about 1.1 at E = 1.2, share 0.15 and deviation
// 0.16, and sqrt(2) less per unit of normal turned. A patch of 0.08 radians of normal
// around the mirror point stays above 0.1 in every channel. Delta orders put only the
// source's small image there, a tenth of that patch, and the far side stays dark.
void orbShadingTests()
{
    const auto grating = moduleGrating();
    if (grating == nullptr)
        return;
    pt::RenderScene orb;
    orb.materials = { diffuse ({ 0.05f, 0.05f, 0.06f }) };
    appendOrb (orb, {}, 0.5f, 0);
    orb.gratingMaterials = { grating };
    orb.gratings = { { 0u, orb.gratingMaterials[0]->description() } };
    const Vec3 toKey = pt::normalizeOrZero ({ 0.5f, 0.5f, 0.7f });
    grid::SceneLightRecord key;
    pt::LightShape unused;
    check (pt::authoredSpectralSourceLight (authoredPlan (toKey, kFlatBands), key, unused), "the key is a light");
    key.intensity = 1.2f;
    orb.lightRecords = { key };
    orb.commit();

    pt::PinholeCamera camera;
    camera.position = { 0.0f, 0.0f, 3.0f };
    camera.tanHalfFov = 0.2f;
    camera.width = 128;
    camera.height = 128;
    const Rendered image = render (orb, camera, 128, 256);
    check (image.stats.nonFiniteSamples == 0, "the orb stays finite");

    const Vec3 mirrorNormal = pt::normalizeOrZero (toKey + Vec3 { 0.0f, 0.0f, 1.0f });
    const auto patch = [&image] (Vec3 normal, double& litShare) {
        const double perUnit = image.size / (2.0 * 0.2 * 3.0);
        const double cx = (image.size - 1) * 0.5 + normal.x * 0.5 * perUnit;
        const double cy = (image.size - 1) * 0.5 - normal.y * 0.5 * perUnit;
        const double radius = 0.04 * perUnit;
        int count = 0, lit = 0;
        for (std::uint32_t y = 0; y < image.size; ++y)
            for (std::uint32_t x = 0; x < image.size; ++x)
                if (std::hypot (x - cx, y - cy) <= radius)
                {
                    const Vec3 c = image.pixels[static_cast<std::size_t> (y) * image.size + x];
                    ++count;
                    lit += std::min ({ c.x, c.y, c.z }) > 0.1f ? 1 : 0;
                }
        litShare = static_cast<double> (lit) / std::max (count, 1);
    };
    double nearShare = 0.0, farShare = 0.0;
    patch (mirrorNormal, nearShare);
    patch (pt::normalizeOrZero ({ -mirrorNormal.x, -mirrorNormal.y, mirrorNormal.z }), farShare);
    check (nearShare > 0.5, "a neutral highlight lies at the key's mirror point, lit share " + std::to_string (nearShare));
    check (farShare < 0.25 * nearShare, "the side away from the key stays dark, lit share " + std::to_string (farShare));
}

// The Diffraction Orb starter's raking source and its key, both on camera left.
const Vec3 kOrbToSource { -0.64f, 0.05f, 0.77f };
const Vec3 kOrbToKey { -0.54f, 0.64f, 0.54f };

// R4 volume: a low source and a key on one side give the metal lobe its highlights
// there. The other half of the ball sees only the sky, 0.1 here, at the metal's 0.9
// reflectance. The lobe's image of E = 24 has a peak of 24 x 0.15 / 0.16 = 22, over
// a spot of about 0.01 of a half-disc area 0.39, so the lit half's mean is near 0.6.
void orbVolumeTests()
{
    const auto grating = moduleGrating();
    if (grating == nullptr)
        return;
    pt::RenderScene orb;
    orb.materials = { diffuse ({ 0.05f, 0.05f, 0.06f }) };
    appendOrb (orb, {}, 0.5f, 0);
    orb.gratingMaterials = { grating };
    orb.gratings = { { 0u, orb.gratingMaterials[0]->description() } };
    std::array<float, 8> bands {};
    bands.fill (24.0f);
    grid::SceneLightRecord source, key;
    pt::LightShape sourceShape, keyShape;
    check (pt::authoredSpectralSourceLight (authoredPlan (pt::normalizeOrZero (kOrbToSource), bands), source, sourceShape), "the source is a light");
    check (pt::authoredSpectralSourceLight (authoredPlan (pt::normalizeOrZero (kOrbToKey), kFlatBands), key, keyShape), "the key is a light");
    key.intensity = 0.6f;
    const pt::EnvironmentPixels pixels = uniformPixels (0.1f, 15);
    orb.lightRecords = { source, key, imageRecord (pixels) };
    orb.lightShapes = { sourceShape };
    orb.environmentImage = pixels;
    orb.commit();

    pt::PinholeCamera camera;
    camera.position = { 0.0f, 0.0f, 3.0f };
    camera.tanHalfFov = 0.2f;
    camera.width = 128;
    camera.height = 128;
    const Rendered image = render (orb, camera, 128, 256);
    const double perUnit = image.size / (2.0 * 0.2 * 3.0);
    double lit = 0.0, shadow = 0.0;
    int litCount = 0, shadowCount = 0;
    for (std::uint32_t y = 0; y < image.size; ++y)
        for (std::uint32_t x = 0; x < image.size; ++x)
        {
            const double px = (x - (image.size - 1) * 0.5) / perUnit;
            const double py = ((image.size - 1) * 0.5 - y) / perUnit;
            if (std::hypot (px, py) > 0.45 || std::abs (px) < 0.1)
                continue;
            const double value = luminance (image.pixels[static_cast<std::size_t> (y) * image.size + x]);
            (px < 0.0 ? lit : shadow) += value;
            ++(px < 0.0 ? litCount : shadowCount);
        }
    const double ratio = (lit / std::max (litCount, 1)) / std::max (shadow / std::max (shadowCount, 1), 1.0e-6);
    check (ratio > 2.5 && ratio < 100.0, "the ball has a light side and a shadow side, ratio " + std::to_string (ratio));
}

// R4 sparks: the metal lobe covers about 0.16 sr and a 3 degree source 0.0086 sr, so
// a lobe sample hits the source about 5% of the time, on 15% of paths, and each hit
// carries a radiance near 929. At 64 samples that is under one hit per pixel around
// the mirror point: speckle. Sampling the light at the grating vertex removes it and
// leaves the mean alone, since multiple importance sampling weighs both routes.
void gratingLobeLightSamplingTests()
{
    const auto grating = moduleGrating();
    if (grating == nullptr)
        return;
    pt::RenderScene orb;
    orb.materials = { diffuse ({ 0.05f, 0.05f, 0.06f }) };
    appendOrb (orb, {}, 0.5f, 0);
    orb.gratingMaterials = { grating };
    orb.gratings = { { 0u, orb.gratingMaterials[0]->description() } };
    const Vec3 toSource = pt::normalizeOrZero ({ 0.5f, 0.5f, 0.7f });
    grid::SceneLightRecord source;
    pt::LightShape shape;
    check (pt::authoredSpectralSourceLight (authoredPlan (toSource, kFlatBands), source, shape), "the source is a light");
    orb.lightRecords = { source };
    orb.lightShapes = { shape };
    orb.commit();

    pt::PinholeCamera camera;
    camera.position = { 0.0f, 0.0f, 3.0f };
    camera.tanHalfFov = 0.2f;
    camera.width = 128;
    camera.height = 128;
    const Vec3 mirrorNormal = pt::normalizeOrZero (toSource + Vec3 { 0.0f, 0.0f, 1.0f });
    struct Noise { double mean = 0.0; double outliers = 0.0; double spread = 0.0; };
    const auto measure = [&] (const Rendered& image) {
        const double perUnit = image.size / (2.0 * 0.2 * 3.0);
        const double cx = (image.size - 1) * 0.5 + mirrorNormal.x * 0.5 * perUnit;
        const double cy = (image.size - 1) * 0.5 - mirrorNormal.y * 0.5 * perUnit;
        std::vector<double> patch;
        for (std::uint32_t y = 0; y < image.size; ++y)
            for (std::uint32_t x = 0; x < image.size; ++x)
                if (std::hypot (x - cx, y - cy) <= 0.08 * perUnit)
                    patch.push_back (luminance (image.pixels[static_cast<std::size_t> (y) * image.size + x]));
        Noise noise;
        for (const double value : patch)
            noise.mean += value / std::max<std::size_t> (patch.size(), 1);
        std::sort (patch.begin(), patch.end());
        const double median = patch.empty() ? 0.0 : patch[patch.size() / 2];
        for (const double value : patch)
            noise.outliers += value > 4.0 * median ? 1.0 / static_cast<double> (patch.size()) : 0.0;
        // The spread around the median, relative to the mean.
        for (const double value : patch)
            noise.spread += std::abs (value - median) / std::max<std::size_t> (patch.size(), 1) / std::max (noise.mean, 1.0e-6);
        return noise;
    };
    const Noise lobeOnly = measure (render (orb, camera, 128, 64, pt::LightStrategy::BsdfOnly));
    const Noise bothRoutes = measure (render (orb, camera, 128, 64));
    check (bothRoutes.spread < 0.5 * lobeOnly.spread,
           "a light sample at the grating quiets the lobe's sparks, spread " + std::to_string (bothRoutes.spread) + " from " + std::to_string (lobeOnly.spread));
    check (bothRoutes.outliers <= lobeOnly.outliers && bothRoutes.outliers < 0.02,
           "few pixels stay outliers around the mirror point, share " + std::to_string (bothRoutes.outliers) + " from " + std::to_string (lobeOnly.outliers));
    check (std::abs (bothRoutes.mean - lobeOnly.mean) < 0.25 * lobeOnly.mean,
           "the light sample leaves the highlight's mean alone, " + std::to_string (bothRoutes.mean) + " against " + std::to_string (lobeOnly.mean));
}

// R4 receiver: the starter's sphere and plate at their real distances, lying with
// the plate in z = 0 and looked at from straight above the orb's foot. Y up in the
// module maps to +Z here, its +Z to -Y.
struct OrbReceiverValues
{
    float plateAlbedo = 0.0f;
    float keyIrradiance = 0.0f;
    float sky = 0.0f;
    float sourceBand = 0.0f;
    Vec3 toSource { 0.2f, 0.55f, 0.8f };
    Vec3 toKey { -0.24f, 0.64f, 0.73f };
};

struct OrbReceiverResult
{
    double depositOverPlate = 0.0;
    double predictedFace = 0.0;
    double renderedFace = 0.0;
    double depositChroma = 0.0;
    double colourSeparation = 0.0;
};

// How far the plate's red deposit lies from its blue one: half the absolute difference
// of the two, each smoothed over 9 pixels and scaled to sum 1. Mean radius cannot show
// this, as a convex mirror spreads the deposit smoothly and orders +1 and -1 move it in
// and out about equally. At 128 samples on 4 threads, seeds 23 to 25, this reads 0.0445,
// 0.0446, 0.0453, and 0.0352, 0.0343, 0.0340 with every order's shift fixed at 550 nm.
double depositColourSeparation (const Rendered& image)
{
    constexpr std::uint32_t window = 9;
    const auto smoothed = [&image] (float Vec3::* channel) {
        std::vector<double> out (image.caustic.size());
        double sum = 0.0;
        for (std::uint32_t y = 0; y < image.size; ++y)
            for (std::uint32_t x = 0; x < image.size; ++x)
            {
                double value = 0.0;
                for (std::uint32_t v = y + 1 > window ? y + 1 - window : 0; v <= y; ++v)
                    for (std::uint32_t u = x + 1 > window ? x + 1 - window : 0; u <= x; ++u)
                        value += image.caustic[static_cast<std::size_t> (v) * image.size + u].*channel;
                out[static_cast<std::size_t> (y) * image.size + x] = value;
                sum += value;
            }
        for (double& value : out)
            value /= std::max (sum, 1.0e-12);
        return out;
    };
    const std::vector<double> red = smoothed (&Vec3::x), blue = smoothed (&Vec3::z);
    double difference = 0.0;
    for (std::size_t i = 0; i < red.size(); ++i)
        difference += std::abs (red[i] - blue[i]);
    return 0.5 * difference;
}

OrbReceiverResult renderOrbReceiver (const OrbReceiverValues& values, std::uint64_t version)
{
    const auto grating = moduleGrating();
    if (grating == nullptr)
        return {};
    pt::RenderScene scene;
    scene.materials = { diffuse ({ values.plateAlbedo, values.plateAlbedo, values.plateAlbedo }), diffuse ({ 0.05f, 0.05f, 0.06f }) };
    const Vec3 p0 { -1.7f, -2.0f, 0.0f }, p1 { 1.7f, -2.0f, 0.0f }, p2 { 1.7f, 0.6f, 0.0f }, p3 { -1.7f, 0.6f, 0.0f };
    appendTriangle (scene, { p0, p1, p2 }, { pt::Vec2 {}, pt::Vec2 {}, pt::Vec2 {} }, 0);
    appendTriangle (scene, { p0, p2, p3 }, { pt::Vec2 {}, pt::Vec2 {}, pt::Vec2 {} }, 0);
    appendOrb (scene, { 0.0f, 0.0f, 0.9f }, 0.5f, 1);
    scene.gratingMaterials = { grating };
    scene.gratings = { { 1u, scene.gratingMaterials[0]->description() } };

    const auto fromModule = [] (Vec3 v) { return pt::normalizeOrZero ({ v.x, -v.z, v.y }); };
    std::array<float, 8> bands {};
    bands.fill (values.sourceBand);
    grid::SceneLightRecord source, key;
    pt::LightShape sourceShape, keyShape;
    check (pt::authoredSpectralSourceLight (authoredPlan (fromModule (values.toSource), bands), source, sourceShape), "the orb's source is a light");
    check (pt::authoredSpectralSourceLight (authoredPlan (fromModule (values.toKey), kFlatBands), key, keyShape), "the orb's key is a light");
    key.intensity = values.keyIrradiance;
    const pt::EnvironmentPixels pixels = uniformPixels (values.sky, version);
    scene.lightRecords = { source, key, imageRecord (pixels) };
    scene.lightShapes = { sourceShape };
    scene.environmentImage = pixels;
    scene.commit();

    pt::PinholeCamera camera;
    camera.position = { 0.0f, 0.0f, 4.5f };
    camera.tanHalfFov = 0.5f;
    camera.width = 128;
    camera.height = 128;
    const Rendered image = render (scene, camera, 128, 128);
    check (image.stats.nonFiniteSamples == 0, "the orb's receiver stays finite");

    // The brightest percent of deposit over the plate's own radiance from key and sky.
    std::vector<double> deposit;
    deposit.reserve (image.caustic.size());
    for (const Vec3 c : image.caustic)
        deposit.push_back (std::max ({ c.x, c.y, c.z }));
    std::sort (deposit.begin(), deposit.end());
    const double depositPeak = deposit[deposit.size() * 99 / 100];
    // The source lights the plate too, by the cosine of its height over the plate.
    const double sourceCos = pt::normalizeOrZero (values.toSource).y;
    const double keyCos = pt::normalizeOrZero (values.toKey).y;
    const double plate = values.plateAlbedo
                       * ((values.sourceBand * sourceCos + values.keyIrradiance * keyCos) / pt::kPi + values.sky);
    OrbReceiverResult result;
    result.predictedFace = plate;
    result.depositOverPlate = depositPeak / std::max (plate, 1.0e-6);
    // The plate's face past the orb's own disc, whose image reaches about 0.63 from
    // the foot, read inside the plate's edges. Half the image spans 0.5 x 4.5 units.
    std::vector<double> face;
    double chroma = 0.0;
    int chromaCount = 0;
    const double perUnit = (image.size - 1) * 0.5 / (0.5 * 4.5);
    for (std::uint32_t y = 0; y < image.size; ++y)
        for (std::uint32_t x = 0; x < image.size; ++x)
        {
            const double px = (x - (image.size - 1) * 0.5) / perUnit;
            const double py = ((image.size - 1) * 0.5 - y) / perUnit;
            if (std::abs (px) > 1.6 || py < -1.9 || py > 0.5 || std::hypot (px, py) < 0.8)
                continue;
            const std::size_t i = static_cast<std::size_t> (y) * image.size + x;
            // The film's mean already holds the light paths' deposit.
            const Vec3 d = image.caustic[i];
            const Vec3 c = image.pixels[i];
            face.push_back (luminance (c));
            if (std::max ({ d.x, d.y, d.z }) >= depositPeak)
            {
                const double peak = std::max ({ c.x, c.y, c.z });
                chroma += (peak - std::min ({ c.x, c.y, c.z })) / std::max (peak, 1.0e-6);
                ++chromaCount;
            }
        }
    std::sort (face.begin(), face.end());
    result.renderedFace = face.empty() ? 0.0 : face[face.size() / 2];
    result.depositChroma = chroma / std::max (chromaCount, 1);
    result.colourSeparation = depositColourSeparation (image);
    return result;
}

// Why the fixtures passed while the frame showed nothing: they assert energy above
// 0 on a 0.8 albedo receiver with no other light, and this test's plate figure left
// the source out. The source is a scene light, so at E = 8 and cos 0.555 it alone
// gives the 0.7 plate 0.7 x 8 x 0.555 / pi = 0.99, and with key and sky the face
// passes display white. The deposit scales with E and the albedo too, so neither
// changes deposit / plate. It scales with the orb's cross-section and not with the
// cosine, so a raking source does: 0.7 x (24 x 0.05 + 0.6 x 0.64) / pi + 0.7 x 0.1
// = 0.42, which leaves 0.58 of display range for the deposit, estimated at 0.24.
void orbReceiverTests()
{
    const OrbReceiverResult before = renderOrbReceiver ({ 0.14f, 1.2f, 0.02f, 1.0f }, 12);
    const OrbReceiverResult blown = renderOrbReceiver ({ 0.7f, 0.3f, 0.1f, 8.0f }, 14);
    const OrbReceiverResult after = renderOrbReceiver ({ 0.7f, 0.6f, 0.1f, 24.0f, kOrbToSource, kOrbToKey }, 13);
    check (before.depositOverPlate < 0.5, "the old starter values wash the deposit out, ratio " + std::to_string (before.depositOverPlate));
    check (blown.renderedFace > 1.0, "the steep source clips the plate white, face " + std::to_string (blown.renderedFace));
    check (blown.depositOverPlate < 0.25, "the steep source drowns the deposit, ratio " + std::to_string (blown.depositOverPlate));
    check (after.renderedFace > 0.7 * after.predictedFace && after.renderedFace < 1.3 * after.predictedFace,
           "the plate's face follows its light budget, " + std::to_string (after.renderedFace) + " against " + std::to_string (after.predictedFace));
    check (after.renderedFace < 0.75, "the plate's face leaves headroom under display white, face " + std::to_string (after.renderedFace));
    check (after.depositOverPlate > 0.25, "the plate carries visible deposited colour, ratio " + std::to_string (after.depositOverPlate));
    check (after.depositChroma > 0.1, "the deposit reads as colour on the plate, chroma " + std::to_string (after.depositChroma));
    check (after.colourSeparation > 0.040, "the grating equation parts red from blue on the plate, separation " + std::to_string (after.colourSeparation));
}

// The cause of the stair-steps on R3. The grating's frame was each triangle's own
// dP/du, constant over a wedge, so a 64-segment face was 64 straight gratings
// whose grooves miss the true circle by half a wedge, 2.8125 degrees or 168.75
// arcminutes, at every wedge edge. A fan crossing that edge jumps. The smooth
// frame from ingest must stay on the circle everywhere, through the seam wedge,
// a hole, a rotated placement and a mirrored one.
double arcminutesBetween (Vec3 a, Vec3 b)
{
    const double crossed = pt::length (pt::cross (a, b));
    return std::atan2 (crossed, std::abs (static_cast<double> (pt::dot (a, b)))) * 180.0 / 3.14159265358979323846 * 60.0;
}

void grooveFrameTests()
{
    constexpr std::uint32_t segments = 64;
    constexpr float outer = 0.65f;
    const auto check3 = [] (float inner, const grid::SceneTransform3D& transform, const std::string& name) {
        auto scene = std::make_unique<grid::Visual3DScene>();
        auto& object = scene->objects[scene->objectCount++];
        object.id.value = 1;
        object.transform = transform;
        const auto at = [&] (std::uint32_t k, float radius, float v) {
            const float phi = 2.0f * pt::kPi * static_cast<float> (k) / static_cast<float> (segments);
            auto& vertex = scene->vertices[scene->vertexCount++];
            vertex.position = { radius * std::cos (phi), 0.0f, radius * std::sin (phi) };
            vertex.uv = { static_cast<float> (k) / static_cast<float> (segments), v };
        };
        const auto triangle = [&] (std::uint32_t a, std::uint32_t b, std::uint32_t c) {
            scene->indices[scene->indexCount++] = a;
            scene->indices[scene->indexCount++] = b;
            scene->indices[scene->indexCount++] = c;
        };
        for (std::uint32_t k = 0; k < segments; ++k)
            at (k, outer, 1.0f);
        if (inner > 0.0f)
        {
            for (std::uint32_t k = 0; k < segments; ++k)
                at (k, inner, inner / outer);
            for (std::uint32_t k = 0; k < segments; ++k)
            {
                const std::uint32_t next = (k + 1) % segments;
                triangle (segments + k, next, k);
                triangle (segments + k, segments + next, next);
            }
        }
        else
        {
            auto& hub = scene->vertices[scene->vertexCount++];
            hub.uv = { 0.5f, 0.0f };
            for (std::uint32_t k = 0; k < segments; ++k)
                triangle (segments, (k + 1) % segments, k);
        }
        object.vertexCount = static_cast<std::uint32_t> (scene->vertexCount);
        object.indexCount = static_cast<std::uint32_t> (scene->indexCount);
        scene->objectInstanceCount = 0;

        pt::IngestedScene ingested;
        pt::ingestScene (*scene, ingested);
        check (ingested.triangleCount() == object.indexCount / 3, name + ": every triangle ingests");
        const pt::Mat4 world = pt::objectWorldMatrix (*scene, 0);
        const Vec3 centre = pt::transformPoint (world, Vec3 {});
        const Vec3 axis = pt::normalizeOrZero (pt::transformPoint (world, Vec3 { 0.0f, 1.0f, 0.0f }) - centre);

        double smoothWorst = 0.0, triangleWorst = 0.0;
        double sign = 0.0;
        bool oneDirection = true;
        for (std::size_t t = 0; t < ingested.triangleCount(); ++t)
        {
            const auto& attributes = ingested.triangles[t];
            const std::array<Vec3, 3> p { ingested.positions[3 * t], ingested.positions[3 * t + 1], ingested.positions[3 * t + 2] };
            const float du1 = attributes.uvs[1].x - attributes.uvs[0].x, dv1 = attributes.uvs[1].y - attributes.uvs[0].y;
            const float du2 = attributes.uvs[2].x - attributes.uvs[0].x, dv2 = attributes.uvs[2].y - attributes.uvs[0].y;
            const Vec3 chord = pt::normalizeOrZero ((p[1] - p[0]) * dv2 - (p[2] - p[0]) * dv1) * (du1 * dv2 - du2 * dv1 < 0.0f ? -1.0f : 1.0f);
            constexpr int steps = 8;
            for (int a = 0; a <= steps; ++a)
                for (int b = 0; b <= steps - a; ++b)
                {
                    const float u = static_cast<float> (a) / steps, v = static_cast<float> (b) / steps, w = 1.0f - u - v;
                    const Vec3 position = p[0] * w + p[1] * u + p[2] * v;
                    const Vec3 toPoint = position - centre;
                    if (pt::length (toPoint) < 0.02f)
                        continue;
                    const Vec3 analytic = pt::normalizeOrZero (pt::cross (axis, toPoint));
                    const Vec3 smooth = attributes.grooveTangents[0] * w + attributes.grooveTangents[1] * u + attributes.grooveTangents[2] * v;
                    smoothWorst = std::max (smoothWorst, arcminutesBetween (smooth, analytic));
                    triangleWorst = std::max (triangleWorst, arcminutesBetween (chord, analytic));
                    const double side = pt::dot (smooth, analytic);
                    if (sign == 0.0)
                        sign = side;
                    oneDirection = oneDirection && side * sign > 0.0;
                }
        }
        check (triangleWorst > 150.0 && triangleWorst < 180.0,
               name + ": a triangle's own frame misses the circle by half a wedge, got " + std::to_string (triangleWorst) + " arcmin");
        check (smoothWorst < 1.0, name + ": the smooth frame stays on the circle, worst " + std::to_string (smoothWorst) + " arcmin");
        check (oneDirection, name + ": the frame runs one way round, through the seam wedge");
    };

    grid::SceneTransform3D placed;
    placed.translation = { 0.2f, 0.75f, -0.1f };
    const float half = 0.5f * 84.0f * pt::kPi / 180.0f;
    placed.rotation = { std::sin (half), 0.0f, 0.0f, std::cos (half) };
    grid::SceneTransform3D mirrored = placed;
    mirrored.scale = { -1.0f, 1.0f, 1.0f };
    check3 (0.0f, {}, "a solid cap");
    check3 (0.24f, {}, "a ring");
    check3 (0.24f, placed, "a placed ring");
    check3 (0.24f, mirrored, "a mirrored ring");
}

// The cause of the rectangles on R3. Order 0 left as a delta, a perfect mirror, so
// the face held a sharp image of whatever the environment shows, and a studio's
// window panes came back as hard vertical strips the width of the pane. Behind a
// straight edge in the environment, a delta's two sides differ by the whole of
// order 0's power. A lobe of 0.16 radians spreads the same edge over about a third
// of the disc, and sides only 0.015 radians apart barely differ.
void zeroOrderSpreadTests()
{
    constexpr std::uint32_t width = 256, height = 128;
    std::vector<float> rgb (static_cast<std::size_t> (width) * height * 3, 0.0f);
    for (std::uint32_t y = 0; y < height; ++y)
        for (std::uint32_t x = width / 2; x < width; ++x)
            for (std::uint32_t channel = 0; channel < 3; ++channel)
                rgb[(static_cast<std::size_t> (y) * width + x) * 3 + channel] = 1.0f;
    const pt::EnvironmentPixels pixels = pt::copyEnvironmentPixels (width, height, rgb.data(),
                                                                    { "pathtrace-grating-environment-test", 23, std::string (64, 'a') });
    auto disc = makeDiscGeometry (false);
    if (disc == nullptr)
        return;
    disc->lightRecords = { imageRecord (pixels) };
    disc->environmentImage = pixels;
    disc->commit();
    const Rendered image = render (*disc, overCamera (128), 128, 256);
    check (image.stats.nonFiniteSamples == 0, "the spread stays finite");
    const double centre = (image.size - 1) * 0.5;
    const double metresPerPixel = 2.0 * 4.0 * 0.15 / image.size;
    double left = 0.0, right = 0.0;
    int leftCount = 0, rightCount = 0;
    for (std::uint32_t y = 0; y < image.size; ++y)
        for (std::uint32_t x = 0; x < image.size; ++x)
        {
            const double wx = (x - centre) * metresPerPixel, wy = (y - centre) * metresPerPixel;
            if (std::abs (wy) > 0.2 || std::abs (wx) < 0.02 || std::abs (wx) > 0.08)
                continue;
            const double value = luminance (image.pixels[static_cast<std::size_t> (y) * image.size + x]);
            (wx < 0.0 ? left : right) += value;
            (wx < 0.0 ? leftCount : rightCount) += 1;
        }
    left /= std::max (leftCount, 1);
    right /= std::max (rightCount, 1);
    const double contrast = std::abs (right - left) / std::max (right + left, 1.0e-9);
    check (right + left > 0.0, "the bright half of the environment reaches the face");
    check (contrast < 0.15, "an environment edge leaves no hard edge in the face, contrast " + std::to_string (contrast));
}

// R3 dark sectors: the metal reflects black where no source sits at the diffraction
// angle. A source at tangential sine 0.393 sends 550 nm through the first order, so
// it lights the sectors of the concentric grooves whose radius points along it, and
// no order at all reaches the sectors at right angles. Only the lobe can light those.
// With L = 50 over a 4-degree cone (0.0153 sr), E = 0.77. At 0.39 rad off the mirror a
// lobe of sigma 0.16 returns about 0.77 x 0.32 x share, an even grey that hides the
// fans. Sigma 0.02 is 19 sigma away, so the sector is black, with the same energy.
void darkSectorTests()
{
    const pt::EnvironmentPixels pixels = lobePixels (kOrderOneSource, 50.0f, { 1.0f, 1.0f, 1.0f }, 31);
    const auto sectors = [&](float rmsSlope)
    {
        auto disc = makeDiscGeometry (false, rmsSlope);
        std::array<double, 2> means { 0.0, 0.0 };
        if (disc == nullptr)
            return means;
        disc->lightRecords = { imageRecord (pixels) };
        disc->environmentImage = pixels;
        disc->commit();
        const Rendered image = render (*disc, overCamera (128), 128, 256);
        check (image.stats.nonFiniteSamples == 0, "the sectors stay finite");
        const double centre = (image.size - 1) * 0.5;
        const double metresPerPixel = 2.0 * 4.0 * 0.15 / image.size;
        std::array<int, 2> counts { 0, 0 };
        for (std::uint32_t y = 0; y < image.size; ++y)
            for (std::uint32_t x = 0; x < image.size; ++x)
            {
                const double wx = (x - centre) * metresPerPixel, wy = (y - centre) * metresPerPixel;
                const double radius = std::hypot (wx, wy);
                if (radius < 0.15 || radius > 0.45)
                    continue;
                // Within 15 degrees of the source's azimuth or of the one across it.
                const bool fan = std::abs (wy) < radius * 0.259;
                const bool dark = std::abs (wx) < radius * 0.259;
                if (! fan && ! dark)
                    continue;
                const std::size_t sector = fan ? 0u : 1u;
                means[sector] += luminance (image.pixels[static_cast<std::size_t> (y) * image.size + x]);
                ++counts[sector];
            }
        means[0] /= std::max (counts[0], 1);
        means[1] /= std::max (counts[1], 1);
        return means;
    };
    const auto wide = sectors (0.08f);
    const auto tight = sectors (0.01f);
    check (wide[1] > 0.02, "the wide lobe greys the sector no order reaches, got " + std::to_string (wide[1]));
    check (tight[1] < 0.05 * wide[1], "the tight lobe leaves that sector black, got " + std::to_string (tight[1])
                                          + " against " + std::to_string (wide[1]));
    check (tight[0] > 0.05, "the source's fan stays lit, got " + std::to_string (tight[0]));
    check (tight[0] > 50.0 * tight[1], "the fan stands out of black, " + std::to_string (tight[0])
                                           + " over " + std::to_string (tight[1]));

    // The darkness is the room's, not lost energy: under a uniform sky the tight lobe
    // still returns the metal's reflectance, as the wide lobe does in discSilverTests.
    constexpr float sky = 0.3f;
    auto furnace = makeDiscGeometry (false, 0.01f);
    if (furnace == nullptr)
        return;
    const pt::EnvironmentPixels uniform = uniformPixels (sky, 32);
    furnace->lightRecords = { imageRecord (uniform) };
    furnace->environmentImage = uniform;
    furnace->commit();
    const Rendered image = render (*furnace, overCamera (96), 96, 256);
    const double centre = (image.size - 1) * 0.5;
    const double metresPerPixel = 2.0 * 4.0 * 0.15 / image.size;
    double sum = 0.0;
    int count = 0;
    for (std::uint32_t y = 0; y < image.size; ++y)
        for (std::uint32_t x = 0; x < image.size; ++x)
            if (std::hypot (x - centre, y - centre) * metresPerPixel <= 0.45)
            {
                sum += luminance (image.pixels[static_cast<std::size_t> (y) * image.size + x]);
                ++count;
            }
    const double reflectance = sum / std::max (count, 1) / sky;
    check (reflectance > 0.85 && reflectance <= 1.0, "the tight lobe keeps the metal's reflectance, got " + std::to_string (reflectance));

    // Through the integrator, per ring and per channel: a band of lost wavelengths
    // moves with the view offset and tints each ring differently. sRGB red weighs
    // blue-green negatively, so losing that band raises red above the reflectance.
    for (int ring = 0; ring < 3; ++ring)
    {
        const double inner = 0.15 * ring, outer = inner + 0.15;
        Vec3 total { 0.0f, 0.0f, 0.0f };
        int inRing = 0;
        for (std::uint32_t y = 0; y < image.size; ++y)
            for (std::uint32_t x = 0; x < image.size; ++x)
            {
                const double radius = std::hypot (x - centre, y - centre) * metresPerPixel;
                if (radius < inner || radius > outer)
                    continue;
                const Vec3 pixel = image.pixels[static_cast<std::size_t> (y) * image.size + x];
                total = { total.x + pixel.x, total.y + pixel.y, total.z + pixel.z };
                ++inRing;
            }
        const double scale = 1.0 / (std::max (inRing, 1) * static_cast<double> (sky));
        const double r = total.x * scale, g = total.y * scale, b = total.z * scale;
        const std::string name = "ring " + std::to_string (ring) + " returns " + std::to_string (r) + ", "
            + std::to_string (g) + ", " + std::to_string (b);
        check (r > 0.85 && g > 0.85 && b > 0.85, name + ", each channel near the metal's reflectance");
        check (r <= 1.0 && g <= 1.0 && b <= 1.0, name + ", no channel above the sky");
        check (std::max ({ r, g, b }) < 1.08 * std::min ({ r, g, b }), name + ", a neutral face");
    }
}

// R5 tear: roughness spreads the diffraction orders too. Delta orders mirrored each
// horizontal edge of the studio HDRI as a hard line across the card, under an
// authored slope of 0.08. Here sigma is 2 x 0.04 = 0.08 rad. 30 degrees off the
// source's azimuth, the nearest order misses the source's centre by
// 0.393 x sin(30) = 0.196, which is 2.45 sigma, or about 0.05 of the fan's peak.
// A delta returns black there, since the 4-degree source covers only 0.07 of that
// gap. Order 0 is 0.393 / 0.08 = 4.9 sigma away, a share of e^-12.
void orderSpreadTests()
{
    const pt::EnvironmentPixels pixels = lobePixels (kOrderOneSource, 50.0f, { 1.0f, 1.0f, 1.0f }, 37);
    auto disc = makeDiscGeometry (false, 0.04f);
    if (disc == nullptr)
        return;
    disc->lightRecords = { imageRecord (pixels) };
    disc->environmentImage = pixels;
    disc->commit();
    const Rendered image = render (*disc, overCamera (128), 128, 256);
    check (image.stats.nonFiniteSamples == 0, "the spread orders stay finite");
    const double centre = (image.size - 1) * 0.5;
    const double metresPerPixel = 2.0 * 4.0 * 0.15 / image.size;
    double fan = 0.0, beside = 0.0;
    int fanCount = 0, besideCount = 0;
    for (std::uint32_t y = 0; y < image.size; ++y)
        for (std::uint32_t x = 0; x < image.size; ++x)
        {
            const double wx = (x - centre) * metresPerPixel, wy = (y - centre) * metresPerPixel;
            const double radius = std::hypot (wx, wy);
            if (radius < 0.15 || radius > 0.45)
                continue;
            const double sine = std::abs (wy) / radius;  // 0.087 is 5 degrees, 0.42 to 0.57 is 25 to 35
            const double value = luminance (image.pixels[static_cast<std::size_t> (y) * image.size + x]);
            if (sine < 0.087)
            {
                fan += value;
                ++fanCount;
            }
            else if (sine > 0.42 && sine < 0.57)
            {
                beside += value;
                ++besideCount;
            }
        }
    fan /= std::max (fanCount, 1);
    beside /= std::max (besideCount, 1);
    check (fan > 0.0, "the source's fan is lit");
    check (beside > 0.01 * fan, "roughness spreads an order off its delta direction, " + std::to_string (beside) + " beside a fan of "
                                    + std::to_string (fan));
}

int main()
{
    boundImageChangesOutputTests();
    concentricOrdersTests();
    dispersedDepositTests();
    keyLightEnergyTests();
    authoredSourceTests();
    discSilverTests();
    tiltedNormalFrameTests();
    brushedReliefTests();
    spatialFoilFieldTests();
    discUnderSlabTests();
    orbShadingTests();
    orbReceiverTests();
    orbVolumeTests();
    gratingLobeLightSamplingTests();
    grooveFrameTests();
    zeroOrderSpreadTests();
    darkSectorTests();
    orderSpreadTests();
    std::cout << checks << " checks, " << failures << " failures\n";
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
