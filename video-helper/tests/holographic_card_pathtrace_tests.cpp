// Holographic Trading Card in Path Trace: the card's committed request traced by the
// real engine on the device ARBIT_PATHTRACE_DEVICE names (Auto by default). Prints
// whether the grating binding is attached, the foil mask the tracer reads at the
// corners and centre, and the hue spread on the card, for the starter's old sun, its
// new sun and a moved sun. The new sun must lay spectral bands across the card, and
// moving it must move them. The card seen from the floor's mirror point, which is what
// the starter's glossy floor reflects, must show bands too. pathtrace_route_tests runs it.

#include "../src/fixture_scene_renderer.h"
#include "../src/pathtrace_fixture_engine.h"
#include "../src/pathtrace_preview_denoise.h"
#include "../src/diffraction_material_admission.h"
#include "support/fixture_scene.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace
{
namespace fixture = videorender::fixture3d;
namespace ptfixture = videorender::fixture3d::pathtracefixture;
namespace grid = HarmonicMIDI::grid;

int cardChecks = 0;
int cardFailures = 0;

void cardCheck (bool condition, const char* message)
{
    ++cardChecks;
    if (! condition)
    {
        ++cardFailures;
        std::fprintf (stderr, "FAIL: %s\n", message);
    }
}

class CardFrame final : public arbitgpu::NativeFixtureSceneFrame
{
public:
    CardFrame (std::string backend, std::uint32_t width, std::uint32_t height) noexcept
        : backend_ (std::move (backend)), width_ (width), height_ (height) {}
    const std::string& backend() const noexcept override { return backend_; }
    std::uint32_t width() const noexcept override { return width_; }
    std::uint32_t height() const noexcept override { return height_; }
    std::uintptr_t colorImageHandle() const noexcept override { return 31; }
    std::uintptr_t colorTextureViewHandle() const noexcept override { return 32; }
    std::uintptr_t depthImageHandle() const noexcept override { return 33; }
    std::uintptr_t depthTextureViewHandle() const noexcept override { return 34; }
    arbitgpu::NativeTextureViewDescriptor colorTextureDescriptor() const noexcept override
    {
        return { backend_, arbitgpu::NativeTextureViewKind::Texture2D,
            arbitgpu::NativeTexturePixelFormat::Rgba8Unorm, colorImageHandle(), colorTextureViewHandle(),
            width_, height_, 1, true, 77, 1 };
    }

private:
    std::string backend_;
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
};

struct CardFilm
{
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<float> rgb;  // linear, rows bottom first
};

CardFilm cardFilm;

std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> captureCard (
    const std::string& backend, const arbitgpu::CpuSceneImage& image, std::string&)
{
    cardFilm.width = image.width;
    cardFilm.height = image.height;
    cardFilm.rgb.assign (static_cast<std::size_t> (image.width) * image.height * 3, 0.0f);
    for (std::size_t pixel = 0; pixel < static_cast<std::size_t> (image.width) * image.height; ++pixel)
        for (std::size_t c = 0; c < 3; ++c)
        {
            if (image.linearColor)
            {
                const auto* bytes = image.color->data() + pixel * 8 + c * 2;
                cardFilm.rgb[pixel * 3 + c] = fixture::pathtracedenoise::halfToFloat (
                    static_cast<std::uint16_t> (bytes[0] | (bytes[1] << 8)));
            }
            else
            {
                cardFilm.rgb[pixel * 3 + c] = (*image.color)[pixel * 4 + c] / 255.0f;
            }
        }
    return std::make_shared<CardFrame> (backend, image.width, image.height);
}

// The foil's corner field at uv, bilinear as the tracer's grooveFieldAt reads it.
std::array<float, 4> foilFieldAt (const std::array<std::array<float, 4>, 4>& field, float u, float v)
{
    std::array<float, 4> local {};
    for (std::size_t i = 0; i < 4; ++i)
        local[i] = (field[0][i] * (1.0f - u) + field[1][i] * u) * (1.0f - v)
                 + (field[2][i] * (1.0f - u) + field[3][i] * u) * v;
    return local;
}

struct HueReport
{
    float saturatedFraction = 0.0f;  // card blocks with saturation above 0.25
    float hueSpreadDegrees = 0.0f;   // 360 minus the largest empty hue arc
    float meanLuminance = 0.0f;
    float coherence = 0.0f;          // share of saturated neighbour pairs within 30 degrees of hue
    std::vector<int> blockHue;       // per card block, row by row, -1 when not saturated
};

// Part of the film, as fractions of its width and height, rows bottom first.
struct FilmRegion
{
    float left = 0.0f;
    float right = 1.0f;
    float bottom = 0.0f;
    float top = 1.0f;
};

// The card is the inner 80% of its projected rectangle: half extents 0.0315 and
// 0.044 at 0.15965 from the camera, under a 0.65 rad vertical field of view.
FilmRegion centredCard (const CardFilm& film)
{
    const float tanHalf = std::tan (0.325f);
    const float aspect = static_cast<float> (film.width) / static_cast<float> (film.height);
    const float halfX = 0.8f * 0.0315f / (0.15965f * tanHalf * aspect);
    const float halfY = 0.8f * 0.044f / (0.15965f * tanHalf);
    return { 0.5f - 0.5f * halfX, 0.5f + 0.5f * halfX, 0.5f - 0.5f * halfY, 0.5f + 0.5f * halfY };
}

grid::SceneVec3 rotated (const grid::SceneQuaternion& q, grid::SceneVec3 v)
{
    // v + 2 w (q x v) + 2 q x (q x v)
    const grid::SceneVec3 u { q.y * v.z - q.z * v.y, q.z * v.x - q.x * v.z, q.x * v.y - q.y * v.x };
    const grid::SceneVec3 t { q.y * u.z - q.z * u.y, q.z * u.x - q.x * u.z, q.x * u.y - q.y * u.x };
    return { v.x + 2.0f * (q.w * u.x + t.x), v.y + 2.0f * (q.w * u.y + t.y), v.z + 2.0f * (q.w * u.z + t.z) };
}

// The inner 80% of the card's face (the GLB's 63 x 88 mm card at the origin, facing +Z)
// as camera projects it onto the film.
FilmRegion projectedCard (const grid::SceneCameraRecord& camera, const CardFilm& film)
{
    const float tanHalf = std::tan (0.5f * camera.verticalFovRadians);
    const float aspect = static_cast<float> (film.width) / static_cast<float> (film.height);
    const grid::SceneQuaternion inverse { -camera.transform.rotation.x, -camera.transform.rotation.y,
                                          -camera.transform.rotation.z, camera.transform.rotation.w };
    FilmRegion region { 1.0f, 0.0f, 1.0f, 0.0f };
    for (const float x : { -0.8f * 0.0315f, 0.8f * 0.0315f })
        for (const float y : { -0.8f * 0.044f, 0.8f * 0.044f })
        {
            const auto& origin = camera.transform.translation;
            const auto view = rotated (inverse, { x - origin.x, y - origin.y, -origin.z });
            const float depth = std::max (1.0e-4f, -view.z);
            const float u = 0.5f + 0.5f * view.x / (depth * tanHalf * aspect);
            const float v = 0.5f + 0.5f * view.y / (depth * tanHalf);
            region = { std::min (region.left, u), std::max (region.right, u), std::min (region.bottom, v), std::max (region.top, v) };
        }
    region = { std::clamp (region.left, 0.0f, 1.0f), std::clamp (region.right, 0.0f, 1.0f),
               std::clamp (region.bottom, 0.0f, 1.0f), std::clamp (region.top, 0.0f, 1.0f) };
    return region;
}

// The camera mirrored in the starter's floor, the plane y = floorHeight: what the
// glossy floor reflects is the scene seen from there, flipped upside down.
grid::SceneCameraRecord mirroredInFloor (grid::SceneCameraRecord camera, float floorHeight)
{
    camera.transform.translation.y = 2.0f * floorHeight - camera.transform.translation.y;
    camera.transform.rotation.x = -camera.transform.rotation.x;
    camera.transform.rotation.z = -camera.transform.rotation.z;
    return camera;
}

// Hue of the card's 4 x 4 pixel blocks, so path noise averages out before the hue is
// read.
HueReport cardHues (const CardFilm& film, const FilmRegion& card)
{
    HueReport report;
    const int x0 = static_cast<int> (card.left * film.width);
    const int x1 = static_cast<int> (card.right * film.width);
    const int y0 = static_cast<int> (card.bottom * film.height);
    const int y1 = static_cast<int> (card.top * film.height);
    std::array<int, 36> bins {};
    int blocks = 0, saturated = 0, columns = 0;
    double luminance = 0.0;
    for (int bx = x0; bx + 4 <= x1; bx += 4)
        ++columns;
    for (int by = y0; by + 4 <= y1; by += 4)
        for (int bx = x0; bx + 4 <= x1; bx += 4)
        {
            float rgb[3] {};
            for (int y = by; y < by + 4; ++y)
                for (int x = bx; x < bx + 4; ++x)
                    for (int c = 0; c < 3; ++c)
                        rgb[c] += film.rgb[(static_cast<std::size_t> (y) * film.width + x) * 3 + c] / 16.0f;
            ++blocks;
            luminance += 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
            const float high = std::max ({ rgb[0], rgb[1], rgb[2] });
            const float low = std::min ({ rgb[0], rgb[1], rgb[2] });
            if (! (high > 0.02f) || (high - low) / high < 0.25f)
            {
                report.blockHue.push_back (-1);
                continue;
            }
            const float delta = high - low;
            float hue = high == rgb[0] ? std::fmod ((rgb[1] - rgb[2]) / delta + 6.0f, 6.0f)
                      : high == rgb[1] ? (rgb[2] - rgb[0]) / delta + 2.0f
                                       : (rgb[0] - rgb[1]) / delta + 4.0f;
            hue *= 60.0f;
            report.blockHue.push_back (static_cast<int> (hue));
            ++saturated;
            ++bins[static_cast<std::size_t> (std::min (35, static_cast<int> (hue / 10.0f)))];
        }
    if (blocks == 0)
        return report;
    // A band is coloured over several blocks with one hue; path noise colours lone
    // blocks with any hue, so its neighbours agree about one time in six.
    int pairs = 0, agreeing = 0;
    for (std::size_t i = 0; i < report.blockHue.size(); ++i)
        for (const std::size_t j : { i + 1, i + static_cast<std::size_t> (columns) })
        {
            if (j >= report.blockHue.size() || (j == i + 1 && (i + 1) % static_cast<std::size_t> (columns) == 0))
                continue;
            const int a = report.blockHue[i], b = report.blockHue[j];
            if (a < 0 || b < 0)
                continue;
            ++pairs;
            const int difference = std::abs (a - b) % 360;
            agreeing += std::min (difference, 360 - difference) <= 30 ? 1 : 0;
        }
    report.coherence = pairs > 0 ? static_cast<float> (agreeing) / pairs : 0.0f;
    report.saturatedFraction = static_cast<float> (saturated) / blocks;
    report.meanLuminance = static_cast<float> (luminance / blocks);
    // A bin counts when it holds at least 1% of the card's blocks.
    std::array<bool, 36> occupied {};
    int occupiedCount = 0;
    for (std::size_t i = 0; i < bins.size(); ++i)
    {
        occupied[i] = bins[i] * 100 >= blocks && bins[i] > 0;
        occupiedCount += occupied[i] ? 1 : 0;
    }
    if (occupiedCount == 0)
        return report;
    int largestGap = 0;
    for (std::size_t start = 0; start < 36; ++start)
    {
        if (occupied[start])
            continue;
        int gap = 0;
        while (gap < 36 && ! occupied[(start + static_cast<std::size_t> (gap)) % 36])
            ++gap;
        largestGap = std::max (largestGap, gap);
    }
    report.hueSpreadDegrees = 360.0f - 10.0f * static_cast<float> (largestGap);
    return report;
}

std::optional<HueReport> traceCard (const videohelper::fixture3d::HolographicTradingCardScene& card,
                                    float sunAzimuth, float sunElevation, const char* label,
                                    const std::optional<grid::SceneCameraRecord>& camera = std::nullopt)
{
    const auto& operation = card.operation;
    const auto& binding = *operation.diffractionMaterial;
    std::string error;
    const auto admitted = diffractionmaterial::admit (binding.material, error);
    if (! admitted)
    {
        std::fprintf (stderr, "%s: the foil's physical material is refused: %s\n", label, error.c_str());
        return std::nullopt;
    }
    std::array<std::array<float, 4>, 4> field {};
    for (std::size_t i = 0; i < field.size(); ++i)
    {
        const auto& sample = binding.spatialFoil->grooveField[i];
        field[i] = { sample.reciprocalDirectionUv[0], sample.reciprocalDirectionUv[1],
                     sample.grooveSpacingNanometres, sample.diffractionCoverage };
    }
    arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
    inputs.linearColor = true;
    inputs.cameraOverride = camera.has_value() ? camera : operation.camera;
    auto light = *operation.light;
    light.sky.sunAzimuthDegrees = sunAzimuth;
    light.sky.sunElevationDegrees = sunElevation;
    // Diagnosis only: ARBIT_CARD_PT_SUN_INTENSITY traces a sun intensity the committed
    // request does not carry yet. The regression run leaves it unset.
    if (const char* intensity = std::getenv ("ARBIT_CARD_PT_SUN_INTENSITY"))
        light.sky.sunIntensity = static_cast<float> (std::atof (intensity));
    inputs.lightOverride = light;

    fixture::SceneEngineDraw draw;
    draw.snapshot = card.scene;
    draw.inputs = &inputs;
    const char* size = std::getenv ("ARBIT_CARD_PT_WIDTH");
    const std::uint32_t width = size != nullptr ? static_cast<std::uint32_t> (std::max (64, std::atoi (size))) : 160u;
    draw.dimensions = { width, width * 9u / 16u };
    draw.use = fixture::RenderUse::Export;
    const char* samples = std::getenv ("ARBIT_CARD_PT_SAMPLES");
    draw.exportSamples = samples != nullptr ? static_cast<std::uint32_t> (std::max (1, std::atoi (samples))) : 8u;
    draw.backend = "opengl";
    draw.diffractionObject = binding.object;
    draw.diffractionMaterial = std::make_shared<const diffractionmaterial::AdmittedDiffractionMaterialIR> (*admitted);
    draw.diffractionLighting = binding.lighting;
    draw.diffractionField = field;
    const auto engine = ptfixture::makeEngineWithUpload (&captureCard);
    fixture::RenderedFrame frame;
    if (engine == nullptr || ! engine->render (draw, frame, error))
    {
        std::fprintf (stderr, "%s: Path Trace refused the card: %s\n", label, error.c_str());
        return std::nullopt;
    }
    const auto report = cardHues (cardFilm, camera.has_value() ? projectedCard (*camera, cardFilm) : centredCard (cardFilm));
    // Diagnosis only: ARBIT_CARD_PT_FRAMES=<directory> writes each film there as a binary
    // PPM, display-encoded with gamma 2.2, top row first.
    if (const char* directory = std::getenv ("ARBIT_CARD_PT_FRAMES"))
    {
        std::string name = std::string (directory) + "/card-" + label + "-" + std::to_string (static_cast<int> (sunAzimuth)) + "-"
                         + std::to_string (static_cast<int> (sunElevation)) + ".ppm";
        std::replace (name.begin() + static_cast<std::ptrdiff_t> (std::string (directory).size()), name.end(), ' ', '-');
        if (std::FILE* file = std::fopen (name.c_str(), "wb"))
        {
            std::fprintf (file, "P6\n%u %u\n255\n", cardFilm.width, cardFilm.height);
            for (std::uint32_t row = cardFilm.height; row-- > 0;)
                for (std::uint32_t x = 0; x < cardFilm.width; ++x)
                    for (std::size_t c = 0; c < 3; ++c)
                    {
                        const float value = cardFilm.rgb[(static_cast<std::size_t> (row) * cardFilm.width + x) * 3 + c];
                        const float encoded = std::pow (std::clamp (value, 0.0f, 1.0f), 1.0f / 2.2f);
                        std::fputc (static_cast<int> (encoded * 255.0f + 0.5f), file);
                    }
            std::fclose (file);
        }
    }
    std::printf ("card %s: sun azimuth %.0f elevation %.0f, %ux%u at %u samples: saturated %.1f%% of the card, "
                 "hue spread %.0f degrees, neighbour hue agreement %.0f%%, mean luminance %.3f, sun intensity %.1f\n",
                 label, sunAzimuth, sunElevation, cardFilm.width, cardFilm.height, draw.exportSamples,
                 100.0f * report.saturatedFraction, report.hueSpreadDegrees, 100.0f * report.coherence,
                 report.meanLuminance, light.sky.sunIntensity);
    return report;
}
} // namespace

int holographicCardPathTraceTests()
{
    std::string error;
    const auto card = videohelper::fixture3d::loadHolographicTradingCardScene (error);
    cardCheck (card.has_value(), "the card's GLB and committed request load");
    if (! card)
    {
        std::fprintf (stderr, "%s\n", error.c_str());
        return cardFailures;
    }
    const auto& operation = card->operation;
    const bool attached = operation.diffractionMaterial.has_value() && operation.diffractionMaterial->spatialFoil.has_value()
                          && card->scene->objectCount > 0 && operation.diffractionMaterial->object == card->scene->objects[0].id;
    std::printf ("card grating binding: %s, object %llu\n", attached ? "attached" : "missing",
                 attached ? static_cast<unsigned long long> (operation.diffractionMaterial->object.value) : 0ull);
    cardCheck (attached, "the card's request binds its foil to the card object");
    cardCheck (operation.camera.has_value() && operation.light.has_value() && operation.light->sky.version != 0,
               "the card's request carries its camera and analytic sky");
    if (! attached || ! operation.camera || ! operation.light)
        return cardFailures;

    std::array<std::array<float, 4>, 4> field {};
    for (std::size_t i = 0; i < field.size(); ++i)
    {
        const auto& sample = operation.diffractionMaterial->spatialFoil->grooveField[i];
        field[i] = { sample.reciprocalDirectionUv[0], sample.reciprocalDirectionUv[1],
                     sample.grooveSpacingNanometres, sample.diffractionCoverage };
    }
    const auto centre = foilFieldAt (field, 0.5f, 0.5f);
    std::printf ("card foil mask: corners %.2f %.2f %.2f %.2f, centre %.3f; centre period %.0f nm\n",
                 field[0][3], field[1][3], field[2][3], field[3][3], centre[3], centre[2]);
    cardCheck (centre[3] > 0.5f, "the foil covers the card's centre");

    // The starter's sun before this change, its sun now, and that sun moved. At azimuth
    // -20 the first order lies across the card's top right corner, and a sun moved 10
    // degrees only slides it off the card's edge. Swung across the camera's axis to
    // azimuth 30, the sun sends the orders of the other corners' grooves into the camera,
    // so the bands move to other places on the card.
    traceCard (*card, 25.0f, 20.0f, "old sun");
    const auto lit = traceCard (*card, -20.0f, 8.0f, "starter sun");
    const auto moved = traceCard (*card, 30.0f, 8.0f, "moved sun");
    // The starter's floor top is 0.0445 below the card's centre.
    const auto mirrored = traceCard (*card, -20.0f, 8.0f, "floor mirror view", mirroredInFloor (*operation.camera, -0.0445f));
    // Diagnosis only: ARBIT_CARD_PT_SWEEP=1 traces the sky's suns, to tell a foil the
    // camera never sees from orders that miss this one sun.
    if (std::getenv ("ARBIT_CARD_PT_SWEEP") != nullptr)
    {
        const auto& camera = *operation.camera;
        std::printf ("card camera: position %.4f %.4f %.4f\n", camera.transform.translation.x, camera.transform.translation.y, camera.transform.translation.z);
        // Each sun from the camera and from the floor's mirror point, which the floor reflects.
        for (float elevation : { 8.0f, 14.0f, 20.0f, 26.0f, 32.0f, 38.0f, 44.0f })
            for (float azimuth = -40.0f; azimuth <= 80.0f; azimuth += 10.0f)
            {
                traceCard (*card, azimuth, elevation, "sweep direct");
                traceCard (*card, azimuth, elevation, "sweep mirror", mirroredInFloor (*operation.camera, -0.0445f));
            }
    }
    cardCheck (lit.has_value() && moved.has_value(), "Path Trace renders the card");
    if (! lit || ! moved)
        return cardFailures;
    cardCheck (lit->saturatedFraction >= 0.10f, "the starter's sun lays spectral colour on at least 10% of the card");
    cardCheck (lit->hueSpreadDegrees >= 120.0f, "the card's bands span at least 120 degrees of hue");
    // The live card at a 5 degree sun and RMS slope 0.08 passed both checks above on
    // coloured path noise, which the denoiser smoothed into a cream card.
    cardCheck (lit->coherence >= 0.5f, "the card's colour lies in bands: half its coloured neighbours share a hue");
    int changed = 0, compared = 0;
    for (std::size_t i = 0; i < std::min (lit->blockHue.size(), moved->blockHue.size()); ++i)
    {
        const int a = lit->blockHue[i], b = moved->blockHue[i];
        if (a < 0 && b < 0)
            continue;
        ++compared;
        const int difference = std::abs (a - b) % 360;
        if (a < 0 || b < 0 || std::min (difference, 360 - difference) > 30)
            ++changed;
    }
    std::printf ("card bands moved on %d of %d coloured blocks when the sun swung from azimuth -20 to 30\n", changed, compared);
    cardCheck (compared > 0 && changed * 4 >= compared, "moving the sun moves the bands on at least a quarter of the coloured card");
    // At the earlier suns (30/12, 25/20) this view held a dim sky tint with a few
    // coloured specks, so the glossy floor under the card mirrored a plain dark card.
    cardCheck (mirrored.has_value() && mirrored->saturatedFraction >= 0.05f,
               "seen from the floor's mirror point the card shows spectral colour on at least 5% of it");
    cardCheck (mirrored.has_value() && mirrored->coherence >= 0.5f,
               "the floor's reflection of the card holds bands: half its coloured neighbours share a hue");
    // The sky's blue alone saturates the card's reflection, so bands also need more than
    // one hue there, as on the card itself.
    cardCheck (mirrored.has_value() && mirrored->hueSpreadDegrees >= 120.0f,
               "the floor's reflection of the card spans at least 120 degrees of hue");
    std::printf ("holographic card: %d checks, %d failures\n", cardChecks, cardFailures);
    return cardFailures;
}
