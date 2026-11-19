// Geometry wired straight into Render 3D, lit by Render 3D's Camera and Light
// (slice georoute). A cube stands on a floor slab under a sun from the right.
// The lighting rig rides in the plan's render settings, and executeNativeGeometry
// hands it to FixtureSceneRenderer for Raster, Realtime and Path Trace on the
// OpenGL backend. Every engine must shade the sunlit top above the unlit front
// face and leave the cube's shadow on the floor. Path Trace alone carries the
// floor's bounce, so the bottom of the front face is brighter than its top there.
// An Environment Light added beside the sun fills the shadow in all three engines.
//
// ARBIT_GEOROUTE_PNG=dir writes every frame there as <case>-<engine>.png.
//
// Slice geolight2: the three direct-Geometry starters (Spectral Ring and Grid
// Instancer, Score-Deformed Imported Mesh, Audio Surface Instances) draw in all
// three engines with the default lighting and again under the starter's Camera
// and Environment Light, with luma percentiles for each. ARBIT_GEOLIGHT_PNG=dir
// writes <starter>-<engine>-<before|fill|fill2|sky|skyhidden|key4>.png. ARBIT_GEOLIGHT_PLANS=dir also
// draws every compiled starter plan (<starter>-<engine>.hex, written by the
// plugin's DirectGeometryRenderEngineTests through ARBIT_DIRECT_GEOMETRY_PLAN_DIR)
// as <starter>-<engine>-plan.png.
// It skips with 77 when no hardware OpenGL 4.3 context exists.
//
// The same context runs cross_engine_exposure_gl_tests.cpp (slice rasterexpo).

#include "../src/geometry_core_backend.h"
#include "../src/fixture_scene_renderer.h"
#include "../src/gl_loader.h"
#include "../src/gpu_backend/backend.h"
#include "../../shared/GeometryCoreRenderSettings.h"
#include "support/direct_starter_scenes.h"
#include "support/lit_cube_on_floor.h"

#include <GLFW/glfw3.h>
#include <GL/gl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

int crossEngineExposureTests();

namespace
{
using namespace HarmonicMIDI::grid;

int checks = 0;
int failures = 0;

void check (bool condition, const std::string& message)
{
    ++checks;
    if (! condition)
    {
        ++failures;
        std::fprintf (stderr, "FAIL: %s\n", message.c_str());
    }
}

constexpr GLenum kGlHalfFloat = 0x140B;
constexpr std::uint32_t kWidth = 192, kHeight = 108;

float halfToFloat (std::uint16_t bits)
{
    const std::uint32_t sign = (bits >> 15) & 1u, exponent = (bits >> 10) & 31u, mantissa = bits & 1023u;
    float value = exponent == 0 ? std::ldexp (static_cast<float> (mantissa), -24)
                : exponent == 31 ? INFINITY
                : std::ldexp (static_cast<float> (mantissa | 1024u), static_cast<int> (exponent) - 25);
    return sign != 0 ? -value : value;
}

std::uint8_t encodeByte (float linear)
{
    const float c = std::clamp (linear, 0.0f, 1.0f);
    const float encoded = c <= 0.0031308f ? 12.92f * c : 1.055f * std::pow (c, 1.0f / 2.4f) - 0.055f;
    return static_cast<std::uint8_t> (encoded * 255.0f + 0.5f);
}

// A frame as the compositor shows it: RGB bytes, top row first.
struct Shown
{
    std::uint32_t width = 0, height = 0;
    std::vector<std::uint8_t> rgb;
    // Mean luma of a 5x5 patch (radius 2), so Path Trace noise does not decide a check.
    float luma (int cx, int cy, int radius = 2) const
    {
        float sum = 0;
        int count = 0;
        for (int y = cy - radius; y <= cy + radius; ++y)
            for (int x = cx - radius; x <= cx + radius; ++x)
            {
                if (x < 0 || y < 0 || x >= static_cast<int> (width) || y >= static_cast<int> (height)) continue;
                const std::size_t i = (static_cast<std::size_t> (y) * width + static_cast<std::size_t> (x)) * 3;
                sum += 0.2126f * rgb[i] + 0.7152f * rgb[i + 1] + 0.0722f * rgb[i + 2];
                ++count;
            }
        return count > 0 ? sum / static_cast<float> (count) : 0.0f;
    }
};

bool readFrame (const arbitgpu::NativeFixtureSceneFrame& frame, Shown& out, std::string& error)
{
    glFinish();
    // Every engine's Color is read from the texture the compositor samples, in its row order.
    arbitgpu::NativeRawPassPixels pixels;
    const auto descriptor = frame.colorTextureDescriptor();
    const bool halfTexture = descriptor.format == arbitgpu::NativeTexturePixelFormat::Rgba16Float;
    if (! halfTexture && descriptor.format != arbitgpu::NativeTexturePixelFormat::Rgba8Unorm)
    {
        error = "unexpected Color format";
        return false;
    }
    pixels.format = descriptor.format;
    pixels.width = descriptor.width;
    pixels.height = descriptor.height;
    pixels.bytes.assign (static_cast<std::size_t> (descriptor.width) * descriptor.height * (halfTexture ? 8 : 4), 0);
    glBindTexture (GL_TEXTURE_2D, static_cast<GLuint> (descriptor.textureViewHandle));
    glPixelStorei (GL_PACK_ALIGNMENT, 1);
    glGetTexImage (GL_TEXTURE_2D, 0, GL_RGBA, halfTexture ? kGlHalfFloat : GL_UNSIGNED_BYTE, pixels.bytes.data());
    glBindTexture (GL_TEXTURE_2D, 0);
    const bool bottomFirst = descriptor.rowOrder == arbitgpu::NativeTextureRowOrder::BottomFirst;
    const bool half = pixels.format == arbitgpu::NativeTexturePixelFormat::Rgba16Float;
    const std::size_t stride = half ? 8 : 4, count = static_cast<std::size_t> (pixels.width) * pixels.height;
    if (pixels.bytes.size() != count * stride)
    {
        error = "unexpected Color size";
        return false;
    }
    out.width = pixels.width;
    out.height = pixels.height;
    out.rgb.assign (count * 3, 0);
    for (std::uint32_t y = 0; y < pixels.height; ++y)
        for (std::uint32_t x = 0; x < pixels.width; ++x)
        {
            const std::uint32_t row = bottomFirst ? pixels.height - 1u - y : y;
            const std::size_t source = (static_cast<std::size_t> (row) * pixels.width + x) * stride;
            std::uint8_t* target = out.rgb.data() + (static_cast<std::size_t> (y) * pixels.width + x) * 3;
            for (std::size_t c = 0; c < 3; ++c)
            {
                if (! half) { target[c] = pixels.bytes[source + c]; continue; }
                std::uint16_t bits = 0;
                std::memcpy (&bits, pixels.bytes.data() + source + c * 2, 2);
                target[c] = encodeByte (halfToFloat (bits));
            }
        }
    return true;
}

// A stored-block PNG, so the frames need no image library.
void writePng (const std::string& path, const Shown& frame)
{
    std::vector<std::uint8_t> raw;
    for (std::uint32_t y = 0; y < frame.height; ++y)
    {
        raw.push_back (0);
        const auto row = frame.rgb.begin() + static_cast<std::ptrdiff_t> (y * frame.width * 3);
        raw.insert (raw.end(), row, row + static_cast<std::ptrdiff_t> (frame.width * 3));
    }
    std::vector<std::uint8_t> z { 0x78, 0x01 };
    std::uint32_t a = 1, b = 0;
    for (const auto byte : raw) { a = (a + byte) % 65521u; b = (b + a) % 65521u; }
    for (std::size_t at = 0; at < raw.size(); at += 65535)
    {
        const std::size_t n = std::min<std::size_t> (65535, raw.size() - at);
        z.push_back (at + n == raw.size() ? 1 : 0);
        z.push_back (static_cast<std::uint8_t> (n)); z.push_back (static_cast<std::uint8_t> (n >> 8));
        z.push_back (static_cast<std::uint8_t> (~n)); z.push_back (static_cast<std::uint8_t> (~n >> 8));
        z.insert (z.end(), raw.begin() + static_cast<std::ptrdiff_t> (at), raw.begin() + static_cast<std::ptrdiff_t> (at + n));
    }
    const std::uint32_t adler = (b << 16) | a;
    for (int s = 24; s >= 0; s -= 8) z.push_back (static_cast<std::uint8_t> (adler >> s));
    std::FILE* file = std::fopen (path.c_str(), "wb");
    if (file == nullptr) return;
    const auto put32 = [file] (std::uint32_t v) { for (int s = 24; s >= 0; s -= 8) std::fputc (static_cast<int> ((v >> s) & 255u), file); };
    const auto chunk = [&] (const char* type, const std::vector<std::uint8_t>& data) {
        put32 (static_cast<std::uint32_t> (data.size()));
        std::vector<std::uint8_t> body (type, type + 4);
        body.insert (body.end(), data.begin(), data.end());
        std::fwrite (body.data(), 1, body.size(), file);
        std::uint32_t crc = 0xffffffffu;
        for (const auto byte : body)
        {
            crc ^= byte;
            for (int k = 0; k < 8; ++k) crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
        }
        put32 (~crc);
    };
    std::fwrite ("\x89PNG\r\n\x1a\n", 1, 8, file);
    std::vector<std::uint8_t> header;
    for (const auto v : { frame.width, frame.height }) for (int s = 24; s >= 0; s -= 8) header.push_back (static_cast<std::uint8_t> (v >> s));
    header.insert (header.end(), { 8, 2, 0, 0, 0 });
    chunk ("IHDR", header);
    chunk ("IDAT", z);
    chunk ("IEND", {});
    std::fclose (file);
}

struct Engine
{
    const char* name;
    visualimportedscenerender::RenderEngine kind;
};

bool draw (arbitgpu::NativeFixtureSceneBackend& backend, const videowire::geometry::PortContract& contract,
           const videowire::geometry::ValueDescriptor& value, const videowire::geometry::RenderSettings& settings,
           std::uint64_t owner, const std::string& label, Shown& frame)
{
    using namespace videohelper::geometry;
    std::string error;
    const auto text = videowire::geometry::encodeRenderSettings (settings);
    const auto decoded = videowire::geometry::decodeRenderSettings (text, error);
    check (decoded.has_value() && decoded->lighting != nullptr, label + ": the settings carry the rig: " + error);
    if (! decoded) return false;
    const auto admitted = videowire::geometry::admitValue (value, contract, error);
    check (admitted.has_value(), label + ": the value admits: " + error);
    if (! admitted) return false;
    NativeGeometryCoreCapabilitySource capabilities (backend);
    GeometryCorePlanRuntime runtime (capabilities);
    auto plan = runtime.admitExport ({1, 1, 1, owner, 1, PlanUse::exportRender},
        videowire::geometry::lowerRuntimePlan (contract, *admitted, {}, {}, false, text), {}, {}, error);
    check (plan.has_value(), label + ": the plan admits: " + error);
    if (! plan) return false;
    const auto execution = executeNativeGeometry (backend, *plan, kWidth, kHeight, error, false, {}, nullptr, {}, 0.0,
        {}, nullptr, 0.0, nullptr, {}, {}, arbitgpu::NativeFixturePrepareUse::Blocking, nullptr, nullptr, nullptr,
        &*decoded, 8170, 7);
    check (execution.has_value() && execution->frame != nullptr, label + " renders: " + error);
    if (! execution || ! execution->frame) return false;
    const bool read = readFrame (*execution->frame, frame, error);
    check (read, label + " reads back: " + error);
    if (const char* dir = std::getenv ("ARBIT_GEOROUTE_PNG"); dir != nullptr && read)
        writePng (std::string (dir) + "/" + label + ".png", frame);
    return read;
}

void litCubeTests()
{
    auto& backend = arbitgpu::nativeFixtureSceneBackend();
    using namespace videohelper::litcube;
    videowire::geometry::PortContract contract;
    std::string error;
    const auto value = cubeOnFloor (contract, error);
    check (value.has_value(), "the cube and floor lower: " + error);
    if (! value) return;
    const Engine engines[] {
        { "raster", visualimportedscenerender::RenderEngine::Raster },
        { "realtime", visualimportedscenerender::RenderEngine::Realtime },
        { "pathtrace", visualimportedscenerender::RenderEngine::PathTrace },
    };
    const auto top = project (kWidth, kHeight, {0, 0.5f, 0}), frontTop = project (kWidth, kHeight, {0, 0.35f, 0.5f}),
        frontBottom = project (kWidth, kHeight, {0, -0.4f, 0.5f}), floorLit = project (kWidth, kHeight, {1.6f, -0.5f, 0.3f}),
        floorShadow = project (kWidth, kHeight, {-0.85f, -0.5f, 0.2f});
    std::array<float, 3> frontGradient {}, frontBottomLuma {};
    std::uint64_t owner = 900;
    for (std::size_t e = 0; e < std::size (engines); ++e)
    {
        const auto& engine = engines[e];
        videowire::geometry::RenderSettings settings;
        settings.engine = engine.kind;
        settings.exportSamples = 128;
        settings.lighting = rig (false);
        check (settings.lighting != nullptr, "the lighting rig is a valid scene record");
        Shown lit;
        if (! draw (backend, contract, *value, settings, ++owner, std::string ("cube-") + engine.name, lit)) continue;
        const float lTop = lit.luma (top.x, top.y), lFrontTop = lit.luma (frontTop.x, frontTop.y),
            lFrontBottom = lit.luma (frontBottom.x, frontBottom.y),
            lFloorLit = lit.luma (floorLit.x, floorLit.y), lFloorShadow = lit.luma (floorShadow.x, floorShadow.y);
        std::printf ("[georoute] %-9s top %.1f front %.1f/%.1f floor lit %.1f shadow %.1f\n", engine.name,
                     lTop, lFrontTop, lFrontBottom, lFloorLit, lFloorShadow);
        const std::string name = engine.name;
        check (lTop > lFrontTop + 25, name + ": the sunlit top is brighter than the unlit front face");
        check (lFloorLit > lFloorShadow + 25, name + ": the cube's shadow darkens the floor");
        frontGradient[e] = lFrontBottom - lFrontTop;
        frontBottomLuma[e] = lFrontBottom;

        settings.lighting = rig (true);
        check (settings.lighting != nullptr, "the sky rig is a valid scene record");
        Shown sky;
        if (! draw (backend, contract, *value, settings, ++owner, std::string ("cube-sky-") + engine.name, sky)) continue;
        const float skyShadow = sky.luma (floorShadow.x, floorShadow.y);
        std::printf ("[georoute] %-9s shadow under the sky %.1f\n", engine.name, skyShadow);
        check (skyShadow > lFloorShadow + 10, name + ": the Environment Light fills the cube's shadow");
    }
    std::printf ("[georoute] front face bottom minus top: raster %.1f realtime %.1f pathtrace %.1f\n",
                 frontGradient[0], frontGradient[1], frontGradient[2]);
    // The lit floor bounces light onto the unlit front face only in Path Trace,
    // most strongly near its bottom edge. Raster and Realtime keep it flat ambient.
    // (Measured on an RTX 4070, 128 samples: bottom 11.8 against 5.0, gradient 3.0 against 0.)
    check (frontBottomLuma[2] > frontBottomLuma[0] + 3, "Path Trace bounces the sunlit floor onto the front face");
    check (frontGradient[2] > frontGradient[0] + 1.5f, "the bounce is strongest beside the floor");
}


// Luma of display bytes, 0 to 255: percentiles over the whole frame and over the
// pixels that differ from the corner, which shows the background.
struct Spread
{
    float p5 = 0, p95 = 0, subjectP5 = 0, subjectP50 = 0, subjectP95 = 0, mean = 0;
    std::size_t subject = 0;
};

// mask, when given, picks the subject as the pixels that differ from its own corner.
Spread spreadOf (const Shown& frame, const Shown* mask = nullptr)
{
    std::vector<float> all, subject;
    const auto luma = [&frame] (std::size_t i) {
        return 0.2126f * frame.rgb[i] + 0.7152f * frame.rgb[i + 1] + 0.0722f * frame.rgb[i + 2];
    };
    const float corner = frame.rgb.size() >= 3 ? luma (0) : 0.0f;
    for (std::size_t i = 0; i + 2 < frame.rgb.size(); i += 3)
    {
        all.push_back (luma (i));
        const auto& m = mask != nullptr && mask->rgb.size() == frame.rgb.size() ? *mask : frame;
        if (std::abs (m.rgb[i] - m.rgb[0]) + std::abs (m.rgb[i + 1] - m.rgb[1])
                + std::abs (m.rgb[i + 2] - m.rgb[2]) > 12)
            subject.push_back (luma (i));
    }
    const auto at = [] (std::vector<float>& v, float q) {
        if (v.empty()) return 0.0f;
        std::sort (v.begin(), v.end());
        return v[std::min (v.size() - 1, static_cast<std::size_t> (q * static_cast<float> (v.size())))];
    };
    Spread out;
    out.p5 = at (all, 0.05f);
    out.p95 = at (all, 0.95f);
    out.subjectP5 = at (subject, 0.05f);
    out.subjectP50 = at (subject, 0.5f);
    out.subjectP95 = at (subject, 0.95f);
    out.subject = subject.size();
    for (const auto v : all) out.mean += v;
    out.mean = all.empty() ? 0.0f : out.mean / static_cast<float> (all.size());
    (void) corner;
    return out;
}

constexpr std::uint32_t kStarterWidth = 480, kStarterHeight = 270;

// A flat level at or above 0 replaces the animated bands, which puts every band at one level.
videohelper::geometry::SpectrumEvaluation starterSpectrum (double seconds, double flat = -1)
{
    videohelper::geometry::SpectrumEvaluation input;
    const auto bandsAt = [flat] (double t) {
        videowire::geometry::spectrum::Bands bands {};
        for (std::size_t i = 0; i < bands.size(); ++i)
            bands[i] = flat >= 0 ? static_cast<float> (flat)
                                 : static_cast<float> (0.35 + 0.3 * std::sin (t * 3.1 + static_cast<double> (i) * 0.7));
        return bands;
    };
    input.featuresAt = bandsAt;
    input.timeSeconds = seconds;
    input.bands = bandsAt (seconds);
    return input;
}

// One starter frame at 0.9 s under the given settings, read back as the compositor shows it.
bool drawStarter (arbitgpu::NativeFixtureSceneBackend& backend, const videohelper::geometry::RuntimeAdmission& plan,
                  const videowire::geometry::RenderSettings& settings, bool spectral, bool score,
                  const std::string& label, Shown& frame, double flatBands = -1)
{
    std::string error;
    arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
    if (score) inputs.canonicalBlockCFrame = scoreFieldFrame();
    auto spectrum = starterSpectrum (0.9, flatBands);
    const auto execution = videohelper::geometry::executeNativeGeometry (backend, plan, kStarterWidth, kStarterHeight,
        error, false, std::move (inputs), spectral ? &spectrum : nullptr, {}, 0.9, {}, nullptr, 0.0, nullptr, {}, {},
        arbitgpu::NativeFixturePrepareUse::Blocking, nullptr, nullptr, nullptr, &settings, 8170, 7);
    check (execution.has_value() && execution->frame != nullptr, label + " renders: " + error);
    if (! execution || ! execution->frame) return false;
    const bool read = readFrame (*execution->frame, frame, error);
    check (read, label + " reads back: " + error);
    if (const char* dir = std::getenv ("ARBIT_GEOLIGHT_PNG"); dir != nullptr && read)
        writePng (std::string (dir) + "/" + label + ".png", frame);
    return read;
}

void report (const std::string& label, const Shown& frame)
{
    const auto s = spreadOf (frame);
    std::printf ("[geolight] %-34s luma p5 %5.1f p95 %5.1f mean %5.1f | subject (%6zu px) p5 %5.1f p95 %5.1f\n",
                 label.c_str(), s.p5, s.p95, s.mean, s.subject, s.subjectP5, s.subjectP95);
}

void starterTests()
{
    using namespace videohelper::starterscenes;
    auto& backend = arbitgpu::nativeFixtureSceneBackend();
    const Engine engines[] {
        { "raster", visualimportedscenerender::RenderEngine::Raster },
        { "realtime", visualimportedscenerender::RenderEngine::Realtime },
        { "pathtrace", visualimportedscenerender::RenderEngine::PathTrace },
    };
    // The cameras each starter wires, as its nodes lower them. The rigs differ in light only.
    Rig ring, card, surface;
    ring.eye = { 0, 0, 6.5f };
    card.eye = { 0, 0, 3.4f };
    card.fovDegrees = 40;
    surface.eye = { 0, 0, 4.5f };
    std::vector<Rig> lights;
    {
        Rig fill;
        fill.environment = { 0.55f, 0.62f, 0.75f };
        fill.environmentIntensity = 1.0f;
        lights.push_back (fill);
        auto bright = fill;
        bright.name = "fill2";
        bright.environmentIntensity = 2.0f;
        lights.push_back (bright);
        auto sky = fill;
        sky.name = "sky";
        sky.sunIntensity = 6.0f;
        lights.push_back (sky);
        auto dark = sky;
        dark.name = "skyhidden";
        dark.cameraVisible = false;
        lights.push_back (dark);
        Rig key;
        key.name = "key4";
        key.key = 4.0f;
        lights.push_back (key);
    }
    struct Case { std::optional<videohelper::starterscenes::Scene> (*build) (std::string&); Rig camera; };
    const Case starters[] { { spectralRing, ring }, { scoreCard, card }, { audioSurface, surface } };
    std::uint64_t owner = 2000;
    for (const auto& starter : starters)
    {
        std::string error;
        const auto scene = starter.build (error);
        check (scene.has_value(), "the starter scene lowers: " + error);
        if (! scene) continue;
        for (const auto& engine : engines)
            for (std::size_t variant = 0; variant <= lights.size(); ++variant)
            {
                // Variant 0 is the default lighting and camera the starters have today.
                Rig rig = variant == 0 ? Rig {} : lights[variant - 1];
                if (variant > 0)
                {
                    rig.eye = starter.camera.eye;
                    rig.fovDegrees = starter.camera.fovDegrees;
                }
                videowire::geometry::RenderSettings settings;
                settings.engine = engine.kind;
                settings.exportSamples = 128;
                if (variant > 0)
                {
                    settings.lighting = lowered (rig);
                    check (settings.lighting != nullptr, scene->name + ": the rig is a valid scene record");
                }
                const auto text = videowire::geometry::encodeRenderSettings (settings);
                const auto decoded = videowire::geometry::decodeRenderSettings (text, error);
                check (decoded.has_value(), scene->name + ": the settings decode: " + error);
                if (! decoded) continue;
                const auto admitted = videowire::geometry::admitValue (scene->value, scene->contract, error);
                check (admitted.has_value(), scene->name + ": the value admits: " + error);
                if (! admitted) continue;
                videohelper::geometry::NativeGeometryCoreCapabilitySource capabilities (backend);
                videohelper::geometry::GeometryCorePlanRuntime runtime (capabilities);
                auto plan = runtime.admitExport ({ 1, 1, 1, ++owner, 1, videohelper::geometry::PlanUse::exportRender },
                    videowire::geometry::lowerRuntimePlan (scene->contract, *admitted, {}, {}, false, text), {}, {}, error);
                check (plan.has_value(), scene->name + ": the plan admits: " + error);
                if (! plan) continue;
                Shown frame;
                const std::string label = scene->name + "-" + engine.name + "-" + (variant == 0 ? "before" : rig.name);
                if (drawStarter (backend, *plan, *decoded, scene->spectral, scene->score, label, frame))
                    report (label, frame);
            }
    }
    const char* plans = std::getenv ("ARBIT_GEOLIGHT_PLANS");
    if (plans == nullptr) return;
    for (const auto& entry : std::filesystem::directory_iterator (plans))
    {
        if (entry.path().extension() != ".hex") continue;
        const auto name = entry.path().stem().string();
        std::ifstream in (entry.path());
        std::stringstream text;
        text << in.rdbuf();
        std::string error;
        const auto bytes = videowire::geometry::decodeLoweredPlanText (text.str(), error);
        const auto lowered = bytes ? videowire::geometry::decodeLoweredRuntimePlan (*bytes, error) : std::nullopt;
        const auto settings = lowered ? videowire::geometry::decodeRenderSettings (lowered->renderSettings, error) : std::nullopt;
        check (settings.has_value(), name + ": the compiled starter plan decodes: " + error);
        if (! settings) continue;
        videohelper::geometry::NativeGeometryCoreCapabilitySource capabilities (backend);
        videohelper::geometry::GeometryCorePlanRuntime runtime (capabilities);
        auto plan = runtime.admitExport ({ 1, 1, 1, ++owner, 1, videohelper::geometry::PlanUse::exportRender }, *bytes, {}, {}, error);
        check (plan.has_value(), name + ": the compiled starter plan admits: " + error);
        if (! plan) continue;
        Shown frame;
        const bool spectral = name.find ("spectral") != std::string::npos || name.find ("audio-surface") != std::string::npos;
        const bool score = name.find ("score") != std::string::npos;
        if (drawStarter (backend, *plan, *settings, spectral, score, name + "-plan", frame)) report (name + "-plan", frame);
    }
}

// Slice geofloor: the spectral ring and the audio surface on their floor stage in
// each engine. A frame reports the subject's luma spread, the floor in the
// objects' shadow just behind their lowest members (contact) and the open floor
// in front of them (open), and how far Path Trace and Realtime stray from Raster.
struct FloorCase
{
    std::optional<videohelper::starterscenes::Scene> (*build) (std::string&);
    videohelper::starterscenes::Stage stage;
    // World points on the floor in the lowest members' shadow, and on the open floor.
    std::vector<std::array<float, 3>> contact, open;
    // The least p95 - p5 luma spread over the objects in Raster and Realtime. The ring's bars
    // differ in band level, so its frame spans far more than the surface's cubes, whose front
    // faces all share one orientation and whose resting emission is a fixed 0.5 luminance.
    float minSpread = 60;
    // Half the edge of a cube in the starter's instances, for the face reads.
    float cubeHalf = 0.0f;
};

std::array<int, 2> stageProject (const videohelper::starterscenes::Stage& stage, std::array<float, 3> p)
{
    const float t = stage.tiltDegrees * 3.14159265f / 180.0f;
    const float y = stage.yawDegrees * 3.14159265f / 180.0f;
    const float ex = p[0] - stage.eye[0], dy = p[1] - stage.eye[1], ez = p[2] - stage.eye[2];
    // The camera's yaw undone first, then its tilt.
    const float dx = ex * std::cos (y) - ez * std::sin (y), dz = ex * std::sin (y) + ez * std::cos (y);
    const float yc = dy * std::cos (t) - dz * std::sin (t), zc = dy * std::sin (t) + dz * std::cos (t);
    const float f = std::tan (stage.fovDegrees * 3.14159265f / 360.0f);
    const float aspect = static_cast<float> (kStarterWidth) / static_cast<float> (kStarterHeight);
    const float nx = dx / -zc / (f * aspect), ny = yc / -zc / f;
    return { static_cast<int> ((nx + 1.0f) * 0.5f * kStarterWidth), static_cast<int> ((1.0f - ny) * 0.5f * kStarterHeight) };
}

// The faces of each cube in a starter's instances, told apart by where the key
// stands: a face is lit when its normal faces the key and shadowed when it
// faces away. Only faces the camera sees, that cover 20 pixels or more and
// are at least 2.5 pixels deep count, each read at its centre pixel. The depth test
// drops the slivers of a top face seen edge-on, where a read at the centre
// lands on the neighbouring surface.
struct FaceMeans
{
    float lit = 0, shadowed = 0;
    int litCount = 0, shadowedCount = 0;
    // Mean luma and count of the faces that point along +X, -X, +Y, +Z (index 0 to 3).
    std::array<float, 4> byNormal {};
    std::array<int, 4> normalCount {};
};

FaceMeans faceMeans (const Shown& frame, const videohelper::starterscenes::Stage& stage,
                     const videowire::geometry::ValueDescriptor& value, float half)
{
    const float e = stage.keyElevation * 3.14159265f / 180.0f, a = stage.keyAzimuth * 3.14159265f / 180.0f;
    const std::array<float, 3> toLight { std::cos (e) * std::sin (a), std::sin (e), std::cos (e) * std::cos (a) };
    FaceMeans out;
    const auto& instances = std::get<videowire::geometry::InstancesData> (value.data).instances;
    for (const auto& instance : instances)
    {
        const std::array<float, 3> centre { instance.transform.translation.x, instance.transform.translation.y,
                                            instance.transform.translation.z };
        for (int axis = 0; axis < 3; ++axis)
            for (int sign = -1; sign <= 1; sign += 2)
            {
                std::array<float, 3> n {}, p = centre;
                n[static_cast<std::size_t> (axis)] = static_cast<float> (sign);
                p[static_cast<std::size_t> (axis)] += static_cast<float> (sign) * half;
                if (n[0] * (stage.eye[0] - p[0]) + n[1] * (stage.eye[1] - p[1]) + n[2] * (stage.eye[2] - p[2]) <= 0) continue;
                const int u = (axis + 1) % 3, v = (axis + 2) % 3;
                // The face's corners in winding order, its area by the shoelace sum and its centre.
                std::array<std::array<int, 2>, 4> corners {};
                for (int corner = 0; corner < 4; ++corner)
                {
                    auto q = p;
                    q[static_cast<std::size_t> (u)] += (corner & 1 ? half : -half);
                    q[static_cast<std::size_t> (v)] += (corner & 2 ? half : -half);
                    corners[static_cast<std::size_t> (corner == 2 ? 3 : corner == 3 ? 2 : corner)] = stageProject (stage, q);
                }
                float twiceArea = 0, perimeter = 0;
                int cx = 0, cy = 0;
                for (std::size_t i = 0; i < 4; ++i)
                {
                    const auto& a = corners[i];
                    const auto& b = corners[(i + 1) % 4];
                    twiceArea += static_cast<float> (a[0] * b[1] - b[0] * a[1]);
                    perimeter += std::hypot (static_cast<float> (a[0] - b[0]), static_cast<float> (a[1] - b[1]));
                    cx += a[0]; cy += a[1];
                }
                if (std::abs (twiceArea) * 0.5f < 20.0f || std::abs (twiceArea) < 2.5f * perimeter) continue;
                const float luma = frame.luma (cx / 4, cy / 4, 0);
                const float facing = n[0] * toLight[0] + n[1] * toLight[1] + n[2] * toLight[2];
                const int slot = axis == 0 ? (sign > 0 ? 0 : 1) : axis == 1 ? 2 : 3;
                if (axis != 1 || sign > 0) { out.byNormal[static_cast<std::size_t> (slot)] += luma; ++out.normalCount[static_cast<std::size_t> (slot)]; }
                if (facing > 0.3f) { out.lit += luma; ++out.litCount; }
                else if (facing <= 0.0f) { out.shadowed += luma; ++out.shadowedCount; }
            }
    }
    for (std::size_t i = 0; i < 4; ++i)
        if (out.normalCount[i] > 0) out.byNormal[i] /= static_cast<float> (out.normalCount[i]);
    if (out.litCount > 0) out.lit /= static_cast<float>(out.litCount);
    if (out.shadowedCount > 0) out.shadowed /= static_cast<float>(out.shadowedCount);
    return out;
}

void floorStageTests()
{
    using namespace videohelper::starterscenes;
    auto& backend = arbitgpu::nativeFixtureSceneBackend();
    const Engine engines[] {
        { "raster", visualimportedscenerender::RenderEngine::Raster },
        { "realtime", visualimportedscenerender::RenderEngine::Realtime },
        { "pathtrace", visualimportedscenerender::RenderEngine::PathTrace },
    };
    // The key's horizontal pull per unit of drop, from Transform 3D's X then Y rotation.
    const auto shadowAt = [] (const Stage& stage, std::array<float, 3> p)
    {
        const float e = stage.keyElevation * 3.14159265f / 180.0f, a = stage.keyAzimuth * 3.14159265f / 180.0f;
        const std::array<float, 3> toLight { std::cos (e) * std::sin (a), std::sin (e), std::cos (e) * std::cos (a) };
        const float drop = (p[1] - stage.floorY) / toLight[1];
        return std::array<float, 3> { p[0] - drop * toLight[0], stage.floorY, p[2] - drop * toLight[2] };
    };
    std::vector<FloorCase> cases;
    {
        FloorCase ring { spectralRing, {}, {}, {} };
        ring.stage.tiltDegrees = 10;
        ring.stage.floorY = -1.9f;
        ring.stage.yawDegrees = 25;
        ring.stage.eye = { 7.0f * std::sin (ring.stage.yawDegrees * 3.14159265f / 180.0f), 1.2f, 7.0f * std::cos (ring.stage.yawDegrees * 3.14159265f / 180.0f) };
        // The bars nearest the bottom of the ring, at rest 0.12 tall.
        for (int i = 44; i <= 52; ++i)
        {
            const float angle = 6.2831853f * static_cast<float> (i) / 64.0f;
            ring.contact.push_back (shadowAt (ring.stage, { 1.6f * std::cos (angle), 1.6f * std::sin (angle) - 0.06f, 0 }));
            ring.open.push_back ({ 1.6f * std::cos (angle), ring.stage.floorY, 1.4f });
        }
        cases.push_back (ring);
        FloorCase surface { audioSurface, {}, {}, {}, 15 };
        surface.stage.tiltDegrees = 8;
        surface.stage.floorY = -1.4f;
        surface.stage.yawDegrees = 25;
        surface.stage.eye = { 4.8f * std::sin (surface.stage.yawDegrees * 3.14159265f / 180.0f), 0.8f, 4.8f * std::cos (surface.stage.yawDegrees * 3.14159265f / 180.0f) };
        // The shadow of the bottom row's lower edge under each of the six columns, and open floor to
        // the right of the surface, which the key lights from that side and which the frame still shows.
        for (float x : { -1.0f, -0.6f, -0.2f, 0.2f, 0.6f, 1.0f })
            surface.contact.push_back (shadowAt (surface.stage, { x, -1.08f, 0 }));
        for (float x : { 1.6f, 2.1f, 2.6f })
            for (float z : { -0.5f, -0.2f })
                surface.open.push_back ({ x, surface.stage.floorY, z });
        surface.cubeHalf = 0.08f;
        cases.push_back (surface);
    }
    std::uint64_t owner = 5000;
    for (const auto& item : cases)
    {
        std::string error;
        const auto scene = item.build (error);
        check (scene.has_value(), "the floor starter scene lowers: " + error);
        if (! scene) continue;
        std::vector<Shown> frames;
        for (const auto& engine : engines)
        {
            videowire::geometry::RenderSettings settings;
            settings.engine = engine.kind;
            settings.exportSamples = 128;
            settings.lighting = staged (item.stage);
            check (settings.lighting != nullptr, scene->name + ": the stage is a valid scene record");
            const auto text = videowire::geometry::encodeRenderSettings (settings);
            const auto decoded = videowire::geometry::decodeRenderSettings (text, error);
            check (decoded.has_value() && decoded->lighting && decoded->lighting->objectCount == 1,
                   scene->name + ": the stage decodes with its floor: " + error);
            if (! decoded) continue;
            const auto admitted = videowire::geometry::admitValue (scene->value, scene->contract, error);
            if (! admitted) { check (false, scene->name + ": the value admits: " + error); continue; }
            videohelper::geometry::NativeGeometryCoreCapabilitySource capabilities (backend);
            videohelper::geometry::GeometryCorePlanRuntime runtime (capabilities);
            auto plan = runtime.admitExport ({ 1, 1, 1, ++owner, 1, videohelper::geometry::PlanUse::exportRender },
                videowire::geometry::lowerRuntimePlan (scene->contract, *admitted, {}, {}, false, text), {}, {}, error);
            if (! plan) { check (false, scene->name + ": the plan admits: " + error); continue; }
            Shown frame;
            const std::string label = scene->name + "-" + engine.name + "-floor";
            if (! drawStarter (backend, *plan, *decoded, scene->spectral, scene->score, label, frame)) continue;
            // The objects' pixels: where the same stage with its floor dropped out
            // of view differs from the background.
            auto away = item.stage;
            away.floorY = -60;
            auto awaySettings = *decoded;
            awaySettings.lighting = staged (away);
            Shown alone;
            if (! drawStarter (backend, *plan, awaySettings, scene->spectral, scene->score, label + "-alone", alone)) continue;
            const auto spread = spreadOf (frame, &alone);
            float contact = 0, open = 0;
            for (const auto& p : item.contact) { const auto at = stageProject (item.stage, p); contact += frame.luma (at[0], at[1]); }
            for (const auto& p : item.open) { const auto at = stageProject (item.stage, p); open += frame.luma (at[0], at[1]); }
            contact /= static_cast<float> (item.contact.size());
            open /= static_cast<float> (item.open.size());
            std::printf ("[geofloor] %-30s subject (%6zu px) p5 %5.1f p50 %5.1f p95 %5.1f spread %5.1f | floor contact %5.1f open %5.1f\n",
                         label.c_str(), spread.subject, spread.subjectP5, spread.subjectP50, spread.subjectP95,
                         spread.subjectP95 - spread.subjectP5, contact, open);
            if (scene->name == "audio-surface")
            {
                // A cube's lit face against its shadowed one, in the engines that shade by the key alone.
                const auto faces = faceMeans (frame, item.stage, scene->value, item.cubeHalf);
                if (engine.kind != visualimportedscenerender::RenderEngine::PathTrace)
                    check (faces.lit - faces.shadowed >= 25, label + ": a lit face is 25 luma levels above a shadowed one");
                std::printf ("[geofloor] %-30s faces lit %5.1f (%d) shadowed %5.1f (%d) difference %5.1f\n", label.c_str(),
                             faces.lit, faces.litCount, faces.shadowed, faces.shadowedCount, faces.lit - faces.shadowed);
                std::printf ("[geofloor] %-30s faces +x %5.1f (%d) -x %5.1f (%d) top %5.1f (%d) front %5.1f (%d)\n", label.c_str(),
                             faces.byNormal[0], faces.normalCount[0], faces.byNormal[1], faces.normalCount[1],
                             faces.byNormal[2], faces.normalCount[2], faces.byNormal[3], faces.normalCount[3]);
            }
            if (engine.kind != visualimportedscenerender::RenderEngine::PathTrace)
                check (spread.subjectP95 - spread.subjectP5 >= item.minSpread,
                       label + ": the subject spans at least " + std::to_string (static_cast<int> (item.minSpread)) + " luma levels");
            if (scene->name == "spectral-ring" && engine.kind != visualimportedscenerender::RenderEngine::PathTrace)
            {
                // Every band at level 0: the bars differ only by how the key shades their faces.
                Shown rest;
                if (drawStarter (backend, *plan, *decoded, scene->spectral, scene->score, label + "-rest", rest, 0.0))
                {
                    Shown restAlone;
                    if (drawStarter (backend, *plan, awaySettings, scene->spectral, scene->score, label + "-rest-alone", restAlone, 0.0))
                    {
                        const auto resting = spreadOf (rest, &restAlone);
                        std::printf ("[geofloor] %-30s at rest (%6zu px) p5 %5.1f p50 %5.1f p95 %5.1f spread %5.1f\n",
                                     label.c_str(), resting.subject, resting.subjectP5, resting.subjectP50, resting.subjectP95,
                                     resting.subjectP95 - resting.subjectP5);
                    }
                }
            }
            check (open > contact + 15, label + ": the floor is darker in the objects' contact shadow");
            frames.push_back (frame);
        }
        if (frames.size() == 3)
            for (std::size_t e = 1; e < 3; ++e)
            {
                double difference = 0;
                for (std::size_t i = 0; i < frames[0].rgb.size(); ++i)
                    difference += std::abs (static_cast<int> (frames[e].rgb[i]) - static_cast<int> (frames[0].rgb[i]));
                const double mean = difference / static_cast<double> (frames[0].rgb.size());
                std::printf ("[geofloor] %-30s mean |%s - raster| %.1f\n", scene->name.c_str(), engines[e].name, mean);
                if (engines[e].kind == visualimportedscenerender::RenderEngine::PathTrace)
                    check (mean >= 5, scene->name + ": Path Trace differs visibly from Raster");
            }
    }
}

int runGl()
{
    if (! glfwInit())
    {
        std::fprintf (stderr, "SKIP: GLFW initialization failed\n");
        return 77;
    }
    glfwWindowHint (GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    auto* window = glfwCreateWindow (64, 64, "geometry-route-lighting", nullptr, nullptr);
    if (window == nullptr)
    {
        std::fprintf (stderr, "SKIP: hidden OpenGL 4.3 core context creation failed\n");
        glfwTerminate();
        return 77;
    }
    glfwMakeContextCurrent (window);
    int result = 0;
    arbitgl::GlFuncs gl;
    std::string missing;
    const auto* rendererName = reinterpret_cast<const char*> (glGetString (GL_RENDERER));
    const std::string name = rendererName != nullptr ? rendererName : "";
    if (! arbitgl::loadGlFunctions (gl, missing))
    {
        std::fprintf (stderr, "FAIL: OpenGL function loading failed: %s\n", missing.c_str());
        result = 1;
    }
    else if (name.find ("llvmpipe") != std::string::npos && std::getenv ("ARBIT_GL_TESTS_ALLOW_SOFTWARE") == nullptr)
    {
        std::fprintf (stderr, "SKIP: software OpenGL renderer %s\n", name.c_str());
        result = 77;
    }
    else
    {
        std::printf ("[georoute] OpenGL renderer %s\n", name.c_str());
        litCubeTests();
        starterTests();
        floorStageTests();
        failures += crossEngineExposureTests();
    }
    glfwDestroyWindow (window);
    glfwTerminate();
    return result;
}
} // namespace

int main()
{
    const int result = runGl();
    if (result != 0)
        return result;
    std::printf ("[georoute] %d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
