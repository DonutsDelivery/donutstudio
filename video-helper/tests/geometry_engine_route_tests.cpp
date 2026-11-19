// Geometry wired straight into Render 3D, drawn by Path Trace: the scene the
// Geometry Core evaluates each frame reaches the tracer through
// executeNativeGeometry's engine route, and a second frame whose positions
// moved refits the acceleration structure instead of rebuilding it.
//
// The scenes stand in for the starters this route serves: Spectral Ring and
// Grid Instancer (spectrum instancing), Score-Deformed Imported Mesh (a score
// field bending a dense mesh), Audio Surface Instances (an audio-deformed
// surface, and instances tinted per instance). ARBIT_PTGEOM_PLANS names a
// directory of compiled starter plans (<starter>-<engine>.hex, written by the
// plugin's DirectGeometryRenderEngineTests); each Path Trace plan there renders
// too. ARBIT_PTGEOM_PNG names where the frames go.
//
// This binary defines arbitgpu::uploadCpuSceneFrame itself, in place of the
// headless stub, so the traced film comes back as bytes.

#include "../src/geometry_core_backend.h"
#include "../src/fixture_scene_renderer.h"
#include "../src/gpu_backend/cpu_scene_frame.h"
#include "../src/pathtrace/devices/device_optix_host.h"
#include "../../shared/GeometryCoreRenderSettings.h"
#include "score_field_fixture.h"
#include "instance_appearance_fixture.h"
#include "support/lit_cube_on_floor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace
{
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

struct Film
{
    std::vector<std::uint8_t> color;
    bool linear = false;
    std::uint32_t width = 0, height = 0;
} film;

class CapturedFrame final : public arbitgpu::NativeFixtureSceneFrame
{
public:
    CapturedFrame (std::string backend, std::uint32_t width, std::uint32_t height) noexcept
        : backend_ (std::move (backend)), width_ (width), height_ (height) {}
    const std::string& backend() const noexcept override { return backend_; }
    std::uint32_t width() const noexcept override { return width_; }
    std::uint32_t height() const noexcept override { return height_; }
    std::uintptr_t colorImageHandle() const noexcept override { return 1; }
    std::uintptr_t colorTextureViewHandle() const noexcept override { return 2; }
    std::uintptr_t depthImageHandle() const noexcept override { return 3; }
    std::uintptr_t depthTextureViewHandle() const noexcept override { return 4; }
    arbitgpu::NativeTextureViewDescriptor colorTextureDescriptor() const noexcept override
    {
        return { backend_, arbitgpu::NativeTextureViewKind::Texture2D,
            arbitgpu::NativeTexturePixelFormat::Rgba8Unorm, 1, 2, width_, height_, 1, true, 77, 1 };
    }

private:
    std::string backend_;
    std::uint32_t width_ = 0, height_ = 0;
};

// The backend the Geometry Core admits against. Path Trace never submits to it.
class TracerOnlyBackend final : public arbitgpu::NativeFixtureSceneBackend
{
public:
    arbitgpu::BackendInfo info() const override { return { true, true, "opengl", "ptgeom", {} }; }
    arbitgpu::GeometryCoreExecutionCapabilities geometryCoreCapabilities() const override
    {
        using namespace videowire::geometry;
        arbitgpu::GeometryCoreExecutionCapabilities capabilities;
        capabilities.immutableSourceBuffers = capabilities.stableElementIds = true;
        capabilities.typedFieldEvaluation = capabilities.gpuInstancingWithoutMeshExpansion = true;
        capabilities.supportedCarriers = (1u << static_cast<unsigned> (CarrierKind::geometry3D))
            | (1u << static_cast<unsigned> (CarrierKind::instances3D));
        capabilities.maxVertices = kMaximumVertices;
        capabilities.maxIndices = kMaximumIndices;
        capabilities.maxPoints = capabilities.maxCurvePoints = capabilities.maxSplines = 1;
        capabilities.maxInstances = kMaximumInstances;
        capabilities.maxFieldElements = kMaximumFieldElements;
        capabilities.maxAttributes = kMaximumAttributes;
        capabilities.maxOperations = kMaximumOperations;
        capabilities.maxDispatches = kMaximumDispatches;
        capabilities.maxBufferBytes = kMaximumBufferBytes;
        return capabilities;
    }
    arbitgpu::NativeFixtureScenePreparation prepare (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>&,
        const std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram>&,
        arbitgpu::NativeFixturePrepareUse = arbitgpu::NativeFixturePrepareUse::Blocking) override
    {
        ++rasterCalls;
        arbitgpu::NativeFixtureScenePreparation result;
        result.error = "the raster backend is not part of this harness";
        return result;
    }
    arbitgpu::NativeFixtureSceneSubmission render (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>&,
        const std::shared_ptr<const arbitgpu::NativeFixtureSceneResources>&,
        std::uint32_t, std::uint32_t, arbitgpu::NativeFixtureSceneRuntimeInputs) override
    {
        ++rasterCalls;
        arbitgpu::NativeFixtureSceneSubmission result;
        result.error = "the raster backend is not part of this harness";
        return result;
    }
    int rasterCalls = 0;
};

float halfToFloat (std::uint16_t bits)
{
    const std::uint32_t sign = (bits >> 15) & 1u, exponent = (bits >> 10) & 31u, mantissa = bits & 1023u;
    float value = exponent == 0 ? std::ldexp (static_cast<float> (mantissa), -24)
                : exponent == 31 ? INFINITY
                : std::ldexp (static_cast<float> (mantissa | 1024u), static_cast<int> (exponent) - 25);
    return sign != 0 ? -value : value;
}

// A stored-block PNG, so the frames need no image library.
void writePng (const std::string& path, std::uint32_t width, std::uint32_t height, const std::vector<std::uint8_t>& rgbTopFirst)
{
    std::vector<std::uint8_t> raw;
    for (std::uint32_t y = 0; y < height; ++y)
    {
        raw.push_back (0);
        raw.insert (raw.end(), rgbTopFirst.begin() + static_cast<std::ptrdiff_t> (y * width * 3),
                    rgbTopFirst.begin() + static_cast<std::ptrdiff_t> ((y + 1) * width * 3));
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
    for (const auto v : { width, height }) for (int s = 24; s >= 0; s -= 8) header.push_back (static_cast<std::uint8_t> (v >> s));
    header.insert (header.end(), { 8, 2, 0, 0, 0 });
    chunk ("IHDR", header);
    chunk ("IDAT", z);
    chunk ("IEND", {});
    std::fclose (file);
}

// The captured film as display sRGB, top row first.
std::vector<std::uint8_t> filmRgb()
{
    std::vector<std::uint8_t> rgb (static_cast<std::size_t> (film.width) * film.height * 3);
    const std::size_t stride = film.linear ? 8 : 4;
    if (film.color.size() != static_cast<std::size_t> (film.width) * film.height * stride) return {};
    for (std::uint32_t y = 0; y < film.height; ++y)
        for (std::uint32_t x = 0; x < film.width; ++x)
        {
            const std::size_t from = (static_cast<std::size_t> (film.height - 1 - y) * film.width + x) * stride;
            const std::size_t to = (static_cast<std::size_t> (y) * film.width + x) * 3;
            for (int c = 0; c < 3; ++c)
            {
                float linear = 0;
                if (film.linear)
                {
                    std::uint16_t bits = 0;
                    std::memcpy (&bits, &film.color[from + static_cast<std::size_t> (c) * 2], 2);
                    linear = halfToFloat (bits);
                }
                else linear = film.color[from + static_cast<std::size_t> (c)] / 255.0f;
                linear = std::clamp (linear, 0.0f, 1.0f);
                const float srgb = linear <= 0.0031308f ? 12.92f * linear : 1.055f * std::pow (linear, 1.0f / 2.4f) - 0.055f;
                rgb[to + static_cast<std::size_t> (c)] = static_cast<std::uint8_t> (srgb * 255.0f + 0.5f);
            }
        }
    return rgb;
}

// Pixels that differ from the top-left corner, which shows the world background.
std::size_t litPixels (const std::vector<std::uint8_t>& rgb)
{
    std::size_t lit = 0;
    if (rgb.size() < 3) return lit;
    for (std::size_t i = 0; i + 2 < rgb.size(); i += 3)
        lit += std::abs (rgb[i] - rgb[0]) + std::abs (rgb[i + 1] - rgb[1]) + std::abs (rgb[i + 2] - rgb[2]) > 24 ? 1 : 0;
    return lit;
}

std::string pngDirectory()
{
    const char* dir = std::getenv ("ARBIT_PTGEOM_PNG");
    return dir != nullptr ? std::string (dir) : std::string {};
}

constexpr std::uint32_t kWidth = 480, kHeight = 270;

videowire::geometry::RenderSettings pathTrace()
{
    videowire::geometry::RenderSettings settings;
    settings.engine = visualimportedscenerender::RenderEngine::PathTrace;
    settings.exportSamples = 64;
    return settings;
}

struct Draw
{
    bool rendered = false;
    std::vector<std::uint8_t> rgb;
    std::string accelBuild, device;
    double totalMs = 0;
    std::uint32_t samples = 0;
};

// One export frame of an admitted plan through the engine route.
Draw drawFrame (arbitgpu::NativeFixtureSceneBackend& backend, const videohelper::geometry::RuntimeAdmission& admission,
                const videowire::geometry::RenderSettings& settings, arbitgpu::NativeFixtureSceneRuntimeInputs inputs,
                videohelper::geometry::SpectrumEvaluation* spectrum, double seconds, const std::string& label)
{
    Draw draw;
    film = {};
    videorender::fixture3d::clearPathTraceExportStats();
    std::string error;
    const auto start = std::chrono::steady_clock::now();
    const auto execution = videohelper::geometry::executeNativeGeometry (backend, admission, kWidth, kHeight, error, false,
        std::move (inputs), spectrum, {}, seconds, {}, nullptr, 0.0, nullptr, {}, {},
        arbitgpu::NativeFixturePrepareUse::Blocking, nullptr, nullptr, nullptr, &settings, 8170, 7);
    const double wall = std::chrono::duration<double, std::milli> (std::chrono::steady_clock::now() - start).count();
    check (execution.has_value() && execution->frame != nullptr, label + " renders through the engine route: " + error);
    if (! execution) return draw;
    draw.rendered = true;
    draw.rgb = filmRgb();
    if (const auto stats = videorender::fixture3d::lastPathTraceStats (videorender::fixture3d::RenderUse::Export))
    {
        draw.accelBuild = stats->phases.accelBuild != nullptr ? stats->phases.accelBuild : "";
        draw.device = stats->device != nullptr ? stats->device : "";
        draw.totalMs = stats->phases.has (videohelper::pathtrace::FramePhase::Total)
            ? stats->phases.get (videohelper::pathtrace::FramePhase::Total) : wall;
        draw.samples = stats->samplesPerPixel;
    }
    std::printf ("[ptgeom] %s: device %s accel %s spp %u total %.1f ms (call %.1f ms), %zu lit pixels\n", label.c_str(),
                 draw.device.c_str(), draw.accelBuild.c_str(), draw.samples, draw.totalMs, wall, litPixels (draw.rgb));
    if (const auto dir = pngDirectory(); ! dir.empty() && ! draw.rgb.empty())
        writePng (dir + "/" + label + ".png", kWidth, kHeight, draw.rgb);
    return draw;
}

std::optional<videohelper::geometry::RuntimeAdmission> admit (
    arbitgpu::NativeFixtureSceneBackend& backend, const videowire::geometry::PortContract& contract,
    const videowire::geometry::ValueDescriptor& value, const std::string& settingsText, std::uint64_t owner)
{
    using namespace videowire::geometry;
    using namespace videohelper::geometry;
    std::string error;
    const auto admitted = admitValue (value, contract, error);
    check (admitted.has_value(), "the fixture value admits: " + error);
    if (! admitted) return std::nullopt;
    static std::vector<std::unique_ptr<NativeGeometryCoreCapabilitySource>> capabilities;
    static std::vector<std::unique_ptr<GeometryCorePlanRuntime>> runtimes;
    capabilities.push_back (std::make_unique<NativeGeometryCoreCapabilitySource> (backend));
    runtimes.push_back (std::make_unique<GeometryCorePlanRuntime> (*capabilities.back()));
    auto plan = runtimes.back()->admitExport ({1, 1, 1, owner, 1, PlanUse::exportRender},
        lowerRuntimePlan (contract, *admitted, {}, {}, false, settingsText), {}, {}, error);
    check (plan.has_value(), "the engine-route plan admits: " + error);
    return plan;
}

videowire::geometry::spectrum::Bands bandsAt (double seconds)
{
    videowire::geometry::spectrum::Bands bands {};
    for (std::size_t i = 0; i < bands.size(); ++i)
        bands[i] = static_cast<float> (0.35 + 0.3 * std::sin (seconds * 3.1 + static_cast<double> (i) * 0.7));
    return bands;
}

videohelper::geometry::SpectrumEvaluation spectrumAt (double seconds)
{
    videohelper::geometry::SpectrumEvaluation input;
    input.featuresAt = bandsAt;
    input.timeSeconds = seconds;
    input.bands = bandsAt (seconds);
    return input;
}

// Two frames of one plan whose positions move between them.
void movingPair (arbitgpu::NativeFixtureSceneBackend& backend, const videohelper::geometry::RuntimeAdmission& plan,
                 const std::string& name, arbitgpu::NativeFixtureSceneRuntimeInputs first,
                 arbitgpu::NativeFixtureSceneRuntimeInputs second, bool spectral)
{
    auto early = spectrumAt (0.2), late = spectrumAt (0.9);
    const auto settings = pathTrace();
    const auto a = drawFrame (backend, plan, settings, std::move (first), spectral ? &early : nullptr, 0.2, name + "-pt-a");
    const auto b = drawFrame (backend, plan, settings, std::move (second), spectral ? &late : nullptr, 0.9, name + "-pt-b");
    check (a.rendered && litPixels (a.rgb) > 500, name + ": the traced frame shows the geometry");
    check (b.rendered && a.rgb != b.rgb, name + ": the second frame shows the moved geometry");
    // OptiX names its refit "update"; the compute devices name it "refit".
    check (b.accelBuild == "update" || b.accelBuild == "refit" || b.accelBuild == "reused" || b.device == "CPU",
           name + ": moving positions refits the acceleration structure (" + b.accelBuild + ")");
}

void spectralRingInstances (arbitgpu::NativeFixtureSceneBackend& backend)
{
    using namespace videowire::geometry;
    std::string error;
    PortContract meshContract;
    meshContract.carrier = CarrierKind::geometry3D; meshContract.overflow = OverflowPolicy::reject;
    meshContract.maxVertices = 24; meshContract.maxIndices = 36;
    PortContract contract;
    contract.carrier = CarrierKind::instances3D; contract.overflow = OverflowPolicy::reject;
    contract.maxInstances = 64;
    const auto cube = lowerMeshGenerator (921, 921, 1, OperationCode::cube, {0.3f, 0.3f, 0.3f, 0}, meshContract, error);
    check (cube.has_value(), "the instance cube lowers: " + error);
    if (! cube) return;
    spectrum::Binding binding;
    binding.firstBand = 0; binding.lastBand = 11;
    binding.baseScale = {0.6f, 0.6f, 0.6f}; binding.scaleResponse = {0, 2.5f, 0};
    binding.direction = {0, 0.4f, 0}; binding.attackSeconds = 0.01f; binding.releaseSeconds = 0.01f;
    const auto value = lowerSpectrumInstancer (*cube, 922, binding, {1, 3.2f, 3.2f, 4}, contract, error);
    check (value.has_value(), "the spectrum instancer lowers: " + error);
    if (! value) return;
    const auto plan = admit (backend, contract, *value, encodeRenderSettings (pathTrace()), 922);
    if (plan) movingPair (backend, *plan, "spectral-ring-instances", {}, {}, true);
}

void scoreDeformedMesh (arbitgpu::NativeFixtureSceneBackend& backend)
{
    using namespace videowire::geometry;
    using namespace videohelper::geometry;
    std::string error;
    PortContract contract;
    contract.carrier = CarrierKind::geometry3D; contract.overflow = OverflowPolicy::reject;
    contract.maxVertices = 4096; contract.maxIndices = 24576;
    const auto mesh = lowerGrid (951, 951, 1, 40, 26, 0.075f, contract, error);
    check (mesh.has_value(), "the card mesh lowers: " + error);
    if (! mesh) return;
    scorefield::Binding binding;
    binding.fieldStableId = 952; binding.sourceGeometryStableId = 951;
    binding.consumerGeometryStableId = 953; binding.scoreSourceStableId = 954;
    binding.mode = scorefield::Mode::velocityInfluence; binding.radius = 2;
    binding.mapping.xScale = 0.25f; binding.mapping.yScale = 0.15f; binding.mapping.zScale = 0.1f;
    binding.direction = {0, 0, 1}; binding.gain = 0.4f;
    for (const auto& p : std::get<GeometryData> (mesh->data).positions)
        binding.positions.push_back ({p.x, p.y, p.z});
    binding.elementIds = std::get<GeometryData> (mesh->data).vertexIds;
    auto value = *mesh;
    value.stableId = 953; value.scoreFields.push_back (binding);
    value.operations.push_back ({952, OperationCode::evaluateField, 952, 952, {}});
    value.operations.push_back ({953, OperationCode::transformTRS, 951, 953, {}});
    value.dispatchCount += 2;
    const auto plan = admit (backend, contract, value, encodeRenderSettings (pathTrace()), 953);
    if (! plan) return;
    arbitgpu::NativeFixtureSceneRuntimeInputs loud, soft;
    loud.canonicalBlockCFrame = scoreFieldFrame (false, false, true, 127.0f);
    soft.canonicalBlockCFrame = scoreFieldFrame (false, false, true, 40.0f);
    movingPair (backend, *plan, "score-deformed-mesh", loud, soft, false);
}

void audioSurface (arbitgpu::NativeFixtureSceneBackend& backend)
{
    using namespace videowire::geometry;
    std::string error;
    PortContract contract;
    contract.carrier = CarrierKind::geometry3D; contract.overflow = OverflowPolicy::reject;
    contract.maxVertices = 1024; contract.maxIndices = 6144;
    const auto source = lowerGrid (941, 941, 1, 24, 16, 0.12f, contract, error);
    check (source.has_value(), "the audio surface lowers: " + error);
    if (! source) return;
    spectrum::Binding binding;
    binding.firstBand = 0; binding.lastBand = 23; binding.direction = {0, 0, 0.5f};
    binding.deformer.mode = static_cast<spectrum::Deformation> (0);
    binding.deformer.frequency = 3.7f; binding.deformer.falloffRadius = 3; binding.deformer.falloffPower = 1.2f;
    binding.attackSeconds = 0.01f; binding.releaseSeconds = 0.01f;
    const auto value = lowerAudioDeformer (*source, 942, binding, contract, error);
    check (value.has_value(), "the audio deformer lowers: " + error);
    if (! value) return;
    const auto plan = admit (backend, contract, *value, encodeRenderSettings (pathTrace()), 942);
    if (plan) movingPair (backend, *plan, "audio-surface", {}, {}, true);
}

// Instance Color and Emission reach the tracer as material variants.
void tintedInstances (arbitgpu::NativeFixtureSceneBackend& backend)
{
    // Variant 1 colours two instances red and two green; variant 2 adds red
    // Emission to the green pair, which then glows orange.
    for (const unsigned variant : { 1u, 2u })
    {
        std::string error;
        const auto value = instanceAppearanceFixture (variant, error);
        check (value.has_value(), "the tinted instances lower: " + error);
        if (! value) return;
        const auto plan = admit (backend, withAttributeContract (appearanceInstanceContract(), *value), *value,
                                 videowire::geometry::encodeRenderSettings (pathTrace()), 9117 + variant);
        if (! plan) return;
        const auto label = std::string (variant == 1 ? "tinted" : "glowing") + "-instances-pt";
        const auto draw = drawFrame (backend, *plan, pathTrace(), {}, nullptr, 0.0, label);
        std::size_t red = 0, green = 0, orange = 0;
        for (std::size_t i = 0; i + 2 < draw.rgb.size(); i += 3)
        {
            const int r = draw.rgb[i], g = draw.rgb[i + 1], b = draw.rgb[i + 2];
            red += r > 60 && r > 2 * g ? 1 : 0;
            green += g > 60 && g > 2 * r ? 1 : 0;
            orange += r > 200 && g > 60 && g < r && b < g ? 1 : 0;
        }
        std::printf ("[ptgeom] %s: %zu red, %zu green, %zu orange pixels\n", label.c_str(), red, green, orange);
        if (variant == 1)
            check (red > 200 && green > 200, "each instance traces in its own Color");
        else
            check (red > 200 && orange > 200 && green == 0, "instance Emission makes the green pair glow orange");
    }
}

// Render 3D's Camera and Light reach the tracer on this route (slice georoute):
// a cube on a floor under a sun from the right, lit by the plan's rig.
void litCube (arbitgpu::NativeFixtureSceneBackend& backend)
{
    using namespace videohelper::litcube;
    videowire::geometry::PortContract contract;
    std::string error;
    const auto value = cubeOnFloor (contract, error);
    check (value.has_value(), "the cube and floor lower: " + error);
    if (! value) return;
    auto settings = pathTrace();
    settings.lighting = rig (false);
    check (settings.lighting != nullptr, "the lighting rig is a valid scene record");
    if (! settings.lighting) return;
    const auto text = videowire::geometry::encodeRenderSettings (settings);
    const auto decoded = videowire::geometry::decodeRenderSettings (text, error);
    check (decoded.has_value() && decoded->lighting != nullptr, "the settings carry the rig: " + error);
    if (! decoded) return;
    const auto plan = admit (backend, contract, *value, text, 9201);
    if (! plan) return;
    const auto draw = drawFrame (backend, *plan, *decoded, {}, nullptr, 0.0, "lit-cube-pt");
    if (draw.rgb.size() != static_cast<std::size_t> (kWidth) * kHeight * 3) return;
    const auto luma = [&draw] (SceneVec3 p) {
        const auto px = project (kWidth, kHeight, p);
        float sum = 0;
        int n = 0;
        for (int dy = -2; dy <= 2; ++dy)
            for (int dx = -2; dx <= 2; ++dx)
            {
                const int x = std::clamp (px.x + dx, 0, static_cast<int> (kWidth) - 1);
                const int y = std::clamp (px.y + dy, 0, static_cast<int> (kHeight) - 1);
                const auto* c = &draw.rgb[(static_cast<std::size_t> (y) * kWidth + static_cast<std::size_t> (x)) * 3];
                sum += 0.2126f * c[0] + 0.7152f * c[1] + 0.0722f * c[2];
                ++n;
            }
        return sum / static_cast<float> (n);
    };
    const float top = luma ({0, 0.5f, 0}), frontTop = luma ({0, 0.35f, 0.5f}), frontBottom = luma ({0, -0.4f, 0.5f}),
        floorLit = luma ({1.6f, -0.5f, 0.3f}), floorShadow = luma ({-0.85f, -0.5f, 0.2f});
    std::printf ("[ptgeom] lit-cube-pt: top %.1f front %.1f/%.1f floor lit %.1f shadow %.1f\n",
                 top, frontTop, frontBottom, floorLit, floorShadow);
    check (top > frontTop + 25, "the sunlit top traces brighter than the unlit front face");
    check (floorLit > floorShadow + 25, "the cube's traced shadow darkens the floor");
    // Raster leaves the unlit front face flat ambient; the tracer bounces the floor onto it.
    check (frontBottom > frontTop, "the floor's bounce brightens the bottom of the front face");
}

// Plans the plugin compiled from the starters themselves.
void starterPlans (arbitgpu::NativeFixtureSceneBackend& backend)
{
    const char* dir = std::getenv ("ARBIT_PTGEOM_PLANS");
    if (dir == nullptr) return;
    for (const auto& entry : std::filesystem::directory_iterator (dir))
    {
        const auto name = entry.path().stem().string();
        if (entry.path().extension() != ".hex" || name.size() < 2 || name.substr (name.size() - 2) != "-1") continue;
        std::ifstream in (entry.path());
        std::stringstream text;
        text << in.rdbuf();
        std::string error;
        const auto bytes = videowire::geometry::decodeLoweredPlanText (text.str(), error);
        const auto lowered = bytes ? videowire::geometry::decodeLoweredRuntimePlan (*bytes, error) : std::nullopt;
        const auto settings = lowered ? videowire::geometry::decodeRenderSettings (lowered->renderSettings, error) : std::nullopt;
        check (settings.has_value() && settings->engine == visualimportedscenerender::RenderEngine::PathTrace,
               name + ": the compiled starter plan carries Path Trace: " + error);
        if (! settings) continue;
        static std::vector<std::unique_ptr<videohelper::geometry::NativeGeometryCoreCapabilitySource>> capabilities;
        static std::vector<std::unique_ptr<videohelper::geometry::GeometryCorePlanRuntime>> runtimes;
        capabilities.push_back (std::make_unique<videohelper::geometry::NativeGeometryCoreCapabilitySource> (backend));
        runtimes.push_back (std::make_unique<videohelper::geometry::GeometryCorePlanRuntime> (*capabilities.back()));
        auto plan = runtimes.back()->admitExport ({1, 1, 1, 7, 1, videohelper::geometry::PlanUse::exportRender},
                                                  *bytes, {}, {}, error);
        check (plan.has_value(), name + ": the compiled starter plan admits: " + error);
        if (! plan) continue;
        arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
        inputs.canonicalBlockCFrame = scoreFieldFrame();
        auto spectrum = spectrumAt (0.9);
        drawFrame (backend, *plan, *settings, inputs, &spectrum, 0.9, name.substr (0, name.size() - 2) + "-starter-pt");
    }
}
} // namespace

std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> arbitgpu::uploadCpuSceneFrame (
    const std::string& backend, const arbitgpu::CpuSceneImage& image, std::string&)
{
    film.color = *image.color;
    film.linear = image.linearColor;
    film.width = image.width;
    film.height = image.height;
    return std::make_shared<CapturedFrame> (backend, image.width, image.height);
}

int main()
{
    TracerOnlyBackend backend;
    spectralRingInstances (backend);
    scoreDeformedMesh (backend);
    audioSurface (backend);
    tintedInstances (backend);
    litCube (backend);
    starterPlans (backend);
    check (backend.rasterCalls == 0, "no engine-route frame drew with the raster backend");
    std::printf ("[ptgeom] %d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
