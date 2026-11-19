// Raster, Realtime and Path Trace show one lit scene at the same brightness (slice
// rasterexpo). A cube stands on a floor under a sun, an analytic sky the camera
// sees and the default rig's ambient colour. Each engine draws to a linear Color
// target, and the sky, the cube's sunlit top and the sunlit floor are compared
// against Path Trace, in scene-linear luminance and after AgX, the display step
// every engine's layer now takes. Before this slice Raster left the ambient out
// of its sky, and its layer skipped AgX, which clipped a lit face to white.
//
// Linked into arbit-opengl-geometry-route-lighting-tests, whose main runs it on
// that harness's OpenGL 4.3 context. ARBIT_EXPOSURE_LOG=1 prints every region.

#include "../src/geometry_core_backend.h"
#include "../src/gpu_backend/backend.h"
#include "../../shared/AgxViewTransform.h"
#include "../../shared/GeometryCoreRenderSettings.h"
#include "support/lit_cube_on_floor.h"

#include <GL/gl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace
{
using namespace HarmonicMIDI::grid;

constexpr std::uint32_t kWidth = 192, kHeight = 108;
constexpr GLenum kGlHalfFloat = 0x140B;

int checks = 0, failures = 0;

void check (bool condition, const std::string& message)
{
    ++checks;
    if (! condition)
    {
        ++failures;
        std::fprintf (stderr, "FAIL: %s\n", message.c_str());
    }
}

float halfToFloat (std::uint16_t bits)
{
    const std::uint32_t sign = (bits >> 15) & 1u, exponent = (bits >> 10) & 31u, mantissa = bits & 1023u;
    const float value = exponent == 0 ? std::ldexp (static_cast<float> (mantissa), -24)
                      : exponent == 31 ? INFINITY
                      : std::ldexp (static_cast<float> (mantissa | 1024u), static_cast<int> (exponent) - 25);
    return sign != 0 ? -value : value;
}

// Scene-linear RGB, top row first.
struct Linear
{
    std::vector<float> rgb;
    // Mean RGB over a box, inclusive.
    agxviewtransform::Rgb mean (int x0, int y0, int x1, int y1) const
    {
        agxviewtransform::Rgb sum { 0, 0, 0 };
        int count = 0;
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x)
            {
                if (x < 0 || y < 0 || x >= static_cast<int> (kWidth) || y >= static_cast<int> (kHeight)) continue;
                const std::size_t i = (static_cast<std::size_t> (y) * kWidth + static_cast<std::size_t> (x)) * 3;
                for (std::size_t c = 0; c < 3; ++c) sum[c] += rgb[i + c];
                ++count;
            }
        for (auto& channel : sum) channel /= count > 0 ? count : 1;
        return sum;
    }
};

double luminance (const agxviewtransform::Rgb& rgb)
{
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
}

// Display luma, 0 to 255, of linear RGB through AgX and the sRGB encode.
double agxLuma (const agxviewtransform::Rgb& radiance)
{
    auto shown = agxviewtransform::apply (radiance);
    for (auto& c : shown)
        c = 255.0 * (c <= 0.0031308 ? 12.92 * c : 1.055 * std::pow (c, 1.0 / 2.4) - 0.055);
    return luminance (shown);
}

bool draw (arbitgpu::NativeFixtureSceneBackend& backend, const videowire::geometry::PortContract& contract,
           const videowire::geometry::ValueDescriptor& value, const videowire::geometry::RenderSettings& settings,
           std::uint64_t owner, const std::string& label, Linear& out)
{
    using namespace videohelper::geometry;
    std::string error;
    const auto text = videowire::geometry::encodeRenderSettings (settings);
    const auto decoded = videowire::geometry::decodeRenderSettings (text, error);
    const auto admitted = decoded ? videowire::geometry::admitValue (value, contract, error) : std::nullopt;
    check (admitted.has_value(), label + ": the settings and value admit: " + error);
    if (! admitted) return false;
    NativeGeometryCoreCapabilitySource capabilities (backend);
    GeometryCorePlanRuntime runtime (capabilities);
    auto plan = runtime.admitExport ({1, 1, 1, owner, 1, PlanUse::exportRender},
        videowire::geometry::lowerRuntimePlan (contract, *admitted, {}, {}, false, text), {}, {}, error);
    check (plan.has_value(), label + ": the plan admits: " + error);
    if (! plan) return false;
    arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
    inputs.linearColor = true;
    const auto execution = executeNativeGeometry (backend, *plan, kWidth, kHeight, error, false, inputs, nullptr, {},
        0.0, {}, nullptr, 0.0, nullptr, {}, {}, arbitgpu::NativeFixturePrepareUse::Blocking, nullptr, nullptr,
        nullptr, &*decoded, 8171, 7);
    check (execution.has_value() && execution->frame != nullptr, label + " renders: " + error);
    if (! execution || ! execution->frame) return false;
    glFinish();
    const auto descriptor = execution->frame->colorTextureDescriptor();
    check (descriptor.format == arbitgpu::NativeTexturePixelFormat::Rgba16Float
               && descriptor.width == kWidth && descriptor.height == kHeight,
           label + ": Color is a linear half-float target");
    if (descriptor.format != arbitgpu::NativeTexturePixelFormat::Rgba16Float) return false;
    std::vector<std::uint16_t> halves (static_cast<std::size_t> (kWidth) * kHeight * 4, 0);
    glBindTexture (GL_TEXTURE_2D, static_cast<GLuint> (descriptor.textureViewHandle));
    glPixelStorei (GL_PACK_ALIGNMENT, 1);
    glGetTexImage (GL_TEXTURE_2D, 0, GL_RGBA, kGlHalfFloat, halves.data());
    glBindTexture (GL_TEXTURE_2D, 0);
    const bool bottomFirst = descriptor.rowOrder == arbitgpu::NativeTextureRowOrder::BottomFirst;
    out.rgb.assign (static_cast<std::size_t> (kWidth) * kHeight * 3, 0.0f);
    for (std::uint32_t y = 0; y < kHeight; ++y)
        for (std::uint32_t x = 0; x < kWidth; ++x)
        {
            const std::uint32_t row = bottomFirst ? kHeight - 1u - y : y;
            for (std::size_t c = 0; c < 3; ++c)
                out.rgb[(static_cast<std::size_t> (y) * kWidth + x) * 3 + c] =
                    halfToFloat (halves[(static_cast<std::size_t> (row) * kWidth + x) * 4 + c]);
        }
    return true;
}
} // namespace

// The failure count; the caller's main adds it to its own.
int crossEngineExposureTests()
{
    using namespace videohelper::litcube;
    auto& backend = arbitgpu::nativeFixtureSceneBackend();
    videowire::geometry::PortContract contract;
    std::string error;
    const auto value = cubeOnFloor (contract, error);
    check (value.has_value(), "the cube and floor lower: " + error);
    if (! value) return failures;
    // The sun from the right, the default rig's ambient 0.12 and an analytic sky.
    auto lighting = rig (true);
    check (lighting != nullptr, "the lighting rig is a valid scene record");
    if (lighting == nullptr) return failures;
    auto scene = std::make_shared<Visual3DScene> (*lighting);
    auto& sky = scene->lights[scene->lightCount - 1];
    sky.color = { 1.0f, 1.0f, 1.0f };
    sky.sky.version = 1;
    sky.sky.zenith = { 0.1f, 0.3f, 0.9f };
    sky.sky.horizon = { 0.6f, 0.7f, 0.95f };
    sky.sky.ground = { 0.08f, 0.08f, 0.1f };
    sky.sky.sunColor = { 1.0f, 0.9f, 0.75f };
    sky.sky.sunIntensity = 2.0f;
    sky.sky.sunAzimuthDegrees = 30;
    sky.sky.sunElevationDegrees = 40;
    sky.sky.sunRadiusDegrees = 3;
    check (validateVisual3DScene (*scene).valid(), "the sky rig is a valid scene record");

    struct Region { const char* name; int x0, y0, x1, y1; };
    const auto top = project (kWidth, kHeight, {0, 0.5f, 0}), floorLit = project (kWidth, kHeight, {1.6f, -0.5f, 0.3f});
    // The top rows see sky above the slab's far edge (about row 40).
    const Region regions[] {
        { "sky", 4, 2, static_cast<int> (kWidth) - 5, 20 },
        { "top", top.x - 3, top.y - 2, top.x + 3, top.y + 2 },
        { "floor", floorLit.x - 3, floorLit.y - 2, floorLit.x + 3, floorLit.y + 2 },
    };
    const struct { const char* name; visualimportedscenerender::RenderEngine kind; } engines[] {
        { "pathtrace", visualimportedscenerender::RenderEngine::PathTrace },
        { "raster", visualimportedscenerender::RenderEngine::Raster },
        { "realtime", visualimportedscenerender::RenderEngine::Realtime },
    };
    std::array<agxviewtransform::Rgb, std::size (regions)> reference {};
    const bool log = std::getenv ("ARBIT_EXPOSURE_LOG") != nullptr;
    for (std::size_t e = 0; e < std::size (engines); ++e)
    {
        videowire::geometry::RenderSettings settings;
        settings.engine = engines[e].kind;
        settings.exportSamples = 128;
        settings.lighting = scene;
        Linear frame;
        if (! draw (backend, contract, *value, settings, 9100 + e, std::string ("exposure-") + engines[e].name, frame))
            continue;
        for (std::size_t r = 0; r < std::size (regions); ++r)
        {
            const auto& region = regions[r];
            const auto mean = frame.mean (region.x0, region.y0, region.x1, region.y1);
            const std::string label = std::string (engines[e].name) + " " + region.name;
            check (std::isfinite (luminance (mean)) && luminance (mean) > 0.01, label + " is lit");
            if (log)
                std::printf ("[exposure] %-18s linear %.4f %.4f %.4f  Y %.4f  AgX luma %.1f\n", label.c_str(),
                             mean[0], mean[1], mean[2], luminance (mean), agxLuma (mean));
            if (e == 0) { reference[r] = mean; continue; }
            const double ratio = luminance (mean) / std::max (luminance (reference[r]), 1e-6);
            const double shown = std::abs (agxLuma (mean) - agxLuma (reference[r]));
            // The sky is one analytic function plus the ambient in every engine. The sunlit
            // faces carry the same direct light; Path Trace alone adds the floor's bounce
            // and the sky's occluded share, so they get a wider band.
            const bool skyRegion = r == 0;
            const double band = skyRegion ? 0.05 : 0.25;
            check (std::abs (ratio - 1.0) <= band,
                   label + ": scene-linear luminance within " + std::to_string (band) + " of Path Trace's, ratio "
                       + std::to_string (ratio));
            check (shown <= (skyRegion ? 3.0 : 12.0),
                   label + ": AgX display luma within reach of Path Trace's, off by " + std::to_string (shown));
        }
    }
    std::printf ("[exposure] %d checks, %d failures\n", checks, failures);
    return failures;
}
