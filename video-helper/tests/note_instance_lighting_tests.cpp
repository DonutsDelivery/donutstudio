// Note Model Array lighting (slice noteinst). A copy that Note Model Array
// draws must be shaded by the scene's lights as the single model is, in Raster,
// Realtime and Path Trace. The bundled Blender knot renders through
// FixtureSceneRenderer on the OpenGL backend under one directional light:
//
//  - single: the model alone;
//  - one note at the model's own place and size (every axis scale 0, mesh
//    scale 1), which must shade as the single draw does;
//
// Two lights: the first Model Key Light the starter had (Transform 3D rotation
// X -90, straight down the camera's view) and an oblique one at (-35, -25).
//
// A second block (slice knotbal) renders the Polyphonic Imported Model Array
// starter's mapping and colours, 8 notes at velocity 100, under the key and glow
// the starter shipped with and under the ones it ships now, and requires the
// lit-to-shadow swing of the copies to be clearly wider while they still glow.
//
// ARBIT_NOTEINST_PNG=dir writes every frame there as <engine>-<light>-<case>.png.
// It skips with 77 when no hardware OpenGL 4.3 context exists.

#include "../../shared/VisualStarterKnotAsset.h"
#include "../src/canonical_block_c_frame.h"
#include "../src/fixture_scene_renderer.h"
#include "../src/gl_loader.h"
#include "../src/glb_scene_adapter.h"
#include "../src/gltf_glb.h"
#include "../src/gpu_backend/backend.h"

#include <GLFW/glfw3.h>
#include <GL/gl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace
{

namespace renderer = videorender::fixture3d;
namespace gltf = videohelper::gltf;
namespace vni = visualnoteinstancing;
using HarmonicMIDI::grid::Visual3DScene;

constexpr GLenum kGlHalfFloat = 0x140B;

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

// A linear frame, top row first, with the bytes the compositor shows.
struct Shown
{
    std::uint32_t width = 0, height = 0;
    std::vector<float> luminance;
    std::vector<std::uint8_t> rgb;
};

bool show (const arbitgpu::NativeRawPassPixels& pixels, bool bottomFirst, Shown& out)
{
    if (pixels.format != arbitgpu::NativeTexturePixelFormat::Rgba16Float)
        return false;
    const std::size_t count = static_cast<std::size_t> (pixels.width) * pixels.height;
    if (pixels.bytes.size() != count * 8)
        return false;
    out.width = pixels.width;
    out.height = pixels.height;
    out.rgb.assign (count * 3, 0);
    out.luminance.assign (count, 0.0f);
    for (std::uint32_t y = 0; y < pixels.height; ++y)
        for (std::uint32_t x = 0; x < pixels.width; ++x)
        {
            const std::uint32_t row = bottomFirst ? pixels.height - 1u - y : y;
            const std::size_t source = (static_cast<std::size_t> (row) * pixels.width + x) * 8;
            const std::size_t at = static_cast<std::size_t> (y) * pixels.width + x;
            float c[3];
            for (std::size_t k = 0; k < 3; ++k)
            {
                std::uint16_t bits = 0;
                std::memcpy (&bits, pixels.bytes.data() + source + k * 2, 2);
                c[k] = halfToFloat (bits);
                out.rgb[at * 3 + k] = encodeByte (c[k]);
            }
            out.luminance[at] = 0.2126f * c[0] + 0.7152f * c[1] + 0.0722f * c[2];
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

struct ScoreFrame
{
    std::shared_ptr<arbitmod::Score> score = std::make_shared<arbitmod::Score>();
    canonicalblockc::FrameProducer producer;
    canonicalblockc::FrameKey key;

    // (MIDI note, start beat) pairs at velocity 100, four beats long, track 0.
    explicit ScoreFrame (const std::vector<std::pair<int, float>>& notes)
    {
        score->scoreRevision = 1;
        for (const auto& [midi, start] : notes)
        {
            arbitmod::Note note;
            note.id = -midi; note.midiNote = static_cast<float> (midi);
            note.freqHz = 440.0f * std::exp2 ((static_cast<float> (midi) - 69.0f) / 12.0f);
            note.velocity = 100.0f; note.startBeat = start; note.lengthBeats = 4;
            score->notes.push_back (note);
        }
        key.projectGeneration = key.sourceGeneration = key.helperGeneration = 1;
        key.backendGeneration = key.deviceGeneration = key.scoreGeneration = 1;
        key.beatMapGeneration = key.fpsGeneration = key.loopGeneration = key.seekGeneration = 1;
        key.fps = 24;
    }

    std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> at (float beat)
    {
        ++key.frame;
        key.beat = beat;
        return producer.evaluate (key, score, beat);
    }
};

std::shared_ptr<const Visual3DScene> knotScene (std::string& error)
{
    gltf::GlbAdmissionOptions options;
    options.supportedRequiredExtensions = { "KHR_lights_punctual" };
    options.admitAnimations = options.admitSkins = true;
    const auto document = gltf::decodeAnimatedGlbBaseScene (visualstarterknot::kGlb.data(),
                                                            visualstarterknot::kGlb.size(), options, error);
    if (! document)
        return {};
    const auto adapted = gltf::adaptStaticGlbToVisual3DScene (*document, error);
    if (! adapted)
        return {};
    return std::make_shared<const Visual3DScene> (*adapted);
}

// A Render 3D directional light in place of the scene's, as the starter's Model
// Key Light: white, its Transform 3D rotation as a quaternion.
HarmonicMIDI::grid::SceneLightRecord keyLight (const Visual3DScene& scene, HarmonicMIDI::grid::SceneQuaternion rotation,
                                               float intensity = 1.0f)
{
    auto light = scene.lights[0];
    light.kind = HarmonicMIDI::grid::SceneLightKind::Directional;
    light.transform.rotation = rotation;
    light.color = { 1.0f, 1.0f, 1.0f };
    light.intensity = intensity;
    return light;
}

// Transform 3D rotationX then rotationY, in degrees, as a quaternion (Y times X).
HarmonicMIDI::grid::SceneQuaternion eulerXY (float degreesX, float degreesY)
{
    const float rad = 3.14159265358979f / 180.0f;
    const float sx = std::sin (0.5f * degreesX * rad), cx = std::cos (0.5f * degreesX * rad);
    const float sy = std::sin (0.5f * degreesY * rad), cy = std::cos (0.5f * degreesY * rad);
    return { cy * sx, sy * cx, -sy * sx, cy * cx };
}

struct EngineCase
{
    const char* name;
    renderer::SceneEngineKind kind;
};

bool renderFrame (const std::shared_ptr<const Visual3DScene>& scene, const EngineCase& engine,
                  const arbitgpu::NativeFixtureSceneRuntimeInputs& inputs, Shown& out, std::string& error)
{
    renderer::FixtureSceneRenderer frames (arbitgpu::nativeFixtureSceneBackend());
    renderer::SceneEngineSelection selection;
    selection.engine = engine.kind;
    selection.exportSamples = 32;
    frames.selectEngine (selection);
    renderer::RenderedFrame frame;
    if (! frames.renderExport (scene, std::shared_ptr<const renderer::AdmittedSurfaceMaterialBinding> {}, {}, inputs,
                               { 320, 180 }, renderer::kNativeGpuCapability, frame, error))
        return false;
    glFinish();
    if (frame.nativeFrame == nullptr)
    {
        error = "no frame";
        return false;
    }
    arbitgpu::NativeRawPassPixels pixels;
    bool bottomFirst = true;
    if (! frame.nativeFrame->readRawPass (renderpassoutput::Output::Color, pixels, error))
    {
        const auto descriptor = frame.nativeFrame->colorTextureDescriptor();
        if (descriptor.format != arbitgpu::NativeTexturePixelFormat::Rgba16Float)
            return false;
        error.clear();
        pixels.format = descriptor.format;
        pixels.width = descriptor.width;
        pixels.height = descriptor.height;
        pixels.bytes.assign (static_cast<std::size_t> (descriptor.width) * descriptor.height * 8, 0);
        glBindTexture (GL_TEXTURE_2D, static_cast<GLuint> (descriptor.textureViewHandle));
        glPixelStorei (GL_PACK_ALIGNMENT, 1);
        glGetTexImage (GL_TEXTURE_2D, 0, GL_RGBA, kGlHalfFloat, pixels.bytes.data());
        glBindTexture (GL_TEXTURE_2D, 0);
        bottomFirst = descriptor.rowOrder == arbitgpu::NativeTextureRowOrder::BottomFirst;
    }
    if (! show (pixels, bottomFirst, out))
    {
        error = "unexpected Color format";
        return false;
    }
    return true;
}

struct Spread { std::size_t pixels = 0; float p5 = 0, p50 = 0, p95 = 0; };

// Luminance percentiles over the pixels that differ from the corner backdrop.
Spread spreadOf (const Shown& frame)
{
    std::vector<float> values;
    const std::uint8_t* corner = frame.rgb.data();
    for (std::size_t i = 0; i < frame.luminance.size(); ++i)
    {
        const std::uint8_t* p = frame.rgb.data() + i * 3;
        if (std::abs (p[0] - corner[0]) + std::abs (p[1] - corner[1]) + std::abs (p[2] - corner[2]) > 12)
            values.push_back (frame.luminance[i]);
    }
    Spread spread;
    spread.pixels = values.size();
    if (values.empty())
        return spread;
    std::sort (values.begin(), values.end());
    const auto at = [&values] (double q) { return values[static_cast<std::size_t> (q * static_cast<double> (values.size() - 1))]; };
    spread.p5 = at (0.05); spread.p50 = at (0.5); spread.p95 = at (0.95);
    return spread;
}

void print (const std::string& label, const Spread& s)
{
    std::printf ("[noteinst] %-38s %6zu px  luminance p5 %.4f  p50 %.4f  p95 %.4f  p95-p5 %.4f  bytes %d/%d/%d\n",
                 label.c_str(), s.pixels, s.p5, s.p50, s.p95, s.p95 - s.p5,
                 encodeByte (s.p5), encodeByte (s.p50), encodeByte (s.p95));
}

vni::Mapping inPlaceMapping()
{
    vni::Mapping mapping;
    mapping.xScale = mapping.yScale = mapping.zScale = 0.0f;
    mapping.meshScale = 1.0f;
    return mapping;
}

// Polyphonic Imported Model Array (BuiltInVisualModules.h).
vni::Mapping starterMapping (float emissionHigh)
{
    vni::Mapping mapping;
    mapping.x = vni::MappingAxis::Onset;
    mapping.y = vni::MappingAxis::LogFrequency;
    mapping.z = vni::MappingAxis::Track;
    mapping.xScale = 0.3f; mapping.yScale = 1.2f; mapping.zScale = 0.5f;
    mapping.meshScale = 0.12f;
    mapping.appearanceLow = { 0.3f, 0.7f, 1.0f, 0.0f };
    mapping.appearanceHigh = { 1.0f, 0.5f, 0.2f, emissionHigh };
    return mapping;
}

// The Polyphonic Imported Model Array starter's Model Key Light and Note Model
// Array glow. "shipped" is what the starter had through 7 Nov 2026: the base
// model's key at intensity 1 straight down the view, glow up to 0.5. "now" is
// what BuiltInVisualModules.h sets today and must stay equal to it.
struct StarterLook
{
    const char* name;
    float rotationX, rotationY, intensity, emissionHigh;
};

constexpr StarterLook kShippedLook { "shipped", -90.0f, 0.0f, 1.0f, 0.5f };
constexpr StarterLook kCurrentLook { "now", -35.0f, -25.0f, 6.0f, 0.2f };

void starterBalanceTests (const std::shared_ptr<const Visual3DScene>& scene, const EngineCase* engines,
                          const char* directory)
{
    std::string error;
    ScoreFrame arpeggio ({ { 48, 0.0f }, { 52, 0.5f }, { 55, 1.0f }, { 60, 1.5f },
                           { 64, 2.0f }, { 67, 2.5f }, { 72, 3.0f }, { 76, 3.5f } });
    const auto frame = arpeggio.at (1.8333f);
    if (const char* sweep = std::getenv ("ARBIT_NOTEINST_SWEEP"))
    {
        // "x,y,intensity,emission;..." looks, printed per engine without checks.
        std::string rest = sweep;
        while (! rest.empty())
        {
            const auto end = rest.find (';');
            const std::string item = rest.substr (0, end);
            rest = end == std::string::npos ? std::string() : rest.substr (end + 1);
            StarterLook look { "sweep", 0, 0, 1, 0 };
            if (std::sscanf (item.c_str(), "%f,%f,%f,%f", &look.rotationX, &look.rotationY, &look.intensity, &look.emissionHigh) != 4)
                continue;
            for (std::size_t e = 0; e < 3; ++e)
            {
                arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
                inputs.linearColor = true;
                inputs.lightOverride = keyLight (*scene, eulerXY (look.rotationX, look.rotationY), look.intensity);
                inputs.canonicalBlockCFrame = frame;
                inputs.noteInstanceMapping = starterMapping (look.emissionHigh);
                Shown shown;
                if (! renderFrame (scene, engines[e], inputs, shown, error))
                    continue;
                const std::string label = std::string (engines[e].name) + "-sweep-" + item;
                if (directory != nullptr)
                    writePng (std::string (directory) + "/" + label + ".png", shown);
                print (label, spreadOf (shown));
            }
        }
        return;
    }
    for (std::size_t e = 0; e < 3; ++e)
    {
        const auto& engine = engines[e];
        Spread spreads[3];
        const char* names[3] = { "shipped", "now", "now-no-glow" };
        for (std::size_t index = 0; index < 3; ++index)
        {
            const auto& look = index == 0 ? kShippedLook : kCurrentLook;
            arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
            inputs.linearColor = true;
            inputs.lightOverride = keyLight (*scene, eulerXY (look.rotationX, look.rotationY), look.intensity);
            inputs.canonicalBlockCFrame = frame;
            inputs.noteInstanceMapping = starterMapping (index == 2 ? 0.0f : look.emissionHigh);
            const std::string label = std::string (engine.name) + "-starter-" + names[index];
            Shown shown;
            if (! renderFrame (scene, engine, inputs, shown, error))
            {
                check (false, label + " renders: " + error);
                continue;
            }
            if (directory != nullptr)
                writePng (std::string (directory) + "/" + label + ".png", shown);
            spreads[index] = spreadOf (shown);
            print (label, spreads[index]);
        }
        const auto& shipped = spreads[0];
        const auto& now = spreads[1];
        const auto& noGlow = spreads[2];
        const std::string label = std::string (engine.name) + " starter";
        // Shadow to lit swing as a share of the median. The shipped starter had 0.27 in Raster and
        // Realtime, 0.99 in Path Trace.
        const auto swing = [] (const Spread& s) { return (s.p95 - s.p5) / std::max (s.p50, 0.01f); };
        check (now.pixels > 1000, label + ": the copies cover pixels");
        check (swing (now) > 1.5f * swing (shipped) && swing (now) > 1.0f,
               label + ": the copies show a clear lit and shadow side");
        check (now.p95 > 1.5f * now.p5 + 0.02f, label + ": the lit side is well above the shadow side");
        check (now.p50 > noGlow.p50 + 0.01f && now.p5 > noGlow.p5 + 0.01f,
               label + ": the copies still glow");
    }
}

void lightingTests()
{
    std::string error;
    const auto scene = knotScene (error);
    check (scene != nullptr, "the bundled knot decodes: " + error);
    if (scene == nullptr)
        return;
    const float h = std::sqrt (0.5f);
    struct Light { const char* name; HarmonicMIDI::grid::SceneQuaternion rotation; };
    const Light lights[] {
        { "key-down", { -h, 0.0f, 0.0f, h } },
        { "key-oblique", { -0.2936f, -0.2064f, -0.0651f, 0.9311f } },
    };
    const EngineCase engines[] {
        { "raster", renderer::SceneEngineKind::Raster },
        { "realtime", renderer::SceneEngineKind::Realtime },
        { "pathtrace", renderer::SceneEngineKind::PathTrace },
    };
    ScoreFrame single ({ { 60, 0.0f } });
    const auto singleFrame = single.at (0.25f);
    const char* directory = std::getenv ("ARBIT_NOTEINST_PNG");
    for (const auto& engine : engines)
        for (const auto& light : lights)
        {
            arbitgpu::NativeFixtureSceneRuntimeInputs plain;
            plain.linearColor = true;
            plain.lightOverride = keyLight (*scene, light.rotation);
            auto inPlace = plain;
            inPlace.canonicalBlockCFrame = singleFrame;
            inPlace.noteInstanceMapping = inPlaceMapping();
            const std::pair<const char*, const arbitgpu::NativeFixtureSceneRuntimeInputs*> cases[] {
                { "single", &plain }, { "copy-in-place", &inPlace } };
            Spread spreads[2];
            for (std::size_t index = 0; index < 2; ++index)
            {
                const std::string label = std::string (engine.name) + "-" + light.name + "-" + cases[index].first;
                Shown frame;
                if (! renderFrame (scene, engine, *cases[index].second, frame, error))
                {
                    check (false, label + " renders: " + error);
                    continue;
                }
                if (directory != nullptr)
                    writePng (std::string (directory) + "/" + label + ".png", frame);
                spreads[index] = spreadOf (frame);
                print (label, spreads[index]);
            }
            const auto& a = spreads[0];
            const auto& b = spreads[1];
            const std::string label = std::string (engine.name) + " " + light.name;
            check (a.pixels > 2000 && a.p95 > 1.5f * a.p5 + 0.02f,
                   label + ": the single model has a lit and a shadow side");
            check (std::abs (static_cast<double> (b.pixels) - static_cast<double> (a.pixels)) <= 0.01 * static_cast<double> (a.pixels),
                   label + ": the in-place copy covers the model's pixels");
            const float tolerance = 0.05f * std::max (a.p50, 0.01f) + 0.002f;
            check (std::abs (b.p5 - a.p5) <= tolerance && std::abs (b.p50 - a.p50) <= tolerance
                       && std::abs (b.p95 - a.p95) <= tolerance,
                   label + ": the in-place copy shades as the single model");
        }
    starterBalanceTests (scene, engines, directory);
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
    auto* window = glfwCreateWindow (64, 64, "note-instance-lighting", nullptr, nullptr);
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
        std::printf ("[noteinst] OpenGL renderer %s\n", name.c_str());
        lightingTests();
    }
    glfwDestroyWindow (window);
    glfwTerminate();
    return result;
}

} // namespace

int main()
{
    const int gl = runGl();
    if (gl != 0)
        return gl;
    std::printf ("[noteinst] %d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
