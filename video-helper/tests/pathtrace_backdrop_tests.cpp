// Raster, Realtime and Path Trace draw the same backdrop (slice ptbg). The cube
// fixture has no sky, so its corners are the backdrop. Each engine renders it
// through FixtureSceneRenderer on the OpenGL backend, with an 8-bit and with a
// linear Color target (the imported scene route), and with an Environment Light
// hidden from the camera. The checks compare the corner pixels as the
// compositor shows them: 8-bit bytes as they are, linear values sRGB encoded.
// The OptiX device needs the kernel wiring this target lacks, so it traces on CPU.
//
// Path Trace also renders denoised. A denoiser must leave the backdrop as the
// raster clears it.
//
// ARBIT_PTBG_PNG=dir writes every frame there as <case>-<engine>.png.
// It skips with 77 when no hardware OpenGL 4.3 context exists.

#include "../src/fixture_scene_renderer.h"
#include "../src/gl_loader.h"
#include "../src/gpu_backend/backend.h"
#include "support/fixture_scene.h"

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
using HarmonicMIDI::grid::SceneLightKind;
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

// A frame as the compositor shows it: RGB bytes, top row first.
struct Shown
{
    std::uint32_t width = 0, height = 0;
    std::vector<std::uint8_t> rgb;
    std::array<std::uint8_t, 3> at (std::uint32_t x, std::uint32_t y) const
    {
        const std::size_t i = (static_cast<std::size_t> (y) * width + x) * 3;
        return { rgb[i], rgb[i + 1], rgb[i + 2] };
    }
};

bool show (const arbitgpu::NativeRawPassPixels& pixels, bool bottomFirst, Shown& out)
{
    const bool half = pixels.format == arbitgpu::NativeTexturePixelFormat::Rgba16Float;
    if (! half && pixels.format != arbitgpu::NativeTexturePixelFormat::Rgba8Unorm)
        return false;
    const std::size_t stride = half ? 8 : 4, count = static_cast<std::size_t> (pixels.width) * pixels.height;
    if (pixels.bytes.size() != count * stride)
        return false;
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
                if (! half)
                {
                    target[c] = pixels.bytes[source + c];
                    continue;
                }
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

struct EngineCase
{
    const char* name;
    renderer::SceneEngineKind kind;
    bool denoise;
};

bool renderFrame (const std::shared_ptr<const Visual3DScene>& scene, const EngineCase& engine, bool linear,
                  Shown& out, std::string& error)
{
    renderer::FixtureSceneRenderer frames (arbitgpu::nativeFixtureSceneBackend());
    renderer::SceneEngineSelection selection;
    selection.engine = engine.kind;
    selection.exportSamples = 16;
    selection.denoise = engine.denoise;
    frames.selectEngine (selection);
    arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
    inputs.linearColor = linear;
    renderer::RenderedFrame frame;
    if (! frames.renderExport (scene, std::shared_ptr<const renderer::AdmittedSurfaceMaterialBinding> {}, {}, inputs,
                               { 96, 54 }, renderer::kNativeGpuCapability, frame, error))
        return false;
    glFinish();
    if (frame.nativeFrame == nullptr)
    {
        error = "no frame";
        return false;
    }
    // The raster keeps its passes readable. A Path Trace frame keeps CPU bytes only
    // when render outputs are read, so its Color comes from the texture the
    // compositor samples.
    arbitgpu::NativeRawPassPixels pixels;
    bool bottomFirst = true;
    if (! frame.nativeFrame->readRawPass (renderpassoutput::Output::Color, pixels, error))
    {
        const auto descriptor = frame.nativeFrame->colorTextureDescriptor();
        const bool half = descriptor.format == arbitgpu::NativeTexturePixelFormat::Rgba16Float;
        if (! half && descriptor.format != arbitgpu::NativeTexturePixelFormat::Rgba8Unorm)
            return false;
        error.clear();
        pixels.format = descriptor.format;
        pixels.width = descriptor.width;
        pixels.height = descriptor.height;
        pixels.bytes.assign (static_cast<std::size_t> (descriptor.width) * descriptor.height * (half ? 8 : 4), 0);
        glBindTexture (GL_TEXTURE_2D, static_cast<GLuint> (descriptor.textureViewHandle));
        glPixelStorei (GL_PACK_ALIGNMENT, 1);
        glGetTexImage (GL_TEXTURE_2D, 0, GL_RGBA, half ? kGlHalfFloat : GL_UNSIGNED_BYTE, pixels.bytes.data());
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

std::string text (const std::array<std::uint8_t, 3>& rgb)
{
    return "(" + std::to_string (rgb[0]) + ", " + std::to_string (rgb[1]) + ", " + std::to_string (rgb[2]) + ")";
}

void backdropTests()
{
    const auto plain = std::make_shared<const Visual3DScene> (videohelper::fixture3d::makeScene());
    auto hiddenScene = videohelper::fixture3d::makeScene();
    auto& environment = hiddenScene.lights[hiddenScene.lightCount];
    environment.id = HarmonicMIDI::grid::SceneLightId { 302 };
    environment.kind = SceneLightKind::Environment;
    environment.color = { 0.2f, 0.3f, 0.5f };
    environment.intensity = 1.0f;
    environment.environmentCameraVisible = false;
    ++hiddenScene.lightCount;
    const auto hidden = std::make_shared<const Visual3DScene> (std::move (hiddenScene));

    struct SceneCase
    {
        const char* name;
        std::shared_ptr<const Visual3DScene> scene;
        bool linear;
        std::array<std::uint8_t, 3> backdrop;
    };
    // The raster clears 7, 10, 18 over 255 into either target. The compositor shows
    // the linear target encoded, as slate. A hidden environment clears to black.
    const SceneCase sceneCases[] {
        { "stage-8bit", plain, false, { 7, 10, 18 } },
        { "stage-linear", plain, true, { 46, 56, 75 } },
        { "hidden-linear", hidden, true, { 0, 0, 0 } },
    };
    const EngineCase engines[] {
        { "raster", renderer::SceneEngineKind::Raster, false },
        { "realtime", renderer::SceneEngineKind::Realtime, false },
        { "pathtrace", renderer::SceneEngineKind::PathTrace, false },
        { "pathtrace-denoised", renderer::SceneEngineKind::PathTrace, true },
    };
    const char* directory = std::getenv ("ARBIT_PTBG_PNG");
    for (const auto& sceneCase : sceneCases)
        for (const auto& engine : engines)
        {
            const std::string label = std::string (sceneCase.name) + " " + engine.name;
            Shown frame;
            std::string error;
            if (! renderFrame (sceneCase.scene, engine, sceneCase.linear, frame, error))
            {
                check (false, label + " renders: " + error);
                continue;
            }
            if (directory != nullptr)
                writePng (std::string (directory) + "/" + sceneCase.name + "-" + engine.name + ".png", frame);
            const std::array<std::array<std::uint8_t, 3>, 4> corners {
                frame.at (0, 0), frame.at (frame.width - 1, 0),
                frame.at (0, frame.height - 1), frame.at (frame.width - 1, frame.height - 1) };
            const std::array<std::uint8_t, 3> centre = frame.at (frame.width / 2, frame.height / 2);
            std::printf ("[ptbg] %-14s %-18s corner %s centre %s\n", sceneCase.name, engine.name,
                         text (corners[0]).c_str(), text (centre).c_str());
            bool same = true;
            for (const auto& corner : corners)
                same = same && corner == sceneCase.backdrop;
            check (same, label + " backdrop is " + text (sceneCase.backdrop) + ", got " + text (corners[0]));
            check (centre != sceneCase.backdrop, label + " draws the cube in the centre");
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
    auto* window = glfwCreateWindow (64, 64, "pathtrace-backdrop", nullptr, nullptr);
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
        std::printf ("[ptbg] OpenGL renderer %s\n", name.c_str());
        backdropTests();
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
    if (failures != 0)
    {
        std::fprintf (stderr, "%d of %d backdrop checks failed\n", failures, checks);
        return 1;
    }
    std::printf ("PASS: %d backdrop checks\n", checks);
    return 0;
}
