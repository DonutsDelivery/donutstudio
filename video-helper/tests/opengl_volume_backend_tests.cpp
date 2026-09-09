#include "../src/volume_native_renderer.h"
#include "../src/gl_loader.h"

#include <GLFW/glfw3.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace
{
std::shared_ptr<const videohelper::volume::AdmittedVolume> admittedSphere (
    const videohelper::volume::VolumeRendererCapabilities& capabilities,
    std::uint8_t density, float translateX, std::string& identity)
{
    constexpr std::uint32_t edge = 32;
    videowire::DenseVolumeDescriptor descriptor;
    descriptor.dimensions = { edge, edge, edge };
    descriptor.bounds = { { -1.0f, -1.0f, -1.0f }, { 1.0f, 1.0f, 1.0f } };
    descriptor.transform.localToWorld[12] = translateX;
    descriptor.voxels.resize (static_cast<std::size_t> (edge) * edge * edge);
    for (std::uint32_t z = 0; z < edge; ++z)
        for (std::uint32_t y = 0; y < edge; ++y)
            for (std::uint32_t x = 0; x < edge; ++x)
            {
                const auto fx = (static_cast<float> (x) + 0.5f) / edge * 2.0f - 1.0f;
                const auto fy = (static_cast<float> (y) + 0.5f) / edge * 2.0f - 1.0f;
                const auto fz = (static_cast<float> (z) + 0.5f) / edge * 2.0f - 1.0f;
                if (fx * fx + fy * fy + fz * fz <= 0.42f * 0.42f)
                    descriptor.voxels[(static_cast<std::size_t> (z) * edge + y) * edge + x]
                        = density;
            }

    std::string error;
    auto admitted = videohelper::volume::admitDenseVolume (
        descriptor, {}, capabilities, error);
    if (! admitted)
    {
        std::fprintf (stderr, "dense volume admission failed: %s\n", error.c_str());
        return {};
    }
    identity = admitted->cacheIdentity();
    return std::make_shared<const videohelper::volume::AdmittedVolume> (
        std::move (*admitted));
}

std::vector<std::uint8_t> readPixels (
    const videohelper::volume::NativeVolumeFrame& frame,
    const arbitgl::GlFuncs& gl)
{
    unsigned framebuffer = 0;
    gl.GenFramebuffers (1, &framebuffer);
    gl.BindFramebuffer (GL_FRAMEBUFFER, framebuffer);
    gl.FramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                             static_cast<unsigned> (frame.colorImageHandle()), 0);
    std::vector<std::uint8_t> pixels (
        static_cast<std::size_t> (frame.width()) * frame.height() * 4u);
    glReadPixels (0, 0, static_cast<int> (frame.width()), static_cast<int> (frame.height()),
                  GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    gl.BindFramebuffer (GL_FRAMEBUFFER, 0);
    gl.DeleteFramebuffers (1, &framebuffer);
    return pixels;
}

std::array<std::uint8_t, 4> pixelAt (
    const std::vector<std::uint8_t>& pixels, std::uint32_t width,
    std::uint32_t x, std::uint32_t y)
{
    const auto offset = (static_cast<std::size_t> (y) * width + x) * 4u;
    return { pixels[offset], pixels[offset + 1], pixels[offset + 2], pixels[offset + 3] };
}

std::uint32_t checksum (const std::vector<std::uint8_t>& pixels)
{
    std::uint32_t value = 2166136261u;
    for (const auto byte : pixels)
    {
        value ^= byte;
        value *= 16777619u;
    }
    return value;
}
} // namespace

int main()
{
    if (glfwInit() != GLFW_TRUE) return 77;
    glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint (GLFW_VISIBLE, GLFW_FALSE);
    auto* window = glfwCreateWindow (96, 64, "native OpenGL volume", nullptr, nullptr);
    if (window == nullptr)
    {
        glfwTerminate();
        return 77;
    }
    glfwMakeContextCurrent (window);

    arbitgl::GlFuncs gl;
    std::string missing;
    if (! arbitgl::loadGlFunctions (gl, missing))
    {
        std::fprintf (stderr, "OpenGL loader failed: %s\n", missing.c_str());
        return 1;
    }

    auto& backend = videohelper::volume::nativeVolumeExecutionBackend();
    const auto capabilities = backend.capabilities();
    if (! capabilities.volume.nativeGpuAvailable
        || ! capabilities.volume.volumeRaymarch
        || ! capabilities.volume.denseVolumeUpload
        || capabilities.volume.sparseBrickUpload
        || capabilities.backend != "opengl")
    {
        std::fprintf (stderr, "strict OpenGL volume capabilities are unavailable\n");
        return 2;
    }

    std::string identity;
    const auto volume = admittedSphere (capabilities.volume, 255, 0.0f, identity);
    if (volume == nullptr) return 3;

    unsigned preserved2D = 0;
    unsigned preserved3D = 0;
    gl.ActiveTexture (GL_TEXTURE3);
    glGenTextures (1, &preserved2D);
    glGenTextures (1, &preserved3D);
    glBindTexture (GL_TEXTURE_2D, preserved2D);
    glBindTexture (GL_TEXTURE_3D, preserved3D);
    glEnable (GL_BLEND);
    glEnable (GL_DEPTH_TEST);
    glEnable (GL_CULL_FACE);
    glEnable (GL_SCISSOR_TEST);
    glEnable (GL_RASTERIZER_DISCARD);
    glEnable (GL_COLOR_LOGIC_OP);
    glEnable (GL_FRAMEBUFFER_SRGB);
    glPolygonMode (GL_FRONT_AND_BACK, GL_LINE);
    glColorMask (GL_FALSE, GL_TRUE, GL_FALSE, GL_TRUE);
    glPixelStorei (GL_UNPACK_ALIGNMENT, 8);

    videohelper::volume::NativeVolumeRenderer renderer;
    videohelper::volume::NativeVolumeRenderedFrame preview;
    videohelper::volume::NativeVolumeRenderedFrame exportFrame;
    std::string error;
    if (! renderer.renderPreview (volume, 96, 64,
                                  videohelper::volume::kNativeVolumeGpuCapability,
                                  preview, error)
        || ! renderer.renderExport (volume, 96, 64,
                                    videohelper::volume::kNativeVolumeGpuCapability,
                                    exportFrame, error))
    {
        std::fprintf (stderr, "native volume preview/export failed: %s\n", error.c_str());
        return 4;
    }
    if (preview.use != videohelper::volume::NativeVolumeRenderUse::Preview
        || exportFrame.use != videohelper::volume::NativeVolumeRenderUse::Export
        || preview.volumeIdentity != identity || exportFrame.volumeIdentity != identity
        || preview.nativeFrame == nullptr || exportFrame.nativeFrame == nullptr)
    {
        std::fprintf (stderr, "preview/export receipts do not own the admitted native frames\n");
        return 5;
    }

    int activeTexture = 0;
    int texture2D = 0;
    int texture3D = 0;
    int unpackAlignment = 0;
    int polygonMode = 0;
    unsigned char colorMask[4] {};
    glGetIntegerv (GL_ACTIVE_TEXTURE, &activeTexture);
    glGetIntegerv (GL_TEXTURE_BINDING_2D, &texture2D);
    glGetIntegerv (GL_TEXTURE_BINDING_3D, &texture3D);
    glGetIntegerv (GL_UNPACK_ALIGNMENT, &unpackAlignment);
    glGetIntegerv (GL_POLYGON_MODE, &polygonMode);
    glGetBooleanv (GL_COLOR_WRITEMASK, colorMask);
    if (activeTexture != GL_TEXTURE3 || texture2D != static_cast<int> (preserved2D)
        || texture3D != static_cast<int> (preserved3D) || unpackAlignment != 8
        || polygonMode != GL_LINE
        || colorMask[0] != GL_FALSE || colorMask[1] != GL_TRUE
        || colorMask[2] != GL_FALSE || colorMask[3] != GL_TRUE
        || glIsEnabled (GL_BLEND) != GL_TRUE || glIsEnabled (GL_DEPTH_TEST) != GL_TRUE
        || glIsEnabled (GL_CULL_FACE) != GL_TRUE || glIsEnabled (GL_SCISSOR_TEST) != GL_TRUE
        || glIsEnabled (GL_RASTERIZER_DISCARD) != GL_TRUE
        || glIsEnabled (GL_COLOR_LOGIC_OP) != GL_TRUE
        || glIsEnabled (GL_FRAMEBUFFER_SRGB) != GL_TRUE)
    {
        std::fprintf (stderr, "OpenGL volume execution did not restore caller state\n");
        return 6;
    }

    glDisable (GL_CULL_FACE);
    glDisable (GL_SCISSOR_TEST);
    glDisable (GL_RASTERIZER_DISCARD);
    glDisable (GL_COLOR_LOGIC_OP);
    glDisable (GL_FRAMEBUFFER_SRGB);
    glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);
    glColorMask (GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glPixelStorei (GL_UNPACK_ALIGNMENT, 4);
    const auto previewPixels = readPixels (*preview.nativeFrame, gl);
    const auto exportPixels = readPixels (*exportFrame.nativeFrame, gl);
    const auto center = pixelAt (previewPixels, 96, 48, 32);
    const auto background = pixelAt (previewPixels, 96, 0, 0);
    if (previewPixels != exportPixels
        || background != std::array<std::uint8_t, 4> { 7, 10, 18, 255 }
        || center == background || center[3] != 255)
    {
        std::fprintf (stderr,
            "volume raymarch pixel mismatch center=%u,%u,%u,%u background=%u,%u,%u,%u\n",
            center[0], center[1], center[2], center[3],
            background[0], background[1], background[2], background[3]);
        return 7;
    }

    std::string emptyIdentity;
    const auto empty = admittedSphere (capabilities.volume, 0, 0.0f, emptyIdentity);
    videohelper::volume::NativeVolumeRenderedFrame emptyFrame;
    if (empty == nullptr
        || ! renderer.renderPreview (empty, 96, 64,
                                     videohelper::volume::kNativeVolumeGpuCapability,
                                     emptyFrame, error)
        || readPixels (*emptyFrame.nativeFrame, gl) == previewPixels)
    {
        std::fprintf (stderr, "3D density upload did not affect raymarched pixels\n");
        return 8;
    }

    std::string translatedIdentity;
    const auto translated = admittedSphere (capabilities.volume, 255, 12.0f,
                                             translatedIdentity);
    videohelper::volume::NativeVolumeRenderedFrame translatedFrame;
    if (translated == nullptr
        || ! renderer.renderExport (translated, 96, 64,
                                    videohelper::volume::kNativeVolumeGpuCapability,
                                    translatedFrame, error)
        || readPixels (*translatedFrame.nativeFrame, gl) != previewPixels)
    {
        std::fprintf (stderr, "volume transform was not consumed by canonical framing\n");
        return 9;
    }

    videowire::SparseVolumeDescriptor sparse;
    sparse.dimensions = { 2, 2, 2 };
    sparse.brickEdge = 2;
    sparse.bricks = { { 0, 0, 0, std::vector<std::uint8_t> (8, 255) } };
    if (videohelper::volume::admitSparseVolume (
            sparse, {}, capabilities.volume, error).has_value()
        || error != "renderer lacks sparse brick upload capability")
    {
        std::fprintf (stderr, "unsupported sparse uploads did not fail closed\n");
        return 10;
    }

    const auto previewTexture = static_cast<unsigned> (
        preview.nativeFrame->colorImageHandle());
    const auto exportTexture = static_cast<unsigned> (
        exportFrame.nativeFrame->colorImageHandle());
    if (glIsTexture (previewTexture) != GL_TRUE || glIsTexture (exportTexture) != GL_TRUE)
    {
        std::fprintf (stderr, "native volume frames do not own live color textures\n");
        return 11;
    }
    emptyFrame = {};
    translatedFrame = {};
    glfwMakeContextCurrent (nullptr);
    preview = {};
    exportFrame = {};
    if (glfwGetCurrentContext() != nullptr)
    {
        std::fprintf (stderr, "volume frame deletion did not restore the caller context\n");
        return 12;
    }
    glfwMakeContextCurrent (window);
    if (glIsTexture (previewTexture) != GL_FALSE || glIsTexture (exportTexture) != GL_FALSE)
    {
        std::fprintf (stderr, "native volume frame leaked a GPU texture\n");
        return 13;
    }

    std::printf (
        "opengl-volume PASS backend=%s volume=%s center=%u,%u,%u,%u "
        "background=%u,%u,%u,%u hash=%u preview_export_equal=yes textures_deleted=yes\n",
        capabilities.backend.c_str(), identity.c_str(),
        center[0], center[1], center[2], center[3],
        background[0], background[1], background[2], background[3],
        checksum (previewPixels));

    glDeleteTextures (1, &preserved2D);
    glDeleteTextures (1, &preserved3D);
    glfwMakeContextCurrent (nullptr);
    glfwDestroyWindow (window);
    glfwTerminate();
    return 0;
}
