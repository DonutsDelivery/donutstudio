#pragma once

#include "gl_loader.h"
#include "gpu_backend/backend.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <atomic>
#include <limits>

namespace videorender
{
// A compositor input copy, not a replacement for the scene's raw attachments.
// The scene owner and readRawPass continue to expose their original resources.
class TopFirstOpenGlTexture final : public arbitgpu::NativeFixtureSceneFrame
{
public:
    explicit TopFirstOpenGlTexture(arbitgpu::NativeTextureViewDescriptor descriptor)
        : descriptor_(std::move(descriptor)) {}
    ~TopFirstOpenGlTexture() override
    {
        auto* context = reinterpret_cast<GLFWwindow*>(descriptor_.deviceOrContextIdentity);
        auto* previous = glfwGetCurrentContext();
        if (previous != context) glfwMakeContextCurrent(context);
        if (glfwGetCurrentContext() == context)
        {
            const auto texture = static_cast<GLuint>(descriptor_.imageHandle);
            glDeleteTextures(1, &texture);
        }
        if (previous != context) glfwMakeContextCurrent(previous);
    }
    const std::string& backend() const noexcept override { return descriptor_.backend; }
    std::uint32_t width() const noexcept override { return descriptor_.width; }
    std::uint32_t height() const noexcept override { return descriptor_.height; }
    std::uintptr_t colorImageHandle() const noexcept override { return descriptor_.imageHandle; }
    std::uintptr_t colorTextureViewHandle() const noexcept override { return descriptor_.textureViewHandle; }
    arbitgpu::NativeTextureViewDescriptor colorTextureDescriptor() const noexcept override { return descriptor_; }
private:
    arbitgpu::NativeTextureViewDescriptor descriptor_;
};

inline std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> copyTopFirstOpenGlTexture(
    const arbitgl::GlFuncs& gl, const arbitgpu::NativeTextureViewDescriptor& source, std::string& error)
{
    using namespace arbitgpu;
    error.clear();
    if (!source.complete() || source.backend != "opengl"
        || source.rowOrder != NativeTextureRowOrder::BottomFirst
        || source.imageHandle != source.textureViewHandle
        || source.imageHandle > std::numeric_limits<GLuint>::max()
        || source.deviceOrContextIdentity != reinterpret_cast<std::uintptr_t>(glfwGetCurrentContext())
        || !nativeFixtureDimensionsWithinBounds(source.width, source.height)
        || !glIsTexture(static_cast<GLuint>(source.imageHandle)))
    { error = "Native row normalization requires a live bottom-first texture on the current GL context"; return {}; }

    GLenum internal = GL_RGBA8, layout = GL_RGBA, type = GL_UNSIGNED_BYTE;
    if (source.format == NativeTexturePixelFormat::Rgba16Float) { internal = GL_RGBA16F; type = GL_HALF_FLOAT; }
    else if (source.format == NativeTexturePixelFormat::R32Float) { internal = GL_R32F; layout = GL_RED; type = GL_FLOAT; }
    else if (source.format != NativeTexturePixelFormat::Rgba8Unorm)
    { error = "Native row normalization requires RGBA8, linear RGBA16F, or R32F storage"; return {}; }

    GLint priorTexture = 0, width = 0, height = 0, format = 0;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &priorTexture);
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(source.imageHandle));
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &width);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &height);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_INTERNAL_FORMAT, &format);
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(priorTexture));
    if (width != static_cast<GLint>(source.width) || height != static_cast<GLint>(source.height)
        || format != static_cast<GLint>(internal))
    { error = "Native row normalization storage differs from its immutable descriptor"; return {}; }

    GLint priorRead = 0, priorDraw = 0, priorUnpack = 0;
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &priorRead);
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &priorDraw);
    glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &priorUnpack);
    const bool scissor = glIsEnabled(GL_SCISSOR_TEST);
    const bool srgb = glIsEnabled(GL_FRAMEBUFFER_SRGB);
    GLuint texture = 0, framebuffers[2] {};
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    gl.BindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glTexImage2D(GL_TEXTURE_2D, 0, internal, width, height, 0, layout, type, nullptr);
    gl.GenFramebuffers(2, framebuffers);
    gl.BindFramebuffer(GL_READ_FRAMEBUFFER, framebuffers[0]);
    gl.FramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                          static_cast<GLuint>(source.imageHandle), 0);
    const bool sourceReady = gl.CheckFramebufferStatus(GL_READ_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
    gl.BindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffers[1]);
    gl.FramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
    const bool targetReady = gl.CheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_FRAMEBUFFER_SRGB);
    if (sourceReady && targetReady)
        gl.BlitFramebuffer(0, height, width, 0, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    const auto status = glGetError();
    gl.BindFramebuffer(GL_READ_FRAMEBUFFER, static_cast<GLuint>(priorRead));
    gl.BindFramebuffer(GL_DRAW_FRAMEBUFFER, static_cast<GLuint>(priorDraw));
    gl.BindBuffer(GL_PIXEL_UNPACK_BUFFER, static_cast<GLuint>(priorUnpack));
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(priorTexture));
    if (scissor) glEnable(GL_SCISSOR_TEST);
    if (srgb) glEnable(GL_FRAMEBUFFER_SRGB);
    gl.DeleteFramebuffers(2, framebuffers);
    if (!sourceReady || !targetReady || status != GL_NO_ERROR || texture == 0)
    {
        glDeleteTextures(1, &texture);
        error = "Native texture row normalization GPU copy failed";
        return {};
    }
    static std::atomic<std::uint64_t> generations {0};
    auto descriptor = source;
    descriptor.imageHandle = descriptor.textureViewHandle = texture;
    descriptor.rendererGeneration = ++generations;
    descriptor.rowOrder = NativeTextureRowOrder::TopFirst;
    return std::make_shared<const TopFirstOpenGlTexture>(std::move(descriptor));
}
} // namespace videorender
