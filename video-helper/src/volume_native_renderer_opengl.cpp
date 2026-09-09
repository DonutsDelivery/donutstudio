#include "volume_native_renderer.h"
#include "gl_loader.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>

namespace videohelper::volume
{
namespace
{
constexpr const char* kUnavailable
    = "native OpenGL volume execution requires a current OpenGL 3.3 context";
constexpr std::size_t kMaximumVolumeBytes = 512u * 1024u * 1024u;
constexpr std::uint32_t kMaximumRenderExtent = 4096;
constexpr std::uint32_t kMaximumRaySteps = 256;

const char* kVertexShader = R"glsl(#version 330 core
const vec2 positions[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
void main() { gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0); }
)glsl";

const char* kFragmentShader = R"glsl(#version 330 core
layout(location = 0) out vec4 outColor;
uniform sampler3D uVolume;
uniform mat4 uWorldToLocal;
uniform vec3 uBoundsMinimum;
uniform vec3 uBoundsMaximum;
uniform vec3 uCameraOrigin;
uniform vec3 uCameraTarget;
uniform vec2 uExtent;
uniform float uCameraHalfHeight;
uniform int uStepCount;

bool intersectBox(vec3 origin, vec3 direction, out float nearT, out float farT)
{
    vec3 inverseDirection = 1.0 / direction;
    vec3 first = (uBoundsMinimum - origin) * inverseDirection;
    vec3 second = (uBoundsMaximum - origin) * inverseDirection;
    vec3 nearAxis = min(first, second);
    vec3 farAxis = max(first, second);
    nearT = max(max(nearAxis.x, nearAxis.y), nearAxis.z);
    farT = min(min(farAxis.x, farAxis.y), farAxis.z);
    return farT >= max(nearT, 0.0);
}

void main()
{
    vec2 uv = (2.0 * gl_FragCoord.xy - uExtent) / uExtent.y;
    vec3 forward = uCameraTarget - uCameraOrigin;
    vec3 worldDirection = normalize(forward
        + vec3(uv.x * uCameraHalfHeight, uv.y * uCameraHalfHeight, 0.0));
    vec3 localOrigin = (uWorldToLocal * vec4(uCameraOrigin, 1.0)).xyz;
    vec3 localDirection = (uWorldToLocal * vec4(worldDirection, 0.0)).xyz;

    float nearT = 0.0;
    float farT = 0.0;
    vec3 background = vec3(7.0 / 255.0, 10.0 / 255.0, 18.0 / 255.0);
    if (!intersectBox(localOrigin, localDirection, nearT, farT))
    {
        outColor = vec4(background, 1.0);
        return;
    }

    nearT = max(nearT, 0.0);
    float stepLength = max((farT - nearT) / float(max(uStepCount, 1)), 0.000001);
    vec3 localExtent = uBoundsMaximum - uBoundsMinimum;
    vec3 accumulated = vec3(0.0);
    float transmittance = 1.0;
    for (int step = 0; step < 256; ++step)
    {
        if (step >= uStepCount || transmittance < 0.004) break;
        float t = nearT + (float(step) + 0.5) * stepLength;
        vec3 localPoint = localOrigin + localDirection * t;
        vec3 texturePoint = (localPoint - uBoundsMinimum) / localExtent;
        float density = max(texture(uVolume, clamp(texturePoint, 0.0, 1.0)).r, 0.0);
        float alpha = 1.0 - exp(-density * 8.0 * stepLength);
        vec3 sampleColor = mix(vec3(0.08, 0.24, 0.62), vec3(1.0, 0.46, 0.12),
                               clamp(density, 0.0, 1.0));
        accumulated += transmittance * alpha * sampleColor;
        transmittance *= 1.0 - alpha;
    }
    outColor = vec4(accumulated + transmittance * background, 1.0);
}
)glsl";

struct TextureFormat final
{
    int internalFormat = 0;
    unsigned type = 0;
};

bool textureFormat (videowire::VolumeVoxelFormat format, TextureFormat& result) noexcept
{
    switch (format)
    {
        case videowire::VolumeVoxelFormat::densityU8:
            result = { GL_R8, GL_UNSIGNED_BYTE };
            return true;
        case videowire::VolumeVoxelFormat::densityF16:
            result = { GL_R16F, GL_HALF_FLOAT };
            return true;
        case videowire::VolumeVoxelFormat::densityF32:
            result = { GL_R32F, GL_FLOAT };
            return true;
    }
    return false;
}

bool invertAffine (const std::array<float, 16>& source,
                   std::array<float, 16>& inverse) noexcept
{
    constexpr double tolerance = 1.0e-7;
    if (std::abs (source[3]) > tolerance || std::abs (source[7]) > tolerance
        || std::abs (source[11]) > tolerance || std::abs (source[15] - 1.0f) > tolerance)
        return false;

    double augmented[4][8] {};
    for (int row = 0; row < 4; ++row)
        for (int column = 0; column < 4; ++column)
        {
            augmented[row][column] = source[static_cast<std::size_t> (column * 4 + row)];
            augmented[row][column + 4] = row == column ? 1.0 : 0.0;
        }

    for (int column = 0; column < 4; ++column)
    {
        int pivot = column;
        for (int row = column + 1; row < 4; ++row)
            if (std::abs (augmented[row][column]) > std::abs (augmented[pivot][column]))
                pivot = row;
        if (! std::isfinite (augmented[pivot][column])
            || std::abs (augmented[pivot][column]) <= tolerance)
            return false;
        if (pivot != column)
            for (int value = 0; value < 8; ++value)
                std::swap (augmented[pivot][value], augmented[column][value]);

        const auto divisor = augmented[column][column];
        for (int value = 0; value < 8; ++value) augmented[column][value] /= divisor;
        for (int row = 0; row < 4; ++row)
        {
            if (row == column) continue;
            const auto multiplier = augmented[row][column];
            for (int value = 0; value < 8; ++value)
                augmented[row][value] -= multiplier * augmented[column][value];
        }
    }

    for (int row = 0; row < 4; ++row)
        for (int column = 0; column < 4; ++column)
        {
            const auto value = augmented[row][column + 4];
            if (! std::isfinite (value)
                || value > std::numeric_limits<float>::max()
                || value < -std::numeric_limits<float>::max())
                return false;
            inverse[static_cast<std::size_t> (column * 4 + row)]
                = static_cast<float> (value);
        }
    return true;
}

std::array<float, 3> transformPoint (const std::array<float, 16>& matrix,
                                     float x, float y, float z) noexcept
{
    return {
        matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12],
        matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13],
        matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14]
    };
}

bool cameraForVolume (const AdmittedVolume& volume,
                      std::array<float, 3>& origin,
                      std::array<float, 3>& target,
                      float& halfHeight) noexcept
{
    const auto& bounds = volume.bounds();
    const auto& matrix = volume.transform().localToWorld;
    std::array<float, 3> minimum {
        std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()
    };
    std::array<float, 3> maximum {
        -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max()
    };
    for (int z = 0; z < 2; ++z)
        for (int y = 0; y < 2; ++y)
            for (int x = 0; x < 2; ++x)
            {
                const auto point = transformPoint (
                    matrix,
                    x == 0 ? bounds.minimum.x : bounds.maximum.x,
                    y == 0 ? bounds.minimum.y : bounds.maximum.y,
                    z == 0 ? bounds.minimum.z : bounds.maximum.z);
                for (int axis = 0; axis < 3; ++axis)
                {
                    minimum[axis] = std::min (minimum[axis], point[axis]);
                    maximum[axis] = std::max (maximum[axis], point[axis]);
                }
            }

    const auto extent = std::max ({ maximum[0] - minimum[0], maximum[1] - minimum[1],
                                    maximum[2] - minimum[2] });
    if (! std::isfinite (extent) || extent <= 1.0e-6f) return false;
    for (int axis = 0; axis < 3; ++axis)
    {
        target[axis] = (minimum[axis] + maximum[axis]) * 0.5f;
        if (! std::isfinite (target[axis])) return false;
    }
    halfHeight = extent * 0.62f;
    origin = { target[0], target[1], target[2] + extent * 2.0f };
    return std::all_of (origin.begin(), origin.end(),
                        [] (float value) { return std::isfinite (value); });
}

unsigned compileShader (const arbitgl::GlFuncs& gl, unsigned type,
                        const char* source, std::string& error)
{
    const auto shader = gl.CreateShader (type);
    gl.ShaderSource (shader, 1, &source, nullptr);
    gl.CompileShader (shader);
    int compiled = 0;
    gl.GetShaderiv (shader, GL_COMPILE_STATUS, &compiled);
    if (compiled == GL_TRUE) return shader;

    char log[1024] {};
    int length = 0;
    gl.GetShaderInfoLog (shader, static_cast<int> (sizeof (log)), &length, log);
    error = std::string ("OpenGL native volume shader compilation failed: ")
          + std::string (log, static_cast<std::size_t> (std::max (length, 0)));
    gl.DeleteShader (shader);
    return 0;
}

class OpenGlVolumeFrame final : public NativeVolumeFrame
{
public:
    ~OpenGlVolumeFrame() override
    {
        if ((colorTexture_ == 0 && volumeTexture_ == 0) || ownerContext_ == nullptr) return;
        auto* previousContext = glfwGetCurrentContext();
        if (previousContext != ownerContext_) glfwMakeContextCurrent (ownerContext_);
        if (glfwGetCurrentContext() == ownerContext_)
        {
            if (colorTexture_ != 0) glDeleteTextures (1, &colorTexture_);
            if (volumeTexture_ != 0) glDeleteTextures (1, &volumeTexture_);
        }
        if (previousContext != ownerContext_) glfwMakeContextCurrent (previousContext);
    }

    const std::string& backend() const noexcept override { return backend_; }
    std::uint32_t width() const noexcept override { return width_; }
    std::uint32_t height() const noexcept override { return height_; }
    std::uintptr_t colorImageHandle() const noexcept override { return colorTexture_; }
    std::uintptr_t colorTextureViewHandle() const noexcept override { return colorTexture_; }

    std::string backend_ = "opengl";
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    unsigned colorTexture_ = 0;
    unsigned volumeTexture_ = 0;
    GLFWwindow* ownerContext_ = nullptr;
};

class OpenGlVolumeExecutionBackend final : public NativeVolumeExecutionBackend
{
public:
    NativeVolumeExecutionCapabilities capabilities() const override
    {
        NativeVolumeExecutionCapabilities result;
        if (glfwGetCurrentContext() == nullptr) return result;

        int major = 0;
        int minor = 0;
        int maximumTextureSize = 0;
        int maximumTexture3DSize = 0;
        glGetIntegerv (GL_MAJOR_VERSION, &major);
        glGetIntegerv (GL_MINOR_VERSION, &minor);
        glGetIntegerv (GL_MAX_TEXTURE_SIZE, &maximumTextureSize);
        glGetIntegerv (GL_MAX_3D_TEXTURE_SIZE, &maximumTexture3DSize);
        if (major < 3 || (major == 3 && minor < 3)
            || maximumTextureSize <= 0 || maximumTexture3DSize <= 0)
            return result;

        const auto maximumExtent = static_cast<std::uint32_t> (
            std::min (maximumTextureSize, static_cast<int> (kMaximumRenderExtent)));
        result.volume.nativeGpuAvailable = true;
        result.volume.volumeRaymarch = true;
        result.volume.denseVolumeUpload = true;
        result.volume.sparseBrickUpload = false;
        result.volume.maxTexture3DDimension = static_cast<std::uint32_t> (
            std::min (maximumTexture3DSize, 2048));
        result.volume.maxBrickEdge = result.volume.maxTexture3DDimension;
        result.volume.maxSparseBrickCount = 65536;
        result.volume.maxVolumeBytes = kMaximumVolumeBytes;
        result.backend = "opengl";
        result.maxRenderExtent = maximumExtent;
        result.maxRenderPixels = std::min<std::uint64_t> (
            static_cast<std::uint64_t> (maximumExtent) * maximumExtent,
            static_cast<std::uint64_t> (kMaximumRenderExtent) * kMaximumRenderExtent);
        return result;
    }

    NativeVolumeSubmission render (const NativeVolumeDrawRequest& request) override
    {
        NativeVolumeSubmission result;
        const auto available = capabilities();
        if (! available.volume.nativeGpuAvailable)
        {
            result.error = kUnavailable;
            return result;
        }
        if (request.volume == nullptr || request.width == 0 || request.height == 0
            || request.width > available.maxRenderExtent
            || request.height > available.maxRenderExtent
            || static_cast<std::uint64_t> (request.width) * request.height
                > available.maxRenderPixels
            || request.volume->bytes().size() > available.volume.maxVolumeBytes)
        {
            result.error = "OpenGL native volume request exceeds backend limits";
            return result;
        }
        if (request.volume->storage() != videowire::VolumeStorage::dense)
        {
            result.error = "OpenGL native volume execution currently admits dense volumes";
            return result;
        }

        const auto& dimensions = request.volume->dimensions();
        if (dimensions.width == 0 || dimensions.height == 0 || dimensions.depth == 0
            || dimensions.width > available.volume.maxTexture3DDimension
            || dimensions.height > available.volume.maxTexture3DDimension
            || dimensions.depth > available.volume.maxTexture3DDimension)
        {
            result.error = "OpenGL native volume dimensions exceed the 3D texture limit";
            return result;
        }

        TextureFormat format;
        std::array<float, 16> worldToLocal {};
        std::array<float, 3> cameraOrigin {};
        std::array<float, 3> cameraTarget {};
        float cameraHalfHeight = 0.0f;
        if (! textureFormat (request.volume->format(), format)
            || ! invertAffine (request.volume->transform().localToWorld, worldToLocal)
            || ! cameraForVolume (*request.volume, cameraOrigin, cameraTarget, cameraHalfHeight))
        {
            result.error = "OpenGL native volume requires an invertible finite affine transform";
            return result;
        }

        arbitgl::GlFuncs gl;
        std::string missing;
        if (! arbitgl::loadGlFunctions (gl, missing))
        {
            result.error = "OpenGL native volume loader failed: " + missing;
            return result;
        }

        int previousDrawFramebuffer = 0;
        int previousReadFramebuffer = 0;
        int previousProgram = 0;
        int previousVertexArray = 0;
        int previousTexture2D = 0;
        int previousTexture3D = 0;
        int previousUnpackAlignment = 0;
        int previousViewport[4] {};
        int previousPolygonMode = 0;
        unsigned char previousColorMask[4] {};
        const bool blendWasEnabled = glIsEnabled (GL_BLEND) == GL_TRUE;
        const bool depthWasEnabled = glIsEnabled (GL_DEPTH_TEST) == GL_TRUE;
        const bool cullWasEnabled = glIsEnabled (GL_CULL_FACE) == GL_TRUE;
        const bool scissorWasEnabled = glIsEnabled (GL_SCISSOR_TEST) == GL_TRUE;
        const bool rasterizerDiscardWasEnabled = glIsEnabled (GL_RASTERIZER_DISCARD) == GL_TRUE;
        const bool colorLogicOpWasEnabled = glIsEnabled (GL_COLOR_LOGIC_OP) == GL_TRUE;
        const bool framebufferSrgbWasEnabled = glIsEnabled (GL_FRAMEBUFFER_SRGB) == GL_TRUE;
        glGetIntegerv (GL_DRAW_FRAMEBUFFER_BINDING, &previousDrawFramebuffer);
        glGetIntegerv (GL_READ_FRAMEBUFFER_BINDING, &previousReadFramebuffer);
        glGetIntegerv (GL_CURRENT_PROGRAM, &previousProgram);
        glGetIntegerv (GL_VERTEX_ARRAY_BINDING, &previousVertexArray);
        glGetIntegerv (GL_TEXTURE_BINDING_2D, &previousTexture2D);
        glGetIntegerv (GL_TEXTURE_BINDING_3D, &previousTexture3D);
        glGetIntegerv (GL_UNPACK_ALIGNMENT, &previousUnpackAlignment);
        glGetIntegerv (GL_VIEWPORT, previousViewport);
        glGetIntegerv (GL_POLYGON_MODE, &previousPolygonMode);
        glGetBooleanv (GL_COLOR_WRITEMASK, previousColorMask);

        unsigned framebuffer = 0;
        unsigned vertexArray = 0;
        unsigned vertexShader = 0;
        unsigned fragmentShader = 0;
        unsigned program = 0;
        auto frame = std::make_shared<OpenGlVolumeFrame>();
        frame->width_ = request.width;
        frame->height_ = request.height;
        frame->ownerContext_ = glfwGetCurrentContext();

        auto restore = [&]
        {
            gl.BindFramebuffer (GL_DRAW_FRAMEBUFFER,
                                static_cast<unsigned> (previousDrawFramebuffer));
            gl.BindFramebuffer (GL_READ_FRAMEBUFFER,
                                static_cast<unsigned> (previousReadFramebuffer));
            gl.UseProgram (static_cast<unsigned> (previousProgram));
            gl.BindVertexArray (static_cast<unsigned> (previousVertexArray));
            glBindTexture (GL_TEXTURE_2D, static_cast<unsigned> (previousTexture2D));
            glBindTexture (GL_TEXTURE_3D, static_cast<unsigned> (previousTexture3D));
            glPixelStorei (GL_UNPACK_ALIGNMENT, previousUnpackAlignment);
            glViewport (previousViewport[0], previousViewport[1],
                        previousViewport[2], previousViewport[3]);
            glPolygonMode (GL_FRONT_AND_BACK, static_cast<unsigned> (previousPolygonMode));
            glColorMask (previousColorMask[0], previousColorMask[1],
                         previousColorMask[2], previousColorMask[3]);
            if (blendWasEnabled) glEnable (GL_BLEND); else glDisable (GL_BLEND);
            if (depthWasEnabled) glEnable (GL_DEPTH_TEST); else glDisable (GL_DEPTH_TEST);
            if (cullWasEnabled) glEnable (GL_CULL_FACE); else glDisable (GL_CULL_FACE);
            if (scissorWasEnabled) glEnable (GL_SCISSOR_TEST); else glDisable (GL_SCISSOR_TEST);
            if (rasterizerDiscardWasEnabled) glEnable (GL_RASTERIZER_DISCARD);
            else glDisable (GL_RASTERIZER_DISCARD);
            if (colorLogicOpWasEnabled) glEnable (GL_COLOR_LOGIC_OP);
            else glDisable (GL_COLOR_LOGIC_OP);
            if (framebufferSrgbWasEnabled) glEnable (GL_FRAMEBUFFER_SRGB);
            else glDisable (GL_FRAMEBUFFER_SRGB);
            if (framebuffer != 0) gl.DeleteFramebuffers (1, &framebuffer);
            if (vertexArray != 0) gl.DeleteVertexArrays (1, &vertexArray);
            if (program != 0) gl.DeleteProgram (program);
            if (vertexShader != 0) gl.DeleteShader (vertexShader);
            if (fragmentShader != 0) gl.DeleteShader (fragmentShader);
        };
        auto fail = [&] (std::string error)
        {
            restore();
            if (frame->colorTexture_ != 0)
            {
                glDeleteTextures (1, &frame->colorTexture_);
                frame->colorTexture_ = 0;
            }
            if (frame->volumeTexture_ != 0)
            {
                glDeleteTextures (1, &frame->volumeTexture_);
                frame->volumeTexture_ = 0;
            }
            result.error = std::move (error);
            return result;
        };

        glPixelStorei (GL_UNPACK_ALIGNMENT, 1);
        glGenTextures (1, &frame->volumeTexture_);
        glBindTexture (GL_TEXTURE_3D, frame->volumeTexture_);
        glTexParameteri (GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri (GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri (GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri (GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri (GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        gl.TexImage3D (GL_TEXTURE_3D, 0, format.internalFormat,
                       static_cast<int> (dimensions.width),
                       static_cast<int> (dimensions.height),
                       static_cast<int> (dimensions.depth), 0,
                       GL_RED, format.type, request.volume->bytes().data());

        glGenTextures (1, &frame->colorTexture_);
        glBindTexture (GL_TEXTURE_2D, frame->colorTexture_);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA8, static_cast<int> (request.width),
                      static_cast<int> (request.height), 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

        gl.GenFramebuffers (1, &framebuffer);
        gl.BindFramebuffer (GL_FRAMEBUFFER, framebuffer);
        gl.FramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                 GL_TEXTURE_2D, frame->colorTexture_, 0);
        if (gl.CheckFramebufferStatus (GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            return fail ("OpenGL native volume framebuffer is incomplete");

        vertexShader = compileShader (gl, GL_VERTEX_SHADER, kVertexShader, result.error);
        if (vertexShader == 0) return fail (result.error);
        fragmentShader = compileShader (gl, GL_FRAGMENT_SHADER, kFragmentShader, result.error);
        if (fragmentShader == 0) return fail (result.error);
        program = gl.CreateProgram();
        gl.AttachShader (program, vertexShader);
        gl.AttachShader (program, fragmentShader);
        gl.LinkProgram (program);
        int linked = 0;
        gl.GetProgramiv (program, GL_LINK_STATUS, &linked);
        if (linked != GL_TRUE)
        {
            char log[1024] {};
            int length = 0;
            gl.GetProgramInfoLog (program, static_cast<int> (sizeof (log)), &length, log);
            return fail (std::string ("OpenGL native volume program link failed: ")
                         + std::string (log, static_cast<std::size_t> (std::max (length, 0))));
        }

        gl.GenVertexArrays (1, &vertexArray);
        gl.BindVertexArray (vertexArray);
        gl.UseProgram (program);
        int activeTexture = 0;
        glGetIntegerv (GL_ACTIVE_TEXTURE, &activeTexture);
        gl.Uniform1i (gl.GetUniformLocation (program, "uVolume"),
                      activeTexture - GL_TEXTURE0);
        gl.UniformMatrix4fv (gl.GetUniformLocation (program, "uWorldToLocal"),
                             1, GL_FALSE, worldToLocal.data());
        const auto& bounds = request.volume->bounds();
        gl.Uniform3f (gl.GetUniformLocation (program, "uBoundsMinimum"),
                      bounds.minimum.x, bounds.minimum.y, bounds.minimum.z);
        gl.Uniform3f (gl.GetUniformLocation (program, "uBoundsMaximum"),
                      bounds.maximum.x, bounds.maximum.y, bounds.maximum.z);
        gl.Uniform3f (gl.GetUniformLocation (program, "uCameraOrigin"),
                      cameraOrigin[0], cameraOrigin[1], cameraOrigin[2]);
        gl.Uniform3f (gl.GetUniformLocation (program, "uCameraTarget"),
                      cameraTarget[0], cameraTarget[1], cameraTarget[2]);
        gl.Uniform2f (gl.GetUniformLocation (program, "uExtent"),
                      static_cast<float> (request.width), static_cast<float> (request.height));
        gl.Uniform1f (gl.GetUniformLocation (program, "uCameraHalfHeight"), cameraHalfHeight);
        const auto steps = static_cast<int> (std::min<std::uint32_t> (
            kMaximumRaySteps, std::max ({ dimensions.width, dimensions.height, dimensions.depth,
                                         32u }) * 2u));
        gl.Uniform1i (gl.GetUniformLocation (program, "uStepCount"), steps);

        glBindTexture (GL_TEXTURE_3D, frame->volumeTexture_);
        glViewport (0, 0, static_cast<int> (request.width), static_cast<int> (request.height));
        glDisable (GL_BLEND);
        glDisable (GL_DEPTH_TEST);
        glDisable (GL_CULL_FACE);
        glDisable (GL_SCISSOR_TEST);
        glDisable (GL_RASTERIZER_DISCARD);
        glDisable (GL_COLOR_LOGIC_OP);
        glDisable (GL_FRAMEBUFFER_SRGB);
        glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);
        glColorMask (GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glDrawArrays (GL_TRIANGLES, 0, 3);
        glFinish();
        if (glGetError() != GL_NO_ERROR)
            return fail ("OpenGL native volume upload or raymarch reported a GPU error");

        restore();
        result.rendered = true;
        result.frame = std::move (frame);
        return result;
    }
};
} // namespace

NativeVolumeExecutionBackend& nativeVolumeExecutionBackend()
{
    static OpenGlVolumeExecutionBackend backend;
    return backend;
}
} // namespace videohelper::volume
