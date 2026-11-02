#include "cpu_scene_frame.h"
#include "../gl_loader.h"

#include <GLFW/glfw3.h>

#include <atomic>
#include <limits>
#include <new>

namespace arbitgpu
{
namespace
{
// Tagged so a CPU frame generation never equals a fixture preparation's.
std::uint64_t nextCpuFrameGeneration() noexcept
{
    static std::atomic<std::uint64_t> counter { 0 };
    return (std::uint64_t { 1 } << 62) | (counter.fetch_add (1, std::memory_order_relaxed) + 1);
}

class OpenGlCpuSceneFrame final : public NativeFixtureSceneFrame
{
public:
    OpenGlCpuSceneFrame (GLFWwindow* context, unsigned color, unsigned depth, std::uint32_t width,
                         std::uint32_t height, bool linearColor) noexcept
        : context_ (context), color_ (color), depth_ (depth), width_ (width), height_ (height),
          linearColor_ (linearColor), generation_ (nextCpuFrameGeneration())
    {
    }

    ~OpenGlCpuSceneFrame() override
    {
        // Same release rule as the raster's frames: delete in the owner context.
        auto* previous = glfwGetCurrentContext();
        if (previous != context_) glfwMakeContextCurrent (context_);
        if (glfwGetCurrentContext() == context_)
        {
            const unsigned textures[2] { color_, depth_ };
            glDeleteTextures (2, textures);
        }
        if (previous != context_) glfwMakeContextCurrent (previous);
    }

    const std::string& backend() const noexcept override { return backend_; }
    std::uint32_t width() const noexcept override { return width_; }
    std::uint32_t height() const noexcept override { return height_; }
    std::uintptr_t colorImageHandle() const noexcept override { return color_; }
    std::uintptr_t colorTextureViewHandle() const noexcept override { return color_; }
    std::uintptr_t depthImageHandle() const noexcept override { return depth_; }
    std::uintptr_t depthTextureViewHandle() const noexcept override { return depth_; }
    NativeTextureViewDescriptor colorTextureDescriptor() const noexcept override
    {
        return { backend_, NativeTextureViewKind::Texture2D,
            linearColor_ ? NativeTexturePixelFormat::Rgba16Float : NativeTexturePixelFormat::Rgba8Unorm,
            color_, color_, width_, height_, 1, true, reinterpret_cast<std::uintptr_t> (context_), generation_,
            linearColor_ ? colortransform::ColorSpace::LinearSRGB : colortransform::ColorSpace::Unspecified,
            linearColor_ ? colortransform::TransferFunction::Linear : colortransform::TransferFunction::Unspecified,
            NativeTextureRowOrder::BottomFirst };
    }
    NativeTextureViewDescriptor depthTextureDescriptor() const noexcept override
    {
        return { backend_, NativeTextureViewKind::Texture2D, NativeTexturePixelFormat::R32Float,
            depth_, depth_, width_, height_, 1, true, reinterpret_cast<std::uintptr_t> (context_), generation_,
            colortransform::ColorSpace::Unspecified, colortransform::TransferFunction::Unspecified,
            NativeTextureRowOrder::BottomFirst };
    }

private:
    GLFWwindow* context_ = nullptr;
    unsigned color_ = 0;
    unsigned depth_ = 0;
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    bool linearColor_ = false;
    std::uint64_t generation_ = 0;
    const std::string backend_ = "opengl";
};

// Restores the unpack and binding state an upload changes.
struct UploadState final
{
    explicit UploadState (const arbitgl::GlFuncs& gl) : gl_ (gl)
    {
        glGetIntegerv (GL_TEXTURE_BINDING_2D, &texture_);
        glGetIntegerv (GL_PIXEL_UNPACK_BUFFER_BINDING, &unpackBuffer_);
        glGetIntegerv (GL_UNPACK_ALIGNMENT, &alignment_);
        glGetIntegerv (GL_UNPACK_ROW_LENGTH, &rowLength_);
        glGetIntegerv (GL_UNPACK_SKIP_ROWS, &skipRows_);
        glGetIntegerv (GL_UNPACK_SKIP_PIXELS, &skipPixels_);
        gl_.BindBuffer (GL_PIXEL_UNPACK_BUFFER, 0);
        glPixelStorei (GL_UNPACK_ALIGNMENT, 1);
        glPixelStorei (GL_UNPACK_ROW_LENGTH, 0);
        glPixelStorei (GL_UNPACK_SKIP_ROWS, 0);
        glPixelStorei (GL_UNPACK_SKIP_PIXELS, 0);
    }
    ~UploadState()
    {
        glPixelStorei (GL_UNPACK_SKIP_PIXELS, skipPixels_);
        glPixelStorei (GL_UNPACK_SKIP_ROWS, skipRows_);
        glPixelStorei (GL_UNPACK_ROW_LENGTH, rowLength_);
        glPixelStorei (GL_UNPACK_ALIGNMENT, alignment_);
        gl_.BindBuffer (GL_PIXEL_UNPACK_BUFFER, static_cast<unsigned> (unpackBuffer_));
        glBindTexture (GL_TEXTURE_2D, static_cast<unsigned> (texture_));
    }
    UploadState (const UploadState&) = delete;
    UploadState& operator= (const UploadState&) = delete;

private:
    const arbitgl::GlFuncs& gl_;
    int texture_ = 0, unpackBuffer_ = 0, alignment_ = 4, rowLength_ = 0, skipRows_ = 0, skipPixels_ = 0;
};

bool uploadTexture (unsigned texture, int internalFormat, unsigned format, unsigned type,
                    std::uint32_t width, std::uint32_t height, const void* pixels)
{
    glBindTexture (GL_TEXTURE_2D, texture);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D (GL_TEXTURE_2D, 0, internalFormat, static_cast<int> (width), static_cast<int> (height), 0,
                  format, type, pixels);
    return glGetError() == GL_NO_ERROR;
}

// Restores everything the frame layer pass changes.
struct LayerPassState final
{
    explicit LayerPassState (const arbitgl::GlFuncs& gl) : gl_ (gl)
    {
        glGetIntegerv (GL_FRAMEBUFFER_BINDING, &framebuffer_);
        glGetIntegerv (GL_VIEWPORT, viewport_);
        glGetIntegerv (GL_CURRENT_PROGRAM, &program_);
        glGetIntegerv (GL_VERTEX_ARRAY_BINDING, &vertexArray_);
        glGetIntegerv (GL_ACTIVE_TEXTURE, &activeTexture_);
        for (int unit = 0; unit < 3; ++unit)
        {
            gl_.ActiveTexture (GL_TEXTURE0 + static_cast<unsigned> (unit));
            glGetIntegerv (GL_TEXTURE_BINDING_2D, &textures_[unit]);
            glGetIntegerv (GL_SAMPLER_BINDING, &samplers_[unit]);
            gl_.BindSampler (static_cast<unsigned> (unit), 0);
        }
        glGetIntegerv (GL_BLEND_SRC_RGB, &blend_[0]);
        glGetIntegerv (GL_BLEND_DST_RGB, &blend_[1]);
        glGetIntegerv (GL_BLEND_SRC_ALPHA, &blend_[2]);
        glGetIntegerv (GL_BLEND_DST_ALPHA, &blend_[3]);
        glGetBooleanv (GL_COLOR_WRITEMASK, colorMask_);
        for (std::size_t index = 0; index < kSwitches; ++index)
            enabled_[index] = glIsEnabled (switches()[index]);
        for (std::size_t index = 0; index < kSwitches; ++index)
            glDisable (switches()[index]);
        glEnable (GL_BLEND);
        gl_.BlendFuncSeparate (GL_ONE, GL_ONE, GL_ZERO, GL_ONE);
        glColorMask (GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    }
    ~LayerPassState()
    {
        for (std::size_t index = 0; index < kSwitches; ++index)
            if (enabled_[index] != GL_FALSE) glEnable (switches()[index]);
            else glDisable (switches()[index]);
        glColorMask (colorMask_[0], colorMask_[1], colorMask_[2], colorMask_[3]);
        gl_.BlendFuncSeparate (static_cast<unsigned> (blend_[0]), static_cast<unsigned> (blend_[1]),
                               static_cast<unsigned> (blend_[2]), static_cast<unsigned> (blend_[3]));
        for (int unit = 0; unit < 3; ++unit)
        {
            gl_.ActiveTexture (GL_TEXTURE0 + static_cast<unsigned> (unit));
            glBindTexture (GL_TEXTURE_2D, static_cast<unsigned> (textures_[unit]));
            gl_.BindSampler (static_cast<unsigned> (unit), static_cast<unsigned> (samplers_[unit]));
        }
        gl_.ActiveTexture (static_cast<unsigned> (activeTexture_));
        gl_.BindVertexArray (static_cast<unsigned> (vertexArray_));
        gl_.UseProgram (static_cast<unsigned> (program_));
        glViewport (viewport_[0], viewport_[1], viewport_[2], viewport_[3]);
        gl_.BindFramebuffer (GL_FRAMEBUFFER, static_cast<unsigned> (framebuffer_));
    }
    LayerPassState (const LayerPassState&) = delete;
    LayerPassState& operator= (const LayerPassState&) = delete;

private:
    static constexpr std::size_t kSwitches = 6;
    static const unsigned* switches() noexcept
    {
        static const unsigned values[kSwitches] { GL_BLEND, GL_DEPTH_TEST, GL_STENCIL_TEST, GL_SCISSOR_TEST,
                                                  GL_CULL_FACE, GL_FRAMEBUFFER_SRGB };
        return values;
    }
    const arbitgl::GlFuncs& gl_;
    int framebuffer_ = 0, viewport_[4] { 0, 0, 0, 0 }, program_ = 0, vertexArray_ = 0, activeTexture_ = GL_TEXTURE0;
    int textures_[3] { 0, 0, 0 }, samplers_[3] { 0, 0, 0 }, blend_[4] { GL_ONE, GL_ZERO, GL_ONE, GL_ZERO };
    GLboolean colorMask_[4] { GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE };
    GLboolean enabled_[kSwitches] {};
};

const char* const kFrameLayerVertex = R"glsl(#version 330 core
void main() {
    vec2 corner = vec2((gl_VertexID & 1) * 4 - 1, (gl_VertexID & 2) * 2 - 1);
    gl_Position = vec4(corner, 0.0, 1.0);
}
)glsl";

// The tint and coverage of the raster's frame-tinted materials (backend_opengl.cpp,
// uTexturePresence.w). Where the raster keeps a partly transparent colour and
// writes its alpha, this frame has alpha 1 (pathtrace/engine.h), so the colour
// lands over the background here.
const char* const kFrameLayerFragment = R"glsl(#version 330 core
uniform sampler2D uLight;
uniform sampler2D uUvCoverage;
uniform sampler2D uFrame;
uniform int uFrameBottomFirst;
uniform int uFrameLinear;
uniform vec3 uBackground;
layout(location = 0) out vec4 outColor;
vec3 srgbToLinear(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c));
}
void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    vec3 uvCoverage = texelFetch(uUvCoverage, pixel, 0).xyz;
    if (uvCoverage.z <= 0.0) discard;
    vec2 uv = uFrameBottomFirst != 0 ? vec2(uvCoverage.x, 1.0 - uvCoverage.y) : uvCoverage.xy;
    vec4 imagery = texture(uFrame, uv);
    vec3 tint = uFrameLinear != 0 ? imagery.rgb : srgbToLinear(imagery.rgb);
    float alpha = clamp(imagery.a, 0.0, 1.0);
    outColor = vec4(texelFetch(uLight, pixel, 0).rgb * tint * alpha
                    + uBackground * uvCoverage.z * (1.0 - alpha), 0.0);
}
)glsl";

struct FrameLayerProgram final
{
    unsigned program = 0;
    unsigned vertexArray = 0;
    int light = -1, uvCoverage = -1, frame = -1, frameBottomFirst = -1, frameLinear = -1, background = -1;
};

unsigned compileLayerShader (const arbitgl::GlFuncs& gl, unsigned stage, const char* source, std::string& error)
{
    const unsigned shader = gl.CreateShader (stage);
    gl.ShaderSource (shader, 1, &source, nullptr);
    gl.CompileShader (shader);
    int compiled = 0;
    gl.GetShaderiv (shader, GL_COMPILE_STATUS, &compiled);
    if (compiled != 0) return shader;
    char log[1024] {};
    gl.GetShaderInfoLog (shader, sizeof (log), nullptr, log);
    gl.DeleteShader (shader);
    error = std::string ("Path Trace frame layer shader did not compile: ") + log;
    return 0;
}

bool buildLayerProgram (const arbitgl::GlFuncs& gl, FrameLayerProgram& out, std::string& error)
{
    const unsigned vertex = compileLayerShader (gl, GL_VERTEX_SHADER, kFrameLayerVertex, error);
    if (vertex == 0) return false;
    const unsigned fragment = compileLayerShader (gl, GL_FRAGMENT_SHADER, kFrameLayerFragment, error);
    if (fragment == 0)
    {
        gl.DeleteShader (vertex);
        return false;
    }
    out.program = gl.CreateProgram();
    gl.AttachShader (out.program, vertex);
    gl.AttachShader (out.program, fragment);
    gl.LinkProgram (out.program);
    gl.DeleteShader (vertex);
    gl.DeleteShader (fragment);
    int linked = 0;
    gl.GetProgramiv (out.program, GL_LINK_STATUS, &linked);
    if (linked == 0)
    {
        char log[1024] {};
        gl.GetProgramInfoLog (out.program, sizeof (log), nullptr, log);
        gl.DeleteProgram (out.program);
        out.program = 0;
        error = std::string ("Path Trace frame layer shader did not link: ") + log;
        return false;
    }
    gl.GenVertexArrays (1, &out.vertexArray);
    out.light = gl.GetUniformLocation (out.program, "uLight");
    out.uvCoverage = gl.GetUniformLocation (out.program, "uUvCoverage");
    out.frame = gl.GetUniformLocation (out.program, "uFrame");
    out.frameBottomFirst = gl.GetUniformLocation (out.program, "uFrameBottomFirst");
    out.frameLinear = gl.GetUniformLocation (out.program, "uFrameLinear");
    out.background = gl.GetUniformLocation (out.program, "uBackground");
    return true;
}

// Adds each layer to the colour texture, sampling its frame on the GPU.
bool drawFrameLayers (const arbitgl::GlFuncs& gl, const FrameLayerProgram& program, unsigned color,
                      const CpuSceneImage& image, std::string& error)
{
    const LayerPassState state (gl);
    const UploadState unpack (gl);
    unsigned framebuffer = 0;
    gl.GenFramebuffers (1, &framebuffer);
    gl.BindFramebuffer (GL_FRAMEBUFFER, framebuffer);
    gl.FramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0);
    const unsigned drawBuffer = GL_COLOR_ATTACHMENT0;
    gl.DrawBuffers (1, &drawBuffer);
    bool drawn = gl.CheckFramebufferStatus (GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
    if (! drawn)
        error = "Path Trace frame layer target is not renderable";
    glViewport (0, 0, static_cast<int> (image.width), static_cast<int> (image.height));
    gl.UseProgram (program.program);
    gl.BindVertexArray (program.vertexArray);
    gl.Uniform1i (program.light, 0);
    gl.Uniform1i (program.uvCoverage, 1);
    gl.Uniform1i (program.frame, 2);
    gl.Uniform3f (program.background, image.background[0], image.background[1], image.background[2]);
    unsigned textures[2] { 0, 0 };
    glGenTextures (2, textures);
    for (const auto& layer : image.frameLayers)
    {
        if (! drawn) break;
        gl.ActiveTexture (GL_TEXTURE0);
        drawn = uploadTexture (textures[0], GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT, image.width, image.height, layer.light->data());
        gl.ActiveTexture (GL_TEXTURE1);
        drawn = drawn && uploadTexture (textures[1], GL_RGB32F, GL_RGB, GL_FLOAT, image.width, image.height, layer.uvCoverage->data());
        if (! drawn)
        {
            error = "Path Trace frame layer upload was refused by the OpenGL driver";
            break;
        }
        gl.ActiveTexture (GL_TEXTURE2);
        glBindTexture (GL_TEXTURE_2D, static_cast<unsigned> (layer.frame->colorTextureViewHandle()));
        const auto descriptor = layer.frame->colorTextureDescriptor();
        gl.Uniform1i (program.frameBottomFirst, descriptor.rowOrder == NativeTextureRowOrder::BottomFirst ? 1 : 0);
        gl.Uniform1i (program.frameLinear, materialFrameIsLinear (descriptor) ? 1 : 0);
        glDrawArrays (GL_TRIANGLES, 0, 3);
        drawn = glGetError() == GL_NO_ERROR;
        if (! drawn)
            error = "Path Trace frame layer draw was refused by the OpenGL driver";
    }
    glDeleteTextures (2, textures);
    gl.DeleteFramebuffers (1, &framebuffer);
    return drawn;
}
} // namespace

std::shared_ptr<const NativeFixtureSceneFrame> uploadCpuSceneFrame (
    const std::string& backend, const CpuSceneImage& image, std::string& error)
{
    if (backend != "opengl")
    {
        error = "Path Trace upload requires the OpenGL fixture backend";
        return nullptr;
    }
    const std::size_t pixels = static_cast<std::size_t> (image.width) * image.height;
    if (image.width == 0 || image.height == 0
        || image.width > static_cast<std::uint32_t> (std::numeric_limits<int>::max())
        || image.height > static_cast<std::uint32_t> (std::numeric_limits<int>::max())
        || image.color == nullptr || image.depth == nullptr
        || image.color->size() != pixels * (image.linearColor ? 8u : 4u) || image.depth->size() != pixels)
    {
        error = "Path Trace image does not match its declared size and format";
        return nullptr;
    }
    const auto contextIdentity = reinterpret_cast<std::uintptr_t> (glfwGetCurrentContext());
    for (const auto& layer : image.frameLayers)
        if (layer.frame == nullptr || layer.light == nullptr || layer.uvCoverage == nullptr
            || layer.light->size() != pixels * 8u || layer.uvCoverage->size() != pixels * 3u
            || layer.frame->colorTextureViewHandle() == 0
            || layer.frame->colorTextureDescriptor().deviceOrContextIdentity != contextIdentity)
        {
            error = "Path Trace frame layer does not match its image or this OpenGL context";
            return nullptr;
        }
    auto* context = glfwGetCurrentContext();
    if (context == nullptr)
    {
        error = "Path Trace upload requires a current OpenGL context";
        return nullptr;
    }
    // Entry points can differ between contexts, so the table is kept per
    // thread for the context it was loaded in.
    thread_local GLFWwindow* loadedContext = nullptr;
    thread_local arbitgl::GlFuncs gl;
    thread_local FrameLayerProgram layerProgram;
    if (loadedContext != context)
    {
        std::string missing;
        loadedContext = nullptr;
        layerProgram = {};
        if (! arbitgl::loadGlFunctions (gl, missing))
        {
            error = "Path Trace upload could not load OpenGL functions: " + missing;
            return nullptr;
        }
        loadedContext = context;
    }
    for (int drained = 0; drained < 16 && glGetError() != GL_NO_ERROR; ++drained) {}
    unsigned textures[2] { 0, 0 };
    glGenTextures (2, textures);
    if (textures[0] == 0 || textures[1] == 0)
    {
        glDeleteTextures (2, textures);
        error = "Path Trace upload could not allocate textures";
        return nullptr;
    }
    bool uploaded = false;
    {
        const UploadState state (gl);
        uploaded = uploadTexture (textures[0], image.linearColor ? GL_RGBA16F : GL_RGBA8, GL_RGBA,
                                  image.linearColor ? GL_HALF_FLOAT : GL_UNSIGNED_BYTE,
                                  image.width, image.height, image.color->data())
            && uploadTexture (textures[1], GL_R32F, GL_RED, GL_FLOAT, image.width, image.height,
                              image.depth->data());
    }
    if (! uploaded)
    {
        glDeleteTextures (2, textures);
        error = "Path Trace upload was refused by the OpenGL driver";
        return nullptr;
    }
    if (! image.frameLayers.empty()
        && ((layerProgram.program == 0 && ! buildLayerProgram (gl, layerProgram, error))
            || ! drawFrameLayers (gl, layerProgram, textures[0], image, error)))
    {
        glDeleteTextures (2, textures);
        return nullptr;
    }
    try
    {
        return std::make_shared<const OpenGlCpuSceneFrame> (context, textures[0], textures[1], image.width,
                                                            image.height, image.linearColor);
    }
    catch (const std::bad_alloc&)
    {
        glDeleteTextures (2, textures);
        error = "Path Trace frame allocation failed";
        return nullptr;
    }
}

} // namespace arbitgpu
