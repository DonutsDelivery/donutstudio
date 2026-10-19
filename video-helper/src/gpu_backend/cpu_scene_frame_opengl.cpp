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
    if (loadedContext != context)
    {
        std::string missing;
        loadedContext = nullptr;
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
