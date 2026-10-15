#include "optical_flow_native_source_upload.h"
#include "color_transform_gl.h"
#include "gl_loader.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <future>
#include <limits>
#include <mutex>
#include <vector>

namespace videoopticalflow
{
namespace
{
struct RetiredTexture
{
    GLFWwindow* context = nullptr;
    unsigned texture = 0;
};

std::mutex& retiredMutex()
{
    static auto* mutex = new std::mutex;
    return *mutex;
}

std::vector<RetiredTexture>& retiredTextures()
{
    static auto* textures = new std::vector<RetiredTexture>;
    return *textures;
}

void drainRetired (GLFWwindow* context)
{
    std::lock_guard<std::mutex> lock (retiredMutex());
    auto& textures = retiredTextures();
    for (auto item = textures.begin(); item != textures.end();)
        if (item->context == context)
        {
            glDeleteTextures (1, &item->texture);
            item = textures.erase (item);
        }
        else ++item;
}

class OpenGlSourceResource final : public UploadedSourceResource
{
public:
    OpenGlSourceResource (GLFWwindow* context, unsigned texture)
        : context_ (context), texture_ (texture) {}

    ~OpenGlSourceResource() override
    {
        const auto texture = texture_.exchange (0, std::memory_order_acq_rel);
        if (texture == 0) return;
        if (glfwGetCurrentContext() == context_)
            glDeleteTextures (1, &texture);
        else
        {
            std::lock_guard<std::mutex> lock (retiredMutex());
            retiredTextures().push_back ({ context_, texture });
        }
    }

    std::uintptr_t imageHandle() const noexcept override
    { return texture_.load (std::memory_order_acquire); }
    std::uintptr_t nativeContextIdentity() const noexcept override
    { return reinterpret_cast<std::uintptr_t> (context_); }

    void invalidateOnOwner() noexcept
    {
        const auto texture = texture_.exchange (0, std::memory_order_acq_rel);
        if (texture != 0) glDeleteTextures (1, &texture);
    }

private:
    GLFWwindow* context_ = nullptr;
    std::atomic<unsigned> texture_ { 0 };
};

std::vector<std::weak_ptr<OpenGlSourceResource>>& activeUploads()
{
    static auto* uploads = new std::vector<std::weak_ptr<OpenGlSourceResource>>;
    return *uploads;
}

class OpenGlSourceUploader;

std::mutex& uploaderMutex()
{
    static auto* mutex = new std::mutex;
    return *mutex;
}

std::vector<OpenGlSourceUploader*>& activeUploaders()
{
    static auto* uploaders = new std::vector<OpenGlSourceUploader*>;
    return *uploaders;
}

class OpenGlSourceUploader final : public NativeSourceUploader
{
public:
    OpenGlSourceUploader (GLFWwindow* context, std::uint64_t helperGeneration,
                          std::uint64_t structuralRevision)
        : context_ (context), helperGeneration_ (helperGeneration),
          structuralRevision_ (structuralRevision)
    {
        std::lock_guard<std::mutex> lock (uploaderMutex());
        activeUploaders().push_back (this);
    }

    ~OpenGlSourceUploader() override
    {
        std::lock_guard<std::mutex> lock (uploaderMutex());
        cancelPreviewOnOwner();
        auto& uploaders = activeUploaders();
        uploaders.erase (std::remove (uploaders.begin(), uploaders.end(), this),
                         uploaders.end());
    }

    std::shared_ptr<const NativeSourceFrame> uploadPreview (
        const DecodedSourceReceipt& decoded,
        const colortransform::AdmittedTransform& transform,
        const SourceFrameV2& expected, bool pixelsValidatedOnWorker,
        std::string& error) override
    {
        return uploadImpl (decoded, transform, expected, pixelsValidatedOnWorker,
                           true, error);
    }

    std::shared_ptr<const NativeSourceFrame> upload (
        const DecodedSourceReceipt& decoded,
        const colortransform::AdmittedTransform& transform,
        const SourceFrameV2& expected, std::string& error) override
    {
        return uploadImpl (decoded, transform, expected, false, false, error);
    }

    void cancelPreviewOnOwner() noexcept
    {
        if (glfwGetCurrentContext() == context_)
        {
            arbitgl::GlFuncs gl;
            std::string missing;
            if (arbitgl::loadGlFunctions (gl, missing))
                for (auto& pending : previewUploads_)
                    releaseReadback (gl, *pending);
        }
        // Futures join here. Their workers only own CPU bytes, so no GL call
        // or context access can survive the context's invalidation.
        previewUploads_.clear();
    }

    std::uintptr_t contextIdentity() const noexcept
    { return reinterpret_cast<std::uintptr_t> (context_); }

private:
    struct PreviewUpload
    {
        SourceFrameV2 expected;
        std::string pixelSha256;
        colortransform::CacheIdentity colorIdentity;
        std::shared_ptr<OpenGlSourceResource> resource;
        unsigned pbo = 0;
        GLsync fence = nullptr;
        std::future<ContentIdentity> digest;
        ContentIdentity content {};
        std::string error;
        bool ready = false;
        std::uint64_t lastUse = 0;
    };

    static void releaseReadback (const arbitgl::GlFuncs& gl,
                                 PreviewUpload& pending) noexcept
    {
        if (pending.fence != nullptr)
        {
            gl.DeleteSync (pending.fence);
            pending.fence = nullptr;
        }
        if (pending.pbo != 0)
        {
            gl.DeleteBuffers (1, &pending.pbo);
            pending.pbo = 0;
        }
    }

    void pollPreview (const arbitgl::GlFuncs& gl)
    {
        int priorPackBuffer = 0;
        glGetIntegerv (GL_PIXEL_PACK_BUFFER_BINDING, &priorPackBuffer);
        for (auto& pending : previewUploads_)
        {
            if (pending->fence != nullptr)
            {
                const auto status = gl.ClientWaitSync (pending->fence, 0, 0);
                if (status == GL_ALREADY_SIGNALED || status == GL_CONDITION_SATISFIED)
                {
                    const auto bytes = static_cast<std::size_t> (
                        pending->expected.frame.extent.width)
                        * pending->expected.frame.extent.height * 8u;
                    std::vector<std::uint8_t> pixels (bytes);
                    gl.BindBuffer (GL_PIXEL_PACK_BUFFER, pending->pbo);
                    const void* mapped = gl.MapBufferRange (
                        GL_PIXEL_PACK_BUFFER, 0, static_cast<GLsizeiptr> (bytes),
                        GL_MAP_READ_BIT);
                    if (mapped != nullptr)
                    {
                        std::memcpy (pixels.data(), mapped, bytes);
                        if (gl.UnmapBuffer (GL_PIXEL_PACK_BUFFER) != GL_TRUE)
                            pending->error = "Optical Flow OpenGL linear source mapping failed";
                    }
                    else pending->error = "Optical Flow OpenGL linear source mapping failed";
                    releaseReadback (gl, *pending);
                    if (pending->error.empty() && glGetError() == GL_NO_ERROR)
                    {
                        try
                        {
                            pending->digest = std::async (std::launch::async,
                                [pixels = std::move (pixels)]
                                {
                                    ContentIdentity content {};
                                    detail::decodeSha256 (videohelper::sha256Bytes (
                                        pixels.data(), pixels.size()), content);
                                    return content;
                                });
                        }
                        catch (...)
                        { pending->error = "Optical Flow OpenGL linear source digest worker failed"; }
                    }
                    else if (pending->error.empty())
                        pending->error = "Optical Flow OpenGL linear source mapping failed";
                }
                else if (status == GL_WAIT_FAILED)
                {
                    pending->error = "Optical Flow OpenGL linear source fence failed";
                    releaseReadback (gl, *pending);
                }
            }
            if (pending->digest.valid()
                && pending->digest.wait_for (std::chrono::seconds (0))
                       == std::future_status::ready)
            {
                try { pending->content = pending->digest.get(); }
                catch (...) { pending->error = "Optical Flow OpenGL linear source digest failed"; }
                if (! detail::anyNonZero (pending->content) && pending->error.empty())
                    pending->error = "Optical Flow OpenGL linear source digest failed";
                pending->ready = true;
            }
            else if (! pending->error.empty()) pending->ready = true;
        }
        gl.BindBuffer (GL_PIXEL_PACK_BUFFER, static_cast<unsigned> (priorPackBuffer));
    }

    std::shared_ptr<const NativeSourceFrame> uploadImpl (
        const DecodedSourceReceipt& decoded,
        const colortransform::AdmittedTransform& transform,
        const SourceFrameV2& expected, bool pixelsValidatedOnWorker,
        bool preview, std::string& error)
    {
        error.clear();
        if (context_ == nullptr || glfwGetCurrentContext() != context_)
        {
            error = "Optical Flow OpenGL source owner has no current context";
            return {};
        }
        if (! validateUploadSource (decoded, transform, expected, error,
                                    preview && pixelsValidatedOnWorker)) return {};
        const auto backend = arbitgpu::nativeOpticalFlowExecutionBackend()
                                 .opticalFlowCapabilities();
        if (backend.kind != BackendKind::NativeGpu || ! backend.supportsOpticalFlow)
        {
            error = "Optical Flow OpenGL source backend is unavailable";
            return {};
        }
        drainRetired (context_);
        arbitgl::GlFuncs gl;
        if (! arbitgl::loadGlFunctions (gl, error)) return {};
        if (preview)
        {
            pollPreview (gl);
            for (auto& pending : previewUploads_)
                if (pending->expected.clipId == expected.clipId
                    && pending->expected.sourceContent == expected.sourceContent
                    && pending->expected.sourceGeneration == expected.sourceGeneration
                    && pending->expected.frame.identity.stream == expected.frame.identity.stream
                    && pending->expected.frame.identity.frameIndex == expected.frame.identity.frameIndex
                    && pending->expected.frame.timestamp.ticks == expected.frame.timestamp.ticks
                    && pending->expected.frame.timestamp.timescale == expected.frame.timestamp.timescale
                    && pending->expected.frame.extent == expected.frame.extent
                    && pending->expected.frame.format == expected.frame.format
                    && pending->expected.frame.colorSpace == expected.frame.colorSpace
                    && pending->pixelSha256 == decoded.pixelSha256
                    && pending->colorIdentity == transform.cacheIdentity())
                {
                    pending->lastUse = ++useSerial_;
                    if (! pending->ready) { error = kPreviewUploadPending; return {}; }
                    if (! pending->error.empty()) { error = pending->error; return {}; }
                    return std::make_shared<const UploadedSourceFrame> (
                        expected, pending->content, backend, helperGeneration_,
                        structuralRevision_, pending->resource);
                }
            if (previewUploads_.size() >= 4)
            {
                auto done = std::min_element (previewUploads_.begin(), previewUploads_.end(),
                    [] (const auto& a, const auto& b)
                    {
                        constexpr auto unavailable = std::numeric_limits<std::uint64_t>::max();
                        return (a->ready ? a->lastUse : unavailable)
                             < (b->ready ? b->lastUse : unavailable);
                    });
                if (done == previewUploads_.end() || ! (*done)->ready)
                { error = kPreviewUploadPending; return {}; }
                previewUploads_.erase (done);
            }
        }
        const int width = decoded.frame.width, height = decoded.frame.height;
        int priorTexture = 0, priorReadFramebuffer = 0;
        int priorPack = 0, priorUnpack = 0, priorPackBuffer = 0;
        int priorRowLength = 0, priorSkipRows = 0, priorSkipPixels = 0, priorSwap = 0;
        glGetIntegerv (GL_TEXTURE_BINDING_2D, &priorTexture);
        glGetIntegerv (GL_READ_FRAMEBUFFER_BINDING, &priorReadFramebuffer);
        glGetIntegerv (GL_PACK_ALIGNMENT, &priorPack);
        glGetIntegerv (GL_UNPACK_ALIGNMENT, &priorUnpack);
        glGetIntegerv (GL_PIXEL_PACK_BUFFER_BINDING, &priorPackBuffer);
        glGetIntegerv (GL_PACK_ROW_LENGTH, &priorRowLength);
        glGetIntegerv (GL_PACK_SKIP_ROWS, &priorSkipRows);
        glGetIntegerv (GL_PACK_SKIP_PIXELS, &priorSkipPixels);
        glGetIntegerv (GL_PACK_SWAP_BYTES, &priorSwap);
        unsigned source = 0, output = 0, readFramebuffer = 0;
        const auto restore = [&]
        {
            gl.BindFramebuffer (GL_READ_FRAMEBUFFER,
                                static_cast<unsigned> (priorReadFramebuffer));
            glBindTexture (GL_TEXTURE_2D, static_cast<unsigned> (priorTexture));
            gl.BindBuffer (GL_PIXEL_PACK_BUFFER, static_cast<unsigned> (priorPackBuffer));
            glPixelStorei (GL_PACK_ALIGNMENT, priorPack);
            glPixelStorei (GL_UNPACK_ALIGNMENT, priorUnpack);
            glPixelStorei (GL_PACK_ROW_LENGTH, priorRowLength);
            glPixelStorei (GL_PACK_SKIP_ROWS, priorSkipRows);
            glPixelStorei (GL_PACK_SKIP_PIXELS, priorSkipPixels);
            glPixelStorei (GL_PACK_SWAP_BYTES, priorSwap);
            if (readFramebuffer != 0) gl.DeleteFramebuffers (1, &readFramebuffer);
            if (source != 0) glDeleteTextures (1, &source);
        };
        glGenTextures (1, &source);
        glBindTexture (GL_TEXTURE_2D, source);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glPixelStorei (GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA,
                      GL_UNSIGNED_BYTE, decoded.frame.rgba.data());
        glGenTextures (1, &output);
        glBindTexture (GL_TEXTURE_2D, output);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0,
                      GL_RGBA, GL_HALF_FLOAT, nullptr);
        videorender::ColorTransformGl color;
        if (source == 0 || output == 0 || glGetError() != GL_NO_ERROR
            || ! color.initialize (&gl, error)
            || ! color.render (source, output, width, height, transform, error))
        {
            if (error.empty()) error = "Optical Flow OpenGL source conversion failed";
            color.shutdown();
            if (output != 0) glDeleteTextures (1, &output);
            restore();
            return {};
        }
        color.shutdown();
        gl.GenFramebuffers (1, &readFramebuffer);
        gl.BindFramebuffer (GL_READ_FRAMEBUFFER, readFramebuffer);
        gl.FramebufferTexture2D (GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                 GL_TEXTURE_2D, output, 0);
        const auto byteCount = static_cast<std::size_t> (width)
                             * static_cast<std::size_t> (height) * 8u;
        std::vector<std::uint8_t> pixels;
        unsigned pbo = 0;
        GLsync fence = nullptr;
        glPixelStorei (GL_PACK_ALIGNMENT, 1);
        glPixelStorei (GL_PACK_ROW_LENGTH, 0);
        glPixelStorei (GL_PACK_SKIP_ROWS, 0);
        glPixelStorei (GL_PACK_SKIP_PIXELS, 0);
        glPixelStorei (GL_PACK_SWAP_BYTES, GL_FALSE);
        if (readFramebuffer == 0
            || gl.CheckFramebufferStatus (GL_READ_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            error = "Optical Flow OpenGL linear source readback is incomplete";
        else
        {
            glReadBuffer (GL_COLOR_ATTACHMENT0);
            if (preview)
            {
                gl.GenBuffers (1, &pbo);
                gl.BindBuffer (GL_PIXEL_PACK_BUFFER, pbo);
                gl.BufferData (GL_PIXEL_PACK_BUFFER,
                               static_cast<GLsizeiptr> (byteCount), nullptr, GL_STREAM_READ);
                if (pbo != 0 && glGetError() == GL_NO_ERROR)
                {
                    glReadPixels (0, 0, width, height, GL_RGBA, GL_HALF_FLOAT, nullptr);
                    if (glGetError() == GL_NO_ERROR)
                        fence = gl.FenceSync (GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
                }
                if (fence == nullptr)
                    error = "Optical Flow OpenGL linear source readback failed";
                else glFlush();
            }
            else
            {
                pixels.resize (byteCount);
                gl.BindBuffer (GL_PIXEL_PACK_BUFFER, 0);
                glReadPixels (0, 0, width, height, GL_RGBA, GL_HALF_FLOAT, pixels.data());
            }
            if (glGetError() != GL_NO_ERROR)
                error = "Optical Flow OpenGL linear source readback failed";
        }
        restore();
        if (! error.empty())
        {
            if (fence != nullptr) gl.DeleteSync (fence);
            if (pbo != 0) gl.DeleteBuffers (1, &pbo);
            glDeleteTextures (1, &output);
            return {};
        }
        auto resource = std::make_shared<OpenGlSourceResource> (context_, output);
        {
            std::lock_guard<std::mutex> lock (retiredMutex());
            auto& uploads = activeUploads();
            uploads.erase (std::remove_if (uploads.begin(), uploads.end(),
                                           [] (const auto& item) { return item.expired(); }),
                           uploads.end());
            uploads.push_back (resource);
        }
        if (preview)
        {
            auto pending = std::make_unique<PreviewUpload>();
            pending->expected = expected;
            pending->pixelSha256 = decoded.pixelSha256;
            pending->colorIdentity = transform.cacheIdentity();
            pending->resource = resource;
            pending->pbo = pbo;
            pending->fence = fence;
            pending->lastUse = ++useSerial_;
            previewUploads_.push_back (std::move (pending));
            error = kPreviewUploadPending;
            return {};
        }
        ContentIdentity content {};
        if (! detail::decodeSha256 (
                videohelper::sha256Bytes (pixels.data(), pixels.size()), content))
        {
            error = "Optical Flow OpenGL linear source digest failed";
            return {};
        }
        return std::make_shared<const UploadedSourceFrame> (
            expected, content, backend, helperGeneration_, structuralRevision_,
            std::move (resource));
    }

private:
    GLFWwindow* context_ = nullptr;
    std::uint64_t helperGeneration_ = 0;
    std::uint64_t structuralRevision_ = 0;
    std::uint64_t useSerial_ = 0;
    std::vector<std::unique_ptr<PreviewUpload>> previewUploads_;
};
} // namespace

std::unique_ptr<NativeSourceUploader> makeNativeSourceUploader (
    std::uint64_t helperGeneration, std::uint64_t structuralRevision)
{
    auto* context = glfwGetCurrentContext();
    if (context == nullptr || helperGeneration == 0 || structuralRevision == 0)
        return {};
    return std::make_unique<OpenGlSourceUploader> (
        context, helperGeneration, structuralRevision);
}

void invalidateNativeSourceUploads (std::uintptr_t contextIdentity) noexcept
{
    auto* context = reinterpret_cast<GLFWwindow*> (contextIdentity);
    if (context == nullptr || glfwGetCurrentContext() != context) return;
    {
        std::lock_guard<std::mutex> lock (uploaderMutex());
        for (auto* uploader : activeUploaders())
            if (uploader->contextIdentity() == contextIdentity)
                uploader->cancelPreviewOnOwner();
    }
    std::lock_guard<std::mutex> lock (retiredMutex());
    auto& uploads = activeUploads();
    for (auto item = uploads.begin(); item != uploads.end();)
    {
        auto resource = item->lock();
        if (! resource) { item = uploads.erase (item); continue; }
        if (resource->nativeContextIdentity() == contextIdentity)
        {
            resource->invalidateOnOwner();
            item = uploads.erase (item);
        }
        else ++item;
    }
    auto& retired = retiredTextures();
    for (auto item = retired.begin(); item != retired.end();)
        if (item->context == context)
        {
            glDeleteTextures (1, &item->texture);
            item = retired.erase (item);
        }
        else ++item;
}
} // namespace videoopticalflow
