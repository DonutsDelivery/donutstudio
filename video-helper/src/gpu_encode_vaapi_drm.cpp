#include "gpu_encode_vaapi_drm.h"

extern "C" {
#include <libavutil/buffer.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
}

#if ARBIT_HAVE_DMABUF
#include "gl_loader.h"
#include "vaapi_drm_import.h"

#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <vector>
#endif

namespace videohelper
{

#if ARBIT_HAVE_DMABUF
namespace
{
bool hasExtension (const char* extensions, const char* name)
{
    if (extensions == nullptr)
        return false;
    const std::size_t length = std::strlen (name);
    for (const char* p = extensions; (p = std::strstr (p, name)) != nullptr; p += length)
        if ((p == extensions || p[-1] == ' ') && (p[length] == '\0' || p[length] == ' '))
            return true;
    return false;
}

std::string eglErrorText()
{
    char text[32] = {};
    std::snprintf (text, sizeof (text), "EGL error 0x%x", static_cast<unsigned> (eglGetError()));
    return text;
}

using ImageTargetTexture2D = void (*) (GLenum, void*);
static_assert (sizeof (EGLint) == sizeof (std::int32_t), "EGL attributes are 32-bit");
} // namespace
#endif

VaapiDrmFrameBridge::~VaapiDrmFrameBridge()
{
    close();
}

bool VaapiDrmFrameBridge::open (AVBufferRef* hwFrames, int bits, std::string& error)
{
    close();
#if ARBIT_HAVE_DMABUF
    if (hwFrames == nullptr)
    {
        error = "the VA-API encoder has no frame pool";
        return false;
    }
    const EGLDisplay display = eglGetCurrentDisplay();
    if (display == EGL_NO_DISPLAY || eglGetCurrentContext() == EGL_NO_CONTEXT)
    {
        error = "the export's GL context is not EGL-backed, so VA-API surfaces cannot import";
        return false;
    }
    const char* extensions = eglQueryString (display, EGL_EXTENSIONS);
    if (! hasExtension (extensions, "EGL_EXT_image_dma_buf_import"))
    {
        error = "EGL lacks EGL_EXT_image_dma_buf_import";
        return false;
    }
    createImage_ = reinterpret_cast<void*> (eglGetProcAddress ("eglCreateImageKHR"));
    destroyImage_ = reinterpret_cast<void*> (eglGetProcAddress ("eglDestroyImageKHR"));
    imageTargetTexture_ = reinterpret_cast<void*> (eglGetProcAddress ("glEGLImageTargetTexture2DOES"));
    if (createImage_ == nullptr || destroyImage_ == nullptr || imageTargetTexture_ == nullptr)
    {
        error = "EGL lacks eglCreateImageKHR or glEGLImageTargetTexture2DOES";
        createImage_ = destroyImage_ = imageTargetTexture_ = nullptr;
        return false;
    }
    modifiers_ = hasExtension (extensions, "EGL_EXT_image_dma_buf_import_modifiers");
    display_ = display;
    bits_ = bits;
    hwFrames_ = av_buffer_ref (hwFrames);
    if (hwFrames_ == nullptr)
    {
        error = "the VA-API frame pool did not take a reference";
        return false;
    }
    return true;
#else
    (void) hwFrames;
    (void) bits;
    error = "this build has no EGL, so VA-API surfaces cannot import into GL";
    return false;
#endif
}

AVFrame* VaapiDrmFrameBridge::acquire (unsigned& lumaTexture, unsigned& chromaTexture, std::string& error)
{
    lumaTexture = chromaTexture = 0;
    if (hwFrames_ == nullptr)
    {
        error = "the VA-API bridge is not open";
        return nullptr;
    }
    AVFrame* frame = av_frame_alloc();
    if (frame == nullptr || av_hwframe_get_buffer (hwFrames_, frame, 0) < 0)
    {
        av_frame_free (&frame);
        error = "the VA-API frame pool is out of surfaces";
        return nullptr;
    }
    // The pool hands its surfaces out again and again: each imports once.
    const auto key = reinterpret_cast<std::uintptr_t> (frame->data[3]);
    auto found = surfaces_.find (key);
    if (found == surfaces_.end())
    {
        Surface surface;
        if (! import (frame, surface, error))
        {
            release (surface);
            av_frame_free (&frame);
            return nullptr;
        }
        found = surfaces_.emplace (key, surface).first;
    }
    lumaTexture = found->second.textures[0];
    chromaTexture = found->second.textures[1];
    return frame;
}

bool VaapiDrmFrameBridge::import (AVFrame* frame, Surface& surface, std::string& error)
{
#if ARBIT_HAVE_DMABUF
    surface.mapping = av_frame_alloc();
    if (surface.mapping == nullptr)
    {
        error = "the DRM-PRIME mapping did not allocate";
        return false;
    }
    surface.mapping->format = AV_PIX_FMT_DRM_PRIME;
    if (const int code = av_hwframe_map (surface.mapping, frame, AV_HWFRAME_MAP_WRITE | AV_HWFRAME_MAP_OVERWRITE);
        code < 0)
    {
        char text[AV_ERROR_MAX_STRING_SIZE] = {};
        av_strerror (code, text, sizeof (text));
        error = std::string ("the VA-API surface did not map to DRM-PRIME: ") + text;
        return false;
    }
    const auto* descriptor = reinterpret_cast<const AVDRMFrameDescriptor*> (surface.mapping->data[0]);
    std::array<vaapidrm::PlaneImport, 2> planes;
    if (descriptor == nullptr || ! vaapidrm::planeImports (*descriptor, frame->width, frame->height, bits_, planes, error))
    {
        if (descriptor == nullptr)
            error = "the DRM-PRIME mapping has no descriptor";
        return false;
    }
    const auto createImage = reinterpret_cast<PFNEGLCREATEIMAGEKHRPROC> (createImage_);
    const auto bindImage = reinterpret_cast<ImageTargetTexture2D> (imageTargetTexture_);
    for (int i = 0; i < 2; ++i)
    {
        std::vector<std::int32_t> attributes;
        if (! vaapidrm::imageAttributes (planes[static_cast<std::size_t> (i)], modifiers_, attributes, error))
            return false;
        EGLImageKHR image = createImage (static_cast<EGLDisplay> (display_), EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT,
                                         nullptr, reinterpret_cast<const EGLint*> (attributes.data()));
        if (image == EGL_NO_IMAGE_KHR)
        {
            error = "EGL did not import the VA-API " + std::string (i == 0 ? "luma" : "chroma") + " plane ("
                + vaapidrm::fourcc::name (planes[static_cast<std::size_t> (i)].fourcc) + "), " + eglErrorText();
            return false;
        }
        surface.images[i] = image;
        glGenTextures (1, &surface.textures[i]);
        glBindTexture (GL_TEXTURE_2D, surface.textures[i]);
        bindImage (GL_TEXTURE_2D, image);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glBindTexture (GL_TEXTURE_2D, 0);
        if (const GLenum glError = glGetError(); glError != GL_NO_ERROR)
        {
            error = "GL did not bind the imported VA-API " + std::string (i == 0 ? "luma" : "chroma")
                + " plane, GL error " + std::to_string (glError);
            return false;
        }
    }
    return true;
#else
    (void) frame;
    (void) surface;
    error = "this build has no EGL, so VA-API surfaces cannot import into GL";
    return false;
#endif
}

void VaapiDrmFrameBridge::finish()
{
#if ARBIT_HAVE_DMABUF
    glFinish();
#endif
}

void VaapiDrmFrameBridge::release (Surface& surface)
{
#if ARBIT_HAVE_DMABUF
    for (int i = 0; i < 2; ++i)
    {
        if (surface.textures[i] != 0)
            glDeleteTextures (1, &surface.textures[i]);
        if (surface.images[i] != nullptr && destroyImage_ != nullptr)
            reinterpret_cast<PFNEGLDESTROYIMAGEKHRPROC> (destroyImage_) (static_cast<EGLDisplay> (display_),
                                                                         static_cast<EGLImageKHR> (surface.images[i]));
        surface.textures[i] = 0;
        surface.images[i] = nullptr;
    }
#endif
    av_frame_free (&surface.mapping);
}

void VaapiDrmFrameBridge::close()
{
    for (auto& entry : surfaces_)
        release (entry.second);
    surfaces_.clear();
    av_buffer_unref (&hwFrames_);
    display_ = createImage_ = destroyImage_ = imageTargetTexture_ = nullptr;
    modifiers_ = false;
}

} // namespace videohelper
