#include "gpu_export.h"

#if ARBIT_HAVE_DMABUF

#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <GLFW/glfw3.h>

#include <cstring>
#include <unistd.h>

namespace gpuexp
{

namespace
{
bool hasExtension (const char* extensions, const char* name)
{
    if (extensions == nullptr || name == nullptr)
        return false;
    const size_t nameLen = std::strlen (name);
    const char* p = extensions;
    while ((p = std::strstr (p, name)) != nullptr)
    {
        const bool startOk = (p == extensions || p[-1] == ' ');
        const bool endOk = (p[nameLen] == '\0' || p[nameLen] == ' ');
        if (startOk && endOk)
            return true;
        p += nameLen;
    }
    return false;
}
} // namespace

bool DmabufExporter::initialize (const arbitgl::GlFuncs* gl, std::string& errorOut)
{
    gl_ = gl;
    ready_ = false;

    EGLDisplay dpy = eglGetCurrentDisplay();
    EGLContext ctx = eglGetCurrentContext();
    if (dpy == EGL_NO_DISPLAY || ctx == EGL_NO_CONTEXT)
    {
        errorOut = "GL context is not EGL-backed (GLX session) — dmabuf export impossible";
        return false;
    }
    display_ = dpy;

    const char* exts = eglQueryString (dpy, EGL_EXTENSIONS);
    if (! hasExtension (exts, "EGL_MESA_image_dma_buf_export"))
    {
        errorOut = "EGL display lacks EGL_MESA_image_dma_buf_export";
        return false;
    }
    if (! hasExtension (exts, "EGL_KHR_gl_texture_2D_image")
        && ! hasExtension (exts, "EGL_KHR_image_base"))
    {
        errorOut = "EGL display lacks EGL_KHR_gl_texture_2D_image";
        return false;
    }

    createImage_     = (void*) eglGetProcAddress ("eglCreateImageKHR");
    destroyImage_    = (void*) eglGetProcAddress ("eglDestroyImageKHR");
    exportQuery_     = (void*) eglGetProcAddress ("eglExportDMABUFImageQueryMESA");
    exportImage_     = (void*) eglGetProcAddress ("eglExportDMABUFImageMESA");
    blitFramebuffer_ = (void*) glfwGetProcAddress ("glBlitFramebuffer");
    if (createImage_ == nullptr || destroyImage_ == nullptr
        || exportQuery_ == nullptr || exportImage_ == nullptr)
    {
        errorOut = "missing EGL dmabuf export entry points";
        return false;
    }
    if (blitFramebuffer_ == nullptr)
    {
        errorOut = "glBlitFramebuffer unavailable";
        return false;
    }

    // Acquire fences for v2 sync (best-effort; present path falls back to
    // glFinish() when unavailable).
    fenceSyncAvailable_ = false;
    if (hasExtension (exts, "EGL_ANDROID_native_fence_sync"))
    {
        createSync_       = (void*) eglGetProcAddress ("eglCreateSyncKHR");
        destroySync_      = (void*) eglGetProcAddress ("eglDestroySyncKHR");
        dupNativeFenceFd_ = (void*) eglGetProcAddress ("eglDupNativeFenceFDANDROID");
        fenceSyncAvailable_ = createSync_ != nullptr && destroySync_ != nullptr
                           && dupNativeFenceFd_ != nullptr;
    }

    // DRM device of the exporting GPU (best-effort; consumer compares to
    // demote on multi-GPU mismatch).
    devicePath_.clear();
    auto queryDisplayAttrib = (PFNEGLQUERYDISPLAYATTRIBEXTPROC) eglGetProcAddress ("eglQueryDisplayAttribEXT");
    auto queryDeviceString = (PFNEGLQUERYDEVICESTRINGEXTPROC) eglGetProcAddress ("eglQueryDeviceStringEXT");
    if (queryDisplayAttrib != nullptr && queryDeviceString != nullptr)
    {
        EGLAttrib devAttr = 0;
        if (queryDisplayAttrib (dpy, EGL_DEVICE_EXT, &devAttr) && devAttr != 0)
        {
            const auto dev = (EGLDeviceEXT) devAttr;
            const char* node = queryDeviceString (dev, EGL_DRM_RENDER_NODE_FILE_EXT);
            if (node == nullptr)
                node = queryDeviceString (dev, EGL_DRM_DEVICE_FILE_EXT);
            if (node != nullptr)
                devicePath_ = node;
        }
    }

    if (readFbo_ == 0)
        gl_->GenFramebuffers (1, &readFbo_);

    ready_ = true;
    return true;
}

bool DmabufExporter::allocate (ExportedBuffer& b, int width, int height, std::string& errorOut)
{
    if (! ready_)
    {
        errorOut = "exporter not initialized";
        return false;
    }

    b = ExportedBuffer {};
    b.width = width;
    b.height = height;

    glGenTextures (1, &b.tex);
    glBindTexture (GL_TEXTURE_2D, b.tex);
    glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                  GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    gl_->GenFramebuffers (1, &b.fbo);
    gl_->BindFramebuffer (GL_FRAMEBUFFER, b.fbo);
    gl_->FramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, b.tex, 0);
    if (gl_->CheckFramebufferStatus (GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        errorOut = "export buffer FBO incomplete";
        gl_->BindFramebuffer (GL_FRAMEBUFFER, 0);
        destroy (b);
        return false;
    }

    // Touch the storage through the FBO before creating the EGLImage —
    // required on NVIDIA for the image to alias live storage (verified),
    // and gives defined contents on every driver.
    glViewport (0, 0, width, height);
    glClearColor (0.0f, 0.0f, 0.0f, 1.0f);
    glClear (GL_COLOR_BUFFER_BIT);
    gl_->BindFramebuffer (GL_FRAMEBUFFER, 0);
    glFinish();

    const auto createImage = (PFNEGLCREATEIMAGEKHRPROC) createImage_;
    const EGLint imageAttribs[] = { EGL_GL_TEXTURE_LEVEL_KHR, 0, EGL_NONE };
    b.image = createImage ((EGLDisplay) display_, eglGetCurrentContext(),
                           EGL_GL_TEXTURE_2D_KHR,
                           (EGLClientBuffer) (uintptr_t) b.tex, imageAttribs);
    if (b.image == EGL_NO_IMAGE_KHR)
    {
        errorOut = "eglCreateImage(GL_TEXTURE_2D) failed: 0x"
                 + std::to_string (eglGetError());
        destroy (b);
        return false;
    }

    const auto exportQuery = (PFNEGLEXPORTDMABUFIMAGEQUERYMESAPROC) exportQuery_;
    const auto exportImage = (PFNEGLEXPORTDMABUFIMAGEMESAPROC) exportImage_;

    int fourcc = 0, planeCount = 0;
    EGLuint64KHR modifier = 0;
    if (! exportQuery ((EGLDisplay) display_, (EGLImageKHR) b.image,
                       &fourcc, &planeCount, &modifier)
        || planeCount < 1 || planeCount > 4)
    {
        errorOut = "eglExportDMABUFImageQueryMESA failed";
        destroy (b);
        return false;
    }

    EGLint strides[4] = {}, offsets[4] = {};
    if (! exportImage ((EGLDisplay) display_, (EGLImageKHR) b.image,
                       b.fds, strides, offsets))
    {
        errorOut = "eglExportDMABUFImageMESA failed";
        destroy (b);
        return false;
    }

    b.fourcc = (uint32_t) fourcc;
    b.modifier = (uint64_t) modifier;
    b.planeCount = planeCount;
    for (int i = 0; i < planeCount; ++i)
    {
        if (b.fds[i] < 0)
        {
            errorOut = "export returned invalid fd for plane " + std::to_string (i);
            destroy (b);
            return false;
        }
        b.strides[i] = (uint32_t) strides[i];
        b.offsets[i] = (uint32_t) offsets[i];
    }
    return true;
}

void DmabufExporter::destroy (ExportedBuffer& b)
{
    if (b.image != nullptr && display_ != nullptr && destroyImage_ != nullptr)
        ((PFNEGLDESTROYIMAGEKHRPROC) destroyImage_) ((EGLDisplay) display_, (EGLImageKHR) b.image);
    for (int& fd : b.fds)
        if (fd >= 0)
        {
            ::close (fd);
            fd = -1;
        }
    if (b.fbo != 0 && gl_ != nullptr)
        gl_->DeleteFramebuffers (1, &b.fbo);
    if (b.tex != 0)
        glDeleteTextures (1, &b.tex);
    b = ExportedBuffer {};
}

bool DmabufExporter::blit (unsigned srcTexture, const ExportedBuffer& b)
{
    if (! ready_ || b.fbo == 0)
        return false;

    gl_->BindFramebuffer (GL_READ_FRAMEBUFFER, readFbo_);
    gl_->FramebufferTexture2D (GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, srcTexture, 0);
    gl_->BindFramebuffer (GL_DRAW_FRAMEBUFFER, b.fbo);
    ((PFNGLBLITFRAMEBUFFERPROC) blitFramebuffer_) (
        0, 0, b.width, b.height, 0, 0, b.width, b.height,
        GL_COLOR_BUFFER_BIT, GL_NEAREST);
    gl_->BindFramebuffer (GL_FRAMEBUFFER, 0);
    return true;
}

int DmabufExporter::createNativeFenceFd()
{
    if (! fenceSyncAvailable_)
        return -1;

    const auto createSync = (PFNEGLCREATESYNCKHRPROC) createSync_;
    const auto destroySync = (PFNEGLDESTROYSYNCKHRPROC) destroySync_;
    const auto dupFenceFd = (PFNEGLDUPNATIVEFENCEFDANDROIDPROC) dupNativeFenceFd_;

    EGLSyncKHR sync = createSync ((EGLDisplay) display_,
                                  EGL_SYNC_NATIVE_FENCE_ANDROID, nullptr);
    if (sync == EGL_NO_SYNC_KHR)
        return -1;

    // The native fence fd only materializes once the fence command reaches
    // the GPU command stream — flush is required by the extension spec.
    glFlush();

    const int fd = dupFenceFd ((EGLDisplay) display_, sync);
    destroySync ((EGLDisplay) display_, sync);
    return fd >= 0 ? fd : -1;
}

void DmabufExporter::shutdown()
{
    if (readFbo_ != 0 && gl_ != nullptr)
    {
        gl_->DeleteFramebuffers (1, &readFbo_);
        readFbo_ = 0;
    }
    ready_ = false;
}

} // namespace gpuexp

#endif // ARBIT_HAVE_DMABUF
