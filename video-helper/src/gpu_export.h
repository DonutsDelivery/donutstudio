// gpu_export.h — DmabufExporter: allocates FBO-attached RGBA8 textures and
// exports them once as dmabuf file descriptors (EGL_MESA_image_dma_buf_export)
// for the zero-copy docked viewport (SharedGpuSurfaceProtocol.h).
//
// Requires the GL context current on the calling thread to be EGL-backed
// (GLFW window created with GLFW_CONTEXT_CREATION_API = GLFW_EGL_CONTEXT_API;
// on Wayland GLFW is EGL anyway). GLX contexts cannot export — initialize()
// fails cleanly and the caller demotes to the glfw-window present path.
//
// Allocation discipline (driver quirks, verified on NVIDIA 595 + Mesa):
//   glTexImage2D(RGBA8) -> FBO attach -> clear -> glFinish -> eglCreateImage
//   -> eglExportDMABUFImageMESA.  The texture is never respecified afterwards
// (respecify would orphan the EGLImage); resize = destroy + reallocate.
// Export happens once per buffer; per frame only ownership moves.
//
// All EGL specifics live in gpu_export.cpp so a future surfaceless-EGL or
// GBM backend swap stays local to this pair of files.
#pragma once

#if ARBIT_HAVE_DMABUF

#include "gl_loader.h"

#include <cstdint>
#include <string>

namespace gpuexp
{

struct ExportedBuffer
{
    unsigned tex = 0, fbo = 0;
    int width = 0, height = 0;

    uint32_t fourcc = 0;          // DRM fourcc from eglExportDMABUFImageQueryMESA
    uint64_t modifier = 0;        // DRM format modifier
    int planeCount = 0;
    int fds[4] = { -1, -1, -1, -1 };
    uint32_t strides[4] = {}, offsets[4] = {};

    void* image = nullptr;        // EGLImageKHR keeping the export alive

    // Announced via FRAME_READY and not yet FRAME_RELEASEd by the consumer.
    // Managed by the viewport render loop, not by the exporter.
    bool busy = false;
};

class DmabufExporter
{
public:
    DmabufExporter() = default;
    ~DmabufExporter() { shutdown(); }
    DmabufExporter (const DmabufExporter&) = delete;
    DmabufExporter& operator= (const DmabufExporter&) = delete;

    // Checks EGL context + extensions and resolves entry points. The GL
    // context must be current. Returns false with errorOut set when the
    // dmabuf path is unavailable (GLX context, non-Mesa-extension driver...).
    bool initialize (const arbitgl::GlFuncs* gl, std::string& errorOut);
    bool available() const { return ready_; }

    // DRM render node of the exporting device (multi-GPU demotion check on
    // the consumer side); empty when the driver does not report one.
    const std::string& devicePath() const { return devicePath_; }

    // Allocates one w*h RGBA8 buffer and exports it. GL context current.
    bool allocate (ExportedBuffer& b, int width, int height, std::string& errorOut);
    void destroy (ExportedBuffer& b); // GL context current; closes fds

    // Copies srcTexture (same size as b) into b's FBO via glBlitFramebuffer.
    // Caller orders the GPU work before announcing FRAME_READY: attach an
    // acquire fence from createNativeFenceFd() when fenceSyncAvailable(),
    // otherwise glFinish() (v1 sync model).
    bool blit (unsigned srcTexture, const ExportedBuffer& b);

    // True when EGL_ANDROID_native_fence_sync is usable on this display.
    bool fenceSyncAvailable() const { return fenceSyncAvailable_; }

    // Creates an EGL native fence covering all GL commands issued so far on
    // this context and returns its fd (caller owns; close after sendmsg).
    // Returns -1 on failure — caller must fall back to glFinish().
    int createNativeFenceFd();

    void shutdown(); // releases the internal read FBO only (buffers are the caller's)

private:
    const arbitgl::GlFuncs* gl_ = nullptr;
    void* display_ = nullptr;     // EGLDisplay
    bool ready_ = false;
    unsigned readFbo_ = 0;
    std::string devicePath_;

    bool fenceSyncAvailable_ = false;

    // Resolved EGL/GL extension entry points (typed in gpu_export.cpp).
    void* createImage_ = nullptr;
    void* destroyImage_ = nullptr;
    void* exportQuery_ = nullptr;
    void* exportImage_ = nullptr;
    void* blitFramebuffer_ = nullptr;
    void* createSync_ = nullptr;
    void* destroySync_ = nullptr;
    void* dupNativeFenceFd_ = nullptr;
};

} // namespace gpuexp

#endif // ARBIT_HAVE_DMABUF
