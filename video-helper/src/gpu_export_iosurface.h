// gpu_export_iosurface.h — IOSurfaceExporter: allocates IOSurface-backed
// GL_TEXTURE_RECTANGLE render targets and exposes their global IOSurfaceIDs
// for the zero-copy docked viewport (SharedGpuSurfaceProtocol.h, macOS).
//
// IOSurfaces created with kIOSurfaceIsGlobal can be opened in any process of
// the same user session via IOSurfaceLookup(id) — the ID is a plain uint32,
// so unlike dmabuf no fd/port passing is needed; it rides in
// BufferInfo.modifier (fourcc = gpusurf::kHandleIOSurface, planeCount = 0).
// kIOSurfaceIsGlobal is deprecated (the blessed replacement is mach-port
// transfer, which our unix-socket channel cannot carry) but remains the
// standard cross-process texture path and works through macOS 15.
//
// The GL side binds each surface with CGLTexImageIOSurface2D, which only
// accepts GL_TEXTURE_RECTANGLE — buffer textures here are rectangle
// textures (core profile OK), attached to FBOs and written via
// glBlitFramebuffer from the renderer's 2D composite texture.
//
// Sync model: v1 only — createNativeFenceFd() always returns -1 and the
// caller glFinish()es before FRAME_READY, which is exactly what IOSurface
// cross-context coherence wants.
#pragma once

#if ARBIT_HAVE_IOSURFACE

#include "gl_loader.h"

#include <cstdint>
#include <string>

namespace gpuexp
{

struct ExportedBuffer
{
    unsigned tex = 0, fbo = 0;    // tex is a GL_TEXTURE_RECTANGLE
    int width = 0, height = 0;

    uint32_t fourcc = 0;          // gpusurf::kHandleIOSurface
    uint64_t modifier = 0;        // global IOSurfaceID
    int planeCount = 0;           // always 0 (no fds on this transport)
    int fds[4] = { -1, -1, -1, -1 };
    uint32_t strides[4] = {}, offsets[4] = {};

    void* surface = nullptr;      // IOSurfaceRef (owned)

    // Announced via FRAME_READY and not yet FRAME_RELEASEd by the consumer.
    // Managed by the viewport render loop, not by the exporter.
    bool busy = false;
};

class IOSurfaceExporter
{
public:
    IOSurfaceExporter() = default;
    ~IOSurfaceExporter() { shutdown(); }
    IOSurfaceExporter (const IOSurfaceExporter&) = delete;
    IOSurfaceExporter& operator= (const IOSurfaceExporter&) = delete;

    // The GL context must be current (CGL — GLFW's only option on macOS).
    bool initialize (const arbitgl::GlFuncs* gl, std::string& errorOut);
    bool available() const { return ready_; }

    // No multi-GPU demotion check on macOS (CGL virtual screens migrate
    // IOSurface textures transparently).
    const std::string& devicePath() const { return devicePath_; }

    // Allocates one w*h BGRA IOSurface, binds it as a rectangle texture and
    // attaches an FBO. GL context current.
    bool allocate (ExportedBuffer& b, int width, int height, std::string& errorOut);
    void destroy (ExportedBuffer& b); // GL context current

    // Copies srcTexture (GL_TEXTURE_2D, same size as b) into b's FBO via
    // glBlitFramebuffer. Caller glFinish()es before announcing FRAME_READY.
    bool blit (unsigned srcTexture, const ExportedBuffer& b);

    bool fenceSyncAvailable() const { return false; } // v1 sync only
    int createNativeFenceFd() { return -1; }

    void shutdown(); // releases the internal read FBO only (buffers are the caller's)

private:
    const arbitgl::GlFuncs* gl_ = nullptr;
    void* cglContext_ = nullptr;  // CGLContextObj
    bool ready_ = false;
    unsigned readFbo_ = 0;
    std::string devicePath_;      // always empty
};

} // namespace gpuexp

#endif // ARBIT_HAVE_IOSURFACE
