// gpu_export_d3d.h — D3DInteropExporter: allocates D3D11 keyed-mutex shared
// textures, bridges them into the helper's WGL context via
// WGL_NV_DX_interop2 (implemented by NVIDIA, AMD and Intel drivers), and
// exposes their legacy DXGI shared HANDLEs for the zero-copy docked viewport
// (SharedGpuSurfaceProtocol.h, Windows).
//
// Legacy shared handles (IDXGIResource::GetSharedHandle, i.e. textures
// created with D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX and *without*
// SHARED_NTHANDLE) are global values: the consumer opens them with
// ID3D11Device::OpenSharedResource directly — no DuplicateHandle, so the
// handle value rides in BufferInfo.modifier (fourcc = gpusurf::kHandleD3D11,
// planeCount = 0) over the plain stream socket.
//
// Sync model: the keyed mutex is the cross-process barrier. Producer per
// frame: AcquireSync(0) → wglDXLockObjectsNV → blit → unlock → ReleaseSync(1)
// (the unlock orders GL writes into D3D; the release publishes them).
// Consumer: AcquireSync(1) → lock → sample (held while displayed) → unlock →
// ReleaseSync(0) before FRAME_RELEASE. createNativeFenceFd() returns -1, so
// the caller additionally glFinish()es (v1 model — harmless on top of the
// mutex). blit() recovers from a consumer that died holding nothing: when
// key 0 is unavailable it tries key 1 (an unconsumed frame being reclaimed).
#pragma once

#if ARBIT_HAVE_D3D

#include "gl_loader.h"

#include <cstdint>
#include <string>

namespace gpuexp
{

struct ExportedBuffer
{
    unsigned tex = 0, fbo = 0;    // tex is the GL_TEXTURE_2D interop wrapper
    int width = 0, height = 0;

    uint32_t fourcc = 0;          // gpusurf::kHandleD3D11
    uint64_t modifier = 0;        // legacy DXGI shared HANDLE value
    int planeCount = 0;           // always 0 (no fds on this transport)
    int fds[4] = { -1, -1, -1, -1 };
    uint32_t strides[4] = {}, offsets[4] = {};

    void* texture = nullptr;      // ID3D11Texture2D* (owned, Release()d)
    void* keyedMutex = nullptr;   // IDXGIKeyedMutex* (owned, Release()d)
    void* interopObject = nullptr;// HANDLE from wglDXRegisterObjectNV

    // Announced via FRAME_READY and not yet FRAME_RELEASEd by the consumer.
    // Managed by the viewport render loop, not by the exporter.
    bool busy = false;
};

class D3DInteropExporter
{
public:
    D3DInteropExporter() = default;
    ~D3DInteropExporter() { shutdown(); }
    D3DInteropExporter (const D3DInteropExporter&) = delete;
    D3DInteropExporter& operator= (const D3DInteropExporter&) = delete;

    // The WGL context must be current. Creates a D3D11 device on the default
    // adapter and opens the GL<->DX interop device. Fails cleanly when the
    // driver lacks WGL_NV_DX_interop2 — caller demotes down the ladder.
    bool initialize (const arbitgl::GlFuncs* gl, std::string& errorOut);
    bool available() const { return ready_; }

    // No producer/consumer device identity check on Windows: both sides use
    // the default adapter, and a mismatch surfaces as OpenSharedResource /
    // import failure on the consumer, which demotes after 3 failures.
    const std::string& devicePath() const { return devicePath_; }

    // Allocates one w*h RGBA8 shared keyed-mutex texture, registers it with
    // the interop device and attaches an FBO to the GL wrapper.
    bool allocate (ExportedBuffer& b, int width, int height, std::string& errorOut);
    void destroy (ExportedBuffer& b);

    // AcquireSync + lock + glBlitFramebuffer + unlock + ReleaseSync(1).
    // Returns false when the keyed mutex is unavailable (frame dropped).
    bool blit (unsigned srcTexture, const ExportedBuffer& b);

    bool fenceSyncAvailable() const { return false; } // keyed mutex syncs
    int createNativeFenceFd() { return -1; }

    void shutdown(); // closes the interop + D3D devices (buffers are the caller's)

private:
    const arbitgl::GlFuncs* gl_ = nullptr;
    bool ready_ = false;
    unsigned readFbo_ = 0;
    std::string devicePath_;      // always empty

    void* d3dDevice_ = nullptr;       // ID3D11Device*
    void* d3dContext_ = nullptr;      // ID3D11DeviceContext*
    void* interopDevice_ = nullptr;   // HANDLE from wglDXOpenDeviceNV

    // WGL_NV_DX_interop2 entry points (typed in gpu_export_d3d.cpp).
    void* dxOpenDevice_ = nullptr;
    void* dxCloseDevice_ = nullptr;
    void* dxRegisterObject_ = nullptr;
    void* dxUnregisterObject_ = nullptr;
    void* dxLockObjects_ = nullptr;
    void* dxUnlockObjects_ = nullptr;
    void* dxSetResourceShareHandle_ = nullptr;
};

} // namespace gpuexp

#endif // ARBIT_HAVE_D3D
