#include "gpu_export_d3d.h"

#if ARBIT_HAVE_D3D

#include "SharedGpuSurfaceProtocol.h"

#include <d3d11.h>
#include <dxgi.h>

#include <GLFW/glfw3.h>

namespace gpuexp
{

namespace
{
// WGL_NV_DX_interop2 — not in the standard wglext headers MinGW ships, so
// declared locally (spec: registry/OpenGL/extensions/NV/WGL_NV_DX_interop2).
using PFNWGLDXOPENDEVICENVPROC = HANDLE (WINAPI*) (void* dxDevice);
using PFNWGLDXCLOSEDEVICENVPROC = BOOL (WINAPI*) (HANDLE hDevice);
using PFNWGLDXREGISTEROBJECTNVPROC = HANDLE (WINAPI*) (HANDLE hDevice, void* dxObject,
                                                       GLuint name, GLenum type, GLenum access);
using PFNWGLDXUNREGISTEROBJECTNVPROC = BOOL (WINAPI*) (HANDLE hDevice, HANDLE hObject);
using PFNWGLDXLOCKOBJECTSNVPROC = BOOL (WINAPI*) (HANDLE hDevice, GLint count, HANDLE* hObjects);
using PFNWGLDXUNLOCKOBJECTSNVPROC = BOOL (WINAPI*) (HANDLE hDevice, GLint count, HANDLE* hObjects);
using PFNWGLDXSETRESOURCESHAREHANDLENVPROC = BOOL (WINAPI*) (void* dxObject, HANDLE shareHandle);

constexpr GLenum WGL_ACCESS_READ_WRITE_NV = 0x0001;

// Interface IIDs spelled out locally — avoids a dxguid/uuid link dependency
// in the MinGW CI build.
constexpr GUID kIID_IDXGIResource =
    { 0x035f3ab4, 0x482e, 0x4e50, { 0xb4, 0x1f, 0x8a, 0x7f, 0x8b, 0xd8, 0x96, 0x0b } };
constexpr GUID kIID_IDXGIKeyedMutex =
    { 0x9d8e1289, 0xd7b3, 0x465f, { 0x81, 0x26, 0x25, 0x0e, 0x34, 0x9a, 0xf8, 0x5d } };

template <typename T>
void safeRelease (T*& p)
{
    if (p != nullptr)
    {
        p->Release();
        p = nullptr;
    }
}
} // namespace

bool D3DInteropExporter::initialize (const arbitgl::GlFuncs* gl, std::string& errorOut)
{
    gl_ = gl;
    ready_ = false;

    dxOpenDevice_             = (void*) glfwGetProcAddress ("wglDXOpenDeviceNV");
    dxCloseDevice_            = (void*) glfwGetProcAddress ("wglDXCloseDeviceNV");
    dxRegisterObject_         = (void*) glfwGetProcAddress ("wglDXRegisterObjectNV");
    dxUnregisterObject_       = (void*) glfwGetProcAddress ("wglDXUnregisterObjectNV");
    dxLockObjects_            = (void*) glfwGetProcAddress ("wglDXLockObjectsNV");
    dxUnlockObjects_          = (void*) glfwGetProcAddress ("wglDXUnlockObjectsNV");
    dxSetResourceShareHandle_ = (void*) glfwGetProcAddress ("wglDXSetResourceShareHandleNV");
    if (dxOpenDevice_ == nullptr || dxCloseDevice_ == nullptr
        || dxRegisterObject_ == nullptr || dxUnregisterObject_ == nullptr
        || dxLockObjects_ == nullptr || dxUnlockObjects_ == nullptr)
    {
        errorOut = "driver lacks WGL_NV_DX_interop2";
        return false;
    }

    ID3D11Device* dev = nullptr;
    ID3D11DeviceContext* ctx = nullptr;
    const D3D_FEATURE_LEVEL levels[] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0 };
    HRESULT hr = D3D11CreateDevice (nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
                                    levels, 2, D3D11_SDK_VERSION, &dev, nullptr, &ctx);
    if (FAILED (hr) || dev == nullptr)
    {
        errorOut = "D3D11CreateDevice failed: 0x" + std::to_string ((unsigned) hr);
        return false;
    }
    d3dDevice_ = dev;
    d3dContext_ = ctx;

    interopDevice_ = ((PFNWGLDXOPENDEVICENVPROC) dxOpenDevice_) (dev);
    if (interopDevice_ == nullptr)
    {
        errorOut = "wglDXOpenDeviceNV failed";
        shutdown();
        return false;
    }

    if (readFbo_ == 0)
        gl_->GenFramebuffers (1, &readFbo_);

    ready_ = true;
    return true;
}

bool D3DInteropExporter::allocate (ExportedBuffer& b, int width, int height,
                                   std::string& errorOut)
{
    if (! ready_)
    {
        errorOut = "exporter not initialized";
        return false;
    }

    b = ExportedBuffer {};
    b.width = width;
    b.height = height;

    D3D11_TEXTURE2D_DESC desc {};
    desc.Width = (UINT) width;
    desc.Height = (UINT) height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM; // GL-friendly channel order
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;

    ID3D11Texture2D* tex2d = nullptr;
    HRESULT hr = ((ID3D11Device*) d3dDevice_)->CreateTexture2D (&desc, nullptr, &tex2d);
    if (FAILED (hr) || tex2d == nullptr)
    {
        errorOut = "CreateTexture2D(shared keyed-mutex) failed: 0x"
                 + std::to_string ((unsigned) hr);
        return false;
    }
    b.texture = tex2d;

    IDXGIResource* res = nullptr;
    HANDLE shared = nullptr;
    if (SUCCEEDED (tex2d->QueryInterface (kIID_IDXGIResource, (void**) &res)) && res != nullptr)
    {
        res->GetSharedHandle (&shared); // legacy global handle — NOT CloseHandle'd
        res->Release();
    }
    if (shared == nullptr)
    {
        errorOut = "GetSharedHandle failed";
        destroy (b);
        return false;
    }
    b.modifier = (uint64_t) (uintptr_t) shared;
    b.fourcc = gpusurf::kHandleD3D11;
    b.planeCount = 0;

    IDXGIKeyedMutex* mutex = nullptr;
    if (FAILED (tex2d->QueryInterface (kIID_IDXGIKeyedMutex, (void**) &mutex)) || mutex == nullptr)
    {
        errorOut = "texture has no keyed mutex";
        destroy (b);
        return false;
    }
    b.keyedMutex = mutex;

    // Optional per spec for D3D11 but required by some drivers before
    // registering a shared resource — best effort.
    if (dxSetResourceShareHandle_ != nullptr)
        ((PFNWGLDXSETRESOURCESHAREHANDLENVPROC) dxSetResourceShareHandle_) (tex2d, shared);

    glGenTextures (1, &b.tex);
    b.interopObject = ((PFNWGLDXREGISTEROBJECTNVPROC) dxRegisterObject_) (
        interopDevice_, tex2d, b.tex, GL_TEXTURE_2D, WGL_ACCESS_READ_WRITE_NV);
    if (b.interopObject == nullptr)
    {
        errorOut = "wglDXRegisterObjectNV failed: 0x"
                 + std::to_string ((unsigned) GetLastError());
        destroy (b);
        return false;
    }

    // First acquire (a fresh keyed mutex starts at key 0): attach the FBO
    // and clear so the consumer never samples garbage. Leave it back on key
    // 0 — the per-frame blit acquires 0 and releases 1.
    auto* km = (IDXGIKeyedMutex*) b.keyedMutex;
    if (km->AcquireSync (0, 1000) == 0)
    {
        HANDLE obj = b.interopObject;
        if (((PFNWGLDXLOCKOBJECTSNVPROC) dxLockObjects_) (interopDevice_, 1, &obj))
        {
            gl_->GenFramebuffers (1, &b.fbo);
            gl_->BindFramebuffer (GL_FRAMEBUFFER, b.fbo);
            gl_->FramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                       GL_TEXTURE_2D, b.tex, 0);
            const bool complete =
                gl_->CheckFramebufferStatus (GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
            if (complete)
            {
                glViewport (0, 0, width, height);
                glClearColor (0.0f, 0.0f, 0.0f, 1.0f);
                glClear (GL_COLOR_BUFFER_BIT);
            }
            gl_->BindFramebuffer (GL_FRAMEBUFFER, 0);
            ((PFNWGLDXUNLOCKOBJECTSNVPROC) dxUnlockObjects_) (interopDevice_, 1, &obj);
            km->ReleaseSync (0);
            if (! complete)
            {
                errorOut = "interop buffer FBO incomplete";
                destroy (b);
                return false;
            }
        }
        else
        {
            km->ReleaseSync (0);
            errorOut = "wglDXLockObjectsNV failed at allocate";
            destroy (b);
            return false;
        }
    }
    else
    {
        errorOut = "initial AcquireSync(0) failed";
        destroy (b);
        return false;
    }

    return true;
}

void D3DInteropExporter::destroy (ExportedBuffer& b)
{
    if (b.interopObject != nullptr && interopDevice_ != nullptr
        && dxUnregisterObject_ != nullptr)
        ((PFNWGLDXUNREGISTEROBJECTNVPROC) dxUnregisterObject_) (interopDevice_, b.interopObject);
    if (b.fbo != 0 && gl_ != nullptr)
        gl_->DeleteFramebuffers (1, &b.fbo);
    if (b.tex != 0)
        glDeleteTextures (1, &b.tex);
    auto* km = (IDXGIKeyedMutex*) b.keyedMutex;
    safeRelease (km);
    auto* tx = (ID3D11Texture2D*) b.texture;
    safeRelease (tx);
    b = ExportedBuffer {};
}

bool D3DInteropExporter::blit (unsigned srcTexture, const ExportedBuffer& b)
{
    if (! ready_ || b.fbo == 0 || b.keyedMutex == nullptr || b.interopObject == nullptr)
        return false;

    // Key 0 = released by the consumer (or never consumed since allocate);
    // key 1 = an announced frame the consumer never acquired (it dropped the
    // connection, or a newer frame superseded it) — reclaim it. WAIT_ABANDONED
    // (consumer died mid-hold) also counts as acquired.
    auto* km = (IDXGIKeyedMutex*) b.keyedMutex;
    HRESULT hr = km->AcquireSync (0, 0);
    if (hr != S_OK && hr != (HRESULT) WAIT_ABANDONED)
        hr = km->AcquireSync (1, 0);
    if (hr != S_OK && hr != (HRESULT) WAIT_ABANDONED)
        return false; // consumer holds it right now — drop this frame

    HANDLE obj = b.interopObject;
    if (! ((PFNWGLDXLOCKOBJECTSNVPROC) dxLockObjects_) (interopDevice_, 1, &obj))
    {
        km->ReleaseSync (0);
        return false;
    }

    gl_->BindFramebuffer (GL_READ_FRAMEBUFFER, readFbo_);
    gl_->FramebufferTexture2D (GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, srcTexture, 0);
    gl_->BindFramebuffer (GL_DRAW_FRAMEBUFFER, b.fbo);
    gl_->BlitFramebuffer (0, 0, b.width, b.height, 0, 0, b.width, b.height,
                          GL_COLOR_BUFFER_BIT, GL_NEAREST);
    gl_->BindFramebuffer (GL_FRAMEBUFFER, 0);

    ((PFNWGLDXUNLOCKOBJECTSNVPROC) dxUnlockObjects_) (interopDevice_, 1, &obj);
    km->ReleaseSync (1); // publish to the consumer (AcquireSync(1) side)
    return true;
}

void D3DInteropExporter::shutdown()
{
    if (readFbo_ != 0 && gl_ != nullptr)
    {
        gl_->DeleteFramebuffers (1, &readFbo_);
        readFbo_ = 0;
    }
    if (interopDevice_ != nullptr && dxCloseDevice_ != nullptr)
    {
        ((PFNWGLDXCLOSEDEVICENVPROC) dxCloseDevice_) (interopDevice_);
        interopDevice_ = nullptr;
    }
    auto* ctx = (ID3D11DeviceContext*) d3dContext_;
    safeRelease (ctx);
    d3dContext_ = nullptr;
    auto* dev = (ID3D11Device*) d3dDevice_;
    safeRelease (dev);
    d3dDevice_ = nullptr;
    ready_ = false;
}

} // namespace gpuexp

#endif // ARBIT_HAVE_D3D
