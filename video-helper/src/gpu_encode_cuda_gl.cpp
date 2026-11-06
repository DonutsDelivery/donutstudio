#include "gpu_encode_cuda_gl.h"

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
}

#include <cstddef>
#include <type_traits>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace videohelper
{
namespace
{

// The CUDA driver API declarations this file needs, without the CUDA headers.
using CuDevicePtr = unsigned long long;
using CuHandle = void*;
constexpr unsigned kGlTexture2D = 0x0DE1;
constexpr unsigned kRegisterReadOnly = 1;     // CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY
constexpr int kMemoryDevice = 2;               // CU_MEMORYTYPE_DEVICE
constexpr int kMemoryArray = 3;                // CU_MEMORYTYPE_ARRAY

struct CuMemcpy2D  // CUDA_MEMCPY2D
{
    std::size_t srcXInBytes = 0, srcY = 0;
    int srcMemoryType = 0;
    const void* srcHost = nullptr;
    CuDevicePtr srcDevice = 0;
    CuHandle srcArray = nullptr;
    std::size_t srcPitch = 0;
    std::size_t dstXInBytes = 0, dstY = 0;
    int dstMemoryType = 0;
    void* dstHost = nullptr;
    CuDevicePtr dstDevice = 0;
    CuHandle dstArray = nullptr;
    std::size_t dstPitch = 0;
    std::size_t widthInBytes = 0, height = 0;
};

struct Interop
{
    int (*ctxPush) (CuHandle) = nullptr;
    int (*ctxPop) (CuHandle*) = nullptr;
    int (*registerImage) (CuHandle*, unsigned, unsigned, unsigned) = nullptr;
    int (*unregister) (CuHandle) = nullptr;
    int (*map) (unsigned, CuHandle*, CuHandle) = nullptr;
    int (*unmap) (unsigned, CuHandle*, CuHandle) = nullptr;
    int (*mappedArray) (CuHandle*, CuHandle, unsigned, unsigned) = nullptr;
    int (*copy2D) (const CuMemcpy2D*) = nullptr;
    std::string problem;

    Interop()
    {
#if defined(_WIN32)
        const char* name = "nvcuda.dll";
        const auto library = LoadLibraryA (name);
        const auto find = [library] (const char* symbol) { return reinterpret_cast<void*> (GetProcAddress (library, symbol)); };
#else
        const char* name = "libcuda.so.1";
        void* const library = dlopen (name, RTLD_NOW | RTLD_LOCAL);
        const auto find = [library] (const char* symbol) { return dlsym (library, symbol); };
#endif
        if (library == nullptr)
        {
            problem = std::string (name) + " did not load";
            return;
        }
        const auto bind = [&] (auto& function, const char* symbol) {
            function = reinterpret_cast<std::remove_reference_t<decltype (function)>> (find (symbol));
            if (function == nullptr && problem.empty())
                problem = std::string (name) + " has no " + symbol;
        };
        bind (ctxPush, "cuCtxPushCurrent_v2");
        bind (ctxPop, "cuCtxPopCurrent_v2");
        bind (registerImage, "cuGraphicsGLRegisterImage");
        bind (unregister, "cuGraphicsUnregisterResource");
        bind (map, "cuGraphicsMapResources");
        bind (unmap, "cuGraphicsUnmapResources");
        bind (mappedArray, "cuGraphicsSubResourceGetMappedArray");
        bind (copy2D, "cuMemcpy2D_v2");
    }
};

const Interop& interop()
{
    static const Interop loaded;
    return loaded;
}

std::string cudaError (const char* step, int code)
{
    return std::string ("CUDA-GL interop: ") + step + " failed with CUDA error " + std::to_string (code);
}

} // namespace

CudaGlFrameBridge::~CudaGlFrameBridge()
{
    close();
}

bool CudaGlFrameBridge::open (AVBufferRef* hwFrames, std::string& error)
{
    close();
    const auto& api = interop();
    if (! api.problem.empty())
    {
        error = "CUDA-GL interop: " + api.problem;
        return false;
    }
    if (hwFrames == nullptr)
    {
        error = "CUDA-GL interop: the encoder has no CUDA frame pool";
        return false;
    }
    const auto* frames = reinterpret_cast<const AVHWFramesContext*> (hwFrames->data);
    if (frames->device_ctx == nullptr || frames->device_ctx->type != AV_HWDEVICE_TYPE_CUDA
        || frames->device_ctx->hwctx == nullptr)
    {
        error = "CUDA-GL interop: the encoder's frame pool is not CUDA";
        return false;
    }
    // AVCUDADeviceContext starts with its CUcontext; hwcontext_cuda.h would
    // pull in cuda.h, which a build without the CUDA SDK does not have.
    context_ = *static_cast<void* const*> (frames->device_ctx->hwctx);
    hwFrames_ = av_buffer_ref (hwFrames);
    if (context_ == nullptr || hwFrames_ == nullptr)
    {
        error = "CUDA-GL interop: FFmpeg's CUDA device has no context";
        close();
        return false;
    }
    return true;
}

AVFrame* CudaGlFrameBridge::capture (unsigned glTexture, int width, int height, std::string& error)
{
    const Plane plane { glTexture, static_cast<std::size_t> (width) * 4, static_cast<std::size_t> (height) };
    return copyPlanes (&plane, 1, error);
}

AVFrame* CudaGlFrameBridge::capturePlanes (unsigned lumaTexture, unsigned chromaTexture, int width, int height,
                                           int bytesPerSample, std::string& error)
{
    // The chroma plane interleaves Cb and Cr: width / 2 pairs per row.
    const auto rowBytes = static_cast<std::size_t> (width) * static_cast<std::size_t> (bytesPerSample);
    const Plane planes[2] = { { lumaTexture, rowBytes, static_cast<std::size_t> (height) },
                              { chromaTexture, rowBytes, static_cast<std::size_t> (height / 2) } };
    return copyPlanes (planes, 2, error);
}

AVFrame* CudaGlFrameBridge::copyPlanes (const Plane* planes, int count, std::string& error)
{
    if (hwFrames_ == nullptr)
    {
        error = "CUDA-GL interop is not open";
        return nullptr;
    }
    error.clear();
    const auto& api = interop();
    if (int code = api.ctxPush (context_); code != 0)
    {
        error = cudaError ("making FFmpeg's CUDA context current", code);
        return nullptr;
    }
    AVFrame* frame = av_frame_alloc();
    if (frame == nullptr || av_hwframe_get_buffer (hwFrames_, frame, 0) < 0)
    {
        av_frame_free (&frame);
        error = "CUDA-GL interop: the CUDA frame pool is out of frames";
    }
    const auto fail = [&] (const char* step, int code) {
        error = cudaError (step, code);
        av_frame_free (&frame);
    };
    for (int i = 0; i < count && error.empty(); ++i)
    {
        CuHandle resource = nullptr;
        if (auto found = registered_.find (planes[i].texture); found != registered_.end())
            resource = found->second;
        else if (int code = api.registerImage (&resource, planes[i].texture, kGlTexture2D, kRegisterReadOnly); code != 0)
        {
            fail ("registering the GL texture (is GL on the NVIDIA GPU?)", code);
            break;
        }
        else
            registered_[planes[i].texture] = resource;

        if (int code = api.map (1, &resource, nullptr); code != 0)
        {
            fail ("mapping the GL texture", code);
            break;
        }
        CuHandle array = nullptr;
        if (int code = api.mappedArray (&array, resource, 0, 0); code != 0)
            fail ("reading the mapped GL texture", code);
        else
        {
            CuMemcpy2D copy;
            copy.srcMemoryType = kMemoryArray;
            copy.srcArray = array;
            copy.dstMemoryType = kMemoryDevice;
            copy.dstDevice = reinterpret_cast<CuDevicePtr> (frame->data[i]);
            copy.dstPitch = static_cast<std::size_t> (frame->linesize[i]);
            copy.widthInBytes = planes[i].widthInBytes;
            copy.height = planes[i].height;
            if (int code = api.copy2D (&copy); code != 0)
                fail ("copying the texture into the encoder frame", code);
        }
        api.unmap (1, &resource, nullptr);
    }
    CuHandle popped = nullptr;
    api.ctxPop (&popped);
    return frame;
}

void CudaGlFrameBridge::release (unsigned glTexture)
{
    const auto found = registered_.find (glTexture);
    if (found == registered_.end())
        return;
    const auto& api = interop();
    if (api.ctxPush (context_) == 0)
    {
        api.unregister (found->second);
        CuHandle popped = nullptr;
        api.ctxPop (&popped);
    }
    registered_.erase (found);
}

void CudaGlFrameBridge::close()
{
    const auto& api = interop();
    if (! registered_.empty() && api.problem.empty() && context_ != nullptr && api.ctxPush (context_) == 0)
    {
        for (const auto& [texture, resource] : registered_)
            api.unregister (resource);
        CuHandle popped = nullptr;
        api.ctxPop (&popped);
    }
    registered_.clear();
    av_buffer_unref (&hwFrames_);
    context_ = nullptr;
}

} // namespace videohelper
