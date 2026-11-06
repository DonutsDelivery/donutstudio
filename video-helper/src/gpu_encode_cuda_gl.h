#pragma once

// NVENC from GPU memory (Plan 3, export and encoders): the export's composite
// is converted on the GPU into NV12 or P010 planes (yuv_encode_math.h), mapped
// into CUDA through CUDA-GL interop and copied, device to device, into a CUDA
// frame from the encoder's frame pool. libcuda loads at run time, so a machine
// without the NVIDIA driver links this file and names what is missing.

#include <cstddef>
#include <map>
#include <string>

struct AVBufferRef;
struct AVFrame;

namespace videohelper
{

class CudaGlFrameBridge
{
public:
    CudaGlFrameBridge() = default;
    ~CudaGlFrameBridge();
    CudaGlFrameBridge (const CudaGlFrameBridge&) = delete;
    CudaGlFrameBridge& operator= (const CudaGlFrameBridge&) = delete;

    // hwFrames is the encoder's CUDA AVHWFramesContext. Returns false with the
    // reason when libcuda or its GL interop entry points are missing.
    bool open (AVBufferRef* hwFrames, std::string& error);
    bool isOpen() const noexcept { return hwFrames_ != nullptr; }

    // Copies an RGBA8 GL_TEXTURE_2D of width x height, top row first, into a new
    // CUDA frame. The GL context that owns the texture must be current. The
    // caller frees the frame. Returns nullptr with the reason on failure.
    AVFrame* capture (unsigned glTexture, int width, int height, std::string& error);

    // Copies a luma plane (R8 or R16, width x height) and a chroma plane (RG8 or
    // RG16, half size) into a new NV12 or P010 CUDA frame. bytesPerSample is 1
    // for NV12 and 2 for P010.
    AVFrame* capturePlanes (unsigned lumaTexture, unsigned chromaTexture, int width, int height,
                            int bytesPerSample, std::string& error);

    // Drops the interop registration of a texture about to be deleted.
    void release (unsigned glTexture);

    void close();

private:
    struct Plane
    {
        unsigned texture = 0;
        std::size_t widthInBytes = 0, height = 0;
    };
    AVFrame* copyPlanes (const Plane* planes, int count, std::string& error);

    AVBufferRef* hwFrames_ = nullptr;
    void* context_ = nullptr;  // FFmpeg's CUcontext, the first field of AVCUDADeviceContext
    std::map<unsigned, void*> registered_;  // GL texture -> CUgraphicsResource
};

} // namespace videohelper
