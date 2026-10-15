#pragma once

// RifeEngineNcnn — RIFE v4 optical-flow frame interpolation via ncnn + Vulkan.
//
// The portable, default-on counterpart to the NVIDIA-only ONNX/CUDA RifeEngine
// (rife.h). Runs on any conformant Vulkan driver (AMD RADV/AMDVLK, Intel ANV,
// NVIDIA, Apple via MoltenVK) — one binary, no CUDA/ONNX/PyTorch. Compiled only
// when the helper is configured with -DARBIT_WITH_NCNN=ON (ARBIT_HAVE_NCNN),
// which ship builds set ON by default.
//
// Mirrors the RifeEngine public contract exactly so the two are drop-in
// interchangeable behind InterpEngine / the exporter. This RGBA-in/RGBA-out
// surface is the correctness baseline; the zero-copy VkImage-in/VkImage-out
// path (no CPU round-trip) is layered on in a follow-up without changing it.
//
// Net source (vendored, MIT): src/rife_net/ — nihui/rife-ncnn-vulkan over
// Tencent/ncnn (BSD-3). Model is a .param/.bin pair (NOT the .onnx); located via
// ARBIT_RIFE_NCNN_MODEL for dev, VPS fetch + SHA pin for ship.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace arbitrife
{

// Handle to a GPU-resident interpolated frame living in the engine's ExportableRing
// (zero-copy path). The worker thread produces it; the render thread imports the
// slot's OPAQUE_FD memory ONCE per slot (glImportMemoryFdEXT) and samples the
// interleaved RGB8 buffer (w*h*3 bytes). POD so it can cross the worker/render
// boundary freely. `slot` lets the consumer cache its per-slot GL import and detect
// reuse (a new frame in the same slot overwrites the old one). Defined
// unconditionally; only populated when the engine is built with ARBIT_RIFE_ZEROCOPY.
struct GpuFrameHandle
{
    int    slot      = -1;   // ring slot index; stable per physical buffer
    int    memFd     = -1;   // OPAQUE_FD for the slot's VkDeviceMemory (import once per slot)
    size_t allocSize = 0;    // VkMemoryRequirements.size — the glImportMemoryFdEXT size
    int    w         = 0;    // interleaved RGB8, tightly packed, w*h*3 bytes
    int    h         = 0;
    bool   valid     = false;
};

class RifeEngineNcnn
{
public:
    RifeEngineNcnn();
    ~RifeEngineNcnn();
    RifeEngineNcnn (const RifeEngineNcnn&) = delete;
    RifeEngineNcnn& operator= (const RifeEngineNcnn&) = delete;

    // Creates the Vulkan instance/device and loads the model. Returns "" on
    // success, error message otherwise (no GPU, model missing, etc.).
    std::string init();

    // "rife-vulkan" after a successful init().
    const std::string& backend() const { return backend_; }

    // Synthesizes the frame at timestep (0..1) between two equally sized
    // straight-RGBA frames (timestep 0 = frame0, 1 = frame1). rgbaOut is
    // width*height*4, tightly packed, opaque alpha. Returns "" on success.
    std::string interpolate (const uint8_t* rgba0, int stride0,
                             const uint8_t* rgba1, int stride1,
                             int width, int height, float timestep,
                             std::vector<uint8_t>& rgbaOut);

#ifdef ARBIT_RIFE_ZEROCOPY
    // True once init() succeeded AND the ncnn VkDevice can export OPAQUE_FD memory
    // (the build enabled zero-copy and the driver advertises VK_KHR_external_memory_fd).
    // When false, callers must use the CPU interpolate() path above.
    bool gpuZeroCopyAvailable() const;

    // Zero-copy variant of interpolate(): synthesizes the in-between frame straight
    // into a GPU-resident, OPAQUE_FD-exportable buffer (no CPU download of the
    // output). `out` describes the slot to import on the render thread. Returns ""
    // on success; on any error returns a message and the caller falls back to the
    // CPU interpolate() path. Input frames are still CPU-uploaded (decode is CPU).
    std::string interpolateToGpu (const uint8_t* rgba0, int stride0,
                                  const uint8_t* rgba1, int stride1,
                                  int width, int height, float timestep,
                                  GpuFrameHandle& out);

    // Place a single decoded frame (no interpolation) into a GPU ring slot — used
    // for the alpha-extreme buckets the worker fills with a source frame directly.
    std::string uploadFrameToGpu (const uint8_t* rgba, int stride,
                                  int width, int height, GpuFrameHandle& out);
#endif

    // Inference statistics (for logs / perf reporting).
    int    inferenceCount() const { return inferCount_; }
    double totalInferenceMs() const { return inferMsTotal_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string backend_;
    int inferCount_ = 0;
    double inferMsTotal_ = 0.0;
};

} // namespace arbitrife
