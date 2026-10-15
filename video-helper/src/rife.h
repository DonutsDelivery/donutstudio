#pragma once

// RifeEngine — RIFE v4 optical-flow frame interpolation via ONNX Runtime.
//
// Compiled only when the helper is configured with -DARBIT_WITH_ONNX=ON
// (ARBIT_HAVE_ONNX). The model binary is NOT shipped: it is downloaded on
// first use from the Arbit download server into the user cache dir and
// verified against a pinned SHA-256 (see video-helper/PROTOCOL.md §RIFE).
//
// Sessions try the CUDA execution provider first and fall back to CPU when
// CUDA/cuDNN are unavailable at runtime.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace arbitrife
{

class RifeEngine
{
public:
    RifeEngine();
    ~RifeEngine();
    RifeEngine (const RifeEngine&) = delete;
    RifeEngine& operator= (const RifeEngine&) = delete;

    // Locates/downloads the model and creates the ORT session.
    // Returns "" on success, error message otherwise.
    std::string init();

    // "rife-cuda" or "rife-cpu" after a successful init().
    const std::string& backend() const { return backend_; }

    // Synthesizes the frame at timestep (0..1) between two equally sized
    // straight-RGBA frames (timestep 0 = frame0, 1 = frame1). rgbaOut is
    // width*height*4, tightly packed, opaque alpha. Returns "" on success.
    std::string interpolate (const uint8_t* rgba0, int stride0,
                             const uint8_t* rgba1, int stride1,
                             int width, int height, float timestep,
                             std::vector<uint8_t>& rgbaOut);

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
