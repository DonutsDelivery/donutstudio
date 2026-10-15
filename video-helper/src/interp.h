#pragma once

// InterpEngine — async RIFE frame-interpolation worker for the LIVE viewport
// (retime tier 2, "Speed Warp"). This is the realtime counterpart to the
// export-only RIFE path in exporter.cpp.
//
// Design (see artifacts/media-machine/live-interp-warp-plan.md §5 Phase 3):
//   * Owns ONE dedicated worker thread + its OWN RifeEngine instance + its own
//     decoders (MediaContext per clip). Fully decoupled from the render thread's
//     ClipStream — the two never share a decoder or a RIFE session.
//   * The render thread interacts non-blocking only: request() posts the time it
//     is heading toward (latest-wins per clip); fetch() copies out a finished
//     interpolated frame if one is cached for (clipId, srcSec). ALL GL uploads
//     stay on the render thread — the worker needs no GL context.
//   * Frames are quantized to a shared time grid (kBucketsPerSec) so render and
//     worker agree on cache keys. On each request the worker fills the whole
//     source-frame bracket straddling srcSec (plus one ahead), so slow playback
//     — which dwells many display frames per source frame — hits the cache after
//     the first warm-up.
//   * Compiled when EITHER RIFE backend is built — ARBIT_HAVE_NCNN (portable
//     ncnn-Vulkan, preferred and default-on) or ARBIT_HAVE_ONNX. The viewport uses
//     it under #if ARBIT_HAVE_ONNX || ARBIT_HAVE_NCNN; Frame Blend (tier 1) stays
//     unconditional.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace arbitinterp
{

// Shared render/worker quantization grid: srcSec -> bucket = round(srcSec * N).
// 120 Hz (~8.3 ms) is finer than any display cadence yet coarse enough that a
// source-frame bracket holds only a handful of buckets.
constexpr double kBucketsPerSec = 120.0;

class InterpEngine
{
public:
    InterpEngine();
    ~InterpEngine();
    InterpEngine (const InterpEngine&) = delete;
    InterpEngine& operator= (const InterpEngine&) = delete;

    // Spawns the worker thread and initializes RIFE on it (model download /
    // session build happen off the render thread). Idempotent; safe to call once.
    void start();

    // True once RIFE initialized with a realtime EP (CUDA / Core ML). A CPU-only
    // session reports false — the caller must NOT drive tier 2 with it (it would
    // crater fps); Frame Blend covers that case instead.
    bool available() const;

    // 2 = rife-cuda, 3 = rife-coreml, 0 otherwise. Matches the viewport's
    // interpBackend tier codes so info() reports the right string.
    int tierCode() const;

    // "rife-cuda" / "rife-coreml" / "rife-cpu" / "" (after start()).
    std::string backend() const;

    // Why the RIFE backend is unavailable ("" while warming up or once ready):
    // the init error (e.g. "ARBIT_RIFE_NCNN_MODEL is not set ...") or the
    // CPU-EP-not-realtime refusal. Surfaced so the UI can show WHY the viewport
    // fell back to Frame Blend instead of silently degrading.
    std::string lastError() const;

    // Render thread: ask the worker to keep frames around srcSec warm for this
    // clip. Overwrites any pending request for clipId (latest wins). Non-blocking.
    struct Request
    {
        int clipId = 0;
        std::string sourcePath;
        double sourceFps = 0.0;     // image-sequence fps hint (0 = container fps)
        int seqStart = -1;
        double srcSec = 0.0;        // the source time the render thread is heading toward
        int previewW = 0, previewH = 0; // decode+inference clamp box (0 = default 360p-class)
    };
    void request (const Request& r);

    // Render thread: fetch a finished interpolated frame for (clipId, srcSec) if
    // one is cached. Copies tightly packed RGBA (stride = w*4) into out. Returns
    // true on a cache hit. Non-blocking.
    bool fetch (int clipId, double srcSec, std::vector<uint8_t>& out, int& w, int& h);

    // Render thread: drop a clip's decoder + cached frames (segment went away).
    void forget (int clipId);

    // Stats for logs / diagnostics.
    uint64_t inferenceCount() const;
    uint64_t cacheHits() const;
    uint64_t cacheMisses() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace arbitinterp
