#pragma once

// Progressive radiance accumulation and the per-pixel random streams that feed
// it. The integrator adds one radiance sample per pixel per pass; the viewport
// resolves the running mean whenever it wants a frame.

#include "devices/device_annotations.h"
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

namespace videohelper::pathtrace
{

struct Rgb
{
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
};

// PCG output permutation (RXS-M-XS, 32-bit) used as an integer hash.
// Jarzynski and Olano, "Hash Functions for GPU Rendering", JCGT 9(3), 2020.
PT_SHARED_FN std::uint32_t pcgHash (std::uint32_t input) noexcept
{
    const std::uint32_t state = input * 747796405u + 2891336453u;
    const std::uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Deterministic stream for one (pixel, sample, generation) triple. The same
// triple always yields the same sequence, independent of thread or tile order,
// so a render with any thread count is bit-identical. A Sobol or other
// low-discrepancy sampler can replace it later behind the same seed.
class PixelRng
{
public:
    PT_SHARED_FN PixelRng (std::uint32_t pixelIndex, std::uint32_t sampleIndex, std::uint64_t generation) noexcept
    {
        std::uint32_t h = pcgHash (static_cast<std::uint32_t> (generation >> 32));
        h = pcgHash (h ^ static_cast<std::uint32_t> (generation));
        h = pcgHash (h ^ sampleIndex);
        state_ = pcgHash (h ^ pixelIndex);
    }

    PT_SHARED_FN std::uint32_t nextUint() noexcept
    {
        const std::uint32_t old = state_;
        state_ = state_ * 747796405u + 2891336453u;
        const std::uint32_t word = ((old >> ((old >> 28u) + 4u)) ^ old) * 277803737u;
        return (word >> 22u) ^ word;
    }

    // Uniform in [0, 1), 24 bits of resolution.
    PT_SHARED_FN float nextFloat() noexcept
    {
        return static_cast<float> (nextUint() >> 8) * (1.0f / 16777216.0f);
    }

private:
    std::uint32_t state_ = 0;
};

// Running per-pixel mean of linear RGB radiance with a per-pixel sample count.
// Host only: the Metal kernel (device_metal.metal) leaves this out.
#if ! defined(PATHTRACE_DEVICE_METAL)
// Storage is 16 bytes per pixel, plus 28 for the guides below. A tile scheduler that gives each pixel to one
// thread per pass may call addSample concurrently for different pixels.
//
// Light paths (caustics) land on whatever pixel they project to, so they add
// through addSplat, which any thread may call for any pixel. Splats sum in
// 64-bit fixed point, so the total does not depend on thread or tile order and
// a render stays bit-identical across thread counts. The splat image is the
// splat sum divided by the number of light paths traced (addLightPath), the
// light-tracing estimator over the whole image. Its 24 bytes per pixel are
// allocated on the first splat, so a scene without caustics pays nothing and
// resolves exactly as before.
//
// Each pixel also keeps the denoiser's guides (Plan 3, Denoising): the albedo and
// the world-space shading normal of the surface the camera ray hit first, summed
// over the pixel's samples. addGuide follows the same tile ownership as addSample.
class Accumulator
{
public:
    static constexpr std::size_t kResolvedBytesPerPixel = 16;

    // Reallocates for a new size and clears every pixel. Keeps the generation.
    void resize (std::uint32_t width, std::uint32_t height);
    // Clears every pixel and starts a new generation, e.g. after a camera move.
    void reset (std::uint64_t generation) noexcept;

    // Adds one sample. A non-finite channel counts as 0 and is tallied in
    // rejectedSamples(), so one bad path cannot poison a pixel forever.
    void addSample (std::uint32_t x, std::uint32_t y, Rgb radiance) noexcept;
    // Adds one light path's contribution to a pixel. Thread-safe for any
    // pixel. A non-finite channel counts as 0. A negative channel subtracts: a
    // single wavelength's colour has negative components, and only their sum
    // over wavelengths is the right colour.
    void addSplat (std::uint32_t x, std::uint32_t y, Rgb radiance) noexcept;
    // Counts one traced light path, splatted or not. Thread-safe.
    void addLightPath() noexcept { lightPaths_.fetch_add (1, std::memory_order_relaxed); }
    // A device pass's film: count samples of this pixel, summing to radiance.
    void addSampleSum (std::uint32_t x, std::uint32_t y, Rgb radiance, std::uint32_t count) noexcept;
    void addLightPaths (std::uint64_t count) noexcept { lightPaths_.fetch_add (count, std::memory_order_relaxed); }
    std::uint64_t lightPaths() const noexcept { return lightPaths_.load (std::memory_order_relaxed); }
    // One sample's first-hit albedo and normal. A miss adds zeros, so the mean of a
    // pixel that saw the sky half the time is half its surface albedo.
    void addGuide (std::uint32_t x, std::uint32_t y, Rgb albedo, Rgb normal) noexcept { addGuideSum (x, y, albedo, normal, 1); }
    // A device pass's guides: count samples of this pixel, summing to albedo and normal.
    void addGuideSum (std::uint32_t x, std::uint32_t y, Rgb albedo, Rgb normal, std::uint32_t count) noexcept;
    // The mean albedo and the normalized mean normal of the pixel's samples. Zeros
    // for a pixel without guide samples.
    void guideMean (std::uint32_t x, std::uint32_t y, Rgb& albedo, Rgb& normal) const noexcept;

    std::uint32_t width() const noexcept { return width_; }
    std::uint32_t height() const noexcept { return height_; }
    std::uint64_t generation() const noexcept { return generation_; }
    std::uint32_t sampleCount (std::uint32_t x, std::uint32_t y) const noexcept;
    Rgb mean (std::uint32_t x, std::uint32_t y) const noexcept;
    // The light-path share of mean(): splat sums over lightPaths().
    Rgb splatMean (std::uint32_t x, std::uint32_t y) const noexcept;
    std::uint64_t rejectedSamples() const noexcept { return rejected_.load (std::memory_order_relaxed); }

    // Writes the mean as RGBA32F: four native-endian 32-bit floats per pixel in
    // R, G, B, A order, linear scene-referred radiance (no tone map, no sRGB
    // encoding). Rows run top to bottom, matching PinholeCamera pixel rows.
    // Row y starts at destination + y * rowPitchBytes; rowPitchBytes 0 means
    // the tight pitch width * 16, and any other pitch must be >= width * 16. Alpha is 1 for a
    // pixel with at least one sample and 0 for an unsampled pixel, whose RGB
    // is 0. Returns false for a null destination or a short pitch.
    bool resolveRgba32f (void* destination, std::size_t rowPitchBytes = 0) const noexcept;
    // Tight-pitch convenience: width * height * 4 floats.
    std::vector<float> resolveRgba32f() const;

private:
    struct Pixel
    {
        float r = 0.0f;
        float g = 0.0f;
        float b = 0.0f;
        std::uint32_t samples = 0;
    };
    static_assert (sizeof (Pixel) == 16, "Accumulator pixel must stay 16 bytes");
    // Fixed-point scale of the splat sums, 2^20.
    static constexpr double kSplatScale = 1048576.0;

    std::atomic<std::int64_t>* splats() noexcept;
    Rgb splatMean (std::size_t index) const noexcept;

    struct GuidePixel
    {
        float albedo[3] {};
        float normal[3] {};
        std::uint32_t samples = 0;
    };

    std::vector<Pixel> pixels_;
    std::vector<GuidePixel> guides_;
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    std::uint64_t generation_ = 0;
    std::atomic<std::uint64_t> rejected_ { 0 };
    // Three channels per pixel, allocated on the first splat.
    std::unique_ptr<std::atomic<std::int64_t>[]> splats_;
    std::atomic<bool> splatsReady_ { false };
    std::mutex splatsMutex_;
    std::atomic<std::uint64_t> lightPaths_ { 0 };
};

// The camera samples of one frame texture: those whose camera ray first hit a
// surface tinted by that frame. The tracer cannot read the frame, which lives on
// the GPU, so it keeps their radiance and uv here untinted and the GPU pass
// multiplies by the sampled frame (gpu_backend/cpu_scene_frame.h). Tiles own
// disjoint pixels, like Accumulator::addSample.
class FrameLayerFilm
{
public:
    struct Pixel
    {
        float r = 0.0f;
        float g = 0.0f;
        float b = 0.0f;
        float u = 0.0f;
        float v = 0.0f;
        std::uint32_t hits = 0;
    };

    void resize (std::uint32_t width, std::uint32_t height)
    {
        width_ = width;
        pixels_.assign (static_cast<std::size_t> (width) * height, Pixel {});
    }
    void add (std::uint32_t x, std::uint32_t y, Rgb radiance, float u, float v) noexcept
    {
        Pixel& pixel = pixels_[static_cast<std::size_t> (y) * width_ + x];
        const float top = std::numeric_limits<float>::max();
        pixel.r = std::min (pixel.r + radiance.r, top);
        pixel.g = std::min (pixel.g + radiance.g, top);
        pixel.b = std::min (pixel.b + radiance.b, top);
        pixel.u += u;
        pixel.v += v;
        ++pixel.hits;
    }
    const Pixel& pixel (std::uint32_t x, std::uint32_t y) const noexcept
    {
        return pixels_[static_cast<std::size_t> (y) * width_ + x];
    }

private:
    struct GuidePixel
    {
        float albedo[3] {};
        float normal[3] {};
        std::uint32_t samples = 0;
    };

    std::vector<Pixel> pixels_;
    std::vector<GuidePixel> guides_;
    std::uint32_t width_ = 0;
};
#endif

} // namespace videohelper::pathtrace
