#pragma once

// Progressive radiance accumulation and the per-pixel random streams that feed
// it. The integrator adds one radiance sample per pixel per pass; the viewport
// resolves the running mean whenever it wants a frame.

#include <atomic>
#include <cstddef>
#include <cstdint>
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
inline std::uint32_t pcgHash (std::uint32_t input) noexcept
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
    PixelRng (std::uint32_t pixelIndex, std::uint32_t sampleIndex, std::uint64_t generation) noexcept
    {
        std::uint32_t h = pcgHash (static_cast<std::uint32_t> (generation >> 32));
        h = pcgHash (h ^ static_cast<std::uint32_t> (generation));
        h = pcgHash (h ^ sampleIndex);
        state_ = pcgHash (h ^ pixelIndex);
    }

    std::uint32_t nextUint() noexcept
    {
        const std::uint32_t old = state_;
        state_ = state_ * 747796405u + 2891336453u;
        const std::uint32_t word = ((old >> ((old >> 28u) + 4u)) ^ old) * 277803737u;
        return (word >> 22u) ^ word;
    }

    // Uniform in [0, 1), 24 bits of resolution.
    float nextFloat() noexcept
    {
        return static_cast<float> (nextUint() >> 8) * (1.0f / 16777216.0f);
    }

private:
    std::uint32_t state_ = 0;
};

// Running per-pixel mean of linear RGB radiance with a per-pixel sample count.
// Storage is 16 bytes per pixel. A tile scheduler that gives each pixel to one
// thread per pass may call addSample concurrently for different pixels.
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

    std::uint32_t width() const noexcept { return width_; }
    std::uint32_t height() const noexcept { return height_; }
    std::uint64_t generation() const noexcept { return generation_; }
    std::uint32_t sampleCount (std::uint32_t x, std::uint32_t y) const noexcept;
    Rgb mean (std::uint32_t x, std::uint32_t y) const noexcept;
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

    std::vector<Pixel> pixels_;
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    std::uint64_t generation_ = 0;
    std::atomic<std::uint64_t> rejected_ { 0 };
};

} // namespace videohelper::pathtrace
