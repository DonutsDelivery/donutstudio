#include "accumulator.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace videohelper::pathtrace
{

void Accumulator::resize (std::uint32_t width, std::uint32_t height)
{
    width_ = width;
    height_ = height;
    pixels_.assign (static_cast<std::size_t> (width) * height, Pixel {});
    rejected_.store (0, std::memory_order_relaxed);
    {
        const std::lock_guard<std::mutex> lock (splatsMutex_);
        splatsReady_.store (false, std::memory_order_release);
        splats_.reset();
    }
    lightPaths_.store (0, std::memory_order_relaxed);
}

void Accumulator::reset (std::uint64_t generation) noexcept
{
    generation_ = generation;
    for (auto& pixel : pixels_)
        pixel = Pixel {};
    rejected_.store (0, std::memory_order_relaxed);
    if (splatsReady_.load (std::memory_order_acquire))
        for (std::size_t i = 0, n = pixels_.size() * 3; i < n; ++i)
            splats_[i].store (0, std::memory_order_relaxed);
    lightPaths_.store (0, std::memory_order_relaxed);
}

std::atomic<std::int64_t>* Accumulator::splats() noexcept
{
    if (splatsReady_.load (std::memory_order_acquire))
        return splats_.get();
    const std::lock_guard<std::mutex> lock (splatsMutex_);
    if (! splatsReady_.load (std::memory_order_relaxed))
    {
        try
        {
            // Value-initialized: every channel starts at 0.
            splats_ = std::make_unique<std::atomic<std::int64_t>[]> (pixels_.size() * 3);
        }
        catch (...)
        {
            return nullptr;
        }
        splatsReady_.store (true, std::memory_order_release);
    }
    return splats_.get();
}

void Accumulator::addSplat (std::uint32_t x, std::uint32_t y, Rgb radiance) noexcept
{
    if (x >= width_ || y >= height_)
        return;
    std::atomic<std::int64_t>* sums = splats();
    if (sums == nullptr)
        return;
    // Signed, because a splat's colour comes through the signed colour matching
    // functions. The range check only keeps llround defined.
    const auto fixed = [] (float value) -> std::int64_t {
        if (! std::isfinite (value))
            return 0;
        const double scaled = static_cast<double> (value) * kSplatScale;
        return std::llround (std::max (-4611686018427387904.0, std::min (scaled, 4611686018427387904.0)));
    };
    const std::size_t index = (static_cast<std::size_t> (y) * width_ + x) * 3;
    sums[index + 0].fetch_add (fixed (radiance.r), std::memory_order_relaxed);
    sums[index + 1].fetch_add (fixed (radiance.g), std::memory_order_relaxed);
    sums[index + 2].fetch_add (fixed (radiance.b), std::memory_order_relaxed);
}

Rgb Accumulator::splatMean (std::size_t index) const noexcept
{
    const std::uint64_t paths = lightPaths_.load (std::memory_order_relaxed);
    if (paths == 0 || ! splatsReady_.load (std::memory_order_acquire))
        return {};
    const double scale = 1.0 / (kSplatScale * static_cast<double> (paths));
    const std::atomic<std::int64_t>* sums = splats_.get() + index * 3;
    return { static_cast<float> (static_cast<double> (sums[0].load (std::memory_order_relaxed)) * scale),
             static_cast<float> (static_cast<double> (sums[1].load (std::memory_order_relaxed)) * scale),
             static_cast<float> (static_cast<double> (sums[2].load (std::memory_order_relaxed)) * scale) };
}

Rgb Accumulator::splatMean (std::uint32_t x, std::uint32_t y) const noexcept
{
    if (x >= width_ || y >= height_)
        return {};
    return splatMean (static_cast<std::size_t> (y) * width_ + x);
}

void Accumulator::addSample (std::uint32_t x, std::uint32_t y, Rgb radiance) noexcept
{
    if (x >= width_ || y >= height_)
        return;
    if (! std::isfinite (radiance.r) || ! std::isfinite (radiance.g) || ! std::isfinite (radiance.b))
    {
        rejected_.fetch_add (1, std::memory_order_relaxed);
        radiance.r = std::isfinite (radiance.r) ? radiance.r : 0.0f;
        radiance.g = std::isfinite (radiance.g) ? radiance.g : 0.0f;
        radiance.b = std::isfinite (radiance.b) ? radiance.b : 0.0f;
    }
    Pixel& pixel = pixels_[static_cast<std::size_t> (y) * width_ + x];
    // Incremental mean: stays accurate past the 2^24 samples where a float
    // running sum would stop growing.
    const std::uint32_t n = pixel.samples + 1;
    const float weight = 1.0f / static_cast<float> (n);
    pixel.r += (radiance.r - pixel.r) * weight;
    pixel.g += (radiance.g - pixel.g) * weight;
    pixel.b += (radiance.b - pixel.b) * weight;
    pixel.samples = n;
}

void Accumulator::addSampleSum (std::uint32_t x, std::uint32_t y, Rgb radiance, std::uint32_t count) noexcept
{
    if (x >= width_ || y >= height_ || count == 0)
        return;
    Pixel& pixel = pixels_[static_cast<std::size_t> (y) * width_ + x];
    const std::uint32_t n = pixel.samples + count;
    const float weight = 1.0f / static_cast<float> (n);
    // The device sums a pass in floats, so light near the largest float can overflow it or the
    // count times the mean. Such a pixel holds the largest float instead of going infinite.
    const auto merged = [&] (float sum, float mean) noexcept {
        const float next = mean + (sum - static_cast<float> (count) * mean) * weight;
        if (std::isfinite (next))
            return next;
        if (! std::isfinite (sum))
            return std::numeric_limits<float>::max();
        const double exact = (static_cast<double> (mean) * pixel.samples + sum) / n;
        return static_cast<float> (std::min (exact, static_cast<double> (std::numeric_limits<float>::max())));
    };
    pixel.r = merged (radiance.r, pixel.r);
    pixel.g = merged (radiance.g, pixel.g);
    pixel.b = merged (radiance.b, pixel.b);
    pixel.samples = n;
}

std::uint32_t Accumulator::sampleCount (std::uint32_t x, std::uint32_t y) const noexcept
{
    if (x >= width_ || y >= height_)
        return 0;
    return pixels_[static_cast<std::size_t> (y) * width_ + x].samples;
}

Rgb Accumulator::mean (std::uint32_t x, std::uint32_t y) const noexcept
{
    if (x >= width_ || y >= height_)
        return {};
    const std::size_t index = static_cast<std::size_t> (y) * width_ + x;
    const Pixel& pixel = pixels_[index];
    const Rgb splat = splatMean (index);
    return { pixel.r + splat.r, pixel.g + splat.g, pixel.b + splat.b };
}

bool Accumulator::resolveRgba32f (void* destination, std::size_t rowPitchBytes) const noexcept
{
    const std::size_t tight = static_cast<std::size_t> (width_) * kResolvedBytesPerPixel;
    if (rowPitchBytes == 0)
        rowPitchBytes = tight;
    if (destination == nullptr || rowPitchBytes < tight)
        return false;
    auto* bytes = static_cast<unsigned char*> (destination);
    for (std::uint32_t y = 0; y < height_; ++y)
    {
        unsigned char* row = bytes + static_cast<std::size_t> (y) * rowPitchBytes;
        const Pixel* source = pixels_.data() + static_cast<std::size_t> (y) * width_;
        for (std::uint32_t x = 0; x < width_; ++x)
        {
            const Pixel& pixel = source[x];
            const Rgb splat = splatMean (static_cast<std::size_t> (y) * width_ + x);
            const float rgba[4] { pixel.r + splat.r, pixel.g + splat.g, pixel.b + splat.b,
                                  pixel.samples != 0 ? 1.0f : 0.0f };
            std::memcpy (row + static_cast<std::size_t> (x) * kResolvedBytesPerPixel, rgba, sizeof rgba);
        }
    }
    return true;
}

std::vector<float> Accumulator::resolveRgba32f() const
{
    std::vector<float> out (static_cast<std::size_t> (width_) * height_ * 4);
    if (! out.empty())
        resolveRgba32f (out.data());
    return out;
}

} // namespace videohelper::pathtrace
