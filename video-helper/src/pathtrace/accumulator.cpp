#include "accumulator.h"

#include <cmath>
#include <cstring>

namespace videohelper::pathtrace
{

void Accumulator::resize (std::uint32_t width, std::uint32_t height)
{
    width_ = width;
    height_ = height;
    pixels_.assign (static_cast<std::size_t> (width) * height, Pixel {});
    rejected_.store (0, std::memory_order_relaxed);
}

void Accumulator::reset (std::uint64_t generation) noexcept
{
    generation_ = generation;
    for (auto& pixel : pixels_)
        pixel = Pixel {};
    rejected_.store (0, std::memory_order_relaxed);
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
    const Pixel& pixel = pixels_[static_cast<std::size_t> (y) * width_ + x];
    return { pixel.r, pixel.g, pixel.b };
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
            const float rgba[4] { pixel.r, pixel.g, pixel.b, pixel.samples != 0 ? 1.0f : 0.0f };
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
