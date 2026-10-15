#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>

namespace videotemporal
{

enum class Mode
{
    frameDelay,
    feedback,
    echo,
    stutter,
    longExposure
};

enum class PixelFormat
{
    r8,
    r16,
    rgba8,
    rgba16f,
    rgba32f
};

struct ImageExtent
{
    uint32_t width = 0;
    uint32_t height = 0;
};

class ResourceContract final
{
public:
    ResourceContract(Mode mode, PixelFormat format, ImageExtent extent,
                     uint32_t historyLength) noexcept
        : mode_(mode), format_(format), extent_(extent), historyLength_(historyLength)
    {
    }

    ResourceContract(const ResourceContract&) = default;
    ResourceContract(ResourceContract&&) = default;
    ResourceContract& operator=(const ResourceContract&) = delete;
    ResourceContract& operator=(ResourceContract&&) = delete;

    Mode mode() const noexcept { return mode_; }
    PixelFormat format() const noexcept { return format_; }
    ImageExtent extent() const noexcept { return extent_; }
    uint32_t historyLength() const noexcept { return historyLength_; }

    static constexpr bool authorizesGraphCycles = false;

private:
    const Mode mode_;
    const PixelFormat format_;
    const ImageExtent extent_;
    const uint32_t historyLength_;
};

static_assert(! std::is_copy_assignable_v<ResourceContract>);
static_assert(! ResourceContract::authorizesGraphCycles);

struct ResourceLimits
{
    uint32_t maximumImageDimension = 8192;
    uint32_t maximumHistoryLength = 256;
    uint64_t maximumHistoryBytes = 512ull * 1024ull * 1024ull;
    uint64_t maximumTransientBytes = 256ull * 1024ull * 1024ull;
};

struct ResourceFootprint
{
    uint32_t retainedImages = 0;
    uint32_t transientImages = 0;
    uint64_t bytesPerImage = 0;
    uint64_t historyBytes = 0;
    uint64_t transientBytes = 0;
};

namespace detail
{
inline uint64_t bytesPerPixel(PixelFormat format) noexcept
{
    switch (format)
    {
        case PixelFormat::r8: return 1;
        case PixelFormat::r16: return 2;
        case PixelFormat::rgba8: return 4;
        case PixelFormat::rgba16f: return 8;
        case PixelFormat::rgba32f: return 16;
    }
    return 0;
}

inline bool checkedMultiply(uint64_t left, uint64_t right, uint64_t& result) noexcept
{
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left)
        return false;
    result = left * right;
    return true;
}

inline bool historyLengthMatchesMode(Mode mode, uint32_t historyLength,
                                     std::string& error)
{
    if (historyLength == 0)
    {
        error = "temporal resource history length must be positive";
        return false;
    }

    switch (mode)
    {
        case Mode::frameDelay:
            // The retained count is the exact frame delay.
            return true;
        case Mode::feedback:
            // Feedback reads the current target and writes the alternate target.
            if (historyLength == 2) return true;
            error = "temporal feedback requires exactly two ping-pong history images";
            return false;
        case Mode::echo:
            // Echo reads the declared number of preceding inputs.
            return true;
        case Mode::stutter:
            // Stutter retains one held input until the evaluator replaces it.
            if (historyLength == 1) return true;
            error = "temporal stutter requires exactly one history image";
            return false;
        case Mode::longExposure:
            // Long exposure uses a bounded rolling window of preceding inputs.
            return true;
    }

    error = "temporal resource mode is invalid";
    return false;
}
} // namespace detail

inline bool admitResource(const ResourceContract& contract, const ResourceLimits& limits,
                          bool graphContainsCycle, ResourceFootprint& footprint,
                          std::string& error)
{
    footprint = {};
    error.clear();

    if (graphContainsCycle)
    {
        error = "temporal resources do not authorize graph cycles";
        return false;
    }

    const auto bytesPerPixel = detail::bytesPerPixel(contract.format());
    if (bytesPerPixel == 0)
    {
        error = "temporal resource requires a declared image format";
        return false;
    }

    const auto extent = contract.extent();
    if (extent.width == 0 || extent.height == 0)
    {
        error = "temporal resource requires a positive explicit image extent";
        return false;
    }
    if (limits.maximumImageDimension == 0
        || extent.width > limits.maximumImageDimension
        || extent.height > limits.maximumImageDimension)
    {
        error = "temporal resource image dimension capacity exceeded";
        return false;
    }

    if (! detail::historyLengthMatchesMode(contract.mode(), contract.historyLength(), error))
        return false;
    if (limits.maximumHistoryLength == 0
        || contract.historyLength() > limits.maximumHistoryLength)
    {
        error = "temporal resource history length capacity exceeded";
        return false;
    }

    uint64_t pixels = 0;
    uint64_t bytesPerImage = 0;
    uint64_t historyBytes = 0;
    if (! detail::checkedMultiply(extent.width, extent.height, pixels)
        || ! detail::checkedMultiply(pixels, bytesPerPixel, bytesPerImage)
        || ! detail::checkedMultiply(bytesPerImage, contract.historyLength(), historyBytes))
    {
        error = "temporal resource byte accounting overflow";
        return false;
    }
    if (historyBytes > limits.maximumHistoryBytes)
    {
        error = "temporal resource history byte capacity exceeded";
        return false;
    }

    // Frame delay needs one persistent result image so a paused preview can
    // present the last delayed frame without advancing or mutating the exact
    // history ring. It is admitted as transient output, never as history.
    const uint32_t transientImages = contract.mode() == Mode::frameDelay ? 1u : 0u;
    uint64_t transientBytes = 0;
    if (! detail::checkedMultiply(bytesPerImage, transientImages, transientBytes))
    {
        error = "temporal resource byte accounting overflow";
        return false;
    }
    if (transientBytes > limits.maximumTransientBytes)
    {
        error = "temporal resource transient byte capacity exceeded";
        return false;
    }

    footprint.retainedImages = contract.historyLength();
    footprint.transientImages = transientImages;
    footprint.bytesPerImage = bytesPerImage;
    footprint.historyBytes = historyBytes;
    footprint.transientBytes = transientBytes;
    return true;
}

} // namespace videotemporal
