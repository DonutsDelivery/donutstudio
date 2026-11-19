#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

namespace renderpassoutput
{
inline constexpr std::uint32_t kWireVersion = 1;
inline constexpr std::size_t kMaximumAttachments = 8;
inline constexpr std::uint32_t kMaximumWidth = 16384;
inline constexpr std::uint32_t kMaximumHeight = 16384;
inline constexpr std::uint64_t kMaximumPixels = 8192ull * 8192ull;
inline constexpr std::uint64_t kMaximumTotalBytes = 2ull * 1024ull * 1024ull * 1024ull;
// The whole-frame byte bound: any representable total. Only a device or
// host allocation failure stops a frame. kMaximumTotalBytes stays as the
// unit of the fixed work envelopes that count complete 2 GiB walks.
inline constexpr std::uint64_t kRepresentableTotalBytes = std::numeric_limits<std::uint64_t>::max();

// Values and tokens are persistent. Add new outputs only immediately before Count.
enum class Output : std::uint8_t
{
    Color = 0,
    Depth = 1,
    Normal = 2,
    Motion = 3,
    Emission = 4,
    Mask = 5,
    MaterialId = 6,
    ObjectId = 7,
    Count = 8
};

inline constexpr std::array<Output, kMaximumAttachments> kOutputs {
    Output::Color,
    Output::Depth,
    Output::Normal,
    Output::Motion,
    Output::Emission,
    Output::Mask,
    Output::MaterialId,
    Output::ObjectId
};

inline constexpr std::string_view token(Output output) noexcept
{
    switch (output)
    {
        case Output::Color: return "color";
        case Output::Depth: return "depth";
        case Output::Normal: return "normal";
        case Output::Motion: return "motion";
        case Output::Emission: return "emission";
        case Output::Mask: return "mask";
        case Output::MaterialId: return "materialId";
        case Output::ObjectId: return "objectId";
        case Output::Count: break;
    }
    return {};
}

inline constexpr std::optional<Output> outputFromToken(std::string_view value) noexcept
{
    for (const auto output : kOutputs)
        if (token(output) == value)
            return output;
    return std::nullopt;
}

enum class PixelFormat : std::uint8_t
{
    Invalid = 0,
    R8Unorm = 1,
    RG16Float = 2,
    RGBA16Float = 3,
    R32Float = 4,
    R32Uint = 5
};

enum class ColorSpace : std::uint8_t
{
    Invalid = 0,
    Data = 1,
    LinearSRGB = 2
};

inline constexpr std::string_view token(PixelFormat format) noexcept
{
    switch (format)
    {
        case PixelFormat::R8Unorm: return "r8Unorm";
        case PixelFormat::RG16Float: return "rg16Float";
        case PixelFormat::RGBA16Float: return "rgba16Float";
        case PixelFormat::R32Float: return "r32Float";
        case PixelFormat::R32Uint: return "r32Uint";
        case PixelFormat::Invalid: break;
    }
    return {};
}

inline constexpr std::string_view token(ColorSpace colorSpace) noexcept
{
    switch (colorSpace)
    {
        case ColorSpace::Data: return "data";
        case ColorSpace::LinearSRGB: return "linearSRGB";
        case ColorSpace::Invalid: break;
    }
    return {};
}

struct Extent
{
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

inline constexpr bool operator==(Extent left, Extent right) noexcept
{
    return left.width == right.width && left.height == right.height;
}

inline constexpr bool operator!=(Extent left, Extent right) noexcept
{
    return !(left == right);
}

struct AttachmentDescription
{
    Output output = Output::Count;
    PixelFormat format = PixelFormat::Invalid;
    ColorSpace colorSpace = ColorSpace::Invalid;
    Extent extent;
};

struct Description
{
    std::uint32_t version = kWireVersion;
    Extent extent;
    std::vector<AttachmentDescription> attachments;
};

struct AdmissionLimits
{
    std::size_t attachments = kMaximumAttachments;
    std::uint32_t width = kMaximumWidth;
    std::uint32_t height = kMaximumHeight;
    std::uint64_t pixels = kMaximumPixels;
    // Any representable total. The device allocation check stops a frame
    // that does not fit, not a fixed byte ceiling.
    std::uint64_t totalBytes = kRepresentableTotalBytes;
};

enum class AdmissionFailure : std::uint8_t
{
    None = 0,
    UnsupportedVersion,
    InvalidLimits,
    InvalidExtent,
    AttachmentCountExceeded,
    UnknownOutput,
    UnknownPixelFormat,
    UnknownColorSpace,
    AttachmentExtentMismatch,
    DuplicateOutput,
    FormatMismatch,
    ColorSpaceMismatch,
    ByteCountOverflow,
    ByteLimitExceeded
};

inline constexpr std::string_view token(AdmissionFailure failure) noexcept
{
    switch (failure)
    {
        case AdmissionFailure::None: return "none";
        case AdmissionFailure::UnsupportedVersion: return "unsupportedVersion";
        case AdmissionFailure::InvalidLimits: return "invalidLimits";
        case AdmissionFailure::InvalidExtent: return "invalidExtent";
        case AdmissionFailure::AttachmentCountExceeded: return "attachmentCountExceeded";
        case AdmissionFailure::UnknownOutput: return "unknownOutput";
        case AdmissionFailure::UnknownPixelFormat: return "unknownPixelFormat";
        case AdmissionFailure::UnknownColorSpace: return "unknownColorSpace";
        case AdmissionFailure::AttachmentExtentMismatch: return "attachmentExtentMismatch";
        case AdmissionFailure::DuplicateOutput: return "duplicateOutput";
        case AdmissionFailure::FormatMismatch: return "formatMismatch";
        case AdmissionFailure::ColorSpaceMismatch: return "colorSpaceMismatch";
        case AdmissionFailure::ByteCountOverflow: return "byteCountOverflow";
        case AdmissionFailure::ByteLimitExceeded: return "byteLimitExceeded";
    }
    return {};
}

struct Requirements
{
    PixelFormat format = PixelFormat::Invalid;
    ColorSpace colorSpace = ColorSpace::Invalid;
};

inline constexpr Requirements requirements(Output output) noexcept
{
    switch (output)
    {
        case Output::Color: return { PixelFormat::RGBA16Float, ColorSpace::LinearSRGB };
        case Output::Depth: return { PixelFormat::R32Float, ColorSpace::Data };
        case Output::Normal: return { PixelFormat::RGBA16Float, ColorSpace::Data };
        case Output::Motion: return { PixelFormat::RG16Float, ColorSpace::Data };
        case Output::Emission: return { PixelFormat::RGBA16Float, ColorSpace::LinearSRGB };
        case Output::Mask: return { PixelFormat::R8Unorm, ColorSpace::Data };
        case Output::MaterialId: return { PixelFormat::R32Uint, ColorSpace::Data };
        case Output::ObjectId: return { PixelFormat::R32Uint, ColorSpace::Data };
        case Output::Count: break;
    }
    return {};
}

inline constexpr std::uint8_t bytesPerPixel(PixelFormat format) noexcept
{
    switch (format)
    {
        case PixelFormat::R8Unorm: return 1;
        case PixelFormat::RG16Float: return 4;
        case PixelFormat::RGBA16Float: return 8;
        case PixelFormat::R32Float: return 4;
        case PixelFormat::R32Uint: return 4;
        case PixelFormat::Invalid: break;
    }
    return 0;
}

struct AdmittedAttachment
{
    Output output = Output::Count;
    PixelFormat format = PixelFormat::Invalid;
    ColorSpace colorSpace = ColorSpace::Invalid;
    Extent extent;
    std::uint64_t byteCount = 0;
};

class AdmittedOutputs final
{
public:
    AdmittedOutputs(const AdmittedOutputs&) = default;
    AdmittedOutputs(AdmittedOutputs&&) = default;
    AdmittedOutputs& operator=(const AdmittedOutputs&) = delete;
    AdmittedOutputs& operator=(AdmittedOutputs&&) = delete;

    std::uint32_t version() const noexcept { return version_; }
    Extent extent() const noexcept { return extent_; }
    const std::vector<AdmittedAttachment>& attachments() const noexcept { return attachments_; }
    std::uint64_t totalByteCount() const noexcept { return totalByteCount_; }

    const AdmittedAttachment* find(Output output) const noexcept
    {
        const auto found = std::lower_bound(
            attachments_.begin(), attachments_.end(), output,
            [] (const AdmittedAttachment& attachment, Output wanted) {
                return static_cast<std::uint8_t>(attachment.output)
                    < static_cast<std::uint8_t>(wanted);
            });
        return found != attachments_.end() && found->output == output ? &*found : nullptr;
    }

private:
    friend std::optional<AdmittedOutputs> admit(
        const Description&, AdmissionFailure&, const AdmissionLimits&);

    AdmittedOutputs(Extent extent,
                    std::vector<AdmittedAttachment> attachments,
                    std::uint64_t totalByteCount)
        : extent_(extent),
          attachments_(std::move(attachments)),
          totalByteCount_(totalByteCount)
    {
    }

    const std::uint32_t version_ = kWireVersion;
    const Extent extent_;
    const std::vector<AdmittedAttachment> attachments_;
    const std::uint64_t totalByteCount_;
};

namespace detail
{
inline constexpr bool known(Output output) noexcept
{
    return static_cast<std::uint8_t>(output) < static_cast<std::uint8_t>(Output::Count);
}

inline constexpr bool known(PixelFormat format) noexcept
{
    return bytesPerPixel(format) != 0;
}

inline constexpr bool known(ColorSpace colorSpace) noexcept
{
    return colorSpace == ColorSpace::Data || colorSpace == ColorSpace::LinearSRGB;
}

inline constexpr bool checkedMultiply(std::uint64_t left,
                                      std::uint64_t right,
                                      std::uint64_t& result) noexcept
{
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left)
        return false;
    result = left * right;
    return true;
}

inline constexpr bool checkedAdd(std::uint64_t left,
                                 std::uint64_t right,
                                 std::uint64_t& result) noexcept
{
    if (right > std::numeric_limits<std::uint64_t>::max() - left)
        return false;
    result = left + right;
    return true;
}

inline constexpr bool limitsWithinHardBounds(const AdmissionLimits& limits) noexcept
{
    return limits.attachments <= kMaximumAttachments
        && limits.width <= kMaximumWidth
        && limits.height <= kMaximumHeight
        && limits.pixels <= kMaximumPixels;
}
} // namespace detail

inline std::optional<AdmittedOutputs> admit(
    const Description& untrusted,
    AdmissionFailure& failure,
    const AdmissionLimits& limits = {})
{
    failure = AdmissionFailure::None;
    const auto reject = [&failure] (AdmissionFailure reason)
        -> std::optional<AdmittedOutputs>
    {
        failure = reason;
        return std::nullopt;
    };

    if (untrusted.version != kWireVersion)
        return reject(AdmissionFailure::UnsupportedVersion);
    if (!detail::limitsWithinHardBounds(limits))
        return reject(AdmissionFailure::InvalidLimits);
    if (untrusted.extent.width == 0 || untrusted.extent.height == 0
        || untrusted.extent.width > limits.width
        || untrusted.extent.height > limits.height)
        return reject(AdmissionFailure::InvalidExtent);

    std::uint64_t pixelCount = 0;
    if (!detail::checkedMultiply(untrusted.extent.width, untrusted.extent.height, pixelCount))
        return reject(AdmissionFailure::ByteCountOverflow);
    if (pixelCount > limits.pixels)
        return reject(AdmissionFailure::InvalidExtent);
    if (untrusted.attachments.size() > limits.attachments)
        return reject(AdmissionFailure::AttachmentCountExceeded);

    std::array<bool, kMaximumAttachments> present {};
    std::uint64_t totalBytes = 0;
    for (const auto& attachment : untrusted.attachments)
    {
        if (!detail::known(attachment.output))
            return reject(AdmissionFailure::UnknownOutput);
        if (!detail::known(attachment.format))
            return reject(AdmissionFailure::UnknownPixelFormat);
        if (!detail::known(attachment.colorSpace))
            return reject(AdmissionFailure::UnknownColorSpace);
        if (attachment.extent != untrusted.extent)
            return reject(AdmissionFailure::AttachmentExtentMismatch);

        const auto outputIndex = static_cast<std::size_t>(attachment.output);
        if (present[outputIndex])
            return reject(AdmissionFailure::DuplicateOutput);
        present[outputIndex] = true;

        const auto exact = requirements(attachment.output);
        if (attachment.format != exact.format)
            return reject(AdmissionFailure::FormatMismatch);
        if (attachment.colorSpace != exact.colorSpace)
            return reject(AdmissionFailure::ColorSpaceMismatch);

        std::uint64_t attachmentBytes = 0;
        if (!detail::checkedMultiply(pixelCount, bytesPerPixel(attachment.format), attachmentBytes)
            || !detail::checkedAdd(totalBytes, attachmentBytes, totalBytes))
            return reject(AdmissionFailure::ByteCountOverflow);
        if (totalBytes > limits.totalBytes)
            return reject(AdmissionFailure::ByteLimitExceeded);
    }

    std::vector<AdmittedAttachment> admitted;
    admitted.reserve(untrusted.attachments.size());
    for (const auto& attachment : untrusted.attachments)
    {
        const auto attachmentBytes = pixelCount * bytesPerPixel(attachment.format);
        admitted.push_back({ attachment.output, attachment.format, attachment.colorSpace,
                             attachment.extent, attachmentBytes });
    }
    std::sort(admitted.begin(), admitted.end(), [] (const auto& left, const auto& right) {
        return static_cast<std::uint8_t>(left.output)
            < static_cast<std::uint8_t>(right.output);
    });
    return AdmittedOutputs(untrusted.extent, std::move(admitted), totalBytes);
}

static_assert(static_cast<std::uint8_t>(Output::Color) == 0);
static_assert(static_cast<std::uint8_t>(Output::Depth) == 1);
static_assert(static_cast<std::uint8_t>(Output::Normal) == 2);
static_assert(static_cast<std::uint8_t>(Output::Motion) == 3);
static_assert(static_cast<std::uint8_t>(Output::Emission) == 4);
static_assert(static_cast<std::uint8_t>(Output::Mask) == 5);
static_assert(static_cast<std::uint8_t>(Output::MaterialId) == 6);
static_assert(static_cast<std::uint8_t>(Output::ObjectId) == 7);
static_assert(static_cast<std::uint8_t>(Output::Count) == kMaximumAttachments);
} // namespace renderpassoutput
