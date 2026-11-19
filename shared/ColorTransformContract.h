#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string_view>
#include <type_traits>

namespace colortransform
{
inline constexpr std::uint32_t kWireVersion = 1;
inline constexpr std::uint32_t kColorMathVersion = 1;
inline constexpr std::uint32_t kMaximumDimension = 16384;
inline constexpr std::uint64_t kMaximumPixels = 8192ull * 8192ull;
inline constexpr std::uint64_t kMaximumResidentBytes = 2ull * 1024ull * 1024ull * 1024ull;
inline constexpr std::size_t kMaximumStages = 8;
inline constexpr double kMinimumReferenceWhiteNits = 80.0;
inline constexpr double kMaximumReferenceWhiteNits = 500.0;
inline constexpr double kMaximumPeakNits = 10000.0;
inline constexpr double kMinimumHdrPeakNits = 400.0;

// These tokens are the existing visual graph PixelFormat vocabulary.
enum class PixelFormat : std::uint8_t
{
    Unspecified = 0,
    R8 = 1,
    R16 = 2,
    RGBA8 = 3,
    RGBA16F = 4,
    RGBA32F = 5
};

// These tokens are the existing visual graph ColorSpace vocabulary.
enum class ColorSpace : std::uint8_t
{
    Unspecified = 0,
    LinearSRGB = 1,
    SRGB = 2,
    DisplayP3 = 3,
    Rec709 = 4,
    Rec2020 = 5
};

enum class TransferFunction : std::uint8_t
{
    Unspecified = 0,
    Linear = 1,
    SRGB = 2,
    Gamma22 = 3,
    PQ = 4,
    HLG = 5,
    Rec709 = 6
};

inline constexpr bool isHdrTransfer(TransferFunction value) noexcept
{
    return value == TransferFunction::PQ || value == TransferFunction::HLG;
}

enum class AlphaMode : std::uint8_t
{
    Unspecified = 0,
    Opaque = 1,
    Straight = 2,
    Premultiplied = 3
};

enum class OutputIntent : std::uint8_t
{
    SceneLinear = 0,
    SdrDisplay = 1,
    HdrPq = 2,
    HdrHlg = 3
};

enum class ToneMap : std::uint8_t
{
    None = 0,
    Reinhard = 1,
    // Scene-referred view transform for a linear sRGB source. shared/AgxViewTransform.h
    AgX = 2
};

enum class BackendCapability : std::uint8_t
{
    Unspecified = 0,
    NativeGpu = 1
};

enum class StageKind : std::uint8_t
{
    UnassociateAlpha = 0,
    DecodeTransfer = 1,
    ConvertToWorkingPrimaries = 2,
    ToneMapLuminance = 3,
    ConvertFromWorkingPrimaries = 4,
    EncodeTransfer = 5,
    AssociateAlpha = 6,
    ViewTransform = 7
};

inline constexpr std::string_view token(PixelFormat value) noexcept
{
    switch (value)
    {
        case PixelFormat::R8: return "r8";
        case PixelFormat::R16: return "r16";
        case PixelFormat::RGBA8: return "rgba8";
        case PixelFormat::RGBA16F: return "rgba16f";
        case PixelFormat::RGBA32F: return "rgba32f";
        case PixelFormat::Unspecified: break;
    }
    return "unspecified";
}

inline constexpr std::string_view token(ColorSpace value) noexcept
{
    switch (value)
    {
        case ColorSpace::LinearSRGB: return "linearSRGB";
        case ColorSpace::SRGB: return "sRGB";
        case ColorSpace::DisplayP3: return "displayP3";
        case ColorSpace::Rec709: return "rec709";
        case ColorSpace::Rec2020: return "rec2020";
        case ColorSpace::Unspecified: break;
    }
    return "unspecified";
}

inline constexpr std::string_view token(TransferFunction value) noexcept
{
    switch (value)
    {
        case TransferFunction::Linear: return "linear";
        case TransferFunction::SRGB: return "sRGB";
        case TransferFunction::Gamma22: return "gamma22";
        case TransferFunction::PQ: return "pq";
        case TransferFunction::HLG: return "hlg";
        case TransferFunction::Rec709: return "rec709";
        case TransferFunction::Unspecified: break;
    }
    return "unspecified";
}

inline constexpr std::string_view token(AlphaMode value) noexcept
{
    switch (value)
    {
        case AlphaMode::Opaque: return "opaque";
        case AlphaMode::Straight: return "straight";
        case AlphaMode::Premultiplied: return "premultiplied";
        case AlphaMode::Unspecified: break;
    }
    return "unspecified";
}

inline constexpr std::string_view token(OutputIntent value) noexcept
{
    switch (value)
    {
        case OutputIntent::SceneLinear: return "sceneLinear";
        case OutputIntent::SdrDisplay: return "sdrDisplay";
        case OutputIntent::HdrPq: return "hdrPq";
        case OutputIntent::HdrHlg: return "hdrHlg";
    }
    return {};
}

inline constexpr std::string_view token(ToneMap value) noexcept
{
    switch (value)
    {
        case ToneMap::None: return "none";
        case ToneMap::Reinhard: return "reinhard";
        case ToneMap::AgX: return "agx";
    }
    return {};
}

inline constexpr std::string_view token(StageKind value) noexcept
{
    switch (value)
    {
        case StageKind::UnassociateAlpha: return "unassociateAlpha";
        case StageKind::DecodeTransfer: return "decodeTransfer";
        case StageKind::ConvertToWorkingPrimaries: return "convertToWorkingPrimaries";
        case StageKind::ToneMapLuminance: return "toneMapLuminance";
        case StageKind::ConvertFromWorkingPrimaries: return "convertFromWorkingPrimaries";
        case StageKind::EncodeTransfer: return "encodeTransfer";
        case StageKind::AssociateAlpha: return "associateAlpha";
        case StageKind::ViewTransform: return "viewTransform";
    }
    return {};
}

struct Extent
{
    std::uint32_t width = 0;
    std::uint32_t height = 0;

    constexpr bool operator==(Extent other) const noexcept
    {
        return width == other.width && height == other.height;
    }

    constexpr bool operator!=(Extent other) const noexcept { return ! (*this == other); }
};

struct ImageDescriptor
{
    Extent extent;
    PixelFormat format = PixelFormat::Unspecified;
    ColorSpace colorSpace = ColorSpace::Unspecified;
    TransferFunction transfer = TransferFunction::Unspecified;
    AlphaMode alpha = AlphaMode::Unspecified;
};

struct LuminanceParameters
{
    double sourceReferenceWhiteNits = 100.0;
    double sourcePeakNits = 100.0;
    double outputReferenceWhiteNits = 100.0;
    double outputPeakNits = 100.0;
};

struct Description
{
    std::uint32_t version = kWireVersion;
    ImageDescriptor input;
    ImageDescriptor output;
    ColorSpace workingColorSpace = ColorSpace::LinearSRGB;
    PixelFormat workingFormat = PixelFormat::RGBA16F;
    OutputIntent outputIntent = OutputIntent::SdrDisplay;
    ToneMap toneMap = ToneMap::None;
    LuminanceParameters luminance;
    // Opt-in source interpretation; old transforms retain inverse-OETF-only HLG.
    bool hlgReferenceDisplayOotf = false;
};

struct AdmissionLimits
{
    std::uint32_t dimension = kMaximumDimension;
    std::uint64_t pixels = kMaximumPixels;
    std::uint64_t residentBytes = kMaximumResidentBytes;
    std::size_t stages = kMaximumStages;
};

struct ResourceFootprint
{
    std::uint64_t pixelCount = 0;
    std::uint64_t inputBytes = 0;
    std::uint64_t outputBytes = 0;
    std::uint64_t workingBytes = 0;
    std::uint64_t peakResidentBytes = 0;
    std::uint32_t intermediateImages = 0;
    std::uint32_t stageCount = 0;
};

enum class AdmissionFailure : std::uint8_t
{
    None = 0,
    UnsupportedVersion,
    InvalidLimits,
    NativeGpuRequired,
    InvalidExtent,
    ExtentMismatch,
    UnknownPixelFormat,
    UnknownColorSpace,
    UnknownTransferFunction,
    UnknownAlphaMode,
    UnsupportedPixelFormat,
    InvalidColorTransferPair,
    InvalidWorkingSpace,
    InvalidWorkingFormat,
    UnknownOutputIntent,
    UnknownToneMap,
    NonFiniteLuminance,
    LuminanceOutOfRange,
    InvalidOutputIntent,
    AlphaWouldBeDiscarded,
    ToneMapRequired,
    ToneMapNotAllowed,
    StageLimitExceeded,
    ByteCountOverflow,
    ResidentByteLimitExceeded,
    MissingHlgSourcePolicy
};

inline constexpr std::string_view token(AdmissionFailure value) noexcept
{
    switch (value)
    {
        case AdmissionFailure::None: return "none";
        case AdmissionFailure::MissingHlgSourcePolicy: return "missingHlgSourcePolicy";
        case AdmissionFailure::UnsupportedVersion: return "unsupportedVersion";
        case AdmissionFailure::InvalidLimits: return "invalidLimits";
        case AdmissionFailure::NativeGpuRequired: return "nativeGpuRequired";
        case AdmissionFailure::InvalidExtent: return "invalidExtent";
        case AdmissionFailure::ExtentMismatch: return "extentMismatch";
        case AdmissionFailure::UnknownPixelFormat: return "unknownPixelFormat";
        case AdmissionFailure::UnknownColorSpace: return "unknownColorSpace";
        case AdmissionFailure::UnknownTransferFunction: return "unknownTransferFunction";
        case AdmissionFailure::UnknownAlphaMode: return "unknownAlphaMode";
        case AdmissionFailure::UnsupportedPixelFormat: return "unsupportedPixelFormat";
        case AdmissionFailure::InvalidColorTransferPair: return "invalidColorTransferPair";
        case AdmissionFailure::InvalidWorkingSpace: return "invalidWorkingSpace";
        case AdmissionFailure::InvalidWorkingFormat: return "invalidWorkingFormat";
        case AdmissionFailure::UnknownOutputIntent: return "unknownOutputIntent";
        case AdmissionFailure::UnknownToneMap: return "unknownToneMap";
        case AdmissionFailure::NonFiniteLuminance: return "nonFiniteLuminance";
        case AdmissionFailure::LuminanceOutOfRange: return "luminanceOutOfRange";
        case AdmissionFailure::InvalidOutputIntent: return "invalidOutputIntent";
        case AdmissionFailure::AlphaWouldBeDiscarded: return "alphaWouldBeDiscarded";
        case AdmissionFailure::ToneMapRequired: return "toneMapRequired";
        case AdmissionFailure::ToneMapNotAllowed: return "toneMapNotAllowed";
        case AdmissionFailure::StageLimitExceeded: return "stageLimitExceeded";
        case AdmissionFailure::ByteCountOverflow: return "byteCountOverflow";
        case AdmissionFailure::ResidentByteLimitExceeded: return "residentByteLimitExceeded";
    }
    return {};
}

struct CacheIdentity
{
    std::array<std::uint8_t, 64> bytes {};
    std::uint8_t size = 0;

    bool operator==(const CacheIdentity& other) const noexcept
    {
        return size == other.size && bytes == other.bytes;
    }

    bool operator!=(const CacheIdentity& other) const noexcept { return ! (*this == other); }

    // This is a lookup accelerator only. Cache equality uses the complete identity bytes.
    std::uint64_t fingerprint() const noexcept
    {
        std::uint64_t value = 1469598103934665603ull;
        for (std::size_t index = 0; index < size; ++index)
        {
            value ^= bytes[index];
            value *= 1099511628211ull;
        }
        return value;
    }
};

class AdmittedTransform final
{
public:
    AdmittedTransform(const AdmittedTransform&) = default;
    AdmittedTransform(AdmittedTransform&&) = default;
    AdmittedTransform& operator=(const AdmittedTransform&) = delete;
    AdmittedTransform& operator=(AdmittedTransform&&) = delete;

    const Description& description() const noexcept { return description_; }
    BackendCapability backend() const noexcept { return BackendCapability::NativeGpu; }
    const std::array<StageKind, kMaximumStages>& stages() const noexcept { return stages_; }
    std::size_t stageCount() const noexcept { return stageCount_; }
    const ResourceFootprint& resources() const noexcept { return resources_; }
    const CacheIdentity& cacheIdentity() const noexcept { return cacheIdentity_; }

    static constexpr bool allowsCpuProductionFallback = false;
    static constexpr bool zeroAlphaForcesZeroRgb = true;

private:
    friend std::optional<AdmittedTransform> admit(const Description&,
                                                   BackendCapability,
                                                   AdmissionFailure&,
                                                   const AdmissionLimits&);

    AdmittedTransform(Description description,
                      std::array<StageKind, kMaximumStages> stages,
                      std::size_t stageCount,
                      ResourceFootprint resources,
                      CacheIdentity cacheIdentity)
        : description_(description),
          stages_(stages),
          stageCount_(stageCount),
          resources_(resources),
          cacheIdentity_(cacheIdentity)
    {
    }

    const Description description_;
    const std::array<StageKind, kMaximumStages> stages_;
    const std::size_t stageCount_;
    const ResourceFootprint resources_;
    const CacheIdentity cacheIdentity_;
};

namespace detail
{
inline constexpr bool known(PixelFormat value) noexcept
{
    return value >= PixelFormat::R8 && value <= PixelFormat::RGBA32F;
}

inline constexpr bool known(ColorSpace value) noexcept
{
    return value >= ColorSpace::LinearSRGB && value <= ColorSpace::Rec2020;
}

inline constexpr bool known(TransferFunction value) noexcept
{
    return value >= TransferFunction::Linear && value <= TransferFunction::Rec709;
}

inline constexpr bool known(AlphaMode value) noexcept
{
    return value >= AlphaMode::Opaque && value <= AlphaMode::Premultiplied;
}

inline constexpr bool known(OutputIntent value) noexcept
{
    return value >= OutputIntent::SceneLinear && value <= OutputIntent::HdrHlg;
}

inline constexpr bool known(ToneMap value) noexcept
{
    return value >= ToneMap::None && value <= ToneMap::AgX;
}

inline constexpr std::uint8_t bytesPerPixel(PixelFormat value) noexcept
{
    switch (value)
    {
        case PixelFormat::R8: return 1;
        case PixelFormat::R16: return 2;
        case PixelFormat::RGBA8: return 4;
        case PixelFormat::RGBA16F: return 8;
        case PixelFormat::RGBA32F: return 16;
        case PixelFormat::Unspecified: break;
    }
    return 0;
}

inline constexpr bool isRgba(PixelFormat value) noexcept
{
    return value == PixelFormat::RGBA8
        || value == PixelFormat::RGBA16F
        || value == PixelFormat::RGBA32F;
}

inline constexpr bool isFloatRgba(PixelFormat value) noexcept
{
    return value == PixelFormat::RGBA16F || value == PixelFormat::RGBA32F;
}

inline constexpr ColorSpace primaries(ColorSpace value) noexcept
{
    return value == ColorSpace::LinearSRGB || value == ColorSpace::SRGB
        ? ColorSpace::SRGB : value;
}

inline constexpr bool validColorTransfer(ColorSpace colorSpace,
                                         TransferFunction transfer) noexcept
{
    switch (colorSpace)
    {
        case ColorSpace::LinearSRGB: return transfer == TransferFunction::Linear;
        case ColorSpace::SRGB: return transfer == TransferFunction::SRGB;
        case ColorSpace::DisplayP3:
            return transfer == TransferFunction::SRGB || transfer == TransferFunction::Linear;
        case ColorSpace::Rec709: return transfer == TransferFunction::Gamma22 || transfer == TransferFunction::Rec709;
        case ColorSpace::Rec2020:
            // Rec.709 here is the BT.2020 SDR encoding, which reuses the BT.709 transfer.
            return transfer == TransferFunction::Linear
                || transfer == TransferFunction::PQ
                || transfer == TransferFunction::HLG
                || transfer == TransferFunction::Rec709;
        case ColorSpace::Unspecified: break;
    }
    return false;
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
    return limits.dimension <= kMaximumDimension
        && limits.pixels <= kMaximumPixels
        && limits.residentBytes <= kMaximumResidentBytes
        && limits.stages <= kMaximumStages;
}

inline bool validLuminance(const LuminanceParameters& value,
                           AdmissionFailure& failure) noexcept
{
    if (! std::isfinite(value.sourceReferenceWhiteNits)
        || ! std::isfinite(value.sourcePeakNits)
        || ! std::isfinite(value.outputReferenceWhiteNits)
        || ! std::isfinite(value.outputPeakNits))
    {
        failure = AdmissionFailure::NonFiniteLuminance;
        return false;
    }

    const auto validReference = [] (double nits) {
        return nits >= kMinimumReferenceWhiteNits
            && nits <= kMaximumReferenceWhiteNits;
    };
    if (! validReference(value.sourceReferenceWhiteNits)
        || ! validReference(value.outputReferenceWhiteNits)
        || value.sourcePeakNits < value.sourceReferenceWhiteNits
        || value.outputPeakNits < value.outputReferenceWhiteNits
        || value.sourcePeakNits > kMaximumPeakNits
        || value.outputPeakNits > kMaximumPeakNits)
    {
        failure = AdmissionFailure::LuminanceOutOfRange;
        return false;
    }
    return true;
}

inline bool validIntent(const Description& value) noexcept
{
    const auto& output = value.output;
    switch (value.outputIntent)
    {
        case OutputIntent::SceneLinear:
            return isFloatRgba(output.format)
                && output.transfer == TransferFunction::Linear
                && (output.colorSpace == ColorSpace::LinearSRGB
                    || output.colorSpace == ColorSpace::DisplayP3
                    || output.colorSpace == ColorSpace::Rec2020);
        case OutputIntent::SdrDisplay:
            // Pairs are already checked by validColorTransfer; this selects the SDR subset.
            return output.format == PixelFormat::RGBA8
                && (((output.colorSpace == ColorSpace::SRGB
                        || output.colorSpace == ColorSpace::DisplayP3)
                        && output.transfer == TransferFunction::SRGB)
                    || (output.colorSpace == ColorSpace::Rec709
                        && (output.transfer == TransferFunction::Gamma22
                            || output.transfer == TransferFunction::Rec709))
                    || (output.colorSpace == ColorSpace::Rec2020
                        && output.transfer == TransferFunction::Rec709))
                && value.luminance.outputPeakNits <= kMaximumReferenceWhiteNits;
        case OutputIntent::HdrPq:
            return isFloatRgba(output.format)
                && output.colorSpace == ColorSpace::Rec2020
                && output.transfer == TransferFunction::PQ
                && output.alpha == AlphaMode::Opaque
                && value.luminance.outputPeakNits >= kMinimumHdrPeakNits;
        case OutputIntent::HdrHlg:
            return isFloatRgba(output.format)
                && output.colorSpace == ColorSpace::Rec2020
                && output.transfer == TransferFunction::HLG
                && output.alpha == AlphaMode::Opaque
                && value.luminance.outputPeakNits >= kMinimumHdrPeakNits;
    }
    return false;
}

inline std::uint64_t doubleBits(double value) noexcept
{
    if (value == 0.0)
        value = 0.0;
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

struct IdentityWriter
{
    CacheIdentity identity;

    void appendByte(std::uint8_t value) noexcept
    {
        identity.bytes[identity.size++] = value;
    }

    void appendU32(std::uint32_t value) noexcept
    {
        for (int shift = 24; shift >= 0; shift -= 8)
            appendByte(static_cast<std::uint8_t>(value >> shift));
    }

    void appendU64(std::uint64_t value) noexcept
    {
        for (int shift = 56; shift >= 0; shift -= 8)
            appendByte(static_cast<std::uint8_t>(value >> shift));
    }
};

inline CacheIdentity makeIdentity(const Description& value) noexcept
{
    IdentityWriter writer;
    writer.appendU32(kWireVersion);
    writer.appendU32(kColorMathVersion);
    writer.appendU32(value.input.extent.width);
    writer.appendU32(value.input.extent.height);
    writer.appendByte(static_cast<std::uint8_t>(value.input.format));
    writer.appendByte(static_cast<std::uint8_t>(value.input.colorSpace));
    writer.appendByte(static_cast<std::uint8_t>(value.input.transfer));
    writer.appendByte(static_cast<std::uint8_t>(value.input.alpha));
    writer.appendByte(static_cast<std::uint8_t>(value.output.format));
    writer.appendByte(static_cast<std::uint8_t>(value.output.colorSpace));
    writer.appendByte(static_cast<std::uint8_t>(value.output.transfer));
    writer.appendByte(static_cast<std::uint8_t>(value.output.alpha));
    writer.appendByte(static_cast<std::uint8_t>(value.workingColorSpace));
    writer.appendByte(static_cast<std::uint8_t>(value.workingFormat));
    writer.appendByte(static_cast<std::uint8_t>(value.outputIntent));
    writer.appendByte(static_cast<std::uint8_t>(value.toneMap));
    writer.appendU64(doubleBits(value.luminance.sourceReferenceWhiteNits));
    writer.appendU64(doubleBits(value.luminance.sourcePeakNits));
    writer.appendU64(doubleBits(value.luminance.outputReferenceWhiteNits));
    writer.appendU64(doubleBits(value.luminance.outputPeakNits));
    // Preserve existing identities, including PQ, byte for byte.
    if (value.hlgReferenceDisplayOotf) writer.appendByte(1);
    return writer.identity;
}
} // namespace detail

inline std::optional<AdmittedTransform> admit(
    const Description& untrusted,
    BackendCapability backend,
    AdmissionFailure& failure,
    const AdmissionLimits& limits = {})
{
    failure = AdmissionFailure::None;
    const auto reject = [&failure] (AdmissionFailure reason)
        -> std::optional<AdmittedTransform>
    {
        failure = reason;
        return std::nullopt;
    };

    if (untrusted.version != kWireVersion)
        return reject(AdmissionFailure::UnsupportedVersion);
    if (! detail::limitsWithinHardBounds(limits))
        return reject(AdmissionFailure::InvalidLimits);
    if (backend != BackendCapability::NativeGpu)
        return reject(AdmissionFailure::NativeGpuRequired);

    const auto& input = untrusted.input;
    const auto& output = untrusted.output;
    if (input.extent.width == 0 || input.extent.height == 0
        || input.extent.width > limits.dimension
        || input.extent.height > limits.dimension)
        return reject(AdmissionFailure::InvalidExtent);
    if (output.extent != input.extent)
        return reject(AdmissionFailure::ExtentMismatch);

    if (! detail::known(input.format) || ! detail::known(output.format)
        || ! detail::known(untrusted.workingFormat))
        return reject(AdmissionFailure::UnknownPixelFormat);
    if (! detail::known(input.colorSpace) || ! detail::known(output.colorSpace)
        || ! detail::known(untrusted.workingColorSpace))
        return reject(AdmissionFailure::UnknownColorSpace);
    if (! detail::known(input.transfer) || ! detail::known(output.transfer))
        return reject(AdmissionFailure::UnknownTransferFunction);
    if (! detail::known(input.alpha) || ! detail::known(output.alpha))
        return reject(AdmissionFailure::UnknownAlphaMode);
    if (! detail::known(untrusted.outputIntent))
        return reject(AdmissionFailure::UnknownOutputIntent);
    if (! detail::known(untrusted.toneMap))
        return reject(AdmissionFailure::UnknownToneMap);

    if (! detail::isRgba(input.format) || ! detail::isRgba(output.format))
        return reject(AdmissionFailure::UnsupportedPixelFormat);
    if (! detail::validColorTransfer(input.colorSpace, input.transfer)
        || ! detail::validColorTransfer(output.colorSpace, output.transfer))
        return reject(AdmissionFailure::InvalidColorTransferPair);
    if (untrusted.workingColorSpace != ColorSpace::LinearSRGB
        && untrusted.workingColorSpace != ColorSpace::Rec2020)
        return reject(AdmissionFailure::InvalidWorkingSpace);
    if (! detail::isFloatRgba(untrusted.workingFormat))
        return reject(AdmissionFailure::InvalidWorkingFormat);
    if (isHdrTransfer(input.transfer) && ! detail::isFloatRgba(input.format))
        return reject(AdmissionFailure::UnsupportedPixelFormat);

    AdmissionFailure luminanceFailure = AdmissionFailure::None;
    if (! detail::validLuminance(untrusted.luminance, luminanceFailure))
        return reject(luminanceFailure);
    if (untrusted.hlgReferenceDisplayOotf
        && (input.transfer != TransferFunction::HLG || input.colorSpace != ColorSpace::Rec2020
            || untrusted.luminance.sourcePeakNits < kMinimumHdrPeakNits))
        return reject(AdmissionFailure::InvalidColorTransferPair);
    if (! detail::validIntent(untrusted))
        return reject(AdmissionFailure::InvalidOutputIntent);
    if (output.alpha == AlphaMode::Opaque && input.alpha != AlphaMode::Opaque)
        return reject(AdmissionFailure::AlphaWouldBeDiscarded);

    const bool needsToneMap = untrusted.outputIntent == OutputIntent::SdrDisplay
        && untrusted.luminance.sourcePeakNits > untrusted.luminance.outputPeakNits;
    // AgX takes scene-linear sRGB to SDR display and sets its own white, so it
    // answers to no luminance pair and to no other input.
    if (untrusted.toneMap == ToneMap::AgX)
    {
        if (untrusted.outputIntent != OutputIntent::SdrDisplay || needsToneMap
            || input.transfer != TransferFunction::Linear || input.colorSpace != ColorSpace::LinearSRGB
            || untrusted.workingColorSpace != ColorSpace::LinearSRGB)
            return reject(AdmissionFailure::ToneMapNotAllowed);
    }
    else
    {
        if (needsToneMap && untrusted.toneMap == ToneMap::None)
            return reject(AdmissionFailure::ToneMapRequired);
        if ((! needsToneMap || untrusted.outputIntent != OutputIntent::SdrDisplay)
            && untrusted.toneMap != ToneMap::None)
            return reject(AdmissionFailure::ToneMapNotAllowed);
    }

    std::array<StageKind, kMaximumStages> stages {};
    std::size_t stageCount = 0;
    const auto appendStage = [&] (StageKind stage) {
        stages[stageCount++] = stage;
    };

    if (input.alpha == AlphaMode::Premultiplied)
        appendStage(StageKind::UnassociateAlpha);
    if (input.transfer != TransferFunction::Linear)
        appendStage(StageKind::DecodeTransfer);
    if (detail::primaries(input.colorSpace) != detail::primaries(untrusted.workingColorSpace))
        appendStage(StageKind::ConvertToWorkingPrimaries);
    if (untrusted.toneMap == ToneMap::AgX)
        appendStage(StageKind::ViewTransform);
    else if (untrusted.toneMap != ToneMap::None)
        appendStage(StageKind::ToneMapLuminance);
    if (detail::primaries(untrusted.workingColorSpace) != detail::primaries(output.colorSpace))
        appendStage(StageKind::ConvertFromWorkingPrimaries);
    if (output.transfer != TransferFunction::Linear)
        appendStage(StageKind::EncodeTransfer);
    if (output.alpha == AlphaMode::Premultiplied)
        appendStage(StageKind::AssociateAlpha);

    if (stageCount > limits.stages)
        return reject(AdmissionFailure::StageLimitExceeded);

    ResourceFootprint resources;
    if (! detail::checkedMultiply(input.extent.width, input.extent.height,
                                  resources.pixelCount)
        || resources.pixelCount > limits.pixels
        || resources.pixelCount > kMaximumPixels)
        return reject(AdmissionFailure::InvalidExtent);
    if (! detail::checkedMultiply(resources.pixelCount,
                                  detail::bytesPerPixel(input.format),
                                  resources.inputBytes)
        || ! detail::checkedMultiply(resources.pixelCount,
                                     detail::bytesPerPixel(output.format),
                                     resources.outputBytes)
        || ! detail::checkedMultiply(resources.pixelCount,
                                     detail::bytesPerPixel(untrusted.workingFormat),
                                     resources.workingBytes))
        return reject(AdmissionFailure::ByteCountOverflow);

    resources.intermediateImages = stageCount == 0 ? 0u : 1u;
    resources.stageCount = static_cast<std::uint32_t>(stageCount);
    std::uint64_t resident = 0;
    if (! detail::checkedAdd(resources.inputBytes, resources.outputBytes, resident)
        || (resources.intermediateImages != 0
            && ! detail::checkedAdd(resident, resources.workingBytes, resident)))
        return reject(AdmissionFailure::ByteCountOverflow);
    resources.peakResidentBytes = resident;
    if (resources.peakResidentBytes > limits.residentBytes)
        return reject(AdmissionFailure::ResidentByteLimitExceeded);

    return AdmittedTransform(untrusted, stages, stageCount, resources,
                             detail::makeIdentity(untrusted));
}

static_assert(! std::is_default_constructible_v<AdmittedTransform>);
static_assert(! std::is_copy_assignable_v<AdmittedTransform>);
static_assert(! AdmittedTransform::allowsCpuProductionFallback);
static_assert(AdmittedTransform::zeroAlphaForcesZeroRgb);
static_assert(static_cast<std::uint8_t>(PixelFormat::RGBA8) == 3);
static_assert(static_cast<std::uint8_t>(ColorSpace::Rec2020) == 5);
static_assert(static_cast<std::uint8_t>(TransferFunction::HLG) == 5);
static_assert(static_cast<std::uint8_t>(AlphaMode::Premultiplied) == 3);
} // namespace colortransform
