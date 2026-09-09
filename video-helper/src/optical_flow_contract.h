#pragma once

#include "../../shared/RenderPassOutputContract.h"
#include "sha256.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace videoopticalflow
{
inline constexpr std::uint32_t kWireVersion = 1;
inline constexpr std::uint32_t kMaximumWidth = 16384;
inline constexpr std::uint32_t kMaximumHeight = 16384;
inline constexpr std::uint64_t kMaximumPixels = 8192ull * 8192ull;
inline constexpr std::uint64_t kMaximumResidentBytes = 2ull * 1024ull * 1024ull * 1024ull;
inline constexpr bool kAllowsCpuProductionFallback = false;

using StreamIdentity = std::array<std::uint8_t, 16>;
using ContentIdentity = std::array<std::uint8_t, 32>;
using BackendIdentity = std::array<std::uint8_t, 16>;
using ResourceIdentity = std::array<std::uint8_t, 16>;

// These fixed-width receipts are supplied by the frame owner. Admission never
// resolves a path, reads frame bytes, or consults a mutable external cache.
struct FrameIdentity
{
    StreamIdentity stream {};
    ContentIdentity content {};
    std::uint64_t frameIndex = 0;
};

struct Timestamp
{
    std::int64_t ticks = 0;
    std::uint32_t timescale = 0;
};

struct FrameDescription
{
    FrameIdentity identity;
    Timestamp timestamp;
    renderpassoutput::Extent extent;
    renderpassoutput::PixelFormat format = renderpassoutput::PixelFormat::RGBA16Float;
    renderpassoutput::ColorSpace colorSpace = renderpassoutput::ColorSpace::LinearSRGB;
};

enum class MotionVectorUnits : std::uint8_t
{
    PixelDisplacement = 1
};

enum class MotionVectorDirection : std::uint8_t
{
    FirstToSecond = 1
};

enum class MotionVectorCoordinates : std::uint8_t
{
    ImageTopLeftPositiveRightDown = 1
};

struct MotionVectorOutputDescription
{
    renderpassoutput::Extent extent;
    renderpassoutput::PixelFormat format = renderpassoutput::PixelFormat::RG16Float;
    renderpassoutput::ColorSpace colorSpace = renderpassoutput::ColorSpace::Data;
    MotionVectorUnits units = MotionVectorUnits::PixelDisplacement;
    MotionVectorDirection direction = MotionVectorDirection::FirstToSecond;
    MotionVectorCoordinates coordinates = MotionVectorCoordinates::ImageTopLeftPositiveRightDown;
};

inline constexpr MotionVectorOutputDescription canonicalOutput(
    renderpassoutput::Extent extent) noexcept
{
    return { extent,
             renderpassoutput::PixelFormat::RG16Float,
             renderpassoutput::ColorSpace::Data,
             MotionVectorUnits::PixelDisplacement,
             MotionVectorDirection::FirstToSecond,
             MotionVectorCoordinates::ImageTopLeftPositiveRightDown };
}

struct Description
{
    std::uint32_t version = kWireVersion;
    FrameDescription first;
    FrameDescription second;
    MotionVectorOutputDescription output;
};

struct ResourceLimits
{
    std::uint32_t width = kMaximumWidth;
    std::uint32_t height = kMaximumHeight;
    std::uint64_t pixels = kMaximumPixels;
    std::uint64_t residentBytes = kMaximumResidentBytes;
};

enum class BackendKind : std::uint8_t
{
    Unavailable = 0,
    NativeGpu = 1,
    CpuReference = 2
};

struct BackendCapabilities
{
    BackendKind kind = BackendKind::Unavailable;
    bool supportsOpticalFlow = false;
    BackendIdentity implementation {};
    std::uint32_t implementationRevision = 0;
    ResourceLimits limits;
    std::uint32_t temporaryBytesPerPixel = 0;
    std::uint64_t fixedTemporaryBytes = 0;
};

struct ResourceFootprint
{
    std::uint64_t pixels = 0;
    std::uint64_t referencedInputBytes = 0;
    std::uint64_t outputBytes = 0;
    std::uint64_t temporaryBytes = 0;
    std::uint64_t residentBytes = 0;
};

enum class AdmissionFailure : std::uint8_t
{
    None = 0,
    UnsupportedVersion,
    InvalidFrameIdentity,
    DuplicateFrameIdentity,
    FrameStreamMismatch,
    FrameIndexOrderMismatch,
    InvalidTimestamp,
    TimestampScaleMismatch,
    TimestampOrderMismatch,
    InvalidInputDescriptor,
    InvalidExtent,
    FrameExtentMismatch,
    OutputDescriptorMismatch,
    InvalidBackendCapabilities,
    BackendUnavailable,
    NativeGpuRequired,
    OpticalFlowUnsupported,
    ByteCountOverflow,
    ByteLimitExceeded
};

inline constexpr std::string_view token(AdmissionFailure failure) noexcept
{
    switch (failure)
    {
        case AdmissionFailure::None: return "none";
        case AdmissionFailure::UnsupportedVersion: return "unsupportedVersion";
        case AdmissionFailure::InvalidFrameIdentity: return "invalidFrameIdentity";
        case AdmissionFailure::DuplicateFrameIdentity: return "duplicateFrameIdentity";
        case AdmissionFailure::FrameStreamMismatch: return "frameStreamMismatch";
        case AdmissionFailure::FrameIndexOrderMismatch: return "frameIndexOrderMismatch";
        case AdmissionFailure::InvalidTimestamp: return "invalidTimestamp";
        case AdmissionFailure::TimestampScaleMismatch: return "timestampScaleMismatch";
        case AdmissionFailure::TimestampOrderMismatch: return "timestampOrderMismatch";
        case AdmissionFailure::InvalidInputDescriptor: return "invalidInputDescriptor";
        case AdmissionFailure::InvalidExtent: return "invalidExtent";
        case AdmissionFailure::FrameExtentMismatch: return "frameExtentMismatch";
        case AdmissionFailure::OutputDescriptorMismatch: return "outputDescriptorMismatch";
        case AdmissionFailure::InvalidBackendCapabilities: return "invalidBackendCapabilities";
        case AdmissionFailure::BackendUnavailable: return "backendUnavailable";
        case AdmissionFailure::NativeGpuRequired: return "nativeGpuRequired";
        case AdmissionFailure::OpticalFlowUnsupported: return "opticalFlowUnsupported";
        case AdmissionFailure::ByteCountOverflow: return "byteCountOverflow";
        case AdmissionFailure::ByteLimitExceeded: return "byteLimitExceeded";
    }
    return {};
}

namespace detail
{
template <std::size_t Size>
inline bool anyNonZero(const std::array<std::uint8_t, Size>& value) noexcept
{
    for (const auto byte : value)
        if (byte != 0)
            return true;
    return false;
}

inline bool sameIdentity(const FrameIdentity& left, const FrameIdentity& right) noexcept
{
    return left.stream == right.stream
        && left.content == right.content
        && left.frameIndex == right.frameIndex;
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

inline constexpr bool limitsWithinHardBounds(const ResourceLimits& limits) noexcept
{
    return limits.width != 0
        && limits.height != 0
        && limits.pixels != 0
        && limits.residentBytes != 0
        && limits.width <= kMaximumWidth
        && limits.height <= kMaximumHeight
        && limits.pixels <= kMaximumPixels
        && limits.residentBytes <= kMaximumResidentBytes;
}

inline constexpr bool exactInputDescriptor(const FrameDescription& frame) noexcept
{
    return frame.format == renderpassoutput::PixelFormat::RGBA16Float
        && frame.colorSpace == renderpassoutput::ColorSpace::LinearSRGB;
}

inline constexpr bool exactOutputDescriptor(
    const MotionVectorOutputDescription& output,
    renderpassoutput::Extent extent) noexcept
{
    return output.extent == extent
        && output.format == renderpassoutput::PixelFormat::RG16Float
        && output.colorSpace == renderpassoutput::ColorSpace::Data
        && output.units == MotionVectorUnits::PixelDisplacement
        && output.direction == MotionVectorDirection::FirstToSecond
        && output.coordinates == MotionVectorCoordinates::ImageTopLeftPositiveRightDown;
}

class CacheKeyWriter final
{
public:
    CacheKeyWriter()
    {
        static constexpr char domain[] = "DonutStudio/OpticalFlowContract";
        hash_.update(domain, sizeof(domain) - 1);
    }

    template <std::size_t Size>
    void bytes(const std::array<std::uint8_t, Size>& value)
    {
        hash_.update(value.data(), value.size());
    }

    void u8(std::uint8_t value) { hash_.update(&value, 1); }

    void u32(std::uint32_t value)
    {
        std::uint8_t encoded[4];
        for (int index = 0; index < 4; ++index)
            encoded[index] = static_cast<std::uint8_t>(value >> (24 - index * 8));
        hash_.update(encoded, sizeof(encoded));
    }

    void u64(std::uint64_t value)
    {
        std::uint8_t encoded[8];
        for (int index = 0; index < 8; ++index)
            encoded[index] = static_cast<std::uint8_t>(value >> (56 - index * 8));
        hash_.update(encoded, sizeof(encoded));
    }

    void i64(std::int64_t value) { u64(static_cast<std::uint64_t>(value)); }

    std::string finish() { return hash_.finishHex(); }

private:
    videohelper::Sha256 hash_;
};

inline void addFrame(CacheKeyWriter& writer, const FrameDescription& frame)
{
    writer.bytes(frame.identity.stream);
    writer.bytes(frame.identity.content);
    writer.u64(frame.identity.frameIndex);
    writer.i64(frame.timestamp.ticks);
    writer.u32(frame.timestamp.timescale);
    writer.u32(frame.extent.width);
    writer.u32(frame.extent.height);
    writer.u8(static_cast<std::uint8_t>(frame.format));
    writer.u8(static_cast<std::uint8_t>(frame.colorSpace));
}

inline std::string makeCacheKey(const Description& description,
                                const BackendCapabilities& backend)
{
    // Fixed-order, big-endian field encoding makes the key independent of
    // structure padding, host endianness, locale, and standard-library hashes.
    // The implementation identity/revision prevents unsafe cross-algorithm hits.
    CacheKeyWriter writer;
    writer.u32(description.version);
    addFrame(writer, description.first);
    addFrame(writer, description.second);
    writer.u32(description.output.extent.width);
    writer.u32(description.output.extent.height);
    writer.u8(static_cast<std::uint8_t>(description.output.format));
    writer.u8(static_cast<std::uint8_t>(description.output.colorSpace));
    writer.u8(static_cast<std::uint8_t>(description.output.units));
    writer.u8(static_cast<std::uint8_t>(description.output.direction));
    writer.u8(static_cast<std::uint8_t>(description.output.coordinates));
    writer.bytes(backend.implementation);
    writer.u32(backend.implementationRevision);
    return writer.finish();
}
} // namespace detail

class AdmittedRequest final
{
public:
    AdmittedRequest(const AdmittedRequest&) = default;
    AdmittedRequest(AdmittedRequest&&) = default;
    AdmittedRequest& operator=(const AdmittedRequest&) = delete;
    AdmittedRequest& operator=(AdmittedRequest&&) = delete;

    const FrameDescription& first() const noexcept { return description_.first; }
    const FrameDescription& second() const noexcept { return description_.second; }
    const MotionVectorOutputDescription& output() const noexcept { return description_.output; }
    const ResourceFootprint& footprint() const noexcept { return footprint_; }
    const std::string& cacheKey() const noexcept { return cacheKey_; }
    const BackendIdentity& backendImplementation() const noexcept { return backendImplementation_; }
    std::uint32_t backendImplementationRevision() const noexcept
    {
        return backendImplementationRevision_;
    }

    static constexpr bool authorizesGraphCycles = false;

private:
    friend std::optional<AdmittedRequest> admit(
        const Description&, const BackendCapabilities&, AdmissionFailure&);

    AdmittedRequest(Description description,
                    ResourceFootprint footprint,
                    std::string cacheKey,
                    BackendIdentity backendImplementation,
                    std::uint32_t backendImplementationRevision)
        : description_(std::move(description)),
          footprint_(footprint),
          cacheKey_(std::move(cacheKey)),
          backendImplementation_(backendImplementation),
          backendImplementationRevision_(backendImplementationRevision)
    {
    }

    const Description description_;
    const ResourceFootprint footprint_;
    const std::string cacheKey_;
    const BackendIdentity backendImplementation_;
    const std::uint32_t backendImplementationRevision_;
};

// Production admission has no CPU overload or fallback path. A CPU reference
// can call the checked arithmetic helpers in tests, but it cannot create an
// AdmittedRequest because only BackendKind::NativeGpu is accepted below.
inline std::optional<AdmittedRequest> admit(const Description& untrusted,
                                            const BackendCapabilities& backend,
                                            AdmissionFailure& failure)
{
    failure = AdmissionFailure::None;
    const auto reject = [&failure](AdmissionFailure reason)
        -> std::optional<AdmittedRequest>
    {
        failure = reason;
        return std::nullopt;
    };

    if (untrusted.version != kWireVersion)
        return reject(AdmissionFailure::UnsupportedVersion);

    for (const auto* frame : { &untrusted.first, &untrusted.second })
    {
        if (!detail::anyNonZero(frame->identity.stream)
            || !detail::anyNonZero(frame->identity.content))
            return reject(AdmissionFailure::InvalidFrameIdentity);
        if (frame->timestamp.timescale == 0)
            return reject(AdmissionFailure::InvalidTimestamp);
        if (!detail::exactInputDescriptor(*frame))
            return reject(AdmissionFailure::InvalidInputDescriptor);
        if (frame->extent.width == 0 || frame->extent.height == 0)
            return reject(AdmissionFailure::InvalidExtent);
    }

    if (detail::sameIdentity(untrusted.first.identity, untrusted.second.identity))
        return reject(AdmissionFailure::DuplicateFrameIdentity);
    if (untrusted.first.identity.stream != untrusted.second.identity.stream)
        return reject(AdmissionFailure::FrameStreamMismatch);
    if (untrusted.second.identity.frameIndex <= untrusted.first.identity.frameIndex)
        return reject(AdmissionFailure::FrameIndexOrderMismatch);
    if (untrusted.first.timestamp.timescale != untrusted.second.timestamp.timescale)
        return reject(AdmissionFailure::TimestampScaleMismatch);
    if (untrusted.second.timestamp.ticks <= untrusted.first.timestamp.ticks)
        return reject(AdmissionFailure::TimestampOrderMismatch);
    if (untrusted.first.extent != untrusted.second.extent)
        return reject(AdmissionFailure::FrameExtentMismatch);
    if (!detail::exactOutputDescriptor(untrusted.output, untrusted.first.extent))
        return reject(AdmissionFailure::OutputDescriptorMismatch);

    const auto extent = untrusted.first.extent;
    if (extent.width > kMaximumWidth || extent.height > kMaximumHeight)
        return reject(AdmissionFailure::InvalidExtent);

    std::uint64_t pixels = 0;
    std::uint64_t oneInputBytes = 0;
    std::uint64_t referencedInputBytes = 0;
    std::uint64_t outputBytes = 0;
    std::uint64_t perPixelTemporaryBytes = 0;
    std::uint64_t temporaryBytes = 0;
    std::uint64_t residentBytes = 0;
    if (!detail::checkedMultiply(extent.width, extent.height, pixels)
        || !detail::checkedMultiply(
            pixels, renderpassoutput::bytesPerPixel(untrusted.first.format), oneInputBytes)
        || !detail::checkedMultiply(oneInputBytes, 2, referencedInputBytes)
        || !detail::checkedMultiply(
            pixels, renderpassoutput::bytesPerPixel(untrusted.output.format), outputBytes)
        || !detail::checkedMultiply(
            pixels, backend.temporaryBytesPerPixel, perPixelTemporaryBytes)
        || !detail::checkedAdd(
            perPixelTemporaryBytes, backend.fixedTemporaryBytes, temporaryBytes)
        || !detail::checkedAdd(referencedInputBytes, outputBytes, residentBytes)
        || !detail::checkedAdd(residentBytes, temporaryBytes, residentBytes))
        return reject(AdmissionFailure::ByteCountOverflow);

    if (pixels > kMaximumPixels || residentBytes > kMaximumResidentBytes)
        return reject(AdmissionFailure::ByteLimitExceeded);

    if (backend.kind == BackendKind::Unavailable)
        return reject(AdmissionFailure::BackendUnavailable);
    if (backend.kind != BackendKind::NativeGpu)
        return reject(AdmissionFailure::NativeGpuRequired);
    if (!backend.supportsOpticalFlow)
        return reject(AdmissionFailure::OpticalFlowUnsupported);
    if (!detail::anyNonZero(backend.implementation)
        || backend.implementationRevision == 0
        || !detail::limitsWithinHardBounds(backend.limits))
        return reject(AdmissionFailure::InvalidBackendCapabilities);
    if (extent.width > backend.limits.width
        || extent.height > backend.limits.height
        || pixels > backend.limits.pixels)
        return reject(AdmissionFailure::InvalidExtent);
    if (residentBytes > backend.limits.residentBytes)
        return reject(AdmissionFailure::ByteLimitExceeded);

    ResourceFootprint footprint;
    footprint.pixels = pixels;
    footprint.referencedInputBytes = referencedInputBytes;
    footprint.outputBytes = outputBytes;
    footprint.temporaryBytes = temporaryBytes;
    footprint.residentBytes = residentBytes;

    return AdmittedRequest(untrusted,
                           footprint,
                           detail::makeCacheKey(untrusted, backend),
                           backend.implementation,
                           backend.implementationRevision);
}

// The backend returns this receipt after allocation. Result admission checks the
// receipt against the request but never allocates, maps, or reads the resource.
struct OutputResourceDescription
{
    ResourceIdentity identity {};
    std::uint64_t helperGeneration = 0;
    MotionVectorOutputDescription descriptor;
    std::uint64_t byteCount = 0;
};

struct ResultDescription
{
    std::uint32_t version = kWireVersion;
    std::string requestCacheKey;
    BackendIdentity backendImplementation {};
    std::uint32_t backendImplementationRevision = 0;
    OutputResourceDescription motionVectors;
};

enum class ResultFailure : std::uint8_t
{
    None = 0,
    UnsupportedVersion,
    RequestIdentityMismatch,
    BackendIdentityMismatch,
    InvalidResourceIdentity,
    InvalidHelperGeneration,
    OutputDescriptorMismatch,
    OutputByteCountMismatch
};

inline constexpr std::string_view token(ResultFailure failure) noexcept
{
    switch (failure)
    {
        case ResultFailure::None: return "none";
        case ResultFailure::UnsupportedVersion: return "unsupportedVersion";
        case ResultFailure::RequestIdentityMismatch: return "requestIdentityMismatch";
        case ResultFailure::BackendIdentityMismatch: return "backendIdentityMismatch";
        case ResultFailure::InvalidResourceIdentity: return "invalidResourceIdentity";
        case ResultFailure::InvalidHelperGeneration: return "invalidHelperGeneration";
        case ResultFailure::OutputDescriptorMismatch: return "outputDescriptorMismatch";
        case ResultFailure::OutputByteCountMismatch: return "outputByteCountMismatch";
    }
    return {};
}

class AdmittedResult;

inline std::optional<AdmittedResult> admitResult(const AdmittedRequest&,
                                                 const ResultDescription&,
                                                 ResultFailure&);

class AdmittedResult final
{
public:
    AdmittedResult(const AdmittedResult&) = default;
    AdmittedResult(AdmittedResult&&) = default;
    AdmittedResult& operator=(const AdmittedResult&) = delete;
    AdmittedResult& operator=(AdmittedResult&&) = delete;

    const std::string& requestCacheKey() const noexcept { return requestCacheKey_; }
    const BackendIdentity& backendImplementation() const noexcept
    {
        return backendImplementation_;
    }
    std::uint32_t backendImplementationRevision() const noexcept
    {
        return backendImplementationRevision_;
    }
    const OutputResourceDescription& motionVectors() const noexcept
    {
        return motionVectors_;
    }

private:
    friend std::optional<AdmittedResult> admitResult(
        const AdmittedRequest&, const ResultDescription&, ResultFailure&);

    explicit AdmittedResult(const ResultDescription& description)
        : requestCacheKey_(description.requestCacheKey),
          backendImplementation_(description.backendImplementation),
          backendImplementationRevision_(description.backendImplementationRevision),
          motionVectors_(description.motionVectors)
    {
    }

    const std::string requestCacheKey_;
    const BackendIdentity backendImplementation_;
    const std::uint32_t backendImplementationRevision_;
    const OutputResourceDescription motionVectors_;
};

inline std::optional<AdmittedResult> admitResult(const AdmittedRequest& request,
                                                 const ResultDescription& untrusted,
                                                 ResultFailure& failure)
{
    failure = ResultFailure::None;
    const auto reject = [&failure](ResultFailure reason) -> std::optional<AdmittedResult>
    {
        failure = reason;
        return std::nullopt;
    };

    if (untrusted.version != kWireVersion)
        return reject(ResultFailure::UnsupportedVersion);
    if (untrusted.requestCacheKey != request.cacheKey())
        return reject(ResultFailure::RequestIdentityMismatch);
    if (untrusted.backendImplementation != request.backendImplementation()
        || untrusted.backendImplementationRevision
               != request.backendImplementationRevision())
        return reject(ResultFailure::BackendIdentityMismatch);
    if (!detail::anyNonZero(untrusted.motionVectors.identity))
        return reject(ResultFailure::InvalidResourceIdentity);
    if (untrusted.motionVectors.helperGeneration == 0)
        return reject(ResultFailure::InvalidHelperGeneration);
    if (!detail::exactOutputDescriptor(untrusted.motionVectors.descriptor,
                                       request.output().extent))
        return reject(ResultFailure::OutputDescriptorMismatch);
    if (untrusted.motionVectors.byteCount != request.footprint().outputBytes)
        return reject(ResultFailure::OutputByteCountMismatch);

    return AdmittedResult(untrusted);
}

enum class EvaluationMode : std::uint8_t
{
    Preview = 1,
    OfflineSequential = 2
};

struct EvaluationContext
{
    std::uint64_t structuralRevision = 0;
    std::uint64_t loopDiscontinuitySerial = 0;
    std::uint64_t helperGeneration = 0;
    EvaluationMode mode = EvaluationMode::Preview;
    bool paused = false;
};

enum class TransitionAction : std::uint8_t
{
    ResetAndDispatch = 1,
    Dispatch = 2,
    Hold = 3,
    Reject = 4
};

enum class ResetCause : std::uint8_t
{
    None = 0,
    FirstEvaluation,
    BackwardSeek,
    SourceDiscontinuity,
    LoopDiscontinuity,
    StructuralRevision,
    HelperRestart,
    EvaluationModeChange
};

enum class EvaluationFailure : std::uint8_t
{
    None = 0,
    InvalidContext,
    OfflineNonSequential
};

struct Transition
{
    TransitionAction action = TransitionAction::Reject;
    ResetCause resetCause = ResetCause::None;
    EvaluationFailure failure = EvaluationFailure::None;
};

// Each viewport lifetime and offline run owns a separate instance. A reset
// action requires the owner to release any prior result before dispatch.
class EvaluationState final
{
public:
    Transition evaluate(const AdmittedRequest& request,
                        const EvaluationContext& context)
    {
        if (context.structuralRevision == 0 || context.helperGeneration == 0
            || (context.mode != EvaluationMode::Preview
                && context.mode != EvaluationMode::OfflineSequential))
            return { TransitionAction::Reject, ResetCause::None,
                     EvaluationFailure::InvalidContext };

        if (context.mode == EvaluationMode::OfflineSequential
            && !offlinePairIsSequential(request))
            return { TransitionAction::Reject, ResetCause::None,
                     EvaluationFailure::OfflineNonSequential };

        if (!initialized_)
            return resetAndDispatch(request, context, ResetCause::FirstEvaluation);
        if (context.helperGeneration != helperGeneration_)
            return resetAndDispatch(request, context, ResetCause::HelperRestart);
        if (context.structuralRevision != structuralRevision_)
            return resetAndDispatch(request, context, ResetCause::StructuralRevision);
        if (context.loopDiscontinuitySerial != loopDiscontinuitySerial_)
            return resetAndDispatch(request, context, ResetCause::LoopDiscontinuity);
        if (context.mode != mode_)
            return resetAndDispatch(request, context, ResetCause::EvaluationModeChange);
        if (request.cacheKey() == requestCacheKey_)
            return { TransitionAction::Hold, ResetCause::None, EvaluationFailure::None };
        if (context.paused)
            return { TransitionAction::Hold, ResetCause::None, EvaluationFailure::None };

        const auto& first = request.first();
        const auto& second = request.second();
        if (first.identity.stream != second_.identity.stream)
            return resetAndDispatch(request, context, ResetCause::SourceDiscontinuity);
        if (second.identity.frameIndex < second_.identity.frameIndex)
            return resetAndDispatch(request, context, ResetCause::BackwardSeek);
        if (second.identity.frameIndex == second_.identity.frameIndex)
            return resetAndDispatch(request, context, ResetCause::SourceDiscontinuity);

        if (context.mode == EvaluationMode::OfflineSequential
            && (!detail::sameIdentity(first.identity, second_.identity)
                || first.timestamp.ticks != second_.timestamp.ticks
                || first.timestamp.timescale != second_.timestamp.timescale))
            return { TransitionAction::Reject, ResetCause::None,
                     EvaluationFailure::OfflineNonSequential };

        remember(request, context);
        return { TransitionAction::Dispatch, ResetCause::None, EvaluationFailure::None };
    }

    void reset() noexcept
    {
        initialized_ = false;
        requestCacheKey_.clear();
        second_ = {};
        structuralRevision_ = 0;
        loopDiscontinuitySerial_ = 0;
        helperGeneration_ = 0;
        mode_ = EvaluationMode::Preview;
    }

    bool initialized() const noexcept { return initialized_; }
    const std::string& requestCacheKey() const noexcept { return requestCacheKey_; }

private:
    static bool offlinePairIsSequential(const AdmittedRequest& request) noexcept
    {
        // Later offline requests must also start with the preceding request's
        // exact second frame. evaluate() checks that chain before mutation.
        return request.first().identity.frameIndex
                   != std::numeric_limits<std::uint64_t>::max()
            && request.second().identity.frameIndex
                   == request.first().identity.frameIndex + 1;
    }

    Transition resetAndDispatch(const AdmittedRequest& request,
                                const EvaluationContext& context,
                                ResetCause cause)
    {
        remember(request, context);
        return { TransitionAction::ResetAndDispatch, cause, EvaluationFailure::None };
    }

    void remember(const AdmittedRequest& request, const EvaluationContext& context)
    {
        initialized_ = true;
        requestCacheKey_ = request.cacheKey();
        second_ = request.second();
        structuralRevision_ = context.structuralRevision;
        loopDiscontinuitySerial_ = context.loopDiscontinuitySerial;
        helperGeneration_ = context.helperGeneration;
        mode_ = context.mode;
    }

    bool initialized_ = false;
    std::string requestCacheKey_;
    FrameDescription second_;
    std::uint64_t structuralRevision_ = 0;
    std::uint64_t loopDiscontinuitySerial_ = 0;
    std::uint64_t helperGeneration_ = 0;
    EvaluationMode mode_ = EvaluationMode::Preview;
};

static_assert(!kAllowsCpuProductionFallback);
static_assert(!AdmittedRequest::authorizesGraphCycles);
static_assert(!std::is_copy_assignable_v<AdmittedRequest>);
static_assert(!std::is_move_assignable_v<AdmittedRequest>);
static_assert(!std::is_copy_assignable_v<AdmittedResult>);
static_assert(!std::is_move_assignable_v<AdmittedResult>);
} // namespace videoopticalflow
