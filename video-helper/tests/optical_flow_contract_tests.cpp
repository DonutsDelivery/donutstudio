#include "../src/optical_flow_contract.h"
#include "../../shared/OpticalFlowOperationContract.h"
#include "../../shared/MotionViewOperationContract.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string_view>
#include <type_traits>
#include <utility>

using namespace videoopticalflow;

static_assert(std::is_copy_constructible_v<AdmittedRequest>);
static_assert(!std::is_copy_assignable_v<AdmittedRequest>);
static_assert(!std::is_move_assignable_v<AdmittedRequest>);
static_assert(!kAllowsCpuProductionFallback);
static_assert(!AdmittedRequest::authorizesGraphCycles);
static_assert(std::is_copy_constructible_v<AdmittedResult>);
static_assert(!std::is_copy_assignable_v<AdmittedResult>);
static_assert(!std::is_move_assignable_v<AdmittedResult>);
static_assert(std::is_same_v<
    decltype(std::declval<const AdmittedRequest&>().first()),
    const FrameDescription&>);
static_assert(std::is_same_v<
    decltype(std::declval<const AdmittedRequest&>().output()),
    const MotionVectorOutputDescription&>);

namespace
{
int failures = 0;

void check(bool condition, const char* message)
{
    if (!condition)
    {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

template <std::size_t Size>
std::array<std::uint8_t, Size> filled(std::uint8_t value)
{
    std::array<std::uint8_t, Size> result;
    result.fill(value);
    return result;
}

Description validDescription()
{
    Description description;
    description.first.identity.stream = filled<16>(0x11);
    description.first.identity.content = filled<32>(0x21);
    description.first.identity.frameIndex = 41;
    description.first.timestamp = { 1001, 24000 };
    description.first.extent = { 640, 360 };

    description.second.identity.stream = description.first.identity.stream;
    description.second.identity.content = filled<32>(0x31);
    description.second.identity.frameIndex = 42;
    description.second.timestamp = { 2002, 24000 };
    description.second.extent = description.first.extent;

    description.output = canonicalOutput(description.first.extent);
    return description;
}

BackendCapabilities validBackend()
{
    BackendCapabilities backend;
    backend.kind = BackendKind::NativeGpu;
    backend.supportsOpticalFlow = true;
    backend.implementation = filled<16>(0x7a);
    backend.implementationRevision = 3;
    backend.temporaryBytesPerPixel = 12;
    backend.fixedTemporaryBytes = 4096;
    return backend;
}

AdmittedRequest admittedRequest(const Description& description = validDescription())
{
    AdmissionFailure failure = AdmissionFailure::None;
    auto admitted = admit(description, validBackend(), failure);
    if (!admitted)
    {
        check(false, "request fixture admission succeeds");
        std::abort();
    }
    return *admitted;
}

Description pairDescription(std::uint64_t firstIndex,
                            std::uint8_t firstContent,
                            std::uint8_t secondContent,
                            std::uint8_t stream = 0x11)
{
    auto description = validDescription();
    description.first.identity.stream = filled<16>(stream);
    description.second.identity.stream = description.first.identity.stream;
    description.first.identity.content = filled<32>(firstContent);
    description.second.identity.content = filled<32>(secondContent);
    description.first.identity.frameIndex = firstIndex;
    description.second.identity.frameIndex = firstIndex + 1;
    description.first.timestamp = { static_cast<std::int64_t>(firstIndex * 1001), 24000 };
    description.second.timestamp = {
        static_cast<std::int64_t>((firstIndex + 1) * 1001), 24000
    };
    return description;
}

ResultDescription validResult(const AdmittedRequest& request)
{
    ResultDescription result;
    result.requestCacheKey = request.cacheKey();
    result.backendImplementation = request.backendImplementation();
    result.backendImplementationRevision = request.backendImplementationRevision();
    result.motionVectors.identity = filled<16>(0x91);
    result.motionVectors.helperGeneration = 7;
    result.motionVectors.descriptor = request.output();
    result.motionVectors.byteCount = request.footprint().outputBytes;
    return result;
}

void expectRejected(const Description& description,
                    const BackendCapabilities& backend,
                    AdmissionFailure expected,
                    const char* message)
{
    AdmissionFailure failure = AdmissionFailure::None;
    const auto result = admit(description, backend, failure);
    check(!result, message);
    check(failure == expected, "rejection reports the exact failed contract");
    check(!token(failure).empty(), "rejection has a stable diagnostic token");
}

void expectResultRejected(const AdmittedRequest& request,
                          const ResultDescription& result,
                          ResultFailure expected,
                          const char* message)
{
    ResultFailure failure = ResultFailure::None;
    const auto admitted = admitResult(request, result, failure);
    check(!admitted, message);
    check(failure == expected, "result rejection reports the exact failed contract");
    check(!token(failure).empty(), "result rejection has a stable diagnostic token");
}

void checkTransition(const Transition& transition,
                     TransitionAction action,
                     ResetCause resetCause,
                     EvaluationFailure failure,
                     const char* message)
{
    check(transition.action == action && transition.resetCause == resetCause
              && transition.failure == failure,
          message);
}

void testExactDescriptorAndFootprint()
{
    AdmissionFailure failure = AdmissionFailure::UnsupportedVersion;
    const auto result = admit(validDescription(), validBackend(), failure);
    check(result.has_value(), "valid bounded native-GPU request is admitted");
    check(failure == AdmissionFailure::None, "successful admission clears the failure");
    if (!result)
        return;

    check(result->output().format == renderpassoutput::PixelFormat::RG16Float,
          "motion vectors use exactly RG16Float");
    check(result->output().colorSpace == renderpassoutput::ColorSpace::Data,
          "motion vectors are data, not color");
    check(result->output().units == MotionVectorUnits::PixelDisplacement,
          "motion vectors are pixel displacement");
    check(result->output().direction == MotionVectorDirection::FirstToSecond,
          "motion vectors point from first frame to second frame");
    check(result->output().coordinates
              == MotionVectorCoordinates::ImageTopLeftPositiveRightDown,
          "motion-vector coordinate axes are exact");

    auto changedOriginal = validDescription();
    AdmissionFailure immutableFailure = AdmissionFailure::None;
    const auto immutable = admit(changedOriginal, validBackend(), immutableFailure);
    changedOriginal.first.timestamp.ticks = 9999;
    check(immutable && immutable->first().timestamp.ticks == 1001,
          "admission freezes frame identity and time inputs");

    const auto& footprint = result->footprint();
    check(footprint.pixels == 640ull * 360ull, "pixel count is exact");
    check(footprint.referencedInputBytes == 640ull * 360ull * 8ull * 2ull,
          "both immutable RGBA16F inputs are charged");
    check(footprint.outputBytes == 640ull * 360ull * 4ull,
          "RG16F output bytes are exact");
    check(footprint.temporaryBytes == 640ull * 360ull * 12ull + 4096ull,
          "backend temporary bytes are charged exactly");
    check(footprint.residentBytes
              == footprint.referencedInputBytes + footprint.outputBytes
                   + footprint.temporaryBytes,
          "resident-byte accounting is complete");
}

void testStableCacheKey()
{
    AdmissionFailure firstFailure = AdmissionFailure::None;
    AdmissionFailure secondFailure = AdmissionFailure::None;
    const auto first = admit(validDescription(), validBackend(), firstFailure);
    const auto second = admit(validDescription(), validBackend(), secondFailure);
    check(first && second, "stable-key fixture admits twice");
    if (!first || !second)
        return;

    check(first->cacheKey() == second->cacheKey(),
          "identical canonical requests produce identical keys");
    check(first->cacheKey().size() == 64, "cache key is a SHA-256 hex digest");
    check(first->cacheKey() == "d5af49e4a8232e9b7ae3e236d328d01437cbd71520afcfdf6ac1a8408d35692c",
          "cache-key canonical encoding remains stable");

    auto changedTimestamp = validDescription();
    ++changedTimestamp.second.timestamp.ticks;
    AdmissionFailure changedTimestampFailure = AdmissionFailure::None;
    const auto changedTimestampResult = admit(
        changedTimestamp, validBackend(), changedTimestampFailure);
    check(changedTimestampResult
              && changedTimestampResult->cacheKey() != first->cacheKey(),
          "timestamp identity participates in the key");

    auto changedContent = validDescription();
    changedContent.second.identity.content[0] ^= 1;
    AdmissionFailure changedContentFailure = AdmissionFailure::None;
    const auto changedContentResult = admit(changedContent,
                                            validBackend(),
                                            changedContentFailure);
    check(changedContentResult && changedContentResult->cacheKey() != first->cacheKey(),
          "content identity participates in the key");

    auto changedBackend = validBackend();
    ++changedBackend.implementationRevision;
    AdmissionFailure changedBackendFailure = AdmissionFailure::None;
    const auto changedBackendResult = admit(validDescription(),
                                            changedBackend,
                                            changedBackendFailure);
    check(changedBackendResult && changedBackendResult->cacheKey() != first->cacheKey(),
          "backend implementation revision participates in the key");
}

void testTimestampAndExtentMismatch()
{
    auto streamMismatch = validDescription();
    streamMismatch.second.identity.stream[0] ^= 1;
    expectRejected(streamMismatch,
                   validBackend(),
                   AdmissionFailure::FrameStreamMismatch,
                   "input frames from different streams are rejected");

    auto frameIndexMismatch = validDescription();
    frameIndexMismatch.second.identity.frameIndex
        = frameIndexMismatch.first.identity.frameIndex;
    frameIndexMismatch.second.identity.content[0] ^= 1;
    expectRejected(frameIndexMismatch,
                   validBackend(),
                   AdmissionFailure::FrameIndexOrderMismatch,
                   "non-forward frame identity is rejected");

    auto timescaleMismatch = validDescription();
    timescaleMismatch.second.timestamp.timescale = 48000;
    expectRejected(timescaleMismatch,
                   validBackend(),
                   AdmissionFailure::TimestampScaleMismatch,
                   "mixed timestamp scales are rejected");

    auto reversedTime = validDescription();
    reversedTime.second.timestamp.ticks = reversedTime.first.timestamp.ticks;
    expectRejected(reversedTime,
                   validBackend(),
                   AdmissionFailure::TimestampOrderMismatch,
                   "non-forward frame timestamps are rejected");

    auto frameExtentMismatch = validDescription();
    ++frameExtentMismatch.second.extent.width;
    expectRejected(frameExtentMismatch,
                   validBackend(),
                   AdmissionFailure::FrameExtentMismatch,
                   "frame extent mismatch is rejected");

    auto outputExtentMismatch = validDescription();
    --outputExtentMismatch.output.extent.height;
    expectRejected(outputExtentMismatch,
                   validBackend(),
                   AdmissionFailure::OutputDescriptorMismatch,
                   "output extent mismatch is rejected");

    auto outputFormatMismatch = validDescription();
    outputFormatMismatch.output.format = renderpassoutput::PixelFormat::RGBA16Float;
    expectRejected(outputFormatMismatch,
                   validBackend(),
                   AdmissionFailure::OutputDescriptorMismatch,
                   "noncanonical output format is rejected");
}

void testDistinctClipSourcePairV2()
{
    SourcePairDescriptionV2 pair;
    pair.graphClipId = 44;
    pair.structuralRevision = 9;
    pair.first.clipId = 44;
    pair.first.sourceContent = filled<32>(0x41);
    pair.first.sourceGeneration = 7;
    pair.first.frame = validDescription().first;
    pair.first.frame.identity.frameIndex = 10;
    pair.first.frame.timestamp = { 9000, 48000 };
    pair.second.clipId = 88;
    pair.second.sourceContent = filled<32>(0x51);
    pair.second.sourceGeneration = 8;
    pair.second.frame = validDescription().second;
    pair.second.frame.identity.stream = filled<16>(0x12);
    pair.second.frame.identity.frameIndex = 2;
    pair.second.frame.timestamp = { 100, 1000 };
    pair.output = canonicalOutput(pair.first.frame.extent);

    SourcePairFailureV2 failure = SourcePairFailureV2::InvalidFrameIdentity;
    const auto validated = validateSourcePairV2(pair, failure);
    check(validated && failure == SourcePairFailureV2::None,
          "v2 accepts ordered frames from distinct saved clips with independent PTS and ordinals");
    if (!validated) return;
    check(validated->selectionReceipt().size() == 64,
          "v2 selection has a fixed-width digest");
    AdmissionFailure nativeFailure = AdmissionFailure::None;
    const auto native = admit(*validated, validBackend(), nativeFailure);
    check(native && nativeFailure == AdmissionFailure::None
          && native->sourcePairV2()
          && native->sourcePairV2()->first.clipId == pair.first.clipId
          && native->sourcePairV2()->second.clipId == pair.second.clipId
          && native->first().identity.stream != native->second().identity.stream
          && native->second().identity.frameIndex < native->first().identity.frameIndex,
          "v2 native admission retains ordered distinct-clip receipts without v1 frame ordering");
    if (native)
    {
        const auto pixels = static_cast<std::uint64_t>(pair.first.frame.extent.width)
                          * pair.first.frame.extent.height;
        check(native->footprint().referencedInputBytes == 2 * pixels * 8
              && native->footprint().outputBytes == pixels * 4,
              "v2 native admission accounts for both linear input frames and RG16F output");
    }
    auto cpu = validBackend();
    cpu.kind = BackendKind::CpuReference;
    check(!admit(*validated, cpu, nativeFailure)
          && nativeFailure == AdmissionFailure::NativeGpuRequired,
          "v2 native admission never grants a CPU production fallback");
    auto tight = validBackend();
    tight.limits.residentBytes = 1;
    check(!admit(*validated, tight, nativeFailure)
          && nativeFailure == AdmissionFailure::ByteLimitExceeded,
          "v2 rejects insufficient native resident-byte limits");
    auto otherBackend = validBackend();
    otherBackend.implementationRevision++;
    const auto otherNative = admit(*validated, otherBackend, nativeFailure);
    check(native && otherNative && native->cacheKey() != otherNative->cacheKey(),
          "v2 native cache key binds backend implementation revision");
    pair.first.frame.timestamp.ticks++;
    check(validated->description().first.frame.timestamp.ticks == 9000,
          "v2 validation freezes the selected frame receipts");
    pair.first.frame.timestamp.ticks--;
    auto changed = pair;
    std::swap(changed.first, changed.second);
    const auto reversed = validateSourcePairV2(changed, failure);
    check(reversed && reversed->selectionReceipt() != validated->selectionReceipt(),
          "v2 selection receipt preserves authored source order");
    if (native && reversed)
    {
        const auto reversedNative = admit(*reversed, validBackend(), nativeFailure);
        check(reversedNative && reversedNative->cacheKey() != native->cacheKey(),
              "v2 native cache key preserves authored source order");
    }
    changed = pair;
    changed.second.sourceContent[0] ^= 1;
    const auto changedSource = validateSourcePairV2(changed, failure);
    check(changedSource && changedSource->selectionReceipt() != validated->selectionReceipt(),
          "v2 selection receipt includes source content");
    if (native && changedSource)
    {
        const auto changedNative = admit(*changedSource, validBackend(), nativeFailure);
        check(changedNative && changedNative->cacheKey() != native->cacheKey(),
              "v2 native cache key binds immutable source content");
    }
    changed = pair;
    changed.second.frame.identity.frameIndex++;
    const auto changedOrdinal = validateSourcePairV2(changed, failure);
    check(changedOrdinal && changedOrdinal->selectionReceipt() != validated->selectionReceipt(),
          "v2 selection receipt includes the presentation ordinal");

    const auto rejected = [&] (const SourcePairDescriptionV2& candidate,
                              SourcePairFailureV2 expected, const char* message)
    {
        SourcePairFailureV2 reason = SourcePairFailureV2::None;
        check(!validateSourcePairV2(candidate, reason) && reason == expected, message);
        check(!token(reason).empty(), "v2 rejection has a diagnostic token");
    };
    changed = pair;
    changed.version = kWireVersion;
    rejected(changed, SourcePairFailureV2::UnsupportedVersion,
             "v2 validator rejects a v1 request version");
    auto legacy = validDescription();
    legacy.version = kSourcePairWireVersion;
    expectRejected(legacy, validBackend(), AdmissionFailure::UnsupportedVersion,
                   "v1 native admission does not grant v2 execution");
    changed = pair;
    changed.graphClipId = 99;
    rejected(changed, SourcePairFailureV2::InvalidSourceOwner,
             "v2 source pair must include its authored graph clip");
    changed = pair;
    changed.second.clipId = changed.first.clipId;
    rejected(changed, SourcePairFailureV2::DuplicateSourceOwner,
             "v2 rejects duplicate saved clip owners");
    changed = pair;
    changed.second.sourceGeneration = 0;
    rejected(changed, SourcePairFailureV2::InvalidSourceReceipt,
             "v2 rejects missing source generation");
    changed = pair;
    changed.second.sourceContent = {};
    rejected(changed, SourcePairFailureV2::InvalidSourceReceipt,
             "v2 rejects missing media content receipt");
    changed = pair;
    changed.second.frame.identity.content = {};
    rejected(changed, SourcePairFailureV2::InvalidFrameIdentity,
             "v2 rejects missing decoded pixel content receipt");
    changed = pair;
    changed.second.frame.identity.frameIndex = std::numeric_limits<std::uint64_t>::max();
    rejected(changed, SourcePairFailureV2::InvalidFrameIdentity,
             "v2 rejects an unknown presentation ordinal");
    changed = pair;
    changed.second.sourceContent = changed.first.sourceContent;
    changed.second.sourceGeneration = changed.first.sourceGeneration;
    changed.second.frame = changed.first.frame;
    const auto sameFrame = validateSourcePairV2(changed, failure);
    check(sameFrame && failure == SourcePairFailureV2::None
              && sameFrame->description().first.clipId != sameFrame->description().second.clipId,
          "v2 accepts distinct clip owners selecting the same underlying frame");
    changed = pair;
    changed.second.frame.timestamp.timescale = 0;
    rejected(changed, SourcePairFailureV2::InvalidTimestamp,
             "v2 rejects missing source time base");
    changed = pair;
    changed.second.frame.extent.width++;
    rejected(changed, SourcePairFailureV2::FrameExtentMismatch,
             "v2 rejects mismatched decoded frame extents");
    changed = pair;
    changed.second.frame.format = renderpassoutput::PixelFormat::R8Unorm;
    rejected(changed, SourcePairFailureV2::InvalidInputDescriptor,
             "v2 rejects a non-linear-float input format");
    changed = pair;
    changed.output.format = renderpassoutput::PixelFormat::RGBA16Float;
    rejected(changed, SourcePairFailureV2::OutputDescriptorMismatch,
             "v2 keeps the exact RG16F motion output descriptor");
}

void testByteOverflowAndLimits()
{
    auto overflowBackend = validBackend();
    overflowBackend.fixedTemporaryBytes = std::numeric_limits<std::uint64_t>::max();
    expectRejected(validDescription(),
                   overflowBackend,
                   AdmissionFailure::ByteCountOverflow,
                   "temporary-byte addition overflow is rejected");

    std::uint64_t product = 0;
    check(!detail::checkedMultiply(std::numeric_limits<std::uint64_t>::max(),
                                   2,
                                   product),
          "checked byte multiplication rejects overflow");

    AdmissionFailure baselineFailure = AdmissionFailure::None;
    const auto baseline = admit(validDescription(), validBackend(), baselineFailure);
    check(baseline.has_value(), "byte-limit baseline admits");
    if (baseline)
    {
        auto memoryLimitedBackend = validBackend();
        memoryLimitedBackend.limits.residentBytes = baseline->footprint().residentBytes - 1;
        expectRejected(validDescription(),
                       memoryLimitedBackend,
                       AdmissionFailure::ByteLimitExceeded,
                       "backend memory budget is enforced");
    }

    auto extentLimitedBackend = validBackend();
    extentLimitedBackend.limits.width = 639;
    expectRejected(validDescription(),
                   extentLimitedBackend,
                   AdmissionFailure::InvalidExtent,
                   "backend extent capability is enforced");

    auto invalidLimitsBackend = validBackend();
    invalidLimitsBackend.limits.pixels = 0;
    expectRejected(validDescription(),
                   invalidLimitsBackend,
                   AdmissionFailure::InvalidBackendCapabilities,
                   "zero backend capability ceilings are rejected");
}

void testUnavailableNativeGpuFailsClosed()
{
    expectRejected(validDescription(),
                   {},
                   AdmissionFailure::BackendUnavailable,
                   "missing GPU backend fails closed");

    auto cpuReference = validBackend();
    cpuReference.kind = BackendKind::CpuReference;
    expectRejected(validDescription(),
                   cpuReference,
                   AdmissionFailure::NativeGpuRequired,
                   "CPU reference cannot become a production fallback");

    auto unsupportedNativeGpu = validBackend();
    unsupportedNativeGpu.supportsOpticalFlow = false;
    expectRejected(validDescription(),
                   unsupportedNativeGpu,
                   AdmissionFailure::OpticalFlowUnsupported,
                   "native GPU without optical-flow support fails closed");
}

void testResultResourceReceipt()
{
    const auto request = admittedRequest();
    auto description = validResult(request);
    ResultFailure failure = ResultFailure::UnsupportedVersion;
    const auto result = admitResult(request, description, failure);
    check(result.has_value(), "exact native motion-vector resource receipt is admitted");
    check(failure == ResultFailure::None, "result admission clears stale diagnostics");
    if (!result)
        return;

    check(result->requestCacheKey() == request.cacheKey(),
          "result remains bound to the exact immutable request");
    check(result->motionVectors().descriptor.format
              == renderpassoutput::PixelFormat::RG16Float
              && result->motionVectors().descriptor.colorSpace
                   == renderpassoutput::ColorSpace::Data
              && result->motionVectors().byteCount == request.footprint().outputBytes,
          "result retains the exact motion resource descriptor and bytes");

    description.motionVectors.identity[0] ^= 1;
    check(result->motionVectors().identity[0] == 0x91,
          "admitted result owns its resource receipt");

    auto malformed = validResult(request);
    ++malformed.version;
    expectResultRejected(request, malformed, ResultFailure::UnsupportedVersion,
                         "unknown result version is rejected");

    malformed = validResult(request);
    malformed.requestCacheKey[0] ^= 1;
    expectResultRejected(request, malformed, ResultFailure::RequestIdentityMismatch,
                         "result for another request is rejected");

    malformed = validResult(request);
    ++malformed.backendImplementationRevision;
    expectResultRejected(request, malformed, ResultFailure::BackendIdentityMismatch,
                         "result from another backend revision is rejected");

    malformed = validResult(request);
    malformed.motionVectors.identity = {};
    expectResultRejected(request, malformed, ResultFailure::InvalidResourceIdentity,
                         "anonymous GPU resources are rejected");

    malformed = validResult(request);
    malformed.motionVectors.helperGeneration = 0;
    expectResultRejected(request, malformed, ResultFailure::InvalidHelperGeneration,
                         "resource without a helper generation is rejected");

    malformed = validResult(request);
    malformed.motionVectors.descriptor.format
        = renderpassoutput::PixelFormat::RGBA16Float;
    expectResultRejected(request, malformed, ResultFailure::OutputDescriptorMismatch,
                         "resource format mismatch is rejected");

    malformed = validResult(request);
    --malformed.motionVectors.byteCount;
    expectResultRejected(request, malformed, ResultFailure::OutputByteCountMismatch,
                         "resource byte mismatch is rejected");
}

EvaluationContext previewContext()
{
    EvaluationContext context;
    context.structuralRevision = 11;
    context.loopDiscontinuitySerial = 2;
    context.helperGeneration = 7;
    return context;
}

void testPreviewSeekAndResetState()
{
    const auto first = admittedRequest(pairDescription(41, 0x21, 0x31));
    const auto next = admittedRequest(pairDescription(42, 0x31, 0x41));
    const auto backwards = admittedRequest(pairDescription(10, 0x51, 0x61));
    const auto otherStream = admittedRequest(pairDescription(70, 0x71, 0x81, 0x22));
    auto context = previewContext();

    EvaluationState state;
    auto invalidContext = context;
    invalidContext.helperGeneration = 0;
    checkTransition(state.evaluate(first, invalidContext), TransitionAction::Reject,
                    ResetCause::None, EvaluationFailure::InvalidContext,
                    "invalid evaluation context is rejected before state mutation");
    check(!state.initialized(), "invalid context leaves preview state empty");
    checkTransition(state.evaluate(first, context), TransitionAction::ResetAndDispatch,
                    ResetCause::FirstEvaluation, EvaluationFailure::None,
                    "first preview pair resets resources before dispatch");
    checkTransition(state.evaluate(first, context), TransitionAction::Hold,
                    ResetCause::None, EvaluationFailure::None,
                    "equal preview request holds its existing result");

    context.paused = true;
    checkTransition(state.evaluate(next, context), TransitionAction::Hold,
                    ResetCause::None, EvaluationFailure::None,
                    "paused preview does not dispatch a new pair");
    check(state.requestCacheKey() == first.cacheKey(),
          "paused preview does not mutate accepted state");
    context.paused = false;
    checkTransition(state.evaluate(next, context), TransitionAction::Dispatch,
                    ResetCause::None, EvaluationFailure::None,
                    "forward preview dispatches the next pair");

    checkTransition(state.evaluate(backwards, context), TransitionAction::ResetAndDispatch,
                    ResetCause::BackwardSeek, EvaluationFailure::None,
                    "backward seek resets before dispatch");
    checkTransition(state.evaluate(otherStream, context), TransitionAction::ResetAndDispatch,
                    ResetCause::SourceDiscontinuity, EvaluationFailure::None,
                    "source replacement resets before dispatch");

    ++context.loopDiscontinuitySerial;
    checkTransition(state.evaluate(first, context), TransitionAction::ResetAndDispatch,
                    ResetCause::LoopDiscontinuity, EvaluationFailure::None,
                    "loop discontinuity resets before dispatch");
    ++context.structuralRevision;
    checkTransition(state.evaluate(next, context), TransitionAction::ResetAndDispatch,
                    ResetCause::StructuralRevision, EvaluationFailure::None,
                    "structural revision resets before dispatch");
    ++context.helperGeneration;
    checkTransition(state.evaluate(first, context), TransitionAction::ResetAndDispatch,
                    ResetCause::HelperRestart, EvaluationFailure::None,
                    "helper restart invalidates resource identities before dispatch");

    context.mode = EvaluationMode::OfflineSequential;
    checkTransition(state.evaluate(next, context), TransitionAction::ResetAndDispatch,
                    ResetCause::EvaluationModeChange, EvaluationFailure::None,
                    "preview and offline modes start separate deterministic sequences");
    context.mode = EvaluationMode::Preview;

    state.reset();
    check(!state.initialized(), "explicit owner reset drops optical-flow state");
    checkTransition(state.evaluate(next, context), TransitionAction::ResetAndDispatch,
                    ResetCause::FirstEvaluation, EvaluationFailure::None,
                    "evaluation after explicit reset starts a fresh sequence");
}

void testOfflineSequentialState()
{
    const auto first = admittedRequest(pairDescription(100, 0x21, 0x31));
    const auto next = admittedRequest(pairDescription(101, 0x31, 0x41));
    const auto gap = admittedRequest(pairDescription(103, 0x51, 0x61));
    auto nonAdjacentDescription = pairDescription(200, 0x71, 0x81);
    nonAdjacentDescription.second.identity.frameIndex = 202;
    nonAdjacentDescription.second.timestamp.ticks = 202 * 1001;
    const auto nonAdjacent = admittedRequest(nonAdjacentDescription);
    const auto backwards = admittedRequest(pairDescription(90, 0x91, 0xa1));

    auto context = previewContext();
    context.mode = EvaluationMode::OfflineSequential;
    EvaluationState state;

    checkTransition(state.evaluate(nonAdjacent, context), TransitionAction::Reject,
                    ResetCause::None, EvaluationFailure::OfflineNonSequential,
                    "offline processing rejects a non-adjacent pair before mutation");
    check(!state.initialized(), "rejected first offline pair leaves state empty");
    checkTransition(state.evaluate(first, context), TransitionAction::ResetAndDispatch,
                    ResetCause::FirstEvaluation, EvaluationFailure::None,
                    "offline range may start at any adjacent pair");
    checkTransition(state.evaluate(gap, context), TransitionAction::Reject,
                    ResetCause::None, EvaluationFailure::OfflineNonSequential,
                    "offline processing rejects a forward gap");
    check(state.requestCacheKey() == first.cacheKey(),
          "rejected offline gap does not mutate state");
    checkTransition(state.evaluate(next, context), TransitionAction::Dispatch,
                    ResetCause::None, EvaluationFailure::None,
                    "offline processing accepts the exact next pair");
    checkTransition(state.evaluate(next, context), TransitionAction::Hold,
                    ResetCause::None, EvaluationFailure::None,
                    "offline repeat holds without duplicate dispatch");
    checkTransition(state.evaluate(backwards, context), TransitionAction::ResetAndDispatch,
                    ResetCause::BackwardSeek, EvaluationFailure::None,
                    "offline backward seek starts a new deterministic sequence");

    EvaluationState independent;
    checkTransition(independent.evaluate(first, context), TransitionAction::ResetAndDispatch,
                    ResetCause::FirstEvaluation, EvaluationFailure::None,
                    "separate offline owner has independent state");
    check(state.requestCacheKey() == backwards.cacheKey()
              && independent.requestCacheKey() == first.cacheKey(),
          "viewport and export owners cannot advance one another");
}

void testSavedPairOfflineTimelineOrdinal()
{
    SourcePairDescriptionV2 pair;
    pair.graphClipId = 44;
    pair.structuralRevision = 9;
    pair.first.clipId = 44;
    pair.first.sourceContent = filled<32>(0x41);
    pair.first.sourceGeneration = 7;
    pair.first.frame = validDescription().first;
    pair.second.clipId = 88;
    pair.second.sourceContent = filled<32>(0x51);
    pair.second.sourceGeneration = 8;
    pair.second.frame = validDescription().second;
    pair.second.frame.identity.stream = filled<16>(0x12);
    pair.output = canonicalOutput(pair.first.frame.extent);
    SourcePairFailureV2 pairFailure = SourcePairFailureV2::None;
    const auto selected = validateSourcePairV2(pair, pairFailure);
    AdmissionFailure admissionFailure = AdmissionFailure::None;
    const auto request = selected ? admit(*selected, validBackend(), admissionFailure)
                                  : std::optional<AdmittedRequest>{};
    check(request.has_value(), "saved pair fixture admits independently timed source clips");
    if (!request) return;

    auto context = previewContext();
    context.mode = EvaluationMode::OfflineSequential;
    EvaluationState state;
    checkTransition(state.evaluate(*request, context), TransitionAction::Reject,
                    ResetCause::None, EvaluationFailure::OfflineNonSequential,
                    "saved pair export requires an authored timeline ordinal");
    context.timelineFrameOrdinal = 71;
    checkTransition(state.evaluate(*request, context), TransitionAction::ResetAndDispatch,
                    ResetCause::FirstEvaluation, EvaluationFailure::None,
                    "saved pair export may begin at a range frame");
    context.timelineFrameOrdinal = 73;
    checkTransition(state.evaluate(*request, context), TransitionAction::Reject,
                    ResetCause::None, EvaluationFailure::OfflineNonSequential,
                    "saved pair export rejects a skipped timeline frame");
    context.timelineFrameOrdinal = 72;
    checkTransition(state.evaluate(*request, context), TransitionAction::Hold,
                    ResetCause::None, EvaluationFailure::None,
                    "unchanged saved frames advance on the next timeline frame");
    checkTransition(state.evaluate(*request, context), TransitionAction::Reject,
                    ResetCause::None, EvaluationFailure::OfflineNonSequential,
                    "saved pair export rejects a repeated timeline frame");
    context.timelineFrameOrdinal = 71;
    checkTransition(state.evaluate(*request, context), TransitionAction::Reject,
                    ResetCause::None, EvaluationFailure::OfflineNonSequential,
                    "saved pair export rejects an out-of-order timeline frame");
}

void testMotionViewDisplayRange()
{
    const auto opticalWire = motionviewoperation::serialize (
        motionviewoperation::Convention::OpticalFirstToSecondTopLeft);
    const auto geometricWire = motionviewoperation::serialize (
        motionviewoperation::Convention::GeometricCurrentMinusPreviousYUp);
    motionviewoperation::Payload presentation;
    check (motionviewoperation::parse (opticalWire, presentation)
           && presentation.displayRangePixels == 0.0
           && motionviewoperation::serialize (presentation) == opticalWire,
           "missing display range retains exact legacy operation bytes");
    const auto rangedWire = motionviewoperation::serialize (motionviewoperation::Payload {
        motionviewoperation::Convention::OpticalFirstToSecondTopLeft, 16.0 });
    check (rangedWire == "<MotionView schemaVersion=\"1\" convention=\"opticalFirstToSecondTopLeft\" displayRangePixels=\"16\"/>"
           && motionviewoperation::parse (rangedWire, presentation)
           && presentation.displayRangePixels == 16.0,
           "positive display range reparses through the existing operation schema");
    const auto withRange = [&] (const std::string& wire, const char* value)
    { return wire.substr (0, wire.size() - 2) + " displayRangePixels=\"" + value + "\"/>"; };
    check (motionviewoperation::parse (withRange (opticalWire, "0"), presentation)
           && motionviewoperation::serialize (presentation) == opticalWire
           && motionviewoperation::serialize (motionviewoperation::Payload {
               motionviewoperation::Convention::GeometricCurrentMinusPreviousYUp, 0.0 }) == geometricWire,
           "explicit zero preserves optical and geometric operation bytes");
    for (const auto* invalid : { "-1", "0.5", "65537", "nan", "inf", "1junk", "", "16\" displayRangePixels=\"32" })
    {
        presentation = { motionviewoperation::Convention::OpticalFirstToSecondTopLeft, 16.0 };
        check (! motionviewoperation::parse (withRange (opticalWire, invalid), presentation)
               && presentation.displayRangePixels == 16.0
               && presentation.convention == motionviewoperation::Convention::OpticalFirstToSecondTopLeft,
               "invalid range parsing refuses without changing the destination");
    }
    check (! motionviewoperation::parse (withRange (geometricWire, "16"), presentation)
           && motionviewoperation::serialize (motionviewoperation::Payload {
               motionviewoperation::Convention::GeometricCurrentMinusPreviousYUp, 16.0 }).empty(),
           "geometric motion refuses an optical pixel display range");
    for (const auto range : { 1.0, 16.25, 65536.0 })
        check (motionviewoperation::parse (motionviewoperation::serialize (motionviewoperation::Payload {
                   motionviewoperation::Convention::OpticalFirstToSecondTopLeft, range }), presentation)
               && presentation.displayRangePixels == range,
               "finite display range endpoints and fractions survive wire round trips");
}
} // namespace

int main()
{
    std::uint32_t sourceClip = 0;
    check (opticalflowoperation::parseSourceClip (
               opticalflowoperation::serializeSourceClip (913), sourceClip)
           && sourceClip == 913,
           "ordered optical-flow source clip binding round-trips canonically");
    for (const std::string_view invalid : {
             "<OpticalFlowSource schemaVersion=\"1\" clipId=\"0\"/>",
             "<OpticalFlowSource schemaVersion=\"1\" clipId=\"0913\"/>",
             "<OpticalFlowSource schemaVersion=\"1\" clipId=\"16777216\"/>",
             "<OpticalFlowSource schemaVersion=\"2\" clipId=\"913\"/>" })
        check (! opticalflowoperation::parseSourceClip (invalid, sourceClip),
               "malformed optical-flow source clip binding is rejected");
    testMotionViewDisplayRange();
    testExactDescriptorAndFootprint();
    testStableCacheKey();
    testTimestampAndExtentMismatch();
    testDistinctClipSourcePairV2();
    testByteOverflowAndLimits();
    testUnavailableNativeGpuFailsClosed();
    testResultResourceReceipt();
    testPreviewSeekAndResetState();
    testOfflineSequentialState();
    testSavedPairOfflineTimelineOrdinal();

    if (failures != 0)
    {
        std::cerr << failures << " optical-flow contract test(s) failed\n";
        return 1;
    }

    std::cout << "optical-flow contract tests passed\n";
    return 0;
}
