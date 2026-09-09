#include "optical_flow_native_executor.h"

#include <utility>

namespace videoopticalflow
{
namespace
{
std::string_view evaluationToken (EvaluationFailure failure) noexcept
{
    switch (failure)
    {
        case EvaluationFailure::None: return "none";
        case EvaluationFailure::InvalidContext: return "invalidContext";
        case EvaluationFailure::OfflineNonSequential: return "offlineNonSequential";
    }
    return {};
}

bool sameFrame (const FrameDescription& left,
                const FrameDescription& right) noexcept
{
    return left.identity.stream == right.identity.stream
        && left.identity.content == right.identity.content
        && left.identity.frameIndex == right.identity.frameIndex
        && left.timestamp.ticks == right.timestamp.ticks
        && left.timestamp.timescale == right.timestamp.timescale
        && left.extent == right.extent
        && left.format == right.format
        && left.colorSpace == right.colorSpace;
}

bool validSource (
    const std::shared_ptr<const NativeSourceFrame>& source,
    const FrameDescription& expected,
    const AdmittedRequest& request,
    const EvaluationContext& context) noexcept
{
    if (source == nullptr
        || ! detail::anyNonZero (source->resourceIdentity())
        || source->helperGeneration() == 0
        || source->structuralRevision() == 0
        || source->imageHandle() == 0
        || source->textureViewHandle() == 0)
        return false;

    const auto expectedInputBytes = request.footprint().referencedInputBytes / 2;
    return sameFrame (source->frame(), expected)
        && source->helperGeneration() == context.helperGeneration
        && source->structuralRevision() == context.structuralRevision
        && source->backendImplementation() == request.backendImplementation()
        && source->backendImplementationRevision()
               == request.backendImplementationRevision()
        && source->byteCount() == expectedInputBytes;
}

arbitgpu::NativeOpticalFlowInputResource makeBackendResource (
    const NativeSourceFrame& source)
{
    arbitgpu::NativeOpticalFlowInputResource result;
    result.identity = source.resourceIdentity();
    result.helperGeneration = source.helperGeneration();
    result.structuralRevision = source.structuralRevision();
    result.descriptor = source.frame();
    result.imageHandle = source.imageHandle();
    result.textureViewHandle = source.textureViewHandle();
    result.immutable = true;
    return result;
}
} // namespace

std::string_view token (NativeExecutionFailure failure) noexcept
{
    switch (failure)
    {
        case NativeExecutionFailure::None: return "none";
        case NativeExecutionFailure::MalformedLoweredOperation:
            return "malformedLoweredOperation";
        case NativeExecutionFailure::LoweredExtentMismatch:
            return "loweredExtentMismatch";
        case NativeExecutionFailure::BackendCapabilityMismatch:
            return "backendCapabilityMismatch";
        case NativeExecutionFailure::InvalidSourceResource:
            return "invalidSourceResource";
        case NativeExecutionFailure::SourceResourceMismatch:
            return "sourceResourceMismatch";
        case NativeExecutionFailure::DuplicateSourceResource:
            return "duplicateSourceResource";
        case NativeExecutionFailure::OwnerModeMismatch:
            return "ownerModeMismatch";
        case NativeExecutionFailure::EvaluationRejected:
            return "evaluationRejected";
        case NativeExecutionFailure::MissingHeldPublication:
            return "missingHeldPublication";
        case NativeExecutionFailure::SubmissionRejected:
            return "submissionRejected";
        case NativeExecutionFailure::InvalidSubmittedResource:
            return "invalidSubmittedResource";
    }
    return {};
}

PublishedNativeResult::PublishedNativeResult (
    AdmittedResult receipt,
    std::uint64_t submissionSerial,
    std::uintptr_t imageHandle,
    std::uintptr_t textureViewHandle,
    std::uint64_t structuralRevision,
    std::uint64_t helperGeneration,
    arbitgpu::NativeOpticalFlowOutputLifecycleHandle lifecycle,
    arbitgpu::NativeOpticalFlowExecutionBackend& backend) noexcept
    : receipt_ (std::move (receipt)),
      submissionSerial_ (submissionSerial),
      imageHandle_ (imageHandle),
      textureViewHandle_ (textureViewHandle),
      structuralRevision_ (structuralRevision),
      helperGeneration_ (helperGeneration),
      lifecycle_ (lifecycle),
      backend_ (&backend)
{
}

PublishedNativeResult::~PublishedNativeResult()
{
    if (backend_ != nullptr && lifecycle_)
        backend_->releaseOpticalFlowOutput (lifecycle_);
}

NativeExecutor::NativeExecutor (
    EvaluationMode ownerMode,
    arbitgpu::NativeOpticalFlowExecutionBackend& backend) noexcept
    : backend_ (backend),
      ownerMode_ (ownerMode)
{
}

bool NativeExecutor::fail (NativeExecutionFailure reason,
                           std::string message,
                           std::string& error,
                           NativeExecutionFailure* failure) const
{
    error = std::move (message);
    if (failure != nullptr) *failure = reason;
    return false;
}

bool NativeExecutor::execute (
    const LoweredOperation& operation,
    const AdmittedRequest& request,
    const std::shared_ptr<const NativeSourceFrame>& first,
    const std::shared_ptr<const NativeSourceFrame>& second,
    const EvaluationContext& context,
    std::string& error,
    NativeExecutionFailure* failure)
{
    error.clear();
    if (failure != nullptr) *failure = NativeExecutionFailure::None;

    if (context.mode != ownerMode_)
    {
        return fail (NativeExecutionFailure::OwnerModeMismatch,
                     "native optical-flow evaluation mode does not match its owner",
                     error, failure);
    }

    opticalflowoperation::Payload payload;
    if (operation.kind != opticalflowoperation::kOperationKind
        || operation.backendCapability != opticalflowoperation::kBackendCapability
        || ! opticalflowoperation::parse (operation.payloadXml, payload)
        || opticalflowoperation::serialize (payload) != operation.payloadXml)
    {
        return fail (NativeExecutionFailure::MalformedLoweredOperation,
                     "native optical flow requires the exact canonical lowered operation",
                     error, failure);
    }
    if (payload.extent != request.output().extent)
    {
        return fail (NativeExecutionFailure::LoweredExtentMismatch,
                     "lowered optical-flow extent does not match the admitted request",
                     error, failure);
    }

    // Re-admit against the backend's current capability receipt. This prevents
    // a request admitted before device replacement/reconfiguration from being
    // submitted to a different implementation or to a CPU/unavailable backend.
    Description description;
    description.first = request.first();
    description.second = request.second();
    description.output = request.output();
    AdmissionFailure admissionFailure = AdmissionFailure::None;
    const auto current = admit (
        description, backend_.opticalFlowCapabilities(), admissionFailure);
    if (! current || current->cacheKey() != request.cacheKey())
    {
        return fail (NativeExecutionFailure::BackendCapabilityMismatch,
                     "native optical-flow backend no longer matches the admitted request: "
                         + std::string (videoopticalflow::token (admissionFailure)),
                     error, failure);
    }

    if (first == nullptr || second == nullptr
        || first->helperGeneration() == 0 || second->helperGeneration() == 0
        || first->structuralRevision() == 0 || second->structuralRevision() == 0
        || first->imageHandle() == 0 || second->imageHandle() == 0
        || first->textureViewHandle() == 0 || second->textureViewHandle() == 0
        || ! detail::anyNonZero (first->resourceIdentity())
        || ! detail::anyNonZero (second->resourceIdentity()))
    {
        return fail (NativeExecutionFailure::InvalidSourceResource,
                     "native optical flow requires two complete immutable GPU source resources",
                     error, failure);
    }
    if (! validSource (first, request.first(), request, context)
        || ! validSource (second, request.second(), request, context))
    {
        return fail (NativeExecutionFailure::SourceResourceMismatch,
                     "native optical-flow source resource does not match frame, extent, format, revision, or backend",
                     error, failure);
    }
    if (first->resourceIdentity() == second->resourceIdentity()
        || first->imageHandle() == second->imageHandle()
        || first->textureViewHandle() == second->textureViewHandle())
    {
        return fail (NativeExecutionFailure::DuplicateSourceResource,
                     "native optical flow requires two distinct immutable source resources",
                     error, failure);
    }

    auto stagedState = evaluationState_;
    const auto transition = stagedState.evaluate (request, context);
    if (transition.action == TransitionAction::Reject)
    {
        return fail (NativeExecutionFailure::EvaluationRejected,
                     "native optical-flow evaluation rejected: "
                         + std::string (evaluationToken (transition.failure)),
                     error, failure);
    }
    if (transition.action == TransitionAction::Hold)
    {
        if (publication_ == nullptr
            || publication_->receipt().requestCacheKey() != request.cacheKey()
            || publication_->structuralRevision() != context.structuralRevision
            || publication_->helperGeneration() != context.helperGeneration)
        {
            return fail (NativeExecutionFailure::MissingHeldPublication,
                         "native optical-flow hold has no matching last-good publication",
                         error, failure);
        }
        return true;
    }

    const bool resetTemporalState =
        transition.action == TransitionAction::ResetAndDispatch;
    if (resetTemporalState)
    {
        // A seek/revision/loop/helper reset makes the old result semantically
        // stale. Drop owner state before dispatch so failure cannot republish it.
        publication_.reset();
        evaluationState_.reset();
    }

    const auto firstResource = makeBackendResource (*first);
    const auto secondResource = makeBackendResource (*second);
    auto submission = backend_.executeOpticalFlow (
        request, firstResource, secondResource, resetTemporalState);
    const bool contradictory = submission.completed
        && (submission.submission == 0 || ! submission.lifecycle
            || submission.imageHandle == 0 || submission.textureViewHandle == 0
            || ! submission.error.empty());
    if (! submission.completed || contradictory)
    {
        if (submission.lifecycle)
            backend_.releaseOpticalFlowOutput (submission.lifecycle);
        return fail (NativeExecutionFailure::SubmissionRejected,
                     contradictory
                         ? "native optical-flow backend returned a contradictory completed submission"
                         : (submission.error.empty()
                             ? "native optical-flow GPU execution did not complete"
                             : std::move (submission.error)),
                     error, failure);
    }

    ResultFailure resultFailure = ResultFailure::None;
    auto admittedResult = admitResult (request, submission.result, resultFailure);
    if (! admittedResult)
    {
        backend_.releaseOpticalFlowOutput (submission.lifecycle);
        return fail (NativeExecutionFailure::InvalidSubmittedResource,
                     "native optical-flow backend returned an incompatible submitted resource: "
                         + std::string (videoopticalflow::token (resultFailure)),
                     error, failure);
    }

    auto published = std::shared_ptr<const PublishedNativeResult> (
        new PublishedNativeResult (
            std::move (*admittedResult), submission.submission,
            submission.imageHandle, submission.textureViewHandle,
            context.structuralRevision, context.helperGeneration,
            submission.lifecycle, backend_));
    publication_ = std::move (published);
    evaluationState_ = std::move (stagedState);
    return true;
}

void NativeExecutor::reset() noexcept
{
    publication_.reset();
    evaluationState_.reset();
}
} // namespace videoopticalflow
