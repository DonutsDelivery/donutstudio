#pragma once

#include "optical_flow_source_lease.h"
#include "../../shared/MotionViewOperationContract.h"

#include <algorithm>
#include <memory>
#include <string>

namespace videoopticalflow
{
struct AuthoredPairSelection final
{
    const videowire::CompiledVisualLayerPlan* plan = nullptr;
    opticalflowoperation::SourcePair pair;
};

// A retained RG16F field and the exact two saved media frames that produced it.
// The native image remains owned by PublishedNativeResult; no Image conversion
// or geometric Motion AOV is implied by this terminal graph output.
struct AuthoredPairResult final
{
    std::shared_ptr<const PublishedNativeResult> field;
    std::shared_ptr<const NativeSourceFrame> firstImage;
    SourcePairDescriptionV2 sources;
    std::uint64_t snapshotGeneration = 0;
    double displaySec = 0.0;
    int nodeId = 0;
};

inline bool sameAuthoredPairPlan (const videowire::CompiledVisualLayerPlan& admitted,
                                  const AuthoredPairSelection& selected)
{
    if (selected.plan == nullptr || admitted.clipId != selected.plan->clipId
        || admitted.structuralRevision != selected.plan->structuralRevision)
        return false;
    if (admitted.operations.size() != selected.plan->operations.size()
        || admitted.edges.size() != selected.plan->edges.size()
        || admitted.ports.size() != selected.plan->ports.size()
        || admitted.nodeIds != selected.plan->nodeIds
        || admitted.nodeKinds != selected.plan->nodeKinds) return false;
    for (std::size_t index = 0; index < admitted.operations.size(); ++index)
        if (admitted.operations[index].nodeId != selected.plan->operations[index].nodeId
            || admitted.operations[index].kind != selected.plan->operations[index].kind
            || admitted.operations[index].backendCapability != selected.plan->operations[index].backendCapability
            || admitted.operations[index].payloadXml != selected.plan->operations[index].payloadXml)
            return false;
    for (std::size_t index = 0; index < admitted.edges.size(); ++index)
    {
        const auto& left = admitted.edges[index];
        const auto& right = selected.plan->edges[index];
        if (left.fromNodeId != right.fromNodeId || left.fromPort != right.fromPort
            || left.toNodeId != right.toNodeId || left.toPort != right.toPort) return false;
    }
    for (std::size_t index = 0; index < admitted.ports.size(); ++index)
    {
        const auto& left = admitted.ports[index];
        const auto& right = selected.plan->ports[index];
        if (left.nodeId != right.nodeId || left.port != right.port
            || left.channels != right.channels || left.direction != right.direction
            || left.carrier != right.carrier || left.dataType != right.dataType
            || left.pixelFormat != right.pixelFormat || left.colorSpace != right.colorSpace
            || left.transfer != right.transfer || left.alpha != right.alpha)
            return false;
    }
    const auto flow = std::find_if (admitted.operations.begin(), admitted.operations.end(),
        [] (const auto& operation)
        { return operation.kind == opticalflowoperation::kPairOperationKind; });
    const auto savedFlow = std::find_if (selected.plan->operations.begin(),
        selected.plan->operations.end(), [] (const auto& operation)
        { return operation.kind == opticalflowoperation::kPairOperationKind; });
    opticalflowoperation::SourcePair pair;
    if (flow == admitted.operations.end() || savedFlow == selected.plan->operations.end()
        || flow->nodeId != savedFlow->nodeId
        || flow->backendCapability != savedFlow->backendCapability
        || flow->payloadXml != savedFlow->payloadXml
        || ! opticalflowoperation::parseSourcePair(flow->payloadXml, pair)
        || pair.images != selected.pair.images)
        return false;
    for (const auto& endpoint : pair.images)
    {
        const auto source = std::find_if (admitted.operations.begin(), admitted.operations.end(),
            [&] (const auto& operation)
            { return operation.nodeId == static_cast<int>(endpoint.nodeId); });
        const auto savedSource = std::find_if (selected.plan->operations.begin(),
            selected.plan->operations.end(), [&] (const auto& operation)
            { return operation.nodeId == static_cast<int>(endpoint.nodeId); });
        if (source == admitted.operations.end() || savedSource == selected.plan->operations.end()
            || source->kind != savedSource->kind
            || source->backendCapability != savedSource->backendCapability
            || source->payloadXml != savedSource->payloadXml)
            return false;
    }
    return true;
}

inline bool hasMotionView (const videowire::CompiledVisualLayerPlan& plan) noexcept
{
    return plan.operations.size() == 5
        && plan.operations[3].kind == motionviewoperation::kNodeKind;
}

inline bool hasMotionBlur (const videowire::CompiledVisualLayerPlan& plan) noexcept
{
    return plan.operations.size() == 5
        && plan.operations[3].kind == visualtemporalsampling::kMotionBlurKind;
}

inline std::string renderAuthoredMotionBlur (
    const videowire::CompiledVisualLayerPlan& plan,
    const AuthoredPairResult& artifact, std::uint64_t snapshotGeneration,
    double displaySec, std::uint64_t helperGeneration,
    arbitgpu::NativeOpticalFlowExecutionBackend& backend,
    std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& image)
{
    image.reset();
    if (! hasMotionBlur(plan) || ! plan.producerValidated
        || plan.clipId != static_cast<int>(artifact.sources.graphClipId)
        || plan.structuralRevision == 0
        || plan.structuralRevision != artifact.sources.structuralRevision
        || artifact.snapshotGeneration != snapshotGeneration
        || artifact.displaySec != displaySec || ! artifact.field || ! artifact.firstImage
        || artifact.field->structuralRevision() != plan.structuralRevision
        || artifact.field->helperGeneration() != helperGeneration
        || artifact.firstImage->structuralRevision() != plan.structuralRevision
        || artifact.firstImage->helperGeneration() != helperGeneration
        || artifact.field->nativeContextIdentity() == 0
        || artifact.field->nativeContextIdentity()
            != artifact.firstImage->nativeContextIdentity()
        || artifact.field->submissionSerial() == 0)
        return "Motion Blur requires the current saved pair, time, helper, and retained field";
    opticalflowoperation::SourcePair pair;
    std::uint32_t firstClip = 0, secondClip = 0;
    if (! opticalflowoperation::parseSourcePair(plan.operations[2].payloadXml, pair)
        || ! opticalflowoperation::parseSourceClip(plan.operations[0].payloadXml, firstClip)
        || ! opticalflowoperation::parseSourceClip(plan.operations[1].payloadXml, secondClip)
        || pair.images[0].nodeId != static_cast<std::uint32_t>(plan.operations[0].nodeId)
        || pair.images[1].nodeId != static_cast<std::uint32_t>(plan.operations[1].nodeId)
        || firstClip != artifact.sources.first.clipId
        || secondClip != artifact.sources.second.clipId
        || artifact.firstImage->sourceClipId() != firstClip
        || artifact.firstImage->sourceContent() != artifact.sources.first.sourceContent
        || artifact.firstImage->sourceGeneration() != artifact.sources.first.sourceGeneration
        || artifact.firstImage->frame().identity.stream
            != artifact.sources.first.frame.identity.stream
        || artifact.firstImage->frame().identity.content
            != artifact.sources.first.frame.identity.content
        || artifact.firstImage->frame().identity.frameIndex
            != artifact.sources.first.frame.identity.frameIndex
        || artifact.firstImage->frame().timestamp.ticks
            != artifact.sources.first.frame.timestamp.ticks
        || artifact.firstImage->frame().timestamp.timescale
            != artifact.sources.first.frame.timestamp.timescale)
        return "Motion Blur Image source differs from its saved A/B operation";
    const auto& field = artifact.field->receipt().motionVectors();
    if (field.descriptor.extent != artifact.firstImage->frame().extent
        || field.descriptor.format != renderpassoutput::PixelFormat::RG16Float
        || field.descriptor.colorSpace != renderpassoutput::ColorSpace::Data
        || field.descriptor.units != MotionVectorUnits::PixelDisplacement
        || field.descriptor.direction != MotionVectorDirection::FirstToSecond
        || field.descriptor.coordinates != MotionVectorCoordinates::ImageTopLeftPositiveRightDown
        || ! detail::anyNonZero(field.identity)
        || field.helperGeneration != helperGeneration
        || artifact.field->imageHandle() == 0 || artifact.field->textureViewHandle() == 0)
        return "Motion Blur requires a complete optical field with top-left pixel displacement";
    visualtemporalsampling::Payload payload;
    std::string error;
    if (plan.operations[3].backendCapability != "temporal-sampling"
        || ! visualtemporalsampling::parseForKind (
            plan.operations[3].kind, plan.operations[3].payloadXml, payload, &error))
        return error.empty() ? "Motion Blur has an invalid saved sampling operation" : error;
    image = backend.renderMotionBlur (artifact.firstImage, artifact.field, payload, error);
    if (! image) return error.empty() ? "Motion Blur native image conversion failed" : error;
    const auto descriptor = image->colorTextureDescriptor();
    if (! descriptor.complete() || descriptor.backend != image->backend()
        || descriptor.deviceOrContextIdentity != artifact.field->nativeContextIdentity()
        || (descriptor.backend != "opengl" && descriptor.backend != "metal")
        || descriptor.format != (descriptor.backend == "metal"
            ? arbitgpu::NativeTexturePixelFormat::Bgra8Unorm
            : arbitgpu::NativeTexturePixelFormat::Rgba8Unorm)
        || descriptor.colorSpace != colortransform::ColorSpace::SRGB
        || descriptor.transfer != colortransform::TransferFunction::SRGB
        || descriptor.rowOrder != arbitgpu::NativeTextureRowOrder::TopFirst
        || descriptor.width != image->width() || descriptor.height != image->height())
    {
        image.reset();
        return "Motion Blur native Image descriptor does not match its retained resource";
    }
    return {};
}

inline std::string renderAuthoredMotionView (
    const videowire::CompiledVisualLayerPlan& plan,
    const AuthoredPairResult& artifact, std::uint64_t snapshotGeneration,
    double displaySec, std::uint64_t helperGeneration,
    arbitgpu::NativeOpticalFlowExecutionBackend& backend,
    std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& image)
{
    image.reset();
    if (! hasMotionView (plan) || ! plan.producerValidated
        || plan.structuralRevision == 0 || plan.structuralRevision != artifact.sources.structuralRevision
        || plan.clipId != static_cast<int> (artifact.sources.graphClipId)
        || artifact.snapshotGeneration != snapshotGeneration
        || artifact.displaySec != displaySec || ! artifact.field
        || artifact.field->structuralRevision() != plan.structuralRevision
        || artifact.field->helperGeneration() != helperGeneration
        || artifact.field->nativeContextIdentity() == 0
        || artifact.field->submissionSerial() == 0)
        return "Motion View requires the current saved graph, time, helper, and retained field";
    motionviewoperation::Payload presentation;
    if (plan.operations[3].backendCapability != motionviewoperation::kBackendCapability
        || ! motionviewoperation::parse (plan.operations[3].payloadXml, presentation)
        || presentation.convention != motionviewoperation::Convention::OpticalFirstToSecondTopLeft)
        return "Motion View requires first-to-second top-left optical motion";
    opticalflowoperation::SourcePair pair;
    std::uint32_t firstClip = 0, secondClip = 0;
    if (! opticalflowoperation::parseSourcePair(plan.operations[2].payloadXml, pair)
        || ! opticalflowoperation::parseSourceClip(plan.operations[0].payloadXml, firstClip)
        || ! opticalflowoperation::parseSourceClip(plan.operations[1].payloadXml, secondClip)
        || pair.images[0].nodeId != static_cast<std::uint32_t>(plan.operations[0].nodeId)
        || pair.images[1].nodeId != static_cast<std::uint32_t>(plan.operations[1].nodeId)
        || firstClip != artifact.sources.first.clipId
        || secondClip != artifact.sources.second.clipId)
        return "Motion View source order differs from its saved A/B operation";
    const auto& field = artifact.field->receipt().motionVectors();
    if (field.descriptor.extent.width == 0 || field.descriptor.extent.height == 0
        || field.descriptor.format != renderpassoutput::PixelFormat::RG16Float
        || field.descriptor.colorSpace != renderpassoutput::ColorSpace::Data
        || field.descriptor.units != MotionVectorUnits::PixelDisplacement
        || field.descriptor.direction != MotionVectorDirection::FirstToSecond
        || field.descriptor.coordinates != MotionVectorCoordinates::ImageTopLeftPositiveRightDown
        || ! detail::anyNonZero(field.identity)
        || field.helperGeneration != helperGeneration
        || artifact.field->imageHandle() == 0 || artifact.field->textureViewHandle() == 0)
        return "Motion View requires a complete optical field with top-left pixel displacement";
    std::string error;
    image = backend.renderMotionView (artifact.field, presentation, error);
    if (! image) return error.empty() ? "Motion View native image conversion failed" : error;
    const auto descriptor = image->colorTextureDescriptor();
    if (! descriptor.complete() || descriptor.backend != image->backend()
        || descriptor.deviceOrContextIdentity != artifact.field->nativeContextIdentity()
        || (descriptor.backend != "opengl" && descriptor.backend != "metal")
        || (descriptor.backend == "opengl"
            && descriptor.imageHandle != descriptor.textureViewHandle)
        || descriptor.format != (descriptor.backend == "metal"
            ? arbitgpu::NativeTexturePixelFormat::Bgra8Unorm
            : arbitgpu::NativeTexturePixelFormat::Rgba8Unorm)
        || descriptor.colorSpace != colortransform::ColorSpace::SRGB
        || descriptor.transfer != colortransform::TransferFunction::SRGB
        || descriptor.rowOrder != arbitgpu::NativeTextureRowOrder::TopFirst
        || descriptor.imageHandle != image->colorImageHandle()
        || descriptor.textureViewHandle != image->colorTextureViewHandle()
        || descriptor.width != image->width() || descriptor.height != image->height())
    {
        image.reset();
        return "Motion View native Image descriptor does not match its retained resource";
    }
    return {};
}

// Select the saved operation from this immutable snapshot. The catalog and
// decoder recheck the same identities when the worker obtains source frames.
inline std::string selectAuthoredPair (
    const videowire::ResolvedVisualSnapshot& snapshot, int clipId,
    AuthoredPairSelection& out)
{
    out = {};
    const auto* plan = videowire::findVisualLayerPlan (snapshot.visualLayerPlans, clipId);
    if (plan == nullptr || ! plan->producerValidated || plan->clipId <= 0
        || plan->structuralRevision == 0 || plan->identityMode != "authoredGraph"
        || snapshot.snapshotGeneration == 0)
        return "Optical Flow has no current saved graph owner";
    const auto flow = std::find_if (plan->operations.begin(), plan->operations.end(),
        [] (const auto& operation)
        { return operation.kind == opticalflowoperation::kPairOperationKind; });
    if (flow == plan->operations.end()
        || std::count_if (plan->operations.begin(), plan->operations.end(),
            [] (const auto& operation)
            { return operation.kind == opticalflowoperation::kPairOperationKind; }) != 1
        || flow->backendCapability != opticalflowoperation::kBackendCapability
        || ! opticalflowoperation::parseSourcePair (flow->payloadXml, out.pair)
        || out.pair.images[0] == out.pair.images[1])
        return "Optical Flow has no exact ordered native source pair";
    std::string error;
    if (! videowire::validateCompiledVisualLayerPlans (
            snapshot.segments, snapshot.visualLayerPlans, true, error)
        || ! videowire::validateOpticalFlowSources (
            snapshot.segments, snapshot.visualLayerPlans,
            snapshot.opticalFlowSources, snapshot.snapshotGeneration, error))
        return error.empty() ? "Optical Flow snapshot admission failed" : error;
    out.plan = plan;
    return {};
}

// The GPU context owner consumes a media worker's immutable decoded receipt.
// The caller must reject obsolete requests before and after this call; it may
// retain the resulting field only while its native context remains alive.
inline std::string executeAuthoredDecodedPair (
    const videowire::ResolvedVisualSnapshot& snapshot, int clipId,
    double displaySec, const DecodedPairReceipt& decoded,
    const EvaluationContext& context, NativeSourceUploader& uploader,
    arbitgpu::NativeOpticalFlowExecutionBackend& backend,
    NativeExecutor& executor,
    std::shared_ptr<const PublishedNativeResult>& out,
    AuthoredPairResult* artifact = nullptr,
    bool pixelsValidatedOnWorker = false)
{
    out.reset();
    if (artifact != nullptr) *artifact = {};
    AuthoredPairSelection selected;
    auto error = selectAuthoredPair (snapshot, clipId, selected);
    if (! error.empty()) return error;
    if (context.structuralRevision != selected.plan->structuralRevision
        || context.helperGeneration == 0)
        return "Optical Flow execution owner changed after snapshot selection";
    for (const auto& source : decoded.sources)
    {
        const auto binding = std::find_if (snapshot.opticalFlowSources.begin(),
            snapshot.opticalFlowSources.end(), [&] (const auto& value)
            { return value.clipId == static_cast<int> (source.clipId); });
        const auto* segment = videowire::resolveClipSegmentAtDisplayTime (
            snapshot.segments, static_cast<int> (source.clipId), displaySec);
        if (binding == snapshot.opticalFlowSources.end() || segment == nullptr
            || source.sourceGeneration != binding->sourceGeneration
            || segment->sourcePath != binding->sourcePath
            || source.sourceTimeSec != segment->inSec
                + (displaySec - segment->displayStartSec) * segment->rate)
            return "Optical Flow decoded receipt belongs to another saved source time";
    }
    std::shared_ptr<const SourcePairLease> lease;
    error = leaseDecodedSourcePair (
        *selected.plan, selected.pair, decoded, snapshot.snapshotGeneration,
        context.helperGeneration, backend.opticalFlowCapabilities(), uploader, lease,
        pixelsValidatedOnWorker, context.mode == EvaluationMode::Preview);
    if (! error.empty()) return error;
    if (lease == nullptr || ! lease->request || ! lease->selection)
        return "Optical Flow worker has no admitted native source pair";

    const auto operationXml = opticalflowoperation::serialize (
        { lease->request->output().extent });
    const LoweredOperation operation {
        opticalflowoperation::kOperationKind,
        opticalflowoperation::kBackendCapability, operationXml };
    if (! executor.execute (operation, *lease->request, lease->frames[0],
                            lease->frames[1], context, error))
        return error;
    out = executor.publication();
    if (out == nullptr) return "Optical Flow worker did not publish motion vectors";
    if (artifact != nullptr)
    {
        artifact->field = out;
        artifact->firstImage = lease->frames[0];
        artifact->sources = lease->selection->description();
        artifact->snapshotGeneration = snapshot.snapshotGeneration;
        artifact->displaySec = displaySec;
        const auto flow = std::find_if (selected.plan->operations.begin(),
            selected.plan->operations.end(), [] (const auto& operation)
            { return operation.kind == opticalflowoperation::kPairOperationKind; });
        artifact->nodeId = flow->nodeId;
    }
    return {};
}

// Offline export has an owned worker and context, so its existing synchronous
// route remains valid. A presentation callback must use the split route above.
inline std::string executeAuthoredPair (
    const videowire::ResolvedVisualSnapshot& snapshot, int clipId,
    double displaySec, int maxW, int maxH,
    const EvaluationContext& context, NativeSourceUploader& uploader,
    arbitgpu::NativeOpticalFlowExecutionBackend& backend,
    NativeExecutor& executor,
    std::shared_ptr<const PublishedNativeResult>& out,
    AuthoredPairResult* artifact = nullptr)
{
    out.reset();
    if (artifact != nullptr) *artifact = {};
    AuthoredPairSelection selected;
    auto error = selectAuthoredPair (snapshot, clipId, selected);
    if (! error.empty()) return error;
    std::shared_ptr<const DecodedPairReceipt> decoded;
    error = resolveDecodedPair (*selected.plan, selected.pair,
        snapshot.opticalFlowSources, snapshot.segments,
        snapshot.snapshotGeneration, displaySec, maxW, maxH, decoded);
    if (! error.empty()) return error;
    return executeAuthoredDecodedPair (snapshot, clipId, displaySec, *decoded,
        context, uploader, backend, executor, out, artifact);
}
} // namespace videoopticalflow
