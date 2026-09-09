#pragma once

#include "render_snapshot.h"
#include "sdf_native_renderer.h"
#include "../../shared/SdfRaymarchOperationContract.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace videohelper::sdf
{
enum class VisualSdfPreparation
{
    notPresent,
    rendered,
    rejected
};

inline const char* operationKind (videowire::SdfOperation operation) noexcept
{
    switch (operation)
    {
        case videowire::SdfOperation::sphere: return "visual.sdf.sphere";
        case videowire::SdfOperation::box: return "visual.sdf.box";
        case videowire::SdfOperation::roundedBox: return "visual.sdf.rounded-box";
        case videowire::SdfOperation::plane: return "visual.sdf.plane";
        case videowire::SdfOperation::torus: return "visual.sdf.torus";
        case videowire::SdfOperation::capsule: return "visual.sdf.capsule";
        case videowire::SdfOperation::cylinder: return "visual.sdf.cylinder";
        case videowire::SdfOperation::cone: return "visual.sdf.cone";
        case videowire::SdfOperation::gyroid: return "visual.sdf.gyroid";
        case videowire::SdfOperation::unionOp: return "visual.sdf.union";
        case videowire::SdfOperation::intersection: return "visual.sdf.intersection";
        case videowire::SdfOperation::subtraction: return "visual.sdf.subtract";
        case videowire::SdfOperation::smoothUnion: return "visual.sdf.smooth-union";
        case videowire::SdfOperation::smoothIntersection: return "visual.sdf.smooth-intersection";
        case videowire::SdfOperation::smoothSubtraction: return "visual.sdf.smooth-subtract";
        case videowire::SdfOperation::translate: return "visual.sdf.translate";
        case videowire::SdfOperation::rotate: return "visual.sdf.rotate";
        case videowire::SdfOperation::scale: return "visual.sdf.scale";
        case videowire::SdfOperation::repeat: return "visual.sdf.repeat";
        case videowire::SdfOperation::polarRepeat: return "visual.sdf.polar-repeat";
        case videowire::SdfOperation::mirror: return "visual.sdf.mirror";
        case videowire::SdfOperation::twist: return "visual.sdf.twist";
        case videowire::SdfOperation::bend: return "visual.sdf.bend";
        case videowire::SdfOperation::taper: return "visual.sdf.taper";
        case videowire::SdfOperation::displacement: return "visual.sdf.displacement";
        case videowire::SdfOperation::domainWarp: return "visual.sdf.domain-warp";
    }
    return nullptr;
}

template <typename Layer>
inline VisualSdfPreparation prepareVisualSdfLayer (
    const std::vector<videowire::CompiledVisualLayerPlan>& plans,
    int clipId,
    int width,
    int height,
    NativeSdfRenderUse use,
    NativeSdfRenderer& renderer,
    Layer& layer,
    std::vector<NativeSdfRenderedFrame>& frameOwners,
    std::string& error, NativeSdfCacheIdentity cacheIdentity = {})
{
    const auto* plan = videowire::findVisualLayerPlan(plans, clipId);
    if (plan == nullptr) return VisualSdfPreparation::notPresent;

    const auto terminal = std::find_if (plan->operations.begin(), plan->operations.end(),
        [] (const auto& operation) { return operation.kind == "visual.sdf.raymarch"; });
    if (terminal == plan->operations.end()) return VisualSdfPreparation::notPresent;

    const auto reject = [&] (const char* message)
    {
        error = message;
        return VisualSdfPreparation::rejected;
    };
    if (width <= 0 || height <= 0 || terminal->backendCapability != "native-gpu")
        return reject ("visual SDF execution requires a bounded native-GPU target");

    videowire::SdfRaymarchOperation operation;
    if (! decodeSdfRaymarchOperation (terminal->payloadXml, operation)
        || operation.terminalStableId != static_cast<std::uint64_t> (terminal->nodeId) + 1u)
        return reject ("visual SDF raymarch payload is malformed or has stale identity");

    const auto output = std::find_if (plan->operations.begin(), plan->operations.end(),
        [] (const auto& candidate) { return candidate.kind == "video.out"; });
    if (output == plan->operations.end() || output->backendCapability != "native-gpu"
        || plan->operations.size() != operation.geometry.records.size() + 2u)
        return reject ("visual SDF execution requires exactly one raymarch terminal and one Output");

    const auto findOperation = [&] (int nodeId) -> const videowire::CompiledVisualOperation*
    {
        const auto found = std::find_if (plan->operations.begin(), plan->operations.end(),
            [nodeId] (const auto& candidate) { return candidate.nodeId == nodeId; });
        return found == plan->operations.end() ? nullptr : &*found;
    };
    const auto exactPort = [&] (int nodeId, int port, const char* direction,
                                const char* carrier, const char* dataType)
    {
        return std::count_if (plan->ports.begin(), plan->ports.end(), [&] (const auto& binding)
        {
            return binding.nodeId == nodeId && binding.port == port && binding.channels == 1
                && binding.direction == direction && binding.carrier == carrier
                && binding.dataType == dataType;
        }) == 1;
    };
    const auto exactEdge = [&] (int fromNode, int fromPort, int toNode, int toPort)
    {
        return std::count_if (plan->edges.begin(), plan->edges.end(), [&] (const auto& edge)
        {
            return edge.fromNodeId == fromNode && edge.fromPort == fromPort
                && edge.toNodeId == toNode && edge.toPort == toPort;
        }) == 1;
    };

    std::size_t expectedEdgeCount = 2;
    for (const auto& record : operation.geometry.records)
    {
        if (record.stableId == 0
            || record.stableId > static_cast<std::uint64_t> (std::numeric_limits<int>::max()) + 1u)
            return reject ("visual SDF payload contains an invalid stable node identity");
        const int nodeId = static_cast<int> (record.stableId - 1u);
        const auto* compiled = findOperation (nodeId);
        const auto* expectedKind = operationKind (record.operation);
        const auto schema = videowire::sdfOperationSchema (record.operation);
        if (compiled == nullptr || expectedKind == nullptr || compiled->kind != expectedKind
            || compiled->backendCapability != "control-eval" || ! schema.valid
            || record.inputCount != schema.inputCount || record.parameterCount != schema.parameterCount)
            return reject ("visual SDF operation identity or immutable shape is incompatible");
        for (int input = 0; input < record.inputCount; ++input)
        {
            const auto inputStableId = record.inputs[static_cast<std::size_t> (input)];
            const auto source = std::find_if (operation.geometry.records.begin(),
                operation.geometry.records.end(), [&] (const auto& candidate)
                { return candidate.stableId == inputStableId; });
            if (! exactPort (nodeId, input, "in", "control", "sdf")
                || inputStableId == 0
                || inputStableId > static_cast<std::uint64_t> (std::numeric_limits<int>::max()) + 1u
                || source == operation.geometry.records.end()
                || ! exactEdge (static_cast<int> (source->stableId - 1u), source->inputCount,
                                nodeId, input))
                return reject ("visual SDF operation has a missing or stale typed input edge");
            ++expectedEdgeCount;
        }
        if (! exactPort (nodeId, record.inputCount, "out", "control", "sdf"))
            return reject ("visual SDF operation has an incompatible typed output port");
    }

    if (! exactPort (terminal->nodeId, 0, "in", "control", "sdf")
        || ! exactPort (terminal->nodeId, 1, "out", "frame", "image")
        || ! exactPort (output->nodeId, 0, "in", "frame", "image")
        || operation.geometry.rootId == 0
        || operation.geometry.rootId > static_cast<std::uint64_t> (std::numeric_limits<int>::max()) + 1u)
        return reject ("visual SDF terminal has incompatible typed ports or root identity");
    const auto root = std::find_if (operation.geometry.records.begin(), operation.geometry.records.end(),
        [&] (const auto& record) { return record.stableId == operation.geometry.rootId; });
    if (root == operation.geometry.records.end()
        || ! exactEdge (static_cast<int> (root->stableId - 1u), root->inputCount,
                        terminal->nodeId, 0)
        || ! exactEdge (terminal->nodeId, 1, output->nodeId, 0)
        || plan->edges.size() != expectedEdgeCount)
        return reject ("visual SDF terminal topology is incomplete or contains extra edges");

    auto admitted = admitSdfIr (operation.geometry, {}, error);
    if (! admitted) return VisualSdfPreparation::rejected;
    NativeSdfRenderControls controls;
    controls.maximumSteps = operation.maximumSteps;
    controls.epsilon = operation.epsilon;
    controls.maximumDistance = operation.maximumDistance;
    controls.adaptiveQuality = static_cast<arbitgpu::NativeSdfQuality> (operation.adaptiveQuality);
    controls.normalQuality = static_cast<arbitgpu::NativeSdfQuality> (operation.normalQuality);
    controls.shadowQuality = static_cast<arbitgpu::NativeSdfQuality> (operation.shadowQuality);
    controls.output = static_cast<arbitgpu::NativeSdfOutput> (operation.outputPass);

    NativeSdfRenderedFrame rendered;
    const auto geometry = std::make_shared<const AdmittedSdfIr> (std::move (*admitted));
    const NativeSdfRenderDimensions dimensions {
        static_cast<std::uint32_t> (width), static_cast<std::uint32_t> (height) };
    cacheIdentity.clipId = plan->clipId;
    cacheIdentity.planRevision = plan->structuralRevision;
    const bool renderedOk = use == NativeSdfRenderUse::Preview
        ? renderer.renderPreview (geometry, dimensions, controls,
                                  kNativeGpuCapability, rendered, error, cacheIdentity)
        : renderer.renderExport (geometry, dimensions, controls,
                                 kNativeGpuCapability, rendered, error, cacheIdentity);
    if (! renderedOk) return VisualSdfPreparation::rejected;

    layer.texture = 0;
    layer.nativeTextureBackend = rendered.nativeFrame->backend();
    layer.nativeTextureView = rendered.nativeFrame->colorTextureViewHandle();
    if (layer.nativeTextureBackend == "opengl")
    {
        if (layer.nativeTextureView > std::numeric_limits<unsigned>::max())
            return reject ("OpenGL native SDF texture handle exceeds compositor width");
        layer.texture = static_cast<unsigned> (layer.nativeTextureView);
    }
    layer.texWidth = width;
    layer.texHeight = height;
    frameOwners.push_back (std::move (rendered));
    return VisualSdfPreparation::rendered;
}
} // namespace videohelper::sdf
