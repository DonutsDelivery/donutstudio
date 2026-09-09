#pragma once

#include "visual_plan_telemetry.h"
#include <nlohmann/json.hpp>

namespace videowire
{
inline nlohmann::json visualTelemetryJson (const VisualTelemetrySnapshot& value)
{
    const auto unavailable = [] { return nlohmann::json(nullptr); };
    const auto duration = [&](const VisualDurationCounter& d) {
        return nlohmann::json{{"available", d.observed},
            {"count", d.observed ? nlohmann::json(d.count) : unavailable()},
            {"totalNs", d.observed ? nlohmann::json(d.totalNs) : unavailable()},
            {"movingNs", d.observed ? nlohmann::json(d.movingNs) : unavailable()}};
    };
    const auto backend = [](VisualBackend value) {
        switch (value) { case VisualBackend::openGL: return "opengl"; case VisualBackend::metal: return "metal";
            case VisualBackend::vulkan: return "vulkan"; case VisualBackend::software: return "software"; }
        return "unknown";
    };
    const char* transport = value.transportMode == VisualTransportMode::zeroCopy ? "zero-copy" : "readback";
    const auto presentationMode = value.presentationMode == VisualPresentationMode::windowPresent
        ? "window-present" : value.presentationMode == VisualPresentationMode::encodedHandoff
        ? "encoded/handoff" : "none";
    nlohmann::json layers = nlohmann::json::array();
    for (const auto& layer : value.layers)
        layers.push_back({ { "clipId", layer.clipId }, { "structuralRevision", layer.structuralRevision },
            { "evaluations", layer.evaluations }, { "measuredTotalNs", layer.measuredTotalNs },
            { "measuredMovingNs", layer.measuredMovingNs } });
    nlohmann::json nodes = nlohmann::json::array();
    for (const auto& node : value.nodes)
        nodes.push_back({ { "clipId", node.clipId }, { "structuralRevision", node.structuralRevision },
            { "stableNodeId", node.stableNodeId }, { "available", node.available },
            { "evaluations", node.available ? nlohmann::json(node.evaluations) : unavailable() },
            { "measuredTotalNs", node.available ? nlohmann::json(node.measuredTotalNs) : unavailable() },
            { "measuredMovingNs", node.available ? nlohmann::json(node.measuredMovingNs) : unavailable() } });
    const auto maybe = [&](bool observed, auto v) { return observed ? nlohmann::json(v) : unavailable(); };
    const auto& budget = value.budgetReceipt;
    const auto& limits = budget.limits;
    const auto& usage = budget.usage;
    return {
        { "graphEvaluations", value.graphEvaluations }, { "graphMeasuredTotalNs", value.graphMeasuredTotalNs },
        { "graphMeasuredMovingNs", value.graphMeasuredMovingNs },
        { "planCacheHits", value.planCacheHits }, { "planCacheMisses", value.planCacheMisses },
        { "lastPlanLoweringNs", value.lastPlanLoweringNs }, { "planLoweringBudgetNs", value.planLoweringBudgetNs },
        { "lastPlanLoweringWithinBudget", value.lastPlanLoweringWithinBudget },
        { "planInstalls", value.planInstalls }, { "droppedSamples", value.droppedSamples },
        { "rejectedPlans", value.rejectedPlans }, { "failedLowerings", value.failedLowerings },
        { "lastLoweringError", value.lastLoweringError },
        { "nodeSubsetLabel", "bounded executable render operations" },
        { "executableNodeTotal", value.executableNodeTotal }, { "executableNodeReported", value.nodes.size() },
        { "executableNodeSubsetTruncated", value.executableNodeSubsetTruncated },
        { "compositor", duration(value.compositor) }, { "presentation", duration(value.presentation) },
        { "presentationSemantic", value.presentationModeObserved ? nlohmann::json(presentationMode) : unavailable() },
        { "particles", duration(value.particles) }, { "generators", duration(value.generators) },
        { "transport", {{"available",value.transportObserved},
            {"allocationAvailable",value.zeroCopyAllocationObserved},
            {"mode", value.transportObserved ? nlohmann::json(transport) : unavailable()},
            {"zeroCopyFrames",maybe(value.transportObserved,value.zeroCopyFrames)},
            {"zeroCopyAllocationBytes",maybe(value.zeroCopyAllocationObserved,value.zeroCopyAllocationBytes)},
            {"readbackFrames",maybe(value.transportObserved,value.readbackFrames)},
            {"readbackCopiedBytes",maybe(value.transportObserved,value.readbackCopiedBytes)},
            {"framesHandedOff",maybe(value.transportObserved,value.transportFramesHandedOff)},
            {"framesRendered",maybe(value.transportObserved,value.framesRendered)},
            {"framesPresented",maybe(value.transportObserved,value.framesPresented)},
            {"exportHandoffFrames",maybe(value.transportObserved,value.exportHandoffFrames)}} },
        { "drops", {{"available",value.dropsObserved},{"framesDropped",maybe(value.dropsObserved,value.framesDropped)},
            {"reasons",{{"noBuffer",maybe(value.dropsObserved,value.droppedNoBuffer)},
                {"transportFailure",maybe(value.dropsObserved,value.droppedTransportFailure)},
                {"renderFailure",maybe(value.dropsObserved,value.droppedRenderFailure)},
                {"dimensionMismatch",maybe(value.dropsObserved,value.droppedDimensionMismatch)},
                {"blitFailure",maybe(value.dropsObserved,value.droppedBlitFailure)},
                {"fenceFailure",maybe(value.dropsObserved,value.droppedFenceFailure)},
                {"socketFailure",maybe(value.dropsObserved,value.droppedSocketFailure)}}}} },
        { "backend", {{"available",value.backendObserved},
            {"initial",value.backendObserved ? nlohmann::json(backend(value.initialBackend)) : unavailable()},
            {"current",value.backendObserved ? nlohmann::json(backend(value.currentBackend)) : unavailable()},
            {"fallbackCount",maybe(value.backendObserved,value.fallbackCount)}} },
        { "resources", {{"available",value.resourcesObserved},
            {"retainedFramesCurrent",maybe(value.resourcesObserved,value.retainedFramesCurrent)},
            {"retainedFramesPeak",maybe(value.resourcesObserved,value.retainedFramesPeak)},
            {"intermediateImagesCurrent",maybe(value.resourcesObserved,value.intermediateImagesCurrent)},
            {"intermediateImagesPeak",maybe(value.resourcesObserved,value.intermediateImagesPeak)},
            {"retainedBytesCurrent",maybe(value.resourcesObserved,value.retainedBytesCurrent)},
            {"retainedBytesPeak",maybe(value.resourcesObserved,value.retainedBytesPeak)}} },
        { "budget", {{"available",value.budgetObserved},
            {"backendProfile",maybe(value.budgetObserved,limits.backendProfile)},
            {"backendDeviceIdentity",maybe(value.budgetObserved,limits.backendDeviceIdentity)},
            {"canvasWidth",maybe(value.budgetObserved,budget.canvasWidth)},
            {"canvasHeight",maybe(value.budgetObserved,budget.canvasHeight)},
            {"limits",{{"descriptors",maybe(value.budgetObserved,limits.descriptors)},
                {"operations",maybe(value.budgetObserved,limits.operations)},
                {"sceneRecords",maybe(value.budgetObserved,limits.sceneRecords)},
                {"frameOutputs",maybe(value.budgetObserved,limits.frameOutputs)},
                {"liveFrames",maybe(value.budgetObserved,limits.liveFrames)},
                {"frameSlots",maybe(value.budgetObserved,limits.frameSlots)},
                {"allocatedFrameBytes",maybe(value.budgetObserved,limits.allocatedFrameBytes)},
                {"maximumImageDimension",maybe(value.budgetObserved,limits.maximumImageDimension)}}},
            {"usage",{{"planCount",maybe(value.budgetObserved,usage.planCount)},
                {"descriptors",maybe(value.budgetObserved,usage.descriptors)},
                {"operations",maybe(value.budgetObserved,usage.operations)},
                {"sceneRecords",maybe(value.budgetObserved,usage.sceneRecords)},
                {"frameOutputs",maybe(value.budgetObserved,usage.frameOutputs)},
                {"peakLiveFrames",maybe(value.budgetObserved,usage.peakLiveFrames)},
                {"frameSlots",maybe(value.budgetObserved,usage.frameSlots)},
                {"allocatedFrameBytes",maybe(value.budgetObserved,usage.allocatedFrameBytes)}}},
            {"provenance",{{"maximumImageDimension",maybe(value.budgetObserved,limits.maximumImageDimensionSource)},
                {"frameSlots",maybe(value.budgetObserved,limits.frameSlotSource)},
                {"allocatedFrameBytes",maybe(value.budgetObserved,limits.allocatedFrameBytesSource)},
                {"observedCombinedTextureImageUnits",maybe(value.budgetObserved,limits.observedCombinedTextureImageUnits)},
                {"observedMaximumBufferLengthBytes",maybe(value.budgetObserved,limits.observedMaximumBufferLengthBytes)},
                {"observedRecommendedWorkingSetBytes",maybe(value.budgetObserved,limits.observedRecommendedWorkingSetBytes)}}}} },
        { "dimensions", {{"available",value.dimensionsObserved},
            {"requestedWidth",maybe(value.dimensionsObserved,value.requestedWidth)},
            {"requestedHeight",maybe(value.dimensionsObserved,value.requestedHeight)},
            {"actualWidth",maybe(value.dimensionsObserved,value.actualWidth)},
            {"actualHeight",maybe(value.dimensionsObserved,value.actualHeight)},
            {"dimensionMismatchCount",maybe(value.dimensionsObserved,value.dimensionMismatchCount)},
            {"halfResolutionMismatchCount",maybe(value.dimensionsObserved,value.halfResolutionMismatchCount)}} },
        { "recordingContentionDrops", value.recordingContentionDrops },
        { "layers", std::move(layers) }, { "nodes", std::move(nodes) }
    };
}
} // namespace videowire
