#pragma once

#include "VisualVolumeOperationContract.h"
#include <set>

namespace visualvolume
{
template <typename Plan>
inline bool isVolumePlan(const Plan& plan)
{
    return std::any_of(plan.nodeKinds.begin(), plan.nodeKinds.end(), [](const auto& kind) {
        return kind == sourceKind || kind == renderKind;
    }) || std::any_of(plan.operations.begin(), plan.operations.end(), [](const auto& operation) {
        return operation.kind == sourceKind || operation.kind == renderKind;
    });
}

// The same validation is used on receipt, during execution compilation, and
// before native preparation. Neither producerValidated nor cached frames grant
// authority to a different topology, resource claim, or density payload.
template <typename Plan>
inline bool validatePlan(const Plan& plan, Operation& destination, std::string& error)
{
    const auto reject = [&] { error = "Volume plan requires an exact canonical Density -> Render Volume -> Video Out graph"; return false; };
    if (plan.clipId <= 0 || plan.structuralRevision == 0 || plan.identityMode != "authoredGraph"
        || !plan.producerValidated || !plan.error.empty() || plan.nodeKinds.size() != 3
        || plan.nodeIds.size() != 3 || plan.operations.size() != 3 || plan.edges.size() != 2
        || plan.ports.size() != 4 || plan.descriptorCount != 3 || plan.operationCount != 3
        || plan.sceneRecordCount != 1 || plan.frameOutputCount != 1
        || plan.peakLiveFrameCount != 1 || plan.allocatedFrameSlotCount != 1) return reject();
    const auto source = std::find_if(plan.operations.begin(), plan.operations.end(), [](const auto& op) { return op.kind == sourceKind; });
    const auto render = std::find_if(plan.operations.begin(), plan.operations.end(), [](const auto& op) { return op.kind == renderKind; });
    const auto output = std::find_if(plan.operations.begin(), plan.operations.end(), [](const auto& op) { return op.kind == "video.out"; });
    if (source == plan.operations.end() || render == plan.operations.end() || output == plan.operations.end()
        || source->backendCapability != "control-eval" || render->backendCapability != "native-gpu"
        || output->backendCapability != "native-gpu" || !output->payloadXml.empty()
        || source->payloadXml != render->payloadXml) return reject();
    std::set<int> ids;
    for (const auto& operation : plan.operations)
    {
        if (operation.nodeId <= 0 || !ids.insert(operation.nodeId).second || !operation.runtimeGrantJson.empty()) return reject();
        const auto node = std::find(plan.nodeIds.begin(), plan.nodeIds.end(), operation.nodeId);
        if (node == plan.nodeIds.end() || plan.nodeKinds[static_cast<std::size_t>(node - plan.nodeIds.begin())] != operation.kind)
            return reject();
    }
    Operation candidate;
    if (!decode(render->payloadXml, candidate)
        || candidate.sourceStableId != static_cast<std::uint32_t>(source->nodeId) + 1u
        || candidate.renderStableId != static_cast<std::uint32_t>(render->nodeId) + 1u) return reject();
    const auto port = [&](int nodeId, int index, const char* direction, const char* carrier,
                          const char* type, const char* format, const char* color) {
        return std::count_if(plan.ports.begin(), plan.ports.end(), [&](const auto& p) {
            return p.nodeId == nodeId && p.port == index && p.channels == 1
                && p.direction == direction && p.carrier == carrier && p.dataType == type
                && p.pixelFormat == format && p.colorSpace == color;
        }) == 1;
    };
    const auto edge = [&](int from, int fromPort, int to, int toPort) {
        return std::count_if(plan.edges.begin(), plan.edges.end(), [&](const auto& e) {
            return e.fromNodeId == from && e.fromPort == fromPort && e.toNodeId == to && e.toPort == toPort;
        }) == 1;
    };
    if (!port(source->nodeId, 0, "out", "control", "volume", "unspecified", "unspecified")
        || !port(render->nodeId, 0, "in", "control", "volume", "unspecified", "unspecified")
        || !port(render->nodeId, 1, "out", "frame", "image", "rgba8", "sRGB")
        || !port(output->nodeId, 0, "in", "frame", "image", "rgba8", "sRGB")
        || !edge(source->nodeId, 0, render->nodeId, 0) || !edge(render->nodeId, 1, output->nodeId, 0)) return reject();
    destination = candidate;
    error.clear();
    return true;
}
} // namespace visualvolume
