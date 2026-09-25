#pragma once

#include "render_snapshot.h"
#include "../../shared/TypedScenePassContract.h"

#include <algorithm>
#include <optional>

namespace videowire
{
inline bool hasTypedScenePass(const CompiledVisualLayerPlan& plan)
{
    return std::any_of(plan.operations.begin(), plan.operations.end(), [](const auto& operation)
    {
        return operation.kind == typedscenepass::kNodeKind
            && operation.payloadXml.compare(0, typedscenepass::kHeader.size(), typedscenepass::kHeader) == 0;
    });
}

// Reduce only the exact producer-authored typed schedule. The ordinary native
// Scene3D compiler still validates the embedded request and retained source.
inline bool lowerTypedScenePass(const CompiledVisualLayerPlan& plan,
                               typedscenepass::Payload& payload,
                               std::optional<aovinspection::Payload>& inspection,
                               CompiledVisualLayerPlan& base, std::string& error)
{
    const auto reject = [&] { error = "Render Passes requires its exact retained scene and typed inspection schedule"; return false; };
    if (!plan.producerValidated || !plan.error.empty() || plan.structuralRevision == 0
        || plan.nodeIds.size() != plan.operations.size() || plan.nodeKinds.size() != plan.operations.size()
        || (plan.operations.size() != 2 && plan.operations.size() != 3)
        || !typedscenepass::decode(plan.operations[1].payloadXml, payload)) return reject();
    const bool inspect = payload.output != renderpassoutput::Output::Color;
    const auto count = inspect ? 3u : 2u;
    if (plan.operations.size() != count || plan.ports.size() != (inspect ? 5u : 3u)
        || plan.edges.size() != (inspect ? 2u : 1u)
        || payload.scene.structuralRevision != plan.structuralRevision) return reject();
    const int source = static_cast<int>(payload.scene.sourceStableId - 1u);
    const int render = static_cast<int>(payload.scene.renderStableId - 1u);
    for (std::size_t i = 0; i < count; ++i)
        if (plan.operations[i].nodeId < 0 || plan.nodeIds[i] != plan.operations[i].nodeId
            || plan.nodeKinds[i] != plan.operations[i].kind) return reject();
    if (plan.operations[0].nodeId != source || plan.operations[0].kind != typedscenepass::kSceneKind
        || plan.operations[0].backendCapability != "control-eval" || !plan.operations[0].payloadXml.empty()
        || plan.operations[1].nodeId != render || plan.operations[1].kind != typedscenepass::kNodeKind
        || plan.operations[1].backendCapability != "native-gpu") return reject();
    const auto port = [&](int node, int id, const char* direction, const char* carrier,
                          const char* type, const char* format, const char* space)
    {
        return std::count_if(plan.ports.begin(), plan.ports.end(), [&](const auto& p) {
            return p.nodeId == node && p.port == id && p.direction == direction && p.channels == 1
                && p.carrier == carrier && p.dataType == type && p.pixelFormat == format && p.colorSpace == space;
        }) == 1;
    };
    const auto edge = [&](int from, int fromPort, int to, int toPort)
    {
        return std::count_if(plan.edges.begin(), plan.edges.end(), [&](const auto& e) {
            return e.fromNodeId == from && e.fromPort == fromPort && e.toNodeId == to && e.toPort == toPort;
        }) == 1;
    };
    const auto type = payload.output == renderpassoutput::Output::Color ? "image"
        : payload.output == renderpassoutput::Output::Depth ? "depth" : "normal";
    const auto format = payload.output == renderpassoutput::Output::Depth ? "r32f" : "rgba16f";
    const auto space = inspect ? "unspecified" : "linearSRGB";
    const auto outputPort = typedscenepass::outputPort(payload.output);
    if (!port(source, 0, "out", "control", "scene3D", "unspecified", "unspecified")
        || !port(render, 0, "in", "control", "scene3D", "unspecified", "unspecified")
        || !port(render, outputPort, "out", "frame", type, format, space)
        || !edge(source, 0, render, 0)) return reject();
    inspection.reset();
    if (inspect)
    {
        const auto& op = plan.operations[2];
        aovinspection::Payload value;
        if (op.nodeId == source || op.nodeId == render || op.backendCapability != "native-gpu"
            || !aovinspection::parse(op.payloadXml, value)
            || aovinspection::serialize(value) != op.payloadXml
            || aovinspection::output(value.source) != payload.output || value.extent != payload.extent
            || !renderpasscomposite::valid(typedscenepass::inspectionParameters(value))
            || op.kind != aovinspection::operationKind(value.source)
            || !port(op.nodeId, 0, "in", "frame", type, format, space)
            || !port(op.nodeId, 1, "out", "frame", "image", "rgba16f", "linearSRGB")
            || !edge(render, outputPort, op.nodeId, 0)) return reject();
        inspection = value;
    }
    base = plan;
    base.nodeIds = {source, render};
    base.nodeKinds = {std::string(typedscenepass::kSceneKind), std::string(visualimportedscenerender::kRenderNodeKind)};
    base.operations = {{source, base.nodeKinds[0], "control-eval", {}},
        {render, base.nodeKinds[1], "native-gpu", visualimportedscenerender::encode(payload.scene)}};
    base.ports = {{source, 0, 1, "out", "control", "scene3D", "unspecified", "unspecified"},
        {render, 0, 1, "in", "control", "scene3D", "unspecified", "unspecified"},
        {render, 1, 1, "out", "frame", "image", "rgba8", "sRGB"}};
    base.edges = {{source, 0, render, 0}};
    error.clear();
    return true;
}
} // namespace videowire
