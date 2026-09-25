#pragma once

#include "material_frame_evaluation.h"
#include "flat_shader_bridge.h"
#include "render_snapshot.h"
#include "../../shared/VisualTemporalOperationContract.h"
#include "../../shared/MaterialFrameSourceContract.h"

#include <map>
#include <set>

namespace videowire::materialframe
{
using Endpoint = surfacematerialbinding::TextureSlotBinding::GraphFrameEndpoint;
inline constexpr std::size_t maximumOperations = 32;
inline constexpr std::size_t maximumEvaluations = 256;
struct Operation
{
    Endpoint output;
    CompiledVisualOperation source;
    std::vector<Endpoint> inputs;
    ImmutableShaderOperationPlan shader;
    std::optional<visualtemporaloperation::Payload> temporal;
};
struct Schedule
{
    std::map<Endpoint,Operation> operations;
    std::vector<Endpoint> outputs;
};

inline bool sourceClip(const CompiledVisualLayerPlan& plan, Endpoint endpoint, int& clip, std::string& error)
{
    const auto operation = std::find_if(plan.operations.begin(),plan.operations.end(),
        [&](const auto& op) { return op.nodeId == endpoint.node; });
    if (endpoint.port != 0 || operation == plan.operations.end())
    { error = "Material Frame source endpoint is absent from its admitted graph"; return false; }
    if (operation->kind == "video.source" && operation->payloadXml.empty())
    { clip = plan.clipId; return true; }
    if (operation->kind == "video.layer.source" && materialframesource::decode(operation->payloadXml,clip)) return true;
    error = "Material Frame source is not an admitted decoded-media dependency"; return false;
}

template <typename ShaderAdmission>
std::optional<Schedule> admit(const CompiledVisualLayerPlan& plan, int render,
    const std::vector<Endpoint>& outputs, ShaderAdmission admitShader, std::string& error)
{
    const auto reject = [&](const char* reason) -> std::optional<Schedule> { error = reason; return std::nullopt; };
    if (outputs.empty() || outputs.size() > maximumOperations || plan.operations.size() > maximumOperations + 1
        || plan.nodeIds.size() != plan.operations.size() || plan.nodeKinds.size() != plan.operations.size())
        return reject("Material Frame schedule exceeds its operation or endpoint budget");
    Schedule result; result.outputs = outputs;
    std::set<int> nodes;
    std::size_t expectedPorts = 1, expectedEdges = outputs.size();
    const auto port = [&](int node,int id,const char* direction) {
        return std::count_if(plan.ports.begin(),plan.ports.end(),[&](const auto& p) {
            return p.nodeId == node && p.port == id && p.direction == direction && p.channels == 1
                && p.carrier == "frame" && p.dataType == "image" && p.pixelFormat == "rgba8" && p.colorSpace == "sRGB";
        }) == 1;
    };
    if (!port(render,1,"out")) return reject("Material Frame schedule lost its Geometry Image output");
    for (std::size_t index = 0; index < plan.operations.size(); ++index)
    {
        const auto& op = plan.operations[index];
        if (!nodes.insert(op.nodeId).second || op.nodeId != plan.nodeIds[index] || op.kind != plan.nodeKinds[index])
            return reject("Material Frame schedule has duplicate or mismatched operation identities");
        if (op.nodeId == render) continue;
        const bool source = op.kind == "video.source" || op.kind == "video.layer.source";
        const bool generator = op.kind == "visual.shader.generator" || op.kind == "visual.shader.custom";
        const bool filter = op.kind == "visual.shader.filter";
        const bool transition = op.kind == shadertransition::operationKind;
        const bool temporal = visualtemporaloperation::isTemporalKind(op.kind);
        if ((!source && !generator && !filter && !transition && !temporal)
            || op.backendCapability != (source ? "source-decode" : "native-gpu")
            || (op.kind == "video.source" && !op.payloadXml.empty()))
            return reject("Material Frame schedule contains an unavailable producer or source override");
        Operation operation; operation.source = op;
        int sourceClip = 0;
        if (op.kind == "video.layer.source" && !materialframesource::decode(op.payloadXml,sourceClip))
            return reject("Material Frame Layer Source requires an exact saved project clip identity");
        operation.output = {op.nodeId,transition ? 2 : filter || temporal ? 1 : 0};
        const int inputs = transition ? 2 : filter || temporal ? 1 : 0;
        if (!port(op.nodeId,operation.output.port,"out")) return reject("Material Frame producer has an incompatible Image output");
        for (int input = 0; input < inputs; ++input)
        {
            const CompiledVisualEdgeBinding* connection = nullptr;
            for (const auto& edge : plan.edges) if (edge.toNodeId == op.nodeId && edge.toPort == input)
            { if (connection) return reject("Material Frame input is multiply bound"); connection = &edge; }
            if (!connection || !port(op.nodeId,input,"in")) return reject("Material Frame producer has no exact Image input");
            operation.inputs.push_back({connection->fromNodeId,connection->fromPort});
        }
        expectedPorts += inputs + 1; expectedEdges += inputs;
        if (temporal)
        {
            visualtemporaloperation::Payload payload;
            if (!visualtemporaloperation::parseForKind(op.kind,op.payloadXml,payload,&error)) return std::nullopt;
            if (payload.reset != 0)
                return reject("Surface Frame temporal reset requires the shared reset-history scheduler");
            operation.temporal = payload;
        }
        else if (!source && !admitShader(operation,error)) return std::nullopt;
        result.operations.emplace(operation.output,std::move(operation));
    }
    for (std::size_t index = 0; index < outputs.size(); ++index)
    {
        const auto endpoint = outputs[index];
        const int input = 12 + static_cast<int>(index);
        if (!port(render,input,"in") || std::count_if(plan.edges.begin(),plan.edges.end(),[&](const auto& edge) {
            return edge.fromNodeId == endpoint.node && edge.fromPort == endpoint.port
                && edge.toNodeId == render && edge.toPort == input;
        }) != 1) return reject("Material Frame endpoint does not match its exact Render dependency");
        ++expectedPorts;
    }
    if (plan.ports.size() != expectedPorts || plan.edges.size() != expectedEdges)
        return reject("Material Frame schedule contains extra ports or edges");
    std::set<Endpoint> visiting, visited;
    std::function<bool(Endpoint)> visit = [&](Endpoint endpoint) {
        if (visited.count(endpoint)) return true;
        const auto found = result.operations.find(endpoint);
        if (found == result.operations.end() || !visiting.insert(endpoint).second) return false;
        for (const auto input : found->second.inputs) if (!visit(input)) return false;
        visiting.erase(endpoint); visited.insert(endpoint); return true;
    };
    for (const auto output : outputs) if (!visit(output)) return reject("Material Frame schedule has a cycle or unknown endpoint");
    if (visited.size() != result.operations.size()) return reject("Material Frame schedule contains an unused producer");
    error.clear(); return result;
}
} // namespace videowire::materialframe
