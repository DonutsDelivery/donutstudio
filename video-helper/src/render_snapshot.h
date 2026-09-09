#pragma once

#include "source_binding.h"
#include "matte_cache.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <map>
#include <vector>

namespace videowire
{
struct RenderSegment
{
    std::string sourcePath;
    SourceKind sourceKind = SourceKind::Unspecified;
    int clipId = 0;
    int trackLayer = 0;
    bool isAdjustment = false;
    double inSec = 0.0;
    double outSec = 0.0;
    double rate = 1.0;
    double displayStartSec = 0.0;
    double clockStartSec = -1.0;
    double clockDurationSec = 0.0;

    int retimeQuality = 0;
    int transitionType = 0;
    double transitionDurationSec = 0.0;
    int transitionFromClipId = -1;
    int transitionToClipId = -1;
    double sourceFps = 0.0;
    int seqStart = -1;
    std::string shaderSource;
    std::map<std::string, std::string> genImages;
    std::string matteAssetId;
    std::string matteAssetVersion;
    std::string matteContentReceipt;
    std::string matteState;
    std::string matteCacheKey;
    std::string matteDir; // helper-resolved only; never accepted from the wire
    std::string matteTrustedRoot; // helper-resolved only; retained for lazy-open revalidation
    uint64_t matteContentRevision = 0; // helper-resolved wire revision
    std::string matteFramePrefix = "matte_";
    std::string matteFrameExtension = ".png";
    int matteFirstFrame = 1;
    int matteFrameDigits = 6;
    double matteFps = 0.0;
    int matteFrames = 0;
};

struct RawRenderSegment : RenderSegment
{
    bool hasStructuredSourceKind = false;
    bool hasStructuredTransitionBinding = false;
};

struct CompiledVisualEdgeBinding
{
    int fromNodeId = 0;
    int fromPort = 0;
    int toNodeId = 0;
    int toPort = 0;
};

struct CompiledVisualPortBinding
{
    int nodeId = 0;
    int port = 0;
    int channels = 1;
    std::string direction;
    std::string carrier;
    std::string dataType;
    std::string pixelFormat;
    std::string colorSpace;
};

struct CompiledVisualOperation
{
    int nodeId = 0;
    std::string kind;
    std::string backendCapability;
    std::string payloadXml;
    // Exact JSON object received over the live processor/helper transport.
    // Render snapshots are in-memory products, not project persistence.
    std::string runtimeGrantJson;

    CompiledVisualOperation() = default;
    CompiledVisualOperation(int id, std::string operationKind,
                            std::string capability, std::string payload,
                            std::string grant = {})
        : nodeId(id), kind(std::move(operationKind)),
          backendCapability(std::move(capability)), payloadXml(std::move(payload)),
          runtimeGrantJson(std::move(grant))
    {
    }
};

inline bool isParameterlessShapeBooleanKind (const std::string& kind) noexcept
{
    return kind == "visual.shape.union" || kind == "visual.shape.intersection"
        || kind == "visual.shape.subtract";
}

inline bool isParameterlessPathMatteKind (const std::string& kind) noexcept
{
    return kind == "visual.shape.invert" || kind == "visual.matte.path";
}

struct CompiledVisualLayerPlan
{
    int clipId = 0;
    uint64_t structuralRevision = 0;
    std::string identityMode = "authoredGraph";
    bool producerValidated = false;
    std::string error;
    size_t descriptorCount = 0;
    size_t operationCount = 0;
    size_t sceneRecordCount = 0;
    size_t frameOutputCount = 0;
    size_t peakLiveFrameCount = 0;
    size_t allocatedFrameSlotCount = 0;
    uint64_t compileDurationMicros = 0;
    std::vector<std::string> nodeKinds;
    std::vector<int> nodeIds;
    std::vector<CompiledVisualEdgeBinding> edges;
    std::vector<CompiledVisualPortBinding> ports;
    std::vector<CompiledVisualOperation> operations;
};

// Select the strictly newest revision for a clip. If revisions are equal,
// select the first plan. This predicate is the canonical normalization rule.
inline bool isSelectedVisualLayerPlan (
    const std::vector<CompiledVisualLayerPlan>& plans, std::size_t index)
{
    for (std::size_t candidate = 0; candidate < plans.size(); ++candidate)
    {
        if (candidate == index || plans[candidate].clipId != plans[index].clipId)
            continue;
        if (plans[candidate].structuralRevision > plans[index].structuralRevision
            || (plans[candidate].structuralRevision == plans[index].structuralRevision
                && candidate < index))
            return false;
    }
    return true;
}

inline std::vector<std::size_t> normalizedVisualLayerPlanIndices (
    const std::vector<CompiledVisualLayerPlan>& plans)
{
    std::vector<std::size_t> selected;
    selected.reserve(plans.size());
    for (std::size_t ownerPosition = 0; ownerPosition < plans.size(); ++ownerPosition)
    {
        const bool firstOwnerPosition = std::none_of(
            plans.begin(), plans.begin() + static_cast<std::ptrdiff_t>(ownerPosition),
            [&] (const auto& prior) { return prior.clipId == plans[ownerPosition].clipId; });
        if (! firstOwnerPosition)
            continue;
        for (std::size_t index = ownerPosition; index < plans.size(); ++index)
            if (plans[index].clipId == plans[ownerPosition].clipId
                && isSelectedVisualLayerPlan(plans, index))
            {
                selected.push_back(index);
                break;
            }
    }
    return selected;
}

inline std::vector<CompiledVisualLayerPlan> normalizedVisualLayerPlans (
    const std::vector<CompiledVisualLayerPlan>& plans)
{
    const auto selected = normalizedVisualLayerPlanIndices(plans);
    std::vector<CompiledVisualLayerPlan> normalized;
    normalized.reserve(selected.size());
    for (const auto index : selected)
        normalized.push_back(plans[index]);
    return normalized;
}

inline const CompiledVisualLayerPlan* findVisualLayerPlan (
    const std::vector<CompiledVisualLayerPlan>& plans, int clipId)
{
    for (std::size_t index = 0; index < plans.size(); ++index)
        if (plans[index].clipId == clipId && isSelectedVisualLayerPlan(plans, index))
            return &plans[index];
    return nullptr;
}

struct CompiledVisualInputBinding
{
    int fromNodeId = 0;
    int fromPort = 0;
    int toPort = 0;
};

struct CompiledVisualScheduledOperation
{
    size_t nodeIndex = 0;
    size_t operationIndex = 0;
    int nodeId = 0;
    std::vector<CompiledVisualInputBinding> inputs;
};

struct CompiledVisualDagSchedule
{
    std::vector<CompiledVisualScheduledOperation> operations;
};

inline bool isExactThreeSourceVisualDag (const std::vector<std::string>& kinds)
{
    return kinds.size() == 5
        && std::count(kinds.begin(), kinds.end(), "video.source") == 1
        && std::count(kinds.begin(), kinds.end(), "video.text") == 1
        && std::count(kinds.begin(), kinds.end(), "video.layer.source") == 1
        && std::count(kinds.begin(), kinds.end(), "video.blend") == 1
        && std::count(kinds.begin(), kinds.end(), "video.out") == 1;
}

inline bool isBoundedNativeCompositorDag (const std::vector<std::string>& kinds)
{
    return std::count(kinds.begin(), kinds.end(), "video.blend") == 1
        && std::count(kinds.begin(), kinds.end(), "video.out") == 1;
}

inline bool admitBoundedNativeImageDescriptors (const CompiledVisualLayerPlan& plan,
                                                std::string& error)
{
    const auto blend = std::find(plan.nodeKinds.begin(), plan.nodeKinds.end(), "video.blend");
    const auto output = std::find(plan.nodeKinds.begin(), plan.nodeKinds.end(), "video.out");
    if (blend == plan.nodeKinds.end() || output == plan.nodeKinds.end()) return true;
    const int blendId = plan.nodeIds[(size_t) std::distance(plan.nodeKinds.begin(), blend)];
    const int outputId = plan.nodeIds[(size_t) std::distance(plan.nodeKinds.begin(), output)];
    std::vector<std::pair<int, int>> imagePorts {
        { blendId, 0 }, { blendId, 1 }, { blendId, 2 }, { blendId, 3 }, { outputId, 0 }
    };
    const auto mark = [&imagePorts] (int nodeId, int port)
    {
        const std::pair<int, int> identity { nodeId, port };
        if (std::find(imagePorts.begin(), imagePorts.end(), identity) == imagePorts.end())
            imagePorts.push_back(identity);
    };
    for (const int blendInput : { 0, 2, 3 })
    {
        const auto edge = std::find_if(plan.edges.begin(), plan.edges.end(), [&](const auto& candidate)
            { return candidate.toNodeId == blendId && candidate.toPort == blendInput; });
        if (edge != plan.edges.end()) mark(edge->fromNodeId, edge->fromPort);
    }
    int successor = blendId;
    for (size_t depth = 0; depth < plan.nodeIds.size(); ++depth)
    {
        const auto edge = std::find_if(plan.edges.begin(), plan.edges.end(), [&](const auto& candidate)
            { return candidate.toNodeId == successor && candidate.toPort == 0; });
        if (edge == plan.edges.end()) break;
        mark(edge->toNodeId, edge->toPort);
        mark(edge->fromNodeId, edge->fromPort);
        successor = edge->fromNodeId;
    }
    for (const auto& port : plan.ports)
        if (std::find(imagePorts.begin(), imagePorts.end(),
                      std::pair<int, int> { port.nodeId, port.port }) != imagePorts.end()
            && (port.channels != 1 || port.carrier != "frame"
                || port.dataType != "image" || port.pixelFormat != "rgba8"
                || port.colorSpace != "sRGB"))
        {
            error = "bounded visual DAG requires RGBA8 sRGB Frame<Image> descriptors";
            return false;
        }
    return true;
}

// Schedule typed native-compositor DAGs independently of serialized document
// order. Stable node identities break ready-node ties. Destination-port
// identities determine input routing.
inline bool compileBoundedVisualDagSchedule (const CompiledVisualLayerPlan& plan,
                                             CompiledVisualDagSchedule& schedule,
                                             std::string& error)
{
    schedule.operations.clear();
    const size_t nodeCount = plan.nodeIds.size();
    const bool exactThreeSource = isExactThreeSourceVisualDag(plan.nodeKinds);
    if ((! exactThreeSource
            && ! isBoundedNativeCompositorDag(plan.nodeKinds))
        || nodeCount != plan.nodeKinds.size() || plan.operations.size() != nodeCount)
    {
        error = "bounded visual DAG has incomplete node or operation identities";
        return false;
    }
    if (! admitBoundedNativeImageDescriptors(plan, error)) return false;

    std::vector<size_t> operationForNode(nodeCount, nodeCount);
    for (size_t nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex)
    {
        if (plan.nodeIds[nodeIndex] <= 0
            || std::find(plan.nodeIds.begin(), plan.nodeIds.begin()
                         + static_cast<std::ptrdiff_t>(nodeIndex), plan.nodeIds[nodeIndex])
                   != plan.nodeIds.begin() + static_cast<std::ptrdiff_t>(nodeIndex))
        {
            error = "bounded visual DAG has duplicate or invalid node identity";
            return false;
        }
    }
    for (size_t operationIndex = 0; operationIndex < plan.operations.size(); ++operationIndex)
    {
        const auto& operation = plan.operations[operationIndex];
        const auto node = std::find(plan.nodeIds.begin(), plan.nodeIds.end(), operation.nodeId);
        if (node == plan.nodeIds.end())
        {
            error = "bounded visual DAG operation references an unknown node";
            return false;
        }
        const size_t nodeIndex = static_cast<size_t>(std::distance(plan.nodeIds.begin(), node));
        if (operationForNode[nodeIndex] != nodeCount || operation.kind != plan.nodeKinds[nodeIndex])
        {
            error = "bounded visual DAG operation identity does not match its node";
            return false;
        }
        if (isParameterlessShapeBooleanKind(operation.kind) && ! operation.payloadXml.empty())
        {
            error = "visual Shape Boolean operation payload must be empty";
            return false;
        }
        if (isParameterlessPathMatteKind(operation.kind) && ! operation.payloadXml.empty())
        {
            error = "visual path-matte operation payload must be empty";
            return false;
        }
        operationForNode[nodeIndex] = operationIndex;
    }

    std::map<std::pair<int, int>, const CompiledVisualPortBinding*> ports;
    for (const auto& port : plan.ports)
    {
        if (std::find(plan.nodeIds.begin(), plan.nodeIds.end(), port.nodeId) == plan.nodeIds.end()
            || port.port < 0 || port.channels <= 0
            || (port.direction != "in" && port.direction != "out")
            || port.carrier.empty() || port.dataType.empty()
            || ! ports.emplace(std::make_pair(port.nodeId, port.port), &port).second)
        {
            error = "bounded visual DAG has an invalid or duplicate typed port";
            return false;
        }
    }

    std::vector<std::vector<size_t>> outgoing(nodeCount);
    std::vector<size_t> indegree(nodeCount, 0);
    std::vector<std::vector<CompiledVisualInputBinding>> inputs(nodeCount);
    for (size_t edgeIndex = 0; edgeIndex < plan.edges.size(); ++edgeIndex)
    {
        const auto& edge = plan.edges[edgeIndex];
        const auto fromNode = std::find(plan.nodeIds.begin(), plan.nodeIds.end(), edge.fromNodeId);
        const auto toNode = std::find(plan.nodeIds.begin(), plan.nodeIds.end(), edge.toNodeId);
        if (fromNode == plan.nodeIds.end() || toNode == plan.nodeIds.end()
            || edge.fromNodeId == edge.toNodeId || edge.fromPort < 0 || edge.toPort < 0)
        {
            error = "bounded visual DAG edge references an invalid endpoint";
            return false;
        }
        if (std::any_of(plan.edges.begin(), plan.edges.begin()
                        + static_cast<std::ptrdiff_t>(edgeIndex),
                        [&edge](const CompiledVisualEdgeBinding& prior)
                        {
                            return (prior.fromNodeId == edge.fromNodeId && prior.fromPort == edge.fromPort
                                    && prior.toNodeId == edge.toNodeId && prior.toPort == edge.toPort)
                                || (prior.toNodeId == edge.toNodeId && prior.toPort == edge.toPort);
                        }))
        {
            error = "bounded visual DAG has a duplicate edge or multiply-bound input port";
            return false;
        }
        const auto sourcePort = ports.find({ edge.fromNodeId, edge.fromPort });
        const auto destinationPort = ports.find({ edge.toNodeId, edge.toPort });
        if (! plan.ports.empty()
            && (sourcePort == ports.end() || destinationPort == ports.end()
                || sourcePort->second->direction != "out" || destinationPort->second->direction != "in"
                || sourcePort->second->carrier != destinationPort->second->carrier
                || sourcePort->second->dataType != destinationPort->second->dataType
                || sourcePort->second->channels != destinationPort->second->channels
                || sourcePort->second->pixelFormat != destinationPort->second->pixelFormat
                || sourcePort->second->colorSpace != destinationPort->second->colorSpace))
        {
            error = "bounded visual DAG has an incompatible typed edge";
            return false;
        }
        const size_t fromIndex = static_cast<size_t>(std::distance(plan.nodeIds.begin(), fromNode));
        const size_t toIndex = static_cast<size_t>(std::distance(plan.nodeIds.begin(), toNode));
        outgoing[fromIndex].push_back(toIndex);
        ++indegree[toIndex];
        inputs[toIndex].push_back({ edge.fromNodeId, edge.fromPort, edge.toPort });
    }

    std::vector<bool> emitted(nodeCount, false);
    schedule.operations.reserve(nodeCount);
    for (size_t emittedCount = 0; emittedCount < nodeCount; ++emittedCount)
    {
        size_t ready = nodeCount;
        for (size_t nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex)
            if (! emitted[nodeIndex] && indegree[nodeIndex] == 0
                && (ready == nodeCount || plan.nodeIds[nodeIndex] < plan.nodeIds[ready]))
                ready = nodeIndex;
        if (ready == nodeCount)
        {
            error = "bounded visual DAG contains a cycle";
            schedule.operations.clear();
            return false;
        }
        emitted[ready] = true;
        auto orderedInputs = inputs[ready];
        std::stable_sort(orderedInputs.begin(), orderedInputs.end(),
            [](const auto& left, const auto& right)
            {
                if (left.toPort != right.toPort) return left.toPort < right.toPort;
                if (left.fromNodeId != right.fromNodeId) return left.fromNodeId < right.fromNodeId;
                return left.fromPort < right.fromPort;
            });
        schedule.operations.push_back(
            { ready, operationForNode[ready], plan.nodeIds[ready], std::move(orderedInputs) });
        for (const auto destination : outgoing[ready]) --indegree[destination];
    }

    const auto outputKind = std::find(plan.nodeKinds.begin(), plan.nodeKinds.end(), "video.out");
    const size_t outputIndex = static_cast<size_t>(std::distance(plan.nodeKinds.begin(), outputKind));
    std::vector<bool> reachesOutput(nodeCount, false);
    reachesOutput[outputIndex] = true;
    for (auto it = schedule.operations.rbegin(); it != schedule.operations.rend(); ++it)
        for (const auto destination : outgoing[it->nodeIndex])
            reachesOutput[it->nodeIndex] = reachesOutput[it->nodeIndex] || reachesOutput[destination];
    if (std::find(reachesOutput.begin(), reachesOutput.end(), false) != reachesOutput.end())
    {
        error = "bounded visual DAG contains a disconnected operation";
        schedule.operations.clear();
        return false;
    }
    error.clear();
    return true;
}

struct VisualTriggerBinding
{
    int sampleOffset = 0;
    double timelineBeat = 0.0;
    float strength = 0.0f;
    uint64_t sequence = 0;
};

struct VisualEventScheduleBinding
{
    int clipId = 0;
    int nodeId = 0;
    int portId = -1;
    uint64_t sessionRevision = 0;
    std::vector<VisualTriggerBinding> triggers;
};

struct RevisionTuple
{
    uint64_t authoring = 0;
    uint64_t accepted = 0;
    uint64_t rejected = 0;
    uint64_t compiled = 0;
    uint64_t lastGood = 0;
    uint64_t exportable = 0;
    uint64_t evaluation = 0;
};

struct ResolvedVisualSnapshot
{
    std::vector<RenderSegment> segments;
    std::vector<CompiledVisualLayerPlan> visualLayerPlans;
    std::vector<VisualEventScheduleBinding> visualEventSchedules;
    uint64_t authoringRevision = 0;
};

inline videohelper::MatteCacheBinding matteCacheBinding (const RenderSegment& segment)
{
    videohelper::MatteCacheBinding binding;
    binding.key = segment.matteCacheKey;
    binding.receipt = segment.matteContentReceipt;
    binding.version = segment.matteAssetVersion;
    binding.prefix = segment.matteFramePrefix;
    binding.extension = segment.matteFrameExtension;
    binding.firstFrame = segment.matteFirstFrame;
    binding.digits = segment.matteFrameDigits;
    binding.frames = segment.matteFrames;
    binding.fps = segment.matteFps;
    binding.contentRevision = segment.matteContentRevision;
    return binding;
}

/** Shared admission guard used immediately before every lazy matte decoder open. */
inline bool revalidateMatteForOpen (const RenderSegment& segment,
                                    std::string& resolvedDirectory,
                                    std::string& pattern,
                                    std::string& error)
{
    if (segment.matteAssetId.empty()) { resolvedDirectory.clear(); pattern.clear(); return true; }
    if (segment.matteContentRevision == 0 || segment.matteTrustedRoot.empty())
    {
        error = "matte cache authority was not retained for lazy open";
        return false;
    }
    auto binding = matteCacheBinding(segment);
    if (! videohelper::validateMatteCache(segment.matteTrustedRoot, binding, resolvedDirectory, error))
        return false;
    pattern = resolvedDirectory + "/" + binding.prefix + "%0" + std::to_string(binding.digits)
        + "d" + binding.extension;
    return true;
}

inline bool validateCompiledVisualLayerPlans (
    const std::vector<RenderSegment>& segments,
    const std::vector<CompiledVisualLayerPlan>& plans,
    bool required,
    std::string& error)
{
    if (! required && plans.empty()) return true;
    if (plans.empty() && ! segments.empty())
    {
        error = "visual snapshot is missing compiled layer plans";
        return false;
    }
    const std::vector<std::string> fixedKinds {
        "video.legacy.source", "video.legacy.retime", "video.legacy.transform",
        "video.legacy.effects", "video.out"
    };
    auto generatorKinds = fixedKinds;
    generatorKinds[0] = "video.legacy.generator";
    const std::vector<std::string> editableTransformKinds {
        "video.source", "video.transform", "video.out"
    };
    const std::vector<std::string> editableEffectsKinds {
        "video.source", "video.transform", "video.effects", "video.out"
    };
    const std::vector<std::string> editableMaskKinds {
        "video.source", "video.transform", "video.effects", "video.mask.shape", "video.out"
    };
    const std::vector<std::string> editableBlendKinds {
        "video.source", "video.transform", "video.effects", "video.mask.shape",
        "video.blend", "video.out"
    };
    const std::vector<std::string> editableTextKinds {
        "video.source", "video.transform", "video.effects", "video.mask.shape",
        "video.text", "video.blend", "video.out"
    };
    const std::vector<std::string> editableLayerKinds {
        "video.source", "video.transform", "video.effects", "video.mask.shape",
        "video.text", "video.layer.source", "video.blend", "video.out"
    };
    const std::vector<std::string> editableDirectKinds {
        "video.source", "video.out"
    };
    for (size_t i = 0; i < plans.size(); ++i)
    {
        const auto& plan = plans[i];
        if (plan.clipId <= 0 || ! plan.producerValidated)
        {
            error = plan.error.empty() ? "visual layer plan failed producer validation"
                                       : plan.error;
            return false;
        }
        const bool hasTypedBindings = ! plan.nodeIds.empty() || ! plan.edges.empty()
                                   || ! plan.ports.empty();
        const bool exactThreeSourceDag = isExactThreeSourceVisualDag(plan.nodeKinds);
        const bool scheduledNativeDag = exactThreeSourceDag
            || isBoundedNativeCompositorDag(plan.nodeKinds);
        const bool importedSceneDag = std::count(
                plan.nodeKinds.begin(), plan.nodeKinds.end(),
                "visual.3d.imported-animation") == 1
            && std::count(plan.nodeKinds.begin(), plan.nodeKinds.end(),
                          "visual.3d.render") == 1;
        const bool nativeSdfPlan = std::find(plan.nodeKinds.begin(), plan.nodeKinds.end(),
            "visual.sdf.raymarch") != plan.nodeKinds.end();
        if (hasTypedBindings)
        {
            if (scheduledNativeDag)
            {
                CompiledVisualDagSchedule schedule;
                if (! compileBoundedVisualDagSchedule(plan, schedule, error))
                    return false;
            }
            if (plan.nodeIds.size() != plan.nodeKinds.size() || plan.ports.empty())
            {
                error = "visual layer plan has incomplete typed bindings";
                return false;
            }
            for (size_t nodeIndex = 0; nodeIndex < plan.nodeIds.size(); ++nodeIndex)
            {
                const auto nodeId = plan.nodeIds[nodeIndex];
                const auto before = plan.nodeIds.begin() + static_cast<std::ptrdiff_t>(nodeIndex);
                if (nodeId <= 0 || std::find(plan.nodeIds.begin(), before, nodeId) != before)
                {
                    error = "visual layer plan has invalid stable node identities";
                    return false;
                }
            }
            for (size_t portIndex = 0; portIndex < plan.ports.size(); ++portIndex)
            {
                const auto& port = plan.ports[portIndex];
                const auto nodePosition = std::find(plan.nodeIds.begin(), plan.nodeIds.end(), port.nodeId);
                const bool depthTombstone = nodePosition != plan.nodeIds.end()
                    && plan.nodeKinds[(size_t) std::distance(plan.nodeIds.begin(), nodePosition)] == "visual.depth.asset"
                    && port.port == 0 && port.channels == 0 && port.direction == "out"
                    && port.carrier == "none";
                if (nodePosition == plan.nodeIds.end()
                    || port.port < 0 || (port.channels <= 0 && ! depthTombstone)
                    || (port.direction != "in" && port.direction != "out")
                    || port.carrier.empty() || (port.carrier == "none" && ! depthTombstone)
                    || (! depthTombstone && (port.dataType.empty() || port.dataType == "unspecified"))
                    || std::any_of(plan.ports.begin(),
                                   plan.ports.begin() + static_cast<std::ptrdiff_t>(portIndex),
                                   [&port](const CompiledVisualPortBinding& prior)
                                   { return prior.nodeId == port.nodeId && prior.port == port.port; }))
                {
                    error = "visual layer plan has invalid typed port descriptors";
                    return false;
                }
            }
            if (! plan.operations.empty())
            {
                if (plan.operations.size() != plan.nodeIds.size())
                {
                    error = "visual layer plan has incomplete compiled operations";
                    return false;
                }
                for (size_t operationIndex = 0; operationIndex < plan.operations.size(); ++operationIndex)
                {
                    const auto& operation = plan.operations[operationIndex];
                    const bool producesFrame = std::any_of(plan.ports.begin(), plan.ports.end(),
                        [&operation](const CompiledVisualPortBinding& port)
                        { return port.nodeId == operation.nodeId && port.direction == "out"
                              && port.carrier == "frame"; });
                    const bool isDecodedVisualResource = ! producesFrame
                        && (operation.kind == "visual.3d.imported-animation"
                            || operation.kind == "visual.matte.asset"
                            || operation.kind == "visual.depth.asset");
                    const bool isNativeGpuControlResource = ! producesFrame
                        && operation.kind == "visual.deformation.imported";
                    const bool isVisualControl = (operation.kind.rfind("visual.", 0) == 0
                        || operation.kind.rfind("tracking.", 0) == 0)
                        && ! producesFrame && ! isDecodedVisualResource
                        && ! isNativeGpuControlResource;
                    const bool capabilityMatchesSemantics =
                        (! producesFrame || operation.backendCapability != "control-eval")
                        && (! isVisualControl || operation.backendCapability == "control-eval")
                        && (! isDecodedVisualResource
                            || operation.backendCapability == "source-decode")
                        && (! isNativeGpuControlResource
                            || operation.backendCapability == "native-gpu");
                    if ((! scheduledNativeDag
                            && (operation.nodeId != plan.nodeIds[operationIndex]
                                || operation.kind != plan.nodeKinds[operationIndex]))
                        || (operation.backendCapability != "source-decode"
                            && operation.backendCapability != "native-gpu"
                            && operation.backendCapability != "control-eval")
                        || ! capabilityMatchesSemantics)
                    {
                        error = "visual layer plan operation identity or backend admission is invalid";
                        return false;
                    }
                }
            }
            for (const auto& edge : plan.edges)
            {
                const auto findPort = [&plan](int nodeId, int portIndex)
                {
                    return std::find_if(plan.ports.begin(), plan.ports.end(),
                        [=](const CompiledVisualPortBinding& port)
                        { return port.nodeId == nodeId && port.port == portIndex; });
                };
                const auto from = findPort(edge.fromNodeId, edge.fromPort);
                const auto to = findPort(edge.toNodeId, edge.toPort);
                if (from == plan.ports.end() || to == plan.ports.end()
                    || from->direction != "out" || to->direction != "in"
                    || from->carrier != to->carrier || from->dataType != to->dataType
                    || from->channels != to->channels
                    || from->pixelFormat != to->pixelFormat
                    || from->colorSpace != to->colorSpace)
                {
                    error = "visual layer plan has an incompatible typed edge binding";
                    return false;
                }
            }
            const std::vector<std::string> supportedKinds {
                "video.legacy.source", "video.legacy.generator", "video.legacy.retime",
                "video.legacy.transform", "video.legacy.effects", "video.source",
                "video.transform", "video.effects", "video.mask.shape", "video.text",
                "video.layer.source", "video.blend", "video.out", "visual.shader.transition",
                "visual.shader.custom", "visual.shader.filter", "visual.effect.sharpen", "visual.effect.blur",
                "visual.effect.vignette", "visual.effect.glow", "visual.effect.bloom",
                "visual.effect.halation", "visual.effect.lens-distortion",
                "visual.effect.denoise", "visual.effect.directional-blur",
                "visual.effect.unsharp-mask", "visual.effect.high-pass-sharpen",
                "visual.color.brightness", "visual.color.contrast",
                "visual.color.saturation", "visual.color.hue", "visual.color.exposure",
                "visual.color.gamma", "visual.color.transform", "visual.scalar.constant",
                "visual.boolean.constant", "visual.scalar.add", "visual.scalar.multiply",
                "visual.scalar.clamp", "visual.scalar.remap", "visual.vec2.constant",
                "visual.vec2.add", "visual.vec2.multiply", "visual.vec2.remap",
                "visual.color.constant", "visual.color.mix", "visual.color.remap",
                "visual.gradient", "visual.shape", "visual.noise",
                "visual.noise.fractal", "visual.field.scalar", "visual.field.vector", "visual.field.color",
                "visual.points.grid", "visual.points.set-position", "visual.points.set-scale",
                "visual.points.set-rotation", "visual.points.set-color",
                "visual.shape.rectangle", "visual.shape.ellipse", "visual.shape.union",
                "visual.shape.intersection", "visual.shape.subtract", "visual.shape.invert",
                "visual.matte.path", "visual.uv.transform",
                "visual.uv.remap", "visual.clone-to-points", "visual.draw.shape", "visual.feedback",
                "visual.matte.asset", "visual.matte.combine", "visual.matte.refine",
                "visual.matte.apply", "visual.depth.asset",
                "visual.particles", "tracking.point.asset", "tracking.planar.asset",
                "tracking.correction", "tracking.point.apply.transform", "tracking.planar.apply.quad",
                "visual.3d.imported-animation", "visual.deformation.imported",
                "visual.score.note-collection", "visual.3d.note-instanced-mesh",
                "visual.3d.render",
                "visual.surface.constant.scalar", "visual.surface.constant.uint",
                "visual.surface.constant.vec2", "visual.surface.constant.vec3",
                "visual.surface.constant.vec4", "visual.surface.texture-sample-2d",
                "visual.surface.compose.vec2", "visual.surface.compose.vec3",
                "visual.surface.compose.vec4", "visual.surface.material",
                "visual.sdf.sphere", "visual.sdf.box", "visual.sdf.rounded-box",
                "visual.sdf.plane", "visual.sdf.torus", "visual.sdf.capsule",
                "visual.sdf.cylinder", "visual.sdf.cone", "visual.sdf.gyroid",
                "visual.sdf.union", "visual.sdf.intersection", "visual.sdf.subtract",
                "visual.sdf.smooth-union", "visual.sdf.smooth-intersection",
                "visual.sdf.smooth-subtract", "visual.sdf.translate", "visual.sdf.rotate",
                "visual.sdf.scale", "visual.sdf.repeat", "visual.sdf.polar-repeat",
                "visual.sdf.mirror", "visual.sdf.twist", "visual.sdf.bend",
                "visual.sdf.taper", "visual.sdf.displacement", "visual.sdf.domain-warp",
                "visual.sdf.raymarch"
            };
            if (plan.nodeKinds.empty()
                || (! scheduledNativeDag
                    && ! nativeSdfPlan
                    && ! importedSceneDag
                    && plan.nodeKinds.front() != "video.source"
                    && plan.nodeKinds.front() != "video.legacy.source"
                    && plan.nodeKinds.front() != "video.legacy.generator"
                    && plan.nodeKinds.front() != "visual.shader.custom"
                    && plan.nodeKinds.front() != "visual.particles")
                || (! scheduledNativeDag && ! importedSceneDag
                    && plan.nodeKinds.back() != "video.out")
                || std::any_of(plan.nodeKinds.begin(), plan.nodeKinds.end(),
                    [&supportedKinds](const std::string& kind)
                    { return std::find(supportedKinds.begin(), supportedKinds.end(), kind)
                             == supportedKinds.end(); }))
            {
                error = "visual layer plan contains an unsupported typed operation";
                return false;
            }
            if (! scheduledNativeDag)
            {
                for (const auto& edge : plan.edges)
                {
                    const auto from = std::find(plan.nodeIds.begin(), plan.nodeIds.end(), edge.fromNodeId);
                    const auto to = std::find(plan.nodeIds.begin(), plan.nodeIds.end(), edge.toNodeId);
                    if (from == plan.nodeIds.end() || to == plan.nodeIds.end() || from >= to)
                    {
                        error = "visual layer plan is not an ordered acyclic graph";
                        return false;
                    }
                }
            }
        }
        if (! hasTypedBindings
            && plan.nodeKinds != fixedKinds && plan.nodeKinds != generatorKinds
            && plan.nodeKinds != editableEffectsKinds
            && plan.nodeKinds != editableMaskKinds
            && plan.nodeKinds != editableBlendKinds
            && plan.nodeKinds != editableTextKinds
            && plan.nodeKinds != editableLayerKinds
            && plan.nodeKinds != editableTransformKinds
            && plan.nodeKinds != editableDirectKinds)
        {
            error = "visual layer plan has unsupported production topology";
            return false;
        }
        if (std::any_of (plans.begin(), plans.begin() + static_cast<std::ptrdiff_t> (i),
                         [&plan] (const CompiledVisualLayerPlan& prior)
                         { return prior.clipId == plan.clipId; }))
        {
            error = "visual layer plan duplicates a clip owner";
            return false;
        }
    }
    for (const auto& segment : segments)
        if (std::none_of (plans.begin(), plans.end(),
                          [&segment] (const CompiledVisualLayerPlan& plan)
                          { return plan.clipId == segment.clipId; }))
        {
            error = "snapshot segment has no compiled visual layer plan";
            return false;
        }
    return true;
}

inline bool normalizeSnapshot (std::vector<RawRenderSegment> raw,
                               std::vector<CompiledVisualLayerPlan> plans,
                               std::vector<VisualEventScheduleBinding> eventSchedules,
                               uint64_t authoringRevision,
                               bool requirePlans,
                               ResolvedVisualSnapshot& result,
                               std::string& error)
{
    ResolvedVisualSnapshot candidate;
    candidate.authoringRevision = authoringRevision;
    candidate.segments.reserve (raw.size());

    for (auto& source : raw)
    {
        if (source.hasStructuredSourceKind
            && source.sourceKind == SourceKind::Unspecified)
        {
            error = "explicit source kind is invalid";
            return false;
        }
        const auto resolvedSourceKind = resolveSourceKind (
            source.hasStructuredSourceKind ? source.sourceKind : SourceKind::Unspecified,
            source.isAdjustment, source.sourcePath);
        RenderSegment segment = std::move (source);
        segment.sourceKind = resolvedSourceKind;
        segment.isAdjustment = segment.sourceKind == SourceKind::Adjustment;

        if (segment.clipId <= 0)
        {
            error = "snapshot segment has no stable clip owner";
            return false;
        }
        if (! std::isfinite(segment.inSec) || ! std::isfinite(segment.outSec)
            || ! std::isfinite(segment.rate) || ! std::isfinite(segment.displayStartSec)
            || ! std::isfinite(segment.clockStartSec) || ! std::isfinite(segment.clockDurationSec)
            || (segment.clockStartSec < 0.0 && segment.clockStartSec != -1.0)
            || segment.clockDurationSec < 0.0
            || (segment.clockStartSec >= 0.0 && segment.clockDurationSec <= 0.0)
            || segment.outSec < segment.inSec || segment.rate <= 0.0)
        {
            error = "snapshot segment has invalid timing";
            return false;
        }
        if (segment.sourceKind == SourceKind::Media && segment.sourcePath.empty())
        {
            error = "media source has no path";
            return false;
        }
        if (! segment.matteCacheKey.empty()
            && (! std::isfinite(segment.matteFps) || segment.matteFps <= 0.0
                || segment.matteFrames <= 0))
        {
            error = "matte binding is incomplete";
            return false;
        }
        if (segment.matteCacheKey.empty()
            && (segment.matteFps != 0.0 || segment.matteFrames != 0))
        {
            error = "matte metadata has no owner";
            return false;
        }
        if (! segment.matteAssetId.empty()
            && segment.matteState != "available"
            && segment.matteState != "missing"
            && segment.matteState != "stale")
        {
            error = "matte asset has invalid state";
            return false;
        }
        if (segment.matteAssetId.empty() && ! segment.matteState.empty())
        {
            error = "matte asset state has no stable owner";
            return false;
        }
        const auto duplicateOwner = std::find_if(
            candidate.segments.begin(), candidate.segments.end(),
            [&segment](const RenderSegment& prior)
            {
                if (prior.clipId != segment.clipId)
                    return false;
                if (prior.trackLayer != segment.trackLayer
                    || prior.sourceKind != segment.sourceKind
                    || prior.isAdjustment != segment.isAdjustment)
                    return true;
                const double priorEnd = prior.displayStartSec
                    + (prior.outSec - prior.inSec) / std::max(prior.rate, 1.0e-9);
                const double segmentEnd = segment.displayStartSec
                    + (segment.outSec - segment.inSec) / std::max(segment.rate, 1.0e-9);
                return segment.displayStartSec < priorEnd - 1.0e-9
                    && prior.displayStartSec < segmentEnd - 1.0e-9;
            });
        if (duplicateOwner != candidate.segments.end())
        {
            error = "snapshot duplicates or overlaps a stable clip owner";
            return false;
        }

        if (source.hasStructuredTransitionBinding
            && (segment.transitionFromClipId < 0
                || segment.transitionToClipId != segment.clipId))
        {
            error = "explicit transition binding is invalid";
            return false;
        }
        candidate.segments.push_back (std::move (segment));
    }

    for (const auto& segment : candidate.segments)
        if (segment.transitionType != 0 && segment.transitionDurationSec > 0.0)
        {
            const bool explicitBinding = segment.transitionFromClipId >= 0
                                      || segment.transitionToClipId >= 0;
            if (explicitBinding)
            {
                const auto found = std::find_if (
                    candidate.segments.begin(), candidate.segments.end(),
                    [&segment] (const RenderSegment& other)
                    { return other.clipId == segment.transitionFromClipId; });
                if (segment.transitionFromClipId != 0
                    && found == candidate.segments.end())
                {
                    error = "explicit transition source is missing";
                    return false;
                }
            }
        }

    std::sort (candidate.segments.begin(), candidate.segments.end(),
               [] (const RenderSegment& a, const RenderSegment& b)
               {
                   if (a.displayStartSec != b.displayStartSec)
                       return a.displayStartSec < b.displayStartSec;
                   if (a.trackLayer != b.trackLayer)
                       return a.trackLayer < b.trackLayer;
                   return a.clipId < b.clipId;
               });
    plans = normalizedVisualLayerPlans(plans);
    if (! validateCompiledVisualLayerPlans (candidate.segments, plans, requirePlans, error))
        return false;
    if (eventSchedules.size() > 256)
    {
        error = "visual Event schedule capacity exceeded";
        return false;
    }
    for (const auto& schedule : eventSchedules)
    {
        const auto plan = std::find_if (plans.begin(), plans.end(), [&] (const auto& value)
            { return value.clipId == schedule.clipId; });
        if (plan == plans.end())
        {
            error = "visual Event schedule has invalid stable sink identity";
            return false;
        }
        const auto node = std::find (plan->nodeIds.begin(), plan->nodeIds.end(), schedule.nodeId);
        const bool particle = node != plan->nodeIds.end()
            && plan->nodeKinds[static_cast<size_t> (std::distance (plan->nodeIds.begin(), node))]
                == "visual.particles";
        const bool eventPort = std::any_of (plan->ports.begin(), plan->ports.end(),
            [&] (const auto& port) { return port.nodeId == schedule.nodeId
                && port.port == schedule.portId && port.direction == "in"
                && port.carrier == "event" && port.channels == 1; });
        if (! particle || ! eventPort || schedule.sessionRevision == 0
            || schedule.triggers.size() > 256)
        {
            error = "visual Event schedule has invalid stable sink identity";
            return false;
        }
        double priorBeat = -std::numeric_limits<double>::infinity();
        uint64_t priorSequence = 0;
        for (const auto& trigger : schedule.triggers)
        {
            if (trigger.sampleOffset < 0 || ! std::isfinite (trigger.timelineBeat)
                || ! std::isfinite (trigger.strength) || trigger.strength < 0.0f
                || trigger.strength > 1.0f || trigger.timelineBeat < priorBeat
                || (trigger.timelineBeat == priorBeat && trigger.sequence < priorSequence))
            {
                error = "visual Event schedule trigger ordering or bounds are invalid";
                return false;
            }
            priorBeat = trigger.timelineBeat;
            priorSequence = trigger.sequence;
        }
    }
    candidate.visualLayerPlans = std::move (plans);
    candidate.visualEventSchedules = std::move (eventSchedules);
    result = std::move (candidate);
    error.clear();
    return true;
}

inline bool normalizeSnapshot (std::vector<RawRenderSegment> raw,
                               std::vector<CompiledVisualLayerPlan> plans,
                               uint64_t authoringRevision,
                               bool requirePlans,
                               ResolvedVisualSnapshot& result,
                               std::string& error)
{
    return normalizeSnapshot (std::move (raw), std::move (plans), {}, authoringRevision,
                              requirePlans, result, error);
}

inline bool normalizeSnapshot (std::vector<RawRenderSegment> raw,
                               uint64_t authoringRevision,
                               ResolvedVisualSnapshot& result,
                               std::string& error)
{
    return normalizeSnapshot (std::move (raw), {}, {}, authoringRevision, false, result, error);
}

inline bool validateSnapshotResources (
    ResolvedVisualSnapshot& snapshot,
    const std::function<bool (const std::string&)>& resourceExists,
    const std::filesystem::path& trustedMatteCacheRoot,
    uint64_t matteContentRevision,
    std::string& error)
{
    for (auto& segment : snapshot.segments)
    {
        if (segment.sourceKind == SourceKind::Media
            && ! resourceExists (segment.sourcePath))
        {
            error = "media source is missing: " + segment.sourcePath;
            return false;
        }
        if (! segment.matteAssetId.empty() && segment.matteState != "available")
        {
            error = "matte asset is " + segment.matteState + ": " + segment.matteAssetId;
            return false;
        }
        if (! segment.matteAssetId.empty())
        {
            segment.matteContentRevision = matteContentRevision;
            segment.matteTrustedRoot = trustedMatteCacheRoot.string();
            auto binding = matteCacheBinding(segment);
            if(matteContentRevision==0 || !videohelper::validateMatteCache(
                    trustedMatteCacheRoot,binding,segment.matteDir,error)) return false;
        }

    }
    error.clear();
    return true;
}

inline bool validateSnapshotResources (
    ResolvedVisualSnapshot& snapshot,
    const std::function<bool (const std::string&)>& resourceExists,
    std::string& error)
{
    return validateSnapshotResources(snapshot, resourceExists, {}, 0, error);
}

class RevisionLedger
{
public:
    const RevisionTuple& state() const noexcept { return state_; }

    bool requestAuthoring (uint64_t revision) noexcept
    {
        if (revision <= state_.authoring)
            return false;
        state_.authoring = revision;
        state_.rejected = 0;
        state_.exportable = 0;
        return true;
    }

    bool accept (uint64_t revision) noexcept
    {
        if (revision > state_.authoring || revision <= state_.accepted
            || revision == state_.rejected)
            return false;
        state_.accepted = revision;
        if (revision == state_.authoring)
            state_.rejected = 0;
        return true;
    }

    bool reject (uint64_t revision) noexcept
    {
        if (revision != state_.authoring)
            return false;
        state_.rejected = revision;
        state_.exportable = 0;
        return true;
    }

    bool compileSucceeded (uint64_t revision, bool exportable) noexcept
    {
        if (revision != state_.accepted || revision <= state_.compiled)
            return false;
        state_.compiled = revision;
        state_.lastGood = revision;
        state_.exportable = exportable && revision == state_.authoring ? revision : 0;
        return true;
    }

    bool evaluate (uint64_t compiledRevision, uint64_t sequence) noexcept
    {
        if (compiledRevision != state_.compiled || sequence <= state_.evaluation)
            return false;
        state_.evaluation = sequence;
        return true;
    }

private:
    RevisionTuple state_;
};
} // namespace videowire
