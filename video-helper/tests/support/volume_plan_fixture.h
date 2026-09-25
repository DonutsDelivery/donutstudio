#pragma once

#include "render_snapshot.h"

namespace volumetest
{
inline videowire::CompiledVisualLayerPlan plan(visualvolume::Operation operation = {})
{
    operation.sourceStableId = 12;
    operation.renderStableId = 13;
    videowire::CompiledVisualLayerPlan result;
    result.clipId = 41;
    result.structuralRevision = 1;
    result.identityMode = "authoredGraph";
    result.producerValidated = true;
    result.nodeIds = {11, 12, 13};
    result.nodeKinds = {visualvolume::sourceKind, visualvolume::renderKind, "video.out"};
    const auto payload = visualvolume::encode(operation);
    result.operations = {{11, visualvolume::sourceKind, "control-eval", payload},
                         {12, visualvolume::renderKind, "native-gpu", payload},
                         {13, "video.out", "native-gpu", ""}};
    result.edges = {{11, 0, 12, 0}, {12, 1, 13, 0}};
    result.ports = {{11, 0, 1, "out", "control", "volume", "unspecified", "unspecified"},
                    {12, 0, 1, "in", "control", "volume", "unspecified", "unspecified"},
                    {12, 1, 1, "out", "frame", "image", "rgba8", "sRGB"},
                    {13, 0, 1, "in", "frame", "image", "rgba8", "sRGB"}};
    result.descriptorCount = result.operationCount = 3;
    result.sceneRecordCount = result.frameOutputCount = 1;
    result.peakLiveFrameCount = result.allocatedFrameSlotCount = 1;
    return result;
}
} // namespace volumetest
