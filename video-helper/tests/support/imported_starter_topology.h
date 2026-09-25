#pragma once

#include "../../src/render_snapshot.h"

namespace videohelper::tests
{
// Full registered ports from Visual3DGraphNodes/SurfaceMaterialGraphNodes.
// The older imported execution fixtures intentionally exercise smaller ABIs.
inline void useRegisteredImportedStarterPorts(videowire::CompiledVisualLayerPlan& plan, int render)
{
    plan.ports.erase(std::remove_if(plan.ports.begin(), plan.ports.end(),
        [&](const auto& port) { return port.nodeId == render; }), plan.ports.end());
    plan.ports.insert(plan.ports.end(), {
        {render, 0, 1, "in", "control", "scene3D", "unspecified", "unspecified"},
        {render, 1, 1, "out", "frame", "image", "rgba8", "sRGB"},
        {render, 2, 1, "in", "control", "material", "unspecified", "unspecified"},
        {render, 3, 1, "in", "control", "mesh", "unspecified", "unspecified"},
        {render, 4, 1, "out", "frame", "depth", "r32f", "unspecified"},
        {render, 5, 1, "in", "control", "camera", "unspecified", "unspecified"},
        {render, 6, 1, "in", "control", "light", "unspecified", "unspecified"},
        {render, 7, 1, "out", "frame", "normal", "rgba16f", "unspecified"},
        {render, 8, 1, "out", "frame", "emission", "rgba16f", "linearSRGB"},
        {render, 9, 1, "out", "frame", "mask", "r8", "unspecified"},
        {render, 10, 1, "out", "frame", "materialId", "r32uint", "unspecified"},
        {render, 11, 1, "out", "frame", "objectId", "r32uint", "unspecified"},
        {render, 12, 2, "out", "frame", "motionVectors", "rg16f", "unspecified"}
    });
    for (const auto& operation : plan.operations)
        if (operation.kind == "visual.surface.material")
            plan.ports.insert(plan.ports.end(), {
                {operation.nodeId, 11, 3, "in", "control", "vec3", "unspecified", "unspecified"},
                {operation.nodeId, 12, 1, "in", "frame", "image", "rgba8", "sRGB"}
            });
}

// matrix-c0ec/receipt.json: each affected module's Image outlet is connected
// to outer Video Out node 6. Module-local identities below remain fixture IDs.
inline void appendImportedStarterOutput(videowire::CompiledVisualLayerPlan& plan, int node, int port)
{
    plan.nodeIds.push_back(6);
    plan.nodeKinds.push_back("video.out");
    plan.operations.push_back({6, "video.out", "native-gpu", "<NodeParams/>"});
    plan.ports.push_back({6, 0, 1, "in", "frame", "image", "rgba8", "sRGB"});
    plan.edges.push_back({node, port, 6, 0});
}
} // namespace videohelper::tests
