#pragma once

#include "../../src/render_snapshot.h"
#include "../../../shared/SdfRaymarchOperationContract.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <string>

namespace sdfdisplayoracle
{
inline videowire::CompiledVisualLayerPlan plan(std::uint8_t output)
{
    videowire::SdfRaymarchOperation operation;
    operation.terminalStableId = 81;
    operation.maximumSteps = 128;
    operation.maximumDistance = 8.0;
    operation.epsilon = 0.001;
    operation.geometry.rootId = 72;
    videowire::SdfRecord sphere;
    sphere.stableId = 71;
    sphere.operation = videowire::SdfOperation::sphere;
    sphere.parameterCount = 1;
    sphere.parameters[0] = 0.45;
    videowire::SdfRecord translation;
    translation.stableId = 72;
    translation.operation = videowire::SdfOperation::translate;
    translation.inputCount = 1;
    translation.inputs[0] = sphere.stableId;
    translation.parameterCount = 3;
    translation.parameters = {0.0, 0.6, 0.0};
    operation.geometry.records = {sphere, translation};
    videowire::CompiledVisualLayerPlan plan;
    plan.clipId = 501;
    plan.structuralRevision = 24;
    plan.producerValidated = true;
    plan.nodeIds = {70, 71, 80, 90};
    plan.nodeKinds = {"visual.sdf.sphere", "visual.sdf.translate", "visual.sdf.raymarch", "video.out"};
    plan.operations = {
        {70, "visual.sdf.sphere", "control-eval", {}},
        {71, "visual.sdf.translate", "control-eval", {}},
        {80, "visual.sdf.raymarch", "native-gpu", {}},
        {90, "video.out", "native-gpu", {}}
    };
    plan.ports = {
        {70, 0, 1, "out", "control", "sdf", "unspecified", "unspecified"},
        {71, 0, 1, "in", "control", "sdf", "unspecified", "unspecified"},
        {71, 1, 1, "out", "control", "sdf", "unspecified", "unspecified"},
        {80, 0, 1, "in", "control", "sdf", "unspecified", "unspecified"},
        {80, 1, 1, "out", "frame", "image", "rgba8", "sRGB"},
        {90, 0, 1, "in", "frame", "image", "rgba8", "sRGB"}
    };
    plan.edges = {{70, 0, 71, 0}, {71, 1, 80, 0}, {80, 1, 90, 0}};
    operation.outputPass = output;
    plan.operations[2].payloadXml = videowire::encodeSdfRaymarchOperation(operation);
    return plan;
}

// Camera (0,0,3), focal length 1.8, sphere radius .45 at (0,.6,0),
// 64x36 pixels, maximum distance 8. Closed-form ray/sphere intersections
// and the declared sRGB transfer give these top-first display values.
// The upper ray hits at normalized depth .327399042269; the lower ray misses.
template <typename ReadPixel>
bool verify(ReadPixel&& read, bool depth, std::string& error)
{
    const std::array<std::array<int, 4>, 2> expected = depth
        ? std::array<std::array<int, 4>, 2>{{{155,155,155,255}, {255,255,255,255}}}
        : std::array<std::array<int, 4>, 2>{{{56,103,146,255}, {46,56,75,255}}};
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        const auto actual = read(32, i == 0 ? 12 : 24);
        for (std::size_t c = 0; c < 4; ++c)
            if (std::abs(static_cast<int>(actual[c]) - expected[i][c]) > (c == 3 ? 0 : 3))
            {
                error = "SDF SDR pixels disagree with independent top-first scene and sRGB transfer goldens";
                return false;
            }
    }
    return true;
}
} // namespace sdfdisplayoracle
