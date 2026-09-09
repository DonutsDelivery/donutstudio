#pragma once

#include "visual_plan_telemetry.h"

#include <cstdint>
#include <functional>

namespace videowire
{
struct VisualRendererTelemetryOperation
{
    VisualExecutionKind kind = VisualExecutionKind::generator;
    int clipId = -1;
    uint64_t structuralRevision = 0;
    int stableNodeId = 0;
    bool telemetryHold = false;
};

unsigned runVisualRendererTelemetryOperation (
    VisualPlanTelemetry* telemetry,
    const VisualRendererTelemetryOperation& operation,
    const std::function<unsigned()>& render);
} // namespace videowire
