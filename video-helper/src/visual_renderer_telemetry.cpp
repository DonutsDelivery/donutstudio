#include "visual_renderer_telemetry.h"

#include <chrono>

namespace videowire
{
unsigned runVisualRendererTelemetryOperation (
    VisualPlanTelemetry* telemetry,
    const VisualRendererTelemetryOperation& operation,
    const std::function<unsigned()>& render)
{
    if (! render)
        return 0;
    const auto started = std::chrono::steady_clock::now();
    const auto texture = render();
    const auto elapsed = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started).count());
    if (telemetry != nullptr)
    {
        telemetry->recordExecutionObservation(operation.kind, elapsed);
        if (operation.kind == VisualExecutionKind::particle && operation.stableNodeId != 0)
            telemetry->recordNodeEvaluation(operation.clipId, operation.structuralRevision,
                                            operation.stableNodeId, elapsed,
                                            operation.telemetryHold);
    }
    return texture;
}
} // namespace videowire
