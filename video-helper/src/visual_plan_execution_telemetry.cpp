#include "visual_plan_executor.h"

namespace videowire
{
const VisualLayerExecution* findAdmittedVisualLayerExecution (
    VisualPlanExecutionState& state, int clipId, uint64_t structuralRevision,
    double evaluationTimeSec, bool& pausedHold)
{
    pausedHold = state.isHold(clipId, structuralRevision, evaluationTimeSec);
    const auto* execution = state.compiled(clipId, structuralRevision);
    if (execution != nullptr && ! pausedHold)
        state.telemetry().recordPlanLowering(true, 0, false);
    return execution;
}
} // namespace videowire
