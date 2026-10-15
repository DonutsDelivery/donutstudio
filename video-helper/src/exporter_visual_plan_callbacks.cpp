#include "exporter_visual_plan_callbacks.h"

#include <cstdio>
#include <utility>

namespace videowire
{
ExportVisualPlanFrameLoop makeExporterVisualPlanFrameLoop (
    ExportInterpolationEngine* interpolationEngine,
    std::string& interpolationBackend,
    ExporterVisualPlanRun run,
    ExportInterpolationStatisticsReader statisticsReader)
{
    ExportVisualPlanFrameLoop frameLoop;
#if ARBIT_HAVE_ONNX
    frameLoop.onnxInterpolationOwner = interpolationEngine;
    frameLoop.publishOnnxStatistics = [&interpolationBackend,
                                       statisticsReader = std::move(statisticsReader)] (void* owner)
    {
        if (owner == nullptr)
            return;
        auto& selectedRife = *static_cast<arbitrife::RifeEngine*> (owner);
        const auto statistics = statisticsReader
            ? statisticsReader(selectedRife)
            : ExportInterpolationStatistics { selectedRife.backend(),
                                              selectedRife.inferenceCount(),
                                              selectedRife.totalInferenceMs() };
        interpolationBackend = statistics.backend;
        if (statistics.inferenceCount > 0)
            std::fprintf (stderr, "[rife] synthesized %d frames, avg %.1f ms (%s)\n",
                          statistics.inferenceCount,
                          statistics.totalInferenceMs / statistics.inferenceCount,
                          statistics.backend.c_str());
    };
#else
    (void) interpolationEngine;
    (void) interpolationBackend;
    (void) statisticsReader;
#endif
    frameLoop.run = [run = std::move (run)] (
                        VisualPlanExecutionSnapshot& snapshot,
                        ExportTelemetryOwner<>& telemetry,
                        void* owner)
    {
#if ARBIT_HAVE_ONNX
        auto* selectedRife = static_cast<arbitrife::RifeEngine*> (owner);
#else
        (void) owner;
        ExportInterpolationEngine* selectedRife = nullptr;
#endif
        return run (snapshot, telemetry, selectedRife);
    };
    return frameLoop;
}

std::string runExporterVisualPlanFrameLoop (
    ExportVisualPlanOrchestration& orchestration,
    ExportInterpolationEngine* interpolationEngine,
    std::string& interpolationBackend,
    ExporterVisualPlanRun run,
    ExportInterpolationStatisticsReader statisticsReader)
{
    return orchestration.runCompiledFrameLoop(makeExporterVisualPlanFrameLoop(
        interpolationEngine, interpolationBackend, std::move(run),
        std::move(statisticsReader)));
}
} // namespace videowire
