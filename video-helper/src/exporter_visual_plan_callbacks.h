#pragma once

#include "export_visual_plan_session.h"

#if ARBIT_HAVE_ONNX
#include "rife.h"
#endif

#include <functional>
#include <string>

namespace videowire
{
#if ARBIT_HAVE_ONNX
using ExportInterpolationEngine = arbitrife::RifeEngine;
#else
struct ExportInterpolationEngine;
#endif

using ExporterVisualPlanRun = std::function<std::string (
    VisualPlanExecutionSnapshot&, ExportTelemetryOwner<>&, ExportInterpolationEngine*)>;

struct ExportInterpolationStatistics
{
    std::string backend;
    int inferenceCount = 0;
    double totalInferenceMs = 0.0;
};

using ExportInterpolationStatisticsReader = std::function<ExportInterpolationStatistics (
    const ExportInterpolationEngine&)>;

ExportVisualPlanFrameLoop makeExporterVisualPlanFrameLoop (
    ExportInterpolationEngine* interpolationEngine,
    std::string& interpolationBackend,
    ExporterVisualPlanRun run,
    ExportInterpolationStatisticsReader statisticsReader = {});

std::string runExporterVisualPlanFrameLoop (
    ExportVisualPlanOrchestration& orchestration,
    ExportInterpolationEngine* interpolationEngine,
    std::string& interpolationBackend,
    ExporterVisualPlanRun run,
    ExportInterpolationStatisticsReader statisticsReader = {});
} // namespace videowire
