#pragma once

#include "export_telemetry_owner.h"
#include "visual_plan_publication.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace videowire
{
class ExportVisualPlanSession
{
public:
    using RendererTelemetryBinding = std::function<void (VisualPlanTelemetry*)>;
    using FrameLoop = std::function<std::string (VisualPlanExecutionSnapshot&,
                                                  ExportTelemetryOwner<>&,
                                                  void*)>;

    ExportVisualPlanSession();
    ~ExportVisualPlanSession();
    ExportVisualPlanSession (const ExportVisualPlanSession&) = delete;
    ExportVisualPlanSession& operator= (const ExportVisualPlanSession&) = delete;

    bool admit (std::vector<CompiledVisualLayerPlan> plans, std::string& diagnostic,
                VisualPlanTelemetry& telemetry, VisualBackend backend,
                int canvasWidth, int canvasHeight,
                const VisualBackendResourceLimits* backendLimits,
                RendererTelemetryBinding bindRenderer);
    void reset();
    std::string runFrameLoop (const FrameLoop& loop, void* interpolationOwner);

    VisualPlanExecutionSnapshot* snapshot() const noexcept;
    std::shared_ptr<VisualPlanExecutionSnapshot> sharedSnapshot() const noexcept;
    ExportTelemetryOwner<>* telemetryOwner() const noexcept;

private:
    VisualPlanExportExecutionOwner executionOwner_;
    std::shared_ptr<VisualPlanExecutionSnapshot> snapshot_;
    std::unique_ptr<ExportTelemetryOwner<>> telemetryOwner_;
    RendererTelemetryBinding rendererBinding_;
};

struct ExportVisualPlanFrameLoop
{
    ExportVisualPlanSession::FrameLoop run;
    void* onnxInterpolationOwner = nullptr;
    std::function<void (void*)> publishOnnxStatistics;
};

class ExportVisualPlanOrchestration
{
public:
    using RendererTelemetryBinding = ExportVisualPlanSession::RendererTelemetryBinding;

    bool admit (std::vector<CompiledVisualLayerPlan> plans, std::string& diagnostic,
                VisualPlanTelemetry& telemetry, VisualBackend backend,
                int canvasWidth, int canvasHeight,
                const VisualBackendResourceLimits* backendLimits,
                RendererTelemetryBinding bindRenderer);

    VisualPlanExecutionSnapshot* snapshot() const noexcept;
    std::shared_ptr<VisualPlanExecutionSnapshot> sharedSnapshot() const noexcept;
    ExportTelemetryOwner<>* telemetryOwner() const noexcept;

    // This compiled production boundary owns ARBIT_HAVE_ONNX selection. Tests build
    // this same translation unit in both configurations and invoke this method.
    std::string runCompiledFrameLoop (const ExportVisualPlanFrameLoop& frameLoop);
    bool drainFinalReadback (bool pendingFinalFrame,
                             const std::function<bool()>& drain,
                             std::string& error);

private:
    ExportVisualPlanSession session_;
};

std::string completeExporterFinalReadback (
    ExportVisualPlanOrchestration* orchestration,
    bool pendingFinalFrame,
    const std::function<bool()>& drain,
    const std::function<std::string()>& publish);
} // namespace videowire
