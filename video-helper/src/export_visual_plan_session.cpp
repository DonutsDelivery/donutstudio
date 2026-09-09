#include "export_visual_plan_session.h"

#include <utility>

#ifndef ARBIT_HAVE_ONNX
#define ARBIT_HAVE_ONNX 0
#endif

namespace videowire
{
ExportVisualPlanSession::ExportVisualPlanSession() = default;
ExportVisualPlanSession::~ExportVisualPlanSession() { reset(); }

bool ExportVisualPlanSession::admit (std::vector<CompiledVisualLayerPlan> plans,
                                     std::string& diagnostic,
                                     VisualPlanTelemetry& telemetry,
                                     VisualBackend backend,
                                     int canvasWidth, int canvasHeight,
                                     const VisualBackendResourceLimits* backendLimits,
                                     RendererTelemetryBinding bindRenderer)
{
    reset();
    if (! executionOwner_.admit(std::move(plans), diagnostic, &telemetry,
                                canvasWidth, canvasHeight, backendLimits))
        return false;
    snapshot_ = executionOwner_.shared();
    telemetryOwner_ = std::make_unique<ExportTelemetryOwner<>>(
        snapshot_->state.telemetry(), backend, canvasWidth, canvasHeight);
    rendererBinding_ = std::move(bindRenderer);
    if (rendererBinding_)
        rendererBinding_(&telemetryOwner_->telemetry());
    return true;
}

void ExportVisualPlanSession::reset()
{
    if (rendererBinding_)
        rendererBinding_(nullptr);
    rendererBinding_ = {};
    telemetryOwner_.reset();
    snapshot_.reset();
}

std::string ExportVisualPlanSession::runFrameLoop (const FrameLoop& loop,
                                                   void* interpolationOwner)
{
    if (snapshot_ == nullptr || telemetryOwner_ == nullptr)
        return "missing admitted visual-plan export session";
    if (! loop)
        return "missing visual-plan export frame loop";
    return loop(*snapshot_, *telemetryOwner_, interpolationOwner);
}

VisualPlanExecutionSnapshot* ExportVisualPlanSession::snapshot() const noexcept
{
    return snapshot_.get();
}

std::shared_ptr<VisualPlanExecutionSnapshot> ExportVisualPlanSession::sharedSnapshot() const noexcept
{
    return snapshot_;
}

ExportTelemetryOwner<>* ExportVisualPlanSession::telemetryOwner() const noexcept
{
    return telemetryOwner_.get();
}

bool ExportVisualPlanOrchestration::admit (std::vector<CompiledVisualLayerPlan> plans,
                                           std::string& diagnostic,
                                           VisualPlanTelemetry& telemetry,
                                           VisualBackend backend,
                                           int canvasWidth, int canvasHeight,
                                           const VisualBackendResourceLimits* backendLimits,
                                           RendererTelemetryBinding bindRenderer)
{
    return session_.admit(std::move(plans), diagnostic, telemetry, backend,
                          canvasWidth, canvasHeight, backendLimits,
                          std::move(bindRenderer));
}

VisualPlanExecutionSnapshot* ExportVisualPlanOrchestration::snapshot() const noexcept
{
    return session_.snapshot();
}

std::shared_ptr<VisualPlanExecutionSnapshot> ExportVisualPlanOrchestration::sharedSnapshot() const noexcept
{
    return session_.sharedSnapshot();
}

ExportTelemetryOwner<>* ExportVisualPlanOrchestration::telemetryOwner() const noexcept
{
    return session_.telemetryOwner();
}

std::string ExportVisualPlanOrchestration::runCompiledFrameLoop (
    const ExportVisualPlanFrameLoop& frameLoop)
{
#if ARBIT_HAVE_ONNX
    auto error = session_.runFrameLoop(frameLoop.run, frameLoop.onnxInterpolationOwner);
    if (error.empty() && frameLoop.onnxInterpolationOwner != nullptr
        && frameLoop.publishOnnxStatistics)
        frameLoop.publishOnnxStatistics(frameLoop.onnxInterpolationOwner);
    return error;
#else
    return session_.runFrameLoop(frameLoop.run, nullptr);
#endif
}

bool ExportVisualPlanOrchestration::drainFinalReadback (
    bool pendingFinalFrame, const std::function<bool()>& drain, std::string& error)
{
    if (! pendingFinalFrame)
        return false;
    if (drain && drain())
        return true;
    error = "native compositor final readback drain failed";
    if (auto* owner = session_.telemetryOwner())
        owner->telemetry().recordDrop(VisualDropReason::transportFailure);
    return false;
}

std::string completeExporterFinalReadback (
    ExportVisualPlanOrchestration* orchestration,
    bool pendingFinalFrame,
    const std::function<bool()>& drain,
    const std::function<std::string()>& publish)
{
    std::string error;
    if (! pendingFinalFrame)
        return error;
    const bool drained = orchestration != nullptr
        ? orchestration->drainFinalReadback(true, drain, error)
        : drain && drain();
    if (! drained)
    {
        if (error.empty())
            error = "native compositor final readback drain failed";
        return error;
    }
    if (! publish)
        return "missing final readback publisher";
    return publish();
}
} // namespace videowire
