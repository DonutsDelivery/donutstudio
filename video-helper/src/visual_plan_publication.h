#pragma once

#include "visual_plan_executor.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace videowire
{
struct VisualPlanExecutionSnapshot
{
    explicit VisualPlanExecutionSnapshot (std::vector<CompiledVisualLayerPlan> normalizedPlans = {})
        : plans(std::move(normalizedPlans)) {}

    const std::vector<CompiledVisualLayerPlan> plans;
    VisualPlanExecutionState state;
};

inline bool makeVisualPlanExecutionSnapshot (
    std::vector<CompiledVisualLayerPlan> plans,
    std::shared_ptr<VisualPlanExecutionSnapshot>& snapshot,
    std::string& diagnostic, VisualPlanTelemetry* telemetryOwner,
    int canvasWidth, int canvasHeight,
    const VisualBackendResourceLimits* backendLimits);

class VisualPlanViewportPublicationOwner
{
public:
    VisualPlanViewportPublicationOwner()
        : publication_(std::make_shared<VisualPlanExecutionSnapshot>()) {}

    VisualPlanExecutionSnapshot* operator->() noexcept { return publication_.get(); }
    const VisualPlanExecutionSnapshot* operator->() const noexcept { return publication_.get(); }
    std::shared_ptr<VisualPlanExecutionSnapshot> shared() const noexcept { return publication_; }
    std::shared_ptr<VisualPlanExecutionSnapshot>& publicationRef() noexcept { return publication_; }
    void replace (std::shared_ptr<VisualPlanExecutionSnapshot> replacement)
    {
        publication_ = std::move(replacement);
    }

private:
    std::shared_ptr<VisualPlanExecutionSnapshot> publication_;
};

class VisualPlanExportExecutionOwner
{
public:
    bool admit (std::vector<CompiledVisualLayerPlan> plans, std::string& diagnostic,
                VisualPlanTelemetry* telemetryOwner = nullptr,
                int canvasWidth = 1920, int canvasHeight = 1080,
                const VisualBackendResourceLimits* backendLimits = nullptr)
    {
        return makeVisualPlanExecutionSnapshot(std::move(plans), snapshot_, diagnostic,
                                               telemetryOwner, canvasWidth, canvasHeight,
                                               backendLimits);
    }
    VisualPlanExecutionSnapshot* get() const noexcept { return snapshot_.get(); }
    std::shared_ptr<VisualPlanExecutionSnapshot> shared() const noexcept { return snapshot_; }
    VisualPlanExecutionSnapshot& snapshot() const noexcept { return *snapshot_; }

private:
    std::shared_ptr<VisualPlanExecutionSnapshot> snapshot_;
};

inline void publishVisualPlanExecutionSnapshotTelemetry (
    VisualPlanExecutionSnapshot& snapshot, VisualPlanTelemetry& telemetryOwner)
{
    snapshot.state.setTelemetryOwner(telemetryOwner);
    std::vector<VisualTelemetryPlanAdmission> telemetryPlans;
    telemetryPlans.reserve(snapshot.plans.size());
    for (const auto& plan : snapshot.plans)
        telemetryPlans.push_back(makeVisualTelemetryAdmission(plan));
    telemetryOwner.admitPlans(telemetryPlans);
    telemetryOwner.recordBudgetReceipt(snapshot.state.budgetReceipt());
}

inline bool publishVisualPlanExecutionSnapshotIfCurrent (
    uint64_t expectedTimelineGeneration, uint64_t expectedCanvasGeneration,
    uint64_t currentTimelineGeneration, uint64_t currentCanvasGeneration,
    std::shared_ptr<VisualPlanExecutionSnapshot> candidate,
    std::shared_ptr<VisualPlanExecutionSnapshot>& publication,
    VisualPlanTelemetry& telemetryOwner)
{
    if (expectedTimelineGeneration != currentTimelineGeneration
        || expectedCanvasGeneration != currentCanvasGeneration)
        return false;
    candidate->state.setTelemetryOwner(telemetryOwner);
    publishVisualPlanExecutionSnapshotTelemetry(*candidate, telemetryOwner);
    publication = std::move(candidate);
    return true;
}

inline bool makeVisualPlanExecutionSnapshot (
    std::vector<CompiledVisualLayerPlan> plans,
    std::shared_ptr<VisualPlanExecutionSnapshot>& snapshot,
    std::string& diagnostic, VisualPlanTelemetry* telemetryOwner = nullptr,
    int canvasWidth = 1920, int canvasHeight = 1080,
    const VisualBackendResourceLimits* backendLimits = nullptr)
{
    auto candidate = std::make_shared<VisualPlanExecutionSnapshot>(
        normalizedVisualLayerPlans(plans));
    if (telemetryOwner != nullptr) candidate->state.setTelemetryOwner(*telemetryOwner);
    diagnostic.clear();
    if (! candidate->state.admitPlans(candidate->plans, &diagnostic,
                                      canvasWidth, canvasHeight, backendLimits))
        return false;


    if (telemetryOwner != nullptr)
        telemetryOwner->recordBudgetReceipt(candidate->state.budgetReceipt());
    snapshot = std::move(candidate);
    return true;
}
} // namespace videowire
