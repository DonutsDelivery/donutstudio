#include "../src/visual_plan_publication.h"

#include <atomic>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace
{
int failures = 0;
void check(bool condition, const char* message)
{
    if (! condition) { ++failures; std::fprintf(stderr, "FAIL: %s\n", message); }
}

videowire::CompiledVisualLayerPlan planFor(int clipId, uint64_t revision)
{
    videowire::CompiledVisualLayerPlan plan;
    plan.clipId = clipId;
    plan.structuralRevision = revision;
    plan.producerValidated = true;
    plan.nodeKinds = { "video.source", "video.out" };
    plan.nodeIds = { 11, 12 };
    plan.ports = {
        { 11, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 12, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    plan.edges = { { 11, 0, 12, 0 } };
    plan.operations = {
        { 11, "video.source", "source-decode", "" },
        { 12, "video.out", "native-gpu", "" }
    };
    return plan;
}
}

int main()
{
    std::string diagnostic;
    std::shared_ptr<videowire::VisualPlanExecutionSnapshot> published;
    check(videowire::makeVisualPlanExecutionSnapshot({ planFor(7, 1) }, published, diagnostic),
          "initial viewport execution snapshot admits synchronously");
    const auto initial = published;

    videowire::VisualPlanTelemetry publicationTelemetry;
    videowire::VisualBackendResourceLimits initialLimits;
    initialLimits.backendProfile = "last-good-device";
    std::shared_ptr<videowire::VisualPlanExecutionSnapshot> lastGood;
    check(videowire::makeVisualPlanExecutionSnapshot(
              { planFor(7, 1) }, lastGood, diagnostic, &publicationTelemetry,
              640, 360, &initialLimits),
          "last-good snapshot publishes its measured budget provenance");
    const auto lastGoodIdentity = lastGood;
    videowire::VisualBackendResourceLimits staleLimits;
    staleLimits.backendProfile = "stale-device";
    std::shared_ptr<videowire::VisualPlanExecutionSnapshot> staleCandidate;
    check(videowire::makeVisualPlanExecutionSnapshot(
              { planFor(7, 2) }, staleCandidate, diagnostic, nullptr,
              1920, 1080, &staleLimits)
          && ! videowire::publishVisualPlanExecutionSnapshotIfCurrent(
              4, 8, 5, 8, staleCandidate, lastGood, publicationTelemetry)
          && lastGood == lastGoodIdentity
          && publicationTelemetry.snapshot().budgetReceipt.limits.backendProfile
                 == "last-good-device"
          && publicationTelemetry.snapshot().budgetReceipt.canvasWidth == 640,
          "stale renderer admission cannot replace publication or budget telemetry");
    check(videowire::publishVisualPlanExecutionSnapshotIfCurrent(
              5, 8, 5, 8, staleCandidate, lastGood, publicationTelemetry)
          && lastGood == staleCandidate
          && &lastGood->state.telemetry() == &publicationTelemetry
          && publicationTelemetry.snapshot().budgetReceipt.limits.backendProfile
                 == "stale-device"
          && publicationTelemetry.snapshot().budgetReceipt.canvasWidth == 1920,
          "current renderer admission atomically publishes its exact measured budget");

    auto unsupported = planFor(7, 2);
    unsupported.operations[1].backendCapability = "cpu-fallback";
    check(! videowire::makeVisualPlanExecutionSnapshot({ unsupported }, published, diagnostic)
          && published == initial && ! diagnostic.empty()
          && initial->plans[0].structuralRevision == 1,
          "failed viewport admission rolls back without replacing the live snapshot");

    videowire::VisualPlanExecutionState exportAdmission;
    std::vector<videowire::CompiledVisualLayerPlan> tooMany;
    for (size_t i = 0; i <= videowire::VisualPlanExecutionState::kMaxAdmittedPlans; ++i)
        tooMany.push_back(planFor(static_cast<int>(i + 1), 1));
    diagnostic.clear();
    check(! exportAdmission.admitPlans(tooMany, &diagnostic)
          && diagnostic == "visual execution admission exceeds fixed plan capacity"
          && exportAdmission.compiled(1, 1) == nullptr
          && exportAdmission.telemetry().snapshot().failedLowerings == 1,
          "export admission refuses over-capacity plans before installing partial state");

    std::shared_ptr<videowire::VisualPlanExecutionSnapshot> normalized;
    videowire::VisualPlanTelemetry normalizedTelemetry;
    const bool normalizedMade = videowire::makeVisualPlanExecutionSnapshot(
        { planFor(7, 1), planFor(8, 4), planFor(7, 3) }, normalized, diagnostic,
        &normalizedTelemetry);
    check(normalizedMade, "duplicate snapshot admits after canonical normalization");
    if (normalizedMade)
    {
        check(normalized->plans.size() == 2
              && normalized->plans[0].clipId == 7
              && normalized->plans[0].structuralRevision == 3
              && normalized->plans[1].clipId == 8
              && normalized->plans[1].structuralRevision == 4
              && normalized->state.compiled(7, 3) != nullptr,
              "publication and admission contain one latest revision per clip");
        const auto normalizedTelemetrySnapshot = normalizedTelemetry.snapshot();
        check(normalizedTelemetrySnapshot.layers.size() == 2
              && normalizedTelemetrySnapshot.layers[0].clipId == 7
              && normalizedTelemetrySnapshot.layers[0].structuralRevision == 3
              && normalizedTelemetrySnapshot.layers[1].clipId == 8
              && normalizedTelemetrySnapshot.layers[1].structuralRevision == 4,
              "telemetry contains one latest revision per clip");
    }
    auto equalFirst = planFor(9, 5);
    auto equalSecond = equalFirst;
    equalSecond.operations[1].backendCapability = "cpu-fallback";
    std::shared_ptr<videowire::VisualPlanExecutionSnapshot> equalRevision;
    check(videowire::makeVisualPlanExecutionSnapshot(
              { equalFirst, equalSecond }, equalRevision, diagnostic)
          && equalRevision->plans.size() == 1
          && equalRevision->plans[0].operations[1].backendCapability == "native-gpu"
          && equalRevision->state.compiled(9, 5) != nullptr,
          "equal-revision snapshot publication consistently preserves the first plan");

    auto telemetryCapacity = std::make_unique<videowire::VisualPlanTelemetry>();
    std::vector<videowire::VisualTelemetryPlanAdmission> telemetryPlans;
    for (int clipId = 1; clipId <= 65; ++clipId)
        telemetryPlans.push_back(videowire::makeVisualTelemetryAdmission(planFor(clipId, 1)));
    check(telemetryCapacity->admitPlans(telemetryPlans, &diagnostic)
          && telemetryCapacity->recordResources(3, 4, 5)
          && telemetryCapacity->snapshot().layers.size() == 65
          && telemetryCapacity->snapshot().resourcesObserved,
          "telemetry admits more than the stale 64-plan capacity without clearing resources");
    for (int clipId = 66; clipId <= 256; ++clipId)
        telemetryPlans.push_back(videowire::makeVisualTelemetryAdmission(planFor(clipId, 1)));
    check(telemetryCapacity->admitPlans(telemetryPlans, &diagnostic)
          && telemetryCapacity->recordResources(7, 8, 9)
          && telemetryCapacity->snapshot().layers.size() == 256,
          "telemetry stages the same 256 independent plans as execution admission");
    telemetryPlans.push_back(videowire::makeVisualTelemetryAdmission(planFor(257, 1)));
    const auto telemetryLastGood = telemetryCapacity->snapshot();
    check(! telemetryCapacity->admitPlans(telemetryPlans, &diagnostic)
          && diagnostic == "visual telemetry admission exceeds fixed plan capacity"
          && telemetryCapacity->snapshot().layers.size() == 256
          && telemetryCapacity->snapshot().retainedFramesCurrent
               == telemetryLastGood.retainedFramesCurrent
          && telemetryCapacity->snapshot().retainedBytesCurrent
               == telemetryLastGood.retainedBytesCurrent,
          "over-capacity telemetry admission preserves all 256 last-good plans and resources");

    std::mutex publicationMutex;
    std::atomic<bool> stop { false };
    std::atomic<bool> staleLifetimeSafe { true };
    std::atomic<uint64_t> leasedRevision { 0 };
    std::atomic<uint64_t> leaseCount { 0 };
    std::thread render([&]
    {
        while (! stop.load(std::memory_order_acquire))
        {
            std::shared_ptr<videowire::VisualPlanExecutionSnapshot> frame;
            {
                std::lock_guard<std::mutex> lock(publicationMutex);
                frame = published;
            }
            const auto revision = frame->plans[0].structuralRevision;
            const auto owner = frame->state.owner(7);
            if (! owner.has_value() || owner->structuralRevision != revision
                || frame->plans[0].structuralRevision != revision)
                staleLifetimeSafe.store(false, std::memory_order_release);
            leasedRevision.store(revision, std::memory_order_release);
            leaseCount.fetch_add(1, std::memory_order_release);
        }
    });
    while (leasedRevision.load(std::memory_order_acquire) != 1)
        std::this_thread::yield();
    for (uint64_t revision = 2; revision < 200; ++revision)
    {
        std::shared_ptr<videowire::VisualPlanExecutionSnapshot> next;
        check(videowire::makeVisualPlanExecutionSnapshot({ planFor(7, revision) }, next, diagnostic),
              "replacement snapshot admits off render thread");
        std::lock_guard<std::mutex> lock(publicationMutex);
        published = std::move(next);
    }
    while (leasedRevision.load(std::memory_order_acquire) < 2)
        std::this_thread::yield();
    stop.store(true, std::memory_order_release);
    render.join();
    check(leaseCount.load(std::memory_order_acquire) >= 2
          && staleLifetimeSafe.load(std::memory_order_acquire)
          && initial->plans[0].structuralRevision == 1
          && initial->state.compiled(7, 1) != nullptr,
          "concurrent timeline replacement preserves every leased stale frame lifetime");

    std::printf("viewport plan concurrency: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
