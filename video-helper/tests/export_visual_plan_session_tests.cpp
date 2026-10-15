#include "export_visual_plan_session.h"
#include "exporter_visual_plan_callbacks.h"
#include "viewport_telemetry_owner.h"
#include "visual_plan_telemetry_json.h"
#include "visual_renderer_telemetry.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#if ARBIT_HAVE_ONNX
namespace arbitrife
{
struct RifeEngine::Impl {};
RifeEngine::RifeEngine() = default;
RifeEngine::~RifeEngine() = default;
std::string RifeEngine::init() { return {}; }
std::string RifeEngine::interpolate (const uint8_t*, int, const uint8_t*, int,
                                     int, int, float, std::vector<uint8_t>&)
{
    return {};
}
} // namespace arbitrife
#endif

namespace
{
int failures = 0;
void check (bool condition, const char* message)
{
    if (condition) return;
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

videowire::CompiledVisualLayerPlan plan()
{
    videowire::CompiledVisualLayerPlan value;
    value.clipId = 73;
    value.structuralRevision = 19;
    value.producerValidated = true;
    value.nodeIds = { 101, 102, 103 };
    value.nodeKinds = { "video.legacy.generator", "video.legacy.effects", "video.out" };
    value.operations.push_back({ 101, value.nodeKinds[0], "native-gpu", {} });
    value.operations.push_back({ 102, value.nodeKinds[1], "native-gpu", {} });
    value.operations.push_back({ 103, value.nodeKinds[2], "native-gpu", {} });
    value.ports.push_back({ 101, 1, 1, "out", "frame", "image", "rgba8", "linearSRGB" });
    value.ports.push_back({ 102, 0, 1, "in", "frame", "image", "rgba8", "linearSRGB" });
    value.ports.push_back({ 102, 1, 1, "out", "frame", "image", "rgba8", "linearSRGB" });
    value.ports.push_back({ 103, 0, 1, "in", "frame", "image", "rgba8", "linearSRGB" });
    value.edges.push_back({ 101, 1, 102, 0 });
    value.edges.push_back({ 102, 1, 103, 0 });
    value.descriptorCount = 3;
    value.operationCount = 3;
    value.frameOutputCount = 2;
    value.peakLiveFrameCount = 2;
    value.allocatedFrameSlotCount = 2;
    return value;
}

videowire::VisualBackendResourceLimits limits()
{
    auto value = videowire::VisualBackendResourceLimits::forCanvas(64, 64, 4096, 7);
    value.backendProfile = "opengl-test";
    value.backendDeviceIdentity = "deterministic-device";
    value.observedCombinedTextureImageUnits = 24;
    value.observedMaximumBufferLengthBytes = 123456;
    value.observedRecommendedWorkingSetBytes = 654321;
    return value;
}

void checkNoSuccess (const videowire::VisualTelemetrySnapshot& value, const char* message)
{
    check(value.framesRendered == 0 && value.framesPresented == 0
              && value.exportHandoffFrames == 0 && value.transportFramesHandedOff == 0
              && value.zeroCopyFrames == 0 && value.readbackFrames == 0
              && value.zeroCopyAllocationBytes == 0 && value.readbackCopiedBytes == 0,
          message);
}

void checkProductionBranch (const char* branchName)
{
    videowire::VisualPlanTelemetry telemetry;
    videowire::VisualPlanTelemetry* rendererTelemetry = nullptr;
    videowire::ExportVisualPlanOrchestration orchestration;
    std::string diagnostic;
    const auto resourceLimits = limits();
    const bool admitted = orchestration.admit({ plan() }, diagnostic, telemetry,
                                               videowire::VisualBackend::openGL, 64, 64,
                                               &resourceLimits,
                                               [&] (videowire::VisualPlanTelemetry* owner)
                                               { rendererTelemetry = owner; });
    check(admitted, branchName);
    if (! admitted)
    {
        std::cerr << "admission diagnostic: " << diagnostic << '\n';
        return;
    }
    auto retained = orchestration.sharedSnapshot();
    bool loopCalled = false;
    bool statisticsRead = false;
    std::string interpolationBackend = "unobserved";
#if ARBIT_HAVE_ONNX
    arbitrife::RifeEngine interpolationEngine;
    auto* expectedInterpolationOwner = &interpolationEngine;
    videowire::ExportInterpolationStatisticsReader statisticsReader =
        [&] (const arbitrife::RifeEngine& selected) {
            statisticsRead = true;
            check(&selected == expectedInterpolationOwner,
                  "statistics reader receives the selected real RifeEngine owner");
            return videowire::ExportInterpolationStatistics { "rife-controlled", 4, 10.0 };
        };
#else
    videowire::ExportInterpolationEngine* expectedInterpolationOwner = nullptr;
#endif
    const auto error = videowire::runExporterVisualPlanFrameLoop(
        orchestration,
#if ARBIT_HAVE_ONNX
        expectedInterpolationOwner,
#else
        nullptr, interpolationBackend,
#endif
#if ARBIT_HAVE_ONNX
        interpolationBackend,
#endif
        [&] (videowire::VisualPlanExecutionSnapshot& snapshot,
             videowire::ExportTelemetryOwner<>& owner,
             videowire::ExportInterpolationEngine* interpolationOwner)
        {
            loopCalled = true;
            check(&snapshot == retained.get(), "selected branch receives admitted snapshot");
            check(&owner == orchestration.telemetryOwner(),
                  "selected branch receives orchestration telemetry owner");
            check(&owner.telemetry() == rendererTelemetry,
                  "selected branch and renderer share telemetry identity");
            check(interpolationOwner == expectedInterpolationOwner,
                  "compiled exporter callback preserves the selected interpolation owner");
            bool published = false;
            const auto drainError = videowire::completeExporterFinalReadback(
                &orchestration, true, [] { return false; },
                [&] { published = true; return std::string{}; });
            check(! published, "failed pending drain cannot publish the final frame");
            return drainError;
        }
#if ARBIT_HAVE_ONNX
        , statisticsReader
#endif
    );
    check(loopCalled, "selected production branch executes");
    check(error == "native compositor final readback drain failed",
          "pending drain returns precise export error");
    check(interpolationBackend == "unobserved",
          "failed loop does not publish interpolation statistics");
    check(! statisticsRead, "failed loop does not read interpolation statistics");
    const auto failed = telemetry.snapshot();
    check(failed.framesDropped == 1 && failed.droppedTransportFailure == 1,
          "pending drain publishes transport failure");
    checkNoSuccess(failed, "pending drain failure publishes no success counters");

    bool drainCalled = false;
    std::string noPendingError;
    check(! orchestration.drainFinalReadback(false, [&] { drainCalled = true; return false; },
                                             noPendingError),
          "no pending frame has no final readback");
    check(! drainCalled && noPendingError.empty()
              && telemetry.snapshot().framesDropped == 1,
          "no pending frame is not reported as a drain failure");

    std::vector<int> completionOrder;
    const auto success = videowire::runExporterVisualPlanFrameLoop(
        orchestration,
#if ARBIT_HAVE_ONNX
        expectedInterpolationOwner,
#else
        nullptr,
#endif
        interpolationBackend,
        [&] (videowire::VisualPlanExecutionSnapshot&,
             videowire::ExportTelemetryOwner<>&,
             videowire::ExportInterpolationEngine*) {
            return videowire::completeExporterFinalReadback(
                &orchestration, true,
                [&] { completionOrder.push_back(1); return true; },
                [&] { completionOrder.push_back(2); return std::string{}; });
        }
#if ARBIT_HAVE_ONNX
        , statisticsReader
#endif
    );
    check(success.empty() && completionOrder == std::vector<int>({ 1, 2 }),
          "export success waits for final drain and publication");
#if ARBIT_HAVE_ONNX
    check(statisticsRead && interpolationBackend == "rife-controlled",
          "successful ONNX export forwards owner statistics after final publication");
#else
    check(interpolationBackend == "unobserved",
          "non-ONNX export has no interpolation statistics");
#endif

    const auto publishFailure = videowire::completeExporterFinalReadback(
        &orchestration, true, [] { return true; },
        [] { return std::string("final frame publication failed"); });
    check(publishFailure == "final frame publication failed",
          "final publication failure fails export");
}

void checkProductionTelemetryOwners()
{
    videowire::VisualPlanTelemetry telemetry;
    videowire::ViewportTelemetryOwner<> viewport(telemetry, false);
    check(viewport.admitBackend(videowire::VisualBackend::openGL)
              && viewport.transitionBackend(videowire::VisualBackend::software),
          "viewport owner publishes backend transition");
    check(viewport.exportedBufferAllocated(8192)
              && viewport.zeroCopyHandoff(true, true, true),
          "viewport owner publishes zero-copy allocation and handoff");
    viewport.noFreeExportedBuffer();
    check(viewport.exportedBufferReleased(8192),
          "viewport owner releases exported allocation");
    auto value = telemetry.snapshot();
    check(value.backendObserved && value.initialBackend == videowire::VisualBackend::openGL
              && value.currentBackend == videowire::VisualBackend::software
              && value.fallbackCount == 1,
          "backend fallback transition is counted once");
    check(value.zeroCopyAllocationObserved && value.zeroCopyAllocationBytes == 8192
              && value.zeroCopyFrames == 1 && value.transportFramesHandedOff == 1,
          "zero-copy allocation and handoff stay distinct");
    check(value.droppedNoBuffer == 1 && value.retainedFramesCurrent == 0
              && value.retainedBytesCurrent == 0,
          "no-buffer drop and resource cleanup are published");

    auto lock = telemetry.lockForTesting();
    check(videowire::runVisualRendererTelemetryOperation(
              &telemetry, { videowire::VisualExecutionKind::generator },
              [] { return 9u; }) == 9u,
          "telemetry contention cannot fail renderer output");
    lock.unlock();
    value = telemetry.snapshot();
    check(value.recordingContentionDrops >= 1,
          "production renderer publication reports telemetry contention");
    const auto json = videowire::visualTelemetryJson(value);
    check(json["transport"]["mode"] == "zero-copy"
              && json["transport"]["allocationAvailable"] == true
              && json["transport"]["zeroCopyAllocationBytes"] == 8192
              && json["backend"]["fallbackCount"] == 1
              && json["drops"]["reasons"]["noBuffer"] == 1
              && json["recordingContentionDrops"].get<uint64_t>() >= 1,
          "JSON publishes zero-copy, fallback, drop, and contention families");
}

void checkPublicationBoundaries()
{
    videowire::VisualPlanTelemetry telemetry;
    videowire::VisualTelemetryPlanAdmission admitted;
    admitted.clipId = 1;
    admitted.structuralRevision = 2;
    admitted.nodeCount = videowire::kMaxCompiledNodesPerGraph;
    admitted.executableNodeTotal = admitted.nodeCount + 17;
    admitted.nodesTruncated = true;
    for (size_t index = 0; index < admitted.nodeCount; ++index)
        admitted.stableNodeIds[index] = static_cast<int>(index + 1);
    std::string diagnostic;
    check(telemetry.admitPlans({ admitted }, &diagnostic),
          "maximum executable publication subset is admitted");
    auto json = videowire::visualTelemetryJson(telemetry.snapshot());
    check(json["executableNodeTotal"] == admitted.executableNodeTotal
              && json["executableNodeReported"] == admitted.nodeCount
              && json["executableNodeSubsetTruncated"] == true,
          "JSON publishes executable totals and truncation");
    for (const auto& node : json["nodes"])
        check(node["available"] == false && node["evaluations"].is_null()
                  && node["measuredTotalNs"].is_null()
                  && node["measuredMovingNs"].is_null(),
              "unmeasured serialized node costs remain unavailable and null");

    std::vector<videowire::VisualTelemetryPlanAdmission> overLimit(
        videowire::kMaxTelemetryPlans + 1);
    for (size_t index = 0; index < overLimit.size(); ++index)
    {
        overLimit[index].clipId = static_cast<int>(index);
        overLimit[index].structuralRevision = 1;
    }
    check(! telemetry.admitPlans(overLimit, &diagnostic) && ! diagnostic.empty(),
          "first over-limit publication plan is rejected");
    json = videowire::visualTelemetryJson(telemetry.snapshot());
    check(json["rejectedPlans"] == 1,
          "JSON publishes rejected plan count");

    videowire::VisualPlanTelemetry emptyTelemetry;
    const auto unavailable = videowire::visualTelemetryJson(emptyTelemetry.snapshot());
    check(unavailable["transport"]["available"] == false
              && unavailable["transport"]["mode"].is_null()
              && unavailable["backend"]["available"] == false
              && unavailable["dimensions"]["requestedWidth"].is_null()
              && unavailable["resources"]["retainedBytesCurrent"].is_null()
              && unavailable["budget"]["limits"]["frameOutputs"].is_null(),
          "unobserved JSON families remain unavailable and null");
    const std::array<const char*, 4> durations { "compositor", "presentation",
        "particles", "generators" };
    for (const auto* name : durations)
        check(unavailable[name]["available"] == false
                  && unavailable[name]["count"].is_null()
                  && unavailable[name]["totalNs"].is_null()
                  && unavailable[name]["movingNs"].is_null(),
              "every unobserved serialized duration is null");
    check(unavailable["presentationSemantic"].is_null(),
          "unobserved presentation semantic is null");
    for (const auto* name : { "mode", "zeroCopyFrames", "zeroCopyAllocationBytes",
                              "readbackFrames", "readbackCopiedBytes", "framesHandedOff",
                              "framesRendered", "framesPresented", "exportHandoffFrames" })
        check(unavailable["transport"][name].is_null(),
              "every unobserved serialized transport value is null");
    check(unavailable["transport"]["allocationAvailable"] == false,
          "unobserved allocation availability is false");
    check(unavailable["drops"]["framesDropped"].is_null(),
          "unobserved serialized drop total is null");
    for (const auto* name : { "noBuffer", "transportFailure", "renderFailure",
                              "dimensionMismatch", "blitFailure", "fenceFailure",
                              "socketFailure" })
        check(unavailable["drops"]["reasons"][name].is_null(),
              "every unobserved serialized drop reason is null");
    for (const auto* name : { "initial", "current", "fallbackCount" })
        check(unavailable["backend"][name].is_null(),
              "every unobserved serialized backend value is null");
    for (const auto* name : { "retainedFramesCurrent", "retainedFramesPeak",
                              "intermediateImagesCurrent", "intermediateImagesPeak",
                              "retainedBytesCurrent", "retainedBytesPeak" })
        check(unavailable["resources"][name].is_null(),
              "every unobserved serialized resource value is null");
    for (const auto* name : { "backendProfile", "backendDeviceIdentity", "canvasWidth",
                              "canvasHeight" })
        check(unavailable["budget"][name].is_null(),
              "every unobserved serialized budget identity is null");
    for (const auto* name : { "descriptors", "operations", "sceneRecords", "frameOutputs",
                              "liveFrames", "frameSlots", "allocatedFrameBytes",
                              "maximumImageDimension" })
        check(unavailable["budget"]["limits"][name].is_null(),
              "every unobserved serialized budget limit is null");
    for (const auto* name : { "planCount", "descriptors", "operations", "sceneRecords",
                              "frameOutputs", "peakLiveFrames", "frameSlots",
                              "allocatedFrameBytes" })
        check(unavailable["budget"]["usage"][name].is_null(),
              "every unobserved serialized budget usage is null");
    for (const auto* name : { "maximumImageDimension", "frameSlots", "allocatedFrameBytes",
                              "observedCombinedTextureImageUnits",
                              "observedMaximumBufferLengthBytes",
                              "observedRecommendedWorkingSetBytes" })
        check(unavailable["budget"]["provenance"][name].is_null(),
              "every unobserved serialized budget provenance value is null");
    for (const auto* name : { "requestedWidth", "requestedHeight", "actualWidth",
                              "actualHeight", "dimensionMismatchCount",
                              "halfResolutionMismatchCount" })
        check(unavailable["dimensions"][name].is_null(),
              "every unobserved serialized dimension value is null");
}

void checkExactSerializedTelemetry (const videowire::VisualTelemetrySnapshot& value,
                                    const videowire::VisualBackendResourceLimits& resourceLimits)
{
    const auto actual = nlohmann::json::parse(videowire::visualTelemetryJson(value).dump());
    const auto duration = [] (const videowire::VisualDurationCounter& counter)
    {
        return nlohmann::json { { "available", true }, { "count", counter.count },
            { "totalNs", counter.totalNs }, { "movingNs", counter.movingNs } };
    };
    const nlohmann::json expected {
        { "graphEvaluations", 1 }, { "graphMeasuredTotalNs", 17 },
        { "graphMeasuredMovingNs", 17.0 },
        { "planCacheHits", value.planCacheHits }, { "planCacheMisses", value.planCacheMisses },
        { "lastPlanLoweringNs", value.lastPlanLoweringNs },
        { "planLoweringBudgetNs", videowire::kPlanLoweringBudgetNs },
        { "lastPlanLoweringWithinBudget", value.lastPlanLoweringWithinBudget },
        { "planInstalls", value.planInstalls }, { "droppedSamples", 0 },
        { "rejectedPlans", 0 }, { "failedLowerings", 0 }, { "lastLoweringError", "" },
        { "nodeSubsetLabel", "bounded executable render operations" },
        { "executableNodeTotal", 3 }, { "executableNodeReported", 3 },
        { "executableNodeSubsetTruncated", false },
        { "compositor", duration(value.compositor) },
        { "presentation", duration(value.presentation) },
        { "presentationSemantic", "encoded/handoff" },
        { "particles", duration(value.particles) }, { "generators", duration(value.generators) },
        { "transport", { { "available", true }, { "allocationAvailable", false },
            { "mode", "readback" }, { "zeroCopyFrames", 0 },
            { "zeroCopyAllocationBytes", nullptr }, { "readbackFrames", 1 },
            { "readbackCopiedBytes", 16384 }, { "framesHandedOff", 0 },
            { "framesRendered", 1 }, { "framesPresented", 1 },
            { "exportHandoffFrames", 1 } } },
        { "drops", { { "available", false }, { "framesDropped", nullptr },
            { "reasons", { { "noBuffer", nullptr }, { "transportFailure", nullptr },
                { "renderFailure", nullptr }, { "dimensionMismatch", nullptr },
                { "blitFailure", nullptr }, { "fenceFailure", nullptr },
                { "socketFailure", nullptr } } } } },
        { "backend", { { "available", true }, { "initial", "opengl" },
            { "current", "opengl" }, { "fallbackCount", 0 } } },
        { "resources", { { "available", true }, { "retainedFramesCurrent", 1 },
            { "retainedFramesPeak", 2 }, { "intermediateImagesCurrent", 2 },
            { "intermediateImagesPeak", 2 }, { "retainedBytesCurrent", 16384 },
            { "retainedBytesPeak", 32768 } } },
        { "budget", { { "available", true }, { "backendProfile", "opengl-test" },
            { "backendDeviceIdentity", "deterministic-device" }, { "canvasWidth", 64 },
            { "canvasHeight", 64 },
            { "limits", { { "descriptors", resourceLimits.descriptors },
                { "operations", resourceLimits.operations },
                { "sceneRecords", resourceLimits.sceneRecords },
                { "frameOutputs", resourceLimits.frameOutputs },
                { "liveFrames", resourceLimits.liveFrames },
                { "frameSlots", 7 },
                { "allocatedFrameBytes", resourceLimits.allocatedFrameBytes },
                { "maximumImageDimension", 4096 } } },
            { "usage", { { "planCount", 1 }, { "descriptors", 3 }, { "operations", 3 },
                { "sceneRecords", 0 }, { "frameOutputs", 2 }, { "peakLiveFrames", 2 },
                { "frameSlots", 2 }, { "allocatedFrameBytes", 32768 } } },
            { "provenance", {
                { "maximumImageDimension", resourceLimits.maximumImageDimensionSource },
                { "frameSlots", resourceLimits.frameSlotSource },
                { "allocatedFrameBytes", resourceLimits.allocatedFrameBytesSource },
                { "observedCombinedTextureImageUnits", 24 },
                { "observedMaximumBufferLengthBytes", 123456 },
                { "observedRecommendedWorkingSetBytes", 654321 } } } } },
        { "dimensions", { { "available", true }, { "requestedWidth", 64 },
            { "requestedHeight", 64 }, { "actualWidth", 64 }, { "actualHeight", 64 },
            { "dimensionMismatchCount", 0 }, { "halfResolutionMismatchCount", 0 } } },
        { "recordingContentionDrops", 0 },
        { "layers", nlohmann::json::array({ { { "clipId", 73 },
            { "structuralRevision", 19 }, { "evaluations", 1 },
            { "measuredTotalNs", 17 }, { "measuredMovingNs", 17.0 } } }) },
        { "nodes", nlohmann::json::array({
            { { "clipId", 73 }, { "structuralRevision", 19 }, { "stableNodeId", 101 },
              { "available", true }, { "evaluations", 1 }, { "measuredTotalNs", 3 },
              { "measuredMovingNs", 3.0 } },
            { { "clipId", 73 }, { "structuralRevision", 19 }, { "stableNodeId", 102 },
              { "available", true }, { "evaluations", 1 }, { "measuredTotalNs", 5 },
              { "measuredMovingNs", 5.0 } },
            { { "clipId", 73 }, { "structuralRevision", 19 }, { "stableNodeId", 103 },
              { "available", true }, { "evaluations", 1 }, { "measuredTotalNs", 7 },
              { "measuredMovingNs", 7.0 } } }) }
    };
    check(actual == expected, "production publisher serializes every telemetry field exactly");
}

void checkFailureTaxonomy()
{
    using Reason = videowire::VisualDropReason;
    const std::array<Reason, 7> reasons { Reason::noBuffer, Reason::transportFailure,
        Reason::renderFailure, Reason::dimensionMismatch, Reason::blitFailure,
        Reason::fenceFailure, Reason::socketFailure };
    for (size_t index = 0; index < reasons.size(); ++index)
    {
        videowire::VisualPlanTelemetry telemetry;
        telemetry.recordDrop(reasons[index]);
        const auto value = telemetry.snapshot();
        const std::array<uint64_t, 7> counts { value.droppedNoBuffer,
            value.droppedTransportFailure, value.droppedRenderFailure,
            value.droppedDimensionMismatch, value.droppedBlitFailure,
            value.droppedFenceFailure, value.droppedSocketFailure };
        check(value.dropsObserved && value.framesDropped == 1,
              "failure taxonomy records one dropped frame");
        for (size_t countIndex = 0; countIndex < counts.size(); ++countIndex)
            check(counts[countIndex] == (countIndex == index ? 1u : 0u),
                  "failure taxonomy keeps each reason distinct");
        const auto json = nlohmann::json::parse(videowire::visualTelemetryJson(value).dump());
        const std::array<const char*, 7> names { "noBuffer", "transportFailure",
            "renderFailure", "dimensionMismatch", "blitFailure", "fenceFailure",
            "socketFailure" };
        check(json["drops"]["available"] == true && json["drops"]["framesDropped"] == 1,
              "failure taxonomy JSON marks drops observed");
        for (size_t countIndex = 0; countIndex < names.size(); ++countIndex)
            check(json["drops"]["reasons"][names[countIndex]]
                      == (countIndex == index ? 1u : 0u),
                  "failure taxonomy JSON keeps every serialized reason distinct");
        checkNoSuccess(value, "failure taxonomy never increments success counters");
    }
}
} // namespace

int main()
{
    checkProductionBranch("compiled production branch admits");
    checkProductionTelemetryOwners();
    checkPublicationBoundaries();

    videowire::VisualPlanTelemetry telemetry;
    videowire::VisualPlanTelemetry* rendererTelemetry = nullptr;
    auto mutablePlan = plan();
    const auto resourceLimits = limits();
    std::string diagnostic;
    std::shared_ptr<videowire::VisualPlanExecutionSnapshot> retained;
    {
        videowire::ExportVisualPlanOrchestration orchestration;
        const bool admitted = orchestration.admit({ mutablePlan }, diagnostic, telemetry,
                                                   videowire::VisualBackend::openGL, 64, 64,
                                                   &resourceLimits,
                                                   [&] (videowire::VisualPlanTelemetry* owner)
                                                   { rendererTelemetry = owner; });
        check(admitted, "valid export plan is admitted");
        if (! admitted)
        {
            std::cerr << "admission diagnostic: " << diagnostic << '\n';
            return EXIT_FAILURE;
        }
        retained = orchestration.sharedSnapshot();
        check(retained && orchestration.snapshot() == retained.get(),
              "orchestration exposes one admitted snapshot identity");
        check(rendererTelemetry == &retained->state.telemetry()
                  && &orchestration.telemetryOwner()->telemetry() == rendererTelemetry,
              "renderer and telemetry owner share admitted telemetry");

        mutablePlan.clipId = 999;
        mutablePlan.nodeIds.clear();
        check(retained->plans.size() == 1 && retained->plans[0].clipId == 73
                  && retained->plans[0].nodeIds.size() == 3,
              "source mutation cannot alter admitted plan");

        bool pausedHold = false;
        check(videowire::findAdmittedVisualLayerExecution(
                  retained->state, 73, 19, 0.0, pausedHold) != nullptr && ! pausedHold,
              "production execution lookup publishes a cache hit");

        rendererTelemetry->recordEvaluation(73, 19, 17, false);
        rendererTelemetry->recordNodeEvaluation(73, 19, 101, 3);
        rendererTelemetry->recordNodeEvaluation(73, 19, 102, 5);
        rendererTelemetry->recordNodeEvaluation(73, 19, 103, 7);
        check(videowire::runVisualRendererTelemetryOperation(
                  rendererTelemetry, { videowire::VisualExecutionKind::generator },
                  [] { return 41u; }) == 41u,
              "production generator renderer seam returns its texture");
        check(videowire::runVisualRendererTelemetryOperation(
                  rendererTelemetry, { videowire::VisualExecutionKind::particle, 73, 19, 0, false },
                  [] { return 42u; }) == 42u,
              "production particle renderer seam returns its texture");
        check(orchestration.telemetryOwner()->renderComposite([] { return true; },
                                                               [] (bool ok) { return ok; }),
              "successful composite is observed");
        check(orchestration.telemetryOwner()->observeFrame(64, 64, 256, 16384, 2),
              "valid readback is observed");
        check(orchestration.telemetryOwner()->encodedHandoff(16384, [] { return true; }),
              "valid encoder handoff is observed");

        const auto accepted = telemetry.snapshot();
        check(accepted.graphEvaluations == 1 && accepted.graphMeasuredTotalNs == 17
                  && accepted.graphMeasuredMovingNs == 17.0,
              "graph and layer costs are complete");
        check(accepted.layers.size() == 1 && accepted.layers[0].clipId == 73
                  && accepted.layers[0].structuralRevision == 19
                  && accepted.layers[0].evaluations == 1
                  && accepted.layers[0].measuredTotalNs == 17
                  && accepted.layers[0].measuredMovingNs == 17.0,
              "layer cost fields are complete");
        check(accepted.nodes.size() == 3, "all admitted node costs are present");
        const std::array<uint64_t, 3> nodeCosts { 3, 5, 7 };
        for (size_t index = 0; index < accepted.nodes.size(); ++index)
            check(accepted.nodes[index].clipId == 73
                      && accepted.nodes[index].structuralRevision == 19
                      && accepted.nodes[index].stableNodeId == 101 + static_cast<int>(index)
                      && accepted.nodes[index].available
                      && accepted.nodes[index].evaluations == 1
                      && accepted.nodes[index].measuredTotalNs == nodeCosts[index]
                      && accepted.nodes[index].measuredMovingNs == static_cast<double>(nodeCosts[index]),
                  "every per-node cost field is complete");
        const auto& receipt = accepted.budgetReceipt;
        check(accepted.budgetObserved && receipt.canvasWidth == 64 && receipt.canvasHeight == 64
                  && receipt.usage.planCount == 1 && receipt.usage.descriptors == 3
                  && receipt.usage.operations == 3 && receipt.usage.sceneRecords == 0
                  && receipt.usage.frameOutputs == 2 && receipt.usage.peakLiveFrames == 2
                  && receipt.usage.frameSlots == 2 && receipt.usage.allocatedFrameBytes == 32768,
              "resource usage includes descriptors, outputs, allocation, and reuse");
        check(receipt.limits.backendProfile == "opengl-test"
                  && receipt.limits.backendDeviceIdentity == "deterministic-device"
                  && receipt.limits.maximumImageDimension == 4096
                  && receipt.limits.frameSlots == 7
                  && receipt.limits.observedCombinedTextureImageUnits == 24
                  && receipt.limits.observedMaximumBufferLengthBytes == 123456
                  && receipt.limits.observedRecommendedWorkingSetBytes == 654321
                  && ! receipt.limits.maximumImageDimensionSource.empty()
                  && ! receipt.limits.frameSlotSource.empty()
                  && ! receipt.limits.allocatedFrameBytesSource.empty(),
              "backend device and resource limit provenance are complete");
        check(accepted.backendObserved
                  && accepted.initialBackend == videowire::VisualBackend::openGL
                  && accepted.currentBackend == videowire::VisualBackend::openGL
                  && accepted.fallbackCount == 0,
              "backend identity and fallback count are complete");
        check(accepted.dimensionsObserved && accepted.requestedWidth == 64
                  && accepted.requestedHeight == 64 && accepted.actualWidth == 64
                  && accepted.actualHeight == 64 && accepted.dimensionMismatchCount == 0
                  && accepted.halfResolutionMismatchCount == 0,
              "requested and actual dimensions are complete");
        check(accepted.resourcesObserved && accepted.retainedFramesCurrent == 1
                  && accepted.retainedFramesPeak == 2
                  && accepted.intermediateImagesCurrent == 2
                  && accepted.intermediateImagesPeak == 2
                  && accepted.retainedBytesCurrent == 16384
                  && accepted.retainedBytesPeak == 32768,
              "current and peak runtime resources are complete");
        check(accepted.transportObserved
                  && accepted.transportMode == videowire::VisualTransportMode::readback
                  && accepted.readbackFrames == 1 && accepted.readbackCopiedBytes == 16384
                  && accepted.zeroCopyFrames == 0 && accepted.transportFramesHandedOff == 0,
              "readback and reuse transport fields are complete");
        check(accepted.generators.observed && accepted.generators.count == 1
                  && accepted.particles.observed && accepted.particles.count == 1,
              "generator and particle timing families are complete");
        check(accepted.compositor.observed && accepted.compositor.count == 1
                  && accepted.presentation.observed && accepted.presentation.count == 1
                  && accepted.framesRendered == 1 && accepted.framesPresented == 1
                  && accepted.exportHandoffFrames == 1
                  && accepted.presentationModeObserved
                  && accepted.presentationMode == videowire::VisualPresentationMode::encodedHandoff,
              "successful render and encoder counters are complete");
        check(accepted.failedLowerings == 0 && accepted.lastLoweringError.empty()
                  && accepted.rejectedPlans == 0 && accepted.droppedSamples == 0
                  && accepted.recordingContentionDrops == 0 && ! accepted.dropsObserved,
              "success diagnostics and failure counters are empty");
        check(accepted.planCacheHits == 1 && accepted.planCacheMisses == 1
                  && accepted.planInstalls == 1
                  && accepted.lastPlanLoweringWithinBudget,
              "cache, lowering, and install counters have exact production values");

        const auto json = nlohmann::json::parse(videowire::visualTelemetryJson(accepted).dump());
        check(json["planCacheHits"].get<uint64_t>() >= 1
                  && json["planCacheMisses"].get<uint64_t>() >= 1
                  && json["planInstalls"].get<uint64_t>() >= 1
                  && json["lastPlanLoweringNs"].is_number_integer()
                  && json["planLoweringBudgetNs"] == videowire::kPlanLoweringBudgetNs
                  && json["lastPlanLoweringWithinBudget"].is_boolean(),
              "published JSON contains cache, lowering budget, and install telemetry");
        check(json["executableNodeTotal"] == 3
                  && json["executableNodeSubsetTruncated"] == false
                  && json["generators"]["available"] == true
                  && json["particles"]["available"] == true,
              "published JSON contains executable and renderer timing families");
        check(json["budget"]["limits"]["frameOutputs"] == resourceLimits.frameOutputs
                  && json["budget"]["usage"]["frameOutputs"] == 2
                  && json["budget"]["backendProfile"] == "opengl-test"
                  && json["budget"]["backendDeviceIdentity"] == "deterministic-device",
              "published JSON contains frame output limits and backend identity");
        check(json["dimensions"]["requestedWidth"] == 64
                  && json["dimensions"]["actualHeight"] == 64
                  && json["resources"]["retainedFramesCurrent"] == 1
                  && json["transport"]["exportHandoffFrames"] == 1,
              "published JSON contains dimensions, resources, and export handoff");
        checkExactSerializedTelemetry(accepted, resourceLimits);
    }
    check(rendererTelemetry == nullptr,
          "orchestration destruction unbinds renderer before telemetry destruction");
    check(retained && retained->plans[0].clipId == 73,
          "admitted snapshot can outlive renderer and telemetry dependents");

    videowire::VisualPlanTelemetry rejectedTelemetry;
    videowire::ExportVisualPlanOrchestration rejectedOrchestration;
    auto invalid = plan();
    invalid.allocatedFrameSlotCount = 1;
    diagnostic.clear();
    check(! rejectedOrchestration.admit({ invalid }, diagnostic, rejectedTelemetry,
                                        videowire::VisualBackend::openGL, 64, 64,
                                        &resourceLimits,
                                        [] (videowire::VisualPlanTelemetry*) {}),
          "invalid resource accounting fails admission");
    const auto rejected = rejectedTelemetry.snapshot();
    check(rejectedOrchestration.snapshot() == nullptr && ! diagnostic.empty()
              && rejected.failedLowerings == 1
              && rejected.lastLoweringError == diagnostic,
          "admission failure publishes its complete diagnostic");
    checkNoSuccess(rejected, "admission failure publishes no success telemetry");

    checkFailureTaxonomy();
    if (failures != 0) return EXIT_FAILURE;
    std::cout << "Export visual plan orchestration semantics passed\n";
    return EXIT_SUCCESS;
}
