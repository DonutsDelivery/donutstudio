#include "../src/visual_plan_executor.h"
#include "../src/visual_plan_telemetry_json.h"

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <new>
#include <string>
#include <thread>

namespace
{
std::atomic<uint64_t> allocations { 0 };
}

void* operator new (std::size_t size)
{
    allocations.fetch_add(1, std::memory_order_relaxed);
    if (void* value = std::malloc(size)) return value;
    throw std::bad_alloc();
}
void operator delete (void* value) noexcept { std::free(value); }
void operator delete (void* value, std::size_t) noexcept { std::free(value); }

namespace
{
int failures = 0;
void check (bool condition, const char* message)
{
    if (! condition) { ++failures; std::fprintf (stderr, "FAIL: %s\n", message); }
}

struct FakeLayer
{
    struct Clock { double time = 0.0, timeDelta = 0.0, beat = 0.0; int frame = 0; bool playing = false; };
    struct Effect { bool enabled = false; int type = -1; float params[9] {}; };
    Clock shaderClock;
    std::map<std::string, double> genParams;
    unsigned texture = 0;
    int texWidth = 0, texHeight = 0;
    std::string nativeTextureBackend;
    std::uintptr_t nativeTextureView = 0;
    arbitgpu::NativeTextureViewDescriptor nativeTextureDescriptor;
    std::shared_ptr<const void> nativeTextureOwner;
    bool shaderSource = false;
    bool flatShaderBridge = false;
    videowire::ImmutableShaderOperationPlan shaderOperationPlan;
    std::map<int, std::map<std::string, double>> shaderOperationParameters;
    bool particleSource = false;
    bool particleStateReset = false;
    bool particleTriggerConnected = false;
    int particleTriggerCount = 0;
    float particleTriggerStrength = 0.0f;
    std::uint64_t visualPlanStructuralRevision = 0;
    bool visualPlanTelemetryHold = false;
    bool graphTemporalActive = false;
    int graphTemporalNodeId = 0;
    visualtemporaloperation::Payload graphTemporalPayload;
    int particleNodeId = 0;
    int drawShapeNodeId = 0;
    bool drawShape = false;
    bool inspectionDrawShapeOutput = false;
    bool drawShapeEllipse = false;
    bool drawShapeHasSecondary = false;
    bool drawShapeSecondaryEllipse = false;
    int drawShapeOperation = 0;
    float drawShapeCx = 0.5f, drawShapeCy = 0.5f, drawShapeW = 1.0f, drawShapeH = 1.0f;
    float drawShape2Cx = 0.5f, drawShape2Cy = 0.5f, drawShape2W = 0.0f, drawShape2H = 0.0f;
    float drawShapeR = 1.0f, drawShapeG = 1.0f, drawShapeB = 1.0f, drawShapeA = 1.0f;
    bool pathMatte = false;
    bool pathMatteEllipse = false, pathMatteHasSecondary = false;
    bool pathMatteSecondaryEllipse = false, pathMatteInvert = false;
    int pathMatteOperation = 0;
    float pathMatteCx = 0.5f, pathMatteCy = 0.5f, pathMatteW = 1.0f, pathMatteH = 1.0f;
    float pathMatte2Cx = 0.5f, pathMatte2Cy = 0.5f, pathMatte2W = 0.0f, pathMatte2H = 0.0f;
    float scale = 2.0f, translateX = 1.0f, translateY = 1.0f, rotationDeg = 4.0f;
    float cropLeft = 0.1f, cropRight = 0.1f, cropTop = 0.1f, cropBottom = 0.1f;
    const Effect* effects = reinterpret_cast<const Effect*> (1);
    int effectCount = 2;
    Effect graphFeedbackEffect;
    Effect graphKeyCleanupEffect;
    Effect graphCommonEffect;
    bool graphKeyCleanupActive = false;
    colortransform::Description graphColorTransform;
    bool graphColorTransformActive = false;
    bool graphMotionBlurActive = false;
    unsigned graphMotionTexture = 0;
    visualtemporalsampling::Payload graphMotionBlur;
    float graphKeyCleanupChoke = 0.0f, graphKeyCleanupFeather = 0.0f;
    float graphKeyCleanupEdgeR = 1.0f, graphKeyCleanupEdgeG = 1.0f;
    float graphKeyCleanupEdgeB = 1.0f, graphKeyCleanupEdgeAmount = 0.0f;
    bool graphKeyCleanupMatteView = false;
    bool feedbackHistoryReset = false;
    bool feedbackHistoryHold = false;
    unsigned lutTexture = 4;
    int lutSize = 17;
    int maskType = 2;
    bool maskInvert = true;
    unsigned matteTexture = 0;
    unsigned matteTextureB = 0;
    int matteWidth = 0, matteHeight = 0;
    int matteWidthB = 0, matteHeightB = 0;
    bool matteApply = false, matteInvert = false;
    int matteCombineMode = -1;
    float matteBlack = 0.0f, matteWhite = 1.0f;
    float matteErodeDilate = 0.0f, matteFeather = 0.0f, matteChoke = 0.0f;
    unsigned depthTexture = 0;
    int depthWidth = 0, depthHeight = 0;
    bool depthFog = false;
    int depthEffect = 0;
    float fogNear = 0.0f, fogFar = 1.0f, fogDensity = 1.0f;
    float fogRed = 1.0f, fogGreen = 1.0f, fogBlue = 1.0f, fogAlpha = 1.0f;
    float depthParam0 = 0.0f, depthParam1 = 0.0f, depthParam2 = 0.0f;
    float depthColorRed = 1.0f, depthColorGreen = 1.0f, depthColorBlue = 1.0f;
    float opacity = 0.4f;
    int blendMode = 3;
};

struct FakeProductRenderer
{
    int colorAovExecutions = 0;
    int motionAovExecutions = 0;
    int motionBlurPreparations = 0;
    int temporalFeedbackPreparations = 0;
    int bridgePreparations = 0;
    bool rejectColorAov = false;
    bool rejectMotionAov = false;
    bool rejectTemporalFeedback = false;
    bool rejectFlatShaderBridge = false;
    renderpassoutput::Description publishedColorAov;
    renderpassoutput::Description publishedMotionAov;
    std::optional<videowire::TemporalFeedbackPass> preparedTemporalFeedback;
    std::optional<videowire::TemporalSamplingExecution> preparedMotionBlur;

    int outputWidth() const { return 1920; }
    int outputHeight() const { return 1080; }

    bool prepareTemporalFeedbackPass (const videowire::TemporalFeedbackPass& pass,
                                      std::string& error)
    {
        ++temporalFeedbackPreparations;
        if (rejectTemporalFeedback)
        {
            error = "native temporal feedback backend is unsupported";
            return false;
        }
        preparedTemporalFeedback = pass;
        return true;
    }

    bool replaceColorAovPass (const renderpassoutput::Description& description,
                              std::string& error)
    {
        ++colorAovExecutions;
        if (rejectColorAov)
        {
            error = "native Color AOV backend is unsupported";
            return false;
        }
        publishedColorAov = description;
        return true;
    }

    bool replaceMotionAovPass (const renderpassoutput::Description& description,
                               std::string& error)
    {
        ++motionAovExecutions;
        if (rejectMotionAov)
        {
            error = "native Motion AOV backend is unsupported";
            return false;
        }
        publishedMotionAov = description;
        return true;
    }

    bool prepareMotionBlurPass (const videowire::TemporalSamplingExecution& execution,
                                FakeLayer& layer, std::string& error)
    {
        ++motionBlurPreparations;
        const auto& payload = execution.pass.payload;
        if (! visualtemporalsampling::validate(payload, &error)
            || payload.mode != visualtemporalsampling::Mode::motionBlur
            || execution.transition.action == visualtemporalsampling::TransitionAction::reject)
            return false;
        preparedMotionBlur = execution;
        layer.graphMotionBlurActive = true;
        layer.graphMotionTexture = 9;
        layer.graphMotionBlur = payload;
        error.clear();
        return true;
    }

    bool replaceSceneAovPass (const sceneaov::Payload&, std::string&)
    {
        return true;
    }

    bool replaceAovInspectionPass (const sceneaov::Payload&,
                                   const aovinspection::Payload&,
                                   std::string&)
    {
        return true;
    }

    bool prepareFlatShaderBridge (FakeLayer&, std::string&)
    {
        ++bridgePreparations;
        return ! rejectFlatShaderBridge;
    }
};

videowire::CompiledVisualLayerPlan colorAovPlan()
{
    videowire::CompiledVisualLayerPlan plan;
    plan.clipId = 91;
    plan.producerValidated = true;
    plan.nodeKinds = { std::string (coloraov::kOperationKind) };
    plan.nodeIds = { 811 };
    plan.ports = {
        { 811, 1, 1, "out", "frame", "image", "rgba16f", "linearSRGB" }
    };
    coloraov::Payload payload;
    payload.extent = { 640, 360 };
    plan.operations = {
        { 811, std::string (coloraov::kOperationKind),
          std::string (coloraov::kBackendCapability), coloraov::serialize (payload) }
    };
    return plan;
}

videowire::CompiledVisualLayerPlan motionAovPlan()
{
    videowire::CompiledVisualLayerPlan plan;
    plan.clipId = 92;
    plan.producerValidated = true;
    plan.nodeKinds = { std::string (opticalflowoperation::kOperationKind) };
    plan.nodeIds = { 812 };
    plan.ports = {
        { 812, 4, 2, "out", "frame", "motionVectors", "rg16f", "data" }
    };
    opticalflowoperation::Payload payload;
    payload.extent = { 640, 360 };
    plan.operations = {
        { 812, std::string (opticalflowoperation::kOperationKind),
          std::string (opticalflowoperation::kBackendCapability),
          opticalflowoperation::serialize (payload) }
    };
    return plan;
}

videowire::CompiledVisualLayerPlan importedAnimationPlan()
{
    videowire::CompiledVisualLayerPlan plan;
    plan.clipId = 7;
    plan.structuralRevision = 9;
    plan.producerValidated = true;
    plan.nodeKinds = {
        std::string(visualanimationoperation::kSourceNodeKind),
        std::string(visualanimationoperation::kDeformationNodeKind)
    };
    plan.nodeIds = { 70, 71 };
    const std::array<const char*, 4> types {
        "mesh", "skeleton", "morphTargets", "animationClip"
    };
    for (int port = 0; port < 4; ++port)
    {
        plan.ports.push_back({ 70, port, 1, "out", "control", types[(size_t) port],
                               "unspecified", "unspecified" });
        plan.ports.push_back({ 71, port, 1, "in", "control", types[(size_t) port],
                               "unspecified", "unspecified" });
        plan.edges.push_back({ 70, port, 71, port });
    }
    plan.ports.push_back({ 70, 4, 1, "out", "control", "scene3D",
                           "unspecified", "unspecified" });
    plan.ports.push_back({ 71, 4, 1, "out", "control", "mesh",
                           "unspecified", "unspecified" });

    visualanimationimport::Request request;
    request.sourceStableId = 71;
    request.deformationStableId = 72;
    request.schedule = { 71, 72 };
    request.asset.id = "model-asset-1";
    request.asset.version = 3;
    request.asset.contentSha256 = std::string(64, 'a');
    request.asset.sourceMediaType = "model/gltf-binary";
    request.asset.sourceByteSize = 4096;
    request.sceneIndex = 0;
    request.clipName = "Walk";
    plan.operations = {
        { 70, std::string(visualanimationoperation::kSourceNodeKind),
          std::string(visualanimationoperation::kSourceBackendCapability), "" },
        { 71, std::string(visualanimationoperation::kDeformationNodeKind),
          std::string(visualanimationoperation::kDeformationBackendCapability),
          visualanimationoperation::encode(request) }
    };
    return plan;
}

videowire::CompiledVisualLayerPlan directPlan()
{
    videowire::CompiledVisualLayerPlan plan;
    plan.clipId = 7;
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

videowire::CompiledVisualLayerPlan mediumBudgetPlan (int clipId, int firstNodeId)
{
    videowire::CompiledVisualLayerPlan plan;
    plan.clipId = clipId;
    plan.structuralRevision = 17;
    plan.producerValidated = true;
    plan.nodeKinds = { "video.source", "video.transform", "video.effects",
                       "video.mask.shape", "video.blend", "video.out" };
    for (int index = 0; index < 6; ++index)
    {
        const int nodeId = firstNodeId + index;
        plan.nodeIds.push_back(nodeId);
        plan.operations.push_back({ nodeId, plan.nodeKinds[(size_t) index],
            index == 0 ? "source-decode" : "native-gpu", "" });
        if (index > 0)
            plan.ports.push_back({ nodeId, 0, 1, "in", "frame", "image", "rgba8", "sRGB" });
        if (index + 1 < 6)
        {
            plan.ports.push_back({ nodeId, 1, 1, "out", "frame", "image", "rgba8", "sRGB" });
            plan.edges.push_back({ nodeId, 1, nodeId + 1, 0 });
        }
    }
    videowire::VisualPlanResourceUsage usage;
    std::string error;
    if (! videowire::accountVisualPlanResources(plan, 64, 64, usage, error))
        return {};
    plan.descriptorCount = usage.descriptors;
    plan.operationCount = usage.operations;
    plan.sceneRecordCount = usage.sceneRecords;
    plan.frameOutputCount = usage.frameOutputs;
    plan.peakLiveFrameCount = usage.peakLiveFrames;
    plan.allocatedFrameSlotCount = usage.frameSlots;
    return plan;
}

videowire::CompiledVisualLayerPlan keyCleanupPlan()
{
    auto plan = directPlan();
    plan.nodeKinds = { "video.source", "visual.key.cleanup", "video.out" };
    plan.nodeIds = { 11, 13, 12 };
    plan.ports = {
        { 11, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 13, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 13, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 12, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    plan.edges = { { 11, 0, 13, 0 }, { 13, 1, 12, 0 } };
    plan.operations = {
        { 11, "video.source", "source-decode", "" },
        { 13, "visual.key.cleanup", "native-gpu",
          "<KeyCleanup schemaVersion=\"1\" keyR=\"0.1\" keyG=\"0.9\" keyB=\"0.2\" tolerance=\"0.25\" softness=\"0.15\" despill=\"0.75\" choke=\"0\" feather=\"0\" edgeRed=\"1\" edgeGreen=\"1\" edgeBlue=\"1\" edgeAmount=\"0\" view=\"0\"/>" },
        { 12, "video.out", "native-gpu", "" }
    };
    return plan;
}

videowire::CompiledVisualLayerPlan mattePlan()
{
    auto plan = directPlan();
    plan.nodeKinds = { "video.source", "visual.matte.asset", "visual.matte.refine",
                       "visual.matte.apply", "video.out" };
    plan.nodeIds = { 11, 41, 42, 43, 12 };
    plan.edges = { { 11, 0, 43, 0 }, { 41, 0, 42, 0 },
                   { 42, 1, 43, 1 }, { 43, 2, 12, 0 } };
    plan.operations = {
        { 11, "video.source", "source-decode", "" },
        { 41, "visual.matte.asset", "source-decode",
          "<MatteAssetBinding matteAssetId=\"matte-1\" state=\"available\" cacheKey=\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\" contentReceipt=\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\" framePrefix=\"subject-\" frameExtension=\".rgba\" firstFrame=\"42\" frameDigits=\"3\" backend=\"rgba-cpu-decode-native-gpu-upload\"/>" },
        { 42, "visual.matte.refine", "native-gpu",
          "<NodeParams invert=\"1\" black=\"-2\" white=\"3\" erodeDilate=\"20\" feather=\"20\" choke=\"-2\"/>" },
        { 43, "visual.matte.apply", "native-gpu", "" },
        { 12, "video.out", "native-gpu", "" }
    };
    return plan;
}

videowire::CompiledVisualLayerPlan combinedMattePlan()
{
    auto plan = mattePlan();
    plan.nodeKinds = { "video.source", "visual.matte.asset", "visual.matte.asset",
                       "visual.matte.combine", "visual.matte.refine", "visual.matte.apply", "video.out" };
    plan.nodeIds = { 11, 41, 44, 45, 42, 43, 12 };
    plan.edges = { {11,0,43,0}, {41,0,45,0}, {44,0,45,1}, {45,2,42,0},
                   {42,1,43,1}, {43,2,12,0} };
    auto bindingB = plan.operations[1];
    bindingB.nodeId = 44;
    bindingB.payloadXml = "<MatteAssetBinding matteAssetId=\"matte-2\" matteAssetVersion=\"v2\" state=\"available\" cacheKey=\"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\" contentReceipt=\"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\" framePrefix=\"other-\" frameExtension=\".rgba\" firstFrame=\"7\" frameDigits=\"3\" fps=\"24\" frames=\"8\" backend=\"rgba-cpu-decode-native-gpu-upload\"/>";
    plan.operations = { plan.operations[0], plan.operations[1], bindingB,
        {45, "visual.matte.combine", "native-gpu", "<NodeParams mode=\"3\"/>"},
        plan.operations[2], plan.operations[3], plan.operations[4] };
    return plan;
}

videowire::CompiledVisualLayerPlan reusedMattePlan()
{
    auto plan = combinedMattePlan();
    plan.nodeKinds.erase(plan.nodeKinds.begin() + 2);
    plan.nodeIds.erase(plan.nodeIds.begin() + 2);
    plan.operations.erase(plan.operations.begin() + 2);
    plan.edges[2].fromNodeId = 41;
    plan.ports = {
        { 11, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 41, 0, 1, "out", "frame", "mask", "r8", "linearSRGB" },
        { 45, 0, 1, "in", "frame", "mask", "r8", "linearSRGB" },
        { 45, 1, 1, "in", "frame", "mask", "r8", "linearSRGB" },
        { 45, 2, 1, "out", "frame", "mask", "r8", "linearSRGB" },
        { 42, 0, 1, "in", "frame", "mask", "r8", "linearSRGB" },
        { 42, 1, 1, "out", "frame", "mask", "r8", "linearSRGB" },
        { 43, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 43, 1, 1, "in", "frame", "mask", "r8", "linearSRGB" },
        { 43, 2, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 12, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    return plan;
}

videowire::CompiledVisualLayerPlan depthPlan()
{
    auto plan = directPlan();
    plan.nodeKinds = { "video.source", "visual.depth.asset", "visual.depth.fog", "video.out" };
    plan.nodeIds = { 11, 61, 62, 12 };
    plan.ports = {
        { 11, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 61, 0, 0, "out", "none", "unspecified", "unspecified", "unspecified" },
        { 61, 1, 1, "out", "frame", "depth", "r16", "linearSRGB" },
        { 62, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 62, 1, 1, "in", "frame", "depth", "r16", "linearSRGB" },
        { 62, 2, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 12, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    plan.edges = { { 11, 0, 62, 0 }, { 61, 1, 62, 1 }, { 62, 2, 12, 0 } };
    plan.operations = {
        { 11, "video.source", "source-decode", "" },
        { 61, "visual.depth.asset", "source-decode",
          "<DepthAssetBinding depthAssetId=\"depth-1\" depthAssetVersion=\"depth-sequence-v1\" state=\"available\" cacheKey=\"depth-job-1-abc/published\" contentReceipt=\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\" analysisReceipt=\"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\" framePrefix=\"depth_\" frameExtension=\".png\" firstFrame=\"1\" frameDigits=\"6\" width=\"2\" height=\"2\" fps=\"24\" frames=\"3\" format=\"r16-unorm\"/>" },
        { 62, "visual.depth.fog", "native-gpu",
          "<NodeParams near=\"-1\" far=\"2\" density=\"99\" red=\"0.2\" green=\"0.3\" blue=\"0.4\" alpha=\"2\"/>" },
        { 12, "video.out", "native-gpu", "" }
    };
    return plan;
}
}

int main()
{
    std::string error;
    error.reserve(256);
    videowire::CompiledVisualLayerPlan multipleGeometry;
    multipleGeometry.structuralRevision = 1;
    multipleGeometry.operations.push_back({1, "geometry.core.runtime", "native-gpu", ""});
    multipleGeometry.operations.push_back({2, "geometry.core.runtime", "native-gpu", ""});
    FakeLayer rejectedGeometryLayer;
    const videowire::VisualPlanEvaluationContext geometryContext {
        11, false, visualtemporalsampling::EvaluationMode::preview, 12, 13
    };
    const auto beforeMultiGeometry = allocations.load(std::memory_order_relaxed);
    check(!videowire::admitGeometryCoreOperations(
              multipleGeometry, videohelper::geometry::PlanUse::preview,
              &geometryContext, 64, 64, rejectedGeometryLayer, error)
              && allocations.load(std::memory_order_relaxed) == beforeMultiGeometry
              && error == "multiple Geometry Core image outputs require an explicit compositor",
          "multiple Geometry Core outputs reject before decode or native allocation");
    videowire::VisualEventScheduleBinding eventSchedule;
    eventSchedule.clipId = 7;
    eventSchedule.nodeId = 70;
    eventSchedule.portId = 0;
    eventSchedule.sessionRevision = 3;
    eventSchedule.triggers = {
        { 0, 1.0, 0.25f, 1 }, { 32, 1.5, 0.75f, 2 }, { 64, 1.5, 0.5f, 3 }
    };
    const std::vector<videowire::VisualEventScheduleBinding> eventSchedules { eventSchedule };
    videowire::VisualEventTriggerCursor viewportCursor (false);
    check (viewportCursor.consume(eventSchedules, 7, 70, 0, 3, 1.25).count == 0,
           "viewport cursor attaches without replaying historical triggers");
    const auto viewportAdvance = viewportCursor.consume(eventSchedules, 7, 70, 0, 3, 1.5);
    check (viewportAdvance.count == 2 && viewportAdvance.strongest == 0.75f
           && viewportCursor.consume(eventSchedules, 7, 70, 0, 3, 1.5).count == 0,
           "viewport cursor consumes equal-beat triggers once in deterministic order");
    check (viewportCursor.consume(eventSchedules, 7, 70, 0, 3, 1.0).count == 0,
           "viewport backward seek resets without bursting retained history");
    check (viewportCursor.consume(eventSchedules, 7, 70, 0, 4, 2.0).count == 0,
           "viewport revision replacement resets independently");
    videowire::VisualEventTriggerCursor exportCursor (true);
    check (exportCursor.consume(eventSchedules, 7, 70, 0, 3, 1.0).count == 1
           && exportCursor.consume(eventSchedules, 7, 70, 0, 3, 1.5).count == 2,
           "export cursor deterministically replays schedules from timeline start");
    check (exportCursor.consume(eventSchedules, 7, 70, 1, 3, 2.0).count == 0
           && exportCursor.consume(eventSchedules, 8, 70, 0, 3, 2.0).count == 0,
           "cursor never retargets a stable sink by presentation identity");
    check (exportCursor.consume(eventSchedules, 7, 70, 0, 3, 1.0).count == 1,
           "export backward seek reconstructs its own trigger cursor");
    const auto colorAov = colorAovPlan();
    const std::vector<videowire::CompiledVisualLayerPlan> colorAovPlans { colorAov };
    FakeLayer viewportColorLayer;
    FakeLayer exportColorLayer;
    FakeProductRenderer viewportRenderer;
    FakeProductRenderer exportRenderer;
    check (videowire::executeVisualLayerPlanForRenderer (
               viewportRenderer, colorAovPlans, colorAov.clipId, viewportColorLayer, error, videohelper::geometry::PlanUse::preview, nullptr)
           && videowire::executeVisualLayerPlanForRenderer (
               exportRenderer, colorAovPlans, colorAov.clipId, exportColorLayer, error, videohelper::geometry::PlanUse::preview, nullptr)
           && viewportRenderer.colorAovExecutions == 1
           && viewportRenderer.bridgePreparations == 1
           && exportRenderer.colorAovExecutions == 1
           && exportRenderer.bridgePreparations == 1
           && viewportRenderer.publishedColorAov.extent == renderpassoutput::Extent { 640, 360 }
           && exportRenderer.publishedColorAov.extent == viewportRenderer.publishedColorAov.extent
           && viewportRenderer.publishedColorAov.attachments.size() == 1
           && exportRenderer.publishedColorAov.attachments.size() == 1
           && viewportRenderer.publishedColorAov.attachments.front().output
                == renderpassoutput::Output::Color
           && exportRenderer.publishedColorAov.attachments.front().output
                == viewportRenderer.publishedColorAov.attachments.front().output
           && exportRenderer.publishedColorAov.attachments.front().format
                == viewportRenderer.publishedColorAov.attachments.front().format
           && exportRenderer.publishedColorAov.attachments.front().colorSpace
                == viewportRenderer.publishedColorAov.attachments.front().colorSpace,
           "viewport and export execute and publish the same typed Color AOV description");
    FakeLayer unsupportedColorLayer;
    FakeProductRenderer unsupportedRenderer;
    unsupportedRenderer.rejectColorAov = true;
    error.clear();
    check (! videowire::executeVisualLayerPlanForRenderer (
               unsupportedRenderer, colorAovPlans, colorAov.clipId, unsupportedColorLayer, error, videohelper::geometry::PlanUse::preview, nullptr)
           && unsupportedRenderer.colorAovExecutions == 1
           && unsupportedRenderer.bridgePreparations == 0
           && error == "native Color AOV backend is unsupported",
           "product execution fails closed when native Color AOV submission is unsupported");

    const auto motionAov = motionAovPlan();
    const std::vector<videowire::CompiledVisualLayerPlan> motionAovPlans { motionAov };
    FakeLayer viewportMotionLayer;
    FakeLayer exportMotionLayer;
    FakeProductRenderer viewportMotionRenderer;
    FakeProductRenderer exportMotionRenderer;
    check (videowire::executeVisualLayerPlanForRenderer (
               viewportMotionRenderer, motionAovPlans, motionAov.clipId,
               viewportMotionLayer, error, videohelper::geometry::PlanUse::preview, nullptr)
           && videowire::executeVisualLayerPlanForRenderer (
               exportMotionRenderer, motionAovPlans, motionAov.clipId,
               exportMotionLayer, error, videohelper::geometry::PlanUse::preview, nullptr)
           && viewportMotionRenderer.motionAovExecutions == 1
           && viewportMotionRenderer.bridgePreparations == 1
           && exportMotionRenderer.motionAovExecutions == 1
           && exportMotionRenderer.bridgePreparations == 1
           && viewportMotionRenderer.publishedMotionAov.extent
                == renderpassoutput::Extent { 640, 360 }
           && exportMotionRenderer.publishedMotionAov.extent
                == viewportMotionRenderer.publishedMotionAov.extent
           && viewportMotionRenderer.publishedMotionAov.attachments.size() == 1
           && exportMotionRenderer.publishedMotionAov.attachments.size() == 1
           && viewportMotionRenderer.publishedMotionAov.attachments.front().output
                == renderpassoutput::Output::Motion
           && viewportMotionRenderer.publishedMotionAov.attachments.front().format
                == renderpassoutput::PixelFormat::RG16Float
           && viewportMotionRenderer.publishedMotionAov.attachments.front().colorSpace
                == renderpassoutput::ColorSpace::Data,
           "viewport and export execute and publish the same typed Motion AOV description");
    FakeLayer unsupportedMotionLayer;
    FakeProductRenderer unsupportedMotionRenderer;
    unsupportedMotionRenderer.rejectMotionAov = true;
    error.clear();
    check (! videowire::executeVisualLayerPlanForRenderer (
               unsupportedMotionRenderer, motionAovPlans, motionAov.clipId,
               unsupportedMotionLayer, error, videohelper::geometry::PlanUse::preview, nullptr)
           && unsupportedMotionRenderer.motionAovExecutions == 1
           && unsupportedMotionRenderer.bridgePreparations == 0
           && error == "native Motion AOV backend is unsupported",
           "product execution fails closed when native Motion AOV submission is unsupported");
    auto direct = directPlan();
    FakeLayer layer;
    videowire::VisualLayerExecution execution;

    const auto importedAnimation = importedAnimationPlan();
    check (videowire::compileVisualLayerExecution(importedAnimation, execution, error)
           && execution.importedAnimation.has_value()
           && execution.importedAnimation->asset.id == "model-asset-1"
           && execution.importedAnimation->asset.sourceByteSize == 4096,
           "exact imported-animation schedule lowers its path-free asset identity");

    auto missingImportedSource = importedAnimation;
    missingImportedSource.nodeKinds.erase(missingImportedSource.nodeKinds.begin());
    missingImportedSource.nodeIds.erase(missingImportedSource.nodeIds.begin());
    missingImportedSource.operations.erase(missingImportedSource.operations.begin());
    error.clear();
    check (! videowire::compileVisualLayerExecution(
                missingImportedSource, execution, error)
           && error == "imported animation requires one exact source-to-deformation typed schedule",
           "imported-animation lowering rejects a deformation without dereferencing a missing source");

    auto missingImportedDeformation = importedAnimation;
    missingImportedDeformation.nodeKinds.pop_back();
    missingImportedDeformation.nodeIds.pop_back();
    missingImportedDeformation.operations.pop_back();
    error.clear();
    check (! videowire::compileVisualLayerExecution(
                missingImportedDeformation, execution, error)
           && error == "imported animation requires one exact source-to-deformation typed schedule",
           "imported-animation lowering rejects a source without dereferencing a missing deformation");

    FakeProductRenderer missingAnimationRenderer;
    FakeLayer missingAnimationLayer;
    error.clear();
    check (! videowire::executeVisualLayerPlanForRenderer(
                missingAnimationRenderer, { importedAnimation }, 7,
                missingAnimationLayer, error, videohelper::geometry::PlanUse::preview, nullptr)
           && missingAnimationRenderer.bridgePreparations == 0
           && error == "imported animation plan requires a retained native compositor frame binding",
           "admitted imported animation fails closed until the product retains and binds its native frame");

    const auto keyCleanup = keyCleanupPlan();
    FakeLayer keyLayer;
    check (videowire::executeVisualLayerPlan({ keyCleanup }, 7, keyLayer, error, videohelper::geometry::PlanUse::preview)
           && keyLayer.effects == &keyLayer.graphKeyCleanupEffect
           && keyLayer.effectCount == 1 && keyLayer.graphKeyCleanupEffect.enabled
           && keyLayer.graphKeyCleanupEffect.type == 17
           && keyLayer.graphKeyCleanupEffect.params[0] == 0.1f
           && keyLayer.graphKeyCleanupEffect.params[1] == 0.9f
           && keyLayer.graphKeyCleanupEffect.params[2] == 0.2f
           && keyLayer.graphKeyCleanupEffect.params[3] == 0.25f
           && keyLayer.graphKeyCleanupEffect.params[4] == 0.15f
           && keyLayer.graphKeyCleanupEffect.params[5] == 0.75f,
           "typed key cleanup lowers to the existing bounded ChromaKey GPU pass");
    FakeLayer repeatedKeyLayer;
    check (videowire::executeVisualLayerPlan({ keyCleanup }, 7, repeatedKeyLayer, error, videohelper::geometry::PlanUse::preview)
           && repeatedKeyLayer.graphKeyCleanupEffect.type == keyLayer.graphKeyCleanupEffect.type
           && std::equal(std::begin(repeatedKeyLayer.graphKeyCleanupEffect.params),
                         std::end(repeatedKeyLayer.graphKeyCleanupEffect.params),
                         std::begin(keyLayer.graphKeyCleanupEffect.params)),
           "typed key cleanup lowering is deterministic");
    auto fullKeyCleanup = keyCleanup;
    fullKeyCleanup.operations[1].payloadXml.replace(
        fullKeyCleanup.operations[1].payloadXml.find("feather=\"0\""),
        std::string("feather=\"0\"").size(), "feather=\"1\"");
    check (videowire::executeVisualLayerPlan({ fullKeyCleanup }, 7, keyLayer, error, videohelper::geometry::PlanUse::preview)
           && keyLayer.graphKeyCleanupActive
           && keyLayer.graphKeyCleanupFeather == 1.0f,
           "authored key cleanup reaches the shared bounded native-GPU pass");
    auto malformedKeyCleanup = keyCleanup;
    malformedKeyCleanup.operations[1].payloadXml.replace(
        malformedKeyCleanup.operations[1].payloadXml.find("despill=\"0.75\""),
        std::string("despill=\"0.75\"").size(), "despill=\"nan\"");
    check (! videowire::executeVisualLayerPlan({ malformedKeyCleanup }, 7, keyLayer, error, videohelper::geometry::PlanUse::preview)
           && error == "visual.key.cleanup has a malformed or out-of-bounds immutable payload",
           "typed key cleanup rejects malformed non-finite payloads");
    auto mistypedKeyCleanup = keyCleanup;
    mistypedKeyCleanup.ports[2].dataType = "depth";
    check (! videowire::executeVisualLayerPlan({ mistypedKeyCleanup }, 7, keyLayer, error, videohelper::geometry::PlanUse::preview)
           && error == "visual.key.cleanup requires exact Frame<Image> input/output descriptors",
           "typed key cleanup rejects inexact Frame output descriptors");
    auto misclassifiedKeyCleanup = keyCleanup;
    misclassifiedKeyCleanup.operations[1].backendCapability = "control-eval";
    check (! videowire::executeVisualLayerPlan({ misclassifiedKeyCleanup }, 7, keyLayer, error, videohelper::geometry::PlanUse::preview),
           "typed key cleanup rejects a false backend capability classification");

    auto sharpen = keyCleanupPlan();
    sharpen.nodeKinds[1] = "visual.effect.sharpen";
    sharpen.operations[1].kind = "visual.effect.sharpen";
    sharpen.operations[1].payloadXml =
        "<CommonEffect schemaVersion=\"1\" effect=\"sharpen\" amount=\"2.5\"/>";
    FakeLayer sharpenLayer;
    check (videowire::executeVisualLayerPlan({ sharpen }, 7, sharpenLayer, error, videohelper::geometry::PlanUse::preview)
           && sharpenLayer.effects == &sharpenLayer.graphCommonEffect
           && sharpenLayer.effectCount == 1 && sharpenLayer.graphCommonEffect.enabled
           && sharpenLayer.graphCommonEffect.type == 7
           && sharpenLayer.graphCommonEffect.params[0] == 2.5f,
           "graph-native sharpen lowers to the shared renderer effect pass");
    auto unsupportedSharpen = sharpen;
    unsupportedSharpen.operations[1].payloadXml =
        "<CommonEffect schemaVersion=\"1\" effect=\"sharpen\" amount=\"2.5\" mix=\"0.5\"/>";
    check (! videowire::executeVisualLayerPlan({ unsupportedSharpen }, 7, sharpenLayer, error, videohelper::geometry::PlanUse::preview),
           "graph-native sharpen fails closed on unsupported controls");

    auto saturation = sharpen;
    saturation.nodeKinds[1] = "visual.color.saturation";
    saturation.operations[1].kind = "visual.color.saturation";
    saturation.operations[1].payloadXml =
        "<CommonEffect schemaVersion=\"1\" effect=\"saturation\" amount=\"1.75\"/>";
    FakeLayer saturationLayer;
    check (videowire::executeVisualLayerPlan({ saturation }, 7, saturationLayer, error, videohelper::geometry::PlanUse::preview)
           && saturationLayer.effects == &saturationLayer.graphCommonEffect
           && saturationLayer.effectCount == 1 && saturationLayer.graphCommonEffect.enabled
           && saturationLayer.graphCommonEffect.type == 2
           && saturationLayer.graphCommonEffect.params[0] == 1.75f,
           "graph-native saturation lowers to the shared renderer color rack");

    auto exposure = sharpen;
    exposure.nodeKinds[1] = "visual.color.exposure";
    exposure.operations[1].kind = "visual.color.exposure";
    exposure.operations[1].payloadXml =
        "<CommonEffect schemaVersion=\"1\" effect=\"exposure\" stops=\"-1.5\"/>";
    FakeLayer exposureLayer;
    check (videowire::executeVisualLayerPlan({ exposure }, 7, exposureLayer, error, videohelper::geometry::PlanUse::preview)
           && exposureLayer.effects == &exposureLayer.graphCommonEffect
           && exposureLayer.effectCount == 1 && exposureLayer.graphCommonEffect.enabled
           && exposureLayer.graphCommonEffect.type == 4
           && exposureLayer.graphCommonEffect.params[0] == -1.5f,
           "graph-native exposure lowers to the shared renderer color rack");

    struct CommonEffectCase
    {
        const char* graphKind;
        const char* payload;
        int rendererType;
        std::array<float, 2> parameters;
        std::size_t parameterCount;
    };
    const std::array<CommonEffectCase, 6> additionalCommonEffects {{
        { "visual.color.brightness",
          "<CommonEffect schemaVersion=\"1\" effect=\"brightness\" amount=\"-0.25\"/>",
          0, { -0.25f, 0.0f }, 1 },
        { "visual.color.contrast",
          "<CommonEffect schemaVersion=\"1\" effect=\"contrast\" amount=\"1.5\"/>",
          1, { 1.5f, 0.0f }, 1 },
        { "visual.color.hue",
          "<CommonEffect schemaVersion=\"1\" effect=\"hue\" shift=\"45\"/>",
          3, { 45.0f, 0.0f }, 1 },
        { "visual.color.gamma",
          "<CommonEffect schemaVersion=\"1\" effect=\"gamma\" value=\"2.2\"/>",
          5, { 2.2f, 0.0f }, 1 },
        { "visual.effect.blur",
          "<CommonEffect schemaVersion=\"1\" effect=\"blur\" radius=\"8.5\"/>",
          6, { 8.5f, 0.0f }, 1 },
        { "visual.effect.vignette",
          "<CommonEffect schemaVersion=\"1\" effect=\"vignette\" amount=\"0.7\" softness=\"0.25\"/>",
          8, { 0.7f, 0.25f }, 2 }
    }};
    for (const auto& test : additionalCommonEffects)
    {
        auto plan = sharpen;
        plan.nodeKinds[1] = test.graphKind;
        plan.operations[1].kind = test.graphKind;
        plan.operations[1].payloadXml = test.payload;
        FakeLayer layer;
        bool parametersMatch = true;
        const bool executed = videowire::executeVisualLayerPlan({ plan }, 7, layer, error, videohelper::geometry::PlanUse::preview);
        for (std::size_t index = 0; index < test.parameterCount; ++index)
            parametersMatch = parametersMatch
                && layer.graphCommonEffect.params[index] == test.parameters[index];
        const auto message = std::string("graph-native ") + test.graphKind
            + " lowers to the shared renderer effect rack";
        check (executed && layer.effects == &layer.graphCommonEffect
               && layer.effectCount == 1 && layer.graphCommonEffect.enabled
               && layer.graphCommonEffect.type == test.rendererType && parametersMatch,
               message.c_str());
    }

    auto incompleteVignette = sharpen;
    incompleteVignette.nodeKinds[1] = "visual.effect.vignette";
    incompleteVignette.operations[1].kind = "visual.effect.vignette";
    incompleteVignette.operations[1].payloadXml =
        "<CommonEffect schemaVersion=\"1\" effect=\"vignette\" amount=\"0.7\"/>";
    FakeLayer incompleteVignetteLayer;
    check (!videowire::executeVisualLayerPlan(
               { incompleteVignette }, 7, incompleteVignetteLayer, error, videohelper::geometry::PlanUse::preview),
           "graph-native vignette rejects an incomplete immutable payload");

    auto convertedGamma = sharpen;
    convertedGamma.nodeKinds[1] = "visual.color.gamma";
    convertedGamma.operations[1].kind = "visual.color.gamma";
    convertedGamma.operations[1].payloadXml =
        "<CommonEffect schemaVersion=\"1\" effect=\"gamma\" value=\"2.2\"/>";
    convertedGamma.ports[2].colorSpace = "linearSRGB";
    FakeLayer convertedGammaLayer;
    check (! videowire::executeVisualLayerPlan(
               { convertedGamma }, 7, convertedGammaLayer, error, videohelper::geometry::PlanUse::preview)
           && error == "visual.color.gamma requires exact Frame<Image> input/output descriptors",
           "graph-native color effects reject implicit typed-edge color conversion");

    auto mismatchedColorPayload = exposure;
    mismatchedColorPayload.operations[1].payloadXml =
        "<CommonEffect schemaVersion=\"1\" effect=\"saturation\" amount=\"1.5\"/>";
    check (!videowire::executeVisualLayerPlan(
               { mismatchedColorPayload }, 7, exposureLayer, error, videohelper::geometry::PlanUse::preview),
           "graph-native color node rejects a payload for another admitted kind");

    videowire::ExecutableCommonEffectPayload unavailablePayload;
    videowire::CommonEffectPayloadFailure unavailableFailure =
        videowire::CommonEffectPayloadFailure::none;
    check (!videowire::parseExecutableCommonEffectPayload(
               exposure.operations[1].payloadXml, unavailablePayload, &unavailableFailure,
               commoneffect::BackendSupport { true, true, false })
           && unavailableFailure
               == videowire::CommonEffectPayloadFailure::productionBackendUnavailable,
           "graph-native exposure reports truthful backend unavailability");

    auto particles = direct;
    particles.nodeKinds = { "visual.particles", "video.out" };
    particles.nodeIds = { 51, 12 };
    particles.edges = { { 51, 1, 12, 0 } };
    particles.operations = {
        { 51, "visual.particles", "native-gpu", "<NodeParams seed=\"70000\" count=\"9000\" lifetime=\"20\" size=\"0\" speed=\"9\" red=\"-1\" green=\"2\" blue=\"0.4\" alpha=\"2\"/>" },
        { 12, "video.out", "native-gpu", "" }
    };
    FakeLayer particleLayer;
    for (const char* key : { "nativeBuiltin", "seed", "count", "lifetime", "size",
                             "speed", "red", "green", "blue", "alpha" })
        particleLayer.genParams.emplace(key, 0.0);
    particleLayer.shaderClock.time = 1.25;
    particleLayer.shaderClock.frame = 30;
    particleLayer.shaderClock.timeDelta = 1.0 / 24.0;
    particleLayer.shaderClock.playing = false;
    check (videowire::executeVisualLayerPlan({ particles }, 7, particleLayer, error,videohelper::geometry::PlanUse::preview,
                                             nullptr, nullptr, nullptr, 2.5)
           && particleLayer.particleSource && ! particleLayer.shaderSource
           && particleLayer.genParams["nativeBuiltin"] == 1.0
           && particleLayer.genParams["seed"] == 65535.0
           && particleLayer.genParams["count"] == 4096.0
           && particleLayer.genParams["lifetime"] == 10.0
           && particleLayer.genParams["size"] == 1.0
           && particleLayer.genParams["speed"] == 4.0
           && particleLayer.genParams["red"] == 0.0
           && particleLayer.genParams["green"] == 1.0
           && particleLayer.shaderClock.time == 1.25
           && particleLayer.shaderClock.frame == 30
           && particleLayer.shaderClock.timeDelta == 1.0 / 24.0
           && ! particleLayer.shaderClock.playing,
           "curated particle source clamps payload and preserves the canonical ShaderClock");
    auto particleState = std::make_unique<videowire::VisualPlanExecutionState>();
    particleState->admitPlans({ particles });
    const std::vector<videowire::CompiledVisualLayerPlan> admittedParticles { particles };
    const auto particleMapSize = particleLayer.genParams.size();
    const auto particleAllocations = allocations.load(std::memory_order_relaxed);
    check (videowire::executeVisualLayerPlan(admittedParticles, 7, particleLayer, error,videohelper::geometry::PlanUse::preview,
                                             nullptr, nullptr, particleState.get(), 3.0)
           && particleLayer.genParams.size() == particleMapSize
           && allocations.load(std::memory_order_relaxed) == particleAllocations,
           "typed particle render updates only pre-existing parameters without allocation or insertion");
    auto particleSchedule = eventSchedule;
    particleSchedule.nodeId = 51;
    const std::vector<videowire::VisualEventScheduleBinding> particleSchedules { particleSchedule };
    videowire::VisualEventTriggerCursor particleCursor (true);
    particleLayer.shaderClock.beat = 1.5;
    check (videowire::executeVisualLayerPlan(admittedParticles, 7, particleLayer, error,videohelper::geometry::PlanUse::preview,
                                             nullptr, nullptr, particleState.get(), 3.0,
                                             &particleSchedules, &particleCursor)
           && particleLayer.particleTriggerConnected
           && particleLayer.particleTriggerCount == 3
           && particleLayer.particleTriggerStrength == 0.75f,
           "typed particle execution consumes the exact Event sink at the canonical beat");
    particleLayer.shaderClock.beat = 1.5;
    check (videowire::executeVisualLayerPlan(admittedParticles, 7, particleLayer, error,videohelper::geometry::PlanUse::preview,
                                             nullptr, nullptr, particleState.get(), 3.0,
                                             &particleSchedules, &particleCursor)
           && particleLayer.particleTriggerConnected && particleLayer.particleTriggerCount == 0,
           "typed particle execution does not retrigger a held frame");
    particleSchedule.nodeId = 52;
    const std::vector<videowire::VisualEventScheduleBinding> staleSchedules { particleSchedule };
    check (videowire::executeVisualLayerPlan(admittedParticles, 7, particleLayer, error,videohelper::geometry::PlanUse::preview,
                                             nullptr, nullptr, particleState.get(), 3.0,
                                             &staleSchedules, &particleCursor)
           && ! particleLayer.particleTriggerConnected,
           "typed particle execution never retargets a stale visual NodeId");
    auto malformedParticles = particles;
    malformedParticles.edges[0].fromPort = 0;
    check (! videowire::executeVisualLayerPlan({ malformedParticles }, 7, particleLayer, error, videohelper::geometry::PlanUse::preview)
           && error == "visual.particles has unsupported production topology",
           "particle source preserves exact image port and fails closed on malformed topology");
    particles.operations[0].backendCapability = "control-eval";
    check (! videowire::executeVisualLayerPlan({ particles }, 7, particleLayer, error, videohelper::geometry::PlanUse::preview),
           "particle image source rejects non-native backend classification");

    auto particleComposite = direct;
    // Deliberately scramble document and operation order. The compiled schedule
    // must derive execution and Blend routing from stable edges and port IDs.
    particleComposite.nodeKinds = { "video.out", "video.blend", "video.mask.shape",
        "visual.draw.shape", "video.layer.source", "visual.particles",
        "video.effects", "visual.shape.rectangle", "video.transform" };
    particleComposite.nodeIds = { 12, 24, 23, 32, 26, 51, 22, 31, 21 };
    particleComposite.operations = {
        { 24, "video.blend", "native-gpu", "" },
        { 51, "visual.particles", "native-gpu",
          "<NodeParams seed=\"42\" count=\"1024\" lifetime=\"3\" size=\"6\" speed=\"2\" red=\"0.1\" green=\"0.2\" blue=\"0.3\" alpha=\"0.8\"/>" },
        { 12, "video.out", "native-gpu", "" },
        { 31, "visual.shape.rectangle", "control-eval",
          "<NodeParams centerX=\"0.2\" centerY=\"0.3\" width=\"0.4\" height=\"0.5\"/>" },
        { 22, "video.effects", "native-gpu", "" },
        { 26, "video.layer.source", "source-decode", "" },
        { 21, "video.transform", "native-gpu", "" },
        { 32, "visual.draw.shape", "native-gpu",
          "<NodeParams red=\"0.6\" green=\"0.7\" blue=\"0.8\" alpha=\"0.9\"/>" },
        { 23, "video.mask.shape", "native-gpu", "" }
    };
    particleComposite.edges = {
        { 24, 1, 12, 0 }, { 32, 1, 24, 2 }, { 51, 1, 21, 0 },
        { 26, 0, 24, 3 }, { 23, 1, 24, 0 }, { 21, 1, 22, 0 },
        { 31, 0, 32, 0 }, { 22, 1, 23, 0 }
    };
    particleComposite.ports = {
        { 51, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 21, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 21, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 22, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 22, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 23, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 23, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 31, 0, 1, "out", "control", "shape", "unspecified", "unspecified" },
        { 32, 0, 1, "in", "control", "shape", "unspecified", "unspecified" },
        { 32, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 26, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 24, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 24, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 24, 2, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 24, 3, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 12, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    check (videowire::validateCompiledVisualLayerPlans({}, { particleComposite }, false, error),
           "particle composite satisfies the complete typed snapshot contract");
    check (videowire::compileVisualLayerExecution(particleComposite, execution, error)
           && execution.particles && execution.particleNodeId == 51
           && execution.transform && execution.effects && execution.mask
           && execution.drawShape && execution.drawShapeNodeId == 32
           && execution.compositeInputCount == 3
           && execution.compositeSourceNodeIds[0] == 51
           && execution.compositeDestinationPorts[0] == 0
           && execution.compositeSourceNodeIds[1] == 31
           && execution.compositeDestinationPorts[1] == 2
           && execution.compositeSourceNodeIds[2] == 26
           && execution.compositeDestinationPorts[2] == 3
           && execution.dagSchedule.operations.size() == particleComposite.operations.size(),
           "particle root lowers with mask, shape and layer inputs in deterministic Blend-port order");
    FakeLayer viewportParticleComposite, exportParticleComposite;
    check (videowire::executeVisualLayerPlan({ particleComposite }, 7,
                                             viewportParticleComposite, error, videohelper::geometry::PlanUse::preview)
           && videowire::executeVisualLayerPlan({ particleComposite }, 7,
                                                exportParticleComposite, error, videohelper::geometry::PlanUse::preview)
           && viewportParticleComposite.particleSource
           && viewportParticleComposite.drawShape
           && viewportParticleComposite.maskType == 2
           && viewportParticleComposite.particleNodeId == exportParticleComposite.particleNodeId
           && viewportParticleComposite.drawShapeNodeId == exportParticleComposite.drawShapeNodeId
           && viewportParticleComposite.particleTriggerConnected
                == exportParticleComposite.particleTriggerConnected,
           "viewport/export shared execution preserves the generalized particle composite topology");
    auto wrongParticlePort = particleComposite;
    const auto particleEdge = std::find_if(wrongParticlePort.edges.begin(), wrongParticlePort.edges.end(),
        [](const auto& edge) { return edge.fromNodeId == 51; });
    particleEdge->fromPort = 0;
    check (! videowire::compileVisualLayerExecution(wrongParticlePort, execution, error)
           && error == "bounded visual DAG has an incompatible typed edge",
           "particle composite fails closed when its exact image output port is not routed");

    check (videowire::executeVisualLayerPlan ({ direct }, 7, layer, error, videohelper::geometry::PlanUse::preview),
           "ordered direct chain executes");
    check (layer.scale == 1.0f && layer.translateX == 0.0f
           && layer.effects == nullptr && layer.effectCount == 0
           && layer.lutTexture == 0 && layer.maskType == 0
           && layer.opacity == 0.4f && layer.blendMode == 3,
           "absent production operations are neutralized through LayerDesc");

    auto matte = mattePlan();
    FakeLayer matteLayer;
    matteLayer.matteTexture = 91;
    matteLayer.matteWidth = 640;
    matteLayer.matteHeight = 360;
    check (videowire::executeVisualLayerPlan({ matte }, 7, matteLayer, error, videohelper::geometry::PlanUse::preview)
           && matteLayer.matteApply && matteLayer.matteInvert
           && matteLayer.matteBlack == 0.0f && matteLayer.matteWhite == 1.0f
           && matteLayer.matteErodeDilate == 4.0f && matteLayer.matteFeather == 4.0f
           && matteLayer.matteChoke == -1.0f,
           "valid matte topology propagates bounded refinement parameters");
    auto combined = combinedMattePlan();
    matteLayer.matteTextureB = 92; matteLayer.matteWidthB = 640; matteLayer.matteHeightB = 360;
    check(videowire::executeVisualLayerPlan({combined}, 7, matteLayer, error, videohelper::geometry::PlanUse::preview)
           && matteLayer.matteCombineMode == 3,
           "two immutable matte receipts lower xor through retained GPU resources");
    auto reorderedCombined = combined;
    std::swap(reorderedCombined.nodeKinds[1], reorderedCombined.nodeKinds[2]);
    std::swap(reorderedCombined.nodeIds[1], reorderedCombined.nodeIds[2]);
    std::swap(reorderedCombined.operations[1], reorderedCombined.operations[2]);
    videowire::RenderSegment primaryMatte;
    primaryMatte.matteAssetId = "matte-1";
    videowire::RenderSegment secondaryMatte;
    check(videowire::visualPlanSecondaryMatte({reorderedCombined}, 7, primaryMatte, secondaryMatte)
          && secondaryMatte.matteAssetId == "matte-2",
          "secondary matte identity follows the exact Combine port across legal source order");
    auto reusedMatte = reusedMattePlan();
    FakeLayer reusedMatteLayer = matteLayer;
    reusedMatteLayer.matteTexture = reusedMatteLayer.matteTextureB = 91;
    check(videowire::validateCompiledVisualLayerPlans({}, {reusedMatte}, false, error)
          && videowire::visualPlanReusesPrimaryMatte({reusedMatte}, 7)
          && videowire::executeVisualLayerPlan({reusedMatte}, 7, reusedMatteLayer, error, videohelper::geometry::PlanUse::preview)
          && reusedMatteLayer.matteCombineMode == 3,
          "one exact typed matte output fans out to both ports and reuses one retained texture");
    videowire::VisualPlanResourceUsage reusedMatteUsage;
    check(videowire::accountVisualPlanResources(reusedMatte, 64, 64, reusedMatteUsage, error)
          && reusedMatteUsage.frameOutputs == 5
          && reusedMatteUsage.peakLiveFrames == 3
          && reusedMatteUsage.frameSlots == 4,
          "fan-out liveness counts one shared output through its last consumer and reuses exact slots");
    auto badReusedMatte = reusedMatte;
    badReusedMatte.edges[2].toPort = 0;
    check(!videowire::compileVisualLayerExecution(badReusedMatte, execution, error)
          && error == "typed matte graph has unsupported production topology",
          "matte fan-out still requires exact distinct typed destination ports");
    const std::array<std::array<float, 4>, 4> oracles {{
        {{0.7f, 0.4f, 0.7f, 0.0f}}, {{0.7f, 0.4f, 0.4f, 1.0f}},
        {{0.7f, 0.4f, 0.3f, 2.0f}}, {{0.7f, 0.4f, 0.3f, 3.0f}} }};
    for (const auto& oracle : oracles)
    {
        const int mode = (int)oracle[3];
        const float actual = mode == 0 ? std::max(oracle[0], oracle[1])
                           : mode == 1 ? std::min(oracle[0], oracle[1])
                           : mode == 2 ? std::max(oracle[0] - oracle[1], 0.0f)
                                       : std::abs(oracle[0] - oracle[1]);
        check(std::abs(actual - oracle[2]) < 1.0e-6f, "tiny independent matte-combine pixel oracle");
    }
    FakeLayer missingSecond = matteLayer; missingSecond.matteTextureB = 0;
    check(!videowire::executeVisualLayerPlan({combined}, 7, missingSecond, error, videohelper::geometry::PlanUse::preview)
          && error == "typed matte combine GPU texture is unavailable",
          "combine rejects a missing retained second GPU texture");
    auto malformedMatte = matte;
    malformedMatte.edges[2].toPort = 0;
    check (! videowire::executeVisualLayerPlan({ malformedMatte }, 7, matteLayer, error, videohelper::geometry::PlanUse::preview)
           && error == "typed matte graph has unsupported production topology",
           "malformed matte topology fails closed");
    auto missingMatte = matte;
    missingMatte.operations[1].payloadXml =
        "<MatteAssetBinding state=\"available\" backend=\"rgba-cpu-decode-native-gpu-upload\"/>";
    check (! videowire::executeVisualLayerPlan({ missingMatte }, 7, matteLayer, error, videohelper::geometry::PlanUse::preview)
           && error == "typed matte asset binding is missing, stale, or unsupported",
           "missing matte binding fails closed");
    auto staleMatte = matte;
    staleMatte.operations[1].payloadXml =
        "<MatteAssetBinding matteAssetId=\"matte-1\" state=\"stale\" backend=\"rgba-cpu-decode-native-gpu-upload\"/>";
    check (! videowire::executeVisualLayerPlan({ staleMatte }, 7, matteLayer, error, videohelper::geometry::PlanUse::preview)
           && error == "typed matte asset binding is missing, stale, or unsupported",
           "stale matte binding fails closed");
    FakeLayer texturelessMatte;
    check (! videowire::executeVisualLayerPlan({ matte }, 7, texturelessMatte, error, videohelper::geometry::PlanUse::preview)
           && error == "typed matte GPU texture is unavailable",
           "unavailable matte texture fails closed");
    auto depth = depthPlan();
    videowire::ExecutableDepthPayload viewportPayload, exportPayload;
    std::string viewportDepthError, exportDepthError;
    check (videowire::visualPlanDepthBinding({ depth }, 7, viewportPayload, viewportDepthError)
           && videowire::visualPlanDepthBinding({ depth }, 7, exportPayload, exportDepthError)
           && viewportPayload.cacheKey == exportPayload.cacheKey
           && viewportPayload.contentReceipt == exportPayload.contentReceipt
           && viewportPayload.width == exportPayload.width
           && viewportPayload.height == exportPayload.height,
           "viewport/export shared depth preparation reads identical immutable descriptors");
    FakeLayer depthLayer;
    depthLayer.depthTexture = 9; depthLayer.depthWidth = 2; depthLayer.depthHeight = 2;
    check (videowire::executeVisualLayerPlan({ depth }, 7, depthLayer, error, videohelper::geometry::PlanUse::preview)
           && depthLayer.depthFog && depthLayer.fogNear == 0.0f && depthLayer.fogFar == 1.0f
           && depthLayer.fogDensity == 32.0f && depthLayer.fogAlpha == 1.0f,
           "depth fog exact topology executes with bounded parameters");
    const std::array<std::pair<const char*, const char*>, 3> depthKinds {{
        { "visual.depth.blur", "<NodeParams radius=\"99\" focus=\"0.25\" falloff=\"0\"/>" },
        { "visual.depth.displace", "<NodeParams amountX=\"1\" amountY=\"-1\" center=\"2\"/>" },
        { "visual.depth.relight", "<NodeParams intensity=\"3\" ambient=\"-1\" red=\"3\" green=\"0.5\" blue=\"1\"/>" } }};
    for (size_t i = 0; i < depthKinds.size(); ++i) {
        auto primitive = depthPlan();
        primitive.nodeKinds[2] = depthKinds[i].first;
        primitive.operations[2].kind = depthKinds[i].first;
        primitive.operations[2].payloadXml = depthKinds[i].second;
        FakeLayer primitiveLayer;
        primitiveLayer.depthTexture = 10 + static_cast<unsigned>(i);
        primitiveLayer.depthWidth = primitiveLayer.depthHeight = 2;
        check(videowire::executeVisualLayerPlan({ primitive }, 7, primitiveLayer, error, videohelper::geometry::PlanUse::preview)
              && primitiveLayer.depthEffect == static_cast<int>(i) + 2 && !primitiveLayer.depthFog,
              "depth primitive exact topology lowers to its stable native wire mode");
    }
    videowire::VisualInspectionTarget depthTarget { 7, 0, 61, 0 };
    check (! videowire::validateVisualInspectionTarget({ depth }, depthTarget, error)
           && error == "inspection target is not an image output port",
           "metadata-only depth node cannot be inspected as a materialized output");

    for (const char* extra : { "path", "file", "uri", "decoder", "upload", "futureField" })
    {
        auto injected = depth;
        injected.operations[1].payloadXml.insert(injected.operations[1].payloadXml.size() - 2,
                                                  std::string(" ") + extra + "=\"x\"");
        check (! videowire::executeVisualLayerPlan({ injected }, 7, depthLayer, error, videohelper::geometry::PlanUse::preview),
               "depth executor rejects every extra payload attribute");
    }
    for (const char* malformed : {
             "<Other depthAssetId=\"depth-1\" depthAssetVersion=\"v1\" state=\"available\" contract=\"parked-metadata\" pendingExecutionSeam=\"depth-consuming-operation\"/>",
             "<DepthAssetBinding depthAssetId=\"depth-1\" depthAssetId=\"duplicate\" depthAssetVersion=\"v1\" state=\"available\" contract=\"parked-metadata\" pendingExecutionSeam=\"depth-consuming-operation\"/>",
             "<DepthAssetBinding depthAssetId=\"depth-1\" depthAssetVersion=\"v1\" state=\"available\" contract=\"parked-metadata\" pendingExecutionSeam=\"depth-consuming-operation\"><child/></DepthAssetBinding>",
             "<DepthAssetBinding depthAssetId=\"depth-1\" depthAssetVersion=\"v1\" state=\"available\" contract=\"parked-metadata\" pendingExecutionSeam=\"depth-consuming-operation\">text</DepthAssetBinding>" })
    {
        auto rejectedPayload = depth;
        rejectedPayload.operations[1].payloadXml = malformed;
        check (! videowire::executeVisualLayerPlan({ rejectedPayload }, 7, depthLayer, error, videohelper::geometry::PlanUse::preview),
               "depth executor rejects malformed exact-schema payload");
    }
    depth.edges[1].fromPort = 0;
    check (! videowire::executeVisualLayerPlan({ depth }, 7, depthLayer, error, videohelper::geometry::PlanUse::preview)
           && error == "typed depth asset binding is missing, stale, or unsupported",
           "depth topology rejects any connection into composition");
    videowire::VisualInspectionTarget inspection { 7, 0, 11, 0 };
    check (videowire::validateVisualInspectionTarget({ direct }, inspection, error),
           "exact executable image output is inspectable");
    inspection.structuralRevision = 1;
    check (! videowire::validateVisualInspectionTarget({ direct }, inspection, error)
           && error == "inspection target revision is stale",
           "stale inspection revisions fail closed");
    inspection = { 7, 0, 12, 0 };
    check (! videowire::validateVisualInspectionTarget({ direct }, inspection, error)
           && error == "inspection target is not an image output port",
           "input ports cannot be selected as intermediate outputs");

    auto sourceThenEffects = direct;
    sourceThenEffects.nodeKinds = { "video.source", "video.transform", "video.effects", "video.out" };
    sourceThenEffects.nodeIds = { 11, 14, 13, 12 };
    sourceThenEffects.operations = {
        { 11, "video.source", "source-decode", "" },
        { 14, "video.transform", "native-gpu", "" },
        { 13, "video.effects", "native-gpu", "" },
        { 12, "video.out", "native-gpu", "" }
    };
    sourceThenEffects.edges = { { 11, 0, 14, 0 }, { 14, 1, 13, 0 }, { 13, 1, 12, 0 } };
    sourceThenEffects.ports.push_back(
        { 14, 1, 1, "out", "frame", "image", "rgba8", "sRGB" });
    inspection = { 7, 0, 11, 0 };
    check (videowire::validateVisualInspectionTarget({ sourceThenEffects }, inspection, error),
           "decoded source output is the admitted non-terminal inspection slice");
    FakeLayer sourceLayer;
    sourceLayer.texture = 77;
    sourceLayer.texWidth = 1920;
    sourceLayer.texHeight = 1080;
    videowire::VisualInspectionResource sourceResource;
    check (videowire::executeVisualLayerPlan({ sourceThenEffects }, 7, sourceLayer, error,videohelper::geometry::PlanUse::preview,
                                             &inspection, &sourceResource)
           && sourceResource.handle == 77 && sourceResource.width == 1920
           && sourceResource.height == 1080 && sourceResource.nodeId == 11,
           "executor retains the genuine decoded GPU texture without readback");
    inspection = { 7, 0, 14, 1 };
    check (! videowire::validateVisualInspectionTarget({ sourceThenEffects }, inspection, error)
           && error == "inspection target has no retainable GPU output resource",
           "transform followed by another operation is not misreported at the renderer boundary");
    auto sourceThenTransform = sourceThenEffects;
    sourceThenTransform.nodeKinds.erase(sourceThenTransform.nodeKinds.begin() + 2);
    sourceThenTransform.nodeIds.erase(sourceThenTransform.nodeIds.begin() + 2);
    sourceThenTransform.operations.erase(sourceThenTransform.operations.begin() + 2);
    sourceThenTransform.edges = { { 11, 0, 14, 0 }, { 14, 1, 12, 0 } };
    check (videowire::validateVisualInspectionTarget({ sourceThenTransform }, inspection, error)
           && videowire::classifyVisualInspectionTarget(sourceThenTransform, inspection)
                == videowire::VisualInspectionSlice::transformedLayer,
           "exact terminal transform output uses the genuine renderer resource seam");

    auto reversed = direct;
    std::reverse (reversed.operations.begin(), reversed.operations.end());
    check (! videowire::executeVisualLayerPlan ({ reversed }, 7, layer, error, videohelper::geometry::PlanUse::preview),
           "ordered non-DAG operation storage must agree with exact edges");

    auto branch = direct;
    branch.nodeKinds.insert (branch.nodeKinds.begin() + 1, "video.effects");
    branch.nodeIds.insert (branch.nodeIds.begin() + 1, 13);
    branch.operations.insert (branch.operations.begin() + 1,
                              { 13, "video.effects", "native-gpu", "" });
    branch.edges = { { 11, 0, 13, 0 }, { 11, 0, 12, 0 } };
    check (! videowire::compileVisualLayerExecution (branch, execution, error),
           "branching image execution fails closed instead of flattening");

    auto threeSource = direct;
    threeSource.nodeKinds = { "video.out", "video.layer.source", "video.blend",
                              "video.source", "video.text" };
    threeSource.nodeIds = { 12, 26, 24, 11, 25 };
    // Deliberately store nodes, operations and edges out of execution order.
    // Stable identities and exact destination ports, not vector position, own
    // lowering.
    threeSource.operations = {
        { 12, "video.out", "native-gpu", "" },
        { 26, "video.layer.source", "source-decode", "" },
        { 24, "video.blend", "native-gpu", "" },
        { 25, "video.text", "native-gpu", "" },
        { 11, "video.source", "source-decode", "" }
    };
    threeSource.ports = {
        { 11, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 25, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 26, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 24, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 24, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 24, 2, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 24, 3, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 12, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    threeSource.edges = {
        { 24, 1, 12, 0 }, { 26, 0, 24, 3 },
        { 11, 0, 24, 0 }, { 25, 0, 24, 2 }
    };
    check (videowire::validateCompiledVisualLayerPlans({}, { threeSource }, false, error),
           "three-source composite satisfies the complete typed snapshot contract");
    check (videowire::compileVisualLayerExecution(threeSource, execution, error)
           && execution.compositeInputCount == 3
           && execution.compositeSourceNodeIds[0] == 11
           && execution.compositeDestinationPorts[0] == 0
           && execution.compositeSourceNodeIds[1] == 25
           && execution.compositeDestinationPorts[1] == 2
           && execution.compositeSourceNodeIds[2] == 26
           && execution.compositeDestinationPorts[2] == 3
           && execution.dagSchedule.operations.size() == 5
           && threeSource.operations[execution.dagSchedule.operations[0].operationIndex].nodeId == 11
           && threeSource.operations[execution.dagSchedule.operations[1].operationIndex].nodeId == 25
           && threeSource.operations[execution.dagSchedule.operations[2].operationIndex].nodeId == 26
           && execution.dagSchedule.operations[3].inputs.size() == 3
           && execution.dagSchedule.operations[3].inputs[0].toPort == 0
           && execution.dagSchedule.operations[3].inputs[0].fromNodeId == 11
           && execution.dagSchedule.operations[3].inputs[1].toPort == 2
           && execution.dagSchedule.operations[3].inputs[1].fromNodeId == 25
           && execution.dagSchedule.operations[3].inputs[2].toPort == 3
           && execution.dagSchedule.operations[3].inputs[2].fromNodeId == 26,
           "three-source composite lowers exact predecessors in deterministic Blend-port order");

    const auto withWholeGraphImageDescriptor = [&] (const char* format,
                                                     const char* colorSpace)
    {
        auto candidate = threeSource;
        for (auto& port : candidate.ports)
            if (port.carrier == "frame" && port.dataType == "image")
            {
                port.pixelFormat = format;
                port.colorSpace = colorSpace;
            }
        return candidate;
    };
    for (const auto& unsupported : {
             std::pair<const char*, const char*> { "rgba16f", "sRGB" },
             std::pair<const char*, const char*> { "rgba8", "linearSRGB" },
             std::pair<const char*, const char*> { "futureFormat", "sRGB" },
             std::pair<const char*, const char*> { "rgba8", "displayP3" },
             std::pair<const char*, const char*> { "rgba8", "futureColorSpace" } })
    {
        const auto candidate = withWholeGraphImageDescriptor(unsupported.first,
                                                              unsupported.second);
        error.clear();
        check (! videowire::validateCompiledVisualLayerPlans({}, { candidate }, false, error)
               && error == "bounded visual DAG requires RGBA8 sRGB Frame<Image> descriptors",
               "snapshot admission rejects a consistently unsupported compositor image descriptor");
        error.clear();
        check (! videowire::compileVisualLayerExecution(candidate, execution, error)
               && error == "bounded visual DAG requires RGBA8 sRGB Frame<Image> descriptors",
               "execution lowering rejects a consistently unsupported compositor image descriptor");
        auto rejectedState = std::make_unique<videowire::VisualPlanExecutionState>();
        error.clear();
        check (! rejectedState->admitPlans({ candidate }, &error)
               && rejectedState->compiled(candidate.clipId,
                                           candidate.structuralRevision) == nullptr
               && error == "bounded visual DAG requires RGBA8 sRGB Frame<Image> descriptors",
               "admission installation rejects a consistently unsupported compositor image descriptor");
    }
    struct ImageDescriptorBoundary
    {
        size_t portIndex;
        const char* name;
    };
    const std::array<ImageDescriptorBoundary, 8> imageDescriptorBoundaries {{
        { 0, "primary output" },
        { 1, "text secondary output" },
        { 2, "layer secondary output" },
        { 3, "Blend primary input" },
        { 4, "Blend output" },
        { 5, "Blend text input" },
        { 6, "Blend layer input" },
        { 7, "Output input" }
    }};
    for (size_t boundaryIndex = 0; boundaryIndex < imageDescriptorBoundaries.size();
         ++boundaryIndex)
    {
        auto candidate = threeSource;
        auto& port = candidate.ports[imageDescriptorBoundaries[boundaryIndex].portIndex];
        const auto boundaryEdge = std::find_if(candidate.edges.begin(), candidate.edges.end(),
            [&](const auto& edge)
            {
                return (edge.fromNodeId == port.nodeId && edge.fromPort == port.port)
                    || (edge.toNodeId == port.nodeId && edge.toPort == port.port);
            });
        for (auto& candidatePort : candidate.ports)
            if ((candidatePort.nodeId == boundaryEdge->fromNodeId
                 && candidatePort.port == boundaryEdge->fromPort)
                || (candidatePort.nodeId == boundaryEdge->toNodeId
                    && candidatePort.port == boundaryEdge->toPort))
                candidatePort.dataType = "futureImage";
        const std::string boundary = imageDescriptorBoundaries[boundaryIndex].name;
        error.clear();
        check (! videowire::validateCompiledVisualLayerPlans({}, { candidate }, false, error)
               && error == "bounded visual DAG requires RGBA8 sRGB Frame<Image> descriptors",
               ("snapshot validation rejects futureImage at the native " + boundary).c_str());
        error.clear();
        check (! videowire::compileVisualLayerExecution(candidate, execution, error)
               && error == "bounded visual DAG requires RGBA8 sRGB Frame<Image> descriptors",
               ("direct lowering rejects futureImage at the native " + boundary).c_str());
        auto rejectedState = std::make_unique<videowire::VisualPlanExecutionState>();
        error.clear();
        check (! rejectedState->admitPlans({ candidate }, &error)
               && rejectedState->compiled(candidate.clipId,
                                           candidate.structuralRevision) == nullptr
               && error == "bounded visual DAG requires RGBA8 sRGB Frame<Image> descriptors",
               ("admission installation rejects futureImage at the native " + boundary).c_str());
    }
    for (int malformedField = 0; malformedField < 3; ++malformedField)
    {
        auto candidate = threeSource;
        auto& port = candidate.ports.front();
        if (malformedField == 0) port.channels = 2;
        if (malformedField == 1) port.carrier = "futureFrame";
        if (malformedField == 2) port.dataType = "futureImage";
        error.clear();
        check (! videowire::validateCompiledVisualLayerPlans({}, { candidate }, false, error)
               && error == "bounded visual DAG requires RGBA8 sRGB Frame<Image> descriptors",
               "snapshot validation requires the exact native image descriptor tuple");
        error.clear();
        check (! videowire::compileVisualLayerExecution(candidate, execution, error)
               && error == "bounded visual DAG requires RGBA8 sRGB Frame<Image> descriptors",
               "direct lowering requires the exact native image descriptor tuple");
        auto rejectedState = std::make_unique<videowire::VisualPlanExecutionState>();
        error.clear();
        check (! rejectedState->admitPlans({ candidate }, &error)
               && rejectedState->compiled(candidate.clipId,
                                           candidate.structuralRevision) == nullptr
               && error == "bounded visual DAG requires RGBA8 sRGB Frame<Image> descriptors",
               "admission installation requires the exact native image descriptor tuple");
    }
    FakeLayer threeSourceLayer;
    check (videowire::executeVisualLayerPlan({ threeSource }, 7, threeSourceLayer, error, videohelper::geometry::PlanUse::preview)
           && threeSourceLayer.scale == 1.0f && threeSourceLayer.effects == nullptr
           && threeSourceLayer.maskType == 0,
           "three-source composite executes through the existing neutral primary descriptor");
    auto permutedThreeSource = threeSource;
    std::rotate(permutedThreeSource.nodeKinds.begin(), permutedThreeSource.nodeKinds.begin() + 2,
                permutedThreeSource.nodeKinds.end());
    std::rotate(permutedThreeSource.nodeIds.begin(), permutedThreeSource.nodeIds.begin() + 2,
                permutedThreeSource.nodeIds.end());
    std::reverse(permutedThreeSource.operations.begin(), permutedThreeSource.operations.end());
    std::reverse(permutedThreeSource.edges.begin(), permutedThreeSource.edges.end());
    videowire::CompiledVisualDagSchedule permutedSchedule;
    check (videowire::compileBoundedVisualDagSchedule(permutedThreeSource, permutedSchedule, error)
           && permutedSchedule.operations.size() == execution.dagSchedule.operations.size()
           && permutedThreeSource.operations[permutedSchedule.operations[0].operationIndex].nodeId == 11
           && permutedThreeSource.operations[permutedSchedule.operations[1].operationIndex].nodeId == 25
           && permutedThreeSource.operations[permutedSchedule.operations[2].operationIndex].nodeId == 26,
           "bounded compositor scheduling is stable across document permutations");

    videowire::VisualPlanResourceUsage threeSourceUsage;
    check (videowire::accountVisualPlanResources(threeSource, 64, 64, threeSourceUsage, error),
           "three-source composite resource usage is independently measured");
    auto accountedThreeSource = threeSource;
    accountedThreeSource.descriptorCount = threeSourceUsage.descriptors;
    accountedThreeSource.operationCount = threeSourceUsage.operations;
    accountedThreeSource.sceneRecordCount = threeSourceUsage.sceneRecords;
    accountedThreeSource.frameOutputCount = threeSourceUsage.frameOutputs;
    accountedThreeSource.peakLiveFrameCount = threeSourceUsage.peakLiveFrames;
    accountedThreeSource.allocatedFrameSlotCount = threeSourceUsage.frameSlots;
    auto threeSourceState = std::make_unique<videowire::VisualPlanExecutionState>();
    auto threeSourceLimits = videowire::VisualBackendResourceLimits::forCanvas(64, 64);
    check (threeSourceState->admitPlans({ accountedThreeSource }, &error, 64, 64, &threeSourceLimits)
           && threeSourceState->compiled(7, accountedThreeSource.structuralRevision) != nullptr,
           "three-source lowering preserves producer resource fields and admission");
    auto constrainedThreeSourceLimits = threeSourceLimits;
    constrainedThreeSourceLimits.descriptors = 4;
    auto rejectedThreeSourceState = std::make_unique<videowire::VisualPlanExecutionState>();
    check (! rejectedThreeSourceState->admitPlans({ accountedThreeSource }, &error, 64, 64,
                                                   &constrainedThreeSourceLimits)
           && error == "visual graph descriptor capacity exceeded: 5 > 4",
           "three-source lowering cannot bypass the descriptor budget");

    const auto mediumA = mediumBudgetPlan(101, 1000);
    const auto mediumB = mediumBudgetPlan(102, 2000);
    const std::vector<videowire::CompiledVisualLayerPlan> mediumPlans { mediumB, mediumA };
    auto mediumLimits = videowire::VisualBackendResourceLimits::forCanvas(64, 64);
    mediumLimits.descriptors = 12;
    mediumLimits.operations = 12;
    mediumLimits.sceneRecords = 0;
    mediumLimits.liveFrames = 4;
    mediumLimits.frameSlots = 4;
    mediumLimits.allocatedFrameBytes = 64ull * 64ull * 4ull * 4ull;
    mediumLimits.backendProfile = "opengl-4.6";
    mediumLimits.maximumImageDimensionSource = "OpenGL.GL_MAX_TEXTURE_SIZE";
    mediumLimits.allocatedFrameBytesSource = "test-exact-backend-budget";
    auto previewBudgetState = std::make_unique<videowire::VisualPlanExecutionState>();
    auto exportBudgetState = std::make_unique<videowire::VisualPlanExecutionState>();
    FakeLayer previewMediumA, previewMediumB, exportMediumA, exportMediumB;
    error.clear();
    check (mediumA.descriptorCount == 6 && mediumA.operationCount == 6
           && mediumA.frameOutputCount == 5 && mediumA.peakLiveFrameCount == 2
           && mediumA.allocatedFrameSlotCount == 2
           && previewBudgetState->admitPlans(mediumPlans, &error, 64, 64, &mediumLimits)
           && exportBudgetState->admitPlans(mediumPlans, &error, 64, 64, &mediumLimits)
           && videowire::executeVisualLayerPlan(mediumPlans, 101, previewMediumA, error,videohelper::geometry::PlanUse::preview,
                                                nullptr, nullptr, previewBudgetState.get(), 1.0)
           && videowire::executeVisualLayerPlan(mediumPlans, 102, previewMediumB, error,videohelper::geometry::PlanUse::preview,
                                                nullptr, nullptr, previewBudgetState.get(), 1.0)
           && videowire::executeVisualLayerPlan(mediumPlans, 101, exportMediumA, error,videohelper::geometry::PlanUse::preview,
                                                nullptr, nullptr, exportBudgetState.get(), 1.0)
           && videowire::executeVisualLayerPlan(mediumPlans, 102, exportMediumB, error,videohelper::geometry::PlanUse::preview,
                                                nullptr, nullptr, exportBudgetState.get(), 1.0),
           "preview and export execute two medium graphs at the exact aggregate budget");
    const videowire::VisualPlanResourceReceipt expectedMediumReceipt {
        2, 12, 12, 0, 10, 4, 4, 64ull * 64ull * 4ull * 4ull
    };
    check (previewBudgetState->resourceReceipt() == expectedMediumReceipt
           && exportBudgetState->resourceReceipt() == expectedMediumReceipt,
           "preview and export publish the same exact liveness, reuse, byte, and work receipt");
    auto sceneA = importedAnimationPlan();
    auto sceneB = sceneA;
    sceneB.clipId = 8;
    sceneB.structuralRevision = 10;
    videowire::VisualPlanResourceUsage sceneUsage;
    check(videowire::accountVisualPlanResources(sceneA, 64, 64, sceneUsage, error)
          && sceneUsage.sceneRecords > 0,
          "scene plan resource usage includes bounded scene records");
    for (auto* scenePlan : { &sceneA, &sceneB })
    {
        scenePlan->descriptorCount = sceneUsage.descriptors;
        scenePlan->operationCount = sceneUsage.operations;
        scenePlan->sceneRecordCount = sceneUsage.sceneRecords;
        scenePlan->frameOutputCount = sceneUsage.frameOutputs;
        scenePlan->peakLiveFrameCount = sceneUsage.peakLiveFrames;
        scenePlan->allocatedFrameSlotCount = sceneUsage.frameSlots;
    }
    auto sceneLimits = videowire::VisualBackendResourceLimits::forCanvas(64, 64);
    sceneLimits.descriptors = sceneUsage.descriptors * 2;
    sceneLimits.operations = sceneUsage.operations * 2;
    sceneLimits.sceneRecords = sceneUsage.sceneRecords * 2;
    auto sceneBudgetState = std::make_unique<videowire::VisualPlanExecutionState>();
    error.clear();
    check(sceneBudgetState->admitPlans({ sceneA, sceneB }, &error, 64, 64, &sceneLimits)
          && sceneBudgetState->resourceReceipt().planCount == 2
          && sceneBudgetState->resourceReceipt().descriptors == sceneUsage.descriptors * 2
          && sceneBudgetState->resourceReceipt().operations == sceneUsage.operations * 2
          && sceneBudgetState->resourceReceipt().sceneRecords == sceneUsage.sceneRecords * 2,
          "descriptor, operation, and scene-record receipts report aggregate owner work");
    const auto expectedBudgetReceipt = previewBudgetState->budgetReceipt();
    const auto previewBudgetTelemetry = previewBudgetState->telemetry().snapshot();
    const auto exportBudgetTelemetry = exportBudgetState->telemetry().snapshot();
    const auto previewBudgetJson = videowire::visualTelemetryJson(previewBudgetTelemetry);
    check (expectedBudgetReceipt == exportBudgetState->budgetReceipt()
           && previewBudgetTelemetry.budgetObserved && exportBudgetTelemetry.budgetObserved
           && previewBudgetTelemetry.budgetReceipt == expectedBudgetReceipt
           && exportBudgetTelemetry.budgetReceipt == expectedBudgetReceipt
           && previewBudgetJson["budget"]["backendProfile"] == "opengl-4.6"
           && previewBudgetJson["budget"]["usage"]["allocatedFrameBytes"] == 65536
           && previewBudgetJson["budget"]["provenance"]["allocatedFrameBytes"]
                == "test-exact-backend-budget",
           "preview and export telemetry carry the same immutable budget and usage receipt");
    auto staleMediumA = mediumA;
    staleMediumA.structuralRevision = mediumA.structuralRevision - 1;
    staleMediumA.descriptorCount = 100;
    staleMediumA.operationCount = 100;
    auto duplicateBudgetState = std::make_unique<videowire::VisualPlanExecutionState>();
    error.clear();
    check (duplicateBudgetState->admitPlans({ mediumB, staleMediumA, mediumA }, &error,
                                            64, 64, &mediumLimits)
           && duplicateBudgetState->resourceReceipt() == expectedMediumReceipt
           && duplicateBudgetState->compiled(101, mediumA.structuralRevision) != nullptr,
           "aggregate accounting normalizes stale clip revisions before admission and receipts");
    const std::vector<videowire::CompiledVisualLayerPlan> duplicateOrder {
        staleMediumA, mediumB, mediumA
    };
    const auto* selectedDuplicate = videowire::findVisualLayerPlan(duplicateOrder, 101);
    check (selectedDuplicate != nullptr
           && selectedDuplicate->structuralRevision == mediumA.structuralRevision,
           "execution selects the same newest clip revision normalized by aggregate admission");
    FakeLayer duplicateExecutionLayer;
    error.clear();
    check(videowire::executeVisualLayerPlan(
              duplicateOrder, 101, duplicateExecutionLayer, error,videohelper::geometry::PlanUse::preview,
              nullptr, nullptr, duplicateBudgetState.get(), 1.0)
          && duplicateBudgetState->compiled(101, mediumA.structuralRevision) != nullptr,
          "stale-first newest-second duplicate execution uses the admitted newest revision end to end");
    const auto stableReceipt = previewBudgetState->resourceReceipt();
    auto insufficientWork = mediumLimits;
    insufficientWork.operations = 11;
    error.clear();
    check (! previewBudgetState->admitPlans(mediumPlans, &error, 64, 64, &insufficientWork)
           && error == "visual graph aggregate operation capacity exceeded: 12 > 11"
           && previewBudgetState->resourceReceipt() == stableReceipt
           && previewBudgetState->compiled(101, mediumA.structuralRevision) != nullptr
           && previewBudgetState->compiled(102, mediumB.structuralRevision) != nullptr,
           "aggregate work rejection preserves both last-good graphs and their receipt");
    check (previewBudgetState->budgetReceipt() == expectedBudgetReceipt
           && previewBudgetState->telemetry().snapshot().budgetReceipt == expectedBudgetReceipt,
           "rejected preview admission preserves last-good budget telemetry");
    uint64_t overflowTotal = std::numeric_limits<uint64_t>::max() - 3;
    check(! videowire::checkedAccumulateVisualPlanBytes(overflowTotal, 2, 2)
          && overflowTotal == std::numeric_limits<uint64_t>::max() - 3,
          "split history and transient byte overflow fails without partial accumulation");
    auto insufficientLiveness = mediumLimits;
    insufficientLiveness.liveFrames = 3;
    error.clear();
    check (! previewBudgetState->admitPlans(mediumPlans, &error, 64, 64,
                                            &insufficientLiveness)
           && error == "visual graph live frame capacity exceeded: 4 > 3"
           && previewBudgetState->resourceReceipt() == stableReceipt,
           "aggregate liveness sums simultaneously admitted graph peaks");
    auto insufficientBytes = mediumLimits;
    --insufficientBytes.allocatedFrameBytes;
    error.clear();
    check (! exportBudgetState->admitPlans(mediumPlans, &error, 64, 64, &insufficientBytes)
           && error == "visual graph allocated frame byte capacity exceeded: 65536 > 65535"
           && exportBudgetState->resourceReceipt() == expectedMediumReceipt,
           "aggregate backend byte rejection preserves the export receipt");
    check (exportBudgetState->budgetReceipt() == expectedBudgetReceipt
           && exportBudgetState->telemetry().snapshot().budgetReceipt == expectedBudgetReceipt,
           "one-over export admission preserves last-good budget telemetry");
    auto swappedThreeSource = threeSource;
    swappedThreeSource.edges[1].toPort = 2;
    swappedThreeSource.edges[3].toPort = 3;
    check (! videowire::compileVisualLayerExecution(swappedThreeSource, execution, error)
           && error == "bounded visual DAG requires exact primary/text/layer Blend bindings",
           "text and layer Blend ports are not interchangeable");
    auto incompleteThreeSource = threeSource;
    incompleteThreeSource.ports.erase(incompleteThreeSource.ports.begin() + 6);
    check (! videowire::compileVisualLayerExecution(incompleteThreeSource, execution, error)
           && error == "bounded visual DAG has an incompatible typed edge",
           "three-source execution rejects a missing typed Blend port");
    auto incompatibleThreeSource = threeSource;
    incompatibleThreeSource.ports[6].colorSpace = "linearSRGB";
    check (! videowire::compileVisualLayerExecution(incompatibleThreeSource, execution, error)
           && error == "bounded visual DAG requires RGBA8 sRGB Frame<Image> descriptors",
           "three-source execution rejects implicit color conversion");
    auto fallbackThreeSource = threeSource;
    fallbackThreeSource.operations[1].backendCapability = "native-gpu";
    check (! videowire::compileVisualLayerExecution(fallbackThreeSource, execution, error)
           && error == "bounded visual DAG has an incompatible native backend capability",
           "three-source execution rejects a layer source without decode capability");
    auto duplicateInputThreeSource = threeSource;
    duplicateInputThreeSource.edges.push_back({ 25, 0, 24, 3 });
    check (! videowire::compileVisualLayerExecution(duplicateInputThreeSource, execution, error)
           && error == "bounded visual DAG has a duplicate edge or multiply-bound input port",
           "three-source execution rejects duplicate Blend input bindings");
    auto cyclicThreeSource = threeSource;
    cyclicThreeSource.ports.push_back({ 12, 1, 1, "out", "frame", "image", "rgba8", "sRGB" });
    cyclicThreeSource.ports.push_back({ 11, 1, 1, "in", "frame", "image", "rgba8", "sRGB" });
    cyclicThreeSource.edges.push_back({ 12, 1, 11, 1 });
    check (! videowire::compileVisualLayerExecution(cyclicThreeSource, execution, error)
           && error == "bounded visual DAG contains a cycle",
           "bounded compositor rejects cycles before execution lowering");

    auto composite = direct;
    composite.nodeKinds = { "video.source", "video.transform", "video.effects",
        "video.mask.shape", "video.text", "video.layer.source", "video.blend", "video.out" };
    composite.nodeIds = { 11, 21, 22, 23, 25, 26, 24, 12 };
    composite.operations.clear();
    for (size_t i = 0; i < composite.nodeIds.size(); ++i)
        composite.operations.push_back({ composite.nodeIds[i], composite.nodeKinds[i],
            i == 0 || i == 5 ? "source-decode" : "native-gpu", "" });
    composite.edges = {
        { 11, 0, 21, 0 }, { 21, 1, 22, 0 }, { 22, 1, 23, 0 },
        { 23, 1, 24, 0 }, { 25, 0, 24, 2 }, { 26, 0, 24, 3 }, { 24, 1, 12, 0 }
    };
    composite.ports = {
        { 11, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 21, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 21, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 22, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 22, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 23, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 23, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 25, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 26, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 24, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 24, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 24, 2, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 24, 3, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 12, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    for (const size_t portIndex : { size_t { 1 }, size_t { 2 } })
    {
        auto candidate = composite;
        const auto& port = candidate.ports[portIndex];
        const auto boundaryEdge = std::find_if(candidate.edges.begin(), candidate.edges.end(),
            [&](const auto& edge)
            {
                return (edge.fromNodeId == port.nodeId && edge.fromPort == port.port)
                    || (edge.toNodeId == port.nodeId && edge.toPort == port.port);
            });
        for (auto& candidatePort : candidate.ports)
            if ((candidatePort.nodeId == boundaryEdge->fromNodeId
                 && candidatePort.port == boundaryEdge->fromPort)
                || (candidatePort.nodeId == boundaryEdge->toNodeId
                    && candidatePort.port == boundaryEdge->toPort))
                candidatePort.dataType = "futureImage";
        error.clear();
        check (! videowire::validateCompiledVisualLayerPlans({}, { candidate }, false, error)
               && error == "bounded visual DAG requires RGBA8 sRGB Frame<Image> descriptors",
               "snapshot validation rejects an inexact intermediate pass image descriptor");
        error.clear();
        check (! videowire::compileVisualLayerExecution(candidate, execution, error)
               && error == "bounded visual DAG requires RGBA8 sRGB Frame<Image> descriptors",
               "direct lowering rejects an inexact intermediate pass image descriptor");
        auto rejectedState = std::make_unique<videowire::VisualPlanExecutionState>();
        error.clear();
        check (! rejectedState->admitPlans({ candidate }, &error)
               && rejectedState->compiled(candidate.clipId,
                                           candidate.structuralRevision) == nullptr
               && error == "bounded visual DAG requires RGBA8 sRGB Frame<Image> descriptors",
               "admission installation rejects an inexact intermediate pass image descriptor");
    }
    FakeLayer compositeLayer;
    check (videowire::executeVisualLayerPlan({ composite }, 7, compositeLayer, error, videohelper::geometry::PlanUse::preview)
           && compositeLayer.scale == 2.0f && compositeLayer.effects != nullptr
           && compositeLayer.maskType == 2,
           "established typed multi-source composite preserves production rendering");

    auto nativeSecondarySources = direct;
    nativeSecondarySources.nodeKinds = { "video.source", "video.layer.source",
                                         "video.text", "video.blend", "video.out" };
    nativeSecondarySources.nodeIds = { 11, 41, 42, 24, 12 };
    nativeSecondarySources.operations = {
        { 11, "video.source", "source-decode", "" },
        { 41, "video.layer.source", "source-decode", "" },
        { 42, "video.text", "native-gpu", "" },
        { 24, "video.blend", "native-gpu", "" },
        { 12, "video.out", "native-gpu", "" }
    };
    // Deliberately reverse source document order at the Blend inputs. Exact
    // destination ports, not document order, own compositor ordering.
    nativeSecondarySources.edges = { { 11, 0, 24, 0 }, { 41, 0, 24, 3 },
                             { 42, 0, 24, 2 }, { 24, 1, 12, 0 } };
    nativeSecondarySources.ports = {
        { 11, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 41, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 42, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 24, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 24, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 24, 2, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 24, 3, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 12, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    check (videowire::validateCompiledVisualLayerPlans({}, { nativeSecondarySources }, false, error),
           "native secondary-source DAG satisfies the complete typed snapshot contract");
    check (videowire::compileVisualLayerExecution(nativeSecondarySources, execution, error)
           && execution.compositeInputCount == 3
           && execution.compositeSourceNodeIds[0] == 11
           && execution.compositeSourceNodeIds[1] == 42
           && execution.compositeSourceNodeIds[2] == 41
           && execution.compositeDestinationPorts[0] == 0
           && execution.compositeDestinationPorts[1] == 2
           && execution.compositeDestinationPorts[2] == 3
           && ! execution.transform && ! execution.effects && ! execution.mask,
           "native text and layer sources execute in deterministic exact port order");
    auto differentSecondaryIdentity = nativeSecondarySources;
    differentSecondaryIdentity.nodeKinds[2] = "video.layer.source";
    differentSecondaryIdentity.operations[2].kind = "video.layer.source";
    differentSecondaryIdentity.operations[2].backendCapability = "source-decode";
    check (! videowire::compileVisualLayerExecution(differentSecondaryIdentity, execution, error)
           && error == "visual composite secondary branch is not an executable native source",
           "native compositor rejects a different source identity at its text-overlay input");

    auto commonEffectComposite = nativeSecondarySources;
    // Keep storage and operation order deliberately unrelated to execution
    // order. Stable node/port bindings own both the primary effect branch and
    // native secondary-source ordering.
    commonEffectComposite.structuralRevision = 8;
    commonEffectComposite.nodeKinds = { "video.out", "video.layer.source", "video.blend",
        "visual.color.saturation", "video.text", "video.source" };
    commonEffectComposite.nodeIds = { 12, 41, 24, 50, 42, 11 };
    commonEffectComposite.operations = {
        { 24, "video.blend", "native-gpu", "" },
        { 50, "visual.color.saturation", "native-gpu",
          "<CommonEffect schemaVersion=\"1\" effect=\"saturation\" amount=\"1.75\"/>" },
        { 41, "video.layer.source", "source-decode", "" },
        { 12, "video.out", "native-gpu", "" },
        { 11, "video.source", "source-decode", "" },
        { 42, "video.text", "native-gpu", "" }
    };
    commonEffectComposite.edges = { { 50, 1, 24, 0 }, { 41, 0, 24, 3 },
        { 24, 1, 12, 0 }, { 11, 0, 50, 0 }, { 42, 0, 24, 2 } };
    commonEffectComposite.ports = {
        { 11, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 50, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 50, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 41, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 42, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 24, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 24, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 24, 2, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 24, 3, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 12, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    check (videowire::validateCompiledVisualLayerPlans(
               {}, { commonEffectComposite }, false, error)
           && videowire::compileVisualLayerExecution(commonEffectComposite, execution, error)
           && execution.commonEffect && ! execution.effects
           && execution.commonEffectType == 2
           && execution.commonEffectParameterCount == 1
           && execution.commonEffectValues[0] == 1.75f
           && execution.compositeInputCount == 3
           && execution.compositeSourceNodeIds[0] == 11
           && execution.compositeSourceNodeIds[1] == 42
           && execution.compositeSourceNodeIds[2] == 41,
           "ordinary multi-source composite admits one exact immutable common-effect stage");
    FakeLayer viewportCommonEffect, exportCommonEffect;
    FakeProductRenderer viewportCommonRenderer, exportCommonRenderer;
    check (videowire::executeVisualLayerPlanForRenderer(
               viewportCommonRenderer, { commonEffectComposite }, 7,
               viewportCommonEffect, error, videohelper::geometry::PlanUse::preview, nullptr)
           && videowire::executeVisualLayerPlanForRenderer(
               exportCommonRenderer, { commonEffectComposite }, 7,
               exportCommonEffect, error, videohelper::geometry::PlanUse::preview, nullptr)
           && viewportCommonRenderer.bridgePreparations == 1
           && exportCommonRenderer.bridgePreparations == 1
           && viewportCommonEffect.effects == &viewportCommonEffect.graphCommonEffect
           && exportCommonEffect.effects == &exportCommonEffect.graphCommonEffect
           && viewportCommonEffect.graphCommonEffect.type
                == exportCommonEffect.graphCommonEffect.type
           && viewportCommonEffect.graphCommonEffect.params[0]
                == exportCommonEffect.graphCommonEffect.params[0],
           "viewport and export share one common-effect composite executor seam");

    auto malformedCompositeEffect = commonEffectComposite;
    malformedCompositeEffect.operations[1].payloadXml =
        "<CommonEffect schemaVersion=\"1\" effect=\"saturation\" amount=\"nan\"/>";
    check (! videowire::compileVisualLayerExecution(
               malformedCompositeEffect, execution, error)
           && error == "visual.color.saturation has a malformed, unsupported, or out-of-bounds immutable payload",
           "ordinary composite rejects malformed common-effect payloads without fallback");
    auto convertedCompositeEffect = commonEffectComposite;
    convertedCompositeEffect.ports[2].colorSpace = "linearSRGB";
    check (! videowire::compileVisualLayerExecution(
               convertedCompositeEffect, execution, error)
           && error == "bounded visual DAG requires RGBA8 sRGB Frame<Image> descriptors",
           "ordinary composite rejects implicit color conversion at typed effect edges");
    auto fallbackCompositeEffect = commonEffectComposite;
    fallbackCompositeEffect.operations[1].backendCapability = "cpu-fallback";
    check (! videowire::compileVisualLayerExecution(
               fallbackCompositeEffect, execution, error)
           && error == "visual composite operation has incompatible native backend capability",
           "ordinary composite rejects CPU effect fallback capability");

    auto unsupportedFanIn = nativeSecondarySources;
    unsupportedFanIn.edges[1].toPort = 2;
    check (! videowire::compileVisualLayerExecution(unsupportedFanIn, execution, error)
           && error == "bounded visual DAG has a duplicate edge or multiply-bound input port",
           "duplicate Blend-port fan-in fails closed with a focused diagnostic");

    auto repeatedTransform = nativeSecondarySources;
    repeatedTransform.nodeKinds = { "video.source", "video.transform", "video.transform",
                                    "video.layer.source", "video.blend", "video.out" };
    repeatedTransform.nodeIds = { 11, 31, 32, 41, 24, 12 };
    repeatedTransform.operations = {
        { 11, "video.source", "source-decode", "" },
        { 31, "video.transform", "native-gpu", "" },
        { 32, "video.transform", "native-gpu", "" },
        { 41, "video.layer.source", "source-decode", "" },
        { 24, "video.blend", "native-gpu", "" },
        { 12, "video.out", "native-gpu", "" }
    };
    repeatedTransform.edges = { { 11, 0, 31, 0 }, { 31, 1, 32, 0 },
                                { 32, 1, 24, 0 }, { 41, 0, 24, 3 },
                                { 24, 1, 12, 0 } };
    repeatedTransform.ports.clear();
    check (! videowire::compileVisualLayerExecution(repeatedTransform, execution, error)
           && error == "visual composite primary branch cannot be represented exactly by LayerDesc",
           "repeated unary passes are rejected instead of collapsing into one descriptor field");

    auto pathMatte = direct;
    pathMatte.nodeKinds = { "video.source", "visual.shape.rectangle", "visual.shape.ellipse",
                            "visual.shape.union", "visual.shape.invert", "visual.matte.path",
                            "video.layer.source", "video.blend", "video.out" };
    pathMatte.nodeIds = { 11, 31, 32, 33, 34, 35, 41, 24, 12 };
    pathMatte.operations = {
        { 11, "video.source", "source-decode", "" },
        { 31, "visual.shape.rectangle", "control-eval",
          "<NodeParams centerX=\"0.25\" centerY=\"0.75\" width=\"0.5\" height=\"0.2\"/>" },
        { 32, "visual.shape.ellipse", "control-eval",
          "<NodeParams centerX=\"0.75\" centerY=\"0.25\" width=\"0.3\" height=\"0.4\"/>" },
        { 33, "visual.shape.union", "control-eval", "" },
        { 34, "visual.shape.invert", "control-eval", "" },
        { 35, "visual.matte.path", "native-gpu", "" },
        { 41, "video.layer.source", "source-decode", "" },
        { 24, "video.blend", "native-gpu", "" },
        { 12, "video.out", "native-gpu", "" }
    };
    pathMatte.edges = { { 11, 0, 35, 0 }, { 31, 0, 33, 0 }, { 32, 0, 33, 1 },
                        { 33, 2, 34, 0 }, { 34, 1, 35, 1 }, { 35, 2, 24, 0 },
                        { 41, 0, 24, 3 },
                        { 24, 1, 12, 0 } };
    pathMatte.ports = {
        { 11, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 31, 0, 1, "out", "control", "shape", "unspecified", "unspecified" },
        { 32, 0, 1, "out", "control", "shape", "unspecified", "unspecified" },
        { 33, 0, 1, "in", "control", "shape", "unspecified", "unspecified" },
        { 33, 1, 1, "in", "control", "shape", "unspecified", "unspecified" },
        { 33, 2, 1, "out", "control", "shape", "unspecified", "unspecified" },
        { 34, 0, 1, "in", "control", "shape", "unspecified", "unspecified" },
        { 34, 1, 1, "out", "control", "shape", "unspecified", "unspecified" },
        { 35, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 35, 1, 1, "in", "control", "shape", "unspecified", "unspecified" },
        { 35, 2, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 41, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 24, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 24, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 24, 3, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 12, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    for (const auto& mode : { std::pair<const char*, int>{ "visual.shape.union", 1 },
                              { "visual.shape.intersection", 2 },
                              { "visual.shape.subtract", 3 } })
    {
        pathMatte.nodeKinds[3] = mode.first;
        pathMatte.operations[3].kind = mode.first;
        const bool pathCompiled = videowire::compileVisualLayerExecution(pathMatte, execution, error);
        check (pathCompiled
               && execution.pathMatte && execution.pathMatteHasSecondary
               && execution.pathMatteInvert && execution.pathMatteOperation == mode.second
               && execution.pathMatteEllipse == false
               && execution.pathMatteSecondaryEllipse,
               "bounded path matte lowers add, intersect, subtract and invert exactly");
    }
    FakeLayer pathLayer;
    const bool pathExecuted = videowire::executeVisualLayerPlan({ pathMatte }, 7, pathLayer, error, videohelper::geometry::PlanUse::preview);
    check (pathExecuted
           && pathLayer.pathMatte && pathLayer.pathMatteOperation == 3
           && pathLayer.pathMatteInvert && pathLayer.maskType == 2,
           "path matte execution preserves the existing clip shape mask");
    auto inventedPathPayload = pathMatte;
    inventedPathPayload.operations[5].payloadXml = "<NodeParams fallback=\"1\"/>";
    const bool inventedValid = videowire::validateCompiledVisualLayerPlans(
        {}, { inventedPathPayload }, false, error);
    check (! inventedValid
           && error == "visual path-matte operation payload must be empty",
           "snapshot admission fails closed on invented path-matte payload");
    auto unboundedPath = pathMatte;
    unboundedPath.operations[1].payloadXml =
        "<NodeParams centerX=\"0.25\" centerY=\"0.75\" width=\"4.01\" height=\"0.2\"/>";
    const bool unboundedCompiled = videowire::compileVisualLayerExecution(
        unboundedPath, execution, error);
    check (! unboundedCompiled
           && error == "visual Path Matte primitive payload is malformed or out of bounds",
           "path-matte execution rejects out-of-bounds primitive payloads");
    auto nestedPath = pathMatte;
    nestedPath.nodeKinds[4] = "visual.shape.union";
    nestedPath.operations[4].kind = "visual.shape.union";
    check (! videowire::compileVisualLayerExecution(nestedPath, execution, error),
           "path-matte execution rejects nested Boolean trees outside the bound");

    auto drawShape = direct;
    drawShape.nodeKinds = { "video.source", "visual.shape.rectangle",
                            "visual.draw.shape", "video.blend", "video.out" };
    drawShape.nodeIds = { 11, 31, 32, 24, 12 };
    drawShape.operations = {
        { 11, "video.source", "source-decode", "" },
        { 31, "visual.shape.rectangle", "control-eval",
          "<NodeParams centerX=\"0.25\" centerY=\"0.75\" width=\"0.5\" height=\"0.2\"/>" },
        { 32, "visual.draw.shape", "native-gpu",
          "<NodeParams red=\"0.1\" green=\"0.2\" blue=\"0.3\" alpha=\"0.4\"/>" },
        { 24, "video.blend", "native-gpu", "" },
        { 12, "video.out", "native-gpu", "" }
    };
    drawShape.edges = { { 11, 0, 24, 0 }, { 31, 0, 32, 0 },
                        { 32, 1, 24, 2 }, { 24, 1, 12, 0 } };
    drawShape.ports = {
        { 11, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 31, 0, 1, "out", "control", "shape", "unspecified", "unspecified" },
        { 32, 0, 1, "in", "control", "shape", "unspecified", "unspecified" },
        { 32, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 24, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 24, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 24, 2, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 12, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    FakeLayer shapeLayer;
    check (videowire::executeVisualLayerPlan({ drawShape }, 7, shapeLayer, error, videohelper::geometry::PlanUse::preview)
           && shapeLayer.drawShape && shapeLayer.drawShapeCx == 0.25f
           && shapeLayer.drawShapeCy == 0.75f && shapeLayer.drawShapeW == 0.5f
           && shapeLayer.drawShapeH == 0.2f && shapeLayer.drawShapeR == 0.1f
           && shapeLayer.drawShapeG == 0.2f && shapeLayer.drawShapeB == 0.3f
           && shapeLayer.drawShapeA == 0.4f,
           "rectangle Draw Shape lowers canonical payload into the shared renderer input");
    inspection = { 7, 0, 32, 1 };
    check (videowire::validateVisualInspectionTarget({ drawShape }, inspection, error)
           && videowire::classifyVisualInspectionTarget(drawShape, inspection)
                == videowire::VisualInspectionSlice::retainedDrawShape,
           "Draw Shape output admits its exact native render-pass boundary");
    FakeLayer inspectedShapeLayer;
    check (videowire::executeVisualLayerPlan({ drawShape }, 7, inspectedShapeLayer, error,videohelper::geometry::PlanUse::preview,
                                             &inspection, nullptr)
           && inspectedShapeLayer.inspectionDrawShapeOutput,
           "executor marks only the exact selected Draw Shape output for retention");
    inspection.outputPort = 0;
    check (! videowire::validateVisualInspectionTarget({ drawShape }, inspection, error),
           "Draw Shape control input cannot alias the retained image output");
    inspection = { 7, 1, 32, 1 };
    FakeLayer staleShapeLayer;
    check (videowire::executeVisualLayerPlan({ drawShape }, 7, staleShapeLayer, error,videohelper::geometry::PlanUse::preview,
                                             &inspection, nullptr)
           && ! staleShapeLayer.inspectionDrawShapeOutput,
           "stale Draw Shape revision never arms backend retention");
    auto drawEllipse = drawShape;
    drawEllipse.nodeKinds[1] = "visual.shape.ellipse";
    drawEllipse.operations[1].kind = "visual.shape.ellipse";
    FakeLayer ellipseLayer;
    check (videowire::executeVisualLayerPlan({ drawEllipse }, 7, ellipseLayer, error, videohelper::geometry::PlanUse::preview)
           && ellipseLayer.drawShape && ellipseLayer.drawShapeEllipse,
           "ellipse Draw Shape lowers to the native shape renderer");

    auto booleanShape = drawShape;
    booleanShape.nodeKinds = { "video.source", "visual.shape.rectangle",
        "visual.shape.ellipse", "visual.shape.subtract", "visual.draw.shape",
        "video.blend", "video.out" };
    booleanShape.nodeIds = { 11, 31, 33, 34, 32, 24, 12 };
    booleanShape.operations = {
        { 11, "video.source", "source-decode", "" },
        { 31, "visual.shape.rectangle", "control-eval",
          "<NodeParams centerX=\"0.4\" centerY=\"0.5\" width=\"0.8\" height=\"0.6\"/>" },
        { 33, "visual.shape.ellipse", "control-eval",
          "<NodeParams centerX=\"0.6\" centerY=\"0.5\" width=\"0.3\" height=\"0.2\"/>" },
        { 34, "visual.shape.subtract", "control-eval", "" },
        { 32, "visual.draw.shape", "native-gpu",
          "<NodeParams red=\"0.1\" green=\"0.2\" blue=\"0.3\" alpha=\"0.4\"/>" },
        { 24, "video.blend", "native-gpu", "" },
        { 12, "video.out", "native-gpu", "" }
    };
    booleanShape.ports = {
        { 11, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 31, 0, 1, "out", "control", "shape", "unspecified", "unspecified" },
        { 33, 0, 1, "out", "control", "shape", "unspecified", "unspecified" },
        { 34, 0, 1, "in", "control", "shape", "unspecified", "unspecified" },
        { 34, 1, 1, "in", "control", "shape", "unspecified", "unspecified" },
        { 34, 2, 1, "out", "control", "shape", "unspecified", "unspecified" },
        { 32, 0, 1, "in", "control", "shape", "unspecified", "unspecified" },
        { 32, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 24, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 24, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 24, 2, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 12, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    booleanShape.edges = { { 11, 0, 24, 0 }, { 31, 0, 34, 0 },
        { 33, 0, 34, 1 }, { 34, 2, 32, 0 }, { 32, 1, 24, 2 }, { 24, 1, 12, 0 } };
    FakeLayer booleanLayer, exportBooleanLayer;
    check (videowire::validateCompiledVisualLayerPlans({}, { booleanShape }, false, error)
           && videowire::compileVisualLayerExecution(booleanShape, execution, error)
           && execution.drawShapePrimaryNodeId == 31
           && execution.drawShapeSecondaryNodeId == 33
           && videowire::executeVisualLayerPlan({ booleanShape }, 7, booleanLayer, error, videohelper::geometry::PlanUse::preview)
           && videowire::executeVisualLayerPlan({ booleanShape }, 7, exportBooleanLayer, error, videohelper::geometry::PlanUse::preview)
           && booleanLayer.drawShape && booleanLayer.drawShapeHasSecondary
           && ! booleanLayer.drawShapeEllipse && booleanLayer.drawShapeSecondaryEllipse
           && booleanLayer.drawShapeOperation == 3
           && booleanLayer.drawShapeCx == 0.4f && booleanLayer.drawShapeW == 0.8f
           && booleanLayer.drawShape2Cx == 0.6f && booleanLayer.drawShape2W == 0.3f
           && booleanLayer.drawShapeOperation == exportBooleanLayer.drawShapeOperation
           && booleanLayer.drawShapeCx == exportBooleanLayer.drawShapeCx
           && booleanLayer.drawShape2Cx == exportBooleanLayer.drawShape2Cx,
           "viewport/export lower the same two distinct immutable primitives into the native Boolean draw pass");

    struct BooleanAliasCase { const char* kind; int operation; const char* message; };
    for (const auto& aliasCase : {
             BooleanAliasCase { "visual.shape.union", 1,
                 "Shape Union admits one primitive identity aliased to both operands" },
             BooleanAliasCase { "visual.shape.intersection", 2,
                 "Shape Intersection admits one primitive identity aliased to both operands" },
             BooleanAliasCase { "visual.shape.subtract", 3,
                 "Shape Subtract admits one primitive identity aliased to both operands" } })
    {
        auto aliasedBoolean = booleanShape;
        aliasedBoolean.nodeKinds.erase(aliasedBoolean.nodeKinds.begin() + 2);
        aliasedBoolean.nodeIds.erase(aliasedBoolean.nodeIds.begin() + 2);
        aliasedBoolean.operations.erase(aliasedBoolean.operations.begin() + 2);
        aliasedBoolean.nodeKinds[2] = aliasCase.kind;
        aliasedBoolean.operations[2].kind = aliasCase.kind;
        aliasedBoolean.ports.erase(std::remove_if(aliasedBoolean.ports.begin(),
            aliasedBoolean.ports.end(), [](const auto& port) { return port.nodeId == 33; }),
            aliasedBoolean.ports.end());
        aliasedBoolean.edges[2].fromNodeId = 31;

        FakeLayer aliasedLayer;
        check (videowire::validateCompiledVisualLayerPlans(
                   {}, { aliasedBoolean }, false, error)
               && videowire::compileVisualLayerExecution(aliasedBoolean, execution, error)
               && execution.drawShapePrimaryNodeId == 31
               && execution.drawShapeSecondaryNodeId == 31
               && videowire::executeVisualLayerPlan(
                   { aliasedBoolean }, 7, aliasedLayer, error, videohelper::geometry::PlanUse::preview)
               && aliasedLayer.drawShape && aliasedLayer.drawShapeHasSecondary
               && aliasedLayer.drawShapeOperation == aliasCase.operation
               && ! aliasedLayer.drawShapeEllipse
               && ! aliasedLayer.drawShapeSecondaryEllipse
               && aliasedLayer.drawShapeCx == aliasedLayer.drawShape2Cx
               && aliasedLayer.drawShapeCy == aliasedLayer.drawShape2Cy
               && aliasedLayer.drawShapeW == aliasedLayer.drawShape2W
               && aliasedLayer.drawShapeH == aliasedLayer.drawShape2H,
               aliasCase.message);
    }

    auto missingPrimitiveBinding = booleanShape;
    missingPrimitiveBinding.ports.erase(std::remove_if(missingPrimitiveBinding.ports.begin(),
        missingPrimitiveBinding.ports.end(), [](const auto& port)
        { return port.nodeId == 33 && port.port == 0; }), missingPrimitiveBinding.ports.end());
    check (! videowire::compileVisualLayerExecution(missingPrimitiveBinding, execution, error)
           && error == "bounded visual DAG has an incompatible typed edge",
           "Boolean Draw Shape rejects a primitive without its exact typed Shape output");

    auto mistypedDrawBinding = booleanShape;
    auto drawInputPort = std::find_if(mistypedDrawBinding.ports.begin(), mistypedDrawBinding.ports.end(),
        [](const auto& port) { return port.nodeId == 32 && port.port == 0; });
    drawInputPort->carrier = "frame";
    check (! videowire::compileVisualLayerExecution(mistypedDrawBinding, execution, error)
           && error == "bounded visual DAG has an incompatible typed edge",
           "Boolean Draw Shape rejects an image-carrier alias for its typed Shape input");

    auto mismatchedDrawOutput = booleanShape;
    auto drawOutputPort = std::find_if(mismatchedDrawOutput.ports.begin(), mismatchedDrawOutput.ports.end(),
        [](const auto& port) { return port.nodeId == 32 && port.port == 1; });
    drawOutputPort->colorSpace = "linearSRGB";
    check (! videowire::compileVisualLayerExecution(mismatchedDrawOutput, execution, error)
           && error == "bounded visual DAG requires RGBA8 sRGB Frame<Image> descriptors",
           "Boolean Draw Shape rejects implicit output color conversion");

    auto outOfBoundsPrimitive = booleanShape;
    outOfBoundsPrimitive.operations[1].payloadXml =
        "<NodeParams centerX=\"0.4\" centerY=\"0.5\" width=\"4.01\" height=\"0.6\"/>";
    check (! videowire::compileVisualLayerExecution(outOfBoundsPrimitive, execution, error)
           && error == "visual Draw Shape primitive payload is malformed or out of bounds",
           "Boolean Draw Shape rejects primitive dimensions outside the graph parameter domain");

    auto malformedPrimitive = booleanShape;
    malformedPrimitive.operations[2].payloadXml =
        "<NodeParams centerX=\"0.6junk\" centerY=\"0.5\" width=\"0.3\" height=\"0.2\"/>";
    check (! videowire::compileVisualLayerExecution(malformedPrimitive, execution, error)
           && error == "visual Draw Shape primitive payload is malformed or out of bounds",
           "Boolean Draw Shape rejects partially parsed primitive values");

    auto nonFinitePrimitive = booleanShape;
    nonFinitePrimitive.operations[2].payloadXml =
        "<NodeParams centerX=\"nan\" centerY=\"0.5\" width=\"0.3\" height=\"0.2\"/>";
    check (! videowire::compileVisualLayerExecution(nonFinitePrimitive, execution, error)
           && error == "visual Draw Shape primitive payload is malformed or out of bounds",
           "Boolean Draw Shape rejects non-finite primitive values");

    auto outOfBoundsColor = booleanShape;
    outOfBoundsColor.operations[4].payloadXml =
        "<NodeParams red=\"0.1\" green=\"0.2\" blue=\"0.3\" alpha=\"1.01\"/>";
    check (! videowire::compileVisualLayerExecution(outOfBoundsColor, execution, error)
           && error == "visual Draw Shape color payload is malformed or out of bounds",
           "Boolean Draw Shape rejects color components outside the graph parameter domain");

    auto inventedBooleanPayload = booleanShape;
    inventedBooleanPayload.operations[3].payloadXml = "<NodeParams invented=\"1\"/>";
    check (! videowire::validateCompiledVisualLayerPlans(
               {}, { inventedBooleanPayload }, false, error)
           && error == "visual Shape Boolean operation payload must be empty",
           "snapshot admission rejects invented Shape Boolean payload keys");
    check (! videowire::compileVisualLayerExecution(inventedBooleanPayload, execution, error)
           && error == "visual Shape Boolean operation payload must be empty",
           "executor rejects invented Shape Boolean payload keys");
    check (! videowire::compileVisualLayerExecutionOrdered(
               inventedBooleanPayload, execution, error)
           && error == "visual Shape Boolean operation payload must be empty",
           "ordered executor rejects invented Shape Boolean payload keys before lowering");

    auto malformedBoolean = booleanShape;
    malformedBoolean.edges.erase(malformedBoolean.edges.begin() + 2);
    check (! videowire::compileVisualLayerExecution(malformedBoolean, execution, error)
           && error == "bounded visual DAG contains a disconnected operation",
           "Boolean Shape admission rejects a missing exact operand binding");

    drawShape.operations[2].backendCapability = "cpu-fallback";
    check (! videowire::compileVisualLayerExecution(drawShape, execution, error),
           "Draw Shape rejects CPU fallback capability");

    auto cpu = direct;
    cpu.operations[1].backendCapability = "cpu-fallback";
    check (! videowire::compileVisualLayerExecution (cpu, execution, error)
           && error == "visual layer plan requires unsupported execution capability",
           "CPU image fallback is rejected");

    auto feedback = direct;
    feedback.structuralRevision = 4;
    feedback.nodeKinds.insert(feedback.nodeKinds.begin() + 1, "visual.feedback");
    feedback.nodeIds.insert(feedback.nodeIds.begin() + 1, 30);
    feedback.operations.insert(feedback.operations.begin() + 1,
        { 30, "visual.feedback", "native-gpu",
          visualtemporaloperation::serialize([]
          {
              visualtemporaloperation::Payload payload;
              payload.mode = visualtemporaloperation::Mode::feedback;
              payload.historyLength = 2;
              payload.decay = 0.95f;
              payload.zoom = 1.05f;
              payload.swirl = -0.05f;
              return payload;
          }()) });
    feedback.ports.push_back({ 30, 0, 1, "in", "frame", "image", "rgba8", "sRGB" });
    feedback.ports.push_back({ 30, 1, 1, "out", "frame", "image", "rgba8", "sRGB" });
    feedback.edges = { { 11, 0, 30, 0 }, { 30, 1, 12, 0 } };
    auto feedbackState = std::make_unique<videowire::VisualPlanExecutionState>();
    check (feedbackState->admitPlans({ feedback }, &error, 640, 360),
           "graph feedback admits one exact bounded native history resource");
    const auto* admittedFeedback = feedbackState->compiled(7, feedback.structuralRevision);
    check (admittedFeedback != nullptr && admittedFeedback->temporalFeedbackPass.has_value()
           && admittedFeedback->temporalFeedbackPass->clipId == 7
           && admittedFeedback->temporalFeedbackPass->nodeId == 30
           && admittedFeedback->temporalFeedbackPass->extent.width == 640
           && admittedFeedback->temporalFeedbackPass->extent.height == 360
           && admittedFeedback->temporalFeedbackPass->format
                == videotemporal::PixelFormat::rgba16f
           && admittedFeedback->temporalFeedbackPass->historyLength == 2
           && admittedFeedback->temporalFeedbackPass->footprint.retainedImages == 2
           && admittedFeedback->temporalFeedbackPass->footprint.historyBytes
                == 640ull * 360ull * 8ull * 2ull,
           "feedback admission transports exact owner, extent, format and footprint");
    check (feedbackState->resourceReceipt().peakLiveFrames == 2
           && feedbackState->resourceReceipt().frameSlots == 2,
           "feedback history contributes to both peak-live and allocated-slot receipts");
    FakeLayer feedbackLayer;
    check (videowire::executeVisualLayerPlan({ feedback }, 7, feedbackLayer, error,videohelper::geometry::PlanUse::preview,
                                             nullptr, nullptr, feedbackState.get(), 1.0)
           && feedbackLayer.effects == &feedbackLayer.graphFeedbackEffect
           && feedbackLayer.effectCount == 1 && feedbackLayer.graphFeedbackEffect.type == 25
           && feedbackLayer.graphTemporalActive && feedbackLayer.graphTemporalNodeId == 30
           && feedbackLayer.graphTemporalPayload.mode == visualtemporaloperation::Mode::feedback
           && feedbackLayer.graphFeedbackEffect.params[0] == 0.95f
           && feedbackLayer.graphFeedbackEffect.params[1] == 1.05f
           && feedbackLayer.graphFeedbackEffect.params[2] == -0.05f
           && feedbackLayer.feedbackHistoryReset && ! feedbackLayer.feedbackHistoryHold,
           "graph feedback lowers validated immutable payload into production FeedbackTrail");
    FakeLayer heldFeedback;
    check (videowire::executeVisualLayerPlan({ feedback }, 7, heldFeedback, error,videohelper::geometry::PlanUse::preview,
                                             nullptr, nullptr, feedbackState.get(), 1.0)
           && heldFeedback.feedbackHistoryHold && ! heldFeedback.feedbackHistoryReset,
           "same-time feedback presentation holds production history");
    FakeLayer resetFeedback;
    check (videowire::executeVisualLayerPlan({ feedback }, 7, resetFeedback, error,videohelper::geometry::PlanUse::preview,
                                             nullptr, nullptr, feedbackState.get(), 0.5)
           && resetFeedback.feedbackHistoryReset && ! resetFeedback.feedbackHistoryHold,
           "backward feedback evaluation resets per-owner production history");
    auto newerWithoutFeedback = feedback;
    ++newerWithoutFeedback.structuralRevision;
    newerWithoutFeedback.operations[1].kind = "visual.shape";
    check(! videowire::visualPlansUseTemporalFeedback(
              { feedback, newerWithoutFeedback }),
          "stale temporal feedback revisions cannot enable export pre-roll");
    auto staleWithoutFeedback = feedback;
    staleWithoutFeedback.operations[1].kind = "visual.shape";
    auto newerFeedback = feedback;
    ++newerFeedback.structuralRevision;
    check(videowire::visualPlansUseTemporalFeedback(
              { staleWithoutFeedback, newerFeedback }),
          "the selected revision alone controls export feedback pre-roll");
    auto equalRevisionWithoutFeedback = feedback;
    equalRevisionWithoutFeedback.operations[1].kind = "visual.shape";
    const std::vector<videowire::CompiledVisualLayerPlan> equalRevisionPlans {
        equalRevisionWithoutFeedback, feedback
    };
    check(! videowire::visualPlansUseTemporalFeedback(equalRevisionPlans)
          && videowire::findVisualLayerPlan(equalRevisionPlans, 7)
                 == &equalRevisionPlans.front(),
          "equal clip revisions consistently preserve the first plan");

    const auto temporalPlan = [&feedback](const char* kind,
                                          const visualtemporaloperation::Payload& payload)
    {
        auto plan = feedback;
        plan.structuralRevision += static_cast<std::uint64_t>(payload.mode) + 10u;
        plan.nodeKinds[1] = kind;
        plan.operations[1].kind = kind;
        plan.operations[1].payloadXml = visualtemporaloperation::serialize(payload);
        return plan;
    };
    const auto checkTemporalMode = [&](const char* kind,
                                       const visualtemporaloperation::Payload& payload)
    {
        auto plan = temporalPlan(kind, payload);
        auto temporalState = std::make_unique<videowire::VisualPlanExecutionState>();
        FakeLayer temporalLayer;
        error.clear();
        if (! temporalState->admitPlans({ plan }, &error, 320, 180))
        {
            std::fprintf(stderr, "temporal admit %s: %s\n", kind, error.c_str());
            return false;
        }
        if (! videowire::executeVisualLayerPlan({ plan }, 7, temporalLayer, error,videohelper::geometry::PlanUse::preview,
                                                nullptr, nullptr, temporalState.get(), 1.0))
        {
            std::fprintf(stderr, "temporal execute %s: %s\n", kind, error.c_str());
            return false;
        }
        return temporalLayer.graphTemporalActive
            && temporalLayer.graphTemporalNodeId == 30
            && temporalLayer.graphTemporalPayload == payload
            && temporalLayer.effects == nullptr
            && temporalState->compiled(7, plan.structuralRevision) != nullptr
            && temporalState->compiled(7, plan.structuralRevision)->temporalFeedbackPass->payload
                 == payload;
    };
    visualtemporaloperation::Payload frameDelayPayload;
    frameDelayPayload.mode = visualtemporaloperation::Mode::frameDelay;
    frameDelayPayload.historyLength = 3;
    frameDelayPayload.decay = 0.0f;
    frameDelayPayload.zoom = 1.0f;
    check(checkTemporalMode("visual.frame-delay", frameDelayPayload),
          "frame delay transports one exact immutable bounded GPU payload");
    visualtemporaloperation::Payload echoPayload;
    echoPayload.mode = visualtemporaloperation::Mode::echo;
    echoPayload.historyLength = 4;
    echoPayload.mix = 0.6f;
    echoPayload.decay = 0.0f;
    echoPayload.zoom = 1.0f;
    check(checkTemporalMode("visual.echo", echoPayload),
          "echo transports one exact immutable bounded GPU payload");
    visualtemporaloperation::Payload stutterPayload;
    stutterPayload.mode = visualtemporaloperation::Mode::stutter;
    stutterPayload.historyLength = 1;
    stutterPayload.holdFrames = 6;
    stutterPayload.decay = 0.0f;
    stutterPayload.zoom = 1.0f;
    check(checkTemporalMode("visual.stutter", stutterPayload),
          "stutter transports one exact immutable bounded GPU payload");
    visualtemporaloperation::Payload exposurePayload;
    exposurePayload.mode = visualtemporaloperation::Mode::longExposure;
    exposurePayload.historyLength = 8;
    exposurePayload.decay = 0.0f;
    exposurePayload.zoom = 1.0f;
    check(checkTemporalMode("visual.long-exposure", exposurePayload),
          "long exposure transports one exact immutable bounded GPU payload");

    auto staleTemporalPlan = temporalPlan("visual.echo", echoPayload);
    auto staleTemporalState = std::make_unique<videowire::VisualPlanExecutionState>();
    check(staleTemporalState->admitPlans({ staleTemporalPlan }, &error, 320, 180),
          "temporal stale-state fixture admits current revision");
    ++staleTemporalPlan.structuralRevision;
    FakeLayer staleTemporalLayer;
    error.clear();
    check(! videowire::executeVisualLayerPlan({ staleTemporalPlan }, 7, staleTemporalLayer, error,videohelper::geometry::PlanUse::preview,
                                              nullptr, nullptr, staleTemporalState.get(), 2.0)
          && error == "render layer has no pre-admitted visual execution",
          "temporal execution rejects stale structural state before native rendering");

    FakeProductRenderer viewportFeedbackRenderer;
    FakeProductRenderer exportFeedbackRenderer;
    FakeLayer viewportFeedbackLayer;
    FakeLayer exportFeedbackLayer;
    auto viewportFeedbackState = std::make_unique<videowire::VisualPlanExecutionState>();
    auto exportFeedbackState = std::make_unique<videowire::VisualPlanExecutionState>();
    check (viewportFeedbackState->admitPlans({ feedback }, &error, 640, 360)
           && exportFeedbackState->admitPlans({ feedback }, &error, 640, 360)
           && videowire::executeVisualLayerPlanForRenderer(
                viewportFeedbackRenderer, { feedback }, 7, viewportFeedbackLayer, error,videohelper::geometry::PlanUse::preview, nullptr,
                nullptr, nullptr, viewportFeedbackState.get(), 1.0)
           && videowire::executeVisualLayerPlanForRenderer(
                exportFeedbackRenderer, { feedback }, 7, exportFeedbackLayer, error,videohelper::geometry::PlanUse::preview, nullptr,
                nullptr, nullptr, exportFeedbackState.get(), 1.0)
           && viewportFeedbackRenderer.temporalFeedbackPreparations == 1
           && exportFeedbackRenderer.temporalFeedbackPreparations == 1
           && viewportFeedbackRenderer.bridgePreparations == 1
           && exportFeedbackRenderer.bridgePreparations == 1
           && viewportFeedbackRenderer.preparedTemporalFeedback.has_value()
           && exportFeedbackRenderer.preparedTemporalFeedback.has_value()
           && viewportFeedbackRenderer.preparedTemporalFeedback->footprint.historyBytes
                == exportFeedbackRenderer.preparedTemporalFeedback->footprint.historyBytes,
           "viewport and export invoke the same pre-admitted native feedback pass");

    FakeProductRenderer missingFeedbackRenderer;
    FakeLayer missingFeedbackLayer;
    error.clear();
    check (! videowire::executeVisualLayerPlanForRenderer(
                missingFeedbackRenderer, { feedback }, 7, missingFeedbackLayer, error, videohelper::geometry::PlanUse::preview, nullptr)
           && missingFeedbackRenderer.temporalFeedbackPreparations == 0
           && missingFeedbackRenderer.bridgePreparations == 0
           && error == "visual temporal operation requires a pre-admitted temporal GPU resource",
           "product execution cannot allocate graph history outside snapshot admission");

    FakeProductRenderer rejectedFeedbackRenderer;
    rejectedFeedbackRenderer.rejectTemporalFeedback = true;
    FakeLayer rejectedFeedbackLayer;
    error.clear();
    check (! videowire::executeVisualLayerPlanForRenderer(
                rejectedFeedbackRenderer, { feedback }, 7, rejectedFeedbackLayer, error,videohelper::geometry::PlanUse::preview, nullptr,
                nullptr, nullptr, viewportFeedbackState.get(), 2.0)
           && rejectedFeedbackRenderer.temporalFeedbackPreparations == 1
           && rejectedFeedbackRenderer.bridgePreparations == 0
           && error == "native temporal feedback backend is unsupported",
           "product execution fails closed when the native feedback backend rejects the pass");

    auto invalidFeedbackPayload = feedback;
    invalidFeedbackPayload.operations[1].payloadXml =
        "<NodeParams decay=\"2\" zoom=\"1.0\" swirl=\"0\"/>";
    check (! videowire::compileVisualLayerExecution(invalidFeedbackPayload, execution, error)
           && error == "visual.feedback payload is malformed or out of bounds",
           "feedback payloads outside the producer domain fail closed instead of clamping");

    auto secondFeedback = feedback;
    secondFeedback.clipId = 8;
    secondFeedback.structuralRevision = 5;
    auto constrainedFeedbackLimits = videowire::VisualBackendResourceLimits::forCanvas(64, 64);
    constrainedFeedbackLimits.frameSlots = 3;
    constrainedFeedbackLimits.liveFrames = 4;
    auto constrainedFeedbackState = std::make_unique<videowire::VisualPlanExecutionState>();
    error.clear();
    check (! constrainedFeedbackState->admitPlans(
                { feedback, secondFeedback }, &error, 64, 64, &constrainedFeedbackLimits)
           && error == "visual graph allocated frame slot capacity exceeded: 4 > 3",
           "retained histories consume the aggregate native frame-slot budget");

    auto viewportState = std::make_unique<videowire::VisualPlanExecutionState>();
    auto exportState = std::make_unique<videowire::VisualPlanExecutionState>();
    auto ownerPlan = directPlan(); ownerPlan.structuralRevision = 4;
    viewportState->admitPlans({ ownerPlan });
    const auto publishTime = [&](videowire::VisualPlanExecutionState& state,
                                 uint64_t revision, double time)
    {
        videowire::TemporalSamplingCommitTransaction transaction;
        visualtemporalsampling::LifecycleState lifecycle;
        std::string transactionError;
        return state.temporalLifecycleSnapshot(7, revision, lifecycle)
            && transaction.add(state, 7, revision, time, std::move(lifecycle), transactionError)
            && transaction.commit(transactionError);
    };
    check (publishTime(*viewportState, 4, 1.0) && viewportState->owner(7)->evaluationSequence == 1,
           "temporal owner starts an explicit evaluation sequence");
    check (publishTime(*viewportState, 4, 1.0) && viewportState->owner(7)->evaluationSequence == 1,
           "same-time presentation does not advance temporal state");
    check (publishTime(*viewportState, 4, 2.0) && viewportState->owner(7)->evaluationSequence == 2,
           "forward evaluation advances exactly once");
    check (publishTime(*viewportState, 4, 0.5) && viewportState->owner(7)->evaluationSequence == 1,
           "backward seek commits a reset with the published frame");
    ownerPlan.structuralRevision = 5;
    viewportState->admitPlans({ ownerPlan });
    exportState->admitPlans({ ownerPlan });
    check (publishTime(*viewportState, 5, 0.5) && viewportState->owner(7)->evaluationSequence == 1,
           "structural revision replacement resets temporal ownership");
    check (publishTime(*exportState, 5, 0.5)
           && publishTime(*viewportState, 5, 1.0)
           && exportState->owner(7)->evaluationSequence == 1,
           "viewport and export owners share semantics without sharing state");
    viewportState->reset(7);
    check (viewportState->owner(7).has_value()
           && viewportState->owner(7)->structuralRevision == 5
           && ! viewportState->owner(7)->hasTime,
           "explicit owner reset publishes an empty state for the admitted revision");

    auto motionBlur = directPlan();
    motionBlur.structuralRevision = 40;
    motionBlur.nodeKinds.insert(motionBlur.nodeKinds.begin() + 1, "visual.motion-blur");
    motionBlur.nodeIds.insert(motionBlur.nodeIds.begin() + 1, 40);
    visualtemporalsampling::Payload motionBlurPayload;
    motionBlurPayload.mode = visualtemporalsampling::Mode::motionBlur;
    motionBlurPayload.sampleCount = 8;
    motionBlurPayload.historyWeight = 0.0f;
    motionBlurPayload.jitterSpread = 0.0f;
    motionBlurPayload.shutterAngleDegrees = 180.0f;
    motionBlurPayload.motionScale = 1.0f;
    motionBlur.operations.insert(motionBlur.operations.begin() + 1,
        { 40, "visual.motion-blur", "temporal-sampling",
          visualtemporalsampling::serialize(motionBlurPayload) });
    motionBlur.ports.push_back({ 40, 0, 1, "in", "frame", "image", "rgba8", "sRGB" });
    motionBlur.ports.push_back({ 40, 1, 2, "in", "frame", "motionVectors", "rg16f", "unspecified" });
    motionBlur.ports.push_back({ 40, 2, 1, "out", "frame", "image", "rgba8", "sRGB" });
    auto motionState = std::make_unique<videowire::VisualPlanExecutionState>();
    check(motionState->admitPlans({ motionBlur }, &error, 320, 180)
          && motionState->compiled(7, 40)->temporalSamplingPass->footprint.transientBytes > 0
          && motionState->resourceReceipt().peakLiveFrames
               == visualtemporalsampling::requirements(motionBlurPayload.mode).retainedHistoryImages
                + visualtemporalsampling::requirements(motionBlurPayload.mode).transientImages
          && motionState->resourceReceipt().frameSlots
               == visualtemporalsampling::requirements(motionBlurPayload.mode).retainedHistoryImages
                + visualtemporalsampling::requirements(motionBlurPayload.mode).transientImages
          && motionState->resourceReceipt().allocatedFrameBytes
               == motionState->compiled(7, 40)->temporalSamplingPass->footprint.historyBytes
                + motionState->compiled(7, 40)->temporalSamplingPass->footprint.transientBytes
          && motionState->telemetry().snapshot().resourcesObserved,
          "motion blur history and transient images publish exact aggregate receipts and telemetry");

    const auto ownerBeforeDiscard = motionState->owner(7);
    videowire::TemporalSamplingCommitTransaction discardedCandidate;
    visualtemporalsampling::LifecycleState discardedLifecycle;
    check(motionState->temporalLifecycleSnapshot(7, 40, discardedLifecycle)
          && discardedCandidate.add(*motionState, 7, 40, 3.0,
                                    std::move(discardedLifecycle), error),
          "a temporal frame candidate is derived from the immutable owner snapshot");
    discardedCandidate.discard();
    const auto ownerAfterDiscard = motionState->owner(7);
    check(ownerBeforeDiscard.has_value() && ownerAfterDiscard.has_value()
          && ownerBeforeDiscard->structuralRevision == ownerAfterDiscard->structuralRevision
          && ownerBeforeDiscard->lastTimeSec == ownerAfterDiscard->lastTimeSec
          && ownerBeforeDiscard->evaluationSequence == ownerAfterDiscard->evaluationSequence
          && ownerBeforeDiscard->hasTime == ownerAfterDiscard->hasTime,
          "discarding preparation leaves durable temporal owner bytes unchanged");

    videowire::TemporalSamplingCommitTransaction resetRaceCandidate;
    visualtemporalsampling::LifecycleState resetRaceLifecycle;
    check(motionState->temporalLifecycleSnapshot(7, 40, resetRaceLifecycle)
          && resetRaceCandidate.add(*motionState, 7, 40, 4.0,
                                    std::move(resetRaceLifecycle), error),
          "reset-race candidate prepares before reset publication");
    std::mutex resetGateMutex;
    std::condition_variable resetGate;
    bool releaseReset = false;
    bool resetFinished = false;
    std::thread resetThread([&]
    {
        std::unique_lock<std::mutex> lock(resetGateMutex);
        resetGate.wait(lock, [&] { return releaseReset; });
        lock.unlock();
        motionState->reset(7);
        lock.lock();
        resetFinished = true;
        lock.unlock();
        resetGate.notify_one();
    });
    {
        std::lock_guard<std::mutex> lock(resetGateMutex);
        releaseReset = true;
    }
    resetGate.notify_one();
    {
        std::unique_lock<std::mutex> lock(resetGateMutex);
        resetGate.wait(lock, [&] { return resetFinished; });
    }
    resetThread.join();
    check(! resetRaceCandidate.commit(error)
          && error == "temporal sampling snapshot changed before frame publication"
          && motionState->owner(7)->structuralRevision == 40
          && ! motionState->owner(7)->hasTime,
          "concurrent reset invalidates the whole candidate with zero candidate mutation");

    auto secondMotionBlur = motionBlur;
    secondMotionBlur.clipId = 8;
    auto atomicState = std::make_unique<videowire::VisualPlanExecutionState>();
    check(atomicState->admitPlans({ motionBlur, secondMotionBlur }, &error, 320, 180),
          "two-owner temporal fixture admits production plans");
    videowire::TemporalSamplingCommitTransaction atomicFrame;
    visualtemporalsampling::LifecycleState firstAtomicLifecycle, secondAtomicLifecycle;
    check(atomicState->temporalLifecycleSnapshot(7, 40, firstAtomicLifecycle)
          && atomicState->temporalLifecycleSnapshot(8, 40, secondAtomicLifecycle)
          && atomicFrame.add(*atomicState, 7, 40, 1.0, std::move(firstAtomicLifecycle), error)
          && atomicFrame.add(*atomicState, 8, 40, 1.0, std::move(secondAtomicLifecycle), error),
          "multi-candidate frame prepares as one transaction");
    atomicState->reset(8);
    check(! atomicFrame.commit(error)
          && ! atomicState->owner(7)->hasTime && ! atomicState->owner(8)->hasTime,
          "one stale candidate rejects the complete frame without partial publication");

    bool promotedPublication = false;
    bool discardedPublication = false;
    videowire::TemporalSamplingCommitTransaction rejectedPublication;
    rejectedPublication.addPublication(
        [](std::string& value)
        { value = "injected presentation rejection"; return false; },
        [&] { promotedPublication = true; },
        [&] { discardedPublication = true; });
    check(! rejectedPublication.commit(error) && ! promotedPublication
          && discardedPublication,
          "failed production publication validation promotes nothing and discards its candidate");
    bool teardownDiscarded = false;
    {
        videowire::TemporalSamplingCommitTransaction abandonedPublication;
        abandonedPublication.addPublication(
            [](std::string&) { return true; }, [] {},
            [&] { teardownDiscarded = true; });
    }
    check(teardownDiscarded,
          "abandoned production publication releases its candidate during teardown");
    videowire::injectTemporalPublicationFailure(
        videowire::TemporalPublicationFailurePoint::exportEncoder);
    check(! videowire::consumeTemporalPublicationFailure(
              videowire::TemporalPublicationFailurePoint::viewportPresentation)
          && videowire::consumeTemporalPublicationFailure(
              videowire::TemporalPublicationFailurePoint::exportEncoder),
          "production publication failure injection is deterministic and stage-specific");

    videowire::TemporalSamplingCommitTransaction destroyedOwnerCandidate;
    {
        auto doomedState = std::make_unique<videowire::VisualPlanExecutionState>();
        check(doomedState->admitPlans({ motionBlur }, &error, 320, 180),
              "owner-destruction fixture admits a production plan");
        visualtemporalsampling::LifecycleState doomedLifecycle;
        check(doomedState->temporalLifecycleSnapshot(7, 40, doomedLifecycle)
              && destroyedOwnerCandidate.add(*doomedState, 7, 40, 1.0,
                                             std::move(doomedLifecycle), error),
              "owner-destruction fixture prepares a weak-safe candidate");
    }
    check(! destroyedOwnerCandidate.commit(error)
          && error == "temporal sampling owner was destroyed before frame publication",
          "destroyed owners cannot be dereferenced or published");

    auto pipelineState = std::make_unique<videowire::VisualPlanExecutionState>();
    check(pipelineState->admitPlans({ motionBlur }, &error, 320, 180),
          "pipeline fixture admits a production plan");
    videowire::TemporalSamplingCommitTransaction predecessor;
    visualtemporalsampling::LifecycleState predecessorLifecycle;
    check(pipelineState->temporalLifecycleSnapshot(7, 40, predecessorLifecycle)
          && predecessor.add(*pipelineState, 7, 40, 1.0,
                             std::move(predecessorLifecycle), error),
          "pipeline predecessor prepares");
    videowire::TemporalSamplingCommitTransaction descendant;
    descendant.seedFrom(predecessor);
    visualtemporalsampling::LifecycleState descendantLifecycle;
    check(pipelineState->temporalLifecycleSnapshot(7, 40, descendantLifecycle)
          && descendant.add(*pipelineState, 7, 40, 2.0,
                            std::move(descendantLifecycle), error),
          "pipeline descendant chains from the uncommitted immutable predecessor");
    check(! descendant.commit(error)
          && error == "temporal sampling predecessor was not committed"
          && ! pipelineState->owner(7)->hasTime,
          "out-of-order descendant publication fails without poisoning the owner");
    check(predecessor.commit(error) && pipelineState->owner(7)->evaluationSequence == 1,
          "a rejected descendant does not poison its predecessor");

    videowire::TemporalSamplingExecution firstSampling;
    check(motionState->prepareTemporalSamplingExecution(7, 40, 1.0, 1, false,
              visualtemporalsampling::EvaluationMode::preview, firstSampling, error)
          && firstSampling.pass.clipId == 7 && firstSampling.pass.nodeId == 40
          && firstSampling.pass.structuralRevision == 40
          && firstSampling.pass.payload == motionBlurPayload
          && firstSampling.point.ownerId == 7 && firstSampling.point.structuralRevision == 40
          && firstSampling.point.width == 320 && firstSampling.point.height == 180
          && firstSampling.point.helperGeneration == 1 && firstSampling.point.timeSec == 1.0
          && firstSampling.point.mode == visualtemporalsampling::EvaluationMode::preview
          && firstSampling.transition.action
               == visualtemporalsampling::TransitionAction::resetAndAdvance
          && firstSampling.transition.resetCause
               == visualtemporalsampling::ResetCause::firstEvaluation,
          "motion blur publishes one complete immutable compositor request");
    const auto publishedSampling = firstSampling;
    visualtemporalsampling::Transition samplingTransition;
    check(motionState->evaluateTemporalSampling(7, 40, 1.0, 1, false,
              visualtemporalsampling::EvaluationMode::preview, samplingTransition, error)
          && samplingTransition.action == visualtemporalsampling::TransitionAction::hold
          && motionState->evaluateTemporalSampling(7, 40, 2.0, 1, false,
              visualtemporalsampling::EvaluationMode::preview, samplingTransition, error)
          && samplingTransition.action == visualtemporalsampling::TransitionAction::advance
          && motionState->evaluateTemporalSampling(7, 40, 0.5, 1, false,
              visualtemporalsampling::EvaluationMode::preview, samplingTransition, error)
          && samplingTransition.action == visualtemporalsampling::TransitionAction::resetAndAdvance,
          "motion blur snapshot lifecycle resets, holds, and advances deterministically");
    check(publishedSampling.point.timeSec == 1.0
          && publishedSampling.transition.action
               == visualtemporalsampling::TransitionAction::resetAndAdvance,
          "later lifecycle evaluations cannot rewrite a published compositor request");
    check(motionState->admitPlans({ motionBlur }, &error, 320, 180)
          && motionState->evaluateTemporalSampling(7, 40, 0.5, 1, false,
              visualtemporalsampling::EvaluationMode::preview, samplingTransition, error)
          && samplingTransition.action == visualtemporalsampling::TransitionAction::hold,
          "readmitting the same immutable plan preserves its temporal sampling lifecycle");
    auto revisedMotionBlur = motionBlur;
    revisedMotionBlur.structuralRevision = 41;
    videowire::TemporalSamplingExecution revisedSampling;
    check(motionState->admitPlans({ revisedMotionBlur }, &error, 320, 180)
          && motionState->prepareTemporalSamplingExecution(7, 41, 0.5, 1, false,
              visualtemporalsampling::EvaluationMode::preview, revisedSampling, error)
          && revisedSampling.pass.structuralRevision == 41
          && revisedSampling.transition.action
               == visualtemporalsampling::TransitionAction::resetAndAdvance
          && revisedSampling.transition.resetCause
               == visualtemporalsampling::ResetCause::structuralRevision,
          "a replacement snapshot reports its structural-revision reset through the shared owner");
    videowire::TemporalSamplingExecution rejectedSampling = publishedSampling;
    rejectedSampling.point.ownerId = 99;
    check(! motionState->prepareTemporalSamplingExecution(7, 41, 0.5, 1, false,
              visualtemporalsampling::EvaluationMode::exportFrame, rejectedSampling, error)
          && error == "temporal sampling lifecycle rejected the evaluation point"
          && rejectedSampling.point.ownerId == 99,
          "mixed preview/export evaluation fails closed without publishing partial state");
    auto exportMotionState = std::make_unique<videowire::VisualPlanExecutionState>();
    videowire::TemporalSamplingExecution exportSampling;
    check(exportMotionState->admitPlans({ motionBlur }, &error, 320, 180)
          && exportMotionState->prepareTemporalSamplingExecution(7, 40, 1.0, 1, false,
              visualtemporalsampling::EvaluationMode::exportFrame, exportSampling, error)
          && exportSampling.pass.payload == publishedSampling.pass.payload
          && exportSampling.transition.action
               == visualtemporalsampling::TransitionAction::resetAndAdvance,
          "preview and export consume the same admitted request through isolated lifecycle owners");
    FakeLayer unconnectedMotionLayer;
    check(! videowire::executeVisualLayerPlan({ revisedMotionBlur }, 7, unconnectedMotionLayer, error,videohelper::geometry::PlanUse::preview,
                                               nullptr, nullptr, motionState.get(), 1.0)
          && error == "visual.motion-blur sampling is admitted but not connected to native viewport/export composition",
          "motion blur sampling fails closed before claiming compositor execution");
    FakeLayer connectedMotionLayer;
    FakeProductRenderer connectedMotionRenderer;
    const videowire::VisualPlanEvaluationContext previewMotionContext {
        7, true, visualtemporalsampling::EvaluationMode::preview
    };
    videowire::TemporalSamplingCommitTransaction previewMotionTransaction;
    check(videowire::executeVisualLayerPlanForRenderer(
              connectedMotionRenderer, { revisedMotionBlur }, 7,
              connectedMotionLayer, error, videohelper::geometry::PlanUse::preview,
              &previewMotionTransaction, nullptr, nullptr, motionState.get(), 1.0,
              nullptr, nullptr, &previewMotionContext)
          && connectedMotionRenderer.motionAovExecutions == 0
          && connectedMotionRenderer.motionBlurPreparations == 1
          && connectedMotionRenderer.preparedMotionBlur.has_value()
          && connectedMotionRenderer.preparedMotionBlur->point.helperGeneration == 7
          && connectedMotionRenderer.preparedMotionBlur->point.paused
          && connectedMotionRenderer.preparedMotionBlur->point.mode
               == visualtemporalsampling::EvaluationMode::preview
          && connectedMotionRenderer.preparedMotionBlur->transition.action
               == visualtemporalsampling::TransitionAction::resetAndAdvance
          && connectedMotionLayer.graphMotionBlurActive
          && connectedMotionLayer.graphMotionTexture == 9
          && connectedMotionLayer.graphMotionBlur == motionBlurPayload,
          "motion blur prepares a publication-owned lifecycle candidate");
    check(previewMotionTransaction.commit(error),
          "preview publication commits its prepared temporal candidate");
    FakeLayer heldMotionLayer;
    videowire::TemporalSamplingCommitTransaction heldMotionTransaction;
    check(videowire::executeVisualLayerPlanForRenderer(
              connectedMotionRenderer, { revisedMotionBlur }, 7,
              heldMotionLayer, error, videohelper::geometry::PlanUse::preview,
              &heldMotionTransaction, nullptr, nullptr, motionState.get(), 1.0,
              nullptr, nullptr, &previewMotionContext)
          && connectedMotionRenderer.preparedMotionBlur->transition.action
               == visualtemporalsampling::TransitionAction::hold,
          "paused viewport motion blur holds its production temporal lifecycle");
    heldMotionTransaction.discard();
    const videowire::VisualPlanEvaluationContext playingPreviewMotionContext {
        7, false, visualtemporalsampling::EvaluationMode::preview
    };
    connectedMotionRenderer.rejectFlatShaderBridge = true;
    videowire::TemporalSamplingCommitTransaction rejectedMotionTransaction;
    check(! videowire::executeVisualLayerPlanForRenderer(
              connectedMotionRenderer, { revisedMotionBlur }, 7,
              heldMotionLayer, error, videohelper::geometry::PlanUse::preview,
              &rejectedMotionTransaction, nullptr, nullptr, motionState.get(), 2.0,
              nullptr, nullptr, &playingPreviewMotionContext),
          "rejected downstream frame preparation does not publish a temporal transition");
    check(rejectedMotionTransaction.empty(),
          "rejected preparation leaves no temporal candidate to commit");
    connectedMotionRenderer.rejectFlatShaderBridge = false;
    videowire::TemporalSamplingCommitTransaction retryMotionTransaction;
    check(videowire::executeVisualLayerPlanForRenderer(
              connectedMotionRenderer, { revisedMotionBlur }, 7,
              heldMotionLayer, error, videohelper::geometry::PlanUse::preview,
              &retryMotionTransaction, nullptr, nullptr, motionState.get(), 2.0,
              nullptr, nullptr, &playingPreviewMotionContext)
          && connectedMotionRenderer.preparedMotionBlur->transition.action
               == visualtemporalsampling::TransitionAction::advance,
          "retrying a rejected frame advances from the last committed temporal point");
    check(retryMotionTransaction.commit(error),
          "successful retry commits only at simulated publication");
    const videowire::VisualPlanEvaluationContext exportMotionContext {
        1, false, visualtemporalsampling::EvaluationMode::exportFrame
    };
    FakeLayer isolatedExportMotionLayer;
    FakeProductRenderer isolatedExportMotionRenderer;
    videowire::TemporalSamplingCommitTransaction exportMotionTransaction;
    check(videowire::executeVisualLayerPlanForRenderer(
              isolatedExportMotionRenderer, { motionBlur }, 7,
              isolatedExportMotionLayer, error, videohelper::geometry::PlanUse::exportRender,
              &exportMotionTransaction, nullptr, nullptr, exportMotionState.get(), 2.0,
              nullptr, nullptr, &exportMotionContext)
          && isolatedExportMotionRenderer.preparedMotionBlur.has_value()
          && isolatedExportMotionRenderer.preparedMotionBlur->point.mode
               == visualtemporalsampling::EvaluationMode::exportFrame
          && isolatedExportMotionRenderer.preparedMotionBlur->transition.action
               == visualtemporalsampling::TransitionAction::advance,
          "export motion blur prepares through its isolated production lifecycle");
    exportMotionTransaction.discard();
    auto tinyMotionLimits = videowire::VisualBackendResourceLimits::forCanvas(320, 180);
    tinyMotionLimits.allocatedFrameBytes = 1;
    auto rejectedMotionState = std::make_unique<videowire::VisualPlanExecutionState>();
    check(! rejectedMotionState->admitPlans({ motionBlur }, &error, 320, 180, &tinyMotionLimits)
          && error == "temporal sampling transient byte capacity exceeded",
          "motion blur snapshot rejects an insufficient transient image budget");

    auto history = direct;
    history.nodeKinds.insert(history.nodeKinds.begin() + 1, "control.history");
    history.nodeIds.insert(history.nodeIds.begin() + 1, 31);
    history.operations.insert(history.operations.begin() + 1,
                              { 31, "control.history", "control-eval", "" });
    check (! videowire::compileVisualLayerExecution(history, execution, error)
           && error == "control.history temporal evaluation is not yet connected to native viewport/export image execution",
           "temporal operation retains a dedicated unsupported production diagnostic");

    auto full = direct;
    full.nodeKinds = { "video.source", "video.transform", "video.effects",
                       "video.mask.shape", "video.blend", "video.out" };
    full.nodeIds = { 11, 21, 22, 23, 24, 12 };
    full.operations.clear();
    full.edges.clear();
    full.ports.clear();
    for (size_t i = 0; i < full.nodeIds.size(); ++i)
    {
        full.operations.push_back ({ full.nodeIds[i], full.nodeKinds[i],
            i == 0 ? "source-decode" : "native-gpu", "" });
        if (i + 1 < full.nodeIds.size())
            full.edges.push_back ({ full.nodeIds[i], 0, full.nodeIds[i + 1], 0 });
    }
    FakeLayer preserved;
    check (videowire::executeVisualLayerPlan ({ full }, 7, preserved, error, videohelper::geometry::PlanUse::preview)
           && preserved.scale == 2.0f && preserved.effects != nullptr
           && preserved.maskType == 2 && preserved.opacity == 0.4f,
           "supported full production chain preserves existing rendering inputs");

    FakeLayer legacy;
    auto legacyPlan = direct;
    legacyPlan.nodeKinds = {
        "video.legacy.source", "video.legacy.retime", "video.legacy.transform",
        "video.legacy.effects", "video.out"
    };
    legacyPlan.nodeIds.clear();
    legacyPlan.ports.clear();
    legacyPlan.edges.clear();
    legacyPlan.operations.clear();
    check (videowire::executeVisualLayerPlan ({ legacyPlan }, 7, legacy, error, videohelper::geometry::PlanUse::preview)
           && legacy.scale == 2.0f && legacy.effects != nullptr && legacy.maskType == 2,
           "fixed legacy plan preserves the complete established renderer chain");

    FakeLayer noPlan;
    check (videowire::executeVisualLayerPlan ({}, 7, noPlan, error, videohelper::geometry::PlanUse::preview)
           && noPlan.scale == 2.0f && noPlan.effects != nullptr,
           "snapshots without compiled plans retain legacy production rendering");

    // Dependency-light production-path telemetry coverage.
    auto measuredPlan = directPlan();
    measuredPlan.structuralRevision = 9;
    auto measuredState = std::make_unique<videowire::VisualPlanExecutionState>();
    videowire::VisualTelemetryPlanAdmission measuredAdmission;
    measuredAdmission.clipId = 7; measuredAdmission.structuralRevision = 9;
    measuredAdmission.nodeCount = 2; measuredAdmission.stableNodeIds[0] = 12;
    measuredAdmission.stableNodeIds[1] = 11; measuredAdmission.intermediateImageCount = 1;
    measuredState->telemetry().admitPlans({ measuredAdmission });
    measuredState->admitPlans({ measuredPlan });
    FakeLayer measuredLayerA, measuredLayerB;
    check (videowire::executeVisualLayerPlan({ measuredPlan }, 7, measuredLayerA, error,videohelper::geometry::PlanUse::preview,
                                             nullptr, nullptr, measuredState.get(), 1.0)
           && videowire::executeVisualLayerPlan({ measuredPlan }, 7, measuredLayerB, error,videohelper::geometry::PlanUse::preview,
                                                nullptr, nullptr, measuredState.get(), 2.0),
           "telemetry fixture executes two monotonic evaluations without sleeps");
    auto measured = measuredState->telemetry().snapshot();
    check (measured.graphEvaluations == 2 && measured.graphMeasuredTotalNs > 0
           && measured.planCacheMisses == 1 && measured.planCacheHits == 2
           && measured.planInstalls == 1,
           "graph accounting and plan lowering/install cache hit/miss are measured");
    check (measured.layers.size() == 1 && measured.nodes.size() == 2
           && measured.layers[0].clipId == 7 && measured.layers[0].structuralRevision == 9
           && measured.nodes[0].clipId == 7 && measured.nodes[0].structuralRevision == 9
           && measured.nodes[0].stableNodeId == 11 && measured.nodes[1].stableNodeId == 12,
           "telemetry preserves exact clip/revision/stable-node identities in deterministic order");
    FakeLayer seekLayer;
    check (videowire::executeVisualLayerPlan({ measuredPlan }, 7, seekLayer, error,videohelper::geometry::PlanUse::preview,
                                             nullptr, nullptr, measuredState.get(), 0.5),
           "backward seek re-evaluates the current plan");
    measured = measuredState->telemetry().snapshot();
    check (measured.graphEvaluations == 1 && measured.planCacheMisses == 0
           && measured.planCacheHits == 1 && measured.layers[0].evaluations == 1,
           "backward seek resets per-revision costs and compile cache accounting");
    auto revisedPlan = measuredPlan;
    revisedPlan.structuralRevision = 10;
    measuredAdmission.structuralRevision = 10;
    measuredState->telemetry().admitPlans({ measuredAdmission });
    measuredState->admitPlans({ revisedPlan });
    measured = measuredState->telemetry().snapshot();
    check (measured.graphEvaluations == 0 && measured.layers.size() == 1
           && measured.layers[0].structuralRevision == 10 && measured.layers[0].evaluations == 0
           && measured.nodes.size() == 2 && measured.nodes[0].evaluations == 0,
           "admitting a new revision immediately replaces stale identities with zeroed slots");
    FakeLayer revisedLayer;
    check (videowire::executeVisualLayerPlan({ revisedPlan }, 7, revisedLayer, error,videohelper::geometry::PlanUse::preview,
                                             nullptr, nullptr, measuredState.get(), 3.0),
           "new structural revision installs independently");
    measured = measuredState->telemetry().snapshot();
    check (measured.layers.size() == 1 && measured.layers[0].structuralRevision == 10
           && measured.nodes.size() == 2 && measured.nodes[0].structuralRevision == 10,
           "revision change removes stale identities");

    videowire::VisualPlanTelemetry bounded;
    videowire::VisualTelemetryPlanAdmission exact;
    exact.clipId = 2; exact.structuralRevision = 1; exact.nodeCount = 8;
    exact.intermediateImageCount = 7;
    for (int i = 0; i < 8; ++i) exact.stableNodeIds[(size_t) i] = 8 - i;
    auto duplicate = exact; duplicate.structuralRevision = 0;
    auto stale = exact; stale.clipId = 1; stale.structuralRevision = 1;
    check (bounded.admitPlans({ exact, duplicate, stale }), "exact 8-node/7-image plans admit");
    bounded.recordEvaluation(2, 1, 100, false);
    check(bounded.recordNodeEvaluation(2, 1, 8, 40)
          && bounded.recordNodeEvaluation(2, 1, 8, 60)
          && ! bounded.recordNodeEvaluation(2, 1, 999, 10),
          "node costs accept only measured samples for admitted stable identities");
    bounded.recordEvaluation(99, 1, 100, false);
    const auto boundedSnapshot = bounded.snapshot();
    check (boundedSnapshot.layers.size() == 2 && boundedSnapshot.nodes.size() == 16
           && boundedSnapshot.layers[0].clipId == 1 && boundedSnapshot.layers[1].clipId == 2
           && boundedSnapshot.nodes[8].stableNodeId == 1,
           "duplicate identities normalize and fixed-capacity snapshot ordering is deterministic");

    auto expandedTelemetry = exact; expandedTelemetry.clipId = 3;
    expandedTelemetry.nodeCount = 20;
    expandedTelemetry.intermediateImageCount = 19;
    for (size_t index = 0; index < expandedTelemetry.nodeCount; ++index)
        expandedTelemetry.stableNodeIds[index] = static_cast<int>(index + 1);
    std::string admissionError;
    check (bounded.admitPlans({ exact, expandedTelemetry }, &admissionError)
           && bounded.snapshot().rejectedPlans == 0
           && ! bounded.snapshot().executableNodeSubsetTruncated,
           "telemetry preserves graphs above the former sixteen-operation ceiling without truncation");

    auto largerComposite = composite;
    largerComposite.nodeKinds = { "video.source", "video.transform", "video.effects",
        "video.mask.shape", "visual.shape.rectangle", "visual.draw.shape",
        "video.layer.source", "video.blend", "video.out" };
    largerComposite.nodeIds = { 11, 21, 22, 23, 31, 32, 99, 24, 12 };
    largerComposite.operations = {
        { 11, "video.source", "source-decode", "" },
        { 21, "video.transform", "native-gpu", "" },
        { 22, "video.effects", "native-gpu", "" },
        { 23, "video.mask.shape", "native-gpu", "" },
        { 31, "visual.shape.rectangle", "control-eval", "" },
        { 32, "visual.draw.shape", "native-gpu", "" },
        { 99, "video.layer.source", "source-decode", "" },
        { 24, "video.blend", "native-gpu", "" },
        { 12, "video.out", "native-gpu", "" }
    };
    largerComposite.edges = {
        { 11, 0, 21, 0 }, { 21, 1, 22, 0 }, { 22, 1, 23, 0 },
        { 23, 1, 24, 0 }, { 31, 0, 32, 0 }, { 32, 1, 24, 2 },
        { 99, 0, 24, 3 }, { 24, 1, 12, 0 }
    };
    largerComposite.ports = {
        { 11, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 21, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 21, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 22, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 22, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 23, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 23, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 31, 0, 1, "out", "control", "shape", "unspecified", "unspecified" },
        { 32, 0, 1, "in", "control", "shape", "unspecified", "unspecified" },
        { 32, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 99, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 24, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 24, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 24, 2, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 24, 3, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 12, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    FakeLayer largerLayer;
    const auto largerAdmission = videowire::makeVisualTelemetryAdmission(largerComposite);
    check (largerComposite.operations.size() > 8 && ! largerAdmission.nodesTruncated
           && videowire::executeVisualLayerPlan({ largerComposite }, 7, largerLayer, error, videohelper::geometry::PlanUse::preview),
           "larger exact typed graph remains executable without telemetry truncation");

    const auto beforeHold = measuredState->telemetry().snapshot();
    FakeLayer heldMeasured;
    check (videowire::executeVisualLayerPlan({ revisedPlan }, 7, heldMeasured, error,videohelper::geometry::PlanUse::preview,
                                             nullptr, nullptr, measuredState.get(), 3.0),
           "paused hold executes");
    const auto afterHold = measuredState->telemetry().snapshot();
    check (afterHold.graphEvaluations == beforeHold.graphEvaluations
           && afterHold.graphMeasuredTotalNs == beforeHold.graphMeasuredTotalNs
           && afterHold.graphMeasuredMovingNs == beforeHold.graphMeasuredMovingNs
           && afterHold.planCacheHits == beforeHold.planCacheHits
           && afterHold.planCacheMisses == beforeHold.planCacheMisses
           && afterHold.planInstalls == beforeHold.planInstalls
           && afterHold.lastPlanLoweringNs == beforeHold.lastPlanLoweringNs,
           "paused hold freezes the complete telemetry cache and evaluation contract");

    videowire::VisualPlanTelemetry contended;
    contended.admitPlans({ exact });
    {
        auto heldLock = contended.lockForTesting();
        contended.recordEvaluation(2, 1, 10, false);
        contended.recordNodeEvaluation(2, 1, 8, 10);
        contended.resetOwner(2, 1);
        contended.recordPlanLowering(false, 123, true);
    }
    const auto contentionSnapshot = contended.snapshot();
    check (contentionSnapshot.droppedSamples == 3 && contentionSnapshot.planCacheMisses == 1
           && contentionSnapshot.planInstalls == 1 && contentionSnapshot.lastPlanLoweringNs == 123,
           "contention drops evaluation but preserves first post-reset plan-lowering/install sample");

    videowire::VisualPlanTelemetry staleContended;
    staleContended.admitPlans({ exact });
    {
        auto heldLock = staleContended.lockForTesting();
        staleContended.recordPlanLowering(false, 456, true);
        staleContended.resetOwner(2, 1);
    }
    const auto staleContention = staleContended.snapshot();
    check (staleContention.planCacheMisses == 0 && staleContention.planInstalls == 0
           && staleContention.lastPlanLoweringNs == 0,
           "reset generation discards a deferred pre-reset lowering sample");
    {
        auto heldLock = staleContended.lockForTesting();
        staleContended.recordPlanLowering(false, 0, true);
    }
    const auto zeroDuration = staleContended.snapshot();
    check (zeroDuration.planCacheMisses == 1 && zeroDuration.planInstalls == 1
           && zeroDuration.lastPlanLoweringNs == 0,
           "explicit pending presence preserves a legitimate zero-duration lowering sample");

    videowire::VisualPlanTelemetry mixedGeneration;
    mixedGeneration.admitPlans({ exact });
    {
        auto heldLock = mixedGeneration.lockForTesting();
        mixedGeneration.recordPlanLowering(false, 41, true);
        mixedGeneration.resetOwner(2, 1);
        mixedGeneration.recordPlanLowering(true, 0, false);
    }
    const auto mixedSnapshot = mixedGeneration.snapshot();
    check (mixedSnapshot.planCacheMisses == 0 && mixedSnapshot.planInstalls == 0
           && mixedSnapshot.planCacheHits == 1 && mixedSnapshot.lastPlanLoweringNs == 0,
           "one drain rejects pre-reset pending data while preserving post-reset data");

    auto filteredPlan = directPlan();
    filteredPlan.operations.insert(filteredPlan.operations.begin() + 1,
        { 90, "control.constant", "control-eval", "" });
    filteredPlan.operations.insert(filteredPlan.operations.begin() + 2,
        { 91, "visual.depth.asset", "parked-metadata", "" });
    const auto filteredAdmission = videowire::makeVisualTelemetryAdmission(filteredPlan);
    check (filteredAdmission.executableNodeTotal == 2 && filteredAdmission.nodeCount == 2
           && filteredAdmission.stableNodeIds[0] == 11
           && filteredAdmission.stableNodeIds[1] == 12 && ! filteredAdmission.nodesTruncated,
           "telemetry reports only executable source-decode/native-gpu operations");

    FakeLayer invariantLayer;
    const std::vector<videowire::CompiledVisualLayerPlan> invariantPlans { revisedPlan };
    const auto mapSizeBefore = invariantLayer.genParams.size();
    const auto allocationsBefore = allocations.load(std::memory_order_relaxed);
    check (videowire::executeVisualLayerPlan(invariantPlans, 7, invariantLayer, error,videohelper::geometry::PlanUse::preview,
                                             nullptr, nullptr, measuredState.get(), 4.0),
           "pre-admitted execution renders through fixed lookup slots");
    check (allocations.load(std::memory_order_relaxed) == allocationsBefore
           && invariantLayer.genParams.size() == mapSizeBefore,
           "pre-admitted render lookup performs no allocation or map insertion");

    contended.seedForSaturationTesting(std::numeric_limits<uint64_t>::max());
    contended.recordEvaluation(2, 1, std::numeric_limits<uint64_t>::max(), false);
    contended.recordNodeEvaluation(2, 1, 1, 1);
    contended.recordPlanLowering(true, 0, false);
    contended.recordRuntime(1, 1, 1, 1, videowire::VisualTransportMode::zeroCopy, true);
    contended.recordZeroCopyAllocationBytes(1); contended.recordReadbackCopiedBytes(1);
    contended.recordDrop(videowire::VisualDropReason::noBuffer);
    contended.recordDimensions(2, 2, 1, 1);
    const auto saturated = contended.snapshot();
    check (saturated.graphEvaluations == std::numeric_limits<uint64_t>::max()
           && saturated.graphMeasuredTotalNs == std::numeric_limits<uint64_t>::max()
           && saturated.nodes[0].evaluations == std::numeric_limits<uint64_t>::max()
           && saturated.nodes[0].measuredTotalNs == std::numeric_limits<uint64_t>::max()
           && saturated.planCacheHits == std::numeric_limits<uint64_t>::max()
           && saturated.compositor.count == std::numeric_limits<uint64_t>::max()
           && saturated.zeroCopyFrames == std::numeric_limits<uint64_t>::max()
           && saturated.zeroCopyAllocationBytes == std::numeric_limits<uint64_t>::max()
           && saturated.readbackCopiedBytes == std::numeric_limits<uint64_t>::max()
           && saturated.framesDropped == std::numeric_limits<uint64_t>::max()
           && saturated.dimensionMismatchCount == std::numeric_limits<uint64_t>::max(),
           "all telemetry counters and measured durations saturate");
    const auto telemetryJson = videowire::visualTelemetryJson(boundedSnapshot);
    videowire::VisualPlanTelemetry runtime;
    const auto unseenJson = videowire::visualTelemetryJson(runtime.snapshot());
    check(! unseenJson["compositor"]["available"].get<bool>() && unseenJson["compositor"]["movingNs"].is_null()
          && ! unseenJson["transport"]["available"].get<bool>() && unseenJson["transport"]["mode"].is_null()
          && ! unseenJson["backend"]["available"].get<bool>() && unseenJson["backend"]["current"].is_null()
          && ! unseenJson["resources"]["available"].get<bool>() && unseenJson["resources"]["retainedFramesCurrent"].is_null()
          && ! unseenJson["dimensions"]["available"].get<bool>() && unseenJson["dimensions"]["actualWidth"].is_null(),
          "unseen runtime families emit available false and null values, never production-looking zeroes");
    check(runtime.admitSessionBackend(videowire::VisualBackend::openGL)
          && runtime.admitSessionBackend(videowire::VisualBackend::metal)
          && runtime.recordBackendTransition(videowire::VisualBackend::metal)
          && runtime.recordBackendTransition(videowire::VisualBackend::metal),
          "bounded backend admission happens once and only explicit transitions are events");
    check(runtime.recordDimensions(1920, 1080, 1920, 1080)
          && ! runtime.recordDimensions(0, 1080, 1920, 1080)
          && ! runtime.recordDimensions(std::numeric_limits<int>::max(), 1080, 1920, 1080),
          "dimensions reject non-positive and oversized values before arithmetic");
    check(! runtime.recordRuntime(1, 1, 1, 1, videowire::VisualTransportMode::none, true)
          && runtime.recordRuntime(100, 40, 0, 0, videowire::VisualTransportMode::zeroCopy, true)
          && runtime.recordZeroCopyAllocationBytes(1920ull*1080ull*4ull)
          && runtime.recordReadbackCopiedBytes(123),
          "transport rejects none, zero durations count, and byte semantics use separate typed APIs");
    check(! runtime.snapshot().particles.observed && ! runtime.snapshot().generators.observed,
          "generic runtime transport does not fabricate generator or particle execution");
    check(runtime.recordExecutionObservation(videowire::VisualExecutionKind::generator, 17)
          && runtime.recordExecutionObservation(videowire::VisualExecutionKind::particle, 23),
          "typed production execution hooks independently record generator and particle work");
    runtime.recordResources(3, 7, 4096); runtime.recordResources(0, 0, 0);
    runtime.recordDrop(videowire::VisualDropReason::noBuffer);
    runtime.recordDimensions(1000, 1000, 900, 900);
    runtime.recordDimensions(3840, 2160, 1920, 1080);
    const auto runtimeSnapshot = runtime.snapshot();
    const auto runtimeJson = videowire::visualTelemetryJson(runtimeSnapshot);
    check(runtimeSnapshot.compositor.count == 1 && runtimeSnapshot.presentation.totalNs == 40
          && runtimeSnapshot.particles.count == 1 && runtimeSnapshot.particles.totalNs == 23
          && runtimeSnapshot.generators.count == 1 && runtimeSnapshot.generators.totalNs == 17
          && runtimeSnapshot.zeroCopyFrames == 1 && runtimeSnapshot.zeroCopyAllocationBytes == 1920ull*1080ull*4ull
          && runtimeSnapshot.readbackCopiedBytes == 123 && runtimeSnapshot.framesRendered == 1
          && runtimeSnapshot.framesPresented == 1 && runtimeSnapshot.framesDropped == 1
          && runtimeSnapshot.droppedNoBuffer == 1 && runtimeSnapshot.fallbackCount == 1,
          "runtime schema records typed helper samples without claiming production owners");
    check(runtimeSnapshot.retainedFramesCurrent == 0 && runtimeSnapshot.retainedFramesPeak == 3
          && runtimeSnapshot.intermediateImagesCurrent == 0 && runtimeSnapshot.intermediateImagesPeak == 7
          && runtimeSnapshot.retainedBytesCurrent == 0 && runtimeSnapshot.retainedBytesPeak == 4096
          && runtimeSnapshot.requestedWidth == 3840 && runtimeSnapshot.actualWidth == 1920
          && runtimeSnapshot.dimensionMismatchCount == 2 && runtimeSnapshot.halfResolutionMismatchCount == 1,
          "resource snapshots preserve peaks/releases and dimensions count all mismatches plus exact halves");
    check(runtimeJson["transport"]["mode"] == "zero-copy" && runtimeJson["drops"]["reasons"]["noBuffer"] == 1
          && runtimeJson["resources"]["retainedFramesPeak"] == 3
          && runtimeJson["dimensions"]["halfResolutionMismatchCount"] == 1,
          "observed runtime JSON projects typed transport, drop, resource and dimension fields");
    runtime.resetOwner(0, 0);
    const auto graphReset = runtime.snapshot();
    check(! graphReset.transportObserved && ! graphReset.resourcesObserved && ! graphReset.dimensionsObserved
          && graphReset.retainedFramesCurrent == 0 && graphReset.retainedFramesPeak == 3
          && ! graphReset.particles.observed && ! graphReset.generators.observed
          && graphReset.backendObserved && graphReset.fallbackCount == 1,
          "graph/seek reset clears runtime/current values while preserving resource peaks and session backend");
    runtime.recordExecutionObservation(videowire::VisualExecutionKind::generator, 31);
    check(runtime.snapshot().generators.observed && runtime.snapshot().generators.count == 1
          && runtime.snapshot().generators.totalNs == 31,
          "the first real execution after a requested reset belongs to the new generation");
    runtime.recordBudgetReceipt(expectedBudgetReceipt);
    runtime.resetSession();
    const auto sessionReset = runtime.snapshot();
    const auto sessionResetJson = videowire::visualTelemetryJson(sessionReset);
    check(! sessionReset.backendObserved && sessionReset.fallbackCount == 0
          && ! sessionReset.budgetObserved
          && ! sessionResetJson["budget"]["available"].get<bool>()
          && sessionResetJson["budget"]["backendProfile"].is_null(),
          "session reset clears backend history and the prior budget receipt");
    videowire::VisualPlanExecutionState failedNewSession;
    failedNewSession.setTelemetryOwner(runtime);
    auto failedSessionLimits = mediumLimits;
    failedSessionLimits.operations = 0;
    error.clear();
    check(! failedNewSession.admitPlans(mediumPlans, &error, 64, 64, &failedSessionLimits)
          && ! runtime.snapshot().budgetObserved
          && videowire::visualTelemetryJson(runtime.snapshot())["budget"]["backendProfile"].is_null(),
          "failed admission in a new session cannot restore the prior session budget");
    videowire::VisualPlanTelemetry runtimeContended;
    {
        auto heldLock = runtimeContended.lockForTesting();
        runtimeContended.recordRuntime(1, 1, 1, 1, videowire::VisualTransportMode::readback, true);
        runtimeContended.recordReadbackCopiedBytes(1); runtimeContended.recordZeroCopyAllocationBytes(1);
        runtimeContended.recordDrop(videowire::VisualDropReason::renderFailure);
        runtimeContended.recordResources(1, 1, 1); runtimeContended.recordDimensions(1, 1, 1, 1);
        runtimeContended.admitSessionBackend(videowire::VisualBackend::openGL);
        runtimeContended.recordFailedLowering("contended");
        runtimeContended.recordExecutionObservation(videowire::VisualExecutionKind::particle, 1);
    }
    check(runtimeContended.snapshot().recordingContentionDrops == 9,
          "every runtime recording API is nonblocking and explicitly counts contention drops");
    check (telemetryJson.is_object() && telemetryJson["layers"].is_array()
           && telemetryJson["nodes"].is_array() && telemetryJson["nodes"].size() == 16
           && telemetryJson["nodes"][0]["clipId"] == 1
           && telemetryJson["nodes"][0]["structuralRevision"] == 1
           && telemetryJson["nodes"][0]["stableNodeId"] == 1
           && ! telemetryJson["nodes"][0]["available"].get<bool>()
           && telemetryJson["nodes"][0]["evaluations"].is_null()
           && telemetryJson["nodes"][0]["measuredTotalNs"].is_null()
           && telemetryJson["nodes"][0]["measuredMovingNs"].is_null()
           && telemetryJson["nodes"][15]["available"].get<bool>()
           && telemetryJson["nodes"][15]["evaluations"] == 2
           && telemetryJson["nodes"][15]["measuredTotalNs"] == 100
           && telemetryJson["nodes"][15]["measuredMovingNs"] == 44.0
           && telemetryJson.contains("planCacheHits")
           && telemetryJson.contains("lastPlanLoweringWithinBudget"),
           "viewport telemetry JSON has stable bounded arrays and exact identity/cache fields");

    std::printf ("visual plan executor: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
