#pragma once

#include "depth_payload_schema.h"
#include "diffraction_material_binding_admission.h"
#include "flat_shader_bridge.h"
#include "geometry_core_backend.h"
#include "key_cleanup_payload_schema.h"
#include "common_effect_payload_schema.h"
#include "shader_transition_admission.h"
#include "temporal_resource_contract.h"
#include "../../shared/ColorAovOperationContract.h"
#include "../../shared/ColorTransformOperationContract.h"
#include "../../shared/VisualImportedAnimationOperationContract.h"
#include "../../shared/VisualImportedSceneRenderOperationContract.h"
#include "../../shared/VisualAnimationDeformationEvaluation.h"
#include "../../shared/OpticalFlowOperationContract.h"
#include "../../shared/VisualTemporalOperationContract.h"
#include "../../shared/VisualTemporalSamplingContract.h"

#include "../../shared/SceneAovOperationContract.h"
#include "../../shared/AovInspectionOperationContract.h"

#include "render_snapshot.h"
#include "visual_plan_resource_budget.h"
#include "visual_plan_telemetry.h"
#include "volume_native_renderer.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace videowire
{
inline constexpr int kMaxVisualParticles = 4096;

// Immutable description of one native temporal image operation. Pixels remain
// renderer-owned GPU images; this value transports the exact owner, mode,
// immutable parameters, extent, format and admitted footprint to the shared
// viewport/export product seam.
struct TemporalResourcePass
{
    int clipId = -1;
    int nodeId = 0;
    uint64_t structuralRevision = 0;
    visualtemporaloperation::Payload payload {};
    videotemporal::ImageExtent extent {};
    videotemporal::PixelFormat format = videotemporal::PixelFormat::rgba16f;
    uint32_t historyLength = 2; // compatibility mirror of payload.historyLength
    videotemporal::ResourceFootprint footprint {};
};

using TemporalFeedbackPass = TemporalResourcePass; // source compatibility for older fixtures

struct TemporalSamplingPass
{
    int clipId = -1;
    int nodeId = 0;
    uint64_t structuralRevision = 0;
    visualtemporalsampling::Payload payload {};
    uint32_t width = 0;
    uint32_t height = 0;
    visualtemporalsampling::ResourceFootprint footprint {};
};

// Complete by-value request for one compositor evaluation. The renderer must
// consume it by const reference. Publishing the admitted pass, exact evaluation
// point and lifecycle decision together prevents either product path from
// reconstructing reset/hold/advance state from mutable timeline state.
struct TemporalSamplingExecution
{
    TemporalSamplingPass pass;
    visualtemporalsampling::EvaluationPoint point;
    visualtemporalsampling::Transition transition;
};

struct VisualPlanEvaluationContext
{
    uint64_t helperGeneration = 0;
    bool paused = false;
    visualtemporalsampling::EvaluationMode mode
        = visualtemporalsampling::EvaluationMode::preview;
    uint64_t projectGeneration = 0;
    uint64_t deviceGeneration = 0;
};

struct VisualTriggerConsumption
{
    int count = 0;
    float strongest = 0.0f;
};

class VisualEventTriggerCursor
{
public:
    explicit VisualEventTriggerCursor (bool replayFromStart) : replayFromStart_ (replayFromStart)
    {
        states_.reserve (256);
    }

    VisualTriggerConsumption consume (const std::vector<VisualEventScheduleBinding>& schedules,
                                      int clipId, int nodeId, int portId,
                                      uint64_t sessionRevision, double currentBeat)
    {
        VisualTriggerConsumption result;
        if (! std::isfinite (currentBeat) || sessionRevision == 0) return result;
        auto found = std::find_if (states_.begin(), states_.end(), [=] (const State& state)
            { return state.clipId == clipId && state.nodeId == nodeId && state.portId == portId; });
        if (found == states_.end())
        {
            if (states_.size() >= 256) return result;
            states_.push_back ({ clipId, nodeId, portId, sessionRevision, currentBeat });
            found = states_.end() - 1;
            if (! replayFromStart_) return result;
            found->priorBeat = -std::numeric_limits<double>::infinity();
        }
        else if (found->sessionRevision != sessionRevision || currentBeat < found->priorBeat)
        {
            found->sessionRevision = sessionRevision;
            found->priorBeat = replayFromStart_ ? -std::numeric_limits<double>::infinity() : currentBeat;
            if (! replayFromStart_) return result;
        }
        else if (currentBeat == found->priorBeat)
            return result;

        const double priorBeat = found->priorBeat;
        for (const auto& schedule : schedules)
            if (schedule.clipId == clipId && schedule.nodeId == nodeId
                && schedule.portId == portId && schedule.sessionRevision == sessionRevision)
                for (const auto& trigger : schedule.triggers)
                    if (trigger.timelineBeat > priorBeat && trigger.timelineBeat <= currentBeat)
                    {
                        ++result.count;
                        result.strongest = std::max (result.strongest, trigger.strength);
                    }
        found->priorBeat = currentBeat;
        return result;
    }

    void reset() noexcept { states_.clear(); }

private:
    struct State
    {
        int clipId = 0, nodeId = 0, portId = -1;
        uint64_t sessionRevision = 0;
        double priorBeat = 0.0;
    };
    bool replayFromStart_ = false;
    std::vector<State> states_;
};

// The helper preserves the existing ordered production paths and adds one exact
// typed DAG shape already represented by FrameRenderer. It does not render
// pixels itself. Viewport and export apply the lowered result to LayerDesc and
// continue through the existing native compositor.
struct VisualLayerExecution
{
    uint64_t structuralRevision = 0;
    CompiledVisualDagSchedule dagSchedule;
    int particleNodeId = 0;
    int drawShapeNodeId = 0;
    int drawShapePrimaryNodeId = 0;
    int drawShapeSecondaryNodeId = 0;
    std::optional<visualanimationimport::Request> importedAnimation;
    std::optional<visualimportedscenerender::Request> importedSceneRender;
    std::optional<visualnoteinstancing::Mapping> noteInstanceMapping;
    // Exact image roots consumed by the established compositor, recorded in
    // stable Blend destination-port order rather than inferred from kind or
    // document order. The native Blend contract has primary, overlay and layer
    // inputs at ports 0, 2 and 3 respectively.
    static constexpr size_t kMaxCompositeInputs = 3;
    std::array<int, kMaxCompositeInputs> compositeSourceNodeIds {};
    std::array<int, kMaxCompositeInputs> compositeDestinationPorts {};
    size_t compositeInputCount = 0;
    bool transform = true;
    bool effects = true;
    bool mask = true;
    bool drawShape = false;
    bool drawShapeEllipse = false;
    bool drawShapeHasSecondary = false;
    bool drawShapeSecondaryEllipse = false;
    // 0 = first shape only, 1 = union, 2 = intersection, 3 = subtraction.
    int drawShapeOperation = 0;
    bool pathMatte = false;
    bool pathMatteEllipse = false;
    bool pathMatteHasSecondary = false;
    bool pathMatteSecondaryEllipse = false;
    bool pathMatteInvert = false;
    int pathMatteOperation = 0;
    float pathCx = 0.5f, pathCy = 0.5f, pathW = 1.0f, pathH = 1.0f;
    float path2Cx = 0.5f, path2Cy = 0.5f, path2W = 0.0f, path2H = 0.0f;
    bool feedback = false;
    int temporalNodeId = 0;
    visualtemporaloperation::Payload temporalPayload;
    bool matteApply = false;
    bool depthFog = false;
    int depthEffect = 0;
    bool particles = false;
    bool keyCleanup = false;
    bool colorTransform = false;
    colortransformoperation::Payload colorTransformPayload;
    bool commonEffect = false;
    int commonEffectType = -1;
    std::size_t commonEffectParameterCount = 0;
    std::array<float, commoneffect::kMaximumParameterCount> commonEffectValues {};
    float keyCleanupKeyR = 0.0f, keyCleanupKeyG = 1.0f, keyCleanupKeyB = 0.0f;
    float keyCleanupTolerance = 0.18f, keyCleanupSoftness = 0.10f;
    float keyCleanupDespill = 0.5f;
    float keyCleanupChoke = 0.0f, keyCleanupFeather = 0.0f;
    float keyCleanupEdgeR = 1.0f, keyCleanupEdgeG = 1.0f, keyCleanupEdgeB = 1.0f;
    float keyCleanupEdgeAmount = 0.0f;
    bool keyCleanupMatteView = false;
    bool flatShaderBridge = false;
    ImmutableShaderOperationPlan shaderOperationPlan;
    int particleSeed = 1, particleCount = 512;
    float particleLifetime = 1.8f, particleSize = 4.0f, particleSpeed = 1.0f;
    float particleRed = 0.2f, particleGreen = 0.7f, particleBlue = 1.0f, particleAlpha = 1.0f;
    bool matteInvert = false;
    int matteCombineMode = -1;
    float matteBlack = 0.0f, matteWhite = 1.0f;
    float matteErodeDilate = 0.0f, matteFeather = 0.0f, matteChoke = 0.0f;
    float fogNear = 0.0f, fogFar = 1.0f, fogDensity = 1.0f;
    float fogRed = 1.0f, fogGreen = 1.0f, fogBlue = 1.0f, fogAlpha = 1.0f;
    float depthParam0 = 0.0f, depthParam1 = 0.0f, depthParam2 = 0.0f;
    float depthColorRed = 1.0f, depthColorGreen = 1.0f, depthColorBlue = 1.0f;
    float feedbackDecay = 0.92f, feedbackZoom = 0.99f, feedbackSwirl = 0.0f;
    float shapeCx = 0.5f, shapeCy = 0.5f, shapeW = 1.0f, shapeH = 1.0f;
    float shape2Cx = 0.5f, shape2Cy = 0.5f, shape2W = 0.0f, shape2H = 0.0f;
    float shapeR = 1.0f, shapeG = 1.0f, shapeB = 1.0f, shapeA = 1.0f;
    std::optional<renderpassoutput::Description> colorAovPass;
    std::optional<renderpassoutput::Description> motionAovPass;
    std::optional<sceneaov::Payload> sceneAovPass;
    std::optional<aovinspection::Payload> aovInspectionPass;
    std::optional<TemporalFeedbackPass> temporalFeedbackPass;
    std::optional<TemporalSamplingPass> temporalSamplingPass;
};

inline bool visualPlanUsesTypedMatte (const std::vector<CompiledVisualLayerPlan>& plans,
                                      int clipId)
{
    const auto* plan = findVisualLayerPlan(plans, clipId);
    return plan != nullptr
        && std::find(plan->nodeKinds.begin(), plan->nodeKinds.end(), "visual.matte.apply")
             != plan->nodeKinds.end();
}

inline bool visualPlanDepthBinding(const std::vector<CompiledVisualLayerPlan>& plans, int clipId,
                                   ExecutableDepthPayload& payload, std::string& error)
{
    const auto* plan = findVisualLayerPlan(plans, clipId);
    if (plan == nullptr) return false;
    const auto operation = std::find_if(plan->operations.begin(), plan->operations.end(), [](const auto& op)
        { return op.kind == "visual.depth.asset"; });
    if (operation == plan->operations.end()) return false;
    if (!parseExecutableDepthPayload(operation->payloadXml, payload))
    {
        error = "typed depth asset receipt metadata is malformed or stale";
        return false;
    }
    return true;
}

inline float visualPayloadFloat (const std::string& xml, const char* name, float fallback)
{
    const std::string key = std::string (name) + "=\"";
    const auto begin = xml.find (key);
    if (begin == std::string::npos) return fallback;
    char* end = nullptr;
    const char* number = xml.c_str() + begin + key.size();
    const float value = std::strtof (number, &end);
    return end == number ? fallback : value;
}

inline bool visualBoundedPayloadFloat (const std::string& xml, const char* name,
                                       float fallback, float minimum, float maximum,
                                       float& result)
{
    const std::string key = std::string (name) + "=\"";
    const auto begin = xml.find (key);
    if (begin == std::string::npos)
    {
        result = fallback;
        return true;
    }
    if (xml.find (key, begin + key.size()) != std::string::npos)
        return false;
    const auto valueBegin = begin + key.size();
    const auto valueEnd = xml.find ('"', valueBegin);
    if (valueEnd == std::string::npos)
        return false;
    errno = 0;
    char* parsedEnd = nullptr;
    const float value = std::strtof (xml.c_str() + valueBegin, &parsedEnd);
    if (errno != 0 || parsedEnd != xml.c_str() + valueEnd || ! std::isfinite (value)
        || value < minimum || value > maximum)
        return false;
    result = value;
    return true;
}

inline std::string visualPayloadString(const std::string& xml, const char* name)
{
    const std::string key = std::string(name) + "=\"";
    const auto begin = xml.find(key);
    if (begin == std::string::npos) return {};
    const auto value = begin + key.size();
    const auto end = xml.find('"', value);
    return end == std::string::npos ? std::string{} : xml.substr(value, end - value);
}

inline bool visualPlanSecondaryMatte(const std::vector<CompiledVisualLayerPlan>& plans, int clipId,
                                     const RenderSegment& primary, RenderSegment& secondary)
{
    const auto* plan = findVisualLayerPlan(plans, clipId);
    if (plan == nullptr) return false;
    std::vector<const CompiledVisualOperation*> assets;
    for (const auto& op : plan->operations)
        if (op.kind == "visual.matte.asset") assets.push_back(&op);
    const auto combine = std::find_if(plan->operations.begin(), plan->operations.end(), [](const auto& op)
        { return op.kind == "visual.matte.combine"; });
    if (combine == plan->operations.end() || assets.empty() || assets.size() > 2) return false;
    const auto inputNode = [&](int port) -> int
    {
        int result = 0;
        for (const auto& edge : plan->edges)
            if (edge.toNodeId == combine->nodeId && edge.toPort == port)
            {
                if (result != 0) return 0;
                result = edge.fromNodeId;
            }
        return result;
    };
    const int primaryNodeId = inputNode(0);
    const int secondaryNodeId = inputNode(1);
    if (primaryNodeId == 0 || secondaryNodeId == 0) return false;
    if (primaryNodeId == secondaryNodeId)
    {
        if (assets.size() != 1 || assets[0]->nodeId != primaryNodeId) return false;
        secondary = primary;
        return true;
    }
    const auto secondaryAsset = std::find_if(assets.begin(), assets.end(), [secondaryNodeId](const auto* op)
        { return op->nodeId == secondaryNodeId; });
    if (assets.size() != 2 || secondaryAsset == assets.end()) return false;
    secondary = primary;
    const auto& xml = (*secondaryAsset)->payloadXml;
    secondary.matteAssetId = visualPayloadString(xml, "matteAssetId");
    secondary.matteAssetVersion = visualPayloadString(xml, "matteAssetVersion");
    secondary.matteContentReceipt = visualPayloadString(xml, "contentReceipt");
    secondary.matteState = visualPayloadString(xml, "state");
    secondary.matteCacheKey = visualPayloadString(xml, "cacheKey");
    secondary.matteFramePrefix = visualPayloadString(xml, "framePrefix");
    secondary.matteFrameExtension = visualPayloadString(xml, "frameExtension");
    secondary.matteFirstFrame = (int)visualPayloadFloat(xml, "firstFrame", -1.0f);
    secondary.matteFrameDigits = (int)visualPayloadFloat(xml, "frameDigits", -1.0f);
    secondary.matteFps = visualPayloadFloat(xml, "fps", 0.0f);
    secondary.matteFrames = (int)visualPayloadFloat(xml, "frames", 0.0f);
    return !secondary.matteAssetId.empty() && secondary.matteState == "available";
}

inline bool visualPlanReusesPrimaryMatte(const std::vector<CompiledVisualLayerPlan>& plans, int clipId)
{
    const auto* plan = findVisualLayerPlan(plans, clipId);
    if (plan == nullptr) return false;
    const auto combine = std::find_if(plan->operations.begin(), plan->operations.end(), [](const auto& op)
        { return op.kind == "visual.matte.combine"; });
    if (combine == plan->operations.end()) return false;
    std::vector<int> inputs;
    for (const auto& edge : plan->edges)
        if (edge.toNodeId == combine->nodeId && (edge.toPort == 0 || edge.toPort == 1))
            inputs.push_back(edge.fromNodeId);
    return inputs.size() == 2 && inputs[0] == inputs[1];
}

// Mutable temporal state belongs to one concrete viewport or export run, never
// to the immutable plan. Revisions and backward seeks explicitly reset an owner.
class TemporalSamplingCommitTransaction;

enum class TemporalPublicationFailurePoint : std::uint8_t
{
    none,
    viewportPresentation,
    exportReadback,
    exportCapture,
    exportEncoder
};

inline std::atomic<TemporalPublicationFailurePoint>& temporalPublicationFailurePoint() noexcept
{
    static std::atomic<TemporalPublicationFailurePoint> point {
        TemporalPublicationFailurePoint::none };
    return point;
}

inline void injectTemporalPublicationFailure(TemporalPublicationFailurePoint point) noexcept
{
    temporalPublicationFailurePoint().store(point, std::memory_order_release);
}

inline bool consumeTemporalPublicationFailure(TemporalPublicationFailurePoint point) noexcept
{
    auto expected = point;
    return temporalPublicationFailurePoint().compare_exchange_strong(
        expected, TemporalPublicationFailurePoint::none, std::memory_order_acq_rel);
}

class VisualPlanExecutionState
{
public:
    static constexpr size_t kMaxAdmittedPlans = 256;
    static constexpr size_t kMaxInjectedVolumeBytes = 512u * 1024u * 1024u;
    struct VolumeProduct
    {
        uint64_t structuralRevision = 0;
        std::shared_ptr<const videohelper::volume::AdmittedVolume> admitted;
    };
    struct Owner
    {
        uint64_t structuralRevision = 0;
        double lastTimeSec = 0.0;
        uint64_t evaluationSequence = 0;
        bool hasTime = false;

    };
    struct TemporalRecord
    {
        int clipId = -1;
        uint64_t structuralRevision = 0;
        Owner owner;
        visualtemporalsampling::LifecycleState samplingLifecycle;
    };
    struct TemporalAuthority
    {
        std::mutex mutex;
        uint64_t generation = 1;
        bool alive = true;
        std::vector<TemporalRecord> records;
    };
    struct Slot
    {
        int clipId = -1;
        uint64_t structuralRevision = 0;
        Owner owner;
        VisualLayerExecution execution;
        VolumeProduct volumeProduct;
        visualtemporalsampling::LifecycleState samplingLifecycle;
        bool lowered = false;
    };

    VisualPlanExecutionState() : temporalAuthority_(std::make_shared<TemporalAuthority>()) {}
    ~VisualPlanExecutionState()
    {
        const auto authority = temporalAuthority_;
        std::lock_guard<std::mutex> lock(authority->mutex);
        authority->alive = false;
        ++authority->generation;
    }
    VisualPlanExecutionState(const VisualPlanExecutionState&) = delete;
    VisualPlanExecutionState& operator=(const VisualPlanExecutionState&) = delete;

    bool admitPlans (const std::vector<CompiledVisualLayerPlan>& plans,
                     std::string* diagnostic = nullptr,
                     int canvasWidth = 1920, int canvasHeight = 1080,
                     const VisualBackendResourceLimits* backendLimits = nullptr);
    void reset()
    {
        const auto authority = temporalAuthority_;
        std::lock_guard<std::mutex> lock(authority->mutex);
        for (auto& record : authority->records)
        {
            record.owner = Owner { record.structuralRevision };
            record.samplingLifecycle.reset();
        }
        ++authority->generation;
    }
    void reset (int clipId)
    {
        const auto authority = temporalAuthority_;
        std::lock_guard<std::mutex> lock(authority->mutex);
        const auto found = findTemporalRecord(*authority, clipId);
        if (found != authority->records.end())
        {
            found->owner = Owner { found->structuralRevision };
            found->samplingLifecycle.reset();
            ++authority->generation;
        }
    }

    bool evaluateTemporalSampling (int clipId, uint64_t revision, double timeSec,
                                   uint64_t helperGeneration, bool paused,
                                   visualtemporalsampling::EvaluationMode mode,
                                   visualtemporalsampling::Transition& transition,
                                   std::string& error)
    {
        TemporalSamplingExecution execution;
        if (! prepareTemporalSamplingExecution (clipId, revision, timeSec,
                                                helperGeneration, paused, mode,
                                                execution, error))
            return false;
        transition = execution.transition;
        return true;
    }

    bool prepareTemporalSamplingExecution (
        int clipId, uint64_t revision, double timeSec, uint64_t helperGeneration,
        bool paused, visualtemporalsampling::EvaluationMode mode,
        TemporalSamplingExecution& destination, std::string& error)
    {
        visualtemporalsampling::LifecycleState nextLifecycle;
        if (! prepareTemporalSamplingCandidate (
                clipId, revision, timeSec, helperGeneration, paused, mode,
                destination, nextLifecycle, error))
            return false;
        return commitTemporalSamplingExecution (
            clipId, revision, std::move (nextLifecycle), error);
    }

    bool prepareTemporalSamplingCandidate (
        int clipId, uint64_t revision, double timeSec, uint64_t helperGeneration,
        bool paused, visualtemporalsampling::EvaluationMode mode,
        TemporalSamplingExecution& destination,
        visualtemporalsampling::LifecycleState& nextLifecycle,
        std::string& error,
        const visualtemporalsampling::LifecycleState* predecessor = nullptr) const
    {
        const auto authority = temporalAuthority_;
        TemporalSamplingPass pass;
        {
            std::lock_guard<std::mutex> lock(authority->mutex);
            const auto* slot = findSlot(clipId);
            if (! authority->alive || slot == nullptr || slot->structuralRevision != revision
                || ! slot->execution.temporalSamplingPass.has_value())
            {
                error = "temporal sampling has no matching admitted snapshot";
                return false;
            }
            pass = *slot->execution.temporalSamplingPass;
            if (predecessor != nullptr)
                nextLifecycle = *predecessor;
            else
            {
                const auto record = findTemporalRecord(*authority, clipId);
                if (record == authority->records.end()
                    || record->structuralRevision != revision)
                {
                    error = "temporal sampling has no matching admitted owner snapshot";
                    return false;
                }
                nextLifecycle = record->samplingLifecycle;
            }
        }
        const visualtemporalsampling::EvaluationPoint point {
            clipId, revision, pass.width, pass.height, helperGeneration,
            timeSec, mode, paused
        };
        const auto transition = nextLifecycle.evaluate(point);
        if (transition.action == visualtemporalsampling::TransitionAction::reject)
        {
            error = "temporal sampling lifecycle rejected the evaluation point";
            return false;
        }
        destination = { pass, point, transition };
        error.clear();
        return true;
    }

    bool commitTemporalSamplingExecution (
        int clipId, uint64_t revision,
        visualtemporalsampling::LifecycleState nextLifecycle,
        std::string& error)
    {
        const auto authority = temporalAuthority_;
        std::lock_guard<std::mutex> lock(authority->mutex);
        const auto record = findTemporalRecord(*authority, clipId);
        if (! authority->alive || record == authority->records.end()
            || record->structuralRevision != revision)
        {
            error = "temporal sampling snapshot changed before lifecycle commit";
            return false;
        }
        record->samplingLifecycle = std::move(nextLifecycle);
        ++authority->generation;
        error.clear();
        return true;
    }

    bool canCommitTemporalSamplingExecution (int clipId, uint64_t revision) const
    {
        const auto authority = temporalAuthority_;
        std::lock_guard<std::mutex> lock(authority->mutex);
        const auto record = findTemporalRecord(*authority, clipId);
        return authority->alive && record != authority->records.end()
            && record->structuralRevision == revision;
    }


    const VisualLayerExecution* compiled (int clipId, uint64_t revision) const
    {
        const auto* slot = findSlot(clipId);
        return slot != nullptr && slot->lowered && slot->structuralRevision == revision
            ? &slot->execution : nullptr;
    }
    void setTelemetryOwner (VisualPlanTelemetry& telemetry) { telemetryOwner_ = &telemetry; }
    VisualPlanTelemetry& telemetry() { return *telemetryOwner_; }
    const VisualPlanTelemetry& telemetry() const { return *telemetryOwner_; }
    const VisualPlanResourceReceipt& resourceReceipt() const noexcept { return resourceReceipt_; }
    const VisualPlanBudgetReceipt& budgetReceipt() const noexcept { return budgetReceipt_; }

    bool injectDenseVolumeProduct (
        int clipId, uint64_t revision, const videowire::DenseVolumeDescriptor& descriptor,
        const videohelper::volume::VolumeAdmissionLimits& limits,
        const videohelper::volume::VolumeRendererCapabilities& capabilities,
        std::string& error)
    {
        return injectVolumeProduct (clipId, revision,
            videohelper::volume::admitDenseVolume (descriptor, limits, capabilities, error), error);
    }

    bool injectSparseVolumeProduct (
        int clipId, uint64_t revision, const videowire::SparseVolumeDescriptor& descriptor,
        const videohelper::volume::VolumeAdmissionLimits& limits,
        const videohelper::volume::VolumeRendererCapabilities& capabilities,
        std::string& error)
    {
        return injectVolumeProduct (clipId, revision,
            videohelper::volume::admitSparseVolume (descriptor, limits, capabilities, error), error);
    }

    const VolumeProduct* volumeProduct (int clipId) const
    {
        const auto* slot = findSlot (clipId);
        return slot != nullptr && slot->volumeProduct.admitted != nullptr
            && slot->volumeProduct.structuralRevision == slot->structuralRevision
            ? &slot->volumeProduct : nullptr;
    }

    void clearVolumeProduct (int clipId)
    {
        if (auto* slot = findSlot (clipId)) slot->volumeProduct = {};
    }

    bool renderVolumePreview (
        int clipId, uint64_t revision, uint32_t width, uint32_t height,
        videohelper::volume::NativeVolumeRenderedFrame& output, std::string& error,
        videohelper::volume::NativeVolumeExecutionBackend& backend
            = videohelper::volume::nativeVolumeExecutionBackend()) const
    {
        return renderVolumeProduct (clipId, revision, width, height,
                                    videohelper::volume::NativeVolumeRenderUse::Preview,
                                    output, error, backend);
    }

    bool renderVolumeExport (
        int clipId, uint64_t revision, uint32_t width, uint32_t height,
        videohelper::volume::NativeVolumeRenderedFrame& output, std::string& error,
        videohelper::volume::NativeVolumeExecutionBackend& backend
            = videohelper::volume::nativeVolumeExecutionBackend()) const
    {
        return renderVolumeProduct (clipId, revision, width, height,
                                    videohelper::volume::NativeVolumeRenderUse::Export,
                                    output, error, backend);
    }

    std::optional<Owner> owner (int clipId) const
    {
        const auto authority = temporalAuthority_;
        std::lock_guard<std::mutex> lock(authority->mutex);
        const auto found = findTemporalRecord(*authority, clipId);
        return found == authority->records.end() ? std::optional<Owner>{}
                                                  : std::optional<Owner>{ found->owner };
    }

    bool publishSuccessfulExecution(int clipId, uint64_t revision,
                                    double timeSec, std::string& error)
    {
        const auto authority = temporalAuthority_;
        std::lock_guard<std::mutex> lock(authority->mutex);
        const auto record = findTemporalRecord(*authority, clipId);
        if (! authority->alive || record == authority->records.end()
            || record->structuralRevision != revision)
        {
            error = "visual execution publication snapshot changed";
            return false;
        }
        auto candidate = record->owner;
        if (candidate.structuralRevision != revision
            || (candidate.hasTime && timeSec < candidate.lastTimeSec))
            candidate = Owner { revision };
        if (! candidate.hasTime || timeSec > candidate.lastTimeSec)
            ++candidate.evaluationSequence;
        candidate.lastTimeSec = timeSec;
        candidate.hasTime = true;
        record->owner = candidate;
        ++authority->generation;
        error.clear();
        return true;
    }

    bool temporalLifecycleSnapshot (int clipId, uint64_t revision,
                                    visualtemporalsampling::LifecycleState& lifecycle) const
    {
        const auto authority = temporalAuthority_;
        std::lock_guard<std::mutex> lock(authority->mutex);
        const auto found = findTemporalRecord(*authority, clipId);
        if (! authority->alive || found == authority->records.end()
            || found->structuralRevision != revision)
            return false;
        lifecycle = found->samplingLifecycle;
        return true;
    }

    bool startsSequence (int clipId, uint64_t revision) const
    {
        const auto value = owner(clipId);
        return value.has_value() && value->structuralRevision == revision && ! value->hasTime;
    }

    bool needsReset (int clipId, uint64_t revision, double timeSec) const
    {
        const auto value = owner(clipId);
        return ! value.has_value() || value->structuralRevision != revision
            || (value->hasTime && timeSec < value->lastTimeSec);
    }

    bool isHold (int clipId, uint64_t revision, double timeSec) const
    {
        const auto value = owner(clipId);
        return value.has_value() && value->structuralRevision == revision
            && value->hasTime && timeSec == value->lastTimeSec;
    }

private:
    friend class TemporalSamplingCommitTransaction;
    static std::vector<TemporalRecord>::iterator findTemporalRecord (TemporalAuthority& authority, int clipId)
    {
        return std::find_if(authority.records.begin(), authority.records.end(),
            [clipId](const TemporalRecord& record) { return record.clipId == clipId; });
    }
    static std::vector<TemporalRecord>::const_iterator findTemporalRecord (const TemporalAuthority& authority, int clipId)
    {
        return std::find_if(authority.records.begin(), authority.records.end(),
            [clipId](const TemporalRecord& record) { return record.clipId == clipId; });
    }

    bool renderVolumeProduct (
        int clipId, uint64_t revision, uint32_t width, uint32_t height,
        videohelper::volume::NativeVolumeRenderUse use,
        videohelper::volume::NativeVolumeRenderedFrame& output, std::string& error,
        videohelper::volume::NativeVolumeExecutionBackend& backend) const
    {
        const auto* slot = findSlot (clipId);
        if (slot == nullptr || ! slot->lowered || slot->structuralRevision != revision
            || slot->volumeProduct.structuralRevision != revision
            || slot->volumeProduct.admitted == nullptr)
        {
            error = "native volume render has no matching admitted visual-plan product";
            return false;
        }

        videohelper::volume::NativeVolumeRenderer renderer (backend);
        return use == videohelper::volume::NativeVolumeRenderUse::Preview
            ? renderer.renderPreview (slot->volumeProduct.admitted, width, height,
                                      videohelper::volume::kNativeVolumeGpuCapability,
                                      output, error)
            : renderer.renderExport (slot->volumeProduct.admitted, width, height,
                                     videohelper::volume::kNativeVolumeGpuCapability,
                                     output, error);
    }

    bool injectVolumeProduct (
        int clipId, uint64_t revision,
        std::optional<videohelper::volume::AdmittedVolume> admitted,
        std::string& error)
    {
        auto* slot = findSlot (clipId);
        if (slot == nullptr || ! slot->lowered || slot->structuralRevision != revision)
        {
            error = "volume product injection has no matching admitted visual-plan owner";
            return false;
        }
        if (! admitted.has_value()) return false;

        size_t retainedBytes = 0;
        for (size_t index = 0; index < slotCount_; ++index)
            if (&slots_[index] != slot && slots_[index].volumeProduct.admitted != nullptr)
            {
                const auto bytes = slots_[index].volumeProduct.admitted->bytes().size();
                if (bytes > kMaxInjectedVolumeBytes - retainedBytes)
                {
                    error = "in-memory volume product byte capacity exceeded";
                    return false;
                }
                retainedBytes += bytes;
            }
        if (admitted->bytes().size() > kMaxInjectedVolumeBytes - retainedBytes)
        {
            error = "in-memory volume product byte capacity exceeded";
            return false;
        }

        slot->volumeProduct = {
            revision,
            std::make_shared<const videohelper::volume::AdmittedVolume> (std::move (*admitted))
        };
        error.clear();
        return true;
    }

    Slot* findSlot (int clipId)
    {
        auto it = std::lower_bound(slots_.begin(), slots_.begin() + (ptrdiff_t) slotCount_, clipId,
            [](const Slot& slot, int id) { return slot.clipId < id; });
        return it != slots_.begin() + (ptrdiff_t) slotCount_ && it->clipId == clipId ? &*it : nullptr;
    }
    const Slot* findSlot (int clipId) const
    {
        auto it = std::lower_bound(slots_.begin(), slots_.begin() + (ptrdiff_t) slotCount_, clipId,
            [](const Slot& slot, int id) { return slot.clipId < id; });
        return it != slots_.begin() + (ptrdiff_t) slotCount_ && it->clipId == clipId ? &*it : nullptr;
    }
    std::array<Slot, kMaxAdmittedPlans> slots_ {};
    size_t slotCount_ = 0;
    VisualPlanTelemetry telemetry_;
    VisualPlanTelemetry* telemetryOwner_ = &telemetry_;
    VisualPlanResourceReceipt resourceReceipt_;
    VisualPlanBudgetReceipt budgetReceipt_;
    std::shared_ptr<TemporalAuthority> temporalAuthority_;

};

class TemporalSamplingCommitTransaction final
{
public:
    enum class Status { pending, committed, failed };
    struct BatchState { std::atomic<Status> status { Status::pending }; };

    TemporalSamplingCommitTransaction() : batch_(std::make_shared<BatchState>()) {}
    ~TemporalSamplingCommitTransaction() { discard(); }
    TemporalSamplingCommitTransaction(const TemporalSamplingCommitTransaction&) = delete;
    TemporalSamplingCommitTransaction& operator=(const TemporalSamplingCommitTransaction&) = delete;
    TemporalSamplingCommitTransaction(TemporalSamplingCommitTransaction&& other) noexcept
        : candidates_(std::move(other.candidates_)), bases_(std::move(other.bases_)),
          batch_(std::move(other.batch_)), publications_(std::move(other.publications_)) {}
    TemporalSamplingCommitTransaction& operator=(TemporalSamplingCommitTransaction&& other) noexcept
    {
        if (this != &other)
        {
            discard();
            candidates_ = std::move(other.candidates_);
            bases_ = std::move(other.bases_);
            batch_ = std::move(other.batch_);
            publications_ = std::move(other.publications_);
        }
        return *this;
    }

    void seedFrom (const TemporalSamplingCommitTransaction& predecessor)
    {
        bases_.clear();
        bases_.reserve(predecessor.candidates_.size());
        for (const auto& candidate : predecessor.candidates_)
            bases_.push_back(candidate);
    }

    const visualtemporalsampling::LifecycleState* predecessorLifecycle (
        const VisualPlanExecutionState& state, int clipId, uint64_t revision) const
    {
        const auto authority = state.temporalAuthority_;
        const auto found = std::find_if(bases_.begin(), bases_.end(),
            [&](const Candidate& candidate)
            {
                return sameOwner(candidate.authority, authority) && candidate.clipId == clipId
                    && candidate.revision == revision;
            });
        return found != bases_.end() ? &found->lifecycle : nullptr;
    }

    bool add (VisualPlanExecutionState& state, int clipId, uint64_t revision,
              double timeSec, visualtemporalsampling::LifecycleState lifecycle,
              std::string& error)
    {
        const auto authority = state.temporalAuthority_;
        const auto duplicate = std::find_if(candidates_.begin(), candidates_.end(),
            [&](const Candidate& candidate)
            {
                return sameOwner(candidate.authority, authority) && candidate.clipId == clipId;
            });
        if (duplicate != candidates_.end())
        {
            error = "temporal sampling frame has duplicate lifecycle candidates";
            return false;
        }

        Candidate candidate;
        candidate.authority = authority;
        candidate.clipId = clipId;
        candidate.revision = revision;
        candidate.lifecycle = std::move(lifecycle);
        candidate.batch = batch_;
        const auto predecessor = std::find_if(bases_.begin(), bases_.end(),
            [&](const Candidate& value)
            {
                return sameOwner(value.authority, authority) && value.clipId == clipId
                    && value.revision == revision;
            });
        if (predecessor != bases_.end())
        {
            candidate.owner = predecessor->owner;
            candidate.expectedGeneration = predecessor->expectedGeneration + 1;
            candidate.dependency = predecessor->batch;
        }
        else
        {
            std::lock_guard<std::mutex> lock(authority->mutex);
            const auto record = VisualPlanExecutionState::findTemporalRecord(*authority, clipId);
            if (! authority->alive || record == authority->records.end()
                || record->structuralRevision != revision)
            {
                error = "temporal sampling has no matching admitted owner snapshot";
                return false;
            }
            candidate.owner = record->owner;
            candidate.expectedGeneration = authority->generation;
        }
        if (candidate.owner.structuralRevision != revision
            || (candidate.owner.hasTime && timeSec < candidate.owner.lastTimeSec))
            candidate.owner = VisualPlanExecutionState::Owner { revision };
        if (! candidate.owner.hasTime || timeSec > candidate.owner.lastTimeSec)
            ++candidate.owner.evaluationSequence;
        candidate.owner.lastTimeSec = timeSec;
        candidate.owner.hasTime = true;
        candidates_.push_back(std::move(candidate));
        error.clear();
        return true;
    }

    void addPublication(std::function<bool(std::string&)> validate,
                        std::function<void()> promote,
                        std::function<void()> discard)
    {
        publications_.push_back({ std::move(validate), std::move(promote),
                                  std::move(discard) });
    }

    bool commit (std::string& error)
    {
        struct LockedOwner
        {
            std::shared_ptr<VisualPlanExecutionState::TemporalAuthority> authority;
            std::unique_lock<std::mutex> lock;
        };
        std::vector<std::shared_ptr<VisualPlanExecutionState::TemporalAuthority>> authorities;
        for (const auto& candidate : candidates_)
        {
            const auto authority = candidate.authority.lock();
            if (! authority)
            {
                error = "temporal sampling owner was destroyed before frame publication";
                discard();
                return false;
            }
            if (candidate.dependency
                && candidate.dependency->status.load(std::memory_order_acquire) != Status::committed)
            {
                error = "temporal sampling predecessor was not committed";
                discard();
                return false;
            }
            if (std::none_of(authorities.begin(), authorities.end(),
                    [&](const auto& value) { return value.get() == authority.get(); }))
                authorities.push_back(authority);
        }
        std::sort(authorities.begin(), authorities.end(),
                  [](const auto& a, const auto& b) { return a.get() < b.get(); });
        std::vector<LockedOwner> locks;
        locks.reserve(authorities.size());
        for (auto& authority : authorities)
            locks.push_back({ authority, std::unique_lock<std::mutex>(authority->mutex) });

        for (const auto& candidate : candidates_)
        {
            const auto authority = candidate.authority.lock();
            const auto record = VisualPlanExecutionState::findTemporalRecord(*authority, candidate.clipId);
            if (! authority->alive || authority->generation != candidate.expectedGeneration
                || record == authority->records.end()
                || record->structuralRevision != candidate.revision)
            {
                error = "temporal sampling snapshot changed before frame publication";
                discardUnlocked();
                return false;
            }
        }
        for (const auto& publication : publications_)
            if (! publication.validate(error))
            {
                discardUnlocked();
                return false;
            }
        for (const auto& publication : publications_)
            publication.promote();
        for (const auto& candidate : candidates_)
        {
            const auto authority = candidate.authority.lock();
            auto record = VisualPlanExecutionState::findTemporalRecord(*authority, candidate.clipId);
            record->owner = candidate.owner;
            record->samplingLifecycle = candidate.lifecycle;
        }
        for (auto& authority : authorities)
            ++authority->generation;
        batch_->status.store(Status::committed, std::memory_order_release);
        candidates_.clear();
        bases_.clear();
        publications_.clear();
        error.clear();
        return true;
    }

    void discard() noexcept
    {
        if (batch_ && batch_->status.load(std::memory_order_acquire) == Status::pending)
            batch_->status.store(Status::failed, std::memory_order_release);
        candidates_.clear();
        bases_.clear();
        for (auto& publication : publications_) publication.discard();
        publications_.clear();
    }
    bool empty() const noexcept { return candidates_.empty(); }

private:
    struct Candidate final
    {
        std::weak_ptr<VisualPlanExecutionState::TemporalAuthority> authority;
        int clipId = -1;
        uint64_t revision = 0;
        uint64_t expectedGeneration = 0;
        VisualPlanExecutionState::Owner owner;
        visualtemporalsampling::LifecycleState lifecycle;
        std::shared_ptr<BatchState> dependency;
        std::shared_ptr<BatchState> batch;
    };
    struct Publication final
    {
        std::function<bool(std::string&)> validate;
        std::function<void()> promote;
        std::function<void()> discard;
    };
    static bool sameOwner(
        const std::weak_ptr<VisualPlanExecutionState::TemporalAuthority>& weak,
        const std::shared_ptr<VisualPlanExecutionState::TemporalAuthority>& strong)
    {
        const auto value = weak.lock();
        return value && value.get() == strong.get();
    }
    void discardUnlocked() noexcept
    {
        if (batch_ && batch_->status.load(std::memory_order_acquire) == Status::pending)
            batch_->status.store(Status::failed, std::memory_order_release);
        candidates_.clear();
        bases_.clear();
        for (auto& publication : publications_) publication.discard();
        publications_.clear();
    }
    std::vector<Candidate> candidates_;
    std::vector<Candidate> bases_;
    std::shared_ptr<BatchState> batch_;
    std::vector<Publication> publications_;
};

inline size_t countVisualPlanIntermediateImages (const CompiledVisualLayerPlan& plan)
{
    size_t count = 0;
    for (size_t i = 0; i < plan.edges.size(); ++i)
    {
        const auto& edge = plan.edges[i];
        const auto producer = std::find_if(plan.operations.begin(), plan.operations.end(),
            [&](const CompiledVisualOperation& operation) { return operation.nodeId == edge.fromNodeId; });
        const auto consumer = std::find_if(plan.operations.begin(), plan.operations.end(),
            [&](const CompiledVisualOperation& operation) { return operation.nodeId == edge.toNodeId; });
        if (producer == plan.operations.end() || consumer == plan.operations.end()
            || producer->backendCapability == "control-eval"
            || producer->backendCapability == "parked-metadata")
            continue;
        bool duplicate = false;
        for (size_t prior = 0; prior < i; ++prior)
            duplicate = duplicate || (plan.edges[prior].fromNodeId == edge.fromNodeId
                && plan.edges[prior].fromPort == edge.fromPort);
        if (! duplicate) ++count;
    }
    return count;
}

inline VisualTelemetryPlanAdmission makeVisualTelemetryAdmission (const CompiledVisualLayerPlan& plan)
{
    VisualTelemetryPlanAdmission admission;
    admission.clipId = plan.clipId;
    admission.structuralRevision = plan.structuralRevision;
    for (const auto& operation : plan.operations)
        if (operation.backendCapability == "native-gpu"
            || operation.backendCapability == "source-decode")
        {
            ++admission.executableNodeTotal;
            if (admission.nodeCount < kMaxCompiledNodesPerGraph)
                admission.stableNodeIds[admission.nodeCount++] = operation.nodeId;
        }
    admission.nodesTruncated = admission.executableNodeTotal > admission.nodeCount;
    admission.intermediateImageCount = std::min(
        countVisualPlanIntermediateImages(plan), kMaxIntermediateImagesPerGraph);
    return admission;
}

inline bool compileVisualLayerExecution (const CompiledVisualLayerPlan& plan,
                                         VisualLayerExecution& execution,
                                         std::string& error);

struct SurfaceGraphOperationContract
{
    surfacematerial::OperationKind kind = surfacematerial::OperationKind::Invalid;
    surfacematerial::ValueType resultType = surfacematerial::ValueType::Invalid;
    surfacematerial::InputSemantic semantic = surfacematerial::InputSemantic::Invalid;
    std::array<surfacematerial::ValueType,
               surfacematerial::kMaximumOperationInputs> inputTypes {};
    std::uint8_t inputCount = 0;
};

inline std::uint32_t surfaceValueWidth (surfacematerial::ValueType type) noexcept
{
    switch (type)
    {
        case surfacematerial::ValueType::Scalar:
        case surfacematerial::ValueType::UInt: return 1;
        case surfacematerial::ValueType::Vec2: return 2;
        case surfacematerial::ValueType::Vec3: return 3;
        case surfacematerial::ValueType::Vec4: return 4;
        case surfacematerial::ValueType::Invalid: break;
    }
    return 0;
}

inline std::string_view surfaceDataTypeToken (surfacematerial::ValueType type) noexcept
{
    return type == surfacematerial::ValueType::UInt
        ? std::string_view ("integer") : surfacematerial::token (type);
}

inline bool exactSurfacePort (const CompiledVisualLayerPlan& plan,
                              int nodeId, int port, const char* direction,
                              surfacematerial::ValueType type) noexcept
{
    return std::count_if (plan.ports.begin(), plan.ports.end(), [&](const auto& binding)
        {
            return binding.nodeId == nodeId && binding.port == port
                && binding.direction == direction && binding.carrier == "control"
                && binding.channels == static_cast<int> (surfaceValueWidth (type))
                && binding.dataType == surfaceDataTypeToken (type)
                && binding.pixelFormat == "unspecified"
                && binding.colorSpace == "unspecified";
        }) == 1;
}

inline bool surfaceGraphOperationContract (std::string_view nodeKind,
                                           SurfaceGraphOperationContract& contract)
{
    contract = {};
    const auto typedKind = [](std::string_view operation, std::string_view type)
    {
        std::string value ("visual.surface.");
        value.append (operation);
        value.push_back ('.');
        value.append (type);
        return value;
    };

    for (const auto type : surfacematerial::kValueTypes)
    {
        const auto typeToken = surfacematerial::token (type);
        if (nodeKind == typedKind ("constant", typeToken))
        {
            contract.kind = type == surfacematerial::ValueType::UInt
                ? surfacematerial::OperationKind::UIntConstant
                : surfacematerial::OperationKind::FloatConstant;
            contract.resultType = type;
            return true;
        }
    }

    for (const auto type : std::array { surfacematerial::ValueType::Scalar,
                                        surfacematerial::ValueType::Vec2,
                                        surfacematerial::ValueType::Vec3,
                                        surfacematerial::ValueType::Vec4 })
    {
        const auto typeToken = surfacematerial::token (type);
        for (const auto operation : std::array {
                 surfacematerial::OperationKind::Add,
                 surfacematerial::OperationKind::Subtract,
                 surfacematerial::OperationKind::Multiply,
                 surfacematerial::OperationKind::Divide,
                 surfacematerial::OperationKind::Minimum,
                 surfacematerial::OperationKind::Maximum })
        {
            if (nodeKind != typedKind (surfacematerial::token (operation), typeToken))
                continue;
            contract.kind = operation;
            contract.resultType = type;
            contract.inputTypes[0] = type;
            contract.inputTypes[1] = type;
            contract.inputCount = 2;
            return true;
        }

        for (const auto operation : std::array {
                 surfacematerial::OperationKind::Clamp,
                 surfacematerial::OperationKind::Mix })
        {
            if (nodeKind != typedKind (surfacematerial::token (operation), typeToken))
                continue;
            contract.kind = operation;
            contract.resultType = type;
            contract.inputTypes[0] = type;
            contract.inputTypes[1] = type;
            contract.inputTypes[2] = operation == surfacematerial::OperationKind::Mix
                ? surfacematerial::ValueType::Scalar : type;
            contract.inputCount = 3;
            return true;
        }

        for (const auto operation : std::array {
                 surfacematerial::OperationKind::Absolute,
                 surfacematerial::OperationKind::Power })
        {
            if (nodeKind != typedKind (surfacematerial::token (operation), typeToken))
                continue;
            contract.kind = operation;
            contract.resultType = type;
            contract.inputTypes[0] = type;
            contract.inputCount = 1;
            if (operation == surfacematerial::OperationKind::Power)
            {
                contract.inputTypes[1] = type;
                contract.inputCount = 2;
            }
            return true;
        }
    }

    for (const auto semantic : surfacematerial::kInputSemantics)
    {
        std::string expected ("visual.surface.input.");
        expected.append (surfacematerial::token (semantic));
        if (nodeKind != expected) continue;
        contract.kind = surfacematerial::OperationKind::Input;
        contract.resultType = surfacematerial::inputType (semantic);
        contract.semantic = semantic;
        return true;
    }

    if (nodeKind == "visual.surface.texture-sample-2d")
    {
        contract.kind = surfacematerial::OperationKind::TextureSample2D;
        contract.resultType = surfacematerial::ValueType::Vec4;
        contract.inputTypes[0] = surfacematerial::ValueType::Vec2;
        contract.inputCount = 1;
        return true;
    }

    for (const auto type : std::array { surfacematerial::ValueType::Vec2,
                                        surfacematerial::ValueType::Vec3,
                                        surfacematerial::ValueType::Vec4 })
    {
        const auto typeToken = surfacematerial::token (type);
        for (const auto operation : std::array {
                 surfacematerial::OperationKind::Dot,
                 surfacematerial::OperationKind::Normalize,
                 surfacematerial::OperationKind::Length })
        {
            if (nodeKind != typedKind (surfacematerial::token (operation), typeToken))
                continue;
            contract.kind = operation;
            contract.resultType = operation == surfacematerial::OperationKind::Dot
                    || operation == surfacematerial::OperationKind::Length
                ? surfacematerial::ValueType::Scalar : type;
            contract.inputTypes[0] = type;
            contract.inputCount = 1;
            if (operation == surfacematerial::OperationKind::Dot)
            {
                contract.inputTypes[1] = type;
                contract.inputCount = 2;
            }
            return true;
        }

        if (nodeKind == typedKind ("component", typeToken))
        {
            contract.kind = surfacematerial::OperationKind::Component;
            contract.resultType = surfacematerial::ValueType::Scalar;
            contract.inputTypes[0] = type;
            contract.inputCount = 1;
            return true;
        }
    }

    for (const auto& [kind, operation, type, inputCount] : std::array {
             std::tuple { std::string_view ("visual.surface.compose.vec2"),
                          surfacematerial::OperationKind::ComposeVec2,
                          surfacematerial::ValueType::Vec2, std::uint8_t (2) },
             std::tuple { std::string_view ("visual.surface.compose.vec3"),
                          surfacematerial::OperationKind::ComposeVec3,
                          surfacematerial::ValueType::Vec3, std::uint8_t (3) },
             std::tuple { std::string_view ("visual.surface.compose.vec4"),
                          surfacematerial::OperationKind::ComposeVec4,
                          surfacematerial::ValueType::Vec4, std::uint8_t (4) } })
    {
        if (nodeKind != kind) continue;
        contract.kind = operation;
        contract.resultType = type;
        contract.inputCount = inputCount;
        contract.inputTypes.fill (surfacematerial::ValueType::Scalar);
        return true;
    }
    return false;
}

inline bool compileVisualLayerExecutionOrdered (const CompiledVisualLayerPlan& plan,
                                                VisualLayerExecution& execution,
                                                std::string& error)
{
    execution = {};
    const auto importedSource = std::find_if(plan.operations.begin(), plan.operations.end(),
        [](const auto& operation) { return operation.kind == visualanimationoperation::kSourceNodeKind; });
    const auto importedDeformation = std::find_if(plan.operations.begin(), plan.operations.end(),
        [](const auto& operation) { return operation.kind == visualanimationoperation::kDeformationNodeKind; });
    const auto importedSceneRender = std::find_if(plan.operations.begin(), plan.operations.end(),
        [](const auto& operation) { return operation.kind == visualimportedscenerender::kRenderNodeKind; });
    if (importedSource != plan.operations.end() || importedDeformation != plan.operations.end()
        || importedSceneRender != plan.operations.end())
    {
        const bool animationRoute = importedSource != plan.operations.end()
            && importedDeformation != plan.operations.end()
            && importedSceneRender == plan.operations.end();
        const bool sceneRenderRoute = importedSource != plan.operations.end()
            && importedSceneRender != plan.operations.end();
        const bool composedSceneRoute = importedSource == plan.operations.end()
            && importedDeformation == plan.operations.end()
            && importedSceneRender != plan.operations.end();
        if (! animationRoute && ! sceneRenderRoute && ! composedSceneRoute)
        {
            if (importedSceneRender == plan.operations.end())
                error = "imported animation requires one exact source-to-deformation typed schedule";
            else
                error = "imported scene execution requires one exact source-to-deformation or source-to-Render 3D schedule";
            return false;
        }
        const auto exactControlPort = [&](int nodeId, int port, const char* direction,
                                          const char* dataType)
        {
            return std::count_if(plan.ports.begin(), plan.ports.end(), [&](const auto& binding)
                { return binding.nodeId == nodeId && binding.port == port
                    && binding.direction == direction && binding.carrier == "control"
                    && binding.channels == 1 && binding.dataType == dataType; }) == 1;
        };
        if (composedSceneRoute)
        {
            visualimportedscenerender::Request request;
            if (!visualimportedscenerender::decode(
                    importedSceneRender->payloadXml, request)
                || request.sceneSnapshot == nullptr
                || visualimportedscenerender::encode(request)
                    != importedSceneRender->payloadXml
                || importedSceneRender->backendCapability
                    != visualimportedscenerender::kRenderBackendCapability
                || importedSceneRender->nodeId < 0
                || request.structuralRevision != plan.structuralRevision
                || request.renderStableId
                    != static_cast<std::uint64_t>(importedSceneRender->nodeId) + 1u)
            {
                error = "composed Scene3D execution requires one canonical exact V7 render request";
                return false;
            }
            const auto sceneProducer = std::find_if(
                plan.operations.begin(), plan.operations.end(), [&](const auto& operation)
                {
                    return operation.nodeId >= 0
                        && static_cast<std::uint64_t>(operation.nodeId) + 1u
                            == request.sourceStableId
                        && (operation.kind == "visual.3d.scene.primitive"
                            || operation.kind == "visual.3d.scene.compose");
                });
            const auto sceneEdge = sceneProducer == plan.operations.end()
                ? plan.edges.end()
                : std::find_if(plan.edges.begin(), plan.edges.end(), [&](const auto& edge)
                {
                    return edge.fromNodeId == sceneProducer->nodeId && edge.fromPort == 4
                        && edge.toNodeId == importedSceneRender->nodeId && edge.toPort == 0;
                });
            const std::array<const char*, 8> kinds {
                "visual.3d.transform", "visual.3d.material.pbr",
                "visual.3d.light.directional", "visual.3d.light.environment",
                "visual.3d.camera.perspective", "visual.3d.scene.primitive",
                "visual.3d.scene.compose", "visual.3d.render" };
            bool exactSchedule = plan.nodeIds.size() == plan.operations.size()
                && plan.nodeKinds.size() == plan.operations.size();
            std::set<int> reachable { importedSceneRender->nodeId };
            for (std::size_t pass = 0; pass < plan.operations.size(); ++pass)
                for (const auto& edge : plan.edges)
                    if (reachable.count(edge.toNodeId) != 0)
                        reachable.insert(edge.fromNodeId);
            for (std::size_t index = 0; exactSchedule && index < plan.operations.size(); ++index)
            {
                const auto& operation = plan.operations[index];
                const bool render = &operation == &*importedSceneRender;
                exactSchedule = plan.nodeIds[index] == operation.nodeId
                    && plan.nodeKinds[index] == operation.kind
                    && reachable.count(operation.nodeId) != 0
                    && std::find(kinds.begin(), kinds.end(), operation.kind) != kinds.end()
                    && (render || (operation.backendCapability == "control-eval"
                                   && operation.payloadXml.empty()));
            }
            const auto exactFramePort = [&](int nodeId, int port, const char* direction,
                                             const char* dataType, const char* pixelFormat,
                                             const char* colorSpace)
            {
                return std::count_if(plan.ports.begin(), plan.ports.end(), [&](const auto& binding)
                    { return binding.nodeId == nodeId && binding.port == port
                        && binding.direction == direction && binding.carrier == "frame"
                        && binding.channels == 1 && binding.dataType == dataType
                        && binding.pixelFormat == pixelFormat
                        && binding.colorSpace == colorSpace; }) == 1;
            };
            const auto operationFor = [&](int nodeId)
            {
                return std::find_if(plan.operations.begin(), plan.operations.end(),
                    [&](const auto& operation) { return operation.nodeId == nodeId; });
            };
            const auto nodePortCount = [&](int nodeId)
            {
                return std::count_if(plan.ports.begin(), plan.ports.end(),
                    [&](const auto& binding) { return binding.nodeId == nodeId; });
            };
            const auto canonicalPorts = [&](const auto& operation)
            {
                const auto& kind = operation.kind;
                if (kind == "visual.3d.transform")
                    return nodePortCount(operation.nodeId) == 1
                        && exactControlPort(operation.nodeId, 0, "out", "transform3D");
                if (kind == "visual.3d.material.pbr")
                    return nodePortCount(operation.nodeId) == 1
                        && exactControlPort(operation.nodeId, 0, "out", "material");
                if (kind == "visual.3d.light.environment")
                    return nodePortCount(operation.nodeId) == 1
                        && exactControlPort(operation.nodeId, 0, "out", "light");
                if (kind == "visual.3d.light.directional")
                    return nodePortCount(operation.nodeId) == 2
                        && exactControlPort(operation.nodeId, 0, "in", "transform3D")
                        && exactControlPort(operation.nodeId, 1, "out", "light");
                if (kind == "visual.3d.camera.perspective")
                    return nodePortCount(operation.nodeId) == 2
                        && exactControlPort(operation.nodeId, 0, "in", "transform3D")
                        && exactControlPort(operation.nodeId, 1, "out", "camera");
                if (kind == "visual.3d.scene.primitive")
                    return nodePortCount(operation.nodeId) == 5
                        && exactControlPort(operation.nodeId, 0, "in", "transform3D")
                        && exactControlPort(operation.nodeId, 1, "in", "material")
                        && exactControlPort(operation.nodeId, 2, "in", "light")
                        && exactControlPort(operation.nodeId, 3, "in", "camera")
                        && exactControlPort(operation.nodeId, 4, "out", "scene3D");
                if (kind == "visual.3d.scene.compose")
                    return nodePortCount(operation.nodeId) == 5
                        && exactControlPort(operation.nodeId, 0, "in", "scene3D")
                        && exactControlPort(operation.nodeId, 1, "in", "scene3D")
                        && exactControlPort(operation.nodeId, 2, "in", "scene3D")
                        && exactControlPort(operation.nodeId, 3, "in", "scene3D")
                        && exactControlPort(operation.nodeId, 4, "out", "scene3D");
                return kind == "visual.3d.render"
                    && nodePortCount(operation.nodeId) == 7
                    && exactControlPort(operation.nodeId, 0, "in", "scene3D")
                    && exactFramePort(operation.nodeId, 1, "out", "image", "rgba8", "sRGB")
                    && exactControlPort(operation.nodeId, 2, "in", "material")
                    && exactControlPort(operation.nodeId, 3, "in", "mesh")
                    && exactFramePort(operation.nodeId, 4, "out", "depth", "r32f", "unspecified")
                    && exactControlPort(operation.nodeId, 5, "in", "camera")
                    && exactControlPort(operation.nodeId, 6, "in", "light");
            };
            std::set<std::tuple<int, int, int, int>> uniqueEdges;
            const auto exactEdge = [&](const auto& edge)
            {
                const auto from = operationFor(edge.fromNodeId);
                const auto to = operationFor(edge.toNodeId);
                if (from == plan.operations.end() || to == plan.operations.end()
                    || !uniqueEdges.emplace(edge.fromNodeId, edge.fromPort,
                                            edge.toNodeId, edge.toPort).second)
                    return false;
                const auto& fromKind = from->kind;
                const auto& toKind = to->kind;
                if (fromKind == "visual.3d.transform" && edge.fromPort == 0)
                    return (toKind == "visual.3d.light.directional"
                            || toKind == "visual.3d.camera.perspective"
                            || toKind == "visual.3d.scene.primitive")
                        && edge.toPort == 0;
                if (fromKind == "visual.3d.material.pbr" && edge.fromPort == 0)
                    return toKind == "visual.3d.scene.primitive" && edge.toPort == 1;
                if (((fromKind == "visual.3d.light.directional" && edge.fromPort == 1)
                     || (fromKind == "visual.3d.light.environment" && edge.fromPort == 0)))
                    return (toKind == "visual.3d.scene.primitive" && edge.toPort == 2)
                        || (toKind == "visual.3d.render" && edge.toPort == 6);
                if (fromKind == "visual.3d.camera.perspective" && edge.fromPort == 1)
                    return (toKind == "visual.3d.scene.primitive" && edge.toPort == 3)
                        || (toKind == "visual.3d.render" && edge.toPort == 5);
                if ((fromKind == "visual.3d.scene.primitive"
                     || fromKind == "visual.3d.scene.compose") && edge.fromPort == 4)
                    return (toKind == "visual.3d.scene.compose"
                            && edge.toPort >= 0 && edge.toPort <= 3)
                        || (toKind == "visual.3d.render" && edge.toPort == 0);
                return false;
            };
            const auto incomingCount = [&](int nodeId, int port)
            {
                return std::count_if(plan.edges.begin(), plan.edges.end(), [&](const auto& edge)
                    { return edge.toNodeId == nodeId && edge.toPort == port; });
            };
            std::set<int> operationIds;
            for (const auto& operation : plan.operations)
            {
                exactSchedule = exactSchedule && operationIds.insert(operation.nodeId).second
                    && canonicalPorts(operation);
                if (operation.kind == "visual.3d.light.directional"
                    || operation.kind == "visual.3d.camera.perspective")
                    exactSchedule = exactSchedule && incomingCount(operation.nodeId, 0) == 1;
                else if (operation.kind == "visual.3d.scene.primitive")
                    for (int port = 0; port < 4; ++port)
                        exactSchedule = exactSchedule && incomingCount(operation.nodeId, port) == 1;
                else if (operation.kind == "visual.3d.scene.compose")
                {
                    int connectedScenes = 0;
                    for (int port = 0; port < 4; ++port)
                    {
                        const auto count = incomingCount(operation.nodeId, port);
                        exactSchedule = exactSchedule && count <= 1;
                        connectedScenes += static_cast<int>(count);
                    }
                    exactSchedule = exactSchedule && connectedScenes >= 2;
                }
            }
            for (const auto& port : plan.ports)
                exactSchedule = exactSchedule
                    && operationFor(port.nodeId) != plan.operations.end();
            for (const auto& edge : plan.edges)
                exactSchedule = exactSchedule && exactEdge(edge);
            exactSchedule = exactSchedule
                && incomingCount(importedSceneRender->nodeId, 0) == 1;
            if (sceneProducer != plan.operations.end()
                && sceneProducer->kind == "visual.3d.scene.compose")
                exactSchedule = exactSchedule
                    && incomingCount(importedSceneRender->nodeId, 5) == 1
                    && incomingCount(importedSceneRender->nodeId, 6) == 1;
            if (!exactSchedule || sceneEdge == plan.edges.end())
            {
                error = "composed Scene3D topology does not match its immutable V7 request";
                return false;
            }
            execution.structuralRevision = plan.structuralRevision;
            execution.transform = execution.effects = execution.mask = false;
            execution.importedSceneRender = std::move(request);
            return true;
        }
        if (sceneRenderRoute)
        {
            visualimportedscenerender::Request request;
            const bool canonicalRequest = visualimportedscenerender::decode(
                importedSceneRender->payloadXml, request,
                diffractionmaterialbinding::migrateLegacyV6);
            const bool hasSurfaceMaterial = canonicalRequest && request.material.has_value();
            const bool hasDiffractionMaterial = canonicalRequest
                && request.diffractionMaterial.has_value();
            const bool hasMaterial = hasSurfaceMaterial || hasDiffractionMaterial;
            const bool hasDeformation = canonicalRequest
                && request.deformation.has_value();
            const bool hasCamera = canonicalRequest && request.camera.has_value();
            const auto noteSource = std::find_if(plan.operations.begin(), plan.operations.end(),
                [](const auto& operation) { return operation.kind == "visual.score.note-collection"; });
            const auto noteInstancer = std::find_if(plan.operations.begin(), plan.operations.end(),
                [](const auto& operation) { return operation.kind == "visual.3d.note-instanced-mesh"; });
            const bool hasNoteInstances = canonicalRequest
                && (noteSource != plan.operations.end() || noteInstancer != plan.operations.end());
            visualnoteinstancing::Mapping noteInstanceMapping;
            const bool exactNoteInstanceMapping = !hasNoteInstances
                || (noteInstancer != plan.operations.end()
                    && visualnoteinstancing::decode(noteInstancer->payloadXml,
                                                    noteInstanceMapping));
            const auto cameraOperation = std::find_if(
                plan.operations.begin(), plan.operations.end(), [](const auto& operation)
                { return operation.kind == "visual.3d.camera.perspective"; });
            const auto cameraTransformOperation = std::find_if(
                plan.operations.begin(), plan.operations.end(), [](const auto& operation)
                { return operation.kind == "visual.3d.transform"; });
            if (hasDeformation
                != (importedDeformation != plan.operations.end()))
            {
                error = "imported scene deformation topology does not match its immutable request";
                return false;
            }
            const auto exactImageOutput = std::count_if(
                plan.ports.begin(), plan.ports.end(), [&](const auto& binding)
                {
                    return binding.nodeId == importedSceneRender->nodeId
                        && binding.port == 1 && binding.direction == "out"
                        && binding.carrier == "frame" && binding.channels == 1
                        && binding.dataType == "image"
                        && binding.pixelFormat == "rgba8"
                        && binding.colorSpace == "sRGB";
                }) == 1;
            const auto exactDepthOutput = std::count_if(
                plan.ports.begin(), plan.ports.end(), [&](const auto& binding)
                {
                    return binding.nodeId == importedSceneRender->nodeId
                        && binding.port == 4 && binding.direction == "out"
                        && binding.carrier == "frame" && binding.channels == 1
                        && binding.dataType == "depth"
                        && binding.pixelFormat == "r32f";
                }) == 1;
            const std::array<const char*, 5> sourceTypes {
                "mesh", "skeleton", "morphTargets", "animationClip", "scene3D" };
            bool exactPorts = true;
            for (int port = 0; port < 5; ++port)
                exactPorts = exactPorts && exactControlPort(
                    importedSource->nodeId, port, "out", sourceTypes[(size_t) port]);
            exactPorts = exactPorts && exactControlPort(
                importedSceneRender->nodeId, 0, "in", "scene3D") && exactImageOutput;
            const bool exactMaterialInput = exactControlPort(
                importedSceneRender->nodeId, 2, "in", "material");
            const bool exactDeformationInput = exactControlPort(
                importedSceneRender->nodeId, 3, "in", "mesh");
            const bool exactCameraInput = exactControlPort(
                importedSceneRender->nodeId, 5, "in", "camera");
            const auto renderPortCount = std::count_if(
                plan.ports.begin(), plan.ports.end(), [&](const auto& binding)
                { return binding.nodeId == importedSceneRender->nodeId; });
            exactPorts = exactPorts
                && (! hasMaterial || exactMaterialInput)
                && (! hasDeformation || exactDeformationInput)
                && (! hasCamera || exactCameraInput)
                && (renderPortCount == 2
                    || (renderPortCount == 3 && exactMaterialInput)
                    || (renderPortCount == 4
                        && exactMaterialInput && exactDeformationInput)
                    || (renderPortCount == 5
                        && exactMaterialInput && exactDeformationInput
                        && exactDepthOutput)
                    || (renderPortCount == 6 && exactMaterialInput
                        && exactDeformationInput && exactDepthOutput
                        && exactCameraInput));
            if (hasCamera)
                exactPorts = exactPorts
                    && cameraOperation != plan.operations.end()
                    && cameraTransformOperation != plan.operations.end()
                    && exactControlPort(cameraTransformOperation->nodeId, 0, "out", "transform3D")
                    && exactControlPort(cameraOperation->nodeId, 0, "in", "transform3D")
                    && exactControlPort(cameraOperation->nodeId, 1, "out", "camera");
            if (hasDeformation)
            {
                const std::array<const char*, 4> deformationTypes {
                    "mesh", "skeleton", "morphTargets", "animationClip" };
                for (int port = 0; port < 4; ++port)
                    exactPorts = exactPorts && exactControlPort(
                        importedDeformation->nodeId, port, "in",
                        deformationTypes[(size_t) port]);
                exactPorts = exactPorts && exactControlPort(
                    importedDeformation->nodeId, 4, "out", "mesh");
            }
            if (hasNoteInstances)
                exactPorts = exactPorts
                    && noteSource != plan.operations.end()
                    && noteInstancer != plan.operations.end()
                    && exactControlPort(noteSource->nodeId, 0, "out", "noteCollection")
                    && exactControlPort(noteInstancer->nodeId, 0, "in", "mesh")
                    && exactControlPort(noteInstancer->nodeId, 1, "in", "noteCollection")
                    && exactControlPort(noteInstancer->nodeId, 2, "out", "mesh")
                    && exactDeformationInput;

            const auto materialTerminal = std::find_if(
                plan.operations.begin(), plan.operations.end(), [](const auto& operation)
                {
                    return operation.kind == "visual.surface.material"
                        || operation.kind == "visual.material.diffraction-grating";
                });
            const int materialOutputPort = hasDiffractionMaterial ? 0 : 10;
            const bool exactMaterialOutput = materialTerminal != plan.operations.end()
                && exactControlPort(materialTerminal->nodeId, materialOutputPort,
                                    "out", "material");
            exactPorts = exactPorts && (! hasMaterial || exactMaterialOutput);
            const auto surfaceOperation = [](const auto& operation)
                { return operation.kind.rfind("visual.surface.", 0) == 0; };
            const bool exactKinds = std::all_of(
                plan.operations.begin(), plan.operations.end(), [&](const auto& operation)
                {
                    return (&operation == &*importedSource)
                        || (&operation == &*importedSceneRender)
                        || (hasDeformation && &operation == &*importedDeformation
                            && operation.backendCapability
                                == visualanimationoperation::kDeformationBackendCapability)
                        || (hasCamera
                            && (&operation == &*cameraOperation
                                || &operation == &*cameraTransformOperation)
                            && operation.backendCapability == "control-eval")
                        || (hasDiffractionMaterial
                            && materialTerminal != plan.operations.end()
                            && &operation == &*materialTerminal
                            && operation.backendCapability == "control-eval")
                        || (surfaceOperation(operation)
                            && operation.backendCapability == "control-eval")
                        || (hasNoteInstances
                            && (&operation == &*noteSource || &operation == &*noteInstancer)
                            && operation.backendCapability == "control-eval");
                });
            bool exactSchedule = plan.nodeIds.size() == plan.operations.size()
                && plan.nodeKinds.size() == plan.operations.size();
            for (std::size_t index = 0; exactSchedule && index < plan.operations.size(); ++index)
                exactSchedule = plan.nodeIds[index] == plan.operations[index].nodeId
                    && plan.nodeKinds[index] == plan.operations[index].kind;

            const auto sceneEdge = std::find_if(plan.edges.begin(), plan.edges.end(),
                [&](const auto& edge)
                {
                    return edge.fromNodeId == importedSource->nodeId && edge.fromPort == 4
                        && edge.toNodeId == importedSceneRender->nodeId && edge.toPort == 0;
                });
            const int materialTerminalNode = materialTerminal != plan.operations.end()
                ? materialTerminal->nodeId : 0;
            const auto materialEdge = std::find_if(plan.edges.begin(), plan.edges.end(),
                [&](const auto& edge)
                {
                    return edge.fromNodeId == materialTerminalNode
                        && edge.fromPort == materialOutputPort
                        && edge.toNodeId == importedSceneRender->nodeId && edge.toPort == 2;
                });
            using EdgeIdentity = std::tuple<int, int, int, int>;
            const auto edgeIdentity = [](const auto& edge)
            {
                return EdgeIdentity { edge.fromNodeId, edge.fromPort,
                                      edge.toNodeId, edge.toPort };
            };
            const EdgeIdentity sceneEdgeIdentity {
                importedSource->nodeId, 4, importedSceneRender->nodeId, 0 };
            const EdgeIdentity materialEdgeIdentity {
                materialTerminalNode, materialOutputPort,
                importedSceneRender->nodeId, 2 };
            const EdgeIdentity deformationEdgeIdentity {
                hasDeformation ? importedDeformation->nodeId : 0, 4,
                importedSceneRender->nodeId, 3 };
            const int cameraNodeId = hasCamera && cameraOperation != plan.operations.end()
                ? cameraOperation->nodeId : 0;
            const int cameraTransformNodeId
                = hasCamera && cameraTransformOperation != plan.operations.end()
                ? cameraTransformOperation->nodeId : 0;
            const EdgeIdentity cameraTransformEdgeIdentity {
                cameraTransformNodeId, 0, cameraNodeId, 0 };
            const EdgeIdentity cameraEdgeIdentity {
                cameraNodeId, 1, importedSceneRender->nodeId, 5 };
            bool exactMaterialTopology = ! hasMaterial;
            std::set<EdgeIdentity> exactMaterialEdges;
            if (hasSurfaceMaterial && materialTerminal != plan.operations.end())
            {
                std::map<int, SurfaceGraphOperationContract> contracts;
                bool validMaterialGraph = true;
                for (const auto& operation : plan.operations)
                {
                    if (! surfaceOperation (operation)
                        || operation.nodeId == materialTerminalNode)
                        continue;
                    SurfaceGraphOperationContract contract;
                    if (operation.backendCapability != "control-eval"
                        || ! surfaceGraphOperationContract (operation.kind, contract)
                        || ! contracts.emplace (operation.nodeId, contract).second)
                    {
                        validMaterialGraph = false;
                        continue;
                    }
                    const auto portCount = std::count_if (
                        plan.ports.begin(), plan.ports.end(), [&](const auto& binding)
                        { return binding.nodeId == operation.nodeId; });
                    validMaterialGraph = validMaterialGraph
                        && portCount == static_cast<int> (contract.inputCount) + 1;
                    for (std::uint8_t input = 0; input < contract.inputCount; ++input)
                        validMaterialGraph = validMaterialGraph && exactSurfacePort (
                            plan, operation.nodeId, input, "in", contract.inputTypes[input]);
                    validMaterialGraph = validMaterialGraph && exactSurfacePort (
                        plan, operation.nodeId, contract.inputCount, "out",
                        contract.resultType);
                }

                const auto terminalPortCount = std::count_if (
                    plan.ports.begin(), plan.ports.end(), [&](const auto& binding)
                    { return binding.nodeId == materialTerminalNode; });
                validMaterialGraph = validMaterialGraph
                    && terminalPortCount == static_cast<int> (
                        surfacematerial::kSurfaceOutputCount + 1u)
                    && contracts.size() == request.material->program.operations.size();
                for (const auto output : surfacematerial::kSurfaceOutputs)
                {
                    const auto port = static_cast<int> (output);
                    validMaterialGraph = validMaterialGraph && exactSurfacePort (
                        plan, materialTerminalNode, port, "in",
                        surfacematerial::outputType (output));
                }

                std::map<int, surfacematerial::ValueId> valuesByNode;
                std::unordered_set<int> visiting;
                std::function<bool (int, surfacematerial::ValueId&)> lowerValue;
                lowerValue = [&](int nodeId, surfacematerial::ValueId& value)
                {
                    const auto existing = valuesByNode.find (nodeId);
                    if (existing != valuesByNode.end())
                    {
                        value = existing->second;
                        return true;
                    }
                    const auto contract = contracts.find (nodeId);
                    if (contract == contracts.end() || ! visiting.insert (nodeId).second)
                        return false;

                    std::array<surfacematerial::ValueId,
                               surfacematerial::kMaximumOperationInputs> inputs {};
                    for (std::uint8_t input = 0; input < contract->second.inputCount; ++input)
                    {
                        const auto incoming = std::find_if (
                            plan.edges.begin(), plan.edges.end(), [&](const auto& edge)
                            { return edge.toNodeId == nodeId && edge.toPort == input; });
                        if (incoming == plan.edges.end()
                            || std::count_if (plan.edges.begin(), plan.edges.end(),
                                [&](const auto& edge)
                                { return edge.toNodeId == nodeId && edge.toPort == input; }) != 1
                            || ! lowerValue (incoming->fromNodeId, inputs[input]))
                            return false;
                        const auto producer = contracts.find (incoming->fromNodeId);
                        if (producer == contracts.end()
                            || incoming->fromPort != producer->second.inputCount)
                            return false;
                        exactMaterialEdges.insert (edgeIdentity (*incoming));
                    }

                    const auto expectedId = static_cast<surfacematerial::ValueId> (
                        valuesByNode.size() + 1u);
                    if (expectedId == 0
                        || expectedId > request.material->program.operations.size())
                        return false;
                    const auto& operation = request.material->program.operations[
                        static_cast<std::size_t> (expectedId - 1u)];
                    if (operation.id != expectedId
                        || operation.kind != contract->second.kind
                        || operation.resultType != contract->second.resultType
                        || operation.inputCount != contract->second.inputCount
                        || operation.semantic != contract->second.semantic)
                        return false;
                    for (std::uint8_t input = 0; input < operation.inputCount; ++input)
                        if (operation.inputs[input] != inputs[input]) return false;
                    visiting.erase (nodeId);
                    valuesByNode.emplace (nodeId, expectedId);
                    value = expectedId;
                    return true;
                };

                for (const auto output : surfacematerial::kSurfaceOutputs)
                {
                    const auto port = static_cast<int> (output);
                    const auto incoming = std::find_if (
                        plan.edges.begin(), plan.edges.end(), [&](const auto& edge)
                        { return edge.toNodeId == materialTerminalNode && edge.toPort == port; });
                    surfacematerial::ValueId value = 0;
                    if (incoming == plan.edges.end()
                        || std::count_if (plan.edges.begin(), plan.edges.end(),
                            [&](const auto& edge)
                            { return edge.toNodeId == materialTerminalNode
                                && edge.toPort == port; }) != 1
                        || ! lowerValue (incoming->fromNodeId, value))
                    {
                        validMaterialGraph = false;
                        continue;
                    }
                    const auto producer = contracts.find (incoming->fromNodeId);
                    validMaterialGraph = validMaterialGraph
                        && producer != contracts.end()
                        && incoming->fromPort == producer->second.inputCount
                        && request.material->program.output (output) == value;
                    exactMaterialEdges.insert (edgeIdentity (*incoming));
                }
                exactMaterialTopology = validMaterialGraph
                    && valuesByNode.size() == contracts.size()
                    && std::count_if(plan.operations.begin(), plan.operations.end(),
                        [](const auto& operation)
                        { return operation.kind == "visual.surface.material"; }) == 1;
            }
            else if (hasDiffractionMaterial && materialTerminal != plan.operations.end())
            {
                const auto terminalPortCount = std::count_if(
                    plan.ports.begin(), plan.ports.end(), [&](const auto& binding)
                    { return binding.nodeId == materialTerminalNode; });
                const auto expectedOperationCount = static_cast<std::size_t>(
                    (hasDeformation ? 3 : 2) + (hasCamera ? 2 : 0) + 1);
                exactMaterialTopology = materialTerminal->kind
                        == "visual.material.diffraction-grating"
                    && terminalPortCount == 1
                    && plan.operations.size() == expectedOperationCount
                    && std::count_if(plan.operations.begin(), plan.operations.end(),
                        [](const auto& operation)
                        {
                            return operation.kind
                                == "visual.material.diffraction-grating";
                        }) == 1;
            }

            std::set<EdgeIdentity> expectedEdges { sceneEdgeIdentity };
            if (hasNoteInstances)
            {
                expectedEdges.insert ({ importedSource->nodeId, 0, noteInstancer->nodeId, 0 });
                expectedEdges.insert ({ noteSource->nodeId, 0, noteInstancer->nodeId, 1 });
                expectedEdges.insert ({ noteInstancer->nodeId, 2,
                                        importedSceneRender->nodeId, 3 });
            }
            if (hasDeformation)
            {
                expectedEdges.insert (deformationEdgeIdentity);
                for (int port = 0; port < 4; ++port)
                    expectedEdges.insert ({ importedSource->nodeId, port,
                                            importedDeformation->nodeId, port });
            }
            if (hasMaterial)
            {
                expectedEdges.insert (materialEdgeIdentity);
                expectedEdges.insert (exactMaterialEdges.begin(), exactMaterialEdges.end());
            }
            if (hasCamera && cameraOperation != plan.operations.end()
                && cameraTransformOperation != plan.operations.end())
            {
                expectedEdges.insert(cameraTransformEdgeIdentity);
                expectedEdges.insert(cameraEdgeIdentity);
            }
            const bool exactEdges = sceneEdge != plan.edges.end()
                && (hasMaterial ? materialEdge != plan.edges.end() : materialEdge == plan.edges.end())
                && plan.edges.size() == expectedEdges.size()
                && std::all_of (plan.edges.begin(), plan.edges.end(), [&](const auto& edge)
                    { return expectedEdges.count (edgeIdentity (edge)) == 1; });
            if (! hasMaterial)
                exactMaterialTopology = plan.operations.size()
                        == static_cast<std::size_t>((hasDeformation ? 3 : 2)
                            + (hasCamera ? 2 : 0) + (hasNoteInstances ? 2 : 0))
                    && plan.edges.size()
                        == static_cast<std::size_t>((hasDeformation ? 6 : 1)
                            + (hasCamera ? 2 : 0) + (hasNoteInstances ? 3 : 0))
                    && materialTerminal == plan.operations.end();
            visualanimationimport::Request deformationRequest;
            std::string deformationError;
            const bool exactDeformation = hasDeformation
                ? importedDeformation != plan.operations.end()
                    && visualanimationoperation::decode(
                        importedDeformation->payloadXml,
                        deformationRequest, deformationError)
                    && visualanimationoperation::encode(deformationRequest)
                        == importedDeformation->payloadXml
                    && visualanimationoperation::encode(deformationRequest)
                        == visualanimationoperation::encode(*request.deformation)
                    && deformationRequest.deformationStableId
                        == static_cast<std::uint64_t>(importedDeformation->nodeId) + 1u
                : importedDeformation == plan.operations.end();
            if (std::count_if(plan.operations.begin(), plan.operations.end(),
                    [](const auto& operation)
                    { return operation.kind == visualimportedscenerender::kSourceNodeKind; }) != 1
                || std::count_if(plan.operations.begin(), plan.operations.end(),
                    [](const auto& operation)
                    { return operation.kind == visualimportedscenerender::kRenderNodeKind; }) != 1
                || std::count_if(plan.operations.begin(), plan.operations.end(),
                    [](const auto& operation)
                    { return operation.kind == visualanimationoperation::kDeformationNodeKind; })
                    != (hasDeformation ? 1 : 0)
                || std::count_if(plan.operations.begin(), plan.operations.end(),
                    [](const auto& operation)
                    { return operation.kind == "visual.3d.camera.perspective"; })
                    != (hasCamera ? 1 : 0)
                || std::count_if(plan.operations.begin(), plan.operations.end(),
                    [](const auto& operation)
                    { return operation.kind == "visual.3d.transform"; })
                    != (hasCamera ? 1 : 0)
                || std::count_if(plan.operations.begin(), plan.operations.end(),
                    [](const auto& operation)
                    { return operation.kind == "visual.surface.material"; })
                    != (hasSurfaceMaterial ? 1 : 0)
                || std::count_if(plan.operations.begin(), plan.operations.end(),
                    [](const auto& operation)
                    {
                        return operation.kind
                            == "visual.material.diffraction-grating";
                    }) != (hasDiffractionMaterial ? 1 : 0)
                || std::count_if(plan.operations.begin(), plan.operations.end(),
                    [](const auto& operation)
                    { return operation.kind == "visual.score.note-collection"; })
                    != (hasNoteInstances ? 1 : 0)
                || std::count_if(plan.operations.begin(), plan.operations.end(),
                    [](const auto& operation)
                    { return operation.kind == "visual.3d.note-instanced-mesh"; })
                    != (hasNoteInstances ? 1 : 0)
                || importedSource->backendCapability
                    != visualimportedscenerender::kSourceBackendCapability
                || importedSceneRender->backendCapability
                    != visualimportedscenerender::kRenderBackendCapability
                || ! exactPorts || ! exactKinds || ! exactSchedule || ! exactEdges
                || ! exactMaterialTopology || ! exactDeformation
                || ! exactNoteInstanceMapping || ! canonicalRequest
                || request.sourceStableId
                    != static_cast<std::uint64_t>(importedSource->nodeId) + 1u
                || request.renderStableId
                    != static_cast<std::uint64_t>(importedSceneRender->nodeId) + 1u
                || (hasCamera
                    && (cameraOperation == plan.operations.end()
                        || request.camera->id.value
                            != static_cast<std::uint32_t>(cameraOperation->nodeId) + 1u)))
            {
                error = "imported scene render requires one exact source-to-Render 3D typed schedule";
                return false;
            }
            if (hasDiffractionMaterial)
            {
                std::string admissionError;
                const auto admitted = diffractionmaterialbinding::admit(
                    *request.diffractionMaterial, admissionError);
                if (! admitted)
                {
                    error = admissionError;
                    return false;
                }
            }
            execution.structuralRevision = plan.structuralRevision;
            execution.transform = execution.effects = execution.mask = false;
            execution.importedSceneRender = std::move(request);
            if (hasNoteInstances)
                execution.noteInstanceMapping = noteInstanceMapping;
            return true;
        }

        visualanimationimport::Request request;
        bool exactEdges = plan.edges.size() == 4;
        for (int port = 0; port < 4; ++port)
            exactEdges = exactEdges && std::count_if(plan.edges.begin(), plan.edges.end(), [&](const auto& edge)
                { return edge.fromNodeId == importedSource->nodeId && edge.fromPort == port
                    && edge.toNodeId == importedDeformation->nodeId && edge.toPort == port; }) == 1;
        const std::array<const char*, 4> types { "mesh", "skeleton", "morphTargets", "animationClip" };
        const bool hasSceneOutput = plan.ports.size() == 10;
        bool exactPorts = hasSceneOutput || plan.ports.size() == 9;
        for (int port = 0; port < 4; ++port)
            exactPorts = exactPorts
                && exactControlPort(importedSource->nodeId, port, "out", types[(size_t) port])
                && exactControlPort(importedDeformation->nodeId, port, "in", types[(size_t) port]);
        exactPorts = exactPorts
            && (! hasSceneOutput
                || exactControlPort(importedSource->nodeId, 4, "out", "scene3D"))
            && exactControlPort(importedDeformation->nodeId, 4, "out", "mesh");
        if (plan.operations.size() != 2 || plan.nodeIds.size() != 2
            || plan.nodeKinds.size() != 2 || plan.nodeIds[0] != importedSource->nodeId
            || plan.nodeIds[1] != importedDeformation->nodeId
            || plan.nodeKinds[0] != visualanimationoperation::kSourceNodeKind
            || plan.nodeKinds[1] != visualanimationoperation::kDeformationNodeKind
            || importedSource->backendCapability != visualanimationoperation::kSourceBackendCapability
            || importedDeformation->backendCapability != visualanimationoperation::kDeformationBackendCapability
            || ! exactPorts || ! exactEdges
            || ! visualanimationoperation::decode(importedDeformation->payloadXml, request, error)
            || request.sourceStableId != static_cast<std::uint64_t>(importedSource->nodeId) + 1u
            || request.deformationStableId != static_cast<std::uint64_t>(importedDeformation->nodeId) + 1u)
        {
            if (error.empty()) error = "imported animation requires one exact source-to-deformation typed schedule";
            return false;
        }
        execution.structuralRevision = plan.structuralRevision;
        execution.transform = execution.effects = execution.mask = false;
        execution.importedAnimation = std::move(request);
        return true;
    }
    const auto exactAovPort = [&] (int nodeId, int portId, const char* direction,
                                   renderpassoutput::Output output)
    {
        const auto port = std::find_if (plan.ports.begin(), plan.ports.end(),
            [&] (const auto& candidate)
            { return candidate.nodeId == nodeId && candidate.port == portId; });
        if (port == plan.ports.end() || port->direction != direction
            || port->carrier != "frame" || port->channels
                != (output == renderpassoutput::Output::Motion ? 2 : 1))
            return false;
        switch (output)
        {
            case renderpassoutput::Output::Color:
                return port->dataType == "image" && port->pixelFormat == "rgba16f"
                    && port->colorSpace == "linearSRGB";
            case renderpassoutput::Output::Depth:
                return port->dataType == "depth" && port->pixelFormat == "r32f"
                    && port->colorSpace == "data";
            case renderpassoutput::Output::Normal:
                return port->dataType == "normal" && port->pixelFormat == "rgba16f"
                    && port->colorSpace == "data";
            case renderpassoutput::Output::Motion:
                return port->dataType == "motionVectors" && port->pixelFormat == "rg16f"
                    && port->colorSpace == "data";
            case renderpassoutput::Output::Emission:
                return port->dataType == "emission" && port->pixelFormat == "rgba16f"
                    && port->colorSpace == "linearSRGB";
            case renderpassoutput::Output::Mask:
                return port->dataType == "mask" && port->pixelFormat == "r8"
                    && port->colorSpace == "data";
            case renderpassoutput::Output::MaterialId:
                return port->dataType == "materialId" && port->pixelFormat == "r32uint"
                    && port->colorSpace == "data";
            case renderpassoutput::Output::ObjectId:
                return port->dataType == "objectId" && port->pixelFormat == "r32uint"
                    && port->colorSpace == "data";
            case renderpassoutput::Output::Count: return false;
        }
        return false;
    };
    const auto inspectionOperation = std::find_if (
        plan.operations.begin(), plan.operations.end(), [] (const auto& operation)
        {
            return operation.kind == aovinspection::kDepthOperationKind
                || operation.kind == aovinspection::kNormalOperationKind;
        });
    if (inspectionOperation != plan.operations.end())
    {
        aovinspection::Payload inspectionPayload;
        const bool canonicalInspection = aovinspection::parse (
            inspectionOperation->payloadXml, inspectionPayload)
            && aovinspection::serialize (inspectionPayload) == inspectionOperation->payloadXml;
        const auto sourceOutput = canonicalInspection
            ? aovinspection::output (inspectionPayload.source)
            : renderpassoutput::Output::Count;
        const auto sceneOperation = std::find_if (
            plan.operations.begin(), plan.operations.end(), [&] (const auto& operation)
            { return operation.kind == sceneaov::operationKind (sourceOutput); });
        sceneaov::Payload scenePayload;
        const bool canonicalScene = sceneOperation != plan.operations.end()
            && sceneaov::parse (sceneOperation->payloadXml, scenePayload)
            && sceneaov::serialize (scenePayload) == sceneOperation->payloadXml;
        const int sourcePort = 1 + static_cast<int> (sourceOutput);
        const bool exactEdge = plan.edges.size() == 1
            && plan.edges.front().fromNodeId == (canonicalScene ? sceneOperation->nodeId : 0)
            && plan.edges.front().fromPort == sourcePort
            && plan.edges.front().toNodeId == inspectionOperation->nodeId
            && plan.edges.front().toPort == 0;
        const bool exactSchedule = canonicalInspection && canonicalScene
            && scenePayload.output == sourceOutput
            && scenePayload.extent == inspectionPayload.extent
            && plan.operations.size() == 2 && plan.nodeIds.size() == 2
            && plan.nodeKinds.size() == 2 && plan.ports.size() == 3
            && plan.nodeIds[0] == sceneOperation->nodeId
            && plan.nodeIds[1] == inspectionOperation->nodeId
            && plan.nodeKinds[0] == sceneOperation->kind
            && plan.nodeKinds[1] == inspectionOperation->kind
            && sceneOperation->backendCapability == sceneaov::kBackendCapability
            && inspectionOperation->backendCapability == aovinspection::kBackendCapability
            && exactEdge
            && exactAovPort (sceneOperation->nodeId, sourcePort, "out", sourceOutput)
            && exactAovPort (inspectionOperation->nodeId, 0, "in", sourceOutput)
            && exactAovPort (inspectionOperation->nodeId, 1, "out",
                             renderpassoutput::Output::Color);
        if (! exactSchedule)
        {
            error = "AOV inspection requires one exact scene-AOV-to-inspector native schedule";
            return false;
        }
        execution.structuralRevision = plan.structuralRevision;
        execution.transform = execution.effects = execution.mask = false;
        execution.sceneAovPass = std::move (scenePayload);
        execution.aovInspectionPass = std::move (inspectionPayload);
        return true;
    }

    const auto renderPass = std::find_if (plan.operations.begin(), plan.operations.end(),
        [] (const auto& operation)
        {
            if (operation.kind == coloraov::kOperationKind
                || operation.kind == opticalflowoperation::kOperationKind) return true;
            for (const auto output : sceneaov::kOutputs)
                if (operation.kind == sceneaov::operationKind(output)) return true;
            return false;
        });
    if (renderPass != plan.operations.end())
    {
        coloraov::Payload colorPayload;
        opticalflowoperation::Payload motionPayload;
        sceneaov::Payload scenePayload;
        const bool isColor = coloraov::parse (renderPass->payloadXml, colorPayload)
            && coloraov::serialize(colorPayload) == renderPass->payloadXml;
        const bool isMotion = opticalflowoperation::parse (renderPass->payloadXml, motionPayload)
            && opticalflowoperation::serialize(motionPayload) == renderPass->payloadXml;
        const bool isScene = sceneaov::parse(renderPass->payloadXml, scenePayload)
            && sceneaov::serialize(scenePayload) == renderPass->payloadXml;
        const auto output = isColor ? renderpassoutput::Output::Color
            : isMotion ? renderpassoutput::Output::Motion
            : isScene ? scenePayload.output : renderpassoutput::Output::Count;
        const auto expectedKind = isColor ? coloraov::kOperationKind
            : isMotion ? opticalflowoperation::kOperationKind
            : sceneaov::operationKind(output);
        const auto expectedCapability = isColor ? coloraov::kBackendCapability
            : isMotion ? opticalflowoperation::kBackendCapability
            : sceneaov::kBackendCapability;
        const bool exactOperation = plan.operations.size() == 1 && plan.nodeIds.size() == 1
            && plan.nodeKinds.size() == 1 && plan.nodeIds.front() == renderPass->nodeId
            && plan.nodeKinds.front() == expectedKind && renderPass->kind == expectedKind
            && plan.edges.empty() && plan.ports.size() == 1
            && renderPass->backendCapability == expectedCapability
            && static_cast<int>(isColor) + static_cast<int>(isMotion)
                + static_cast<int>(isScene) == 1;
        const auto port = std::find_if (plan.ports.begin(), plan.ports.end(),
            [&] (const auto& candidate)
            {
                return candidate.nodeId == renderPass->nodeId
                    && candidate.port == 1 + static_cast<int>(output);
            });
        const auto exactDescriptor = [&]()
        {
            if (port == plan.ports.end() || port->direction != "out"
                || port->carrier != "frame" || port->channels != (isMotion ? 2 : 1))
                return false;
            switch (output)
            {
                case renderpassoutput::Output::Color:
                    return port->dataType == "image" && port->pixelFormat == "rgba16f"
                        && port->colorSpace == "linearSRGB";
                case renderpassoutput::Output::Depth:
                    return port->dataType == "depth" && port->pixelFormat == "r32f"
                        && port->colorSpace == "data";
                case renderpassoutput::Output::Normal:
                    return port->dataType == "normal" && port->pixelFormat == "rgba16f"
                        && port->colorSpace == "data";
                case renderpassoutput::Output::Motion:
                    return port->dataType == "motionVectors" && port->pixelFormat == "rg16f"
                        && port->colorSpace == "data";
                case renderpassoutput::Output::Emission:
                    return port->dataType == "emission" && port->pixelFormat == "rgba16f"
                        && port->colorSpace == "linearSRGB";
                case renderpassoutput::Output::Mask:
                    return port->dataType == "mask" && port->pixelFormat == "r8"
                        && port->colorSpace == "data";
                case renderpassoutput::Output::MaterialId:
                    return port->dataType == "materialId" && port->pixelFormat == "r32uint"
                        && port->colorSpace == "data";
                case renderpassoutput::Output::ObjectId:
                    return port->dataType == "objectId" && port->pixelFormat == "r32uint"
                        && port->colorSpace == "data";
                case renderpassoutput::Output::Count: return false;
            }
            return false;
        }();
        if (! exactOperation || ! exactDescriptor)
        {
            error = "Render Passes requires one exact native AOV operation and canonical payload";
            return false;
        }
        execution.structuralRevision = plan.structuralRevision;
        execution.transform = execution.effects = execution.mask = false;
        if (isColor)
            execution.colorAovPass = coloraov::description (colorPayload);
        else if (isMotion)
            execution.motionAovPass = opticalflowoperation::description (motionPayload);
        else
            execution.sceneAovPass = std::move(scenePayload);
        return true;
    }
    const auto isTrackingControl = [] (const std::string& kind)
    {
        return kind == "tracking.point.asset" || kind == "tracking.planar.asset"
            || kind == "tracking.correction" || kind == "tracking.point.apply.transform"
            || kind == "tracking.planar.apply.quad";
    };
    if (std::any_of(plan.operations.begin(), plan.operations.end(), [&] (const auto& operation)
        { return isTrackingControl(operation.kind); }))
    {
        CompiledVisualLayerPlan framePlan = plan;
        std::vector<int> removed;
        for (const auto& operation : plan.operations)
            if (isTrackingControl(operation.kind)) removed.push_back(operation.nodeId);
        const auto removedNode = [&] (int id)
        { return std::find(removed.begin(), removed.end(), id) != removed.end(); };
        framePlan.nodeKinds.clear(); framePlan.nodeIds.clear(); framePlan.operations.clear(); framePlan.ports.clear();
        for (size_t i = 0; i < plan.nodeIds.size(); ++i)
            if (! removedNode(plan.nodeIds[i])) { framePlan.nodeIds.push_back(plan.nodeIds[i]); framePlan.nodeKinds.push_back(plan.nodeKinds[i]); }
        for (const auto& operation : plan.operations)
            if (! removedNode(operation.nodeId)) framePlan.operations.push_back(operation);
        for (const auto& port : plan.ports)
            if (! removedNode(port.nodeId)) framePlan.ports.push_back(port);
        framePlan.edges.erase(std::remove_if(framePlan.edges.begin(), framePlan.edges.end(), [&] (const auto& edge)
            { return removedNode(edge.fromNodeId) || removedNode(edge.toNodeId); }), framePlan.edges.end());
        return compileVisualLayerExecution(framePlan, execution, error);
    }
    execution.structuralRevision = plan.structuralRevision;
    const bool typed = ! plan.nodeIds.empty() || ! plan.edges.empty()
                    || ! plan.ports.empty() || ! plan.operations.empty();
    std::vector<std::string> kinds;
    std::vector<int> ids;
    if (typed)
    {
        if (plan.operations.empty() || plan.operations.size() != plan.nodeIds.size())
        {
            error = "typed visual layer plan has no executable operation order";
            return false;
        }
        kinds.reserve (plan.operations.size());
        ids.reserve (plan.operations.size());
        const auto sampling = std::find_if(plan.operations.begin(), plan.operations.end(),
            [] (const auto& operation)
            {
                visualtemporalsampling::Mode mode {};
                return visualtemporalsampling::modeForKind(operation.kind, mode);
            });
        if (sampling != plan.operations.end())
        {
            if (sampling->kind != visualtemporalsampling::kMotionBlurKind
                || sampling->backendCapability != "temporal-sampling"
                || std::count_if(plan.operations.begin(), plan.operations.end(), [] (const auto& operation)
                   { visualtemporalsampling::Mode mode {}; return visualtemporalsampling::modeForKind(operation.kind, mode); }) != 1)
            {
                error = "temporal sampling requires one motion-blur operation with sampling-only capability";
                return false;
            }
            visualtemporalsampling::Payload payload;
            if (! visualtemporalsampling::parseForKind(sampling->kind, sampling->payloadXml, payload, &error))
                return false;
            const auto exactPort = [&] (int index, const char* direction, const char* dataType,
                                        const char* format, const char* colorSpace)
            {
                return std::count_if(plan.ports.begin(), plan.ports.end(), [&] (const auto& port)
                {
                    return port.nodeId == sampling->nodeId && port.port == index
                        && port.direction == direction && port.carrier == "frame"
                        && port.dataType == dataType && port.pixelFormat == format
                        && port.colorSpace == colorSpace;
                }) == 1;
            };
            if (! exactPort(0, "in", "image", "rgba8", "sRGB")
                || ! exactPort(1, "in", "motionVectors", "rg16f", "unspecified")
                || ! exactPort(2, "out", "image", "rgba8", "sRGB"))
            {
                error = "visual.motion-blur requires exact image, motion-vector, and image ports";
                return false;
            }
            TemporalSamplingPass pass;
            pass.clipId = plan.clipId;
            pass.nodeId = sampling->nodeId;
            pass.structuralRevision = plan.structuralRevision;
            pass.payload = payload;
            execution.transform = execution.effects = execution.mask = false;
            execution.temporalSamplingPass = pass;
            return true;
        }
        for (const auto& operation : plan.operations)
        {
            if (operation.kind == "control.history")
            {
                error = "control.history temporal evaluation is not yet connected to native viewport/export image execution";
                return false;
            }
            if (isParameterlessShapeBooleanKind(operation.kind)
                && ! operation.payloadXml.empty())
            {
                error = "visual Shape Boolean operation payload must be empty";
                return false;
            }
            // This slice admits only passes which the existing compositor runs on
            // the native GPU (source decode remains the established source path).
            const bool shapeControl = (operation.kind == "visual.shape.rectangle"
                || operation.kind == "visual.shape.ellipse"
                || operation.kind == "visual.shape.union"
                || operation.kind == "visual.shape.intersection"
                || operation.kind == "visual.shape.subtract")
                && operation.backendCapability == "control-eval";
            if (! shapeControl && operation.backendCapability != "native-gpu"
                && operation.backendCapability != "source-decode"
                && ! isBoundedNativeCompositorDag(plan.nodeKinds))
            {
                error = "visual layer plan requires unsupported execution capability";
                return false;
            }
            kinds.push_back (operation.kind);
            ids.push_back (operation.nodeId);
        }

        const auto colorTransform = std::find_if(
            plan.operations.begin(), plan.operations.end(), [](const auto& operation)
            { return operation.kind == colortransformoperation::kOperationKind; });
        if (colorTransform != plan.operations.end())
        {
            const size_t index = static_cast<size_t>(
                std::distance(plan.operations.begin(), colorTransform));
            const auto edge = [&](size_t from, int fromPort, size_t to, int toPort)
            {
                return std::any_of(plan.edges.begin(), plan.edges.end(), [&](const auto& value)
                {
                    return value.fromNodeId == ids[from] && value.fromPort == fromPort
                        && value.toNodeId == ids[to] && value.toPort == toPort;
                });
            };
            const bool exact = kinds.size() == 3 && index == 1
                && kinds[0] == "video.source" && kinds[2] == "video.out"
                && plan.edges.size() == 2 && edge(0, 0, 1, 0) && edge(1, 1, 2, 0)
                && colorTransform->backendCapability
                    == colortransformoperation::kBackendCapability;
            if (! exact)
            {
                error = "visual.color.transform requires exactly Source -> Color Transform -> Output";
                return false;
            }
            const auto exactPort = [&](int nodeId, int portIndex, const char* direction)
                -> const CompiledVisualPortBinding*
            {
                const CompiledVisualPortBinding* result = nullptr;
                for (const auto& port : plan.ports)
                    if (port.nodeId == nodeId && port.port == portIndex)
                    {
                        if (result != nullptr) return nullptr;
                        result = &port;
                    }
                if (result == nullptr || result->channels != 1
                    || result->direction != direction || result->carrier != "frame"
                    || result->dataType != "image" || result->pixelFormat != "rgba8"
                    || result->colorSpace != "sRGB")
                    return nullptr;
                return result;
            };
            if (exactPort(ids[0], 0, "out") == nullptr
                || exactPort(colorTransform->nodeId, 0, "in") == nullptr
                || exactPort(colorTransform->nodeId, 1, "out") == nullptr
                || exactPort(ids[2], 0, "in") == nullptr)
            {
                error = "visual.color.transform requires exact RGBA8 sRGB Frame<Image> ports";
                return false;
            }
            colortransformoperation::Payload payload;
            if (! colortransformoperation::parse(colorTransform->payloadXml, payload)
                || payload.inputFormat != colortransform::PixelFormat::RGBA8
                || payload.inputColorSpace != colortransform::ColorSpace::SRGB
                || payload.inputTransfer != colortransform::TransferFunction::SRGB
                || payload.inputAlpha != colortransform::AlphaMode::Straight
                || payload.outputFormat != colortransform::PixelFormat::RGBA8
                || payload.outputColorSpace != colortransform::ColorSpace::SRGB
                || payload.outputTransfer != colortransform::TransferFunction::SRGB
                || payload.outputAlpha != colortransform::AlphaMode::Straight
                || payload.workingColorSpace != colortransform::ColorSpace::LinearSRGB
                || payload.workingFormat != colortransform::PixelFormat::RGBA16F
                || payload.outputIntent != colortransform::OutputIntent::SdrDisplay)
            {
                error = "visual.color.transform has an unsupported immutable payload";
                return false;
            }
            colortransform::AdmissionFailure failure = colortransform::AdmissionFailure::None;
            const auto admitted = colortransform::admit(
                colortransformoperation::descriptionForExtent(payload, { 1, 1 }),
                colortransform::BackendCapability::NativeGpu, failure);
            if (! admitted)
            {
                error = "visual.color.transform admission failed: "
                    + std::string(colortransform::token(failure));
                return false;
            }
            execution.transform = execution.effects = execution.mask = false;
            execution.colorTransform = true;
            execution.colorTransformPayload = payload;
            return true;
        }

        const auto keyCleanup = std::find_if(plan.operations.begin(), plan.operations.end(), [](const auto& op)
            { return op.kind == "visual.key.cleanup"; });
        if (keyCleanup != plan.operations.end())
        {
            const size_t index = static_cast<size_t>(std::distance(plan.operations.begin(), keyCleanup));
            const auto edge = [&](size_t from, int fromPort, size_t to, int toPort)
            {
                return std::any_of(plan.edges.begin(), plan.edges.end(), [&](const auto& value)
                    { return value.fromNodeId == ids[from] && value.fromPort == fromPort
                        && value.toNodeId == ids[to] && value.toPort == toPort; });
            };
            const bool exact = kinds.size() == 3 && index == 1
                && kinds[0] == "video.source" && kinds[2] == "video.out"
                && plan.edges.size() == 2 && edge(0, 0, 1, 0) && edge(1, 1, 2, 0);
            if (!exact)
            {
                error = "visual.key.cleanup requires exactly Source -> Key Cleanup -> Output";
                return false;
            }
            std::vector<const CompiledVisualPortBinding*> ports;
            for (const auto& port : plan.ports)
                if (port.nodeId == keyCleanup->nodeId) ports.push_back(&port);
            const auto exactPort = [&](int indexToFind, const char* direction)
            {
                return std::count_if(ports.begin(), ports.end(), [&](const auto* port)
                    { return port->port == indexToFind && port->channels == 1
                        && port->direction == direction && port->carrier == "frame"
                        && port->dataType == "image"; }) == 1;
            };
            if (ports.size() != 2 || !exactPort(0, "in") || !exactPort(1, "out"))
            {
                error = "visual.key.cleanup requires exact Frame<Image> input/output descriptors";
                return false;
            }
            ExecutableKeyCleanupPayload payload;
            KeyCleanupPayloadFailure failure = KeyCleanupPayloadFailure::none;
            if (!parseExecutableKeyCleanupPayload(keyCleanup->payloadXml, payload, &failure))
            {
                error = failure == KeyCleanupPayloadFailure::productionBackendUnavailable
                    ? "visual.key.cleanup requests cleanup controls unavailable on the production native-GPU backend"
                    : "visual.key.cleanup has a malformed or out-of-bounds immutable payload";
                return false;
            }
            execution.transform = execution.effects = execution.mask = false;
            execution.keyCleanup = true;
            execution.keyCleanupKeyR = payload.keyR;
            execution.keyCleanupKeyG = payload.keyG;
            execution.keyCleanupKeyB = payload.keyB;
            execution.keyCleanupTolerance = payload.tolerance;
            execution.keyCleanupSoftness = payload.softness;
            execution.keyCleanupDespill = payload.despill;
            execution.keyCleanupChoke = payload.choke;
            execution.keyCleanupFeather = payload.feather;
            execution.keyCleanupEdgeR = payload.edgeRed;
            execution.keyCleanupEdgeG = payload.edgeGreen;
            execution.keyCleanupEdgeB = payload.edgeBlue;
            execution.keyCleanupEdgeAmount = payload.edgeAmount;
            execution.keyCleanupMatteView = payload.matteView;
            return true;
        }

        const auto commonEffect = std::find_if(plan.operations.begin(), plan.operations.end(), [](const auto& op)
        {
            return commoneffect::kindForGraphName(op.kind).has_value();
        });
        // Direct chains retain their established exact admission. A common
        // effect on the primary branch of an ordinary native composite is
        // validated below together with the exact Blend-port topology.
        if (commonEffect != plan.operations.end() && ! isBoundedNativeCompositorDag(kinds))
        {
            const size_t index = static_cast<size_t>(std::distance(plan.operations.begin(), commonEffect));
            const auto edge = [&](size_t from, int fromPort, size_t to, int toPort)
            {
                return std::any_of(plan.edges.begin(), plan.edges.end(), [&](const auto& value)
                    { return value.fromNodeId == ids[from] && value.fromPort == fromPort
                        && value.toNodeId == ids[to] && value.toPort == toPort; });
            };
            const bool exact = kinds.size() == 3 && index == 1
                && kinds[0] == "video.source" && kinds[2] == "video.out"
                && plan.edges.size() == 2 && edge(0, 0, 1, 0) && edge(1, 1, 2, 0);
            if (!exact || commonEffect->backendCapability != "native-gpu")
            {
                error = commonEffect->kind + " has unsupported production topology";
                return false;
            }
            std::vector<const CompiledVisualPortBinding*> ports;
            for (const auto& port : plan.ports)
                if (port.nodeId == commonEffect->nodeId) ports.push_back(&port);
            const auto findExactPort = [&](int nodeId, int portIndex)
                -> const CompiledVisualPortBinding*
            {
                const CompiledVisualPortBinding* result = nullptr;
                for (const auto& port : plan.ports)
                    if (port.nodeId == nodeId && port.port == portIndex)
                    {
                        if (result != nullptr) return nullptr;
                        result = &port;
                    }
                return result;
            };
            const auto exactImagePort = [](const CompiledVisualPortBinding* port,
                                           const char* direction)
            {
                return port != nullptr && port->channels == 1
                    && port->direction == direction && port->carrier == "frame"
                    && port->dataType == "image" && ! port->pixelFormat.empty()
                    && port->pixelFormat != "unspecified" && ! port->colorSpace.empty()
                    && port->colorSpace != "unspecified";
            };
            const auto* sourcePort = findExactPort(ids[0], 0);
            const auto* inputPort = findExactPort(commonEffect->nodeId, 0);
            const auto* outputPort = findExactPort(commonEffect->nodeId, 1);
            const auto* sinkPort = findExactPort(ids[2], 0);
            if (ports.size() != 2 || ! exactImagePort(sourcePort, "out")
                || ! exactImagePort(inputPort, "in")
                || ! exactImagePort(outputPort, "out")
                || ! exactImagePort(sinkPort, "in")
                || sourcePort->pixelFormat != inputPort->pixelFormat
                || sourcePort->colorSpace != inputPort->colorSpace
                || inputPort->pixelFormat != outputPort->pixelFormat
                || inputPort->colorSpace != outputPort->colorSpace
                || outputPort->pixelFormat != sinkPort->pixelFormat
                || outputPort->colorSpace != sinkPort->colorSpace)
            {
                error = commonEffect->kind + " requires exact Frame<Image> input/output descriptors";
                return false;
            }
            ExecutableCommonEffectPayload payload;
            if (!parseExecutableCommonEffectPayload(commonEffect->payloadXml, payload)
                || commonEffect->kind != commoneffect::graphKindName(payload.kind))
            {
                error = commonEffect->kind
                    + " has a malformed, unsupported, or out-of-bounds immutable payload";
                return false;
            }
            execution.transform = execution.effects = execution.mask = false;
            execution.commonEffect = true;
            execution.commonEffectType = payload.rendererEffectType;
            execution.commonEffectParameterCount = payload.parameterCount;
            execution.commonEffectValues = payload.parameters;
            return true;
        }

        const bool hasShaderOperation = std::any_of(plan.operations.begin(), plan.operations.end(), [](const auto& op)
            { return op.kind == "visual.shader.generator" || op.kind == "visual.shader.filter"
                || op.kind == "visual.shader.custom"
                || op.kind == shadertransition::operationKind; });
        if (hasShaderOperation && ! isBoundedNativeCompositorDag(kinds))
        {
            const auto operationIndex = [&](int nodeId) -> int
            {
                for (std::size_t i = 0; i < plan.operations.size(); ++i)
                    if (plan.operations[i].nodeId == nodeId) return static_cast<int>(i);
                return -1;
            };
            const auto incomingEdge = [&](int nodeId, int port) -> const CompiledVisualEdgeBinding*
            {
                const CompiledVisualEdgeBinding* result = nullptr;
                for (const auto& edge : plan.edges)
                    if (edge.toNodeId == nodeId && edge.toPort == port)
                    {
                        if (result != nullptr) return nullptr;
                        result = &edge;
                    }
                return result;
            };
            const auto outgoingEdge = [&](int nodeId, int port) -> const CompiledVisualEdgeBinding*
            {
                const CompiledVisualEdgeBinding* result = nullptr;
                for (const auto& edge : plan.edges)
                    if (edge.fromNodeId == nodeId && edge.fromPort == port)
                    {
                        if (result != nullptr) return nullptr;
                        result = &edge;
                    }
                return result;
            };
            const auto exactImagePort = [&](int nodeId, int portIndex,
                                            const char* direction)
            {
                const CompiledVisualPortBinding* result = nullptr;
                for (const auto& port : plan.ports)
                {
                    if (port.nodeId != nodeId || port.port != portIndex
                        || port.direction != direction)
                        continue;
                    if (result != nullptr) return static_cast<const CompiledVisualPortBinding*>(nullptr);
                    result = &port;
                }
                return result != nullptr && result->channels == 1
                    && result->carrier == "frame" && result->dataType == "image"
                    && !result->pixelFormat.empty() && result->pixelFormat != "unspecified"
                    && !result->colorSpace.empty() && result->colorSpace != "unspecified"
                    ? result : nullptr;
            };
            const auto sameImageDescriptor = [](const CompiledVisualPortBinding* left,
                                                const CompiledVisualPortBinding* right)
            {
                return left != nullptr && right != nullptr
                    && left->pixelFormat == right->pixelFormat
                    && left->colorSpace == right->colorSpace;
            };
            const auto portCount = [&](int nodeId)
            {
                return std::count_if(plan.ports.begin(), plan.ports.end(),
                    [nodeId](const auto& port) { return port.nodeId == nodeId; });
            };
            const auto exactFilterPorts = [&](int nodeId)
            {
                const auto* input = exactImagePort(nodeId, 0, "in");
                const auto* outputPort = exactImagePort(nodeId, 1, "out");
                return portCount(nodeId) == 2 && sameImageDescriptor(input, outputPort);
            };
            const auto exactTransitionPorts = [&](int nodeId)
            {
                const auto* start = exactImagePort(nodeId, 0, "in");
                const auto* end = exactImagePort(nodeId, 1, "in");
                const auto* outputPort = exactImagePort(nodeId, 2, "out");
                return portCount(nodeId) == 3 && sameImageDescriptor(start, end)
                    && sameImageDescriptor(start, outputPort);
            };

            const CompiledVisualOperation* output = nullptr;
            for (const auto& op : plan.operations) if (op.kind == "video.out") output = &op;
            if (output == nullptr || output->backendCapability != "native-gpu")
            { error = "shader operation plan requires one native Output"; return false; }
            const auto* finalEdge = incomingEdge(output->nodeId, 0);
            if (finalEdge == nullptr)
            { error = "shader operation plan requires one exact Output input"; return false; }

            std::vector<const CompiledVisualOperation*> ordered;
            std::set<int> visiting, visited, sourceResources;
            std::function<bool(int)> visit = [&](int nodeId)
            {
                const int index = operationIndex(nodeId);
                if (index < 0)
                { error = "shader operation plan contains an unknown resource identity"; return false; }
                const auto& op = plan.operations[static_cast<std::size_t>(index)];
                if (op.kind == "video.source" || op.kind == "video.layer.source")
                {
                    if (op.backendCapability != "source-decode")
                    { error = "shader operation source requires the decode backend"; return false; }
                    sourceResources.insert(nodeId);
                    return sourceResources.size() <= 2;
                }
                if (visited.count(nodeId) != 0) return true;
                if (!visiting.insert(nodeId).second)
                { error = "shader operation plan contains a cycle"; return false; }
                const bool filter = op.kind == "visual.shader.filter";
                const bool transition = op.kind == shadertransition::operationKind;
                const bool generator = op.kind == "visual.shader.generator"
                    || op.kind == "visual.shader.custom";
                if (!filter && !transition && !generator)
                { error = "shader operation plan has unsupported production topology"; return false; }
                if (filter && !exactFilterPorts(nodeId))
                { error = "shader filter requires exact compatible Frame<Image> ports"; return false; }
                const std::size_t inputCount = transition ? 2u : filter ? 1u : 0u;
                for (std::size_t input = 0; input < inputCount; ++input)
                {
                    const auto* edge = incomingEdge(nodeId, static_cast<int>(input));
                    if (edge == nullptr || !visit(edge->fromNodeId))
                    { if (error.empty()) error = "shader operation has no exact input resource"; return false; }
                }
                if (transition)
                {
                    const auto* start = incomingEdge(nodeId, 0);
                    const auto* end = incomingEdge(nodeId, 1);
                    if (start->fromNodeId == end->fromNodeId)
                    { error = "shader transition requires two distinct exact frame producers"; return false; }
                }
                visiting.erase(nodeId);
                visited.insert(nodeId);
                ordered.push_back(&op);
                if (ordered.size() > ShaderOperationPlan::maximumOperations)
                { error = "shader operation plan exceeds the bounded operation budget"; return false; }
                return true;
            };
            if (!visit(finalEdge->fromNodeId) || ordered.empty()) return false;
            std::size_t expectedEdges = 1;
            for (const auto* op : ordered)
                expectedEdges += op->kind == shadertransition::operationKind ? 2u
                    : op->kind == "visual.shader.filter" ? 1u : 0u;
            if (plan.edges.size() != expectedEdges)
            { error = "shader operation plan has unsupported branching or unused edges"; return false; }

            auto shaderPlan = std::make_shared<ShaderOperationPlan>();
            for (const auto* admitted : ordered)
            {
                const auto& admittedOp = *admitted;
                const bool custom = admittedOp.kind == "visual.shader.custom";
                const bool transition = admittedOp.kind == shadertransition::operationKind;
                const bool generator = admittedOp.kind != "visual.shader.filter" && !transition;
                if (admittedOp.backendCapability != "native-gpu")
                { error = "shader operation plan requires the native GPU backend"; return false; }
                FlatShaderBridgePayload bridge;
                std::optional<shadertransition::Payload> transitionPayload;
                if (transition)
                {
                    if (!exactTransitionPorts(admittedOp.nodeId))
                    { error = "shader transition requires exactly two Frame<Image> inputs and one output with exact compatible descriptors"; return false; }
                    AdmittedCuratedShaderTransition admittedTransition;
                    if (!admitCuratedShaderTransitionPayload(
                            admittedOp.payloadXml, admittedTransition, error)) return false;
                    const auto* catalog = shadercatalog::find(
                        admittedTransition.payload.catalogPackId,
                        admittedTransition.payload.catalogProgramId);
                    std::map<std::string, double> values;
                    if (catalog == nullptr || !shadercatalog::validateParameterWire(
                            *catalog, admittedTransition.payload.parameters, values, error)) return false;
                    bridge.schemaVersion = 2;
                    bridge.role = FlatShaderRole::filter;
                    bridge.language = FlatShaderLanguage::isf;
                    bridge.source = admittedTransition.payload.source;
                    bridge.sourceSha256 = admittedTransition.payload.sourceSha256;
                    bridge.catalogPackId = admittedTransition.payload.catalogPackId;
                    bridge.catalogProgramId = admittedTransition.payload.catalogProgramId;
                    bridge.parameters = admittedTransition.payload.parameters;
                    bridge.parameterValues = std::move(values);
                    transitionPayload = admittedTransition.payload;
                }
                else if (custom)
                {
                    if (!admitCustomFlatShaderBridgeOperation(admittedOp, bridge, error)) return false;
                }
                else if (generator)
                {
                    if (!admitFlatShaderBridgeOperation(admittedOp, bridge, error)) return false;
                }
                else if (!admitCatalogIsfFilterOperation(admittedOp, bridge, error)) return false;
                if (!transition && generator != (bridge.role == FlatShaderRole::generator))
                { error = "shader operation role differs from its graph topology"; return false; }
                ShaderOperation operation;
                operation.kind = transition ? ShaderOperationKind::transition
                    : generator ? ShaderOperationKind::generator : ShaderOperationKind::filter;
                operation.nodeId = admittedOp.nodeId;
                operation.inputCount = transition ? 2u : generator ? 0u : 1u;
                for (std::size_t input = 0; input < operation.inputCount; ++input)
                    operation.inputNodeIds[input] = incomingEdge(
                        admittedOp.nodeId, static_cast<int>(input))->fromNodeId;
                const int outputPort = transition ? 2 : generator ? 0 : 1;
                const auto* next = outgoingEdge(admittedOp.nodeId, outputPort);
                if (next == nullptr)
                { error = "shader operation has no exact output resource"; return false; }
                operation.outputNodeId = next->toNodeId;
                operation.payload = std::move(bridge);
                operation.generatedParameters = operation.payload.parameterValues;
                operation.transitionPayload = std::move(transitionPayload);
                if (custom)
                {
                    const auto grantJson = nlohmann::json::parse(admittedOp.runtimeGrantJson);
                    programmableruntime::Grant grant;
                    if (!programmableadmission::parseGrant(grantJson, grant, error)) return false;
                    operation.customGrant = std::move(grant);
                }
                shaderPlan->passTargetCount += operation.payload.passResources.passes.size();
                if (shaderPlan->passTargetCount > ShaderOperationPlan::maximumPassTargets)
                { error = "shader operation plan exceeds the bounded multipass resource budget"; return false; }
                shaderPlan->operations.push_back(std::move(operation));
            }
            execution.transform = execution.effects = execution.mask = false;
            execution.flatShaderBridge = true;
            shaderPlan->revision = plan.structuralRevision;
            shaderPlan->digest = shaderOperationPlanDigest(*shaderPlan, shaderPlan->revision);
            execution.shaderOperationPlan = std::move(shaderPlan);
            return true;
        }

        if (kinds.size() == 2 && kinds[0] == "visual.particles" && kinds[1] == "video.out")
        {
            const bool exact = plan.edges.size() == 1 && plan.edges[0].fromNodeId == ids[0]
                && plan.edges[0].fromPort == 1 && plan.edges[0].toNodeId == ids[1]
                && plan.edges[0].toPort == 0;
            if (! exact)
            {
                error = "visual.particles has unsupported production topology";
                return false;
            }
            const auto& payload = plan.operations[0].payloadXml;
            execution.transform = execution.effects = execution.mask = false;
            execution.particles = true;
            execution.particleNodeId = ids[0];
            execution.particleSeed = std::clamp((int) visualPayloadFloat(payload, "seed", 1.0f), 0, 65535);
            execution.particleCount = std::clamp((int) visualPayloadFloat(payload, "count", 512.0f), 1,
                                                 kMaxVisualParticles);
            execution.particleLifetime = std::clamp(visualPayloadFloat(payload, "lifetime", 1.8f), 0.1f, 10.0f);
            execution.particleSize = std::clamp(visualPayloadFloat(payload, "size", 4.0f), 1.0f, 32.0f);
            execution.particleSpeed = std::clamp(visualPayloadFloat(payload, "speed", 1.0f), 0.0f, 4.0f);
            execution.particleRed = std::clamp(visualPayloadFloat(payload, "red", 0.2f), 0.0f, 1.0f);
            execution.particleGreen = std::clamp(visualPayloadFloat(payload, "green", 0.7f), 0.0f, 1.0f);
            execution.particleBlue = std::clamp(visualPayloadFloat(payload, "blue", 1.0f), 0.0f, 1.0f);
            execution.particleAlpha = std::clamp(visualPayloadFloat(payload, "alpha", 1.0f), 0.0f, 1.0f);
            return true;
        }

        // The existing LayerDesc compositor represents one image value flowing
        // through a linear chain. Require every exact compiled edge to be the
        // corresponding consecutive image operation; branching/control/image
        // fan-in remains unsupported rather than being flattened or CPU-rendered.
        const bool linear = plan.edges.size() + 1 == ids.size()
            && std::all_of(ids.begin(), ids.end() - 1, [&](int id)
            {
                const auto index = (size_t) std::distance(ids.begin(),
                    std::find(ids.begin(), ids.end(), id));
                return std::any_of(plan.edges.begin(), plan.edges.end(),
                    [&](const CompiledVisualEdgeBinding& edge)
                    { return edge.fromNodeId == id && edge.toNodeId == ids[index + 1]; });
            });
        if (! linear)
        {
            const auto indexOf = [&](const char* kind)
            {
                const auto found = std::find(kinds.begin(), kinds.end(), kind);
                return found == kinds.end() ? -1 : (int) std::distance(kinds.begin(), found);
            };
            const int matteAsset = indexOf("visual.matte.asset");
            const int matteRefine = indexOf("visual.matte.refine");
            const int matteApply = indexOf("visual.matte.apply");
            const int matteCombine = indexOf("visual.matte.combine");
            const int matteSource = indexOf("video.source");
            const int matteOutput = indexOf("video.out");
            const int depthAsset = indexOf("visual.depth.asset");
            if (depthAsset >= 0)
            {
                int fog = indexOf("visual.depth.fog");
                if (fog < 0) fog = indexOf("visual.depth.blur");
                if (fog < 0) fog = indexOf("visual.depth.displace");
                if (fog < 0) fog = indexOf("visual.depth.relight");
                const auto edge = [&](int from, int fromPort, int to, int toPort)
                { return from >= 0 && to >= 0 && std::any_of(plan.edges.begin(), plan.edges.end(),
                    [&](const CompiledVisualEdgeBinding& e) { return e.fromNodeId == ids[(size_t)from]
                        && e.fromPort == fromPort && e.toNodeId == ids[(size_t)to] && e.toPort == toPort; }); };
                const bool exact = kinds.size() == 4 && ids.size() == 4 && kinds[0] == "video.source"
                    && kinds.back() == "video.out" && fog >= 0 && plan.edges.size() == 3
                    && edge(0, 0, fog, 0) && edge(depthAsset, 1, fog, 1)
                    && edge(fog, 2, 3, 0);
                const auto& binding = plan.operations[(size_t) depthAsset];
                const auto required = [&](const char* value) { return binding.payloadXml.find(value) != std::string::npos; };
                if (! exact || binding.backendCapability != "source-decode"
                    || binding.payloadXml.rfind("<DepthAssetBinding ", 0) != 0
                    || binding.payloadXml.size() < 3
                    || binding.payloadXml.compare(binding.payloadXml.size() - 2, 2, "/>") != 0
                    || ! required("state=\"available\"") || ! required("cacheKey=\"")
                    || ! required("contentReceipt=\"") || ! required("analysisReceipt=\"")
                    || ! required("framePrefix=\"") || ! required("frameExtension=\"")
                    || ! required("width=\"") || ! required("height=\"")
                    || ! required("fps=\"") || ! required("frames=\"")
                    || ! required("format=\"r16-unorm\"")
                    || binding.payloadXml.find("sequencePath=\"") != std::string::npos
                    || binding.payloadXml.find(" path=\"") != std::string::npos
                    || binding.payloadXml.find(" file=\"") != std::string::npos
                    || binding.payloadXml.find(" uri=\"") != std::string::npos
                    || binding.payloadXml.find(" decoder=\"") != std::string::npos
                    || binding.payloadXml.find(" upload=\"") != std::string::npos
                    || binding.payloadXml.find(" futureField=\"") != std::string::npos)
                {
                    error = "typed depth asset binding is missing, stale, or unsupported";
                    return false;
                }
                execution.transform = execution.effects = execution.mask = false;
                execution.depthFog = kinds[(size_t) fog] == "visual.depth.fog";
                execution.depthEffect = execution.depthFog ? 1
                    : kinds[(size_t) fog] == "visual.depth.blur" ? 2
                    : kinds[(size_t) fog] == "visual.depth.displace" ? 3 : 4;
                const auto& fogPayload = plan.operations[(size_t) fog].payloadXml;
                execution.fogNear = std::clamp(visualPayloadFloat(fogPayload, "near", 0.0f), 0.0f, 1.0f);
                execution.fogFar = std::clamp(visualPayloadFloat(fogPayload, "far", 1.0f), 0.0f, 1.0f);
                execution.fogDensity = std::clamp(visualPayloadFloat(fogPayload, "density", 1.0f), 0.0f, 32.0f);
                execution.fogRed = std::clamp(visualPayloadFloat(fogPayload, "red", 1.0f), 0.0f, 1.0f);
                execution.fogGreen = std::clamp(visualPayloadFloat(fogPayload, "green", 1.0f), 0.0f, 1.0f);
                execution.fogBlue = std::clamp(visualPayloadFloat(fogPayload, "blue", 1.0f), 0.0f, 1.0f);
                execution.fogAlpha = std::clamp(visualPayloadFloat(fogPayload, "alpha", 1.0f), 0.0f, 1.0f);
                if (execution.depthEffect == 2) {
                    execution.depthParam0 = std::clamp(visualPayloadFloat(fogPayload, "radius", 4.0f), 0.0f, 32.0f);
                    execution.depthParam1 = std::clamp(visualPayloadFloat(fogPayload, "focus", 0.5f), 0.0f, 1.0f);
                    execution.depthParam2 = std::clamp(visualPayloadFloat(fogPayload, "falloff", 0.25f), 0.001f, 1.0f);
                } else if (execution.depthEffect == 3) {
                    execution.depthParam0 = std::clamp(visualPayloadFloat(fogPayload, "amountX", 0.0f), -0.25f, 0.25f);
                    execution.depthParam1 = std::clamp(visualPayloadFloat(fogPayload, "amountY", 0.0f), -0.25f, 0.25f);
                    execution.depthParam2 = std::clamp(visualPayloadFloat(fogPayload, "center", 0.5f), 0.0f, 1.0f);
                } else if (execution.depthEffect == 4) {
                    execution.depthParam0 = std::clamp(visualPayloadFloat(fogPayload, "intensity", 1.0f), -2.0f, 2.0f);
                    execution.depthParam1 = std::clamp(visualPayloadFloat(fogPayload, "ambient", 1.0f), 0.0f, 2.0f);
                    execution.depthColorRed = std::clamp(visualPayloadFloat(fogPayload, "red", 1.0f), 0.0f, 2.0f);
                    execution.depthColorGreen = std::clamp(visualPayloadFloat(fogPayload, "green", 1.0f), 0.0f, 2.0f);
                    execution.depthColorBlue = std::clamp(visualPayloadFloat(fogPayload, "blue", 1.0f), 0.0f, 2.0f);
                }
                return true;
            }
            if ((matteAsset >= 0 || matteRefine >= 0 || matteApply >= 0)
                && indexOf("video.blend") < 0)
            {
                const auto hasExactEdge = [&](int from, int fromPort, int to, int toPort)
                {
                    return from >= 0 && to >= 0 && std::any_of(plan.edges.begin(), plan.edges.end(),
                        [&](const CompiledVisualEdgeBinding& edge)
                        { return edge.fromNodeId == ids[(size_t) from] && edge.fromPort == fromPort
                              && edge.toNodeId == ids[(size_t) to] && edge.toPort == toPort; });
                };
                std::vector<int> matteAssets;
                for (size_t i = 0; i < kinds.size(); ++i)
                    if (kinds[i] == "visual.matte.asset") matteAssets.push_back((int)i);
                const bool combined = matteCombine >= 0;
                const bool refined = matteRefine >= 0;
                const bool reusedAsset = combined && matteAssets.size() == 1;
                const size_t expectedNodes = (refined ? 5u : 4u)
                    + (combined ? (reusedAsset ? 1u : 2u) : 0u);
                const int maskSource = combined ? matteCombine : matteAsset;
                const bool exact = kinds.size() == expectedNodes && matteSource == 0
                    && matteAssets.size() == (combined ? (reusedAsset ? 1u : 2u) : 1u)
                    && matteApply > matteAsset
                    && matteOutput == (int) kinds.size() - 1
                    && hasExactEdge(matteSource, 0, matteApply, 0)
                    && (!combined || (hasExactEdge(matteAssets[0], 0, matteCombine, 0)
                                   && hasExactEdge(matteAssets[reusedAsset ? 0 : 1], 0,
                                                   matteCombine, 1)))
                    && hasExactEdge(maskSource, combined ? 2 : 0,
                                    refined ? matteRefine : matteApply, refined ? 0 : 1)
                    && (! refined || hasExactEdge(matteRefine, 1, matteApply, 1))
                    && hasExactEdge(matteApply, 2, matteOutput, 0)
                    && plan.edges.size() == (refined ? 4u : 3u) + (combined ? 2u : 0u);
                if (! exact)
                {
                    error = "typed matte graph has unsupported production topology";
                    return false;
                }
                const auto validBinding = [&](int index)
                {
                    const auto& binding = plan.operations[(size_t)index];
                    return binding.backendCapability == "source-decode"
                        && binding.payloadXml.find("matteAssetId=\"") != std::string::npos
                        && binding.payloadXml.find("state=\"available\"") != std::string::npos
                        && binding.payloadXml.find("cacheKey=\"") != std::string::npos
                        && binding.payloadXml.find("contentReceipt=\"") != std::string::npos
                        && binding.payloadXml.find("framePrefix=\"") != std::string::npos
                        && binding.payloadXml.find("frameExtension=\"") != std::string::npos
                        && binding.payloadXml.find("firstFrame=\"") != std::string::npos
                        && binding.payloadXml.find("frameDigits=\"") != std::string::npos
                        && binding.payloadXml.find("sequencePath=\"") == std::string::npos
                        && binding.payloadXml.find("backend=\"rgba-cpu-decode-native-gpu-upload\"") != std::string::npos;
                };
                if (!std::all_of(matteAssets.begin(), matteAssets.end(), validBinding))
                {
                    error = "typed matte asset binding is missing, stale, or unsupported";
                    return false;
                }
                execution.transform = execution.effects = execution.mask = false;
                execution.matteApply = true;
                if (combined)
                    execution.matteCombineMode = std::clamp((int)visualPayloadFloat(
                        plan.operations[(size_t)matteCombine].payloadXml, "mode", 0.0f), 0, 3);
                if (refined)
                {
                    const auto& payload = plan.operations[(size_t) matteRefine].payloadXml;
                    execution.matteInvert = visualPayloadFloat(payload, "invert", 0.0f) >= 0.5f;
                    execution.matteBlack = std::clamp(visualPayloadFloat(payload, "black", 0.0f), 0.0f, 1.0f);
                    execution.matteWhite = std::clamp(visualPayloadFloat(payload, "white", 1.0f), 0.0f, 1.0f);
                    execution.matteErodeDilate = std::clamp(visualPayloadFloat(payload, "erodeDilate", 0.0f), -4.0f, 4.0f);
                    execution.matteFeather = std::clamp(visualPayloadFloat(payload, "feather", 0.0f), 0.0f, 4.0f);
                    execution.matteChoke = std::clamp(visualPayloadFloat(payload, "choke", 0.0f), -1.0f, 1.0f);
                }
                return true;
            }

            // The established native compositor exposes exactly three inputs:
            // the descriptor-owned primary frame, one text overlay, and one
            // referenced layer. Keep this first multi-source slice exact; kinds
            // alone are insufficient because the two secondary ports are not
            // interchangeable in the renderer.
            const bool exactThreeSourceComposite = isExactThreeSourceVisualDag(kinds);
            if (exactThreeSourceComposite)
            {
                const int sourceIndex = indexOf("video.source");
                const int textIndex = indexOf("video.text");
                const int layerIndex = indexOf("video.layer.source");
                const int blendIndex = indexOf("video.blend");
                const int outputIndex = indexOf("video.out");
                const auto capabilityIs = [&](int index, const char* capability)
                {
                    return index >= 0 && plan.operations[(size_t) index].backendCapability == capability;
                };
                if (! capabilityIs(sourceIndex, "source-decode")
                    || ! capabilityIs(textIndex, "native-gpu")
                    || ! capabilityIs(layerIndex, "source-decode")
                    || ! capabilityIs(blendIndex, "native-gpu")
                    || ! capabilityIs(outputIndex, "native-gpu"))
                {
                    error = "bounded visual DAG has an incompatible native backend capability";
                    return false;
                }

                const auto findPort = [&](int index, int port)
                {
                    return std::find_if(plan.ports.begin(), plan.ports.end(),
                        [&](const CompiledVisualPortBinding& binding)
                        { return binding.nodeId == ids[(size_t) index] && binding.port == port; });
                };
                const auto exactImagePort = [&](int index, int port, const char* direction)
                {
                    const auto found = findPort(index, port);
                    return found != plan.ports.end() && found->direction == direction
                        && found->channels == 1 && found->carrier == "frame"
                        && found->dataType == "image" && ! found->pixelFormat.empty()
                        && found->pixelFormat != "unspecified" && ! found->colorSpace.empty()
                        && found->colorSpace != "unspecified";
                };
                if (! exactImagePort(sourceIndex, 0, "out")
                    || ! exactImagePort(textIndex, 0, "out")
                    || ! exactImagePort(layerIndex, 0, "out")
                    || ! exactImagePort(blendIndex, 0, "in")
                    || ! exactImagePort(blendIndex, 1, "out")
                    || ! exactImagePort(blendIndex, 2, "in")
                    || ! exactImagePort(blendIndex, 3, "in")
                    || ! exactImagePort(outputIndex, 0, "in"))
                {
                    error = "bounded visual DAG requires complete typed image ports";
                    return false;
                }

                const std::array<CompiledVisualEdgeBinding, 4> expectedEdges {{
                    { ids[(size_t) sourceIndex], 0, ids[(size_t) blendIndex], 0 },
                    { ids[(size_t) textIndex], 0, ids[(size_t) blendIndex], 2 },
                    { ids[(size_t) layerIndex], 0, ids[(size_t) blendIndex], 3 },
                    { ids[(size_t) blendIndex], 1, ids[(size_t) outputIndex], 0 }
                }};
                for (const auto& edge : expectedEdges)
                {
                    const auto from = std::find_if(plan.ports.begin(), plan.ports.end(),
                        [&](const auto& port) { return port.nodeId == edge.fromNodeId
                            && port.port == edge.fromPort; });
                    const auto to = std::find_if(plan.ports.begin(), plan.ports.end(),
                        [&](const auto& port) { return port.nodeId == edge.toNodeId
                            && port.port == edge.toPort; });
                    if (from->pixelFormat != to->pixelFormat || from->colorSpace != to->colorSpace)
                    {
                        error = "bounded visual DAG does not perform pixel-format or color conversion";
                        return false;
                    }
                }
                std::array<bool, 4> matched {};
                bool exactEdges = plan.edges.size() == expectedEdges.size();
                for (const auto& edge : plan.edges)
                {
                    bool found = false;
                    for (size_t i = 0; i < expectedEdges.size(); ++i)
                    {
                        const auto& expected = expectedEdges[i];
                        if (! matched[i] && edge.fromNodeId == expected.fromNodeId
                            && edge.fromPort == expected.fromPort
                            && edge.toNodeId == expected.toNodeId
                            && edge.toPort == expected.toPort)
                        {
                            matched[i] = true;
                            found = true;
                            break;
                        }
                    }
                    exactEdges = exactEdges && found;
                }
                exactEdges = exactEdges
                    && std::all_of(matched.begin(), matched.end(), [](bool value) { return value; });
                if (! exactEdges)
                {
                    error = "bounded visual DAG requires exact primary/text/layer Blend bindings";
                    return false;
                }

                execution.transform = execution.effects = execution.mask = false;
                execution.compositeInputCount = expectedEdges.size() - 1;
                for (size_t i = 0; i < execution.compositeInputCount; ++i)
                {
                    execution.compositeSourceNodeIds[i] = expectedEdges[i].fromNodeId;
                    execution.compositeDestinationPorts[i] = expectedEdges[i].toPort;
                }
                return true;
            }
            std::vector<size_t> blendIndices;
            std::vector<size_t> outputIndices;
            for (size_t i = 0; i < kinds.size(); ++i)
            {
                if (kinds[i] == "video.blend") blendIndices.push_back(i);
                if (kinds[i] == "video.out") outputIndices.push_back(i);
            }
            if (blendIndices.size() != 1 || outputIndices.size() != 1)
            {
                error = "visual composite requires exactly one native Blend and one Output";
                return false;
            }
            const size_t blendIndex = blendIndices.front();
            const size_t outputIndex = outputIndices.front();
            if (outputIndex + 1 != kinds.size())
            {
                error = "visual composite Output must be the terminal ordered operation";
                return false;
            }

            const auto findExactPort = [&](int nodeId, int portIndex)
                -> const CompiledVisualPortBinding*
            {
                const CompiledVisualPortBinding* result = nullptr;
                for (const auto& port : plan.ports)
                    if (port.nodeId == nodeId && port.port == portIndex)
                    {
                        if (result != nullptr) return nullptr;
                        result = &port;
                    }
                return result;
            };
            const auto operationIndex = [&](int nodeId) -> int
            {
                const auto found = std::find(ids.begin(), ids.end(), nodeId);
                return found == ids.end() ? -1 : (int) std::distance(ids.begin(), found);
            };
            const auto expectedCapability = [](const std::string& kind)
            {
                if (kind == "video.source" || kind == "video.layer.source"
                    || kind == "visual.matte.asset")
                    return "source-decode";
                if (kind == "visual.shape.rectangle" || kind == "visual.shape.ellipse"
                    || kind == "visual.shape.union" || kind == "visual.shape.intersection"
                    || kind == "visual.shape.subtract" || kind == "visual.shape.invert")
                    return "control-eval";
                return "native-gpu";
            };
            for (const auto& operation : plan.operations)
                if (operation.backendCapability != expectedCapability(operation.kind))
                {
                    error = "visual composite operation has incompatible native backend capability";
                    return false;
                }

            std::vector<CompiledVisualEdgeBinding> expectedEdges;
            const auto expectEdge = [&](size_t from, int fromPort, size_t to, int toPort)
            {
                expectedEdges.push_back({ ids[from], fromPort, ids[to], toPort });
            };
            expectEdge(blendIndex, 1, outputIndex, 0);

            std::vector<bool> consumed(kinds.size(), false);
            consumed[blendIndex] = true;
            consumed[outputIndex] = true;
            const auto incoming = [&](size_t to, int toPort,
                                      CompiledVisualEdgeBinding& result) -> bool
            {
                bool found = false;
                for (const auto& edge : plan.edges)
                    if (edge.toNodeId == ids[to] && edge.toPort == toPort)
                    {
                        if (found) return false;
                        result = edge;
                        found = true;
                    }
                return found;
            };
            const auto recordInput = [&](int sourceNodeId, int destinationPort) -> bool
            {
                if (execution.compositeInputCount >= VisualLayerExecution::kMaxCompositeInputs)
                    return false;
                const size_t position = execution.compositeInputCount++;
                execution.compositeSourceNodeIds[position] = sourceNodeId;
                execution.compositeDestinationPorts[position] = destinationPort;
                return true;
            };

            // Port 0 is the descriptor-owned primary image. Its exact native
            // representation is Source/Particles -> [Transform] -> [Effects]
            // -> [Mask or Path Matte]. Repeated unary passes require distinct intermediate
            // descriptors and therefore remain fail-closed.
            CompiledVisualEdgeBinding primaryEdge;
            if (! incoming(blendIndex, 0, primaryEdge))
            {
                error = "visual composite is missing the primary Blend input at port 0";
                return false;
            }
            int current = operationIndex(primaryEdge.fromNodeId);
            if (current < 0)
            {
                error = "visual composite primary input has no stable operation identity";
                return false;
            }
            expectEdge((size_t) current, primaryEdge.fromPort, blendIndex, 0);
            execution.transform = false;
            execution.effects = false;
            execution.mask = false;
            int previousStage = 4;
            while (current >= 0 && kinds[(size_t) current] != "video.source"
                   && kinds[(size_t) current] != "visual.particles")
            {
                const auto& kind = kinds[(size_t) current];
                const bool commonEffectStage =
                    commoneffect::kindForGraphName(kind).has_value();
                const bool catalogFilterStage = kind == "visual.shader.filter";
                const bool matteApplyStage = kind == "visual.matte.apply";
                const int stage = catalogFilterStage ? 0
                    : kind == "video.transform" ? 1
                    : (kind == "video.effects" || commonEffectStage) ? 2
                    : (kind == "video.mask.shape" || kind == "visual.matte.path"
                       || matteApplyStage) ? 3 : -1;
                const int stageOutputPort = (kind == "visual.matte.path" || matteApplyStage) ? 2 : 1;
                if (stage < 0 || stage >= previousStage || consumed[(size_t) current]
                    || primaryEdge.fromPort != stageOutputPort)
                {
                    error = "visual composite primary branch cannot be represented exactly by LayerDesc";
                    return false;
                }
                consumed[(size_t) current] = true;
                if (kind == "video.transform") execution.transform = true;
                if (kind == "video.effects") execution.effects = true;
                if (kind == "video.mask.shape") execution.mask = true;
                if (kind == "visual.matte.path")
                {
                    const auto exactTypedPort = [&] (int nodeId, int portIndex,
                                                     const char* direction,
                                                     const char* carrier,
                                                     const char* dataType)
                        -> const CompiledVisualPortBinding*
                    {
                        const CompiledVisualPortBinding* result = nullptr;
                        for (const auto& port : plan.ports)
                            if (port.nodeId == nodeId && port.port == portIndex)
                            {
                                if (result != nullptr || port.channels != 1
                                    || port.direction != direction || port.carrier != carrier
                                    || port.dataType != dataType)
                                    return nullptr;
                                result = &port;
                            }
                        return result;
                    };
                    const auto portCount = [&] (int nodeId)
                    {
                        return std::count_if(plan.ports.begin(), plan.ports.end(),
                            [=](const auto& port) { return port.nodeId == nodeId; });
                    };
                    const auto* frameInput = exactTypedPort(ids[(size_t) current], 0,
                                                            "in", "frame", "image");
                    const auto* shapeInput = exactTypedPort(ids[(size_t) current], 1,
                                                            "in", "control", "shape");
                    const auto* frameOutput = exactTypedPort(ids[(size_t) current], 2,
                                                             "out", "frame", "image");
                    const auto* successor = findExactPort(primaryEdge.toNodeId, primaryEdge.toPort);
                    if (portCount(ids[(size_t) current]) != 3 || frameInput == nullptr
                        || shapeInput == nullptr || frameOutput == nullptr || successor == nullptr
                        || frameInput->pixelFormat != frameOutput->pixelFormat
                        || frameInput->colorSpace != frameOutput->colorSpace
                        || frameOutput->pixelFormat != successor->pixelFormat
                        || frameOutput->colorSpace != successor->colorSpace)
                    {
                        error = "visual Path Matte requires exact compatible Frame<Image> and Shape bindings";
                        return false;
                    }

                    CompiledVisualEdgeBinding shapeEdge;
                    if (! incoming((size_t) current, 1, shapeEdge))
                    {
                        error = "visual Path Matte has no exact Shape binding";
                        return false;
                    }
                    int root = operationIndex(shapeEdge.fromNodeId);
                    if (root < 0)
                    {
                        error = "visual Path Matte Shape binding has no stable operation identity";
                        return false;
                    }
                    const auto isPrimitive = [&] (int index)
                    {
                        return index >= 0 && (kinds[(size_t) index] == "visual.shape.rectangle"
                            || kinds[(size_t) index] == "visual.shape.ellipse");
                    };
                    if (kinds[(size_t) root] == "visual.shape.invert")
                    {
                        CompiledVisualEdgeBinding invertEdge;
                        if (shapeEdge.fromPort != 1 || portCount(ids[(size_t) root]) != 2
                            || exactTypedPort(ids[(size_t) root], 0, "in", "control", "shape") == nullptr
                            || exactTypedPort(ids[(size_t) root], 1, "out", "control", "shape") == nullptr
                            || ! incoming((size_t) root, 0, invertEdge))
                        {
                            error = "visual Path Matte invert requires one exact typed Shape binding";
                            return false;
                        }
                        const int invertOperand = operationIndex(invertEdge.fromNodeId);
                        if (invertOperand < 0)
                        {
                            error = "visual Path Matte invert operand has no stable operation identity";
                            return false;
                        }
                        expectEdge((size_t) invertOperand, invertEdge.fromPort,
                                   (size_t) root, 0);
                        consumed[(size_t) root] = true;
                        root = invertOperand;
                        shapeEdge = invertEdge;
                        execution.pathMatteInvert = true;
                    }
                    const auto booleanOperation = [&] (int index)
                    {
                        if (index < 0) return 0;
                        if (kinds[(size_t) index] == "visual.shape.union") return 1;
                        if (kinds[(size_t) index] == "visual.shape.intersection") return 2;
                        if (kinds[(size_t) index] == "visual.shape.subtract") return 3;
                        return 0;
                    }(root);
                    int first = root, second = -1;
                    if (booleanOperation != 0)
                    {
                        CompiledVisualEdgeBinding a, b;
                        if (shapeEdge.fromPort != 2 || portCount(ids[(size_t) root]) != 3
                            || exactTypedPort(ids[(size_t) root], 0, "in", "control", "shape") == nullptr
                            || exactTypedPort(ids[(size_t) root], 1, "in", "control", "shape") == nullptr
                            || exactTypedPort(ids[(size_t) root], 2, "out", "control", "shape") == nullptr
                            || ! incoming((size_t) root, 0, a) || ! incoming((size_t) root, 1, b))
                        {
                            error = "visual Path Matte Boolean requires exact typed a, b and Shape bindings";
                            return false;
                        }
                        first = operationIndex(a.fromNodeId);
                        second = operationIndex(b.fromNodeId);
                        if (! isPrimitive(first) || ! isPrimitive(second)
                            || a.fromPort != 0 || b.fromPort != 0)
                        {
                            error = "visual Path Matte Boolean requires two rectangle or ellipse operands";
                            return false;
                        }
                        expectEdge((size_t) first, 0, (size_t) root, 0);
                        expectEdge((size_t) second, 0, (size_t) root, 1);
                        consumed[(size_t) root] = true;
                    }
                    else if (! isPrimitive(first) || shapeEdge.fromPort != 0)
                    {
                        error = "visual Path Matte requires one primitive or one bounded Boolean Shape";
                        return false;
                    }
                    const auto parsePrimitive = [&] (int index, float& cx, float& cy,
                                                     float& width, float& height)
                    {
                        const auto& payload = plan.operations[(size_t) index].payloadXml;
                        return portCount(ids[(size_t) index]) == 1
                            && exactTypedPort(ids[(size_t) index], 0, "out", "control", "shape") != nullptr
                            && visualBoundedPayloadFloat(payload, "centerX", 0.5f, -2.0f, 2.0f, cx)
                            && visualBoundedPayloadFloat(payload, "centerY", 0.5f, -2.0f, 2.0f, cy)
                            && visualBoundedPayloadFloat(payload, "width", 1.0f, 0.0f, 4.0f, width)
                            && visualBoundedPayloadFloat(payload, "height", 1.0f, 0.0f, 4.0f, height);
                    };
                    if (! parsePrimitive(first, execution.pathCx, execution.pathCy,
                                         execution.pathW, execution.pathH)
                        || (second >= 0 && ! parsePrimitive(second, execution.path2Cx,
                            execution.path2Cy, execution.path2W, execution.path2H)))
                    {
                        error = "visual Path Matte primitive payload is malformed or out of bounds";
                        return false;
                    }
                    expectEdge((size_t) (execution.pathMatteInvert
                        ? operationIndex(shapeEdge.toNodeId) : root),
                        execution.pathMatteInvert ? 1 : shapeEdge.fromPort, (size_t) current, 1);
                    consumed[(size_t) first] = true;
                    if (second >= 0) consumed[(size_t) second] = true;
                    execution.pathMatte = true;
                    execution.pathMatteEllipse = kinds[(size_t) first] == "visual.shape.ellipse";
                    execution.pathMatteHasSecondary = second >= 0;
                    execution.pathMatteSecondaryEllipse = second >= 0
                        && kinds[(size_t) second] == "visual.shape.ellipse";
                    execution.pathMatteOperation = booleanOperation;
                }
                if (matteApplyStage)
                {
                    const auto* imageInput = findExactPort(ids[(size_t) current], 0);
                    const auto* maskInput = findExactPort(ids[(size_t) current], 1);
                    const auto* imageOutput = findExactPort(ids[(size_t) current], 2);
                    const auto* successor = findExactPort(primaryEdge.toNodeId, primaryEdge.toPort);
                    const auto portCount = std::count_if(plan.ports.begin(), plan.ports.end(),
                        [&](const auto& port) { return port.nodeId == ids[(size_t) current]; });
                    const auto exactPort = [](const CompiledVisualPortBinding* port,
                                              const char* direction, const char* dataType)
                    {
                        return port != nullptr && port->channels == 1
                            && port->direction == direction && port->carrier == "frame"
                            && port->dataType == dataType && ! port->pixelFormat.empty()
                            && port->pixelFormat != "unspecified" && ! port->colorSpace.empty()
                            && port->colorSpace != "unspecified";
                    };
                    CompiledVisualEdgeBinding maskEdge;
                    if (portCount != 3 || ! exactPort(imageInput, "in", "image")
                        || ! exactPort(maskInput, "in", "mask")
                        || ! exactPort(imageOutput, "out", "image")
                        || ! exactPort(successor, "in", "image")
                        || imageInput->pixelFormat != imageOutput->pixelFormat
                        || imageInput->colorSpace != imageOutput->colorSpace
                        || imageOutput->pixelFormat != successor->pixelFormat
                        || imageOutput->colorSpace != successor->colorSpace
                        || ! incoming((size_t) current, 1, maskEdge))
                    {
                        error = "visual composite Apply Matte requires exact compatible Image and Mask bindings";
                        return false;
                    }
                    const int maskSource = operationIndex(maskEdge.fromNodeId);
                    if (maskSource < 0 || kinds[(size_t) maskSource] != "visual.matte.asset"
                        || maskEdge.fromPort != 0 || consumed[(size_t) maskSource])
                    {
                        error = "visual composite Apply Matte requires one exact immutable matte source";
                        return false;
                    }
                    const auto* maskOutput = findExactPort(maskEdge.fromNodeId, 0);
                    const auto maskPortCount = std::count_if(plan.ports.begin(), plan.ports.end(),
                        [&](const auto& port) { return port.nodeId == maskEdge.fromNodeId; });
                    const auto& binding = plan.operations[(size_t) maskSource];
                    const bool validBinding = binding.backendCapability == "source-decode"
                        && binding.payloadXml.find("matteAssetId=\"") != std::string::npos
                        && binding.payloadXml.find("state=\"available\"") != std::string::npos
                        && binding.payloadXml.find("cacheKey=\"") != std::string::npos
                        && binding.payloadXml.find("contentReceipt=\"") != std::string::npos
                        && binding.payloadXml.find("framePrefix=\"") != std::string::npos
                        && binding.payloadXml.find("frameExtension=\"") != std::string::npos
                        && binding.payloadXml.find("firstFrame=\"") != std::string::npos
                        && binding.payloadXml.find("frameDigits=\"") != std::string::npos
                        && binding.payloadXml.find("sequencePath=\"") == std::string::npos
                        && binding.payloadXml.find("backend=\"rgba-cpu-decode-native-gpu-upload\"")
                            != std::string::npos;
                    if (maskPortCount != 1 || ! exactPort(maskOutput, "out", "mask")
                        || maskOutput->pixelFormat != maskInput->pixelFormat
                        || maskOutput->colorSpace != maskInput->colorSpace || ! validBinding)
                    {
                        error = "visual composite matte source binding is missing, stale, or incompatible";
                        return false;
                    }
                    expectEdge((size_t) maskSource, 0, (size_t) current, 1);
                    consumed[(size_t) maskSource] = true;
                    execution.matteApply = true;
                }
                if (commonEffectStage)
                {
                    const auto* inputPort = findExactPort(ids[(size_t) current], 0);
                    const auto* outputPort = findExactPort(ids[(size_t) current], 1);
                    const auto* successorPort = findExactPort(
                        primaryEdge.toNodeId, primaryEdge.toPort);
                    const auto portCount = std::count_if(plan.ports.begin(), plan.ports.end(),
                        [&](const auto& port) { return port.nodeId == ids[(size_t) current]; });
                    const auto exactImagePort = [](const CompiledVisualPortBinding* port,
                                                   const char* direction)
                    {
                        return port != nullptr && port->channels == 1
                            && port->direction == direction && port->carrier == "frame"
                            && port->dataType == "image" && ! port->pixelFormat.empty()
                            && port->pixelFormat != "unspecified" && ! port->colorSpace.empty()
                            && port->colorSpace != "unspecified";
                    };
                    if (portCount != 2 || ! exactImagePort(inputPort, "in")
                        || ! exactImagePort(outputPort, "out")
                        || ! exactImagePort(successorPort, "in")
                        || inputPort->pixelFormat != outputPort->pixelFormat
                        || inputPort->colorSpace != outputPort->colorSpace
                        || outputPort->pixelFormat != successorPort->pixelFormat
                        || outputPort->colorSpace != successorPort->colorSpace)
                    {
                        error = "visual composite requires exact compatible typed edge bindings";
                        return false;
                    }
                    ExecutableCommonEffectPayload payload;
                    if (! parseExecutableCommonEffectPayload(
                            plan.operations[(size_t) current].payloadXml, payload)
                        || kind != commoneffect::graphKindName(payload.kind))
                    {
                        error = kind
                            + " has a malformed, unsupported, or out-of-bounds immutable payload";
                        return false;
                    }
                    execution.effects = false;
                    execution.commonEffect = true;
                    execution.commonEffectType = payload.rendererEffectType;
                    execution.commonEffectParameterCount = payload.parameterCount;
                    execution.commonEffectValues = payload.parameters;
                }
                if (catalogFilterStage)
                {
                    const auto* inputPort = findExactPort(ids[(size_t) current], 0);
                    const auto* outputPort = findExactPort(ids[(size_t) current], 1);
                    const auto* successorPort = findExactPort(primaryEdge.toNodeId, primaryEdge.toPort);
                    const auto portCount = std::count_if(plan.ports.begin(), plan.ports.end(),
                        [&](const auto& port) { return port.nodeId == ids[(size_t) current]; });
                    const auto exactImagePort = [](const CompiledVisualPortBinding* port,
                                                   const char* direction)
                    {
                        return port != nullptr && port->channels == 1
                            && port->direction == direction && port->carrier == "frame"
                            && port->dataType == "image" && ! port->pixelFormat.empty()
                            && port->pixelFormat != "unspecified" && ! port->colorSpace.empty()
                            && port->colorSpace != "unspecified";
                    };
                    if (portCount != 2 || ! exactImagePort(inputPort, "in")
                        || ! exactImagePort(outputPort, "out")
                        || ! exactImagePort(successorPort, "in")
                        || inputPort->pixelFormat != outputPort->pixelFormat
                        || inputPort->colorSpace != outputPort->colorSpace
                        || outputPort->pixelFormat != successorPort->pixelFormat
                        || outputPort->colorSpace != successorPort->colorSpace)
                    {
                        error = "catalog ISF filter requires exact compatible Frame<Image> bindings";
                        return false;
                    }
                    FlatShaderBridgePayload bridge;
                    if (! admitCatalogIsfFilterOperation(
                            plan.operations[(size_t) current], bridge, error))
                        return false;
                    CompiledVisualEdgeBinding filterInput;
                    if (! incoming((size_t) current, 0, filterInput))
                    {
                        error = "catalog ISF filter has no exact primary frame input";
                        return false;
                    }
                    error = "catalog ISF filters inside the legacy composite topology are unsupported";
                    return false;
                }
                previousStage = stage;
                CompiledVisualEdgeBinding predecessor;
                if (! incoming((size_t) current, 0, predecessor))
                {
                    error = "visual composite primary branch has a missing exact input edge";
                    return false;
                }
                const int predecessorIndex = operationIndex(predecessor.fromNodeId);
                if (predecessorIndex < 0)
                {
                    error = "visual composite primary branch has an unknown source identity";
                    return false;
                }
                if (commonEffectStage || catalogFilterStage)
                {
                    const auto* predecessorPort = findExactPort(
                        predecessor.fromNodeId, predecessor.fromPort);
                    const auto* inputPort = findExactPort(ids[(size_t) current], 0);
                    if (predecessorPort == nullptr || inputPort == nullptr
                        || predecessorPort->channels != 1 || predecessorPort->direction != "out"
                        || predecessorPort->carrier != "frame" || predecessorPort->dataType != "image"
                        || predecessorPort->pixelFormat != inputPort->pixelFormat
                        || predecessorPort->colorSpace != inputPort->colorSpace)
                    {
                        error = "visual composite requires exact compatible typed edge bindings";
                        return false;
                    }
                }
                expectEdge((size_t) predecessorIndex, predecessor.fromPort, (size_t) current, 0);
                primaryEdge = predecessor;
                current = predecessorIndex;
            }
            const bool decodedRoot = current >= 0 && kinds[(size_t) current] == "video.source";
            const bool particleRoot = current >= 0 && kinds[(size_t) current] == "visual.particles";
            const int requiredRootPort = particleRoot ? 1 : 0;
            if ((! decodedRoot && ! particleRoot)
                || primaryEdge.fromPort != requiredRootPort || consumed[(size_t) current])
            {
                error = "visual composite primary branch requires one exact native image source";
                return false;
            }
            consumed[(size_t) current] = true;
            if (particleRoot)
            {
                const auto& payload = plan.operations[(size_t) current].payloadXml;
                execution.particles = true;
                execution.particleNodeId = ids[(size_t) current];
                execution.particleSeed = std::clamp((int) visualPayloadFloat(payload, "seed", 1.0f), 0, 65535);
                execution.particleCount = std::clamp((int) visualPayloadFloat(payload, "count", 512.0f), 1,
                                                     kMaxVisualParticles);
                execution.particleLifetime = std::clamp(visualPayloadFloat(payload, "lifetime", 1.8f), 0.1f, 10.0f);
                execution.particleSize = std::clamp(visualPayloadFloat(payload, "size", 4.0f), 1.0f, 32.0f);
                execution.particleSpeed = std::clamp(visualPayloadFloat(payload, "speed", 1.0f), 0.0f, 4.0f);
                execution.particleRed = std::clamp(visualPayloadFloat(payload, "red", 0.2f), 0.0f, 1.0f);
                execution.particleGreen = std::clamp(visualPayloadFloat(payload, "green", 0.7f), 0.0f, 1.0f);
                execution.particleBlue = std::clamp(visualPayloadFloat(payload, "blue", 1.0f), 0.0f, 1.0f);
                execution.particleAlpha = std::clamp(visualPayloadFloat(payload, "alpha", 1.0f), 0.0f, 1.0f);
            }
            if (! recordInput(ids[(size_t) current], 0))
            {
                error = "visual composite exceeds native input capacity";
                return false;
            }

            // The existing compositor does not resolve arbitrary graph node
            // identities to secondary textures. Port 2 owns its one text or
            // retained Draw Shape overlay, and port 3 owns its one referenced
            // layer. Keep admission exact until the renderer has node-to-texture
            // routing for general secondary sources.
            bool haveDrawShape = false;
            for (int blendPort : { 2, 3 })
            {
                CompiledVisualEdgeBinding branchEdge;
                if (! incoming(blendIndex, blendPort, branchEdge)) continue;
                const int terminal = operationIndex(branchEdge.fromNodeId);
                if (terminal < 0 || consumed[(size_t) terminal])
                {
                    error = "visual composite secondary input has an invalid or reused source identity";
                    return false;
                }
                expectEdge((size_t) terminal, branchEdge.fromPort, blendIndex, blendPort);
                int source = terminal;
                const auto& terminalKind = kinds[(size_t) terminal];
                const bool directSource = (blendPort == 2 && terminalKind == "video.text")
                    || (blendPort == 3 && terminalKind == "video.layer.source");
                if (directSource)
                {
                    if (branchEdge.fromPort != 0)
                    {
                        error = "visual composite secondary source uses an unsupported output port";
                        return false;
                    }
                    consumed[(size_t) terminal] = true;
                }
                else if (blendPort == 2 && terminalKind == "visual.draw.shape")
                {
                    if (haveDrawShape || branchEdge.fromPort != 1)
                    {
                        error = "visual composite supports one exact native Draw Shape input";
                        return false;
                    }
                    CompiledVisualEdgeBinding shapeEdge;
                    if (! incoming((size_t) terminal, 0, shapeEdge))
                    {
                        error = "visual Draw Shape input has no exact Shape binding";
                        return false;
                    }
                    const auto nodePortCount = [&] (int nodeId)
                    {
                        return std::count_if(plan.ports.begin(), plan.ports.end(),
                            [=](const auto& port) { return port.nodeId == nodeId; });
                    };
                    const auto exactTypedPort = [&] (int nodeId, int portIndex,
                                                     const char* direction,
                                                     const char* carrier,
                                                     const char* dataType)
                        -> const CompiledVisualPortBinding*
                    {
                        const CompiledVisualPortBinding* result = nullptr;
                        for (const auto& port : plan.ports)
                            if (port.nodeId == nodeId && port.port == portIndex)
                            {
                                if (result != nullptr || port.channels != 1
                                    || port.direction != direction || port.carrier != carrier
                                    || port.dataType != dataType)
                                    return nullptr;
                                result = &port;
                            }
                        return result;
                    };
                    const auto* drawInput = exactTypedPort(ids[(size_t) terminal], 0,
                                                           "in", "control", "shape");
                    const auto* drawOutput = exactTypedPort(ids[(size_t) terminal], 1,
                                                            "out", "frame", "image");
                    const auto* blendInput = exactTypedPort(ids[blendIndex], blendPort,
                                                            "in", "frame", "image");
                    if (nodePortCount(ids[(size_t) terminal]) != 2 || drawInput == nullptr
                        || drawOutput == nullptr || blendInput == nullptr
                        || drawOutput->pixelFormat != blendInput->pixelFormat
                        || drawOutput->colorSpace != blendInput->colorSpace)
                    {
                        error = "visual Draw Shape requires exact typed Shape input and Frame<Image> output bindings";
                        return false;
                    }
                    source = operationIndex(shapeEdge.fromNodeId);
                    const auto isPrimitive = [&] (int index)
                    {
                        return index >= 0 && (kinds[(size_t) index] == "visual.shape.rectangle"
                            || kinds[(size_t) index] == "visual.shape.ellipse");
                    };
                    const auto shapeOperation = [&] (int index)
                    {
                        if (index < 0) return 0;
                        if (kinds[(size_t) index] == "visual.shape.union") return 1;
                        if (kinds[(size_t) index] == "visual.shape.intersection") return 2;
                        if (kinds[(size_t) index] == "visual.shape.subtract") return 3;
                        return 0;
                    };
                    int firstPrimitive = source;
                    int secondPrimitive = -1;
                    const int booleanOperation = shapeOperation(source);
                    if (booleanOperation != 0)
                    {
                        CompiledVisualEdgeBinding firstEdge, secondEdge;
                        if (consumed[(size_t) source] || shapeEdge.fromPort != 2
                            || ! incoming((size_t) source, 0, firstEdge)
                            || ! incoming((size_t) source, 1, secondEdge))
                        {
                            error = "visual Draw Shape Boolean input requires exact a, b and shape bindings";
                            return false;
                        }
                        firstPrimitive = operationIndex(firstEdge.fromNodeId);
                        secondPrimitive = operationIndex(secondEdge.fromNodeId);
                        if (! isPrimitive(firstPrimitive) || ! isPrimitive(secondPrimitive)
                            || consumed[(size_t) firstPrimitive] || consumed[(size_t) secondPrimitive]
                            || firstEdge.fromPort != 0 || secondEdge.fromPort != 0)
                        {
                            error = "visual Draw Shape Boolean input requires two rectangle or ellipse operands";
                            return false;
                        }
                        if (nodePortCount(ids[(size_t) firstPrimitive]) != 1
                            || nodePortCount(ids[(size_t) secondPrimitive]) != 1
                            || nodePortCount(ids[(size_t) source]) != 3
                            || exactTypedPort(ids[(size_t) firstPrimitive], 0,
                                              "out", "control", "shape") == nullptr
                            || exactTypedPort(ids[(size_t) secondPrimitive], 0,
                                              "out", "control", "shape") == nullptr
                            || exactTypedPort(ids[(size_t) source], 0,
                                              "in", "control", "shape") == nullptr
                            || exactTypedPort(ids[(size_t) source], 1,
                                              "in", "control", "shape") == nullptr
                            || exactTypedPort(ids[(size_t) source], 2,
                                              "out", "control", "shape") == nullptr)
                        {
                            error = "visual Draw Shape Boolean input requires exact typed primitive and Boolean bindings";
                            return false;
                        }
                        expectEdge((size_t) firstPrimitive, 0, (size_t) source, 0);
                        expectEdge((size_t) secondPrimitive, 0, (size_t) source, 1);
                        consumed[(size_t) firstPrimitive] = true;
                        consumed[(size_t) secondPrimitive] = true;
                        consumed[(size_t) source] = true;
                    }
                    else if (! isPrimitive(source) || consumed[(size_t) source]
                             || shapeEdge.fromPort != 0)
                    {
                        error = "visual Draw Shape input requires one primitive or one bounded Boolean shape";
                        return false;
                    }
                    else
                    {
                        firstPrimitive = source;
                        if (nodePortCount(ids[(size_t) source]) != 1
                            || exactTypedPort(ids[(size_t) source], 0,
                                              "out", "control", "shape") == nullptr)
                        {
                            error = "visual Draw Shape primitive requires one exact typed Shape output";
                            return false;
                        }
                        consumed[(size_t) source] = true;
                    }
                    expectEdge((size_t) source, shapeEdge.fromPort, (size_t) terminal, 0);
                    consumed[(size_t) terminal] = true;
                    haveDrawShape = true;
                    const auto& primitive = plan.operations[(size_t) firstPrimitive];
                    const auto& draw = plan.operations[(size_t) terminal];
                    execution.drawShape = true;
                    execution.drawShapeNodeId = ids[(size_t) terminal];
                    execution.drawShapePrimaryNodeId = ids[(size_t) firstPrimitive];
                    execution.drawShapeSecondaryNodeId = secondPrimitive >= 0
                        ? ids[(size_t) secondPrimitive] : 0;
                    execution.drawShapeEllipse = kinds[(size_t) firstPrimitive] == "visual.shape.ellipse";
                    execution.drawShapeOperation = booleanOperation;
                    execution.drawShapeHasSecondary = secondPrimitive >= 0;
                    const auto parsePrimitive = [&] (const CompiledVisualOperation& operation,
                                                     float& cx, float& cy,
                                                     float& width, float& height)
                    {
                        return visualBoundedPayloadFloat(operation.payloadXml, "centerX", 0.5f,
                                                         -2.0f, 2.0f, cx)
                            && visualBoundedPayloadFloat(operation.payloadXml, "centerY", 0.5f,
                                                         -2.0f, 2.0f, cy)
                            && visualBoundedPayloadFloat(operation.payloadXml, "width", 1.0f,
                                                         0.0f, 4.0f, width)
                            && visualBoundedPayloadFloat(operation.payloadXml, "height", 1.0f,
                                                         0.0f, 4.0f, height);
                    };
                    if (! parsePrimitive(primitive, execution.shapeCx, execution.shapeCy,
                                         execution.shapeW, execution.shapeH))
                    {
                        error = "visual Draw Shape primitive payload is malformed or out of bounds";
                        return false;
                    }
                    if (secondPrimitive >= 0)
                    {
                        const auto& secondary = plan.operations[(size_t) secondPrimitive];
                        execution.drawShapeSecondaryEllipse = kinds[(size_t) secondPrimitive] == "visual.shape.ellipse";
                        if (! parsePrimitive(secondary, execution.shape2Cx, execution.shape2Cy,
                                             execution.shape2W, execution.shape2H))
                        {
                            error = "visual Draw Shape primitive payload is malformed or out of bounds";
                            return false;
                        }
                    }
                    if (! visualBoundedPayloadFloat(draw.payloadXml, "red", 1.0f,
                                                    0.0f, 1.0f, execution.shapeR)
                        || ! visualBoundedPayloadFloat(draw.payloadXml, "green", 1.0f,
                                                       0.0f, 1.0f, execution.shapeG)
                        || ! visualBoundedPayloadFloat(draw.payloadXml, "blue", 1.0f,
                                                       0.0f, 1.0f, execution.shapeB)
                        || ! visualBoundedPayloadFloat(draw.payloadXml, "alpha", 1.0f,
                                                       0.0f, 1.0f, execution.shapeA))
                    {
                        error = "visual Draw Shape color payload is malformed or out of bounds";
                        return false;
                    }
                }
                else
                {
                    error = "visual composite secondary branch is not an executable native source";
                    return false;
                }
                if (! recordInput(ids[(size_t) source], blendPort))
                {
                    error = "visual composite exceeds native input capacity";
                    return false;
                }
            }

            if (execution.compositeInputCount < 2
                || std::any_of(consumed.begin(), consumed.end(), [](bool value) { return ! value; })
                || expectedEdges.size() != plan.edges.size())
            {
                error = "visual composite contains disconnected operations or unsupported fan-in";
                return false;
            }
            std::vector<bool> matched(plan.edges.size(), false);
            for (const auto& expected : expectedEdges)
            {
                bool found = false;
                for (size_t i = 0; i < plan.edges.size(); ++i)
                    if (! matched[i] && plan.edges[i].fromNodeId == expected.fromNodeId
                        && plan.edges[i].fromPort == expected.fromPort
                        && plan.edges[i].toNodeId == expected.toNodeId
                        && plan.edges[i].toPort == expected.toPort)
                    {
                        matched[i] = true;
                        found = true;
                        break;
                    }
                if (! found)
                {
                    error = "visual composite exact edge bindings do not match native execution";
                    return false;
                }
            }
            return true;
        }
    }
    else
    {
        // Absence-only compatibility for snapshots produced before operation and
        // edge transport. Their already-admitted node order is the fixed chain.
        kinds = plan.nodeKinds;
    }

    if (kinds.empty() || kinds.back() != "video.out")
    {
        error = "visual layer plan has no executable output";
        return false;
    }
    const bool source = kinds.front() == "video.source"
                     || kinds.front() == "video.legacy.source"
                     || kinds.front() == "video.legacy.generator";
    if (! source)
    {
        error = "visual layer plan has no executable production source";
        return false;
    }

    size_t cursor = 1;
    const auto consume = [&](const char* kind, size_t& index)
    {
        if (index < kinds.size() && kinds[index] == kind) { ++index; return true; }
        return false;
    };
    const bool legacyChain = kinds.front().rfind ("video.legacy.", 0) == 0;
    if (consume ("video.legacy.retime", cursor)) {}
    execution.transform = consume (legacyChain ? "video.legacy.transform"
                                               : "video.transform", cursor);
    execution.effects = consume (legacyChain ? "video.legacy.effects"
                                             : "video.effects", cursor);
    visualtemporaloperation::Mode temporalMode {};
    if (! legacyChain && cursor < kinds.size()
        && visualtemporaloperation::modeForKind(kinds[cursor], temporalMode))
    {
        const auto operation = std::find_if(plan.operations.begin(), plan.operations.end(),
            [&kinds, cursor](const CompiledVisualOperation& value)
            { return value.kind == kinds[cursor]; });
        execution.feedback = true;
        visualtemporaloperation::Payload payload;
        if (operation == plan.operations.end()
            || ! visualtemporaloperation::parseForKind(operation->kind, operation->payloadXml,
                                                       payload, &error))
        {
            error = kinds[cursor] + " payload is malformed or out of bounds";
            return false;
        }
        execution.temporalNodeId = operation->nodeId;
        execution.temporalPayload = payload;
        execution.feedbackDecay = payload.decay;
        execution.feedbackZoom = payload.zoom;
        execution.feedbackSwirl = payload.swirl;
        ++cursor;
    }
    execution.mask = legacyChain;
    if (! legacyChain)
        execution.mask = consume ("video.mask.shape", cursor);
    consume ("video.blend", cursor); // existing compositor owns the admitted pass

    if (cursor + 1 != kinds.size() || kinds[cursor] != "video.out")
    {
        error = "visual layer plan contains an unsupported ordered production operation";
        return false;
    }
    if (execution.feedback && execution.effects)
    {
        error = "visual.feedback cannot be combined with video.effects until LayerDesc owns a merged immutable rack";
        return false;
    }
    if (std::count_if(kinds.begin(), kinds.end(), [](const std::string& kind)
        { visualtemporaloperation::Mode mode {}; return visualtemporaloperation::modeForKind(kind, mode); }) > 1)
    {
        error = "visual temporal execution supports exactly one production history pass per clip";
        return false;
    }
    return true;
}

inline bool compileVisualLayerExecution (const CompiledVisualLayerPlan& plan,
                                         VisualLayerExecution& execution,
                                         std::string& error)
{
    const bool typed = ! plan.nodeIds.empty() || ! plan.edges.empty()
                    || ! plan.ports.empty() || ! plan.operations.empty();
    if (! typed || (! isExactThreeSourceVisualDag(plan.nodeKinds)
                    && ! isBoundedNativeCompositorDag(plan.nodeKinds)))
        return compileVisualLayerExecutionOrdered(plan, execution, error);

    CompiledVisualDagSchedule schedule;
    if (! compileBoundedVisualDagSchedule(plan, schedule, error)) return false;

    CompiledVisualLayerPlan ordered = plan;
    ordered.nodeIds.clear();
    ordered.nodeKinds.clear();
    ordered.operations.clear();
    ordered.nodeIds.reserve(schedule.operations.size());
    ordered.nodeKinds.reserve(schedule.operations.size());
    ordered.operations.reserve(schedule.operations.size());
    for (const auto& scheduled : schedule.operations)
    {
        ordered.nodeIds.push_back(plan.nodeIds[scheduled.nodeIndex]);
        ordered.nodeKinds.push_back(plan.nodeKinds[scheduled.nodeIndex]);
        ordered.operations.push_back(plan.operations[scheduled.operationIndex]);
    }
    if (! compileVisualLayerExecutionOrdered(ordered, execution, error)) return false;
    execution.dagSchedule = std::move(schedule);
    return true;
}

inline bool visualPlanContainsDirectedCycle (const CompiledVisualLayerPlan& plan)
{
    if (plan.nodeIds.empty()) return false;
    std::vector<size_t> indegree(plan.nodeIds.size(), 0);
    const auto nodeIndex = [&plan] (int nodeId) -> size_t
    {
        const auto found = std::find(plan.nodeIds.begin(), plan.nodeIds.end(), nodeId);
        return found == plan.nodeIds.end() ? plan.nodeIds.size()
                                           : static_cast<size_t>(found - plan.nodeIds.begin());
    };
    for (const auto& edge : plan.edges)
    {
        const auto from = nodeIndex(edge.fromNodeId);
        const auto to = nodeIndex(edge.toNodeId);
        if (from < plan.nodeIds.size() && to < plan.nodeIds.size()) ++indegree[to];
    }
    std::vector<bool> emitted(plan.nodeIds.size(), false);
    size_t emittedCount = 0;
    for (;;)
    {
        size_t ready = plan.nodeIds.size();
        for (size_t i = 0; i < plan.nodeIds.size(); ++i)
            if (! emitted[i] && indegree[i] == 0) { ready = i; break; }
        if (ready == plan.nodeIds.size()) break;
        emitted[ready] = true;
        ++emittedCount;
        for (const auto& edge : plan.edges)
            if (nodeIndex(edge.fromNodeId) == ready)
            {
                const auto to = nodeIndex(edge.toNodeId);
                if (to < indegree.size() && indegree[to] != 0) --indegree[to];
            }
    }
    return emittedCount != plan.nodeIds.size();
}

inline bool admitTemporalSamplingResource (
    const CompiledVisualLayerPlan& plan, int canvasWidth, int canvasHeight,
    const VisualBackendResourceLimits& backendLimits,
    std::optional<TemporalSamplingPass>& admitted, std::string& error)
{
    admitted.reset();
    const auto operation = std::find_if(plan.operations.begin(), plan.operations.end(), [] (const auto& candidate)
        { visualtemporalsampling::Mode mode {}; return visualtemporalsampling::modeForKind(candidate.kind, mode); });
    if (operation == plan.operations.end()) return true;
    if (operation->backendCapability != "temporal-sampling")
    {
        error = "temporal sampling operation requires sampling-only admission";
        return false;
    }
    visualtemporalsampling::Payload payload;
    if (! visualtemporalsampling::parseForKind(operation->kind, operation->payloadXml, payload, &error))
        return false;
    visualtemporalsampling::ResourceLimits limits;
    limits.maximumImageDimension = backendLimits.maximumImageDimension;
    limits.maximumHistoryBytes = backendLimits.allocatedFrameBytes;
    limits.maximumTransientBytes = backendLimits.allocatedFrameBytes;
    TemporalSamplingPass pass;
    pass.clipId = plan.clipId;
    pass.nodeId = operation->nodeId;
    pass.structuralRevision = plan.structuralRevision;
    pass.payload = payload;
    pass.width = static_cast<uint32_t>(canvasWidth);
    pass.height = static_cast<uint32_t>(canvasHeight);
    if (canvasWidth <= 0 || canvasHeight <= 0
        || ! visualtemporalsampling::admitResources(payload, pass.width, pass.height,
                                                    limits, pass.footprint, error))
        return false;
    admitted = pass;
    return true;
}

inline bool admitTemporalFeedbackResource (
    const CompiledVisualLayerPlan& plan, int canvasWidth, int canvasHeight,
    const VisualBackendResourceLimits& backendLimits,
    std::optional<TemporalFeedbackPass>& admitted, std::string& error)
{
    admitted.reset();
    const auto first = std::find_if(plan.operations.begin(), plan.operations.end(),
        [] (const auto& operation)
        { visualtemporaloperation::Mode mode {}; return visualtemporaloperation::modeForKind(operation.kind, mode); });
    if (first == plan.operations.end()) return true;
    if (std::count_if(plan.operations.begin(), plan.operations.end(), [] (const auto& operation)
        { visualtemporaloperation::Mode mode {}; return visualtemporaloperation::modeForKind(operation.kind, mode); }) != 1)
    {
        error = "visual temporal execution supports exactly one production history pass per clip";
        return false;
    }
    if (first->backendCapability != "native-gpu")
    {
        error = "visual temporal operation requires the native GPU history backend";
        return false;
    }
    if (canvasWidth <= 0 || canvasHeight <= 0
        || static_cast<uint64_t>(canvasWidth) > std::numeric_limits<uint32_t>::max()
        || static_cast<uint64_t>(canvasHeight) > std::numeric_limits<uint32_t>::max())
    {
        error = "temporal resource requires a positive explicit image extent";
        return false;
    }

    visualtemporaloperation::Payload payload;
    if (! visualtemporaloperation::parseForKind(first->kind, first->payloadXml, payload, &error))
        return false;
    videotemporal::Mode resourceMode {};
    switch (payload.mode)
    {
        case visualtemporaloperation::Mode::frameDelay:
            resourceMode = videotemporal::Mode::frameDelay;
            break;
        case visualtemporaloperation::Mode::feedback:
            resourceMode = videotemporal::Mode::feedback;
            break;
        case visualtemporaloperation::Mode::echo:
            resourceMode = videotemporal::Mode::echo;
            break;
        case visualtemporaloperation::Mode::stutter:
            resourceMode = videotemporal::Mode::stutter;
            break;
        case visualtemporaloperation::Mode::longExposure:
            resourceMode = videotemporal::Mode::longExposure;
            break;
    }
    videotemporal::ResourceLimits limits;
    limits.maximumImageDimension = backendLimits.maximumImageDimension > 0
        ? static_cast<uint32_t>(std::min<uint64_t>(
            static_cast<uint64_t>(backendLimits.maximumImageDimension),
            std::numeric_limits<uint32_t>::max())) : 0;
    limits.maximumHistoryLength = visualtemporaloperation::kMaximumHistoryLength;
    limits.maximumHistoryBytes = backendLimits.allocatedFrameBytes;
    limits.maximumTransientBytes = backendLimits.allocatedFrameBytes;
    const videotemporal::ResourceContract contract(
        resourceMode, videotemporal::PixelFormat::rgba16f,
        { static_cast<uint32_t>(canvasWidth), static_cast<uint32_t>(canvasHeight) },
        payload.historyLength);
    videotemporal::ResourceFootprint footprint;
    if (! videotemporal::admitResource(contract, limits,
                                       visualPlanContainsDirectedCycle(plan), footprint, error))
        return false;
    TemporalResourcePass pass;
    pass.clipId = plan.clipId;
    pass.nodeId = first->nodeId;
    pass.structuralRevision = plan.structuralRevision;
    pass.payload = payload;
    pass.extent = contract.extent();
    pass.format = contract.format();
    pass.historyLength = contract.historyLength();
    pass.footprint = footprint;
    admitted.emplace(std::move(pass));
    return true;
}

inline bool checkedAccumulateVisualPlanBytes (
    uint64_t& total, uint64_t first, uint64_t second) noexcept
{
    if (first > std::numeric_limits<uint64_t>::max() - second)
        return false;
    const auto increment = first + second;
    if (total > std::numeric_limits<uint64_t>::max() - increment)
        return false;
    total += increment;
    return true;
}

inline bool VisualPlanExecutionState::admitPlans (
    const std::vector<CompiledVisualLayerPlan>& plans, std::string* diagnostic,
    int canvasWidth, int canvasHeight, const VisualBackendResourceLimits* backendLimits)
{
    const auto normalizedPlanIndices = normalizedVisualLayerPlanIndices(plans);
    if (normalizedPlanIndices.size() > kMaxAdmittedPlans)
    {
        const std::string reason = "visual execution admission exceeds fixed plan capacity";
        telemetry().recordFailedLowering(reason);
        if (diagnostic != nullptr) *diagnostic = reason;
        return false;
    }
    const auto derivedLimits = VisualBackendResourceLimits::forCanvas(canvasWidth, canvasHeight);
    const auto& limits = backendLimits != nullptr ? *backendLimits : derivedLimits;
    uint64_t totalAllocatedFrameBytes = 0;
    uint64_t totalAllocatedFrameSlots = 0;
    uint64_t totalPeakLiveFrames = 0;
    uint64_t totalDescriptors = 0;
    uint64_t totalOperations = 0;
    uint64_t totalSceneRecords = 0;
    uint64_t totalFrameOutputs = 0;
    for (const auto planIndex : normalizedPlanIndices)
    {
        const auto& plan = plans[planIndex];
        VisualPlanResourceUsage usage;
        std::string resourceError;
        std::optional<TemporalFeedbackPass> temporalFeedback;
        std::optional<TemporalSamplingPass> temporalSampling;
        if (! admitTemporalSamplingResource(plan, canvasWidth, canvasHeight, limits,
                                            temporalSampling, resourceError)
            || ! admitTemporalFeedbackResource(plan, canvasWidth, canvasHeight, limits,
                                            temporalFeedback, resourceError))
        {
            telemetry().recordFailedLowering(resourceError);
            if (diagnostic != nullptr) *diagnostic = resourceError;
            return false;
        }
        const bool directFixtureWithoutProducerAccounting = plan.descriptorCount == 0
            && plan.operationCount == 0 && plan.sceneRecordCount == 0
            && plan.frameOutputCount == 0 && plan.peakLiveFrameCount == 0
            && plan.allocatedFrameSlotCount == 0;
        if (! directFixtureWithoutProducerAccounting
            && ! admitVisualPlanResources(plan, canvasWidth, canvasHeight, limits, usage, resourceError))
        {
            telemetry().recordFailedLowering(resourceError);
            if (diagnostic != nullptr) *diagnostic = resourceError;
            return false;
        }
        if (! directFixtureWithoutProducerAccounting)
        {
            if (! checkedAccumulateVisualPlanBytes(
                    totalAllocatedFrameBytes, usage.allocatedFrameBytes, 0))
            {
                const std::string reason = "visual graph allocated frame byte accounting overflow";
                telemetry().recordFailedLowering(reason);
                if (diagnostic != nullptr) *diagnostic = reason;
                return false;
            }
            if (! checkedAccumulateVisualPlanBytes(totalAllocatedFrameSlots, usage.frameSlots, 0)
                || ! checkedAccumulateVisualPlanBytes(totalPeakLiveFrames, usage.peakLiveFrames, 0)
                || ! checkedAccumulateVisualPlanBytes(totalDescriptors, usage.descriptors, 0)
                || ! checkedAccumulateVisualPlanBytes(totalOperations, usage.operations, 0)
                || ! checkedAccumulateVisualPlanBytes(totalSceneRecords, usage.sceneRecords, 0)
                || ! checkedAccumulateVisualPlanBytes(totalFrameOutputs, usage.frameOutputs, 0))
            {
                const std::string reason = "visual graph aggregate resource accounting overflow";
                telemetry().recordFailedLowering(reason);
                if (diagnostic != nullptr) *diagnostic = reason;
                return false;
            }
        }
        if (temporalSampling.has_value())
        {
            if (! checkedAccumulateVisualPlanBytes(
                    totalAllocatedFrameBytes,
                    temporalSampling->footprint.historyBytes,
                    temporalSampling->footprint.transientBytes))
            {
                const std::string reason = "temporal sampling byte accounting overflow";
                telemetry().recordFailedLowering(reason);
                if (diagnostic != nullptr) *diagnostic = reason;
                return false;
            }
            const auto temporalImages = visualtemporalsampling::requirements(
                temporalSampling->payload.mode).retainedHistoryImages
                + visualtemporalsampling::requirements(temporalSampling->payload.mode).transientImages;
            if (! checkedAccumulateVisualPlanBytes(totalAllocatedFrameSlots, temporalImages, 0)
                || ! checkedAccumulateVisualPlanBytes(totalPeakLiveFrames, temporalImages, 0))
            {
                const std::string reason = "temporal sampling image accounting overflow";
                telemetry().recordFailedLowering(reason);
                if (diagnostic != nullptr) *diagnostic = reason;
                return false;
            }
        }
        if (temporalFeedback.has_value())
        {
            const auto historyBytes = temporalFeedback->footprint.historyBytes;
            const auto transientBytes = temporalFeedback->footprint.transientBytes;
            if (! checkedAccumulateVisualPlanBytes(
                    totalAllocatedFrameBytes, historyBytes, transientBytes))
            {
                const std::string reason = "temporal resource byte accounting overflow";
                telemetry().recordFailedLowering(reason);
                if (diagnostic != nullptr) *diagnostic = reason;
                return false;
            }

            const auto temporalImages = temporalFeedback->footprint.retainedImages
                + temporalFeedback->footprint.transientImages;
            if (! checkedAccumulateVisualPlanBytes(totalAllocatedFrameSlots, temporalImages, 0)
                || ! checkedAccumulateVisualPlanBytes(totalPeakLiveFrames, temporalImages, 0))
            {
                const std::string reason = "temporal feedback image accounting overflow";
                telemetry().recordFailedLowering(reason);
                if (diagnostic != nullptr) *diagnostic = reason;
                return false;
            }
        }
    }
    const auto rejectAggregate = [&] (const char* resource, uint64_t used, uint64_t limit)
    {
        const std::string reason = std::string("visual graph aggregate ") + resource
            + " capacity exceeded: " + std::to_string(used) + " > " + std::to_string(limit);
        telemetry().recordFailedLowering(reason);
        if (diagnostic != nullptr) *diagnostic = reason;
        return false;
    };
    if (totalDescriptors > limits.descriptors)
        return rejectAggregate("descriptor", totalDescriptors, limits.descriptors);
    if (totalOperations > limits.operations)
        return rejectAggregate("operation", totalOperations, limits.operations);
    if (totalSceneRecords > limits.sceneRecords)
        return rejectAggregate("scene record", totalSceneRecords, limits.sceneRecords);
    if (totalFrameOutputs > limits.frameOutputs)
        return rejectAggregate("frame output", totalFrameOutputs, limits.frameOutputs);
    if (totalAllocatedFrameSlots > limits.frameSlots)
    {
        const std::string reason = "visual graph allocated frame slot capacity exceeded: "
            + std::to_string(totalAllocatedFrameSlots) + " > "
            + std::to_string(limits.frameSlots);
        telemetry().recordFailedLowering(reason);
        if (diagnostic != nullptr) *diagnostic = reason;
        return false;
    }
    if (totalPeakLiveFrames > limits.liveFrames)
    {
        const std::string reason = "visual graph live frame capacity exceeded: "
            + std::to_string(totalPeakLiveFrames) + " > "
            + std::to_string(limits.liveFrames);
        telemetry().recordFailedLowering(reason);
        if (diagnostic != nullptr) *diagnostic = reason;
        return false;
    }
    if (totalAllocatedFrameBytes > limits.allocatedFrameBytes)
    {
        const std::string reason = "visual graph allocated frame byte capacity exceeded: "
            + std::to_string(totalAllocatedFrameBytes) + " > "
            + std::to_string(limits.allocatedFrameBytes);
        telemetry().recordFailedLowering(reason);
        if (diagnostic != nullptr) *diagnostic = reason;
        return false;
    }
    std::array<Slot, kMaxAdmittedPlans> next {};
    struct LoweringSample { uint64_t durationNs = 0; bool installed = false; };
    std::vector<LoweringSample> loweringSamples;
    loweringSamples.reserve(normalizedPlanIndices.size());
    size_t count = 0;
    bool complete = true;
    for (const auto planIndex : normalizedPlanIndices)
    {
        const auto& plan = plans[planIndex];
        auto duplicate = std::find_if(next.begin(), next.begin() + (ptrdiff_t) count,
            [&](const Slot& slot) { return slot.clipId == plan.clipId; });
        if (duplicate != next.begin() + (ptrdiff_t) count
            && duplicate->structuralRevision >= plan.structuralRevision)
            continue;
        if (duplicate == next.begin() + (ptrdiff_t) count)
        {
            if (count == kMaxAdmittedPlans)
            {
                complete = false;
                telemetry().recordFailedLowering("visual execution admission exceeds fixed plan capacity");
                if (diagnostic != nullptr && diagnostic->empty())
                    *diagnostic = "visual execution admission exceeds fixed plan capacity";
                continue;
            }
            duplicate = next.begin() + (ptrdiff_t) count++;
        }
        Slot slot;
        slot.clipId = plan.clipId;
        slot.structuralRevision = plan.structuralRevision;
        slot.owner.structuralRevision = plan.structuralRevision;
        std::string loweringError;
        const auto begin = std::chrono::steady_clock::now();
        slot.lowered = compileVisualLayerExecution(plan, slot.execution, loweringError);
        if (slot.lowered
            && ! admitTemporalSamplingResource(plan, canvasWidth, canvasHeight, limits,
                                                slot.execution.temporalSamplingPass,
                                                loweringError))
            slot.lowered = false;
        if (slot.lowered
            && ! admitTemporalFeedbackResource(plan, canvasWidth, canvasHeight, limits,
                                                slot.execution.temporalFeedbackPass,
                                                loweringError))
            slot.lowered = false;
        if (slot.lowered && slot.execution.colorAovPass.has_value()
            && slot.execution.colorAovPass->extent != renderpassoutput::Extent {
                static_cast<std::uint32_t> (canvasWidth),
                static_cast<std::uint32_t> (canvasHeight) })
        {
            slot.lowered = false;
            loweringError = "Color AOV extent does not match the admitted canvas";
        }
        if (slot.lowered && slot.execution.motionAovPass.has_value()
            && slot.execution.motionAovPass->extent != renderpassoutput::Extent {
                static_cast<std::uint32_t> (canvasWidth),
                static_cast<std::uint32_t> (canvasHeight) })
        {
            slot.lowered = false;
            loweringError = "Motion AOV extent does not match the admitted canvas";
        }
        if (slot.lowered && slot.execution.aovInspectionPass.has_value()
            && slot.execution.aovInspectionPass->extent != renderpassoutput::Extent {
                static_cast<std::uint32_t> (canvasWidth),
                static_cast<std::uint32_t> (canvasHeight) })
        {
            slot.lowered = false;
            loweringError = "AOV inspection extent does not match the admitted canvas";
        }
        const auto loweringNs = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - begin).count());
        loweringSamples.push_back({ loweringNs, slot.lowered });
        if (! slot.lowered)
        {
            complete = false;
            telemetry().recordFailedLowering(loweringError);
            if (diagnostic != nullptr && diagnostic->empty()) *diagnostic = loweringError;
        }
        // Temporal ownership is published separately by the synchronized
        // generation authority after the complete replacement is ready.
        *duplicate = slot;
    }
    if (! complete)
    {
        for (const auto& sample : loweringSamples)
            telemetry().recordPlanLowering(false, sample.durationNs, sample.installed);
        return false;
    }
    std::vector<VisualTelemetryPlanAdmission> telemetryAdmissions;
    telemetryAdmissions.reserve(normalizedPlanIndices.size());
    for (const auto planIndex : normalizedPlanIndices)
        telemetryAdmissions.push_back(makeVisualTelemetryAdmission(plans[planIndex]));
    if (! telemetry().admitPlans(telemetryAdmissions, diagnostic))
        return false;
    for (const auto& sample : loweringSamples)
        telemetry().recordPlanLowering(false, sample.durationNs, sample.installed);
    if (totalAllocatedFrameBytes != 0)
        telemetry().recordResources(totalPeakLiveFrames, totalAllocatedFrameSlots,
                                    totalAllocatedFrameBytes);
    std::sort(next.begin(), next.begin() + (ptrdiff_t) count,
        [](const Slot& a, const Slot& b) { return a.clipId < b.clipId; });
    {
        const auto authority = temporalAuthority_;
        std::lock_guard<std::mutex> lock(authority->mutex);
        std::vector<TemporalRecord> records;
        records.reserve(count);
        for (size_t index = 0; index < count; ++index)
        {
            TemporalRecord record;
            record.clipId = next[index].clipId;
            record.structuralRevision = next[index].structuralRevision;
            record.owner = Owner { record.structuralRevision };
            const auto previous = findTemporalRecord(*authority, record.clipId);
            if (previous != authority->records.end())
            {
                if (previous->structuralRevision == record.structuralRevision)
                    record = *previous;
                else
                    record.samplingLifecycle = previous->samplingLifecycle;
            }
            records.push_back(std::move(record));
        }
        slots_ = next;
        slotCount_ = count;
        authority->records = std::move(records);
        ++authority->generation;
    }
    resourceReceipt_ = {
        count, static_cast<size_t>(totalDescriptors), static_cast<size_t>(totalOperations),
        static_cast<size_t>(totalSceneRecords), static_cast<size_t>(totalFrameOutputs),
        static_cast<size_t>(totalPeakLiveFrames),
        static_cast<size_t>(totalAllocatedFrameSlots), totalAllocatedFrameBytes
    };
    budgetReceipt_ = { canvasWidth, canvasHeight, limits, resourceReceipt_ };
    telemetry().recordBudgetReceipt(budgetReceipt_);
    return true;
}

inline bool visualPlansUseTemporalFeedback (
    const std::vector<CompiledVisualLayerPlan>& plans)
{
    const auto selected = normalizedVisualLayerPlanIndices(plans);
    return std::any_of(selected.begin(), selected.end(), [&plans] (std::size_t index)
    {
        const auto& plan = plans[index];
        return std::any_of(plan.operations.begin(), plan.operations.end(), [] (const auto& operation)
            { visualtemporaloperation::Mode mode {}; return visualtemporaloperation::modeForKind(operation.kind, mode); });
    });
}

struct VisualInspectionTarget
{
    int clipId = -1;
    uint64_t structuralRevision = 0;
    int nodeId = 0;
    int outputPort = -1;
};

// A borrowed backend-native image produced while lowering one layer. The
// renderer remains the owner; this record pins the exact handle for only the
// current frame/generation and never maps or copies its pixels.
struct VisualInspectionResource
{
    unsigned handle = 0;
    int width = 0;
    int height = 0;
    int nodeId = 0;
    int outputPort = -1;
};

enum class VisualInspectionSlice
{
    unsupported,
    decodedSource,
    parkedDepthMetadata,
    transformedLayer,
    retainedDrawShape,
    terminalComposite
};

inline VisualInspectionSlice classifyVisualInspectionTarget (
    const CompiledVisualLayerPlan& plan, const VisualInspectionTarget& target)
{
    const auto operation = std::find_if(plan.operations.begin(), plan.operations.end(),
        [&target](const CompiledVisualOperation& value) { return value.nodeId == target.nodeId; });
    if (operation == plan.operations.end()) return VisualInspectionSlice::unsupported;
    const auto output = std::find_if(plan.operations.begin(), plan.operations.end(),
        [](const CompiledVisualOperation& value) { return value.kind == "video.out"; });
    const bool terminal = output != plan.operations.end()
        && std::any_of(plan.edges.begin(), plan.edges.end(), [&](const CompiledVisualEdgeBinding& edge)
        { return edge.fromNodeId == target.nodeId && edge.toNodeId == output->nodeId; });
    // FrameRenderer exposes its genuine post-geometry layer target. Admit only
    // the exact source->transform->out shape: later operations would make the
    // fixed renderer boundary a different semantic resource.
    if (operation->kind == "video.transform" && operation->backendCapability == "native-gpu"
        && terminal && plan.operations.size() == 3
        && plan.operations.front().kind == "video.source"
        && plan.operations.back().kind == "video.out")
        return VisualInspectionSlice::transformedLayer;
    // visual.draw.shape materializes its image in a dedicated native render pass
    // immediately before Blend. This is the exact operation boundary, not a
    // reconstruction from the terminal composite.
    if (operation->kind == "visual.draw.shape"
        && operation->backendCapability == "native-gpu"
        && std::any_of(plan.edges.begin(), plan.edges.end(), [&](const CompiledVisualEdgeBinding& edge)
        {
            const auto consumer = std::find_if(plan.operations.begin(), plan.operations.end(),
                [&](const CompiledVisualOperation& value) { return value.nodeId == edge.toNodeId; });
            return edge.fromNodeId == target.nodeId && edge.fromPort == target.outputPort
                && consumer != plan.operations.end() && consumer->kind == "video.blend";
        }))
        return VisualInspectionSlice::retainedDrawShape;
    if (terminal) return VisualInspectionSlice::terminalComposite;
    // FrameRenderer receives decoded video.source as LayerDesc::texture. That
    // is the only exact non-terminal operation output exposed at this boundary.
    if (operation->kind == "video.source" && operation->backendCapability == "source-decode")
        return VisualInspectionSlice::decodedSource;
    if (operation->kind == "visual.depth.asset" && operation->backendCapability == "parked-metadata")
        return VisualInspectionSlice::parkedDepthMetadata;
    return VisualInspectionSlice::unsupported;
}

inline bool validateVisualInspectionTarget (
    const std::vector<CompiledVisualLayerPlan>& plans,
    const VisualInspectionTarget& target, std::string& error)
{
    const auto* plan = findVisualLayerPlan(plans, target.clipId);
    if (plan == nullptr)
    {
        error = "inspection target has no compiled visual plan";
        return false;
    }
    if (plan->structuralRevision != target.structuralRevision)
    {
        error = "inspection target revision is stale";
        return false;
    }
    const auto operation = std::find_if(plan->operations.begin(), plan->operations.end(),
        [&target](const CompiledVisualOperation& value) { return value.nodeId == target.nodeId; });
    if (operation == plan->operations.end())
    {
        error = "inspection target node is not executable";
        return false;
    }
    const auto port = std::find_if(plan->ports.begin(), plan->ports.end(),
        [&target](const CompiledVisualPortBinding& value)
        {
            return value.nodeId == target.nodeId && value.port == target.outputPort;
        });
    if (port == plan->ports.end() || port->direction != "out" || port->carrier != "frame")
    {
        error = "inspection target is not an image output port";
        return false;
    }
    const auto slice = classifyVisualInspectionTarget(*plan, target);
    if (slice == VisualInspectionSlice::parkedDepthMetadata)
    {
        error = "depth is parked metadata pending a depth-consuming execution seam; inspection is unavailable";
        return false;
    }
    if (slice == VisualInspectionSlice::unsupported)
    {
        error = "inspection target has no retainable GPU output resource";
        return false;
    }
    return true;
}

inline void seedFlatShaderRuntimeParameters(
    const VisualLayerExecution* execution,
    std::map<std::string, double>& parameters)
{
    if (execution == nullptr || !execution->flatShaderBridge
        || execution->shaderOperationPlan == nullptr)
        return;
    for (const auto& operation : execution->shaderOperationPlan->operations)
    {
        const auto alias = shadercatalog::runtimeNodeAlias(operation.nodeId);
        if (alias.empty()) continue;
        for (const auto& [componentId, value] : operation.generatedParameters)
            parameters.emplace(alias + "/" + componentId, value);
    }
}

inline std::uint64_t runtimeDeviceGeneration (
    const videohelper::geometry::BackendCapabilities& capabilities) noexcept
{
    std::uint64_t digest = 1469598103934665603ull;
    for (const auto character : capabilities.backendIdentity + "\n" + capabilities.deviceIdentity)
    {
        digest ^= static_cast<unsigned char> (character);
        digest *= 1099511628211ull;
    }
    return digest == 0 ? 1 : digest;
}

inline std::uint64_t geometryCoreHelperGeneration() noexcept
{
    static const auto generation = []
    {
        const auto ticks = static_cast<std::uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
        const auto address = reinterpret_cast<std::uintptr_t>(&runtimeDeviceGeneration);
        const auto value = ticks ^ static_cast<std::uint64_t>(address);
        return value == 0 ? std::uint64_t{1} : value;
    }();
    return generation;
}

inline std::uint64_t geometryCoreDeviceGeneration()
{
    videohelper::geometry::NativeGeometryCoreCapabilitySource source(
        arbitgpu::nativeFixtureSceneBackend());
    return runtimeDeviceGeneration(source.geometryCoreCapabilities());
}

template <typename LayerDesc>
inline bool admitGeometryCoreOperations(const CompiledVisualLayerPlan& plan,
                                        videohelper::geometry::PlanUse use,
                                        const VisualPlanEvaluationContext* evaluationContext,
                                        int width, int height, LayerDesc& layer,
                                        std::string& error)
{
    const CompiledVisualOperation* geometryOperation = nullptr;
    for(const auto& operation:plan.operations)
        if(operation.kind=="geometry.core.runtime")
        {
            if(geometryOperation != nullptr)
            {
                error="multiple Geometry Core image outputs require an explicit compositor";
                return false;
            }
            geometryOperation=&operation;
        }
    if(geometryOperation == nullptr) return true;
    if(evaluationContext == nullptr || evaluationContext->projectGeneration == 0
        || evaluationContext->helperGeneration == 0
        || evaluationContext->deviceGeneration == 0)
    {
        error="Geometry Core requires exact project, helper, and device generations";
        return false;
    }
    if(plan.clipId <= 0)
    {
        error="Geometry Core requires a stable positive clip identity";
        return false;
    }
    if(width<=0||height<=0)
    {
        error="Geometry Core requires the compositor output extent";
        return false;
    }
    static videohelper::geometry::NativeGeometryCoreCapabilitySource source(arbitgpu::nativeFixtureSceneBackend());
    static videohelper::geometry::GeometryCorePlanRuntime runtime(source);
    struct RetainedExecutionCache final
    {
        struct Entry final
        {
            std::uint64_t lastUse = 0;
            std::shared_ptr<const videohelper::geometry::AdmittedPlanValue> admittedPlan;
            std::shared_ptr<const videohelper::geometry::NativeGeometryExecution> execution;
            bool reserved = false;
        };
        std::mutex mutex;
        std::map<videohelper::geometry::PlanOwnerIdentity, Entry> owners;
        std::uint64_t clock = 0;
        std::uint64_t projectGeneration = 0;
        std::uint64_t helperGeneration = 0;
        std::uint64_t deviceGeneration = 0;
    };
    static RetainedExecutionCache retained;
    if(!runtime.reconcileGeneration(evaluationContext->projectGeneration,
                                    evaluationContext->helperGeneration,
                                    evaluationContext->deviceGeneration,error))
        return false;
    {
        std::lock_guard<std::mutex> lock(retained.mutex);
        const bool changed=retained.projectGeneration!=evaluationContext->projectGeneration
            || retained.helperGeneration!=evaluationContext->helperGeneration
            || retained.deviceGeneration!=evaluationContext->deviceGeneration;
        for(auto it=retained.owners.begin();it!=retained.owners.end();)
        {
            const bool current=it->first.projectGeneration==evaluationContext->projectGeneration
                && it->first.helperGeneration==evaluationContext->helperGeneration
                && it->first.deviceGeneration==evaluationContext->deviceGeneration;
            if(!current && !it->second.reserved && it->second.execution
                && it->second.execution.use_count()==1)
                it=retained.owners.erase(it);
            else
                ++it;
        }
        if(changed)
        {
            retained.projectGeneration=evaluationContext->projectGeneration;
            retained.helperGeneration=evaluationContext->helperGeneration;
            retained.deviceGeneration=evaluationContext->deviceGeneration;
            retained.clock=0;
        }
    }
    videowire::geometry::AdmissionContext context;
    const auto admitPass=[&](bool instances)
    {
        for(const auto* operation : { geometryOperation })
        {
            auto bytes=videowire::geometry::decodeLoweredPlanText(operation->payloadXml,error);
            if(!bytes) return false;
            auto lowered=videowire::geometry::decodeLoweredRuntimePlan(*bytes,error);
            if(!lowered) return false;
            if((lowered->contract.carrier==videowire::geometry::CarrierKind::instances3D)!=instances) continue;
            // The owner is part of admission, rather than a cache hint.  A
            // plan cannot cross a project revision, helper lifetime, device
            // binding, or preview/export owner boundary.
            const videohelper::geometry::PlanOwnerIdentity owner {
                evaluationContext->projectGeneration, evaluationContext->helperGeneration,
                evaluationContext->deviceGeneration,
                static_cast<std::uint64_t>(plan.clipId), plan.structuralRevision, use };
            const auto admitted=use==videohelper::geometry::PlanUse::preview
                ? runtime.admitPreview(owner,*bytes,videowire::geometry::ResourceLimits{},context,error)
                : runtime.admitExport(owner,*bytes,videowire::geometry::ResourceLimits{},context,error);
            if(!admitted) return false;
            {
                std::lock_guard<std::mutex> lock(retained.mutex);
                const auto existing=retained.owners.find(owner);
                if(existing!=retained.owners.end() && !existing->second.reserved
                    && existing->second.admittedPlan == admitted->plan
                    && existing->second.execution->frame->width()==static_cast<std::uint32_t>(width)
                    && existing->second.execution->frame->height()==static_cast<std::uint32_t>(height))
                {
                    existing->second.lastUse=++retained.clock;
                    const auto& frame=existing->second.execution->frame;
                    const auto view=frame->colorTextureViewHandle();
                    if(frame->backend()=="opengl"
                        && view>std::numeric_limits<unsigned>::max())
                    {
                        error="Geometry Core OpenGL texture view exceeds compositor width";
                        return false;
                    }
                    layer.texture=frame->backend()=="opengl"
                        ? static_cast<unsigned>(view):0;
                    layer.nativeTextureBackend=frame->backend();
                    layer.nativeTextureView=view;
                    layer.nativeTextureDescriptor=frame->colorTextureDescriptor();
                    layer.nativeTextureOwner=existing->second.execution;
                    layer.texWidth=width;
                    layer.texHeight=height;
                    context.admittedGeometryOrSceneSources.insert(
                        admitted->plan->value().descriptor().stableId);
                    continue;
                }
                if(existing!=retained.owners.end())
                {
                    if(existing->second.reserved || existing->second.execution.use_count()!=1)
                    {
                        error="Geometry Core cannot replace a live compositor lease at a new extent";
                        return false;
                    }
                    retained.owners.erase(existing);
                }
                if(retained.owners.size()>=32)
                {
                    auto victim=retained.owners.end();
                    for(auto candidate=retained.owners.begin();candidate!=retained.owners.end();++candidate)
                        if(!candidate->second.reserved
                            && candidate->second.execution.use_count()==1
                            && (victim==retained.owners.end()
                                || candidate->second.lastUse<victim->second.lastUse))
                            victim=candidate;
                    if(victim==retained.owners.end())
                    {
                        error="Geometry Core retained frame cache is full with live compositor leases";
                        return false;
                    }
                    retained.owners.erase(victim);
                }
                // Reserve admission before native decode/allocation/draw. A full
                // live cache therefore rejects without touching backend memory.
                retained.owners.emplace(owner,typename RetainedExecutionCache::Entry{
                    ++retained.clock,admitted->plan,{},true});
            }
            auto execution = videohelper::geometry::executeNativeGeometry(
                arbitgpu::nativeFixtureSceneBackend(), *admitted,
                static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height), error);
            if(!execution)
            {
                std::lock_guard<std::mutex> lock(retained.mutex);
                retained.owners.erase(owner);
                return false;
            }
            {
                std::lock_guard<std::mutex> lock(retained.mutex);
                const auto retainedExecution =
                    std::make_shared<const videohelper::geometry::NativeGeometryExecution>(
                        std::move(*execution));
                const auto inserted=retained.owners.find(owner);
                if(inserted==retained.owners.end() || !inserted->second.reserved)
                {
                    error="Geometry Core retained reservation was invalidated before publication";
                    return false;
                }
                inserted->second.execution=retainedExecution;
                inserted->second.reserved=false;
                inserted->second.lastUse=++retained.clock;
                const auto& frame=inserted->second.execution->frame;
                const auto view=frame->colorTextureViewHandle();
                if(frame->backend()=="opengl"
                    && view>std::numeric_limits<unsigned>::max())
                {
                    retained.owners.erase(inserted);
                    error="Geometry Core OpenGL texture view exceeds compositor width";
                    return false;
                }
                layer.texture=frame->backend()=="opengl"
                    ? static_cast<unsigned>(view):0;
                layer.nativeTextureBackend=frame->backend();
                layer.nativeTextureView=view;
                layer.nativeTextureDescriptor=frame->colorTextureDescriptor();
                layer.nativeTextureOwner=inserted->second.execution;
                layer.texWidth=width;
                layer.texHeight=height;
            }
            context.admittedGeometryOrSceneSources.insert(admitted->plan->value().descriptor().stableId);
        }
        return true;
    };
    return admitPass(false)&&admitPass(true);
}

const VisualLayerExecution* findAdmittedVisualLayerExecution (
    VisualPlanExecutionState& state, int clipId, uint64_t structuralRevision,
    double evaluationTimeSec, bool& pausedHold);

template <typename LayerDesc>
inline bool executeVisualLayerPlan (const std::vector<CompiledVisualLayerPlan>& plans,
                                    int clipId, LayerDesc& layer, std::string& error,
                                    videohelper::geometry::PlanUse geometryUse,
                                    const VisualInspectionTarget* inspection = nullptr,
                                    VisualInspectionResource* resource = nullptr,
                                    VisualPlanExecutionState* state = nullptr,
                                    double evaluationTimeSec = 0.0,
                                    const std::vector<VisualEventScheduleBinding>* eventSchedules = nullptr,
                                    VisualEventTriggerCursor* eventCursor = nullptr,
                                    std::optional<renderpassoutput::Description>* colorAovPass = nullptr,
                                    std::optional<TemporalFeedbackPass>* temporalFeedbackPass = nullptr,
                                    bool* temporalFeedbackRequired = nullptr,
                                    std::optional<renderpassoutput::Description>* motionAovPass = nullptr,
                                    std::optional<sceneaov::Payload>* sceneAovPass = nullptr,
                                    std::optional<aovinspection::Payload>* aovInspectionPass = nullptr,
                                    std::optional<visualtemporalsampling::Payload>* motionBlurPass = nullptr,
                                    const std::map<std::string, double>* runtimeParameters = nullptr,
                                    int geometryWidth = 0, int geometryHeight = 0,
                                    const VisualPlanEvaluationContext* evaluationContext = nullptr,
                                    bool deferTemporalPublication = false)
{
    const auto evaluationBegin = std::chrono::steady_clock::now();
    // Old snapshots which carry no plans retain their established compositor
    // path. Once any compiled plans are supplied, every rendered owner is strict.
    if (plans.empty()) return true;
    const auto* plan = findVisualLayerPlan (plans, clipId);
    if (plan == nullptr)
    {
        error = "render layer has no compiled visual execution plan";
        return false;
    }
    if (! admitGeometryCoreOperations(*plan, geometryUse, evaluationContext, geometryWidth,
                                      geometryHeight, layer, error)) return false;
    bool pausedHold = false;
    pausedHold = state != nullptr
        && state->isHold(clipId, plan->structuralRevision, evaluationTimeSec);
    const bool ownerReset = state != nullptr
        && state->needsReset(clipId, plan->structuralRevision, evaluationTimeSec);
    if (ownerReset && ! deferTemporalPublication)
        state->telemetry().resetOwner(clipId, plan->structuralRevision);
    VisualLayerExecution uncachedExecution;
    const auto* execution = state != nullptr
        ? findAdmittedVisualLayerExecution(*state, clipId, plan->structuralRevision,
                                           evaluationTimeSec, pausedHold)
        : nullptr;
    if (state != nullptr)
    {
        if (execution == nullptr)
        {
            error = "render layer has no pre-admitted visual execution";
            return false;
        }
    }
    else
    {
        if (! compileVisualLayerExecution (*plan, uncachedExecution, error)) return false;
        execution = &uncachedExecution;
    }
    if (colorAovPass != nullptr)
        *colorAovPass = execution->colorAovPass;
    if (motionAovPass != nullptr)
        *motionAovPass = execution->motionAovPass;
    if (sceneAovPass != nullptr)
        *sceneAovPass = execution->sceneAovPass;
    if (aovInspectionPass != nullptr)
        *aovInspectionPass = execution->aovInspectionPass;
    if (temporalFeedbackPass != nullptr)
        *temporalFeedbackPass = execution->temporalFeedbackPass;
    if (temporalFeedbackRequired != nullptr)
        *temporalFeedbackRequired = execution->feedback;
    if (execution->temporalSamplingPass.has_value())
    {
        if (motionBlurPass == nullptr)
        {
            error = "visual.motion-blur sampling is admitted but not connected to native viewport/export composition";
            return false;
        }
        *motionBlurPass = execution->temporalSamplingPass->payload;
    }
    if (execution->importedAnimation.has_value())
    {
        error = "imported animation plan requires a retained native compositor frame binding";
        return false;
    }
    if (execution->matteApply
        && (layer.matteTexture == 0 || layer.matteWidth <= 0 || layer.matteHeight <= 0))
    {
        error = "typed matte GPU texture is unavailable";
        return false;
    }
    if (execution->matteCombineMode >= 0
        && (layer.matteTextureB == 0 || layer.matteWidthB <= 0 || layer.matteHeightB <= 0))
    {
        error = "typed matte combine GPU texture is unavailable";
        return false;
    }
    if (state != nullptr)
    {
        layer.feedbackHistoryReset = state->startsSequence(clipId, plan->structuralRevision)
            || state->needsReset(clipId, plan->structuralRevision, evaluationTimeSec);
        layer.feedbackHistoryHold = state->isHold(clipId, plan->structuralRevision, evaluationTimeSec);
    }

    if (inspection != nullptr && resource != nullptr && inspection->clipId == clipId
        && inspection->structuralRevision == plan->structuralRevision
        && classifyVisualInspectionTarget(*plan, *inspection) == VisualInspectionSlice::decodedSource)
    {
        // Borrow the genuine source texture already uploaded/decoded for this
        // frame. Generated sources create their texture later inside FrameRenderer.
        if (! layer.shaderSource && ! layer.particleSource && layer.texture != 0
            && layer.texWidth > 0 && layer.texHeight > 0)
        {
            resource->handle = layer.texture;
            resource->width = layer.texWidth;
            resource->height = layer.texHeight;
            resource->nodeId = inspection->nodeId;
            resource->outputPort = inspection->outputPort;
        }
    }

    if (! execution->transform)
    {
        layer.scale = 1.0f;
        layer.translateX = layer.translateY = layer.rotationDeg = 0.0f;
        layer.cropLeft = layer.cropRight = layer.cropTop = layer.cropBottom = 0.0f;
    }
    if (! execution->effects)
    {
        layer.effects = nullptr;
        layer.effectCount = 0;
        layer.lutTexture = 0;
        layer.lutSize = 0;
    }
    layer.graphKeyCleanupActive = false;
    layer.graphColorTransformActive = false;
    if (execution->colorTransform)
    {
        if (layer.texture == 0 || layer.texWidth <= 0 || layer.texHeight <= 0)
        {
            error = "visual.color.transform has no concrete decoded frame extent";
            return false;
        }
        const auto description = colortransformoperation::descriptionForExtent(
            execution->colorTransformPayload,
            { static_cast<std::uint32_t>(layer.texWidth),
              static_cast<std::uint32_t>(layer.texHeight) });
        colortransform::AdmissionFailure failure = colortransform::AdmissionFailure::None;
        const auto admitted = colortransform::admit(
            description, colortransform::BackendCapability::NativeGpu, failure);
        if (! admitted)
        {
            error = "visual.color.transform frame admission failed: "
                + std::string(colortransform::token(failure));
            return false;
        }
        layer.graphColorTransform = admitted->description();
        layer.graphColorTransformActive = true;
    }
    layer.graphTemporalActive = execution->feedback;
    layer.graphTemporalNodeId = execution->temporalNodeId;
    layer.graphTemporalPayload = execution->temporalPayload;
    if (execution->feedback && execution->temporalPayload.mode == visualtemporaloperation::Mode::feedback)
    {
        layer.graphFeedbackEffect = {};
        layer.graphFeedbackEffect.enabled = true;
        layer.graphFeedbackEffect.type = 25; // stable FeedbackTrail wire value
        layer.graphFeedbackEffect.params[0] = execution->feedbackDecay;
        layer.graphFeedbackEffect.params[1] = execution->feedbackZoom;
        layer.graphFeedbackEffect.params[2] = execution->feedbackSwirl;
        layer.effects = &layer.graphFeedbackEffect;
        layer.effectCount = 1;
    }
    if (execution->keyCleanup)
    {
        layer.graphKeyCleanupEffect = {};
        layer.graphKeyCleanupEffect.enabled = true;
        layer.graphKeyCleanupEffect.type = 17; // stable ChromaKey wire value
        layer.graphKeyCleanupEffect.params[0] = execution->keyCleanupKeyR;
        layer.graphKeyCleanupEffect.params[1] = execution->keyCleanupKeyG;
        layer.graphKeyCleanupEffect.params[2] = execution->keyCleanupKeyB;
        layer.graphKeyCleanupEffect.params[3] = execution->keyCleanupTolerance;
        layer.graphKeyCleanupEffect.params[4] = execution->keyCleanupSoftness;
        layer.graphKeyCleanupEffect.params[5] = execution->keyCleanupDespill;
        layer.graphKeyCleanupActive = true;
        layer.graphKeyCleanupChoke = execution->keyCleanupChoke;
        layer.graphKeyCleanupFeather = execution->keyCleanupFeather;
        layer.graphKeyCleanupEdgeR = execution->keyCleanupEdgeR;
        layer.graphKeyCleanupEdgeG = execution->keyCleanupEdgeG;
        layer.graphKeyCleanupEdgeB = execution->keyCleanupEdgeB;
        layer.graphKeyCleanupEdgeAmount = execution->keyCleanupEdgeAmount;
        layer.graphKeyCleanupMatteView = execution->keyCleanupMatteView;
        layer.effects = &layer.graphKeyCleanupEffect;
        layer.effectCount = 1;
    }
    if (execution->commonEffect)
    {
        layer.graphCommonEffect = {};
        layer.graphCommonEffect.enabled = true;
        layer.graphCommonEffect.type = execution->commonEffectType;
        for (std::size_t index = 0;
             index < execution->commonEffectParameterCount; ++index)
            layer.graphCommonEffect.params[index] = execution->commonEffectValues[index];
        layer.effects = &layer.graphCommonEffect;
        layer.effectCount = 1;
    }
    if (! execution->mask && ! execution->pathMatte)
    {
        layer.maskType = 0;
        layer.maskInvert = false;
    }
    layer.matteApply = execution->matteApply;
    layer.depthFog = execution->depthFog;
    layer.depthEffect = execution->depthEffect;
    layer.fogNear = execution->fogNear; layer.fogFar = execution->fogFar;
    layer.fogDensity = execution->fogDensity;
    layer.fogRed = execution->fogRed; layer.fogGreen = execution->fogGreen;
    layer.fogBlue = execution->fogBlue; layer.fogAlpha = execution->fogAlpha;
    layer.depthParam0 = execution->depthParam0; layer.depthParam1 = execution->depthParam1;
    layer.depthParam2 = execution->depthParam2; layer.depthColorRed = execution->depthColorRed;
    layer.depthColorGreen = execution->depthColorGreen; layer.depthColorBlue = execution->depthColorBlue;
    layer.particleSource = execution->particles;
    layer.flatShaderBridge = execution->flatShaderBridge;
    layer.shaderOperationPlan = execution->shaderOperationPlan;
    layer.shaderOperationParameters.clear();
    if (execution->shaderOperationPlan != nullptr)
    {
        for (const auto& operation : execution->shaderOperationPlan->operations)
        {
            auto values = operation.generatedParameters;
            if (operation.transitionPayload)
                values["progress"] = shadertransition::evaluateProgress(
                    operation.transitionPayload->progress,
                    operation.transitionPayload->direction,
                    operation.transitionPayload->easing);
            if (runtimeParameters != nullptr && operation.kind != ShaderOperationKind::transition)
                if (!applyFlatShaderBridgeRuntimeParameters(operation.payload, operation.nodeId,
                        operation.customGrant.has_value(), *runtimeParameters, values, error))
                    return false;
            layer.shaderOperationParameters.emplace(operation.nodeId, std::move(values));
        }
    }
    layer.visualPlanStructuralRevision = execution->structuralRevision;
    layer.visualPlanTelemetryHold = pausedHold;
    layer.particleNodeId = execution->particleNodeId;
    layer.drawShapeNodeId = execution->drawShapeNodeId;
    if (execution->particles)
    {
        layer.particleTriggerConnected = false;
        layer.particleTriggerCount = 0;
        layer.particleTriggerStrength = 0.0f;
        if (eventSchedules != nullptr && eventCursor != nullptr)
        {
            const auto binding = std::find_if (eventSchedules->begin(), eventSchedules->end(),
                [&] (const auto& schedule)
                {
                    return schedule.clipId == clipId && schedule.nodeId == execution->particleNodeId
                        && schedule.portId == 0;
                });
            if (binding != eventSchedules->end())
            {
                layer.particleTriggerConnected = true;
                const auto consumed = eventCursor->consume (*eventSchedules, clipId,
                    execution->particleNodeId, 0, binding->sessionRevision, layer.shaderClock.beat);
                layer.particleTriggerCount = consumed.count;
                layer.particleTriggerStrength = consumed.strongest;
            }
        }
        layer.particleStateReset = state != nullptr && layer.feedbackHistoryReset;
        layer.shaderSource = false;
        layer.texture = 0;
        const auto update = [&](const char* name, double value)
        {
            const auto found = layer.genParams.find(name);
            if (found != layer.genParams.end()) found->second = value;
        };
        update("nativeBuiltin", 1.0); update("seed", execution->particleSeed);
        update("count", execution->particleCount); update("lifetime", execution->particleLifetime);
        update("size", execution->particleSize); update("speed", execution->particleSpeed);
        update("red", execution->particleRed); update("green", execution->particleGreen);
        update("blue", execution->particleBlue); update("alpha", execution->particleAlpha);
        // The caller already populated the canonical viewport/export ShaderClock
        // (project FPS, playing/hold and seek position). Never synthesize 60 Hz.
    }
    layer.matteInvert = execution->matteInvert;
    layer.matteCombineMode = execution->matteCombineMode;
    layer.matteBlack = execution->matteBlack;
    layer.matteWhite = execution->matteWhite;
    layer.matteErodeDilate = execution->matteErodeDilate;
    layer.matteFeather = execution->matteFeather;
    layer.matteChoke = execution->matteChoke;
    layer.drawShape = execution->drawShape;
    layer.inspectionDrawShapeOutput = inspection != nullptr
        && inspection->clipId == clipId
        && inspection->structuralRevision == plan->structuralRevision
        && classifyVisualInspectionTarget(*plan, *inspection)
            == VisualInspectionSlice::retainedDrawShape;
    layer.drawShapeEllipse = execution->drawShapeEllipse;
    layer.drawShapeHasSecondary = execution->drawShapeHasSecondary;
    layer.drawShapeSecondaryEllipse = execution->drawShapeSecondaryEllipse;
    layer.drawShapeOperation = execution->drawShapeOperation;
    layer.pathMatte = execution->pathMatte;
    layer.pathMatteEllipse = execution->pathMatteEllipse;
    layer.pathMatteHasSecondary = execution->pathMatteHasSecondary;
    layer.pathMatteSecondaryEllipse = execution->pathMatteSecondaryEllipse;
    layer.pathMatteInvert = execution->pathMatteInvert;
    layer.pathMatteOperation = execution->pathMatteOperation;
    layer.pathMatteCx = execution->pathCx; layer.pathMatteCy = execution->pathCy;
    layer.pathMatteW = execution->pathW; layer.pathMatteH = execution->pathH;
    layer.pathMatte2Cx = execution->path2Cx; layer.pathMatte2Cy = execution->path2Cy;
    layer.pathMatte2W = execution->path2W; layer.pathMatte2H = execution->path2H;
    if (execution->drawShape)
    {
        layer.drawShapeCx = execution->shapeCx; layer.drawShapeCy = execution->shapeCy;
        layer.drawShapeW = execution->shapeW; layer.drawShapeH = execution->shapeH;
        layer.drawShape2Cx = execution->shape2Cx; layer.drawShape2Cy = execution->shape2Cy;
        layer.drawShape2W = execution->shape2W; layer.drawShape2H = execution->shape2H;
        layer.drawShapeR = execution->shapeR; layer.drawShapeG = execution->shapeG;
        layer.drawShapeB = execution->shapeB; layer.drawShapeA = execution->shapeA;
    }
    if (state != nullptr)
    {
        if (! deferTemporalPublication)
        {
            if (! state->publishSuccessfulExecution(
                    clipId, plan->structuralRevision, evaluationTimeSec, error))
                return false;
        }
        const auto elapsed = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - evaluationBegin).count());
        state->telemetry().recordEvaluation(clipId, plan->structuralRevision, elapsed, pausedHold);
    }
    return true;
}

// Product-owned viewport/export seam. Lowering alone is not execution: a Color
// AOV becomes available only after the renderer has submitted the native pass
// and retained its published output. Renderer rejection is terminal; there is
// deliberately no CPU production fallback.
template <typename Renderer, typename LayerDesc>
inline bool executeVisualLayerPlanForRenderer (
    Renderer& renderer, const std::vector<CompiledVisualLayerPlan>& plans,
    int clipId, LayerDesc& layer, std::string& error,
    videohelper::geometry::PlanUse geometryUse,
    TemporalSamplingCommitTransaction* temporalSamplingTransaction,
    const VisualInspectionTarget* inspection = nullptr,
    VisualInspectionResource* resource = nullptr,
    VisualPlanExecutionState* state = nullptr,
    double evaluationTimeSec = 0.0,
    const std::vector<VisualEventScheduleBinding>* eventSchedules = nullptr,
    VisualEventTriggerCursor* eventCursor = nullptr,
    const VisualPlanEvaluationContext* evaluationContext = nullptr,
    const std::map<std::string, double>* runtimeParameters = nullptr)
{
    TemporalSamplingCommitTransaction immediateTemporalTransaction;
    auto* publicationTransaction = temporalSamplingTransaction != nullptr
        ? temporalSamplingTransaction : &immediateTemporalTransaction;
    std::optional<renderpassoutput::Description> colorAovPass;
    std::optional<renderpassoutput::Description> motionAovPass;
    std::optional<sceneaov::Payload> sceneAovPass;
    std::optional<aovinspection::Payload> aovInspectionPass;
    std::optional<TemporalFeedbackPass> temporalFeedbackPass;
    std::optional<visualtemporalsampling::Payload> motionBlurPass;
    bool temporalFeedbackRequired = false;
    if (! executeVisualLayerPlan (plans, clipId, layer, error, geometryUse, inspection, resource,
                                  state, evaluationTimeSec, eventSchedules, eventCursor,
                                  &colorAovPass, &temporalFeedbackPass,
                                  &temporalFeedbackRequired, &motionAovPass, &sceneAovPass,
                                  &aovInspectionPass, &motionBlurPass, runtimeParameters,
                                  renderer.outputWidth(), renderer.outputHeight(),
                                  evaluationContext, true))
        return false;
    if (temporalFeedbackRequired && ! temporalFeedbackPass.has_value())
    {
        error = "visual temporal operation requires a pre-admitted temporal GPU resource";
        return false;
    }
    if (temporalFeedbackPass.has_value()
        && ! renderer.prepareTemporalFeedbackPass(*temporalFeedbackPass, error))
        return false;
    if (colorAovPass.has_value()
        && ! renderer.replaceColorAovPass (*colorAovPass, error))
        return false;
    if (motionAovPass.has_value()
        && ! renderer.replaceMotionAovPass (*motionAovPass, error))
        return false;
    std::optional<visualtemporalsampling::LifecycleState> pendingSamplingLifecycle;
    uint64_t pendingSamplingRevision = 0;
    if (motionBlurPass.has_value())
    {
        if (state == nullptr)
        {
            error = "motion blur requires viewport/export-owned temporal execution state";
            return false;
        }

        const VisualPlanEvaluationContext defaultContext;
        const auto& context = evaluationContext != nullptr
            ? *evaluationContext : defaultContext;
        TemporalSamplingExecution execution;
        pendingSamplingLifecycle.emplace();
        const auto* plan = findVisualLayerPlan(plans, clipId);
        if (plan == nullptr
            || ! state->prepareTemporalSamplingCandidate(
                clipId, plan->structuralRevision, evaluationTimeSec,
                context.helperGeneration, context.paused, context.mode,
                execution, *pendingSamplingLifecycle, error,
                publicationTransaction->predecessorLifecycle(
                    *state, clipId, plan->structuralRevision)))
            return false;
        pendingSamplingRevision = plan->structuralRevision;
        if (execution.pass.payload != *motionBlurPass)
        {
            error = "motion-blur execution payload differs from the admitted temporal pass";
            return false;
        }
        if (! renderer.prepareMotionBlurPass(execution, layer, error))
            return false;
    }
    if (aovInspectionPass.has_value())
    {
        if (! sceneAovPass.has_value())
        {
            error = "AOV inspection requires its exact native scene AOV source";
            return false;
        }
        if (! renderer.replaceAovInspectionPass (*sceneAovPass,
                                                  *aovInspectionPass, error))
            return false;
    }
    else if (sceneAovPass.has_value()
             && ! renderer.replaceSceneAovPass (*sceneAovPass, error))
        return false;
    if (! renderer.prepareFlatShaderBridge (layer, error))
        return false;
    if (pendingSamplingLifecycle.has_value())
    {
        if (! publicationTransaction->add (
                *state, clipId, pendingSamplingRevision, evaluationTimeSec,
                std::move (*pendingSamplingLifecycle), error))
            return false;
    }
    else if (state != nullptr)
    {
        const auto* plan = findVisualLayerPlan(plans, clipId);
        visualtemporalsampling::LifecycleState lifecycle;
        if (plan == nullptr
            || ! state->temporalLifecycleSnapshot(clipId, plan->structuralRevision, lifecycle)
            || ! publicationTransaction->add(*state, clipId, plan->structuralRevision,
                                             evaluationTimeSec, std::move(lifecycle), error))
        {
            if (error.empty())
                error = "visual execution has no matching admitted owner snapshot";
            return false;
        }
    }
    return temporalSamplingTransaction != nullptr
        || immediateTemporalTransaction.commit(error);
}
} // namespace videowire
