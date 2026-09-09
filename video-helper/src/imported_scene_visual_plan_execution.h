#pragma once

#include "imported_animation_visual_plan_execution.h"
#include "imported_scene_payload_execution.h"
#include "visual_plan_executor.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace videohelper::importedscene
{

inline void seedImportedSceneRuntimeParameters(
    const std::vector<videowire::CompiledVisualLayerPlan>& plans,
    int clipId,
    std::map<std::string, double>& parameters)
{
    const auto* plan = videowire::findVisualLayerPlan(plans, clipId);
    if (plan == nullptr) return;
    videowire::VisualLayerExecution compiled;
    std::string error;
    if (!videowire::compileVisualLayerExecution(*plan, compiled, error)
        || !compiled.importedSceneRender)
        return;
    if (compiled.importedSceneRender->deformation)
    {
        const auto& deformation = *compiled.importedSceneRender->deformation;
        const auto prefix = "visual" + std::to_string(deformation.deformationStableId) + "/";
        parameters.emplace(prefix + "timelineSeconds", deformation.playback.timelineSeconds);
        parameters.emplace(prefix + "speed", deformation.playback.speed);
        parameters.emplace(prefix + "trimStartSeconds",
                           deformation.playback.trimStartSeconds);
        parameters.emplace(prefix + "trimEndSeconds",
                           deformation.playback.trimEndSeconds);
        parameters.emplace(prefix + "weight", deformation.playback.weight);
    }
    const auto renderPrefix = "visual"
        + std::to_string(compiled.importedSceneRender->renderStableId) + "/";
    for (const auto* parameter : { "objectX", "objectY", "objectZ",
                                  "objectRotationX", "objectRotationY", "objectRotationZ",
                                  "cameraX", "cameraY", "cameraZ" })
        parameters.emplace(renderPrefix + parameter, 0.0);
    parameters.emplace(renderPrefix + "objectScale", 1.0);
    parameters.emplace(renderPrefix + "emissionGain", 1.0);
}

enum class NativeImportedSceneRenderUse : std::uint8_t
{
    Preview = 0,
    Export = 1
};

struct ImportedSceneStaticResourceIdentity final
{
    std::uint64_t projectGeneration = 0;
    std::uint64_t helperGeneration = 0;
};

inline std::string exactImportedScenePayloadIdentity(
    const videowire::CompiledVisualLayerPlan& plan)
{
    for (const auto& operation : plan.operations)
        if (operation.kind == visualimportedscenerender::kRenderNodeKind)
            return operation.payloadXml;
    return {};
}

enum class VisualImportedScenePreparation : std::uint8_t
{
    notApplicable = 0,
    rendered = 1,
    rejected = 2
};

inline bool sameCompiledVisualPlan(
    const videowire::CompiledVisualLayerPlan& left,
    const videowire::CompiledVisualLayerPlan& right) noexcept
{
    if (left.clipId != right.clipId
        || left.structuralRevision != right.structuralRevision
        || left.identityMode != right.identityMode
        || left.producerValidated != right.producerValidated
        || left.error != right.error
        || left.descriptorCount != right.descriptorCount
        || left.operationCount != right.operationCount
        || left.sceneRecordCount != right.sceneRecordCount
        || left.frameOutputCount != right.frameOutputCount
        || left.peakLiveFrameCount != right.peakLiveFrameCount
        || left.allocatedFrameSlotCount != right.allocatedFrameSlotCount
        || left.nodeKinds != right.nodeKinds
        || left.nodeIds != right.nodeIds
        || left.edges.size() != right.edges.size()
        || left.ports.size() != right.ports.size()
        || left.operations.size() != right.operations.size())
        return false;

    for (std::size_t index = 0; index < left.edges.size(); ++index)
    {
        const auto& a = left.edges[index];
        const auto& b = right.edges[index];
        if (a.fromNodeId != b.fromNodeId || a.fromPort != b.fromPort
            || a.toNodeId != b.toNodeId || a.toPort != b.toPort)
            return false;
    }
    for (std::size_t index = 0; index < left.ports.size(); ++index)
    {
        const auto& a = left.ports[index];
        const auto& b = right.ports[index];
        if (a.nodeId != b.nodeId || a.port != b.port || a.channels != b.channels
            || a.direction != b.direction || a.carrier != b.carrier
            || a.dataType != b.dataType || a.pixelFormat != b.pixelFormat
            || a.colorSpace != b.colorSpace)
            return false;
    }
    for (std::size_t index = 0; index < left.operations.size(); ++index)
    {
        const auto& a = left.operations[index];
        const auto& b = right.operations[index];
        if (a.nodeId != b.nodeId || a.kind != b.kind
            || a.backendCapability != b.backendCapability
            || a.payloadXml != b.payloadXml
            || a.runtimeGrantJson != b.runtimeGrantJson)
            return false;
    }
    return true;
}

class VisualImportedScenePlanCache final
{
public:
    void publish(const videowire::CompiledVisualLayerPlan& plan,
                 ImportedSceneStaticResourceIdentity identity) noexcept
    {
        if (!entries_.empty()
            && (identity.projectGeneration != projectGeneration_
                || identity.helperGeneration != helperGeneration_))
            entries_.clear();
        projectGeneration_ = identity.projectGeneration;
        helperGeneration_ = identity.helperGeneration;
        const auto found = entries_.find(plan.clipId);
        if (found != entries_.end() && !sameCompiledVisualPlan(found->second.plan, plan))
            entries_.erase(found);
    }

    const videowire::VisualLayerExecution* resolve(
        const videowire::CompiledVisualLayerPlan& plan,
        std::string& error)
    {
        auto found = entries_.find(plan.clipId);
        if (found != entries_.end() && sameCompiledVisualPlan(found->second.plan, plan))
        {
            found->second.lastUse = ++useSerial_;
            return &found->second.execution;
        }

        videowire::VisualLayerExecution compiled;
        if (!videowire::compileVisualLayerExecution(plan, compiled, error))
            return nullptr;

        if (entries_.size() >= kMaximumEntries)
        {
            auto victim = entries_.begin();
            for (auto candidate = std::next(victim); candidate != entries_.end(); ++candidate)
                if (candidate->second.lastUse < victim->second.lastUse
                    || (candidate->second.lastUse == victim->second.lastUse
                        && candidate->first < victim->first))
                    victim = candidate;
            entries_.erase(victim);
            found = entries_.end();
        }
        Entry replacement { plan, std::move(compiled), ++useSerial_ };
        if (found == entries_.end())
            found = entries_.emplace(plan.clipId, std::move(replacement)).first;
        else
            found->second = std::move(replacement);
        return &found->second.execution;
    }

    void reset() noexcept
    {
        entries_.clear();
        projectGeneration_ = 0;
        helperGeneration_ = 0;
        useSerial_ = 0;
    }

private:
    struct Entry final
    {
        videowire::CompiledVisualLayerPlan plan;
        videowire::VisualLayerExecution execution;
        std::uint64_t lastUse = 0;
    };
    static constexpr std::size_t kMaximumEntries = 16;
    std::map<int, Entry> entries_;
    std::uint64_t projectGeneration_ = 0;
    std::uint64_t helperGeneration_ = 0;
    std::uint64_t useSerial_ = 0;
};

template <typename Execution, typename Layer, typename FrameOwnerContainer>
VisualImportedScenePreparation prepareVisualImportedSceneLayerAtTime(
    const std::vector<videowire::CompiledVisualLayerPlan>& plans,
    int clipId,
    int width,
    int height,
    double sourceSeconds,
    double framesPerSecond,
    NativeImportedSceneRenderUse use,
    Execution* execution,
    VisualImportedScenePlanCache* planCache,
    Layer& layer,
    FrameOwnerContainer& frameOwners,
    std::string& error,
    const std::map<std::string, double>* runtimeParameters = nullptr,
    ImportedSceneStaticResourceIdentity staticIdentity = {})
{
    const auto canonicalBlockCFrame = layer.canonicalBlockCFrame;
    layer = Layer {};
    const auto reject = [&error](const char* message)
    {
        error = message;
        return VisualImportedScenePreparation::rejected;
    };
    const auto* plan = videowire::findVisualLayerPlan(plans, clipId);
    if (plan == nullptr) return VisualImportedScenePreparation::notApplicable;

    const auto exactPayloadIdentity = exactImportedScenePayloadIdentity(*plan);
    if (execution != nullptr)
        execution->publishStaticPayload(clipId, staticIdentity.projectGeneration,
                                        staticIdentity.helperGeneration,
                                        exactPayloadIdentity);

    if (planCache == nullptr)
        return reject("imported scene render requires a production-owned compiled plan cache");
    planCache->publish(*plan, staticIdentity);
    const auto* compiled = planCache->resolve(*plan, error);
    if (compiled == nullptr)
        return VisualImportedScenePreparation::rejected;
    if (!compiled->importedSceneRender.has_value())
        return VisualImportedScenePreparation::notApplicable;
    if (execution == nullptr)
        return reject("imported scene render requires the processor-owned exact payload store");
    if (width <= 0 || height <= 0 || width > 8192 || height > 8192
        || static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height)
            > 67108864u)
        return reject("imported scene render dimensions exceed the product bound");

    modelpayload::ImportedSceneRequest request;
    request.asset = compiled->importedSceneRender->asset;
    request.sceneIndex = compiled->importedSceneRender->sceneIndex;
    request.dimensions = { static_cast<std::uint32_t>(width),
                           static_cast<std::uint32_t>(height) };
    request.deformation = compiled->importedSceneRender->deformation;
    request.projectGeneration = staticIdentity.projectGeneration;
    request.helperGeneration = staticIdentity.helperGeneration;
    request.clipId = clipId;
    request.staticPayloadIdentity = exactPayloadIdentity;
    if (compiled->noteInstanceMapping)
    {
        if (!canonicalBlockCFrame)
            return reject("note-instanced imported scene requires one canonical Block C frame");
        request.runtimeInputs.canonicalBlockCFrame = canonicalBlockCFrame;
        request.runtimeInputs.noteInstanceMapping = compiled->noteInstanceMapping;
    }
    if (request.deformation && runtimeParameters != nullptr)
    {
        const auto prefix = "visual"
            + std::to_string(request.deformation->deformationStableId) + "/";
        const auto value = [&] (const char* parameter, double fallback)
        {
            const auto found = runtimeParameters->find(prefix + parameter);
            return found != runtimeParameters->end() && std::isfinite(found->second)
                ? found->second : fallback;
        };
        request.deformation->playback.timelineSeconds = std::clamp(
            value("timelineSeconds", request.deformation->playback.timelineSeconds),
            -365.0 * 24.0 * 60.0 * 60.0, 365.0 * 24.0 * 60.0 * 60.0);
        request.deformation->playback.speed = std::clamp(
            value("speed", request.deformation->playback.speed), -64.0, 64.0);
        request.deformation->playback.trimStartSeconds = std::clamp(
            value("trimStartSeconds", request.deformation->playback.trimStartSeconds),
            0.0, 86400.0);
        request.deformation->playback.trimEndSeconds = std::clamp(
            value("trimEndSeconds", request.deformation->playback.trimEndSeconds),
            0.0, 86400.0);
        request.deformation->playback.weight = std::clamp(
            value("weight", request.deformation->playback.weight), 0.0, 1.0);
    }
    if (runtimeParameters != nullptr)
    {
        const auto prefix = "visual"
            + std::to_string(compiled->importedSceneRender->renderStableId) + "/";
        const auto value = [&] (const char* parameter, double fallback,
                                double minimum, double maximum)
        {
            const auto found = runtimeParameters->find(prefix + parameter);
            const auto candidate = found != runtimeParameters->end()
                && std::isfinite(found->second) ? found->second : fallback;
            return static_cast<float>(std::clamp(candidate, minimum, maximum));
        };
        request.runtimeInputs.objectTranslationOffset = std::array<float, 3> {
            value("objectX", 0.0, -1000000.0, 1000000.0),
            value("objectY", 0.0, -1000000.0, 1000000.0),
            value("objectZ", 0.0, -1000000.0, 1000000.0)
        };
        request.runtimeInputs.objectRotationDegrees = std::array<float, 3> {
            value("objectRotationX", 0.0, -360.0, 360.0),
            value("objectRotationY", 0.0, -360.0, 360.0),
            value("objectRotationZ", 0.0, -360.0, 360.0)
        };
        request.runtimeInputs.objectScale = value("objectScale", 1.0, 0.01, 100.0);
        request.runtimeInputs.cameraTranslationOffset = std::array<float, 3> {
            value("cameraX", 0.0, -1000000.0, 1000000.0),
            value("cameraY", 0.0, -1000000.0, 1000000.0),
            value("cameraZ", 0.0, -1000000.0, 1000000.0)
        };
        request.runtimeInputs.emissionGain = value("emissionGain", 1.0, 0.0, 64.0);
    }
    if (request.deformation
        && !importedanimation::importedAnimationFrameIdentity(
            sourceSeconds, framesPerSecond, request.frame, error))
        return VisualImportedScenePreparation::rejected;
    request.structuralRevision = plan->structuralRevision;
    request.evaluationRevision = compiled->importedSceneRender->evaluationRevision;
    request.material = compiled->importedSceneRender->material;
    request.diffractionMaterial = compiled->importedSceneRender->diffractionMaterial;
    request.camera = compiled->importedSceneRender->camera;
    request.light = compiled->importedSceneRender->light;
    request.sceneSnapshot = compiled->importedSceneRender->sceneSnapshot;

    typename Execution::Receipt receipt;
    const bool executed = use == NativeImportedSceneRenderUse::Preview
        ? execution->executePreview(request, receipt, error)
        : execution->executeExport(request, receipt, error);
    if (!executed) return VisualImportedScenePreparation::rejected;
    if (!Execution::validReceipt(receipt))
        return reject("imported scene renderer returned an incomplete native frame receipt");

    const auto& frame = Execution::nativeFrame(receipt);
    if (!frame)
        return reject("imported scene renderer returned no native compositor frame");
    if (frame->width() != static_cast<std::uint32_t>(width)
        || frame->height() != static_cast<std::uint32_t>(height))
        return reject("imported scene native frame dimensions do not match the layer");
    const auto backend = frame->backend();
    if (backend != "opengl" && backend != "metal")
        return reject("imported scene native frame backend is unsupported by the compositor");
    const auto nativeView = frame->colorTextureViewHandle();
    const auto colorDescriptor = frame->colorTextureDescriptor();
    const auto depthDescriptor = frame->depthTextureDescriptor();
    const auto exactDescriptor = [&] (const arbitgpu::NativeTextureViewDescriptor& descriptor,
                                      arbitgpu::NativeTexturePixelFormat format,
                                      std::uintptr_t image, std::uintptr_t view)
    {
        return descriptor.complete() && descriptor.backend == backend
            && descriptor.format == format && descriptor.imageHandle == image
            && descriptor.textureViewHandle == view
            && descriptor.width == static_cast<std::uint32_t>(width)
            && descriptor.height == static_cast<std::uint32_t>(height);
    };
    if (nativeView == 0)
        return reject("imported scene native frame has no texture view");
    if (!exactDescriptor(colorDescriptor,
                         backend == "metal" ? arbitgpu::NativeTexturePixelFormat::Bgra8Unorm
                                              : arbitgpu::NativeTexturePixelFormat::Rgba8Unorm,
                         frame->colorImageHandle(), nativeView))
        return reject("imported scene native color descriptor violates the compositor contract");
    if (frame->depthImageHandle() == 0 || frame->depthTextureViewHandle() == 0)
        return reject("imported scene native frame has no published depth resource");
    const auto nativeDepthView = frame->depthTextureViewHandle();
    if (!exactDescriptor(depthDescriptor, arbitgpu::NativeTexturePixelFormat::R32Float,
                         frame->depthImageHandle(), nativeDepthView))
        return reject("imported scene native depth descriptor must be exact sampleable R32F");

    unsigned openGlTexture = 0;
    if (backend == "opengl")
    {
        if (nativeView > std::numeric_limits<unsigned>::max())
            return reject("imported scene compositor OpenGL texture view exceeds compositor width");
        if (nativeDepthView > std::numeric_limits<unsigned>::max())
            return reject("imported scene compositor OpenGL depth view exceeds compositor width");
        openGlTexture = static_cast<unsigned>(nativeView);
    }

    frameOwners.push_back(std::move(receipt));
    layer.texture = openGlTexture;
    layer.nativeTextureBackend = backend;
    layer.nativeTextureView = nativeView;
    layer.nativeTextureDescriptor = colorDescriptor;
    layer.depthTexture = backend == "opengl" ? static_cast<unsigned>(nativeDepthView) : 0;
    layer.nativeDepthTextureBackend = backend;
    layer.nativeDepthTextureView = nativeDepthView;
    layer.nativeDepthTextureDescriptor = depthDescriptor;
    layer.depthWidth = width;
    layer.depthHeight = height;
    layer.texWidth = width;
    layer.texHeight = height;
    layer.canonicalBlockCFrame = receipt.canonicalBlockCFrame;
    error.clear();
    return VisualImportedScenePreparation::rendered;
}

template <typename Execution, typename Layer, typename FrameOwnerContainer>
VisualImportedScenePreparation prepareVisualImportedSceneLayerAtTime(
    const std::vector<videowire::CompiledVisualLayerPlan>& plans,
    int clipId,
    int width,
    int height,
    double sourceSeconds,
    double framesPerSecond,
    NativeImportedSceneRenderUse use,
    Execution* execution,
    Layer& layer,
    FrameOwnerContainer& frameOwners,
    std::string& error,
    const std::map<std::string, double>* runtimeParameters = nullptr)
{
    VisualImportedScenePlanCache oneShotCache;
    return prepareVisualImportedSceneLayerAtTime(
        plans, clipId, width, height, sourceSeconds, framesPerSecond, use,
        execution, &oneShotCache, layer, frameOwners, error, runtimeParameters);
}

template <typename Execution, typename Layer, typename FrameOwnerContainer>
VisualImportedScenePreparation prepareVisualImportedSceneLayer(
    const std::vector<videowire::CompiledVisualLayerPlan>& plans,
    int clipId,
    int width,
    int height,
    NativeImportedSceneRenderUse use,
    Execution* execution,
    VisualImportedScenePlanCache* planCache,
    Layer& layer,
    FrameOwnerContainer& frameOwners,
    std::string& error)
{
    return prepareVisualImportedSceneLayerAtTime(
        plans, clipId, width, height, 0.0, 60.0, use,
        execution, planCache, layer, frameOwners, error);
}

template <typename Execution, typename Layer, typename FrameOwnerContainer>
VisualImportedScenePreparation prepareVisualImportedSceneLayer(
    const std::vector<videowire::CompiledVisualLayerPlan>& plans,
    int clipId,
    int width,
    int height,
    NativeImportedSceneRenderUse use,
    Execution* execution,
    Layer& layer,
    FrameOwnerContainer& frameOwners,
    std::string& error)
{
    VisualImportedScenePlanCache oneShotCache;
    return prepareVisualImportedSceneLayer(
        plans, clipId, width, height, use, execution, &oneShotCache,
        layer, frameOwners, error);
}

} // namespace videohelper::importedscene
