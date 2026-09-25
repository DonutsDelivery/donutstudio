#include "imported_animated_scene_payload_execution.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <utility>

namespace videohelper::modelpayload
{
namespace
{
bool reject(std::string& error, const char* message)
{
    error = message;
    return false;
}

} // namespace

ImportedAnimatedScenePayloadExecution::ImportedAnimatedScenePayloadExecution(
    Store& store, arbitgpu::NativeDeformationBackend& backend,
    arbitgpu::NativeFixtureSceneBackend& sceneBackend) noexcept
    : store_(store), renderer_(backend), sceneRenderer_(sceneBackend)
{
}

bool ImportedAnimatedScenePayloadExecution::inspectCompatibility(
    const visualmodelassetpayload::PayloadPtr& payload,
    std::optional<std::size_t> sceneIndex,
    ImportedAnimatedSceneCompatibility& output,
    std::string& error)
{
    output = {};
    if (!payload)
        return reject(error, "imported animated scene compatibility payload is unavailable");

    const auto& bytes = payload->bytes();
    auto animationOptions = nativeImportedAnimationDecodeOptions(bytes.size(), sceneIndex);
    auto decodedScene = gltf::decodeAnimatedGlbBaseScene(
        bytes, animationOptions.admission, error);
    if (!decodedScene)
    {
        error = "imported scene base geometry rejected: " + error;
        return false;
    }
    if (decodedScene->metadata.animations == 0)
    {
        if (!gltf::adaptStaticGlbToVisual3DScene(*decodedScene, error))
        {
            error = "imported scene native adaptation rejected: " + error;
            return false;
        }
        error.clear();
        return true;
    }

    auto document = gltf::decodeGlbAnimations(bytes, animationOptions, error);
    if (!document)
    {
        error = "imported animation/deformation decode rejected: " + error;
        return false;
    }
    ImportedAnimatedSceneCompatibility inspected;
    inspected.clips.reserve(document->clips.size());
    std::map<visualanimationimport::StableId, bool> nativeMeshAdmission;
    for (std::size_t clipIndex = 0; clipIndex < document->clips.size(); ++clipIndex)
    {
        const auto& clip = document->clips[clipIndex];
        ImportedAnimationClipCompatibility candidate;
        candidate.stableId = static_cast<visualanimationimport::StableId>(clipIndex + 1u);
        candidate.name = clip.name;
        for (const auto meshStableId
             : importedAnimationCompatibleMeshStableIds(*document, clip))
        {
            if (!clip.sceneTransformTargets.empty())
            {
                if (gltf::adaptAnimatedGlbMeshToVisual3DScene(*decodedScene, error,
                        static_cast<std::size_t>(meshStableId - 1u), true))
                    candidate.compatibleMeshStableIds.push_back(meshStableId);
                error.clear();
                continue;
            }
            auto admitted = nativeMeshAdmission.find(meshStableId);
            if (admitted == nativeMeshAdmission.end())
            {
                const auto binding = std::find_if(
                    document->renderBindings.begin(), document->renderBindings.end(),
                    [meshStableId] (const auto& entry)
                    { return entry.mesh.value == meshStableId; });
                const auto deformationMesh = binding == document->renderBindings.end()
                    ? document->deformation->meshes().end()
                    : std::find_if(
                        document->deformation->meshes().begin(),
                        document->deformation->meshes().end(),
                        [&binding] (const auto& entry) { return entry.id() == binding->mesh; });
                const auto selectedMeshIndex = static_cast<std::size_t>(meshStableId - 1u);
                auto adapted = gltf::adaptAnimatedGlbMeshToVisual3DScene(
                    *decodedScene, error, selectedMeshIndex, true);
                const auto expectedObject = binding == document->renderBindings.end()
                    ? 0u : static_cast<std::uint64_t>(binding->nodeIndex) * 65536u + 1u;
                const auto nativeAdmitted = adapted && binding != document->renderBindings.end()
                    && deformationMesh != document->deformation->meshes().end()
                    && std::any_of(adapted->objects.begin(), adapted->objects.begin() + adapted->objectCount,
                        [expectedObject](const auto& object) { return object.id.value == expectedObject; });
                admitted = nativeMeshAdmission.emplace(meshStableId, nativeAdmitted).first;
                error.clear();
            }
            if (admitted->second)
                candidate.compatibleMeshStableIds.push_back(meshStableId);
        }
        inspected.clips.push_back(std::move(candidate));
    }
    output = std::move(inspected);
    error.clear();
    return true;
}

bool ImportedAnimatedScenePayloadExecution::executePreview(
    const ImportedAnimatedSceneRequest& request,
    ImportedAnimatedSceneReceipt& output,
    std::string& error)
{
    return execute(ImportedAnimationEvaluationOwner::Preview, request, output, error);
}

bool ImportedAnimatedScenePayloadExecution::executeExport(
    const ImportedAnimatedSceneRequest& request,
    ImportedAnimatedSceneReceipt& output,
    std::string& error)
{
    return execute(ImportedAnimationEvaluationOwner::Export, request, output, error);
}

void ImportedAnimatedScenePayloadExecution::reset() noexcept
{
    clearSurfaceMaterial();
    source_.reset();
    scene_.reset();
    baseScene_.reset();
    bindings_.clear();
    admittedIdentity_ = {};
}

bool ImportedAnimatedScenePayloadExecution::admit(
    const visualmodelassetpayload::PayloadPtr& payload,
    const visualanimationimport::Request& operation,
    std::string& error)
{
    const ImportedAnimatedSceneAdmissionIdentity identity {
        payload, operation.sceneIndex, operation.meshStableId,
        operation.animationClipStableId
    };
    if (identity == admittedIdentity_) return true;
    ImportedAnimationDeformationConsumer candidateConsumer;
    if (!candidateConsumer.admit(
            operation, payload->bytes().data(), payload->bytes().size(), error))
        return false;

    auto options = nativeImportedAnimationDecodeOptions(
        payload->bytes().size(), operation.sceneIndex);
    auto decodedScene = gltf::decodeAnimatedGlbBaseScene(
        payload->bytes(), options.admission, error);
    if (!decodedScene)
    {
        error = "imported animated scene base geometry rejected: " + error;
        return false;
    }
    const auto selectedMeshIndex = operation.meshStableId == 0
        ? std::optional<std::size_t> {}
        : std::optional<std::size_t> {
            static_cast<std::size_t>(operation.meshStableId - 1u) };
    auto adapted = gltf::adaptAnimatedGlbMeshToVisual3DScene(
        *decodedScene, error, selectedMeshIndex, true);
    if (!adapted)
    {
        error = "imported animated scene native adaptation rejected: " + error;
        return false;
    }
    auto document = candidateConsumer.admittedDocument();
    if (!document)
        return reject(error, "imported animated scene clip data is unavailable");
    std::vector<gltf::GlbDeformationRenderBinding> bindings;
    for (const auto& binding : document->renderBindings)
    {
        if (operation.meshStableId != 0 && binding.mesh.value != operation.meshStableId) continue;
        const auto expectedObject = static_cast<std::uint64_t>(binding.nodeIndex) * 65536u + 1u;
        const auto object = std::find_if(adapted->objects.begin(), adapted->objects.begin() + adapted->objectCount,
            [expectedObject](const auto& candidate) { return candidate.id.value == expectedObject; });
        if (object == adapted->objects.begin() + adapted->objectCount) continue;
        const auto* mesh = document->deformation->findMesh(binding.mesh);
        if (!mesh || mesh->vertexCount() > adapted->vertexCount - object->firstVertex)
            return reject(error, "imported animated mesh has incompatible or changing topology");
        std::size_t vertices = 0;
        for (auto part = object; part != adapted->objects.begin() + adapted->objectCount
             && (part->id.value - 1u) / 65536u == binding.nodeIndex; ++part)
        {
            if (part->firstVertex != object->firstVertex + vertices)
                return reject(error, "imported animated mesh primitives are not contiguous");
            vertices += part->vertexCount;
        }
        if (vertices != mesh->vertexCount())
            return reject(error, "imported animated mesh primitive topology does not match its deformation");
        bindings.push_back(binding);
    }
    if (bindings.empty() && std::none_of(document->clips.begin(), document->clips.end(),
            [](const auto& clip) { return !clip.sceneTransformTargets.empty(); }))
        return reject(error, "imported animated scene has no selected mesh, camera or light animation bindings");
    for (std::size_t index = 0; index < adapted->objectCount; ++index)
    {
        const auto nodeIndex = (adapted->objects[index].id.value - 1u) / 65536u;
        const auto meshIndex = decodedScene->nodes[nodeIndex].mesh;
        if (meshIndex && document->deformation && document->deformation->findMesh({*meshIndex + 1u})
            && std::none_of(bindings.begin(), bindings.end(),
                [nodeIndex](const auto& binding) { return binding.nodeIndex == nodeIndex; }))
            return reject(error, "animated mesh instances require a distinct mesh per scene node; this node has no exact deformation binding");
    }

    clearSurfaceMaterial();
    consumer_ = std::move(candidateConsumer);
    scene_ = std::make_shared<const HarmonicMIDI::grid::Visual3DScene>(std::move(*adapted));
    baseScene_ = std::make_shared<const gltf::GlbStaticMeshDocument>(std::move(*decodedScene));
    bindings_ = std::move(bindings);
    admittedIdentity_ = identity;
    source_.reset();
    error.clear();
    return true;
}

bool ImportedAnimatedScenePayloadExecution::publishSurfaceMaterial(
    const surfacematerialbinding::ImportedSceneMaterialRequest& request,
    std::string& error)
{
    const auto rejectMaterial = [&] (std::string diagnostic)
    {
        if (request.programRevision != 0
            && request.programRevision == highestMaterialRevisionSeen_
            && request.programRevision > lastGoodMaterialRevision_)
            rejectedMaterialRevision_ = request.programRevision;
        error = "native animated material publication rejected: " + std::move(diagnostic);
        return false;
    };
    if (request.programRevision != 0
        && request.programRevision < highestMaterialRevisionSeen_)
        return rejectMaterial("surface material request revision is stale");
    if (request.programRevision > highestMaterialRevisionSeen_)
        highestMaterialRevisionSeen_ = request.programRevision;
    if (request.programRevision != 0
        && request.programRevision == rejectedMaterialRevision_)
        return rejectMaterial("surface material request revision was already rejected");
    if (scene_ == nullptr)
        return rejectMaterial("animated scene admission is empty");

    using videohelper::materialprogram::BackendTarget;
    const auto backendName = renderer_.backendInfo().backend;
    const auto target = backendName == "metal" ? BackendTarget::Metal
                      : backendName == "opengl" ? BackendTarget::OpenGl
                                                 : BackendTarget::Invalid;
    if (target == BackendTarget::Invalid)
        return rejectMaterial("native deformation backend has no material compiler target");

    std::string diagnostic;
    auto candidate = videorender::fixture3d::admitSurfaceMaterialBinding(
        scene_, request, target, diagnostic);
    if (candidate == nullptr)
        return rejectMaterial(std::move(diagnostic));

    const auto materialScene = materialScene_.lock();
    if (request.programRevision == lastGoodMaterialRevision_)
    {
        if (materialScene == scene_ && lastGoodMaterial_ != nullptr
            && lastGoodMaterial_->sceneRevision() == candidate->sceneRevision()
            && lastGoodMaterial_->structuralRevision() == candidate->structuralRevision()
            && lastGoodMaterial_->evaluationRevision() == candidate->evaluationRevision()
            && lastGoodMaterial_->bindingDigest() == candidate->bindingDigest())
        {
            error.clear();
            return true;
        }
        const bool nextEvaluation = materialScene == scene_ && lastGoodMaterial_ != nullptr
            && lastGoodMaterial_->sceneRevision() == candidate->sceneRevision()
            && lastGoodMaterial_->structuralRevision() == candidate->structuralRevision()
            && candidate->evaluationRevision() > lastGoodMaterial_->evaluationRevision();
        if (! nextEvaluation)
            return rejectMaterial(
                "surface material revision already names a different immutable request");
    }
    if (lastGoodMaterial_ != nullptr && materialScene == scene_
        && lastGoodMaterial_->sceneRevision() != candidate->sceneRevision())
        return rejectMaterial(
            "surface material scene revision does not match the immutable admission");

    materialScene_ = scene_;
    lastGoodMaterial_ = std::move(candidate);
    lastGoodMaterialRevision_ = request.programRevision;
    rejectedMaterialRevision_ = 0;
    error.clear();
    return true;
}

void ImportedAnimatedScenePayloadExecution::clearSurfaceMaterial() noexcept
{
    materialScene_.reset();
    lastGoodMaterial_.reset();
    highestMaterialRevisionSeen_ = 0;
    lastGoodMaterialRevision_ = 0;
    rejectedMaterialRevision_ = 0;
}

bool ImportedAnimatedScenePayloadExecution::execute(
    ImportedAnimationEvaluationOwner owner,
    const ImportedAnimatedSceneRequest& request,
    ImportedAnimatedSceneReceipt& output,
    std::string& error)
{
    if (request.structuralRevision == 0 || request.frame.rateNumerator == 0
        || request.frame.rateDenominator == 0)
        return reject(error, "imported animated scene frame identity is invalid");
    if (request.width == 0 || request.height == 0
        || request.width > 8192 || request.height > 8192
        || static_cast<std::uint64_t>(request.width) * request.height > 67108864u)
        return reject(error, "imported animated scene dimensions exceed the product bound");

    const auto payload = owner == ImportedAnimationEvaluationOwner::Preview
        ? store_.resolvePreview(request.operation.asset)
        : store_.resolveExport(request.operation.asset);
    if (!payload)
        return reject(error, "imported animated scene exact payload is not resident");
    if (!admit(payload, request.operation, error)) return false;
    if (request.operation.pose.nodeStableId != 0
        && std::none_of(scene_->objects.begin(), scene_->objects.begin() + scene_->objectCount,
            [&](const auto& object) { return (object.id.value - 1u) / 65536u + 1u
                == request.operation.pose.nodeStableId; }))
        return reject(error, "selected pose object is not drawn by the imported scene selection");

    ImportedAnimationDeformationEvaluation evaluation;
    const auto evaluated = owner == ImportedAnimationEvaluationOwner::Preview
        ? consumer_.evaluatePreview(request.operation, request.frame,
                                    request.structuralRevision, evaluation, error)
        : consumer_.evaluateExport(request.operation, request.frame,
                                   request.structuralRevision, evaluation, error);
    if (!evaluated) return false;

    bool usedLastGoodMaterial = false;
    std::string materialDiagnostic;
    if (request.material)
    {
        auto material = *request.material;
        if (material.sceneSnapshot != nullptr)
            return reject(error,
                "imported animated material transport must not carry a foreign scene owner");
        material.sceneSnapshot = scene_;
        if (!publishSurfaceMaterial(material, error))
        {
            if (surfacematerialbinding::hasGraphFrameInput(material.binding))
                return false;
            if (owner != ImportedAnimationEvaluationOwner::Preview
                || lastGoodMaterial_ == nullptr)
                return false;
            usedLastGoodMaterial = true;
            materialDiagnostic = error;
        }
    }
    else
    {
        clearSurfaceMaterial();
    }
    if (owner == ImportedAnimationEvaluationOwner::Export
        && rejectedMaterialRevision_ > lastGoodMaterialRevision_)
        return reject(error,
            "current animated surface material revision is not exportable");

    if (!source_ || source_->sourceStableId != request.operation.sourceStableId
        || source_->deformationStableId != request.operation.deformationStableId
        || source_->structuralRevision != request.structuralRevision
        || source_->clip != (evaluation.deformation ? evaluation.deformation->clip() : evaluation.sceneAnimation->clip()))
    {
        auto source = std::make_shared<arbitgpu::NativeDeformationScene>();
        source->sourceStableId = request.operation.sourceStableId;
        source->deformationStableId = request.operation.deformationStableId;
        source->structuralRevision = request.structuralRevision;
        source->clip = evaluation.deformation ? evaluation.deformation->clip() : evaluation.sceneAnimation->clip();
        source->scene = scene_;
        source->deformation = consumer_.admittedDocument()->deformation;
        for (const auto& binding : bindings_)
        {
            auto draw = std::make_shared<arbitgpu::NativeDeformationScene>(*source);
            draw->draws.clear();
            draw->mesh = binding.mesh;
            draw->skin = binding.skin;
            draw->animationNodeStableId = binding.nodeIndex + 1u;
            draw->object = {static_cast<std::uint32_t>(binding.nodeIndex * 65536u + 1u)};
            draw->morphBaseWeights = binding.morphBaseWeights;
            draw->batchMember = true;
            for (const auto& base : binding.jointBaseTransforms)
                draw->jointBaseTransforms.push_back({base.skin, base.joint,
                    base.translation, base.rotation, base.scale, base.matrix});
            source->draws.push_back(std::move(draw));
        }
        if (!source->draws.empty())
        {
            source->mesh = source->draws.front()->mesh;
            source->object = source->draws.front()->object;
        }
        if (source->draws.size() == 1 && scene_->objectCount == 1
            && scene_->materialCount == 1 && scene_->cameraCount == 1 && scene_->lightCount == 1
            && scene_->materials[0].opacity == 1.0f
            && scene_->lights[0].kind == HarmonicMIDI::grid::SceneLightKind::Directional)
        {
            const auto draw = source->draws.front();
            *source = *draw;
            source->batchMember = false;
        }
        source_ = std::move(source);
    }

    videorender::animation3d::RenderedDeformationFrame frame;
    const auto materialScene = materialScene_.lock();
    const auto material = materialScene == scene_ ? lastGoodMaterial_ : nullptr;
    const auto exactSeconds = static_cast<double>(request.frame.frame)
        * static_cast<double>(request.frame.rateDenominator)
        / static_cast<double>(request.frame.rateNumerator);
    if (!std::isfinite(exactSeconds)
        || std::abs(exactSeconds) > surfacematerial::kMaximumEvaluationMagnitude)
        return reject(error, "imported animated material time is out of bounds");
    auto runtimeInputs = request.runtimeInputs;
    if (evaluation.sceneAnimation)
    {
        std::vector<HarmonicMIDI::grid::SceneCameraRecord> cameras;
        if (!gltf::sampleGlbCameraLights(*baseScene_, *evaluation.sceneAnimation, request.operation.combinationMode,
                static_cast<float>(evaluation.playback.weight), *scene_, cameras, runtimeInputs.animatedLights, error))
            return false;
        const auto camera = std::find_if(cameras.begin(), cameras.end(),
            [&](const auto& value) { return value.id == scene_->activeCamera; });
        if (camera != cameras.end()) runtimeInputs.animatedCamera = *camera;
    }
    runtimeInputs.objectNodeStableId = request.operation.pose.nodeStableId;
    runtimeInputs.timeSeconds = static_cast<float>(exactSeconds);
    runtimeInputs.morphWeight = request.operation.pose.morphEnabled || request.operation.combinationMode
            == visualanimation::CombinationMode::WeightedBlend
        ? 1.0f : static_cast<float>(request.operation.playback.weight);
    const auto renderFrame = [&]() {
        if (bindings_.empty())
        {
            videorender::fixture3d::RenderedFrame sceneFrame;
            surfacematerial::MaterialEvaluationInputs materialInputs;
            materialInputs.timeSeconds = runtimeInputs.timeSeconds;
            const auto rendered = owner == ImportedAnimationEvaluationOwner::Preview
                ? sceneRenderer_.renderPreview(scene_, material, materialInputs, runtimeInputs,
                    {request.width, request.height}, videorender::animation3d::kNativeGpuCapability, sceneFrame, error)
                : sceneRenderer_.renderExport(scene_, material, materialInputs, runtimeInputs,
                    {request.width, request.height}, videorender::animation3d::kNativeGpuCapability, sceneFrame, error);
            if (!rendered) return false;
            frame.use = owner == ImportedAnimationEvaluationOwner::Preview
                ? videorender::animation3d::RenderUse::Preview : videorender::animation3d::RenderUse::Export;
            frame.source = source_; frame.nativeFrame = std::move(sceneFrame.nativeFrame);
            frame.stats.drawCount = sceneFrame.stats.drawCount;
            frame.stats.clipId = source_->clip.value; frame.stats.time = request.frame;
            frame.stats.sourceStableId = source_->sourceStableId;
            frame.stats.deformationStableId = source_->deformationStableId;
            frame.stats.revision = request.structuralRevision;
            frame.stats.staticUploadCount = sceneFrame.stats.staticUploadCount;
            frame.stats.reusedStaticResources = sceneFrame.stats.reusedStaticResources;
            return true;
        }
        return owner == ImportedAnimationEvaluationOwner::Preview
        ? renderer_.renderPreview(source_, evaluation.deformation,
                                  material ? material->nativeProgram() : nullptr,
                                  runtimeInputs, request.width, request.height,
                                  videorender::animation3d::kNativeGpuCapability, frame, error)
        : renderer_.renderExport(source_, evaluation.deformation,
                                 material ? material->nativeProgram() : nullptr,
                                 runtimeInputs, request.width, request.height,
                                 videorender::animation3d::kNativeGpuCapability, frame, error);
    };
    if (!renderFrame()) return false;

    ImportedAnimatedSceneReceipt receipt;
    receipt.owner = owner;
    receipt.payload = payload;
    receipt.source = source_;
    receipt.evaluation = std::move(evaluation);
    receipt.frame = std::move(frame);
    receipt.runtimeInputs = std::move(runtimeInputs);
    receipt.usedLastGoodMaterial = usedLastGoodMaterial;
    receipt.materialDiagnostic = std::move(materialDiagnostic);
    output = std::move(receipt);
    error.clear();
    return true;
}
} // namespace videohelper::modelpayload
