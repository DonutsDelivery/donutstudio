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
    Store& store, arbitgpu::NativeDeformationBackend& backend) noexcept
    : store_(store), renderer_(backend)
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
    if (!document->deformation)
        return reject(error, "imported animation/deformation payload is unavailable");

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
                auto adapted = gltf::adaptStaticGlbToVisual3DScene(
                    *decodedScene, error, selectedMeshIndex);
                const auto expectedObject = binding == document->renderBindings.end()
                    ? 0u : static_cast<std::uint64_t>(binding->nodeIndex) * 65536u + 1u;
                const auto nativeAdmitted = adapted && binding != document->renderBindings.end()
                    && deformationMesh != document->deformation->meshes().end()
                    && adapted->objectCount == 1 && adapted->materialCount == 1
                    && adapted->lightCount == 1 && adapted->cameraCount == 1
                    && adapted->objects[0].id.value == expectedObject;
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
    binding_ = {};
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
    auto adapted = gltf::adaptStaticGlbToVisual3DScene(
        *decodedScene, error, selectedMeshIndex);
    if (!adapted)
    {
        error = "imported animated scene native adaptation rejected: " + error;
        return false;
    }
    auto document = candidateConsumer.admittedDocument();
    if (!document || !document->deformation
        || adapted->objectCount != 1 || adapted->materialCount != 1
        || adapted->lightCount != 1 || adapted->cameraCount != 1)
        return reject(error, "imported animated scene is outside the single-draw native subset");

    const auto bindingIt = operation.meshStableId == 0
        ? (document->renderBindings.size() == 1
            ? document->renderBindings.begin() : document->renderBindings.end())
        : std::find_if(document->renderBindings.begin(), document->renderBindings.end(),
            [&operation] (const auto& candidate)
            { return candidate.mesh.value == operation.meshStableId; });
    if (bindingIt == document->renderBindings.end())
        return reject(error, "imported animated scene mesh selector is unavailable");
    const auto& binding = *bindingIt;
    const auto deformationMesh = std::find_if(
        document->deformation->meshes().begin(), document->deformation->meshes().end(),
        [&binding] (const auto& candidate) { return candidate.id() == binding.mesh; });
    const auto expectedObject = static_cast<std::uint64_t>(binding.nodeIndex) * 65536u + 1u;
    if (adapted->objects[0].id.value != expectedObject
        || deformationMesh == document->deformation->meshes().end())
        return reject(error, "imported animated scene mesh and object identities are incompatible");

    clearSurfaceMaterial();
    consumer_ = std::move(candidateConsumer);
    scene_ = std::make_shared<const HarmonicMIDI::grid::Visual3DScene>(std::move(*adapted));
    binding_ = binding;
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
        || source_->clip != evaluation.deformation->clip())
    {
        auto source = std::make_shared<arbitgpu::NativeDeformationScene>();
        source->sourceStableId = request.operation.sourceStableId;
        source->deformationStableId = request.operation.deformationStableId;
        source->structuralRevision = request.structuralRevision;
        source->clip = evaluation.deformation->clip();
        source->mesh = binding_.mesh;
        source->object = scene_->objects[0].id;
        source->scene = scene_;
        source->deformation = consumer_.admittedDocument()->deformation;
        source->morphBaseWeights = binding_.morphBaseWeights;
        for (const auto& base : binding_.jointBaseTransforms)
            source->jointBaseTransforms.push_back({base.skin, base.joint,
                base.translation, base.rotation, base.scale});
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
    runtimeInputs.timeSeconds = static_cast<float>(exactSeconds);
    runtimeInputs.morphWeight = request.operation.combinationMode
            == visualanimation::CombinationMode::WeightedBlend
        ? 1.0f : static_cast<float>(request.operation.playback.weight);
    const auto rendered = owner == ImportedAnimationEvaluationOwner::Preview
        ? renderer_.renderPreview(source_, evaluation.deformation,
                                  material ? material->nativeProgram() : nullptr,
                                  runtimeInputs, request.width, request.height,
                                  videorender::animation3d::kNativeGpuCapability, frame, error)
        : renderer_.renderExport(source_, evaluation.deformation,
                                 material ? material->nativeProgram() : nullptr,
                                 runtimeInputs, request.width, request.height,
                                 videorender::animation3d::kNativeGpuCapability, frame, error);
    if (!rendered) return false;

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
