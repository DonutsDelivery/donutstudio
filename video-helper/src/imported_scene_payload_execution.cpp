#include "imported_scene_payload_execution.h"
#include "VisualImportedAnimationOperationContract.h"
#include "VisualImportedSceneIdentity.h"

#include <limits>
#include <iterator>
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

[[maybe_unused]] std::string exactAssetIdentity(const ImportedSceneRequest& request)
{
    if (request.sceneSnapshot != nullptr)
        return "composed";
    return request.asset.id + "\n" + std::to_string(request.asset.version) + "\n"
        + request.asset.contentSha256 + "\n" + request.asset.sourceMediaType + "\n"
        + std::to_string(request.asset.sourceByteSize) + "\n"
        + (request.sceneIndex ? std::to_string(*request.sceneIndex) : "default");
}

} // namespace

ImportedScenePayloadExecution::ImportedScenePayloadExecution(
    Store& store, arbitgpu::NativeFixtureSceneBackend& backend) noexcept
    : store_(store), backend_(backend)
{
}

ImportedScenePayloadExecution::ImportedScenePayloadExecution(
    Store& store, arbitgpu::NativeFixtureSceneBackend& backend,
    arbitgpu::NativeDeformationBackend& deformationBackend) noexcept
    : store_(store), backend_(backend),
      deformationExecution_(
          std::make_unique<ImportedAnimatedScenePayloadExecution>(
              store, deformationBackend))
{
}

bool ImportedScenePayloadExecution::executePreview(
    const ImportedSceneRequest& request,
    ImportedSceneExecutionReceipt& output,
    std::string& error)
{
    return execute(ImportedSceneUse::Preview, request, output, error);
}

bool ImportedScenePayloadExecution::executeExport(
    const ImportedSceneRequest& request,
    ImportedSceneExecutionReceipt& output,
    std::string& error)
{
    return execute(ImportedSceneUse::Export, request, output, error);
}

void ImportedScenePayloadExecution::reset() noexcept
{
    if (deformationExecution_ != nullptr)
        deformationExecution_->reset();
    staticOwners_.clear();
    projectGeneration_ = 0;
    helperGeneration_ = 0;
    useSerial_ = 0;
    deviceIdentity_.clear();
}

void ImportedScenePayloadExecution::synchronizeGenerations(
    std::uint64_t projectGeneration, std::uint64_t helperGeneration) noexcept
{
    if (!staticOwners_.empty()
        && (projectGeneration != projectGeneration_
            || helperGeneration != helperGeneration_))
        staticOwners_.clear();
    projectGeneration_ = projectGeneration;
    helperGeneration_ = helperGeneration;
}

void ImportedScenePayloadExecution::synchronizeDevice() noexcept
{
    const auto info = backend_.info();
    const auto identity = info.backend + "\n" + info.device;
    if (!deviceIdentity_.empty() && identity != deviceIdentity_)
        staticOwners_.clear();
    deviceIdentity_ = identity;
}

void ImportedScenePayloadExecution::publishStaticPayload(
    int clipId, std::uint64_t projectGeneration, std::uint64_t helperGeneration,
    const std::string& exactPayloadIdentity) noexcept
{
    synchronizeGenerations(projectGeneration, helperGeneration);
    synchronizeDevice();
    for (auto found = staticOwners_.begin(); found != staticOwners_.end();)
    {
        if (found->first.first == clipId
            && found->second.exactPayloadIdentity != exactPayloadIdentity)
            found = staticOwners_.erase(found);
        else
            ++found;
    }
}

void ImportedScenePayloadExecution::evictForInsertion() noexcept
{
    if (staticOwners_.size() < kMaximumStaticOwners)
        return;
    auto victim = staticOwners_.begin();
    for (auto candidate = std::next(victim); candidate != staticOwners_.end(); ++candidate)
        if (candidate->second.lastUse < victim->second.lastUse
            || (candidate->second.lastUse == victim->second.lastUse
                && candidate->first < victim->first))
            victim = candidate;
    staticOwners_.erase(victim);
}

bool ImportedScenePayloadExecution::execute(
    ImportedSceneUse use,
    const ImportedSceneRequest& request,
    ImportedSceneExecutionReceipt& output,
    std::string& error)
{
    publishStaticPayload(request.clipId, request.projectGeneration,
                         request.helperGeneration, request.staticPayloadIdentity);
    const bool composedScene = request.sceneSnapshot != nullptr;
    if (composedScene)
    {
        if (!HarmonicMIDI::grid::validateVisual3DScene(*request.sceneSnapshot).valid()
            || !request.asset.id.empty() || request.sceneIndex || request.deformation
            || request.camera || request.light || request.structuralRevision == 0
            || request.evaluationRevision < request.structuralRevision)
            return reject(error, "composed Scene3D request is invalid or mixes imported-scene state");
        visualimportedscenerender::Request operation;
        operation.sourceStableId = request.sceneSnapshot->id.value;
        operation.renderStableId = operation.sourceStableId == 1 ? 2 : 1;
        operation.sceneSnapshot = request.sceneSnapshot;
        operation.structuralRevision = request.structuralRevision;
        operation.evaluationRevision = request.evaluationRevision;
        operation.material = request.material;
        operation.diffractionMaterial = request.diffractionMaterial;
        if (operation.material && operation.material->sceneSnapshot == nullptr)
            operation.material->sceneSnapshot = request.sceneSnapshot;
        if (!visualimportedscenerender::valid(operation))
            return reject(error, "composed Scene3D material identity or revision is stale");
    }
    else
    {
        if (!visualmodelasset::validStableAssetId(request.asset.id)
            || request.asset.version == 0
            || request.asset.version > static_cast<std::uint64_t>(
                   std::numeric_limits<std::int64_t>::max())
            || !visualmodelasset::validSha256(request.asset.contentSha256))
            return reject(error, "imported scene exact-content identity is invalid");
        if (request.asset.sourceMediaType != "model/gltf-binary")
            return reject(error, "imported scene payload must be model/gltf-binary");
        if (request.asset.sourceByteSize == 0
            || request.asset.sourceByteSize > visualmodelassetpayload::kMaxPayloadBytes)
            return reject(error, "imported scene payload size exceeds the product bound");
        if (request.sceneIndex && *request.sceneIndex >= 64u)
            return reject(error, "imported scene index exceeds the product bound");
    }
    if (request.dimensions.width == 0 || request.dimensions.height == 0
        || request.dimensions.width > videorender::fixture3d::FixtureSceneRenderer::kMaxExtent
        || request.dimensions.height > videorender::fixture3d::FixtureSceneRenderer::kMaxExtent
        || static_cast<std::uint64_t>(request.dimensions.width)
             * static_cast<std::uint64_t>(request.dimensions.height)
           > videorender::fixture3d::FixtureSceneRenderer::kMaxPixels)
        return reject(error, "imported scene dimensions exceed the product bound");
    if (request.camera.has_value()
        && !visualimportedscenerender::detail::validCamera(*request.camera))
        return reject(error, "imported scene graph camera is invalid");
    if (request.light.has_value()
        && !visualimportedscenerender::detail::validLight(*request.light))
        return reject(error, "imported scene graph light is invalid");
    if (request.material && request.diffractionMaterial)
        return reject(error, "imported scene request admits one exact material terminal");
    if (request.deformation && request.diffractionMaterial)
        return reject(error, "animated imported scenes do not yet admit native diffraction materials");
    if (request.diffractionMaterial)
    {
        const auto expectedScene = composedScene
            ? std::optional<HarmonicMIDI::grid::Scene3DId> { request.sceneSnapshot->id }
            : request.sceneIndex
                ? HarmonicMIDI::grid::visualimportedsceneidentity::sceneId(*request.sceneIndex)
                : std::nullopt;
        const auto expectedRevision = composedScene
            ? request.structuralRevision : request.asset.version;
        if (!expectedScene || request.diffractionMaterial->scene != *expectedScene
            || request.diffractionMaterial->sceneRevision != expectedRevision)
            return reject(error, "imported diffraction material does not match the exact scene identity");
        if (request.diffractionMaterial->structuralRevision
            != request.structuralRevision)
            return reject(error, "imported diffraction material does not match the exact plan revision");
    }

    if (request.deformation)
    {
        if (!visualanimationoperation::valid(*request.deformation)
            || !visualanimationimport::sameAsset(
                request.deformation->asset, request.asset)
            || request.deformation->sceneIndex != request.sceneIndex)
            return reject(error, "imported scene deformation does not match the exact scene request");
        if (deformationExecution_ == nullptr)
            return reject(error, "native imported scene deformation execution is unavailable");

        ImportedAnimatedSceneRequest deformationRequest;
        deformationRequest.operation = *request.deformation;
        deformationRequest.frame = request.frame;
        deformationRequest.structuralRevision = request.structuralRevision;
        deformationRequest.width = request.dimensions.width;
        deformationRequest.height = request.dimensions.height;
        deformationRequest.material = request.material;
        deformationRequest.runtimeInputs = request.runtimeInputs;
        deformationRequest.runtimeInputs.cameraOverride = request.camera;
        deformationRequest.runtimeInputs.lightOverride = request.light;
        ImportedAnimatedSceneReceipt deformationFrame;
        const auto rendered = use == ImportedSceneUse::Preview
            ? deformationExecution_->executePreview(
                deformationRequest, deformationFrame, error)
            : deformationExecution_->executeExport(
                deformationRequest, deformationFrame, error);
        if (!rendered) return false;

        ImportedSceneExecutionReceipt receipt;
        receipt.use = use;
        receipt.payload = deformationFrame.payload;
        receipt.usedLastGoodMaterial = deformationFrame.usedLastGoodMaterial;
        receipt.materialDiagnostic = deformationFrame.materialDiagnostic;
        receipt.deformationFrame = std::move(deformationFrame);
        receipt.canonicalBlockCFrame = request.runtimeInputs.canonicalBlockCFrame;
        output = std::move(receipt);
        error.clear();
        return true;
    }

    visualmodelassetpayload::PayloadPtr payload;
    if (!composedScene)
    {
        payload = use == ImportedSceneUse::Preview
            ? store_.resolvePreview(request.asset)
            : store_.resolveExport(request.asset);
        if (!payload)
            return reject(error, "imported scene exact payload is not resident");
        if (payload->bytes().size() != request.asset.sourceByteSize)
            return reject(error, "imported scene resident payload size is inconsistent");

    }

    const StaticOwnerKey ownerKey { request.clipId, use };
    auto foundOwner = staticOwners_.find(ownerKey);
    if (foundOwner != staticOwners_.end()
        && ((!composedScene && foundOwner->second.payload != payload)
            || foundOwner->second.sceneIndex != request.sceneIndex
            || (composedScene
                && foundOwner->second.admission.scene != request.sceneSnapshot)))
    {
        staticOwners_.erase(foundOwner);
        foundOwner = staticOwners_.end();
    }
    if (foundOwner == staticOwners_.end())
    {
        evictForInsertion();
        auto [inserted, didInsert] = staticOwners_.try_emplace(ownerKey, backend_);
        if (!didInsert)
            return reject(error, "imported scene static owner insertion failed");
        auto& owner = inserted->second;
        owner.exactPayloadIdentity = request.staticPayloadIdentity;
        owner.payload = payload;
        owner.sceneIndex = request.sceneIndex;
        if (composedScene)
            owner.admission.scene = request.sceneSnapshot;
        else if (!owner.seam.admit(payload->bytes(), request.sceneIndex,
                                   owner.admission, error))
        {
            staticOwners_.erase(inserted);
            return false;
        }
        foundOwner = inserted;
    }
    auto& owner = foundOwner->second;
    owner.lastUse = ++useSerial_;
    auto& seam = owner.seam;
    auto& admission = owner.admission;

    bool usedLastGoodMaterial = false;
    std::string materialDiagnostic;
    if (request.material)
    {
        auto material = *request.material;
        if (material.sceneSnapshot != nullptr
            && material.sceneSnapshot != admission.scene)
            return reject(error, "imported scene material transport must not carry a foreign scene owner");
        material.sceneSnapshot = admission.scene;
        if (!seam.publishSurfaceMaterial(admission, material, error))
        {
            if (use != ImportedSceneUse::Preview
                || seam.lastGoodMaterialRevision() == 0)
                return false;
            usedLastGoodMaterial = true;
            materialDiagnostic = error;
        }
    }
    else if (request.diffractionMaterial)
    {
        if (!seam.publishDiffractionMaterial(
                admission, *request.diffractionMaterial, error))
        {
            if (use != ImportedSceneUse::Preview
                || seam.lastGoodMaterialRevision() == 0)
                return false;
            usedLastGoodMaterial = true;
            materialDiagnostic = error;
        }
    }
    else
    {
        seam.clearSurfaceMaterial(admission);
        seam.clearDiffractionMaterial(admission);
    }

    auto sceneInputs = request.runtimeInputs;
    sceneInputs.cameraOverride = request.camera;
    sceneInputs.lightOverride = request.light;
    gltf::GlbNativeFrameReceipt frame;
    const auto rendered = use == ImportedSceneUse::Preview
        ? seam.renderPreview(admission, request.dimensions, sceneInputs, frame, error)
        : seam.renderExport(admission, request.dimensions, sceneInputs, frame, error);
    if (!rendered)
        return false;

    ImportedSceneExecutionReceipt receipt;
    receipt.use = use;
    receipt.payload = payload;
    receipt.admission = admission;
    receipt.frame = std::move(frame);
    receipt.usedLastGoodMaterial = usedLastGoodMaterial;
    receipt.materialDiagnostic = std::move(materialDiagnostic);
    receipt.canonicalBlockCFrame = request.runtimeInputs.canonicalBlockCFrame;
    output = std::move(receipt);
    error.clear();
    return true;
}
} // namespace videohelper::modelpayload
