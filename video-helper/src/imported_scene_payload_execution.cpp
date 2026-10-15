#include "imported_scene_payload_execution.h"
#include "VisualImportedAnimationOperationContract.h"
#include "VisualImportedSceneIdentity.h"
#include "geometry_score_field.h"
#include "glb_scene_adapter.h"
#include "../../shared/Visual3DSceneComposition.h"
#include "../../shared/GeometrySurfaceMaterial.h"
#include "loop_step_probe.h"

#include <algorithm>
#include <limits>
#include <iterator>
#include <new>
#include <type_traits>
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

template <typename T>
std::shared_ptr<const T> retainParentPayloadValue(std::shared_ptr<const T> value,
    const std::shared_ptr<const NativeCompositionRetentionBudget::Charge>& charge,
    const std::shared_ptr<NativeSceneAnimationCompositionBudget>& budget)
{
    struct Retained final
    {
        std::shared_ptr<const NativeCompositionRetentionBudget::Charge> charge;
        std::shared_ptr<NativeSceneAnimationCompositionBudget> budget;
        std::shared_ptr<const T> value;
    };
    auto retained=std::make_shared<Retained>(Retained {charge,budget,value});
    return {std::move(retained),value.get()};
}

template <typename T>
struct RetainedStaticLocalAllocation final
{
    std::shared_ptr<const NativeSceneAnimationCompositionBudget::Charge> charge;
    std::shared_ptr<const T> value;
};
template <typename T>
std::shared_ptr<const T> retainStaticLocalAllocation(std::shared_ptr<const T> value,
    const std::shared_ptr<const NativeSceneAnimationCompositionBudget::Charge>& charge)
{
    auto retained = std::make_shared<RetainedStaticLocalAllocation<T>>(RetainedStaticLocalAllocation<T>{charge, value});
    return {std::move(retained), value.get()};
}

// Three complete outer header visits exchange fixed members. The nested
// optional adds three move-construction and three moved-from disposal visits,
// three engagement tests, two resets and temporary setup/disposal controls.
constexpr auto parentReceiptExchangeWork=3*sizeof(ImportedSceneExecutionReceipt)
    +6*sizeof(ImportedAnimatedSceneReceipt)+7;
// One empty outgoing receipt, one nested optional relocation and the largest
// scalar record swapped below. No owning value uses move assignment.
constexpr auto parentReceiptDisposalBytes=sizeof(ImportedSceneExecutionReceipt)
    +sizeof(std::optional<ImportedAnimatedSceneReceipt>)+sizeof(arbitgpu::NativeFixtureSceneStats);
constexpr auto parentReceiptClearWork=parentReceiptExchangeWork
    +2*sizeof(ImportedSceneExecutionReceipt)+2;

[[maybe_unused]] std::string exactAssetIdentity(const ImportedSceneRequest& request)
{
    if (request.sceneSnapshot != nullptr)
        return "composed";
    return request.asset.id + "\n" + std::to_string(request.asset.version) + "\n"
        + request.asset.contentSha256 + "\n" + request.asset.sourceMediaType + "\n"
        + std::to_string(request.asset.sourceByteSize) + "\n"
        + (request.sceneIndex ? std::to_string(*request.sceneIndex) : "default");
}

// Only a canonical, otherwise identical plain imported source can defer
// retirement while its new object/scene selection is admitted. All other
// publications retain the existing immediate retirement behavior.
bool sameStaticObjectSelectionSource(const std::string& previous, const std::string& incoming) noexcept
{
    if (std::string_view(incoming).substr(0, visualimportedscenerender::kCompiledOperationHeaderV24.size())
        != visualimportedscenerender::kCompiledOperationHeaderV24) return false;
    try
    {
        visualimportedscenerender::Request old, next;
        std::string error;
        if (!visualimportedscenerender::decodeCanonical(previous, old, error)
            || !visualimportedscenerender::decodeCanonical(incoming, next, error)
            || !next.objectNodeStableId || !next.objectMeshStableId
            || old.importedScene || next.importedScene || old.sceneSnapshot || next.sceneSnapshot
            || old.importedParentIntent || next.importedParentIntent || old.deformation || next.deformation
            || old.glassCaustic || next.glassCaustic) return false;
        next.objectNodeStableId = old.objectNodeStableId;
        next.objectMeshStableId = old.objectMeshStableId;
        next.sceneIndex = old.sceneIndex;
        return visualimportedscenerender::sameRequest(old, next);
    }
    catch (...)
    {
        // Publication is noexcept. Failed bounded decoding supplies no authority
        // to preserve an owner from a changed incoming payload.
        return false;
    }
}

// An executed deformation may change playback controls and the runtime pose
// values, never the selection the Graph compiled for its leaf.
bool sameImportedDeformationSelection(const visualanimationimport::Request& executed,
    const visualanimationimport::Request& compiled)
{
    return executed.sourceStableId == compiled.sourceStableId
        && executed.deformationStableId == compiled.deformationStableId
        && executed.meshStableId == compiled.meshStableId
        && executed.animationClipStableId == compiled.animationClipStableId
        && executed.crossfadeClipStableId == compiled.crossfadeClipStableId
        && executed.clipName == compiled.clipName
        && executed.crossfadeClipName == compiled.crossfadeClipName
        && executed.combinationMode == compiled.combinationMode
        && executed.schedule == compiled.schedule
        && executed.pose.boneEnabled == compiled.pose.boneEnabled
        && executed.pose.morphEnabled == compiled.pose.morphEnabled
        && executed.pose.morphTargetIndex == compiled.pose.morphTargetIndex
        && executed.pose.nodeStableId == compiled.pose.nodeStableId
        && executed.pose.meshStableId == compiled.pose.meshStableId
        && executed.pose.boneStableId == compiled.pose.boneStableId
        && executed.pose.morphTargetStableId == compiled.pose.morphTargetStableId;
}

} // namespace

void ImportedSceneExecutionReceipt::exchangeOwnedValues(ImportedSceneExecutionReceipt& other) noexcept
{
    using std::swap;
    swap(use,other.use);payload.swap(other.payload);
    admission.documentRetention.swap(other.admission.documentRetention);
    admission.document.swap(other.admission.document);
    admission.sourceScene.swap(other.admission.sourceScene);
    swap(admission.retainedDocumentBytes,other.admission.retainedDocumentBytes);
    admission.scene.swap(other.admission.scene);
    swap(admission.metadata.containerBytes,other.admission.metadata.containerBytes);
    swap(admission.metadata.jsonBytes,other.admission.metadata.jsonBytes);
    swap(admission.metadata.binBytes,other.admission.metadata.binBytes);
    swap(admission.metadata.scenes,other.admission.metadata.scenes);
    swap(admission.metadata.nodes,other.admission.metadata.nodes);
    swap(admission.metadata.meshes,other.admission.metadata.meshes);
    swap(admission.metadata.primitives,other.admission.metadata.primitives);
    swap(admission.metadata.materials,other.admission.metadata.materials);
    swap(admission.metadata.textures,other.admission.metadata.textures);
    swap(admission.metadata.images,other.admission.metadata.images);
    swap(admission.metadata.samplers,other.admission.metadata.samplers);
    swap(admission.metadata.cameras,other.admission.metadata.cameras);
    swap(admission.metadata.animations,other.admission.metadata.animations);
    swap(admission.metadata.skins,other.admission.metadata.skins);
    swap(admission.metadata.accessors,other.admission.metadata.accessors);
    swap(admission.metadata.bufferViews,other.admission.metadata.bufferViews);
    swap(admission.metadata.maxHierarchyDepth,other.admission.metadata.maxHierarchyDepth);
    swap(admission.metadata.defaultScene,other.admission.metadata.defaultScene);
    admission.metadata.generator.swap(other.admission.metadata.generator);
    admission.metadata.extensionsUsed.swap(other.admission.metadata.extensionsUsed);
    admission.metadata.extensionsRequired.swap(other.admission.metadata.extensionsRequired);
    admission.importedLightDefinitions.swap(other.admission.importedLightDefinitions);
    swap(admission.selectedScene,other.admission.selectedScene);
    swap(admission.sourceBytes,other.admission.sourceBytes);
    swap(admission.decodedVertexCount,other.admission.decodedVertexCount);
    swap(admission.decodedIndexCount,other.admission.decodedIndexCount);
    swap(admission.decodedBytes,other.admission.decodedBytes);
    frame.swap(other.frame);
    // optional::swap would move-assign an engaged receipt, retaining some old
    // string buffers. Relocate instead. Each destination is disengaged before
    // construction, so all nested strings, containers and private associations
    // transfer through their default-allocator move constructors. Both outer
    // stamps stay alive throughout; moved-from resets own no transferred value.
    std::optional<ImportedAnimatedSceneReceipt> nested;
    if (deformationFrame) nested.emplace(std::move(*deformationFrame));
    deformationFrame.reset();
    if (other.deformationFrame) deformationFrame.emplace(std::move(*other.deformationFrame));
    other.deformationFrame.reset();
    if (nested) other.deformationFrame.emplace(std::move(*nested));
    swap(usedLastGoodMaterial,other.usedLastGoodMaterial);
    materialDiagnostic.swap(other.materialDiagnostic);
    canonicalBlockCFrame.swap(other.canonicalBlockCFrame);
    materialFrameTexture.swap(other.materialFrameTexture);
    // No actual outgoing buffer is destroyed during exchange. Finally exchange
    // the stamps so each complete result retains its own charge until disposal.
    parentPayloadIdentity_.swap(other.parentPayloadIdentity_);
}

void ImportedScenePayloadExecution::clearCallerReceipt(Receipt& output) noexcept
{
    if (!output.parentPayloadIdentity_) { output={};return; }
    // Already-authorized retirement cannot depend on new admission. Record the
    // entire fixed attempt even when consumeWork saturates the surviving ledger.
    output.parentPayloadIdentity_->budget->consumeWork(parentReceiptClearWork,1);
    Receipt outgoing;
    output.exchangeOwnedValues(outgoing);
    // Reverse member destruction disposes all diagnostic/native values first,
    // then the outgoing stamp. The caller now owns fresh empty string capacity.
}

#if defined(ARBIT_TEST_IMPORTED_PARENT_STAGING_FAILURES)
void ImportedSceneParentStagingFailureProbe::allocation(Site site,
    const ImportedAnimatedSceneReceipt& frame,bool diagnosticAllocates)
{
    auto& fault=state();
    if (fault.selected!=site || (site==Site::DiagnosticCopy && !diagnosticAllocates)) return;
    fault.candidateOwner=frame.runtimeInputs.compositionFrame->owner();
    fault.candidateNative=frame.frame.nativeFrame;
    fault.diagnosticAllocates=diagnosticAllocates;
    fault.hit=true;
    throw std::bad_alloc();
}

ImportedSceneParentStagingFailureProbe::Diagnostic
ImportedSceneParentStagingFailureProbe::diagnostic(const ImportedSceneExecutionReceipt& receipt) noexcept
{
    Diagnostic result;result.capacity=receipt.materialDiagnostic.capacity();
    if (receipt.parentPayloadIdentity_ && receipt.parentPayloadIdentity_->diagnosticRetention)
    {
        const auto& charge=receipt.parentPayloadIdentity_->diagnosticRetention;
        result.retention=charge;result.pool=charge->budget;result.bytes=charge->bytes;
    }
    return result;
}

ImportedSceneParentStagingFailureProbe::Witness
ImportedSceneParentStagingFailureProbe::inspect(const ImportedScenePayloadExecution& execution) noexcept
{
    Witness result;
    const auto& inner=*execution.deformationExecution_;
    const auto sample=[](const std::optional<arbitgpu::NativeSceneMotionSample>& value) {
        Sample result;
        result.present=value.has_value();
        if (value)
        {
            result.translation=value->objectTranslationOffset;result.rotation=value->objectRotationDegrees;
            result.cameraTranslation=value->cameraTranslationOffset;result.scale=value->objectScale;
            result.morph=value->morphWeight;result.camera=value->cameraOverride;result.owner=value->owner;
            result.deformation=value->deformation;result.composition=value->compositionFrame;
            result.worlds=value->importedNodeWorldMatrices;result.notes=value->noteTransforms;
        }
        return result;
    };
    for (std::size_t i=0;i<result.history.size();++i)
    {
        const auto& actual=inner.motion_[i];auto& saved=result.history[i];
        saved.current=sample(actual.current);saved.previous=sample(actual.previous);
        saved.frame=actual.frame;saved.numerator=actual.numerator;saved.denominator=actual.denominator;
        saved.width=actual.width;saved.height=actual.height;saved.owner=actual.owner;
        saved.mappingEmpty=actual.mapping.empty();saved.noteOwnerAbsent=!actual.noteOwner;
    }
    result.source=inner.source_;result.parent=inner.parentOwner_;result.material=inner.lastGoodMaterial_;
    result.retentionBudget=inner.compositionBudget_;result.workBudget=inner.sceneAnimationBudget_;
    result.attemptedWork=inner.sceneAnimationBudget_->logicalWork();
    result.materialRevisions={inner.highestMaterialRevisionSeen_,inner.lastGoodMaterialRevision_,inner.rejectedMaterialRevision_};
    return result;
}

bool ImportedSceneParentStagingFailureProbe::conserved(const ImportedScenePayloadExecution& execution,
    const Witness& witness) noexcept
{
    const auto current=inspect(execution);
    const auto same=[](const auto& a,const auto& b) {
        return a.lock().get()==b.lock().get() && !a.owner_before(b) && !b.owner_before(a);
    };
    const auto sample=[&](const Sample& a,const Sample& b) {
        return a.present==b.present && (!a.present || (a.translation==b.translation && a.rotation==b.rotation
            && a.cameraTranslation==b.cameraTranslation && a.scale==b.scale && a.morph==b.morph
            && a.owner==b.owner && a.camera.has_value()==b.camera.has_value()
            && (!a.camera || HarmonicMIDI::grid::scenecomposition::sameCameraRecord(*a.camera,*b.camera))
            && same(a.deformation,b.deformation) && same(a.composition,b.composition)
            && same(a.worlds,b.worlds) && same(a.notes,b.notes)));
    };
    if (!same(current.source,witness.source) || !same(current.parent,witness.parent)
        || !same(current.material,witness.material) || current.materialRevisions!=witness.materialRevisions
        || !same(current.retentionBudget,witness.retentionBudget) || !same(current.workBudget,witness.workBudget)
        || current.attemptedWork<=witness.attemptedWork) return false;
    for (std::size_t i=0;i<current.history.size();++i)
    {
        const auto& a=current.history[i];const auto& b=witness.history[i];
        // The restricted parent route rejects note mappings. No proportional
        // test string capture or new owner/authority is needed for its history.
        if (!a.mappingEmpty || !b.mappingEmpty || !a.noteOwnerAbsent || !b.noteOwnerAbsent
            || a.frame!=b.frame || a.numerator!=b.numerator || a.denominator!=b.denominator
            || a.width!=b.width || a.height!=b.height || !(a.owner==b.owner)
            || !sample(a.current,b.current) || !sample(a.previous,b.previous)) return false;
    }
    return true;
}
#endif

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
              store, deformationBackend, backend))
{
}

bool ImportedSceneParentReceiptStaging::prepareDiagnostic(const std::string& diagnostic,std::string& error)
{
    diagnosticCapacity_=diagnostic.capacity();
    if (diagnosticCapacity_>=NativeCompositionRetentionBudget::kMaximumBytes)
        return reject(error,"imported parent diagnostic exceeds logical CPU capacity");
    if (!identity_->budget->work(diagnosticCapacity_+1,2,error)) return false;
    identity_->diagnosticRetention=NativeCompositionRetentionBudget::reserve(
        execution_.deformationExecution_->compositionBudget_,diagnosticCapacity_+1,error);
    return static_cast<bool>(identity_->diagnosticRetention);
}

bool ImportedSceneParentReceiptStaging::stage(ImportedAnimatedSceneReceipt& frame,std::string& error)
{
    static_assert(std::is_nothrow_move_constructible_v<ImportedAnimatedSceneReceipt>);
    const auto& owner=frame.runtimeInputs.compositionFrame->owner();
    const auto& incumbent=execution_.parentPublication_;
    const bool sameSource=incumbent && !incumbent->presentation->retired_
        && !incumbent->nativeOwner.owner_before(owner) && !owner.owner_before(incumbent->nativeOwner);
    if (sameSource) identity_->presentation=incumbent->presentation;
    else
    {
#if defined(ARBIT_TEST_IMPORTED_PARENT_STAGING_FAILURES)
        ImportedSceneParentStagingFailureProbe::allocation(ImportedSceneParentStagingFailureProbe::Site::TokenObject,frame);
#endif
        auto tokenObject=std::unique_ptr<ImportedSceneParentPresentation>(new ImportedSceneParentPresentation());
#if defined(ARBIT_TEST_IMPORTED_PARENT_STAGING_FAILURES)
        ImportedSceneParentStagingFailureProbe::allocation(ImportedSceneParentStagingFailureProbe::Site::TokenControlBlock,frame);
#endif
        auto token=std::shared_ptr<ImportedSceneParentPresentation>(std::move(tokenObject));
        token->retention_=identity_->retention;token->budget_=identity_->budget;
        identity_->presentation=std::move(token);
        replacePresentation_=true;
    }
    identity_->nativeOwner=owner;
    // All fallible outer assembly stays within the original inner rollback.
    // The three aliases retain the original outer debit through escaped values.
#if defined(ARBIT_TEST_IMPORTED_PARENT_STAGING_FAILURES)
    ImportedSceneParentStagingFailureProbe::allocation(ImportedSceneParentStagingFailureProbe::Site::PayloadAlias,frame);
#endif
    frame.payload=retainParentPayloadValue(frame.payload,identity_->retention,identity_->budget);
#if defined(ARBIT_TEST_IMPORTED_PARENT_STAGING_FAILURES)
    ImportedSceneParentStagingFailureProbe::allocation(ImportedSceneParentStagingFailureProbe::Site::SourceAlias,frame);
#endif
    frame.source=retainParentPayloadValue(frame.source,identity_->retention,identity_->budget);
    frame.frame.source=frame.source;
#if defined(ARBIT_TEST_IMPORTED_PARENT_STAGING_FAILURES)
    ImportedSceneParentStagingFailureProbe::allocation(ImportedSceneParentStagingFailureProbe::Site::NativeAlias,frame);
#endif
    frame.frame.nativeFrame=retainParentPayloadValue(frame.frame.nativeFrame,identity_->retention,identity_->budget);
    candidate_.use=use_;
    candidate_.payload=frame.payload;
    candidate_.usedLastGoodMaterial=frame.usedLastGoodMaterial;
#if defined(ARBIT_TEST_IMPORTED_PARENT_STAGING_FAILURES)
    ImportedSceneParentStagingFailureProbe::allocation(ImportedSceneParentStagingFailureProbe::Site::DiagnosticCopy,frame,
        frame.materialDiagnostic.size()>candidate_.materialDiagnostic.capacity());
#endif
    candidate_.materialDiagnostic=frame.materialDiagnostic;
    if (candidate_.materialDiagnostic.capacity()>diagnosticCapacity_)
        return reject(error,"imported parent copied diagnostic exceeds its reserved capacity");
    candidate_.deformationFrame=std::move(frame);
    candidate_.canonicalBlockCFrame=request_.runtimeInputs.canonicalBlockCFrame;
    candidate_.materialFrameTexture=request_.runtimeInputs.materialFrameTexture;
    return true;
}

void ImportedSceneParentReceiptStaging::publish(ImportedSceneExecutionReceipt& output) noexcept
{
    if (replacePresentation_ && execution_.parentPublication_)
        execution_.parentPublication_->presentation->retired_=true;
    execution_.parentPublication_=std::move(identity_);
    output.exchangeOwnedValues(candidate_);
}

bool ImportedScenePayloadExecution::inspectParentReceipt(const Receipt& receipt,const ImportedSceneRequest& expected,
    ImportedSceneUse use) const noexcept
{
    if (!deformationExecution_ || !deformationExecution_->chargeParentInspection()) return false;
    return receipt.parentPayloadIdentity_
        && receipt.parentPayloadIdentity_->presentation
        && !receipt.parentPayloadIdentity_->presentation->retired_
        && parentPublication_ && receipt.parentPayloadIdentity_->presentation==parentPublication_->presentation
        && expected.parentIntent && expected.deformation && receipt.use==use && receipt.deformationFrame
        && receipt.payload==receipt.deformationFrame->payload
        && receipt.deformationFrame->inspectParent(*expected.parentIntent,expected.frame,expected.structuralRevision,
            expected.evaluationRevision,expected.projectGeneration,expected.helperGeneration,expected.clipId,
            use==ImportedSceneUse::Preview?ImportedAnimationEvaluationOwner::Preview:ImportedAnimationEvaluationOwner::Export,
            expected.dimensions.width,expected.dimensions.height,&*expected.deformation,expected.sceneSnapshot,expected.sourceCameraNodeId,expected.camera,&expected.runtimeInputs);
}

// Named refusals for object-local requests this function does not admit. A
// transform that is invalid or not version 1, animation deformation, parenting,
// and source or render stable ids that are missing or equal all need a native
// evaluation owner. Several Scene Compose leaves are refused. So are a procedural
// scene without its leaf, a leaf without its procedural scene, note instancing,
// glass caustics, and diffraction gratings. One exact static leaf, alone or as
// the only leaf of a Scene Compose, passes this gate.
static const char* staticLocalRefusal(const ImportedSceneRequest& request)
{
    if (!visualimportedobjectlocal::valid(request.objectLocalTransform)
        || request.objectLocalTransform.version != 1 || request.deformation || request.parentIntent
        || !request.sourceStableId || !request.renderStableId
        || request.sourceStableId == request.renderStableId)
        return "Imported object local transforms require a native evaluation owner";
    if (!request.additionalImportedScenes.empty())
        return "Static local object transforms require one exact imported leaf; several Scene Compose leaves are not rendered yet";
    // A procedural snapshot alone has no imported leaf to move.
    if (request.sceneSnapshot && !request.importedScene)
        return "Static local object transforms require one exact imported leaf";
    // A Scene Compose leaf moves only beside its procedural scene.
    if (request.importedScene && !request.sceneSnapshot)
        return "Static local object transforms in a Scene Compose need its procedural scene";
    if (request.runtimeInputs.noteInstanceMapping)
        return "Static local object transforms do not yet admit note instancing";
    if (request.glassCaustic || request.diffractionMaterial)
        return "Static local object transforms do not yet admit glass caustics or diffraction gratings";
    return nullptr;
}

bool ImportedScenePayloadExecution::stageStaticLocalFrame(
    ImportedSceneUse use, const ImportedSceneRequest& request, const StaticOwner* incumbent,
    std::shared_ptr<const NativeSceneAnimationCompositionFrame>& candidate, std::string& error)
{
    using Frame = NativeSceneAnimationCompositionFrame;
    using Owner = NativeSceneAnimationCompositionOwner;
    using Budget = NativeSceneAnimationCompositionBudget;
    using namespace HarmonicMIDI::grid;
    const auto& local = request.objectLocalTransform;
    if (!request.sceneIndex || *request.sceneIndex >= 64 || !request.structuralRevision
        || request.evaluationRevision < request.structuralRevision || !request.projectGeneration
        || !request.helperGeneration || request.clipId <= 0 || !request.frame.rateNumerator
        || !request.frame.rateDenominator || !request.dimensions.width || !request.dimensions.height
        || request.dimensions.width > videorender::fixture3d::FixtureSceneRenderer::kMaxExtent
        || request.dimensions.height > videorender::fixture3d::FixtureSceneRenderer::kMaxExtent
        || std::uint64_t(request.dimensions.width) * request.dimensions.height
            > videorender::fixture3d::FixtureSceneRenderer::kMaxPixels
        || (request.objectNodeStableId == 0) != (request.objectMeshStableId == 0)
        || request.runtimeInputs.objectNodeStableId != request.objectNodeStableId
        || request.runtimeInputs.compositionFrame || request.runtimeInputs.localSceneFrame
        || request.runtimeInputs.previousMotion
        || request.runtimeInputs.animatedCamera || !request.runtimeInputs.animatedLights.empty()
        || !std::isfinite(request.runtimeInputs.timeSeconds)
        || !std::isfinite(request.runtimeInputs.objectScale)
        || request.runtimeInputs.objectScale < .0001f || request.runtimeInputs.objectScale > 1000)
        return reject(error, "Static local candidate request identity or runtime is invalid");
    for (const auto value : request.runtimeInputs.objectTranslationOffset)
        if (!std::isfinite(value) || std::abs(value) > 10000)
            return reject(error, "Static local overlay translation is invalid");
    for (const auto value : request.runtimeInputs.objectRotationDegrees)
        if (!std::isfinite(value) || std::abs(value) > 3600)
            return reject(error, "Static local overlay rotation is invalid");
    for (const auto value : request.runtimeInputs.cameraRotationDegrees)
        if (!std::isfinite(value) || std::abs(value) > 360)
            return reject(error, "Static local camera rotation is invalid");
    if (!arbitgpu::validRuntimeCameraFov(request.runtimeInputs.cameraVerticalFovDegrees))
        return reject(error, "Static local camera field of view is invalid");
    if (request.importedScene)
    {
        const auto& leaf = *request.importedScene;
        if (!request.sceneSnapshot || leaf.importedScene || leaf.sceneSnapshot || leaf.deformation
            || leaf.sourceStableId != request.sourceStableId || leaf.renderStableId != request.renderStableId
            || !visualanimationimport::sameAsset(leaf.asset, request.asset)
            || leaf.sceneIndex != request.sceneIndex || leaf.sourceCameraNodeId != request.sourceCameraNodeId
            || leaf.objectNodeStableId != request.objectNodeStableId
            || leaf.objectMeshStableId != request.objectMeshStableId
            || !visualimportedobjectlocal::same(leaf.objectLocalTransform, local)
            || request.light || !request.importedLights.empty())
            return reject(error, "Static local composition leaf is foreign");
    }
    else if (request.sceneSnapshot)
        return reject(error, "Static local candidate needs one exact imported leaf");
    const auto payload = use == ImportedSceneUse::Preview
        ? store_.resolvePreview(request.asset) : store_.resolveExport(request.asset);
    if (!payload || !visualanimationimport::sameAsset(payload->key(), request.asset)
        || payload->bytes().size() != request.asset.sourceByteSize)
        return reject(error, "Static local exact payload is not resident");
    if (!staticLocalBudget_) staticLocalBudget_ = std::make_shared<Budget>();
    if (!staticLocalCallerBudget_) staticLocalCallerBudget_=std::make_shared<NativeCompositionRetentionBudget>();
    const auto& budget = staticLocalBudget_;
    if (!budget->work(kImportedSceneRequestChargeBytes + sizeof(Receipt) + sizeof(StaticOwner), 4, error)
        || !budget->work(request.importedLights.intensityMultipliers.size(), Frame::kVisit, error)) return false;
    if (!importedlightcontrol::valid(request.importedLights))
        return reject(error, "Static local imported light controls are invalid");

    std::shared_ptr<const gltf::GlbNativeSceneAdmission> borrowedAdmission;
    gltf::GlbNativeSceneAdmission coldAdmission;
    const bool borrow = incumbent && incumbent->payload == payload && incumbent->sceneIndex == request.sceneIndex
        && incumbent->proceduralScene == (request.importedScene ? request.sceneSnapshot : nullptr)
        && (!incumbent->localAuthority || (incumbent->localAuthority->projectGeneration() == request.projectGeneration
            && incumbent->localAuthority->helperGeneration() == request.helperGeneration
            && incumbent->localAuthority->sourceStableId_ == request.sourceStableId
            && incumbent->localAuthority->renderStableId_ == request.renderStableId));
    if (borrow)
    {
        if (!incumbent->admission.document || !incumbent->admission.sourceScene)
            return reject(error,"Static local first document could not fit its retained admission; no second decode is allowed");
        if (incumbent->localAuthority) borrowedAdmission = incumbent->localAuthority->staticAdmission_;
    }
    else
    {
        gltf::GlbNativeRenderSeam seam(backend_);
        if (!seam.admit(payload->bytes(), request.sceneIndex, coldAdmission, error,
                request.objectNodeStableId, request.objectMeshStableId, budget)) return false;
    }
    const auto& admission = borrow ? incumbent->admission : coldAdmission;
    if (!budget->work(admission.importedLightDefinitions.size(), Frame::kVisit, error)) return false;
    if (request.importedLights.selectedDefinition && std::none_of(admission.importedLightDefinitions.begin(),
            admission.importedLightDefinitions.end(), [&](const auto& entry) { return entry.second == request.importedLights.selectedDefinition; }))
        return reject(error, "Static local selected light definition is outside the exact source scene");
    const auto& document = *admission.document;
    if (document.nodes.size() > 88 || document.selectedScene >= document.scenes.size())
        return reject(error, "Static local admitted source scene is invalid");
    if (!local.nodeStableId || local.nodeStableId > document.nodes.size())
        return reject(error, "Static local target node is outside the admitted original document");
    const auto selectedNode = local.nodeStableId - 1u;
    const auto& node = document.nodes[selectedNode];
    if ((node.mesh ? *node.mesh + 1u : 0u) != local.meshStableId)
        return reject(error, "Static local target mesh does not match its exact original node");
    const auto imported = admission.sourceScene ? admission.sourceScene : admission.scene;
    if (!imported || imported->objectCount > imported->objects.size()
        || imported->objectCount > Visual3DScene::kMaxObjects)
        return reject(error, "Static local retained source draw count is invalid");
    const auto n = document.nodes.size();
    const auto o = imported->objectCount;
    if (request.objectNodeStableId)
    {
        if (request.objectNodeStableId > n
            || document.nodes[request.objectNodeStableId-1u].mesh != std::optional<std::size_t>{request.objectMeshStableId-1u})
            return reject(error, "Static local old overlay target is outside the exact retained source scene");
        // One source-draw scan, independently of the later ancestry visits.
        const auto oldOverlaySourceDrawScanVisits = o;
        if (!budget->work(oldOverlaySourceDrawScanVisits, Frame::kVisit, error)) return false;
        if (std::none_of(imported->objects.begin(), imported->objects.begin()+o, [&](const auto& object) {
#if defined(ARBIT_TEST_IMPORTED_PARENT_STAGING_FAILURES)
                ++gltf::GlbColdAdmissionProbe::state().oldOverlaySourceDrawVisits;
#endif
                return object.indexCount && (object.id.value-1u)/visualimportedsceneidentity::kObjectPrimitiveStride+1u==request.objectNodeStableId;
            })) return reject(error, "Static local old overlay target is outside the exact retained source scene");
    }
    if (!budget->work(n*n + o*n, Frame::kVisit, error)
        || !budget->work(6, Frame::kProduct, error) || !budget->work(1, Frame::kInverse, error)) return false;
    // Membership follows original parents to the selected-scene roots.
    auto root = std::size_t(selectedNode);
    for (std::size_t depth=0; document.nodes[root].parent; ++depth)
    {
        if (depth >= n || *document.nodes[root].parent >= n)
            return reject(error, "Static local target original ancestry is invalid");
        root = *document.nodes[root].parent;
    }
    const auto& roots = document.scenes[document.selectedScene].rootNodes;
    if (std::find(roots.begin(), roots.end(), root) == roots.end())
        return reject(error, "Static local target is outside the exact selected scene");
    bool retainedDescendant = false;
    for (std::size_t i=0; i<o; ++i)
    {
        const auto& object = imported->objects[i];
        if (!object.indexCount || (object.id.value & 0x80000000u)) continue;
        auto sourceNode = std::size_t((object.id.value-1u) / visualimportedsceneidentity::kObjectPrimitiveStride);
        for (std::size_t depth=0; depth<n && sourceNode<n; ++depth)
        {
            if (sourceNode == selectedNode) { retainedDescendant = true; break; }
            if (!document.nodes[sourceNode].parent) break;
            sourceNode = *document.nodes[sourceNode].parent;
        }
    }
    if (!retainedDescendant)
        return reject(error, "Static local target has no retained draw descendant");

    // Candidate-local input census. Aggregate incumbent/caller/history charges
    // remain an activation prerequisite; escaped candidate aliases hold this charge.
    std::size_t bytes = sizeof(Owner) + sizeof(Frame) + 2*sizeof(gltf::GlbNativeSceneAdmission)
        + sizeof(Budget::Charge) + sizeof(payload) + sizeof(candidate) + sizeof(request.runtimeInputs)
        + sizeof(*payload)
        + sizeof(RetainedStaticLocalAllocation<visualmodelassetpayload::ImmutablePayload>)
        + sizeof(RetainedStaticLocalAllocation<Visual3DScene>)
        + sizeof(RetainedStaticLocalAllocation<gltf::GlbNativeSceneAdmission>);
    const auto add = [&](std::size_t count, std::size_t stride) {
        if (!stride || bytes > Budget::kMaximumBytes || count > (Budget::kMaximumBytes-bytes)/stride)
        { error = "Static local candidate retained inputs exceed logical CPU capacity"; return false; }
        if (!budget->work(count, 2*stride, error)) return false;
        bytes += count*stride; return true;
    };
    const auto text = [&](const std::string& value) {
        return value.capacity() != std::numeric_limits<std::size_t>::max() && add(value.capacity()+1, 1);
    };
    const auto sceneBytes = [&](const Visual3DScene& scene) {
        if (!add(1, sizeof(scene)) || !add(scene.textureTexels.capacity(), sizeof(SceneTexelRgba8))) return false;
        for (const auto& light : scene.lights)
            if (!text(light.environmentImage.asset.id) || !text(light.environmentImage.asset.contentSha256)) return false;
        return true;
    };
    if (!budget->work(bytes, 2, error) || !add(payload->bytes().capacity(), sizeof(std::uint8_t))
        || !text(payload->key().id) || !text(payload->key().contentSha256) || !text(payload->key().sourceMediaType)
        || !text(admission.metadata.generator)
        || !add(admission.metadata.extensionsUsed.capacity()+admission.metadata.extensionsRequired.capacity(), sizeof(std::string))
        || (admission.importedLightDefinitions.size() && !add(admission.importedLightDefinitions.size(),
            gltf::GlbNativeRenderSeam::lightDefinitionNodeBytes()))
        || !sceneBytes(*admission.scene)) return false;
    for (const auto* list : {&admission.metadata.extensionsUsed, &admission.metadata.extensionsRequired})
        for (const auto& value : *list) if (!text(value)) return false;
    if (request.light && (!text(request.light->environmentImage.asset.id)
        || !text(request.light->environmentImage.asset.contentSha256))) return false;
    if (!add(imported->cameraCount, sizeof(SceneCameraRecord))
        || !add(imported->lightCount, sizeof(SceneLightRecord))
        || !add(imported->objectCount, sizeof(arbitgpu::NativeImportedObjectWorld))) return false;
    for (std::size_t i=0; i<admission.scene->lightCount; ++i)
        if (!text(admission.scene->lights[i].environmentImage.asset.id)
            || !text(admission.scene->lights[i].environmentImage.asset.contentSha256)) return false;
    if (request.importedScene)
    {
        if (!add(2, sizeof(RetainedStaticLocalAllocation<Visual3DScene>))
            || !add(request.sceneSnapshot->lightCount, sizeof(SceneLightRecord))
            || !sceneBytes(*request.sceneSnapshot) || !add(2, sizeof(Visual3DScene))
            || !add(request.sceneSnapshot->textureTexels.capacity(), sizeof(SceneTexelRgba8))
            || !add(2*imported->textureTexels.capacity(), sizeof(SceneTexelRgba8))) return false;
        for (const auto& scene : {request.sceneSnapshot, imported})
            for (const auto& light : scene->lights)
                if (!text(light.environmentImage.asset.id) || !text(light.environmentImage.asset.contentSha256)) return false;
        // The retained procedural source, composed scene and frame vector are
        // distinct identity allocations. Reserve the third copy before issuance.
        for (std::size_t i=0; i<request.sceneSnapshot->lightCount; ++i)
            if (!text(request.sceneSnapshot->lights[i].environmentImage.asset.id)
                || !text(request.sceneSnapshot->lights[i].environmentImage.asset.contentSha256)) return false;
        // Scene Compose's existing validation/copy/identity visits, bounded by
        // fixed scene records and all retained texture lanes before allocation.
        if (!budget->work(2*sizeof(Visual3DScene), 4, error)
            || !budget->work(request.sceneSnapshot->textureTexels.size()+imported->textureTexels.size(), 4*sizeof(SceneTexelRgba8), error)) return false;
    }
    auto charge = Budget::reserve(budget, bytes, error);
    if (!charge) return false;
    auto retainedScene = borrow ? incumbent->admission.scene : admission.scene;
    if (request.importedScene && !borrow)
    {
        retainedScene = gltf::composeImportedVisual3DScene(*request.sceneSnapshot, *imported, error);
        if (!retainedScene) return false;
        // The actual result capacity must fit the prospective texture closure.
        if (retainedScene->textureTexels.capacity()
            > request.sceneSnapshot->textureTexels.capacity()+imported->textureTexels.capacity())
            return reject(error, "Static local composed texture capacity exceeds its prospective charge");
        for (std::size_t i=0; i<retainedScene->lights.size(); ++i)
            if (retainedScene->lights[i].environmentImage.asset.id.capacity() > request.sceneSnapshot->lights[i].environmentImage.asset.id.capacity()
                || retainedScene->lights[i].environmentImage.asset.contentSha256.capacity() > request.sceneSnapshot->lights[i].environmentImage.asset.contentSha256.capacity())
                return reject(error, "Static local composed light identity exceeds its prospective charge");
    }
#if defined(ARBIT_TEST_IMPORTED_PARENT_STAGING_FAILURES)
    const auto allocation = [](ImportedSceneParentStagingFailureProbe::Site site) {
        auto& probe = ImportedSceneParentStagingFailureProbe::state();
        if (probe.selected == site) { probe.hit = true; throw std::bad_alloc(); }
    };
    allocation(ImportedSceneParentStagingFailureProbe::Site::StaticLocalOwner);
#endif
    auto authority = std::shared_ptr<Owner>(new Owner(*this));
    authority->charge_ = charge;
#if defined(ARBIT_TEST_IMPORTED_PARENT_STAGING_FAILURES)
    allocation(ImportedSceneParentStagingFailureProbe::Site::StaticLocalSource);
#endif
    auto retainedAdmission=borrowedAdmission ? borrowedAdmission
        : borrow ? std::make_shared<const gltf::GlbNativeSceneAdmission>(admission)
        : std::make_shared<const gltf::GlbNativeSceneAdmission>(std::move(coldAdmission));
    if (borrow && !borrowedAdmission)
    {
        const auto& metadata=retainedAdmission->metadata;
        if (metadata.generator.capacity()>admission.metadata.generator.capacity()
            || metadata.extensionsUsed.capacity()>admission.metadata.extensionsUsed.capacity()
            || metadata.extensionsRequired.capacity()>admission.metadata.extensionsRequired.capacity())
            return reject(error,"Static local copied admission exceeds its prospective capacity");
        const std::pair<const std::vector<std::string>*,const std::vector<std::string>*> identities[]={
            {&metadata.extensionsUsed,&admission.metadata.extensionsUsed},
            {&metadata.extensionsRequired,&admission.metadata.extensionsRequired}};
        for (const auto& pair : identities)
            for (std::size_t i=0;i<pair.first->size();++i)
                if ((*pair.first)[i].capacity()>(*pair.second)[i].capacity())
                    return reject(error,"Static local copied metadata identity exceeds its prospective capacity");
    }
    authority->staticAdmission_=retainStaticLocalAllocation(std::move(retainedAdmission),charge);
    authority->payload_ = retainStaticLocalAllocation(payload, charge);
    authority->document_ = authority->staticAdmission_->document;
    authority->imported_ = retainStaticLocalAllocation(imported, charge);
    authority->scene_ = authority->imported_;
    if (retainedScene != imported) authority->scene_ = retainStaticLocalAllocation(retainedScene, charge);
    authority->procedural_ = request.importedScene ? retainStaticLocalAllocation(request.sceneSnapshot, charge) : nullptr;
    authority->sourceStableId_ = request.sourceStableId; authority->renderStableId_ = request.renderStableId;
    authority->sourceCameraNodeId_ = request.sourceCameraNodeId;
    authority->structural_ = request.structuralRevision; authority->evaluation_ = request.evaluationRevision;
    authority->project_ = request.projectGeneration; authority->helper_ = request.helperGeneration;
    authority->renderClip_ = request.clipId;
#if defined(ARBIT_TEST_IMPORTED_PARENT_STAGING_FAILURES)
    allocation(ImportedSceneParentStagingFailureProbe::Site::StaticLocalFrame);
#endif
    auto frame = std::shared_ptr<Frame>(new Frame(*this));
    frame->charge_ = charge; frame->owner_ = authority; frame->localTransform_ = local;
    frame->callerBudget_=staticLocalCallerBudget_;
    frame->callerMaximumBytes_=NativeCompositionRetentionBudget::kCallerMaximumBytes;
    frame->reserveCaller_=[](const std::shared_ptr<const void>& opaque,std::size_t bytes,std::string& diagnostic)->std::shared_ptr<const void> {
        auto pool=std::const_pointer_cast<NativeCompositionRetentionBudget>(std::static_pointer_cast<const NativeCompositionRetentionBudget>(opaque));
        if (bytes>NativeCompositionRetentionBudget::kCallerMaximumBytes-sizeof(NativeCompositionRetentionBudget::Charge))
        { diagnostic="static local caller charge exceeds its existing cap"; return {}; }
        return NativeCompositionRetentionBudget::reserve(pool,bytes+sizeof(NativeCompositionRetentionBudget::Charge),diagnostic);
    };
    frame->cameraTranslation_=request.runtimeInputs.cameraTranslationOffset;
    frame->cameraRotation_=request.runtimeInputs.cameraRotationDegrees;
    frame->cameraFov_=request.runtimeInputs.cameraVerticalFovDegrees;
    frame->use_ =use == ImportedSceneUse::Preview ? ImportedAnimationEvaluationOwner::Preview : ImportedAnimationEvaluationOwner::Export;
    frame->time_ = request.frame; frame->width_ = request.dimensions.width; frame->height_ = request.dimensions.height;
    frame->sourceCameraNodeId_ = request.sourceCameraNodeId;
    frame->overlayNodeStableId_ = request.objectNodeStableId;
    frame->overlayTranslation_ = request.runtimeInputs.objectTranslationOffset;
    frame->overlayRotation_ = request.runtimeInputs.objectRotationDegrees; frame->overlayScale_ = request.runtimeInputs.objectScale;
    frame->timeSeconds_ = request.runtimeInputs.timeSeconds; frame->imageOutput_ = request.runtimeInputs.imageOutput;
    frame->rawExportMask_ = request.runtimeInputs.rawExportMask; frame->linearColor_ = request.runtimeInputs.linearColor;
    SceneTransform3D offset;
    offset.translation = {float(local.translation[0]), float(local.translation[1]), float(local.translation[2])};
    offset.rotation = arbitgpu::runtimeEulerQuaternion({float(local.rotationDegrees[0]), float(local.rotationDegrees[1]), float(local.rotationDegrees[2])});
    offset.scale = {float(local.scale[0]), float(local.scale[1]), float(local.scale[2])};
    Frame::Matrix inverse;
    if (!Frame::inverseAffine(node.worldTransform, inverse))
        return reject(error, "Static local selected original world is singular or nonfinite");
    frame->correction_ = Frame::multiply(Frame::multiply(node.worldTransform, Frame::compose(offset)), inverse);
    if (!budget->work(4, Frame::kProduct, error)) return false;
    SceneTransform3D overlay;
    overlay.translation = {frame->overlayTranslation_[0], frame->overlayTranslation_[1], frame->overlayTranslation_[2]};
    overlay.rotation = arbitgpu::runtimeEulerQuaternion(frame->overlayRotation_);
    overlay.scale = {frame->overlayScale_, frame->overlayScale_, frame->overlayScale_};
    frame->overlay_ = Frame::compose(overlay);
    if ((request.camera && !visualimportedscenerender::detail::validCamera(*request.camera))
        || (request.light && !visualimportedscenerender::detail::validLight(*request.light)))
        return reject(error, "Static local graph record replacement is invalid");
    frame->cameraOverride_ = request.camera; frame->lightOverride_ = request.light;
    // Stored for identity so a frame rendered with modulation is never reused without it.
    // Its bytes are already in the reserve charge above.
    frame->lightOverrides_ = request.runtimeInputs.lightOverrides;
    const auto lightCount = request.importedScene ? request.sceneSnapshot->lightCount : imported->lightCount;
    frame->cameras_.reserve(imported->cameraCount); frame->lights_.reserve(lightCount);
    if (frame->cameras_.capacity() != imported->cameraCount || frame->lights_.capacity() != lightCount)
        return reject(error, "Static local record capacity exceeds its prospective charge");
    // A float rigid matrix has round-off from quaternion products. This tight
    // Gram/determinant check admits that representation only; authored local
    // scale is checked exactly before interpreting a camera as unit scale.
    const auto rigid = [](const Frame::Matrix& matrix) {
        constexpr double tolerance = 16*std::numeric_limits<float>::epsilon();
        for (std::size_t a=0; a<3; ++a)
            for (std::size_t b=0; b<3; ++b)
            {
                double dot = 0;
                for (std::size_t row=0; row<3; ++row) dot += double(matrix[4*a+row])*matrix[4*b+row];
                if (!std::isfinite(dot) || std::abs(dot-(a==b?1.0:0.0)) > tolerance) return false;
            }
        const double determinant = double(matrix[0])*(double(matrix[5])*matrix[10]-double(matrix[9])*matrix[6])
            -double(matrix[4])*(double(matrix[1])*matrix[10]-double(matrix[9])*matrix[2])
            +double(matrix[8])*(double(matrix[1])*matrix[6]-double(matrix[5])*matrix[2]);
        return std::isfinite(determinant) && std::abs(determinant-1) <= tolerance;
    };
    const auto record = [&](std::uint32_t nodeId, auto& value, bool camera) {
        if (!nodeId || nodeId > n) return true;
        const auto& original = document.nodes[nodeId-1u];
        if (camera ? !original.camera : !original.light) return true; // synthetic records stay original
        if (!budget->work(n, Frame::kVisit, error) || !budget->work(1, Frame::kProduct, error)
            || !budget->work(3, Frame::kInverse, error) || !budget->work(sizeof(value), 4, error)) return false;
        bool affected = false;
        if (!frame->descendant(nodeId-1u, affected)) return false;
        if (!affected) return true;
        const auto world = Frame::multiply(frame->correction_, original.worldTransform);
        if (!Frame::decompose(world, value.transform))
            return reject(error, "Static local camera or light world is not representable");
        if (camera)
        {
            if (local.scale != std::array<double,3>{1,1,1}
                || !rigid(original.worldTransform) || !rigid(frame->correction_) || !rigid(world))
                return reject(error, "Static local camera world is not representable as an unscaled rigid transform");
            // Unit scale follows the exact authored offset and proved rigid
            // source/correction. Decomposition length drift is not authored scale.
            value.transform.scale = {1,1,1};
        }
        return true;
    };
    for (std::size_t i=0; i<imported->cameraCount; ++i)
    {
        auto value = imported->cameras[i];
        if (!record(value.id.value, value, true)) return false;
        if (!visualimportedscenerender::detail::validCamera(value))
            return reject(error, "Static local final camera record is invalid");
        frame->cameras_.push_back(value);
        if (request.sourceCameraNodeId ? value.id.value == *request.sourceCameraNodeId : value.id == authority->scene_->activeCamera)
            frame->camera_ = value;
    }
    if (request.sourceCameraNodeId && (!gltf::GlbNativeRenderSeam::selectedSourceCamera(*authority->staticAdmission_, *request.sourceCameraNodeId)
        || !frame->camera_)) return reject(error, "Static local selected camera is not a genuine source record");
    for (std::size_t i=0; i<lightCount; ++i)
    {
        if (!budget->work(sizeof(SceneLightRecord), 4, error)
            || !budget->consumeLightIdentityWork(request.importedScene ? authority->scene_->lights[i] : imported->lights[i], 4)) return false;
        auto value = request.importedScene ? authority->scene_->lights[i] : imported->lights[i];
        const auto& sourceIdentity = request.importedScene ? authority->scene_->lights[i] : imported->lights[i];
        if (value.environmentImage.asset.id.capacity() > sourceIdentity.environmentImage.asset.id.capacity()
            || value.environmentImage.asset.contentSha256.capacity() > sourceIdentity.environmentImage.asset.contentSha256.capacity())
            return reject(error, "Static local frame light identity exceeds its prospective charge");
        if (!request.importedScene && !record(value.id.value, value, false)) return false;
        const auto definition = authority->staticAdmission_->importedLightDefinitions.find(value.id.value);
        if (!request.importedScene && definition != authority->staticAdmission_->importedLightDefinitions.end())
        {
            const auto multiplier = request.importedLights.intensityMultipliers.find(definition->second);
            if (multiplier != request.importedLights.intensityMultipliers.end())
            {
                const auto intensity = double(value.intensity)*multiplier->second;
                if (!std::isfinite(intensity) || intensity < 0 || intensity > 1000000)
                    return reject(error, "Static local light intensity is out of bounds");
                value.intensity = float(intensity);
            }
        }
        if (!visualimportedscenerender::detail::validLight(value))
            return reject(error, "Static local final light record is invalid");
        frame->lights_.push_back(std::move(value));
        if (frame->lights_.back().environmentImage.asset.id.capacity() > sourceIdentity.environmentImage.asset.id.capacity()
            || frame->lights_.back().environmentImage.asset.contentSha256.capacity() > sourceIdentity.environmentImage.asset.contentSha256.capacity())
            return reject(error, "Static local retained frame light identity exceeds its prospective charge");
    }
    if (request.importedScene && !request.sourceCameraNodeId)
    {
        const auto* camera = visual3d_detail::findById(authority->scene_->cameras,
            authority->scene_->cameraCount, authority->scene_->activeCamera);
        frame->camera_ = camera ? std::optional<SceneCameraRecord>{*camera} : std::nullopt;
    }
    if (frame->camera_ && !visualimportedscenerender::detail::validCamera(*frame->camera_))
        return reject(error, "Static local selected final camera record is invalid");
    // Checked once per listed object, before publication, never per draw. The
    // overlay stays out: each raster applies its runtime transform itself. A
    // single leaf lists every drawn object. A Scene Compose lists only the
    // imported objects under the moved node, by raw id. Procedural objects and
    // unaffected imported objects keep their authored chains, which a raster
    // already uses for any object without an entry.
    auto worlds = std::make_shared<arbitgpu::NativeImportedObjectWorlds>();
    worlds->reserve(imported->objectCount);
    if (worlds->capacity() != imported->objectCount)
        return reject(error, "Static local world list capacity exceeds its prospective charge");
    for (std::size_t i=0; i<authority->scene_->objectCount; ++i)
    {
        const auto& object = authority->scene_->objects[i];
        if (!object.indexCount) continue;
        if (request.importedScene)
        {
            // Procedural ids carry the high bit, as in the retained-descendant scan.
            if (object.id.value & 0x80000000u) continue;
            bool affected = false;
            const auto objectNode = std::size_t((object.id.value-1u) / visualimportedsceneidentity::kObjectPrimitiveStride);
            if (!frame->descendant(objectNode, affected))
                return reject(error, "Static local composed object ancestry is invalid");
            if (!affected) continue;
        }
        Frame::Matrix world {};
        if (!frame->checkedStaticWorld(authority->scene_, object.id, world, error, false)) return false;
        if (worlds->size() >= imported->objectCount)
            return reject(error, "Static local world list exceeds its prospective charge");
        worlds->push_back({object.id, world});
    }
    // A single leaf names each drawn object exactly once. A Scene Compose names at
    // least one imported object. The retained-descendant scan above already requires
    // one, so this second check should not fire.
    if (!request.importedScene && !arbitgpu::nativeImportedObjectWorldsExactCover(*authority->scene_, *worlds))
        return reject(error, "Static local single leaf worlds must name every drawn object exactly");
    if (request.importedScene && worlds->empty())
        return reject(error, "Static local Scene Compose moves no drawn imported object");
    frame->objectWorlds_ = std::move(worlds);
    candidate = std::move(frame);
    return true;
}

bool ImportedScenePayloadExecution::executePreview(
    const ImportedSceneRequest& request,
    ImportedSceneExecutionReceipt& output,
    std::string& error)
{
    if ((request.objectNodeStableId != 0 || request.objectMeshStableId != 0) && !request.parentIntent
        && !request.deformation && !request.importedScene && !request.sceneSnapshot && !output.deformationFrame)
    {
        ImportedSceneExecutionReceipt candidate;
        if (!execute(ImportedSceneUse::Preview, request, candidate, error))
        {
            const auto incumbent = staticOwners_.find(request.clipId);
            if (!output.deformationFrame && output.admission.scene
                && (incumbent == staticOwners_.end() || incumbent->second.admission.scene != output.admission.scene
                    || incumbent->second.payload != output.payload))
                clearCallerReceipt(output);
            return false;
        }
        if (candidate.pending)
        {
            // Pending keeps the caller's receipt and changes only the flag.
            output.pending = true;
            error.clear();
            return true;
        }
        clearCallerReceipt(output);
        output = std::move(candidate);
        return true;
    }
    return execute(ImportedSceneUse::Preview, request, output, error);
}

bool ImportedScenePayloadExecution::executeExport(
    const ImportedSceneRequest& request,
    ImportedSceneExecutionReceipt& output,
    std::string& error)
{
    if ((request.objectNodeStableId != 0 || request.objectMeshStableId != 0) && !request.parentIntent
        && !request.deformation && !request.importedScene && !request.sceneSnapshot && !output.deformationFrame)
    {
        ImportedSceneExecutionReceipt candidate;
        if (!execute(ImportedSceneUse::Export, request, candidate, error))
        {
            const auto incumbent = staticOwners_.find(request.clipId);
            if (!output.deformationFrame && output.admission.scene
                && (incumbent == staticOwners_.end() || incumbent->second.admission.scene != output.admission.scene
                    || incumbent->second.payload != output.payload))
                clearCallerReceipt(output);
            return false;
        }
        if (candidate.pending)
        {
            // Pending keeps the caller's receipt and changes only the flag.
            output.pending = true;
            error.clear();
            return true;
        }
        clearCallerReceipt(output);
        output = std::move(candidate);
        return true;
    }
    return execute(ImportedSceneUse::Export, request, output, error);
}

ImportedScenePayloadExecution::Receipt::ParentPresentationGuard
ImportedScenePayloadExecution::parentPresentationGuard(int clipId) const noexcept
{
    Receipt::ParentPresentationGuard guard;
    if (parentPublication_)
    {
        // Saturation refuses new proportional work. These fixed reads/copies
        // still record attempted work and honor previously proved retirement.
        parentPublication_->budget->consumeWork(1,16);
        if (parentPublication_->clip==clipId) guard.identity_=parentPublication_->presentation;
    }
    return guard;
}

ImportedScenePayloadExecution::Receipt::ParentPresentationGuard
ImportedScenePayloadExecution::parentPresentationGuard(
    const std::shared_ptr<const ImportedSceneParentPresentation>& token,int clipId) const noexcept
{
    if (!token) return parentPresentationGuard(clipId);
    token->budget_->consumeWork(1,24);
    Receipt::ParentPresentationGuard guard;
    guard.identity_=token;
    guard.foreign_=!parentPublication_ || parentPublication_->clip!=clipId
        || parentPublication_->presentation!=token;
    return guard;
}

std::shared_ptr<const ImportedSceneParentPresentation>
ImportedScenePayloadExecution::parentPresentationToken(const Receipt& receipt) const noexcept
{
    if (!receipt.parentPayloadIdentity_) return {};
    receipt.parentPayloadIdentity_->budget->consumeWork(1,24);
    const auto& token=receipt.parentPayloadIdentity_->presentation;
    return token && !token->retired_ && parentPublication_ && parentPublication_->presentation==token?token:nullptr;
}

void ImportedScenePayloadExecution::retireParentPresentation() noexcept
{
    if (!parentPublication_ || parentPublication_->presentation->retired_) return;
    // No text comparison, Store walk, clone or allocation occurs on this path.
    // The fixed cleanup debit saturates the same permanent attempted ledger.
    parentPublication_->budget->consumeWork(1,32);
    parentPublication_->presentation->retired_=true;
    if (deformationExecution_) deformationExecution_->reset();
}

void ImportedScenePayloadExecution::reset() noexcept
{
    retireParentPresentation();
    parentPublication_.reset();
    parentPublicationReady_=true;
    if (deformationExecution_ != nullptr)
        deformationExecution_->reset();
    staticOwners_.clear();
    staticLocalBudget_.reset();
    staticLocalCallerBudget_.reset();
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
    parentPublicationReady_=true;
    if (parentPublication_ && !parentPublication_->presentation->retired_)
    {
        const auto& old=*parentPublication_;
        // Positive public generation publication is fixed retirement authority,
        // even when there is no remaining allowance for the incoming payload.
        old.budget->consumeWork(1,32);
        if ((projectGeneration && old.project!=projectGeneration)
            || (helperGeneration && old.helper!=helperGeneration))
            retireParentPresentation();
        else if (old.clip==clipId)
        {
            // A length mismatch proves changed text with fixed work. Equal-size
            // contents require prospective work before any proportional scan.
            if (old.exactPayloadIdentity.size()!=exactPayloadIdentity.size())
                retireParentPresentation();
            else if (exactPayloadIdentity.capacity()>=NativeSceneAnimationCompositionBudget::kMaximumBytes
                || !old.budget->consumeWork(old.exactPayloadIdentity.capacity()+1,2))
                parentPublicationReady_=false;
            else if (old.exactPayloadIdentity!=exactPayloadIdentity)
                retireParentPresentation();
        }
    }
    if (parentPublication_ && staticOwners_.empty())
    {
        // The animated renderer owns its device authority. With no static
        // owners, no legacy device-string construction or owner scan is needed.
        projectGeneration_=projectGeneration;helperGeneration_=helperGeneration;
        retiredOwnerSinceLastAttempt_=false;
        return;
    }
    if (!parentPublicationReady_) return;
    synchronizeGenerations(projectGeneration, helperGeneration);
    synchronizeDevice();
    bool retired = false;
    for (auto found = staticOwners_.begin(); found != staticOwners_.end();)
    {
        if (found->first == clipId
            && found->second.exactPayloadIdentity != exactPayloadIdentity
            && !(found->second.payload && !found->second.proceduralScene
                && sameStaticObjectSelectionSource(found->second.exactPayloadIdentity, exactPayloadIdentity)))
        {
            retired = true;
            found = staticOwners_.erase(found);
        }
        else
            ++found;
    }
    retiredOwnerSinceLastAttempt_ = retired;
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
    // Early successes below return before the final publish, so a pending flag
    // from an earlier call must not survive into them.
    output.pending = false;
    const bool localTransform = !visualimportedobjectlocal::same(request.objectLocalTransform, {})
        || (request.importedScene && !visualimportedobjectlocal::same(request.importedScene->objectLocalTransform, {}));
    if (localTransform)
    {
        if (const auto refusal = staticLocalRefusal(request)) return reject(error, refusal);
    }
    // Several imported leaves select at most one object. Execution refuses a
    // second selecting leaf on every frame, so refuse it here, before any
    // receipt or owner changes, with the same message.
    if (!request.additionalImportedScenes.empty())
    {
        std::size_t selecting = (request.deformation ? request.deformation->pose.nodeStableId
            : request.objectNodeStableId) != 0 ? 1u : 0u;
        for (const auto& entry : request.additionalImportedScenes)
        {
            if (!entry.scene)
                return reject(error, "Scene Compose imported request or explicit lighting replacement is invalid");
            if ((entry.deformation ? entry.deformation->pose.nodeStableId : entry.scene->objectNodeStableId) != 0)
                ++selecting;
        }
        if (selecting > 1)
            return reject(error, "Scene Compose refused: more than one imported leaf selects an object");
    }
    std::shared_ptr<const NativeCompositionRetentionBudget::Charge> parentPayloadCharge;
    std::shared_ptr<ImportedSceneExecutionReceipt::ParentPayloadIdentity> parentPayloadIdentity;
    // A non-parent attempt clears the caller's receipt only when it fails. A pending
    // or published attempt keeps it, so the viewport holds the last frame.
    bool keepCallerReceipt = false;
    const auto clearCallerReceiptUnlessKept = [&] {
        if (!request.parentIntent && !keepCallerReceipt) clearCallerReceipt(output);
    };
    struct ClearCallerReceiptUnlessKept final
    {
        const decltype(clearCallerReceiptUnlessKept)& action;
        ~ClearCallerReceiptUnlessKept() { action(); }
    } clearCallerReceiptOnExit {clearCallerReceiptUnlessKept};
    // A non-parent static owner inserted by an object-local attempt belongs to
    // that attempt alone. A refused local attempt erases it, so its admitted
    // charge returns to the executor budget. A released v0 attempt (no object-
    // local transform) keeps its owner and first document: its caller may
    // activate locally later, and that activation must reuse the document
    // instead of decoding again. Pending and published attempts keep the owner.
    bool insertedStaticOwner = false, keepStaticOwner = false;
    const auto rollbackStaticOwnerUnlessKept = [&] {
        if (insertedStaticOwner && !keepStaticOwner) staticOwners_.erase(request.clipId);
    };
    struct RollbackStaticOwnerUnlessKept final
    {
        const decltype(rollbackStaticOwnerUnlessKept)& action;
        ~RollbackStaticOwnerUnlessKept() { action(); }
    } rollbackStaticOwnerOnExit {rollbackStaticOwnerUnlessKept};
    if (request.parentIntent)
    {
        if (!deformationExecution_)
            return reject(error,"native imported scene deformation execution is unavailable");
        auto& execution=*deformationExecution_;
        if (!execution.compositionBudget_) execution.compositionBudget_=std::make_shared<NativeCompositionRetentionBudget>();
        if (!execution.sceneAnimationBudget_) execution.sceneAnimationBudget_=std::make_shared<NativeSceneAnimationCompositionBudget>();
        using Identity=ImportedSceneExecutionReceipt::ParentPayloadIdentity;
        execution.sceneAnimationBudget_->consumeWork(1,160);
        const auto& leaf=request.importedScene;
        // Outer optional/zero/contradictory IDs are request companions.
        // Only a coherent positive retained leaf supplies source authority.
        const bool coherentLeaf=leaf && leaf->deformation && request.deformation
            && leaf->sourceStableId && leaf->deformation->deformationStableId
            && leaf->sourceStableId==leaf->deformation->sourceStableId
            && request.deformation->sourceStableId==leaf->sourceStableId
            && request.deformation->deformationStableId==leaf->deformation->deformationStableId
            && request.parentIntent->importedSourceStableId==leaf->sourceStableId;
        const auto fixedReplacement=[&](const Identity& old) {
            return (request.structuralRevision && old.structural!=request.structuralRevision)
                || (request.projectGeneration && old.project!=request.projectGeneration)
                || (request.helperGeneration && old.helper!=request.helperGeneration)
                || (request.clipId && old.clip!=request.clipId)
                || (coherentLeaf && (old.source!=leaf->sourceStableId
                    || old.deformation!=leaf->deformation->deformationStableId));
        };
        if (parentPublication_ && !parentPublication_->presentation->retired_ && fixedReplacement(*parentPublication_))
            retireParentPresentation();
        if (output.parentPayloadIdentity_)
        {
            const auto& old=*output.parentPayloadIdentity_;
            if (old.presentation->retired_ || fixedReplacement(old))
                clearCallerReceipt(output);
        }
        // The incumbent stays in the caller. No receipt or native/source owner
        // is copied for rollback. Charge both caller/result headers, generation
        // metadata, incoming identity copy, alias temporaries and the animated
        // request/frame/runtime staging before constructing any new value.
        constexpr auto fixedBytes=2*sizeof(ImportedSceneExecutionReceipt)+sizeof(Identity)
            +kImportedAnimatedSceneRequestChargeBytes+sizeof(ImportedAnimatedSceneReceipt)
            +sizeof(arbitgpu::NativeImportedSceneRuntimeInputs)+sizeof(NativeCompositionRetentionBudget::Charge)
            +10*sizeof(std::shared_ptr<const void>)+3*10*sizeof(std::shared_ptr<const void>)
            +3*sizeof(Receipt::ParentPresentationGuard)+sizeof(ImportedSceneParentPresentation)
            +3*sizeof(std::shared_ptr<const ImportedSceneParentPresentation>)
            +sizeof(ImportedSceneParentReceiptStaging)+sizeof(NativeCompositionRetentionBudget::Charge)
            +sizeof(std::unique_ptr<ImportedSceneParentPresentation>)+parentReceiptDisposalBytes;
#if defined(ARBIT_TEST_IMPORTED_PARENT_STAGING_FAILURES)
        // Fixed fault state and original/current read-only witnesses overlap.
        constexpr auto testBytes=sizeof(ImportedSceneParentStagingFailureProbe::State)
            +4*sizeof(ImportedSceneParentStagingFailureProbe::Witness)
            +3*sizeof(ImportedSceneParentStagingFailureProbe::Sample)
            +sizeof(ImportedSceneParentStagingFailureProbe::Scope)+16*sizeof(std::shared_ptr<const void>)
            +3*sizeof(ImportedSceneParentStagingFailureProbe::Diagnostic)+sizeof(std::string);
        if (!execution.sceneAnimationBudget_->work(testBytes,2,error)) return false;
#else
        constexpr std::size_t testBytes=0;
#endif
        if (!execution.sceneAnimationBudget_->work(fixedBytes,2,error)
            || !execution.sceneAnimationBudget_->work(parentReceiptExchangeWork,1,error)) return false;
        if (request.staticPayloadIdentity.capacity()>=NativeSceneAnimationCompositionBudget::kMaximumBytes)
            return reject(error,"imported parent outer payload identity exceeds logical CPU capacity");
        const auto identityBytes=request.staticPayloadIdentity.capacity()+1;
        const auto incumbentBytes=output.parentPayloadIdentity_
            ? output.parentPayloadIdentity_->exactPayloadIdentity.capacity()+1:0;
        if (!execution.sceneAnimationBudget_->work(identityBytes+incumbentBytes,2,error)) return false;
        // Explicit retirement never rolls an old generation back into view.
        // Intent, evaluation epoch, extent and attempted procedural replacements
        // are transactional and do not themselves retire the incumbent source.
        if (output.parentPayloadIdentity_)
        {
            const auto& old=*output.parentPayloadIdentity_;
            // Store resolution visits at most 16 already admitted exact keys.
            // String equality first checks length; matching operands are bounded
            // by the Store's stable-ID/hash/GLB media-type admission contract.
            constexpr auto resolveWork=visualmodelassetpayload::kMaxResidentPayloads
                *(24+2*(visualmodelasset::kMaximumStableAssetIdBytes+visualmodelasset::kSha256HexBytes+17));
            if (!execution.sceneAnimationBudget_->work(resolveWork,1,error)) return false;
            const auto payload=use==ImportedSceneUse::Preview?store_.resolvePreview(request.asset):store_.resolveExport(request.asset);
            if (old.exactPayloadIdentity!=request.staticPayloadIdentity || payload!=output.payload)
            {
                if (parentPublication_ && old.presentation==parentPublication_->presentation) retireParentPresentation();
                clearCallerReceipt(output);
            }
        }
        parentPayloadCharge=NativeCompositionRetentionBudget::reserve(execution.compositionBudget_,fixedBytes+identityBytes+testBytes,error);
        if (!parentPayloadCharge) return false;
        parentPayloadIdentity=std::make_shared<Identity>();
        parentPayloadIdentity->retention=parentPayloadCharge;
        parentPayloadIdentity->budget=execution.sceneAnimationBudget_;
        parentPayloadIdentity->exactPayloadIdentity=request.staticPayloadIdentity;
        if (parentPayloadIdentity->exactPayloadIdentity.capacity()>request.staticPayloadIdentity.capacity())
            return reject(error,"imported parent copied outer identity exceeds its reserved capacity");
        parentPayloadIdentity->structural=request.structuralRevision;
        parentPayloadIdentity->project=request.projectGeneration;parentPayloadIdentity->helper=request.helperGeneration;
        parentPayloadIdentity->clip=request.clipId;
        if (request.deformation)
        {
            parentPayloadIdentity->source=request.deformation->sourceStableId;
            parentPayloadIdentity->deformation=request.deformation->deformationStableId;
        }
    }
    publishStaticPayload(request.clipId, request.projectGeneration,
                         request.helperGeneration, request.staticPayloadIdentity);
    if (request.parentIntent && !parentPublicationReady_)
        return reject(error,"imported parent publication identity exceeds permanent work or logical CPU capacity");
    // When this attempt retires the clip's previous owner, any frame from the
    // retired owner is stale: a rejected execution must not leave it visible
    // through the caller's receipt.
    const auto previousOwner = staticOwners_.find(request.clipId);
    if (previousOwner != staticOwners_.end()
        && (previousOwner->second.glassCaustic.has_value() != request.glassCaustic.has_value()
            || (request.glassCaustic
                && !visualimportedscenerender::sameGlassCaustic(
                    *previousOwner->second.glassCaustic, *request.glassCaustic))))
    {
        staticOwners_.erase(previousOwner);
        clearCallerReceipt(output);
    }
    // A selected caustic request must publish only its own complete native
    // result, including when a malformed caller reuses the serialized identity.
    // A retired owner clears only a parent receipt here. A non-parent receipt is
    // kept on pending and cleared on failure by clearCallerReceiptOnExit.
    if (request.parentIntent ? (retiredOwnerSinceLastAttempt_ || request.glassCaustic.has_value()) : request.glassCaustic.has_value())
        clearCallerReceipt(output);
    const bool hasMaterialFrame = (request.material
        && surfacematerialbinding::hasGraphFrameInput(request.material->binding))
        || (request.diffractionMaterial && request.diffractionMaterial->graphFrame.has_value())
        || geometrysurfacematerial::objectFrameEndpoint(request.surfacePrograms).has_value()
        || std::any_of(request.additionalImportedScenes.begin(), request.additionalImportedScenes.end(),
            [](const auto& entry) { return entry.scene && entry.scene->material
                && surfacematerialbinding::hasGraphFrameInput(entry.scene->material->binding); });
    if (hasMaterialFrame != static_cast<bool>(request.runtimeInputs.materialFrameTexture)
        || (hasMaterialFrame
            && !arbitgpu::validMaterialFrameTexture(request.runtimeInputs.materialFrameTexture)))
        return reject(error, "imported scene material Frame requires its exact owned native texture");
    const bool importedComposition = request.importedScene != nullptr;
    const bool composedScene = request.sceneSnapshot != nullptr && !importedComposition;
    // Only a composed timeline scene without imported deformation keeps prior
    // positions. Other routes ignore the flag unless Motion is requested.
    const bool deformedSceneMotion = request.deformedSceneMotion && composedScene && !request.deformation;
    if (request.deformedSceneMotion && !deformedSceneMotion && arbitgpu::sceneUsesMotionPass(request.runtimeInputs))
        return reject(error, "Timeline Geometry3D motion requires a composed scene without imported deformation");
    if (importedComposition)
    {
        const auto& leaf = *request.importedScene;
        if (!request.sceneSnapshot || !visualanimationimport::sameAsset(request.asset, leaf.asset)
            || request.sceneIndex != leaf.sceneIndex
            || !visualimportedobjectlocal::same(request.objectLocalTransform, leaf.objectLocalTransform)
            || request.objectNodeStableId != leaf.objectNodeStableId
            || request.objectMeshStableId != leaf.objectMeshStableId
            || request.sourceCameraNodeId != leaf.sourceCameraNodeId
            || request.deformation.has_value() != leaf.deformation.has_value()
            || request.material.has_value() != leaf.material.has_value()
            || (request.material && (!visualimportedscenerender::sameMaterialRequest(*request.material, *leaf.material)
                || request.material->sceneSnapshot != leaf.material->sceneSnapshot))
            || (request.deformation && (request.deformation->sourceStableId != leaf.deformation->sourceStableId
                || request.deformation->deformationStableId != leaf.deformation->deformationStableId
                || request.deformation->meshStableId != leaf.deformation->meshStableId
                || request.deformation->animationClipStableId != leaf.deformation->animationClipStableId
                || request.deformation->crossfadeClipStableId != leaf.deformation->crossfadeClipStableId
                || request.deformation->clipName != leaf.deformation->clipName
                || request.deformation->crossfadeClipName != leaf.deformation->crossfadeClipName
                || request.deformation->combinationMode != leaf.deformation->combinationMode
                || request.deformation->schedule != leaf.deformation->schedule
                || request.deformation->pose.boneEnabled != leaf.deformation->pose.boneEnabled
                || request.deformation->pose.morphEnabled != leaf.deformation->pose.morphEnabled
                || request.deformation->pose.morphTargetIndex != leaf.deformation->pose.morphTargetIndex
                || request.deformation->pose.nodeStableId != leaf.deformation->pose.nodeStableId
                || request.deformation->pose.meshStableId != leaf.deformation->pose.meshStableId
                || request.deformation->pose.boneStableId != leaf.deformation->pose.boneStableId
                || request.deformation->pose.morphTargetStableId != leaf.deformation->pose.morphTargetStableId))
            || request.light || request.diffractionMaterial || request.glassCaustic
            || !request.importedLights.intensityMultipliers.empty())
            return reject(error, "Scene Compose imported request or explicit lighting replacement is invalid");
        visualimportedscenerender::Request operation;
        operation.sourceStableId = request.sceneSnapshot->id.value;
        operation.renderStableId = leaf.renderStableId;
        operation.sceneSnapshot = request.sceneSnapshot;
        operation.camera = request.camera;
        operation.structuralRevision = request.structuralRevision;
        operation.evaluationRevision = request.evaluationRevision;
        operation.surfacePrograms = request.surfacePrograms;
        operation.importedScene = request.importedScene;
        // Leaves 1+ keep their compiled requests. Each executed deformation
        // differs from its leaf only in playback controls and runtime pose.
        bool animatedLeaf = request.deformation.has_value();
        for (std::size_t index = 0; index < request.additionalImportedScenes.size(); ++index)
        {
            const auto& entry = request.additionalImportedScenes[index];
            const auto leafNumber = std::to_string(index + 1);
            if (!entry.scene)
                return reject(error, "Scene Compose imported request or explicit lighting replacement is invalid");
            const auto& compiled = *entry.scene;
            if (!visualimportedobjectlocal::same(compiled.objectLocalTransform, {}))
                return reject(error, "Imported object local transforms require a native evaluation owner");
            if (entry.deformation.has_value() != compiled.deformation.has_value()
                || (entry.deformation && (!sameImportedDeformationSelection(*entry.deformation, *compiled.deformation)
                    || !visualanimationoperation::valid(*entry.deformation)
                    || !visualanimationimport::sameAsset(entry.deformation->asset, compiled.asset)
                    || entry.deformation->sceneIndex != compiled.sceneIndex)))
            {
                error = "Scene Compose refused: imported leaf " + leafNumber + " deformation does not match its compiled leaf";
                return false;
            }
            // A leaf material arrives lowered in its leaf range, object and
            // MaterialId literal, with that program's digest; execution binds
            // it to the composed scene. Vertex processors stay one-leaf.
            if (compiled.material && compiled.material->vertexModifier)
                return reject(error, "Scene Compose refused: a material vertex processor with multiple imported leaves is not rendered yet");
            if (!compiled.importedLights.intensityMultipliers.empty())
            {
                error = "Scene Compose refused: imported leaf " + leafNumber + " light controls are not rendered yet";
                return false;
            }
            animatedLeaf = animatedLeaf || entry.deformation.has_value();
            operation.additionalImportedScenes.push_back(entry.scene);
        }
        if (!visualimportedscenerender::valid(operation))
            return reject(error, "Scene Compose imported selection, procedural material or revision is stale");
        if (!request.additionalImportedScenes.empty())
        {
            if (request.sourceCameraNodeId || std::any_of(request.additionalImportedScenes.begin(),
                    request.additionalImportedScenes.end(), [](const auto& entry) { return entry.scene->sourceCameraNodeId.has_value(); }))
                return reject(error, "Scene Compose refused: a selected source camera with multiple imported leaves is not rendered yet");
            if (request.parentIntent)
                return reject(error, "Scene Compose refused: imported parenting with multiple imported leaves is not rendered yet");
            if (request.runtimeInputs.noteInstanceMapping)
                return reject(error, "Scene Compose refused: note instancing of multiple imported leaves is not rendered yet");
            if (request.material && request.material->vertexModifier)
                return reject(error, "Scene Compose refused: a material vertex processor with multiple imported leaves is not rendered yet");
            if (!animatedLeaf)
                return reject(error, "Scene Compose refused: multiple imported leaves without animation are not rendered yet");
        }
    }
    else if (!request.additionalImportedScenes.empty())
        return reject(error, "Scene Compose imported request or explicit lighting replacement is invalid");
    if (request.parentIntent && (!importedComposition || !request.deformation
        || request.parentIntent->composeStableId!=request.sceneSnapshot->id.value
        || request.parentIntent->importedSourceStableId!=request.importedScene->sourceStableId
        || request.runtimeInputs.noteInstanceMapping || request.runtimeInputs.glassCaustic
        || (request.material && request.material->vertexModifier)))
        return reject(error,"Imported parenting requires one genuine deformation composition without family, notes or vertex processors");
    if (composedScene)
    {
        if (!HarmonicMIDI::grid::validateVisual3DScene(*request.sceneSnapshot).valid()
            || !request.asset.id.empty() || request.sceneIndex || request.sourceCameraNodeId || request.deformation
            || request.camera || request.light || !request.importedLights.empty()
            || request.structuralRevision == 0
            || request.evaluationRevision < request.structuralRevision)
            return reject(error, "composed Scene3D request is invalid or mixes imported-scene state");
        visualimportedscenerender::Request operation;
        operation.sourceStableId = request.sceneSnapshot->id.value;
        operation.renderStableId = operation.sourceStableId == 1 ? 2 : 1;
        operation.sceneSnapshot = request.sceneSnapshot;
        operation.structuralRevision = request.structuralRevision;
        operation.evaluationRevision = request.evaluationRevision;
        operation.material = request.material;
        operation.surfacePrograms = request.surfacePrograms;
        operation.glassCaustic = request.glassCaustic;
        operation.diffractionMaterial = request.diffractionMaterial;
        if (operation.material && operation.material->sceneSnapshot == nullptr)
            operation.material->sceneSnapshot = request.sceneSnapshot;
        if (!visualimportedscenerender::valid(operation))
            return reject(error, "composed Scene3D material identity or revision is stale");
    }
    else
    {
        if (request.glassCaustic)
            return reject(error, "Glass caustics require the exact retained Scene3D snapshot owner");
        if (!request.surfacePrograms.empty() && !importedComposition)
            return reject(error, "Surface object collections require the retained Scene3D snapshot owner");
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
    // Several leaves select at most one object, in the leaf ID scheme; entry
    // refused a second selecting leaf.
    std::uint64_t expectedObjectNode = request.deformation
        ? request.deformation->pose.nodeStableId : request.objectNodeStableId;
    for (const auto& entry : request.additionalImportedScenes)
        if (expectedObjectNode == 0)
            expectedObjectNode = entry.deformation ? entry.deformation->pose.nodeStableId : entry.scene->objectNodeStableId;
    if ((request.objectNodeStableId == 0) != (request.objectMeshStableId == 0)
        || request.objectNodeStableId > 16384 || request.objectMeshStableId > 4096
        || (request.objectNodeStableId != 0 && (composedScene || !request.sceneIndex
            || (request.deformation && (request.deformation->pose.nodeStableId != request.objectNodeStableId
                || request.deformation->pose.meshStableId != request.objectMeshStableId))))
        || (request.runtimeInputs.objectNodeStableId != 0
            && request.runtimeInputs.objectNodeStableId != expectedObjectNode))
        return reject(error, "imported object modulation requires the exact source node and mesh selection");
    if (request.sourceCameraNodeId && (!request.sceneIndex || *request.sourceCameraNodeId == 0
        || *request.sourceCameraNodeId > 16384))
        return reject(error, "imported source camera selector is malformed");
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

    if (request.light && !request.importedLights.intensityMultipliers.empty())
        return reject(error, "Render 3D Light input replaces imported lighting");
    if (!importedlightcontrol::valid(request.importedLights))
        return reject(error, "Imported light intensity controls are malformed");
    auto sceneInputs = request.runtimeInputs;
    // Checked worlds come only from a staged candidate below, never from the caller.
    sceneInputs.objectWorlds.reset();
    sceneInputs.objectNodeStableId = expectedObjectNode;
    sceneInputs.cameraOverride = request.camera;
    sceneInputs.lightOverride = request.light;
    sceneInputs.glassCaustic = request.glassCaustic;
    // Material Fields consume the retained request frame during evaluation.
    // As in Geometry Core, only an explicit note-instance mapping carries that
    // frame into the native draw. The receipt still retains its original owner.
    if (!sceneInputs.noteInstanceMapping)
        sceneInputs.canonicalBlockCFrame.reset();

    if (request.deformation || !request.additionalImportedScenes.empty())
    {
        if (request.deformation && (!visualanimationoperation::valid(*request.deformation)
            || !visualanimationimport::sameAsset(
                request.deformation->asset, request.asset)
            || request.deformation->sceneIndex != request.sceneIndex))
            return reject(error, "imported scene deformation does not match the exact scene request");
        if (deformationExecution_ == nullptr)
            return reject(error, "native imported scene deformation execution is unavailable");

        // A static leaf keeps its exact source and selection without a clip.
        const auto staticLeaf = [](const visualimportedscenerender::Request& leaf) {
            visualanimationimport::Request operation;
            operation.sourceStableId = leaf.sourceStableId;
            operation.asset = leaf.asset;
            operation.sceneIndex = leaf.sceneIndex;
            return operation;
        };
        ImportedAnimatedSceneRequest deformationRequest;
        deformationRequest.parentIntent = request.parentIntent;
        deformationRequest.operation = request.deformation ? *request.deformation : staticLeaf(*request.importedScene);
        if (!request.additionalImportedScenes.empty())
        {
            deformationRequest.compositionLeaves.reserve(request.additionalImportedScenes.size() + 1);
            ImportedAnimatedSceneLeafRequest first;
            first.operation = deformationRequest.operation;
            first.animated = request.deformation.has_value();
            first.objectNodeStableId = first.animated ? 0u : request.objectNodeStableId;
            deformationRequest.compositionLeaves.push_back(std::move(first));
            for (const auto& entry : request.additionalImportedScenes)
            {
                ImportedAnimatedSceneLeafRequest leaf;
                leaf.animated = entry.deformation.has_value();
                leaf.operation = leaf.animated ? *entry.deformation : staticLeaf(*entry.scene);
                leaf.objectNodeStableId = leaf.animated ? 0u : entry.scene->objectNodeStableId;
                leaf.material = entry.scene->material;
                deformationRequest.compositionLeaves.push_back(std::move(leaf));
            }
        }
        deformationRequest.objectLocalTransform = request.objectLocalTransform;
        deformationRequest.sourceCameraNodeId = request.sourceCameraNodeId;
        deformationRequest.importedLights = request.importedLights;
        deformationRequest.frame = request.frame;
        deformationRequest.structuralRevision = request.structuralRevision;
        deformationRequest.evaluationRevision = request.evaluationRevision;
        deformationRequest.renderClipId = request.clipId;
        deformationRequest.projectGeneration = request.projectGeneration;
        deformationRequest.helperGeneration = request.helperGeneration;
        deformationRequest.width = request.dimensions.width;
        deformationRequest.height = request.dimensions.height;
        deformationRequest.material = request.material;
        deformationRequest.proceduralScene = importedComposition ? request.sceneSnapshot : nullptr;
        deformationRequest.surfacePrograms = request.surfacePrograms;
        deformationRequest.runtimeInputs = sceneInputs;
        if (!request.surfacePrograms.empty())
            deformationRequest.runtimeInputs.canonicalBlockCFrame = request.runtimeInputs.canonicalBlockCFrame;
        ImportedAnimatedSceneReceipt deformationFrame;
        if (parentPayloadIdentity)
        {
            ImportedSceneExecutionReceipt candidate;
            ImportedSceneParentReceiptStaging staging(*this,request,use,candidate,std::move(parentPayloadIdentity));
            if (!deformationExecution_->execute(
                    use==ImportedSceneUse::Preview?ImportedAnimationEvaluationOwner::Preview:ImportedAnimationEvaluationOwner::Export,
                    deformationRequest,deformationFrame,error,&staging)) return false;
            // A pending animated frame is not staged or published. The staging
            // object is discarded with its identity, and the caller keeps its receipt.
            if (deformationFrame.pending)
            {
                output.pending = true;
                error.clear();
                return true;
            }
            staging.publish(output);
            error.clear();
            return true;
        }
        const auto rendered = use == ImportedSceneUse::Preview
            ? deformationExecution_->executePreview(
                deformationRequest, deformationFrame, error)
            : deformationExecution_->executeExport(
                deformationRequest, deformationFrame, error);
        if (!rendered) return false;
        if (deformationFrame.pending)
        {
            // Retry state from the animated route. No receipt is replaced.
            if (use != ImportedSceneUse::Preview)
                return reject(error, "imported scene export delegation returned pending");
            output.pending = true;
            error.clear();
            keepCallerReceipt = true;
            return true;
        }

        ImportedSceneExecutionReceipt receipt;
        receipt.use = use;
        receipt.payload = deformationFrame.payload;
        receipt.usedLastGoodMaterial = deformationFrame.usedLastGoodMaterial;
        receipt.materialDiagnostic = deformationFrame.materialDiagnostic;
        receipt.deformationFrame = std::move(deformationFrame);
        receipt.canonicalBlockCFrame = request.runtimeInputs.canonicalBlockCFrame;
        receipt.materialFrameTexture = request.runtimeInputs.materialFrameTexture;
        // The animated transaction checks the complete expected request before
        // committing history. The outer receipt adds no changed companion.
        keepCallerReceipt = true;
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

    if (!composedScene && !staticLocalBudget_)
    {
        try {staticLocalBudget_=std::make_shared<NativeSceneAnimationCompositionBudget>();}
        catch (const std::bad_alloc&)
        {
            // Optional first-document retention cannot change released v0
            // admission. A later local request refuses a missing document.
        }
    }

    // One immutable static owner per clip: preview and export share the exact
    // prepared scene so an export re-render never re-uploads static resources.
    const StaticOwnerKey ownerKey = request.clipId;
    auto foundOwner = staticOwners_.find(ownerKey);
    // Validate a replacement target before retiring an existing prepared owner.
    // Admission uses the same bounded source decode and publishes only on success.
    std::optional<gltf::GlbNativeSceneAdmission> selectedAdmission;
    if (request.objectNodeStableId != 0 && (foundOwner == staticOwners_.end()
        || foundOwner->second.payload != payload || foundOwner->second.sceneIndex != request.sceneIndex
        || foundOwner->second.objectNodeStableId != request.objectNodeStableId
        || foundOwner->second.objectMeshStableId != request.objectMeshStableId))
    {
        gltf::GlbNativeRenderSeam admissionSeam(backend_);
        gltf::GlbNativeSceneAdmission candidate;
        if (!admissionSeam.admit(payload->bytes(), request.sceneIndex, candidate, error,
                request.objectNodeStableId, request.objectMeshStableId,staticLocalBudget_,true)) return false;
        selectedAdmission = std::move(candidate);
    }
    std::array<arbitgpu::NativeSceneMotionHistory, 2> carriedMotion;
    std::optional<gltf::GlbNativeSceneAdmission> carriedImport;
    std::optional<HarmonicMIDI::grid::SceneCameraRecord> carriedCamera;
    if (foundOwner != staticOwners_.end()
        && ((!composedScene && foundOwner->second.payload != payload)
            || foundOwner->second.sceneIndex != request.sceneIndex
            || foundOwner->second.objectNodeStableId != request.objectNodeStableId
            || foundOwner->second.objectMeshStableId != request.objectMeshStableId
            || foundOwner->second.sourceCameraNodeId != request.sourceCameraNodeId
            || (composedScene
                && foundOwner->second.admission.scene != request.sceneSnapshot)
            || foundOwner->second.proceduralScene != (importedComposition ? request.sceneSnapshot : nullptr)))
    {
        // A timeline Geometry3D scene is a new snapshot every frame. Its motion
        // history moves to the replacement owner. The history key still checks
        // scene id, revision, generations, clip and node before any reuse.
        if (deformedSceneMotion
            && foundOwner->second.sceneIndex == request.sceneIndex
            && foundOwner->second.objectNodeStableId == request.objectNodeStableId
            && foundOwner->second.objectMeshStableId == request.objectMeshStableId
            && foundOwner->second.sourceCameraNodeId == request.sourceCameraNodeId
            && foundOwner->second.proceduralScene == nullptr)
            carriedMotion = std::move(foundOwner->second.motion);
        // Scene Compose with the same exact import and selection only needs a
        // new composition. A modulated PBR value makes a new procedural
        // snapshot on each changed quantum, and must not decode the GLB again
        // or draw on the lifetime ledger. The replacement owner gets a fresh
        // seam, so material publication starts clean for the composed scene.
        if (importedComposition && foundOwner->second.importedScene
            && foundOwner->second.payload == payload
            && foundOwner->second.sceneIndex == request.sceneIndex
            && foundOwner->second.objectNodeStableId == request.objectNodeStableId
            && foundOwner->second.objectMeshStableId == request.objectMeshStableId
            && foundOwner->second.sourceCameraNodeId == request.sourceCameraNodeId)
        {
            carriedImport = std::move(foundOwner->second.admission);
            carriedImport->scene = std::move(foundOwner->second.importedScene);
            carriedCamera = std::move(foundOwner->second.selectedSourceCamera);
        }
        staticOwners_.erase(foundOwner);
        foundOwner = staticOwners_.end();
    }
    if (foundOwner == staticOwners_.end())
    {
        // Cold admit: eviction, GLB admission and the camera decode below.
        loopstep::Scope coldAdmitScope ("advance:imported-scene:cold-admit");
        evictForInsertion();
        auto [inserted, didInsert] = staticOwners_.try_emplace(ownerKey, backend_);
        if (!didInsert)
            return reject(error, "imported scene static owner insertion failed");
        auto& owner = inserted->second;
        owner.motion = std::move(carriedMotion);
        owner.exactPayloadIdentity = request.staticPayloadIdentity;
        owner.glassCaustic = request.glassCaustic;
        owner.payload = payload;
        owner.sceneIndex = request.sceneIndex;
        owner.sourceCameraNodeId = request.sourceCameraNodeId;
        owner.objectNodeStableId = request.objectNodeStableId;
        owner.objectMeshStableId = request.objectMeshStableId;
        if (composedScene)
            owner.admission.scene = request.sceneSnapshot;
        else if (carriedImport)
            owner.admission = std::move(*carriedImport);
        else if (selectedAdmission)
            owner.admission = std::move(*selectedAdmission);
        else if (!owner.seam.admit(payload->bytes(), request.sceneIndex,
                                   owner.admission, error, request.objectNodeStableId,
                                   request.objectMeshStableId,staticLocalBudget_,true))
        {
            staticOwners_.erase(inserted);
            return false;
        }
        if (request.sourceCameraNodeId && carriedCamera)
            owner.selectedSourceCamera = std::move(carriedCamera);
        else if (request.sourceCameraNodeId)
        {
            // Reuse bounded source decode. The admitted scene alone cannot
            // distinguish a real node camera from the framed fallback.
            auto options = nativeImportedAnimationDecodeOptions(payload->bytes().size(), request.sceneIndex);
            loopstep::Scope cameraDecodeScope ("advance:imported-scene:glb-decode");
            auto decoded = owner.admission.document ? std::optional<gltf::GlbStaticMeshDocument>{}
                : gltf::decodeAnimatedGlbBaseScene(payload->bytes(), options.admission, error);
            const auto* document = owner.admission.document ? owner.admission.document.get() : decoded ? &*decoded : nullptr;
            const auto* camera = document ? sourceCameraRecord(*document, *owner.admission.scene, *request.sourceCameraNodeId) : nullptr;
            if (!camera)
            {
                staticOwners_.erase(inserted);
                return reject(error, "selected imported camera is not a genuine camera in the admitted source scene");
            }
            owner.selectedSourceCamera = *camera;
        }
        if (importedComposition)
        {
            auto composed = gltf::composeImportedVisual3DScene(*request.sceneSnapshot, *owner.admission.scene, error);
            if (!composed) { staticOwners_.erase(inserted); return false; }
            owner.proceduralScene = request.sceneSnapshot;
            owner.importedScene = std::move(owner.admission.scene);
            owner.admission.scene = std::move(composed);
        }
        foundOwner = inserted;
        insertedStaticOwner = !request.parentIntent && request.objectLocalTransform.version != 0;
    }
    auto& owner = foundOwner->second;
    owner.lastUse = ++useSerial_;
    auto& seam = owner.seam;
    auto& admission = owner.admission;
    if (request.sourceCameraNodeId)
    {
        if (!owner.selectedSourceCamera || owner.selectedSourceCamera->id.value != *request.sourceCameraNodeId)
            return reject(error, "static imported camera owner does not match the request");
        sceneInputs.animatedCamera = owner.selectedSourceCamera;
        if (importedComposition && sceneInputs.cameraOverride)
        {
            const auto* retained=HarmonicMIDI::grid::visual3d_detail::findById(admission.scene->cameras,
                admission.scene->cameraCount,admission.scene->activeCamera);
            if (!retained || !HarmonicMIDI::grid::scenecomposition::sameCameraRecord(*retained,*request.camera))
                return reject(error,"static composition camera override does not match its exact retained record");
            sceneInputs.cameraOverride=*retained;
        }
    }

    if (!importedComposition && !importedscene::evaluateImportedLights(request.importedLights,
            admission.importedLightDefinitions, *admission.scene, sceneInputs.animatedLights, error)) return false;

    bool usedLastGoodMaterial = false;
    std::string materialDiagnostic;
    if (!request.surfacePrograms.empty())
    {
        auto programs = request.surfacePrograms;
        if (importedComposition)
            for (auto& object : programs)
            {
                const auto mapped = HarmonicMIDI::grid::scenecomposition::composedProceduralObjectId(object.program.material.binding.object);
                if (!mapped) return reject(error, "Scene Compose procedural Surface target has no object identity");
                object.program.material.binding.object = *mapped;
                object.program.material.sceneSnapshot = admission.scene;
            }
        if (importedComposition && request.material)
        {
            std::string overrideError;
            if (!geometrysurfacematerial::appendImportedOverride(programs, *request.material,
                    admission.scene, request.structuralRevision, request.evaluationRevision, overrideError))
                return reject(error, overrideError.c_str());
        }
        if (request.frame.rateNumerator == 0 || request.frame.rateDenominator == 0)
            return reject(error, "Surface collection requires the exact rational preview or export frame");
        videowire::geometry::RuntimeFieldEvaluation evaluation;
        evaluation.timelineSeconds = static_cast<double>(request.frame.frame) * request.frame.rateDenominator / request.frame.rateNumerator;
        evaluation.scoreAt = [&](const auto& operation, const auto& positions, auto& result, auto& diagnostic) {
            return scorefield::evaluateSampleField(operation, positions, request.runtimeInputs.canonicalBlockCFrame, result, diagnostic);
        };
        if (!seam.publishSurfacePrograms(admission, programs, evaluation, error,
                request.diffractionMaterial ? &*request.diffractionMaterial : nullptr,
                request.glassCaustic ? &*request.glassCaustic : nullptr)) return false;
    }
    else if (request.material)
    {
        auto material = *request.material;
        if (importedComposition)
        {
            material.scene = admission.scene->id;
            material.sceneRevision = request.structuralRevision;
        }
        if (material.sceneSnapshot != nullptr
            && material.sceneSnapshot != admission.scene)
            return reject(error, "imported scene material transport must not carry a foreign scene owner");
        material.sceneSnapshot = admission.scene;
        if (!seam.publishSurfaceMaterial(admission, material, error))
        {
            if (request.glassCaustic || hasMaterialFrame || use != ImportedSceneUse::Preview
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
            if (request.glassCaustic || hasMaterialFrame || use != ImportedSceneUse::Preview
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

    // A non-neutral object-local transform stages one candidate over this
    // owner's exact admission, so nothing decodes twice. Its checked worlds
    // drive the native draw. The candidate itself ends with this attempt.
    if (localTransform)
    {
        std::shared_ptr<const NativeSceneAnimationCompositionFrame> localCandidate;
        try
        {
            if (!stageStaticLocalFrame(use, request, &owner, localCandidate, error)) return false;
        }
        catch (const std::bad_alloc&)
        { return reject(error, "Static local object candidate allocation failed"); }
#if defined(ARBIT_TEST_IMPORTED_PARENT_STAGING_FAILURES)
        auto& probe = ImportedSceneParentStagingFailureProbe::state();
        probe.candidateLocalFrame = localCandidate;
        if (probe.localObserver) probe.localObserver(localCandidate);
#endif
        if (!localCandidate->objectWorlds() || localCandidate->owner()->scene() != admission.scene)
            return reject(error, "Static local candidate does not match the admitted scene");
        sceneInputs.objectWorlds = localCandidate->objectWorlds();
    }
    // Samples keep the vertex program time and spectrum, so a graph vertex
    // program shares this history. The backend runs it again at those inputs.
    auto& motion = owner.motion[use == ImportedSceneUse::Preview ? 0 : 1];
    if (arbitgpu::sceneUsesMotionPass(sceneInputs))
    {
        if (request.frame.rateNumerator == 0 || request.frame.rateDenominator == 0)
        {
            motion = {};
            return reject(error, "Motion pass requires an exact rational presentation frame");
        }
        if (sceneInputs.noteInstanceMapping && (composedScene || !backend_.supportsRetainedNoteMotion()))
        {
            motion = {};
            return reject(error, "Motion pass requires an admitted static imported prototype and native stable-note consumer");
        }
    }
    const arbitgpu::NativeSceneMotionOwner motionOwner {{ admission.scene->id.value, 0, 0,
        request.structuralRevision, request.projectGeneration, request.helperGeneration,
        static_cast<std::uint64_t>(request.clipId), sceneInputs.objectNodeStableId }};
    sceneInputs.previousMotion = motion.predecessor(request.frame.frame,
        request.frame.rateNumerator, request.frame.rateDenominator,
        request.dimensions.width, request.dimensions.height, motionOwner, sceneInputs);
    std::shared_ptr<const std::vector<HarmonicMIDI::grid::SceneVec3>> scenePositions;
    if (deformedSceneMotion && arbitgpu::sceneUsesMotionPass(sceneInputs))
    {
        // A deformed predecessor without positions cannot be projected, so it
        // gives zero motion like a first frame. Frames without Motion keep no
        // positions, so the first Motion frame after them starts at zero.
        if (sceneInputs.previousMotion && !sceneInputs.previousMotion->scenePositions)
            sceneInputs.previousMotion.reset();
        // Only the scene part keeps positions. Appended solids past it carry
        // their own prior transforms, so their varying count never shifts it.
        const auto retained = request.retainedMotionVertexCount != 0
            ? request.retainedMotionVertexCount : admission.scene->vertexCount;
        if (!arbitgpu::sceneMotionPositionsMatch(*admission.scene, retained)
            || (sceneInputs.previousMotion && sceneInputs.previousMotion->scenePositions->size() != retained))
        {
            motion = {};
            return reject(error, "Motion output prior Geometry3D positions do not match the scene vertices");
        }
        try
        {
            auto positions = std::make_shared<std::vector<HarmonicMIDI::grid::SceneVec3>>(retained);
            for (std::size_t index = 0; index < positions->size(); ++index)
                (*positions)[index] = admission.scene->vertices[index].position;
            scenePositions = std::move(positions);
        }
        catch (const std::bad_alloc&)
        {
            motion = {};
            return reject(error, "Timeline Geometry3D motion position retention failed");
        }
    }
    gltf::GlbNativeFrameReceipt frame;
    bool rendered = false;
    {
        loopstep::Scope renderScope (use == ImportedSceneUse::Preview
            ? "advance:imported-scene:preview-render" : "advance:imported-scene:export-render");
        rendered = use == ImportedSceneUse::Preview
            ? seam.renderPreview(admission, request.dimensions, sceneInputs, frame, error)
            : seam.renderExport(admission, request.dimensions, sceneInputs, frame, error);
    }
    if (!rendered)
        return false;
    if (frame.rendered.pending)
    {
        // Retry state. Nothing below this point runs: no motion history is reset or
        // committed, no frame is published, and the caller keeps its receipt.
        // Export never sees pending, so it refuses instead.
        if (use != ImportedSceneUse::Preview)
            return reject(error, "native imported scene export render returned pending");
        output.pending = true;
        error.clear();
        keepCallerReceipt = true;
        keepStaticOwner = true;
        return true;
    }
    if (!motion.commit(request.frame.frame, request.frame.rateNumerator,
        request.frame.rateDenominator, request.dimensions.width, request.dimensions.height,
        sceneInputs, {}, motionOwner, std::move(scenePositions)))
    {
        // The next frame cannot project deformed positions it never received,
        // so the history resets. A Motion frame that needed them fails now
        // instead of rendering with silently zero motion.
        motion = {};
        if (deformedSceneMotion && arbitgpu::sceneUsesMotionPass(sceneInputs))
            return reject(error, "Motion output could not retain the timeline Geometry3D positions for the next frame");
    }

    ImportedSceneExecutionReceipt receipt;
    receipt.use = use;
    receipt.payload = payload;
    receipt.admission = admission;
    receipt.frame = std::move(frame);
    receipt.usedLastGoodMaterial = usedLastGoodMaterial;
    receipt.materialDiagnostic = std::move(materialDiagnostic);
    receipt.canonicalBlockCFrame = request.runtimeInputs.canonicalBlockCFrame;
    receipt.materialFrameTexture = request.runtimeInputs.materialFrameTexture;
    keepCallerReceipt = true;
    keepStaticOwner = true;
    output = std::move(receipt);
    error.clear();
    return true;
}
} // namespace videohelper::modelpayload
