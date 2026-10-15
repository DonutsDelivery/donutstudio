#pragma once

#include "../../shared/ImportedLightControl.h"
#include "../../shared/VisualImportedObjectLocalTransform.h"
#include "imported_animation_deformation_consumer.h"
#include "fixture_scene_renderer.h"
#include "glb_scene_adapter.h"
#include "model_payload_transport.h"
#include "native_animation_deformation_renderer.h"
#include "native_scene_animation_composition.h"
#include "imported_scene_parent_binding.h"

#include <cstddef>
#include <optional>
#include <vector>

namespace videohelper::modelpayload
{
class NativeCompositionRetentionBudget;
class ImportedSceneParentReceiptStaging;
// Defined only by tests that inspect composition leaf admission.
struct ImportedAnimatedSceneCompositionProbe;
#if defined(ARBIT_TEST_IMPORTED_PARENT_STAGING_FAILURES)
struct ImportedSceneParentStagingFailureProbe;
#endif
struct ImportedAnimatedSceneAdmissionIdentity final
{
    visualmodelassetpayload::PayloadPtr payload;
    std::optional<std::size_t> sceneIndex;
    visualanimationimport::StableId meshStableId = 0;
    visualanimationimport::StableId animationClipStableId = 0;
    visualanimationimport::StableId crossfadeClipStableId = 0;
    std::string clipName {};
    std::string crossfadeClipName {};
    std::optional<std::uint32_t> sourceCameraNodeId {};
    std::optional<visualimportedscenerender::ImportedParentIntentV1> parentIntent {};

    friend bool operator==(const ImportedAnimatedSceneAdmissionIdentity& left,
                           const ImportedAnimatedSceneAdmissionIdentity& right) noexcept
    {
        return left.payload == right.payload
            && left.sceneIndex == right.sceneIndex
            && left.sourceCameraNodeId == right.sourceCameraNodeId
            && left.meshStableId == right.meshStableId
            && left.animationClipStableId == right.animationClipStableId
            && left.crossfadeClipStableId == right.crossfadeClipStableId
            && left.clipName == right.clipName && left.crossfadeClipName == right.crossfadeClipName
            && left.parentIntent.has_value() == right.parentIntent.has_value()
            && (!left.parentIntent || visualimportedscenerender::sameImportedParentIntent(*left.parentIntent,*right.parentIntent));
    }

    friend bool operator!=(const ImportedAnimatedSceneAdmissionIdentity& left,
                           const ImportedAnimatedSceneAdmissionIdentity& right) noexcept
    {
        return !(left == right);
    }
};

struct ImportedAnimationClipCompatibility final
{
    visualanimationimport::StableId stableId = 0;
    std::string name;
    std::vector<visualanimationimport::StableId> compatibleMeshStableIds;
};

struct ImportedAnimatedSceneCompatibility final
{
    std::vector<ImportedAnimationClipCompatibility> clips;
};

// One imported leaf of a multi-leaf Scene Compose, as authored. Its pose
// selector keeps the leaf ID scheme; execution decodes it to the leaf's raw
// node before the leaf's consumer is evaluated.
struct ImportedAnimatedSceneLeafRequest final
{
    visualanimationimport::Request operation;
    // A static leaf's object selection in the leaf ID scheme, or zero. An
    // animated leaf selects through operation.pose.nodeStableId.
    std::uint32_t objectNodeStableId = 0;
    bool animated = true;
    // Leaf 1+'s Render Material, lowered in the leaf ID scheme: its object
    // and MaterialId literal, with the digest of that program. Leaf 0's
    // material is the request's material, so this stays empty for leaf 0.
    std::optional<surfacematerialbinding::ImportedSceneMaterialRequest> material;
};

struct ImportedAnimatedSceneRequest final
{
    // Borrowed for this synchronous call. Retained authorities own a sealed copy.
    const visualimportedscenerender::ImportedParentIntentV1* parentIntent = nullptr;
    visualanimationimport::Request operation;
    visualimportedobjectlocal::ObjectLocalTransformV1 objectLocalTransform;
    std::optional<std::uint32_t> sourceCameraNodeId {};
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> proceduralScene;
    geometrysurfacematerial::ObjectPrograms surfacePrograms;
    importedlightcontrol::Controls importedLights;
    visualdeformation::RationalFrameTime frame;
    std::uint64_t structuralRevision = 0;
    int renderClipId = 0;
    std::uint64_t projectGeneration = 0, helperGeneration = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::optional<surfacematerialbinding::ImportedSceneMaterialRequest> material;
    arbitgpu::NativeImportedSceneRuntimeInputs runtimeInputs;
    // Zero remains compatible only with the older non-composition internal route.
    std::uint64_t evaluationRevision = 0;
    // Empty for one leaf. Otherwise every leaf of the composition in leaf
    // order; operation and sourceCameraNodeId above are then unused.
    std::vector<ImportedAnimatedSceneLeafRequest> compositionLeaves;
};

// Request bytes without the leaf list. One-leaf charges keep the size they had
// before multi-leaf compositions.
inline constexpr std::size_t kImportedAnimatedSceneRequestChargeBytes =
    sizeof(ImportedAnimatedSceneRequest) - sizeof(std::vector<ImportedAnimatedSceneLeafRequest>);

struct ImportedAnimatedSceneReceipt final
{
    ImportedAnimationEvaluationOwner owner = ImportedAnimationEvaluationOwner::Preview;
    visualmodelassetpayload::PayloadPtr payload;
    std::shared_ptr<const arbitgpu::NativeDeformationScene> source;
    ImportedAnimationDeformationEvaluation evaluation;
    videorender::animation3d::RenderedDeformationFrame frame;
    arbitgpu::NativeImportedSceneRuntimeInputs runtimeInputs;
    std::optional<std::uint32_t> sourceCameraNodeId {};
    bool usedLastGoodMaterial = false;
    std::string materialDiagnostic;
    std::shared_ptr<const NativeSceneAnimationCompositionFrame> sceneAnimationComposition;
    // Preview only. The frame is still compiling, so the receipt carries no frame and
    // the caller retries. A pending receipt is not valid(), and no history is committed.
    bool pending = false;

    bool valid() const noexcept
    {
        const bool requiresSceneAnimationAuthority = source && !source->deformation
            && !source->compositionOwner && source->evaluationRevision != 0;
        return (!parentBinding_ || parentBinding_->matches(payload,source,evaluation,frame,runtimeInputs,owner,sourceCameraNodeId))
            && payload && source && (evaluation.deformation || evaluation.sceneAnimation) && frame.nativeFrame
            && (!sourceCameraNodeId || (runtimeInputs.animatedCamera
                && runtimeInputs.animatedCamera->id.value == *sourceCameraNodeId))
            && (!evaluation.deformation || !sourceCameraNodeId || selectedDeformationCamera_)
            && (!selectedDeformationCamera_ || selectedDeformationCamera_->matches(
                payload,source,evaluation,frame.nativeFrame,owner,frame.stats.time,runtimeInputs,
                sourceCameraNodeId,frame.source,frame.deformation,
                static_cast<std::uint8_t>(frame.use)))
            && (!requiresSceneAnimationAuthority || sceneAnimationComposition)
            && (!sceneAnimationComposition || sceneAnimationComposition->matches(
                payload, source, evaluation, frame.nativeFrame, owner, frame.stats.time, runtimeInputs, sourceCameraNodeId))
            && (!sceneAnimationComposition || frame.source == source)
            && frame.nativeFrame->depthImageHandle() != 0
            && frame.nativeFrame->depthTextureViewHandle() != 0;
    }
private:
    friend class ImportedAnimatedScenePayloadExecution;
    std::shared_ptr<const NativeSelectedDeformationCameraFrame> selectedDeformationCamera_;
    std::shared_ptr<const ImportedParentBindingFrame> parentBinding_;
public:
    bool inspectParent(const visualimportedscenerender::ImportedParentIntentV1& expected,
        const visualdeformation::RationalFrameTime& time,std::uint64_t structural,std::uint64_t evaluation,
        std::uint64_t project,std::uint64_t helper,int clip,ImportedAnimationEvaluationOwner use,
        std::uint32_t width,std::uint32_t height,
        const visualanimationimport::Request* operation = nullptr,
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& procedural = {},
        std::optional<std::uint32_t> selector = {},
        const std::optional<HarmonicMIDI::grid::SceneCameraRecord>& override = {},
        const arbitgpu::NativeImportedSceneRuntimeInputs* runtime = nullptr) const noexcept
    {
        return parentBinding_ && parentBinding_->inspectExpected(expected,time,structural,evaluation,project,helper,
            clip,use,width,height,operation,procedural,selector,override,runtime) && valid();
    }
};

// Exact product seam: resident processor bytes -> immutable animation/source
// snapshot -> one shared native preview/export renderer. No path, URI, catalog,
// grant, or CPU fallback exists on this surface.
class ImportedAnimatedScenePayloadExecution final
{
public:
    using Receipt = ImportedAnimatedSceneReceipt;

    ImportedAnimatedScenePayloadExecution(Store& store,
        arbitgpu::NativeDeformationBackend& backend,
        arbitgpu::NativeFixtureSceneBackend& sceneBackend = arbitgpu::nativeFixtureSceneBackend()) noexcept;

    static bool validReceipt(const Receipt& receipt) noexcept { return receipt.valid(); }
    static const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& nativeFrame(
        const Receipt& receipt) noexcept
    {
        return receipt.frame.nativeFrame;
    }
    static bool inspectCompatibility(
        const visualmodelassetpayload::PayloadPtr& payload,
        std::optional<std::size_t> sceneIndex,
        ImportedAnimatedSceneCompatibility& output,
        std::string& error);

    bool executePreview(const ImportedAnimatedSceneRequest& request,
                        ImportedAnimatedSceneReceipt& output,
                        std::string& error);
    bool executeExport(const ImportedAnimatedSceneRequest& request,
                       ImportedAnimatedSceneReceipt& output,
                       std::string& error);
    bool chargeParentInspection() const noexcept;
    bool inspectParentReceipt(const ImportedAnimatedSceneReceipt& receipt,
        const ImportedAnimatedSceneRequest& expected,ImportedAnimationEvaluationOwner use) const noexcept;
    void reset() noexcept;

private:
    friend class ImportedScenePayloadExecution;
    friend class ImportedSceneParentReceiptStaging;
    friend struct ImportedAnimatedSceneCompositionProbe;
#if defined(ARBIT_TEST_IMPORTED_PARENT_STAGING_FAILURES)
    friend struct ImportedSceneParentStagingFailureProbe;
#endif
    bool execute(ImportedAnimationEvaluationOwner owner,
                 const ImportedAnimatedSceneRequest& request,
                 ImportedAnimatedSceneReceipt& output,
                 std::string& error,
                 ImportedSceneParentReceiptStaging* parentStaging = nullptr);
    bool admit(const visualmodelassetpayload::PayloadPtr& payload,
               const visualanimationimport::Request& operation,
               const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& proceduralScene,
               std::optional<std::uint32_t> sourceCameraNodeId,
               const ImportedAnimatedSceneRequest& request,
               std::string& error);
    // One imported leaf of a multi-leaf composition: its own exact payload,
    // request (clip, playback speed and trim), consumer, decoded base
    // document, adapted rest scene and render bindings. A static leaf has no
    // consumer clip and no bindings. Each leaf holds its own reservations in
    // the scene animation ledger; destroying the leaf refunds them.
    struct CompositionLeaf final
    {
        // Members are destroyed in reverse order, so these reservations,
        // declared first, are released last: the leaf drops its values first.
        // Owner aliases of a value keep its reservation alive with them.
        // [0] fixed storage, [1] decoded closure, [2] the render binding copy
        // (empty when the leaf keeps no binding).
        std::array<std::shared_ptr<const NativeSceneAnimationCompositionBudget::Charge>, 3> charges;
        visualmodelassetpayload::PayloadPtr payload;
        visualanimationimport::Request operation;
        ImportedAnimationDeformationConsumer consumer;
        std::shared_ptr<const gltf::GlbStaticMeshDocument> document;
        std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> scene;
        std::vector<gltf::GlbDeformationRenderBinding> bindings;
        bool animated = true;
    };
    struct CompositionLeafInput final
    {
        visualmodelassetpayload::PayloadPtr payload;
        visualanimationimport::Request operation;
        bool animated = true;
    };
    // Reserves the leaf's fixed storage before decode, then its decoded
    // closure, and decodes and binds it the way admit() does for one leaf.
    static bool admitCompositionLeaf(const std::shared_ptr<NativeSceneAnimationCompositionBudget>& budget,
        const CompositionLeafInput& input, CompositionLeaf& output, std::string& error);
    // Admits two to eight leaves together, or keeps every previous state.
    // Success replaces any admitted single leaf; admit() success clears these.
    bool admitCompositionLeaves(const std::vector<CompositionLeafInput>& inputs, std::string& error);
    // Drops the admitted single-leaf state, as reset() does, without touching
    // the composition leaves or the shared ledgers.
    void clearSingleLeafState() noexcept;
    // Evaluates one leaf at its own clip time for the shared presentation
    // frame, with this frame's playback controls and the leaf's raw selectors.
    static bool evaluateCompositionLeaf(ImportedAnimationEvaluationOwner owner, const CompositionLeaf& leaf,
        const visualanimationimport::Request& operation,
        const visualdeformation::RationalFrameTime& frame, std::uint64_t structuralRevision,
        ImportedAnimationDeformationEvaluation& evaluation, std::string& error);
    // Renders a multi-leaf Scene Compose: every leaf joins one composed scene
    // and one native owner, and each animated leaf keeps its own clip.
    bool executeComposition(ImportedAnimationEvaluationOwner owner,
        const ImportedAnimatedSceneRequest& request, ImportedAnimatedSceneReceipt& output, std::string& error);
    bool publishSurfaceMaterial(
        const surfacematerialbinding::ImportedSceneMaterialRequest& request,
        std::string& error);
    void clearSurfaceMaterial() noexcept;
    bool executeSceneAnimationComposition(ImportedAnimationEvaluationOwner owner,
        const ImportedAnimatedSceneRequest& request, ImportedAnimationDeformationEvaluation evaluation,
        bool usedLastGoodMaterial, std::string materialDiagnostic,
        ImportedAnimatedSceneReceipt& output, std::string& error);

    // Shared across resets so escaped receipts remain charged.
    std::shared_ptr<NativeCompositionRetentionBudget> compositionBudget_;
    std::weak_ptr<const visualmodelassetpayload::ImmutablePayload> compositionEpochPayload_;
    std::array<std::uint64_t,5> compositionEpochIdentity_ {};
    std::uint64_t highestCompositionEvaluationRevision_ = 0;
    // This ledger survives source replacement and reset. Immutable aliases and
    // issued frame authorities retain debits after executor histories retire.
    std::shared_ptr<NativeSceneAnimationCompositionBudget> sceneAnimationBudget_;
    std::shared_ptr<const NativeSceneAnimationCompositionBudget::Charge> sceneAnimationAdmissionCharge_;
    std::shared_ptr<const NativeSceneAnimationCompositionOwner> sceneAnimationOwner_;
    std::array<std::shared_ptr<const NativeSceneAnimationCompositionFrame>, 2> sceneAnimationFrames_;
    Store& store_;
    ImportedAnimationDeformationConsumer consumer_;
    videorender::animation3d::NativeAnimationDeformationRenderer renderer_;
    videorender::fixture3d::FixtureSceneRenderer sceneRenderer_;
    std::shared_ptr<const ImportedParentBindingOwner> parentOwner_;
    ImportedAnimatedSceneAdmissionIdentity admittedIdentity_;
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> scene_;
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> importedScene_;
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> proceduralScene_;
    std::shared_ptr<const gltf::GlbStaticMeshDocument> baseScene_;
    // Reserves the kept binding copy below; declared first so the copy is
    // destroyed before its reservation is refunded.
    std::shared_ptr<const NativeSceneAnimationCompositionBudget::Charge> bindingCopyCharge_;
    std::vector<gltf::GlbDeformationRenderBinding> bindings_;
    std::shared_ptr<const arbitgpu::NativeDeformationScene> source_;
    // Empty for a single leaf; otherwise every admitted leaf in leaf order.
    std::vector<CompositionLeaf> compositionLeaves_;
    // Resource admission stays shared; timeline histories belong to each use.
    std::array<arbitgpu::NativeSceneMotionHistory, 2> motion_;
    std::optional<arbitgpu::NativeSceneMotionOwner> motionOwner_;
    std::string motionDevice_;
    std::weak_ptr<const HarmonicMIDI::grid::Visual3DScene> materialScene_;
    std::shared_ptr<const videorender::fixture3d::AdmittedSurfaceMaterialBinding>
        lastGoodMaterial_;
    std::uint64_t highestMaterialRevisionSeen_ = 0;
    std::uint64_t lastGoodMaterialRevision_ = 0;
    std::uint64_t rejectedMaterialRevision_ = 0;
};
} // namespace videohelper::modelpayload
