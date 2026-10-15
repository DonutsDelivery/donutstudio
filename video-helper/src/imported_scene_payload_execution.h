#pragma once

#include "imported_animated_scene_payload_execution.h"
#include "glb_native_render_seam.h"
#include "model_payload_transport.h"
#include "../../shared/DiffractionMaterialBindingContract.h"
#include "../../shared/VisualImportedSceneRenderOperationContract.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <map>
#include <optional>
#include <string>
#include <vector>
#if defined(ARBIT_TEST_IMPORTED_PARENT_STAGING_FAILURES)
#include <new>
#endif

namespace canonicalblockc { class CanonicalBlockCFrame; }

namespace videohelper::modelpayload
{

enum class ImportedSceneUse : std::uint8_t
{
    Preview = 0,
    Export = 1
};

// One further imported leaf of a multi-leaf Scene Compose: its compiled leaf
// request and its deformation after the leaf's own runtime controls and
// playback clock. Selectors keep the leaf ID scheme.
struct ImportedSceneLeafRequest final
{
    std::shared_ptr<const visualimportedscenerender::Request> scene;
    std::optional<visualanimationimport::Request> deformation;
};

struct ImportedSceneRequest final
{
    // The compiled request owns this fixed value for the synchronous execution.
    const visualimportedscenerender::ImportedParentIntentV1* parentIntent = nullptr;
    std::shared_ptr<const visualimportedscenerender::Request> importedScene;
    visualanimationimport::ExactContentAssetKey asset;
    std::uint64_t sourceStableId = 0, renderStableId = 0;
    std::optional<std::size_t> sceneIndex;
    std::optional<std::uint32_t> sourceCameraNodeId {};
    std::uint32_t objectNodeStableId = 0;
    std::uint32_t objectMeshStableId = 0;
    visualimportedobjectlocal::ObjectLocalTransformV1 objectLocalTransform;
    importedlightcontrol::Controls importedLights;
    videorender::fixture3d::RenderDimensions dimensions;
    std::optional<visualanimationimport::Request> deformation;
    visualdeformation::RationalFrameTime frame;
    std::uint64_t structuralRevision = 0;
    std::uint64_t evaluationRevision = 0;
    std::uint64_t projectGeneration = 0;
    std::uint64_t helperGeneration = 0;
    int clipId = 0;
    std::string staticPayloadIdentity;
    std::optional<surfacematerialbinding::ImportedSceneMaterialRequest> material;
    geometrysurfacematerial::ObjectPrograms surfacePrograms;
    std::optional<diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest>
        diffractionMaterial;
    std::optional<visualimportedscenerender::GlassCausticSelection> glassCaustic;
    std::optional<HarmonicMIDI::grid::SceneCameraRecord> camera;
    std::optional<HarmonicMIDI::grid::SceneLightRecord> light;
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> sceneSnapshot;
    // Set for a timeline Geometry3D scene evaluated each frame with a fixed
    // vertex order. Motion history then survives the per-frame snapshot and
    // retains vertex positions so the next frame can project them.
    bool deformedSceneMotion = false;
    // Vertices whose positions that history keeps. Zero keeps every vertex.
    // Appended solid bodies past it take Motion from their own prior transforms.
    std::size_t retainedMotionVertexCount = 0;
    arbitgpu::NativeImportedSceneRuntimeInputs runtimeInputs;
    // Leaves 1..N-1 of a multi-leaf Scene Compose in leaf order; importedScene
    // stays leaf 0. Empty for one leaf.
    std::vector<ImportedSceneLeafRequest> additionalImportedScenes;
};

// Request bytes without the leaf list. One-leaf charges keep the size they had
// before multi-leaf compositions.
inline constexpr std::size_t kImportedSceneRequestChargeBytes =
    sizeof(ImportedSceneRequest) - sizeof(std::vector<ImportedSceneLeafRequest>);

// Opaque source-presentation bookkeeping. Only the existing payload executor
// issues or retires it. It grants no native capability or historical validity.
class ImportedSceneParentPresentation final
{
    friend class ImportedScenePayloadExecution;
    friend class ImportedSceneParentReceiptStaging;
    friend struct ImportedSceneExecutionReceipt;
    ImportedSceneParentPresentation()=default;
    ImportedSceneParentPresentation(const ImportedSceneParentPresentation&)=delete;
    ImportedSceneParentPresentation& operator=(const ImportedSceneParentPresentation&)=delete;
    std::shared_ptr<const NativeCompositionRetentionBudget::Charge> retention_;
    std::shared_ptr<NativeSceneAnimationCompositionBudget> budget_;
    mutable bool retired_=false;
public:
    ~ImportedSceneParentPresentation()=default;
};

// Owns every immutable object involved in one successful product execution:
// exact processor-supplied bytes, the admitted scene snapshot, and the native
// frame. No path or catalog record can substitute for `payload`.
struct ImportedSceneExecutionReceipt final
{
private:
    friend class ImportedScenePayloadExecution;
    friend class ImportedSceneParentReceiptStaging;
#if defined(ARBIT_TEST_IMPORTED_PARENT_STAGING_FAILURES)
    friend struct ImportedSceneParentStagingFailureProbe;
#endif
    void exchangeOwnedValues(ImportedSceneExecutionReceipt& other) noexcept;
    // Outer source retirement identity, not a native issuance authority. Member
    // order destroys the copied identity before refunding its retained charge.
    struct ParentPayloadIdentity final
    {
        std::shared_ptr<const NativeCompositionRetentionBudget::Charge> retention;
        std::shared_ptr<const NativeCompositionRetentionBudget::Charge> diagnosticRetention;
        std::shared_ptr<NativeSceneAnimationCompositionBudget> budget;
        std::string exactPayloadIdentity;
        std::uint64_t structural=0,project=0,helper=0,source=0,deformation=0;
        // Same-source receipts and Layers share one original publication token.
        std::shared_ptr<const ImportedSceneParentPresentation> presentation;
        // Weak bookkeeping binds successful typed source-owner replacement
        // without extending or guessing native/source lifetime authority.
        std::weak_ptr<const arbitgpu::NativeDeformationCompositionOwner> nativeOwner;
        int clip=0;
    };
    std::shared_ptr<const ParentPayloadIdentity> parentPayloadIdentity_;
public:
    class ParentPresentationGuard final
    {
        friend class ImportedScenePayloadExecution;
        std::shared_ptr<const ImportedSceneParentPresentation> identity_;
        bool foreign_=false;
    public:
        bool present() const noexcept { return static_cast<bool>(identity_); }
        bool retired() const noexcept { return foreign_ || (identity_ && identity_->retired_); }
    };
    ImportedSceneUse use = ImportedSceneUse::Preview;
    visualmodelassetpayload::PayloadPtr payload;
    gltf::GlbNativeSceneAdmission admission;
    gltf::GlbNativeFrameReceipt frame;
    std::optional<ImportedAnimatedSceneReceipt> deformationFrame;
    bool usedLastGoodMaterial = false;
    std::string materialDiagnostic;
    std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> canonicalBlockCFrame;
    // Keep the prerequisite alive through asynchronous submission and through
    // the compositor's ownership of this output, not just executePreview().
    std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> materialFrameTexture;
    // Preview only. The frame is still compiling, so no frame is present and the caller retries.
    bool pending = false;

    bool valid() const noexcept
    {
        return deformationFrame ? deformationFrame->valid()
                                : admission.valid() && frame.rendered.nativeFrame
                                    && (payload || admission.sourceBytes == 0);
    }
};

/** Resolves resident exact bytes into the shared preview/export GLB seam. */
class ImportedScenePayloadExecution final
{
public:
    using Receipt = ImportedSceneExecutionReceipt;

    ImportedScenePayloadExecution(Store& store,
                                  arbitgpu::NativeFixtureSceneBackend& backend) noexcept;
    ImportedScenePayloadExecution(Store& store,
                                  arbitgpu::NativeFixtureSceneBackend& backend,
                                  arbitgpu::NativeDeformationBackend& deformationBackend) noexcept;

    static bool validReceipt(const Receipt& receipt) noexcept { return receipt.valid(); }
    bool inspectParentReceipt(const Receipt& receipt,const ImportedSceneRequest& expected,ImportedSceneUse use) const noexcept;
    static const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& nativeFrame(
        const Receipt& receipt) noexcept
    {
        return receipt.deformationFrame
            ? receipt.deformationFrame->frame.nativeFrame
            : receipt.frame.rendered.nativeFrame;
    }
    static std::uintptr_t nativeDepthTextureView(const Receipt& receipt) noexcept
    {
        const auto& frame = nativeFrame(receipt);
        return frame ? frame->depthTextureViewHandle() : 0;
    }

    bool executePreview(const ImportedSceneRequest& request,
                        ImportedSceneExecutionReceipt& output,
                        std::string& error);
    bool executeExport(const ImportedSceneRequest& request,
                       ImportedSceneExecutionReceipt& output,
                       std::string& error);

    // Called before plan compilation. Changed publications retire the prior
    // owner, except an otherwise identical canonical static object/scene
    // selection, which must first pass exact source admission.
    void publishStaticPayload(int clipId,
                              std::uint64_t projectGeneration,
                              std::uint64_t helperGeneration,
                              const std::string& exactPayloadIdentity) noexcept;

    // Fixed-size presentation bookkeeping only. Historical native receipts
    // remain valid, but cannot be adopted again after source publication retires.
    Receipt::ParentPresentationGuard parentPresentationGuard(int clipId) const noexcept;
    Receipt::ParentPresentationGuard parentPresentationGuard(
        const std::shared_ptr<const ImportedSceneParentPresentation>& token,int clipId) const noexcept;
    std::shared_ptr<const ImportedSceneParentPresentation> parentPresentationToken(const Receipt& receipt) const noexcept;
    bool parentPublicationReady() const noexcept { return parentPublicationReady_; }
    // Lifetime work that static admissions have drawn on the persistent
    // ledger. Only reset() clears it, so steady playback must not grow it.
    std::uint64_t staticLedgerWork() const noexcept
    { return staticLocalBudget_ ? staticLocalBudget_->logicalWork() : 0; }

    void reset() noexcept;

private:
    friend class ImportedSceneParentReceiptStaging;
#if defined(ARBIT_TEST_IMPORTED_PARENT_STAGING_FAILURES)
    friend struct ImportedSceneParentStagingFailureProbe;
#endif
    bool execute(ImportedSceneUse use,
                 const ImportedSceneRequest& request,
                 ImportedSceneExecutionReceipt& output,
                 std::string& error);

    struct StaticOwner final
    {
        explicit StaticOwner(arbitgpu::NativeFixtureSceneBackend& backend) noexcept
            : seam(backend) {}
        gltf::GlbNativeRenderSeam seam;
        visualmodelassetpayload::PayloadPtr payload;
        std::optional<std::size_t> sceneIndex;
        std::optional<std::uint32_t> sourceCameraNodeId {};
        std::optional<HarmonicMIDI::grid::SceneCameraRecord> selectedSourceCamera;
        std::uint32_t objectNodeStableId = 0;
        std::uint32_t objectMeshStableId = 0;
        gltf::GlbNativeSceneAdmission admission;
        std::shared_ptr<const NativeSceneAnimationCompositionOwner> localAuthority;
        std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> proceduralScene;
        // Scene Compose keeps the admitted imported scene here, because
        // admission.scene holds the composed result. A new procedural
        // snapshot recomposes from it without a second decode or admission.
        std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> importedScene;
        std::string exactPayloadIdentity;
        std::optional<visualimportedscenerender::GlassCausticSelection> glassCaustic;
        std::uint64_t lastUse = 0;
        std::array<arbitgpu::NativeSceneMotionHistory, 2> motion;
    };

    bool stageStaticLocalFrame(ImportedSceneUse use, const ImportedSceneRequest& request,
        const StaticOwner* incumbent,
        std::shared_ptr<const NativeSceneAnimationCompositionFrame>& candidate, std::string& error);
    std::shared_ptr<NativeSceneAnimationCompositionBudget> staticLocalBudget_;
    std::shared_ptr<NativeCompositionRetentionBudget> staticLocalCallerBudget_;
    static void clearCallerReceipt(Receipt& output) noexcept;
    void retireParentPresentation() noexcept;
    void synchronizeGenerations(std::uint64_t projectGeneration,
                                std::uint64_t helperGeneration) noexcept;
    void synchronizeDevice() noexcept;
    void evictForInsertion() noexcept;

    static constexpr std::size_t kMaximumStaticOwners = 16;
    Store& store_;
    arbitgpu::NativeFixtureSceneBackend& backend_;
    std::unique_ptr<ImportedAnimatedScenePayloadExecution> deformationExecution_;
    using StaticOwnerKey = int;
    std::map<StaticOwnerKey, StaticOwner> staticOwners_;
    std::uint64_t projectGeneration_ = 0;
    std::uint64_t helperGeneration_ = 0;
    std::uint64_t useSerial_ = 0;
    std::string deviceIdentity_;
    bool retiredOwnerSinceLastAttempt_ = false;
    std::shared_ptr<const Receipt::ParentPayloadIdentity> parentPublication_;
    bool parentPublicationReady_ = true;
};

// The two existing executors use this concrete synchronous continuation only
// while the animated executor's original incumbent rollback remains active.
class ImportedSceneParentReceiptStaging final
{
    friend class ImportedScenePayloadExecution;
    friend class ImportedAnimatedScenePayloadExecution;
    ImportedSceneParentReceiptStaging(ImportedScenePayloadExecution& execution,
        const ImportedSceneRequest& request, ImportedSceneUse use,
        ImportedSceneExecutionReceipt& candidate,
        std::shared_ptr<ImportedSceneExecutionReceipt::ParentPayloadIdentity> identity) noexcept
        : execution_(execution),request_(request),use_(use),candidate_(candidate),identity_(std::move(identity))
        { candidate_.parentPayloadIdentity_=identity_; }
    bool prepareDiagnostic(const std::string& diagnostic,std::string& error);
    bool stage(ImportedAnimatedSceneReceipt& frame,std::string& error);
    void publish(ImportedSceneExecutionReceipt& output) noexcept;
    ImportedScenePayloadExecution& execution_;
    const ImportedSceneRequest& request_;
    ImportedSceneUse use_;
    ImportedSceneExecutionReceipt& candidate_;
    std::shared_ptr<ImportedSceneExecutionReceipt::ParentPayloadIdentity> identity_;
    std::size_t diagnosticCapacity_=0;
    bool replacePresentation_=false;
};

#if defined(ARBIT_TEST_IMPORTED_PARENT_STAGING_FAILURES)
// Fault selection exists only in the two owning native test executables. Weak
// witnesses observe actual production-issued candidates and never issue them.
struct ImportedSceneParentStagingFailureProbe final
{
    enum class Site { None,TokenObject,TokenControlBlock,PayloadAlias,SourceAlias,NativeAlias,DiagnosticCopy,StaticLocalOwner,StaticLocalFrame,StaticLocalSource };
    struct Sample final
    {
        bool present=false;
        std::array<float,3> translation{},rotation{},cameraTranslation{};
        float scale=1,morph=1;
        std::optional<HarmonicMIDI::grid::SceneCameraRecord> camera;
        arbitgpu::NativeSceneMotionOwner owner;
        std::weak_ptr<const visualdeformation::AnimationDeformationSnapshot> deformation;
        std::weak_ptr<const arbitgpu::NativeDeformationCompositionFrame> composition;
        std::weak_ptr<const arbitgpu::NativeSceneMotionSample::ImportedNodeWorldMatrices> worlds;
        std::weak_ptr<const arbitgpu::NativeSceneNoteMotionTransforms> notes;
    };
    struct History final
    {
        Sample current,previous;
        std::int64_t frame=0;
        std::uint32_t numerator=0,denominator=0,width=0,height=0;
        arbitgpu::NativeSceneMotionOwner owner;
        bool mappingEmpty=true,noteOwnerAbsent=true;
    };
    struct Witness final
    {
        std::array<History,2> history;
        std::weak_ptr<const arbitgpu::NativeDeformationScene> source;
        std::weak_ptr<const ImportedParentBindingOwner> parent;
        std::weak_ptr<const videorender::fixture3d::AdmittedSurfaceMaterialBinding> material;
        std::weak_ptr<NativeCompositionRetentionBudget> retentionBudget;
        std::weak_ptr<NativeSceneAnimationCompositionBudget> workBudget;
        std::uint64_t attemptedWork=0;
        std::array<std::uint64_t,3> materialRevisions{};
    };
    struct State final
    {
        Site selected=Site::None;
        bool hit=false;
        std::weak_ptr<const arbitgpu::NativeDeformationCompositionOwner> candidateOwner;
        std::weak_ptr<const arbitgpu::NativeFixtureSceneFrame> candidateNative;
        bool diagnosticAllocates=false;
        std::weak_ptr<const NativeSceneAnimationCompositionFrame> candidateLocalFrame;
        void (*localObserver)(const std::shared_ptr<const NativeSceneAnimationCompositionFrame>&) = nullptr;
    };
    static std::weak_ptr<NativeSceneAnimationCompositionBudget> staticLocalBudget(
        const ImportedScenePayloadExecution& execution) noexcept { return execution.staticLocalBudget_; }
    static std::weak_ptr<NativeCompositionRetentionBudget> staticLocalCallerBudget(
        const ImportedScenePayloadExecution& execution) noexcept { return execution.staticLocalCallerBudget_; }
    static State& state() noexcept { static thread_local State value;return value; }
    class Scope final
    {
    public:
        explicit Scope(Site site) noexcept { state()={};state().selected=site; }
        ~Scope() { state()={}; }
        Scope(const Scope&)=delete;
        Scope& operator=(const Scope&)=delete;
    };
    struct Diagnostic final
    {
        std::weak_ptr<const NativeCompositionRetentionBudget::Charge> retention;
        std::weak_ptr<NativeCompositionRetentionBudget> pool;
        std::size_t capacity=0,bytes=0;
    };
    static Diagnostic diagnostic(const ImportedSceneExecutionReceipt& receipt) noexcept;
    static Witness inspect(const ImportedScenePayloadExecution& execution) noexcept;
    static bool conserved(const ImportedScenePayloadExecution& execution,const Witness& witness) noexcept;
private:
    friend class ImportedSceneParentReceiptStaging;
    static void allocation(Site site,const ImportedAnimatedSceneReceipt& frame,bool diagnosticAllocates=false);
};
#endif
} // namespace videohelper::modelpayload
