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

namespace canonicalblockc { class CanonicalBlockCFrame; }

namespace videohelper::modelpayload
{

enum class ImportedSceneUse : std::uint8_t
{
    Preview = 0,
    Export = 1
};

struct ImportedSceneRequest final
{
    visualanimationimport::ExactContentAssetKey asset;
    std::optional<std::size_t> sceneIndex;
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
    std::optional<diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest>
        diffractionMaterial;
    std::optional<HarmonicMIDI::grid::SceneCameraRecord> camera;
    std::optional<HarmonicMIDI::grid::SceneLightRecord> light;
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> sceneSnapshot;
    arbitgpu::NativeImportedSceneRuntimeInputs runtimeInputs;
};

// Owns every immutable object involved in one successful product execution:
// exact processor-supplied bytes, the admitted scene snapshot, and the native
// frame. No path or catalog record can substitute for `payload`.
struct ImportedSceneExecutionReceipt final
{
    ImportedSceneUse use = ImportedSceneUse::Preview;
    visualmodelassetpayload::PayloadPtr payload;
    gltf::GlbNativeSceneAdmission admission;
    gltf::GlbNativeFrameReceipt frame;
    std::optional<ImportedAnimatedSceneReceipt> deformationFrame;
    bool usedLastGoodMaterial = false;
    std::string materialDiagnostic;
    std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> canonicalBlockCFrame;

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

    // Called before plan compilation so a same-generation malformed replacement
    // cannot fall back to the prior clip owner.
    void publishStaticPayload(int clipId,
                              std::uint64_t projectGeneration,
                              std::uint64_t helperGeneration,
                              const std::string& exactPayloadIdentity) noexcept;

    void reset() noexcept;

private:
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
        gltf::GlbNativeSceneAdmission admission;
        std::string exactPayloadIdentity;
        std::uint64_t lastUse = 0;
    };

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
};
} // namespace videohelper::modelpayload
