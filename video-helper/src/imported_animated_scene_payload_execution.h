#pragma once

#include "imported_animation_deformation_consumer.h"
#include "fixture_scene_renderer.h"
#include "glb_scene_adapter.h"
#include "model_payload_transport.h"
#include "native_animation_deformation_renderer.h"

#include <optional>

namespace videohelper::modelpayload
{
struct ImportedAnimatedSceneAdmissionIdentity final
{
    visualmodelassetpayload::PayloadPtr payload;
    std::optional<std::size_t> sceneIndex;
    visualanimationimport::StableId meshStableId = 0;
    visualanimationimport::StableId animationClipStableId = 0;

    friend bool operator==(const ImportedAnimatedSceneAdmissionIdentity& left,
                           const ImportedAnimatedSceneAdmissionIdentity& right) noexcept
    {
        return left.payload == right.payload
            && left.sceneIndex == right.sceneIndex
            && left.meshStableId == right.meshStableId
            && left.animationClipStableId == right.animationClipStableId;
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

struct ImportedAnimatedSceneRequest final
{
    visualanimationimport::Request operation;
    visualdeformation::RationalFrameTime frame;
    std::uint64_t structuralRevision = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::optional<surfacematerialbinding::ImportedSceneMaterialRequest> material;
    arbitgpu::NativeImportedSceneRuntimeInputs runtimeInputs;
};

struct ImportedAnimatedSceneReceipt final
{
    ImportedAnimationEvaluationOwner owner = ImportedAnimationEvaluationOwner::Preview;
    visualmodelassetpayload::PayloadPtr payload;
    std::shared_ptr<const arbitgpu::NativeDeformationScene> source;
    ImportedAnimationDeformationEvaluation evaluation;
    videorender::animation3d::RenderedDeformationFrame frame;
    arbitgpu::NativeImportedSceneRuntimeInputs runtimeInputs;
    bool usedLastGoodMaterial = false;
    std::string materialDiagnostic;

    bool valid() const noexcept
    {
        return payload && source && evaluation.deformation && frame.nativeFrame
            && frame.nativeFrame->depthImageHandle() != 0
            && frame.nativeFrame->depthTextureViewHandle() != 0;
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
                                          arbitgpu::NativeDeformationBackend& backend) noexcept;

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
    void reset() noexcept;

private:
    bool execute(ImportedAnimationEvaluationOwner owner,
                 const ImportedAnimatedSceneRequest& request,
                 ImportedAnimatedSceneReceipt& output,
                 std::string& error);
    bool admit(const visualmodelassetpayload::PayloadPtr& payload,
               const visualanimationimport::Request& operation,
               std::string& error);
    bool publishSurfaceMaterial(
        const surfacematerialbinding::ImportedSceneMaterialRequest& request,
        std::string& error);
    void clearSurfaceMaterial() noexcept;

    Store& store_;
    ImportedAnimationDeformationConsumer consumer_;
    videorender::animation3d::NativeAnimationDeformationRenderer renderer_;
    ImportedAnimatedSceneAdmissionIdentity admittedIdentity_;
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> scene_;
    gltf::GlbDeformationRenderBinding binding_;
    std::shared_ptr<const arbitgpu::NativeDeformationScene> source_;
    std::weak_ptr<const HarmonicMIDI::grid::Visual3DScene> materialScene_;
    std::shared_ptr<const videorender::fixture3d::AdmittedSurfaceMaterialBinding>
        lastGoodMaterial_;
    std::uint64_t highestMaterialRevisionSeen_ = 0;
    std::uint64_t lastGoodMaterialRevision_ = 0;
    std::uint64_t rejectedMaterialRevision_ = 0;
};
} // namespace videohelper::modelpayload
