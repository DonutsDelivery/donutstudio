#pragma once

#include "fixture_scene_renderer.h"
#include "glb_scene_adapter.h"
#include "imported_light_evaluation.h"
#include "../../shared/SurfaceMaterialBindingContract.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace videohelper::modelpayload { class NativeSceneAnimationCompositionBudget; }

namespace videohelper::gltf
{

// This seam stops at a backend-owned native frame. It retains every admitted
// static draw, hierarchy identity, material, embedded texture and sampler.
// External URI/path resources, compositor attachment, readback and encoding
// remain outside this contract.
inline constexpr const char* kGlbNativeRenderSubset
    = "static-retained-rich-scene-v4";

enum class GlbTextureRole : std::uint8_t
{
    BaseColor, MetallicRoughness, Normal, Occlusion, Emissive
};
enum class GlbTextureColorSpace : std::uint8_t { Linear, Srgb };
struct GlbNativeTextureRoleReceipt
{
    HarmonicMIDI::grid::SceneMaterialId material {};
    HarmonicMIDI::grid::SceneTextureId texture {};
    GlbTextureRole role = GlbTextureRole::BaseColor;
    GlbTextureColorSpace colorSpace = GlbTextureColorSpace::Linear;
};
struct GlbNativeAttributeReceipt
{
    bool position = true;
    bool normal = true;
    bool tangentWithBitangentSign = true;
    bool uv0 = true;
    bool color0 = true;
};

struct GlbNativeDrawReceipt
{
    HarmonicMIDI::grid::SceneObjectId object {};
    HarmonicMIDI::grid::SceneObjectId parent {};
    HarmonicMIDI::grid::SceneMaterialId material {};
    HarmonicMIDI::grid::SceneTransform3D transform {};
    std::uint32_t firstVertex = 0;
    std::uint32_t vertexCount = 0;
    std::uint32_t firstIndex = 0;
    std::uint32_t indexCount = 0;
};

struct GlbNativeMaterialReceipt
{
    HarmonicMIDI::grid::SceneMaterialId material {};
    HarmonicMIDI::grid::SceneTextureId baseColorTexture {};
    HarmonicMIDI::grid::SceneTextureId metallicRoughnessTexture {};
    HarmonicMIDI::grid::SceneTextureId normalTexture {};
    HarmonicMIDI::grid::SceneTextureId occlusionTexture {};
    HarmonicMIDI::grid::SceneTextureId emissiveTexture {};
    HarmonicMIDI::grid::SceneVec3 baseColor {};
    float opacity = 1.0f;
    float metallic = 0.0f;
    float roughness = 1.0f;
    HarmonicMIDI::grid::SceneVec3 emissive {};
    float normalScale = 1.0f;
    float occlusionStrength = 1.0f;
    HarmonicMIDI::grid::SceneAlphaMode alphaMode = HarmonicMIDI::grid::SceneAlphaMode::Opaque;
    float alphaCutoff = 0.5f;
    bool doubleSided = false;
};

struct GlbNativeTextureReceipt
{
    HarmonicMIDI::grid::SceneTextureId texture {};
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t firstTexel = 0;
    std::uint32_t texelCount = 0;
    std::uint32_t magFilter = 0;
    std::uint32_t minFilter = 0;
    std::uint32_t wrapS = 0;
    std::uint32_t wrapT = 0;
};

struct GlbNativeSceneAdmission
{
    // The first decoder document and its logical debit have one lifetime.
    std::shared_ptr<const void> documentRetention;
    std::shared_ptr<const GlbStaticMeshDocument> document;
    std::size_t retainedDocumentBytes = 0;
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> scene;
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> sourceScene;
    GlbMetadata metadata;
    importedscene::LightDefinitions importedLightDefinitions;
    std::size_t selectedScene = 0;
    std::size_t sourceBytes = 0;
    std::size_t decodedVertexCount = 0;
    std::size_t decodedIndexCount = 0;
    std::size_t decodedBytes = 0;

    bool valid() const noexcept { return scene != nullptr; }
};

struct GlbNativeFrameReceipt
{
private:
    struct LocalRetention;
    std::shared_ptr<const LocalRetention> localRetention_;
    void exchange(GlbNativeFrameReceipt&) noexcept;
    void copyValues(const GlbNativeFrameReceipt&);
public:
    GlbNativeFrameReceipt() = default;
    GlbNativeFrameReceipt(const GlbNativeFrameReceipt&);
    GlbNativeFrameReceipt& operator=(const GlbNativeFrameReceipt&);
    GlbNativeFrameReceipt(GlbNativeFrameReceipt&&) noexcept;
    GlbNativeFrameReceipt& operator=(GlbNativeFrameReceipt&&) noexcept;
    void swap(GlbNativeFrameReceipt& other) noexcept { exchange(other); }
    // Allocation preparation only; this does not issue or authorize a native frame.
    bool prepareLocal(const std::shared_ptr<const modelpayload::NativeSceneAnimationCompositionFrame>&,
        const HarmonicMIDI::grid::Visual3DScene&, const std::string& backend, std::string& error);
    std::shared_ptr<const modelpayload::NativeSceneAnimationCompositionFrame> localFrame() const noexcept;

    const char* supportedSubset = kGlbNativeRenderSubset;
    bool downstreamHandoffComplete = false;
    GlbNativeAttributeReceipt attributes;
    std::vector<GlbNativeDrawReceipt> draws;
    std::vector<GlbNativeMaterialReceipt> materials;
    std::vector<GlbNativeTextureReceipt> textures;
    std::vector<GlbNativeTextureRoleReceipt> textureRoles;
    std::vector<HarmonicMIDI::grid::SceneCameraRecord> cameras;
    std::vector<HarmonicMIDI::grid::SceneLightRecord> lights;
    HarmonicMIDI::grid::SceneCameraId activeCamera {};
    std::string nativeBackend;
    std::uintptr_t nativeResourceCacheIdentity = 0;
    std::uintptr_t nativeDeviceOrContextIdentity = 0;
    std::uint64_t nativeRendererGeneration = 0;
    videorender::fixture3d::RenderedFrame rendered;
};

#if defined(ARBIT_TEST_IMPORTED_PARENT_STAGING_FAILURES)
struct GlbColdAdmissionProbe final
{
    std::size_t decodes = 0, adaptations = 0, sharedScenes = 0, metadataCopies = 0, lightMaps = 0;
    std::size_t textureTexels = 0, textureGrowthOverlapTexels = 0, reservedBytes = 0;
    std::size_t oldOverlaySourceDrawVisits = 0;
    static GlbColdAdmissionProbe& state() noexcept { static thread_local GlbColdAdmissionProbe value; return value; }
};
#endif

class GlbNativeRenderSeam
{
public:
    static constexpr std::size_t kMaxSourceBytes = 16u * 1024u * 1024u;
    static constexpr std::size_t kMaxJsonBytes = 1u * 1024u * 1024u;
    static constexpr std::size_t kMaxBinBytes = kMaxSourceBytes - kMaxJsonBytes;

    // Structural node storage for the selected standard library, excluding
    // allocator bookkeeping. Zero means that library boundary is unaccounted.
    static std::size_t lightDefinitionNodeBytes() noexcept;

    explicit GlbNativeRenderSeam (
        arbitgpu::NativeFixtureSceneBackend& backend) noexcept;

    bool admit (const std::uint8_t* bytes,
                std::size_t size,
                std::optional<std::size_t> sceneIndex,
                GlbNativeSceneAdmission& output,
                std::string& error,
                std::uint32_t objectNodeStableId = 0,
                std::uint32_t objectMeshStableId = 0,
                std::shared_ptr<modelpayload::NativeSceneAnimationCompositionBudget> documentBudget = {},
                bool optionalDocumentRetention = false) const;

    bool admit (const std::vector<std::uint8_t>& bytes,
                std::optional<std::size_t> sceneIndex,
                GlbNativeSceneAdmission& output,
                std::string& error,
                std::uint32_t objectNodeStableId = 0,
                std::uint32_t objectMeshStableId = 0,
                std::shared_ptr<modelpayload::NativeSceneAnimationCompositionBudget> documentBudget = {},
                bool optionalDocumentRetention = false) const
    {
        return admit (bytes.data(), bytes.size(), sceneIndex, output, error,
                      objectNodeStableId, objectMeshStableId, std::move(documentBudget), optionalDocumentRetention);
    }

    // Borrows the first drawn primitive matching the supplied document's node
    // and mesh. The pointer requires this exact scene to remain alive and
    // unchanged; it grants no source identity or native preparation authority.
    static const HarmonicMIDI::grid::SceneObjectRecord* selectedImportedPrototype(
        const GlbStaticMeshDocument& document,
        const HarmonicMIDI::grid::Visual3DScene& scene,
        std::uint32_t nodeStableId, std::uint32_t meshStableId) noexcept;

    // Borrows only the first admitted document, excluding synthetic cameras.
    static const HarmonicMIDI::grid::SceneCameraRecord* selectedSourceCamera(
        const GlbNativeSceneAdmission& admission, std::uint32_t node) noexcept;

    bool renderPreview (const GlbNativeSceneAdmission& admission,
                        videorender::fixture3d::RenderDimensions dimensions,
                        arbitgpu::NativeFixtureSceneRuntimeInputs sceneInputs,
                        GlbNativeFrameReceipt& output,
                        std::string& error);

    bool renderPreview (const GlbNativeSceneAdmission& admission,
                        videorender::fixture3d::RenderDimensions dimensions,
                        GlbNativeFrameReceipt& output,
                        std::string& error);

    bool renderExport (const GlbNativeSceneAdmission& admission,
                       videorender::fixture3d::RenderDimensions dimensions,
                       arbitgpu::NativeFixtureSceneRuntimeInputs sceneInputs,
                       GlbNativeFrameReceipt& output,
                       std::string& error);

    bool renderExport (const GlbNativeSceneAdmission& admission,
                       videorender::fixture3d::RenderDimensions dimensions,
                       GlbNativeFrameReceipt& output,
                       std::string& error);

    bool publishSurfaceMaterial (
        const GlbNativeSceneAdmission& admission,
        const surfacematerialbinding::ImportedSceneMaterialRequest& request,
        std::string& error);

    bool publishSurfacePrograms (
        const GlbNativeSceneAdmission& admission,
        const geometrysurfacematerial::ObjectPrograms& programs,
        const videowire::geometry::RuntimeFieldEvaluation& evaluation, std::string& error,
        const diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest* diffraction = nullptr,
        const visualimportedscenerender::GlassCausticSelection* caustic = nullptr);

    bool publishDiffractionMaterial (
        const GlbNativeSceneAdmission& admission,
        const diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest& request,
        std::string& error);

    void clearSurfaceMaterial (
        const GlbNativeSceneAdmission& admission) noexcept;

    void clearDiffractionMaterial (
        const GlbNativeSceneAdmission& admission) noexcept;

    std::uint64_t lastGoodMaterialRevision() const noexcept
    {
        return lastGoodMaterialRevision_;
    }

    std::uint64_t rejectedMaterialRevision() const noexcept
    {
        return rejectedMaterialRevision_;
    }

    bool surfaceUsesVertexProgram() const noexcept
    {
        const auto program = lastGoodMaterial_ ? lastGoodMaterial_->nativeProgram() : nullptr;
        return program && (program->vertexProgram
            || std::any_of(program->objectPrograms.begin(), program->objectPrograms.end(),
                [](const auto& value) { return value && value->vertexProgram; }));
    }

private:
    bool render (const GlbNativeSceneAdmission& admission,
                 videorender::fixture3d::RenderDimensions dimensions,
                 videorender::fixture3d::RenderUse use,
                 arbitgpu::NativeFixtureSceneRuntimeInputs sceneInputs,
                 GlbNativeFrameReceipt& output,
                 std::string& error);

    arbitgpu::NativeFixtureSceneBackend& backend_;
    videorender::fixture3d::FixtureSceneRenderer renderer_;
    std::weak_ptr<const HarmonicMIDI::grid::Visual3DScene> materialScene_;
    std::shared_ptr<const videorender::fixture3d::AdmittedSurfaceMaterialBinding>
        lastGoodMaterial_;
    std::string lastSurfaceProgramsIdentity_;
    std::shared_ptr<const videorender::fixture3d::AdmittedDiffractionMaterialBinding>
        lastGoodDiffractionMaterial_;
    std::uint64_t lastGoodMaterialRevision_ = 0;
    std::uint64_t rejectedMaterialRevision_ = 0;
    std::uint64_t highestMaterialRevisionSeen_ = 0;
};

} // namespace videohelper::gltf
