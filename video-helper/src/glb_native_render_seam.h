#pragma once

#include "fixture_scene_renderer.h"
#include "glb_scene_adapter.h"
#include "../../shared/SurfaceMaterialBindingContract.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

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
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> scene;
    GlbMetadata metadata;
    std::size_t selectedScene = 0;
    std::size_t sourceBytes = 0;
    std::size_t decodedVertexCount = 0;
    std::size_t decodedIndexCount = 0;
    std::size_t decodedBytes = 0;

    bool valid() const noexcept { return scene != nullptr; }
};

struct GlbNativeFrameReceipt
{
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

class GlbNativeRenderSeam
{
public:
    static constexpr std::size_t kMaxSourceBytes = 16u * 1024u * 1024u;
    static constexpr std::size_t kMaxJsonBytes = 1u * 1024u * 1024u;
    static constexpr std::size_t kMaxBinBytes = kMaxSourceBytes - kMaxJsonBytes;

    explicit GlbNativeRenderSeam (
        arbitgpu::NativeFixtureSceneBackend& backend) noexcept;

    bool admit (const std::uint8_t* bytes,
                std::size_t size,
                std::optional<std::size_t> sceneIndex,
                GlbNativeSceneAdmission& output,
                std::string& error) const;

    bool admit (const std::vector<std::uint8_t>& bytes,
                std::optional<std::size_t> sceneIndex,
                GlbNativeSceneAdmission& output,
                std::string& error) const
    {
        return admit (bytes.data(), bytes.size(), sceneIndex, output, error);
    }

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
    std::shared_ptr<const videorender::fixture3d::AdmittedDiffractionMaterialBinding>
        lastGoodDiffractionMaterial_;
    std::uint64_t lastGoodMaterialRevision_ = 0;
    std::uint64_t rejectedMaterialRevision_ = 0;
    std::uint64_t highestMaterialRevisionSeen_ = 0;
};

} // namespace videohelper::gltf
