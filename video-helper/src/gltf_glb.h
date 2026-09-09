#pragma once

#include "../../shared/VisualAnimationClip.h"
#include "../../shared/VisualDeformationContract.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace videohelper::gltf
{
struct GlbLimits
{
    std::size_t maxContainerBytes = 256u * 1024u * 1024u;
    std::size_t maxJsonBytes = 4u * 1024u * 1024u;
    std::size_t maxBinBytes = 252u * 1024u * 1024u;
    std::size_t maxJsonNestingDepth = 64;
    std::size_t maxScenes = 64;
    std::size_t maxNodes = 16384;
    std::size_t maxNodeDepth = 256;
    std::size_t maxMeshes = 4096;
    std::size_t maxPrimitives = 16384;
    std::size_t maxMaterials = 4096;
    std::size_t maxTextures = 4096;
    std::size_t maxImages = 4096;
    std::size_t maxSamplers = 4096;
    std::size_t maxCameras = 256;
    std::size_t maxLights = 256;
    std::size_t maxAccessors = 65536;
    std::size_t maxBufferViews = 65536;
    std::size_t maxDecodedVertices = 4u * 1024u * 1024u;
    std::size_t maxDecodedIndices = 12u * 1024u * 1024u;
    std::size_t maxDecodedBytes = 512u * 1024u * 1024u;
    std::size_t maxEmbeddedImageBytes = 128u * 1024u * 1024u;
    std::size_t maxImageWidth = 8192;
    std::size_t maxImageHeight = 8192;
    std::size_t maxDecodedImageBytes = 256u * 1024u * 1024u;
};

struct GlbAdmissionOptions
{
    GlbLimits limits;
    std::vector<std::string> supportedRequiredExtensions;
    bool admitAnimations = false;
    bool admitSkins = false;
    std::optional<std::size_t> sceneIndex;
};

struct GlbMetadata
{
    std::size_t containerBytes = 0;
    std::size_t jsonBytes = 0;
    std::size_t binBytes = 0;
    std::size_t scenes = 0;
    std::size_t nodes = 0;
    std::size_t meshes = 0;
    std::size_t primitives = 0;
    std::size_t materials = 0;
    std::size_t textures = 0;
    std::size_t images = 0;
    std::size_t samplers = 0;
    std::size_t cameras = 0;
    std::size_t animations = 0;
    std::size_t skins = 0;
    std::size_t accessors = 0;
    std::size_t bufferViews = 0;
    std::size_t maxHierarchyDepth = 0;
    std::optional<std::size_t> defaultScene;
    std::string generator;
    std::vector<std::string> extensionsUsed;
    std::vector<std::string> extensionsRequired;
};

enum class GlbAccessorType
{
    Scalar,
    Vec2,
    Vec3,
    Vec4,
    Mat2,
    Mat3,
    Mat4
};

enum class GlbAlphaMode
{
    Opaque,
    Mask,
    Blend
};

enum class GlbCameraType
{
    Perspective,
    Orthographic
};

enum class GlbLightType
{
    Directional,
    Point,
    Spot
};

struct GlbBufferViewRecord
{
    std::size_t byteOffset = 0;
    std::size_t byteLength = 0;
    std::optional<std::size_t> byteStride;
    std::optional<std::uint32_t> target;
};

struct GlbAccessorRecord
{
    std::size_t bufferView = 0;
    std::size_t byteOffset = 0;
    std::uint32_t componentType = 0;
    std::size_t count = 0;
    GlbAccessorType type = GlbAccessorType::Scalar;
    bool normalized = false;
};

struct GlbTextureInfo
{
    std::size_t texture = 0;
    std::size_t texCoord = 0;
};

struct GlbSamplerRecord
{
    std::string name;
    std::uint32_t magFilter = 9729;
    std::uint32_t minFilter = 9987;
    std::uint32_t wrapS = 10497;
    std::uint32_t wrapT = 10497;
};

struct GlbImageRecord
{
    std::string name;
    std::size_t bufferView = 0;
    std::string mimeType;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    // Row-major RGBA8 texels with (0, 0) at the top left.
    std::vector<std::uint8_t> decodedRgba8;
};

struct GlbTextureRecord
{
    std::string name;
    std::size_t source = 0;
    std::optional<std::size_t> sampler;
};

struct GlbMaterialRecord
{
    std::string name;
    std::array<float, 4> baseColorFactor {1.0f, 1.0f, 1.0f, 1.0f};
    float metallicFactor = 1.0f;
    float roughnessFactor = 1.0f;
    std::array<float, 3> emissiveFactor {0.0f, 0.0f, 0.0f};
    std::optional<GlbTextureInfo> baseColorTexture;
    std::optional<GlbTextureInfo> metallicRoughnessTexture;
    std::optional<GlbTextureInfo> normalTexture;
    std::optional<GlbTextureInfo> occlusionTexture;
    std::optional<GlbTextureInfo> emissiveTexture;
    float normalScale = 1.0f;
    float occlusionStrength = 1.0f;
    GlbAlphaMode alphaMode = GlbAlphaMode::Opaque;
    float alphaCutoff = 0.5f;
    bool doubleSided = false;
};

struct GlbPrimitiveRecord
{
    std::size_t positionAccessor = 0;
    std::optional<std::size_t> normalAccessor;
    std::optional<std::size_t> tangentAccessor;
    std::optional<std::size_t> texCoord0Accessor;
    std::optional<std::size_t> color0Accessor;
    std::size_t indexAccessor = 0;
    std::optional<std::size_t> material;
    std::vector<float> positions;
    std::vector<float> normals;
    std::vector<float> tangents;
    std::vector<float> texCoords0;
    std::vector<float> colors0;
    std::vector<std::uint32_t> indices;
};

struct GlbMeshRecord
{
    std::string name;
    std::vector<GlbPrimitiveRecord> primitives;
};

struct GlbCameraRecord
{
    std::string name;
    GlbCameraType type = GlbCameraType::Perspective;
    float aspectRatio = 0.0f;
    float verticalFovRadians = 0.0f;
    float xMagnification = 0.0f;
    float yMagnification = 0.0f;
    float nearPlane = 0.0f;
    std::optional<float> farPlane;
};

struct GlbLightRecord
{
    std::string name;
    GlbLightType type = GlbLightType::Directional;
    std::array<float, 3> color {1.0f, 1.0f, 1.0f};
    float intensity = 1.0f;
    std::optional<float> range;
    float innerConeAngle = 0.0f;
    float outerConeAngle = 0.7853981634f;
};

struct GlbNodeRecord
{
    std::string name;
    std::optional<std::size_t> mesh;
    std::optional<std::size_t> camera;
    std::optional<std::size_t> light;
    std::optional<std::size_t> parent;
    std::vector<std::size_t> children;
    std::array<float, 16> localTransform {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    std::array<float, 16> worldTransform {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
};

struct GlbSceneRecord
{
    std::string name;
    std::vector<std::size_t> rootNodes;
};

struct GlbStaticMeshDocument
{
    GlbMetadata metadata;
    std::size_t selectedScene = 0;
    std::size_t decodedVertexCount = 0;
    std::size_t decodedIndexCount = 0;
    std::size_t decodedBytes = 0;
    std::vector<GlbBufferViewRecord> bufferViews;
    std::vector<GlbAccessorRecord> accessors;
    std::vector<GlbSamplerRecord> samplers;
    std::vector<GlbImageRecord> images;
    std::vector<GlbTextureRecord> textures;
    std::vector<GlbMaterialRecord> materials;
    std::vector<GlbMeshRecord> meshes;
    std::vector<GlbCameraRecord> cameras;
    std::vector<GlbLightRecord> lights;
    std::vector<GlbNodeRecord> nodes;
    std::vector<GlbSceneRecord> scenes;
};

std::optional<GlbMetadata> admitGlbMetadata(const std::uint8_t* bytes,
                                             std::size_t size,
                                             const GlbAdmissionOptions& options,
                                             std::string& error);

inline std::optional<GlbMetadata> admitGlbMetadata(const std::vector<std::uint8_t>& bytes,
                                                    const GlbAdmissionOptions& options,
                                                    std::string& error)
{
    return admitGlbMetadata(bytes.data(), bytes.size(), options, error);
}

std::optional<GlbMetadata> admitGlbMetadataFile(std::string_view path,
                                                 const GlbAdmissionOptions& options,
                                                 std::string& error);

std::optional<GlbStaticMeshDocument> decodeStaticGlb(const std::uint8_t* bytes,
                                                      std::size_t size,
                                                      const GlbAdmissionOptions& options,
                                                      std::string& error);

// Decodes the renderable base scene from an animated/skinned GLB while keeping
// the public static decoder fail-closed for those payloads. This is an
// in-memory-only companion to decodeGlbAnimations: it does not acquire content
// and only exposes the same bounded geometry/material/camera/light records used
// by native scene adaptation.
std::optional<GlbStaticMeshDocument> decodeAnimatedGlbBaseScene(
    const std::uint8_t* bytes,
    std::size_t size,
    const GlbAdmissionOptions& options,
    std::string& error);

inline std::optional<GlbStaticMeshDocument> decodeStaticGlb(
    const std::vector<std::uint8_t>& bytes,
    const GlbAdmissionOptions& options,
    std::string& error)
{
    return decodeStaticGlb(bytes.data(), bytes.size(), options, error);
}

inline std::optional<GlbStaticMeshDocument> decodeAnimatedGlbBaseScene(
    const std::vector<std::uint8_t>& bytes,
    const GlbAdmissionOptions& options,
    std::string& error)
{
    return decodeAnimatedGlbBaseScene(bytes.data(), bytes.size(), options, error);
}

// Animation decode is a separate in-memory admission surface. It never opens a
// path and does not weaken the static decoder's animation/skin rejection.
struct GlbAnimationDecodeLimits
{
    std::size_t maxAnimations = 256;
    std::size_t maxSamplersPerAnimation = 4096;
    std::size_t maxChannelsPerAnimation = 4096;
    std::size_t maxTotalSamplers = 16384;
    std::size_t maxTotalChannels = 16384;
    std::size_t maxClipNameBytes = 1024;
    std::size_t maxTotalClipNameBytes = 65536;
    std::size_t maxDecodedBytes = 256u * 1024u * 1024u;
};

struct GlbAnimationDecodeOptions
{
    GlbAdmissionOptions admission;
    GlbAnimationDecodeLimits limits;
    visualanimation::Limits animationLimits;
    visualdeformation::Limits deformationLimits;
};

struct GlbJointAnimationBinding
{
    visualanimation::TargetId animationTarget;
    visualdeformation::SkinId skin;
    visualdeformation::JointId joint;
};

struct GlbMorphAnimationBinding
{
    visualanimation::TargetId animationTarget;
    visualdeformation::MeshId mesh;
    std::vector<visualdeformation::MorphTargetId> targets;
};

// Immutable render-side values decoded from the same admitted GLB bytes as the
// animation/deformation contracts. They carry data, never content authority.
struct GlbJointBaseTransform
{
    visualdeformation::SkinId skin;
    visualdeformation::JointId joint;
    std::array<float, 3> translation {0.0f, 0.0f, 0.0f};
    std::array<float, 4> rotation {0.0f, 0.0f, 0.0f, 1.0f};
    std::array<float, 3> scale {1.0f, 1.0f, 1.0f};
};

struct GlbDeformationRenderBinding
{
    visualdeformation::MeshId mesh;
    std::size_t nodeIndex = 0;
    std::vector<GlbJointBaseTransform> jointBaseTransforms;
    std::vector<float> morphBaseWeights;
};

struct GlbNamedAnimationClip
{
    std::string name;
    std::shared_ptr<const visualanimation::Clip> clip;
    std::vector<GlbJointAnimationBinding> jointBindings;
    std::vector<GlbMorphAnimationBinding> morphBindings;
};

struct GlbAnimationDocument
{
    GlbMetadata metadata;
    std::vector<GlbNamedAnimationClip> clips;
    std::shared_ptr<const visualdeformation::DeformationAsset> deformation;
    std::vector<GlbDeformationRenderBinding> renderBindings;
};

namespace detail
{
using AnimationClipFactory = std::shared_ptr<const visualanimation::Clip> (*) (
    const visualanimation::ClipView&, const visualanimation::Limits&, std::string&);
using DeformationAssetFactory = std::shared_ptr<const visualdeformation::DeformationAsset> (*) (
    const visualdeformation::AssetView&, const visualdeformation::Limits&, std::string&);

std::optional<GlbAnimationDocument> decodeGlbAnimationsWithFactories(
    const std::uint8_t* bytes,
    std::size_t size,
    const GlbAnimationDecodeOptions& options,
    AnimationClipFactory clipFactory,
    DeformationAssetFactory deformationFactory,
    std::string& error);
} // namespace detail

// The factories remain inline so gltf_glb.cpp can stay linked into the helper
// before the two contract implementation files join that product target.
inline std::optional<GlbAnimationDocument> decodeGlbAnimations(
    const std::uint8_t* bytes,
    std::size_t size,
    const GlbAnimationDecodeOptions& options,
    std::string& error)
{
    return detail::decodeGlbAnimationsWithFactories(
        bytes, size, options,
        [] (const visualanimation::ClipView& source,
            const visualanimation::Limits& limits,
            std::string& factoryError)
        {
            return visualanimation::Clip::create(source, limits, factoryError);
        },
        [] (const visualdeformation::AssetView& source,
            const visualdeformation::Limits& limits,
            std::string& factoryError)
        {
            return visualdeformation::DeformationAsset::create(source, limits, factoryError);
        },
        error);
}

inline std::optional<GlbAnimationDocument> decodeGlbAnimations(
    const std::vector<std::uint8_t>& bytes,
    const GlbAnimationDecodeOptions& options,
    std::string& error)
{
    return decodeGlbAnimations(bytes.data(), bytes.size(), options, error);
}
} // namespace videohelper::gltf
