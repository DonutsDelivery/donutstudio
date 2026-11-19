#pragma once

#include "Visual3DScene.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>

namespace HarmonicMIDI::grid::scenecomposition
{
// Imported GLB admission bounds nodes below 32768. Composition leaves their
// original node/primitive IDs intact and maps procedural IDs into the other
// half of the object space. The authored snapshot retains the original ID.
inline std::optional<SceneObjectId> proceduralObjectId(SceneObjectId authored) noexcept
{
    if (!authored.isValid() || authored.value >= 0x80000000u) return std::nullopt;
    return SceneObjectId {0x80000000u | authored.value};
}

// Mapping for every object record of a procedural snapshot. Geometry Core
// instance and material-draw records already carry the high bit (see
// GeometryCoreScene.h sceneInstanceId and GeometryMaterialTable.h), so they
// keep their value. Object IDs, Surface targets, and generated parents all use
// this mapping. Parent-intent anchors and compose roots use the strict
// proceduralObjectId instead.
inline std::optional<SceneObjectId> composedProceduralObjectId(SceneObjectId procedural) noexcept
{
    if (!procedural.isValid()) return std::nullopt;
    return SceneObjectId {0x80000000u | procedural.value};
}

// An authored ID and a generated ID can share their low 31 bits and would then
// map to one composed record. Such a snapshot cannot join an imported leaf.
inline bool distinctComposedProceduralObjects(const Visual3DScene& scene) noexcept
{
    if (scene.objectCount > scene.objects.size()) return false;
    for (std::size_t index = 0; index < scene.objectCount; ++index)
    {
        if (!scene.objects[index].id.isValid()) return false;
        const auto mapped = scene.objects[index].id.value | 0x80000000u;
        for (std::size_t other = 0; other < index; ++other)
            if ((scene.objects[other].id.value | 0x80000000u) == mapped) return false;
    }
    return true;
}

// Several imported leaves of one Scene Compose share the imported half of the
// object space. Leaf 0 keeps its raw GLB IDs, so a one-leaf composition and its
// selectors stay unchanged. In a multi-leaf composition, leaf k maps raw ID r
// to k * stride + r, and every leaf must have fewer than stride nodes,
// materials and textures.
//
// Node IDs always use kImportedLeafIdStride, because an object ID packs its
// node as node * 65536 below 0x80000000. Material and texture IDs have no such
// packing. The Narrow scheme gives them the node stride too; saved Scene
// Compose nodes keep it, so their Material ID literals name the same
// materials. Scene Compose nodes placed since the Wide scheme give materials
// and textures kImportedLeafWideIdStride, a format limit of 2^27 - 1 per leaf.
inline constexpr std::uint32_t kImportedLeafIdStride = 2048;
inline constexpr std::uint32_t kImportedLeafWideIdStride = 1u << 27;
inline constexpr std::size_t kMaximumImportedLeaves = 8;
static_assert(kMaximumImportedLeaves * kImportedLeafIdStride <= 16384,
              "Composed imported leaf IDs must stay within the object selector bound");
static_assert(kMaximumImportedLeaves * kImportedLeafWideIdStride <= 0x80000000u,
              "Wide composed material IDs must stay in the imported half of the ID space");

enum class ImportedLeafIdScheme : std::uint8_t
{
    Narrow = 0,
    Wide = 1
};

inline constexpr bool validImportedLeafIdScheme(std::uint64_t value) noexcept
{
    return value <= static_cast<std::uint64_t>(ImportedLeafIdScheme::Wide);
}

// Stride for material and texture IDs under a scheme. Node IDs pass Narrow.
inline constexpr std::uint32_t importedLeafIdStride(ImportedLeafIdScheme scheme) noexcept
{
    return scheme == ImportedLeafIdScheme::Wide ? kImportedLeafWideIdStride : kImportedLeafIdStride;
}

struct ImportedLeafId final
{
    std::size_t leaf = 0;
    std::uint32_t raw = 0;
};

inline bool validImportedLeafCount(std::size_t leafCount) noexcept
{
    return leafCount >= 1 && leafCount <= kMaximumImportedLeaves;
}

// Count is one leaf's node, material or texture count.
inline bool importedLeafCountFits(std::size_t leafCount, std::size_t count,
                                  ImportedLeafIdScheme scheme = ImportedLeafIdScheme::Narrow) noexcept
{
    return validImportedLeafCount(leafCount) && (leafCount == 1 || count < importedLeafIdStride(scheme));
}

inline std::optional<std::uint32_t> encodeImportedLeafId(std::size_t leaf, std::size_t leafCount,
                                                         std::uint32_t raw,
                                                         ImportedLeafIdScheme scheme = ImportedLeafIdScheme::Narrow) noexcept
{
    if (!validImportedLeafCount(leafCount) || leaf >= leafCount || raw == 0 || raw >= 0x80000000u)
        return std::nullopt;
    if (leafCount == 1) return raw;
    const auto stride = importedLeafIdStride(scheme);
    if (raw >= stride) return std::nullopt;
    return static_cast<std::uint32_t>(leaf) * stride + raw;
}

inline std::optional<ImportedLeafId> decodeImportedLeafId(std::uint32_t id, std::size_t leafCount,
                                                          ImportedLeafIdScheme scheme = ImportedLeafIdScheme::Narrow) noexcept
{
    if (!validImportedLeafCount(leafCount) || id == 0 || id >= 0x80000000u) return std::nullopt;
    if (leafCount == 1) return ImportedLeafId {0, id};
    const auto stride = importedLeafIdStride(scheme);
    const auto leaf = static_cast<std::size_t>(id / stride);
    const auto raw = id % stride;
    if (leaf >= leafCount || raw == 0) return std::nullopt;
    return ImportedLeafId {leaf, raw};
}

inline bool importedLeafOwnsId(std::uint32_t id, std::size_t leaf, std::size_t leafCount,
                               ImportedLeafIdScheme scheme = ImportedLeafIdScheme::Narrow) noexcept
{
    const auto decoded = decodeImportedLeafId(id, leafCount, scheme);
    return decoded && decoded->leaf == leaf;
}

// Imported object IDs are node * 65536 + primitive + 1, so an object follows
// the leaf scheme through its node: leaf k's object (node n, primitive p)
// becomes (encoded node n + 1, minus one) * 65536 + p + 1. Leaf 0 keeps raw IDs.
inline constexpr std::uint32_t kImportedObjectPrimitiveStride = 65536;

inline std::optional<SceneObjectId> composedImportedLeafObjectId(std::size_t leaf, std::size_t leafCount,
                                                                SceneObjectId raw) noexcept
{
    if (!raw.isValid() || raw.value >= 0x80000000u) return std::nullopt;
    const auto node = (raw.value - 1u) / kImportedObjectPrimitiveStride;
    const auto primitive = (raw.value - 1u) % kImportedObjectPrimitiveStride;
    const auto encoded = encodeImportedLeafId(leaf, leafCount, node + 1u);
    if (!encoded) return std::nullopt;
    const auto value = static_cast<std::uint64_t>(*encoded - 1u) * kImportedObjectPrimitiveStride + primitive + 1u;
    if (value >= 0x80000000u) return std::nullopt;
    return SceneObjectId {static_cast<std::uint32_t>(value)};
}

inline bool importedLeafOwnsObjectId(std::uint32_t id, std::size_t leaf, std::size_t leafCount) noexcept
{
    if (id == 0 || id >= 0x80000000u) return false;
    return importedLeafOwnsId((id - 1u) / kImportedObjectPrimitiveStride + 1u, leaf, leafCount);
}

inline bool sameVec3 (const SceneVec3& left, const SceneVec3& right) noexcept
{
    return left.x == right.x && left.y == right.y && left.z == right.z;
}

inline bool sameTransform (const SceneTransform3D& left,
                           const SceneTransform3D& right) noexcept
{
    return sameVec3 (left.translation, right.translation)
        && left.rotation.x == right.rotation.x
        && left.rotation.y == right.rotation.y
        && left.rotation.z == right.rotation.z
        && left.rotation.w == right.rotation.w
        && sameVec3 (left.scale, right.scale);
}

inline bool sameMaterialRecord (const SceneMaterialRecord& left,
                                const SceneMaterialRecord& right) noexcept
{
    return left.id == right.id && sameVec3 (left.baseColor, right.baseColor)
        && left.baseColorTexture == right.baseColorTexture
        && left.opacity == right.opacity && left.metallic == right.metallic
        && left.roughness == right.roughness && sameVec3 (left.emissive, right.emissive);
}

inline bool sameTextureRecord (const Visual3DScene& leftScene,
                               const SceneTextureRecord& left,
                               const Visual3DScene& rightScene,
                               const SceneTextureRecord& right) noexcept
{
    if (left.id != right.id || left.width != right.width || left.height != right.height
        || left.magFilter != right.magFilter || left.minFilter != right.minFilter
        || left.wrapS != right.wrapS || left.wrapT != right.wrapT)
        return false;
    const auto count = static_cast<std::size_t> (left.width) * left.height;
    if (left.firstTexel > leftScene.textureTexels.size()
        || count > leftScene.textureTexels.size() - left.firstTexel
        || right.firstTexel > rightScene.textureTexels.size()
        || count > rightScene.textureTexels.size() - right.firstTexel)
        return false;
    for (std::size_t index = 0; index < count; ++index)
    {
        const auto& leftTexel = leftScene.textureTexels[left.firstTexel + index];
        const auto& rightTexel = rightScene.textureTexels[right.firstTexel + index];
        if (leftTexel.red != rightTexel.red || leftTexel.green != rightTexel.green
            || leftTexel.blue != rightTexel.blue || leftTexel.alpha != rightTexel.alpha)
            return false;
    }
    return true;
}

inline bool sameLightRecord (const SceneLightRecord& left,
                             const SceneLightRecord& right) noexcept
{
    const auto bits = [](float a, float b) { return std::memcmp(&a, &b, sizeof(float)) == 0; };
    const auto vec = [&](SceneVec3 a, SceneVec3 b) { return bits(a.x,b.x) && bits(a.y,b.y) && bits(a.z,b.z); };
    const auto& a = left.transform;
    const auto& b = right.transform;
    return left.id == right.id && left.kind == right.kind
        && vec(a.translation,b.translation) && vec(a.scale,b.scale)
        && bits(a.rotation.x,b.rotation.x) && bits(a.rotation.y,b.rotation.y)
        && bits(a.rotation.z,b.rotation.z) && bits(a.rotation.w,b.rotation.w)
        && vec(left.color,right.color)
        && bits(left.intensity,right.intensity) && bits(left.range,right.range)
        && bits(left.innerConeAngle,right.innerConeAngle) && bits(left.outerConeAngle,right.outerConeAngle)
        && sameAnalyticSky(left.sky,right.sky)
        && left.environmentCameraVisible == right.environmentCameraVisible
        && sameCameraBackground(left.cameraBackground,right.cameraBackground)
        && sameEnvironmentImage(left.environmentImage,right.environmentImage)
        && sameSpectralSource(left.spectralSource,right.spectralSource)
        && sameLightShapeAndVisibility(left,right);
}

// Native preparation uploads the HDR image its scene's lights name. An imported scene
// whose image comes from the Render request's light is prepared from this copy, which
// names that image. The draw still reads the request's record each frame.
inline std::shared_ptr<const Visual3DScene> sceneWithEnvironmentImageLight (
    const Visual3DScene& scene, const SceneLightRecord& light)
{
    if (scene.lights.size() == 0) return nullptr;
    auto bound = std::make_shared<Visual3DScene> (scene);
    bound->lights[0] = light;
    bound->lightCount = 1;
    return bound;
}

inline bool sameCameraRecord (const SceneCameraRecord& left,
                              const SceneCameraRecord& right) noexcept
{
    return left.id == right.id && sameTransform (left.transform, right.transform)
        && left.verticalFovRadians == right.verticalFovRadians
        && left.nearPlane == right.nearPlane && left.farPlane == right.farPlane
        && left.orthographicHeight == right.orthographicHeight
        && sameCameraLens (left, right);
}

inline bool appendScene (const Visual3DScene& source, Visual3DScene& destination,
                         bool first, std::string& error,
                         SceneObjectId rootParent = {})
{
    if (! validateVisual3DScene (source).valid())
    {
        error = "Scene 3D composition received an invalid source scene";
        return false;
    }
    if (source.instancedObject.isValid() || destination.instancedObject.isValid())
    {
        error = "Scene 3D composition does not carry instanced objects";
        return false;
    }
    if (destination.textureCount > destination.textures.size()
        || destination.textureTexelCount != destination.textureTexels.size())
    {
        error = "Scene 3D composition received invalid destination texture storage";
        return false;
    }
    // Ambient belongs to the composed scene, which has one term. It takes each channel's
    // brightest source, so source order doesn't matter and a source at 0 dims nothing.
    if (! first)
    {
        destination.ambientColor.x = std::max (destination.ambientColor.x, source.ambientColor.x);
        destination.ambientColor.y = std::max (destination.ambientColor.y, source.ambientColor.y);
        destination.ambientColor.z = std::max (destination.ambientColor.z, source.ambientColor.z);
    }
    if (first)
    {
        destination.ambientColor = source.ambientColor;
        destination.activeCamera = source.activeCamera;
        destination.worldVolume = source.worldVolume;
    }
    // The composed scene has one fog too: the densest source's, so a source
    // without fog clears nothing.
    else if (sceneVolumeActive (source.worldVolume)
             && (! sceneVolumeActive (destination.worldVolume)
                 || source.worldVolume.density > destination.worldVolume.density))
        destination.worldVolume = source.worldVolume;

    // Room for everything the source could add. Records that merge by identity
    // leave the extra room unused.
    Visual3DSceneGrowth added;
    added.vertices = source.vertexCount;
    added.indices = source.indexCount;
    added.objects = source.objectCount;
    added.materials = source.materialCount;
    added.textures = source.textureCount;
    added.lights = source.lightCount;
    added.cameras = source.cameraCount;
    if (! growVisual3DScene (destination, added, error))
        return false;
    if (destination.vertexCount + source.vertexCount > std::numeric_limits<std::uint32_t>::max())
    {
        error = "Scene 3D composition vertex offset overflows its 32-bit index: "
            + std::to_string (destination.vertexCount) + " + " + std::to_string (source.vertexCount) + " vertices.";
        return false;
    }
    if (destination.indexCount + source.indexCount > std::numeric_limits<std::uint32_t>::max())
    {
        error = "Scene 3D composition index offset overflows its 32-bit range: "
            + std::to_string (destination.indexCount) + " + " + std::to_string (source.indexCount) + " indices.";
        return false;
    }

    for (std::size_t index = 0; index < source.textureCount; ++index)
    {
        const auto& record = source.textures[index];
        const auto* existing = visual3d_detail::findById (
            destination.textures, destination.textureCount, record.id);
        if (existing != nullptr)
        {
            if (! sameTextureRecord (source, record, destination, *existing))
            {
                error = "Scene 3D composition has a conflicting texture identity";
                return false;
            }
            continue;
        }
        const auto texelCount = static_cast<std::size_t> (record.width) * record.height;
        if (destination.textureTexelCount + texelCount > std::numeric_limits<std::uint32_t>::max())
        {
            error = "Scene 3D composition texel offset overflows its 32-bit first texel: "
                + std::to_string (destination.textureTexelCount) + " + " + std::to_string (texelCount) + " texels.";
            return false;
        }
        auto copy = record;
        copy.firstTexel = static_cast<std::uint32_t> (destination.textureTexelCount);
        destination.textureTexels.reserve (destination.textureTexelCount + texelCount);
        destination.textureTexels.resize (destination.textureTexelCount + texelCount);
        for (std::size_t texel = 0; texel < texelCount; ++texel)
            destination.textureTexels[destination.textureTexelCount + texel]
                = source.textureTexels[record.firstTexel + texel];
        destination.textureTexelCount += texelCount;
        destination.textures[destination.textureCount++] = copy;
    }

    for (std::size_t index = 0; index < source.materialCount; ++index)
    {
        const auto& record = source.materials[index];
        const auto* existing = visual3d_detail::findById (
            destination.materials, destination.materialCount, record.id);
        if (existing != nullptr)
        {
            if (! sameMaterialRecord (*existing, record))
            {
                error = "Scene 3D composition has a conflicting material identity";
                return false;
            }
            continue;
        }
        destination.materials[destination.materialCount++] = record;
    }

    const auto vertexBase = destination.vertexCount;
    const auto indexBase = destination.indexCount;
    for (std::size_t index = 0; index < source.vertexCount; ++index)
        destination.vertices[vertexBase + index] = source.vertices[index];
    for (std::size_t index = 0; index < source.indexCount; ++index)
        destination.indices[indexBase + index] = source.indices[index];
    destination.vertexCount += source.vertexCount;
    destination.indexCount += source.indexCount;

    for (std::size_t index = 0; index < source.objectCount; ++index)
    {
        auto record = source.objects[index];
        if (visual3d_detail::findById (
                destination.objects, destination.objectCount, record.id) != nullptr)
        {
            error = "Scene 3D composition has a duplicate object identity";
            return false;
        }
        record.firstVertex += static_cast<std::uint32_t> (vertexBase);
        record.firstIndex += static_cast<std::uint32_t> (indexBase);
        if (! record.parent.isValid() && rootParent.isValid())
            record.parent = rootParent;
        destination.objects[destination.objectCount++] = record;
    }

    for (std::size_t index = 0; index < source.lightCount; ++index)
    {
        const auto& record = source.lights[index];
        const auto* existing = visual3d_detail::findById (
            destination.lights, destination.lightCount, record.id);
        if (existing != nullptr)
        {
            if (! sameLightRecord (*existing, record))
            {
                error = "Scene 3D composition has a conflicting light identity";
                return false;
            }
            continue;
        }
        destination.lights[destination.lightCount++] = record;
    }

    for (std::size_t index = 0; index < source.cameraCount; ++index)
    {
        const auto& record = source.cameras[index];
        const auto* existing = visual3d_detail::findById (
            destination.cameras, destination.cameraCount, record.id);
        if (existing != nullptr)
        {
            if (! sameCameraRecord (*existing, record))
            {
                error = "Scene 3D composition has a conflicting camera identity";
                return false;
            }
            continue;
        }
        destination.cameras[destination.cameraCount++] = record;
    }
    return true;
}

} // namespace HarmonicMIDI::grid::scenecomposition
