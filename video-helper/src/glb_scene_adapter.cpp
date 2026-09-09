#include "glb_scene_adapter.h"
#include "../../shared/VisualImportedSceneIdentity.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace videohelper::gltf
{
namespace
{
using namespace HarmonicMIDI::grid;

static_assert(GlbLimits {}.maxPrimitives
                  < visualimportedsceneidentity::kObjectPrimitiveStride,
              "GLB primitive indices must fit the stable object identity stride");
static_assert((GlbLimits {}.maxNodes - 1)
                  * visualimportedsceneidentity::kObjectPrimitiveStride
                  + GlbLimits {}.maxPrimitives
              <= std::numeric_limits<std::uint32_t>::max(),
              "admitted GLB node and primitive indices must fit SceneObjectId");

bool fail(std::string& error, const std::string& message)
{
    error = message;
    return false;
}

bool checkedMultiply(std::size_t left, std::size_t right, std::size_t& result)
{
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
        return false;
    result = left * right;
    return true;
}

template <typename Value>
bool appendFits(std::size_t current, std::size_t addition, Value capacity)
{
    const auto available = static_cast<std::size_t>(capacity);
    return current <= available && addition <= available - current;
}

const char* validationCodeName(Visual3DSceneValidationCode code)
{
    using Code = Visual3DSceneValidationCode;
    switch (code)
    {
        case Code::Valid: return "Valid";
        case Code::UnsupportedSchema: return "UnsupportedSchema";
        case Code::MissingSceneIdentity: return "MissingSceneIdentity";
        case Code::VertexCapacityExceeded: return "VertexCapacityExceeded";
        case Code::IndexCapacityExceeded: return "IndexCapacityExceeded";
        case Code::ObjectCapacityExceeded: return "ObjectCapacityExceeded";
        case Code::MaterialCapacityExceeded: return "MaterialCapacityExceeded";
        case Code::TextureCapacityExceeded: return "TextureCapacityExceeded";
        case Code::TextureTexelCapacityExceeded: return "TextureTexelCapacityExceeded";
        case Code::LightCapacityExceeded: return "LightCapacityExceeded";
        case Code::CameraCapacityExceeded: return "CameraCapacityExceeded";
        case Code::InvalidAmbientColor: return "InvalidAmbientColor";
        case Code::InvalidObjectIdentity: return "InvalidObjectIdentity";
        case Code::DuplicateObjectIdentity: return "DuplicateObjectIdentity";
        case Code::InvalidParentObject: return "InvalidParentObject";
        case Code::CyclicObjectHierarchy: return "CyclicObjectHierarchy";
        case Code::InvalidObjectTransform: return "InvalidObjectTransform";
        case Code::InvalidObjectHierarchyTransform: return "InvalidObjectHierarchyTransform";
        case Code::InvalidGeometryRange: return "InvalidGeometryRange";
        case Code::InvalidTriangleIndexCount: return "InvalidTriangleIndexCount";
        case Code::IndexOutsideObjectVertexRange: return "IndexOutsideObjectVertexRange";
        case Code::InvalidVertexValue: return "InvalidVertexValue";
        case Code::InvalidMaterialIdentity: return "InvalidMaterialIdentity";
        case Code::DuplicateMaterialIdentity: return "DuplicateMaterialIdentity";
        case Code::InvalidTextureIdentity: return "InvalidTextureIdentity";
        case Code::DuplicateTextureIdentity: return "DuplicateTextureIdentity";
        case Code::InvalidTextureRange: return "InvalidTextureRange";
        case Code::MissingBaseColorTexture: return "MissingBaseColorTexture";
        case Code::MissingObjectMaterial: return "MissingObjectMaterial";
        case Code::InvalidMaterialValue: return "InvalidMaterialValue";
        case Code::InvalidLightIdentity: return "InvalidLightIdentity";
        case Code::DuplicateLightIdentity: return "DuplicateLightIdentity";
        case Code::InvalidLightKind: return "InvalidLightKind";
        case Code::InvalidLightValue: return "InvalidLightValue";
        case Code::InvalidCameraIdentity: return "InvalidCameraIdentity";
        case Code::DuplicateCameraIdentity: return "DuplicateCameraIdentity";
        case Code::InvalidCameraTransform: return "InvalidCameraTransform";
        case Code::InvalidCameraProjection: return "InvalidCameraProjection";
        case Code::MissingActiveCamera: return "MissingActiveCamera";
    }
    return "Unknown";
}

bool near(float left, float right)
{
    const auto scale = std::max({1.0f, std::abs(left), std::abs(right)});
    return std::abs(left - right) <= 0.0001f * scale;
}

bool finiteMatrix(const std::array<float, 16>& matrix)
{
    return std::all_of(matrix.begin(), matrix.end(), [] (float value)
    {
        return std::isfinite(value);
    });
}

bool affineMatrix(const std::array<float, 16>& matrix)
{
    return finiteMatrix(matrix)
        && near(matrix[3], 0.0f) && near(matrix[7], 0.0f)
        && near(matrix[11], 0.0f) && near(matrix[15], 1.0f);
}

std::array<float, 16> multiply(const std::array<float, 16>& left,
                               const std::array<float, 16>& right)
{
    std::array<float, 16> result {};
    for (std::size_t column = 0; column < 4; ++column)
        for (std::size_t row = 0; row < 4; ++row)
            for (std::size_t inner = 0; inner < 4; ++inner)
                result[column * 4 + row] += left[inner * 4 + row]
                                               * right[column * 4 + inner];
    return result;
}

bool sameMatrix(const std::array<float, 16>& left,
                const std::array<float, 16>& right)
{
    for (std::size_t index = 0; index < left.size(); ++index)
        if (!near(left[index], right[index]))
            return false;
    return true;
}

std::array<float, 16> compose(const SceneTransform3D& transform)
{
    const auto x = transform.rotation.x;
    const auto y = transform.rotation.y;
    const auto z = transform.rotation.z;
    const auto w = transform.rotation.w;
    const auto xx = x * x, yy = y * y, zz = z * z;
    const auto xy = x * y, xz = x * z, yz = y * z;
    const auto wx = w * x, wy = w * y, wz = w * z;
    return {
        (1.0f - 2.0f * (yy + zz)) * transform.scale.x,
        (2.0f * (xy + wz)) * transform.scale.x,
        (2.0f * (xz - wy)) * transform.scale.x, 0.0f,
        (2.0f * (xy - wz)) * transform.scale.y,
        (1.0f - 2.0f * (xx + zz)) * transform.scale.y,
        (2.0f * (yz + wx)) * transform.scale.y, 0.0f,
        (2.0f * (xz + wy)) * transform.scale.z,
        (2.0f * (yz - wx)) * transform.scale.z,
        (1.0f - 2.0f * (xx + yy)) * transform.scale.z, 0.0f,
        transform.translation.x, transform.translation.y,
        transform.translation.z, 1.0f
    };
}

bool decompose(const std::array<float, 16>& matrix,
               SceneTransform3D& transform)
{
    if (!affineMatrix(matrix))
        return false;

    transform.translation = {matrix[12], matrix[13], matrix[14]};
    std::array<float, 3> x {matrix[0], matrix[1], matrix[2]};
    std::array<float, 3> y {matrix[4], matrix[5], matrix[6]};
    std::array<float, 3> z {matrix[8], matrix[9], matrix[10]};
    const auto length = [] (const std::array<float, 3>& value)
    {
        return std::sqrt(value[0] * value[0] + value[1] * value[1]
                         + value[2] * value[2]);
    };
    auto sx = length(x), sy = length(y), sz = length(z);
    if (!std::isfinite(sx) || !std::isfinite(sy) || !std::isfinite(sz)
        || sx <= 0.000001f || sy <= 0.000001f || sz <= 0.000001f)
        return false;
    for (auto& value : x) value /= sx;
    for (auto& value : y) value /= sy;
    for (auto& value : z) value /= sz;

    const auto dot = [] (const auto& left, const auto& right)
    {
        return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
    };
    if (!near(dot(x, y), 0.0f) || !near(dot(x, z), 0.0f)
        || !near(dot(y, z), 0.0f))
        return false;

    auto determinant = x[0] * (y[1] * z[2] - y[2] * z[1])
                     - y[0] * (x[1] * z[2] - x[2] * z[1])
                     + z[0] * (x[1] * y[2] - x[2] * y[1]);
    if (!near(std::abs(determinant), 1.0f))
        return false;
    if (determinant < 0.0f)
    {
        sx = -sx;
        for (auto& value : x) value = -value;
        determinant = -determinant;
    }
    if (!near(determinant, 1.0f))
        return false;

    const auto r00 = x[0], r01 = y[0], r02 = z[0];
    const auto r10 = x[1], r11 = y[1], r12 = z[1];
    const auto r20 = x[2], r21 = y[2], r22 = z[2];
    SceneQuaternion rotation;
    const auto trace = r00 + r11 + r22;
    if (trace > 0.0f)
    {
        const auto scale = 2.0f * std::sqrt(trace + 1.0f);
        rotation.w = 0.25f * scale;
        rotation.x = (r21 - r12) / scale;
        rotation.y = (r02 - r20) / scale;
        rotation.z = (r10 - r01) / scale;
    }
    else if (r00 > r11 && r00 > r22)
    {
        const auto scale = 2.0f * std::sqrt(1.0f + r00 - r11 - r22);
        rotation.w = (r21 - r12) / scale;
        rotation.x = 0.25f * scale;
        rotation.y = (r01 + r10) / scale;
        rotation.z = (r02 + r20) / scale;
    }
    else if (r11 > r22)
    {
        const auto scale = 2.0f * std::sqrt(1.0f + r11 - r00 - r22);
        rotation.w = (r02 - r20) / scale;
        rotation.x = (r01 + r10) / scale;
        rotation.y = 0.25f * scale;
        rotation.z = (r12 + r21) / scale;
    }
    else
    {
        const auto scale = 2.0f * std::sqrt(1.0f + r22 - r00 - r11);
        rotation.w = (r10 - r01) / scale;
        rotation.x = (r02 + r20) / scale;
        rotation.y = (r12 + r21) / scale;
        rotation.z = 0.25f * scale;
    }

    const auto quaternionLength = std::sqrt(rotation.x * rotation.x
                                          + rotation.y * rotation.y
                                          + rotation.z * rotation.z
                                          + rotation.w * rotation.w);
    if (!std::isfinite(quaternionLength) || quaternionLength <= 0.000001f)
        return false;
    rotation.x /= quaternionLength;
    rotation.y /= quaternionLength;
    rotation.z /= quaternionLength;
    rotation.w /= quaternionLength;
    if (rotation.w < 0.0f)
    {
        rotation.x = -rotation.x;
        rotation.y = -rotation.y;
        rotation.z = -rotation.z;
        rotation.w = -rotation.w;
    }
    transform.rotation = rotation;
    transform.scale = {sx, sy, sz};
    return sameMatrix(matrix, compose(transform));
}

bool collectSelectedNodes(const GlbStaticMeshDocument& asset,
                          std::vector<std::uint8_t>& selected,
                          std::string& error)
{
    if (asset.selectedScene >= asset.scenes.size())
        return fail(error, "selected GLB scene index is out of range");
    if (asset.nodes.size() > GlbLimits {}.maxNodes)
        return fail(error, "GLB node count exceeds the adapter bound");

    selected.assign(asset.nodes.size(), 0);
    std::vector<std::uint8_t> state(asset.nodes.size(), 0);
    struct Frame { std::size_t node; std::size_t nextChild; std::size_t depth; };
    std::vector<Frame> stack;
    stack.reserve(std::min(asset.nodes.size(), GlbLimits {}.maxNodeDepth));

    for (const auto root : asset.scenes[asset.selectedScene].rootNodes)
    {
        if (root >= asset.nodes.size())
            return fail(error, "selected GLB scene root node is out of range");
        if (asset.nodes[root].parent || state[root] != 0)
            return fail(error, "selected GLB scene roots do not form an independent tree");
        stack.push_back({root, 0, 1});
        while (!stack.empty())
        {
            auto& frame = stack.back();
            if (state[frame.node] == 0)
            {
                if (frame.depth > GlbLimits {}.maxNodeDepth)
                    return fail(error, "selected GLB scene hierarchy exceeds the adapter depth bound");
                state[frame.node] = 1;
                selected[frame.node] = 1;
            }
            const auto& children = asset.nodes[frame.node].children;
            if (frame.nextChild == children.size())
            {
                state[frame.node] = 2;
                stack.pop_back();
                continue;
            }
            const auto child = children[frame.nextChild++];
            if (child >= asset.nodes.size())
                return fail(error, "selected GLB node child is out of range");
            if (asset.nodes[child].parent != std::optional<std::size_t>(frame.node))
                return fail(error, "selected GLB node parent and child references disagree");
            if (state[child] != 0)
                return fail(error, "selected GLB node hierarchy is cyclic or shared");
            stack.push_back({child, 0, frame.depth + 1});
        }
    }

    for (std::size_t index = 0; index < asset.nodes.size(); ++index)
    {
        if (selected[index] == 0)
            continue;
        const auto& node = asset.nodes[index];
        if (!affineMatrix(node.localTransform) || !affineMatrix(node.worldTransform))
            return fail(error, "selected GLB node has a non-finite or non-affine transform");
        const auto expected = node.parent
            ? multiply(asset.nodes[*node.parent].worldTransform, node.localTransform)
            : node.localTransform;
        if (!sameMatrix(expected, node.worldTransform))
            return fail(error, "selected GLB node world transform disagrees with its hierarchy");
    }
    return true;
}

bool copyMaterials(const GlbStaticMeshDocument& asset,
                   Visual3DScene& scene,
                   std::string& error)
{
    if (asset.materials.size() > scene.materials.size())
        return fail(error, "GLB material count exceeds Visual3DScene capacity");

    std::vector<SceneTextureId> adaptedTextureIds(asset.textures.size());
    const auto copyTexture = [&] (const GlbTextureInfo& info) -> bool
    {
        const auto textureIndex = info.texture;
        if (textureIndex >= asset.textures.size())
            return fail(error, "GLB material texture index is out of range");
        if (info.texCoord != 0)
            return fail(error, "GLB material texture requires unsupported TEXCOORD set");
        if (adaptedTextureIds[textureIndex].isValid())
            return true;
        if (scene.textureCount >= Visual3DScene::kMaxTextures)
            return fail(error, "GLB material textures exceed Visual3DScene capacity");

        const auto& sourceTexture = asset.textures[textureIndex];
        if (sourceTexture.source >= asset.images.size())
            return fail(error, "GLB texture image source is out of range");
        if (!sourceTexture.sampler || *sourceTexture.sampler >= asset.samplers.size())
            return fail(error, "GLB material texture requires an explicit admitted sampler");
        const auto& sourceSampler = asset.samplers[*sourceTexture.sampler];

        const auto& sourceImage = asset.images[sourceTexture.source];
        std::size_t texelCount = 0;
        std::size_t expectedBytes = 0;
        if (sourceImage.width == 0 || sourceImage.height == 0
            || sourceImage.width > Visual3DScene::kMaxTextureTexels
            || sourceImage.height > Visual3DScene::kMaxTextureTexels
            || !checkedMultiply(static_cast<std::size_t>(sourceImage.width),
                                static_cast<std::size_t>(sourceImage.height), texelCount)
            || !checkedMultiply(texelCount, 4u, expectedBytes)
            || expectedBytes != sourceImage.decodedRgba8.size())
            return fail(error, "GLB decoded image texels are malformed or exceed Visual3DScene dimensions");
        if (texelCount > Visual3DScene::kMaxTextureTexels - scene.textureTexelCount)
            return fail(error, "GLB decoded image texels exceed Visual3DScene capacity");

        SceneTextureRecord texture;
        texture.id = SceneTextureId {static_cast<std::uint32_t>(textureIndex + 1)};
        texture.width = sourceImage.width;
        texture.height = sourceImage.height;
        texture.magFilter = sourceSampler.magFilter;
        texture.minFilter = sourceSampler.minFilter;
        texture.wrapS = sourceSampler.wrapS;
        texture.wrapT = sourceSampler.wrapT;
        texture.firstTexel = scene.textureTexelCount;
        for (std::size_t texelIndex = 0; texelIndex < texelCount; ++texelIndex)
        {
            const auto byte = texelIndex * 4u;
            scene.textureTexels[scene.textureTexelCount + texelIndex] = {
                sourceImage.decodedRgba8[byte],
                sourceImage.decodedRgba8[byte + 1],
                sourceImage.decodedRgba8[byte + 2],
                sourceImage.decodedRgba8[byte + 3]
            };
        }
        scene.textureTexelCount += texelCount;
        scene.textures[scene.textureCount++] = texture;
        adaptedTextureIds[textureIndex] = texture.id;
        return true;
    };

    for (const auto& material : asset.materials)
    {
        const std::array<const std::optional<GlbTextureInfo>*, 5> infos {{
            &material.baseColorTexture, &material.metallicRoughnessTexture,
            &material.normalTexture, &material.occlusionTexture,
            &material.emissiveTexture
        }};
        for (const auto* info : infos)
            if (*info && !copyTexture(**info))
                return false;
    }

    for (std::size_t index = 0; index < asset.materials.size(); ++index)
    {
        const auto& source = asset.materials[index];
        auto& target = scene.materials[index];
        target.id = SceneMaterialId {static_cast<std::uint32_t>(index + 1)};
        target.baseColor = {source.baseColorFactor[0], source.baseColorFactor[1],
                            source.baseColorFactor[2]};
        if (source.baseColorTexture)
            target.baseColorTexture = adaptedTextureIds[source.baseColorTexture->texture];
        if (source.metallicRoughnessTexture)
            target.metallicRoughnessTexture = adaptedTextureIds[source.metallicRoughnessTexture->texture];
        if (source.normalTexture)
            target.normalTexture = adaptedTextureIds[source.normalTexture->texture];
        if (source.occlusionTexture)
            target.occlusionTexture = adaptedTextureIds[source.occlusionTexture->texture];
        if (source.emissiveTexture)
            target.emissiveTexture = adaptedTextureIds[source.emissiveTexture->texture];
        target.opacity = source.alphaMode == GlbAlphaMode::Opaque
            ? 1.0f : source.baseColorFactor[3];
        target.metallic = source.metallicFactor;
        target.roughness = source.roughnessFactor;
        target.emissive = {source.emissiveFactor[0], source.emissiveFactor[1],
                           source.emissiveFactor[2]};
        target.normalScale = source.normalScale;
        target.occlusionStrength = source.occlusionStrength;
        target.alphaMode = source.alphaMode == GlbAlphaMode::Mask
            ? SceneAlphaMode::Mask
            : source.alphaMode == GlbAlphaMode::Blend
                ? SceneAlphaMode::Blend : SceneAlphaMode::Opaque;
        target.alphaCutoff = source.alphaCutoff;
        target.doubleSided = source.doubleSided;
    }
    scene.materialCount = asset.materials.size();
    return true;
}

bool copyPrimitive(const GlbPrimitiveRecord& primitive,
                   std::size_t nodeIndex,
                   std::size_t primitiveIndex,
                   const SceneTransform3D& transform,
                   SceneObjectId parent,
                   Visual3DScene& scene,
                   std::string& error)
{
    if (!primitive.material || *primitive.material >= scene.materialCount)
        return fail(error, "GLB primitive material reference is missing or out of range");

    if (!primitive.normalAccessor || primitive.normals.empty())
        return fail(error, "Visual3DScene adaptation requires a NORMAL attribute");
    if (primitive.positions.empty() || primitive.positions.size() % 3 != 0)
        return fail(error, "GLB POSITION data has invalid cardinality");

    const auto vertexCount = primitive.positions.size() / 3;
    if (primitive.normals.size() != vertexCount * 3)
        return fail(error, "GLB NORMAL cardinality does not match POSITION");
    if ((primitive.texCoord0Accessor && primitive.texCoords0.size() != vertexCount * 2)
        || (!primitive.texCoord0Accessor && !primitive.texCoords0.empty()))
        return fail(error, "GLB TEXCOORD_0 cardinality does not match POSITION");
    if ((primitive.tangentAccessor && primitive.tangents.size() != vertexCount * 4)
        || (!primitive.tangentAccessor && !primitive.tangents.empty()))
        return fail(error, "GLB TANGENT cardinality does not match POSITION");
    if ((primitive.color0Accessor && primitive.colors0.size() != vertexCount * 4)
        || (!primitive.color0Accessor && !primitive.colors0.empty()))
        return fail(error, "GLB COLOR_0 cardinality does not match POSITION");
    if (primitive.indices.empty() || primitive.indices.size() % 3 != 0)
        return fail(error, "GLB primitive topology is not a non-empty triangle list");
    if (!appendFits(scene.vertexCount, vertexCount, scene.vertices.size()))
        return fail(error, "GLB vertex data exceeds Visual3DScene capacity");
    if (!appendFits(scene.indexCount, primitive.indices.size(), scene.indices.size()))
        return fail(error, "GLB index data exceeds Visual3DScene capacity");
    if (scene.objectCount == scene.objects.size())
        return fail(error, "GLB primitive count exceeds Visual3DScene object capacity");

    const auto identity = visualimportedsceneidentity::objectId(nodeIndex, primitiveIndex);
    if (!identity)
        return fail(error, "GLB node and primitive indices exceed the stable object identity range");

    auto& object = scene.objects[scene.objectCount];
    object.id = *identity;
    object.parent = parent;
    object.material = SceneMaterialId {static_cast<std::uint32_t>(*primitive.material + 1)};
    object.transform = transform;
    object.firstVertex = static_cast<std::uint32_t>(scene.vertexCount);
    object.vertexCount = static_cast<std::uint32_t>(vertexCount);
    object.firstIndex = static_cast<std::uint32_t>(scene.indexCount);
    object.indexCount = static_cast<std::uint32_t>(primitive.indices.size());

    for (std::size_t index = 0; index < vertexCount; ++index)
    {
        auto& vertex = scene.vertices[scene.vertexCount + index];
        vertex.position = {primitive.positions[index * 3], primitive.positions[index * 3 + 1],
                           primitive.positions[index * 3 + 2]};
        vertex.normal = {primitive.normals[index * 3], primitive.normals[index * 3 + 1],
                         primitive.normals[index * 3 + 2]};
        if (!primitive.tangents.empty())
            vertex.tangent = {primitive.tangents[index * 4], primitive.tangents[index * 4 + 1],
                              primitive.tangents[index * 4 + 2], primitive.tangents[index * 4 + 3]};
        if (!primitive.texCoords0.empty())
            vertex.uv = {primitive.texCoords0[index * 2], primitive.texCoords0[index * 2 + 1]};
        if (!primitive.colors0.empty())
            vertex.color = {primitive.colors0[index * 4], primitive.colors0[index * 4 + 1],
                            primitive.colors0[index * 4 + 2], primitive.colors0[index * 4 + 3]};
    }
    for (std::size_t index = 0; index < primitive.indices.size(); ++index)
    {
        if (primitive.indices[index] >= vertexCount)
            return fail(error, "GLB primitive index is outside its POSITION range");
        scene.indices[scene.indexCount + index] = primitive.indices[index];
    }

    scene.vertexCount += vertexCount;
    scene.indexCount += primitive.indices.size();
    ++scene.objectCount;
    return true;
}
} // namespace

std::optional<Visual3DScene> adaptStaticGlbToVisual3DScene(
    const GlbStaticMeshDocument& asset,
    std::string& error,
    std::optional<std::size_t> meshIndex)
{
    error.clear();
    if (meshIndex && *meshIndex >= asset.meshes.size())
    {
        fail(error, "selected GLB mesh identity is unavailable");
        return std::nullopt;
    }
    std::vector<std::uint8_t> selected;
    if (!collectSelectedNodes(asset, selected, error))
        return std::nullopt;

    Visual3DScene scene;
    const auto sceneIdentity = visualimportedsceneidentity::sceneId(asset.selectedScene);
    if (!sceneIdentity)
    {
        fail(error, "selected GLB scene index exceeds the stable scene identity range");
        return std::nullopt;
    }
    scene.id = *sceneIdentity;
    if (!copyMaterials(asset, scene, error))
        return std::nullopt;

    std::vector<std::uint8_t> retainedObjectNodes(asset.nodes.size(), 0);
    for (std::size_t nodeIndex = 0; nodeIndex < asset.nodes.size(); ++nodeIndex)
    {
        const auto& node = asset.nodes[nodeIndex];
        if (selected[nodeIndex] == 0 || !node.mesh || !node.parent
            || (meshIndex && *node.mesh != *meshIndex))
            continue;
        for (auto retained = std::optional<std::size_t>(nodeIndex); retained;
             retained = asset.nodes[*retained].parent)
            retainedObjectNodes[*retained] = 1;
    }

    for (std::size_t nodeIndex = 0; nodeIndex < asset.nodes.size(); ++nodeIndex)
    {
        if (retainedObjectNodes[nodeIndex] == 0)
            continue;
        if (scene.objectCount == scene.objects.size())
        {
            fail(error, "GLB node hierarchy exceeds Visual3DScene object capacity");
            return std::nullopt;
        }
        const auto identity = visualimportedsceneidentity::nodeTransformObjectId(nodeIndex);
        if (!identity)
        {
            fail(error, "GLB node index exceeds the stable hierarchy identity range");
            return std::nullopt;
        }
        auto& object = scene.objects[scene.objectCount++];
        object.id = *identity;
        if (!decompose(asset.nodes[nodeIndex].localTransform, object.transform))
        {
            fail(error, "selected GLB node local transform cannot be represented as Visual3DScene TRS");
            return std::nullopt;
        }
        if (asset.nodes[nodeIndex].parent
            && retainedObjectNodes[*asset.nodes[nodeIndex].parent] != 0)
        {
            const auto parent = visualimportedsceneidentity::nodeTransformObjectId(
                *asset.nodes[nodeIndex].parent);
            if (!parent)
            {
                fail(error, "GLB parent node index exceeds the stable hierarchy identity range");
                return std::nullopt;
            }
            object.parent = *parent;
        }
    }

    for (std::size_t nodeIndex = 0; nodeIndex < asset.nodes.size(); ++nodeIndex)
    {
        if (selected[nodeIndex] == 0)
            continue;
        const auto& node = asset.nodes[nodeIndex];
        SceneTransform3D transform;
        if (!decompose(node.worldTransform, transform))
        {
            fail(error, "selected GLB node transform cannot be represented as Visual3DScene TRS");
            return std::nullopt;
        }

        if (node.mesh && (!meshIndex || *node.mesh == *meshIndex))
        {
            if (*node.mesh >= asset.meshes.size())
            {
                fail(error, "selected GLB node mesh reference is out of range");
                return std::nullopt;
            }
            const auto& mesh = asset.meshes[*node.mesh];
            SceneObjectId parentObject;
            SceneTransform3D objectTransform = transform;
            if (retainedObjectNodes[nodeIndex] != 0)
            {
                const auto nodeObject
                    = visualimportedsceneidentity::nodeTransformObjectId(nodeIndex);
                if (!nodeObject)
                {
                    fail(error, "GLB node index exceeds the stable hierarchy identity range");
                    return std::nullopt;
                }
                parentObject = *nodeObject;
                objectTransform = {};
            }
            for (std::size_t primitiveIndex = 0; primitiveIndex < mesh.primitives.size(); ++primitiveIndex)
                if (!copyPrimitive(mesh.primitives[primitiveIndex], nodeIndex, primitiveIndex,
                                   objectTransform, parentObject, scene, error))
                    return std::nullopt;
        }

        if (node.camera)
        {
            if (*node.camera >= asset.cameras.size())
            {
                fail(error, "selected GLB node camera reference is out of range");
                return std::nullopt;
            }
            if (scene.cameraCount == scene.cameras.size())
            {
                fail(error, "GLB camera instances exceed Visual3DScene capacity");
                return std::nullopt;
            }
            const auto& source = asset.cameras[*node.camera];
            if (source.type != GlbCameraType::Perspective || !source.farPlane)
            {
                fail(error, "Visual3DScene adaptation requires finite perspective cameras");
                return std::nullopt;
            }
            if (!near(transform.scale.x, 1.0f) || !near(transform.scale.y, 1.0f)
                || !near(transform.scale.z, 1.0f))
            {
                fail(error, "Visual3DScene cameras cannot inherit scale");
                return std::nullopt;
            }
            auto& target = scene.cameras[scene.cameraCount++];
            target.id = SceneCameraId {static_cast<std::uint32_t>(nodeIndex + 1)};
            target.transform = transform;
            target.transform.scale = {1.0f, 1.0f, 1.0f};
            target.verticalFovRadians = source.verticalFovRadians;
            target.nearPlane = source.nearPlane;
            target.farPlane = *source.farPlane;
            if (!scene.activeCamera)
                scene.activeCamera = target.id;
        }

        if (node.light)
        {
            if (*node.light >= asset.lights.size())
            {
                fail(error, "selected GLB node light reference is out of range");
                return std::nullopt;
            }
            if (scene.lightCount == scene.lights.size())
            {
                fail(error, "GLB light instances exceed Visual3DScene capacity");
                return std::nullopt;
            }
            const auto& source = asset.lights[*node.light];
            auto& target = scene.lights[scene.lightCount++];
            target.id = SceneLightId {static_cast<std::uint32_t>(nodeIndex + 1)};
            target.kind = source.type == GlbLightType::Directional
                ? SceneLightKind::Directional
                : source.type == GlbLightType::Spot
                    ? SceneLightKind::Spot : SceneLightKind::Point;
            target.transform = transform;
            target.color = {source.color[0], source.color[1], source.color[2]};
            target.intensity = source.intensity;
            target.range = source.range.value_or(0.0f);
            target.innerConeAngle = source.innerConeAngle;
            target.outerConeAngle = source.outerConeAngle;
        }
    }

    if (scene.cameraCount == 0)
    {
        fail(error, "selected GLB scene has no supported camera");
        return std::nullopt;
    }
    const auto validation = validateVisual3DScene(scene);
    if (!validation.valid())
    {
        fail(error, "translated GLB scene failed Visual3DScene validation: "
                    + std::string(validationCodeName(validation.code)) + " at record "
                    + std::to_string(validation.recordIndex));
        return std::nullopt;
    }
    return scene;
}
} // namespace videohelper::gltf
