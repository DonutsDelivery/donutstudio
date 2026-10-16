#include "scene_ingest.h"

#include <algorithm>
#include <cmath>
#include <utility>

// Behavioural reference for flattening object instances into one world-space
// triangle set: intern/cycles/bvh/build.cpp (Blender Cycles). Matrix and camera
// conventions follow this repository's raster backends, not Cycles.

namespace videohelper::pathtrace
{
namespace
{

using HarmonicMIDI::grid::SceneCameraRecord;
using HarmonicMIDI::grid::SceneObjectRecord;
using HarmonicMIDI::grid::SceneTransform3D;
using HarmonicMIDI::grid::SceneVec3;
using HarmonicMIDI::grid::Visual3DScene;

Vec3 toVec3 (const SceneVec3& v) noexcept { return { v.x, v.y, v.z }; }

template <typename Record, std::size_t N, typename Id>
std::size_t indexOfId (const std::array<Record, N>& records, std::size_t count, Id id) noexcept
{
    if (! id.isValid())
        return N;
    count = std::min (count, N);
    for (std::size_t i = 0; i < count; ++i)
        if (records[i].id == id)
            return i;
    return N;
}

bool usableNormal (Vec3 n) noexcept
{
    return isFinite (n) && dot (n, n) > 0.0f;
}

void appendObject (const Visual3DScene& scene, std::size_t objectIndex, const Mat4& world,
                   std::uint32_t instanceIndex, IngestedScene& out)
{
    const SceneObjectRecord& object = scene.objects[objectIndex];
    const std::size_t sceneVertices = std::min (scene.vertexCount, Visual3DScene::kMaxVertices);
    const std::size_t sceneIndices = std::min (scene.indexCount, Visual3DScene::kMaxIndices);
    const std::size_t firstVertex = object.firstVertex;
    const std::size_t vertexCount = object.vertexCount;
    const std::size_t firstIndex = object.firstIndex;
    const std::size_t indexCount = object.indexCount;
    if (firstVertex + vertexCount > sceneVertices || firstIndex + indexCount > sceneIndices)
    {
        ++out.stats.invalidObjectRanges;
        return;
    }

    const std::size_t materialSlot = indexOfId (scene.materials, scene.materialCount, object.material);
    const bool hasMaterial = materialSlot < Visual3DScene::kMaxMaterials;
    const float determinant = determinant3x3 (world);
    const bool mirrored = determinant < 0.0f;
    const NormalMatrix normals = normalMatrixOf (world);
    const float tangentSign = mirrored ? -1.0f : 1.0f;

    for (std::size_t tri = 0; tri + 3 <= indexCount; tri += 3)
    {
        std::array<std::uint32_t, 3> local {
            scene.indices[firstIndex + tri],
            scene.indices[firstIndex + tri + 1],
            scene.indices[firstIndex + tri + 2]
        };
        if (local[0] >= vertexCount || local[1] >= vertexCount || local[2] >= vertexCount)
        {
            ++out.stats.invalidIndices;
            continue;
        }
        // Stored order keeps the authored front face counter-clockwise in world space.
        if (mirrored)
            std::swap (local[1], local[2]);

        std::array<Vec3, 3> p {};
        for (std::size_t k = 0; k < 3; ++k)
            p[k] = transformPoint (world, toVec3 (scene.vertices[firstVertex + local[k]].position));
        if (! isFinite (p[0]) || ! isFinite (p[1]) || ! isFinite (p[2]))
        {
            ++out.stats.nonFiniteTriangles;
            continue;
        }
        const Vec3 ng = normalizeOrZero (cross (p[1] - p[0], p[2] - p[0]));
        if (! usableNormal (ng))
        {
            ++out.stats.degenerateTriangles;
            continue;
        }

        TriangleAttributes attributes;
        attributes.objectIndex = static_cast<std::uint32_t> (objectIndex);
        attributes.objectId = object.id.value;
        attributes.materialIndex = hasMaterial ? static_cast<std::uint32_t> (materialSlot) : kNoMaterial;
        attributes.materialId = hasMaterial ? object.material.value : 0u;
        attributes.instanceIndex = instanceIndex;
        attributes.primitiveIndex = static_cast<std::uint32_t> (tri / 3);
        attributes.mirrored = mirrored;
        attributes.geometricNormal = ng;
        bool allNormals = true;
        for (std::size_t k = 0; k < 3; ++k)
        {
            const auto& vertex = scene.vertices[firstVertex + local[k]];
            const Vec3 n = normalizeOrZero (normals.apply (toVec3 (vertex.normal)));
            allNormals = allNormals && usableNormal (n);
            attributes.shadingNormals[k] = n;
            attributes.uvs[k] = { vertex.uv.x, vertex.uv.y };
            const Vec3 t = normalizeOrZero (transformVector (
                world, { vertex.tangent[0], vertex.tangent[1], vertex.tangent[2] }));
            const float w = vertex.tangent[3] < 0.0f ? -1.0f : 1.0f;
            attributes.tangents[k] = { t.x, t.y, t.z, w * tangentSign };
        }
        attributes.hasShadingNormals = allNormals;
        if (! allNormals)
            attributes.shadingNormals = { ng, ng, ng };

        for (const auto& vertex : p)
        {
            out.positions.push_back (vertex);
            out.bounds.grow (vertex);
        }
        out.triangles.push_back (attributes);
    }
}

} // namespace

void IngestedScene::clear() noexcept
{
    positions.clear();
    triangles.clear();
    bounds = {};
    stats = {};
}

Mat4 transformMatrix (const SceneTransform3D& transform) noexcept
{
    const float x = transform.rotation.x, y = transform.rotation.y;
    const float z = transform.rotation.z, w = transform.rotation.w;
    const float sx = transform.scale.x, sy = transform.scale.y, sz = transform.scale.z;
    Mat4 m;
    m.m = { (1.0f - 2.0f * (y * y + z * z)) * sx, (2.0f * (x * y + w * z)) * sx, (2.0f * (x * z - w * y)) * sx, 0.0f,
            (2.0f * (x * y - w * z)) * sy, (1.0f - 2.0f * (x * x + z * z)) * sy, (2.0f * (y * z + w * x)) * sy, 0.0f,
            (2.0f * (x * z + w * y)) * sz, (2.0f * (y * z - w * x)) * sz, (1.0f - 2.0f * (x * x + y * y)) * sz, 0.0f,
            transform.translation.x, transform.translation.y, transform.translation.z, 1.0f };
    return m;
}

Mat4 objectWorldMatrix (const Visual3DScene& scene, std::size_t objectIndex) noexcept
{
    const std::size_t objectCount = std::min (scene.objectCount, Visual3DScene::kMaxObjects);
    if (objectIndex >= objectCount)
        return {};
    const SceneObjectRecord& object = scene.objects[objectIndex];
    Mat4 result = transformMatrix (object.transform);
    auto parent = object.parent;
    for (std::size_t depth = 0; parent.isValid() && depth < objectCount; ++depth)
    {
        const std::size_t slot = indexOfId (scene.objects, objectCount, parent);
        if (slot >= objectCount)
            break;
        result = transformMatrix (scene.objects[slot].transform) * result;
        parent = scene.objects[slot].parent;
    }
    return result;
}

void ingestScene (const Visual3DScene& scene, IngestedScene& out, const IngestOptions& options)
{
    out.clear();
    const std::size_t objectCount = std::min (scene.objectCount, Visual3DScene::kMaxObjects);
    const bool overrides = options.objectWorlds != nullptr && options.objectWorldCount == scene.objectCount;
    const std::size_t instanceCount = std::min (scene.objectInstanceCount, Visual3DScene::kMaxObjectInstances);
    for (std::size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex)
    {
        const SceneObjectRecord& object = scene.objects[objectIndex];
        if (object.indexCount < 3)
            continue;
        ++out.stats.objectsWithGeometry;
        const Mat4 world = overrides ? options.objectWorlds[objectIndex] : objectWorldMatrix (scene, objectIndex);
        if (scene.instancedObject.isValid() && object.id == scene.instancedObject)
        {
            for (std::size_t instance = 0; instance < instanceCount; ++instance)
            {
                appendObject (scene, objectIndex, world * transformMatrix (scene.objectInstances[instance]),
                              static_cast<std::uint32_t> (instance), out);
                ++out.stats.instancesExpanded;
            }
            continue;
        }
        appendObject (scene, objectIndex, world, kNoInstance, out);
    }
}

Ray PinholeCamera::generateRay (float pixelX, float pixelY) const noexcept
{
    const float ndcX = 2.0f * pixelX / static_cast<float> (width) - 1.0f;
    const float ndcY = 1.0f - 2.0f * pixelY / static_cast<float> (height);
    const Vec3 local { ndcX * tanHalfFov * aspect, ndcY * tanHalfFov, -1.0f };
    const float localLength = length (local);
    Ray ray;
    ray.origin = position;
    ray.direction = rotate (rotation, local) * (1.0f / localLength);
    ray.tMin = nearPlane * localLength;
    ray.tMax = farPlane * localLength;
    return ray;
}

bool PinholeCamera::project (Vec3 world, float& pixelX, float& pixelY, float& viewDepth) const noexcept
{
    const Vec3 view = rotate (conjugate (rotation), world - position);
    viewDepth = -view.z;
    if (! (viewDepth > 0.0f))
        return false;
    const float ndcX = view.x / (tanHalfFov * aspect * viewDepth);
    const float ndcY = view.y / (tanHalfFov * viewDepth);
    pixelX = (ndcX + 1.0f) * 0.5f * static_cast<float> (width);
    pixelY = (1.0f - ndcY) * 0.5f * static_cast<float> (height);
    return std::isfinite (pixelX) && std::isfinite (pixelY);
}

bool makePinholeCamera (const SceneCameraRecord& record, std::uint32_t width, std::uint32_t height,
                        PinholeCamera& camera) noexcept
{
    const auto& q = record.transform.rotation;
    const float norm = std::sqrt (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    const Vec3 position = toVec3 (record.transform.translation);
    if (width == 0 || height == 0 || ! isFinite (position) || ! std::isfinite (norm) || norm <= 0.0f
        || ! std::isfinite (record.verticalFovRadians) || ! (record.verticalFovRadians > 0.0f)
        || ! (record.verticalFovRadians < 3.1415926535f)
        || ! std::isfinite (record.nearPlane) || ! std::isfinite (record.farPlane)
        || ! (record.nearPlane > 0.0f) || ! (record.farPlane > record.nearPlane))
        return false;
    camera.position = position;
    camera.rotation = { q.x / norm, q.y / norm, q.z / norm, q.w / norm };
    camera.tanHalfFov = std::tan (record.verticalFovRadians * 0.5f);
    camera.aspect = static_cast<float> (width) / static_cast<float> (height);
    camera.nearPlane = record.nearPlane;
    camera.farPlane = record.farPlane;
    camera.width = width;
    camera.height = height;
    return true;
}

bool resolveActiveCamera (const Visual3DScene& scene, std::uint32_t width, std::uint32_t height,
                          PinholeCamera& camera) noexcept
{
    const std::size_t slot = indexOfId (scene.cameras, scene.cameraCount, scene.activeCamera);
    if (slot >= std::min (scene.cameraCount, Visual3DScene::kMaxCameras))
        return false;
    return makePinholeCamera (scene.cameras[slot], width, height, camera);
}

} // namespace videohelper::pathtrace
