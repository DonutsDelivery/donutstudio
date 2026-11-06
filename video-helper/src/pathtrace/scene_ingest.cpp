#include "scene_ingest.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

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

// Returns records.size() when no live record has the id.
template <typename Records, typename Id>
std::size_t indexOfId (const Records& records, std::size_t count, Id id) noexcept
{
    const std::size_t none = records.size();
    if (! id.isValid())
        return none;
    count = std::min (count, none);
    for (std::size_t i = 0; i < count; ++i)
        if (records[i].id == id)
            return i;
    return none;
}

// Names the first live count that its allocated buffer cannot hold. Ingest
// refuses such a scene whole rather than drop the records past the buffer.
std::string capacityRefusal (const Visual3DScene& scene)
{
    const struct { const char* name; std::size_t count; std::size_t allocated; } buffers[] {
        { "vertices", scene.vertexCount, scene.vertices.size() },
        { "indices", scene.indexCount, scene.indices.size() },
        { "objects", scene.objectCount, scene.objects.size() },
        { "materials", scene.materialCount, scene.materials.size() },
        { "cameras", scene.cameraCount, scene.cameras.size() },
        { "object instances", scene.objectInstanceCount, scene.objectInstances.size() },
    };
    for (const auto& buffer : buffers)
        if (buffer.count > buffer.allocated)
            return std::string ("3D scene refused by path-trace ingest: it counts ")
                + std::to_string (buffer.count) + " " + buffer.name + " but allocates "
                + std::to_string (buffer.allocated) + ".";
    return {};
}

bool usableNormal (Vec3 n) noexcept
{
    return isFinite (n) && dot (n, n) > 0.0f;
}

// One dP/du per object vertex, in world space, with the handedness the grating
// frame requires, so a polar cap's seam wedge, whose u steps back, agrees with its
// neighbours. A vertex takes both from its constant-v edges, which measure dP/du
// at that radius. Each edge counts once per side of the vertex however the quads
// split into triangles, so the sum is the circle's tangent and not tilted toward
// the wedge holding more triangles. The longest wins, which leaves out the seam
// edge's shrunken value. A vertex with no such edge uses its triangles.
std::vector<Vec3> smoothGrooveTangents (const Visual3DScene& scene, const SceneObjectRecord& object, const Mat4& world, bool mirrored)
{
    const std::size_t count = object.vertexCount;
    std::vector<Vec3> directions (count), edgeDirections (count);
    std::vector<float> edgeLengths (count, 0.0f), triangleLengths (count, 0.0f);
    for (std::size_t tri = 0; tri + 3 <= object.indexCount; tri += 3)
    {
        std::array<std::uint32_t, 3> local { scene.indices[object.firstIndex + tri],
                                             scene.indices[object.firstIndex + tri + 1],
                                             scene.indices[object.firstIndex + tri + 2] };
        if (local[0] >= count || local[1] >= count || local[2] >= count)
            continue;
        if (mirrored)
            std::swap (local[1], local[2]);
        std::array<Vec3, 3> p {};
        std::array<Vec2, 3> uv {};
        for (std::size_t k = 0; k < 3; ++k)
        {
            const auto& vertex = scene.vertices[object.firstVertex + local[k]];
            p[k] = transformPoint (world, toVec3 (vertex.position));
            uv[k] = { vertex.uv.x, vertex.uv.y };
        }
        const Vec3 edge1 = p[1] - p[0], edge2 = p[2] - p[0];
        const float du1 = uv[1].x - uv[0].x, dv1 = uv[1].y - uv[0].y;
        const float du2 = uv[2].x - uv[0].x, dv2 = uv[2].y - uv[0].y;
        const double area = static_cast<double> (du1) * dv2 - static_cast<double> (du2) * dv1;
        const Vec3 normal = normalizeOrZero (cross (edge1, edge2));
        if (! (std::abs (area) > 1.0e-14) || ! usableNormal (normal))
            continue;
        const float inverse = static_cast<float> (1.0 / area);
        Vec3 tangent = (edge1 * dv2 - edge2 * dv1) * inverse;
        const Vec3 bitangent = (edge2 * du1 - edge1 * du2) * inverse;
        const float size = length (tangent);
        if (! std::isfinite (size) || ! (size > 0.0f))
            continue;
        if (dot (bitangent, cross (normal, tangent)) < 0.0f)
            tangent = tangent * -1.0f;
        for (std::size_t k = 0; k < 3; ++k)
        {
            directions[local[k]] = directions[local[k]] + tangent * (1.0f / size);
            triangleLengths[local[k]] = std::max (triangleLengths[local[k]], size);
            const std::size_t next = (k + 1) % 3;
            const float du = std::abs (uv[next].x - uv[k].x), dv = std::abs (uv[next].y - uv[k].y);
            if (du > 1.0e-6f && dv <= 1.0e-6f)
            {
                const float edgeSize = length (p[next] - p[k]) / du;
                edgeLengths[local[k]] = std::max (edgeLengths[local[k]], edgeSize);
                edgeLengths[local[next]] = std::max (edgeLengths[local[next]], edgeSize);
                edgeDirections[local[k]] = edgeDirections[local[k]] + tangent * (1.0f / size);
                edgeDirections[local[next]] = edgeDirections[local[next]] + tangent * (1.0f / size);
            }
        }
    }
    for (std::size_t v = 0; v < count; ++v)
    {
        const bool onEdge = edgeLengths[v] > 0.0f;
        const Vec3 sum = onEdge ? edgeDirections[v] : directions[v];
        // Around a cap's hub the unit directions cancel to rounding noise.
        directions[v] = length (sum) > 1.0e-3f ? normalizeOrZero (sum) * (onEdge ? edgeLengths[v] : triangleLengths[v]) : Vec3 {};
    }
    return directions;
}

// The earlier frame of one draw while motion blur is on.
struct DrawMotion
{
    Mat4 previousWorld;
    const SceneVec3* previousPositions = nullptr;  // scene vertices, a prefix
    std::size_t previousPositionCount = 0;
    float open = 0.0f;
    float close = 0.0f;
};

void appendObject (const Visual3DScene& scene, std::size_t objectIndex, const Mat4& world,
                   std::uint32_t instanceIndex, IngestedScene& out, const DrawMotion* motion = nullptr)
{
    const SceneObjectRecord& object = scene.objects[objectIndex];
    const std::size_t sceneVertices = scene.vertexCount;
    const std::size_t sceneIndices = scene.indexCount;
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
    const bool hasMaterial = materialSlot < scene.materials.size();
    const float determinant = determinant3x3 (world);
    const bool mirrored = determinant < 0.0f;
    const NormalMatrix normals = normalMatrixOf (world);
    const float tangentSign = mirrored ? -1.0f : 1.0f;
    const std::vector<Vec3> grooveTangents = smoothGrooveTangents (scene, object, world, mirrored);

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
        attributes.visibility = object.visibility;
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
            attributes.grooveTangents[k] = grooveTangents[local[k]];
            attributes.colors[k] = vertex.color;
            attributes.roughness[k] = vertex.roughness;
        }
        attributes.hasShadingNormals = allNormals;
        attributes.authoredShading = allNormals && object.authoredShading;
        if (! allNormals)
            attributes.shadingNormals = { ng, ng, ng };

        for (const auto& vertex : p)
        {
            out.positions.push_back (vertex);
            out.bounds.grow (vertex);
        }
        if (motion != nullptr)
            for (std::size_t k = 0; k < 3; ++k)
            {
                const std::size_t sceneVertex = firstVertex + local[k];
                const Vec3 before = transformPoint (motion->previousWorld, sceneVertex < motion->previousPositionCount
                    ? toVec3 (motion->previousPositions[sceneVertex]) : toVec3 (scene.vertices[sceneVertex].position));
                // A non-finite earlier vertex holds the triangle still.
                const Vec3 open = isFinite (before) ? pointAtFrame (before, p[k], motion->open) : p[k];
                const Vec3 close = isFinite (before) ? pointAtFrame (before, p[k], motion->close) : p[k];
                out.motionOpen.push_back (isFinite (open) ? open : p[k]);
                out.motionClose.push_back (isFinite (close) ? close : p[k]);
                out.bounds.grow (out.motionOpen.back());
                out.bounds.grow (out.motionClose.back());
            }
        out.triangles.push_back (attributes);
    }
}

} // namespace

void IngestedScene::clear() noexcept
{
    positions.clear();
    motionOpen.clear();
    motionClose.clear();
    triangles.clear();
    bounds = {};
    stats = {};
    refusal.clear();
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
    const std::size_t objectCount = std::min (scene.objectCount, scene.objects.size());
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
    out.refusal = capacityRefusal (scene);
    if (! out.refusal.empty())
        return;
    const std::size_t objectCount = scene.objectCount;
    const bool overrides = options.objectWorlds != nullptr && options.objectWorldCount == scene.objectCount;
    const std::size_t instanceCount = scene.objectInstanceCount;
    const bool shutter = options.shutterClose > options.shutterOpen
        && std::isfinite (options.shutterOpen) && std::isfinite (options.shutterClose);
    const bool previousWorlds = options.previousObjectWorlds != nullptr
        && options.previousObjectWorldCount == scene.objectCount;
    const bool previousInstances = options.previousInstances != nullptr
        && options.previousInstanceCount == instanceCount;
    DrawMotion motion;
    motion.previousPositions = options.previousPositions;
    motion.previousPositionCount = options.previousPositions != nullptr
        ? std::min (options.previousPositionCount, scene.vertexCount) : 0;
    motion.open = options.shutterOpen;
    motion.close = options.shutterClose;
    for (std::size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex)
    {
        const SceneObjectRecord& object = scene.objects[objectIndex];
        if (object.indexCount < 3)
            continue;
        ++out.stats.objectsWithGeometry;
        const Mat4 world = overrides ? options.objectWorlds[objectIndex] : objectWorldMatrix (scene, objectIndex);
        const Mat4 previous = previousWorlds ? options.previousObjectWorlds[objectIndex] : world;
        if (scene.instancedObject.isValid() && object.id == scene.instancedObject)
        {
            for (std::size_t instance = 0; instance < instanceCount; ++instance)
            {
                motion.previousWorld = previous * transformMatrix (previousInstances
                    ? options.previousInstances[instance] : scene.objectInstances[instance]);
                appendObject (scene, objectIndex, world * transformMatrix (scene.objectInstances[instance]),
                              static_cast<std::uint32_t> (instance), out, shutter ? &motion : nullptr);
                ++out.stats.instancesExpanded;
            }
            continue;
        }
        motion.previousWorld = previous;
        appendObject (scene, objectIndex, world, kNoInstance, out, shutter ? &motion : nullptr);
    }
    // Nothing moved over the shutter: trace the still scene.
    bool moved = false;
    for (std::size_t i = 0; i < out.motionOpen.size() && ! moved; ++i)
        moved = out.motionOpen[i].x != out.motionClose[i].x || out.motionOpen[i].y != out.motionClose[i].y
             || out.motionOpen[i].z != out.motionClose[i].z;
    if (! moved)
    {
        out.motionOpen.clear();
        out.motionClose.clear();
    }
}

void setCameraShutter (PinholeCamera& camera, const SceneCameraRecord& previous, float open, float close) noexcept
{
    camera.moving = false;
    if (! (close > open) || ! std::isfinite (open) || ! std::isfinite (close))
        return;
    const auto& q = previous.transform.rotation;
    const float norm = std::sqrt (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    const Vec3 before = toVec3 (previous.transform.translation);
    if (! isFinite (before) || ! std::isfinite (norm) || ! (norm > 0.0f))
        return;
    const Quat rotationBefore { q.x / norm, q.y / norm, q.z / norm, q.w / norm };
    const Vec3 now = camera.position;
    const Quat rotationNow = camera.rotation;
    const bool still = before.x == now.x && before.y == now.y && before.z == now.z
        && rotationBefore.x == rotationNow.x && rotationBefore.y == rotationNow.y
        && rotationBefore.z == rotationNow.z && rotationBefore.w == rotationNow.w;
    if (still)
        return;
    camera.position = pointAtFrame (before, now, open);
    camera.closePosition = pointAtFrame (before, now, close);
    camera.rotation = nlerpRotation (rotationBefore, rotationNow, 1.0f + open);
    camera.closeRotation = nlerpRotation (rotationBefore, rotationNow, 1.0f + close);
    camera.moving = true;
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
        || ! (record.nearPlane > 0.0f) || ! (record.farPlane > record.nearPlane)
        || ! std::isfinite (record.orthographicHeight) || record.orthographicHeight < 0.0f)
        return false;
    camera.position = position;
    camera.orthographicHalfHeight = record.orthographicHeight * 0.5f;
    camera.rotation = { q.x / norm, q.y / norm, q.z / norm, q.w / norm };
    camera.tanHalfFov = std::tan (record.verticalFovRadians * 0.5f);
    camera.aspect = static_cast<float> (width) / static_cast<float> (height);
    camera.nearPlane = record.nearPlane;
    camera.farPlane = record.farPlane;
    camera.width = width;
    camera.height = height;
    if (! HarmonicMIDI::grid::validSceneCameraLens (record))
        return false;
    camera.apertureRadius = HarmonicMIDI::grid::sceneCameraApertureRadius (record);
    camera.apertureRatio = record.apertureRatio;
    camera.focusDistance = record.focusDistance;
    camera.apertureBlades = record.apertureBlades;
    camera.bladeRotation = record.bladeRotationRadians;
    return true;
}

bool resolveActiveCamera (const Visual3DScene& scene, std::uint32_t width, std::uint32_t height,
                          PinholeCamera& camera) noexcept
{
    const std::size_t slot = indexOfId (scene.cameras, scene.cameraCount, scene.activeCamera);
    if (slot >= std::min (scene.cameraCount, scene.cameras.size()))
        return false;
    HarmonicMIDI::grid::SceneCameraRecord record = scene.cameras[slot];
    HarmonicMIDI::grid::resolveSceneCameraFocus (scene, record);
    return makePinholeCamera (record, width, height, camera);
}

} // namespace videohelper::pathtrace
