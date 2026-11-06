#pragma once

#include "fixture_surface_source_admission.h"

#include "../../../shared/Visual3DScene.h"
#include "backend.h"
#include "../material_program_compiler.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace arbitgpu::opaquevisibility
{
// Renderer-owned acceleration, never a serialized scene authority. Exact range
// instances share a BLAS; the TLAS holds their current world transforms. Both are
// sized from the scene and flattened into one RGBA32F texture of this width. Its
// height comes from the payload and is checked against the device's reported
// maximum texture size where the texture is created.
constexpr std::size_t kTextureWidth = 1024;
// The payload stores node, leaf and instance indices in float components, which
// are exact below 2^24. This is the encoding's range, not a scene budget.
constexpr std::size_t kExactFloatIndexRange = std::size_t(1) << 24;
using Texel = std::array<float, 4>;
using Matrix = std::array<float, 16>;
using Point = std::array<float, 3>;
// Median splitting halves the primitive range at each level, so a tree over any
// indexable range is at most 24 levels deep and each level leaves one entry pending.
constexpr std::size_t kClosestHitStackCapacity = 32;
// The range of the query's 32-bit work counter. A frame's allowance comes from
// its actual topology and saturates here.
constexpr std::size_t kMaximumClosestHitWork = UINT32_MAX;
inline bool payloadIndexable(std::size_t texels, std::string& error)
{
    if (texels <= kExactFloatIndexRange)
        return true;
    error = "Scene query payload has " + std::to_string(texels) + " texels and its float index encoding is exact to "
        + std::to_string(kExactFloatIndexRange);
    return false;
}
inline std::size_t treeNodes(std::size_t count, std::size_t leafSize)
{
    return count <= leafSize ? 1 : 1 + treeNodes(count / 2, leafSize) + treeNodes(count - count / 2, leafSize);
}
struct Bounds
{
    Point low{INFINITY, INFINITY, INFINITY};
    Point high{-INFINITY, -INFINITY, -INFINITY};
    void add(const Point& p)
    {
        for (int i = 0; i < 3; ++i)
        {
            low[i] = std::min(low[i], p[i]);
            high[i] = std::max(high[i], p[i]);
        }
    }
    void add(const Bounds& b)
    {
        add(b.low);
        add(b.high);
    }
};
struct Primitive
{
    Bounds bounds;
    std::array<Point, 3> triangle{};
    std::size_t record = 0;
    // First of three vertex-normal texels, or 0 when the face normal is the answer.
    std::size_t normals = 0;
};
struct Mesh
{
    std::size_t root = 0;
    Bounds bounds;
    std::size_t triangles = 0;
    std::size_t nodes = 0;
};
// Default admission preserves the released opaque-shadow payload. allSurfaces
// produces a separate frame for sceneClosestHit, never for opaqueVisibility.
enum class AdmissionMode { opaqueOnly, allSurfaces };
struct Prepared
{
    AdmissionMode mode = AdmissionMode::opaqueOnly;
    std::size_t objectCount = 0;
    std::vector<Texel> identities;
    std::vector<Texel> texels;
    std::vector<Mesh> meshes;
    std::vector<int> objectMesh;
};
struct Instance
{
    Matrix world;
    std::size_t object = 0;
};
struct Frame
{
    std::vector<Texel> texels;
    std::array<float, 4> control{};
    std::size_t height = 1;
};
inline Point point(const HarmonicMIDI::grid::SceneVec3& p)
{
    return {p.x, p.y, p.z};
}
inline Point transform(const Matrix& m, const Point& p, bool direction = false)
{
    Point q{};
    for (int r = 0; r < 3; ++r)
        q[r] = m[r] * p[0] + m[4 + r] * p[1] + m[8 + r] * p[2] + (direction ? 0 : m[12 + r]);
    return q;
}
inline bool inverse(const Matrix& m, Matrix& out)
{
    const double a = m[0], b = m[4], c = m[8], d = m[1], e = m[5], f = m[9], g = m[2], h = m[6], i = m[10];
    const double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if (!std::isfinite(det) || det == 0)
        return false;
    out = {float((e * i - f * h) / det),
           float((f * g - d * i) / det),
           float((d * h - e * g) / det),
           0,
           float((c * h - b * i) / det),
           float((a * i - c * g) / det),
           float((b * g - a * h) / det),
           0,
           float((b * f - c * e) / det),
           float((c * d - a * f) / det),
           float((a * e - b * d) / det),
           0,
           0,
           0,
           0,
           1};
    const auto t = transform(out, {m[12], m[13], m[14]}, true);
    for (int r = 0; r < 3; ++r)
        out[12 + r] = -t[r];
    return std::all_of(out.begin(), out.end(), [](float v) { return std::isfinite(v); });
}
inline std::size_t tree(std::vector<Texel>& texels, std::vector<Primitive>& primitives, std::size_t begin,
                        std::size_t end, bool triangles, bool identities = false)
{
    Bounds bounds;
    for (auto i = begin; i < end; ++i)
        bounds.add(primitives[i].bounds);
    const auto node = texels.size();
    texels.resize(node + 2);
    std::size_t leaf = 0;
    if (end - begin <= (triangles ? 4u : 1u))
    {
        if (triangles)
        {
            leaf = texels.size() + 1;
            for (auto i = begin; i < end; ++i)
                for (int v = 0; v < 3; ++v)
                {
                    const auto& p = primitives[i].triangle[v];
                    texels.push_back({p[0], p[1], p[2], v == 0 && i == begin ? float(end - begin) :
                        (identities && v == 1 ? float(primitives[i].record)
                            : v == 2 ? float(primitives[i].normals) : 0.0f)});
                }
        }
        else
            leaf = primitives[begin].record + 1;
    }
    else
    {
        int axis = 0;
        for (int a = 1; a < 3; ++a)
            if (bounds.high[a] - bounds.low[a] > bounds.high[axis] - bounds.low[axis])
                axis = a;
        const auto middle = begin + (end - begin) / 2;
        std::nth_element(
            primitives.begin() + begin, primitives.begin() + middle, primitives.begin() + end,
            [axis](const auto& a, const auto& b)
            { return a.bounds.low[axis] + a.bounds.high[axis] < b.bounds.low[axis] + b.bounds.high[axis]; });
        tree(texels, primitives, begin, middle, triangles, identities);
        tree(texels, primitives, middle, end, triangles, identities);
    }
    texels[node] = {bounds.low[0], bounds.low[1], bounds.low[2], float(texels.size())};
    texels[node + 1] = {bounds.high[0], bounds.high[1], bounds.high[2], float(leaf)};
    return node;
}
// Upper bound of one query table for this scene: seven texels per triangle of each
// distinct mesh range, one identity per object and eight per instance matrix (notes is
// the drawn note count, one without note instancing), rounded
// up to a whole row. Saturates rather than wraps, so an impossible table fails its charge.
// Glass receiver normal blocks are not counted. They add three texels per smooth receiver
// triangle, and the callers charge this figure four times over.
inline std::uint64_t sceneQueryPayloadTexels(const HarmonicMIDI::grid::Visual3DScene& scene,
                                             std::uint64_t notes = 1)
{
    std::set<std::array<std::uint32_t, 4>> ranges;
    std::uint64_t triangles = 0;
    for (std::size_t o = 0; o < scene.objectCount; ++o)
    {
        const auto& object = scene.objects[o];
        if (ranges.insert({object.firstVertex, object.vertexCount, object.firstIndex, object.indexCount}).second)
            triangles += (object.indexCount ? object.indexCount : object.vertexCount) / 3;
    }
    const auto instances = (std::uint64_t(scene.objectCount) + scene.objectInstanceCount) * (notes + 1);
    const auto saturated = std::numeric_limits<std::uint64_t>::max() / 16;
    if (triangles > saturated / 8 || instances > saturated / 16) return saturated;
    const auto texels = 7 * triangles + scene.objectCount + 8 * instances + 1;
    return (texels + kTextureWidth - 1) / kTextureWidth * kTextureWidth;
}

inline bool prepare(const HarmonicMIDI::grid::Visual3DScene& scene, const std::vector<char>& opaque,
                    Prepared& output, std::string& error, AdmissionMode mode = AdmissionMode::opaqueOnly,
                    const std::vector<char>* receiverNormals = nullptr)
{
    using namespace HarmonicMIDI::grid;
    Prepared candidate;
    candidate.mode = mode;
    candidate.objectCount = scene.objectCount;
    std::map<std::array<std::uint32_t, 4>, int> ranges;
    if (opaque.size() != scene.objectCount)
    {
        error = "Opaque visibility object roster does not match the scene";
        return false;
    }
    candidate.identities.resize(scene.objectCount);
    candidate.objectMesh.assign(scene.objectCount, -1);
    for (std::size_t o = 0; o < scene.objectCount; ++o)
    {
        const auto& object = scene.objects[o];
        if (!opaque[o] || (object.indexCount == 0 &&
            (mode == AdmissionMode::opaqueOnly || object.vertexCount == 0)))
            continue;
        const auto count = object.indexCount ? object.indexCount : object.vertexCount;
        const auto objectId = object.id.value, materialId = object.material.value;
        candidate.identities[o] = {float(objectId & 65535u), float(objectId >> 16),
                                   float(materialId & 65535u), float(materialId >> 16)};
        if (object.firstVertex > scene.vertexCount ||
            object.vertexCount > scene.vertexCount - object.firstVertex ||
            object.firstIndex > scene.indexCount ||
            object.indexCount > scene.indexCount - object.firstIndex || count % 3)
        {
            error = "Opaque visibility mesh range is malformed";
            return false;
        }
        const std::array<std::uint32_t, 4> key{object.firstVertex, object.vertexCount, object.firstIndex,
                                               object.indexCount};
        const auto found = ranges.find(key);
        if (found != ranges.end())
        {
            candidate.objectMesh[o] = found->second;
            continue;
        }
        std::vector<Primitive> triangles;
        triangles.reserve(count / 3);
        for (std::size_t index = 0; index < count; index += 3)
        {
            Primitive p;
            p.record = index / 3;
            for (int v = 0; v < 3; ++v)
            {
                const auto local = object.indexCount ? scene.indices[object.firstIndex + index + v] : index + v;
                if (local >= object.vertexCount)
                {
                    error = "Opaque visibility index exceeds its object range";
                    return false;
                }
                p.triangle[v] = point(scene.vertices[object.firstVertex + local].position);
                for (float component : p.triangle[v])
                    if (!std::isfinite(component))
                    {
                        error = "Opaque visibility vertex is non-finite";
                        return false;
                    }
                p.bounds.add(p.triangle[v]);
            }
            // A face whose vertex normals all equal its geometric normal stores none,
            // so a flat mesh keeps the same texels and the same shading.
            if (receiverNormals && (*receiverNormals)[o])
            {
                std::array<Point, 3> normals{};
                bool smooth = false;
                const auto& t = p.triangle;
                const Point e1{t[1][0] - t[0][0], t[1][1] - t[0][1], t[1][2] - t[0][2]};
                const Point e2{t[2][0] - t[0][0], t[2][1] - t[0][1], t[2][2] - t[0][2]};
                const Point face{e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
                const auto faceLength = std::sqrt(face[0] * face[0] + face[1] * face[1] + face[2] * face[2]);
                for (int v = 0; v < 3; ++v)
                {
                    const auto local = object.indexCount ? scene.indices[object.firstIndex + index + v] : index + v;
                    normals[v] = point(scene.vertices[object.firstVertex + local].normal);
                    for (int axis = 0; axis < 3; ++axis)
                        smooth = smooth || !(std::abs(normals[v][axis] - face[axis] / faceLength) <= 1e-5f);
                }
                if (smooth)
                {
                    // Index 0 means no block, so a block never starts there.
                    if (candidate.texels.empty())
                        candidate.texels.push_back({});
                    p.normals = candidate.texels.size();
                    for (const auto& normal : normals)
                        candidate.texels.push_back({normal[0], normal[1], normal[2], 0.0f});
                }
            }
            triangles.push_back(p);
        }
        Mesh mesh;
        for (const auto& p : triangles)
            mesh.bounds.add(p.bounds);
        mesh.triangles = triangles.size();
        mesh.nodes = treeNodes(triangles.size(), 4);
        mesh.root = tree(candidate.texels, triangles, 0, triangles.size(), true, mode == AdmissionMode::allSurfaces);
        const auto id = int(candidate.meshes.size());
        candidate.meshes.push_back(mesh);
        ranges[key] = id;
        candidate.objectMesh[o] = id;
    }
    if (!payloadIndexable(candidate.texels.size(), error))
        return false;
    output = std::move(candidate);
    error.clear();
    return true;
}
inline std::vector<Instance> noteInstances(const std::vector<Matrix>& objects,
                                           const NativeNoteInstanceBatch& notes)
{
    std::vector<Instance> instances;
    instances.reserve(objects.size() * notes.count);
    for (std::size_t o = 0; o < objects.size(); ++o)
        for (std::size_t n = 0; n < notes.count; ++n)
        {
            Matrix matrix = objects[o];
            const auto translation = transform(
                objects[o], {notes.transforms[n][0], notes.transforms[n][1], notes.transforms[n][2]});
            for (int c = 0; c < 3; ++c)
                for (int r = 0; r < 3; ++r)
                    matrix[c * 4 + r] *= notes.meshScale;
            for (int r = 0; r < 3; ++r)
                matrix[12 + r] = translation[r];
            instances.push_back({matrix, o});
        }
    return instances;
}
// The scene's instanced object becomes one query instance per transform. Each
// shares the object's mesh and index, so shadows match the instanced draw.
inline std::vector<Instance> sceneInstances(const HarmonicMIDI::grid::Visual3DScene& scene,
                                            const std::vector<Matrix>& objects,
                                            const NativeNoteInstanceBatch& notes)
{
    auto expanded = noteInstances(objects, notes);
    if (!scene.instancedObject.isValid())
        return expanded;
    std::vector<Instance> instances;
    instances.reserve(expanded.size() + scene.objectInstanceCount);
    const auto count = std::min(scene.objectInstanceCount, scene.objectInstances.size());
    for (const auto& entry : expanded)
    {
        if (entry.object >= scene.objectCount || scene.objects[entry.object].id != scene.instancedObject)
        {
            instances.push_back(entry);
            continue;
        }
        for (std::size_t i = 0; i < count; ++i)
            instances.push_back({nativeObjectInstanceMatrix(entry.world, scene.objectInstances[i]), entry.object});
    }
    return instances;
}
inline bool frame(const Prepared& prepared, const std::vector<Instance>& matrices, Frame& output,
                  std::string& error)
{
    Frame candidate;
    candidate.texels = prepared.texels;
    const auto metadata = candidate.texels.size();
    if (prepared.mode == AdmissionMode::allSurfaces)
        candidate.texels.insert(candidate.texels.end(), prepared.identities.begin(), prepared.identities.end());
    std::vector<Primitive> instances;
    std::size_t queryWork = 0;
    for (std::size_t o = 0; o < matrices.size(); ++o)
    {
        if (matrices[o].object >= prepared.objectCount)
        {
            error = "Opaque visibility instance object is invalid";
            return false;
        }
        const auto meshIndex = prepared.objectMesh[matrices[o].object];
        if (meshIndex < 0)
            continue;
        if (std::size_t(meshIndex) >= prepared.meshes.size())
        {
            error = "Scene query mesh index is invalid";
            return false;
        }
        const auto& mesh = prepared.meshes[std::size_t(meshIndex)];
        queryWork += mesh.nodes + mesh.triangles;
        if (mesh.root + 1 >= prepared.texels.size())
        {
            error = "Scene query mesh root is invalid";
            return false;
        }
        const auto& world = matrices[o].world;
        Matrix local;
        if (prepared.mode == AdmissionMode::allSurfaces &&
            (!std::all_of(world.begin(), world.end(), [](float v) { return std::isfinite(v); }) ||
             world[3] != 0 || world[7] != 0 || world[11] != 0 || world[15] != 1))
        { error = "Scene closest-hit transform is not finite affine"; return false; }
        if (!inverse(world, local))
        {
            error = "Opaque visibility transform is singular or non-finite";
            return false;
        }
        Primitive p;
        p.record = candidate.texels.size();
        for (int r = 0; r < 3; ++r)
            candidate.texels.push_back({local[r], local[4 + r], local[8 + r], local[12 + r]});
        candidate.texels.push_back({float(mesh.root), float(prepared.texels[mesh.root][3]),
            prepared.mode == AdmissionMode::allSurfaces ? float(metadata + matrices[o].object) : 0.0f,
            prepared.mode == AdmissionMode::allSurfaces ? float(o) : 0.0f});
        for (int corner = 0; corner < 8; ++corner)
        {
            Point v;
            for (int a = 0; a < 3; ++a)
                v[a] = (corner & (1 << a)) ? mesh.bounds.high[a] : mesh.bounds.low[a];
            const auto q = transform(world, v);
            for (float c : q)
                if (!std::isfinite(c))
                {
                    error = "Opaque visibility world bounds are non-finite";
                    return false;
                }
            p.bounds.add(q);
        }
        instances.push_back(p);
    }
    if (prepared.mode == AdmissionMode::allSurfaces)
        candidate.control = {float(candidate.texels.size()), float(candidate.texels.size()), float(kTextureWidth), 2};
    if (!instances.empty())
    {
        if (prepared.mode == AdmissionMode::allSurfaces)
        {
            queryWork = std::min(queryWork + treeNodes(instances.size(), 1), kMaximumClosestHitWork);
            // Count every instance even when its BLAS range is reused. Split the
            // integer into exact float words; the public control remains unchanged.
            candidate.texels.push_back({float(queryWork & 65535u), float(queryWork >> 16),
                                        float(kClosestHitStackCapacity), 3.0f});
        }
        const auto root = tree(candidate.texels, instances, 0, instances.size(), false);
        candidate.control = {float(root), float(candidate.texels.size()), float(kTextureWidth), prepared.mode == AdmissionMode::allSurfaces ? 2.0f : 1.0f};
    }
    if (!payloadIndexable(candidate.texels.size(), error))
        return false;
    candidate.height =
        std::max(std::size_t(1), (candidate.texels.size() + kTextureWidth - 1) / kTextureWidth);
    candidate.texels.resize(kTextureWidth * candidate.height);
    output = std::move(candidate);
    error.clear();
    return true;
}

// The instance tree of tree(..., triangles=false), written into a vector that
// holds only the texels from base on. Every stored index is absolute.
inline std::size_t instanceTree(std::vector<Texel>& texels, std::size_t base, std::vector<Primitive>& primitives,
                                std::size_t begin, std::size_t end)
{
    Bounds bounds;
    for (auto i = begin; i < end; ++i)
        bounds.add(primitives[i].bounds);
    const auto node = texels.size();
    texels.resize(node + 2);
    std::size_t leaf = 0;
    if (end - begin <= 1u)
        leaf = primitives[begin].record + 1;
    else
    {
        int axis = 0;
        for (int a = 1; a < 3; ++a)
            if (bounds.high[a] - bounds.low[a] > bounds.high[axis] - bounds.low[axis])
                axis = a;
        const auto middle = begin + (end - begin) / 2;
        std::nth_element(
            primitives.begin() + begin, primitives.begin() + middle, primitives.begin() + end,
            [axis](const auto& a, const auto& b)
            { return a.bounds.low[axis] + a.bounds.high[axis] < b.bounds.low[axis] + b.bounds.high[axis]; });
        instanceTree(texels, base, primitives, begin, middle);
        instanceTree(texels, base, primitives, middle, end);
    }
    texels[node] = {bounds.low[0], bounds.low[1], bounds.low[2], float(base + texels.size())};
    texels[node + 1] = {bounds.high[0], bounds.high[1], bounds.high[2], float(leaf)};
    return base + node;
}

// Preview resident query. frame() copies the static BLAS texels into every
// frame. A resident texture keeps them on the GPU, so the tail holds only what
// frame() appends after them: identities, instance records, the work allowance
// and the TLAS. texels[i] is frame().texels[begin + i]. The world matrix of each
// emitted instance record is kept, in record order, for a GPU TLAS refit.
struct FrameTail
{
    std::size_t begin = 0;
    std::size_t instanceBegin = 0;
    std::vector<Texel> texels;
    std::vector<Matrix> worlds;
    std::array<float, 4> control{};
    std::size_t height = 1;
};
inline bool frameTail(const Prepared& prepared, const std::vector<Instance>& matrices, FrameTail& output,
                      std::string& error)
{
    FrameTail candidate;
    const auto base = prepared.texels.size();
    candidate.begin = base;
    const auto metadata = base;
    if (prepared.mode == AdmissionMode::allSurfaces)
        candidate.texels.insert(candidate.texels.end(), prepared.identities.begin(), prepared.identities.end());
    candidate.instanceBegin = base + candidate.texels.size();
    std::vector<Primitive> instances;
    std::size_t queryWork = 0;
    for (std::size_t o = 0; o < matrices.size(); ++o)
    {
        if (matrices[o].object >= prepared.objectCount)
        {
            error = "Opaque visibility instance object is invalid";
            return false;
        }
        const auto meshIndex = prepared.objectMesh[matrices[o].object];
        if (meshIndex < 0)
            continue;
        if (std::size_t(meshIndex) >= prepared.meshes.size())
        {
            error = "Scene query mesh index is invalid";
            return false;
        }
        const auto& mesh = prepared.meshes[std::size_t(meshIndex)];
        queryWork += mesh.nodes + mesh.triangles;
        if (mesh.root + 1 >= prepared.texels.size())
        {
            error = "Scene query mesh root is invalid";
            return false;
        }
        const auto& world = matrices[o].world;
        Matrix local;
        if (prepared.mode == AdmissionMode::allSurfaces &&
            (!std::all_of(world.begin(), world.end(), [](float v) { return std::isfinite(v); }) ||
             world[3] != 0 || world[7] != 0 || world[11] != 0 || world[15] != 1))
        { error = "Scene closest-hit transform is not finite affine"; return false; }
        if (!inverse(world, local))
        {
            error = "Opaque visibility transform is singular or non-finite";
            return false;
        }
        Primitive p;
        p.record = base + candidate.texels.size();
        for (int r = 0; r < 3; ++r)
            candidate.texels.push_back({local[r], local[4 + r], local[8 + r], local[12 + r]});
        candidate.texels.push_back({float(mesh.root), float(prepared.texels[mesh.root][3]),
            prepared.mode == AdmissionMode::allSurfaces ? float(metadata + matrices[o].object) : 0.0f,
            prepared.mode == AdmissionMode::allSurfaces ? float(o) : 0.0f});
        candidate.worlds.push_back(world);
        for (int corner = 0; corner < 8; ++corner)
        {
            Point v;
            for (int a = 0; a < 3; ++a)
                v[a] = (corner & (1 << a)) ? mesh.bounds.high[a] : mesh.bounds.low[a];
            const auto q = transform(world, v);
            for (float c : q)
                if (!std::isfinite(c))
                {
                    error = "Opaque visibility world bounds are non-finite";
                    return false;
                }
            p.bounds.add(q);
        }
        instances.push_back(p);
    }
    if (prepared.mode == AdmissionMode::allSurfaces)
        candidate.control = {float(base + candidate.texels.size()), float(base + candidate.texels.size()),
                             float(kTextureWidth), 2};
    if (!instances.empty())
    {
        if (prepared.mode == AdmissionMode::allSurfaces)
        {
            queryWork = std::min(queryWork + treeNodes(instances.size(), 1), kMaximumClosestHitWork);
            candidate.texels.push_back({float(queryWork & 65535u), float(queryWork >> 16),
                                        float(kClosestHitStackCapacity), 3.0f});
        }
        const auto root = instanceTree(candidate.texels, base, instances, 0, instances.size());
        candidate.control = {float(root), float(base + candidate.texels.size()), float(kTextureWidth),
                             prepared.mode == AdmissionMode::allSurfaces ? 2.0f : 1.0f};
    }
    const auto total = base + candidate.texels.size();
    if (!payloadIndexable(total, error))
        return false;
    candidate.height = std::max(std::size_t(1), (total + kTextureWidth - 1) / kTextureWidth);
    output = std::move(candidate);
    error.clear();
    return true;
}

// Float companion for the production query. The independent test oracle uses
// double world triangles and does not consume this payload or intersection.
struct ClosestHit
{
    int status = -1;
    float distance = 0;
    Point position{}, outwardNormal{}, shadingNormal{};
    std::uint32_t objectId = 0, materialId = 0, instanceId = 0, triangleId = 0;
    bool entering = false;
    std::uint32_t work = 0; // One unit per popped node or triangle intersection.
};
// preferred, when nonzero, is the shader's preferred object: its hits win over
// any other object's hit in [minimum, maximum]. 0 is the ordinary closest hit.
// avoid, when nonzero, is the shader's avoided object: its instances are
// skipped before their mesh walk and never hit.
inline ClosestHit closestHitCpu(const Prepared& prepared, const Frame& frame, Point origin,
                                Point rayDirection, float minimum, float maximum, std::uint32_t preferred = 0,
                                std::uint32_t avoid = 0)
{
    ClosestHit hit; hit.distance = maximum;
    float reach = maximum;
    auto finite = [](const Point& p) { return std::all_of(p.begin(), p.end(), [](float v) { return std::isfinite(v); }); };
    auto xyz = [](const Texel& p) -> Point { return {p[0], p[1], p[2]}; };
    auto dot = [](Point a, Point b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; };
    auto sub = [](Point a, Point b) -> Point { return {a[0]-b[0],a[1]-b[1],a[2]-b[2]}; };
    auto cross = [](Point a, Point b) -> Point { return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}; };
    auto index = [](float v, std::size_t end) { return std::isfinite(v) && v>=0 && v<float(end) && std::floor(v)==v; };
    const auto& control = frame.control;
    const float rayLength = std::sqrt(dot(rayDirection, rayDirection));
    if (prepared.mode != AdmissionMode::allSurfaces || !finite(origin) || !finite(rayDirection) ||
        !std::isfinite(rayLength) || rayLength<=0 || !std::isfinite(minimum) || !std::isfinite(maximum) ||
        minimum<0 || maximum<minimum || control[3]!=2 || control[2]!=1024 ||
        !index(control[1], kExactFloatIndexRange+1) || control[1]>frame.texels.size() ||
        control[1]>double(frame.height)*kTextureWidth ||
        !index(control[0], std::size_t(control[1])+1)) return hit;
    for (auto& c : rayDirection) c /= rayLength;
    const auto root = std::size_t(control[0]), end = std::size_t(control[1]);
    auto refused = [&](int status) { ClosestHit out; out.status=status; out.distance=maximum; out.work=hit.work; return out; };
    hit.status=0;
    if (root==end) return hit;
    if (root<1) return refused(-1);
    const auto& allowance=frame.texels[root-1];
    if (!index(allowance[0],65536) || !index(allowance[1],65536) ||
        !index(allowance[2],kClosestHitStackCapacity+1) || allowance[2]<1 || allowance[3]!=3) return refused(-1);
    const auto budget=std::uint32_t(allowance[0])+(std::uint32_t(allowance[1])<<16);
    if (budget==0) return refused(-1);
    const auto capacity=std::size_t(allowance[2]);
    auto range = [&](const Texel& low, const Texel& high, std::size_t node, std::size_t limit) {
        return finite(xyz(low)) && finite(xyz(high)) && low[0]<=high[0] && low[1]<=high[1] && low[2]<=high[2] &&
            index(low[3],limit+1) && low[3]>float(node+1) && index(high[3],limit+1);
    };
    auto bounds = [](Point origin, Point direction, const Texel& low, const Texel& high, float limit, float& nearT) {
        nearT=0; float farT=limit;
        for (int axis=0; axis<3; ++axis) {
            if (direction[axis]==0) { if (origin[axis]<low[axis] || origin[axis]>high[axis]) return false; }
            else {
                const float a=(low[axis]-origin[axis])/direction[axis], b=(high[axis]-origin[axis])/direction[axis];
                nearT=std::max(nearT,std::min(a,b)); farT=std::min(farT,std::max(a,b));
                if (nearT>farT) return false;
            }
        }
        return true;
    };
    // TLAS and BLAS each have their own fixed stack, sharing one work counter.
    auto walk = [&](std::size_t start, std::size_t limit, Point localOrigin, Point localDirection, const auto& leaf) {
        std::array<std::size_t,kClosestHitStackCapacity> nodes{}; std::size_t pending=1; nodes[0]=start;
        while (pending>0) {
            if (hit.work>=budget) return -2;
            ++hit.work; const auto node=nodes[--pending];
            if (node<start || node+1>=limit) return -1;
            const auto& low=frame.texels[node]; const auto& high=frame.texels[node+1];
            if (!range(low,high,node,limit)) return -1;
            const auto escape=std::size_t(low[3]); float nearT=0;
            if (!bounds(localOrigin,localDirection,low,high,reach,nearT)) continue;
            if (high[3]>0) { const auto status=leaf(node,escape,std::size_t(high[3])-1); if (status<0) return status; continue; }
            const auto left=node+2;
            if (left+1>=escape) return -1;
            const auto& ll=frame.texels[left]; const auto& lh=frame.texels[left+1];
            if (!range(ll,lh,left,escape)) return -1;
            const auto right=std::size_t(ll[3]);
            if (right+1>=escape) return -1;
            const auto& rl=frame.texels[right]; const auto& rh=frame.texels[right+1];
            if (!range(rl,rh,right,escape) || rl[3]!=float(escape)) return -1;
            float leftNear=0,rightNear=0;
            const bool useLeft=bounds(localOrigin,localDirection,ll,lh,reach,leftNear);
            const bool useRight=bounds(localOrigin,localDirection,rl,rh,reach,rightNear);
            if (pending+std::size_t(useLeft)+std::size_t(useRight)>capacity) return -2;
            if (useLeft && useRight) {
                nodes[pending++]=leftNear<=rightNear?right:left; nodes[pending++]=leftNear<=rightNear?left:right;
            } else if (useLeft) nodes[pending++]=left;
            else if (useRight) nodes[pending++]=right;
        }
        return 0;
    };
    std::size_t bestObject=kExactFloatIndexRange;
    const int status=walk(root,end,origin,rayDirection,[&](std::size_t node,std::size_t,std::size_t leaf) {
        if (leaf+3>=end) return -1;
        const auto& r0=frame.texels[leaf]; const auto& r1=frame.texels[leaf+1]; const auto& r2=frame.texels[leaf+2];
        const auto& mesh=frame.texels[leaf+3];
        if (!finite(xyz(r0)) || !std::isfinite(r0[3]) || !finite(xyz(r1)) || !std::isfinite(r1[3]) ||
            !finite(xyz(r2)) || !std::isfinite(r2[3]) || !index(mesh[0],end) || !index(mesh[1],node+1) ||
            mesh[1]<=mesh[0] || !index(mesh[2],end) || !index(mesh[3],kExactFloatIndexRange)) return -1;
        const auto& ids=frame.texels[std::size_t(mesh[2])];
        for (float value : ids) if (!index(value,65536)) return -1;
        const auto objectId=std::uint32_t(ids[0])+(std::uint32_t(ids[1])<<16);
        const auto materialId=std::uint32_t(ids[2])+(std::uint32_t(ids[3])<<16);
        const auto instanceId=std::uint32_t(mesh[3]);
        if (!objectId || !materialId) return -1;
        if (avoid!=0 && objectId==avoid) return 0;
        const Point localOrigin{dot(xyz(r0),origin)+r0[3],dot(xyz(r1),origin)+r1[3],dot(xyz(r2),origin)+r2[3]};
        const Point localDirection{dot(xyz(r0),rayDirection),dot(xyz(r1),rayDirection),dot(xyz(r2),rayDirection)};
        if (!finite(localOrigin) || !finite(localDirection) || std::sqrt(dot(localDirection,localDirection))==0) return -1;
        return walk(std::size_t(mesh[0]),std::size_t(mesh[1]),localOrigin,localDirection,
            [&](std::size_t at,std::size_t next,std::size_t tri) {
                if (tri<at+2 || tri>=next) return -1;
                const float count=frame.texels[tri][3];
                if (!index(count,5) || count<1 || tri+std::size_t(count)*3>next) return -1;
                for (std::size_t t=0;t<std::size_t(count);++t) {
                    if (hit.work>=budget) return -2;
                    ++hit.work; const auto offset=tri+t*3;
                    const auto a=xyz(frame.texels[offset]),b=xyz(frame.texels[offset+1]),c=xyz(frame.texels[offset+2]);
                    const auto triangle=frame.texels[offset+1][3];
                    if (!finite(a) || !finite(b) || !finite(c) || !index(triangle,kExactFloatIndexRange)) return -1;
                    const auto e1=sub(b,a),e2=sub(c,a),p=cross(localDirection,e2),normal=cross(e1,e2);
                    const float det=dot(e1,p);
                    if (!finite(e1) || !finite(e2) || !finite(p) || !finite(normal) || !std::isfinite(det)) return -1;
                    if (det==0) continue;
                    const auto delta=sub(localOrigin,a); const float u=dot(delta,p)/det;
                    if (!std::isfinite(u)) return -1;
                    if (u<0 || u>1) continue;
                    const auto q=cross(delta,e1); const float v=dot(localDirection,q)/det;
                    if (!std::isfinite(v)) return -1;
                    if (v<0 || u+v>1) continue;
                    const float distance=dot(e2,q)/det;
                    if (!std::isfinite(distance)) return -1;
                    if (distance<minimum || distance>reach) continue;
                    const auto objectIndex=std::size_t(mesh[2]); const auto triangleId=std::uint32_t(triangle);
                    const bool candidatePreferred=preferred!=0 && objectId==preferred;
                    const bool bestPreferred=preferred!=0 && hit.status==1 && hit.objectId==preferred;
                    const bool wins=hit.status==0 || (candidatePreferred && !bestPreferred) ||
                        (candidatePreferred==bestPreferred && (distance<hit.distance || (distance==hit.distance &&
                        (objectIndex<bestObject || (objectIndex==bestObject && (instanceId<hit.instanceId ||
                        (instanceId==hit.instanceId && triangleId<hit.triangleId)))))));
                    if (!wins) continue;
                    Point worldNormal{r0[0]*normal[0]+r1[0]*normal[1]+r2[0]*normal[2],
                        r0[1]*normal[0]+r1[1]*normal[1]+r2[1]*normal[2],r0[2]*normal[0]+r1[2]*normal[1]+r2[2]*normal[2]};
                    const float length=std::sqrt(dot(worldNormal,worldNormal));
                    if (!std::isfinite(length) || length<=0) return -1;
                    for (int axis=0;axis<3;++axis) { worldNormal[axis]/=length; hit.position[axis]=origin[axis]+rayDirection[axis]*distance; }
                    hit.status=1; hit.distance=distance; hit.outwardNormal=worldNormal; hit.entering=dot(rayDirection,worldNormal)<0;
                    hit.shadingNormal=worldNormal;
                    const auto block=frame.texels[offset+2][3];
                    if (!index(block,kExactFloatIndexRange)) return -1;
                    if (block>0) {
                        if (std::size_t(block)+2>=frame.texels.size()) return -1;
                        const auto n0=xyz(frame.texels[std::size_t(block)]),n1=xyz(frame.texels[std::size_t(block)+1]),n2=xyz(frame.texels[std::size_t(block)+2]);
                        const Point blended{n0[0]*(1-u-v)+n1[0]*u+n2[0]*v,n0[1]*(1-u-v)+n1[1]*u+n2[1]*v,n0[2]*(1-u-v)+n1[2]*u+n2[2]*v};
                        Point world{r0[0]*blended[0]+r1[0]*blended[1]+r2[0]*blended[2],
                            r0[1]*blended[0]+r1[1]*blended[1]+r2[1]*blended[2],r0[2]*blended[0]+r1[2]*blended[1]+r2[2]*blended[2]};
                        const float worldLength=std::sqrt(dot(world,world));
                        if (std::isfinite(worldLength) && worldLength>0) { for (auto& component:world) component/=worldLength; hit.shadingNormal=world; }
                    }
                    hit.objectId=objectId; hit.materialId=materialId; hit.instanceId=instanceId; hit.triangleId=triangleId; bestObject=objectIndex;
                    if (preferred==0 || candidatePreferred) reach=distance;
                }
                return 0;
            });
    });
    return status<0?refused(status):hit;
}

inline bool vertexDeformation(const NativeFixtureSurfaceMaterialProgram* program);
inline bool prepareScene(const HarmonicMIDI::grid::Visual3DScene& scene,
                         const NativeFixtureSurfaceMaterialProgram* program,
                         const std::vector<videowire::geometry::InstanceAppearance>& appearances,
                         Prepared& output, std::string& error, AdmissionMode mode = AdmissionMode::opaqueOnly,
                         bool sceneReceiverTransport = false,
                         bool sceneReceiverConsumer = false,
                         NativeFixtureDiffractionConsumer consumer=NativeFixtureDiffractionConsumer::FiniteLobe,
                         bool glassReceiverNormals=false)
{
    using namespace HarmonicMIDI::grid;
    const bool discrete=consumer==NativeFixtureDiffractionConsumer::SmoothGlassDiscreteOrders;
    if((consumer!=NativeFixtureDiffractionConsumer::FiniteLobe&&!discrete)
        ||(discrete&&(sceneReceiverTransport||mode!=AdmissionMode::allSurfaces||!program
            ||program->kind!=NativeFixtureMaterialKind::DiffractionReflective
            ||!validNativeFixtureMaterialPrograms(*program,scene,program->backend,true,consumer))))
    {error="Smooth glass grating query requires exact private discrete material owners";return false;}
    if (sceneReceiverTransport && (mode != AdmissionMode::allSurfaces || !program
        || !program->diffractionLightingAdmission
        || program->diffractionLightingAdmission->description().version != 3
        || !validNativeFixtureMaterialPrograms(*program, scene, program->backend, true)))
    { error = "Scene receiver query requires the exact admitted v3 grating"; return false; }
    if (!appearances.empty() && appearances.size() != scene.objectCount)
    {
        error = "Opaque visibility appearance count is malformed";
        return false;
    }
    std::vector<char> opaque(scene.objectCount, 0);
    for (std::size_t o = 0; o < scene.objectCount; ++o)
    {
        const auto& object = scene.objects[o];
        if (object.firstVertex > scene.vertexCount ||
            object.vertexCount > scene.vertexCount - object.firstVertex)
        {
            error = "Opaque visibility vertex range is malformed";
            return false;
        }
        const auto* material =
            visual3d_detail::findById(scene.materials, scene.materialCount, object.material);
        const auto* surface = nativeSurfaceForObject(program, object.id);
        if (mode == AdmissionMode::allSurfaces && object.vertexCount > 0)
        {
            for (std::size_t previous = 0; previous < o; ++previous)
                if (scene.objects[previous].id == object.id)
                { error = "Scene closest-hit duplicate object identity is invalid"; return false; }
            if (!material || !object.id.isValid() || !object.material.isValid())
            { error = "Scene closest-hit object or material identity is invalid"; return false; }
            const bool grating = (sceneReceiverTransport||discrete) && surface == program
                && surface->kind == NativeFixtureMaterialKind::DiffractionReflective;
            if (sceneReceiverTransport && (material->alphaMode != SceneAlphaMode::Opaque
                || material->opacity != 1.0f
                || (surface && !grating && (surface->parameters.normalOpacity[3] != 1.0f
                    || surface->parameters.transmissionIorClearcoat[0] != 0.0f
                    || surface->parameters.identifiers[1] != static_cast<std::uint32_t>(surfacematerial::TransportMode::ThinSurface)
                    || surface->parameters.identifiers[0] != object.material.value))
                || (!appearances.empty() && appearances[o].color[3] != 1.0f)))
            { error = "Scene receiver query requires complete opaque coverage"; return false; }
            if (vertexDeformation(surface) || vertexDeformation(program))
            { error = "Scene closest-hit unevaluated vertex deformation is unsupported"; return false; }
            if (material->alphaMode != SceneAlphaMode::Opaque && material->alphaMode != SceneAlphaMode::Mask &&
                material->alphaMode != SceneAlphaMode::Blend)
            { error = "Scene closest-hit material alpha mode is invalid"; return false; }
            if (material->alphaMode == SceneAlphaMode::Mask ||
                material->baseColorTexture.isValid() ||
                (surface && (surface->baseColorSource == NativeFixtureSurfaceMaterialProgram::BaseColorSource::ImportedSrgbTexture ||
                             surface->baseColorSource == NativeFixtureSurfaceMaterialProgram::BaseColorSource::GraphFrameSrgbTexture)))
            { error = "Scene closest-hit masked or textured-alpha coverage is unsupported"; return false; }
            if (!std::isfinite(material->opacity) || material->opacity <= 0.0f ||
                (surface && !grating && (!validNativeFixtureSurfaceParameters(surface->parameters) || surface->parameters.normalOpacity[3] <= 0.0f)) ||
                (!appearances.empty() && (!std::isfinite(appearances[o].color[3]) || appearances[o].color[3] <= 0.0f)))
            { error = "Scene closest-hit zero or invalid coverage is unsupported"; return false; }
            for (std::size_t v = 0; v < object.vertexCount; ++v)
                if (!std::isfinite(scene.vertices[object.firstVertex + v].color[3]) ||
                    scene.vertices[object.firstVertex + v].color[3] <= 0.0f
                    || (sceneReceiverTransport && scene.vertices[object.firstVertex + v].color[3] != 1.0f))
                { error = "Scene closest-hit vertex coverage is unsupported"; return false; }
            opaque[o] = true;
            continue;
        }
        if (surface && surface->kind == NativeFixtureMaterialKind::DiffractionReflective)
        {
            if (!validNativeFixtureDiffractionProgram(*surface, surface->backend, object.id, sceneReceiverConsumer))
            { error = "Opaque visibility diffraction identity is invalid"; return false; }
            opaque[o] = surface->baseColorSource == NativeFixtureSurfaceMaterialProgram::BaseColorSource::ConstantLinear;
            continue;
        }
        opaque[o] = material && material->alphaMode == SceneAlphaMode::Opaque &&
                    !(surface && surface->parameters.identifiers[1]
                        == static_cast<std::uint32_t>(surfacematerial::TransportMode::ThinInterface)) &&
                    (surface ? !nativeSurfaceRequiresBlend(surface) : material->opacity >= 1.0f) &&
                    (appearances.empty() || appearances[o].color[3] >= 1.0f);
        for (std::size_t v = 0; opaque[o] && v < object.vertexCount; ++v)
            opaque[o] = scene.vertices[object.firstVertex + v].color[3] >= 1.0f;
        // An object hidden from shadow rays casts no raster shadow.
        if ((object.visibility & sceneRayVisibility::Shadow) == 0) opaque[o] = false;
    }
    if (!glassReceiverNormals)
        return prepare(scene, opaque, output, error, mode);
    // Only a receiver reads a shading normal. Water and glass refract at their own
    // evaluated normals, and a 1024 by 1024 sea must not pay for a block per face.
    std::vector<char> receivers(scene.objectCount, 0);
    for (std::size_t o = 0; o < scene.objectCount; ++o)
    {
        const auto* surface = nativeSurfaceForObject(program, scene.objects[o].id);
        receivers[o] = !surface || (surface->kind == NativeFixtureMaterialKind::SurfacePbr
            && surface->parameters.identifiers[1] != static_cast<std::uint32_t>(surfacematerial::TransportMode::ThinInterface)
            && surface->parameters.identifiers[1] != static_cast<std::uint32_t>(surfacematerial::TransportMode::ClosedVolume));
    }
    return prepare(scene, opaque, output, error, mode, &receivers);
}
inline bool vertexDeformation(const NativeFixtureSurfaceMaterialProgram* program)
{
    if (!program || !program->vertexProgram || !program->vertexProgram->hasVertexProgram())
        return false;
    const auto& vertex = *program->vertexProgram;
    for (const auto& instruction : vertex.vertexInstructions())
        if (instruction.resultSlot == vertex.vertexOutputSlot())
            return instruction.operation != videowire::VertexModifierOperation::importedPosition;
    return true;
}

// Both native backends compile these same traversal and intersection operations.
// Opaque any-hit retains its escape-link traversal. Closest-hit visits nearer
// child bounds first with fixed stacks and an actual-topology work allowance.
// allSurfaces is a geometric interface query. It does not evaluate a BSDF,
// require closed topology, or certify that an entering hit encloses a volume.
// Outward normals preserve authored local winding through inverse transpose,
// including mirrored transforms. Callers own volume/topology admission.
// opaqueVisibility takes the receiver's outward normal, not a screen-derivative
// one: raster callers pass the interpolated vertex normal, glass callers the hit
// normal. A receiver whose normal is not strictly toward the light is unlit
// before any ray. The origin moves 1e-4 * max(1, max|p|) along that normal.
// Float32 epsilon is 1.19e-7, so this is about 840 ulps of the largest
// coordinate. It covers varying interpolation, the world-to-local TLAS
// transform and the Moller-Trumbore distance error, which all scale with the
// coordinate (and triangle) magnitude. At metre scale it is 0.1-0.3 mm, well
// below a 4K pixel, so contact shadows do not visibly detach.
inline constexpr char shader[] = R"shader(
SV4 opaqueRead(int index ST_ARGS) { return ST_READ; }
bool opaqueBounds(SV3 origin,SV3 direction,SV3 low,SV3 high,float limit) {
    float nearT=0.0,farT=limit;
    #pragma optionNV(unroll none)
    for(int axis=0;axis<3;++axis) {
        if(abs(direction[axis])<1.0e-30) { if(origin[axis]<low[axis]||origin[axis]>high[axis]) return false; }
        else { float a=(low[axis]-origin[axis])/direction[axis];float b=(high[axis]-origin[axis])/direction[axis];
            nearT=max(nearT,min(a,b));farT=min(farT,max(a,b)); if(nearT>farT) return false; }
    }
    return true;
}
bool opaqueTriangle(SV3 origin,SV3 direction,SV3 a,SV3 b,SV3 c,float limit) {
    SV3 e1=b-a,e2=c-a,p=cross(direction,e2);float det=dot(e1,p);
    if(abs(det)<=1.0e-7*length(e1)*length(e2)*length(direction)) return false;
    SV3 t=origin-a;float u=dot(t,p)/det;if(u<0.0||u>1.0) return false;
    SV3 q=cross(t,e1);float v=dot(direction,q)/det;if(v<0.0||u+v>1.0) return false;
    float distance=dot(e2,q)/det;return distance>0.0&&distance<limit;
}
bool opaqueMesh(SV3 origin,SV3 direction,float limit,int root,int end ST_ARGS) {
    int node=root;
    #pragma optionNV(unroll none)
    for(int visit=0;visit<OPAQUE_WALK_VISITS;++visit) {
        if(node>=end) return false;
        SV4 low=opaqueRead(node ST_PASS),high=opaqueRead(node+1 ST_PASS);
        int escape=int(low.w),leaf=int(high.w)-1;
        if(opaqueBounds(origin,direction,low.xyz,high.xyz,limit)) {
            if(leaf<0) { node+=2;continue; }
            int count=int(opaqueRead(leaf ST_PASS).w);
            #pragma optionNV(unroll none)
            for(int triangle=0;triangle<4;++triangle) {
                if(triangle>=count) break;int at=leaf+triangle*3;
                if(opaqueTriangle(origin,direction,opaqueRead(at ST_PASS).xyz,
                    opaqueRead(at+1 ST_PASS).xyz,opaqueRead(at+2 ST_PASS).xyz,limit)) return true;
            }
        }
        node=escape;
    }
    return false;
}
float opaqueVisibility(SV3 position,SV3 geometricNormal,SV3 toLight,float limit,SV4 control ST_ARGS) {
    if(control.w<0.5) return 1.0;
    SV3 direction=normalize(toLight);
    float bias=max(1.0e-4,1.0e-4*max(max(abs(position.x),abs(position.y)),abs(position.z)));
    SV3 n=normalize(geometricNormal);if(!(dot(n,direction)>0.0)) return 0.0;
    SV3 origin=position+n*bias;
    limit=max(0.0,limit-bias);
    int node=int(control.x),end=int(control.y);
    #pragma optionNV(unroll none)
    for(int visit=0;visit<OPAQUE_WALK_VISITS;++visit) {
        if(node>=end) return 1.0;
        SV4 low=opaqueRead(node ST_PASS),high=opaqueRead(node+1 ST_PASS);
        int escape=int(low.w),leaf=int(high.w)-1;
        if(opaqueBounds(origin,direction,low.xyz,high.xyz,limit)) {
            if(leaf<0) {node+=2;continue;}
            SV4 r0=opaqueRead(leaf ST_PASS),r1=opaqueRead(leaf+1 ST_PASS),r2=opaqueRead(leaf+2 ST_PASS);
            SV4 mesh=opaqueRead(leaf+3 ST_PASS);
            SV3 localOrigin=SV3(dot(r0.xyz,origin)+r0.w,dot(r1.xyz,origin)+r1.w,dot(r2.xyz,origin)+r2.w);
            SV3 localDirection=SV3(dot(r0.xyz,direction),dot(r1.xyz,direction),dot(r2.xyz,direction));
            if(opaqueMesh(localOrigin,localDirection,limit,int(mesh.x),int(mesh.y) ST_PASS)) return 0.0;
        }
        node=escape;
    }
    return 1.0;
}
// status -2 is exhausted, -1 is invalid/unsupported, 0 is a complete miss, 1 is a complete hit.
// Bounds are inclusive finite world distances along the normalized world ray.
// Equal distances use scene object order, input instance order, then triangle order.
struct SceneClosestHit {
    int status;
    float distance;
    SV3 position;
    SV3 outwardNormal;
    // The interpolated vertex normal on a face that stores a normal block, otherwise
    // exactly outwardNormal. Interface crossings keep using outwardNormal.
    SV3 shadingNormal;
    uint objectId;
    uint materialId;
    uint instanceId;
    uint triangleId;
    bool entering;
};
bool sceneFinite(float v) { return !isnan(v)&&!isinf(v); }
bool sceneFinite3(SV3 v) { return sceneFinite(v.x)&&sceneFinite(v.y)&&sceneFinite(v.z); }
bool sceneIndex(float v,int end) { return sceneFinite(v)&&v>=0.0&&v<float(end)&&floor(v)==v; }
bool sceneRange(SV4 low,SV4 high,int node,int end) {
    return sceneFinite3(low.xyz)&&sceneFinite3(high.xyz)&&
        low.x<=high.x&&low.y<=high.y&&low.z<=high.z&&
        sceneFinite(low.w)&&floor(low.w)==low.w&&low.w>float(node+1)&&low.w<=float(end)&&
        sceneFinite(high.w)&&floor(high.w)==high.w&&high.w>=0.0&&high.w<=float(end);
}
SceneClosestHit sceneEmptyHit(int status,float limit) {
    SceneClosestHit hit;
    hit.status=status;hit.distance=limit;hit.position=SV3(0.0);hit.outwardNormal=SV3(0.0);hit.shadingNormal=SV3(0.0);
    hit.objectId=uint(0);hit.materialId=uint(0);hit.instanceId=uint(0);hit.triangleId=uint(0);hit.entering=false;
    return hit;
}
bool sceneBoundsNear(SV3 origin,SV3 direction,SV3 low,SV3 high,float limit SC_OUT(float,nearDistance)) {
    float nearT=0.0,farT=limit;
    #pragma optionNV(unroll none)
    for(int axis=0;axis<3;++axis) {
        if(direction[axis]==0.0) {if(origin[axis]<low[axis]||origin[axis]>high[axis]) return false;}
        else {float a=(low[axis]-origin[axis])/direction[axis],b=(high[axis]-origin[axis])/direction[axis];
            nearT=max(nearT,min(a,b));farT=min(farT,max(a,b));if(nearT>farT) return false;}
    }
    nearDistance=nearT;return true;
}
int sceneTriangle(SV3 origin,SV3 direction,SV3 a,SV3 b,SV3 c,
                  float minimum,float maximum SC_OUT(float,distance) SC_OUT(SV3,normal) SC_OUT(float,hitU) SC_OUT(float,hitV)) {
    SV3 e1=b-a,e2=c-a,p=cross(direction,e2);float det=dot(e1,p);
    normal=cross(e1,e2);
    if(!sceneFinite3(e1)||!sceneFinite3(e2)||!sceneFinite3(p)||!sceneFinite3(normal)||!sceneFinite(det)) return -1;
    if(det==0.0) return 0;
    SV3 t=origin-a;float u=dot(t,p)/det;
    if(!sceneFinite(u)) return -1;if(u<0.0||u>1.0) return 0;
    SV3 q=cross(t,e1);float v=dot(direction,q)/det;
    if(!sceneFinite(v)) return -1;if(v<0.0||u+v>1.0) return 0;
    distance=dot(e2,q)/det;
    if(!sceneFinite(distance)) return -1;
    hitU=u;hitV=v;
    return distance>=minimum&&distance<=maximum?1:0;
}
int sceneChildren(SV3 origin,SV3 direction,float limit,int node,int escape SC_STACK_ARGS ST_ARGS) {
    int left=node+2;
    if(left+1>=escape) return -1;
    SV4 ll=opaqueRead(left ST_PASS),lh=opaqueRead(left+1 ST_PASS);
    if(!sceneRange(ll,lh,left,escape)) return -1;
    int right=int(ll.w);
    if(right+1>=escape) return -1;
    SV4 rl=opaqueRead(right ST_PASS),rh=opaqueRead(right+1 ST_PASS);
    if(!sceneRange(rl,rh,right,escape)||int(rl.w)!=escape) return -1;
    float leftNear=0.0,rightNear=0.0;
    bool useLeft=sceneBoundsNear(origin,direction,ll.xyz,lh.xyz,limit,leftNear);
    bool useRight=sceneBoundsNear(origin,direction,rl.xyz,rh.xyz,limit,rightNear);
    if(pending+(useLeft?1:0)+(useRight?1:0)>capacity) return -2;
    // Push the farther child first, so the next pop visits the nearer one.
    if(useLeft&&useRight) {
        bool leftFirst=leftNear<=rightNear;
        nodes[pending++]=leftFirst?right:left;nodes[pending++]=leftFirst?left:right;
    } else if(useLeft) nodes[pending++]=left;
    else if(useRight) nodes[pending++]=right;
    return 0;
}
// preferred, when nonzero, names an object whose hits win over any other
// object's hit in [minimum, maximum], whatever their distances. Among hits of
// the same kind the nearest wins and ties go to the lower object index. With
// preferred 0 the query is the ordinary closest hit: reach is hit.distance.
// avoid, when nonzero, names an object the query passes through: its instances
// are skipped before their mesh walk, so they cost one TLAS node each and
// never hit. closestHitCpu skips them at the same point.
SceneClosestHit sceneClosestHit(SV3 origin,SV3 rayDirection,float minimum,float maximum,SV4 control,uint preferred,uint avoid ST_ARGS) {
    SceneClosestHit hit=sceneEmptyHit(-1,maximum);
    float rayLength=length(rayDirection);
    if(!sceneFinite3(origin)||!sceneFinite3(rayDirection)||!sceneFinite(rayLength)||rayLength<=0.0||
        !sceneFinite(minimum)||!sceneFinite(maximum)||minimum<0.0||maximum<minimum||
        control.w!=2.0||control.z!=1024.0||!sceneFinite(control.y)||floor(control.y)!=control.y||
        control.y<0.0||control.y>float(SC_TEXELS)||!sceneFinite(control.x)||floor(control.x)!=control.x||
        control.x<0.0||control.x>control.y) return hit;
    SV3 direction=rayDirection/rayLength;
    int root=int(control.x),end=int(control.y);
    hit.status=0;
    if(root==end) return hit;
    if(root<1) return sceneEmptyHit(-1,maximum);
    SV4 allowance=opaqueRead(root-1 ST_PASS);
    if(!sceneIndex(allowance.x,65536)||!sceneIndex(allowance.y,65536)||
        !sceneIndex(allowance.z,SC_STACK_CAPACITY+1)||allowance.z<1.0||allowance.w!=3.0)
        return sceneEmptyHit(-1,maximum);
    uint budget=uint(allowance.x)+uint(allowance.y)*uint(65536),work=uint(0);
    if(budget==uint(0)) return sceneEmptyHit(-1,maximum);
    int capacity=int(allowance.z);
    int nodes[SC_STACK_CAPACITY];int pending=1;nodes[0]=root;
    int bestObject=2147483647;
    float reach=maximum;
    while(pending>0) {
        if(work>=budget) return sceneEmptyHit(-2,maximum);
        ++work;int node=nodes[--pending];
        if(node<root||node+1>=end) return sceneEmptyHit(-1,maximum);
        SV4 low=opaqueRead(node ST_PASS),high=opaqueRead(node+1 ST_PASS);
        if(!sceneRange(low,high,node,end)) return sceneEmptyHit(-1,maximum);
        int escape=int(low.w),leaf=int(high.w)-1;float nearDistance=0.0;
        if(sceneBoundsNear(origin,direction,low.xyz,high.xyz,reach,nearDistance)) {
            if(leaf<0) {
                int pushed=sceneChildren(origin,direction,reach,node,escape,nodes,pending,capacity ST_PASS);
                if(pushed<0) return sceneEmptyHit(pushed,maximum);
                continue;
            }
            if(leaf+3>=end) return sceneEmptyHit(-1,maximum);
            SV4 r0=opaqueRead(leaf ST_PASS),r1=opaqueRead(leaf+1 ST_PASS),r2=opaqueRead(leaf+2 ST_PASS);
            SV4 mesh=opaqueRead(leaf+3 ST_PASS);
            if(!sceneFinite3(r0.xyz)||!sceneFinite(r0.w)||!sceneFinite3(r1.xyz)||!sceneFinite(r1.w)||
                !sceneFinite3(r2.xyz)||!sceneFinite(r2.w)||!sceneIndex(mesh.x,end)||
                !sceneFinite(mesh.y)||floor(mesh.y)!=mesh.y||mesh.y<=mesh.x||mesh.y>float(node)||
                !sceneIndex(mesh.z,end)||!sceneIndex(mesh.w,SC_EXACT_INDEX_RANGE)) return sceneEmptyHit(-1,maximum);
            SV4 ids=opaqueRead(int(mesh.z) ST_PASS);
            if(!sceneIndex(ids.x,65536)||!sceneIndex(ids.y,65536)||!sceneIndex(ids.z,65536)||!sceneIndex(ids.w,65536))
                return sceneEmptyHit(-1,maximum);
            uint objectId=uint(ids.x)+uint(ids.y)*uint(65536),materialId=uint(ids.z)+uint(ids.w)*uint(65536);
            uint instanceId=uint(mesh.w);
            if(objectId==uint(0)||materialId==uint(0)) return sceneEmptyHit(-1,maximum);
            if(avoid!=uint(0)&&objectId==avoid) continue;
            SV3 localOrigin=SV3(dot(r0.xyz,origin)+r0.w,dot(r1.xyz,origin)+r1.w,dot(r2.xyz,origin)+r2.w);
            // Do not normalize this transformed direction. Its parameter is world distance.
            SV3 localDirection=SV3(dot(r0.xyz,direction),dot(r1.xyz,direction),dot(r2.xyz,direction));
            if(!sceneFinite3(localOrigin)||!sceneFinite3(localDirection)||length(localDirection)==0.0)
                return sceneEmptyHit(-1,maximum);
            int meshRoot=int(mesh.x),meshEnd=int(mesh.y);
            int meshNodes[SC_STACK_CAPACITY];int meshPending=1;meshNodes[0]=meshRoot;
            while(meshPending>0) {
                if(work>=budget) return sceneEmptyHit(-2,maximum);
                ++work;int at=meshNodes[--meshPending];
                if(at<meshRoot||at+1>=meshEnd) return sceneEmptyHit(-1,maximum);
                SV4 ml=opaqueRead(at ST_PASS),mh=opaqueRead(at+1 ST_PASS);
                if(!sceneRange(ml,mh,at,meshEnd)) return sceneEmptyHit(-1,maximum);
                int next=int(ml.w),tri=int(mh.w)-1;
                if(sceneBoundsNear(localOrigin,localDirection,ml.xyz,mh.xyz,reach,nearDistance)) {
                    if(tri<0) {
                        int pushed=sceneChildren(localOrigin,localDirection,reach,at,next,meshNodes,meshPending,capacity ST_PASS);
                        if(pushed<0) return sceneEmptyHit(pushed,maximum);
                        continue;
                    }
                    if(tri<at+2||tri>=next) return sceneEmptyHit(-1,maximum);
                    float countValue=opaqueRead(tri ST_PASS).w;
                    if(!sceneIndex(countValue,5)||countValue<1.0||tri+int(countValue)*3>next)
                        return sceneEmptyHit(-1,maximum);
                    #pragma optionNV(unroll none)
                    for(int t=0;t<4;++t) {
                        if(t>=int(countValue)) break;
                        if(work>=budget) return sceneEmptyHit(-2,maximum);
                        ++work;int offset=tri+t*3;
                        SV4 a=opaqueRead(offset ST_PASS),b=opaqueRead(offset+1 ST_PASS),c=opaqueRead(offset+2 ST_PASS);
                        if(!sceneFinite3(a.xyz)||!sceneFinite3(b.xyz)||!sceneFinite3(c.xyz)||!sceneIndex(b.w,SC_EXACT_INDEX_RANGE)||!sceneIndex(c.w,SC_EXACT_INDEX_RANGE))
                            return sceneEmptyHit(-1,maximum);
                        float distance=0.0;SV3 normal=SV3(0.0);
                        float hitU=0.0,hitV=0.0;
                        int triangleStatus=sceneTriangle(localOrigin,localDirection,a.xyz,b.xyz,c.xyz,minimum,reach,distance,normal,hitU,hitV);
                        if(triangleStatus<0) return sceneEmptyHit(-1,maximum);
                        if(triangleStatus>0) {
                            uint triangleId=uint(b.w);
                            int objectIndex=int(mesh.z);
                            bool candidatePreferred=preferred!=uint(0)&&objectId==preferred;
                            bool bestPreferred=preferred!=uint(0)&&hit.status==1&&hit.objectId==preferred;
                            bool wins=hit.status==0||(candidatePreferred&&!bestPreferred)||
                                (candidatePreferred==bestPreferred&&(distance<hit.distance||
                                (distance==hit.distance&&(objectIndex<bestObject||
                                (objectIndex==bestObject&&(instanceId<hit.instanceId||
                                (instanceId==hit.instanceId&&triangleId<hit.triangleId)))))));
                            if(wins) {
                                SV3 worldNormal=SV3(r0.x*normal.x+r1.x*normal.y+r2.x*normal.z,
                                    r0.y*normal.x+r1.y*normal.y+r2.y*normal.z,
                                    r0.z*normal.x+r1.z*normal.y+r2.z*normal.z);
                                float normalLength=length(worldNormal);
                                if(!sceneFinite(normalLength)||normalLength<=0.0) return sceneEmptyHit(-1,maximum);
                                hit.status=1;hit.distance=distance;hit.position=origin+direction*distance;
                                hit.outwardNormal=worldNormal/normalLength;hit.entering=dot(direction,hit.outwardNormal)<0.0;
                                hit.objectId=objectId;hit.materialId=materialId;hit.instanceId=instanceId;hit.triangleId=triangleId;
                                hit.shadingNormal=hit.outwardNormal;
                                if(c.w>0.0) {
                                    int block=int(c.w);
                                    if(block+2>=end) return sceneEmptyHit(-1,maximum);
                                    SV3 blended=opaqueRead(block ST_PASS).xyz*(1.0-hitU-hitV)
                                        +opaqueRead(block+1 ST_PASS).xyz*hitU+opaqueRead(block+2 ST_PASS).xyz*hitV;
                                    SV3 worldBlended=SV3(r0.x*blended.x+r1.x*blended.y+r2.x*blended.z,
                                        r0.y*blended.x+r1.y*blended.y+r2.y*blended.z,
                                        r0.z*blended.x+r1.z*blended.y+r2.z*blended.z);
                                    float blendedLength=length(worldBlended);
                                    if(sceneFinite(blendedLength)&&blendedLength>0.0) hit.shadingNormal=worldBlended/blendedLength;
                                }
                                bestObject=objectIndex;
                                if(preferred==uint(0)||candidatePreferred) reach=distance;
                            }
                        }
                    }
                }
            }
        }
    }
    return hit;
}
SceneClosestHit sceneClosestHit(SV3 origin,SV3 rayDirection,float minimum,float maximum,SV4 control,uint preferred ST_ARGS) {
    return sceneClosestHit(origin,rayDirection,minimum,maximum,control,preferred,uint(0) ST_PASS);
}
SceneClosestHit sceneClosestHit(SV3 origin,SV3 rayDirection,float minimum,float maximum,SV4 control ST_ARGS) {
    return sceneClosestHit(origin,rayDirection,minimum,maximum,control,uint(0) ST_PASS);
}

)shader";
template<class String> inline String shaderBoundsRecipe()
{
    return String("\n#define OPAQUE_WALK_VISITS ") + surfacesource::number<String>(kExactFloatIndexRange) +
           "\n#define SC_EXACT_INDEX_RANGE " + surfacesource::number<String>(kExactFloatIndexRange) +
           "\n#define SC_STACK_CAPACITY " + surfacesource::number<String>(kClosestHitStackCapacity) + "\n";
}
// GLSL 4.30 only. Qualify the existing ordered stages in their own functions;
// a precise caller result would not constrain unqualified callee arithmetic.
// Built-in numeric error bounds still apply. Default GLSL/MSL bytes are kept.
inline std::string shaderBounds() { return shaderBoundsRecipe<std::string>(); }
inline std::string preciseGl43ShaderBody()
{
    std::string source=shader;
    const auto closestHitBegin=source.find("bool sceneBoundsNear(");
    if(closestHitBegin==std::string::npos)return {};
    const std::pair<const char*,const char*> stages[]{
        {"float nearT=0.0,farT=limit;","precise float nearT=0.0,farT=limit;"},
        {"float a=(low[axis]-origin[axis])/direction[axis],b=(high[axis]-origin[axis])/direction[axis];",
         "precise float a=(low[axis]-origin[axis])/direction[axis],b=(high[axis]-origin[axis])/direction[axis];"},
        {"SC_OUT(float,nearDistance)","SC_OUT(precise float,nearDistance)"},
        {"SC_OUT(float,distance) SC_OUT(SV3,normal)","SC_OUT(precise float,distance) SC_OUT(precise SV3,normal)"},
        {"SV3 e1=b-a,e2=c-a,p=cross(direction,e2);float det=dot(e1,p);",
         "precise SV3 e1=b-a,e2=c-a,p=cross(direction,e2);precise float det=dot(e1,p);"},
        {"SV3 t=origin-a;float u=dot(t,p)/det;","precise SV3 t=origin-a;precise float u=dot(t,p)/det;"},
        {"SV3 q=cross(t,e1);float v=dot(direction,q)/det;","precise SV3 q=cross(t,e1);precise float v=dot(direction,q)/det;"},
        {"if(!sceneFinite(v)) return -1;if(v<0.0||u+v>1.0) return 0;",
         "if(!sceneFinite(v)) return -1;if(v<0.0) return 0;precise float barycentricSum=u+v;if(barycentricSum>1.0) return 0;"},
        {"float leftNear=0.0,rightNear=0.0;","precise float leftNear=0.0,rightNear=0.0;"},
        {"float rayLength=length(rayDirection);","precise float rayLength=length(rayDirection);"},
        {"SV3 direction=rayDirection/rayLength;","precise SV3 direction=rayDirection/rayLength;"},
        {"SV3 localOrigin=SV3(dot(r0.xyz,origin)+r0.w,dot(r1.xyz,origin)+r1.w,dot(r2.xyz,origin)+r2.w);",
         "precise SV3 localOrigin=SV3(dot(r0.xyz,origin)+r0.w,dot(r1.xyz,origin)+r1.w,dot(r2.xyz,origin)+r2.w);"},
        {"SV3 localDirection=SV3(dot(r0.xyz,direction),dot(r1.xyz,direction),dot(r2.xyz,direction));",
         "precise SV3 localDirection=SV3(dot(r0.xyz,direction),dot(r1.xyz,direction),dot(r2.xyz,direction));"},
        {"if(!sceneFinite3(localOrigin)||!sceneFinite3(localDirection)||length(localDirection)==0.0)\n                return sceneEmptyHit(-1,maximum);",
         "if(!sceneFinite3(localOrigin)||!sceneFinite3(localDirection)) return sceneEmptyHit(-1,maximum);\n            precise float localDirectionLength=length(localDirection);\n            if(localDirectionLength==0.0) return sceneEmptyHit(-1,maximum);"},
        {"float distance=0.0;SV3 normal=SV3(0.0);","precise float distance=0.0;precise SV3 normal=SV3(0.0);"},
        {"SV3 worldNormal=SV3(r0.x*normal.x+r1.x*normal.y+r2.x*normal.z,",
         "precise SV3 worldNormal=SV3(r0.x*normal.x+r1.x*normal.y+r2.x*normal.z,"},
        {"float normalLength=length(worldNormal);","precise float normalLength=length(worldNormal);"},
        {"hit.status=1;hit.distance=distance;hit.position=origin+direction*distance;\n                                hit.outwardNormal=worldNormal/normalLength;hit.entering=dot(direction,hit.outwardNormal)<0.0;",
         "precise SV3 position=origin+direction*distance;\n                                precise SV3 outwardNormal=worldNormal/normalLength;\n                                precise float enteringDot=dot(direction,outwardNormal);\n                                hit.status=1;hit.distance=distance;hit.position=position;\n                                hit.outwardNormal=outwardNormal;hit.entering=enteringDot<0.0;"}
    };
    for(const auto& stage:stages) {
        std::size_t at=closestHitBegin;bool found=false;
        while((at=source.find(stage.first,at))!=std::string::npos) {
            found=true;source.replace(at,std::char_traits<char>::length(stage.first),stage.second);
            at+=std::char_traits<char>::length(stage.second);
        }
        if(!found)return {}; // Source drift cannot silently omit a contract stage.
    }
    return source;
}
inline std::string glslShader(bool precise43=false)
{
    const char* header="\nuniform sampler2D uOpaqueVisibility;\nuniform vec4 uOpaqueControl;\n#define SV3 "
                       "vec3\n#define SV4 vec4\n#define SC_TEXELS (textureSize(uOpaqueVisibility,0).x*textureSize(uOpaqueVisibility,0).y)\n#define SC_OUT(T,N) , out T N\n#define SC_STACK_ARGS , inout int nodes[SC_STACK_CAPACITY], inout int pending, int capacity\n#define ST_ARGS\n#define ST_PASS\n#define ST_READ "
                       "texelFetch(uOpaqueVisibility,ivec2(index%1024,index/1024),0)\n";
    if(!precise43)return std::string(header)+shaderBounds()+shader;
    const auto body=preciseGl43ShaderBody();
    return body.empty()?std::string{}:std::string(header)+shaderBounds()+body;
}
template<class String> inline String metalShaderRecipe()
{
    String text = String("\n#define SV3 float3\n#define SV4 float4\n#define SC_TEXELS (opaqueTexture.get_width()*opaqueTexture.get_height())\n#define SC_OUT(T,N) , thread T& N\n#define SC_STACK_ARGS , thread int* nodes, thread int& pending, int capacity\n#define ST_ARGS , texture2d<float> "
                       "opaqueTexture\n#define ST_PASS , opaqueTexture\n#define ST_READ "
                       "opaqueTexture.read(uint2(index%1024,index/1024))\n") +
           shaderBoundsRecipe<String>() + shader;
    // The NVIDIA unroll pragmas live in the shared shader text. MSL does not use them.
    surfacesource::replace(text, "#pragma optionNV(unroll none)\n", "");
    return text;
}
inline std::string metalShader() { return metalShaderRecipe<std::string>(); }
} // namespace arbitgpu::opaquevisibility
