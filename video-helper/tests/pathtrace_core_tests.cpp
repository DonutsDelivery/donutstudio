// Core tests for the CPU path tracer foundation: scene ingest, BVH, camera,
// tile pool and progressive accumulation.

#include "pathtrace/accumulator.h"
#include "pathtrace/bvh.h"
#include "pathtrace/scene_ingest.h"
#include "pathtrace/thread_pool.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

namespace pt = videohelper::pathtrace;
namespace grid = HarmonicMIDI::grid;
using pt::Vec3;

int checks = 0;
int failures = 0;

void check (bool condition, const std::string& message)
{
    ++checks;
    if (! condition)
    {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

bool approx (float a, float b, float tolerance)
{
    return std::fabs (a - b) <= tolerance;
}

bool nearVec (Vec3 a, Vec3 b, float tolerance)
{
    return approx (a.x, b.x, tolerance) && approx (a.y, b.y, tolerance) && approx (a.z, b.z, tolerance);
}

float uniform (std::mt19937& rng, float lo, float hi)
{
    return std::uniform_real_distribution<float> (lo, hi) (rng);
}

Vec3 randomUnit (std::mt19937& rng)
{
    std::normal_distribution<float> normal (0.0f, 1.0f);
    for (;;)
    {
        const Vec3 v { normal (rng), normal (rng), normal (rng) };
        const float len = pt::length (v);
        if (len > 1.0e-3f)
            return v * (1.0f / len);
    }
}

grid::SceneQuaternion randomRotation (std::mt19937& rng)
{
    std::normal_distribution<float> normal (0.0f, 1.0f);
    float x = normal (rng), y = normal (rng), z = normal (rng), w = normal (rng);
    const float n = std::sqrt (x * x + y * y + z * z + w * w);
    return { x / n, y / n, z / n, w / n };
}

pt::Hit bruteClosest (const std::vector<Vec3>& p, const pt::Ray& ray)
{
    pt::Hit best;
    pt::PreparedRay prepared;
    if (! pt::prepareRay (ray, prepared))
        return best;
    float tMax = ray.tMax;
    for (std::uint32_t i = 0; i < p.size() / 3; ++i)
    {
        if (i == ray.ignoreTriangle)
            continue;
        pt::Hit hit;
        if (pt::intersectTriangle (prepared, p[3 * i], p[3 * i + 1], p[3 * i + 2], ray.tMin, tMax, hit))
        {
            tMax = hit.t;
            best = hit;
            best.triangle = i;
        }
    }
    return best;
}

bool bruteOccluded (const std::vector<Vec3>& p, const pt::Ray& ray)
{
    pt::PreparedRay prepared;
    if (! pt::prepareRay (ray, prepared))
        return false;
    for (std::uint32_t i = 0; i < p.size() / 3; ++i)
    {
        pt::Hit hit;
        if (i != ray.ignoreTriangle
            && pt::intersectTriangle (prepared, p[3 * i], p[3 * i + 1], p[3 * i + 2], ray.tMin, ray.tMax, hit))
            return true;
    }
    return false;
}

Vec3 hitPoint (const std::vector<Vec3>& p, const pt::Hit& hit)
{
    const std::size_t i = 3 * std::size_t { hit.triangle };
    return p[i] * (1.0f - hit.u - hit.v) + p[i + 1] * hit.u + p[i + 2] * hit.v;
}

// Unit cube centred on the origin, counter-clockwise outward winding.
const std::array<std::uint32_t, 36> kCubeIndices {
    1, 3, 7, 1, 7, 5,  0, 4, 6, 0, 6, 2,  2, 6, 7, 2, 7, 3,
    0, 1, 5, 0, 5, 4,  4, 5, 7, 4, 7, 6,  0, 2, 3, 0, 3, 1 };

Vec3 cubeCorner (std::uint32_t i)
{
    return { (i & 1u) ? 0.5f : -0.5f, (i & 2u) ? 0.5f : -0.5f, (i & 4u) ? 0.5f : -0.5f };
}

std::size_t addCube (grid::Visual3DScene& scene, std::uint32_t id, const grid::SceneTransform3D& transform,
                     std::uint32_t parent = 0, std::uint32_t material = 0)
{
    const std::size_t index = scene.objectCount++;
    auto& object = scene.objects[index];
    object.id.value = id;
    object.parent.value = parent;
    object.material.value = material;
    object.transform = transform;
    object.firstVertex = static_cast<std::uint32_t> (scene.vertexCount);
    object.vertexCount = 8;
    object.firstIndex = static_cast<std::uint32_t> (scene.indexCount);
    object.indexCount = 36;
    for (std::uint32_t v = 0; v < 8; ++v)
    {
        const Vec3 c = cubeCorner (v);
        const Vec3 n = pt::normalizeOrZero (c);
        auto& vertex = scene.vertices[scene.vertexCount++];
        vertex.position = { c.x, c.y, c.z };
        vertex.normal = { n.x, n.y, n.z };
        vertex.uv = { c.x + 0.5f, c.y + 0.5f };
    }
    for (const auto i : kCubeIndices)
        scene.indices[scene.indexCount++] = i;
    return index;
}

std::unique_ptr<grid::Visual3DScene> emptyScene()
{
    auto scene = std::make_unique<grid::Visual3DScene>();
    scene->materialCount = 1;
    scene->materials[0].id.value = 1;
    return scene;
}

grid::SceneTransform3D translation (float x, float y, float z)
{
    grid::SceneTransform3D t;
    t.translation = { x, y, z };
    return t;
}

// Indexed closed mesh expanded into a triangle soup with bit-identical shared vertices.
struct Mesh
{
    std::vector<Vec3> vertices;
    std::vector<std::uint32_t> indices;

    std::vector<Vec3> soup() const
    {
        std::vector<Vec3> out;
        for (const auto i : indices)
            out.push_back (vertices[i]);
        return out;
    }
};

Mesh uvSphere (std::uint32_t stacks, std::uint32_t slices)
{
    Mesh mesh;
    const float pi = 3.14159265358979f;
    mesh.vertices.push_back ({ 0.0f, 1.0f, 0.0f });
    for (std::uint32_t i = 1; i < stacks; ++i)
    {
        const float theta = pi * static_cast<float> (i) / static_cast<float> (stacks);
        for (std::uint32_t j = 0; j < slices; ++j)
        {
            const float phi = 2.0f * pi * static_cast<float> (j) / static_cast<float> (slices);
            mesh.vertices.push_back ({ std::sin (theta) * std::cos (phi), std::cos (theta),
                                       std::sin (theta) * std::sin (phi) });
        }
    }
    mesh.vertices.push_back ({ 0.0f, -1.0f, 0.0f });
    const auto ring = [slices] (std::uint32_t i, std::uint32_t j) { return 1 + (i - 1) * slices + j % slices; };
    const auto south = static_cast<std::uint32_t> (mesh.vertices.size() - 1);
    for (std::uint32_t j = 0; j < slices; ++j)
    {
        mesh.indices.insert (mesh.indices.end(), { 0u, ring (1, j + 1), ring (1, j) });
        for (std::uint32_t i = 1; i + 1 < stacks; ++i)
        {
            mesh.indices.insert (mesh.indices.end(), { ring (i, j), ring (i, j + 1), ring (i + 1, j) });
            mesh.indices.insert (mesh.indices.end(), { ring (i, j + 1), ring (i + 1, j + 1), ring (i + 1, j) });
        }
        mesh.indices.insert (mesh.indices.end(), { ring (stacks - 1, j), ring (stacks - 1, j + 1), south });
    }
    return mesh;
}

Mesh cubeMesh()
{
    Mesh mesh;
    for (std::uint32_t v = 0; v < 8; ++v)
        mesh.vertices.push_back (cubeCorner (v));
    mesh.indices.assign (kCubeIndices.begin(), kCubeIndices.end());
    return mesh;
}

void transformMesh (Mesh& mesh, const pt::Mat4& m)
{
    for (auto& v : mesh.vertices)
        v = pt::transformPoint (m, v);
}

pt::Ray randomRay (std::mt19937& rng, const std::vector<Vec3>& soup, float extent)
{
    pt::Ray ray;
    ray.origin = { uniform (rng, -extent, extent), uniform (rng, -extent, extent), uniform (rng, -extent, extent) };
    const float pick = uniform (rng, 0.0f, 1.0f);
    if (pick < 0.15f)
    {
        // Axis-aligned, with exact zero components.
        const int axis = static_cast<int> (uniform (rng, 0.0f, 2.999f));
        ray.direction = {};
        ray.direction[axis] = uniform (rng, 0.0f, 1.0f) < 0.5f ? -1.0f : 1.0f;
    }
    else if (pick < 0.6f && ! soup.empty())
    {
        // Aimed at a random point of a random triangle, so most of these hit.
        const auto tri = static_cast<std::size_t> (uniform (rng, 0.0f, 1.0f) * static_cast<float> (soup.size() / 3)) % (soup.size() / 3);
        float u = uniform (rng, 0.0f, 1.0f), v = uniform (rng, 0.0f, 1.0f);
        if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
        const Vec3 target = soup[3 * tri] * (1.0f - u - v) + soup[3 * tri + 1] * u + soup[3 * tri + 2] * v;
        ray.direction = target - ray.origin;
    }
    else
    {
        ray.direction = randomUnit (rng);
    }
    return ray;
}

void compareWithBruteForce (const std::vector<Vec3>& soup, std::mt19937& rng, std::uint32_t rayCount,
                            float extent, const std::string& label)
{
    pt::Bvh bvh;
    bvh.build (soup);
    std::uint32_t mismatches = 0;
    std::uint32_t occlusionMismatches = 0;
    std::uint32_t skipMismatches = 0;
    std::uint32_t hits = 0;
    std::uint32_t badPoints = 0;
    for (std::uint32_t r = 0; r < rayCount; ++r)
    {
        pt::Ray ray = randomRay (rng, soup, extent);
        const pt::Hit expected = bruteClosest (soup, ray);
        pt::Hit actual;
        const bool found = bvh.intersect (ray, actual);
        if (found != expected.valid() || (found && actual.t != expected.t))
            ++mismatches;
        if (found)
        {
            ++hits;
            const Vec3 onRay = ray.origin + ray.direction * actual.t;
            const std::size_t i = 3 * std::size_t { actual.triangle };
            const Vec3 e1 = soup[i + 1] - soup[i], e2 = soup[i + 2] - soup[i];
            const float longest = std::max (pt::dot (e1, e1), pt::dot (e2, e2));
            // Slivers have ill-conditioned barycentrics; check the rest.
            const bool wellShaped = pt::length (pt::cross (e1, e2)) > 0.05f * longest;
            if (wellShaped && ! nearVec (hitPoint (soup, actual), onRay, 1.0e-3f * (1.0f + pt::length (onRay))))
                ++badPoints;
            // Skipping the first hit's triangle must expose the same next hit.
            pt::Ray skip = ray;
            skip.ignoreTriangle = actual.triangle;
            pt::Hit next;
            const pt::Hit expectedNext = bruteClosest (soup, skip);
            const bool nextFound = bvh.intersect (skip, next);
            if (nextFound != expectedNext.valid() || (nextFound && next.t != expectedNext.t))
                ++skipMismatches;
        }
        pt::Ray shadow = ray;
        shadow.tMax = uniform (rng, 0.0f, 2.0f * extent);
        if (bvh.occluded (shadow) != bruteOccluded (soup, shadow))
            ++occlusionMismatches;
    }
    check (mismatches == 0, label + ": BVH closest hit equals brute force (" + std::to_string (mismatches) + " mismatches)");
    check (skipMismatches == 0, label + ": ignoreTriangle agrees with brute force");
    check (occlusionMismatches == 0, label + ": occlusion agrees with brute force ("
                                         + std::to_string (occlusionMismatches) + " mismatches)");
    check (badPoints == 0, label + ": barycentrics reproduce the hit point");
    check (soup.size() < 6 || hits > rayCount / 10, label + ": the ray set exercises hits");

    const auto& stats = bvh.stats();
    check (bvh.triangleCount() == soup.size() / 3, label + ": every triangle is in the BVH");
    check (stats.maxDepth < pt::Bvh::kMaxDepth, label + ": depth stays inside the traversal stack");
    check (soup.empty() || stats.nodeCount == 2 * stats.leafCount - 1, label + ": binary tree node count");
}

void bvhTests()
{
    std::mt19937 rng (0x5eed1234u);
    for (const std::uint32_t count : { 1u, 2u, 7u, 64u, 1000u, 4000u })
    {
        std::vector<Vec3> soup;
        for (std::uint32_t i = 0; i < count; ++i)
        {
            const Vec3 centre { uniform (rng, -2.0f, 2.0f), uniform (rng, -2.0f, 2.0f), uniform (rng, -2.0f, 2.0f) };
            const float size = uniform (rng, 0.02f, 0.6f);
            for (int k = 0; k < 3; ++k)
                soup.push_back (centre + randomUnit (rng) * size);
        }
        compareWithBruteForce (soup, rng, 3000, 3.0f, "random soup of " + std::to_string (count));
    }

    // A thin, elongated soup stresses the SAH on one axis.
    {
        std::vector<Vec3> soup;
        for (std::uint32_t i = 0; i < 2000; ++i)
        {
            const float x = uniform (rng, -50.0f, 50.0f);
            soup.push_back ({ x, uniform (rng, -0.1f, 0.1f), 0.0f });
            soup.push_back ({ x + 0.05f, uniform (rng, -0.1f, 0.1f), 0.01f });
            soup.push_back ({ x, uniform (rng, -0.1f, 0.1f), 0.05f });
        }
        compareWithBruteForce (soup, rng, 2000, 50.0f, "elongated soup");
    }

    // Identical centroids defeat every binned split; the median fallback must still terminate.
    {
        std::vector<Vec3> soup;
        for (std::uint32_t i = 0; i < 300; ++i)
        {
            const float s = 0.1f + 0.002f * static_cast<float> (i);
            soup.push_back ({ -s, -s, 0.0f });
            soup.push_back ({ 2.0f * s, -s, 0.0f });
            soup.push_back ({ -s, 2.0f * s, 0.0f });
        }
        compareWithBruteForce (soup, rng, 1000, 2.0f, "coincident centroids");
        pt::Bvh bvh;
        bvh.build (soup);
        check (bvh.stats().medianSplits > 0, "coincident centroids use median splits");
        check (bvh.stats().maxLeafSize <= 8, "median splits respect the leaf size");
    }

    // Cube scene through the full ingest path.
    {
        auto scene = emptyScene();
        grid::SceneTransform3D t;
        t.rotation = randomRotation (rng);
        t.translation = { 0.3f, -0.2f, 0.1f };
        addCube (*scene, 1, t, 0, 1);
        pt::IngestedScene ingested;
        pt::ingestScene (*scene, ingested);
        check (ingested.triangleCount() == 12, "cube ingests 12 triangles");
        compareWithBruteForce (ingested.positions, rng, 4000, 2.0f, "cube scene");
    }

    // Empty and malformed inputs.
    {
        pt::Bvh bvh;
        bvh.build (std::vector<Vec3> {});
        pt::Hit hit;
        pt::Ray ray;
        check (! bvh.intersect (ray, hit) && ! bvh.occluded (ray), "empty BVH reports no hit");
        std::vector<Vec3> one { { -1.0f, -1.0f, -1.0f }, { 1.0f, -1.0f, -1.0f }, { 0.0f, 1.0f, -1.0f } };
        bvh.build (one);
        pt::Ray zero;
        zero.direction = {};
        check (! bvh.intersect (zero, hit), "zero direction is refused");
        pt::Ray badOrigin;
        badOrigin.origin.x = std::nanf ("");
        check (! bvh.intersect (badOrigin, hit), "non-finite origin is refused");
        check (bvh.intersect (ray, hit) && approx (hit.t, 1.0f, 1.0e-6f), "single triangle hit at t = 1");
        pt::Ray limited = ray;
        limited.tMax = 0.999f;
        check (! bvh.intersect (limited, hit) && ! bvh.occluded (limited), "tMax excludes a farther triangle");
        limited.tMax = 2.0f;
        limited.tMin = 1.001f;
        check (! bvh.occluded (limited), "tMin excludes a nearer triangle");
    }

    // The watertight test against a double-precision reference away from edges.
    {
        std::uint32_t wrong = 0;
        for (int i = 0; i < 5000; ++i)
        {
            const Vec3 a = randomUnit (rng) * 2.0f, b = randomUnit (rng) * 2.0f, c = randomUnit (rng) * 2.0f;
            if (pt::length (pt::cross (b - a, c - a)) < 0.05f)
                continue;
            float u = uniform (rng, -0.3f, 1.0f), v = uniform (rng, -0.3f, 1.0f);
            const float w = 1.0f - u - v;
            const float margin = std::min (u, std::min (v, w));
            if (std::fabs (margin) < 0.01f)
                continue;
            const Vec3 target = a * w + b * u + c * v;
            pt::Ray ray;
            ray.origin = target + randomUnit (rng) * 3.0f;
            ray.direction = target - ray.origin;
            const Vec3 normal = pt::normalizeOrZero (pt::cross (b - a, c - a));
            if (std::fabs (pt::dot (pt::normalizeOrZero (ray.direction), normal)) < 0.1f)
                continue;
            pt::PreparedRay prepared;
            pt::Hit hit;
            const bool found = pt::prepareRay (ray, prepared)
                && pt::intersectTriangle (prepared, a, b, c, 0.0f, 10.0f, hit);
            if (found != (margin > 0.0f) || (found && (! approx (hit.t, 1.0f, 1.0e-4f)
                                                       || ! approx (hit.u, u, 1.0e-3f) || ! approx (hit.v, v, 1.0e-3f))))
                ++wrong;
        }
        check (wrong == 0, "triangle test matches analytic barycentrics (" + std::to_string (wrong) + " wrong)");
    }
}

// Rays from inside a closed mesh aimed exactly at its vertices and edges must hit.
void watertightTests()
{
    std::mt19937 rng (0xa7e27167u);
    std::vector<std::pair<std::string, Mesh>> meshes;
    meshes.emplace_back ("cube", cubeMesh());
    meshes.emplace_back ("sphere", uvSphere (13, 17));
    meshes.emplace_back ("dense sphere", uvSphere (24, 37));
    for (auto& [name, mesh] : meshes)
    {
        grid::SceneTransform3D t;
        t.rotation = randomRotation (rng);
        t.translation = { 0.137f, -12.71f, 3.3331f };
        t.scale = { 1.7f, 0.93f, 1.21f };
        const pt::Mat4 m = pt::transformMatrix (t);
        transformMesh (mesh, m);
        const Vec3 centre = pt::transformPoint (m, {});
        const std::vector<Vec3> soup = mesh.soup();
        pt::Bvh bvh;
        bvh.build (soup);
        std::uint32_t bvhMisses = 0;
        std::uint32_t bruteMisses = 0;
        std::uint32_t rays = 0;
        for (std::size_t tri = 0; tri < mesh.indices.size(); tri += 3)
        {
            for (int edge = 0; edge < 3; ++edge)
            {
                const Vec3 a = mesh.vertices[mesh.indices[tri + static_cast<std::size_t> (edge)]];
                const Vec3 b = mesh.vertices[mesh.indices[tri + static_cast<std::size_t> ((edge + 1) % 3)]];
                for (const float s : { 0.0f, 0.5f, uniform (rng, 0.0f, 1.0f), uniform (rng, 0.0f, 1.0f) })
                {
                    const Vec3 target = a * (1.0f - s) + b * s;
                    pt::Ray ray;
                    ray.origin = centre + randomUnit (rng) * uniform (rng, 0.0f, 0.4f);
                    ray.direction = target - ray.origin;
                    pt::Hit hit;
                    ++rays;
                    bvhMisses += bvh.intersect (ray, hit) ? 0u : 1u;
                    bruteMisses += bruteClosest (soup, ray).valid() ? 0u : 1u;
                    bvhMisses += bvh.occluded (ray) ? 0u : 1u;
                }
            }
        }
        check (bruteMisses == 0, name + ": no ray through a shared edge or vertex escapes the triangle test ("
                                     + std::to_string (bruteMisses) + " of " + std::to_string (rays) + ")");
        check (bvhMisses == 0, name + ": no ray through a shared edge or vertex escapes the BVH ("
                                   + std::to_string (bvhMisses) + ")");
    }
}

void rayOffsetTests()
{
    std::mt19937 rng (0x0ff5e7u);
    std::uint32_t selfHits = 0;
    std::uint32_t wrongSide = 0;
    for (int i = 0; i < 2000; ++i)
    {
        const float scale = std::pow (10.0f, uniform (rng, -3.0f, 4.0f));
        const Vec3 a = randomUnit (rng) * scale + Vec3 { scale, -scale, 0.5f * scale };
        const Vec3 b = a + randomUnit (rng) * scale;
        const Vec3 c = a + randomUnit (rng) * scale;
        const Vec3 ng = pt::normalizeOrZero (pt::cross (b - a, c - a));
        if (pt::length (ng) == 0.0f)
            continue;
        const Vec3 p = a * 0.3f + b * 0.3f + c * 0.4f;
        Vec3 dir = randomUnit (rng);
        if (std::fabs (pt::dot (dir, ng)) < 0.05f)
            continue;
        pt::Ray ray;
        ray.origin = pt::spawnRayOrigin (p, ng, dir);
        ray.direction = dir;
        pt::PreparedRay prepared;
        pt::Hit hit;
        if (pt::prepareRay (ray, prepared) && pt::intersectTriangle (prepared, a, b, c, 0.0f, 1.0e30f, hit))
            ++selfHits;
        if ((pt::dot (ray.origin - p, ng) > 0.0f) != (pt::dot (dir, ng) > 0.0f))
            ++wrongSide;
    }
    check (selfHits == 0, "offset origins never re-hit their own triangle (" + std::to_string (selfHits) + ")");
    check (wrongSide == 0, "offset origins move to the side the ray leaves from");
}

void ingestTests()
{
    auto scene = emptyScene();
    scene->materialCount = 2;
    scene->materials[1].id.value = 9;

    // Instanced cube: the object world (+1 in x) applies after each instance transform.
    const std::size_t instanced = addCube (*scene, 5, translation (1.0f, 0.0f, 0.0f), 0, 9);
    scene->instancedObject.value = 5;
    scene->objectInstanceCount = 3;
    scene->objectInstances[0] = translation (5.0f, 0.0f, 0.0f);
    scene->objectInstances[1] = translation (0.0f, 5.0f, 0.0f);
    scene->objectInstances[1].rotation = { 0.0f, 0.38268343f, 0.0f, 0.92387953f };  // 45 degrees about +Y
    scene->objectInstances[2] = translation (0.0f, 0.0f, -5.0f);
    scene->objectInstances[2].scale = { 2.0f, 2.0f, 2.0f };

    // Parent chain: parent turns 90 degrees about +Y and lifts by 10; child sits at +2 x.
    grid::SceneTransform3D parent = translation (0.0f, 10.0f, 0.0f);
    parent.rotation = { 0.0f, 0.70710678f, 0.0f, 0.70710678f };
    {
        const std::size_t index = scene->objectCount++;
        scene->objects[index].id.value = 20;
        scene->objects[index].transform = parent;
    }
    addCube (*scene, 21, translation (2.0f, 0.0f, 0.0f), 20, 1);

    // Mirrored cube.
    grid::SceneTransform3D mirror = translation (0.0f, -10.0f, 0.0f);
    mirror.scale = { -1.0f, 1.0f, 1.0f };
    addCube (*scene, 30, mirror, 0, 1);

    // Object with an index past its vertex range and one with a bad index range.
    {
        const std::size_t index = addCube (*scene, 40, translation (0.0f, 0.0f, 40.0f), 0, 1);
        scene->indices[scene->objects[index].firstIndex + 4] = 8;
        const std::size_t broken = addCube (*scene, 41, translation (0.0f, 0.0f, 50.0f), 0, 1);
        scene->objects[broken].indexCount = static_cast<std::uint32_t> (grid::Visual3DScene::kMaxIndices);
    }

    pt::IngestedScene ingested;
    pt::ingestScene (*scene, ingested);
    check (ingested.stats.instancesExpanded == 3, "three instances expanded");
    check (ingested.stats.invalidObjectRanges == 1, "an out-of-range object is skipped");
    check (ingested.stats.invalidIndices == 1, "a triangle with an out-of-range index is skipped");
    check (ingested.triangleCount() == 12 * 3 + 12 + 12 + 11, "triangle count after instancing and validation");
    check (ingested.positions.size() == 3 * ingested.triangleCount(), "three positions per triangle");

    pt::Bvh bvh;
    bvh.build (ingested.positions);
    const auto shoot = [&] (Vec3 origin, Vec3 direction, pt::Hit& hit) {
        pt::Ray ray;
        ray.origin = origin;
        ray.direction = direction;
        return bvh.intersect (ray, hit);
    };
    pt::Hit hit;
    check (shoot ({ 6.0f, 0.0f, 10.0f }, { 0.0f, 0.0f, -1.0f }, hit) && approx (hit.t, 9.5f, 1.0e-4f)
               && ingested.triangles[hit.triangle].instanceIndex == 0
               && ingested.triangles[hit.triangle].objectId == 5
               && ingested.triangles[hit.triangle].materialId == 9
               && ingested.triangles[hit.triangle].materialIndex == 1,
           "instance 0 sits at object world times instance (x = 6)");
    // A cube turned 45 degrees about Y shows an edge at z = sqrt(0.5).
    check (shoot ({ 1.0f, 5.0f, 10.0f }, { 0.0f, 0.0f, -1.0f }, hit)
               && approx (hit.t, 10.0f - 0.70710678f, 1.0e-4f)
               && ingested.triangles[hit.triangle].instanceIndex == 1,
           "instance 1 is rotated about its own centre");
    check (shoot ({ 1.0f, 0.0f, 10.0f }, { 0.0f, 0.0f, -1.0f }, hit) && approx (hit.t, 14.0f, 1.0e-4f)
               && ingested.triangles[hit.triangle].instanceIndex == 2,
           "instance 2 is scaled, and the bare instanced object is not drawn");
    check (shoot ({ 0.0f, 10.0f, 10.0f }, { 0.0f, 0.0f, -1.0f }, hit) && approx (hit.t, 11.5f, 1.0e-4f)
               && ingested.triangles[hit.triangle].objectId == 21
               && ingested.triangles[hit.triangle].instanceIndex == pt::kNoInstance,
           "parent rotation carries the child from +x to -z");
    check (! shoot ({ 2.0f, 10.0f, 10.0f }, { 0.0f, 0.0f, -1.0f }, hit),
           "the child is not at its local position");

    // Mirrored and plain cubes both keep outward geometric and shading normals.
    std::uint32_t inward = 0;
    std::uint32_t inwardShading = 0;
    std::uint32_t mirroredTriangles = 0;
    for (std::size_t i = 0; i < ingested.triangleCount(); ++i)
    {
        const auto& attributes = ingested.triangles[i];
        if (attributes.objectId != 21 && attributes.objectId != 30)
            continue;
        const Vec3 centre = attributes.objectId == 21 ? Vec3 { 0.0f, 10.0f, -2.0f } : Vec3 { 0.0f, -10.0f, 0.0f };
        const Vec3 triCentre = (ingested.positions[3 * i] + ingested.positions[3 * i + 1] + ingested.positions[3 * i + 2]) * (1.0f / 3.0f);
        inward += pt::dot (attributes.geometricNormal, triCentre - centre) > 0.0f ? 0u : 1u;
        for (std::size_t k = 0; k < 3; ++k)
            inwardShading += pt::dot (attributes.shadingNormals[k], ingested.positions[3 * i + k] - centre) > 0.0f ? 0u : 1u;
        mirroredTriangles += attributes.mirrored ? 1u : 0u;
        check (attributes.hasShadingNormals, "authored normals are carried");
    }
    check (inward == 0, "geometric normals face out, mirrored or not");
    check (inwardShading == 0, "shading normals face out, mirrored or not");
    check (mirroredTriangles == 12, "the mirrored cube is flagged");

    // Mirroring flips the tangent handedness.
    bool flipped = true;
    for (const auto& attributes : ingested.triangles)
        if (attributes.objectId == 30)
            flipped = flipped && attributes.tangents[0][3] < 0.0f;
    check (flipped, "mirroring flips the bitangent sign");

    // World-matrix overrides replace the authored chain.
    std::vector<pt::Mat4> worlds (scene->objectCount);
    for (std::size_t i = 0; i < scene->objectCount; ++i)
        worlds[i] = pt::objectWorldMatrix (*scene, i);
    worlds[instanced] = pt::transformMatrix (translation (100.0f, 0.0f, 0.0f));
    pt::IngestOptions options;
    options.objectWorlds = worlds.data();
    options.objectWorldCount = worlds.size();
    pt::IngestedScene overridden;
    pt::ingestScene (*scene, overridden, options);
    bvh.build (overridden.positions);
    check (shoot ({ 105.0f, 0.0f, 10.0f }, { 0.0f, 0.0f, -1.0f }, hit)
               && overridden.triangles[hit.triangle].instanceIndex == 0,
           "object world override moves the instances");
}

grid::SceneQuaternion lookAtRotation (Vec3 eye, Vec3 target)
{
    const Vec3 f = pt::normalizeOrZero (target - eye);
    const Vec3 r = pt::normalizeOrZero (pt::cross (f, { 0.0f, 1.0f, 0.0f }));
    const Vec3 u = pt::cross (r, f);
    const Vec3 b = -f;
    // Columns r, u, b map local x, y, z.
    const float m00 = r.x, m01 = u.x, m02 = b.x;
    const float m10 = r.y, m11 = u.y, m12 = b.y;
    const float m20 = r.z, m21 = u.z, m22 = b.z;
    const float trace = m00 + m11 + m22;
    grid::SceneQuaternion q;
    if (trace > 0.0f)
    {
        const float s = std::sqrt (trace + 1.0f) * 2.0f;
        q = { (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25f * s };
    }
    else if (m00 > m11 && m00 > m22)
    {
        const float s = std::sqrt (1.0f + m00 - m11 - m22) * 2.0f;
        q = { 0.25f * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s };
    }
    else if (m11 > m22)
    {
        const float s = std::sqrt (1.0f + m11 - m00 - m22) * 2.0f;
        q = { (m01 + m10) / s, 0.25f * s, (m12 + m21) / s, (m02 - m20) / s };
    }
    else
    {
        const float s = std::sqrt (1.0f + m22 - m00 - m11) * 2.0f;
        q = { (m02 + m20) / s, (m12 + m21) / s, 0.25f * s, (m10 - m01) / s };
    }
    return q;
}

void cameraTests()
{
    auto scene = emptyScene();
    addCube (*scene, 1, {}, 0, 1);
    // Ground plane at y = -1.
    {
        const std::size_t index = scene->objectCount++;
        auto& object = scene->objects[index];
        object.id.value = 2;
        object.material.value = 1;
        object.firstVertex = static_cast<std::uint32_t> (scene->vertexCount);
        object.vertexCount = 4;
        object.firstIndex = static_cast<std::uint32_t> (scene->indexCount);
        object.indexCount = 6;
        const float e = 50.0f;
        const std::array<grid::SceneVec3, 4> corners {{ { -e, -1.0f, -e }, { e, -1.0f, -e }, { e, -1.0f, e }, { -e, -1.0f, e } }};
        for (const auto& c : corners)
            scene->vertices[scene->vertexCount++].position = c;
        for (const std::uint32_t i : { 0u, 2u, 1u, 0u, 3u, 2u })
            scene->indices[scene->indexCount++] = i;
    }
    scene->cameraCount = 1;
    scene->cameras[0].id.value = 7;
    scene->cameras[0].transform.translation = { 0.0f, 0.0f, 5.0f };
    scene->cameras[0].verticalFovRadians = 1.0471975512f;
    scene->cameras[0].nearPlane = 0.1f;
    scene->cameras[0].farPlane = 100.0f;

    pt::PinholeCamera camera;
    check (! pt::resolveActiveCamera (*scene, 64, 48, camera), "an unset active camera is refused");
    scene->activeCamera.value = 7;
    check (pt::resolveActiveCamera (*scene, 64, 48, camera), "the active camera resolves");
    check (approx (camera.aspect, 64.0f / 48.0f, 1.0e-6f) && approx (camera.tanHalfFov, std::tan (0.5235987756f), 1.0e-6f),
           "camera keeps the raster aspect and tan(fov / 2)");

    pt::IngestedScene ingested;
    pt::ingestScene (*scene, ingested);
    pt::Bvh bvh;
    bvh.build (ingested.positions);

    float px = 0.0f, py = 0.0f, depth = 0.0f;
    check (camera.project ({}, px, py, depth) && approx (px, 32.0f, 1.0e-4f) && approx (py, 24.0f, 1.0e-4f)
               && approx (depth, 5.0f, 1.0e-5f),
           "the scene centre projects to the image centre");
    pt::Hit hit;
    pt::Ray centre = camera.generateRay (32.0f, 24.0f);
    check (bvh.intersect (centre, hit) && approx (hit.t, 4.5f, 1.0e-4f)
               && nearVec (centre.origin + centre.direction * hit.t, { 0.0f, 0.0f, 0.5f }, 1.0e-4f),
           "the centre ray hits the cube's front face at the scene centre");
    check (camera.project ({ 1.0f, 0.0f, 4.0f }, px, py, depth) && px > 32.0f,
           "+x projects to the right half");
    check (camera.project ({ 0.0f, 1.0f, 4.0f }, px, py, depth) && py < 24.0f,
           "+y projects to the top half (row 0 is the top row)");
    check (! camera.project ({ 0.0f, 0.0f, 6.0f }, px, py, depth), "points behind the eye do not project");

    // Off-axis camera aimed at the origin.
    const Vec3 eye { 3.0f, 2.0f, 4.0f };
    scene->cameras[0].transform.translation = { eye.x, eye.y, eye.z };
    scene->cameras[0].transform.rotation = lookAtRotation (eye, {});
    check (pt::resolveActiveCamera (*scene, 64, 48, camera), "the rotated camera resolves");
    check (nearVec (pt::rotate (camera.rotation, { 0.0f, 0.0f, -1.0f }), pt::normalizeOrZero (-eye), 1.0e-5f),
           "look-at rotation maps local -Z to the view direction");
    check (camera.project ({}, px, py, depth) && approx (px, 32.0f, 1.0e-3f) && approx (py, 24.0f, 1.0e-3f),
           "the rotated camera frames the origin at the image centre");
    centre = camera.generateRay (32.0f, 24.0f);
    check (pt::dot (centre.direction, pt::normalizeOrZero (-eye)) > 1.0f - 1.0e-6f && bvh.intersect (centre, hit),
           "the rotated centre ray runs through the scene centre and hits the cube");

    // Every traced point reprojects to the pixel that generated its ray.
    std::mt19937 rng (0xca3e7a5u);
    std::uint32_t reprojectionErrors = 0;
    std::uint32_t nearErrors = 0;
    std::uint32_t hits = 0;
    for (int i = 0; i < 500; ++i)
    {
        const float x = uniform (rng, 0.0f, 64.0f);
        const float y = uniform (rng, 0.0f, 48.0f);
        const pt::Ray ray = camera.generateRay (x, y);
        check (approx (pt::length (ray.direction), 1.0f, 1.0e-5f), "camera directions are unit length");
        if (camera.project (ray.origin + ray.direction * ray.tMin, px, py, depth))
            nearErrors += approx (depth, camera.nearPlane, 1.0e-5f) ? 0u : 1u;
        else
            ++nearErrors;
        if (! bvh.intersect (ray, hit))
            continue;
        ++hits;
        if (! camera.project (ray.origin + ray.direction * hit.t, px, py, depth)
            || ! approx (px, x, 1.0e-2f) || ! approx (py, y, 1.0e-2f))
            ++reprojectionErrors;
    }
    check (hits > 100, "random camera rays hit the scene");
    check (reprojectionErrors == 0, "hit points reproject to their pixel through the raster transform");
    check (nearErrors == 0, "ray tMin lies on the raster near plane");

    grid::SceneCameraRecord bad;
    bad.verticalFovRadians = 0.0f;
    check (! pt::makePinholeCamera (bad, 64, 48, camera), "zero field of view is refused");
    bad = {};
    bad.farPlane = bad.nearPlane;
    check (! pt::makePinholeCamera (bad, 64, 48, camera), "far must exceed near");
    bad = {};
    check (! pt::makePinholeCamera (bad, 0, 48, camera), "zero-width image is refused");
}

void threadPoolTests()
{
    const pt::TileGrid grid (1000, 700, 32);
    check (grid.tileCount() == 32u * 22u, "tile grid counts clipped edge tiles");
    const pt::Tile last = grid.tile (grid.tileCount() - 1);
    check (last.x1 == 1000 && last.y1 == 700 && last.x0 == 992 && last.y0 == 672, "edge tiles clip to the image");

    for (const unsigned threads : { 1u, 2u, 4u, 0u })
    {
        pt::ThreadPool pool (threads);
        const std::string label = "pool(" + std::to_string (threads) + ")";
        check (pool.concurrency() >= 1 && (threads == 0 || pool.concurrency() == threads), label + " concurrency");
        for (int pass = 0; pass < 3; ++pass)
        {
            std::vector<std::atomic<int>> tileRuns (grid.tileCount());
            std::vector<std::uint8_t> pixels (std::size_t { grid.width } * grid.height, 0);
            std::atomic<bool> badThread { false };
            pt::CancellationToken token;
            const auto result = pool.runTiles (grid, token, [&] (const pt::Tile& tile, unsigned threadIndex) {
                tileRuns[tile.index].fetch_add (1);
                if (threadIndex >= pool.concurrency())
                    badThread = true;
                for (std::uint32_t y = tile.y0; y < tile.y1; ++y)
                    for (std::uint32_t x = tile.x0; x < tile.x1; ++x)
                        ++pixels[std::size_t { y } * grid.width + x];
            });
            bool once = true;
            for (const auto& runs : tileRuns)
                once = once && runs.load() == 1;
            check (once, label + ": every tile runs exactly once");
            check (std::all_of (pixels.begin(), pixels.end(), [] (std::uint8_t v) { return v == 1; }),
                   label + ": every pixel is covered exactly once");
            check (result.tilesCompleted == grid.tileCount() && ! result.cancelled, label + ": full pass reports completion");
            check (! badThread, label + ": thread indices stay below concurrency()");
        }

        // Cancellation part way through a pass.
        pt::CancellationToken token;
        std::vector<std::atomic<int>> tileRuns (grid.tileCount());
        std::atomic<std::uint32_t> started { 0 };
        std::atomic<std::uint32_t> startedAfterCancel { 0 };
        const auto cancelled = pool.runTiles (grid, token, [&] (const pt::Tile& tile, unsigned) {
            // A thread can pass the scheduler's check just before the cancel
            // lands, then see it here: at most once per thread.
            if (token.isCancelled())
                startedAfterCancel.fetch_add (1);
            tileRuns[tile.index].fetch_add (1);
            if (started.fetch_add (1) == 10)
                token.cancel();
        });
        bool atMostOnce = true;
        for (const auto& runs : tileRuns)
            atMostOnce = atMostOnce && runs.load() <= 1;
        check (cancelled.cancelled && cancelled.tilesCompleted < grid.tileCount() && cancelled.tilesCompleted >= 11,
               label + ": cancellation stops the pass early");
        check (startedAfterCancel.load() <= pool.concurrency(), label + ": cancellation stops within one tile per thread");
        check (atMostOnce, label + ": no tile runs twice under cancellation");

        // An already-cancelled token runs nothing; reset re-arms it.
        std::atomic<std::uint32_t> ran { 0 };
        const auto none = pool.runTiles (grid, token, [&] (const pt::Tile&, unsigned) { ++ran; });
        check (ran == 0 && none.tilesCompleted == 0 && none.cancelled, label + ": a cancelled token runs no tiles");
        token.reset();
        const auto again = pool.runTiles (grid, token, [&] (const pt::Tile&, unsigned) { ++ran; });
        check (ran == grid.tileCount() && again.tilesCompleted == grid.tileCount(), label + ": reset re-arms the token");

        // A throwing tile stops the pass and the pool stays usable.
        bool threw = false;
        try
        {
            pool.runTiles (grid, token, [&] (const pt::Tile& tile, unsigned) {
                if (tile.index == 5)
                    throw std::runtime_error ("tile failure");
            });
        }
        catch (const std::runtime_error&)
        {
            threw = true;
        }
        check (threw, label + ": a tile exception reaches the caller");
        ran = 0;
        pool.runTiles (grid, token, [&] (const pt::Tile&, unsigned) { ++ran; });
        check (ran == grid.tileCount(), label + ": the pool runs again after an exception");

        // An empty image is a valid, empty pass.
        const auto empty = pool.runTiles (pt::TileGrid (0, 0), token, [&] (const pt::Tile&, unsigned) { ++ran; });
        check (empty.tileCount == 0 && empty.tilesCompleted == 0 && ! empty.cancelled, label + ": empty grid");
    }
}

void accumulatorTests()
{
    // Streams are deterministic per (pixel, sample, generation) and distinct across them.
    {
        pt::PixelRng a (17, 3, 0x100000002ull), b (17, 3, 0x100000002ull);
        pt::PixelRng c (17, 3, 0x100000003ull), d (18, 3, 0x100000002ull), e (17, 4, 0x100000002ull);
        bool same = true;
        for (int i = 0; i < 16; ++i)
            same = same && a.nextUint() == b.nextUint();
        check (same, "equal seeds give equal streams");
        pt::PixelRng base (17, 3, 0x100000002ull);
        const std::uint32_t first = base.nextUint();
        check (first != c.nextUint() && first != d.nextUint() && first != e.nextUint(),
               "generation, pixel and sample each change the stream");
        pt::PixelRng stream (1, 2, 3);
        double sum = 0.0;
        float lo = 1.0f, hi = 0.0f;
        for (int i = 0; i < 200000; ++i)
        {
            const float u = stream.nextFloat();
            sum += u;
            lo = std::min (lo, u);
            hi = std::max (hi, u);
        }
        check (approx (static_cast<float> (sum / 200000.0), 0.5f, 0.005f) && lo >= 0.0f && hi < 1.0f,
               "nextFloat is uniform on [0, 1)");
    }

    pt::Accumulator accumulator;
    accumulator.resize (8, 4);
    accumulator.reset (42);
    check (accumulator.generation() == 42 && accumulator.sampleCount (3, 2) == 0, "reset starts an empty generation");
    const std::uint32_t samples = 20000;
    for (std::uint32_t y = 0; y < 4; ++y)
        for (std::uint32_t x = 0; x < 8; ++x)
            for (std::uint32_t s = 0; s < samples; ++s)
            {
                pt::PixelRng rng (y * 8 + x, s, accumulator.generation());
                const float u = rng.nextFloat();
                accumulator.addSample (x, y, { u, 2.0f * u, 3.0f });
            }
    bool converged = true;
    for (std::uint32_t y = 0; y < 4; ++y)
        for (std::uint32_t x = 0; x < 8; ++x)
        {
            const pt::Rgb m = accumulator.mean (x, y);
            converged = converged && accumulator.sampleCount (x, y) == samples
                && approx (m.r, 0.5f, 0.015f) && approx (m.g, 1.0f, 0.03f) && m.b == 3.0f;
        }
    check (converged, "the running mean converges to the expected value");

    // Resolve with a padded pitch leaves the padding alone.
    const std::size_t pitch = 8 * pt::Accumulator::kResolvedBytesPerPixel + 32;
    std::vector<unsigned char> buffer (pitch * 4, 0xab);
    check (accumulator.resolveRgba32f (buffer.data(), pitch), "resolve into a padded pitch");
    bool layout = true;
    for (std::uint32_t y = 0; y < 4; ++y)
    {
        for (std::uint32_t x = 0; x < 8; ++x)
        {
            float rgba[4] {};
            std::memcpy (rgba, buffer.data() + y * pitch + x * 16, sizeof rgba);
            const pt::Rgb m = accumulator.mean (x, y);
            layout = layout && rgba[0] == m.r && rgba[1] == m.g && rgba[2] == m.b && rgba[3] == 1.0f;
        }
        for (std::size_t pad = 8 * 16; pad < pitch; ++pad)
            layout = layout && buffer[y * pitch + pad] == 0xab;
    }
    check (layout, "resolved rows are RGBA32F at the requested pitch");
    check (! accumulator.resolveRgba32f (buffer.data(), 8 * 16 - 1), "a short pitch is refused");
    check (! accumulator.resolveRgba32f (nullptr), "a null destination is refused");

    accumulator.addSample (0, 0, { std::nanf (""), 1.0f, std::numeric_limits<float>::infinity() });
    check (accumulator.rejectedSamples() == 1 && accumulator.sampleCount (0, 0) == samples + 1,
           "non-finite samples are counted as rejected zeros");
    accumulator.addSample (8, 0, { 1.0f, 1.0f, 1.0f });
    check (accumulator.sampleCount (7, 0) == samples, "out-of-range samples are ignored");

    accumulator.reset (43);
    const std::vector<float> cleared = accumulator.resolveRgba32f();
    check (cleared.size() == 8 * 4 * 4 && std::all_of (cleared.begin(), cleared.end(), [] (float v) { return v == 0.0f; }),
           "reset clears colour and alpha");
    check (accumulator.generation() == 43 && accumulator.rejectedSamples() == 0, "reset records the new generation");
}

// A small progressive render is bit-identical for any thread count.
void deterministicRenderTests()
{
    auto scene = emptyScene();
    grid::SceneTransform3D t;
    t.rotation = { 0.2f, 0.3f, 0.1f, 0.927362f };
    addCube (*scene, 1, t, 0, 1);
    scene->cameraCount = 1;
    scene->cameras[0].id.value = 3;
    scene->cameras[0].transform.translation = { 0.0f, 0.0f, 3.0f };
    scene->activeCamera.value = 3;
    pt::IngestedScene ingested;
    pt::ingestScene (*scene, ingested);
    pt::Bvh bvh;
    bvh.build (ingested.positions);
    pt::PinholeCamera camera;
    check (pt::resolveActiveCamera (*scene, 96, 64, camera), "render camera resolves");

    const auto render = [&] (unsigned threads) {
        pt::ThreadPool pool (threads);
        pt::Accumulator accumulator;
        accumulator.resize (96, 64);
        accumulator.reset (7);
        pt::CancellationToken token;
        const pt::TileGrid grid (96, 64, 16);
        for (std::uint32_t sample = 0; sample < 4; ++sample)
            pool.runTiles (grid, token, [&] (const pt::Tile& tile, unsigned) {
                for (std::uint32_t y = tile.y0; y < tile.y1; ++y)
                    for (std::uint32_t x = tile.x0; x < tile.x1; ++x)
                    {
                        pt::PixelRng rng (y * 96 + x, sample, accumulator.generation());
                        const float jx = rng.nextFloat(), jy = rng.nextFloat();
                        const pt::Ray ray = camera.generateRay (static_cast<float> (x) + jx, static_cast<float> (y) + jy);
                        pt::Hit hit;
                        pt::Rgb value;
                        if (bvh.intersect (ray, hit))
                        {
                            const Vec3 n = ingested.triangles[hit.triangle].geometricNormal;
                            // Shadow-style query toward a fixed light, offset off the surface.
                            pt::Ray shadow;
                            shadow.direction = pt::normalizeOrZero (Vec3 { 1.0f, 2.0f, 3.0f });
                            shadow.origin = pt::spawnRayOrigin (ray.origin + ray.direction * hit.t, n, shadow.direction);
                            shadow.ignoreTriangle = hit.triangle;
                            const float lit = bvh.occluded (shadow) ? 0.2f : 1.0f;
                            value = { lit * (0.5f + 0.5f * n.x), lit * (0.5f + 0.5f * n.y), lit * (0.5f + 0.5f * n.z) };
                        }
                        accumulator.addSample (x, y, value);
                    }
            });
        return accumulator.resolveRgba32f();
    };
    const std::vector<float> single = render (1);
    const std::vector<float> multi = render (4);
    check (single.size() == multi.size()
               && std::memcmp (single.data(), multi.data(), single.size() * sizeof (float)) == 0,
           "render is bit-identical across thread counts");
    const std::size_t centre = (32 * 96 + 48) * 4;
    check (single[centre] + single[centre + 1] + single[centre + 2] > 0.0f, "the centre pixel sees the cube");
    check (single[0] == 0.0f && single[3] == 1.0f, "a corner pixel sees the background");
}

} // namespace

int main()
{
    bvhTests();
    watertightTests();
    rayOffsetTests();
    ingestTests();
    cameraTests();
    threadPoolTests();
    accumulatorTests();
    deterministicRenderTests();
    std::cout << checks << " checks, " << failures << " failures\n";
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
