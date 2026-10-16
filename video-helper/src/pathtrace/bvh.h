#pragma once

// Two-wide bounding volume hierarchy over a flat world-space triangle list,
// built with a binned surface area heuristic. Closest-hit and any-hit
// traversal use a watertight ray-triangle test, so a ray that passes through a
// shared edge or vertex of a closed mesh always reports a hit.
//
// Behavioural reference (studied, not copied): Blender Cycles
// intern/cycles/bvh/binning.cpp, intern/cycles/bvh/build.cpp and
// intern/cycles/bvh/params.h for bin count, SAH costs and leaf rules;
// intern/cycles/util/math_intersect.h for the edge-function triangle test.
// The triangle test itself follows Woop, Benthin and Wald, "Watertight
// Ray/Triangle Intersection", JCGT 2(1), 2013. The secondary-ray origin offset
// follows Waechter and Binder, "A Fast and Robust Method for Avoiding
// Self-Intersection", Ray Tracing Gems, 2019.

#include "vec_math.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace videohelper::pathtrace
{

constexpr std::uint32_t kNoTriangle = 0xffffffffu;

// A ray covers origin + t * direction for tMin < t < tMax. The direction need
// not be unit length; t is in units of the direction. ignoreTriangle skips one
// triangle, usually the one a secondary ray starts on.
struct Ray
{
    Vec3 origin {};
    Vec3 direction { 0.0f, 0.0f, -1.0f };
    float tMin = 0.0f;
    float tMax = std::numeric_limits<float>::infinity();
    std::uint32_t ignoreTriangle = kNoTriangle;
};

// Barycentrics: the hit point is (1 - u - v) * p0 + u * p1 + v * p2.
// triangle is the caller's index into the triangle list given to build().
struct Hit
{
    float t = std::numeric_limits<float>::infinity();
    float u = 0.0f;
    float v = 0.0f;
    std::uint32_t triangle = kNoTriangle;

    bool valid() const noexcept { return triangle != kNoTriangle; }
};

// Per-ray constants shared by every box and triangle test of one query.
struct PreparedRay
{
    Vec3 origin {};
    Vec3 direction {};
    Vec3 inverseDirection {};
    int kx = 0;
    int ky = 1;
    int kz = 2;
    float shearX = 0.0f;
    float shearY = 0.0f;
    float shearZ = 1.0f;
};

// False when the origin or direction is not finite or the direction is zero.
bool prepareRay (const Ray& ray, PreparedRay& prepared) noexcept;

// Watertight test of one triangle against tMin < t < tMax. Both faces hit.
// On a hit, writes t, u and v (triangle is left unchanged).
bool intersectTriangle (const PreparedRay& ray, Vec3 p0, Vec3 p1, Vec3 p2,
                        float tMin, float tMax, Hit& hit) noexcept;

// Moves a surface point off its triangle along n, the geometric normal facing
// the side the new ray leaves from. The step grows with |p|, so it stays a few
// float ulps at any scale.
Vec3 offsetRayOrigin (Vec3 p, Vec3 n) noexcept;

// Origin for a ray leaving a surface at p in direction dir. Flips ng toward dir.
inline Vec3 spawnRayOrigin (Vec3 p, Vec3 ng, Vec3 dir) noexcept
{
    return offsetRayOrigin (p, dot (ng, dir) < 0.0f ? -ng : ng);
}

// 32 bytes. An interior node (count == 0) has children at firstOrLeft and
// firstOrLeft + 1. A leaf (count > 0) owns ordered triangles
// [firstOrLeft, firstOrLeft + count).
struct BvhNode
{
    float boundsMin[3];
    std::uint32_t firstOrLeft;
    float boundsMax[3];
    std::uint32_t count;
};
static_assert (sizeof (BvhNode) == 32, "BvhNode must stay 32 bytes");

struct BvhBuildOptions
{
    std::uint32_t maxLeafSize = 8;
    float nodeCost = 1.0f;
    float triangleCost = 1.0f;
};

struct BvhStats
{
    std::uint32_t nodeCount = 0;
    std::uint32_t leafCount = 0;
    std::uint32_t maxDepth = 0;
    std::uint32_t maxLeafSize = 0;
    std::uint32_t medianSplits = 0;
};

class Bvh
{
public:
    // Deepest leaf the build may produce. Traversal keeps one stack entry per
    // level, so this bounds the traversal stack.
    static constexpr std::uint32_t kMaxDepth = 64;

    // positions holds three world-space vertices per triangle. Rebuilding
    // replaces any earlier hierarchy. Triangles keep their caller indices.
    void build (const std::vector<Vec3>& positions, const BvhBuildOptions& options = {});
    void build (const Vec3* positions, std::size_t triangleCount, const BvhBuildOptions& options = {});
    void clear() noexcept;

    // Closest hit in (ray.tMin, ray.tMax). Leaves hit unchanged on a miss.
    bool intersect (const Ray& ray, Hit& hit) const noexcept;
    // True when any triangle lies in (ray.tMin, ray.tMax).
    bool occluded (const Ray& ray) const noexcept;

    std::size_t triangleCount() const noexcept { return triangleIndices_.size(); }
    const std::vector<BvhNode>& nodes() const noexcept { return nodes_; }
    Bounds bounds() const noexcept;
    const BvhStats& stats() const noexcept { return stats_; }

private:
    std::vector<BvhNode> nodes_;
    // Leaf order. orderedPositions_ holds three vertices per ordered triangle.
    std::vector<Vec3> orderedPositions_;
    std::vector<std::uint32_t> triangleIndices_;
    BvhStats stats_ {};
};

} // namespace videohelper::pathtrace
