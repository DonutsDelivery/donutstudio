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

#include "devices/device_annotations.h"
#include "vec_math.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace videohelper::pathtrace
{

constexpr std::uint32_t kNoTriangle = 0xffffffffu;

// A ray covers origin + t * direction for tMin < t < tMax. The direction need
// not be unit length; t is in units of the direction. ignoreTriangle skips one
// triangle, usually the one a secondary ray starts on. time is the ray's
// moment in the shutter, 0 at the open and 1 at the close keys, where moving
// triangles are placed (CPU motion BVH and the OptiX motion GAS alike).
struct Ray
{
    Vec3 origin {};
    Vec3 direction { 0.0f, 0.0f, -1.0f };
    float tMin = 0.0f;
    float tMax = std::numeric_limits<float>::infinity();
    std::uint32_t ignoreTriangle = kNoTriangle;
    float time = 0.0f;
};

// Barycentrics: the hit point is (1 - u - v) * p0 + u * p1 + v * p2.
// triangle is the caller's index into the triangle list given to build().
// time copies the ray's shutter time, so the hit point can be rebuilt from
// the triangle's vertices at that moment.
struct Hit
{
    float t = std::numeric_limits<float>::infinity();
    float u = 0.0f;
    float v = 0.0f;
    std::uint32_t triangle = kNoTriangle;
    float time = 0.0f;

    PT_SHARED_FN bool valid() const noexcept { return triangle != kNoTriangle; }
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

// prepareRay and intersectTriangle, the ray setup and the watertight triangle
// test, are in bvh_traversal.h, which every device compiles.

// Linear vertex motion: the vertex at shutter time `time` between its open and
// close positions. Rigid transforms interpolated elementwise move every vertex
// along this line, so the lerp is exact for them.
PT_SHARED_FN Vec3 motionLerp (Vec3 open, Vec3 close, float time) noexcept
{
    return open + (close - open) * time;
}

// Bounds of one triangle over the whole shutter: the union of its open and
// close triangles, which contains every in-between triangle because each
// vertex moves on a straight line.
inline Bounds motionTriangleBounds (const Vec3* open, const Vec3* close) noexcept
{
    Bounds b;
    for (int k = 0; k < 3; ++k)
    {
        b.grow (open[k]);
        if (close != nullptr)
            b.grow (close[k]);
    }
    return b;
}

// Moves a surface point off its triangle along n, the geometric normal facing
// the side the new ray leaves from. The step grows with |p|, so it stays a few
// float ulps at any scale.
PT_SHARED_FN Vec3 offsetRayOrigin (Vec3 p, Vec3 n) noexcept
{
    // Near the origin a fixed float step; elsewhere an integer step in the
    // float's bit pattern, which scales with the exponent.
    constexpr float originRegion = 1.0f / 32.0f;
    constexpr float floatStep = 1.0f / 65536.0f;
    constexpr float intStep = 256.0f;
    Vec3 out;
    for (int axis = 0; axis < 3; ++axis)
    {
        const float pc = p[axis];
        const float nc = n[axis];
        if (std::fabs (pc) < originRegion)
        {
            out[axis] = pc + floatStep * nc;
            continue;
        }
        const auto steps = static_cast<std::int32_t> (intStep * nc);
        const auto bits = static_cast<std::int32_t> (floatBits (pc));
        const std::int32_t moved = bits + (pc < 0.0f ? -steps : steps);
        out[axis] = bitsToFloat (static_cast<std::uint32_t> (moved));
    }
    return out;
}

// Origin for a ray leaving a surface at p in direction dir. Flips ng toward dir.
PT_SHARED_FN Vec3 spawnRayOrigin (Vec3 p, Vec3 ng, Vec3 dir) noexcept
{
    return offsetRayOrigin (p, dot (ng, dir) < 0.0f ? -ng : ng);
}

// Host only: the Metal kernel (device_metal.metal) leaves this out.
#if ! defined(PATHTRACE_DEVICE_METAL)
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

// A built Bvh's arrays, on the host or copied to a device. Leaf order: the
// ordered triangle i has vertices orderedPositions[3i..3i+2] and the caller's
// index triangleIndices[i]. bvh_traversal.h traverses it, placing a moving
// triangle's vertices at ray.time between orderedPositions and orderedClose.
struct BvhView
{
    const BvhNode* nodes = nullptr;
    std::size_t nodeCount = 0;
    const Vec3* orderedPositions = nullptr;
    const std::uint32_t* triangleIndices = nullptr;
    // Close-of-shutter vertices in leaf order, or null for a static build.
    const Vec3* orderedClose = nullptr;
};

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
    // Motion build: open and close hold three vertices per triangle at shutter
    // open and close. Node bounds cover the whole sweep and the leaf test
    // places each vertex at ray.time. close == nullptr is a static build.
    void buildMotion (const Vec3* open, const Vec3* close, std::size_t triangleCount,
                      const BvhBuildOptions& options = {});
    void clear() noexcept;
    bool hasMotion() const noexcept { return ! orderedClose_.empty(); }

    // Closest hit in (ray.tMin, ray.tMax). Leaves hit unchanged on a miss.
    bool intersect (const Ray& ray, Hit& hit) const noexcept;
    // True when any triangle lies in (ray.tMin, ray.tMax).
    bool occluded (const Ray& ray) const noexcept;

    std::size_t triangleCount() const noexcept { return triangleIndices_.size(); }
    const std::vector<BvhNode>& nodes() const noexcept { return nodes_; }
    // The arrays a device copies; a GPU device uploads each and traverses the copy.
    BvhView view() const noexcept
    {
        return { nodes_.data(), nodes_.size(), orderedPositions_.data(), triangleIndices_.data(),
                 orderedClose_.empty() ? nullptr : orderedClose_.data() };
    }
    const std::vector<Vec3>& orderedPositions() const noexcept { return orderedPositions_; }
    const std::vector<Vec3>& orderedClose() const noexcept { return orderedClose_; }
    const std::vector<std::uint32_t>& triangleIndices() const noexcept { return triangleIndices_; }
    Bounds bounds() const noexcept;
    const BvhStats& stats() const noexcept { return stats_; }

private:
    std::vector<BvhNode> nodes_;
    // Leaf order. orderedPositions_ holds three vertices per ordered triangle.
    std::vector<Vec3> orderedPositions_;
    // Close-of-shutter vertices in the same order; empty without motion.
    std::vector<Vec3> orderedClose_;
    std::vector<std::uint32_t> triangleIndices_;
    BvhStats stats_ {};
};
#endif

} // namespace videohelper::pathtrace

// Metal traces against its own acceleration structure.
#if ! defined(PATHTRACE_DEVICE_METAL)
#include "bvh_traversal.h"
#endif
