#pragma once

// Traversal of the two-wide BVH (bvh.h), written once for every device. The
// CPU Bvh traverses through it, and the CUDA and HIP devices run the same code
// over a copy of the CPU build uploaded to device memory (BvhView). OptiX
// traverses its own GAS instead.
//
// The triangle test follows Woop, Benthin and Wald, "Watertight Ray/Triangle
// Intersection", JCGT 2(1), 2013, as bvh.h describes.

#include "bvh.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace videohelper::pathtrace
{

// Widens a box's far distance so float rounding in the slab test never culls a
// box the ray touches: 1 + 2 * gamma(3) with gamma(n) = n u / (1 - n u).
inline constexpr float kSlabRobustness = 1.0f + 2.0f * (3.0f * 5.9604645e-8f) / (1.0f - 3.0f * 5.9604645e-8f);

PT_SHARED_FN int largestAxis (Vec3 e) noexcept
{
    return e.x >= e.y && e.x >= e.z ? 0 : e.y >= e.z ? 1 : 2;
}

// False when the origin or direction is not finite or the direction is zero.
PT_SHARED_FN bool prepareRay (const Ray& ray, PreparedRay& prepared) noexcept
{
    if (! isFinite (ray.origin) || ! isFinite (ray.direction) || std::isnan (ray.tMin) || std::isnan (ray.tMax))
        return false;
    const Vec3 d = ray.direction;
    const Vec3 a { std::fabs (d.x), std::fabs (d.y), std::fabs (d.z) };
    const int kz = largestAxis (a);
    if (! (a[kz] > 0.0f))
        return false;
    int kx = kz == 2 ? 0 : kz + 1;
    int ky = kx == 2 ? 0 : kx + 1;
    // Keep the winding of the projected triangle when the major axis is negative.
    if (d[kz] < 0.0f)
    {
        const int swapped = kx;
        kx = ky;
        ky = swapped;
    }
    prepared.origin = ray.origin;
    prepared.direction = d;
    prepared.kx = kx;
    prepared.ky = ky;
    prepared.kz = kz;
    prepared.shearX = d[kx] / d[kz];
    prepared.shearY = d[ky] / d[kz];
    prepared.shearZ = 1.0f / d[kz];
    // A zero component becomes a tiny one, so the slab test never forms 0 * inf.
    constexpr float tiny = 1.0e-30f;
    for (int axis = 0; axis < 3; ++axis)
    {
        const float c = d[axis];
        prepared.inverseDirection[axis] = 1.0f / (std::fabs (c) < tiny ? std::copysign (tiny, c) : c);
    }
    return true;
}

// Watertight test of one triangle against tMin < t < tMax. Both faces hit.
// On a hit, writes t, u and v (triangle is left unchanged).
PT_SHARED_FN bool intersectTriangle (const PreparedRay& ray, Vec3 p0, Vec3 p1, Vec3 p2,
                                     float tMin, float tMax, Hit& hit) noexcept
{
    // Vertices relative to the origin, sheared so the ray runs along +z.
    const Vec3 a = p0 - ray.origin;
    const Vec3 b = p1 - ray.origin;
    const Vec3 c = p2 - ray.origin;
    const int kx = ray.kx, ky = ray.ky, kz = ray.kz;
    const float ax = a[kx] - ray.shearX * a[kz];
    const float ay = a[ky] - ray.shearY * a[kz];
    const float bx = b[kx] - ray.shearX * b[kz];
    const float by = b[ky] - ray.shearY * b[kz];
    const float cx = c[kx] - ray.shearX * c[kz];
    const float cy = c[ky] - ray.shearY * c[kz];

    // 2D edge functions. Each is the weight of the opposite vertex.
    float e0 = cx * by - cy * bx;
    float e1 = ax * cy - ay * cx;
    float e2 = bx * ay - by * ax;
    // A float zero may be a rounding artefact on an edge; decide it in double.
    if (e0 == 0.0f || e1 == 0.0f || e2 == 0.0f)
    {
        e0 = static_cast<float> (static_cast<double> (cx) * by - static_cast<double> (cy) * bx);
        e1 = static_cast<float> (static_cast<double> (ax) * cy - static_cast<double> (ay) * cx);
        e2 = static_cast<float> (static_cast<double> (bx) * ay - static_cast<double> (by) * ax);
    }
    if ((e0 < 0.0f || e1 < 0.0f || e2 < 0.0f) && (e0 > 0.0f || e1 > 0.0f || e2 > 0.0f))
        return false;
    float det = e0 + e1 + e2;
    if (det == 0.0f || ! std::isfinite (det))
        return false;

    const float az = ray.shearZ * a[kz];
    const float bz = ray.shearZ * b[kz];
    const float cz = ray.shearZ * c[kz];
    float scaledT = e0 * az + e1 * bz + e2 * cz;
    // Compare t against the interval before dividing.
    if (det < 0.0f)
    {
        det = -det;
        scaledT = -scaledT;
        e0 = -e0; e1 = -e1; e2 = -e2;
    }
    if (! (scaledT > tMin * det) || ! (scaledT < tMax * det))
        return false;
    const float inv = 1.0f / det;
    const float t = scaledT * inv;
    if (! (t > tMin) || ! (t < tMax))
        return false;
    hit.t = t;
    hit.u = e1 * inv;
    hit.v = e2 * inv;
    return true;
}

PT_SHARED_FN bool hitBox (const BvhNode& node, const PreparedRay& ray, float tMin, float tMax, float& entry) noexcept
{
    float nearT = tMin;
    float farT = tMax;
    for (int axis = 0; axis < 3; ++axis)
    {
        const float o = ray.origin[axis];
        const float inv = ray.inverseDirection[axis];
        float t0 = (node.boundsMin[axis] - o) * inv;
        float t1 = (node.boundsMax[axis] - o) * inv;
        if (t0 > t1)
        {
            const float swapped = t0;
            t0 = t1;
            t1 = swapped;
        }
        nearT = t0 > nearT ? t0 : nearT;
        t1 *= kSlabRobustness;
        farT = t1 < farT ? t1 : farT;
    }
    entry = nearT;
    return nearT <= farT;
}

// Ordered triangle i at shutter time `time`: the still vertices, or each
// vertex on its line from open to close for a motion build.
PT_SHARED_FN bool leafTriangle (const BvhView& bvh, const PreparedRay& prepared, float time, std::uint32_t i,
                                float tMin, float tMax, Hit& candidate) noexcept
{
    const std::size_t base = 3 * std::size_t { i };
    const Vec3* open = bvh.orderedPositions;
    if (bvh.orderedClose == nullptr)
        return intersectTriangle (prepared, open[base], open[base + 1], open[base + 2], tMin, tMax, candidate);
    const Vec3* close = bvh.orderedClose;
    return intersectTriangle (prepared, motionLerp (open[base], close[base], time),
                              motionLerp (open[base + 1], close[base + 1], time),
                              motionLerp (open[base + 2], close[base + 2], time), tMin, tMax, candidate);
}

struct BvhStackEntry
{
    std::uint32_t node;
    float entry;
};

// Closest hit in (ray.tMin, ray.tMax), skipping ray.ignoreTriangle. Leaves hit
// unchanged on a miss.
PT_SHARED_FN bool traverseClosest (const BvhView& bvh, const Ray& ray, Hit& hit) noexcept
{
    PreparedRay prepared;
    if (bvh.nodeCount == 0 || ! prepareRay (ray, prepared))
        return false;
    const float tMin = ray.tMin;
    float tMax = ray.tMax;
    float entry = 0.0f;
    if (! hitBox (bvh.nodes[0], prepared, tMin, tMax, entry))
        return false;

    // One entry per level: the build never goes deeper than Bvh::kMaxDepth.
    BvhStackEntry stack[Bvh::kMaxDepth + 1];
    std::size_t depth = 0;
    std::uint32_t current = 0;
    Hit best;
    for (;;)
    {
        const BvhNode& node = bvh.nodes[current];
        if (node.count != 0)
        {
            const std::uint32_t end = node.firstOrLeft + node.count;
            for (std::uint32_t i = node.firstOrLeft; i < end; ++i)
            {
                const std::uint32_t triangle = bvh.triangleIndices[i];
                if (triangle == ray.ignoreTriangle)
                    continue;
                Hit candidate;
                if (leafTriangle (bvh, prepared, ray.time, i, tMin, tMax, candidate))
                {
                    tMax = candidate.t;
                    best = candidate;
                    best.triangle = triangle;
                    best.time = ray.time;
                }
            }
        }
        else
        {
            const std::uint32_t left = node.firstOrLeft;
            float leftEntry = 0.0f;
            float rightEntry = 0.0f;
            const bool hitLeft = hitBox (bvh.nodes[left], prepared, tMin, tMax, leftEntry);
            const bool hitRight = hitBox (bvh.nodes[left + 1], prepared, tMin, tMax, rightEntry);
            if (hitLeft && hitRight)
            {
                // Visit the nearer child first and keep the other for later.
                const bool leftFirst = leftEntry <= rightEntry;
                stack[depth++] = leftFirst ? BvhStackEntry { left + 1, rightEntry } : BvhStackEntry { left, leftEntry };
                current = leftFirst ? left : left + 1;
                continue;
            }
            if (hitLeft || hitRight)
            {
                current = hitLeft ? left : left + 1;
                continue;
            }
        }
        // Pop the next subtree that can still beat the closest hit.
        bool resumed = false;
        while (depth > 0)
        {
            const BvhStackEntry next = stack[--depth];
            if (next.entry <= tMax)
            {
                current = next.node;
                resumed = true;
                break;
            }
        }
        if (! resumed)
            break;
    }
    if (! best.valid())
        return false;
    hit = best;
    return true;
}

// True when any triangle other than ray.ignoreTriangle lies in (ray.tMin, ray.tMax).
PT_SHARED_FN bool traverseAny (const BvhView& bvh, const Ray& ray) noexcept
{
    PreparedRay prepared;
    if (bvh.nodeCount == 0 || ! prepareRay (ray, prepared))
        return false;
    const float tMin = ray.tMin;
    const float tMax = ray.tMax;
    float entry = 0.0f;
    if (! hitBox (bvh.nodes[0], prepared, tMin, tMax, entry))
        return false;

    std::uint32_t stack[Bvh::kMaxDepth + 1];
    std::size_t depth = 0;
    std::uint32_t current = 0;
    for (;;)
    {
        const BvhNode& node = bvh.nodes[current];
        if (node.count != 0)
        {
            const std::uint32_t end = node.firstOrLeft + node.count;
            for (std::uint32_t i = node.firstOrLeft; i < end; ++i)
            {
                if (bvh.triangleIndices[i] == ray.ignoreTriangle)
                    continue;
                Hit candidate;
                if (leafTriangle (bvh, prepared, ray.time, i, tMin, tMax, candidate))
                    return true;
            }
        }
        else
        {
            const std::uint32_t left = node.firstOrLeft;
            const bool hitLeft = hitBox (bvh.nodes[left], prepared, tMin, tMax, entry);
            const bool hitRight = hitBox (bvh.nodes[left + 1], prepared, tMin, tMax, entry);
            if (hitLeft && hitRight)
            {
                stack[depth++] = left + 1;
                current = left;
                continue;
            }
            if (hitLeft || hitRight)
            {
                current = hitLeft ? left : left + 1;
                continue;
            }
        }
        if (depth == 0)
            return false;
        current = stack[--depth];
    }
}

} // namespace videohelper::pathtrace
