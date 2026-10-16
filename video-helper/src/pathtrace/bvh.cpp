#include "bvh.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

// Behavioural reference: intern/cycles/bvh/binning.cpp, intern/cycles/bvh/build.cpp,
// intern/cycles/bvh/params.h (Blender Cycles). See bvh.h for the papers.

namespace videohelper::pathtrace
{
namespace
{

constexpr std::uint32_t kMaxBins = 32;

// Widens a box's far distance so float rounding in the slab test never culls a
// box the ray touches: 1 + 2 * gamma(3) with gamma(n) = n u / (1 - n u).
constexpr float kSlabRobustness = 1.0f + 2.0f * (3.0f * 5.9604645e-8f) / (1.0f - 3.0f * 5.9604645e-8f);

int largestAxis (Vec3 e) noexcept
{
    return e.x >= e.y && e.x >= e.z ? 0 : e.y >= e.z ? 1 : 2;
}

struct BuildTask
{
    std::uint32_t node;
    std::uint32_t begin;
    std::uint32_t end;
    std::uint32_t depth;
};

struct Bin
{
    Bounds bounds;
    std::uint32_t count = 0;
};

class Builder
{
public:
    Builder (const Vec3* positions, std::size_t triangleCount, const BvhBuildOptions& options,
             std::vector<BvhNode>& nodes, std::vector<std::uint32_t>& order, BvhStats& stats)
        : options_ (options), nodes_ (nodes), order_ (order), stats_ (stats)
    {
        triangleBounds_.resize (triangleCount);
        centroids_.resize (triangleCount);
        order_.resize (triangleCount);
        for (std::size_t i = 0; i < triangleCount; ++i)
        {
            Bounds b;
            b.grow (positions[3 * i]);
            b.grow (positions[3 * i + 1]);
            b.grow (positions[3 * i + 2]);
            triangleBounds_[i] = b;
            centroids_[i] = b.centre();
            order_[i] = static_cast<std::uint32_t> (i);
        }
        if (options_.maxLeafSize == 0)
            options_.maxLeafSize = 1;
        // From this depth on, splits are object medians. A median split halves
        // the range, so the deepest leaf stays below Bvh::kMaxDepth.
        std::uint32_t halvings = 0;
        while ((std::size_t { 1 } << halvings) < triangleCount && halvings < 62)
            ++halvings;
        medianDepth_ = Bvh::kMaxDepth - 1 > halvings ? Bvh::kMaxDepth - 1 - halvings : 0;
    }

    void run()
    {
        const auto count = static_cast<std::uint32_t> (order_.size());
        nodes_.clear();
        nodes_.reserve (count == 0 ? 0 : 2 * count);
        if (count == 0)
            return;
        nodes_.push_back ({});
        std::vector<BuildTask> tasks;
        tasks.push_back ({ 0, 0, count, 0 });
        while (! tasks.empty())
        {
            const BuildTask task = tasks.back();
            tasks.pop_back();
            process (task, tasks);
        }
        stats_.nodeCount = static_cast<std::uint32_t> (nodes_.size());
    }

private:
    void setBounds (std::uint32_t node, const Bounds& b)
    {
        auto& n = nodes_[node];
        n.boundsMin[0] = b.lo.x; n.boundsMin[1] = b.lo.y; n.boundsMin[2] = b.lo.z;
        n.boundsMax[0] = b.hi.x; n.boundsMax[1] = b.hi.y; n.boundsMax[2] = b.hi.z;
    }

    void makeLeaf (const BuildTask& task)
    {
        auto& n = nodes_[task.node];
        n.firstOrLeft = task.begin;
        n.count = task.end - task.begin;
        ++stats_.leafCount;
        stats_.maxDepth = std::max (stats_.maxDepth, task.depth);
        stats_.maxLeafSize = std::max (stats_.maxLeafSize, n.count);
    }

    static std::uint32_t binOf (float c, float lo, float scale, std::uint32_t binCount) noexcept
    {
        const float f = (c - lo) * scale;
        if (! (f > 0.0f))
            return 0;
        const auto b = static_cast<std::uint32_t> (f);
        return std::min (b, binCount - 1);
    }

    void process (const BuildTask& task, std::vector<BuildTask>& tasks)
    {
        Bounds bounds;
        Bounds centroidBounds;
        for (std::uint32_t i = task.begin; i < task.end; ++i)
        {
            bounds.grow (triangleBounds_[order_[i]]);
            centroidBounds.grow (centroids_[order_[i]]);
        }
        setBounds (task.node, bounds);
        const std::uint32_t count = task.end - task.begin;
        if (count <= 1)
        {
            makeLeaf (task);
            return;
        }

        // Binned SAH over all three axes. The bin count grows slowly with the
        // range size, as the reference builder does.
        const auto binCount = std::min<std::uint32_t> (
            kMaxBins, static_cast<std::uint32_t> (4.0f + 0.05f * static_cast<float> (count)));
        const Vec3 centroidExtent = centroidBounds.extent();
        float bestCost = std::numeric_limits<float>::infinity();
        int bestAxis = -1;
        std::uint32_t bestSplit = 0;
        if (task.depth < medianDepth_)
        {
            for (int axis = 0; axis < 3; ++axis)
            {
                const float extent = centroidExtent[axis];
                if (! (extent > 0.0f) || ! std::isfinite (extent))
                    continue;
                const float scale = static_cast<float> (binCount) / extent;
                const float lo = centroidBounds.lo[axis];
                std::array<Bin, kMaxBins> bins {};
                for (std::uint32_t i = task.begin; i < task.end; ++i)
                {
                    const auto t = order_[i];
                    auto& bin = bins[binOf (centroids_[t][axis], lo, scale, binCount)];
                    bin.bounds.grow (triangleBounds_[t]);
                    ++bin.count;
                }
                // Right-to-left prefix of area and count, then a left-to-right sweep.
                std::array<float, kMaxBins> rightArea {};
                std::array<std::uint32_t, kMaxBins> rightCount {};
                Bounds right;
                std::uint32_t rightN = 0;
                for (std::uint32_t b = binCount - 1; b > 0; --b)
                {
                    right.grow (bins[b].bounds);
                    rightN += bins[b].count;
                    rightArea[b] = right.halfArea();
                    rightCount[b] = rightN;
                }
                Bounds left;
                std::uint32_t leftN = 0;
                for (std::uint32_t b = 1; b < binCount; ++b)
                {
                    left.grow (bins[b - 1].bounds);
                    leftN += bins[b - 1].count;
                    if (leftN == 0 || rightCount[b] == 0)
                        continue;
                    const float cost = left.halfArea() * static_cast<float> (leftN)
                                     + rightArea[b] * static_cast<float> (rightCount[b]);
                    if (cost < bestCost)
                    {
                        bestCost = cost;
                        bestAxis = axis;
                        bestSplit = b;
                    }
                }
            }
        }

        const float area = bounds.halfArea();
        const float leafCost = options_.triangleCost * area * static_cast<float> (count);
        const float splitCost = options_.nodeCost * area + options_.triangleCost * bestCost;
        if (count <= options_.maxLeafSize && (bestAxis < 0 || leafCost <= splitCost))
        {
            // A small range with no binned split, or one SAH prefers as a leaf.
            makeLeaf (task);
            return;
        }

        std::uint32_t mid = task.begin;
        if (bestAxis >= 0)
        {
            const float scale = static_cast<float> (binCount) / centroidExtent[bestAxis];
            const float lo = centroidBounds.lo[bestAxis];
            auto* split = std::partition (order_.data() + task.begin, order_.data() + task.end,
                [&] (std::uint32_t t) { return binOf (centroids_[t][bestAxis], lo, scale, binCount) < bestSplit; });
            mid = static_cast<std::uint32_t> (split - order_.data());
        }
        if (mid == task.begin || mid == task.end)
        {
            if (count <= options_.maxLeafSize)
            {
                makeLeaf (task);
                return;
            }
            // Object median on the widest centroid axis. Equal centroids still
            // split by position in the range, so the build always terminates.
            ++stats_.medianSplits;
            const int axis = largestAxis (centroidExtent);
            mid = task.begin + count / 2;
            std::nth_element (order_.begin() + task.begin, order_.begin() + mid, order_.begin() + task.end,
                [&] (std::uint32_t a, std::uint32_t b) { return centroids_[a][axis] < centroids_[b][axis]; });
        }

        const auto left = static_cast<std::uint32_t> (nodes_.size());
        nodes_.push_back ({});
        nodes_.push_back ({});
        nodes_[task.node].firstOrLeft = left;
        nodes_[task.node].count = 0;
        tasks.push_back ({ left + 1, mid, task.end, task.depth + 1 });
        tasks.push_back ({ left, task.begin, mid, task.depth + 1 });
    }

    BvhBuildOptions options_;
    std::vector<BvhNode>& nodes_;
    std::vector<std::uint32_t>& order_;
    BvhStats& stats_;
    std::vector<Bounds> triangleBounds_;
    std::vector<Vec3> centroids_;
    std::uint32_t medianDepth_ = 0;
};

// Slab test against [tMin, tMax]. Writes the entry distance on a hit.
inline bool hitBox (const BvhNode& node, const PreparedRay& ray, float tMin, float tMax, float& entry) noexcept
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
            std::swap (t0, t1);
        nearT = t0 > nearT ? t0 : nearT;
        t1 *= kSlabRobustness;
        farT = t1 < farT ? t1 : farT;
    }
    entry = nearT;
    return nearT <= farT;
}

struct StackEntry
{
    std::uint32_t node;
    float entry;
};

} // namespace

bool prepareRay (const Ray& ray, PreparedRay& prepared) noexcept
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
        std::swap (kx, ky);
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

bool intersectTriangle (const PreparedRay& ray, Vec3 p0, Vec3 p1, Vec3 p2,
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

Vec3 offsetRayOrigin (Vec3 p, Vec3 n) noexcept
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

void Bvh::build (const std::vector<Vec3>& positions, const BvhBuildOptions& options)
{
    build (positions.data(), positions.size() / 3, options);
}

void Bvh::build (const Vec3* positions, std::size_t triangleCount, const BvhBuildOptions& options)
{
    clear();
    if (positions == nullptr || triangleCount == 0)
        return;
    std::vector<std::uint32_t> order;
    Builder builder (positions, triangleCount, options, nodes_, order, stats_);
    builder.run();
    triangleIndices_ = std::move (order);
    orderedPositions_.resize (triangleIndices_.size() * 3);
    for (std::size_t i = 0; i < triangleIndices_.size(); ++i)
        for (std::size_t k = 0; k < 3; ++k)
            orderedPositions_[3 * i + k] = positions[3 * std::size_t { triangleIndices_[i] } + k];
}

void Bvh::clear() noexcept
{
    nodes_.clear();
    orderedPositions_.clear();
    triangleIndices_.clear();
    stats_ = {};
}

Bounds Bvh::bounds() const noexcept
{
    Bounds b;
    if (! nodes_.empty())
    {
        const auto& root = nodes_.front();
        b.grow (Vec3 { root.boundsMin[0], root.boundsMin[1], root.boundsMin[2] });
        b.grow (Vec3 { root.boundsMax[0], root.boundsMax[1], root.boundsMax[2] });
    }
    return b;
}

bool Bvh::intersect (const Ray& ray, Hit& hit) const noexcept
{
    PreparedRay prepared;
    if (nodes_.empty() || ! prepareRay (ray, prepared))
        return false;
    const float tMin = ray.tMin;
    float tMax = ray.tMax;
    float entry = 0.0f;
    if (! hitBox (nodes_[0], prepared, tMin, tMax, entry))
        return false;

    std::array<StackEntry, Bvh::kMaxDepth + 1> stack {};
    std::size_t depth = 0;
    std::uint32_t current = 0;
    Hit best;
    for (;;)
    {
        const BvhNode& node = nodes_[current];
        if (node.count != 0)
        {
            const std::uint32_t end = node.firstOrLeft + node.count;
            for (std::uint32_t i = node.firstOrLeft; i < end; ++i)
            {
                const std::uint32_t triangle = triangleIndices_[i];
                if (triangle == ray.ignoreTriangle)
                    continue;
                Hit candidate;
                if (intersectTriangle (prepared, orderedPositions_[3 * std::size_t { i }],
                        orderedPositions_[3 * std::size_t { i } + 1],
                        orderedPositions_[3 * std::size_t { i } + 2], tMin, tMax, candidate))
                {
                    tMax = candidate.t;
                    best = candidate;
                    best.triangle = triangle;
                }
            }
        }
        else
        {
            const std::uint32_t left = node.firstOrLeft;
            float leftEntry = 0.0f;
            float rightEntry = 0.0f;
            const bool hitLeft = hitBox (nodes_[left], prepared, tMin, tMax, leftEntry);
            const bool hitRight = hitBox (nodes_[left + 1], prepared, tMin, tMax, rightEntry);
            if (hitLeft && hitRight)
            {
                // Visit the nearer child first and keep the other for later.
                const bool leftFirst = leftEntry <= rightEntry;
                stack[depth++] = leftFirst ? StackEntry { left + 1, rightEntry } : StackEntry { left, leftEntry };
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
            const StackEntry next = stack[--depth];
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

bool Bvh::occluded (const Ray& ray) const noexcept
{
    PreparedRay prepared;
    if (nodes_.empty() || ! prepareRay (ray, prepared))
        return false;
    const float tMin = ray.tMin;
    const float tMax = ray.tMax;
    float entry = 0.0f;
    if (! hitBox (nodes_[0], prepared, tMin, tMax, entry))
        return false;

    std::array<std::uint32_t, Bvh::kMaxDepth + 1> stack {};
    std::size_t depth = 0;
    std::uint32_t current = 0;
    for (;;)
    {
        const BvhNode& node = nodes_[current];
        if (node.count != 0)
        {
            const std::uint32_t end = node.firstOrLeft + node.count;
            for (std::uint32_t i = node.firstOrLeft; i < end; ++i)
            {
                if (triangleIndices_[i] == ray.ignoreTriangle)
                    continue;
                Hit candidate;
                if (intersectTriangle (prepared, orderedPositions_[3 * std::size_t { i }],
                        orderedPositions_[3 * std::size_t { i } + 1],
                        orderedPositions_[3 * std::size_t { i } + 2], tMin, tMax, candidate))
                    return true;
            }
        }
        else
        {
            const std::uint32_t left = node.firstOrLeft;
            const bool hitLeft = hitBox (nodes_[left], prepared, tMin, tMax, entry);
            const bool hitRight = hitBox (nodes_[left + 1], prepared, tMin, tMax, entry);
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
