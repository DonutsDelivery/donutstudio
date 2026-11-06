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
    Builder (const Vec3* positions, const Vec3* close, std::size_t triangleCount, const BvhBuildOptions& options,
             std::vector<BvhNode>& nodes, std::vector<std::uint32_t>& order, BvhStats& stats)
        : options_ (options), nodes_ (nodes), order_ (order), stats_ (stats)
    {
        triangleBounds_.resize (triangleCount);
        centroids_.resize (triangleCount);
        order_.resize (triangleCount);
        for (std::size_t i = 0; i < triangleCount; ++i)
        {
            const Bounds b = motionTriangleBounds (positions + 3 * i, close != nullptr ? close + 3 * i : nullptr);
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
} // namespace

void Bvh::build (const std::vector<Vec3>& positions, const BvhBuildOptions& options)
{
    build (positions.data(), positions.size() / 3, options);
}

void Bvh::build (const Vec3* positions, std::size_t triangleCount, const BvhBuildOptions& options)
{
    buildMotion (positions, nullptr, triangleCount, options);
}

void Bvh::buildMotion (const Vec3* open, const Vec3* close, std::size_t triangleCount, const BvhBuildOptions& options)
{
    clear();
    if (open == nullptr || triangleCount == 0)
        return;
    std::vector<std::uint32_t> order;
    Builder builder (open, close, triangleCount, options, nodes_, order, stats_);
    builder.run();
    triangleIndices_ = std::move (order);
    orderedPositions_.resize (triangleIndices_.size() * 3);
    if (close != nullptr)
        orderedClose_.resize (triangleIndices_.size() * 3);
    for (std::size_t i = 0; i < triangleIndices_.size(); ++i)
        for (std::size_t k = 0; k < 3; ++k)
        {
            const std::size_t source = 3 * std::size_t { triangleIndices_[i] } + k;
            orderedPositions_[3 * i + k] = open[source];
            if (close != nullptr)
                orderedClose_[3 * i + k] = close[source];
        }
}

void Bvh::clear() noexcept
{
    nodes_.clear();
    orderedPositions_.clear();
    orderedClose_.clear();
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
    return traverseClosest (view(), ray, hit);
}

bool Bvh::occluded (const Ray& ray) const noexcept
{
    return traverseAny (view(), ray);
}

} // namespace videohelper::pathtrace
