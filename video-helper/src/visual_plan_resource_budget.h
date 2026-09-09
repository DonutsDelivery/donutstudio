#pragma once

#include "render_snapshot.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace videowire
{
struct VisualBackendResourceLimits
{
    size_t descriptors = 256;
    size_t operations = 256;
    size_t sceneRecords = 128;
    size_t frameOutputs = 256;
    size_t liveFrames = 32;
    size_t frameSlots = 32;
    uint64_t allocatedFrameBytes = 512ull * 1024ull * 1024ull;
    int maximumImageDimension = 8192;
    std::string backendProfile = "unqueried";
    std::string backendDeviceIdentity;
    std::string maximumImageDimensionSource = "conservative-default";
    std::string frameSlotSource = "conservative-default-no-safe-capability-mapping";
    std::string allocatedFrameBytesSource = "conservative-default-no-safe-byte-capability";
    int observedCombinedTextureImageUnits = 0;
    uint64_t observedMaximumBufferLengthBytes = 0;
    uint64_t observedRecommendedWorkingSetBytes = 0;

    struct Capabilities
    {
        std::string backendProfile = "unqueried";
        std::string backendDeviceIdentity;
        int maximumImageDimension = 0;
        int combinedTextureImageUnits = 0;
        uint64_t maximumBufferLengthBytes = 0;
        uint64_t recommendedWorkingSetBytes = 0;
        std::string maximumImageDimensionQuery;
        std::string recommendedWorkingSetQuery;
    };

    static VisualBackendResourceLimits fromCapabilities (int width, int height,
                                                         const Capabilities& capabilities)
    {
        constexpr uint64_t conservativeFrameByteBudget = 512ull * 1024ull * 1024ull;
        VisualBackendResourceLimits limits;
        limits.backendProfile = capabilities.backendProfile.empty()
            ? "unqueried" : capabilities.backendProfile;
        limits.backendDeviceIdentity = capabilities.backendDeviceIdentity;
        limits.observedCombinedTextureImageUnits = std::max(0, capabilities.combinedTextureImageUnits);
        limits.observedMaximumBufferLengthBytes = capabilities.maximumBufferLengthBytes;
        limits.observedRecommendedWorkingSetBytes = capabilities.recommendedWorkingSetBytes;
        if (capabilities.maximumImageDimension > 0)
        {
            limits.maximumImageDimension = capabilities.maximumImageDimension;
            limits.maximumImageDimensionSource = capabilities.maximumImageDimensionQuery.empty()
                ? "backend-query" : capabilities.maximumImageDimensionQuery;
        }
        if (capabilities.recommendedWorkingSetBytes > 0)
        {
            // Reserve 25% for drawables, persistent renderer buffers, encoder
            // surfaces, and textures that visual-plan admission does not own.
            const auto planBudget = capabilities.recommendedWorkingSetBytes / 4 * 3;
            if (planBudget < conservativeFrameByteBudget)
            {
                limits.allocatedFrameBytes = planBudget;
                limits.allocatedFrameBytesSource = capabilities.recommendedWorkingSetQuery.empty()
                    ? "backend-recommended-working-set*0.75-plan-budget"
                    : capabilities.recommendedWorkingSetQuery + "*0.75-plan-budget";
            }
        }
        if (width > 0 && height > 0)
        {
            const auto pixels = static_cast<uint64_t> (width) * static_cast<uint64_t> (height);
            const uint64_t thirtyTwoRgba16fFrames = pixels > std::numeric_limits<uint64_t>::max() / 256ull
                ? std::numeric_limits<uint64_t>::max() : pixels * 256ull;
            const auto canvasBudget = std::max<uint64_t>(64ull * 1024ull * 1024ull,
                std::min<uint64_t>(thirtyTwoRgba16fFrames, conservativeFrameByteBudget));
            if (canvasBudget < limits.allocatedFrameBytes)
            {
                limits.allocatedFrameBytes = canvasBudget;
                limits.allocatedFrameBytesSource = "conservative-canvas-default-no-safe-byte-capability";
            }
        }
        return limits;
    }

    static VisualBackendResourceLimits forCanvas (int width, int height,
                                                   int backendMaximumImageDimension = 8192,
                                                   size_t backendFrameSlots = 32)
    {
        Capabilities capabilities;
        capabilities.maximumImageDimension = std::max (1, backendMaximumImageDimension);
        capabilities.maximumImageDimensionQuery = "caller-supplied-capability";
        auto limits = fromCapabilities(width, height, capabilities);
        limits.frameSlots = std::clamp (backendFrameSlots, size_t { 1 }, size_t { 32 });
        limits.liveFrames = limits.frameSlots;
        limits.frameSlotSource = "caller-supplied-capability-clamped-to-scheduler-capacity";
        return limits;
    }
};

struct VisualPlanResourceUsage
{
    size_t descriptors = 0;
    size_t operations = 0;
    size_t sceneRecords = 0;
    size_t frameOutputs = 0;
    size_t peakLiveFrames = 0;
    size_t frameSlots = 0;
    uint64_t allocatedFrameBytes = 0;
};

struct VisualPlanResourceReceipt
{
    size_t planCount = 0;
    size_t descriptors = 0;
    size_t operations = 0;
    size_t sceneRecords = 0;
    size_t frameOutputs = 0;
    size_t peakLiveFrames = 0;
    size_t frameSlots = 0;
    uint64_t allocatedFrameBytes = 0;

    bool operator== (const VisualPlanResourceReceipt& other) const noexcept
    {
        return planCount == other.planCount && descriptors == other.descriptors
            && operations == other.operations && sceneRecords == other.sceneRecords
            && frameOutputs == other.frameOutputs
            && peakLiveFrames == other.peakLiveFrames && frameSlots == other.frameSlots
            && allocatedFrameBytes == other.allocatedFrameBytes;
    }
};

struct VisualPlanBudgetReceipt
{
    int canvasWidth = 0;
    int canvasHeight = 0;
    VisualBackendResourceLimits limits;
    VisualPlanResourceReceipt usage;

    bool operator== (const VisualPlanBudgetReceipt& other) const noexcept
    {
        return canvasWidth == other.canvasWidth && canvasHeight == other.canvasHeight
            && limits.descriptors == other.limits.descriptors
            && limits.operations == other.limits.operations
            && limits.sceneRecords == other.limits.sceneRecords
            && limits.frameOutputs == other.limits.frameOutputs
            && limits.liveFrames == other.limits.liveFrames
            && limits.frameSlots == other.limits.frameSlots
            && limits.allocatedFrameBytes == other.limits.allocatedFrameBytes
            && limits.maximumImageDimension == other.limits.maximumImageDimension
            && limits.backendProfile == other.limits.backendProfile
            && limits.backendDeviceIdentity == other.limits.backendDeviceIdentity
            && limits.maximumImageDimensionSource == other.limits.maximumImageDimensionSource
            && limits.frameSlotSource == other.limits.frameSlotSource
            && limits.allocatedFrameBytesSource == other.limits.allocatedFrameBytesSource
            && limits.observedCombinedTextureImageUnits == other.limits.observedCombinedTextureImageUnits
            && limits.observedMaximumBufferLengthBytes == other.limits.observedMaximumBufferLengthBytes
            && limits.observedRecommendedWorkingSetBytes == other.limits.observedRecommendedWorkingSetBytes
            && usage == other.usage;
    }
};

namespace visualresource_detail
{
inline bool isSceneRecord (const std::string& dataType)
{
    static const std::set<std::string> types {
        "scene3D", "transform3D", "material", "light", "camera", "sdf", "animationClip"
    };
    return types.count (dataType) != 0;
}

inline uint64_t bytesPerPixel (const std::string& pixelFormat)
{
    if (pixelFormat == "r8") return 1;
    if (pixelFormat == "r16") return 2;
    if (pixelFormat == "rgba8") return 4;
    if (pixelFormat == "rgba16f") return 8;
    if (pixelFormat == "rgba32f") return 16;
    return 0;
}

struct FrameDescriptor
{
    int channels = 0;
    std::string dataType;
    std::string pixelFormat;
    std::string colorSpace;

    bool operator== (const FrameDescriptor& other) const noexcept
    {
        return channels == other.channels && dataType == other.dataType
            && pixelFormat == other.pixelFormat && colorSpace == other.colorSpace;
    }
};

struct Lifetime
{
    FrameDescriptor descriptor;
    size_t producedAt = 0;
    size_t lastUsedAt = 0;
    size_t slot = 0;
};
} // namespace visualresource_detail

inline bool accountVisualPlanResources (const CompiledVisualLayerPlan& plan,
                                        int canvasWidth, int canvasHeight,
                                        VisualPlanResourceUsage& usage, std::string& error)
{
    using namespace visualresource_detail;
    usage = {};
    error.clear();
    if (canvasWidth <= 0 || canvasHeight <= 0)
    {
        error = "visual resource admission requires positive canvas dimensions";
        return false;
    }
    if (plan.nodeKinds.size() != plan.nodeIds.size())
    {
        error = "visual resource admission found mismatched node descriptors";
        return false;
    }

    std::map<int, size_t> scheduleIndex;
    for (size_t index = 0; index < plan.nodeIds.size(); ++index)
        if (plan.nodeIds[index] <= 0 || ! scheduleIndex.emplace(plan.nodeIds[index], index).second)
        {
            error = "visual resource admission found a duplicate or invalid node identity";
            return false;
        }

    usage.descriptors = plan.nodeKinds.size();
    usage.operations = plan.operations.size();
    std::map<std::pair<int, int>, const CompiledVisualPortBinding*> outputs;
    for (const auto& port : plan.ports)
    {
        if (scheduleIndex.count(port.nodeId) == 0) continue;
        if (port.direction == "out")
            outputs[{ port.nodeId, port.port }] = &port;
    }

    struct Use
    {
        const CompiledVisualPortBinding* port = nullptr;
        size_t producedAt = 0;
        size_t lastUsedAt = 0;
    };
    std::map<std::pair<int, int>, Use> usedOutputs;
    for (const auto& edge : plan.edges)
    {
        const auto source = outputs.find({ edge.fromNodeId, edge.fromPort });
        const auto destination = scheduleIndex.find(edge.toNodeId);
        if (source == outputs.end() || destination == scheduleIndex.end())
        {
            error = "visual resource admission found an invalid edge endpoint";
            return false;
        }
        const auto producedAt = scheduleIndex.at(edge.fromNodeId);
        auto& use = usedOutputs[{ edge.fromNodeId, edge.fromPort }];
        use.port = source->second;
        use.producedAt = producedAt;
        use.lastUsedAt = std::max(use.lastUsedAt, destination->second);
    }

    std::vector<Lifetime> lifetimes;
    std::vector<FrameDescriptor> slots;
    std::vector<size_t> slotLiveUntil;
    for (size_t schedulePosition = 0; schedulePosition < plan.nodeIds.size(); ++schedulePosition)
        for (const auto& entry : usedOutputs)
        {
            const auto& use = entry.second;
            if (use.producedAt != schedulePosition) continue;
            if (use.port->carrier != "frame")
            {
                if (isSceneRecord(use.port->dataType)) ++usage.sceneRecords;
                continue;
            }
            const auto pixelBytes = bytesPerPixel(use.port->pixelFormat);
            if (use.port->dataType.empty() || pixelBytes == 0)
            {
                error = "visual resource admission requires exact frame descriptors";
                return false;
            }
            ++usage.frameOutputs;
            const FrameDescriptor descriptor {
                use.port->channels, use.port->dataType, use.port->pixelFormat, use.port->colorSpace
            };
            size_t slot = slots.size();
            for (size_t candidate = 0; candidate < slots.size(); ++candidate)
                if (slotLiveUntil[candidate] < schedulePosition && slots[candidate] == descriptor)
                {
                    slot = candidate;
                    break;
                }
            if (slot == slots.size())
            {
                slots.push_back(descriptor);
                slotLiveUntil.push_back(use.lastUsedAt);
                const auto pixels = static_cast<uint64_t>(canvasWidth) * static_cast<uint64_t>(canvasHeight);
                if (pixelBytes > 0 && pixels > std::numeric_limits<uint64_t>::max() / pixelBytes)
                {
                    error = "visual resource byte accounting overflow";
                    return false;
                }
                const auto bytes = pixels * pixelBytes;
                if (usage.allocatedFrameBytes > std::numeric_limits<uint64_t>::max() - bytes)
                {
                    error = "visual resource byte accounting overflow";
                    return false;
                }
                usage.allocatedFrameBytes += bytes;
            }
            else
                slotLiveUntil[slot] = use.lastUsedAt;
            lifetimes.push_back({ descriptor, use.producedAt, use.lastUsedAt, slot });
        }

    usage.frameSlots = slots.size();
    for (size_t schedulePosition = 0; schedulePosition < plan.nodeIds.size(); ++schedulePosition)
    {
        const auto live = static_cast<size_t>(std::count_if(lifetimes.begin(), lifetimes.end(),
            [schedulePosition](const Lifetime& lifetime)
            { return lifetime.producedAt <= schedulePosition && lifetime.lastUsedAt >= schedulePosition; }));
        usage.peakLiveFrames = std::max(usage.peakLiveFrames, live);
    }
    return true;
}

inline bool admitVisualPlanResources (const CompiledVisualLayerPlan& plan,
                                      int canvasWidth, int canvasHeight,
                                      const VisualBackendResourceLimits& limits,
                                      VisualPlanResourceUsage& usage, std::string& error)
{
    if (! accountVisualPlanResources(plan, canvasWidth, canvasHeight, usage, error)) return false;
    const auto capacityError = [&error](const char* name, uint64_t used, uint64_t limit)
    {
        error = std::string("visual graph ") + name + " capacity exceeded: "
            + std::to_string(used) + " > " + std::to_string(limit);
        return false;
    };
    if (canvasWidth > limits.maximumImageDimension || canvasHeight > limits.maximumImageDimension)
        return capacityError("image dimension", static_cast<uint64_t>(std::max(canvasWidth, canvasHeight)),
                             static_cast<uint64_t>(limits.maximumImageDimension));
    if (usage.descriptors > limits.descriptors) return capacityError("descriptor", usage.descriptors, limits.descriptors);
    if (usage.operations > limits.operations) return capacityError("operation", usage.operations, limits.operations);
    if (usage.sceneRecords > limits.sceneRecords) return capacityError("scene record", usage.sceneRecords, limits.sceneRecords);
    if (usage.frameOutputs > limits.frameOutputs) return capacityError("frame output", usage.frameOutputs, limits.frameOutputs);
    if (usage.peakLiveFrames > limits.liveFrames) return capacityError("live frame", usage.peakLiveFrames, limits.liveFrames);
    if (usage.frameSlots > limits.frameSlots) return capacityError("frame slot", usage.frameSlots, limits.frameSlots);
    if (usage.allocatedFrameBytes > limits.allocatedFrameBytes)
        return capacityError("allocated frame byte", usage.allocatedFrameBytes, limits.allocatedFrameBytes);
    if (plan.descriptorCount != usage.descriptors || plan.operationCount != usage.operations
        || plan.sceneRecordCount != usage.sceneRecords || plan.frameOutputCount != usage.frameOutputs
        || plan.peakLiveFrameCount != usage.peakLiveFrames
        || plan.allocatedFrameSlotCount != usage.frameSlots)
    {
        error = "visual resource admission disagrees with producer accounting";
        return false;
    }
    return true;
}
} // namespace videowire
