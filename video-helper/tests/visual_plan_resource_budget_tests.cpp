#include "visual_plan_resource_budget.h"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

namespace
{
int failures = 0;

void check (bool condition, const char* message)
{
    if (condition) return;
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

videowire::CompiledVisualLayerPlan linearPlan (int nodeCount, const std::string& pixelFormat = "rgba8")
{
    videowire::CompiledVisualLayerPlan plan;
    plan.clipId = 7;
    plan.structuralRevision = 1;
    plan.producerValidated = true;
    for (int node = 0; node < nodeCount; ++node)
    {
        const int nodeId = node + 1;
        plan.nodeIds.push_back(nodeId);
        plan.nodeKinds.push_back(node + 1 == nodeCount ? "video.out" : "visual.test.pass");
        plan.operations.push_back({ nodeId, plan.nodeKinds.back(), "native-gpu", {} });
        if (node > 0)
            plan.ports.push_back({ nodeId, 0, 1, "in", "frame", "image", pixelFormat, "linearSRGB" });
        if (node + 1 < nodeCount)
        {
            plan.ports.push_back({ nodeId, 1, 1, "out", "frame", "image", pixelFormat, "linearSRGB" });
            plan.edges.push_back({ nodeId, 1, nodeId + 1, 0 });
        }
    }
    plan.descriptorCount = static_cast<size_t>(nodeCount);
    plan.operationCount = static_cast<size_t>(nodeCount);
    plan.sceneRecordCount = 0;
    plan.frameOutputCount = static_cast<size_t>(nodeCount - 1);
    plan.peakLiveFrameCount = nodeCount > 1 ? 2 : 0;
    plan.allocatedFrameSlotCount = nodeCount > 1 ? 2 : 0;
    return plan;
}
} // namespace

int main()
{
    auto plan = linearPlan(20);
    auto limits = videowire::VisualBackendResourceLimits::forCanvas(1280, 720);
    videowire::VisualPlanResourceUsage usage;
    std::string error;
    check(videowire::admitVisualPlanResources(plan, 1280, 720, limits, usage, error),
          "a linear graph with more than sixteen operations is admitted when liveness fits");
    check(usage.descriptors == 20 && usage.operations == 20 && usage.frameOutputs == 19,
          "descriptor, operation, and frame-output counts are independent");
    check(usage.peakLiveFrames == 2 && usage.frameSlots == 2,
          "linear frame lifetimes reuse two compatible slots");
    check(usage.allocatedFrameBytes == 1280ull * 720ull * 4ull * 2ull,
          "allocated bytes are derived from measured canvas dimensions and slot formats");

    auto disagreeing = plan;
    disagreeing.peakLiveFrameCount = 1;
    error.clear();
    check(! videowire::admitVisualPlanResources(disagreeing, 1280, 720, limits, usage, error)
              && error == "visual resource admission disagrees with producer accounting",
          "helper independently rejects producer resource-accounting disagreement");

    auto constrained = limits;
    constrained.liveFrames = 1;
    error.clear();
    check(! videowire::admitVisualPlanResources(plan, 1280, 720, constrained, usage, error)
              && error == "visual graph live frame capacity exceeded: 2 > 1",
          "backend live-frame capacity is enforced separately from operation count");

    auto byteHeavy = linearPlan(20, "rgba32f");
    auto byteLimits = videowire::VisualBackendResourceLimits::forCanvas(8192, 8192);
    error.clear();
    check(! videowire::admitVisualPlanResources(byteHeavy, 8192, 8192, byteLimits, usage, error)
              && error.find("allocated frame byte capacity exceeded") != std::string::npos,
          "measured frame bytes enforce the backend allocation envelope");

    auto exactFit = linearPlan(3, "rgba8");
    auto exactLimits = videowire::VisualBackendResourceLimits::forCanvas(64, 64);
    exactLimits.allocatedFrameBytes = 64ull * 64ull * 4ull * 2ull;
    error.clear();
    check(videowire::admitVisualPlanResources(exactFit, 64, 64, exactLimits, usage, error)
              && usage.allocatedFrameBytes == exactLimits.allocatedFrameBytes,
          "an exact byte-budget fit is admitted");
    --exactLimits.allocatedFrameBytes;
    error.clear();
    check(! videowire::admitVisualPlanResources(exactFit, 64, 64, exactLimits, usage, error)
              && error == "visual graph allocated frame byte capacity exceeded: 32768 > 32767",
          "one byte over budget is rejected before allocation");

    auto overflowPlan = linearPlan(3, "rgba32f");
    auto overflowLimits = videowire::VisualBackendResourceLimits::forCanvas(
        std::numeric_limits<int>::max(), std::numeric_limits<int>::max(),
        std::numeric_limits<int>::max());
    error.clear();
    check(! videowire::admitVisualPlanResources(
              overflowPlan, std::numeric_limits<int>::max(), std::numeric_limits<int>::max(),
              overflowLimits, usage, error)
              && error == "visual resource byte accounting overflow",
          "frame-byte multiplication overflow is rejected");

    videowire::VisualBackendResourceLimits::Capabilities glCapabilities;
    glCapabilities.backendProfile = "opengl-4.6";
    glCapabilities.maximumImageDimension = 16384;
    glCapabilities.combinedTextureImageUnits = 192;
    glCapabilities.maximumImageDimensionQuery = "OpenGL.GL_MAX_TEXTURE_SIZE";
    const auto glLimits = videowire::VisualBackendResourceLimits::fromCapabilities(
        1280, 720, glCapabilities);
    check(glLimits.maximumImageDimension == 16384 && glLimits.frameSlots == 32
              && glLimits.observedCombinedTextureImageUnits == 192
              && glLimits.frameSlotSource == "conservative-default-no-safe-capability-mapping"
              && glLimits.allocatedFrameBytesSource
                    == "conservative-canvas-default-no-safe-byte-capability",
          "OpenGL profile uses queried dimensions without treating binding counts as memory limits");

    videowire::VisualBackendResourceLimits::Capabilities metalCapabilities;
    metalCapabilities.backendProfile = "metal-renderer-device";
    metalCapabilities.backendDeviceIdentity = "Apple M3 Max@0x1234";
    metalCapabilities.maximumImageDimension = 16384;
    metalCapabilities.maximumImageDimensionQuery =
        "Sokol.sg_query_limits.max_image_size_2d";
    metalCapabilities.maximumBufferLengthBytes = 256ull * 1024ull * 1024ull;
    metalCapabilities.recommendedWorkingSetBytes = 48ull * 1024ull * 1024ull;
    metalCapabilities.recommendedWorkingSetQuery = "Metal.MTLDevice.recommendedMaxWorkingSetSize";
    const auto metalLimits = videowire::VisualBackendResourceLimits::fromCapabilities(
        1280, 720, metalCapabilities);
    check(metalLimits.allocatedFrameBytes == 36ull * 1024ull * 1024ull
              && metalLimits.backendDeviceIdentity == "Apple M3 Max@0x1234"
              && metalLimits.maximumImageDimension == 16384
              && metalLimits.maximumImageDimensionSource
                    == "Sokol.sg_query_limits.max_image_size_2d"
              && metalLimits.allocatedFrameBytesSource
                    == "Metal.MTLDevice.recommendedMaxWorkingSetSize*0.75-plan-budget"
              && metalLimits.observedRecommendedWorkingSetBytes
                    == 48ull * 1024ull * 1024ull
              && metalLimits.observedMaximumBufferLengthBytes
                    == 256ull * 1024ull * 1024ull,
          "Metal profile reserves renderer headroom and records exact device provenance");

    metalCapabilities.recommendedWorkingSetBytes = 600ull * 1024ull * 1024ull;
    const auto largeMetalLimits = videowire::VisualBackendResourceLimits::fromCapabilities(
        8192, 8192, metalCapabilities);
    check(largeMetalLimits.allocatedFrameBytes == 450ull * 1024ull * 1024ull,
          "Metal headroom reserve applies before the conservative global ceiling");

    if (failures != 0) return EXIT_FAILURE;
    std::cout << "Visual plan resource budget checks passed\n";
    return EXIT_SUCCESS;
}
