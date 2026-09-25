#include "visual_plan_resource_budget.h"
#include "reactive_surface_fixture.h"

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

videowire::CompiledVisualLayerPlan animatedModelPlan()
{
    videowire::CompiledVisualLayerPlan plan;
    plan.clipId = 8;
    plan.structuralRevision = 4;
    plan.producerValidated = true;
    plan.nodeIds = { 1, 2, 3, 4 };
    plan.nodeKinds = { "visual.3d.imported-animation", "visual.3d.imported-deformation",
                       "visual.3d.render", "video.out" };
    for (size_t i = 0; i < plan.nodeIds.size(); ++i)
        plan.operations.push_back({ plan.nodeIds[i], plan.nodeKinds[i], "native-gpu", {} });
    const char* sourceTypes[] = { "mesh", "skeleton", "morphTargets", "animationClip", "scene3D" };
    for (int port = 0; port < 5; ++port)
        plan.ports.push_back({ 1, port, 1, "out", "control", sourceTypes[port], "unspecified", "unspecified" });
    for (int port = 0; port < 4; ++port)
    {
        plan.ports.push_back({ 2, port, 1, "in", "control", sourceTypes[port], "unspecified", "unspecified" });
        plan.edges.push_back({ 1, port, 2, port });
    }
    plan.ports.push_back({ 2, 4, 1, "out", "control", "mesh", "unspecified", "unspecified" });
    plan.ports.push_back({ 3, 0, 1, "in", "control", "scene3D", "unspecified", "unspecified" });
    plan.ports.push_back({ 3, 3, 1, "in", "control", "mesh", "unspecified", "unspecified" });
    plan.ports.push_back({ 3, 1, 1, "out", "frame", "image", "rgba8", "sRGB" });
    plan.ports.push_back({ 4, 0, 1, "in", "frame", "image", "rgba8", "sRGB" });
    plan.edges.push_back({ 1, 4, 3, 0 });
    plan.edges.push_back({ 2, 4, 3, 3 });
    plan.edges.push_back({ 3, 1, 4, 0 });
    plan.descriptorCount = 4;
    plan.operationCount = 4;
    // Producer counts each consumed output once, including source outputs that
    // fan out downstream: 5 source records + the deformed mesh.
    plan.sceneRecordCount = 6;
    plan.frameOutputCount = 1;
    plan.peakLiveFrameCount = 1;
    plan.allocatedFrameSlotCount = 1;
    return plan;
}

videowire::CompiledVisualLayerPlan retainedScenePlan (bool withFrame, bool& fixtureValid)
{
    using namespace HarmonicMIDI::grid;
    std::string fixtureError;
    const auto fixture = retainedSurfaceFixture(fixtureError);
    if (!fixture) { fixtureValid = false; return {}; }
    auto request = *fixture;
    request.renderStableId = 43;
    const auto payload = visualimportedscenerender::encode(request);
    fixtureValid = !payload.empty();

    videowire::CompiledVisualLayerPlan plan;
    plan.clipId = 9;
    plan.structuralRevision = 1;
    plan.producerValidated = true;
    plan.nodeIds = withFrame ? std::vector<int> { 40, 91, 42 } : std::vector<int> { 40, 42 };
    plan.nodeKinds = withFrame
        ? std::vector<std::string> { videowire::geometry::kRetainedSceneOperation, "video.source", "visual.3d.render" }
        : std::vector<std::string> { videowire::geometry::kRetainedSceneOperation, "visual.3d.render" };
    plan.operations = { { 40, videowire::geometry::kRetainedSceneOperation, "control-eval", {} } };
    if (withFrame) plan.operations.push_back({ 91, "video.source", "source-decode", {} });
    plan.operations.push_back({ 42, "visual.3d.render", "native-gpu", payload });
    plan.ports = { { 40, 0, 1, "out", "control", "scene3D", "unspecified", "unspecified" },
                   { 42, 0, 1, "in", "control", "scene3D", "unspecified", "unspecified" },
                   { 42, 1, 1, "out", "frame", "image", "rgba8", "sRGB" } };
    plan.edges = { { 40, 0, 42, 0 } };
    if (withFrame)
    {
        plan.ports.push_back({ 91, 0, 1, "out", "frame", "image", "rgba8", "sRGB" });
        plan.ports.push_back({ 42, 12, 1, "in", "frame", "image", "rgba8", "sRGB" });
        plan.edges.push_back({ 91, 0, 42, 12 });
    }
    plan.descriptorCount = plan.nodeKinds.size();
    plan.operationCount = plan.operations.size();
    plan.sceneRecordCount = request.sceneSnapshot->objectCount + request.sceneSnapshot->materialCount
        + request.sceneSnapshot->lightCount + request.sceneSnapshot->cameraCount;
    plan.frameOutputCount = withFrame ? 2 : 1;
    plan.peakLiveFrameCount = withFrame ? 2 : 1;
    plan.allocatedFrameSlotCount = withFrame ? 2 : 1;
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

    auto collapsed = linearPlan(1);
    collapsed.nodeKinds = { "geometry.core.runtime" };
    collapsed.operations = { { collapsed.nodeIds.front(), "geometry.core.runtime",
                               "native-gpu", {} } };
    collapsed.ports = { { collapsed.nodeIds.front(), 0, 1, "out", "frame",
                          "image", "rgba8", "linearSRGB" } };
    collapsed.edges.clear();
    collapsed.frameOutputCount = 1;
    collapsed.peakLiveFrameCount = 1;
    collapsed.allocatedFrameSlotCount = 1;
    error.clear();
    check(videowire::admitVisualPlanResources(
              collapsed, 1280, 720, limits, usage, error)
              && usage.frameOutputs == 1 && usage.peakLiveFrames == 1
              && usage.frameSlots == 1,
          "collapsed native terminal retains its unconsumed published Frame output");

    auto animated = animatedModelPlan();
    auto animatedLimits = videowire::VisualBackendResourceLimits::forCanvas(160, 90);
    error.clear();
    check(videowire::admitVisualPlanResources(animated, 160, 90, animatedLimits, usage, error)
              && usage.sceneRecords == 6,
          "animated-model mesh, skeleton, morph, animation, and scene records match producer accounting");
    auto animatedWithUnusedDepth = animated;
    animatedWithUnusedDepth.ports.push_back(
        { 3, 4, 1, "out", "frame", "depth", "r32f", "linear" });
    error.clear();
    check(videowire::admitVisualPlanResources(
              animatedWithUnusedDepth, 160, 90, animatedLimits, usage, error)
              && usage.frameOutputs == 1,
          "unused Render 3D pass outputs do not become retained collapsed frame resources");
    auto animatedFanout = animated;
    animatedFanout.ports.push_back({ 5, 0, 1, "in", "control", "mesh", "unspecified", "unspecified" });
    animatedFanout.nodeIds.push_back(5);
    animatedFanout.nodeKinds.push_back("visual.test.mesh-consumer");
    animatedFanout.operations.push_back({ 5, "visual.test.mesh-consumer", "native-gpu", {} });
    animatedFanout.edges.push_back({ 1, 0, 5, 0 });
    animatedFanout.descriptorCount = 5;
    animatedFanout.operationCount = 5;
    error.clear();
    check(videowire::admitVisualPlanResources(animatedFanout, 160, 90, animatedLimits, usage, error)
              && usage.sceneRecords == 6,
          "fanout counts one consumed scene-record output once");
    auto sceneConstrained = animatedLimits;
    sceneConstrained.sceneRecords = 5;
    error.clear();
    check(! videowire::admitVisualPlanResources(animated, 160, 90, sceneConstrained, usage, error)
              && error == "visual graph scene record capacity exceeded: 6 > 5",
          "animated-model scene records remain subject to the backend budget");

    bool retainedFixtureValid = false;
    auto retained = retainedScenePlan(false, retainedFixtureValid);
    check(retainedFixtureValid, "retained resource fixture has a canonical encoded payload");
    error.clear();
    check(videowire::admitVisualPlanResources(retained, 160, 90, animatedLimits, usage, error)
              && usage.descriptors == 2 && usage.sceneRecords == 10
              && usage.frameOutputs == 1 && usage.peakLiveFrames == 1,
          "retained scene accounts exact payload records and its published Render image");
    const auto retainedPayload = retained.operations.back().payloadXml;
    const auto projectLimits = videowire::VisualBackendResourceLimits::forCanvas(3840, 2160);
    error.clear();
    check(videowire::admitVisualPlanResources(retained, 3840, 2160, projectLimits, usage, error)
              && usage.allocatedFrameBytes == 3840ull * 2160ull * 4ull,
          "project-size retained scene bytes derive from its render extent");
    auto diagnosticLimits = videowire::VisualBackendResourceLimits::forCanvas(320, 180);
    error.clear();
    check(videowire::admitVisualPlanResources(retained, 320, 180, diagnosticLimits, usage, error)
              && usage.allocatedFrameBytes == 320ull * 180ull * 4ull
              && retained.operations.back().payloadXml == retainedPayload
              && usage.sceneRecords == 10 && usage.frameSlots == 1,
          "the unchanged retained scene recomputes transient bytes at the diagnostic extent");
    diagnosticLimits.allocatedFrameBytes = 320ull * 180ull * 4ull - 1;
    error.clear();
    check(!videowire::admitVisualPlanResources(retained, 320, 180, diagnosticLimits, usage, error)
              && error.find("allocated frame byte capacity exceeded") != std::string::npos,
          "diagnostic extents still enforce backend allocation limits");
    diagnosticLimits = videowire::VisualBackendResourceLimits::forCanvas(320, 180);
    auto forgedDiagnostic = retained;
    ++forgedDiagnostic.allocatedFrameSlotCount;
    error.clear();
    check(!videowire::admitVisualPlanResources(forgedDiagnostic, 320, 180, diagnosticLimits, usage, error)
              && error.find("disagrees with producer accounting") != std::string::npos,
          "a smaller diagnostic render never bypasses producer resource-count validation");
    auto retainedWithUnusedDepth = retained;
    retainedWithUnusedDepth.ports.push_back(
        { 42, 4, 1, "out", "frame", "depth", "r32f", "unspecified" });
    error.clear();
    check(videowire::admitVisualPlanResources(
              retainedWithUnusedDepth, 160, 90, animatedLimits, usage, error)
              && usage.frameOutputs == 1 && usage.frameSlots == 1,
          "retained Render image is published without retaining an unused depth output");
    bool retainedFrameFixtureValid = false;
    auto retainedWithFrame = retainedScenePlan(true, retainedFrameFixtureValid);
    check(retainedFrameFixtureValid, "retained Frame resource fixture has a canonical encoded payload");
    error.clear();
    check(videowire::admitVisualPlanResources(retainedWithFrame, 160, 90, animatedLimits, usage, error)
              && usage.descriptors == 3 && usage.sceneRecords == 10
              && usage.frameOutputs == 2 && usage.peakLiveFrames == 2 && usage.frameSlots == 2,
          "retained scene with a Frame prerequisite preserves topological liveness and exact records");
    auto forgedRetained = retainedWithFrame;
    forgedRetained.sceneRecordCount = 1;
    error.clear();
    check(! videowire::admitVisualPlanResources(forgedRetained, 160, 90, animatedLimits, usage, error)
              && error.find("sceneRecords: producer=1, helper=10") != std::string::npos,
          "retained scene rejects forged producer scene-record accounting");
    auto malformedRetained = retained;
    malformedRetained.operations.back().payloadXml = "not-a-retained-scene";
    error.clear();
    check(! videowire::admitVisualPlanResources(malformedRetained, 160, 90, animatedLimits, usage, error)
              && error == "visual resource admission requires the exact retained scene payload",
          "retained scene resource accounting requires the authoritative payload");

    auto disagreeing = plan;
    disagreeing.peakLiveFrameCount = 1;
    error.clear();
    check(! videowire::admitVisualPlanResources(disagreeing, 1280, 720, limits, usage, error)
              && error == "visual resource admission disagrees with producer accounting; peakLiveFrames: producer=1, helper=2",
          "helper rejects resource-accounting disagreement with the precise differing counts");

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

    for (const auto& format : { "r32f", "rg16f", "r32uint" })
    {
        auto typed = collapsed;
        typed.ports[0].pixelFormat = format;
        typed.ports[0].dataType = format == std::string("r32f") ? "depth"
            : format == std::string("rg16f") ? "motionVectors" : "objectId";
        error.clear();
        check(videowire::admitVisualPlanResources(typed, 160, 90, animatedLimits, usage, error)
                  && usage.allocatedFrameBytes == 160ull * 90ull * 4ull,
              "typed depth, motion, and ID AOV formats have exact byte accounting");
    }

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
