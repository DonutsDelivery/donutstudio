#include "../src/visual_plan_executor.h"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>

namespace
{
int failures = 0;

void check (bool condition, const char* message)
{
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
}

videowire::CompiledVisualLayerPlan fourWayPlan()
{
    videowire::CompiledVisualLayerPlan plan;
    plan.clipId = 7;
    plan.structuralRevision = 12;
    plan.producerValidated = true;
    plan.nodeKinds = { "video.out", "video.layer.source", "video.blend", "video.source",
                       "video.text", "visual.matte.asset", "visual.matte.apply" };
    plan.nodeIds = { 25, 26, 24, 11, 22, 27, 28 };
    plan.operations = {
        { 25, "video.out", "native-gpu", "" },
        { 26, "video.layer.source", "source-decode", "" },
        { 24, "video.blend", "native-gpu", "" },
        { 11, "video.source", "source-decode", "" },
        { 22, "video.text", "native-gpu", "" },
        { 27, "visual.matte.asset", "source-decode",
          "<MatteAssetBinding matteAssetId=\"matte-four-way\" state=\"available\" "
          "cacheKey=\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\" "
          "contentReceipt=\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\" "
          "framePrefix=\"mask-\" frameExtension=\".rgba\" firstFrame=\"1\" frameDigits=\"4\" "
          "backend=\"rgba-cpu-decode-native-gpu-upload\"/>" },
        { 28, "visual.matte.apply", "native-gpu", "" }
    };
    plan.ports = {
        {25,0,1,"in","frame","image","rgba8","sRGB"},
        {26,0,1,"out","frame","image","rgba8","sRGB"},
        {24,0,1,"in","frame","image","rgba8","sRGB"},
        {24,1,1,"out","frame","image","rgba8","sRGB"},
        {24,2,1,"in","frame","image","rgba8","sRGB"},
        {24,3,1,"in","frame","image","rgba8","sRGB"},
        {11,0,1,"out","frame","image","rgba8","sRGB"},
        {22,0,1,"out","frame","image","rgba8","sRGB"},
        {27,0,1,"out","frame","mask","r8","linearSRGB"},
        {28,0,1,"in","frame","image","rgba8","sRGB"},
        {28,1,1,"in","frame","mask","r8","linearSRGB"},
        {28,2,1,"out","frame","image","rgba8","sRGB"}
    };
    plan.edges = {
        {28,2,24,0}, {22,0,24,2}, {26,0,24,3}, {24,1,25,0},
        {11,0,28,0}, {27,0,28,1}
    };
    return plan;
}
}

int main()
{
    std::string error;
    auto plan = fourWayPlan();
    videowire::VisualLayerExecution execution;
    check(videowire::compileVisualLayerExecution(plan, execution, error)
          && execution.matteApply && execution.compositeInputCount == 3
          && execution.dagSchedule.operations.size() == 7
          && execution.dagSchedule.operations[4].inputs.size() == 2
          && execution.dagSchedule.operations[5].inputs.size() == 3,
          "four-way image and mask topology compiles to one native schedule");

    videowire::VisualPlanResourceUsage usage;
    check(videowire::accountVisualPlanResources(plan, 64, 64, usage, error),
          "four-way topology has deterministic resource accounting");
    plan.descriptorCount = usage.descriptors;
    plan.operationCount = usage.operations;
    plan.sceneRecordCount = usage.sceneRecords;
    plan.frameOutputCount = usage.frameOutputs;
    plan.peakLiveFrameCount = usage.peakLiveFrames;
    plan.allocatedFrameSlotCount = usage.frameSlots;
    auto limits = videowire::VisualBackendResourceLimits::forCanvas(64, 64);
    auto preview = std::make_unique<videowire::VisualPlanExecutionState>();
    auto exportState = std::make_unique<videowire::VisualPlanExecutionState>();
    check(preview->admitPlans({plan}, &error, 64, 64, &limits)
          && exportState->admitPlans({plan}, &error, 64, 64, &limits)
          && preview->compiled(7, 12) != nullptr && exportState->compiled(7, 12) != nullptr,
          "preview and export admit the same immutable plan and exact budget");

    auto undersized = limits;
    undersized.descriptors = usage.descriptors - 1;
    auto rejected = std::make_unique<videowire::VisualPlanExecutionState>();
    check(! rejected->admitPlans({plan}, &error, 64, 64, &undersized)
          && error.find("visual graph descriptor capacity exceeded") == 0,
          "oversize topology fails closed before backend execution");

    auto malformed = fourWayPlan();
    malformed.ports[10].dataType = "image";
    check(! videowire::compileVisualLayerExecution(malformed, execution, error)
          && error == "bounded visual DAG has an incompatible typed edge",
          "malformed mask port fails closed");

    auto multiplyBound = fourWayPlan();
    multiplyBound.edges.push_back({26,0,28,1});
    check(! videowire::compileVisualLayerExecution(multiplyBound, execution, error)
          && error == "bounded visual DAG has a duplicate edge or multiply-bound input port",
          "multiply-bound mask input fails closed");

    if (failures != 0)
        std::fprintf(stderr, "%d four-way compositor checks failed\n", failures);
    return failures == 0 ? 0 : 1;
}
