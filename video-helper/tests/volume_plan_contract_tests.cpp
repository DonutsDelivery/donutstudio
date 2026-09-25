#include "support/volume_plan_fixture.h"
#include "visual_plan_resource_budget.h"
#include <functional>
#include <iostream>

int main()
{
    int failures = 0;
    const auto check = [&](bool condition, const char* message) {
        if (!condition) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
    };
    const auto original = volumetest::plan();
    visualvolume::Operation operation;
    std::string error;
    check(visualvolume::validatePlan(original, operation, error), "canonical graph is admitted");
    check(videowire::validateCompiledVisualLayerPlans({}, {original}, true, error), "helper whitelist admits native Volume");
    videowire::VisualPlanResourceUsage usage;
    check(videowire::accountVisualPlanResources(original, 96, 64, usage, error),
          "helper independently accounts the exact density and image resources");
    const auto reject = [&](const char* label, const std::function<void(videowire::CompiledVisualLayerPlan&)>& mutate) {
        auto malformed = original;
        mutate(malformed);
        visualvolume::Operation unchanged;
        unchanged.sourceStableId = 700;
        check(!visualvolume::validatePlan(malformed, unchanged, error)
              && !error.empty() && unchanged.sourceStableId == 700, label);
        check(!videowire::validateCompiledVisualLayerPlans({}, {malformed}, true, error), label);
    };
    reject("density/render controls cannot disagree", [](auto& p) {
        visualvolume::Operation altered; visualvolume::decode(p.operations[0].payloadXml, altered);
        altered.density = 0.1f; p.operations[0].payloadXml = visualvolume::encode(altered);
    });
    reject("forged stable IDs reject", [](auto& p) {
        visualvolume::Operation altered; visualvolume::decode(p.operations[0].payloadXml, altered);
        altered.sourceStableId += 1;
        p.operations[0].payloadXml = p.operations[1].payloadXml = visualvolume::encode(altered);
    });
    reject("zero node identity rejects", [](auto& p) { p.nodeIds[0] = p.operations[0].nodeId = 0; });
    reject("disconnected render rejects", [](auto& p) { p.edges.erase(p.edges.begin()); });
    reject("wrong image port rejects", [](auto& p) { p.edges[1].fromPort = 0; });
    reject("duplicate connection rejects", [](auto& p) { p.edges.push_back(p.edges.front()); });
    reject("extra output rejects", [](auto& p) { p.ports.push_back(p.ports[2]); });
    reject("forged density descriptor rejects", [](auto& p) { p.ports[0].dataType = "sdf"; });
    reject("forged image format rejects", [](auto& p) { p.ports[2].pixelFormat = "rgba16f"; });
    reject("extra operation rejects", [](auto& p) { p.operations.push_back({14, "video.out", "native-gpu", ""}); });
    reject("render backend cannot fall back", [](auto& p) { p.operations[1].backendCapability = "source-decode"; });
    reject("untrusted grant rejects", [](auto& p) { p.operations[0].runtimeGrantJson = "{}"; });
    reject("output payload rejects", [](auto& p) { p.operations[2].payloadXml = "<NodeParams path='x'/>"; });
    reject("extra wire bytes reject", [](auto& p) { p.operations[0].payloadXml += "00"; p.operations[1].payloadXml += "00"; });
    reject("unaccounted density rejects", [](auto& p) { p.sceneRecordCount = 0; });
    reject("forged frame count rejects", [](auto& p) { p.frameOutputCount = 2; });
    reject("forged liveness rejects", [](auto& p) { p.peakLiveFrameCount = 0; });
    reject("unpublished graph rejects", [](auto& p) { p.producerValidated = false; });
    reject("missing persisted revision rejects", [](auto& p) { p.structuralRevision = 0; });
    reject("transient projection rejects", [](auto& p) { p.identityMode = "transientLegacyProjection"; });
    std::cout << (failures ? "FAIL" : "PASS") << ": strict Volume plan admission\n";
    return failures ? 1 : 0;
}
