#include "../src/render_snapshot_json.h"
#include "../src/optical_flow_product_execution.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>

using json = nlohmann::json;

namespace
{
int failures = 0;
int checks = 0;

void check (bool condition, const char* message)
{
    ++checks;
    if (! condition)
    {
        ++failures;
        std::fprintf (stderr, "FAIL: %s\n", message);
    }
}

std::string noShader (const json&) { return {}; }

struct NoOpticalBackend final : arbitgpu::NativeOpticalFlowExecutionBackend
{
    videoopticalflow::BackendCapabilities opticalFlowCapabilities() const override { return {}; }
    arbitgpu::NativeOpticalFlowSubmission executeOpticalFlow (
        const videoopticalflow::AdmittedRequest&,
        const arbitgpu::NativeOpticalFlowInputResource&,
        const arbitgpu::NativeOpticalFlowInputResource&, bool) override { return {}; }
    void releaseOpticalFlowOutput (
        arbitgpu::NativeOpticalFlowOutputLifecycleHandle) noexcept override {}
};

json representativeSnapshot()
{
    return {
        { "authoringRevision", 12 },
        { "matteContentRevision", 7 },
        { "segments", {
            {
                { "sourceKind", "media" }, { "sourcePath", "/media/a.mov" },
                { "clipId", 10 }, { "trackLayer", 0 },
                { "inSec", 0.0 }, { "outSec", 3.0 }, { "rate", 1.0 },

                { "displayStartSec", 0.0 },
                { "matteAssetId", "matte-shared" }, { "matteState", "available" },
                { "matteAssetVersion", "content-v1" },
                { "matteContentReceipt", std::string(64, 'a') },
                { "matteFramePrefix", "mask-" }, { "matteFrameExtension", ".webp" },
                { "matteFirstFrame", 17 }, { "matteFrameDigits", 4 },
                { "matteCacheKey", std::string(64, 'a') }, { "matteFps", 30.0 }, { "matteFrames", 90 }
            },
            {
                { "sourceKind", "media" }, { "sourcePath", "/media/b.mov" },
                { "clipId", 20 }, { "trackLayer", 0 },
                { "inSec", 0.0 }, { "outSec", 3.0 }, { "rate", 1.0 },
                { "displayStartSec", 2.0 },
                { "matteAssetId", "matte-shared" }, { "matteState", "available" },
                { "matteAssetVersion", "content-v1" },
                { "matteContentReceipt", std::string(64, 'a') },
                { "matteFramePrefix", "mask-" }, { "matteFrameExtension", ".webp" },
                { "matteFirstFrame", 17 }, { "matteFrameDigits", 4 },
                { "matteCacheKey", std::string(64, 'a') }, { "matteFps", 30.0 },
                { "matteFrames", 90 },
                { "transition", {
                    { "type", 1 }, { "durationSec", 1.0 },
                    { "fromClipId", 10 }, { "toClipId", 20 }
                } }
            },
            {
                { "sourceKind", "adjustment" },
                { "sourcePath", "gen://adjustment" },
                { "clipId", 30 }, { "trackLayer", 1 },
                { "inSec", 0.0 }, { "outSec", 3.0 }, { "rate", 1.0 },
                { "displayStartSec", 0.0 }
            }
        } }
    };
}
}

int main()
{
    std::string error;
    {
        videowire::CompiledVisualLayerPlan view;
        view.clipId = 913;
        view.structuralRevision = 4;
        view.producerValidated = true;
        view.nodeIds = { 11, 12, 13, 14, 15 };
        view.nodeKinds = { "video.source", "video.layer.source", "visual.optical-flow",
                           "visual.motion.view", "video.out" };
        const opticalflowoperation::SourcePair pair { { { { 11, 0 }, { 12, 0 } } } };
        view.operations = {
            { 11, "video.source", "source-decode", opticalflowoperation::serializeSourceClip(913) },
            { 12, "video.layer.source", "source-decode", opticalflowoperation::serializeSourceClip(914) },
            { 13, "visual.optical-flow", "native-gpu", opticalflowoperation::serializeSourcePair(pair) },
            { 14, "visual.motion.view", "native-gpu", motionviewoperation::serialize(
                motionviewoperation::Convention::OpticalFirstToSecondTopLeft) },
            { 15, "video.out", "native-gpu", "" }
        };
        view.edges = { { 11, 0, 13, 0 }, { 12, 0, 13, 1 },
                       { 13, 2, 14, 0 }, { 14, 1, 15, 0 } };
        view.ports = {
            { 11, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
            { 12, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
            { 13, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
            { 13, 1, 1, "in", "frame", "image", "rgba8", "sRGB" },
            { 13, 2, 1, "out", "frame", "motionVectors", "rg16f", "unspecified" },
            { 14, 0, 1, "in", "frame", "motionVectors", "rg16f", "unspecified" },
            { 14, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
            { 15, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
        };
        check (videowire::validateCompiledVisualLayerPlans({}, {view}, false, error),
               "helper admits exact Optical Flow to Motion View Image route");
        videoopticalflow::AuthoredPairSelection selected;
        selected.plan = &view;
        selected.pair = pair;
        check (videoopticalflow::sameAuthoredPairPlan(view, selected),
               "preview and export retain the same saved Motion View plan");
        auto malformed = view;
        malformed.structuralRevision = 5;
        check (!videoopticalflow::sameAuthoredPairPlan(malformed, selected),
               "Motion View rejects a stale structural revision");
        malformed = view;
        malformed.edges[3].fromPort = 0;
        check (!videoopticalflow::sameAuthoredPairPlan(malformed, selected),
               "Motion View rejects an in-flight route change");
        malformed = view;
        malformed.operations[3].payloadXml = motionviewoperation::serialize(
            motionviewoperation::Convention::GeometricCurrentMinusPreviousYUp);
        check (!videowire::validateCompiledVisualLayerPlans({}, {malformed}, false, error),
               "helper rejects geometric Motion AOV convention on optical field");
        malformed = view;
        malformed.edges[3].fromPort = 0;
        check (!videowire::validateCompiledVisualLayerPlans({}, {malformed}, false, error),
               "helper rejects wrong Motion View output edge");
        malformed = view;
        malformed.ports[6].colorSpace = "linearSRGB";
        check (!videowire::validateCompiledVisualLayerPlans({}, {malformed}, false, error),
               "helper rejects a Motion View Image without sRGB semantics");
        malformed = view;
        malformed.operations[2].payloadXml = opticalflowoperation::serializeSourcePair(
            opticalflowoperation::SourcePair { { opticalflowoperation::SourceEndpoint { 12, 0 },
                                                  opticalflowoperation::SourceEndpoint { 11, 0 } } });
        check (!videowire::validateCompiledVisualLayerPlans({}, {malformed}, false, error),
               "helper rejects a source-pair payload that reverses saved A and B");
        malformed = view;
        malformed.structuralRevision = 0;
        check (!videowire::validateOpticalFlowSources({}, {malformed}, {}, 1, error),
               "helper rejects Motion View without a saved structural owner");
    }
    {
        videowire::CompiledVisualLayerPlan geometric;
        geometric.clipId = 912;
        geometric.structuralRevision = 4;
        geometric.producerValidated = true;
        geometric.nodeIds = {11, 12, 13};
        geometric.nodeKinds = {"visual.3d.scene.retained", "visual.3d.render.passes", "visual.motion.view"};
        geometric.operations = {
            {11, geometric.nodeKinds[0], "control-eval", {}},
            {12, geometric.nodeKinds[1], "native-gpu", {}},
            {13, geometric.nodeKinds[2], "native-gpu", motionviewoperation::serialize(
                motionviewoperation::Convention::GeometricCurrentMinusPreviousYUp)}
        };
        geometric.edges = {{11, 0, 12, 0}, {12, 4, 13, 0}};
        geometric.ports = {
            {11, 0, 1, "out", "control", "scene3D", "unspecified", "unspecified"},
            {12, 0, 1, "in", "control", "scene3D", "unspecified", "unspecified"},
            {12, 4, 1, "out", "frame", "motionVectors", "rg16f", "unspecified", "unspecified", "opaque"},
            {13, 0, 1, "in", "frame", "motionVectors", "rg16f", "unspecified", "linear", "opaque"},
            {13, 1, 1, "out", "frame", "image", "rgba8", "sRGB", "sRGB", "straight"}
        };
        check (!videowire::validateCompiledVisualLayerPlans({}, {geometric}, false, error),
               "snapshot refuses geometric Motion View despite existing compiler native-gpu classification");
        check (error == "Motion View requires a saved Optical Flow field",
               "geometric Motion View retains the exact unsupported snapshot refusal");
    }
    videowire::ResolvedVisualSnapshot preview;
    videowire::ResolvedVisualSnapshot exportSnapshot;
    videowire::ResolvedVisualSnapshot rejected;
    const auto wire = representativeSnapshot();
    check (videowire::parseSnapshotJson (wire, noShader, preview, error),
           "preview snapshot normalizes");
    check (videowire::parseSnapshotJson (wire, noShader, exportSnapshot, error),
           "export snapshot normalizes");
    auto automated=wire;
    automated["paramTimeline"]={{{"paramId","clip10/visual7/gravity"},{"atSec",0},{"value",1}},
                                {{"paramId","clip10/visual7/gravity"},{"atSec",.5},{"value",-1}}};
    videowire::ResolvedVisualSnapshot automatedPreview,automatedExport;
    check(videowire::parseSnapshotJson(automated,noShader,automatedPreview,error)
        && videowire::parseSnapshotJson(automated,noShader,automatedExport,error),
        "preview and export admit the same immutable automation timeline");
    if (automatedPreview.parameterTimeline && automatedExport.parameterTimeline)
    {
        double previewValue=0,exportValue=0;
        automatedPreview.parameterTimeline->sample("clip10/visual7/gravity",.5,previewValue);
        automatedExport.parameterTimeline->sample("clip10/visual7/gravity",.5,exportValue);
        check(previewValue==-1 && previewValue==exportValue,"historical values retain their exact destination and step");
    }
    automated["paramTimeline"][1]["atSec"]=0;
    check(!videowire::parseSnapshotJson(automated,noShader,rejected,error),
          "duplicate automation timestamps reject the candidate snapshot");
    automated["paramTimeline"]=json::array({{{"paramId","bad"},{"atSec",0},{"value","nan"}}});
    check(!videowire::parseSnapshotJson(automated,noShader,rejected,error),
          "malformed automation scalar rejects before publication");
    check (preview.authoringRevision == exportSnapshot.authoringRevision
           && preview.segments.size() == exportSnapshot.segments.size(),
           "preview and export receive equivalent snapshot identity");
    const auto previewB = std::find_if (preview.segments.begin(), preview.segments.end(),
                                        [] (const auto& segment) { return segment.clipId == 20; });
    const auto exportB = std::find_if (exportSnapshot.segments.begin(), exportSnapshot.segments.end(),
                                       [] (const auto& segment) { return segment.clipId == 20; });
    const auto adjustment = std::find_if (preview.segments.begin(), preview.segments.end(),
                                           [] (const auto& segment) { return segment.clipId == 30; });
    check (previewB != preview.segments.end() && exportB != exportSnapshot.segments.end()
           && previewB->transitionFromClipId == exportB->transitionFromClipId
           && previewB->matteCacheKey == exportB->matteCacheKey
           && adjustment != preview.segments.end()
           && adjustment->sourceKind == videowire::SourceKind::Adjustment,
           "owner inputs normalize identically");
    const auto previewA = std::find_if (preview.segments.begin(), preview.segments.end(),
                                        [] (const auto& segment) { return segment.clipId == 10; });
    check (previewA != preview.segments.end() && previewB != preview.segments.end()
           && previewA->matteAssetId == "matte-shared"
           && previewA->matteAssetVersion == "content-v1"
           && previewA->matteContentReceipt == std::string(64, 'a')
           && previewA->matteFramePrefix == "mask-" && previewA->matteFrameExtension == ".webp"
           && previewA->matteFirstFrame == 17 && previewA->matteFrameDigits == 4
           && previewA->matteAssetId == previewB->matteAssetId,
           "full matte identity and naming bind identically to two clips");

    const json runtimeGrant = {
        { "version", 1 }, { "kind", "shader" }, { "fingerprint", "exact-source" }
    };
    json customTransport;
    customTransport["visualLayerPlans"] = json::array({ {
        { "clipId", 10 }, { "structuralRevision", 1 }, { "valid", true },
        { "descriptorCount", 2 }, { "operationCount", 2 }, { "sceneRecordCount", 0 },
        { "frameOutputCount", 1 }, { "peakLiveFrameCount", 1 },
        { "allocatedFrameSlotCount", 1 }, { "compileDurationMicros", 1 },
        { "nodeKinds", json::array({ "visual.shader.custom", "video.out" }) },
        { "nodeIds", json::array({ 61, 62 }) },
        { "ports", json::array({
            { { "nodeId", 61 }, { "port", 0 }, { "channels", 1 }, { "direction", "out" },
              { "carrier", "frame" }, { "dataType", "image" }, { "pixelFormat", "rgba8" },
              { "colorSpace", "sRGB" } },
            { { "nodeId", 62 }, { "port", 0 }, { "channels", 1 }, { "direction", "in" },
              { "carrier", "frame" }, { "dataType", "image" }, { "pixelFormat", "rgba8" },
              { "colorSpace", "sRGB" } }
        }) },
        { "edges", json::array({ {
            { "fromNodeId", 61 }, { "fromPort", 0 }, { "toNodeId", 62 }, { "toPort", 0 }
        } }) },
        { "operations", json::array({
            { { "nodeId", 61 }, { "kind", "visual.shader.custom" },
              { "backendCapability", "native-gpu" }, { "payloadXml", "exact-payload" },
              { "runtimeGrant", runtimeGrant } },
            { { "nodeId", 62 }, { "kind", "video.out" },
              { "backendCapability", "native-gpu" }, { "payloadXml", "" } }
        }) }
    } });
    const bool customTransportAccepted = videowire::parseSnapshotJson(
        customTransport, noShader, rejected, error);
    check (customTransportAccepted
           && rejected.visualLayerPlans.size() == 1
           && rejected.visualLayerPlans[0].operations.size() == 2
           && json::parse(rejected.visualLayerPlans[0].operations[0].runtimeGrantJson) == runtimeGrant,
           "custom shader runtime grant survives processor-to-helper JSON admission");
    check (customTransportAccepted
           && rejected.visualLayerPlans[0].ports[0].transfer == "unspecified"
           && rejected.visualLayerPlans[0].ports[0].alpha == "unspecified",
           "older compiled port wire defaults transfer and alpha without changing admission");

    auto typedTransport = customTransport;
    for (auto& port : typedTransport["visualLayerPlans"][0]["ports"])
    {
        port["transfer"] = "sRGB";
        port["alpha"] = "straight";
    }
    check (videowire::parseSnapshotJson(typedTransport, noShader, rejected, error)
           && rejected.visualLayerPlans[0].ports[0].transfer == "sRGB"
           && rejected.visualLayerPlans[0].ports[1].alpha == "straight",
           "compiled port transfer and alpha survive snapshot JSON admission");
    auto mismatchedTransport = typedTransport;
    mismatchedTransport["visualLayerPlans"][0]["ports"][1]["transfer"] = "linear";
    check (! videowire::parseSnapshotJson(mismatchedTransport, noShader, rejected, error)
           && error.find("incompatible typed edge") != std::string::npos,
           "helper rejects an implicit transfer conversion on a typed edge");
    mismatchedTransport = typedTransport;
    mismatchedTransport["visualLayerPlans"][0]["ports"][1]["alpha"] = "premultiplied";
    check (! videowire::parseSnapshotJson(mismatchedTransport, noShader, rejected, error)
           && error.find("incompatible typed edge") != std::string::npos,
           "helper rejects an implicit alpha conversion on a typed edge");
    mismatchedTransport = typedTransport;
    mismatchedTransport["visualLayerPlans"][0]["ports"][0]["transfer"] = 1;
    check (! videowire::parseSnapshotJson(mismatchedTransport, noShader, rejected, error)
           && error == "compiled visual port has malformed transfer or alpha metadata",
           "helper rejects malformed transfer metadata");
    mismatchedTransport = typedTransport;
    mismatchedTransport["visualLayerPlans"][0]["ports"][0]["alpha"] = "unknown-alpha";
    check (! videowire::parseSnapshotJson(mismatchedTransport, noShader, rejected, error),
           "helper rejects unknown alpha metadata");

    auto rejectedDepthOperation = wire;
    rejectedDepthOperation["visualLayerPlans"] = json::array({ {
        { "clipId", 10 }, { "operations", json::array({ {
            { "nodeId", 61 }, { "kind", "visual.depth.asset" },
            { "backendCapability", "source-decode" },
            { "payloadXml", "<DepthAssetBinding sequencePath=\"/untrusted\"/>" }
        } }) }
    } });
    check (! videowire::parseSnapshotJson(rejectedDepthOperation, noShader, rejected, error)
           && error == "depth payload does not match the executable receipt schema",
           "helper rejects malformed executable depth capability and payload");

    videowire::CompiledVisualLayerPlan geometryPlan;
    geometryPlan.clipId = 10;
    geometryPlan.structuralRevision = 3;
    geometryPlan.producerValidated = true;
    // The producer intentionally serializes the collapsed Render 3D operation
    // before a reactive Frame prerequisite. Admission must schedule the typed
    // dependency edge rather than treating document order as execution order.
    geometryPlan.nodeKinds = { "geometry.core.runtime", "video.source" };
    geometryPlan.nodeIds = { 71, 91 };
    geometryPlan.operations = {
        { 71, "geometry.core.runtime", "native-gpu", "lowered-geometry" },
        { 91, "video.source", "source-decode", "" }
    };
    geometryPlan.ports = {
        { 71, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 71, 12, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 91, 0, 1, "out", "frame", "image", "rgba8", "sRGB" }
    };
    geometryPlan.edges = { { 91, 0, 71, 12 } };
    error.clear();
    check (videowire::validateCompiledVisualLayerPlans({}, { geometryPlan }, false, error),
           "snapshot admits collapsed Geometry Core with a later-serialized reactive Frame prerequisite");

    auto harmonicPlan = geometryPlan;
    harmonicPlan.nodeKinds = { "geometry.harmonic-links.runtime" };
    harmonicPlan.nodeIds = { 72 };
    harmonicPlan.operations = { { 72, "geometry.harmonic-links.runtime", "native-gpu", "harmonic" } };
    harmonicPlan.ports = { { 72, 1, 1, "out", "frame", "image", "rgba8", "sRGB" } };
    harmonicPlan.edges.clear();
    error.clear();
    check (videowire::validateCompiledVisualLayerPlans({}, { harmonicPlan }, false, error),
           "snapshot admits the executable collapsed Harmonic Link Geometry terminal without video.out");

    auto retainedScenePlan = geometryPlan;
    retainedScenePlan.nodeKinds = { "visual.3d.render", "visual.3d.scene.retained" };
    retainedScenePlan.nodeIds = { 81, 80 };
    retainedScenePlan.operations = {
        { 81, "visual.3d.render", "native-gpu", "retained-render" },
        { 80, "visual.3d.scene.retained", "control-eval", "" }
    };
    retainedScenePlan.ports = {
        { 81, 0, 1, "in", "control", "scene3D", "unspecified", "unspecified" },
        { 81, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 80, 0, 1, "out", "control", "scene3D", "unspecified", "unspecified" }
    };
    retainedScenePlan.edges = { { 80, 0, 81, 0 } };
    error.clear();
    check (videowire::validateCompiledVisualLayerPlans({}, { retainedScenePlan }, false, error),
           "snapshot topologically admits retained Scene3D before its collapsed native render terminal");

    auto advancedPassPlan = harmonicPlan;
    advancedPassPlan.nodeKinds = { "visual.3d.render.passes" };
    advancedPassPlan.nodeIds = { 82 };
    advancedPassPlan.operations = { { 82, "visual.3d.render.passes", "native-gpu", "passes" } };
    advancedPassPlan.ports = { { 82, 0, 1, "out", "frame", "image", "rgba8", "sRGB" } };
    error.clear();
    check (videowire::validateCompiledVisualLayerPlans({}, { advancedPassPlan }, false, error),
           "snapshot admits the executable collapsed advanced Render Passes terminal");

    auto sdfOutputPlan = harmonicPlan;
    sdfOutputPlan.nodeKinds = { "visual.sdf.sphere", "visual.sdf.raymarch", "video.out" };
    sdfOutputPlan.nodeIds = { 70, 80, 90 };
    // Operation order is deliberately independent of node serialization order.
    sdfOutputPlan.operations = {
        { 90, "video.out", "native-gpu", "" },
        { 70, "visual.sdf.sphere", "control-eval", "sphere" },
        { 80, "visual.sdf.raymarch", "native-gpu", "raymarch" }
    };
    sdfOutputPlan.ports = {
        { 70, 0, 1, "out", "control", "sdf", "unspecified", "unspecified" },
        { 80, 0, 1, "in", "control", "sdf", "unspecified", "unspecified" },
        { 80, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 80, 2, 1, "out", "frame", "depth", "r32f", "unspecified" },
        { 90, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    sdfOutputPlan.edges = { { 70, 0, 80, 0 }, { 80, 1, 90, 0 } };
    error.clear();
    check (videowire::validateCompiledVisualLayerPlans({}, { sdfOutputPlan }, false, error),
           "snapshot admits an exact typed SDF raymarch feeding video.out with independent operation order");

    videowire::CompiledVisualLayerPlan sourcelessLegacy;
    sourcelessLegacy.producerValidated = true;
    sourcelessLegacy.clipId = 1;
    sourcelessLegacy.nodeKinds = { "video.effects", "video.out" };
    sourcelessLegacy.nodeIds = { 61, 62 };
    sourcelessLegacy.operations = {
        { 61, "video.effects", "native-gpu", "" },
        { 62, "video.out", "native-gpu", "" }
    };
    sourcelessLegacy.ports = {
        { 61, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 62, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    sourcelessLegacy.edges = { { 61, 1, 62, 0 } };
    error.clear();
    const bool sourcelessLegacyAccepted = videowire::validateCompiledVisualLayerPlans(
        {}, { sourcelessLegacy }, false, error);
    if (sourcelessLegacyAccepted || error != "visual layer plan has an unsupported production terminal")
        std::fprintf (stderr, "sourceless legacy plan: accepted=%d error=%s\n",
                      sourcelessLegacyAccepted, error.c_str());
    check (! sourcelessLegacyAccepted
           && error == "visual layer plan has an unsupported production terminal",
           "video.out alone does not classify a sourceless regular graph as a collapsed native plan");

    auto malformedGeometry = geometryPlan;
    malformedGeometry.ports[2].dataType = "depth";
    error.clear();
    check (! videowire::validateCompiledVisualLayerPlans({}, { malformedGeometry }, false, error)
           && error == "bounded visual DAG has an incompatible typed edge",
           "collapsed Geometry Core still rejects malformed typed Frame dependencies");

    auto unknownCollapsed = harmonicPlan;
    unknownCollapsed.nodeKinds = { "geometry.future.runtime" };
    unknownCollapsed.operations[0].kind = "geometry.future.runtime";
    error.clear();
    check (! videowire::validateCompiledVisualLayerPlans({}, { unknownCollapsed }, false, error)
           && error == "visual layer plan contains unsupported typed operation: geometry.future.runtime",
           "snapshot names and rejects unknown collapsed operation kinds");

    auto importedTyped = retainedScenePlan;
    importedTyped.nodeKinds = { "visual.3d.imported-animation", "visual.3d.transform",
        "visual.3d.camera.perspective", "visual.3d.light.directional", "visual.3d.render" };
    importedTyped.nodeIds = { 201, 202, 203, 204, 205 };
    importedTyped.operations = {
        { 201, "visual.3d.imported-animation", "source-decode", "" },
        { 202, "visual.3d.transform", "control-eval", "" },
        { 203, "visual.3d.camera.perspective", "control-eval", "" },
        { 204, "visual.3d.light.directional", "control-eval", "" },
        { 205, "visual.3d.render", "native-gpu", "" } };
    importedTyped.ports = {
        { 201, 0, 1, "out", "control", "scene3D", "unspecified", "unspecified" },
        { 205, 0, 1, "in", "control", "scene3D", "unspecified", "unspecified" },
        { 205, 1, 1, "out", "frame", "image", "rgba8", "sRGB" }
    };
    importedTyped.edges = { { 201, 0, 205, 0 } };
    error.clear();
    check (videowire::validateCompiledVisualLayerPlans({}, { importedTyped }, false, error),
           "snapshot whitelist admits imported transform camera and directional-light operations");
    importedTyped.nodeKinds.insert(importedTyped.nodeKinds.begin() + 1, "visual.3d.light.point");
    importedTyped.nodeIds.insert(importedTyped.nodeIds.begin() + 1, 206);
    importedTyped.operations.insert(importedTyped.operations.begin() + 1,
        { 206, "visual.3d.light.point", "control-eval", "" });
    error.clear();
    check (! videowire::validateCompiledVisualLayerPlans({}, { importedTyped }, false, error)
           && error == "visual layer plan contains unsupported typed operation: visual.3d.light.point",
           "snapshot whitelist rejects imported operation without native executor admission");

    const std::string validDepth = "<DepthAssetBinding depthAssetId=\"depth-1\" depthAssetVersion=\"v1\" state=\"available\" contract=\"parked-metadata\" pendingExecutionSeam=\"depth-consuming-operation\"/>";
    for (const char* extra : { "path", "file", "uri", "decoder", "upload", "futureField" })
    {
        auto injected = rejectedDepthOperation;
        injected["visualLayerPlans"][0]["operations"][0]["backendCapability"] = "parked-metadata";
        auto payload = validDepth;
        payload.insert(payload.size() - 2, std::string(" ") + extra + "=\"x\"");
        injected["visualLayerPlans"][0]["operations"][0]["payloadXml"] = payload;
        check (! videowire::parseSnapshotJson(injected, noShader, rejected, error),
               "snapshot admission rejects unknown depth payload attributes");
    }

    auto invalidTransition = wire;
    invalidTransition["segments"][1]["transition"]["fromClipId"] = 999;
    check (! videowire::parseSnapshotJson (invalidTransition, noShader, rejected, error),
           "explicit missing transition owner fails without legacy inference");

    auto invalidKind = wire;
    invalidKind["segments"][0]["sourceKind"] = "future-kind";
    check (! videowire::parseSnapshotJson (invalidKind, noShader, rejected, error),
           "explicit unknown source kind fails without path fallback");

    auto invalidMatte = wire;
    invalidMatte["segments"][1].erase ("matteFrames");
    check (! videowire::parseSnapshotJson (invalidMatte, noShader, rejected, error),
           "incomplete explicit matte binding fails");
    auto conflictingMatte = wire;
    conflictingMatte["segments"][1]["matteCacheKey"] = std::string(64, 'b');
    check (! videowire::parseSnapshotJson(conflictingMatte, noShader, rejected, error)
           && error == "duplicate matte asset authority",
           "duplicate matte IDs cannot add conflicting path authority");
    auto conflictingVersion = wire;
    conflictingVersion["segments"][1]["matteAssetVersion"] = "content-v2";
    check (! videowire::parseSnapshotJson(conflictingVersion, noShader, rejected, error)
           && error == "duplicate matte asset authority",
           "duplicate matte IDs cannot add conflicting version authority");
    auto invalidNaming = wire;
    invalidNaming["segments"][0]["matteFramePrefix"] = "../escape";
    check (! videowire::parseSnapshotJson(invalidNaming, noShader, rejected, error),
           "unsafe matte naming is rejected");
    auto nonFiniteMatte = wire;
    nonFiniteMatte["segments"][0]["matteFps"] = std::numeric_limits<double>::infinity();
    check (! videowire::parseSnapshotJson(nonFiniteMatte, noShader, rejected, error),
           "non-finite matte fps is rejected");
    auto duplicateClip = wire;
    duplicateClip["segments"][1]["clipId"] = 10;
    check (! videowire::parseSnapshotJson(duplicateClip, noShader, rejected, error)
           && error == "snapshot duplicates or overlaps a stable clip owner",
           "overlapping segments for one stable clip owner are rejected");
    auto splitClip = wire;
    splitClip["segments"][1]["clipId"] = 10;
    splitClip["segments"][1]["displayStartSec"] = 3.0;
    splitClip["segments"][1].erase("transition");
    check (videowire::parseSnapshotJson(splitClip, noShader, rejected, error)
           && rejected.segments.size() == 3,
           "non-overlapping segments for one clip owner are accepted");
    auto traversalMatte = wire;
    traversalMatte["segments"][0]["matteCacheKey"] = "../outside";
    check (! videowire::parseSnapshotJson(traversalMatte, noShader, rejected, error)
           && error == "malformed or unauthorised matte binding",
           "matte path additions reject traversal");
    auto staleMatte = wire;
    staleMatte["segments"][0]["matteState"] = "stale";
    staleMatte["segments"][1]["matteState"] = "stale";
    check (videowire::parseSnapshotJson (staleMatte, noShader, rejected, error)
           && ! videowire::validateSnapshotResources (rejected,
               [] (const std::string&) { return true; }, error)
           && error == "matte asset is stale: matte-shared",
           "stale asset is preserved and rejected without regeneration");

    json legacy = {
        { "segments", {{
            { "sourcePath", "gen://adjustment" }, { "clipId", 7 },
            { "inSec", 0.0 }, { "outSec", 1.0 }, { "rate", 1.0 }
        }} }
    };
    check (videowire::parseSnapshotJson (legacy, noShader, rejected, error)
           && rejected.segments[0].sourceKind == videowire::SourceKind::Adjustment,
           "legacy sentinel normalizes through absence-only compatibility path");

    check (! videowire::validateSnapshotResources (
               preview,
               [] (const std::string& path)
               { return path == "/media/a.mov" || path == "/media/b.mov"
                     || path == "/mattes/b"; },
               error),
           "matte authority remains non-executable without a trusted cache root");
    check (! videowire::validateSnapshotResources (
               preview,
               [] (const std::string& path) { return path != "/media/a.mov"; },
               error)
           && error == "media source is missing: /media/a.mov",
           "resource that became stale fails before matte execution admission");
    check (videowire::validateSnapshotResources (
               rejected,
               [] (const std::string&) { return false; },
               error),
           "absence-only generator compatibility does not require media resources");
    check (! videowire::validateSnapshotResources (
               preview,
               [] (const std::string& path)
               { return path == "/media/a.mov" || path == "/media/b.mov"
                     || path == "/mattes/b"; },
               error),
           "matte binding cannot bypass fail-closed execution admission");

    std::vector<videowire::RawRenderSegment> raw (1);
    raw[0].clipId = 44;
    raw[0].sourcePath = "/media/unchanged.wav";
    raw[0].inSec = 0.0;
    raw[0].outSec = 1.0;
    const auto originalPath = raw[0].sourcePath;
    videowire::CompiledVisualLayerPlan compiledPlan;
    compiledPlan.clipId = 44;
    compiledPlan.producerValidated = true;
    compiledPlan.nodeKinds = {
        "video.legacy.source", "video.legacy.retime", "video.legacy.transform",
        "video.legacy.effects", "video.out"
    };
    check (videowire::normalizeSnapshot (raw, std::vector<videowire::CompiledVisualLayerPlan> { compiledPlan }, {}, 1, true, rejected, error)
           && rejected.visualLayerPlans.size() == 1
           && rejected.segments.size() == 1,
           "fixed visual layer plan lowers onto the existing render segment");
    auto typedPlan = compiledPlan;
    typedPlan.nodeKinds = { "video.source", "video.out" };
    typedPlan.nodeIds = { 101, 102 };
    typedPlan.ports = {
        { 101, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 102, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    typedPlan.edges = { { 101, 0, 102, 0 } };
    typedPlan.operations = {
        { 101, "video.source", "source-decode", "" },
        { 102, "video.out", "native-gpu", "" }
    };
    check (videowire::normalizeSnapshot (raw, std::vector<videowire::CompiledVisualLayerPlan> { typedPlan }, {}, 2, true, rejected, error),
           "typed node, port, and edge bindings are admitted");
    opticalflowoperation::SourcePair flowPair;
    flowPair.images = { opticalflowoperation::SourceEndpoint { 101, 0 },
                        opticalflowoperation::SourceEndpoint { 103, 0 } };
    const auto flowPayload = opticalflowoperation::serializeSourcePair(flowPair);
    auto flowWire = json {
        { "authoringRevision", 2 },
        { "snapshotGeneration", 7 },
        { "opticalFlowSources", json::array({
            { { "clipId", 44 }, { "sourceKind", "media" },
              { "sourcePath", "/media/original-a.mov" }, { "sourceGeneration", 3 },
              { "snapshotGeneration", 7 } },
            { { "clipId", 45 }, { "sourceKind", "media" },
              { "sourcePath", "/media/original-b.mov" }, { "sourceGeneration", 5 },
              { "snapshotGeneration", 7 } } }) },
        { "segments", json::array({ {
            { "sourceKind", "media" }, { "sourcePath", "/media/a.mov" },
            { "clipId", 44 }, { "inSec", 0.0 }, { "outSec", 1.0 },
            { "rate", 1.0 }, { "displayStartSec", 0.0 } } }) },
        { "visualLayerPlans", json::array({ {
            { "clipId", 44 }, { "structuralRevision", 2 }, { "valid", true },
            { "descriptorCount", 5 }, { "operationCount", 3 },
            { "sceneRecordCount", 0 }, { "frameOutputCount", 3 },
            { "peakLiveFrameCount", 3 }, { "allocatedFrameSlotCount", 3 },
            { "compileDurationMicros", 1 },
            { "nodeKinds", json::array({ "video.source", "video.layer.source", "visual.optical-flow" }) },
            { "nodeIds", json::array({ 101, 103, 102 }) },
            { "edges", json::array({
                { { "fromNodeId", 101 }, { "fromPort", 0 }, { "toNodeId", 102 }, { "toPort", 0 } },
                { { "fromNodeId", 103 }, { "fromPort", 0 }, { "toNodeId", 102 }, { "toPort", 1 } } }) },
            { "ports", json::array({
                { { "nodeId", 101 }, { "port", 0 }, { "channels", 1 }, { "direction", "out" },
                  { "carrier", "frame" }, { "dataType", "image" }, { "pixelFormat", "rgba8" }, { "colorSpace", "sRGB" } },
                { { "nodeId", 103 }, { "port", 0 }, { "channels", 1 }, { "direction", "out" },
                  { "carrier", "frame" }, { "dataType", "image" }, { "pixelFormat", "rgba8" }, { "colorSpace", "sRGB" } },
                { { "nodeId", 102 }, { "port", 0 }, { "channels", 1 }, { "direction", "in" },
                  { "carrier", "frame" }, { "dataType", "image" }, { "pixelFormat", "rgba8" }, { "colorSpace", "sRGB" } },
                { { "nodeId", 102 }, { "port", 1 }, { "channels", 1 }, { "direction", "in" },
                  { "carrier", "frame" }, { "dataType", "image" }, { "pixelFormat", "rgba8" }, { "colorSpace", "sRGB" } },
                { { "nodeId", 102 }, { "port", 2 }, { "channels", 1 }, { "direction", "out" },
                  { "carrier", "frame" }, { "dataType", "motionVectors" }, { "pixelFormat", "rg16f" }, { "colorSpace", "unspecified" } } }) },
            { "operations", json::array({
                { { "nodeId", 101 }, { "kind", "video.source" }, { "backendCapability", "source-decode" }, { "payloadXml", opticalflowoperation::serializeSourceClip(44) } },
                { { "nodeId", 103 }, { "kind", "video.layer.source" }, { "backendCapability", "source-decode" }, { "payloadXml", opticalflowoperation::serializeSourceClip(45) } },
                { { "nodeId", 102 }, { "kind", "visual.optical-flow" }, { "backendCapability", "native-gpu" }, { "payloadXml", flowPayload } } }) }
        } }) }
    };
    check (videowire::parseSnapshotJson(flowWire, noShader, rejected, error)
           && rejected.visualLayerPlans[0].operations[2].payloadXml == flowPayload
           && rejected.opticalFlowSources.size() == 2
           && rejected.opticalFlowSources[1].sourcePath == "/media/original-b.mov"
           && rejected.snapshotGeneration == 7,
           "snapshot parse retains the ordered Optical Flow sources and original media catalog");
    videoopticalflow::AuthoredPairSelection flowSelection;
    check (videoopticalflow::selectAuthoredPair(rejected, 44, flowSelection).empty()
           && flowSelection.plan == &rejected.visualLayerPlans[0]
           && flowSelection.pair.images == flowPair.images,
           "Optical Flow worker selects the saved ordered pair from its current snapshot");
    check (videoopticalflow::sameAuthoredPairPlan(
               rejected.visualLayerPlans[0], flowSelection),
           "export plan matches its exact saved Optical Flow pair");
    auto blurredWire = flowWire;
    auto& blurredPlan = blurredWire["visualLayerPlans"][0];
    visualtemporalsampling::Payload blurPayload;
    blurPayload.mode = visualtemporalsampling::Mode::motionBlur;
    blurPayload.historyWeight = 0.0f;
    blurPayload.jitterSpread = 0.0f;
    blurPayload.sampleCount = 8;
    blurPayload.shutterAngleDegrees = 180.0f;
    blurPayload.motionScale = 1.0f;
    blurredPlan["nodeKinds"] = json::array({ "video.source", "video.layer.source",
        "visual.optical-flow", "visual.motion-blur", "video.out" });
    blurredPlan["nodeIds"] = json::array({ 101, 103, 102, 104, 105 });
    blurredPlan["descriptorCount"] = 9;
    blurredPlan["operationCount"] = 5;
    blurredPlan["frameOutputCount"] = 4;
    blurredPlan["peakLiveFrameCount"] = 4;
    blurredPlan["allocatedFrameSlotCount"] = 4;
    blurredPlan["edges"].push_back({ { "fromNodeId", 101 }, { "fromPort", 0 },
        { "toNodeId", 104 }, { "toPort", 0 } });
    blurredPlan["edges"].push_back({ { "fromNodeId", 102 }, { "fromPort", 2 },
        { "toNodeId", 104 }, { "toPort", 1 } });
    blurredPlan["edges"].push_back({ { "fromNodeId", 104 }, { "fromPort", 2 },
        { "toNodeId", 105 }, { "toPort", 0 } });
    blurredPlan["ports"].push_back({ { "nodeId", 104 }, { "port", 0 }, { "channels", 1 },
        { "direction", "in" }, { "carrier", "frame" }, { "dataType", "image" },
        { "pixelFormat", "rgba8" }, { "colorSpace", "sRGB" } });
    blurredPlan["ports"].push_back({ { "nodeId", 104 }, { "port", 1 }, { "channels", 2 },
        { "direction", "in" }, { "carrier", "frame" }, { "dataType", "motionVectors" },
        { "pixelFormat", "rg16f" }, { "colorSpace", "unspecified" } });
    blurredPlan["ports"].push_back({ { "nodeId", 104 }, { "port", 2 }, { "channels", 1 },
        { "direction", "out" }, { "carrier", "frame" }, { "dataType", "image" },
        { "pixelFormat", "rgba8" }, { "colorSpace", "sRGB" } });
    blurredPlan["ports"].push_back({ { "nodeId", 105 }, { "port", 0 }, { "channels", 1 },
        { "direction", "in" }, { "carrier", "frame" }, { "dataType", "image" },
        { "pixelFormat", "rgba8" }, { "colorSpace", "sRGB" } });
    blurredPlan["operations"].push_back({ { "nodeId", 104 }, { "kind", "visual.motion-blur" },
        { "backendCapability", "temporal-sampling" },
        { "payloadXml", visualtemporalsampling::serialize (blurPayload) } });
    blurredPlan["operations"].push_back({ { "nodeId", 105 }, { "kind", "video.out" },
        { "backendCapability", "native-gpu" }, { "payloadXml", "" } });
    videowire::ResolvedVisualSnapshot blurredSnapshot;
    check (videowire::parseSnapshotJson (blurredWire, noShader, blurredSnapshot, error)
        && videoopticalflow::hasMotionBlur (blurredSnapshot.visualLayerPlans[0]),
        "saved Optical Flow MotionVectors feed Motion Blur");
    check (videowire::parseSnapshotJson (json::parse (blurredWire.dump()), noShader,
        blurredSnapshot, error), "Optical Flow Motion Blur survives snapshot serialization");
    std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> blankBlur;
    NoOpticalBackend noOpticalBackend;
    check (! blurredSnapshot.visualLayerPlans.empty()
        && videoopticalflow::renderAuthoredMotionBlur (
               blurredSnapshot.visualLayerPlans[0], {}, 7, 0.0, 1,
               noOpticalBackend, blankBlur)
               == "Motion Blur requires the current saved pair, time, helper, and retained field"
        && blankBlur == nullptr,
        "Motion Blur rejects a blank Image without a retained field");
    auto wrongBlur = blurredWire;
    wrongBlur["visualLayerPlans"][0]["edges"][2]["fromNodeId"] = 103;
    check (! videowire::parseSnapshotJson (wrongBlur, noShader, blurredSnapshot, error)
        && error == "Optical Flow Motion Blur requires its exact typed Image and MotionVectors route",
        "Motion Blur rejects the wrong saved Image source");
    wrongBlur = blurredWire;
    wrongBlur["visualLayerPlans"][0]["operations"][2]["payloadXml"] = "<OpticalFlowPair/>";
    check (! videowire::parseSnapshotJson (wrongBlur, noShader, blurredSnapshot, error)
        && error == "Optical Flow has a malformed ordered source-pair payload",
        "Motion Blur rejects a malformed pair");
    wrongBlur = blurredWire;
    wrongBlur["visualLayerPlans"][0]["structuralRevision"] = 0;
    check (! videowire::parseSnapshotJson (wrongBlur, noShader, blurredSnapshot, error)
        && error == "Optical Flow requires its saved graph owner and structural revision",
        "Motion Blur rejects a stale graph revision");
    auto changedFlowPlan = rejected.visualLayerPlans[0];
    changedFlowPlan.operations[1].payloadXml = opticalflowoperation::serializeSourceClip(46);
    check (! videoopticalflow::sameAuthoredPairPlan(changedFlowPlan, flowSelection),
           "export rejects a changed saved source under the same graph revision");
    changedFlowPlan = rejected.visualLayerPlans[0];
    changedFlowPlan.operations[1].backendCapability = "native-gpu";
    check (! videoopticalflow::sameAuthoredPairPlan(changedFlowPlan, flowSelection),
           "export rejects a changed saved source backend under the same graph revision");
    auto staleFlowSnapshot = rejected;
    staleFlowSnapshot.opticalFlowSources[1].snapshotGeneration = 8;
    check (!videoopticalflow::selectAuthoredPair(staleFlowSnapshot, 44, flowSelection).empty()
           && flowSelection.plan == nullptr,
           "Optical Flow worker rejects a stale source catalog before decode");
    check (!videoopticalflow::selectAuthoredPair(rejected, 45, flowSelection).empty()
           && flowSelection.plan == nullptr,
           "Optical Flow worker rejects a different graph owner");
    auto invalidCatalog = flowWire;
    invalidCatalog.erase("opticalFlowSources");
    check (! videowire::parseSnapshotJson(invalidCatalog, noShader, rejected, error)
           && error == "Optical Flow source catalog does not match its operations",
           "Optical Flow requires a source catalog");
    invalidCatalog = flowWire;
    invalidCatalog["opticalFlowSources"][1] = invalidCatalog["opticalFlowSources"][0];
    check (! videowire::parseSnapshotJson(invalidCatalog, noShader, rejected, error)
           && error == "Optical Flow source catalog does not match its operations",
           "Optical Flow rejects duplicate catalog entries");
    invalidCatalog = flowWire;
    std::swap(invalidCatalog["opticalFlowSources"][0], invalidCatalog["opticalFlowSources"][1]);
    check (! videowire::parseSnapshotJson(invalidCatalog, noShader, rejected, error)
           && error == "Optical Flow source catalog does not match its operations",
           "Optical Flow preserves the ordered source operation identities");
    invalidCatalog = flowWire;
    invalidCatalog["opticalFlowSources"][1]["sourceKind"] = "generator";
    check (! videowire::parseSnapshotJson(invalidCatalog, noShader, rejected, error)
           && error == "Optical Flow source catalog has an invalid saved media identity",
           "Optical Flow rejects a non-media catalog entry");
    invalidCatalog = flowWire;
    invalidCatalog["opticalFlowSources"][1]["sourcePath"] = "";
    check (! videowire::parseSnapshotJson(invalidCatalog, noShader, rejected, error)
           && error == "Optical Flow source catalog has an invalid saved media identity",
           "Optical Flow rejects a missing saved media path");
    invalidCatalog = flowWire;
    invalidCatalog["opticalFlowSources"][1]["sourceGeneration"] = 0;
    check (! videowire::parseSnapshotJson(invalidCatalog, noShader, rejected, error)
           && error == "Optical Flow source catalog has an invalid saved media identity",
           "Optical Flow rejects a source without a generation");
    invalidCatalog = flowWire;
    invalidCatalog["opticalFlowSources"][1]["snapshotGeneration"] = 8;
    check (! videowire::parseSnapshotJson(invalidCatalog, noShader, rejected, error)
           && error == "Optical Flow source catalog has an invalid saved media identity",
           "Optical Flow rejects a catalog from another snapshot");
    invalidCatalog = flowWire;
    invalidCatalog["visualLayerPlans"][0]["operations"][1]["payloadXml"]
        = opticalflowoperation::serializeSourceClip(46);
    check (! videowire::parseSnapshotJson(invalidCatalog, noShader, rejected, error)
           && error == "Optical Flow source catalog does not match its operations",
           "Optical Flow rejects an operation and catalog mismatch");
    invalidCatalog = flowWire;
    invalidCatalog["visualLayerPlans"][0]["structuralRevision"] = 0;
    check (! videowire::parseSnapshotJson(invalidCatalog, noShader, rejected, error)
           && error == "Optical Flow requires its saved graph owner and structural revision",
           "Optical Flow retains the authored owner and structural revision");
    auto invalidFlow = flowWire;
    invalidFlow["visualLayerPlans"][0]["operations"][2]["payloadXml"] = "<OpticalFlowPair/>";
    check (! videowire::parseSnapshotJson(invalidFlow, noShader, rejected, error)
           && error == "Optical Flow has a malformed ordered source-pair payload",
           "snapshot parse rejects a malformed Optical Flow pair");
    invalidFlow = flowWire;
    invalidFlow["visualLayerPlans"][0]["edges"].erase(1);
    check (! videowire::parseSnapshotJson(invalidFlow, noShader, rejected, error)
           && error == "Optical Flow source-pair payload does not match its ordered Image edges",
           "snapshot parse rejects a missing ordered Optical Flow edge");
    invalidFlow = flowWire;
    auto duplicatePair = flowPair;
    duplicatePair.images[1] = duplicatePair.images[0];
    invalidFlow["visualLayerPlans"][0]["operations"][2]["payloadXml"]
        = opticalflowoperation::serializeSourcePair(duplicatePair);
    check (! videowire::parseSnapshotJson(invalidFlow, noShader, rejected, error)
           && error == "Optical Flow has duplicate Image source endpoints",
           "snapshot parse rejects duplicate Optical Flow source identities");
    auto importedScenePlan = typedPlan;
    importedScenePlan.nodeKinds = {
        "visual.3d.imported-animation", "visual.surface.constant.vec3",
        "visual.surface.material", "visual.3d.render"
    };
    importedScenePlan.nodeIds = { 201, 203, 204, 202 };
    importedScenePlan.ports = {
        { 201, 4, 1, "out", "control", "scene3D", "unspecified", "unspecified" },
        { 203, 0, 3, "out", "control", "vec3", "unspecified", "unspecified" },
        { 204, 0, 3, "in", "control", "vec3", "unspecified", "unspecified" },
        { 204, 10, 1, "out", "control", "surfaceMaterial", "unspecified", "unspecified" },
        { 202, 0, 1, "in", "control", "scene3D", "unspecified", "unspecified" },
        { 202, 2, 1, "in", "control", "surfaceMaterial", "unspecified", "unspecified" },
        { 202, 1, 1, "out", "frame", "image", "rgba8", "linearSRGB" }
    };
    importedScenePlan.edges = {
        { 201, 4, 202, 0 }, { 203, 0, 204, 0 }, { 204, 10, 202, 2 }
    };
    importedScenePlan.operations = {
        { 201, "visual.3d.imported-animation", "source-decode", "" },
        { 203, "visual.surface.constant.vec3", "control-eval", "" },
        { 204, "visual.surface.material", "control-eval", "" },
        { 202, "visual.3d.render", "native-gpu", "" }
    };
    const bool importedSceneAdmitted = videowire::normalizeSnapshot (
               raw, std::vector<videowire::CompiledVisualLayerPlan> { importedScenePlan }, {}, 2, true, rejected, error);
    if (! importedSceneAdmitted)
        std::fprintf(stderr, "imported scene admission: %s\n", error.c_str());
    check (importedSceneAdmitted,
           "typed non-frame scene resources admit only through source decode");
    importedScenePlan.operations[0].backendCapability = "control-eval";
    check (! videowire::normalizeSnapshot (raw, std::vector<videowire::CompiledVisualLayerPlan> { importedScenePlan }, {}, 2, true, rejected, error),
           "typed scene resources reject deterministic control evaluation");
    auto deformedImportedScenePlan = importedScenePlan;
    deformedImportedScenePlan.nodeKinds = {
        "visual.3d.imported-animation", "visual.deformation.imported",
        "visual.surface.constant.vec3", "visual.surface.material", "visual.3d.render"
    };
    deformedImportedScenePlan.nodeIds = { 201, 205, 203, 204, 202 };
    deformedImportedScenePlan.ports = {
        { 201, 0, 1, "out", "control", "mesh", "unspecified", "unspecified" },
        { 201, 4, 1, "out", "control", "scene3D", "unspecified", "unspecified" },
        { 205, 0, 1, "in", "control", "mesh", "unspecified", "unspecified" },
        { 205, 4, 1, "out", "control", "mesh", "unspecified", "unspecified" },
        { 203, 0, 3, "out", "control", "vec3", "unspecified", "unspecified" },
        { 204, 0, 3, "in", "control", "vec3", "unspecified", "unspecified" },
        { 204, 10, 1, "out", "control", "surfaceMaterial", "unspecified", "unspecified" },
        { 202, 0, 1, "in", "control", "scene3D", "unspecified", "unspecified" },
        { 202, 2, 1, "in", "control", "surfaceMaterial", "unspecified", "unspecified" },
        { 202, 3, 1, "in", "control", "mesh", "unspecified", "unspecified" },
        { 202, 1, 1, "out", "frame", "image", "rgba8", "linearSRGB" }
    };
    deformedImportedScenePlan.edges = {
        { 201, 0, 205, 0 }, { 201, 4, 202, 0 }, { 205, 4, 202, 3 },
        { 203, 0, 204, 0 }, { 204, 10, 202, 2 }
    };
    deformedImportedScenePlan.operations = {
        { 201, "visual.3d.imported-animation", "source-decode", "" },
        { 205, "visual.deformation.imported", "native-gpu", "" },
        { 203, "visual.surface.constant.vec3", "control-eval", "" },
        { 204, "visual.surface.material", "control-eval", "" },
        { 202, "visual.3d.render", "native-gpu", "" }
    };
    check (videowire::normalizeSnapshot (
               raw, std::vector<videowire::CompiledVisualLayerPlan> { deformedImportedScenePlan }, {}, 2, true, rejected, error),
           "native GPU deformation admits as a non-frame control resource");
    deformedImportedScenePlan.operations[1].backendCapability = "control-eval";
    check (! videowire::normalizeSnapshot (
               raw, std::vector<videowire::CompiledVisualLayerPlan> { deformedImportedScenePlan }, {}, 2, true, rejected, error),
           "imported deformation rejects CPU control evaluation");
    auto particlePlan = typedPlan;
    particlePlan.nodeKinds = { "visual.particles", "video.out" };
    particlePlan.nodeIds = { 201, 202 };
    particlePlan.ports = {
        { 201, 0, 1, "in", "event", "event", "unspecified", "unspecified" },
        { 201, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 202, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    particlePlan.edges = { { 201, 1, 202, 0 } };
    particlePlan.operations = {
        { 201, "visual.particles", "native-gpu", "" },
        { 202, "video.out", "native-gpu", "" }
    };
    videowire::VisualEventScheduleBinding particleEvents;
    particleEvents.clipId = 44;
    particleEvents.nodeId = 201;
    particleEvents.portId = 0;
    particleEvents.sessionRevision = 9;
    particleEvents.triggers = { { 0, 4.0, 0.5f, 1 }, { 32, 4.01, 1.0f, 2 } };
    check (videowire::normalizeSnapshot (raw, std::vector<videowire::CompiledVisualLayerPlan> { particlePlan },
                                         std::vector<videowire::VisualEventScheduleBinding> { particleEvents },
                                         2, true, rejected, error)
           && rejected.visualEventSchedules.size() == 1,
           "particle Event schedules retain exact stable sink identity and order");
    auto connectedWithoutTriggers = particleEvents;
    connectedWithoutTriggers.triggers.clear();
    check (videowire::normalizeSnapshot (raw, std::vector<videowire::CompiledVisualLayerPlan> { particlePlan },
                                         std::vector<videowire::VisualEventScheduleBinding> { connectedWithoutTriggers },
                                         2, true, rejected, error)
           && rejected.visualEventSchedules.size() == 1
           && rejected.visualEventSchedules.front().triggers.empty(),
           "legacy connected-empty Event metadata parses without claiming historical coverage");
    particleEvents.portId = 1;
    check (! videowire::normalizeSnapshot (raw, std::vector<videowire::CompiledVisualLayerPlan> { particlePlan },
                                         std::vector<videowire::VisualEventScheduleBinding> { particleEvents },
                                         2, true, rejected, error)
           && error == "visual Event schedule has invalid stable sink identity",
           "particle Event schedules reject a Frame port masquerading as an Event sink");
    particleEvents.portId = 0;
    particleEvents.triggers[1].timelineBeat = 3.0;
    check (! videowire::normalizeSnapshot (raw, std::vector<videowire::CompiledVisualLayerPlan> { particlePlan },
                                         std::vector<videowire::VisualEventScheduleBinding> { particleEvents },
                                         2, true, rejected, error)
           && error == "visual Event schedule trigger ordering or bounds are invalid",
           "particle Event schedules reject non-deterministic timeline order");
    auto pointsPlan = typedPlan;
    pointsPlan.nodeKinds = { "video.source", "visual.points.grid",
                             "visual.points.set-color", "visual.clone-to-points", "video.out" };
    pointsPlan.nodeIds = { 101, 103, 104, 105, 102 };
    pointsPlan.ports.insert(pointsPlan.ports.begin() + 1, {
        { 103, 0, 1, "out", "control", "points", "unspecified", "unspecified" },
        { 104, 0, 1, "in", "control", "points", "unspecified", "unspecified" },
        { 104, 1, 4, "in", "control", "color", "unspecified", "linearSRGB" },
        { 104, 2, 1, "out", "control", "points", "unspecified", "unspecified" },
        { 105, 0, 1, "in", "control", "shape", "unspecified", "unspecified" },
        { 105, 1, 1, "in", "control", "points", "unspecified", "unspecified" },
        { 105, 2, 1, "out", "control", "shape", "unspecified", "unspecified" }
    });
    pointsPlan.edges.push_back({ 103, 0, 104, 0 });
    pointsPlan.edges.push_back({ 104, 2, 105, 1 });
    pointsPlan.operations = {
        { 101, "video.source", "source-decode", "" },
        { 103, "visual.points.grid", "control-eval", "<NodeParams columns=\"4\" rows=\"3\"/>" },
        { 104, "visual.points.set-color", "control-eval", "<NodeParams red=\"1\"/>" },
        { 105, "visual.clone-to-points", "control-eval", "" },
        { 102, "video.out", "native-gpu", "" }
    };
    const bool pointsAdmitted = videowire::normalizeSnapshot(
        raw, std::vector<videowire::CompiledVisualLayerPlan> { pointsPlan }, {}, 2, true, rejected, error);
    if (! pointsAdmitted) std::fprintf(stderr, "points admission: %s\n", error.c_str());
    check (pointsAdmitted,
           "clone-to-points admits only as typed control geometry");
    pointsPlan.operations[0].backendCapability = "control-eval";
    check (! videowire::normalizeSnapshot (raw, std::vector<videowire::CompiledVisualLayerPlan> { pointsPlan }, {}, 2, true, rejected, error),
           "frame-producing operations cannot claim control evaluation");
    typedPlan.operations[1].backendCapability = "cpu-fallback";
    check (! videowire::normalizeSnapshot (raw, std::vector<videowire::CompiledVisualLayerPlan> { typedPlan }, {}, 2, true, rejected, error)
           && error == "visual layer plan operation identity or backend admission is invalid",
           "unadmitted operation backend rejects before rendering");
    typedPlan.operations[1].backendCapability = "native-gpu";
    auto proceduralPlan = typedPlan;
    proceduralPlan.nodeKinds = {
        "video.source", "visual.noise", "visual.field.scalar", "visual.field.vector",
        "visual.field.color", "video.out"
    };
    proceduralPlan.nodeIds = { 101, 103, 104, 105, 106, 102 };
    proceduralPlan.ports = {
        { 101, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 103, 0, 2, "in", "control", "vec2", "unspecified", "unspecified" },
        { 103, 1, 1, "out", "control", "scalar", "unspecified", "unspecified" },
        { 104, 0, 1, "in", "control", "scalar", "unspecified", "unspecified" },
        { 104, 1, 1, "out", "control", "scalarField", "unspecified", "unspecified" },
        { 105, 0, 2, "in", "control", "vec2", "unspecified", "unspecified" },
        { 105, 1, 2, "out", "control", "vectorField", "unspecified", "unspecified" },
        { 106, 0, 4, "in", "control", "color", "unspecified", "linearSRGB" },
        { 106, 1, 4, "out", "control", "colorField", "unspecified", "linearSRGB" },
        { 102, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    proceduralPlan.edges = { { 101, 0, 102, 0 }, { 103, 1, 104, 0 } };
    proceduralPlan.operations = {
        { 101, "video.source", "source-decode", "" },
        { 103, "visual.noise", "control-eval", "<NodeParams seed='17' frequency='2'/>" },
        { 104, "visual.field.scalar", "control-eval", "" },
        { 105, "visual.field.vector", "control-eval", "" },
        { 106, "visual.field.color", "control-eval", "" },
        { 102, "video.out", "native-gpu", "" }
    };
    check (videowire::normalizeSnapshot (raw, std::vector<videowire::CompiledVisualLayerPlan> { proceduralPlan }, {}, 2, true, rejected, error),
           "procedural noise and scalar/vector/color fields admit as control evaluation");
    proceduralPlan.operations[2].backendCapability = "native-gpu";
    check (! videowire::normalizeSnapshot (raw, std::vector<videowire::CompiledVisualLayerPlan> { proceduralPlan }, {}, 2, true, rejected, error)
           && error == "visual layer plan operation identity or backend admission is invalid",
           "field operation cannot claim native renderer execution");

    auto valueMathPlan = typedPlan;
    valueMathPlan.nodeKinds = {
        "video.source", "visual.vec2.add", "visual.vec2.multiply", "visual.vec2.remap",
        "visual.color.mix", "visual.color.remap", "video.out"
    };
    valueMathPlan.nodeIds = { 101, 103, 104, 105, 106, 107, 102 };
    valueMathPlan.ports = {
        { 101, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 103, 0, 2, "in", "control", "vec2", "unspecified", "unspecified" },
        { 103, 1, 2, "in", "control", "vec2", "unspecified", "unspecified" },
        { 103, 2, 2, "out", "control", "vec2", "unspecified", "unspecified" },
        { 104, 0, 2, "in", "control", "vec2", "unspecified", "unspecified" },
        { 104, 1, 2, "in", "control", "vec2", "unspecified", "unspecified" },
        { 104, 2, 2, "out", "control", "vec2", "unspecified", "unspecified" },
        { 105, 0, 2, "in", "control", "vec2", "unspecified", "unspecified" },
        { 105, 5, 2, "out", "control", "vec2", "unspecified", "unspecified" },
        { 106, 0, 4, "in", "control", "color", "unspecified", "linearSRGB" },
        { 106, 3, 4, "out", "control", "color", "unspecified", "linearSRGB" },
        { 107, 0, 4, "in", "control", "color", "unspecified", "linearSRGB" },
        { 107, 5, 4, "out", "control", "color", "unspecified", "linearSRGB" },
        { 102, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    valueMathPlan.edges = { { 101, 0, 102, 0 }, { 103, 2, 104, 0 }, { 104, 2, 105, 0 } };
    valueMathPlan.operations = {
        { 101, "video.source", "source-decode", "" },
        { 103, "visual.vec2.add", "control-eval", "" },
        { 104, "visual.vec2.multiply", "control-eval", "" },
        { 105, "visual.vec2.remap", "control-eval", "" },
        { 106, "visual.color.mix", "control-eval", "" },
        { 107, "visual.color.remap", "control-eval", "" },
        { 102, "video.out", "native-gpu", "" }
    };
    check (videowire::normalizeSnapshot (raw, std::vector<videowire::CompiledVisualLayerPlan> { valueMathPlan }, {}, 2, true, rejected, error),
           "typed vector and color math admit only as deterministic control evaluation");
    valueMathPlan.operations[4].backendCapability = "native-gpu";
    check (! videowire::normalizeSnapshot (raw, std::vector<videowire::CompiledVisualLayerPlan> { valueMathPlan }, {}, 2, true, rejected, error)
           && error == "visual layer plan operation identity or backend admission is invalid",
           "color math cannot claim native image execution");

    auto shapeUvPlan = typedPlan;
    shapeUvPlan.nodeKinds = {
        "video.source", "visual.shape.rectangle", "visual.shape.ellipse",
        "visual.shape.union", "visual.uv.transform", "visual.uv.remap", "video.out"
    };
    shapeUvPlan.nodeIds = { 101, 103, 104, 105, 106, 107, 102 };
    shapeUvPlan.ports = {
        { 101, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 103, 0, 1, "out", "control", "shape", "unspecified", "unspecified" },
        { 104, 0, 1, "out", "control", "shape", "unspecified", "unspecified" },
        { 105, 0, 1, "in", "control", "shape", "unspecified", "unspecified" },
        { 105, 1, 1, "in", "control", "shape", "unspecified", "unspecified" },
        { 105, 2, 1, "out", "control", "shape", "unspecified", "unspecified" },
        { 106, 0, 2, "in", "control", "vec2", "unspecified", "unspecified" },
        { 106, 4, 2, "out", "control", "vec2", "unspecified", "unspecified" },
        { 107, 0, 2, "in", "control", "vec2", "unspecified", "unspecified" },
        { 107, 5, 2, "out", "control", "vec2", "unspecified", "unspecified" },
        { 102, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    shapeUvPlan.edges = { { 101, 0, 102, 0 }, { 103, 0, 105, 0 }, { 104, 0, 105, 1 },
                          { 106, 4, 107, 0 } };
    shapeUvPlan.operations = {
        { 101, "video.source", "source-decode", "" },
        { 103, "visual.shape.rectangle", "control-eval", "<NodeParams width='1'/>" },
        { 104, "visual.shape.ellipse", "control-eval", "<NodeParams width='0.5'/>" },
        { 105, "visual.shape.union", "control-eval", "" },
        { 106, "visual.uv.transform", "control-eval", "<NodeParams pivotX='0.5'/>" },
        { 107, "visual.uv.remap", "control-eval", "" },
        { 102, "video.out", "native-gpu", "" }
    };
    check (videowire::normalizeSnapshot (raw, std::vector<videowire::CompiledVisualLayerPlan> { shapeUvPlan }, {}, 2, true, rejected, error),
           "shape geometry and UV operations admit as typed control evaluation");
    shapeUvPlan.operations[3].backendCapability = "native-gpu";
    check (! videowire::normalizeSnapshot (raw, std::vector<videowire::CompiledVisualLayerPlan> { shapeUvPlan }, {}, 2, true, rejected, error)
           && error == "visual layer plan operation identity or backend admission is invalid",
           "shape geometry cannot claim native image execution");

    typedPlan.nodeKinds[1] = "video.future";
    typedPlan.operations[1].kind = "video.future";
    check (! videowire::normalizeSnapshot (raw, std::vector<videowire::CompiledVisualLayerPlan> { typedPlan }, {}, 2, true, rejected, error)
           && error == "visual layer plan contains unsupported typed operation: video.future",
           "unsupported typed operation rejects without a fixed-chain comparison");
    typedPlan.nodeKinds[1] = "video.out";
    typedPlan.operations[1].kind = "video.out";
    typedPlan.ports[1].dataType = "mask";
    check (! videowire::normalizeSnapshot (raw, std::vector<videowire::CompiledVisualLayerPlan> { typedPlan }, {}, 2, true, rejected, error)
           && error == "visual layer plan has an incompatible typed edge binding",
           "incompatible typed edge bindings reject before rendering");
    auto editablePlan = compiledPlan;
    editablePlan.nodeKinds = { "video.source", "video.transform", "video.out" };
    check (videowire::normalizeSnapshot (raw, std::vector<videowire::CompiledVisualLayerPlan> { editablePlan }, {}, 2, true, rejected, error)
           && rejected.visualLayerPlans.size() == 1
           && rejected.visualLayerPlans[0].nodeKinds == editablePlan.nodeKinds,
           "editable transform plan lowers onto existing production operations");
    editablePlan.nodeKinds = {
        "video.source", "video.transform", "video.effects", "video.out"
    };
    check (videowire::normalizeSnapshot (raw, std::vector<videowire::CompiledVisualLayerPlan> { editablePlan }, {}, 3, true, rejected, error),
           "editable effects plan lowers onto the production effect rack");
    editablePlan.nodeKinds = {
        "video.source", "video.transform", "video.effects", "video.mask.shape", "video.out"
    };
    check (videowire::normalizeSnapshot (raw, std::vector<videowire::CompiledVisualLayerPlan> { editablePlan }, {}, 4, true, rejected, error),
           "editable shape-mask plan lowers onto production mask operations");
    editablePlan.nodeKinds = {
        "video.source", "video.transform", "video.effects", "video.mask.shape",
        "video.blend", "video.out"
    };
    check (videowire::normalizeSnapshot (raw, std::vector<videowire::CompiledVisualLayerPlan> { editablePlan }, {}, 5, true, rejected, error),
           "editable blend plan lowers onto production layer compositing");
    editablePlan.nodeKinds = {
        "video.source", "video.transform", "video.effects", "video.mask.shape",
        "video.text", "video.blend", "video.out"
    };
    check (videowire::normalizeSnapshot (raw, std::vector<videowire::CompiledVisualLayerPlan> { editablePlan }, {}, 6, true, rejected, error),
           "editable text plan lowers onto production text compositing");
    editablePlan.nodeKinds = {
        "video.source", "video.transform", "video.effects", "video.mask.shape",
        "video.text", "video.layer.source", "video.blend", "video.out"
    };
    check (videowire::normalizeSnapshot (raw, std::vector<videowire::CompiledVisualLayerPlan> { editablePlan }, {}, 7, true, rejected, error),
           "editable second-source plan lowers onto production layer compositing");
    editablePlan.nodeKinds = { "video.source", "video.out" };
    check (videowire::normalizeSnapshot (raw, std::vector<videowire::CompiledVisualLayerPlan> { editablePlan }, {}, 8, true, rejected, error),
           "delete-healed direct plan lowers onto existing production operations");
    compiledPlan.producerValidated = false;
    compiledPlan.error = "malformed fixed topology";
    check (! videowire::normalizeSnapshot (raw, std::vector<videowire::CompiledVisualLayerPlan> { compiledPlan }, {}, 2, true, rejected, error)
           && error == "malformed fixed topology",
           "malformed visual layer plan rejects the current structural candidate");
    videowire::RevisionLedger planRevisions;
    check (planRevisions.requestAuthoring (1) && planRevisions.accept (1)
           && planRevisions.compileSucceeded (1, true)
           && planRevisions.requestAuthoring (2) && planRevisions.reject (2)
           && planRevisions.state().rejected == 2
           && planRevisions.state().lastGood == 1
           && planRevisions.state().exportable == 0,
           "fixed-topology rejection preserves last-good preview and clears exportability");
    check (videowire::normalizeSnapshot (raw, 1, rejected, error)
           && raw[0].sourcePath == originalPath
           && raw[0].sourceKind == videowire::SourceKind::Unspecified,
           "snapshot construction does not mutate canonical input");

    videowire::RevisionLedger revisions;
    check (revisions.requestAuthoring (1) && revisions.accept (1)
           && revisions.compileSucceeded (1, true),
           "accepted authoring becomes compiled last-good and exportable");
    const auto structuralBeforeRuntime = revisions.state();
    check (revisions.evaluate (1, 100) && revisions.evaluate (1, 101),
           "runtime evaluations advance monotonically");
    check (revisions.state().authoring == structuralBeforeRuntime.authoring
           && revisions.state().compiled == structuralBeforeRuntime.compiled
           && revisions.state().evaluation == 101,
           "runtime evaluation does not bump structural revisions");

    check (revisions.requestAuthoring (2), "new structural authoring advances");
    check (revisions.state().compiled == 1 && revisions.state().lastGood == 1
           && revisions.state().exportable == 0,
           "uncompiled authoring keeps last-good preview but clears exportability");
    check (! revisions.accept (1) && ! revisions.compileSucceeded (1, true),
           "stale acknowledgement cannot become current");
    check (revisions.reject (2)
           && revisions.state().rejected == 2
           && revisions.state().compiled == 1
           && revisions.state().lastGood == 1
           && revisions.state().exportable == 0,
           "rejection preserves last-good and refuses current export");

    videowire::RevisionLedger replay;
    replay.requestAuthoring (1);
    replay.accept (1);
    replay.compileSucceeded (1, true);
    check (replay.state().accepted == 1 && replay.state().compiled == 1
           && replay.state().lastGood == 1 && replay.state().exportable == 1,
           "restart replay recreates the accepted revision tuple");

    videowire::RevisionLedger race;
    check (race.requestAuthoring (10) && race.requestAuthoring (11)
           && race.requestAuthoring (12),
           "compile race publishes revisions 10, 11, and 12");
    check (race.accept (11) && race.compileSucceeded (11, true)
           && race.state().compiled == 11 && race.state().lastGood == 11
           && race.state().exportable == 0,
           "newest completed stale revision becomes last-good but not exportable");
    check (! race.accept (10) && ! race.compileSucceeded (10, true)
           && race.state().compiled == 11,
           "older completion cannot roll compiled state backward");
    check (race.accept (12) && race.compileSucceeded (12, true)
           && race.state().accepted == 12 && race.state().compiled == 12
           && race.state().lastGood == 12 && race.state().exportable == 12,
           "current completion becomes compiled and exportable");

    videowire::RevisionLedger rejectedRace;
    check (rejectedRace.requestAuthoring (10) && rejectedRace.requestAuthoring (11)
           && rejectedRace.requestAuthoring (12),
           "rejected race publishes revisions 10, 11, and 12");
    check (rejectedRace.accept (11) && rejectedRace.compileSucceeded (11, true)
           && ! rejectedRace.accept (10) && ! rejectedRace.compileSucceeded (10, true),
           "rejected race retains the newest usable out-of-order completion");
    check (rejectedRace.reject (12)
           && rejectedRace.state().authoring == 12
           && rejectedRace.state().accepted == 11
           && rejectedRace.state().rejected == 12
           && rejectedRace.state().compiled == 11
           && rejectedRace.state().lastGood == 11
           && rejectedRace.state().exportable == 0,
           "current rejection preserves revision 11 only as last-good");

    // Exact lazy-open boundary: custom naming is retained, and post-admission
    // frame/manifest mutation is rejected by the same validator used by viewport/export.
    const auto cacheRoot = std::filesystem::canonical(std::filesystem::temp_directory_path())
        / ("arbit-matte-cache-test-" + std::to_string(reinterpret_cast<uintptr_t>(&checks)));
    videohelper::MatteCacheBinding cacheBinding;
    cacheBinding.version = "content-v1";
    cacheBinding.prefix = "subject-";
    cacheBinding.extension = ".rgba";
    cacheBinding.firstFrame = 42;
    cacheBinding.digits = 3;
    cacheBinding.frames = 1;
    cacheBinding.fps = 24.0;
    cacheBinding.contentRevision = 99;
    const std::string frameBytes = "immutable-frame-bytes";
    const std::vector<std::string> frameHashes { videohelper::sha256Text(frameBytes) };
    const auto manifest = videohelper::bindingMaterial(cacheBinding, frameHashes);
    cacheBinding.key = cacheBinding.receipt = videohelper::sha256Text(manifest);
    const auto cacheDir = cacheRoot / cacheBinding.key;
    std::filesystem::create_directories(cacheDir);
    const auto framePath = cacheDir / videohelper::matteFrameName(cacheBinding, 0);
    { std::ofstream frame(framePath, std::ios::binary); frame << frameBytes; }
    { std::ofstream manifestFile(cacheDir / "manifest", std::ios::binary); manifestFile << manifest; }
    std::filesystem::permissions(framePath, std::filesystem::perms::owner_write, std::filesystem::perm_options::remove);
    std::filesystem::permissions(cacheDir / "manifest", std::filesystem::perms::owner_write, std::filesystem::perm_options::remove);
    std::filesystem::permissions(cacheDir, std::filesystem::perms::owner_write, std::filesystem::perm_options::remove);
    videowire::RenderSegment lazySegment;
    lazySegment.matteAssetId = "matte-custom";
    lazySegment.matteAssetVersion = cacheBinding.version;
    lazySegment.matteContentReceipt = cacheBinding.receipt;
    lazySegment.matteCacheKey = cacheBinding.key;
    lazySegment.matteFramePrefix = cacheBinding.prefix;
    lazySegment.matteFrameExtension = cacheBinding.extension;
    lazySegment.matteFirstFrame = cacheBinding.firstFrame;
    lazySegment.matteFrameDigits = cacheBinding.digits;
    lazySegment.matteFrames = cacheBinding.frames;
    lazySegment.matteFps = cacheBinding.fps;
    lazySegment.matteContentRevision = cacheBinding.contentRevision;
    lazySegment.matteTrustedRoot = cacheRoot.string();
    std::string resolvedDir, customPattern;
    check (videowire::revalidateMatteForOpen(lazySegment, resolvedDir, customPattern, error)
           && customPattern == cacheDir.string() + "/subject-%03d.rgba",
           "viewport/export lazy-open validator derives the complete custom naming tuple");
    const auto originalSource = cacheRoot / "project-controlled-original.mov";
    { std::ofstream source(originalSource, std::ios::binary); source << "original-source-v1"; }
    { std::ofstream source(originalSource, std::ios::binary | std::ios::trunc); source << "mutated-source-v2"; }
    check (videowire::revalidateMatteForOpen(lazySegment, resolvedDir, customPattern, error),
           "original-source mutation cannot alter receipt-addressed cached decode authority");
    std::filesystem::permissions(framePath, std::filesystem::perms::owner_write, std::filesystem::perm_options::add);
    { std::ofstream frame(framePath, std::ios::binary | std::ios::trunc); frame << "tampered"; }
    check (! videowire::revalidateMatteForOpen(lazySegment, resolvedDir, customPattern, error),
           "frame mutation after admission is rejected immediately before lazy open");
    { std::ofstream frame(framePath, std::ios::binary | std::ios::trunc); frame << frameBytes; }
    std::filesystem::permissions(framePath, std::filesystem::perms::owner_write, std::filesystem::perm_options::remove);
    std::filesystem::permissions(cacheDir / "manifest", std::filesystem::perms::owner_write, std::filesystem::perm_options::add);
    { std::ofstream manifestFile(cacheDir / "manifest", std::ios::binary | std::ios::app); manifestFile << "tamper"; }
    check (! videowire::revalidateMatteForOpen(lazySegment, resolvedDir, customPattern, error),
           "manifest mutation after admission is rejected immediately before lazy open");
    std::filesystem::permissions(cacheDir / "manifest", std::filesystem::perms::owner_write,
                                 std::filesystem::perm_options::add);
    std::filesystem::permissions(framePath, std::filesystem::perms::owner_write,
                                 std::filesystem::perm_options::add);
    std::filesystem::permissions(cacheDir, std::filesystem::perms::owner_write,
                                 std::filesystem::perm_options::add);
    std::filesystem::remove_all(cacheRoot);


    // Plan 3 Event coverage admission is independent of particle simulation.
    {
        videowire::VisualEventScheduleBinding covered;
        covered.clipId = 44; covered.nodeId = 201; covered.portId = 0; covered.sessionRevision = 9;
        covered.coverageVersion = 1; covered.coverageComplete = true;
        covered.planGeneration = 3; covered.transportEpoch = 2; covered.streamId = 1;
        covered.routeId = 10; covered.sourceNodeId = 100; covered.sourcePortId = 0;
        covered.sampleRate = 1024; covered.bpm = 120;
        covered.coverageStartSample = 0; covered.coverageEndSample = 1024;
        covered.coverageStartBeat = 0; covered.coverageEndBeat = 2;
        const auto admit = [&] (std::vector<videowire::VisualEventScheduleBinding> schedules)
        {
            return videowire::normalizeSnapshot(raw, std::vector<videowire::CompiledVisualLayerPlan>{particlePlan},
                std::move(schedules),2,true,rejected,error);
        };
        check (admit({covered}) && rejected.visualEventSchedules[0].coverageComplete,
               "event-coverage coherent silence has an explicit historical interval");
        auto second = covered; second.routeId = 11; second.sourceNodeId = 101;
        check (admit({covered,second}) && rejected.visualEventSchedules.size() == 2,
               "event-coverage distinct source routes retain the same sink without coalescing");
        second.routeId = 10;
        check (!admit({covered,second}), "event-coverage duplicate route declaration is refused");
        for (int index = 0; index < 512; ++index)
            covered.triggers.push_back({index,static_cast<double>(index)/512,0.5f,static_cast<uint64_t>(index),index});
        check (admit({covered}), "event-coverage exactly 512 ordered triggers are admitted");
        covered.triggers.push_back({512,1.0,0.5f,512,512});
        check (!admit({covered}), "event-coverage 513 triggers refuse instead of truncating");
        covered.triggers.clear(); covered.coverageEndSample = 10241;
        check (!admit({covered}), "event-coverage historical interval cannot exceed ten seconds");
        covered.coverageEndSample = 1024; covered.coverageVersion = 2;
        check (!admit({covered}), "event-coverage unsupported metadata version is refused");
        covered.coverageVersion = 1; covered.coverageComplete = false; covered.lossFlags = 8;
        check (admit({covered}) && !rejected.visualEventSchedules[0].coverageComplete,
               "event-coverage external MIDI uncertainty remains explicit incomplete metadata");

        // Malformed historical fields must be rejected before numeric narrowing or sink admission.
        auto malformed = wire;
        const json historical = {{"clipId",44},{"nodeId",201},{"portId",0},{"sessionRevision",9},
            {"coverageVersion",1},{"coverageComplete",true},{"planGeneration",3},{"transportEpoch",2},
            {"streamId",1},{"routeId",10},{"sourceNodeId",100},{"sourcePortId",0},{"lossFlags",0},
            {"sampleRate",1024},{"bpm",120},{"coverageStartSample",0},{"coverageEndSample",1024},
            {"coverageStartBeat",0},{"coverageEndBeat",2},{"triggers",json::array()}};
        for (const auto* field : {"planGeneration","sourceNodeId","coverageEndSample","coverageComplete"})
        {
            auto row = historical; row.erase(field); malformed["visualEventSchedules"] = json::array({row});
            check (!videowire::parseSnapshotJson(malformed,noShader,rejected,error)
                   && error.find("coverage metadata") != std::string::npos,
                   "event-coverage missing historical metadata refuses before normalization");
        }
        auto row = historical; row["coverageVersion"] = uint64_t{4294967297ULL};
        malformed["visualEventSchedules"] = json::array({row});
        check (!videowire::parseSnapshotJson(malformed,noShader,rejected,error)
               && error.find("version") != std::string::npos,
               "event-coverage wide version cannot narrow into a supported identity");
        row = historical; row["sourceNodeId"] = uint64_t{4294967296ULL};
        malformed["visualEventSchedules"] = json::array({row});
        check (!videowire::parseSnapshotJson(malformed,noShader,rejected,error)
               && error.find("exact representation") != std::string::npos,
               "event-coverage source identity refuses numeric truncation");
    }

    std::printf ("render-snapshot: %d/%d checks passed\n", checks - failures, checks);
    return failures;
}
