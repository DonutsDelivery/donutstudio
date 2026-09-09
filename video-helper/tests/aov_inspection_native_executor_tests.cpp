#include "../src/visual_plan_executor.h"
#include "support/fixture_scene.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

namespace
{
[[noreturn]] void fail (const char* message)
{
    std::cerr << "FAIL: " << message << '\n';
    std::exit (1);
}

void require (bool condition, const char* message)
{
    if (! condition)
        fail (message);
}

videowire::CompiledVisualLayerPlan depthInspectionPlan()
{
    auto fixture = videohelper::fixture3d::makeScene();
    fixture.textures[0].magFilter = 9728;
    fixture.textures[0].minFilter = 9729;
    fixture.textures[0].wrapS = 33071;
    fixture.textures[0].wrapT = 33648;
    fixture.materials[0].metallicRoughnessTexture = fixture.textures[0].id;
    fixture.materials[0].normalTexture = fixture.textures[0].id;
    fixture.materials[0].occlusionTexture = fixture.textures[0].id;
    fixture.materials[0].emissiveTexture = fixture.textures[0].id;
    fixture.materials[0].normalScale = 0.75f;
    fixture.materials[0].occlusionStrength = 0.5f;
    fixture.materials[0].alphaMode = HarmonicMIDI::grid::SceneAlphaMode::Mask;
    fixture.materials[0].alphaCutoff = 0.25f;
    fixture.materials[0].doubleSided = true;
    fixture.lights[0].kind = HarmonicMIDI::grid::SceneLightKind::Spot;
    fixture.lights[0].innerConeAngle = 0.1f;
    fixture.lights[0].outerConeAngle = 0.2f;
    sceneaov::Payload scenePayload;
    scenePayload.version = sceneaov::kWireVersion;
    scenePayload.output = renderpassoutput::Output::Depth;
    scenePayload.extent = { 640, 360 };
    scenePayload.scene = std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (fixture);

    aovinspection::Payload inspection;
    inspection.schemaVersion = aovinspection::kSchemaVersion;
    inspection.source = aovinspection::Source::Depth;
    inspection.extent = scenePayload.extent;
    inspection.depthNear = 0.1f;
    inspection.depthFar = 1.0f;

    videowire::CompiledVisualLayerPlan plan;
    plan.clipId = 41;
    plan.structuralRevision = 9;
    plan.producerValidated = true;
    plan.descriptorCount = 2;
    plan.operationCount = 2;
    plan.frameOutputCount = 1;
    plan.peakLiveFrameCount = 1;
    plan.allocatedFrameSlotCount = 1;
    plan.nodeKinds = { std::string (sceneaov::operationKind (renderpassoutput::Output::Depth)),
                       std::string (aovinspection::kDepthOperationKind) };
    plan.nodeIds = { 100, 200 };
    plan.ports = {
        { 100, 2, 1, "out", "frame", "depth", "r32f", "data" },
        { 200, 0, 1, "in", "frame", "depth", "r32f", "data" },
        { 200, 1, 1, "out", "frame", "image", "rgba16f", "linearSRGB" }
    };
    plan.edges = { { 100, 2, 200, 0 } };
    plan.operations = {
        { 100, std::string (sceneaov::operationKind (renderpassoutput::Output::Depth)),
          std::string (sceneaov::kBackendCapability), sceneaov::serialize (scenePayload) },
        { 200, std::string (aovinspection::kDepthOperationKind),
          std::string (aovinspection::kBackendCapability),
          aovinspection::serialize (inspection) }
    };
    return plan;
}
} // namespace

int main()
{
    auto plan = depthInspectionPlan();
    videowire::VisualLayerExecution execution;
    std::string error;
    const auto compiled = videowire::compileVisualLayerExecution (plan, execution, error);
    if (! compiled)
    {
        sceneaov::Payload parsedScene;
        aovinspection::Payload parsedInspection;
        const auto sceneParsed = sceneaov::parse (plan.operations[0].payloadXml, parsedScene);
        const auto inspectionParsed = aovinspection::parse (
            plan.operations[1].payloadXml, parsedInspection);
        std::cerr << "compile diagnostic: " << error
                  << "; sceneParsed=" << sceneParsed
                  << "; sceneCanonical=" << (sceneParsed
                      && sceneaov::serialize (parsedScene) == plan.operations[0].payloadXml)
                  << "; inspectionParsed=" << inspectionParsed
                  << "; inspectionCanonical=" << (inspectionParsed
                      && aovinspection::serialize (parsedInspection) == plan.operations[1].payloadXml)
                  << "; nodes=" << plan.nodeKinds.size()
                  << "; ports=" << plan.ports.size()
                  << "; edges=" << plan.edges.size() << '\n';
    }
    require (compiled
             && error.empty() && execution.sceneAovPass.has_value()
             && execution.aovInspectionPass.has_value()
             && execution.sceneAovPass->output == renderpassoutput::Output::Depth
             && execution.sceneAovPass->scene->textures[0].magFilter == 9728
             && execution.sceneAovPass->scene->textures[0].minFilter == 9729
             && execution.sceneAovPass->scene->textures[0].wrapS == 33071
             && execution.sceneAovPass->scene->textures[0].wrapT == 33648
             && execution.sceneAovPass->scene->materials[0].normalTexture
                    == execution.sceneAovPass->scene->textures[0].id
             && execution.sceneAovPass->scene->materials[0].alphaMode
                    == HarmonicMIDI::grid::SceneAlphaMode::Mask
             && execution.sceneAovPass->scene->materials[0].doubleSided
             && execution.sceneAovPass->scene->lights[0].kind
                    == HarmonicMIDI::grid::SceneLightKind::Spot
             && execution.sceneAovPass->scene->lights[0].innerConeAngle == 0.1f
             && execution.sceneAovPass->scene->lights[0].outerConeAngle == 0.2f
             && execution.aovInspectionPass->source == aovinspection::Source::Depth,
             "the exact scene-AOV-to-inspector schedule must compile for native execution");

    auto disconnected = plan;
    disconnected.edges.clear();
    require (! videowire::compileVisualLayerExecution (disconnected, execution, error)
             && error == "AOV inspection requires one exact scene-AOV-to-inspector native schedule",
             "a disconnected inspector must fail closed");

    auto substituted = plan;
    sceneaov::Payload normalScene;
    require (sceneaov::parse (substituted.operations.front().payloadXml, normalScene),
             "scene AOV fixture payload must parse");
    normalScene.output = renderpassoutput::Output::Normal;
    substituted.operations.front().kind = std::string (
        sceneaov::operationKind (renderpassoutput::Output::Normal));
    substituted.operations.front().payloadXml = sceneaov::serialize (normalScene);
    substituted.nodeKinds.front() = substituted.operations.front().kind;
    require (! videowire::compileVisualLayerExecution (substituted, execution, error)
             && error == "AOV inspection requires one exact scene-AOV-to-inspector native schedule",
             "an inspector must reject a source-kind substitution");

    auto forged = plan;
    forged.operations.back().payloadXml.insert (
        forged.operations.back().payloadXml.size() - 2, " runtimeGrant=\"forged\"");
    require (! videowire::compileVisualLayerExecution (forged, execution, error)
             && error == "AOV inspection requires one exact scene-AOV-to-inspector native schedule",
             "an inspector must reject forged payload authority");

    std::cout << "native AOV inspection executor: PASS\n";
    return 0;
}
