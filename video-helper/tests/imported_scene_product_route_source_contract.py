#!/usr/bin/env python3
from pathlib import Path
import sys


root = Path(sys.argv[1])
viewport = (root / "src/viewport.cpp").read_text()
exporter = (root / "src/exporter.cpp").read_text()
execution = (root / "src/imported_scene_visual_plan_execution.h").read_text()
payload_execution = (root / "src/imported_scene_payload_execution.cpp").read_text()
helper_main = (root / "src/main.cpp").read_text()
wire_contract = (root / "../shared/VisualImportedSceneRenderOperationContract.h").read_text()

# Ordinary imported scenes and imported particle overlays use the same
# exact-payload/native-frame seam at their own preview and export times.
prepare_call = "prepareVisualImportedSceneLayerAtTime("
assert viewport.count(prepare_call) == 2
assert exporter.count(prepare_call) == 2
viewport_scene = viewport.partition("const auto importedSceneResult =")[2].partition("const auto importedResult =")[0]
viewport_particles = viewport.partition("if (videowire::isImportedParticleOverlayPlan(frameVisualPlans->plans, al.seg.clipId))")[2].partition("else if (isScore)")[0]
export_scene = exporter.partition("const auto importedSceneResult =")[2].partition("const auto importedResult =")[0]
export_particles = exporter.partition("if (videowire::isImportedParticleOverlayPlan(visualLayerPlans, al.seg->clipId))")[2].partition("else if (sourceKind == videowire::SourceKind::Score)")[0]
assert all(route.count(prepare_call) == 1 for route in
           (viewport_scene, viewport_particles, export_scene, export_particles))
assert "sourceSec, valueFps," in viewport_scene
assert "NativeImportedSceneRenderUse::Preview" in viewport_scene
assert "&materialFrameResolver" in viewport_scene
assert "VisualImportedScenePreparation::notApplicable" in viewport_scene
assert "sceneSeconds, valueFps," in viewport_particles
assert "sceneSeconds = al.seg.inSec" in viewport_particles
assert "(displaySec - al.seg.displayStartSec) * al.seg.rate" in viewport_particles
assert "NativeImportedSceneRenderUse::Preview" in viewport_particles
assert "&al.params.visualParams" in viewport_particles
assert "prepared != videohelper::importedscene::VisualImportedScenePreparation::rendered" in viewport_particles
assert "im.rendererError = sceneError" in viewport_particles
assert "sourceSec, job.fps," in export_scene
assert "NativeImportedSceneRenderUse::Export" in export_scene
assert "&materialFrameResolver" in export_scene
assert "glctx.hdrProfile != nullptr" in export_scene
assert "sceneSeconds, job.fps," in export_particles
assert "sceneSeconds = al.seg->inSec" in export_particles
assert "(t - al.seg->displayStartSec) * al.seg->rate" in export_particles
assert "NativeImportedSceneRenderUse::Export" in export_particles
assert "&al.params.visualParams" in export_particles
assert "VisualImportedScenePreparation::rendered)" in export_particles
assert "glctx.hdrProfile != nullptr" in export_particles
assert all("importedSceneFrameOwners" in route for route in
           (viewport_scene, viewport_particles, export_scene, export_particles))
assert all("importedScenePlanCache" in route for route in
           (viewport_scene, viewport_particles, export_scene, export_particles))
assert "ImportedScenePayloadExecution" in viewport
assert "ImportedScenePayloadExecution" in exporter
assert "nativeFixtureSceneBackend()" in viewport
assert "nativeFixtureSceneBackend()" in exporter
assert "deformationRequest.material = request.material;" in payload_execution
assert "deformation with Surface Material is not admitted" not in payload_execution

# Current diffraction has one production wire version. V6 remains reachable only
# through its named legacy encoder and guarded migration callback. V8 owns scene
# composition and explicit light transport.
legacy_encoder = wire_contract.index("inline std::string encodeLegacyV6")
v7_producer = "encodeBase(encoded, kCompiledOperationHeaderV7, request)"
v6_producer = "encodeBase(encoded, kCompiledOperationHeaderV6, request)"
assert wire_contract.count(v7_producer) == 1
assert wire_contract.index(v7_producer) < legacy_encoder
assert wire_contract.count(v6_producer) == 1
assert wire_contract.index(v6_producer) > legacy_encoder
assert "encodeBase(encoded, kCompiledOperationHeaderV8, request)" in wire_contract
assert "encoded << kCompiledOperationHeaderV8 << '\\n'" in wire_contract
assert "encodeLegacyV6(request) != encoded || migrateLegacyV6 == nullptr" in wire_contract

# The ordinary compositor receives only the backend-owned native view. The
# complete payload/render receipt remains retained until the layer is consumed.
assert "Execution::nativeFrame(receipt)" in execution
assert "frameOwners.push_back(std::move(receipt));" in execution
assert 'backend != "opengl" && backend != "metal"' in execution
assert "layer.nativeTextureBackend = backend;" in execution
assert "layer.nativeTextureView = nativeView;" in execution
assert 'if (backend == "opengl")' in execution
assert "std::numeric_limits<unsigned>::max()" in execution
assert "openGlTexture = static_cast<unsigned>(nativeView);" in execution
assert "layer.texture = openGlTexture;" in execution

# Both bounded compositor entry points retain the processor-owned model bytes.
assert "renderCompositeFrame(job, 0.0, frame, &g_modelPayloads)" in helper_main
assert "renderCompositeFrame (job, timelineSec, frame, &g_modelPayloads)" in helper_main

# No CPU image or readback fallback is part of the production route.
assert "readback" not in execution.lower()
assert "cpu" not in execution.lower()
