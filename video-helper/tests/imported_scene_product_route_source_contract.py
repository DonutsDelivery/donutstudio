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
# A scene-prep failure holds the frame and keeps the channel open. Closing it here
# ended the Metal stream for a scene edit.
assert "holdFrameForPlanError(sceneError);" in viewport_particles and "im.wantClose" not in viewport_particles
# A layer plan that fails to execute holds the frame too. Imported scenes take this path.
general_plan = viewport.partition("if (! planExecuted")[2].partition("if (d.shaderOperationPlan != nullptr")[0]
assert "holdFrameForPlanError(planError);" in general_plan and "im.wantClose" not in general_plan
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
# composition and explicit light transport; spectral materials select V16.
legacy_encoder = wire_contract.index("inline std::string encodeLegacyV6")
v7_producer = "encodeBase(encoded, kCompiledOperationHeaderV7, request)"
v6_producer = "encodeBase(encoded, kCompiledOperationHeaderV6, request)"
assert wire_contract.count(v7_producer) == 1
assert wire_contract.index(v7_producer) < legacy_encoder
assert wire_contract.count(v6_producer) == 1
assert wire_contract.index(v6_producer) > legacy_encoder
wire_tokens = " ".join(wire_contract.split())
spectral_selector = "request.material && request.material->program.version == surfacematerial::kSpectralWireVersion"
assert ("detail::encodeBase(encoded, " + spectral_selector
        + " ? kCompiledOperationHeaderV16 : kCompiledOperationHeaderV8, request);") in wire_tokens
assert ("encoded << (" + spectral_selector
        + " ? kCompiledOperationHeaderV16 : kCompiledOperationHeaderV8) << '\\n'") in wire_tokens
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
assert "renderCompositeFrame(job, job.startSec, frame, &g_modelPayloads)" in helper_main
assert "renderCompositeFrame (job, timelineSec, frame, &g_modelPayloads)" in helper_main

# No CPU image or readback fallback is part of the production route.
assert "readback" not in execution.lower()
capacity_diagnostic = ('return reject("imported parent publication identity exceeds permanent work '
                       'or logical CPU capacity");')
assert execution.count(capacity_diagnostic) == 1
assert "cpu" not in execution.replace(capacity_diagnostic, "").lower()

# V19 retains the selected source/glass/policy in the ordinary request. Native
# dispatch owns support admission; a selection cannot become an ordinary draw.
assert "request.glassCaustic = compiled->importedSceneRender->glassCaustic;" in execution
assert "Glass caustics require a compositor layer with an owned linear native texture" in execution
assert "operation.glassCaustic = request.glassCaustic;" in payload_execution
assert "sceneInputs.glassCaustic = request.glassCaustic;" in payload_execution
assert "visualimportedscenerender::valid(operation)" in payload_execution
assert "Glass caustics require the exact retained Scene3D snapshot owner" in payload_execution
assert "sameGlassCaustic(" in payload_execution
assert "owner.glassCaustic = request.glassCaustic;" in payload_execution
assert "retiredOwnerSinceLastAttempt_ || request.glassCaustic" in payload_execution
assert "if (request.glassCaustic || hasMaterialFrame || use != ImportedSceneUse::Preview" in payload_execution
assert "this renderer does not yet support that request" not in execution

# The actual imported consumers compose the shared factory before destination
# rebuilding. Geometry Core continues through its existing compatibility entry.
factory = (root / "src/imported_material_frame_resolver.h").read_text()
executor = (root / "src/visual_plan_executor.h").read_text()
for consumer in (viewport, exporter):
    assert "ImportedMaterialFrameResolver::make<" in consumer
    assert "const auto importedFrameInputs = desc;" in consumer
    assert "importedFrameCompiled->importedSceneRender && importedFrameCompiled->materialFrameSource" in consumer
    assert "importedFrameInputs,&params.visualParams,invocationContext,temporalSamplingTransaction,activity" in consumer
    assert consumer.index("const auto importedFrameInputs = desc;") < consumer.index("const auto importedSceneResult =")
    route = consumer.partition("const auto importedSceneResult =")[2].partition("const auto importedResult =")[0]
    assert "importedFrameResolver ? &*importedFrameResolver : nullptr" in route
    assert route.index("prepareVisualImportedSceneLayerAtTime(") < route.index("desc.materialInputsAt = {};")
assert "NativeImportedSceneRenderUse::Preview>(renderer," in viewport
assert "NativeImportedSceneRenderUse::Export>(glctx.renderer," in exporter
assert "savedSegments = im.segments" in viewport
assert "segment,job.segments,t,sourceSec,job.fps" in exporter
assert "count != 1" in factory
assert "leaf.frame.frame == display.frame ? sourceSeconds" in factory
assert "selected.inSec + (at-selected.displayStartSec)*selected.rate" in factory
assert "plan.structuralRevision,compiled.importedSceneRender->evaluationRevision" in factory
assert "staging->bundle = std::make_shared<const Bundle>" in factory
assert "transaction.addPublication(" in factory
assert "[staging,activity]" in factory
assert "[staging] { staging->bundle.reset(); }" in factory
assert "&& !composedMaterialFrameResolver)" in execution
assert "Imported Surface Layer Source requires its saved clip and frame-time resolver" in execution
assert "composedMaterialFrameResolver->resolvePrepared(evaluation,request.frame,resolved,error)" in execution
assert "frameDependencyRollback" in execution
assert "prepareMaterialFrameDag" in factory and "prepareGeometryMaterialFrame" in executor
assert "materialFrameImageBytes(renderer.outputWidth(),renderer.outputHeight(),false" in executor
assert "pendingReadbackTransaction.emplace(std::move(frameTransaction));" in exporter
assert "pendingReadbackTransaction->commit(error)" in exporter
assert "pendingReadbackTransaction->commit(publicationError)" in exporter
assert "temporalSamplingTransaction.discard();" in viewport
assert "temporalSamplingTransaction.commit(commitError)" in viewport
