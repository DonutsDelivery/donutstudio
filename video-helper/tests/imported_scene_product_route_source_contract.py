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

# Preview and export enter the same exact-payload/native-frame seam.
assert viewport.count("prepareVisualImportedSceneLayerAtTime(") == 1
assert exporter.count("prepareVisualImportedSceneLayerAtTime(") == 1
assert "sourceSec, valueFps," in viewport
assert "sourceSec, job.fps," in exporter
assert "NativeImportedSceneRenderUse::Preview" in viewport
assert "NativeImportedSceneRenderUse::Export" in exporter
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
