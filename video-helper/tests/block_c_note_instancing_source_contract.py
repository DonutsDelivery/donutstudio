#!/usr/bin/env python3
"""Production-route checks for canonical Block C note instancing."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]


def source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


class BlockCNoteInstancingSourceContract(unittest.TestCase):
    def test_compiled_mapping_and_frame_reach_fixture_runtime(self) -> None:
        compiler = source("video-helper/src/visual_plan_executor.h")
        preparation = source("video-helper/src/imported_scene_visual_plan_execution.h")
        execution = source("video-helper/src/imported_scene_payload_execution.cpp")

        self.assertIn("visualnoteinstancing::decode(noteInstancer->payloadXml", compiler)
        self.assertIn("execution.noteInstanceMapping = noteInstanceMapping", compiler)
        self.assertIn("request.runtimeInputs.canonicalBlockCFrame = canonicalBlockCFrame", preparation)
        self.assertIn("request.runtimeInputs.noteInstanceMapping = compiled->noteInstanceMapping", preparation)
        self.assertIn("auto sceneInputs = request.runtimeInputs", execution)
        self.assertIn("deformationRequest.runtimeInputs = request.runtimeInputs", execution)
        self.assertIn("receipt.canonicalBlockCFrame = request.runtimeInputs.canonicalBlockCFrame", execution)
        self.assertIn("layer.canonicalBlockCFrame = receipt.canonicalBlockCFrame", preparation)
        self.assertNotIn("frameOwners.clear()", preparation)

    def test_malformed_mapping_fails_closed(self) -> None:
        contract = source("shared/VisualNoteInstancingContract.h")
        compiler = source("video-helper/src/visual_plan_executor.h")
        backend = source("video-helper/src/gpu_backend/backend.h")

        self.assertIn("output = {}", contract)
        self.assertIn("encode(candidate)!=encoded", contract)
        self.assertIn("! exactNoteInstanceMapping", compiler)
        self.assertIn("== inputs.noteInstanceMapping.has_value()", backend)
        self.assertIn("visualnoteinstancing::valid(*inputs.noteInstanceMapping)", backend)

    def test_both_production_backends_issue_bounded_instanced_draws(self) -> None:
        backend = source("video-helper/src/gpu_backend/backend.h")
        opengl = source("video-helper/src/gpu_backend/backend_opengl.cpp")
        metal = source("video-helper/src/gpu_backend/backend_sokol.mm")

        self.assertIn("visualnoteinstancing::kMaximumInstances", backend)
        self.assertIn("frame.noteIdentities[row]", backend)
        self.assertIn("batch.canonicalRows[index]", backend)
        self.assertIn("prepareNativeNoteInstances(runtimeInputs)", opengl)
        self.assertIn("uNoteInstanceTransforms[noteIndex]", opengl)
        self.assertIn("uNoteInstanceIndex >= 0 ? uNoteInstanceIndex : gl_InstanceID", opengl)
        self.assertIn("gl.DrawElementsInstanced", opengl)
        self.assertIn("prepareNativeNoteInstances(runtimeInputs)", metal)
        self.assertIn("uint instance [[instance_id]]", metal)
        self.assertIn("u.noteInstanceTransforms[noteIndex]", metal)
        self.assertIn("static_cast<int>(noteInstances.count)", metal)

    def test_geometry_and_note_instance_authorities_remain_separate(self) -> None:
        backend = source("video-helper/src/gpu_backend/backend.h")
        geometry = source("video-helper/src/geometry_core_backend.cpp")
        opengl = source("video-helper/src/gpu_backend/backend_opengl.cpp")
        metal = source("video-helper/src/gpu_backend/backend_sokol.mm")

        self.assertIn("prepareGeometryInstances", backend)
        self.assertIn("instanceBufferUploadCount", backend)
        self.assertIn("submittedInstanceCount", backend)
        self.assertIn("noteInstanceTransformUploadCount", backend)
        self.assertIn("submittedNoteInstanceCount", backend)
        self.assertIn("std::move(runtimeInputs)", geometry)
        self.assertIn("resources->instanceBuffer", opengl)
        self.assertIn("locations.noteInstanceIndex", opengl)
        self.assertIn("for (std::size_t noteIndex = 0; noteIndex < noteInstances.count", opengl)
        self.assertIn("noteInstanceTransformUploadCount = noteInstances.admitted ? 1u : 0u", opengl)
        self.assertIn("noteInstanceTransformUploadCount = noteInstances.admitted ? 1u : 0u", metal)
        self.assertIn("NativeFixtureScenePreparation prepareGeometryInstances", metal)
        self.assertIn("resources->geometryAdmission = geometryAdmission", metal)
        self.assertIn("arbit-metal-geometry-core-instances", metal)
        self.assertIn("sg_update_buffer (resources->geometryInstanceBuffer", metal)
        self.assertIn("static_cast<int> (scene->objectCount)", metal)
        self.assertIn("uniforms.geometryInstanceControl[1]", metal)
        self.assertIn("result.stats.instancedDrawCount = instancedDrawCount", metal)
        self.assertIn("result.stats.ordinaryDrawCount = resources->instancedSharedGeometry", metal)
        self.assertIn("diffractionFoilField", metal)
        self.assertIn("degreesToRadians", metal)

    def test_apple_headers_name_the_global_canonical_type(self) -> None:
        for path in (
            "video-helper/src/gpu_backend/metal_shader_generator.h",
            "video-helper/src/gpu_backend/particle_engine_metal.h",
        ):
            header = source(path)
            self.assertIn("namespace canonicalblockc { class CanonicalBlockCFrame; }", header)
            self.assertIn("::canonicalblockc::CanonicalBlockCFrame", header)
            videorender = header.index("namespace videorender")
            declaration = header.index("namespace canonicalblockc")
            self.assertLess(declaration, videorender)


if __name__ == "__main__":
    unittest.main()
