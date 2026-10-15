#!/usr/bin/env python3
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
GL = ROOT / "src/gpu_backend/backend_opengl.cpp"
METAL = ROOT / "src/gpu_backend/backend_sokol.mm"
NATIVE = ROOT / "src/native_animation_deformation.cpp"
ANIMATED = ROOT / "src/imported_animated_scene_payload_execution.cpp"

# Zero lights, or one Directional light, are admitted on GL and Metal. Two or
# more lights, and one non-Directional light, stay refused by the non-batch checks.
GL_OWNER_REFUSAL = "native OpenGL deformation requires one exact bounded scene owner"
METAL_OWNER_REFUSAL = "native Metal deformation requires one exact bounded scene owner"
OWNER_LIGHT_CLAUSE = "|| source->scene->lightCount > 1 || source->scene->cameraCount != 1"
CPU_LIGHT_CLAUSE = "|| source.scene->lightCount > 1 || source.scene->cameraCount != 1))"
CPU_DIRECTIONAL_CLAUSE = (
    "|| (source.scene->lightCount == 1 && source.scene->lights[0].kind "
    "!= HarmonicMIDI::grid::SceneLightKind::Directional)))"
)
CPU_ADMISSION_MESSAGE = (
    "native deformation admits one object, one material, one camera, and at most one light"
)
CPU_SUBSET_MESSAGE = "native deformation scene is outside the opaque fixture subset: "
CPU_SUBSET_RULE = "opaque material, and zero lights or one Directional light"
COLLAPSE_CLAUSE = (
    "scene_->lightCount <= 1 && scene_->materials[0].opacity == 1.0f "
    "&& (scene_->lightCount == 0 || scene_->lights[0].kind "
    "== HarmonicMIDI::grid::SceneLightKind::Directional))"
)


def normalized(text):
    return re.sub(r"\s+", " ", text)


def owner_gates(source, refusal):
    """Return the normalized if-gate that guards each occurrence of a refusal."""
    gates = []
    for match in re.finditer(re.escape(refusal), source):
        starts = list(re.finditer(r"\bif\s*\(", source[:match.start()]))
        gates.append(normalized(source[starts[-1].start():match.start()]))
    return gates


class LightlessDeformationAdmissionSourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gl = GL.read_text(encoding="utf-8")
        cls.metal = METAL.read_text(encoding="utf-8")
        cls.native = NATIVE.read_text(encoding="utf-8")
        cls.animated = ANIMATED.read_text(encoding="utf-8")

    def assertOwnerGate(self, source, refusal):
        gates = owner_gates(source, refusal)
        self.assertTrue(gates, refusal)
        for gate in gates:
            self.assertIn(OWNER_LIGHT_CLAUSE, gate, refusal)
            self.assertNotIn("lightCount != 1", gate, refusal)

    def test_gl_owner_gate_refuses_two_or_more_lights_only(self):
        self.assertOwnerGate(self.gl, GL_OWNER_REFUSAL)

    def test_metal_owner_gate_refuses_two_or_more_lights_only(self):
        self.assertOwnerGate(self.metal, METAL_OWNER_REFUSAL)

    def test_cpu_fixture_admission_takes_zero_or_one_directional_light(self):
        source = normalized(self.native)
        self.assertIn(CPU_LIGHT_CLAUSE, source)
        self.assertIn(CPU_DIRECTIONAL_CLAUSE, source)
        self.assertIn(CPU_ADMISSION_MESSAGE, self.native)
        self.assertIn(CPU_SUBSET_MESSAGE, self.native)
        self.assertIn(CPU_SUBSET_RULE, self.native)
        self.assertNotIn("admits one object, material, light, and camera", self.native)
        self.assertNotIn("|| source.scene->lights[0].kind", source)

    def test_single_source_collapse_takes_zero_or_one_directional_light(self):
        source = normalized(self.animated)
        self.assertIn(COLLAPSE_CLAUSE, source)
        self.assertNotIn("scene_->lightCount == 1 && scene_->materials[0].opacity", source)

    def test_zero_light_paths_never_read_a_light_record(self):
        gl = normalized(self.gl)
        metal = normalized(self.metal)
        self.assertIn("if (uLightCount == 0) return vec3(0.0);", gl)
        self.assertIn("if (lightCount == 1) { const auto& light = lightAt(0);", gl)
        self.assertIn("for (std::size_t i = 0; i < lightCount; ++i) hasDirectLight |= lightAt(i).kind", gl)
        self.assertIn("lightCount == 1 && lightAt(0).kind == SceneLightKind::Point ? 1 : 0", gl)
        self.assertIn("if (u.ambient.w == 0.0f) return float3(0.0f);", metal)
        self.assertIn("if (lightIndex >= uint(u.ambient.w)) break;", metal)
        self.assertIn(
            "storeVec3(uniforms.ambient, scene->ambientColor, static_cast<float>(lightCount));",
            metal)

    def test_dropped_node_world_diagnostic_reaches_the_local_transform_refusal(self):
        source = normalized(self.animated)
        for fragment in (
            "composition = {}, std::string* failure = nullptr)",
            "if (!arbitgpu::prepareNativeDeformationFrame(source, *evaluation.deformation, "
            "composition, frame, diagnostic, &sampled)) { if (failure) *failure = diagnostic; return {}; }",
            "std::string nodeWorldFailure;",
            "importedNodeWorlds(*baseScene_, *source_, *importedScene_, evaluation,compositionFrame,&nodeWorldFailure);",
            'error = "Imported object local transform has no exact animated node world"; '
            'if (!nodeWorldFailure.empty()) error += ": " + nodeWorldFailure; return false;',
        ):
            self.assertIn(fragment, source)

    def test_motion_refusals_keep_the_expected_worlds_diagnostic(self):
        source = normalized(self.animated)
        for fragment in (
            "const auto rejectMotion = [&](const char* message) { "
            "motion_[owner == ImportedAnimationEvaluationOwner::Preview ? 0 : 1] = {}; "
            "return reject(error, message); };",
            "const auto rejectMotionWith = [&](const std::string& message) { "
            "motion_[owner == ImportedAnimationEvaluationOwner::Preview ? 0 : 1] = {}; "
            "error = message; return false; };",
            "std::string expectedFailure;",
            "importedNodeWorlds(*baseScene_, *source_, *importedScene_, "
            "previousEvaluation,previous.compositionFrame,&expectedFailure);",
            'return rejectMotionWith(expectedFailure.empty() ? std::string("imported Motion node worlds '
            'disagree with the exact source snapshot") : "imported Motion node worlds disagree with '
            'the exact source snapshot: " + expectedFailure);',
        ):
            self.assertIn(fragment, source)


if __name__ == "__main__":
    unittest.main()
