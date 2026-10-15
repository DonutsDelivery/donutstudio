#!/usr/bin/env python3
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
GL = ROOT / "src/gpu_backend/backend_opengl.cpp"
METAL = ROOT / "src/gpu_backend/backend_sokol.mm"
PAYLOAD = ROOT / "src/imported_scene_payload_execution.cpp"
ANIMATED = ROOT / "src/imported_animated_scene_payload_execution.cpp"
SEAM = ROOT / "src/glb_native_render_seam.cpp"
BACKEND = ROOT / "src/gpu_backend/backend.h"
COMPOSITION = ROOT / "src/native_scene_animation_composition.h"

# A raster backend that cannot consume a checked object world refuses it
# behind an if gate that reads objectWorlds.
GL_REFUSALS = (
    "OpenGL static local object worlds do not match the drawn scene objects",
    "OpenGL static local object worlds are not finite affine worlds with distinct object ids",
    "OpenGL static local object worlds do not yet admit diffraction gratings",
    "OpenGL static local object worlds do not yet admit instanced draws",
    "OpenGL static local object worlds require one rigid imported deformation leaf",
    "OpenGL static local object worlds do not yet admit glass caustics",
)
METAL_REFUSALS = (
    "Metal static local object worlds do not yet admit glass caustics",
    "Metal static local object worlds do not yet admit diffraction gratings",
    "Metal static local object worlds do not match the drawn scene objects",
    "Metal static local object worlds are not finite affine worlds with distinct object ids",
    "Metal static local object worlds do not yet admit instanced draws",
    "Metal static local object worlds require one rigid imported deformation leaf",
)
# The payload executor refuses these before any candidate is staged.
PAYLOAD_REFUSALS = (
    "Imported object local transforms require a native evaluation owner",
    "Static local object transforms require one exact imported leaf; several Scene Compose leaves are not rendered yet",
    "Static local object transforms require one exact imported leaf",
    "Static local object transforms in a Scene Compose need its procedural scene",
    "Static local object transforms do not yet admit note instancing",
    "Static local object transforms do not yet admit glass caustics or diffraction gratings",
)
# Staging refuses these after it builds the candidate frame, before it publishes it.
SINGLE_LEAF_COVER_REFUSAL = "Static local single leaf worlds must name every drawn object exactly"
EMPTY_COMPOSE_REFUSAL = "Static local Scene Compose moves no drawn imported object"
# Motion output under checked worlds is stepped by the pre-check, not refused.
REMOVED_MOTION_REFUSAL = "do not yet admit the Motion pass"
REMOVED_SEAM_REFUSAL = "static local object worlds await typed backend consumers"
RETENTION_REFUSAL = "static local backend retention and typed consumers are not admitted"
COVER_CHECK = "!nativeImportedObjectWorldsCover(*scene, *runtimeInputs.objectWorlds)"
VALIDITY_CHECK = "!validNativeImportedObjectWorlds(*runtimeInputs.objectWorlds)"
FIXTURE_CHECK = "!validFixtureRuntimeInputs(runtimeInputs)"
EXACT_COVER_GATE = (
    "if (!request.importedScene && !arbitgpu::nativeImportedObjectWorldsExactCover(*authority->scene_, *worlds))"
)
COMPOSED_EMPTY_GATE = "if (request.importedScene && worlds->empty())"
PUBLISH = "frame->objectWorlds_ = std::move(worlds);"
# The animated executor admits one rigid imported leaf with a non-neutral object-local
# transform. Every other use is refused by name, before or after leaf admission.
ANIMATED_LOCAL_REFUSALS = (
    "Imported object local transform is malformed",
    "Imported object local transforms with imported parenting are not rendered yet",
    "Imported object local transforms with procedural composition are not rendered yet",
    "Imported object local transforms with a selected source camera are not rendered yet",
    "Imported object local transforms with note instancing are not rendered yet",
    "Imported object local transforms with glass caustics are not rendered yet",
    "Imported object local transforms with diffraction gratings are not rendered yet",
    "Imported object local transforms cannot reuse a composition frame or caller object worlds",
)
ANIMATED_LEAF_REFUSALS = (
    "Imported object local transform has no admitted deformation leaf",
    "Imported object local transform does not name a node of the selected leaf",
    "Imported object local transform does not name the selected mesh",
    "Imported object local transforms with imported parenting are not rendered yet",
    "Imported object local transforms with skinned meshes are not rendered yet",
    "Imported object local transforms with skin joints in the moved subtree are not rendered yet",
    "Imported object local transforms with morph targets are not rendered yet",
    "Imported object local transforms require one rigid imported deformation leaf",
    "Imported object local transform names no drawn object of the selected leaf",
)
ANIMATED_FRAME_REFUSALS = (
    "Imported object local transform has no exact animated node world",
    "Imported object local transform animated world is singular or nonfinite",
    "Imported object local transform draw identity is invalid",
    "Imported object local transform draw ancestry is invalid",
    "Imported object local transform worlds do not name every drawn object exactly",
)


class StaticLocalObjectWorldSourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gl = GL.read_text(encoding="utf-8")
        cls.metal = METAL.read_text(encoding="utf-8")
        cls.payload = PAYLOAD.read_text(encoding="utf-8")
        cls.animated = ANIMATED.read_text(encoding="utf-8")
        cls.seam = SEAM.read_text(encoding="utf-8")
        cls.backend = BACKEND.read_text(encoding="utf-8")
        cls.composition = COMPOSITION.read_text(encoding="utf-8")

    def assertGatedBy(self, source, messages, token):
        for message in messages:
            self.assertIn(message, source)
            for match in re.finditer(re.escape(message), source):
                gates = list(re.finditer(r"\bif\s*\(", source[:match.start()]))
                self.assertTrue(gates, message)
                gate = source[gates[-1].start():match.start()]
                self.assertIn(token, gate, message)

    def assertGatedRefusals(self, source, messages):
        self.assertGatedBy(source, messages, "objectWorlds")

    def assertGatedCall(self, source, call):
        matches = list(re.finditer(re.escape(call), source))
        self.assertTrue(matches, call)
        for match in matches:
            gates = list(re.finditer(r"\bif\s*\(", source[:match.start()]))
            self.assertTrue(gates, call)
            self.assertIn("objectWorlds", source[gates[-1].start():match.start()], call)

    def test_both_raster_backends_check_the_drawn_object_cover(self):
        self.assertIn(COVER_CHECK, self.gl)
        self.assertIn(COVER_CHECK, self.metal)

    def test_both_raster_backends_read_checked_worlds_for_raster_and_visibility(self):
        gl_uses = re.findall(
            r"nativeObjectWorldMatrix\s*\(\s*runtimeInputs\.objectWorlds\.get\(\)", self.gl)
        metal_uses = re.findall(
            r"nativeObjectWorldMatrix\s*\(\s*runtime(?:Inputs)?\.objectWorlds\.get\(\)", self.metal)
        self.assertGreaterEqual(len(gl_uses), 4)
        self.assertGreaterEqual(len(metal_uses), 4)

    def test_gl_refusals_sit_behind_the_object_world_gate(self):
        self.assertGatedRefusals(self.gl, GL_REFUSALS)

    def test_metal_refusals_sit_behind_the_object_world_gate(self):
        self.assertGatedRefusals(self.metal, METAL_REFUSALS)

    def test_raster_world_validity_has_its_own_refusal_before_the_fixture_check(self):
        for name, source in (("gl", self.gl), ("metal", self.metal)):
            self.assertIn(VALIDITY_CHECK, source, name)
            self.assertIn(FIXTURE_CHECK, source, name)
            self.assertLess(source.index(VALIDITY_CHECK), source.index(FIXTURE_CHECK), name)

    def test_payload_refuses_unsupported_static_local_paths_before_staging(self):
        self.assertIn(
            "if (const auto refusal = staticLocalRefusal(request)) return reject(error, refusal);",
            self.payload)
        for message in PAYLOAD_REFUSALS:
            self.assertIn(message, self.payload)
        for gate in (
            "request.objectLocalTransform.version != 1",
            "request.deformation || request.parentIntent",
            "request.runtimeInputs.noteInstanceMapping",
            "request.glassCaustic || request.diffractionMaterial",
        ):
            self.assertIn(gate, self.payload)

    def test_payload_comment_names_only_the_refusals_it_keeps(self):
        match = re.search(r"((?://[^\n]*\n)+)static const char\* staticLocalRefusal\(", self.payload)
        self.assertIsNotNone(match, "staticLocalRefusal comment is missing")
        comment = re.sub(r"\s+", " ", match.group(1).replace("//", " "))
        for phrase in (
            "animation deformation",
            "parenting",
            "Several Scene Compose leaves are refused",
            "a procedural scene without its leaf",
            "a leaf without its procedural scene",
            "note instancing",
            "glass caustics",
            "diffraction gratings",
            "One exact static leaf",
        ):
            self.assertIn(phrase, comment)
        self.assertNotIn("Motion", comment)

    def test_motion_refusal_is_lifted_from_every_layer(self):
        for name, source in (("gl", self.gl), ("metal", self.metal), ("payload", self.payload)):
            self.assertNotIn(REMOVED_MOTION_REFUSAL, source, f"Motion refusal remains in {name}")
        refusal = re.search(r"static const char\* staticLocalRefusal\(.*?\n\}", self.payload, re.S)
        self.assertIsNotNone(refusal, "staticLocalRefusal is missing")
        self.assertNotIn("sceneUsesMotionPass", refusal.group(0), "staticLocalRefusal still gates Motion")

    def test_motion_pass_steps_changed_worlds_at_both_raster_backends(self):
        for helper in (
            "inline bool nativeObjectWorldMotionSupported(",
            "inline bool nativeObjectWorldStep(",
            "inline NativeSceneMotionUniforms sceneObjectWorldMotionUniforms(",
        ):
            self.assertIn(helper, self.backend)
        self.assertGatedCall(self.gl, "!nativeObjectWorldMotionSupported(*scene, runtimeInputs, result.error)")
        self.assertGatedCall(self.metal, "!nativeObjectWorldMotionSupported(*scene, runtimeInputs, result.error)")
        self.assertRegex(self.gl, r"objectWorlds\s*\?\s*sceneObjectWorldMotionUniforms\(motion,")
        self.assertRegex(self.metal, r"objectWorlds\s*\?\s*sceneObjectWorldMotionUniforms\(frameMotion,")
        for source in (self.gl, self.metal):
            self.assertIn("previousMotion->objectWorlds.get()", source)

    def test_admitted_candidate_worlds_reach_the_runtime_inputs(self):
        self.assertIn("sceneInputs.objectWorlds = localCandidate->objectWorlds();", self.payload)
        self.assertIn(
            "if (!localCandidate->objectWorlds() || localCandidate->owner()->scene() != admission.scene)",
            self.payload)

    def test_seam_no_longer_refuses_object_worlds(self):
        self.assertNotIn(REMOVED_SEAM_REFUSAL, self.seam)
        self.assertIn(RETENTION_REFUSAL, self.seam)

    def test_composed_leaf_lists_only_the_moved_leaf_worlds(self):
        for fragment in (
            "if (request.importedScene)",
            "if (object.id.value & 0x80000000u) continue;",
            "if (!frame->descendant(objectNode, affected))",
            "if (!affected) continue;",
            PUBLISH,
        ):
            self.assertIn(fragment, self.payload)

    def test_single_leaf_gate_refuses_several_leaves_and_unpaired_scenes(self):
        self.assertIn("if (!request.additionalImportedScenes.empty())", self.payload)
        self.assertIn("if (request.sceneSnapshot && !request.importedScene)", self.payload)
        self.assertIn("if (request.importedScene && !request.sceneSnapshot)", self.payload)
        self.assertNotIn(
            "request.importedScene || request.sceneSnapshot || !request.additionalImportedScenes.empty()",
            self.payload)
        self.assertNotIn("(!request.importedScene && !add(imported->objectCount", self.payload)

    def test_execute_drops_caller_worlds_before_the_candidate_supplies_its_own(self):
        reset = self.payload.index("sceneInputs.objectWorlds.reset();")
        attach = self.payload.index("sceneInputs.objectWorlds = localCandidate->objectWorlds();")
        self.assertLess(reset, attach)

    def test_cover_accepts_a_nonempty_subset_of_drawn_objects(self):
        self.assertIn("if (worlds.empty()) return false;", self.backend)
        self.assertNotIn("if (worlds.size() != drawn) return false;", self.backend)

    def test_single_leaf_worlds_must_name_every_drawn_object_exactly(self):
        self.assertIn("inline bool nativeImportedObjectWorldsExactCover(", self.backend)
        self.assertIn("return drawn == worlds.size();", self.backend)
        self.assertIn(EXACT_COVER_GATE, self.payload)

    def test_staged_refusals_sit_behind_their_leaf_gates_before_publication(self):
        publish = self.payload.index(PUBLISH)
        self.assertLess(self.payload.index(SINGLE_LEAF_COVER_REFUSAL), publish)
        self.assertLess(self.payload.index(EMPTY_COMPOSE_REFUSAL), publish)
        self.assertGatedBy(self.payload, (SINGLE_LEAF_COVER_REFUSAL,), EXACT_COVER_GATE)
        self.assertGatedBy(self.payload, (EMPTY_COMPOSE_REFUSAL,), COMPOSED_EMPTY_GATE)

    def test_static_candidate_binding_compares_the_checked_worlds(self):
        self.assertIn(
            "arbitgpu::sameNativeImportedObjectWorlds(objectWorlds_, runtime.objectWorlds)",
            self.composition)

    def test_animated_route_keeps_the_neutral_gate_and_the_compose_refusal(self):
        self.assertIn(
            "const bool localTransform = !visualimportedobjectlocal::same(request.objectLocalTransform, {});",
            self.animated)
        self.assertRegex(
            self.animated,
            r"if \(!visualimportedobjectlocal::same\(request\.objectLocalTransform, \{\}\)\)\s*"
            r"return reject\(error, \"Imported object local transforms require a native evaluation owner\"\);")

    def test_animated_local_refusals_sit_in_the_object_local_block(self):
        start = self.animated.index("const bool localTransform = ")
        end = self.animated.index("const bool composition = request.proceduralScene != nullptr;")
        self.assertLess(start, end)
        for message in ANIMATED_LOCAL_REFUSALS:
            self.assertIn(message, self.animated)
            self.assertTrue(
                any(start < match.start() < end for match in re.finditer(re.escape(message), self.animated)),
                message)

    def test_animated_motion_refusal_precedes_the_structural_checks(self):
        reject_motion = self.animated.index("const auto rejectMotion = [&](const char* message) {")
        gate = self.animated.index("if (localTransform && usesMotion)")
        structural = self.animated.index("if (request.structuralRevision == 0 || request.frame.rateNumerator == 0")
        self.assertLess(reject_motion, gate)
        self.assertLess(gate, structural)
        self.assertIn(
            "return rejectMotion(\"Imported object local transforms with Motion are not rendered yet\");",
            self.animated)

    def test_animated_leaf_refusals_follow_admission_in_the_helper(self):
        helper = self.animated.index("const char* localTransformRefusal(")
        self.assertIn("std::optional<bool> localTransformAffects(", self.animated)
        call = self.animated.index("localTransformRefusal(*baseScene_, *scene_, bindings_")
        self.assertLess(helper, call)
        for message in ANIMATED_LEAF_REFUSALS:
            self.assertTrue(
                any(match.start() < call for match in re.finditer(re.escape(message), self.animated)),
                message)

    def test_animated_rigid_leaf_predicate_checks_the_synthetic_skin(self):
        for fragment in (
            "if (binding->authoredJointCount != 0)",
            "if (!mesh->morphTargets().empty())",
            "if (!binding->skin.isValid() || (mesh->skin().isValid() && mesh->skin() != binding->skin))",
            "skin->joints().front().id().value != node + 1u",
            "!std::equal(identity.begin(), identity.end(), inverse.begin())",
        ):
            self.assertIn(fragment, self.animated)

    def test_animated_frame_worlds_are_checked_before_they_are_published(self):
        publish = self.animated.index("runtimeInputs.objectWorlds = std::move(worlds);")
        self.assertLess(
            self.animated.index("arbitgpu::nativeImportedObjectWorldsExactCover(*scene_, *worlds)"), publish)
        self.assertLess(publish, self.animated.index("runtimeInputs.previousMotion.reset();"))
        self.assertIn("const auto correction = NativeSceneAnimationCompositionFrame::multiply(", self.animated)
        self.assertIn(
            "NativeSceneAnimationCompositionFrame::multiply(correction, authored) : authored",
            self.animated)
        for message in ANIMATED_FRAME_REFUSALS:
            self.assertIn(message, self.animated)

    def test_deformation_refusal_admits_only_the_rigid_leaf_shape(self):
        gate = re.compile(
            r"if \(runtimeInputs\.objectWorlds\s*&&\s*\(source == nullptr \|\| runtimeInputs\.compositionFrame"
            r"\s*\|\| source->compositionOwner\s*\|\|\s*!validNativeImportedObjectWorlds\(\*runtimeInputs\.objectWorlds\)\)\)")
        for name, source in (("gl", self.gl), ("metal", self.metal)):
            self.assertRegex(source, gate, name)
            self.assertNotIn("do not apply to imported geometry deformation", source, name)

    def test_touched_sources_carry_no_merge_markers(self):
        marker = re.compile(r"^(<{7} |={7}$|>{7} )", re.M)
        for name, source in (
            ("payload", self.payload),
            ("animated", self.animated),
            ("composition", self.composition),
            ("backend", self.backend),
            ("gl", self.gl),
            ("metal", self.metal),
        ):
            self.assertIsNone(marker.search(source), name)


if __name__ == "__main__":
    unittest.main()
