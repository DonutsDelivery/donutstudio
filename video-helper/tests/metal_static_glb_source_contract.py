#!/usr/bin/env python3
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/gpu_backend/backend_sokol.mm"
CMAKE = ROOT / "CMakeLists.txt"
RICH_TEST = ROOT / "tests/metal_rich_static_glb_backend_tests.mm"


class MetalStaticGlbSourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_text(encoding="utf-8")
        cls.cmake = CMAKE.read_text(encoding="utf-8")
        cls.rich_test = RICH_TEST.read_text(encoding="utf-8")
        start = cls.source.index("class MetalFixtureSceneBackend final")
        end = cls.source.index("class MetalDeformationBackend final", start)
        cls.backend = cls.source[start:end]
        shader_start = cls.source.index('const char* kFixtureVertexShader')
        cls.shaders = cls.source[shader_start:start]

    def test_rich_scene_admission_and_exact_receipt_identity(self):
        for token in (
            "validateVisual3DScene (*scene).valid()",
            "scene->cameraCount > Visual3DScene::kMaxCameras",
            "scene->activeCamera",
            "resources->rendererGeneration = nextMetalRendererGeneration()",
            "frame->rendererGeneration_ = resources->rendererGeneration",
            "reinterpret_cast<std::uintptr_t> (staticResources.get())",
        ):
            self.assertIn(token, self.source)
        self.assertNotIn("frame->rendererGeneration_ = nextMetalRendererGeneration()", self.backend)

    def test_material_textures_preserve_color_roles_and_samplers(self):
        for token in (
            "1 + scene->textureCount * 2",
            "SG_PIXELFORMAT_SRGB8A8 : SG_PIXELFORMAT_RGBA8",
            "source->minFilter",
            "source->magFilter",
            "source->wrapS",
            "source->wrapT",
            '"metallicRoughnessTexture"',
            '"normalTexture"',
            '"occlusionTexture"',
            '"emissiveTexture"',
        ):
            self.assertIn(token, self.backend)

    def test_shader_has_tangent_pbr_alpha_and_all_light_kinds(self):
        for token in (
            "float4 tangent [[attribute(4)]]",
            "in.bitangentSign",
            "in.tangent.w * sign(objectDeterminant)",
            "metallicRoughnessTexel.g",
            "metallicRoughnessTexel.b",
            "discard_fragment()",
            "kind == 1u || kind == 3u",
            "smoothstep(u.lightCones[lightIndex].y",
            "kind == 2u",
        ):
            self.assertIn(token, self.shaders)

    def test_hierarchy_winding_and_stable_alpha_queue_are_explicit(self):
        for token in (
            "metalFixtureWorldMatrix",
            "parentObject->parent",
            "metalFixtureDeterminant3x3",
            "SG_FACEWINDING_CW",
            "SG_FACEWINDING_CCW",
            "std::stable_sort (blendedDraws.begin(), blendedDraws.end()",
            "drawOrder.push_back (objectIndex)",
            "pipelineDesc.depth.write_enabled = blend == 0",
            "SG_BLENDFACTOR_SRC_ALPHA",
        ):
            self.assertIn(token, self.source)

    def test_fixture_metal_uses_current_layout_and_resource_collections(self):
        for token in (
            "sizeof (FixtureUniforms) == 4160",
            "offsetof (FixtureUniforms, diffractionEvaluationSchedule) == 2080",
            "offsetof (FixtureUniforms, noteInstanceTransforms) == 2096",
            "offsetof (FixtureUniforms, geometryInstanceControl) == 4144",
        ):
            self.assertIn(token, self.source)
        for token in (
            "validNativeFixtureDiffractionProgram (",
            "static_cast<std::uint8_t> (slot)",
            "resources->pipelines[winding]",
            "return 1 + index * 2 + (srgb ? 1 : 0)",
            "const auto slot = textureSlot (textureIds[unit], unit == 0 || unit == 4)",
            "bindings.views[unit] = resources->textureViews[slot]",
            "bindings.samplers[unit] = resources->samplers[slot]",
            "drawObject.indexCount",
        ):
            self.assertIn(token, self.backend)
        for stale in (
            "materialProgram->diffraction.",
            "resources->pipeline)",
            "resources->textureView;",
            "resources->sampler;",
            "object.indexCount",
        ):
            self.assertNotIn(stale, self.backend)

    def test_strict_rich_static_glb_native_target_has_no_skip_path(self):
        for token in (
            "add_executable(arbit-metal-rich-static-glb-tests",
            "tests/metal_rich_static_glb_backend_tests.mm",
            "add_test(NAME metal_rich_static_glb",
            'LABELS "strict-static-glb-runtime;physical-apple-silicon"',
        ):
            self.assertIn(token, self.cmake)
        for token in (
            "decodeStaticGlb",
            "adaptStaticGlbToVisual3DScene",
            "spot cone",
            "finite light range cutoff",
            "simultaneous sRGB and linear texture roles",
            "reflected single-sided geometry",
            "overlapping Metal alpha draws",
            "one preparation lifetime",
        ):
            self.assertIn(token, self.rich_test)
        self.assertNotIn("return 77", self.rich_test)
        self.assertNotIn("SKIP", self.rich_test)


if __name__ == "__main__":
    unittest.main()
