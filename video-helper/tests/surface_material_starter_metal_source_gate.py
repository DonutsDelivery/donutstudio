#!/usr/bin/env python3
"""Fail-closed source gate for the starter pack's physical Metal row."""

from pathlib import Path
import json
import sys


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def between(text: str, start: str, end: str) -> str:
    first = text.find(start)
    require(first >= 0, f"missing {start}")
    last = text.find(end, first + len(start))
    require(last >= 0, f"missing {end} after {start}")
    return text[first:last]


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1])
    cmake = (root / "CMakeLists.txt").read_text(encoding="utf-8")
    test = (root / "tests/metal_fixture_scene_backend_tests.mm").read_text(encoding="utf-8")
    opengl_test = (root / "tests/opengl_fixture_scene_backend_tests.cpp").read_text(encoding="utf-8")
    backend = (root / "src/gpu_backend/backend_sokol.mm").read_text(encoding="utf-8")
    generated = (root.parent / "shared/generated/SurfaceMaterialStarterPrograms.h").read_text(encoding="utf-8")
    oracle = json.loads((root / "tests/fixtures/surface_material_starter_expected.json").read_text(encoding="utf-8"))
    oracle_loader = (root / "tests/support/surface_material_starter_oracle.h").read_text(encoding="utf-8")

    fixture_targets = {
        "arbit-metal-fixture-diffraction-tests": between(
            cmake,
            "add_executable(arbit-metal-fixture-diffraction-tests",
            "add_executable(arbit-metal-rich-static-glb-tests"),
        "arbit-holographic-trading-card-metal-strict-tests": between(
            cmake,
            "add_executable(arbit-holographic-trading-card-metal-strict-tests",
            "if(GLFW3_FOUND AND OpenGL_FOUND)"),
    }
    oracle_definition = (
        'SURFACE_MATERIAL_STARTER_ORACLE_PATH="${CMAKE_CURRENT_SOURCE_DIR}'
        '/tests/fixtures/surface_material_starter_expected.json"')
    for target_name, target in fixture_targets.items():
        require("metal_fixture_scene_backend_tests.mm" in target
                and "fixture_scene_renderer.cpp" in target and "backend_sokol.mm" in target,
                f"{target_name} must compile production renderer and Metal backend owners")
        require("ARBIT_HAVE_VIEWPORT=0" in target
                and "ARBIT_HAVE_METAL_BACKEND=1" in target,
                f"{target_name} must select the headless Metal backend")
        require(oracle_definition in target
                and "nlohmann_json::nlohmann_json" in target,
                f"{target_name} must load the exact checked-in JSON oracle")
        require('"-framework Metal"' in target and '"-framework Cocoa"' in target,
                f"{target_name} must link native Metal and Cocoa")

    opengl_fixture_targets = {
        "arbit-opengl-fixture-scene-backend-tests": between(
            cmake,
            "add_executable(arbit-opengl-fixture-scene-backend-tests",
            "add_executable(arbit-opengl-packaged-multi-object-acceptance"),
        "arbit-opengl-packaged-multi-object-acceptance": between(
            cmake,
            "add_executable(arbit-opengl-packaged-multi-object-acceptance",
            "add_executable(arbit-opengl-animation-deformation-backend-tests"),
    }
    for target_name, target in opengl_fixture_targets.items():
        require("opengl_fixture_scene_backend_tests.cpp" in target,
                f"{target_name} must compile the shared OpenGL fixture test")
        require("ARBIT_HAVE_VIEWPORT=1" in target
                and "ARBIT_HAVE_METAL_BACKEND=0" in target
                and "ARBIT_HAVE_OPENGL_SDF_BACKEND=1" in target,
                f"{target_name} must select the OpenGL fixture backend")
        require(oracle_definition in target
                and "nlohmann_json::nlohmann_json" in target,
                f"{target_name} must load the exact checked-in JSON oracle")

    target = fixture_targets["arbit-metal-fixture-diffraction-tests"]
    require("RUN_SERIAL TRUE" in target and "native-surface-material-runtime" in target,
            "strict target must be labeled and serialized")
    require("SKIP_RETURN_CODE" not in target and "backend_stub" not in target,
            "strict Metal target must have no skip or fallback")
    require("strict physical Metal fixture backend unavailable" in test
            and "requires physical Apple Silicon" in test
            and "__aarch64__" in test and "__arm64__" in test,
            "runtime must fail off physical Apple Silicon or when Metal is unavailable")
    strict_target = fixture_targets["arbit-holographic-trading-card-metal-strict-tests"]
    require("ARBIT_HOLOGRAPHIC_TRADING_CARD_STRICT=1" in strict_target,
            "holographic card target must enable its strict-only assertions")
    require("surfacematerialstarterfixture::kPrograms" in test
            and generated.count("visual.material.") == 5
            and len(oracle["materials"]) == 5
            and oracle["schema"] == 3
            and "kSchemaVersion = 3" in oracle_loader,
            "physical target must consume the one generated five-program table")
    require("14695981039346656037ull" in oracle_loader
            and "surface_material_starter_oracle_cross_language" in cmake,
            "oracle must bind standard FNV-1a through a compiled cross-language check")
    require("renderPreview" in test and "renderExport" in test
            and "FixtureSceneRenderer previewOwner" in test
            and "FixtureSceneRenderer exportOwner" in test,
            "physical target must use distinct production preview and export owners")
    require("actualRgba == expected.centerRgba8" in test
            and "oracle.sampledRgba8Fnv1a64" in test
            and "rejectedAlternatives" in test,
            "physical target must enforce pinned JSON pixels, digest, and wrong-shading alternatives")
    require("malformed Surface PBR values" in test
            and "malformed Surface PBR identity" in test,
            "physical target must reject malformed starter values and identity before allocation")
    for source, backend_name in ((test, "Metal"), (opengl_test, "OpenGL")):
        require("{ -0.16773126f, 0.25488700f, 0.04494346f, 0.95125124f }" in source
                and "ambientColor = { 0.12f, 0.12f, 0.12f }" in source
                and "lights[0].transform.rotation" in source
                and "lights[0].intensity = 0.88f" in source,
                f"{backend_name} starter fixture must use the shared explicit transforms and lighting")
        require('"/base-color"' in source,
                f"{backend_name} starter fixture must include a controlled Base Color alternative")
    require("Metal fixture preparation requires an exact bounded material binding" in backend,
            "production Metal backend must retain fail-closed material admission")
    vertex_shader = between(backend, "const char* kFixtureVertexShader", "const char* kFixtureFragmentShader")
    fragment_shader = between(backend, "const char* kFixtureFragmentShader", "[[maybe_unused]] const char* kDeformationComputeShader")
    require(vertex_shader.count("float3 litBase;") == 1
            and fragment_shader.count("float3 litBase;") == 1
            and "out.litBase = out.baseColor * u.ambient.xyz;" in vertex_shader
            and "in.litBase" in fragment_shader,
            "Metal vertex and fragment interfaces must carry the diffraction fallback color")
    print("Surface Material Starter Pack strict physical Metal source gate PASS; execution remains Apple-Silicon-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
