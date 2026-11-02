#!/usr/bin/env python3
"""Pins the opaque black backdrop of a camera-hidden environment in both backends.

Path Trace sees no world on a camera ray when the Environment Light hides itself.
The raster clear has to match it, and the hidden flag must not reach the lighting.
The native pixels are checked by support/environment_image_cases.h on each backend.
"""
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
opengl = (root / "src/gpu_backend/backend_opengl.cpp").read_text(encoding="utf-8")
sokol = (root / "src/gpu_backend/backend_sokol.mm").read_text(encoding="utf-8")
cases = (root / "tests/support/environment_image_cases.h").read_text(encoding="utf-8")
failures = []


def check(condition, message):
    if not condition:
        failures.append(message)


def before(text, first, second, message):
    a, b = text.find(first), text.find(second)
    check(0 <= a < b, message)


# OpenGL: the scene clear and the diffraction resolve clear share one choice.
gl_scan = "!lightAt(i).environmentCameraVisible"
gl_black = "const float hiddenBackdrop[4] { 0, 0, 0, 1 };"
gl_choice = "const float* background = cameraHiddenBackdrop ? hiddenBackdrop : stageBackground;"
gl_scene = ("gl.ClearBufferfv(GL_COLOR, 0, diffractionFrame ? emptyPass\n"
            "            : runtimeInputs.imageOutput == renderpassoutput::Output::Color\n"
            "                && (!runtimeInputs.passComposite || cameraHiddenBackdrop) ? background : emptyPass);")
gl_resolve = ("const auto clearColor = runtimeInputs.imageOutput == renderpassoutput::Output::Color\n"
              "                && (!runtimeInputs.passComposite || cameraHiddenBackdrop) ? background : emptyPass;")
check(gl_black in opengl, "OpenGL has an opaque black backdrop")
check(gl_choice in opengl, "OpenGL picks black over the stage colour for a hidden environment")
check(gl_scene in opengl, "OpenGL scene clear uses the backdrop choice")
check(gl_resolve in opengl, "OpenGL diffraction resolve clear uses the backdrop choice")
before(opengl, gl_scan, gl_choice, "OpenGL reads the hidden flag before the clear")
before(opengl, gl_choice, "gl.ClearBufferfv(GL_COLOR, 0, diffractionFrame", "OpenGL chooses the backdrop before the clear")

# Metal: the black clear comes after the stage colour and before the resolve captures it.
mtl_scan = "light.kind == SceneLightKind::Environment && !light.environmentCameraVisible"
mtl_black = ("if (cameraHiddenBackdrop)\n"
             "                pass.action.colors[0].clear_value = { 0.0f, 0.0f, 0.0f, 1.0f };")
check(mtl_black in sokol, "Metal clears a hidden environment's backdrop to opaque black")
before(sokol, mtl_scan, mtl_black, "Metal reads the hidden flag before the clear")
before(sokol, "if (runtimeInputs.imageOutput != renderpassoutput::Output::Color || runtimeInputs.passComposite)",
       mtl_black, "Metal black overrides the stage and composite clears")
before(sokol, mtl_black, "const auto resolvedColorClear = pass.action.colors[0].clear_value;",
       "Metal diffraction resolve keeps the black clear")

# An inspection image keeps its own clear in both backends.
for label, text, choice, inspection in (
        ("OpenGL", opengl, gl_choice, "const auto clear = renderpasscomposite::inspectionClear(*runtimeInputs.passComposite);"),
        ("Metal", sokol, mtl_black, "const auto clear = renderpasscomposite::inspectionClear(*runtimeInputs.passComposite);")):
    at = text.find(choice)
    check(at >= 0 and text.find(inspection, at) > at, label + " inspection clear still runs after the backdrop choice")
for label, text in (("OpenGL", opengl), ("Metal", sokol)):
    check("const bool cameraHiddenBackdrop = !skyCameraVisible" in text
          and "!runtimeInputs.passComposite->inspectionImage" in text,
          label + " leaves inspection images and other outputs alone")

# The hidden flag gates only the sky draw. Lighting still reads the same environment.
check(opengl.count("skyCameraVisible") == 3 + 1, "OpenGL uses the hidden flag only for the backdrop")
check(sokol.count("skyCameraVisible") == 3 + 1, "Metal uses the hidden flag only for the backdrop")
check("sky=environmentimage::uniforms(lightAt(i));" in opengl, "OpenGL still binds the image for lighting")
check("uniforms.sky=environmentimage::uniforms(light);" in sokol, "Metal still binds the image for lighting")

check("7.0/255" not in cases, "the shared native case no longer accepts the stage colour")
check("const std::array<double,3> black {0,0,0};" in cases, "the shared native case requires black")

if failures:
    print("\n".join("FAIL: " + failure for failure in failures))
    sys.exit(1)
print("camera hidden backdrop source contract: passed")
