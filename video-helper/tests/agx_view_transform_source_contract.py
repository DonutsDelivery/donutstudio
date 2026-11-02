#!/usr/bin/env python3
"""Pins the GPU AgX to shared/AgxViewTransform.h and the display route to one conversion.

The numeric tests run the scalar header. This file makes their figures hold for the
GLSL and Metal shaders, which no headless test executes on every platform.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
header = (ROOT / "shared/AgxViewTransform.h").read_text()
gl = (ROOT / "video-helper/src/color_transform_gl.cpp").read_text()
metal = (ROOT / "video-helper/src/gpu_backend/backend_sokol.mm").read_text()
renderer_cpp = (ROOT / "video-helper/src/renderer.cpp").read_text()
renderer_h = (ROOT / "video-helper/src/renderer.h").read_text()

failures = []


def require(condition, message):
    if not condition:
        failures.append(message)


NUMBER = re.compile(r"-?\d+\.\d+(?:e-?\d+)?")


def numbers(text):
    return [float(value) for value in NUMBER.findall(text)]


def header_matrix(name):
    body = re.search(r"kMatrixName \{\{(.*?)\}\};".replace("kMatrixName", name), header, re.S).group(1)
    return numbers(body)


def transpose(values):
    return [values[column * 3 + row] for row in range(3) for column in range(3)]


def shader_function(source, start):
    begin = source.index(start)
    return source[begin:source.index("\n}\n", begin)]


for label, source, start in (
    ("GLSL", gl, "vec3 agxViewTransform(vec3 radiance)"),
    ("Metal", metal, "static float3 agxViewTransform(float3 radiance)"),
):
    body = shader_function(source, start)
    values = numbers(body)
    # Shader matrices are columns, the header's are rows.
    expected = (transpose(header_matrix("kLinearSrgbToRec2020"))
                + transpose(header_matrix("kRec2020ToLinearSrgb"))
                + transpose(header_matrix("kInset"))
                + transpose(header_matrix("kOutset")))
    require(values[:36] == expected, f"{label} matrices differ from shared/AgxViewTransform.h")
    require(values[36:38] == [-12.47393, 4.026069], f"{label} EV range differs")
    require(1.0e-10 in values, f"{label} log floor differs")
    for coefficient in (15.5, 40.14, 31.96, 6.868, 0.4298, 0.1191, 0.00232):
        require(coefficient in values or -coefficient in values, f"{label} sigmoid lost {coefficient}")
    require(2.2 in values, f"{label} display power differs")

require("if (toneMap == 2) working = agxViewTransform(working);" in gl,
        "GLSL must run AgX on the working colour before the output encode")
require("if (p.rendering.x == 2) working = agxViewTransform(working);" in metal,
        "Metal must run AgX on the working colour before the output encode")
require(gl.index("agxViewTransform(working)") < gl.index("vec3 outputEncoded = encodeTransfer(outputLinear);"),
        "GLSL encodes after the transform")
require(metal.index("agxViewTransform(working)")
        < metal.index("float3 outputEncoded = encodeTransfer(outputLinear, p.options.y, p.luminance.z);"),
        "Metal encodes after the transform")

# One conversion per composite on each backend, reached only through SdrNativeImageLayers.
require(renderer_cpp.count("displayInputs.prepare(") == 1, "OpenGL composites through one display conversion")
require(metal.count("displayInputs.prepare(") == 1, "Metal composites through one display conversion")
require(renderer_cpp.count("nativeLinearImageToSrgb(") == 1 and metal.count("nativeLinearImageToSrgb(") == 1,
        "each backend admits the display transform in one place")
prepare = renderer_h[renderer_h.index("struct SdrNativeImageLayers"):]
require(prepare.index("if (hdrCapture) return true;")
        < prepare.index("convert(frame, source.nativeLinearViewTransform, error)"),
        "HDR capture returns before the view transform")
require("target.nativeLinearImage.reset();" in prepare, "a converted layer stops being linear, so it can't convert twice")
require("params.rendering[0] = static_cast<int>(transform.toneMap);"
        in metal[metal.index("leaseLinearImageForDisplay(\n        const std::shared_ptr"):metal.index("bool ensureOutputs (const arbitgl::GlFuncs* gl, int w, int h)")],
        "the Metal display conversion passes the view transform to its shader")

# The param recompiles the plan, so both immutable-parameter lists must name it.
for path in ("plugin/Source/VideoVisualGraph.h", "plugin/Source/PluginProcessorVideoGraph.cpp"):
    require('parameterId == "viewTransform"' in (ROOT / path).read_text(), f"{path} must recompile on viewTransform")

if failures:
    print("\n".join("FAIL: " + failure for failure in failures))
    sys.exit(1)
print("agx_view_transform_source_contract: pass")
