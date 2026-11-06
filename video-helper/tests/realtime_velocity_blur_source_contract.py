#!/usr/bin/env python3
"""Pins the OpenGL and Metal velocity motion blur and lens passes that need a GPU to run.

Raster, Realtime, Raster or Realtime with gratings, closed glass and rough glass
blur a Color image along its Motion over the Render node's shutter, then apply
the camera's depth of field, and a shutter keeps the previous frame's motion for
every engine. Metal runs the same two passes at the end of every Color frame.
"""
from pathlib import Path
import sys


root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
metal = (root / "src/gpu_backend/backend_sokol.mm").read_text()
backend = (root / "src/gpu_backend/backend.h").read_text()
gl = (root / "src/gpu_backend/backend_opengl.cpp").read_text()
shader = (root / "src/gpu_backend/fixture_realtime_opengl.h").read_text()
plan = (root / "src/imported_scene_visual_plan_execution.h").read_text()

uses = backend[backend.index("inline bool sceneUsesMotionPass"):]
uses = uses[:uses.index("}")]
assert "inputs.motionBlur.active()" in uses, "a shutter keeps the motion history"
assert "visualimportedscenerender::ShutterInterval motionBlur {};" in backend
assert "request.runtimeInputs.motionBlur = visualimportedscenerender::shutterInterval(" in plan

assert "inline const char* velocityBlurFragmentShader()" in shader
blur = shader[shader.index("inline const char* velocityBlurFragmentShader()"):]
assert "gl_FragCoord.xy - motion * t" in blur, "taps follow the motion back over the shutter"
assert "mix(uShutter.x, uShutter.y" in blur

assert "glBindTexture (GL_TEXTURE_2D, frame->rawTextures[5]);" in gl, "the blur reads the Motion attachment"
assert "runtimeInputs.motionBlur.open, runtimeInputs.motionBlur.close" in gl
lens = gl[gl.index("const auto lensAndShutter = [&]() -> bool {"):]
lens = lens[:lens.index("};")]
assert lens.index("velocityMotionBlur()") < lens.index("realtimeDepthOfField()"), "blur, then the lens"
assert "lensshutter::colorPasses (colorImage, runtimeInputs.motionBlur, camera)" in lens
# Realtime, Realtime with gratings, Raster with gratings, closed glass, rough glass
# and Raster each run both passes.
assert gl.count("if (!lensAndShutter())") == 5 and gl.count("if (!lensAndShutter()) return fail(result.error);") == 3
assert gl.count("if (!realtimeGratingPasses && !lensAndShutter()) return fail(result.error);") == 1
for marker in ("// Raster: the same lens and shutter passes over its single draw.",
               "// Raster with gratings: the lens and shutter passes over the accumulated radiance.",
               "// Closed glass: the lens and shutter passes over the finished bands.",
               "// Rough glass: the lens and shutter passes over the published tiles."):
    assert gl[gl.index(marker):].index("lensAndShutter()") < 200, marker
assert "lensshutter::bokehParameters (camera, width, height)" in gl

# Metal: no refusal, the passes run on the finished Color with the focus resolved.
assert "no motion blur pass yet" not in metal and "no depth of field pass yet" not in metal
assert "resolveSceneCameraFocus (*scene, evaluatedCamera);" in metal
call = metal.index("if (!applyMetalLensAndShutter(*frame, evaluatedCamera")
assert call < metal.index("result.rendered = true;", call) < call + 800
apply = metal[metal.index("static bool applyMetalLensAndShutter("):]
apply = apply[:apply.index("\n}\n")]
assert apply.index("run(1,frame.rawTextureViews[5]") < apply.index("run(2,frame.depthTextureView"), "blur, then the lens"
assert "arbitgpu::lensshutter::bokehParameters(camera,frame.width_,frame.height_)" in apply
print("velocity blur and lens source contract holds")
