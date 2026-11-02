#!/usr/bin/env python3
"""Path Trace samples a material's frame texture on the GPU (R5).

The route tests link the stub upload, so they cannot run the OpenGL pass. This
pins what the pass must do and that nothing brings the frame to the CPU.
"""
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
upload = (root / "src/gpu_backend/cpu_scene_frame_opengl.cpp").read_text()
engine = (root / "src/pathtrace_fixture_engine.cpp").read_text()

assert "does not sample frame textures" not in engine, "the frame texture refusal is gone"
for call in ("glReadPixels", "glGetTexImage", "GetTextureImage", "glGetTextureSubImage", "MapBuffer"):
    assert call not in upload and call not in engine, f"no frame readback: {call}"

assert "texture(uFrame, uv)" in upload, "the fragment pass samples the frame"
assert "texelFetch(uLight, pixel, 0).rgb * tint * alpha" in upload, "the frame tints the traced light"
assert "1.0 - uvCoverage.y" in upload and "uFrameBottomFirst" in upload, "a bottom-first frame flips like the raster"
assert "srgbToLinear(imagery.rgb)" in upload and "uFrameLinear" in upload, "an sRGB frame decodes, a linear one does not"
assert "colorTextureViewHandle()" in upload, "the pass binds the frame's own GPU texture"
assert "drawFrameLayers (gl, layerProgram, textures[0], image, error)" in upload, "the pass draws into the uploaded colour"
assert "GL_ONE, GL_ONE, GL_ZERO, GL_ONE" in upload, "the layers add to colour and keep its alpha"

assert "bindFrameLayers (draw, setup, frames, error)" in engine, "the engine binds frames to objects"
assert "upload.frameLayers.push_back" in engine, "the engine hands every layer to the upload"

print("pathtrace frame layer source contract passed")
