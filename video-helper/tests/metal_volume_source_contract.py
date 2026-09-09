#!/usr/bin/env python3
from pathlib import Path

root = Path(__file__).resolve().parents[1]
cmake = (root / "CMakeLists.txt").read_text()
header = (root / "src/volume_native_renderer.h").read_text()
wire = (root.parent / "shared/VisualVolumeData.h").read_text()
metal = (root / "src/volume_native_renderer_metal.mm").read_text()
opengl = (root / "src/volume_native_renderer_opengl.cpp").read_text()
test = (root / "tests/metal_volume_backend_tests.mm").read_text()

assert "if(APPLE AND ARBIT_WITH_METAL)\n    target_sources(arbit-video-helper PRIVATE src/volume_native_renderer_metal.mm)" in cmake
assert "elseif(GLFW3_FOUND AND OpenGL_FOUND AND NOT APPLE)\n    target_sources(arbit-video-helper PRIVATE src/volume_native_renderer_opengl.cpp)" in cmake
assert "src/volume_native_renderer_stub.cpp" in cmake
assert "arbit-metal-volume-backend-tests" in cmake
assert "-Wall -Wextra -Werror" in cmake
assert 'LABELS "native-volume-runtime;physical-apple-silicon"' in cmake

assert "kAllowsCpuProductionFallback" in wire
assert "deliberately no CPU implementation" in header
assert "SG_IMAGETYPE_3D" in metal
assert "texture3d<float> volume" in metal
assert "sg_make_image (&volumeDesc)" in metal
assert "request.volume->bytes().data()" in metal
assert "sg_begin_pass (&pass)" in metal
assert "sg_draw (0, 3, 1)" in metal
assert "sg_commit()" in metal
assert "result.frame = std::move (frame)" in metal
assert "sg_destroy_image (volumeImage)" in metal
assert "sg_destroy_image (colorImage)" in metal
assert "sparseBrickUpload = false" in metal
assert "kMaximumVolumeBytes = 512u * 1024u * 1024u" in metal
assert "kMaximumTexture3DExtent = 2048" in metal
assert "kMaximumRenderExtent = 4096" in metal
assert "expectedBytes != request.volume->bytes().size()" in metal
assert "Metal native volume request exceeds backend limits" in metal

# Preserve the established OpenGL limits and strict GPU path as an explicit
# regression boundary while adding the separate Apple implementation.
assert "kMaximumVolumeBytes = 512u * 1024u * 1024u" in opengl
assert "kMaximumRenderExtent = 4096" in opengl
assert "gl.TexImage3D" in opengl
assert "glDrawArrays (GL_TRIANGLES, 0, 3)" in opengl
assert "sparseBrickUpload = false" in opengl

assert "sg_mtl_query_image_info" in test
assert "previewPixels != exportPixels" in test
assert "renderer lacks sparse brick upload capability" in test
assert "imageIsValid" in test
assert "return 77" in test

print("strict Metal volume source contract passed")
