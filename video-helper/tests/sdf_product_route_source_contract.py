#!/usr/bin/env python3
from pathlib import Path
import sys


root = Path(sys.argv[1])
viewport = (root / "src/viewport.cpp").read_text()
exporter = (root / "src/exporter.cpp").read_text()
execution = (root / "src/sdf_visual_plan_execution.h").read_text()
renderer_header = (root / "src/renderer.h").read_text()
renderer = (root / "src/renderer.cpp").read_text()
metal = (root / "src/gpu_backend/backend_sokol.mm").read_text()
stub = (root / "src/gpu_backend/backend_stub.cpp").read_text()

# Preview and export both enter the same strict renderer seam.
assert viewport.count("prepareVisualSdfLayer(") == 1
assert exporter.count("prepareVisualSdfLayer(") == 1
assert "NativeSdfRenderUse::Preview" in viewport
assert "NativeSdfRenderUse::Export" in exporter

# The product layer keeps the backend identity and opaque native view. Metal
# views are never narrowed into an OpenGL texture name by the execution helper.
assert "layer.nativeTextureBackend = rendered.nativeFrame->backend();" in execution
assert "layer.nativeTextureView = rendered.nativeFrame->colorTextureViewHandle();" in execution
assert 'if (layer.nativeTextureBackend == "opengl")' in execution
assert "layer.texture = rendered.nativeFrame->colorTextureViewHandle();" not in execution
assert "std::string nativeTextureBackend;" in renderer_header
assert "std::uintptr_t nativeTextureView = 0;" in renderer_header

# The Metal compositor validates and samples the exact live sg_view. If native
# Metal consumption is rejected, FrameRenderer must not reinterpret the handle
# through its OpenGL fallback.
assert 'layer.nativeTextureBackend != "metal"' in metal
assert "sg_query_view_state (nativeView)" in metal
assert 'else if (layer.nativeTextureBackend == "metal")' in metal
assert "sourceView.id = static_cast<std::uint32_t> (layer.nativeTextureView);" in metal
assert 'requiresMetalTexture = layers[i].nativeTextureBackend == "metal";' in renderer
assert 'compositorBackend_ = "metal-rejected";' in renderer

# Unsupported builds remain explicit and allocate no CPU-rendered substitute.
assert stub.count("native GPU SDF execution is not compiled in") >= 2
assert "CPU" not in stub[stub.index("class StubSdfExecutionBackend"):]
