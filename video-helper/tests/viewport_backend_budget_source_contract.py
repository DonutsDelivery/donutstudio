#!/usr/bin/env python3
from pathlib import Path
import sys

root = Path(sys.argv[1])
viewport = (root / "src/viewport.cpp").read_text()
metal = (root / "src/gpu_backend/backend_sokol.mm").read_text()

set_canvas = viewport[
    viewport.index("std::string Viewport::setCanvas ("):
    viewport.index("Viewport::CanvasExtent Viewport::canvasExtent()")
]
admission = set_canvas.index("makeVisualPlanExecutionSnapshot(")
accepted = set_canvas.index("if (! admitted)")
publish = set_canvas.index("publishVisualPlanExecutionSnapshotTelemetry(")
resize = set_canvas.index("canvasW_ = width", accepted)
assert admission < accepted < resize < publish
assert "publication->state.setTelemetryOwner(impl_->visualTelemetry)" in set_canvas
assert "plans = impl_->visualPlanPublication->plans" in set_canvas
assert "currentCaps.maxTextureSize != gpuCaps.maxTextureSize" in set_canvas
assert "currentCaps.deviceIdentity != gpuCaps.deviceIdentity" in set_canvas
assert "impl_->timelineGeneration != timelineGeneration" in set_canvas

renderer_admission = viewport[
    viewport.index("videowire::VisualBackendResourceLimits::Capabilities rendererCapabilities"):
    viewport.index("videowire::ViewportTelemetryOwner<>")
]
assert "renderer-owned OpenGL device identity query failed" in renderer_admission
assert "const auto canvas = canvasExtent()" in renderer_admission
assert "publicationGeneration, canvas.generation" in renderer_admission
assert "im.timelineGeneration, canvasGeneration_" in renderer_admission
assert renderer_admission.count("makeVisualPlanExecutionSnapshot(") == 1

metal_query = metal[
    metal.index("bool MetalFrameRenderer::queryResourceCapabilities ("):
    metal.index("void* MetalFrameRenderer::retainedDevice()")
]
assert "arbitgpu::sokolmetal::device()" in metal_query
assert "sg_query_limits().max_image_size_2d" in metal_query
assert "device.recommendedMaxWorkingSetSize" in metal_query
assert "device.registryID" in metal_query
assert "MTLCreateSystemDefaultDevice" not in metal_query

print("viewport backend budget source contract: PASS")
