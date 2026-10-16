#!/usr/bin/env python3
"""Render-thread stack contract (macOS SIGBUS, 15 Oct).

The viewport thread crashed in ___chkstk_darwin under renderLoop, particle
body replay, ImportedGeometryExecution::evaluate, Metal deformation and
MetalFixtureSceneBackend::renderCaustic. evaluate held a ~330 KiB
Visual3DScene in a stack optional and renderCaustic held 60 KiB of uniform
copies and instance records, on a 512 KiB thread. This keeps the scenes and
records on the heap and the render threads on an explicit stack.
"""
from pathlib import Path
import re
import sys

src = Path(__file__).resolve().parents[1] / "src"
failures = []


def read(name):
    return (src / name).read_text(encoding="utf-8")


for name in ("imported_geometry_execution.h", "glb_geometry_core_adapter.h"):
    if re.search(r"adaptAnimatedGlbMeshToVisual3DScene\s*\(", read(name)):
        failures.append(f"{name} builds a stack Visual3DScene; use adaptAnimatedGlbMeshToVisual3DSceneHeap")

metal = read("gpu_backend/backend_sokol.mm")
start = metal.find("NativeFixtureSceneSubmission renderCaustic(")
end = metal.find("class MetalDeformationBackend", start)
caustic = metal[start:end] if start >= 0 and end > start else ""
if not caustic:
    failures.append("MetalFixtureSceneBackend::renderCaustic was not found")
else:
    for pattern, what in (
        (r"(?m)^\s*(?:const\s+)?FixtureUniforms\s+\w+\s*[{;=]", "a FixtureUniforms local"),
        (r"const\s+auto\s+\w+\s*=\s*uniforms\s*;", "a FixtureUniforms copy"),
        (r"std::array<MetalFixtureInstanceGpuRecord", "an instance record array"),
    ):
        if re.search(pattern, caustic):
            failures.append(f"renderCaustic keeps {what} on the stack; use MetalFixtureRenderScratch")
    # RoughDraw members live in std::vector storage, so they are not stack locals.
    if "std::make_unique<MetalFixtureRenderScratch>()" not in caustic:
        failures.append("renderCaustic no longer allocates MetalFixtureRenderScratch")

viewport_h = read("viewport.h")
if "videohelper::RenderThread thread_;" not in viewport_h:
    failures.append("Viewport render thread is not a RenderThread")
if "std::thread (" in read("viewport.cpp").replace("std::this_thread", ""):
    failures.append("viewport.cpp starts a std::thread with the platform default stack")
main = read("main.cpp")
for state in ("g_export", "g_renderCache"):
    if f"{state}.worker = videohelper::RenderThread (" not in main:
        failures.append(f"{state} worker does not start on a RenderThread")

if "kRenderThreadStackBytes = 8u * 1024u * 1024u" not in read("render_thread.h"):
    failures.append("RenderThread no longer requests an 8 MiB stack")

if failures:
    for failure in failures:
        print("FAIL: " + failure, file=sys.stderr)
    raise SystemExit(1)
print("PASS: render threads have explicit stacks and the caustic replay path keeps large scenes on the heap")
