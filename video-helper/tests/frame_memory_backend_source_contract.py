#!/usr/bin/env python3
"""Require backend byte admission before native attachment allocation."""

from pathlib import Path
import sys

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
checks = {
    root / "src/gpu_backend/backend_opengl.cpp": (
        "RenderPassOutputAdmission admitRenderPassOutputs",
        "admitRenderPassOutputFrameMemory",
        "glGenTextures (1, &texture)",
    ),
    root / "src/gpu_backend/backend_sokol.mm": (
        "RenderPassOutputAdmission admitRenderPassOutputs",
        "admitRenderPassOutputFrameMemory",
        "owned.image = sg_make_image",
    ),
}

for path, markers in checks.items():
    source = path.read_text(encoding="utf-8")
    method = source.find(markers[0])
    admission = source.find(markers[1], method)
    allocation = source.find(markers[2], method)
    assert method >= 0, f"missing render-pass backend method in {path}"
    assert admission > method, f"missing frame-memory admission in {path}"
    assert allocation > admission, f"native allocation precedes frame-memory admission in {path}"

print("frame-memory backend source contract: PASS")
