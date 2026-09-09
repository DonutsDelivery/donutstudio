#!/usr/bin/env python3
from pathlib import Path
import sys

source = Path(__file__).resolve().parents[1] / "src/gpu_backend/backend_sokol.mm"
text = source.read_text(encoding="utf-8")
required = {
    "back-face culling": "winding == 0 ? SG_CULLMODE_NONE : SG_CULLMODE_BACK",
    "ordinary winding": "winding == 2 ? SG_FACEWINDING_CW",
    "mirrored winding": "metalFixtureDeterminant3x3 (objectMatrix) < 0.0f ? 2u : 1u",
    "pipeline selection": "resources->pipelines[blend * 3u + winding]",
}
missing = [name for name, token in required.items() if token not in text]
if missing:
    print("FAIL: Metal fixture scene culling contract missing " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(1)
print("PASS: Metal fixture scene uses back-face culling with mirrored-transform winding")
