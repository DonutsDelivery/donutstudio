#!/usr/bin/env python3
from pathlib import Path
import sys

source = Path(__file__).resolve().parents[1] / "src/gpu_backend/backend_sokol.mm"
text = source.read_text(encoding="utf-8")
required = {
    "back-face culling": "winding == 0 ? SG_CULLMODE_NONE : SG_CULLMODE_BACK",
    "ordinary winding": "winding == 2 ? SG_FACEWINDING_CW",
    "mirrored winding": "metalFixtureDeterminant3x3 (objectMatrix) < 0.0f ? 2u : 1u",
    "SDR/HDR pipeline offset": "!diffractionDraw && runtimeInputs.linearColor\n                ? resources->pipelines.size() / 2u : 0u",
    "diffraction resolve picks HDR": "diffractionResolvePipelines[runtimeInputs.linearColor ? 1u : 0u]",
    "SDR/HDR pipeline selection": "resources->pipelines[colorPipelineOffset + (realtimePhase == 2 ? 0u : blend) * 3u + winding]",
    "SDR/HDR reflection pipeline selection": "realtimeReflectionPipelines[(colorPipelineOffset != 0 ? 3u : 0u) + winding]",
    "SDR/HDR diffraction winding": "resources->pipelines[colorPipelineOffset + winding]",
}
missing = [name for name, token in required.items() if token not in text]
if missing:
    print("FAIL: Metal fixture scene culling contract missing " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(1)
print("PASS: Metal fixture scene uses back-face culling with mirrored-transform winding")
