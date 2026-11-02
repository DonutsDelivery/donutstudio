#!/usr/bin/env python3
"""A transmissive surface reflects the sky in Raster, on both GPU backends.

visual.material.frosted-glass drew no sphere on 31 October 2026. Opacity 0.35 and
transmission 0.92 leave 4% coverage, and the sky term was gated on zero transmission,
so only ambient reached the surface. An opaque surface must keep its gate, and the
transmitted light must leave the diffuse base.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
CASES = {
    "video-helper/src/gpu_backend/backend_opengl.cpp": (
        "if (transmission>0.0 || (uAlphaModeCutoff.x==0.0 && alpha>=1.0))",
        "environmentImageBase(vBaseColor*texel.rgb*(1.0-transmission),",
        "analyticSkyBase(vBaseColor*texel.rgb*(1.0-transmission),"),
    "video-helper/src/gpu_backend/backend_sokol.mm": (
        "if (transmission>0.0f || (u.materialKind.z==0u && alpha>=1.0f))",
        "environmentImageBase(in.baseColor*baseTexel.rgb*(1.0f-transmission),",
        "analyticSkyBase(in.baseColor*baseTexel.rgb*(1.0f-transmission),"),
}
for name, needles in CASES.items():
    source = (ROOT / name).read_text(encoding="utf-8")
    for needle in needles:
        if needle not in source:
            print(f"FAIL: {name} lacks {needle}")
            sys.exit(1)
print("transmissive sky reflection: PASS")
