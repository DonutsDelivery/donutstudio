#!/usr/bin/env python3
# Contract for the splat receiver-surface raster: glass triangles never raster
# into the surface the gather reads, because the raster answers for a
# glass-penetrating camera ray - the same view splatCpuQuery resolves - and a
# glass texel can only overwrite the receiver behind it under depth test while
# causticSplatSurface rejects it anyway. That overwrite turned dome-covered
# water texels into zeros the reference still counted as radiance.
from pathlib import Path
import sys

root = Path(sys.argv[1])
backend = (root / "src/gpu_backend/backend_opengl.cpp").read_text()
caustics = (root / "src/gpu_backend/fixture_glass_caustics.h").read_text()
opengl = (root / "src/gpu_backend/fixture_glass_caustics_opengl.h").read_text()
errors = []

if "if(scheduled.triangles[t].glass)continue;" not in backend:
    errors.append("the receiver-surface raster must skip glass triangles")
skip_at = backend.find("if(scheduled.triangles[t].glass)continue;")
if skip_at < 0:
    errors.append("glass raster skip missing")
else:
    window = backend[max(0, skip_at - 900):skip_at]
    if 'location("uTriangle")' not in backend[skip_at:skip_at + 400]:
        errors.append("the receiver draws must stay behind the glass skip")
    if "glass-penetrating" not in window:
        errors.append("the skip must document its glass-penetrating contract")
if "triangle.glass=glass" not in caustics:
    errors.append("frame triangles must carry the glass flag the raster reads")
if "shape.z!=0.0||shape.w!=1.0" not in opengl:
    errors.append("the surface gate must keep rejecting non-receivers")

if errors:
    print("fixture splat receiver raster contract: " + "; ".join(errors), file=sys.stderr)
    raise SystemExit(1)
print("fixture splat receiver raster contract: PASS")
