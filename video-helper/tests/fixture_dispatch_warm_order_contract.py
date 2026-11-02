#!/usr/bin/env python3
# Contract for the cold lighthouse window: the caustic dispatch warm must run the
# moment the admitted material's keys exist, ahead of the grating admission and
# the smooth-glass preparation the first publication spends its cold nine seconds
# on, so its compile overlaps them. The heavy sections must each stay named.
from pathlib import Path
import sys

root = Path(sys.argv[1])
seam = (root / "src/glb_native_render_seam.cpp").read_text()
errors = []

warm = seam.find("arbitgpu::prepareOpenGlGlassCausticPrograms")
grating = seam.find("admitDiffractionMaterialBinding")
smooth = seam.find("arbitgpu::smoothglass::prepare")
link = seam.find("backend_.startFixtureProgramLink")
publish = seam.find("GlbNativeRenderSeam::publishSurfacePrograms")
if publish < 0:
    errors.append("publishSurfacePrograms not found")
elif not (publish < grating < smooth < link < warm):
    errors.append(
        "the fixture link must precede the dispatch warm so its context "
        f"initializes while the pool is idle (publish={publish} grating={grating} "
        f"smooth={smooth} link={link} warm={warm})")

for needle in (
    "advance:imported-scene:seam-material-admit",
    "advance:imported-scene:seam-grating",
    "advance:imported-scene:seam-smoothglass",
    "advance:imported-scene:seam-fixture-link",
    "NativeFixturePrepareUse::Preview",
):
    if needle not in seam:
        errors.append("missing: " + needle)

# The fixture link keeps its post-grating position: its key needs the grating.
link = seam.find("backend_.startFixtureProgramLink")
if link < 0 or link < grating:
    errors.append("fixture program link must stay after the grating admission")

if errors:
    print("fixture dispatch warm order contract: " + "; ".join(errors), file=sys.stderr)
    raise SystemExit(1)
print("fixture dispatch warm order contract: PASS")
