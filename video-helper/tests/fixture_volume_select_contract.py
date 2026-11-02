#!/usr/bin/env python3
# Contract for the caustic-volume selection record. The lighthouse row's cold
# caustic-volume fixture link can outlive its pre-seek window, and the only
# frames that can start the link earlier are the ones that did not select a
# volume. This line records why each frame passed: no Glass Caustics in the
# plan, no admitted material, no free second slot, or a camera-only selection.
from pathlib import Path
import sys

root = Path(sys.argv[1])
renderer = (root / "src/fixture_scene_renderer.cpp").read_text()
errors = []

if '"volume-select glass="' not in renderer:
    errors.append("missing volume-select record line")
for field in (" material=", " slot=", " selected=", " cameraOnly="):
    if field not in renderer:
        errors.append("volume-select record missing field " + field.strip())
# The record prints only when its text changes, so a settling row cannot flood
# the run log, and it reads the same needsSlot predicate the selection uses.
if "lastVolumeSelect != volumeSelectText" not in renderer:
    errors.append("volume-select record must print only on state change")
if renderer.count("causticVolumeNeedsSlot") < 2:
    errors.append("volume-select record must read the selection's own needsSlot predicate")

if errors:
    print("fixture volume-select contract: " + "; ".join(errors), file=sys.stderr)
    raise SystemExit(1)
print("fixture volume-select contract: PASS")
