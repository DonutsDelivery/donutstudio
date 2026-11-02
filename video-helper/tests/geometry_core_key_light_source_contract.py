#!/usr/bin/env python3
# Contract for the fixed Geometry Core key light. Starter scenes carry no light input port, so
# this single directional record is their only direct light. Its direction must stay toward the
# upper left and the viewer (toLight z is positive), so every camera-facing starter surface
# receives a positive NdotL whatever the starter's tilt and displaced slopes shade. The previous
# raking direction left most front faces unlit: a 1920x1080 spectral composite held 4 unique
# colors, and zeroing the resting glow left the interior at 2 levels with no response at all.
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
source = (root / "src/geometry_core_backend.cpp").read_text(encoding="utf-8")

errors = []


def require(needle, message):
    if needle not in source:
        errors.append(message)


require("scene->lights[0].transform.rotation = {-0.2064f, -0.1548f, 0.0f, 0.9661f};",
        "the fixed Geometry Core light must keep the camera-facing key rotation")
require("scene->lights[0].intensity = 1.5f;",
        "the fixed Geometry Core key light must keep intensity 1.5")
require("scene->lights[0].kind = SceneLightKind::Directional;",
        "the fixed Geometry Core light must stay directional")
require("scene->lightCount = 1;",
        "the fixed Geometry Core light must stay the scene's single light")
if "does not reach most front faces" in source:
    errors.append("the stale unlit-front-face claim must not return")

if errors:
    for line in errors:
        print("FAIL:", line)
    raise SystemExit(1)
print("geometry_core_key_light_source_contract: PASS")
