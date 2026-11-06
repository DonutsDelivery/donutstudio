#!/usr/bin/env python3
"""Pins how OpenGL and Metal tell the medium marches a light is hidden from volume scatter.

OpenGL adds 8 to the light's kind texel and reads the kind with the flag masked
off; Metal sets the light's lightKinds y. The world fog, material media and the
Realtime Depth Fog light shafts all skip such a light.
"""
from pathlib import Path
import sys


root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
gl = (root / "src/gpu_backend/backend_opengl.cpp").read_text()
realtime = (root / "src/gpu_backend/fixture_realtime_opengl.h").read_text()
metal = (root / "src/gpu_backend/backend_sokol.mm").read_text()

assert "#define fixtureLightKind(i) (int(sceneTableRead(4 * (i) + 3).z) & 7)" in gl, "kind reads mask the flag"
assert "#define fixtureLightInVolumes(i) (int(sceneTableRead(4 * (i) + 3).z) < 8)" in gl
assert "sceneRayVisibility::VolumeScatter) == 0) record[3][2] += 8.0f;" in gl, "the table carries the flag"
assert "if (kind == 2 || !fixtureLightInVolumes(lightIndex)) continue;" in realtime, "OpenGL Depth Fog shafts"

assert "uniforms.lightKinds[lightIndex][1] =" in metal
assert "sceneRayVisibility::VolumeScatter) != 0 ? 0u : 1u;" in metal, "Metal uniforms carry the flag"
assert "if (kind == 2u || u.lightKinds[lightIndex].y != 0u) continue;" in metal, "Metal Depth Fog shafts"
print("volume light visibility source contract holds")
