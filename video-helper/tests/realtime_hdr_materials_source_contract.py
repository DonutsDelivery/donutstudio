#!/usr/bin/env python3
"""An HDR environment refuses no material.

Each lobe reads the sky through the same uSky branch whether the sky is analytic
or an image, so the image needs no material admission of its own.
"""

from pathlib import Path
import re
import sys

root = Path(__file__).resolve().parents[2]
gpu = root / "video-helper/src/gpu_backend"
environment = (gpu / "fixture_environment_image.h").read_text()
shader = (gpu / "fixture_realtime_opengl.h").read_text()
opengl = (gpu / "backend_opengl.cpp").read_text()
metal = (gpu / "backend_sokol.mm").read_text()

failures = []


def need(text, snippet, what):
    if snippet not in text:
        failures.append(f"missing: {what}: {snippet}")


def gone(text, snippet, what):
    if snippet in text:
        failures.append(f"still present: {what}: {snippet}")


# The refusals themselves, on both backends.
gone(environment, "HDR environment currently supports", "the material refusal")
gone(environment, "HDR environment currently requires", "the opaque scene refusal")
gone(opengl, "HDR environment requires opaque", "the OpenGL instance refusal")
gone(metal, "HDR environment requires opaque", "the Metal instance refusal")

# prepare() reads no material lobe. The secondary owner check names the material
# only to compare its identity.
body = environment[environment.index("inline bool prepare("):environment.index("inline analyticsky::Uniforms uniforms(")]
for lobe in ("transmissionIorClearcoat", "normalOpacity", "NativeFixtureMaterialKind", "alphaMode", ".opacity"):
    gone(body, lobe, "prepare() inspects a material lobe")

# The lobes read the image through the analytic branch.
need(opengl, "uSky[0].w>1.5 ? environmentImageRadiance(vWorldPosition,uSky[0],uSky[1])",
     "the backdrop that Realtime refraction snapshots draws the image")
need(shader, "uSky[0].w > 1.5", "Realtime glass reflection reads the image")
need(shader, "environmentImageBase(", "Realtime glass reflection uses the image GGX integral")
need(opengl, "(directBase + skyBase) * (1.0 - coat)", "clearcoat scales image and analytic sky alike")
for name, text in (("OpenGL", opengl), ("Metal", metal)):
    if not re.search(r"transmission\s*==\s*0\.0f?\s*&&.*alpha\s*>=\s*1\.0", text):
        failures.append(f"missing: {name} per-fragment sky guard for blended and transmissive fragments")

# Behavior lives in the pixel cases. This contract keeps them running.
fixture = (root / "video-helper/tests/opengl_fixture_scene_backend_tests.cpp").read_text()
need(fixture, "environmentmaterialfixture::run(", "the glass and clearcoat pixel cases run")
need(fixture, "Realtime grating keeps its orders under an HDR environment", "the diffraction pixel case runs")

if failures:
    print("\n".join(failures))
    sys.exit(1)
print("realtime hdr materials source contract ok")
