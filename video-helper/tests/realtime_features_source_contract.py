#!/usr/bin/env python3
"""The Realtime program's four approximations, read from their sources.

Each feature is a shader function and the site that uses it, so a feature that
loses its wiring fails here. None of them may walk the glass BVH.
"""

from pathlib import Path
import re
import sys

root = Path(__file__).resolve().parents[2]
shader = (root / "video-helper/src/gpu_backend/fixture_realtime_opengl.h").read_text()
backend = (root / "video-helper/src/gpu_backend/backend_opengl.cpp").read_text()
renderer = (root / "video-helper/src/fixture_scene_renderer.cpp").read_text()
header = (root / "video-helper/src/fixture_scene_renderer.h").read_text()

failures = []


def need(text, snippet, what):
    if snippet not in text:
        failures.append(f"missing: {what}: {snippet}")


# Soft shadows: every light kind goes through several jittered rays, and the sky
# through a hemisphere of them.
need(shader, "float realtimeSoftVisibility(", "the soft shadow function")
need(backend, "realtimeSoftVisibility(vWorldPosition,vWorldNormal,", "direct lights use soft shadows")
need(shader, "float realtimeSkyVisibility(", "the sky shadow function")
need(backend, "skyBase *= realtimeSkyVisibility(", "the sky light uses soft shadows")

# Object reflections: a screen-space march over a snapshot, blended over the base.
need(shader, "bool realtimeTrace(", "the screen-space march")
need(shader, "vec4 realtimeReflection(", "the reflection term")
need(backend, "outColor = realtimeReflection(", "the reflection phase writes it")
need(backend, "gl.BlendFuncSeparate (GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);",
     "reflections blend over the opaque image")

# Refraction with fringes: one refracted direction per wavelength.
need(shader, "vec3 realtimeRefraction(", "the refraction term")
for wavelength in ("0.6563", "0.5461", "0.4358"):
    need(shader, wavelength, "the fringe wavelength")
need(shader, "uGlassTransport.y", "the material's dispersion")
need(backend, "outColor.rgb = realtimeGlass(", "the glass phase writes it")
need(backend, "realtimeSnapshot();", "each glass draw sees what is behind it")

# Fog: an in-scatter march toward the lights lights the Depth Fog pass.
need(shader, "vec3 realtimeFogLight(", "the fog march")
need(backend, "gRealtimeFog = realtimeFogLight(vWorldPosition);", "main lights the fog")
need(backend, "fog *= gRealtimeFog;", "the fog pass uses the light")

# The program: its own feature bit, no glass walk, one route for Preview and Export.
need(backend, "kFixtureFragmentRealtime = 1u << 6", "the feature bit")
need(backend, '"#define FIXTURE_REALTIME 1\\n"', "the program define")
need(backend, "realtimeengine::glslShader()", "the program text")
need(backend, "realtimeengine::active()", "preparation sees the engine")
need(backend, "result.stats.realtimeProgramApplied = realtimeFrame;", "the telemetry is truthful")
need(backend, "else if (realtimePasses)", "the phased draw")
need(renderer, "realtimeengine::Scope", "the renderer opens the scope")
need(renderer, "cache_.realtime == realtimeEngine", "an engine change prepares again")
need(header, "bool realtime = false;", "the prepared cache knows its engine")
if re.search(r"glassWalk|walkGlass|smoothglass::", shader):
    failures.append("the Realtime shader must not use the glass walk")
if "previewUse" in shader:
    failures.append("the Realtime shader must not branch on Preview")
start = backend.find("const auto realtimeReflectionAndGlass = [&]")
end = backend.find("if (materialKind == NativeFixtureMaterialKind::DiffractionReflective)", start)
plain = backend.find("else if (realtimePasses)")
if start < 0 or end < 0 or plain < 0:
    failures.append("the phased draw is missing")
elif "previewUse" in backend[start:end] or "previewUse" in backend[plain:plain + 300]:
    failures.append("the phased draw must not branch on Preview")

# Diffraction: a grating frame keeps the raster order sum, whose eight spectral bands
# give each order its own wavelength colors, and joins the phased draw around it.
need(backend, "(features & ~static_cast<unsigned> (kFixtureFragmentSmoothGlass)) | kFixtureFragmentRealtime",
     "Realtime keeps the diffraction program feature")
need(backend, "realtimePasses = realtimePhased && !diffractionFrame", "a plain frame takes the phased branch")
need(backend, "realtimeGratingPasses = realtimePhased && mixedMaterials", "a mixed grating frame takes the phases")
grating = backend.find("if (materialKind == NativeFixtureMaterialKind::DiffractionReflective)\n        {\n            // Probe")
scene = backend.find('validateDiffractionDraw(0, "scene-complete")', grating)
branch = backend[grating:scene] if 0 <= grating < scene else ""
for snippet, what in (
    ("realtimePhase = realtimeGratingPasses ? 0 : -1;", "the surfaces under the grating draw as phase 0"),
    ("!realtimeReflectionAndGlass()", "reflections and glass follow the grating"),
):
    if snippet not in branch:
        failures.append("the grating branch needs " + what)
if branch.find("drawSurfaces(1);") > branch.find("!realtimeReflectionAndGlass()"):
    failures.append("blended surfaces must reach the snapshot that reflections and glass read")
need(backend, "frame->linearColor || diffractionFrame ? GL_RGBA16F", "the snapshot holds the float accumulation")
need(backend, "receiverTransport = false", "the reflection phase blends over the receiver transport")
# The dithered gather is the Realtime cost control. Its size comes from the table.
need(backend, "int(sqrt(uDiffractionSampleControl.y))", "the receiver gathers a stride sized by the scene")
need(backend, "int(sqrt(uDiffractionSampleControl.x))", "the grating gathers a stride sized by the scene")
need(backend, "bands[b]+=realtimeStride*weight", "the stride keeps the receiver's energy")
need(backend, "bands[b] += realtimeStride*weight", "the stride keeps the grating's energy")
# Caustics stay Path Trace only.
need(renderer, "const bool realtimeDroppedCaustic = realtimeEngine && sceneInputs.glassCaustic.has_value();",
     "Realtime still names the dropped caustic")
need(renderer, "sceneInputs.glassCaustic.reset();", "Realtime still draws no caustic")

if failures:
    print("\n".join(failures))
    sys.exit(1)
print("realtime features source contract: ok")
