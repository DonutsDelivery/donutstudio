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
# GL and Metal pick the phases through one helper (dc4d81420b): a plain frame
# takes the surface phases, a grating frame with surfaces the phases around it.
need(shader, "if (! grating) return FramePhases::kSurfaces;", "a plain frame takes the phased branch")
need(shader, "return gratingWithSurfaces ? FramePhases::kAroundGrating : FramePhases::kSingle;",
     "a mixed grating frame takes the phases")
need(backend, "arbitgpu::realtimeengine::framePhases(realtimePhased, diffractionFrame, mixedMaterials)",
     "the GL frame picks its phases from the grating and mixed materials")
need(backend, "realtimePasses = realtimeFramePhases == arbitgpu::realtimeengine::FramePhases::kSurfaces",
     "a plain frame takes the phased branch")
need(backend, "realtimeGratingPasses = realtimeFramePhases == arbitgpu::realtimeengine::FramePhases::kAroundGrating",
     "a mixed grating frame takes the phases")
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
# The orb's Realtime refusal (6 Nov): the whole-output probe read the base Color
# before any draw and found an unlisted NaN payload a pooled or fresh texture
# carried in. A diffraction frame clears its base Color when it binds its targets,
# before the probe at the start of the grating branch can read it.
setup = backend.find("if (diffractionFrame && frame->texture_ != 0)")
clear = backend.find("gl.ClearBufferfv (GL_COLOR, 0, transparent);", setup)
reattach = backend.find("GL_TEXTURE_2D, frame->accumulationTexture_, 0);", clear)
probe = backend.find('validateDiffractionDraw(frame->texture_, "before-draws@texture")')
if not (0 <= setup < clear < reattach) or clear - setup > 600:
    failures.append("a diffraction frame must clear its base Color, then attach its accumulation again")
elif not (0 <= reattach < probe):
    failures.append("the base Color clear must come before the before-draws probe")

# Diffraction under the environment: Realtime reads the sky, HDR image or flat
# environment where each order's light comes from, on GL and Metal alike.
need(backend, "void realtimeDiffractionSky(", "the GL environment-through-orders term")
need(backend, "if (uDiffractionPathIndex == 0 && !reused) realtimeDiffractionSky(vUv,tangent,bitangent,normal,outgoing,bands);",
     "the GL transport adds the environment term once")
need(backend, "lit += realtimeDiffractionSkyRgb(tangent, bitangent, normal, outgoing);",
     "the GL lit grating adds the environment term")
# The orders also reach the scene's glowing objects (realtime_diffraction_emitters.h):
# a Realtime grating frame appends their triangles to the scene table, and an
# environment-path grating adds the sky and emitters through its orders.
need(backend, "bool realtimeDiffractionEmitter(vec3 position, vec3 normal, vec3 direction, out vec3 radiance) {",
     "the GL order ray's emitter cast")
need(backend, "if (realtimeFrame && diffractionFrame)", "a Realtime grating frame lists its emitters")
# The environment-path grating takes the scene's light through its orders alone, as
# Path Trace does: Raster's analytic environment stand-in on top lit it twice. The
# scene's ambient colour is environment radiance, as Path Trace adds it.
need(backend, "if (uDiffractionEnvironment.x == 2.0 && uDiffractionPathKindAndBounce.x != 1.0)\n"
     "        return realtimeDiffractionSkyRgb(tangent, bitangent, normal, outgoing);",
     "the environment-path grating takes the sky and emitters through its orders")
if "return integrateDiffractionEnvironment(outgoing, tangent, bitangent, normal)\n            + realtimeDiffractionSkyRgb(" in backend:
    failures.append("Realtime must not add Raster's environment stand-in to the scene's light through the orders")
need(backend, "vec3 flatSky = uAmbient;", "the orders read the scene's ambient as environment")
need(backend, "&& (materialKind!=NativeFixtureMaterialKind::DiffractionReflective || realtimeFrame)",
     "a Realtime grating frame draws its sky")
# Each band is read across its wavelength interval, and the analytic sun is integrated
# over each order's arc, so lit orders are neither flat wedges nor missed by a small sun.
need(backend, "const int kRealtimeDiffractionSubbands = 8;", "the GL band reads its whole wavelength interval")
need(backend, "arriving += sunVisible * realtimeDiffractionSunArc(edge, next, sigma);", "the GL sun is integrated over the order's arc")
need(backend, "vec3 origin = vWorldPosition + side * 1.0e-4 * max(1.0, uProjection.w - uProjection.z);",
     "the GL order rays start off the grating")
need(backend, "|| any(isnan(vNoteAppearance)) || any(isinf(vNoteAppearance)) ? 16u : 17u);",
     "the GL grating output probes name a bad base apart from bad arithmetic")
# Each sub-wavelength lands on the band centres around it, so a saturated order keeps
# its own hue instead of one of eight band hues.
need(backend, "realtimeDiffractionDeposit(b, wavelength, arriving,", "the GL order light lands by its own wavelength")
need((root / "video-helper/src/gpu_backend/realtime_diffraction_emitters.h").read_text(),
     "inline TentBands tentBands (", "the host form of the band split")
# Each grating draw names its route once per change, and the banded readback names its pass.
need(backend, "[gl] diffraction draw prog=%u features=0x%x realtimeDefine=%d", "the grating route log")
need(backend, "\" bands of \" + std::to_string(diffractionBands.bandHeight) + \" rows, \"", "a banded frame is read back once, naming its bands")
# Realtime bands are budgeted by the rows the shader gathers, now that a gathered row
# reads tables and casts no ray; Raster keeps every row.
need(backend, "? nativeFixtureRealtimeDiffractionGatherRows(std::size_t(diffractionTransportFrame.sampleControl[0]))",
     "Realtime bands are budgeted by the rows a pixel gathers")
need(backend, "if (!nativeFixtureDiffractionPixelQueries(receiverRowsGathered, gratingRowsGathered, diffractionPixelQueries))",
     "the band budget reads the gathered rows")
need(backend, "uniform int uDiffractionReceiverRow;", "a receiver pixel reads its row from the host")
need(backend, "bool reused = realtimeGatherMode()==2 && realtimeGatherUpsample(upsampled);",
     "a full-resolution pixel reuses the reduced-resolution gather term")
need(backend, "for (realtimeGatherRound = gatherScale > 1 ? 1 : 0; ; ++realtimeGatherRound)",
     "Realtime draws the gather pass before the full pass")
# The GL BSDF's orders share what the metal reflects, as Path Trace's do, instead of
# refusing the frame where the scalar model sums above 1.
need(backend, "if (efficiencySum > 1.0) radiance /= efficiencySum;", "the GL orders share the reflection")
# A pressed disc's orders and pit scatter, and the scene's directional lights, reach the
# Realtime orders; a frame without lights keeps the sky through them.
need(backend, "? realtimePressedDiscEfficiency(primary, uDiffractionMicrostructure.x, phase)", "the GL pressed-disc orders")
need(backend, "realtimePitScatter(b, centre, geometry,", "the GL pit scatter")
# Fans read the exact share of each arc piece an emitter covers, and the pit-scatter lobe
# is integrated on fixed nodes, so the disc's fans are continuous and free of speckle.
need(backend, "arriving = realtimeDiffractionEmitterChord(origin, side, edge, next, sigma, covered)", "the GL orders read emitter shares through the lobe")
need(backend, "const int kRealtimePitScatterPieces = 12;", "the GL pit scatter on fixed nodes")
# The grating's vertices carry the smooth groove direction and Realtime's frame follows it,
# so a polar cap's fans do not step wedge by wedge.
need(backend, "realtimediffraction::smoothGrooveTangents (positions, uvs, local)", "the grating uploads smooth groove directions")
need(backend, "if (dot(grooveTangent, tangent) > 0.7) tangent = grooveTangent;", "Realtime's grating frame follows them")
# Realtime's receiver gathers bound each sample as a coaxial disc of its area and skip a
# segment the exact query cannot resolve, so a contact neither overflows nor refuses.
need(backend, "/(pi*distance*distance+diffractionSample(r,0).w);", "the grating gather's disc form factor")
need(backend, "*max(outgoing.z,0.0)*pi/(pi*distance*distance+diffractionSample(index,0).w)", "the receiver gather's disc form factor")
need(backend, "arriving += realtimeDiffractionLightArcs(edge, next, lightVisible, sigma);", "the GL directional lights through the orders")
need(backend, "if (uLightCount == 0) return realtimeDiffractionSkyRgb(", "no light still leaves the sky")
need((root / "video-helper/src/diffraction_material_execution.h").read_text(),
     "description.pitStatistics == PitStatistics::PressedDisc ? 1.0f : 0.0f", "the pressed-disc flag reaches the GPU")
# Probe labels survive the RGBA16F diffraction target in alpha.
need(backend, "vec4 fixtureProbeColor(uint reason) {", "the GL probe colour carries its reason in alpha")
for reason in ("8u", "9u", "12u", "13u", "14u", "15u", "18u"):
    need(backend, "fixtureProbeColor(" + reason + ")", "the GL probe " + reason + " names its pass")
need((root / "video-helper/src/gpu_backend/fixture_diffraction_transport.h").read_text(),
     "inline std::uint32_t pixelDiagnosticReason(float value, float alpha) noexcept",
     "the readback names a payload lost to a half-float target by its alpha")
metal = (root / "video-helper/src/gpu_backend/backend_sokol.mm").read_text()
need(metal, "float3 realtimeDiffractionSky(VertexOut in, constant Uniforms& u,", "the Metal environment-through-orders term")
need(metal, "? realtimeDiffractionSky(in, u, environmentTexture, opaqueTexture) : float3(0.0f))) * scatterWeight",
     "the Metal grating adds the environment term")

# Caustics stay Path Trace only.
# Solid viewport shading draws through the Realtime engine but is not a Realtime
# render, so it names no dropped caustic (3889dd5632).
need(renderer, "const bool realtimeDroppedCaustic = realtimeEngine && ! viewportSolid && sceneInputs.glassCaustic.has_value();",
     "Realtime still names the dropped caustic")
need(renderer, "sceneInputs.glassCaustic.reset();", "Realtime still draws no caustic")

if failures:
    print("\n".join(failures))
    sys.exit(1)
print("realtime features source contract: ok")
