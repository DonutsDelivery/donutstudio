#!/usr/bin/env python3
# Contract for the direct-light shadow ray in the OpenGL and Metal fixture scene shaders. Each
# direct light traces the opaque BVH from the receiver toward the light. A directional light
# uses an unbounded ray along its per-draw rotation. A point or spot light stops at the
# receiver's distance. The ray is composed into every fixture fragment program, and the
# host gates it through uOpaqueControl.w. The exact GPU oracle that covers this path lives in
# tests/support/opaque_visibility_cases.h, and this contract checks that it stays wired.
from pathlib import Path
import math
import random
import re
import struct
import sys

root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
backend = (root / "src/gpu_backend/backend_opengl.cpp").read_text(encoding="utf-8")
visibility = (root / "src/gpu_backend/fixture_opaque_visibility.h").read_text(encoding="utf-8")
oracle = (root / "tests/support/opaque_visibility_cases.h").read_text(encoding="utf-8")
gl_tests = (root / "tests/opengl_fixture_scene_backend_tests.cpp").read_text(encoding="utf-8")

errors = []


def collapse(text):
    return re.sub(r"\s+", " ", text)


metal = (root / "src/gpu_backend/backend_sokol.mm").read_text(encoding="utf-8")
flat = collapse(backend)
flat_metal = collapse(metal)
flat_visibility = collapse(visibility)
flat_oracle = collapse(oracle)


def require(text, needle, message):
    if needle not in text:
        errors.append(message)


# Light loop: the directional light is kind 0 with a per-draw rotation, so its ray direction
# follows the current scene table light rotation on every frame.
require(flat, "vec3 toLight = kind == 0 ? rotateQ(fixtureLightRotation(lightIndex), vec3(0.0, 0.0, 1.0)) : baseSurfaceUnit(delta);",
        "the light loop must derive a directional toLight from the scene table light rotation")

# The visibility term is gated by the opaque alpha and transmission conditions, then uses the
# unbounded ray for directional lights and the receiver distance for the others. The receiver
# normal is the interpolated vertex normal on both backends. The old screen-derivative normal
# was per 2x2 quad and evaluated inside the light loop, so near grazing light its sign could
# flip and put the ray origin inside the solid. That matches the stripes and black/lit
# checkerboard on the lighthouse boxes.
require(flat, "float visibility = uAlphaModeCutoff.x == 0.0 && transmission == 0.0 && alpha >= 1.0 "
              "? opaqueVisibility(vWorldPosition,vWorldNormal, "
              "toLight,kind == 0 ? 3.402823e38 : length(delta),uOpaqueControl) : 1.0;",
        "the direct visibility term must use the opaque ray with an unbounded limit for kind 0")
require(flat_metal, "float visibility = u.materialKind.z == 0u && transmission == 0.0f && alpha >= 1.0f "
                    "? opaqueVisibility(in.worldPosition,in.worldNormal, "
                    "toLight,kind == 0u ? 3.402823e38f : length(delta),u.opaqueControl,opaqueTexture) : 1.0f;",
        "the Metal direct visibility term must pass the interpolated world normal like OpenGL")
for name, text in (("OpenGL", flat), ("Metal", flat_metal)):
    if re.search(r"opaqueVisibility\([^;]*d[Ff]d[xy]", text):
        errors.append(name + " opaque visibility must not take a screen-derivative normal")

# The shadowed term multiplies only the direct base radiance. Ambient stays in `lighting`, so a
# blocked receiver keeps the ambient term.
require(flat, "directBase += radiance * baseSurfaceDirect(vBaseColor * texel.rgb, metallic, roughness, "
              "dielectric, transmission, normal, toLight, toView) * visibility;",
        "the shadow visibility must multiply the direct base term only")
require(flat, "vec3 lighting = uAmbient;", "ambient must start the lighting term outside the shadowed direct term")

# The ray shader is composed into every fixture program. Its feature gate is the host-side
# uOpaqueControl.w, not a compile-time branch, so plain programs keep one code path.
require(flat, "text += opaquevisibility::glslShader() + analyticsky::shader (false)",
        "fixtureFragmentShaderSource must compose the opaque visibility shader unconditionally")
if re.search(r"if \(\s*smoothGlass\s*\)\s*text \+= opaquevisibility", backend) or \
        re.search(r"if \(\s*diffraction\s*\)\s*text \+= opaquevisibility", backend):
    errors.append("the opaque visibility shader must not be gated by the smooth-glass or diffraction bit")

# Shader-side gate: a zero control.w returns full visibility, which is the path when no opaque
# frame was admitted.
require(flat_visibility, "float opaqueVisibility(SV3 position,SV3 geometricNormal,SV3 toLight,float limit,SV4 control ST_ARGS) { "
                         "if(control.w<0.5) return 1.0;",
        "opaqueVisibility must return full visibility when control.w is below 0.5")
# Receiver rule shared by the raster and glass callers: the normal is not flipped toward the
# light. A receiver turned away from it is unlit before any ray, and the origin moves along
# the normal by 1e-4 * max(1, max|p|).
require(flat_visibility, "float bias=max(1.0e-4,1.0e-4*max(max(abs(position.x),abs(position.y)),abs(position.z))); "
                         "SV3 n=normalize(geometricNormal);if(!(dot(n,direction)>0.0)) return 0.0; "
                         "SV3 origin=position+n*bias;",
        "opaqueVisibility must classify back-facing receivers unlit and use the 1e-4 relative bias")
if "if(dot(n,direction)<0.0) n=-n;" in flat_visibility:
    errors.append("opaqueVisibility must not flip the receiver normal toward the light")

# Host gate: the frame is built only for Color output with a direct light and a non-empty
# opaque mesh set. Vertex deformation disables it and reports the boundary. A disabled frame
# keeps control at zero, so the shader returns full visibility.
require(flat, "bool directVisibility = (!diffractionFrame || !materialProgram->objectPrograms.empty()) && "
              "runtimeInputs.imageOutput == renderpassoutput::Output::Color && "
              "!resources->opaqueVisibility.meshes.empty() && hasDirectLight;",
        "the direct visibility gate must require Color output, opaque meshes and a direct light")
require(flat, "hasDirectLight |= lightAt(i).kind != SceneLightKind::Environment && lightAt(i).intensity > 0;",
        "hasDirectLight must count only non-environment lights with positive intensity")
require(flat, "result.stats.opaqueVisibilityUnsupportedVertexDeformation = true; directVisibility = false;",
        "vertex deformation must disable the direct shadow ray and report the boundary")
require(flat, "gl.Uniform4fv(locations.opaqueControl, 1, visibilityFrame.control.data());",
        "the per-draw opaque control must be uploaded from the admitted visibility frame")

# The exact oracle stays wired to both the OpenGL test and the directional case that expects a
# blocked receiver to be zero.
require(flat_oracle, "for (const auto kind : {SceneLightKind::Directional, SceneLightKind::Point, SceneLightKind::Spot})",
        "the opaque visibility oracle must cover the directional light kind")
require(flat_oracle, "kind == SceneLightKind::Directional ? std::numeric_limits<double>::infinity() : std::sqrt(distanceSquared)",
        "the oracle must trace directional lights with an unbounded distance")
require(flat_oracle, "const double expected = blocked ? 0 :",
        "the oracle must expect a blocked receiver to receive zero direct light")
require(collapse(gl_tests), "opaqueVisibilityCases(backend, visibilityError)",
        "opengl_fixture_scene_backend_tests.cpp must run opaqueVisibilityCases")

# Float32 self-hit check. Emulate the shader's any-hit triangle test with every operation
# rounded to float32. Receivers are points on the light-facing faces of a 1 m box whose
# coordinates reach 3, moved off the face by up to 64 float32 ulps as a margin for hardware
# varying interpolation and fast division. The TLAS moves the origin into box-local space.
# With the shader bias no ray may hit the box; the old 1e-5 bias self-hits here.
def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def sub3(a, b):
    return [f32(a[i] - b[i]) for i in range(3)]


def dot3(a, b):
    return f32(f32(f32(a[0] * b[0]) + f32(a[1] * b[1])) + f32(a[2] * b[2]))


def cross3(a, b):
    return [f32(f32(a[1] * b[2]) - f32(a[2] * b[1])), f32(f32(a[2] * b[0]) - f32(a[0] * b[2])),
            f32(f32(a[0] * b[1]) - f32(a[1] * b[0]))]


def triangle_hit(origin, direction, a, b, c):
    e1, e2 = sub3(b, a), sub3(c, a)
    p = cross3(direction, e2)
    det = dot3(e1, p)
    if abs(det) <= 1.0e-7 * math.sqrt(dot3(e1, e1)) * math.sqrt(dot3(e2, e2)):
        return False
    t = sub3(origin, a)
    u = f32(dot3(t, p) / det)
    if u < 0.0 or u > 1.0:
        return False
    q = cross3(t, e1)
    v = f32(dot3(direction, q) / det)
    if v < 0.0 or f32(u + v) > 1.0:
        return False
    return f32(dot3(e2, q) / det) > 0.0


def shader_bias(position):
    return f32(max(1.0e-4, f32(1.0e-4 * max(abs(c) for c in position))))


centre = [f32(2.5), f32(2.5), f32(2.5)]
corners = [[f32(x * 0.5), f32(y * 0.5), f32(z * 0.5)] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
quads = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
box = []
for quad in quads:
    box.append((corners[quad[0]], corners[quad[1]], corners[quad[2]]))
    box.append((corners[quad[0]], corners[quad[2]], corners[quad[3]]))
generator = random.Random(20261016)
self_hits = 0
for sample in range(1500):
    axis, side = sample % 3, 1.0 if sample % 2 else -1.0
    normal = [0.0, 0.0, 0.0]
    normal[axis] = side
    position = [f32(centre[i] + generator.uniform(-0.499, 0.499)) for i in range(3)]
    position[axis] = f32(centre[axis] + side * 0.5)
    ulp = 2.0 ** (math.frexp(position[axis])[1] - 24)
    position[axis] = f32(position[axis] + generator.randint(-64, 64) * ulp)
    # Light from the receiver's side, down to about 0.06 degrees above the face.
    elevation = generator.choice((1.0e-3, 1.0e-2, 0.1, 0.5, 1.0)) * generator.uniform(0.5, 1.0)
    tangent = [generator.uniform(-1, 1) for _ in range(3)]
    tangent[axis] = 0.0
    length = math.sqrt(sum(c * c for c in tangent)) or 1.0
    direction = [tangent[i] / length + normal[i] * elevation for i in range(3)]
    length = math.sqrt(sum(c * c for c in direction))
    direction = [f32(c / length) for c in direction]
    bias = shader_bias(position)
    origin = [f32(position[i] + f32(normal[i] * bias)) for i in range(3)]
    local = [f32(origin[i] - centre[i]) for i in range(3)]
    if any(triangle_hit(local, direction, *triangle) for triangle in box):
        self_hits += 1
if self_hits:
    errors.append("float32 rays from a 1 m box at |p| ~ 3 self-hit %d times with the shader bias" % self_hits)

# Opaque alpha snap. The sky term and the shadow ray both require alpha >= 1. Alpha starts from
# the interpolated vertex opacity, which is 1.0 at every vertex of an opaque mesh, yet the
# interpolated value can land a few ulps below 1.0. Those pixels dropped the sky and shadow, and
# because the error follows view depth, the lighthouse box faces showed row stripes (the camera
# only pitches) and a checkerboard on the receding side face, even with the key light at zero.
# Both backends snap that error to 1.0 before the discard and before any alpha >= 1 gate.
snaps = (
    ("OpenGL", flat, "float alpha = vOpacity * texel.a * coverage;",
     "if (alpha > 1.0 - 1.0e-5 && alpha < 1.0) alpha = 1.0;",
     "if (alpha <= 0.0 || (uAlphaModeCutoff.x == 1.0 && alpha < uAlphaModeCutoff.y))",
     ("skyBase+=uSky[0].w>1.5", "float visibility = uAlphaModeCutoff.x == 0.0")),
    ("Metal", flat_metal, "float alpha = in.opacity * baseTexel.a * coverage;",
     "if (alpha > 1.0f - 1.0e-5f && alpha < 1.0f) alpha = 1.0f;",
     "if (alpha <= 0.0f || (u.materialKind.z == 1u && alpha < as_type<float>(u.materialKind.w)))",
     ("skyBase+=u.sky[0].w>1.5f", "float visibility = u.materialKind.z == 0u")),
)
for name, text, declaration, snap, discard, gates in snaps:
    at = text.find(declaration)
    snap_at = text.find(snap, at)
    discard_at = text.find(discard, at)
    if at < 0 or snap_at < 0 or discard_at < 0 or not at < snap_at < discard_at:
        errors.append(name + " must snap the interpolated opaque alpha to 1.0 right after computing it")
        continue
    if text.count(declaration) != 1:
        errors.append(name + " must compute the fixture surface alpha in exactly one place")
    for gate in gates:
        gate_at = text.find(gate, snap_at)
        if gate_at < 0:
            errors.append(name + " alpha >= 1 gate '" + gate + "' must follow the opaque alpha snap")
if "const float alpha = in.opacity * baseTexel.a * coverage;" in flat_metal:
    errors.append("Metal alpha must stay mutable so the opaque snap can apply")

# The snap window covers the interpolation error. Interpolate a constant 1.0 in float32 as
# b0*1 + b1*1 + b2*1 with b2 = 1 - b0 - b1, the way a rasterizer spends rounding. Some
# pixels land below 1.0, every one stays inside the 1e-5 window, and none exceed 1.0 by more.
below = 0
worst = 0.0
for sample in range(20000):
    b0 = f32(generator.random())
    b1 = f32(generator.random() * f32(1.0 - b0))
    b2 = f32(f32(1.0 - b0) - b1)
    w0, w1, w2 = f32(generator.uniform(0.1, 0.9)), f32(generator.uniform(0.1, 0.9)), f32(generator.uniform(0.1, 0.9))
    # Perspective-correct weights: (b_i / w_i) / sum(b_j / w_j).
    p0, p1, p2 = f32(b0 / w0), f32(b1 / w1), f32(b2 / w2)
    total = f32(f32(p0 + p1) + p2)
    value = f32(f32(f32(p0 / total) + f32(p1 / total)) + f32(p2 / total))
    if value < 1.0:
        below += 1
        worst = max(worst, 1.0 - value)
if below == 0:
    errors.append("the float32 interpolation model no longer reproduces sub-1.0 opacity; recheck the snap rationale")
if worst >= 1.0e-5:
    errors.append("float32 interpolated opacity fell %.3g below 1.0, outside the 1e-5 snap window" % worst)

if errors:
    for error in errors:
        print("FAIL: " + error, file=sys.stderr)
    raise SystemExit(1)
print("PASS: fixture lights trace the opaque shadow ray from the outward vertex normal on OpenGL and Metal")
