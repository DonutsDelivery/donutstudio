#!/usr/bin/env python3
# Contract for the directional-light shadow ray in the OpenGL fixture scene shader. Each
# direct light traces the opaque BVH from the receiver toward the light. A directional light
# uses an unbounded ray along its per-draw rotation. A point or spot light stops at the
# receiver's distance. The ray is composed into every fixture fragment program, and the
# host gates it through uOpaqueControl.w. The exact GPU oracle that covers this path lives in
# tests/support/opaque_visibility_cases.h, and this contract checks that it stays wired.
from pathlib import Path
import re
import sys

root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
backend = (root / "src/gpu_backend/backend_opengl.cpp").read_text(encoding="utf-8")
visibility = (root / "src/gpu_backend/fixture_opaque_visibility.h").read_text(encoding="utf-8")
oracle = (root / "tests/support/opaque_visibility_cases.h").read_text(encoding="utf-8")
gl_tests = (root / "tests/opengl_fixture_scene_backend_tests.cpp").read_text(encoding="utf-8")

errors = []


def collapse(text):
    return re.sub(r"\s+", " ", text)


flat = collapse(backend)
flat_visibility = collapse(visibility)
flat_oracle = collapse(oracle)


def require(text, needle, message):
    if needle not in text:
        errors.append(message)


# Light loop: the directional light is kind 0 with a per-draw rotation, so its ray direction
# follows the current uLightRotations value on every frame.
require(flat, "vec3 toLight = kind == 0 ? rotateQ(uLightRotations[lightIndex], vec3(0.0, 0.0, 1.0)) : baseSurfaceUnit(delta);",
        "the light loop must derive a directional toLight from the per-draw uLightRotations")

# The visibility term is gated by the opaque alpha and transmission conditions, then uses the
# unbounded ray for directional lights and the receiver distance for the others.
require(flat, "float visibility = uAlphaModeCutoff.x == 0.0 && transmission == 0.0 && alpha >= 1.0 "
              "? opaqueVisibility(vWorldPosition,cross(dFdx(vWorldPosition),dFdy(vWorldPosition)), "
              "toLight,kind == 0 ? 3.402823e38 : length(delta),uOpaqueControl) : 1.0;",
        "the direct visibility term must use the opaque ray with an unbounded limit for kind 0")

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

if errors:
    for error in errors:
        print("FAIL: " + error, file=sys.stderr)
    raise SystemExit(1)
print("PASS: directional fixture lights trace the opaque shadow ray, gated by uOpaqueControl")
