#!/usr/bin/env python3
"""Pins the open-sea cloud layer in the one recipe both raster backends compile.

Background and BRDF integration call analyticSkyRadiance, so the cloud layer has
to sit inside that function. Authored parameters have to reach it through the
scene record, the node, the runtime parameters and both wire formats.
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
failures = []


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def check(condition, message):
    if not condition:
        failures.append(message)


sky = read("video-helper/src/gpu_backend/fixture_analytic_sky.h")
recipe = sky[sky.index('R"shader('):sky.index(')shader"')]
radiance = recipe[recipe.index("vec3 analyticSkyRadiance"):recipe.index("float analyticSkyRadicalInverse")]
check("analyticSkyCloudOpacity(d,horizon,ground)" in radiance,
      "analyticSkyRadiance composites the cloud layer")
check(recipe.index("float analyticSkyCloudOpacity") < recipe.index("vec3 analyticSkyRadiance"),
      "cloud helpers come before analyticSkyRadiance")
check(recipe.index("vec3 analyticSkyRadiance") < recipe.index("vec3 analyticSkyBase"),
      "the environment image recipe can still cut every sky helper off at analyticSkyBase")
check("mod(" not in recipe, "the shared recipe avoids mod(), which MSL lacks")
check(not re.search(r"\b(packed|half|ref|texture|type|kernel)\b", recipe),
      "the shared recipe avoids names reserved in GLSL or MSL")
check("if(cloud<=0.0) return clear;" in radiance, "a clear sky keeps the released arithmetic")
check("analyticSkyCloudOpacity(d" in radiance and "uTime" not in recipe,
      "cloud motion comes from an authored parameter, so a seek replays exactly")
check("inline float cloudOpacity(" in sky and "inline std::array<float,3> radiance(" in sky,
      "a C++ twin lets the CPU engine and the tests evaluate the same field")

fields = ("cloudCoverage", "cloudType", "cloudDrift", "cloudHeadingDegrees")
scene = read("shared/Visual3DScene.h")
struct = scene[scene.index("struct SceneAnalyticSky"):scene.index("inline bool sameAnalyticSky")]
same = scene[scene.index("inline bool sameAnalyticSky"):scene.index("struct SceneEnvironmentImage")]
for field in fields:
    check(f"float {field} = 0.0f;" in struct, f"the scene record stores {field}, off by default")
    check(f"a.{field},b.{field}" in same, f"sameAnalyticSky compares {field}")

node = read("plugin/Source/graph/Visual3DGraphNodes.h")
runtime = read("video-helper/src/imported_scene_visual_plan_execution.h")
aov = read("shared/SceneAovOperationContract.h")
plan = read("shared/VisualImportedSceneRenderOperationContract.h")
for field in fields:
    check(f'"{field}"' in node, f"the Environment Light node authors {field}")
    check(f'"{field}",&s.{field}' in runtime, f"modulation reaches {field}")
    check(f"sky.{field}" in aov, f"the scene wire format carries {field}")
    check(f"sky.{field}" in plan, f"the compiled plan carries {field}")
check("kSkyCloudExtensionMagic" in aov, "scene clouds use an optional trailer, so clear scenes keep their bytes")

if failures:
    print("\n".join(f"FAIL: {failure}" for failure in failures))
    sys.exit(1)
print("analytic sky clouds source contract: passed")
