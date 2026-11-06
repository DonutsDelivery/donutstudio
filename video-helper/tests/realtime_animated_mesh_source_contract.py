#!/usr/bin/env python3
"""Realtime draws an animated imported mesh through the fixture scene renderer.

The Note-Driven Particle and Light Rig starter renders the bundled animated
model. Its bound meshes drew through the deformation renderer, which has no
engine choice, so renderEngine 2 (Realtime) gave the Raster frame byte for byte.
Path Trace already baked the frame's vertices and drew through the fixture
renderer. This pins that Realtime takes the same route on the one-leaf and
multi-leaf paths, that the starter takes that route with renderEngine 2
lowering to Realtime, and that the fixture renderer composes a Realtime program
(FIXTURE_REALTIME) for it on OpenGL and selects the Realtime surface on Metal.
"""

from pathlib import Path
import re
import sys

root = Path(__file__).resolve().parents[2]


def read(path):
    return (root / path).read_text()


failures = []


def need(text, needle, what):
    if needle not in text:
        failures.append(f"missing: {what}: {needle}")


animated = read("video-helper/src/imported_animated_scene_payload_execution.cpp")
contract = read("shared/VisualImportedSceneRenderOperationContract.h")
static_route = read("video-helper/src/imported_scene_payload_execution.cpp")
renderer = read("video-helper/src/fixture_scene_renderer.cpp")
opengl = read("video-helper/src/gpu_backend/backend_opengl.cpp")
sokol = read("video-helper/src/gpu_backend/backend_sokol.mm")
starters = read("plugin/Source/graph/BuiltInVisualModules.h")
nodes = read("plugin/Source/graph/Visual3DGraphNodes.h")

# The starter: the animated model starter with a Render node, so the animated route draws it.
rig = starters[starters.find("inline GraphModulePackage noteDrivenParticleAndLightRigStarter()"):]
rig = rig[:rig.find("\n}\n")]
need(rig, "animatedBlenderModelStarter()", "the light rig renders the bundled animated model")
need(rig, "kinds::Visual3DRender", "the light rig draws through a Render 3D node")
need(nodes, "engine = visualimportedscenerender::RenderEngine::Realtime;", "renderEngine 2 lowers to Realtime")
need(static_route, "? videorender::fixture3d::SceneEngineKind::Realtime", "the helper selects the Realtime engine")
need(static_route, "deformationExecution_->selectEngine(sceneEngineSelection(request));",
     "the animated route receives the node's engine")

# The rule and both bake sites.
need(contract, "inline constexpr bool animatedMeshDrawsBaked(RenderEngine engine) noexcept", "the baked-mesh rule")
need(contract, "return engine != RenderEngine::Raster;", "every engine but Raster bakes")
need(animated, "const bool bakedEngine = visualimportedscenerender::animatedMeshDrawsBaked(engine);",
     "the one-leaf route asks the rule")
need(animated, "if (bakedEngine && !bindings_.empty() && !restScene)", "the one-leaf route bakes for Realtime")
need(animated, "if (visualimportedscenerender::animatedMeshDrawsBaked(compositionEngine))",
     "the multi-leaf route bakes for Realtime")
for stale in ("const bool pathTrace = sceneRenderer_.selectedEngine().engine",
              "if (pathTrace && !bindings_.empty()",
              "if (sceneRenderer_.selectedEngine().engine == videorender::fixture3d::SceneEngineKind::PathTrace)"):
    if stale in animated:
        failures.append(f"a Path Trace-only bake condition is back: {stale}")

# The fixture renderer turns a Realtime selection into the Realtime program.
need(renderer, "sceneInputs.realtimeEngine = true;", "the fixture renderer marks Realtime frames")
need(renderer, "realtimeScope.emplace();", "the fixture renderer opens the Realtime scope")
need(opengl, "if (realtimeengine::active())", "the OpenGL program features read the Realtime scope")
need(opengl, "| kFixtureFragmentRealtime;", "a Realtime frame takes the Realtime feature bit")
if not re.search(r'kFixtureFragmentRealtime\) != 0 \? "#define FIXTURE_REALTIME 1\\n"', opengl):
    failures.append("the OpenGL composer no longer defines FIXTURE_REALTIME for the Realtime feature bit")
need(opengl, "if ((features & kFixtureFragmentRealtime) != 0) text += realtimeengine::glslShader();",
     "the OpenGL composer appends the Realtime text")
need(sokol, "realtimeSoftVisibility(in.worldPosition,in.worldNormal,", "Metal Realtime draws soft light shadows")

if failures:
    print("\n".join(failures))
    sys.exit(1)
print("Realtime draws animated imported meshes through the fixture renderer")
