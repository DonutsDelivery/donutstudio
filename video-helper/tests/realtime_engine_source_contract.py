#!/usr/bin/env python3
"""Check that Realtime is its own engine from the Render node to the HUD.

Realtime is a distinct engine value, so a saved Raster graph keeps its bytes.
It renders without caustics and says that they need Path Trace.
"""

from pathlib import Path
import re

root = Path(__file__).resolve().parents[2]


def read(path):
    return (root / path).read_text()


def require(text, needle, what):
    if needle not in text:
        raise SystemExit(f"missing: {what}: {needle}")


def refuse(text, needle, what):
    if needle in text:
        raise SystemExit(f"still present: {what}: {needle}")


contract = read("shared/VisualImportedSceneRenderOperationContract.h")
nodes = read("plugin/Source/graph/Visual3DGraphNodes.h")
animation = read("plugin/Source/graph/ImportedAnimationGraphNodes.h")
graph = read("plugin/Source/VideoVisualGraph.cpp")
selection = read("video-helper/src/imported_scene_payload_execution.cpp")
animated = read("video-helper/src/imported_animated_scene_payload_execution.cpp")
renderer_h = read("video-helper/src/fixture_scene_renderer.h")
renderer = read("video-helper/src/fixture_scene_renderer.cpp")
reply = read("video-helper/src/path_trace_stats_reply.h")
main = read("video-helper/src/main.cpp")
command = read("plugin/Source/CommandServerVideo.cpp")
hud = read("plugin/Source/VideoEditor/VideoEditorContent.cpp")
backend = read("video-helper/src/gpu_backend/backend.h")

# Selection: appended after the existing choices, the default stays Raster, and
# the value reserved for Workbench stays free.
require(contract, "Realtime = 3", "the Realtime wire value")
require(contract, "kReservedWorkbenchRenderEngine = 2", "the Workbench reservation")
require(contract, "case RenderEngine::Realtime:", "Realtime request validation")
require(nodes, '{ "renderEngine", 0, 2, 0 }', "the Render node's engine range")
require(nodes, 'TRANS ("Raster"), TRANS ("Path Trace"), TRANS ("Realtime")', "the engine choice")
require(nodes, "RenderEngine::Realtime", "Realtime lowering")

# A Realtime request is not limited to what the path tracer can write.
for name, text in (("nodes", nodes), ("animation", animation), ("graph", graph)):
    refuse(text, "enderEngine != visualimportedscenerender::RenderEngine::Raster",
           f"a Path Trace refusal that also catches Realtime in {name}")

# Routing reaches the fixture renderer as its own engine kind.
require(renderer_h, "Realtime = 2", "the Realtime engine kind")
require(selection, "SceneEngineKind::Realtime", "the request to engine mapping")
require(backend, "bool realtimeEngine = false;", "the backend's Realtime input")
require(renderer, "sceneInputs.realtimeEngine", "the Realtime input set by the renderer")

# The caustic is left out and named, on every route that refused it before.
require(contract, 'kRealtimeCausticsNeedPathTrace = "caustic off: caustics need Path Trace"',
        "the caustic message")
require(renderer, "kRealtimeCausticsNeedPathTrace", "the renderer's caustic notice")
require(contract, 'kRealtimeReliefNeedsPathTrace = "relief scatter off: Relief Height needs Path Trace"',
        "the relief notice's text")
require(renderer, "kRealtimeReliefNeedsPathTrace", "the renderer's relief notice")
strip = renderer.index("sceneInputs.glassCaustic.reset();")
if strip > renderer.index("std::optional<arbitgpu::glasscaustics::AdmittedBranchPolicy> causticPolicy;"):
    raise SystemExit("Realtime must drop the caustic before caustic admission")
require(animated, "SceneEngineKind::Realtime", "the animated route's Realtime case")

# Telemetry names the engine, and a Raster reply keeps exactly its keys.
require(reply, 'reply["realtimeStats"]', "the Realtime reply")
require(reply, '{ "engine", "realtime" }', "the Realtime engine name")
require(reply, '{ "causticsNeedPathTrace", stats->causticsNeedPathTrace }', "the caustic flag")
if not re.search(r"appendRealtimeStats \(nlohmann::json& reply, const std::optional<RealtimeFrameStats>& stats\)\s*\{\s*if \(! stats\.has_value\(\)\)\s*return;", reply):
    raise SystemExit("a render without Realtime stats must add no key")
if main.count("appendRealtimeStats") != 2:
    raise SystemExit("the probe and viewport_info replies must both name Realtime")
require(command, '"realtimeStats"', "the MCP probe forward")
require(hud, 'TRANS("Realtime render: %TIME% ms/frame', "the HUD render row")
require(hud, 'TRANS("Image: Realtime, drawn by %BACKEND%, fallbacks %COUNT%")', "the HUD image row")

print("realtime engine source contract: ok")
