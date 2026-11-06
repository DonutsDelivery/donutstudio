#!/usr/bin/env python3
"""The Render node's renderMode must reach both consumers and invalidate the plan.

Usage: render_mode_plumbing_source_contract.py <video-helper dir>
"""
from pathlib import Path
import sys

helper = Path(sys.argv[1]).resolve()
root = helper.parent
checks = {
    "plugin/Source/graph/Visual3DGraphNodes.h": [
        'ParamSpec::choice ("renderMode"',
        '{ "renderMode", 0, 1, 0 }',
        "lowerRenderMode (*terminal, candidate.renderMode, error)",
    ],
    "plugin/Source/graph/ImportedAnimationGraphNodes.h": [
        "lowerRenderMode (*render, request.renderMode, error)",
        "leaf.renderMode = visualimportedscenerender::RenderMode::Accurate",
    ],
    "plugin/Source/VideoVisualGraph.cpp": [
        "request.renderMode = lowered.renderMode;",
        "request.renderMode = sceneRequest.renderMode;",
    ],
    "plugin/Source/VideoVisualGraph.h": ['parameterId == "renderMode"'],
    "plugin/Source/PluginProcessorVideoGraph.cpp": ['parameterId == "renderMode"'],
    "video-helper/src/imported_scene_visual_plan_execution.h": [
        "request.renderMode = compiled->importedSceneRender->renderMode;",
    ],
    "video-helper/src/imported_scene_payload_execution.cpp": [
        "selection.mode = request.renderMode == visualimportedscenerender::RenderMode::Fast",
    ],
}
missing = []
for relative, tokens in checks.items():
    text = (root / relative).read_text(encoding="utf-8")
    missing += [f"{relative}: {token}" for token in tokens if token not in text]

# Only the files that declare, lower, route, carry or (since tier K S2) read the
# mode may name it, each with an exact count of mentions, so a new read anywhere,
# including inside a listed renderer file, fails here. The slice that adds a
# read edits its count on purpose. Counts are (all mentions, renderMode mentions):
# the regex below finds renderMode, the SceneEngineMode type and RenderMode::.
import re

expected_mentions = {
    "shared/VisualImportedSceneRenderOperationContract.h": (24, 14),
    "plugin/Source/graph/Visual3DGraphNodes.h": (8, 5),
    "plugin/Source/graph/ImportedAnimationGraphNodes.h": (3, 2),
    "plugin/Source/VideoVisualGraph.cpp": (4, 4),
    "plugin/Source/VideoVisualGraph.h": (1, 1),
    "plugin/Source/PluginProcessorVideoGraph.cpp": (1, 1),
    "video-helper/src/imported_scene_visual_plan_execution.h": (2, 2),
    "video-helper/src/imported_scene_payload_execution.h": (4, 3),
    "video-helper/src/imported_scene_payload_execution.cpp": (4, 1),
    # The renderer declares SceneEngineMode and carries it on the selection, the
    # Path Trace draw and the stats it reports (enum plus four fields, each naming
    # the type twice). It never names the node's renderMode.
    "video-helper/src/fixture_scene_renderer.h": (9, 0),
    # S2 readers: the Preview budget scope and the Path Trace sample seeding. The
    # renderer also reads the mode for Realtime's reflection pass (Fast drops it,
    # ab89cd4c36), and the Path Trace engine reads it for the preview sample
    # target, the Fast cost ladder and the export sample count (5bda16a43e).
    "video-helper/src/fixture_scene_renderer.cpp": (2, 0),
    "video-helper/src/pathtrace_fixture_engine.cpp": (3, 0),
    # The probe and viewport reply name the mode of the drawn frame (a8b3b18dd3,
    # ab89cd4c36). It only reports; it never branches.
    "video-helper/src/path_trace_stats_reply.h": (2, 0),
    # The Render properties panel lists the node's renderMode row (ed57c5f791).
    "plugin/Source/VideoEditor/Render3DPropertiesModel.h": (1, 1),
    # The Path Trace engine's own EngineRenderMode accessor (5cca289464) shares
    # the identifier but is the engine's playback/settle mode, not the node's.
    "video-helper/src/pathtrace/engine.h": (1, 1),
}
# The UI's own render-mode setting (SettingsMenu, set_render_mode) is unrelated
# and never matches: these patterns need the exact lowercase identifier, the
# SceneEngineMode type or the contract's RenderMode:: enumerators.
mode_use = re.compile(r"\brenderMode\b|\bSceneEngineMode\b|\bRenderMode::")
node_mode = re.compile(r"\brenderMode\b")
extensions = {".h", ".hpp", ".cpp", ".cc", ".cxx", ".mm"}
seen = set()
for tree in ("video-helper/src", "plugin/Source", "shared"):
    for path in sorted((root / tree).rglob("*")):
        if path.suffix not in extensions or not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        found = (len(mode_use.findall(text)), len(node_mode.findall(text)))
        if relative not in expected_mentions:
            if found[0]:
                missing.append(f"{relative}: names renderMode outside the plumbing allowlist")
            continue
        seen.add(relative)
        if found != expected_mentions[relative]:
            missing.append(f"{relative}: mode mentions {found}, expected {expected_mentions[relative]}")
for relative in sorted(set(expected_mentions) - seen):
    missing.append(f"{relative}: allowlisted mode file is missing")

# S2: an Accurate Preview draws with the export budgets, a Fast one keeps the
# Preview budgets, and neither adds a readback or a wait. The renderer opens the
# budget scope from the selection; the backend reads it only at the gated sites.
# S2 and S4: the mode picks the budget scope for both draw classes; a Fast
# draw - Preview or export - takes the Preview budgets and the default path
# (Accurate, Export budgets) keeps the Export scope, plan and program exactly.
s2_checks = {
    "video-helper/src/fixture_scene_renderer.cpp": [
        "const bool accurateMode = engineSelection_.mode == SceneEngineMode::Accurate;",
        "const bool previewExportBudgets = use == RenderUse::Preview && accurateMode;",
        "const arbitgpu::previewbudget::Scope previewBudgetScope (accurateMode\n        ? arbitgpu::previewbudget::Budget::Export : arbitgpu::previewbudget::Budget::Preview);",
        "arbitgpu::openGlCausticRenderPlan (causticUse, dimensions.width, dimensions.height,\n"
        "                                                                    previewExportBudgets);",
        "sceneInputs.previewUse = use == RenderUse::Preview;",
        "draw.mode = engineSelection_.mode;",
    ],
    "video-helper/src/pathtrace_fixture_engine.cpp": [
        "const bool accurate = draw.mode == SceneEngineMode::Accurate;",
        "engine_.previewSeededLikeExport = accurate;",
        "engine_.maximumPreviewSamples = pt::previewSampleTarget (accurate, draw.previewSamples, draw.exportSamples);",
    ],
    "video-helper/src/pathtrace/engine.cpp": [
        "previewSeeded_ ? kExportGeneration : 0);",
        "|| previewSeededLikeExport != previewSeeded_",
        "|| previewSamples_ > maximumPreviewSamples;",
    ],
    "video-helper/src/gpu_backend/backend_opengl.cpp": [
        "const bool lightKeyed=sourceLight&&(runtime.lightOverride?runtime.lightOverride->id==source",
        "if(!animatedLightsOwnedByScene(*scene,runtime))return false;",
        "const bool previewGlassFrame = !previewbudget::exportBudgets()\n            && resources->smoothGlass.enabled && !roughGlassFrame && !diffractionFrame;",
        "const bool previewGlassWalk = previewGlassFrame && !previewbudget::exportBudgets();",
        "const bool previewExportBudgets=preview&&previewbudget::exportBudgets();",
        "const bool fastExportDraw=!preview&&!previewbudget::exportBudgets();",
        "const bool previewDegrades=(preview||fastExportDraw)&&!previewExportBudgets;",
        # A refused Accurate frame is latched and answered before any build or GPU work.
        "if(const auto* latched=resources->accurateCausticRefusal.find(refusalKey))",
        "refused.error=openGlAccurateCausticRefusal(message);",
        "resources->accurateCausticRefusal.latch(refusalKey,refused.error);",
        "return refuseLimit(\"OpenGL caustic admitted storage exceeds native GL limits: \"+failure);",
        "return refuseLimit(\"OpenGL caustic deposition exceeds native GL limits: \"+failure);",
        "return refuseLimit(\"OpenGL caustic splat storage exceeds native GL limits: \"+failure);",
        # The Preview wait and readback rules stay keyed on previewUse alone.
        # A test run also finishes every band so it can yield after each one (231b51c672).
        "const bool bandFinish = smoothglass::cameraWalkBudget(runtimeInputs.previewUse, width, height).bandFinish\n"
        "                || testpacing::enabled();",
        "if (resources->smoothGlass.enabled && !roughGlassFrame && !runtimeInputs.previewUse) {",
        "if (!authoredDiffraction || runtimeInputs.previewUse) return true;",
    ],
    "video-helper/src/gpu_backend/fixture_glass_caustics_opengl.h": [
        '#include "fixture_preview_budget.h"',
        "if(use!=OpenGlCausticRenderUse::Preview && previewbudget::exportBudgets())return plan;",
        "if(!exportSchedule)plan.scheduling.entrySubdivisions=kPreviewCausticEntrySubdivisions;",
        "plan.hostReceipt=false;plan.stageFences=false;",
        "switch Render Mode to Fast to preview this scene",
    ],
}
for relative, tokens in s2_checks.items():
    text = (root / relative).read_text(encoding="utf-8")
    missing += [f"{relative}: {token}" for token in tokens if token not in text]
backend_text = (root / "video-helper/src/gpu_backend/backend_opengl.cpp").read_text(encoding="utf-8")
# Every budget read is one of the gated sites named above, plus the schedule
# check: the S2 walk, plan and scope reads and the S4 Fast-export walk-frame
# and ladder reads.
if backend_text.count("previewbudget::exportBudgets()") != 5:
    missing.append("backend_opengl.cpp: previewbudget::exportBudgets() read outside the gated sites")
# Nothing in the budget scope may wait or read back.
budget_header = (root / "video-helper/src/gpu_backend/fixture_preview_budget.h").read_text(encoding="utf-8")
for forbidden in ("glFinish", "readRawPass", "ClientWaitSync", "MapBufferRange"):
    if forbidden in budget_header:
        missing.append(f"fixture_preview_budget.h: names {forbidden}")
# Metal draws Preview and export through one path today, so it has no Preview
# reduction to gate. A Metal Preview budget would have to read the scope too.
metal = (root / "video-helper/src/gpu_backend/backend_sokol.mm").read_text(encoding="utf-8")
# Metal reads previewUse only to pick Preview-only solid viewport shading and
# X-ray (3d695c76c5), a display choice that reduces no budget, so those two
# calls are set aside before the search.
metal_budget_text = re.sub(
    r"fixtureviewportsolid::(?:solid|xray)Applies\s*\(\s*runtimeInputs\.previewUse", "", metal)
for reduction in ("previewUse", "GLASS_PREVIEW_VISITS", "kPreviewCameraVisits", "kPreviewCausticEntrySubdivisions"):
    if reduction in metal_budget_text and "previewbudget::" not in metal:
        missing.append(f"backend_sokol.mm: Metal Preview reduction {reduction} is not gated on the mode")

# The helper's engine selection stores the mode and nothing else reads it.
payload_lines = [line for line in
                 (root / "video-helper/src/imported_scene_payload_execution.cpp")
                 .read_text(encoding="utf-8").splitlines() if "renderMode" in line]
if len(payload_lines) != 1:
    missing.append("video-helper/src/imported_scene_payload_execution.cpp: renderMode read more than once")

if missing:
    print("FAIL: renderMode plumbing contract:\n  " + "\n  ".join(missing), file=sys.stderr)
    raise SystemExit(1)
print("PASS: renderMode is lowered, routed, carried to the preview and export request, "
      "and only the S2 budget readers branch on it")
