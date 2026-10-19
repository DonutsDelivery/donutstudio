#!/usr/bin/env python3
"""Pins the imported scene render timer, its JSON field, and the HUD row.

Usage: imported_scene_render_telemetry_source_contract.py <video-helper dir>
"""
from pathlib import Path
import re
import sys

root = Path(sys.argv[1]).resolve()
execution = (root / "src/imported_scene_visual_plan_execution.h").read_text()
telemetry = (root / "src/visual_plan_telemetry.h").read_text()
telemetry_json = (root / "src/visual_plan_telemetry_json.h").read_text()
viewport = (root / "src/viewport.cpp").read_text()
exporter = (root / "src/exporter.cpp").read_text()
fixture_header = (root / "src/fixture_scene_renderer.h").read_text()
fixture = (root / "src/fixture_scene_renderer.cpp").read_text()
seam = (root / "src/glb_native_render_seam.cpp").read_text()
payload_header = (root / "src/imported_scene_payload_execution.h").read_text()
animated_header = (root / "src/imported_animated_scene_payload_execution.h").read_text()
animated = (root / "src/imported_animated_scene_payload_execution.cpp").read_text()
strings = (root.parent / "tools/i18n/strings.json").read_text()
hud = (root.parent / "plugin/Source/VideoEditor/VideoEditorContent.cpp").read_text()

errors = []

def need(ok, message):
    if not ok:
        errors.append(message)

# The timer wraps the execute call and only reports for completed preview frames.
timed = re.search(
    r"previewRenderStart = std::chrono::steady_clock::now\(\);\s*"
    r"const bool executed = use == NativeImportedSceneRenderUse::Preview\s*"
    r"\? execution->executePreview\(request, receipt, error\)\s*"
    r": execution->executeExport\(request, receipt, error\);\s*"
    r"const auto previewRenderElapsed", execution)
need(timed, "preview render timer must wrap executePreview/executeExport")
need(re.search(r"receipt\.pending\) return VisualImportedScenePreparation::pending;\s*"
               r"// [^\n]*\n\s*if \(previewRenderNs != nullptr && use == NativeImportedSceneRenderUse::Preview\)",
               execution),
     "duration must be reported after the pending check and for Preview only")

# Telemetry owns the counter and resets it with the other duration counters.
need("bool recordImportedSceneRender (uint64_t durationNs)" in telemetry,
     "VisualPlanTelemetry::recordImportedSceneRender missing")
need("addDuration(importedSceneRender_, durationNs)" in telemetry, "counter must use addDuration")
need("importedSceneRender_ = {};" in telemetry, "resetUnlocked must clear importedSceneRender_")
need("out.importedSceneRender = importedSceneRender_;" in telemetry, "snapshot must copy importedSceneRender_")

# One JSON field name shared with the plugin reader.
need('{ "importedSceneRender", duration(value.importedSceneRender) }' in telemetry_json,
     "JSON field importedSceneRender missing")
need('telemetry->getProperty("importedSceneRender")' in hud, "HUD must read importedSceneRender")

# Both viewport prepare calls hand the duration to the viewport telemetry.
need(len(re.findall(r"im\.visualTelemetry\.recordImportedSceneRender \(", viewport)) == 2,
     "both viewport imported scene prepare calls must record the render duration")
need("&importedSceneRenderNs" in viewport and "&particleSceneRenderNs" in viewport,
     "viewport must pass the duration out-parameter")

# Export timing is untouched.
need("recordImportedSceneRender" not in exporter and "previewRenderNs" not in exporter,
     "export path must not report preview render time")

# HUD row label.
need('TRANS("Imported scene render: %TIME% ms/frame")' in hud, "HUD row label missing")
need('TRANS("Imported scene render: Not observed")' in hud, "HUD unobserved label missing")
need('TRANS("Compositor / present: %COMP% us / no window present")' in hud,
     "docked compositor-only HUD label missing")

# The glass caustic degradation reason travels from the fixture RenderedFrame to the HUD row.
# 1. The fixture renderer sets it on the Preview frame, and the GLB receipt exchange keeps it.
need("std::string causticDegradation;" in fixture_header, "RenderedFrame::causticDegradation missing")
need("rendered.causticDegradation = std::move (causticDegradation);" in fixture,
     "fixture renderer must hand its reason to the RenderedFrame")
need("rendered.causticDegradation.swap(other.rendered.causticDegradation);" in seam,
     "GlbNativeFrameReceipt::exchange must carry the caustic reason")
# 2. Each execution receipt exposes it: the static route through the GLB frame, the animated
#    routes through their own receipt field, taken from the scene renderer frame.
need(re.search(r"static const std::string& causticDegradation\(const Receipt& receipt\) noexcept\s*"
               r"\{\s*return receipt\.deformationFrame\s*"
               r"\? receipt\.deformationFrame->causticDegradation\s*"
               r": receipt\.frame\.rendered\.causticDegradation;", payload_header),
     "ImportedScenePayloadExecution::causticDegradation must read the animated or the GLB frame")
need("std::string causticDegradation;" in animated_header,
     "ImportedAnimatedSceneReceipt::causticDegradation missing")
need("receipt.causticDegradation=std::move(rendered.causticDegradation);" in animated,
     "camera/light composition receipt must take the renderer's reason")
need("causticDegradation = std::move(sceneFrame.causticDegradation);" in animated
     and "receipt.causticDegradation = std::move(causticDegradation);" in animated,
     "single-leaf animated receipt must take the renderer's reason")
# 3. The prepare step reports it for completed Preview frames only, after the pending check.
need("std::string* causticDegradation = nullptr)" in execution
     and len(re.findall(r"std::string\* causticDegradation = nullptr", execution)) == 1,
     "prepare must default to no caustic reason out-parameter")
need(re.search(r"if \(causticDegradation != nullptr && use == NativeImportedSceneRenderUse::Preview\)\s*\{\s*"
               r"if constexpr \(HasImportedCausticDegradation<Execution>::value\)\s*"
               r"\*causticDegradation = Execution::causticDegradation\(receipt\);\s*"
               r"else\s*causticDegradation->clear\(\);", execution),
     "the reason must be assigned, or cleared, for completed Preview frames only")
need(execution.find("receipt.pending) return VisualImportedScenePreparation::pending;")
     < execution.find("*causticDegradation = Execution::causticDegradation(receipt);"),
     "the reason must be reported after the pending check")
# 4. The viewport records it beside the render time and extent, so an empty reason clears it.
need(len(re.findall(r"im\.visualTelemetry\.recordImportedSceneCausticDegradation \(", viewport)) == 2,
     "both viewport imported scene prepare calls must record the caustic reason")
need("&importedSceneCausticDegradation" in viewport and "&particleSceneCausticDegradation" in viewport,
     "viewport must pass the caustic reason out-parameter")
# 5. Telemetry keeps the latest reason under its mutex, clears it on reset, and bounds it.
need("bool recordImportedSceneCausticDegradation (const std::string& reason)" in telemetry,
     "VisualPlanTelemetry::recordImportedSceneCausticDegradation missing")
need("importedSceneCausticDegradation_.assign (reason, 0, kMaxCausticDegradationBytes);" in telemetry,
     "the reason must be assigned, not appended, and bounded")
need("out.importedSceneCausticDegradation = importedSceneCausticDegradation_;" in telemetry,
     "snapshot must copy the caustic reason")
need("importedSceneCausticDegradation_.clear();" in telemetry, "resetUnlocked must clear the caustic reason")
# 6. One JSON field name shared with the plugin reader, and one HUD warning row below the render row.
need('{ "importedSceneCausticDegradation", value.importedSceneCausticDegradation }' in telemetry_json,
     "JSON field importedSceneCausticDegradation missing")
need('telemetry->getProperty("importedSceneCausticDegradation")' in hud,
     "HUD must read importedSceneCausticDegradation")
need(re.search(r'TRANS\("Imported scene render: Not observed"\), cDim \}\);.*?'
               r'TRANS\("Caustic: %REASON%"\)\s*\.replace\("%REASON%", [^\n]*\), cWarn \}\);.*?'
               r'TRANS\("Generators: %TIME% us"\)', hud, re.S),
     "the Caustic row must sit below the imported scene render row, in the warning colour")
need("causticReason.isNotEmpty()" in hud, "the Caustic row must show only a non-empty reason")
need('"Caustic: %REASON%"' in strings, "HUD label missing from tools/i18n/strings.json")
# Export never reports a reason.
need("causticDegradation" not in exporter and "CausticDegradation" not in exporter,
     "export path must not report the caustic reason")

# Zero-copy release counts a presented frame, handoff alone does not.
need("bool recordTransportRelease()" in telemetry, "recordTransportRelease missing")
need("bool zeroCopyReleased()" in (root / "src/viewport_telemetry_owner.h").read_text(),
     "ViewportTelemetryOwner::zeroCopyReleased missing")
need(len(re.findall(r"viewportTelemetry\.zeroCopyReleased\(\)", viewport)) == 2,
     "both FrameRelease branches must report to zeroCopyReleased")

if errors:
    print("imported scene render telemetry contract failures:\n  " + "\n  ".join(errors), file=sys.stderr)
    raise SystemExit(1)
print("imported scene render telemetry source contract: PASS")
