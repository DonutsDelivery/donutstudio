#!/usr/bin/env python3
from pathlib import Path
import sys


root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
main = (root / "src/main.cpp").read_text()
handler = (root / "../plugin/Source/CommandServerVideo.cpp").read_text()


def section(text, start, end):
    first = text.index(start)
    return text[first:text.index(end, first)]


probe = section(main, 'if (method == "composite_frame_probe")', 'if (method == "proxy_progress")')
viewport = section(main, 'if (method == "viewport_info"', "// ---- Video scopes")

# The probe clears first, so its reply never reports an earlier job's paths.
clear = probe.index("clearPathTraceExportStats()")
render = probe.index("renderCompositeFrame (")
append = probe.index("appendPathTraceStats (")
assert clear < render < append, "the probe clears, renders, then reports"
assert "RenderUse::Export" in probe[append:], "the probe reports its own Export counters"

assert "appendPathTraceStats (" in viewport, "viewport_info reports the counters"
assert "RenderUse::Preview" in viewport, "viewport_info reports the Preview counters"

# The inline probe reply is rebuilt key by key. Without this line a 1080p probe
# drops the object that a 4K artifact probe keeps through result.clone().
inline = section(handler, 'object->setProperty("pixelFormat", "rgba8");\n    object->setProperty("snapshotGeneration"',
                 "return frameProbeResultJson(frame, object,")
assert 'result.hasProperty("pathTraceStats")' in inline, "a Raster inline reply must omit the key"
assert 'object->setProperty("pathTraceStats"' in inline, "the inline reply forwards pathTraceStats"

# Both replies name the size the film really was, and the HUD shows it.
reply_header = (root / "src/path_trace_stats_reply.h").read_text()
assert '"filmWidth", stats->filmWidth' in reply_header and '"filmHeight", stats->filmHeight' in reply_header
assert '"temporalDenoise", stats->temporalDenoise' in reply_header, "the reply says whether the image blends history"
# A GPU device that failed under Auto stays named after the CPU takes over: the
# header notice reads fallbackReason, set only on that fallback.
assert '"fallbackReason", stats->fallbackReason' in reply_header, "the reply names a failed GPU device"
engine = (root / "src/pathtrace_fixture_engine.cpp").read_text()
assert "stats_.fallbackReason = fellBackToCpu_ ? deviceReason_ : nullptr;" in engine, "only the device fallback sets fallbackReason"
renderer = (root / "src/fixture_scene_renderer.cpp").read_text()
assert "traced.filmWidth = dimensions.width;" in renderer, "the film size comes from the rendered dimensions"

# A Path Trace frame's HUD names the film, not a raster scene render or backend.
hud = (root / "../plugin/Source/VideoEditor/VideoEditorContent.cpp").read_text()
assert "Path Trace film on %DEVICE%: %TIME% ms at %WIDTH%x%HEIGHT% for" in hud, "the film row names the device and the film size"
assert '.replace("%DEVICE%", pathTraceStats->getProperty("device").toString())' in hud, "the device is the film's, from its stats"
scene_row = section(hud, 'measuredDurationUs(telemetry->getProperty("importedSceneRender")', "Imported scene render: Not observed")
assert scene_row.index("Path Trace film on") < scene_row.index("Imported scene render:"), "the film row wins for a Path Trace frame"
assert "Image: Path Trace film, presented by %BACKEND%, display fallbacks %COUNT%" in hud, "the backend row names the film as the image source"
assert '"Display path: "' in hud and '"Display GPU: "' in hud, "the presentation rows do not read as the tracer"
# The viewport header says which of these the tracer is doing, with its target.
for row in ("Path Trace %MODE% converging: %SAMPLES% of %TARGET%", "Path Trace %MODE% settled: %SAMPLES% of %TARGET%",
            "Path Trace %MODE% playing: %SAMPLES% of %TARGET%", "Path Trace %MODE% export: %SAMPLES% of %TARGET%"):
    assert row in scene_row, "the header row names the activity and the target: " + row
assert scene_row.count("%DENOISER%") >= 4, "every header row names the denoiser that ran"
# The viewport tells the tracer when the transport plays, so a repeated time stays
# on the playback budget. The hold has no frame budget, only the target and a stall.
viewport = (root / "src/viewport.cpp").read_text()
assert "setPathTraceTransportPlaying (playing);" in viewport, "the viewport publishes the transport state"
assert "pathTracePreviewHoldBudget" not in viewport, "the Path Trace hold has no frame budget"
