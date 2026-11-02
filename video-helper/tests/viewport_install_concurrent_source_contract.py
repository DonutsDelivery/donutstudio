#!/usr/bin/env python3
from pathlib import Path
import re
import sys


root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
render = (root / "../plugin/Source/PluginProcessorVideoRender.cpp").read_text()
viewport = (root / "src/viewport.cpp").read_text()
exporter = (root / "src/exporter.cpp").read_text()
scene = (root / "src/imported_scene_visual_plan_execution.h").read_text()

# R4's video_editor_close waited behind a 78 s install on the message thread and
# missed the client's 90 s window. The editor's syncs install on a worker instead.
start = render.index("bool HarmonicMIDIProcessor::videoViewportSyncTimelineNow(")
sync = render[start:render.index("\nvoid HarmonicMIDIProcessor::videoViewportSyncScore", start)]
background = sync.index("if (installInBackground)\n")
assert "installWorker.schedule" in sync[background:], "the editor's install must run on the worker"
assert sync.index("installWorker.schedule") < sync.rindex("videoSidecar->viewportSetTimeline"), \
    "the worker path must return before the synchronous send"
assert "installs.admitWhileRunning" in sync, "a sync during an install must not start a second one"
assert "installCancelled.store (true)" in render and "installWorker.stop()" in render, \
    "teardown must end the unclocked wait and then join the worker"

# Render-loop and export seeding read the admitted execution. Lowering a 276 MB plan
# again on every frame, under the viewport state lock, stalled the install behind it.
for name, source in (("viewport", viewport), ("exporter", exporter)):
    assert not re.search(r"seedImportedSceneRuntimeParameters\(\s*(frameVisualPlans->plans|visualLayerPlans|impl_->visualPlanPublication->plans)\s*,",
                         source), f"{name} must not recompile the plan to seed parameters"
assert "importedScenePlanCache.adopt" in viewport, "the plan cache must take the admitted execution"
assert "bool adopt(" in scene or "void adopt(" in scene, "the plan cache needs adopt"

# The old label covered everything up to the next step, so the log blamed a trivial bind.
bind = viewport.index('enterLoopStep ("advance:control-plan-bind");')
assert viewport.index('enterLoopStep ("advance:segment-gather");') > bind
assert "[diagnostic] viewport_set_timeline install phases" in viewport
print("viewport_install_concurrent_source_contract passed")
