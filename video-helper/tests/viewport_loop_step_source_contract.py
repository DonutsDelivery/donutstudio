#!/usr/bin/env python3
# Contract for the Advance sub-step probe (diagnostic only). The render thread names
# the call it enters inside the advance phase, and viewport_info exports that name.
# It pins the export, the label shape, and the labels before the imported-scene and
# audio-feature calls, so a refactor cannot drop them without failing here.
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
viewport = (root / "src/viewport.cpp").read_text()
header = (root / "src/viewport.h").read_text()
main = (root / "src/main.cpp").read_text()

errors = []


def region(text, begin, end):
    start = text.find(begin)
    if start < 0:
        errors.append("missing: " + begin)
        return "", 0
    stop = text.find(end, start + len(begin))
    return (text[start:stop], start) if stop >= 0 else (text[start:], start)


# The stamp lives in the render-loop probe and the phase change resets it.
if "std::atomic<const char*> loopStep" not in viewport:
    errors.append("loopStep is not an atomic string literal pointer")
if "std::atomic<int64_t> loopStepEnterNs" not in viewport:
    errors.append("loopStepEnterNs is missing")
if "const auto enterLoopStep = [&im, &recordSlowLoopStep] (const char* step)" not in viewport:
    errors.append("enterLoopStep stamp is missing")
phase_body, _ = region(viewport, "const auto enterLoopPhase = [&im, &recordSlowLoopStep] (LoopPhase next)",
                       "const auto enterLoopStep")
if "im.loopStep.store (kLoopPhaseNames[(int) next]" not in phase_body:
    errors.append("a phase change does not reset loopStep to the phase name")

# info() exports the step and its age next to the phase fields.
info_body, _ = region(viewport, "ViewportInfo Viewport::info() const", "\n}\n")
for needle in ("vi.loopStep = step;", "vi.loopStepAgeMs ="):
    if needle not in info_body:
        errors.append("info() does not set: " + needle)

for decl in ("std::string loopStep", "double loopStepAgeMs"):
    if decl not in header:
        errors.append("viewport.h does not declare: " + decl)
for key in ("loopStep", "loopStepAgeMs"):
    if '{ "%s", vi.%s }' % (key, key) not in main:
        errors.append("viewport_info RPC reply does not serialize: " + key)

# The 2 s watchdog line names the last step as well as the phase.
if "loop phase %s took %lld ms (last step %s)" not in viewport:
    errors.append("watchdog line does not name the last step")

# Every label is a literal of the form advance:<name>, and all sit inside the advance phase.
advance, advance_at = region(viewport, "enterLoopPhase (LoopPhase::Advance);",
                             "enterLoopPhase (LoopPhase::Composite);")
labels = re.findall(r'enterLoopStep \("([^"]*)"\);', viewport)
if not labels:
    errors.append("no advance sub-step labels found")
bad = [label for label in labels if not re.fullmatch(r"advance:[a-z-]+", label)]
if bad:
    errors.append("labels not in advance:<name> form: " + ", ".join(bad))
outside = re.findall(r'enterLoopStep \("([^"]*)"\);', viewport.replace(advance, ""))
if outside:
    errors.append("labels outside the advance phase: " + ", ".join(outside))
if len(re.findall(r'enterLoopStep \(', advance)) != len(labels):
    errors.append("advance region has a step call that is not a literal label")


def label_before(call, label):
    """True when the label is stamped before the first use of call in the advance phase."""
    at = advance.find(call)
    if at < 0:
        errors.append("advance phase no longer calls: " + call)
        return
    stamp = advance.find('enterLoopStep ("%s");' % label)
    if stamp < 0 or stamp > at:
        errors.append("%s is not stamped before %s" % (label, call))


# The imported scene execution and the audio feature lookups are named before they run.
label_before("prepareVisualImportedSceneLayerAtTime(", "advance:imported-scene-exec")
label_before("previewMixFeatureAt ((double) f / clkFps)", "advance:audio-feature-lookup")

# The render thread installs the advance sub-step atomics as its loopstep sink, inside
# renderLoop and before its loop starts. Scopes elsewhere then name their call.
probe = (root / "src/loop_step_probe.h").read_text()
imported_header = (root / "src/imported_scene_visual_plan_execution.h").read_text()
imported_payload = (root / "src/imported_scene_payload_execution.cpp").read_text()
backend = (root / "src/gpu_backend/backend_opengl.cpp").read_text()
fixture = (root / "src/fixture_scene_renderer.cpp").read_text()
animated = (root / "src/imported_animated_scene_payload_execution.cpp").read_text()
metal = (root / "src/gpu_backend/backend_sokol.mm").read_text()

render_start = viewport.find("void Viewport::renderLoop (int width")
render_loop_head = ""
if render_start >= 0:
    render_loop_head = viewport[render_start:viewport.find("while (running_.load()", render_start)]
if '#include "loop_step_probe.h"' not in viewport:
    errors.append("viewport.cpp does not include loop_step_probe.h")
if ("loopstep::Sink loopSink { &im.loopStep, &im.loopStepEnterNs,\n"
        "                              &im.lastSlowScope, &im.lastSlowScopeMs };") not in render_loop_head:
    errors.append("renderLoop does not name the loop-step atomics as its sink")
if "loopstep::SinkInstall loopSinkInstall (&loopSink);" not in render_loop_head:
    errors.append("renderLoop does not install the sink with a restoring guard")
if "inline thread_local Sink* currentSink" not in probe:
    errors.append("loop_step_probe.h does not declare the thread_local sink as inline")
if '#include "loop_step_probe.h"' not in imported_header:
    errors.append("imported_scene_visual_plan_execution.h does not include loop_step_probe.h")
if '#include "loop_step_probe.h"' not in imported_payload:
    errors.append("imported_scene_payload_execution.cpp does not include loop_step_probe.h")
if '#include "../loop_step_probe.h"' not in backend:
    errors.append("backend_opengl.cpp does not include ../loop_step_probe.h")
if '#include "loop_step_probe.h"' not in fixture:
    errors.append("fixture_scene_renderer.cpp does not include loop_step_probe.h")
if '#include "loop_step_probe.h"' not in animated:
    errors.append("imported_animated_scene_payload_execution.cpp does not include loop_step_probe.h")
if '#include "../loop_step_probe.h"' not in metal:
    errors.append("backend_sokol.mm does not include ../loop_step_probe.h")

scope_call = re.compile(r"loopstep::Scope\s+\w+\s*\((.*?)\);", re.S)


def scope_labels(text):
    return [label for args in scope_call.findall(text)
            for label in re.findall(r'"([^"]*)"', args)]


label_shape = re.compile(r"advance:imported-scene:[a-z-]+|gl:[a-z-]+|metal:[a-z-]+")
header_labels = scope_labels(imported_header)
payload_labels = scope_labels(imported_payload)
backend_labels = scope_labels(backend)
fixture_labels = re.findall(r'"(advance:imported-scene:fixture-[a-z-]+)"', fixture)
animated_labels = scope_labels(animated)
metal_labels = re.findall(r'"(metal:[a-z-]+)"', metal)
for label in header_labels + payload_labels + backend_labels + fixture_labels + animated_labels + metal_labels:
    if not label_shape.fullmatch(label):
        errors.append("scope label not in advance:imported-scene:<name>, gl:<name> or metal:<name> form: " + label)
for label in ("advance:imported-scene:plan-publish", "advance:imported-scene:resolve-compile"):
    if label not in header_labels:
        errors.append("missing plan scope: " + label)
for label in ("advance:imported-scene:cold-admit", "advance:imported-scene:glb-decode",
              "advance:imported-scene:preview-render", "advance:imported-scene:animated-delegate"):
    if label not in payload_labels:
        errors.append("missing payload scope: " + label)
for label in ("gl:fixture-scene-program-compile", "gl:fixture-deformation-compute-compile",
              "gl:fixture-draw-finish", "gl:caustic-coverage-fence-wait"):
    if label not in backend_labels:
        errors.append("missing GL scope: " + label)

# The scene program compile is timed inside the fixtureMutex-held prepare, not before it.
prepare_at = backend.find("NativeFixtureScenePreparation prepareGeometryInstancesImpl(")
lock_at = backend.find("std::lock_guard<std::mutex> lock (fixtureMutex());", prepare_at) if prepare_at >= 0 else -1
compile_at = backend.find('loopstep::Scope sceneCompileScope ("gl:fixture-scene-program-compile")', prepare_at) if prepare_at >= 0 else -1
if prepare_at < 0 or lock_at < 0 or compile_at < lock_at:
    errors.append("gl:fixture-scene-program-compile is not inside the fixtureMutex-held prepare")

# A long imported-scene step is split down to the call that ran long: the
# animated route, the fixture prepare and submit, the Metal shader compile and
# pipeline creation, and the material Frame decode and lease.
for label in ("advance:imported-scene:animated-admit", "advance:imported-scene:animated-evaluate",
              "advance:imported-scene:animated-render"):
    if label not in animated_labels:
        errors.append("missing animated scope: " + label)
for label in ("advance:imported-scene:fixture-prepare", "advance:imported-scene:fixture-submit",
              "advance:imported-scene:fixture-caustic-submit"):
    if label not in fixture_labels:
        errors.append("missing fixture scope: " + label)
for label in ("metal:fixture-scene-shader-compile", "metal:fixture-scene-pipeline-create"):
    if label not in metal_labels:
        errors.append("missing Metal prepare scope: " + label)
for label in ("advance:imported-scene:frame-decode", "advance:imported-scene:frame-lease"):
    if 'loopstep::Scope' not in viewport or '"%s"' % label not in viewport:
        errors.append("missing material Frame scope: " + label)
# The Metal prepare scopes sit inside the fixture backend's prepare, before renderCaustic.
metal_class = metal.find("class MetalFixtureSceneBackend final")
metal_caustic = metal.find("NativeFixtureSceneSubmission renderCaustic(", metal_class)
for label in ("metal:fixture-scene-shader-compile", "metal:fixture-scene-pipeline-create"):
    at = metal.find('"%s"' % label)
    if metal_class < 0 or metal_caustic < 0 or not (metal_class < at < metal_caustic):
        errors.append(label + " is not inside the Metal fixture prepare")

# A frame-work step that does not return is reported as a stall in viewport_info,
# and the last slow step, the last slow scope and the last close wait stay named.
for needle in ("isFrameWorkPhase (phase) && vi.loopStepAgeMs >= kViewportStallReportMs",
               "vi.stalledStep = vi.loopStep;", "viewport render stalled: ",
               "vi.lastSlowStep = impl_->lastSlowStep.load", "vi.lastSlowScope = impl_->lastSlowScope.load",
               "vi.lastCloseWaitStep = lastCloseWaitStep_;"):
    if needle not in info_body:
        errors.append("info() does not report the stall: " + needle)
info_head = info_body[:info_body.find("if (impl_ == nullptr) return vi;")]
if "vi.lastCloseWaitMs = lastCloseWaitMs_;" not in info_head:
    errors.append("info() drops the close wait while no viewport is open")
for key in ("stalledStep", "stalledStepMs", "lastSlowStep", "lastSlowStepMs",
            "lastSlowScope", "lastSlowScopeMs", "lastCloseWaitStep", "lastCloseWaitMs"):
    if '{ "%s", vi.%s }' % (key, key) not in main:
        errors.append("viewport_info RPC reply does not serialize: " + key)
close_body, _ = region(viewport, "void Viewport::close()", "\n}\n")
for needle in ("impl_->loopStep.load", "thread_.join();", "lastCloseWaitMs_ = (double) waitedMs;",
               "close waited %lld ms for the render thread (step %s)"):
    if needle not in close_body:
        errors.append("close() does not record its wait: " + needle)
for needle in ("recordSlowLoopStep (now);",):
    if viewport.count(needle) != 2:
        errors.append("enterLoopPhase and enterLoopStep do not both record the slow step they leave")

if errors:
    print("missing viewport loop-step contracts: " + "; ".join(errors), file=sys.stderr)
    raise SystemExit(1)
print("viewport loop step source contract: PASS")
