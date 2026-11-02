#!/usr/bin/env python3
# Contract for the card-lifecycle item (e) readiness seam: the backend exposes
# a report-only pending-program completion query with the same extension rule as
# the caustic acquire, the renderer surfaces it through the fixture3d facade
# beside previewCausticPending, and the viewport's idle gate polls it while a
# Preview caustic is pending so a completed link wakes one adopting frame
# instead of spending kMaxCausticWakeFrames first.
from pathlib import Path
import sys

root = Path(sys.argv[1])
backend = (root / "src/gpu_backend/backend_opengl.cpp").read_text()
header = (root / "src/gpu_backend/fixture_glass_caustics_opengl.h").read_text()
renderer = (root / "src/fixture_scene_renderer.cpp").read_text()
renderer_header = (root / "src/fixture_scene_renderer.h").read_text()
viewport = (root / "src/viewport.cpp").read_text()
errors = []

if "bool takeOpenGlCausticProgramCompletion()" not in header:
    errors.append("the readiness query must be declared beside prepareOpenGlGlassCausticPrograms")
q_begin = backend.find("bool takeOpenGlCausticProgramCompletion()\n{")
q_end = backend.find("OpenGlCausticProgramReadiness prepareOpenGlGlassCausticPrograms", q_begin)
if q_begin < 0 or q_end < 0:
    errors.append("the backend readiness query body is missing")
else:
    body = backend[q_begin:q_end]
    for needle, why in (
        ("std::lock_guard<std::mutex> lock (fixtureMutex());",
         "the query must take fixtureMutex like the acquire"),
        ("GL_COMPLETION_STATUS_KHR",
         "the query must poll the completion status"),
        ("if (cache.parallelCompile != 1) continue;",
         "the query must use the acquire's extension-absent rule"),
        ("if (stage.loadedBinary) continue;",
         "the query must skip stages loaded from binary"),
        ("completionReported",
         "the query must latch one true per completed set"),
        ('loopstep::Scope pollScope ("gl:caustic-completion-poll");',
         "the poll must keep its loop-step probe"),
    ):
        if needle not in body:
            errors.append(why)
    for forbidden, why in (
        ("inFlight.erase",
         "the query must not adopt the set; the next acquire does"),
        ("finishCausticStage",
         "the query must not finish stages"),
        ("OpenGlCausticProgramSet",
         "the query must not publish programs"),
    ):
        if forbidden in body:
            errors.append(why)
if "bool completionReported = false;" not in backend:
    errors.append("the per-context caustic cache must carry the completion latch")

if "bool takePendingCausticProgramCompletionOnThisThread() noexcept;" not in renderer_header:
    errors.append("the fixture3d facade must declare the readiness seam")
impl_at = renderer.find("bool takePendingCausticProgramCompletionOnThisThread() noexcept\n{")
if impl_at < 0:
    errors.append("the fixture3d facade must define the readiness seam")
else:
    impl = renderer[impl_at:impl_at + 420]
    if "arbitgpu::takeOpenGlCausticProgramCompletion();" not in impl:
        errors.append("the facade must reach the backend query")
    if "#if ! ARBIT_HAVE_METAL_BACKEND && ARBIT_HAVE_OPENGL_SDF_BACKEND && ! defined(__APPLE__)" not in impl:
        errors.append("the facade must guard the GL query like the renderer's caustic program block")
    if "return false;" not in impl:
        errors.append("the facade must stay false where there is no GL link to poll")

poll_at = viewport.find("videorender::fixture3d::takePendingCausticProgramCompletionOnThisThread()")
if poll_at < 0:
    errors.append("the viewport idle gate must poll the readiness seam")
elif viewport.count("takePendingCausticProgramCompletionOnThisThread") != 1:
    errors.append("the seam must be polled in exactly one place, once per settled evaluation")
else:
    before = viewport[max(0, poll_at - 460):poll_at]
    after = viewport[poll_at:poll_at + 760]
    if "previewCausticPendingOnThisThread()" not in before:
        errors.append("the poll must only run while a Preview caustic is pending")
    if "const bool settled = frameSettled" not in after:
        errors.append("the poll must sit in the idle-gate settled evaluation")
    if "&& ! causticProgramsCompleted" not in after:
        errors.append("a completed link must enter the settled expression as its own wake term")

if errors:
    print("fixture caustic completion wake contract: " + "; ".join(errors), file=sys.stderr)
    raise SystemExit(1)
print("fixture caustic completion wake contract: PASS")
