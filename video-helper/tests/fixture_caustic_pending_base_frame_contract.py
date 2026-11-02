#!/usr/bin/env python3
# Contract for the pending-caustic overlap: a Preview frame whose caustic dispatch
# programs are still linking renders the base scene instead of holding the whole
# frame, keeps the glass scene's scene-linear contract, and reserves the caustic
# validation and accounting for frames that actually took the caustic submit.
# Blocking use keeps its pending error, so export never changes.
from pathlib import Path
import sys

root = Path(sys.argv[1])
renderer = (root / "src/fixture_scene_renderer.cpp").read_text()
backend = (root / "src/gpu_backend/backend_opengl.cpp").read_text()
errors = []

# A Preview caustic frame must never block the frame on the dispatch compile:
# the acquire passes wait=false and reports pending instead of failing.
if backend.count("arbitgl::hasParallelShaderCompile (gl)") < 2:
    errors.append("per-context availability queries missing")
# The completion poll must not run without KHR_parallel_shader_compile: the
# pname is an invalid enum there and done never leaves GL_FALSE, so a Preview
# would stay Pending for ever and only an input would ever advance it. The
# acquire takes its submitted stages as complete instead -- a driver without
# the extension links synchronously inside the submit.
if "if (cache.parallelCompile == 1)\n                for (const auto& stage : flying->second.stages)" not in backend:
    errors.append("the caustic acquire must poll GL_COMPLETION_STATUS_KHR only when the extension exists")
if "parallelShaderCompileAvailable" in backend:
    errors.append("the shared availability helper must stay gone")
if "if (cache.parallelCompile != 1) wait = true;" in backend:
    errors.append("the acquire must never escalate a pending Preview to a blocking wait")
if "sceneInputs.glassCaustic && submission.rendered && submission.stats.drawCount == 0" not in renderer:
    errors.append("the zero-draw gate must skip a failed submission so its real error stands")
    errors.append("a zero-draw glass frame must never publish")
if "native fixture GPU backend rendered a blocking glass frame with no draws" not in renderer:
    errors.append("the blocking zero-draw path must report the pending-class failure")
if "attempt < 4 && submission.rendered\n                 && submission.stats.drawCount == 0" not in renderer:
    errors.append("a blocking zero-draw caustic submission must retry before failing")
if "renderpasscomposite::kInputs[i] == output && rawTextures[i - 2] != 0" not in backend:
    errors.append("a frame without pooled raw textures must fall back to its receipt descriptor")
if "const bool wait = use!=OpenGlCausticRenderUse::Preview;" not in backend:
    errors.append("the caustic submit must wait only for export")
if "if(!wait&&acquired==OpenGlCausticProgramAcquire::Pending)\n            { result.causticPending=true; return result; }" not in backend:
    errors.append("a pending Preview acquire must return causticPending, not fail")

begin = renderer.find("if (causticProgramsPending)")
end = renderer.find("NativeFixtureSceneSubmission submission;", begin)
if begin < 0 or end < 0:
    errors.append("pending-caustic block not found")
else:
    block = renderer[begin:end]
    if "pending caustic programs for a blocking render" not in block:
        errors.append("blocking use lost its pending error")
    if "output.pending = true" in block:
        errors.append("a Preview must not hold the frame while the dispatch compiles")
    if "++previewPendingResults" in block:
        errors.append("counting a pending result re-imposes the viewport's frame hold")

for needle in (
    "submitScope.emplace (sceneInputs.glassCaustic && !causticProgramsPending",
    "if (sceneInputs.glassCaustic && !causticProgramsPending)",
    "baseInputs.linearColor = true;",
    "sceneInputs.glassCaustic.has_value() && !causticProgramsPending",
):
    if needle not in renderer:
        errors.append("missing overlap guard: " + needle)

# The plain base submit must exist as the fall-through the pending frame takes.
if "submission = backend_.render (\n            snapshot, cache_.native, dimensions.width, dimensions.height, baseInputs)" not in renderer:
    errors.append("base render no longer receives the deferred-glass inputs")

if errors:
    print("fixture caustic pending base-frame contract: " + "; ".join(errors), file=sys.stderr)
    raise SystemExit(1)
print("fixture caustic pending base-frame contract: PASS")
