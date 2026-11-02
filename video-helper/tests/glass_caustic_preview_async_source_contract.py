#!/usr/bin/env python3
# Contract for the asynchronous Preview caustic. A Preview frame submits the raster, trace
# and deposition with a fence and never waits on it: later frames poll the fence with
# glGetSynciv, read the copied splat counters only once it has signalled, and resolve the
# spectrum of the same schedule. Export keeps its counter mapping and resolve fence.
from pathlib import Path
import re
import sys

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
backend = (root / "src/gpu_backend/backend_opengl.cpp").read_text()
header = (root / "src/gpu_backend/fixture_glass_caustics_opengl.h").read_text()
loader = (root / "src/gl_loader.cpp").read_text()
cmake = (root / "CMakeLists.txt").read_text()
tests = (root / "tests/glass_caustic_contract_tests.cpp").read_text()
rough_cases = (root / "tests/support/authored_rough_caustic_renderer_cases.h").read_text()
gl_tests = (root / "tests/opengl_fixture_scene_backend_tests.cpp").read_text()

errors = []


def require(text, needle, label):
    if needle not in text:
        errors.append(label + " is missing " + needle)


def forbid(text, needle, label):
    if needle in text:
        errors.append(label + " must not contain " + needle)


def region(text, begin, end, label):
    start = text.find(begin)
    if start < 0:
        errors.append(label + " is missing " + begin)
        return ""
    stop = text.find(end, start + len(begin))
    if stop < 0:
        errors.append(label + " is missing " + end)
        return ""
    return text[start:stop]


def forbid_waiting_sync(text, label):
    # Only a zero timeout polls. Any other ClientWaitSync blocks the render thread.
    for match in re.finditer(r"ClientWaitSync\s*\(([^;]*)\)", text):
        arguments = [part.strip() for part in match.group(1).split(",")]
        if len(arguments) < 3 or arguments[-1] not in ("0", "0u", "0ull"):
            errors.append(label + " waits on the GPU: " + match.group(0))


# The budget decision is pure and sits beside the static GL limit admission.
admission = header.find("inline OpenGlCausticAdmissionDecision decideOpenGlCausticAdmission(")
if admission < 0:
    admission = header.find("decideOpenGlCausticAdmission(")
budget = header.find("inline bool decidePreviewCausticOverBudget(double elapsedMs,double budgetMs)")
if admission < 0 or budget < 0 or budget < admission:
    errors.append("decidePreviewCausticOverBudget must follow decideOpenGlCausticAdmission in the header")
for needle in ("inline constexpr double kPreviewCausticSubmitBudgetMilliseconds=250;",
               "inline constexpr unsigned kPreviewCausticOverBudgetSteps=2;",
               "inline constexpr unsigned kPreviewCausticRetryWithinSteps=120;",
               "{return budgetMs>0&&!(elapsedMs<=budgetMs);}",
               "{return \"caustic submit over budget (\"+previewCausticMilliseconds(elapsedMs)+\")\";}",
               "enum class OpenGlPreviewCausticLevel : std::uint8_t { Full=0, Reduced, Off };",
               "inline bool stepPreviewCausticBudget(OpenGlPreviewCausticBudget& state,double elapsedMs,double nowMs,",
               "inline bool retryPreviewCausticBudget(OpenGlPreviewCausticBudget& state,double nowMs)"):
    require(header, needle, "preview caustic budget")
schedule_key = region(header, "inline bool operator==(const OpenGlCausticScheduleKey& a,", "\n}\n", "schedule key")
for field in ("context", "rendererGeneration", "width", "height", "scheduleIdentity"):
    require(schedule_key, "a." + field + "==b." + field, "schedule key")
forbid(schedule_key, "timeSeconds", "schedule key")

# The counter read refuses before the fence signals, so its mapping never waits.
counters = region(backend, "bool readPreviewCausticCountersAfterFence(", "\n}\n", "late counter read")
signalled = counters.find("if(signalled!=GL_SIGNALED)return false;")
mapped = counters.find("MapBufferRange(")
if signalled < 0 or mapped < 0 or mapped < signalled:
    errors.append("the late counter read must check GL_SIGNALED before its MapBufferRange")
require(counters, "syncStatus(job.fence,GL_SYNC_STATUS,1,nullptr,&signalled);", "late counter read")
forbid_waiting_sync(counters, "late counter read")

# The poll reads the fence status and nothing else that waits.
poll = region(backend, "OpenGlPreviewCausticPoll pollPreviewCausticJobUnlocked(", "\n}\n", "caustic poll")
require(poll, "loopstep::Scope pollScope(\"gl:caustic-submit-poll\");", "caustic poll")
# The declaration stays function-scoped so signalled outlives the timing scope
# that names this status read; the read form itself is the invariant.
require(poll, "int signalled=0;", "caustic poll")
require(poll, "syncStatus(job.fence,GL_SYNC_STATUS,1,nullptr,&signalled);", "caustic poll")
require(poll, "if(signalled!=GL_SIGNALED)", "caustic poll")
require(poll, "readPreviewCausticCountersAfterFence(gl,syncStatus,job,overflow,gatherWork)", "caustic poll")
require(poll, "if(usable&&current&&job.schedule==*current&&job.spectrum[0]&&job.spectrum[1])", "caustic poll")
require(poll, "caustic preview submit over budget: %s still on the GPU after %.0f ms", "caustic poll")
require(poll, "if(previewCausticBudgetLogDue())", "caustic poll")
forbid(poll, "MapBufferRange", "caustic poll")
forbid(poll, "glFinish", "caustic poll")
forbid(poll, "causticFence(", "caustic poll")
forbid_waiting_sync(poll, "caustic poll")
# A query result is read only once its availability reports it landed.
signal_at = poll.find("if(signalled!=GL_SIGNALED)")
result_at = poll.find("queryResult(job.stamps[i],GL_QUERY_RESULT,&at[i])")
if result_at < 0 or result_at < signal_at:
    errors.append("the poll must read timestamps only after the fence signalled")
require(poll, "if(ready(i)&&queryResult)", "caustic poll")
require(backend, "bool previewCausticBudgetLogDue()", "caustic poll")
require(backend, "if(logged&&now-*logged<std::chrono::seconds(1))return false;", "caustic poll")

# The caustic render: counters, submission and resolve.
render = region(backend, "NativeFixtureSceneSubmission renderOpenGlGlassCausticScene(", "\n#endif\n}\n", "caustic render")
splat_counters = region(render, "std::uint32_t overflow=0,gatherWork=0;\n            if(!preview)\n",
                        "if(overflow&gc::opengl::kSplatOverflowKeySlots)", "splat counters")
export_counters, _, preview_counters = splat_counters.partition("            else\n")
require(export_counters, "loopstep::Scope counterWaitScope(\"gl:caustic-splat-counter-wait\");", "export splat counters")
require(export_counters, "MapBufferRange(GL_SHADER_STORAGE_BUFFER,", "export splat counters")
require(preview_counters, "copyBuffer(GL_COPY_READ_BUFFER,GL_COPY_WRITE_BUFFER,", "Preview splat counters")
forbid(preview_counters, "MapBufferRange", "Preview splat counters")
forbid(preview_counters, "causticFence(", "Preview splat counters")

submission = region(render, "// The submission is fenced and left on the GPU.", "resources->previewCausticJob=std::move(job);",
                    "Preview submission")
require(submission, "const auto fence=gl.FenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE,0);", "Preview submission")
require(submission, "glFlush();", "Preview submission")
require(submission, "job.counters=previewCounters;previewCounters=0;", "Preview submission")
require(submission, "job.schedule=*previewSchedule;", "Preview submission")
forbid(submission, "MapBufferRange", "Preview submission")
forbid(submission, "causticFence(", "Preview submission")
forbid(submission, "glFinish", "Preview submission")
forbid_waiting_sync(submission, "Preview submission")

resolve = region(render, "// Only validated GPU spectral storage is resolved into product Color.", "NativeRawPassPixels color;",
                 "caustic resolve")
require(resolve, "if(resolveCaustic)", "caustic resolve")
require(resolve, "gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,2,resources->previewSpectrumBuffers[0]);", "caustic resolve")
require(resolve, "if(!preview)\n        {\n            loopstep::Scope resolveWaitScope(\"gl:caustic-resolve-wait\");\n"
                 "            if(!causticFence(gl,result.error))return fail(result.error);\n        }\n        else glFlush();",
        "caustic resolve")
if resolve.count("causticFence(") != 1:
    errors.append("caustic resolve must hold exactly the one export resolve fence")

# Export keeps a fence after each stage; Preview fences only bands past its pixel cap.
require(render, "if(plan.stageFences||(band&&plan.bandFences))return causticFence(gl,result.error);", "stage fences")
require(render, "const bool trace=!preview||(!spectrumReused&&!previewPoll.inFlight&&!previewPoll.declined&&!resources->previewCausticJob);", "caustic render")
require(render, "previewPoll=pollPreviewCausticJobUnlocked(*resources,&*previewSchedule);", "caustic render")
require(render, "const bool completedQuery=trace&&!preview;", "caustic render")
require(render, "previewCausticOverBudgetReason(previewPoll.elapsedMs)", "caustic render")
# Every Preview base, with or without the caustic, flushes instead of its final glFinish.
if render.count("baseRuntime.previewUse=preview;baseRuntime.previewCallerWaits=preview;") != 2:
    errors.append("caustic render: the caustic base and the fallback base must both skip the Preview wait")
forbid(render, "previewCallerWaits=false", "caustic render")
require(backend, "        if (!(runtimeInputs.previewUse && runtimeInputs.previewCallerWaits))\n        {\n"
                 "            loopstep::Scope drawFinishScope (\"gl:fixture-draw-finish\");\n            glFinish();\n        }\n"
                 "        else glFlush();", "scene draw")
require(render, "const auto level=openGlPreviewCausticBudgetLevel(previewBudgetReason);", "caustic budget ladder")
require(render, "if(smaller)scheduling.entrySubdivisions=smaller;", "caustic budget ladder")
require(backend, "observeOpenGlPreviewCausticBudget(", "caustic budget ladder")
for name in ("stepPreviewCausticBudget", "retryPreviewCausticBudget"):
    if not re.search(r"\b" + name + r"\s*\(state,", backend):
        errors.append("caustic budget ladder is missing a call of " + name)

# Scene resources release an unfinished submission with the context.
require(backend, "void releasePreviewCausticJobUnlocked() const noexcept", "scene resources")
release = region(backend, "void releasePreviewCausticJobUnlocked() const noexcept", "\n    }\n", "scene resources")
require(release, "DeleteSync(", "scene resources")

# The driver is named once at context creation.
require(loader, "static std::atomic<bool> contextLogged { false };", "GL context log")
require(loader, "\"[gl] context GL_RENDERER=%s GL_VERSION=%s\\n\"", "GL context log")

require(tests, "void previewCausticSubmitBudget()", "glass caustic tests")
require(tests, "previewCausticSubmitBudget();causticLimitAdmission();previewCausticPlan();", "glass caustic tests")
require(tests, "\"caustic submit over budget (21437 ms)\"", "glass caustic tests")
# The native GL Preview tests poll a bounded number of frames for the caustic, and read
# receipts from Export only. The test waits between frames; the Preview never does.
require(rough_cases, "for (unsigned attempt = 0; attempt < std::max(1u, publication.frames); ++attempt)", "rough caustic cases")
require(rough_cases, "frame.nativeFrame && frame.causticDegradation.empty()", "rough caustic cases")
require(gl_tests, "publication.receipts=false;publication.waitForGpu=[] {glFinish();};", "OpenGL rough caustic test")
require(gl_tests, "BackendTarget::OpenGl,read,error,publication);", "OpenGL rough caustic test")
require(cmake, "add_test(NAME glass_caustic_preview_async_source_contract", "CMake")

if errors:
    for error in errors:
        print("FAIL: " + error)
    sys.exit(1)
print("glass caustic preview async source contract: PASS")
