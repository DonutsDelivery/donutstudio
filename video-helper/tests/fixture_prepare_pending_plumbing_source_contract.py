#!/usr/bin/env python3
# Contract for the fixture prepare pending plumbing. A Preview prepare may report
# pending as a retry state. No backend returns pending yet, so this pins the route
# from the backend to the viewport, and the blocking paths that must never see it.
from pathlib import Path
import sys

root = Path(sys.argv[1])
backend = (root / "src/gpu_backend/backend.h").read_text()
renderer = (root / "src/fixture_scene_renderer.cpp").read_text()
renderer_header = (root / "src/fixture_scene_renderer.h").read_text()
glb = (root / "src/glb_native_render_seam.cpp").read_text()
payload_header = (root / "src/imported_scene_payload_execution.h").read_text()
payload = (root / "src/imported_scene_payload_execution.cpp").read_text()
animated_header = (root / "src/imported_animated_scene_payload_execution.h").read_text()
animated = (root / "src/imported_animated_scene_payload_execution.cpp").read_text()
visual = (root / "src/imported_scene_visual_plan_execution.h").read_text()
viewport = (root / "src/viewport.cpp").read_text()
viewport_header = (root / "src/viewport.h").read_text()
main = (root / "src/main.cpp").read_text()
cmake = (root / "CMakeLists.txt").read_text()

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


# Backend seam: the use argument is declared once, and every backend accepts it.
require(backend, "enum class NativeFixturePrepareUse : std::uint8_t { Blocking = 0, Preview };", "backend.h")
require(backend, "NativeFixturePrepareUse use = NativeFixturePrepareUse::Blocking) = 0;", "backend.h base prepare")
require(region(backend, "struct NativeFixtureScenePreparation\n{", "\n};", "backend.h"),
        "bool pending = false;", "NativeFixtureScenePreparation")
for name in ("gpu_backend/backend_opengl.cpp", "gpu_backend/backend_sokol.mm", "gpu_backend/backend_stub.cpp"):
    require((root / "src" / name).read_text(),
            "NativeFixturePrepareUse = NativeFixturePrepareUse::Blocking) override", name)

# Renderer: only Preview asks for pending, and a pending result touches no cache.
require(renderer, "backend_.prepare (snapshot, nativeMaterial, prepareUse)", "fixture_scene_renderer.cpp")
require(renderer, "arbitgpu::NativeFixturePrepareUse::Preview", "fixture_scene_renderer.cpp")
pending_block = region(renderer, "if (preparation.pending)", "if (! preparation.prepared",
                       "fixture_scene_renderer.cpp")
require(pending_block, "output.pending = true;", "renderer pending block")
require(pending_block, "return true;", "renderer pending block")
forbid(pending_block, "cache_", "renderer pending block")
require(region(renderer_header, "struct RenderedFrame\n{", "\n};", "fixture_scene_renderer.h"),
        "bool pending = false;", "RenderedFrame")

# GLB seam: a pending frame publishes no draws, materials, or textures.
glb_block = region(glb, "if (frame.pending)", "GlbNativeFrameReceipt receipt;", "glb_native_render_seam.cpp")
require(glb_block, "output.rendered = std::move (frame);", "GLB pending block")
require(glb_block, "return true;", "GLB pending block")
forbid(glb_block, "receipt.draws", "GLB pending block")

# Payload: the flag is reset on entry, and pending returns before motion commit.
require(region(payload_header, "struct ImportedSceneExecutionReceipt final", "bool valid() const noexcept",
               "imported_scene_payload_execution.h"),
        "bool pending = false;", "ImportedSceneExecutionReceipt")
require(payload, "output.pending = false;", "payload execute entry")
if payload.count("if (candidate.pending)") != 2:
    errors.append("payload candidate branches must each test candidate.pending")
search = 0
for _ in range(2):
    at = payload.find("if (candidate.pending)", search)
    if at < 0:
        break
    block = payload[at:payload.find("clearCallerReceipt(output);", at)]
    require(block, "output.pending = true;", "payload candidate pending block")
    forbid(block, "output = std::move(candidate);", "payload candidate pending block")
    search = at + 1
render_block = region(payload, "if (frame.rendered.pending)", "if (!motion.commit(request.frame.frame",
                      "imported_scene_payload_execution.cpp")
require(render_block, "output.pending = true;", "payload render pending block")
require(render_block, "return true;", "payload render pending block")
forbid(render_block, "motion = {}", "payload render pending block")
forbid(render_block, "motion.commit(", "payload render pending block")


def ordered(text, needles, label):
    # Each needle must appear, and each must follow the one before it by text order.
    at = -1
    for needle in needles:
        found = text.find(needle, at + 1)
        if found < 0:
            errors.append(label + " is missing, or out of order, " + needle)
            return
        at = found


# Static render site: the render, then the pending return, then the only motion commit.
# Motion is never reset between the render and the pending return.
ordered(payload, [
    "seam.renderPreview(admission, request.dimensions, sceneInputs, frame, error)",
    "if (frame.rendered.pending)",
    "output.pending = true;",
    "if (!motion.commit(request.frame.frame",
], "static render pending order")
require(payload, "return reject(error, \"native imported scene export render returned pending\");",
        "static render export refusal")
# A non-parent receipt is cleared only on failure. The unconditional clear is gone.
forbid(payload, "if (!request.parentIntent) clearCallerReceipt(output);", "payload entry")
require(payload, "clearCallerReceiptUnlessKept", "payload entry guard")
if payload.count("keepCallerReceipt = true;") < 3:
    errors.append("payload success and pending exits must keep the caller receipt")
# Animated delegation: a pending animated frame is never staged or published.
ordered(payload, ["if (deformationFrame.pending)", "staging.publish(output);"],
        "payload staging pending order")
if payload.count("if (deformationFrame.pending)") != 2:
    errors.append("payload delegation must test deformationFrame.pending on both routes")

# Animated payload: the preview sites check pending before they publish, export
# refuses it, and the one-leaf motion reset waits until the render has returned.
require(region(animated_header, "struct ImportedAnimatedSceneReceipt final", "bool valid() const noexcept",
               "imported_animated_scene_payload_execution.h"),
        "bool pending = false;", "ImportedAnimatedSceneReceipt")
ordered(animated, [
    "sceneRenderer_.renderPreview(scene_,material",
    "if (rendered.pending)",
    "output.pending=true;",
    "auto issued=std::shared_ptr<NativeSceneAnimationCompositionFrame>",
], "animated composition preview pending order")
require(animated, "camera/light composition export render returned pending", "animated composition export refusal")
ordered(animated, [
    "sceneRenderer_.renderPreview(scene_, material",
    "if (sceneFrame.pending)",
    "renderPending = true;",
], "animated single-leaf preview pending order")
require(animated, "imported animated export render returned pending", "animated single-leaf export refusal")
ordered(animated, [
    "if (motionHistoryReset) runtimeInputs.previousMotion = std::nullopt;",
    "if (!renderFrame()) return false;",
    "if (renderPending)",
    "if (motionHistoryReset) motion_ = {};",
    "motion.commit(request.frame.frame",
], "animated single-leaf motion order")
pre_render = region(animated, "const bool motionHistoryReset", "if (!renderFrame()) return false;",
                    "imported_animated_scene_payload_execution.cpp")
forbid(pre_render, "motion_ = {}", "animated pre-render motion")
forbid(pre_render, "motionOwner_ = motionOwner", "animated pre-render motion")
forbid(animated, "if (!composition && !request.parentIntent) output = {};", "animated entry")
ordered(animated, ["if (!executeSceneAnimationComposition(", "if (output.pending) return true;", "published=true;"],
        "animated composition route pending order")

# Visual plan: pending is a value, the template returns it before validReceipt, and
# the trait keeps receipts without a pending member compiling.
require(visual, "pending = 3", "imported_scene_visual_plan_execution.h enum")
require(visual, "struct HasImportedPendingReceipt", "imported_scene_visual_plan_execution.h trait")
pending_return = "if (receipt.pending) return VisualImportedScenePreparation::pending;"
require(visual, pending_return, "imported_scene_visual_plan_execution.h template")
if visual.find(pending_return) < 0 or visual.find("if (!Execution::validReceipt(receipt))") < visual.find(pending_return):
    errors.append("the pending return must come before validReceipt in the template")

# Viewport: pending holds the frame through frameDeferred and never records an error.
require(viewport, "bool importedPendingThisFrame = false;", "viewport.cpp frame local")
imported = region(viewport, 'enterLoopStep ("advance:imported-scene-plan-exec");',
                  "VisualImportedScenePreparation::rendered", "viewport.cpp imported branch")
require(imported, "VisualImportedScenePreparation::pending)", "viewport.cpp imported branch")
require(imported, "importedPendingThisFrame = true;", "viewport.cpp imported branch")
require(imported, "frameDeferred = true;", "viewport.cpp imported branch")
for needle in ("rejectPlan", "holdFrameForPlanError", "rendererError", "planFrameError"):
    forbid(imported, needle, "viewport.cpp imported pending block")
particle = region(viewport, "prepared == videohelper::importedscene::VisualImportedScenePreparation::pending",
                  "holdFrameForPlanError(sceneError);", "viewport.cpp particle branch")
require(particle, "importedPendingThisFrame = true;", "viewport.cpp particle pending block")
require(particle, "frameDeferred = true;", "viewport.cpp particle pending block")
forbid(particle, "rendererError", "viewport.cpp particle pending block")
# A shader transition's referenced layer that is pending breaks before it sets planError.
transition = region(viewport, "else if (!prepareDecodedDesc(referenced, referencedParams",
                    'planError = "shader transition Layer Source could not produce', "viewport.cpp transition")
require(transition, "if (importedPendingThisFrame) break;", "viewport.cpp transition pending block")

store_at = viewport.find("im.importedScenePending.store(importedPendingThisFrame);")
defer_at = viewport.find("if (frameDeferred)", store_at) if store_at >= 0 else -1
if store_at < 0 or defer_at < 0 or defer_at - store_at > 400:
    errors.append("the pending status store must sit directly before the frame deferral check")
require(viewport, "std::atomic<bool> importedScenePending { false };", "viewport.cpp Impl")
require(viewport, "vi.importedScenePending = impl_->importedScenePending.load();", "viewport.cpp info()")
require(viewport_header, "bool importedScenePending = false;", "viewport.h ViewportInfo")
require(main, '{ "importedScenePending", vi.importedScenePending },', "main.cpp viewport_info")
require(main, "if (receipt.pending)", "main.cpp model payload guard")

# Animated compositor preparation: pending is a value, the template returns it after
# the executed check and before validReceipt, and export refuses it instead.
anim_header = (root / "src/imported_animation_visual_plan_execution.h").read_text()
require(anim_header, "    pending\n};", "imported_animation_visual_plan_execution.h enum")
require(anim_header, "struct HasImportedAnimationPendingReceipt", "imported_animation_visual_plan_execution.h trait")
ordered(anim_header, [
    "if (! executed)",
    "if constexpr (HasImportedAnimationPendingReceipt<typename Execution::Receipt>::value)",
    "if (receipt.pending)",
    "return VisualImportedAnimationPreparation::pending;",
    "if (! execution->validReceipt(receipt))",
], "animation template pending order")
require(anim_header, "imported animation compositor export returned pending", "animation export pending refusal")

# Viewport: the animated pending branch holds through frameDeferred with no plan error.
anim_pending = region(viewport, "VisualImportedAnimationPreparation::pending)",
                      "VisualImportedAnimationPreparation::rejected)", "viewport.cpp animated pending block")
require(anim_pending, "importedPendingThisFrame = true;", "viewport.cpp animated pending block")
require(anim_pending, "frameDeferred = true;", "viewport.cpp animated pending block")
for needle in ("rejectPlan", "holdFrameForPlanError", "rendererError", "planError"):
    forbid(anim_pending, needle, "viewport.cpp animated pending block")

# Animated model payload handler: the pending guard returns before any payload or frame use.
anim_handler = region(main, 'if (method == "model_payload_animation_preview"',
                      'if (method == "gpu_backend_info"', "main.cpp animation handler")
ordered(anim_handler, [
    "g_importedAnimatedScenePayloadExecution.executePreview(request, receipt, error)",
    "if (receipt.pending)",
    "receipt.payload->key().id",
], "main.cpp animation handler pending order")

# Every fixture backend override in the test fakes accepts the use argument.
for name, minimum in (
    ("tests/diffraction_triangle_transport_tests.cpp", 1),
    ("tests/glb_native_render_seam_tests.cpp", 1),
    ("tests/fixture_scene_renderer_tests.cpp", 1),
    ("tests/native_composition_production_join_fixture.h", 2),
    ("tests/imported_scene_visual_plan_execution_tests.cpp", 2),
    ("tests/particle_points_fixture.h", 1),
):
    text = (root / name).read_text()
    if text.count("NativeFixturePrepareUse") < minimum:
        errors.append(name + " has fewer NativeFixturePrepareUse overrides than expected")

require(cmake, "add_test(NAME fixture_prepare_pending_plumbing_source_contract", "CMakeLists.txt")
require(cmake, "tests/fixture_prepare_pending_plumbing_source_contract.py", "CMakeLists.txt")

if errors:
    print("fixture prepare pending plumbing contract: " + "; ".join(errors), file=sys.stderr)
    raise SystemExit(1)
print("fixture prepare pending plumbing source contract: PASS")
