#!/usr/bin/env python3
from pathlib import Path
import sys


root = Path(sys.argv[1])
viewport = (root / "src/viewport.cpp").read_text()
exporter = (root / "src/exporter.cpp").read_text()
execution = (root / "src/sdf_visual_plan_execution.h").read_text()
renderer_header = (root / "src/renderer.h").read_text()
renderer = (root / "src/renderer.cpp").read_text()
metal = (root / "src/gpu_backend/backend_sokol.mm").read_text()
stub = (root / "src/gpu_backend/backend_stub.cpp").read_text()

# Preview and export both enter the same strict renderer seam.
assert viewport.count("prepareVisualSdfLayer(") == 1
assert exporter.count("prepareVisualSdfLayer(") == 1
assert "NativeSdfRenderUse::Preview" in viewport
assert "NativeSdfRenderUse::Export" in exporter

# Native graph sources must be selected before generator dispatch, including
# Layer Source references. These are source-routing checks, not pixel evidence.
def region(source, begin, end):
    start = source.index(begin)
    return source[start:source.index(end, start)]


native_classifiers = []
for source, plans, clip, preparation in (
    (viewport, "frameVisualPlans->plans", "al.seg.clipId", "Preview"),
    (exporter, "visualLayerPlans", "al.seg->clipId", "Export"),
):
    prepared = region(source, "auto prepareDecodedDesc =", "\n        for (const auto& al : act)")
    sdf = prepared.index("prepareVisualSdfLayer(")
    assert sdf < prepared.index("decodeLayer("), f"{preparation}: SDF must precede media decode"
    assert sdf < prepared.index("prepareVisualImportedSceneLayerAtTime("), (
        f"{preparation}: unrelated scene preparation must not reset the SDF layer"
    )
    sdf_branch = region(prepared, "const auto sdfResult =", "const auto volumeResult =")
    assert "VisualSdfPreparation::notPresent" in sdf_branch
    assert "VisualSdfPreparation::rendered" in sdf_branch
    assert "executeVisualLayerPlanForRenderer" not in sdf_branch
    if preparation == "Preview":
        assert "return rejectPlan();" in sdf_branch
        rejection = region(prepared, "const auto rejectPlan =", "fillDescCommon(")
        assert "im.rendererError = planError;" in rejection and "im.wantClose = true;" in rejection
    else:
        assert "return sdfResult == videohelper::sdf::VisualSdfPreparation::rendered;" in sdf_branch
    native_dispatch = region(source[source.index(prepared) + len(prepared):],
                             f"if (hasNativeVisualSource({plans}, {clip}))", "else if")
    assert "prepareDecodedDesc(" in native_dispatch and "planExecuted = true;" in native_dispatch
    assert "0.0, d" not in native_dispatch, f"{preparation}: native scenes need mapped source time"
    referenced = region(source, "const auto referencedKind =", "if (referencedKind ==")
    assert "hasNativeVisualSource(" in referenced
    shader_setup = region(source, "std::set<int> shaderClips;", "// M5: pack" if preparation == "Preview"
                          else "if (! glctx.renderer.setClipShader")
    assert "hasNativeVisualSource(" in shader_setup
    classifier = region(source, "bool hasNativeVisualSource(", "\n}\n")
    assert "videohelper::sdf::hasVisualSdf(plans, clipId)" in classifier
    exclusions = region(classifier, "if (plan == nullptr", "return false;")
    assert "videowire::isTypedParticlePlan(plans, clipId)" in exclusions
    assert "videowire::isGeometryCoreRenderPlan(plans, clipId)" in exclusions
    assert "hasOperation(visualimportedscenerender::kSourceNodeKind)" in classifier
    assert "&& hasOperation(visualimportedscenerender::kRenderNodeKind)" in classifier
    native_classifiers.append(classifier)
    scene = region(prepared, "prepareVisualImportedSceneLayerAtTime(", "const auto importedResult =")
    source_clock = "sourceSec, valueFps," if preparation == "Preview" else "sourceSec, job.fps,"
    assert source_clock in scene
    assert "&params.visualParams" in scene and "&materialFrameResolver" in scene

assert native_classifiers[0] == native_classifiers[1], "preview/export must select the same native graph sources"

assert "videohelper::sdf::hasVisualSdf(job.visualLayerPlans, segment.clipId)" in exporter

# The product layer publishes the validated backend descriptor and its frame
# lease. Metal views are never narrowed into an OpenGL texture name.
assert "const auto descriptor = rendered.nativeFrame->colorTextureDescriptor();" in execution
assert "layer.nativeTextureBackend = descriptor.backend;" in execution
assert "layer.nativeTextureView = descriptor.textureViewHandle;" in execution
assert "layer.nativeTextureDescriptor = descriptor;" in execution
assert "layer.nativeTextureOwner = rendered.nativeFrame;" in execution
assert 'if (descriptor.backend == "opengl")' in execution
assert "layer.texture = rendered.nativeFrame->colorTextureViewHandle();" not in execution
assert "std::string nativeTextureBackend;" in renderer_header
assert "std::uintptr_t nativeTextureView = 0;" in renderer_header

# The Metal compositor validates and samples the exact live sg_view. If native
# Metal consumption is rejected, FrameRenderer must not reinterpret the handle
# through its OpenGL fallback.
assert 'layer.nativeTextureBackend != "metal"' in metal
assert "sg_query_view_state (nativeView)" in metal
assert 'else if (layer.nativeTextureBackend == "metal")' in metal
assert "sourceView.id = static_cast<std::uint32_t> (layer.nativeTextureView);" in metal
assert 'requiresMetalTexture = layers[i].nativeTextureBackend == "metal";' in renderer
assert 'compositorBackend_ = "metal-rejected";' in renderer

# Unsupported builds remain explicit and allocate no CPU-rendered substitute.
assert stub.count("native GPU SDF execution is not compiled in") >= 2
assert "CPU" not in stub[stub.index("class StubSdfExecutionBackend"):]
