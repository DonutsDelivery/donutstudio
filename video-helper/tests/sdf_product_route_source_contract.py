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
        # A rejected SDF plan holds the last frame. Closing the channel ends the Metal stream.
        assert "holdFrameForPlanError(planError);" in rejection and "im.wantClose" not in rejection
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
    assert "return hasOperation(visualimportedscenerender::kRenderNodeKind)" in classifier
    assert "(hasOperation(typedscenepass::kSceneKind) && hasOperation(typedscenepass::kNodeKind))" in classifier
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

# A Geometry Core plan that fails to render holds the last presented frame. The
# failure must not reach a closing assignment, which ends the Metal surface channel.
geometry_core = region(viewport, "if (videowire::isGeometryCoreRenderPlan(frameVisualPlans->plans, segment.clipId))",
                       "const auto* importedFramePlan")
assert "holdFrameForPlanError(planError);" in geometry_core
assert "im.wantClose" not in geometry_core
assert "if (spectrumBakePending || frameDeferred)" in viewport
assert "if (im.rendererError == planFrameError) im.rendererError.clear();" in viewport

# A stopped Audio Spectrum field waits for the baked mix after the history wait.
spectrum_header = (root / "src/visual_plan_executor.h").read_text()
assert "inline bool requiresBakedSpectrumAnalysis(" in spectrum_header
assert "Audio Spectrum Field is waiting for the project's baked audio analysis" in geometry_core
assert geometry_core.index("Spectral History Field is waiting") < geometry_core.index(
    "videowire::requiresBakedSpectrumAnalysis(*plan)")
# The wait holds the frame itself. A transition's outgoing layer returns false
# without a hold of its own, so the flag must be set here.
spectrum_wait = region(geometry_core, "videowire::requiresBakedSpectrumAnalysis(*plan))", "stream = nullptr;")
assert "frameDeferred = true;" in spectrum_wait

# Content plan failures hold the frame through holdFrameForPlanError. The hold sets
# frameDeferred, so the tick presents nothing and retries, and it never closes.
hold = region(viewport, "const auto holdFrameForPlanError = [&]", "const auto prepareShaderClip")
assert "planFrameError = error.empty()" in hold and "im.rendererError = planFrameError;" in hold
assert "frameDeferred = true;" in hold and "im.wantClose" not in hold
# A transition's outgoing layer that fails under a hold stops the layer loop.
assert "if (im.wantClose.load() || frameDeferred) break;" in viewport

# A device or resource failure closes the channel only after it names the failure.
# closeForDeviceFailure publishes rendererError before wantClose. The cleanup then
# sends that reason in a final BuffersGone, before the socket closes.
device_close = region(viewport, "const auto closeForDeviceFailure = [&]", "// M6 mod matrix")
assert "std::lock_guard<std::mutex> lock(im.mutex);" in device_close
assert device_close.index("im.rendererError = deviceCloseReason;") < device_close.index("im.wantClose = true;")
assert viewport.count("im.wantClose = true;") == 1
device_reasons = (
    "linear IOSurface buffers could not be reallocated for a new consumer",
    "linear IOSurface buffers could not be allocated for the selected format",
    "shared buffers could not be resized",
    "strict Metal surface could not be resized",
    "strict Metal compositor rejected frame",
    "strict Metal present failed",
    "temporal sampling could not be committed",
)
assert viewport.count("closeForDeviceFailure(") == len(device_reasons)
for reason in device_reasons:
    assert viewport.count(f'closeForDeviceFailure("{reason}"') == 1, reason

# The final BuffersGone carries the reason, capped to fit the Linux receive buffer.
# It goes out before the socket closes.
cleanup = region(viewport, "// A device failure names its reason in this final BuffersGone", "renderer.shutdown();")
assert "gpusurf::closeReasonPrefixBytes (" in cleanup
assert "reinterpret_cast<const uint8_t*> (deviceCloseReason.data()), deviceCloseReason.size());" in cleanup
assert "reasonBytes > 0 ? deviceCloseReason.data() : nullptr, reasonBytes" in cleanup
assert cleanup.index("gpusurf::MsgType::BuffersGone") < cleanup.index("im.fdSocket.close();")

# The macOS consumer keeps the reason from BuffersGone and appends it to the bare EOF
# text. The append waits until after the parse, so a parse error keeps its own text.
mac_surface = (root / "../plugin/Source/MacMetalVideoSurface.mm").read_text()
assert 'static constexpr const char* kProducerClosed = "producer closed the Metal surface channel";' in mac_surface
mac_connect = region(mac_surface, "bool connect (const juce::String& endpoint)", "void disconnect()")
assert "producerReason.clear();" in mac_connect
mac_read = region(mac_surface, "void readMessages()", "void parseMessages()")
assert "error = kProducerClosed;" in mac_read and "producerReason" not in mac_read
mac_buffers_gone = region(mac_surface, "case gpusurf::MsgType::BuffersGone:", "case gpusurf::MsgType::FrameReady:")
assert "reinterpret_cast<const char*> (payload), (int) reasonBytes" in mac_buffers_gone
assert "clearBuffers();" in mac_buffers_gone
mac_pump = region(mac_surface, "bool pumpAndPresent()", "if (connectionLost || ! havePending)")
assert mac_pump.index("parseMessages();") < mac_pump.index("producerReason.isNotEmpty()")
assert "error == kProducerClosed" in mac_pump
