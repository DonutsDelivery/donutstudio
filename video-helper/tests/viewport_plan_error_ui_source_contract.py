#!/usr/bin/env python3
"""Pins how a rejected visual plan and a refused module placement reach the user.

A rejected plan holds the last frame. The helper publishes the reason and the clip
in viewport_info, and the editor shows it on the viewport, the graph status line,
the island badge and the selected-node diagnostics. A refused Place keeps its
reason in the Modules tab beside the button.
"""
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
plugin = root / "../plugin/Source"
viewport = (root / "src/viewport.cpp").read_text()
viewport_h = (root / "src/viewport.h").read_text()
main_cpp = (root / "src/main.cpp").read_text()
content = (plugin / "VideoEditor/VideoEditorContent.cpp").read_text()
window = (plugin / "VideoEditor/VideoEditorWindow.h").read_text()
component = (plugin / "VideoViewportComponent.h").read_text()
patch_editor = (plugin / "graph/PatchEditor.h").read_text()
graph_h = (plugin / "VideoEditor/VideoVisualGraphEditor.h").read_text()
graph_cpp = (plugin / "VideoEditor/VideoVisualGraphEditor.cpp").read_text()
strings = json.loads((root / "../tools/i18n/strings.json").read_text(encoding="utf-8"))
errors = []


def require(source, needle, label):
    if needle not in source:
        errors.append(label + ": missing " + repr(needle))


def forbid(source, needle, label):
    if needle in source:
        errors.append(label + ": must not contain " + repr(needle))


def ordered(source, needles, label):
    at = -1
    for needle in needles:
        found = source.find(needle, at + 1)
        if found < 0:
            errors.append(label + ": missing or out of order " + repr(needle))
            return
        at = found


def region(source, begin, end, label):
    start = source.find(begin)
    if start < 0:
        errors.append(label + ": missing region start " + repr(begin))
        return ""
    stop = source.find(end, start + len(begin))
    if stop < 0:
        errors.append(label + ": missing region end " + repr(end))
        return ""
    return source[start:stop]


# Helper: the hold names its reason and its clip, and a presented frame ends both.
require(viewport_h, "std::string planError;", "viewport.h ViewportInfo")
require(viewport_h, "int planErrorClipId = -1;", "viewport.h ViewportInfo")
require(viewport, "vi.planError = impl_->planError;", "viewport.cpp info()")
require(viewport, "vi.planErrorClipId = impl_->planErrorClipId;", "viewport.cpp info()")
hold = region(viewport, "const auto holdFrameForPlanError = [&](const std::string& error)",
              "const auto prepareShaderClip", "viewport.cpp holdFrameForPlanError")
ordered(hold, [
    "if (geometryPendingThisFrame())",
    "return;",
    "im.planError = planFrameError;",
    "im.planErrorClipId = planLayerClipId;",
], "viewport.cpp hold publishes the reason after the pending deferral")
ordered(viewport, [
    "int planLayerClipId = -1;",
    "const auto holdFrameForPlanError",
    "planLayerClipId = al.seg.clipId;",
], "viewport.cpp layer clip tracked for plan errors")
cleared = region(viewport, "if (!planFrameError.empty())", "planFrameError.clear();",
                 "viewport.cpp presented-frame clear")
require(cleared, "im.planError.clear();", "viewport.cpp presented-frame clear")
require(cleared, "im.planErrorClipId = -1;", "viewport.cpp presented-frame clear")
require(viewport, "A reopened viewport starts with no rejected plan", "viewport.cpp reopen reset")
require(main_cpp, '{ "planError", vi.planError },', "main.cpp viewport_info")
require(main_cpp, '{ "planErrorClipId", vi.planErrorClipId },', "main.cpp viewport_info")

# Editor poll: the reason is read before the HUD gate, so a hidden HUD still shows it.
update = region(content, "void VideoEditorContent::updateGpuReadout()", "using HudLine",
                "VideoEditorContent.cpp updateGpuReadout")
ordered(update, [
    "sc->viewportInfo(err, 500)",
    'o->getProperty("planErrorClipId")',
    'o->getProperty("planError").toString()',
    "noteRenderFailure(named, message, rejections.syncedFingerprint(), 2)",
    "rejections.rejectionAt(playheadSec, rejected)",
    "applyRenderRejection(vp, clipId, reason);",
    "if (! hudUserVisible_)",
], "VideoEditorContent.cpp reads planError before the HUD gate")
apply_fn = region(content, "void VideoEditorContent::applyRenderRejection(", "void VideoEditorContent::updateGpuReadout()",
                  "VideoEditorContent.cpp applyRenderRejection")
require(apply_fn, "vp->setFrameNotice(reason);", "VideoEditorContent.cpp applyRenderRejection")
require(apply_fn, "visualGraphEditor->setRenderRejection(clipId, reason);", "VideoEditorContent.cpp applyRenderRejection")
require(window, "void applyRenderRejection(VideoViewportComponent* vp, int clipId, const juce::String& reason);",
        "VideoEditorWindow.h declaration")

# Viewport banner: painted on every rung with the HUD, and independent of the HUD toggle.
require(component, "void setFrameNotice (const juce::String& reason, FrameNoticeKind kind", "VideoViewportComponent.h")
paint = region(component, "void paintHud (juce::Graphics& g)", "void paintFrameNotice", "VideoViewportComponent.h paintHud")
ordered(paint, ["paintFrameNotice (g);", "paintMetricsHud (g);"], "VideoViewportComponent.h paintHud")
gl = region(component, "void drawHudGL (int fbW, int fbH, double scale)", "juce::createOpenGLGraphicsContext",
            "VideoViewportComponent.h drawHudGL")
require(gl, "frameNotice_.isEmpty()", "VideoViewportComponent.h drawHudGL draws for a notice with the HUD off")
banner = region(component, "void paintFrameNotice (juce::Graphics& g)", "void paintMetricsHud",
                "VideoViewportComponent.h paintFrameNotice")
for token in ("ColorPalette::accentRed", "ColorPalette::textPrimary", "M::radiusSmall", "M::paddingSmall",
              'TRANS ("This frame is not rendering")'):
    require(banner, token, "VideoViewportComponent.h banner uses design tokens")

# Graph status line: a passing compile does not hide the helper's verdict.
require(patch_editor, "void setRenderRejection (const juce::String& reason)", "PatchEditor.h")
status = region(patch_editor, "if (cg->runnable && renderRejection_.isNotEmpty())", "// A5: append host-supplied",
                "PatchEditor.h refreshDiagnosticsCurrent")
require(status, 'msg = TRANS ("Not rendering") + ": " + renderRejection_;', "PatchEditor.h status line")

# Video graph editor: island badge, tooltip, selected-node line, and the clip match.
require(graph_h, "decorateRenderRejection(node, model);", "VideoVisualGraphEditor.h node decorator")
badge = region(graph_cpp, "void VideoVisualGraphEditor::decorateRenderRejection(", "void VideoVisualGraphEditor::setRenderRejection(",
               "VideoVisualGraphEditor.cpp decorateRenderRejection")
for token in ("kinds::Island", 'model.badge = "! "', "ColorPalette::accentRed", "model.tooltip = TRANS(\"Not rendering\")",
              "editor.layerDepth() != 0"):
    require(badge, token, "VideoVisualGraphEditor.cpp island badge")
sync = region(graph_cpp, "void VideoVisualGraphEditor::syncRenderRejection()", "void VideoVisualGraphEditor::timerCallback()",
              "VideoVisualGraphEditor.cpp syncRenderRejection")
for token in ("selection.objectId == renderRejectionClipId", "editor.setRenderRejection(wanted);",
              "editor.refreshNodeDecoration(node.id);"):
    require(sync, token, "VideoVisualGraphEditor.cpp syncRenderRejection")
timer = region(graph_cpp, "void VideoVisualGraphEditor::timerCallback()", "selectedPreview.setTooltip(shownText);",
               "VideoVisualGraphEditor.cpp timerCallback")
require(timer, "syncRenderRejection();", "VideoVisualGraphEditor.cpp timerCallback")
require(timer, 'TRANS("Not rendering") + ": " + appliedRenderRejection', "VideoVisualGraphEditor.cpp selected-node line")

# Modules tab: every refusal path keeps its reason beside Place until the next action.
place = region(graph_cpp, "auto VideoVisualGraphEditor::placeModuleWithOutcome(", "void VideoVisualGraphEditor::showModuleMenu(",
               "VideoVisualGraphEditor.cpp placeModuleWithOutcome")
if place.count("postPlacementRefusal(") != 4:
    errors.append("placeModuleWithOutcome must report through postPlacementRefusal on all four exits, found "
                  + str(place.count("postPlacementRefusal(")))
require(place, "postPlacementRefusal(error, false);", "placeModuleWithOutcome unplaceable module")
require(place, "postPlacementRefusal(error.isNotEmpty()", "placeModuleWithOutcome helper refusal")
placement_refusal = region(graph_cpp, "void VideoVisualGraphEditor::postPlacementRefusal(",
                           "void VideoVisualGraphEditor::syncPlacementRefusal()",
                           "VideoVisualGraphEditor.cpp postPlacementRefusal")
ordered(placement_refusal, ["moduleBrowser.showPlacementRefusal(reason);", "placementRefusalTracker.post("],
        "VideoVisualGraphEditor.cpp postPlacementRefusal shows the refusal before the tracker post")
refresh = region(graph_h, "void refreshButtons()\n    {\n        const int index = moduleIndexForRow(list.getSelectedRow());",
                 "placeButton.setDescription(availability);", "VideoVisualGraphEditor.h module browser refreshButtons")
require(refresh, "refused ? placementRefusal : availability", "module browser shows the refusal in the status line")
require(refresh, "refused ? ColorPalette::accentRed : ColorPalette::textDim", "module browser refusal colour")
click = region(graph_h, "placeButton.onClick = [this]", "addAndMakeVisible(placeButton);", "module browser Place click")
ordered(click, ["showPlacementRefusal({});", "onPlace(*package);"], "module browser Place click clears first")
require(graph_h, "void selectedRowsChanged(int) override\n    {\n        placementRefusal.clear();",
        "module browser selection clears the refusal")

# A kernel compile says so; only the helper's judgeStep verdict moves it out of the stall report.
require(viewport, "if (verdict == kernelcompile::StepVerdict::Stalled)", "viewport.cpp keeps the stall report")
require(viewport, "vi.kernelCompiling = verdict == kernelcompile::StepVerdict::Compiling;", "viewport.cpp names the compile")
require(banner, "g.setColour (compiling ? ColorPalette::textPrimary : noticeColour);", "the compile heading reads against the panel")
require(banner, "g.fillRect (getLocalBounds());", "the compile dims the old frame")
require(main_cpp, '{ "kernelCompiling", vi.kernelCompiling }', "viewport_info carries the compile")
require((root / "src/pathtrace/devices/device_optix_host.cpp").read_text(),
        "compiling.emplace();", "the OptiX device marks a wait on its compile")
require(viewport, "previewWaiting().load()", "a preview holding its frame shows the compile")
require(viewport, "previewStopped().load()", "a cancelled compile reaches the notice")
require(content, 'o->getProperty("kernelBuildStopped")', "the notice says the preview is waiting")
require(banner, "g.drawText (buttonText, buttonBounds", "the notice paints its own button: no child shows on the zero-copy rung")
require((root / "src/pathtrace/devices/device_optix_host.cpp").read_text(), "if (gate.stopped())",
        "a preview after Cancel waits instead of trying the device")
ordered(content, ('rendererError.isEmpty()\n            && static_cast<bool>(o->getProperty("kernelCompiling"))',
                  "FrameNoticeKind::compilingKernel"), "a rendererError or rejection outranks the compile notice")
require(banner, 'TRANS ("Preparing the renderer for this graphics driver")', "the banner names the compile")

require(main_cpp, '{ "scoreDropWarning", vi.scoreDropWarning }', "viewport_info carries the score warning")
ordered(content, ('o->getProperty("scoreDropWarning")', "FrameNoticeKind::scoreLeftOut"), "the score warning takes the banner")
require(banner, 'TRANS ("Part of the score is left out")', "the banner names the score warning")

# Every new string is registered for translation.
for text in ("This frame is not rendering", "Not rendering", "Preparing the renderer for this graphics driver",
             "Part of the score is left out", "The renderer isn't ready", "Start again",
             "You cancelled preparing it, so the preview is waiting. Start again when you want the preview.",
             "This happens once per update and can take a few minutes. You can keep working.",
             "Select a video layer first, then place the module.", "The module could not be placed."):
    if text not in strings:
        errors.append("tools/i18n/strings.json missing " + repr(text))

if errors:
    print("viewport plan error UI source contract failed:\n  " + "\n  ".join(errors), file=sys.stderr)
    raise SystemExit(1)
print("viewport plan error UI source contract: PASS")
