#!/usr/bin/env python3
"""Pins the live viewport's smaller imported scene render and keeps export at canvas size.

Usage: imported_scene_preview_extent_source_contract.py <video-helper dir>
"""
from pathlib import Path
import re
import sys

root = Path(sys.argv[1]).resolve()
src = root / "src"
header = (src / "preview_render_extent.h").read_text()
execution = (src / "imported_scene_visual_plan_execution.h").read_text()
viewport = (src / "viewport.cpp").read_text()
telemetry = (src / "visual_plan_telemetry.h").read_text()
telemetry_json = (src / "visual_plan_telemetry_json.h").read_text()
hud = (root.parent / "plugin/Source/VideoEditor/VideoEditorContent.cpp").read_text()
strings = (root.parent / "tools/i18n/strings.json").read_text()

errors = []

def need(ok, message):
    if not ok:
        errors.append(message)

# The pure fit rule lives in one header with no GL or project dependencies.
need("inline Extent fitToPresentation (int canvasWidth, int canvasHeight," in header,
     "fitToPresentation missing")
need("kFloorShortSide = 64" in header, "64 px short-side floor missing")
need("inline std::uint32_t zoomBucket (double zoom)" in header, "zoomBucket missing")
need("#include \"" not in header, "preview_render_extent.h must stay free of project includes")

# The scene prepare step renders at the requested extent, for Preview only, and keeps
# the canvas or pass extent wherever the render has to match them exactly.
need(re.search(r"const bool smallerPreview = previewExtent != nullptr && previewExtent->valid\(\)\s*"
               r"&& use == NativeImportedSceneRenderUse::Preview\s*"
               r"&& !compiled->typedScenePassExtent && !compiled->passComposite && !solidOverlay\s*"
               r"&& previewExtent->width <= static_cast<std::uint32_t>\(width\)\s*"
               r"&& previewExtent->height <= static_cast<std::uint32_t>\(height\)", execution),
     "smallerPreview must be Preview-only, never larger than the canvas, and skip pass and Solid 3D renders")
need(re.search(r"const auto renderExtent = smallerPreview\s*"
               r"\? renderpassoutput::Extent \{ previewExtent->width, previewExtent->height \}\s*"
               r": compiled->typedScenePassExtent\.value_or\(", execution),
     "renderExtent must take the preview extent only when smallerPreview")
need("request.dimensions = {renderExtent.width, renderExtent.height};" in execution,
     "the native request must render at renderExtent")
need(len(re.findall(r"const previewextent::Extent\* previewExtent = nullptr", execution)) == 1
     and "previewextent::Extent* renderedExtent = nullptr" in execution,
     "prepare must default to no preview extent")
need(re.search(r"if \(renderedExtent != nullptr && use == NativeImportedSceneRenderUse::Preview\)\s*"
               r"\*renderedExtent = \{ renderExtent\.width, renderExtent\.height \};", execution),
     "the extent actually rendered must be reported for Preview only")

# The viewport computes the extent from the canvas, the presentation framebuffer
# and the zoom bucket, and hands it to both imported scene prepare calls.
need(re.search(r"previewSceneExtent = videohelper::previewextent::fitToPresentation \(\s*"
               r"cw > 0 \? cw : fbW, ch > 0 \? ch : fbH, fbW, fbH,\s*"
               r"videohelper::previewextent::zoomBucket \(viewZoom_\.load\(\)\)\);", viewport),
     "viewport must fit the canvas into the framebuffer with the zoom bucket")
need(len(re.findall(r"&previewSceneExtent, &(?:importedScene|particleScene)RenderedExtent,", viewport)) == 2,
     "both viewport imported scene prepare calls must pass the preview extent")
need(len(re.findall(r"recordImportedScenePreviewExtent \(", viewport)) == 2,
     "both viewport prepare calls must record the rendered extent")
need(re.search(r"bucketChanged && impl_ != nullptr\)\s*impl_->contentRevision\.fetch_add \(1\)", viewport),
     "a zoom bucket change must wake the idle gate so a paused scene re-renders once")

# Export and the composite probe never see the preview extent.
users = sorted(p.name for p in src.iterdir()
               if p.suffix in (".cpp", ".h", ".mm") and "previewextent" in p.read_text())
allowed = ["imported_scene_visual_plan_execution.h", "preview_render_extent.h", "viewport.cpp"]
need(users == allowed, f"only {allowed} may use previewextent, found {users}")
for name in ("exporter.cpp", "main.cpp", "composite_probe_contract.h"):
    text = (src / name).read_text()
    need("preview_render_extent" not in text and "previewSceneExtent" not in text
         and "recordImportedScenePreviewExtent" not in text,
         f"{name} must not use the preview extent")
exporter = (src / "exporter.cpp").read_text()
for call in re.finditer(r"prepareVisualImportedSceneLayerAtTime\(", exporter):
    tail = exporter[call.start():call.start() + 1500]
    need("NativeImportedSceneRenderUse::Export" in tail.split(";")[0],
         "exporter must keep calling the prepare step with the Export use")

# One telemetry field and one HUD reader share the name.
need("bool recordImportedScenePreviewExtent (uint32_t width, uint32_t height)" in telemetry,
     "recordImportedScenePreviewExtent missing")
need("out.importedScenePreviewWidth = importedScenePreviewWidth_;" in telemetry,
     "snapshot must copy the preview extent")
need("importedScenePreviewWidth_ = importedScenePreviewHeight_ = 0;" in telemetry,
     "reset must clear the preview extent")
need('{ "importedScenePreviewExtent", value.importedScenePreviewWidth != 0' in telemetry_json,
     "JSON field importedScenePreviewExtent missing")
need('telemetry->getProperty("importedScenePreviewExtent")' in hud,
     "HUD must read importedScenePreviewExtent")
label = "Imported scene render: %TIME% ms/frame at %WIDTH%x%HEIGHT%"
need(f'TRANS("{label}")' in hud, "HUD label with the extent missing")
need(f'"{label}"' in strings, "HUD label missing from tools/i18n/strings.json")

if errors:
    print("imported scene preview extent contract failures:\n  " + "\n  ".join(errors), file=sys.stderr)
    raise SystemExit(1)
print("imported scene preview extent source contract: PASS")
