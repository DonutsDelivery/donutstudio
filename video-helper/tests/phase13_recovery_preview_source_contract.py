#!/usr/bin/env python3
"""Source contract for Phase 13 recovery and recipe preview."""

from pathlib import Path
import sys


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def section(text: str, start: str, end: str) -> str:
    begin = text.find(start)
    require(begin >= 0, f"missing section start: {start}")
    finish = text.find(end, begin + len(start))
    require(finish >= 0, f"missing section end after {start}: {end}")
    return text[begin:finish]


def main() -> int:
    helper_root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1])
    repo_root = helper_root.parent
    processor = (repo_root / "plugin/Source/PluginProcessor.cpp").read_text(encoding="utf-8")
    sidecar = (repo_root / "plugin/Source/VideoSidecar.h").read_text(encoding="utf-8")
    editor = (repo_root / "plugin/Source/VideoEditor/VideoVisualGraphEditor.h").read_text(encoding="utf-8")
    helper_main = (helper_root / "src/main.cpp").read_text(encoding="utf-8")

    module_diagnosis = section(
        processor,
        "HarmonicMIDIProcessor::videoDiagnoseVisualModule",
        "bool HarmonicMIDIProcessor::videoPlaceVisualModule",
    )
    placement = section(
        processor,
        "bool HarmonicMIDIProcessor::videoPlaceVisualModule",
        "bool HarmonicMIDIProcessor::videoPasteVisualRecipe",
    )
    require("makeVisualModuleAssetResolver (getVisualModelAssetCatalog())" in module_diagnosis,
            "product diagnostics must resolve module assets from the processor catalog")
    require("validateGraphModule(package, assetResolver)" in placement,
            "production placement must validate exact asset dependencies")
    require("package, x, y, false, assetResolver" in placement,
            "staged placement must use the same exact asset resolver")
    require("dependency.type != \"visual-model\"" in processor
            and "compileVisualModelAssetReference" in processor,
            "module recovery must fail closed for unknown types and non-exact model identity")
    require("processor.videoDiagnoseVisualModule(package)" in editor,
            "the product browser must show the production placement diagnosis")

    prepare = section(
        processor,
        "bool HarmonicMIDIProcessor::videoPrepareVisualRecipePreview",
        "bool HarmonicMIDIProcessor::videoPublishVisualGraphParameter",
    )
    require("recipe.builtIn" in prepare
            and "canonicalBuiltInRenderedPreviewRecipe (recipe.id)" in prepare,
            "preview must admit only source-owned built-in identities")
    require("canonicalRecipe->fragment" in prepare,
            "preview must rebuild the canonical built-in graph")
    require("recipe.fragment" not in prepare,
            "caller recipe pixels or graph bytes must not enter canonical preview compilation")
    require("captureVideoRenderSnapshot" in prepare
            and "compileLayerPlanThroughProductionHandlers" in prepare
            and "previewClip" in prepare
            and "serializeVisualLayerPlans({ plan })" in prepare,
            "preview must use the production snapshot and compiler plan contract")
    require("isHelperInstalled()" in prepare,
            "preview must fail closed without the packaged helper")

    receipt = section(
        sidecar,
        "static bool validateVisualRecipePreviewReceipt",
        "/** Render one recipe thumbnail",
    )
    require("gpuComposited" in receipt and "format" in receipt
            and "compositorBackend" in receipt,
            "plugin receipt validation must require GPU, format, and backend receipts")
    require("expectedBase64Characters" in receipt and "rgbaOut.getSize() != expectedBytes" in receipt,
            "plugin receipt validation must require a complete bounded RGBA payload")

    helper_preview = section(
        helper_main,
        'if (method == "recipe_preview")',
        'if (method == "thumbnails")',
    )
    require("parseExportJob(params, job)" in helper_preview,
            "recipe preview must parse the production export-shaped job")
    require("renderCompositeFrame(job, 0.0, frame, &g_modelPayloads)" in helper_preview,
            "recipe preview must call the shared production compositor with retained model payloads")
    require("validateRecipePreviewPixels" in helper_preview
            and '"compositorBackend"' in helper_preview
            and '"rgbaBase64"' in helper_preview,
            "helper must return validated compositor pixels and backend identity")

    print("Phase 13 recovery and recipe preview source contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
