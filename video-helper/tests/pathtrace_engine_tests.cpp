// Tests for the Path Trace render engine (Plan 3, PT-C1): the engine field on
// the 3D Render wire, the RenderScene cache, progressive reset rules, the
// output format the compositor takes, export determinism, and the failure
// policy (paths end at light-path limits, non-finite samples are dropped, and
// only invalid scene input refuses a frame).

#include "pathtrace/engine.h"
#include "support/pathtrace_glass_scene.h"
#include "../../shared/VisualImportedSceneRenderOperationContract.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace
{

namespace pt = videohelper::pathtrace;
namespace grid = HarmonicMIDI::grid;
namespace render = visualimportedscenerender;
namespace glass = pathtraceglassscene;

int checks = 0;
int failures = 0;

void check (bool condition, const std::string& message)
{
    ++checks;
    if (! condition)
    {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

bool startsWith (const std::string& text, std::string_view prefix)
{
    return text.size() >= prefix.size() && text.compare (0, prefix.size(), prefix) == 0;
}

// ---------------------------------------------------------------- wire contract

render::Request rasterRequest()
{
    render::Request request;
    request.sourceStableId = 71;
    request.renderStableId = 72;
    request.asset.id = "model-asset-1";
    request.asset.version = 3;
    request.asset.contentSha256 = std::string (64, 'a');
    request.asset.sourceMediaType = "model/gltf-binary";
    request.asset.sourceByteSize = 4096;
    request.sceneIndex = 0;
    return request;
}

// Bytes the encoder produced BEFORE renderMode existed. They were captured by
// compiling a scratch program against the headers of ebc94bd545 (the parent of
// the renderMode commit) and printing render::encode of rasterRequest(), of the
// same request as Path Trace with 128 export samples (V27), and with a 128
// sample preview target as well (V28). They are literals on purpose: if the
// Accurate encoder drifts, these comparisons fail instead of moving with it.
const std::string kParentRasterWire =
    "visual.imported-scene.render.v1\n71 72\n\"model-asset-1\"\n3\n"
    "\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"\n"
    "\"model/gltf-binary\"\n4096\n1 0\n";
const std::string kParentV27Prefix = "visual.imported-scene.render.v27\n1 128 152\n";
const std::string kParentV28Prefix = "visual.imported-scene.render.v28\n1 128 128 152\n";

void parentEncoderPins (const std::string& rasterWire, const std::string& tracedWire,
                        const std::string& targetedWire)
{
    check (kParentRasterWire.size() == 152, "the pinned Raster wire is the 152 bytes the parent encoder wrote");
    check (rasterWire == kParentRasterWire, "the Accurate Raster wire is byte-identical to the parent encoder's");
    check (tracedWire == kParentV27Prefix + kParentRasterWire,
           "the Accurate V27 wire is byte-identical to the parent encoder's");
    check (targetedWire == kParentV28Prefix + kParentRasterWire,
           "the Accurate V28 wire is byte-identical to the parent encoder's");
}

// V29: the Render node's quality mode. Accurate is absent from the wire, so
// every bytes above stay exactly as they were; Fast wraps them.
void renderModeTests (const render::Request& raster, const std::string& rasterWire,
                      const render::Request& traced, const std::string& tracedWire,
                      const render::Request& targeted, const std::string& targetedWire)
{
    check (raster.renderMode == render::RenderMode::Accurate && traced.renderMode == render::RenderMode::Accurate,
           "a new request is Accurate");
    check (! startsWith (rasterWire, render::kCompiledOperationHeaderV29)
               && ! startsWith (tracedWire, render::kCompiledOperationHeaderV29)
               && ! startsWith (targetedWire, render::kCompiledOperationHeaderV29),
           "Accurate requests never use the V29 envelope, so their bytes are unchanged");
    auto explicitAccurate = raster;
    explicitAccurate.renderMode = render::RenderMode::Accurate;
    check (render::encode (explicitAccurate) == rasterWire, "an explicit Accurate encodes the exact old bytes");

    auto fast = raster;
    fast.renderMode = render::RenderMode::Fast;
    const auto fastWire = render::encode (fast);
    const std::string fastEnvelope = std::string (render::kCompiledOperationHeaderV29) + "\n1 "
        + std::to_string (rasterWire.size()) + "\n";
    check (fastWire == fastEnvelope + rasterWire, "V29 carries the mode and the exact Accurate bytes");
    render::Request decodedFast;
    check (render::decode (fastWire, decodedFast) && decodedFast.renderMode == render::RenderMode::Fast
               && decodedFast.renderEngine == render::RenderEngine::Raster
               && render::sameRequest (fast, decodedFast) && render::encode (decodedFast) == fastWire,
           "a Fast request round-trips");
    check (! render::sameRequest (raster, decodedFast), "the mode is part of request identity");

    auto fastTraced = traced;
    fastTraced.renderMode = render::RenderMode::Fast;
    const auto fastTracedWire = render::encode (fastTraced);
    check (fastTracedWire == std::string (render::kCompiledOperationHeaderV29) + "\n1 "
                                 + std::to_string (tracedWire.size()) + "\n" + tracedWire,
           "V29 wraps the V27 Path Trace bytes");
    render::Request decodedFastTraced;
    check (render::decode (fastTracedWire, decodedFastTraced)
               && decodedFastTraced.renderMode == render::RenderMode::Fast
               && decodedFastTraced.renderEngine == render::RenderEngine::PathTrace
               && decodedFastTraced.pathTraceExportSamples == 128
               && render::sameRequest (fastTraced, decodedFastTraced),
           "Fast keeps the Path Trace engine and its samples");
    auto fastTargeted = targeted;
    fastTargeted.renderMode = render::RenderMode::Fast;
    const auto fastTargetedWire = render::encode (fastTargeted);
    render::Request decodedFastTargeted;
    check (fastTargetedWire == std::string (render::kCompiledOperationHeaderV29) + "\n1 "
                                   + std::to_string (targetedWire.size()) + "\n" + targetedWire
               && render::decode (fastTargetedWire, decodedFastTargeted)
               && decodedFastTargeted.pathTracePreviewSamples == 128
               && decodedFastTargeted.renderMode == render::RenderMode::Fast,
           "V29 wraps the V28 preview-target bytes");

    // Refusals: the same one-byte-at-a-time shape as the V27 envelope above.
    auto refused = raster;
    refused.renderMode = static_cast<render::RenderMode> (2);
    check (render::encode (refused).empty(), "a mode past Fast is refused");
    render::Request ignored;
    const std::string header = std::string (render::kCompiledOperationHeaderV29) + "\n";
    const std::string size = std::to_string (rasterWire.size());
    check (render::decode (header + "1 " + size + "\n" + rasterWire, ignored),
           "the unmodified V29 envelope below decodes");
    check (! render::decode (header + "0 " + size + "\n" + rasterWire, ignored),
           "a V29 envelope naming Accurate is refused");
    check (! render::decode (header + "2 " + size + "\n" + rasterWire, ignored), "a mode past Fast is refused");
    check (! render::decode (header + "01 " + size + "\n" + rasterWire, ignored),
           "a leading zero in the mode is refused");
    check (! render::decode (header + "1 " + std::to_string (rasterWire.size() + 1) + "\n" + rasterWire, ignored),
           "a nested length past the end is refused");
    check (! render::decode (header + "1 " + size + "\n" + rasterWire.substr (0, rasterWire.size() - 1), ignored),
           "truncated nested bytes are refused");
    check (! render::decode (header + "1 " + size + rasterWire, ignored),
           "a missing newline before the nested bytes is refused");
    check (! render::decode (header + "1  " + size + "\n" + rasterWire, ignored), "a doubled separator is refused");
    check (! render::decode (std::string (render::kCompiledOperationHeaderV29) + " 1 " + size + "\n" + rasterWire,
                             ignored),
           "a header without its newline is refused");
    check (! render::decode (header + "1 " + std::to_string (fastWire.size()) + "\n" + fastWire, ignored),
           "a nested V29 envelope is refused");
    // V29 is outermost. Inside V27 or V28 it is not canonical.
    check (! render::decode (std::string (render::kCompiledOperationHeaderV27) + "\n1 128 "
                                 + std::to_string (fastWire.size()) + "\n" + fastWire,
                             ignored),
           "V29 inside V27 is refused");

    // Composed leaves stay Accurate: the node owns the mode.
    auto composed = raster;
    auto leaf = std::make_shared<render::Request> (raster);
    leaf->renderMode = render::RenderMode::Fast;
    composed.importedScene = leaf;
    check (! render::detail::validRenderEngine (composed), "an imported leaf may not name Fast");
    composed.importedScene = std::make_shared<render::Request> (raster);
    check (render::detail::validRenderEngine (composed), "an Accurate leaf stays admitted");
}

// V34: the bounce limit wraps every other envelope and is absent on the default.
void maxBouncesWireTests (const render::Request& raster, const render::Request& traced,
                          const std::string& tracedWire)
{
    check (traced.pathTraceMaxBounces == render::kDefaultPathTraceMaxBounces
               && tracedWire.find (render::kCompiledOperationHeaderV34) == std::string::npos,
           "a graph on the default bounce limit keeps its bytes");
    auto deep = traced;
    deep.pathTraceMaxBounces = 48;
    const auto deepWire = render::encode (deep);
    check (deepWire == std::string (render::kCompiledOperationHeaderV34) + "\n48 "
                           + std::to_string (tracedWire.size()) + "\n" + tracedWire,
           "V34 carries the limit and the exact bytes of the default");
    render::Request decoded;
    check (render::decode (deepWire, decoded) && decoded.pathTraceMaxBounces == 48
               && render::sameRequest (deep, decoded) && render::encode (decoded) == deepWire,
           "a bounce limit round-trips");
    check (! render::sameRequest (traced, decoded), "the bounce limit is part of request identity");
    auto oldDefault = traced;
    oldDefault.pathTraceMaxBounces = 12;
    check (render::decode (render::encode (oldDefault), decoded) && decoded.pathTraceMaxBounces == 12,
           "a stored 12 stays 12 under the new default");

    const auto header = std::string (render::kCompiledOperationHeaderV34) + "\n";
    const auto size = std::to_string (tracedWire.size());
    render::Request refused;
    check (! render::decode (header + "65536 " + size + "\n" + tracedWire, refused),
           "a limit past the field is refused");
    check (! render::decode (header + std::to_string (render::kDefaultPathTraceMaxBounces) + " " + size + "\n"
                                 + tracedWire, refused),
           "the default written out is not canonical");
    auto rasterDeep = raster;
    rasterDeep.pathTraceMaxBounces = 48;
    check (render::encode (rasterDeep).empty(), "Raster may not set a bounce limit");
    auto noBounces = traced;
    noBounces.pathTraceMaxBounces = 0;
    check (render::decode (render::encode (noBounces), decoded) && decoded.pathTraceMaxBounces == 0,
           "zero bounces round-trip");
}

// V30: the denoise switch wraps every older envelope and is absent when off.
void denoiseWireTests (const render::Request& raster, const std::string& rasterWire,
                       const render::Request& traced, const std::string& tracedWire,
                       const render::Request& targeted, const std::string& targetedWire)
{
    auto off = traced;
    off.pathTraceDenoise = false;
    check (render::encode (off) == tracedWire, "denoise off keeps the exact V27 bytes");

    const auto wrap = [] (const std::string& nested) {
        return std::string (render::kCompiledOperationHeaderV30) + "\n" + std::to_string (nested.size()) + "\n" + nested;
    };
    auto on = traced;
    on.pathTraceDenoise = true;
    const auto onWire = render::encode (on);
    check (onWire == wrap (tracedWire), "V30 carries the exact V27 bytes");
    render::Request decodedOn;
    check (render::decode (onWire, decodedOn) && decodedOn.pathTraceDenoise
               && decodedOn.renderEngine == render::RenderEngine::PathTrace
               && decodedOn.pathTraceExportSamples == 128
               && render::sameRequest (on, decodedOn) && render::encode (decodedOn) == onWire,
           "a denoised request round-trips");
    check (! render::sameRequest (traced, decodedOn), "the denoise switch is part of request identity");

    auto onTargeted = targeted;
    onTargeted.pathTraceDenoise = true;
    const auto onTargetedWire = render::encode (onTargeted);
    render::Request decodedTargeted;
    check (onTargetedWire == wrap (targetedWire) && render::decode (onTargetedWire, decodedTargeted)
               && decodedTargeted.pathTraceDenoise && decodedTargeted.pathTracePreviewSamples == 128,
           "V30 wraps the V28 bytes");

    auto onFast = traced;
    onFast.renderMode = render::RenderMode::Fast;
    const auto fastWire = render::encode (onFast);
    onFast.pathTraceDenoise = true;
    const auto onFastWire = render::encode (onFast);
    render::Request decodedFast;
    check (onFastWire == wrap (fastWire) && render::decode (onFastWire, decodedFast)
               && decodedFast.pathTraceDenoise && decodedFast.renderMode == render::RenderMode::Fast,
           "V30 wraps the V29 bytes");

    auto rasterOn = raster;
    rasterOn.pathTraceDenoise = true;
    check (render::encode (rasterOn).empty(), "Raster refuses the denoise switch");
    render::Request refused;
    check (! render::decode (wrap (rasterWire), refused), "a V30 envelope around a Raster request is refused");
    check (! render::decode (wrap (onWire), refused), "a nested V30 envelope is refused");
    check (! render::decode (std::string (render::kCompiledOperationHeaderV30) + "\n1\n" + tracedWire, refused),
           "a V30 envelope with the wrong length is refused");
    const std::string v27InsideOut = std::string (render::kCompiledOperationHeaderV27) + "\n1 128 "
        + std::to_string (onWire.size()) + "\n" + onWire;
    check (! render::decode (v27InsideOut, refused), "V30 inside an older envelope is refused");
}

void realtimeEngineTests()
{
    const auto raster = rasterRequest();
    const auto rasterWire = render::encode (raster);
    auto realtime = raster;
    realtime.renderEngine = render::RenderEngine::Realtime;
    const auto realtimeWire = render::encode (realtime);
    check (realtimeWire == std::string (render::kCompiledOperationHeaderV27) + "\n3 "
                               + std::to_string (render::kDefaultPathTraceExportSamples) + " "
                               + std::to_string (rasterWire.size()) + "\n" + rasterWire,
           "V27 carries the Realtime engine and the exact Raster bytes");
    render::Request decoded;
    check (render::decode (realtimeWire, decoded) && decoded.renderEngine == render::RenderEngine::Realtime
               && render::sameRequest (realtime, decoded) && render::encode (decoded) == realtimeWire,
           "a Realtime request round-trips");
    check (! render::sameRequest (raster, decoded), "the engine is part of request identity");

    auto samples = realtime;
    samples.pathTraceExportSamples = 128;
    check (render::encode (samples).empty(), "Realtime refuses path trace samples");
    auto denoised = realtime;
    denoised.pathTraceDenoise = true;
    check (render::encode (denoised).empty(), "Realtime refuses the denoise switch");
    render::Request refused;
    const auto replaceEngine = [&] (const char* engine)
    {
        return std::string (render::kCompiledOperationHeaderV27) + "\n" + engine + " "
            + std::to_string (render::kDefaultPathTraceExportSamples) + " "
            + std::to_string (rasterWire.size()) + "\n" + rasterWire;
    };
    check (! render::decode (replaceEngine ("2"), refused), "the Workbench value stays refused");
    check (! render::decode (replaceEngine ("4"), refused), "an unknown engine is refused");

    auto fast = realtime;
    fast.renderMode = render::RenderMode::Fast;
    render::Request decodedFast;
    check (render::decode (render::encode (fast), decodedFast)
               && decodedFast.renderEngine == render::RenderEngine::Realtime
               && decodedFast.renderMode == render::RenderMode::Fast,
           "Fast keeps the Realtime engine");
    check (render::kRealtimeCausticsNeedPathTrace == std::string_view ("caustic off: caustics need Path Trace"),
           "the caustic notice names Path Trace");
}

void contractTests()
{
    realtimeEngineTests();
    const auto raster = rasterRequest();
    check (raster.renderEngine == render::RenderEngine::Raster
               && raster.pathTraceExportSamples == render::kDefaultPathTraceExportSamples,
           "a new request is Raster with the default export samples");
    const auto rasterWire = render::encode (raster);
    check (! rasterWire.empty() && ! startsWith (rasterWire, render::kCompiledOperationHeaderV27),
           "a Raster request keeps its pre-engine wire");
    render::Request decodedRaster;
    check (render::decode (rasterWire, decodedRaster) && decodedRaster.renderEngine == render::RenderEngine::Raster
               && decodedRaster.pathTraceExportSamples == render::kDefaultPathTraceExportSamples,
           "an old wire decodes as Raster");

    auto traced = raster;
    traced.renderEngine = render::RenderEngine::PathTrace;
    traced.pathTraceExportSamples = 128;
    const auto tracedWire = render::encode (traced);
    check (startsWith (tracedWire, render::kCompiledOperationHeaderV27), "a Path Trace request uses the V27 envelope");
    const std::string envelope = std::string (render::kCompiledOperationHeaderV27) + "\n1 128 "
        + std::to_string (rasterWire.size()) + "\n";
    check (tracedWire == envelope + rasterWire, "V27 carries engine, samples and the exact Raster bytes");
    render::Request decodedTraced;
    check (render::decode (tracedWire, decodedTraced) && decodedTraced.renderEngine == render::RenderEngine::PathTrace
               && decodedTraced.pathTraceExportSamples == 128 && render::sameRequest (traced, decodedTraced)
               && render::encode (decodedTraced) == tracedWire,
           "a Path Trace request round-trips");
    check (! render::sameRequest (raster, decodedTraced), "the engine is part of request identity");

    // V28: the preview target beside the selection, emitted only when it
    // differs, so a graph on the default keeps the V27 bytes above.
    auto targeted = traced;
    targeted.pathTracePreviewSamples = 128;
    const auto targetedWire = render::encode (targeted);
    check (startsWith (targetedWire, render::kCompiledOperationHeaderV28),
           "a non-default preview target uses the V28 envelope");
    const std::string targetedEnvelope = std::string (render::kCompiledOperationHeaderV28)
        + "\n1 128 128 " + std::to_string (rasterWire.size()) + "\n";
    check (targetedWire == targetedEnvelope + rasterWire,
           "V28 carries engine, export samples, preview target and the exact Raster bytes");
    render::Request decodedTargeted;
    check (render::decode (targetedWire, decodedTargeted)
               && decodedTargeted.pathTracePreviewSamples == 128
               && render::sameRequest (targeted, decodedTargeted)
               && render::encode (decodedTargeted) == targetedWire,
           "a V28 request round-trips");
    auto refusedTarget = traced;
    refusedTarget.pathTracePreviewSamples = 0;
    check (render::encode (refusedTarget).empty(), "zero preview samples are refused");
    // Sample counts stop at the 16-bit field and nowhere earlier.
    auto largest = traced;
    largest.pathTraceExportSamples = 65535;
    largest.pathTracePreviewSamples = 65535;
    render::Request decodedLargest;
    check (render::decode (render::encode (largest), decodedLargest)
               && decodedLargest.pathTraceExportSamples == 65535
               && decodedLargest.pathTracePreviewSamples == 65535,
           "the largest sample counts the field holds round-trip");
    refusedTarget = raster;
    refusedTarget.pathTracePreviewSamples = 32;
    check (render::encode (refusedTarget).empty(), "Raster keeps the default preview samples");

    auto refused = traced;
    refused.pathTraceExportSamples = 0;
    check (render::encode (refused).empty(), "zero export samples are refused");
    auto outputs = traced;
    outputs.imageOutput = renderpassoutput::Output::Depth;
    outputs.rawExportMask = 1u << static_cast<unsigned> (renderpassoutput::Output::Motion);
    render::Request decodedOutputs;
    check (render::decode (render::encode (outputs), decodedOutputs)
               && decodedOutputs.imageOutput == renderpassoutput::Output::Depth
               && decodedOutputs.rawExportMask == outputs.rawExportMask,
           "Path Trace keeps an Image view and raw data exports");
    refused = raster;
    refused.pathTraceExportSamples = 32;
    check (render::encode (refused).empty(), "Raster keeps the default export samples");

    render::Request ignored;
    const auto replaceEngine = [&] (const std::string& line) {
        return std::string (render::kCompiledOperationHeaderV27) + "\n" + line + " "
            + std::to_string (rasterWire.size()) + "\n" + rasterWire;
    };
    check (! render::decode (replaceEngine ("2 64"), ignored), "the reserved Workbench engine is refused");
    check (! render::decode (replaceEngine ("0 64"), ignored), "a V27 envelope naming Raster is refused");
    check (! render::decode (replaceEngine ("1 0"), ignored), "a V27 envelope with zero samples is refused");
    const std::string nested = std::string (render::kCompiledOperationHeaderV27) + "\n1 64 "
        + std::to_string (tracedWire.size()) + "\n" + tracedWire;
    check (! render::decode (nested, ignored), "a nested V27 envelope is refused");

    // Malformed V27 envelopes. Each wire differs from a valid one in one place.
    const std::string header = std::string (render::kCompiledOperationHeaderV27) + "\n";
    const std::string size = std::to_string (rasterWire.size());
    check (render::decode (header + "1 128 " + size + "\n" + rasterWire, ignored),
           "the unmodified envelope below decodes");
    check (! render::decode (header + "1 128 " + size + "\n" + rasterWire.substr (0, rasterWire.size() - 1), ignored),
           "truncated nested bytes are refused");
    check (! render::decode (header + "1 128 " + std::to_string (rasterWire.size() - 1) + "\n"
                                 + rasterWire.substr (0, rasterWire.size() - 1),
                             ignored),
           "a nested length that cuts the raster request short is refused");
    check (! render::decode (header + "1 128 " + std::to_string (rasterWire.size() + 1) + "\n" + rasterWire, ignored),
           "a nested length past the end is refused");
    check (! render::decode (header + "01 128 " + size + "\n" + rasterWire, ignored),
           "a leading zero in the engine is refused");
    check (! render::decode (header + "1 0128 " + size + "\n" + rasterWire, ignored),
           "a leading zero in the samples is refused");
    check (! render::decode (header + "1 128 0" + size + "\n" + rasterWire, ignored),
           "a leading zero in the nested length is refused");
    check (render::decode (header + "1 65535 " + size + "\n" + rasterWire, ignored)
               && ignored.pathTraceExportSamples == 65535,
           "the largest 16-bit export samples decode");
    check (! render::decode (header + "1 65536 " + size + "\n" + rasterWire, ignored),
           "export samples past the 16-bit field are refused");
    check (! render::decode (header + "1 65600 " + size + "\n" + rasterWire, ignored),
           "export samples past 16 bits are refused, not truncated");
    check (! render::decode (header + "1128 " + size + "\n" + rasterWire, ignored),
           "a missing separator between engine and samples is refused");
    check (! render::decode (header + "1 128" + size + "\n" + rasterWire, ignored),
           "a missing separator between samples and length is refused");
    check (! render::decode (header + "1 128 " + size + rasterWire, ignored),
           "a missing newline before the nested bytes is refused");
    check (! render::decode (header + "1  128 " + size + "\n" + rasterWire, ignored),
           "a doubled separator is refused");
    check (! render::decode (std::string (render::kCompiledOperationHeaderV27) + " 1 128 " + size + "\n" + rasterWire,
                             ignored),
           "a header without its newline is refused");

    parentEncoderPins (rasterWire, tracedWire, targetedWire);
    renderModeTests (raster, rasterWire, traced, tracedWire, targeted, targetedWire);
    maxBouncesWireTests (raster, traced, tracedWire);
    denoiseWireTests (raster, rasterWire, traced, tracedWire, targeted, targetedWire);
}

// ---------------------------------------------------------------- engine helpers

struct Fixture
{
    std::shared_ptr<const grid::Visual3DScene> scene = glass::makeScene();
    pt::EngineSceneSetup setup = glass::makeSetup (scene);
    pt::EngineView view = glass::makeView (*scene, 24, 16);
};

bool preview (pt::SceneEngine& engine, const pt::EngineSceneSetup& setup, const pt::EngineView& view,
              pt::EngineImage& image, pt::EngineColor color = pt::EngineColor::DisplayRgba8)
{
    std::string error;
    const bool ok = engine.renderPreview (setup, view, color, image, error);
    check (ok, "preview renders: " + error);
    return ok;
}

// ---------------------------------------------------------------- cache

void cacheTests()
{
    Fixture f;
    pt::SceneEngine engine (2);
    pt::EngineImage image;
    preview (engine, f.setup, f.view, image);
    check (engine.sceneBuilds() == 1 && image.sceneRebuilt, "the first preview builds the RenderScene");
    preview (engine, f.setup, f.view, image);
    check (engine.sceneBuilds() == 1 && ! image.sceneRebuilt, "the same setup reuses the RenderScene and BVH");

    auto view = f.view;
    view.camera.transform.translation.x += 0.25f;
    preview (engine, f.setup, view, image);
    check (engine.sceneBuilds() == 1 && ! image.sceneRebuilt, "a camera move keeps the RenderScene");
    view.width = 20;
    preview (engine, f.setup, view, image);
    check (engine.sceneBuilds() == 1, "a size change keeps the RenderScene");
    view.timeSeconds = 2.0f;
    preview (engine, f.setup, view, image);
    check (engine.sceneBuilds() == 1, "a time change keeps the RenderScene");

    auto setup = f.setup;
    setup.lights[0].intensity *= 2.0f;
    preview (engine, setup, view, image);
    check (engine.sceneBuilds() == 2 && image.sceneRebuilt, "a light change rebuilds");
    setup.objectMaterials[0].material.ior = 1.33f;
    preview (engine, setup, view, image);
    check (engine.sceneBuilds() == 3, "a material change rebuilds");
    setup.emissionGain = 0.5f;
    preview (engine, setup, view, image);
    check (engine.sceneBuilds() == 4, "an emission gain change rebuilds");
    setup.objectWorlds.clear();
    for (std::size_t index = 0; index < setup.scene->objectCount; ++index)
        setup.objectWorlds.push_back (pt::objectWorldMatrix (*setup.scene, index));
    preview (engine, setup, view, image);
    check (engine.sceneBuilds() == 5, "runtime object worlds rebuild");
    setup.objectWorlds[3].m[13] += 0.1f;
    preview (engine, setup, view, image);
    check (engine.sceneBuilds() == 6, "a moved object rebuilds");
    preview (engine, setup, view, image);
    check (engine.sceneBuilds() == 6, "an unchanged setup with object worlds is reused");

    // The plan builds a new scene snapshot every frame. A light record holds
    // the environment image's asset strings, so its raw bytes include pointers
    // into its own snapshot and the padding after the light kind; neither is
    // part of the light, and an equal snapshot must keep the accumulation.
    setup.scene = std::make_shared<const grid::Visual3DScene> (*f.scene);
    preview (engine, setup, view, image);
    check (engine.sceneBuilds() == 6 && ! image.sceneRebuilt, "an equal new scene snapshot is reused");
    auto padded = std::make_shared<grid::Visual3DScene> (*f.scene);
    for (std::size_t index = 0; index < padded->lightCount; ++index)
        std::fill (reinterpret_cast<unsigned char*> (&padded->lights[index].kind) + 1,
                   reinterpret_cast<unsigned char*> (&padded->lights[index].transform), static_cast<unsigned char> (0xa5));
    setup.scene = padded;
    preview (engine, setup, view, image);
    check (engine.sceneBuilds() == 6 && ! image.sceneRebuilt, "light padding bytes do not rebuild");
    auto brighter = std::make_shared<grid::Visual3DScene> (*f.scene);
    brighter->lights[0].intensity *= 2.0f;
    setup.scene = brighter;
    preview (engine, setup, view, image);
    check (engine.sceneBuilds() == 7 && image.sceneRebuilt, "a changed scene light rebuilds");

    auto bad = setup;
    bad.objectWorlds.pop_back();
    std::string error;
    check (! engine.renderPreview (bad, view, pt::EngineColor::DisplayRgba8, image, error) && ! error.empty(),
           "object worlds that miss objects are refused");
    bad = setup;
    bad.scene = nullptr;
    check (! engine.renderPreview (bad, view, pt::EngineColor::DisplayRgba8, image, error), "no scene is refused");
    auto badView = view;
    badView.camera.nearPlane = 0.0f;
    check (! engine.renderPreview (setup, badView, pt::EngineColor::DisplayRgba8, image, error),
           "an invalid camera is refused");
}

// ---------------------------------------------------------------- progressive

void progressiveTests()
{
    Fixture f;
    pt::SceneEngine engine (2);
    engine.samplesPerPass = 2;
    engine.maximumPreviewSamples = 6;
    pt::EngineImage image;

    preview (engine, f.setup, f.view, image);
    const std::uint64_t first = image.generation;
    check (image.restarted && image.traced && image.samplesPerPixel == 2, "the first preview runs one pass");
    check (first != pt::kExportGeneration, "preview generations never use the export generation");
    preview (engine, f.setup, f.view, image);
    check (! image.restarted && image.generation == first && image.samplesPerPixel == 4,
           "an unchanged frame adds one pass to the same generation");
    check (image.stats.samplesAdded + image.stats.nonFiniteSamples == 2ull * f.view.width * f.view.height,
           "one pass adds samplesPerPass per pixel");

    auto view = f.view;
    view.camera.transform.rotation = glass::axisAngle (1.0f, 0.0f, 0.0f, -0.2f);
    preview (engine, f.setup, view, image);
    check (image.restarted && image.generation != first && image.samplesPerPixel == 2, "a camera change restarts");
    std::uint64_t generation = image.generation;
    view.width = 30;
    preview (engine, f.setup, view, image);
    check (image.restarted && image.generation != generation && image.width == 30, "a size change restarts");
    generation = image.generation;
    view.timeSeconds = 0.5f;
    preview (engine, f.setup, view, image);
    check (image.restarted && image.generation != generation && ! image.sceneRebuilt, "a time change restarts");
    generation = image.generation;
    auto setup = f.setup;
    setup.lights[1].intensity = 0.5f;
    preview (engine, setup, view, image);
    check (image.restarted && image.sceneRebuilt && image.generation != generation, "a scene change restarts");
    generation = image.generation;
    engine.settings().maxDepth = 3;
    preview (engine, setup, view, image);
    check (image.restarted && ! image.sceneRebuilt && image.generation != generation && image.samplesPerPixel == 2,
           "a changed bounce limit restarts without rebuilding the scene");
    generation = image.generation;

    engine.cancel();
    preview (engine, setup, view, image);
    check (image.restarted && image.generation != generation && image.samplesPerPixel == 2,
           "a cancel restarts the next preview");

    preview (engine, setup, view, image);
    preview (engine, setup, view, image);
    check (image.samplesPerPixel == 6 && image.traced, "preview accumulates up to its maximum");
    preview (engine, setup, view, image);
    check (image.unchanged && ! image.traced && ! image.restarted && image.samplesPerPixel == 6
               && image.colorBytes.empty() && image.depth.empty(),
           "a capped preview reports unchanged without tracing");
    preview (engine, setup, view, image, pt::EngineColor::LinearRgba16f);
    check (! image.unchanged && ! image.traced && image.colorBytes.size() == 8u * view.width * view.height,
           "a colour format change resolves the capped image again");

    // The preview target rides each draw: raising it resumes the same
    // accumulation, a target equal to the samples taken stops progress so the
    // viewport's idle gate can settle, and a target below them restarts, so the
    // settled image never holds more samples than the target.
    engine.maximumPreviewSamples = 8;
    check (engine.previewProgressing(), "a raised preview target wants more samples");
    preview (engine, setup, view, image);
    check (! image.unchanged && image.traced && image.samplesPerPixel == 8,
           "a raised preview target resumes the capped accumulation");
    check (! engine.previewProgressing(), "a target equal to the samples taken stops progress");
    generation = image.generation;
    engine.maximumPreviewSamples = 4;
    check (engine.previewProgressing(), "a target below the samples taken wants a restart");
    preview (engine, setup, view, image);
    check (image.restarted && image.generation != generation && image.samplesPerPixel == 2,
           "a lowered preview target restarts the accumulation");
    preview (engine, setup, view, image);
    check (! image.restarted && image.samplesPerPixel == 4 && ! engine.previewProgressing(),
           "the restarted accumulation settles at the lowered target");
    preview (engine, setup, view, image);
    check (image.unchanged && image.samplesPerPixel == 4, "a settled lowered target traces no more");

    pt::SceneEngine single (2);
    single.samplesPerPass = 0;
    preview (single, f.setup, f.view, image);
    check (image.samplesPerPixel == 1, "a zero pass size still adds one sample");
}

// ---------------------------------------------------------------- playback

void playbackTests()
{
    Fixture f;
    pt::SceneEngine engine (2);
    engine.playbackSamples = 4;
    engine.maximumPreviewSamples = 16;
    pt::EngineImage image;

    // (a) Two frames at the same time with playback active hold the budget.
    engine.playbackActive = true;
    preview (engine, f.setup, f.view, image);
    const std::uint64_t generation = image.generation;
    check (image.samplesPerPixel == 4 && image.mode == pt::EngineRenderMode::PlaybackFixed,
           "the first playback frame reaches the fixed budget in one call");
    check (engine.renderMode() == pt::EngineRenderMode::PlaybackFixed, "the engine reports the playback mode");
    check (image.stats.samplesAdded + image.stats.nonFiniteSamples == 4ull * f.view.width * f.view.height,
           "a playback frame traces the budget per pixel");
    preview (engine, f.setup, f.view, image);
    check (image.samplesPerPixel == 4 && ! image.restarted && image.generation == generation && ! image.traced
               && image.unchanged,
           "a second playback frame at the same time keeps the budget and does not restart");

    // A moved time gets a new generation at the same sample count, with or
    // without the flag, because the time alone marks the frame as playback.
    auto moved = f.view;
    moved.timeSeconds += 0.25f;
    preview (engine, f.setup, moved, image);
    check (image.restarted && image.generation != generation && image.samplesPerPixel == 4,
           "a playback frame at a new time restarts at the same budget, not at one sample");
    engine.playbackActive = false;
    moved.timeSeconds += 0.25f;
    preview (engine, f.setup, moved, image);
    check (image.restarted && image.samplesPerPixel == 4 && image.mode == pt::EngineRenderMode::PlaybackFixed,
           "a moved time without the flag renders the fixed budget");

    // (b) Stopped: the same accumulation grows one pass per tick to the target.
    const std::uint64_t stoppedGeneration = image.generation;
    preview (engine, f.setup, moved, image);
    check (! image.restarted && image.generation == stoppedGeneration && image.samplesPerPixel == 5
               && image.traced && image.mode == pt::EngineRenderMode::Converging,
           "stopping resumes the accumulation without an authored change");
    check (engine.previewProgressing(), "the viewport keeps ticking below the target after playback stops");
    preview (engine, f.setup, moved, image);
    check (! image.restarted && image.samplesPerPixel == 6, "each tick after stopping adds samples");
    while (engine.previewProgressing())
        preview (engine, f.setup, moved, image);
    check (image.samplesPerPixel == 16 && image.generation == stoppedGeneration,
           "a stopped viewport converges to its target in one generation");
    preview (engine, f.setup, moved, image);
    check (image.unchanged && image.samplesPerPixel == 16 && image.mode == pt::EngineRenderMode::Converging,
           "a settled viewport traces nothing");

    // Playback resuming at the same time keeps the converged image.
    engine.playbackActive = true;
    preview (engine, f.setup, moved, image);
    check (image.unchanged && image.samplesPerPixel == 16 && ! image.restarted,
           "playback at the same time does not throw away a converged image");
    engine.playbackActive = false;

    // (c) An authored change resets, in either mode, and costs its mode's samples.
    auto setup = f.setup;
    setup.lights[0].intensity *= 2.0f;
    preview (engine, setup, moved, image);
    check (image.restarted && image.sceneRebuilt && image.samplesPerPixel == 1
               && image.mode == pt::EngineRenderMode::Converging,
           "an authored scene change resets a stopped accumulation");
    engine.playbackActive = true;
    setup.lights[0].intensity *= 2.0f;
    preview (engine, setup, moved, image);
    check (image.restarted && image.sceneRebuilt && image.samplesPerPixel == 4,
           "an authored scene change during playback renders the budget again");
    auto turned = moved;
    turned.camera.transform.rotation = glass::axisAngle (1.0f, 0.0f, 0.0f, -0.2f);
    preview (engine, setup, turned, image);
    check (image.restarted && image.samplesPerPixel == 4, "a camera change during playback renders the budget again");
    turned.width = 30;
    preview (engine, setup, turned, image);
    check (image.restarted && image.width == 30 && image.samplesPerPixel == 4,
           "a size change during playback renders the budget again");

    // The budget never passes the target. Zero still traces one sample.
    engine.maximumPreviewSamples = 2;
    preview (engine, setup, turned, image);
    check (image.restarted && image.samplesPerPixel == 2 && ! engine.previewProgressing(),
           "a target below the budget caps the playback frame");
    engine.maximumPreviewSamples = 16;
    pt::SceneEngine zero (2);
    zero.playbackActive = true;
    zero.playbackSamples = 0;
    preview (zero, f.setup, f.view, image);
    check (image.samplesPerPixel == 1, "a zero budget still adds one sample");

    // (d) Export reads neither the flag nor the budget.
    const auto view = glass::makeView (*f.scene, 20, 12);
    std::string error;
    pt::SceneEngine plain (2), playing (2);
    playing.playbackActive = true;
    playing.playbackSamples = 3;
    pt::EngineImage plainExport, playingExport;
    check (plain.renderExport (f.setup, view, pt::EngineColor::LinearRgba16f, 8, plainExport, error)
               && playing.renderExport (f.setup, view, pt::EngineColor::LinearRgba16f, 8, playingExport, error),
           "export renders during playback: " + error);
    check (playingExport.samplesPerPixel == 8 && playingExport.mode == pt::EngineRenderMode::Export
               && playing.renderMode() == pt::EngineRenderMode::Export,
           "export renders its own sample count and reports the export mode");
    check (playingExport.stats.samplesAdded + playingExport.stats.nonFiniteSamples == 8ull * view.width * view.height,
           "export traces its count with a different playback budget set");
    check (playingExport.colorBytes == plainExport.colorBytes, "the playback budget does not change export bytes");
    preview (playing, f.setup, view, image);
    check (image.restarted && image.samplesPerPixel == 3, "playback after an export renders its own budget");

    // An Accurate playback frame draws the export streams, so it equals the export at N.
    pt::SceneEngine seeded (2);
    seeded.previewSeededLikeExport = true;
    seeded.playbackActive = true;
    seeded.playbackSamples = 8;
    pt::EngineImage seededFrame;
    preview (seeded, f.setup, view, seededFrame, pt::EngineColor::LinearRgba16f);
    check (seededFrame.samplesPerPixel == 8 && seededFrame.colorBytes == plainExport.colorBytes,
           "an Accurate playback frame at N samples is the export at N samples");
}

// ---------------------------------------------------------------- temporal history

float halfToFloat (std::uint16_t half)
{
    const int exponent = (half >> 10) & 0x1f;
    const float magnitude = exponent == 0 ? static_cast<float> (half & 0x3ff) * std::ldexp (1.0f, -24)
                                          : static_cast<float> ((half & 0x3ff) | 0x400) * std::ldexp (1.0f, exponent - 25);
    return (half & 0x8000) != 0 ? -magnitude : magnitude;
}

// Mean over every pixel and RGB channel of a linear image.
double meanRadiance (const std::vector<std::uint8_t>& image)
{
    double sum = 0.0;
    const std::size_t pixels = image.size() / 8;
    for (std::size_t pixel = 0; pixel < pixels; ++pixel)
        for (int channel = 0; channel < 3; ++channel)
        {
            std::uint16_t half = 0;
            std::memcpy (&half, image.data() + pixel * 8 + static_cast<std::size_t> (channel) * 2, sizeof (half));
            sum += static_cast<double> (halfToFloat (half));
        }
    return sum / static_cast<double> (pixels * 3);
}

// Mean over every RGB channel of the squared difference of two linear images.
double meanSquaredDifference (const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b)
{
    double sum = 0.0;
    const std::size_t pixels = a.size() / 8;
    for (std::size_t pixel = 0; pixel < pixels; ++pixel)
        for (int channel = 0; channel < 3; ++channel)
        {
            std::uint16_t x = 0, y = 0;
            std::memcpy (&x, a.data() + pixel * 8 + static_cast<std::size_t> (channel) * 2, sizeof (x));
            std::memcpy (&y, b.data() + pixel * 8 + static_cast<std::size_t> (channel) * 2, sizeof (y));
            const double difference = static_cast<double> (halfToFloat (x)) - static_cast<double> (halfToFloat (y));
            sum += difference * difference;
        }
    return sum / static_cast<double> (pixels * 3);
}

void temporalHistoryTests()
{
    const auto linear = pt::EngineColor::LinearRgba16f;
    const auto newEngine = [] (float currentWeight, std::uint32_t samples) {
        auto engine = std::make_unique<pt::SceneEngine> (2);
        engine->playbackCurrentWeight = currentWeight;
        engine->playbackActive = true;
        engine->playbackSamples = samples;
        engine->maximumPreviewSamples = 64;
        return engine;
    };

    // (a) A static scene at moving times: the distance between consecutive frames,
    // with history against without it. Frames draw independent noise in Fast mode.
    {
        Fixture f;
        const auto consecutiveDistance = [&] (float weight, bool& blended) {
            auto engine = newEngine (weight, 4);
            auto view = f.view;
            pt::EngineImage previous, frame;
            preview (*engine, f.setup, view, previous, linear);
            double total = 0.0;
            blended = true;
            for (int step = 0; step < 4; ++step)
            {
                view.timeSeconds += 0.25f;
                preview (*engine, f.setup, view, frame, linear);
                blended = blended && frame.temporalDenoise;
                total += meanSquaredDifference (previous.colorBytes, frame.colorBytes);
                previous = frame;
            }
            return total;
        };
        bool blended = false, plain = true;
        const double withHistory = consecutiveDistance (pt::kDefaultPlaybackCurrentWeight, blended);
        const double withoutHistory = consecutiveDistance (1.0f, plain);
        check (blended, "every playback frame after the first reports the temporal step");
        check (! plain, "a weight of one turns the temporal step off");
        check (withoutHistory > 0.0 && withHistory < 0.5 * withoutHistory,
               "history brings consecutive frames of a static scene at least twice as close: "
                   + std::to_string (withHistory) + " against " + std::to_string (withoutHistory));
    }

    // (b) Stopping resumes the accumulator's own convergence: every call matches an
    // engine that never had history, because both draw the same generations.
    {
        Fixture f;
        auto withHistory = newEngine (pt::kDefaultPlaybackCurrentWeight, 4);
        auto without = newEngine (1.0f, 4);
        auto view = f.view;
        pt::EngineImage a, b;
        for (int step = 0; step < 3; ++step)
        {
            view.timeSeconds += 0.25f;
            preview (*withHistory, f.setup, view, a, linear);
            preview (*without, f.setup, view, b, linear);
        }
        check (a.temporalDenoise && a.colorBytes != b.colorBytes, "the last playback frame blends history");
        withHistory->playbackActive = false;
        without->playbackActive = false;
        for (int step = 0; step < 3; ++step)
        {
            preview (*withHistory, f.setup, view, a, linear);
            preview (*without, f.setup, view, b, linear);
            check (a.mode == pt::EngineRenderMode::Converging && ! a.temporalDenoise && a.colorBytes == b.colorBytes,
                   "a converging frame after playback has the bytes of a history-free engine");
        }

        // A blended frame at the target has no samples left to add, yet stopping
        // still has to replace it with the accumulation.
        auto settled = newEngine (pt::kDefaultPlaybackCurrentWeight, 4);
        settled->maximumPreviewSamples = 4;
        auto reference = newEngine (1.0f, 4);
        reference->maximumPreviewSamples = 4;
        auto settledView = f.view;
        for (int step = 0; step < 2; ++step)
        {
            settledView.timeSeconds += 0.25f;
            preview (*settled, f.setup, settledView, a, linear);
            preview (*reference, f.setup, settledView, b, linear);
        }
        settled->playbackActive = false;
        preview (*settled, f.setup, settledView, a, linear);
        check (! a.unchanged && ! a.traced && ! a.temporalDenoise && a.colorBytes == b.colorBytes,
               "stopping at the target resolves the accumulation instead of keeping the blend");
    }

    // (c) Export after playback has the bytes of an engine that never played.
    {
        Fixture f;
        auto played = newEngine (pt::kDefaultPlaybackCurrentWeight, 4);
        auto view = f.view;
        pt::EngineImage image, fresh;
        for (int step = 0; step < 3; ++step)
        {
            view.timeSeconds += 0.25f;
            preview (*played, f.setup, view, image, linear);
        }
        std::string error;
        pt::SceneEngine clean (2);
        check (played->renderExport (f.setup, view, linear, 8, image, error)
                   && clean.renderExport (f.setup, view, linear, 8, fresh, error),
               "export renders after playback: " + error);
        check (! image.temporalDenoise && image.colorBytes == fresh.colorBytes && image.depth == fresh.depth,
               "an export after playback has the bytes of a history-free export");
        view.timeSeconds += 0.25f;
        preview (*played, f.setup, view, image, linear);
        check (! image.temporalDenoise, "playback after an export starts without history");
    }

    // (d) A lighting step keeps the geometry, so only the clip can stop the smear. The
    // image mean ignores per-pixel noise. With light eight times brighter, an unclipped
    // blend reaches 0.8 / 8 + 0.2 = 0.3 of a fresh render's mean, and the check wants 0.6.
    {
        Fixture f;
        auto engine = newEngine (pt::kDefaultPlaybackCurrentWeight, 16);
        auto view = glass::makeView (*f.scene, 48, 32);
        pt::EngineImage before, after, fresh;
        for (int step = 0; step < 3; ++step)
        {
            view.timeSeconds += 0.25f;
            preview (*engine, f.setup, view, before, linear);
        }
        auto brighter = f.setup;
        for (auto& light : brighter.lights)
            light.intensity *= 8.0f;
        view.timeSeconds += 0.25f;
        preview (*engine, brighter, view, after, linear);
        check (after.sceneRebuilt && after.temporalDenoise, "a scene change during playback keeps the history");
        auto reference = newEngine (1.0f, 16);
        preview (*reference, brighter, view, fresh, linear);
        const double reached = meanRadiance (after.colorBytes), target = meanRadiance (fresh.colorBytes);
        check (target > meanRadiance (before.colorBytes) && reached > 0.6 * target,
               "the frame after a lighting step reaches the new brightness: "
                   + std::to_string (reached) + " against " + std::to_string (target));
    }

    // (d) Moved geometry: a pixel whose centre ray now hits something else takes the
    // current frame exactly, so an object leaves no ghost where it was.
    {
        Fixture f;
        const auto worldsWithGlassAt = [&] (float offset) {
            std::vector<pt::Mat4> worlds;
            grid::SceneTransform3D move {};
            move.translation = { offset, 0.0f, 0.0f };
            for (std::size_t index = 0; index < f.scene->objectCount; ++index)
            {
                const auto world = pt::objectWorldMatrix (*f.scene, index);
                worlds.push_back (f.scene->objects[index].id.value == glass::kGlassObject
                                      ? pt::transformMatrix (move) * world : world);
            }
            return worlds;
        };
        auto withHistory = newEngine (pt::kDefaultPlaybackCurrentWeight, 4);
        auto without = newEngine (1.0f, 4);
        auto setup = f.setup;
        setup.objectWorlds = worldsWithGlassAt (0.0f);
        auto view = f.view;
        pt::EngineImage previous, a, b;
        for (int step = 0; step < 2; ++step)
        {
            view.timeSeconds += 0.25f;
            preview (*withHistory, setup, view, previous, linear);
            preview (*without, setup, view, b, linear);
        }
        setup.objectWorlds = worldsWithGlassAt (1.0f);
        view.timeSeconds += 0.25f;
        preview (*withHistory, setup, view, a, linear);
        preview (*without, setup, view, b, linear);
        std::size_t rejected = 0, smeared = 0;
        for (std::size_t pixel = 0; pixel < a.depth.size(); ++pixel)
        {
            const float depth = a.depth[pixel], earlier = previous.depth[pixel];
            if (std::abs (depth - earlier) <= pt::kPlaybackHistoryDepthTolerance * std::max (depth, earlier))
                continue;
            ++rejected;
            if (std::memcmp (a.colorBytes.data() + pixel * 8, b.colorBytes.data() + pixel * 8, 8) != 0)
                ++smeared;
        }
        check (rejected > 0, "moving the glass changes the depth of some pixels");
        check (smeared == 0, "a pixel whose depth moved keeps none of its history: "
                                 + std::to_string (smeared) + " of " + std::to_string (rejected));
    }

    // A camera or size change drops the history, as nothing reprojects it.
    {
        Fixture f;
        auto engine = newEngine (pt::kDefaultPlaybackCurrentWeight, 4);
        auto view = f.view;
        pt::EngineImage image;
        for (int step = 0; step < 2; ++step)
        {
            view.timeSeconds += 0.25f;
            preview (*engine, f.setup, view, image, linear);
        }
        check (image.temporalDenoise, "the second playback frame blends");
        view.camera.transform.rotation = glass::axisAngle (1.0f, 0.0f, 0.0f, -0.2f);
        view.timeSeconds += 0.25f;
        preview (*engine, f.setup, view, image, linear);
        check (! image.temporalDenoise, "a camera change during playback starts without history");
        view.timeSeconds += 0.25f;
        preview (*engine, f.setup, view, image, linear);
        check (image.temporalDenoise, "the next frame at the new camera blends again");
        view.width = 30;
        view.timeSeconds += 0.25f;
        preview (*engine, f.setup, view, image, linear);
        check (! image.temporalDenoise && image.width == 30, "a size change during playback starts without history");
        view.timeSeconds += 0.25f;
        preview (*engine, f.setup, view, image, linear);
        check (image.temporalDenoise && image.width == 30, "the history follows the new film size");
    }

    // The history is sized from the film. A film far past the fixtures still blends.
    {
        Fixture f;
        auto engine = newEngine (pt::kDefaultPlaybackCurrentWeight, 1);
        auto view = glass::makeView (*f.scene, 640, 360);
        pt::EngineImage image;
        for (int step = 0; step < 2; ++step)
        {
            view.timeSeconds += 0.25f;
            preview (*engine, f.setup, view, image, linear);
        }
        check (image.temporalDenoise && image.colorBytes.size() == 640u * 360u * 8u,
               "a larger film blends its whole image");
    }

    // A repeated colour request on a held playback frame resolves the same blend.
    {
        Fixture f;
        auto engine = newEngine (pt::kDefaultPlaybackCurrentWeight, 4);
        engine->playbackActive = true;
        auto view = f.view;
        pt::EngineImage image, again;
        for (int step = 0; step < 2; ++step)
        {
            view.timeSeconds += 0.25f;
            preview (*engine, f.setup, view, image, linear);
        }
        preview (*engine, f.setup, view, again, linear);
        check (again.unchanged && again.temporalDenoise, "a held playback frame stays the blend and traces nothing");
        preview (*engine, f.setup, view, again, pt::EngineColor::DisplayRgba8);
        check (! again.unchanged && again.temporalDenoise && ! again.traced,
               "a new colour format on a held frame resolves the blend again");
    }
}

// ---------------------------------------------------------------- output format

std::uint16_t halfAt (const std::vector<std::uint8_t>& bytes, std::size_t pixel, int channel)
{
    std::uint16_t value = 0;
    std::memcpy (&value, bytes.data() + pixel * 8 + static_cast<std::size_t> (channel) * 2, sizeof (value));
    return value;
}

void formatTests()
{
    check (pt::halfBits (1.0f) == 0x3c00 && pt::halfBits (0.5f) == 0x3800 && pt::halfBits (0.0f) == 0
               && pt::halfBits (-0.0f) == 0x8000 && pt::halfBits (-2.0f) == 0xc000,
           "half conversion of exact values");
    check (pt::halfBits (65504.0f) == 0x7bff && pt::halfBits (1.0e9f) == 0x7bff
               && pt::halfBits (std::numeric_limits<float>::infinity()) == 0x7bff,
           "half conversion saturates to the largest finite half");
    check (pt::halfBits (std::nanf ("")) == 0, "NaN becomes zero");
    check (pt::halfBits (1.0f / 3.0f) == 0x3555 && pt::halfBits (5.9604645e-8f) == 0x0001
               && pt::halfBits (6.1035156e-5f) == 0x0400,
           "half conversion rounds to nearest and keeps subnormals");

    // Looking away from the cube and backdrops toward the floor's near edge:
    // the bottom rows hit the floor, the top rows miss everything.
    auto scene = std::make_shared<grid::Visual3DScene> (*glass::makeScene());
    scene->cameras[0].transform.rotation = glass::axisAngle (0.0f, 1.0f, 0.0f, 3.14159265f);
    scene->cameras[0].transform.translation = { 0.0f, 0.3f, 2.0f };
    std::shared_ptr<const grid::Visual3DScene> frozen = scene;
    auto setup = glass::makeSetup (frozen);
    setup.lights.resize (1);  // the sun only: no sky behind the scene
    const auto view = glass::makeView (*frozen, 16, 8);
    pt::SceneEngine engine (2);
    pt::EngineImage image;
    preview (engine, setup, view, image);
    const std::size_t pixels = 16u * 8u;
    check (image.width == 16 && image.height == 8 && image.color == pt::EngineColor::DisplayRgba8
               && image.colorBytes.size() == pixels * 4 && image.depth.size() == pixels,
           "Display output is 4 bytes per pixel with one depth value per pixel");
    bool opaque = true, depthRange = true;
    for (std::size_t i = 0; i < pixels; ++i)
    {
        opaque = opaque && image.colorBytes[i * 4 + 3] == 255;
        depthRange = depthRange && image.depth[i] >= 0.0f && image.depth[i] <= 1.0f;
    }
    check (opaque, "alpha is 1 everywhere");
    check (depthRange, "depth stays in [0, 1]");
    const std::size_t bottom = 0, top = static_cast<std::size_t> (7) * 16;
    check (image.depth[bottom + 8] < 1.0f, "row 0 is the bottom of the picture, where the floor is");
    check (image.depth[top + 8] == 1.0f, "the last row is the top of the picture, where rays miss");
    check (image.colorBytes[(top + 8) * 4 + 0] == 7 && image.colorBytes[(top + 8) * 4 + 1] == 10
               && image.colorBytes[(top + 8) * 4 + 2] == 18,
           "a miss without a sky shows the raster clear colour");

    pt::SceneEngine hdr (2);
    preview (hdr, setup, view, image, pt::EngineColor::LinearRgba16f);
    check (image.color == pt::EngineColor::LinearRgba16f && image.colorBytes.size() == pixels * 8,
           "linear output is 8 bytes per pixel");
    bool halfAlpha = true;
    for (std::size_t i = 0; i < pixels; ++i)
        halfAlpha = halfAlpha && halfAt (image.colorBytes, i, 3) == 0x3c00;
    check (halfAlpha, "linear alpha is half 1.0");
    check (halfAt (image.colorBytes, top + 8, 0) == pt::halfBits (pt::kLinearRasterBackground[0])
               && halfAt (image.colorBytes, top + 8, 2) == pt::halfBits (pt::kLinearRasterBackground[2]),
           "a linear miss carries the clear colour in linear values");
    // The compositor encodes a linear scene as sRGB. The miss must come out as
    // the clear colour, not as that colour encoded a second time (46, 56, 75).
    const auto displayByte = [] (float linear) {
        const float encoded = linear <= 0.0031308f ? 12.92f * linear
                                                   : 1.055f * std::pow (linear, 1.0f / 2.4f) - 0.055f;
        return static_cast<int> (encoded * 255.0f + 0.5f);
    };
    check (displayByte (pt::kLinearRasterBackground[0]) == 7 && displayByte (pt::kLinearRasterBackground[1]) == 10
               && displayByte (pt::kLinearRasterBackground[2]) == 18,
           "the linear clear colour displays as (7, 10, 18)");

    // With the analytic sky the raster draws the sky, and so does the tracer.
    auto skySetup = glass::makeSetup (frozen);
    check (pt::rasterDrawsSkyBackground (skySetup.lights) && ! pt::rasterDrawsSkyBackground (setup.lights),
           "the sky background follows the raster's rule");
    pt::SceneEngine sky (2);
    preview (sky, skySetup, view, image);
    const std::uint8_t* missed = image.colorBytes.data() + (top + 8) * 4;
    check (! (missed[0] == 7 && missed[1] == 10 && missed[2] == 18) && missed[2] > missed[0],
           "a miss under a sky shows the sky");
    check (image.depth[top + 8] == 1.0f, "sky pixels keep depth 1");
}

// ---------------------------------------------------------------- environment hidden from camera

// The Environment Light's "Visible to camera" toggle. Off, a camera ray that
// misses shows the raster clear colour, while a mirror floor still reflects
// the sky in the direction the top rows see directly.
void environmentCameraTests()
{
    auto scene = std::make_shared<grid::Visual3DScene> (*glass::makeScene());
    scene->cameras[0].transform.rotation = glass::axisAngle (0.0f, 1.0f, 0.0f, 3.14159265f);
    scene->cameras[0].transform.translation = { 0.0f, 0.3f, 2.0f };
    std::shared_ptr<const grid::Visual3DScene> frozen = scene;
    pt::Material mirror;
    mirror.baseColor = { 1.0f, 1.0f, 1.0f };
    mirror.metallic = 1.0f;
    mirror.roughness = 0.0f;
    mirror.opacity = 1.0f;
    auto visible = glass::makeSetup (frozen);
    visible.objectMaterials.push_back ({ glass::kFloorObject, mirror });
    auto hidden = visible;
    hidden.lights[1].environmentCameraVisible = false;
    auto sunOnly = visible;
    sunOnly.lights.resize (1);

    check (! pt::rasterDrawsSkyBackground (hidden.lights) && pt::rasterDrawsSkyBackground (visible.lights),
           "a hidden environment draws no camera background");
    auto image = hidden.lights;
    image[1].sky = {};
    image[1].environmentImage.version = 1;
    check (! pt::rasterDrawsSkyBackground (image), "a hidden environment image draws no camera background");
    image[1].environmentCameraVisible = true;
    check (pt::rasterDrawsSkyBackground (image), "a visible environment image draws the camera background");

    const auto view = glass::makeView (*frozen, 16, 8);
    std::string error;
    const auto render = [&] (const pt::EngineSceneSetup& setup) {
        pt::SceneEngine engine (2);
        pt::EngineImage result;
        check (engine.renderExport (setup, view, pt::EngineColor::LinearRgba16f, 4, result, error),
               "environment camera export renders: " + error);
        return result;
    };
    const auto shown = render (visible);
    const auto blank = render (hidden);
    const auto unlit = render (sunOnly);
    const std::size_t pixels = 16u * 8u, floorPixel = 8, skyPixel = static_cast<std::size_t> (7) * 16 + 8;
    if (shown.colorBytes.size() != pixels * 8 || blank.colorBytes.size() != pixels * 8
        || unlit.colorBytes.size() != pixels * 8)
    {
        check (false, "environment camera exports have one half pixel per pixel");
        return;
    }
    check (blank.depth[skyPixel] == 1.0f && blank.depth[floorPixel] < 1.0f,
           "the top row misses and the bottom row hits the mirror floor");
    check (halfAt (blank.colorBytes, skyPixel, 0) == pt::halfBits (pt::kLinearRasterBackground[0])
               && halfAt (blank.colorBytes, skyPixel, 1) == pt::halfBits (pt::kLinearRasterBackground[1])
               && halfAt (blank.colorBytes, skyPixel, 2) == pt::halfBits (pt::kLinearRasterBackground[2]),
           "a camera miss under a hidden environment is the raster clear colour");
    check (halfAt (shown.colorBytes, skyPixel, 2) != pt::halfBits (pt::kLinearRasterBackground[2]),
           "a camera miss under a visible environment is the sky");
    // This view's floor reaches only the bottom row, whose pixels all cross the floor's far
    // edge, and a sample that misses there sees what hiding changes. So hiding is judged on a
    // view pitched down at the floor. A pixel's samples land anywhere inside it and depth
    // traces the centre only, so a fine grid of the same view says which pixels hold no miss.
    auto pitchedScene = std::make_shared<grid::Visual3DScene> (*scene);
    pitchedScene->cameras[0].transform.rotation = glass::multiply (scene->cameras[0].transform.rotation,
                                                                   glass::axisAngle (1.0f, 0.0f, 0.0f, -0.35f));
    std::shared_ptr<const grid::Visual3DScene> pitchedFrozen = pitchedScene;
    auto pitchedVisible = glass::makeSetup (pitchedFrozen);
    pitchedVisible.objectMaterials.push_back ({ glass::kFloorObject, mirror });
    auto pitchedHidden = pitchedVisible;
    pitchedHidden.lights[1].environmentCameraVisible = false;
    const auto renderPitched = [&] (const pt::EngineSceneSetup& setup, std::uint32_t scale, std::uint32_t samples) {
        pt::SceneEngine engine (2);
        pt::EngineImage result;
        check (engine.renderExport (setup, glass::makeView (*pitchedFrozen, 16 * scale, 8 * scale),
                                    pt::EngineColor::LinearRgba16f, samples, result, error),
               "pitched environment export renders: " + error);
        return result;
    };
    constexpr int kCells = 16;
    const auto fine = renderPitched (pitchedHidden, kCells, 1);
    const auto pitchedShown = renderPitched (pitchedVisible, 1, 4);
    const auto pitchedBlank = renderPitched (pitchedHidden, 1, 4);
    if (fine.depth.size() != pixels * kCells * kCells || pitchedShown.colorBytes.size() != pixels * 8
        || pitchedBlank.colorBytes.size() != pixels * 8)
    {
        check (false, "the pitched exports have the view's pixels and one depth per grid cell");
        return;
    }
    const auto fullyCovered = [&] (std::size_t pixel) {
        const int x = static_cast<int> (pixel % 16) * kCells, y = static_cast<int> (pixel / 16) * kCells;
        for (int cy = std::max (y - 1, 0); cy <= std::min (y + kCells, 8 * kCells - 1); ++cy)
            for (int cx = std::max (x - 1, 0); cx <= std::min (x + kCells, 16 * kCells - 1); ++cx)
                if (fine.depth[static_cast<std::size_t> (cy) * 16 * kCells + static_cast<std::size_t> (cx)] >= 1.0f)
                    return false;
        return true;
    };
    bool coveredSame = true;
    std::size_t covered = 0;
    for (std::size_t i = 0; i < pixels; ++i)
        if (fullyCovered (i))
        {
            ++covered;
            for (int c = 0; c < 4; ++c)
                coveredSame = coveredSame && halfAt (pitchedBlank.colorBytes, i, c) == halfAt (pitchedShown.colorBytes, i, c);
        }
    check (covered >= 16 && covered < pixels, "the pitched view has fully covered pixels and pixels that miss");
    check (coveredSame, "every fully covered pixel, mirror reflections included, is unchanged by hiding the environment");
    check (halfAt (blank.colorBytes, floorPixel, 2) != halfAt (unlit.colorBytes, floorPixel, 2),
           "the mirror floor still reflects the hidden environment's radiance");
}

// ---------------------------------------------------------------- export

float halfValue (const std::uint8_t* bytes)
{
    std::uint16_t bits = 0;
    std::memcpy (&bits, bytes, sizeof (bits));
    const int exponent = (bits >> 10) & 0x1f;
    const int fraction = bits & 0x3ff;
    const float magnitude = exponent == 0 ? std::ldexp (static_cast<float> (fraction), -24)
                                          : std::ldexp (static_cast<float> (1024 + fraction), exponent - 25);
    return (bits & 0x8000u) != 0 ? -magnitude : magnitude;
}

// R5: a frame-tinted object's camera samples leave the colour for its layer.
// The two add up to the film without the binding, which the same sample streams
// trace, so a white frame reproduces that film.
void frameLayerTests()
{
    auto authored = *glass::makeScene();
    for (std::size_t index = 0; index < authored.vertexCount; ++index)
        authored.vertices[index].uv = { 0.25f, 0.75f };
    const auto scene = std::make_shared<const grid::Visual3DScene> (std::move (authored));
    auto plainSetup = glass::makeSetup (scene);
    const auto view = glass::makeView (*scene, 20, 12);
    auto boundSetup = plainSetup;
    boundSetup.objectFrames.push_back ({ glass::kFloorObject, 0 });
    check (! pt::sameSceneSetup (plainSetup, boundSetup), "a frame binding is part of the scene setup");

    std::string error;
    pt::SceneEngine plainEngine (2), boundEngine (2);
    pt::EngineImage plain, bound;
    check (plainEngine.renderExport (plainSetup, view, pt::EngineColor::LinearRgba16f, 8, plain, error), "plain export: " + error);
    check (boundEngine.renderExport (boundSetup, view, pt::EngineColor::LinearRgba16f, 8, bound, error), "bound export: " + error);
    check (plain.frameLayers.empty(), "a scene without a frame binding has no layers");
    check (bound.frameLayers.size() == 1, "one binding gives one layer");
    if (bound.frameLayers.size() != 1) return;
    const auto& layer = bound.frameLayers[0];
    const std::size_t pixels = static_cast<std::size_t> (view.width) * view.height;
    check (layer.lightBytes.size() == pixels * 8 && layer.uvCoverage.size() == pixels * 3, "the layer covers the film");
    check (bound.depth == plain.depth, "the binding does not change the depth");

    bool sums = true, uvExact = true, coverageInRange = true;
    float covered = 0.0f, moved = 0.0f;
    for (std::size_t pixel = 0; pixel < pixels; ++pixel)
    {
        for (std::size_t channel = 0; channel < 3; ++channel)
        {
            const float whole = halfValue (plain.colorBytes.data() + (pixel * 4 + channel) * 2);
            const float rest = halfValue (bound.colorBytes.data() + (pixel * 4 + channel) * 2);
            const float light = halfValue (layer.lightBytes.data() + (pixel * 4 + channel) * 2);
            moved += light;
            if (std::abs (rest + light - whole) > 0.002f * std::max (1.0f, whole))
                sums = false;
        }
        const float* uv = layer.uvCoverage.data() + pixel * 3;
        covered += uv[2];
        if (uv[2] < 0.0f || uv[2] > 1.0f) coverageInRange = false;
        if (uv[2] > 0.0f && (std::abs (uv[0] - 0.25f) > 1.0e-3f || std::abs (uv[1] - 0.75f) > 1.0e-3f))
            uvExact = false;
    }
    check (covered > 0.0f && moved > 0.0f, "the bound object's samples reach the layer");
    check (covered < static_cast<float> (pixels), "the rest of the film stays out of the layer");
    check (sums, "colour plus layer light is the unbound film");
    check (uvExact, "the layer's uv is the surface's uv");
    check (coverageInRange, "coverage is a share of the pixel's samples");

    // A new generation starts the layer over.
    pt::SceneEngine previewEngine (1);
    pt::EngineImage first, second;
    preview (previewEngine, boundSetup, view, first, pt::EngineColor::LinearRgba16f);
    auto nextTime = view;
    nextTime.timeSeconds += 1.0f;
    preview (previewEngine, boundSetup, nextTime, second, pt::EngineColor::LinearRgba16f);
    check (second.restarted && second.samplesPerPixel >= 1 && second.frameLayers.size() == 1, "a new generation resolves its layer");
    float maxCoverage = 0.0f;
    if (second.frameLayers.size() == 1)
        for (std::size_t pixel = 0; pixel < pixels; ++pixel)
            maxCoverage = std::max (maxCoverage, second.frameLayers[0].uvCoverage[pixel * 3 + 2]);
    check (maxCoverage <= 1.0f, "a restarted layer keeps no samples from the last generation");
}

void exportTests()
{
    Fixture f;
    const auto view = glass::makeView (*f.scene, 20, 12);
    std::string error;
    const auto render = [&] (pt::SceneEngine& engine, std::uint32_t spp) {
        pt::EngineImage image;
        check (engine.renderExport (f.setup, view, pt::EngineColor::LinearRgba16f, spp, image, error),
               "export renders: " + error);
        return image;
    };
    pt::SceneEngine a (1);
    pt::SceneEngine b (3);
    const auto first = render (a, 8);
    check (first.generation == pt::kExportGeneration && first.samplesPerPixel == 8 && first.traced,
           "export renders the requested samples with the export generation");
    check (first.stats.samplesAdded + first.stats.nonFiniteSamples == 8ull * view.width * view.height,
           "export traces exactly samplesPerPixel per pixel");
    const auto second = render (b, 8);
    check (first.colorBytes == second.colorBytes && first.depth == second.depth,
           "export is identical on any thread count");

    pt::EngineImage scratch;
    preview (a, f.setup, view, scratch);
    preview (a, f.setup, view, scratch);
    const auto third = render (a, 8);
    check (third.colorBytes == first.colorBytes, "preview between exports does not change the export");
    preview (a, f.setup, view, scratch);
    check (scratch.restarted && scratch.samplesPerPixel == 1, "preview starts over after an export");

    const auto more = render (a, 16);
    check (more.colorBytes != first.colorBytes, "more samples give a different image");

    pt::EngineImage refused;
    check (! a.renderExport (f.setup, view, pt::EngineColor::DisplayRgba8, 0, refused, error) && ! error.empty(),
           "zero export samples are refused");
}

// ---------------------------------------------------------------- failure policy

bool finiteHalves (const std::vector<std::uint8_t>& bytes)
{
    for (std::size_t pixel = 0; pixel < bytes.size() / 8; ++pixel)
        for (int channel = 0; channel < 4; ++channel)
            if ((halfAt (bytes, pixel, channel) & 0x7c00u) == 0x7c00u)
                return false;
    return true;
}

void limitTests()
{
    Fixture f;
    const auto view = glass::makeView (*f.scene, 20, 12);
    const std::uint64_t samples = 4ull * view.width * view.height;
    const auto render = [&] (pt::SceneEngine& engine, pt::EngineImage& image, std::string& error) {
        return engine.renderExport (f.setup, view, pt::EngineColor::LinearRgba16f, 4, image, error);
    };

    pt::SceneEngine reference (2);
    pt::EngineImage unlimited;
    std::string error;
    check (render (reference, unlimited, error), "the glass scene renders with the default limits: " + error);

    // Zero bounces end every camera path at its first hit, before it gathers
    // any light there. The frame still renders.
    {
        pt::SceneEngine engine (2);
        engine.settings().maxDepth = 0;
        pt::EngineImage image;
        error = "unchanged";
        check (render (engine, image, error) && error.empty() && image.samplesPerPixel == 4,
               "zero bounces render every sample: " + error);
        check (image.stats.nonFiniteSamples == 0, "zero bounces stay finite");
        check (image.stats.bounceLimitPaths > 0, "zero bounces end paths at the limit");
    }

    // Through the glass cube a camera path needs more bounces than one, so
    // with maxDepth 1 most of them reach the limit. They end there and the
    // frame still renders every sample.
    {
        pt::SceneEngine engine (2);
        engine.settings().maxDepth = 1;
        pt::EngineImage image;
        error = "unchanged";
        const bool ok = render (engine, image, error);
        check (ok && error.empty(), "paths past maxDepth do not refuse the frame: " + error);
        check (image.traced && image.samplesPerPixel == 4 && image.stats.tilesCompleted == image.stats.tileCount
                   && ! image.stats.cancelled,
               "the depth-limited frame is complete");
        check (image.stats.samplesAdded == samples && image.stats.nonFiniteSamples == 0,
               "every depth-limited sample reaches the image");
        check (image.stats.bounceLimitPaths > 0 && image.stats.limitEndedPaths >= image.stats.bounceLimitPaths,
               "paths that ended at maxDepth are counted");
        check (image.colorBytes.size() == static_cast<std::size_t> (view.width) * view.height * 8
                   && finiteHalves (image.colorBytes),
               "the depth-limited frame is a finite image");
        check (image.colorBytes != unlimited.colorBytes, "ending paths early changes the image, it does not drop it");
    }

    // The per-kind limits. Light that crosses the cube transmits twice, so a
    // transmission limit of 1 ends those paths inside the glass.
    {
        pt::SceneEngine engine (2);
        engine.settings().maxTransmissionBounces = 1;
        pt::EngineImage image;
        check (render (engine, image, error), "paths past the transmission limit do not refuse the frame: " + error);
        check (image.stats.samplesAdded == samples && image.stats.bounceLimitPaths > 0,
               "transmission-limited paths contribute and are counted");
        check (image.colorBytes != unlimited.colorBytes, "the transmission limit changes the glass");
    }
    {
        pt::SceneEngine engine (2);
        engine.settings().maxDiffuseBounces = 0;
        engine.settings().maxGlossyBounces = 0;
        pt::EngineImage image;
        check (render (engine, image, error), "zero diffuse and glossy bounces still render: " + error);
        check (image.stats.samplesAdded == samples && image.stats.bounceLimitPaths > 0,
               "diffuse- and glossy-limited paths contribute and are counted");
    }
    {
        // Limits that never bind leave the image bit for bit as it was.
        pt::SceneEngine engine (2);
        engine.settings().maxDiffuseBounces = 1000;
        engine.settings().maxGlossyBounces = 1000;
        engine.settings().maxTransmissionBounces = 1000;
        pt::EngineImage image;
        check (render (engine, image, error) && image.colorBytes == unlimited.colorBytes,
               "per-kind limits that never bind do not change the export");
    }

    // The transparent limit. As a thin sheet the cube's faces pass light
    // straight through, and each crossing counts against maxTransparentHits.
    {
        auto setup = f.setup;
        pt::Material sheet = glass::glassMaterial();
        sheet.transport = pt::DielectricTransport::ThinSheet;
        setup.objectMaterials = { { glass::kGlassObject, sheet } };
        pt::SceneEngine engine (2);
        engine.settings().maxTransparentHits = 0;
        pt::EngineImage image;
        check (engine.renderExport (setup, view, pt::EngineColor::LinearRgba16f, 4, image, error),
               "paths past the transparent limit do not refuse the frame: " + error);
        check (image.stats.samplesAdded == samples && image.stats.transparentLimitPaths > 0
                   && image.stats.limitEndedPaths
                          == image.stats.bounceLimitPaths + image.stats.transparentLimitPaths,
               "transparent-limited paths contribute and are counted with the others");
    }
}

void nonFiniteTests()
{
    // An infinite emitter on the warm backdrop: camera paths that see it turn
    // non-finite. Those samples are dropped and counted; the frame renders.
    Fixture f;
    auto setup = f.setup;
    pt::Material broken;
    broken.emission = { std::numeric_limits<float>::infinity(), 1.0f, 1.0f };
    setup.objectMaterials.push_back ({ glass::kWarmBackdropObject, broken });
    const auto view = glass::makeView (*f.scene, 20, 12);

    pt::SceneEngine engine (2);
    pt::EngineImage image;
    std::string error = "unchanged";
    const bool ok = engine.renderExport (setup, view, pt::EngineColor::LinearRgba16f, 4, image, error);
    check (ok && error.empty(), "non-finite samples do not refuse the frame: " + error);
    check (image.stats.nonFiniteSamples > 0, "non-finite samples are counted per frame");
    check (image.stats.samplesAdded > 0
               && image.stats.samplesAdded + image.stats.nonFiniteSamples == 4ull * view.width * view.height,
           "every sample is either added or dropped");
    check (finiteHalves (image.colorBytes), "the frame with dropped samples is a finite image");

    pt::EngineImage preview;
    check (engine.renderPreview (setup, view, pt::EngineColor::DisplayRgba8, preview, error) && error.empty()
               && preview.stats.nonFiniteSamples > 0,
           "a preview pass also drops and counts non-finite samples: " + error);
}

void invalidInputTests()
{
    // Invalid scene input still refuses, with its cause, whatever the limits.
    Fixture f;
    pt::SceneEngine engine (2);
    engine.settings().maxDepth = 0;
    engine.settings().maxTransparentHits = 0;
    pt::EngineImage image;
    std::string error;

    auto setup = f.setup;
    setup.scene = nullptr;
    check (! engine.renderExport (setup, f.view, pt::EngineColor::DisplayRgba8, 2, image, error)
               && error == "Path Trace requires a scene snapshot",
           "a missing scene refuses with its cause");
    check (! engine.renderPreview (setup, f.view, pt::EngineColor::DisplayRgba8, image, error)
               && error == "Path Trace requires a scene snapshot",
           "a missing scene refuses the preview with its cause");

    setup = f.setup;
    setup.objectWorlds.assign (1, pt::Mat4 {});
    check (f.scene->objectCount != 1
               && ! engine.renderExport (setup, f.view, pt::EngineColor::DisplayRgba8, 2, image, error)
               && error == "Path Trace object worlds must list every scene object",
           "object worlds that miss objects refuse with their cause");

    auto view = f.view;
    view.width = 0;
    check (! engine.renderExport (f.setup, view, pt::EngineColor::DisplayRgba8, 2, image, error)
               && error == "Path Trace requires a non-empty image",
           "an empty image refuses with its cause");

    view = f.view;
    view.camera.nearPlane = std::numeric_limits<float>::quiet_NaN();
    check (! engine.renderExport (f.setup, view, pt::EngineColor::DisplayRgba8, 2, image, error)
               && error == "Path Trace requires a finite camera with a near plane in front of the far plane",
           "a non-finite camera refuses with its cause");

    // The same engine renders once the input is valid.
    check (engine.renderExport (f.setup, f.view, pt::EngineColor::DisplayRgba8, 2, image, error) && error.empty(),
           "valid input renders under the tightest limits: " + error);
}

} // namespace

// ---------------------------------------------------------------- export film hook

void setFilmPath (const char* path)
{
#if defined(_WIN32)
    _putenv_s ("ARBIT_PATHTRACE_EXPORT_FILM", path);
#else
    if (*path != '\0')
        setenv ("ARBIT_PATHTRACE_EXPORT_FILM", path, 1);
    else
        unsetenv ("ARBIT_PATHTRACE_EXPORT_FILM");
#endif
}

void exportFilmTests()
{
    Fixture f;
    const auto path = std::filesystem::temp_directory_path() / "arbit-pathtrace-export-film-test.pfm";
    const std::string pathText = path.string();
    std::filesystem::remove (path);
    std::string error;

    pt::SceneEngine unhooked (2);
    pt::EngineImage plain;
    check (unhooked.renderExport (f.setup, f.view, pt::EngineColor::LinearRgba16f, 8, plain, error),
           "an export renders without the film hook: " + error);
    check (! std::filesystem::exists (path), "an export writes no film unless asked");

    setFilmPath (pathText.c_str());
    pt::SceneEngine hooked (2);
    pt::EngineImage image;
    check (hooked.renderExport (f.setup, f.view, pt::EngineColor::LinearRgba16f, 8, image, error),
           "an export renders with the film hook: " + error);
    check (image.colorBytes == plain.colorBytes, "the film hook does not change the export");

    const std::size_t pixels = static_cast<std::size_t> (f.view.width) * f.view.height;
    const std::string header = "PF\n" + std::to_string (f.view.width) + " " + std::to_string (f.view.height) + "\n-1.0\n";
    std::string bytes (header.size() + pixels * 12, '\0');
    if (std::FILE* file = std::fopen (pathText.c_str(), "rb"))
    {
        check (std::fread (bytes.data(), 1, bytes.size(), file) == bytes.size() && std::fgetc (file) == EOF,
               "the film file holds exactly its header and pixels");
        std::fclose (file);
    }
    else
    {
        check (false, "the film hook writes its file");
    }
    check (bytes.compare (0, header.size(), header) == 0, "the film file has a PFM header");

    // Both run bottom row first. The half-float export rounds to about one part in a thousand.
    bool matches = true;
    for (std::size_t value = 0; value < pixels * 3; ++value)
    {
        float film = 0.0f;
        std::memcpy (&film, bytes.data() + header.size() + value * 4, sizeof (film));
        const float shown = halfValue (image.colorBytes.data() + (value / 3 * 4 + value % 3) * 2);
        matches = matches && std::abs (film - shown) <= 0.002f * std::max (1.0f, std::abs (shown));
    }
    check (matches, "the film holds the linear export's radiance");

    setFilmPath ((path / "missing" / "film.pfm").string().c_str());
    pt::SceneEngine refused (2);
    pt::EngineImage refusedImage;
    check (! refused.renderExport (f.setup, f.view, pt::EngineColor::LinearRgba16f, 8, refusedImage, error)
               && error.find ("could not write its film") != std::string::npos,
           "a film that can't be written fails the export");

    setFilmPath ("");
    std::filesystem::remove (path);
}

int main()
{
    contractTests();
    cacheTests();
    progressiveTests();
    playbackTests();
    temporalHistoryTests();
    formatTests();
    environmentCameraTests();
    exportTests();
    frameLayerTests();
    limitTests();
    nonFiniteTests();
    invalidInputTests();
    exportFilmTests();
    std::cout << checks << " checks, " << failures << " failures\n";
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
