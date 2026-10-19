// Tests for the Path Trace render engine (Plan 3, PT-C1): the engine field on
// the 3D Render wire, the RenderScene cache, progressive reset rules, the
// output format the compositor takes, and export determinism.

#include "pathtrace/engine.h"
#include "support/pathtrace_glass_scene.h"
#include "../../shared/VisualImportedSceneRenderOperationContract.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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

void contractTests()
{
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

    auto refused = traced;
    refused.pathTraceExportSamples = 0;
    check (render::encode (refused).empty(), "zero export samples are refused");
    refused.pathTraceExportSamples = render::kMaximumPathTraceExportSamples + 1;
    check (render::encode (refused).empty(), "export samples above the maximum are refused");
    refused = traced;
    refused.imageOutput = renderpassoutput::Output::Depth;
    check (render::encode (refused).empty(), "Path Trace refuses a non-Color image output");
    refused = traced;
    refused.rawExportMask = 1u << static_cast<unsigned> (renderpassoutput::Output::Depth);
    check (render::encode (refused).empty(), "Path Trace refuses raw data exports");
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
    check (render::decode (header + "1 4096 " + size + "\n" + rasterWire, ignored)
               && ignored.pathTraceExportSamples == render::kMaximumPathTraceExportSamples,
           "the maximum export samples decode");
    check (! render::decode (header + "1 4097 " + size + "\n" + rasterWire, ignored),
           "export samples above 4096 are refused");
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

    setup.scene = std::make_shared<const grid::Visual3DScene> (*f.scene);
    preview (engine, setup, view, image);
    check (engine.sceneBuilds() == 7, "a new scene snapshot rebuilds, even with equal content");

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

    pt::SceneEngine single (2);
    single.samplesPerPass = 0;
    preview (single, f.setup, f.view, image);
    check (image.samplesPerPixel == 1, "a zero pass size still adds one sample");
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
    check (halfAt (image.colorBytes, top + 8, 0) == pt::halfBits (pt::kRasterBackground[0])
               && halfAt (image.colorBytes, top + 8, 2) == pt::halfBits (pt::kRasterBackground[2]),
           "a linear miss carries the clear colour in linear values");

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

// ---------------------------------------------------------------- export

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

} // namespace

int main()
{
    contractTests();
    cacheTests();
    progressiveTests();
    formatTests();
    exportTests();
    std::cout << checks << " checks, " << failures << " failures\n";
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
