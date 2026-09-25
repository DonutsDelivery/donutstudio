#pragma once

#include "../../src/imported_scene_visual_plan_execution.h"
#include "../../src/render_snapshot_json.h"
#include "../../src/renderer.h"
#include "../../../shared/HdrImageOutputContract.h"

#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <vector>

namespace typedscenepasstests
{
inline bool run(const std::string& file, const arbitgl::GlFuncs& gl, std::string& error)
{
    using namespace videohelper::importedscene;
    using namespace videohelper::modelpayload;
    using namespace arbitgpu;
    const auto reject = [&](const std::string& message) { error = message; return false; };
    renderpasscomposite::Parameters parameters;
    parameters.mode = renderpasscomposite::Mode::DepthView;
    parameters.nearDepth = 0.25f; parameters.farDepth = 0.75f;
    parameters.amount = 1; parameters.fogColor = {0.5f, 0.5f, 0.5f}; parameters.identity = 0;
    if (renderpasscomposite::encode(parameters) != "RenderPassCompositeV1 2 0.25 0.75 1 0.5 0.5 0.5 0")
        return reject("Typed inspection changed the existing pass serialization");
    parameters.inspectionImage = parameters.invertDepth = true;
    renderpasscomposite::Parameters decoded;
    const std::string inspectionWire = "RenderPassCompositeV3 2 0.25 0.75 1 0.5 0.5 0.5 0 0 0 0 1";
    if (renderpasscomposite::encode(parameters) != inspectionWire
        || !renderpasscomposite::decode(inspectionWire, decoded)
        || !decoded.inspectionImage || !decoded.invertDepth
        || renderpasscomposite::gpuProgram(renderpasscomposite::single(decoded)).transforms[0][3] != 3)
        return reject("Typed inspection lost opaque-image or invert semantics across pass serialization");
    decoded.mode = renderpasscomposite::Mode::NormalView;
    if (renderpasscomposite::valid(decoded)
        || renderpasscomposite::decode("RenderPassCompositeV3 2 0.25 0.75 1 0.5 0.5 0.5 0 0 0 0 2", decoded))
        return reject("Typed inspection accepted an invalid mode or non-boolean inversion");
    std::ifstream input(file);
    const auto document = nlohmann::json::parse(input, nullptr, false);
    if (document.is_discarded() || document.value("schema", "") != "typed-scene-pass-production-fixtures-v1"
        || !document.contains("rows") || !document["rows"].is_array() || document["rows"].size() != 10)
        return reject("Native typed-pass checks require the ten fixtures emitted by the real plugin compiler");
    const int width = document.value("width", 0), height = document.value("height", 0);
    if (width < 8 || height < 8 || width > 4096 || height > 4096) return reject("Invalid fixture extent");
    videorender::FrameRenderer compositor;
    if (!compositor.initialize(&gl, width, height, error)) return false;
    compositor.setHdrImageCapture(false);
    Store store;
    ImportedScenePayloadExecution execution(store, nativeFixtureSceneBackend());
    VisualImportedScenePlanCache cache;
    std::map<std::string, std::vector<std::uint8_t>> rendered;
    for (const auto& row : document["rows"])
    {
        const auto name = row.value("name", "");
        videowire::ResolvedVisualSnapshot snapshot;
        if (!videowire::parseSnapshotJson(row, [](const auto&) { return std::string{}; }, snapshot, error)
            || snapshot.visualLayerPlans.size() != 1) return false;
        const auto& plan = snapshot.visualLayerPlans.front();
        typedscenepass::Payload payload;
        std::optional<aovinspection::Payload> inspection;
        videowire::CompiledVisualLayerPlan base;
        if (!videowire::lowerTypedScenePass(plan, payload, inspection, base, error)) return false;
        if (payload.extent != renderpassoutput::Extent {
                static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height)})
            return reject(name + ": plugin fixture lost its authored Render Passes extent");
        if (inspection && inspection->source == aovinspection::Source::Depth)
        {
            if (!row.contains("authoredNear") || !row.contains("authoredFar") || !row.contains("authoredInvert")
                || std::abs(inspection->depthNear - row.value("authoredNear", -1.0f)) > 0.000001f
                || std::abs(inspection->depthFar - row.value("authoredFar", -1.0f)) > 0.000001f
                || inspection->invertDepth != (row.value("authoredInvert", 0) == 1))
                return reject(name + ": plugin fixture controls did not persist into immutable typed inspection IR");
        }
        videowire::VisualPlanResourceUsage usage;
        if (!videowire::admitVisualPlanResources(plan, width, height, {}, usage, error)) return false;
        if (usage.frameSlots != 10 || usage.allocatedFrameBytes != static_cast<std::uint64_t>(width) * height * 53)
            return reject(name + ": resource admission does not cover the eight native attachments and two display targets");
        for (int fault = 0; fault < 7; ++fault)
        {
            auto malformed = plan;
            if (fault == 0) malformed.ports.back().pixelFormat = "rgba8";
            if (fault == 1) malformed.edges.clear();
            if (fault == 2) ++malformed.structuralRevision;
            if (fault == 3) malformed.operations[1].payloadXml.push_back('x');
            if (fault == 4) malformed.ports.push_back(malformed.ports.back());
            if (fault == 5) malformed.operations[0].nodeId += 100;
            if (fault == 6) malformed.operations[1].backendCapability = "cpu";
            videowire::VisualLayerExecution ignored;
            std::string diagnostic;
            if (videowire::compileVisualLayerExecution(malformed, ignored, diagnostic))
                return reject(name + ": malformed typed schedule reached native execution");
        }
        auto wrongCount = plan;
        ++wrongCount.allocatedFrameSlotCount;
        if (videowire::admitVisualPlanResources(wrongCount, width, height, {}, usage, error))
            return reject(name + ": forged producer resource accounting was accepted");
        std::vector<std::uint8_t> preview;
        std::optional<videorender::LayerDesc> priorFrameLayer;
        for (const auto use : {NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export})
        {
            videorender::LayerDesc layer;
            layer.clipId = plan.clipId;
            std::vector<ImportedSceneExecutionReceipt> owners;
            if (prepareVisualImportedSceneLayerAtTime(snapshot.visualLayerPlans, plan.clipId, width, height,
                    0.5, 24, use, &execution, &cache, layer, owners, error, nullptr, {12, 34})
                    != VisualImportedScenePreparation::rendered || owners.size() != 1
                || !layer.nativeLinearImage || !layer.rawExportFrame || !layer.nativeTextureOwner)
                return reject(name + ": real scene preparation failed: " + error);
            const auto frame = layer.rawExportFrame;
            NativeRawPassPixels color, depth, normal, mask;
            if (!frame->readRawPass(renderpassoutput::Output::Color, color, error)
                || !frame->readRawPass(renderpassoutput::Output::Depth, depth, error)
                || !frame->readRawPass(renderpassoutput::Output::Normal, normal, error)
                || !frame->readRawPass(renderpassoutput::Output::Mask, mask, error)) return false;
            const auto pixels = static_cast<std::size_t>(width) * height;
            if (color.format != NativeTexturePixelFormat::Rgba16Float || color.bytes.size() != pixels * 8
                || depth.format != NativeTexturePixelFormat::R32Float || depth.bytes.size() != pixels * 4
                || normal.format != NativeTexturePixelFormat::Rgba16Float || normal.bytes.size() != pixels * 8
                || mask.format != NativeTexturePixelFormat::R8Unorm || mask.bytes.size() != pixels)
                return reject(name + ": typed passes lost their real native attachment formats");
            const auto half = [](const std::vector<std::uint8_t>& bytes, std::size_t scalar)
            {
                std::uint16_t bits;
                std::memcpy(&bits, bytes.data() + scalar * 2, 2);
                return hdrimage::halfToFloat(bits);
            };
            std::size_t covered = 0;
            for (std::size_t i = 0; i < pixels; ++i)
            {
                covered += mask.bytes[i] != 0;
                if (mask.bytes[i] && name != "mesh")
                {
                    // The authored triangle is at Z=0, with a camera at Z=2.5
                    // and planes .1/10. A +30-degree Y rotation has normal
                    // (sin(30),0,cos(30)); these oracles do not read the payload.
                    if (name != "normal-rotate")
                    {
                        float rawDepth;
                        std::memcpy(&rawDepth, depth.bytes.data() + i * 4, 4);
                        if (!std::isfinite(rawDepth) || std::abs(rawDepth - 0.242424242424f) > 0.0001f)
                            return reject(name + ": raw Depth disagrees with the independent authored camera/plane distance");
                    }
                    const std::array<float, 3> expectedNormal = name == "normal-rotate"
                        ? std::array<float,3>{0.5f, 0.0f, 0.866025403784f} : std::array<float,3>{0,0,1};
                    for (std::size_t c = 0; c < 3; ++c)
                        if (!std::isfinite(half(normal.bytes, i * 4 + c))
                            || std::abs(half(normal.bytes, i * 4 + c) - expectedNormal[c]) > 0.001f)
                            return reject(name + ": raw Normal disagrees with the independent authored rotation");
                }
                if (inspection)
                {
                    float rawDepth;
                    std::memcpy(&rawDepth, depth.bytes.data() + i * 4, 4);
                    for (std::size_t c = 0; c < 4; ++c)
                    {
                        double expected = 1;
                        if (c < 3 && inspection->source == aovinspection::Source::Depth)
                        {
                            expected = std::clamp((rawDepth - inspection->depthNear)
                                / (inspection->depthFar - inspection->depthNear), 0.0f, 1.0f);
                            if (inspection->invertDepth) expected = 1 - expected;
                        }
                        if (c < 3 && inspection->source == aovinspection::Source::Normal)
                            expected = half(normal.bytes, i * 4 + c) * 0.5 + 0.5;
                        if (!std::isfinite(expected) || std::abs(half(color.bytes, i * 4 + c) - expected) > 0.003)
                            return reject(name + ": inspection pixels disagree with the same-frame raw AOV and authored mapping");
                    }
                }
                else if (mask.bytes[i] && (name == "color" || name == "material"))
                {
                    const std::array<float, 3> expected = name == "color"
                        ? std::array<float,3>{0.02f, 0.1f, 0.25f} : std::array<float,3>{0.25f, 0.03f, 0.01f};
                    for (std::size_t c = 0; c < 3; ++c)
                        if (std::abs(half(color.bytes, i * 4 + c) - expected[c]) > 0.003f)
                            return reject(name + ": authored emission does not reach native Color");
                }
            }
            if (covered <= pixels / 200 || covered >= pixels * 3 / 4)
                return reject(name + ": scene fixture lost visible geometry or uncovered background");
            owners.clear();
            if (!glIsTexture(layer.texture)) return reject(name + ": frame-list release expired the compositor image");
            std::vector<std::uint8_t> sdr;
            if (compositor.hdrImageCaptureEnabled()
                || !compositor.renderToPixels(&layer, 1, sdr, error) || sdr.size() != pixels * 4) return false;
            for (std::size_t i = 0; i < sdr.size(); ++i)
            {
                const double raw = half(color.bytes, i);
                if (!std::isfinite(raw)) return reject(name + ": native Color contains a nonfinite pixel");
                const double linear = std::clamp(raw, 0.0, 1.0);
                const auto encoded = i % 4 == 3 ? linear
                    : linear <= 0.0031308 ? 12.92 * linear : 1.055 * std::pow(linear, 1.0 / 2.4) - 0.055;
                if (std::abs(static_cast<int>(sdr[i]) - std::lround(encoded * 255)) > 3)
                    return reject(name + ": normal SDR compositor lost canonical rows or sRGB conversion");
            }
            if (use == NativeImportedSceneRenderUse::Preview) preview = sdr;
            else if (preview != sdr) return reject(name + ": preview and video export disagree");
            NativeRawPassPixels after;
            if (!frame->readRawPass(renderpassoutput::Output::Color, after, error) || after.bytes != color.bytes)
                return reject(name + ": display conversion mutated the retained HDR source");
            for (int fault = 0; fault < 5; ++fault)
            {
                auto invalid = layer;
                if (fault == 0) invalid.nativeTextureDescriptor = {};
                if (fault == 1) ++invalid.nativeTextureDescriptor.rendererGeneration;
                if (fault == 2) invalid.nativeTextureDescriptor.deviceOrContextIdentity ^= 1u;
                if (fault == 3) invalid.nativeTextureDescriptor.format = NativeTexturePixelFormat::Rgba8Unorm;
                if (fault == 4) invalid.nativeTextureOwner.reset();
                if (compositor.renderComposite(&invalid, 1) != 0 || compositor.lastError().empty())
                    return reject(name + ": compositor accepted forged or unowned native metadata");
            }
            if (priorFrameLayer)
            {
                auto wrongFrame = layer;
                wrongFrame.nativeLinearImage = priorFrameLayer->nativeLinearImage;
                wrongFrame.nativeTextureOwner = priorFrameLayer->nativeTextureOwner;
                if (compositor.renderComposite(&wrongFrame, 1) != 0 || compositor.lastError().empty())
                    return reject(name + ": compositor accepted a retained frame that does not match the current descriptor");
            }
            priorFrameLayer = layer;
        }
        if (name == "color")
        {
            const int composedWidth = std::max(8, width / 2 + 3);
            const int composedHeight = std::max(8, height / 2 + 5);
            videowire::VisualPlanResourceUsage composedUsage;
            if (!videowire::admitVisualPlanResources(
                    plan, composedWidth, composedHeight, {}, composedUsage, error)) return false;
            if (composedUsage.allocatedFrameBytes != static_cast<std::uint64_t>(width) * height * 53)
                return reject("different composition extent changed authored typed-pass ownership accounting");
            videorender::LayerDesc layer;
            layer.clipId = plan.clipId;
            std::vector<ImportedSceneExecutionReceipt> owners;
            if (prepareVisualImportedSceneLayerAtTime(snapshot.visualLayerPlans, plan.clipId,
                    composedWidth, composedHeight, 0.5, 24, NativeImportedSceneRenderUse::Preview,
                    &execution, &cache, layer, owners, error, nullptr, {12, 34})
                    != VisualImportedScenePreparation::rendered
                || layer.texWidth != width || layer.texHeight != height
                || !layer.nativeLinearImage || layer.nativeLinearImage->width() != static_cast<std::uint32_t>(width)
                || layer.nativeLinearImage->height() != static_cast<std::uint32_t>(height))
                return reject("typed pass did not retain its authored extent before composition");
            videorender::FrameRenderer resizedCompositor;
            if (!resizedCompositor.initialize(&gl, composedWidth, composedHeight, error)) return false;
            resizedCompositor.setHdrImageCapture(false);
            std::vector<std::uint8_t> composed;
            if (!resizedCompositor.renderToPixels(&layer, 1, composed, error)
                || composed.size() != static_cast<std::size_t>(composedWidth) * composedHeight * 4)
                return reject("authored typed pass did not compose into a different viewport/export extent");
        }
        if (!rendered.emplace(name, std::move(preview)).second) return reject("Duplicate typed scene fixture name");
    }
    for (const auto* name : {"color", "material", "object", "camera", "mesh", "depth",
                             "depth-range", "depth-invert", "normal", "normal-rotate"})
        if (!rendered.count(name)) return reject(std::string("Missing producer fixture: ") + name);
    for (const auto* variant : {"material", "object", "camera", "mesh"})
        if (!rendered.count(variant) || rendered.at(variant) == rendered.at("color"))
            return reject(std::string(variant) + ": editing the actual scene did not change Color pixels");
    if (rendered.at("depth") == rendered.at("depth-range") || rendered.at("depth-range") == rendered.at("depth-invert")
        || rendered.at("normal") == rendered.at("normal-rotate"))
        return reject("Typed inspection ignores its authored range, inversion or object orientation");
    error.clear();
    return true;
}
} // namespace typedscenepasstests
