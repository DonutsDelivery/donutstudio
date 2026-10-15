#include "imported_scene_visual_plan_execution.h"
#include "support/vertex_modifier_fixture.h"
#include "support/render_pass_program_cases.h"
#include "support/raw_pass_export_checks.h"
#include "support/material_field_fixture.h"
#include "support/imported_starter_topology.h"
#include "../../shared/DiffractionMaterialPresets.h"
#include "../../shared/DiffractionProductPlans.h"
#include "geometry_scene_composition_fixture.h"
#include "composed_timeline_geometry_fixture.h"
#include "timeline_scene_family_evaluation_fixture.h"
#include "water_surface_replay_fixture.h"
#include "support/generated_surface_scene.h"
#include "support/smooth_glass_authored_material.h"
#include "reactive_frame_fixture.h"
#include "../src/renderer.h"
#include "../src/viewport_project_generation.h"
#include "../src/imported_animation_deformation_consumer.h"
#include "../src/sha256.h"
#include "../../shared/VisualImportedAnimationOperationContract.h"
#include <nlohmann/json.hpp>
#include "reactive_surface_fixture.h"
#include "mixed_surface_diffraction_fixture.h"
#include "../../shared/generated/SurfaceMaterialStarterPrograms.h"
#include <filesystem>
#include <fstream>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <tuple>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace
{
int failures = 0;

void check(bool condition, const std::string& message)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

// Failure-only PROBE output prints once per distinct conjunct pattern.
bool probeFirstTime(const std::string& pattern)
{
    static std::set<std::string> printed;
    return printed.insert(pattern).second;
}

// Two-action GLB used by the crossfade block: one joint and two morph targets,
// with "Walk" and "Reach" both driving the same mesh. Walk samples to
// joint (0.5,1,1.5) and morph (0.25,0.75) at 0.5s; Reach samples to
// joint (3.5,4.5,5.5) and morph (0.65,0.35) at 0.5s.
void crossfadeAppendU32(std::vector<std::uint8_t>& bytes, std::uint32_t value)
{
    for (unsigned shift = 0; shift < 32u; shift += 8u)
        bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void crossfadeAppendFloat(std::vector<std::uint8_t>& bytes, float value)
{
    std::uint32_t encoded = 0;
    std::memcpy(&encoded, &value, sizeof(encoded));
    crossfadeAppendU32(bytes, encoded);
}

std::vector<std::uint8_t> crossfadeGlb(nlohmann::json root, std::vector<std::uint8_t> bin)
{
    auto text = root.dump();
    while ((text.size() & 3u) != 0)
        text.push_back(' ');
    while ((bin.size() & 3u) != 0)
        bin.push_back(0);

    std::vector<std::uint8_t> bytes;
    const auto total = 12u + 8u + static_cast<unsigned>(text.size())
                     + (bin.empty() ? 0u : 8u + static_cast<unsigned>(bin.size()));
    crossfadeAppendU32(bytes, 0x46546c67u);
    crossfadeAppendU32(bytes, 2u);
    crossfadeAppendU32(bytes, total);
    crossfadeAppendU32(bytes, static_cast<std::uint32_t>(text.size()));
    crossfadeAppendU32(bytes, 0x4e4f534au);
    bytes.insert(bytes.end(), text.begin(), text.end());
    if (!bin.empty())
    {
        crossfadeAppendU32(bytes, static_cast<std::uint32_t>(bin.size()));
        crossfadeAppendU32(bytes, 0x004e4942u);
        bytes.insert(bytes.end(), bin.begin(), bin.end());
    }
    return bytes;
}

std::vector<std::uint8_t> crossfadeBin()
{
    std::vector<std::uint8_t> bin;
    for (const auto value : {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f})
        crossfadeAppendFloat(bin, value);
    for (int vertex = 0; vertex < 3; ++vertex)
        for (int component = 0; component < 4; ++component)
            bin.push_back(0);
    for (int vertex = 0; vertex < 3; ++vertex)
        for (const auto value : {1.0f, 0.0f, 0.0f, 0.0f})
            crossfadeAppendFloat(bin, value);
    for (int target = 0; target < 2; ++target)
        for (int value = 0; value < 9; ++value)
            crossfadeAppendFloat(bin, target == 0 && value == 0 ? 0.25f : 0.0f);
    for (int row = 0; row < 4; ++row)
        for (int column = 0; column < 4; ++column)
            crossfadeAppendFloat(bin, row == column ? 1.0f : 0.0f);
    for (const auto value : {0.0f, 1.0f})
        crossfadeAppendFloat(bin, value);
    for (const auto value : {0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f})
        crossfadeAppendFloat(bin, value);
    for (const auto value : {0.0f, 1.0f, 0.5f, 0.5f})
        crossfadeAppendFloat(bin, value);
    for (const auto value : {4.0f, 5.0f, 6.0f, 2.0f, 3.0f, 4.0f,
                             0.8f, 0.2f, 0.2f, 0.8f, 0.0f, 2.0f})
        crossfadeAppendFloat(bin, value);
    return bin;
}

nlohmann::json crossfadeRoot()
{
    const nlohmann::json channels = nlohmann::json::array({
        {{"sampler", 0}, {"target", {{"node", 1}, {"path", "translation"}}}},
        {{"sampler", 1}, {"target", {{"node", 0}, {"path", "weights"}}}}
    });
    const nlohmann::json walk = {
        {"name", "Walk"},
        {"samplers", nlohmann::json::array({
            {{"input", 6}, {"output", 7}, {"interpolation", "LINEAR"}},
            {{"input", 6}, {"output", 8}, {"interpolation", "LINEAR"}}
        })},
        {"channels", channels}
    };
    const nlohmann::json reach = {
        {"name", "Reach"},
        {"samplers", nlohmann::json::array({
            {{"input", 11}, {"output", 9}, {"interpolation", "LINEAR"}},
            {{"input", 11}, {"output", 10}, {"interpolation", "LINEAR"}}
        })},
        {"channels", channels}
    };
    return {
        {"asset", {{"version", "2.0"}}},
        {"nodes", nlohmann::json::array({
            {{"mesh", 0}, {"skin", 0}, {"children", nlohmann::json::array({1})}},
            nlohmann::json::object()
        })},
        {"meshes", nlohmann::json::array({{{"weights", nlohmann::json::array({0.0, 0.0})},
            {"primitives", nlohmann::json::array({{
                {"attributes", {{"POSITION", 0}, {"JOINTS_0", 1}, {"WEIGHTS_0", 2}}},
                {"targets", nlohmann::json::array({{{"POSITION", 3}}, {{"POSITION", 4}}})}
            }})}}})},
        {"skins", nlohmann::json::array({{{"joints", nlohmann::json::array({1})},
                                            {"inverseBindMatrices", 5}}})},
        {"animations", nlohmann::json::array({walk, reach})},
        {"buffers", nlohmann::json::array({{{"byteLength", 328}}})},
        {"bufferViews", nlohmann::json::array({
            {{"buffer", 0}, {"byteOffset", 0}, {"byteLength", 36}, {"target", 34962}},
            {{"buffer", 0}, {"byteOffset", 36}, {"byteLength", 12}, {"target", 34962}},
            {{"buffer", 0}, {"byteOffset", 48}, {"byteLength", 48}, {"target", 34962}},
            {{"buffer", 0}, {"byteOffset", 96}, {"byteLength", 36}, {"target", 34962}},
            {{"buffer", 0}, {"byteOffset", 132}, {"byteLength", 36}, {"target", 34962}},
            {{"buffer", 0}, {"byteOffset", 168}, {"byteLength", 64}},
            {{"buffer", 0}, {"byteOffset", 232}, {"byteLength", 8}},
            {{"buffer", 0}, {"byteOffset", 240}, {"byteLength", 24}},
            {{"buffer", 0}, {"byteOffset", 264}, {"byteLength", 16}},
            {{"buffer", 0}, {"byteOffset", 280}, {"byteLength", 24}},
            {{"buffer", 0}, {"byteOffset", 304}, {"byteLength", 16}},
            {{"buffer", 0}, {"byteOffset", 320}, {"byteLength", 8}}
        })},
        {"accessors", nlohmann::json::array({
            {{"bufferView", 0}, {"componentType", 5126}, {"count", 3}, {"type", "VEC3"}},
            {{"bufferView", 1}, {"componentType", 5121}, {"count", 3}, {"type", "VEC4"}},
            {{"bufferView", 2}, {"componentType", 5126}, {"count", 3}, {"type", "VEC4"}},
            {{"bufferView", 3}, {"componentType", 5126}, {"count", 3}, {"type", "VEC3"}},
            {{"bufferView", 4}, {"componentType", 5126}, {"count", 3}, {"type", "VEC3"}},
            {{"bufferView", 5}, {"componentType", 5126}, {"count", 1}, {"type", "MAT4"}},
            {{"bufferView", 6}, {"componentType", 5126}, {"count", 2}, {"type", "SCALAR"}},
            {{"bufferView", 7}, {"componentType", 5126}, {"count", 2}, {"type", "VEC3"}},
            {{"bufferView", 8}, {"componentType", 5126}, {"count", 4}, {"type", "SCALAR"}},
            {{"bufferView", 9}, {"componentType", 5126}, {"count", 2}, {"type", "VEC3"}},
            {{"bufferView", 10}, {"componentType", 5126}, {"count", 4}, {"type", "SCALAR"}},
            {{"bufferView", 11}, {"componentType", 5126}, {"count", 2}, {"type", "SCALAR"}}
        })}
    };
}

bool nearCrossfadeValue(float left, float right)
{
    return std::abs(left - right) < 0.0001f;
}

template <std::size_t N>
bool nearCrossfadeValues(const std::array<float, N>& left, const std::array<float, N>& right)
{
    for (std::size_t index = 0; index < N; ++index)
        if (!nearCrossfadeValue(left[index], right[index])) return false;
    return true;
}

// Evaluates one selection through the production consumer at timeline 0.5s and
// returns the single joint translation and the two morph weights it produced.
bool crossfadeEvaluate(const videohelper::ImportedAnimationDeformationConsumer& consumer,
                       const visualanimationimport::Request& request,
                       std::array<float, 3>& joint,
                       std::array<float, 2>& morph,
                       std::string& error)
{
    videohelper::ImportedAnimationDeformationEvaluation evaluation;
    if (!consumer.evaluatePreview(request, { 12, 24, 1 }, 9, evaluation, error)
        || !evaluation.deformation)
        return false;
    const auto& joints = evaluation.deformation->jointTransforms();
    const auto& morphs = evaluation.deformation->morphWeights();
    if (joints.size() != 1 || morphs.size() != 1 || morphs[0].values().size() != 2)
        return false;
    joint = joints[0].translation();
    morph = { morphs[0].values()[0], morphs[0].values()[1] };
    return true;
}

class FakeFrame final : public arbitgpu::NativeFixtureSceneFrame
{
public:
    std::string api = "opengl";
    std::uint32_t frameWidth = 640;
    std::uint32_t frameHeight = 360;
    std::uintptr_t image = 33;
    std::uintptr_t view = 44;
    std::uintptr_t depthImage = 55;
    std::uintptr_t depthView = 66;
    arbitgpu::NativeTexturePixelFormat depthFormat = arbitgpu::NativeTexturePixelFormat::R32Float;
    std::uint32_t descriptorWidth = 640;
    std::uint32_t descriptorSampleCount = 1;
    std::uintptr_t contextIdentity = 77;
    bool rawPasses = false;
    bool linearColor = false;
    int content = 0;

    const std::string& backend() const noexcept override { return api; }
    std::uint32_t width() const noexcept override { return frameWidth; }
    std::uint32_t height() const noexcept override { return frameHeight; }
    std::uintptr_t colorImageHandle() const noexcept override { return image; }
    std::uintptr_t colorTextureViewHandle() const noexcept override { return view; }
    std::uintptr_t depthImageHandle() const noexcept override { return depthImage; }
    std::uintptr_t depthTextureViewHandle() const noexcept override { return depthView; }
    arbitgpu::NativeTextureViewDescriptor colorTextureDescriptor() const noexcept override
    {
        return { api, arbitgpu::NativeTextureViewKind::Texture2D,
                 linearColor ? arbitgpu::NativeTexturePixelFormat::Rgba16Float
                 : api == "metal" ? arbitgpu::NativeTexturePixelFormat::Bgra8Unorm
                                  : arbitgpu::NativeTexturePixelFormat::Rgba8Unorm,
                 image, view, descriptorWidth, frameHeight, descriptorSampleCount, true, contextIdentity, 1,
                 linearColor ? colortransform::ColorSpace::LinearSRGB : colortransform::ColorSpace::Unspecified,
                 linearColor ? colortransform::TransferFunction::Linear : colortransform::TransferFunction::Unspecified };
    }
    arbitgpu::NativeTextureViewDescriptor decodedLinearMaterialDescriptor() const noexcept override
    { return linearColor ? colorTextureDescriptor() : arbitgpu::NativeTextureViewDescriptor{}; }
    arbitgpu::NativeTextureViewDescriptor depthTextureDescriptor() const noexcept override
    {
        return { api, arbitgpu::NativeTextureViewKind::Texture2D,
                 depthFormat, depthImage, depthView, descriptorWidth, frameHeight,
                 descriptorSampleCount, true, 77, 1 };
    }
    arbitgpu::NativeTextureViewDescriptor passTextureDescriptor(renderpassoutput::Output output) const noexcept override
    {
        if (!rawPasses || output == renderpassoutput::Output::Color || output == renderpassoutput::Output::Depth)
            return NativeFixtureSceneFrame::passTextureDescriptor(output);
        const auto id = static_cast<unsigned>(output);
        const auto format = output == renderpassoutput::Output::Mask ? arbitgpu::NativeTexturePixelFormat::R8Unorm
            : output == renderpassoutput::Output::Motion ? arbitgpu::NativeTexturePixelFormat::Rg16Float
            : output == renderpassoutput::Output::MaterialId || output == renderpassoutput::Output::ObjectId
                ? arbitgpu::NativeTexturePixelFormat::R32Uint : arbitgpu::NativeTexturePixelFormat::Rgba16Float;
        return { api, arbitgpu::NativeTextureViewKind::Texture2D, format, 100 + id, 200 + id,
            descriptorWidth, frameHeight, descriptorSampleCount, true, 77, 1 };
    }
};

struct FakeReceipt
{
    bool complete = true;
    std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> frame;
    int owner = 0;
    std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> canonicalBlockCFrame;
    std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> materialFrameTexture;
};

struct FakeExecution
{
    using Receipt = FakeReceipt;

    int previewCalls = 0;
    int exportCalls = 0;
    int staticGpuUploads = 0;
    const HarmonicMIDI::grid::Visual3DScene* admittedScene = nullptr;
    videohelper::modelpayload::ImportedSceneRequest lastRequest;
    std::shared_ptr<FakeFrame> nextFrame = std::make_shared<FakeFrame>();
    bool succeed = true;
    std::string publishedIdentity;
    int identityRetirements = 0;

    static bool validReceipt(const Receipt& receipt) noexcept
    {
        return receipt.complete && receipt.frame != nullptr;
    }

    static const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& nativeFrame(
        const Receipt& receipt) noexcept
    {
        return receipt.frame;
    }

    void publishStaticPayload(int, std::uint64_t, std::uint64_t,
                              const std::string& identity) noexcept
    {
        if (!publishedIdentity.empty() && publishedIdentity != identity)
            ++identityRetirements;
        publishedIdentity = identity;
    }

    bool executePreview(const videohelper::modelpayload::ImportedSceneRequest& request,
                        Receipt& receipt, std::string& error)
    {
        ++previewCalls;
        lastRequest = request;
        if (request.sceneSnapshot != nullptr
            && request.sceneSnapshot.get() != admittedScene)
        {
            admittedScene = request.sceneSnapshot.get();
            ++staticGpuUploads;
        }
        if (!succeed)
        {
            error = "preview rejected";
            return false;
        }
        receipt = { true, nextFrame, 1, request.runtimeInputs.canonicalBlockCFrame,
                    request.runtimeInputs.materialFrameTexture };
        return true;
    }

    bool executeExport(const videohelper::modelpayload::ImportedSceneRequest& request,
                       Receipt& receipt, std::string& error)
    {
        ++exportCalls;
        lastRequest = request;
        if (request.sceneSnapshot != nullptr
            && request.sceneSnapshot.get() != admittedScene)
        {
            admittedScene = request.sceneSnapshot.get();
            ++staticGpuUploads;
        }
        if (!succeed)
        {
            error = "export rejected";
            return false;
        }
        receipt = { true, nextFrame, 2, request.runtimeInputs.canonicalBlockCFrame,
                    request.runtimeInputs.materialFrameTexture };
        return true;
    }
};

struct FakeLayer
{
    std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> rawExportFrame;
    std::uint32_t rawExportMask = 0, rawExportRenderId = 0;
    int rawExportClipId = -1;
    renderpassoutput::Output rawExportImageOutput = renderpassoutput::Output::Color;
    struct Clock { double time = 0.0; int frame = 0; } shaderClock;
    bool audioPresent = false;
    struct AudioFeatures { std::vector<float> bands; } audioFeatures;
    std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> canonicalBlockCFrame;
    int shaderTransitionFromClipId = 0;
    unsigned texture = 0;
    std::string nativeTextureBackend;
    std::uintptr_t nativeTextureView = 0;
    arbitgpu::NativeTextureViewDescriptor nativeTextureDescriptor;
    unsigned depthTexture = 0;
    std::string nativeDepthTextureBackend;
    std::uintptr_t nativeDepthTextureView = 0;
    arbitgpu::NativeTextureViewDescriptor nativeDepthTextureDescriptor;
    int depthWidth = 0;
    int depthHeight = 0;
    int texWidth = 0;
    int texHeight = 0;
};


// Capture the production CPU-to-native scene boundary without claiming GPU execution.
class WaterMeshCaptureBackend final : public arbitgpu::NativeFixtureSceneBackend
{
public:
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> captured;
    int prepareCalls = 0;
    arbitgpu::BackendInfo info() const override { return {}; }
    arbitgpu::NativeFixtureScenePreparation prepare(
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
        const std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram>&,
        arbitgpu::NativeFixturePrepareUse = arbitgpu::NativeFixturePrepareUse::Blocking) override
    {
        ++prepareCalls; captured = scene;
        arbitgpu::NativeFixtureScenePreparation result;
        result.error = "water fixture stops before native GPU submission";
        return result;
    }
    arbitgpu::NativeFixtureSceneSubmission render(
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>&,
        const std::shared_ptr<const arbitgpu::NativeFixtureSceneResources>&,
        std::uint32_t,std::uint32_t,arbitgpu::NativeFixtureSceneRuntimeInputs) override { return {}; }
};

void waterFrameConsumers()
{
    using namespace videohelper::geometry;
    using namespace videowire::geometry;
    class Capabilities final : public BackendCapabilitySource {
    public:
        BackendCapabilities geometryCoreCapabilities() const override {
            BackendCapabilities c; c.backendIdentity = "owned-water-fixture"; c.deviceIdentity = "fixture-device";
            c.nativeGpuAvailable = c.geometryCoreExecution = c.immutableSourceBuffers = c.stableElementIds = c.typedFieldEvaluation = true;
            c.supportedCarriers = carrierBit(CarrierKind::geometry3D);
            return c;
        }
    } capabilities;
    GeometryCorePlanRuntime runtime(capabilities);
    const PlanOwnerIdentity preview {31,41,51,8,9,PlanUse::preview};
    auto exportOwner = preview; exportOwner.use = PlanUse::exportRender;
    std::string error;
    WaterSurfaceFixtureSettings settings;
    auto request = composedTimelineGeometryFixture(error,nullptr,&settings);
    check(request.has_value(), "owned water scene fixture lowers: " + error);
    if (!request) return;
    const auto authored = sceneaov::detail::encodeScene(*request->sceneSnapshot);
    auto history = waterreplayfixture::source();
    auto score = std::make_shared<arbitmod::Score>(*history.score);
    score->notes = {waterreplayfixture::note(-7,0,0,63.5f)}; history.score = score;
    int parameterCalls = 0;
    history.parameterAt = [&](const auto&,double,double&,std::string&) { ++parameterCalls; return false; };
    constexpr double project = 3.0/120, scene = 23.5;
    auto currentFrame = waterreplayfixture::frame(history,project);
    const auto evaluate = [&](const PlanOwnerIdentity& owner,const videorender::ParticleHistorySource* source,
                              double projectTime,double sceneTime) {
        return evaluateComposedTimelineGeometry(runtime,owner,*request->timelineGeometry,
            request->sceneSnapshot,sceneTime,error,source,projectTime,currentFrame);
    };
    const auto changed = evaluate(preview,&history,project,scene);
    const auto exported = evaluate(exportOwner,&history,project,scene);
    const auto repeated = evaluate(preview,&history,project,scene);
    currentFrame = waterreplayfixture::frame(history,0);
    const auto reset = evaluate(preview,&history,0,scene);
    currentFrame = waterreplayfixture::frame(history,project);
    check(changed && exported && repeated && reset, "owned composed water preview/export frames replay: " + error);
    // Each refusal names its own cause and happens before any admission.
    std::string refusal;
    {
        const bool conditionOk = !evaluateComposedTimelineGeometry(runtime,preview,*request->timelineGeometry,
              nullptr,scene,refusal,&history,project,currentFrame)
            && refusal == "Timeline Scene3D requires an authored scene";
        check(conditionOk, "missing authored scene has its own refusal: " + refusal);
    }
    {
        const bool conditionOk = !evaluateComposedTimelineGeometry(runtime,PlanOwnerIdentity{0,41,51,8,9,PlanUse::preview},
              *request->timelineGeometry,request->sceneSnapshot,scene,refusal,&history,project,currentFrame)
            && refusal == "Timeline Scene3D owner is invalid because it has zero projectGeneration";
        check(conditionOk, "zero project generation is named in the owner refusal: " + refusal);
    }
    {
        const bool conditionOk = !evaluateComposedTimelineGeometry(runtime,PlanOwnerIdentity{0,41,0,8,9,PlanUse::preview},
              *request->timelineGeometry,request->sceneSnapshot,scene,refusal,&history,project,currentFrame)
            && refusal == "Timeline Scene3D owner is invalid because it has zero projectGeneration, deviceGeneration";
        check(conditionOk, "every zero owner field is named in declaration order: " + refusal);
    }
    {
        const bool conditionOk = !evaluateComposedTimelineGeometry(runtime,PlanOwnerIdentity{0,0,0,0,0,PlanUse::preview},
              *request->timelineGeometry,request->sceneSnapshot,scene,refusal,&history,project,currentFrame)
            && refusal == "Timeline Scene3D owner is invalid because it has zero projectGeneration, helperGeneration, deviceGeneration, clipIdentity, planGeneration";
        check(conditionOk, "all zero owner fields are named in declaration order: " + refusal);
    }
    {
        const bool conditionOk = !evaluateComposedTimelineGeometry(runtime,preview,*request->timelineGeometry,
              request->sceneSnapshot,std::numeric_limits<double>::quiet_NaN(),refusal,&history,project,currentFrame)
            && refusal == "Timeline Scene3D requires a finite frame time";
        check(conditionOk, "non-finite frame time has its own refusal: " + refusal);
    }
    if (!changed || !exported || !repeated || !reset) return;
    const auto encoded = sceneaov::detail::encodeScene(*changed);
    check(encoded == sceneaov::detail::encodeScene(*exported)
        && encoded == sceneaov::detail::encodeScene(*repeated)
        && authored == sceneaov::detail::encodeScene(*reset) && encoded != authored,
        "composed water keeps byte-identical preview/export replay and project-zero rest at an offset scene clock");
    double total = 0;
    for (std::size_t i=0;i<4;++i) total += changed->vertices[i].position.z;
    check(std::abs(total - (1.0/120)*(0.5+0.25+0.125)) < 1e-8,
        "composed water uses independently predicted damped score impulse at project time");
    for (std::size_t i=4;i<changed->vertexCount;++i)
        check(sameSceneVertex(changed->vertices[i],request->sceneSnapshot->vertices[i]),
            "composed water preserves unrelated cube vertices");
    check(changed->vertexCount==request->sceneSnapshot->vertexCount
        && changed->indexCount==request->sceneSnapshot->indexCount
        && changed->id.value==request->sceneSnapshot->id.value
        && std::equal(changed->indices.begin(),changed->indices.begin()+changed->indexCount,
            request->sceneSnapshot->indices.begin()),
        "composed water keeps scene id, vertex order and indices, so Motion pairs retained positions by index");
    history.key.loopGeneration++; history.key.seekGeneration++;
    currentFrame = waterreplayfixture::frame(history,project);
    const auto generationChanged = evaluate(preview,&history,project,scene);
    check(generationChanged && sceneaov::detail::encodeScene(*generationChanged) == encoded,
        "seek and loop generations do not invent a nonzero water reset origin");
    const auto reject = [&](const videorender::ParticleHistorySource* candidate,double seconds,const char* message) {
        check(!evaluate(preview,candidate,seconds,scene) && !error.empty()
            && sceneaov::detail::encodeScene(*request->sceneSnapshot) == authored
            && sceneaov::detail::encodeScene(*changed) == encoded,message);
    };
    reject(nullptr,project,"composed water missing history leaves authored and published scenes untouched");
    auto invalid = history; invalid.key.helperGeneration = 0;
    reject(&invalid,project,"composed water invalid history generation refuses atomically");
    currentFrame = waterreplayfixture::frame(history,10.001);
    reject(&history,10.001,"composed water over-horizon request refuses atomically");
    currentFrame = waterreplayfixture::frame(history,project);

    const auto bytes = decodeLoweredPlanText(request->timelineGeometry->loweredPlan,error);
    const auto admission = bytes ? runtime.admitPreview(preview,*bytes,{}, {},error) : std::nullopt;
    const auto exportAdmission = bytes ? runtime.admitExport(exportOwner,*bytes,{}, {},error) : std::nullopt;
    check(admission && exportAdmission,"image water geometry keeps the existing admission owner: " + error);
    if (!admission || !exportAdmission) return;
    WaterMeshCaptureBackend backend;
    const auto capture = [&](const RuntimeAdmission& owned,const videorender::ParticleHistorySource* candidate,double seconds) {
        backend.captured.reset();
        arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
        inputs.canonicalBlockCFrame = currentFrame;
        const auto execution = executeNativeGeometry(backend,owned,64,64,error,false,std::move(inputs),nullptr,{},
            scene,{},candidate,seconds);
        check(!execution,"CPU water fixture stops before any native GPU submission");
        return backend.captured;
    };
    const auto nativePreview = capture(*admission,&history,project);
    const auto nativeExport = capture(*exportAdmission,&history,project);
    check(nativePreview && nativeExport,"Geometry Core image consumer supplies water replay before native preparation: " + error);
    if (nativePreview && nativeExport) {
        check(sceneaov::detail::encodeScene(*nativePreview) == sceneaov::detail::encodeScene(*nativeExport),
            "image water consumer produces byte-identical preview/export scenes");
        for (std::size_t i=0;i<4;++i) {
            const auto& a = nativePreview->vertices[i].position; const auto& b = changed->vertices[i].position;
            check(a.x==b.x && a.y==b.y && a.z==b.z,"image and composed consumers share exact water positions at project time");
        }
    }
    constexpr double nonGridProject = 0.079;
    const auto directRest = waterreplayfixture::mesh(2,2);
    WaterSurfaceGrid directGrid;
    check(validateWaterSurfaceGrid(directRest,directGrid,error),"non-grid consumer oracle retains the authored grid");
    const auto directOperation = waterreplayfixture::operation(settings.speed,settings.damping,settings.scoreGain,settings.audioGain);
    for (const double fps : {24.0,30000.0/1001.0}) {
        currentFrame = waterreplayfixture::frame(history,nonGridProject,fps);
        const auto composedPreview = evaluate(preview,&history,nonGridProject,scene);
        const auto imagePreview = capture(*admission,&history,nonGridProject);
        currentFrame = waterreplayfixture::frame(history,nonGridProject,fps,true);
        const auto composedExport = evaluate(exportOwner,&history,nonGridProject,scene);
        const auto imageExport = capture(*exportAdmission,&history,nonGridProject);
        std::vector<double> direct,snapped;
        const bool replayed = replayWaterSurface(history,directOperation,directRest,directGrid,nonGridProject,direct,error);
        check(composedPreview && composedExport && imagePreview && imageExport && replayed && direct.size()==4,
            "both water consumers admit non-grid 24 and rational 29.97 fps FrameOwner requests");
        if (composedPreview && composedExport && imagePreview && imageExport && replayed && direct.size()==4) {
            check(sceneaov::detail::encodeScene(*composedPreview)==sceneaov::detail::encodeScene(*composedExport)
                && sceneaov::detail::encodeScene(*imagePreview)==sceneaov::detail::encodeScene(*imageExport),
                "non-grid production consumers retain byte-identical preview/export scene replay");
            for (std::size_t i=0;i<4;++i) {
                const float directZ = static_cast<float>(directRest.positions[i].z+direct[i]);
                check(std::memcmp(&composedPreview->vertices[i].position.z,&directZ,sizeof(float))==0
                    && std::memcmp(&composedExport->vertices[i].position.z,&directZ,sizeof(float))==0
                    && std::memcmp(&imagePreview->vertices[i].position.z,&directZ,sizeof(float))==0
                    && std::memcmp(&imageExport->vertices[i].position.z,&directZ,sizeof(float))==0,
                    "both production consumers match independent exact-time replay position bytes");
            }
            const double snappedProject = static_cast<double>(currentFrame->key.frame)/fps;
            check(replayWaterSurface(history,directOperation,directRest,directGrid,snappedProject,snapped,error)
                && snapped!=direct,"production consumers preserve the exact water endpoint after canonical frame quantization");
        }
        const auto nonGridPrepares = backend.prepareCalls;
        auto foreign = history; ++foreign.key.loopGeneration;
        reject(&foreign,nonGridProject,"non-grid foreign lifecycle refuses composed water atomically");
        check(!capture(*admission,&foreign,nonGridProject) && backend.prepareCalls==nonGridPrepares,
            "non-grid foreign lifecycle refuses image water before backend preparation");
        currentFrame = waterreplayfixture::frame(history,nonGridProject+1/fps,fps);
        reject(&history,nonGridProject,"non-grid wrong canonical frame refuses composed water atomically");
        check(!capture(*admission,&history,nonGridProject) && backend.prepareCalls==nonGridPrepares,
            "non-grid wrong canonical frame refuses image water before backend preparation");
        auto wrongTimeline = history; wrongTimeline.timeline.reset(90,4);
        currentFrame = waterreplayfixture::frame(wrongTimeline,nonGridProject,fps);
        reject(&history,nonGridProject,"non-grid wrong packed beat refuses composed water atomically");
        check(!capture(*admission,&history,nonGridProject) && backend.prepareCalls==nonGridPrepares,
            "non-grid wrong packed beat refuses image water before backend preparation");
    }
    currentFrame = waterreplayfixture::frame(history,project);
    const auto prepares = backend.prepareCalls;
    check(!capture(*admission,nullptr,project) && backend.prepareCalls==prepares,
        "missing water history refuses before image backend preparation");
    check(!capture(*admission,&invalid,project) && backend.prepareCalls==prepares,
        "invalid water generation refuses before image backend preparation");
    currentFrame = waterreplayfixture::frame(history,10.001);
    check(!capture(*admission,&history,10.001) && backend.prepareCalls==prepares,
        "over-horizon water refuses before image backend preparation");
    currentFrame = waterreplayfixture::frame(history,project);
    // Every candidate is valid on its own, but belongs to another Block C lifecycle.
    for (const auto member : {&canonicalblockc::FrameKey::projectGeneration,
            &canonicalblockc::FrameKey::helperGeneration,&canonicalblockc::FrameKey::deviceGeneration,
            &canonicalblockc::FrameKey::sourceGeneration,&canonicalblockc::FrameKey::backendGeneration,
            &canonicalblockc::FrameKey::scoreGeneration,&canonicalblockc::FrameKey::beatMapGeneration,
            &canonicalblockc::FrameKey::fpsGeneration,&canonicalblockc::FrameKey::loopGeneration,
            &canonicalblockc::FrameKey::seekGeneration}) {
        auto foreign = history; ++(foreign.key.*member);
        reject(&foreign,project,"valid nonzero foreign Block C history refuses composed water atomically");
        check(!capture(*admission,&foreign,project) && backend.prepareCalls==prepares,
            "valid nonzero foreign Block C history refuses before image backend preparation");
        const auto savedFrame = currentFrame;
        currentFrame = waterreplayfixture::frame(foreign,project);
        reject(&history,project,"valid foreign current canonical frame refuses composed water atomically");
        check(!capture(*admission,&history,project) && backend.prepareCalls==prepares,
            "valid foreign current canonical frame refuses before image backend preparation");
        currentFrame = savedFrame;
    }
    const auto savedFrame = currentFrame;
    currentFrame.reset();
    reject(&history,project,"missing current canonical frame refuses composed water atomically");
    check(!capture(*admission,&history,project) && backend.prepareCalls==prepares,
        "missing current canonical frame refuses image water before native preparation");
    currentFrame = waterreplayfixture::frame(history,project+1.0/120);
    reject(&history,project,"wrong canonical project time refuses composed water atomically");
    check(!capture(*admission,&history,project) && backend.prepareCalls==prepares,
        "wrong canonical project time refuses image water before native preparation");
    currentFrame = savedFrame;
    check(changed && nativePreview && currentFrame->key.projectGeneration!=preview.projectGeneration
        && currentFrame->key.helperGeneration!=preview.helperGeneration
        && currentFrame->key.deviceGeneration!=preview.deviceGeneration,
        "matching Block C history/frame remains valid with independent Geometry generation counters");
    check(parameterCalls==0,"both production water consumers retain static controls without parameterAt sampling");

    settings.audioGain = 1;
    request = composedTimelineGeometryFixture(error,nullptr,&settings);
    check(request.has_value(),"positive-audio water scene fixture lowers");
    if (!request) return;
    const auto audioAuthored = sceneaov::detail::encodeScene(*request->sceneSnapshot);
    GeometryCorePlanRuntime audioRuntime(capabilities);
    const auto audioEvaluate = [&](const PlanOwnerIdentity& owner) {
        return evaluateComposedTimelineGeometry(audioRuntime,owner,*request->timelineGeometry,
            request->sceneSnapshot,scene,error,&history,project,currentFrame);
    };
    check(!audioEvaluate(exportOwner)
        && sceneaov::detail::encodeScene(*request->sceneSnapshot)==audioAuthored,
        "midrange export without master project-zero history refuses composed water atomically");
    const auto audioBytes = decodeLoweredPlanText(request->timelineGeometry->loweredPlan,error);
    const auto audioAdmission = audioBytes ? audioRuntime.admitExport(exportOwner,*audioBytes,{}, {},error) : std::nullopt;
    check(audioAdmission && !capture(*audioAdmission,&history,project) && backend.prepareCalls==prepares,
        "midrange export without master project-zero history refuses image water before native preparation");
    auto frames = waterreplayfixture::audio(3); frames[0].onset = 1;
    history.audioFrames = &frames; history.audioSampleRate = 61440;
    const auto audioPreview = audioEvaluate(preview), audioExport = audioEvaluate(exportOwner);
    check(audioPreview && audioExport && sceneaov::detail::encodeScene(*audioPreview)==sceneaov::detail::encodeScene(*audioExport),
        "contiguous master audio and canonical score produce byte-identical composed preview/export water");
    check(audioAdmission && capture(*audioAdmission,&history,project),
        "image water consumer uses contiguous project-zero master audio before native preparation");
    auto ordinary = composedTimelineGeometryFixture(error);
    GeometryCorePlanRuntime ordinaryRuntime(capabilities);
    check(ordinary.has_value(),"non-water timeline fixture lowers");
    if (ordinary) {
        const auto plain = evaluateComposedTimelineGeometry(ordinaryRuntime,preview,*ordinary->timelineGeometry,
            ordinary->sceneSnapshot,0.5,error,&invalid,project,{});
        check(plain!=nullptr,"non-water composed geometry ignores unused invalid water provenance");
        const auto plainBytes = decodeLoweredPlanText(ordinary->timelineGeometry->loweredPlan,error);
        const auto plainAdmission = plainBytes ? ordinaryRuntime.admitPreview(preview,*plainBytes,{}, {},error) : std::nullopt;
        currentFrame.reset();
        check(plainAdmission && capture(*plainAdmission,&invalid,project),
            "non-water image geometry ignores unused invalid water provenance");
        currentFrame = savedFrame;
    }
    frames.resize(2);
    const auto audioPrepares = backend.prepareCalls;
    check(!audioEvaluate(exportOwner) && audioAdmission && !capture(*audioAdmission,&history,project)
        && backend.prepareCalls==audioPrepares && sceneaov::detail::encodeScene(*request->sceneSnapshot)==audioAuthored,
        "partial master coverage refuses both consumers atomically without clamping project time");
}

std::uint32_t floatBits(float value)
{
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

int surfaceChannels(surfacematerial::ValueType type)
{
    switch (type)
    {
        case surfacematerial::ValueType::Scalar:
        case surfacematerial::ValueType::UInt: return 1;
        case surfacematerial::ValueType::Vec2: return 2;
        case surfacematerial::ValueType::Vec3: return 3;
        case surfacematerial::ValueType::Vec4: return 4;
        case surfacematerial::ValueType::Invalid: break;
    }
    return 0;
}

std::string surfaceDataType(surfacematerial::ValueType type)
{
    return type == surfacematerial::ValueType::UInt
        ? "integer" : std::string(surfacematerial::token(type));
}

surfacematerialbinding::ImportedSceneMaterialRequest materialRequest()
{
    surfacematerialbinding::ImportedSceneMaterialRequest material;
    material.scene.value = 17;
    material.sceneRevision = 31;
    material.structuralRevision = 32;
    material.evaluationRevision = 33;
    material.programRevision = 32;
    material.program.textureSlotCount = 1;
    for (const auto output : surfacematerial::kSurfaceOutputs)
    {
        surfacematerial::Operation operation;
        operation.id = static_cast<surfacematerial::ValueId>(
            material.program.operations.size() + 1u);
        operation.kind = output == surfacematerial::OutputSemantic::MaterialId
            ? surfacematerial::OperationKind::UIntConstant
            : surfacematerial::OperationKind::FloatConstant;
        operation.resultType = surfacematerial::outputType(output);
        if (output == surfacematerial::OutputSemantic::BaseColor)
            operation.literal = { 0.25f, -0.0f, 0.75f, 0.0f };
        else if (output == surfacematerial::OutputSemantic::Normal)
            operation.literal = { 0.0f, 0.0f, 1.0f, 0.0f };
        else if (output == surfacematerial::OutputSemantic::Opacity)
            operation.literal[0] = 1.0f;
        else if (output == surfacematerial::OutputSemantic::Ior)
            operation.literal[0] = 1.5f;
        else if (output == surfacematerial::OutputSemantic::MaterialId)
            operation.unsignedLiteral = 23;
        material.program.outputs[static_cast<std::size_t>(output)] = operation.id;
        material.program.operations.push_back(operation);
    }
    material.binding.targetKind
        = surfacematerialbinding::BindingTargetKind::ObjectOverride;
    material.binding.object.value = 23;
    material.binding.surfaceMaterialDigest = std::string(64, 'b');
    material.binding.surfaceMaterialRevision = 32;
    surfacematerialbinding::TextureSlotBinding texture;
    texture.slot = 0;
    texture.source = surfacematerialbinding::TextureSourceKind::ImportedBaseColor;
    for (std::size_t index = 0; index < texture.videoResource.size(); ++index)
        texture.videoResource[index] = static_cast<std::uint8_t>(index + 1u);
    material.binding.textures.push_back(texture);
    return material;
}

diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest
diffractionMaterialRequest()
{
    diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest material;
    material.scene.value = 1;
    material.sceneRevision = 3;
    material.structuralRevision = 9;
    material.evaluationRevision = 33;
    material.materialRevision = 9;
    material.object.value = 23;
    material.material = diffractionmaterial::makeAluminiumBinaryGratingPreset();
    material.lighting = diffractionmaterial::makeReferenceLighting();
    std::string error;
    const auto admitted = diffractionmaterial::admit(material.material, error);
    if (admitted) material.structuralDigest = admitted->structuralDigest();
    return material;
}

diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest
spatialDiffractionMaterialRequest()
{
    auto material = diffractionMaterialRequest();
    diffractivefoil::Description foil;
    foil.physicalBsdf = material.material;
    foil.workBudget.maximumEvaluations = diffractivefoil::kMaximumEvaluations;
    foil.grooveField = {{
        { { 1.0f, 0.0f }, foil.physicalBsdf.geometry.grooveSpacingNanometres, 0.0f },
        { { 0.0f, 1.0f }, diffractionmaterial::kMaximumGrooveSpacingNanometres, 1.0f },
        { { 0.6f, 0.8f }, 800.0f, 0.25f },
        { { 0.6f, -0.8f }, 2000.0f, 0.75f }
    }};
    material.spatialFoil = foil;
    std::string foilError;
    const auto admittedFoil = diffractivefoil::admit(foil, foilError);
    if (admittedFoil)
        material.structuralDigest = admittedFoil->structuralDigest();
    return material;
}

videowire::VertexModifierIr vertexModifierRequest()
{
    using Operation = videowire::VertexModifierOperation;
    videowire::VertexModifierIr vertex;
    const auto append = [&vertex](videowire::VertexModifierStableId id,
                                  Operation operation,
                                  std::initializer_list<videowire::VertexModifierStableId> inputs,
                                  std::initializer_list<double> parameters)
    {
        const auto schema = videowire::vertexModifierOperationSchema(operation);
        videowire::VertexModifierRecord record;
        record.stableId = id;
        record.operation = operation;
        record.resultType = schema.resultType;
        record.inputCount = schema.inputCount;
        record.parameterCount = schema.parameterCount;
        std::copy(inputs.begin(), inputs.end(), record.inputs.begin());
        std::copy(parameters.begin(), parameters.end(), record.parameters.begin());
        vertex.records.push_back(record);
    };
    append(101, Operation::importedPosition, {}, {});
    append(102, Operation::vec3Constant, {}, { 0.0, 0.25, -0.0 });
    append(103, Operation::boundedDisplacementOutput, { 101, 102 }, { 0.5 });
    vertex.rootId = 103;
    return vertex;
}

std::string replaceLine(std::string value, std::size_t lineIndex,
                        const std::string& replacement)
{
    std::size_t start = 0;
    for (std::size_t index = 0; index < lineIndex; ++index)
    {
        start = value.find('\n', start);
        if (start == std::string::npos) return {};
        ++start;
    }
    const auto end = value.find('\n', start);
    if (end == std::string::npos) return {};
    value.replace(start, end - start, replacement);
    return value;
}

enum class CrossVersionAdmissionStage
{
    baseSyntax,
    flagSchema,
    cameraSyntax,
    lightSyntax,
    deformationSyntax,
    materialSyntax,
    diffractionSyntax,
    trailingBytes,
    destinationAuthority
};

struct DirectionalRelabelFixture
{
    std::string_view identity;
    std::string_view sourceHeader;
    std::string_view destinationHeader;
    std::string sourceBytes;
    std::string relabeledBytes;
    CrossVersionAdmissionStage expectedStage;
};

std::string relabelHeaderOnly(const std::string& sourceBytes,
                              std::string_view destinationHeader)
{
    return replaceLine(sourceBytes, 0, std::string(destinationHeader));
}

CrossVersionAdmissionStage probeCrossVersionAdmission(
    const std::string& encoded, std::string_view destinationHeader)
{
    std::istringstream input(encoded);
    std::string header;
    visualimportedscenerender::Request request;
    if (!std::getline(input, header)
        || header != destinationHeader
        || !visualimportedscenerender::detail::decodeBase(input, request))
        return CrossVersionAdmissionStage::baseSyntax;

    const bool v8 = destinationHeader
        == visualimportedscenerender::kCompiledOperationHeaderV8;
    const bool v7 = destinationHeader
        == visualimportedscenerender::kCompiledOperationHeaderV7;
    if (v8)
    {
        int deformation = -1, material = -1, vertex = -1;
        int diffraction = -1, camera = -1, light = -1;
        if (!(input >> deformation >> material >> vertex >> diffraction >> camera >> light)
            || (deformation != 0 && deformation != 1)
            || (material != 0 && material != 1)
            || (vertex != 0 && vertex != 1)
            || (diffraction != 0 && diffraction != 1)
            || (camera != 0 && camera != 1) || light != 1
            || (vertex != 0 && material == 0)
            || (material != 0 && diffraction != 0))
            return CrossVersionAdmissionStage::flagSchema;
        if (camera != 0)
        {
            HarmonicMIDI::grid::SceneCameraRecord value;
            if (!visualimportedscenerender::detail::decodeCamera(input, value))
                return CrossVersionAdmissionStage::cameraSyntax;
            request.camera = value;
        }
        HarmonicMIDI::grid::SceneLightRecord value;
        if (!visualimportedscenerender::detail::decodeLight(input, value))
            return CrossVersionAdmissionStage::lightSyntax;
        request.light = value;
        if (deformation != 0)
            return CrossVersionAdmissionStage::deformationSyntax;
        if (material != 0)
        {
            surfacematerialbinding::ImportedSceneMaterialRequest value;
            if (!visualimportedscenerender::detail::decodeMaterial(input, value, vertex != 0))
                return CrossVersionAdmissionStage::materialSyntax;
            request.material = value;
        }
        if (diffraction != 0)
        {
            diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest value;
            if (!visualimportedscenerender::detail::decodeDiffractionMaterial(input, value))
                return CrossVersionAdmissionStage::diffractionSyntax;
            request.diffractionMaterial = value;
        }
    }
    else
    {
        int deformation = -1, diffraction = -1, camera = -1;
        if (!(input >> deformation >> diffraction >> camera)
            || (deformation != 0 && deformation != 1) || diffraction != 1
            || (camera != 0 && camera != 1))
            return CrossVersionAdmissionStage::flagSchema;
        if (camera != 0)
        {
            HarmonicMIDI::grid::SceneCameraRecord value;
            if (!visualimportedscenerender::detail::decodeCamera(input, value))
                return CrossVersionAdmissionStage::cameraSyntax;
            request.camera = value;
        }
        if (deformation != 0)
            return CrossVersionAdmissionStage::deformationSyntax;
        diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest value;
        const bool decoded = v7
            ? visualimportedscenerender::detail::decodeDiffractionMaterial(input, value)
            : visualimportedscenerender::detail::decodeLegacyV6DiffractionMaterial(input, value);
        if (!decoded) return CrossVersionAdmissionStage::diffractionSyntax;
        request.diffractionMaterial = value;
    }
    input >> std::ws;
    if (!input.eof()) return CrossVersionAdmissionStage::trailingBytes;
    return CrossVersionAdmissionStage::destinationAuthority;
}

visualanimationimport::Request deformationRequest()
{
    visualanimationimport::Request request;
    request.sourceStableId = 71;
    request.deformationStableId = 73;
    request.asset.id = "model-asset-1";
    request.asset.version = 3;
    request.asset.contentSha256 = std::string(64, 'a');
    request.asset.sourceMediaType = "model/gltf-binary";
    request.asset.sourceByteSize = 4096;
    request.sceneIndex = 0;
    request.clipName = "Idle";
    request.playback.timelineSeconds = 1.0;
    request.playback.speed = 1.5;
    request.playback.trimStartSeconds = 0.25;
    request.playback.trimEndSeconds = 0.75;
    request.playback.weight = 0.75;
    request.schedule = { request.sourceStableId, request.deformationStableId };
    return request;
}

visualimportedscenerender::Request renderRequest(
    bool withMaterial = false, bool withDeformation = false,
    bool withDiffractionMaterial = false)
{
    visualimportedscenerender::Request request;
    request.sourceStableId = 71;
    request.renderStableId = 72;
    request.asset.id = "model-asset-1";
    request.asset.version = 3;
    request.asset.contentSha256 = std::string(64, 'a');
    request.asset.sourceMediaType = "model/gltf-binary";
    request.asset.sourceByteSize = 4096;
    request.sceneIndex = 0;
    if (withMaterial) request.material = materialRequest();
    if (withDiffractionMaterial)
        request.diffractionMaterial = diffractionMaterialRequest();
    if (withDeformation) request.deformation = deformationRequest();
    return request;
}

std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> composedScene()
{
    using namespace HarmonicMIDI::grid;
    Visual3DScene scene;
    scene.id.value = 7;
    scene.activeCamera.value = 50;
    scene.ambientColor = { 0.1f, 0.2f, 0.3f };
    scene.vertexCount = 6;
    scene.indexCount = 6;
    scene.objectCount = 2;
    scene.materialCount = 2;
    scene.lightCount = 1;
    scene.cameraCount = 1;
    for (std::size_t object = 0; object < 2; ++object)
    {
        const auto first = static_cast<std::uint32_t>(object * 3);
        scene.vertices[first] = { { -1.0f, -1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f }, {} };
        scene.vertices[first + 1] = { { 1.0f, -1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f }, {} };
        scene.vertices[first + 2] = { { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f }, {} };
        scene.indices[first] = 0;
        scene.indices[first + 1] = 1;
        scene.indices[first + 2] = 2;
        scene.materials[object].id.value = static_cast<std::uint32_t>(20 + object);
        scene.objects[object].id.value = static_cast<std::uint32_t>(10 + object);
        scene.objects[object].material = scene.materials[object].id;
        scene.objects[object].firstVertex = first;
        scene.objects[object].vertexCount = 3;
        scene.objects[object].firstIndex = first;
        scene.objects[object].indexCount = 3;
    }
    scene.objects[1].transform.translation.x = 2.0f;
    scene.lights[0].id.value = 40;
    scene.lights[0].kind = SceneLightKind::Directional;
    scene.cameras[0].id = scene.activeCamera;
    scene.cameras[0].transform.translation.z = 4.0f;
    return std::make_shared<const Visual3DScene>(std::move(scene));
}

videowire::CompiledVisualLayerPlan composedScenePlan()
{
    videowire::CompiledVisualLayerPlan plan;
    plan.clipId = 8;
    plan.structuralRevision = 9;
    plan.producerValidated = true;
    plan.nodeKinds = {
        "visual.3d.transform", "visual.3d.material.pbr", "visual.3d.scene.primitive",
        "visual.3d.transform", "visual.3d.material.pbr", "visual.3d.scene.primitive",
        "visual.3d.scene.compose", "visual.3d.transform", "visual.3d.camera.perspective",
        "visual.3d.transform", "visual.3d.light.directional", "visual.3d.render"
    };
    for (int id = 0; id < static_cast<int>(plan.nodeKinds.size()); ++id)
    {
        plan.nodeIds.push_back(id);
        plan.operations.push_back({ id, plan.nodeKinds[static_cast<std::size_t>(id)],
            id == 11 ? "native-gpu" : "control-eval", "" });
    }
    const auto port = [&](int node, int id, const char* direction, const char* carrier,
                          const char* type, const char* format = "unspecified",
                          const char* color = "unspecified")
    {
        plan.ports.push_back({ node, id, 1, direction, carrier, type, format, color });
    };
    port(0, 0, "out", "control", "transform3D");
    port(1, 0, "out", "control", "material");
    port(2, 0, "in", "control", "transform3D");
    port(2, 1, "in", "control", "material");
    port(2, 2, "in", "control", "light");
    port(2, 3, "in", "control", "camera");
    port(2, 4, "out", "control", "scene3D");
    port(3, 0, "out", "control", "transform3D");
    port(4, 0, "out", "control", "material");
    port(5, 0, "in", "control", "transform3D");
    port(5, 1, "in", "control", "material");
    port(5, 2, "in", "control", "light");
    port(5, 3, "in", "control", "camera");
    port(5, 4, "out", "control", "scene3D");
    port(6, 0, "in", "control", "scene3D");
    port(6, 1, "in", "control", "scene3D");
    port(6, 2, "in", "control", "scene3D");
    port(6, 3, "in", "control", "scene3D");
    port(6, 4, "out", "control", "scene3D");
    port(7, 0, "out", "control", "transform3D");
    port(8, 0, "in", "control", "transform3D");
    port(8, 1, "out", "control", "camera");
    port(9, 0, "out", "control", "transform3D");
    port(10, 0, "in", "control", "transform3D");
    port(10, 1, "out", "control", "light");
    port(11, 0, "in", "control", "scene3D");
    port(11, 1, "out", "frame", "image", "rgba8", "sRGB");
    port(11, 2, "in", "control", "material");
    port(11, 3, "in", "control", "mesh");
    port(11, 4, "out", "frame", "depth", "r32f");
    port(11, 5, "in", "control", "camera");
    port(11, 6, "in", "control", "light");
    plan.edges = {
        { 0, 0, 2, 0 }, { 1, 0, 2, 1 }, { 3, 0, 5, 0 }, { 4, 0, 5, 1 },
        { 2, 4, 6, 0 }, { 5, 4, 6, 1 }, { 6, 4, 11, 0 },
        { 7, 0, 8, 0 }, { 8, 1, 2, 3 }, { 8, 1, 5, 3 }, { 8, 1, 11, 5 },
        { 9, 0, 10, 0 }, { 10, 1, 2, 2 }, { 10, 1, 5, 2 }, { 10, 1, 11, 6 }
    };
    visualimportedscenerender::Request request;
    request.sourceStableId = 7;
    request.renderStableId = 12;
    request.sceneSnapshot = composedScene();
    request.structuralRevision = 9;
    request.evaluationRevision = 12;
    plan.operations.back().payloadXml = visualimportedscenerender::encode(request);
    return plan;
}


void testImportedParentIntentTransport()
{
    using namespace visualimportedscenerender;
    using namespace HarmonicMIDI::grid;
    using namespace videohelper::importedscene;
    Request base;
    base.sourceStableId = 7; base.renderStableId = 12;
    base.structuralRevision = 31; base.evaluationRevision = 35;
    base.sceneSnapshot = composedScene();
    auto leaf = renderRequest(); leaf.renderStableId = 12;
    leaf.structuralRevision = 31; leaf.evaluationRevision = 35;
    base.importedScene = std::make_shared<const Request>(leaf);
    const auto ordinary = encode(base);
    check(!ordinary.empty(), "parent transport fixture retains a valid legacy composition");
    if (ordinary.empty()) return;
    ImportedParentIntentV1 intent;
    intent.direction = 2; intent.composeStableId = 7; intent.importedSourceStableId = 71;
    intent.connectedInputMask = 7; intent.requestedParentMask = 2;
    intent.rootMembershipCount = 2; intent.rootMembership[0] = {10, 1}; intent.rootMembership[1] = {11, 2};
    auto request = base; request.importedParentIntent = intent;
    // Independent packed little-endian golden: 48 header bytes, two roots,
    // then 62 zero slots. No production intent encoder supplies these bytes.
    const std::string headerGolden = "0100000002000000"
        "07000000000000004700000000000000"
        "070000000200000000000000000000000000000002000000";
    const std::string rootsGolden = "0a000000010000000b00000002000000";
    const std::string intentGolden = headerGolden + rootsGolden + std::string(62 * 16, '0');
    const auto expected = std::string("visual.imported-scene.render.v23\n") + intentGolden + "\n" + ordinary;
    Request decoded;
    std::string error;
    check(intentGolden.size() == 1120 && encode(request) == expected
        && expected.size() == ordinary.size() + 1154
        && decodeCanonical(expected, decoded, error) && sameRequest(request, decoded)
        && encode(decoded) == expected && decoded.structuralRevision == 31 && decoded.evaluationRevision == 35,
        "V23 adds exactly the independently packed 560-byte complete-root intent and 1154 text bytes");
    check(decoded.sceneSnapshot && decoded.sceneSnapshot->objects[0].parent.value == 0
        && decoded.sceneSnapshot->objects[1].parent.value == 0,
        "V23 retains unchanged unflagged and flagged procedural parent words until genuine helper assembly");
    const auto sentinel = decoded;
    auto selectedLeaf = leaf; selectedLeaf.sourceCameraNodeId = 7;
    auto cameraBase = base; cameraBase.importedScene = std::make_shared<const Request>(selectedLeaf);
    cameraBase.camera = cameraBase.sceneSnapshot->cameras[0];
    const auto cameraNested = encode(cameraBase);
    auto parentCamera = cameraBase; parentCamera.importedParentIntent = intent;
    check(!cameraNested.empty() && cameraNested.find("visual.imported-scene.render.v22\noverride ") == 0
        && encode(parentCamera) == std::string("visual.imported-scene.render.v23\n") + intentGolden + "\n" + cameraNested
        && decode(encode(parentCamera), decoded) && sameRequest(parentCamera, decoded)
        && decoded.camera && decoded.camera->id.value == 50 && decoded.importedScene->sourceCameraNodeId == 7,
        "V23 wraps the unchanged full V22 composition and retains explicit Render camera priority");
    auto changedCameraLeaf = selectedLeaf; changedCameraLeaf.sourceCameraNodeId = 8;
    auto changedCamera = parentCamera; changedCamera.importedScene = std::make_shared<const Request>(changedCameraLeaf);
    check(valid(changedCamera) && !sameRequest(parentCamera, changedCamera),
        "parent source equality keeps the exact selected-camera leaf separate from parenting");
    decoded = sentinel;
    const auto packed = [&](ImportedParentIntentV1 value, const std::string& nested)
    {
        // Malformed values must reach the decoder even when encode() refuses.
        std::string bytes = "visual.imported-scene.render.v23\n";
        const auto integer = [&](std::uint64_t number, unsigned count)
        {
            constexpr char hex[] = "0123456789abcdef";
            for (unsigned i = 0; i < count; ++i)
            {
                const auto byte = static_cast<unsigned>((number >> (8 * i)) & 255);
                bytes += hex[byte / 16]; bytes += hex[byte % 16];
            }
        };
        integer(value.version, 4); integer(value.direction, 4);
        integer(value.composeStableId, 8); integer(value.importedSourceStableId, 8);
        integer(value.connectedInputMask, 4); integer(value.requestedParentMask, 4);
        integer(value.importedInputIndex, 4); integer(value.reserved, 4);
        integer(value.proceduralAnchorId, 4); integer(value.rootMembershipCount, 4);
        for (const auto& root : value.rootMembership)
        { integer(root.objectId, 4); integer(root.inputIndex, 4); }
        return bytes + "\n" + nested;
    };
    const auto refuses = [&](const Request& wrong, const std::string& wire)
    {
        check(!valid(wrong) && encode(wrong).empty() && !sameRequest(wrong, wrong),
            "invalid parent intent refuses validation, encoding and source reuse");
        const auto incumbent = decoded;
        check(!decodeCanonical(wire, decoded, error) && sameRequest(incumbent, decoded),
            "malformed parent wire refuses without replacing the incumbent request");
    };
    for (unsigned field = 0; field < 17; ++field)
    {
        auto bad = intent;
        switch (field)
        {
            case 0: bad.version = 0; break;
            case 1: bad.direction = 3; break;
            case 2: bad.composeStableId = 8; break;
            case 3: bad.importedSourceStableId = 72; break;
            case 4: bad.connectedInputMask = 15; break; // Missing connected group 3.
            case 5: bad.connectedInputMask = 3; break; // Entry names disconnected input 2.
            case 6: bad.requestedParentMask = 1; break;
            case 7: bad.requestedParentMask = 8; break;
            case 8: bad.importedInputIndex = 1; break;
            case 9: bad.reserved = 1; break;
            case 10: bad.proceduralAnchorId = 10; break;
            case 11: bad.rootMembershipCount = 65; break;
            case 12: bad.rootMembership[2] = {12, 3}; break; // Nonzero unused slot.
            case 13: bad.rootMembership[1] = {10, 2}; break; // Duplicate replaces root 11.
            case 14: bad.rootMembership[0].objectId = 0x8000000a; break;
            case 15: bad.rootMembership[1].objectId = 999; break;
            case 16: std::swap(bad.rootMembership[0], bad.rootMembership[1]); break;
        }
        auto wrong = request; wrong.importedParentIntent = bad;
        refuses(wrong, packed(bad, ordinary));
    }
    // Removing either flagged or unflagged root must fail full-union coverage.
    for (unsigned removed = 0; removed < 2; ++removed)
    {
        auto bad = intent; bad.rootMembershipCount = 1;
        bad.rootMembership[0] = intent.rootMembership[1 - removed]; bad.rootMembership[1] = {};
        auto wrong = request; wrong.importedParentIntent = bad;
        refuses(wrong, packed(bad, ordinary));
    }
    // A same-input sibling deletion also fails, even when group coverage remains.
    auto sibling = request; sibling.importedParentIntent->connectedInputMask = 3;
    sibling.importedParentIntent->rootMembership[1].inputIndex = 1;
    check(valid(sibling), "complete same-input sibling roots are representable");
    auto missingSibling = sibling; missingSibling.importedParentIntent->rootMembershipCount = 1;
    missingSibling.importedParentIntent->rootMembership[1] = {};
    refuses(missingSibling, packed(*missingSibling.importedParentIntent, ordinary));
    auto descendantScene = std::make_shared<Visual3DScene>(*base.sceneSnapshot);
    descendantScene->objectCount = 3;
    descendantScene->objects[2].id = {12}; descendantScene->objects[2].parent = {10};
    auto nonroot = request; nonroot.sceneSnapshot = descendantScene;
    nonroot.importedParentIntent->rootMembership[1].objectId = 12;
    auto descendantBase = base; descendantBase.sceneSnapshot = descendantScene;
    refuses(nonroot, packed(*nonroot.importedParentIntent, encode(descendantBase)));
    auto malformed = expected; malformed[32 + 1 + 48 * 2 + 1] = 'A';
    const auto v20 = [](const std::string& scene, const std::string& imported)
    { return std::string("visual.imported-scene.render.v20\n") + std::to_string(scene.size()) + " "
        + std::to_string(imported.size()) + "\n" + scene + imported; };
    auto sceneOnly = base; sceneOnly.importedScene.reset();
    for (const auto& wire : {malformed, expected + "extra", expected.substr(0, expected.size() - 1),
        expected.substr(0, 1153), packed(intent, expected), packed(intent, encode(leaf)),
        packed(intent, encode(*base.importedScene)), packed(*nonroot.importedParentIntent, encode(descendantBase)),
        v20(expected, encode(leaf)), v20(encode(sceneOnly), expected),
        std::string("visual.imported-scene.render.v11\n1\n") + expected})
        check(!decodeCanonical(wire, decoded, error) && sameRequest(sentinel, decoded),
            "uppercase, truncated, trailing, nested, leaf and nonroot V23 envelopes refuse transactionally");
    auto huge = expected;
    huge.resize(kMaximumV8EncodedBytes + 1, '0');
    check(!decodeCanonical(huge, decoded, error) && sameRequest(sentinel, decoded),
        "V23 retains the existing whole-wire ceiling");
    auto unrelated = leaf; unrelated.importedParentIntent = intent;
    refuses(unrelated, packed(intent, encode(leaf)));
    auto leafWithIntent = leaf; leafWithIntent.importedParentIntent = intent;
    auto duplicated = request; duplicated.importedScene = std::make_shared<const Request>(leafWithIntent);
    check(!valid(duplicated) && encode(duplicated).empty(), "parent intent belongs only to its one composed outer request");
    auto timeline = request; timeline.timelineGeometry.emplace();
    check(!valid(timeline), "parented transport refuses TimelineSceneBinding");
    auto family = request; family.timelineGeometryFamily.emplace_back();
    check(!valid(family), "parented transport refuses TimelineSceneBinding family");
    auto vertexLeaf = leaf; vertexLeaf.material = materialRequest(); vertexLeaf.material->vertexModifier = vertexModifierRequest();
    auto vertex = request; vertex.importedScene = std::make_shared<const Request>(vertexLeaf);
    check(!valid(vertex), "parented transport refuses the imported material vertex processor");
    auto staleLeaf = leaf; staleLeaf.evaluationRevision = 34;
    auto stale = request; stale.importedScene = std::make_shared<const Request>(staleLeaf);
    check(!valid(stale) && !sameRequest(stale, stale), "parented source equality includes outer/leaf evaluation identity");
    for (const auto mask : {2u, 4u, 6u})
    {
        auto chosen = request; chosen.importedParentIntent->requestedParentMask = mask;
        check(valid(chosen) && decode(encode(chosen), decoded) && sameRequest(chosen, decoded),
            "parent masks select a flagged subset without discarding complete unflagged membership");
        if (mask != 2) check(!sameRequest(request, chosen), "changed requested mask has a distinct source-admission identity");
    }
    auto regrouped = request;
    std::swap(regrouped.importedParentIntent->rootMembership[0].objectId,
              regrouped.importedParentIntent->rootMembership[1].objectId);
    check(valid(regrouped) && !sameRequest(request, regrouped),
        "complete compiler-issued regrouping is structurally valid but never equals the prior issued labels");
    auto unused = intent; unused.rootMembership[63].objectId = 1;
    check(!sameImportedParentIntent(intent, unused), "fixed-value equality retains every unused slot");
    auto maximumScene = std::make_shared<Visual3DScene>(*base.sceneSnapshot);
    maximumScene->objectCount = 64;
    ImportedParentIntentV1 maximum = intent;
    maximum.connectedInputMask = 3; maximum.rootMembershipCount = 64;
    for (std::size_t i = 0; i < 64; ++i)
    {
        if (i >= 2) maximumScene->objects[i].id = {static_cast<std::uint32_t>(10 + i)};
        maximum.rootMembership[i] = {static_cast<std::uint32_t>(10 + i), 1};
    }
    auto maximumRequest = base; maximumRequest.sceneSnapshot = maximumScene; maximumRequest.importedParentIntent = maximum;
    check(valid(maximumRequest) && decode(encode(maximumRequest), decoded) && sameRequest(maximumRequest, decoded)
        && decoded.importedParentIntent->rootMembershipCount == 64,
        "all 64 actual procedural roots fit the same fixed payload without unused or variable slots");
    auto emptyScene = std::make_shared<Visual3DScene>(*base.sceneSnapshot);
    emptyScene->objects = {}; emptyScene->objectCount = emptyScene->vertexCount = emptyScene->indexCount = 0;
    auto emptyBase = base; emptyBase.sceneSnapshot = emptyScene;
    check(valid(emptyBase), "legacy no-parent composed empty scene retains its released validity");
    auto emptyParent = emptyBase; emptyParent.importedParentIntent = intent;
    check(!valid(emptyParent), "parented complete-map route refuses connected empty procedural branches");
    auto direction1 = base;
    ImportedParentIntentV1 first;
    first.direction = 1; first.composeStableId = 7; first.importedSourceStableId = 71;
    first.connectedInputMask = 3; first.requestedParentMask = 2; first.importedInputIndex = 1;
    first.proceduralAnchorId = 10; direction1.importedParentIntent = first;
    check(valid(direction1) && decode(encode(direction1), decoded) && sameRequest(direction1, decoded),
        "procedural-first transport retains first-stored anchor and zero imported-root entries");
    auto nonmonotonic = std::make_shared<Visual3DScene>(*base.sceneSnapshot);
    nonmonotonic->objects[0].id = {90}; nonmonotonic->objects[0].parent = {11};
    direction1.sceneSnapshot = nonmonotonic; direction1.importedParentIntent->proceduralAnchorId = 90;
    check(valid(direction1), "procedural-first anchor may be a nonroot first record and need not have the lowest ID");
    auto wrongAnchor = direction1; wrongAnchor.importedParentIntent->proceduralAnchorId = 11;
    check(!valid(wrongAnchor), "procedural-first parenting never substitutes the lowest root for the first stored object");
    wrongAnchor.importedParentIntent->proceduralAnchorId = base.sceneSnapshot->activeCamera.value;
    check(!valid(wrongAnchor), "a camera selector supplies no procedural parent anchor");
    first.rootMembershipCount = 1; first.rootMembership[0] = {10, 1};
    auto wrongFirst = base; wrongFirst.importedParentIntent = first;
    refuses(wrongFirst, packed(first, ordinary));
    request.importedParentIntent.reset();
    check(encode(request) == ordinary, "removing optional parent intent restores exact V20 bytes");

    // This exercises the real production cache/consumer refusal. No native
    // frame preparation/publication runs. Existing source retirement still runs.
    auto plan = composedScenePlan(); plan.structuralRevision = 31;
    plan.nodeIds = {6, 11}; plan.nodeKinds = {videowire::geometry::kRetainedSceneOperation, "visual.3d.render"};
    plan.operations = {{6, plan.nodeKinds[0], "control-eval", ""}, {11, plan.nodeKinds[1], "native-gpu", expected}};
    plan.edges = {{6, 0, 11, 0}};
    plan.ports = {{6, 0, 1, "out", "control", "scene3D", "unspecified", "unspecified"},
        {11, 0, 1, "in", "control", "scene3D", "unspecified", "unspecified"},
        {11, 1, 1, "out", "frame", "image", "rgba8", "sRGB"}};
    FakeExecution execution; FakeLayer layer; std::vector<FakeReceipt> receipts;
    VisualImportedScenePlanCache cache;
    for (const auto use : {NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export})
    {
        auto noParentPlan = plan; noParentPlan.operations.back().payloadXml = ordinary;
        {
            const bool conditionOk = prepareVisualImportedSceneLayerAtTime({noParentPlan}, 8, 640, 360, 1.25, 24, use,
            &execution, &cache, layer, receipts, error, nullptr, {12, 34}) == VisualImportedScenePreparation::rendered;
            check(conditionOk, "unchanged no-parent production composition keeps its existing consumer lane: " + error);
        }
        layer.depthTexture = static_cast<unsigned>(execution.nextFrame->depthView);
        layer.nativeDepthTextureBackend = execution.nextFrame->backend();
        layer.nativeDepthTextureView = execution.nextFrame->depthView;
        layer.nativeDepthTextureDescriptor = execution.nextFrame->depthTextureDescriptor();
        layer.depthWidth = 640; layer.depthHeight = 360;
        layer.rawExportFrame = execution.nextFrame;
        layer.rawExportMask = 1; layer.rawExportRenderId = 11; layer.rawExportClipId = 8;
        layer.rawExportImageOutput = renderpassoutput::Output::Depth;
        const auto incumbent = layer;
        const auto sameDescriptor = [](const auto& a, const auto& b)
        {
            return a.backend == b.backend && a.viewKind == b.viewKind && a.format == b.format
                && a.imageHandle == b.imageHandle && a.textureViewHandle == b.textureViewHandle
                && a.width == b.width && a.height == b.height && a.sampleCount == b.sampleCount
                && a.sampledBinding == b.sampledBinding && a.deviceOrContextIdentity == b.deviceOrContextIdentity
                && a.rendererGeneration == b.rendererGeneration && a.colorSpace == b.colorSpace
                && a.transfer == b.transfer && a.rowOrder == b.rowOrder;
        };
        const auto calls = execution.previewCalls + execution.exportCalls;
        const auto receiptCount = receipts.size();
        const auto uploads = execution.staticGpuUploads;
        const auto retirement = execution.identityRetirements;
        const auto priorIdentity = execution.publishedIdentity;
        check(prepareVisualImportedSceneLayerAtTime({plan}, 8, 640, 360, 1.25, 24, use,
            &execution, &cache, layer, receipts, error, nullptr, {12, 34}) == VisualImportedScenePreparation::rejected
            && error == "Imported Scene Compose parenting requires its native parent binding owner and receipt",
            "typed parent intent refuses before production forwarding until the exact private helper join exists");
        check(execution.previewCalls + execution.exportCalls == calls && execution.staticGpuUploads == uploads
            && receipts.size() == receiptCount && layer.texture == incumbent.texture
            && layer.nativeTextureBackend == incumbent.nativeTextureBackend
            && layer.nativeTextureView == incumbent.nativeTextureView
            && sameDescriptor(layer.nativeTextureDescriptor, incumbent.nativeTextureDescriptor)
            && layer.texWidth == incumbent.texWidth && layer.texHeight == incumbent.texHeight
            && layer.depthTexture == incumbent.depthTexture
            && layer.nativeDepthTextureBackend == incumbent.nativeDepthTextureBackend
            && layer.nativeDepthTextureView == incumbent.nativeDepthTextureView
            && sameDescriptor(layer.nativeDepthTextureDescriptor, incumbent.nativeDepthTextureDescriptor)
            && layer.depthWidth == incumbent.depthWidth && layer.depthHeight == incumbent.depthHeight
            && layer.rawExportFrame == incumbent.rawExportFrame && layer.rawExportMask == incumbent.rawExportMask
            && layer.rawExportRenderId == incumbent.rawExportRenderId && layer.rawExportClipId == incumbent.rawExportClipId
            && layer.rawExportImageOutput == incumbent.rawExportImageOutput,
            "parent refusal preserves incumbent output and makes no executor/upload/receipt call");
        check(execution.publishedIdentity != priorIdentity && execution.identityRetirements == retirement + 1,
            "failed parent request retains the existing exact source-generation retirement policy");
    }
}

// Plugin-shaped request: the imported leaf carries the Surface override, and the
// outer request carries the procedural scene with one program on object 11.
visualimportedscenerender::Request importedOverrideRequest()
{
    using namespace HarmonicMIDI::grid;
    visualimportedscenerender::Request request;
    request.sourceStableId = 7; request.renderStableId = 12;
    request.structuralRevision = 31; request.evaluationRevision = 35;
    request.sceneSnapshot = composedScene();
    auto leaf = renderRequest(true, true);
    leaf.renderStableId = request.renderStableId;
    leaf.structuralRevision = request.structuralRevision;
    leaf.evaluationRevision = request.evaluationRevision;
    leaf.material->scene = *visualimportedsceneidentity::sceneId(0);
    leaf.material->sceneRevision = leaf.asset.version;
    leaf.material->structuralRevision = 31;
    leaf.material->evaluationRevision = 35;
    leaf.material->programRevision = leaf.material->binding.surfaceMaterialRevision = 31;
    leaf.material->binding.object = *visualimportedsceneidentity::objectId(0, 0);
    request.importedScene = std::make_shared<const visualimportedscenerender::Request>(leaf);

    geometrysurfacematerial::ObjectProgram procedural;
    procedural.program.sourceMaterial = 1;
    procedural.program.material = tableSurfaceRequest(request.sceneSnapshot->objects[1].material.value, true);
    auto& material = procedural.program.material;
    material.scene = request.sceneSnapshot->id;
    material.sceneSnapshot = request.sceneSnapshot;
    material.binding.object = request.sceneSnapshot->objects[1].id;
    material.sceneRevision = material.structuralRevision = 31;
    material.programRevision = material.binding.surfaceMaterialRevision = 31;
    material.evaluationRevision = 35;
    request.surfacePrograms.push_back(std::move(procedural));
    return request;
}

// The composed scene the executor builds: the imported object keeps its
// identity and takes the override's material, and the procedural object moves
// to its composed half of the identity space.
std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> importedOverrideComposedScene(
    const surfacematerialbinding::ImportedSceneMaterialRequest& material)
{
    using namespace HarmonicMIDI::grid;
    auto scene = std::make_shared<Visual3DScene>(*composedScene());
    scene->objects[0].id = *visualimportedsceneidentity::objectId(0, 0);
    scene->materials[0].id.value = geometrysurfacematerial::materialId(material);
    scene->materials[0].baseColorTexture.value = 1;
    scene->objects[0].material = scene->materials[0].id;
    scene->objects[1].id = SceneObjectId{0x80000000u | scene->objects[1].id.value};
    return scene;
}

// The executor's composition: procedural entries move to their composed targets
// and the imported override joins the same validated collection.
geometrysurfacematerial::ObjectPrograms importedOverrideComposition(
    const geometrysurfacematerial::ObjectPrograms& procedural,
    const surfacematerialbinding::ImportedSceneMaterialRequest& material,
    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& composed)
{
    using namespace HarmonicMIDI::grid;
    geometrysurfacematerial::ObjectPrograms programs = procedural;
    for (auto& entry : programs)
    {
        entry.program.material.binding.object = SceneObjectId{0x80000000u | entry.program.material.binding.object.value};
        entry.program.material.sceneSnapshot = composed;
    }
    std::string error;
    {
        const bool conditionOk = geometrysurfacematerial::appendImportedOverride(programs, material, composed, 31, 35, error);
        check(conditionOk, "the composed collection admits the imported override beside the procedural program: " + error);
    }
    return programs;
}

void testImportedSurfaceOverrideBesideProceduralPrograms()
{
    using namespace HarmonicMIDI::grid;
    const auto request = importedOverrideRequest();
    const auto encoded = visualimportedscenerender::encode(request);
    visualimportedscenerender::Request decoded;
    check(visualimportedscenerender::valid(request) && !encoded.empty()
        && encoded.find(visualimportedscenerender::kCompiledOperationHeaderV20) == 0
        && visualimportedscenerender::decode(encoded, decoded)
        && visualimportedscenerender::sameRequest(request, decoded)
        && visualimportedscenerender::encode(decoded) == encoded,
        "an imported Surface override beside procedural Surface programs is one admitted composed payload");

    const auto imported = *visualimportedsceneidentity::objectId(0, 0);
    const auto composed = importedOverrideComposedScene(*request.importedScene->material);
    const auto programs = importedOverrideComposition(request.surfacePrograms, *request.importedScene->material, composed);
    const auto importedEntry = std::find_if(programs.begin(), programs.end(),
        [](const auto& entry) { return entry.program.sourceMaterial == 0; });
    const auto proceduralEntry = std::find_if(programs.begin(), programs.end(),
        [](const auto& entry) { return entry.program.sourceMaterial == 1; });
    check(programs.size() == 2 && importedEntry != programs.end() && proceduralEntry != programs.end()
        && importedEntry->program.material.binding.object == imported
        && proceduralEntry->program.material.binding.object == composed->objects[1].id
        && importedEntry->program.material.sceneSnapshot == composed
        && importedEntry->program.material.scene == composed->id
        && importedEntry->program.material.programRevision == 31
        && importedEntry->program.material.evaluationRevision == 35,
        "the imported override binds only its imported object and the procedural program only its target");
}

void testImportedSurfaceOverrideReachesCompositionExecutor()
{
    using namespace HarmonicMIDI::grid;
    using videohelper::importedscene::NativeImportedSceneRenderUse;
    using videohelper::importedscene::VisualImportedScenePlanCache;
    using videohelper::importedscene::VisualImportedScenePreparation;
    using videohelper::importedscene::prepareVisualImportedSceneLayerAtTime;
    const auto request = importedOverrideRequest();
    const auto encoded = visualimportedscenerender::encode(request);
    check(!encoded.empty(), "the plugin-shaped imported override payload is admitted");
    if (encoded.empty()) return;
    auto plan = composedScenePlan();
    plan.structuralRevision = 31;
    plan.nodeIds = {6,11};
    plan.nodeKinds = {videowire::geometry::kRetainedSceneOperation, "visual.3d.render"};
    plan.operations = {{6,plan.nodeKinds[0],"control-eval",""},
        {11,plan.nodeKinds[1],"native-gpu",encoded}};
    plan.edges = {{6,0,11,0}};
    plan.ports = {{6,0,1,"out","control","scene3D","unspecified","unspecified"},
        {11,0,1,"in","control","scene3D","unspecified","unspecified"},
        {11,1,1,"out","frame","image","rgba8","sRGB"}};
    FakeExecution execution;
    FakeLayer layer;
    std::vector<FakeReceipt> receipts;
    VisualImportedScenePlanCache cache;
    std::string error;
    {
        const bool conditionOk = prepareVisualImportedSceneLayerAtTime({plan},8,640,360,1.25,24,NativeImportedSceneRenderUse::Preview,
        &execution,&cache,layer,receipts,error,nullptr,{12,34}) == VisualImportedScenePreparation::rendered;
        check(conditionOk, "the executor receives the plugin-shaped imported override: " + error);
    }
    const auto& actual = execution.lastRequest;
    check(actual.importedScene && actual.material && actual.surfacePrograms.size() == 1
        && actual.material->binding.object == *visualimportedsceneidentity::objectId(0, 0)
        && actual.surfacePrograms[0].program.material.binding.object == request.sceneSnapshot->objects[1].id
        && !actual.material->sceneSnapshot,
        "the executor request carries the leaf override and the untouched procedural program");
    const auto composed = importedOverrideComposedScene(*actual.material);
    const auto programs = importedOverrideComposition(actual.surfacePrograms, *actual.material, composed);
    check(programs.size() == 2 && programs[0].program.sourceMaterial == 0
        && programs[0].program.material.binding.object == *visualimportedsceneidentity::objectId(0, 0)
        && programs[1].program.sourceMaterial == 1
        && programs[1].program.material.binding.object == composed->objects[1].id,
        "the composed list keeps the override on its imported object and the procedural program on its target");
}

void testImportedSceneCompositionHandoff()
{
    using namespace videohelper::importedscene;
    using namespace HarmonicMIDI::grid;
    auto plan = composedScenePlan();
    plan.structuralRevision = 31;
    visualimportedscenerender::Request request;
    request.sourceStableId = 7; request.renderStableId = 12;
    request.structuralRevision = 31; request.evaluationRevision = 35;
    request.sceneSnapshot = composedScene();
    auto leaf = renderRequest(true, true);
    leaf.renderStableId = request.renderStableId;
    leaf.structuralRevision = request.structuralRevision;
    leaf.evaluationRevision = request.evaluationRevision;
    leaf.material->scene = *visualimportedsceneidentity::sceneId(0);
    leaf.material->sceneRevision = leaf.asset.version;
    leaf.material->structuralRevision = 31;
    leaf.material->evaluationRevision = 35;
    leaf.material->programRevision = leaf.material->binding.surfaceMaterialRevision = 31;
    request.importedScene = std::make_shared<const visualimportedscenerender::Request>(leaf);
    const auto encoded = visualimportedscenerender::encode(request);
    visualimportedscenerender::Request decoded;
    check(encoded.find(visualimportedscenerender::kCompiledOperationHeaderV20) == 0
        && visualimportedscenerender::decode(encoded, decoded)
        && visualimportedscenerender::sameRequest(request, decoded)
        && decoded.structuralRevision == 31 && decoded.evaluationRevision == 35
        && decoded.importedScene && decoded.importedScene->structuralRevision == 31
        && decoded.importedScene->evaluationRevision == 35
        && visualimportedscenerender::encode(decoded) == encoded,
        "V20 round trip preserves imported selection, material target, playback and procedural owner");
    if (encoded.empty()) return;
    check(visualimportedscenerender::modelAssets(decoded).size() == 1
        && visualanimationimport::sameAsset(visualimportedscenerender::modelAssets(decoded)[0], leaf.asset),
        "composition payload transfer enumerates its exact original asset");
    for (unsigned authoredText = 1; authoredText < 4; ++authoredText)
    {
        auto authoredLeaf = leaf;
        if (authoredText & 1)
            authoredLeaf.asset.id = "model-" + std::string(visualimportedscenerender::kCompiledOperationHeaderV20) + "-1";
        authoredLeaf.deformation->asset = authoredLeaf.asset;
        if (authoredText & 2)
            authoredLeaf.deformation->clipName = "Idle " + std::string(visualimportedscenerender::kCompiledOperationHeaderV20);
        auto authoredRequest = request;
        authoredRequest.importedScene = std::make_shared<const visualimportedscenerender::Request>(authoredLeaf);
        const auto authoredBytes = visualimportedscenerender::encode(authoredRequest);
        visualimportedscenerender::Request restored;
        check(!authoredBytes.empty() && visualimportedscenerender::decode(authoredBytes, restored)
            && visualimportedscenerender::sameRequest(authoredRequest, restored)
            && visualimportedscenerender::encode(restored) == authoredBytes,
            "V20 header text in authored asset IDs and clip names round trips unchanged");
    }
    plan.nodeIds = {6,11};
    plan.nodeKinds = {videowire::geometry::kRetainedSceneOperation, "visual.3d.render"};
    plan.operations = {{6,plan.nodeKinds[0],"control-eval",""},
        {11,plan.nodeKinds[1],"native-gpu",encoded}};
    plan.edges = {{6,0,11,0}};
    plan.ports = {{6,0,1,"out","control","scene3D","unspecified","unspecified"},
        {11,0,1,"in","control","scene3D","unspecified","unspecified"},
        {11,1,1,"out","frame","image","rgba8","sRGB"}};
    FakeExecution execution;
    FakeLayer layer;
    std::vector<FakeReceipt> receipts;
    VisualImportedScenePlanCache cache;
    std::string error;
    for (const auto use : {NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export})
    {
        {
            const bool conditionOk = prepareVisualImportedSceneLayerAtTime({plan},8,640,360,1.25,24,use,
            &execution,&cache,layer,receipts,error,nullptr,{12,34}) == VisualImportedScenePreparation::rendered;
            check(conditionOk, "unparented imported composition reaches the existing preview/export executor: " + error);
        }
        const auto& actual = execution.lastRequest;
        check(actual.importedScene && actual.sceneSnapshot && actual.sceneSnapshot->objectCount == 2
            && visualanimationimport::sameAsset(actual.asset, leaf.asset)
            && actual.sceneIndex == leaf.sceneIndex && actual.deformation && actual.material
            && actual.deformation->sourceStableId == leaf.deformation->sourceStableId
            && actual.deformation->deformationStableId == leaf.deformation->deformationStableId
            && actual.deformation->clipName == "Idle" && actual.deformation->playback.speed == 1.5
            && actual.deformation->playback.timelineSeconds == 1.25
            && actual.frame.frame == 30 && actual.frame.rateNumerator == 24 && actual.frame.rateDenominator == 1
            && actual.structuralRevision == 31 && actual.evaluationRevision == 35
            && actual.projectGeneration == 12 && actual.helperGeneration == 34
            && actual.material->binding.object == leaf.material->binding.object
            && actual.material->scene == leaf.material->scene && !actual.material->sceneSnapshot,
            "preview/export keep GLB identity and material target separate from the procedural scene at the exact clock");
    }
    const auto sentinel = decoded;
    auto sceneOnly = request; sceneOnly.importedScene.reset();
    const auto sceneBytes = visualimportedscenerender::encode(sceneOnly);
    const auto leafBytes = visualimportedscenerender::encode(leaf);
    const auto envelope = [](const std::string& scene, const std::string& imported)
    {
        return std::string(visualimportedscenerender::kCompiledOperationHeaderV20) + "\n"
            + std::to_string(scene.size()) + " " + std::to_string(imported.size()) + "\n" + scene + imported;
    };
    check(encoded == envelope(sceneBytes,leafBytes),
        "V20 retains the exact released nested scene and leaf bytes");
    auto cameraLeaf = leaf;
    cameraLeaf.material.reset(); cameraLeaf.deformation.reset();
    const std::string leafGolden = "visual.imported-scene.render.v1\n71 12\n"
        "\"model-asset-1\"\n3\n\""
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        "\"\n\"model/gltf-binary\"\n4096\n1 0\n";
    check(visualimportedscenerender::encode(cameraLeaf) == leafGolden,
        "no-parent leaf retains the literal V1 asset and selection bytes at revisions 31 and 35");
    cameraLeaf.sourceCameraNodeId = 7;
    const auto selectedLeafGolden = std::string("visual.imported-scene.render.v22\nsource 7 ")
        + std::to_string(leafGolden.size()) + "\n" + leafGolden;
    check(visualimportedscenerender::encode(cameraLeaf) == selectedLeafGolden,
        "no-parent selected camera adds only its released V22 source wrapper");
    auto cameraComposition = request;
    cameraComposition.importedScene = std::make_shared<const visualimportedscenerender::Request>(cameraLeaf);
    const auto cameraCompositionGolden = envelope(sceneBytes, selectedLeafGolden);
    check(visualimportedscenerender::encode(cameraComposition) == cameraCompositionGolden
        && visualimportedscenerender::decode(cameraCompositionGolden, decoded)
        && visualimportedscenerender::sameRequest(cameraComposition, decoded)
        && !decoded.importedParentIntent && decoded.structuralRevision == 31 && decoded.evaluationRevision == 35,
        "no-parent V20 camera composition conserves its literal leaf and procedural bytes");
    cameraComposition.camera = cameraComposition.sceneSnapshot->cameras[0];
    // Authored independently from the fixture's default unit TRS, z=4 camera,
    // binary32 pi/3 FOV, near=0.1 and far=1000.
    const std::string cameraGolden = "50 0 0 1082130432 0 0 0 1065353216 1065353216 "
        "1065353216 1065353216 1065749138 1036831949 1148846080\n";
    const auto overrideGolden = std::string("visual.imported-scene.render.v22\noverride ")
        + std::to_string(cameraCompositionGolden.size()) + "\n" + cameraGolden + cameraCompositionGolden;
    check(visualimportedscenerender::encode(cameraComposition) == overrideGolden
        && visualimportedscenerender::decode(overrideGolden, decoded)
        && visualimportedscenerender::sameRequest(cameraComposition, decoded)
        && decoded.camera && decoded.importedScene->sourceCameraNodeId == 7,
        "no-parent explicit Render camera preserves literal V22 override priority and leaf selection");
    decoded = sentinel;

    const std::array<std::string_view,5> leafHeaders {
        visualimportedscenerender::kCompiledOperationHeader,
        visualimportedscenerender::kCompiledOperationHeaderV2,
        visualimportedscenerender::kCompiledOperationHeaderV3,
        visualimportedscenerender::kCompiledOperationHeaderV3,
        visualimportedscenerender::kCompiledOperationHeaderV4
    };
    for (std::size_t variant = 0; variant < leafHeaders.size(); ++variant)
    {
        auto contextualLeaf = leaf;
        if (variant == 0 || variant == 2) contextualLeaf.material.reset();
        if (variant < 2) contextualLeaf.deformation.reset();
        if (variant == 4) contextualLeaf.material->vertexModifier = vertexModifierRequest();
        auto legacyLeaf = contextualLeaf;
        legacyLeaf.structuralRevision = legacyLeaf.evaluationRevision = 0;
        const auto legacyBytes = visualimportedscenerender::encode(legacyLeaf);
        visualimportedscenerender::Request legacyDecoded;
        check(!legacyBytes.empty() && legacyBytes.find(leafHeaders[variant]) == 0
            && visualimportedscenerender::encode(contextualLeaf) == legacyBytes
            && visualimportedscenerender::decode(legacyBytes,legacyDecoded)
            && legacyDecoded.structuralRevision == 0 && legacyDecoded.evaluationRevision == 0
            && visualimportedscenerender::sameRequest(legacyLeaf,legacyDecoded),
            "standalone V1-V4 leaves conserve their bytes and default outer revisions");
        if (variant == 0)
        {
            const auto expected = std::string("visual.imported-scene.render.v1\n71 12\n"
                "\"model-asset-1\"\n3\n\"") + std::string(64,'a')
                + "\"\n\"model/gltf-binary\"\n4096\n1 0\n";
            check(legacyBytes == expected,"V20 embeds the independently authored legacy V1 bytes unchanged");
        }
        auto contextual = request;
        contextual.importedScene = std::make_shared<const visualimportedscenerender::Request>(contextualLeaf);
        const auto contextualBytes = visualimportedscenerender::encode(contextual);
        visualimportedscenerender::Request restored;
        check(contextualBytes == envelope(sceneBytes,legacyBytes)
            && visualimportedscenerender::decode(contextualBytes,restored)
            && restored.importedScene && restored.importedScene->structuralRevision == 31
            && restored.importedScene->evaluationRevision == 35
            && visualimportedscenerender::sameRequest(contextual,restored)
            && visualimportedscenerender::encode(restored) == contextualBytes,
            "V20 restores literal scene revisions for each released leaf variant without changing nested bytes");
    }
    for (unsigned mismatch = 0; mismatch < 6; ++mismatch)
    {
        auto wrongLeaf = leaf;
        auto& material = *wrongLeaf.material;
        switch (mismatch)
        {
            case 0: material.scene = *visualimportedsceneidentity::sceneId(1); break;
            case 1: material.sceneRevision = 4; break;
            case 2: material.structuralRevision = 30; break;
            case 3: material.evaluationRevision = 34; break;
            case 4: material.programRevision = 30; break;
            case 5: material.binding.surfaceMaterialRevision = 30; break;
        }
        const auto wrongBytes = visualimportedscenerender::encode(wrongLeaf);
        visualimportedscenerender::Request standalone;
        check(!wrongBytes.empty() && visualimportedscenerender::decode(wrongBytes,standalone),
            "contextual material mismatch remains a valid standalone leaf before V20 admission");
        check(!visualimportedscenerender::decode(envelope(sceneBytes,wrongBytes),decoded)
            && visualimportedscenerender::sameRequest(sentinel,decoded),
            "V20 refuses each mismatched embedded material identity and revision without replacing output");
    }
    const auto wrappedComposition = std::string(visualimportedscenerender::kCompiledOperationHeaderV11) + "\n1\n" + encoded;
    const auto twiceWrappedComposition = std::string(visualimportedscenerender::kCompiledOperationHeaderV11)
        + "\n1\n" + std::string(visualimportedscenerender::kCompiledOperationHeaderV9) + "\n1\n" + encoded;
    for (const auto& bad : {encoded + "extra", encoded.substr(0, encoded.size() - 1),
        std::string(visualimportedscenerender::kCompiledOperationHeaderV20) + "\n999999999 1\nx",
        std::string(visualimportedscenerender::kCompiledOperationHeaderV20) + "\n" +
            std::to_string(encoded.size()) + " " + std::to_string(encoded.size()) + "\n" + encoded + encoded,
        envelope(encoded, leafBytes), envelope(sceneBytes, encoded),
        envelope(wrappedComposition, leafBytes), envelope(sceneBytes, wrappedComposition),
        envelope(twiceWrappedComposition, leafBytes), envelope(sceneBytes, twiceWrappedComposition),
        envelope(sceneBytes,leafBytes.substr(0,leafBytes.size()-1)),
        envelope(sceneBytes,replaceLine(leafBytes,8,"1 999999999")),
        envelope(sceneBytes,replaceLine(leafBytes,0,"visual.imported-scene.render.v999")),
        envelope(sceneBytes,leafBytes + "\n"), envelope(sceneBytes + "\n",leafBytes)})
        check(!visualimportedscenerender::decode(bad, decoded)
            && visualimportedscenerender::sameRequest(sentinel, decoded),
            "V20 rejects trailing, truncated, oversized and recursive envelopes without publishing partial state");
    auto nested = request;
    nested.importedScene = std::make_shared<const visualimportedscenerender::Request>(request);
    check(!visualimportedscenerender::sameRequest(nested, nested)
        && !visualimportedscenerender::sameRequest(request, nested)
        && !visualimportedscenerender::sameRequest(nested, request),
        "request equality refuses nested composition shapes on either side");
    auto self = std::make_shared<visualimportedscenerender::Request>(request);
    self->importedScene = self;
    check(!visualimportedscenerender::sameRequest(*self, *self),
        "request equality refuses a self-referencing composition without descending");
    self->importedScene.reset();
    auto first = std::make_shared<visualimportedscenerender::Request>(request);
    auto second = std::make_shared<visualimportedscenerender::Request>(request);
    first->importedScene = second; second->importedScene = first;
    check(!visualimportedscenerender::sameRequest(*first, *second)
        && !visualimportedscenerender::sameRequest(*second, *first),
        "request equality refuses a two-node composition cycle on either side");
    first->importedScene.reset(); second->importedScene.reset();
    auto stale = request;
    auto staleLeaf = leaf; staleLeaf.evaluationRevision = 34;
    stale.importedScene = std::make_shared<const visualimportedscenerender::Request>(staleLeaf);
    check(!visualimportedscenerender::valid(stale) && visualimportedscenerender::encode(stale).empty(),
        "composition refuses a stale imported revision");
    auto duplicate = request; duplicate.asset = leaf.asset;
    check(!visualimportedscenerender::valid(duplicate), "composition refuses a second asset authority in the outer request");
    auto wrongAsset = leaf; wrongAsset.asset.contentSha256 = std::string(64,'c');
    auto mismatched = request;
    mismatched.importedScene = std::make_shared<const visualimportedscenerender::Request>(wrongAsset);
    check(!visualimportedscenerender::valid(mismatched), "composition refuses an animation bound to a different asset fingerprint");
    auto oversized = std::make_shared<Visual3DScene>(*request.sceneSnapshot);
    oversized->objectCount = Visual3DScene::kMaxObjects + 1;
    mismatched = request; mismatched.sceneSnapshot = oversized;
    check(!visualimportedscenerender::valid(mismatched) && visualimportedscenerender::encode(mismatched).empty(),
        "composition refuses scene capacity overflow before import admission");
    auto procedural = request; procedural.importedScene.reset();
    check(visualimportedscenerender::encode(procedural).find(visualimportedscenerender::kCompiledOperationHeaderV8) == 0
        && visualimportedscenerender::encode(renderRequest()).find(visualimportedscenerender::kCompiledOperationHeader) == 0,
        "unextended procedural and standalone imported requests retain their legacy versions");
    auto object = request.sceneSnapshot->objects[0];
    const auto mapped = scenecomposition::proceduralObjectId(object.id);
    check(mapped.has_value() && mapped->value == (0x80000000u | object.id.value)
        && !scenecomposition::proceduralObjectId(SceneObjectId{0x80000001u}),
        "procedural draw membership uses an explicit bounded namespace derived from its authored identity");
    if (mapped) object.id = *mapped;
    arbitgpu::NativeImportedSceneRuntimeInputs runtime;
    runtime.objectNodeStableId = 1;
    check(!arbitgpu::runtimeTargetsObject(runtime, object),
        "selected original GLB node membership cannot reinterpret a procedural draw ID");
}

videowire::CompiledVisualLayerPlan importedScenePlan(
    bool withMaterial = false, bool withDeformation = false,
    bool withDiffractionMaterial = false)
{
    videowire::CompiledVisualLayerPlan plan;
    plan.clipId = 7;
    plan.structuralRevision = 9;
    plan.producerValidated = true;
    plan.nodeKinds = {
        std::string(visualimportedscenerender::kSourceNodeKind),
        std::string(visualimportedscenerender::kRenderNodeKind)
    };
    plan.nodeIds = { 70, 71 };
    const std::array<const char*, 5> sourceTypes {
        "mesh", "skeleton", "morphTargets", "animationClip", "scene3D"
    };
    for (int port = 0; port < 5; ++port)
        plan.ports.push_back({ 70, port, 1, "out", "control",
                              sourceTypes[static_cast<std::size_t>(port)],
                              "unspecified", "unspecified" });
    plan.ports.push_back({ 71, 0, 1, "in", "control", "scene3D",
                           "unspecified", "unspecified" });
    plan.ports.push_back({ 71, 1, 1, "out", "frame", "image",
                           "rgba8", "sRGB" });
    plan.edges = { { 70, 4, 71, 0 } };

    const auto request = renderRequest(
        withMaterial, withDeformation, withDiffractionMaterial);
    plan.operations.push_back({
        70, std::string(visualimportedscenerender::kSourceNodeKind),
        std::string(visualimportedscenerender::kSourceBackendCapability), "" });
    if (withDeformation)
    {
        plan.nodeKinds.insert(plan.nodeKinds.end() - 1,
                              std::string(visualanimationoperation::kDeformationNodeKind));
        plan.nodeIds.insert(plan.nodeIds.end() - 1, 72);
        const std::array<const char*, 5> deformationTypes {
            "mesh", "skeleton", "morphTargets", "animationClip", "mesh"
        };
        for (int port = 0; port < 5; ++port)
            plan.ports.push_back({
                72, port, 1, port == 4 ? "out" : "in", "control",
                deformationTypes[static_cast<std::size_t>(port)],
                "unspecified", "unspecified" });
        plan.ports.push_back({ 71, 2, 1, "in", "control", "material",
                               "unspecified", "unspecified" });
        plan.ports.push_back({ 71, 3, 1, "in", "control", "mesh",
                               "unspecified", "unspecified" });
        plan.ports.push_back({ 71, 4, 1, "out", "frame", "depth",
                               "r32f", "unspecified" });
        for (int port = 0; port < 4; ++port)
            plan.edges.push_back({ 70, port, 72, port });
        plan.edges.push_back({ 72, 4, 71, 3 });
        plan.operations.push_back({
            72, std::string(visualanimationoperation::kDeformationNodeKind),
            std::string(visualanimationoperation::kDeformationBackendCapability),
            visualanimationoperation::encode(*request.deformation) });
    }
    if (withMaterial)
    {
        const int firstMaterialValueNode = withDeformation ? 73 : 72;
        const int materialTerminalNode = firstMaterialValueNode + 10;
        const auto& program = request.material->program;
        for (std::size_t index = 0; index < program.operations.size(); ++index)
        {
            const auto& value = program.operations[index];
            const auto nodeId = firstMaterialValueNode + static_cast<int>(index);
            const auto kind = value.resultType == surfacematerial::ValueType::UInt
                ? "visual.surface.constant.uint"
                : std::string("visual.surface.constant.")
                    + std::string(surfacematerial::token(value.resultType));
            plan.nodeKinds.insert(plan.nodeKinds.end() - 1, kind);
            plan.nodeIds.insert(plan.nodeIds.end() - 1, nodeId);
            plan.ports.push_back({
                nodeId, 0, surfaceChannels(value.resultType),
                "out", "control", surfaceDataType(value.resultType),
                "unspecified", "unspecified" });
            plan.operations.push_back({ nodeId, kind, "control-eval", "" });
        }
        plan.nodeKinds.insert(plan.nodeKinds.end() - 1, "visual.surface.material");
        plan.nodeIds.insert(plan.nodeIds.end() - 1, materialTerminalNode);
        for (const auto output : surfacematerial::kSurfaceOutputs)
        {
            const auto port = static_cast<int>(output);
            const auto type = surfacematerial::outputType(output);
            const auto valueNode = firstMaterialValueNode + port;
            plan.ports.push_back({
                materialTerminalNode, port,
                surfaceChannels(type), "in", "control", surfaceDataType(type),
                "unspecified", "unspecified" });
            plan.edges.push_back({ valueNode, 0, materialTerminalNode, port });
        }
        plan.ports.push_back({ materialTerminalNode, 10, 1, "out", "control", "material",
                               "unspecified", "unspecified" });
        if (!withDeformation)
            plan.ports.push_back({ 71, 2, 1, "in", "control", "material",
                                   "unspecified", "unspecified" });
        plan.edges.push_back({ materialTerminalNode, 10, 71, 2 });
        plan.operations.push_back({
            materialTerminalNode, "visual.surface.material", "control-eval", "" });
    }
    if (withDiffractionMaterial)
    {
        const int materialTerminalNode = withDeformation ? 73 : 72;
        plan.nodeKinds.insert(
            plan.nodeKinds.end() - 1, "visual.material.diffraction-grating");
        plan.nodeIds.insert(plan.nodeIds.end() - 1, materialTerminalNode);
        plan.ports.push_back({ materialTerminalNode, 0, 1, "out", "control",
                               "material", "unspecified", "unspecified" });
        if (! withDeformation)
            plan.ports.push_back({ 71, 2, 1, "in", "control", "material",
                                   "unspecified", "unspecified" });
        plan.edges.push_back({ materialTerminalNode, 0, 71, 2 });
        plan.operations.push_back({
            materialTerminalNode, "visual.material.diffraction-grating",
            "control-eval", "" });
    }
    plan.operations.push_back({
        71, std::string(visualimportedscenerender::kRenderNodeKind),
        std::string(visualimportedscenerender::kRenderBackendCapability),
        visualimportedscenerender::encode(request) });
    return plan;
}

void testImportedSceneCompositionGeneratedIdentities()
{
    using namespace videohelper::importedscene;
    using namespace HarmonicMIDI::grid;
    // Object 1 stands in for a Geometry Core instance: its ID already carries
    // the high bit and its parent is the authored Instances root.
    auto generated = *composedScene();
    generated.objects[1].id.value = 0x80000000u | 0x1234567u;
    generated.objects[1].parent = generated.objects[0].id;
    visualimportedscenerender::Request request;
    request.sourceStableId = 7; request.renderStableId = 12;
    request.structuralRevision = 31; request.evaluationRevision = 35;
    request.sceneSnapshot = std::make_shared<const Visual3DScene>(generated);
    auto leaf = renderRequest();
    leaf.renderStableId = request.renderStableId;
    leaf.structuralRevision = request.structuralRevision;
    leaf.evaluationRevision = request.evaluationRevision;
    request.importedScene = std::make_shared<const visualimportedscenerender::Request>(leaf);
    check(scenecomposition::distinctComposedProceduralObjects(generated)
        && !scenecomposition::proceduralObjectId(generated.objects[1].id)
        && scenecomposition::composedProceduralObjectId(generated.objects[1].id)->value == generated.objects[1].id.value
        && scenecomposition::composedProceduralObjectId(generated.objects[0].id)->value == (0x80000000u | 10u),
        "generated identities keep their value while authored identities gain the high bit");
    const auto encoded = visualimportedscenerender::encode(request);
    visualimportedscenerender::Request decoded;
    check(encoded.find(visualimportedscenerender::kCompiledOperationHeaderV20) == 0
        && visualimportedscenerender::decode(encoded, decoded)
        && visualimportedscenerender::sameRequest(request, decoded)
        && visualimportedscenerender::encode(decoded) == encoded,
        "V20 round trip admits a generated instance record beside the imported leaf");
    if (encoded.empty()) return;

    auto plan = composedScenePlan();
    plan.structuralRevision = 31;
    plan.nodeIds = {6,11};
    plan.nodeKinds = {videowire::geometry::kRetainedSceneOperation, "visual.3d.render"};
    plan.operations = {{6,plan.nodeKinds[0],"control-eval",""},
        {11,plan.nodeKinds[1],"native-gpu",encoded}};
    plan.edges = {{6,0,11,0}};
    plan.ports = {{6,0,1,"out","control","scene3D","unspecified","unspecified"},
        {11,0,1,"in","control","scene3D","unspecified","unspecified"},
        {11,1,1,"out","frame","image","rgba8","sRGB"}};
    FakeExecution execution;
    FakeLayer layer;
    std::vector<FakeReceipt> receipts;
    VisualImportedScenePlanCache cache;
    std::string error;
    for (const auto use : {NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export})
    {
        {
            const bool conditionOk = prepareVisualImportedSceneLayerAtTime({plan},8,640,360,1.25,24,use,
            &execution,&cache,layer,receipts,error,nullptr,{12,34}) == VisualImportedScenePreparation::rendered;
            check(conditionOk, "imported composition with a generated instance reaches the preview/export executor: " + error);
        }
        const auto& actual = execution.lastRequest;
        check(actual.importedScene && actual.sceneSnapshot && actual.sceneSnapshot->objectCount == 2
            && actual.sceneSnapshot->objects[0].id.value == 10u
            && actual.sceneSnapshot->objects[1].id.value == (0x80000000u | 0x1234567u)
            && actual.sceneSnapshot->objects[1].parent.value == 10u
            && visualanimationimport::sameAsset(actual.asset, leaf.asset)
            && !actual.deformation && !actual.material
            && actual.structuralRevision == 31 && actual.evaluationRevision == 35,
            "preview/export receive the generated instance record and the imported asset unchanged");
    }
    check(execution.previewCalls == 1 && execution.exportCalls == 1,
        "generated-identity composition executes once per use");

    // Authored 10 and generated 0x8000000a would both map to 0x8000000a.
    auto collision = generated;
    collision.objects[1].id.value = 0x80000000u | 10u;
    auto refused = request;
    refused.sceneSnapshot = std::make_shared<const Visual3DScene>(collision);
    check(!scenecomposition::distinctComposedProceduralObjects(collision)
        && !visualimportedscenerender::valid(refused)
        && visualimportedscenerender::encode(refused).empty(),
        "authored and generated identities that share their low bits are refused");
}

void testAuthoredImportedStarterTopologies()
{
    using videohelper::tests::appendImportedStarterOutput;
    using videohelper::tests::useRegisteredImportedStarterPorts;
    using videohelper::importedscene::NativeImportedSceneRenderUse;
    using videohelper::importedscene::VisualImportedScenePreparation;
    using videohelper::importedscene::prepareVisualImportedSceneLayer;
    std::string error;
    videowire::VisualLayerExecution compiled;
    const auto operation = [](auto& plan, int node) -> auto& {
        return *std::find_if(plan.operations.begin(), plan.operations.end(),
            [&](const auto& item) { return item.nodeId == node; });
    };
    const auto addBefore = [](auto& plan, int before, int node, const char* kind,
                              const char* backend, const std::string& payload = "") {
        const auto found = std::find(plan.nodeIds.begin(), plan.nodeIds.end(), before);
        const auto index = found - plan.nodeIds.begin();
        plan.nodeIds.insert(found, node);
        plan.nodeKinds.insert(plan.nodeKinds.begin() + index, kind);
        plan.operations.insert(plan.operations.begin() + index, {node, kind, backend, payload});
    };
    const auto accepted = [&](const auto& plan) {
        error.clear();
        return videowire::validateCompiledVisualLayerPlans({}, {plan}, true, error)
            && videowire::compileVisualLayerExecution(plan, compiled, error);
    };
    const auto reject = [&](const auto& plan, const char* reason) {
        error.clear();
        check(!videowire::compileVisualLayerExecution(plan, compiled, error), reason);
    };

    // BuiltInVisualModules::depthReactiveFogSceneStarter keeps the complete
    // animated material graph and binds both fog inputs to its one Render 3D.
    auto fog = importedScenePlan(true, true);
    useRegisteredImportedStarterPorts(fog, 71);
    addBefore(fog, -1, 90, "visual.depth.fog", "native-gpu",
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<NodeParams near=\"0\" far=\"1\" density=\"1.35\" "
        "red=\"0.72\" green=\"0.82\" blue=\"0.95\" alpha=\"1\"/>");
    fog.ports.insert(fog.ports.end(), {
        {90, 0, 1, "in", "frame", "image", "rgba8", "sRGB"},
        {90, 1, 1, "in", "frame", "depth", "r32f", "unspecified"},
        {90, 2, 1, "out", "frame", "image", "rgba8", "sRGB"}
    });
    fog.edges.insert(fog.edges.end(), {{71, 1, 90, 0}, {71, 4, 90, 1}});
    {
        const bool conditionOk = accepted(fog) && compiled.depthFog && compiled.depthEffect == 1;
        check(conditionOk, "authored fog admits the module image before its outer output is connected: " + error);
    }
    appendImportedStarterOutput(fog, 90, 2);
    {
        const bool conditionOk = accepted(fog) && compiled.importedSceneRender && compiled.importedSceneRender->deformation
        && compiled.importedSceneRender->material && compiled.depthFog
        && std::abs(compiled.fogDensity - 1.35f) < 0.0001f;
        check(conditionOk, "authored Depth-Reactive Fog retains animation, material and outer Video Out: " + error);
    }
    for (const auto use : {NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export})
    {
        FakeExecution native;
        videorender::LayerDesc layer;
        std::vector<FakeReceipt> owners;
        const auto prepared = prepareVisualImportedSceneLayer({fog}, 7, 640, 360, use,
            &native, layer, owners, error);
        {
            const bool conditionOk = prepared == VisualImportedScenePreparation::rendered
            && videowire::executeVisualLayerPlan({fog}, 7, layer, error,
                use == NativeImportedSceneRenderUse::Preview ? videohelper::geometry::PlanUse::preview
                    : videohelper::geometry::PlanUse::exportRender)
            && layer.depthFog && layer.depthEffect == 1 && layer.depthTexture == 66
            && layer.nativeDepthTextureView == 66 && !owners.empty()
            && std::abs(layer.fogBlue - 0.95f) < 0.0001f;
            check(conditionOk, "preview and export compose fog using their retained imported depth: " + error);
        }
    }
    auto invalid = fog;
    invalid.edges.erase(invalid.edges.end() - 2);
    reject(invalid, "fog rejects an unbound depth input");
    invalid = fog;
    invalid.edges[invalid.edges.size() - 2].fromNodeId = 70;
    reject(invalid, "fog rejects depth from a different source owner");
    invalid = fog;
    invalid.edges.push_back({71, 1, 6, 0});
    reject(invalid, "fog rejects an extra direct Render-to-output edge");
    invalid = fog;
    invalid.ports.push_back({90, 3, 1, "out", "frame", "image", "rgba8", "sRGB"});
    reject(invalid, "fog rejects extra ports");
    invalid = fog;
    operation(invalid, 90).kind = "visual.depth.blur";
    *std::find(invalid.nodeKinds.begin(), invalid.nodeKinds.end(), "visual.depth.fog") = "visual.depth.blur";
    reject(invalid, "fog lowering cannot admit another depth effect kind");
    invalid = fog;
    operation(invalid, 90).payloadXml = "<NodeParams density=\"nan\"/>";
    reject(invalid, "fog rejects nonfinite authored parameters");

    // PolyphonicImportedModelArray has no material terminal. Its exact output
    // must count alongside the imported source, score source and instancer.
    auto notes = importedScenePlan();
    useRegisteredImportedStarterPorts(notes, 71);
    visualnoteinstancing::Mapping mapping;
    mapping.appearanceSource = visualnoteinstancing::AppearanceSource::StableNoteIdentity;
    addBefore(notes, 71, 90, "visual.score.note-collection", "control-eval");
    addBefore(notes, 71, 91, "visual.3d.note-instanced-mesh", "control-eval", visualnoteinstancing::encode(mapping));
    notes.ports.insert(notes.ports.end(), {
        {90, 0, 1, "out", "control", "noteCollection", "unspecified", "unspecified"},
        {91, 0, 1, "in", "control", "mesh", "unspecified", "unspecified"},
        {91, 1, 1, "in", "control", "noteCollection", "unspecified", "unspecified"},
        {91, 2, 1, "out", "control", "mesh", "unspecified", "unspecified"}
    });
    notes.edges.insert(notes.edges.end(), {{70, 0, 91, 0}, {90, 0, 91, 1}, {91, 2, 71, 3}});
    appendImportedStarterOutput(notes, 71, 1);
    {
        const bool conditionOk = accepted(notes) && compiled.noteInstanceMapping && compiled.importedSceneRender
        && !compiled.importedSceneRender->material;
        check(conditionOk, "authored polyphonic imported array admits its exact outer output without a material: " + error);
    }
    invalid = notes;
    invalid.edges.erase(invalid.edges.end() - 3);
    reject(invalid, "imported note array rejects an unbound score producer");
    invalid = notes;
    invalid.edges[2] = invalid.edges.front();
    reject(invalid, "duplicate scene edges cannot hide an unbound note source");
    invalid = notes;
    operation(invalid, 90).kind = "visual.score.future-collection";
    *std::find(invalid.nodeKinds.begin(), invalid.nodeKinds.end(), "visual.score.note-collection")
        = "visual.score.future-collection";
    reject(invalid, "imported instancer rejects a wrong-kind note source before dereferencing it");
    invalid = notes;
    addBefore(invalid, 6, 92, "visual.surface.constant.scalar", "control-eval");
    invalid.ports.push_back({92, 0, 1, "out", "control", "scalar", "unspecified", "unspecified"});
    reject(invalid, "no-material array rejects an extra unbound operation");

    // The current Note-Driven Particle and Light Rig uses all eight registered
    // particle ports; optional geometry and Points3D ports are unconnected.
    auto rig = importedScenePlan(true, true);
    useRegisteredImportedStarterPorts(rig, 71);
    auto rigRequest = renderRequest(true, true);
    HarmonicMIDI::grid::SceneLightRecord light;
    light.id.value = 86;
    light.kind = HarmonicMIDI::grid::SceneLightKind::Directional;
    rigRequest.light = light;
    rigRequest.noteLight = visualimportedscenerender::NoteLight {0, 2, 45};
    operation(rig, 71).payloadXml = visualimportedscenerender::encode(rigRequest);
    addBefore(rig, 71, 84, "visual.3d.transform", "control-eval");
    addBefore(rig, 71, 85, "visual.3d.light.directional", "control-eval");
    rig.ports.insert(rig.ports.end(), {
        {84, 0, 1, "out", "control", "transform3D", "unspecified", "unspecified"},
        {85, 0, 1, "in", "control", "transform3D", "unspecified", "unspecified"},
        {85, 1, 1, "out", "control", "light", "unspecified", "unspecified"}
    });
    rig.edges.insert(rig.edges.end(), {{84, 0, 85, 0}, {85, 1, 71, 6}});
    addBefore(rig, -1, 90, "visual.particles", "native-gpu",
        "<NodeParams motionMode=\"1\" count=\"256\" lifetime=\"3\" size=\"7\" speed=\"0.4\" "
        "drag=\"0.25\" attraction=\"0.3\" alpha=\"0.75\"/>");
    addBefore(rig, -1, 91, "video.blend", "native-gpu", "<NodeParams opacity=\"1\" mode=\"0\"/>");
    rig.ports.insert(rig.ports.end(), {
        {90, 0, 1, "in", "event", "unspecified", "unspecified", "unspecified"},
        {90, 1, 1, "out", "frame", "image", "rgba8", "sRGB"},
        {90, 2, 1, "in", "control", "points3D", "unspecified", "unspecified"},
        {90, 3, 1, "in", "control", "points3D", "unspecified", "unspecified"},
        {90, 4, 1, "in", "control", "geometry3D", "unspecified", "unspecified"},
        {90, 5, 1, "in", "control", "points3D", "unspecified", "unspecified"},
        {90, 6, 1, "in", "control", "geometry3D", "unspecified", "unspecified"},
        {90, 7, 1, "out", "control", "points3D", "unspecified", "unspecified"},
        {91, 0, 1, "in", "frame", "image", "rgba8", "sRGB"},
        {91, 1, 1, "out", "frame", "image", "rgba8", "sRGB"},
        {91, 2, 1, "in", "frame", "image", "rgba8", "sRGB"},
        {91, 3, 1, "in", "frame", "image", "rgba8", "sRGB"}
    });
    rig.edges.insert(rig.edges.end(), {{71, 1, 91, 0}, {90, 1, 91, 2}});
    appendImportedStarterOutput(rig, 91, 1);
    {
        const bool conditionOk = accepted(rig) && compiled.importedParticleOverlay && compiled.particles
        && compiled.importedSceneRender->material && compiled.importedSceneRender->deformation
        && compiled.importedSceneRender->noteLight && compiled.particleParameters.count == 256;
        check(conditionOk, "authored eight-port particle rig retains animation, surface material and note light: " + error);
    }
    for (int port : {2, 3, 4, 5, 6})
    {
        invalid = rig;
        auto& binding = *std::find_if(invalid.ports.begin(), invalid.ports.end(), [&](const auto& item)
            { return item.nodeId == 90 && item.port == port; });
        binding.dataType = binding.dataType == "points3D" ? "geometry3D" : "points3D";
        reject(invalid, "particle rig rejects a changed optional geometry kind");
    }
    invalid = rig;
    invalid.ports.push_back({90, 8, 1, "in", "control", "geometry3D", "unspecified", "unspecified"});
    reject(invalid, "particle rig rejects extra ports beyond the registered eight");
    invalid = rig;
    std::find_if(invalid.ports.begin(), invalid.ports.end(), [](const auto& item)
        { return item.nodeId == 90 && item.port == 7; })->direction = "in";
    reject(invalid, "particle rig rejects a Points3D port that is not an output");
    invalid = rig;
    invalid.edges.push_back({90, 7, 71, 6});
    reject(invalid, "particle rig rejects a connected Points3D output");
    invalid = rig;
    invalid.edges[invalid.edges.size() - 2].toPort = 3;
    reject(invalid, "particle rig rejects the layer input in place of the overlay input");
    invalid = rig;
    invalid.edges.push_back({70, 0, 90, 4});
    reject(invalid, "particle overlay cannot consume an unlowered mesh-barrier source");

    // HolographicTradingCard uses a spatial foil, separate camera/light
    // transforms, an unconnected imagery port and the looping vertex field.
    auto card = importedScenePlan(false, false, true);
    operation(card, 72).kind = "visual.material.diffractive-foil";
    *std::find(card.nodeKinds.begin(), card.nodeKinds.end(), "visual.material.diffraction-grating")
        = "visual.material.diffractive-foil";
    useRegisteredImportedStarterPorts(card, 71);
    auto cardRequest = renderRequest(false, false, true);
    cardRequest.diffractionMaterial = spatialDiffractionMaterialRequest();
    HarmonicMIDI::grid::SceneCameraRecord camera;
    camera.id.value = 75;
    camera.transform.translation.z = 0.16f;
    camera.verticalFovRadians = 0.65f;
    camera.nearPlane = 0.01f;
    camera.farPlane = 10;
    cardRequest.camera = camera;
    light.id.value = 77;
    cardRequest.light = light;
    addBefore(card, 72, 73, "visual.3d.transform", "control-eval");
    addBefore(card, 72, 74, "visual.3d.camera.perspective", "control-eval");
    addBefore(card, 72, 75, "visual.3d.transform", "control-eval");
    addBefore(card, 72, 76, "visual.3d.light.directional", "control-eval");
    card.ports.insert(card.ports.end(), {
        {73, 0, 1, "out", "control", "transform3D", "unspecified", "unspecified"},
        {74, 0, 1, "in", "control", "transform3D", "unspecified", "unspecified"},
        {74, 1, 1, "out", "control", "camera", "unspecified", "unspecified"},
        {75, 0, 1, "out", "control", "transform3D", "unspecified", "unspecified"},
        {76, 0, 1, "in", "control", "transform3D", "unspecified", "unspecified"},
        {76, 1, 1, "out", "control", "light", "unspecified", "unspecified"},
        {72, 1, 1, "in", "frame", "image", "rgba8", "sRGB"},
        {72, 2, 1, "in", "control", "field", "unspecified", "unspecified"}
    });
    card.edges.insert(card.edges.end(), {{73, 0, 74, 0}, {74, 1, 71, 5}, {75, 0, 76, 0}, {76, 1, 71, 6}});
    cardRequest.materialField = videohelper::tests::materialFieldFixture(72);
    cardRequest.materialField->target = 13;
    cardRequest.materialField->gain = 30;
    const auto field = materialfield::admit(*cardRequest.materialField, error);
    check(field.has_value(), "card's looping foil field is canonical: " + error);
    if (!field) return;
    const auto topology = materialfield::topology(*cardRequest.materialField, field->descriptor());
    for (const auto& [id, kind] : topology.nodes)
    {
        addBefore(card, 72, id, kind.c_str(), "control-eval");
        const int output = topology.outputPorts.at(id);
        for (int port = 0; port <= output; ++port)
            card.ports.push_back({id, port, 1, port == output ? "out" : "in", "control", "field", "unspecified", "unspecified"});
    }
    for (const auto& [from, output, to, input] : topology.edges) card.edges.push_back({from, output, to, input});
    operation(card, 71).payloadXml = visualimportedscenerender::encode(cardRequest);
    appendImportedStarterOutput(card, 71, 1);
    {
        const bool conditionOk = accepted(card) && compiled.importedSceneRender && compiled.importedSceneRender->diffractionMaterial
        && compiled.importedSceneRender->diffractionMaterial->spatialFoil && compiled.importedSceneRender->materialField;
        check(conditionOk, "authored holographic card reaches strict execution with its exact foil field and outer output: " + error);
    }
    invalid = card;
    operation(invalid, 72).kind = "visual.material.diffraction-grating";
    *std::find(invalid.nodeKinds.begin(), invalid.nodeKinds.end(), "visual.material.diffractive-foil")
        = "visual.material.diffraction-grating";
    reject(invalid, "a grating terminal cannot own the spatial foil request");
    auto analyticFoil = card;
    auto analyticRequest = cardRequest;
    analyticRequest.diffractionMaterial = diffractionMaterialRequest();
    analyticRequest.materialField->target = 4;
    operation(analyticFoil, 71).payloadXml = visualimportedscenerender::encode(analyticRequest);
    {
        const bool conditionOk = accepted(analyticFoil);
        check(conditionOk, "foil's authored analytic mode retains the physical grating route: " + error);
    }
    auto grating = analyticFoil;
    operation(grating, 72).kind = "visual.material.diffraction-grating";
    *std::find(grating.nodeKinds.begin(), grating.nodeKinds.end(), "visual.material.diffractive-foil")
        = "visual.material.diffraction-grating";
    {
        const bool conditionOk = accepted(grating);
        check(conditionOk, "physical grating snapshot admits only its exact material topology: " + error);
    }
    invalid = card;
    operation(invalid, 72).kind = "visual.material.diffractive-foil.future";
    *std::find(invalid.nodeKinds.begin(), invalid.nodeKinds.end(), "visual.material.diffractive-foil")
        = "visual.material.diffractive-foil.future";
    check(!accepted(invalid), "unknown foil kinds fail snapshot admission");
    invalid = card;
    invalid.edges.erase(std::find_if(invalid.edges.begin(), invalid.edges.end(),
        [](const auto& edge) { return edge.fromNodeId == 72 && edge.toNodeId == 71; }));
    reject(invalid, "foil must own the render material input");
    invalid = card;
    addBefore(invalid, 71, 92, "visual.material.diffractive-foil", "control-eval");
    invalid.ports.push_back({92, 0, 1, "out", "control", "material", "unspecified", "unspecified"});
    reject(invalid, "card rejects an extra unbound foil terminal");
    invalid = card;
    invalid.edges.erase(std::find_if(invalid.edges.begin(), invalid.edges.end(),
        [](const auto& edge) { return edge.toNodeId == 72 && edge.toPort == 2; }));
    reject(invalid, "foil field request cannot survive an unbound field input");

    // AnimatedBlenderModel starter: animation, Render 3D light and a ten-constant
    // Surface Material. Its registered terminal also plans the unconnected vertex,
    // texture-frame and scalar Dispersion inputs, so it has 14 ports.
    auto blender = importedScenePlan(true, true);
    useRegisteredImportedStarterPorts(blender, 71);
    const int blenderTerminal = std::find_if(blender.operations.begin(), blender.operations.end(),
        [](const auto& item) { return item.kind == "visual.surface.material"; })->nodeId;
    auto blenderRequest = renderRequest(true, true);
    HarmonicMIDI::grid::SceneLightRecord blenderLight;
    blenderLight.id.value = 86;
    blenderLight.kind = HarmonicMIDI::grid::SceneLightKind::Directional;
    blenderRequest.light = blenderLight;
    operation(blender, 71).payloadXml = visualimportedscenerender::encode(blenderRequest);
    addBefore(blender, 71, 84, "visual.3d.transform", "control-eval");
    addBefore(blender, 71, 85, "visual.3d.light.directional", "control-eval");
    blender.ports.insert(blender.ports.end(), {
        {84, 0, 1, "out", "control", "transform3D", "unspecified", "unspecified"},
        {85, 0, 1, "in", "control", "transform3D", "unspecified", "unspecified"},
        {85, 1, 1, "out", "control", "light", "unspecified", "unspecified"}
    });
    blender.edges.insert(blender.edges.end(), {{84, 0, 85, 0}, {85, 1, 71, 6}});
    appendImportedStarterOutput(blender, 71, 1);
    {
        const bool conditionOk = accepted(blender) && compiled.importedSceneRender
        && compiled.importedSceneRender->deformation && compiled.importedSceneRender->material;
        check(conditionOk, "authored animated blender starter admits its registered 14-port material terminal: " + error);
    }

    // Dispersion stays exact: its schema must be scalar and it must not be wired.
    auto wrongDispersion = blender;
    auto& dispersionPort = *std::find_if(wrongDispersion.ports.begin(), wrongDispersion.ports.end(),
        [&](const auto& item) { return item.nodeId == blenderTerminal && item.port == 13; });
    dispersionPort.dataType = "vec3";
    error.clear();
    {
        const bool conditionOk = !videowire::compileVisualLayerExecution(wrongDispersion, compiled, error)
        && error.find("failed=material") != std::string::npos
        && error.find("materialPorts=14") != std::string::npos;
        check(conditionOk, "a non-scalar Dispersion input is refused and the diagnostic names the material terminal: " + error);
    }
    auto wiredDispersion = blender;
    wiredDispersion.edges.push_back({73, 0, blenderTerminal, 13});
    reject(wiredDispersion, "an unsupported wired Dispersion input fails exact edge admission");
}
videowire::CompiledVisualLayerPlan importedMaterialFramePlan(bool diffraction = false)
{
    auto plan = importedScenePlan(!diffraction, false, diffraction);
    visualimportedscenerender::Request request;
    visualimportedscenerender::decode(plan.operations.back().payloadXml, request);
    if (diffraction)
    {
        request.diffractionMaterial->version = diffractionmaterialbinding::kGraphFrameWireVersion;
        request.diffractionMaterial->graphFrame = surfacematerialbinding::TextureSlotBinding::GraphFrameEndpoint {91, 0};
    }
    else
    {
        request.material->version = surfacematerialbinding::kGraphFrameWireVersion;
        auto& texture = request.material->binding.textures.front();
        texture.source = surfacematerialbinding::TextureSourceKind::GraphFrame;
        texture.videoResource = {};
        texture.graphFrame = surfacematerialbinding::TextureSlotBinding::GraphFrameEndpoint { 91, 0 };
    }
    plan.operations.back().payloadXml = visualimportedscenerender::encode(request);
    // Deliberately serialize the prerequisite after Render 3D. The common
    // preparation seam, not incidental document order, must run it first.
    plan.nodeKinds.push_back("video.source");
    plan.nodeIds.push_back(91);
    plan.operations.push_back({ 91, "video.source", "source-decode", "" });
    plan.ports.push_back({ 91, 0, 1, "out", "frame", "image", "rgba8", "sRGB" });
    if (!diffraction)
        plan.ports.push_back({ 82, 11, 3, "in", "control", "vec3", "unspecified", "unspecified" });
    plan.ports.push_back({ diffraction ? 72 : 82, diffraction ? 1 : 12, 1, "in", "frame", "image", "rgba8", "sRGB" });
    plan.edges.push_back({ 91, 0, diffraction ? 72 : 82, diffraction ? 1 : 12 });
    return plan;
}

void testMaterialFramePrerequisite(bool diffraction = false, bool generated = false)
{
    using namespace videohelper::importedscene;
    auto plan = importedMaterialFramePlan(diffraction);
    if (generated)
    {
        std::string generatedError;
        const auto lighting = geometryCompositionFixture(generatedError);
        const auto scene = lighting ? videohelper::tests::generatedSurfaceScene(*lighting, generatedError) : std::nullopt;
        check(scene.has_value(), "generated Frame schedule retains a real Geometry Core mesh");
        if (!scene) return;
        visualimportedscenerender::Request request;
        visualimportedscenerender::decode(plan.operations[plan.operations.size() - 2].payloadXml, request);
        request.sourceStableId = scene->id.value;
        request.renderStableId = 12;
        request.asset = {};
        request.sceneIndex.reset();
        request.structuralRevision = request.evaluationRevision = plan.structuralRevision;
        request.sceneSnapshot = std::make_shared<const HarmonicMIDI::grid::Visual3DScene>(*scene);
        auto& material = *request.material;
        material.scene = scene->id;
        material.sceneSnapshot = request.sceneSnapshot;
        material.sceneRevision = material.structuralRevision = material.evaluationRevision
            = material.programRevision = material.binding.surfaceMaterialRevision = plan.structuralRevision;
        material.binding.object = scene->objects[0].id;
        const int source = static_cast<int>(request.sourceStableId - 1);
        plan.nodeIds = {source, 91, 11};
        plan.nodeKinds = {videowire::geometry::kRetainedSceneOperation, "video.source", "visual.3d.render"};
        plan.operations = {{source, plan.nodeKinds[0], "control-eval", ""},
            {91, "video.source", "source-decode", ""},
            {11, plan.nodeKinds[2], "native-gpu", visualimportedscenerender::encode(request)}};
        plan.edges = {{source, 0, 11, 0}, {91, 0, 11, 12}};
        plan.ports = {{source, 0, 1, "out", "control", "scene3D", "unspecified", "unspecified"},
            {11, 0, 1, "in", "control", "scene3D", "unspecified", "unspecified"},
            {11, 1, 1, "out", "frame", "image", "rgba8", "sRGB"},
            {91, 0, 1, "out", "frame", "image", "rgba8", "sRGB"},
            {11, 12, 1, "in", "frame", "image", "rgba8", "sRGB"}};
        check(!plan.operations[2].payloadXml.empty(), "generated Frame material serializes the existing Surface binding");
    }
    if (diffraction)
    {
        visualimportedscenerender::Request request;
        check(visualimportedscenerender::decode(plan.operations[plan.operations.size() - 2].payloadXml, request),
              "Frame diffraction v5 decodes with the legacy canonical lighting plan");
        if (request.diffractionMaterial)
            for (int fault = 0; fault < 4; ++fault)
            {
                auto invalid = *request.diffractionMaterial;
                if (fault == 0) invalid.version = diffractionmaterialbinding::kWireVersion;
                if (fault == 1) invalid.graphFrame.reset();
                if (fault == 2) invalid.graphFrame->node = -1;
                if (fault == 3) invalid.graphFrame->port = 1;
                std::string admissionError;
                check(!visualimportedscenerender::detail::validDiffractionMaterial(invalid)
                          && !diffractionmaterialbinding::admit(invalid, admissionError),
                      "diffraction Frame identity and version must agree at wire and native admission");
            }
    }
    videowire::VisualLayerExecution compiled;
    std::string error;
    const auto frameSource = std::find_if(plan.operations.begin(), plan.operations.end(),
        [](const auto& operation) { return operation.nodeId == 91; });
    check(frameSource != plan.operations.end(), "Frame prerequisite fixture contains its exact source");
    if (frameSource == plan.operations.end()) return;
    const auto frameSourceIndex = static_cast<std::size_t>(frameSource - plan.operations.begin());
    {
        const bool conditionOk = videowire::compileVisualLayerExecution(plan, compiled, error)
              && compiled.materialFrameSource && compiled.materialFrameSource->nodeId == 91
              && compiled.materialFrameSource->operationIndex == frameSourceIndex
              && compiled.materialFrameSource->nodeIndex < plan.nodeIds.size()
              && plan.nodeIds[compiled.materialFrameSource->nodeIndex] == 91;
        check(conditionOk, "Frame dependency identifies its exact authored source for Render 3D preparation: " + error);
    }
    for (int fault = 0; fault < 6; ++fault)
    {
        auto malformed = plan;
        if (fault == 0) malformed.edges.back().fromNodeId = 70;
        if (fault == 1) malformed.edges.push_back(malformed.edges.back());
        if (fault == 2) malformed.operations[frameSourceIndex].backendCapability = "native-gpu";
        if (fault == 3)
            for (auto& port : malformed.ports)
                if (port.nodeId == 91) port.colorSpace = "linear";
        if (fault == 4) malformed.ports.pop_back();
        if (fault == 5) malformed.operations[frameSourceIndex].payloadXml = "unadmitted source override";
        {
            const bool conditionOk = !videowire::compileVisualLayerExecution(malformed, compiled, error);
            check(conditionOk, "Frame prerequisite rejects topology, descriptor, or source-authority mismatch " + std::to_string(fault));
        }
    }

    FakeExecution execution;
    FakeLayer layer;
    std::vector<FakeReceipt> owners;
    VisualImportedScenePlanCache cache;
    const auto prepare = [&](NativeImportedSceneRenderUse use, const MaterialFrameResolver* resolver)
    {
        return prepareVisualImportedSceneLayerAtTime({plan}, 7, 640, 360, 1.25, 24.0,
            use, &execution, &cache, layer, owners, error, nullptr, { 12, 34 }, resolver);
    };
    check(prepare(NativeImportedSceneRenderUse::Preview, nullptr) == VisualImportedScenePreparation::rejected
              && execution.previewCalls == 0 && owners.empty(),
          "missing native Frame resolver rejects before scene submission or imported-texture fallback");

    if (!diffraction)
    {
        auto savedPlan = plan;
        auto& savedSource = savedPlan.operations[frameSourceIndex];
        savedSource.kind = savedPlan.nodeKinds[std::find(savedPlan.nodeIds.begin(), savedPlan.nodeIds.end(), 91)
            - savedPlan.nodeIds.begin()] = "video.layer.source";
        savedSource.payloadXml = materialframesource::encode(111);
        const auto savedRenderPayload = std::find_if(savedPlan.operations.begin(), savedPlan.operations.end(),
            [](const auto& operation) { return operation.kind == "visual.3d.render"; })->payloadXml;
        videowire::VisualLayerExecution savedCompiled;
        videosourcecolor::Policy savedPolicy;
        int savedClip = 0;
        check(videowire::compileVisualLayerExecution(savedPlan, savedCompiled, error)
                  && savedCompiled.geometryFrameSchedule
                  && savedCompiled.geometryFrameSchedule->outputs == std::vector<videowire::materialframe::Endpoint>{{91, 0}}
                  && videowire::materialframe::sourcePolicy(savedPlan, {91, 0}, savedPolicy, error)
                  && videowire::materialframe::sourceClip(savedPlan, {91, 0}, savedClip, error) && savedClip == 111
                  && !videowire::isGeometryCoreRenderPlan({savedPlan}, 7),
              "ordinary and retained imported saved Layer Source keeps prepared admission and exact source identity");
        for (const auto use : {NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export})
        {
            execution = {}; owners.clear();
            int evaluations = 0;
            MaterialFrameResolver currentClipFallback = [&](const auto& evaluation, auto& receipt, std::string&) {
                ++evaluations;
                receipt = {evaluation, std::make_shared<FakeFrame>()};
                return true;
            };
            check(prepareVisualImportedSceneLayerAtTime({savedPlan}, 7, 640, 360, 1.25, 24.0,
                      use, &execution, &cache, layer, owners, error, nullptr, {12, 34}, &currentClipFallback)
                      == VisualImportedScenePreparation::rejected
                      && error == "Imported Surface Layer Source requires its saved clip and frame-time resolver"
                      && evaluations == 0 && execution.previewCalls == 0 && execution.exportCalls == 0 && owners.empty(),
                  "ordinary and retained imported saved Layer Source refuses before decoder callback or native scene draw");
            check(savedPlan.operations[frameSourceIndex].payloadXml == materialframesource::encode(111)
                      && std::find_if(savedPlan.operations.begin(), savedPlan.operations.end(),
                          [](const auto& operation) { return operation.kind == "visual.3d.render"; })->payloadXml == savedRenderPayload,
                  "saved Layer Source refusal preserves the authored endpoint and required material payload");
        }
    }

    for (const auto* backend : { "opengl", "metal" })
        for (const auto use : { NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export })
        {
            execution = {};
            execution.nextFrame->api = backend;
            auto input = std::make_shared<FakeFrame>();
            input->api = backend;
            input->frameWidth = input->descriptorWidth = 160;
            input->frameHeight = 90;
            const std::weak_ptr<FakeFrame> retained = input;
            int evaluations = 0;
            MaterialFrameResolver resolver = [&](const MaterialFrameEvaluation& evaluation,
                                                 MaterialFrameReceipt& receipt, std::string&)
            {
                ++evaluations;
                check(execution.previewCalls == 0 && execution.exportCalls == 0
                          && evaluation.endpoint.node == 91 && evaluation.endpoint.port == 0
                          && evaluation.frame.frame == 30 && evaluation.frame.rateNumerator == 24
                          && evaluation.frame.rateDenominator == 1 && evaluation.clipId == 7
                          && evaluation.structuralRevision == plan.structuralRevision
                          && evaluation.evaluationRevision == (generated ? plan.structuralRevision : 0)
                          && evaluation.projectGeneration == 12 && evaluation.helperGeneration == 34
                          && evaluation.use == use,
                      "Frame is evaluated at the exact publication/time before preview or export draw");
                receipt = { evaluation, input };
                return true;
            };
            {
                const bool conditionOk = prepare(use, &resolver) == VisualImportedScenePreparation::rendered
                      && evaluations == 1 && owners.size() == 1
                      && execution.lastRequest.runtimeInputs.materialFrameTexture == input
                      && owners.front().materialFrameTexture == input;
                check(conditionOk, std::string(backend) + " shared prerequisite seam carries the owner into submission and receipt: " + error);
            }
            input.reset();
            execution.lastRequest = {};
            check(!retained.expired(), "output receipt keeps the Frame dependency alive after evaluator/request release");
            owners.clear();
            check(retained.expired(), "releasing the output receipt releases its final Frame lease");
        }

    for (const auto* backend : {"opengl","metal"})
        for (const auto use : {NativeImportedSceneRenderUse::Preview,NativeImportedSceneRenderUse::Export})
        {
            execution = {}; owners.clear();
            execution.nextFrame->api = backend; execution.nextFrame->linearColor = true;
            auto linearInput = std::make_shared<FakeFrame>();
            linearInput->api = backend; linearInput->linearColor = true;
            MaterialFrameResolver linearResolver = [&](const auto& evaluation,auto& receipt,std::string&) {
                check(evaluation.frame.frame == 30 && evaluation.frame.rateNumerator == 24 && evaluation.use == use,
                    "direct linear material keeps the exact preview/export evaluation");
                receipt = {evaluation,linearInput}; return true;
            };
            {
                const bool conditionOk = prepare(use,&linearResolver) == VisualImportedScenePreparation::rendered
                && execution.lastRequest.runtimeInputs.linearColor
                && execution.lastRequest.runtimeInputs.materialFrameTexture == linearInput;
                check(conditionOk, "declared direct linear material selects scene-linear rendering in preview/export: " + error);
            }
            const std::weak_ptr<FakeFrame> retained = linearInput;
            linearInput.reset(); execution.lastRequest = {};
            check(!retained.expired(),"scene receipt retains linear material owner after evaluator advancement");
            owners.clear(); check(retained.expired(),"linear material releases after its last scene receipt");
        }

    execution = {};
    auto input = std::make_shared<FakeFrame>();
    for (int fault = 0; fault < 12; ++fault)
    {
        MaterialFrameResolver resolver = [&](const MaterialFrameEvaluation& evaluation,
                                             MaterialFrameReceipt& receipt, std::string&)
        {
            receipt = { evaluation, input };
            if (fault == 0) ++receipt.evaluation.endpoint.node;
            if (fault == 1) ++receipt.evaluation.endpoint.port;
            if (fault == 2) ++receipt.evaluation.frame.frame;
            if (fault == 3) ++receipt.evaluation.frame.rateNumerator;
            if (fault == 4) ++receipt.evaluation.clipId;
            if (fault == 5) ++receipt.evaluation.structuralRevision;
            if (fault == 6) ++receipt.evaluation.evaluationRevision;
            if (fault == 7) ++receipt.evaluation.projectGeneration;
            if (fault == 8) ++receipt.evaluation.helperGeneration;
            if (fault == 9) receipt.evaluation.use = NativeImportedSceneRenderUse::Export;
            if (fault == 10) receipt.nativeFrame.reset();
            if (fault == 11) input->descriptorSampleCount = 2;
            return true;
        };
        {
            const bool conditionOk = prepare(NativeImportedSceneRenderUse::Preview, &resolver) == VisualImportedScenePreparation::rejected
                  && execution.previewCalls == 0 && owners.empty();
            check(conditionOk, "stale/unowned Frame prerequisite cannot submit or retain a scene frame " + std::to_string(fault));
        }
    }
    input->descriptorSampleCount = 1;
    MaterialFrameResolver resolver = [&](const MaterialFrameEvaluation& evaluation,
                                         MaterialFrameReceipt& receipt, std::string&)
    { receipt = { evaluation, input }; return true; };
    input->contextIdentity = 999;
    check(prepare(NativeImportedSceneRenderUse::Preview, &resolver) == VisualImportedScenePreparation::rejected
              && owners.empty(), "different native contexts cannot publish a material Frame scene result");
    execution = {};
    input->contextIdentity = 77;
    input->api = "metal";
    check(prepare(NativeImportedSceneRenderUse::Preview, &resolver) == VisualImportedScenePreparation::rejected
              && owners.empty(), "different native backends cannot publish a material Frame scene result");

    for (const auto use : { NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export })
    {
        execution = {};
        owners.clear();
        std::vector<std::int64_t> sourceFrames;
        MaterialFrameResolver seekResolver = [&](const MaterialFrameEvaluation& evaluation,
                                                MaterialFrameReceipt& receipt, std::string&)
        {
            sourceFrames.push_back(evaluation.frame.frame);
            auto sought = std::make_shared<FakeFrame>();
            sought->image = sought->view = static_cast<std::uintptr_t>(1000 + evaluation.frame.frame);
            receipt = { evaluation, std::move(sought) };
            return true;
        };
        // Forward playback, backwards seek, then a loop back to the same time.
        for (const double seconds : { 0.5, 1.5, 0.25, 0.5 })
        {
            owners.clear();
            check(prepareVisualImportedSceneLayerAtTime({plan}, 7, 640, 360, seconds, 24.0,
                      use, &execution, &cache, layer, owners, error, nullptr, {12, 34}, &seekResolver)
                      == VisualImportedScenePreparation::rendered
                      && execution.lastRequest.runtimeInputs.materialFrameTexture->colorImageHandle()
                          == static_cast<std::uintptr_t>(1000 + std::llround(seconds * 24)),
                  "preview/export seek and loop resolve the exact source frame before each scene draw");
        }
        check(sourceFrames == std::vector<std::int64_t> {12, 36, 6, 12},
              "preview and offline export have identical stateless Frame evaluation order");
        const auto draws = execution.previewCalls + execution.exportCalls;
        owners.clear();
        MaterialFrameResolver missing = [](const MaterialFrameEvaluation&, MaterialFrameReceipt&, std::string& error)
        { error = "video source missing"; return false; };
        check(prepare(use, &missing) == VisualImportedScenePreparation::rejected
                  && execution.previewCalls + execution.exportCalls == draws && owners.empty(),
              "a missing decoder frame after a successful seek cannot publish the previous video texture");
    }

    plan = importedScenePlan(!diffraction, false, diffraction);
    execution = {};
    MaterialFrameResolver forbidden = [](const MaterialFrameEvaluation&, MaterialFrameReceipt&, std::string&)
    { check(false, "imported texture fallback must not invoke a Frame resolver"); return false; };
    check(prepare(NativeImportedSceneRenderUse::Preview, &forbidden) == VisualImportedScenePreparation::rendered
              && !execution.lastRequest.runtimeInputs.materialFrameTexture,
          "legacy imported slot-zero texture remains unchanged without a Frame edge");
}

videowire::CompiledVisualLayerPlan importedVertexPlan()
{
    auto plan = importedScenePlan(true, true);
    auto render = std::move(plan.operations.back());
    plan.operations.pop_back();
    plan.nodeIds.pop_back();
    plan.nodeKinds.pop_back();
    auto request = renderRequest(true, true);
    request.material->vertexModifier = videohelper::test::audioNormalDisplacement();
    const auto& vertex = *request.material->vertexModifier;
    constexpr int material = 83;
    plan.ports.push_back({material, 11, 3, "in", "control", "vec3", "unspecified", "unspecified"});
    for (const auto& record : vertex.records)
    {
        const auto id = static_cast<int>(record.stableId - 1u);
        const auto kind = videowire::vertexModifierGraphKindToken(record.operation);
        plan.operations.push_back({id, kind, "control-eval", ""});
        plan.nodeIds.push_back(id);
        plan.nodeKinds.push_back(kind);
        const auto schema = videowire::vertexModifierOperationSchema(record.operation);
        const auto addPort = [&](int port, const char* direction, videowire::VertexModifierValueType type)
        {
            const auto scalar = type == videowire::VertexModifierValueType::scalar;
            plan.ports.push_back({id, port, scalar ? 1 : 3, direction, "control",
                                 scalar ? "scalar" : "vec3", "unspecified", "unspecified"});
        };
        for (std::uint8_t input = 0; input < schema.inputCount; ++input)
        {
            addPort(input, "in", schema.inputTypes[input]);
            const auto& source = vertex.records[record.inputs[input] - 1u];
            plan.edges.push_back({static_cast<int>(source.stableId - 1u), source.inputCount, id, input});
        }
        addPort(schema.inputCount, "out", schema.resultType);
    }
    plan.edges.push_back({6, 2, material, 11});
    render.payloadXml = visualimportedscenerender::encode(request);
    plan.nodeIds.push_back(render.nodeId);
    plan.nodeKinds.push_back(render.kind);
    plan.operations.push_back(std::move(render));
    return plan;
}
} // namespace

void testDiffractionRuntimeParameters()
{
    using namespace videohelper::importedscene;
    const auto plan = importedScenePlan(false, false, true);
    const std::vector<videowire::CompiledVisualLayerPlan> plans {plan};
    std::map<std::string, double> parameters;
    seedImportedSceneRuntimeParameters(plans, 7, parameters);
    check(parameters.count("visual73/grooveSpacingNanometres") == 1
              && parameters.count("visual73/rmsSlope") == 1
              && parameters.count("visual73/emissionGain") == 0,
          "the exact diffraction terminal seeds physical scalar targets, not a false emission channel");
    const auto originalSpacing = parameters.at("visual73/grooveSpacingNanometres");
    visualimportedscenerender::Request authored;
    check(visualimportedscenerender::decode(plan.operations.back().payloadXml, authored),
          "runtime parameter tests retain the canonical authored material request");
    if (authored.diffractionMaterial)
    {
        using namespace diffractionmaterialbinding;
        const auto original = *authored.diffractionMaterial;
        auto evaluated = original;
        std::string diagnostic;
        check(applyRuntimeParameters(original, {{"grooveDepthNanometres", 90}, {"rmsSlope", 0.2},
                {"grooveSpacingNanometres", 1300}, {"directionAngleDegrees", 45}}, evaluated, diagnostic)
                  && diffractionmaterial::admit(evaluated.material, diagnostic).has_value()
                  && evaluated.material.geometry.grooveSpacingNanometres == 1300
                  && evaluated.scene == original.scene && evaluated.materialRevision == original.materialRevision,
              "physical scalar modulation preserves material identity and re-admits valid values");
        for (const auto& invalid : std::vector<RuntimeParameters> {{{"profile", 1}}, {{"lattice", 1}},
                {{"coating", 1}}, {{"grooveSpacingNanometres", 0}}, {{"maximumOrder", 0}},
                {{"coatingThicknessNanometres", 200}}, {{"not-a-parameter", 1}}})
            check(!applyRuntimeParameters(original, invalid, evaluated, diagnostic),
                  "runtime controls reject unadmitted modes, inactive coating and unsupported destinations");
        auto blazed = original;
        blazed.material.microstructure.profile = diffractionmaterial::GrooveProfile::BlazedSawtooth;
        check(applyRuntimeParameters(blazed, {{"grooveSpacingNanometres", 1400}, {"grooveDepthNanometres", 100}},
                    evaluated, diagnostic) && diffractionmaterial::admit(evaluated.material, diagnostic).has_value(),
              "blazed scalar modulation recomputes its constrained blaze angle");
        blazed.material.geometry.lattice = diffractionmaterial::GratingLattice::CrossedTwoDimensional;
        blazed.material.geometry.secondaryDirectionUv = {0, 1};
        blazed.material.geometry.secondaryGrooveSpacingNanometres = 1000;
        check(applyRuntimeParameters(blazed, {{"grooveSpacingNanometres", 1400}}, evaluated, diagnostic)
                  && !diffractionmaterial::admit(evaluated.material, diagnostic),
              "unequal crossed blazed periods cannot pass physical admission");
        check(applyRuntimeParameters(blazed, {{"grooveSpacingNanometres", 1400},
                {"secondaryGrooveSpacingNanometres", 1400}}, evaluated, diagnostic)
                  && diffractionmaterial::admit(evaluated.material, diagnostic).has_value(),
              "compatible crossed blazed periods remain editable together");
        auto coated = original;
        coated.material.coating = {diffractionmaterial::CoatingModel::IncoherentDielectric, 1000, {1.5f, 0.001f}};
        check(applyRuntimeParameters(coated, {{"coatingThicknessNanometres", 200},
                {"coatingRefractiveIndex", 1.7}, {"coatingExtinctionCoefficient", 0.01}}, evaluated, diagnostic)
                  && diffractionmaterial::admit(evaluated.material, diagnostic).has_value(),
              "the active coating's optical parameters remain physical runtime values");
    }
    FakeExecution execution;
    FakeLayer layer;
    std::vector<FakeReceipt> owners;
    std::string error;
    for (const auto use : {NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export})
        for (const double seconds : {0.0, 1.0, 0.5, 0.0})
        {
            parameters["visual73/grooveSpacingNanometres"] = originalSpacing + 100 * seconds;
            check(prepareVisualImportedSceneLayerAtTime(plans, 7, 640, 360, seconds, 24,
                    use, &execution, layer, owners, error, &parameters)
                      == VisualImportedScenePreparation::rendered,
                  "timeline/score/audio parameter values reach diffraction at forward, seek and loop samples");
            const auto& request = execution.lastRequest;
            check(request.frame.frame == static_cast<std::int64_t>(seconds * 24)
                      && request.diffractionMaterial->material.geometry.grooveSpacingNanometres == originalSpacing,
                  "modulation keeps the published material immutable and carries the rational frame identity");
            if (seconds == 0)
                check(request.runtimeInputs.diffractionParameters.empty(), "returning to the authored value clears frame-local modulation");
            else
                check(request.runtimeInputs.diffractionParameters.at("grooveSpacingNanometres") == originalSpacing + 100 * seconds,
                      "the frame carries only the exact changed material scalar");
        }
    for (const auto& key : {"visual73/not-a-parameter", "visual73/rmsSlope"})
    {
        auto invalid = parameters;
        invalid[key] = std::numeric_limits<double>::quiet_NaN();
        check(prepareVisualImportedSceneLayerAtTime(plans, 7, 640, 360, 0, 24,
                NativeImportedSceneRenderUse::Preview, &execution, layer, owners, error, &invalid)
                  == VisualImportedScenePreparation::rejected,
              "nonfinite and unknown modulation cannot reuse a previous diffraction frame");
    }
}

namespace
{
// composedScenePlan() with material records named by their PBR Material nodes,
// as the producer lowers them: node 1 owns material 2 and node 4 owns material 5.
videowire::CompiledVisualLayerPlan pbrComposedScenePlan()
{
    auto plan = composedScenePlan();
    visualimportedscenerender::Request request;
    check(visualimportedscenerender::decode(plan.operations.back().payloadXml, request) && request.sceneSnapshot,
          "PBR modulation fixture decodes its composed request");
    auto scene = std::make_shared<HarmonicMIDI::grid::Visual3DScene>(*request.sceneSnapshot);
    for (std::size_t index = 0; index < 2; ++index)
    {
        scene->materials[index].id.value = index == 0 ? 2u : 5u;
        scene->objects[index].material = scene->materials[index].id;
    }
    scene->materials[0].baseColor = { 0.5f, 0.25f, 0.125f };
    scene->materials[0].roughness = 0.5f;
    request.sceneSnapshot = std::move(scene);
    plan.operations.back().payloadXml = visualimportedscenerender::encode(request);
    return plan;
}
} // namespace

// The lights the backend reader sees for one frame, through the same shared
// selection and override helpers the fixture readers call.
std::vector<HarmonicMIDI::grid::SceneLightRecord> readFrameLights(
    const HarmonicMIDI::grid::Visual3DScene& scene, const arbitgpu::NativeImportedSceneRuntimeInputs& runtime)
{
    std::vector<HarmonicMIDI::grid::SceneLightRecord> lights;
    if (runtime.lightOverride)
    {
        lights.push_back(*runtime.lightOverride);
        return lights;
    }
    const auto count = arbitgpu::effectiveNativeLightCount(scene, runtime);
    for (std::size_t i = 0; i < count; ++i)
        lights.push_back(arbitgpu::effectiveNativeLightRecord(scene, runtime, i));
    return lights;
}

void testLightOverrideApplication()
{
    using namespace videohelper::importedscene;
    using namespace HarmonicMIDI::grid;
    // A point light: only the recoloured and moved fields are flagged, and the override applies exactly.
    SceneLightRecord point;
    point.id.value = 91;
    point.kind = SceneLightKind::Point;
    point.transform.translation = { 1.0f, 2.0f, 3.0f };
    point.color = { 0.2f, 0.4f, 0.6f };
    point.intensity = 2.0f;
    const std::map<std::string, double> pointValues { { "visual91/positionY", 0.5 }, { "visual91/red", 1.0 } };
    arbitgpu::NativeImportedLightOverride pointOverride;
    check(directionalOrPointOverride(pointValues, point, 0, pointOverride),
          "a finite point light override builds");
    check(pointOverride.hasColor && pointOverride.hasPositionOffset && !pointOverride.hasIntensity
              && !pointOverride.hasRotationOffset && pointOverride.positionOffset.y == 0.5f,
          "unchanged point fields are not flagged on the override");
    auto pointApplied = point;
    arbitgpu::applyNativeImportedLightOverride(pointOverride, pointApplied);
    check(pointApplied.transform.translation.x == 1.0f && pointApplied.transform.translation.y == 2.5f
              && pointApplied.color.x == 1.0f && pointApplied.color.y == 0.4f && pointApplied.intensity == 2.0f,
          "the point override applies additively and absolutely to the sampled record");

    // A directional light: the rotation override reproduces the direct evaluation.
    SceneLightRecord directional;
    directional.id.value = 92;
    directional.kind = SceneLightKind::Directional;
    const std::map<std::string, double> turned { { "visual92/rotationY", 90.0 } };
    arbitgpu::NativeImportedLightOverride turnedOverride;
    check(directionalOrPointOverride(turned, directional, 3, turnedOverride) && turnedOverride.index == 3
              && turnedOverride.hasRotationOffset && !turnedOverride.hasColor && !turnedOverride.hasIntensity,
          "a directional rotation override carries only the rotation");
    auto expected = directional;
    check(evaluateDirectionalOrPointLight(turned, expected), "the direct directional evaluation succeeds");
    auto viaOverride = directional;
    arbitgpu::applyNativeImportedLightOverride(turnedOverride, viaOverride);
    check(std::abs(viaOverride.transform.rotation.x - expected.transform.rotation.x) < 1e-6f
              && std::abs(viaOverride.transform.rotation.y - expected.transform.rotation.y) < 1e-6f
              && std::abs(viaOverride.transform.rotation.z - expected.transform.rotation.z) < 1e-6f
              && std::abs(viaOverride.transform.rotation.w - expected.transform.rotation.w) < 1e-6f,
          "the rotation override applied to the sampled light matches the direct evaluation");

    // An authored light yields no override, and a non-finite control is refused.
    arbitgpu::NativeImportedLightOverride none;
    check(directionalOrPointOverride({}, directional, 0, none) && !none.any(),
          "an authored light yields no override");
    const std::map<std::string, double> nonfinite { { "visual91/intensity", std::nan("") } };
    check(!directionalOrPointOverride(nonfinite, point, 0, none),
          "a non-finite override control is refused");
}

void testLightOverrideIdentityAndAcceptance()
{
    using namespace HarmonicMIDI::grid;
    // A changed override is never reused, including a changed field that its flag does not name.
    arbitgpu::NativeImportedLightOverride a;
    a.index = 0;
    a.hasIntensity = true;
    a.intensity = 2.0f;
    auto b = a;
    check(arbitgpu::sameNativeImportedLightOverride(a, b), "identical overrides compare equal");
    b.intensity = 3.0f;
    check(!arbitgpu::sameNativeImportedLightOverride(a, b), "a changed intensity is not reused");
    b = a;
    b.positionOffset.x = 1.0f;
    check(!arbitgpu::sameNativeImportedLightOverride(a, b), "a changed unflagged field is not reused");
    check(!arbitgpu::sameNativeImportedLightOverrides({ a }, { a, b }) && arbitgpu::sameNativeImportedLightOverrides({ a }, { a }),
          "override lists compare by length and element");

    // Acceptance: each override names one ordered directional or point light and its own kind of field.
    SceneLightRecord directional;
    directional.kind = SceneLightKind::Directional;
    directional.id.value = 1;
    SceneLightRecord point;
    point.kind = SceneLightKind::Point;
    point.id.value = 2;
    SceneLightRecord spot;
    spot.kind = SceneLightKind::Spot;
    spot.id.value = 3;
    const std::vector<SceneLightRecord> lights { directional, point, spot };
    arbitgpu::NativeImportedLightOverride directionalIntensity;
    directionalIntensity.index = 0;
    directionalIntensity.hasIntensity = true;
    directionalIntensity.intensity = 1.0f;
    check(arbitgpu::validNativeImportedLightOverrides({ directionalIntensity }, lights),
          "a directional intensity override is accepted");
    auto pointPosition = directionalIntensity;
    pointPosition.index = 1;
    pointPosition.hasIntensity = false;
    pointPosition.hasPositionOffset = true;
    check(arbitgpu::validNativeImportedLightOverrides({ pointPosition }, lights),
          "a point position override is accepted");
    auto directionalPosition = directionalIntensity;
    directionalPosition.hasPositionOffset = true;
    check(!arbitgpu::validNativeImportedLightOverrides({ directionalPosition }, lights),
          "a position offset on a directional light is refused");
    auto spotTarget = directionalIntensity;
    spotTarget.index = 2;
    check(!arbitgpu::validNativeImportedLightOverrides({ spotTarget }, lights),
          "a spot light override is refused");
    auto outOfRange = directionalIntensity;
    outOfRange.index = 3;
    check(!arbitgpu::validNativeImportedLightOverrides({ outOfRange }, lights),
          "an override naming no light is refused");
    auto pointPositionIndexOne = pointPosition;
    check(!arbitgpu::validNativeImportedLightOverrides({ pointPosition, directionalIntensity }, lights)
              && !arbitgpu::validNativeImportedLightOverrides({ pointPosition, pointPositionIndexOne }, lights),
          "overrides must be strictly increasing and unique per light");
    auto empty = directionalIntensity;
    empty.hasIntensity = false;
    check(!arbitgpu::validNativeImportedLightOverrides({ empty }, lights),
          "an override that flags no field is refused");
    auto nonfinite = directionalIntensity;
    nonfinite.intensity = std::nan("");
    check(!arbitgpu::validNativeImportedLightOverrides({ nonfinite }, lights),
          "a non-finite override value is refused");
    auto zeroRotation = pointPosition;
    zeroRotation.index = 0;
    zeroRotation.hasPositionOffset = false;
    zeroRotation.hasRotationOffset = true;
    zeroRotation.rotationOffset = { 0.0f, 0.0f, 0.0f, 0.0f };
    check(!arbitgpu::validNativeImportedLightOverrides({ zeroRotation }, lights),
          "a zero rotation offset is refused");
}

void testLightOverrideBackendRead()
{
    using namespace HarmonicMIDI::grid;
    // A static scene with no sampled animatedLights: the override must reach the record
    // the backend reads, and the authored scene record must stay unchanged.
    auto scene = std::make_unique<Visual3DScene>();
    scene->lightCount = 1;
    scene->lights[0].id.value = 501;
    scene->lights[0].kind = SceneLightKind::Point;
    scene->lights[0].intensity = 1.0f;
    scene->lights[0].color = { 0.25f, 0.5f, 0.75f };
    arbitgpu::NativeImportedSceneRuntimeInputs staticRuntime;
    arbitgpu::NativeImportedLightOverride boosted;
    boosted.index = 0;
    boosted.hasIntensity = true;
    boosted.intensity = 3.0f;
    staticRuntime.lightOverrides.push_back(boosted);
    std::string error;
    check(staticRuntime.animatedLights.empty()
              && arbitgpu::validNativeImportedLightOverrideTargets(*scene, staticRuntime, error)
              && arbitgpu::effectiveNativeLightCount(*scene, staticRuntime) == 1,
          "a static scene reads its own light when animatedLights is empty");
    const auto staticRead = arbitgpu::effectiveNativeLightRecord(*scene, staticRuntime, 0);
    check(staticRead.intensity == 3.0f && staticRead.kind == SceneLightKind::Point
              && staticRead.color.x == 0.25f && staticRead.color.y == 0.5f,
          "the static override reaches the effective record and keeps untouched fields");
    check(scene->lights[0].intensity == 1.0f,
          "the authored static scene record is not rewritten by the override");
    check(readFrameLights(*scene, staticRuntime).size() == 1
              && readFrameLights(*scene, staticRuntime)[0].intensity == 3.0f,
          "the reader list carries the modulated static record");

    // An out-of-range index is refused with an error, never ignored.
    auto outOfRange = staticRuntime;
    outOfRange.lightOverrides[0].index = 1;
    error.clear();
    check(!arbitgpu::validNativeImportedLightOverrideTargets(*scene, outOfRange, error) && !error.empty(),
          "an override past the effective light count is refused with an error");

    // A position offset on a point is valid; the same offset on a directional light is refused.
    auto movedRuntime = staticRuntime;
    movedRuntime.lightOverrides[0] = {};
    movedRuntime.lightOverrides[0].index = 0;
    movedRuntime.lightOverrides[0].hasPositionOffset = true;
    movedRuntime.lightOverrides[0].positionOffset = { 1.0f, 0.0f, 0.0f };
    error.clear();
    check(arbitgpu::validNativeImportedLightOverrideTargets(*scene, movedRuntime, error),
          "a point position offset is accepted on a static point light");
    scene->lights[0].kind = SceneLightKind::Directional;
    check(!arbitgpu::validNativeImportedLightOverrideTargets(*scene, movedRuntime, error) && !error.empty(),
          "a position offset on a static directional light is refused");
    scene->lights[0].kind = SceneLightKind::Point;

    // With sampled animatedLights, the override applies to the sampled record, which
    // stays authored, and the count follows the sampled list.
    arbitgpu::NativeImportedSceneRuntimeInputs sampledRuntime;
    SceneLightRecord sampled = scene->lights[0];
    sampled.intensity = 2.0f;
    sampledRuntime.animatedLights.push_back(sampled);
    arbitgpu::NativeImportedLightOverride sampledBoost = boosted;
    sampledRuntime.lightOverrides.push_back(sampledBoost);
    error.clear();
    check(arbitgpu::validNativeImportedLightOverrideTargets(*scene, sampledRuntime, error)
              && arbitgpu::effectiveNativeLightRecord(*scene, sampledRuntime, 0).intensity == 3.0f
              && sampledRuntime.animatedLights[0].intensity == 2.0f,
          "a sampled light takes the override and the sampled record stays unmodified");

    // An Environment light override replaces the whole read list, so it is read as-is.
    arbitgpu::NativeImportedSceneRuntimeInputs environmentRuntime;
    SceneLightRecord environment = scene->lights[0];
    environment.intensity = 7.0f;
    environmentRuntime.lightOverride = environment;
    environmentRuntime.lightOverrides.push_back(boosted);
    error.clear();
    check(arbitgpu::validNativeImportedLightOverrideTargets(*scene, environmentRuntime, error)
              && readFrameLights(*scene, environmentRuntime).size() == 1
              && readFrameLights(*scene, environmentRuntime)[0].intensity == 7.0f,
          "an Environment light replaces the read list and is not modulated by scene-light overrides");
}

void testDirectionalLightModulation()
{
    using namespace videohelper::importedscene;
    using namespace HarmonicMIDI::grid;
    {
        // A point light takes clamped position offsets and clamped colour and intensity.
        SceneLightRecord point;
        point.id.value = 77;
        point.kind = SceneLightKind::Point;
        point.transform.translation = { 1.0f, 2.0f, 3.0f };
        point.transform.rotation = { 0.0f, 0.0f, 0.0f, 1.0f };
        const std::map<std::string, double> values {
            { "visual77/positionX", 1.0e7 }, { "visual77/positionY", -0.5 },
            { "visual77/intensity", -4.0 }, { "visual77/red", 99.0 }
        };
        check(evaluateDirectionalOrPointLight(values, point)
                  && point.transform.translation.x == 1000001.0f
                  && point.transform.translation.y == 1.5f
                  && point.transform.translation.z == 3.0f
                  && point.intensity == 0.0f && point.color.x == 64.0f,
              "point light offsets and absolute values clamp to their ranges");
        const std::map<std::string, double> nonfinite { { "visual77/intensity", std::nan("") } };
        check(!evaluateDirectionalOrPointLight(nonfinite, point),
              "a non-finite light value is refused");
    }
    {
        // A directional light turned 90 degrees about Y, composed over the identity.
        SceneLightRecord directional;
        directional.id.value = 78;
        directional.kind = SceneLightKind::Directional;
        directional.transform.rotation = { 0.0f, 0.0f, 0.0f, 1.0f };
        const std::map<std::string, double> values { { "visual78/rotationY", 90.0 } };
        const bool ok = evaluateDirectionalOrPointLight(values, directional);
        check(ok && std::abs(directional.transform.rotation.y - 0.70710678f) < 1e-5f
                  && std::abs(directional.transform.rotation.w - 0.70710678f) < 1e-5f
                  && directional.transform.rotation.x == 0.0f && directional.transform.rotation.z == 0.0f,
              "a directional rotation offset composes into the light quaternion");
    }

    const auto plan = composedScenePlan();
    const std::vector<videowire::CompiledVisualLayerPlan> plans { plan };
    FakeExecution execution;
    FakeLayer layer;
    std::vector<FakeReceipt> owners;
    VisualImportedScenePlanCache cache;
    std::string error;
    const auto render = [&](const std::map<std::string, double>& values) {
        return prepareVisualImportedSceneLayerAtTime(plans, 8, 640, 360, 0.0, 24.0,
            NativeImportedSceneRenderUse::Preview, &execution, &cache, layer, owners, error, &values);
    };
    std::map<std::string, double> parameters;
    seedImportedSceneRuntimeParameters(plans, 8, parameters);
    {
        const bool conditionOk = render(parameters) == VisualImportedScenePreparation::rendered;
        check(conditionOk, "seeded light controls render the authored composed scene: " + error);
    }
    const auto authored = execution.lastRequest.sceneSnapshot;
    check(authored && authored->lightCount == 1
              && authored->lights[0].kind == SceneLightKind::Directional,
          "the composed plan carries one directional light");
    const auto key = "visual" + std::to_string(authored->lights[0].id.value) + "/";
    check(parameters.count(key + "intensity") == 1 && parameters.count(key + "red") == 1
              && parameters.count(key + "rotationY") == 1 && parameters.count(key + "positionX") == 0,
          "a directional light seeds colour, intensity and rotation offsets only");
    check(execution.lastRequest.runtimeInputs.animatedLights.empty()
              && execution.lastRequest.runtimeInputs.lightOverrides.empty(),
          "authored light values derive no per-frame light record or override");

    auto modulated = parameters;
    modulated[key + "intensity"] = 3.75;
    modulated[key + "red"] = 0.5;
    {
        const bool conditionOk = render(modulated) == VisualImportedScenePreparation::rendered;
        check(conditionOk, "a modulated directional intensity and colour render: " + error);
    }
    const auto& overrides = execution.lastRequest.runtimeInputs.lightOverrides;
    check(overrides.size() == 1 && overrides[0].hasIntensity && overrides[0].hasColor
              && !overrides[0].hasRotationOffset && !overrides[0].hasPositionOffset,
          "only the modulated fields are flagged on the light override");
    const auto readLights = [&] {
        return readFrameLights(*execution.lastRequest.sceneSnapshot, execution.lastRequest.runtimeInputs);
    };
    const auto frameLights = readLights();
    check(frameLights.size() == 1 && frameLights[0].intensity == 3.75f
              && frameLights[0].color.x == 0.5f && frameLights[0].color.y == authored->lights[0].color.y
              && authored->lights[0].intensity != 3.75f,
          "the modulated intensity and colour reach the executed frame light, not the authored scene");

    const auto firstOverrides = execution.lastRequest.runtimeInputs.lightOverrides;
    modulated[key + "intensity"] = 2.0e6;
    modulated[key + "red"] = -3.0;
    check(render(modulated) == VisualImportedScenePreparation::rendered
              && !arbitgpu::sameNativeImportedLightOverrides(firstOverrides, execution.lastRequest.runtimeInputs.lightOverrides)
              && readLights().size() == 1
              && readLights()[0].intensity == 1000000.0f
              && readLights()[0].color.x == 0.0f,
          "out-of-range light values clamp to the node ranges");

    auto turned = parameters;
    turned[key + "rotationY"] = 90.0;
    check(render(turned) == VisualImportedScenePreparation::rendered
              && readLights().size() == 1
              && std::abs(readLights()[0].transform.rotation.y) > 0.5f,
          "a directional rotation offset reaches the executed frame light");

    auto broken = parameters;
    broken[key + "intensity"] = std::nan("");
    check(render(broken) == VisualImportedScenePreparation::rejected,
          "a non-finite directional light value is refused by the preparation");
}

void testBakedPbrModulation()
{
    using namespace videohelper::importedscene;
    const auto plan = pbrComposedScenePlan();
    const std::vector<videowire::CompiledVisualLayerPlan> plans { plan };
    std::map<std::string, double> parameters;
    seedImportedSceneRuntimeParameters(plans, 8, parameters);
    check(parameters.count("visual2/baseRed") == 1 && parameters.at("visual2/baseRed") == 0.5
              && parameters.count("visual2/roughness") == 1 && parameters.at("visual2/roughness") == 0.5
              && parameters.count("visual5/metallic") == 1 && parameters.count("visual5/emissiveBlue") == 1,
          "each PBR Material node seeds its numeric material values");
    check(parameters.count("visual2/opacity") == 0 && parameters.count("visual20/baseRed") == 0
              && parameters.count("visual12/baseRed") == 0,
          "opacity and records without a PBR Material node are not modulation targets");

    FakeExecution execution;
    FakeLayer layer;
    std::vector<FakeReceipt> owners;
    VisualImportedScenePlanCache cache;
    std::string error;
    const auto render = [&](const std::map<std::string, double>& values) {
        return prepareVisualImportedSceneLayerAtTime(plans, 8, 640, 360, 0.0, 24.0,
            NativeImportedSceneRenderUse::Preview, &execution, &cache, layer, owners, error, &values);
    };
    {
        const bool conditionOk = render(parameters) == VisualImportedScenePreparation::rendered;
        check(conditionOk, "seeded PBR values render the authored composed scene: " + error);
    }
    const auto authored = execution.lastRequest.sceneSnapshot;
    const auto* slot = cache.bakedTargetSlot(8);
    check(authored && slot && slot->derivations == 0 && !slot->scene
              && authored->materials[0].baseColor.x == 0.5f,
          "values equal to the authored material derive nothing");

    auto modulated = parameters;
    modulated["visual2/baseRed"] = 2.0;
    modulated["visual5/roughness"] = 0.25;
    {
        const bool conditionOk = render(modulated) == VisualImportedScenePreparation::rendered;
        check(conditionOk, "a modulated PBR value renders: " + error);
    }
    const auto derived = execution.lastRequest.sceneSnapshot;
    check(derived && derived != authored && slot && slot->derivations == 1
              && derived->materials[0].baseColor.x == 2.0f
              && derived->materials[0].baseColor.y == 0.25f
              && derived->materials[0].roughness == 0.5f
              && derived->materials[1].roughness == 0.25f
              && authored->materials[0].baseColor.x == 0.5f
              && execution.lastRequest.evaluationRevision == 12,
          "the routed value reaches the evaluated scene and leaves the compiled scene untouched");
    const auto uploads = execution.staticGpuUploads;
    check(render(modulated) == VisualImportedScenePreparation::rendered
              && execution.lastRequest.sceneSnapshot == derived && slot && slot->derivations == 1
              && execution.staticGpuUploads == uploads,
          "an unchanged modulated value reuses the derived scene without a replan");
    auto nearby = modulated;
    nearby["visual2/baseRed"] = 2.0005;
    check(render(nearby) == VisualImportedScenePreparation::rendered
              && execution.lastRequest.sceneSnapshot == derived && slot && slot->derivations == 1,
          "a change below the quantization step keeps the derived scene");
    auto clamped = modulated;
    clamped["visual2/baseRed"] = 100.0;
    check(render(clamped) == VisualImportedScenePreparation::rendered
              && execution.lastRequest.sceneSnapshot->materials[0].baseColor.x == 64.0f
              && slot && slot->derivations == 2,
          "a modulated colour clamps to the PBR Material range");
    check(render(parameters) == VisualImportedScenePreparation::rendered
              && execution.lastRequest.sceneSnapshot == authored && slot && !slot->scene,
          "returning to the authored values restores the compiled scene");

    auto unknown = parameters;
    unknown["visual2/opacity"] = 0.5;
    unknown["visual20/baseRed"] = 3.0;
    check(render(unknown) == VisualImportedScenePreparation::rendered
              && execution.lastRequest.sceneSnapshot == authored
              && execution.lastRequest.sceneSnapshot->materials[0].opacity == 1.0f,
          "opacity and unknown material keys are skipped");
    auto invalid = parameters;
    invalid["visual5/metallic"] = std::numeric_limits<double>::quiet_NaN();
    check(render(invalid) == VisualImportedScenePreparation::rejected,
          "a nonfinite PBR value refuses the frame");

    // A retained Geometry Scene keeps each PBR node identity on its record.
    visualimportedscenerender::Request request;
    check(visualimportedscenerender::decode(plan.operations.back().payloadXml, request),
          "retained PBR fixture decodes");
    auto retainedPlan = plan;
    retainedPlan.nodeIds = { 6, 11 };
    retainedPlan.nodeKinds = { videowire::geometry::kRetainedSceneOperation, "visual.3d.render" };
    retainedPlan.operations = { { 6, retainedPlan.nodeKinds[0], "control-eval", "" },
                                { 11, retainedPlan.nodeKinds[1], "native-gpu", visualimportedscenerender::encode(request) } };
    retainedPlan.edges = { { 6, 0, 11, 0 } };
    retainedPlan.ports = { { 6, 0, 1, "out", "control", "scene3D", "unspecified", "unspecified" },
                           { 11, 0, 1, "in", "control", "scene3D", "unspecified", "unspecified" },
                           { 11, 1, 1, "out", "frame", "image", "rgba8", "sRGB" } };
    const std::vector<videowire::CompiledVisualLayerPlan> retainedPlans { retainedPlan };
    std::map<std::string, double> retainedParameters;
    seedImportedSceneRuntimeParameters(retainedPlans, 8, retainedParameters);
    check(retainedParameters.count("visual2/baseGreen") == 1 && retainedParameters.count("visual5/emissiveRed") == 1,
          "a retained Geometry Scene seeds its PBR material records");
    retainedParameters["visual5/emissiveRed"] = 4.0;
    FakeExecution retainedExecution;
    VisualImportedScenePlanCache retainedCache;
    {
        const bool conditionOk = prepareVisualImportedSceneLayerAtTime(retainedPlans, 8, 640, 360, 0.0, 24.0,
              NativeImportedSceneRenderUse::Export, &retainedExecution, &retainedCache, layer, owners, error,
              &retainedParameters) == VisualImportedScenePreparation::rendered
              && retainedExecution.lastRequest.sceneSnapshot
              && retainedExecution.lastRequest.sceneSnapshot->materials[1].emissive.x == 4.0f;
        check(conditionOk, "a routed emission value reaches the retained scene export: " + error);
    }
}

namespace {
// The key table entry for a published Surface constant name, or null.
const geometrysurfacematerial::ConstantField* surfaceConstantField(const std::string& name)
{
    for (const auto& field : geometrysurfacematerial::kConstantFields)
        if (name == field.name) return &field;
    return nullptr;
}

// The literal a published Surface constant holds in one executed program. A
// program without that constant reads NaN, which fails every equality check.
double surfaceConstantValue(const geometrysurfacematerial::ObjectProgram& object, const std::string& name)
{
    const auto* field = surfaceConstantField(name);
    if (field == nullptr) return std::numeric_limits<double>::quiet_NaN();
    const auto& program = object.program.material.program;
    const auto operation = geometrysurfacematerial::constantOperation(program, *field);
    return operation ? static_cast<double>(program.operations[*operation].literal[field->lane])
                     : std::numeric_limits<double>::quiet_NaN();
}

// Both objects of the composed scene, each drawn by a per-object Surface Material
// that lowers from source node 7. Every program is admitted, so its digest is the
// one the compiler stamps for those constants.
geometrysurfacematerial::ObjectPrograms composedSurfacePrograms(
    const visualimportedscenerender::Request& request, std::string& error)
{
    geometrysurfacematerial::ObjectPrograms programs;
    const auto& scene = request.sceneSnapshot;
    for (std::size_t index = 0; index < 2; ++index)
    {
        geometrysurfacematerial::ObjectProgram object;
        object.program.sourceMaterial = 7;
        auto& material = object.program.material;
        material.scene = scene->id;
        material.sceneSnapshot = scene;
        material.sceneRevision = material.structuralRevision = material.programRevision = request.structuralRevision;
        material.evaluationRevision = request.evaluationRevision;
        material.binding.targetKind = surfacematerialbinding::BindingTargetKind::ObjectOverride;
        material.binding.object = scene->objects[index].id;
        material.binding.surfaceMaterialRevision = material.programRevision;
        material.program = videohelper::tests::smoothglassfixture::authoredProgram(
            scene->objects[index].material.value);
        const auto admitted = surfacematerial::admit(material.program, error);
        if (!admitted) return {};
        material.binding.surfaceMaterialDigest = admitted->structuralDigest();
        programs.push_back(std::move(object));
    }
    return programs;
}

// The same composed programs as a thin surface, the one transport whose native pass
// blends. The dispersion literal is zero, as a thin surface requires.
geometrysurfacematerial::ObjectPrograms thinSurfacePrograms(
    const visualimportedscenerender::Request& request, std::string& error)
{
    auto programs = composedSurfacePrograms(request, error);
    for (auto& object : programs)
    {
        auto& program = object.program.material.program;
        program.transport = surfacematerial::TransportMode::ThinSurface;
        program.operations.back().literal[0] = 0.0f;
        const auto admitted = surfacematerial::admit(program, error);
        if (!admitted) return {};
        object.program.material.binding.surfaceMaterialDigest = admitted->structuralDigest();
    }
    return programs;
}
} // namespace

void testBakedSurfaceConstantModulation()
{
    using namespace videohelper::importedscene;
    auto plan = composedScenePlan();
    visualimportedscenerender::Request request;
    check(visualimportedscenerender::decode(plan.operations.back().payloadXml, request) && request.sceneSnapshot,
          "Surface constant fixture decodes its composed request");
    std::string error;
    request.surfacePrograms = composedSurfacePrograms(request, error);
    check(request.surfacePrograms.size() == 2,
          "per-object Surface programs admit on the composed scene: " + error);
    plan.operations.back().payloadXml = visualimportedscenerender::encode(request);
    {
        // Failure-only PROBE: each conjunct of the executor canonical gate
        // (visual_plan_executor.h, composed Scene3D route) on the re-encoded render operation.
        const auto& gateOp = plan.operations.back();
        visualimportedscenerender::Request gateRequest;
        const bool gateDecoded = visualimportedscenerender::decode(gateOp.payloadXml, gateRequest);
        const bool gateRoundTrip = gateDecoded && visualimportedscenerender::encode(gateRequest) == gateOp.payloadXml;
        const bool gateScene = gateDecoded && gateRequest.sceneSnapshot != nullptr;
        const bool gateCapability = gateOp.backendCapability == visualimportedscenerender::kRenderBackendCapability;
        const bool gateNode = gateOp.nodeId >= 0;
        const bool gateRevision = gateDecoded && gateRequest.structuralRevision == plan.structuralRevision;
        const bool gateStable = gateDecoded
            && gateRequest.renderStableId == static_cast<std::uint64_t>(gateOp.nodeId) + 1u;
        const bool gateOk = gateRoundTrip && gateScene && gateCapability && gateNode && gateRevision && gateStable;
        if (!gateOk)
        {
            const std::string pattern = std::string("gate:") + (gateDecoded ? '1' : '0') + (gateRoundTrip ? '1' : '0')
                + (gateScene ? '1' : '0') + (gateCapability ? '1' : '0') + (gateNode ? '1' : '0')
                + (gateRevision ? '1' : '0') + (gateStable ? '1' : '0');
            if (probeFirstTime(pattern))
                std::cerr << "PROBE: canonical gate kind=" << gateOp.kind << " nodeId=" << gateOp.nodeId
                    << " decoded=" << gateDecoded << " roundTrip=" << gateRoundTrip << " sceneSnapshot=" << gateScene
                    << " backendCapability='" << gateOp.backendCapability << "' (want native-gpu) capabilityMatches=" << gateCapability
                    << " nodeIdNonNegative=" << gateNode
                    << " structuralRevision=" << gateRequest.structuralRevision << " planRevision=" << plan.structuralRevision
                    << " revisionMatches=" << gateRevision
                    << " renderStableId=" << static_cast<unsigned long long>(gateRequest.renderStableId)
                    << " (want nodeId+1=" << gateOp.nodeId + 1 << ") stableMatches=" << gateStable << '\n';
        }
    }
    const std::vector<videowire::CompiledVisualLayerPlan> plans { plan };

    // Each constant seeds under its source Surface Material. Scene material ids
    // publish nothing, because a Surface program keys its constants by source.
    std::map<std::string, double> parameters;
    seedImportedSceneRuntimeParameters(plans, 8, parameters);
    check(parameters.count("visual7/roughness") == 1 && parameters.at("visual7/roughness") == 0.0
              && parameters.count("visual7/baseRed") == 1 && parameters.at("visual7/baseRed") == 1.0
              && parameters.count("visual7/emissiveBlue") == 1,
          "each per-object Surface constant seeds under its source material");
    check(parameters.count("visual20/roughness") == 0 && parameters.count("visual21/roughness") == 0,
          "scene material ids publish no Surface constants");
    check(parameters.count("visual7/opacity") == 0 && parameters.count("visual7/baseRed") == 1,
          "a closed-volume Surface seeds no opacity target and keeps its tint");

    FakeExecution execution;
    FakeLayer layer;
    std::vector<FakeReceipt> owners;
    VisualImportedScenePlanCache cache;
    const auto render = [&](const std::map<std::string, double>& values) {
        return prepareVisualImportedSceneLayerAtTime(plans, 8, 640, 360, 0.0, 24.0,
            NativeImportedSceneRenderUse::Preview, &execution, &cache, layer, owners, error, &values);
    };
    {
        const bool conditionOk = render(parameters) == VisualImportedScenePreparation::rendered;
        check(conditionOk, "seeded Surface constants render the authored programs: " + error);
    }
    const auto* slot = cache.bakedTargetSlot(8);
    const auto authoredScene = execution.lastRequest.sceneSnapshot;
    const auto authoredRevision = execution.lastRequest.evaluationRevision;
    const auto authoredPrograms = execution.lastRequest.surfacePrograms;
    const auto authoredDigest = authoredPrograms.empty() ? std::string {}
        : authoredPrograms[0].program.material.binding.surfaceMaterialDigest;
    check(slot && slot->derivations == 0 && !slot->surfacePrograms && authoredPrograms.size() == 2
              && surfaceConstantValue(authoredPrograms[0], "roughness") == 0.0,
          "values equal to the authored literals derive nothing");

    // A routed value writes the literal in every program its source node lowers to,
    // lifts the evaluation revision, and binds the programs to the unchanged scene.
    auto modulated = parameters;
    modulated["visual7/roughness"] = 0.5;
    modulated["visual7/baseRed"] = 0.25;
    {
        const bool conditionOk = render(modulated) == VisualImportedScenePreparation::rendered;
        check(conditionOk, "a modulated Surface constant renders: " + error);
    }
    const auto derived = execution.lastRequest.surfacePrograms;
    check(slot && slot->derivations == 1 && derived.size() == 2
              && surfaceConstantValue(derived[0], "roughness") == 0.5
              && surfaceConstantValue(derived[1], "roughness") == 0.5
              && surfaceConstantValue(derived[0], "baseRed") == 0.25
              && surfaceConstantValue(derived[1], "baseRed") == 0.25,
          "a routed constant reaches every program its source node lowers to");
    const auto admittedDerived = derived.size() == 2
        ? surfacematerial::admit(derived[0].program.material.program, error)
        : decltype(surfacematerial::admit(derived[0].program.material.program, error)) {};
    check(admittedDerived && derived.size() == 2
              && derived[0].program.material.binding.surfaceMaterialDigest == admittedDerived->structuralDigest()
              && derived[0].program.material.binding.surfaceMaterialDigest != authoredDigest,
          "the derived program carries the digest it admits with: " + error);
    check(derived.size() == 2 && execution.lastRequest.evaluationRevision == authoredRevision + 1
              && derived[0].program.material.evaluationRevision == authoredRevision + 1
              && derived[1].program.material.evaluationRevision == authoredRevision + 1,
          "a derived Surface program lifts the evaluation revision of its request");
    check(derived.size() == 2 && execution.lastRequest.sceneSnapshot == authoredScene
              && derived[0].program.material.sceneSnapshot == authoredScene
              && derived[1].program.material.sceneSnapshot == authoredScene,
          "the derived programs bind the unchanged scene");

    check(render(modulated) == VisualImportedScenePreparation::rendered && slot && slot->derivations == 1
              && execution.lastRequest.evaluationRevision == authoredRevision + 1,
          "an unchanged modulated value reuses the derived programs");
    auto nearby = modulated;
    nearby["visual7/roughness"] = 0.5004;
    check(render(nearby) == VisualImportedScenePreparation::rendered && slot && slot->derivations == 1,
          "a change below the quantization step keeps the derived programs");

    auto clamped = modulated;
    clamped["visual7/roughness"] = 1.5;
    clamped["visual7/baseRed"] = -2.0;
    check(render(clamped) == VisualImportedScenePreparation::rendered && execution.lastRequest.surfacePrograms.size() == 2
              && slot && slot->derivations == 2
              && surfaceConstantValue(execution.lastRequest.surfacePrograms[0], "roughness") == 1.0
              && surfaceConstantValue(execution.lastRequest.surfacePrograms[0], "baseRed") == 0.0
              && execution.lastRequest.evaluationRevision == authoredRevision + 2,
          "a modulated constant clamps to its range and lifts the revision");

    // A nonfinite value keeps the last good programs and their revision.
    auto invalid = clamped;
    invalid["visual7/roughness"] = std::numeric_limits<double>::quiet_NaN();
    {
        const bool conditionOk = render(invalid) == VisualImportedScenePreparation::rendered && execution.lastRequest.surfacePrograms.size() == 2
              && slot && slot->derivations == 2
              && surfaceConstantValue(execution.lastRequest.surfacePrograms[0], "roughness") == 1.0
              && surfaceConstantValue(execution.lastRequest.surfacePrograms[0], "baseRed") == 0.0
              && execution.lastRequest.evaluationRevision == authoredRevision + 2;
        check(conditionOk, "a nonfinite Surface constant keeps the last good programs: " + error);
    }

    // Returning to the authored literals restores the compiled programs. Their
    // revision still rises, so the native owner never sees it go backwards.
    check(render(parameters) == VisualImportedScenePreparation::rendered && execution.lastRequest.surfacePrograms.size() == 2
              && slot && !slot->surfacePrograms
              && slot->derivations == 3
              && surfaceConstantValue(execution.lastRequest.surfacePrograms[0], "roughness") == 0.0
              && execution.lastRequest.surfacePrograms[0].program.material.binding.surfaceMaterialDigest
                  == authoredDigest
              && execution.lastRequest.evaluationRevision > authoredRevision + 2
              && execution.lastRequest.surfacePrograms[0].program.material.evaluationRevision
                  == execution.lastRequest.evaluationRevision,
          "returning to the authored literals restores the compiled programs");

    // Opacity is a Surface constant only on a thin-surface program, the one transport
    // whose native pass blends. The composed programs above are closed-volume glass, so
    // a routed opacity is neither offered nor derived for them.
    {
        videohelper::bakedtarget::Slot::SurfaceKey closedKey;
        std::map<std::string, double> closedTranslucent;
        closedTranslucent["visual7/opacity"] = 0.5;
        check(videohelper::bakedtarget::surfaceKeyFor(request, closedTranslucent, closedKey)
                  && closedKey.empty(),
              "a closed-volume Surface does not take a routed opacity");
    }

    // The thin-surface variant of the same composed scene. A routed opacity writes the
    // Surface Opacity literal in both programs and round-trips the surface wire. A value
    // above one clamps to the authored one, so it derives nothing.
    std::string thinError;
    auto thinRequest = request;
    thinRequest.surfacePrograms = thinSurfacePrograms(request, thinError);
    check(thinRequest.surfacePrograms.size() == 2,
          "thin-surface Surface programs admit on the composed scene: " + thinError);
    {
        std::map<std::string, double> translucent;
        translucent["visual7/opacity"] = 0.5;
        videohelper::bakedtarget::Slot::SurfaceKey key;
        geometrysurfacematerial::ObjectPrograms derived;
        check(videohelper::bakedtarget::surfaceKeyFor(thinRequest, translucent, key) && key.size() == 2
                  && videohelper::bakedtarget::deriveSurfacePrograms(thinRequest, key, derived, thinError)
                  && derived.size() == 2
                  && surfaceConstantValue(derived[0], "opacity") == 0.5
                  && surfaceConstantValue(derived[1], "opacity") == 0.5,
              "a routed opacity reaches every thin-surface program its source node lowers to: " + thinError);
        std::ostringstream encoded;
        encoded << visualimportedscenerender::detail::encodeSurfacePrograms(derived);
        std::istringstream wire(encoded.str());
        geometrysurfacematerial::ObjectPrograms decoded;
        check(visualimportedscenerender::detail::decodeSurfacePrograms(wire, decoded)
                  && decoded.size() == derived.size()
                  && surfaceConstantValue(decoded[0], "opacity") == 0.5
                  && visualimportedscenerender::detail::encodeSurfacePrograms(decoded) == encoded.str(),
              "a translucent thin-surface program round-trips its opacity through the wire");
    }
    {
        std::map<std::string, double> over;
        over["visual7/opacity"] = 2.0;
        videohelper::bakedtarget::Slot::SurfaceKey key;
        check(videohelper::bakedtarget::surfaceKeyFor(thinRequest, over, key) && key.empty(),
              "a routed opacity above one clamps to the authored one and derives nothing");
    }

    // A constant that a second output also reads is not a target, because a write
    // would move that output too. Seeding and key derivation share this predicate.
    auto shared = videohelper::tests::smoothglassfixture::authoredProgram(20);
    shared.outputs[static_cast<std::size_t>(surfacematerial::OutputSemantic::Clearcoat)]
        = shared.output(surfacematerial::OutputSemantic::Roughness);
    check(!geometrysurfacematerial::constantOperation(shared, *surfaceConstantField("roughness"))
              && geometrysurfacematerial::constantOperation(shared, *surfaceConstantField("baseRed")).has_value(),
          "a constant that a second output also reads is not a Surface modulation target");
}

void testBakedDisplacementModulation()
{
    using namespace videohelper::importedscene;
    const auto plan = importedVertexPlan();
    const std::vector<videowire::CompiledVisualLayerPlan> plans { plan };
    std::map<std::string, double> parameters;
    seedImportedSceneRuntimeParameters(plans, 7, parameters);
    check(parameters.count("visual7/maximumDisplacement") == 1
              && parameters.at("visual7/maximumDisplacement") == 0.25
              && parameters.count("visual6/maximumDisplacement") == 0,
          "the Bounded Displacement root seeds its maximum displacement");

    FakeExecution execution;
    FakeLayer layer;
    std::vector<FakeReceipt> owners;
    VisualImportedScenePlanCache cache;
    std::string error;
    const auto render = [&](const std::map<std::string, double>& values) {
        return prepareVisualImportedSceneLayerAtTime(plans, 7, 640, 360, 0.5, 60.0,
            NativeImportedSceneRenderUse::Preview, &execution, &cache, layer, owners, error, &values);
    };
    const auto bound = [&]() {
        const auto& material = execution.lastRequest.material;
        if (!material || !material->vertexModifier) return -1.0;
        for (const auto& record : material->vertexModifier->records)
            if (record.stableId == material->vertexModifier->rootId) return record.parameters[0];
        return -1.0;
    };
    {
        const bool conditionOk = render(parameters) == VisualImportedScenePreparation::rendered;
        check(conditionOk, "the seeded displacement renders: " + error);
    }
    const auto baseRevision = execution.lastRequest.evaluationRevision;
    const auto* slot = cache.bakedTargetSlot(7);
    check(slot && slot->derivations == 0 && bound() == 0.25 && slot->evaluationRevision == 0,
          "the authored bound derives no program");

    auto modulated = parameters;
    modulated["visual7/maximumDisplacement"] = 0.5;
    {
        const bool conditionOk = render(modulated) == VisualImportedScenePreparation::rendered;
        check(conditionOk, "a modulated displacement bound renders: " + error);
    }
    const auto liftedRevision = execution.lastRequest.evaluationRevision;
    check(bound() == 0.5 && slot && slot->derivations == 1 && liftedRevision > baseRevision
              && execution.lastRequest.material->evaluationRevision == liftedRevision
              && execution.lastRequest.structuralRevision == plan.structuralRevision,
          "the routed bound reaches the vertex program under a newer evaluation revision");
    std::string admission;
    {
        const bool conditionOk = videohelper::vertexmodifier::admitVertexModifierIr(
              *execution.lastRequest.material->vertexModifier, {}, admission).has_value();
        check(conditionOk, "the derived vertex program passes the same admission: " + admission);
    }
    for (const double value : { 0.5, 0.50001 })
    {
        modulated["visual7/maximumDisplacement"] = value;
        check(render(modulated) == VisualImportedScenePreparation::rendered && bound() == 0.5
                  && slot && slot->derivations == 1 && execution.lastRequest.evaluationRevision == liftedRevision,
              "an unchanged quantized bound reuses the derived program without a replan");
    }
    modulated["visual7/maximumDisplacement"] = 5000.0;
    check(render(modulated) == VisualImportedScenePreparation::rendered && bound() == 1000.0
              && slot && slot->derivations == 2 && execution.lastRequest.evaluationRevision > liftedRevision,
          "an out-of-range bound clamps to the admitted maximum");
    const auto clampedRevision = execution.lastRequest.evaluationRevision;
    check(render(parameters) == VisualImportedScenePreparation::rendered && bound() == 0.25
              && slot && slot->derivations == 3 && execution.lastRequest.evaluationRevision > clampedRevision,
          "returning to the authored bound keeps the evaluation revision moving forward");

    VisualImportedScenePlanCache evicted;
    auto floorCarried = cache.bakedRevisionFloor(7);
    evicted.bakedRevisionFloor(7) = floorCarried;
    check(prepareVisualImportedSceneLayerAtTime(plans, 7, 640, 360, 0.5, 60.0,
              NativeImportedSceneRenderUse::Preview, &execution, &evicted, layer, owners, error, &parameters)
              == VisualImportedScenePreparation::rendered
              && execution.lastRequest.evaluationRevision > floorCarried.evaluationRevision && bound() == 0.25,
          "a fresh plan entry for a clip that lifted its revision keeps lifting");

    auto unknown = parameters;
    unknown["visual6/maximumDisplacement"] = 0.75;
    unknown["visual7/gain"] = 2.0;
    const auto derivations = slot ? slot->derivations : decltype(slot->derivations) {};
    check(render(unknown) == VisualImportedScenePreparation::rendered && bound() == 0.25
              && slot && slot->derivations == derivations,
          "keys outside the Bounded Displacement root are skipped");
    auto invalid = parameters;
    invalid["visual7/maximumDisplacement"] = std::numeric_limits<double>::infinity();
    check(render(invalid) == VisualImportedScenePreparation::rejected,
          "a nonfinite displacement bound refuses the frame");
}

void testMaterialFieldBinding()
{
    using namespace videohelper::importedscene;
    auto plan = importedScenePlan(false, false, true);
    visualimportedscenerender::Request request;
    check(visualimportedscenerender::decode(plan.operations.back().payloadXml, request), "field fixture has an authored request");
    request.materialField = videohelper::tests::materialFieldFixture(72);
    std::string error;
    const auto field = materialfield::admit(*request.materialField, error);
    check(field.has_value(), "canonical Geometry Core field transport admits without a new field identity: " + error);
    if (!field) return;
    const auto topology = materialfield::topology(*request.materialField, field->descriptor());
    check(std::any_of(topology.nodes.begin(), topology.nodes.end(), [](const auto& node)
            { return node.second == "visual.geometry.field.float"; }),
        "canonical material field topology carries the exact field-float kind");
    for (const auto& [id, kind] : topology.nodes)
    {
        plan.nodeIds.insert(plan.nodeIds.end()-1, id);
        plan.nodeKinds.insert(plan.nodeKinds.end()-1, kind);
        plan.operations.insert(plan.operations.end()-1, {id, kind, "control-eval", ""});
        const auto output = topology.outputPorts.at(id);
        for (int port = 0; port <= output; ++port)
            plan.ports.push_back({id, port, 1, port == output ? "out" : "in", "control", "field", "unspecified", "unspecified"});
    }
    for (const auto& [from, output, to, input] : topology.edges) plan.edges.push_back({from, output, to, input});
    plan.ports.push_back({72, 1, 1, "in", "frame", "image", "rgba8", "sRGB"});
    plan.ports.push_back({72, 2, 1, "in", "control", "field", "unspecified", "unspecified"});
    plan.operations.back().payloadXml = visualimportedscenerender::encode(request);
    visualimportedscenerender::Request restored;
    check(!plan.operations.back().payloadXml.empty()
              && visualimportedscenerender::decode(plan.operations.back().payloadXml, restored)
              && visualimportedscenerender::sameRequest(request, restored), "v12 preserves the field adapter and existing request exactly");
    FakeExecution execution; FakeLayer layer; std::vector<FakeReceipt> owners;
    const auto base = request.diffractionMaterial->material.roughness.rmsHeightNanometres;
    for (auto use : {NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export})
        for (double seconds : {0.0, 1.0, 0.5, 4.0, 0.0})
        {
            std::map<std::string,double> parameters{{"visual73/rmsHeightNanometres", base + 5}};
            {
                const bool conditionOk = prepareVisualImportedSceneLayerAtTime({plan}, 7, 640, 360, seconds, 24,
                    use, &execution, layer, owners, error, &parameters) == VisualImportedScenePreparation::rendered;
                check(conditionOk, "field scheduler admits forward, backward and loop samples: " + error);
            }
            const auto& evaluated = execution.lastRequest.runtimeInputs.diffractionParameters;
            check(evaluated.count("rmsHeightNanometres")
                && std::abs(evaluated.at("rmsHeightNanometres") - (base + 5 + 10*std::sin(seconds*1.5707963267948966))) < 1e-6,
                "rational-time field evaluation adds after Patch Bay and repeats on seek/export");
        }
    for (int mutation = 0; mutation < 3; ++mutation)
    {
        auto invalid = plan;
        if (mutation == 0) invalid.edges.back().toPort = 1;
        if (mutation == 1) invalid.operations[2].kind = "visual.geometry.field-timeline.point";
        if (mutation == 2) invalid.ports.back().dataType = "vector";
        check(prepareVisualImportedSceneLayerAtTime({invalid}, 7, 640, 360, 0, 24,
                NativeImportedSceneRenderUse::Preview, &execution, layer, owners, error)
                  == VisualImportedScenePreparation::rejected, "material field port, node kind and endpoint tampering fail admission");
    }
    for (int mutation = 0; mutation < 4; ++mutation)
    {
        auto invalid = *request.materialField;
        if (mutation == 0) invalid.index = materialfield::kMaximumElements;
        if (mutation == 1) invalid.gain = std::numeric_limits<double>::quiet_NaN();
        if (mutation == 2) { invalid.reduction = materialfield::Reduction::index; invalid.index = 4; }
        if (mutation == 3) invalid.materialNode = 9000;
        check(!materialfield::admit(invalid, error), "out-of-range samples, nonfinite gain and overlapping identities reject");
    }
    for (auto reduction : {materialfield::Reduction::index, materialfield::Reduction::mean,
                           materialfield::Reduction::minimum, materialfield::Reduction::maximum})
    {
        auto binding = *request.materialField; binding.reduction = reduction;
        videowire::geometry::RuntimeFieldEvaluation clock; clock.timelineSeconds = 1;
        diffractionmaterialbinding::RuntimeParameters parameters;
        check(materialfield::evaluate(binding, clock, *request.diffractionMaterial, parameters, error)
                  && std::abs(parameters.at("rmsHeightNanometres") - (base+10)) < 1e-6,
              "each bounded reduction uses the canonical evaluated Field samples");
    }
    const auto score = videohelper::tests::materialFieldFixture(72, true);
    check(!materialfield::admit(videohelper::tests::materialFieldFixture(72, false, 2049), error),
          "Material Field rejects excess cardinality before runtime evaluation");
    videowire::geometry::RuntimeFieldEvaluation evaluation;
    diffractionmaterialbinding::RuntimeParameters parameters;
    check(!materialfield::evaluate(score, evaluation, *request.diffractionMaterial, parameters, error),
          "Score Sample Field cannot fall back to a constant without the canonical score evaluator");
    evaluation.scoreAt = [&](const auto& operation, const auto& positions, auto& output, auto& diagnostic)
    {
        check(operation.field.values[4] == 9003 && positions.elements.size() == 4,
              "existing Score Sample identity and sample positions reach RuntimeFieldEvaluation");
        output.elements = {{{1,0,0,0}},{{2,0,0,0}},{{3,0,0,0}},{{4,0,0,0}}};
        diagnostic.clear(); return true;
    };
    for (auto reduction : {materialfield::Reduction::index, materialfield::Reduction::mean,
                           materialfield::Reduction::minimum, materialfield::Reduction::maximum})
    {
        auto binding = score; binding.reduction = reduction; binding.index = 2;
        parameters.clear();
        const auto expected = reduction == materialfield::Reduction::index ? 3.0
            : reduction == materialfield::Reduction::mean ? 2.5
            : reduction == materialfield::Reduction::minimum ? 1.0 : 4.0;
        check(materialfield::evaluate(binding, evaluation, *request.diffractionMaterial, parameters, error)
                  && parameters.at("rmsHeightNanometres") == base + 10*expected,
              "nonuniform canonical score samples use defined index, mean, minimum and maximum semantics");
    }
}

videowire::CompiledVisualLayerPlan nativeResolverPlan(bool retained,int mode,std::string& error)
{
    using namespace videowire;
    using namespace HarmonicMIDI::grid;
    const auto sourceRoot = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() / "content/shader-packs";
    const auto read = [&](const char* path) {
        std::ifstream input(sourceRoot / path,std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(input),std::istreambuf_iterator<char>());
    };
    const auto generatorSource = read("arbit-isf-starters/shaders/plasma.fs");
    const auto filterSource = read("vidvox-isf/ISF/RGB Invert.fs");
    const auto transitionSource = read("vidvox-isf/ISF/Fade.fs");
    check(!generatorSource.empty() && !filterSource.empty() && !transitionSource.empty(),"imported Frame prepared admission loads actual bundled shader sources");
    if (generatorSource.empty() || filterSource.empty() || transitionSource.empty()) return {};
    const auto escape = [](const std::string& value) {
        std::string result;
        for (const char c : value) switch(c) {
            case '&': result += "&amp;"; break; case '<': result += "&lt;"; break;
            case '>': result += "&gt;"; break; case '\"': result += "&quot;"; break;
            case '\'': result += "&apos;"; break; default: result += c; break;
        }
        return result;
    };
    const auto shaderPayload = [&](const char* role,const char* pack,const char* program,const std::string& source) {
        const auto* entry = shadercatalog::find(pack,program);
        check(entry != nullptr,"imported Frame shader identities exist in the current catalog");
        return "<FlatShaderBridge schemaVersion=\"2\" role=\"" + std::string(role)
            + "\" language=\"isf\" source=\"" + escape(source) + "\" sourceSha256=\"" + videohelper::sha256Text(source)
            + "\" catalogPackId=\"" + pack + "\" catalogProgramId=\"" + program + "\" parameters=\""
            + escape(entry ? shadercatalog::defaultParameterWire(*entry) : std::string{}) + "\"/>";
    };
    const auto generator = shaderPayload("generator","arbit-isf-starters-v1","plasma",generatorSource);
    const auto filter = shaderPayload("filter","vidvox-isf","rgb-invert",filterSource);
    shadertransition::Payload transition;
    transition.catalogPackId = "vidvox-isf"; transition.catalogProgramId = "fade";
    transition.source = transitionSource; transition.sourceSha256 = videohelper::sha256Text(transitionSource); transition.progress = .25;
    const auto* transitionEntry = shadercatalog::find("vidvox-isf","fade");
    if (!transitionEntry) { check(false,"imported Frame transition catalog is available"); return {}; }
    transition.parameters = shadertransition::parametersForProgress(*transitionEntry,transition.progress,error);
    const auto transitionPayload = shadertransition::serialize(transition,error);
        CompiledVisualLayerPlan plan; plan.clipId = 93; plan.structuralRevision = 9; plan.producerValidated = true;
        visualimportedscenerender::Request request; request.structuralRevision = 9; request.evaluationRevision = 17;
        request.renderStableId = 12;
        const int source = retained ? 6 : 70, render = 11, terminal = 82;
        request.sourceStableId = static_cast<std::uint64_t>(source) + 1;
        if (retained)
        {
            Visual3DScene lighting; lighting.id = {7}; lighting.objectCount = lighting.materialCount = lighting.cameraCount = 1;
            lighting.objects[0].id = {1}; lighting.objects[0].material = {1}; lighting.materials[0].id = {1};
            lighting.activeCamera = {50}; lighting.cameras[0].id = {50}; lighting.cameras[0].transform.translation.z = 4;
            const auto scene = videohelper::tests::generatedSurfaceScene(lighting,error);
            check(scene.has_value(),"retained imported admission fixture uses actual admitted Geometry Core scene transport");
            if (!scene) return {};
            request.sceneSnapshot = std::make_shared<const Visual3DScene>(*scene);
        }
        else
        {
            request.asset = {"model-asset-1",3,std::string(64,'a'),"model/gltf-binary",4096}; request.sceneIndex = 0;
        }
        surfacematerialbinding::ImportedSceneMaterialRequest material;
        material.scene = retained ? request.sceneSnapshot->id : Scene3DId{1}; material.sceneSnapshot = request.sceneSnapshot;
        material.sceneRevision = material.structuralRevision = material.evaluationRevision = material.programRevision = 9;
        material.binding.targetKind = surfacematerialbinding::BindingTargetKind::ObjectOverride; material.binding.object = {1};
        material.binding.surfaceMaterialRevision = 9; material.version = surfacematerialbinding::kGraphFrameWireVersion;
        material.program = surfacematerialstarterfixture::kPrograms[0].program(); material.program.textureSlotCount = 1;
        for (auto& op : material.program.operations) if (op.id == material.program.outputs[9]) op.unsignedLiteral = 1;
        const auto admitted = surfacematerial::admit(material.program,error);
        if (!admitted) { check(false,"imported Frame material fixture has admitted Surface IR"); return {}; }
        material.binding.surfaceMaterialDigest = admitted->structuralDigest();
        const materialframe::Endpoint endpoint {mode == 6 ? 94 : mode == 2 || mode == 3 || mode == 4 ? 92 : 91,
            mode == 3 ? 2 : mode == 2 || mode == 4 || mode == 6 ? 1 : 0};
        surfacematerialbinding::TextureSlotBinding texture; texture.source = surfacematerialbinding::TextureSourceKind::GraphFrame;
        texture.graphFrame = endpoint; material.binding.textures = {texture}; request.material = material;
        const auto addOperation = [&](int id,const std::string& kind,const char* capability,const std::string& payload = {}) {
            plan.nodeIds.push_back(id); plan.nodeKinds.push_back(kind); plan.operations.push_back({id,kind,capability,payload});
        };
        const auto imagePort = [&](int id,int port,const char* direction) {
            plan.ports.push_back({id,port,1,direction,"frame","image","rgba8","sRGB","sRGB","straight"});
        };
        const auto controlPort = [&](int id,int port,int channels,const char* direction,const std::string& type) {
            plan.ports.push_back({id,port,channels,direction,"control",type,"unspecified","unspecified"});
        };
        addOperation(source,retained ? "visual.3d.scene.retained" : "visual.3d.imported-animation","control-eval");
        if (!retained) plan.operations.back().backendCapability = visualimportedscenerender::kSourceBackendCapability;
        if (retained) controlPort(source,0,1,"out","scene3D");
        else
        {
            const char* types[] = {"mesh","skeleton","morphTargets","animationClip","scene3D"};
            for (int port = 0; port < 5; ++port) controlPort(source,port,1,"out",types[port]);
            for (std::size_t index = 0; index < material.program.operations.size(); ++index)
            {
                const auto& op = material.program.operations[index]; const int id = 100 + static_cast<int>(index);
                const auto token = surfacematerial::token(op.resultType);
                addOperation(id,"visual.surface.constant." + std::string(token),"control-eval");
                const int channels = op.resultType == surfacematerial::ValueType::Vec3 ? 3 : 1;
                controlPort(id,0,channels,"out",std::string(op.resultType == surfacematerial::ValueType::UInt ? "integer" : token));
                controlPort(terminal,static_cast<int>(index),channels,"in",std::string(op.resultType == surfacematerial::ValueType::UInt ? "integer" : token));
                plan.edges.push_back({id,0,terminal,static_cast<int>(index)});
            }
            addOperation(terminal,"visual.surface.material","control-eval"); controlPort(terminal,10,1,"out","material");
            controlPort(terminal,11,3,"in","vec3"); controlPort(render,2,1,"in","material");
            plan.edges.push_back({terminal,10,render,2});
        }
        addOperation(91,mode == 1 || mode == 6 ? "visual.shader.generator" : mode == 5 ? "video.layer.source" : "video.source",
            mode == 1 || mode == 6 ? "native-gpu" : "source-decode",
            mode == 1 || mode == 6 ? generator : mode == 5 ? materialframesource::encode(94) : videosourcecolor::encode(videosourcecolor::Policy{4000.0}));
        imagePort(91,0,"out");
        if (mode == 2 || mode == 3 || mode == 4 || mode == 6)
        {
            const auto kind = mode == 3 ? shadertransition::operationKind : mode == 4 ? "visual.frame-delay" : "visual.shader.filter";
            visualtemporaloperation::Payload temporal; temporal.mode = visualtemporaloperation::Mode::frameDelay;
            temporal.historyLength = 3; temporal.decay = 0; temporal.zoom = 1;
            addOperation(92,kind,"native-gpu",mode == 3 ? transitionPayload : mode == 4 ? visualtemporaloperation::serialize(temporal) : filter);
            imagePort(92,0,"in"); imagePort(92,mode == 3 ? 2 : 1,"out"); plan.edges.push_back({91,0,92,0});
            if (mode == 3) {
                addOperation(93,"video.layer.source","source-decode",materialframesource::encode(94)); imagePort(93,0,"out");
                imagePort(92,1,"in"); plan.edges.push_back({93,0,92,1});
            }
            if (mode == 6) {
                addOperation(94,"visual.shader.filter","native-gpu",filter); imagePort(94,0,"in"); imagePort(94,1,"out");
                plan.edges.push_back({92,1,94,0});
            }
        }
        addOperation(render,"visual.3d.render","native-gpu",visualimportedscenerender::encode(request));
        controlPort(render,0,1,"in","scene3D"); imagePort(render,1,"out"); imagePort(retained ? render : terminal,12,"in");
        plan.edges.insert(plan.edges.begin(),{source,retained ? 0 : 4,render,0});
        plan.edges.push_back({endpoint.node,endpoint.port,retained ? render : terminal,12});
        plan.descriptorCount = static_cast<int>(plan.ports.size()); plan.operationCount = static_cast<int>(plan.operations.size());
        plan.frameOutputCount = plan.peakLiveFrameCount = plan.allocatedFrameSlotCount = 5;
        return plan;
}

struct ResolverSegment
{
    int clipId = 0;
    std::string path;
    double inSec = 0, outSec = 40, displayStartSec = 0, rate = 1, mediaFps = 60;
};
struct ResolverDecoded
{
    struct Hdr { int width = 0, height = 0; } hdrFrame;
    int texW = 640, texH = 360;
    std::shared_ptr<FakeFrame> image;
};
struct ResolverRenderer
{
    int targetWidth = 640, targetHeight = 360, shaderCalls = 0, temporalCalls = 0;
    bool failShader = false;
    std::vector<std::weak_ptr<const arbitgpu::NativeFixtureSceneFrame>> weak;
    std::vector<std::pair<int,int>> transitionInputs;
    std::vector<double> shaderTimes;
    using WeakImage = std::weak_ptr<const arbitgpu::NativeFixtureSceneFrame>;
    std::map<int,std::pair<WeakImage,WeakImage>> filterOwners;
    std::vector<std::pair<WeakImage,WeakImage>> transitionOwners;
    int outputWidth() const { return targetWidth; }
    int outputHeight() const { return targetHeight; }
    static int content(const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& image)
    { const auto* frame = dynamic_cast<const FakeFrame*>(image.get()); return frame ? frame->content : 0; }
    static int content(const std::shared_ptr<const void>& image)
    { return content(std::static_pointer_cast<const arbitgpu::NativeFixtureSceneFrame>(image)); }
    std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> image(int marker)
    {
        auto value = std::make_shared<FakeFrame>(); value->content = marker;
        value->frameWidth = value->descriptorWidth = targetWidth; value->frameHeight = targetHeight;
        weak.push_back(value); return value;
    }
    std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> leaseShaderFrame(
        const videorender::LayerDesc& layer,std::string& error)
    {
        ++shaderCalls; shaderTimes.push_back(layer.shaderClock.timeSec);
        if (failShader) { error = "fixture shader lease refused"; return {}; }
        const auto to = content(layer.nativeTextureOwner);
        if (layer.fromLayer)
        {
            transitionInputs.emplace_back(content(layer.fromLayer->nativeTextureOwner),to);
            transitionOwners.emplace_back(std::static_pointer_cast<const arbitgpu::NativeFixtureSceneFrame>(layer.fromLayer->nativeTextureOwner),
                std::static_pointer_cast<const arbitgpu::NativeFixtureSceneFrame>(layer.nativeTextureOwner));
        }
        auto output = image(layer.fromLayer ? content(layer.fromLayer->nativeTextureOwner)*1000 + to
            : to ? to + 1 : 500);
        if (!layer.fromLayer && layer.nativeTextureOwner)
            filterOwners.emplace(layer.shaderOperationPlan->operations.front().nodeId,
                std::make_pair(std::static_pointer_cast<const arbitgpu::NativeFixtureSceneFrame>(layer.nativeTextureOwner),output));
        return output;
    }
    std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> leaseTemporalFrame(
        const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& current,
        const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& previous,
        const visualtemporaloperation::Payload&,float,std::string&)
    { ++temporalCalls; return image(content(current) + content(previous)); }
};

void testProductionMaterialFrameResolver()
{
    using namespace videohelper::importedscene;
    using namespace videowire;
    for (bool retained : {false,true})
    for (auto use : {NativeImportedSceneRenderUse::Preview,NativeImportedSceneRenderUse::Export})
    for (int mode = 0; mode < 13; ++mode)
    {
        std::string error;
        auto plan = nativeResolverPlan(retained,mode >= 7 ? (mode == 12 ? 3 : mode == 11 ? 1 : 4) : mode,error);
        if (mode >= 7 && mode <= 10)
        {
            auto& operation = *std::find_if(plan.operations.begin(),plan.operations.end(),[](const auto& op){return op.nodeId==92;});
            visualtemporaloperation::Payload payload;
            const auto kind = mode == 7 ? "visual.stutter" : mode == 8 ? "visual.feedback"
                : mode == 9 ? "visual.echo" : "visual.long-exposure";
            payload.mode = mode == 7 ? visualtemporaloperation::Mode::stutter : mode == 8 ? visualtemporaloperation::Mode::feedback
                : mode == 9 ? visualtemporaloperation::Mode::echo : visualtemporaloperation::Mode::longExposure;
            payload.historyLength = 3; payload.holdFrames = 3;
            payload.decay = mode == 8 ? .9f : 0; payload.zoom = 1;
            operation.kind = kind; operation.payloadXml = visualtemporaloperation::serialize(payload);
            plan.nodeKinds[std::find(plan.nodeIds.begin(),plan.nodeIds.end(),92)-plan.nodeIds.begin()] = kind;
        }
        if (mode == 11)
        {
            const std::string source = "void mainImage(out vec4 c, in vec2 p) { c = vec4(0.25, 0.5, 1.0, 1.0); }";
            programmableruntime::SessionSecret secret{}; secret.fill(21);
            programmableruntime::Grant grant; grant.kind = programmableruntime::PayloadKind::shader;
            grant.fingerprint = programmableruntime::fingerprint(grant.kind,source);
            grant.sessionGeneration = 7001; grant.nonce = 91; grant.issuedAtMs = programmableadmission::nowMs();
            grant.approved = true; grant.cpuMs = programmableruntime::defaultCpuMs;
            grant.gpuMs = programmableruntime::defaultGpuMs; grant.memoryMiB = programmableruntime::defaultMemoryMiB;
            grant.mac = programmableruntime::sign(secret,grant,grant.kind);
            programmableadmission::verifier().reset(secret,grant.sessionGeneration);
            auto& operation = *std::find_if(plan.operations.begin(),plan.operations.end(),[](const auto& op){return op.nodeId==91;});
            operation.kind = "visual.shader.custom";
            plan.nodeKinds[std::find(plan.nodeIds.begin(),plan.nodeIds.end(),91)-plan.nodeIds.begin()] = operation.kind;
            operation.payloadXml = "<FlatShaderBridge schemaVersion=\"2\" role=\"generator\" language=\"glsl\" source=\""
                + source + "\" sourceSha256=\"" + videohelper::sha256Text(source) + "\"/>";
            operation.runtimeGrantJson = programmableadmission::toJson(grant).dump();
        }
        if (mode == 12)
        {
            // Equal direct producers remain refused by real shader admission.
            plan.edges.erase(std::remove_if(plan.edges.begin(),plan.edges.end(),[](const auto& edge){return edge.fromNodeId==93;}),plan.edges.end());
            plan.edges.push_back({91,0,92,1});
            const auto index = std::find(plan.nodeIds.begin(),plan.nodeIds.end(),93)-plan.nodeIds.begin();
            plan.nodeIds.erase(plan.nodeIds.begin()+index); plan.nodeKinds.erase(plan.nodeKinds.begin()+index);
            plan.operations.erase(plan.operations.begin()+index);
            plan.ports.erase(std::remove_if(plan.ports.begin(),plan.ports.end(),[](const auto& port){return port.nodeId==93;}),plan.ports.end());
            plan.operationCount = plan.operations.size(); plan.descriptorCount = plan.ports.size();
            VisualLayerExecution refused;
            check(!compileVisualLayerExecution(plan,refused,error) && !refused.geometryFrameSchedule,
                "equal direct transition producers retain their production admission refusal");
            error.clear();
            // One leaf fans out through two distinct filters. The transition
            // consumes their actual outputs under the unchanged shader cap.
            const auto filterPlan = nativeResolverPlan(retained,2,error);
            const auto filter = *std::find_if(filterPlan.operations.begin(),filterPlan.operations.end(),
                [](const auto& operation){return operation.nodeId==92;});
            plan.edges.erase(std::remove_if(plan.edges.begin(),plan.edges.end(),
                [](const auto& edge){return edge.toNodeId==92;}),plan.edges.end());
            for (int node : {95,96})
            {
                auto operation = filter; operation.nodeId = node;
                plan.operations.insert(plan.operations.end()-1,operation);
                plan.nodeIds.insert(plan.nodeIds.end()-1,node);
                plan.nodeKinds.insert(plan.nodeKinds.end()-1,operation.kind);
                plan.ports.push_back({node,0,1,"in","frame","image","rgba8","sRGB","sRGB","straight"});
                plan.ports.push_back({node,1,1,"out","frame","image","rgba8","sRGB","sRGB","straight"});
                plan.edges.push_back({91,0,node,0});
                plan.edges.push_back({node,1,92,node==95 ? 0 : 1});
            }
            plan.operationCount = plan.operations.size(); plan.descriptorCount = plan.ports.size();
        }
        VisualLayerExecution compiled;
        {
            const bool conditionOk = compileVisualLayerExecution(plan,compiled,error);
            check(conditionOk, "production resolver matrix admits real authored payloads: " + error);
        }
        if (!compiled.geometryFrameSchedule) continue;
        const auto admittedPlan=plan;
        const auto admittedExecution=compiled;
        for (const auto timing : {std::pair<double,double>{12.125,24}, {12.13,24}, {11.125,29.97}, {12.13,29.97}, {12.125,24}})
        for (const int fault : mode == 0 ? std::vector<int>{0,3,4,5,6,8,9,10,11,28,29}
            : mode == 5 ? std::vector<int>{0,1,2,7,26,27}
            : mode == 2 ? std::vector<int>{0,13,14,18}
            : mode == 3 ? std::vector<int>{0,12,15,19,20,21,22,23}
            : mode == 4 ? std::vector<int>{0,16,17}
            : mode == 8 ? std::vector<int>{0,24,25} : std::vector<int>{0})
        {
            error.clear();
            auto plan=admittedPlan;
            auto compiled=admittedExecution;
            const double project = fault==25 ? 19.0 : timing.first, fps = timing.second, source = 7.25+(project-10)*2;
            auto current = ResolverSegment {93,"current.mp4",7.25,40,10,2,60};
            auto saved = ResolverSegment {94,"saved.mp4",3.5,20,8,.5,25};
            if (fault==3) current.rate=0;
            if (fault==26) saved.rate=std::numeric_limits<double>::quiet_NaN();
            if (fault==27) saved.displayStartSec=project+1;
            if (fault==7) saved.outSec=3.5+(project-8)*.5;
            std::vector<ResolverSegment> segments {current,saved};
            if (fault==1) segments.pop_back();
            if (fault==2) segments.push_back(saved);
            ResolverRenderer renderer;
            if (fault==13) renderer.targetWidth=4097;
            if (fault==14) renderer.failShader=true;
            if (fault==24) renderer.targetWidth=renderer.targetHeight=4096;
            ResolverDecoded decoded;
            if (fault==24) decoded.texW=decoded.texH=4096;
            if (fault==29) decoded.texW=4097;
            std::vector<std::tuple<int,std::string,double>> decodes;
            std::vector<std::weak_ptr<const arbitgpu::NativeFixtureSceneFrame>> sources;
            int leases = 0;
            auto live = std::make_shared<bool>(true);
            const auto activity = [live](std::string& failure) { if (*live) return true; failure="fixture invocation retired"; return false; };
            VisualPlanEvaluationContext context; context.projectGeneration=12; context.helperGeneration=34;
            context.materialFrameClock = [fps](double seconds,auto& frame,std::string& failure) {
                return videohelper::importedanimation::importedAnimationFrameIdentity(seconds,fps,frame,failure);
            };
            videorender::LayerDesc inputs; inputs.materialFrameStartSeconds = 10;
            inputs.shaderClock.timeSec = project; inputs.audioPresent = true; inputs.audioFeatures.bands = {float(project)};
            inputs.materialInputsAt = [](double at,auto& layer,auto& values,std::string&) {
                layer.shaderClock.timeSec=at; layer.audioPresent=true; layer.audioFeatures.bands={float(at)}; values.clear(); return true;
            };
            if (fault==16) {
                auto history=std::make_shared<videorender::ParticleHistorySource>();
                history->parameterAt=[project](const auto&,double at,double& value,std::string&) {
                    value=at > project-.03 && at < project-.01 ? 1 : 0; return true;
                }; inputs.particleHistory=history;
            }
            if (fault==17) {
                inputs.materialInputsAt=[](double,auto&,auto&,std::string& failure){ failure="fixture historical audio/script parameters unavailable";return false; };
                auto generated=nativeResolverPlan(retained,1,error);
                auto& original=*std::find_if(plan.operations.begin(),plan.operations.end(),[](const auto& op){return op.nodeId==91;});
                original=*std::find_if(generated.operations.begin(),generated.operations.end(),[](const auto& op){return op.nodeId==91;});
                plan.nodeKinds[std::find(plan.nodeIds.begin(),plan.nodeIds.end(),91)-plan.nodeIds.begin()]=original.kind;
                check(compileVisualLayerExecution(plan,compiled,error),"history-negative keeps the real generated source admission");
            }
            const auto decode = [&](const ResolverSegment& selected,double at,const MaterialFrameEvaluation& leaf,std::string&) -> ResolverDecoded* {
                visualdeformation::RationalFrameTime expectedDisplay;
                context.materialFrameClock(project,expectedDisplay,error);
                const double anchored = leaf.frame.frame==expectedDisplay.frame ? project
                    : project+(leaf.frame.frame-expectedDisplay.frame)*double(expectedDisplay.rateDenominator)/expectedDisplay.rateNumerator;
                const int wantNode = selected.clipId==94 ? (mode==5 ? 91 : 93) : 91;
                const bool leafClip = leaf.clipId==93;
                const bool leafRevisions = leaf.structuralRevision==9 && leaf.evaluationRevision==17;
                const bool leafInvocation = leaf.projectGeneration==12 && leaf.helperGeneration==34 && leaf.use==use;
                const bool leafClock = leaf.frame.rateNumerator==expectedDisplay.rateNumerator
                    && leaf.frame.rateDenominator==expectedDisplay.rateDenominator && leaf.frame.frame<=expectedDisplay.frame;
                const bool leafEndpoint = leaf.endpoint.port==0 && leaf.endpoint.node==wantNode;
                const bool leafOk = leafClip && leafRevisions && leafInvocation && leafClock && leafEndpoint;
                check(leafOk,
                    "every actual factory decoded leaf carries its endpoint, internal display clock and full invocation identity");
                if (!leafOk)
                {
                    const std::string pattern = std::string("leaf:") + (leafClip ? '1' : '0') + (leafRevisions ? '1' : '0')
                        + (leafInvocation ? '1' : '0') + (leafClock ? '1' : '0') + (leafEndpoint ? '1' : '0');
                    if (probeFirstTime(pattern))
                        std::cerr << "PROBE: leaf mode=" << mode << " fault=" << fault << " selectedClip=" << selected.clipId
                            << " clipId=" << leaf.clipId << " (want 93)"
                            << " structuralRevision=" << leaf.structuralRevision << " (want 9)"
                            << " evaluationRevision=" << leaf.evaluationRevision << " (want 17)"
                            << " projectGeneration=" << leaf.projectGeneration << " (want 12)"
                            << " helperGeneration=" << leaf.helperGeneration << " (want 34)"
                            << " useMatches=" << (leaf.use==use)
                            << " rateNumerator=" << leaf.frame.rateNumerator << " (want " << expectedDisplay.rateNumerator << ")"
                            << " rateDenominator=" << leaf.frame.rateDenominator << " (want " << expectedDisplay.rateDenominator << ")"
                            << " frame=" << leaf.frame.frame << " (want <=" << expectedDisplay.frame << ")"
                            << " endpoint.node=" << leaf.endpoint.node << " (want " << wantNode << ")"
                            << " endpoint.port=" << leaf.endpoint.port << " (want 0)\n";
                }
                const double expectedMedia=selected.clipId==94 ? 3.5+(anchored-8)*.5
                    : leaf.frame.frame==expectedDisplay.frame ? source : source+(anchored-project)*2;
                check(std::abs(at-expectedMedia)<1e-11,"every decoded leaf uses its own anchored trim/rate mapping");
                check(selected.mediaFps!=fps,"selected decoder retains its independently different media frame rate");
                decodes.emplace_back(selected.clipId,selected.path,at);
                if (fault==8) return nullptr;
                decoded.image = std::make_shared<FakeFrame>(); decoded.image->content=selected.clipId==93 ? 93 : 94;
                decoded.image->image=selected.clipId==93 ? 931 : 941; decoded.image->view=selected.clipId==93 ? 932 : 942;
                if (fault==10) decoded.image->descriptorWidth=0;
                if (fault==28) decoded.image->api="unsupported";
                if (fault==12 && selected.clipId==94) decoded.image->contextIdentity=999;
                if (fault==18) { decoded.hdrFrame.width=640;decoded.hdrFrame.height=360;decoded.image->linearColor=true; }
                if (fault==24) { decoded.image->frameWidth=decoded.image->descriptorWidth=4096;decoded.image->frameHeight=4096; }
                sources.push_back(decoded.image); return &decoded;
            };
            const auto lease = [&](ResolverDecoded& value,const auto& policy,std::string&) {
                check(value.image->content==94 ? !policy.hlgNominalDisplayPeakNits : policy.hlgNominalDisplayPeakNits==4000.0,
                    "source policy stays bound to the selected decoded leaf, without current HLG inheritance");
                ++leases; std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> image=value.image; value.image.reset();
                if (fault==9 || fault==11) return std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>{};
                if (fault==15) *live=false;
                return image;
            };
            TemporalSamplingCommitTransaction transaction;
            auto resolver = use == NativeImportedSceneRenderUse::Preview
                ? ImportedMaterialFrameResolver::make<NativeImportedSceneRenderUse::Preview>(renderer,plan,compiled,current,segments,
                    fault==5 ? std::numeric_limits<double>::quiet_NaN() : project,
                    fault==4 ? std::numeric_limits<double>::quiet_NaN() : fault==6 ? current.outSec : source,
                    fps,inputs,nullptr,context,transaction,activity,decode,lease)
                : ImportedMaterialFrameResolver::make<NativeImportedSceneRenderUse::Export>(renderer,plan,compiled,current,segments,
                    fault==5 ? std::numeric_limits<double>::quiet_NaN() : project,
                    fault==4 ? std::numeric_limits<double>::quiet_NaN() : fault==6 ? current.outSec : source,
                    fps,inputs,nullptr,context,transaction,activity,decode,lease);
            FakeExecution execution; videorender::LayerDesc layer; std::vector<FakeReceipt> owners; VisualImportedScenePlanCache cache;
            if (fault==0 && mode==0) {
                visualdeformation::RationalFrameTime expectedFrame;
                videohelper::importedanimation::importedAnimationFrameIdentity(source,fps,expectedFrame,error);
                const MaterialFrameEvaluation expected{compiled.geometryFrameSchedule->outputs.front(),expectedFrame,93,9,17,12,34,use};
                for (int field=0;field<11;++field) {
                    auto stale=expected;
                    if (field==0) {++stale.endpoint.node;} if (field==1) {++stale.endpoint.port;}
                    if (field==2) {++stale.frame.frame;} if (field==3) {++stale.frame.rateNumerator;}
                    if (field==4) {++stale.frame.rateDenominator;} if (field==5) {++stale.clipId;}
                    if (field==6) {++stale.structuralRevision;} if (field==7) {++stale.evaluationRevision;}
                    if (field==8) {++stale.projectGeneration;} if (field==9) {++stale.helperGeneration;}
                    if (field==10) stale.use=use==NativeImportedSceneRenderUse::Preview ? NativeImportedSceneRenderUse::Export : NativeImportedSceneRenderUse::Preview;
                    MaterialFrameReceipt refused;
                    check(!resolver.resolve(stale,refused,error) && !refused.nativeFrame && decodes.empty() && leases==0,
                        "each full outer identity mismatch refuses before media work");
                }
            }
            execution.succeed=fault!=19;
            if (fault==20) execution.nextFrame->frameWidth=1;
            auto incumbent=std::make_shared<FakeFrame>();layer.nativeTextureOwner=incumbent;layer.texture=718;
            const auto prepared=prepareVisualImportedSceneLayerAtTime({plan},93,640,360,source,fps,use,&execution,&cache,layer,owners,error,
                nullptr,{12,34},nullptr,false,nullptr,project,nullptr,&resolver);
            const bool settlementFailure=fault==21 || fault==22 || fault==23;
            if (fault && !settlementFailure) {
                check(prepared==VisualImportedScenePreparation::rejected && owners.empty()
                    && layer.nativeTextureOwner==incumbent && layer.texture==718,
                    "production resolver failure preserves incumbent and publishes no imported receipt: fault="+std::to_string(fault)+" "+error);
                if (fault!=19 && fault!=20) check(execution.previewCalls==0 && execution.exportCalls==0,"rejected dependency issues no scene draw");
                if (fault==15) check(decodes.size()==1 && leases==1,"cancellation between transition leaves prevents second decode/lease/draw");
                if (fault==13) check(renderer.shaderCalls==0,"oversized generated target refuses before its native lease");
                if (fault==24) check(leases<=32 && renderer.temporalCalls==0,"logical memo exhaustion refuses before another reserved lease");
                decoded.image.reset();execution.lastRequest={};transaction.discard();
                for(auto& weak:sources) check(weak.expired(),"failed source/intermediate staging releases all unpublished owners");
                for(auto& weak:renderer.weak) check(weak.expired(),"failed DAG releases every generated owner");
                continue;
            }
            check(prepared==VisualImportedScenePreparation::rendered,
                "actual Preview/Export factory feeds ordinary/retained imported draw: " + error);
            if (settlementFailure) {
                execution.lastRequest={};owners.clear();
                TemporalSamplingCommitTransaction pending(std::move(transaction));
                if(fault==23) { *live=false; check(!pending.commit(error),"retired invocation refuses transaction settlement"); }
                else pending.discard();
                for(auto& weak:sources) check(weak.expired(),"nonpresented Preview or failed readback/encode releases sources");
                for(auto& weak:renderer.weak) check(weak.expired(),"nonpublication releases generated dependencies");
                continue;
            }
            const auto& request=execution.lastRequest;
            visualdeformation::RationalFrameTime outer,display;
            videohelper::importedanimation::importedAnimationFrameIdentity(source,fps,outer,error); context.materialFrameClock(project,display,error);
            const bool compositionClocks = request.frame==outer && outer.frame!=display.frame;
            const bool compositionRevisions = request.structuralRevision==9 && request.evaluationRevision==17;
            const bool compositionOk = compositionClocks && compositionRevisions;
            check(compositionOk, "production composition conserves distinct source/display clocks and distinct revisions");
            if (!compositionOk)
            {
                const std::string pattern = std::string("composition:") + (compositionClocks ? '1' : '0')
                    + (compositionRevisions ? '1' : '0');
                if (probeFirstTime(pattern))
                    std::cerr << "PROBE: composition frame=" << request.frame.frame << " outer=" << outer.frame
                        << " display=" << display.frame << " (want outer!=display)"
                        << " requestRate=" << request.frame.rateNumerator << "/" << request.frame.rateDenominator
                        << " outerRate=" << outer.rateNumerator << "/" << outer.rateDenominator
                        << " structuralRevision=" << request.structuralRevision << " (want 9)"
                        << " evaluationRevision=" << request.evaluationRevision << " (want 17)\n";
            }
            check(!owners.empty() && owners.front().materialFrameTexture==request.runtimeInputs.materialFrameTexture,
                "final Frame owner is retained by the imported scene receipt");
            if (mode==1 || mode==11) check(renderer.shaderTimes==std::vector<double>{project},
                "current shader reads the captured input clock before destination rebuild");
            if (mode==0) check(decodes.size()==1 && std::get<0>(decodes[0])==93 && std::get<2>(decodes[0])==source,
                "current decode preserves exact off-grid seconds independent of media fps");
            if (mode==5) check(decodes.size()==1 && std::get<0>(decodes[0])==94 && std::get<1>(decodes[0])=="saved.mp4"
                && std::abs(std::get<2>(decodes[0])-(3.5+(project-8)*.5))<1e-12
                && sources.front().lock()==request.runtimeInputs.materialFrameTexture && ResolverRenderer::content(request.runtimeInputs.materialFrameTexture)==94,
                "saved-only source uses the saved decoded content and owner with zero current decode");
            if (mode==3) check(renderer.transitionInputs==std::vector<std::pair<int,int>>{{93,94}}
                && decodes.size()==2,"transition inputs retain selected current/saved order");
            if (mode==12)
            {
                check(compiled.geometryFrameShaders && compiled.geometryFrameShaders->operations.size()==3
                    && ShaderOperationPlan::maximumOperations==4 && renderer.shaderCalls==3
                    && decodes.size()==1 && leases==1 && renderer.transitionInputs==std::vector<std::pair<int,int>>{{94,94}},
                    "admitted shared leaf fan-out uses one decode/lease and three shaders under the four-operation cap");
                check(renderer.filterOwners.size()==2 && renderer.filterOwners.count(95) && renderer.filterOwners.count(96)
                    && renderer.transitionOwners.size()==1,
                    "both actual filter nodes and transition input owners are observed");
                if (renderer.filterOwners.count(95) && renderer.filterOwners.count(96) && renderer.transitionOwners.size()==1)
                {
                    const auto& left=renderer.filterOwners.at(95); const auto& right=renderer.filterOwners.at(96);
                    check(left.first.lock()==sources.front().lock() && right.first.lock()==sources.front().lock()
                        && left.second.lock() && right.second.lock() && left.second.lock()!=right.second.lock()
                        && renderer.transitionOwners.front().first.lock()==left.second.lock()
                        && renderer.transitionOwners.front().second.lock()==right.second.lock(),
                        "two distinct filter outputs from the same decoded owner reach their exact transition sides");
                }
            }
            if (mode==4) check(decodes.size()==1 && std::abs(std::get<2>(decodes[0])-(source-3.0*display.rateDenominator/display.rateNumerator*2))<1e-11,
                "Frame Delay applies the anchored project period before current media rate");
            for (auto& weak : sources) check(!weak.expired(),"decoded dependencies survive scene submission");
            for (auto& weak : renderer.weak) check(!weak.expired(),"all generated/intermediate dependencies survive scene submission");
            // Simulate Export's actual transaction move across per-frame owner
            // clearing. This is the transaction used by production, not a fixture ledger.
            TemporalSamplingCommitTransaction pending(std::move(transaction));
            execution.lastRequest={}; owners.clear();
            for (auto& weak : sources) check(!weak.expired(),"dependency bundle survives pending settlement");
            for (auto& weak : renderer.weak) check(!weak.expired(),"every filter/transition owner survives the real transaction move");
            {
                const bool conditionOk = pending.commit(error);
                check(conditionOk, "successful publication/readback/encoder settlement releases the bundle: " + error);
            }
            for (auto& weak : sources) check(weak.expired(),"source owner retires after transaction settlement");
            for (auto& weak : renderer.weak) check(weak.expired(),"intermediate owner retires after transaction settlement");
        }
    }
}

void testProductionDiffractionFrameCompatibility()
{
    using namespace videohelper::importedscene;
    using namespace videowire;
    auto plan=importedMaterialFramePlan(true);
    std::string error; VisualLayerExecution compiled;
    check(compileVisualLayerExecution(plan,compiled,error) && !compiled.geometryFrameSchedule,
        "direct diffraction keeps its existing decoder-only admission");
    for(auto use:{NativeImportedSceneRenderUse::Preview,NativeImportedSceneRenderUse::Export}) {
        ResolverRenderer renderer; ResolverDecoded decoded;
        ResolverSegment current{7,"diffraction-current.mp4",7.25,40,10,2,60};
        VisualPlanEvaluationContext context;context.projectGeneration=12;context.helperGeneration=34;
        context.materialFrameClock=[](double at,auto& frame,std::string& failure){return videohelper::importedanimation::importedAnimationFrameIdentity(at,24,frame,failure);};
        videorender::LayerDesc inputs;inputs.materialFrameStartSeconds=10;
        TemporalSamplingCommitTransaction transaction;
        int decodes=0;
        auto decode=[&](const auto& selected,double at,const auto&,std::string&)->ResolverDecoded* {
            ++decodes;check(selected.clipId==7 && selected.path==current.path && at==11.51,
                "direct diffraction preserves the current exact off-grid media sample");
            decoded.image=std::make_shared<FakeFrame>();return &decoded;
        };
        auto lease=[&](auto& value,const auto&,std::string&){auto image=value.image;value.image.reset();return image;};
        const auto active=[](std::string&){return true;};
        auto resolver=use==NativeImportedSceneRenderUse::Preview
            ? ImportedMaterialFrameResolver::make<NativeImportedSceneRenderUse::Preview>(renderer,plan,compiled,current,std::vector<ResolverSegment>{current},12.13,11.51,24,inputs,nullptr,context,transaction,active,decode,lease)
            : ImportedMaterialFrameResolver::make<NativeImportedSceneRenderUse::Export>(renderer,plan,compiled,current,std::vector<ResolverSegment>{current},12.13,11.51,24,inputs,nullptr,context,transaction,active,decode,lease);
        FakeExecution execution;FakeLayer layer;std::vector<FakeReceipt> owners;VisualImportedScenePlanCache cache;
        {
            const bool conditionOk = prepareVisualImportedSceneLayerAtTime({plan},7,640,360,11.51,24,use,&execution,&cache,layer,owners,error,
            nullptr,{12,34},nullptr,false,nullptr,12.13,nullptr,&resolver)==VisualImportedScenePreparation::rendered
            && decodes==1 && renderer.shaderCalls==0 && transaction.commit(error);
            check(conditionOk, "production factory preserves direct current-video diffraction compatibility: "+error);
        }
    }
    // Both released timeline callers clamp the outgoing clip at its trim.
    // Media remains decodable beyond that trim, so this is not an EOF case.
    for (auto use : {NativeImportedSceneRenderUse::Preview,NativeImportedSceneRenderUse::Export})
    {
        const ResolverSegment current{7,"outgoing-trimmed.mp4",7.25,11.5,10,2,60};
        constexpr double project=12.13;
        const double mapped=current.inSec+(project-current.displayStartSec)*current.rate;
        const double captured=std::clamp(mapped,current.inSec,std::max(current.inSec,current.outSec));
        check(std::abs(mapped-11.51)<1e-12 && captured==11.5
            && project>current.displayStartSec+(current.outSec-current.inSec)/current.rate,
            "outgoing trim oracle separates caller-held source 11.5 from mapped 11.51 after the owner interval");
        VisualPlanEvaluationContext context; context.projectGeneration=12; context.helperGeneration=34;
        // No project replay clock is required by the released direct callback.
        videorender::LayerDesc inputs; inputs.materialFrameStartSeconds=10;
        std::vector<double> legacySeconds;
        ResolverDecoded legacyDecoded;
        const auto legacyDecode=[&](const ResolverSegment& selected,double at,const auto&,std::string&)->ResolverDecoded* {
            check(selected.clipId==7 && selected.path==current.path,"legacy decoder selects the actual outgoing clip");
            legacySeconds.push_back(at); legacyDecoded.image=std::make_shared<FakeFrame>(); return &legacyDecoded;
        };
        MaterialFrameResolver legacy=[&](const MaterialFrameEvaluation& query,MaterialFrameReceipt& receipt,std::string& failure) {
            videosourcecolor::Policy policy;
            if (!materialframe::sourcePolicy(plan,query.endpoint,policy,failure)) return false;
            auto* decoded=legacyDecode(current,captured,query,failure);
            receipt={query,decoded->image}; decoded->image.reset(); return true;
        };
        FakeExecution legacyExecution; FakeLayer legacyLayer; std::vector<FakeReceipt> legacyOwners;
        VisualImportedScenePlanCache legacyCache;
        check(prepareVisualImportedSceneLayerAtTime({plan},7,640,360,captured,24,use,&legacyExecution,&legacyCache,
            legacyLayer,legacyOwners,error,nullptr,{12,34},&legacy,false,nullptr,project)==VisualImportedScenePreparation::rendered
            && legacySeconds==std::vector<double>{11.5},
            "legacy direct callback receives the exact caller-held outgoing trim sample");
        ResolverRenderer renderer; ResolverDecoded decoded; TemporalSamplingCommitTransaction transaction;
        std::vector<double> actualSeconds;
        std::weak_ptr<const arbitgpu::NativeFixtureSceneFrame> sourceOwner;
        bool live=true;
        const auto active=[&](std::string& failure) { if (live) return true; failure="outgoing fixture retired"; return false; };
        auto decode=[&](const auto& selected,double at,const MaterialFrameEvaluation& leaf,std::string&)->ResolverDecoded* {
            check(selected.clipId==7 && selected.path==current.path && leaf.frame==legacyExecution.lastRequest.frame,
                "direct compatibility selects the actual outgoing owner and preserves the full outer source frame");
            actualSeconds.push_back(at); decoded.image=std::make_shared<FakeFrame>(); sourceOwner=decoded.image; return &decoded;
        };
        auto lease=[&](auto& value,const auto&,std::string&) { auto image=value.image; value.image.reset(); return image; };
        auto resolver=use==NativeImportedSceneRenderUse::Preview
            ? ImportedMaterialFrameResolver::make<NativeImportedSceneRenderUse::Preview>(renderer,plan,compiled,current,
                std::vector<ResolverSegment>{current},project,captured,24,inputs,nullptr,context,transaction,active,decode,lease)
            : ImportedMaterialFrameResolver::make<NativeImportedSceneRenderUse::Export>(renderer,plan,compiled,current,
                std::vector<ResolverSegment>{current},project,captured,24,inputs,nullptr,context,transaction,active,decode,lease);
        visualdeformation::RationalFrameTime outer;
        videohelper::importedanimation::importedAnimationFrameIdentity(captured,24,outer,error);
        const MaterialFrameEvaluation expected{*visualimportedscenerender::materialFrameEndpoint(*compiled.importedSceneRender),
            outer,7,plan.structuralRevision,compiled.importedSceneRender->evaluationRevision,12,34,use};
        for (int field=0;field<11;++field)
        {
            auto stale=expected;
            if (field==0) {++stale.endpoint.node;} if (field==1) {++stale.endpoint.port;}
            if (field==2) {++stale.frame.frame;} if (field==3) {++stale.frame.rateNumerator;}
            if (field==4) {++stale.frame.rateDenominator;} if (field==5) {++stale.clipId;}
            if (field==6) {++stale.structuralRevision;} if (field==7) {++stale.evaluationRevision;}
            if (field==8) {++stale.projectGeneration;} if (field==9) {++stale.helperGeneration;}
            if (field==10) stale.use=use==NativeImportedSceneRenderUse::Preview ? NativeImportedSceneRenderUse::Export : NativeImportedSceneRenderUse::Preview;
            MaterialFrameReceipt refused;
            check(!resolver.resolve(stale,refused,error) && !refused.nativeFrame && actualSeconds.empty(),
                "direct outgoing compatibility refuses every full outer identity mutation before decode");
        }
        MaterialFrameReceipt inactive;
        live=false;
        check(!resolver.resolve(expected,inactive,error) && !inactive.nativeFrame && actualSeconds.empty(),
            "direct outgoing compatibility preserves activity refusal before decode");
        live=true;
        FakeExecution execution; FakeLayer layer; std::vector<FakeReceipt> owners; VisualImportedScenePlanCache cache;
        {
            const bool conditionOk = prepareVisualImportedSceneLayerAtTime({plan},7,640,360,captured,24,use,&execution,&cache,layer,owners,error,
            nullptr,{12,34},nullptr,false,nullptr,project,nullptr,&resolver)==VisualImportedScenePreparation::rendered
            && actualSeconds==legacySeconds && renderer.shaderCalls==0 && renderer.temporalCalls==0
            && execution.lastRequest.frame==outer && sourceOwner.lock()==execution.lastRequest.runtimeInputs.materialFrameTexture;
            check(conditionOk, "actual Preview/Export factory and imported helper preserve the legacy outgoing decoder argument and owner: "+error);
        }
        TemporalSamplingCommitTransaction pending(std::move(transaction)); execution.lastRequest={}; owners.clear();
        check(!sourceOwner.expired(),"outgoing direct owner survives the actual pending transaction move");
        check(pending.commit(error) && sourceOwner.expired(),"outgoing direct owner releases only after publication/readback settlement");

        for (bool retained : {false,true})
        {
            auto scheduledPlan=nativeResolverPlan(retained,0,error); VisualLayerExecution scheduled;
            check(compileVisualLayerExecution(scheduledPlan,scheduled,error) && scheduled.geometryFrameSchedule,
                "scheduled outgoing refusal retains actual admitted ordinary/retained Frame dependency");
            auto scheduledCurrent=current; scheduledCurrent.clipId=93;
            TemporalSamplingCommitTransaction scheduledTransaction;
            auto strict=use==NativeImportedSceneRenderUse::Preview
                ? ImportedMaterialFrameResolver::make<NativeImportedSceneRenderUse::Preview>(renderer,scheduledPlan,scheduled,scheduledCurrent,
                    std::vector<ResolverSegment>{scheduledCurrent},project,captured,24,inputs,nullptr,context,scheduledTransaction,active,decode,lease)
                : ImportedMaterialFrameResolver::make<NativeImportedSceneRenderUse::Export>(renderer,scheduledPlan,scheduled,scheduledCurrent,
                    std::vector<ResolverSegment>{scheduledCurrent},project,captured,24,inputs,nullptr,context,scheduledTransaction,active,decode,lease);
            FakeExecution refusedExecution; FakeLayer refusedLayer; std::vector<FakeReceipt> refusedOwners; VisualImportedScenePlanCache refusedCache;
            const auto before=actualSeconds.size();
            check(prepareVisualImportedSceneLayerAtTime({scheduledPlan},93,640,360,captured,24,use,&refusedExecution,&refusedCache,
                refusedLayer,refusedOwners,error,nullptr,{12,34},nullptr,false,nullptr,project,nullptr,&strict)==VisualImportedScenePreparation::rejected
                && actualSeconds.size()==before && refusedExecution.previewCalls==0 && refusedExecution.exportCalls==0 && refusedOwners.empty(),
                "scheduled GF2 keeps strict owning interval/mapping refusal for the same outgoing held sample");
            scheduledTransaction.discard();
        }
    }
}

void testReactiveMaterialFramePrerequisite()
{
    using namespace videohelper::importedscene;
    std::string error;
    const auto plan = reactiveFramePlan(error);
    videowire::VisualLayerExecution execution;
    check(videowire::compileVisualLayerExecution(plan,execution,error) && execution.geometryFramePlan,
        "reactive Geometry Image admits a separately scheduled canonical Frame prerequisite");
    if (!execution.geometryFramePlan) return;
    struct Renderer
    {
        std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> leaseShaderFrame(const videorender::LayerDesc&,std::string&) { return {}; }
        bool prepareFlatShaderBridge(const videorender::LayerDesc&,std::string&) { return true; }
        int outputWidth() const { return 640; }
        int outputHeight() const { return 360; }
    } renderer;
    videorender::LayerDesc layer;
    videowire::VisualPlanEvaluationContext context;
    context.projectGeneration = 12; context.helperGeneration = 13; context.deviceGeneration = 14;
    context.materialFrameClock = [](double seconds,auto& frame,std::string& failure) {
        return videohelper::importedanimation::importedAnimationFrameIdentity(seconds,29.97,frame,failure);
    };
    auto image = std::make_shared<FakeFrame>();
    int calls = 0;
    context.materialFrameResolver = [&](const MaterialFrameEvaluation& evaluation,MaterialFrameReceipt& receipt,std::string&) {
        ++calls; receipt = {evaluation,image}; return true;
    };
    for (auto use : {videohelper::geometry::PlanUse::preview,videohelper::geometry::PlanUse::exportRender})
        for (double seconds : {0.0,1.0,0.0})
        {
            const auto before = calls;
            check(videowire::prepareGeometryMaterialFrame(renderer,plan,layer,use,seconds,nullptr,context,error)
                && calls == before + 1,"reactive Frame dependency refreshes on playback, repeat seek and preview/export evaluation");
            visualdeformation::RationalFrameTime expected;
            context.materialFrameClock(seconds,expected,error);
            check(context.geometryMaterialFrame && context.geometryMaterialFrame->evaluation.frame == expected
                && context.geometryMaterialFrame->evaluation.clipId == plan.clipId
                && context.geometryMaterialFrame->evaluation.endpoint.node == 91
                && context.geometryMaterialFrame->evaluation.use == (use == videohelper::geometry::PlanUse::preview
                    ? NativeImportedSceneRenderUse::Preview : NativeImportedSceneRenderUse::Export),
                "reactive Frame uses the same rational timeline and owned receipt contract as imported Scene3D");
        }
    for (int fault = 0; fault < 7; ++fault)
    {
        context.materialFrameResolver = [&](const auto& evaluation,auto& receipt,std::string&) {
            receipt = {evaluation,image};
            if (fault == 0) ++receipt.evaluation.frame.frame;
            if (fault == 1) ++receipt.evaluation.endpoint.node;
            if (fault == 2) ++receipt.evaluation.structuralRevision;
            if (fault == 3) ++receipt.evaluation.projectGeneration;
            if (fault == 4) ++receipt.evaluation.helperGeneration;
            if (fault == 5) receipt.evaluation.use = NativeImportedSceneRenderUse::Export;
            if (fault == 6) receipt.nativeFrame.reset();
            return true;
        };
        check(!videowire::prepareGeometryMaterialFrame(renderer,plan,layer,videohelper::geometry::PlanUse::preview,
            0,nullptr,context,error),"reactive Frame rejects stale timeline, endpoint, revision, reopen, helper and mode receipts");
    }
    context.materialFrameResolver = {};
    check(!videowire::prepareGeometryMaterialFrame(renderer,plan,layer,videohelper::geometry::PlanUse::preview,
        0,nullptr,context,error),"missing reactive Frame decoder fails closed despite a previous live receipt");
    for (int fault = 0; fault < 4; ++fault)
    {
        auto bad = plan;
        if (fault == 0) bad.edges.clear();
        if (fault == 1) bad.edges.push_back(bad.edges.front());
        if (fault == 2) bad.edges[0].fromNodeId = 999;
        if (fault == 3) bad.operations[1].payloadXml = "foreign-decoder";
        check(!videowire::compileVisualLayerExecution(bad,execution,error),
            "reactive Frame schedule cannot drop, duplicate or substitute the authored producer");
    }
}

void testMaterialFrameDagAndTime()
{
    using namespace videowire;
    using namespace videohelper::importedscene;
    std::string error;
    auto plan = reactiveFramePlan(error);
    plan.nodeIds.push_back(92); plan.nodeKinds.push_back("video.layer.source");
    plan.operations.push_back({92,"video.layer.source","source-decode",materialframesource::encode(111)});
    plan.ports.push_back({92,0,1,"out","frame","image","rgba8","sRGB"});
    plan.ports.push_back({71,13,1,"in","frame","image","rgba8","sRGB"});
    plan.edges.push_back({92,0,71,13});
    const std::vector<materialframe::Endpoint> endpoints {{91,0},{92,0}};
    auto schedule = materialframe::admit(plan,71,endpoints,[](auto&,auto&) { return false; },error);
    check(schedule && schedule->operations.size() == 2,"independent decoded sources admit a bounded Frame DAG");
    if (!schedule) return;
    videosourcecolor::Policy policy;
    check(materialframe::sourcePolicy(plan,{91,0},policy,error) && !policy.hlgNominalDisplayPeakNits,
        "old Video Source remains unspecified without deriving peak from export");
    auto policyPlan = plan;
    auto policySource = std::find_if(policyPlan.operations.begin(),policyPlan.operations.end(),
        [](const auto& op) { return op.nodeId == 91; });
    policySource->payloadXml = videosourcecolor::encode(videosourcecolor::Policy{4000.0});
    check(materialframe::admit(policyPlan,71,endpoints,[](auto&,auto&) { return false; },error).has_value()
        && materialframe::sourcePolicy(policyPlan,{91,0},policy,error) && policy.hlgNominalDisplayPeakNits == 4000.0,
        "exact decoded Video Source carries authored HLG policy through its schedule");
    policySource->payloadXml += "|exportPeak=10000";
    check(!materialframe::admit(policyPlan,71,endpoints,[](auto&,auto&) { return false; },error),
        "source policy rejects extra and export-mastering fields");
    check(materialframe::sourcePolicy(plan,{92,0},policy,error) && !policy.hlgNominalDisplayPeakNits,
        "alternate clip-ID-only source format remains unchanged and unspecified");
    int clip = 0;
    check(materialframe::sourceClip(plan,{92,0},clip,error) && clip == 111,"Layer Source preserves the canonical project clip ID");
    struct Renderer {
        std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> leaseShaderFrame(const videorender::LayerDesc&,std::string&) { return {}; }
        std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> leaseTemporalFrame(
            const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& current,
            const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& previous,
            const visualtemporaloperation::Payload&,float,std::string&) { return previous ? previous : current; }
    } renderer;
    VisualLayerExecution execution; execution.geometryFrameSchedule = schedule;
    videorender::LayerDesc layer;
    VisualPlanEvaluationContext context;
    context.materialFrameClock = [](double seconds,auto& frame,std::string&) {
        frame = {static_cast<std::int64_t>(std::llround(seconds*30)),30,1}; return true;
    };
    std::vector<std::pair<int,std::int64_t>> requested;
    context.materialFrameResolver = [&](const auto& request,auto& receipt,std::string&) {
        requested.emplace_back(request.endpoint.node,request.frame.frame);
        auto image = std::make_shared<FakeFrame>(); image->image = 100 + request.endpoint.node;
        receipt = {request,std::move(image)}; return true;
    };
    for (auto use : {videohelper::geometry::PlanUse::preview,videohelper::geometry::PlanUse::exportRender})
        for (double seconds : {1.0,0.0,1.0}) {
            requested.clear();
            check(prepareGeometryMaterialFrame(renderer,plan,layer,use,seconds,nullptr,context,error,&execution)
                && requested.size() == 2 && context.geometryMaterialFrames.size() == 2
                && context.geometryMaterialFrames.at({91,0}).nativeFrame != context.geometryMaterialFrames.at({92,0}).nativeFrame,
                "independent Frame leases survive reverse seeks and preview/export without source aliasing");
        }
    auto temporalPlan = plan;
    temporalPlan.nodeIds.push_back(93); temporalPlan.nodeKinds.push_back("visual.frame-delay");
    visualtemporaloperation::Payload payload; payload.mode = visualtemporaloperation::Mode::frameDelay;
    payload.historyLength = 2; payload.decay = 0; payload.zoom = 1;
    temporalPlan.operations.push_back({93,"visual.frame-delay","native-gpu",visualtemporaloperation::serialize(payload)});
    temporalPlan.ports.push_back({93,0,1,"in","frame","image","rgba8","sRGB"});
    temporalPlan.ports.push_back({93,1,1,"out","frame","image","rgba8","sRGB"});
    temporalPlan.edges[0] = {93,1,71,12}; temporalPlan.edges.push_back({91,0,93,0});
    schedule = materialframe::admit(temporalPlan,71,{{93,1},{92,0}},[](auto&,auto&) { return false; },error);
    check(schedule.has_value(),"Frame Delay participates in a multi-source dependency DAG");
    if (schedule) {
        execution.geometryFrameSchedule = schedule; requested.clear();
        check(prepareGeometryMaterialFrame(renderer,temporalPlan,layer,videohelper::geometry::PlanUse::preview,
            1.0,nullptr,context,error,&execution)
            && requested == std::vector<std::pair<int,std::int64_t>>{{91,28},{92,30}},
            "Frame Delay requests its source at the exact prior rational sample, independently of the other endpoint");
        const auto sdrResolver = context.materialFrameResolver;
        context.materialFrameResolver = [&](const auto& request,auto& receipt,std::string&) {
            auto linear = std::make_shared<FakeFrame>(); linear->linearColor = true;
            receipt = {request,std::move(linear)}; return true;
        };
        check(!prepareGeometryMaterialFrame(renderer,temporalPlan,layer,videohelper::geometry::PlanUse::preview,
            1.0,nullptr,context,error,&execution) && context.geometryMaterialFrames.empty()
            && error.find("temporal intermediates are unavailable") != std::string::npos,
            "HDR material temporal intermediates reject rather than render an SDR substitute");
        auto shaderSchedule = *execution.geometryFrameSchedule;
        auto& shaderOperation = shaderSchedule.operations.at({93,1});
        shaderOperation.temporal.reset();
        shaderOperation.shader = std::make_shared<const videowire::ShaderOperationPlan>();
        execution.geometryFrameSchedule = shaderSchedule;
        check(!prepareGeometryMaterialFrame(renderer,temporalPlan,layer,videohelper::geometry::PlanUse::exportRender,
            1.0,nullptr,context,error,&execution) && context.geometryMaterialFrames.empty()
            && error.find("shader, effect and transition intermediates are unavailable") != std::string::npos,
            "HDR material shader intermediates reject before the byte shader renderer executes");
        execution.geometryFrameSchedule = schedule;
        context.materialFrameResolver = sdrResolver;
        auto resetPlan = temporalPlan;
        auto resetPayload = payload; resetPayload.reset = 1;
        resetPlan.operations.back().payloadXml = visualtemporaloperation::serialize(resetPayload);
        check(!materialframe::admit(resetPlan,71,{{93,1},{92,0}},[](auto&,auto&) { return false; },error),
            "Surface DAG cannot silently ignore an authored shared temporal reset");
        auto resetHistory = std::make_shared<videorender::ParticleHistorySource>();
        resetHistory->parameterAt = [](const auto& destination,double at,double& value,std::string&) {
            if (destination.find("/visual93/reset") != std::string::npos && at > .97 && at < .99) value = 1;
            return true;
        };
        layer.particleHistory = resetHistory;
        check(!prepareGeometryMaterialFrame(renderer,temporalPlan,layer,videohelper::geometry::PlanUse::preview,
            1.0,nullptr,context,error,&execution) && context.geometryMaterialFrames.empty(),
            "Surface DAG rejects a transient shared reset edge even when the current reset value is neutral");
        layer.particleHistory.reset();
        auto cyclic = temporalPlan; cyclic.edges.back() = {93,1,93,0};
        check(!materialframe::admit(cyclic,71,{{93,1},{92,0}},[](auto&,auto&) { return false; },error),
            "temporal Frame cycles cannot bypass dependency admission");
        payload.mode = visualtemporaloperation::Mode::feedback; payload.decay = 0.92f; payload.zoom = 0.99f;
        temporalPlan.operations.back().kind = temporalPlan.nodeKinds.back() = "visual.feedback";
        temporalPlan.operations.back().payloadXml = visualtemporaloperation::serialize(payload);
        execution.geometryFrameSchedule = materialframe::admit(temporalPlan,71,{{93,1},{92,0}},[](auto&,auto&) { return false; },error);
        requested.clear();
        check(!prepareGeometryMaterialFrame(renderer,temporalPlan,layer,videohelper::geometry::PlanUse::exportRender,
            20.0,nullptr,context,error,&execution) && context.geometryMaterialFrames.empty(),
            "long feedback replay rejects its budget before publishing partial material images");
    }
    colortransform::AdmissionFailure failure;
    for (auto transfer : {colortransform::TransferFunction::Rec709,colortransform::TransferFunction::Gamma22})
        check(decodedframecolor::toMaterialSrgb({colortransform::ColorSpace::Rec709,transfer},{32,32},failure).has_value(),
            "declared SDR decoder transfer uses the shared native colour transform admission");
    for (auto transfer : {colortransform::TransferFunction::PQ,colortransform::TransferFunction::HLG,
                          colortransform::TransferFunction::Unspecified})
        check(!decodedframecolor::toMaterialSrgb({colortransform::ColorSpace::Rec2020,transfer},{32,32},failure),
            "HDR and undeclared decoder transfers cannot silently enter the SDR texture path");
}

void testMixedRetainedSceneSchedule()
{
    using namespace videohelper::importedscene;
    std::string error;
    const auto fixture = mixedSurfaceDiffractionFixture(error);
    check(fixture.has_value(), "mixed retained scene schedule has a valid authored fixture");
    if (!fixture) return;
    const auto planFor = [&](const visualimportedscenerender::Request& request, bool frame) {
        videowire::CompiledVisualLayerPlan plan;
        plan.clipId = 8; plan.structuralRevision = request.structuralRevision;
        plan.nodeIds = {700,701}; plan.nodeKinds = {videowire::geometry::kRetainedSceneOperation,"visual.3d.render"};
        plan.operations = {{700,plan.nodeKinds[0],"control-eval",""},
            {701,plan.nodeKinds[1],"native-gpu",visualimportedscenerender::encode(request)}};
        plan.ports = {{700,0,1,"out","control","scene3D","unspecified","unspecified"},
            {701,0,1,"in","control","scene3D","unspecified","unspecified"},
            {701,1,1,"out","frame","image","rgba8","sRGB"}};
        plan.edges = {{700,0,701,0}};
        if (frame) {
            plan.nodeIds.insert(plan.nodeIds.begin()+1,91);
            plan.nodeKinds.insert(plan.nodeKinds.begin()+1,"video.source");
            plan.operations.insert(plan.operations.begin()+1,{91,"video.source","source-decode",videosourcecolor::encode(videosourcecolor::Policy{})});
            plan.ports.push_back({91,0,1,"out","frame","image","rgba8","sRGB"});
            plan.ports.push_back({701,12,1,"in","frame","image","rgba8","sRGB"});
            plan.edges.push_back({91,0,701,12});
        }
        return plan;
    };
    auto plan = planFor(*fixture, false);
    videowire::VisualLayerExecution compiled;
    {
        const bool conditionOk = videowire::compileVisualLayerExecution(plan, compiled, error)
        && compiled.importedSceneRender && compiled.importedSceneRender->diffractionMaterial
        && compiled.importedSceneRender->surfacePrograms.size() == 1;
        check(conditionOk, "V18 mixed request enters the exact retained-scene schedule: " + error);
    }
    auto legacyOnly = *fixture; legacyOnly.surfacePrograms.clear();
    check(!videowire::compileVisualLayerExecution(planFor(legacyOnly,false), compiled, error),
        "legacy diffraction-only retained schedules preserve their released refusal");
    auto framed = *fixture; mixedSurfaceFrame(framed);
    plan = planFor(framed,true);
    {
        const bool conditionOk = videowire::compileVisualLayerExecution(plan, compiled, error)
        && compiled.materialFrameSource && compiled.materialFrameSource->nodeId == 91;
        check(conditionOk, "Surface-only Frame in a mixed scene retains the exact scheduled source: " + error);
    }
    FakeExecution execution;
    FakeLayer layer;
    VisualImportedScenePlanCache cache;
    std::vector<FakeReceipt> owners;
    auto input = std::make_shared<FakeFrame>();
    int evaluations = 0;
    MaterialFrameResolver resolver = [&](const MaterialFrameEvaluation& evaluation, MaterialFrameReceipt& receipt, std::string&) {
        ++evaluations;
        check(evaluation.endpoint.node == 91 && evaluation.endpoint.port == 0
            && evaluation.frame.frame == 30 && evaluation.frame.rateNumerator == 24
            && evaluation.frame.rateDenominator == 1 && evaluation.structuralRevision == 3
            && evaluation.evaluationRevision == 7 && evaluation.projectGeneration == 12
            && evaluation.helperGeneration == 34,
            "mixed Frame prerequisite retains exact endpoint, rational time, revisions and generations");
        receipt = {evaluation,input}; return true;
    };
    for (auto use : {NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export})
        {
            const bool conditionOk = prepareVisualImportedSceneLayerAtTime({plan},8,640,360,1.25,24,use,&execution,&cache,
            layer,owners,error,nullptr,{12,34},&resolver) == VisualImportedScenePreparation::rendered
            && execution.lastRequest.runtimeInputs.materialFrameTexture == input
            && execution.lastRequest.surfacePrograms.front().program.material.sceneSnapshot == execution.lastRequest.sceneSnapshot
            && !owners.empty() && owners.back().materialFrameTexture == input;
            check(conditionOk, "mixed Surface Frame dependency reaches one owned preview/export scene submission: " + error);
        }
    check(evaluations == 2, "mixed preview and export each resolve their same-time Frame prerequisite");
    check(prepareVisualImportedSceneLayerAtTime({plan},8,640,360,1.25,24,NativeImportedSceneRenderUse::Preview,
        &execution,&cache,layer,owners,error,nullptr,{12,34},nullptr) == VisualImportedScenePreparation::rejected,
        "mixed Surface Frame cannot reuse a prior lease without its resolver");
    framed.diffractionMaterial->version = diffractionmaterialbinding::kGraphFrameWireVersion;
    framed.diffractionMaterial->graphFrame = surfacematerialbinding::TextureSlotBinding::GraphFrameEndpoint {92,0};
    check(!videowire::compileVisualLayerExecution(planFor(framed,true), compiled, error),
        "mismatched dual Frame endpoint ownership refuses the mixed schedule");
}

void testGlassCausticRequestHandoff()
{
    using namespace videohelper::importedscene;
    using namespace HarmonicMIDI::grid;
    auto plan = composedScenePlan();
    visualimportedscenerender::Request request;
    check(visualimportedscenerender::decode(plan.operations.back().payloadXml, request),
        "caustic transport fixture starts from the canonical retained scene");
    std::string error;
    const auto generated = videohelper::tests::generatedSurfaceScene(*request.sceneSnapshot, error);
    check(generated.has_value(), "caustic transport uses a retained generated closed cube: " + error);
    if (!generated) return;
    auto scene = std::make_shared<Visual3DScene>(*generated);
    scene->lights[0].spectralSource.version = 1;
    scene->lights[0].spectralSource.wavelengthsNanometres = diffractionmaterial::kProductionWavelengthsNanometres;
    scene->lights[0].spectralSource.radiance.fill(1);
    scene->lights[0].spectralSource.traceMaximumDistance = 75;
    request.sceneSnapshot = scene;
    geometrysurfacematerial::ObjectProgram glass;
    auto& material = glass.program.material;
    glass.program.sourceMaterial = 1;
    material.scene = scene->id;
    material.sceneSnapshot = scene;
    material.sceneRevision = material.structuralRevision = material.programRevision = request.structuralRevision;
    material.evaluationRevision = request.evaluationRevision;
    material.binding.targetKind = surfacematerialbinding::BindingTargetKind::ObjectOverride;
    material.binding.object = scene->objects[0].id;
    material.binding.surfaceMaterialRevision = material.programRevision;
    material.program = videohelper::tests::smoothglassfixture::authoredProgram(scene->objects[0].material.value);
    const auto admitted = surfacematerial::admit(material.program, error);
    check(admitted.has_value(), "caustic transport retains an admitted closed-glass material: " + error);
    if (!admitted) return;
    material.binding.surfaceMaterialDigest = admitted->structuralDigest();
    request.surfacePrograms.push_back(glass);
    request.glassCaustic = visualimportedscenerender::GlassCausticSelection {
        1, scene->lights[0].id, scene->objects[0].id, scene->objects[0].material,
        {1, 1.0e-4f, 1.0e-6f, 32, 64, 16}};
    plan.nodeIds = {6,11};
    plan.nodeKinds = {videowire::geometry::kRetainedSceneOperation, "visual.3d.render"};
    plan.operations = {{6, plan.nodeKinds[0], "control-eval", ""},
        {11, plan.nodeKinds[1], "native-gpu", visualimportedscenerender::encode(request)}};
    plan.edges = {{6,0,11,0}};
    plan.ports = {{6,0,1,"out","control","scene3D","unspecified","unspecified"},
        {11,0,1,"in","control","scene3D","unspecified","unspecified"},
        {11,1,1,"out","frame","image","rgba8","sRGB"}};
    videowire::VisualLayerExecution compiled;
    {
        const bool conditionOk = !plan.operations[1].payloadXml.empty() && videowire::compileVisualLayerExecution(plan, compiled, error)
        && compiled.importedSceneRender && compiled.importedSceneRender->glassCaustic;
        check(conditionOk, "valid caustic v19 reaches the ordinary helper preparation boundary: " + error);
    }
    const auto encoded = plan.operations[1].payloadXml;
    FakeExecution execution;
    execution.nextFrame->linearColor = true;
    VisualImportedScenePlanCache cache;
    for (const auto use : {NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export})
    {
        videorender::LayerDesc layer;
        std::vector<FakeReceipt> owners;
        const auto result = prepareVisualImportedSceneLayerAtTime({plan}, 8, 640, 360, 1.25, 24,
            use, &execution, &cache, layer, owners, error, nullptr, {12,34});
        const auto& actual = execution.lastRequest;
        {
            const bool conditionOk = result == VisualImportedScenePreparation::rendered
            && actual.glassCaustic
            && visualimportedscenerender::sameGlassCaustic(*actual.glassCaustic, *request.glassCaustic)
            && actual.sceneSnapshot
            && sceneaov::detail::encodeScene(*actual.sceneSnapshot) == sceneaov::detail::encodeScene(*scene)
            && actual.surfacePrograms.size() == 1
            && actual.surfacePrograms[0].program.material.sceneSnapshot == actual.sceneSnapshot
            && actual.frame.frame == 30 && actual.frame.rateNumerator == 24
            && actual.projectGeneration == 12 && actual.helperGeneration == 34
            && actual.staticPayloadIdentity == exactImportedScenePayloadIdentity(plan)
            && actual.runtimeInputs.linearColor
            && layer.nativeLinearImage == execution.nextFrame
            && layer.nativeTextureOwner == execution.nextFrame && owners.size() == 1;
            check(conditionOk, "ordinary preview/export preserves exact caustic source, glass, policy, frame and linear ownership: " + error);
        }
    }
    check(execution.staticGpuUploads == 1,
        "equal-time preview/export reuse one exact retained scene owner");

    auto changed = request;
    changed.glassCaustic->policy.maximumVisits = 63;
    auto changedPlan = plan;
    changedPlan.operations[1].payloadXml = visualimportedscenerender::encode(changed);
    videorender::LayerDesc layer;
    std::vector<FakeReceipt> owners;
    {
        const bool conditionOk = !changedPlan.operations[1].payloadXml.empty()
        && prepareVisualImportedSceneLayerAtTime({changedPlan}, 8, 640, 360, 1.25, 24,
            NativeImportedSceneRenderUse::Preview, &execution, &cache, layer, owners, error, nullptr, {12,34})
                == VisualImportedScenePreparation::rendered
        && execution.identityRetirements == 1 && execution.lastRequest.glassCaustic
        && execution.lastRequest.glassCaustic->policy.maximumVisits == 63
        && execution.lastRequest.staticPayloadIdentity != exactImportedScenePayloadIdentity(plan);
        check(conditionOk, "a same-revision caustic policy replacement changes exact serialized ownership: " + error);
    }

    const auto selectionLine = [](const visualimportedscenerender::GlassCausticSelection& selection)
    {
        const auto& p = selection.policy;
        std::ostringstream out;
        out << selection.version << ' ' << selection.sourceLight.value << ' '
            << selection.glassObject.value << ' ' << selection.glassMaterial.value << ' '
            << p.version << ' ' << floatBits(p.maximumUnresolvedFraction) << ' '
            << floatBits(p.minimumBranchFraction) << ' ' << p.maximumInterfaces << ' '
            << p.maximumVisits << ' ' << p.maximumPending;
        return out.str();
    };
    std::vector<std::string> malformed;
    for (int mutation = 0; mutation < 5; ++mutation)
    {
        auto selection = *request.glassCaustic;
        if (mutation == 0) selection.sourceLight.value = 41;
        if (mutation == 1) selection.glassObject.value = 11;
        if (mutation == 2) selection.glassMaterial.value = 21;
        if (mutation == 3) selection.policy.maximumVisits = 0;
        if (mutation == 4) selection.version = 2;
        malformed.push_back(replaceLine(encoded, 1, selectionLine(selection)));
    }
    malformed.push_back(encoded + "malformed");
    auto noLi8 = request;
    auto noLi8Scene = std::make_shared<Visual3DScene>(*scene);
    noLi8Scene->lights[0].spectralSource = {};
    noLi8.sceneSnapshot = noLi8Scene;
    noLi8.surfacePrograms[0].program.material.sceneSnapshot = noLi8Scene;
    noLi8.glassCaustic.reset();
    malformed.push_back(std::string(visualimportedscenerender::kCompiledOperationHeaderV19)
        + "\n" + selectionLine(*request.glassCaustic) + "\n" + visualimportedscenerender::encode(noLi8));
    for (const auto& payload : malformed)
    {
        auto invalid = plan;
        invalid.operations[1].payloadXml = payload;
        const auto calls = execution.previewCalls + execution.exportCalls;
        layer.nativeTextureView = 999;
        layer.nativeLinearImage = execution.nextFrame;
        owners.clear();
        check(prepareVisualImportedSceneLayerAtTime({invalid}, 8, 640, 360, 1.25, 24,
            NativeImportedSceneRenderUse::Preview, &execution, &cache, layer, owners, error, nullptr, {12,34})
                == VisualImportedScenePreparation::rejected
            && execution.previewCalls + execution.exportCalls == calls
            && layer.nativeTextureView == 0 && !layer.nativeLinearImage && owners.empty()
            && execution.publishedIdentity == exactImportedScenePayloadIdentity(invalid),
            "malformed or foreign caustic selection retires serialized ownership before execution");
    }
    for (const auto use : {NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export})
    {
        execution.succeed = false;
        owners.clear();
        layer.nativeTextureView = 999;
        layer.nativeLinearImage = execution.nextFrame;
        check(prepareVisualImportedSceneLayerAtTime({plan}, 8, 640, 360, 1.25, 24,
            use, &execution, &cache, layer, owners, error, nullptr, {12,34})
                == VisualImportedScenePreparation::rejected
            && execution.lastRequest.glassCaustic
            && layer.nativeTextureView == 0 && !layer.nativeLinearImage && owners.empty(),
            "a native caustic refusal publishes no preceding layer or frame");
    }
    execution.succeed = true;
    auto absent = request;
    absent.glassCaustic.reset();
    auto absentPlan = plan;
    absentPlan.operations[1].payloadXml = visualimportedscenerender::encode(absent);
    execution.nextFrame->linearColor = false;
    {
        const bool conditionOk = prepareVisualImportedSceneLayerAtTime({absentPlan}, 8, 640, 360, 1.25, 24,
        NativeImportedSceneRenderUse::Preview, &execution, &cache, layer, owners, error, nullptr, {12,34})
            == VisualImportedScenePreparation::rendered
        && !execution.lastRequest.glassCaustic && !execution.lastRequest.runtimeInputs.linearColor;
        check(conditionOk, "a retained scene without caustic selection preserves the ordinary SDR route: " + error);
    }
}

void testImportedSolidComposition()
{
    using namespace videorender;
    using namespace videowire::geometry;
    using namespace videohelper::geometry;
    using namespace videohelper::importedscene;
    using namespace HarmonicMIDI::grid;
    std::string error;
    auto request=composedTimelineGeometryFixture(error);
    check(request.has_value(),"imported solid fixture retains the timeline source: "+error);
    if (!request) return;
    const auto authored=sceneaov::detail::encodeScene(*request->sceneSnapshot);
    auto p=ParticleParams{};
    const auto xml="<NodeParams motionMode=\"2\" simulationSpace=\"1\" historicalReplay=\"1\" count=\"2\" "
        "speed=\"0\" lifetime=\"2\" resetMode=\"1\" colliderGeometryPlan=\""+request->timelineGeometry->loweredPlan+"\"/>";
    {
        const bool conditionOk = videowire::admitParticleParameters(xml,p,error);
        check(conditionOk, "imported solid parameters admit: "+error);
    }
    p.history=std::make_shared<const ParticleHistorySource>(waterreplayfixture::source());
    p.historyProjectSeconds=.5;
    ParticleSolidState solids;
    {
        const bool conditionOk = sampleParticleGeometry(p,kParticleReplayTicksPerSecond/2,solids.colliders,error);
        check(conditionOk, "imported solid collider evaluates at the exact project tick: "+error);
    }
    solids.count=2;
    solids.bodies[0].x=.5f; solids.bodies[0].y=.5f; solids.bodies[0].z=.75f;
    solids.bodies[1].x=.6f; solids.bodies[1].y=.4f; solids.bodies[1].z=.8f;
    class Capabilities final : public BackendCapabilitySource {
    public:
        BackendCapabilities geometryCoreCapabilities() const override {
            BackendCapabilities c; c.backendIdentity="solid-fixture"; c.deviceIdentity="owned-fixture";
            c.nativeGpuAvailable=c.geometryCoreExecution=c.immutableSourceBuffers=c.stableElementIds=c.typedFieldEvaluation=true;
            c.supportedCarriers=carrierBit(CarrierKind::geometry3D); return c;
        }
    } capabilities;
    GeometryCorePlanRuntime runtime(capabilities);
    const PlanOwnerIdentity owner{3,4,5,8,9,PlanUse::preview};
    auto current=evaluateComposedTimelineGeometry(runtime,owner,*request->timelineGeometry,request->sceneSnapshot,.5,error);
    visualdeformation::RationalFrameTime roundedA,roundedB;
    check(videohelper::importedanimation::importedAnimationFrameIdentity(.079,30,roundedA,error)
        && videohelper::importedanimation::importedAnimationFrameIdentity(.080,30,roundedB,error)
        && roundedA==roundedB,"nearby exact scene queries share one rounded imported frame identity");
    const auto exactA=importedTimelineEvaluationSeconds(.079,roundedA,true);
    const auto exactB=importedTimelineEvaluationSeconds(.080,roundedB,true);
    auto sceneA=evaluateComposedTimelineGeometry(runtime,owner,*request->timelineGeometry,request->sceneSnapshot,exactA,error);
    auto sceneB=evaluateComposedTimelineGeometry(runtime,owner,*request->timelineGeometry,request->sceneSnapshot,exactB,error);
    auto sample=p; ParticleSolidState exactSolids=solids;
    check(exactA==.079 && exactB==.080 && importedTimelineEvaluationSeconds(.079,roundedA,false)==2.0/30
        && sampleParticleGeometry(sample,std::llround(exactA*kParticleReplayTicksPerSecond),exactSolids.colliders,error)
        && sceneA && sceneB && sceneA!=sceneB
        && sceneaov::detail::encodeScene(*sceneA)!=sceneaov::detail::encodeScene(*sceneB)
        && appendParticleSolidsToScene(sceneA,*request->sceneSnapshot,*request->timelineGeometry,sample,exactSolids,90,error),
        "exact combined clocks preserve subframe collider alignment and regenerate distinct scene snapshots");
    FakeExecution subframeNative; FakeReceipt subframeReceipt;
    videohelper::modelpayload::ImportedSceneRequest subframeRequest;
    subframeRequest.frame=roundedA; subframeRequest.sceneSnapshot=sceneA;
    check(subframeNative.executePreview(subframeRequest,subframeReceipt,error),"first exact subframe reaches native input");
    subframeRequest.frame=roundedB; subframeRequest.sceneSnapshot=sceneB;
    check(subframeNative.executePreview(subframeRequest,subframeReceipt,error) && subframeNative.staticGpuUploads==2,
        "same-rounded-frame exact scene changes cannot reuse the preceding native scene input pointer");
    auto appended=current ? appendParticleSolidsToScene(current,*request->sceneSnapshot,*request->timelineGeometry,p,solids,90,error):nullptr;
    check(appended && appended!=current,"solid bodies append to an owned evaluated scene: "+error);
    if (!appended) return;
    check(appended->objectCount==4 && appended->vertexCount==current->vertexCount+6
        && appended->indexCount==current->indexCount+24 && appended->materialCount==3
        && appended->activeCamera==current->activeCamera && appended->cameraCount==current->cameraCount
        && appended->lightCount==current->lightCount,
        "two sphere bodies share six vertices and 24 indices under the imported camera and lights");
    auto preserved=*appended;
    preserved.objectCount=current->objectCount; preserved.materialCount=current->materialCount;
    preserved.vertexCount=current->vertexCount; preserved.indexCount=current->indexCount;
    check(sceneaov::detail::encodeScene(preserved)==sceneaov::detail::encodeScene(*current)
        && sceneaov::detail::encodeScene(*request->sceneSnapshot)==authored,
        "composition preserves every imported record and the authored source bytes");
    const auto& body=appended->objects[2];
    check(body.parent==request->timelineGeometry->object && body.id.value!=10 && body.id.value!=11
        && body.transform.translation.x==0 && body.transform.translation.y==0 && body.transform.translation.z==1
        && body.transform.scale.x==p.bodyRadius/p.geometryBinding->scale
        && appended->objects[3].firstVertex==body.firstVertex,
        "body transforms invert the exact simulation binding inside the imported hierarchy without private-camera centering");
    {
        // With a Motion prior the surviving body takes its prior replay pose.
        // The imported objects and the body born this frame repeat theirs.
        auto identified=std::make_unique<ParticleSolidState>(solids);
        identified->bodies[0].identity=1; identified->bodies[1].identity=2;
        auto previous=std::make_unique<ParticleSolidState>(*identified);
        previous->count=1; previous->bodies[0].x-=.05f;
        const std::array<double,kParticleSolidMaxBodies> births {};
        const ParticleSolidMotionPrior prior{*previous,births,births,p.bodyRadius};
        const auto moving=appendParticleSolidsToScene(current,*request->sceneSnapshot,*request->timelineGeometry,
            p,*identified,90,error,&prior);
        const auto same=[&](std::size_t i) {
            return arbitgpu::sameSceneTransform(moving->previousObjectTransforms[i],moving->objects[i].transform);
        };
        {
            const bool conditionOk = moving && moving->previousObjectTransformCount==4 && same(0) && same(1) && same(3)
            && std::abs(moving->objects[2].transform.translation.x-moving->previousObjectTransforms[2].translation.x
                -.05f/p.geometryBinding->scale)<1e-5f
            && moving->previousObjectTransforms[2].scale.x==moving->objects[2].transform.scale.x
            && arbitgpu::sceneMotionPositionsMatch(*moving,current->vertexCount)
            && !arbitgpu::sceneMotionPositionsMatch(*moving,current->vertexCount+1)
            && !arbitgpu::sceneMotionPositionsMatch(*appended,current->vertexCount);
            check(conditionOk, "imported Solid Motion keeps the scene part's retained positions and each body's own prior transform: "+error);
        }
    }
    auto selected=std::make_shared<ParticleGeometryBinding>(*p.geometryBinding);
    PortContract contract; contract.carrier=CarrierKind::geometry3D; contract.maxVertices=256; contract.maxIndices=384;
    auto rigid=lowerMeshGenerator(609,609,1,OperationCode::cube,{1,1,1,0},contract,error);
    selected->body=rigid ? std::make_shared<const ValueDescriptor>(*rigid):nullptr;
    auto rigidParams=p; rigidParams.geometryBinding=selected;
    auto rigidScene=appendParticleSolidsToScene(current,*request->sceneSnapshot,*request->timelineGeometry,rigidParams,solids,90,error);
    check(rigidScene && rigidScene->vertexCount==current->vertexCount+24
        && rigidScene->objects[2].vertexCount==24,
        "selected static rigid appearance uses its released mesh builder instead of substituted spheres");
    auto refuse=[&](std::shared_ptr<const Visual3DScene> candidate,const ParticleParams& parameters,const ParticleSolidState& state,const char* message) {
        check(!appendParticleSolidsToScene(candidate,*request->sceneSnapshot,*request->timelineGeometry,parameters,state,90,error)
            && !error.empty() && sceneaov::detail::encodeScene(*request->sceneSnapshot)==authored
            && appended->objectCount==4,message);
    };
    check(appendParticleSolidsToScene(current,*request->sceneSnapshot,*request->timelineGeometry,p,ParticleSolidState{},90,error)==current,
        "pre-reset or zero-emission body replay keeps the ordinary imported scene frame");
    auto wrong=solids; wrong.colliders.meshVertices[0][2]+=.001f;
    refuse(current,p,wrong,"retimed or mismapped collider refuses without changing retained scenes");
    auto nonuniform=std::make_shared<Visual3DScene>(*current); nonuniform->objects[0].transform.scale={1,2,1};
    refuse(nonuniform,p,solids,"nonuniform bound object scale refuses before native preparation");
    auto projected=p; selected=std::make_shared<ParticleGeometryBinding>(*p.geometryBinding); selected->projection=1; projected.geometryBinding=selected;
    refuse(current,projected,solids,"non-XYZ simulation mapping refuses explicitly");
    auto overCount=solids; overCount.count=videorender::kParticleImportedSolidMaxBodies+1;
    refuse(current,p,overCount,"imported Solid composition above 64 bodies refuses before appending objects");
    // An oriented box replay draws its static body Geometry with the same
    // bounding-radius normalization and bodyRadius scale as its contact box.
    auto boxParams=rigidParams; boxParams.bodyShape=2; boxParams.collisionMode=2;
    SolidVector boxHalf{};
    const auto boxScene=appendParticleSolidsToScene(current,*request->sceneSnapshot,*request->timelineGeometry,boxParams,solids,90,error);
    {
        const bool conditionOk = rigid && solidBodyBoxHalfExtents(boxParams,boxHalf) && boxScene && rigidScene
        && sceneaov::detail::encodeScene(*boxScene)==sceneaov::detail::encodeScene(*rigidScene)
        && boxScene->objects[2].transform.scale.x==boxParams.bodyRadius/boxParams.geometryBinding->scale;
        check(conditionOk, "an oriented box replay appends its static body Geometry under the imported hierarchy: "+error);
    }
    auto bodylessBox=p; bodylessBox.bodyShape=2;
    refuse(current,bodylessBox,solids,"an oriented box without static body Geometry refuses before composition");
    auto full=std::make_shared<Visual3DScene>(*current);
    while (full->objectCount<Visual3DScene::kMaxObjects) {
        auto object=full->objects[1]; object.id={static_cast<std::uint32_t>(100+full->objectCount)};
        full->objects[full->objectCount++]=object;
    }
    refuse(full,p,solids,"remaining object capacity refuses without raising the 64-object cap");
    auto materialFull=std::make_shared<Visual3DScene>(*current);
    while (materialFull->materialCount<Visual3DScene::kMaxMaterials) {
        auto material=materialFull->materials[0]; material.id={static_cast<std::uint32_t>(100+materialFull->materialCount)};
        materialFull->materials[materialFull->materialCount++]=material;
    }
    refuse(materialFull,p,solids,"remaining material capacity refuses before native preparation");
    auto vertexFull=std::make_shared<Visual3DScene>(*current); vertexFull->vertexCount=Visual3DScene::kMaxVertices;
    refuse(vertexFull,p,solids,"remaining vertex capacity refuses before native preparation");
    auto indexFull=std::make_shared<Visual3DScene>(*current);
    while (indexFull->indexCount<Visual3DScene::kMaxIndices) indexFull->indices[indexFull->indexCount++]=0;
    refuse(indexFull,p,solids,"remaining index capacity refuses before native preparation");

    auto plan=composedScenePlan();
    plan.nodeIds={6,11,90,91,92};
    plan.nodeKinds={kRetainedSceneOperation,"visual.3d.render","visual.particles","video.blend","video.out"};
    plan.operations={{6,plan.nodeKinds[0],"control-eval",""},
        {11,plan.nodeKinds[1],"native-gpu",visualimportedscenerender::encode(*request)},
        {90,plan.nodeKinds[2],"native-gpu",xml},{91,plan.nodeKinds[3],"native-gpu","<NodeParams opacity=\"1\" mode=\"0\"/>"},
        {92,plan.nodeKinds[4],"native-gpu",""}};
    plan.edges={{6,0,11,0},{11,1,91,0},{90,1,91,2},{91,1,92,0}};
    plan.ports={{6,0,1,"out","control","scene3D","unspecified","unspecified"},
        {11,0,1,"in","control","scene3D","unspecified","unspecified"},{11,1,1,"out","frame","image","rgba8","sRGB"},
        {90,1,1,"out","frame","image","rgba8","sRGB"},{91,0,1,"in","frame","image","rgba8","sRGB"},
        {91,1,1,"out","frame","image","rgba8","sRGB"},{91,2,1,"in","frame","image","rgba8","sRGB"},
        {91,3,1,"in","frame","image","rgba8","sRGB"},{92,0,1,"in","frame","image","rgba8","sRGB"}};
    videowire::VisualLayerExecution compiled;
    {
        const bool conditionOk = videowire::compileVisualLayerExecution(plan,compiled,error) && compiled.importedParticleOverlay;
        check(conditionOk, "actual retained timeline and Solid 3D graph admit the composition schedule: "+error);
    }
    for (const char* option : {"imageOutput","exportMotion"}) {
        const std::string name=option;
        auto motionPlan=plan;
        auto& motionXml=motionPlan.operations[2].payloadXml;
        motionXml.replace(motionXml.rfind("/>"),2," "+name+"=\"1\"/>");
        videowire::VisualLayerExecution motionCompiled;
        {
            const bool conditionOk = videowire::compileVisualLayerExecution(motionPlan,motionCompiled,error) && motionCompiled.importedParticleOverlay
            && (name=="imageOutput" ? motionCompiled.particleParameters.solidMotionView
                                    : motionCompiled.particleParameters.solidExportMotion);
            check(conditionOk, "the Solid 3D overlay admits "+name+" over a timeline Geometry3D scene: "+error);
        }
    }
    if (const auto rigidValue=rigid ? admitValue(*rigid,contract,error):std::nullopt) {
        auto boxPlan=plan;
        boxPlan.operations[2].payloadXml="<NodeParams motionMode=\"2\" simulationSpace=\"1\" historicalReplay=\"1\" count=\"2\" "
            "speed=\"0\" lifetime=\"2\" resetMode=\"1\" collisionMode=\"2\" bodyShape=\"2\" bodyGeometryPlan=\""
            +encodeLoweredPlanText(lowerRuntimePlan(contract,*rigidValue))+"\" colliderGeometryPlan=\""
            +request->timelineGeometry->loweredPlan+"\"/>";
        videowire::VisualLayerExecution boxCompiled;
        {
            const bool conditionOk = videowire::compileVisualLayerExecution(boxPlan,boxCompiled,error) && boxCompiled.importedParticleOverlay
            && boxCompiled.particleParameters.bodyShape==2 && boxCompiled.particleParameters.geometryBinding
            && boxCompiled.particleParameters.geometryBinding->body && boxCompiled.particleParameters.geometryBinding->mesh;
            check(conditionOk, "imported Solid 3D admits oriented box bodies against the exact TimelineScene3D mesh: "+error);
        }
    }
    FakeExecution native;
    videohelper::modelpayload::ImportedSceneRequest input;
    input.sceneSnapshot=appended;
    FakeReceipt nativeReceipt;
    check(native.executePreview(input,nativeReceipt,error) && native.admittedScene==appended.get(),
        "the native request boundary receives the final composed scene pointer");
    auto receipt=std::make_shared<ParticleImportedSolidFrame>();
    receipt->owner=owner; receipt->nodeId=90; receipt->projectSeconds=receipt->clipSeconds=.5;
    receipt->sourceScene=current; receipt->scene=appended; receipt->frame=nativeReceipt.frame;
    receipt->bodies=p.preparedBodies=std::make_shared<const ParticleBodyState>();
    receipt->state=p.preparedSolids=std::make_shared<const ParticleSolidState>(solids);
    receipt->geometry=p.geometryBinding; p.preparedImportedSolids=receipt;
    LayerDesc layer;
    layer.particleParameters=p; layer.particleProjectSeconds=.5; layer.shaderClock.timeSec=.5;
    layer.texture=44; layer.nativeTextureView=44; layer.nativeTextureBackend="opengl";
    layer.nativeTextureDescriptor=receipt->frame->colorTextureDescriptor(); layer.texWidth=640; layer.texHeight=360;
    layer.depthTexture=66; layer.depthWidth=640; layer.depthHeight=360;
    layer.nativeDepthTextureBackend="opengl"; layer.nativeDepthTextureView=66;
    layer.nativeDepthTextureDescriptor=receipt->frame->depthTextureDescriptor();
    auto context=videowire::VisualPlanEvaluationContext{};
    context.projectGeneration=3; context.helperGeneration=4; context.deviceGeneration=5;
    {
        const bool conditionOk = videowire::executeVisualLayerPlan({plan},8,layer,error,PlanUse::preview,
        nullptr,nullptr,nullptr,.5,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,640,360,&context)
        && !layer.particleSource && !layer.importedParticleOverlay && layer.texture==44 && layer.depthTexture==66
        && layer.particleParameters.preparedImportedSolids==receipt;
        check(conditionOk, "shared execution consumes the exact imported frame and suppresses the later standalone solid render: "+error);
    }
    auto exportedReceipt=std::make_shared<ParticleImportedSolidFrame>(*receipt);
    exportedReceipt->owner.use=PlanUse::exportRender;
    layer.particleParameters.preparedImportedSolids=exportedReceipt;
    {
        const bool conditionOk = videowire::executeVisualLayerPlan({plan},8,layer,error,PlanUse::exportRender,
        nullptr,nullptr,nullptr,.5,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,640,360,&context)
        && !layer.particleSource && layer.texture==44 && layer.depthTexture==66;
        check(conditionOk, "export consumes its own retained scene owner without a second solid render: "+error);
    }
    ++context.helperGeneration;
    check(!videowire::executeVisualLayerPlan({plan},8,layer,error,PlanUse::exportRender,
        nullptr,nullptr,nullptr,.5,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,640,360,&context)
        && layer.texture==44 && layer.particleParameters.preparedImportedSolids==exportedReceipt,
        "shared execution refuses a stale scene receipt before changing the published layer");
    std::vector<videowire::VisualEventScheduleBinding> schedules(1);
    schedules[0].clipId=8; schedules[0].nodeId=90; schedules[0].portId=0;
    --context.helperGeneration;
    check(!videowire::executeVisualLayerPlan({plan},8,layer,error,PlanUse::exportRender,
        nullptr,nullptr,nullptr,.5,&schedules,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,640,360,&context)
        && layer.texture==44 && error.find("Event-triggered")!=std::string::npos,
        "unsupported Event schedules refuse before consuming or replacing the prepared imported frame");
    auto stale=owner; ++stale.helperGeneration;
    check(!validateImportedSolidFrame(p,stale,90,.5,.5,640,360,layer,error),"stale imported scene owner refuses");
    check(!validateImportedSolidFrame(p,owner,90,.5,.25,640,360,layer,error),"stale imported scene time refuses");
    ++layer.nativeDepthTextureDescriptor.rendererGeneration;
    check(!validateImportedSolidFrame(p,owner,90,.5,.5,640,360,layer,error),"mixed native depth generation refuses");
    layer.nativeDepthTextureDescriptor=receipt->frame->depthTextureDescriptor();
    ++layer.nativeTextureDescriptor.rendererGeneration;
    check(!validateImportedSolidFrame(p,owner,90,.5,.5,640,360,layer,error),"mixed native resource generation refuses");
    auto admitted=p;
    check(videowire::admitParticleParameters(xml,admitted,error) && !admitted.preparedImportedSolids,
        "ordinary parameter admission clears the request receipt");
    LayerDesc missing; missing.texture=123;
    std::vector<FakeReceipt> owners;
    VisualImportedScenePlanCache cache;
    const auto calls=native.previewCalls;
    missing.particleParameters=p;
    check(prepareVisualImportedSceneLayerAtTime({plan},8,640,360,.5,30,NativeImportedSceneRenderUse::Preview,
        &native,nullptr,missing,owners,error,nullptr,{3,4})==VisualImportedScenePreparation::rejected
        && missing.texture==123 && missing.particleParameters.preparedImportedSolids==receipt
        && owners.empty() && native.previewCalls==calls,
        "missing compiled cache preserves the previous solid layer and receipt before admission");
    auto malformed=plan; malformed.operations[1].payloadXml="invalid retained scene";
    check(prepareVisualImportedSceneLayerAtTime({malformed},8,640,360,.5,30,NativeImportedSceneRenderUse::Preview,
        &native,&cache,missing,owners,error,nullptr,{3,4})==VisualImportedScenePreparation::rejected
        && missing.texture==123 && missing.particleParameters.preparedImportedSolids==receipt
        && owners.empty() && native.previewCalls==calls,
        "compiled scene refusal preserves the previous solid layer and receipt before resolution");
    check(prepareVisualImportedSceneLayerAtTime({plan},8,640,360,.5,30,NativeImportedSceneRenderUse::Preview,
        &native,&cache,missing,owners,error,nullptr,{3,4})==VisualImportedScenePreparation::rejected
        && missing.texture==123 && owners.empty() && native.previewCalls==calls,
        "missing historical replay refuses atomically before native scene preparation");

    PortContract meshContract; meshContract.carrier=CarrierKind::geometry3D;
    meshContract.maxVertices=256; meshContract.maxIndices=384;
    const auto rest=lowerGrid(18000,18000,1,9,9,.25f,meshContract,error);
    const auto water=rest ? lowerWaterSurfaceSimulation(*rest,18001,18002,0,0,0,0,meshContract,error,100):std::nullopt;
    const auto waterValue=water ? admitValue(*water,meshContract,error):std::nullopt;
    check(waterValue.has_value(),"imported coupled-water fixture admits its bounded positive-mass rest grid");
    if (!waterValue) return;
    auto waterScene=std::make_shared<Visual3DScene>(*request->sceneSnapshot);
    waterScene->objectCount=waterScene->materialCount=waterScene->vertexCount=waterScene->indexCount=0;
    auto cube=lowerMeshGenerator(18003,18003,1,OperationCode::cube,{.7f,.7f,.7f,0},meshContract,error);
    auto cubeValue=cube ? admitValue(*cube,meshContract,error):std::nullopt;
    {
        const bool conditionOk = appendGeometryToScene(*waterValue,request->sceneSnapshot->objects[0],request->sceneSnapshot->materials[0],*waterScene,error)
        && cubeValue && appendGeometryToScene(*cubeValue,request->sceneSnapshot->objects[1],request->sceneSnapshot->materials[1],*waterScene,error);
        check(conditionOk, "imported water and unchanged scene camera/light state compose at authored rest: "+error);
    }
    auto waterRequest=*request; waterRequest.sceneSnapshot=waterScene;
    waterRequest.timelineGeometry=TimelineSceneBinding{{10},encodeLoweredPlanText(lowerRuntimePlan(meshContract,*waterValue))};
    ValueDescriptor emitter; emitter.carrier=CarrierKind::points3D; emitter.stableId=18100;
    emitter.sourceStableId=18100; emitter.sourceRevision=1; emitter.data=PointsData{};
    std::get<PointsData>(emitter.data).points.push_back({{-.4f,-.2f,.12f},18101,1,{0,0,0,1}});
    PortContract pointsContract; pointsContract.carrier=CarrierKind::points3D; pointsContract.maxPoints=64;
    const auto emitted=admitValue(emitter,pointsContract,error);
    check(emitted.has_value(),"exact water body emitter admits");
    if (!emitted) return;
    const auto coupledXml="<NodeParams motionMode=\"2\" simulationSpace=\"1\" historicalReplay=\"1\" count=\"1\" "
        "speed=\"0\" lifetime=\"1\" resetMode=\"1\" impulseZ=\"-4\" restitution=\"1\" colliderThickness=\"0\" "
        "colliderGeometryPlan=\""+waterRequest.timelineGeometry->loweredPlan+"\" emitterGeometryPlan=\""
        +encodeLoweredPlanText(lowerRuntimePlan(pointsContract,*emitted))+"\"/>";
    auto coupledPlan=plan;
    coupledPlan.operations[1].payloadXml=visualimportedscenerender::encode(waterRequest);
    coupledPlan.operations[2].payloadXml=coupledXml;
    {
        const bool conditionOk = videowire::compileVisualLayerExecution(coupledPlan,compiled,error);
        check(conditionOk, "imported positive-mass water graph admits its existing scene route: "+error);
    }
    ParticleParams coupled;
    {
        const bool conditionOk = videowire::admitParticleParameters(coupledXml,coupled,error);
        check(conditionOk, "imported water/body parameters admit: "+error);
    }
    auto history=std::make_shared<ParticleHistorySource>(waterreplayfixture::source());
    coupled.history=history;
    struct CoupledResult {
        ParticleParams parameters;
        std::shared_ptr<const WaterCoupledFrame> water;
        std::shared_ptr<const Visual3DScene> source,scene;
    };
    const auto composeWater=[&](double seconds,PlanUse use)->CoupledResult {
        CoupledResult result; result.parameters=coupled; result.parameters.historyProjectSeconds=seconds;
        auto canonical=waterreplayfixture::frame(*history,seconds,30,use==PlanUse::exportRender);
        ParticleBodyState bodies; auto state=std::make_shared<ParticleSolidState>(); WaterCoupledFrame completed;
        if (!replayWaterCoupledBodies(result.parameters,seconds,640.0f/360,canonical.get(),bodies,*state,nullptr,error,nullptr,&completed)) return result;
        result.water=std::make_shared<const WaterCoupledFrame>(std::move(completed));
        result.parameters.preparedBodies=std::make_shared<const ParticleBodyState>(std::move(bodies));
        result.parameters.preparedSolids=state;
        auto frameOwner=owner; frameOwner.use=use;
        result.source=evaluateComposedTimelineGeometry(runtime,frameOwner,*waterRequest.timelineGeometry,waterScene,
            seconds,error,history.get(),seconds,canonical,result.water.get());
        if (result.source)
            result.scene=appendParticleSolidsToScene(result.source,*waterScene,*waterRequest.timelineGeometry,result.parameters,*state,90,error);
        return result;
    };
    auto impact=composeWater(1.0/120,PlanUse::preview), following=composeWater(2.0/120,PlanUse::preview);
    {
        const bool conditionOk = impact.scene && following.scene && impact.parameters.preparedSolids->bodies[0].vz>0
        && std::any_of(following.water->heights.begin(),following.water->heights.end(),[](double height) { return height<0; });
        check(conditionOk, "same imported replay transfers contact impulse to the sphere and following-step water displacement: "+error);
    }
    if (!following.scene) return;
    const auto& collider=following.parameters.preparedSolids->colliders;
    check(std::equal(following.source->vertices.begin(),following.source->vertices.begin()+81,collider.meshVertices.begin(),
        [&](const auto& vertex,const auto& point) {
            const auto& binding=*following.parameters.geometryBinding;
            return point==SolidVector{binding.centerX+binding.scale*vertex.position.x,binding.centerY+binding.scale*vertex.position.y,
                binding.centerZ+binding.scale*vertex.position.z};
        }) && following.scene->objectCount==3 && following.scene->activeCamera==waterScene->activeCamera,
        "Timeline mesh, completed collider and appended body share the same water result and imported camera");
    auto exported=composeWater(2.0/120,PlanUse::exportRender), sought=composeWater(0,PlanUse::preview);
    sought=composeWater(2.0/120,PlanUse::preview);
    check(exported.scene && sought.scene
        && sceneaov::detail::encodeScene(*following.scene)==sceneaov::detail::encodeScene(*exported.scene)
        && sceneaov::detail::encodeScene(*following.scene)==sceneaov::detail::encodeScene(*sought.scene)
        && following.water->heights==exported.water->heights,
        "same-time direct seek, preview and export regenerate identical coupled imported scenes");
    auto unaligned=composeWater(.079,PlanUse::preview), unalignedExport=composeWater(.079,PlanUse::exportRender);
    {
        const bool conditionOk = unaligned.scene && unalignedExport.scene
        && sceneaov::detail::encodeScene(*unaligned.scene)==sceneaov::detail::encodeScene(*unalignedExport.scene);
        check(conditionOk, "common project-zero 0.079-second queries retain exact water/body alignment at 30 fps: "+error);
    }
    auto currentCanonical=waterreplayfixture::frame(*history,2.0/120,30);
    const auto waterBytes=sceneaov::detail::encodeScene(*waterScene), completedBytes=sceneaov::detail::encodeScene(*following.scene);
    check(!evaluateComposedTimelineGeometry(runtime,owner,*waterRequest.timelineGeometry,waterScene,.079,error,
        history.get(),2.0/120,currentCanonical,following.water.get())
        && sceneaov::detail::encodeScene(*waterScene)==waterBytes && sceneaov::detail::encodeScene(*following.scene)==completedBytes,
        "incompatible current scene time refuses before replacing either authored or completed imported scene");
    check(!evaluateComposedTimelineGeometry(runtime,owner,*waterRequest.timelineGeometry,waterScene,2.0/120,error,
        history.get(),2.0/120,currentCanonical),"positive-mass imported water refuses without the completed coordinator frame");
    auto wrongWater=std::make_shared<WaterCoupledFrame>(*following.water); ++wrongWater->operation.stableId;
    check(!evaluateComposedTimelineGeometry(runtime,owner,*waterRequest.timelineGeometry,waterScene,2.0/120,error,
        history.get(),2.0/120,currentCanonical,wrongWater.get()),"another exact water operation cannot borrow the imported completed frame");
    auto coupledReceipt=std::make_shared<ParticleImportedSolidFrame>();
    coupledReceipt->owner=owner; coupledReceipt->nodeId=90; coupledReceipt->projectSeconds=coupledReceipt->clipSeconds=2.0/120;
    coupledReceipt->bodies=following.parameters.preparedBodies; coupledReceipt->state=following.parameters.preparedSolids;
    coupledReceipt->geometry=following.parameters.geometryBinding; coupledReceipt->water=following.water;
    coupledReceipt->sourceScene=following.source; coupledReceipt->scene=following.scene; coupledReceipt->frame=nativeReceipt.frame;
    layer.particleParameters=following.parameters; layer.particleParameters.preparedImportedSolids=coupledReceipt;
    layer.particleProjectSeconds=layer.shaderClock.timeSec=2.0/120;
    layer.nativeTextureDescriptor=coupledReceipt->frame->colorTextureDescriptor();
    int replayCalls=0;
    history->parameterAt=[&](const auto&,double,double&,std::string& diagnostic) {
        ++replayCalls; diagnostic="second coupled replay is forbidden"; return false;
    };
    {
        const bool conditionOk = videowire::executeVisualLayerPlan({coupledPlan},8,layer,error,PlanUse::preview,
        nullptr,nullptr,nullptr,2.0/120,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,640,360,&context)
        && replayCalls==0 && !layer.particleSource && layer.particleParameters.preparedImportedSolids->water==following.water
        && layer.texture==44 && layer.depthTexture==66;
        check(conditionOk, "imported shared execution retains the exact coordinated pointer and performs no second water/body replay: "+error);
    }
    auto absentWater=std::make_shared<ParticleImportedSolidFrame>(*coupledReceipt); absentWater->water.reset();
    layer.particleParameters.preparedImportedSolids=absentWater;
    check(!videowire::executeVisualLayerPlan({coupledPlan},8,layer,error,PlanUse::preview,
        nullptr,nullptr,nullptr,2.0/120,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,640,360,&context)
        && replayCalls==0 && layer.texture==44,"missing coupled frame refuses without replaying or publishing an uncoupled result");
    history->parameterAt={};
    missing.particleHistory=history; missing.particleProjectSeconds=.079; missing.shaderClock.timeSec=.079;
    missing.canonicalBlockCFrame=waterreplayfixture::frame(*history,.079,30);
    check(prepareVisualImportedSceneLayerAtTime({coupledPlan},8,640,360,.080,30,NativeImportedSceneRenderUse::Preview,
        &native,&cache,missing,owners,error,nullptr,{3,4},nullptr,false,nullptr,.079)==VisualImportedScenePreparation::rejected
        && missing.texture==123 && missing.particleParameters.preparedImportedSolids==receipt
        && owners.empty() && native.previewCalls==calls && error.find("project-zero source")!=std::string::npos,
        "actual imported preparation rejects a retimed coupled clock before any native frame publication");
    auto cyclicPlan=coupledPlan;
    const auto resetAt=cyclicPlan.operations[2].payloadXml.find("resetMode=\"1\"");
    cyclicPlan.operations[2].payloadXml.replace(resetAt,std::string("resetMode=\"1\"").size(),"resetMode=\"0\"");
    check(prepareVisualImportedSceneLayerAtTime({cyclicPlan},8,640,360,.079,30,NativeImportedSceneRenderUse::Preview,
        &native,&cache,missing,owners,error,nullptr,{3,4},nullptr,false,nullptr,.079)==VisualImportedScenePreparation::rejected
        && missing.texture==123 && missing.particleParameters.preparedImportedSolids==receipt
        && owners.empty() && native.previewCalls==calls,
        "cyclic coupled replay refuses before replacing the prior body or native frame");
    missing.shaderClock.timeSec=.04;
    check(prepareVisualImportedSceneLayerAtTime({coupledPlan},8,640,360,.079,30,NativeImportedSceneRenderUse::Preview,
        &native,&cache,missing,owners,error,nullptr,{3,4},nullptr,false,nullptr,.079)==VisualImportedScenePreparation::rejected
        && missing.texture==123 && missing.particleParameters.preparedImportedSolids==receipt
        && owners.empty() && native.previewCalls==calls,
        "late clip body clock refuses before replacing the prior coupled layer");
}

void testTimelineFamilyTransport()
{
    using namespace videowire::geometry;
    using namespace HarmonicMIDI::grid;
    std::string error;
    const auto source=timelinefamilyfixture::source(error);
    {
        const bool conditionOk = bool(source);
        check(conditionOk, "family transport starts with two genuine different lowered timeline plans: "+error);
    }
    if (!source) return;
    visualimportedscenerender::Request family;
    family.sourceStableId=7; family.renderStableId=12;
    family.structuralRevision=family.evaluationRevision=9;
    family.sceneSnapshot=source->scene; family.timelineGeometryFamily=source->bindings;
    const auto encoded=visualimportedscenerender::encode(family);
    visualimportedscenerender::Request decoded;
    check(!encoded.empty() && encoded.find(visualimportedscenerender::kCompiledOperationHeaderV21)==0
        && visualimportedscenerender::decode(encoded,decoded)
        && visualimportedscenerender::sameRequest(family,decoded)
        && visualimportedscenerender::encode(decoded)==encoded,
        "V21 conserves every sorted child ID, exact lowered bytes and authored scene record");
    if (encoded.empty()) return;
    auto base=family; base.timelineGeometryFamily.clear();
    const auto oldV8=visualimportedscenerender::encode(base);
    const auto wire=[&](const std::vector<TimelineSceneBinding>& bindings,const std::string& nested) {
        std::string result=std::string(visualimportedscenerender::kCompiledOperationHeaderV21)+"\n"
            +std::to_string(bindings.size())+" "+std::to_string(nested.size())+"\n";
        for (const auto& binding : bindings)
            result+=std::to_string(binding.object.value)+" "+std::to_string(binding.loweredPlan.size())
                +"\n"+binding.loweredPlan+"\n";
        return result+nested;
    };
    check(encoded==wire(source->bindings,oldV8),"V21 is exactly the new length-delimited wrapper around unchanged V8 bytes");
    const auto singleFixture=composedTimelineGeometryFixture(error);
    check(bool(singleFixture),"legacy wire conservation uses the existing genuine single-child producer");
    if (!singleFixture) return;
    {
        auto staticSingle=*singleFixture; staticSingle.timelineGeometry.reset();
        const auto previousV8=visualimportedscenerender::encode(staticSingle);
        const auto& binding=*singleFixture->timelineGeometry;
        const auto previousV14=std::string(visualimportedscenerender::kCompiledOperationHeaderV14)+"\n"
            +std::to_string(binding.object.value)+" "+std::to_string(binding.loweredPlan.size())+"\n"
            +binding.loweredPlan+"\n"+previousV8;
        check(visualimportedscenerender::encode(*singleFixture)==previousV14
            && visualimportedscenerender::decode(previousV14,decoded)
            && decoded.timelineGeometry && decoded.timelineGeometryFamily.empty()
            && visualimportedscenerender::encode(decoded)==previousV14,
            "one-child V14 retains its exact old wrapper and V8 nested bytes");
    }
    auto leaf=renderRequest(); leaf.renderStableId=family.renderStableId;
    leaf.structuralRevision=leaf.evaluationRevision=9;
    auto composed=family; composed.importedScene=std::make_shared<const visualimportedscenerender::Request>(leaf);
    const auto leafBytes=visualimportedscenerender::encode(leaf);
    const auto wrap=[&](const std::string& scene) {
        return std::string(visualimportedscenerender::kCompiledOperationHeaderV20)+"\n"
            +std::to_string(scene.size())+" "+std::to_string(leafBytes.size())+"\n"+scene+leafBytes;
    };
    const auto composedBytes=visualimportedscenerender::encode(composed);
    check(!composedBytes.empty() && composedBytes==wrap(encoded)
        && visualimportedscenerender::decode(composedBytes,decoded)
        && visualimportedscenerender::sameRequest(composed,decoded),
        "V20 can contain exact V21 procedural bytes while preserving the old leaf wire");
    auto oldComposition=base; oldComposition.importedScene=composed.importedScene;
    check(visualimportedscenerender::encode(oldComposition)==wrap(oldV8),
        "V20 without a family retains its released scene/leaf concatenation byte for byte");
    auto selectedLeaf=leaf; selectedLeaf.sourceCameraNodeId=7;
    auto selectedComposition=composed;
    selectedComposition.importedScene=std::make_shared<const visualimportedscenerender::Request>(selectedLeaf);
    const auto selectedBytes=visualimportedscenerender::encode(selectedComposition);
    check(!selectedBytes.empty() && visualimportedscenerender::decode(selectedBytes,decoded)
        && decoded.importedScene && decoded.importedScene->sourceCameraNodeId==7
        && visualimportedscenerender::sameRequest(selectedComposition,decoded),
        "selected imported camera conserves the exact V21 geometry family and its separate leaf owner");
    selectedComposition.camera=selectedComposition.sceneSnapshot->cameras[0];
    const auto overrideBytes=visualimportedscenerender::encode(selectedComposition);
    check(!overrideBytes.empty() && visualimportedscenerender::decode(overrideBytes,decoded)
        && decoded.camera && decoded.importedScene->sourceCameraNodeId==7
        && visualimportedscenerender::sameRequest(selectedComposition,decoded),
        "V22 full authored override preflights the V20/V21 family while retaining selected source camera 7");
    auto mismatchedOverride=selectedComposition; mismatchedOverride.camera->farPlane+=1;
    check(!visualimportedscenerender::valid(mismatchedOverride)
        && visualimportedscenerender::encode(mismatchedOverride).empty(),
        "composition override must equal the actual active authored camera record");
    selectedLeaf.sourceCameraNodeId.reset();
    selectedComposition.importedScene=std::make_shared<const visualimportedscenerender::Request>(selectedLeaf);
    selectedComposition.camera.reset();
    check(visualimportedscenerender::encode(selectedComposition)==composedBytes,
        "removing selected camera and override restores the original V20/V21 bytes exactly");
    auto namedLeaf=leaf;
    namedLeaf.asset.id="model-"+std::string(visualimportedscenerender::kCompiledOperationHeaderV21)+"-1";
    auto namedLegacy=oldComposition;
    namedLegacy.importedScene=std::make_shared<const visualimportedscenerender::Request>(namedLeaf);
    const auto namedBytes=visualimportedscenerender::encode(namedLegacy);
    check(!namedBytes.empty() && visualimportedscenerender::decode(namedBytes,decoded)
        && visualimportedscenerender::encode(decoded)==namedBytes,
        "legacy V20 authored IDs containing V21 header text retain their old bytes");
    const auto refuseWire=[&](const std::string& bytes) {
        auto output=base;
        return !visualimportedscenerender::decode(bytes,output)
            && visualimportedscenerender::encode(output)==oldV8;
    };
    const auto refuseRequest=[&](const visualimportedscenerender::Request& request) {
        return !visualimportedscenerender::valid(request) && visualimportedscenerender::encode(request).empty();
    };
    auto wrong=family; wrong.timelineGeometry=source->bindings[0];
    check(refuseRequest(wrong),"single and family representations cannot be mixed");
    check(refuseWire(wire(source->bindings,visualimportedscenerender::encode(*singleFixture))),
        "V21 refuses a nested single binding without replacing output");
    wrong=family; wrong.timelineGeometryFamily[1].object=wrong.timelineGeometryFamily[0].object;
    check(refuseRequest(wrong) && refuseWire(wire(wrong.timelineGeometryFamily,oldV8)),
        "duplicate child IDs refuse before expensive decode and preserve output");
    wrong=family; std::reverse(wrong.timelineGeometryFamily.begin(),wrong.timelineGeometryFamily.end());
    check(refuseRequest(wrong) && refuseWire(wire(wrong.timelineGeometryFamily,oldV8)),
        "noncanonical child ordering refuses instead of selecting a first child");
    wrong=family; wrong.timelineGeometryFamily[1].object={999};
    check(refuseRequest(wrong) && refuseWire(wire(wrong.timelineGeometryFamily,oldV8)),
        "an absent object cannot claim a lowered plan");
    auto overlapping=std::make_shared<Visual3DScene>(*source->scene);
    overlapping->objects[1].firstVertex=overlapping->objects[0].firstVertex;
    overlapping->objects[1].firstIndex=overlapping->objects[0].firstIndex;
    wrong=family; wrong.sceneSnapshot=overlapping;
    check(refuseRequest(wrong),"family children require mesh ranges exclusive of all bound and unbound objects");
    wrong=family; wrong.timelineGeometryFamily[1].loweredPlan=wrong.timelineGeometryFamily[0].loweredPlan;
    check(refuseRequest(wrong) && refuseWire(wire(wrong.timelineGeometryFamily,oldV8)),
        "different baseline meshes cannot silently become one cloned plan");
    wrong=family; wrong.timelineGeometryFamily[1].loweredPlan[0]='g';
    check(refuseRequest(wrong) && refuseWire(wire(wrong.timelineGeometryFamily,oldV8)),"noncanonical lowered plan hex refuses before decoder staging");
    wrong=family; wrong.timelineGeometryFamily.resize(1);
    check(refuseRequest(wrong) && refuseWire(wire(wrong.timelineGeometryFamily,oldV8)),"a one-child family refuses in favor of old V14");
    wrong=family; wrong.timelineGeometryFamily.resize(65,source->bindings[0]);
    check(refuseRequest(wrong) && refuseWire(wire(wrong.timelineGeometryFamily,oldV8)),"family count above 64 refuses before record allocation");
    wrong=family; wrong.timelineGeometryFamily[0].loweredPlan.resize(kMaximumTimelineScenePlanBytes+2,'0');
    check(refuseRequest(wrong) && refuseWire(wire(wrong.timelineGeometryFamily,oldV8)),"oversized child text refuses without replacing output");
    wrong=family;
    for (auto& binding : wrong.timelineGeometryFamily) binding.loweredPlan.resize(kMaximumTimelineScenePlanBytes/2+2,'0');
    check(refuseRequest(wrong) && refuseWire(wire(wrong.timelineGeometryFamily,oldV8)),"aggregate text and decoded byte capacity is checked before any child decode");
    check(refuseWire(encoded.substr(0,encoded.size()-1)) && refuseWire(encoded+"extra")
        && refuseWire(wire(source->bindings,encoded)),"truncated, trailing and recursively nested families preserve output");
    // Add real identity Transform Geometry operations to both admitted sources.
    // Each source is valid on its own. Together their actual 130 operations
    // exceed the family bound before runtime descriptor allocation.
    auto crowded=source->bindings;
    PortContract contract; contract.carrier=CarrierKind::geometry3D;
    contract.maxVertices=4096; contract.maxIndices=12288;
    for (std::size_t i=0;i<crowded.size();++i) {
        const auto admitted=decodeTimelineScenePlan(crowded[i],error);
        if (!admitted) { check(false,"aggregate fixture retains genuine timeline admission"); return; }
        auto value=admitted->descriptor();
        while (value.operations.size()<65) {
            const auto transformed=lowerTransformGeometry(value,20000+i*100+value.operations.size(),Transform{},contract,error);
            if (!transformed) { check(false,"aggregate fixture lowers real identity transforms: "+error); return; }
            value=*transformed;
        }
        const auto final=admitValue(value,contract,error);
        if (!final) { check(false,"each 65-operation source independently admits: "+error); return; }
        crowded[i].loweredPlan=encodeLoweredPlanText(lowerRuntimePlan(contract,*final));
    }
    wrong=family; wrong.timelineGeometryFamily=crowded;
    check(refuseRequest(wrong) && refuseWire(wire(crowded,oldV8)),"130 genuine operations refuse the aggregate 128-operation budget");
    visualimportedscenerender::detail::TimelineFamilyCodecEnvelope envelope;
    check(visualimportedscenerender::detail::timelineFamilyEnvelope(source->bindings,1,composedBytes.size(),envelope)
        && envelope.familyValidationPasses==10 && envelope.planDecodeCalls==20
        && envelope.decodedBytes==envelope.textBytes/2 && envelope.operations==11
        && envelope.canonicalByteVisits==40*composedBytes.size(),
        "codec recipe separately counts V20 nested validation, canonical passes, exact plans and byte copies");
    auto timePlan=composedScenePlan(); timePlan.nodeIds={6,11};
    timePlan.nodeKinds={kRetainedSceneOperation,"visual.3d.render"};
    timePlan.operations={{6,timePlan.nodeKinds[0],"control-eval",""},{11,timePlan.nodeKinds[1],"native-gpu",encoded}};
    timePlan.edges={{6,0,11,0}};
    timePlan.ports={{6,0,1,"out","control","scene3D","unspecified","unspecified"},
        {11,0,1,"in","control","scene3D","unspecified","unspecified"},
        {11,1,1,"out","frame","image","rgba8","sRGB"}};
    videowire::VisualLayerExecution compiled;
    check(videowire::compileVisualLayerExecution(timePlan,compiled,error)
        && compiled.importedSceneRender && compiled.importedSceneRender->timelineGeometryFamily.size()==2,
        "released visual plan compilation transfers both exact children without helper evaluation");
    for (const auto use : {videohelper::importedscene::NativeImportedSceneRenderUse::Preview,
                           videohelper::importedscene::NativeImportedSceneRenderUse::Export}) {
        FakeLayer layer; layer.texture=731; layer.nativeTextureView=732; layer.shaderClock.time=0.375;
        FakeExecution execution; std::vector<FakeReceipt> receipts;
        videohelper::importedscene::VisualImportedScenePlanCache cache;
        {
            const bool conditionOk = videohelper::importedscene::prepareVisualImportedSceneLayerAtTime(
            std::vector<videowire::CompiledVisualLayerPlan>{timePlan},8,640,360,0.5,24,use,
            &execution,&cache,layer,receipts,error)==videohelper::importedscene::VisualImportedScenePreparation::rejected
            && error.find("identity is incomplete")!=std::string::npos
            && layer.texture==731 && layer.nativeTextureView==732 && layer.shaderClock.time==0.375
            && receipts.empty() && execution.previewCalls==0 && execution.exportCalls==0 && execution.staticGpuUploads==0;
            check(conditionOk, "production family with no project/helper owner preserves the layer without rendering its baseline: "+error);
        }
    }
}

// Owned native capability interface for algorithm dispatch evidence. The real
// payload executor can capture preparation, but this backend never submits GPU work.
class FamilyProductionBackend final : public arbitgpu::NativeFixtureSceneBackend
{
public:
    std::string device = "family-production-fixture-device";
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> captured;
    int prepareCalls = 0;
    arbitgpu::BackendInfo info() const override { return {true,false,"opengl",device,{}}; }
    arbitgpu::GeometryCoreExecutionCapabilities geometryCoreCapabilities() const override
    {
        const videowire::geometry::ResourceLimits limits;
        return {true,true,true,false,videohelper::geometry::carrierBit(videowire::geometry::CarrierKind::geometry3D),
            limits.maxVertices,limits.maxIndices,limits.maxPoints,limits.maxCurvePoints,limits.maxSplines,
            limits.maxInstances,limits.maxFieldElements,limits.maxAttributes,limits.maxOperations,
            limits.maxDispatches,limits.maxBufferBytes};
    }
    arbitgpu::NativeFixtureScenePreparation prepare(
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
        const std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram>&,
        arbitgpu::NativeFixturePrepareUse = arbitgpu::NativeFixturePrepareUse::Blocking) override
    {
        ++prepareCalls; captured = scene;
        arbitgpu::NativeFixtureScenePreparation result;
        result.error = "family fixture stops before native GPU submission";
        return result;
    }
    arbitgpu::NativeFixtureSceneSubmission render(
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>&,
        const std::shared_ptr<const arbitgpu::NativeFixtureSceneResources>&,
        std::uint32_t,std::uint32_t,arbitgpu::NativeFixtureSceneRuntimeInputs) override { return {}; }
};

videowire::CompiledVisualLayerPlan familyProductionPlan(const visualimportedscenerender::Request& request)
{
    auto plan = composedScenePlan();
    plan.structuralRevision = request.structuralRevision;
    plan.nodeIds = {6,11};
    plan.nodeKinds = {videowire::geometry::kRetainedSceneOperation,"visual.3d.render"};
    plan.operations = {{6,plan.nodeKinds[0],"control-eval",""},
        {11,plan.nodeKinds[1],"native-gpu",visualimportedscenerender::encode(request)}};
    plan.edges = {{6,0,11,0}};
    plan.ports = {{6,0,1,"out","control","scene3D","unspecified","unspecified"},
        {11,0,1,"in","control","scene3D","unspecified","unspecified"},
        {11,1,1,"out","frame","image","rgba8","sRGB"}};
    return plan;
}

visualimportedscenerender::Request familyProductionRequest(const timelinefamilyfixture::Source& source)
{
    visualimportedscenerender::Request request;
    request.sourceStableId = 7; request.renderStableId = 12;
    request.structuralRevision = 9; request.evaluationRevision = 17;
    request.sceneSnapshot = source.scene; request.timelineGeometryFamily = source.bindings;
    return request;
}

void testTimelineFamilyProductionDispatch()
{
    using namespace videohelper::importedscene;
    using namespace videohelper::geometry;
    std::string error;
    const auto source = timelinefamilyfixture::source(error);
    {
        const bool conditionOk = bool(source);
        check(conditionOk, "production family has two genuine independent timeline children: " + error);
    }
    if (!source) return;
    const auto authored = sceneaov::detail::encodeScene(*source->scene);
    auto request = familyProductionRequest(*source);
    const auto plan = familyProductionPlan(request);
    FamilyProductionBackend backend;
    auto geometryCache = std::make_unique<GeometryExecutionCache>(backend);
    VisualImportedScenePlanCache cache;
    FakeExecution execution;
    videorender::LayerDesc layer;
    std::vector<FakeReceipt> owners;
    const auto prepare = [&](const auto& candidate,double seconds,NativeImportedSceneRenderUse use,
                             ImportedSceneStaticResourceIdentity identity = {31,32}) {
        return prepareVisualImportedSceneLayerAtTime({candidate},8,640,360,seconds,24,use,
            &execution,&cache,layer,owners,error,nullptr,identity,nullptr,false,nullptr,3.125,geometryCache.get());
    };
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> escaped;
    for (const auto use : {NativeImportedSceneRenderUse::Preview,NativeImportedSceneRenderUse::Export})
    {
        {
            const bool conditionOk = prepare(plan,0.5,use) == VisualImportedScenePreparation::rendered
            && execution.lastRequest.sceneSnapshot
            && timelinefamilyfixture::literal(*execution.lastRequest.sceneSnapshot,*source->scene,0,0.25,0.25);
            check(conditionOk, "production preview/export evaluates both children before native boundary: " + error);
        }
        const auto& actual = execution.lastRequest;
        check(actual.projectGeneration == 31 && actual.helperGeneration == 32 && actual.clipId == 8
            && actual.structuralRevision == 9 && actual.evaluationRevision == 17
            && actual.frame.frame == 12 && actual.frame.rateNumerator == 24 && actual.frame.rateDenominator == 1
            && actual.runtimeInputs.timeSeconds == 0.5f && actual.staticPayloadIdentity == plan.operations[1].payloadXml,
            "production preserves separate evaluation revision, rational source frame and exact payload/project/helper/clip identity");
        escaped = actual.sceneSnapshot;
    }
    for (const double seconds : {4.5,0.25,0.0,0.5})
    {
        {
            const bool conditionOk = prepare(plan,seconds,NativeImportedSceneRenderUse::Preview) == VisualImportedScenePreparation::rendered;
            check(conditionOk, "production family supports repeated forward, backward and reset frame requests: " + error);
        }
        const auto& scene = execution.lastRequest.sceneSnapshot;
        const bool literal = scene && (seconds == 4.5 ? timelinefamilyfixture::literal(*scene,*source->scene,0,0.25,2.25)
            : seconds == 0.25 ? timelinefamilyfixture::literal(*scene,*source->scene,-0.09567085809127245,0.23096988312782168,0.125)
            : seconds == 0 ? timelinefamilyfixture::literal(*scene,*source->scene,-0.17677669529663688,0.17677669529663688,0)
            : timelinefamilyfixture::literal(*scene,*source->scene,0,0.25,0.25));
        check(literal,"production family seeks from each exact authored rest mesh and conserves unbound scene records");
    }
    const auto refusal = [&](const auto& candidate,const char* reason,bool nativeFailure = false) {
        layer.texture = 731; layer.nativeTextureView = 732; layer.shaderClock.timeSec = 0.375;
        const auto previousOwners = owners.size();
        const auto calls = execution.previewCalls;
        {
            const bool conditionOk = prepare(candidate,0.5,NativeImportedSceneRenderUse::Preview) == VisualImportedScenePreparation::rejected
            && error.find(reason) != std::string::npos && layer.texture == 731 && layer.nativeTextureView == 732
            && layer.shaderClock.timeSec == 0.375 && owners.size() == previousOwners && execution.previewCalls == calls + (nativeFailure ? 1 : 0);
            check(conditionOk, "family refusal restores the whole caller layer and conserves prior native publication: " + error);
        }
    };
    auto changedEvaluation = request; changedEvaluation.evaluationRevision = 18;
    check(prepare(familyProductionPlan(changedEvaluation),0.51,NativeImportedSceneRenderUse::Preview) == VisualImportedScenePreparation::rendered
        && execution.lastRequest.evaluationRevision == 18 && execution.lastRequest.frame.frame == 12
        && timelinefamilyfixture::literal(*execution.lastRequest.sceneSnapshot,*source->scene,0,0.25,0.25),
        "opaque evaluation revision changes without replacing the structural owner; geometry samples its rational source frame");
    auto solidPlan = plan;
    const auto solidXml = "<NodeParams motionMode=\"2\" simulationSpace=\"1\" historicalReplay=\"1\" count=\"2\" "
        "speed=\"0\" lifetime=\"2\" resetMode=\"1\" colliderGeometryPlan=\"" + source->bindings[0].loweredPlan + "\"/>";
    solidPlan.nodeIds.insert(solidPlan.nodeIds.end(),{90,91,92});
    solidPlan.nodeKinds.insert(solidPlan.nodeKinds.end(),{"visual.particles","video.blend","video.out"});
    solidPlan.operations.insert(solidPlan.operations.end(),{{90,"visual.particles","native-gpu",solidXml},
        {91,"video.blend","native-gpu","<NodeParams opacity=\"1\" mode=\"0\"/>"},{92,"video.out","native-gpu",""}});
    solidPlan.edges.insert(solidPlan.edges.end(),{{11,1,91,0},{90,1,91,2},{91,1,92,0}});
    solidPlan.ports.insert(solidPlan.ports.end(),{{90,1,1,"out","frame","image","rgba8","sRGB"},
        {91,0,1,"in","frame","image","rgba8","sRGB"},{91,1,1,"out","frame","image","rgba8","sRGB"},
        {91,2,1,"in","frame","image","rgba8","sRGB"},{91,3,1,"in","frame","image","rgba8","sRGB"},
        {92,0,1,"in","frame","image","rgba8","sRGB"}});
    refusal(solidPlan,"Imported Solid 3D over a timeline Geometry3D family requires one collider plan per child object");
    auto motion = request; motion.imageOutput = renderpassoutput::Output::Motion;
    refusal(familyProductionPlan(motion),"authenticated previous");
    geometryCache->familyEvaluationLimits.maxAdditionalBytes = 1;
    refusal(plan,"retained/scratch storage budget");
    geometryCache->familyEvaluationLimits = GeometryExecutionCache::productionFamilyEvaluationLimits();
    execution.succeed = false;
    refusal(plan,"preview rejected",true);
    execution.succeed = true;

    // All material owners must alias the completed full scene, never a baseline
    // or one child's partially updated snapshot.
    auto withSurface = request;
    for (std::size_t i = 0; i < 3; ++i)
    {
        geometrysurfacematerial::ObjectProgram entry;
        entry.program.sourceMaterial = 200 + i;
        auto& material = entry.program.material;
        material = tableSurfaceRequest(source->scene->objects[i].material.value,i != 0);
        material.scene = source->scene->id; material.sceneSnapshot = source->scene;
        material.binding.object = source->scene->objects[i].id;
        material.sceneRevision = material.structuralRevision = material.programRevision = material.binding.surfaceMaterialRevision = 9;
        material.evaluationRevision = 17;
        withSurface.surfacePrograms.push_back(std::move(entry));
    }
    {
        const bool conditionOk = prepare(familyProductionPlan(withSurface),0.5,NativeImportedSceneRenderUse::Preview) == VisualImportedScenePreparation::rendered
        && execution.lastRequest.surfacePrograms.size() == 3;
        check(conditionOk, "all bound and unbound authored Surface programs reach the production family request: " + error);
    }
    for (const auto& entry : execution.lastRequest.surfacePrograms)
        check(entry.program.material.sceneSnapshot == execution.lastRequest.sceneSnapshot,
            "each Surface material retains the same completed family alias");
    auto withMaterial = request;
    withMaterial.material = withSurface.surfacePrograms[0].program.material;
    {
        const bool conditionOk = prepare(familyProductionPlan(withMaterial),0.5,NativeImportedSceneRenderUse::Preview) == VisualImportedScenePreparation::rendered
        && execution.lastRequest.material && execution.lastRequest.material->sceneSnapshot == execution.lastRequest.sceneSnapshot;
        check(conditionOk, "ordinary object material also aliases the completed family scene: " + error);
    }

    // Preserve the released imported leaf material owner inside V20 composition.
    auto composition = request;
    auto leaf = renderRequest(true); leaf.renderStableId = 12; leaf.structuralRevision = 9; leaf.evaluationRevision = 17;
    leaf.material->scene = *HarmonicMIDI::grid::visualimportedsceneidentity::sceneId(0);
    leaf.material->sceneRevision = leaf.asset.version;
    leaf.material->structuralRevision = leaf.material->programRevision
        = leaf.material->binding.surfaceMaterialRevision = 9;
    leaf.material->evaluationRevision = 17;
    composition.importedScene = std::make_shared<const visualimportedscenerender::Request>(leaf);
    {
        const bool conditionOk = prepare(familyProductionPlan(composition),0.5,NativeImportedSceneRenderUse::Preview) == VisualImportedScenePreparation::rendered
        && execution.lastRequest.importedScene && execution.lastRequest.material
        && !execution.lastRequest.material->sceneSnapshot
        && timelinefamilyfixture::literal(*execution.lastRequest.sceneSnapshot,*source->scene,0,0.25,0.25);
        check(conditionOk, "V20 imported composition conserves its leaf material owner and the completed full family scene: " + error);
    }

    // Changing a camera outside its active range is still an exact-anchor change.
    auto changed = request;
    auto changedScene = std::make_shared<HarmonicMIDI::grid::Visual3DScene>(*source->scene);
    changedScene->cameras.back().farPlane = 78; changed.sceneSnapshot = changedScene;
    refusal(familyProductionPlan(changed),"different immutable scene family");
    ++changed.structuralRevision; changed.evaluationRevision = 19;
    {
        const bool conditionOk = prepare(familyProductionPlan(changed),0.5,NativeImportedSceneRenderUse::Preview) == VisualImportedScenePreparation::rendered
        && execution.lastRequest.sceneSnapshot->cameras.back().farPlane == 78;
        check(conditionOk, "new structural owner admits the entire changed frozen scene: " + error);
    }
    backend.device = "family-production-restarted-device";
    {
        const bool conditionOk = prepare(plan,0.5,NativeImportedSceneRenderUse::Preview,{41,42}) == VisualImportedScenePreparation::rendered;
        check(conditionOk, "actual project/helper/device replacement admits a new exact owner: " + error);
    }
    owners.clear(); execution.lastRequest = {};
    geometryCache.reset();
    check(escaped && timelinefamilyfixture::literal(*escaped,*source->scene,0,0.25,0.25)
        && authored == sceneaov::detail::encodeScene(*source->scene),
        "escaped completed scene outlives production cache and keeps both child leases without changing authored input");
}

// One Solid 3D replay meets both timeline family children. Child 10 tilts and
// child 11 slides, and each has its own emitter above it in parent space.
void testTimelineFamilySolidComposition()
{
    using namespace videohelper::importedscene;
    using namespace videohelper::geometry;
    using namespace videowire::geometry;
    using namespace videorender;
    std::string error;
    const auto source = timelinefamilyfixture::source(error);
    {
        const bool conditionOk = bool(source);
        check(conditionOk, "family Solid fixture has two independent animated children: " + error);
    }
    if (!source) return;
    const auto authored = sceneaov::detail::encodeScene(*source->scene);
    ValueDescriptor emitter; emitter.carrier=CarrierKind::points3D; emitter.stableId=18100;
    emitter.sourceStableId=18100; emitter.sourceRevision=1; emitter.data=PointsData{};
    std::get<PointsData>(emitter.data).points.push_back({{-1,-.2f,.6f},18101,1,{0,0,0,1}});
    std::get<PointsData>(emitter.data).points.push_back({{1,-.2f,.6f},18102,1,{0,0,0,1}});
    PortContract pointsContract; pointsContract.carrier=CarrierKind::points3D; pointsContract.maxPoints=64;
    const auto emitted=admitValue(emitter,pointsContract,error);
    check(emitted.has_value(),"family Solid emitters admit: " + error);
    if (!emitted) return;
    const auto familyPlans="10:"+source->bindings[0].loweredPlan+";11:"+source->bindings[1].loweredPlan;
    const auto particleXml=[&](const std::string& colliders) {
        return "<NodeParams motionMode=\"2\" simulationSpace=\"1\" historicalReplay=\"1\" count=\"2\" "
            "speed=\"0\" lifetime=\"4\" resetMode=\"1\" impulseZ=\"-4\" restitution=\"1\" colliderThickness=\"0\" "
            +colliders+" emitterGeometryPlan=\""+encodeLoweredPlanText(lowerRuntimePlan(pointsContract,*emitted))+"\"/>";
    };
    const auto familyXml=particleXml("colliderFamilyPlans=\""+familyPlans+"\"");

    // Admission: only the imported overlay gate may admit a family payload.
    ParticleParams admitted;
    check(!videowire::admitParticleParameters(familyXml,admitted,error)
        && error.find("Collider families require Imported Solid 3D")!=std::string::npos,
        "ordinary particle admission refuses collider families");
    {
        const bool conditionOk = videowire::admitParticleParameters(familyXml,admitted,error,true) && admitted.geometryBinding
        && admitted.geometryBinding->meshFamily.size()==2 && admitted.geometryBinding->meshFamily[0].object==10
        && admitted.geometryBinding->meshFamily[1].object==11
        && admitted.geometryBinding->mesh==admitted.geometryBinding->meshFamily[0].mesh;
        check(conditionOk, "the overlay gate admits one collider plan per family child in object order: " + error);
    }
    const auto refusedFamily=[&](const std::string& colliders,const char* reason,const char* message) {
        ParticleParams candidate;
        check(!videowire::admitParticleParameters(particleXml(colliders),candidate,error,true)
            && error.find(reason)!=std::string::npos,message);
    };
    refusedFamily("colliderFamilyPlans=\"10:"+source->bindings[0].loweredPlan+"\"","at least two",
        "a one-child collider family refuses");
    refusedFamily("colliderFamilyPlans=\"11:"+source->bindings[1].loweredPlan+";10:"+source->bindings[0].loweredPlan+"\"",
        "increasing order","collider family objects must increase");
    refusedFamily("colliderFamilyPlans=\"x:"+source->bindings[0].loweredPlan+";11:"+source->bindings[1].loweredPlan+"\"",
        "object:plan pairs","a malformed collider family entry refuses");
    refusedFamily("colliderFamilyPlans=\""+familyPlans+";\"","object:plan pairs","a trailing empty collider family entry refuses");
    refusedFamily("colliderFamilyPlans=\""+familyPlans+"\" colliderGeometryPlan=\""+source->bindings[0].loweredPlan+"\"",
        "replace the single collider plan","a collider family cannot also carry a single collider plan");
    if (const auto water = timelinefamilyfixture::source(error,0.5,true))
        refusedFamily("colliderFamilyPlans=\"10:"+water->bindings[0].loweredPlan+";11:"+water->bindings[1].loweredPlan+"\"",
            "Water children","Water family children refuse because coupled family Water has no shared replay");
    else check(false,"family Water fixture authors: " + error);
    // A 9x9 grid has 128 triangles and a 2x2 grid has 2, so the pair passes
    // the per-child checks and exceeds the 128-triangle family limit.
    PortContract gridContract; gridContract.carrier=CarrierKind::geometry3D;
    gridContract.maxVertices=4096; gridContract.maxIndices=12288;
    const auto dense=lowerGrid(18300,18300,1,9,9,.25f,gridContract,error);
    const auto sparse=lowerGrid(18301,18301,1,2,2,.25f,gridContract,error);
    const auto denseValue=dense ? admitValue(*dense,gridContract,error):std::nullopt;
    const auto sparseValue=sparse ? admitValue(*sparse,gridContract,error):std::nullopt;
    if (denseValue && sparseValue)
        refusedFamily("colliderFamilyPlans=\"10:"+encodeLoweredPlanText(lowerRuntimePlan(gridContract,*denseValue))
            +";11:"+encodeLoweredPlanText(lowerRuntimePlan(gridContract,*sparseValue))+"\"","exceed 256 vertices and 128 triangles",
            "family children that together exceed 128 triangles refuse");
    else check(false,"oversized family grids author: " + error);

    auto request = familyProductionRequest(*source);
    auto plan = familyProductionPlan(request);
    plan.nodeIds.insert(plan.nodeIds.end(),{90,91,92});
    plan.nodeKinds.insert(plan.nodeKinds.end(),{"visual.particles","video.blend","video.out"});
    plan.operations.insert(plan.operations.end(),{{90,"visual.particles","native-gpu",familyXml},
        {91,"video.blend","native-gpu","<NodeParams opacity=\"1\" mode=\"0\"/>"},{92,"video.out","native-gpu",""}});
    plan.edges.insert(plan.edges.end(),{{11,1,91,0},{90,1,91,2},{91,1,92,0}});
    plan.ports.insert(plan.ports.end(),{{90,1,1,"out","frame","image","rgba8","sRGB"},
        {91,0,1,"in","frame","image","rgba8","sRGB"},{91,1,1,"out","frame","image","rgba8","sRGB"},
        {91,2,1,"in","frame","image","rgba8","sRGB"},{91,3,1,"in","frame","image","rgba8","sRGB"},
        {92,0,1,"in","frame","image","rgba8","sRGB"}});
    videowire::VisualLayerExecution compiled;
    check(videowire::compileVisualLayerExecution(plan,compiled,error) && compiled.importedParticleOverlay
        && compiled.particleParameters.geometryBinding && compiled.particleParameters.geometryBinding->meshFamily.size()==2
        && compiled.particleParameters.geometryBinding->meshFamily[0].translation==std::array<float,3>{-1,0,0}
        && compiled.particleParameters.geometryBinding->meshFamily[1].translation==std::array<float,3>{1,0,0}
        && compiled.particleParameters.geometryBinding->meshFamily[1].linear==std::array<float,9>{1,0,0,0,1,0,0,0,1},
        "the overlay gate binds each collider child to its authored scene transform: " + error);
    // The plugin emits every Image port with sRGB transfer and straight alpha
    // and writes payloads through juce::XmlElement::toString, header included.
    // Snapshot validation and the overlay gate both admit that shape.
    auto pluginShaped = plan;
    for (auto& port : pluginShaped.ports)
        if (port.carrier == "frame") { port.transfer = "sRGB"; port.alpha = "straight"; }
    const std::string xmlHeader = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\n";
    pluginShaped.operations[2].payloadXml = xmlHeader + familyXml;
    pluginShaped.operations[3].payloadXml = xmlHeader + "<NodeParams opacity=\"1.0\" mode=\"0\"/>";
    error.clear();
    {
        const bool conditionOk = videowire::validateCompiledVisualLayerPlans({}, {pluginShaped}, false, error);
        check(conditionOk, "the plugin-shaped family overlay passes snapshot validation: " + error);
    }
    {
        const bool conditionOk = videowire::compileVisualLayerExecution(pluginShaped,compiled,error) && compiled.importedParticleOverlay
        && compiled.particleParameters.geometryBinding && compiled.particleParameters.geometryBinding->meshFamily.size()==2
        && compiled.particleParameters.geometryBinding->meshFamily[0].object==10
        && compiled.particleParameters.geometryBinding->meshFamily[1].object==11;
        check(conditionOk, "the overlay gate admits the plugin-shaped family overlay: " + error);
    }
    auto single = plan;
    single.operations[2].payloadXml=particleXml("colliderGeometryPlan=\""+source->bindings[0].loweredPlan+"\"");
    check(!videowire::compileVisualLayerExecution(single,compiled,error)
        && error.find("one collider plan per child object")!=std::string::npos,
        "one collider plan over a family refuses instead of colliding with one child");
    // Each variant edits child 11 of a fresh fixture, then carries that scene
    // in the production plan.
    const auto childIndex=[](const timelinefamilyfixture::Source& variant,std::uint32_t id) {
        std::size_t index=0;
        while (index<variant.scene->objectCount && variant.scene->objects[index].id.value!=id) ++index;
        return index;
    };
    // Child 10 sits under parent 100 (y 0.2). Child 11 is reparented under object
    // 12, which the fixture also keeps under 100 at y 1. The children therefore have
    // different parents, so the frame is the scene root and each collider carries
    // its full world transform.
    if (auto reparented=timelinefamilyfixture::source(error)) {
        reparented->scene->objects[childIndex(*reparented,11)].parent={12};
        auto reparentedPlan=plan;
        reparentedPlan.operations[1].payloadXml=visualimportedscenerender::encode(familyProductionRequest(*reparented));
        error.clear();
        {
            const bool conditionOk = videowire::compileVisualLayerExecution(reparentedPlan,compiled,error) && compiled.importedParticleOverlay
            && compiled.particleParameters.geometryBinding && compiled.particleParameters.geometryBinding->meshFamily.size()==2;
            check(conditionOk, "a family whose children have different parents admits: " + error);
        }
        const auto nearVector=[](const std::array<float,3>& a,const std::array<float,3>& b) {
            return std::abs(a[0]-b[0])<1e-5f && std::abs(a[1]-b[1])<1e-5f && std::abs(a[2]-b[2])<1e-5f; };
        const std::array<float,9> identity {1,0,0,0,1,0,0,0,1};
        if (compiled.importedParticleOverlay && compiled.particleParameters.geometryBinding
            && compiled.particleParameters.geometryBinding->meshFamily.size()==2)
        {
            const auto& children=compiled.particleParameters.geometryBinding->meshFamily;
            check(nearVector(children[0].translation,{-1,.2f,0}) && children[0].linear==identity
                && nearVector(children[1].translation,{1,1.2f,0}) && children[1].linear==identity,
                "different-parent children carry their full world transforms in the scene-root frame");
            // The composition checks each child's transform against the scene. With no bodies
            // it returns the source scene only when every transform matches.
            const auto reparentedScene=std::make_shared<const HarmonicMIDI::grid::Visual3DScene>(*reparented->scene);
            const ParticleSolidState noBodies{};
            const auto composed=appendParticleSolidsToFamilyScene(reparentedScene,*reparented->scene,reparented->bindings,
                compiled.particleParameters,noBodies,90,error);
            check(composed==reparentedScene,"different-parent children compose against their world transforms: " + error);
            // An ancestor of a child moving changes that child's world transform, so the
            // composition refuses rather than place the collider wrongly.
            auto moved=std::make_shared<HarmonicMIDI::grid::Visual3DScene>(*reparented->scene);
            moved->objects[childIndex(*reparented,100)].transform.translation.x+=0.5f;
            const auto movedScene=std::make_shared<const HarmonicMIDI::grid::Visual3DScene>(*moved);
            error.clear();
            {
                const bool conditionOk = !appendParticleSolidsToFamilyScene(movedScene,*moved,reparented->bindings,
                    compiled.particleParameters,noBodies,90,error)
                && error.find("collider transforms")!=std::string::npos;
                check(conditionOk, "an ancestor that moves under a different-parent family refuses at composition: " + error);
            }
        }
    }
    else check(false,"reparented family fixture authors: " + error);
    // A non-uniform child scale admits under a shared parent, and the collider keeps that scale.
    if (auto scaled=timelinefamilyfixture::source(error)) {
        scaled->scene->objects[childIndex(*scaled,11)].transform.scale.y=2.f;
        auto scaledPlan=plan;
        scaledPlan.operations[1].payloadXml=visualimportedscenerender::encode(familyProductionRequest(*scaled));
        error.clear();
        check(videowire::compileVisualLayerExecution(scaledPlan,compiled,error) && compiled.importedParticleOverlay
            && compiled.particleParameters.geometryBinding && compiled.particleParameters.geometryBinding->meshFamily.size()==2
            && compiled.particleParameters.geometryBinding->meshFamily[1].linear==std::array<float,9>{1,0,0,0,2,0,0,0,1},
            "a family child with non-uniform scale admits and keeps its scale in its collider transform: " + error);
    }
    else check(false,"scaled family fixture authors: " + error);

    FamilyProductionBackend backend;
    auto geometryCache = std::make_unique<GeometryExecutionCache>(backend);
    VisualImportedScenePlanCache cache;
    FakeExecution execution;
    LayerDesc layer;
    std::vector<FakeReceipt> owners;
    auto history = std::make_shared<ParticleHistorySource>(waterreplayfixture::source());
    struct Composed final {
        bool rendered=false;
        std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> scene;
        ParticleParams parameters;
    };
    const auto compose = [&](const videowire::CompiledVisualLayerPlan& candidate,double seconds,NativeImportedSceneRenderUse use) {
        layer.particleHistory=history; layer.particleProjectSeconds=seconds; layer.shaderClock.timeSec=seconds;
        layer.canonicalBlockCFrame=waterreplayfixture::frame(*history,seconds,24,use==NativeImportedSceneRenderUse::Export);
        Composed result;
        result.rendered = prepareVisualImportedSceneLayerAtTime({candidate},8,640,360,seconds,24,use,
            &execution,&cache,layer,owners,error,nullptr,{31,32},nullptr,false,nullptr,seconds,geometryCache.get())
            == VisualImportedScenePreparation::rendered;
        if (result.rendered) { result.scene=execution.lastRequest.sceneSnapshot; result.parameters=layer.particleParameters; }
        return result;
    };
    constexpr double bounced = 6.0/120;
    const auto direct = compose(plan,bounced,NativeImportedSceneRenderUse::Preview);
    check(direct.rendered && direct.scene && direct.parameters.preparedSolids && direct.parameters.preparedImportedSolids,
        "imported Solid 3D renders over a two-child timeline family: " + error);
    if (!direct.rendered || !direct.scene || !direct.parameters.preparedSolids) return;
    const auto& state = *direct.parameters.preparedSolids;
    check(state.count==2 && state.colliders.meshVertexCount==8 && state.colliders.meshTriangleCount==4
        && state.colliders.meshIdentity==particleColliderMeshIdentity(*direct.parameters.geometryBinding),
        "one replay frame merges both children's vertices and triangles under one family identity");
    check(state.bodies[0].vz>0 && state.bodies[0].vx<0 && state.bodies[1].vz>0 && std::abs(state.bodies[1].vx)<1e-3f,
        "one body bounces off the tilted child toward -x and the other straight up off the sliding child");
    const auto& composed = *direct.scene;
    check(composed.objectCount==6 && composed.objects[4].parent.value==100 && composed.objects[5].parent.value==100
        && composed.objects[4].transform.translation.x==(state.bodies[0].x-.5f)/.25f
        && composed.objects[5].transform.translation.x==(state.bodies[1].x-.5f)/.25f
        && authored==sceneaov::detail::encodeScene(*source->scene),
        "bodies append under the children's shared parent and the authored scene stays unchanged");
    const auto bytes = sceneaov::detail::encodeScene(composed);
    const auto exported = compose(plan,bounced,NativeImportedSceneRenderUse::Export);
    {
        const bool conditionOk = exported.rendered && exported.scene && sceneaov::detail::encodeScene(*exported.scene)==bytes;
        check(conditionOk, "preview and export compose identical family Solid scenes: " + error);
    }
    for (const double seconds : {0.5,0.0,bounced})
        {
            const bool conditionOk = compose(plan,seconds,NativeImportedSceneRenderUse::Preview).rendered;
            check(conditionOk, "family Solid composition supports forward seek, reset and loop back: " + error);
        }
    check(execution.lastRequest.sceneSnapshot && sceneaov::detail::encodeScene(*execution.lastRequest.sceneSnapshot)==bytes,
        "seeking back to the same time regenerates the same family Solid scene");

    // Motion over the family is one draw. The scene part keeps its retained
    // positions and each appended body carries its prior replay transform.
    const auto withMotion=[&](const char* name) {
        auto candidate=plan;
        auto& xml=candidate.operations[2].payloadXml;
        xml.replace(xml.rfind("/>"),2," "+std::string(name)+"=\"1\"/>");
        return candidate;
    };
    const auto motionView=withMotion("imageOutput");
    const auto viewed=compose(motionView,bounced,NativeImportedSceneRenderUse::Preview);
    const auto& motionRequest=execution.lastRequest;
    check(viewed.rendered && viewed.scene && viewed.parameters.preparedSolidMotion
        && motionRequest.runtimeInputs.imageOutput==renderpassoutput::Output::Motion
        && motionRequest.deformedSceneMotion && motionRequest.runtimeInputs.rawExportMask==0
        && motionRequest.retainedMotionVertexCount==direct.parameters.preparedImportedSolids->sourceScene->vertexCount,
        "Motion View over a timeline family admits one composed Motion draw: " + error);
    if (viewed.rendered && viewed.scene) {
        const auto& scene=*viewed.scene;
        bool sceneStill=true;
        for (std::size_t i=0;i<4;++i)
            sceneStill=sceneStill && arbitgpu::sameSceneTransform(scene.previousObjectTransforms[i],scene.objects[i].transform);
        check(scene.objectCount==6 && scene.previousObjectTransformCount==6 && sceneStill
            && !arbitgpu::sameSceneTransform(scene.previousObjectTransforms[4],scene.objects[4].transform)
            && arbitgpu::sceneMotionPositionsMatch(scene,motionRequest.retainedMotionVertexCount)
            && !arbitgpu::sceneMotionPositionsMatch(scene,motionRequest.retainedMotionVertexCount+1),
            "scene objects keep their transforms and a moving body takes its prior replay pose");
        const arbitgpu::NativeSceneMotionUniforms frame;
        const HarmonicMIDI::grid::SceneTransform3D runtime;
        check(arbitgpu::sceneObjectMotionUniforms(frame,scene,0,runtime,true).translationScale==frame.translationScale
            && arbitgpu::sceneObjectMotionUniforms(frame,scene,4,runtime,true).translationScale!=frame.translationScale,
            "scene objects draw with the frame Motion and a moving body with its own step");
        // An object-targeted transform moves only the draws it targets. For every
        // other draw the frame object step is identity, and the camera step stays.
        auto moving=frame;
        moving.rotation={0.0f,0.0f,std::sqrt(0.5f),std::sqrt(0.5f)};
        moving.translationScale={0.5f,0.25f,0.0f,1.0f};
        moving.cameraTranslation={1.0f,2.0f,3.0f,0.0f};
        const auto untargeted=arbitgpu::sceneObjectMotionUniforms(moving,scene,0,runtime,false);
        const auto targeted=arbitgpu::sceneObjectMotionUniforms(moving,scene,0,runtime,true);
        check(untargeted.rotation==std::array<float,4>{0,0,0,1}
            && untargeted.translationScale==std::array<float,4>{0,0,0,1}
            && untargeted.cameraRotation==moving.cameraRotation && untargeted.cameraTranslation==moving.cameraTranslation
            && untargeted.projection==moving.projection && untargeted.viewport==moving.viewport
            && targeted.rotation==moving.rotation && targeted.translationScale==moving.translationScale,
            "an untargeted object takes an identity object step, keeps the camera step, and a targeted one moves");
        check(arbitgpu::sceneObjectMotionUniforms(moving,scene,4,runtime,false).translationScale
            ==arbitgpu::sceneObjectMotionUniforms(frame,scene,4,runtime,true).translationScale,
            "an untargeted body with its own prior keeps only its own step");
        arbitgpu::NativeImportedSceneRuntimeInputs nodeTarget;
        nodeTarget.objectNodeStableId=1;
        check(!arbitgpu::runtimeTargetsObject(nodeTarget,scene.objects[4])
            && !arbitgpu::runtimeTargetsObject(nodeTarget,scene.objects[5])
            && arbitgpu::runtimeTargetsObject(arbitgpu::NativeImportedSceneRuntimeInputs{},scene.objects[4]),
            "appended solid bodies are never targeted by an object-targeted transform node");
        auto plain=std::make_shared<HarmonicMIDI::grid::Visual3DScene>(scene); plain->previousObjectTransformCount=0;
        check(!arbitgpu::sceneMotionPositionsMatch(*plain,motionRequest.retainedMotionVertexCount),
            "retained positions short of the scene need every object's prior transform");
    }
    const auto exportMotion=withMotion("exportMotion");
    const auto exportedMotion=compose(exportMotion,bounced,NativeImportedSceneRenderUse::Export);
    {
        const bool conditionOk = exportedMotion.rendered && exportedMotion.scene
        && execution.lastRequest.runtimeInputs.imageOutput==renderpassoutput::Output::Color
        && execution.lastRequest.runtimeInputs.rawExportMask==(1u<<static_cast<unsigned>(renderpassoutput::Output::Motion))
        && exportedMotion.scene->previousObjectTransformCount==6
        && layer.rawExportMask==execution.lastRequest.runtimeInputs.rawExportMask
        && layer.rawExportRenderId==render3dimage::renderStableIdForNodeValue(90);
        check(conditionOk, "Export Motion Data writes raw Motion from the composed export draw, keyed by the particle node: " + error);
    }
    const auto previewedExport=compose(exportMotion,bounced,NativeImportedSceneRenderUse::Preview);
    check(previewedExport.rendered && previewedExport.scene
        && execution.lastRequest.runtimeInputs.imageOutput==renderpassoutput::Output::Color
        && execution.lastRequest.runtimeInputs.rawExportMask==0
        && previewedExport.scene->previousObjectTransformCount==0 && !previewedExport.parameters.preparedSolidMotion,
        "Export Motion Data alone leaves preview as an ordinary Color draw: " + error);

    // Every child's range of the merged collider is checked against its own
    // evaluated mesh, transform and binding.
    const auto& receipt = *direct.parameters.preparedImportedSolids;
    check(receipt.sourceScene && receipt.scene==direct.scene && receipt.geometry==direct.parameters.geometryBinding,
        "the receipt retains the family binding, source scene and composed scene");
    if (!receipt.sourceScene) return;
    // A body's prior composes with its parent's current world. A recorded
    // parent that moved between frames must refuse rather than compose a wrong prior.
    {
        std::size_t parentIndex=receipt.sourceScene->objectCount;
        for (std::size_t i=0;i<receipt.sourceScene->objectCount;++i)
            if (receipt.sourceScene->objects[i].id.value==100) parentIndex=i;
        check(parentIndex<receipt.sourceScene->objectCount,"the family parent is an object of the source scene");
        if (parentIndex<receipt.sourceScene->objectCount) {
            auto recorded=std::make_shared<HarmonicMIDI::grid::Visual3DScene>(*receipt.sourceScene);
            recorded->previousObjectTransformCount=recorded->objectCount;
            for (std::size_t i=0;i<recorded->objectCount;++i) recorded->previousObjectTransforms[i]=recorded->objects[i].transform;
            const std::array<double,videorender::kParticleSolidMaxBodies> births {};
            const videorender::ParticleSolidMotionPrior prior{state,births,births,direct.parameters.bodyRadius};
            auto moved=std::make_shared<HarmonicMIDI::grid::Visual3DScene>(*recorded);
            moved->previousObjectTransforms[parentIndex].translation.x+=0.1f;
            std::string motionError;
            check(!videorender::appendParticleSolidsToFamilyScene(moved,*source->scene,source->bindings,
                    direct.parameters,state,90,motionError,&prior)
                && motionError.find("parent or ancestor")!=std::string::npos,
                "a body whose parent changed between frames refuses Motion instead of composing a wrong prior");
            const auto unchanged=videorender::appendParticleSolidsToFamilyScene(recorded,*source->scene,source->bindings,
                direct.parameters,state,90,motionError,&prior);
            check(unchanged && motionError.empty() && unchanged->previousObjectTransformCount==unchanged->objectCount,
                "an unchanged parent keeps the body prior");
        }
    }
    const auto append = [&](const ParticleParams& parameters,const ParticleSolidState& candidate) {
        return appendParticleSolidsToFamilyScene(receipt.sourceScene,*source->scene,source->bindings,
            parameters,candidate,90,error);
    };
    const auto repeated = append(direct.parameters,state);
    {
        const bool conditionOk = repeated && sceneaov::detail::encodeScene(*repeated)==bytes;
        check(conditionOk, "direct family append reproduces the composed scene: " + error);
    }
    for (const std::size_t vertex : {std::size_t(0),std::size_t(5)}) {
        auto moved = state; moved.colliders.meshVertices[vertex][2] += .001f;
        check(!append(direct.parameters,moved) && error.find("timeline family meshes")!=std::string::npos,
            "a collider vertex that left either child's evaluated mesh refuses");
    }
    auto reordered = state; std::swap(reordered.colliders.meshTriangles[2][0],reordered.colliders.meshTriangles[2][1]);
    check(!append(direct.parameters,reordered),"the second child's triangle order is checked after its vertex offset");
    auto shifted = direct.parameters;
    auto binding = std::make_shared<ParticleGeometryBinding>(*shifted.geometryBinding);
    binding->meshFamily[1].translation[0] += 1; shifted.geometryBinding = binding;
    check(!append(shifted,state) && error.find("collider transforms")!=std::string::npos,
        "a collider child transform that differs from the scene refuses");
    auto swapped = direct.parameters;
    binding = std::make_shared<ParticleGeometryBinding>(*swapped.geometryBinding);
    std::swap(binding->meshFamily[0].mesh,binding->meshFamily[1].mesh); swapped.geometryBinding = binding;
    check(!append(swapped,state),"collider plans bound to the wrong child refuse");

    // A changed child plan in the scene invalidates the particle's retained plans.
    if (const auto changedSource = timelinefamilyfixture::source(error,0.75)) {
        auto changed = familyProductionRequest(*changedSource);
        changed.structuralRevision = 10;
        auto changedPlan = plan;
        changedPlan.structuralRevision = 10;
        changedPlan.operations[1].payloadXml = visualimportedscenerender::encode(changed);
        const auto calls = execution.previewCalls;
        {
            const bool conditionOk = !compose(changedPlan,bounced,NativeImportedSceneRenderUse::Preview).rendered
            && error.find("each child's exact retained")!=std::string::npos && execution.previewCalls==calls;
            check(conditionOk, "a changed child plan refuses before native preparation: " + error);
        }
    }
    else check(false,"changed family fixture authors: " + error);
    // Holding at the end of a short lifetime would replay a different time
    // than the scene shows, so the collider check refuses it.
    auto held = plan;
    const auto at = held.operations[2].payloadXml.find("lifetime=\"4\"");
    held.operations[2].payloadXml.replace(at,std::string("lifetime=\"4\"").size(),"lifetime=\"0.25\"");
    {
        const bool conditionOk = !compose(held,0.5,NativeImportedSceneRenderUse::Preview).rendered
        && error.find("timeline family meshes")!=std::string::npos;
        check(conditionOk, "a held replay past its lifetime cannot meet a later family scene: " + error);
    }
}

void testTimelineFamilyWaterProductionDispatch()
{
    using namespace videohelper::importedscene;
    using namespace videohelper::geometry;
    std::string error;
    const auto source = timelinefamilyfixture::source(error,0.5,true);
    {
        const bool conditionOk = bool(source);
        check(conditionOk, "production Water family has distinct released grids and gains: " + error);
    }
    if (!source) return;
    auto request = familyProductionRequest(*source);
    const auto plan = familyProductionPlan(request);
    FamilyProductionBackend backend;
    GeometryExecutionCache geometryCache(backend);
    VisualImportedScenePlanCache cache;
    FakeExecution execution;
    videorender::LayerDesc layer;
    std::vector<FakeReceipt> owners;
    auto history = std::make_shared<videorender::ParticleHistorySource>(waterreplayfixture::source());
    auto score = std::make_shared<arbitmod::Score>(*history->score);
    score->notes = {waterreplayfixture::note(1,0,63.5f)}; history->score = score;
    constexpr double dt = 1.0/120, local = 23.5;
    layer.particleHistory = history;
    const auto prepare = [&](const auto& candidate,double project,double sourceTime,NativeImportedSceneRenderUse use) {
        layer.particleProjectSeconds = project;
        layer.canonicalBlockCFrame = waterreplayfixture::frame(*history,project,120,use == NativeImportedSceneRenderUse::Export);
        return prepareVisualImportedSceneLayerAtTime({candidate},8,640,360,sourceTime,24,use,&execution,&cache,
            layer,owners,error,nullptr,{31,32},nullptr,false,nullptr,project,&geometryCache);
    };
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> escaped;
    for (const auto use : {NativeImportedSceneRenderUse::Preview,NativeImportedSceneRenderUse::Export})
    {
        layer.particleHistory = history;
        {
            const bool conditionOk = prepare(plan,dt,local,use) == VisualImportedScenePreparation::rendered
            && execution.lastRequest.sceneSnapshot && timelinefamilyfixture::waterLiteral(*execution.lastRequest.sceneSnapshot,*source->scene,1);
            check(conditionOk, "production Water preview/export consumes actual history at a separate project clock: " + error);
        }
        check(execution.lastRequest.frame.frame == 564 && execution.lastRequest.frame.rateNumerator == 24
            && execution.lastRequest.runtimeInputs.timeSeconds == 23.5f
            && execution.lastRequest.runtimeInputs.canonicalBlockCFrame == layer.canonicalBlockCFrame,
            "production Water retains exact rational local frame and immutable project canonical frame");
        escaped = execution.lastRequest.sceneSnapshot;
    }
    for (const auto project : {2*dt,dt,0.0})
    {
        layer.particleHistory = history;
        {
            const bool conditionOk = prepare(plan,project,-7,NativeImportedSceneRenderUse::Preview) == VisualImportedScenePreparation::rendered
            && timelinefamilyfixture::waterLiteral(*execution.lastRequest.sceneSnapshot,*source->scene,project == 0 ? 0 : project == dt ? 1 : 2);
            check(conditionOk, "production Water seek/reset uses project replay while conserving the negative local source anchor: " + error);
        }
    }
    ++history->key.seekGeneration; ++history->key.loopGeneration;
    layer.particleHistory = history;
    {
        const bool conditionOk = prepare(plan,dt,local,NativeImportedSceneRenderUse::Preview) == VisualImportedScenePreparation::rendered
        && timelinefamilyfixture::waterLiteral(*execution.lastRequest.sceneSnapshot,*source->scene,1);
        check(conditionOk, "production Water uses the new actual seek/loop canonical lifecycle: " + error);
    }
    const auto refusal = [&](const auto& candidate,const char* reason) {
        layer.particleHistory = history; layer.texture = 731; layer.nativeTextureView = 732; layer.shaderClock.timeSec = .375;
        const auto count = owners.size(); const auto calls = execution.previewCalls;
        {
            const bool conditionOk = prepare(candidate,dt,local,NativeImportedSceneRenderUse::Preview) == VisualImportedScenePreparation::rejected
            && error.find(reason) != std::string::npos && layer.texture == 731 && layer.nativeTextureView == 732
            && layer.shaderClock.timeSec == .375 && layer.particleHistory == history
            && owners.size() == count && execution.previewCalls == calls
            && escaped && timelinefamilyfixture::waterLiteral(*escaped,*source->scene,1);
            check(conditionOk, "Water family failure preserves caller history/layer and all prior publication: " + error);
        }
    };
    const auto frameRefusal = [&](const std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame>& frame,
                                  const std::shared_ptr<const videorender::ParticleHistorySource>& borrowed,double project) {
        layer.particleHistory = borrowed; layer.canonicalBlockCFrame = frame; layer.texture = 731;
        const auto count = owners.size(); const auto calls = execution.previewCalls;
        {
            const bool conditionOk = prepareVisualImportedSceneLayerAtTime({plan},8,640,360,local,24,NativeImportedSceneRenderUse::Preview,
            &execution,&cache,layer,owners,error,nullptr,{31,32},nullptr,false,nullptr,project,&geometryCache)
                == VisualImportedScenePreparation::rejected && layer.texture == 731
            && layer.particleHistory == borrowed && layer.canonicalBlockCFrame == frame
            && owners.size() == count && execution.previewCalls == calls;
            check(conditionOk, "actual stale project/source/frame identity refuses before native preparation and restores the input: " + error);
        }
    };
    const auto current = waterreplayfixture::frame(*history,dt);
    frameRefusal(current,history,2*dt);
    auto staleHistory = std::make_shared<videorender::ParticleHistorySource>(*history);
    ++staleHistory->key.sourceGeneration;
    frameRefusal(current,staleHistory,dt);
    frameRefusal({},history,dt);
    frameRefusal(current,{},dt);
    check(geometryCache.familyEvaluationLimits.maxWaterIdentityVisits == 10000000
        && geometryCache.familyEvaluationLimits.maxWaterRestPositionVisits == 10000000
        && geometryCache.familyEvaluationLimits.maxWaterMemoCopyElements == 10000000
        && geometryCache.familyEvaluationLimits.maxWaterColumnDiscoveryVisits == 10000000,
        "production sets four finite logical pass allowances instead of compatibility-unbounded defaults");
    const std::pair<std::size_t timelineevaluation::Limits::*,std::size_t> shortages[] = {
        {&timelineevaluation::Limits::maxWaterIdentityVisits,99},
        {&timelineevaluation::Limits::maxWaterRestPositionVisits,15},
        {&timelineevaluation::Limits::maxWaterMemoCopyElements,27},
        {&timelineevaluation::Limits::maxWaterColumnDiscoveryVisits,23}
    };
    for (const auto& shortage : shortages)
    {
        geometryCache.familyEvaluationLimits = GeometryExecutionCache::productionFamilyEvaluationLimits();
        geometryCache.familyEvaluationLimits.*shortage.first = shortage.second;
        refusal(plan,"prospective rest/memo work budget");
    }
    geometryCache.familyEvaluationLimits = GeometryExecutionCache::productionFamilyEvaluationLimits();
    geometryCache.familyEvaluationLimits.waterSource.logicalBytes[static_cast<std::size_t>(videorender::watersourcebudget::Storage::waveArrays)] = 32*sizeof(double)-1;
    refusal(plan,"source payload allowance");
    geometryCache.familyEvaluationLimits = GeometryExecutionCache::productionFamilyEvaluationLimits();
    geometryCache.familyEvaluationLimits.waterSource.work[static_cast<std::size_t>(videorender::watersourcebudget::Work::censusNotes)] = 0;
    refusal(plan,"cumulative note census allowance");
    geometryCache.familyEvaluationLimits = GeometryExecutionCache::productionFamilyEvaluationLimits();

    // Derive exact storage from the same admitted production owner without
    // allocating another output. A live escaped production result stays charged.
    const PlanOwnerIdentity owner{31,32,videowire::runtimeDeviceGeneration(geometryCache.source.geometryCoreCapabilities()),8,9,PlanUse::preview};
    auto family = geometryCache.runtime.admitTimelineScenePreview(owner,*source->scene,source->bindings,{},error);
    timelineevaluation::Envelope storage;
    auto noSpace = GeometryExecutionCache::productionFamilyEvaluationLimits(); noSpace.maxAdditionalBytes = 0;
    geometryCache.familyEvaluator.evaluate(owner,family,source->scene,local,noSpace,error,&storage,dt,nullptr,
        history.get(),waterreplayfixture::frame(*history,dt));
    check(storage.concurrentBytes > 0,"production evaluator reports the prospective live escaped storage cost");
    geometryCache.familyEvaluationLimits.maxAdditionalBytes = storage.concurrentBytes-1;
    refusal(plan,"retained/scratch storage budget");
    geometryCache.familyEvaluationLimits = GeometryExecutionCache::productionFamilyEvaluationLimits();

    const auto coupled = timelinefamilyfixture::source(error,0.5,true,1,0,1);
    check(bool(coupled),"coupled family refusal uses real positive-mass lowering");
    // Use a fresh structural owner so coupling refuses its own lowering rather
    // than merely the incumbent plan-identity check.
    if (coupled) {
        auto coupledRequest = familyProductionRequest(*coupled); coupledRequest.structuralRevision = 10; coupledRequest.evaluationRevision = 18;
        refusal(familyProductionPlan(coupledRequest),"independently owned child frames");
    }
    const auto audioSource = timelinefamilyfixture::source(error,0.5,true,1,1);
    check(bool(audioSource),"second-child failure uses real positive-audio Water lowering");
    if (audioSource)
    {
        auto audioRequest = familyProductionRequest(*audioSource); audioRequest.structuralRevision = 11; audioRequest.evaluationRevision = 19;
        auto audio = waterreplayfixture::audio(2); history->audioFrames = &audio; history->audioSampleRate = 61440;
        audio[0].bands[17] = std::numeric_limits<float>::quiet_NaN();
        refusal(familyProductionPlan(audioRequest),"malformed");
        audio[0].bands[17] = 0; layer.particleHistory = history;
        {
            const bool conditionOk = prepare(familyProductionPlan(audioRequest),dt,local,NativeImportedSceneRenderUse::Preview) == VisualImportedScenePreparation::rendered
            && timelinefamilyfixture::waterLiteral(*execution.lastRequest.sceneSnapshot,*audioSource->scene,1);
            check(conditionOk, "second-child failure rolls back first replacement and releases candidate storage for retry: " + error);
        }
        history->audioFrames = nullptr;
    }

    // The real payload execution seam sees the completed family, then rejects at
    // the explicit capture backend. It publishes no native output on this path.
    FamilyProductionBackend capture;
    GeometryExecutionCache capturedGeometry(capture);
    videohelper::modelpayload::Store store;
    videohelper::modelpayload::ImportedScenePayloadExecution payloadExecution(store,capture);
    VisualImportedScenePlanCache capturedPlans;
    std::vector<videohelper::modelpayload::ImportedSceneExecutionReceipt> nativeOwners;
    layer.particleHistory = history; layer.canonicalBlockCFrame = waterreplayfixture::frame(*history,dt);
    layer.texture = 731;
    {
        const bool conditionOk = prepareVisualImportedSceneLayerAtTime({plan},8,640,360,local,24,NativeImportedSceneRenderUse::Preview,
        &payloadExecution,&capturedPlans,layer,nativeOwners,error,nullptr,{31,32},nullptr,false,nullptr,dt,&capturedGeometry)
            == VisualImportedScenePreparation::rejected && capture.prepareCalls == 1 && capture.captured
        && timelinefamilyfixture::waterLiteral(*capture.captured,*source->scene,1) && layer.texture == 731 && nativeOwners.empty();
        check(conditionOk, "real payload preparation receives both completed Water children and rolls back its native rejection: " + error);
    }
}

void testMultiLeafComposition()
{
    using namespace videohelper::importedscene;
    using namespace videohelper::modelpayload;
    namespace composition = HarmonicMIDI::grid::scenecomposition;
    check(composition::encodeImportedLeafId(0, 1, 9000u) == 9000u
            && composition::encodeImportedLeafId(0, 2, 5u) == 5u
            && composition::encodeImportedLeafId(1, 2, 5u) == 2053u
            && composition::encodeImportedLeafId(7, 8, 2047u) == 7u * 2048u + 2047u
            && !composition::encodeImportedLeafId(1, 2, 2048u)
            && !composition::encodeImportedLeafId(0, 2, 2048u)
            && !composition::encodeImportedLeafId(2, 2, 5u)
            && !composition::encodeImportedLeafId(0, 9, 5u)
            && !composition::encodeImportedLeafId(0, 0, 5u)
            && !composition::encodeImportedLeafId(1, 2, 0u)
            && !composition::encodeImportedLeafId(0, 1, 0x80000000u),
        "imported leaf IDs keep leaf 0 raw and stride later leaves by 2048");
    bool roundTrip = true;
    for (std::size_t count = 2; count <= composition::kMaximumImportedLeaves; ++count)
        for (std::size_t leaf = 0; leaf < count; ++leaf)
            for (const std::uint32_t raw : {1u, 5u, 2047u})
            {
                const auto id = composition::encodeImportedLeafId(leaf, count, raw);
                const auto decoded = id ? composition::decodeImportedLeafId(*id, count) : std::nullopt;
                roundTrip = roundTrip && decoded && decoded->leaf == leaf && decoded->raw == raw
                    && composition::importedLeafOwnsId(*id, leaf, count)
                    && !composition::importedLeafOwnsId(*id, (leaf + 1) % count, count);
            }
    const auto single = composition::decodeImportedLeafId(9000u, 1);
    check(roundTrip && single && single->leaf == 0 && single->raw == 9000u
            && !composition::decodeImportedLeafId(2048u, 2) && !composition::decodeImportedLeafId(4097u, 2)
            && !composition::decodeImportedLeafId(0u, 2),
        "imported leaf IDs decode back to their leaf and raw GLB ID");
    check(composition::importedLeafCountFits(1, 5000) && composition::importedLeafCountFits(2, 2047)
            && !composition::importedLeafCountFits(2, 2048) && !composition::importedLeafCountFits(9, 1)
            && !composition::importedLeafCountFits(0, 1)
            && composition::validImportedLeafCount(8) && !composition::validImportedLeafCount(9),
        "multi-leaf compositions cap leaves at eight and leaf node, material and texture counts below 2048");

    const auto makeLeaf = [](std::uint64_t source, double speed)
    {
        auto leaf = renderRequest(false, true);
        leaf.sourceStableId = source;
        leaf.renderStableId = 12; leaf.structuralRevision = 31; leaf.evaluationRevision = 35;
        leaf.objectNodeStableId = 5; leaf.objectMeshStableId = 1;
        leaf.deformation->sourceStableId = source;
        leaf.deformation->deformationStableId = source + 2;
        leaf.deformation->schedule = {leaf.deformation->sourceStableId, leaf.deformation->deformationStableId};
        leaf.deformation->playback.speed = speed;
        leaf.deformation->pose.nodeStableId = 5; leaf.deformation->pose.meshStableId = 1;
        return leaf;
    };
    visualimportedscenerender::Request operation;
    operation.sourceStableId = 7; operation.renderStableId = 12;
    operation.structuralRevision = 31; operation.evaluationRevision = 35;
    operation.sceneSnapshot = composedScene();
    auto first = makeLeaf(71, 1.5);
    auto second = makeLeaf(81, 0.5);
    check(visualimportedscenerender::remapComposedLeafSelectors(first, 0, 2)
            && visualimportedscenerender::remapComposedLeafSelectors(second, 1, 2)
            && first.objectNodeStableId == 5 && first.deformation->pose.nodeStableId == 5
            && second.objectNodeStableId == 2053 && second.deformation->pose.nodeStableId == 2053,
        "leaf selectors remap through the leaf ID scheme");
    auto oversized = makeLeaf(91, 1.0);
    oversized.objectNodeStableId = 2048; oversized.deformation->pose.nodeStableId = 2048;
    check(!visualimportedscenerender::remapComposedLeafSelectors(oversized, 1, 2)
            && oversized.objectNodeStableId == 2048 && oversized.deformation->pose.nodeStableId == 2048,
        "a selector at or above 2048 refuses the multi-leaf remap without changing the leaf");

    auto oneLeaf = operation;
    oneLeaf.importedScene = std::make_shared<const visualimportedscenerender::Request>(first);
    const auto oneLeafWire = visualimportedscenerender::encode(oneLeaf);
    // The V20 layout on main: header, chunk sizes, then the procedural scene
    // and the standalone leaf, each encoded by paths V26 does not touch.
    auto procedural = oneLeaf;
    procedural.importedScene.reset();
    const auto sceneChunk = visualimportedscenerender::encode(procedural);
    const auto firstChunk = visualimportedscenerender::encode(first);
    const auto expectedOneLeaf = std::string(visualimportedscenerender::kCompiledOperationHeaderV20) + "\n"
        + std::to_string(sceneChunk.size()) + " " + std::to_string(firstChunk.size()) + "\n" + sceneChunk + firstChunk;
    check(!sceneChunk.empty() && !firstChunk.empty() && oneLeafWire == expectedOneLeaf
            && !visualimportedscenerender::detail::containsRequestHeader(oneLeafWire,
                visualimportedscenerender::kCompiledOperationHeaderV26),
        "a one-leaf composition encodes the exact main V20 bytes without a V26 header");

    auto twoLeaves = oneLeaf;
    twoLeaves.additionalImportedScenes.push_back(std::make_shared<const visualimportedscenerender::Request>(second));
    const auto wire = visualimportedscenerender::encode(twoLeaves);
    const auto secondChunk = visualimportedscenerender::encode(second);
    const auto expectedTwoLeaves = std::string(visualimportedscenerender::kCompiledOperationHeaderV26) + "\n2 plain "
        + std::to_string(expectedOneLeaf.size()) + " " + std::to_string(secondChunk.size()) + "\n"
        + expectedOneLeaf + secondChunk;
    check(!secondChunk.empty() && wire == expectedTwoLeaves,
        "a two-leaf composition encodes the pinned V26 layout: count, plain mode, chunk sizes, V20 base, leaf 1");
    visualimportedscenerender::Request decoded;
    check(visualimportedscenerender::decode(expectedTwoLeaves, decoded)
            && decoded.sourceStableId == 7 && decoded.renderStableId == 12
            && decoded.structuralRevision == 31 && decoded.evaluationRevision == 35 && !decoded.camera
            && decoded.importedScene && decoded.additionalImportedScenes.size() == 1
            && decoded.importedScene->sourceStableId == 71
            && decoded.additionalImportedScenes[0]->sourceStableId == 81
            && decoded.importedScene->objectNodeStableId == 5
            && decoded.additionalImportedScenes[0]->structuralRevision == 31
            && decoded.additionalImportedScenes[0]->evaluationRevision == 35
            && decoded.importedScene->deformation && decoded.additionalImportedScenes[0]->deformation
            && decoded.importedScene->deformation->playback.speed == 1.5
            && decoded.additionalImportedScenes[0]->deformation->playback.speed == 0.5
            && decoded.additionalImportedScenes[0]->objectNodeStableId == 2053,
        "a two-leaf composition round-trips through V26 with per-leaf selectors and playback");
    auto withoutExtras = decoded;
    withoutExtras.additionalImportedScenes.clear();
    check(visualimportedscenerender::encode(withoutExtras) == expectedOneLeaf,
        "dropping the extra leaves restores the exact one-leaf V20 bytes");

    auto refused = twoLeaves;
    refused.additionalImportedScenes = {std::make_shared<const visualimportedscenerender::Request>(makeLeaf(81, 0.5))};
    check(visualimportedscenerender::encode(refused).empty(),
        "an unmapped leaf 1 selector is not owned by leaf 1 and refuses to encode");
    refused = twoLeaves;
    auto duplicate = second;
    duplicate.sourceStableId = 71; duplicate.deformation->sourceStableId = 71;
    duplicate.deformation->schedule = {duplicate.deformation->sourceStableId, duplicate.deformation->deformationStableId};
    refused.additionalImportedScenes = {std::make_shared<const visualimportedscenerender::Request>(duplicate)};
    check(visualimportedscenerender::encode(refused).empty(), "two leaves with the same source refuse to encode");
    refused = twoLeaves;
    for (std::uint64_t source = 101; refused.additionalImportedScenes.size() < composition::kMaximumImportedLeaves; source += 10)
    {
        auto extra = makeLeaf(source, 1.0);
        extra.objectNodeStableId = 0; extra.objectMeshStableId = 0;
        extra.deformation->pose.nodeStableId = 0; extra.deformation->pose.meshStableId = 0;
        refused.additionalImportedScenes.push_back(std::make_shared<const visualimportedscenerender::Request>(extra));
    }
    check(refused.additionalImportedScenes.size() + 1 == 9 && visualimportedscenerender::encode(refused).empty(),
        "a ninth imported leaf refuses to encode");
    auto tampered = wire;
    tampered.replace(visualimportedscenerender::kCompiledOperationHeaderV26.size() + 1, 1, "3");
    check(!visualimportedscenerender::decode(tampered, decoded), "a V26 leaf count that disagrees with its chunks refuses");
    tampered = wire;
    tampered.replace(visualimportedscenerender::kCompiledOperationHeaderV26.size() + 1, 1, "1");
    check(!visualimportedscenerender::decode(tampered, decoded), "a V26 wire claiming one leaf refuses");
    const auto header = std::string(visualimportedscenerender::kCompiledOperationHeaderV26) + "\n";
    bool truncatedRefused = true;
    for (const auto cut : {header.size(), header.size() + 2, wire.size() - secondChunk.size(),
                           wire.size() - secondChunk.size() / 2, wire.size() - 1})
        truncatedRefused = truncatedRefused && cut < wire.size()
            && !visualimportedscenerender::decode(std::string_view(wire).substr(0, cut), decoded);
    check(truncatedRefused, "a truncated V26 wire refuses at the header, the size line, a missing leaf and a cut leaf");
    check(!visualimportedscenerender::decode(wire + "\n", decoded), "a V26 wire with trailing bytes refuses");

    {
        // A Render Material on several leaves writes each leaf's MaterialId
        // literal in that leaf's range before admission, so the digest that
        // travels is the digest of the remapped program.
        const auto setLiteral = [](surfacematerialbinding::ImportedSceneMaterialRequest& material, std::uint32_t value)
        {
            const auto output = material.program.outputs[static_cast<std::size_t>(surfacematerial::OutputSemantic::MaterialId)];
            for (auto& operation : material.program.operations)
                if (operation.id == output) operation.unsignedLiteral = value;
        };
        auto oneLeafMaterial = materialRequest(), leafZero = materialRequest(), leafOne = materialRequest(), lastLeaf = materialRequest();
        check(geometrysurfacematerial::remapImportedLeafMaterialId(oneLeafMaterial.program, 0, 1)
                && geometrysurfacematerial::remapImportedLeafMaterialId(leafZero.program, 0, 2)
                && geometrysurfacematerial::remapImportedLeafMaterialId(leafOne.program, 1, 2)
                && geometrysurfacematerial::remapImportedLeafMaterialId(lastLeaf.program, 7, 8)
                && geometrysurfacematerial::materialId(oneLeafMaterial) == 23u
                && geometrysurfacematerial::materialId(leafZero) == 23u
                && geometrysurfacematerial::materialId(leafOne) == 2048u + 23u
                && geometrysurfacematerial::materialId(lastLeaf) == 7u * 2048u + 23u,
            "a leaf Render Material literal moves into its leaf's material range and leaf 0 keeps it");
        auto large = materialRequest(); setLiteral(large, 9000u);
        auto unbound = materialRequest(); setLiteral(unbound, 0u);
        check(geometrysurfacematerial::remapImportedLeafMaterialId(large.program, 0, 1)
                && geometrysurfacematerial::materialId(large) == 9000u
                && geometrysurfacematerial::remapImportedLeafMaterialId(unbound.program, 1, 2)
                && geometrysurfacematerial::materialId(unbound) == 0u,
            "one leaf keeps any literal and a zero literal stays zero in every leaf");
        bool refusedUnchanged = true;
        for (const std::size_t leaf : {std::size_t {0}, std::size_t {1}})
        {
            auto oversized = materialRequest(); setLiteral(oversized, 2048u);
            refusedUnchanged = refusedUnchanged
                && !geometrysurfacematerial::remapImportedLeafMaterialId(oversized.program, leaf, 2)
                && geometrysurfacematerial::materialId(oversized) == 2048u;
        }
        auto outside = materialRequest(), tooMany = materialRequest(), computed = materialRequest();
        const auto materialOutput = computed.program.outputs[static_cast<std::size_t>(surfacematerial::OutputSemantic::MaterialId)];
        for (auto& operation : computed.program.operations)
            if (operation.id == materialOutput) operation.kind = surfacematerial::OperationKind::FloatConstant;
        check(refusedUnchanged
                && !geometrysurfacematerial::remapImportedLeafMaterialId(outside.program, 2, 2)
                && !geometrysurfacematerial::remapImportedLeafMaterialId(tooMany.program, 0, 9)
                && !geometrysurfacematerial::remapImportedLeafMaterialId(computed.program, 1, 2)
                && geometrysurfacematerial::materialId(outside) == 23u,
            "a literal at or above 2048, a foreign leaf or a computed Material ID refuses the leaf remap");

        std::string admissionError;
        const auto original = surfacematerial::admit(leafZero.program, admissionError);
        const auto remapped = surfacematerial::admit(leafOne.program, admissionError);
        auto restored = leafOne; setLiteral(restored, 23u);
        const auto restoredAdmitted = surfacematerial::admit(restored.program, admissionError);
        check(original && remapped && restoredAdmitted
                && remapped->structuralDigest() != original->structuralDigest()
                && restoredAdmitted->structuralDigest() == original->structuralDigest(),
            "the remapped literal is part of the structural digest, so a literal patched after admission "
            "fails the helper's exact digest check: " + admissionError);

        // Both leaves carry the same node and primitive in their own ranges.
        const auto leafMaterial = [&](visualimportedscenerender::Request leaf, std::size_t index)
        {
            leaf.material = materialRequest();
            auto& material = *leaf.material;
            material.scene = *HarmonicMIDI::grid::visualimportedsceneidentity::sceneId(0);
            material.sceneRevision = leaf.asset.version;
            material.structuralRevision = 31;
            material.evaluationRevision = 35;
            material.programRevision = material.binding.surfaceMaterialRevision = 31;
            material.binding.object = *composition::composedImportedLeafObjectId(
                index, 2, *HarmonicMIDI::grid::visualimportedsceneidentity::objectId(0, 0));
            geometrysurfacematerial::remapImportedLeafMaterialId(material.program, index, 2);
            return leaf;
        };
        auto materials = oneLeaf;
        materials.importedScene = std::make_shared<const visualimportedscenerender::Request>(leafMaterial(first, 0));
        materials.additionalImportedScenes = {std::make_shared<const visualimportedscenerender::Request>(leafMaterial(second, 1))};
        check(materials.additionalImportedScenes[0]->material->binding.object.value == 2048u * 65536u + 1u
                && geometrysurfacematerial::materialId(*materials.additionalImportedScenes[0]->material) == 2048u + 23u
                && visualimportedscenerender::valid(materials) && !visualimportedscenerender::encode(materials).empty(),
            "two leaves each carry a Render Material bound in their own object and material ranges");
        auto foreignObject = leafMaterial(second, 1);
        foreignObject.material->binding.object = *HarmonicMIDI::grid::visualimportedsceneidentity::objectId(0, 0);
        auto foreignLiteral = leafMaterial(second, 1);
        setLiteral(*foreignLiteral.material, 23u);
        auto refusedMaterial = materials;
        refusedMaterial.additionalImportedScenes = {std::make_shared<const visualimportedscenerender::Request>(foreignObject)};
        const bool objectRefused = !visualimportedscenerender::valid(refusedMaterial);
        refusedMaterial.additionalImportedScenes = {std::make_shared<const visualimportedscenerender::Request>(foreignLiteral)};
        check(objectRefused && !visualimportedscenerender::valid(refusedMaterial),
            "a leaf 1 Render Material naming a leaf 0 object or Material ID refuses");
        // A MaterialId output that is not one unsigned constant reads as zero
        // through materialId(), so validation must refuse it explicitly.
        auto computedLeaf = leafMaterial(second, 1);
        const auto computedOutput = computedLeaf.material->program.outputs[
            static_cast<std::size_t>(surfacematerial::OutputSemantic::MaterialId)];
        for (auto& operation : computedLeaf.material->program.operations)
            if (operation.id == computedOutput) operation.kind = surfacematerial::OperationKind::FloatConstant;
        auto refusedComputed = materials;
        refusedComputed.additionalImportedScenes = {std::make_shared<const visualimportedscenerender::Request>(computedLeaf)};
        check(!geometrysurfacematerial::materialIdIsConstant(*computedLeaf.material)
                && !visualimportedscenerender::valid(refusedComputed),
            "a leaf Render Material whose Material ID is not one constant refuses instead of reading as no material");
        auto remappedSelectors = leafMaterial(second, 1);
        const auto boundObject = remappedSelectors.material->binding.object;
        remappedSelectors.objectNodeStableId = 5;
        remappedSelectors.deformation->pose.nodeStableId = 5;
        check(visualimportedscenerender::remapComposedLeafSelectors(remappedSelectors, 1, 2)
                && remappedSelectors.material->binding.object == boundObject
                && geometrysurfacematerial::materialId(*remappedSelectors.material) == 2048u + 23u,
            "selector remap leaves the already lowered leaf material unchanged");

        // The helper adds one Render override per leaf beside the procedural
        // programs; source zero carries a different program for each leaf.
        auto composed = std::make_shared<HarmonicMIDI::grid::Visual3DScene>(*composedScene());
        // As in importedOverrideComposedScene, each target material owns the
        // imported base color texture its override samples.
        composed->materials[0].baseColorTexture.value = 1;
        composed->materials[1].baseColorTexture.value = 1;
        geometrysurfacematerial::ObjectPrograms programs;
        auto zeroOverride = *materials.importedScene->material;
        auto oneOverride = *materials.additionalImportedScenes[0]->material;
        zeroOverride.binding.object = composed->objects[0].id;
        oneOverride.binding.object = composed->objects[1].id;
        setLiteral(zeroOverride, composed->objects[0].material.value);
        setLiteral(oneOverride, composed->objects[1].material.value);
        // Each override's digest is the admission digest of the program it
        // carries after its literal moved, as the plugin computes it.
        const auto composedDigest = [](const surfacematerialbinding::ImportedSceneMaterialRequest& material)
        {
            std::string admissionError;
            const auto admitted = surfacematerial::admit(material.program, admissionError);
            return admitted ? admitted->structuralDigest() : std::string {};
        };
        zeroOverride.binding.surfaceMaterialDigest = composedDigest(zeroOverride);
        oneOverride.binding.surfaceMaterialDigest = composedDigest(oneOverride);
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> owner = composed;
        std::string overrideError;
        {
            const bool conditionOk = composed->objectCount >= 2 && composed->objects[0].material != composed->objects[1].material
                && geometrysurfacematerial::appendImportedOverride(programs, zeroOverride, owner, 31, 35, overrideError)
                && geometrysurfacematerial::appendImportedOverride(programs, oneOverride, owner, 31, 35, overrideError)
                && programs.size() == 2;
            check(conditionOk, "two Render overrides with different leaf programs share one admitted collection: " + overrideError);
        }
        check(programs.size() == 2
                && !programs[0].program.material.binding.surfaceMaterialDigest.empty()
                && !programs[1].program.material.binding.surfaceMaterialDigest.empty()
                && programs[0].program.material.binding.surfaceMaterialDigest
                    == composedDigest(programs[0].program.material)
                && programs[1].program.material.binding.surfaceMaterialDigest
                    == composedDigest(programs[1].program.material),
            "each Render override's surfaceMaterialDigest equals the structural digest of its composed program");
    }

    typedscenepass::Payload pass;
    pass.extent = {640, 360};
    pass.scene = oneLeaf;
    const bool oneLeafPass = typedscenepass::valid(pass);
    pass.scene = twoLeaves;
    check(oneLeafPass && !typedscenepass::valid(pass) && typedscenepass::encode(pass).empty(),
        "Render Passes refuse a multi-leaf scene and keep the one-leaf scene");
    check(std::string(visualimportedscenerender::kMultipleImportedLeavesUnavailable).find(
            "multiple imported leaves are not rendered yet") != std::string::npos
            && std::string(visualimportedscenerender::kRenderPassesMultipleImportedLeavesUnavailable).find(
                "Render Passes with multiple imported leaves are not rendered yet") != std::string::npos,
        "the shared multi-leaf diagnostics name their refusals");

    auto baseline = composedScenePlan(); baseline.structuralRevision = 31;
    baseline.nodeIds = {6,11};
    baseline.nodeKinds = {videowire::geometry::kRetainedSceneOperation, "visual.3d.render"};
    baseline.operations = {{6,baseline.nodeKinds[0],"control-eval",""}, {11,baseline.nodeKinds[1],"native-gpu",""}};
    baseline.edges = {{6,0,11,0}};
    baseline.ports = {{6,0,1,"out","control","scene3D","unspecified","unspecified"},
        {11,0,1,"in","control","scene3D","unspecified","unspecified"},
        {11,1,1,"out","frame","image","rgba8","sRGB"}};
    auto candidate = baseline;
    candidate.operations.back().payloadXml = wire;
    auto oneLeafPlan = baseline;
    oneLeafPlan.operations.back().payloadXml = oneLeafWire;
    {
        // Resource admission counts the procedural snapshot's records, as the
        // plugin does, so a second leaf leaves the scene record count unchanged.
        videowire::VisualPlanResourceUsage oneLeafUsage, twoLeafUsage;
        std::string usageError;
        {
            const bool conditionOk = videowire::accountVisualPlanResources(oneLeafPlan, 640, 360, oneLeafUsage, usageError)
                && videowire::accountVisualPlanResources(candidate, 640, 360, twoLeafUsage, usageError)
                && oneLeafUsage.sceneRecords > 0 && twoLeafUsage.sceneRecords == oneLeafUsage.sceneRecords
                && twoLeafUsage.operations == oneLeafUsage.operations;
            check(conditionOk, "resource admission accepts a V26 composition and counts the procedural records only: " + usageError);
        }
    }
    {
        // The executor is the shared compile step for direct export, viewport
        // and Geometry Core routes that skip preparation.
        videowire::VisualLayerExecution execution;
        std::string executorError;
        {
            const bool conditionOk = videowire::compileVisualLayerExecution(oneLeafPlan, execution, executorError);
            check(conditionOk, "the executor still compiles a one-leaf composition: " + executorError);
        }
        executorError.clear();
        {
            const bool conditionOk = videowire::compileVisualLayerExecution(candidate, execution, executorError);
            check(conditionOk, "the executor compiles a V26 Render node: " + executorError);
        }
        auto passPlan = candidate;
        passPlan.operations.back().kind = std::string(typedscenepass::kNodeKind);
        executorError.clear();
        {
            const bool conditionOk = !videowire::compileVisualLayerExecution(passPlan, execution, executorError)
                && executorError == visualimportedscenerender::kRenderPassesMultipleImportedLeavesUnavailable;
            check(conditionOk, "the executor refuses a V26 payload on a Render Passes operation: " + executorError);
        }
        VisualImportedScenePlanCache directCache;
        executorError.clear();
        {
            const bool conditionOk = directCache.resolve(candidate, executorError) != nullptr;
            check(conditionOk, "the compiled plan cache resolves a V26 Render node: " + executorError);
        }
        executorError.clear();
        {
            const bool conditionOk = directCache.resolve(passPlan, executorError) == nullptr
                && executorError == visualimportedscenerender::kRenderPassesMultipleImportedLeavesUnavailable;
            check(conditionOk, "the compiled plan cache refuses V26 Render Passes through the executor: " + executorError);
        }
    }
    FakeExecution execution;
    FakeLayer layer;
    layer.texture = 913;
    std::vector<FakeReceipt> owners;
    VisualImportedScenePlanCache cache;
    std::string error;
    for (const auto use : {NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export})
    {
        error.clear();
        {
            const bool conditionOk = prepareVisualImportedSceneLayerAtTime({oneLeafPlan},oneLeafPlan.clipId,640,360,1.25,24,use,
                &execution,&cache,layer,owners,error,nullptr,{12,34}) == VisualImportedScenePreparation::rendered
                && execution.lastRequest.additionalImportedScenes.empty();
            check(conditionOk, "a one-leaf composition prepares with no further leaves: " + error);
        }
        error.clear();
        {
            const bool conditionOk = prepareVisualImportedSceneLayerAtTime({candidate},candidate.clipId,640,360,1.25,24,use,
                &execution,&cache,layer,owners,error,nullptr,{12,34}) == VisualImportedScenePreparation::rendered;
            check(conditionOk, "the helper prepares a V26 multi-leaf composition: " + error);
        }
        const auto& prepared = execution.lastRequest;
        check(prepared.importedScene && prepared.importedScene->sourceStableId == 71
                && prepared.deformation && prepared.deformation->playback.speed == 1.5
                && prepared.additionalImportedScenes.size() == 1
                && prepared.additionalImportedScenes[0].scene
                && prepared.additionalImportedScenes[0].scene->sourceStableId == 81
                && prepared.additionalImportedScenes[0].deformation
                && prepared.additionalImportedScenes[0].deformation->playback.speed == 0.5
                && prepared.additionalImportedScenes[0].deformation->pose.nodeStableId == 2053
                && prepared.additionalImportedScenes[0].deformation->playback.timelineSeconds
                    == prepared.deformation->playback.timelineSeconds,
            "each prepared leaf keeps its own playback controls, its leaf selector and the shared clock");
    }
    auto passPlan = candidate;
    passPlan.operations.back().kind = std::string(typedscenepass::kNodeKind);
    const auto calls = execution.previewCalls + execution.exportCalls;
    const auto texture = layer.texture;
    error.clear();
    {
        const bool conditionOk = prepareVisualImportedSceneLayerAtTime({passPlan},passPlan.clipId,640,360,1.25,24,
            NativeImportedSceneRenderUse::Preview,&execution,&cache,layer,owners,error,nullptr,{12,34})
                == VisualImportedScenePreparation::rejected
            && error == visualimportedscenerender::kRenderPassesMultipleImportedLeavesUnavailable
            && execution.previewCalls + execution.exportCalls == calls && layer.texture == texture;
        check(conditionOk, "V26 Render Passes refuse before any layer reset, publication or render: " + error);
    }

    {
        // Execution admits one selected object across all leaves. Entry
        // refuses a second selecting leaf with execution's message before any
        // backend work, instead of passing preparation and failing per frame.
        FamilyProductionBackend sceneBackend;
        Store store;
        ImportedScenePayloadExecution payloadExecution(store, sceneBackend);
        ImportedSceneRequest selecting;
        selecting.importedScene = std::make_shared<const visualimportedscenerender::Request>(first);
        selecting.deformation = first.deformation;
        ImportedSceneLeafRequest secondLeaf;
        secondLeaf.scene = std::make_shared<const visualimportedscenerender::Request>(second);
        secondLeaf.deformation = second.deformation;
        selecting.additionalImportedScenes.push_back(secondLeaf);
        ImportedSceneExecutionReceipt receipt;
        error.clear();
        {
            const bool conditionOk = first.deformation->pose.nodeStableId == 5 && second.deformation->pose.nodeStableId == 2053
                && !payloadExecution.executePreview(selecting, receipt, error)
                && error == "Scene Compose refused: more than one imported leaf selects an object"
                && sceneBackend.prepareCalls == 0;
            check(conditionOk, "two selecting leaves refuse at entry with execution's message: " + error);
        }
    }
}

void testUnsupportedObjectLocalRefusal()
{
    using namespace videohelper::importedscene;
    using namespace videohelper::modelpayload;
    const std::string diagnostic = "Imported object local transforms require a native evaluation owner";
    std::string error;
    for (const bool composition : { false, true })
    {
        auto baseline = importedScenePlan();
        auto leaf = renderRequest();
        leaf.objectNodeStableId = 5; leaf.objectMeshStableId = 1;
        auto operation = leaf;
        if (composition)
        {
            baseline = composedScenePlan(); baseline.structuralRevision = 31;
            operation = {};
            operation.sourceStableId = 7; operation.renderStableId = 12;
            operation.structuralRevision = 31; operation.evaluationRevision = 35;
            operation.sceneSnapshot = composedScene();
            leaf.renderStableId = 12; leaf.structuralRevision = 31; leaf.evaluationRevision = 35;
            operation.importedScene = std::make_shared<const visualimportedscenerender::Request>(leaf);
            baseline.nodeIds = {6,11};
            baseline.nodeKinds = {videowire::geometry::kRetainedSceneOperation, "visual.3d.render"};
            baseline.operations = {{6,baseline.nodeKinds[0],"control-eval",""},
                {11,baseline.nodeKinds[1],"native-gpu",""}};
            baseline.edges = {{6,0,11,0}};
            baseline.ports = {{6,0,1,"out","control","scene3D","unspecified","unspecified"},
                {11,0,1,"in","control","scene3D","unspecified","unspecified"},
                {11,1,1,"out","frame","image","rgba8","sRGB"}};
        }
        baseline.operations.back().payloadXml = visualimportedscenerender::encode(operation);
        check(!baseline.operations.back().payloadXml.empty(), "neutral V24 refusal baseline encodes");
        auto candidate = baseline;
        leaf.objectLocalTransform.version = 1; leaf.objectLocalTransform.nodeStableId = 7;
        leaf.objectLocalTransform.translation[0] = 1;
        if (composition) operation.importedScene = std::make_shared<const visualimportedscenerender::Request>(leaf);
        else operation = leaf;
        candidate.operations.back().payloadXml = visualimportedscenerender::encode(operation);
        check(!candidate.operations.back().payloadXml.empty(), "active V25 standalone/composition candidate encodes");
        FakeExecution execution;
        FakeLayer layer;
        std::vector<FakeReceipt> owners;
        VisualImportedScenePlanCache cache;
        for (const auto use : {NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export})
        {
            {
                const bool conditionOk = prepareVisualImportedSceneLayerAtTime({baseline},baseline.clipId,640,360,1.25,24,use,
                &execution,&cache,layer,owners,error,nullptr,{12,34}) == VisualImportedScenePreparation::rendered;
                check(conditionOk, "neutral V24 standalone/composition preserves the released preparation route: " + error);
            }
            const auto* compiled = cache.resolve(baseline,error);
            check(compiled != nullptr && !owners.empty(), "neutral V24 retains the accepted compiled plan and receipt");
            if (!compiled || owners.empty()) continue;
            layer.shaderClock = {2.25,54}; layer.audioPresent = true; layer.audioFeatures.bands = {0.25f,0.5f};
            layer.rawExportFrame = execution.nextFrame; layer.rawExportMask = 3; layer.rawExportClipId = baseline.clipId;
            layer.rawExportRenderId = 12; layer.rawExportImageOutput = renderpassoutput::Output::Depth;
            const auto incumbent = layer;
            const auto count = owners.size(); const auto priorReceipt = owners.back();
            const auto priorRequest = execution.lastRequest;
            const auto calls = execution.previewCalls + execution.exportCalls;
            const auto uploads = execution.staticGpuUploads; const auto retired = execution.identityRetirements;
            const auto identity = execution.publishedIdentity; const auto scene = execution.admittedScene;
            check(prepareVisualImportedSceneLayerAtTime({candidate},baseline.clipId,640,360,2.5,24,use,
                &execution,&cache,layer,owners,error,nullptr,{12,34}) == VisualImportedScenePreparation::rejected
                && error == diagnostic, "active V25 refuses explicitly before preparation publication");
            check(execution.previewCalls + execution.exportCalls == calls && execution.staticGpuUploads == uploads
                && execution.identityRetirements == retired && execution.publishedIdentity == identity
                && execution.admittedScene == scene && execution.lastRequest.sceneSnapshot == priorRequest.sceneSnapshot
                && execution.lastRequest.importedScene == priorRequest.importedScene
                && visualimportedobjectlocal::same(execution.lastRequest.objectLocalTransform, priorRequest.objectLocalTransform)
                && owners.size() == count && owners.back().frame == priorReceipt.frame && owners.back().owner == priorReceipt.owner
                && layer.texture == incumbent.texture && layer.nativeTextureView == incumbent.nativeTextureView
                && layer.nativeTextureBackend == incumbent.nativeTextureBackend
                && layer.nativeTextureDescriptor.imageHandle == incumbent.nativeTextureDescriptor.imageHandle
                && layer.nativeTextureDescriptor.deviceOrContextIdentity == incumbent.nativeTextureDescriptor.deviceOrContextIdentity
                && layer.depthTexture == incumbent.depthTexture && layer.nativeDepthTextureView == incumbent.nativeDepthTextureView
                && layer.texWidth == incumbent.texWidth && layer.texHeight == incumbent.texHeight
                && layer.depthWidth == incumbent.depthWidth && layer.depthHeight == incumbent.depthHeight
                && layer.shaderClock.time == incumbent.shaderClock.time && layer.shaderClock.frame == incumbent.shaderClock.frame
                && layer.audioPresent == incumbent.audioPresent && layer.audioFeatures.bands == incumbent.audioFeatures.bands
                && layer.canonicalBlockCFrame == incumbent.canonicalBlockCFrame
                && layer.rawExportFrame == incumbent.rawExportFrame && layer.rawExportMask == incumbent.rawExportMask
                && layer.rawExportClipId == incumbent.rawExportClipId && layer.rawExportRenderId == incumbent.rawExportRenderId
                && layer.rawExportImageOutput == incumbent.rawExportImageOutput,
                "active V25 preserves caller layer, executor, receipt and prior source publication");
            check(cache.resolve(baseline,error) == compiled, "active V25 leaves the accepted cache entry intact");
        }
    }

    class CaptureDeformationBackend final : public arbitgpu::NativeDeformationBackend
    {
    public:
        int prepareCalls = 0, renderCalls = 0;
        arbitgpu::BackendInfo info() const override { return {}; }
        arbitgpu::NativeDeformationPreparation prepare(const std::shared_ptr<const arbitgpu::NativeDeformationScene>&,
            const std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram>&) override
        { ++prepareCalls; return {}; }
        arbitgpu::NativeDeformationSubmission render(const std::shared_ptr<const arbitgpu::NativeDeformationScene>&,
            const std::shared_ptr<const visualdeformation::AnimationDeformationSnapshot>&,
            const std::shared_ptr<const arbitgpu::NativeDeformationResources>&,
            std::uint32_t,std::uint32_t,arbitgpu::NativeDeformationRuntimeInputs) override
        { ++renderCalls; return {}; }
    } deformationBackend;
    FamilyProductionBackend sceneBackend;
    Store store;
    ImportedScenePayloadExecution staticExecution(store,sceneBackend,deformationBackend);
    ImportedAnimatedScenePayloadExecution animatedExecution(store,deformationBackend,sceneBackend);
    ImportedSceneRequest staticRequest;
    staticRequest.objectLocalTransform.version = 1; staticRequest.objectLocalTransform.nodeStableId = 7;
    ImportedAnimatedSceneRequest animatedRequest;
    animatedRequest.objectLocalTransform = staticRequest.objectLocalTransform;
    // A procedural scene stops the animated entry at its named refusal, before any leaf is admitted.
    animatedRequest.proceduralScene = std::make_shared<HarmonicMIDI::grid::Visual3DScene>();
    const std::string animatedDiagnostic = "Imported object local transforms with procedural composition are not rendered yet";
    const auto oldScene = composedScene(); const auto oldFrame = std::make_shared<FakeFrame>();
    ImportedSceneExecutionReceipt staticReceipt;
    staticReceipt.use = ImportedSceneUse::Export; staticReceipt.admission.scene = oldScene;
    staticReceipt.frame.rendered.nativeFrame = oldFrame; staticReceipt.frame.nativeBackend = "retained-backend";
    staticReceipt.materialDiagnostic = "retained-static-diagnostic"; staticReceipt.usedLastGoodMaterial = true;
    ImportedAnimatedSceneReceipt animatedReceipt;
    animatedReceipt.owner = videohelper::ImportedAnimationEvaluationOwner::Export; animatedReceipt.frame.nativeFrame = oldFrame;
    animatedReceipt.materialDiagnostic = "retained-animated-diagnostic"; animatedReceipt.usedLastGoodMaterial = true;
    // Sentinel receipts test pre-mutation refusal only, not native rendering.
    for (const bool preview : {true,false})
    {
        check(!(preview ? staticExecution.executePreview(staticRequest,staticReceipt,error)
                        : staticExecution.executeExport(staticRequest,staticReceipt,error)) && error == diagnostic,
            "direct static active-local entry refuses before receipt staging");
        check(staticReceipt.use == ImportedSceneUse::Export && staticReceipt.admission.scene == oldScene
            && staticReceipt.frame.rendered.nativeFrame == oldFrame && staticReceipt.frame.nativeBackend == "retained-backend"
            && staticReceipt.materialDiagnostic == "retained-static-diagnostic" && staticReceipt.usedLastGoodMaterial,
            "direct static active-local refusal preserves the caller receipt");
        check(!(preview ? animatedExecution.executePreview(animatedRequest,animatedReceipt,error)
                        : animatedExecution.executeExport(animatedRequest,animatedReceipt,error)) && error == animatedDiagnostic,
            "direct animated active-local entry refuses before receipt reset");
        check(animatedReceipt.owner == videohelper::ImportedAnimationEvaluationOwner::Export && animatedReceipt.frame.nativeFrame == oldFrame
            && animatedReceipt.materialDiagnostic == "retained-animated-diagnostic" && animatedReceipt.usedLastGoodMaterial,
            "direct animated active-local refusal preserves the caller receipt");
    }
    staticRequest.objectLocalTransform = {};
    auto nested = renderRequest(); nested.objectLocalTransform = animatedRequest.objectLocalTransform;
    staticRequest.importedScene = std::make_shared<const visualimportedscenerender::Request>(nested);
    check(!staticExecution.executePreview(staticRequest,staticReceipt,error) && error == diagnostic
        && staticReceipt.admission.scene == oldScene && staticReceipt.frame.rendered.nativeFrame == oldFrame,
        "direct imported composition refuses an active leaf even if caller omitted its fixed-value copy");
    check(sceneBackend.prepareCalls == 0 && deformationBackend.prepareCalls == 0 && deformationBackend.renderCalls == 0,
        "active local refusal makes no scene/deformation backend preparation or submission");
}

// A family collider's composed transform carries non-uniform scale and a
// parent chain into the frame its bodies attach to. The replay point must land
// where the scene places that vertex.
void testFamilyColliderWorldTransform()
{
    using namespace videorender;
    using namespace HarmonicMIDI::grid;
    struct Point { float x, y, z; };
    const auto nearPoint=[](const std::array<float,3>& a,float x,float y,float z) {
        return std::abs(a[0]-x)<1e-4f && std::abs(a[1]-y)<1e-4f && std::abs(a[2]-z)<1e-4f;
    };
    auto scene=std::make_shared<Visual3DScene>();
    // Parent 100 turns 90 degrees about Z, stretches X by 2, halves Z, and sits at y=0.2.
    SceneObjectRecord parentRecord; parentRecord.id={100};
    parentRecord.transform.translation={0,0.2f,0};
    parentRecord.transform.rotation={0,0,0.70710678f,0.70710678f};
    parentRecord.transform.scale={2,1,0.5f};
    scene->objects[scene->objectCount++]=parentRecord;
    // Child 11 is under the parent with identity local transform.
    SceneObjectRecord parented; parented.id={11}; parented.parent={100};
    parented.transform.rotation={0,0,0,1};
    scene->objects[scene->objectCount++]=parented;
    // Child 10 is a scene root with non-uniform scale and an offset.
    SceneObjectRecord root; root.id={10};
    root.transform.translation={-1,0,0};
    root.transform.rotation={0,0,0,1};
    root.transform.scale={1,3,1};
    scene->objects[scene->objectCount++]=root;
    ParticleGeometryBinding binding;
    binding.centerX=binding.centerY=binding.centerZ=.5f; binding.scale=.25f;
    const auto replay=[&](const std::array<float,3>& translation,const std::array<float,9>& linear,Point point) {
        ParticleColliderFamilyChild child; child.translation=translation; child.linear=linear;
        return particleFamilyColliderPoint(binding,child,point);
    };
    // Scene space: the parented child maps (1,0,0) to (0,2.2,0), and the root
    // child maps (1,1,1) to (0,3,1). Replay coordinates are center + scale * world.
    std::array<float,3> translation {}; std::array<float,9> linear {};
    check(particleFamilyWorldTransform(*scene,scene->objects[1],SceneObjectId{},translation,linear)
        && nearPoint(replay(translation,linear,Point{1,0,0}),.5f,.5f+.25f*2.2f,.5f),
        "a child under a stretched, turned parent lands at its world position in the scene-root frame");
    check(particleFamilyWorldTransform(*scene,scene->objects[2],SceneObjectId{},translation,linear)
        && nearPoint(replay(translation,linear,Point{1,1,1}),.5f,.5f+.25f*3,.5f+.25f),
        "a root child with non-uniform scale lands at its world position in the scene-root frame");
    // Under its shared parent, the same child keeps its local transform.
    check(particleFamilyWorldTransform(*scene,scene->objects[1],SceneObjectId{100},translation,linear)
        && nearPoint(translation,0,0,0) && linear==std::array<float,9>{1,0,0,0,1,0,0,0,1},
        "a child of the shared parent keeps its transform relative to that parent");
    check(!particleFamilyFrameUniform(*scene,SceneObjectId{100}) && particleFamilyFrameUniform(*scene,SceneObjectId{}),
        "a shared parent with non-uniform scale is not a uniform frame, and the scene root is");
    // Zero, negative and non-finite scale refuse, on the child and on a parent it multiplies through.
    SceneObjectRecord zero=root; zero.transform.scale.x=0;
    SceneObjectRecord negative=root; negative.transform.scale.y=-1;
    SceneObjectRecord nonFinite=root; nonFinite.transform.scale.z=std::numeric_limits<float>::infinity();
    check(!particleFamilyWorldTransform(*scene,zero,SceneObjectId{},translation,linear)
        && !particleFamilyWorldTransform(*scene,negative,SceneObjectId{},translation,linear)
        && !particleFamilyWorldTransform(*scene,nonFinite,SceneObjectId{},translation,linear),
        "zero, negative and non-finite child scale refuse");
    scene->objects[0].transform.scale.x=0;
    check(!particleFamilyWorldTransform(*scene,scene->objects[1],SceneObjectId{},translation,linear),
        "zero scale on a parent that a child multiplies through refuses");
    scene->objects[0].transform.scale.x=2;
    // A chain that never reaches the frame refuses.
    SceneObjectRecord orphan=parented; orphan.parent={999};
    check(!particleFamilyWorldTransform(*scene,orphan,SceneObjectId{},translation,linear),
        "a parent chain missing its object refuses");
}

// The live viewport has no timeline bump yet at generation zero. Its evaluation
// context and its Timeline Scene3D owner must share one nonzero project
// generation, so the owner is admitted instead of refused as zero.
void viewportSceneOwnerBeforeTimelineBump()
{
    using namespace videohelper::geometry;
    using namespace videowire::geometry;
    check(videohelper::viewportgeneration::projectGeneration(0)==1
        && videohelper::viewportgeneration::projectGeneration(1)==1
        && videohelper::viewportgeneration::projectGeneration(7)==7,
        "viewport project generation maps an unbumped timeline to one and keeps bumped values");
    class Capabilities final : public BackendCapabilitySource {
    public:
        BackendCapabilities geometryCoreCapabilities() const override {
            BackendCapabilities c; c.backendIdentity="viewport-owner-fixture"; c.deviceIdentity="owned-fixture";
            c.nativeGpuAvailable=c.geometryCoreExecution=c.immutableSourceBuffers=c.stableElementIds=c.typedFieldEvaluation=true;
            c.supportedCarriers=carrierBit(CarrierKind::geometry3D); return c;
        }
    } capabilities;
    GeometryCorePlanRuntime runtime(capabilities);
    std::string error;
    auto request=composedTimelineGeometryFixture(error);
    check(request.has_value(),"viewport owner fixture lowers the timeline scene: "+error);
    if (!request) return;
    const auto projectGeneration=videohelper::viewportgeneration::projectGeneration(0);
    // Helper, device, clip, plan revision and use follow the viewport's owner fields.
    const PlanOwnerIdentity owner{projectGeneration,41,51,8,9,PlanUse::preview};
    videowire::VisualPlanEvaluationContext context;
    context.projectGeneration=projectGeneration;
    check(owner.valid() && owner.projectGeneration==1 && context.projectGeneration==owner.projectGeneration,
        "viewport owner before a timeline bump is valid and matches the evaluation context generation");
    const auto admitted=evaluateComposedTimelineGeometry(runtime,owner,*request->timelineGeometry,
        request->sceneSnapshot,.5,error);
    check(admitted && admitted->vertexCount==request->sceneSnapshot->vertexCount,
        "Timeline Scene3D owner before a timeline bump is admitted: "+error);
    // The raw zero generation is still refused, so the coerced value is what admits the frame.
    std::string refusal;
    const PlanOwnerIdentity zeroOwner{0,41,51,8,9,PlanUse::preview};
    {
        const bool conditionOk = !evaluateComposedTimelineGeometry(runtime,zeroOwner,*request->timelineGeometry,
              request->sceneSnapshot,.5,refusal)
            && refusal == "Timeline Scene3D owner is invalid because it has zero projectGeneration";
        check(conditionOk, "a zero project generation owner is still refused by name: "+refusal);
    }
}

void testParentLightRecordIdentity()
{
    // The imported parent route renders modulated directional and point lights.
    // Its reusable frame must compare the exact per-frame records, so a changed
    // value cannot reuse a frame rendered at the previous value.
    using namespace HarmonicMIDI::grid;
    using videohelper::modelpayload::sameParentLightRecords;
    SceneLightRecord directional;
    directional.id.value = 78;
    directional.kind = SceneLightKind::Directional;
    directional.transform.rotation = { 0.0f, 0.0f, 0.0f, 1.0f };
    directional.intensity = 1.0f;
    const std::vector<SceneLightRecord> stored { directional };
    check(sameParentLightRecords(stored, stored) && sameParentLightRecords({}, {}),
        "identical per-frame light records, or none on both sides, reuse a parent frame");
    auto modulated = stored;
    modulated[0].intensity = 0.5f;
    check(!sameParentLightRecords(stored, modulated),
        "a modulated light value does not reuse a parent frame rendered at another value");
    check(!sameParentLightRecords(stored, {}) && !sameParentLightRecords({}, stored),
        "a light set present on only one side does not reuse a parent frame");
}

int main(int argc, char** argv)
{
    if (argc == 2 && std::string(argv[1]) == "--imported-material-frame-resolver")
    { testProductionMaterialFrameResolver(); testProductionDiffractionFrameCompatibility(); return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE; }
    testProductionMaterialFrameResolver();
    testProductionDiffractionFrameCompatibility();
    timelinefamilyfixture::verify(check);
    testTimelineFamilyTransport();
    testTimelineFamilyProductionDispatch();
    testTimelineFamilyWaterProductionDispatch();
    testTimelineFamilySolidComposition();
    testFamilyColliderWorldTransform();
    waterFrameConsumers();
    viewportSceneOwnerBeforeTimelineBump();
    testImportedSolidComposition();
    testImportedSceneCompositionHandoff();
    testImportedSurfaceOverrideBesideProceduralPrograms();
    testImportedSurfaceOverrideReachesCompositionExecutor();
    testImportedSceneCompositionGeneratedIdentities();
    testUnsupportedObjectLocalRefusal();
    testMultiLeafComposition();
    testImportedParentIntentTransport();
    testParentLightRecordIdentity();
    rawexportchecks::verify(check);
    testGlassCausticRequestHandoff();
    check(videowire::isAdmittedVertexModifierKind("visual.vertex.imported-position")
            && !videowire::isAdmittedVertexModifierKind("visual.vertex.future-position"),
        "snapshot kind admission covers the exact native vertex IR without accepting unknown extensions");
    videowire::SurfaceGraphOperationContract surfaceInputContract;
    check(videowire::isAdmittedSurfaceProgramKind("visual.surface.input.tex-coord-0")
            && videowire::surfaceGraphOperationContract(
                "visual.surface.input.tex-coord-0", surfaceInputContract)
            && surfaceInputContract.kind == surfacematerial::OperationKind::Input
            && surfaceInputContract.semantic == surfacematerial::InputSemantic::TexCoord0
            && !videowire::isAdmittedSurfaceProgramKind("visual.surface.input.future-coordinate"),
        "producer surface input slugs map to the immutable native material IR and unknown inputs reject");
    testMaterialFramePrerequisite();
    testReactiveMaterialFramePrerequisite();
    testMaterialFrameDagAndTime();
    testMaterialFramePrerequisite(true);
    testMaterialFramePrerequisite(false, true);
    testDiffractionRuntimeParameters();
    testBakedPbrModulation();
    testBakedSurfaceConstantModulation();
    testLightOverrideApplication();
    testLightOverrideIdentityAndAcceptance();
    testLightOverrideBackendRead();
    testDirectionalLightModulation();
    testBakedDisplacementModulation();
    testMaterialFieldBinding();
    testAuthoredImportedStarterTopologies();
    using videohelper::importedscene::NativeImportedSceneRenderUse;
    using videohelper::importedscene::VisualImportedScenePreparation;
    using videohelper::importedscene::prepareVisualImportedSceneLayer;
    using videohelper::importedscene::prepareVisualImportedSceneLayerAtTime;

    const auto legacyRequest = renderRequest();
    {
        auto producerPlan = importedScenePlan();
        producerPlan.ports.insert(producerPlan.ports.end(), {
            { 71, 2, 1, "in", "control", "material", "unspecified", "unspecified" },
            { 71, 3, 1, "in", "control", "mesh", "unspecified", "unspecified" },
            { 71, 4, 1, "out", "frame", "depth", "r32f", "unspecified" },
            { 71, 5, 1, "in", "control", "camera", "unspecified", "unspecified" },
            { 71, 6, 1, "in", "control", "light", "unspecified", "unspecified" },
            { 71, 7, 1, "out", "frame", "normal", "rgba16f", "unspecified" },
            { 71, 8, 1, "out", "frame", "emission", "rgba16f", "linearSRGB" },
            { 71, 9, 1, "out", "frame", "mask", "r8", "unspecified" },
            { 71, 10, 1, "out", "frame", "materialId", "r32uint", "unspecified" },
            { 71, 11, 1, "out", "frame", "objectId", "r32uint", "unspecified" },
            { 71, 12, 2, "out", "frame", "motionVectors", "rg16f", "unspecified" }
        });
        videowire::VisualLayerExecution compiled;
        std::string error;
        {
            const bool conditionOk = videowire::compileVisualLayerExecution(producerPlan, compiled, error);
            check(conditionOk, "producer-compatible 13-port Render 3D contract is admitted: " + error);
        }

        auto malformed = producerPlan;
        auto outputPlan = producerPlan;
        outputPlan.operations.push_back({ 72, "video.out", "native-gpu", "<NodeParams/>", "" });
        outputPlan.nodeIds.push_back(72);
        outputPlan.nodeKinds.push_back("video.out");
        outputPlan.ports.push_back({72,0,1,"in","frame","image","rgba8","sRGB"});
        outputPlan.edges.push_back({71,1,72,0});
        {
            const bool conditionOk = videowire::compileVisualLayerExecution(outputPlan, compiled, error);
            check(conditionOk, "producer Render 3D may retain its exact outer VideoOut sink: " + error);
        }
        malformed = outputPlan;
        malformed.edges.back().fromPort = 4;
        check(!videowire::compileVisualLayerExecution(malformed, compiled, error),
            "outer VideoOut rejects a non-image Render 3D source port");
        malformed = outputPlan;
        malformed.edges.push_back({71,1,72,0});
        check(!videowire::compileVisualLayerExecution(malformed, compiled, error),
            "outer VideoOut rejects duplicate fan-in");
        malformed = outputPlan;
        malformed.operations.back().payloadXml = "<NodeParams future=\"1\"/>";
        check(!videowire::compileVisualLayerExecution(malformed, compiled, error),
            "outer VideoOut rejects unknown parameter payloads");

        malformed = producerPlan;
        malformed.ports.back().channels = 1;
        check(!videowire::compileVisualLayerExecution(malformed, compiled, error),
            "producer Render 3D motion output rejects changed channel count");
        malformed = producerPlan;
        malformed.ports.back().pixelFormat = "rgba16f";
        check(!videowire::compileVisualLayerExecution(malformed, compiled, error),
            "producer Render 3D motion output rejects changed data format");
        malformed = producerPlan;
        malformed.ports.back().port = 13;
        check(!videowire::compileVisualLayerExecution(malformed, compiled, error),
            "producer Render 3D motion output rejects changed port identity");

        auto legacyRawPassPlan = producerPlan;
        legacyRawPassPlan.ports.pop_back();
        {
            const bool conditionOk = videowire::compileVisualLayerExecution(legacyRawPassPlan, compiled, error);
            check(conditionOk, "producer-compatible legacy 12-port Render 3D raw-pass contract is admitted: " + error);
        }
        malformed = legacyRawPassPlan;
        malformed.ports.back().colorSpace = "linearSRGB";
        check(!videowire::compileVisualLayerExecution(malformed, compiled, error),
            "legacy Render 3D raw object-ID output rejects changed color space");
    }
    {
        arbitgpu::NativeSceneMotionHistory history;
        arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
        check(!history.predecessor(0, 60, 1, 640, 360), "first motion sample is reset");
        history.commit(0, 60, 1, 640, 360, inputs);
        inputs.objectTranslationOffset[0] = 0.25f;
        check(history.predecessor(1, 60, 1, 640, 360).has_value(), "adjacent motion sample retains the predecessor");
        history.commit(1, 60, 1, 640, 360, inputs);
        check(history.predecessor(1, 60, 1, 640, 360)->objectTranslationOffset[0] == 0,
            "same-frame export keeps the original predecessor");
        check(!history.predecessor(10, 60, 1, 640, 360)
            && !history.predecessor(0, 60, 1, 640, 360)
            && !history.predecessor(2, 30, 1, 640, 360)
            && !history.predecessor(2, 60, 1, 1280, 720),
            "seek, loop, rate and extent changes reset motion");
        history.commit(10, 60, 1, 640, 360, inputs);
        check(!history.predecessor(10, 60, 1, 640, 360), "repeated seek frame remains reset");
    }
    {
        // Deformed Motion keeps the vertex program inputs and the timeline
        // Geometry3D positions of the prior frame with its sample.
        arbitgpu::NativeSceneMotionHistory history;
        arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
        inputs.timeSeconds = 0.5f;
        inputs.vertexSpectrum[6] = 0.25f;
        const auto positions = std::make_shared<const std::vector<HarmonicMIDI::grid::SceneVec3>>(
            std::vector<HarmonicMIDI::grid::SceneVec3> {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}});
        check(history.commit(0, 60, 1, 640, 360, inputs, {}, {}, positions),
            "a rendered frame commits its deformed positions");
        inputs.timeSeconds = 0.5f + 1.0f / 60.0f;
        inputs.vertexSpectrum[6] = 0.75f;
        const auto prior = history.predecessor(1, 60, 1, 640, 360, {}, inputs);
        check(prior && prior->scenePositions == positions && prior->timeSeconds == 0.5f
                && prior->vertexSpectrum[6] == 0.25f,
            "the adjacent frame receives the prior positions, time and audio");
        auto invalid = std::make_shared<std::vector<HarmonicMIDI::grid::SceneVec3>>(1);
        (*invalid)[0].x = std::numeric_limits<float>::quiet_NaN();
        check(!history.commit(1, 60, 1, 640, 360, inputs, {}, {}, invalid)
                && history.predecessor(1, 60, 1, 640, 360, {}, inputs)->scenePositions == positions,
            "non-finite prior positions are refused and leave the history unchanged");
        auto previousInputs = inputs;
        previousInputs.previousMotion = prior;
        check(arbitgpu::validFixtureRuntimeInputs(previousInputs),
            "retained prior positions and audio pass native input validation");
        previousInputs.previousMotion->vertexSpectrum[6] = 2.0f;
        check(!arbitgpu::validFixtureRuntimeInputs(previousInputs),
            "out-of-range prior audio is refused before a native draw");
    }
    {
        // A Motion frame with non-finite timeline positions cannot reach the
        // deformed-position commit: Scene3D validation refuses the composed
        // request first. Preview and Export both refuse it before native work.
        auto invalidScene = std::make_shared<HarmonicMIDI::grid::Visual3DScene>(*composedScene());
        invalidScene->vertices[0].position.x = std::numeric_limits<float>::quiet_NaN();
        videohelper::modelpayload::Store store;
        FamilyProductionBackend backend;
        videohelper::modelpayload::ImportedScenePayloadExecution execution(store, backend);
        videohelper::modelpayload::ImportedSceneRequest request;
        request.sceneSnapshot = invalidScene;
        request.deformedSceneMotion = true;
        request.structuralRevision = 31;
        request.evaluationRevision = 35;
        request.clipId = 8;
        request.dimensions.width = 640;
        request.dimensions.height = 360;
        request.frame.frame = 0;
        request.frame.rateNumerator = 60;
        request.frame.rateDenominator = 1;
        request.runtimeInputs.imageOutput = renderpassoutput::Output::Motion;
        for (const bool preview : {true, false})
        {
            videohelper::modelpayload::ImportedSceneExecutionReceipt receipt;
            std::string error;
            {
                const bool conditionOk = !(preview ? execution.executePreview(request, receipt, error)
                            : execution.executeExport(request, receipt, error))
                && error.find("composed Scene3D request is invalid") != std::string::npos
                && !receipt.frame.rendered.nativeFrame && backend.prepareCalls == 0;
                check(conditionOk, "a Motion frame with non-finite deformed positions is refused before native output: " + error);
            }
        }
    }
    {
        auto plan = importedScenePlan();
        plan.ports.push_back({ 71, 2, 1, "in", "control", "material", "unspecified", "unspecified" });
        plan.ports.push_back({ 71, 3, 1, "in", "control", "mesh", "unspecified", "unspecified" });
        plan.ports.push_back({ 71, 4, 1, "out", "frame", "depth", "r32f", "unspecified" });
        plan.ports.push_back({ 71, 5, 1, "in", "control", "camera", "unspecified", "unspecified" });
        plan.ports.push_back({ 71, 6, 1, "in", "control", "light", "unspecified", "unspecified" });
        const std::array<const char*, 7> types { "image", "depth", "normal", "emission", "mask", "materialId", "objectId" };
        const std::array<const char*, 7> formats { "rgba8", "r32f", "rgba16f", "rgba16f", "r8", "r32uint", "r32uint" };
        for (std::size_t i = 0; i < 7; ++i)
        {
            const auto space = i == 0 ? "sRGB" : i == 3 ? "linearSRGB" : "unspecified";
            const auto renderPort = renderpasscomposite::renderPort(renderpasscomposite::kInputs[i]);
            if (i >= 2) plan.ports.push_back({ 71, renderPort, 1, "out", "frame", types[i], formats[i], space });
            plan.ports.push_back({ 90, static_cast<int>(i), 1, "in", "frame", types[i], formats[i], space });
            plan.edges.push_back({ 71, renderPort, 90, static_cast<int>(i) });
        }
        plan.ports.push_back({ 90, 7, 1, "out", "frame", "image", "rgba8", "sRGB" });
        plan.nodeIds.push_back(90); plan.nodeKinds.push_back(renderpasscomposite::kNodeKind);
        renderpasscomposite::Parameters parameters;
        parameters.fogColor = { 0.2f, 0.4f, 0.6f };
        plan.operations.push_back({ 90, renderpasscomposite::kNodeKind, "native-gpu", renderpasscomposite::encode(parameters) });
        videowire::VisualLayerExecution compiled;
        std::string error;
        {
            const bool conditionOk = videowire::compileVisualLayerExecution(plan, compiled, error) && compiled.passComposite.has_value();
            check(conditionOk, "raw Render 3D fan-out compiles with one immutable composite consumer: " + error);
        }
        FakeExecution native;
        native.nextFrame->rawPasses = true;
        FakeLayer layer;
        std::vector<FakeReceipt> owners;
        for (const auto use : { NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export })
            {
                const bool conditionOk = prepareVisualImportedSceneLayer({ plan }, 7, 640, 360, use, &native, layer, owners, error)
                == VisualImportedScenePreparation::rendered && native.lastRequest.runtimeInputs.passComposite
                && renderpasscomposite::encode(*native.lastRequest.runtimeInputs.passComposite) == renderpasscomposite::encode(parameters);
                check(conditionOk, "preview and export forward identical raw-pass composite parameters: " + error);
            }
        native.nextFrame->rawPasses = false;
        check(prepareVisualImportedSceneLayer({ plan }, 7, 640, 360, NativeImportedSceneRenderUse::Preview,
            &native, layer, owners, error) == VisualImportedScenePreparation::rejected,
            "a flattened color frame cannot satisfy raw AOV publication");
        auto corrupted = plan;
        for (auto& port : corrupted.ports)
            if (port.nodeId == 71 && port.port == 10) port.pixelFormat = "rgba8";
        check(!videowire::compileVisualLayerExecution(corrupted, compiled, error),
            "integer ID attachments reject palette-image formats");
        corrupted = plan;
        corrupted.edges.back().fromNodeId = 70;
        check(!videowire::compileVisualLayerExecution(corrupted, compiled, error),
            "composite inputs must share the exact scene render identity");
        auto motionPlan = plan;
        motionPlan.ports.push_back({71, 12, 2, "out", "frame", "motionVectors", "rg16f", "unspecified"});
        motionPlan.ports.push_back({90, 8, 2, "in", "frame", "motionVectors", "rg16f", "unspecified"});
        motionPlan.edges.push_back({71, 12, 90, 8});
        parameters.mode = renderpasscomposite::Mode::MotionView;
        motionPlan.operations.back().payloadXml = renderpasscomposite::encode(parameters);
        {
            const bool conditionOk = videowire::compileVisualLayerExecution(motionPlan, compiled, error)
            && compiled.passComposite->mode == renderpasscomposite::Mode::MotionView;
            check(conditionOk, "motion inspection admits exact RG16F fan-out from the same render: " + error);
        }
        native.nextFrame->rawPasses = true;
        videohelper::importedscene::VisualImportedScenePlanCache motionCache;
        for (const auto use : { NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export })
            {
                const bool conditionOk = prepareVisualImportedSceneLayerAtTime({motionPlan}, 7, 640, 360, 0.5, 60, use,
                &native, &motionCache, layer, owners, error) == VisualImportedScenePreparation::rendered
                && native.lastRequest.frame.frame == 30;
                check(conditionOk, "static motion preview/export receive the same rational project sample: " + error);
            }
        {
            auto branched = motionPlan;
            const auto program = renderpassfixture::hdrBranches();
            auto oversized = program;
            oversized.count = renderpasscomposite::kMaximumPasses + 1;
            check(!renderpasscomposite::valid(oversized), "pass programs enforce their fixed GPU register capacity");
            for (int id = 90; id <= 93; ++id)
            {
                if (id != 90)
                {
                    branched.nodeIds.push_back(id);
                    branched.nodeKinds.push_back(renderpasscomposite::kNodeKind);
                    for (const auto& port : motionPlan.ports)
                        if (port.nodeId == 90)
                        { auto copy = port; copy.nodeId = id; branched.ports.push_back(copy); }
                    branched.operations.push_back({id, renderpasscomposite::kNodeKind, "native-gpu", {}});
                }
                branched.ports.push_back({id, 9, 1, "out", "frame", "image", "rgba32f", "linearSRGB"});
                branched.ports.push_back({id, 10, 1, "in", "frame", "image", "rgba32f", "linearSRGB"});
                branched.ports.push_back({id, 11, 1, "in", "frame", "image", "rgba32f", "linearSRGB"});
                const auto& step = program.steps[static_cast<std::size_t>(id - 90)];
                for (auto& operation : branched.operations)
                    if (operation.nodeId == id) operation.payloadXml = renderpasscomposite::encode(step.parameters);
                if (id == 91)
                    for (const auto& edge : motionPlan.edges)
                        if (edge.toNodeId == 90)
                        { auto copy = edge; copy.toNodeId = id; branched.edges.push_back(copy); }
                if (id >= 92)
                {
                    branched.edges.push_back({90 + step.inputA, 9, id, 10});
                    branched.edges.push_back({90 + step.inputB, 9, id, 11});
                }
            }
            branched.nodeIds.push_back(94); branched.nodeKinds.push_back("video.out");
            branched.operations.push_back({94, "video.out", "native-gpu", {}});
            branched.ports.push_back({94, 0, 1, "in", "frame", "image", "rgba8", "sRGB"});
            branched.edges.push_back({93, 7, 94, 0});
            {
                const bool conditionOk = videowire::compileVisualLayerExecution(branched, compiled, error)
                && compiled.passProgram && compiled.passProgram->count == 4;
                check(conditionOk, "raw AOV consumers share deterministic linear HDR branch execution: " + error);
            }
            const auto expected = renderpasscomposite::gpuProgram(program);
            native.nextFrame->rawPasses = true;
            for (const auto use : {NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export})
            {
                {
                    const bool conditionOk = prepareVisualImportedSceneLayer({branched}, 7, 640, 360, use,
                    &native, layer, owners, error) == VisualImportedScenePreparation::rendered
                    && native.lastRequest.runtimeInputs.passProgram.has_value();
                    check(conditionOk, "preview and export carry the same retained branch program: " + error);
                }
                if (native.lastRequest.runtimeInputs.passProgram)
                {
                    const auto actual = renderpasscomposite::gpuProgram(*native.lastRequest.runtimeInputs.passProgram);
                    check(actual.control == expected.control && actual.modes == expected.modes
                        && actual.colors == expected.colors && actual.references == expected.references
                        && actual.transforms == expected.transforms,
                        "branch wiring, HDR parameters and output transform survive production transport");
                }
            }
            auto reordered = branched;
            std::reverse(reordered.operations.end() - 5, reordered.operations.end() - 1);
            {
                const bool conditionOk = videowire::compileVisualLayerExecution(reordered, compiled, error)
                && compiled.passProgram && renderpasscomposite::gpuProgram(*compiled.passProgram).references == expected.references;
                check(conditionOk, "branch scheduling is independent of serialized pass order: " + error);
            }
            auto invalid = branched;
            invalid.edges.back().fromPort = 9;
            check(!videowire::compileVisualLayerExecution(invalid, compiled, error)
                && error.find("HDR output encoding") != std::string::npos,
                "raw linear HDR is rejected as an SDR file output with an explicit reason");
            invalid = branched;
            for (auto& port : invalid.ports)
                if (port.nodeId == 92 && port.port == 10) port.colorSpace = "sRGB";
            check(!videowire::compileVisualLayerExecution(invalid, compiled, error),
                "linear branch inputs cannot silently reinterpret sRGB values");
            invalid = branched;
            for (auto& edge : invalid.edges)
                if (edge.toNodeId == 92 && edge.toPort == 10) edge.fromNodeId = 93;
            check(!videowire::compileVisualLayerExecution(invalid, compiled, error),
                "feedback cycles are rejected before the native pass program executes");
        }
        motionPlan.ports.back().pixelFormat = "rgba8";
        check(!videowire::compileVisualLayerExecution(motionPlan, compiled, error),
            "motion never admits display-color pixels as raw vectors");
    }
    for (const auto output : { renderpassoutput::Output::Color, renderpassoutput::Output::Depth,
                              renderpassoutput::Output::Normal, renderpassoutput::Output::Motion,
                              renderpassoutput::Output::Emission, renderpassoutput::Output::Mask,
                              renderpassoutput::Output::MaterialId, renderpassoutput::Output::ObjectId })
    {
        auto request = legacyRequest;
        request.imageOutput = output;
        request.rawExportMask = 1u << static_cast<unsigned>(output);
        const auto encoded = visualimportedscenerender::encode(request);
        visualimportedscenerender::Request decoded;
        check(!encoded.empty() && visualimportedscenerender::decode(encoded, decoded)
                  && visualimportedscenerender::sameRequest(request, decoded),
              "every supported image output retains the exact imported request through transport");
        auto outputPlan = importedScenePlan();
        for (auto& operation : outputPlan.operations)
            if (operation.kind == visualimportedscenerender::kRenderNodeKind)
                operation.payloadXml = encoded;
        FakeExecution outputExecution;
        FakeLayer outputLayer;
        std::vector<FakeReceipt> outputOwners;
        std::string outputError;
        for (const auto use : { NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export })
        {
            check(prepareVisualImportedSceneLayer(
                      { outputPlan }, 7, 640, 360, use, &outputExecution,
                      outputLayer, outputOwners, outputError) == VisualImportedScenePreparation::rendered
                      && outputExecution.lastRequest.runtimeInputs.imageOutput == output,
                  "preview and export pass the same output to the native draw and image compositor");
            const auto expectedMask = use == NativeImportedSceneRenderUse::Export ? request.rawExportMask : 0u;
            check(outputExecution.lastRequest.runtimeInputs.rawExportMask == expectedMask
                && outputLayer.rawExportMask == expectedMask && outputLayer.rawExportClipId == 7
                && outputLayer.rawExportRenderId == request.renderStableId
                && !outputOwners.empty() && outputLayer.rawExportFrame == outputOwners.back().frame,
                "only export receives persisted pass selections and retains the exact frame and clip identities");
        }
        for (const auto header : { visualimportedscenerender::kCompiledOperationHeaderV9,
                                  visualimportedscenerender::kCompiledOperationHeaderV11 })
            check(!visualimportedscenerender::decode(std::string(header) + "\n2\n" + encoded, decoded),
                "out-of-order or repeated raw selection envelopes cannot recurse");
        request.rawExportMask = 256;
        check(visualimportedscenerender::encode(request).empty(), "unknown raw output selection bits fail closed");
        if (output != renderpassoutput::Output::Color)
        {
            const auto nested = std::string(visualimportedscenerender::kCompiledOperationHeaderV9)
                + "\n2\n" + encoded;
            check(!visualimportedscenerender::decode(nested, decoded), "nested output envelopes are rejected");
        }
    }
    const auto legacyEncoded = visualimportedscenerender::encode(legacyRequest);
    const auto expectedLegacy = std::string(
        "visual.imported-scene.render.v1\n"
        "71 72\n"
        "\"model-asset-1\"\n"
        "3\n"
        "\"") + std::string(64, 'a')
        + "\"\n\"model/gltf-binary\"\n4096\n1 0\n";
    visualimportedscenerender::Request decodedLegacy;
    check(legacyEncoded == expectedLegacy,
          "a request without material preserves the v1 bytes exactly");
    check(visualimportedscenerender::decode(legacyEncoded, decodedLegacy)
              && visualimportedscenerender::sameRequest(
                  legacyRequest, decodedLegacy)
              && !decodedLegacy.material,
          "the legacy v1 request still decodes without material");

    {
        auto authored = legacyRequest;
        authored.importedLights.selectedDefinition = 2;
        authored.importedLights.intensityMultipliers = {{1, 3.0f}, {2, 0.25f}};
        const auto bytes = visualimportedscenerender::encode(authored);
        const auto prefix = std::string(visualimportedscenerender::kCompiledOperationHeaderV17)
            + "\n2 2\n1 1077936128\n2 1048576000\n";
        visualimportedscenerender::Request reopened;
        check(bytes == prefix + expectedLegacy && visualimportedscenerender::decode(bytes, reopened)
            && visualimportedscenerender::sameRequest(authored, reopened),
            "v17 imported light definitions use canonical sorted multipliers and preserve nested legacy bytes");
        reopened.importedLights.selectedDefinition = 1;
        check(!visualimportedscenerender::sameRequest(authored, reopened),
            "selected imported definition participates in immutable request identity");
        check(!visualimportedscenerender::decode(replaceLine(bytes, 2, "2 1077936128"), reopened),
            "duplicate or unsorted imported definitions reject");
        check(!visualimportedscenerender::decode(replaceLine(bytes, 2, "1 2147483648"), reopened),
            "negative zero imported multiplier rejects");
        check(!visualimportedscenerender::decode(prefix + bytes, reopened),
            "imported-light envelopes cannot recurse");
        auto bad = authored;
        bad.importedLights.intensityMultipliers[1] = 65;
        check(visualimportedscenerender::encode(bad).empty(), "out-of-bound authored multiplier rejects");
        bad = authored;
        bad.sceneSnapshot = composedScene();
        check(visualimportedscenerender::encode(bad).empty(), "imported definitions cannot attach to an unrelated composed scene");
        auto plan = importedScenePlan();
        plan.operations.back().payloadXml = bytes;
        std::vector<videowire::CompiledVisualLayerPlan> plans {plan};
        std::map<std::string, double> parameters;
        videohelper::importedscene::seedImportedSceneRuntimeParameters(plans, 7, parameters);
        check(parameters["visual71/selectedLightIntensity"] == 0.25,
            "selected imported light runtime target seeds from its source stable ID and saved definition");
        parameters["visual71/selectedLightIntensity"] = 0.5;
        FakeExecution execution; FakeLayer layer; std::vector<FakeReceipt> owners;
        videohelper::importedscene::VisualImportedScenePlanCache cache;
        std::string error;
        for (const auto use : {NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export})
        {
            {
                const bool conditionOk = prepareVisualImportedSceneLayerAtTime(plans, 7, 640, 360, 2.5, 24, use,
                &execution, &cache, layer, owners, error, &parameters) == VisualImportedScenePreparation::rendered;
                check(conditionOk, "selected light control reaches same-time ordinary preview/export request: " + error);
            }
            check(execution.lastRequest.frame.frame == 60
                && execution.lastRequest.importedLights.selectedDefinition == 2
                && execution.lastRequest.importedLights.intensityMultipliers.at(1) == 3
                && execution.lastRequest.importedLights.intensityMultipliers.at(2) == 0.5f,
                "modulation resolves only the selected definition and retains the other's authored override");
        }
        parameters["visual71/selectedLightIntensity"] = std::numeric_limits<double>::quiet_NaN();
        const auto calls = execution.previewCalls;
        check(prepareVisualImportedSceneLayerAtTime(plans, 7, 640, 360, 2.5, 24,
            NativeImportedSceneRenderUse::Preview, &execution, &cache, layer, owners, error, &parameters)
                == VisualImportedScenePreparation::rejected && execution.previewCalls == calls,
            "nonfinite selected imported light modulation refuses before execution");
        authored.importedLights = {};
        check(visualimportedscenerender::encode(authored) == expectedLegacy,
            "no imported override retains the exact released render payload");
    }

    {
        auto constant=legacyRequest;
        HarmonicMIDI::grid::SceneLightRecord environment;
        environment.id.value=702; environment.kind=HarmonicMIDI::grid::SceneLightKind::Environment;
        environment.color={.25f,.5f,.75f}; environment.intensity=8;
        constant.light=environment;
        auto expected=expectedLegacy;
        expected.replace(0,std::string(visualimportedscenerender::kCompiledOperationHeader).size(),
            visualimportedscenerender::kCompiledOperationHeaderV8);
        expected+="0 0 0 0 0 1\n702 2 0 0 0 0 0 0 1065353216 1065353216 1065353216 1065353216 "
            "1048576000 1056964608 1061158912 1090519040 0\n";
        visualimportedscenerender::Request reopened;
        check(visualimportedscenerender::encode(constant)==expected
            && visualimportedscenerender::decode(expected,reopened)
            && visualimportedscenerender::sameRequest(constant,reopened),
            "Frozen constant-environment override v8 bytes retain exact legacy encoding/decoding");
        auto sky=constant; sky.light->sky.version=1; sky.light->sky.orientationDegrees=45;
        const auto bytes=visualimportedscenerender::encode(sky);
        check(bytes.find(visualimportedscenerender::kCompiledOperationHeaderV15)==0
            && visualimportedscenerender::decode(bytes,reopened)
            && visualimportedscenerender::sameRequest(sky,reopened)
            && visualimportedscenerender::encode(reopened)==bytes,
            "Version 15 sky override preserves exact bounded descriptor through nested released request");
        reopened.light->sky.sunAzimuthDegrees+=1;
        check(!visualimportedscenerender::sameRequest(sky,reopened),
            "Sky override direction participates in exact request identity");
        auto invalid=bytes;
        invalid[invalid.find('\n')+1]='2';
        check(!visualimportedscenerender::decode(invalid,reopened),"Unknown override sky version rejects");
        invalid=bytes; invalid[invalid.find('\n')+1]='0';
        check(!visualimportedscenerender::decode(invalid,reopened),"Inconsistent sky envelope flag rejects");
        check(!visualimportedscenerender::decode(bytes.substr(0,bytes.find('\n')+1)+"1\n"+bytes,reopened),
            "Nested sky override envelopes cannot recurse");
        sky.light->sky.sunRadiusDegrees=0;
        check(visualimportedscenerender::encode(sky).empty(),"Invalid override descriptor bounds reject");
        constant.light->sky.orientationDegrees=1;
        check(visualimportedscenerender::encode(constant).empty(),
            "Inactive override descriptor cannot hide noncanonical sky fields");
    }
    auto boundSceneRequest = renderRequest();
    HarmonicMIDI::grid::SceneCameraRecord boundCamera;
    boundCamera.id.value = 701;
    boundCamera.transform.translation.z = 4.0f;
    boundCamera.verticalFovRadians = 0.75f;
    boundCamera.nearPlane = 0.25f;
    boundCamera.farPlane = 500.0f;
    boundSceneRequest.camera = boundCamera;
    HarmonicMIDI::grid::SceneLightRecord boundLight;
    boundLight.id.value = 702;
    boundLight.kind = HarmonicMIDI::grid::SceneLightKind::Point;
    boundLight.transform.translation = { 1.0f, 2.0f, 3.0f };
    boundLight.color = { 0.25f, 0.5f, 0.75f };
    boundLight.intensity = 8.0f;
    boundLight.range = 16.0f;
    boundSceneRequest.light = boundLight;
    const auto boundSceneEncoded = visualimportedscenerender::encode(boundSceneRequest);
    visualimportedscenerender::Request decodedBoundScene;
    check(boundSceneEncoded.rfind(
              std::string(visualimportedscenerender::kCompiledOperationHeaderV8) + "\n",
              0) == 0
              && visualimportedscenerender::decode(boundSceneEncoded, decodedBoundScene)
              && visualimportedscenerender::sameRequest(boundSceneRequest, decodedBoundScene)
              && visualimportedscenerender::encode(decodedBoundScene) == boundSceneEncoded,
          "the immutable camera and light request has one canonical exact round trip");
    auto invalidBoundScene = boundSceneRequest;
    {
        auto request = renderRequest();
        request.light = boundLight;
        request.light->kind = HarmonicMIDI::grid::SceneLightKind::Directional;
        request.light->id.value = 74;
        request.noteLight = visualimportedscenerender::NoteLight { 9, 2, 60 };
        const auto payload = visualimportedscenerender::encode(request);
        visualimportedscenerender::Request decoded;
        check(!payload.empty() && visualimportedscenerender::decode(payload, decoded)
            && visualimportedscenerender::sameRequest(request, decoded),
            "note-light controls have an exact V10 round trip");
        const auto nested = std::string(visualimportedscenerender::kCompiledOperationHeaderV10)
            + "\n9 1073741824 1114636288\n" + payload;
        check(!visualimportedscenerender::decode(nested, decoded),
            "note-light envelopes cannot recursively nest");
        auto invalid = request;
        invalid.noteLight->track = 65536;
        check(visualimportedscenerender::encode(invalid).empty(), "note-light track capacity is explicit");
        invalid = request; invalid.sceneSnapshot = composedScene();
        check(visualimportedscenerender::encode(invalid).empty(),
            "unsupported composed-scene note light is rejected before execution");

        auto plan = importedScenePlan();
        plan.nodeKinds.insert(plan.nodeKinds.end() - 1,
            { "visual.3d.transform", "visual.3d.light.directional" });
        plan.nodeIds.insert(plan.nodeIds.end() - 1, { 72, 73 });
        plan.operations.insert(plan.operations.end() - 1, {
            { 72, "visual.3d.transform", "control-eval", "" },
            { 73, "visual.3d.light.directional", "control-eval", "" } });
        plan.operations.back().payloadXml = payload;
        plan.ports.insert(plan.ports.end(), {
            { 72, 0, 1, "out", "control", "transform3D", "unspecified", "unspecified" },
            { 73, 0, 1, "in", "control", "transform3D", "unspecified", "unspecified" },
            { 73, 1, 1, "out", "control", "light", "unspecified", "unspecified" },
            { 71, 2, 1, "in", "control", "material", "unspecified", "unspecified" },
            { 71, 3, 1, "in", "control", "mesh", "unspecified", "unspecified" },
            { 71, 4, 1, "out", "frame", "depth", "r32f", "unspecified" },
            { 71, 5, 1, "in", "control", "camera", "unspecified", "unspecified" },
            { 71, 6, 1, "in", "control", "light", "unspecified", "unspecified" } });
        plan.edges.insert(plan.edges.end(), { { 72, 0, 73, 0 }, { 73, 1, 71, 6 } });
        auto score = std::make_shared<arbitmod::Score>();
        score->scoreRevision = 1;
        arbitmod::Note note;
        note.id = -101; note.midiNote = 84; note.velocity = 63.5f;
        note.freqHz = 1046.5f; note.startBeat = 0; note.lengthBeats = 4; note.trackId = 9;
        score->notes.push_back(note);
        canonicalblockc::FrameKey key;
        key.projectGeneration = key.sourceGeneration = key.helperGeneration = 1;
        key.backendGeneration = key.deviceGeneration = key.scoreGeneration = 1;
        key.beatMapGeneration = key.fpsGeneration = key.loopGeneration = key.seekGeneration = 1;
        key.fps = 60;
        canonicalblockc::FrameProducer producer;
        const auto notes = producer.evaluate(key, score, 0.0f);
        FakeExecution lightExecution;
        FakeLayer lightLayer;
        std::vector<FakeReceipt> lightOwners;
        std::string lightError;
        for (const auto use : { NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export })
        {
            lightLayer.canonicalBlockCFrame = notes;
            {
                const bool conditionOk = prepareVisualImportedSceneLayer({ plan }, 7, 640, 360, use,
                    &lightExecution, lightLayer, lightOwners, lightError) == VisualImportedScenePreparation::rendered;
                check(conditionOk, "note light reaches preview and export: " + lightError);
            }
            check(lightExecution.lastRequest.light
                && std::abs(lightExecution.lastRequest.light->intensity - 8.0f) < 0.001f
                && lightExecution.lastRequest.light->transform.rotation.y > 0.3f,
                "sounding note velocity and pitch change native light intensity and direction");
        }
        lightLayer.canonicalBlockCFrame.reset();
        check(prepareVisualImportedSceneLayer({ plan }, 7, 640, 360,
                NativeImportedSceneRenderUse::Preview, &lightExecution, lightLayer, lightOwners, lightError)
                == VisualImportedScenePreparation::rejected
                && lightError.find("canonical score frame") != std::string::npos,
            "missing note-light score input fails before native submission");
        auto otherTrack = *request.light;
        check(videohelper::applyNoteReactiveLight(otherTrack, { 8, 2, 60 }, notes)
                && otherTrack.intensity == 0, "unmatched note track produces no direct light");
        auto replay = *request.light;
        check(videohelper::applyNoteReactiveLight(replay, *request.noteLight, notes)
                && lightExecution.lastRequest.light
                && visualimportedscenerender::detail::sameLight(replay, *lightExecution.lastRequest.light),
            "note-light replay depends only on the canonical frame and authored controls");

        auto rig = plan;
        rig.nodeIds.insert(rig.nodeIds.end(), { 90, 91, 92 });
        rig.nodeKinds.insert(rig.nodeKinds.end(), { "visual.particles", "video.blend", "video.out" });
        rig.operations.insert(rig.operations.end(), {
            { 90, "visual.particles", "native-gpu", "<NodeParams motionMode=\"1\" spawnTrack=\"9\" alpha=\"0.4\"/>" },
            { 91, "video.blend", "native-gpu", "<NodeParams opacity=\"1\" mode=\"0\"/>" },
            { 92, "video.out", "native-gpu", "" } });
        rig.ports.insert(rig.ports.end(), {
            { 90, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
            { 91, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
            { 91, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
            { 91, 2, 1, "in", "frame", "image", "rgba8", "sRGB" },
            { 91, 3, 1, "in", "frame", "image", "rgba8", "sRGB" },
            { 92, 0, 1, "in", "frame", "image", "rgba8", "sRGB" } });
        rig.edges.insert(rig.edges.end(), { { 71, 1, 91, 0 }, { 90, 1, 91, 2 }, { 91, 1, 92, 0 } });
        videowire::VisualLayerExecution rigExecution;
        {
            const bool conditionOk = videowire::compileVisualLayerExecution(rig, rigExecution, lightError)
            && rigExecution.importedParticleOverlay && rigExecution.particles
            && rigExecution.particleNodeId == 90 && rigExecution.importedSceneRender->noteLight
            && rigExecution.particleParameters.spawnTrack == 9;
            check(conditionOk, "note light and particles admit one native overlay schedule: " + lightError);
        }
        check(videowire::isImportedParticleOverlayPlan({ rig }, 7),
            "viewport and export select scene preparation for the particle rig");
        rig.ports.insert(rig.ports.end(), {
            { 90, 0, 1, "in", "event", "unspecified", "unspecified", "unspecified" },
            { 90, 2, 1, "in", "control", "points3D", "unspecified", "unspecified" },
            { 90, 3, 1, "in", "control", "points3D", "unspecified", "unspecified" } });
        check(videowire::compileVisualLayerExecution(rig, rigExecution, lightError)
            && rigExecution.importedParticleOverlay && rigExecution.particleParameters.geometryCount == 0,
            "new unconnected geometry ports preserve the existing note particle and light rig");
        auto invalidGeometryPort = rig;
        invalidGeometryPort.ports.back().dataType = "geometry3D";
        check(!videowire::compileVisualLayerExecution(invalidGeometryPort, rigExecution, lightError),
            "particle overlay rejects a forged optional geometry descriptor");
        for (const auto use : { NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export })
        {
            lightLayer.canonicalBlockCFrame = notes;
            lightLayer.shaderClock = { 1.25, 30 };
            {
                const bool conditionOk = prepareVisualImportedSceneLayer({ rig }, 7, 640, 360, use,
                    &lightExecution, lightLayer, lightOwners, lightError) == VisualImportedScenePreparation::rendered;
                check(conditionOk, "particle rig prepares the same native scene in preview and export: " + lightError);
            }
            check(lightLayer.texture == 44 && lightLayer.nativeTextureView == 44
                    && lightLayer.canonicalBlockCFrame == notes
                    && lightLayer.shaderClock.frame == 30 && lightLayer.shaderClock.time == 1.25,
                "scene preparation retains particle clock, canonical note identity and native background");
        }
        for (const auto wrongPort : { 0, 3 })
        {
            auto invalidRig = rig;
            invalidRig.edges[invalidRig.edges.size() - 2].toPort = wrongPort;
            check(!videowire::compileVisualLayerExecution(invalidRig, rigExecution, lightError),
                "particle rig rejects an aliased or swapped compositor input");
        }
        auto invalidRig = rig;
        invalidRig.operations[invalidRig.operations.size() - 3].payloadXml = "<NodeParams motionMode=\"0\"/>";
        check(!videowire::compileVisualLayerExecution(invalidRig, rigExecution, lightError)
                && lightError.find("deterministic timeline motion mode") != std::string::npos,
            "particle rig rejects stateful legacy particles rather than claiming repeatable export");
    }
    invalidBoundScene.light->id = {};
    check(visualimportedscenerender::encode(invalidBoundScene).empty(),
          "light transport rejects a missing stable identity");
    auto malformedBoundScene = boundSceneEncoded;
    const auto lightIdentity = malformedBoundScene.find("\n702 ");
    check(lightIdentity != std::string::npos,
          "v8 encodes the exact light identity");
    if (lightIdentity != std::string::npos)
        malformedBoundScene.replace(lightIdentity + 1, 3, "0");
    check(lightIdentity != std::string::npos
              && !visualimportedscenerender::decode(
                  malformedBoundScene, decodedBoundScene),
          "v8 rejects a malformed light identity deterministically");

    visualimportedscenerender::Request composedRequest;
    composedRequest.sourceStableId = 7;
    composedRequest.renderStableId = 12;
    composedRequest.sceneSnapshot = composedScene();
    composedRequest.structuralRevision = 9;
    composedRequest.evaluationRevision = 12;
    const auto composedEncoded = visualimportedscenerender::encode(composedRequest);
    visualimportedscenerender::Request decodedComposed;
    check(composedEncoded.rfind(
              std::string(visualimportedscenerender::kCompiledOperationHeaderV8)
                  + "\nscene ", 0) == 0
              && visualimportedscenerender::decode(composedEncoded, decodedComposed)
              && visualimportedscenerender::sameRequest(composedRequest, decodedComposed)
              && decodedComposed.sceneSnapshot
              && decodedComposed.sceneSnapshot->objectCount == 2
              && decodedComposed.sceneSnapshot->objects[1].firstVertex == 3
              && decodedComposed.sceneSnapshot->objects[1].firstIndex == 3
              && decodedComposed.sceneSnapshot->activeCamera.value == 50
              && decodedComposed.sceneSnapshot->lights[0].id.value == 40
              && visualimportedscenerender::encode(decodedComposed) == composedEncoded,
          "v8 preserves the exact composed object camera and light topology");
    auto invalidComposed = composedRequest;
    auto invalidScene = std::make_shared<HarmonicMIDI::grid::Visual3DScene>(
        *composedRequest.sceneSnapshot);
    invalidScene->objects[1].id = invalidScene->objects[0].id;
    invalidComposed.sceneSnapshot = std::move(invalidScene);
    check(visualimportedscenerender::encode(invalidComposed).empty(),
          "v8 rejects duplicate composed object identity before publication");

    const auto exactMaterialRequest = renderRequest(true);
    const auto materialEncoded
        = visualimportedscenerender::encode(exactMaterialRequest);
    visualimportedscenerender::Request decodedMaterial;
    check(materialEncoded.rfind(
              std::string(visualimportedscenerender::kCompiledOperationHeaderV2)
                  + "\n",
              0) == 0
              && visualimportedscenerender::decode(
                  materialEncoded, decodedMaterial)
              && visualimportedscenerender::sameRequest(
                  exactMaterialRequest, decodedMaterial)
              && visualimportedscenerender::encode(decodedMaterial)
                  == materialEncoded,
          "the v2 material request has one canonical exact round trip");
    check(decodedMaterial.material
              && decodedMaterial.material->sceneSnapshot == nullptr
              && floatBits(decodedMaterial.material->program.operations.front()
                               .literal[1])
                  == floatBits(-0.0f)
              && decodedMaterial.material->binding.textures.size() == 1
              && decodedMaterial.material->binding.textures.front()
                     .videoResource[15] == 16,
          "v2 preserves float bits and value fields without a scene owner");

    const auto exactDeformationRequest = renderRequest(false, true);
    const auto deformationEncoded
        = visualimportedscenerender::encode(exactDeformationRequest);
    visualimportedscenerender::Request decodedDeformation;
    check(deformationEncoded.rfind(
              std::string(visualimportedscenerender::kCompiledOperationHeaderV3)
                  + "\n",
              0) == 0
              && visualimportedscenerender::decode(
                  deformationEncoded, decodedDeformation)
              && visualimportedscenerender::sameRequest(
                  exactDeformationRequest, decodedDeformation)
              && visualimportedscenerender::encode(decodedDeformation)
                  == deformationEncoded,
          "the v3 deformation request has one canonical exact round trip");
    check(decodedDeformation.deformation
              && decodedDeformation.deformation->deformationStableId == 73
              && decodedDeformation.deformation->clipName == "Idle",
          "v3 retains the exact deformation identity and playback request");

    auto exactVertexRequest = renderRequest(true, true);
    exactVertexRequest.material->vertexModifier = vertexModifierRequest();
    const auto vertexEncoded = visualimportedscenerender::encode(exactVertexRequest);
    visualimportedscenerender::Request decodedVertex;
    check(vertexEncoded.rfind(
              std::string(visualimportedscenerender::kCompiledOperationHeaderV4) + "\n",
              0) == 0
              && visualimportedscenerender::decode(vertexEncoded, decodedVertex)
              && visualimportedscenerender::sameRequest(exactVertexRequest, decodedVertex)
              && visualimportedscenerender::encode(decodedVertex) == vertexEncoded,
          "the v4 material and vertex request has one canonical exact round trip");
    check(decodedVertex.material && decodedVertex.material->vertexModifier
              && decodedVertex.material->vertexModifier->rootId == 103
              && floatBits(static_cast<float>(
                     decodedVertex.material->vertexModifier->records[1].parameters[2]))
                   == floatBits(-0.0f),
          "v4 preserves the graph-native vertex stage and signed parameter bits");

    auto unsupportedVertex = exactVertexRequest;
    unsupportedVertex.material->vertexModifier->records[1].parameters[3] = 1.0;
    check(visualimportedscenerender::encode(unsupportedVertex).empty(),
          "vertex transport rejects data outside the typed parameter shape");
    unsupportedVertex = exactVertexRequest;
    unsupportedVertex.material->vertexModifier->records.push_back(
        unsupportedVertex.material->vertexModifier->records.front());
    check(visualimportedscenerender::encode(unsupportedVertex).empty(),
          "vertex transport rejects duplicate or unreachable operations");

    auto foreignOwner = exactMaterialRequest;
    foreignOwner.material->sceneSnapshot
        = std::make_shared<const HarmonicMIDI::grid::Visual3DScene>();
    visualimportedscenerender::Request decodedForeignOwner;
    check(visualimportedscenerender::encode(foreignOwner) == materialEncoded
              && visualimportedscenerender::decode(
                  visualimportedscenerender::encode(foreignOwner),
                  decodedForeignOwner)
              && decodedForeignOwner.material
              && decodedForeignOwner.material->sceneSnapshot == nullptr,
          "material transport excludes rather than serializes a scene snapshot owner");
    auto tooManyOperations = exactMaterialRequest;
    tooManyOperations.material->program.operations.resize(
        surfacematerial::kMaximumOperations + 1u);
    check(visualimportedscenerender::encode(tooManyOperations).empty(),
          "material transport rejects an oversized operation count");
    auto tooManyTextures = exactMaterialRequest;
    tooManyTextures.material->binding.textures.resize(
        surfacematerialbinding::kMaximumTextureBindings + 1u);
    check(visualimportedscenerender::encode(tooManyTextures).empty(),
          "material transport rejects an oversized texture-binding count");
    visualimportedscenerender::Request rejectedMaterial;
    check(!visualimportedscenerender::decode(
              replaceLine(materialEncoded, 9, "1 1 129"), rejectedMaterial),
          "v2 rejects an operation count above the hard bound before allocation");
    check(!visualimportedscenerender::decode(
              replaceLine(materialEncoded, 0,
                          std::string(visualimportedscenerender::kCompiledOperationHeader)),
              rejectedMaterial),
          "material fields cannot be smuggled through the v1 header");

    std::string error;
    FakeExecution execution;
    FakeLayer previewLayer;
    std::vector<FakeReceipt> previewOwners;

    check(prepareVisualImportedSceneLayer(
              {}, 7, 640, 360, NativeImportedSceneRenderUse::Preview,
              &execution, previewLayer, previewOwners, error)
              == VisualImportedScenePreparation::notApplicable,
          "a clip without an imported scene render stays on its ordinary route");

    auto plan = importedScenePlan();
    std::vector<videowire::CompiledVisualLayerPlan> plans { plan };
    for (const auto& extentPlan : { importedScenePlan(), composedScenePlan() })
    {
        const std::vector<videowire::CompiledVisualLayerPlan> extentPlans { extentPlan };
        const auto originalIdentity = videohelper::importedscene::exactImportedScenePayloadIdentity(extentPlan);
        videohelper::importedscene::VisualImportedScenePlanCache extentCache;
        FakeExecution extentExecution;
        for (const auto& dimensions : { std::pair<int, int>{3840, 2160}, {320, 180} })
        {
            extentExecution.nextFrame->frameWidth = static_cast<std::uint32_t>(dimensions.first);
            extentExecution.nextFrame->descriptorWidth = extentExecution.nextFrame->frameWidth;
            extentExecution.nextFrame->frameHeight = static_cast<std::uint32_t>(dimensions.second);
            FakeLayer layer;
            std::vector<FakeReceipt> owners;
            error.clear();
            {
                const bool conditionOk = prepareVisualImportedSceneLayerAtTime(extentPlans, extentPlan.clipId,
                      dimensions.first, dimensions.second, 0.5, 30.0,
                      NativeImportedSceneRenderUse::Export, &extentExecution, &extentCache,
                      layer, owners, error) == VisualImportedScenePreparation::rendered
                      && extentExecution.lastRequest.dimensions.width == static_cast<std::uint32_t>(dimensions.first)
                      && extentExecution.lastRequest.dimensions.height == static_cast<std::uint32_t>(dimensions.second)
                      && layer.texWidth == dimensions.first && layer.texHeight == dimensions.second
                      && videohelper::importedscene::exactImportedScenePayloadIdentity(extentPlans.front())
                          == originalIdentity;
                check(conditionOk, "project and diagnostic extents reuse exact scene payload identity with newly sized native frames: " + error);
            }
        }
        // Matching byte counts alone would not distinguish these swapped extents.
        extentExecution.nextFrame->frameWidth = 180;
        extentExecution.nextFrame->descriptorWidth = 180;
        extentExecution.nextFrame->frameHeight = 320;
        FakeLayer wrongExtentLayer;
        std::vector<FakeReceipt> wrongExtentOwners;
        error.clear();
        check(prepareVisualImportedSceneLayerAtTime(extentPlans, extentPlan.clipId,
                  320, 180, 0.5, 30.0, NativeImportedSceneRenderUse::Export,
                  &extentExecution, &extentCache, wrongExtentLayer, wrongExtentOwners, error)
                  == VisualImportedScenePreparation::rejected
                  && error == "imported scene native frame dimensions do not match the declared render extent",
              "diagnostic rendering cannot relabel an equal-byte-count native frame of the wrong shape");
    }
    for (const auto& hdrPlan : { importedScenePlan(), composedScenePlan() })
        for (const auto* api : { "opengl", "metal" })
        {
            FakeExecution hdrExecution;
            hdrExecution.nextFrame->api = api;
            hdrExecution.nextFrame->linearColor = true;
            const std::vector<videowire::CompiledVisualLayerPlan> hdrPlans {hdrPlan};
            for (const auto use : {NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export})
            {
                // Fresh cache models reopen; the persisted plan stays identical,
                // while the selected export profile chooses native precision.
                videohelper::importedscene::VisualImportedScenePlanCache reopened;
                FakeLayer hdrLayer;
                std::vector<FakeReceipt> owners;
                {
                    const bool conditionOk = prepareVisualImportedSceneLayerAtTime(hdrPlans, hdrPlan.clipId, 640, 360, 0, 24,
                    use, &hdrExecution, &reopened, hdrLayer, owners, error, nullptr, {}, nullptr, true)
                        == VisualImportedScenePreparation::rendered
                    && hdrExecution.lastRequest.runtimeInputs.linearColor
                    && arbitgpu::isLinearSceneColor(hdrLayer.nativeTextureDescriptor);
                    check(conditionOk, "imported/generated reopened plans retain explicit linear native preview/export color: " + error);
                }
                hdrExecution.nextFrame->linearColor = false;
                check(prepareVisualImportedSceneLayerAtTime(hdrPlans, hdrPlan.clipId, 640, 360, 0, 24,
                    use, &hdrExecution, &reopened, hdrLayer, owners, error, nullptr, {}, nullptr, true)
                        == VisualImportedScenePreparation::rejected,
                    "HDR scene preparation rejects an SDR attachment from the native executor");
                hdrExecution.nextFrame->linearColor = true;
            }
        }
    check(prepareVisualImportedSceneLayer(
              plans, 7, 640, 360, NativeImportedSceneRenderUse::Preview,
              &execution, previewLayer, previewOwners, error)
              == VisualImportedScenePreparation::rendered,
          "the exact imported scene plan renders for preview");
    check(execution.previewCalls == 1 && execution.exportCalls == 0,
          "preview uses the preview execution owner");
    check(previewOwners.size() == 1 && previewOwners.front().owner == 1,
          "preview retains its complete frame receipt");
    check(previewOwners.front().frame->depthImageHandle() == 55
              && previewOwners.front().frame->depthTextureViewHandle() == 66,
          "preview retains the native Render 3D depth publication");
    check(previewLayer.texture == 44 && previewLayer.nativeTextureView == 44
              && previewLayer.nativeTextureBackend == "opengl"
              && previewLayer.texWidth == 640 && previewLayer.texHeight == 360,
          "the exact native OpenGL texture view reaches the ordinary layer");
    check(previewLayer.depthTexture == 66
              && previewLayer.nativeDepthTextureBackend == "opengl"
              && previewLayer.nativeDepthTextureView == 66
              && previewLayer.depthWidth == 640 && previewLayer.depthHeight == 360,
          "the exact native OpenGL R32F depth view reaches the ordinary layer");
    check(execution.lastRequest.asset.id == "model-asset-1"
              && execution.lastRequest.asset.version == 3
              && execution.lastRequest.sceneIndex == 0,
          "the compiled exact asset identity reaches payload execution");

    {
        using namespace videohelper::geometry;
        const auto fixture = composedTimelineGeometryFixture(error);
        check(fixture.has_value(), "timeline Geometry3D fixture admits its fixed mesh: " + error);
        if (fixture) {
            const auto encoded = visualimportedscenerender::encode(*fixture);
            visualimportedscenerender::Request decoded;
            check(encoded.find(visualimportedscenerender::kCompiledOperationHeaderV14) == 0
                && visualimportedscenerender::decode(encoded, decoded)
                && visualimportedscenerender::sameRequest(*fixture, decoded),
                "V14 retains exact timeline mesh/object binding and all authored scene state");
            auto timePlan = composedScenePlan();
            timePlan.nodeIds = {6,11}; timePlan.nodeKinds = {videowire::geometry::kRetainedSceneOperation,"visual.3d.render"};
            timePlan.operations = {{6,timePlan.nodeKinds[0],"control-eval",""},{11,timePlan.nodeKinds[1],"native-gpu",encoded}};
            timePlan.edges = {{6,0,11,0}};
            timePlan.ports = {{6,0,1,"out","control","scene3D","unspecified","unspecified"},
                {11,0,1,"in","control","scene3D","unspecified","unspecified"},
                {11,1,1,"out","frame","image","rgba8","sRGB"}};
            videowire::VisualLayerExecution timeCompiled;
            {
                const bool conditionOk = videowire::compileVisualLayerExecution(timePlan,timeCompiled,error)
                && timeCompiled.importedSceneRender && timeCompiled.importedSceneRender->timelineGeometry;
                check(conditionOk, "timeline mesh binding keeps the exact two-operation retained scene schedule: " + error);
            }
            auto old = *fixture; old.timelineGeometry.reset();
            const auto oldBytes = visualimportedscenerender::encode(old);
            check(oldBytes.find(visualimportedscenerender::kCompiledOperationHeaderV8) == 0
                && visualimportedscenerender::decode(oldBytes, decoded) && !decoded.timelineGeometry,
                "static retained scenes keep the existing V8 bytes and decoder");
            videowire::geometry::OperationRecord oversized {601,videowire::geometry::OperationCode::grid,601,601,{1024,1024,1,0}};
            check(!videowire::geometry::admitTimelineSceneOperation(oversized,error),
                "timeline consumer rejects oversized intermediate generators before materialization");
            auto wrong = *fixture; wrong.timelineGeometry->object = {11};
            check(!visualimportedscenerender::valid(wrong) && visualimportedscenerender::encode(wrong).empty(),
                "timeline plan cannot bind the static cube's different mesh");
            wrong = *fixture; wrong.timelineGeometry->loweredPlan[0] = 'g';
            check(!visualimportedscenerender::valid(wrong), "timeline binding rejects malformed plan hex");
            wrong = *fixture; wrong.timelineGeometry->loweredPlan.assign(videowire::geometry::kMaximumTimelineScenePlanBytes + 1, '0');
            check(!visualimportedscenerender::valid(wrong), "timeline binding rejects an oversized lowered plan before decoding");
            check(!visualimportedscenerender::decode(encoded + "extra", decoded), "V14 rejects trailing bytes");
            auto shared = std::make_shared<HarmonicMIDI::grid::Visual3DScene>(*fixture->sceneSnapshot);
            shared->objects[1].firstVertex = shared->objects[0].firstVertex;
            shared->objects[1].vertexCount = shared->objects[0].vertexCount;
            shared->objects[1].firstIndex = shared->objects[0].firstIndex;
            shared->objects[1].indexCount = shared->objects[0].indexCount;
            wrong = *fixture; wrong.sceneSnapshot = shared;
            check(!visualimportedscenerender::valid(wrong), "timeline deformation rejects a mesh shared by another draw");
            class Capabilities final : public BackendCapabilitySource {
                BackendCapabilities geometryCoreCapabilities() const override {
                    BackendCapabilities c; c.backendIdentity = "owned-fixture"; c.deviceIdentity = "fixture-device";
                    c.nativeGpuAvailable = c.geometryCoreExecution = c.immutableSourceBuffers = c.stableElementIds = c.typedFieldEvaluation = true;
                    c.supportedCarriers = carrierBit(videowire::geometry::CarrierKind::geometry3D);
                    return c;
                }
            } source;
            GeometryCorePlanRuntime runtime(source);
            const PlanOwnerIdentity preview {3,4,5,8,9,PlanUse::preview};
            auto exportOwner = preview; exportOwner.use = PlanUse::exportRender;
            auto evaluate = [&](const PlanOwnerIdentity& owner, double seconds) {
                return evaluateComposedTimelineGeometry(runtime, owner, *fixture->timelineGeometry,
                    fixture->sceneSnapshot, seconds, error);
            };
            auto before = evaluate(preview,0), changed = evaluate(preview,0.5), repeated = evaluate(preview,0.5);
            auto looped = evaluate(preview,4.5), exported = evaluate(exportOwner,0.5), sought = evaluate(preview,0);
            check(before && changed && repeated && looped && exported && sought,
                "preview, seek, loop and export use the existing admission owner: " + error);
            if (before && changed && repeated && looped && exported && sought) {
                {
                    const bool conditionOk = verifyComposedTimelinePositions(*changed,*fixture->sceneSnapshot,error);
                    check(conditionOk, error);
                }
                check(sceneaov::detail::encodeScene(*changed) == sceneaov::detail::encodeScene(*repeated)
                    && sceneaov::detail::encodeScene(*changed) == sceneaov::detail::encodeScene(*looped)
                    && sceneaov::detail::encodeScene(*changed) == sceneaov::detail::encodeScene(*exported)
                    && sceneaov::detail::encodeScene(*before) == sceneaov::detail::encodeScene(*sought)
                    && sceneaov::detail::encodeScene(*before) == sceneaov::detail::encodeScene(*fixture->sceneSnapshot),
                    "owned snapshots retain topology, replay and authored state after later frames");
                auto incomplete = preview; incomplete.deviceGeneration = 0;
                check(!evaluate(incomplete,0.5), "timeline evaluation rejects an incomplete device owner");
                check(!evaluate(preview,std::numeric_limits<double>::quiet_NaN()), "timeline evaluation rejects nonfinite time");
            }
        }
    }

    {
        auto snapshot=geometryCompositionFixture(error);
        check(snapshot!=nullptr,"procedural mesh and instance composition produces valid scene records");
        auto retainedPlan=composedScenePlan();
        retainedPlan.nodeIds={6,11};
        retainedPlan.nodeKinds={videowire::geometry::kRetainedSceneOperation,"visual.3d.render"};
        visualimportedscenerender::Request request;
        request.sourceStableId=7; request.renderStableId=12;
        request.structuralRevision=9; request.evaluationRevision=12;
        request.sceneSnapshot=snapshot;
        retainedPlan.operations={{6,retainedPlan.nodeKinds[0],"control-eval",""},
                                 {11,retainedPlan.nodeKinds[1],"native-gpu",visualimportedscenerender::encode(request)}};
        retainedPlan.edges={{6,0,11,0}};
        retainedPlan.ports={{6,0,1,"out","control","scene3D","unspecified","unspecified"},
                            {11,0,1,"in","control","scene3D","unspecified","unspecified"},
                            {11,1,1,"out","frame","image","rgba8","sRGB"}};
        videowire::VisualLayerExecution retainedCompiled;
        {
            const bool conditionOk = videowire::compileVisualLayerExecution(retainedPlan, retainedCompiled, error);
            check(conditionOk, "canonical retained scene without a Frame prerequisite compiles: " + error);
        }
        FakeExecution retainedExecution;
        videohelper::importedscene::VisualImportedScenePlanCache cache;
        FakeLayer layer;
        std::vector<FakeReceipt> owners;
        for (const auto use : {NativeImportedSceneRenderUse::Preview,NativeImportedSceneRenderUse::Export}) {
            const std::vector<videowire::CompiledVisualLayerPlan> plans{retainedPlan};
            check(prepareVisualImportedSceneLayer(plans,8,640,360,use,&retainedExecution,&cache,
                      layer,owners,error)==VisualImportedScenePreparation::rendered,
                  "retained Geometry Core scene reaches production preview and export without media");
            check(retainedExecution.lastRequest.sceneSnapshot
                      && retainedExecution.lastRequest.sceneSnapshot->objectCount==6
                      && retainedExecution.lastRequest.sceneSnapshot->vertexCount==66
                      && retainedExecution.lastRequest.asset.id.empty(),
                  "production retains independent object materials and shared instance geometry");
        }
        const auto surfaces=retainedSurfaceFixture(error);
        check(surfaces.has_value(),"retained Surface execution fixture is canonical");
        if (surfaces) {
            request.surfacePrograms=surfaces->surfacePrograms;
            for (auto& object : request.surfacePrograms) {
                auto& material=object.program.material;
                material.sceneSnapshot=snapshot;
                material.sceneRevision=material.structuralRevision=material.programRevision=9;
                material.binding.surfaceMaterialRevision=9; material.evaluationRevision=12;
            }
            auto surfacePlan=retainedPlan;
            surfacePlan.operations[1].payloadXml=visualimportedscenerender::encode(request);
            for (const auto use : {NativeImportedSceneRenderUse::Preview,NativeImportedSceneRenderUse::Export})
                {
                    const bool conditionOk = prepareVisualImportedSceneLayer({surfacePlan},8,640,360,use,&retainedExecution,&cache,
                    layer,owners,error)==VisualImportedScenePreparation::rendered
                    && retainedExecution.lastRequest.surfacePrograms.size()==5
                    && retainedExecution.lastRequest.surfacePrograms.front().program.material.sceneSnapshot
                        ==retainedExecution.lastRequest.sceneSnapshot;
                    check(conditionOk, "production retained-scene preparation preserves exact Surface collection ownership in preview/export: "+error);
                }
        }
        auto compositePlan = retainedPlan;
        renderpasscomposite::Parameters compositeParameters;
        compositeParameters.mode = renderpasscomposite::Mode::ObjectMatte;
        compositeParameters.identity = 0xf1234567u;
        compositePlan.nodeIds.push_back(90);
        compositePlan.nodeKinds.push_back(renderpasscomposite::kNodeKind);
        compositePlan.operations.push_back({90, renderpasscomposite::kNodeKind, "native-gpu",
            renderpasscomposite::encode(compositeParameters)});
        const std::array<const char*, 7> types {"image", "depth", "normal", "emission", "mask", "materialId", "objectId"};
        const std::array<const char*, 7> formats {"rgba8", "r32f", "rgba16f", "rgba16f", "r8", "r32uint", "r32uint"};
        for (std::size_t i = 0; i < 7; ++i)
        {
            const auto space = i == 0 ? "sRGB" : i == 3 ? "linearSRGB" : "unspecified";
            const auto port = renderpasscomposite::renderPort(renderpasscomposite::kInputs[i]);
            if (i != 0) compositePlan.ports.push_back({11, port, 1, "out", "frame", types[i], formats[i], space});
            compositePlan.ports.push_back({90, static_cast<int>(i), 1, "in", "frame", types[i], formats[i], space});
            compositePlan.edges.push_back({11, port, 90, static_cast<int>(i)});
        }
        compositePlan.ports.push_back({90, 7, 1, "out", "frame", "image", "rgba8", "sRGB"});
        compositePlan.nodeIds.push_back(91); compositePlan.nodeKinds.push_back("video.out");
        compositePlan.operations.push_back({91, "video.out", "native-gpu", ""});
        compositePlan.ports.push_back({91, 0, 1, "in", "frame", "image", "rgba8", "sRGB"});
        compositePlan.edges.push_back({90, 7, 91, 0});
        retainedExecution.nextFrame->rawPasses = true;
        for (const auto use : {NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export})
            {
                const bool conditionOk = prepareVisualImportedSceneLayer({compositePlan}, 8, 640, 360, use, &retainedExecution,
                &cache, layer, owners, error) == VisualImportedScenePreparation::rendered
                && retainedExecution.lastRequest.runtimeInputs.passComposite
                && retainedExecution.lastRequest.runtimeInputs.passComposite->identity == 0xf1234567u;
                check(conditionOk, "retained generated scenes preserve full-width ID matte fan-out and Video Output in preview/export: " + error);
            }
        auto broken=retainedPlan;
        broken.edges[0].fromNodeId=5;
        check(prepareVisualImportedSceneLayer({broken},8,640,360,NativeImportedSceneRenderUse::Preview,
                  &retainedExecution,&cache,layer,owners,error)==VisualImportedScenePreparation::rejected,
              "retained scene rejects forged source identity");
        broken=retainedPlan;
        broken.operations[0].payloadXml="unexpected";
        check(prepareVisualImportedSceneLayer({broken},8,640,360,NativeImportedSceneRenderUse::Preview,
                  &retainedExecution,&cache,layer,owners,error)==VisualImportedScenePreparation::rejected,
              "retained scene rejects additional producer payload");
    }
    auto composedPlan = composedScenePlan();
    std::vector<videowire::CompiledVisualLayerPlan> composedPlans { composedPlan };
    FakeLayer composedLayer;
    std::vector<FakeReceipt> composedOwners;
    videohelper::importedscene::VisualImportedScenePlanCache composedPlanCache;
    check(prepareVisualImportedSceneLayer(
              composedPlans, 8, 640, 360, NativeImportedSceneRenderUse::Export,
              &execution, &composedPlanCache, composedLayer, composedOwners, error)
              == VisualImportedScenePreparation::rendered,
          "the production-shaped composed Scene3D V7 plan reaches export execution");
    check(execution.exportCalls == 1 && execution.lastRequest.sceneSnapshot
              && execution.lastRequest.sceneSnapshot->objectCount == 2
              && execution.lastRequest.sceneSnapshot->objects[1].firstVertex == 3
              && execution.lastRequest.sceneSnapshot->cameraCount == 1
              && execution.lastRequest.sceneSnapshot->lightCount == 1
              && execution.lastRequest.asset.id.empty(),
          "export receives the exact composed topology without an imported asset surrogate");
    const auto* firstComposedOwner = execution.lastRequest.sceneSnapshot.get();
    std::map<std::string, double> secondFrameParameters {
        { "visual12/objectX", 3.5 },
        { "visual12/objectRotationY", 45.0 },
        { "visual12/cameraZ", -2.0 },
        { "visual12/cameraRotateX", 15.0 },
        { "visual12/cameraFovDegrees", 50.0 },
        { "visual12/emissionGain", 4.0 }
    };
    check(prepareVisualImportedSceneLayerAtTime(
              composedPlans, 8, 640, 360, 1.0 / 24.0, 24.0,
              NativeImportedSceneRenderUse::Export, &execution, &composedPlanCache,
              composedLayer, composedOwners, error, &secondFrameParameters)
              == VisualImportedScenePreparation::rendered,
          "a second composed-scene frame evaluates through the cached production plan");
    check(execution.lastRequest.sceneSnapshot.get() == firstComposedOwner
              && execution.staticGpuUploads == 1
              && execution.lastRequest.runtimeInputs.objectTranslationOffset[0] == 3.5f
              && execution.lastRequest.runtimeInputs.objectRotationDegrees[1] == 45.0f
              && execution.lastRequest.runtimeInputs.cameraTranslationOffset[2] == -2.0f
              && execution.lastRequest.runtimeInputs.cameraRotationDegrees[0] == 15.0f
              && execution.lastRequest.runtimeInputs.cameraRotationDegrees[1] == 0.0f
              && execution.lastRequest.runtimeInputs.cameraVerticalFovDegrees == 50.0f
              && execution.lastRequest.runtimeInputs.emissionGain == 4.0f,
          "two frame evaluations reuse static GPU ownership while transforms and lights update");
    {
        using namespace HarmonicMIDI::grid;
        auto skyPlan=composedScenePlan();
        // Retain all owned scene edges, changing only the light authority to the
        // released environment port layout (one output, no transform input).
        const auto transform=std::find(skyPlan.nodeIds.begin(),skyPlan.nodeIds.end(),9);
        const auto transformIndex=static_cast<std::size_t>(transform-skyPlan.nodeIds.begin());
        skyPlan.nodeIds.erase(transform);
        skyPlan.nodeKinds.erase(skyPlan.nodeKinds.begin()+transformIndex);
        skyPlan.operations.erase(skyPlan.operations.begin()+transformIndex);
        for (std::size_t i=0;i<skyPlan.nodeIds.size();++i)
            if (skyPlan.nodeIds[i]==10) skyPlan.nodeKinds[i]=skyPlan.operations[i].kind="visual.3d.light.environment";
        skyPlan.ports.erase(std::remove_if(skyPlan.ports.begin(),skyPlan.ports.end(),[](const auto& port) {
            return port.nodeId==9 || (port.nodeId==10 && port.direction=="in");}),skyPlan.ports.end());
        for (auto& port:skyPlan.ports) if (port.nodeId==10) port.port=0;
        skyPlan.edges.erase(std::remove_if(skyPlan.edges.begin(),skyPlan.edges.end(),[](const auto& edge) {
            return edge.fromNodeId==9;}),skyPlan.edges.end());
        for (auto& edge:skyPlan.edges) if (edge.fromNodeId==10) edge.fromPort=0;
        visualimportedscenerender::Request skyRequest;
        check(visualimportedscenerender::decode(skyPlan.operations.back().payloadXml,skyRequest),"Existing composed fixture decodes");
        auto scene=std::make_shared<Visual3DScene>(*skyRequest.sceneSnapshot);
        scene->lights[0].id.value=11; scene->lights[0].kind=SceneLightKind::Environment;
        scene->lights[0].sky.version=1; scene->lights[0].sky.sunAzimuthDegrees=15;
        skyRequest.sceneSnapshot=scene;
        skyPlan.operations.back().payloadXml=visualimportedscenerender::encode(skyRequest);
        const std::vector<videowire::CompiledVisualLayerPlan> skyPlans {skyPlan};
        std::map<std::string,double> skyParameters;
        videohelper::importedscene::seedImportedSceneRuntimeParameters(skyPlans,8,skyParameters);
        check(skyParameters["visual11/sunAzimuthDegrees"]==15
            && skyParameters["visual11/sunRadiusDegrees"]==2,
            "Sky runtime controls seed from the exact authored environment identity");
        skyParameters["visual11/sunAzimuthDegrees"]=75;
        skyParameters["visual11/skyOrientationDegrees"]=-20;
        FakeExecution skyExecution; FakeLayer skyLayer; std::vector<FakeReceipt> skyOwners;
        videohelper::importedscene::VisualImportedScenePlanCache skyCache;
        const Visual3DScene* owner=nullptr;
        for (const auto use:{NativeImportedSceneRenderUse::Preview,NativeImportedSceneRenderUse::Export})
        {
            {
                const bool conditionOk = prepareVisualImportedSceneLayerAtTime(skyPlans,8,640,360,2.5,24,use,
                &skyExecution,&skyCache,skyLayer,skyOwners,error,&skyParameters)==VisualImportedScenePreparation::rendered;
                check(conditionOk, "Same-time authored sky reaches ordinary preview/export request: "+error);
            }
            const auto& request=skyExecution.lastRequest;
            check(request.frame.frame==60 && request.sceneSnapshot
                && request.sceneSnapshot->lights[0].sky.sunAzimuthDegrees==15
                && request.runtimeInputs.animatedLights.size()==1
                && request.runtimeInputs.animatedLights[0].sky.sunAzimuthDegrees==75
                && request.runtimeInputs.animatedLights[0].sky.orientationDegrees==-20,
                "Same rational frame evaluates sky controls without mutating immutable authored scene");
            if (owner) check(owner==request.sceneSnapshot.get(),"Sky preview/export retain the exact static scene owner");
            owner=request.sceneSnapshot.get();
        }
        check(skyExecution.staticGpuUploads==1,"Sky modulation reuses the existing GPU resource owner");
        skyParameters["visual11/sunAzimuthDegrees"]=std::numeric_limits<double>::quiet_NaN();
        check(prepareVisualImportedSceneLayerAtTime(skyPlans,8,640,360,2.5,24,NativeImportedSceneRenderUse::Preview,
            &skyExecution,&skyCache,skyLayer,skyOwners,error,&skyParameters)==VisualImportedScenePreparation::rejected,
            "Nonfinite animated sky control fails closed");
    }
    {
        auto imagePlan=composedPlan;
        const auto transform=std::find(imagePlan.nodeIds.begin(),imagePlan.nodeIds.end(),9);
        const auto transformIndex=static_cast<std::size_t>(transform-imagePlan.nodeIds.begin());
        imagePlan.nodeIds.erase(transform); imagePlan.nodeKinds.erase(imagePlan.nodeKinds.begin()+transformIndex);
        imagePlan.operations.erase(imagePlan.operations.begin()+transformIndex);
        for (std::size_t i=0;i<imagePlan.nodeIds.size();++i)
            if (imagePlan.nodeIds[i]==10) imagePlan.nodeKinds[i]=imagePlan.operations[i].kind="visual.3d.light.environment";
        imagePlan.ports.erase(std::remove_if(imagePlan.ports.begin(),imagePlan.ports.end(),[](const auto& port) {
            return port.nodeId==9 || (port.nodeId==10 && port.direction=="in");}),imagePlan.ports.end());
        for (auto& port:imagePlan.ports) if (port.nodeId==10) port.port=0;
        imagePlan.edges.erase(std::remove_if(imagePlan.edges.begin(),imagePlan.edges.end(),[](const auto& edge) {
            return edge.fromNodeId==9;}),imagePlan.edges.end());
        for (auto& edge:imagePlan.edges) if (edge.fromNodeId==10) edge.fromPort=0;
        visualimportedscenerender::Request authored;
        check(visualimportedscenerender::decode(imagePlan.operations.back().payloadXml,authored),"Composed HDR fixture decodes");
        auto scene=std::make_shared<HarmonicMIDI::grid::Visual3DScene>(*authored.sceneSnapshot);
        scene->lights[0].id.value=11; scene->lights[0].kind=HarmonicMIDI::grid::SceneLightKind::Environment; scene->lights[0].sky={};
        scene->lights[0].environmentImage={1,{"saved-hdr",7,std::string(64,'a')},30,2};
        authored.sceneSnapshot=scene;
        imagePlan.operations.back().payloadXml=visualimportedscenerender::encode(authored);
        const std::vector<videowire::CompiledVisualLayerPlan> imagePlans {imagePlan};
        const auto prefix="visual"+std::to_string(scene->lights[0].id.value)+"/";
        std::map<std::string,double> parameters;
        videohelper::importedscene::seedImportedSceneRuntimeParameters(imagePlans,8,parameters);
        check(parameters[prefix+"environmentOrientationDegrees"]==30 && parameters[prefix+"environmentExposureStops"]==2,
            "HDR controls seed from exact authored scene identity");
        parameters[prefix+"environmentOrientationDegrees"]=-75; parameters[prefix+"environmentExposureStops"]=1;
        FakeExecution imageExecution; FakeLayer layer; std::vector<FakeReceipt> owners;
        videohelper::importedscene::VisualImportedScenePlanCache cache;
        const HarmonicMIDI::grid::Visual3DScene* staticOwner=nullptr;
        for (const auto use:{NativeImportedSceneRenderUse::Preview,NativeImportedSceneRenderUse::Export})
        {
            {
                const bool conditionOk = prepareVisualImportedSceneLayerAtTime(imagePlans,8,640,360,2.5,24,use,
                &imageExecution,&cache,layer,owners,error,&parameters)==VisualImportedScenePreparation::rendered;
                check(conditionOk, "HDR image binding reaches the production preview/export route: "+error);
            }
            const auto& request=imageExecution.lastRequest;
            check(request.frame.frame==60 && request.sceneSnapshot && request.runtimeInputs.animatedLights.size()==1
                && request.sceneSnapshot->lights[0].environmentImage.orientationDegrees==30
                && request.sceneSnapshot->lights[0].environmentImage.exposureStops==2
                && request.runtimeInputs.animatedLights[0].environmentImage.orientationDegrees==-75
                && request.runtimeInputs.animatedLights[0].environmentImage.exposureStops==1
                && sameEnvironmentImage(request.sceneSnapshot->lights[0].environmentImage,scene->lights[0].environmentImage),
                "HDR controls evaluate at the exact rational frame without replacing immutable pixels or identity");
            if (staticOwner) check(staticOwner==request.sceneSnapshot.get(),"HDR preview/export retain one static scene owner");
            staticOwner=request.sceneSnapshot.get();
        }
        check(imageExecution.staticGpuUploads==1,"HDR controls reuse the native static resource owner");
        parameters[prefix+"environmentExposureStops"]=17;
        check(prepareVisualImportedSceneLayerAtTime(imagePlans,8,640,360,2.5,24,NativeImportedSceneRenderUse::Preview,
            &imageExecution,&cache,layer,owners,error,&parameters)==VisualImportedScenePreparation::rejected,
            "Out of bound animated HDR exposure refuses instead of clamping");
        parameters[prefix+"environmentExposureStops"]=std::numeric_limits<double>::quiet_NaN();
        check(prepareVisualImportedSceneLayerAtTime(imagePlans,8,640,360,2.5,24,NativeImportedSceneRenderUse::Export,
            &imageExecution,&cache,layer,owners,error,&parameters)==VisualImportedScenePreparation::rejected,
            "Nonfinite HDR export control refuses publication");
    }
    auto disconnectedComposedPlan = composedPlan;
    disconnectedComposedPlan.edges.erase(disconnectedComposedPlan.edges.begin());
    std::vector<videowire::CompiledVisualLayerPlan> disconnectedComposedPlans {
        disconnectedComposedPlan
    };
    check(prepareVisualImportedSceneLayer(
              disconnectedComposedPlans, 8, 640, 360,
              NativeImportedSceneRenderUse::Preview, &execution,
              &composedPlanCache, composedLayer, composedOwners, error)
              == VisualImportedScenePreparation::rejected,
          "the cache revalidates and rejects changed V7 topology at the same revision");
    auto mistypedComposedPlan = composedPlan;
    const auto cameraPrimitiveEdge = std::find_if(
        mistypedComposedPlan.edges.begin(), mistypedComposedPlan.edges.end(),
        [](const auto& edge)
        { return edge.fromNodeId == 8 && edge.toNodeId == 2; });
    check(cameraPrimitiveEdge != mistypedComposedPlan.edges.end(),
          "the production-shaped fixture contains the primitive camera edge");
    if (cameraPrimitiveEdge != mistypedComposedPlan.edges.end())
        cameraPrimitiveEdge->toPort = 2;
    std::vector<videowire::CompiledVisualLayerPlan> mistypedComposedPlans {
        mistypedComposedPlan
    };
    check(prepareVisualImportedSceneLayer(
              mistypedComposedPlans, 8, 640, 360,
              NativeImportedSceneRenderUse::Preview, &execution,
              composedLayer, composedOwners, error)
              == VisualImportedScenePreparation::rejected,
          "helper execution rejects a reachable but mistyped Scene3D edge");
    auto detachedRenderStatePlan = composedPlan;
    detachedRenderStatePlan.edges.erase(
        std::remove_if(detachedRenderStatePlan.edges.begin(),
                       detachedRenderStatePlan.edges.end(), [](const auto& edge)
                       { return edge.fromNodeId == 8 && edge.toNodeId == 11; }),
        detachedRenderStatePlan.edges.end());
    std::vector<videowire::CompiledVisualLayerPlan> detachedRenderStatePlans {
        detachedRenderStatePlan
    };
    check(prepareVisualImportedSceneLayer(
              detachedRenderStatePlans, 8, 640, 360,
              NativeImportedSceneRenderUse::Preview, &execution,
              composedLayer, composedOwners, error)
              == VisualImportedScenePreparation::rejected,
          "helper execution requires explicit composed-scene camera topology");

    std::map<std::string, double> staticRuntimeParameters;
    videohelper::importedscene::seedImportedSceneRuntimeParameters(
        plans, 7, staticRuntimeParameters);
    check(staticRuntimeParameters["visual72/objectRotationY"] == 0.0
              && staticRuntimeParameters["visual72/objectScale"] == 1.0,
          "static imported scenes seed Render 3D transform targets without deformation");

    auto selectedStaticPlans = plans;
    for (auto& operation : selectedStaticPlans.front().operations)
        if (operation.kind == visualimportedscenerender::kRenderNodeKind)
        {
            visualimportedscenerender::Request selected;
            check(visualimportedscenerender::decode(operation.payloadXml, selected), "static render request decodes");
            selected.objectNodeStableId = 5; selected.objectMeshStableId = 1;
            operation.payloadXml = visualimportedscenerender::encode(selected);
        }
    staticRuntimeParameters["visual72/objectX"] = 0.5;
    for (const auto use : {NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export})
    {
        FakeLayer selectedLayer; std::vector<FakeReceipt> selectedOwners;
        check(prepareVisualImportedSceneLayerAtTime(selectedStaticPlans, 7, 640, 360, 1.25, 24.0, use,
                  &execution, selectedLayer, selectedOwners, error, &staticRuntimeParameters)
                  == VisualImportedScenePreparation::rendered
              && !execution.lastRequest.deformation && execution.lastRequest.objectNodeStableId == 5
              && execution.lastRequest.objectMeshStableId == 1 && execution.lastRequest.frame.frame == 30
              && execution.lastRequest.runtimeInputs.objectTranslationOffset[0] == 0.5f,
              "static Preview and Export transport the same exact selected object and rational frame");
    }

    auto deformationPlan = importedScenePlan(false, true);
    std::vector<videowire::CompiledVisualLayerPlan> deformationPlans {
        deformationPlan
    };
    FakeLayer deformationLayer;
    std::vector<FakeReceipt> deformationOwners;
    std::map<std::string, double> runtimeParameters;
    videohelper::importedscene::seedImportedSceneRuntimeParameters(
        deformationPlans, 7, runtimeParameters);
    check(runtimeParameters["visual73/timelineSeconds"] == 1.0
              && runtimeParameters["visual73/speed"] == 1.5
              && runtimeParameters["visual73/trimStartSeconds"] == 0.25
              && runtimeParameters["visual73/trimEndSeconds"] == 0.75
              && runtimeParameters["visual73/weight"] == 0.75
              && runtimeParameters["visual72/objectX"] == 0.0
              && runtimeParameters["visual72/objectRotationY"] == 0.0
              && runtimeParameters["visual72/objectScale"] == 1.0
              && runtimeParameters["visual72/cameraZ"] == 0.0
              && runtimeParameters["visual72/cameraRotateX"] == 0.0
              && runtimeParameters["visual72/cameraRotateY"] == 0.0
              && runtimeParameters["visual72/cameraRotateZ"] == 0.0
              && runtimeParameters["visual72/cameraFovDegrees"] == 0.0
              && runtimeParameters["visual72/emissionGain"] == 1.0,
          "the immutable deformation values seed stable per-node runtime targets");
    runtimeParameters["visual73/timelineSeconds"] = 4.0;
    runtimeParameters["visual73/speed"] = 2.0;
    runtimeParameters["visual73/trimStartSeconds"] = 0.5;
    runtimeParameters["visual73/trimEndSeconds"] = 1.5;
    runtimeParameters["visual73/weight"] = 0.25;
    runtimeParameters["visual73/selectedMorphWeight"] = 0.7;
    runtimeParameters["visual73/boneTranslateX"] = 2.5;
    runtimeParameters["visual73/boneRotateY"] = 30.0;
    runtimeParameters["visual73/boneScaleZ"] = 1.5;
    runtimeParameters["visual72/objectX"] = 0.5;
    runtimeParameters["visual72/objectRotationY"] = 45.0;
    runtimeParameters["visual72/objectScale"] = 1.5;
    runtimeParameters["visual72/cameraZ"] = -0.25;
    runtimeParameters["visual72/cameraRotateY"] = 30.0;
    runtimeParameters["visual72/cameraRotateX"] = -400.0;
    runtimeParameters["visual72/cameraRotateZ"] = 500.0;
    runtimeParameters["visual72/cameraFovDegrees"] = 500.0;
    runtimeParameters["visual72/emissionGain"] = 2.0;
    check(prepareVisualImportedSceneLayerAtTime(
              deformationPlans, 7, 640, 360, 2.5, 24.0,
              NativeImportedSceneRenderUse::Preview,
              &execution, deformationLayer, deformationOwners, error,
              &runtimeParameters)
              == VisualImportedScenePreparation::rendered,
          "the exact Source -> Deformation -> Render 3D plan reaches preview");
    check(execution.lastRequest.deformation
              && execution.lastRequest.deformation->deformationStableId == 73
              && execution.lastRequest.frame.frame == 60
              && execution.lastRequest.frame.rateNumerator == 24
              && execution.lastRequest.frame.rateDenominator == 1
              && execution.lastRequest.structuralRevision == 9
              && execution.lastRequest.deformation->playback.timelineSeconds == 6.5
              && execution.lastRequest.deformation->playback.speed == 2.0
              && execution.lastRequest.deformation->playback.trimStartSeconds == 0.5
              && execution.lastRequest.deformation->playback.trimEndSeconds == 1.5
              && execution.lastRequest.deformation->playback.weight == 0.25
              && execution.lastRequest.deformation->pose.morphWeight == 0.7
              && execution.lastRequest.deformation->pose.translation[0] == 2.5
              && execution.lastRequest.deformation->pose.rotationDegrees[1] == 30.0
              && execution.lastRequest.deformation->pose.scale[2] == 1.5
              && execution.lastRequest.runtimeInputs.objectTranslationOffset[0] == 0.5f
              && execution.lastRequest.runtimeInputs.objectRotationDegrees[1] == 45.0f
              && execution.lastRequest.runtimeInputs.objectScale == 1.5f
              && execution.lastRequest.runtimeInputs.cameraTranslationOffset[2] == -0.25f
              && execution.lastRequest.runtimeInputs.emissionGain == 2.0f,
          "preview receives the immutable deformation payload and exact frame time");
    check(execution.lastRequest.runtimeInputs.cameraRotationDegrees[0] == -360.0f
              && execution.lastRequest.runtimeInputs.cameraRotationDegrees[1] == 30.0f
              && execution.lastRequest.runtimeInputs.cameraRotationDegrees[2] == 360.0f
              && execution.lastRequest.runtimeInputs.cameraVerticalFovDegrees == 179.0f,
          "modulated camera rotation reaches the request, clamped to the lowered ranges");

    for (const auto use : {NativeImportedSceneRenderUse::Preview,
                           NativeImportedSceneRenderUse::Export})
    {
        check(prepareVisualImportedSceneLayerAtTime(
                  deformationPlans, 7, 640, 360, 1.25, 24.0, use,
                  &execution, deformationLayer, deformationOwners, error,
                  &runtimeParameters) == VisualImportedScenePreparation::rendered
                  && execution.lastRequest.deformation
                  && execution.lastRequest.deformation->playback.timelineSeconds == 5.25,
              "preview and export seek to the rational frame plus authored offset without accumulation");
    }

    // selectedBoneEnabled gates the runtime bone values. The request always
    // carries them, but native evaluation applies them only to an enabled pose.
    const std::array<bool, 2> boneGateStates { false, true };
    for (const bool boneEnabled : boneGateStates)
    {
        auto gatedRequest = renderRequest(false, true);
        gatedRequest.deformation->meshStableId = 501;
        gatedRequest.deformation->pose.meshStableId = 501;
        gatedRequest.deformation->pose.boneStableId = 301;
        gatedRequest.deformation->pose.boneEnabled = boneEnabled;
        auto gatedPlan = importedScenePlan(false, true);
        for (auto& operation : gatedPlan.operations)
        {
            if (operation.kind == std::string(visualimportedscenerender::kRenderNodeKind))
                operation.payloadXml = visualimportedscenerender::encode(gatedRequest);
            else if (operation.kind == std::string(visualanimationoperation::kDeformationNodeKind))
                operation.payloadXml = visualanimationoperation::encode(*gatedRequest.deformation);
        }
        std::vector<videowire::CompiledVisualLayerPlan> gatedPlans { gatedPlan };
        std::map<std::string, double> gatedParameters;
        videohelper::importedscene::seedImportedSceneRuntimeParameters(
            gatedPlans, 7, gatedParameters);
        gatedParameters["visual73/boneRotateY"] = 30.0;
        FakeLayer gatedLayer;
        std::vector<FakeReceipt> gatedOwners;
        check(prepareVisualImportedSceneLayerAtTime(
                  gatedPlans, 7, 640, 360, 2.5, 24.0,
                  NativeImportedSceneRenderUse::Preview,
                  &execution, gatedLayer, gatedOwners, error,
                  &gatedParameters) == VisualImportedScenePreparation::rendered
              && execution.lastRequest.deformation
              && execution.lastRequest.deformation->pose.boneEnabled == boneEnabled
              && execution.lastRequest.deformation->pose.rotationDegrees[1] == 30.0,
              boneEnabled
                  ? "selectedBoneEnabled=1 hands the runtime boneRotateY to an enabled pose"
                  : "selectedBoneEnabled=0 hands native evaluation a disabled bone pose");
    }

    // Source seconds and project seconds are deliberately different. Beat Sync
    // follows the project tempo map at the rational project frame.
    auto beatPlan = deformationPlan;
    for (auto& operation : beatPlan.operations)
        if (operation.kind == visualimportedscenerender::kRenderNodeKind)
        {
            visualimportedscenerender::Request request;
            check(visualimportedscenerender::decode(operation.payloadXml, request), "beat plan render decodes");
            request.deformation->playback.timeSource = visualanimation::TimeSource::BeatSync;
            request.deformation->playback.timelineBeat = 0.5;
            request.deformation->playback.clipStartBeat = 1.0;
            request.deformation->playback.beatsPerLoop = 2.0;
            operation.payloadXml = visualimportedscenerender::encode(request);
        }
    visualimportedscenerender::Request beatRenderRequest;
    check(visualimportedscenerender::decode(beatPlan.operations.back().payloadXml, beatRenderRequest),
          "beat render owns the exact authored deformation");
    for (auto& operation : beatPlan.operations)
        if (operation.kind == visualanimationoperation::kDeformationNodeKind)
            operation.payloadXml = visualanimationoperation::encode(*beatRenderRequest.deformation);
    std::vector<videowire::CompiledVisualLayerPlan> beatPlans { beatPlan };
    videotime::BeatTimeline beatTimeline;
    beatTimeline.set({{0,60,true},{2,120,true}},{{0,4,4}});
    videohelper::importedscene::VisualImportedScenePlanCache beatCache;
    std::map<std::string,double> beatParameters;
    videohelper::importedscene::seedImportedSceneRuntimeParameters(beatPlans,7,beatParameters);
    check(beatParameters["visual73/timelineBeat"] == 0.5
              && beatParameters["visual73/clipStartBeat"] == 1.0
              && beatParameters["visual73/beatsPerLoop"] == 2.0,
          "Beat Sync seeds authored offsets and anchors without replacing graph values");
    for (const auto use : {NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export})
        for (const auto projectSeconds : {2.25, 0.5, 3.0, 0.5})
        {
            const auto result = videohelper::importedscene::prepareVisualImportedSceneLayerAtTime(
                beatPlans,7,640,360,7.0,24.0,use,&execution,&beatCache,
                deformationLayer,deformationOwners,error,&beatParameters,{},nullptr,false,&beatTimeline,projectSeconds);
            const auto expectedBeat = projectSeconds <= 2 ? projectSeconds : 2 + 2*(projectSeconds-2);
            check(result == VisualImportedScenePreparation::rendered
                      && execution.lastRequest.deformation
                      && execution.lastRequest.frame.frame == 168
                      && execution.lastRequest.deformation->playback.timelineBeat == expectedBeat+0.5
                      && execution.lastRequest.deformation->playback.clipStartBeat == 1.0
                      && execution.lastRequest.deformation->playback.beatsPerLoop == 2.0
                      && beatParameters["visual73/timelineBeat"] == 0.5,
                  "normal Scene3D Beat Sync follows project time, tempo steps and backward seeks without accumulation");
        }
    for (const auto invalid : {0.0, -1.0, 1000001.0, std::numeric_limits<double>::infinity()})
    {
        beatParameters["visual73/beatsPerLoop"] = invalid;
        const auto before = execution.previewCalls;
        check(videohelper::importedscene::prepareVisualImportedSceneLayerAtTime(
                  beatPlans,7,640,360,7.0,24.0,NativeImportedSceneRenderUse::Preview,&execution,&beatCache,
                  deformationLayer,deformationOwners,error,&beatParameters,{},nullptr,false,&beatTimeline,0.5)
                  == VisualImportedScenePreparation::rejected && execution.previewCalls == before,
              "invalid beat loop lengths refuse before native scene publication");
    }
    beatParameters["visual73/beatsPerLoop"] = 2.0;
    check(videohelper::importedscene::prepareVisualImportedSceneLayerAtTime(
              beatPlans,7,640,360,7.0,24.0,NativeImportedSceneRenderUse::Preview,&execution,&beatCache,
              deformationLayer,deformationOwners,error,&beatParameters)
              == VisualImportedScenePreparation::rejected,
          "Beat Sync cannot substitute a default tempo when the project timeline is missing");

    // Crossfade end to end: the v8 operation carries a second exact clip and
    // progress through the wire contract, the production consumer mixes the two
    // independently sampled actions, and invalid selections refuse.
    {
        const auto glb = crossfadeGlb(crossfadeRoot(), crossfadeBin());
        videohelper::Sha256 hash;
        hash.update(glb.data(), glb.size());
        visualanimationimport::Request base;
        base.sourceStableId = 41;
        base.deformationStableId = 42;
        base.schedule = { 41, 42 };
        base.asset = { "crossfade-plan-fixture", 1, hash.finishHex(),
                       "model/gltf-binary", glb.size() };
        base.animationClipStableId = 1;
        base.clipName = "Walk";
        base.meshStableId = 1;
        base.playback.timelineSeconds = 0.5;
        auto mixRequest = base;
        mixRequest.crossfadeClipStableId = 2;
        mixRequest.crossfadeClipName = "Reach";
        mixRequest.crossfadeProgress = 0.5;
        auto reachRequest = base;
        reachRequest.animationClipStableId = 2;
        reachRequest.clipName = "Reach";

        videohelper::ImportedAnimationDeformationConsumer consumer;
        std::string crossfadeError;
        check(consumer.admit(mixRequest, glb.data(), glb.size(), crossfadeError)
                  && crossfadeError.empty(),
              "the two-action crossfade GLB is admitted through the imported consumer");

        const auto encoded = visualanimationoperation::encode(mixRequest);
        visualanimationimport::Request reopened;
        check(encoded.rfind("visual.imported-animation.deformation.v8\n", 0) == 0
                  && visualanimationoperation::decode(encoded, reopened, crossfadeError)
                  && reopened.animationClipStableId == 1 && reopened.clipName == "Walk"
                  && reopened.crossfadeClipStableId == 2
                  && reopened.crossfadeClipName == "Reach"
                  && reopened.crossfadeProgress == 0.5
                  && visualanimationoperation::encode(reopened) == encoded,
              "crossfade operation round-trips both clip selectors and progress exactly");

        const std::string pair = "2 \"Reach\"";
        auto tamperedProgress = encoded;
        tamperedProgress.replace(tamperedProgress.rfind("0.5"), 3, "1.5");
        auto unpairedWire = encoded;
        unpairedWire.replace(unpairedWire.rfind(pair), pair.size(), "0 \"\"");
        visualanimationimport::Request refused;
        check(!visualanimationoperation::decode(tamperedProgress, refused, crossfadeError)
                  && !visualanimationoperation::decode(unpairedWire, refused, crossfadeError),
              "the wire decoder refuses progress outside [0,1] and progress without a paired clip");

        auto overProgress = mixRequest;
        overProgress.crossfadeProgress = 1.01;
        auto unpaired = mixRequest;
        unpaired.crossfadeClipStableId = 0;
        unpaired.crossfadeClipName.clear();
        auto unknown = mixRequest;
        unknown.crossfadeClipStableId = 3;
        check(visualanimationoperation::encode(overProgress).empty()
                  && visualanimationoperation::encode(unpaired).empty(),
              "the operation encoder refuses out-of-range progress and unpaired progress");

        std::array<float, 3> joint {};
        std::array<float, 2> morph {};
        check(!crossfadeEvaluate(consumer, overProgress, joint, morph, crossfadeError)
                  && !crossfadeEvaluate(consumer, unpaired, joint, morph, crossfadeError),
              "the consumer refuses out-of-range progress and progress without a paired clip");
        check(!crossfadeEvaluate(consumer, unknown, joint, morph, crossfadeError)
                  && crossfadeError == "imported animation/deformation clip selector is unavailable",
              "an unknown crossfade clip id is refused at evaluation with the exact selector diagnostic");
        videohelper::ImportedAnimationDeformationConsumer rejecting;
        check(!rejecting.admit(unknown, glb.data(), glb.size(), crossfadeError)
                  && crossfadeError == "imported animation/deformation clip selector is unavailable",
              "an unknown crossfade clip id is refused at admission with the exact selector diagnostic");

        std::array<float, 3> jointA {};
        std::array<float, 2> morphA {};
        std::array<float, 3> jointB {};
        std::array<float, 2> morphB {};
        check(crossfadeEvaluate(consumer, base, jointA, morphA, crossfadeError)
                  && nearCrossfadeValues(jointA, std::array<float, 3> { 0.5f, 1.0f, 1.5f })
                  && nearCrossfadeValues(morphA, std::array<float, 2> { 0.25f, 0.75f }),
              "clip A samples to its own joint and morph values at 0.5s");
        check(crossfadeEvaluate(consumer, reachRequest, jointB, morphB, crossfadeError)
                  && nearCrossfadeValues(jointB, std::array<float, 3> { 3.5f, 4.5f, 5.5f })
                  && nearCrossfadeValues(morphB, std::array<float, 2> { 0.65f, 0.35f }),
              "clip B samples to its own joint and morph values at 0.5s");
        for (const auto progress : { 0.0, 0.5, 1.0 })
        {
            mixRequest.crossfadeProgress = progress;
            std::array<float, 3> mixedJoint {};
            std::array<float, 2> mixedMorph {};
            const bool evaluated = crossfadeEvaluate(consumer, mixRequest, mixedJoint, mixedMorph,
                                                     crossfadeError);
            const auto weight = static_cast<float>(progress);
            std::array<float, 3> expectedJoint {};
            std::array<float, 2> expectedMorph {};
            for (std::size_t index = 0; index < 3; ++index)
                expectedJoint[index] = jointA[index] * (1.0f - weight) + jointB[index] * weight;
            for (std::size_t index = 0; index < 2; ++index)
                expectedMorph[index] = morphA[index] * (1.0f - weight) + morphB[index] * weight;
            check(evaluated && nearCrossfadeValues(mixedJoint, expectedJoint)
                      && nearCrossfadeValues(mixedMorph, expectedMorph),
                  "the crossfade deformation equals the progress-weighted mix of both clips at the same sample time");
            if (progress == 0.0)
                check(evaluated && nearCrossfadeValues(mixedJoint, jointA)
                          && nearCrossfadeValues(mixedMorph, morphA),
                      "crossfade progress 0 reproduces clip A exactly");
            if (progress == 1.0)
                check(evaluated && nearCrossfadeValues(mixedJoint, jointB)
                          && nearCrossfadeValues(mixedMorph, morphB),
                      "crossfade progress 1 reproduces clip B exactly");
            if (progress == 0.5)
                check(evaluated && nearCrossfadeValues(mixedJoint, std::array<float, 3> { 2.0f, 2.75f, 3.5f })
                          && nearCrossfadeValues(mixedMorph, std::array<float, 2> { 0.45f, 0.55f }),
                      "crossfade progress 0.5 is the midpoint of clip A and clip B");
        }

        // Graph lowering: the render node's deformation payload carries the same
        // selectors and progress into the request the plan execution receives.
        auto crossfadePlan = deformationPlan;
        for (auto& operation : crossfadePlan.operations)
            if (operation.kind == visualimportedscenerender::kRenderNodeKind)
            {
                visualimportedscenerender::Request request;
                check(visualimportedscenerender::decode(operation.payloadXml, request),
                      "crossfade render operation decodes");
                request.deformation->crossfadeClipStableId = 2;
                request.deformation->crossfadeClipName = "Reach";
                request.deformation->crossfadeProgress = 0.5;
                operation.payloadXml = visualimportedscenerender::encode(request);
            }
        visualimportedscenerender::Request crossfadeRender;
        check(visualimportedscenerender::decode(crossfadePlan.operations.back().payloadXml,
                                                crossfadeRender),
              "crossfade render owns the exact modified deformation");
        for (auto& operation : crossfadePlan.operations)
            if (operation.kind == visualanimationoperation::kDeformationNodeKind)
                operation.payloadXml = visualanimationoperation::encode(*crossfadeRender.deformation);
        std::vector<videowire::CompiledVisualLayerPlan> crossfadePlans { crossfadePlan };
        check(prepareVisualImportedSceneLayerAtTime(
                  crossfadePlans, 7, 640, 360, 2.5, 24.0,
                  NativeImportedSceneRenderUse::Preview,
                  &execution, deformationLayer, deformationOwners, error)
                  == VisualImportedScenePreparation::rendered
              && execution.lastRequest.deformation
              && execution.lastRequest.deformation->crossfadeClipStableId == 2
              && execution.lastRequest.deformation->crossfadeClipName == "Reach"
              && execution.lastRequest.deformation->crossfadeProgress == 0.5,
              "the plan-executed render request carries the crossfade clip and progress");
    }

    std::vector<videowire::CompiledVisualLayerPlan> materialDeformationPlans {
        importedScenePlan(true, true)
    };
    check(prepareVisualImportedSceneLayerAtTime(
              materialDeformationPlans, 7, 640, 360, 2.5, 24.0,
              NativeImportedSceneRenderUse::Preview,
              &execution, deformationLayer, deformationOwners, error)
              == VisualImportedScenePreparation::rendered,
          "the ordinary graph admits deformation and Surface Material together");
    check(execution.lastRequest.deformation
              && execution.lastRequest.material
              && execution.lastRequest.material->evaluationRevision == 33,
          "the shared request carries both immutable deformation and material payloads");

    auto tamperedDeformationPlan = deformationPlan;
    tamperedDeformationPlan.edges.pop_back();
    std::vector<videowire::CompiledVisualLayerPlan> tamperedDeformationPlans {
        tamperedDeformationPlan
    };
    check(prepareVisualImportedSceneLayerAtTime(
              tamperedDeformationPlans, 7, 640, 360, 2.5, 24.0,
              NativeImportedSceneRenderUse::Preview,
              &execution, deformationLayer, deformationOwners, error)
              == VisualImportedScenePreparation::rejected,
          "the deformation route rejects a missing exact graph edge");

    for (const auto hostile : { arbitgpu::NativeTexturePixelFormat::Rgba8Unorm,
                                arbitgpu::NativeTexturePixelFormat::Invalid })
    {
        execution.nextFrame = std::make_shared<FakeFrame>();
        execution.nextFrame->depthFormat = hostile;
        FakeLayer hostileLayer;
        std::vector<FakeReceipt> hostileOwners;
        check(prepareVisualImportedSceneLayer(
                  plans, 7, 640, 360, NativeImportedSceneRenderUse::Preview,
                  &execution, hostileLayer, hostileOwners, error)
                  == VisualImportedScenePreparation::rejected
                  && hostileOwners.empty(),
              "wrong native depth formats reject before ownership publication");
    }
    execution.nextFrame = std::make_shared<FakeFrame>();
    execution.nextFrame->descriptorWidth = 639;
    FakeLayer hostileDimensions;
    std::vector<FakeReceipt> hostileDimensionOwners;
    check(prepareVisualImportedSceneLayer(
              plans, 7, 640, 360, NativeImportedSceneRenderUse::Preview,
              &execution, hostileDimensions, hostileDimensionOwners, error)
              == VisualImportedScenePreparation::rejected,
          "mismatched native descriptor dimensions reject before composition");
    execution.nextFrame = std::make_shared<FakeFrame>();
    execution.nextFrame->descriptorSampleCount = 4;
    FakeLayer hostileSamples;
    std::vector<FakeReceipt> hostileSampleOwners;
    check(prepareVisualImportedSceneLayer(
              plans, 7, 640, 360, NativeImportedSceneRenderUse::Preview,
              &execution, hostileSamples, hostileSampleOwners, error)
              == VisualImportedScenePreparation::rejected,
          "multisampled native descriptors reject before composition");

    execution.nextFrame = std::make_shared<FakeFrame>();
    execution.nextFrame->api = "metal";
    execution.nextFrame->view = 0x12345678u;
    FakeLayer exportLayer;
    std::vector<FakeReceipt> exportOwners;
    const auto exportCallsBefore = execution.exportCalls;
    const auto previewCallsBefore = execution.previewCalls;
    const auto previewFrameBefore = previewOwners.empty() ? nullptr : previewOwners.front().frame;
    check(prepareVisualImportedSceneLayer(
              plans, 7, 640, 360, NativeImportedSceneRenderUse::Export,
              &execution, exportLayer, exportOwners, error)
              == VisualImportedScenePreparation::rendered,
          "the same exact plan renders for export");
    check(execution.exportCalls == exportCallsBefore + 1
              && execution.previewCalls == previewCallsBefore
              && previewOwners.size() == 1 && previewOwners.front().owner == 1
              && previewOwners.front().frame == previewFrameBefore
              && exportOwners.size() == 1 && exportOwners.front().owner == 2
              && exportOwners.front().frame != previewFrameBefore,
          "preview and export retain independent owners");
    check(exportLayer.texture == 0 && exportLayer.nativeTextureBackend == "metal"
              && exportLayer.nativeTextureView == 0x12345678u,
          "Metal preserves native texture view identity without narrowing");
    check(exportLayer.depthTexture == 0
              && exportLayer.nativeDepthTextureBackend == "metal"
              && exportLayer.nativeDepthTextureView == 66
              && exportLayer.depthWidth == 640 && exportLayer.depthHeight == 360,
          "Metal preserves the native R32F depth view without readback or narrowing");

    execution.nextFrame = std::make_shared<FakeFrame>();
    FakeLayer materialLayer;
    std::vector<FakeReceipt> materialOwners;
    std::vector<videowire::CompiledVisualLayerPlan> materialPlans {
        importedScenePlan(true)
    };
    check(prepareVisualImportedSceneLayer(
              materialPlans, 7, 640, 360,
              NativeImportedSceneRenderUse::Preview,
              &execution, materialLayer, materialOwners, error)
              == VisualImportedScenePreparation::rendered
              && execution.lastRequest.material
              && execution.lastRequest.material->sceneSnapshot == nullptr
              && visualimportedscenerender::sameMaterialRequest(
                  *exactMaterialRequest.material,
                  *execution.lastRequest.material),
          "the decoded material value reaches payload execution without a scene owner");

    auto unequalCrossedBlazedRequest = renderRequest(false, false, true);
    {
        const auto vertexPlan = importedVertexPlan();
        FakeLayer vertexLayer;
        vertexLayer.audioPresent = true;
        vertexLayer.audioFeatures.bands.assign(64, 0.0f);
        vertexLayer.audioFeatures.bands[6] = 0.8f;
        std::vector<FakeReceipt> vertexOwners;
        videohelper::importedscene::VisualImportedScenePlanCache vertexCache;
        const auto previewResult = prepareVisualImportedSceneLayerAtTime(
            {vertexPlan}, 7, 640, 360, 0.5, 60.0, NativeImportedSceneRenderUse::Preview,
            &execution, &vertexCache, vertexLayer, vertexOwners, error);
        check(previewResult == VisualImportedScenePreparation::rendered
              && execution.lastRequest.material && execution.lastRequest.material->vertexModifier
              && execution.lastRequest.runtimeInputs.vertexSpectrum[6] == 0.8f
              && execution.lastRequest.runtimeInputs.timeSeconds == 0.5f,
              "authored vertex topology and the current audio frame reach native preview: " + error);
        const auto previewRequest = execution.lastRequest;
        check(prepareVisualImportedSceneLayerAtTime(
            {vertexPlan}, 7, 640, 360, 0.5, 60.0, NativeImportedSceneRenderUse::Export,
            &execution, &vertexCache, vertexLayer, vertexOwners, error)
                == VisualImportedScenePreparation::rendered
              && execution.lastRequest.runtimeInputs.vertexSpectrum == previewRequest.runtimeInputs.vertexSpectrum
              && visualimportedscenerender::sameMaterialRequest(*execution.lastRequest.material, *previewRequest.material),
              "export retains the same vertex graph, frame time and canonical audio samples");
        vertexLayer.audioFeatures.bands.resize(63);
        check(prepareVisualImportedSceneLayerAtTime(
            {vertexPlan}, 7, 640, 360, 0.5, 60.0, NativeImportedSceneRenderUse::Preview,
            &execution, &vertexCache, vertexLayer, vertexOwners, error)
                == VisualImportedScenePreparation::rejected
              && error.find("64-band") != std::string::npos,
              "a truncated spectrum frame cannot silently change the band's meaning");
        vertexLayer.audioPresent = false;
        check(prepareVisualImportedSceneLayerAtTime(
            {vertexPlan}, 7, 640, 360, 0.5, 60.0, NativeImportedSceneRenderUse::Preview,
            &execution, &vertexCache, vertexLayer, vertexOwners, error)
                == VisualImportedScenePreparation::rendered
              && std::all_of(execution.lastRequest.runtimeInputs.vertexSpectrum.begin(),
                             execution.lastRequest.runtimeInputs.vertexSpectrum.end(),
                             [](float value) { return value == 0.0f; }),
              "missing audio keeps the imported vertex graph silent");
        auto motionVertexPlan = vertexPlan;
        for (auto& operation : motionVertexPlan.operations)
            if (operation.kind == visualimportedscenerender::kRenderNodeKind)
            {
                visualimportedscenerender::Request motionRequest;
                check(visualimportedscenerender::decode(operation.payloadXml, motionRequest),
                    "vertex motion fixture decodes");
                motionRequest.imageOutput = renderpassoutput::Output::Motion;
                operation.payloadXml = visualimportedscenerender::encode(motionRequest);
            }
        for (const auto use : {NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export})
            {
                const bool conditionOk = prepareVisualImportedSceneLayerAtTime(
                {motionVertexPlan}, 7, 640, 360, 0.5, 60.0, use,
                &execution, &vertexCache, vertexLayer, vertexOwners, error)
                    == VisualImportedScenePreparation::rendered
                && execution.lastRequest.runtimeInputs.imageOutput == renderpassoutput::Output::Motion
                && execution.lastRequest.material && execution.lastRequest.material->vertexModifier
                && !execution.lastRequest.deformedSceneMotion;
                check(conditionOk, "Motion selection reaches preview and export for a vertex program over static geometry: " + error);
            }
        auto mismatched = vertexPlan;
        mismatched.edges.back().fromPort = 0;
        videowire::VisualLayerExecution rejected;
        check(!videowire::compileVisualLayerExecution(mismatched, rejected, error),
              "vertex topology rejects a material edge from the wrong typed output");
        mismatched = vertexPlan;
        for (auto& operation : mismatched.operations)
            if (operation.nodeId == 2) operation.kind = "visual.vertex.control-parameter";
        check(!videowire::compileVisualLayerExecution(mismatched, rejected, error),
              "vertex topology rejects a node kind that disagrees with its immutable IR");
    }
    auto& unequalCrossedBlazed
        = unequalCrossedBlazedRequest.diffractionMaterial->material;
    unequalCrossedBlazed = diffractionmaterial::makeBlazedGratingPreset();
    unequalCrossedBlazed.geometry.lattice
        = diffractionmaterial::GratingLattice::CrossedTwoDimensional;
    unequalCrossedBlazed.geometry.secondaryDirectionUv = { 0.0f, 1.0f };
    unequalCrossedBlazed.geometry.secondaryGrooveSpacingNanometres
        = unequalCrossedBlazed.geometry.grooveSpacingNanometres + 1.0f;
    check(visualimportedscenerender::encode(unequalCrossedBlazedRequest).empty(),
          "the imported-scene wire validator rejects unequal crossed blazed periods");

    auto currentDiffractionRequest = renderRequest(false, false, true);
    currentDiffractionRequest.diffractionMaterial = spatialDiffractionMaterialRequest();
    const auto canonicalLighting = diffractionmaterial::makeReferenceLighting();
    std::string currentFoilAdmissionError;
    const auto admittedCurrentFoil = diffractivefoil::admit(
        *currentDiffractionRequest.diffractionMaterial->spatialFoil,
        currentFoilAdmissionError);
    std::string currentLightingAdmissionError;
    const auto admittedCurrentLighting = diffractionmaterial::AdmittedLightingPlan::admit(
        currentDiffractionRequest.diffractionMaterial->lighting,
        currentLightingAdmissionError);
    std::string currentBindingAdmissionError;
    const auto admittedCurrentBinding = diffractionmaterialbinding::admit(
        *currentDiffractionRequest.diffractionMaterial,
        currentBindingAdmissionError);
    check(admittedCurrentFoil.has_value(),
          "production admission accepts the current V7 spatial foil before encoding: "
              + currentFoilAdmissionError);
    {
        const bool conditionOk = admittedCurrentLighting.has_value()
              && admittedCurrentLighting->hasIndirectBounce()
              && diffractionmaterial::sameLightingDescription(
                  admittedCurrentLighting->description(), canonicalLighting);
        check(conditionOk, "production admission accepts the canonical three-path V7 diffraction lighting before encoding: "
              + currentLightingAdmissionError);
    }
    {
        const bool conditionOk = admittedCurrentBinding.has_value()
              && admittedCurrentBinding->spatialFoil().has_value()
              && diffractionmaterial::sameLightingDescription(
                  admittedCurrentBinding->lighting().description(), canonicalLighting);
        check(conditionOk, "binding admission owns the current V7 spatial foil before encoding: "
              + currentBindingAdmissionError);
    }
    auto noncanonicalLightingRequest = currentDiffractionRequest;
    noncanonicalLightingRequest.diffractionMaterial->lighting.paths[0]
        .incident.radiance[0] = 0.64f;
    std::string noncanonicalLightingError;
    check(visualimportedscenerender::encode(noncanonicalLightingRequest).empty()
              && !diffractionmaterialbinding::admit(
                  *noncanonicalLightingRequest.diffractionMaterial,
                  noncanonicalLightingError)
              && noncanonicalLightingError.find("version-defined canonical plan")
                  != std::string::npos,
          "V7 rejects valid but noncanonical caller lighting instead of hiding it behind version-defined restoration");
    auto malformedTrailingLightingRequest = currentDiffractionRequest;
    malformedTrailingLightingRequest.diffractionMaterial->lighting.paths.back().kind
        = diffractionmaterial::LightingPathKind::Direct;
    std::string malformedTrailingLightingError;
    check(visualimportedscenerender::encode(malformedTrailingLightingRequest).empty()
              && !diffractionmaterialbinding::admit(
                  *malformedTrailingLightingRequest.diffractionMaterial,
                  malformedTrailingLightingError)
              && malformedTrailingLightingError.find("noncanonical trailing paths")
                  != std::string::npos,
          "V7 rejects noncanonical unused lighting paths before wire encoding or binding admission");
    const auto currentDiffraction
        = visualimportedscenerender::encode(currentDiffractionRequest);
    visualimportedscenerender::Request decodedCurrentDiffraction;
    const bool decodedCurrent = currentDiffraction.rfind(
              std::string(visualimportedscenerender::kCompiledOperationHeaderV7) + "\n",
              0) == 0
              && visualimportedscenerender::decode(
                  currentDiffraction, decodedCurrentDiffraction);
    check(decodedCurrent, "the current diffraction producer decodes canonical V7 bytes");
    if (decodedCurrent)
    {
        const auto& expected = *currentDiffractionRequest.diffractionMaterial;
        const auto& actual = *decodedCurrentDiffraction.diffractionMaterial;
        check(currentDiffractionRequest.sourceStableId
                  == decodedCurrentDiffraction.sourceStableId
                  && currentDiffractionRequest.renderStableId
                      == decodedCurrentDiffraction.renderStableId
                  && currentDiffractionRequest.asset.id
                      == decodedCurrentDiffraction.asset.id
                  && currentDiffractionRequest.asset.version
                      == decodedCurrentDiffraction.asset.version
                  && currentDiffractionRequest.asset.contentSha256
                      == decodedCurrentDiffraction.asset.contentSha256
                  && currentDiffractionRequest.asset.sourceMediaType
                      == decodedCurrentDiffraction.asset.sourceMediaType
                  && currentDiffractionRequest.asset.sourceByteSize
                      == decodedCurrentDiffraction.asset.sourceByteSize
                  && currentDiffractionRequest.sceneIndex
                      == decodedCurrentDiffraction.sceneIndex
                  && expected.version == actual.version
                  && expected.scene == actual.scene
                  && expected.sceneRevision == actual.sceneRevision
                  && expected.structuralRevision == actual.structuralRevision
                  && expected.evaluationRevision == actual.evaluationRevision
                  && expected.materialRevision == actual.materialRevision
                  && expected.object == actual.object
                  && expected.structuralDigest == actual.structuralDigest
                  && visualimportedscenerender::detail::sameDiffractionDescription(
                      expected.material, actual.material)
                  && diffractionmaterial::sameLightingDescription(
                      expected.lighting, actual.lighting)
                  && diffractionmaterial::sameLightingDescription(
                      actual.lighting, canonicalLighting),
              "V7 preserves imported-scene identities, every diffraction material field, and every bit of the implicit canonical lighting plan");

        const auto& expectedFoil = *expected.spatialFoil;
        const auto& actualFoil = *actual.spatialFoil;
        bool sameFoilField = expectedFoil.version == actualFoil.version
            && expectedFoil.workBudget.maximumEvaluations
                == actualFoil.workBudget.maximumEvaluations
            && visualimportedscenerender::detail::sameDiffractionDescription(
                expectedFoil.physicalBsdf, actualFoil.physicalBsdf);
        for (std::size_t index = 0; index < expectedFoil.grooveField.size(); ++index)
        {
            const auto& expectedSample = expectedFoil.grooveField[index];
            const auto& actualSample = actualFoil.grooveField[index];
            sameFoilField = sameFoilField
                && visualimportedscenerender::detail::sameFloatArray(
                    expectedSample.reciprocalDirectionUv,
                    actualSample.reciprocalDirectionUv)
                && visualimportedscenerender::detail::floatBits(
                    expectedSample.grooveSpacingNanometres)
                    == visualimportedscenerender::detail::floatBits(
                        actualSample.grooveSpacingNanometres)
                && visualimportedscenerender::detail::floatBits(
                    expectedSample.diffractionCoverage)
                    == visualimportedscenerender::detail::floatBits(
                        actualSample.diffractionCoverage);
        }
        check(sameFoilField,
              "V7 preserves the foil version, immutable ceiling, physical BSDF, and all four ordered field samples");

        constexpr std::uint32_t graphAuthoredWidth = 4096;
        constexpr std::uint32_t graphAuthoredHeight = 4096;
        constexpr std::uint32_t graphAuthoredPathCount = 4;
        diffractivefoil::EvaluationSchedule schedule;
        check(diffractivefoil::makeEvaluationSchedule(
                  actualFoil.workBudget.maximumEvaluations,
                  graphAuthoredWidth, graphAuthoredHeight,
                  graphAuthoredPathCount, schedule)
                  && schedule.width == graphAuthoredWidth
                  && schedule.height == graphAuthoredHeight
                  && schedule.pathCount == graphAuthoredPathCount
                  && schedule.maximumEvaluations
                      == diffractivefoil::kMaximumEvaluations
                  && schedule.requiredEvaluations
                      == diffractivefoil::kMaximumEvaluations,
              "the decoded graph-authored ceiling admits the exact width, height, and path boundary schedule");
        check(visualimportedscenerender::sameRequest(
                  currentDiffractionRequest, decodedCurrentDiffraction)
                  && visualimportedscenerender::encode(decodedCurrentDiffraction)
                      == currentDiffraction,
              "the full V7 diffraction-plus-foil value round-trips to identical canonical bytes");
    }


    auto legacyV6Request = renderRequest(false, false, true);
    legacyV6Request.diffractionMaterial->version
        = visualimportedscenerender::detail::kLegacyV6BindingWireVersion;
    legacyV6Request.diffractionMaterial->material.version
        = visualimportedscenerender::detail::kLegacyV6MaterialWireVersion;
    legacyV6Request.diffractionMaterial->structuralDigest = std::string(64, 'd');
    auto noncanonicalLegacyLightingRequest = legacyV6Request;
    noncanonicalLegacyLightingRequest.diffractionMaterial->lighting.paths[1]
        .incident.radiance[7] = 0.15f;
    check(visualimportedscenerender::encodeLegacyV6(
              noncanonicalLegacyLightingRequest).empty(),
          "V6 rejects noncanonical caller lighting instead of silently restoring over it");
    const auto legacyV6 = visualimportedscenerender::encodeLegacyV6(legacyV6Request);
    visualimportedscenerender::Request migratedV6;
    check(!legacyV6.empty()
              && !visualimportedscenerender::decode(legacyV6, migratedV6)
              && visualimportedscenerender::decode(
                  legacyV6, migratedV6,
                  diffractionmaterialbinding::migrateLegacyV6)
              && migratedV6.diffractionMaterial
              && migratedV6.diffractionMaterial->version
                  == diffractionmaterialbinding::kWireVersion
              && migratedV6.diffractionMaterial->material.version
                  == diffractionmaterial::kWireVersion
              && migratedV6.diffractionMaterial->material.grooveField.mode
                  == diffractionmaterial::GrooveFieldMode::Constant
              && migratedV6.diffractionMaterial->material.grooveField.originUv
                  == std::array<float, 2> {}
              && migratedV6.diffractionMaterial->structuralDigest
                  == diffractionMaterialRequest().structuralDigest
              && diffractionmaterial::sameLightingDescription(
                  migratedV6.diffractionMaterial->lighting, canonicalLighting),
          "V6 fieldless diffraction decodes only through owning admission and migrates to canonical V7 identity");
    const auto legacyFixture = [](std::uint32_t sourceId)
    {
        auto request = renderRequest(false, false, true);
        request.sourceStableId = sourceId;
        request.diffractionMaterial->version
            = visualimportedscenerender::detail::kLegacyV6BindingWireVersion;
        request.diffractionMaterial->material.version
            = visualimportedscenerender::detail::kLegacyV6MaterialWireVersion;
        request.diffractionMaterial->structuralDigest = std::string(64, 'd');
        return visualimportedscenerender::encodeLegacyV6(request);
    };
    const auto currentFixture = [](std::uint32_t sourceId)
    {
        auto request = renderRequest(false, false, true);
        request.sourceStableId = sourceId;
        return visualimportedscenerender::encode(request);
    };
    const auto compositionFixture = [&boundLight](std::uint32_t sourceId)
    {
        auto request = renderRequest(true);
        request.sourceStableId = sourceId;
        request.light = boundLight;
        return visualimportedscenerender::encode(request);
    };

    // V6 and V7 carry three flags: deformation, mandatory diffraction, camera.
    // V8 carries six. A canonical V6 body always presents 0 1 0 1 to V8,
    // which violates V8's material/diffraction exclusion. A canonical V7 body
    // presents binding version 3 as V8's fourth flag. Those two directions
    // therefore cannot pass V8 flag admission. The other four collisions enter
    // the destination diffraction grammar before its version-specific fields
    // reject the shifted source tokens.
    std::array<DirectionalRelabelFixture, 6> crossVersionFixtures {{
        { "V6-to-V7", visualimportedscenerender::kCompiledOperationHeaderV6,
          visualimportedscenerender::kCompiledOperationHeaderV7, legacyFixture(601), {},
          CrossVersionAdmissionStage::diffractionSyntax },
        { "V7-to-V6", visualimportedscenerender::kCompiledOperationHeaderV7,
          visualimportedscenerender::kCompiledOperationHeaderV6, currentFixture(701), {},
          CrossVersionAdmissionStage::diffractionSyntax },
        { "V6-to-V8", visualimportedscenerender::kCompiledOperationHeaderV6,
          visualimportedscenerender::kCompiledOperationHeaderV8, legacyFixture(608), {},
          CrossVersionAdmissionStage::flagSchema },
        { "V8-to-V6", visualimportedscenerender::kCompiledOperationHeaderV8,
          visualimportedscenerender::kCompiledOperationHeaderV6, compositionFixture(806), {},
          CrossVersionAdmissionStage::diffractionSyntax },
        { "V7-to-V8", visualimportedscenerender::kCompiledOperationHeaderV7,
          visualimportedscenerender::kCompiledOperationHeaderV8, currentFixture(708), {},
          CrossVersionAdmissionStage::flagSchema },
        { "V8-to-V7", visualimportedscenerender::kCompiledOperationHeaderV8,
          visualimportedscenerender::kCompiledOperationHeaderV7, compositionFixture(807), {},
          CrossVersionAdmissionStage::diffractionSyntax }
    }};

    std::set<std::string> sourceBodies;
    for (auto& fixture : crossVersionFixtures)
    {
        fixture.relabeledBytes = relabelHeaderOnly(
            fixture.sourceBytes, fixture.destinationHeader);
        visualimportedscenerender::Request canonicalSource;
        const bool sourceDecoded = fixture.sourceHeader
                == visualimportedscenerender::kCompiledOperationHeaderV6
            ? visualimportedscenerender::decode(
                fixture.sourceBytes, canonicalSource,
                diffractionmaterialbinding::migrateLegacyV6)
            : visualimportedscenerender::decode(fixture.sourceBytes, canonicalSource);
        const auto sourceReencoded = fixture.sourceHeader
                == visualimportedscenerender::kCompiledOperationHeaderV6
            ? visualimportedscenerender::encodeLegacyV6([&]
              {
                  auto source = canonicalSource;
                  source.diffractionMaterial->version
                      = visualimportedscenerender::detail::kLegacyV6BindingWireVersion;
                  source.diffractionMaterial->material.version
                      = visualimportedscenerender::detail::kLegacyV6MaterialWireVersion;
                  source.diffractionMaterial->material.grooveField = {};
                  source.diffractionMaterial->structuralDigest = std::string(64, 'd');
                  return source;
              }())
            : visualimportedscenerender::encode(canonicalSource);
        check(sourceDecoded && !fixture.sourceBytes.empty()
                  && sourceReencoded == fixture.sourceBytes,
              std::string(fixture.identity) + " starts from source-canonical bytes");
        {
            const bool conditionOk = relabelHeaderOnly(fixture.relabeledBytes, fixture.sourceHeader)
                  == fixture.sourceBytes;
            check(conditionOk, std::string(fixture.identity) + " changes only the version header");
        }
        {
            const bool conditionOk = probeCrossVersionAdmission(
                  fixture.relabeledBytes, fixture.destinationHeader)
                  == fixture.expectedStage;
            check(conditionOk, std::string(fixture.identity)
                  + " reaches its exact wrong-schema admission boundary");
        }
        const bool rejected = fixture.destinationHeader
                == visualimportedscenerender::kCompiledOperationHeaderV6
            ? !visualimportedscenerender::decode(
                fixture.relabeledBytes, migratedV6,
                diffractionmaterialbinding::migrateLegacyV6)
            : !visualimportedscenerender::decode(fixture.relabeledBytes, migratedV6);
        check(rejected,
              std::string(fixture.identity)
                  + " is rejected by destination schema authority");
        sourceBodies.insert(fixture.sourceBytes.substr(
            fixture.sourceBytes.find('\n') + 1));
    }
    check(sourceBodies.size() == crossVersionFixtures.size(),
          "all six directional fixtures have byte-distinct source-shaped bodies");
    const auto malformedLegacyV6 = replaceLine(
        legacyV6, 9,
        "2 1 3 9 33 9 23 \"" + std::string(64, 'd') + "\"");
    check(!visualimportedscenerender::decode(
              malformedLegacyV6, migratedV6,
              diffractionmaterialbinding::migrateLegacyV6),
          "the frozen V6 decoder rejects a non-V1 legacy binding schema");
    auto legacyV6Plan = importedScenePlan(false, false, true);
    legacyV6Plan.operations.back().payloadXml = legacyV6;
    std::vector<videowire::CompiledVisualLayerPlan> legacyV6Plans {
        std::move(legacyV6Plan)
    };
    {
        const bool conditionOk = prepareVisualImportedSceneLayer(
              legacyV6Plans, 7, 640, 360,
              NativeImportedSceneRenderUse::Preview,
              &execution, materialLayer, materialOwners, error)
              == VisualImportedScenePreparation::rendered
              && execution.lastRequest.diffractionMaterial
              && execution.lastRequest.diffractionMaterial->structuralDigest
                  == diffractionMaterialRequest().structuralDigest
              && diffractionmaterial::sameLightingDescription(
                  execution.lastRequest.diffractionMaterial->lighting,
                  canonicalLighting);
        check(conditionOk, "the imported-scene owner admits the migrated V6 material before execution: "
              + error);
    }

    std::vector<videowire::CompiledVisualLayerPlan> diffractionPlans {
        importedScenePlan(false, false, true)
    };
    const auto diffractionPreparation = prepareVisualImportedSceneLayer(
              diffractionPlans, 7, 640, 360,
              NativeImportedSceneRenderUse::Preview,
              &execution, materialLayer, materialOwners, error);
    const auto expectedDiffractionDigest = diffractionMaterialRequest().structuralDigest;
    const auto actualDiffractionDigest = execution.lastRequest.diffractionMaterial
        ? execution.lastRequest.diffractionMaterial->structuralDigest : std::string {};
    {
        const bool conditionOk = diffractionPreparation == VisualImportedScenePreparation::rendered
              && execution.lastRequest.diffractionMaterial
              && actualDiffractionDigest == expectedDiffractionDigest
              && diffractionmaterial::sameLightingDescription(
                  execution.lastRequest.diffractionMaterial->lighting,
                  canonicalLighting);
        check(conditionOk, "a valid diffraction binding reaches the shared native preview execution owner: preparation="
              + std::to_string(static_cast<int>(diffractionPreparation))
              + ", expectedDigest=" + expectedDiffractionDigest
              + ", actualDigest=" + actualDiffractionDigest
              + ", error=" + error);
    }
    const int previewCallsAfterDiffraction = execution.previewCalls;

    auto tamperedDiffractionPlan = importedScenePlan(false, false, true);
    auto tamperedDiffractionRequest = renderRequest(false, false, true);
    tamperedDiffractionRequest.diffractionMaterial->structuralDigest
        = std::string(64, 'c');
    tamperedDiffractionPlan.operations.back().payloadXml
        = visualimportedscenerender::encode(tamperedDiffractionRequest);
    std::vector<videowire::CompiledVisualLayerPlan> tamperedDiffractionPlans {
        std::move(tamperedDiffractionPlan)
    };
    check(prepareVisualImportedSceneLayer(
              tamperedDiffractionPlans, 7, 640, 360,
              NativeImportedSceneRenderUse::Preview,
              &execution, materialLayer, materialOwners, error)
              == VisualImportedScenePreparation::rejected
              && error.find("structural digest does not match") != std::string::npos
              && execution.previewCalls == previewCallsAfterDiffraction,
          "helper admission recomputes diffraction identity before execution");

    auto detachedDiffractionPlan = importedScenePlan(false, false, true);
    detachedDiffractionPlan.edges.pop_back();
    std::vector<videowire::CompiledVisualLayerPlan> detachedDiffractionPlans {
        std::move(detachedDiffractionPlan)
    };
    check(prepareVisualImportedSceneLayer(
              detachedDiffractionPlans, 7, 640, 360,
              NativeImportedSceneRenderUse::Preview,
              &execution, materialLayer, materialOwners, error)
              == VisualImportedScenePreparation::rejected
              && error.find("exact source-to-Render 3D typed schedule")
                  != std::string::npos,
          "a disconnected diffraction material terminal fails topology admission");

    auto detachedMaterialPlan = importedScenePlan(true);
    detachedMaterialPlan.edges.erase(detachedMaterialPlan.edges.begin() + 1);
    std::vector<videowire::CompiledVisualLayerPlan> detachedMaterialPlans {
        std::move(detachedMaterialPlan)
    };
    FakeLayer detachedMaterialLayer;
    std::vector<FakeReceipt> detachedMaterialOwners;
    check(prepareVisualImportedSceneLayer(
              detachedMaterialPlans, 7, 640, 360,
              NativeImportedSceneRenderUse::Preview,
              &execution, detachedMaterialLayer, detachedMaterialOwners, error)
              == VisualImportedScenePreparation::rejected,
          "a disconnected ordinary material terminal input fails closed");

    auto terminalOnlyMaterialPlan = importedScenePlan(true);
    const auto materialValueNode = [](int nodeId)
        { return nodeId >= 72 && nodeId <= 81; };
    terminalOnlyMaterialPlan.nodeKinds.erase(
        terminalOnlyMaterialPlan.nodeKinds.begin() + 1,
        terminalOnlyMaterialPlan.nodeKinds.begin() + 11);
    terminalOnlyMaterialPlan.nodeIds.erase(
        terminalOnlyMaterialPlan.nodeIds.begin() + 1,
        terminalOnlyMaterialPlan.nodeIds.begin() + 11);
    terminalOnlyMaterialPlan.operations.erase(
        terminalOnlyMaterialPlan.operations.begin() + 1,
        terminalOnlyMaterialPlan.operations.begin() + 11);
    terminalOnlyMaterialPlan.ports.erase(
        std::remove_if(terminalOnlyMaterialPlan.ports.begin(),
                       terminalOnlyMaterialPlan.ports.end(),
                       [&](const auto& port) { return materialValueNode(port.nodeId); }),
        terminalOnlyMaterialPlan.ports.end());
    terminalOnlyMaterialPlan.edges.erase(
        std::remove_if(terminalOnlyMaterialPlan.edges.begin(),
                       terminalOnlyMaterialPlan.edges.end(),
                       [&](const auto& edge) { return materialValueNode(edge.fromNodeId); }),
        terminalOnlyMaterialPlan.edges.end());
    std::vector<videowire::CompiledVisualLayerPlan> terminalOnlyMaterialPlans {
        std::move(terminalOnlyMaterialPlan)
    };
    check(prepareVisualImportedSceneLayer(
              terminalOnlyMaterialPlans, 7, 640, 360,
              NativeImportedSceneRenderUse::Preview,
              &execution, detachedMaterialLayer, detachedMaterialOwners, error)
              == VisualImportedScenePreparation::rejected,
          "a terminal-only schedule cannot claim a transported material program");

    auto mismatchedMaterialPlan = importedScenePlan(true);
    mismatchedMaterialPlan.nodeKinds[1] = "visual.surface.input.position";
    mismatchedMaterialPlan.operations[1].kind = "visual.surface.input.position";
    std::vector<videowire::CompiledVisualLayerPlan> mismatchedMaterialPlans {
        std::move(mismatchedMaterialPlan)
    };
    check(prepareVisualImportedSceneLayer(
              mismatchedMaterialPlans, 7, 640, 360,
              NativeImportedSceneRenderUse::Preview,
              &execution, detachedMaterialLayer, detachedMaterialOwners, error)
              == VisualImportedScenePreparation::rejected,
          "a transported material program cannot disagree with the ordinary DAG");

    FakeLayer rejectedLayer;
    std::vector<FakeReceipt> rejectedOwners;
    check(prepareVisualImportedSceneLayer(
              plans, 7, 640, 360, NativeImportedSceneRenderUse::Preview,
              static_cast<FakeExecution*>(nullptr), rejectedLayer, rejectedOwners, error)
              == VisualImportedScenePreparation::rejected,
          "missing processor payload execution fails closed");

    execution.succeed = false;
    check(prepareVisualImportedSceneLayer(
              plans, 7, 640, 360, NativeImportedSceneRenderUse::Preview,
              &execution, rejectedLayer, rejectedOwners, error)
              == VisualImportedScenePreparation::rejected
              && error == "preview rejected",
          "missing exact resident payload execution fails closed");
    execution.succeed = true;

    check(prepareVisualImportedSceneLayer(
              plans, 7, 0, 360, NativeImportedSceneRenderUse::Preview,
              &execution, rejectedLayer, rejectedOwners, error)
              == VisualImportedScenePreparation::rejected,
          "out-of-bounds dimensions fail closed before execution");

    execution.nextFrame = std::make_shared<FakeFrame>();
    execution.nextFrame->api = "vulkan";
    check(prepareVisualImportedSceneLayer(
              plans, 7, 640, 360, NativeImportedSceneRenderUse::Preview,
              &execution, rejectedLayer, rejectedOwners, error)
              == VisualImportedScenePreparation::rejected,
          "an unsupported native frame backend fails closed");

    execution.nextFrame = std::make_shared<FakeFrame>();
    execution.nextFrame->view = 0;
    check(prepareVisualImportedSceneLayer(
              plans, 7, 640, 360, NativeImportedSceneRenderUse::Preview,
              &execution, rejectedLayer, rejectedOwners, error)
              == VisualImportedScenePreparation::rejected,
          "a missing native texture view fails closed");

    if (std::numeric_limits<std::uintptr_t>::max()
        > std::numeric_limits<unsigned>::max())
    {
        execution.nextFrame = std::make_shared<FakeFrame>();
        execution.nextFrame->view = static_cast<std::uintptr_t>(
            std::numeric_limits<unsigned>::max()) + 1u;
        check(prepareVisualImportedSceneLayer(
                  plans, 7, 640, 360, NativeImportedSceneRenderUse::Preview,
                  &execution, rejectedLayer, rejectedOwners, error)
                  == VisualImportedScenePreparation::rejected,
              "an oversized OpenGL texture view is never narrowed");
    }

    auto malformed = importedScenePlan();
    malformed.operations.back().payloadXml += "x";
    videowire::VisualLayerExecution compiled;
    check(!videowire::compileVisualLayerExecution(malformed, compiled, error),
          "a noncanonical imported scene operation fails closed");

    auto sourceOnly = importedScenePlan();
    sourceOnly.nodeKinds.pop_back();
    sourceOnly.nodeIds.pop_back();
    sourceOnly.operations.pop_back();
    sourceOnly.ports.resize(5);
    sourceOnly.edges.clear();
    check(!videowire::compileVisualLayerExecution(sourceOnly, compiled, error),
          "source-only imported scene execution fails closed");

    auto deformationOnly = sourceOnly;
    deformationOnly.nodeKinds.front() =
        std::string(visualanimationoperation::kDeformationNodeKind);
    deformationOnly.operations.front().kind =
        std::string(visualanimationoperation::kDeformationNodeKind);
    deformationOnly.operations.front().backendCapability =
        std::string(visualanimationoperation::kDeformationBackendCapability);
    check(!videowire::compileVisualLayerExecution(deformationOnly, compiled, error),
          "deformation-only imported scene execution fails closed");

    auto wrongEdge = importedScenePlan();
    wrongEdge.edges.front().fromPort = 3;
    check(!videowire::compileVisualLayerExecution(wrongEdge, compiled, error),
          "a non-Scene source edge fails closed");

    {
        visualnoteinstancing::Mapping appearance;
        appearance.appearanceSource = visualnoteinstancing::AppearanceSource::StableNoteIdentity;
        appearance.appearanceLow = { 1.0f, 0.0f, 0.0f, 0.25f };
        appearance.appearanceHigh = { 0.0f, 0.0f, 1.0f, 2.0f };
        auto notesPlan = importedScenePlan();
        notesPlan.nodeKinds.insert(notesPlan.nodeKinds.end() - 1,
            { "visual.score.note-collection", "visual.3d.note-instanced-mesh" });
        notesPlan.nodeIds.insert(notesPlan.nodeIds.end() - 1, { 90, 91 });
        notesPlan.operations.insert(notesPlan.operations.end() - 1, {
            { 90, "visual.score.note-collection", "control-eval", "" },
            { 91, "visual.3d.note-instanced-mesh", "control-eval", visualnoteinstancing::encode(appearance) }
        });
        notesPlan.ports.insert(notesPlan.ports.end(), {
            { 90, 0, 1, "out", "control", "noteCollection", "unspecified", "unspecified" },
            { 91, 0, 1, "in", "control", "mesh", "unspecified", "unspecified" },
            { 91, 1, 1, "in", "control", "noteCollection", "unspecified", "unspecified" },
            { 91, 2, 1, "out", "control", "mesh", "unspecified", "unspecified" },
            { 71, 2, 1, "in", "control", "material", "unspecified", "unspecified" },
            { 71, 3, 1, "in", "control", "mesh", "unspecified", "unspecified" },
            { 71, 4, 1, "out", "frame", "depth", "r32f", "unspecified" },
            { 71, 5, 1, "in", "control", "camera", "unspecified", "unspecified" },
            { 71, 6, 1, "in", "control", "light", "unspecified", "unspecified" }
        });
        notesPlan.edges.insert(notesPlan.edges.end(), {
            { 70, 0, 91, 0 }, { 90, 0, 91, 1 }, { 91, 2, 71, 3 }
        });
        videowire::VisualLayerExecution notesCompiled;
        {
            const bool conditionOk = videowire::compileVisualLayerExecution(notesPlan, notesCompiled, error);
            check(conditionOk, "authored note appearance and current Render 3D ports compile: " + error);
        }
        check(notesCompiled.noteInstanceMapping.has_value()
                  && visualnoteinstancing::encode(*notesCompiled.noteInstanceMapping)
                      == visualnoteinstancing::encode(appearance),
              "helper compilation preserves the exact note appearance mapping");
        auto score = std::make_shared<arbitmod::Score>();
        score->rootFreq = 440.0f;
        score->notationVersion = score->scoreRevision = 1;
        canonicalblockc::FrameKey frameKey;
        frameKey.projectGeneration = frameKey.sourceGeneration = frameKey.helperGeneration = 1;
        frameKey.backendGeneration = frameKey.deviceGeneration = frameKey.scoreGeneration = 1;
        frameKey.beatMapGeneration = frameKey.fpsGeneration = frameKey.loopGeneration = frameKey.seekGeneration = 1;
        frameKey.fps = 60.0;
        canonicalblockc::FrameProducer frameProducer;
        const auto noteFrame = frameProducer.evaluate(frameKey, score, 0.0f);
        check(canonicalblockc::valid(noteFrame) && noteFrame->noteRows() == 0,
              "an empty score produces a valid zero-row canonical frame");
        FakeExecution notesExecution;
        FakeLayer notesLayer;
        notesLayer.canonicalBlockCFrame = noteFrame;
        std::vector<FakeReceipt> notesOwners;
        for (const auto use : { NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export })
        {
            {
                const bool conditionOk = prepareVisualImportedSceneLayer({ notesPlan }, 7, 640, 360, use,
                      &notesExecution, notesLayer, notesOwners, error)
                      == VisualImportedScenePreparation::rendered;
                check(conditionOk, "note appearance reaches both preview and export: " + error);
            }
            check(notesExecution.lastRequest.runtimeInputs.noteInstanceMapping.has_value()
                      && visualnoteinstancing::encode(*notesExecution.lastRequest.runtimeInputs.noteInstanceMapping)
                          == visualnoteinstancing::encode(appearance)
                      && notesExecution.lastRequest.runtimeInputs.canonicalBlockCFrame == noteFrame
                      && notesLayer.canonicalBlockCFrame == noteFrame,
                  "preview and export retain the same mapping and canonical frame owner");
            const auto batch = arbitgpu::prepareNativeNoteInstances(notesExecution.lastRequest.runtimeInputs);
            check(batch.admitted && batch.count == 0,
                  "the empty canonical frame renders zero note instances");
        }
        FakeLayer missingFrameLayer;
        check(prepareVisualImportedSceneLayer({ notesPlan }, 7, 640, 360,
                  NativeImportedSceneRenderUse::Preview, &notesExecution,
                  missingFrameLayer, notesOwners, error)
                  == VisualImportedScenePreparation::rejected
                  && error == "note-instanced imported scene requires one canonical Block C frame",
              "a note-instanced scene still rejects a missing canonical frame");
        auto velocityScale = appearance;
        velocityScale.schemaVersion = visualnoteinstancing::kVelocityScaleMappingSchemaVersion;
        velocityScale.velocityScaleLow = 0.5f;
        velocityScale.velocityScaleHigh = 2.0f;
        auto scalePlan = notesPlan;
        scalePlan.operations[2].payloadXml = visualnoteinstancing::encode(velocityScale);
        videowire::VisualLayerExecution scaleCompiled;
        {
            const bool conditionOk = videowire::compileVisualLayerExecution(scalePlan, scaleCompiled, error)
                  && scaleCompiled.importedSceneRender && scaleCompiled.noteInstanceMapping
                  && scaleCompiled.importedSceneRender->sourceStableId == 71
                  && scaleCompiled.importedSceneRender->renderStableId == 72
                  && visualnoteinstancing::encode(*scaleCompiled.noteInstanceMapping)
                      == scalePlan.operations[2].payloadXml;
            check(conditionOk, "v4 refusal uses the authenticated imported source and decoded mapping: " + error);
        }
        const std::string scaleDiagnostic = "Native per-note velocity scale transport is unavailable";
        for (const auto use : { NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export })
            for (const bool seeded : { false, true })
                for (const bool presentFrame : { false, true })
                {
                    FakeExecution execution;
                    FakeLayer layer;
                    std::vector<FakeReceipt> owners;
                    videohelper::importedscene::VisualImportedScenePlanCache cache;
                    const videowire::VisualLayerExecution* incumbentCompiled = nullptr;
                    if (seeded)
                    {
                        layer.canonicalBlockCFrame = noteFrame;
                        {
                            const bool conditionOk = prepareVisualImportedSceneLayerAtTime({notesPlan}, 7, 640, 360, 0.0, 60.0, use,
                                  &execution, &cache, layer, owners, error, nullptr, {12, 34})
                                  == VisualImportedScenePreparation::rendered;
                            check(conditionOk, "legacy note mapping seeds the real cache and receipt route: " + error);
                        }
                        incumbentCompiled = cache.resolve(notesPlan, error);
                    }
                    layer.canonicalBlockCFrame = presentFrame ? noteFrame : nullptr;
                    layer.texture = 731; layer.texWidth = 640; layer.texHeight = 360;
                    layer.shaderClock = {2.25, 135}; layer.shaderTransitionFromClipId = 19;
                    layer.audioPresent = true; layer.audioFeatures.bands = {0.25f, 0.5f};
                    layer.rawExportFrame = execution.nextFrame; layer.rawExportMask = 3;
                    layer.rawExportClipId = 7; layer.rawExportRenderId = 72;
                    layer.rawExportImageOutput = renderpassoutput::Output::Depth;
                    const auto incumbent = layer;
                    const auto count = owners.size();
                    const auto priorFrame = owners.empty() ? nullptr : owners.back().frame;
                    const auto priorOwner = owners.empty() ? 0 : owners.back().owner;
                    const auto previewCalls = execution.previewCalls, exportCalls = execution.exportCalls;
                    const auto uploads = execution.staticGpuUploads, retired = execution.identityRetirements;
                    const auto identity = execution.publishedIdentity;
                    const auto priorRequestFrame = execution.lastRequest.runtimeInputs.canonicalBlockCFrame;
                    check(prepareVisualImportedSceneLayerAtTime({scalePlan}, 7, 640, 360, 2.5, 60.0, use,
                              &execution, &cache, layer, owners, error, nullptr, {56, 78})
                              == VisualImportedScenePreparation::rejected && error == scaleDiagnostic,
                          "authenticated v4 refuses before cold or seeded preview/export preparation");
                    check(execution.previewCalls == previewCalls && execution.exportCalls == exportCalls
                              && execution.staticGpuUploads == uploads && execution.identityRetirements == retired
                              && execution.publishedIdentity == identity
                              && execution.lastRequest.runtimeInputs.canonicalBlockCFrame == priorRequestFrame
                              && owners.size() == count
                              && (owners.empty() || (owners.back().frame == priorFrame && owners.back().owner == priorOwner))
                              && layer.texture == incumbent.texture && layer.nativeTextureView == incumbent.nativeTextureView
                              && layer.nativeTextureBackend == incumbent.nativeTextureBackend
                              && layer.nativeTextureDescriptor.imageHandle == incumbent.nativeTextureDescriptor.imageHandle
                              && layer.nativeTextureDescriptor.deviceOrContextIdentity == incumbent.nativeTextureDescriptor.deviceOrContextIdentity
                              && layer.depthTexture == incumbent.depthTexture && layer.nativeDepthTextureView == incumbent.nativeDepthTextureView
                              && layer.nativeDepthTextureBackend == incumbent.nativeDepthTextureBackend
                              && layer.nativeDepthTextureDescriptor.imageHandle == incumbent.nativeDepthTextureDescriptor.imageHandle
                              && layer.texWidth == incumbent.texWidth && layer.texHeight == incumbent.texHeight
                              && layer.depthWidth == incumbent.depthWidth && layer.depthHeight == incumbent.depthHeight
                              && layer.shaderClock.time == incumbent.shaderClock.time && layer.shaderClock.frame == incumbent.shaderClock.frame
                              && layer.shaderTransitionFromClipId == incumbent.shaderTransitionFromClipId
                              && layer.audioPresent == incumbent.audioPresent && layer.audioFeatures.bands == incumbent.audioFeatures.bands
                              && layer.canonicalBlockCFrame == incumbent.canonicalBlockCFrame
                              && layer.rawExportFrame == incumbent.rawExportFrame && layer.rawExportMask == incumbent.rawExportMask
                              && layer.rawExportClipId == incumbent.rawExportClipId && layer.rawExportRenderId == incumbent.rawExportRenderId
                              && layer.rawExportImageOutput == incumbent.rawExportImageOutput,
                          "v4 refusal preserves callbacks, static identity, layer and retained frame owners");
                    if (seeded)
                        check(cache.resolve(notesPlan, error) == incumbentCompiled,
                              "v4 refusal leaves the seeded compiled owner intact across generation changes");
                }
        for (const auto version : {1u, 2u, 3u})
        {
            auto legacy = visualnoteinstancing::Mapping {};
            legacy.schemaVersion = version;
            auto legacyPlan = notesPlan;
            legacyPlan.operations[2].payloadXml = visualnoteinstancing::encode(legacy);
            FakeExecution execution;
            FakeLayer layer;
            layer.canonicalBlockCFrame = noteFrame;
            std::vector<FakeReceipt> owners;
            auto foreignClipPlan = scalePlan;
            foreignClipPlan.clipId = 8;
            for (const auto use : { NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export })
                {
                    const bool conditionOk = prepareVisualImportedSceneLayer({legacyPlan, foreignClipPlan}, 7, 640, 360, use,
                          &execution, layer, owners, error) == VisualImportedScenePreparation::rendered
                          && execution.lastRequest.runtimeInputs.noteInstanceMapping
                          && visualnoteinstancing::encode(*execution.lastRequest.runtimeInputs.noteInstanceMapping)
                              == legacyPlan.operations[2].payloadXml;
                    check(conditionOk, "v1-v3 retain their exact mapping and ignore another clip's v4 producer: " + error);
                }
        }
        auto malformedScale = scalePlan;
        malformedScale.operations[2].payloadXml += "0\n";
        auto foreignSourceScale = scalePlan;
        foreignSourceScale.edges[1].fromNodeId = 90;
        for (const auto& invalid : { malformedScale, foreignSourceScale })
        {
            videowire::VisualLayerExecution invalidCompiled;
            std::string compilerError;
            check(!videowire::compileVisualLayerExecution(invalid, invalidCompiled, compilerError)
                      && !compilerError.empty() && compilerError != scaleDiagnostic,
                  "malformed or foreign-source v4 has a canonical compiler diagnostic");
            FakeExecution execution;
            FakeLayer layer;
            layer.texture = 731;
            std::vector<FakeReceipt> owners;
            videohelper::importedscene::VisualImportedScenePlanCache cache;
            check(prepareVisualImportedSceneLayerAtTime({invalid}, 7, 640, 360, 0.0, 60.0,
                      NativeImportedSceneRenderUse::Preview, &execution, &cache, layer, owners, error)
                      == VisualImportedScenePreparation::rejected && error == compilerError
                      && execution.previewCalls == 0 && execution.exportCalls == 0
                      && execution.publishedIdentity.empty() && execution.identityRetirements == 0
                      && layer.texture == 731 && owners.empty(),
                  "raw v4 header cannot replace canonical malformed or foreign-source diagnostics");
        }
        WaterMeshCaptureBackend capture;
        videohelper::modelpayload::Store store;
        videohelper::modelpayload::ImportedScenePayloadExecution payloadExecution(store, capture);
        videohelper::importedscene::VisualImportedScenePlanCache productionCache;
        FakeLayer productionLayer;
        productionLayer.texture = 731;
        productionLayer.canonicalBlockCFrame = noteFrame;
        std::vector<videohelper::modelpayload::ImportedSceneExecutionReceipt> productionOwners;
        check(prepareVisualImportedSceneLayerAtTime({scalePlan}, 7, 640, 360, 0.0, 60.0,
                  NativeImportedSceneRenderUse::Preview, &payloadExecution, &productionCache,
                  productionLayer, productionOwners, error) == VisualImportedScenePreparation::rejected
                  && error == scaleDiagnostic && capture.prepareCalls == 0
                  && productionLayer.texture == 731 && productionLayer.canonicalBlockCFrame == noteFrame
                  && productionOwners.empty(),
              "real payload execution refuses authenticated v4 before its native prepare callback");
        notesPlan.operations[2].payloadXml += "0\n";
        check(!videowire::compileVisualLayerExecution(notesPlan, notesCompiled, error),
              "malformed note appearance is rejected before native submission");
    }

    testMixedRetainedSceneSchedule();

    if (failures != 0)
    {
        std::cerr << failures << " imported scene visual plan checks failed\n";
        return 1;
    }
    std::cout << "imported scene visual plan execution: all checks passed\n";
    return 0;
}
