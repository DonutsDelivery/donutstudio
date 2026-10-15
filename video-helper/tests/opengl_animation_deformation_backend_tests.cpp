#include "VisualAnimationDeformationEvaluation.h"
#include "../src/gl_loader.h"
#include "../src/native_animation_deformation_renderer.h"
#include "../src/gpu_backend/backend.h"
#include "support/fixture_scene.h"
#include "support/native_deformation_composition_fixture.h"
#include "native_composition_production_join_fixture.h"
#include "animated_geometry_fixture.h"
#include "imported_animation_beat_fixture.h"
#include "multiobject_animation_fixture.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <GL/gl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <cstring>
#include <nlohmann/json.hpp>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace
{
using namespace visualdeformation;
namespace animation = visualanimation;

bool expect(bool condition, const char* message)
{
    if (! condition)
        std::cerr << "FAIL: " << message << '\n';
    return condition;
}

struct Fixture final
{
    HarmonicMIDI::grid::Visual3DScene sceneValue = videohelper::fixture3d::makeScene();
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> scene;
    std::array<float, 24 * 3> morphPositions {};
    MorphTargetView morphTarget;
    MeshView mesh;
    AssetView assetView;
    std::shared_ptr<const DeformationAsset> asset;
    std::array<double, 2> keyTimes { 0.0, 1.0 };
    std::array<float, 2> morphValues { 0.0f, 0.75f };
    animation::TrackView track;
    animation::ClipView clipView;
    std::shared_ptr<const animation::Clip> clip;
    std::array<MorphTargetId, 1> targetIds { MorphTargetId { 601 } };
    MorphAnimationBindingView morphBinding;
    AnimationDeformationBindingView bindings;
    std::shared_ptr<const arbitgpu::NativeDeformationScene> source;

    std::array<JointView,1> joints { JointView { JointId { 301 }, {} } };
    std::array<float,16> inverseBind { 1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1 };
    std::array<std::uint32_t,24*4> jointIndices {};
    std::array<float,24*4> jointWeights {};
    JointWeightSetView weightSet;
    SkinView skinView;
    std::array<float,6> jointValues { 0,0,0,0,.75f,0 };
    JointAnimationBindingView jointBinding;

    explicit Fixture(bool skin = false,bool axisAligned = false)
    {
        if (skin || axisAligned) sceneValue.objects[0].transform = {};
        for (std::size_t vertex = 0; vertex < 24; ++vertex)
            morphPositions[vertex * 3] = 1.0f;
        morphTarget = { targetIds[0], morphPositions.data(), morphPositions.size(),
                        nullptr, 0, nullptr, 0 };
        mesh = { MeshId { 501 }, 24, {}, nullptr, 0, &morphTarget, 1 };
        if (skin)
        {
            for (std::size_t vertex=0;vertex<24;++vertex) jointWeights[vertex*4] = 1;
            weightSet = { jointIndices.data(),jointIndices.size(),jointWeights.data(),jointWeights.size() };
            skinView = { SkinId { 401 },joints.data(),joints.size(),inverseBind.data(),inverseBind.size() };
            mesh.skin = skinView.id;
            mesh.jointWeightSets = &weightSet;
            mesh.jointWeightSetCount = 1;
        }
        assetView = { skin ? &skinView : nullptr, skin ? 1u : 0u, &mesh, 1 };
        std::string error;
        asset = DeformationAsset::create(assetView, {}, error);
        if (asset == nullptr)
            std::cerr << "fixture deformation asset error: " << error << '\n';

        track = { animation::TrackId { 12 }, animation::TargetId { 1002 },
                  animation::Channel::MorphWeights, animation::Interpolation::Linear,
                  keyTimes.data(), morphValues.data(), keyTimes.size(),
                  morphValues.size(), 1 };
        if (skin)
            track = { animation::TrackId { 12 },animation::TargetId { 1002 },
                animation::Channel::Translation,animation::Interpolation::Linear,
                keyTimes.data(),jointValues.data(),keyTimes.size(),jointValues.size(),0 };
        clipView = { animation::ClipId { 701 }, 1.0, &track, 1 };
        clip = animation::Clip::create(clipView, {}, error);
        if (clip == nullptr)
            std::cerr << "fixture animation clip error: " << error << '\n';

        morphBinding = { animation::TargetId { 1002 }, mesh.id,
                         targetIds.data(), targetIds.size() };
        jointBinding = { animation::TargetId { 1002 },skinView.id,joints[0].id };
        bindings = skin ? AnimationDeformationBindingView { &jointBinding,1,nullptr,0 }
                        : AnimationDeformationBindingView { nullptr,0,&morphBinding,1 };
        scene = std::make_shared<const HarmonicMIDI::grid::Visual3DScene>(sceneValue);
        auto mutableSource = std::make_shared<arbitgpu::NativeDeformationScene>();
        mutableSource->sourceStableId = 801;
        mutableSource->deformationStableId = 802;
        mutableSource->structuralRevision = 77;
        mutableSource->clip = animation::ClipId { 701 };
        mutableSource->mesh = mesh.id;
        mutableSource->object = videohelper::fixture3d::kCubeObjectId;
        mutableSource->scene = scene;
        mutableSource->deformation = asset;
        mutableSource->morphBaseWeights = { 0.0f };
        if (skin)
            mutableSource->jointBaseTransforms = { arbitgpu::NativeDeformationJointBaseTransform {
                skinView.id,joints[0].id,{0,0,0},{0,0,0,1},{1,1,1} } };
        mutableSource->retainDeformedGeometry = true;
        source = std::move(mutableSource);
    }

    std::shared_ptr<const AnimationDeformationSnapshot> snapshot(std::int64_t frame,std::uint32_t rate = 30) const
    {
        std::string error;
        const AnimationDeformationRequest request {
            RationalFrameTime { frame, rate, 1 }, 77,
            animation::Playback::Clamp, std::nullopt
        };
        auto result = evaluateAnimationDeformation(*asset, *clip, bindings,
                                                   request, {}, error);
        if (result == nullptr)
            std::cerr << "fixture snapshot error: " << error << '\n';
        return result;
    }
};

namespace importedmotionfixture
{
void appendU16 (std::vector<std::uint8_t>& bytes, std::uint16_t value)
{
    bytes.push_back (static_cast<std::uint8_t> (value));
    bytes.push_back (static_cast<std::uint8_t> (value >> 8u));
}

void appendU32 (std::vector<std::uint8_t>& bytes, std::uint32_t value)
{
    bytes.push_back (static_cast<std::uint8_t> (value));
    bytes.push_back (static_cast<std::uint8_t> (value >> 8u));
    bytes.push_back (static_cast<std::uint8_t> (value >> 16u));
    bytes.push_back (static_cast<std::uint8_t> (value >> 24u));
}

void appendFloat (std::vector<std::uint8_t>& bytes, float value)
{
    std::uint32_t encoded = 0;
    std::memcpy (&encoded, &value, sizeof (encoded));
    appendU32 (bytes, encoded);
}

std::vector<std::uint8_t> makeGlb (nlohmann::json root,
                                   std::vector<std::uint8_t> bin)
{
    auto text = root.dump();
    while ((text.size() & 3u) != 0)
        text.push_back (' ');
    while ((bin.size() & 3u) != 0)
        bin.push_back (0);

    std::vector<std::uint8_t> bytes;
    const auto total = 12u + 8u + static_cast<unsigned> (text.size())
                     + 8u + static_cast<unsigned> (bin.size());
    appendU32 (bytes, 0x46546c67u);
    appendU32 (bytes, 2u);
    appendU32 (bytes, total);
    appendU32 (bytes, static_cast<std::uint32_t> (text.size()));
    appendU32 (bytes, 0x4e4f534au);
    bytes.insert (bytes.end(), text.begin(), text.end());
    appendU32 (bytes, static_cast<std::uint32_t> (bin.size()));
    appendU32 (bytes, 0x004e4942u);
    bytes.insert (bytes.end(), bin.begin(), bin.end());
    return bytes;
}

std::string encodeBase64 (const std::vector<std::uint8_t>& bytes)
{
    static constexpr char alphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string encoded;
    encoded.reserve ((bytes.size() + 2u) / 3u * 4u);
    for (std::size_t offset = 0; offset < bytes.size(); offset += 3u)
    {
        const auto remaining = bytes.size() - offset;
        const auto value = static_cast<std::uint32_t> (bytes[offset]) << 16u
            | (remaining > 1u ? static_cast<std::uint32_t> (bytes[offset + 1u]) << 8u : 0u)
            | (remaining > 2u ? static_cast<std::uint32_t> (bytes[offset + 2u]) : 0u);
        encoded.push_back (alphabet[(value >> 18u) & 63u]);
        encoded.push_back (alphabet[(value >> 12u) & 63u]);
        encoded.push_back (remaining > 1u ? alphabet[(value >> 6u) & 63u] : '=');
        encoded.push_back (remaining > 2u ? alphabet[value & 63u] : '=');
    }
    return encoded;
}

visualanimationimport::ExactContentAssetKey exactKey (
    const char* id, const std::vector<std::uint8_t>& bytes)
{
    return {id, 1u, videohelper::modelpayload::detail::digestHex (bytes),
            "model/gltf-binary", bytes.size()};
}

bool publish (videohelper::modelpayload::Store& store,
              const char* transferId,
              const visualanimationimport::ExactContentAssetKey& key,
              const std::vector<std::uint8_t>& bytes)
{
    return store.begin (transferId, key)
        && store.appendBase64 (transferId, 0u, encodeBase64 (bytes))
        && store.commit (transferId);
}

std::vector<std::uint8_t> triangleBin()
{
    std::vector<std::uint8_t> bin;
    for (const auto value : {-0.5f, -0.5f, 0.0f,
                              0.5f, -0.5f, 0.0f,
                              0.0f,  0.5f, 0.0f})
        appendFloat (bin, value);
    for (int vertex = 0; vertex < 3; ++vertex)
        for (const auto value : {0.0f, 0.0f, 1.0f})
            appendFloat (bin, value);
    for (const auto value : {0.0f, 0.0f, 1.0f, 0.0f, 0.5f, 1.0f})
        appendFloat (bin, value);
    appendU16 (bin, 0);
    appendU16 (bin, 1);
    appendU16 (bin, 2);
    return bin;
}

nlohmann::json triangleRoot()
{
    return {
        {"asset", {{"version", "2.0"}, {"generator", "native seam fixture"}}},
        {"scene", 0},
        {"scenes", nlohmann::json::array ({
            {{"name", "Primary"}, {"nodes", nlohmann::json::array ({0, 1, 2})}},
            {{"name", "Alternate"}, {"nodes", nlohmann::json::array ({0, 1, 2})}}
        })},
        {"nodes", nlohmann::json::array ({
            {{"name", "Triangle"}, {"mesh", 0}},
            {{"name", "Camera"}, {"camera", 0},
             {"translation", nlohmann::json::array ({0.0, 0.0, 2.0})}},
            {{"name", "Key"},
             {"extensions", {{"KHR_lights_punctual", {{"light", 0}}}}}}
        })},
        {"meshes", nlohmann::json::array ({
            {{"primitives", nlohmann::json::array ({
                {{"attributes", {{"POSITION", 0}, {"NORMAL", 1}, {"TEXCOORD_0", 2}}},
                 {"indices", 3}, {"material", 0}}
            })}}
        })},
        {"materials", nlohmann::json::array ({
            {{"pbrMetallicRoughness", {
                {"baseColorFactor", nlohmann::json::array ({0.2, 0.4, 0.8, 1.0})},
                {"metallicFactor", 0.1}, {"roughnessFactor", 0.7}}}}
        })},
        {"cameras", nlohmann::json::array ({
            {{"type", "perspective"}, {"perspective", {
                {"yfov", 1.0}, {"znear", 0.1}, {"zfar", 20.0}}}}
        })},
        {"extensionsUsed", nlohmann::json::array ({"KHR_lights_punctual"})},
        {"extensionsRequired", nlohmann::json::array ({"KHR_lights_punctual"})},
        {"extensions", {{"KHR_lights_punctual", {
            {"lights", nlohmann::json::array ({
                {{"type", "directional"}, {"intensity", 2.0}}
            })}}}}},
        {"buffers", nlohmann::json::array ({
            {{"byteLength", 102}}
        })},
        {"bufferViews", nlohmann::json::array ({
            {{"buffer", 0}, {"byteOffset", 0}, {"byteLength", 36}, {"target", 34962}},
            {{"buffer", 0}, {"byteOffset", 36}, {"byteLength", 36}, {"target", 34962}},
            {{"buffer", 0}, {"byteOffset", 72}, {"byteLength", 24}, {"target", 34962}},
            {{"buffer", 0}, {"byteOffset", 96}, {"byteLength", 6}, {"target", 34963}}
        })},
        {"accessors", nlohmann::json::array ({
            {{"bufferView", 0}, {"componentType", 5126}, {"count", 3}, {"type", "VEC3"}},
            {{"bufferView", 1}, {"componentType", 5126}, {"count", 3}, {"type", "VEC3"}},
            {{"bufferView", 2}, {"componentType", 5126}, {"count", 3}, {"type", "VEC2"}},
            {{"bufferView", 3}, {"componentType", 5123}, {"count", 3}, {"type", "SCALAR"}}
        })}
    };
}

std::vector<std::uint8_t> importedObjectMotionFixture(bool parentTarget, bool moving)
{
    auto root = triangleRoot();
    auto bin = triangleBin();
    bin.resize(104, 0);
    for (const auto time : {0.0f, 1.0f}) appendFloat(bin, time);
    const std::array<float, 3> first = parentTarget
        ? std::array<float, 3>{0.1f, 0.2f, 0.0f}
        : std::array<float, 3>{0.25f, -0.125f, 0.0f};
    for (int key = 0; key < 2; ++key)
        for (std::size_t axis = 0; axis < 3; ++axis)
            appendFloat(bin, first[axis] + (moving && axis == 0 ? 0.5f * key : 0.0f));
    root["buffers"][0]["byteLength"] = bin.size();
    root["bufferViews"].push_back({{"buffer",0}, {"byteOffset",104}, {"byteLength",8}});
    root["bufferViews"].push_back({{"buffer",0}, {"byteOffset",112}, {"byteLength",24}});
    root["accessors"].push_back({{"bufferView",4}, {"componentType",5126}, {"count",2}, {"type","SCALAR"}});
    root["accessors"].push_back({{"bufferView",5}, {"componentType",5126}, {"count",2}, {"type","VEC3"}});
    root["nodes"][0]["translation"] = {0.25, -0.125, 0.0};
    root["nodes"][0]["scale"] = {0.5, 0.75, 1.0};
    root["nodes"].push_back({{"name","Animated Parent"}, {"children",{0}},
        {"translation",{0.1,0.2,0.0}}, {"rotation",{0.0,0.0,std::sqrt(0.5),std::sqrt(0.5)}},
        {"scale",{2.0,2.0,1.0}}});
    for (auto& scene : root["scenes"]) scene["nodes"] = {3,1,2};
    root["nodes"][1]["translation"] = {0.0,0.0,4.0};
    root["cameras"][0]["perspective"]["yfov"] = 1.5707963267948966;
    root["animations"] = nlohmann::json::array({{
        {"name","Imported Object Motion"},
        {"samplers",nlohmann::json::array({{{"input",4}, {"output",5}, {"interpolation","LINEAR"}}})},
        {"channels",nlohmann::json::array({{{"sampler",0},
            {"target",{{"node",parentTarget ? 3 : 0}, {"path","translation"}}}}})}
    }});
    return makeGlb(std::move(root), std::move(bin));
}

} // namespace importedmotionfixture

std::vector<float> readMotion(const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& frame)
{
    if (!frame) return {};
    const auto view = frame->passTextureDescriptor(renderpassoutput::Output::Motion);
    if (!view.complete() || view.format != arbitgpu::NativeTexturePixelFormat::Rg16Float) return {};
    GLint previous = 0;
    glGetIntegerv(GL_TEXTURE_BINDING_2D,&previous);
    glBindTexture(GL_TEXTURE_2D,static_cast<GLuint>(view.textureViewHandle));
    std::vector<float> values(frame->width()*frame->height()*2u);
    glGetTexImage(GL_TEXTURE_2D,0,GL_RG,GL_FLOAT,values.data());
    const auto error = glGetError();
    glBindTexture(GL_TEXTURE_2D,static_cast<GLuint>(previous));
    if (error != GL_NO_ERROR) return {};
    return values;
}

bool verifyDeformationMotion(arbitgpu::NativeDeformationBackend& backend)
{
    bool ok = true;
    const auto neutral = [](const auto& pixels) {
        return !pixels.empty() && std::all_of(pixels.begin(),pixels.end(),[](float value){return value==0;});
    };
    for (const bool skin : {false,true})
    {
        Fixture fixture(skin,true);
        const auto previous = fixture.snapshot(0), current = fixture.snapshot(1);
        const auto resources = backend.prepare(fixture.source);
        if (!resources.prepared || !previous || !current) return false;
        arbitgpu::NativeDeformationRuntimeInputs inputs;
        inputs.imageOutput = renderpassoutput::Output::Motion;
        const auto first = backend.render(fixture.source,current,resources.resources,96,96,inputs);
        ok &= expect(first.rendered && neutral(readMotion(first.frame)),"first deformed frame has neutral Motion");
        inputs.previousMotion = arbitgpu::sceneMotionSample(inputs,previous);
        const auto moving = backend.render(fixture.source,current,resources.resources,96,96,inputs);
        const auto repeated = backend.render(fixture.source,current,resources.resources,96,96,inputs);
        const auto pixels = readMotion(moving.frame);
        const auto center = (48u*96u+48u)*2u;
        const auto expected = .025f*48.0f/(std::tan(fixture.scene->cameras[0].verticalFovRadians*.5f)*4.0f);
        ok &= expect(moving.rendered && moving.stats.dispatchCount==2 && pixels.size()==96u*96u*2u
            && pixels==readMotion(repeated.frame),"preceding deformation is consumed by repeatable native GPU draws");
        if (pixels.size()==96u*96u*2u)
            ok &= expect(std::abs(pixels[center+(skin ? 1u : 0u)]-expected)<.01f
                && std::abs(pixels[center+(skin ? 0u : 1u)])<.001f,
                "one-frame morph X or skin Y displacement matches independent pixel projection");
        if (!skin)
        {
            inputs.previousMotion = arbitgpu::sceneMotionSample(inputs,current);
            inputs.morphWeight = 0;
            const auto restored = backend.render(fixture.source,fixture.snapshot(2),resources.resources,96,96,inputs);
            const auto restoredPixels = readMotion(restored.frame);
            ok &= expect(restored.rendered && restoredPixels.size()==96u*96u*2u
                && std::abs(restoredPixels[center]+expected)<.01f,
                "prior morph gain is retained independently of the current gain");
            inputs.morphWeight = 1;
        }
        inputs.previousMotion = arbitgpu::sceneMotionSample(inputs,fixture.snapshot(30));
        const auto unchanged = backend.render(fixture.source,fixture.snapshot(31),resources.resources,96,96,inputs);
        ok &= expect(unchanged.rendered && neutral(readMotion(unchanged.frame)),"unchanged adjacent deformation has zero Motion");
        inputs.previousMotion = arbitgpu::sceneMotionSample(inputs,previous);
        const auto sought = backend.render(fixture.source,fixture.snapshot(30),resources.resources,96,96,inputs);
        const auto rateReset = backend.render(fixture.source,fixture.snapshot(1,60),resources.resources,96,96,inputs);
        ok &= expect(sought.rendered && rateReset.rendered && neutral(readMotion(sought.frame))
            && neutral(readMotion(rateReset.frame)),"nonadjacent or rate-changed deformation resets Motion");
        videorender::animation3d::NativeAnimationDeformationRenderer preview(backend),offline(backend);
        videorender::animation3d::RenderedDeformationFrame live,exported;
        std::string error;
        inputs.previousMotion.reset();
        ok &= expect(offline.renderExport(fixture.source,current,{},inputs,96,96,
            videorender::animation3d::kNativeGpuCapability,exported,error) && neutral(readMotion(exported.nativeFrame)),
            "separate export owner starts with neutral deformation Motion");
        inputs.previousMotion = arbitgpu::sceneMotionSample(inputs,previous);
        ok &= expect(preview.renderPreview(fixture.source,current,{},inputs,96,96,
            videorender::animation3d::kNativeGpuCapability,live,error)
            && offline.renderExport(fixture.source,current,{},inputs,96,96,
            videorender::animation3d::kNativeGpuCapability,exported,error)
            && readMotion(live.nativeFrame)==readMotion(exported.nativeFrame),
            "equal retained predecessors match between isolated Preview and Export renderers");
        ok &= expect(!preview.renderPreview(fixture.source,current,{},inputs,0,96,
            videorender::animation3d::kNativeGpuCapability,live,error) && !live.nativeFrame,
            "failed deformation submission retires the previous published frame");
    }
    return ok;
}

std::vector<std::uint8_t> readRgba8(
    const arbitgl::GlFuncs& gl,
    const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& frame)
{
    GLint previousFramebuffer = 0;
    GLint previousPackAlignment = 0;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &previousFramebuffer);
    glGetIntegerv(GL_PACK_ALIGNMENT, &previousPackAlignment);
    GLuint framebuffer = 0;
    gl.GenFramebuffers(1, &framebuffer);
    gl.BindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    gl.FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                            static_cast<GLuint>(frame->colorImageHandle()), 0);
    std::vector<std::uint8_t> pixels(
        static_cast<std::size_t>(frame->width()) * frame->height() * 4u);
    if (gl.CheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE)
    {
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, static_cast<GLsizei>(frame->width()),
                     static_cast<GLsizei>(frame->height()),
                     GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    }
    glPixelStorei(GL_PACK_ALIGNMENT, previousPackAlignment);
    gl.BindFramebuffer(GL_FRAMEBUFFER, static_cast<GLuint>(previousFramebuffer));
    gl.DeleteFramebuffers(1, &framebuffer);
    return pixels;
}

std::vector<float> readDepth32f(
    const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& frame)
{
    if (!frame) return {};
    GLint previousTexture = 0;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &previousTexture);
    glBindTexture(GL_TEXTURE_2D,
                  static_cast<GLuint>(frame->depthTextureViewHandle()));
    std::vector<float> pixels(
        static_cast<std::size_t>(frame->width()) * frame->height());
    // The depth resource is an R32F colour attachment holding linear depth, so it is
    // read as a single red float channel. Clear stale errors first so the check below
    // reflects only this read.
    for (int attempt = 0; attempt < 64 && glGetError() != GL_NO_ERROR; ++attempt) {}
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, pixels.data());
    const auto error = glGetError();
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(previousTexture));
    if (error != GL_NO_ERROR) return {};
    return pixels;
}

namespace importedmotionfixture
{
constexpr std::array<std::array<double,3>,3> triangle {{
    {{-0.5,-0.5,0.0}}, {{0.5,-0.5,0.0}}, {{0.0,0.5,0.0}}
}};

std::array<double,2> project(bool parentTarget, bool moving, double seconds,
    const std::array<double,3>& vertex, const arbitgpu::NativeSceneMotionSample& outer,
    std::uint32_t width, std::uint32_t height)
{
    // Compose the authored parent T Rz(90) S and child T S in binary64.
    // Neither palettes nor production matrix/projection helpers enter this oracle.
    const double x = 0.35 - 1.5 * vertex[1] + (parentTarget && moving ? 0.5 * seconds : 0.0);
    const double y = 0.7 + vertex[0] + (!parentTarget && moving ? seconds : 0.0);
    const double angle = outer.objectRotationDegrees[2] * std::acos(-1.0) / 180.0;
    const double worldX = outer.objectScale * (std::cos(angle) * x - std::sin(angle) * y)
        + outer.objectTranslationOffset[0];
    const double worldY = outer.objectScale * (std::sin(angle) * x + std::cos(angle) * y)
        + outer.objectTranslationOffset[1];
    const double distance = 4.0 - outer.objectScale * vertex[2] - outer.objectTranslationOffset[2];
    const double scale = 0.5 * height / distance;
    return {0.5 * width + worldX * scale, 0.5 * height + worldY * scale};
}

bool neutral(const std::vector<float>& pixels)
{
    return !pixels.empty() && std::all_of(pixels.begin(), pixels.end(), [](float value) { return value == 0; });
}

bool checkPixels(const videohelper::modelpayload::ImportedAnimatedSceneReceipt& receipt,
                 bool parentTarget, bool moving)
{
    const auto& frame = receipt.frame.nativeFrame;
    if (!frame || !receipt.evaluation.deformation) return false;
    const auto motion = readMotion(frame);
    const auto depth = readDepth32f(frame);
    const auto width = frame->width(), height = frame->height();
    if (motion.size() != width * height * 2u || depth.size() != width * height) return false;
    if (!receipt.runtimeInputs.previousMotion) return neutral(motion);
    const auto& prior = *receipt.runtimeInputs.previousMotion;
    if (!prior.deformation || !prior.importedNodeWorldMatrices) return false;
    const auto current = arbitgpu::sceneMotionSample(receipt.runtimeInputs);
    std::array<std::array<double,2>,3> now {}, before {};
    for (std::size_t vertex = 0; vertex < triangle.size(); ++vertex)
    {
        now[vertex] = project(parentTarget, moving, receipt.evaluation.deformation->sampleTimeSeconds(),
                              triangle[vertex], current, width, height);
        before[vertex] = project(parentTarget, moving, prior.deformation->sampleTimeSeconds(),
                                 triangle[vertex], prior, width, height);
    }
    const auto cross = [](const auto& a, const auto& b, const auto& c) {
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]);
    };
    const double area = cross(now[0], now[1], now[2]);
    if (std::abs(area) < 1.0) return false;
    std::size_t covered = 0;
    std::array<std::size_t,3> vertexRegions {};
    bool ok = true;
    for (std::uint32_t y = 0; y < height; ++y)
        for (std::uint32_t x = 0; x < width; ++x)
        {
            const std::size_t pixel = static_cast<std::size_t>(y) * width + x;
            const std::array<double,2> center {x + 0.5, y + 0.5};
            const std::array<double,3> barycentric {
                cross(now[1], now[2], center) / area,
                cross(now[2], now[0], center) / area,
                cross(now[0], now[1], center) / area
            };
            const auto minimum = *std::min_element(barycentric.begin(), barycentric.end());
            // Exclude only the rasterizer's edge convention from coverage equality.
            if (minimum > 0.02) ok &= depth[pixel] < 1.0f;
            if (depth[pixel] >= 1.0f)
            {
                ok &= motion[pixel*2] == 0 && motion[pixel*2+1] == 0;
                continue;
            }
            ++covered;
            ok &= minimum > -0.002;
            for (std::size_t vertex = 0; vertex < 3; ++vertex)
                if (barycentric[vertex] > 0.7) ++vertexRegions[vertex];
            for (std::size_t axis = 0; axis < 2; ++axis)
            {
                double expected = 0;
                for (std::size_t vertex = 0; vertex < 3; ++vertex)
                    expected += barycentric[vertex] * (now[vertex][axis] - before[vertex][axis]);
                // RG16F rounding below 4 pixels is at most .002 pixels. The .01
                // bound also covers float palette, FOV and interpolation rounding.
                ok &= std::abs(motion[pixel*2+axis] - expected) < 0.01;
            }
        }
    return ok && covered > 30
        && std::all_of(vertexRegions.begin(), vertexRegions.end(), [](auto count) { return count > 0; });
}

bool verifyAuthoredSkin(arbitgpu::NativeDeformationBackend& backend)
{
    using namespace videohelper::modelpayload;
    Store store;
    visualanimationimport::ExactContentAssetKey key;
    if (!importedanimationbeatfixture::publish(store, key)) return false;
    ImportedAnimatedScenePayloadExecution execution(store, backend);
    ImportedAnimatedSceneRequest request;
    request.operation.sourceStableId = 901; request.operation.deformationStableId = 902;
    request.operation.schedule = {901,902}; request.operation.asset = key;
    request.operation.sceneIndex = 0;
    request.operation.meshStableId = request.operation.animationClipStableId = 1;
    request.operation.clipName = "Joint translation";
    request.frame = {0,24,1}; request.structuralRevision = 1;
    request.width = request.height = 96;
    request.runtimeInputs.imageOutput = renderpassoutput::Output::Motion;
    ImportedAnimatedSceneReceipt receipt;
    std::string error;
    if (!execution.executePreview(request, receipt, error) || !neutral(readMotion(receipt.frame.nativeFrame))) return false;
    request.frame.frame = 1; request.operation.playback.timelineSeconds = 1.0 / 24.0;
    if (!execution.executePreview(request, receipt, error) || !receipt.runtimeInputs.previousMotion) return false;
    const auto worlds = receipt.runtimeInputs.previousMotion->importedNodeWorldMatrices;
    // Drawn mesh is node 1. Its authored joint is node 2, outside retained ancestry.
    if (!worlds || worlds->size() != 1 || worlds->front().nodeStableId != 1) return false;
    const auto pixels = readMotion(receipt.frame.nativeFrame);
    const auto depth = readDepth32f(receipt.frame.nativeFrame);
    if (pixels.size() != 96u*96u*2u || depth.size() != 96u*96u) return false;
    const double expectedX = (0.4 / 24.0) * 48.0 / (4.0 * std::tan(0.4));
    std::size_t covered = 0;
    for (std::size_t pixel = 0; pixel < depth.size(); ++pixel)
        if (depth[pixel] < 1.0f)
        {
            ++covered;
            if (std::abs(pixels[pixel*2] - expectedX) >= 0.01
                || std::abs(pixels[pixel*2+1]) >= 0.001) return false;
        }
    return covered > 100;
}

bool verify(arbitgpu::NativeDeformationBackend& backend)
{
    using namespace videohelper::modelpayload;
    using Worlds = arbitgpu::NativeSceneMotionSample::ImportedNodeWorldMatrices;
    bool ok = expect(verifyAuthoredSkin(backend),
        "authored skin joints outside retained drawn ancestry preserve native Motion");
    for (const int mode : {0,1,2})
    {
        const bool parentTarget = mode == 1, moving = mode != 2;
        Store store;
        const auto bytes = importedObjectMotionFixture(parentTarget, moving);
        const auto key = exactKey("native-imported-motion", bytes);
        if (!publish(store, key.id.c_str(), key, bytes)) return false;
        ImportedAnimatedScenePayloadExecution execution(store, backend);
        ImportedAnimatedSceneRequest request;
        request.operation.sourceStableId = 801; request.operation.deformationStableId = 802;
        request.operation.schedule = {801,802}; request.operation.asset = key;
        request.operation.sceneIndex = 0;
        request.operation.meshStableId = request.operation.animationClipStableId = 1;
        request.operation.clipName = "Imported Object Motion";
        request.frame = {0,24,1}; request.structuralRevision = 1;
        request.renderClipId = 83; request.projectGeneration = request.helperGeneration = 1;
        request.width = 96; request.height = 64;
        request.runtimeInputs.imageOutput = renderpassoutput::Output::Motion;
        ImportedAnimatedSceneReceipt receipt;
        std::string error;
        const auto render = [&](bool exported, const char* message) {
            const bool rendered = exported ? execution.executeExport(request, receipt, error)
                                           : execution.executePreview(request, receipt, error);
            if (!rendered) std::cerr << "imported Motion fixture: " << error << '\n';
            return expect(rendered && receipt.valid() && checkPixels(receipt, parentTarget, moving), message);
        };
        if (!render(false, "first imported Motion is exactly zero without a predecessor")) return false;
        request.frame.frame = 1; request.operation.playback.timelineSeconds = 1.0 / 24.0;
        if (!render(false, "mesh, rotated/scaled ancestor and static Motion match every covered raw RG16F pixel")) return false;
        const auto ordinaryPixels = readMotion(receipt.frame.nativeFrame);
        if (!expect(receipt.runtimeInputs.previousMotion
            && receipt.runtimeInputs.previousMotion->importedNodeWorldMatrices
            && receipt.runtimeInputs.previousMotion->importedNodeWorldMatrices->size() == 2
            && (*receipt.runtimeInputs.previousMotion->importedNodeWorldMatrices)[0].nodeStableId == 1
            && (*receipt.runtimeInputs.previousMotion->importedNodeWorldMatrices)[1].nodeStableId == 4,
            "native imported Motion receives the exact drawn-node and ancestor world identities")) return false;
        if (!parentTarget && moving)
            ok &= expect(std::any_of(ordinaryPixels.begin(), ordinaryPixels.end(), [](float value) { return value > .3f; })
                && std::none_of(ordinaryPixels.begin(), ordinaryPixels.end(), [](float value) { return value < -.001f; }),
                "mesh translation under Rz(90) is positive Y upwards in raw pixel units");
        ok &= render(false, "same-frame native imported Motion redraw uses its retained predecessor")
            && expect(ordinaryPixels == readMotion(receipt.frame.nativeFrame), "same-frame raw Motion is byte-repeatable");

        // Direct physical-backend submissions must validate provenance before dispatch.
        const auto prepared = backend.prepare(receipt.source);
        if (!prepared.prepared) return false;
        const auto validInputs = receipt.runtimeInputs;
        for (const int corruption : {0,1,2})
        {
            auto inputs = validInputs;
            if (corruption == 2) inputs.previousMotion->deformation.reset();
            else
            {
                auto worlds = std::make_shared<Worlds>(*inputs.previousMotion->importedNodeWorldMatrices);
                if (corruption == 0) (*worlds)[0].world[12] += 0.125f;
                if (corruption == 1) (*worlds)[1].world[13] += 0.125f;
                inputs.previousMotion->importedNodeWorldMatrices = std::move(worlds);
            }
            const auto rejected = backend.render(receipt.source, receipt.evaluation.deformation,
                prepared.resources, request.width, request.height, inputs);
            ok &= expect(!rejected.rendered && !rejected.error.empty() && rejected.stats.dispatchCount == 0,
                "inconsistent mesh or ancestor world and missing retained snapshot refuse before GPU dispatch");
        }

        request.frame.frame = 2; request.operation.playback.timelineSeconds = 2.0 / 24.0;
        if (moving)
        {
            request.runtimeInputs.objectRotationDegrees[2] = 12.0f;
            request.runtimeInputs.objectScale = 0.85f;
            request.runtimeInputs.objectTranslationOffset = {0.025f,-0.05f,0.0f};
        }
        request.runtimeInputs.objectScale = 0;
        ok &= expect(!execution.executePreview(request, receipt, error) && !receipt.valid(),
            "failed native imported submission publishes no receipt");
        request.runtimeInputs.objectScale = moving ? 0.85f : 1.0f;
        if (!render(false, "retry combines prior palette with independently projected outer graph motion")) return false;
        const auto retryPixels = readMotion(receipt.frame.nativeFrame);
        ok &= expect(receipt.runtimeInputs.previousMotion->deformation->time().frame == 1,
            "failed submission preserves the last successful frame-one predecessor");
        ok &= render(true, "interleaved Export starts with exactly neutral imported Motion")
            && expect(!receipt.runtimeInputs.previousMotion, "Export cannot borrow Preview Motion history");
        ok &= render(false, "Preview redraw survives interleaved Export")
            && expect(retryPixels == readMotion(receipt.frame.nativeFrame), "Export leaves Preview raw Motion unchanged");
        request.frame.frame = 3; request.operation.playback.timelineSeconds = 3.0 / 24.0;
        if (!render(true, "Export has independently projected adjacent imported Motion")
            || !expect(receipt.runtimeInputs.previousMotion.has_value(), "Export retains its own adjacent predecessor")) return false;
        const auto exportPixels = readMotion(receipt.frame.nativeFrame);
        ok &= render(false, "Preview has independently projected adjacent imported Motion")
            && expect(receipt.runtimeInputs.previousMotion.has_value(), "Preview retains its own adjacent predecessor")
            && expect(exportPixels == readMotion(receipt.frame.nativeFrame), "equal Preview/Export histories produce equal raw Motion");
        for (int reset = 0; reset < 12; ++reset)
        {
            ++request.frame.frame;
            if (reset == 0) request.frame.frame += 10;
            if (reset == 1) request.frame.frame = 0;
            if (reset == 2) request.frame.rateNumerator = 30;
            if (reset == 3) request.frame.rateDenominator = 2;
            if (reset == 4) request.width = 80;
            if (reset == 5) request.height = 60;
            if (reset == 6) ++request.projectGeneration;
            if (reset == 7) ++request.helperGeneration;
            if (reset == 8) ++request.renderClipId;
            if (reset == 9) ++request.structuralRevision;
            if (reset == 10) ++request.operation.sourceStableId;
            if (reset == 11) ++request.operation.deformationStableId;
            request.operation.schedule = {request.operation.sourceStableId, request.operation.deformationStableId};
            request.operation.playback.timelineSeconds = static_cast<double>(request.frame.frame)
                * request.frame.rateDenominator / request.frame.rateNumerator;
            ok &= render(false, "seek, loop, rational rate, extent and source generation reset native imported Motion")
                && expect(!receipt.runtimeInputs.previousMotion, "reset never leaks a stale imported predecessor buffer");
        }
    }
    return ok;
}
} // namespace importedmotionfixture

std::array<int, 8> shaderStorageBindings(const arbitgl::GlFuncs& gl)
{
    std::array<int, 8> bindings {};
    for (unsigned index = 0; index < bindings.size(); ++index)
        gl.GetIntegeri_v(GL_SHADER_STORAGE_BUFFER_BINDING, index, &bindings[index]);
    return bindings;
}

std::uint64_t fnv1a64(const std::vector<std::uint8_t>& bytes)
{
    std::uint64_t value = 14695981039346656037ull;
    for (const auto byte : bytes)
    {
        value ^= byte;
        value *= 1099511628211ull;
    }
    return value;
}
} // namespace

int main()
{
    if (! glfwInit())
    {
        std::cerr << "SKIP: GLFW initialization failed\n";
        return 77;
    }
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    auto* window = glfwCreateWindow(96, 96, "opengl-deformation-test", nullptr, nullptr);
    if (window == nullptr)
    {
        std::cerr << "SKIP: hidden OpenGL 4.3 core context creation failed\n";
        glfwTerminate();
        return 77;
    }
    glfwMakeContextCurrent(window);

    arbitgl::GlFuncs gl;
    std::string missing;
    if (! arbitgl::loadGlFunctions(gl, missing) || ! arbitgl::loadGl43Functions(gl))
    {
        std::cerr << "SKIP: OpenGL 4.3 function loading failed: " << missing << '\n';
        glfwDestroyWindow(window);
        glfwTerminate();
        return 77;
    }

    bool ok = true;
    Fixture fixture;
    ok &= expect(fixture.asset != nullptr && fixture.clip != nullptr && fixture.source != nullptr,
                 "the physical deformation fixture must be admitted");
    auto firstSnapshot = fixture.snapshot(0);
    auto movedSnapshot = fixture.snapshot(30);
    ok &= expect(firstSnapshot != nullptr && movedSnapshot != nullptr,
                 "both exact deformation snapshots must evaluate");

    std::array<GLuint, 8> callerBuffers {};
    gl.GenBuffers(static_cast<int>(callerBuffers.size()), callerBuffers.data());
    std::array<std::uint32_t, 4> sentinel { 1, 2, 3, 4 };
    for (unsigned index = 0; index < callerBuffers.size(); ++index)
    {
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, callerBuffers[index]);
        gl.BufferData(GL_SHADER_STORAGE_BUFFER,
                      static_cast<std::ptrdiff_t>(sizeof(sentinel)),
                      sentinel.data(), GL_STATIC_DRAW);
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, index, callerBuffers[index]);
    }
    gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, callerBuffers[3]);
    const auto callerBindings = shaderStorageBindings(gl);
    GLint callerGenericBinding = 0;
    glGetIntegerv(GL_SHADER_STORAGE_BUFFER_BINDING, &callerGenericBinding);

    auto& backend = arbitgpu::nativeDeformationBackend();
    ok &= expect(backend.supportsRetainedDeformationMotion(),
                 "OpenGL deformation explicitly supports retained prior-pose Motion");
    ok &= expect(backend.supportsRetainedNoteMotion(),
                 "OpenGL deformation explicitly supports retained stable-note Motion");
    std::string extractionError;
    const bool beatSync=importedanimationbeatfixture::verify(backend,
        [&](const auto& frame) { return readRgba8(gl, frame); }, extractionError);
    if (!beatSync) std::cerr << "imported Beat Sync refusal: "
        << (extractionError.empty() ? std::string("no refusal text was set") : extractionError) << '\n';
    ok &= expect(beatSync,"imported Beat Sync matches independent native joint geometry across tempo changes, seeks and export");
    extractionError.clear();
    const bool extracted=verifyAnimatedGeometryExtraction(extractionError);
    if (!extracted) std::cerr << "imported Geometry3D extraction refusal: "
        << (extractionError.empty() ? std::string("no refusal text was set") : extractionError) << '\n';
    ok &= expect(extracted,"imported animation reaches retained Geometry3D through the native GPU path");
    const bool multiobject = multiobjectanimationfixture::verify(backend,
        [&](const auto& frame) { return readRgba8(gl, frame); }, extractionError);
    if (!multiobject) std::cerr << extractionError << '\n';
    ok &= expect(multiobject, "multi-object imported animation preserves GPU preview, export and seek");
    const bool rigidScene = multiobjectanimationfixture::verify(backend,
        [&](const auto& frame) { return readRgba8(gl, frame); }, extractionError, true);
    if (!rigidScene) std::cerr << extractionError << '\n';
    ok &= expect(rigidScene, "rigid-only parent animation is selectable and rendered on OpenGL");
    for (const bool rigid : {false, true}) {
        const bool repeated = multiobjectanimationfixture::verify(backend,
            [&](const auto& frame) { return readRgba8(gl, frame); }, extractionError, rigid, true);
        if (!repeated) std::cerr << extractionError << '\n';
        ok &= expect(repeated, "repeated imported nodes retain independent whole-scene poses on OpenGL");
    }
    for (unsigned target : {1u,2u}) {
        const bool sceneOnly=multiobjectanimationfixture::verifySceneOnly(backend,
            [&](const auto& frame) { return readRgba8(gl,frame); },target,extractionError);
        if (!sceneOnly) std::cerr << extractionError << '\n';
        ok &= expect(sceneOnly,"camera/light-only clips render without mesh deformation on OpenGL");
    }
    ok &= expect(verifyDeformationMotion(backend),"skin and morph native Motion fixtures");
    std::string compositionError;
    ok &= expect(videohelper::fixture3d::verifyNativeComposition(backend,
        [&](const auto& frame) {return readRgba8(gl,frame);},compositionError),
        "typed full-scene composition, predecessor Motion and normal/tangent reference");
    if (!compositionError.empty()) std::cerr << compositionError << '\n';
    const bool productionJoin=videohelper::fixture3d::verifyNativeCompositionProductionJoin(backend,
        [&](const auto& frame) {return readRgba8(gl,frame);},compositionError);
    if (!productionJoin) std::cerr << "production composition join refusal: "
        << (compositionError.empty() ? std::string("no refusal text was set") : compositionError) << '\n';
    ok &= expect(productionJoin,
        "actual production composition provenance, epochs, independent histories and bounded escaped receipts");
    const bool parentedProduction=videohelper::fixture3d::verifyParentedImportedProductionJoin(backend,
        [&](const auto& frame) {return readRgba8(gl,frame);},compositionError);
    if (!parentedProduction) std::cerr << compositionError << '\n';
    ok &= expect(parentedProduction,"production-issued parent maps, real deformation, native Color/Motion and complete-root conservation");
    const bool selectedCamera=videohelper::fixture3d::verifyImportedCameraSelectionProductionJoin(backend,
        [&](const auto& frame) {return readRgba8(gl,frame);},compositionError);
    if (!selectedCamera) std::cerr << compositionError << '\n';
    ok &= expect(selectedCamera,"exact source camera selection, sampled ancestry, override and Preview/Export native join");
    const bool selectedSkin=videohelper::fixture3d::verifySelectedDeformationCameraReceipt(backend,
        [&](const auto& frame) {return readRgba8(gl,frame);},compositionError);
    if (!selectedSkin) std::cerr << compositionError << '\n';
    ok &= expect(selectedSkin,"genuine skin selected-camera receipt rejects copied camera, selector and native-frame mutations");
    for (const unsigned target : {1u,2u})
    {
        const bool composed=videohelper::fixture3d::verifyCameraLightCompositionProductionJoin(backend,
            [&](const auto& frame) {return readRgba8(gl,frame);},target,compositionError);
        if (!composed) std::cerr << compositionError << '\n';
        ok &= expect(composed,"authentic camera/light-only static composition on OpenGL");
    }
    ok &= expect(backend.supportsRetainedImportedObjectMotion(),
                 "OpenGL explicitly supports validated retained imported-object Motion");
    ok &= expect(importedmotionfixture::verify(backend), "native imported-object and ancestor raw Motion fixtures");
    const auto info = backend.info();
    ok &= expect(info.available && info.compute && info.backend == "opengl",
                 "the production deformation backend must report live OpenGL compute");
    auto preparation = backend.prepare(fixture.source);
    if (! preparation.prepared)
        std::cerr << "OpenGL deformation preparation error: " << preparation.error << '\n';
    ok &= expect(preparation.prepared && preparation.resources != nullptr,
                 "the exact textured morph scene must prepare on OpenGL");
    ok &= expect(shaderStorageBindings(gl) == callerBindings,
                 "deformation preparation must preserve caller indexed SSBO bindings");
    GLint afterPrepareGeneric = 0;
    glGetIntegerv(GL_SHADER_STORAGE_BUFFER_BINDING, &afterPrepareGeneric);
    ok &= expect(afterPrepareGeneric == callerGenericBinding,
                 "deformation preparation must preserve caller generic SSBO binding");

    auto first = backend.render(fixture.source, firstSnapshot,
                                preparation.resources, 96, 96);
    auto repeated = backend.render(fixture.source, firstSnapshot,
                                   preparation.resources, 96, 96);
    auto moved = backend.render(fixture.source, movedSnapshot,
                                preparation.resources, 96, 96);
    ok &= expect(first.deformedVertices.size()==24u*8u
                 && first.deformedVertices==repeated.deformedVertices
                 && moved.deformedVertices.size()==first.deformedVertices.size(),
                 "GPU geometry extraction retains topology and repeats the same frame exactly");
    if (first.deformedVertices.size()==24u*8u && moved.deformedVertices.size()==24u*8u)
        for (std::size_t vertex=0;vertex<24;++vertex)
        {
            ok &= expect(std::abs(moved.deformedVertices[vertex*8]-first.deformedVertices[vertex*8]-0.75f)<0.00001f,
                         "retained geometry contains the GPU evaluated morph displacement");
            ok &= expect(first.deformedVertices[vertex*8+6]==fixture.sceneValue.vertices[vertex].uv.x
                         && first.deformedVertices[vertex*8+7]==fixture.sceneValue.vertices[vertex].uv.y,
                         "GPU extraction uses the full tangent and color vertex stride without corrupting UVs");
        }
    auto sought=backend.render(fixture.source,firstSnapshot,preparation.resources,96,96);
    ok &= expect(sought.deformedVertices==first.deformedVertices,
                 "seeking backwards resets extracted geometry to the exact original GPU frame");
    arbitgpu::NativeDeformationRuntimeInputs mutedMorphInputs;
    mutedMorphInputs.timeSeconds = 0.0f;
    mutedMorphInputs.morphWeight = 0.0f;
    auto mutedMorph = backend.render(fixture.source, movedSnapshot,
                                     preparation.resources, 96, 96, mutedMorphInputs);
    arbitgpu::NativeDeformationRuntimeInputs sceneRuntime;
    sceneRuntime.objectTranslationOffset[0] = 0.5f;
    sceneRuntime.cameraTranslationOffset[2] = 0.25f;
    auto movedScene = backend.render(
        fixture.source, firstSnapshot, preparation.resources, 96, 96, sceneRuntime);
    arbitgpu::NativeDeformationRuntimeInputs rotatedScene;
    rotatedScene.objectRotationDegrees[1] = 35.0f;
    auto rotated = backend.render(
        fixture.source, firstSnapshot, preparation.resources, 96, 96, rotatedScene);
    arbitgpu::NativeDeformationRuntimeInputs scaledScene;
    scaledScene.objectScale = 0.7f;
    auto scaled = backend.render(
        fixture.source, firstSnapshot, preparation.resources, 96, 96, scaledScene);
    arbitgpu::NativeDeformationRuntimeInputs graphCamera;
    graphCamera.cameraOverride = fixture.source->scene->cameras.front();
    graphCamera.cameraOverride->transform.translation.x += 0.75f;
    auto cameraOverride = backend.render(
        fixture.source, firstSnapshot, preparation.resources, 96, 96, graphCamera);
    auto invalidGraphCamera = graphCamera;
    invalidGraphCamera.cameraOverride->nearPlane = -1.0f;
    auto rejectedCameraOverride = backend.render(
        fixture.source, firstSnapshot, preparation.resources, 96, 96, invalidGraphCamera);
    if (! first.rendered) std::cerr << "first render error: " << first.error << '\n';
    if (! repeated.rendered) std::cerr << "repeated render error: " << repeated.error << '\n';
    if (! moved.rendered) std::cerr << "moved render error: " << moved.error << '\n';
    ok &= expect(first.rendered && repeated.rendered && moved.rendered
                     && mutedMorph.rendered && movedScene.rendered
                     && rotated.rendered && scaled.rendered && cameraOverride.rendered,
                 "native OpenGL deformation must render exact snapshots");
    // validFixtureRuntimeInputs rejects the nearPlane=-1 camera, and the OpenGL fixture
    // render reports that rejection with this exact message (backend_opengl.cpp).
    if (rejectedCameraOverride.rendered || rejectedCameraOverride.error
            != "OpenGL fixture scene modulation is non-finite or out of bounds")
        std::cerr << "rejected camera override error: " << rejectedCameraOverride.error << '\n';
    ok &= expect(!rejectedCameraOverride.rendered
                     && rejectedCameraOverride.error
                          == "OpenGL fixture scene modulation is non-finite or out of bounds",
                 "invalid graph camera override must fail closed");
    ok &= expect(first.frame && first.frame->depthImageHandle() != 0
                     && first.frame->depthTextureViewHandle() != 0,
                 "Render 3D must publish its native depth resource");
    ok &= expect(first.stats.dispatchCount == 1 && first.stats.drawCount == 1
                     && first.stats.reusedStaticResources,
                 "each render must report one GPU deformation dispatch and draw");
    ok &= expect(shaderStorageBindings(gl) == callerBindings,
                 "deformation rendering must restore caller indexed SSBO bindings");
    GLint afterRenderGeneric = 0;
    glGetIntegerv(GL_SHADER_STORAGE_BUFFER_BINDING, &afterRenderGeneric);
    ok &= expect(afterRenderGeneric == callerGenericBinding,
                 "deformation rendering must restore caller generic SSBO binding");

    std::uint64_t firstHash = 0;
    std::uint64_t movedHash = 0;
    if (first.frame && repeated.frame && moved.frame && rotated.frame && scaled.frame
        && cameraOverride.frame)
    {
        const auto firstPixels = readRgba8(gl, first.frame);
        const auto repeatedPixels = readRgba8(gl, repeated.frame);
        const auto movedPixels = readRgba8(gl, moved.frame);
        const auto mutedMorphPixels = readRgba8(gl, mutedMorph.frame);
        const auto movedScenePixels = readRgba8(gl, movedScene.frame);
        const auto rotatedPixels = readRgba8(gl, rotated.frame);
        const auto scaledPixels = readRgba8(gl, scaled.frame);
        const auto cameraOverridePixels = readRgba8(gl, cameraOverride.frame);
        firstHash = fnv1a64(firstPixels);
        movedHash = fnv1a64(movedPixels);
        ok &= expect(firstPixels == repeatedPixels,
                     "equal immutable snapshots must produce identical pixels");
        ok &= expect(firstPixels != movedPixels,
                     "a changed morph weight must change rendered pixels");
        ok &= expect(firstPixels == mutedMorphPixels,
                     "zero runtime morph weight must restore the imported base shape");
        ok &= expect(firstPixels != movedScenePixels,
                     "object and camera runtime offsets must change rendered pixels");
        ok &= expect(firstPixels != rotatedPixels,
                     "runtime object rotation must change rendered pixels");
        ok &= expect(firstPixels != scaledPixels,
                     "runtime object scale must change rendered pixels");
        ok &= expect(firstPixels != cameraOverridePixels,
                     "graph camera override must change rendered pixels");
        const auto firstDepth = readDepth32f(first.frame);
        const auto repeatedDepth = readDepth32f(repeated.frame);
        const auto movedDepth = readDepth32f(moved.frame);
        // An empty readback is a failed read, not a depth result. Require it explicitly,
        // because two empty vectors would otherwise compare equal.
        ok &= expect(!firstDepth.empty() && !repeatedDepth.empty() && !movedDepth.empty(),
                     "published depth readback succeeds");
        ok &= expect(firstDepth == repeatedDepth,
                     "equal immutable snapshots must produce identical depth");
        ok &= expect(firstDepth != movedDepth,
                     "a changed morph weight must change published depth");
    }

    auto material = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>();
    material->backend = arbitgpu::NativeFixtureMaterialBackend::OpenGl;
    material->object = videohelper::fixture3d::kCubeObjectId;
    material->bindingDigest = "opengl-deformed-time-material-binding";
    material->programIdentity = "opengl-deformed-time-material-program";
    material->baseColorSource
        = arbitgpu::NativeFixtureSurfaceMaterialProgram::BaseColorSource::TimeLinearMix;
    material->parameters.baseColorMetallic = { 1.0f, 0.0f, 0.0f, 0.0f };
    material->parameters.emissionRoughness = { 0.1f, 0.0f, 0.0f, 0.5f };
    material->parameters.normalOpacity = { 0.0f, 0.0f, 1.0f, 1.0f };
    material->parameters.transmissionIorClearcoat = { 0.0f, 1.5f, 0.0f, 0.0f };
    material->parameters.identifiers[0] = videohelper::fixture3d::kCubeMaterialId.value;
    material->timeMixEndColor = { 0.0f, 0.0f, 1.0f };
    auto materialPreparation = backend.prepare(fixture.source, material);
    if (! materialPreparation.prepared)
        std::cerr << "OpenGL deformed material preparation error: "
                  << materialPreparation.error << '\n';
    ok &= expect(materialPreparation.prepared && materialPreparation.resources != nullptr,
                 "authored Surface Material must prepare with deformation resources");
    arbitgpu::NativeDeformationRuntimeInputs materialStartInputs;
    materialStartInputs.timeSeconds = 0.0f;
    auto materialStart = backend.render(
        fixture.source, firstSnapshot, materialPreparation.resources, 96, 96,
        materialStartInputs);
    arbitgpu::NativeDeformationRuntimeInputs materialEndInputs;
    materialEndInputs.timeSeconds = 1.0f;
    auto materialEnd = backend.render(
        fixture.source, firstSnapshot, materialPreparation.resources, 96, 96,
        materialEndInputs);
    arbitgpu::NativeDeformationRuntimeInputs darkEmission;
    darkEmission.emissionGain = 0.0f;
    auto materialDark = backend.render(
        fixture.source, firstSnapshot, materialPreparation.resources, 96, 96, darkEmission);
    if (! materialStart.rendered)
        std::cerr << "deformed material start error: " << materialStart.error << '\n';
    if (! materialEnd.rendered)
        std::cerr << "deformed material end error: " << materialEnd.error << '\n';
    ok &= expect(materialStart.rendered && materialEnd.rendered && materialDark.rendered,
                 "deformed geometry must render the authored material at exact times");
    if (materialStart.frame && materialEnd.frame)
    {
        const auto startPixels = readRgba8(gl, materialStart.frame);
        const auto endPixels = readRgba8(gl, materialEnd.frame);
        ok &= expect(startPixels != endPixels,
                     "the authored time-dependent material must change deformed pixels");
        ok &= expect(startPixels != readRgba8(gl, materialDark.frame),
                     "runtime emission gain must change deformed material pixels");
    }

    auto wrongOwner = std::make_shared<arbitgpu::NativeDeformationScene>(*fixture.source);
    wrongOwner->deformationStableId += 1;
    const auto rejected = backend.render(wrongOwner, firstSnapshot,
                                         preparation.resources, 96, 96);
    ok &= expect(! rejected.rendered && ! rejected.error.empty(),
                 "prepared resources must reject a different immutable owner");

    first.frame.reset();
    repeated.frame.reset();
    moved.frame.reset();
    mutedMorph.frame.reset();
    movedScene.frame.reset();
    rotated.frame.reset();
    scaled.frame.reset();
    materialStart.frame.reset();
    materialEnd.frame.reset();
    materialDark.frame.reset();
    materialPreparation.resources.reset();
    preparation.resources.reset();
    gl.DeleteBuffers(static_cast<int>(callerBuffers.size()), callerBuffers.data());
    if (ok)
    {
        std::cout << "OpenGL animation deformation backend: PASS; device="
                  << info.device << "; base-rgba8-fnv1a64=" << std::hex << firstHash
                  << "; morphed-rgba8-fnv1a64=" << movedHash << std::dec << '\n';
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    return ok ? 0 : 1;
}
