#include "VisualAnimationDeformationEvaluation.h"
#include "gpu_backend/backend.h"
#include "support/fixture_scene.h"
#include "support/native_deformation_composition_fixture.h"
#include "native_composition_production_join_fixture.h"
#include "animated_geometry_fixture.h"
#include "imported_animation_beat_fixture.h"
#include "multiobject_animation_fixture.h"
#include "glb_native_render_seam.h"
#include "imported_animated_scene_payload_execution.h"
#include "../../shared/VisualStarterModelAssets.h"

#define SOKOL_METAL
#include "sokol_gfx.h"
#import <Metal/Metal.h>

#include <nlohmann/json.hpp>

#include <array>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <limits>
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

    Fixture()
    {
        for (std::size_t vertex = 0; vertex < 24; ++vertex)
            morphPositions[vertex * 3] = 1.0f;
        morphTarget = { targetIds[0], morphPositions.data(), morphPositions.size(),
                        nullptr, 0, nullptr, 0 };
        mesh = { MeshId { 501 }, 24, {}, nullptr, 0, &morphTarget, 1 };
        assetView = { nullptr, 0, &mesh, 1 };
        std::string error;
        asset = DeformationAsset::create(assetView, {}, error);
        if (asset == nullptr)
            std::cerr << "fixture deformation asset error: " << error << '\n';
        track = { animation::TrackId { 12 }, animation::TargetId { 1002 },
                  animation::Channel::MorphWeights, animation::Interpolation::Linear,
                  keyTimes.data(), morphValues.data(), keyTimes.size(), morphValues.size(), 1 };
        clipView = { animation::ClipId { 701 }, 1.0, &track, 1 };
        clip = animation::Clip::create(clipView, {}, error);
        if (clip == nullptr)
            std::cerr << "fixture animation clip error: " << error << '\n';
        morphBinding = { animation::TargetId { 1002 }, mesh.id,
                         targetIds.data(), targetIds.size() };
        bindings = { nullptr, 0, &morphBinding, 1 };
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
        mutableSource->retainDeformedGeometry = true;
        source = std::move(mutableSource);
    }

    std::shared_ptr<const AnimationDeformationSnapshot> snapshot(std::int64_t frame) const
    {
        std::string error;
        const AnimationDeformationRequest request {
            RationalFrameTime { frame, 30, 1 }, 77,
            animation::Playback::Clamp, std::nullopt
        };
        auto result = evaluateAnimationDeformation(*asset, *clip, bindings,
                                                   request, {}, error);
        if (result == nullptr)
            std::cerr << "fixture snapshot error: " << error << '\n';
        return result;
    }
};

std::vector<std::uint8_t> readBgra8(
    const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& frame)
{
    const sg_image image { static_cast<std::uint32_t>(frame->colorImageHandle()) };
    const auto native = sg_mtl_query_image_info(image);
    id<MTLTexture> texture = (__bridge id<MTLTexture>) native.tex[native.active_slot];
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>) sg_mtl_command_queue();
    const auto byteCount = static_cast<std::size_t>(frame->width()) * frame->height() * 4u;
    id<MTLBuffer> readback = [texture.device newBufferWithLength:byteCount
                                                        options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    [blit copyFromTexture:texture
              sourceSlice:0
              sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(frame->width(), frame->height(), 1)
                 toBuffer:readback
        destinationOffset:0
   destinationBytesPerRow:frame->width() * 4u
 destinationBytesPerImage:byteCount];
    [blit endEncoding];
    [command commit];
    [command waitUntilCompleted];
    std::vector<std::uint8_t> pixels(byteCount);
    std::memcpy(pixels.data(), readback.contents, byteCount);
    return pixels;
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

std::string encodeBase64(const std::vector<std::uint8_t>& bytes)
{
    constexpr char alphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string encoded;
    encoded.reserve((bytes.size() + 2u) / 3u * 4u);
    for (std::size_t offset = 0; offset < bytes.size(); offset += 3u)
    {
        const auto remaining = bytes.size() - offset;
        const auto value = (static_cast<std::uint32_t>(bytes[offset]) << 16u)
            | (remaining > 1u ? static_cast<std::uint32_t>(bytes[offset + 1u]) << 8u : 0u)
            | (remaining > 2u ? bytes[offset + 2u] : 0u);
        encoded += alphabet[(value >> 18u) & 63u];
        encoded += alphabet[(value >> 12u) & 63u];
        encoded += remaining > 1u ? alphabet[(value >> 6u) & 63u] : '=';
        encoded += remaining > 2u ? alphabet[value & 63u] : '=';
    }
    return encoded;
}

void reportStarterPasses(const char* label,
    const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& frame,
    std::uint32_t draws)
{
    std::cerr << "animated-triangle " << label << " draws=" << draws;
    if (!frame)
    {
        std::cerr << " frame=absent\n";
        return;
    }
    const auto extent = static_cast<std::size_t>(frame->width()) * frame->height();
    const auto passName = [](renderpassoutput::Output output)
    {
        switch (output)
        {
            case renderpassoutput::Output::Color: return "color";
            case renderpassoutput::Output::Mask: return "mask";
            case renderpassoutput::Output::ObjectId: return "object-id";
            case renderpassoutput::Output::Depth: return "depth";
            default: return "unexpected";
        }
    };
    const auto formatName = [](arbitgpu::NativeTexturePixelFormat format)
    {
        using Format = arbitgpu::NativeTexturePixelFormat;
        switch (format)
        {
            case Format::Rgba8Unorm: return "rgba8";
            case Format::Bgra8Unorm: return "bgra8";
            case Format::Rgba16Float: return "rgba16f";
            case Format::R8Unorm: return "r8";
            case Format::R32Uint: return "r32ui";
            case Format::R32Float: return "r32f";
            default: return "unexpected";
        }
    };
    for (const auto output : {renderpassoutput::Output::Color,
                              renderpassoutput::Output::Mask,
                              renderpassoutput::Output::ObjectId,
                              renderpassoutput::Output::Depth})
    {
        arbitgpu::NativeRawPassPixels pixels;
        std::string error;
        if (!frame->readRawPass(output, pixels, error))
        {
            std::cerr << " " << passName(output) << "=error:" << error;
            continue;
        }
        const auto scalarBytes = arbitgpu::rawPassScalarBytes(pixels.format);
        const auto pixelBytes = scalarBytes * arbitgpu::rawPassChannels(pixels.format);
        const bool validLayout = arbitgpu::rawPassFormatMatches(output, pixels.format)
            && pixels.width == frame->width() && pixels.height == frame->height()
            && pixelBytes != 0 && pixels.bytes.size() == extent * pixelBytes;
        std::cerr << " " << passName(output) << "-format=" << formatName(pixels.format);
        if (!validLayout)
        {
            std::cerr << " " << passName(output) << "=invalid-layout bytes="
                      << pixels.bytes.size();
            continue;
        }
        std::size_t foreground = 0;
        if (output == renderpassoutput::Output::Color)
        {
            const auto colorBytes = 3u * scalarBytes;
            for (std::size_t i = 0; i < extent; ++i)
                foreground += !std::equal(pixels.bytes.begin() + pixelBytes * i,
                    pixels.bytes.begin() + pixelBytes * i + colorBytes, pixels.bytes.begin());
        }
        else if (output == renderpassoutput::Output::Mask)
        {
            foreground = std::count_if(pixels.bytes.begin(), pixels.bytes.end(),
                [](std::uint8_t value) { return value != 0; });
        }
        else
        {
            for (std::size_t i = 0; i < extent; ++i)
            {
                if (output == renderpassoutput::Output::ObjectId)
                {
                    std::uint32_t id = 0;
                    std::memcpy(&id, pixels.bytes.data() + pixelBytes * i, sizeof(id));
                    foreground += id != 0;
                }
                else if (output == renderpassoutput::Output::Depth)
                {
                    float depth = 1.0f;
                    std::memcpy(&depth, pixels.bytes.data() + pixelBytes * i, sizeof(depth));
                    foreground += depth < 1.0f;
                }
            }
        }
        std::cerr << " " << passName(output) << "=" << foreground
                  << "/" << extent << " bytes=" << pixels.bytes.size();
    }
    std::cerr << '\n';
}

struct StarterShape
{
    std::vector<std::uint8_t> mask;
    std::vector<std::uint8_t> color;
    bool visible = false;
};

StarterShape observeStarterShape(
    const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& frame,
    const char* label)
{
    StarterShape result;
    if (!frame) return result;
    arbitgpu::NativeRawPassPixels mask, objectIds;
    std::string error;
    if (!frame->readRawPass(renderpassoutput::Output::Mask, mask, error)
        || !frame->readRawPass(renderpassoutput::Output::ObjectId, objectIds, error))
    {
        std::cerr << "animated-triangle " << label << " shape readback error=" << error << '\n';
        return result;
    }
    const auto width = static_cast<std::size_t>(frame->width());
    const auto height = static_cast<std::size_t>(frame->height());
    const auto extent = width * height;
    result.color = readBgra8(frame);
    if (mask.format != arbitgpu::NativeTexturePixelFormat::R8Unorm
        || objectIds.format != arbitgpu::NativeTexturePixelFormat::R32Uint
        || mask.bytes.size() != extent || objectIds.bytes.size() != extent * sizeof(std::uint32_t)
        || result.color.size() != extent * 4u)
    {
        std::cerr << "animated-triangle " << label << " shape attachment layout changed\n";
        return result;
    }
    result.mask = std::move(mask.bytes);
    std::size_t covered = 0, colored = 0;
    std::size_t minX = width, minY = height, maxX = 0, maxY = 0;
    bool identitiesAgree = true;
    for (std::size_t index = 0; index < extent; ++index)
    {
        std::uint32_t objectId = 0;
        std::memcpy(&objectId, objectIds.bytes.data() + index * sizeof(objectId), sizeof(objectId));
        const bool occupied = result.mask[index] != 0;
        identitiesAgree &= occupied == (objectId != 0);
        if (!occupied) continue;
        ++covered;
        const auto x = index % width, y = index / width;
        minX = std::min(minX, x); maxX = std::max(maxX, x);
        minY = std::min(minY, y); maxY = std::max(maxY, y);
        const auto* pixel = result.color.data() + index * 4u;
        colored += pixel[0] != result.color[0] || pixel[1] != result.color[1]
            || pixel[2] != result.color[2];
    }
    result.visible = identitiesAgree && covered >= 64 && covered < extent / 2u
        && maxX >= minX + 8u && maxY >= minY + 8u
        && minX > 0 && maxX + 1u < width && maxY + 1u < height
        && colored >= covered / 2u;
    std::cerr << "animated-triangle " << label << " authored-shape=" << result.visible
              << " pixels=" << covered << " bounds=" << minX << ',' << minY << '-'
              << maxX << ',' << maxY << " object-ids-agree=" << identitiesAgree
              << " colored=" << colored << '\n';
    return result;
}

bool probeAnimatedTriangle(arbitgpu::NativeDeformationBackend& deformationBackend)
{
    constexpr std::uint32_t width = 192, height = 108;
    const std::vector<std::uint8_t> bytes(visualstartermodel::kAnimatedTriangleGlb.begin(),
        visualstartermodel::kAnimatedTriangleGlb.end());
    std::string error;
    auto& sceneBackend = arbitgpu::nativeFixtureSceneBackend();
    videohelper::gltf::GlbNativeRenderSeam staticSeam(sceneBackend);
    videohelper::gltf::GlbNativeSceneAdmission staticAdmission;
    if (staticSeam.admit(bytes, 0u, staticAdmission, error))
    {
        std::cerr << "animated-triangle static-admission objects="
                  << staticAdmission.scene->objectCount << " vertices="
                  << staticAdmission.scene->vertexCount << " indices="
                  << staticAdmission.scene->indexCount << " cameras="
                  << staticAdmission.scene->cameraCount << '\n';
        videohelper::gltf::GlbNativeFrameReceipt raw;
        if (staticSeam.renderPreview(staticAdmission, {width, height}, raw, error))
            reportStarterPasses("static-raw", raw.rendered.nativeFrame, raw.rendered.stats.drawCount);
        else
            std::cerr << "animated-triangle static-raw error=" << error << '\n';
        arbitgpu::NativeFixtureSceneRuntimeInputs composite;
        composite.passComposite = renderpasscomposite::Parameters {};
        composite.passComposite->mode = renderpasscomposite::Mode::DepthFog;
        videohelper::gltf::GlbNativeFrameReceipt composed;
        if (staticSeam.renderPreview(staticAdmission, {width, height}, composite, composed, error))
            reportStarterPasses("static-depth-fog", composed.rendered.nativeFrame,
                composed.rendered.stats.drawCount);
        else
            std::cerr << "animated-triangle static-depth-fog error=" << error << '\n';
    }
    else
        std::cerr << "animated-triangle static-admission error=" << error << '\n';

    videohelper::modelpayload::Store store;
    const visualanimationimport::ExactContentAssetKey key {
        "metal-animated-triangle", 1u, std::string(visualstartermodel::kContentSha256),
        "model/gltf-binary", bytes.size()};
    if (!store.begin("metal-animated-triangle", key)
        || !store.appendBase64("metal-animated-triangle", 0u, encodeBase64(bytes))
        || !store.commit("metal-animated-triangle"))
    {
        std::cerr << "animated-triangle payload-publication failed\n";
        return false;
    }
    videohelper::modelpayload::ImportedAnimatedScenePayloadExecution execution(
        store, deformationBackend, sceneBackend);
    videohelper::modelpayload::ImportedAnimatedSceneRequest request;
    request.operation.sourceStableId = 11;
    request.operation.deformationStableId = 12;
    request.operation.schedule = {11, 12};
    request.operation.asset = key;
    request.operation.sceneIndex = 0u;
    request.operation.meshStableId = 1u;
    request.operation.animationClipStableId = 1u;
    request.operation.clipName = std::string(visualstartermodel::kClipName);
    request.frame = {0, 24, 1};
    request.structuralRevision = 1;
    request.width = width;
    request.height = height;
    std::array<StarterShape, 2> shapes;
    std::array<float, 2> sampledWeights {};
    bool ok = true;
    for (const auto frame : {0, 12})
    {
        request.frame.frame = frame;
        request.operation.playback.timelineSeconds = static_cast<double>(frame) / 24.0;
        videohelper::modelpayload::ImportedAnimatedSceneReceipt raw;
        if (execution.executePreview(request, raw, error))
        {
            std::cerr << "animated-triangle animated-admission-" << frame
                      << " objects=" << raw.source->scene->objectCount
                      << " vertices=" << raw.source->scene->vertexCount
                      << " indices=" << raw.source->scene->indexCount
                      << " cameras=" << raw.source->scene->cameraCount << '\n';
            reportStarterPasses(frame == 0 ? "animated-raw-0" : "animated-raw-12",
                raw.frame.nativeFrame, raw.frame.stats.drawCount);
            const auto label = frame == 0 ? "animated-raw-0" : "animated-raw-12";
            auto& shape = shapes[frame == 0 ? 0 : 1];
            shape = observeStarterShape(raw.frame.nativeFrame, label);
            ok &= expect(raw.frame.stats.drawCount > 0 && shape.visible,
                "bundled animated triangle must have an identified native Metal shape");
            const auto& draw = raw.source->draws.empty() ? raw.source : raw.source->draws.front();
            arbitgpu::NativeDeformationFrameData sampled;
            const bool sampledPose = raw.evaluation.deformation
                && arbitgpu::prepareNativeDeformationFrame(
                    *draw, *raw.evaluation.deformation, sampled, error)
                && sampled.morphTargetCount == 1u;
            ok &= expect(sampledPose,
                "bundled animated triangle must sample its one authored morph target");
            if (sampledPose) sampledWeights[frame == 0 ? 0 : 1] = sampled.morphWeights[0];
        }
        else
        {
            std::cerr << "animated-triangle animated-raw-" << frame << " error=" << error << '\n';
            ok = false;
        }
    }
    std::cerr << "animated-triangle sampled-morph-weights=" << sampledWeights[0]
              << ',' << sampledWeights[1] << '\n';
    ok &= expect(std::abs(sampledWeights[0]) < 0.0001f
        && std::abs(sampledWeights[1] - 0.5f) < 0.0001f,
        "bundled animated triangle must sample the authored 0 and 0.5 second morph keys");
    ok &= expect(shapes[0].visible && shapes[1].visible
        && (shapes[0].mask != shapes[1].mask || shapes[0].color != shapes[1].color),
        "bundled animated triangle must change native Metal pixels at 0 and 12/24 seconds");
    request.frame.frame = 0;
    request.runtimeInputs.passComposite = renderpasscomposite::Parameters {};
    request.runtimeInputs.passComposite->mode = renderpasscomposite::Mode::DepthFog;
    videohelper::modelpayload::ImportedAnimatedSceneReceipt composed;
    if (execution.executePreview(request, composed, error))
        reportStarterPasses("animated-depth-fog", composed.frame.nativeFrame,
            composed.frame.stats.drawCount);
    else
        std::cerr << "animated-triangle animated-depth-fog error=" << error << '\n';
    return ok;
}
void appendMotionU16 (std::vector<std::uint8_t>& bytes, std::uint16_t value)
{
    bytes.push_back (static_cast<std::uint8_t> (value));
    bytes.push_back (static_cast<std::uint8_t> (value >> 8u));
}

void appendMotionU32 (std::vector<std::uint8_t>& bytes, std::uint32_t value)
{
    bytes.push_back (static_cast<std::uint8_t> (value));
    bytes.push_back (static_cast<std::uint8_t> (value >> 8u));
    bytes.push_back (static_cast<std::uint8_t> (value >> 16u));
    bytes.push_back (static_cast<std::uint8_t> (value >> 24u));
}

void appendMotionFloat (std::vector<std::uint8_t>& bytes, float value)
{
    std::uint32_t encoded = 0;
    std::memcpy (&encoded, &value, sizeof (encoded));
    appendMotionU32 (bytes, encoded);
}

std::vector<std::uint8_t> makeMotionGlb (nlohmann::json root,
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
    appendMotionU32 (bytes, 0x46546c67u);
    appendMotionU32 (bytes, 2u);
    appendMotionU32 (bytes, total);
    appendMotionU32 (bytes, static_cast<std::uint32_t> (text.size()));
    appendMotionU32 (bytes, 0x4e4f534au);
    bytes.insert (bytes.end(), text.begin(), text.end());
    appendMotionU32 (bytes, static_cast<std::uint32_t> (bin.size()));
    appendMotionU32 (bytes, 0x004e4942u);
    bytes.insert (bytes.end(), bin.begin(), bin.end());
    return bytes;
}

std::vector<std::uint8_t> motionTriangleBin()
{
    std::vector<std::uint8_t> bin;
    for (const auto value : {-0.5f, -0.5f, 0.0f,
                              0.5f, -0.5f, 0.0f,
                              0.0f,  0.5f, 0.0f})
        appendMotionFloat (bin, value);
    for (int vertex = 0; vertex < 3; ++vertex)
        for (const auto value : {0.0f, 0.0f, 1.0f})
            appendMotionFloat (bin, value);
    for (const auto value : {0.0f, 0.0f, 1.0f, 0.0f, 0.5f, 1.0f})
        appendMotionFloat (bin, value);
    appendMotionU16 (bin, 0);
    appendMotionU16 (bin, 1);
    appendMotionU16 (bin, 2);
    return bin;
}

nlohmann::json motionTriangleRoot()
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
    auto root = motionTriangleRoot();
    auto bin = motionTriangleBin();
    bin.resize(104, 0);
    for (const auto time : {0.0f, 1.0f}) appendMotionFloat(bin, time);
    const std::array<float, 3> first = parentTarget
        ? std::array<float, 3>{0.1f, 0.2f, 0.0f}
        : std::array<float, 3>{0.25f, -0.125f, 0.0f};
    for (int key = 0; key < 2; ++key)
        for (std::size_t axis = 0; axis < 3; ++axis)
            appendMotionFloat(bin, first[axis] + (moving && axis == 0 ? 0.5f * key : 0.0f));
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
    return makeMotionGlb(std::move(root), std::move(bin));
}

std::array<double,2> importedMotionVertex(bool parentTarget, bool moving, double seconds,
    const std::array<double,3>& vertex, std::uint32_t width, std::uint32_t height)
{
    // Binary64 composition of parent T(.1,.2) Rz(90) S(2,2), child
    // T(.25,-.125) S(.5,.75), then the authored outer translation.
    // Camera T(0,0,4), identity rotation, and 90 degree vertical FOV.
    const double x = .35 - 1.5 * vertex[1] + (parentTarget && moving ? .5 * seconds : 0.0)
        + (moving ? .12 * seconds : 0.0);
    const double y = .7 + vertex[0] + (!parentTarget && moving ? seconds : 0.0) - (moving ? .08 * seconds : 0.0);
    const double scale = .5 * height / (4.0 - vertex[2]);
    return {.5 * width + x * scale, .5 * height - y * scale};
}

bool checkImportedMotionPixels(const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& frame,
    bool parentTarget, bool moving, double currentSeconds, std::optional<double> previousSeconds,
    std::string& error)
{
    arbitgpu::NativeRawPassPixels motion, mask;
    if (!frame || !frame->readRawPass(renderpassoutput::Output::Motion, motion, error)
        || !frame->readRawPass(renderpassoutput::Output::Mask, mask, error)) return false;
    const auto width = frame->width(), height = frame->height();
    if (motion.format != arbitgpu::NativeTexturePixelFormat::Rg16Float
        || mask.format != arbitgpu::NativeTexturePixelFormat::R8Unorm
        || motion.width != width || motion.height != height || mask.width != width || mask.height != height
        || motion.bytes.size() != std::size_t(width) * height * 4
        || mask.bytes.size() != std::size_t(width) * height)
    { error = "imported Metal Motion requires owned raw RG16F and Mask extents"; return false; }
    constexpr std::array<std::array<double,3>,3> vertices {{
        {{-.5,-.5,0}}, {{.5,-.5,0}}, {{0,.5,0}}
    }};
    std::array<std::array<double,2>,3> current {}, previous {};
    for (std::size_t i = 0; i < vertices.size(); ++i) {
        current[i] = importedMotionVertex(parentTarget, moving, currentSeconds, vertices[i], width, height);
        previous[i] = importedMotionVertex(parentTarget, moving, previousSeconds.value_or(currentSeconds),
            vertices[i], width, height);
    }
    const auto cross = [](double ax, double ay, double bx, double by) { return ax * by - ay * bx; };
    const double area = cross(current[1][0] - current[0][0], current[1][1] - current[0][1],
        current[2][0] - current[0][0], current[2][1] - current[0][1]);
    const auto half = [](std::uint16_t bits) {
        const unsigned exponent = (bits >> 10) & 31, fraction = bits & 1023;
        const double value = exponent == 0 ? std::ldexp(double(fraction), -24)
            : exponent == 31 ? std::numeric_limits<double>::infinity()
            : std::ldexp(double(1024 + fraction), int(exponent) - 25);
        return bits & 32768 ? -value : value;
    };
    std::array<double,3> nearest {1e9,1e9,1e9};
    std::size_t covered = 0, interior = 0;
    for (std::uint32_t row = 0; row < height; ++row)
        for (std::uint32_t column = 0; column < width; ++column) {
            const auto index = std::size_t(row) * width + column;
            const double x = column + .5, y = row + .5;
            const double b = cross(x - current[0][0], y - current[0][1],
                current[2][0] - current[0][0], current[2][1] - current[0][1]) / area;
            const double c = cross(current[1][0] - current[0][0], current[1][1] - current[0][1],
                x - current[0][0], y - current[0][1]) / area;
            const double a = 1.0 - b - c;
            if (std::min({a,b,c}) > .15) {
                ++interior;
                if (mask.bytes[index] == 0)
                { error = "imported Metal triangle misses independently projected interior"; return false; }
            }
            std::uint16_t rawX = 0, rawY = 0;
            std::memcpy(&rawX, motion.bytes.data() + index * 4, 2);
            std::memcpy(&rawY, motion.bytes.data() + index * 4 + 2, 2);
            if (!previousSeconds || !moving || mask.bytes[index] == 0) {
                if ((rawX & 0x7fff) != 0 || (rawY & 0x7fff) != 0)
                { error = "imported Metal missing predecessor or background Motion is not exact zero"; return false; }
            }
            if (mask.bytes[index] == 0) continue;
            ++covered;
            if (std::min({a,b,c}) < -.015)
            { error = "imported Metal triangle coverage disagrees with composed world projection"; return false; }
            for (std::size_t i = 0; i < nearest.size(); ++i)
                nearest[i] = std::min(nearest[i], std::hypot(x - current[i][0], y - current[i][1]));
            // Flat camera depth makes screen barycentrics exact for this triangle.
            const std::array<double,3> weights {a,b,c};
            double expectedX = 0, expectedY = 0;
            for (std::size_t i = 0; i < weights.size(); ++i) {
                expectedX += weights[i] * (current[i][0] - previous[i][0]);
                expectedY += weights[i] * (previous[i][1] - current[i][1]); // Raw Motion is positive Y upwards.
            }
            // Keep the existing Metal half-float motion tolerance in pixel units.
            if (!std::isfinite(half(rawX)) || !std::isfinite(half(rawY))
                || std::abs(half(rawX) - expectedX) > .005
                || std::abs(half(rawY) - expectedY) > .005)
            { error = "imported Metal raw Motion differs from independent pixel displacement"; return false; }
        }
    if (!covered || !interior || *std::max_element(nearest.begin(), nearest.end()) > 2.5)
    { error = "imported Metal triangle must cover its interior and all three projected vertices"; return false; }
    return true;
}

bool verifyMetalImportedObjectMotion(arbitgpu::NativeDeformationBackend& backend, std::string& error)
{
    for (int mode : {0,1,2}) {
        const bool parentTarget = mode == 1, moving = mode != 2;
        const auto bytes = importedObjectMotionFixture(parentTarget, moving);
        videohelper::modelpayload::Store store;
        const visualanimationimport::ExactContentAssetKey key {"metal-imported-object-motion", 1,
            videohelper::modelpayload::detail::digestHex(bytes), "model/gltf-binary", bytes.size()};
        if (!store.begin(key.id, key) || !store.appendBase64(key.id, 0, encodeBase64(bytes)) || !store.commit(key.id))
        { error = "imported Metal Motion payload publication failed"; return false; }
        videohelper::modelpayload::ImportedAnimatedScenePayloadExecution execution(store, backend);
        videohelper::modelpayload::ImportedAnimatedSceneRequest request;
        request.operation.sourceStableId = 801; request.operation.deformationStableId = 802;
        request.operation.schedule = {801,802}; request.operation.asset = key; request.operation.sceneIndex = 0;
        request.operation.meshStableId = request.operation.animationClipStableId = 1;
        request.operation.clipName = "Imported Object Motion";
        request.frame = {0,24,1}; request.structuralRevision = 1; request.renderClipId = 83;
        request.projectGeneration = request.helperGeneration = 1; request.width = 96; request.height = 64;
        request.runtimeInputs.imageOutput = renderpassoutput::Output::Motion;
        videohelper::modelpayload::ImportedAnimatedSceneReceipt receipt;
        const auto atFrame = [&](std::int64_t frame) {
            request.frame.frame = frame;
            const double seconds = double(frame) * request.frame.rateDenominator / request.frame.rateNumerator;
            request.operation.playback.timelineSeconds = seconds;
            request.runtimeInputs.objectTranslationOffset = moving
                ? std::array<float,3>{float(.12 * seconds), float(-.08 * seconds), 0.0f}
                : std::array<float,3>{};
            return seconds;
        };
        const auto draw = [&](bool exported, double seconds, std::optional<double> prior) {
            if (!(exported ? execution.executeExport(request, receipt, error) : execution.executePreview(request, receipt, error)))
                return false;
            if (bool(receipt.runtimeInputs.previousMotion) != bool(prior))
            { error = "imported Metal Motion lane retained an unexpected predecessor"; return false; }
            return checkImportedMotionPixels(receipt.frame.nativeFrame, parentTarget, moving, seconds, prior, error);
        };
        if (!draw(false, atFrame(0), std::nullopt) || !draw(false, atFrame(1), 0.0)) return false;
        arbitgpu::NativeRawPassPixels moved;
        if (!receipt.frame.nativeFrame->readRawPass(renderpassoutput::Output::Motion, moved, error)) return false;
        const auto priorWorlds = receipt.runtimeInputs.previousMotion->importedNodeWorldMatrices;
        if (!priorWorlds || priorWorlds->size() != 2 || (*priorWorlds)[0].nodeStableId != 1 || (*priorWorlds)[1].nodeStableId != 4)
        { error = "imported Metal Motion must retain exact drawn and ancestor node identities"; return false; }
        const auto prepared = backend.prepare(receipt.source);
        if (!prepared.prepared) { error = prepared.error; return false; }
        for (std::size_t node : {0u,1u})
        for (bool missing : {false,true}) {
            if (node == 1 && missing) continue; // Full ancestor completeness belongs to imported admission.
            auto invalidInputs = receipt.runtimeInputs;
            auto invalidWorlds = std::make_shared<arbitgpu::NativeSceneMotionSample::ImportedNodeWorldMatrices>(*priorWorlds);
            if (missing) invalidWorlds->erase(invalidWorlds->begin() + node);
            else (*invalidWorlds)[node].world[12] += .125f;
            invalidInputs.previousMotion->importedNodeWorldMatrices = invalidWorlds;
            const auto rejected = backend.render(receipt.source, receipt.evaluation.deformation,
                prepared.resources, request.width, request.height, invalidInputs);
            if (rejected.rendered || rejected.frame || rejected.stats.dispatchCount != 0 || rejected.error.empty())
            { error = "imported Metal prior-world mismatch must refuse before compute dispatch"; return false; }
        }
        auto missingDeformation = receipt.runtimeInputs;
        missingDeformation.previousMotion->deformation.reset();
        const auto rejected = backend.render(receipt.source, receipt.evaluation.deformation,
            prepared.resources, request.width, request.height, missingDeformation);
        if (rejected.rendered || rejected.frame || rejected.stats.dispatchCount != 0 || rejected.error.empty())
        { error = "imported Metal world table without prior deformation must refuse before dispatch"; return false; }
        if (!draw(false, atFrame(1), 0.0)) return false;
        arbitgpu::NativeRawPassPixels redraw;
        if (!receipt.frame.nativeFrame->readRawPass(renderpassoutput::Output::Motion, redraw, error) || redraw.bytes != moved.bytes)
        { error = "imported Metal same-frame redraw changed raw Motion"; return false; }
        atFrame(2); const auto savedWidth = request.width; request.width = 0;
        if (execution.executePreview(request, receipt, error) || receipt.valid())
        { error = "failed imported Metal draw published a receipt"; return false; }
        request.width = savedWidth;
        if (!draw(false, atFrame(2), 1.0 / 24.0)) return false;
        if (!draw(true, atFrame(2), std::nullopt) || !draw(true, atFrame(3), 2.0 / 24.0)) return false;
        if (!draw(false, atFrame(2), 1.0 / 24.0) || !draw(false, atFrame(3), 2.0 / 24.0)) return false;
        for (int reset = 0; reset < 12; ++reset) {
            std::int64_t next = request.frame.frame + 1;
            if (reset == 0) next += 10;
            if (reset == 1) next = 0;
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
            request.operation.schedule = {request.operation.sourceStableId,request.operation.deformationStableId};
            if (!draw(false, atFrame(next), std::nullopt)) return false;
        }
        request.operation.sourceStableId = 801; request.operation.deformationStableId = 802;
        request.operation.schedule = {801,802};
        if (!draw(false, atFrame(request.frame.frame + 1), std::nullopt)) return false;
    }
    return true;
}

bool checkMotionTranslation(const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& frame,
    const HarmonicMIDI::grid::SceneCameraRecord& camera,double dx,double dy,bool neutral)
{
    if(!frame)return false;
    arbitgpu::NativeRawPassPixels motion,depth,mask;
    std::string error;
    if(!frame->readRawPass(renderpassoutput::Output::Motion,motion,error)
        ||!frame->readRawPass(renderpassoutput::Output::Depth,depth,error)
        ||!frame->readRawPass(renderpassoutput::Output::Mask,mask,error)
        ||motion.format!=arbitgpu::NativeTexturePixelFormat::Rg16Float
        ||motion.bytes.size()!=std::size_t(frame->width())*frame->height()*4
        ||depth.bytes.size()!=std::size_t(frame->width())*frame->height()*4
        ||mask.bytes.size()!=std::size_t(frame->width())*frame->height())return false;
    const auto half=[](std::uint16_t bits) {
        const unsigned exponent=(bits>>10)&31,fraction=bits&1023;
        const float value=exponent==0?std::ldexp(float(fraction),-24)
            :exponent==31?std::numeric_limits<float>::infinity()
            :std::ldexp(float(1024+fraction),int(exponent)-25);
        return bits&32768?-value:value;
    };
    std::size_t covered=0;
    for(std::size_t i=0;i<mask.bytes.size();++i) {
        std::uint16_t x=0,y=0;
        std::memcpy(&x,motion.bytes.data()+i*4,2);std::memcpy(&y,motion.bytes.data()+i*4+2,2);
        if(neutral) {if(x!=0&&x!=0x8000)return false;if(y!=0&&y!=0x8000)return false;continue;}
        if(mask.bytes[i]==0)continue;
        ++covered;
        float normalizedDepth=0;std::memcpy(&normalizedDepth,depth.bytes.data()+i*4,4);
        const double distance=camera.nearPlane+normalizedDepth*(camera.farPlane-camera.nearPlane);
        const double scale=frame->height()/(2*std::tan(double(camera.verticalFovRadians)*.5)*distance);
        if(!std::isfinite(half(x))||!std::isfinite(half(y))
            ||std::abs(half(x)-dx*scale)>.005||std::abs(half(y)-dy*scale)>.005)return false;
    }
    return neutral||covered>0;
}

bool verifyMetalRetainedPoseMotion(arbitgpu::NativeDeformationBackend& backend,const Fixture& fixture)
{
    for(bool skin : {false,true}) {
        auto scene=std::make_shared<HarmonicMIDI::grid::Visual3DScene>(*fixture.scene);
        scene->objects[0].transform={};
        auto source=std::make_shared<arbitgpu::NativeDeformationScene>(*fixture.source);
        source->scene=scene;source->retainDeformedGeometry=false;
        std::shared_ptr<const AnimationDeformationSnapshot> prior=fixture.snapshot(0),current=fixture.snapshot(1);
        std::string error;
        if(skin) {
            const std::array<JointView,1> joints {{JointView{JointId{301},{}}}};
            const std::array<float,16> inverseBind {1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
            std::array<std::uint32_t,24*4> indices{};
            std::array<float,24*4> weights{};
            for(std::size_t v=0;v<24;++v)weights[v*4]=1;
            const JointWeightSetView influences{indices.data(),indices.size(),weights.data(),weights.size()};
            const SkinView skinned{SkinId{401},joints.data(),joints.size(),inverseBind.data(),inverseBind.size()};
            const MeshView mesh{MeshId{501},24,skinned.id,&influences,1,nullptr,0};
            const AssetView view{&skinned,1,&mesh,1};
            const auto asset=DeformationAsset::create(view,{},error);
            const std::array<double,2> times{0,1};
            const std::array<float,6> values{0,0,0,.75f,.9f,0};
            const animation::TrackView track{animation::TrackId{11},animation::TargetId{1001},
                animation::Channel::Translation,animation::Interpolation::Linear,times.data(),values.data(),2,6,0};
            const animation::ClipView clipView{animation::ClipId{701},1,&track,1};
            const auto clip=animation::Clip::create(clipView,{},error);
            const JointAnimationBindingView joint{animation::TargetId{1001},skinned.id,joints[0].id};
            const AnimationDeformationBindingView bindings{&joint,1,nullptr,0};
            if(!asset||!clip)return false;
            prior=evaluateAnimationDeformation(*asset,*clip,bindings,
                {RationalFrameTime{0,30,1},77,animation::Playback::Clamp,std::nullopt},{},error);
            current=evaluateAnimationDeformation(*asset,*clip,bindings,
                {RationalFrameTime{1,30,1},77,animation::Playback::Clamp,std::nullopt},{},error);
            source->deformation=asset;source->morphBaseWeights.clear();
        }
        if(!prior||!current)return false;
        const auto prepared=backend.prepare(source);
        if(!prepared.prepared)return false;
        arbitgpu::NativeDeformationRuntimeInputs inputs;
        inputs.imageOutput=renderpassoutput::Output::Motion;
        const auto first=backend.render(source,prior,prepared.resources,96,96,inputs);
        if(!first.rendered||!checkMotionTranslation(first.frame,scene->cameras[0],0,0,true))return false;
        arbitgpu::NativeSceneMotionHistory preview,exported;
        if(!preview.commit(0,30,1,96,96,inputs,prior))return false;
        inputs.previousMotion=preview.predecessor(1,30,1,96,96);
        const auto moved=backend.render(source,current,prepared.resources,96,96,inputs);
        if(!moved.rendered||moved.stats.dispatchCount!=2
            ||!checkMotionTranslation(moved.frame,scene->cameras[0],.025,skin?.03:0,false))return false;
        // Retained prior runtime gain is independent of the current gain.
        if(!skin) {
            inputs.previousMotion=arbitgpu::sceneMotionSample(inputs,current);
            inputs.previousMotion->morphWeight=.5f;
            const auto gain=backend.render(source,current,prepared.resources,96,96,inputs);
            if(!gain.rendered||!checkMotionTranslation(gain.frame,scene->cameras[0],.0125,0,false))return false;
        }
        inputs.previousMotion=arbitgpu::sceneMotionSample(inputs,current);
        const auto unchanged=backend.render(source,current,prepared.resources,96,96,inputs);
        if(!unchanged.rendered||!checkMotionTranslation(unchanged.frame,scene->cameras[0],0,0,true))return false;
        const auto saved=preview.current->deformation;
        const auto rejected=backend.render(source,current,prepared.resources,0,96,inputs);
        if(rejected.rendered||rejected.frame||preview.current->deformation!=saved)return false;
        for(auto predecessor : {preview.predecessor(4,30,1,96,96),preview.predecessor(1,60,1,96,96),
                preview.predecessor(1,30,1,48,96),exported.predecessor(1,30,1,96,96)}) {
            inputs.previousMotion=predecessor;
            const auto reset=backend.render(source,current,prepared.resources,96,96,inputs);
            if(!reset.rendered||!checkMotionTranslation(reset.frame,scene->cameras[0],0,0,true))return false;
        }
        inputs.previousMotion.reset();
        if(!exported.commit(0,30,1,96,96,inputs,prior))return false;
        inputs.previousMotion=exported.predecessor(1,30,1,96,96);
        const auto exportFrame=backend.render(source,current,prepared.resources,96,96,inputs);
        arbitgpu::NativeRawPassPixels previewPixels,exportPixels;
        if(!exportFrame.rendered||!moved.frame->readRawPass(renderpassoutput::Output::Motion,previewPixels,error)
            ||!exportFrame.frame->readRawPass(renderpassoutput::Output::Motion,exportPixels,error)
            ||previewPixels.bytes!=exportPixels.bytes)return false;
    }
    return true;
}
} // namespace

int main()
{
    bool ok = true;
    Fixture fixture;
    auto firstSnapshot = fixture.snapshot(0);
    auto movedSnapshot = fixture.snapshot(30);
    ok &= expect(fixture.source && firstSnapshot && movedSnapshot,
                 "the exact Metal deformation fixture must be admitted");

    auto& backend = arbitgpu::nativeDeformationBackend();
    std::string compositionError;
    ok &= expect(videohelper::fixture3d::verifyNativeComposition(backend,
        [](const auto& frame) {return readBgra8(frame);},compositionError),
        "typed full-scene composition, predecessor Motion and normal/tangent reference");
    if (!compositionError.empty()) std::cerr << compositionError << '\n';
    const bool productionJoin=videohelper::fixture3d::verifyNativeCompositionProductionJoin(backend,
        [](const auto& frame) {return readBgra8(frame);},compositionError);
    if (!productionJoin) std::cerr << "production composition join refusal: "
        << (compositionError.empty() ? std::string("no refusal text was set") : compositionError) << '\n';
    ok &= expect(productionJoin,
        "actual production composition provenance, epochs, independent histories and bounded escaped receipts");
    const bool parentedProduction=videohelper::fixture3d::verifyParentedImportedProductionJoin(backend,
        [](const auto& frame) {return readBgra8(frame);},compositionError);
    if (!parentedProduction) std::cerr << compositionError << '\n';
    ok &= expect(parentedProduction,"production-issued parent maps, real deformation, native Color/Motion and complete-root conservation");
    const bool selectedCamera=videohelper::fixture3d::verifyImportedCameraSelectionProductionJoin(backend,
        [](const auto& frame) {return readBgra8(frame);},compositionError);
    if (!selectedCamera) std::cerr << compositionError << '\n';
    ok &= expect(selectedCamera,"exact source camera selection, sampled ancestry, override and Preview/Export native join");
    const bool selectedSkin=videohelper::fixture3d::verifySelectedDeformationCameraReceipt(backend,
        [](const auto& frame) {return readBgra8(frame);},compositionError);
    if (!selectedSkin) std::cerr << compositionError << '\n';
    ok &= expect(selectedSkin,"genuine skin selected-camera receipt rejects copied camera, selector and native-frame mutations");
    for (const unsigned target : {1u,2u})
    {
        const bool composed=videohelper::fixture3d::verifyCameraLightCompositionProductionJoin(backend,
            [](const auto& frame) {return readBgra8(frame);},target,compositionError);
        if (!composed) std::cerr << compositionError << '\n';
        ok &= expect(composed,"authentic camera/light-only static composition on Metal");
    }
    std::string extractionError;
    const bool beatSync=importedanimationbeatfixture::verify(backend,
        [](const auto& frame) { return readBgra8(frame); }, extractionError);
    if (!beatSync) std::cerr << "imported Beat Sync refusal: "
        << (extractionError.empty() ? std::string("no refusal text was set") : extractionError) << '\n';
    ok &= expect(beatSync,"imported Beat Sync matches independent native joint geometry across tempo changes, seeks and export");
    extractionError.clear();
    const bool extracted=verifyAnimatedGeometryExtraction(extractionError);
    if (!extracted) std::cerr << "imported Geometry3D extraction refusal: "
        << (extractionError.empty() ? std::string("no refusal text was set") : extractionError) << '\n';
    ok &= expect(extracted,"imported animation reaches retained Geometry3D through the native GPU path");
    const bool multiobject = multiobjectanimationfixture::verify(backend,
        [](const auto& frame) { return readBgra8(frame); }, extractionError);
    if (!multiobject) std::cerr << extractionError << '\n';
    ok &= expect(multiobject, "multi-object imported animation preserves GPU preview, export and seek");
    const bool rigidScene = multiobjectanimationfixture::verify(backend,
        [](const auto& frame) { return readBgra8(frame); }, extractionError, true);
    if (!rigidScene) std::cerr << extractionError << '\n';
    ok &= expect(rigidScene, "rigid-only parent animation is selectable and rendered on Metal");
    for (const bool rigid : {false, true}) {
        const bool repeated = multiobjectanimationfixture::verify(backend,
            [](const auto& frame) { return readBgra8(frame); }, extractionError, rigid, true);
        if (!repeated) std::cerr << extractionError << '\n';
        ok &= expect(repeated, "repeated imported nodes retain independent whole-scene poses on Metal");
    }
    for (unsigned target : {1u,2u}) {
        const bool sceneOnly=multiobjectanimationfixture::verifySceneOnly(backend,
            [](const auto& frame) { return readBgra8(frame); },target,extractionError);
        if (!sceneOnly) std::cerr << extractionError << '\n';
        ok &= expect(sceneOnly,"camera/light-only clips render without mesh deformation on Metal");
    }
    const auto info = backend.info();
    if (! info.available || ! info.compute || info.backend != "metal")
    {
        std::cerr << "SKIP: strict Metal deformation unavailable: " << info.error << '\n';
        return 77;
    }
    ok &= expect(backend.supportsRetainedDeformationMotion() && backend.supportsRetainedNoteMotion()
        && backend.supportsRetainedImportedObjectMotion(),
        "Metal deformation consumer explicitly supports retained pose, note and imported object Motion");
    const bool importedMotion = verifyMetalImportedObjectMotion(backend, extractionError);
    if (!importedMotion) std::cerr << extractionError << '\n';
    ok &= expect(importedMotion,
        "Metal imported object and ancestor Motion matches independent raw pixel projection, resets and lanes");
    ok &= probeAnimatedTriangle(backend);
    ok &= expect(verifyMetalRetainedPoseMotion(backend,fixture),
        "Metal Motion consumes retained skin/morph poses and gains, with neutral resets and isolated preview/export histories");
    auto preparation = backend.prepare(fixture.source);
    if (! preparation.prepared)
        std::cerr << "Metal deformation preparation error: " << preparation.error << '\n';
    ok &= expect(preparation.prepared && preparation.resources,
                 "the exact textured morph scene must prepare on Metal");
    auto first = backend.render(fixture.source, firstSnapshot, preparation.resources, 96, 96);
    auto repeated = backend.render(fixture.source, firstSnapshot, preparation.resources, 96, 96);
    auto moved = backend.render(fixture.source, movedSnapshot, preparation.resources, 96, 96);
    ok &= expect(first.deformedVertices.size()==24u*8u
                 && first.deformedVertices==repeated.deformedVertices
                 && moved.deformedVertices.size()==first.deformedVertices.size(),
                 "GPU geometry extraction preserves topology and exact frame repetition");
    if (first.deformedVertices.size()==24u*8u && moved.deformedVertices.size()==24u*8u)
        for (std::size_t vertex=0;vertex<24;++vertex)
        {
            ok &= expect(std::abs(moved.deformedVertices[vertex*8]-first.deformedVertices[vertex*8]-0.75f)<0.00001f,
                         "retained geometry contains the GPU evaluated morph displacement");
            ok &= expect(first.deformedVertices[vertex*8+6]==fixture.sceneValue.vertices[vertex].uv.x
                         && first.deformedVertices[vertex*8+7]==fixture.sceneValue.vertices[vertex].uv.y,
                         "GPU extraction preserves UVs in the full tangent and color vertex layout");
        }
    auto sought=backend.render(fixture.source,firstSnapshot,preparation.resources,96,96);
    ok &= expect(sought.deformedVertices==first.deformedVertices,
                 "seeking backwards restores the original retained GPU geometry");
    arbitgpu::NativeDeformationRuntimeInputs mutedMorphRuntime;
    mutedMorphRuntime.morphWeight = 0.0f;
    auto mutedMorph = backend.render(
        fixture.source, movedSnapshot, preparation.resources, 96, 96, mutedMorphRuntime);
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
    if (! first.rendered) std::cerr << "Metal first render error: " << first.error << '\n';
    if (! repeated.rendered) std::cerr << "Metal repeated render error: " << repeated.error << '\n';
    if (! moved.rendered) std::cerr << "Metal moved render error: " << moved.error << '\n';
    ok &= expect(first.rendered && repeated.rendered && moved.rendered
                     && mutedMorph.rendered && movedScene.rendered
                     && rotated.rendered && scaled.rendered,
                 "strict Metal deformation must render exact snapshots");
    ok &= expect(first.frame && first.frame->depthImageHandle() != 0
                     && first.frame->depthTextureViewHandle() != 0,
                 "strict Metal Render 3D must publish its native depth resource");

    std::uint64_t firstHash = 0;
    std::uint64_t movedHash = 0;
    if (first.frame && repeated.frame && moved.frame && rotated.frame && scaled.frame)
    {
        const auto firstPixels = readBgra8(first.frame);
        const auto repeatedPixels = readBgra8(repeated.frame);
        const auto movedPixels = readBgra8(moved.frame);
        const auto mutedMorphPixels = readBgra8(mutedMorph.frame);
        const auto movedScenePixels = readBgra8(movedScene.frame);
        const auto rotatedPixels = readBgra8(rotated.frame);
        const auto scaledPixels = readBgra8(scaled.frame);
        firstHash = fnv1a64(firstPixels);
        movedHash = fnv1a64(movedPixels);
        ok &= expect(firstPixels == repeatedPixels,
                     "equal Metal snapshots must produce identical pixels");
        ok &= expect(firstPixels != movedPixels,
                     "a changed morph weight must change Metal pixels");
        ok &= expect(firstPixels == mutedMorphPixels,
                     "zero Metal runtime morph weight must restore the imported base shape");
        ok &= expect(firstPixels != movedScenePixels,
                     "Metal object and camera runtime offsets must change pixels");
        ok &= expect(firstPixels != rotatedPixels,
                     "Metal runtime object rotation must change pixels");
        ok &= expect(firstPixels != scaledPixels,
                     "Metal runtime object scale must change pixels");
    }

    auto material = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>();
    material->backend = arbitgpu::NativeFixtureMaterialBackend::Metal;
    material->object = videohelper::fixture3d::kCubeObjectId;
    material->bindingDigest = "metal-deformed-time-material-binding";
    material->programIdentity = "metal-deformed-time-material-program";
    material->baseColorSource
        = arbitgpu::NativeFixtureSurfaceMaterialProgram::BaseColorSource::TimeLinearMix;
    material->parameters.baseColorMetallic = { 1.0f, 0.0f, 0.0f, 0.0f };
    material->parameters.emissionRoughness = { 0.1f, 0.0f, 0.0f, 0.5f };
    material->parameters.normalOpacity = { 0.0f, 0.0f, 1.0f, 1.0f };
    material->parameters.transmissionIorClearcoat = { 0.0f, 1.5f, 0.0f, 0.0f };
    material->parameters.identifiers[0] = videohelper::fixture3d::kCubeMaterialId.value;
    material->timeMixEndColor = { 0.0f, 0.0f, 1.0f };
    auto materialPreparation = backend.prepare(fixture.source, material);
    arbitgpu::NativeDeformationRuntimeInputs materialStartRuntime;
    materialStartRuntime.timeSeconds = 0.0f;
    auto materialStart = backend.render(
        fixture.source, firstSnapshot, materialPreparation.resources, 96, 96, materialStartRuntime);
    arbitgpu::NativeDeformationRuntimeInputs materialEndRuntime;
    materialEndRuntime.timeSeconds = 1.0f;
    auto materialEnd = backend.render(
        fixture.source, firstSnapshot, materialPreparation.resources, 96, 96, materialEndRuntime);
    arbitgpu::NativeDeformationRuntimeInputs darkEmission;
    darkEmission.emissionGain = 0.0f;
    auto materialDark = backend.render(
        fixture.source, firstSnapshot, materialPreparation.resources, 96, 96, darkEmission);
    if (! materialStart.rendered)
        std::cerr << "Metal material start error: " << materialStart.error << '\n';
    if (! materialEnd.rendered)
        std::cerr << "Metal material end error: " << materialEnd.error << '\n';
    ok &= expect(materialPreparation.prepared && materialStart.rendered
                     && materialEnd.rendered && materialDark.rendered,
                 "strict Metal must shade deformed geometry with Surface Material");
    if (materialStart.frame && materialEnd.frame)
        ok &= expect(readBgra8(materialStart.frame) != readBgra8(materialEnd.frame),
                     "the time-dependent Surface Material must change Metal pixels");
    if (materialStart.frame && materialDark.frame)
        ok &= expect(readBgra8(materialStart.frame) != readBgra8(materialDark.frame),
                     "Metal runtime emission gain must change material pixels");

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
    if (ok)
    {
        std::cout << "Metal animation deformation backend: PASS; device=" << info.device
                  << "; base-bgra8-fnv1a64=" << std::hex << firstHash
                  << "; morphed-bgra8-fnv1a64=" << movedHash << std::dec << '\n';
    }
    return ok ? 0 : 1;
}
