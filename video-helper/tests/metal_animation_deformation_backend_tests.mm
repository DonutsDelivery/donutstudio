#include "VisualAnimationDeformationEvaluation.h"
#include "gpu_backend/backend.h"
#include "support/fixture_scene.h"

#define SOKOL_METAL
#include "sokol_gfx.h"
#import <Metal/Metal.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
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
    const auto info = backend.info();
    if (! info.available || ! info.compute || info.backend != "metal")
    {
        std::cerr << "SKIP: strict Metal deformation unavailable: " << info.error << '\n';
        return 77;
    }
    auto preparation = backend.prepare(fixture.source);
    if (! preparation.prepared)
        std::cerr << "Metal deformation preparation error: " << preparation.error << '\n';
    ok &= expect(preparation.prepared && preparation.resources,
                 "the exact textured morph scene must prepare on Metal");
    auto first = backend.render(fixture.source, firstSnapshot, preparation.resources, 96, 96);
    auto repeated = backend.render(fixture.source, firstSnapshot, preparation.resources, 96, 96);
    auto moved = backend.render(fixture.source, movedSnapshot, preparation.resources, 96, 96);
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
