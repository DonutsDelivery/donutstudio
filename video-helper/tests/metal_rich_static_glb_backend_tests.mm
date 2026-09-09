#include "../src/glb_scene_adapter.h"
#include "../src/gpu_backend/backend.h"
#include "../../plugin/Tests/fixtures/visual-model/StaticVisualModelWorkflowFixture.h"

#include "sokol_gfx.h"

#import <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

namespace
{
using namespace HarmonicMIDI::grid;

std::vector<std::uint8_t> readPixels(
    const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& frame)
{
    if (!frame || frame->colorImageHandle() == 0)
        return {};
    sg_image image {};
    image.id = static_cast<std::uint32_t>(frame->colorImageHandle());
    const auto info = sg_mtl_query_image_info(image);
    if (info.active_slot < 0 || info.active_slot >= SG_NUM_INFLIGHT_FRAMES)
        return {};
    id<MTLTexture> texture = (__bridge id<MTLTexture>) info.tex[info.active_slot];
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>) sg_mtl_command_queue();
    if (texture == nil || queue == nil)
        return {};
    const auto tightRowBytes = static_cast<std::size_t>(frame->width()) * 4u;
    const auto rowBytes = (tightRowBytes + 255u) & ~std::size_t(255u);
    id<MTLBuffer> buffer = [texture.device newBufferWithLength:rowBytes * frame->height()
                                                   options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    if (buffer == nil || command == nil || blit == nil)
        return {};
    [blit copyFromTexture:texture sourceSlice:0 sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(frame->width(), frame->height(), 1)
                 toBuffer:buffer destinationOffset:0 destinationBytesPerRow:rowBytes
 destinationBytesPerImage:rowBytes * frame->height()];
    [blit endEncoding];
    [command commit];
    [command waitUntilCompleted];
    std::vector<std::uint8_t> pixels(tightRowBytes * frame->height());
    const auto* source = static_cast<const std::uint8_t*>(buffer.contents);
    for (std::uint32_t y = 0; source != nullptr && y < frame->height(); ++y)
        std::copy_n(source + y * rowBytes, tightRowBytes,
                    pixels.data() + y * tightRowBytes);
#if ! __has_feature(objc_arc)
    [buffer release];
#endif
    return pixels;
}

struct Rendered final
{
    arbitgpu::NativeFixtureScenePreparation preparation;
    arbitgpu::NativeFixtureSceneSubmission render;
    std::vector<std::uint8_t> pixels;
};

Rendered render(arbitgpu::NativeFixtureSceneBackend& backend, const Visual3DScene& value)
{
    Rendered result;
    auto scene = std::make_shared<const Visual3DScene>(value);
    result.preparation = backend.prepare(scene, nullptr);
    if (result.preparation.prepared)
        result.render = backend.render(scene, result.preparation.resources, 96, 96, {});
    if (result.render.rendered)
        result.pixels = readPixels(result.render.frame);
    return result;
}

bool check(bool condition, const char* message)
{
    if (!condition)
        std::cerr << "FAIL: " << message << '\n';
    return condition;
}

std::size_t nonBackgroundCount(const std::vector<std::uint8_t>& pixels)
{
    std::size_t count = 0;
    for (std::size_t index = 0; index + 3 < pixels.size(); index += 4)
        if (pixels[index] != 7u || pixels[index + 1] != 10u
            || pixels[index + 2] != 18u || pixels[index + 3] != 255u)
            ++count;
    return count;
}
} // namespace

int main()
{
    @autoreleasepool
    {
        auto& backend = arbitgpu::nativeFixtureSceneBackend();
        const auto info = backend.info();
        if (!info.available || info.backend != "metal")
        {
            std::cerr << "FAIL: strict rich static GLB Metal backend unavailable: "
                      << info.error << '\n';
            return 1;
        }

        std::string error;
        videohelper::gltf::GlbAdmissionOptions options;
        auto document = videohelper::gltf::decodeStaticGlb(
            staticvisualmodelworkflowfixture::kGlb.data(),
            staticvisualmodelworkflowfixture::kGlb.size(), options, error);
        auto adapted = document
            ? videohelper::gltf::adaptStaticGlbToVisual3DScene(*document, error)
            : std::nullopt;
        if (!adapted)
        {
            std::cerr << "FAIL: rich static GLB decode/adaptation failed: " << error << '\n';
            return 1;
        }
        const auto baseline = render(backend, *adapted);

        auto flattened = *adapted;
        for (std::size_t index = 0; index < flattened.objectCount; ++index)
            flattened.objects[index].parent = {};
        const auto flattenedRender = render(backend, flattened);

        auto lit = *adapted;
        lit.ambientColor = {};
        lit.lightCount = 1;
        lit.lights[0].kind = SceneLightKind::Spot;
        lit.lights[0].transform = {};
        lit.lights[0].transform.translation = {0.0f, 0.0f, 1.0f};
        lit.lights[0].range = 5.0f;
        lit.lights[0].innerConeAngle = 0.15f;
        lit.lights[0].outerConeAngle = 0.35f;
        const auto spotInside = render(backend, lit);
        auto spotOutsideScene = lit;
        spotOutsideScene.lights[0].transform.rotation = {0.0f, 1.0f, 0.0f, 0.0f};
        const auto spotOutside = render(backend, spotOutsideScene);
        auto infiniteRangeScene = lit;
        infiniteRangeScene.lights[0].kind = SceneLightKind::Point;
        infiniteRangeScene.lights[0].range = 0.0f;
        const auto infiniteRange = render(backend, infiniteRangeScene);
        auto finiteRangeScene = infiniteRangeScene;
        finiteRangeScene.lights[0].range = 0.25f;
        const auto rangeCutoff = render(backend, finiteRangeScene);

        auto textured = *adapted;
        auto* meshObject = std::find_if(textured.objects.begin(),
            textured.objects.begin() + textured.objectCount,
            [] (const auto& object) { return object.indexCount != 0; });
        if (meshObject == textured.objects.begin() + textured.objectCount)
            return 1;
        auto* material = const_cast<SceneMaterialRecord*>(visual3d_detail::findById(
            textured.materials, textured.materialCount, meshObject->material));
        if (material == nullptr || textured.textureCount == 0)
            return 1;
        const auto firstRoleTexture = textured.textureCount++;
        const auto secondRoleTexture = textured.textureCount++;
        const auto normalTexture = textured.textureCount++;
        const auto firstRoleTexel = textured.textureTexelCount++;
        const auto secondRoleTexel = textured.textureTexelCount++;
        const auto normalTexel = textured.textureTexelCount++;
        textured.textures[firstRoleTexture].id = SceneTextureId {9001};
        textured.textures[firstRoleTexture].firstTexel = static_cast<std::uint32_t>(firstRoleTexel);
        textured.textures[firstRoleTexture].width = 1;
        textured.textures[firstRoleTexture].height = 1;
        textured.textures[secondRoleTexture] = textured.textures[firstRoleTexture];
        textured.textures[secondRoleTexture].id = SceneTextureId {9002};
        textured.textures[secondRoleTexture].firstTexel = static_cast<std::uint32_t>(secondRoleTexel);
        textured.textures[normalTexture] = textured.textures[firstRoleTexture];
        textured.textures[normalTexture].id = SceneTextureId {9003};
        textured.textures[normalTexture].firstTexel = static_cast<std::uint32_t>(normalTexel);
        textured.textureTexels[firstRoleTexel] = {128, 96, 192, 255};
        textured.textureTexels[secondRoleTexel] = textured.textureTexels[firstRoleTexel];
        textured.textureTexels[normalTexel] = {255, 128, 255, 255};
        const auto sharedImage = textured.textures[firstRoleTexture].id;
        material->baseColorTexture = sharedImage;
        material->metallicRoughnessTexture = sharedImage;
        material->normalTexture = textured.textures[normalTexture].id;
        material->occlusionTexture = {};
        material->emissiveTexture = {};
        material->emissive = {};
        const auto allRoles = render(backend, textured);
        auto splitRoleReference = textured;
        auto* splitMaterial = const_cast<SceneMaterialRecord*>(visual3d_detail::findById(
            splitRoleReference.materials, splitRoleReference.materialCount, meshObject->material));
        splitMaterial->baseColorTexture = splitRoleReference.textures[secondRoleTexture].id;
        const auto splitRoleRender = render(backend, splitRoleReference);
        auto wronglyLinearBase = splitRoleReference;
        wronglyLinearBase.textureTexels[secondRoleTexel] = {188, 165, 225, 255};
        const auto wronglyLinearBaseRender = render(backend, wronglyLinearBase);
        auto wronglySrgbMetallicRoughness = splitRoleReference;
        wronglySrgbMetallicRoughness.textureTexels[firstRoleTexel] = {55, 30, 134, 255};
        const auto wronglySrgbMetallicRoughnessRender = render(backend, wronglySrgbMetallicRoughness);

        auto tangentFixture = textured;
        tangentFixture.objects[0] = *meshObject;
        tangentFixture.objects[0].parent = {};
        tangentFixture.objects[0].transform = {};
        tangentFixture.objectCount = 1;
        auto tangentWithoutNormal = tangentFixture;
        auto* tangentWithoutNormalMaterial = const_cast<SceneMaterialRecord*>(visual3d_detail::findById(
            tangentWithoutNormal.materials, tangentWithoutNormal.materialCount,
            tangentWithoutNormal.objects[0].material));
        tangentWithoutNormalMaterial->normalTexture = {};
        const auto tangentWithoutNormalRender = render(backend, tangentWithoutNormal);
        const auto tangentFixtureRender = render(backend, tangentFixture);
        auto reflected = tangentFixture;
        auto* reflectedObject = std::find_if(reflected.objects.begin(),
            reflected.objects.begin() + reflected.objectCount,
            [] (const auto& object) { return object.indexCount != 0; });
        reflectedObject->transform.scale.x *= -1.0f;
        const auto reflectedNormal = render(backend, reflected);
        auto analyticReflection = tangentFixture;
        const auto& analyticObject = analyticReflection.objects[0];
        for (std::size_t index = analyticObject.firstVertex;
             index < analyticObject.firstVertex + analyticObject.vertexCount; ++index)
        {
            analyticReflection.vertices[index].position.x *= -1.0f;
            analyticReflection.vertices[index].normal.x *= -1.0f;
            analyticReflection.vertices[index].tangent[0] *= -1.0f;
            analyticReflection.vertices[index].tangent[3] *= -1.0f;
        }
        for (std::size_t index = analyticObject.firstIndex;
             index + 2 < analyticObject.firstIndex + analyticObject.indexCount; index += 3)
            std::swap(analyticReflection.indices[index], analyticReflection.indices[index + 1]);
        const auto analyticReflectionRender = render(backend, analyticReflection);
        auto reflectedCulling = reflected;
        auto* cullingMaterial = const_cast<SceneMaterialRecord*>(visual3d_detail::findById(
            reflectedCulling.materials, reflectedCulling.materialCount, reflectedObject->material));
        cullingMaterial->normalTexture = {};
        const auto reflectedCullingRender = render(backend, reflectedCulling);

        auto blend = textured;
        for (std::size_t index = 0; index < blend.materialCount; ++index)
        {
            blend.materials[index].alphaMode = SceneAlphaMode::Blend;
            blend.materials[index].opacity = 0.5f;
        }
        for (std::size_t index = 0; index < blend.objectCount; ++index)
            if (blend.objects[index].indexCount != 0)
                blend.objects[index].transform.translation = {};
        const auto blendA = render(backend, blend);
        const auto blendB = render(backend, blend);
        auto reversedBlend = blend;
        std::reverse(reversedBlend.objects.begin(),
                     reversedBlend.objects.begin() + reversedBlend.objectCount);
        const auto blendReversed = render(backend, reversedBlend);

        auto scene = std::make_shared<const Visual3DScene>(*adapted);
        const auto preparedAgain = backend.prepare(scene, nullptr);
        const auto repeatA = backend.render(scene, preparedAgain.resources, 96, 96, {});
        const auto repeatB = backend.render(scene, preparedAgain.resources, 96, 96, {});
        const auto preparedThird = backend.prepare(scene, nullptr);
        const auto repeatC = backend.render(scene, preparedThird.resources, 96, 96, {});

        bool ok = true;
        ok &= check(baseline.preparation.prepared && baseline.render.rendered
                        && !baseline.pixels.empty(),
                    "the checked-in rich GLB must render through native Metal");
        ok &= check(flattenedRender.pixels != baseline.pixels,
                    "nested transform-only GLB hierarchy must affect Metal pixels");
        ok &= check(spotInside.pixels != spotOutside.pixels,
                    "Metal pixels must respond to the spot cone");
        ok &= check(infiniteRange.pixels != rangeCutoff.pixels,
                    "Metal pixels must respond to finite light range cutoff");
        ok &= check(allRoles.pixels == splitRoleRender.pixels
                        && allRoles.pixels != wronglyLinearBaseRender.pixels
                        && allRoles.pixels != wronglySrgbMetallicRoughnessRender.pixels,
                    "one GLB image must execute in simultaneous sRGB and linear texture roles");
        ok &= check(tangentFixtureRender.pixels != tangentWithoutNormalRender.pixels,
                    "Metal tangent fixture must keep normal mapping active");
        ok &= check(reflectedNormal.pixels == analyticReflectionRender.pixels,
                    "Metal reflected tangents must match the analytic handedness reference");
        ok &= check(nonBackgroundCount(reflectedCullingRender.pixels) > 0,
                    "Metal reflected single-sided geometry must survive culling");
        ok &= check(blendA.pixels == blendB.pixels
                        && blendA.pixels != blendReversed.pixels,
                    "overlapping Metal alpha draws must retain stable source order");
        ok &= check(repeatA.rendered && repeatB.rendered && repeatC.rendered
                        && repeatA.frame->colorTextureDescriptor().rendererGeneration
                            == repeatB.frame->colorTextureDescriptor().rendererGeneration
                        && repeatA.frame->colorTextureDescriptor().rendererGeneration
                            != repeatC.frame->colorTextureDescriptor().rendererGeneration,
                    "Metal renderer generation must identify one preparation lifetime");
        if (!ok)
            return 1;
        std::cout << "Metal rich static GLB backend: PASS; device=" << info.device << '\n';
        return 0;
    }
}
