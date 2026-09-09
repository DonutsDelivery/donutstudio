#include "../src/gpu_backend/backend.h"
#include "../src/diffraction_material_admission.h"
#include "../src/diffraction_material_execution.h"
#include "../src/fixture_scene_renderer.h"
#include "support/fixture_scene.h"
#include "support/surface_material_starter_oracle.h"
#include "../src/sha256.h"
#include "../../shared/DiffractionMaterialPresets.h"
#include "../../shared/generated/SurfaceMaterialStarterPrograms.h"

#include "sokol_gfx.h"

#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#if defined(ARBIT_HOLOGRAPHIC_TRADING_CARD_STRICT) \
    && ARBIT_HOLOGRAPHIC_TRADING_CARD_STRICT
#include <cstddef>
#endif
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace
{
bool expect(bool condition, const char* message)
{
    if (!condition)
        std::cerr << "FAIL: " << message << '\n';
    return condition;
}

std::vector<std::uint8_t> readBgra8(
    const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& frame)
{
    if (!frame || frame->colorImageHandle() == 0)
        return {};
    sg_image image {};
    image.id = static_cast<std::uint32_t>(frame->colorImageHandle());
    const auto imageInfo = sg_mtl_query_image_info(image);
    if (imageInfo.active_slot < 0 || imageInfo.active_slot >= SG_NUM_INFLIGHT_FRAMES)
        return {};
    id<MTLTexture> texture = (__bridge id<MTLTexture>) imageInfo.tex[imageInfo.active_slot];
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>) sg_mtl_command_queue();
    if (texture == nil || queue == nil || texture.pixelFormat != MTLPixelFormatBGRA8Unorm)
        return {};

    const auto tightRowBytes = static_cast<std::size_t>(frame->width()) * 4u;
    const auto rowBytes = (tightRowBytes + 255u) & ~std::size_t(255u);
    id<MTLBuffer> buffer = [texture.device newBufferWithLength:rowBytes * frame->height()
                                                   options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    if (buffer == nil || command == nil || blit == nil)
    {
#if ! __has_feature(objc_arc)
        [buffer release];
#endif
        return {};
    }
    [blit copyFromTexture:texture
              sourceSlice:0
              sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(frame->width(), frame->height(), 1)
                 toBuffer:buffer
        destinationOffset:0
   destinationBytesPerRow:rowBytes
 destinationBytesPerImage:rowBytes * frame->height()];
    [blit endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted)
    {
        std::cerr << "Metal fixture readback command failed";
        if (command.error != nil && command.error.localizedDescription != nil)
            std::cerr << ": " << command.error.localizedDescription.UTF8String;
        std::cerr << '\n';
#if ! __has_feature(objc_arc)
        [buffer release];
#endif
        return {};
    }

    std::vector<std::uint8_t> pixels(tightRowBytes * frame->height());
    const auto* source = static_cast<const std::uint8_t*>(buffer.contents);
    if (source != nullptr)
        for (std::uint32_t y = 0; y < frame->height(); ++y)
            std::copy_n(source + y * rowBytes, tightRowBytes,
                        pixels.data() + y * tightRowBytes);
#if ! __has_feature(objc_arc)
    [buffer release];
#endif
    return pixels;
}

std::shared_ptr<arbitgpu::NativeFixtureSurfaceMaterialProgram> makeProgram(
    const diffractionmaterial::Description& description,
    HarmonicMIDI::grid::SceneObjectId object,
    const char* identity)
{
    (void) identity;
    std::string error;
    const auto admitted = diffractionmaterial::admit(description, error);
    diffractionmaterial::SpectralIncidentLight incident;
    incident.radiance.fill(0.01f);
    diffractionmaterial::SpectralLightingPath lightingPath;
    lightingPath.kind = diffractionmaterial::LightingPathKind::Direct;
    lightingPath.incident = incident;
    auto program = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>();
    program->backend = arbitgpu::NativeFixtureMaterialBackend::Metal;
    program->kind = arbitgpu::NativeFixtureMaterialKind::DiffractionReflective;
    program->object = object;
    program->bindingDigest = admitted ? admitted->structuralDigest() : std::string {};

    program->diffractionPathCount = 1;
    diffractionmaterial::LightingDescription lightingDescription;
    lightingDescription.pathCount = 1;
    lightingDescription.paths[0] = lightingPath;
    const auto admittedLighting = diffractionmaterial::AdmittedLightingPlan::admit(
        lightingDescription, error);
    if (admittedLighting)
    {
        program->diffractionLightingAdmission
            = std::make_shared<const diffractionmaterial::AdmittedLightingPlan>(
                *admittedLighting);
        program->programIdentity = arbitgpu::nativeFixtureDiffractionProgramIdentity(
            program->bindingDigest, *admittedLighting);
    }
    if (admitted)
        program->diffractionPaths[0] = diffractionmaterial::makeGpuLightingPath(
            diffractionmaterial::physicalcheckpoint::makeGpuParameters(*admitted, incident),
            lightingPath);

    return program;
}

bool backgroundOnly(const std::vector<std::uint8_t>& pixels)
{
    for (std::size_t index = 0; index + 3 < pixels.size(); index += 4)
        if (pixels[index] != 18u || pixels[index + 1] != 10u
            || pixels[index + 2] != 7u || pixels[index + 3] != 255u)
            return false;
    return true;
}

std::size_t nonBackgroundPixelCount(const std::vector<std::uint8_t>& pixels)
{
    std::size_t count = 0;
    for (std::size_t index = 0; index + 3 < pixels.size(); index += 4)
        if (pixels[index] != 18u || pixels[index + 1] != 10u
            || pixels[index + 2] != 7u || pixels[index + 3] != 255u)
            ++count;
    return count;
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

#if defined(ARBIT_HOLOGRAPHIC_TRADING_CARD_STRICT) \
    && ARBIT_HOLOGRAPHIC_TRADING_CARD_STRICT
std::size_t differingPixelCount(const std::vector<std::uint8_t>& left,
                                const std::vector<std::uint8_t>& right)
{
    if (left.size() != right.size() || left.size() % 4u != 0)
        return 0;
    std::size_t count = 0;
    for (std::size_t offset = 0; offset < left.size(); offset += 4u)
        if (!std::equal(left.begin() + static_cast<std::ptrdiff_t>(offset),
                        left.begin() + static_cast<std::ptrdiff_t>(offset + 4u),
                        right.begin() + static_cast<std::ptrdiff_t>(offset)))
            ++count;
    return count;
}
#endif
} // namespace

int main()
{
#if ! defined(__aarch64__) && ! defined(__arm64__)
    std::cerr << "FAIL: strict fixture material target requires physical Apple Silicon\n";
    return EXIT_FAILURE;
#else
    @autoreleasepool
    {
        auto& backend = arbitgpu::nativeFixtureSceneBackend();
        const auto info = backend.info();
        if (!info.available || info.backend != "metal")
        {
            std::cerr << "FAIL: strict physical Metal fixture backend unavailable: "
                      << info.error << '\n';
            return EXIT_FAILURE;
        }

        auto rasterSceneValue = videohelper::fixture3d::makeScene();
        rasterSceneValue.vertexCount = 3;
        rasterSceneValue.indexCount = 3;
        // Metal treats counter-clockwise clip-space triangles as front-facing here.
        // Keep the indexed control winding explicit so either index-order or
        // face-winding mutations break the paired positive/negative controls.
        rasterSceneValue.indices[0] = 0;
        rasterSceneValue.indices[1] = 1;
        rasterSceneValue.indices[2] = 2;
        rasterSceneValue.objects[0].vertexCount = 3;
        rasterSceneValue.objects[0].indexCount = 3;
        rasterSceneValue.objects[0].transform = {};
        rasterSceneValue.materials[0].baseColor = { 1.0f, 0.0f, 0.0f };
        rasterSceneValue.materials[0].baseColorTexture = {};
        rasterSceneValue.materials[0].doubleSided = true;
        rasterSceneValue.textureCount = 0;
        rasterSceneValue.textureTexelCount = 0;
        rasterSceneValue.lightCount = 0;
        rasterSceneValue.ambientColor = { 1.0f, 1.0f, 1.0f };
        auto rasterScene = std::make_shared<const HarmonicMIDI::grid::Visual3DScene>(
            rasterSceneValue);
        const auto rasterPreparation = backend.prepare(rasterScene, nullptr);
        const auto rasterFrame = backend.render(
            rasterScene, rasterPreparation.resources, 96, 96, {});
        const auto rasterPixels = readBgra8(rasterFrame.frame);

        rasterSceneValue.materials[0].doubleSided = false;
        auto frontFacingRasterScene
            = std::make_shared<const HarmonicMIDI::grid::Visual3DScene>(rasterSceneValue);
        const auto frontFacingRasterPreparation = backend.prepare(frontFacingRasterScene, nullptr);
        const auto frontFacingRasterFrame = backend.render(
            frontFacingRasterScene, frontFacingRasterPreparation.resources, 96, 96, {});
        const auto frontFacingRasterPixels = readBgra8(frontFacingRasterFrame.frame);

        auto backFacingRasterSceneValue = rasterSceneValue;
        backFacingRasterSceneValue.indices[1] = 2;
        backFacingRasterSceneValue.indices[2] = 1;
        auto backFacingRasterScene
            = std::make_shared<const HarmonicMIDI::grid::Visual3DScene>(backFacingRasterSceneValue);
        const auto backFacingRasterPreparation = backend.prepare(backFacingRasterScene, nullptr);
        const auto backFacingRasterFrame = backend.render(
            backFacingRasterScene, backFacingRasterPreparation.resources, 96, 96, {});
        const auto backFacingRasterPixels = readBgra8(backFacingRasterFrame.frame);

        const auto rasterControlOk = expect(
            rasterPreparation.prepared && rasterFrame.rendered
                && rasterPixels.size() == 96u * 96u * 4u
                && nonBackgroundPixelCount(rasterPixels) > 0,
            "the double-sided constant-red triangle must prove indexed Metal rasterization and BGRA readback")
            && expect(frontFacingRasterPreparation.prepared && frontFacingRasterFrame.rendered
                && frontFacingRasterPixels.size() == 96u * 96u * 4u
                && nonBackgroundPixelCount(frontFacingRasterPixels) > 0,
                "the counter-clockwise front-facing triangle must survive production back-face culling")
            && expect(backFacingRasterPreparation.prepared && backFacingRasterFrame.rendered
                && backFacingRasterPixels.size() == 96u * 96u * 4u
                && nonBackgroundPixelCount(backFacingRasterPixels) == 0,
                "the reversed-winding back-facing triangle must be fully culled by the production pipeline");

        auto surfaceSceneValue = videohelper::fixture3d::makeScene();
        surfaceSceneValue.objects[0].transform.rotation
            = { -0.16773126f, 0.25488700f, 0.04494346f, 0.95125124f };
        surfaceSceneValue.lightCount = 1;
        surfaceSceneValue.ambientColor = { 0.12f, 0.12f, 0.12f };
        surfaceSceneValue.lights[0].kind = HarmonicMIDI::grid::SceneLightKind::Directional;
        surfaceSceneValue.lights[0].transform = {};
        surfaceSceneValue.lights[0].transform.rotation
            = surfaceSceneValue.objects[0].transform.rotation;
        surfaceSceneValue.lights[0].color = { 1.0f, 1.0f, 1.0f };
        surfaceSceneValue.lights[0].intensity = 0.88f;
        surfaceSceneValue.materials[0].baseColorTexture = {};

        const auto oracle = surfacematerialstarteroracle::load(
            SURFACE_MATERIAL_STARTER_ORACLE_PATH);
        bool starterOk = true;
        std::vector<std::uint8_t> sampledStarterPixels;
        const auto center = (32u * 64u + 32u) * 4u;
        for (std::size_t starterIndex = 0;
             starterIndex < surfacematerialstarterfixture::kPrograms.size(); ++starterIndex)
        {
            const auto& starter = surfacematerialstarterfixture::kPrograms[starterIndex];
            const auto& expected = oracle.materials[starterIndex];
            starterOk = expect(expected.id == starter.id,
                               "generated programs and independent JSON oracle must share stable IDs")
                && starterOk;
            auto starterSceneValue = surfaceSceneValue;
            starterSceneValue.materials[0].id = {starter.materialIdValue()};
            starterSceneValue.objects[0].material = {starter.materialIdValue()};
            const auto starterScene
                = std::make_shared<const HarmonicMIDI::grid::Visual3DScene>(
                    std::move(starterSceneValue));

            surfacematerialbinding::ImportedSceneMaterialRequest request;
            request.scene = starterScene->id;
            request.sceneRevision = 1;
            request.structuralRevision = 1;
            request.evaluationRevision = 1;
            request.programRevision = 1;
            request.sceneSnapshot = starterScene;
            request.program = starter.program();
            request.binding.targetKind
                = surfacematerialbinding::BindingTargetKind::ObjectOverride;
            request.binding.object = videohelper::fixture3d::kCubeObjectId;
            request.binding.surfaceMaterialRevision = 1;
            std::string starterError;
            const auto admittedProgram = surfacematerial::admit(request.program, starterError);
            if (admittedProgram.has_value())
                request.binding.surfaceMaterialDigest = admittedProgram->structuralDigest();
            const auto binding = videorender::fixture3d::admitSurfaceMaterialBinding(
                starterScene, request, videohelper::materialprogram::BackendTarget::Metal,
                starterError);
            starterOk = expect(binding != nullptr,
                               "each generated exact starter program must pass Metal admission")
                && starterOk;
            if (binding == nullptr)
                continue;

            const auto& block = binding->nativeProgram()->parameters;
            const std::array<std::array<float, 4>, 4> actualFloatLanes {{
                block.baseColorMetallic, block.emissionRoughness,
                block.normalOpacity, block.transmissionIorClearcoat
            }};
            bool exactBlock = true;
            for (std::size_t lane = 0; lane < actualFloatLanes.size(); ++lane)
                for (std::size_t channel = 0; channel < 4; ++channel)
                    exactBlock &= std::abs(actualFloatLanes[lane][channel]
                        - expected.pbr[lane * 4 + channel]) < 0.000001f;
            exactBlock &= block.identifiers[0]
                == static_cast<std::uint32_t>(expected.pbr[16]);
            starterOk = expect(exactBlock,
                               "native Metal lowering must match the independent JSON PBR oracle")
                && starterOk;

            videorender::fixture3d::FixtureSceneRenderer previewOwner(backend);
            videorender::fixture3d::FixtureSceneRenderer exportOwner(backend);
            videorender::fixture3d::RenderedFrame preview;
            videorender::fixture3d::RenderedFrame exportFrame;
            const bool previewRendered = previewOwner.renderPreview(
                starterScene, binding, { 64, 64 },
                videorender::fixture3d::kNativeGpuCapability, preview, starterError);
            const bool exportRendered = exportOwner.renderExport(
                starterScene, binding, { 64, 64 },
                videorender::fixture3d::kNativeGpuCapability, exportFrame, starterError);
            const auto previewPixels = previewRendered
                ? readBgra8(preview.nativeFrame) : std::vector<std::uint8_t> {};
            const auto exportPixels = exportRendered
                ? readBgra8(exportFrame.nativeFrame) : std::vector<std::uint8_t> {};
            starterOk = expect(previewRendered && exportRendered
                                   && preview.use == videorender::fixture3d::RenderUse::Preview
                                   && exportFrame.use == videorender::fixture3d::RenderUse::Export
                                   && preview.nativeFrame != nullptr
                                   && exportFrame.nativeFrame != nullptr
                                   && preview.nativeFrame->colorImageHandle()
                                       != exportFrame.nativeFrame->colorImageHandle(),
                               "distinct production preview and export owners must render Metal frames")
                && expect(!previewPixels.empty() && previewPixels == exportPixels,
                          "preview and export owners must produce exact matching Metal pixels")
                && starterOk;
            if (previewPixels.size() > center + 3u)
            {
                const std::array<std::uint8_t, 4> actualRgba {{
                    previewPixels[center + 2u], previewPixels[center + 1u],
                    previewPixels[center], previewPixels[center + 3u]
                }};
                starterOk = expect(actualRgba == expected.centerRgba8,
                                   "Metal center pixels must match the pinned JSON oracle")
                    && expect(std::find(expected.rejectedAlternatives.begin(),
                                        expected.rejectedAlternatives.end(), actualRgba)
                                  == expected.rejectedAlternatives.end(),
                              "Metal pixels must reject wrong shading alternatives")
                    && starterOk;
                sampledStarterPixels.insert(sampledStarterPixels.end(),
                                            actualRgba.begin(), actualRgba.end());

                const auto rejectsControlledMaterialAlternative = [&] (const char* label,
                                                                        auto mutate)
                {
                    auto alternative = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(
                        *binding->nativeProgram());
                    alternative->bindingDigest += label;
                    alternative->programIdentity += label;
                    mutate(alternative->parameters);
                    const auto prepared = backend.prepare(starterScene, alternative);
                    if (!prepared.prepared) return false;
                    const auto rendered = backend.render(starterScene, prepared.resources, 64, 64, {});
                    const auto pixels = rendered.rendered
                        ? readBgra8(rendered.frame) : std::vector<std::uint8_t> {};
                    if (pixels.size() <= center + 3u) return false;
                    const std::array<std::uint8_t, 4> value {{
                        pixels[center + 2u], pixels[center + 1u],
                        pixels[center], pixels[center + 3u] }};
                    return value != expected.centerRgba8;
                };
                if (starterIndex == 0)
                {
                    starterOk = expect(rejectsControlledMaterialAlternative("/base-color", [] (auto& p)
                        { p.baseColorMetallic[0] = 0.9f; }),
                        "Metal lit pixels must detect a wrong base-color binding") && starterOk;
                    starterOk = expect(rejectsControlledMaterialAlternative("/roughness", [] (auto& p)
                        { p.emissionRoughness[3] = 0.0f; }),
                        "Metal lit pixels must detect a wrong roughness binding") && starterOk;
                    starterOk = expect(rejectsControlledMaterialAlternative("/emission", [] (auto& p)
                        { p.emissionRoughness[0] = 0.25f; }),
                        "Metal lit pixels must detect a wrong emission binding") && starterOk;
                }
                if (starterIndex == 1)
                    starterOk = expect(rejectsControlledMaterialAlternative("/metallic", [] (auto& p)
                        { p.baseColorMetallic[3] = 0.0f; }),
                        "Metal lit pixels must detect a wrong metallic binding") && starterOk;
                if (starterIndex == 2)
                    starterOk = expect(rejectsControlledMaterialAlternative("/opacity", [] (auto& p)
                        { p.normalOpacity[3] = 1.0f; }),
                        "Metal lit pixels must detect a wrong opacity binding") && starterOk;

                if (starterIndex != 0)
                    continue;
                auto wrongNormalSceneValue = *starterScene;
                for (std::size_t vertex = 0; vertex < wrongNormalSceneValue.vertexCount; ++vertex)
                    wrongNormalSceneValue.vertices[vertex].normal
                        = { -wrongNormalSceneValue.vertices[vertex].normal.x,
                            -wrongNormalSceneValue.vertices[vertex].normal.y,
                            -wrongNormalSceneValue.vertices[vertex].normal.z };
                const auto wrongNormalScene
                    = std::make_shared<const HarmonicMIDI::grid::Visual3DScene>(
                        std::move(wrongNormalSceneValue));
                const auto wrongNormalPreparation = backend.prepare(
                    wrongNormalScene, binding->nativeProgram());
                std::vector<std::uint8_t> wrongNormalPixels;
                if (wrongNormalPreparation.prepared)
                {
                    const auto wrongNormalRender = backend.render(
                        wrongNormalScene, wrongNormalPreparation.resources, 64, 64, {});
                    if (wrongNormalRender.rendered)
                        wrongNormalPixels = readBgra8(wrongNormalRender.frame);
                }
                const std::array<std::uint8_t, 4> wrongNormalRgba {{
                    wrongNormalPixels.size() > center + 3u ? wrongNormalPixels[center + 2u] : std::uint8_t { 0 },
                    wrongNormalPixels.size() > center + 3u ? wrongNormalPixels[center + 1u] : std::uint8_t { 0 },
                    wrongNormalPixels.size() > center + 3u ? wrongNormalPixels[center] : std::uint8_t { 0 },
                    wrongNormalPixels.size() > center + 3u ? wrongNormalPixels[center + 3u] : std::uint8_t { 0 } }};
                starterOk = expect(wrongNormalPixels.size() > center + 3u
                                       && wrongNormalRgba != actualRgba,
                                   "Metal lit pixels must detect a wrong transformed normal")
                    && starterOk;
            }
        }
        const auto sampledDigest = fnv1a64(sampledStarterPixels);
        char sampledDigestText[17] {};
        std::snprintf(sampledDigestText, sizeof(sampledDigestText), "%016llx",
                      static_cast<unsigned long long>(sampledDigest));
        starterOk = expect(sampledStarterPixels.size() == 20
                               && oracle.sampledRgba8Fnv1a64 == sampledDigestText,
                           "all five Metal samples must match the pinned JSON digest")
            && starterOk;
        videohelper::Sha256 sampledSha256;
        sampledSha256.update(sampledStarterPixels.data(), sampledStarterPixels.size());
        starterOk = expect(oracle.sampledRgba8Sha256 == sampledSha256.finishHex(),
                           "all five Metal samples must match the pinned SHA-256 digest")
            && starterOk;

        auto malformedSceneValue = surfaceSceneValue;
        malformedSceneValue.lightCount = 0;
        malformedSceneValue.materials[0].id = { 1u };
        malformedSceneValue.objects[0].material = { 1u };
        const auto malformedScene
            = std::make_shared<const HarmonicMIDI::grid::Visual3DScene>(malformedSceneValue);
        surfacematerialbinding::ImportedSceneMaterialRequest malformedRequest;
        malformedRequest.scene = malformedScene->id;
        malformedRequest.sceneRevision = 1;
        malformedRequest.structuralRevision = 1;
        malformedRequest.evaluationRevision = 1;
        malformedRequest.programRevision = 1;
        malformedRequest.sceneSnapshot = malformedScene;
        malformedRequest.program = surfacematerialstarterfixture::kPrograms[0].program();
        malformedRequest.binding.targetKind
            = surfacematerialbinding::BindingTargetKind::ObjectOverride;
        malformedRequest.binding.object = videohelper::fixture3d::kCubeObjectId;
        malformedRequest.binding.surfaceMaterialRevision = 1;
        std::string malformedError;
        const auto validMalformedBase = surfacematerial::admit(
            malformedRequest.program, malformedError);
        if (validMalformedBase.has_value())
            malformedRequest.binding.surfaceMaterialDigest
                = validMalformedBase->structuralDigest();
        malformedRequest.program.operations[0].literal[0]
            = std::numeric_limits<float>::quiet_NaN();
        const auto malformedValueBinding
            = videorender::fixture3d::admitSurfaceMaterialBinding(
                malformedScene, malformedRequest,
                videohelper::materialprogram::BackendTarget::Metal, malformedError);
        starterOk = expect(malformedValueBinding == nullptr,
                           "malformed Surface PBR values must reject before Metal allocation")
            && starterOk;
        malformedRequest.program = surfacematerialstarterfixture::kPrograms[0].program();
        malformedRequest.binding.surfaceMaterialDigest = std::string(64, '0');
        const auto malformedIdentityBinding
            = videorender::fixture3d::admitSurfaceMaterialBinding(
                malformedScene, malformedRequest,
                videohelper::materialprogram::BackendTarget::Metal, malformedError);
        starterOk = expect(malformedIdentityBinding == nullptr,
                           "malformed Surface PBR identity must reject before Metal allocation")
            && starterOk;

        std::string cardLoadError;
        const auto cardAsset = videohelper::fixture3d::loadHolographicTradingCardScene(
            cardLoadError);
        if (!cardAsset)
        {
            std::cerr << "FAIL: exact card GLB production decode/adaptation failed: "
                      << cardLoadError << '\n';
            return EXIT_FAILURE;
        }
        const auto& scene = cardAsset->scene;
        bool ok = expect(cardAsset->assetId
                             == "builtin.visual-model.holographic-trading-card"
                         && cardAsset->contentSha256
                             == "5e3433aa19cc4636c33a1f6dd53a4a05fdffa39d788ad565b530d8dfdcce0287"
                         && cardAsset->sceneName == "Holographic Trading Card Scene"
                         && cardAsset->objectName == "Trading Card"
                         && cardAsset->materialName == "Physical Diffraction Target"
                         && cardAsset->cameraName == "Card Camera"
                         && cardAsset->lightName == "Card Key"
                         && scene->vertexCount == 24u && scene->indexCount == 36u,
                         "strict Metal must retain the exact card asset and named decoded records");

#if defined(ARBIT_HOLOGRAPHIC_TRADING_CARD_STRICT) \
    && ARBIT_HOLOGRAPHIC_TRADING_CARD_STRICT
        const auto& cardRequest = *cardAsset->operation.diffractionMaterial;
        std::string previewError, exportError, wrongObjectError, strictError;
        const auto previewBinding = videorender::fixture3d::admitDiffractionMaterialBinding(
            scene, cardRequest, videohelper::materialprogram::BackendTarget::Metal,
            previewError);
        const auto exportBinding = videorender::fixture3d::admitDiffractionMaterialBinding(
            scene, cardRequest, videohelper::materialprogram::BackendTarget::Metal,
            exportError);
        auto zeroSlope = cardRequest;
        zeroSlope.spatialFoil->physicalBsdf.roughness.rmsSlope = 0.0f;
        zeroSlope.material = zeroSlope.spatialFoil->physicalBsdf;
        const auto admittedZeroSlope = diffractivefoil::admit(*zeroSlope.spatialFoil,
                                                              strictError);
        zeroSlope.structuralDigest = admittedZeroSlope
            ? admittedZeroSlope->structuralDigest() : std::string {};
        const auto zeroSlopeBinding = videorender::fixture3d::admitDiffractionMaterialBinding(
            scene, zeroSlope, videohelper::materialprogram::BackendTarget::Metal,
            strictError);
        auto wrongObject = cardRequest;
        wrongObject.object = HarmonicMIDI::grid::SceneObjectId { 2 };
        const auto wrongObjectBinding = videorender::fixture3d::admitDiffractionMaterialBinding(
            scene, wrongObject, videohelper::materialprogram::BackendTarget::Metal,
            wrongObjectError);
        auto changedGroove = cardRequest;
        changedGroove.spatialFoil->grooveField[2].grooveSpacingNanometres += 113.0f;
        const auto admittedGroove = diffractivefoil::admit(*changedGroove.spatialFoil, strictError);
        changedGroove.structuralDigest = admittedGroove
            ? admittedGroove->structuralDigest() : std::string {};
        const auto grooveBinding = videorender::fixture3d::admitDiffractionMaterialBinding(
            scene, changedGroove, videohelper::materialprogram::BackendTarget::Metal,
            strictError);
        auto changedOccupancy = cardRequest;
        changedOccupancy.spatialFoil->grooveField[2].diffractionCoverage = 0.0f;
        const auto admittedOccupancy = diffractivefoil::admit(
            *changedOccupancy.spatialFoil, strictError);
        changedOccupancy.structuralDigest = admittedOccupancy
            ? admittedOccupancy->structuralDigest() : std::string {};
        const auto occupancyBinding = videorender::fixture3d::admitDiffractionMaterialBinding(
            scene, changedOccupancy, videohelper::materialprogram::BackendTarget::Metal,
            strictError);
        videorender::fixture3d::FixtureSceneRenderer previewOwner(backend);
        videorender::fixture3d::FixtureSceneRenderer exportOwner(backend);
        videorender::fixture3d::FixtureSceneRenderer grooveOwner(backend);
        videorender::fixture3d::FixtureSceneRenderer occupancyOwner(backend);
        videorender::fixture3d::RenderedFrame previewFrame, exportFrame, grooveFrame,
            occupancyFrame;
        const auto capability = videorender::fixture3d::kNativeGpuCapability;
        const bool rendered = previewBinding && exportBinding && grooveBinding
            && occupancyBinding && previewBinding != exportBinding
            && previewBinding->nativeProgram() != exportBinding->nativeProgram()
            && wrongObjectBinding == nullptr
            && previewOwner.renderPreview(scene, previewBinding, {}, { 320, 180 },
                                          capability, previewFrame, strictError)
            && exportOwner.renderExport(scene, exportBinding, {}, { 320, 180 },
                                        capability, exportFrame, strictError)
            && grooveOwner.renderPreview(scene, grooveBinding, {}, { 320, 180 },
                                         capability, grooveFrame, strictError)
            && occupancyOwner.renderPreview(scene, occupancyBinding, {}, { 320, 180 },
                                            capability, occupancyFrame, strictError);
        const auto previewPixels = readBgra8(previewFrame.nativeFrame);
        const auto exportPixels = readBgra8(exportFrame.nativeFrame);
        const auto groovePixels = readBgra8(grooveFrame.nativeFrame);
        const auto occupancyPixels = readBgra8(occupancyFrame.nativeFrame);
        const auto disabledPreparation = backend.prepare(scene, nullptr);
        const auto disabledFrame = backend.render(
            scene, disabledPreparation.resources, 320, 180, {});
        const auto disabledPixels = readBgra8(disabledFrame.frame);
        const auto previewOwnerId = previewFrame.nativeFrame
            ? previewFrame.nativeFrame->nativeResourceCacheIdentity() : 0;
        const auto exportOwnerId = exportFrame.nativeFrame
            ? exportFrame.nativeFrame->nativeResourceCacheIdentity() : 0;
        const auto grooveDifferingPixels = differingPixelCount(previewPixels, groovePixels);
        const auto occupancyDifferingPixels = differingPixelCount(previewPixels, occupancyPixels);
        const auto disabledDifferingPixels = differingPixelCount(previewPixels, disabledPixels);
        const bool strictCardAccepted = rendered && disabledPreparation.prepared
            && disabledFrame.rendered && zeroSlopeBinding == nullptr
            && previewOwnerId != 0 && exportOwnerId != 0 && previewOwnerId != exportOwnerId
            && previewBinding->bindingDigest() == exportBinding->bindingDigest()
            && previewBinding->nativeProgram()->programIdentity
                == exportBinding->nativeProgram()->programIdentity
            && previewPixels.size() == 320u * 180u * 4u
            && exportPixels.size() == previewPixels.size()
            && groovePixels.size() == previewPixels.size()
            && occupancyPixels.size() == previewPixels.size()
            && disabledPixels.size() == previewPixels.size()
            && previewPixels == exportPixels
            && grooveDifferingPixels > 100u
            && occupancyDifferingPixels > 100u
            && disabledDifferingPixels > 100u;
        if (!strictCardAccepted)
        {
            std::cerr << "Strict Metal card operands: rendered=" << rendered
                      << " disabled-prepared=" << disabledPreparation.prepared
                      << " disabled-rendered=" << disabledFrame.rendered
                      << " preview-owner=" << previewOwnerId
                      << " export-owner=" << exportOwnerId
                      << " source-operation=" << cardAsset->operation.sourceStableId
                      << " render-operation=" << cardAsset->operation.renderStableId
                      << " preview-bytes=" << previewPixels.size()
                      << " export-bytes=" << exportPixels.size()
                      << " groove-bytes=" << groovePixels.size()
                      << " occupancy-bytes=" << occupancyPixels.size()
                      << " disabled-bytes=" << disabledPixels.size()
                      << " preview-digest=" << std::hex << fnv1a64(previewPixels)
                      << " export-digest=" << fnv1a64(exportPixels)
                      << " groove-digest=" << fnv1a64(groovePixels)
                      << " occupancy-digest=" << fnv1a64(occupancyPixels)
                      << " disabled-digest=" << fnv1a64(disabledPixels) << std::dec
                      << " groove-differing-pixels=" << grooveDifferingPixels
                      << " occupancy-differing-pixels=" << occupancyDifferingPixels
                      << " disabled-differing-pixels=" << disabledDifferingPixels
                      << " preview-error=" << previewError
                      << " export-error=" << exportError
                      << " wrong-object-error=" << wrongObjectError
                      << " final-error=" << strictError
                      << " preview-program="
                      << (previewBinding ? previewBinding->nativeProgram()->programIdentity
                                         : std::string {})
                      << " export-program="
                      << (exportBinding ? exportBinding->nativeProgram()->programIdentity
                                        : std::string {})
                      << '\n';
        }
        else
        {
            std::cout << "Metal holographic card: PASS; preview-owner=" << previewOwnerId
                      << "; export-owner=" << exportOwnerId
                      << "; source-operation=" << cardAsset->operation.sourceStableId
                      << "; render-operation=" << cardAsset->operation.renderStableId
                      << "; bytes=" << previewPixels.size()
                      << "; preview-export-digest=" << std::hex << fnv1a64(previewPixels)
                      << "; groove-digest=" << fnv1a64(groovePixels)
                      << "; occupancy-digest=" << fnv1a64(occupancyPixels)
                      << "; disabled-digest=" << fnv1a64(disabledPixels) << std::dec
                      << "; groove-differing-pixels=" << grooveDifferingPixels
                      << "; occupancy-differing-pixels=" << occupancyDifferingPixels
                      << "; disabled-differing-pixels=" << disabledDifferingPixels
                      << "; zero-slope-rejected=1\n";
        }
        ok &= expect(strictCardAccepted,
                     "strict Metal card fixture must bind one handler operation to distinct preview/export owners and meaningful same-size alternatives");
        return ok ? 0 : 1;
#endif

        auto description = diffractionmaterial::makeAluminiumBinaryGratingPreset();
        auto program = makeProgram(description, scene->objects[0].id,
                                   "metal-card-diffraction-program");
        auto preparation = backend.prepare(scene, program);
        auto replayedLightingProgram
            = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*program);
        auto replayedLightingDescription
            = replayedLightingProgram->diffractionLightingAdmission->description();
        replayedLightingDescription.paths[0].incident.radiance[1]
            = std::nextafter(
                replayedLightingDescription.paths[0].incident.radiance[1], 0.0f);
        std::string replayedLightingError;
        const auto replayedLighting = diffractionmaterial::AdmittedLightingPlan::admit(
            replayedLightingDescription, replayedLightingError);
        const auto replayedMaterial = diffractionmaterial::admit(
            description, replayedLightingError);
        if (replayedLighting && replayedMaterial)
        {
            replayedLightingProgram->diffractionLightingAdmission
                = std::make_shared<const diffractionmaterial::AdmittedLightingPlan>(
                    *replayedLighting);
            replayedLightingProgram->diffractionPaths[0]
                = diffractionmaterial::makeGpuLightingPath(
                    diffractionmaterial::physicalcheckpoint::makeGpuParameters(
                        *replayedMaterial,
                        replayedLightingDescription.paths[0].incident),
                    replayedLightingDescription.paths[0]);
        }
        const auto replayedLightingPreparation = backend.prepare(
            scene, replayedLightingProgram);
        auto malformedProgram
            = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*program);
        malformedProgram->programIdentity = "metal-fixture-malformed-diffraction-program";
        malformedProgram->diffractionPaths[0].material.spectralZ[7].w
            = std::numeric_limits<float>::quiet_NaN();
        const auto malformedPreparation = backend.prepare(scene, malformedProgram);
        auto layoutSentinelProgram
            = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*program);
        layoutSentinelProgram->programIdentity
            = "metal-fixture-layout-sentinel-diffraction-program";
        layoutSentinelProgram->parameters.identifiers[3] = 1u;
        const auto layoutSentinelPreparation = backend.prepare(
            scene, layoutSentinelProgram);
        auto divergentBlazeProgram
            = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*program);
        divergentBlazeProgram->programIdentity
            = "metal-fixture-divergent-local-blaze-program";
        divergentBlazeProgram->diffractionPaths[0].material.control.y = static_cast<float>(
            diffractionmaterial::GrooveProfile::BlazedSawtooth);
        divergentBlazeProgram->diffractionPaths[0].material.grooveField = {
            static_cast<float>(diffractionmaterial::GrooveFieldMode::Linear),
            0.5f, 0.5f, 0.0f
        };
        divergentBlazeProgram->diffractionPaths[0].material.grooveVariation
            = { 1.0f, 0.0f, 100.0f, 0.0f };
        const auto divergentBlazePreparation = backend.prepare(
            scene, divergentBlazeProgram);
        program->diffractionPaths[0].material.roughness.y = 0.0f;
        if (!preparation.prepared)
            std::cerr << "Metal diffraction preparation error: "
                      << preparation.error << '\n';
        auto frame = backend.render(scene, preparation.resources, 96, 96, {});
        if (!frame.rendered)
            std::cerr << "Metal diffraction render error: " << frame.error << '\n';
        const auto pixels = readBgra8(frame.frame);
        auto repeated = backend.render(scene, preparation.resources, 96, 96, {});
        const auto repeatedPixels = readBgra8(repeated.frame);

        auto transportProgram
            = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*program);

        transportProgram->diffractionPathCount = 3;
        transportProgram->diffractionMaximumBounceDepth = 1;
        diffractionmaterial::LightingDescription transportLighting;
        transportLighting.pathCount = 3;
        const std::array<diffractionmaterial::LightingPathKind, 3> transportKinds {
            diffractionmaterial::LightingPathKind::Direct,
            diffractionmaterial::LightingPathKind::Environment,
            diffractionmaterial::LightingPathKind::Indirect };
        const std::array<std::array<float, 3>, 3> transportDirections {{
            {{ 0.0f, 0.0f, 1.0f }}, {{ 0.36f, 0.0f, 0.9329523f }},
            {{ -0.28f, 0.28f, 0.918259f }} }};
        std::string transportError;
        const auto admittedTransport = diffractionmaterial::admit(description, transportError);
        for (std::size_t pathIndex = 0; pathIndex < transportKinds.size(); ++pathIndex)
        {
            diffractionmaterial::SpectralLightingPath path;
            path.kind = transportKinds[pathIndex];
            path.bounceDepth = path.kind == diffractionmaterial::LightingPathKind::Indirect ? 1 : 0;
            path.incident.direction = transportDirections[pathIndex];
            path.incident.radiance.fill(0.01f);
            transportLighting.paths[pathIndex] = path;
            if (admittedTransport)
                transportProgram->diffractionPaths[pathIndex]
                    = diffractionmaterial::makeGpuLightingPath(
                        diffractionmaterial::physicalcheckpoint::makeGpuParameters(
                            *admittedTransport, path.incident), path);
        }
        const auto admittedTransportLighting
            = diffractionmaterial::AdmittedLightingPlan::admit(
                transportLighting, transportError);
        if (admittedTransportLighting)
        {
            transportProgram->diffractionLightingAdmission
                = std::make_shared<const diffractionmaterial::AdmittedLightingPlan>(
                    *admittedTransportLighting);
            transportProgram->programIdentity
                = arbitgpu::nativeFixtureDiffractionProgramIdentity(
                    transportProgram->bindingDigest, *admittedTransportLighting);
        }
        const auto transportPreparation = backend.prepare(scene, transportProgram);
        const auto transportFrame = backend.render(
            scene, transportPreparation.resources, 96, 96, {});
        const bool everyPathSubmitted = transportPreparation.prepared
            && transportFrame.rendered
            && transportFrame.stats.drawCount == transportKinds.size();

        arbitgpu::NativeFixtureSceneRuntimeInputs shiftedCamera;
        shiftedCamera.cameraTranslationOffset[0] = 0.75f;
        auto shifted = backend.render(
            scene, preparation.resources, 96, 96, shiftedCamera);
        const auto shiftedPixels = readBgra8(shifted.frame);

        auto rotatedDescription = description;
        rotatedDescription.geometry.directionUv = { 0.0f, 1.0f };
        auto rotatedProgram = makeProgram(
            rotatedDescription, scene->objects[0].id,
            "metal-fixture-diffraction-rotated-program");
        auto rotatedPreparation = backend.prepare(scene, rotatedProgram);
        auto rotated = backend.render(
            scene, rotatedPreparation.resources, 96, 96, {});
        const auto rotatedPixels = readBgra8(rotated.frame);

        auto spatialDescription = description;
        spatialDescription.grooveField.mode
            = diffractionmaterial::GrooveFieldMode::Linear;
        spatialDescription.grooveField.originUv = { 0.5f, 0.5f };
        spatialDescription.grooveField.axisUv = { 1.0f, 0.0f };
        spatialDescription.grooveField.grooveSpacingDeltaNanometresPerUnit = 500.0f;
        spatialDescription.grooveField.orientationDegreesPerUnit = 90.0f;
        auto spatialProgram = makeProgram(
            spatialDescription, scene->objects[0].id,
            "metal-fixture-diffraction-spatial-program");
        auto spatialPreparation = backend.prepare(scene, spatialProgram);
        auto spatial = backend.render(
            scene, spatialPreparation.resources, 96, 96, {});
        const auto spatialPixels = readBgra8(spatial.frame);

        auto crossedDescription
            = diffractionmaterial::makeRealtimeCrossedTwoDimensionalGratingPreset();
        auto crossedProgram = makeProgram(
            crossedDescription, scene->objects[0].id,
            "metal-fixture-diffraction-crossed-program");
        auto unequalCrossedBlazedProgram
            = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(
                *crossedProgram);
        unequalCrossedBlazedProgram->programIdentity
            = "metal-fixture-unequal-crossed-blazed-program";
        unequalCrossedBlazedProgram->diffractionPaths[0].material.control.y
            = static_cast<float>(diffractionmaterial::GrooveProfile::BlazedSawtooth);
        unequalCrossedBlazedProgram->diffractionPaths[0].material.secondaryGeometry.z
            = unequalCrossedBlazedProgram->diffractionPaths[0].material.geometry.z + 1.0f;
        const auto unequalCrossedBlazedPreparation = backend.prepare(
            scene, unequalCrossedBlazedProgram);
        auto crossedPreparation = backend.prepare(scene, crossedProgram);
        auto crossed = backend.render(
            scene, crossedPreparation.resources, 96, 96, {});
        const auto crossedPixels = readBgra8(crossed.frame);

        auto crossedDirectionDescription = crossedDescription;
        crossedDirectionDescription.geometry.directionUv
            = { 0.70710678f, 0.70710678f };
        crossedDirectionDescription.geometry.secondaryDirectionUv
            = { -0.70710678f, 0.70710678f };
        auto crossedDirectionProgram = makeProgram(
            crossedDirectionDescription,
            scene->objects[0].id,
            "metal-fixture-diffraction-crossed-direction-program");
        auto crossedDirectionPreparation = backend.prepare(
            scene, crossedDirectionProgram);
        auto crossedDirection = backend.render(
            scene, crossedDirectionPreparation.resources, 96, 96, {});
        const auto crossedDirectionPixels = readBgra8(crossedDirection.frame);

        auto primaryPeriodDescription = crossedDescription;
        primaryPeriodDescription.geometry.grooveSpacingNanometres = 1350.0f;
        auto primaryPeriodProgram = makeProgram(
            primaryPeriodDescription,
            scene->objects[0].id,
            "metal-fixture-diffraction-primary-period-program");
        auto primaryPeriodPreparation = backend.prepare(scene, primaryPeriodProgram);
        auto primaryPeriod = backend.render(
            scene, primaryPeriodPreparation.resources, 96, 96, {});
        const auto primaryPeriodPixels = readBgra8(primaryPeriod.frame);

        auto crossedPeriodDescription = crossedDescription;
        crossedPeriodDescription.geometry.secondaryGrooveSpacingNanometres = 900.0f;
        auto crossedPeriodProgram = makeProgram(
            crossedPeriodDescription,
            scene->objects[0].id,
            "metal-fixture-diffraction-crossed-period-program");
        auto crossedPeriodPreparation = backend.prepare(scene, crossedPeriodProgram);
        auto crossedPeriod = backend.render(
            scene, crossedPeriodPreparation.resources, 96, 96, {});
        const auto crossedPeriodPixels = readBgra8(crossedPeriod.frame);
        const auto overBudgetCrossed = backend.render(
            scene, crossedPreparation.resources, 3840, 2160, {});

        const auto reportDraw = [](const char* name, const auto& prepared,
                                   const auto& rendered)
        {
            if (!prepared.prepared || !rendered.rendered)
                std::cerr << "Metal diffraction draw " << name
                          << " preparation=" << prepared.prepared
                          << " preparationError=" << prepared.error
                          << " rendered=" << rendered.rendered
                          << " renderError=" << rendered.error << '\n';
        };
        reportDraw("base", preparation, frame);
        reportDraw("transport", transportPreparation, transportFrame);
        reportDraw("rotated", rotatedPreparation, rotated);
        reportDraw("spatial", spatialPreparation, spatial);
        reportDraw("crossed", crossedPreparation, crossed);
        reportDraw("crossed-direction", crossedDirectionPreparation, crossedDirection);
        reportDraw("primary-period", primaryPeriodPreparation, primaryPeriod);
        reportDraw("crossed-period", crossedPeriodPreparation, crossedPeriod);

        ok &= starterOk && rasterControlOk
            && expect(preparation.prepared && frame.rendered && repeated.rendered
                                   && everyPathSubmitted
                                   && shifted.rendered && rotatedPreparation.prepared
                                   && rotated.rendered && spatialPreparation.prepared
                                   && spatial.rendered && crossedPreparation.prepared
                                   && crossed.rendered
                                   && crossedDirectionPreparation.prepared
                                   && crossedDirection.rendered
                                   && primaryPeriodPreparation.prepared
                                   && primaryPeriod.rendered
                                   && crossedPeriodPreparation.prepared
                                   && crossedPeriod.rendered,
                               "the strict Metal fixture path must submit every diffraction draw")
            && expect(pixels.size() == 96u * 96u * 4u && !backgroundOnly(pixels),
                      "the Metal diffraction draw must write physical surface pixels")
            && expect(repeatedPixels == pixels
                          && frame.frame->nativeResourceCacheIdentity()
                              == repeated.frame->nativeResourceCacheIdentity()
                          && frame.frame->colorTextureDescriptor().rendererGeneration
                              == repeated.frame->colorTextureDescriptor().rendererGeneration,
                      "repeated Metal draws must retain exact cache and renderer-generation identity")
            && expect(pixels != shiftedPixels,
                      "Metal diffraction pixels must change with camera position")
            && expect(pixels != rotatedPixels,
                      "Metal diffraction pixels must change with reciprocal-lattice direction")
            && expect(spatialPixels != pixels && spatialPixels != rotatedPixels,
                      "Metal diffraction pixels must evaluate local UV groove fields")
            && expect(!backgroundOnly(crossedPixels) && crossedPixels != pixels,
                      "Metal diffraction pixels must execute crossed-lattice topology")
            && expect(crossedDirectionPixels != crossedPixels
                          && primaryPeriodPixels != crossedPixels
                          && crossedPeriodPixels != crossedPixels,
                      "Metal crossed diffraction must independently respond to lattice orientation and both periods")
            && expect(!overBudgetCrossed.rendered
                          && overBudgetCrossed.frame == nullptr
                          && overBudgetCrossed.error
                               == "Metal fixture diffraction workload exceeds backend limits",
                      "Metal rejects over-budget crossed diffraction before frame allocation");
        const bool malformedRejected = expect(
            !malformedPreparation.prepared && malformedPreparation.resources == nullptr
                && malformedPreparation.stats.vertexBytes == 0
                && malformedPreparation.stats.indexBytes == 0
                && malformedPreparation.stats.materialBytes == 0
                && malformedPreparation.stats.textureBytes == 0
                && malformedPreparation.error
                    == "Metal fixture preparation requires an exact bounded material binding",
            "Metal rejects malformed diffraction GPU parameters before resource allocation")
            && expect(!replayedLightingPreparation.prepared
                && replayedLightingPreparation.resources == nullptr
                && replayedLightingPreparation.stats.vertexBytes == 0
                && replayedLightingPreparation.stats.indexBytes == 0
                && replayedLightingPreparation.stats.materialBytes == 0
                && replayedLightingPreparation.stats.textureBytes == 0
                && replayedLightingPreparation.error
                    == "Metal fixture preparation requires an exact bounded material binding",
            "Metal rejects a replayed spectral receipt before resource allocation")
            && expect(!layoutSentinelPreparation.prepared
                && layoutSentinelPreparation.resources == nullptr
                && layoutSentinelPreparation.stats.vertexBytes == 0
                && layoutSentinelPreparation.stats.indexBytes == 0
                && layoutSentinelPreparation.stats.materialBytes == 0
                && layoutSentinelPreparation.stats.textureBytes == 0
                && layoutSentinelPreparation.error
                    == "Metal fixture preparation requires an exact bounded material binding",
            "Metal rejects reused layout-sentinel data before GPU allocation")
            && expect(!divergentBlazePreparation.prepared
                && divergentBlazePreparation.resources == nullptr
                && divergentBlazePreparation.stats.vertexBytes == 0
                && divergentBlazePreparation.stats.indexBytes == 0
                && divergentBlazePreparation.stats.materialBytes == 0
                && divergentBlazePreparation.stats.textureBytes == 0
                && divergentBlazePreparation.error
                    == "Metal fixture preparation requires an exact bounded material binding",
            "Metal rejects local period/blaze divergence before GPU allocation")
            && expect(!unequalCrossedBlazedPreparation.prepared
                && unequalCrossedBlazedPreparation.resources == nullptr
                && unequalCrossedBlazedPreparation.error
                    == "Metal fixture preparation requires an exact bounded material binding",
            "Metal preflight rejects unequal crossed blazed periods before GPU allocation")
            && expect(frame.rendered,
                      "Metal preparation owns an immutable material snapshot after caller alias mutation");
        if (!malformedRejected)
            return 1;
        if (!ok)
            return 1;
        std::cout << "Metal fixture diffraction backend: PASS; device="
                  << info.device << "; raster-control-pixels="
                  << nonBackgroundPixelCount(rasterPixels)
                  << "; front-facing-control-pixels="
                  << nonBackgroundPixelCount(frontFacingRasterPixels)
                  << "; culled-control-pixels="
                  << nonBackgroundPixelCount(backFacingRasterPixels)
                  << "; diffraction-pixels=" << nonBackgroundPixelCount(pixels)
                  << '\n';
        return 0;
    }
#endif
}
