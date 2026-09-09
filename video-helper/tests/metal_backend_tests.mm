#include "gpu_backend/backend.h"
#include "sdf_native_renderer.h"
#include "support/fixture_scene.h"
#include "support/sdf_native_backend_parity.h"

#import <Metal/Metal.h>

#define SOKOL_METAL
#include "sokol_gfx.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <vector>

namespace
{
std::vector<std::uint8_t> readBgra8 (
    const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& frame)
{
    const sg_image image { static_cast<std::uint32_t> (frame->colorImageHandle()) };
    const auto metal = sg_mtl_query_image_info (image);
    if (metal.active_slot < 0 || metal.active_slot >= SG_NUM_INFLIGHT_FRAMES)
        return {};
    id<MTLTexture> texture = (__bridge id<MTLTexture>) metal.tex[metal.active_slot];
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>) sg_mtl_command_queue();
    if (texture == nil || queue == nil)
        return {};
    const auto byteCount = static_cast<NSUInteger> (frame->width()) * frame->height() * 4u;
    id<MTLBuffer> readback = [texture.device newBufferWithLength:byteCount
                                                        options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    [blit copyFromTexture:texture
              sourceSlice:0
              sourceLevel:0
             sourceOrigin:MTLOriginMake (0, 0, 0)
               sourceSize:MTLSizeMake (frame->width(), frame->height(), 1)
                 toBuffer:readback
        destinationOffset:0
   destinationBytesPerRow:frame->width() * 4u
 destinationBytesPerImage:byteCount];
    [blit endEncoding];
    [command commit];
    [command waitUntilCompleted];
    std::vector<std::uint8_t> pixels (byteCount);
    if (command.status == MTLCommandBufferStatusCompleted && readback.contents != nullptr)
        std::copy_n (static_cast<const std::uint8_t*> (readback.contents), byteCount,
                     pixels.begin());
    else
        pixels.clear();
#if ! __has_feature(objc_arc)
    [readback release];
#endif
    return pixels;
}

videowire::SdfRecord record (videowire::SdfStableId id,
                             videowire::SdfOperation operation,
                             std::initializer_list<videowire::SdfStableId> inputs,
                             std::initializer_list<double> parameters)
{
    videowire::SdfRecord result;
    result.stableId = id;
    result.operation = operation;
    result.inputCount = static_cast<std::uint8_t> (inputs.size());
    result.parameterCount = static_cast<std::uint8_t> (parameters.size());
    std::copy (inputs.begin(), inputs.end(), result.inputs.begin());
    std::copy (parameters.begin(), parameters.end(), result.parameters.begin());
    return result;
}

std::uint32_t checksum (const std::vector<std::uint8_t>& pixels)
{
    std::uint32_t hash = 2166136261u;
    for (const auto value : pixels)
    {
        hash ^= value;
        hash *= 16777619u;
    }
    return hash;
}
} // namespace

int main()
{
    const auto info = arbitgpu::queryNativeBackend();
    if (! info.available || info.backend != "metal" || ! info.compute)
    {
        std::fprintf (stderr, "Metal capability probe failed: %s\n", info.error.c_str());
        return 1;
    }

    const auto test = arbitgpu::runNativeBackendSelfTest();
    if (! test.available || ! test.computePassed || ! test.renderPassed)
    {
        std::fprintf (stderr, "Metal self-test failed: %s\n", test.error.c_str());
        return 2;
    }

    renderpassoutput::Description outputDescription;
    outputDescription.extent = { 64, 32 };
    for (const auto output : { renderpassoutput::Output::Color,
                               renderpassoutput::Output::Depth })
    {
        const auto required = renderpassoutput::requirements (output);
        outputDescription.attachments.push_back (
            { output, required.format, required.colorSpace, outputDescription.extent });
    }
    renderpassoutput::AdmissionFailure outputFailure = renderpassoutput::AdmissionFailure::None;
    auto admittedOutputs = renderpassoutput::admit (outputDescription, outputFailure);
    auto& outputBackend = arbitgpu::nativeRenderPassOutputBackend();
    const auto outputCapabilities = outputBackend.renderPassOutputCapabilities();
    auto outputAdmission = admittedOutputs.has_value()
        ? outputBackend.admitRenderPassOutputs (*admittedOutputs)
        : arbitgpu::RenderPassOutputAdmission {};
    if (! outputCapabilities.available || ! outputAdmission.lifecycle
        || outputAdmission.resources.size() != 2
        || outputAdmission.frameMemory.failure
               != arbitgpu::FrameMemoryAdmissionFailure::none
        || outputAdmission.frameMemory.requestedBytes != 64ull * 32ull * 12ull
        || outputAdmission.frameMemory.allocatedSlots != 2)
    {
        std::fprintf (stderr, "Metal frame-memory admission failed: %s\n",
                      outputAdmission.error.c_str());
        return 20;
    }
    outputBackend.releaseRenderPassOutputs (outputAdmission.lifecycle);

    const auto sdf = arbitgpu::queryNativeSdfExecution();
    if (! sdf.available || sdf.backend != "metal"
        || ! sdf.supports (videowire::SdfOperation::sphere)
        || ! sdf.supports (videowire::SdfOperation::box)
        || ! sdf.supports (videowire::SdfOperation::roundedBox)
        || ! sdf.supports (videowire::SdfOperation::plane)
        || ! sdf.supports (videowire::SdfOperation::torus)
        || ! sdf.supports (videowire::SdfOperation::capsule)
        || ! sdf.supports (videowire::SdfOperation::cylinder)
        || ! sdf.supports (videowire::SdfOperation::cone)
        || ! sdf.supports (videowire::SdfOperation::gyroid)
        || ! sdf.supports (arbitgpu::NativeSdfOutput::color)
        || ! sdf.supports (arbitgpu::NativeSdfOutput::depth)
        || ! sdf.supports (arbitgpu::NativeSdfOutput::normal))
    {
        std::fprintf (stderr, "Metal SDF capability probe failed: %s\n",
                      sdf.error.c_str());
        return 3;
    }

    arbitgpu::NativeSdfDrawRequest request;
    request.geometry.rootId = 1;
    videowire::SdfRecord sphere;
    sphere.stableId = request.geometry.rootId;
    sphere.operation = videowire::SdfOperation::sphere;
    sphere.parameterCount = 1;
    sphere.parameters[0] = 1.0;
    request.geometry.records.push_back (sphere);
    request.width = 64;
    request.height = 64;
    request.maximumSteps = 128;
    request.epsilon = 0.001;
    request.maximumDistance = 100.0;
    std::string rendererError;
    auto admitted = videohelper::sdf::admitSdfIr (request.geometry, {}, rendererError);
    videohelper::sdf::NativeSdfRenderedFrame renderedByOwner;
    videohelper::sdf::NativeSdfRenderedFrame exportedByOwner;
    const auto geometry = admitted
        ? std::make_shared<const videohelper::sdf::AdmittedSdfIr> (std::move (*admitted))
        : nullptr;
    const auto ownerRendered = geometry && videohelper::sdf::nativeSdfRenderer().renderPreview (
        geometry, { request.width, request.height }, {},
        videohelper::sdf::kNativeGpuCapability, renderedByOwner, rendererError);
    const auto ownerExported = geometry && videohelper::sdf::nativeSdfRenderer().renderExport (
        geometry, { request.width, request.height }, {},
        videohelper::sdf::kNativeGpuCapability, exportedByOwner, rendererError);
    const auto rendered = ownerRendered
        ? arbitgpu::NativeSdfSceneSubmission { true, renderedByOwner.nativeFrame, {} }
        : arbitgpu::NativeSdfSceneSubmission { false, {}, rendererError };
    if (! rendered.rendered || rendered.frame == nullptr
        || rendered.frame->backend() != "metal"
        || rendered.frame->width() != request.width
        || rendered.frame->height() != request.height
        || rendered.frame->colorImageHandle() == 0
        || rendered.frame->colorTextureViewHandle() == 0
        || rendered.frame->frameMemoryAdmission().failure
               != arbitgpu::FrameMemoryAdmissionFailure::none
        || rendered.frame->frameMemoryAdmission().requestedBytes != 64ull * 64ull * 8ull
        || rendered.frame->frameMemoryAdmission().allocatedSlots != 1
        || renderedByOwner.use != videohelper::sdf::NativeSdfRenderUse::Preview
        || ! ownerExported || exportedByOwner.use != videohelper::sdf::NativeSdfRenderUse::Export
        || exportedByOwner.nativeFrame == nullptr
        || exportedByOwner.nativeFrame->backend() != "metal"
        || exportedByOwner.nativeFrame->colorImageHandle() == 0
        || exportedByOwner.nativeFrame->colorImageHandle()
               == rendered.frame->colorImageHandle())
    {
        std::fprintf (stderr, "Metal SDF draw failed: %s\n", rendered.error.c_str());
        return 4;
    }
    const auto& receipt = rendered.frame->sdfResourceReceipt();
    if (receipt.compiledRecordCount != 1 || receipt.compiledRecordBytes == 0
        || receipt.geometryCacheBytes < receipt.compiledRecordBytes
        || receipt.backendProgramBytes == 0 || receipt.uniformBytes == 0
        || receipt.attachmentBytes != 64ull * 64ull * 8ull || receipt.totalBytes == 0)
    {
        std::fprintf (stderr, "Metal native SDF resource receipt mismatch\n");
        return 4;
    }

    auto boxRequest = request;
    boxRequest.geometry = {};
    boxRequest.geometry.rootId = 2;
    videowire::SdfRecord box;
    box.stableId = boxRequest.geometry.rootId;
    box.operation = videowire::SdfOperation::box;
    box.parameterCount = 3;
    box.parameters[0] = 0.75;
    box.parameters[1] = 0.5;
    box.parameters[2] = 0.25;
    boxRequest.geometry.records.push_back (box);
    const auto renderedBox = arbitgpu::nativeSdfExecutionBackend().render (boxRequest);
    if (! renderedBox.rendered || renderedBox.frame == nullptr
        || renderedBox.frame->backend() != "metal"
        || renderedBox.frame->colorImageHandle() == 0
        || renderedBox.frame->colorTextureViewHandle() == 0)
    {
        std::fprintf (stderr, "Metal SDF box draw failed: %s\n", renderedBox.error.c_str());
        return 5;
    }

    auto compositeRequest = request;
    compositeRequest.geometry = {};
    compositeRequest.geometry.rootId = 5;
    compositeRequest.geometry.records = {
        record (1, videowire::SdfOperation::sphere, {}, { 0.55 }),
        record (2, videowire::SdfOperation::translate, { 1 }, { -0.7, 0.0, 0.0 }),
        record (3, videowire::SdfOperation::box, {}, { 0.38, 0.5, 0.45 }),
        record (4, videowire::SdfOperation::translate, { 3 }, { 0.7, 0.0, 0.0 }),
        record (5, videowire::SdfOperation::smoothUnion, { 2, 4 }, { 0.18 })
    };
    const auto renderedComposite = arbitgpu::nativeSdfExecutionBackend().render (compositeRequest);
    std::vector<std::uint8_t> spherePixels;
    std::vector<std::uint8_t> compositePixels;
    if (! renderedComposite.rendered || renderedComposite.frame == nullptr
        || ! rendered.frame->readColorPixels (spherePixels)
        || ! renderedComposite.frame->readColorPixels (compositePixels)
        || spherePixels.size() != 64u * 64u * 4u
        || compositePixels.size() != spherePixels.size()
        || checksum (spherePixels) == checksum (compositePixels))
    {
        std::fprintf (stderr, "Metal transformed composite did not affect pixels: %s\n",
                      renderedComposite.error.c_str());
        return 6;
    }

    auto malformed = compositeRequest;
    malformed.geometry.rootId = 2;
    malformed.geometry.records = {
        record (1, videowire::SdfOperation::translate, { 2 }, { 0.0, 0.0, 0.0 }),
        record (2, videowire::SdfOperation::translate, { 1 }, { 0.0, 0.0, 0.0 })
    };
    const auto rejectedCycle = arbitgpu::nativeSdfExecutionBackend().render (malformed);
    malformed.geometry.records[0].inputs[0] = 999;
    const auto rejectedReference = arbitgpu::nativeSdfExecutionBackend().render (malformed);
    if (rejectedCycle.rendered || rejectedCycle.frame != nullptr
        || rejectedReference.rendered || rejectedReference.frame != nullptr)
    {
        std::fprintf (stderr, "Metal malformed SDF request allocated a frame\n");
        return 7;
    }

    videohelper::sdf::NativeSdfRenderControls parityControls;
    parityControls.maximumSteps = 128;
    parityControls.epsilon = 0.001;
    parityControls.maximumDistance = 100.0;
    parityControls.output = arbitgpu::NativeSdfOutput::depth;
    std::string parityError;
    auto parityFixtures = videohelper::sdf::test::nativeOperationFixtures (parityError);
    auto polarFixtures = videohelper::sdf::test::largeOffsetPolarRepeatFixtures (parityError);
    parityFixtures.insert (parityFixtures.end(),
                           std::make_move_iterator (polarFixtures.begin()),
                           std::make_move_iterator (polarFixtures.end()));
    if (parityFixtures.size() != 28)
    {
        std::fprintf (stderr, "Metal SDF parity fixtures failed: %s\n", parityError.c_str());
        return 8;
    }
    for (const auto& fixture : parityFixtures)
    {
        arbitgpu::NativeSdfDrawRequest parityRequest;
        parityRequest.geometry.schemaVersion = fixture.geometry->schemaVersion();
        parityRequest.geometry.rootId = fixture.geometry->rootId();
        parityRequest.geometry.records = fixture.geometry->records();
        parityRequest.width = 17;
        parityRequest.height = 17;
        parityRequest.maximumSteps = parityControls.maximumSteps;
        parityRequest.epsilon = parityControls.epsilon;
        parityRequest.maximumDistance = parityControls.maximumDistance;
        parityRequest.output = parityControls.output;
        const auto parityFrame = arbitgpu::nativeSdfExecutionBackend().render (parityRequest);
        std::vector<std::uint8_t> parityPixels;
        if (! parityFrame.rendered || parityFrame.frame == nullptr
            || ! parityFrame.frame->readColorPixels (parityPixels)
            || ! videohelper::sdf::test::verifyNativeDepthParity (
                   fixture, parityControls, parityPixels, 17, 17, parityError))
        {
            std::fprintf (stderr, "Metal SDF operation parity failed: %s\n",
                          parityError.empty() ? parityFrame.error.c_str() : parityError.c_str());
            return 9;
        }
    }

    auto fixtureScene = std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (
        videohelper::fixture3d::makeScene());
    auto& fixtureBackend = arbitgpu::nativeFixtureSceneBackend();
    auto fixturePreparation = fixtureBackend.prepare (fixtureScene, nullptr);
    arbitgpu::NativeFixtureSceneRuntimeInputs fixtureInputs;
    HarmonicMIDI::grid::SceneLightRecord lightOverride;
    lightOverride.id.value = 9001;
    lightOverride.color = { 0.25f, 0.5f, 0.75f };
    lightOverride.intensity = 8.0f;
    fixtureInputs.lightOverride = lightOverride;
    auto fixtureFrame = fixtureBackend.render (
        fixtureScene, fixturePreparation.resources, 64, 64, fixtureInputs);
    if (! fixturePreparation.prepared || ! fixtureFrame.rendered
        || fixtureFrame.frame == nullptr || fixtureFrame.frame->backend() != "metal"
        || fixtureFrame.frame->colorImageHandle() == 0
        || fixtureFrame.frame->colorTextureViewHandle() == 0)
    {
        std::fprintf (stderr, "Metal fixture Light binding failed: %s %s\n",
                      fixturePreparation.error.c_str(), fixtureFrame.error.c_str());
        return 6;
    }

    const auto fixturePixels = readBgra8 (fixtureFrame.frame);
    auto composedSceneValue = *fixtureScene;
    composedSceneValue.objectCount = 2;
    composedSceneValue.materialCount = 2;
    composedSceneValue.objects[1] = composedSceneValue.objects[0];
    composedSceneValue.objects[1].id.value = 9002;
    composedSceneValue.objects[1].parent = composedSceneValue.objects[0].id;
    // The local offset is the parent's inverse rotation of world +X, so the
    // child lands in a separate right-side draw region while exercising hierarchy.
    composedSceneValue.objects[1].transform.translation
        = { 2.5980762f, -0.5130302f, 1.4095389f };
    composedSceneValue.materials[1] = composedSceneValue.materials[0];
    composedSceneValue.materials[1].id.value = 9003;
    composedSceneValue.materials[1].emissive = { 0.01f, 0.8f, 0.02f };
    composedSceneValue.objects[1].material = composedSceneValue.materials[1].id;
    auto composedScene = std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (
        std::move (composedSceneValue));
    auto objectZeroSceneValue = *composedScene;
    objectZeroSceneValue.objectCount = 1;
    auto objectZeroScene = std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (
        std::move (objectZeroSceneValue));
    const auto objectZeroPreparation = fixtureBackend.prepare (objectZeroScene, nullptr);
    const auto objectZeroFrame = fixtureBackend.render (
        objectZeroScene, objectZeroPreparation.resources, 64, 64, fixtureInputs);
    const auto objectZeroPixels = objectZeroFrame.rendered
        ? readBgra8 (objectZeroFrame.frame) : std::vector<std::uint8_t> {};
    const auto composedPreparation = fixtureBackend.prepare (composedScene, nullptr);
    const auto composedFrame = fixtureBackend.render (
        composedScene, composedPreparation.resources, 64, 64, fixtureInputs);
    const auto composedPixels = composedFrame.rendered
        ? readBgra8 (composedFrame.frame) : std::vector<std::uint8_t> {};
    const auto composedExport = fixtureBackend.render (
        composedScene, composedPreparation.resources, 64, 64, fixtureInputs);
    const auto composedExportPixels = composedExport.rendered
        ? readBgra8 (composedExport.frame) : std::vector<std::uint8_t> {};
    const std::array<std::uint8_t, 4> background { 18, 10, 7, 255 };
    std::size_t secondObjectPixels = 0;
    bool objectZeroUnchanged = composedPixels.size() == objectZeroPixels.size();
    for (std::uint32_t y = 0; objectZeroUnchanged && y < 64u; ++y)
        for (std::uint32_t x = 0; x < 64u; ++x)
        {
            const auto offset = (y * 64u + x) * 4u;
            if (! std::equal (composedPixels.begin() + offset,
                              composedPixels.begin() + offset + 4u,
                              objectZeroPixels.begin() + offset))
            {
                objectZeroUnchanged &= std::equal (
                    objectZeroPixels.begin() + offset, objectZeroPixels.begin() + offset + 4u,
                    background.begin());
                secondObjectPixels += composedPixels[offset + 1u]
                    > composedPixels[offset + 2u] * 4u ? 1u : 0u;
            }
        }
    if (! objectZeroPreparation.prepared || ! objectZeroFrame.rendered
        || objectZeroFrame.stats.drawCount != 1
        || ! composedPreparation.prepared || ! composedFrame.rendered
        || ! composedExport.rendered
        || composedPreparation.stats.staticUploadCount != 2
        || composedFrame.stats.drawCount != 2
        || ! composedFrame.stats.reusedStaticResources
        || ! composedExport.stats.reusedStaticResources
        || composedPixels.size() != 64u * 64u * 4u
        || ! objectZeroUnchanged || secondObjectPixels == 0
        || composedPixels != composedExportPixels)
    {
        std::fprintf (stderr, "Metal composed Scene3D execution failed: %s %s\n",
                      composedPreparation.error.c_str(), composedFrame.error.c_str());
        return 7;
    }

    auto transparentSceneValue = *fixtureScene;
    transparentSceneValue.materials[0].opacity = 0.5f;
    const auto transparentPreparation = fixtureBackend.prepare (
        std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (transparentSceneValue),
        nullptr);
    if (transparentPreparation.prepared || transparentPreparation.resources != nullptr
        || transparentPreparation.stats.staticUploadCount != 0)
    {
        std::fprintf (stderr, "Metal accepted unsupported opacity before GPU upload\n");
        return 8;
    }

    std::printf ("backend=%s device=%s compute=%u render=%u sdfImage=%zu sdfBoxImage=%zu fixtureImage=%zu\n",
                 test.backend.c_str(), test.device.c_str(),
                 test.computeChecksum, test.renderChecksum,
                 rendered.frame->colorImageHandle(), renderedBox.frame->colorImageHandle(),
                 fixtureFrame.frame->colorImageHandle());
    return 0;
}
