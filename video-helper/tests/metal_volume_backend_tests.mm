#include "../src/volume_native_renderer.h"
#include "../src/gpu_backend/sokol_metal_context.h"

#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#define SOKOL_METAL
#include "sokol_gfx.h"

namespace
{
std::shared_ptr<const videohelper::volume::AdmittedVolume> admittedSphere (
    const videohelper::volume::VolumeRendererCapabilities& capabilities,
    std::uint8_t density, float translateX, std::string& identity)
{
    constexpr std::uint32_t edge = 32;
    videowire::DenseVolumeDescriptor descriptor;
    descriptor.dimensions = { edge, edge, edge };
    descriptor.bounds = { { -1.0f, -1.0f, -1.0f }, { 1.0f, 1.0f, 1.0f } };
    descriptor.transform.localToWorld[12] = translateX;
    descriptor.voxels.resize (static_cast<std::size_t> (edge) * edge * edge);
    for (std::uint32_t z = 0; z < edge; ++z)
        for (std::uint32_t y = 0; y < edge; ++y)
            for (std::uint32_t x = 0; x < edge; ++x)
            {
                const auto fx = (static_cast<float> (x) + 0.5f) / edge * 2.0f - 1.0f;
                const auto fy = (static_cast<float> (y) + 0.5f) / edge * 2.0f - 1.0f;
                const auto fz = (static_cast<float> (z) + 0.5f) / edge * 2.0f - 1.0f;
                if (fx * fx + fy * fy + fz * fz <= 0.42f * 0.42f)
                    descriptor.voxels[(static_cast<std::size_t> (z) * edge + y) * edge + x]
                        = density;
            }

    std::string error;
    auto admitted = videohelper::volume::admitDenseVolume (
        descriptor, {}, capabilities, error);
    if (! admitted)
    {
        std::fprintf (stderr, "dense volume admission failed: %s\n", error.c_str());
        return {};
    }
    identity = admitted->cacheIdentity();
    return std::make_shared<const videohelper::volume::AdmittedVolume> (
        std::move (*admitted));
}

std::vector<std::uint8_t> readPixels (
    const videohelper::volume::NativeVolumeFrame& frame)
{
    std::lock_guard<std::mutex> lock (arbitgpu::sokolmetal::mutex());
    const sg_image image { static_cast<std::uint32_t> (frame.colorImageHandle()) };
    const auto info = sg_mtl_query_image_info (image);
    id<MTLTexture> texture = (__bridge id<MTLTexture>) info.tex[info.active_slot];
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>) sg_mtl_command_queue();
    id<MTLDevice> device = (__bridge id<MTLDevice>) arbitgpu::sokolmetal::device();
    const auto byteCount = static_cast<NSUInteger> (frame.width()) * frame.height() * 4u;
    id<MTLBuffer> readback = [device newBufferWithLength:byteCount
                                                options:MTLResourceStorageModeShared];
    if (texture == nil || queue == nil || readback == nil) return {};

    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    [blit copyFromTexture:texture
              sourceSlice:0
              sourceLevel:0
             sourceOrigin:MTLOriginMake (0, 0, 0)
               sourceSize:MTLSizeMake (frame.width(), frame.height(), 1)
                 toBuffer:readback
        destinationOffset:0
   destinationBytesPerRow:static_cast<NSUInteger> (frame.width()) * 4u
 destinationBytesPerImage:byteCount];
    [blit endEncoding];
    [command commit];
    [command waitUntilCompleted];
    std::vector<std::uint8_t> pixels (byteCount);
    if (command.status == MTLCommandBufferStatusCompleted)
        std::memcpy (pixels.data(), readback.contents, pixels.size());
    else
        pixels.clear();
#if ! __has_feature(objc_arc)
    [readback release];
#endif
    return pixels;
}

std::array<std::uint8_t, 4> pixelAt (
    const std::vector<std::uint8_t>& pixels, std::uint32_t width,
    std::uint32_t x, std::uint32_t y)
{
    const auto offset = (static_cast<std::size_t> (y) * width + x) * 4u;
    return { pixels[offset], pixels[offset + 1], pixels[offset + 2], pixels[offset + 3] };
}

std::uint32_t checksum (const std::vector<std::uint8_t>& pixels)
{
    std::uint32_t value = 2166136261u;
    for (const auto byte : pixels)
    {
        value ^= byte;
        value *= 16777619u;
    }
    return value;
}

bool imageIsValid (std::uintptr_t handle)
{
    std::lock_guard<std::mutex> lock (arbitgpu::sokolmetal::mutex());
    return sg_query_image_state ({ static_cast<std::uint32_t> (handle) })
        == SG_RESOURCESTATE_VALID;
}
} // namespace

int main()
{
    static_assert (! videowire::VolumeRenderContract::kAllowsCpuProductionFallback,
                   "Metal volume rendering must not gain a CPU production fallback");

    auto& backend = videohelper::volume::nativeVolumeExecutionBackend();
    const auto capabilities = backend.capabilities();
    if (! capabilities.volume.nativeGpuAvailable
        || ! capabilities.volume.volumeRaymarch
        || ! capabilities.volume.denseVolumeUpload
        || capabilities.volume.sparseBrickUpload
        || capabilities.volume.maxTexture3DDimension > 2048
        || capabilities.volume.maxVolumeBytes != 512u * 1024u * 1024u
        || capabilities.maxRenderExtent > 4096
        || capabilities.maxRenderPixels > 4096ull * 4096ull
        || capabilities.backend != "metal")
    {
        std::fprintf (stderr, "strict Metal volume capabilities are unavailable\n");
        return 77;
    }

    std::string identity;
    const auto volume = admittedSphere (capabilities.volume, 255, 0.0f, identity);
    if (volume == nullptr) return 1;

    videohelper::volume::NativeVolumeRenderer renderer;
    videohelper::volume::NativeVolumeRenderedFrame preview;
    videohelper::volume::NativeVolumeRenderedFrame exportFrame;
    std::string error;
    if (! renderer.renderPreview (volume, 96, 64,
                                  videohelper::volume::kNativeVolumeGpuCapability,
                                  preview, error)
        || ! renderer.renderExport (volume, 96, 64,
                                    videohelper::volume::kNativeVolumeGpuCapability,
                                    exportFrame, error))
    {
        std::fprintf (stderr, "native Metal volume preview/export failed: %s\n",
                      error.c_str());
        return 2;
    }
    if (preview.use != videohelper::volume::NativeVolumeRenderUse::Preview
        || exportFrame.use != videohelper::volume::NativeVolumeRenderUse::Export
        || preview.volumeIdentity != identity || exportFrame.volumeIdentity != identity
        || preview.nativeFrame == nullptr || exportFrame.nativeFrame == nullptr)
    {
        std::fprintf (stderr, "preview/export did not publish admitted Metal frames\n");
        return 3;
    }

    const auto previewPixels = readPixels (*preview.nativeFrame);
    const auto exportPixels = readPixels (*exportFrame.nativeFrame);
    if (previewPixels.size() != 96u * 64u * 4u || previewPixels != exportPixels)
    {
        std::fprintf (stderr, "Metal preview/export readback mismatch\n");
        return 4;
    }
    const auto center = pixelAt (previewPixels, 96, 48, 32);
    const auto background = pixelAt (previewPixels, 96, 0, 0);
    if (background != std::array<std::uint8_t, 4> { 18, 10, 7, 255 }
        || center == background || center[3] != 255)
    {
        std::fprintf (stderr,
            "Metal volume raymarch pixel mismatch center=%u,%u,%u,%u background=%u,%u,%u,%u\n",
            center[0], center[1], center[2], center[3],
            background[0], background[1], background[2], background[3]);
        return 5;
    }

    std::string emptyIdentity;
    const auto empty = admittedSphere (capabilities.volume, 0, 0.0f, emptyIdentity);
    videohelper::volume::NativeVolumeRenderedFrame emptyFrame;
    if (empty == nullptr
        || ! renderer.renderPreview (empty, 96, 64,
                                     videohelper::volume::kNativeVolumeGpuCapability,
                                     emptyFrame, error)
        || readPixels (*emptyFrame.nativeFrame) == previewPixels)
    {
        std::fprintf (stderr, "Metal 3D density upload did not affect raymarched pixels\n");
        return 6;
    }

    std::string translatedIdentity;
    const auto translated = admittedSphere (capabilities.volume, 255, 12.0f,
                                             translatedIdentity);
    videohelper::volume::NativeVolumeRenderedFrame translatedFrame;
    if (translated == nullptr
        || ! renderer.renderExport (translated, 96, 64,
                                    videohelper::volume::kNativeVolumeGpuCapability,
                                    translatedFrame, error)
        || readPixels (*translatedFrame.nativeFrame) != previewPixels)
    {
        std::fprintf (stderr, "Metal volume transform was not consumed by canonical framing\n");
        return 7;
    }

    videowire::SparseVolumeDescriptor sparse;
    sparse.dimensions = { 2, 2, 2 };
    sparse.brickEdge = 2;
    sparse.bricks = { { 0, 0, 0, std::vector<std::uint8_t> (8, 255) } };
    if (videohelper::volume::admitSparseVolume (
            sparse, {}, capabilities.volume, error).has_value()
        || error != "renderer lacks sparse brick upload capability")
    {
        std::fprintf (stderr, "unsupported Metal sparse uploads did not fail closed\n");
        return 8;
    }

    auto oversized = backend.render ({ volume, capabilities.maxRenderExtent + 1u, 1u });
    if (oversized.rendered
        || oversized.error != "Metal native volume request exceeds backend limits")
    {
        std::fprintf (stderr, "oversized Metal render target did not fail closed\n");
        return 9;
    }

    const auto previewImage = preview.nativeFrame->colorImageHandle();
    const auto exportImage = exportFrame.nativeFrame->colorImageHandle();
    if (! imageIsValid (previewImage) || ! imageIsValid (exportImage))
    {
        std::fprintf (stderr, "published Metal volume image handles are not live\n");
        return 10;
    }
    emptyFrame = {};
    translatedFrame = {};
    preview = {};
    exportFrame = {};
    if (imageIsValid (previewImage) || imageIsValid (exportImage))
    {
        std::fprintf (stderr, "Metal volume frames leaked GPU image resources\n");
        return 11;
    }

    std::printf (
        "metal-volume PASS backend=%s volume=%s center=%u,%u,%u,%u "
        "background=%u,%u,%u,%u hash=%u preview_export_equal=yes images_deleted=yes\n",
        capabilities.backend.c_str(), identity.c_str(),
        center[0], center[1], center[2], center[3],
        background[0], background[1], background[2], background[3],
        checksum (previewPixels));
    return 0;
}
