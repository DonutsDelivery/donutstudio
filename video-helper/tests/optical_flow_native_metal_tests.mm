#import <Metal/Metal.h>

#include "../src/optical_flow_native_executor.h"
#include "optical_flow_native_test_fixture.h"
#include "sokol_gfx.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace
{
constexpr std::uint32_t kWidth = opticalflowtest::width;
constexpr std::uint32_t kHeight = opticalflowtest::height;
constexpr std::uint64_t kStructuralRevision = 23;
constexpr std::uint64_t kHelperGeneration = 7;

std::uint16_t floatToHalf (float value)
{
    std::uint32_t bits = 0;
    std::memcpy (&bits, &value, sizeof (bits));
    const auto sign = static_cast<std::uint16_t> ((bits >> 16u) & 0x8000u);
    auto exponent = static_cast<int> ((bits >> 23u) & 0xffu) - 127 + 15;
    auto mantissa = bits & 0x7fffffu;
    if (exponent <= 0)
    {
        if (exponent < -10) return sign;
        mantissa = (mantissa | 0x800000u) >> static_cast<unsigned> (1 - exponent);
        return static_cast<std::uint16_t> (sign | ((mantissa + 0x1000u) >> 13u));
    }
    if (exponent >= 31)
        return static_cast<std::uint16_t> (sign | 0x7c00u);
    return static_cast<std::uint16_t> (
        sign | (static_cast<std::uint16_t> (exponent) << 10u)
        | static_cast<std::uint16_t> ((mantissa + 0x1000u) >> 13u));
}

float halfToFloat (std::uint16_t value)
{
    const std::uint32_t sign = static_cast<std::uint32_t> (value & 0x8000u) << 16u;
    std::uint32_t exponent = (value >> 10u) & 0x1fu;
    std::uint32_t mantissa = value & 0x3ffu;
    std::uint32_t bits = 0;
    if (exponent == 0)
    {
        if (mantissa == 0)
            bits = sign;
        else
        {
            exponent = 1;
            while ((mantissa & 0x400u) == 0)
            {
                mantissa <<= 1u;
                --exponent;
            }
            mantissa &= 0x3ffu;
            bits = sign | ((exponent + 127u - 15u) << 23u) | (mantissa << 13u);
        }
    }
    else if (exponent == 31)
        bits = sign | 0x7f800000u | (mantissa << 13u);
    else
        bits = sign | ((exponent + 127u - 15u) << 23u) | (mantissa << 13u);
    float result = 0.0f;
    std::memcpy (&result, &bits, sizeof (result));
    return result;
}

videoopticalflow::Description fixtureDescription()
{
    videoopticalflow::Description description;
    description.first.identity.stream[0] = 0x61;
    description.second.identity.stream = description.first.identity.stream;
    description.first.identity.content[0] = 0x71;
    description.second.identity.content[0] = 0x72;
    description.first.identity.frameIndex = 20;
    description.second.identity.frameIndex = 21;
    description.first.timestamp = { 20020, 24000 };
    description.second.timestamp = { 21021, 24000 };
    description.first.extent = { kWidth, kHeight };
    description.second.extent = description.first.extent;
    description.output = videoopticalflow::canonicalOutput (description.first.extent);
    return description;
}

id<MTLTexture> makeFixtureTexture (id<MTLDevice> device, int horizontalShift,
                                   int verticalShift = 0)
{
    MTLTextureDescriptor* descriptor = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                     width:kWidth
                                    height:kHeight
                                 mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
    if (texture == nil) return nil;

    std::vector<std::uint16_t> pixels (
        static_cast<std::size_t> (kWidth) * kHeight * 4u, floatToHalf (0.0f));
    for (std::size_t pixel = 0; pixel < static_cast<std::size_t> (kWidth) * kHeight;
         ++pixel)
        pixels[pixel * 4u + 3u] = floatToHalf (1.0f);

    constexpr int centerX = 5;
    constexpr int centerY = 5;
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x)
        {
            const auto targetX = centerX + x + horizontalShift;
            const auto targetY = centerY + y + verticalShift;
            const auto offset = (static_cast<std::size_t> (targetY) * kWidth
                                + static_cast<std::size_t> (targetX)) * 4u;
            pixels[offset] = floatToHalf (static_cast<float> (x + 2) * 0.25f);
            pixels[offset + 1u] = floatToHalf (static_cast<float> (y + 2) * 0.25f);
            pixels[offset + 2u] = floatToHalf (
                static_cast<float> ((x + 1) * 3 + (y + 1) + 1) * 0.0625f);
        }

    [texture replaceRegion:MTLRegionMake2D (0, 0, kWidth, kHeight)
               mipmapLevel:0
                 withBytes:pixels.data()
               bytesPerRow:static_cast<NSUInteger> (kWidth * 4u * sizeof (std::uint16_t))];
    return texture;
}

id<MTLTexture> makeAuthoredTexture (id<MTLDevice> device, const std::vector<float>& authored)
{
    MTLTextureDescriptor* descriptor = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                     width:kWidth
                                    height:kHeight
                                 mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
    if (texture == nil) return nil;
    std::vector<std::uint16_t> pixels;
    pixels.reserve(authored.size());
    for (const auto value : authored) pixels.push_back(floatToHalf(value));
    [texture replaceRegion:MTLRegionMake2D (0, 0, kWidth, kHeight)
               mipmapLevel:0
                 withBytes:pixels.data()
               bytesPerRow:static_cast<NSUInteger> (kWidth * 4u * sizeof (std::uint16_t))];
    return texture;
}

class MetalSourceFrame final : public videoopticalflow::NativeSourceFrame
{
public:
    MetalSourceFrame (id<MTLTexture> texture,
                      videoopticalflow::FrameDescription description,
                      videoopticalflow::ResourceIdentity identity,
                      videoopticalflow::BackendIdentity implementation,
                      std::uint32_t implementationRevision)
#if __has_feature(objc_arc)
        : texture_ (texture),
#else
        : texture_ ([texture retain]),
#endif
          description_ (std::move (description)),
          identity_ (identity),
          implementation_ (implementation),
          implementationRevision_ (implementationRevision)
    {
    }

    ~MetalSourceFrame() override
    {
#if ! __has_feature(objc_arc)
        [texture_ release];
#endif
    }

    const videoopticalflow::FrameDescription& frame() const noexcept override
    {
        return description_;
    }
    const videoopticalflow::ResourceIdentity& resourceIdentity() const noexcept override
    {
        return identity_;
    }
    std::uint64_t helperGeneration() const noexcept override
    {
        return kHelperGeneration;
    }
    std::uint64_t structuralRevision() const noexcept override
    {
        return kStructuralRevision;
    }
    const videoopticalflow::BackendIdentity& backendImplementation() const noexcept override
    {
        return implementation_;
    }
    std::uint32_t backendImplementationRevision() const noexcept override
    {
        return implementationRevision_;
    }
    std::uint64_t byteCount() const noexcept override
    {
        return static_cast<std::uint64_t> (kWidth) * kHeight * 8u;
    }
    std::uintptr_t imageHandle() const noexcept override
    {
        return reinterpret_cast<std::uintptr_t> ((__bridge void*) texture_);
    }
    std::uintptr_t textureViewHandle() const noexcept override
    {
        return imageHandle();
    }
    std::uintptr_t nativeContextIdentity() const noexcept override
    {
        return reinterpret_cast<std::uintptr_t> ((__bridge void*) texture_.device);
    }

private:
    id<MTLTexture> texture_ = nil;
    const videoopticalflow::FrameDescription description_;
    const videoopticalflow::ResourceIdentity identity_;
    const videoopticalflow::BackendIdentity implementation_;
    const std::uint32_t implementationRevision_ = 0;
};

std::shared_ptr<const MetalSourceFrame> makeSource (
    id<MTLTexture> texture,
    const videoopticalflow::FrameDescription& description,
    std::uint8_t identityByte,
    const videoopticalflow::BackendCapabilities& capabilities)
{
    videoopticalflow::ResourceIdentity identity {};
    identity[0] = identityByte;
    return std::make_shared<const MetalSourceFrame> (
        texture, description, identity, capabilities.implementation,
        capabilities.implementationRevision);
}

bool readMotion (id<MTLDevice> device, std::uintptr_t outputHandle,
                 std::uint32_t x, std::uint32_t y, float& dx, float& dy,
                 std::vector<float>* field = nullptr)
{
    id<MTLTexture> texture = (__bridge id<MTLTexture>)
        reinterpret_cast<void*> (outputHandle);
    if (texture == nil || texture.pixelFormat != MTLPixelFormatRG16Float)
        return false;

    constexpr NSUInteger bytesPerRow = 256;
    const NSUInteger byteCount = bytesPerRow * kHeight;
    id<MTLBuffer> buffer = [device newBufferWithLength:byteCount
                                               options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    if (buffer == nil || queue == nil || command == nil || blit == nil)
    {
#if ! __has_feature(objc_arc)
        [buffer release];
        [queue release];
#endif
        return false;
    }
    [blit copyFromTexture:texture
              sourceSlice:0
              sourceLevel:0
             sourceOrigin:MTLOriginMake (0, 0, 0)
               sourceSize:MTLSizeMake (kWidth, kHeight, 1)
                 toBuffer:buffer
        destinationOffset:0
   destinationBytesPerRow:bytesPerRow
 destinationBytesPerImage:byteCount];
    [blit endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted)
    {
#if ! __has_feature(objc_arc)
        [buffer release];
        [queue release];
#endif
        return false;
    }
    if (field != nullptr)
    {
        field->resize(kWidth * kHeight * 2u);
        for (std::uint32_t rowIndex = 0; rowIndex < kHeight; ++rowIndex)
        {
            const auto* bytes = static_cast<const std::uint8_t*>(buffer.contents)
                              + rowIndex * bytesPerRow;
            const auto* values = reinterpret_cast<const std::uint16_t*>(bytes);
            for (std::uint32_t column = 0; column < kWidth; ++column)
            {
                (*field)[(rowIndex * kWidth + column) * 2u] = halfToFloat(values[column * 2u]);
                (*field)[(rowIndex * kWidth + column) * 2u + 1u] = halfToFloat(values[column * 2u + 1u]);
            }
        }
    }
    const auto* row = static_cast<const std::uint8_t*> (buffer.contents)
                    + static_cast<std::size_t> (y) * bytesPerRow;
    const auto* values = reinterpret_cast<const std::uint16_t*> (
        row + static_cast<std::size_t> (x) * 4u);
    dx = halfToFloat (values[0]);
    dy = halfToFloat (values[1]);
#if ! __has_feature(objc_arc)
    [buffer release];
    [queue release];
#endif
    return std::isfinite (dx) && std::isfinite (dy);
}

bool readBlurColor (id<MTLDevice> device,
                    const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& frame,
                    std::uint32_t x, std::uint32_t y, float& red,
                    float* green = nullptr,
                    std::vector<std::uint8_t>* rgba = nullptr)
{
    if (!frame || frame->colorTextureDescriptor().format
            != arbitgpu::NativeTexturePixelFormat::Bgra8Unorm)
        return false;
    sg_image image {};
    image.id = static_cast<std::uint32_t> (frame->colorImageHandle());
    const auto native = sg_mtl_query_image_info (image);
    if (native.active_slot < 0 || native.active_slot >= SG_NUM_INFLIGHT_FRAMES)
        return false;
    id<MTLTexture> texture = (__bridge id<MTLTexture>) native.tex[native.active_slot];
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>) sg_mtl_command_queue();
    constexpr NSUInteger bytesPerRow = 256;
    id<MTLBuffer> buffer = [device newBufferWithLength:bytesPerRow * kHeight
                                               options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    if (texture == nil || queue == nil || buffer == nil || command == nil || blit == nil)
    {
#if ! __has_feature(objc_arc)
        [buffer release];
#endif
        return false;
    }
    [blit copyFromTexture:texture sourceSlice:0 sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(kWidth, kHeight, 1)
                 toBuffer:buffer destinationOffset:0
   destinationBytesPerRow:bytesPerRow destinationBytesPerImage:bytesPerRow * kHeight];
    [blit endEncoding];
    [command commit]; [command waitUntilCompleted];
    const bool complete = command.status == MTLCommandBufferStatusCompleted;
    if (complete)
    {
        const auto* bytes = static_cast<const std::uint8_t*> (buffer.contents);
        if (rgba != nullptr)
        {
            rgba->resize(kWidth * kHeight * 4u);
            for (std::uint32_t row = 0; row < kHeight; ++row)
                for (std::uint32_t column = 0; column < kWidth; ++column)
                {
                    const auto source = row * bytesPerRow + column * 4u;
                    const auto target = (row * kWidth + column) * 4u;
                    (*rgba)[target] = bytes[source + 2u];
                    (*rgba)[target + 1u] = bytes[source + 1u];
                    (*rgba)[target + 2u] = bytes[source];
                    (*rgba)[target + 3u] = bytes[source + 3u];
                }
        }
        red = static_cast<float> (bytes[y * bytesPerRow + x * 4u + 2u]) / 255.0f;
        if (green != nullptr)
            *green = static_cast<float> (bytes[y * bytesPerRow + x * 4u + 1u]) / 255.0f;
    }
#if ! __has_feature(objc_arc)
    [buffer release];
#endif
    return complete && std::isfinite(red);
}
} // namespace

int main()
{
    @autoreleasepool
    {
        auto& backend = arbitgpu::nativeOpticalFlowExecutionBackend();
        const auto capabilities = backend.opticalFlowCapabilities();
        if (capabilities.kind != videoopticalflow::BackendKind::NativeGpu
            || ! capabilities.supportsOpticalFlow)
            return 77;

        const auto description = fixtureDescription();
        videoopticalflow::AdmissionFailure admissionFailure =
            videoopticalflow::AdmissionFailure::None;
        auto admitted = videoopticalflow::admit (
            description, capabilities, admissionFailure);
        if (! admitted)
        {
            std::fprintf (stderr, "fixture admission failed: %s\n",
                          std::string (videoopticalflow::token (admissionFailure)).c_str());
            return 1;
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLTexture> firstTexture = makeFixtureTexture (device, 0);
        id<MTLTexture> secondTexture = makeFixtureTexture (device, 1);
        if (device == nil || firstTexture == nil || secondTexture == nil)
            return 77;
        auto first = makeSource (
            firstTexture, admitted->first(), 0x31, capabilities);
        auto second = makeSource (
            secondTexture, admitted->second(), 0x32, capabilities);
#if ! __has_feature(objc_arc)
        [firstTexture release];
        [secondTexture release];
#endif

        opticalflowoperation::Payload payload;
        payload.extent = admitted->output().extent;
        const std::string payloadXml = opticalflowoperation::serialize (payload);
        const videoopticalflow::LoweredOperation operation {
            opticalflowoperation::kOperationKind,
            opticalflowoperation::kBackendCapability,
            payloadXml };
        videoopticalflow::EvaluationContext previewContext;
        previewContext.structuralRevision = kStructuralRevision;
        previewContext.helperGeneration = kHelperGeneration;
        previewContext.mode = videoopticalflow::EvaluationMode::Preview;
        auto offlineContext = previewContext;
        offlineContext.mode = videoopticalflow::EvaluationMode::OfflineSequential;

        videoopticalflow::NativeExecutor preview (
            videoopticalflow::EvaluationMode::Preview, backend);
        videoopticalflow::NativeExecutor exportRun (
            videoopticalflow::EvaluationMode::OfflineSequential, backend);
        std::string error;
        videoopticalflow::NativeExecutionFailure failure =
            videoopticalflow::NativeExecutionFailure::None;
        if (! preview.execute (operation, *admitted, first, second,
                              previewContext, error, &failure))
        {
            std::fprintf (stderr, "preview native execution failed: %s (%s)\n",
                          error.c_str(),
                          std::string (videoopticalflow::token (failure)).c_str());
            return 2;
        }
        if (! exportRun.execute (operation, *admitted, first, second,
                                offlineContext, error, &failure))
        {
            std::fprintf (stderr, "export-owner native execution failed: %s (%s)\n",
                          error.c_str(),
                          std::string (videoopticalflow::token (failure)).c_str());
            return 3;
        }

        const auto previewResult = preview.publication();
        const auto exportResult = exportRun.publication();
        float dx = 0.0f;
        float dy = 0.0f;
        if (previewResult == nullptr || exportResult == nullptr
            || previewResult->imageHandle() == exportResult->imageHandle()
            || previewResult->receipt().motionVectors().identity
                   == exportResult->receipt().motionVectors().identity
            || ! readMotion (device, previewResult->imageHandle(), 5, 5, dx, dy)
            || std::abs (dx - 1.0f) > 0.01f || std::abs (dy) > 0.01f
            || ! readMotion (device, exportResult->imageHandle(), 5, 5, dx, dy)
            || std::abs (dx - 1.0f) > 0.01f || std::abs (dy) > 0.01f)
        {
            std::fprintf (stderr,
                          "Metal optical-flow pixel/isolation mismatch dx=%g dy=%g\n",
                          static_cast<double> (dx), static_cast<double> (dy));
            return 4;
        }

        visualtemporalsampling::Payload blur;
        blur.mode = visualtemporalsampling::Mode::motionBlur;
        blur.sampleCount = 3;
        blur.historyWeight = 0.0f;
        blur.jitterSpread = 0.0f;
        blur.shutterAngleDegrees = 360.0f;
        blur.motionScale = 2.0f;
        auto previewBlur = backend.renderMotionBlur (first, previewResult, blur, error);
        auto exportBlur = backend.renderMotionBlur (first, exportResult, blur, error);
        float previewRed = 0.0f, exportRed = 0.0f;
        if (!previewBlur || !exportBlur
            || !readMotion (device, previewResult->imageHandle(), 6, 5, dx, dy)
            || std::abs(dx - 1.0f) > 0.01f
            || !readBlurColor (device, previewBlur, 6, 5, previewRed)
            || !readBlurColor (device, exportBlur, 6, 5, exportRed)
            || std::abs(previewRed - exportRed) > 1.0f / 255.0f
            || previewRed < 0.55f || previewRed > 0.8f)
        {
            std::fprintf (stderr, "Metal optical blur pixel mismatch preview=%g export=%g\n",
                          double(previewRed), double(exportRed));
            return 12;
        }
        auto invalidBlur = blur;
        invalidBlur.sampleCount = 0;
        if (backend.renderMotionBlur(first, previewResult, invalidBlur, error)
            || error != "Metal Motion Blur requires the exact retained image and optical field")
            return 13;

        auto checkVertical = [&](int direction)
        {
            auto vertical = description;
            vertical.first.identity.content[0] = direction > 0 ? 0x51 : 0x53;
            vertical.second.identity.content[0] = direction > 0 ? 0x52 : 0x54;
            vertical.first.identity.frameIndex = direction > 0 ? 30 : 40;
            vertical.second.identity.frameIndex = vertical.first.identity.frameIndex + 1;
            vertical.first.timestamp = {
                static_cast<std::int64_t>(vertical.first.identity.frameIndex) * 1001, 24000 };
            vertical.second.timestamp = {
                static_cast<std::int64_t>(vertical.second.identity.frameIndex) * 1001, 24000 };
            videoopticalflow::AdmissionFailure verticalFailure {};
            auto admittedVertical = videoopticalflow::admit (
                vertical, capabilities, verticalFailure);
            if (!admittedVertical) return false;
            id<MTLTexture> imageTexture = makeFixtureTexture(device, 0);
            id<MTLTexture> movedTexture = makeFixtureTexture(device, 0, direction);
            if (imageTexture == nil || movedTexture == nil)
            {
#if ! __has_feature(objc_arc)
                [imageTexture release];
                [movedTexture release];
#endif
                return false;
            }
            auto image = makeSource(imageTexture, admittedVertical->first(),
                                    direction > 0 ? 0x41 : 0x43, capabilities);
            auto moved = makeSource(movedTexture, admittedVertical->second(),
                                    direction > 0 ? 0x42 : 0x44, capabilities);
#if ! __has_feature(objc_arc)
            [imageTexture release];
            [movedTexture release];
#endif
            videoopticalflow::NativeExecutor live (
                videoopticalflow::EvaluationMode::Preview, backend);
            videoopticalflow::NativeExecutor offline (
                videoopticalflow::EvaluationMode::OfflineSequential, backend);
            if (!live.execute(operation, *admittedVertical, image, moved,
                              previewContext, error, &failure)
                || !offline.execute(operation, *admittedVertical, image, moved,
                                    offlineContext, error, &failure)
                || !readMotion(device, live.publication()->imageHandle(), 5, 5, dx, dy)
                || std::abs(dx) > 0.01f || std::abs(dy - direction) > 0.01f
                || !readMotion(device, offline.publication()->imageHandle(), 5, 5, dx, dy)
                || std::abs(dx) > 0.01f || std::abs(dy - direction) > 0.01f
                || !readMotion(device, live.publication()->imageHandle(), 5, 6, dx, dy)
                || std::abs(dx) > 0.01f || std::abs(dy - direction) > 0.01f
                || !readMotion(device, offline.publication()->imageHandle(), 5, 6, dx, dy)
                || std::abs(dx) > 0.01f || std::abs(dy - direction) > 0.01f)
                return false;
            auto liveImage = backend.renderMotionBlur(image, live.publication(), blur, error);
            auto offlineImage = backend.renderMotionBlur(image, offline.publication(), blur, error);
            float liveRed = 0.0f, offlineRed = 0.0f;
            float lowerGreen = 0.0f, upperGreen = 0.0f, offlineGreen = 0.0f;
            // Top-first rows 6 and 4 average to sRGB green about 0.678 and 0.537.
            return liveImage && offlineImage
                && readBlurColor(device, liveImage, 5, 6, liveRed, &lowerGreen)
                && readBlurColor(device, offlineImage, 5, 6, offlineRed, &offlineGreen)
                && readBlurColor(device, liveImage, 5, 4, liveRed, &upperGreen)
                && lowerGreen > 0.62f && lowerGreen < 0.74f
                && upperGreen > 0.48f && upperGreen < 0.59f
                && lowerGreen > upperGreen + 0.1f
                && std::abs(lowerGreen - offlineGreen) <= 1.0f / 255.0f;
        };
        if (!checkVertical(1) || !checkVertical(-1)) return 14;

        const auto previewHandle = previewResult->imageHandle();
        if (preview.execute (operation, *admitted, first, second,
                            offlineContext, error, &failure)
            || failure != videoopticalflow::NativeExecutionFailure::OwnerModeMismatch
            || preview.publication()->imageHandle() != previewHandle)
        {
            std::fprintf (stderr, "viewport/export owner-mode isolation failed\n");
            return 5;
        }

        auto aliasedSecond = makeSource (
            (__bridge id<MTLTexture>) reinterpret_cast<void*> (first->imageHandle()),
            admitted->second(), 0x32, capabilities);
        if (preview.execute (operation, *admitted, first, aliasedSecond,
                            previewContext, error, &failure)
            || failure != videoopticalflow::NativeExecutionFailure::DuplicateSourceResource)
        {
            std::fprintf (stderr, "aliased immutable input resource was not rejected\n");
            return 6;
        }

        // Start independent runs for the stationary pair. The retained results
        // must keep both translated outputs readable after their owners reset.
        preview.reset();
        exportRun.reset();
        if (! readMotion (device, previewResult->imageHandle(), 5, 5, dx, dy)
            || std::abs (dx - 1.0f) > 0.01f || std::abs (dy) > 0.01f
            || ! readMotion (device, exportResult->imageHandle(), 5, 5, dx, dy)
            || std::abs (dx - 1.0f) > 0.01f || std::abs (dy) > 0.01f)
        {
            std::fprintf (stderr, "retained Metal publications were invalidated by owner reset\n");
            return 7;
        }
        auto stationaryDescription = description;
        stationaryDescription.second.identity.content = stationaryDescription.first.identity.content;
        const auto stationaryRequest = videoopticalflow::admit (
            stationaryDescription, capabilities, admissionFailure);
        if (! stationaryRequest) return 8;
        id<MTLTexture> stationaryTexture = makeFixtureTexture (device, 0);
        if (stationaryTexture == nil) return 9;
        const auto stationary = makeSource (
            stationaryTexture, stationaryRequest->second(), 0x33, capabilities);
#if ! __has_feature(objc_arc)
        [stationaryTexture release];
#endif
        if (! preview.execute (operation, *stationaryRequest, first, stationary,
                               previewContext, error, &failure)
            || ! exportRun.execute (operation, *stationaryRequest, first, stationary,
                                    offlineContext, error, &failure))
        {
            std::fprintf (stderr, "stationary native execution failed: %s\n", error.c_str());
            return 10;
        }
        if (! readMotion (device, preview.publication()->imageHandle(), 5, 5, dx, dy)
            || std::abs (dx) > 0.01f || std::abs (dy) > 0.01f
            || ! readMotion (device, exportRun.publication()->imageHandle(), 5, 5, dx, dy)
            || std::abs (dx) > 0.01f || std::abs (dy) > 0.01f)
        {
            std::fprintf (stderr, "stationary Metal optical-flow mismatch dx=%g dy=%g\n",
                          static_cast<double> (dx), static_cast<double> (dy));
            return 11;
        }

        for (const auto& pair : opticalflowtest::pairs())
        {
            id<MTLTexture> authoredFirstTexture = makeAuthoredTexture(device, pair.first);
            id<MTLTexture> authoredSecondTexture = makeAuthoredTexture(device, pair.second);
            if (authoredFirstTexture == nil || authoredSecondTexture == nil) return 15;
            auto authoredFirst = makeSource(authoredFirstTexture, admitted->first(), 0x81, capabilities);
            auto authoredSecond = makeSource(authoredSecondTexture, admitted->second(), 0x82, capabilities);
#if ! __has_feature(objc_arc)
            [authoredFirstTexture release];
            [authoredSecondTexture release];
#endif
            videoopticalflow::NativeExecutor live(videoopticalflow::EvaluationMode::Preview, backend);
            videoopticalflow::NativeExecutor offline(videoopticalflow::EvaluationMode::OfflineSequential, backend);
            if (!live.execute(operation, *admitted, authoredFirst, authoredSecond, previewContext, error, &failure)
                || !offline.execute(operation, *admitted, authoredFirst, authoredSecond, offlineContext, error, &failure))
                return 16;
            std::vector<float> field;
            if (!readMotion(device, live.publication()->imageHandle(), 0, 0, dx, dy, &field)
                || !opticalflowtest::verify(pair, field, "metal-preview")
                || !readMotion(device, offline.publication()->imageHandle(), 0, 0, dx, dy, &field)
                || !opticalflowtest::verify(pair, field, "metal-export")) return 17;
            const auto verifyViews = [&](const std::shared_ptr<const videoopticalflow::PublishedNativeResult>& publication,
                                         const char* mode)
            {
                std::vector<float> before, after;
                if (!readMotion(device, publication->imageHandle(), 0, 0, dx, dy, &before)) return false;
                const auto colorBytes = [&](const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& image,
                                            std::vector<std::uint8_t>& bytes)
                { return readBlurColor(device, image, 0, 0, previewRed, nullptr, &bytes); };
                std::vector<std::uint8_t> legacy, zero, ranged;
                if (!colorBytes(backend.renderMotionView(publication, error), legacy)
                    || !colorBytes(backend.renderMotionView(publication, motionviewoperation::Payload {}, error), zero)
                    || legacy != zero || !opticalflowtest::verifyView(pair, legacy, 0.0, mode, 1)
                    || !colorBytes(backend.renderMotionView(publication, motionviewoperation::Payload {
                        motionviewoperation::Convention::OpticalFirstToSecondTopLeft, 16.0 }, error), ranged)
                    || !opticalflowtest::verifyView(pair, ranged, 16.0, mode, 0)) return false;
                if (!colorBytes(backend.renderMotionView(publication, motionviewoperation::Payload {
                        motionviewoperation::Convention::OpticalFirstToSecondTopLeft, 1.0 }, error), ranged)
                    || !opticalflowtest::verifyView(pair, ranged, 1.0, mode, 0)) return false;
                for (const auto invalid : { -1.0, 0.5, 65537.0,
                        std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN() })
                    if (backend.renderMotionView(publication, motionviewoperation::Payload {
                            motionviewoperation::Convention::OpticalFirstToSecondTopLeft, invalid }, error)) return false;
                if (backend.renderMotionView(publication, motionviewoperation::Payload {
                        motionviewoperation::Convention::GeometricCurrentMinusPreviousYUp, 16.0 }, error)) return false;
                return readMotion(device, publication->imageHandle(), 0, 0, dx, dy, &after)
                    && before == after && opticalflowtest::verify(pair, after, mode);
            };
            if (!verifyViews(live.publication(), "metal-preview")
                || !verifyViews(offline.publication(), "metal-export")) return 21;
            auto stale = previewContext;
            ++stale.structuralRevision;
            const auto retainedHandle = live.publication()->imageHandle();
            if (live.execute(operation, *admitted, authoredFirst, authoredSecond, stale, error, &failure)
                || failure != videoopticalflow::NativeExecutionFailure::SourceResourceMismatch
                || live.publication()->imageHandle() != retainedHandle) return 18;
            if (live.execute(operation, *admitted, authoredFirst, authoredSecond, offlineContext, error, &failure)
                || failure != videoopticalflow::NativeExecutionFailure::OwnerModeMismatch
                || live.publication()->imageHandle() != retainedHandle) return 19;
            if (live.execute(operation, *admitted, authoredSecond, authoredFirst, previewContext, error, &failure)
                || failure != videoopticalflow::NativeExecutionFailure::SourceResourceMismatch
                || live.publication()->imageHandle() != retainedHandle) return 20;
        }

        std::printf (
            "optical-flow-metal PASS backend=metal extent=%ux%u translated=1,0 stationary=0,0 "
            "preview_export_pixels=yes textured_xy_oracle=yes stationary_exact_zero=yes occluded_background_zero=yes ambiguous_zero=yes "
            "motion_view_range16_pixels=yes motion_view_clamp_pixels=yes legacy_zero_image_bytes=yes raw_field_unchanged=yes optical_blur_xy_pixels=yes preview_export_isolated=yes finite_readback=yes "
            "retained_publications=yes aliased_input_rejected=yes\n",
            kWidth, kHeight);
        return 0;
    }
}
