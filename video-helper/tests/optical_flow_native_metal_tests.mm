#import <Metal/Metal.h>

#include "../src/optical_flow_native_executor.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace
{
constexpr std::uint32_t kWidth = 12;
constexpr std::uint32_t kHeight = 10;
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

id<MTLTexture> makeFixtureTexture (id<MTLDevice> device, int horizontalShift)
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
            const auto targetY = centerY + y;
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
                 std::uint32_t x, std::uint32_t y, float& dx, float& dy)
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
    return true;
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
            || std::abs (dx - 1.0f) > 0.01f || std::abs (dy) > 0.01f)
        {
            std::fprintf (stderr,
                          "Metal optical-flow pixel/isolation mismatch dx=%g dy=%g\n",
                          static_cast<double> (dx), static_cast<double> (dy));
            return 4;
        }

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

        std::printf (
            "optical-flow-metal PASS backend=metal extent=%ux%u vector=%g,%g "
            "preview_export_isolated=yes aliased_input_rejected=yes\n",
            kWidth, kHeight, static_cast<double> (dx), static_cast<double> (dy));
        return 0;
    }
}
