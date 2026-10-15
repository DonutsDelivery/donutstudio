#include "optical_flow_native_source_upload.h"
#include "../../shared/ColorTransformGpuMath.h"

#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <memory>
#include <vector>

namespace arbitgpu
{
std::uintptr_t nativeMetalOpticalFlowDeviceHandle() noexcept;
}

namespace videoopticalflow
{
namespace
{
constexpr const char* kSourceConversion = R"METAL(
#include <metal_stdlib>
using namespace metal;
struct Params {
    float4x4 inputToWorking;
    float4x4 workingToOutput;
    float4 luminance;
    int4 options;
};
float3 decode(float3 value, int transfer, float referenceWhite) {
    value = max(value, float3(0.0f));
    if (transfer == 1) return value;
    if (transfer == 2) return select(pow((value + 0.055f) / 1.055f, float3(2.4f)),
                                     value / 12.92f, value <= float3(0.04045f));
    if (transfer == 3) return pow(value, float3(2.2f));
    if (transfer == 6) return select(pow((value + 0.099f) / 1.099f,
                                     float3(1.0f / 0.45f)), value / 4.5f,
                                     value < float3(0.081f));
    if (transfer == 4) {
        const float m1 = 2610.0f / 16384.0f, m2 = 2523.0f / 32.0f;
        const float c1 = 3424.0f / 4096.0f, c2 = 2413.0f / 128.0f;
        const float c3 = 2392.0f / 128.0f;
        float3 p = pow(value, float3(1.0f / m2));
        return 10000.0f * pow(max(p - c1, float3(0.0f))
            / max(c2 - c3 * p, float3(1.0e-6f)), float3(1.0f / m1))
            / referenceWhite;
    }
    const float a = 0.17883277f, b = 0.28466892f, c = 0.55991073f;
    return select((exp((value - c) / a) + b) / 12.0f,
                  value * value / 3.0f, value <= float3(0.5f));
}
kernel void uploadSource(texture2d<float, access::read> source [[texture(0)]],
                         texture2d<half, access::write> target [[texture(1)]],
                         constant Params& params [[buffer(0)]],
                         uint2 pixel [[thread_position_in_grid]]) {
    if (pixel.x >= target.get_width() || pixel.y >= target.get_height()) return;
    float4 rgba = source.read(pixel);
    float3 working = (params.inputToWorking
        * float4(decode(rgba.rgb, params.options.x, params.luminance.x), 1.0f)).rgb;
    float3 linear = (params.workingToOutput
        * float4(working * (params.luminance.x / params.luminance.y), 1.0f)).rgb;
    target.write(half4(half3(linear), half(rgba.a)), pixel);
}
)METAL";

struct alignas(16) ConversionParams
{
    float inputToWorking[16] {};
    float workingToOutput[16] {};
    float luminance[4] {};
    std::int32_t options[4] {};
};

class MetalSourceResource final : public UploadedSourceResource
{
public:
    explicit MetalSourceResource (id<MTLTexture> texture) : texture_ (texture) {}
    ~MetalSourceResource() override
    {
#if ! __has_feature(objc_arc)
        [texture_ release];
#endif
    }
    std::uintptr_t imageHandle() const noexcept override
    { return reinterpret_cast<std::uintptr_t> ((__bridge void*) texture_); }
    std::uintptr_t nativeContextIdentity() const noexcept override
    { return reinterpret_cast<std::uintptr_t> ((__bridge void*) texture_.device); }

private:
    id<MTLTexture> texture_ = nil;
};

class MetalSourceUploader final : public NativeSourceUploader
{
public:
    MetalSourceUploader (id<MTLDevice> device, std::uint64_t helperGeneration,
                         std::uint64_t structuralRevision)
        : device_ (device), helperGeneration_ (helperGeneration),
          structuralRevision_ (structuralRevision), queue_ ([device newCommandQueue]) {}

    ~MetalSourceUploader() override
    {
#if ! __has_feature(objc_arc)
        [pipeline_ release];
        [queue_ release];
#endif
    }

    std::shared_ptr<const NativeSourceFrame> uploadPreview (
        const DecodedSourceReceipt& decoded,
        const colortransform::AdmittedTransform& transform,
        const SourceFrameV2& expected, bool pixelsValidatedOnWorker,
        std::string& error) override
    {
        preview_ = true;
        previewPixelsValidated_ = pixelsValidatedOnWorker;
        auto result = upload (decoded, transform, expected, error);
        preview_ = false;
        previewPixelsValidated_ = false;
        return result;
    }

    std::shared_ptr<const NativeSourceFrame> upload (
        const DecodedSourceReceipt& decoded,
        const colortransform::AdmittedTransform& transform,
        const SourceFrameV2& expected, std::string& error) override
    {
        error.clear();
        if (! validateUploadSource (decoded, transform, expected, error,
                                    preview_ && previewPixelsValidated_)) return {};
        const auto backend = arbitgpu::nativeOpticalFlowExecutionBackend()
                                 .opticalFlowCapabilities();
        if (backend.kind != BackendKind::NativeGpu || ! backend.supportsOpticalFlow
            || device_ == nil
            || reinterpret_cast<std::uintptr_t> ((__bridge void*) device_)
                   != arbitgpu::nativeMetalOpticalFlowDeviceHandle()
            || queue_ == nil || ! ensurePipeline (error))
        {
            if (error.empty()) error = "Optical Flow Metal source backend is unavailable";
            return {};
        }
        if (preview_)
        {
            for (const auto& pending : previewUploads_)
                if (pending->expected.clipId == expected.clipId
                    && pending->expected.sourceContent == expected.sourceContent
                    && pending->expected.sourceGeneration == expected.sourceGeneration
                    && pending->expected.frame.identity.stream == expected.frame.identity.stream
                    && pending->expected.frame.identity.frameIndex == expected.frame.identity.frameIndex
                    && pending->expected.frame.timestamp.ticks == expected.frame.timestamp.ticks
                    && pending->expected.frame.timestamp.timescale == expected.frame.timestamp.timescale
                    && pending->expected.frame.extent == expected.frame.extent
                    && pending->pixelSha256 == decoded.pixelSha256
                    && pending->colorIdentity == transform.cacheIdentity())
                {
                    if (! pending->ready.load (std::memory_order_acquire))
                    { error = kPreviewUploadPending; return {}; }
                    if (! pending->error.empty())
                    { error = pending->error; return {}; }
                    return std::make_shared<const UploadedSourceFrame> (
                        expected, pending->content, backend, helperGeneration_,
                        structuralRevision_, pending->resource);
                }
        }
        const auto& color = transform.description();
        if (color.output.transfer != colortransform::TransferFunction::Linear
            || color.output.colorSpace != colortransform::ColorSpace::LinearSRGB
            || color.output.format != colortransform::PixelFormat::RGBA16F
            || color.input.format != colortransform::PixelFormat::RGBA8
            || color.input.alpha != colortransform::AlphaMode::Straight
            || color.output.alpha != colortransform::AlphaMode::Straight
            || color.toneMap != colortransform::ToneMap::None)
        {
            error = "Optical Flow Metal source transform has an unsupported output";
            return {};
        }
        @autoreleasepool
        {
            const auto width = static_cast<NSUInteger> (decoded.frame.width);
            const auto height = static_cast<NSUInteger> (decoded.frame.height);
            MTLTextureDescriptor* sourceDesc =
                [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                   width:width height:height mipmapped:NO];
            sourceDesc.storageMode = MTLStorageModeShared;
            sourceDesc.usage = MTLTextureUsageShaderRead;
            MTLTextureDescriptor* targetDesc =
                [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                   width:width height:height mipmapped:NO];
            targetDesc.storageMode = MTLStorageModeShared;
            targetDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
            id<MTLTexture> source = [device_ newTextureWithDescriptor:sourceDesc];
            id<MTLTexture> target = [device_ newTextureWithDescriptor:targetDesc];
            if (source == nil || target == nil)
            {
#if ! __has_feature(objc_arc)
                [source release]; [target release];
#endif
                error = "Optical Flow Metal source texture allocation failed";
                return {};
            }
            [source replaceRegion:MTLRegionMake2D(0, 0, width, height)
                    mipmapLevel:0 withBytes:decoded.frame.rgba.data()
                    bytesPerRow:width * 4u];
            ConversionParams params;
            const auto input = colortransform::gpumath::inputToWorking (color);
            const auto output = colortransform::gpumath::workingToOutput (color);
            std::copy (input.begin(), input.end(), params.inputToWorking);
            std::copy (output.begin(), output.end(), params.workingToOutput);
            params.luminance[0] = static_cast<float> (color.luminance.sourceReferenceWhiteNits);
            params.luminance[1] = static_cast<float> (color.luminance.outputReferenceWhiteNits);
            params.options[0] = static_cast<std::int32_t> (color.input.transfer);
            id<MTLCommandBuffer> command = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            if (command == nil || encoder == nil)
                error = "Optical Flow Metal source command allocation failed";
            else
            {
                [encoder setComputePipelineState:pipeline_];
                [encoder setTexture:source atIndex:0];
                [encoder setTexture:target atIndex:1];
                [encoder setBytes:&params length:sizeof (params) atIndex:0];
                const auto groupWidth = pipeline_.threadExecutionWidth;
                const auto groupHeight = std::max<NSUInteger> (
                    1, pipeline_.maxTotalThreadsPerThreadgroup / groupWidth);
                [encoder dispatchThreads:MTLSizeMake(width, height, 1)
                     threadsPerThreadgroup:MTLSizeMake(groupWidth, groupHeight, 1)];
                [encoder endEncoding];
                if (preview_)
                {
                    auto pending = std::make_shared<PreviewUpload>();
                    pending->expected = expected;
                    pending->pixelSha256 = decoded.pixelSha256;
                    pending->colorIdentity = transform.cacheIdentity();
                    pending->resource = std::make_shared<MetalSourceResource> (target);
                    [command addCompletedHandler:^(id<MTLCommandBuffer> completed)
                    {
                        if (completed.status != MTLCommandBufferStatusCompleted)
                            pending->error = completed.error != nil
                                ? std::string ([[completed.error localizedDescription] UTF8String])
                                : "Optical Flow Metal source conversion did not complete";
                        else
                        {
                            std::vector<std::uint8_t> pixels (width * height * 8u);
                            id<MTLTexture> result = (__bridge id<MTLTexture>)
                                reinterpret_cast<void*> (pending->resource->imageHandle());
                            [result getBytes:pixels.data() bytesPerRow:width * 8u
                                fromRegion:MTLRegionMake2D(0, 0, width, height) mipmapLevel:0];
                            if (! detail::decodeSha256 (
                                videohelper::sha256Bytes (pixels.data(), pixels.size()),
                                pending->content))
                                pending->error = "Optical Flow Metal linear source digest failed";
                        }
                        pending->ready.store (true, std::memory_order_release);
                    }];
                    [command commit];
#if ! __has_feature(objc_arc)
                    [source release];
#endif
                    if (previewUploads_.size() >= 4)
                        previewUploads_.erase (previewUploads_.begin());
                    previewUploads_.push_back (std::move (pending));
                    error = kPreviewUploadPending;
                    return {};
                }
                [command commit];
                [command waitUntilCompleted];
                if (command.status != MTLCommandBufferStatusCompleted)
                    error = command.error != nil
                        ? std::string ([[command.error localizedDescription] UTF8String])
                        : "Optical Flow Metal source conversion did not complete";
            }
#if ! __has_feature(objc_arc)
            [source release];
#endif
            if (! error.empty())
            {
#if ! __has_feature(objc_arc)
                [target release];
#endif
                return {};
            }
            std::vector<std::uint8_t> pixels (width * height * 8u);
            [target getBytes:pixels.data() bytesPerRow:width * 8u
                    fromRegion:MTLRegionMake2D(0, 0, width, height) mipmapLevel:0];
            ContentIdentity content {};
            if (! detail::decodeSha256 (
                    videohelper::sha256Bytes (pixels.data(), pixels.size()), content))
            {
#if ! __has_feature(objc_arc)
                [target release];
#endif
                error = "Optical Flow Metal linear source digest failed";
                return {};
            }
            auto resource = std::make_shared<MetalSourceResource> (target);
            return std::make_shared<const UploadedSourceFrame> (
                expected, content, backend, helperGeneration_, structuralRevision_,
                std::move (resource));
        }
    }

private:
    struct PreviewUpload
    {
        SourceFrameV2 expected;
        std::string pixelSha256;
        colortransform::CacheIdentity colorIdentity;
        std::shared_ptr<MetalSourceResource> resource;
        ContentIdentity content {};
        std::string error;
        std::atomic<bool> ready { false };
    };

    bool ensurePipeline (std::string& error)
    {
        if (pipeline_ != nil) return true;
        NSError* failure = nil;
        NSString* source = [NSString stringWithUTF8String:kSourceConversion];
        id<MTLLibrary> library = [device_ newLibraryWithSource:source options:nil error:&failure];
        id<MTLFunction> function = [library newFunctionWithName:@"uploadSource"];
        if (function != nil)
            pipeline_ = [device_ newComputePipelineStateWithFunction:function error:&failure];
#if ! __has_feature(objc_arc)
        [function release]; [library release];
#endif
        if (pipeline_ != nil) return true;
        error = failure != nil
            ? std::string ([[failure localizedDescription] UTF8String])
            : "Optical Flow Metal source conversion pipeline is unavailable";
        return false;
    }

    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    bool preview_ = false;
    bool previewPixelsValidated_ = false;
    std::vector<std::shared_ptr<PreviewUpload>> previewUploads_;
    id<MTLComputePipelineState> pipeline_ = nil;
    std::uint64_t helperGeneration_ = 0;
    std::uint64_t structuralRevision_ = 0;
};
} // namespace

std::unique_ptr<NativeSourceUploader> makeNativeSourceUploader (
    std::uint64_t helperGeneration, std::uint64_t structuralRevision)
{
    if (helperGeneration == 0 || structuralRevision == 0) return {};
    auto* device = reinterpret_cast<void*> (
        arbitgpu::nativeMetalOpticalFlowDeviceHandle());
    if (device == nullptr) return {};
    return std::make_unique<MetalSourceUploader> (
        (__bridge id<MTLDevice>) device, helperGeneration, structuralRevision);
}
void invalidateNativeSourceUploads (std::uintptr_t) noexcept {}
} // namespace videoopticalflow
