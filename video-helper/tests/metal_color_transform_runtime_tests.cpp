#include "gpu_backend/frame_renderer_metal.h"
#include "renderer.h"
#include "visual_plan_executor.h"
#include "../../shared/ColorTransformOperationContract.h"

#include <CoreFoundation/CoreFoundation.h>
#include <IOSurface/IOSurface.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace
{
constexpr int kWidth = 8;
constexpr int kHeight = 2;
// The production path performs nonlinear sRGB/Reinhard math in an RGBA16F
// Metal target, then crosses the IOSurface/OpenGL RGBA8 quantization boundary.
// Three code values bound those independent half-float and UNORM roundings;
// the physical acceptance output is also reported so the observed error stays visible.
constexpr int kTolerance = 3;
constexpr int kClipId = 19;
constexpr unsigned kSourceHandle = 7301;

constexpr std::array<std::uint8_t, kWidth> kEncoded {
    24, 48, 72, 96, 128, 160, 188, 224
};

bool require(bool condition, const std::string& message)
{
    if (! condition)
        std::cerr << "FAIL: " << message << '\n';
    return condition;
}

colortransformoperation::Payload colorTransformPayload()
{
    colortransformoperation::Payload payload;
    payload.inputFormat = colortransform::PixelFormat::RGBA8;
    payload.inputColorSpace = colortransform::ColorSpace::SRGB;
    payload.inputTransfer = colortransform::TransferFunction::SRGB;
    payload.inputAlpha = colortransform::AlphaMode::Straight;
    payload.outputFormat = colortransform::PixelFormat::RGBA8;
    payload.outputColorSpace = colortransform::ColorSpace::SRGB;
    payload.outputTransfer = colortransform::TransferFunction::SRGB;
    payload.outputAlpha = colortransform::AlphaMode::Straight;
    payload.workingColorSpace = colortransform::ColorSpace::LinearSRGB;
    payload.workingFormat = colortransform::PixelFormat::RGBA16F;
    payload.outputIntent = colortransform::OutputIntent::SdrDisplay;
    payload.toneMap = colortransform::ToneMap::Reinhard;
    payload.luminance = { 100.0, 1000.0, 100.0, 100.0 };
    return payload;
}

videowire::CompiledVisualLayerPlan compiledPlan()
{
    videowire::CompiledVisualLayerPlan plan;
    plan.clipId = kClipId;
    plan.structuralRevision = 7;
    plan.producerValidated = true;
    plan.nodeKinds = { "video.source", "visual.color.transform", "video.out" };
    plan.nodeIds = { 11, 12, 13 };
    plan.edges = { { 11, 0, 12, 0 }, { 12, 1, 13, 0 } };
    plan.ports = {
        { 11, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 12, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 12, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 13, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    plan.operations = {
        videowire::CompiledVisualOperation { 11, "video.source", "source-decode", "" },
        videowire::CompiledVisualOperation {
            12, std::string(colortransformoperation::kOperationKind),
            std::string(colortransformoperation::kBackendCapability),
            colortransformoperation::serialize(colorTransformPayload()) },
        videowire::CompiledVisualOperation { 13, "video.out", "native-gpu", "" }
    };
    return plan;
}

int referenceChannel(std::uint8_t encoded)
{
    // Independent CPU oracle for the admitted sRGB -> linear-sRGB Reinhard ->
    // sRGB operation. It follows the public transform contract rather than the
    // shared GPU matrix/uniform helpers or Metal shader source.
    const double value = static_cast<double>(encoded) / 255.0;
    const double linear = value <= 0.04045
        ? value / 12.92
        : std::pow((value + 0.055) / 1.055, 2.4);
    const double absoluteNits = linear * 100.0;
    const double shoulder = 1.0 / 100.0 - 1.0 / 1000.0;
    const double mappedLinear = absoluteNits
        / (1.0 + shoulder * absoluteNits) / 100.0;
    const double output = mappedLinear <= 0.0031308
        ? mappedLinear * 12.92
        : 1.055 * std::pow(mappedLinear, 1.0 / 2.4) - 0.055;
    return static_cast<int>(std::lround(std::clamp(output, 0.0, 1.0) * 255.0));
}

std::vector<std::uint8_t> sourcePixels()
{
    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(kWidth) * kHeight * 4u);
    for (int y = 0; y < kHeight; ++y)
        for (int x = 0; x < kWidth; ++x)
        {
            const auto offset = (static_cast<std::size_t>(y) * kWidth + x) * 4u;
            pixels[offset] = kEncoded[static_cast<std::size_t>(x)];
            pixels[offset + 1] = kEncoded[static_cast<std::size_t>(x)];
            pixels[offset + 2] = kEncoded[static_cast<std::size_t>(x)];
            pixels[offset + 3] = 255;
        }
    return pixels;
}


bool matchesReference(const std::vector<std::uint8_t>& pixels)
{
    if (pixels.size() != static_cast<std::size_t>(kWidth) * kHeight * 4u)
        return false;
    for (int y = 0; y < kHeight; ++y)
        for (int x = 0; x < kWidth; ++x)
        {
            const auto offset = (static_cast<std::size_t>(y) * kWidth + x) * 4u;
            const int expected = referenceChannel(kEncoded[static_cast<std::size_t>(x)]);
            for (int channel = 0; channel < 3; ++channel)
                if (std::abs(static_cast<int>(pixels[offset + channel]) - expected)
                    > kTolerance)
                {
                    std::cerr << "pixel mismatch x=" << x << " y=" << y
                              << " channel=" << channel
                              << " actual=" << static_cast<int>(pixels[offset + channel])
                              << " expected=" << expected
                              << " tolerance=" << kTolerance << '\n';
                    return false;
                }
            if (pixels[offset + 3] != 255)
            {
                std::cerr << "alpha mismatch x=" << x << " y=" << y
                          << " actual=" << static_cast<int>(pixels[offset + 3]) << '\n';
                return false;
            }
        }
    return true;
}

void printReadbackEvidence(const std::vector<std::uint8_t>& pixels)
{
    std::cout << "source=";
    for (int x = 0; x < kWidth; ++x)
        std::cout << (x == 0 ? "[" : ",") << static_cast<int>(kEncoded[static_cast<std::size_t>(x)]);
    std::cout << "] oracle=";
    for (int x = 0; x < kWidth; ++x)
        std::cout << (x == 0 ? "[" : ",") << referenceChannel(kEncoded[static_cast<std::size_t>(x)]);
    std::cout << "] gpu-row0=";
    for (int x = 0; x < kWidth; ++x)
    {
        const auto offset = static_cast<std::size_t>(x) * 4u;
        std::cout << (x == 0 ? "[" : ",") << static_cast<int>(pixels[offset]);
    }
    std::cout << "] gpu-row1=";
    for (int x = 0; x < kWidth; ++x)
    {
        const auto offset = (static_cast<std::size_t>(kWidth) + x) * 4u;
        std::cout << (x == 0 ? "[" : ",") << static_cast<int>(pixels[offset]);
    }
    std::cout << "]\n";
}
} // namespace

int main()
{
#if !defined(__APPLE__) || !defined(__arm64__)
#error "strict Metal color-transform acceptance must compile and run on Apple Silicon"
#endif
    bool ok = true;
    const std::vector<videowire::CompiledVisualLayerPlan> plans { compiledPlan() };
    std::string error;
    const auto rejectBeforeMetalAllocation = [&] (
        const std::vector<videowire::CompiledVisualLayerPlan>& malformed,
        const std::string& label)
    {
        videorender::LayerDesc output;
        output.texture = kSourceHandle;
        output.graphColorTransformActive = true;
        videorender::FrameRenderer rejectedRenderer;
        std::string preflightError;
        videorender::MetalFrameRenderer::resetAllocationCountersForTesting();
        const bool admitted = rejectedRenderer.initializeForVisualPlans(
            nullptr, kWidth, kHeight, malformed, preflightError, true,
            kClipId, kSourceHandle, &output);
        ok &= require(! admitted && ! rejectedRenderer.ready()
                          && ! output.graphColorTransformActive && output.texture == 0
                          && videorender::MetalFrameRenderer::allocationCountersForTesting().empty(),
                      label + " was not rejected with cleared output before every Metal allocation");
    };

    auto malformedBackend = plans;
    malformedBackend[0].operations[1].backendCapability = "opengl";
    rejectBeforeMetalAllocation(malformedBackend, "malformed backend preflight");
    auto malformedDescriptor = plans;
    malformedDescriptor[0].ports[1].colorSpace = "linearSRGB";
    rejectBeforeMetalAllocation(malformedDescriptor, "malformed descriptor preflight");
    auto malformedPayload = plans;
    malformedPayload[0].operations[1].payloadXml = "{";
    rejectBeforeMetalAllocation(malformedPayload, "malformed payload preflight");
    if (! ok)
        return EXIT_FAILURE;

    videorender::LayerDesc previewLayer;
    videorender::FrameRenderer renderer;
    if (! renderer.initializeForVisualPlans(
            nullptr, kWidth, kHeight, plans, error, true,
            kClipId, kSourceHandle, &previewLayer))
    {
        std::cerr << "strict Metal compositor initialization failed: " << error << '\n';
        return EXIT_FAILURE;
    }
    ok &= require(previewLayer.texture == kSourceHandle
                      && previewLayer.graphColorTransformActive
                      && previewLayer.graphColorTransform.toneMap
                          == colortransform::ToneMap::Reinhard
                      && previewLayer.graphColorTransform.input.extent
                          == colortransform::Extent { kWidth, kHeight }
                      && previewLayer.graphColorTransform.output.extent
                          == colortransform::Extent { kWidth, kHeight },
                  "production Metal entry point did not bind the complete admitted transform");
    if (! renderer.ready())
    {
        std::cerr << "FAIL: Metal compositor initialization returned without a ready renderer: "
                  << renderer.lastError() << '\n';
        renderer.shutdown();
        return EXIT_FAILURE;
    }
    renderer.setCanvas(kWidth, kHeight);
    renderer.setBackgroundColor(0.0f, 0.0f, 0.0f, 1.0f);
    const auto source = sourcePixels();
    previewLayer.texture = renderer.uploadRgba(
        source.data(), kWidth, kHeight, kWidth * 4, kSourceHandle);

    CFMutableDictionaryRef properties = CFDictionaryCreateMutable(
        kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks,
        &kCFTypeDictionaryValueCallBacks);
    auto setSurfaceInt = [properties](CFStringRef key, int value)
    {
        CFNumberRef number = CFNumberCreate(kCFAllocatorDefault, kCFNumberIntType, &value);
        CFDictionarySetValue(properties, key, number);
        CFRelease(number);
    };
    setSurfaceInt(kIOSurfaceWidth, kWidth);
    setSurfaceInt(kIOSurfaceHeight, kHeight);
    setSurfaceInt(kIOSurfaceBytesPerElement, 4);
    setSurfaceInt(kIOSurfacePixelFormat, static_cast<int32_t>('BGRA'));
    IOSurfaceRef surface = IOSurfaceCreate(properties);
    CFRelease(properties);
    if (surface == nullptr)
    {
        std::cerr << "FAIL: strict Metal test IOSurface allocation failed\n";
        renderer.shutdown();
        return EXIT_FAILURE;
    }

    std::vector<std::uint8_t> actual;
    for (int frame = 0; frame < 3; ++frame)
    {
        const bool rendered = renderer.renderCompositeToIOSurface(
            surface, kWidth, kHeight, &previewLayer, 1);
        ok &= require(rendered,
                      "strict Metal compositor rejected the admitted transform: "
                          + renderer.lastError());
        std::vector<std::uint8_t> framePixels(static_cast<size_t>(kWidth * kHeight * 4));
        IOSurfaceLock(surface, kIOSurfaceLockReadOnly, nullptr);
        const auto* bytes = static_cast<const std::uint8_t*>(IOSurfaceGetBaseAddress(surface));
        const size_t stride = IOSurfaceGetBytesPerRow(surface);
        for (int y = 0; y < kHeight; ++y)
            for (int x = 0; x < kWidth; ++x)
            {
                const auto* bgra = bytes + static_cast<size_t>(y) * stride + x * 4;
                auto* rgba = framePixels.data() + static_cast<size_t>(y * kWidth + x) * 4;
                rgba[0] = bgra[2]; rgba[1] = bgra[1]; rgba[2] = bgra[0]; rgba[3] = bgra[3];
            }
        IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, nullptr);
        ok &= require(matchesReference(framePixels),
                      "Metal GPU readback did not match the independent CPU color-transform oracle");
        if (frame == 0)
            actual = framePixels;
        else
            ok &= require(framePixels == actual,
                          "Metal GPU readback changed across identical production frames");
    }
    ok &= require(actual != source,
                  "Metal output matched the source bytes; shader execution was not proven");

    auto invalidBackendLayer = previewLayer;
    invalidBackendLayer.graphColorTransform.toneMap = colortransform::ToneMap::None;
    const bool invalidOutput = renderer.renderCompositeToIOSurface(
        surface, kWidth, kHeight, &invalidBackendLayer, 1);
    ok &= require(! invalidOutput
                      && renderer.lastError()
                          == "Metal color transform admission failed: toneMapRequired",
                  "Metal execution did not fail closed on an inadmissible exact descriptor");

    CFRelease(surface);
    renderer.deleteTexture(previewLayer.texture);
    renderer.shutdown();

    if (! ok)
        return EXIT_FAILURE;
    printReadbackEvidence(actual);
    std::cout << "strict Metal color transform readback: PASS; tolerance="
              << kTolerance << " RGBA8 code values\n";
    return EXIT_SUCCESS;
}
