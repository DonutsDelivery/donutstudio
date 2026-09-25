#include "renderer.h"
#include "visual_plan_executor.h"
#include "volume_visual_plan_execution.h"
#include "support/volume_plan_fixture.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#if defined(__APPLE__)
#include "gpu_backend/sokol_metal_context.h"
#import <Metal/Metal.h>
#include <CoreFoundation/CoreFoundation.h>
#include <IOSurface/IOSurface.h>
#define SOKOL_METAL
#include "sokol_gfx.h"
#else
#include <GLFW/glfw3.h>
#endif

namespace
{
using namespace videohelper::volume;
using Pixels = std::vector<std::uint8_t>;
constexpr int width = 96;
constexpr int height = 64;
constexpr std::array<std::uint8_t, 4> background {19, 43, 173, 255};

void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

// Readback is test evidence only. The production preparation and composition
// below pass the native frame directly, without copying it through the CPU.
class Context final
{
public:
    ~Context() { close(); }
    bool open()
    {
#if defined(__APPLE__)
        if (!nativeVolumeExecutionBackend().capabilities().volume.nativeGpuAvailable) return false;
        auto properties = CFDictionaryCreateMutable(nullptr, 0,
            &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
        const auto put = [&](CFStringRef key, int value) {
            auto number = CFNumberCreate(nullptr, kCFNumberIntType, &value);
            CFDictionarySetValue(properties, key, number);
            CFRelease(number);
        };
        put(kIOSurfaceWidth, width); put(kIOSurfaceHeight, height);
        put(kIOSurfaceBytesPerElement, 4);
        put(kIOSurfacePixelFormat, 0x42475241); // BGRA
        surface_ = IOSurfaceCreate(properties);
        CFRelease(properties);
        return surface_ != nullptr;
#else
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        window_ = glfwCreateWindow(width, height, "Volume graph compositor", nullptr, nullptr);
        if (!window_) return false;
        glfwMakeContextCurrent(window_);
        std::string error;
        return arbitgl::loadGlFunctions(gl_, error);
#endif
    }
    void close()
    {
#if defined(__APPLE__)
        if (surface_) { CFRelease(surface_); surface_ = nullptr; }
#else
        if (window_) { glfwDestroyWindow(window_); window_ = nullptr; }
#endif
    }
    bool initialize(videorender::FrameRenderer& renderer,
                    const std::vector<videowire::CompiledVisualLayerPlan>& plans, std::string& error)
    {
#if defined(__APPLE__)
        return renderer.initializeForVisualPlans(nullptr, width, height, plans, error, true);
#else
        return renderer.initializeForVisualPlans(&gl_, width, height, plans, error, false);
#endif
    }
    Pixels composite(videorender::FrameRenderer& renderer,
                     const videorender::LayerDesc* layers, int count)
    {
        Pixels pixels;
#if defined(__APPLE__)
        require(renderer.renderCompositeToIOSurface(surface_, width, height, layers, count),
            "Metal compositor: " + renderer.lastError());
        pixels.resize(width * height * 4);
        IOSurfaceLock(surface_, kIOSurfaceLockReadOnly, nullptr);
        const auto* bytes = static_cast<const std::uint8_t*>(IOSurfaceGetBaseAddress(surface_));
        const auto stride = IOSurfaceGetBytesPerRow(surface_);
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
            {
                const auto* bgra = bytes + y * stride + x * 4;
                auto* rgba = pixels.data() + (y * width + x) * 4;
                rgba[0] = bgra[2]; rgba[1] = bgra[1]; rgba[2] = bgra[0]; rgba[3] = bgra[3];
            }
        IOSurfaceUnlock(surface_, kIOSurfaceLockReadOnly, nullptr);
#else
        std::string error;
        require(renderer.renderToPixels(layers, count, pixels, error), "GL compositor: " + error);
#endif
        require(pixels.size() == width * height * 4, "compositor returned the wrong extent");
        return pixels;
    }
    Pixels nativePixels(const NativeVolumeFrame& frame)
    {
        require(frame.colorTextureDescriptor().complete(), "native Volume descriptor is incomplete");
        Pixels pixels(static_cast<std::size_t>(frame.width()) * frame.height() * 4);
#if defined(__APPLE__)
        std::lock_guard<std::mutex> lock(arbitgpu::sokolmetal::mutex());
        const auto info = sg_mtl_query_image_info({static_cast<std::uint32_t>(frame.colorImageHandle())});
        id<MTLTexture> texture = (__bridge id<MTLTexture>) info.tex[info.active_slot];
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>) sg_mtl_command_queue();
        id<MTLDevice> device = (__bridge id<MTLDevice>) arbitgpu::sokolmetal::device();
        const NSUInteger stride = (static_cast<NSUInteger>(frame.width()) * 4u + 255u) & ~NSUInteger(255u);
        id<MTLBuffer> buffer = [device newBufferWithLength:stride * frame.height()
            options:MTLResourceStorageModeShared];
        require(texture != nil && queue != nil && buffer != nil, "Metal readback allocation failed");
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
        [blit copyFromTexture:texture sourceSlice:0 sourceLevel:0
            sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(frame.width(), frame.height(), 1)
            toBuffer:buffer destinationOffset:0 destinationBytesPerRow:stride
            destinationBytesPerImage:stride * frame.height()];
        [blit endEncoding]; [command commit]; [command waitUntilCompleted];
        const bool complete = command.status == MTLCommandBufferStatusCompleted;
        if (complete)
            for (std::uint32_t y = 0; y < frame.height(); ++y)
                for (std::uint32_t x = 0; x < frame.width(); ++x)
                {
                    const auto* bgra = static_cast<const std::uint8_t*>(buffer.contents) + y * stride + x * 4;
                    auto* rgba = pixels.data() + (y * frame.width() + x) * 4;
                    rgba[0] = bgra[2]; rgba[1] = bgra[1]; rgba[2] = bgra[0]; rgba[3] = bgra[3];
                }
#if !__has_feature(objc_arc)
        [buffer release];
#endif
        require(complete, "Metal native Volume readback failed");
#else
        int previous = 0;
        glGetIntegerv(GL_TEXTURE_BINDING_2D, &previous);
        glBindTexture(GL_TEXTURE_2D, static_cast<unsigned>(frame.colorTextureViewHandle()));
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
        glBindTexture(GL_TEXTURE_2D, static_cast<unsigned>(previous));
        require(glGetError() == GL_NO_ERROR, "GL native Volume readback failed");
#endif
        return pixels;
    }
    bool imageAlive(std::uintptr_t image) const
    {
#if defined(__APPLE__)
        std::lock_guard<std::mutex> lock(arbitgpu::sokolmetal::mutex());
        return sg_query_image_state({static_cast<std::uint32_t>(image)}) == SG_RESOURCESTATE_VALID;
#else
        return glIsTexture(static_cast<unsigned>(image)) == GL_TRUE;
#endif
    }
    void rejectOtherContext(VisualVolumeRenderCache& cache, const std::string& payload)
    {
#if !defined(__APPLE__)
        auto& backend = nativeVolumeExecutionBackend();
        NativeVolumeRenderedFrame unchanged;
        unchanged.width = 777;
        std::string error;
        glfwMakeContextCurrent(nullptr);
        const bool withoutContext = cache.render(payload, width, height,
            NativeVolumeRenderUse::Preview, backend, unchanged, error);
        glfwMakeContextCurrent(window_);
        require(!withoutContext && unchanged.width == 777,
            "a cache hit without its context must reject without changing output");
        auto* second = glfwCreateWindow(width, height, "Volume unrelated context", nullptr, nullptr);
        require(second != nullptr, "could not create the unrelated GL context");
        glfwMakeContextCurrent(second);
        const bool otherContext = cache.render(payload, width, height,
            NativeVolumeRenderUse::Preview, backend, unchanged, error);
        glfwDestroyWindow(second);
        glfwMakeContextCurrent(window_);
        require(!otherContext && unchanged.width == 777, "cache reused a texture from an unrelated context");
#else
        (void) cache; (void) payload;
#endif
    }
private:
#if defined(__APPLE__)
    IOSurfaceRef surface_ = nullptr;
#else
    GLFWwindow* window_ = nullptr;
    arbitgl::GlFuncs gl_;
#endif
};

visualvolume::Operation controls()
{
    visualvolume::Operation operation;
    operation.noiseAmount = 0.0f;
    operation.density = 0.2f;
    return operation;
}

videorender::LayerDesc prepare(videorender::FrameRenderer& renderer,
                              visualvolume::Operation operation, NativeVolumeRenderUse use)
{
    const std::vector<videowire::CompiledVisualLayerPlan> plans {volumetest::plan(operation)};
    std::string error;
    videowire::VisualPlanExecutionState state;
    require(state.admitPlans(plans, &error, width, height), "Volume execution admission: " + error);
    videorender::LayerDesc layer;
    layer.clipId = plans.front().clipId;
    // A new generator's legacy flags must be replaced by the native Volume.
    layer.shaderSource = layer.particleSource = layer.isAdjustment = true;
    layer.scale = 0.5f; layer.maskType = 1;
    require(prepareVisualVolumeLayer(plans, layer.clipId, width, height, use,
                renderer.volumeRenderCache(), layer, error) == VisualVolumePreparation::rendered,
        "Volume preparation: " + error);
    require(videowire::executeVisualLayerPlanForRenderer(renderer, plans, layer.clipId, layer, error,
                use == NativeVolumeRenderUse::Export ? videohelper::geometry::PlanUse::exportRender
                                                     : videohelper::geometry::PlanUse::preview,
                nullptr, nullptr, nullptr, &state), "Volume execution: " + error);
    require(layer.nativeTextureOwner && layer.nativeTextureDescriptor.complete()
            && !layer.shaderSource && !layer.particleSource && !layer.isAdjustment
            && layer.scale == 1.0f && layer.maskType == 0,
        "Volume did not replace the generator source and legacy effects");
    return layer;
}

std::shared_ptr<const NativeVolumeFrame> owner(const videorender::LayerDesc& layer)
{
    return std::static_pointer_cast<const NativeVolumeFrame>(layer.nativeTextureOwner);
}

videorender::LayerDesc lowerLayer(videorender::FrameRenderer& renderer)
{
    Pixels pixels(width * height * 4);
    for (std::size_t i = 0; i < pixels.size(); i += 4)
        std::copy(background.begin(), background.end(), pixels.begin() + i);
    videorender::LayerDesc layer;
    layer.clipId = 7; layer.texWidth = width; layer.texHeight = height;
    layer.texture = renderer.uploadRgba(pixels.data(), width, height, width * 4, 0);
    require(layer.texture != 0, "lower clip upload failed");
    return layer;
}

std::size_t occupied(const Pixels& pixels)
{
    std::size_t result = 0;
    for (std::size_t i = 3; i < pixels.size(); i += 4) result += pixels[i] > 3;
    return result;
}

std::uint64_t run(Context& context, std::shared_ptr<const NativeVolumeFrame>& lateOwner, Pixels& reference)
{
    std::string error;
    const auto plan = volumetest::plan(controls());
    videorender::FrameRenderer renderer;
    require(context.initialize(renderer, {plan}, error), "Volume compositor initialization: " + error);
    renderer.setBackgroundColor(0, 0, 0, 1);
    auto volume = prepare(renderer, controls(), NativeVolumeRenderUse::Preview);
    const auto rejectsBinding = [&](videorender::LayerDesc candidate, const char* reason) {
        require(!videowire::executeVisualLayerPlan({plan}, plan.clipId, candidate, error,
                    videohelper::geometry::PlanUse::preview), reason);
    };
    rejectsBinding({}, "an unprepared Volume must not pass execution");
    auto forged = volume; forged.nativeTextureOwner.reset();
    rejectsBinding(forged, "a borrowed texture without its frame owner must reject");
    forged = volume; ++forged.nativeTextureView;
    rejectsBinding(forged, "a forged native texture view must reject");
    forged = volume; ++forged.visualPlanStructuralRevision;
    rejectsBinding(forged, "a native frame from a different structural revision must reject");
    forged = volume; forged.nativeTextureDescriptor.transfer = colortransform::TransferFunction::Unspecified;
    rejectsBinding(forged, "Volume composition requires the exact SDR transfer declaration");
    lateOwner = owner(volume);
    const auto generation = lateOwner->colorTextureDescriptor().rendererGeneration;
    const auto oldImage = lateOwner->colorImageHandle();
    reference = context.nativePixels(*lateOwner);
    const std::size_t centerAlpha = (height / 2 * width + width / 2) * 4 + 3;
    require(reference[centerAlpha] > 10 && reference[centerAlpha] < 240,
        "authored low density must produce partial transmittance");
    // Every sample at density <= 0.2 has blue in [0.52, 0.62]. Its
    // opacity-weighted straight color stays in that interval. Multiplying RGB
    // by coverage in the Volume shader would violate this independent bound.
    require(reference[centerAlpha - 1] >= 131 && reference[centerAlpha - 1] <= 160,
        "low-density Volume RGB was premultiplied instead of straight alpha");
    require(occupied(reference) > 30 && occupied(reference) < width * height / 2,
        "authored sphere must have bounded visible area");

    for (int pausedFrame = 0; pausedFrame < 12; ++pausedFrame)
    {
        const auto repeated = prepare(renderer, controls(), NativeVolumeRenderUse::Preview);
        require(repeated.nativeTextureOwner == volume.nativeTextureOwner
                && repeated.nativeTextureDescriptor.rendererGeneration
                    == volume.nativeTextureDescriptor.rendererGeneration,
            "paused frames must reuse the upload and native render");
    }
    context.rejectOtherContext(renderer.volumeRenderCache(), plan.operations[0].payloadXml);

    auto emptyControls = controls(); emptyControls.density = 0;
    const auto empty = prepare(renderer, emptyControls, NativeVolumeRenderUse::Preview);
    const auto emptyPixels = context.nativePixels(*owner(empty));
    require(std::all_of(emptyPixels.begin(), emptyPixels.end(), [](auto value) { return value == 0; }),
        "density zero must emit transparent black, including RGB");
    auto smallControls = controls(); smallControls.radius = 0.18f;
    const auto small = prepare(renderer, smallControls, NativeVolumeRenderUse::Preview);
    require(occupied(context.nativePixels(*owner(small))) < occupied(reference) / 2,
        "radius edit must shrink the occupied pixel area");
    auto denseControls = controls(); denseControls.density = 0.8f;
    const auto dense = prepare(renderer, denseControls, NativeVolumeRenderUse::Preview);
    require(context.nativePixels(*owner(dense))[centerAlpha] > reference[centerAlpha] + 20,
        "density edit must reduce transmittance");

    auto lower = lowerLayer(renderer);
    std::array<videorender::LayerDesc, 2> layers {lower, empty};
    const auto transparentComposite = context.composite(renderer, layers.data(), 2);
    for (std::size_t i = 0; i < transparentComposite.size(); i += 4)
        require(std::equal(background.begin(), background.end(), transparentComposite.begin() + i),
            "zero-density Volume covered the lower clip");
    layers[1] = volume;
    const auto composite = context.composite(renderer, layers.data(), 2);
    // Independent source-over oracle, evaluated on GPU readback. The color
    // bound above checks that the Volume itself publishes straight RGB.
    for (std::size_t i = 0; i < composite.size(); i += 4)
    {
        const double alpha = reference[i + 3] / 255.0;
        for (int channel = 0; channel < 3; ++channel)
        {
            const int expected = static_cast<int>(std::lround(
                reference[i + channel] * alpha + background[channel] * (1.0 - alpha)));
            require(std::abs(static_cast<int>(composite[i + channel]) - expected) <= 3,
                "native Volume does not compose with straight alpha over the lower clip");
        }
        require(composite[i + 3] == 255, "opaque lower clip must keep the final composite opaque");
    }
    // Exercise the retained secondary LayerDesc used by referenced clip inputs.
    auto borrowed = lower;
    borrowed.clipId = 8; borrowed.transitionType = 2; borrowed.transitionProgress = 0;
    borrowed.fromLayer = &volume;
    layers[1] = borrowed;
    require(context.composite(renderer, layers.data(), 2) == composite,
        "referenced Volume frame lost transparency through the compositor");

    videorender::FrameRenderer exported;
    require(context.initialize(exported, {plan}, error), "export compositor initialization: " + error);
    const auto exportVolume = prepare(exported, controls(), NativeVolumeRenderUse::Export);
    require(exportVolume.nativeTextureOwner != volume.nativeTextureOwner
            && context.nativePixels(*owner(exportVolume)) == reference,
        "independent preview/export native renders differ");
    auto exportLower = lowerLayer(exported);
    std::array<videorender::LayerDesc, 2> exportLayers {exportLower, exportVolume};
    require(context.composite(exported, exportLayers.data(), 2) == composite,
        "preview and export composites differ");
    exported.deleteTexture(exportLower.texture);
    exported.shutdown();
    require(!owner(exportVolume)->colorTextureDescriptor().complete(),
        "export shutdown left a live borrowed native resource");

    NativeVolumeRenderedFrame resized;
    require(renderer.volumeRenderCache().render(plan.operations[0].payloadXml, 64, 32,
                NativeVolumeRenderUse::Preview, nativeVolumeExecutionBackend(), resized, error)
            && resized.width == 64 && resized.height == 32 && resized.nativeFrame != lateOwner,
        "cache must render the requested extent");
    NativeVolumeRenderedFrame unchanged; unchanged.width = 777;
    require(!renderer.volumeRenderCache().render(plan.operations[0].payloadXml, 100000, 100000,
                NativeVolumeRenderUse::Preview, nativeVolumeExecutionBackend(), unchanged, error)
            && unchanged.width == 777, "oversized cache request must reject without replacing output");
    for (std::size_t edit = 0; edit < VisualVolumeRenderCache::maximumEntries + 2; ++edit)
    {
        auto changed = controls(); changed.phase = static_cast<float>(edit) + 0.5f;
        prepare(renderer, changed, NativeVolumeRenderUse::Preview);
    }
    require(renderer.volumeRenderCache().size() == VisualVolumeRenderCache::maximumEntries
            && renderer.volumeRenderCache().retainedBytes() <= VisualVolumeRenderCache::maximumBytes,
        "Volume cache exceeded its entry or byte bound");
    require(context.imageAlive(oldImage), "LRU eviction destroyed an in-flight layer's native frame");
    layers[1] = volume;
    require(context.composite(renderer, layers.data(), 2) == composite,
        "evicted frame owner did not survive composition");
    renderer.deleteTexture(lower.texture);
    renderer.shutdown();
    require(renderer.volumeRenderCache().size() == 0 && renderer.volumeRenderCache().retainedBytes() == 0
            && !context.imageAlive(oldImage) && lateOwner->colorImageHandle() == 0
            && !lateOwner->colorTextureDescriptor().complete(),
        "shutdown must retire evicted and borrowed frames before context destruction");
    return generation;
}
} // namespace

int main()
{
#if !defined(__APPLE__)
    if (!glfwInit()) { std::cerr << "SKIP: GLFW unavailable\n"; return 77; }
#endif
    int result = 0;
    {
        Context context;
        if (!context.open()) result = 77;
        else
        {
            try
            {
                std::shared_ptr<const NativeVolumeFrame> lateOwner;
                Pixels reference;
                const auto priorGeneration = run(context, lateOwner, reference);
                context.close();
                require(context.open(), "replacement native context is unavailable");
                videorender::FrameRenderer replacement;
                std::string error;
                require(context.initialize(replacement, {volumetest::plan(controls())}, error), error);
                const auto restored = prepare(replacement, controls(), NativeVolumeRenderUse::Preview);
                require(owner(restored)->colorTextureDescriptor().rendererGeneration != priorGeneration
                        && context.nativePixels(*owner(restored)) == reference,
                    "reopened context reused a stale Volume resource or changed its pixels");
                lateOwner.reset();
                require(context.imageAlive(owner(restored)->colorImageHandle()),
                    "late release from the old context destroyed a replacement texture");
                replacement.shutdown();
                std::cout << "PASS: Volume graph native transparency, composition, cache and context lifetime\n";
            }
            catch (const std::exception& failure)
            {
                std::cerr << "FAIL: " << failure.what() << '\n';
                result = 1;
            }
        }
    }
#if !defined(__APPLE__)
    glfwTerminate();
#endif
    return result;
}
