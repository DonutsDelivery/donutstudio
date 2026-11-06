// Host side of the Metal device (Plan 3, Render engines and devices). Manual
// retain and release, as the helper's other Metal files.
//
// One device, queue and pipeline per process, compiled on the build gate's
// thread from metalKernelSource() with fast math off, as the Metal backend
// compiles its shaders. Each renderer keeps its own scene buffers, acceleration
// structure and film. Apple silicon shares memory with the GPU, so the buffers
// are shared storage: the host writes and reads their contents directly, and the
// kernel reaches them through their gpuAddress in the launch parameters.

#include "device_metal_host.h"
#include "device_metal_params.h"
#include "device_registry.h"
#include "kernel_build_gate.h"
#include "kernel_compile_state.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace videohelper::pathtrace::device
{
namespace
{

std::string describe (NSError* error)
{
    if (error == nil)
        return "no reason given";
    const char* text = [[error localizedDescription] UTF8String];
    return text != nullptr ? std::string (text) : std::string ("no reason given");
}

struct SharedMetal
{
    std::mutex lock;
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLIntersectionFunctionTable> functions = nil;
    // The kernel was compiled with ARBIT_METAL_MOTION: the GPU has primitive
    // motion blur, and every structure is built with shutter keyframes.
    bool motion = false;
    kernelcompile::BuildGate* gate = new kernelcompile::BuildGate();
};

SharedMetal& sharedMetal()
{
    static auto* shared = new SharedMetal();
    return *shared;
}

void releaseShared (SharedMetal& shared)
{
    [shared.functions release];
    [shared.pipeline release];
    [shared.queue release];
    [shared.device release];
    shared.functions = nil;
    shared.pipeline = nil;
    shared.queue = nil;
    shared.device = nil;
}

std::string deviceProblem (id<MTLDevice> device)
{
    if (device == nil)
        return "this Mac has no Metal GPU";
    if (! [device supportsRaytracing])
        return std::string ([[device name] UTF8String]) + " has no Metal ray tracing";
    return {};
}

// The device, the kernel and its intersection function, made once. A failure
// releases what it made, and the next try starts from nothing.
bool buildPipeline (SharedMetal& shared, std::string& error)
{
    const std::lock_guard<std::mutex> lock (shared.lock);
    if (shared.pipeline != nil)
        return true;
    @autoreleasepool
    {
        shared.device = MTLCreateSystemDefaultDevice();
        if (const auto problem = deviceProblem (shared.device); ! problem.empty())
        {
            error = "Metal device: " + problem;
            releaseShared (shared);
            return false;
        }
        const std::string source = metalKernelSource();
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.fastMathEnabled = NO;
        options.languageVersion = MTLLanguageVersion3_0;
        shared.motion = [shared.device supportsPrimitiveMotionBlur];
        if (shared.motion)
            options.preprocessorMacros = @{ @"ARBIT_METAL_MOTION" : @"1" };
        NSError* failure = nil;
        id<MTLLibrary> library = [shared.device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                                             options:options
                                                               error:&failure];
        [options release];
        if (library == nil)
        {
            error = "Metal device: the kernel did not compile: " + describe (failure);
            releaseShared (shared);
            return false;
        }
        id<MTLFunction> kernel = [library newFunctionWithName:[NSString stringWithUTF8String:metal::kKernelName]];
        id<MTLFunction> skip = [library newFunctionWithName:[NSString stringWithUTF8String:metal::kSkipTriangleName]];
        [library release];
        if (kernel == nil || skip == nil)
        {
            error = "Metal device: the kernel source has no " + std::string (kernel == nil ? metal::kKernelName : metal::kSkipTriangleName);
            [kernel release];
            [skip release];
            releaseShared (shared);
            return false;
        }
        MTLLinkedFunctions* linked = [[MTLLinkedFunctions alloc] init];
        linked.functions = @[ skip ];
        MTLComputePipelineDescriptor* descriptor = [[MTLComputePipelineDescriptor alloc] init];
        descriptor.computeFunction = kernel;
        descriptor.linkedFunctions = linked;
        shared.pipeline = [shared.device newComputePipelineStateWithDescriptor:descriptor
                                                                       options:MTLPipelineOptionNone
                                                                    reflection:nil
                                                                         error:&failure];
        [descriptor release];
        [linked release];
        [kernel release];
        if (shared.pipeline == nil)
        {
            [skip release];
            error = "Metal device: the kernel did not link: " + describe (failure);
            releaseShared (shared);
            return false;
        }
        MTLIntersectionFunctionTableDescriptor* tableDescriptor = [[MTLIntersectionFunctionTableDescriptor alloc] init];
        tableDescriptor.functionCount = 1;
        shared.functions = [shared.pipeline newIntersectionFunctionTableWithDescriptor:tableDescriptor];
        [tableDescriptor release];
        id<MTLFunctionHandle> handle = [shared.pipeline functionHandleWithFunction:skip];
        [skip release];
        if (shared.functions == nil || handle == nil)
        {
            error = "Metal device: the intersection function table could not be made";
            releaseShared (shared);
            return false;
        }
        [shared.functions setFunction:handle atIndex:0];
        shared.queue = [shared.device newCommandQueue];
        if (shared.queue == nil)
        {
            error = "Metal device: the command queue could not be made";
            releaseShared (shared);
            return false;
        }
    }
    return true;
}

std::function<bool (std::string&)> buildBody()
{
    return [] (std::string& error) { return buildPipeline (sharedMetal(), error); };
}

class MetalRenderer
{
public:
    MetalRenderer (SharedMetal& shared, std::function<std::uint64_t()> sceneBuild)
        : shared_ (shared), sceneBuild_ (std::move (sceneBuild)) {}

    ~MetalRenderer()
    {
        releaseScene();
        releaseFilm();
    }

    bool render (PathTracer::DevicePass& pass)
    {
        const auto waitStart = PhaseClock::now();
        if (deviceLost())
            return lostError (pass.error);
        {
            std::optional<kernelcompile::Scope> compiling;
            if (shared_.gate->status() != kernelcompile::BuildGate::Status::Ready)
                compiling.emplace();
            ++kernelcompile::exportsWaiting();
            const bool built = shared_.gate->join (buildBody(), pass.error);
            --kernelcompile::exportsWaiting();
            if (! built)
                return false;
        }
        const std::lock_guard<std::mutex> lock (shared_.lock);
        pass.stats.phases.set (FramePhase::PipelineWait, millisecondsSince (waitStart));
        @autoreleasepool
        {
            return ensureScene (pass, pass.error) && ensureFilm (pass.film.width(), pass.film.height(), pass.error) && launch (pass);
        }
    }

private:
    bool lostError (std::string& error) const
    {
        error = "Metal device: the GPU stopped responding earlier in this session";
        return false;
    }

    id<MTLBuffer> makeBuffer (const void* data, std::size_t size, std::vector<id<MTLBuffer>>& owner, std::string& error)
    {
        // Metal makes no zero-length buffer; an empty view keeps a null address.
        if (size == 0)
            return nil;
        id<MTLBuffer> buffer = data != nullptr
                                 ? [shared_.device newBufferWithBytes:data length:size options:MTLResourceStorageModeShared]
                                 : [shared_.device newBufferWithLength:size options:MTLResourceStorageModeShared];
        if (buffer == nil)
            error = "Metal device: the GPU has no room for " + std::to_string (size) + " bytes";
        else
            owner.push_back (buffer);
        return buffer;
    }

    template <typename T>
    bool uploadView (BufferView<T> source, BufferView<T>& destination, std::string& error)
    {
        static_assert (std::is_trivially_copyable_v<T>, "a device buffer is copied as bytes");
        destination = {};
        if (source.count == 0)
            return true;
        id<MTLBuffer> buffer = makeBuffer (source.data, source.count * sizeof (T), scene_, error);
        if (buffer == nil)
            return false;
        destination = { reinterpret_cast<const T*> ([buffer gpuAddress]), source.count };
        return true;
    }

    // The kernel reads double as float (device_metal.metal), so a double buffer
    // goes up as floats behind the same view.
    bool uploadView (BufferView<double> source, BufferView<double>& destination, std::string& error)
    {
        destination = {};
        if (source.count == 0)
            return true;
        std::vector<float> narrowed (source.count);
        for (std::size_t i = 0; i < source.count; ++i)
            narrowed[i] = static_cast<float> (source.data[i]);
        id<MTLBuffer> buffer = makeBuffer (narrowed.data(), narrowed.size() * sizeof (float), scene_, error);
        if (buffer == nil)
            return false;
        destination = { reinterpret_cast<const double*> ([buffer gpuAddress]), source.count };
        return true;
    }

    template <typename T>
    bool uploadOne (const T& value, const T*& destination, std::string& error)
    {
        id<MTLBuffer> buffer = makeBuffer (&value, sizeof (T), scene_, error);
        if (buffer == nil)
            return false;
        destination = reinterpret_cast<const T*> ([buffer gpuAddress]);
        return true;
    }

    void releaseScene()
    {
        for (id<MTLBuffer> buffer : scene_)
            [buffer release];
        scene_.clear();
        [structure_ release];
        structure_ = nil;
        haveScene_ = false;
    }

    void releaseFilm()
    {
        for (id<MTLBuffer> buffer : film_)
            [buffer release];
        film_.clear();
        filmWidth_ = filmHeight_ = 0;
    }

    bool ensureScene (const PathTracer::DevicePass& pass, std::string& error)
    {
        const std::uint64_t build = sceneBuild_();
        auto& phases = pass.stats.phases;
        if (haveScene_ && build == sceneUploaded_ && &pass.scene == sceneAddress_)
        {
            phases.set (FramePhase::Upload, 0.0);
            phases.set (FramePhase::AccelBuild, 0.0);
            phases.accelBuild = "reused";
            return true;
        }
        // Shared storage: an upload is the copy into a new buffer.
        const auto uploadStart = PhaseClock::now();
        releaseScene();
        const RenderScene& scene = pass.scene;
        view_ = {};
        if (! uploadView (BufferView<Vec3> { scene.geometry.positions.data(), scene.geometry.positions.size() }, view_.positions, error)
            || ! uploadView (BufferView<TriangleAttributes> { scene.geometry.triangles.data(), scene.geometry.triangles.size() }, view_.triangles, error)
            || ! uploadView (BufferView<Material> { scene.materials.data(), scene.materials.size() }, view_.materials, error)
            || ! uploadView (BufferView<GratingBinding> { scene.gratings.data(), scene.gratings.size() }, view_.gratings, error)
            || ! uploadView (BufferView<Vec3> { scene.geometry.motionOpen.data(), scene.geometry.motionOpen.size() }, view_.motionOpen, error)
            || ! uploadView (BufferView<Vec3> { scene.geometry.motionClose.data(), scene.geometry.motionClose.size() }, view_.motionClose, error)
            || ! uploadOne (scene.fallbackMaterial, view_.fallbackMaterial, error))
            return false;
        if (scene.hasWorldVolume && ! uploadOne (scene.worldVolume, view_.worldMedium, error))
            return false;
        view_.worldVolumeRadius = scene.worldVolumeExtent;
        view_.anyVolume = scene.anyVolume;
        view_.anyHiddenObjects = scene.anyHiddenObjects;

        bool uploaded = true;
        const LightSetView lights = scene.lights.deviceCopy ([&] (auto source) {
            decltype (source) copy {};
            uploaded = uploadView (source, copy, error) && uploaded;
            return copy;
        });
        if (! uploaded || ! uploadOne (lights, view_.lights, error))
            return false;

        view_.anyPartialCoverage = scene.anyPartialCoverage;
        view_.anyThinSheet = scene.anyThinSheet;
        view_.anyDispersive = scene.anyDispersive;
        view_.anyMediumBoundary = scene.anyMediumBoundary;
        view_.boundsCentre = scene.boundsCentre;
        view_.boundsRadius = scene.boundsRadius;
        phases.set (FramePhase::Upload, millisecondsSince (uploadStart));
        // buildStructure waits for its command buffer, so this is the whole build.
        const auto accelStart = PhaseClock::now();
        const bool built = buildStructure (scene, error);
        phases.set (FramePhase::AccelBuild, millisecondsSince (accelStart));
        phases.accelBuild = "build";
        if (! built)
            return false;
        haveScene_ = true;
        sceneUploaded_ = build;
        sceneAddress_ = &pass.scene;
        return true;
    }

    // A primitive acceleration structure over the scene's triangles, three
    // vertices each, in scene order, so a primitive id is the triangle index. The
    // geometry is not opaque, so the skip function sees every candidate hit. An
    // empty scene gets one zero-area triangle, which no ray hits. With motion
    // blur compiled in, the triangles have two keyframes, shutter open and
    // close, the same vertices for a still scene.
    bool buildStructure (const RenderScene& scene, std::string& error)
    {
        static_assert (sizeof (Vec3) == 3 * sizeof (float), "Metal reads the positions as packed float3");
        const Vec3 empty[3] {};
        const bool none = scene.geometry.positions.size() < 3;
        const bool moving = ! none && ! scene.geometry.motionOpen.empty();
        if (moving && ! shared_.motion)
        {
            error = "Metal device: " + std::string ([[shared_.device name] UTF8String])
                  + " has no primitive motion blur, which this scene's moving objects need";
            return false;
        }
        const Vec3* positions = none ? empty : moving ? scene.geometry.motionOpen.data() : scene.geometry.positions.data();
        const std::size_t triangles = none ? 1 : scene.geometry.positions.size() / 3;
        id<MTLBuffer> vertices = makeBuffer (positions, triangles * 3 * sizeof (Vec3), scene_, error);
        if (vertices == nil)
            return false;
        id<MTLBuffer> closeVertices = vertices;
        if (moving)
        {
            closeVertices = makeBuffer (scene.geometry.motionClose.data(), triangles * 3 * sizeof (Vec3), scene_, error);
            if (closeVertices == nil)
                return false;
        }

        MTLPrimitiveAccelerationStructureDescriptor* descriptor = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
        if (shared_.motion)
        {
            MTLAccelerationStructureMotionTriangleGeometryDescriptor* geometry =
                [MTLAccelerationStructureMotionTriangleGeometryDescriptor descriptor];
            MTLMotionKeyframeData* open = [MTLMotionKeyframeData data];
            open.buffer = vertices;
            MTLMotionKeyframeData* close = [MTLMotionKeyframeData data];
            close.buffer = closeVertices;
            geometry.vertexBuffers = @[ open, close ];
            geometry.vertexStride = sizeof (Vec3);
            geometry.vertexFormat = MTLAttributeFormatFloat3;
            geometry.triangleCount = triangles;
            geometry.opaque = NO;
            geometry.intersectionFunctionTableOffset = 0;
            descriptor.geometryDescriptors = @[ geometry ];
            descriptor.motionStartTime = 0.0f;
            descriptor.motionEndTime = 1.0f;
            descriptor.motionKeyframeCount = 2;
        }
        else
        {
            MTLAccelerationStructureTriangleGeometryDescriptor* geometry = [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
            geometry.vertexBuffer = vertices;
            geometry.vertexStride = sizeof (Vec3);
            geometry.vertexFormat = MTLAttributeFormatFloat3;
            geometry.triangleCount = triangles;
            geometry.opaque = NO;
            geometry.intersectionFunctionTableOffset = 0;
            descriptor.geometryDescriptors = @[ geometry ];
        }

        const MTLAccelerationStructureSizes sizes = [shared_.device accelerationStructureSizesWithDescriptor:descriptor];
        structure_ = [shared_.device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
        id<MTLBuffer> scratch = [shared_.device newBufferWithLength:std::max<NSUInteger> (sizes.buildScratchBufferSize, 1)
                                                            options:MTLResourceStorageModePrivate];
        if (structure_ == nil || scratch == nil)
        {
            [scratch release];
            error = "Metal device: the GPU has no room for the acceleration structure ("
                  + std::to_string (sizes.accelerationStructureSize) + " bytes)";
            return false;
        }
        id<MTLCommandBuffer> commands = [shared_.queue commandBuffer];
        id<MTLAccelerationStructureCommandEncoder> encoder = [commands accelerationStructureCommandEncoder];
        [encoder buildAccelerationStructure:structure_ descriptor:descriptor scratchBuffer:scratch scratchBufferOffset:0];
        [encoder endEncoding];
        [commands commit];
        [commands waitUntilCompleted];
        [scratch release];
        return finished (commands, "building the acceleration structure", error);
    }

    bool finished (id<MTLCommandBuffer> commands, const char* what, std::string& error)
    {
        if ([commands status] != MTLCommandBufferStatusError)
            return true;
        NSError* failure = [commands error];
        if (failure != nil && [failure code] == MTLCommandBufferErrorDeviceRemoved)
            noteDeviceLost();
        error = std::string ("Metal device: ") + what + " failed: " + describe (failure);
        return false;
    }

    bool ensureFilm (std::uint32_t width, std::uint32_t height, std::string& error)
    {
        if (filmWidth_ == width && filmHeight_ == height)
            return true;
        releaseFilm();
        const std::size_t bytes = static_cast<std::size_t> (width) * height * sizeof (gpu::FilmPixel);
        pixels_ = makeBuffer (nullptr, bytes, film_, error);
        splats_ = makeBuffer (nullptr, bytes, film_, error);
        guideAlbedo_ = makeBuffer (nullptr, bytes, film_, error);
        guideNormal_ = makeBuffer (nullptr, bytes, film_, error);
        lightPaths_ = makeBuffer (nullptr, sizeof (unsigned long long), film_, error);
        totals_ = makeBuffer (nullptr, sizeof (gpu::LaunchTotals), film_, error);
        if (! error.empty())
        {
            releaseFilm();
            return false;
        }
        filmWidth_ = width;
        filmHeight_ = height;
        return true;
    }

    bool launch (PathTracer::DevicePass& pass)
    {
        std::string& error = pass.error;
        const std::uint32_t width = pass.film.width();
        const std::uint32_t height = pass.film.height();
        const auto traceStart = PhaseClock::now();
        for (id<MTLBuffer> buffer : film_)
            std::memset ([buffer contents], 0, [buffer length]);

        metal::LaunchParams params;
        params.view = view_;
        params.settings = pass.settings;
        params.camera = pass.camera;
        params.film = { reinterpret_cast<gpu::FilmPixel*> ([pixels_ gpuAddress]), reinterpret_cast<gpu::FilmPixel*> ([splats_ gpuAddress]),
                        reinterpret_cast<unsigned long long*> ([lightPaths_ gpuAddress]),
                        reinterpret_cast<gpu::FilmPixel*> ([guideAlbedo_ gpuAddress]),
                        reinterpret_cast<gpu::FilmPixel*> ([guideNormal_ gpuAddress]), width };
        params.totals = reinterpret_cast<gpu::LaunchTotals*> ([totals_ gpuAddress]);
        params.height = height;
        params.samples = 1;
        params.generation = pass.generation;

        // A command buffer per sample, so a cancel stops the frame between samples.
        // The queue runs them in order; the last one's end means all have ended.
        id<MTLCommandBuffer> last = nil;
        for (std::uint32_t s = 0; s < pass.samples; ++s)
        {
            if (pass.cancel.isCancelled())
            {
                pass.stats.cancelled = true;
                break;
            }
            params.firstSample = pass.firstSample + s;
            id<MTLCommandBuffer> commands = [shared_.queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
            [encoder setComputePipelineState:shared_.pipeline];
            id<MTLBuffer> parameters = [shared_.device newBufferWithBytes:&params length:sizeof (params)
                                                                  options:MTLResourceStorageModeShared];
            if (parameters == nil)
            {
                [encoder endEncoding];
                error = "Metal device: the GPU has no room for the launch parameters";
                return false;
            }
            [encoder setBuffer:parameters offset:0 atIndex:metal::kParamsIndex];
            [parameters release];
            [encoder setAccelerationStructure:structure_ atBufferIndex:metal::kAccelIndex];
            [encoder setIntersectionFunctionTable:shared_.functions atBufferIndex:metal::kFunctionTableIndex];
            // The kernel reaches these through addresses, not bindings.
            for (id<MTLBuffer> buffer : scene_)
                [encoder useResource:buffer usage:MTLResourceUsageRead];
            for (id<MTLBuffer> buffer : film_)
                [encoder useResource:buffer usage:MTLResourceUsageRead | MTLResourceUsageWrite];
            [encoder dispatchThreads:MTLSizeMake (width, height, 1)
               threadsPerThreadgroup:MTLSizeMake (metal::kGroupSide, metal::kGroupSide, 1)];
            [encoder endEncoding];
            [commands commit];
            last = commands;
        }
        if (last == nil)
        {
            pass.stats.phases.set (FramePhase::Trace, millisecondsSince (traceStart));
            pass.stats.phases.set (FramePhase::Readback, 0.0);
            return true;
        }
        [last waitUntilCompleted];
        pass.stats.phases.set (FramePhase::Trace, millisecondsSince (traceStart));
        if (! finished (last, "running the kernel", error))
            return false;
        // The film is in shared memory, so this is the host's read of it.
        const auto readbackStart = PhaseClock::now();
        readBack (pass);
        pass.stats.phases.set (FramePhase::Readback, millisecondsSince (readbackStart));
        return true;
    }

    void readBack (PathTracer::DevicePass& pass)
    {
        const std::uint32_t width = pass.film.width();
        const std::size_t count = static_cast<std::size_t> (width) * pass.film.height();
        const auto* pixels = static_cast<const gpu::FilmPixel*> ([pixels_ contents]);
        const auto* splats = static_cast<const gpu::FilmPixel*> ([splats_ contents]);
        const auto* albedos = static_cast<const gpu::FilmPixel*> ([guideAlbedo_ contents]);
        const auto* normals = static_cast<const gpu::FilmPixel*> ([guideNormal_ contents]);
        for (std::size_t i = 0; i < count; ++i)
        {
            const auto x = static_cast<std::uint32_t> (i % width);
            const auto y = static_cast<std::uint32_t> (i / width);
            const gpu::FilmPixel& pixel = pixels[i];
            if (pixel.w > 0.0f)
                pass.film.addSampleSum (x, y, { pixel.x, pixel.y, pixel.z }, static_cast<std::uint32_t> (pixel.w));
            const gpu::FilmPixel& splat = splats[i];
            if (splat.x != 0.0f || splat.y != 0.0f || splat.z != 0.0f)
                pass.film.addSplat (x, y, { splat.x, splat.y, splat.z });
            const gpu::FilmPixel& albedo = albedos[i];
            const gpu::FilmPixel& normal = normals[i];
            if (albedo.w > 0.0f)
                pass.film.addGuideSum (x, y, { albedo.x, albedo.y, albedo.z }, { normal.x, normal.y, normal.z },
                                       static_cast<std::uint32_t> (albedo.w));
        }
        unsigned long long lightPaths = 0;
        std::memcpy (&lightPaths, [lightPaths_ contents], sizeof (lightPaths));
        pass.film.addLightPaths (lightPaths);
        gpu::LaunchTotals totals;
        std::memcpy (&totals, [totals_ contents], sizeof (totals));
        PathCounters counters;
        std::memcpy (&counters, totals.counters, sizeof (counters));
        pass.counters += counters;
        pass.stats.samplesAdded += totals.samplesAdded;
        pass.stats.nonFiniteSamples += totals.nonFinite;
        // Apple silicon has no separate GPU memory: the readout is the share Metal
        // recommends for this process and what it holds.
        const std::uint64_t total = [shared_.device recommendedMaxWorkingSetSize];
        const std::uint64_t used = [shared_.device currentAllocatedSize];
        pass.stats.deviceMemoryTotalBytes = total;
        pass.stats.deviceMemoryFreeBytes = total > used ? total - used : 0;
    }

    SharedMetal& shared_;
    std::function<std::uint64_t()> sceneBuild_;
    std::vector<id<MTLBuffer>> scene_;
    std::vector<id<MTLBuffer>> film_;
    id<MTLAccelerationStructure> structure_ = nil;
    SceneView<metal::AccelSlot> view_;
    bool haveScene_ = false;
    std::uint64_t sceneUploaded_ = 0;
    const RenderScene* sceneAddress_ = nullptr;
    std::uint32_t filmWidth_ = 0;
    std::uint32_t filmHeight_ = 0;
    id<MTLBuffer> pixels_ = nil, splats_ = nil, guideAlbedo_ = nil, guideNormal_ = nil, lightPaths_ = nil, totals_ = nil;
};

} // namespace

PathTracer::DevicePassFunction makeMetalDevicePass (std::function<std::uint64_t()> sceneBuild)
{
    auto renderer = std::make_shared<MetalRenderer> (sharedMetal(), std::move (sceneBuild));
    return [renderer] (PathTracer::DevicePass& pass) { return renderer->render (pass); };
}

bool metalPipelineReady()
{
    if (deviceLost())
        return true;
    auto& gate = *sharedMetal().gate;
    if (gate.stopped())
    {
        kernelcompile::previewWaiting() = false;
        kernelcompile::previewStopped() = true;
        return false;
    }
    kernelcompile::previewStopped() = false;
    gate.start (buildBody());
    if (gate.status() != kernelcompile::BuildGate::Status::Running)
    {
        kernelcompile::previewWaiting() = false;
        return true;
    }
    if (gate.runningForMs() >= 250.0)
        kernelcompile::previewWaiting() = true;
    return false;
}

void startMetalPipelineBuild()
{
    sharedMetal().gate->start (buildBody());
}

bool stopMetalPipelineBuildByUser()
{
    if (! sharedMetal().gate->stop())
        return false;
    kernelcompile::previewWaiting() = false;
    kernelcompile::previewStopped() = true;
    return true;
}

void startMetalPipelineBuildByUser()
{
    auto& gate = *sharedMetal().gate;
    gate.resume();
    kernelcompile::previewStopped() = false;
    gate.start (buildBody());
}

void cancelMetalPipelineBuild()
{
    auto& gate = *sharedMetal().gate;
    std::string ignored;
    if (gate.status() == kernelcompile::BuildGate::Status::Running)
        gate.await (ignored);
}

std::string metalDeviceProblem()
{
    static const std::string problem = []
    {
        @autoreleasepool
        {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            std::string result = deviceProblem (device);
            [device release];
            return result;
        }
    }();
    return problem;
}

} // namespace videohelper::pathtrace::device
