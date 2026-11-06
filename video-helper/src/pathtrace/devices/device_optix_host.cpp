// Host side of the OptiX device. Compiled only when configure finds the SDK
// (cmake/OptiXDiscovery.cmake). The SDK is headers alone: optixInit finds the
// driver's library at run time, and the CUDA calls below go through dlopen, so
// nothing here links against NVIDIA code and the helper still starts without a
// driver.

#define OPTIX_DONT_INCLUDE_CUDA 1
typedef struct CUctx_st* CUcontext;
typedef struct CUstream_st* CUstream;
typedef unsigned long long CUdeviceptr;

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include "device_optix_host.h"
#include "device_registry.h"
#include "setup_unwind.h"
#include "kernel_build_gate.h"
#include "kernel_compile_state.h"
#include "device_optix_params.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

// The kernel's PTX, generated at build time (cmake/OptiXEmbedKernel.cmake).
#if defined(ARBIT_HAVE_OPTIX_KERNEL)
extern const char* const kArbitOptixKernelPtx[];
extern const std::size_t kArbitOptixKernelPtxChunks;
#endif

namespace videohelper::pathtrace::device
{

std::string optixInitProblem();

// The driver holds the function table for the ABI these SDK headers name, and
// refuses when it is older than the SDK.
std::string optixInitProblem()
{
    const OptixResult result = optixInit();
    if (result == OPTIX_SUCCESS)
        return {};
    if (result == OPTIX_ERROR_UNSUPPORTED_ABI_VERSION)
        return "the NVIDIA driver is older than the OptiX SDK this build used (ABI " + std::to_string (OPTIX_ABI_VERSION) + ")";
    return "optixInit failed with OptiX error " + std::to_string (static_cast<int> (result));
}

#if defined(ARBIT_HAVE_OPTIX_KERNEL)
namespace
{

// The few CUDA driver calls OptiX needs around it, resolved from libcuda at run time.
struct CudaApi
{
    int (*cuInit) (unsigned int) = nullptr;
    int (*cuDeviceGet) (int*, int) = nullptr;
    int (*cuCtxCreate) (CUcontext*, unsigned int, int) = nullptr;
    int (*cuCtxSetCurrent) (CUcontext) = nullptr;
    int (*cuCtxDestroy) (CUcontext) = nullptr;
    int (*cuMemAlloc) (CUdeviceptr*, std::size_t) = nullptr;
    int (*cuMemFree) (CUdeviceptr) = nullptr;
    int (*cuMemcpyHtoD) (CUdeviceptr, const void*, std::size_t) = nullptr;
    int (*cuMemcpyDtoH) (void*, CUdeviceptr, std::size_t) = nullptr;
    int (*cuMemsetD8) (CUdeviceptr, unsigned char, std::size_t) = nullptr;
    int (*cuMemcpyDtoD) (CUdeviceptr, CUdeviceptr, std::size_t) = nullptr;
    int (*cuCtxSetLimit) (int, std::size_t) = nullptr;
    // Optional: a driver without it leaves the memory readout empty.
    int (*cuMemGetInfo) (std::size_t*, std::size_t*) = nullptr;

    bool load (std::string& error)
    {
#if defined(_WIN32)
        const auto library = LoadLibraryA ("nvcuda.dll");
        const auto find = [library] (const char* name) { return reinterpret_cast<void*> (GetProcAddress (library, name)); };
#else
        void* const library = dlopen ("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        const auto find = [library] (const char* name) { return dlsym (library, name); };
#endif
        if (library == nullptr)
        {
            error = "the CUDA driver library did not load";
            return false;
        }
        const auto bind = [&] (auto& function, const char* name) {
            function = reinterpret_cast<std::remove_reference_t<decltype (function)>> (find (name));
            if (function == nullptr)
                error = std::string ("the CUDA driver library has no ") + name;
        };
        bind (cuInit, "cuInit");
        bind (cuDeviceGet, "cuDeviceGet");
        bind (cuCtxCreate, "cuCtxCreate_v2");
        bind (cuCtxSetCurrent, "cuCtxSetCurrent");
        bind (cuCtxDestroy, "cuCtxDestroy_v2");
        bind (cuMemAlloc, "cuMemAlloc_v2");
        bind (cuMemFree, "cuMemFree_v2");
        bind (cuMemcpyHtoD, "cuMemcpyHtoD_v2");
        bind (cuMemcpyDtoH, "cuMemcpyDtoH_v2");
        bind (cuMemsetD8, "cuMemsetD8_v2");
        bind (cuMemcpyDtoD, "cuMemcpyDtoD_v2");
        bind (cuCtxSetLimit, "cuCtxSetLimit");
        cuMemGetInfo = reinterpret_cast<decltype (cuMemGetInfo)> (find ("cuMemGetInfo_v2"));
        return error.empty();
    }
};

struct EmptyRecord
{
    alignas (OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE] {};
};

// The OptiX AI denoiser (Plan 3, Denoising), one per model kind, on the shared
// context. Each keeps its state, scratch and images for the last frame size.
// Outside OptixRenderer: its inline static array needs the type complete.
struct DenoiserSlot
{
    OptixDenoiser denoiser = nullptr;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    OptixDenoiserSizes sizes {};
    std::vector<CUdeviceptr> memory;
    CUdeviceptr state = 0, scratch = 0, color = 0, albedo = 0, normal = 0, output = 0, flow = 0;
    CUdeviceptr guides[2] {};
    int currentGuide = 0;
    bool hasPrevious = false;
};

class OptixRenderer
{
public:
    explicit OptixRenderer (std::function<std::uint64_t()> sceneBuild) : sceneBuild_ (std::move (sceneBuild)) {}

    // An engine's scene and film go with it. The pipeline stays for the next engine.
    ~OptixRenderer()
    {
        // No scene or film reached the device without a pipeline, and the build's thread
        // owns the context until it ends.
        if (buildGate().status() != kernelcompile::BuildGate::Status::Ready)
            return;
        const std::lock_guard<std::mutex> lock (deviceLock_);
        if (cudaContext_ == nullptr || cu_.cuCtxSetCurrent (cudaContext_) != 0)
            return;
        freeScene();
        for (const CUdeviceptr address : film_)
            cu_.cuMemFree (address);
    }

    bool render (PathTracer::DevicePass& pass)
    {
        const auto waitStart = PhaseClock::now();
        if (deviceLost())
        {
            pass.error = "OptiX device: the GPU context was lost earlier in this session";
            return false;
        }
        {
            // An export comes here during a build, and after a Cancel it builds anyway. A
            // preview comes after pipelineReady.
            std::optional<kernelcompile::Scope> compiling;
            if (buildGate().status() != kernelcompile::BuildGate::Status::Ready)
            {
                compiling.emplace();
                allowCompile();
            }
            ++kernelcompile::exportsWaiting();
            const bool built = buildGate().join (buildBody(), pass.error);
            --kernelcompile::exportsWaiting();
            if (! built)
                return false;
        }
        const std::lock_guard<std::mutex> lock (deviceLock_);
        pass.stats.phases.set (FramePhase::PipelineWait, millisecondsSince (waitStart));
        if (deviceLost())
        {
            pass.error = "OptiX device: the GPU context was lost earlier in this session";
            return false;
        }
        return enter (pass.error) && ensureScene (pass, pass.error) && launch (pass);
    }

    // The kernel's cache hit takes about 0.1 s, too short to announce.
    static constexpr double kAnnounceBuildAfterMs = 250.0;

    static bool pipelineReady()
    {
        if (deviceLost())
            return true;
        auto& gate = buildGate();
        // Cancel leaves the preview waiting by name. It never falls to the CPU for this.
        if (gate.stopped())
        {
            kernelcompile::previewWaiting() = false;
            kernelcompile::previewStopped() = true;
            return false;
        }
        kernelcompile::previewStopped() = false;
        startBuild();
        if (gate.status() != kernelcompile::BuildGate::Status::Running)
        {
            kernelcompile::previewWaiting() = false;
            return true;
        }
        if (gate.runningForMs() >= kAnnounceBuildAfterMs)
            kernelcompile::previewWaiting() = true;
        return false;
    }

    static std::function<bool (std::string&)> buildBody()
    {
        return [] (std::string& error)
        {
            // No renderer outlives this for certain, and setup uses only the shared objects.
            OptixRenderer builder ({});
            return builder.ensurePipeline (error);
        };
    }

    static void startBuild() { buildGate().start (buildBody()); }

    // The compile stops about 2 s after this (measured), on its own thread. This doesn't wait.
    static bool stopBuildByUser()
    {
        if (! buildGate().stop())
            return false;
        cancelCompile();
        kernelcompile::previewWaiting() = false;
        kernelcompile::previewStopped() = true;
        return true;
    }

    static void startBuildByUser()
    {
        allowCompile();
        buildGate().resume();
        kernelcompile::previewStopped() = false;
        startBuild();
    }

    static void cancelBuild()
    {
        cancelCompile();
        std::string ignored;
        if (buildGate().status() == kernelcompile::BuildGate::Status::Running)
            buildGate().await (ignored);
    }

private:
    static void cancelCompile()
    {
        const std::lock_guard<std::mutex> lock (creatingLock_);
        cancelled_ = true;
        if (creating_ != nullptr)
            optixModuleCancelCreation (creating_, OPTIX_CREATION_FLAG_NONE);
    }

    // Lets the next compile run. One still ending from a Cancel has already seen that Cancel.
    static void allowCompile()
    {
        const std::lock_guard<std::mutex> lock (creatingLock_);
        cancelled_ = false;
    }

    static kernelcompile::BuildGate& buildGate()
    {
        static auto* gate = new kernelcompile::BuildGate();
        return *gate;
    }

    // One task holds nearly all the work, so this thread does what a pool would. Tasks
    // leave the module cancellable.
    bool compileModule (const OptixModuleCompileOptions& moduleOptions, const OptixPipelineCompileOptions& pipelineOptions,
                        const std::string& ptx, std::string& error)
    {
        logSize_ = sizeof (log_);
        OptixTask first = nullptr;
        if (! check (optixModuleCreateWithTasks (context_, &moduleOptions, &pipelineOptions, ptx.data(), ptx.size(),
                                                 log_, &logSize_, &module_, &first), "compiling the kernel", error))
            return false;
        {
            const std::lock_guard<std::mutex> lock (creatingLock_);
            creating_ = module_;
            if (cancelled_)
                optixModuleCancelCreation (creating_, OPTIX_CREATION_FLAG_NONE);
        }
        std::vector<OptixTask> tasks { first };
        OptixResult result = OPTIX_SUCCESS;
        while (! tasks.empty() && result == OPTIX_SUCCESS)
        {
            OptixTask more[16] {};
            unsigned int added = 0;
            const OptixTask task = tasks.back();
            tasks.pop_back();
            result = optixTaskExecute (task, more, 16, &added);
            tasks.insert (tasks.end(), more, more + added);
        }
        {
            const std::lock_guard<std::mutex> lock (creatingLock_);
            creating_ = nullptr;
        }
        return check (result, "compiling the kernel", error);
    }

    static bool cuda (int result, const char* what, std::string& error)
    {
        if (result == 0)
            return true;
        error = std::string ("OptiX device: ") + what + " failed with CUDA error " + std::to_string (result);
        if (cudaErrorLosesDevice (result))
            noteDeviceLost();
        return false;
    }

    static bool check (OptixResult result, const char* what, std::string& error)
    {
        if (result == OPTIX_SUCCESS)
            return true;
        error = std::string ("OptiX device: ") + what + " failed: " + optixGetErrorName (result);
        if (result == OPTIX_ERROR_LAUNCH_FAILURE)
            noteDeviceLost();
        if (logSize_ > 1 && result != OPTIX_SUCCESS)
            error += std::string (" ") + log_;
        return false;
    }

    // Every OptiX and CUDA call needs the context current on the calling thread,
    // and render threads change.
    static bool enter (std::string& error) { return cuda (cu_.cuCtxSetCurrent (cudaContext_), "making the CUDA context current", error); }

    // pipeline_ is published last, so a set pipeline_ means the whole setup finished.
    // A setup that fails releases all it made, and the next engine's try starts from
    // nothing, with one context. A lost device keeps its objects: no pass touches the
    // driver again, and releasing them could hang on it.
    bool ensurePipeline (std::string& error)
    {
        if (pipeline_ != nullptr)
            return true;
        SetupUnwind undo;
        const bool built = buildPipeline (undo, error);
        if (built || deviceLost())
            undo.commit();
        return built;
    }

    bool buildPipeline (SetupUnwind& undo, std::string& error)
    {
        const std::string init = optixInitProblem();
        if (! init.empty())
        {
            error = "OptiX device: " + init;
            return false;
        }
        if (! cu_.load (error))
            return false;
        int gpu = 0;
        if (! cuda (cu_.cuInit (0), "cuInit", error) || ! cuda (cu_.cuDeviceGet (&gpu, 0), "cuDeviceGet", error)
            || ! cuda (cu_.cuCtxCreate (&cudaContext_, 0, gpu), "cuCtxCreate", error))
            return false;
        undo.add ([] {
            cu_.cuCtxDestroy (cudaContext_);
            cudaContext_ = nullptr;
        });
        undo.add ([] {
            for (const CUdeviceptr address : permanent_)
                cu_.cuMemFree (address);
            permanent_.clear();
            sbt_ = {};
            tables_ = {};
            logSize_ = 0;
        });
        OptixDeviceContextOptions contextOptions {};
        if (! check (optixDeviceContextCreate (cudaContext_, &contextOptions, &context_), "creating the device context", error))
            return false;
        undo.add ([] {
            optixDeviceContextDestroy (context_);
            context_ = nullptr;
        });

        OptixModuleCompileOptions moduleOptions {};
        moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        moduleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        moduleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
        OptixPipelineCompileOptions pipelineOptions {};
        pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipelineOptions.numPayloadValues = 5;  // t, u, v, triangle, ignored triangle
        pipelineOptions.numAttributeValues = 2;
        pipelineOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipelineOptions.pipelineLaunchParamsVariableName = "params";
        pipelineOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
        // Motion blur: a moving scene's GAS has two motion keys (shutter open and
        // close) and rays carry their time. A still GAS traces the same in this pipeline.
        pipelineOptions.usesMotionBlur = 1;
        std::string ptx;
        for (std::size_t chunk = 0; chunk < kArbitOptixKernelPtxChunks; ++chunk)
            ptx += kArbitOptixKernelPtx[chunk];
        // Added first, as a failed compile still has a module to destroy.
        undo.add ([] {
            if (module_ != nullptr)
                optixModuleDestroy (module_);
            module_ = nullptr;
        });
        if (! compileModule (moduleOptions, pipelineOptions, ptx, error))
            return false;

        OptixProgramGroupDesc descriptions[5] {};
        descriptions[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        descriptions[0].raygen = { module_, "__raygen__pathtrace" };
        descriptions[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        descriptions[1].miss = { module_, "__miss__pathtrace" };
        descriptions[2].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        descriptions[2].miss = { module_, "__miss__shadow" };
        descriptions[3].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        descriptions[3].hitgroup.moduleCH = module_;
        descriptions[3].hitgroup.entryFunctionNameCH = "__closesthit__pathtrace";
        descriptions[3].hitgroup.moduleAH = module_;
        descriptions[3].hitgroup.entryFunctionNameAH = "__anyhit__pathtrace";
        descriptions[4].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        descriptions[4].hitgroup.moduleAH = module_;
        descriptions[4].hitgroup.entryFunctionNameAH = "__anyhit__shadow";
        OptixProgramGroupOptions groupOptions {};
        logSize_ = sizeof (log_);
        if (! check (optixProgramGroupCreate (context_, descriptions, 5, &groupOptions, log_, &logSize_, groups_),
                     "creating the program groups", error))
            return false;
        undo.add ([] {
            for (OptixProgramGroup& group : groups_)
            {
                optixProgramGroupDestroy (group);
                group = nullptr;
            }
        });

        OptixPipelineLinkOptions linkOptions {};
        linkOptions.maxTraceDepth = 1;  // raygen traces every ray; hit programs trace none
        logSize_ = sizeof (log_);
        OptixPipeline pipeline = nullptr;
        if (! check (optixPipelineCreate (context_, &pipelineOptions, &linkOptions, groups_, 5, log_, &logSize_, &pipeline),
                     "linking the pipeline", error))
            return false;
        undo.add ([pipeline] { optixPipelineDestroy (pipeline); });
        logSize_ = 0;

        // Sizes the stacks from the compiled programs, so a path's locals always fit.
        OptixStackSizes sizes {};
        for (const OptixProgramGroup group : groups_)
            if (! check (optixUtilAccumulateStackSizes (group, &sizes, pipeline), "reading the stack sizes", error))
                return false;
        unsigned int fromTraversal = 0, fromState = 0, continuation = 0;
        if (! check (optixUtilComputeStackSizes (&sizes, linkOptions.maxTraceDepth, 0, 0, &fromTraversal, &fromState, &continuation),
                     "computing the stack sizes", error)
            || ! check (optixPipelineSetStackSize (pipeline, fromTraversal, fromState, continuation, 1), "setting the stack size", error))
            return false;
        stackBytesPerThread_ = continuation;

        EmptyRecord raygen, misses[2], hits[2];
        if (! check (optixSbtRecordPackHeader (groups_[0], &raygen), "packing an SBT record", error)
            || ! check (optixSbtRecordPackHeader (groups_[1], &misses[0]), "packing an SBT record", error)
            || ! check (optixSbtRecordPackHeader (groups_[2], &misses[1]), "packing an SBT record", error)
            || ! check (optixSbtRecordPackHeader (groups_[3], &hits[0]), "packing an SBT record", error)
            || ! check (optixSbtRecordPackHeader (groups_[4], &hits[1]), "packing an SBT record", error))
            return false;
        CUdeviceptr raygenRecord = 0, missRecords = 0, hitRecords = 0;
        if (! uploadBytes (&raygen, sizeof (raygen), permanent_, raygenRecord, error)
            || ! uploadBytes (misses, sizeof (misses), permanent_, missRecords, error)
            || ! uploadBytes (hits, sizeof (hits), permanent_, hitRecords, error))
            return false;
        sbt_.raygenRecord = raygenRecord;
        sbt_.missRecordBase = missRecords;
        sbt_.missRecordStrideInBytes = sizeof (EmptyRecord);
        sbt_.missRecordCount = 2;
        sbt_.hitgroupRecordBase = hitRecords;
        sbt_.hitgroupRecordStrideInBytes = sizeof (EmptyRecord);
        sbt_.hitgroupRecordCount = 2;

        DeviceLookupTables tables;
        exportSpectrumTables (tables);
        exportGgxTables (tables);
        tables_ = tables;
        if (! uploadTables (tables, error))
            return false;
        pipeline_ = pipeline;
        return true;
    }

    // A launch holds the pipeline's stack in local memory for every thread the GPU
    // can run at once: 46 KB a thread, 3.1 GB on an RTX 4070 (measured). Left to
    // itself the driver grows the context's local memory at the first launch, and
    // when the GPU has less free than that it does not refuse the launch: the GPU
    // faults (Xid 13, SKEDCHECK05_LOCAL_MEMORY_TOTAL_SIZE) and the context is lost,
    // CUDA error 719 at the next copy. A stack limit of the same size makes the driver
    // reserve the memory before the launch, and a GPU without room says so with an
    // out-of-memory code that leaves the context usable for a later pass
    // (pathtrace_optix_preview_sequence_tests.cpp, ARBIT_OPTIX_SEQUENCE_LEAVE_MIB).
    // The first launch does it rather than the build, which runs at helper startup
    // and would hold the memory for a session that never traces.
    static bool reserveStack (std::string& error)
    {
        if (stackReserved_)
            return true;
        const unsigned int bytesPerThread = stackBytesPerThread_;
        std::size_t freeBefore = 0, freeAfter = 0, total = 0;
        if (cu_.cuMemGetInfo != nullptr)
            cu_.cuMemGetInfo (&freeBefore, &total);
        const int result = cu_.cuCtxSetLimit (0x00 /* CU_LIMIT_STACK_SIZE */, bytesPerThread);
        if (cu_.cuMemGetInfo != nullptr)
            cu_.cuMemGetInfo (&freeAfter, &total);
        std::fprintf (stderr, "[pt] optix stack %u bytes per thread, reserved %zu MiB (free %zu MiB before, %zu MiB after, result %d)\n",
                      bytesPerThread, freeBefore > freeAfter ? (freeBefore - freeAfter) >> 20 : 0, freeBefore >> 20, freeAfter >> 20, result);
        if (result == 0)
        {
            stackReserved_ = true;
            return true;
        }
        if (cudaErrorLosesDevice (result))
            noteDeviceLost();
        error = "OptiX device: reserving " + std::to_string (static_cast<unsigned long long> (bytesPerThread))
              + " bytes of path stack per GPU thread failed with CUDA error " + std::to_string (result)
              + (result == 2 ? ": the GPU has too little free memory; close other GPU apps" : "");
        return false;
    }

    bool uploadBytes (const void* data, std::size_t size, std::vector<CUdeviceptr>& owner, CUdeviceptr& address, std::string& error)
    {
        address = 0;
        if (size == 0)
            return true;
        if (! cuda (cu_.cuMemAlloc (&address, size), "allocating device memory", error))
            return false;
        owner.push_back (address);
        return cuda (cu_.cuMemcpyHtoD (address, data, size), "uploading to the device", error);
    }

    bool uploadTables (const DeviceLookupTables& tables, std::string& error)
    {
        CUdeviceptr weight = 0, basis = 0, scale = 0, bias = 0;
        if (! uploadBytes (tables.spectrumWeight, tables.spectrumRows * sizeof (float[3]), permanent_, weight, error)
            || ! uploadBytes (tables.spectrumBasis, tables.spectrumRows * sizeof (float[3]), permanent_, basis, error)
            || ! uploadBytes (tables.ggxScale, tables.ggxEntries * sizeof (float), permanent_, scale, error)
            || ! uploadBytes (tables.ggxBias, tables.ggxEntries * sizeof (float), permanent_, bias, error))
            return false;
        tables_.spectrumWeight = reinterpret_cast<const float (*)[3]> (weight);
        tables_.spectrumBasis = reinterpret_cast<const float (*)[3]> (basis);
        tables_.ggxScale = reinterpret_cast<const float*> (scale);
        tables_.ggxBias = reinterpret_cast<const float*> (bias);
        return true;
    }

    void freeScene()
    {
        for (const CUdeviceptr address : scene_)
            cu_.cuMemFree (address);
        scene_.clear();
    }

    template <typename T>
    bool uploadView (BufferView<T> source, BufferView<T>& destination, std::string& error)
    {
        static_assert (std::is_trivially_copyable_v<T>, "a device buffer is copied as bytes");
        CUdeviceptr address = 0;
        if (! uploadBytes (source.data, source.count * sizeof (T), scene_, address, error))
            return false;
        destination = { reinterpret_cast<const T*> (address), source.count };
        return true;
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
        const auto uploadStart = PhaseClock::now();
        freeScene();
        haveScene_ = false;
        const RenderScene& scene = pass.scene;
        view_ = {};
        if (! uploadView (BufferView<Vec3> { scene.geometry.positions.data(), scene.geometry.positions.size() }, view_.positions, error)
            || ! uploadView (BufferView<TriangleAttributes> { scene.geometry.triangles.data(), scene.geometry.triangles.size() }, view_.triangles, error)
            || ! uploadView (BufferView<Material> { scene.materials.data(), scene.materials.size() }, view_.materials, error)
            || ! uploadView (BufferView<GratingBinding> { scene.gratings.data(), scene.gratings.size() }, view_.gratings, error)
            || ! uploadView (BufferView<Vec3> { scene.geometry.motionOpen.data(), scene.geometry.motionOpen.size() }, view_.motionOpen, error)
            || ! uploadView (BufferView<Vec3> { scene.geometry.motionClose.data(), scene.geometry.motionClose.size() }, view_.motionClose, error))
            return false;
        CUdeviceptr fallback = 0;
        if (! uploadBytes (&scene.fallbackMaterial, sizeof (Material), scene_, fallback, error))
            return false;
        view_.fallbackMaterial = reinterpret_cast<const Material*> (fallback);
        if (scene.hasWorldVolume)
        {
            CUdeviceptr world = 0;
            if (! uploadBytes (&scene.worldVolume, sizeof (Material), scene_, world, error))
                return false;
            view_.worldMedium = reinterpret_cast<const Material*> (world);
            view_.worldVolumeRadius = scene.worldVolumeExtent;
        }

        // The light set's own buffers first, then the view that points at them.
        bool uploaded = true;
        const LightSetView lights = scene.lights.deviceCopy ([&] (auto source) {
            decltype (source) copy {};
            uploaded = uploadView (source, copy, error) && uploaded;
            return copy;
        });
        CUdeviceptr lightView = 0;
        if (! uploaded || ! uploadBytes (&lights, sizeof (LightSetView), scene_, lightView, error))
            return false;
        view_.lights = reinterpret_cast<const LightSetView*> (lightView);

        view_.anyPartialCoverage = scene.anyPartialCoverage;
        view_.anyThinSheet = scene.anyThinSheet;
        view_.anyDispersive = scene.anyDispersive;
        view_.anyMediumBoundary = scene.anyMediumBoundary;
        view_.anyVolume = scene.anyVolume;
        view_.anyHiddenObjects = scene.anyHiddenObjects;
        view_.boundsCentre = scene.boundsCentre;
        view_.boundsRadius = scene.boundsRadius;
        // cuMemcpyHtoD from pageable memory returns once the bytes are staged, so the
        // last copies may still be in flight; the build below queues behind them.
        phases.set (FramePhase::Upload, millisecondsSince (uploadStart));
        const auto accelStart = PhaseClock::now();
        const bool accelBuilt = buildAccel (scene, error);
        // optixAccelBuild only queues the build on the default stream. Its GPU time
        // finishes inside the first launch's parameter copy, so traceMs carries it.
        phases.set (FramePhase::AccelBuild, millisecondsSince (accelStart));
        phases.accelBuild = "build";
        if (! accelBuilt)
            return false;
        haveScene_ = true;
        sceneUploaded_ = build;
        sceneAddress_ = &pass.scene;
        return true;
    }

    bool buildAccel (const RenderScene& scene, std::string& error)
    {
        view_.accel = 0;
        const auto vertexCount = static_cast<unsigned int> (scene.geometry.positions.size());
        if (vertexCount == 0)
            return true;  // a scene of lights alone: rays miss a null handle
        // A moving scene builds a motion GAS from its shutter-open and shutter-close
        // vertices. Ingest has already expanded instances and deformations into
        // world-space triangles, so two linear keys move every vertex exactly as
        // the CPU BVH does: rigid objects, instances and deforming meshes alike.
        const bool moving = ! view_.motionOpen.empty();
        CUdeviceptr vertices[2] {
            reinterpret_cast<CUdeviceptr> (moving ? view_.motionOpen.data : view_.positions.data),
            reinterpret_cast<CUdeviceptr> (moving ? view_.motionClose.data : view_.positions.data) };
        const std::uint32_t flags = OPTIX_GEOMETRY_FLAG_NONE;  // keeps any-hit, which honours Ray::ignoreTriangle
        OptixBuildInput input {};
        input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        input.triangleArray.vertexStrideInBytes = sizeof (Vec3);
        input.triangleArray.numVertices = vertexCount;
        input.triangleArray.vertexBuffers = vertices;
        input.triangleArray.flags = &flags;
        input.triangleArray.numSbtRecords = 1;
        OptixAccelBuildOptions options {};
        options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        options.operation = OPTIX_BUILD_OPERATION_BUILD;
        if (moving)
        {
            // Ray time runs 0 at shutter open to 1 at close, as Ray::time.
            options.motionOptions.numKeys = 2;
            options.motionOptions.flags = OPTIX_MOTION_FLAG_NONE;
            options.motionOptions.timeBegin = 0.0f;
            options.motionOptions.timeEnd = 1.0f;
        }
        OptixAccelBufferSizes sizes {};
        if (! check (optixAccelComputeMemoryUsage (context_, &options, &input, 1, &sizes), "sizing the acceleration structure", error))
            return false;
        CUdeviceptr temporary = 0, output = 0;
        if (! cuda (cu_.cuMemAlloc (&temporary, sizes.tempSizeInBytes), "allocating the acceleration build", error))
            return false;
        scene_.push_back (temporary);
        if (! cuda (cu_.cuMemAlloc (&output, sizes.outputSizeInBytes), "allocating the acceleration structure", error))
            return false;
        scene_.push_back (output);
        // No stream is the default one, and the launch that follows waits on it.
        return check (optixAccelBuild (context_, nullptr, &options, &input, 1, temporary, sizes.tempSizeInBytes, output,
                                       sizes.outputSizeInBytes, &view_.accel, nullptr, 0), "building the acceleration structure", error);
    }

    bool ensureFilm (std::uint32_t width, std::uint32_t height, std::string& error)
    {
        if (filmWidth_ == width && filmHeight_ == height)
            return true;
        for (const CUdeviceptr address : film_)
            cu_.cuMemFree (address);
        film_.clear();
        filmWidth_ = filmHeight_ = 0;
        const std::size_t bytes = static_cast<std::size_t> (width) * height * sizeof (optix::FilmPixel);
        for (CUdeviceptr* address : { &pixels_, &splats_, &guideAlbedo_, &guideNormal_, &lightPaths_, &totals_, &paramsAddress_ })
        {
            const std::size_t size = address == &pixels_ || address == &splats_ || address == &guideAlbedo_ || address == &guideNormal_ ? bytes
                                   : address == &lightPaths_ ? sizeof (unsigned long long)
                                   : address == &totals_ ? sizeof (optix::LaunchTotals) : sizeof (optix::LaunchParams);
            if (! cuda (cu_.cuMemAlloc (address, size), "allocating the film", error))
                return false;
            film_.push_back (*address);
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
        if (! reserveStack (error) || ! ensureFilm (width, height, error))
            return false;
        const std::size_t pixelBytes = static_cast<std::size_t> (width) * height * sizeof (optix::FilmPixel);
        const unsigned long long noPaths = 0;
        const optix::LaunchTotals noTotals {};
        if (! cuda (cu_.cuMemsetD8 (pixels_, 0, pixelBytes), "clearing the film", error)
            || ! cuda (cu_.cuMemsetD8 (splats_, 0, pixelBytes), "clearing the film", error)
            || ! cuda (cu_.cuMemsetD8 (guideAlbedo_, 0, pixelBytes), "clearing the film", error)
            || ! cuda (cu_.cuMemsetD8 (guideNormal_, 0, pixelBytes), "clearing the film", error)
            || ! cuda (cu_.cuMemcpyHtoD (lightPaths_, &noPaths, sizeof (noPaths)), "clearing the film", error)
            || ! cuda (cu_.cuMemcpyHtoD (totals_, &noTotals, sizeof (noTotals)), "clearing the film", error))
            return false;

        optix::LaunchParams params;
        params.view = view_;
        params.settings = pass.settings;
        params.camera = pass.camera;
        params.film = { reinterpret_cast<optix::FilmPixel*> (pixels_), reinterpret_cast<optix::FilmPixel*> (splats_),
                        reinterpret_cast<unsigned long long*> (lightPaths_),
                        reinterpret_cast<optix::FilmPixel*> (guideAlbedo_), reinterpret_cast<optix::FilmPixel*> (guideNormal_), width };
        params.totals = reinterpret_cast<optix::LaunchTotals*> (totals_);
        params.tables = tables_;
        params.samples = 1;
        params.generation = pass.generation;
        // A launch per sample, so a cancel stops the frame between samples.
        std::uint32_t launched = 0;
        for (std::uint32_t s = 0; s < pass.samples; ++s)
        {
            if (pass.cancel.isCancelled())
            {
                pass.stats.cancelled = true;
                break;
            }
            params.firstSample = pass.firstSample + s;
            if (! cuda (cu_.cuMemcpyHtoD (paramsAddress_, &params, sizeof (params)), "uploading the launch parameters", error)
                || ! check (optixLaunch (pipeline_, nullptr, paramsAddress_, sizeof (params), &sbt_, width, height, 1), "launching the kernel", error))
                return false;
            ++launched;
        }
        // Each parameter copy waits for the launch before it, so this covers every
        // sample but the last one's GPU work, which finishes inside the readback.
        pass.stats.phases.set (FramePhase::Trace, millisecondsSince (traceStart));
        if (launched == 0)
        {
            pass.stats.phases.set (FramePhase::Readback, 0.0);
            return true;
        }
        const auto readbackStart = PhaseClock::now();
        const bool read = readBack (pass, pixelBytes);
        pass.stats.phases.set (FramePhase::Readback, millisecondsSince (readbackStart));
        return read;
    }

public:
    // The OptiX AI denoiser, which the denoiser choice runs on any device's image.
    static std::string denoiserProblem()
    {
        const std::lock_guard<std::mutex> lock (deviceLock_);
        if (deviceLost())
            return "the GPU context was lost earlier in this session";
        if (context_ == nullptr || pipeline_ == nullptr)
            return "the OptiX device has not started in this session";
        return {};
    }

private:
    static void freeDenoiserImages (DenoiserSlot& slot)
    {
        for (const CUdeviceptr address : slot.memory)
            cu_.cuMemFree (address);
        slot.memory.clear();
        slot.width = slot.height = 0;
        slot.hasPrevious = false;
    }

    static bool denoiserAllocate (DenoiserSlot& slot, CUdeviceptr& address, std::size_t bytes, std::string& error)
    {
        if (! cuda (cu_.cuMemAlloc (&address, bytes), "allocating the denoiser's memory", error))
            return false;
        slot.memory.push_back (address);
        return true;
    }

    static bool ensureDenoiser (DenoiserSlot& slot, bool temporal, std::uint32_t width, std::uint32_t height, std::string& error)
    {
        if (slot.denoiser == nullptr)
        {
            OptixDenoiserOptions options {};
            options.guideAlbedo = 1;
            options.guideNormal = 1;
            options.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
            // TEMPORAL_AOV is the SDK 9 name of OPTIX_DENOISER_MODEL_KIND_TEMPORAL, which maps to it.
            if (! check (optixDenoiserCreate (context_, temporal ? OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV
                                                                         : OPTIX_DENOISER_MODEL_KIND_AOV,
                                                      &options, &slot.denoiser), "creating the AI denoiser", error))
                return false;
        }
        if (slot.width == width && slot.height == height)
            return true;
        freeDenoiserImages (slot);
        if (! check (optixDenoiserComputeMemoryResources (slot.denoiser, width, height, &slot.sizes),
                             "sizing the AI denoiser", error))
            return false;
        const std::size_t pixels = static_cast<std::size_t> (width) * height;
        const std::size_t image = pixels * 3 * sizeof (float);
        if (! denoiserAllocate (slot, slot.state, slot.sizes.stateSizeInBytes, error)
            || ! denoiserAllocate (slot, slot.scratch, slot.sizes.withoutOverlapScratchSizeInBytes, error)
            || ! denoiserAllocate (slot, slot.color, image, error)
            || ! denoiserAllocate (slot, slot.albedo, image, error)
            || ! denoiserAllocate (slot, slot.normal, image, error)
            || ! denoiserAllocate (slot, slot.output, image, error))
        {
            freeDenoiserImages (slot);
            return false;
        }
        if (temporal)
        {
            const std::size_t guide = pixels * slot.sizes.internalGuideLayerPixelSizeInBytes;
            if (! denoiserAllocate (slot, slot.flow, pixels * 2 * sizeof (float), error)
                || ! denoiserAllocate (slot, slot.guides[0], guide, error)
                || ! denoiserAllocate (slot, slot.guides[1], guide, error))
            {
                freeDenoiserImages (slot);
                return false;
            }
        }
        if (! check (optixDenoiserSetup (slot.denoiser, nullptr, width, height, slot.state, slot.sizes.stateSizeInBytes,
                                                 slot.scratch, slot.sizes.withoutOverlapScratchSizeInBytes),
                             "setting up the AI denoiser", error))
        {
            freeDenoiserImages (slot);
            return false;
        }
        slot.width = width;
        slot.height = height;
        return true;
    }

public:
    static bool denoise (const OptixDenoiseImage& frame, std::string& error)
    {
        const std::lock_guard<std::mutex> lock (deviceLock_);
        if (deviceLost())
        {
            error = "OptiX AI denoiser: the GPU context was lost earlier in this session";
            return false;
        }
        if (context_ == nullptr || pipeline_ == nullptr)
        {
            error = "OptiX AI denoiser: the OptiX device has not started in this session";
            return false;
        }
        if (frame.width == 0 || frame.height == 0 || frame.color == nullptr || frame.albedo == nullptr
            || frame.normal == nullptr || frame.output == nullptr || (frame.temporal && frame.flow == nullptr))
        {
            error = "OptiX AI denoiser: the frame is incomplete";
            return false;
        }
        if (! enter (error))
            return false;
        DenoiserSlot& slot = denoisers_[frame.temporal ? 1 : 0];
        if (! ensureDenoiser (slot, frame.temporal, frame.width, frame.height, error))
            return false;
        const std::size_t pixels = static_cast<std::size_t> (frame.width) * frame.height;
        const std::size_t bytes = pixels * 3 * sizeof (float);
        if (! cuda (cu_.cuMemcpyHtoD (slot.color, frame.color, bytes), "uploading the denoiser's input", error)
            || ! cuda (cu_.cuMemcpyHtoD (slot.albedo, frame.albedo, bytes), "uploading the denoiser's input", error)
            || ! cuda (cu_.cuMemcpyHtoD (slot.normal, frame.normal, bytes), "uploading the denoiser's input", error))
            return false;
        const bool usePrevious = frame.temporal && frame.usePrevious && slot.hasPrevious;
        if (frame.temporal)
        {
            const std::size_t guideBytes = pixels * slot.sizes.internalGuideLayerPixelSizeInBytes;
            // A new sequence: the noisy frame stands in for the previous output, with zero flow
            // and a cleared guide layer, as optixDenoiserInvoke asks of a first frame.
            if (! cuda (cu_.cuMemcpyHtoD (slot.flow, frame.flow, pixels * 2 * sizeof (float)), "uploading the flow", error)
                || (! usePrevious && (! cuda (cu_.cuMemcpyDtoD (slot.output, slot.color, bytes), "starting the sequence", error)
                                      || ! cuda (cu_.cuMemsetD8 (slot.guides[slot.currentGuide], 0, guideBytes), "starting the sequence", error)
                                      || ! cuda (cu_.cuMemsetD8 (slot.flow, 0, pixels * 2 * sizeof (float)), "starting the sequence", error))))
                return false;
        }
        const auto image = [&] (CUdeviceptr data, OptixPixelFormat format, unsigned int pixelBytes) {
            OptixImage2D result {};
            result.data = data;
            result.width = frame.width;
            result.height = frame.height;
            result.rowStrideInBytes = frame.width * pixelBytes;
            result.pixelStrideInBytes = pixelBytes;
            result.format = format;
            return result;
        };
        constexpr unsigned int float3Bytes = 3 * sizeof (float);
        OptixDenoiserGuideLayer guides {};
        guides.albedo = image (slot.albedo, OPTIX_PIXEL_FORMAT_FLOAT3, float3Bytes);
        guides.normal = image (slot.normal, OPTIX_PIXEL_FORMAT_FLOAT3, float3Bytes);
        OptixDenoiserLayer layer {};
        layer.input = image (slot.color, OPTIX_PIXEL_FORMAT_FLOAT3, float3Bytes);
        layer.output = image (slot.output, OPTIX_PIXEL_FORMAT_FLOAT3, float3Bytes);
        layer.type = OPTIX_DENOISER_AOV_TYPE_BEAUTY;
        OptixDenoiserParams params {};
        if (frame.temporal)
        {
            const auto guidePixel = static_cast<unsigned int> (slot.sizes.internalGuideLayerPixelSizeInBytes);
            guides.flow = image (slot.flow, OPTIX_PIXEL_FORMAT_FLOAT2, 2 * sizeof (float));
            guides.previousOutputInternalGuideLayer = image (slot.guides[slot.currentGuide], OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER, guidePixel);
            guides.outputInternalGuideLayer = image (slot.guides[1 - slot.currentGuide], OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER, guidePixel);
            // The output buffer holds the previous frame's output; the invoke reads it first.
            layer.previousOutput = layer.output;
            params.temporalModeUsePreviousLayers = usePrevious ? 1u : 0u;
        }
        if (! check (optixDenoiserInvoke (slot.denoiser, nullptr, &params, slot.state, slot.sizes.stateSizeInBytes,
                                                  &guides, &layer, 1, 0, 0, slot.scratch, slot.sizes.withoutOverlapScratchSizeInBytes),
                             "running the AI denoiser", error)
            || ! cuda (cu_.cuMemcpyDtoH (frame.output, slot.output, bytes), "reading the denoised frame back", error))
        {
            slot.hasPrevious = false;
            return false;
        }
        if (frame.temporal)
        {
            slot.currentGuide = 1 - slot.currentGuide;
            slot.hasPrevious = true;
        }
        return true;
    }

private:
    bool readBack (PathTracer::DevicePass& pass, std::size_t pixelBytes)
    {
        std::string& error = pass.error;
        hostPixels_.resize (pixelBytes / sizeof (optix::FilmPixel));
        hostSplats_.resize (hostPixels_.size());
        hostAlbedo_.resize (hostPixels_.size());
        hostNormal_.resize (hostPixels_.size());
        unsigned long long lightPaths = 0;
        optix::LaunchTotals totals;
        if (! cuda (cu_.cuMemcpyDtoH (hostPixels_.data(), pixels_, pixelBytes), "reading the film back", error)
            || ! cuda (cu_.cuMemcpyDtoH (hostSplats_.data(), splats_, pixelBytes), "reading the film back", error)
            || ! cuda (cu_.cuMemcpyDtoH (hostAlbedo_.data(), guideAlbedo_, pixelBytes), "reading the guides back", error)
            || ! cuda (cu_.cuMemcpyDtoH (hostNormal_.data(), guideNormal_, pixelBytes), "reading the guides back", error)
            || ! cuda (cu_.cuMemcpyDtoH (&lightPaths, lightPaths_, sizeof (lightPaths)), "reading the film back", error)
            || ! cuda (cu_.cuMemcpyDtoH (&totals, totals_, sizeof (totals)), "reading the film back", error))
            return false;
        const std::uint32_t width = pass.film.width();
        for (std::size_t i = 0; i < hostPixels_.size(); ++i)
        {
            const auto x = static_cast<std::uint32_t> (i % width);
            const auto y = static_cast<std::uint32_t> (i / width);
            const optix::FilmPixel& pixel = hostPixels_[i];
            if (pixel.w > 0.0f)
                pass.film.addSampleSum (x, y, { pixel.x, pixel.y, pixel.z }, static_cast<std::uint32_t> (pixel.w));
            const optix::FilmPixel& splat = hostSplats_[i];
            if (splat.x != 0.0f || splat.y != 0.0f || splat.z != 0.0f)
                pass.film.addSplat (x, y, { splat.x, splat.y, splat.z });
            const optix::FilmPixel& albedo = hostAlbedo_[i];
            const optix::FilmPixel& normal = hostNormal_[i];
            if (albedo.w > 0.0f)
                pass.film.addGuideSum (x, y, { albedo.x, albedo.y, albedo.z }, { normal.x, normal.y, normal.z },
                                       static_cast<std::uint32_t> (albedo.w));
        }
        pass.film.addLightPaths (lightPaths);
        PathCounters counters;
        std::memcpy (&counters, totals.counters, sizeof (counters));
        pass.counters += counters;
        pass.stats.samplesAdded += totals.samplesAdded;
        pass.stats.nonFiniteSamples += totals.nonFinite;
        std::size_t freeBytes = 0, totalBytes = 0;
        if (cu_.cuMemGetInfo != nullptr && cu_.cuMemGetInfo (&freeBytes, &totalBytes) == 0)
        {
            pass.stats.deviceMemoryFreeBytes = freeBytes;
            pass.stats.deviceMemoryTotalBytes = totalBytes;
        }
        return true;
    }

    // One context and pipeline for the process. Every engine with its own pipeline
    // reserved its own stack memory, about 3.9 GB, and with two the GPU refused the
    // launch (Xid 13, SKEDCHECK05_LOCAL_MEMORY_TOTAL_SIZE). Engines keep their own
    // scene and film, and take turns on the device.
    inline static std::mutex deviceLock_;
    inline static CudaApi cu_;
    inline static CUcontext cudaContext_ = nullptr;
    inline static OptixDeviceContext context_ = nullptr;
    inline static OptixModule module_ = nullptr;
    inline static std::mutex creatingLock_;
    inline static OptixModule creating_ = nullptr;
    inline static bool cancelled_ = false;
    inline static OptixProgramGroup groups_[5] {};
    inline static OptixPipeline pipeline_ = nullptr;
    inline static unsigned int stackBytesPerThread_ = 0;
    inline static bool stackReserved_ = false;
    inline static OptixShaderBindingTable sbt_ {};
    inline static DeviceLookupTables tables_;
    inline static char log_[4096] {};
    inline static std::size_t logSize_ = 0;
    inline static std::vector<CUdeviceptr> permanent_;
    inline static DenoiserSlot denoisers_[2];

    std::function<std::uint64_t()> sceneBuild_;
    std::vector<CUdeviceptr> scene_;
    std::vector<CUdeviceptr> film_;
    optix::SceneViewForLaunch view_;
    bool haveScene_ = false;
    std::uint64_t sceneUploaded_ = 0;
    const RenderScene* sceneAddress_ = nullptr;
    std::uint32_t filmWidth_ = 0;
    std::uint32_t filmHeight_ = 0;
    CUdeviceptr pixels_ = 0, splats_ = 0, guideAlbedo_ = 0, guideNormal_ = 0, lightPaths_ = 0, totals_ = 0, paramsAddress_ = 0;
    std::vector<optix::FilmPixel> hostPixels_;
    std::vector<optix::FilmPixel> hostSplats_;
    std::vector<optix::FilmPixel> hostAlbedo_;
    std::vector<optix::FilmPixel> hostNormal_;
};

} // namespace

PathTracer::DevicePassFunction makeOptixDevicePass (std::function<std::uint64_t()> sceneBuild)
{
    auto renderer = std::make_shared<OptixRenderer> (std::move (sceneBuild));
    return [renderer] (PathTracer::DevicePass& pass) { return renderer->render (pass); };
}

bool optixPipelineReady() { return OptixRenderer::pipelineReady(); }
void startOptixPipelineBuild() { OptixRenderer::startBuild(); }
void cancelOptixPipelineBuild() { OptixRenderer::cancelBuild(); }
bool stopOptixPipelineBuildByUser() { return OptixRenderer::stopBuildByUser(); }
void startOptixPipelineBuildByUser() { OptixRenderer::startBuildByUser(); }
std::string optixDenoiserProblem() { return OptixRenderer::denoiserProblem(); }
bool optixDenoise (const OptixDenoiseImage& image, std::string& error) { return OptixRenderer::denoise (image, error); }
#else
std::string optixDenoiserProblem() { return "this build of DonutStudio has no OptiX kernel"; }
bool optixDenoise (const OptixDenoiseImage&, std::string& error)
{
    error = "OptiX AI denoiser: this build of DonutStudio has no OptiX kernel";
    return false;
}
#endif // ARBIT_HAVE_OPTIX_KERNEL

} // namespace videohelper::pathtrace::device
