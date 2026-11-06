// Host side of the CUDA, HIP and oneAPI devices. The two driver APIs match call
// for call in what this needs, and the oneAPI device library exports the same
// shape over SYCL (device_oneapi_abi.h), so one class drives all three through a
// table of entry points found with dlopen. Nothing here links against vendor code,
// and the helper starts without any of them.
//
// The kernel (device_compute_kernel.inl) is embedded at build time: PTX for CUDA,
// which the driver compiles for the GPU on first load and caches, and a code
// object for HIP (cmake/GpuEmbedKernel.cmake). The oneAPI kernel lives in its
// library, as SPIR-V the GPU driver compiles on first load.

#include "device_compute_host.h"
#include "device_compute_params.h"
#include "device_oneapi_abi.h"
#include "device_registry.h"
#include "kernel_build_gate.h"
#include "kernel_compile_state.h"
#include "setup_unwind.h"
#include "../bsdf.h"
#include "../spectrum.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <cstddef>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#if defined(ARBIT_HAVE_CUDA_KERNEL)
extern const unsigned char kArbitCudaKernel[];
extern const std::size_t kArbitCudaKernelSize;
#endif
#if defined(ARBIT_HAVE_HIP_KERNEL)
extern const unsigned char kArbitHipKernel[];
extern const std::size_t kArbitHipKernelSize;
#endif

namespace videohelper::pathtrace::device
{
namespace
{

// A device address. CUdeviceptr is an unsigned 64-bit integer and hipDeviceptr_t a
// pointer; both are eight bytes on every platform the helper builds for.
using DeviceAddress = unsigned long long;
static_assert (sizeof (DeviceAddress) == sizeof (void*), "device addresses are pointer sized");
using Handle = void*;

// The driver calls the renderer makes, by the names each vendor gives them.
struct DriverNames
{
    const char* device;          // "CUDA" or "HIP"
    const char* label;           // "CUDA device" or "HIP device", which starts every error
    const char* errorName;       // "CUDA error" or "HIP error"
    const char* library;
    const char* init;
    const char* deviceGet;
    const char* ctxCreate;
    const char* ctxSetCurrent;
    const char* ctxDestroy;
    const char* memAlloc;
    const char* memFree;
    const char* memcpyHtoD;
    const char* memcpyDtoH;
    const char* memsetD8;
    const char* moduleLoadData;
    const char* moduleUnload;
    const char* moduleGetFunction;
    const char* moduleGetGlobal;
    const char* launchKernel;
    const char* synchronize;
    // Optional: a driver without it leaves the memory readout empty.
    const char* memGetInfo;
    // Optional: the last failure in the runtime's words, added to the error.
    const char* lastError = nullptr;
    // library is a file beside the helper rather than a system library.
    bool besideHelper = false;
};

constexpr DriverNames kCudaNames {
    "CUDA", "CUDA device", "CUDA error",
#if defined(_WIN32)
    "nvcuda.dll",
#else
    "libcuda.so.1",
#endif
    "cuInit", "cuDeviceGet", "cuCtxCreate_v2", "cuCtxSetCurrent", "cuCtxDestroy_v2", "cuMemAlloc_v2", "cuMemFree_v2",
    "cuMemcpyHtoD_v2", "cuMemcpyDtoH_v2", "cuMemsetD8_v2", "cuModuleLoadData", "cuModuleUnload", "cuModuleGetFunction",
    "cuModuleGetGlobal_v2", "cuLaunchKernel", "cuCtxSynchronize", "cuMemGetInfo_v2"
};

// HIP keeps the CUDA driver API's shape under hip names. hipCtxCreate is deprecated
// in ROCm but still exported; it gives each process one context as CUDA does.
constexpr DriverNames kHipNames {
    "HIP", "HIP device", "HIP error",
#if defined(_WIN32)
    "amdhip64.dll",
#else
    "libamdhip64.so",
#endif
    "hipInit", "hipDeviceGet", "hipCtxCreate", "hipCtxSetCurrent", "hipCtxDestroy", "hipMalloc", "hipFree",
    "hipMemcpyHtoD", "hipMemcpyDtoH", "hipMemsetD8", "hipModuleLoadData", "hipModuleUnload", "hipModuleGetFunction",
    "hipModuleGetGlobal", "hipModuleLaunchKernel", "hipCtxSynchronize", "hipMemGetInfo"
};

// The oneAPI device library, beside the helper. It links the SYCL runtime, so a
// machine without oneAPI fails this load and the error names the missing library.
constexpr DriverNames kOneApiNames {
    "oneAPI", "oneAPI device", "oneAPI error", oneapi::kLibraryFile,
    oneapi::kInit, oneapi::kDeviceGet, oneapi::kCtxCreate, oneapi::kCtxSetCurrent, oneapi::kCtxDestroy,
    oneapi::kMemAlloc, oneapi::kMemFree, oneapi::kMemcpyHtoD, oneapi::kMemcpyDtoH, oneapi::kMemsetD8,
    oneapi::kModuleLoadData, oneapi::kModuleUnload, oneapi::kModuleGetFunction, oneapi::kModuleGetGlobal,
    oneapi::kLaunchKernel, oneapi::kSynchronize, oneapi::kMemGetInfo, oneapi::kLastError, true
};

// The library handle and a lookup in it, for the driver table and the oneAPI probe.
struct LoadedLibrary
{
    void* handle = nullptr;
    std::string path;
    std::string problem;  // why it did not load, empty when it did

    explicit LoadedLibrary (const DriverNames& names)
        : path (names.besideHelper ? besideHelperPath (names.library) : std::string (names.library))
    {
#if defined(_WIN32)
        handle = reinterpret_cast<void*> (LoadLibraryA (path.c_str()));
        if (handle == nullptr)
            problem = path + " did not load (Windows error " + std::to_string (GetLastError()) + ")";
#else
        handle = dlopen (path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (handle == nullptr)
        {
            const char* const reason = dlerror();
            problem = path + " did not load: " + (reason != nullptr ? reason : "no reason given");
        }
#endif
    }

    void* find (const char* name) const
    {
#if defined(_WIN32)
        return reinterpret_cast<void*> (GetProcAddress (static_cast<HMODULE> (handle), name));
#else
        return dlsym (handle, name);
#endif
    }
};

struct DriverApi
{
    int (*init) (unsigned int) = nullptr;
    int (*deviceGet) (int*, int) = nullptr;
    int (*ctxCreate) (Handle*, unsigned int, int) = nullptr;
    int (*ctxSetCurrent) (Handle) = nullptr;
    int (*ctxDestroy) (Handle) = nullptr;
    int (*memAlloc) (DeviceAddress*, std::size_t) = nullptr;
    int (*memFree) (DeviceAddress) = nullptr;
    int (*memcpyHtoD) (DeviceAddress, const void*, std::size_t) = nullptr;
    int (*memcpyDtoH) (void*, DeviceAddress, std::size_t) = nullptr;
    int (*memsetD8) (DeviceAddress, unsigned char, std::size_t) = nullptr;
    int (*moduleLoadData) (Handle*, const void*) = nullptr;
    int (*moduleUnload) (Handle) = nullptr;
    int (*moduleGetFunction) (Handle*, Handle, const char*) = nullptr;
    int (*moduleGetGlobal) (DeviceAddress*, std::size_t*, Handle, const char*) = nullptr;
    int (*launchKernel) (Handle, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
                         unsigned int, Handle, void**, void**) = nullptr;
    int (*synchronize)() = nullptr;
    int (*memGetInfo) (std::size_t*, std::size_t*) = nullptr;
    const char* (*lastError)() = nullptr;

    bool load (const DriverNames& names, std::string& error)
    {
        const LoadedLibrary library (names);
        if (library.handle == nullptr)
        {
            error = std::string (names.label) + ": " + library.problem;
            return false;
        }
        const auto find = [&library] (const char* name) { return library.find (name); };
        const auto bind = [&] (auto& function, const char* name) {
            function = reinterpret_cast<std::remove_reference_t<decltype (function)>> (find (name));
            if (function == nullptr)
                error = std::string (names.label) + ": " + library.path + " has no " + name;
        };
        bind (init, names.init);
        bind (deviceGet, names.deviceGet);
        bind (ctxCreate, names.ctxCreate);
        bind (ctxSetCurrent, names.ctxSetCurrent);
        bind (ctxDestroy, names.ctxDestroy);
        bind (memAlloc, names.memAlloc);
        bind (memFree, names.memFree);
        bind (memcpyHtoD, names.memcpyHtoD);
        bind (memcpyDtoH, names.memcpyDtoH);
        bind (memsetD8, names.memsetD8);
        bind (moduleLoadData, names.moduleLoadData);
        bind (moduleUnload, names.moduleUnload);
        bind (moduleGetFunction, names.moduleGetFunction);
        bind (moduleGetGlobal, names.moduleGetGlobal);
        bind (launchKernel, names.launchKernel);
        bind (synchronize, names.synchronize);
        memGetInfo = reinterpret_cast<decltype (memGetInfo)> (find (names.memGetInfo));
        if (names.lastError != nullptr)
            lastError = reinterpret_cast<decltype (lastError)> (find (names.lastError));
        return error.empty();
    }
};

// One driver, context and module per device kind for the process, as OptiX keeps
// one pipeline. Engines keep their own scene and film and take turns on the device.
struct SharedDevice
{
    const DriverNames& names;
    const unsigned char* kernel = nullptr;  // null when this build has no kernel for the device
    std::size_t kernelSize = 0;
    std::mutex lock;
    DriverApi api;
    Handle context = nullptr;
    Handle module = nullptr;
    Handle function = nullptr;
    DeviceAddress params = 0;
    std::size_t paramsSize = 0;
    DeviceLookupTables tables;
    std::vector<DeviceAddress> permanent;
    kernelcompile::BuildGate* gate = new kernelcompile::BuildGate();

    explicit SharedDevice (const DriverNames& driverNames) : names (driverNames) {}
};

SharedDevice& sharedDevice (DeviceKind kind)
{
    static auto* cuda = []
    {
        auto* shared = new SharedDevice (kCudaNames);
#if defined(ARBIT_HAVE_CUDA_KERNEL)
        shared->kernel = kArbitCudaKernel;
        shared->kernelSize = kArbitCudaKernelSize;
#endif
        return shared;
    }();
    static auto* hipDevice = []
    {
        auto* shared = new SharedDevice (kHipNames);
#if defined(ARBIT_HAVE_HIP_KERNEL)
        shared->kernel = kArbitHipKernel;
        shared->kernelSize = kArbitHipKernelSize;
#endif
        return shared;
    }();
    static auto* oneApi = []
    {
        auto* shared = new SharedDevice (kOneApiNames);
#if defined(ARBIT_HAVE_ONEAPI_DEVICE)
        shared->kernel = reinterpret_cast<const unsigned char*> (oneapi::kModuleImage);
        shared->kernelSize = sizeof (oneapi::kModuleImage);
#endif
        return shared;
    }();
    return kind == DeviceKind::Hip ? *hipDevice : kind == DeviceKind::OneApi ? *oneApi : *cuda;
}

class ComputeRenderer
{
public:
    ComputeRenderer (SharedDevice& shared, std::function<std::uint64_t()> sceneBuild)
        : shared_ (shared), sceneBuild_ (std::move (sceneBuild)) {}

    ~ComputeRenderer()
    {
        if (shared_.gate->status() != kernelcompile::BuildGate::Status::Ready)
            return;
        const std::lock_guard<std::mutex> lock (shared_.lock);
        if (deviceLost() || shared_.context == nullptr || shared_.api.ctxSetCurrent (shared_.context) != 0)
            return;
        freeAll (scene_);
        freeAll (film_);
    }

    bool render (PathTracer::DevicePass& pass)
    {
        const auto waitStart = PhaseClock::now();
        if (deviceLost())
            return lostError (pass.error);
        {
            // An export comes here while the module loads and waits for it; a preview
            // comes after moduleReady.
            std::optional<kernelcompile::Scope> compiling;
            if (shared_.gate->status() != kernelcompile::BuildGate::Status::Ready)
                compiling.emplace();
            ++kernelcompile::exportsWaiting();
            const bool built = shared_.gate->join (buildBody (shared_), pass.error);
            --kernelcompile::exportsWaiting();
            if (! built)
                return false;
        }
        const std::lock_guard<std::mutex> lock (shared_.lock);
        pass.stats.phases.set (FramePhase::PipelineWait, millisecondsSince (waitStart));
        if (deviceLost())
            return lostError (pass.error);
        return call (shared_.api.ctxSetCurrent (shared_.context), "making the context current", pass.error)
            && ensureScene (pass, pass.error) && launch (pass);
    }

    static std::function<bool (std::string&)> buildBody (SharedDevice& shared)
    {
        return [&shared] (std::string& error) { return ensureModule (shared, error); };
    }

    static bool moduleReady (SharedDevice& shared)
    {
        if (deviceLost())
            return true;
        auto& gate = *shared.gate;
        if (gate.stopped())
        {
            kernelcompile::previewWaiting() = false;
            kernelcompile::previewStopped() = true;
            return false;
        }
        kernelcompile::previewStopped() = false;
        gate.start (buildBody (shared));
        if (gate.status() != kernelcompile::BuildGate::Status::Running)
        {
            kernelcompile::previewWaiting() = false;
            return true;
        }
        if (gate.runningForMs() >= kAnnounceBuildAfterMs)
            kernelcompile::previewWaiting() = true;
        return false;
    }

    static constexpr double kAnnounceBuildAfterMs = 250.0;

private:
    bool lostError (std::string& error) const
    {
        error = std::string (shared_.names.label) + ": the GPU context was lost earlier in this session";
        return false;
    }

    static bool check (const SharedDevice& shared, int result, const char* what, std::string& error)
    {
        if (result == 0)
            return true;
        error = std::string (shared.names.label) + ": " + what + " failed with " + shared.names.errorName + " "
              + std::to_string (result);
        if (shared.api.lastError != nullptr)
            if (const char* const detail = shared.api.lastError(); detail != nullptr && detail[0] != '\0')
                error += std::string (": ") + detail;
        // HIP keeps CUDA's numbers for the codes that lose a context.
        if (cudaErrorLosesDevice (result))
            noteDeviceLost();
        return false;
    }

    bool call (int result, const char* what, std::string& error) const { return check (shared_, result, what, error); }

    // The context, module, tables and parameter block, made once. A failure releases
    // what it made, and the next try starts from nothing.
    static bool ensureModule (SharedDevice& shared, std::string& error)
    {
        const std::lock_guard<std::mutex> lock (shared.lock);
        if (shared.function != nullptr)
            return true;
        if (shared.kernel == nullptr)
        {
            error = std::string (shared.names.label) + ": this build of DonutStudio has no " + shared.names.label + " kernel";
            return false;
        }
        SetupUnwind undo;
        const bool built = buildModule (shared, undo, error);
        if (built || deviceLost())
            undo.commit();
        return built;
    }

    static bool buildModule (SharedDevice& shared, SetupUnwind& undo, std::string& error)
    {
        if (! shared.api.load (shared.names, error))
            return false;
        DriverApi& api = shared.api;
        int gpu = 0;
        if (! check (shared, api.init (0), "initialising the driver", error)
            || ! check (shared, api.deviceGet (&gpu, 0), "finding the GPU", error)
            || ! check (shared, api.ctxCreate (&shared.context, 0, gpu), "creating the context", error))
            return false;
        undo.add ([&shared] {
            shared.api.ctxDestroy (shared.context);
            shared.context = nullptr;
        });
        undo.add ([&shared] {
            for (const DeviceAddress address : shared.permanent)
                shared.api.memFree (address);
            shared.permanent.clear();
            shared.tables = {};
            shared.params = 0;
        });
        if (! check (shared, api.moduleLoadData (&shared.module, shared.kernel), "loading the kernel", error))
            return false;
        undo.add ([&shared] {
            shared.api.moduleUnload (shared.module);
            shared.module = nullptr;
        });
        Handle function = nullptr;
        if (! check (shared, api.moduleGetFunction (&function, shared.module, compute::kKernelName), "finding the kernel", error)
            || ! check (shared, api.moduleGetGlobal (&shared.params, &shared.paramsSize, shared.module, compute::kParamsName),
                        "finding the launch parameters", error))
            return false;
        if (shared.paramsSize != sizeof (compute::LaunchParams))
        {
            error = std::string (shared.names.label) + ": the kernel's launch parameters are "
                  + std::to_string (shared.paramsSize) + " bytes, the host's " + std::to_string (sizeof (compute::LaunchParams));
            return false;
        }
        DeviceLookupTables tables;
        exportSpectrumTables (tables);
        exportGgxTables (tables);
        shared.tables = tables;
        DeviceAddress weight = 0, basis = 0, scale = 0, bias = 0;
        if (! upload (shared, tables.spectrumWeight, tables.spectrumRows * sizeof (float[3]), shared.permanent, weight, error)
            || ! upload (shared, tables.spectrumBasis, tables.spectrumRows * sizeof (float[3]), shared.permanent, basis, error)
            || ! upload (shared, tables.ggxScale, tables.ggxEntries * sizeof (float), shared.permanent, scale, error)
            || ! upload (shared, tables.ggxBias, tables.ggxEntries * sizeof (float), shared.permanent, bias, error))
            return false;
        shared.tables.spectrumWeight = reinterpret_cast<const float (*)[3]> (weight);
        shared.tables.spectrumBasis = reinterpret_cast<const float (*)[3]> (basis);
        shared.tables.ggxScale = reinterpret_cast<const float*> (scale);
        shared.tables.ggxBias = reinterpret_cast<const float*> (bias);
        shared.function = function;
        return true;
    }

    static bool upload (SharedDevice& shared, const void* data, std::size_t size, std::vector<DeviceAddress>& owner,
                        DeviceAddress& address, std::string& error)
    {
        address = 0;
        if (size == 0)
            return true;
        if (! check (shared, shared.api.memAlloc (&address, size), "allocating device memory", error))
            return false;
        owner.push_back (address);
        return check (shared, shared.api.memcpyHtoD (address, data, size), "uploading to the device", error);
    }

    void freeAll (std::vector<DeviceAddress>& addresses)
    {
        for (const DeviceAddress address : addresses)
            shared_.api.memFree (address);
        addresses.clear();
    }

    template <typename T>
    bool uploadView (BufferView<T> source, BufferView<T>& destination, std::string& error)
    {
        static_assert (std::is_trivially_copyable_v<T>, "a device buffer is copied as bytes");
        DeviceAddress address = 0;
        if (! upload (shared_, source.data, source.count * sizeof (T), scene_, address, error))
            return false;
        destination = { reinterpret_cast<const T*> (address), source.count };
        return true;
    }

    template <typename T>
    bool uploadArray (const std::vector<T>& source, const T*& destination, std::string& error)
    {
        static_assert (std::is_trivially_copyable_v<T>, "a device buffer is copied as bytes");
        DeviceAddress address = 0;
        if (! upload (shared_, source.data(), source.size() * sizeof (T), scene_, address, error))
            return false;
        destination = reinterpret_cast<const T*> (address);
        return true;
    }

    bool ensureScene (const PathTracer::DevicePass& pass, std::string& error)
    {
        // This device traces the CPU's BVH, so it builds no structure of its own and
        // the engine reports that BVH's build as AccelBuild.
        const std::uint64_t build = sceneBuild_();
        if (haveScene_ && build == sceneUploaded_ && &pass.scene == sceneAddress_)
        {
            pass.stats.phases.set (FramePhase::Upload, 0.0);
            return true;
        }
        const auto uploadStart = PhaseClock::now();
        freeAll (scene_);
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
        DeviceAddress fallback = 0;
        if (! upload (shared_, &scene.fallbackMaterial, sizeof (Material), scene_, fallback, error))
            return false;
        view_.fallbackMaterial = reinterpret_cast<const Material*> (fallback);
        if (scene.hasWorldVolume)
        {
            DeviceAddress worldMedium = 0;
            if (! upload (shared_, &scene.worldVolume, sizeof (Material), scene_, worldMedium, error))
                return false;
            view_.worldMedium = reinterpret_cast<const Material*> (worldMedium);
        }
        view_.worldVolumeRadius = scene.worldVolumeExtent;
        view_.anyVolume = scene.anyVolume;
        view_.anyHiddenObjects = scene.anyHiddenObjects;

        bool uploaded = true;
        const LightSetView lights = scene.lights.deviceCopy ([&] (auto source) {
            decltype (source) copy {};
            uploaded = uploadView (source, copy, error) && uploaded;
            return copy;
        });
        DeviceAddress lightView = 0;
        if (! uploaded || ! upload (shared_, &lights, sizeof (LightSetView), scene_, lightView, error))
            return false;
        view_.lights = reinterpret_cast<const LightSetView*> (lightView);

        // The CPU's BVH as it is: nodes, leaf-ordered vertices, close-of-shutter
        // vertices when the scene moves, and caller indices.
        view_.accel = {};
        view_.accel.nodeCount = scene.bvh.nodes().size();
        if (! uploadArray (scene.bvh.nodes(), view_.accel.nodes, error)
            || ! uploadArray (scene.bvh.orderedPositions(), view_.accel.orderedPositions, error)
            || ! uploadArray (scene.bvh.triangleIndices(), view_.accel.triangleIndices, error)
            || ! uploadArray (scene.bvh.orderedClose(), view_.accel.orderedClose, error))
            return false;

        view_.anyPartialCoverage = scene.anyPartialCoverage;
        view_.anyThinSheet = scene.anyThinSheet;
        view_.anyDispersive = scene.anyDispersive;
        view_.anyMediumBoundary = scene.anyMediumBoundary;
        view_.boundsCentre = scene.boundsCentre;
        view_.boundsRadius = scene.boundsRadius;
        pass.stats.phases.set (FramePhase::Upload, millisecondsSince (uploadStart));
        haveScene_ = true;
        sceneUploaded_ = build;
        sceneAddress_ = &pass.scene;
        return true;
    }

    bool ensureFilm (std::uint32_t width, std::uint32_t height, std::string& error)
    {
        if (filmWidth_ == width && filmHeight_ == height)
            return true;
        freeAll (film_);
        filmWidth_ = filmHeight_ = 0;
        const std::size_t bytes = static_cast<std::size_t> (width) * height * sizeof (gpu::FilmPixel);
        const auto allocate = [&] (DeviceAddress& address, std::size_t size) {
            if (! call (shared_.api.memAlloc (&address, size), "allocating the film", error))
                return false;
            film_.push_back (address);
            return true;
        };
        if (! allocate (pixels_, bytes) || ! allocate (splats_, bytes) || ! allocate (guideAlbedo_, bytes)
            || ! allocate (guideNormal_, bytes) || ! allocate (lightPaths_, sizeof (unsigned long long))
            || ! allocate (totals_, sizeof (gpu::LaunchTotals)))
            return false;
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
        if (! ensureFilm (width, height, error))
            return false;
        const DriverApi& api = shared_.api;
        const std::size_t pixelBytes = static_cast<std::size_t> (width) * height * sizeof (gpu::FilmPixel);
        const unsigned long long noPaths = 0;
        const gpu::LaunchTotals noTotals {};
        if (! call (api.memsetD8 (pixels_, 0, pixelBytes), "clearing the film", error)
            || ! call (api.memsetD8 (splats_, 0, pixelBytes), "clearing the film", error)
            || ! call (api.memsetD8 (guideAlbedo_, 0, pixelBytes), "clearing the film", error)
            || ! call (api.memsetD8 (guideNormal_, 0, pixelBytes), "clearing the film", error)
            || ! call (api.memcpyHtoD (lightPaths_, &noPaths, sizeof (noPaths)), "clearing the film", error)
            || ! call (api.memcpyHtoD (totals_, &noTotals, sizeof (noTotals)), "clearing the film", error))
            return false;

        compute::LaunchParams params;
        params.view = view_;
        params.settings = pass.settings;
        params.camera = pass.camera;
        params.film = { reinterpret_cast<gpu::FilmPixel*> (pixels_), reinterpret_cast<gpu::FilmPixel*> (splats_),
                        reinterpret_cast<unsigned long long*> (lightPaths_),
                        reinterpret_cast<gpu::FilmPixel*> (guideAlbedo_), reinterpret_cast<gpu::FilmPixel*> (guideNormal_), width };
        params.totals = reinterpret_cast<gpu::LaunchTotals*> (totals_);
        params.tables = shared_.tables;
        params.height = height;
        params.samples = 1;
        params.generation = pass.generation;
        const unsigned blocksX = (width + compute::kBlockSide - 1) / compute::kBlockSide;
        const unsigned blocksY = (height + compute::kBlockSide - 1) / compute::kBlockSide;
        // A launch per sample, so a cancel stops the frame between samples. The copy to
        // the parameter block waits for the launch before it on the default stream.
        std::uint32_t launched = 0;
        for (std::uint32_t s = 0; s < pass.samples; ++s)
        {
            if (pass.cancel.isCancelled())
            {
                pass.stats.cancelled = true;
                break;
            }
            params.firstSample = pass.firstSample + s;
            if (! call (api.memcpyHtoD (shared_.params, &params, sizeof (params)), "uploading the launch parameters", error)
                || ! call (api.launchKernel (shared_.function, blocksX, blocksY, 1, compute::kBlockSide, compute::kBlockSide, 1,
                                             0, nullptr, nullptr, nullptr), "launching the kernel", error))
                return false;
            ++launched;
        }
        if (launched > 0 && ! call (api.synchronize(), "running the kernel", error))
            return false;
        // The synchronize above already waits for the last sample, so this is all of it.
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

    bool readBack (PathTracer::DevicePass& pass, std::size_t pixelBytes)
    {
        std::string& error = pass.error;
        const DriverApi& api = shared_.api;
        hostPixels_.resize (pixelBytes / sizeof (gpu::FilmPixel));
        hostSplats_.resize (hostPixels_.size());
        hostAlbedo_.resize (hostPixels_.size());
        hostNormal_.resize (hostPixels_.size());
        unsigned long long lightPaths = 0;
        gpu::LaunchTotals totals;
        if (! call (api.memcpyDtoH (hostPixels_.data(), pixels_, pixelBytes), "reading the film back", error)
            || ! call (api.memcpyDtoH (hostSplats_.data(), splats_, pixelBytes), "reading the film back", error)
            || ! call (api.memcpyDtoH (hostAlbedo_.data(), guideAlbedo_, pixelBytes), "reading the guides back", error)
            || ! call (api.memcpyDtoH (hostNormal_.data(), guideNormal_, pixelBytes), "reading the guides back", error)
            || ! call (api.memcpyDtoH (&lightPaths, lightPaths_, sizeof (lightPaths)), "reading the film back", error)
            || ! call (api.memcpyDtoH (&totals, totals_, sizeof (totals)), "reading the film back", error))
            return false;
        const std::uint32_t width = pass.film.width();
        for (std::size_t i = 0; i < hostPixels_.size(); ++i)
        {
            const auto x = static_cast<std::uint32_t> (i % width);
            const auto y = static_cast<std::uint32_t> (i / width);
            const gpu::FilmPixel& pixel = hostPixels_[i];
            if (pixel.w > 0.0f)
                pass.film.addSampleSum (x, y, { pixel.x, pixel.y, pixel.z }, static_cast<std::uint32_t> (pixel.w));
            const gpu::FilmPixel& splat = hostSplats_[i];
            if (splat.x != 0.0f || splat.y != 0.0f || splat.z != 0.0f)
                pass.film.addSplat (x, y, { splat.x, splat.y, splat.z });
            const gpu::FilmPixel& albedo = hostAlbedo_[i];
            const gpu::FilmPixel& normal = hostNormal_[i];
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
        if (shared_.api.memGetInfo != nullptr && shared_.api.memGetInfo (&freeBytes, &totalBytes) == 0)
        {
            pass.stats.deviceMemoryFreeBytes = freeBytes;
            pass.stats.deviceMemoryTotalBytes = totalBytes;
        }
        return true;
    }

    SharedDevice& shared_;
    std::function<std::uint64_t()> sceneBuild_;
    std::vector<DeviceAddress> scene_;
    std::vector<DeviceAddress> film_;
    compute::SceneViewForLaunch view_;
    bool haveScene_ = false;
    std::uint64_t sceneUploaded_ = 0;
    const RenderScene* sceneAddress_ = nullptr;
    std::uint32_t filmWidth_ = 0;
    std::uint32_t filmHeight_ = 0;
    DeviceAddress pixels_ = 0, splats_ = 0, guideAlbedo_ = 0, guideNormal_ = 0, lightPaths_ = 0, totals_ = 0;
    std::vector<gpu::FilmPixel> hostPixels_;
    std::vector<gpu::FilmPixel> hostSplats_;
    std::vector<gpu::FilmPixel> hostAlbedo_;
    std::vector<gpu::FilmPixel> hostNormal_;
};

PathTracer::DevicePassFunction makePass (DeviceKind kind, std::function<std::uint64_t()> sceneBuild)
{
    auto renderer = std::make_shared<ComputeRenderer> (sharedDevice (kind), std::move (sceneBuild));
    return [renderer] (PathTracer::DevicePass& pass) { return renderer->render (pass); };
}

} // namespace

PathTracer::DevicePassFunction makeCudaDevicePass (std::function<std::uint64_t()> sceneBuild)
{
    return makePass (DeviceKind::Cuda, std::move (sceneBuild));
}

PathTracer::DevicePassFunction makeHipDevicePass (std::function<std::uint64_t()> sceneBuild)
{
    return makePass (DeviceKind::Hip, std::move (sceneBuild));
}

PathTracer::DevicePassFunction makeOneApiDevicePass (std::function<std::uint64_t()> sceneBuild)
{
    return makePass (DeviceKind::OneApi, std::move (sceneBuild));
}

bool isComputeDevice (DeviceKind kind) noexcept
{
    return kind == DeviceKind::Cuda || kind == DeviceKind::Hip || kind == DeviceKind::OneApi;
}

bool cudaModuleReady() { return ComputeRenderer::moduleReady (sharedDevice (DeviceKind::Cuda)); }
bool hipModuleReady() { return ComputeRenderer::moduleReady (sharedDevice (DeviceKind::Hip)); }
bool oneApiModuleReady() { return ComputeRenderer::moduleReady (sharedDevice (DeviceKind::OneApi)); }

void startComputeModuleBuild (DeviceKind kind)
{
    SharedDevice& shared = sharedDevice (kind);
    shared.gate->start (ComputeRenderer::buildBody (shared));
}

bool stopComputeModuleBuildByUser (DeviceKind kind)
{
    if (! sharedDevice (kind).gate->stop())
        return false;
    kernelcompile::previewWaiting() = false;
    kernelcompile::previewStopped() = true;
    return true;
}

void startComputeModuleBuildByUser (DeviceKind kind)
{
    SharedDevice& shared = sharedDevice (kind);
    shared.gate->resume();
    kernelcompile::previewStopped() = false;
    shared.gate->start (ComputeRenderer::buildBody (shared));
}

void cancelComputeModuleBuild (DeviceKind kind)
{
    auto& gate = *sharedDevice (kind).gate;
    std::string ignored;
    if (gate.status() == kernelcompile::BuildGate::Status::Running)
        gate.await (ignored);
}

std::string oneApiDeviceProblem()
{
    // Listing SYCL devices starts every runtime adapter, so the answer is kept.
    static const std::string problem = []
    {
        const LoadedLibrary library (kOneApiNames);
        if (library.handle == nullptr)
            return library.problem;
        using Probe = int (*) (char*, std::size_t);
        const auto probe = reinterpret_cast<Probe> (library.find (oneapi::kProbe));
        if (probe == nullptr)
            return library.path + " has no " + oneapi::kProbe;
        char text[512] {};
        if (probe (text, sizeof (text)) == oneapi::kOk)
            return std::string();
        return std::string (text[0] != '\0' ? text : "the oneAPI runtime found no device");
    }();
    return problem;
}

} // namespace videohelper::pathtrace::device
