// The OptiX device the way a converging preview runs it: one sample, the OptiX AI
// denoiser on the same context, then the next sample, up to eight. On 6 Nov 2026 the
// live preview faulted the GPU (Xid 13, SKEDCHECK05_LOCAL_MEMORY_TOTAL_SIZE, then
// CUDA error 719) and Auto moved Path Trace to the CPU.
//
// The cause was the pipeline's stack: 46 KB for every thread the GPU runs at once,
// about 3.1 GB that the driver took at the first launch. With less than that free
// the launch faulted instead of failing. The device now reserves the stack before
// its first launch (device_optix_host.cpp, reserveStack).
//
// Each step logs the context's stack limit and the free device memory. A machine
// without an NVIDIA GPU, or a build without the OptiX kernel, skips by name.
//
// ARBIT_OPTIX_SEQUENCE_SIZE=WxH sets the film (default 960x540).
// ARBIT_OPTIX_SEQUENCE_DENOISE=0 leaves the denoiser out, for an A/B run.
// ARBIT_OPTIX_SEQUENCE_LEAVE_MIB=N fills the GPU from another context until N MiB
// stay free, after the build and before the first launch. Before the fix, N=2000
// reproduced the Xid on an RTX 4070. Now a GPU without room for the stack refuses
// the pass by name with an out-of-memory code and keeps the device usable.
// ARBIT_OPTIX_SEQUENCE_CROWD_FIRST=1 fills it before the build instead.
// ARBIT_OPTIX_SEQUENCE_SQUEEZE=1 fills the GPU from the device's own context after
// the first sample and gives it back, and ARBIT_OPTIX_SEQUENCE_STACK_LIMIT=bytes
// sets the stack limit there, for probing the driver.
// ARBIT_OPTIX_SEQUENCE_EXPORT=N renders an N sample export at the end and logs its
// samples per second, and ARBIT_OPTIX_SEQUENCE_DUMP=path writes its RGBA16F pixels
// there, so two builds' images compare bit for bit.

#include "pathtrace/devices/device_optix_host.h"
#include "pathtrace/devices/device_registry.h"
#include "pathtrace/engine.h"
#include "support/pathtrace_glass_scene.h"

#if ! defined(_WIN32)
#include <dlfcn.h>
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace
{

namespace pt = videohelper::pathtrace;
namespace glass = pathtraceglassscene;

int checks = 0;
int failures = 0;

void check (bool condition, const std::string& message)
{
    ++checks;
    if (! condition)
    {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

// The CUDA calls the log reads, from the driver the device already loaded.
struct CudaProbe
{
    int (*ctxGetCurrent) (void**) = nullptr;
    int (*ctxGetLimit) (std::size_t*, int) = nullptr;
    int (*memGetInfo) (std::size_t*, std::size_t*) = nullptr;
    int (*ctxSetLimit) (int, std::size_t) = nullptr;
    int (*memAlloc) (unsigned long long*, std::size_t) = nullptr;
    int (*memFree) (unsigned long long) = nullptr;
    int (*init) (unsigned int) = nullptr;
    int (*deviceGet) (int*, int) = nullptr;
    int (*ctxCreate) (void**, unsigned int, int) = nullptr;
    int (*ctxSetCurrent) (void*) = nullptr;

    CudaProbe()
    {
#if ! defined(_WIN32)
        if (void* library = dlopen ("libcuda.so.1", RTLD_NOW | RTLD_LOCAL))
        {
            ctxGetCurrent = reinterpret_cast<decltype (ctxGetCurrent)> (dlsym (library, "cuCtxGetCurrent"));
            ctxGetLimit = reinterpret_cast<decltype (ctxGetLimit)> (dlsym (library, "cuCtxGetLimit"));
            memGetInfo = reinterpret_cast<decltype (memGetInfo)> (dlsym (library, "cuMemGetInfo_v2"));
            ctxSetLimit = reinterpret_cast<decltype (ctxSetLimit)> (dlsym (library, "cuCtxSetLimit"));
            memAlloc = reinterpret_cast<decltype (memAlloc)> (dlsym (library, "cuMemAlloc_v2"));
            memFree = reinterpret_cast<decltype (memFree)> (dlsym (library, "cuMemFree_v2"));
            init = reinterpret_cast<decltype (init)> (dlsym (library, "cuInit"));
            deviceGet = reinterpret_cast<decltype (deviceGet)> (dlsym (library, "cuDeviceGet"));
            ctxCreate = reinterpret_cast<decltype (ctxCreate)> (dlsym (library, "cuCtxCreate_v2"));
            ctxSetCurrent = reinterpret_cast<decltype (ctxSetCurrent)> (dlsym (library, "cuCtxSetCurrent"));
        }
#endif
    }

    void log (const char* step) const
    {
        void* context = nullptr;
        std::size_t stack = 0, freeBytes = 0, totalBytes = 0;
        const int current = ctxGetCurrent != nullptr ? ctxGetCurrent (&context) : -1;
        const int limit = context != nullptr && ctxGetLimit != nullptr ? ctxGetLimit (&stack, 0x00) : -1;  // CU_LIMIT_STACK_SIZE
        const int memory = context != nullptr && memGetInfo != nullptr ? memGetInfo (&freeBytes, &totalBytes) : -1;
        std::printf ("[seq] %-14s context=%p(%d) stackLimit=%zu(%d) freeMiB=%zu/%zu(%d)\n", step, context, current, stack, limit,
                     freeBytes >> 20, totalBytes >> 20, memory);
        std::fflush (stdout);
    }

    // Fills the device until an allocation fails, as a busy desktop GPU does, then
    // gives it all back. The log shows whether the driver took memory back from the
    // context's local memory reservation to make an allocation fit.
    void squeeze() const
    {
        if (memAlloc == nullptr || memFree == nullptr)
            return;
        std::vector<unsigned long long> blocks;
        int result = 0;
        while (blocks.size() < 400)
        {
            unsigned long long block = 0;
            result = memAlloc (&block, std::size_t (32) << 20);
            if (result != 0)
                break;
            blocks.push_back (block);
        }
        std::printf ("[seq] squeeze took %zu MiB before CUDA error %d\n", blocks.size() * 32, result);
        log ("squeezed");
        for (const unsigned long long block : blocks)
            memFree (block);
        log ("released");
    }

    // Before the first launch, takes device memory in a context of its own until
    // only leaveMiB stay free, as other apps on a desktop GPU do. Held until the run ends.
    std::vector<unsigned long long> crowd (std::size_t leaveMiB) const
    {
        std::vector<unsigned long long> blocks;
        void* own = nullptr;
        int gpu = 0;
        if (init == nullptr || ctxCreate == nullptr || init (0) != 0 || deviceGet (&gpu, 0) != 0 || ctxCreate (&own, 0, gpu) != 0)
            return blocks;
        log ("before crowd");
        std::size_t freeBytes = 0, totalBytes = 0;
        while (memGetInfo (&freeBytes, &totalBytes) == 0 && (freeBytes >> 20) > leaveMiB + 32)
        {
            unsigned long long block = 0;
            if (memAlloc (&block, std::size_t (32) << 20) != 0)
                break;
            blocks.push_back (block);
        }
        std::printf ("[seq] crowd holds %zu MiB\n", blocks.size() * 32);
        log ("crowded");
        return blocks;
    }
};

std::uint32_t envSize (std::uint32_t& height)
{
    std::uint32_t width = 960;
    height = 540;
    if (const char* text = std::getenv ("ARBIT_OPTIX_SEQUENCE_SIZE"))
        std::sscanf (text, "%ux%u", &width, &height);
    return width;
}

void previewSequence()
{
#if defined(ARBIT_HAVE_OPTIX_KERNEL)
    const auto environment = pt::device::productDeviceEnvironment();
    const auto selection = pt::device::selectDevice ("optix", environment);
    if (! selection.ok)
    {
        std::cout << "SKIP: OptiX does not load here: " << selection.error << '\n';
        return;
    }
    const bool denoise = std::getenv ("ARBIT_OPTIX_SEQUENCE_DENOISE") == nullptr
                      || std::strcmp (std::getenv ("ARBIT_OPTIX_SEQUENCE_DENOISE"), "0") != 0;
    std::uint32_t height = 0;
    const std::uint32_t width = envSize (height);

    const auto scene = glass::makeScene();
    const auto setup = glass::makeSetup (scene);
    const auto view = glass::makeView (*scene, width, height);
    pt::SceneEngine engine (1);
    engine.maximumPreviewSamples = 64;
    engine.setDevicePass (pt::device::makeOptixDevicePass ([&engine] { return engine.sceneBuilds(); }),
                          [] { return pt::device::optixPipelineReady(); });
    const CudaProbe probe;
    std::printf ("[seq] film=%ux%u denoise=%d\n", width, height, denoise ? 1 : 0);

    std::vector<unsigned long long> crowd;
    // CROWD_FIRST takes the memory before the kernel builds; otherwise the crowd comes
    // between the build and the first launch.
    if (const char* leave = std::getenv ("ARBIT_OPTIX_SEQUENCE_LEAVE_MIB"))
    {
        const bool first = std::getenv ("ARBIT_OPTIX_SEQUENCE_CROWD_FIRST") != nullptr;
        if (first)
            crowd = probe.crowd (std::strtoull (leave, nullptr, 10));
        for (int wait = 0; wait < 1200 && ! pt::device::optixPipelineReady(); ++wait)
            std::this_thread::sleep_for (std::chrono::milliseconds (100));
        if (! first)
            crowd = probe.crowd (std::strtoull (leave, nullptr, 10));
    }

    std::vector<float> output;
    for (int sample = 1; sample <= 8; ++sample)
    {
        pt::EngineImage image;
        std::string error;
        bool rendered = false;
        // The first call waits for the kernel build, as the preview does.
        for (int wait = 0; wait < 1200; ++wait)
        {
            image = {};
            rendered = engine.renderPreview (setup, view, pt::EngineColor::LinearRgba16f, image, error);
            if (! rendered || ! image.pending)
                break;
            std::this_thread::sleep_for (std::chrono::milliseconds (100));
        }
        const std::string traced = "trace " + std::to_string (sample);
        probe.log (traced.c_str());
        if (sample == 1 && ! crowd.empty() && image.stats.deviceFailed)
        {
            check (error.find ("reserving") != std::string::npos
                       && error.find ("CUDA error 2") != std::string::npos,
                   "a GPU without room for the stack refuses the pass by name: " + error);
            check (! pt::device::deviceLost(), "the refusal leaves the device usable");
            return;
        }
        check (rendered && ! image.pending && ! image.stats.deviceFailed,
               traced + " ran on OptiX: " + (error.empty() ? std::string (image.stats.deviceFailed ? "device failed" : "pending") : error));
        if (! rendered || image.stats.deviceFailed)
            return;
        check (image.samplesPerPixel == static_cast<std::uint32_t> (sample), traced + " added one sample per pixel");
        std::printf ("[seq] %-14s samplesPerSecond=%.0f\n", traced.c_str(), pt::samplesPerSecond (image.stats.samplesAdded, image.stats.phases));
        if (sample == 1)
        {
            if (const char* limit = std::getenv ("ARBIT_OPTIX_SEQUENCE_STACK_LIMIT"); limit != nullptr && probe.ctxSetLimit != nullptr)
            {
                const int set = probe.ctxSetLimit (0x00, std::strtoull (limit, nullptr, 10));
                std::printf ("[seq] cuCtxSetLimit(CU_LIMIT_STACK_SIZE, %s) = %d\n", limit, set);
                probe.log ("stack limit");
            }
            if (std::getenv ("ARBIT_OPTIX_SEQUENCE_SQUEEZE") != nullptr)
                probe.squeeze();
        }
        if (! denoise)
            continue;

        // The preview denoises the colour with these guides; any finite colour runs the same launches.
        output.assign (image.albedo.size(), 0.0f);
        pt::device::OptixDenoiseImage frame;
        frame.width = image.width;
        frame.height = image.height;
        frame.color = image.albedo.data();
        frame.albedo = image.albedo.data();
        frame.normal = image.normal.data();
        frame.output = output.data();
        std::string failure;
        const bool denoised = pt::device::optixDenoise (frame, failure);
        const std::string step = "denoise " + std::to_string (sample);
        probe.log (step.c_str());
        check (denoised, step + " ran: " + failure);
        if (! denoised)
            return;
    }
    check (! pt::device::deviceLost(), "the context is still usable after eight samples");
    if (const char* samples = std::getenv ("ARBIT_OPTIX_SEQUENCE_EXPORT"))
    {
        pt::EngineImage image;
        std::string error;
        const bool exported = engine.renderExport (setup, view, pt::EngineColor::LinearRgba16f,
                                                   static_cast<std::uint32_t> (std::strtoul (samples, nullptr, 10)), image, error);
        check (exported && ! image.stats.deviceFailed, "the export ran on OptiX: " + error);
        std::printf ("[seq] export %u spp samplesPerSecond=%.0f\n", image.samplesPerPixel,
                     pt::samplesPerSecond (image.stats.samplesAdded, image.stats.phases));
        if (const char* path = std::getenv ("ARBIT_OPTIX_SEQUENCE_DUMP"))
            std::ofstream (path, std::ios::binary).write (reinterpret_cast<const char*> (image.colorBytes.data()),
                                                          static_cast<std::streamsize> (image.colorBytes.size()));
    }
    pt::device::cancelOptixPipelineBuild();
#else
    std::cout << "SKIP: this build has no OptiX kernel\n";
#endif
}

} // namespace

int main()
{
    previewSequence();
    std::cout << checks << " checks, " << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}
