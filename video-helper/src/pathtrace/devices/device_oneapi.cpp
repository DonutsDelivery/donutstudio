// oneAPI device library (Plan 3, Render engines and devices): the integrator body
// as a SYCL kernel for Intel GPUs, and the driver entries device_oneapi_abi.h
// lists. icpx -fsycl builds this file alone into libarbit-pathtrace-oneapi; the
// helper loads it at run time, and the SYCL runtime it links loads Level Zero.
//
// The kernel is the CUDA and HIP one: the same device struct
// (device_compute_device.h), launch parameters and film, over the CPU build of
// the BVH. The shared code picks its device paths on __CUDACC__, as for HIP, so
// this file defines it after the SYCL headers and supplies the few CUDA names
// those paths use (atomicAdd, __float_as_uint, __uint_as_float) over SYCL.

#define PATHTRACE_DEVICE_ONEAPI 1

#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <math.h>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if ! defined(__CUDACC__)
#define __CUDACC__ 1
#endif
#define __host__
#define __device__
#define __constant__
#define __forceinline__ inline __attribute__ ((always_inline))

inline unsigned int __float_as_uint (float value) { return sycl::bit_cast<unsigned int> (value); }
inline float __uint_as_float (unsigned int bits) { return sycl::bit_cast<float> (bits); }

template <typename T>
inline T atomicAdd (T* address, T value)
{
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> ref (*address);
    return ref.fetch_add (value);
}

#include "device_compute_device.h"
#include "device_compute_params.h"
#include "device_oneapi_abi.h"

namespace arbitoneapi
{
namespace device = ::videohelper::pathtrace::device;
// The launch's lookup tables, which the shared code reads through
// deviceLookupTables(). It points into the parameter block.
sycl::ext::oneapi::experimental::device_global<const device::DeviceLookupTables*> tables;
} // namespace arbitoneapi

namespace videohelper::pathtrace::device
{
PT_SHARED_ENTRY const DeviceLookupTables& deviceLookupTables() noexcept
{
    return *arbitoneapi::tables.get();
}
} // namespace videohelper::pathtrace::device

// One TU, so the out-of-line shared code comes in as source, as for CUDA.
#include "../spectrum.cpp"
#include "../bsdf.cpp"
#include "../environment_map.cpp"
#include "../lights.cpp"

#define PT_DEVICE ::videohelper::pathtrace::device::ComputeDevice
#define PT_DEVICE_NAMESPACE compute
#include "integrator_body.inl"

class ArbitPathTraceOneApi;

namespace arbitoneapi
{
namespace pt = ::videohelper::pathtrace;
namespace compute = pt::device::compute;
namespace abi = pt::device::oneapi;
using DeviceAddress = unsigned long long;

// The CUDA kernel's body, for one pixel.
inline void renderPixel (const compute::LaunchParams& params, std::uint32_t x, std::uint32_t y)
{
    if (x >= params.film.width || y >= params.height)
        return;
    pt::device::gpu::Film film = params.film;
    pt::PathCounters counters;
    unsigned long long added = 0, nonFinite = 0;
    for (std::uint32_t s = 0; s < params.samples; ++s)
    {
        const auto outcome = compute::renderSample (params.view, params.settings, params.camera, x, y, params.film.width,
                                                    params.firstSample + s, params.generation, film, counters);
        if (outcome == pt::device::SampleOutcome::NonFinite)
            ++nonFinite;
        else
            ++added;
    }
    pt::device::gpu::addTotals (params.totals, counters, added, nonFinite);
}

thread_local std::string lastError;
std::string asyncError;

int failed (int code, std::string message)
{
    lastError = std::move (message);
    return code;
}

bool wantsCpu()
{
    const char* const type = std::getenv ("ARBIT_ONEAPI_DEVICE_TYPE");
    return type != nullptr && (std::strcmp (type, "cpu") == 0 || std::strcmp (type, "CPU") == 0);
}

constexpr std::uint32_t kIntelVendor = 0x8086;

// An Intel GPU through Level Zero, else through OpenCL. Another vendor's OpenCL GPU
// cannot run the SPIR-V this library carries, so it never counts.
std::optional<sycl::device> chooseDevice (std::string& why)
{
    std::vector<sycl::device> devices;
    try
    {
        devices = sycl::device::get_devices();
    }
    catch (const sycl::exception& error)
    {
        why = std::string ("the SYCL runtime could not list devices: ") + error.what();
        return std::nullopt;
    }
    const bool cpu = wantsCpu();
    std::optional<sycl::device> chosen;
    int rank = 0;
    std::string seen;
    for (const auto& candidate : devices)
    {
        const auto name = candidate.get_info<sycl::info::device::name>();
        seen += (seen.empty() ? "" : ", ") + name;
        if (candidate.get_info<sycl::info::device::vendor_id>() != kIntelVendor)
            continue;
        int candidateRank = 0;
        if (candidate.is_gpu())
            candidateRank = candidate.get_backend() == sycl::backend::ext_oneapi_level_zero ? 3 : 2;
        else if (cpu && candidate.is_cpu())
            candidateRank = 1;
        if (candidateRank > rank)
        {
            rank = candidateRank;
            chosen = candidate;
        }
    }
    if (! chosen)
        why = std::string ("the oneAPI runtime found no Intel GPU; install Intel's GPU driver with Level Zero")
            + (seen.empty() ? " (it found no devices)" : " (it found " + seen + ")");
    return chosen;
}

void handleAsync (sycl::exception_list errors);

struct Context
{
    explicit Context (const sycl::device& chosen)
        : gpu (chosen), queue (chosen, handleAsync, { sycl::property::queue::in_order() }) {}

    sycl::device gpu;
    sycl::queue queue;
    std::optional<sycl::kernel_bundle<sycl::bundle_state::executable>> bundle;
    DeviceAddress params = 0;
};

std::optional<sycl::device> initialised;
Context* current = nullptr;

void handleAsync (sycl::exception_list errors)
{
    for (const auto& error : errors)
    {
        try
        {
            std::rethrow_exception (error);
        }
        catch (const sycl::exception& e)
        {
            asyncError = e.what();
        }
    }
}

template <typename Body>
int guarded (Body&& body)
{
    try
    {
        return body();
    }
    catch (const sycl::exception& error)
    {
        const bool memory = error.code() == sycl::errc::memory_allocation;
        return failed (memory ? abi::kOutOfMemory : abi::kFailed, error.what());
    }
    catch (const std::exception& error)
    {
        return failed (abi::kFailed, error.what());
    }
}

int needContext()
{
    return current == nullptr ? failed (abi::kNotInitialised, "no oneAPI context is current") : abi::kOk;
}

void* devicePointer (DeviceAddress address) { return reinterpret_cast<void*> (static_cast<std::uintptr_t> (address)); }

} // namespace arbitoneapi

#if defined(_WIN32)
#define ARBIT_ONEAPI_EXPORT extern "C" __declspec (dllexport)
#else
#define ARBIT_ONEAPI_EXPORT extern "C" __attribute__ ((visibility ("default")))
#endif

using arbitoneapi::DeviceAddress;
namespace abi = ::videohelper::pathtrace::device::oneapi;

ARBIT_ONEAPI_EXPORT const char* arbitOneApiLastError()
{
    return arbitoneapi::lastError.c_str();
}

ARBIT_ONEAPI_EXPORT int arbitOneApiProbe (char* text, std::size_t size)
{
    return arbitoneapi::guarded ([&] () -> int {
        std::string why;
        const auto chosen = arbitoneapi::chooseDevice (why);
        const std::string message = chosen ? chosen->get_info<sycl::info::device::name>() : why;
        if (text != nullptr && size > 0)
        {
            const std::size_t length = std::min (message.size(), size - 1);
            std::memcpy (text, message.data(), length);
            text[length] = '\0';
        }
        return chosen ? abi::kOk : abi::kNoDevice;
    });
}

ARBIT_ONEAPI_EXPORT int arbitOneApiInit (unsigned int)
{
    return arbitoneapi::guarded ([] () -> int {
        if (arbitoneapi::initialised)
            return abi::kOk;
        std::string why;
        arbitoneapi::initialised = arbitoneapi::chooseDevice (why);
        return arbitoneapi::initialised ? abi::kOk : arbitoneapi::failed (abi::kNoDevice, why);
    });
}

ARBIT_ONEAPI_EXPORT int arbitOneApiDeviceGet (int* ordinal, int index)
{
    if (! arbitoneapi::initialised)
        return arbitoneapi::failed (abi::kNotInitialised, "the oneAPI device is not initialised");
    if (index != 0)
        return arbitoneapi::failed (abi::kNoDevice, "the oneAPI device has one GPU");
    *ordinal = 0;
    return abi::kOk;
}

ARBIT_ONEAPI_EXPORT int arbitOneApiCtxCreate (void** context, unsigned int, int)
{
    return arbitoneapi::guarded ([&] () -> int {
        if (! arbitoneapi::initialised)
            return arbitoneapi::failed (abi::kNotInitialised, "the oneAPI device is not initialised");
        auto made = std::make_unique<arbitoneapi::Context> (*arbitoneapi::initialised);
        arbitoneapi::current = made.get();
        *context = made.release();
        return abi::kOk;
    });
}

ARBIT_ONEAPI_EXPORT int arbitOneApiCtxSetCurrent (void* context)
{
    arbitoneapi::current = static_cast<arbitoneapi::Context*> (context);
    return abi::kOk;
}

ARBIT_ONEAPI_EXPORT int arbitOneApiCtxDestroy (void* context)
{
    return arbitoneapi::guarded ([&] () -> int {
        auto* owned = static_cast<arbitoneapi::Context*> (context);
        if (arbitoneapi::current == owned)
            arbitoneapi::current = nullptr;
        if (owned != nullptr)
        {
            owned->queue.wait();
            if (owned->params != 0)
                sycl::free (arbitoneapi::devicePointer (owned->params), owned->queue);
        }
        delete owned;
        return abi::kOk;
    });
}

ARBIT_ONEAPI_EXPORT int arbitOneApiMemAlloc (DeviceAddress* address, std::size_t size)
{
    if (const int problem = arbitoneapi::needContext(); problem != abi::kOk)
        return problem;
    return arbitoneapi::guarded ([&] () -> int {
        void* const memory = sycl::malloc_device (size, arbitoneapi::current->queue);
        if (memory == nullptr)
            return arbitoneapi::failed (abi::kOutOfMemory, "the GPU has no room for " + std::to_string (size) + " bytes");
        *address = reinterpret_cast<std::uintptr_t> (memory);
        return abi::kOk;
    });
}

ARBIT_ONEAPI_EXPORT int arbitOneApiMemFree (DeviceAddress address)
{
    if (const int problem = arbitoneapi::needContext(); problem != abi::kOk)
        return problem;
    return arbitoneapi::guarded ([&] () -> int {
        arbitoneapi::current->queue.wait();
        sycl::free (arbitoneapi::devicePointer (address), arbitoneapi::current->queue);
        return abi::kOk;
    });
}

// Copies wait, as CUDA's pageable copies do: the caller reuses its buffer at once.
ARBIT_ONEAPI_EXPORT int arbitOneApiMemcpyHtoD (DeviceAddress destination, const void* source, std::size_t size)
{
    if (const int problem = arbitoneapi::needContext(); problem != abi::kOk)
        return problem;
    return arbitoneapi::guarded ([&] () -> int {
        arbitoneapi::current->queue.memcpy (arbitoneapi::devicePointer (destination), source, size).wait_and_throw();
        return abi::kOk;
    });
}

ARBIT_ONEAPI_EXPORT int arbitOneApiMemcpyDtoH (void* destination, DeviceAddress source, std::size_t size)
{
    if (const int problem = arbitoneapi::needContext(); problem != abi::kOk)
        return problem;
    return arbitoneapi::guarded ([&] () -> int {
        arbitoneapi::current->queue.memcpy (destination, arbitoneapi::devicePointer (source), size).wait_and_throw();
        return abi::kOk;
    });
}

ARBIT_ONEAPI_EXPORT int arbitOneApiMemsetD8 (DeviceAddress destination, unsigned char value, std::size_t size)
{
    if (const int problem = arbitoneapi::needContext(); problem != abi::kOk)
        return problem;
    return arbitoneapi::guarded ([&] () -> int {
        arbitoneapi::current->queue.memset (arbitoneapi::devicePointer (destination), value, size).wait_and_throw();
        return abi::kOk;
    });
}

// The driver compiles the kernel's SPIR-V for the GPU here, on the host's module
// thread, and caches it.
ARBIT_ONEAPI_EXPORT int arbitOneApiModuleLoadData (void** module, const void* image)
{
    if (const int problem = arbitoneapi::needContext(); problem != abi::kOk)
        return problem;
    if (image == nullptr || std::strcmp (static_cast<const char*> (image), abi::kModuleImage) != 0)
        return arbitoneapi::failed (abi::kWrongImage, std::string ("the helper and the oneAPI library come from different builds (the library wants ")
                                                          + abi::kModuleImage + ")");
    return arbitoneapi::guarded ([&] () -> int {
        auto& context = *arbitoneapi::current;
        context.bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable> (
            context.queue.get_context(), { context.gpu }, { sycl::get_kernel_id<ArbitPathTraceOneApi>() });
        *module = &context;
        return abi::kOk;
    });
}

ARBIT_ONEAPI_EXPORT int arbitOneApiModuleUnload (void* module)
{
    if (auto* context = static_cast<arbitoneapi::Context*> (module); context != nullptr)
        context->bundle.reset();
    return abi::kOk;
}

ARBIT_ONEAPI_EXPORT int arbitOneApiModuleGetFunction (void** function, void* module, const char* name)
{
    if (module == nullptr || std::strcmp (name, arbitoneapi::compute::kKernelName) != 0)
        return arbitoneapi::failed (abi::kUnknownName, std::string ("the oneAPI library has no kernel ") + name);
    *function = module;
    return abi::kOk;
}

// The parameter block lives in device memory; the device_global the shared code
// reads its tables through points into it.
ARBIT_ONEAPI_EXPORT int arbitOneApiModuleGetGlobal (DeviceAddress* address, std::size_t* size, void* module, const char* name)
{
    if (module == nullptr || std::strcmp (name, arbitoneapi::compute::kParamsName) != 0)
        return arbitoneapi::failed (abi::kUnknownName, std::string ("the oneAPI library has no global ") + name);
    return arbitoneapi::guarded ([&] () -> int {
        auto& context = *static_cast<arbitoneapi::Context*> (module);
        if (context.params == 0)
        {
            void* const block = sycl::malloc_device (sizeof (arbitoneapi::compute::LaunchParams), context.queue);
            if (block == nullptr)
                return arbitoneapi::failed (abi::kOutOfMemory, "the GPU has no room for the launch parameters");
            context.params = reinterpret_cast<std::uintptr_t> (block);
            const auto* tables = reinterpret_cast<const ::videohelper::pathtrace::device::DeviceLookupTables*> (
                static_cast<const char*> (block) + offsetof (arbitoneapi::compute::LaunchParams, tables));
            context.queue.memcpy (arbitoneapi::tables, &tables).wait_and_throw();
        }
        *address = context.params;
        *size = sizeof (arbitoneapi::compute::LaunchParams);
        return abi::kOk;
    });
}

ARBIT_ONEAPI_EXPORT int arbitOneApiLaunchKernel (void* function, unsigned int gridX, unsigned int gridY, unsigned int,
                                                 unsigned int blockX, unsigned int blockY, unsigned int, unsigned int,
                                                 void*, void**, void**)
{
    if (function == nullptr)
        return arbitoneapi::failed (abi::kUnknownName, "no oneAPI kernel to launch");
    return arbitoneapi::guarded ([&] () -> int {
        auto& context = *static_cast<arbitoneapi::Context*> (function);
        const auto* params = reinterpret_cast<const arbitoneapi::compute::LaunchParams*> (static_cast<std::uintptr_t> (context.params));
        const sycl::nd_range<2> range ({ std::size_t (gridY) * blockY, std::size_t (gridX) * blockX }, { blockY, blockX });
        context.queue.submit ([&] (sycl::handler& handler) {
            if (context.bundle)
                handler.use_kernel_bundle (*context.bundle);
            handler.parallel_for<ArbitPathTraceOneApi> (range, [params] (sycl::nd_item<2> item) {
                arbitoneapi::renderPixel (*params, static_cast<std::uint32_t> (item.get_global_id (1)),
                                          static_cast<std::uint32_t> (item.get_global_id (0)));
            });
        });
        return abi::kOk;
    });
}

ARBIT_ONEAPI_EXPORT int arbitOneApiSynchronize()
{
    if (const int problem = arbitoneapi::needContext(); problem != abi::kOk)
        return problem;
    return arbitoneapi::guarded ([] () -> int {
        arbitoneapi::asyncError.clear();
        arbitoneapi::current->queue.wait_and_throw();
        if (! arbitoneapi::asyncError.empty())
            return arbitoneapi::failed (abi::kFailed, arbitoneapi::asyncError);
        return abi::kOk;
    });
}

// Free memory needs the GPU's sysman (ZES_ENABLE_SYSMAN=1); without it the host
// leaves the readout empty.
ARBIT_ONEAPI_EXPORT int arbitOneApiMemGetInfo (std::size_t* freeBytes, std::size_t* totalBytes)
{
    if (const int problem = arbitoneapi::needContext(); problem != abi::kOk)
        return problem;
    return arbitoneapi::guarded ([&] () -> int {
        const auto& gpu = arbitoneapi::current->gpu;
        if (! gpu.has (sycl::aspect::ext_intel_free_memory))
            return arbitoneapi::failed (abi::kFailed, "the GPU does not report free memory");
        *totalBytes = gpu.get_info<sycl::info::device::global_mem_size>();
        *freeBytes = gpu.get_info<sycl::ext::intel::info::device::free_memory>();
        return abi::kOk;
    });
}
