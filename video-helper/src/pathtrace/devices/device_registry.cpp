#include "device_registry.h"
#include "device_oneapi_abi.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <cctype>
#include <utility>

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
#include <unistd.h>
#endif

namespace videohelper::pathtrace::device
{

#if defined(ARBIT_HAVE_OPTIX)
std::string optixInitProblem();  // device_optix_host.cpp
#endif
#if defined(ARBIT_HAVE_ONEAPI_DEVICE)
std::string oneApiDeviceProblem();  // device_compute_host.cpp
#endif
#if defined(ARBIT_HAVE_METAL_PATHTRACE)
std::string metalDeviceProblem();  // device_metal_host.mm
#endif

namespace
{

#if defined(_WIN32)
constexpr const char* kOptixRuntimeLibrary = "nvoptix.dll";
constexpr const char* kCudaRuntimeLibrary = "nvcuda.dll";
constexpr const char* kHipRuntimeLibrary = "amdhip64.dll";
#else
constexpr const char* kOptixRuntimeLibrary = "libnvoptix.so.1";
constexpr const char* kCudaRuntimeLibrary = "libcuda.so.1";
constexpr const char* kHipRuntimeLibrary = "libamdhip64.so";
#endif
constexpr const char* kMetalFramework = "/System/Library/Frameworks/Metal.framework/Metal";

std::string optixProblem (const DeviceEnvironment& environment)
{
    if (! environment.optixSdkBuilt)
        return "this build of DonutStudio has no OptiX device";
    if (environment.loadProblem)
    {
        // The OptiX entry point the SDK's own loader looks up in the driver.
        auto problem = environment.loadProblem (kOptixRuntimeLibrary, "optixQueryFunctionTable");
        if (problem.empty())
            problem = environment.loadProblem (kCudaRuntimeLibrary, "cuInit");
        if (! problem.empty())
            return problem;
    }
    if (environment.optixInitProblem)
        if (auto problem = environment.optixInitProblem(); ! problem.empty())
            return problem;
    if (! environment.optixKernelBuilt)
        return "this build of DonutStudio has no OptiX kernel";
    return {};
}

std::string cudaProblem (const DeviceEnvironment& environment)
{
    if (! environment.cudaKernelBuilt)
        return "this build of DonutStudio has no CUDA kernel (it was built without nvcc)";
    if (environment.loadProblem)
        return environment.loadProblem (kCudaRuntimeLibrary, "cuInit");
    return {};
}

std::string hipProblem (const DeviceEnvironment& environment)
{
    if (! environment.hipKernelBuilt)
        return "this build of DonutStudio has no HIP kernel (it was built without ROCm's hipcc)";
    if (environment.loadProblem)
        return environment.loadProblem (kHipRuntimeLibrary, "hipInit");
    return {};
}

std::string metalProblem (const DeviceEnvironment& environment)
{
    if (! environment.applePlatform)
        return "Metal runs on macOS only";
    if (! environment.metalDeviceBuilt)
        return "this build of DonutStudio has no Metal device (it needs the macOS Metal toolchain)";
    if (environment.loadProblem)
        if (auto problem = environment.loadProblem (kMetalFramework, "MTLCreateSystemDefaultDevice"); ! problem.empty())
            return problem;
    if (environment.metalDeviceProblem)
        return environment.metalDeviceProblem();
    return {};
}

// The device library beside the helper links the SYCL runtime, which loads Level
// Zero, so a failed load names whichever of them is missing.
std::string oneApiProblem (const DeviceEnvironment& environment)
{
    if (! environment.oneApiDeviceBuilt)
        return "this build of DonutStudio has no oneAPI device (it was built without Intel's oneAPI DPC++ compiler)";
    if (environment.loadProblem)
        if (auto problem = environment.loadProblem (besideHelperPath (oneapi::kLibraryFile).c_str(), oneapi::kProbe); ! problem.empty())
            return withOneApiRuntimeHint (std::move (problem));
    if (environment.oneApiDeviceProblem)
        return environment.oneApiDeviceProblem();
    return {};
}

std::string lowered (const char* text)
{
    std::string result (text != nullptr ? text : "");
    std::transform (result.begin(), result.end(), result.begin(),
                    [] (unsigned char c) { return static_cast<char> (std::tolower (c)); });
    return result;
}

} // namespace

std::string withOneApiRuntimeHint (std::string problem)
{
    // The libraries icpx's runtime brings: SYCL, the Unified Runtime and its
    // adapters, Level Zero, OpenCL and the compiler's math and support libraries.
    static constexpr const char* kRuntimeNames[] = { "sycl", "ur_loader", "ur_adapter", "ze_loader", "opencl",
                                                     "svml", "libimf", "intlc", "irng", "umf" };
    const auto text = lowered (problem.c_str());
    for (const char* name : kRuntimeNames)
        if (text.find (name) != std::string::npos)
            return problem + ". DonutStudio does not include Intel's oneAPI DPC++ runtime: install the Intel oneAPI DPC++/C++ "
                             "runtime (Intel's intel-oneapi-runtime-dpcpp-cpp package or installer, from intel.com/oneapi) "
                             "and an Intel GPU driver with Level Zero, then start DonutStudio again";
    return problem;
}

const char* deviceName (DeviceKind kind) noexcept
{
    switch (kind)
    {
        case DeviceKind::OptiX:  return "OptiX";
        case DeviceKind::Cuda:   return "CUDA";
        case DeviceKind::Hip:    return "HIP";
        case DeviceKind::Metal:  return "Metal";
        case DeviceKind::OneApi: return "oneAPI";
        case DeviceKind::Cpu:    break;
    }
    return "CPU";
}

std::vector<DeviceStatus> probeDevices (const DeviceEnvironment& environment)
{
    std::vector<DeviceStatus> devices;
    devices.reserve (kAutoOrder.size());
    for (const auto kind : kAutoOrder)
    {
        DeviceStatus status;
        status.kind = kind;
        if (kind == DeviceKind::Cpu)
            status.loadable = true;
        else if (kind == DeviceKind::OptiX)
        {
            status.problem = optixProblem (environment);
            status.loadable = status.problem.empty();
        }
        else
        {
            status.problem = kind == DeviceKind::Cuda ? cudaProblem (environment)
                           : kind == DeviceKind::Hip  ? hipProblem (environment)
                           : kind == DeviceKind::Metal ? metalProblem (environment)
                                                       : oneApiProblem (environment);
            status.loadable = status.problem.empty();
        }
        devices.push_back (std::move (status));
    }
    return devices;
}

DeviceSelection selectDevice (const char* text, const DeviceEnvironment& environment)
{
    DeviceSelection selection;
    const auto name = lowered (text);
    const bool automatic = name.empty() || name == "auto";

    bool known = automatic;
    DeviceKind named = DeviceKind::Cpu;
    for (const auto kind : kAutoOrder)
        if (name == lowered (deviceName (kind)))
        {
            known = true;
            named = kind;
        }
    if (name == "none")
        known = true;  // Cycles' panel calls the CPU device None
    if (! known)
    {
        selection.error = "Path Trace device \"" + std::string (text) + "\" is unknown; use Auto, OptiX, CUDA, HIP, Metal, oneAPI or None";
        return selection;
    }

    selection.devices = probeDevices (environment);
    if (automatic)
    {
        const auto first = std::find_if (selection.devices.begin(), selection.devices.end(),
                                         [&environment] (const DeviceStatus& status) {
                                             return status.loadable && (status.kind != DeviceKind::OptiX || environment.optixInAuto);
                                         });
        selection.ok = true;
        selection.device = first->kind;  // CPU is always loadable
        return selection;
    }

    selection.requested = deviceName (named);
    const auto status = std::find_if (selection.devices.begin(), selection.devices.end(),
                                      [named] (const DeviceStatus& entry) { return entry.kind == named; });
    if (! status->loadable)
    {
        selection.error = "Path Trace device " + std::string (deviceName (named)) + " is unavailable: " + status->problem;
        return selection;
    }
    selection.ok = true;
    selection.device = named;
    return selection;
}

const char* deviceSettingValue (DeviceKind kind) noexcept
{
    switch (kind)
    {
        case DeviceKind::OptiX:  return "optix";
        case DeviceKind::Cuda:   return "cuda";
        case DeviceKind::Hip:    return "hip";
        case DeviceKind::Metal:  return "metal";
        case DeviceKind::OneApi: return "oneapi";
        case DeviceKind::Cpu:    break;
    }
    return "cpu";
}

const char* effectiveDeviceSetting (const char* environmentOverride, const char* preference) noexcept
{
    if (environmentOverride != nullptr && environmentOverride[0] != '\0')
        return environmentOverride;
    return preference;
}

std::string besideHelperPath (const char* file)
{
#if defined(_WIN32)
    char path[MAX_PATH] {};
    const DWORD length = GetModuleFileNameA (nullptr, path, MAX_PATH);
    std::string executable (path, length);
    const auto slash = executable.find_last_of ("\\/");
#elif defined(__linux__)
    char path[4096] {};
    const auto length = readlink ("/proc/self/exe", path, sizeof (path) - 1);
    std::string executable (path, length > 0 ? static_cast<std::size_t> (length) : 0);
    const auto slash = executable.rfind ('/');
#else
    std::string executable;
    const auto slash = std::string::npos;
#endif
    if (slash == std::string::npos)
        return file;
    return executable.substr (0, slash + 1) + file;
}

std::string libraryLoadProblem (const char* library, const char* symbol)
{
#if defined(_WIN32)
    const auto handle = LoadLibraryA (library);
    if (handle == nullptr)
        return std::string (library) + " did not load (Windows error " + std::to_string (GetLastError()) + ")";
    if (GetProcAddress (handle, symbol) == nullptr)
        return std::string (library) + " has no " + symbol;
#else
    void* const handle = dlopen (library, RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr)
    {
        const char* const reason = dlerror();
        return std::string (library) + " did not load: " + (reason != nullptr ? reason : "no reason given");
    }
    if (dlsym (handle, symbol) == nullptr)
        return std::string (library) + " has no " + symbol;
#endif
    return {};
}

bool cudaErrorLosesDevice (int code) noexcept
{
    // ECC_UNCORRECTABLE, ILLEGAL_ADDRESS, LAUNCH_TIMEOUT, CONTEXT_IS_DESTROYED,
    // LAUNCH_FAILED, UNKNOWN.
    return code == 214 || code == 700 || code == 702 || code == 709 || code == 719 || code == 999;
}

namespace
{
std::atomic<bool> lostDevice { false };
}

void noteDeviceLost() noexcept { lostDevice.store (true); }
bool deviceLost() noexcept { return lostDevice.load(); }

std::string autoPassedOverReason (const DeviceSelection& selection)
{
    if (! selection.ok || selection.device != DeviceKind::Cpu || std::strcmp (selection.requested, "Auto") != 0)
        return {};
    for (const auto& status : selection.devices)
        if (! status.loadable)
            return "because the GPU renderer could not start. (" + std::string (deviceName (status.kind)) + ": " + status.problem + ")";
    return {};
}

std::string deviceFailureReason (const std::string& error, bool lost, DeviceKind kind)
{
    // Only the code goes in the brackets when there is one: an OptiX compile log runs long.
    std::string detail = error;
    for (const char* const prefix : { "Path Trace device failed: ", "OptiX device: ", "CUDA device: ", "HIP device: ",
                                      "oneAPI device: ", "Metal device: " })
        if (detail.rfind (prefix, 0) == 0)
            detail.erase (0, std::strlen (prefix));
    if (const auto code = detail.find ("OPTIX_ERROR_"); code != std::string::npos)
        detail = detail.substr (code, detail.find (' ', code) - code);
    else if (const auto cuda = detail.find ("CUDA error "); cuda != std::string::npos)
        detail = detail.substr (cuda);
    else if (const auto hip = detail.find ("HIP error "); hip != std::string::npos)
        detail = detail.substr (hip);
    else if (const auto oneApi = detail.find ("oneAPI error "); oneApi != std::string::npos)
        detail = detail.substr (oneApi);
    const std::string device = std::string (" (") + deviceName (kind) + ": ";
    return (lost ? "because the GPU stopped responding. Restart DonutStudio to try the GPU again." + device
                 : "because the GPU renderer could not start. Switch Render Engine away from Path Trace and back to try the GPU again." + device)
        + detail + ")";
}

DeviceEnvironment productDeviceEnvironment()
{
    DeviceEnvironment environment;
    environment.loadProblem = libraryLoadProblem;
#if defined(ARBIT_HAVE_OPTIX)
    environment.optixSdkBuilt = true;
    environment.optixInitProblem = optixInitProblem;
#endif
#if defined(ARBIT_HAVE_OPTIX_KERNEL)
    environment.optixKernelBuilt = true;
#endif
#if defined(ARBIT_HAVE_CUDA_KERNEL)
    environment.cudaKernelBuilt = true;
#endif
#if defined(ARBIT_HAVE_HIP_KERNEL)
    environment.hipKernelBuilt = true;
#endif
#if defined(__APPLE__)
    environment.applePlatform = true;
#endif
#if defined(ARBIT_HAVE_METAL_PATHTRACE)
    environment.metalDeviceBuilt = true;
    environment.metalDeviceProblem = metalDeviceProblem;
#endif
#if defined(ARBIT_HAVE_ONEAPI_DEVICE)
    environment.oneApiDeviceBuilt = true;
    environment.oneApiDeviceProblem = oneApiDeviceProblem;
#endif
    return environment;
}

} // namespace videohelper::pathtrace::device
