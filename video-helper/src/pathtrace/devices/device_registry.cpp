#include "device_registry.h"

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
#endif

namespace videohelper::pathtrace::device
{

#if defined(ARBIT_HAVE_OPTIX)
std::string optixInitProblem();  // device_optix_host.cpp
#endif

namespace
{

#if defined(_WIN32)
constexpr const char* kOptixRuntimeLibrary = "nvoptix.dll";
constexpr const char* kCudaRuntimeLibrary = "nvcuda.dll";
#else
constexpr const char* kOptixRuntimeLibrary = "libnvoptix.so.1";
constexpr const char* kCudaRuntimeLibrary = "libcuda.so.1";
#endif

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

std::string lowered (const char* text)
{
    std::string result (text != nullptr ? text : "");
    std::transform (result.begin(), result.end(), result.begin(),
                    [] (unsigned char c) { return static_cast<char> (std::tolower (c)); });
    return result;
}

} // namespace

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
            status.problem = std::string ("this build has no ") + deviceName (kind) + " device yet";
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

std::string deviceFailureReason (const std::string& error, bool lost)
{
    // Only the code goes in the brackets when there is one: an OptiX compile log runs long.
    std::string detail = error;
    for (const char* const prefix : { "Path Trace device failed: ", "OptiX device: " })
        if (detail.rfind (prefix, 0) == 0)
            detail.erase (0, std::strlen (prefix));
    if (const auto code = detail.find ("OPTIX_ERROR_"); code != std::string::npos)
        detail = detail.substr (code, detail.find (' ', code) - code);
    else if (const auto cuda = detail.find ("CUDA error "); cuda != std::string::npos)
        detail = detail.substr (cuda);
    return (lost ? "because the GPU stopped responding. Restart DonutStudio to try the GPU again. (OptiX: "
                 : "because the GPU renderer could not start. Switch Render Engine away from Path Trace and back to try the GPU again. (OptiX: ")
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
    return environment;
}

} // namespace videohelper::pathtrace::device
