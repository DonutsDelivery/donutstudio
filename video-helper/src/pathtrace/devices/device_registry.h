#pragma once

// The Path Trace device list (Plan 3): which devices exist, which of them can
// run on this machine, and which one a request selects.
//
// The device is a machine setting, Auto or a named device. Auto takes the first
// loadable device in kAutoOrder, and CPU always loads. A named device that does
// not load is an error the user sees, never a silent switch to another device.
//
// A device is loadable when this build holds its code and the machine holds its
// runtime. Vendor runtimes load at run time, so nothing here links against them.

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace videohelper::pathtrace::device
{

enum class DeviceKind : std::uint8_t
{
    OptiX = 0,
    Cuda,
    Hip,
    Metal,
    OneApi,
    Cpu
};

inline constexpr std::array<DeviceKind, 6> kAutoOrder { DeviceKind::OptiX, DeviceKind::Cuda, DeviceKind::Hip,
                                                        DeviceKind::Metal, DeviceKind::OneApi, DeviceKind::Cpu };

// "OptiX", "CUDA", "HIP", "Metal", "oneAPI" or "CPU". The HUD and the stats reply show it.
const char* deviceName (DeviceKind kind) noexcept;

// What a device's availability depends on, so tests can stand in a machine.
struct DeviceEnvironment
{
    // This build compiled the OptiX host code against an SDK (ARBIT_HAVE_OPTIX).
    bool optixSdkBuilt = false;
    // This build holds the OptiX integrator kernel and a path that launches it.
    bool optixKernelBuilt = false;
    // Auto may pick OptiX. An explicit request can use it either way.
    bool optixInAuto = true;
    // This build holds the compute kernel nvcc made (ARBIT_HAVE_CUDA_KERNEL), and the
    // one hipcc made (ARBIT_HAVE_HIP_KERNEL).
    bool cudaKernelBuilt = false;
    bool hipKernelBuilt = false;
    // The machine is a Mac, where Metal runs.
    bool applePlatform = false;
    // This build holds the Metal device (ARBIT_HAVE_METAL_PATHTRACE, macOS) and the
    // oneAPI device library icpx made (ARBIT_HAVE_ONEAPI_DEVICE).
    bool metalDeviceBuilt = false;
    bool oneApiDeviceBuilt = false;
    // Empty when the device finds a GPU it can use, otherwise why not. Unset where
    // the device is not built.
    std::function<std::string()> metalDeviceProblem;
    std::function<std::string()> oneApiDeviceProblem;
    // Empty when library loads and exports symbol, otherwise what went wrong.
    std::function<std::string (const char* library, const char* symbol)> loadProblem;
    // Empty when the driver accepts the SDK's ABI. Unset without an SDK.
    std::function<std::string()> optixInitProblem;
};

// A failed load of the oneAPI device library with, when the loader names one of
// Intel's runtime libraries as missing, which runtime that is and where to get it:
// DonutStudio ships its own device library but not Intel's DPC++ runtime.
std::string withOneApiRuntimeHint (std::string problem);

struct DeviceStatus
{
    DeviceKind kind = DeviceKind::Cpu;
    bool loadable = false;
    std::string problem;  // why not, naming the missing build or library
};

// One status per device, in kAutoOrder.
std::vector<DeviceStatus> probeDevices (const DeviceEnvironment& environment);

struct DeviceSelection
{
    bool ok = false;
    const char* requested = "Auto";     // "Auto" or a deviceName
    DeviceKind device = DeviceKind::Cpu;  // valid when ok
    std::string error;                  // for the user, when not ok
    std::vector<DeviceStatus> devices;
};

// text is the setting's value, matched without regard to case: empty or null or
// "auto", "none" or "cpu", "optix", "cuda", "hip", "metal", "oneapi". An unknown
// name is an error, not Auto.
DeviceSelection selectDevice (const char* text, const DeviceEnvironment& environment);

// The setting's stored value for kind: "optix", "cuda", "hip", "metal", "oneapi"
// or "cpu". selectDevice reads each back as kind; "auto" stands for Auto.
const char* deviceSettingValue (DeviceKind kind) noexcept;

// Which value decides the device: ARBIT_PATHTRACE_DEVICE when set, for tests and
// launch scripts, otherwise the machine setting Preferences passes to the helper.
const char* effectiveDeviceSetting (const char* environmentOverride, const char* preference) noexcept;

// Why Auto settled on the CPU, for the user: a clause that continues "Path Trace is
// running on CPU", with the first device it passed over and that device's problem
// in brackets at the end. Empty unless Auto chose the CPU.
std::string autoPassedOverReason (const DeviceSelection& selection);

// The same clause for a device that failed after Auto chose it. Plain words and what
// to do first, then the device and the driver's code in brackets so a user can report it.
std::string deviceFailureReason (const std::string& error, bool lost, DeviceKind kind = DeviceKind::OptiX);

// CUDA codes after which a context stays broken: a kernel fault, a lost device or
// a destroyed context. Anything else, out of memory say, leaves the process able
// to try again.
bool cudaErrorLosesDevice (int code) noexcept;
// A device reports the loss here, and from then on no pass in this process touches
// the driver. A new helper process gets a new context.
void noteDeviceLost() noexcept;
bool deviceLost() noexcept;

// file in the helper executable's directory, where the build and the installers
// put the oneAPI device library.
std::string besideHelperPath (const char* file);

// Tries to load library and find symbol. Empty on success. The library stays
// loaded, because unloading vendor driver libraries is not safe.
std::string libraryLoadProblem (const char* library, const char* symbol);

// This build on this machine.
DeviceEnvironment productDeviceEnvironment();

} // namespace videohelper::pathtrace::device
