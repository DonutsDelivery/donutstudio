// Tests for the Path Trace device list (Plan 3): the Auto order, what an
// unavailable device reports and the load probe. No OptiX SDK or driver needed.

#include "pathtrace/devices/device_registry.h"

#include <iostream>
#include <string>

namespace
{

namespace device = videohelper::pathtrace::device;
using device::DeviceKind;

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

bool contains (const std::string& text, const std::string& part)
{
    return text.find (part) != std::string::npos;
}

// A machine where every library loads.
device::DeviceEnvironment workingMachine()
{
    device::DeviceEnvironment environment;
    environment.optixSdkBuilt = true;
    environment.optixKernelBuilt = true;
    environment.loadProblem = [] (const char*, const char*) { return std::string(); };
    environment.optixInitProblem = [] { return std::string(); };
    return environment;
}

device::DeviceEnvironment missingLibrary (const std::string& part)
{
    auto environment = workingMachine();
    environment.loadProblem = [part] (const char* library, const char*)
    {
        return contains (library, part) ? std::string (library) + " did not load" : std::string();
    };
    return environment;
}

void orderTests()
{
    check (device::kAutoOrder.front() == DeviceKind::OptiX && device::kAutoOrder.back() == DeviceKind::Cpu,
           "the Auto order starts at OptiX and ends at CPU");
    const std::string names[] = { "OptiX", "CUDA", "HIP", "Metal", "oneAPI", "CPU" };
    const auto statuses = device::probeDevices ({});
    check (statuses.size() == 6, "every device has a status");
    for (std::size_t i = 0; i < statuses.size() && i < 6; ++i)
        check (device::deviceName (statuses[i].kind) == names[i], "status order: " + names[i]);
}

void failureReasonTests()
{
    const auto link = device::deviceFailureReason ("Path Trace device failed: OptiX device: linking the pipeline failed: OPTIX_ERROR_PIPELINE_LINK_ERROR a long compile log", false);
    check (link == "because the GPU renderer could not start. Switch Render Engine away from Path Trace and back to try the GPU again. (OptiX: OPTIX_ERROR_PIPELINE_LINK_ERROR)",
           "a failed setup reads plain first, with the code last and no compile log");
    const auto lost = device::deviceFailureReason ("Path Trace device failed: OptiX device: launching failed with CUDA error 999", true);
    check (lost == "because the GPU stopped responding. Restart DonutStudio to try the GPU again. (OptiX: CUDA error 999)",
           "a lost device says to restart, and keeps the CUDA code");
    check (contains (device::deviceFailureReason ("OptiX device: the GPU context was lost earlier in this session", true), "(OptiX: the GPU context was lost earlier in this session)"),
           "an error with no code keeps its words");
}

void autoTests()
{
    auto selection = device::selectDevice (nullptr, workingMachine());
    check (selection.ok && selection.device == DeviceKind::OptiX, "Auto takes OptiX when it loads");
    check (std::string (selection.requested) == "Auto", "the selection says Auto was requested");

    selection = device::selectDevice ("AUTO", {});
    check (selection.ok && selection.device == DeviceKind::Cpu, "Auto without an SDK build falls to CPU");
    check (contains (selection.devices[0].problem, "no OptiX device"), "the skipped OptiX names what the build lacks");
    check (! contains (selection.devices[0].problem, "OptiX_INSTALL_DIR"), "the reason a user reads names no build setting");

    selection = device::selectDevice ("", missingLibrary ("optix"));
    check (selection.ok && selection.device == DeviceKind::Cpu, "Auto without the driver library falls to CPU");
    check (contains (selection.devices[0].problem, "optix"), "the skipped OptiX names the missing library");
    check (contains (device::autoPassedOverReason (selection), "OptiX: ")
               && contains (device::autoPassedOverReason (selection), "optix"),
           "the reason for the user names the device Auto passed over and its problem");
    check (device::autoPassedOverReason (device::selectDevice (nullptr, workingMachine())).empty(),
           "Auto on OptiX has no reason to give");
    check (device::autoPassedOverReason (device::selectDevice ("none", workingMachine())).empty(),
           "a CPU asked for by name is not a fallback");
    check (device::cudaErrorLosesDevice (719) && device::cudaErrorLosesDevice (999), "a kernel fault loses the device");
    check (! device::cudaErrorLosesDevice (2) && ! device::cudaErrorLosesDevice (0), "out of memory does not");
    check (! device::deviceLost(), "no device is lost at the start");
    device::noteDeviceLost();
    check (device::deviceLost(), "a reported loss holds for the process");

    auto noKernel = workingMachine();
    noKernel.optixKernelBuilt = false;
    selection = device::selectDevice ("auto", noKernel);
    check (selection.ok && selection.device == DeviceKind::Cpu, "Auto never names a device that cannot trace");
    check (contains (selection.devices[0].problem, "kernel"), "the skipped OptiX names the missing kernel");
}

void explicitTests()
{
    auto selection = device::selectDevice ("optix", workingMachine());
    check (selection.ok && selection.device == DeviceKind::OptiX, "named OptiX selects it");
    check (std::string (selection.requested) == "OptiX", "the selection names the request");

    selection = device::selectDevice ("OptiX", {});
    check (! selection.ok, "named OptiX without an SDK build does not switch to CPU");
    check (contains (selection.error, "OptiX is unavailable") && contains (selection.error, "no OptiX device"),
           "the error names what the build lacks: " + selection.error);

    selection = device::selectDevice ("OptiX", missingLibrary ("optix"));
    check (! selection.ok && contains (selection.error, "optix"), "the error names the missing driver library: " + selection.error);

    selection = device::selectDevice ("optix", missingLibrary ("cuda"));
    check (! selection.ok && contains (selection.error, "cuda"), "the error names the missing CUDA library: " + selection.error);

    auto oldDriver = workingMachine();
    oldDriver.optixInitProblem = [] { return std::string ("the NVIDIA driver is older"); };
    selection = device::selectDevice ("optix", oldDriver);
    check (! selection.ok && contains (selection.error, "driver is older"), "the error carries the driver's refusal");

    for (const char* name : { "cuda", "hip" })
    {
        selection = device::selectDevice (name, workingMachine());
        check (! selection.ok && contains (selection.error, "this build of DonutStudio has no") && contains (selection.error, "kernel"),
               std::string (name) + " without its kernel reports that this build has none: " + selection.error);
    }
    for (const char* name : { "metal", "oneapi" })
    {
        selection = device::selectDevice (name, workingMachine());
        check (! selection.ok && contains (selection.error, "is unavailable"),
               std::string (name) + " reports why it is unavailable: " + selection.error);
    }

    for (const char* name : { "none", "None", "cpu", "CPU" })
    {
        selection = device::selectDevice (name, {});
        check (selection.ok && selection.device == DeviceKind::Cpu && std::string (selection.requested) == "CPU",
               std::string (name) + " selects the CPU device");
    }

    selection = device::selectDevice ("vulkan", workingMachine());
    check (! selection.ok && contains (selection.error, "\"vulkan\" is unknown"), "an unknown name is an error, not Auto");
}

void productTests()
{
    // Whatever this machine holds, the build's own environment always renders on Auto.
    const auto selection = device::selectDevice (nullptr, device::productDeviceEnvironment());
    check (selection.ok, "Auto selects a device in the real environment");

    const auto missing = device::libraryLoadProblem ("libarbit-no-such-library.so.1", "anything");
    check (contains (missing, "libarbit-no-such-library.so.1"), "a failed load names the library");
#if defined(__linux__)
    check (device::libraryLoadProblem ("libm.so.6", "cos").empty(), "a real library and symbol load");
    check (contains (device::libraryLoadProblem ("libm.so.6", "arbit_no_such_symbol"), "arbit_no_such_symbol"),
           "a missing symbol is named");
#endif
}

} // namespace

int main()
{
    orderTests();
    failureReasonTests();
    autoTests();
    explicitTests();
    productTests();
    std::cout << (checks - failures) << "/" << checks << " pathtrace device registry checks passed\n";
    return failures == 0 ? 0 : 1;
}
