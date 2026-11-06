// The CUDA and HIP devices (Plan 3, Render engines and devices) and the device
// setting Preferences passes to the helper: the Auto order across every device,
// a named device that cannot load, the ARBIT_PATHTRACE_DEVICE override, the
// stored values, and the failure reason naming the device that failed.
//
// Then the equality proof: the glass scene at 4 samples per pixel on the CPU and
// on each compute device this machine can run. A machine without the GPU, the
// driver or the kernel prints why it skipped and passes.

#include "pathtrace/devices/device_compute_host.h"
#include "pathtrace/devices/device_registry.h"
#include "pathtrace/engine.h"
#include "support/pathtrace_glass_scene.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace
{

namespace pt = videohelper::pathtrace;
namespace device = pt::device;
namespace glass = pathtraceglassscene;
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

// Every library loads and every kernel is built, unless a test takes one away.
device::DeviceEnvironment fullMachine()
{
    device::DeviceEnvironment environment;
    environment.optixSdkBuilt = true;
    environment.optixKernelBuilt = true;
    environment.cudaKernelBuilt = true;
    environment.hipKernelBuilt = true;
    environment.applePlatform = true;
    environment.loadProblem = [] (const char*, const char*) { return std::string(); };
    environment.optixInitProblem = [] { return std::string(); };
    return environment;
}

device::DeviceEnvironment withoutLibrary (device::DeviceEnvironment environment, const std::string& part)
{
    environment.loadProblem = [part] (const char* library, const char*)
    {
        return contains (library, part) ? std::string (library) + " did not load" : std::string();
    };
    return environment;
}

void autoOrderTests()
{
    auto machine = fullMachine();
    check (device::selectDevice ("auto", machine).device == DeviceKind::OptiX, "Auto takes OptiX first");

    machine.optixKernelBuilt = false;
    auto selection = device::selectDevice ("auto", machine);
    check (selection.ok && selection.device == DeviceKind::Cuda, "Auto takes CUDA when OptiX cannot load");

    machine = withoutLibrary (fullMachine(), "cuda");  // OptiX needs libcuda too
    selection = device::selectDevice ("auto", machine);
    check (selection.ok && selection.device == DeviceKind::Hip, "Auto takes HIP when the CUDA driver is missing");

    machine.hipKernelBuilt = false;
    selection = device::selectDevice ("auto", machine);
    check (selection.ok && selection.device == DeviceKind::Cpu,
           "Auto passes Metal and oneAPI, which this machine's build lacks, and takes the CPU");
    check (contains (device::autoPassedOverReason (selection), "OptiX: "), "the reason names the first device passed over");

    const auto statuses = device::probeDevices (fullMachine());
    check (statuses.size() == device::kAutoOrder.size(), "every device has a status");
    for (std::size_t i = 0; i < statuses.size(); ++i)
        check (statuses[i].kind == device::kAutoOrder[i], std::string ("status in Auto order: ") + device::deviceName (statuses[i].kind));
    check (statuses[1].loadable && statuses[2].loadable, "CUDA and HIP load where their kernels are built and their drivers load");
    check (! statuses[3].loadable && ! statuses[4].loadable, "Metal and oneAPI do not load where they are not built");
}

void namedDeviceTests()
{
    auto selection = device::selectDevice ("CUDA", fullMachine());
    check (selection.ok && selection.device == DeviceKind::Cuda && std::string (selection.requested) == "CUDA", "named CUDA selects it");
    selection = device::selectDevice ("hip", fullMachine());
    check (selection.ok && selection.device == DeviceKind::Hip, "named HIP selects it");

    auto noKernels = fullMachine();
    noKernels.cudaKernelBuilt = false;
    noKernels.hipKernelBuilt = false;
    selection = device::selectDevice ("cuda", noKernels);
    check (! selection.ok && contains (selection.error, "CUDA is unavailable") && contains (selection.error, "nvcc"),
           "named CUDA without its kernel stops by name: " + selection.error);
    selection = device::selectDevice ("hip", noKernels);
    check (! selection.ok && contains (selection.error, "HIP is unavailable") && contains (selection.error, "hipcc"),
           "named HIP without its kernel stops by name: " + selection.error);

    selection = device::selectDevice ("cuda", withoutLibrary (fullMachine(), "cuda"));
    check (! selection.ok && (contains (selection.error, "libcuda") || contains (selection.error, "nvcuda")),
           "named CUDA names the missing driver: " + selection.error);
    selection = device::selectDevice ("hip", withoutLibrary (fullMachine(), "amdhip"));
    check (! selection.ok && contains (selection.error, "amdhip64"), "named HIP names the missing runtime: " + selection.error);

    selection = device::selectDevice ("metal", fullMachine());
    check (! selection.ok && contains (selection.error, "no Metal device") && contains (selection.error, "Metal toolchain"),
           "named Metal on a Mac says the build has none: " + selection.error);
    auto elsewhere = fullMachine();
    elsewhere.applePlatform = false;
    selection = device::selectDevice ("metal", elsewhere);
    check (! selection.ok && contains (selection.error, "macOS only"), "named Metal elsewhere says where it runs: " + selection.error);
    selection = device::selectDevice ("oneapi", fullMachine());
    check (! selection.ok && contains (selection.error, "no oneAPI device") && contains (selection.error, "DPC++"),
           "named oneAPI says the build has none: " + selection.error);
    auto oneApiBuilt = fullMachine();
    oneApiBuilt.oneApiDeviceBuilt = true;
    selection = device::selectDevice ("oneapi", withoutLibrary (oneApiBuilt, "arbit-pathtrace-oneapi"));
    check (! selection.ok && contains (selection.error, "arbit-pathtrace-oneapi"),
           "named oneAPI names the missing device library: " + selection.error);
}

void settingTests()
{
    for (const auto kind : device::kAutoOrder)
    {
        const auto selection = device::selectDevice (device::deviceSettingValue (kind), fullMachine());
        check (std::string (selection.requested) == device::deviceName (kind),
               std::string ("the stored value reads back as ") + device::deviceName (kind));
    }
    check (std::string (device::effectiveDeviceSetting (nullptr, "cuda")) == "cuda", "the preference decides without an override");
    check (std::string (device::effectiveDeviceSetting ("", "hip")) == "hip", "an empty override is none");
    check (std::string (device::effectiveDeviceSetting ("cpu", "cuda")) == "cpu", "ARBIT_PATHTRACE_DEVICE overrides the preference");

    const auto cuda = device::deviceFailureReason ("Path Trace device failed: CUDA device: launching the kernel failed with CUDA error 700", true, DeviceKind::Cuda);
    check (contains (cuda, "(CUDA: CUDA error 700)") && contains (cuda, "Restart DonutStudio"), "a lost CUDA device names CUDA: " + cuda);
    const auto hip = device::deviceFailureReason ("HIP device: loading the kernel failed with HIP error 218", false, DeviceKind::Hip);
    check (contains (hip, "(HIP: HIP error 218)"), "a failed HIP device names HIP: " + hip);
    check (contains (device::deviceFailureReason ("OptiX device: x", false), "(OptiX: x)"), "OptiX stays the default name");
}

// ---------------------------------------------------------------- equality

float halfToFloat (std::uint16_t half)
{
    const int exponent = (half >> 10) & 0x1f;
    const float magnitude = exponent == 0 ? static_cast<float> (half & 0x3ff) * std::ldexp (1.0f, -24)
                                          : static_cast<float> ((half & 0x3ff) | 0x400) * std::ldexp (1.0f, exponent - 25);
    return (half & 0x8000) != 0 ? -magnitude : magnitude;
}

std::vector<float> linearPixels (const pt::EngineImage& image)
{
    std::vector<float> values (image.colorBytes.size() / sizeof (std::uint16_t));
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        std::uint16_t half = 0;
        std::memcpy (&half, image.colorBytes.data() + i * sizeof (half), sizeof (half));
        values[i] = halfToFloat (half);
    }
    return values;
}

// The heat rule: CPU renders for equality proofs stay at 16 samples or fewer.
constexpr std::uint32_t kEqualitySamples = 4;
constexpr std::uint32_t kEqualitySide = 16;

void equalityOn (DeviceKind kind)
{
    const char* name = device::deviceName (kind);
    const auto selection = device::selectDevice (device::deviceSettingValue (kind), device::productDeviceEnvironment());
    if (! selection.ok)
    {
        std::cout << "SKIP " << name << " equals CPU: " << selection.error << '\n';
        return;
    }
    const auto scene = glass::makeScene();
    const auto setup = glass::makeSetup (scene);
    const auto view = glass::makeView (*scene, kEqualitySide, kEqualitySide);
    std::string error;

    pt::EngineImage cpu;
    {
        pt::SceneEngine engine (1);
        check (engine.renderExport (setup, view, pt::EngineColor::LinearRgba16f, kEqualitySamples, cpu, error),
               std::string ("the CPU export renders: ") + error);
    }
    pt::EngineImage gpu;
    {
        pt::SceneEngine engine (1);
        auto* source = &engine;
        engine.setDevicePass (kind == DeviceKind::Cuda ? device::makeCudaDevicePass ([source] { return source->sceneBuilds(); })
                                                       : device::makeHipDevicePass ([source] { return source->sceneBuilds(); }));
        const bool rendered = engine.renderExport (setup, view, pt::EngineColor::LinearRgba16f, kEqualitySamples, gpu, error);
        if (! rendered && gpu.stats.deviceFailed)
        {
            std::cout << "SKIP " << name << " equals CPU: the device did not start: " << error << '\n';
            return;
        }
        check (rendered, std::string (name) + " export renders: " + error);
    }
    check (gpu.samplesPerPixel == cpu.samplesPerPixel, std::string (name) + " adds the same samples per pixel");
    const auto a = linearPixels (cpu);
    const auto b = linearPixels (gpu);
    check (a.size() == b.size() && ! a.empty(), std::string (name) + " image has the CPU's size");
    if (a.size() != b.size() || a.empty())
        return;
    // Same sample streams and integrator source. GPU float contraction and the
    // film's float atomics move values a little; a path that branches differently
    // moves one pixel more. The mean over the image stays close.
    double sumA = 0.0, sumDifference = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        sumA += std::abs (a[i]);
        sumDifference += std::abs (a[i] - b[i]);
    }
    const double relative = sumA > 0.0 ? sumDifference / sumA : sumDifference;
    std::cout << name << " vs CPU: mean relative difference " << relative << " over " << a.size() << " channels\n";
    check (sumA > 0.0, "the CPU image is not black");
    check (relative < 0.02, std::string (name) + " matches the CPU within 2% of the image's mean");
}

} // namespace

int main()
{
    autoOrderTests();
    namedDeviceTests();
    settingTests();
    equalityOn (DeviceKind::Cuda);
    equalityOn (DeviceKind::Hip);
    std::cout << (checks - failures) << "/" << checks << " pathtrace compute device checks passed\n";
    return failures == 0 ? 0 : 1;
}
