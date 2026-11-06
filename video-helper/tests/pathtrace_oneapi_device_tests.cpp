// The oneAPI Path Trace device (Plan 3, Render engines and devices): where Auto
// puts it, what a user sees when the build, the library beside the helper, the
// SYCL runtime or an Intel GPU is missing, and what a failed pass says.
//
// Then the device itself: the glass scene at 4 samples per pixel on the CPU and
// through the oneAPI library. Without an Intel GPU the library may take Intel's
// OpenCL CPU device (ARBIT_ONEAPI_DEVICE_TYPE=cpu), which runs the same SYCL
// kernel. Without the library or any device the test prints why and checks the
// error a render gets instead.

#include "pathtrace/devices/device_compute_host.h"
#include "pathtrace/devices/device_oneapi_abi.h"
#include "pathtrace/devices/device_registry.h"
#include "pathtrace/engine.h"
#include "support/pathtrace_glass_scene.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
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

// A PC with an Intel GPU and nothing else: no NVIDIA or AMD device, not a Mac.
device::DeviceEnvironment intelMachine()
{
    device::DeviceEnvironment environment;
    environment.oneApiDeviceBuilt = true;
    environment.loadProblem = [] (const char*, const char*) { return std::string(); };
    environment.oneApiDeviceProblem = [] { return std::string(); };
    return environment;
}

void selectionTests()
{
    auto selection = device::selectDevice ("auto", intelMachine());
    check (selection.ok && selection.device == DeviceKind::OneApi, "Auto takes oneAPI when no device before it loads");
    selection = device::selectDevice ("oneapi", intelMachine());
    check (selection.ok && selection.device == DeviceKind::OneApi && std::string (selection.requested) == "oneAPI",
           "named oneAPI selects it");

    auto withCuda = intelMachine();
    withCuda.cudaKernelBuilt = true;
    check (device::selectDevice ("auto", withCuda).device == DeviceKind::Cuda, "Auto keeps CUDA ahead of oneAPI");

    auto noGpu = intelMachine();
    noGpu.oneApiDeviceProblem = [] { return std::string ("the oneAPI runtime found no Intel GPU; install Intel's GPU driver with Level Zero"); };
    selection = device::selectDevice ("oneapi", noGpu);
    check (! selection.ok && contains (selection.error, "oneAPI is unavailable") && contains (selection.error, "no Intel GPU"),
           "named oneAPI without an Intel GPU stops and says so: " + selection.error);
    selection = device::selectDevice ("auto", noGpu);
    check (selection.ok && selection.device == DeviceKind::Cpu, "Auto passes oneAPI without an Intel GPU");
    check (contains (device::autoPassedOverReason (selection), "OptiX: "), "the CPU reason names the first device passed over");

    auto unbuilt = intelMachine();
    unbuilt.oneApiDeviceBuilt = false;
    selection = device::selectDevice ("oneapi", unbuilt);
    check (! selection.ok && contains (selection.error, "no oneAPI device") && contains (selection.error, "DPC++"),
           "named oneAPI in a build without it says the build lacks it: " + selection.error);

    auto noLibrary = intelMachine();
    noLibrary.loadProblem = [] (const char* library, const char*)
    {
        return contains (library, device::oneapi::kLibraryFile)
                 ? std::string (library) + " did not load: libsycl.so.9: cannot open shared object file"
                 : std::string();
    };
    selection = device::selectDevice ("oneapi", noLibrary);
    check (! selection.ok && contains (selection.error, device::oneapi::kLibraryFile) && contains (selection.error, "libsycl"),
           "named oneAPI names the library and the runtime it misses: " + selection.error);
    check (contains (selection.error, "intel-oneapi-runtime-dpcpp-cpp") && contains (selection.error, "Level Zero"),
           "a missing Intel runtime says what to install: " + selection.error);
    const auto ownLibraryMissing = device::withOneApiRuntimeHint ("libarbit-pathtrace-oneapi.so did not load: no such file");
    check (! contains (ownLibraryMissing, "intel-oneapi-runtime"), "a missing device library alone gets no runtime hint: " + ownLibraryMissing);

    const auto path = device::besideHelperPath (device::oneapi::kLibraryFile);
    check (contains (path, device::oneapi::kLibraryFile), "the library path ends in the library's file: " + path);
#if defined(__linux__)
    check (path.front() == '/' && contains (path, "/") && path != device::oneapi::kLibraryFile,
           "the library sits in the helper's directory: " + path);
#endif
}

void failureReasonTests()
{
    const auto reason = device::deviceFailureReason (
        "Path Trace device failed: oneAPI device: running the kernel failed with oneAPI error 1: UR_RESULT_ERROR_DEVICE_LOST", false,
        DeviceKind::OneApi);
    check (contains (reason, "(oneAPI: oneAPI error 1: UR_RESULT_ERROR_DEVICE_LOST)"), "a failed oneAPI pass names oneAPI and its error: " + reason);
    check (device::isComputeDevice (DeviceKind::OneApi) && ! device::isComputeDevice (DeviceKind::Metal),
           "oneAPI runs through the compute device host");
}

// ---------------------------------------------------------------- the device

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

void deviceTests()
{
    const auto scene = glass::makeScene();
    const auto setup = glass::makeSetup (scene);
    const auto view = glass::makeView (*scene, kEqualitySide, kEqualitySide);
    std::string error;

    pt::EngineImage gpu;
    bool rendered = false;
    {
        pt::SceneEngine engine (1);
        auto* source = &engine;
        engine.setDevicePass (device::makeOneApiDevicePass ([source] { return source->sceneBuilds(); }));
        rendered = engine.renderExport (setup, view, pt::EngineColor::LinearRgba16f, kEqualitySamples, gpu, error);
    }
    const auto selection = device::selectDevice ("oneapi", device::productDeviceEnvironment());
    if (! selection.ok)
    {
        // What a user who names oneAPI sees, and what a pass forced onto it says.
        std::cout << "SKIP oneAPI equals CPU: " << selection.error << '\n';
        check (contains (selection.error, "oneAPI is unavailable"), "the unavailable device says which: " + selection.error);
        check (! rendered && gpu.stats.deviceFailed && contains (error, "oneAPI device: "),
               "a pass without the device fails by name: " + error);
        return;
    }
    if (! rendered && gpu.stats.deviceFailed)
    {
        std::cout << "SKIP oneAPI equals CPU: the device did not start: " << error << '\n';
        check (contains (error, "oneAPI device: "), "a failed start names the oneAPI device: " + error);
        return;
    }
    check (rendered, "oneAPI export renders: " + error);

    pt::EngineImage cpu;
    {
        pt::SceneEngine engine (1);
        check (engine.renderExport (setup, view, pt::EngineColor::LinearRgba16f, kEqualitySamples, cpu, error),
               "the CPU export renders: " + error);
    }
    check (gpu.samplesPerPixel == cpu.samplesPerPixel, "oneAPI adds the same samples per pixel");
    const auto a = linearPixels (cpu);
    const auto b = linearPixels (gpu);
    check (a.size() == b.size() && ! a.empty(), "oneAPI image has the CPU's size");
    if (a.size() != b.size() || a.empty())
        return;
    double sumA = 0.0, sumDifference = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        sumA += std::abs (a[i]);
        sumDifference += std::abs (a[i] - b[i]);
    }
    const double relative = sumA > 0.0 ? sumDifference / sumA : sumDifference;
    std::cout << "oneAPI vs CPU: mean relative difference " << relative << " over " << a.size() << " channels\n";
    check (sumA > 0.0, "the CPU image is not black");
    check (relative < 0.02, "oneAPI matches the CPU within 2% of the image's mean");
}

} // namespace

int main()
{
#if ! defined(_WIN32)
    // An Intel GPU when there is one, otherwise Intel's OpenCL CPU device.
    setenv ("ARBIT_ONEAPI_DEVICE_TYPE", "cpu", 0);
#endif
    selectionTests();
    failureReasonTests();
    deviceTests();
    std::cout << (checks - failures) << "/" << checks << " pathtrace oneAPI device checks passed\n";
    return failures == 0 ? 0 : 1;
}
