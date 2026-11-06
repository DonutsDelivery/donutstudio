// The Metal Path Trace device (Plan 3, Render engines and devices): where Auto
// puts it, what a user sees without a Metal build, without macOS or without Metal
// ray tracing, and what a failed pass says.
//
// The kernel source the Mac compiles is read on every platform: one text with no
// quoted include left, the kernel and its intersection function, and the lookup
// tables written in as constant arrays that read back as the host's floats.
//
// On a Mac with Metal ray tracing the device then renders the glass scene against
// the CPU at 4 samples per pixel. Elsewhere that part prints why it skipped.

#include "pathtrace/bsdf.h"
#include "pathtrace/devices/device_metal_host.h"
#include "pathtrace/devices/device_metal_params.h"
#include "pathtrace/devices/device_registry.h"
#include "pathtrace/devices/device_tables.h"
#include "pathtrace/engine.h"
#include "pathtrace/spectrum.h"
#include "support/pathtrace_glass_scene.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace
{

namespace pt = videohelper::pathtrace;
namespace device = pt::device;
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

// An Apple silicon Mac: Metal with ray tracing and nothing from NVIDIA or AMD.
device::DeviceEnvironment appleMachine()
{
    device::DeviceEnvironment environment;
    environment.applePlatform = true;
    environment.metalDeviceBuilt = true;
    environment.loadProblem = [] (const char*, const char*) { return std::string(); };
    environment.metalDeviceProblem = [] { return std::string(); };
    return environment;
}

void selectionTests()
{
    auto selection = device::selectDevice ("auto", appleMachine());
    check (selection.ok && selection.device == DeviceKind::Metal, "Auto takes Metal on a Mac");
    selection = device::selectDevice ("Metal", appleMachine());
    check (selection.ok && selection.device == DeviceKind::Metal && std::string (selection.requested) == "Metal",
           "named Metal selects it");

    auto noRayTracing = appleMachine();
    noRayTracing.metalDeviceProblem = [] { return std::string ("AMD Radeon Pro 560 has no Metal ray tracing"); };
    selection = device::selectDevice ("metal", noRayTracing);
    check (! selection.ok && contains (selection.error, "Metal is unavailable") && contains (selection.error, "no Metal ray tracing"),
           "named Metal without ray tracing stops and says why: " + selection.error);
    selection = device::selectDevice ("auto", noRayTracing);
    check (selection.ok && selection.device == DeviceKind::Cpu, "Auto passes Metal without ray tracing");

    auto unbuilt = appleMachine();
    unbuilt.metalDeviceBuilt = false;
    selection = device::selectDevice ("metal", unbuilt);
    check (! selection.ok && contains (selection.error, "no Metal device"), "a build without Metal says so: " + selection.error);

    auto pc = appleMachine();
    pc.applePlatform = false;
    selection = device::selectDevice ("metal", pc);
    check (! selection.ok && contains (selection.error, "macOS only"), "Metal elsewhere says where it runs: " + selection.error);

    auto noFramework = appleMachine();
    noFramework.loadProblem = [] (const char* library, const char*)
    {
        return contains (library, "Metal.framework") ? std::string (library) + " did not load" : std::string();
    };
    selection = device::selectDevice ("metal", noFramework);
    check (! selection.ok && contains (selection.error, "Metal.framework"), "a missing framework is named: " + selection.error);

    const auto reason = device::deviceFailureReason (
        "Path Trace device failed: Metal device: the kernel did not compile: program_source:12: error", false, DeviceKind::Metal);
    check (contains (reason, "(Metal: the kernel did not compile: program_source:12: error)"),
           "a failed Metal pass names Metal and the compiler's words: " + reason);
}

// Every quoted include is inlined, and only Metal's own headers are left.
void sourceTests()
{
    const std::string source = device::metalKernelSource();
    check (! contains (source, device::metal::kTablesMarker), "the tables replace their marker");
    check (contains (source, std::string ("kernel void ") + device::metal::kKernelName), "the source holds the kernel");
    check (contains (source, std::string ("bool ") + device::metal::kSkipTriangleName)
               && contains (source, "[[intersection (triangle, triangle_data)]]"),
           "the source holds the triangle intersection function");
    check (contains (source, "primitive_acceleration_structure") && contains (source, "intersector<"),
           "the kernel traces with Metal ray tracing");
    check (contains (source, "PT_DEVICE_ENTRY SampleOutcome renderSample"), "the integrator body is in the source");
    std::istringstream lines (source);
    std::string line;
    bool quoted = false, foreign = false, once = false;
    while (std::getline (lines, line))
    {
        const auto start = line.find_first_not_of (" \t");
        if (start == std::string::npos || line[start] != '#')
            continue;
        const auto directive = line.substr (start);
        quoted = quoted || directive.rfind ("#include \"", 0) == 0;
        foreign = foreign || (directive.rfind ("#include <", 0) == 0 && directive.rfind ("#include <metal_", 0) != 0);
        once = once || directive.rfind ("#pragma once", 0) == 0;
    }
    check (! quoted, "no quoted include is left");
    check (! foreign, "no C++ standard header is left");
    check (! once, "no #pragma once is left");

    device::DeviceLookupTables tables;
    pt::exportSpectrumTables (tables);
    pt::exportGgxTables (tables);
    const auto declared = [&source] (const std::string& declaration) { return contains (source, declaration); };
    check (declared ("constant float arbitSpectrumWeight[" + std::to_string (tables.spectrumRows) + "][3]"),
           "the spectrum weights have the host's rows");
    check (declared ("constant float arbitGgxScale[" + std::to_string (tables.ggxEntries) + "]"),
           "the GGX scale has the host's entries");
    check (declared ("constant ulong kArbitSpectrumRows = " + std::to_string (tables.spectrumRows) + ";"),
           "the row count is written");

    // Each literal reads back as the host's float.
    const auto table = device::metalLookupTableSource();
    const auto readList = [&table] (const std::string& name) {
        std::vector<float> values;
        auto at = table.find ("constant float " + name + "[");
        if (at == std::string::npos)
            return values;
        at = table.find ('{', at);
        const auto end = table.find ("};", at);
        for (const char* cursor = table.c_str() + at; cursor < table.c_str() + end; ++cursor)
            if (*cursor == '-' || (*cursor >= '0' && *cursor <= '9'))
            {
                char* next = nullptr;
                values.push_back (std::strtof (cursor, &next));
                cursor = next;
            }
        return values;
    };
    const auto scale = readList ("arbitGgxScale");
    bool scaleExact = scale.size() == tables.ggxEntries;
    for (std::size_t i = 0; scaleExact && i < scale.size(); ++i)
        scaleExact = scale[i] == tables.ggxScale[i];
    check (scaleExact, "the GGX scale reads back bit for bit");
    const auto weight = readList ("arbitSpectrumWeight");
    bool weightExact = weight.size() == tables.spectrumRows * 3;
    for (std::size_t i = 0; weightExact && i < weight.size(); ++i)
        weightExact = weight[i] == tables.spectrumWeight[i / 3][i % 3];
    check (weightExact, "the spectrum weights read back bit for bit");
}

#if defined(ARBIT_HAVE_METAL_PATHTRACE)
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
#endif

// The heat rule: CPU renders for equality proofs stay at 16 samples or fewer.
void deviceTests()
{
#if defined(ARBIT_HAVE_METAL_PATHTRACE)
    namespace glass = pathtraceglassscene;
    const auto selection = device::selectDevice ("metal", device::productDeviceEnvironment());
    if (! selection.ok)
    {
        std::cout << "SKIP Metal equals CPU: " << selection.error << '\n';
        return;
    }
    constexpr std::uint32_t samples = 4, side = 16;
    const auto scene = glass::makeScene();
    const auto setup = glass::makeSetup (scene);
    const auto view = glass::makeView (*scene, side, side);
    std::string error;
    pt::EngineImage gpu;
    {
        pt::SceneEngine engine (1);
        auto* source = &engine;
        engine.setDevicePass (device::makeMetalDevicePass ([source] { return source->sceneBuilds(); }));
        const bool rendered = engine.renderExport (setup, view, pt::EngineColor::LinearRgba16f, samples, gpu, error);
        check (rendered, "Metal export renders: " + error);
        if (! rendered)
            return;
    }
    pt::EngineImage cpu;
    {
        pt::SceneEngine engine (1);
        check (engine.renderExport (setup, view, pt::EngineColor::LinearRgba16f, samples, cpu, error), "the CPU export renders: " + error);
    }
    const auto a = linearPixels (cpu);
    const auto b = linearPixels (gpu);
    check (a.size() == b.size() && ! a.empty(), "Metal image has the CPU's size");
    if (a.size() != b.size() || a.empty())
        return;
    double sumA = 0.0, sumDifference = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        sumA += std::abs (a[i]);
        sumDifference += std::abs (a[i] - b[i]);
    }
    const double relative = sumA > 0.0 ? sumDifference / sumA : sumDifference;
    std::cout << "Metal vs CPU: mean relative difference " << relative << " over " << a.size() << " channels\n";
    check (sumA > 0.0, "the CPU image is not black");
    check (relative < 0.02, "Metal matches the CPU within 2% of the image's mean");
#else
    std::cout << "SKIP Metal equals CPU: this build has no Metal device (macOS only)\n";
#endif
}

} // namespace

int main()
{
    selectionTests();
    sourceTests();
    deviceTests();
    std::cout << (checks - failures) << "/" << checks << " pathtrace Metal device checks passed\n";
    return failures == 0 ? 0 : 1;
}
