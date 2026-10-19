// arbit-pathtrace-timing: wall-clock cost of the Path Trace engine on the
// Smooth Dispersive Glass starter scene (tests/support/pathtrace_glass_scene.h).
// Not a test and not run by ctest. Run it by hand on the machine you want
// numbers for.
//
// Usage: arbit-pathtrace-timing [samples] [width] [height] [threads]
// Defaults: 64 samples, 1920 x 1080, every hardware thread.
//
// It prints:
// - the RenderScene and BVH build time,
// - the time to a usable viewport frame, which this tool defines as 16
//   samples per pixel accumulated one sample per frame, as the viewport does,
// - the time to the requested samples per pixel as one fixed export.

#include "pathtrace/engine.h"
#include "support/pathtrace_glass_scene.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

namespace
{
namespace pt = videohelper::pathtrace;
namespace glass = pathtraceglassscene;

constexpr std::uint32_t kUsableSamples = 16;

double secondsSince (std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double> (std::chrono::steady_clock::now() - start).count();
}

std::uint32_t argument (int argc, char** argv, int index, std::uint32_t fallback)
{
    if (argc <= index) return fallback;
    const long value = std::strtol (argv[index], nullptr, 10);
    return value > 0 ? static_cast<std::uint32_t> (value) : fallback;
}
} // namespace

int main (int argc, char** argv)
{
    const std::uint32_t samples = argument (argc, argv, 1, 64);
    const std::uint32_t width = argument (argc, argv, 2, 1920);
    const std::uint32_t height = argument (argc, argv, 3, 1080);
    const std::uint32_t threads = argument (argc, argv, 4, 0);

    const auto scene = glass::makeScene();
    const auto setup = glass::makeSetup (scene);
    const auto view = glass::makeView (*scene, width, height);
    pt::SceneEngine engine (threads);
    engine.samplesPerPass = 1;
    std::printf ("Smooth Dispersive Glass starter scene, %u x %u, %u threads\n", width, height,
                 threads != 0 ? threads : std::thread::hardware_concurrency());
    std::printf ("Usable frame: %u samples per pixel (this tool's definition)\n", kUsableSamples);

    std::string error;
    pt::EngineImage image;
    const auto start = std::chrono::steady_clock::now();
    double firstPass = 0.0;
    for (std::uint32_t pass = 0; pass < kUsableSamples; ++pass)
    {
        if (! engine.renderPreview (setup, view, pt::EngineColor::DisplayRgba8, image, error))
        {
            std::fprintf (stderr, "preview failed: %s\n", error.c_str());
            return EXIT_FAILURE;
        }
        if (pass == 0) firstPass = secondsSince (start);
    }
    const double usable = secondsSince (start);
    std::printf ("First viewport frame (scene build + 1 spp): %.3f s\n", firstPass);
    std::printf ("Time to usable frame (%u spp, one pass per frame): %.3f s, %.3f s per frame\n",
                 kUsableSamples, usable, (usable - firstPass) / static_cast<double> (kUsableSamples - 1));

    const auto exportStart = std::chrono::steady_clock::now();
    if (! engine.renderExport (setup, view, pt::EngineColor::LinearRgba16f, samples, image, error))
    {
        std::fprintf (stderr, "export failed: %s\n", error.c_str());
        return EXIT_FAILURE;
    }
    const double exported = secondsSince (exportStart);
    std::printf ("Time to %u spp (export): %.3f s, %.1f Mrays/s\n", samples, exported,
                 static_cast<double> (image.stats.closestRays + image.stats.shadowRays) / exported * 1.0e-6);
    std::printf ("Non-finite samples dropped: %llu\n",
                 static_cast<unsigned long long> (image.stats.nonFiniteSamples));
    return EXIT_SUCCESS;
}
