// Tests for the Render node's Render Mode on previews and export: Fast lowers the
// Path Trace bounce limits and playback samples, exports at the preview sample
// count, denoises a converging preview only at power-of-two sample counts, and
// drops Realtime's reflection pass; Accurate keeps the node's full settings.

#include "render_mode_policy.h"

#include <iostream>
#include <string>

namespace
{

namespace fx = videorender::fixture3d;

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

void pathTraceCostTests()
{
    const auto accurate = fx::pathTracePreviewCost (false, 12, 6, 16);
    check (accurate.maxBounces == 12 && accurate.maxVolumeBounces == 6 && accurate.playbackSamples == 16,
           "Accurate keeps the node's bounces and samples");

    const auto fast = fx::pathTracePreviewCost (true, 12, 6, 16);
    check (fast.maxBounces == fx::kFastPreviewBounces, "Fast lowers the bounce limit");
    check (fast.maxVolumeBounces == fx::kFastPreviewVolumeBounces, "Fast lowers the volume bounce limit");
    check (fast.playbackSamples == 4, "Fast plays a quarter of the samples per frame");

    const auto low = fx::pathTracePreviewCost (true, 2, 0, 2);
    check (low.maxBounces == 2 && low.maxVolumeBounces == 0, "Fast never raises a lower limit");
    check (low.playbackSamples == 1, "Fast keeps at least one sample per frame");
    check (fx::pathTracePreviewCost (false, 2, 0, 0).playbackSamples == 1, "no draw traces zero samples");
}

void denoiseTests()
{
    for (std::uint32_t spp : { 1u, 3u, 5u, 6u, 7u })
        check (fx::denoiseThisImage (false, true, spp), "Accurate denoises every converging image");
    check (fx::denoiseThisImage (true, false, 7), "a settled Fast preview denoises");
    check (fx::denoiseThisImage (true, true, 1) && fx::denoiseThisImage (true, true, 8),
           "Fast denoises at power-of-two counts");
    check (! fx::denoiseThisImage (true, true, 6) && ! fx::denoiseThisImage (true, true, 0),
           "Fast skips other converging counts");
}

void exportTests()
{
    check (fx::exportSamples (false, 256, 16) == 256, "Accurate exports the node's export samples");
    check (fx::exportSamples (true, 256, 16) == 16, "Fast exports the preview samples");
    check (fx::exportSamples (true, 8, 64) == 8, "Fast never exports more than export asks for");
    check (fx::exportSamples (true, 0, 0) == 1 && fx::exportSamples (false, 0, 16) == 1, "no export traces zero samples");
    const auto fast = fx::pathTracePreviewCost (true, 12, 6, 0);
    check (fast.maxBounces == fx::kFastPreviewBounces && fast.maxVolumeBounces == fx::kFastPreviewVolumeBounces,
           "a Fast export takes the Fast bounce limits");
}

void realtimeTests()
{
    check (fx::realtimeReflections (false), "Accurate Realtime draws reflections");
    check (! fx::realtimeReflections (true), "Fast Realtime skips reflections");
}

} // namespace

int main()
{
    pathTraceCostTests();
    denoiseTests();
    exportTests();
    realtimeTests();
    std::cout << checks - failures << " of " << checks << " checks passed\n";
    return failures == 0 ? 0 : 1;
}
