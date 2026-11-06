// The speed fields of pathTraceStats and realtimeStats, the median/p95 helper and
// the measureFrames summary. Pure host code: no GPU, no helper link.

#include "../src/render_speed_stats.h"

#include <cmath>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

namespace
{
namespace pt = videohelper::pathtrace;
namespace ss = videorender::speedstats;

int checks = 0;
int failures = 0;
void check (bool condition, const char* message)
{
    ++checks;
    if (! condition) { ++failures; std::fprintf (stderr, "FAIL: %s\n", message); }
}

bool closeTo (double a, double b) { return std::fabs (a - b) < 1.0e-9; }

void summaryIsMedianAndNearestRankP95()
{
    check (ss::summarizeFrameTimes ({}).count == 0, "an empty set has no frames");
    const auto odd = ss::summarizeFrameTimes ({ 5.0, 1.0, 3.0 });
    check (odd.count == 3 && closeTo (odd.median, 3.0), "odd median is the middle time");
    const auto even = ss::summarizeFrameTimes ({ 4.0, 1.0, 3.0, 2.0 });
    check (closeTo (even.median, 2.5), "even median averages the two middle times");
    std::vector<double> hundred;
    for (int i = 1; i <= 100; ++i) hundred.push_back (static_cast<double> (i));
    const auto ranked = ss::summarizeFrameTimes (hundred);
    check (closeTo (ranked.p95, 95.0), "p95 of 1..100 is the 95th time");
    check (closeTo (ss::summarizeFrameTimes ({ 7.0 }).p95, 7.0), "p95 of one time is that time");
    const auto skipped = ss::summarizeFrameTimes ({ 2.0, std::nan (""), 4.0 });
    check (skipped.count == 2 && closeTo (skipped.median, 3.0), "non-finite times are skipped");
}

void windowKeepsTheLast120()
{
    ss::FrameTimeWindow window;
    check (window.summary().count == 0, "a new window is empty");
    for (int i = 0; i < 200; ++i) window.push (static_cast<double> (i));
    check (window.size() == ss::FrameTimeWindow::kCapacity, "the window holds 120 times");
    const auto summary = window.summary();
    check (closeTo (summary.median, 139.5), "the window drops the oldest times (median of 80..199)");
    window.push (std::nan (""));
    check (window.size() == ss::FrameTimeWindow::kCapacity, "a non-finite time is not pushed");
    window.clear();
    check (window.size() == 0, "clear empties the window");
}

void phasesSerialiseMeasuredOrAbsent()
{
    pt::FramePhaseTimes phases;
    phases.set (pt::FramePhase::SceneLower, 1.5);
    phases.set (pt::FramePhase::Trace, 500.0);
    phases.set (pt::FramePhase::Upload, 0.0);
    phases.markAbsent (pt::FramePhase::Readback, "the CPU device accumulates into host memory");
    phases.accelBuild = "reused";
    nlohmann::json stats = nlohmann::json::object();
    ss::appendPhaseTimes (stats, phases, 1000000);
    check (stats["sceneLowerMs"] == 1.5 && stats["traceMs"] == 500.0, "measured phases carry their time");
    check (stats.contains ("uploadMs") && stats["uploadMs"] == 0.0, "a measured zero stays a zero");
    check (stats["accelBuild"] == "reused", "accelBuild names what the device did");
    check (closeTo (stats["samplesPerSecond"].get<double>(), 2000000.0), "samplesPerSecond is samples over trace seconds");
    check (! stats.contains ("readbackMs"), "an absent phase has no key");
    check (stats["timingsAbsent"]["readbackMs"] == "the CPU device accumulates into host memory",
           "an absent phase names its reason");
    check (stats["timingsAbsent"].contains ("denoiseMs"), "an unmeasured phase without a reason still says so");

    pt::FramePhaseTimes untraced;
    nlohmann::json empty = nlohmann::json::object();
    ss::appendPhaseTimes (empty, untraced, 0);
    check (! empty.contains ("samplesPerSecond") && empty["timingsAbsent"].contains ("samplesPerSecond"),
           "no trace phase leaves samplesPerSecond absent with a reason");
    check (! empty.contains ("accelBuild") && empty["timingsAbsent"].contains ("accelBuild"),
           "no pass leaves accelBuild absent with a reason");
}

void phasesMergeAcrossLayers()
{
    pt::FramePhaseTimes first;
    first.set (pt::FramePhase::Trace, 10.0);
    first.markAbsent (pt::FramePhase::Readback, "cpu");
    pt::FramePhaseTimes second;
    second.set (pt::FramePhase::Trace, 5.0);
    second.set (pt::FramePhase::Readback, 2.0);
    second.accelBuild = "build";
    first.merge (second);
    check (closeTo (first.get (pt::FramePhase::Trace), 15.0), "layer traces add");
    check (closeTo (first.get (pt::FramePhase::Readback), 2.0) && first.absentReason (pt::FramePhase::Readback) == nullptr,
           "a phase one layer measured replaces the other's absence");
    check (std::string (first.accelBuild) == "build", "the merged frame names the latest layer's build");
}

void convergenceSerialises()
{
    nlohmann::json stats = nlohmann::json::object();
    ss::appendConvergence (stats, 12.0, std::nullopt, "the accumulation has not reached its target yet");
    check (stats["firstSampleMs"] == 12.0, "firstSampleMs carries its time");
    check (! stats.contains ("targetMs")
               && stats["timingsAbsent"]["targetMs"] == "the accumulation has not reached its target yet",
           "a pending targetMs is absent with the reason");
    nlohmann::json done = nlohmann::json::object();
    ss::appendConvergence (done, 12.0, 900.0, nullptr);
    check (done["targetMs"] == 900.0 && ! done.contains ("timingsAbsent"), "both measured leaves no absence object");
}

void realtimeSerialises()
{
    ss::RealtimeFrameTimes metal;
    metal.gpuAbsent = "not measured on Metal yet";
    metal.cpuMs = 3.0;
    nlohmann::json stats = nlohmann::json::object();
    ss::appendRealtimeTimes (stats, metal);
    check (! stats.contains ("gpuMs") && stats["timingsAbsent"]["gpuMs"] == "not measured on Metal yet",
           "Metal reports gpuMs absent, never zero");
    check (stats["cpuMs"] == 3.0, "cpuMs is reported");
    check (! stats.contains ("gpuMedianMs"), "an export has no window");

    ss::RealtimeFrameTimes viewport;
    viewport.gpuMs = 8.0;
    viewport.cpuMs = 1.0;
    viewport.window = ss::summarizeFrameTimes ({ 8.0, 10.0, 12.0 });
    nlohmann::json live = nlohmann::json::object();
    ss::appendRealtimeTimes (live, viewport);
    check (live["gpuMs"] == 8.0 && live["gpuMedianMs"] == 10.0 && live["gpuP95Ms"] == 12.0,
           "the viewport reports the frame and the window");
    check (closeTo (live["fps"].get<double>(), 100.0) && live["gpuWindowFrames"] == 3, "fps is 1000 over the median");
    check (! live.contains ("timingsAbsent"), "all measured leaves no absence object");
}

void measuredFramesSummarise()
{
    nlohmann::json first { { "pathTraceStats", { { "traceMs", 10.0 }, { "samplesAdded", 4 }, { "accelBuild", "build" },
                                                 { "engine", "pathTrace" } } } };
    nlohmann::json second { { "pathTraceStats", { { "traceMs", 30.0 }, { "samplesAdded", 4 }, { "accelBuild", "build" } } } };
    nlohmann::json records = nlohmann::json::array();
    records.push_back (ss::measuredFrameRecord (first, 0.0, 100.0));
    records.push_back (ss::measuredFrameRecord (second, 1.0 / 30.0, 300.0));
    check (! records[0]["pathTraceStats"].contains ("engine"), "a frame record keeps only timing keys");
    check (records[0]["pathTraceStats"]["accelBuild"] == "build", "a frame record keeps accelBuild");
    const auto reply = ss::measuredFramesReply (records);
    check (reply["frames"] == 2 && reply["frameRecords"].size() == 2, "the reply lists every frame");
    check (reply["summary"]["wallMs"]["median"] == 200.0 && reply["summary"]["wallMs"]["p95"] == 300.0,
           "wall time is summarised");
    check (reply["summary"]["pathTrace.traceMs"]["median"] == 20.0, "each phase is summarised under its engine");
    check (! reply["summary"].contains ("pathTrace.samplesAdded"), "sample counts are not summarised as times");
}
} // namespace

int main()
{
    summaryIsMedianAndNearestRankP95();
    windowKeepsTheLast120();
    phasesSerialiseMeasuredOrAbsent();
    phasesMergeAcrossLayers();
    convergenceSerialises();
    realtimeSerialises();
    measuredFramesSummarise();
    std::printf ("render speed stats: %d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
