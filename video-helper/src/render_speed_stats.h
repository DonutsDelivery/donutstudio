#pragma once

// The speed fields of the pathTraceStats and realtimeStats replies. Only
// nlohmann and two standard-library headers, so the reply tests link nothing else.
//
// A measurement that is missing leaves its key out and names why under
// timingsAbsent. A missing time is never written as zero.

#include "frame_time_window.h"
#include "pathtrace/phase_times.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace videorender::speedstats
{

namespace pt = videohelper::pathtrace;

inline void writeSummary (nlohmann::json& object, const char* prefix, const FrameTimeSummary& summary)
{
    object[std::string (prefix) + "MedianMs"] = summary.median;
    object[std::string (prefix) + "P95Ms"] = summary.p95;
}

// The phase times, accelBuild and samplesPerSecond into a pathTraceStats object.
inline void appendPhaseTimes (nlohmann::json& stats, const pt::FramePhaseTimes& phases, std::uint64_t traceSamples)
{
    auto absent = nlohmann::json::object();
    for (std::size_t i = 0; i < pt::kFramePhaseCount; ++i)
    {
        const auto phase = static_cast<pt::FramePhase> (i);
        const char* key = pt::framePhaseKey (phase);
        if (phases.has (phase))
            stats[key] = phases.get (phase);
        else
            absent[key] = phases.absentReason (phase) != nullptr ? phases.absentReason (phase) : "not measured by this frame";
    }
    if (phases.accelBuild != nullptr)
        stats["accelBuild"] = phases.accelBuild;
    else
        absent["accelBuild"] = "no pass traced this frame";
    const double rate = pt::samplesPerSecond (traceSamples, phases);
    if (std::isfinite (rate))
        stats["samplesPerSecond"] = rate;
    else
        absent["samplesPerSecond"] = phases.has (pt::FramePhase::Trace) ? "the trace phase added no time" : "the trace phase was not measured";
    if (! absent.empty())
        stats["timingsAbsent"] = std::move (absent);
}

// firstSampleMs and targetMs of a Preview accumulation, or their absence.
inline void appendConvergence (nlohmann::json& stats, const std::optional<double>& firstSampleMs,
                               const std::optional<double>& targetMs, const char* absentReason)
{
    const char* reason = absentReason != nullptr ? absentReason : "not measured";
    auto& absent = stats["timingsAbsent"];
    if (firstSampleMs.has_value())
        stats["firstSampleMs"] = *firstSampleMs;
    else
        absent["firstSampleMs"] = reason;
    if (targetMs.has_value())
        stats["targetMs"] = *targetMs;
    else
        absent["targetMs"] = reason;
    if (absent.empty())
        stats.erase ("timingsAbsent");
}

inline void appendRealtimeTimes (nlohmann::json& stats, const RealtimeFrameTimes& times)
{
    auto absent = nlohmann::json::object();
    if (times.gpuMs.has_value())
        stats["gpuMs"] = *times.gpuMs;
    else
        absent["gpuMs"] = times.gpuAbsent != nullptr ? times.gpuAbsent : "not measured";
    if (times.cpuMs.has_value())
        stats["cpuMs"] = *times.cpuMs;
    else
        absent["cpuMs"] = "not measured";
    if (times.window.has_value() && times.window->count > 0)
    {
        stats["gpuWindowFrames"] = times.window->count;
        writeSummary (stats, "gpu", *times.window);
        if (times.window->median > 0.0)
            stats["fps"] = 1000.0 / times.window->median;
        else
            absent["fps"] = "the median GPU time is zero";
    }
    else if (times.window.has_value())
        absent["gpuMedianMs"] = times.gpuAbsent != nullptr ? times.gpuAbsent : "no frame has been timed yet";
    if (! absent.empty())
        stats["timingsAbsent"] = std::move (absent);
}

// One frame of a measureFrames probe: its timeline time, the composite call's wall
// time, and the timing keys of the pathTraceStats and realtimeStats in stats.
inline nlohmann::json measuredFrameRecord (const nlohmann::json& stats, double timelineSec, double wallMs)
{
    nlohmann::json entry { { "timelineSec", timelineSec }, { "wallMs", wallMs } };
    for (const char* engine : { "pathTraceStats", "realtimeStats" })
    {
        if (! stats.contains (engine))
            continue;
        auto times = nlohmann::json::object();
        for (const auto& [key, value] : stats[engine].items())
            if ((key.size() > 2 && key.compare (key.size() - 2, 2, "Ms") == 0)
                || key == "accelBuild" || key == "samplesPerSecond" || key == "samplesAdded"
                || key == "device" || key == "timingsAbsent")
                times[key] = value;
        entry[engine] = std::move (times);
    }
    return entry;
}

// The measure object: the frame records, and the median and p95 of each time over
// the frames that reported it, keyed "wallMs", "pathTrace.<key>" or "realtime.<key>".
inline nlohmann::json measuredFramesReply (nlohmann::json records)
{
    std::map<std::string, std::vector<double>> series;
    for (const auto& entry : records)
    {
        if (entry.contains ("wallMs") && entry["wallMs"].is_number())
            series["wallMs"].push_back (entry["wallMs"].get<double>());
        for (const auto& [engine, prefix] : { std::pair<const char*, const char*> { "pathTraceStats", "pathTrace." },
                                              std::pair<const char*, const char*> { "realtimeStats", "realtime." } })
            if (entry.contains (engine))
                for (const auto& [key, value] : entry[engine].items())
                    if (value.is_number() && key != "samplesAdded")
                        series[prefix + key].push_back (value.get<double>());
    }
    auto summary = nlohmann::json::object();
    for (const auto& [key, values] : series)
    {
        const auto summarized = summarizeFrameTimes (values);
        summary[key] = { { "frames", summarized.count }, { "median", summarized.median }, { "p95", summarized.p95 } };
    }
    const auto frames = records.size();
    return { { "frames", frames }, { "frameRecords", std::move (records) }, { "summary", std::move (summary) } };
}

} // namespace videorender::speedstats
