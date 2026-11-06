#pragma once

// The additive pathTraceStats object of the composite_frame_probe and
// viewport_info replies. Raster renders report nothing, so their replies keep
// exactly the keys they had.

#include "fixture_scene_renderer.h"
#include "render_speed_stats.h"

#include <nlohmann/json.hpp>

#include <optional>
#include <string>

namespace videorender::fixture3d
{

inline void appendPathTraceStats (nlohmann::json& reply, const std::optional<PathTraceFrameStats>& stats)
{
    if (! stats.has_value())
        return;
    reply["pathTraceStats"] = {
        { "engine", "pathTrace" },
        // The frame's pixels are this film. compositorBackend only presents them.
        { "imageSource", "pathTraceFilm" },
        { "mode", stats->mode == SceneEngineMode::Fast ? "fast" : "accurate" },
        { "use", stats->use == RenderUse::Export ? "export" : "preview" },
        { "samplesAdded", stats->samplesAdded },
        { "samplesPerPixel", stats->samplesPerPixel },
        { "filmWidth", stats->filmWidth },
        { "filmHeight", stats->filmHeight },
        { "nonFiniteSamples", stats->nonFiniteSamples },
        { "bounceLimitPaths", stats->bounceLimitPaths },
        { "transparentLimitPaths", stats->transparentLimitPaths },
        { "limitEndedPaths", stats->limitEndedPaths },
        { "lightPaths", stats->lightPaths },
        { "causticSplats", stats->causticSplats },
        { "deviceMemoryFreeBytes", stats->deviceMemoryFreeBytes },
        { "deviceMemoryTotalBytes", stats->deviceMemoryTotalBytes },
        { "temporalDenoise", stats->temporalDenoise },
        { "renderActivity", stats->renderActivity },
        { "sampleTarget", stats->sampleTarget },
        { "maxBounces", stats->maxBounces },
        { "denoiser", stats->denoiser },
        { "denoiserRequested", stats->denoiserRequested },
        { "denoiserReason", stats->denoiserReason != nullptr ? *stats->denoiserReason : std::string() },
        { "device", stats->device },
        { "deviceRequested", stats->deviceRequested },
        { "deviceReason", stats->deviceReason != nullptr ? *stats->deviceReason : std::string() },
        { "fallbackReason", stats->fallbackReason != nullptr ? *stats->fallbackReason : std::string() },
        { "obsoletePassesCancelled", stats->obsoletePasses }
    };
    // Where the frame's time went (pathtrace/phase_times.h), and for the viewport how
    // long its accumulation took to show a sample and to reach the target.
    auto& traced = reply["pathTraceStats"];
    speedstats::appendPhaseTimes (traced, stats->phases, stats->phaseSamples);
    if (stats->use == RenderUse::Preview)
        speedstats::appendConvergence (traced, stats->firstSampleMs, stats->targetMs, stats->convergenceAbsent);
}

// The additive realtimeStats object of the same replies. A Raster or Path Trace
// render has none.
inline void appendRealtimeStats (nlohmann::json& reply, const std::optional<RealtimeFrameStats>& stats)
{
    if (! stats.has_value())
        return;
    reply["realtimeStats"] = {
        { "engine", "realtime" },
        { "imageSource", stats->realtimeProgram ? "realtimeProgram" : "rasterProgram" },
        { "mode", stats->mode == SceneEngineMode::Fast ? "fast" : "accurate" },
        { "use", stats->use == RenderUse::Export ? "export" : "preview" },
        { "causticsNeedPathTrace", stats->causticsNeedPathTrace },
        { "reflectionsSkipped", stats->reflectionsSkipped },
        { "width", stats->width },
        { "height", stats->height }
    };
    // Named so the header can say solid is on, or that this device can't draw it.
    if (stats->viewportSolidRequested)
        reply["realtimeStats"]["viewportShading"] = stats->viewportSolidApplied ? "solid" : "solidUnavailable";
    // The frame's GPU and host cost, and for the viewport the window's median and p95.
    speedstats::appendRealtimeTimes (reply["realtimeStats"], stats->times);
}

// Both objects of the latest Export render, for each later frame of a measureFrames probe.
inline void appendLastExportStats (nlohmann::json& reply)
{
    appendPathTraceStats (reply, lastPathTraceStats (RenderUse::Export));
    appendRealtimeStats (reply, lastRealtimeStats (RenderUse::Export));
}

} // namespace videorender::fixture3d
