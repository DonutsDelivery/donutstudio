#pragma once

// The additive pathTraceStats object of the composite_frame_probe and
// viewport_info replies. Raster renders report nothing, so their replies keep
// exactly the keys they had.

#include "fixture_scene_renderer.h"

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
        { "temporalDenoise", stats->temporalDenoise },
        { "renderActivity", stats->renderActivity },
        { "sampleTarget", stats->sampleTarget },
        { "denoiser", stats->denoiser },
        { "device", stats->device },
        { "deviceRequested", stats->deviceRequested },
        { "deviceReason", stats->deviceReason != nullptr ? *stats->deviceReason : std::string() }
    };
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
        { "width", stats->width },
        { "height", stats->height }
    };
}

} // namespace videorender::fixture3d
