#pragma once

#include "VisualAnimationClip.h"

#include <optional>
#include <string>

namespace visualanimation
{
enum class TimeSource
{
    Timeline,
    BeatSync
};

struct PlaybackControlLimits
{
    double maxAbsoluteTimelineSeconds = 365.0 * 24.0 * 60.0 * 60.0;
    double maxAbsoluteBeat = 1000000000.0;
    double maxBeatsPerLoop = 1000000.0;
    double maxAbsoluteSpeed = 64.0;
    double maxAbsoluteOffsetSeconds = 24.0 * 60.0 * 60.0;
    double maxTrimEndpointSeconds = 24.0 * 60.0 * 60.0;
};

struct PlaybackControlRequest
{
    TimeSource timeSource = TimeSource::Timeline;
    double timelineSeconds = 0.0;
    double clipStartTimelineSeconds = 0.0;
    double timelineBeat = 0.0;
    double clipStartBeat = 0.0;
    double beatsPerLoop = 4.0;
    double speed = 1.0;
    double offsetSeconds = 0.0;
    double trimStartSeconds = 0.0;
    // Zero selects the immutable clip duration.
    double trimEndSeconds = 0.0;
    double weight = 1.0;
    Playback playback = Playback::Clamp;
};

struct ResolvedPlaybackControl
{
    SampleRequest sample;
    double weight = 1.0;
};

std::optional<ResolvedPlaybackControl> resolvePlaybackControl (
    const Clip& clip,
    const PlaybackControlRequest& request,
    const PlaybackControlLimits& limits,
    std::string& error) noexcept;
} // namespace visualanimation
