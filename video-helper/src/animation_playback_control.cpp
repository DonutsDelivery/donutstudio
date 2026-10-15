#include "../../shared/VisualAnimationPlaybackControl.h"

#include <algorithm>
#include <cmath>

namespace visualanimation
{
namespace
{
double boundedNonnegative (double requested, double hard) noexcept
{
    if (! std::isfinite(requested) || requested < 0.0)
        return 0.0;
    return std::min(requested, hard);
}

bool inAbsoluteRange (double value, double maximum) noexcept
{
    return std::isfinite(value) && std::abs(value) <= maximum;
}
} // namespace

std::optional<ResolvedPlaybackControl> resolvePlaybackControl (
    const Clip& clip,
    const PlaybackControlRequest& request,
    const PlaybackControlLimits& requestedLimits,
    std::string& error) noexcept
{
    error.clear();
    const PlaybackControlLimits hard;
    const PlaybackControlLimits limits {
        boundedNonnegative(requestedLimits.maxAbsoluteTimelineSeconds,
                           hard.maxAbsoluteTimelineSeconds),
        boundedNonnegative(requestedLimits.maxAbsoluteBeat, hard.maxAbsoluteBeat),
        boundedNonnegative(requestedLimits.maxBeatsPerLoop, hard.maxBeatsPerLoop),
        boundedNonnegative(requestedLimits.maxAbsoluteSpeed, hard.maxAbsoluteSpeed),
        boundedNonnegative(requestedLimits.maxAbsoluteOffsetSeconds,
                           hard.maxAbsoluteOffsetSeconds),
        boundedNonnegative(requestedLimits.maxTrimEndpointSeconds,
                           hard.maxTrimEndpointSeconds)
    };

    if (! inAbsoluteRange(request.speed, limits.maxAbsoluteSpeed))
    {
        error = "animation playback speed is nonfinite or outside the admitted range";
        return std::nullopt;
    }
    if (! inAbsoluteRange(request.offsetSeconds, limits.maxAbsoluteOffsetSeconds))
    {
        error = "animation playback offset is nonfinite or outside the admitted range";
        return std::nullopt;
    }
    const auto trimEnd = request.trimEndSeconds == 0.0
        ? clip.durationSeconds() : request.trimEndSeconds;
    const bool zeroDurationDefault = clip.durationSeconds() == 0.0
        && request.trimStartSeconds == 0.0 && request.trimEndSeconds == 0.0;
    if (! std::isfinite(request.trimStartSeconds) || request.trimStartSeconds < 0.0
        || request.trimStartSeconds > limits.maxTrimEndpointSeconds
        || ! std::isfinite(trimEnd) || trimEnd <= request.trimStartSeconds
        || trimEnd > clip.durationSeconds()
        || (request.trimEndSeconds != 0.0
            && trimEnd > limits.maxTrimEndpointSeconds))
    {
        if (! zeroDurationDefault)
        {
            error = "animation playback trim range is invalid or outside the admitted clip";
            return std::nullopt;
        }
    }
    if (! std::isfinite(request.weight) || request.weight < 0.0 || request.weight > 1.0)
    {
        error = "animation playback weight must be finite and in [0,1]";
        return std::nullopt;
    }
    if (request.playback != Playback::Clamp && request.playback != Playback::Loop)
    {
        error = "animation playback mode is unsupported";
        return std::nullopt;
    }

    double sourcePosition = 0.0;
    if (request.timeSource == TimeSource::Timeline)
    {
        if (! inAbsoluteRange(request.timelineSeconds, limits.maxAbsoluteTimelineSeconds)
            || ! inAbsoluteRange(request.clipStartTimelineSeconds,
                                 limits.maxAbsoluteTimelineSeconds))
        {
            error = "animation timeline position is nonfinite or outside the admitted range";
            return std::nullopt;
        }
        sourcePosition = request.timelineSeconds - request.clipStartTimelineSeconds;
    }
    else if (request.timeSource == TimeSource::BeatSync)
    {
        if (! inAbsoluteRange(request.timelineBeat, limits.maxAbsoluteBeat)
            || ! inAbsoluteRange(request.clipStartBeat, limits.maxAbsoluteBeat))
        {
            error = "animation beat position is nonfinite or outside the admitted range";
            return std::nullopt;
        }
        if (! std::isfinite(request.beatsPerLoop) || request.beatsPerLoop <= 0.0
            || request.beatsPerLoop > limits.maxBeatsPerLoop)
        {
            error = "animation beat-sync length is outside the admitted range";
            return std::nullopt;
        }
        sourcePosition = (request.timelineBeat - request.clipStartBeat)
                       * (trimEnd - request.trimStartSeconds)
                       / request.beatsPerLoop;
    }
    else
    {
        error = "animation time source is unsupported";
        return std::nullopt;
    }

    const auto sampleTime = sourcePosition * request.speed + request.offsetSeconds;
    if (! std::isfinite(sampleTime))
    {
        error = "animation playback mapping produced a nonfinite sample time";
        return std::nullopt;
    }

    return ResolvedPlaybackControl {
        SampleRequest { sampleTime + request.trimStartSeconds, request.playback,
                        request.trimStartSeconds, trimEnd }, request.weight
    };
}
} // namespace visualanimation
