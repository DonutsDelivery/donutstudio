#pragma once

#include "beat_timeline.h"
#include "../../shared/VisualAnimationPlaybackControl.h"
#include "../../shared/VisualAnimationDeformationEvaluation.h"

#include <cmath>
#include <limits>

namespace videohelper::importedanimation
{
// Beat Sync uses the project tempo map at an exact project frame, independently
// of the media source frame. Authored beat values remain immutable offsets.
inline bool applyImportedAnimationPlaybackClock(
    visualanimation::PlaybackControlRequest& playback,
    const visualdeformation::RationalFrameTime& sourceFrame,
    const videotime::BeatTimeline* beatTimeline,
    double projectSeconds,
    std::string& error)
{
    if (sourceFrame.rateNumerator == 0 || sourceFrame.rateDenominator == 0)
    { error = "Imported animation frame rate is invalid"; return false; }
    const visualanimation::PlaybackControlLimits limits;
    if (playback.timeSource == visualanimation::TimeSource::Timeline)
    {
        playback.timelineSeconds += static_cast<double>(sourceFrame.frame)
            * sourceFrame.rateDenominator / sourceFrame.rateNumerator;
        if (!std::isfinite(playback.timelineSeconds)
            || std::abs(playback.timelineSeconds) > limits.maxAbsoluteTimelineSeconds
            || !std::isfinite(playback.clipStartTimelineSeconds)
            || std::abs(playback.clipStartTimelineSeconds) > limits.maxAbsoluteTimelineSeconds)
        { error = "Imported animation timeline position exceeds the playback bound"; return false; }
        return true;
    }
    if (playback.timeSource != visualanimation::TimeSource::BeatSync || !beatTimeline)
    { error = "Imported animation Beat Sync requires the project beat timeline"; return false; }
    const auto projectFrame = projectSeconds * sourceFrame.rateNumerator / sourceFrame.rateDenominator;
    if (!std::isfinite(projectFrame)
        || projectFrame < static_cast<double>(std::numeric_limits<std::int64_t>::min())
        || projectFrame >= static_cast<double>(std::numeric_limits<std::int64_t>::max()))
    { error = "Imported animation project beat frame exceeds its signed bound"; return false; }
    const auto exactSeconds = static_cast<double>(std::llround(projectFrame))
        * sourceFrame.rateDenominator / sourceFrame.rateNumerator;
    playback.timelineBeat += beatTimeline->secondsToBeat(exactSeconds);
    if (!std::isfinite(playback.timelineBeat) || std::abs(playback.timelineBeat) > limits.maxAbsoluteBeat)
    { error = "Imported animation beat position exceeds the playback bound"; return false; }
    return true;
}
} // namespace videohelper::importedanimation
