#pragma once

#include <cmath>
#include <string>

namespace videowire
{
enum class SourceKind
{
    Unspecified,
    Media,
    Shader,
    Particles,
    Score,
    Adjustment,
};

inline SourceKind sourceKindFromWire (const std::string& value) noexcept
{
    if (value == "media")      return SourceKind::Media;
    if (value == "shader")     return SourceKind::Shader;
    if (value == "particles")  return SourceKind::Particles;
    if (value == "score")      return SourceKind::Score;
    if (value == "adjustment") return SourceKind::Adjustment;
    return SourceKind::Unspecified;
}

inline const char* sourceKindToWire (SourceKind value) noexcept
{
    switch (value)
    {
        case SourceKind::Media:      return "media";
        case SourceKind::Shader:     return "shader";
        case SourceKind::Particles:  return "particles";
        case SourceKind::Score:      return "score";
        case SourceKind::Adjustment: return "adjustment";
        default:                     return "unspecified";
    }
}

inline SourceKind resolveSourceKind (SourceKind structuredKind,
                                     bool structuredAdjustment,
                                     const std::string& legacyPath) noexcept
{
    // Structured source identity is authoritative. Compatibility fields and
    // sentinels are consulted only when the typed field is absent.
    if (structuredKind != SourceKind::Unspecified)
        return structuredKind;
    if (structuredAdjustment)
        return SourceKind::Adjustment;
    if (legacyPath.rfind ("gen://adjustment", 0) == 0)
        return SourceKind::Adjustment;
    if (legacyPath.rfind ("gen://particles", 0) == 0)
        return SourceKind::Particles;
    if (legacyPath.rfind ("gen://score", 0) == 0)
        return SourceKind::Score;
    if (legacyPath.rfind ("gen://shader", 0) == 0)
        return SourceKind::Shader;
    return SourceKind::Media;
}

/** Tolerance for clip boundaries on the display timeline.

    An export frame time is rangeStart + n / fps, while a clip start is the
    tempo map's seconds for its beat. Both name the same instant at a seam but
    reach it through different float paths, so they can differ by a few ulps.
    At 105 bpm and 30 fps, frame 840 of an export from beat 16 came out at
    37.14285714285714 s and the clip starting at beat 65 at 37.142857142857146 s,
    so the old clip played one frame late. At the video-to-shader cut at beat
    58 the video was still active instead of the shader, sampling its out
    point, and the frame came out as bare canvas. One microsecond is far
    below any frame period and far above double error. */
inline constexpr double kDisplayBoundaryEpsilonSec = 1.0e-6;

/** Whether displaySeconds falls in [startSeconds, startSeconds + durationSeconds)
    with both edges snapped by kDisplayBoundaryEpsilonSec: a frame at a clip's
    start shows that clip, and a frame at a clip's end no longer shows it. */
inline bool displayTimeInSpan (double displaySeconds,
                               double startSeconds,
                               double durationSeconds) noexcept
{
    return displaySeconds >= startSeconds - kDisplayBoundaryEpsilonSec
        && displaySeconds < startSeconds + durationSeconds - kDisplayBoundaryEpsilonSec;
}

template <typename Segment>
bool segmentCoversDisplayTime (const Segment& segment, double displaySeconds) noexcept
{
    const double duration = (segment.outSec - segment.inSec)
                          / (segment.rate > 1.0e-9 ? segment.rate : 1.0e-9);
    return displayTimeInSpan (displaySeconds, segment.displayStartSec, duration);
}

/** A frame time within kDisplayBoundaryEpsilonSec of a segment start, moved
    onto that start. Code downstream of the activity test maps the frame into
    the clip (source seconds, scene time, frame owners) and some of it rejects
    a sample a few ulps before the clip's in point, so the export puts the
    frame exactly on the seam instead of only widening the activity test. */
template <typename SegmentRange>
double snapDisplayTimeToSegmentStart (const SegmentRange& segments,
                                      double displaySeconds) noexcept
{
    for (const auto& segment : segments)
        if (std::abs (displaySeconds - segment.displayStartSec) <= kDisplayBoundaryEpsilonSec)
            return segment.displayStartSec;
    return displaySeconds;
}

template <typename SegmentRange, typename Segment>
const Segment* resolveTransitionFrom (const SegmentRange& segments,
                                      const Segment& current,
                                      double abutEpsilon = 1.0e-3) noexcept
{
    const bool hasBinding = current.transitionFromClipId >= 0
                         || current.transitionToClipId >= 0;
    const Segment* best = nullptr;
    if (hasBinding)
    {
        if (current.transitionToClipId >= 0
            && current.transitionToClipId != current.clipId)
            return nullptr;
        for (const auto& candidate : segments)
            if (&candidate != &current
                && candidate.clipId == current.transitionFromClipId
                && candidate.displayStartSec <= current.displayStartSec
                && (best == nullptr
                    || candidate.displayStartSec > best->displayStartSec))
                best = &candidate;
        return best;
    }

    for (const auto& candidate : segments)
        if (&candidate != &current && candidate.trackLayer == current.trackLayer
            && candidate.displayStartSec < current.displayStartSec
            && (best == nullptr
                || candidate.displayStartSec > best->displayStartSec))
            best = &candidate;
    if (best == nullptr)
        return nullptr;
    const double duration = (best->outSec - best->inSec)
                          / (best->rate > 1.0e-9 ? best->rate : 1.0e-9);
    return best->displayStartSec + duration >= current.displayStartSec - abutEpsilon
         ? best : nullptr;
}

template <typename SegmentRange>
const typename SegmentRange::value_type* resolveClipSegmentAtDisplayTime (
    const SegmentRange& segments, int clipId, double displaySeconds) noexcept
{
    using Segment = typename SegmentRange::value_type;
    const Segment* best = nullptr;
    for (const auto& candidate : segments)
    {
        if (candidate.clipId != clipId || candidate.rate <= 1.0e-9)
            continue;
        const double duration = (candidate.outSec - candidate.inSec) / candidate.rate;
        if (! displayTimeInSpan (displaySeconds, candidate.displayStartSec, duration))
            continue;
        if (best == nullptr || candidate.displayStartSec > best->displayStartSec)
            best = &candidate;
    }
    return best;
}

/** Segment a shader transition's Layer Source reads at display time.

    The saved clip is normally the one that ends where the transition clip
    starts, so at every moment the transition plays that clip is already past
    its end. resolveClipSegmentAtDisplayTime() rejects it there, which left
    every abutting Fade Transition with "Layer Source clip is unavailable".
    This returns the exact segment when one covers the time. Otherwise it
    returns the latest segment of that clip that ended at or after the start
    of the transition clip (within abutEpsilon), so the caller can hold its
    last frame. A clip that ended well before the transition clip started, or
    that has not started yet, still resolves to nothing. */
template <typename SegmentRange>
const typename SegmentRange::value_type* resolveTransitionSourceSegmentAtDisplayTime (
    const SegmentRange& segments, int clipId, double displaySeconds,
    double transitionClipStartSeconds, double abutEpsilon = 1.0e-3) noexcept
{
    if (const auto* exact = resolveClipSegmentAtDisplayTime (segments, clipId, displaySeconds))
        return exact;

    using Segment = typename SegmentRange::value_type;
    const Segment* best = nullptr;
    double bestEnd = 0.0;
    for (const auto& candidate : segments)
    {
        if (candidate.clipId != clipId || candidate.rate <= 1.0e-9)
            continue;
        const double end = candidate.displayStartSec
                         + (candidate.outSec - candidate.inSec) / candidate.rate;
        if (end > displaySeconds || end < transitionClipStartSeconds - abutEpsilon)
            continue;
        if (best == nullptr || end > bestEnd)
        {
            best = &candidate;
            bestEnd = end;
        }
    }
    return best;
}

/** Duration of one source frame for holding a transition source's last
    frame. It uses the segment's source frame rate when it carries one, and a
    small fixed step otherwise. */
inline double transitionHeldFrameSeconds (double sourceFps) noexcept
{
    return sourceFps > 1.0e-3 ? 1.0 / sourceFps : 1.0e-3;
}

/** Source time a transition's Layer Source reads at display time. It follows
    the clip's own trim and speed. Past the out point it holds the last
    decodable frame, one frame before the out point and never before the in
    point, because a decoder has no frame at the out point itself. Before the
    in point it holds the in point. Only media sources use this time. Shader
    and generator sources run on the display-time clock. */
template <typename Segment>
double transitionSourceSeconds (const Segment& segment, double displaySeconds,
                                double frameSeconds = 1.0e-3) noexcept
{
    const double raw = segment.inSec + (displaySeconds - segment.displayStartSec) * segment.rate;
    if (raw < segment.inSec)
        return segment.inSec;
    if (raw < segment.outSec)
        return raw;
    const double last = segment.outSec - frameSeconds;
    return last > segment.inSec ? last : segment.inSec;
}

/** Seconds a time-driven shader transition runs. A leading-edge transition
    the clip already carries sets the span. Without one the transition runs
    for defaultSeconds, or for the whole clip when it is shorter. */
inline double shaderTransitionSpanSeconds (double leadingTransitionSeconds,
                                           double clipSeconds,
                                           double defaultSeconds = 1.0) noexcept
{
    if (leadingTransitionSeconds > 1.0e-6)
        return leadingTransitionSeconds;
    return clipSeconds > 1.0e-6 && clipSeconds < defaultSeconds ? clipSeconds : defaultSeconds;
}

/** Linear progress of a time-driven shader transition at display time, from 0
    at startSeconds to 1 after spanSeconds. */
inline double shaderTransitionFraction (double displaySeconds, double startSeconds,
                                        double spanSeconds) noexcept
{
    if (! (spanSeconds > 1.0e-9))
        return 1.0;
    const double value = (displaySeconds - startSeconds) / spanSeconds;
    return value < 0.0 ? 0.0 : (value > 1.0 ? 1.0 : value);
}
} // namespace videowire
