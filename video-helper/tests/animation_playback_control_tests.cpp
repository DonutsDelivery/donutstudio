#include "VisualAnimationPlaybackControl.h"
#include "VisualImportedAnimationOperationContract.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>

namespace
{
using namespace visualanimation;

int failures = 0;

void check (bool condition, const char* message)
{
    if (condition)
        return;
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", message);
}

bool near (double actual, double expected, double tolerance = 1.0e-9)
{
    return std::abs(actual - expected) <= tolerance;
}
} // namespace

int main()
{
    const std::array<double, 2> times { 0.0, 2.0 };
    const std::array<float, 6> values { 0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f };
    const TrackView track {
        TrackId { 1 }, TargetId { 1 }, Channel::Translation, Interpolation::Linear,
        times.data(), values.data(), times.size(), values.size(), 0
    };
    const ClipView view { ClipId { 1 }, 2.0, &track, 1 };
    std::string error;
    const auto clip = Clip::create(view, {}, error);
    check(clip && error.empty(), "playback-control fixture is admitted");
    if (! clip)
        return 1;

    PlaybackControlRequest timeline;
    timeline.timelineSeconds = 3.0;
    timeline.clipStartTimelineSeconds = 1.0;
    timeline.speed = 2.0;
    timeline.offsetSeconds = 0.25;
    timeline.weight = 0.75;
    timeline.playback = Playback::Clamp;
    const auto mappedTimeline = resolvePlaybackControl(*clip, timeline, {}, error);
    const auto timelineSample = mappedTimeline
        ? sample(*clip, mappedTimeline->sample, error) : std::nullopt;
    check(mappedTimeline && near(mappedTimeline->sample.timeSeconds, 4.25)
          && near(mappedTimeline->weight, 0.75) && timelineSample
          && near(timelineSample->sampleTimeSeconds(), 2.0),
          "timeline sync applies clip start, speed, offset, weight, and clamp explicitly");

    PlaybackControlRequest beat;
    beat.timeSource = TimeSource::BeatSync;
    beat.timelineBeat = 3.0;
    beat.clipStartBeat = 1.0;
    beat.beatsPerLoop = 4.0;
    beat.playback = Playback::Loop;
    const auto mappedBeat = resolvePlaybackControl(*clip, beat, {}, error);
    const auto beatSample = mappedBeat ? sample(*clip, mappedBeat->sample, error) : std::nullopt;
    check(mappedBeat && near(mappedBeat->sample.timeSeconds, 1.0)
          && beatSample && near(beatSample->sampleTimeSeconds(), 1.0),
          "beat sync maps an authored beat span onto one clip duration");

    beat.timelineBeat = 5.0;
    const auto loopBoundary = resolvePlaybackControl(*clip, beat, {}, error);
    const auto boundarySample = loopBoundary
        ? sample(*clip, loopBoundary->sample, error) : std::nullopt;
    check(boundarySample && near(boundarySample->sampleTimeSeconds(), 0.0),
          "beat-sync loop endpoint maps to the first clip frame");

    beat.speed = -1.0;
    beat.timelineBeat = 3.0;
    beat.offsetSeconds = 2.0;
    const auto reverse = resolvePlaybackControl(*clip, beat, {}, error);
    const auto repeated = resolvePlaybackControl(*clip, beat, {}, error);
    check(reverse && repeated && near(reverse->sample.timeSeconds, 1.0)
          && reverse->sample.timeSeconds == repeated->sample.timeSeconds
          && reverse->weight == repeated->weight,
          "reverse playback and equal-time reevaluation are deterministic");

    auto trimmed = timeline;
    trimmed.timelineSeconds = 1.75;
    trimmed.clipStartTimelineSeconds = 1.0;
    trimmed.speed = 1.0;
    trimmed.offsetSeconds = 0.0;
    trimmed.trimStartSeconds = 0.5;
    trimmed.trimEndSeconds = 1.5;
    trimmed.playback = Playback::Loop;
    const auto mappedTrim = resolvePlaybackControl(*clip, trimmed, {}, error);
    const auto trimSample = mappedTrim
        ? sample(*clip, mappedTrim->sample, error) : std::nullopt;
    check(mappedTrim && near(mappedTrim->sample.timeSeconds, 1.25)
          && near(mappedTrim->sample.rangeStartSeconds, 0.5)
          && near(mappedTrim->sample.rangeEndSeconds, 1.5)
          && trimSample && near(trimSample->sampleTimeSeconds(), 1.25),
          "trimmed playback offsets into the admitted clip range");
    trimmed.timelineSeconds = 2.25;
    const auto trimLoop = resolvePlaybackControl(*clip, trimmed, {}, error);
    const auto trimLoopSample = trimLoop
        ? sample(*clip, trimLoop->sample, error) : std::nullopt;
    check(trimLoopSample && near(trimLoopSample->sampleTimeSeconds(), 0.75),
          "trimmed loop playback wraps inside the selected range");
    trimmed.timeSource = TimeSource::BeatSync;
    trimmed.timelineBeat = 2.0;
    trimmed.clipStartBeat = 0.0;
    trimmed.beatsPerLoop = 4.0;
    const auto trimBeat = resolvePlaybackControl(*clip, trimmed, {}, error);
    const auto trimBeatSample = trimBeat
        ? sample(*clip, trimBeat->sample, error) : std::nullopt;
    check(trimBeatSample && near(trimBeatSample->sampleTimeSeconds(), 1.0),
          "beat sync maps one beat span onto the trimmed duration");

    auto invalid = timeline;
    invalid.speed = std::numeric_limits<double>::infinity();
    check(! resolvePlaybackControl(*clip, invalid, {}, error)
          && error == "animation playback speed is nonfinite or outside the admitted range",
          "nonfinite playback speed is rejected");
    invalid = timeline;
    invalid.weight = -0.1;
    check(! resolvePlaybackControl(*clip, invalid, {}, error)
          && error == "animation playback weight must be finite and in [0,1]",
          "out-of-range blend weight is rejected");
    invalid = beat;
    invalid.beatsPerLoop = 0.0;
    check(! resolvePlaybackControl(*clip, invalid, {}, error)
          && error == "animation beat-sync length is outside the admitted range",
          "zero beat-sync duration is rejected");
    invalid = timeline;
    invalid.trimStartSeconds = 1.5;
    invalid.trimEndSeconds = 1.0;
    check(! resolvePlaybackControl(*clip, invalid, {}, error)
          && error == "animation playback trim range is invalid or outside the admitted clip",
          "reversed trim ranges are rejected");
    invalid = timeline;
    invalid.timeSource = static_cast<TimeSource>(99);
    check(! resolvePlaybackControl(*clip, invalid, {}, error)
          && error == "animation time source is unsupported",
          "unknown time-source values fail closed");

    PlaybackControlLimits narrow;
    narrow.maxAbsoluteSpeed = 1.0;
    check(! resolvePlaybackControl(*clip, timeline, narrow, error)
          && error == "animation playback speed is nonfinite or outside the admitted range",
          "caller playback bounds are enforced");

    visualanimationimport::Request operation;
    operation.sourceStableId = 1;
    operation.deformationStableId = 2;
    operation.schedule = { 1, 2 };
    operation.asset = { "trimmed-model", 3, std::string(64, 'a'),
                        "model/gltf-binary", 4096 };
    operation.animationClipStableId = 4;
    operation.meshStableId = 5;
    operation.clipName = "Walk";
    operation.playback.trimStartSeconds = 0.5;
    operation.playback.trimEndSeconds = 1.5;
    const auto encoded = visualanimationoperation::encode(operation);
    visualanimationimport::Request decoded;
    check(encoded.rfind("visual.imported-animation.deformation.v5\n", 0) == 0
          && visualanimationoperation::decode(encoded, decoded, error)
          && decoded.animationClipStableId == operation.animationClipStableId
          && decoded.meshStableId == operation.meshStableId
          && near(decoded.playback.trimStartSeconds, 0.5)
          && near(decoded.playback.trimEndSeconds, 1.5),
          "operation V5 preserves typed selectors and the admitted trim range");

    std::fprintf(stderr, failures == 0
        ? "Animation playback-control checks passed\n"
        : "%d animation playback-control checks failed\n", failures);
    return failures == 0 ? 0 : 1;
}
