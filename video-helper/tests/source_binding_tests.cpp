#include "../src/source_binding.h"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

struct Segment
{
    int clipId = 0;
    int trackLayer = 0;
    double inSec = 0.0;
    double outSec = 1.0;
    double rate = 1.0;
    double displayStartSec = 0.0;
    int transitionFromClipId = -1;
    int transitionToClipId = -1;
};

int main()
{
    int failures = 0;
    auto check = [&failures] (bool condition, const char* message)
    {
        if (! condition)
        {
            ++failures;
            std::fprintf (stderr, "FAIL: %s\n", message);
        }
    };

    using videowire::SourceKind;
    using videowire::resolveSourceKind;

    check (resolveSourceKind (SourceKind::Media, true, "gen://adjustment")
               == SourceKind::Media,
           "typed media overrides adjustment compatibility fields");
    check (resolveSourceKind (SourceKind::Shader, false, "/tmp/movie.mov")
               == SourceKind::Shader,
           "typed shader overrides media-looking path");
    check (resolveSourceKind (SourceKind::Unspecified, true, "/tmp/movie.mov")
               == SourceKind::Adjustment,
           "structured legacy adjustment precedes path fallback");
    check (resolveSourceKind (SourceKind::Unspecified, false, "gen://particles")
               == SourceKind::Particles,
           "legacy particle sentinel remains readable");
    check (resolveSourceKind (SourceKind::Unspecified, false, "gen://shader")
               == SourceKind::Shader,
           "legacy shader sentinel remains readable");
    check (resolveSourceKind (SourceKind::Unspecified, false, "gen://score")
               == SourceKind::Score,
           "legacy score sentinel remains readable");
    check (videowire::sourceKindFromWire("score") == SourceKind::Score,
           "structured score source kind round-trips");
    check (resolveSourceKind (SourceKind::Unspecified, false, "/tmp/movie.mov")
               == SourceKind::Media,
           "ordinary legacy path resolves to media");
    check (videowire::sourceKindFromWire ("future") == SourceKind::Unspecified,
           "unknown wire values degrade to legacy fallback");

    std::vector<Segment> segments {
        { 10, 0, 0.0, 3.0, 1.0, 0.0, -1, -1 },
        { 20, 0, 0.0, 3.0, 1.0, 2.0, 10, 20 },
        { 30, 1, 0.0, 3.0, 1.0, 1.0, -1, -1 },
    };
    check (videowire::resolveTransitionFrom (segments, segments[1]) == &segments[0],
           "explicit transition A wins across timeline candidates");
    segments[1].transitionFromClipId = 30;
    check (videowire::resolveTransitionFrom (segments, segments[1]) == &segments[2],
           "explicit transition A is not constrained by track ordering");
    segments[1].transitionToClipId = 99;
    check (videowire::resolveTransitionFrom (segments, segments[1]) == nullptr,
           "conflicting explicit B fails closed instead of inferring");
    segments[1].transitionFromClipId = -1;
    segments[1].transitionToClipId = -1;
    check (videowire::resolveTransitionFrom (segments, segments[1]) == &segments[0],
           "legacy transition falls back to previous overlapping track segment");

    std::vector<Segment> splitClip {
        { 40, 0, 2.0, 4.0, 1.0, 5.0, -1, -1 },
        { 40, 0, 8.0, 12.0, 2.0, 7.0, -1, -1 },
    };
    const auto* firstSplit = videowire::resolveClipSegmentAtDisplayTime(splitClip, 40, 6.0);
    const auto* secondSplit = videowire::resolveClipSegmentAtDisplayTime(splitClip, 40, 7.5);
    check (firstSplit == &splitClip[0]
               && firstSplit->inSec + (6.0 - firstSplit->displayStartSec) * firstSplit->rate == 3.0,
           "Layer Source resolves the exact first trim segment at display time");
    check (secondSplit == &splitClip[1]
               && secondSplit->inSec + (7.5 - secondSplit->displayStartSec) * secondSplit->rate == 9.0,
           "Layer Source resolves the exact speed-ramp segment at display time");
    check (videowire::resolveClipSegmentAtDisplayTime(splitClip, 40, 9.0) == nullptr,
           "Layer Source rejects a referenced clip outside every saved display segment");

    // A transition's Layer Source ends where the transition clip starts (6.0 s).
    std::vector<Segment> abutting {
        { 50, 0, 1.0, 5.0, 1.0, 2.0, -1, -1 },
        { 60, 0, 0.0, 4.0, 1.0, 6.0, -1, -1 },
    };
    check (videowire::resolveClipSegmentAtDisplayTime(abutting, 50, 7.0) == nullptr,
           "strict Layer Source lookup is past the abutting clip end");
    const auto* held = videowire::resolveTransitionSourceSegmentAtDisplayTime(abutting, 50, 7.0, 6.0);
    check (held == &abutting[0]
               && videowire::transitionSourceSeconds(*held, 7.0, 0.25) == 4.75
               && std::fabs (videowire::transitionSourceSeconds(*held, 7.0) - 4.999) < 1.0e-9,
           "transition Layer Source holds the last decodable frame, not the out point");
    check (videowire::transitionSourceSeconds(*held, 7.0, 100.0) == 1.0,
           "held transition source time never precedes the in point");
    check (videowire::resolveTransitionSourceSegmentAtDisplayTime(abutting, 50, 3.0, 6.0) == &abutting[0],
           "transition Layer Source still prefers the exact segment");
    abutting[0].outSec = 3.0;
    check (videowire::resolveTransitionSourceSegmentAtDisplayTime(abutting, 50, 7.0, 6.0) == nullptr,
           "transition Layer Source rejects a clip that ended long before the transition clip");

    // Export seams at 105 bpm and 30 fps, exported from beat 16. The frame
    // time rangeStart + n / fps and the tempo map's clip start land a few
    // ulps apart at beats 58, 65 and 72; the clip that starts on the frame
    // must own it.
    {
        const auto secondsAtBeat = [] (double beat) { return beat * 60.0 / 105.0; };
        const double rangeStart = secondsAtBeat (16.0);
        const auto frameTime = [rangeStart] (int n) { return rangeStart + static_cast<double> (n) / 30.0; };
        const double videoSourceSec = 12.0;
        std::vector<Segment> seam {
            { 2, 0, 0.0, videoSourceSec,
              videoSourceSec / (secondsAtBeat (58.0) - secondsAtBeat (37.0)),
              secondsAtBeat (37.0), -1, -1 },
            { 3, 0, 0.0, secondsAtBeat (65.0) - secondsAtBeat (58.0), 1.0,
              secondsAtBeat (58.0), -1, -1 },
            { 4, 0, 0.0, secondsAtBeat (72.0) - secondsAtBeat (65.0), 1.0,
              secondsAtBeat (65.0), -1, -1 },
            { 5, 0, 0.0, secondsAtBeat (93.0) - secondsAtBeat (72.0), 1.0,
              secondsAtBeat (72.0), -1, -1 },
        };
        int seamFailures = 0;
        const struct { int frame; std::size_t ending; std::size_t starting; } cuts[] {
            { 720, 0, 1 }, { 840, 1, 2 }, { 960, 2, 3 },
        };
        for (const auto& cut : cuts)
        {
            const double at = frameTime (cut.frame);
            const double before = frameTime (cut.frame - 1);
            if (! videowire::segmentCoversDisplayTime (seam[cut.starting], at)
                || videowire::segmentCoversDisplayTime (seam[cut.ending], at)
                || ! videowire::segmentCoversDisplayTime (seam[cut.ending], before)
                || videowire::segmentCoversDisplayTime (seam[cut.starting], before)
                || videowire::snapDisplayTimeToSegmentStart (seam, at)
                       != seam[cut.starting].displayStartSec
                || videowire::snapDisplayTimeToSegmentStart (seam, before) != before)
                ++seamFailures;
        }
        check (seamFailures == 0,
               "the clip that starts on an export frame owns that frame, the ending clip does not");
        check (videowire::resolveClipSegmentAtDisplayTime (seam, 3, frameTime (720)) == &seam[1]
                   && videowire::resolveClipSegmentAtDisplayTime (seam, 2, frameTime (720)) == nullptr,
               "Layer Source lookup uses the same seam rule as the export");
        check (! videowire::displayTimeInSpan (std::nan (""), 0.0, 1.0)
                   && videowire::displayTimeInSpan (0.0, 0.0, 1.0)
                   && ! videowire::displayTimeInSpan (1.0, 0.0, 1.0)
                   && videowire::displayTimeInSpan (0.5, 0.0, 1.0),
               "display span is half-open and rejects NaN");
    }

    std::printf ("source-binding: %d/24 checks passed\n", 24 - failures);
    return failures;
}
