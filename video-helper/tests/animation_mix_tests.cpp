#include "VisualAnimationMix.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <type_traits>

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

bool near (float actual, float expected, float tolerance = 1.0e-5f)
{
    return std::abs(actual - expected) <= tolerance;
}

struct ClipFixture
{
    std::array<double, 2> times { 0.0, 2.0 };
    std::array<float, 6> translation;
    std::array<float, 8> rotation;
    std::array<float, 4> morph;
    std::array<TrackView, 3> tracks;
    ClipView view;

    ClipFixture (ClipId clipId, TrackId trackBase, bool reverseOrder,
                 float translationEnd, float rotationSign, float morphEnd)
        : translation { 0.0f, 0.0f, 0.0f, translationEnd, 0.0f, 0.0f },
          rotation { 0.0f, 0.0f, 0.0f, 1.0f,
                     0.0f, 0.0f, rotationSign, 0.0f },
          morph { 0.0f, 1.0f, morphEnd, 1.0f - morphEnd }
    {
        const TrackView translationTrack {
            trackBase, TargetId { 20 }, Channel::Translation, Interpolation::Linear,
            times.data(), translation.data(), times.size(), translation.size(), 0
        };
        const TrackView rotationTrack {
            TrackId { trackBase.value + 1 }, TargetId { 10 }, Channel::Rotation,
            Interpolation::Linear, times.data(), rotation.data(), times.size(),
            rotation.size(), 0
        };
        const TrackView morphTrack {
            TrackId { trackBase.value + 2 }, TargetId { 30 }, Channel::MorphWeights,
            Interpolation::Linear, times.data(), morph.data(), times.size(), morph.size(), 2
        };
        tracks = reverseOrder
            ? std::array<TrackView, 3> { morphTrack, translationTrack, rotationTrack }
            : std::array<TrackView, 3> { rotationTrack, morphTrack, translationTrack };
        view = { clipId, 2.0, tracks.data(), tracks.size() };
    }
};
} // namespace

int main()
{
    static_assert(std::is_const_v<std::remove_reference_t<decltype(
        std::declval<const CrossfadeSnapshot&>().channels())>>,
        "crossfade channels must be immutable");

    std::string error;
    ClipFixture fromSource(ClipId { 1 }, TrackId { 101 }, false, 4.0f, 1.0f, 1.0f);
    ClipFixture toSource(ClipId { 2 }, TrackId { 201 }, true, 12.0f, -1.0f, 0.0f);
    const auto from = Clip::create(fromSource.view, {}, error);
    const auto to = Clip::create(toSource.view, {}, error);
    check(from && to && error.empty(), "crossfade fixtures are admitted");
    if (! from || ! to)
        return 1;

    const CrossfadeRequest midpoint {
        SampleRequest { 1.0, Playback::Clamp },
        SampleRequest { 1.0, Playback::Clamp },
        0.5
    };
    const auto mixed = crossfade(*from, *to, midpoint, {}, error);
    const auto repeated = crossfade(*from, *to, midpoint, {}, error);
    check(mixed && repeated && error.empty(), "compatible clips crossfade");
    check(mixed && mixed->fromClip() == ClipId { 1 } && mixed->toClip() == ClipId { 2 }
          && mixed->progress() == 0.5 && mixed->fromSampleTimeSeconds() == 1.0
          && mixed->toSampleTimeSeconds() == 1.0,
          "crossfade snapshot retains clip, progress, and sample-time identity");
    check(mixed && mixed->channels().size() == 3
          && mixed->channels()[0].target() == TargetId { 10 }
          && mixed->channels()[1].target() == TargetId { 20 }
          && mixed->channels()[2].target() == TargetId { 30 },
          "crossfade channels use canonical target order independent of clip track order");
    check(mixed && mixed->channels()[1].fromTrack() == TrackId { 101 }
          && mixed->channels()[1].toTrack() == TrackId { 201 }
          && near(mixed->channels()[1].values()[0], 4.0f),
          "translation crossfade retains both source track identities");
    check(mixed && near(mixed->channels()[2].values()[0], 0.25f)
          && near(mixed->channels()[2].values()[1], 0.75f),
          "morph weights crossfade component-wise");
    if (mixed)
    {
        const auto& rotation = mixed->channels()[0].values();
        const auto norm = std::sqrt(rotation[0] * rotation[0] + rotation[1] * rotation[1]
                                  + rotation[2] * rotation[2] + rotation[3] * rotation[3]);
        check(near(norm, 1.0f) && near(rotation[2], 0.0f)
              && near(rotation[3], 1.0f),
              "opposed sampled rotations crossfade through the normalized identity");
    }
    check(mixed && repeated
          && mixed->channels()[0].values() == repeated->channels()[0].values()
          && mixed->channels()[1].values() == repeated->channels()[1].values()
          && mixed->channels()[2].values() == repeated->channels()[2].values(),
          "equal crossfade requests are bit-identical");

    auto endpoints = midpoint;
    endpoints.progress = 0.0;
    const auto atFrom = crossfade(*from, *to, endpoints, {}, error);
    endpoints.progress = 1.0;
    const auto atTo = crossfade(*from, *to, endpoints, {}, error);
    check(atFrom && atTo && near(atFrom->channels()[1].values()[0], 2.0f)
          && near(atTo->channels()[1].values()[0], 6.0f),
          "crossfade endpoints reproduce independently sampled clips");

    auto looped = midpoint;
    looped.from = { 2.5, Playback::Loop };
    looped.to = { -0.5, Playback::Loop };
    const auto loopResult = crossfade(*from, *to, looped, {}, error);
    check(loopResult && loopResult->fromSampleTimeSeconds() == 0.5
          && loopResult->toSampleTimeSeconds() == 1.5,
          "each crossfade input applies its own deterministic playback mapping");

    auto invalidProgress = midpoint;
    invalidProgress.progress = std::numeric_limits<double>::quiet_NaN();
    check(! crossfade(*from, *to, invalidProgress, {}, error)
          && error == "animation crossfade progress must be finite and in [0,1]",
          "nonfinite crossfade progress fails before sampling");

    CrossfadeLimits channelLimit;
    channelLimit.maxChannels = 2;
    check(! crossfade(*from, *to, midpoint, channelLimit, error)
          && error == "animation crossfade channel capacity exceeded",
          "crossfade channel capacity is bounded before sampling");

    CrossfadeLimits scalarLimit;
    scalarLimit.maxScalarValues = 10;
    check(! crossfade(*from, *to, midpoint, scalarLimit, error)
          && error == "animation crossfade scalar capacity exceeded",
          "crossfade scalar capacity includes both sampled clips");

    auto incompatibleTracks = toSource.tracks;
    incompatibleTracks[0].target = TargetId { 99 };
    const ClipView incompatibleView {
        ClipId { 3 }, 2.0, incompatibleTracks.data(), incompatibleTracks.size()
    };
    const auto incompatible = Clip::create(incompatibleView, {}, error);
    check(incompatible != nullptr, "incompatible topology fixture remains a valid clip");
    check(incompatible && ! crossfade(*from, *incompatible, midpoint, {}, error)
          && error == "animation crossfade clips have incompatible channel sets",
          "crossfade rejects clips with different target-channel topology");

    std::fprintf(stderr, failures == 0
        ? "Animation crossfade checks passed\n"
        : "%d animation crossfade checks failed\n", failures);
    return failures == 0 ? 0 : 1;
}
