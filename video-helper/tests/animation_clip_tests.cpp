#include "VisualAnimationClip.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

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

TrackView trackView (TrackId id, TargetId target, Channel channel,
                     Interpolation interpolation, const std::vector<double>& times,
                     const std::vector<float>& values, std::size_t morphWeights = 0)
{
    return { id, target, channel, interpolation, times.data(), values.data(),
             times.size(), values.size(), morphWeights };
}

std::shared_ptr<const Clip> createSingle (TrackView track, double duration,
                                          std::string& error, Limits limits = {})
{
    const ClipView source { ClipId { 90 }, duration, &track, 1 };
    return Clip::create(source, limits, error);
}

bool rejected (TrackView track, double duration, std::string* diagnostic = nullptr,
               Limits limits = {})
{
    std::string error;
    const auto clip = createSingle(track, duration, error, limits);
    if (diagnostic != nullptr)
        *diagnostic = error;
    return clip == nullptr && ! error.empty();
}
} // namespace

int main()
{
    const std::vector<double> times { 0.0, 1.0, 2.0 };
    const std::vector<float> translation {
        0.0f, 0.0f, 0.0f,
        2.0f, 4.0f, 6.0f,
        4.0f, 8.0f, 12.0f
    };
    const std::vector<float> scale {
        1.0f, 1.0f, 1.0f,
        2.0f, 2.0f, 2.0f,
        3.0f, 3.0f, 3.0f
    };
    const float halfSqrt = std::sqrt(0.5f);
    const std::vector<float> rotation {
        0.0f, 0.0f, 0.0f, 2.0f,
        0.0f, 0.0f, -halfSqrt, -halfSqrt,
        0.0f, 0.0f, 1.0f, 0.0f
    };
    const std::vector<float> morph {
        0.0f, 1.0f,
        0.5f, 0.5f,
        1.0f, 0.0f
    };
    const std::vector<TrackView> tracks {
        trackView(TrackId { 11 }, TargetId { 101 }, Channel::Translation,
                  Interpolation::Linear, times, translation),
        trackView(TrackId { 12 }, TargetId { 101 }, Channel::Scale,
                  Interpolation::Step, times, scale),
        trackView(TrackId { 13 }, TargetId { 102 }, Channel::Rotation,
                  Interpolation::Linear, times, rotation),
        trackView(TrackId { 14 }, TargetId { 103 }, Channel::MorphWeights,
                  Interpolation::Linear, times, morph, 2)
    };

    std::string error;
    const ClipView source { ClipId { 77 }, 2.0, tracks.data(), tracks.size() };
    const auto clip = Clip::create(source, {}, error);
    check(clip != nullptr && error.empty(), "valid bounded animation clip is admitted");
    check(clip && clip->id() == ClipId { 77 } && clip->tracks().size() == 4,
          "stable clip and track identities survive immutable admission");
    check(clip && clip->tracks()[0].id() == TrackId { 11 }
          && clip->tracks()[0].target() == TargetId { 101 },
          "stable track and target identities remain explicit");
    check(clip && near(clip->tracks()[2].values()[3], 1.0f),
          "rotation keys are normalized once during admission");

    if (clip)
    {
        auto exact = sample(*clip, { 1.0, Playback::Clamp }, error);
        check(exact && exact->sampleTimeSeconds() == 1.0,
              "exact key time remains the explicit sample time");
        check(exact && exact->tracks()[0].values()
                          == std::vector<float>({ 2.0f, 4.0f, 6.0f }),
              "linear translation returns the exact authored key");
        check(exact && exact->tracks()[1].values()
                          == std::vector<float>({ 2.0f, 2.0f, 2.0f }),
              "STEP interpolation changes at the exact key time");
        check(exact && near(exact->tracks()[3].values()[0], 0.5f)
                    && near(exact->tracks()[3].values()[1], 0.5f),
              "morph-weight channels sample every declared weight");

        auto before = sample(*clip, { -4.0, Playback::Clamp }, error);
        auto after = sample(*clip, { 8.0, Playback::Clamp }, error);
        check(before && before->sampleTimeSeconds() == 0.0
                     && before->tracks()[0].values()[0] == 0.0f,
              "clamp sampling holds the first boundary");
        check(after && after->sampleTimeSeconds() == 2.0
                    && after->tracks()[0].values()[0] == 4.0f,
              "clamp sampling holds the final boundary");

        auto looped = sample(*clip, { 2.5, Playback::Loop }, error);
        auto negativeLoop = sample(*clip, { -0.5, Playback::Loop }, error);
        auto exactLoop = sample(*clip, { 2.0, Playback::Loop }, error);
        check(looped && looped->sampleTimeSeconds() == 0.5
                     && near(looped->tracks()[0].values()[0], 1.0f),
              "positive loop time wraps deterministically");
        check(negativeLoop && negativeLoop->sampleTimeSeconds() == 1.5
                            && near(negativeLoop->tracks()[0].values()[0], 3.0f),
              "negative loop time wraps into clip-local time");
        check(exactLoop && exactLoop->sampleTimeSeconds() == 0.0,
              "the loop endpoint maps to the first frame");

        auto quaternionMidpoint = sample(*clip, { 0.5, Playback::Clamp }, error);
        check(quaternionMidpoint
              && quaternionMidpoint->tracks()[2].values()[2] > 0.0f
              && quaternionMidpoint->tracks()[2].values()[3] > 0.0f,
              "quaternion LINEAR interpolation follows the shortest hemisphere");
        if (quaternionMidpoint)
        {
            const auto& q = quaternionMidpoint->tracks()[2].values();
            const auto norm = std::sqrt(q[0] * q[0] + q[1] * q[1]
                                      + q[2] * q[2] + q[3] * q[3]);
            check(near(norm, 1.0f), "quaternion interpolation normalizes its result");
        }

        const auto baseline = sample(*clip, { 0.375, Playback::Loop }, error);
        bool repeatedEqual = baseline.has_value();
        for (int repeat = 0; repeat < 1000 && repeatedEqual; ++repeat)
        {
            const auto repeated = sample(*clip, { 0.375, Playback::Loop }, error);
            repeatedEqual = repeated
                         && repeated->sampleTimeSeconds() == baseline->sampleTimeSeconds()
                         && repeated->tracks().size() == baseline->tracks().size();
            for (std::size_t track = 0; repeatedEqual && track < repeated->tracks().size(); ++track)
                repeatedEqual = repeated->tracks()[track].values()
                             == baseline->tracks()[track].values();
        }
        check(repeatedEqual, "repeated samples at the same explicit time are bit-identical");

        const auto invalidSample = sample(
            *clip, { std::numeric_limits<double>::infinity(), Playback::Clamp }, error);
        check(! invalidSample && error == "animation sample time is nonfinite",
              "nonfinite sample time fails before output allocation");
    }

    {
        const std::vector<double> badTimes { 0.0, 0.75, 0.5 };
        check(rejected(trackView(TrackId { 1 }, TargetId { 1 }, Channel::Translation,
                                 Interpolation::Linear, badTimes, translation), 2.0),
              "unsorted keys are rejected");
    }
    {
        const std::vector<double> duplicateTimes { 0.0, 1.0, 1.0 };
        check(rejected(trackView(TrackId { 1 }, TargetId { 1 }, Channel::Translation,
                                 Interpolation::Linear, duplicateTimes, translation), 2.0),
              "duplicate key times with incompatible values are rejected");
    }
    {
        auto badTimes = times;
        badTimes[1] = std::numeric_limits<double>::quiet_NaN();
        check(rejected(trackView(TrackId { 1 }, TargetId { 1 }, Channel::Translation,
                                 Interpolation::Linear, badTimes, translation), 2.0),
              "nonfinite key times are rejected");
    }
    {
        auto badValues = translation;
        badValues[2] = std::numeric_limits<float>::infinity();
        check(rejected(trackView(TrackId { 1 }, TargetId { 1 }, Channel::Translation,
                                 Interpolation::Linear, times, badValues), 2.0),
              "nonfinite key values are rejected");
    }
    {
        const std::vector<float> zeroQuaternion(12, 0.0f);
        check(rejected(trackView(TrackId { 1 }, TargetId { 1 }, Channel::Rotation,
                                 Interpolation::Linear, times, zeroQuaternion), 2.0),
              "zero quaternions are rejected");
    }
    {
        Limits limits;
        limits.maxKeysPerTrack = 2;
        check(rejected(trackView(TrackId { 1 }, TargetId { 1 }, Channel::Translation,
                                 Interpolation::Linear, times, translation), 2.0, nullptr, limits),
              "key capacity is checked before clip allocation");
    }
    {
        Limits limits;
        limits.maxMorphWeightsPerTrack = 1;
        check(rejected(trackView(TrackId { 1 }, TargetId { 1 }, Channel::MorphWeights,
                                 Interpolation::Linear, times, morph, 2), 2.0, nullptr, limits),
              "morph-weight capacity is checked before clip allocation");
    }
    {
        Limits limits;
        limits.maxTracks = 1;
        const ClipView excessive { ClipId { 1 }, 1.0, nullptr, 2 };
        const auto rejectedClip = Clip::create(excessive, limits, error);
        check(! rejectedClip && error == "animation track capacity exceeded",
              "track capacity is checked before track storage is accessed or allocated");
    }
    {
        const auto first = trackView(TrackId { 1 }, TargetId { 9 }, Channel::Translation,
                                     Interpolation::Linear, times, translation);
        const auto second = trackView(TrackId { 2 }, TargetId { 9 }, Channel::Translation,
                                      Interpolation::Step, times, translation);
        const TrackView duplicates[] { first, second };
        const ClipView duplicateTargets { ClipId { 2 }, 2.0, duplicates, 2 };
        const auto rejectedClip = Clip::create(duplicateTargets, {}, error);
        check(! rejectedClip && error == "animation clip has duplicate target-channel tracks",
              "duplicate target-channel tracks are rejected as incompatible");
    }

    std::fprintf(stderr, failures == 0
        ? "Animation clip sampling checks passed\n"
        : "%d animation clip sampling checks failed\n", failures);
    return failures == 0 ? 0 : 1;
}
