#include "../../shared/VisualAnimationClip.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>

namespace visualanimation
{
namespace
{
bool addWithin (std::size_t value, std::size_t& total, std::size_t limit) noexcept
{
    if (value > limit || total > limit - value)
        return false;
    total += value;
    return true;
}

std::size_t widthFor (const TrackView& track, std::string& error)
{
    switch (track.channel)
    {
        case Channel::Translation:
        case Channel::Scale:
            if (track.morphWeightCount != 0)
                error = "non-morph animation track declares morph weights";
            return error.empty() ? 3 : 0;
        case Channel::Rotation:
            if (track.morphWeightCount != 0)
                error = "rotation animation track declares morph weights";
            return error.empty() ? 4 : 0;
        case Channel::MorphWeights:
            if (track.morphWeightCount == 0)
                error = "morph animation track has no weights";
            return track.morphWeightCount;
    }
    error = "animation track has an unknown channel";
    return 0;
}

bool interpolationIsValid (Interpolation interpolation) noexcept
{
    return interpolation == Interpolation::Step || interpolation == Interpolation::Linear;
}

bool playbackIsValid (Playback playback) noexcept
{
    return playback == Playback::Clamp || playback == Playback::Loop;
}

bool duplicateValuesMatch (const TrackView& track, std::size_t first,
                           std::size_t second, std::size_t width) noexcept
{
    const auto firstOffset = first * width;
    const auto secondOffset = second * width;
    for (std::size_t component = 0; component < width; ++component)
        if (track.values[firstOffset + component] != track.values[secondOffset + component])
            return false;
    return true;
}

bool validateTrack (const TrackView& track, double durationSeconds,
                    const Limits& limits, std::size_t& totalKeys,
                    std::size_t& totalValues, std::string& error)
{
    if (! track.id.isValid() || ! track.target.isValid())
    {
        error = "animation track has an invalid stable identity";
        return false;
    }
    if (! interpolationIsValid(track.interpolation))
    {
        error = "animation track has an unknown interpolation";
        return false;
    }

    const auto width = widthFor(track, error);
    if (! error.empty())
        return false;
    if (track.channel == Channel::MorphWeights
        && width > limits.maxMorphWeightsPerTrack)
    {
        error = "animation morph weight capacity exceeded";
        return false;
    }
    if (track.keyCount == 0 || track.keyCount > limits.maxKeysPerTrack)
    {
        error = "animation key capacity exceeded";
        return false;
    }
    if (track.keyTimesSeconds == nullptr || track.values == nullptr)
    {
        error = "animation track has missing key storage";
        return false;
    }
    if (width > std::numeric_limits<std::size_t>::max() / track.keyCount)
    {
        error = "animation value count overflow";
        return false;
    }
    const auto expectedValues = width * track.keyCount;
    if (track.valueCount != expectedValues)
    {
        error = "animation track value count does not match its channel";
        return false;
    }
    if (! addWithin(track.keyCount, totalKeys, limits.maxTotalKeys)
        || ! addWithin(expectedValues, totalValues, limits.maxTotalValues))
    {
        error = "animation clip key or value capacity exceeded";
        return false;
    }

    for (std::size_t key = 0; key < track.keyCount; ++key)
    {
        const auto time = track.keyTimesSeconds[key];
        if (! std::isfinite(time) || time < 0.0 || time > durationSeconds)
        {
            error = "animation key time is nonfinite or outside the clip";
            return false;
        }
        if (key > 0 && time < track.keyTimesSeconds[key - 1])
        {
            error = "animation key times are unsorted";
            return false;
        }
        if (key > 0 && time == track.keyTimesSeconds[key - 1]
            && ! duplicateValuesMatch(track, key - 1, key, width))
        {
            error = "duplicate animation key times have incompatible values";
            return false;
        }

        double quaternionNormSquared = 0.0;
        for (std::size_t component = 0; component < width; ++component)
        {
            const auto value = track.values[key * width + component];
            if (! std::isfinite(value))
            {
                error = "animation key value is nonfinite";
                return false;
            }
            if (track.channel == Channel::Rotation)
                quaternionNormSquared += static_cast<double>(value) * value;
        }
        if (track.channel == Channel::Rotation
            && (! std::isfinite(quaternionNormSquared)
                || quaternionNormSquared <= std::numeric_limits<double>::min()))
        {
            error = "animation rotation key has a zero or invalid quaternion";
            return false;
        }
    }
    return true;
}

std::vector<float> sampleTrack (const Track& track, double timeSeconds)
{
    const auto& times = track.keyTimesSeconds();
    const auto& values = track.values();
    const auto width = track.valueWidth();
    std::vector<float> result(width);

    auto copyKey = [&] (std::size_t key)
    {
        std::copy_n(values.begin() + static_cast<std::ptrdiff_t>(key * width),
                    static_cast<std::ptrdiff_t>(width), result.begin());
    };

    if (times.size() == 1 || timeSeconds <= times.front())
    {
        copyKey(0);
        return result;
    }
    if (timeSeconds >= times.back())
    {
        copyKey(times.size() - 1);
        return result;
    }

    const auto upper = std::upper_bound(times.begin(), times.end(), timeSeconds);
    const auto right = static_cast<std::size_t>(upper - times.begin());
    const auto left = right - 1;
    if (track.interpolation() == Interpolation::Step)
    {
        copyKey(left);
        return result;
    }

    const auto fraction = (timeSeconds - times[left]) / (times[right] - times[left]);
    const auto leftOffset = left * width;
    const auto rightOffset = right * width;
    if (track.channel() != Channel::Rotation)
    {
        for (std::size_t component = 0; component < width; ++component)
        {
            const auto a = static_cast<double>(values[leftOffset + component]);
            const auto b = static_cast<double>(values[rightOffset + component]);
            result[component] = static_cast<float>(a + (b - a) * fraction);
        }
        return result;
    }

    double dot = 0.0;
    for (std::size_t component = 0; component < 4; ++component)
        dot += static_cast<double>(values[leftOffset + component])
             * static_cast<double>(values[rightOffset + component]);
    const auto direction = dot < 0.0 ? -1.0 : 1.0;
    double normSquared = 0.0;
    for (std::size_t component = 0; component < 4; ++component)
    {
        const auto a = static_cast<double>(values[leftOffset + component]);
        const auto b = direction * static_cast<double>(values[rightOffset + component]);
        const auto value = a + (b - a) * fraction;
        result[component] = static_cast<float>(value);
        normSquared += value * value;
    }
    const auto inverseNorm = 1.0 / std::sqrt(normSquared);
    for (auto& value : result)
        value = static_cast<float>(value * inverseNorm);
    return result;
}
} // namespace

struct SampleBuilder
{
    static Sample build (const Clip& clip, const SampleRequest& request, double sampleTime)
    {
        Sample result;
        result.clip_ = clip.id();
        result.requestedTimeSeconds_ = request.timeSeconds;
        result.sampleTimeSeconds_ = sampleTime;
        result.playback_ = request.playback;
        result.tracks_.reserve(clip.tracks().size());
        for (const auto& track : clip.tracks())
        {
            TrackSample trackSample;
            trackSample.track_ = track.id();
            trackSample.target_ = track.target();
            trackSample.channel_ = track.channel();
            trackSample.values_ = sampleTrack(track, sampleTime);
            result.tracks_.push_back(std::move(trackSample));
        }
        return result;
    }
};

std::shared_ptr<const Clip> Clip::create (const ClipView& source,
                                          const Limits& limits,
                                          std::string& error)
{
    error.clear();
    if (! source.id.isValid())
    {
        error = "animation clip has an invalid stable identity";
        return {};
    }
    if (! std::isfinite(source.durationSeconds) || source.durationSeconds < 0.0)
    {
        error = "animation clip duration is nonfinite or negative";
        return {};
    }
    if (source.trackCount == 0 || source.trackCount > limits.maxTracks)
    {
        error = "animation track capacity exceeded";
        return {};
    }
    if (source.tracks == nullptr)
    {
        error = "animation clip has missing track storage";
        return {};
    }

    std::size_t totalKeys = 0;
    std::size_t totalValues = 0;
    for (std::size_t index = 0; index < source.trackCount; ++index)
    {
        const auto& track = source.tracks[index];
        for (std::size_t previous = 0; previous < index; ++previous)
        {
            if (track.id == source.tracks[previous].id)
            {
                error = "animation clip has a duplicate track identity";
                return {};
            }
            if (track.target == source.tracks[previous].target
                && track.channel == source.tracks[previous].channel)
            {
                error = "animation clip has duplicate target-channel tracks";
                return {};
            }
        }
        if (! validateTrack(track, source.durationSeconds, limits,
                            totalKeys, totalValues, error))
            return {};
    }

    try
    {
        std::shared_ptr<Clip> clip(new Clip());
        clip->id_ = source.id;
        clip->durationSeconds_ = source.durationSeconds;
        clip->tracks_.reserve(source.trackCount);
        for (std::size_t index = 0; index < source.trackCount; ++index)
        {
            const auto& sourceTrack = source.tracks[index];
            Track track;
            track.id_ = sourceTrack.id;
            track.target_ = sourceTrack.target;
            track.channel_ = sourceTrack.channel;
            track.interpolation_ = sourceTrack.interpolation;
            track.valueWidth_ = widthFor(sourceTrack, error);
            track.keyTimesSeconds_.assign(sourceTrack.keyTimesSeconds,
                                          sourceTrack.keyTimesSeconds + sourceTrack.keyCount);
            track.values_.assign(sourceTrack.values,
                                 sourceTrack.values + sourceTrack.valueCount);
            if (sourceTrack.channel == Channel::Rotation)
                for (std::size_t key = 0; key < sourceTrack.keyCount; ++key)
                {
                    const auto offset = key * 4;
                    double normSquared = 0.0;
                    for (std::size_t component = 0; component < 4; ++component)
                    {
                        const auto value = track.values_[offset + component];
                        normSquared += static_cast<double>(value) * value;
                    }
                    const auto inverseNorm = 1.0 / std::sqrt(normSquared);
                    for (std::size_t component = 0; component < 4; ++component)
                        track.values_[offset + component]
                            = static_cast<float>(track.values_[offset + component] * inverseNorm);
                }
            clip->tracks_.push_back(std::move(track));
        }
        return clip;
    }
    catch (const std::bad_alloc&)
    {
        error = "animation clip allocation failed within admitted limits";
        return {};
    }
}

std::optional<Sample> sample (const Clip& clip,
                              const SampleRequest& request,
                              std::string& error)
{
    error.clear();
    if (! std::isfinite(request.timeSeconds))
    {
        error = "animation sample time is nonfinite";
        return std::nullopt;
    }
    if (! playbackIsValid(request.playback))
    {
        error = "animation sample has an unknown playback mode";
        return std::nullopt;
    }

    const auto rangeEnd = request.rangeEndSeconds == 0.0
        ? clip.durationSeconds() : request.rangeEndSeconds;
    const bool zeroDurationDefault = clip.durationSeconds() == 0.0
        && request.rangeStartSeconds == 0.0 && request.rangeEndSeconds == 0.0;
    if (! std::isfinite(request.rangeStartSeconds) || request.rangeStartSeconds < 0.0
        || ! std::isfinite(rangeEnd) || rangeEnd <= request.rangeStartSeconds
        || rangeEnd > clip.durationSeconds())
    {
        if (! zeroDurationDefault)
        {
            error = "animation sample range is invalid or outside the clip";
            return std::nullopt;
        }
    }

    double sampleTime = 0.0;
    if (request.playback == Playback::Clamp)
        sampleTime = std::clamp(request.timeSeconds, request.rangeStartSeconds, rangeEnd);
    else if (! zeroDurationDefault)
    {
        const auto rangeDuration = rangeEnd - request.rangeStartSeconds;
        sampleTime = std::fmod(request.timeSeconds - request.rangeStartSeconds,
                               rangeDuration);
        if (sampleTime < 0.0)
            sampleTime += rangeDuration;
        sampleTime += request.rangeStartSeconds;
        if (sampleTime == request.rangeStartSeconds)
            sampleTime = request.rangeStartSeconds;
    }

    try
    {
        return SampleBuilder::build(clip, request, sampleTime);
    }
    catch (const std::bad_alloc&)
    {
        error = "animation sample allocation failed";
        return std::nullopt;
    }
}
} // namespace visualanimation
