#include "../../shared/VisualAnimationMix.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <tuple>
#include <utility>

namespace visualanimation
{
namespace
{
struct ChannelKey
{
    TargetId target;
    Channel channel = Channel::Translation;

    bool operator< (const ChannelKey& other) const noexcept
    {
        return std::tie(target.value, channel) < std::tie(other.target.value, other.channel);
    }

    bool operator== (const ChannelKey& other) const noexcept
    {
        return target == other.target && channel == other.channel;
    }
};

struct SampledChannel
{
    ChannelKey key;
    TrackId track;
    const std::vector<float>* values = nullptr;
};

CrossfadeLimits boundedLimits (const CrossfadeLimits& requested) noexcept
{
    const CrossfadeLimits hard;
    return {
        std::min(requested.maxChannels, hard.maxChannels),
        std::min(requested.maxScalarValues, hard.maxScalarValues)
    };
}

bool addWithin (std::size_t value, std::size_t& total, std::size_t limit) noexcept
{
    if (value > limit || total > limit - value)
        return false;
    total += value;
    return true;
}

bool validateClipShape (const Clip& clip, const CrossfadeLimits& limits,
                        std::size_t& scalarValues, std::string& error)
{
    if (clip.tracks().size() > limits.maxChannels)
    {
        error = "animation crossfade channel capacity exceeded";
        return false;
    }
    for (const auto& track : clip.tracks())
        if (! addWithin(track.valueWidth(), scalarValues, limits.maxScalarValues))
        {
            error = "animation crossfade scalar capacity exceeded";
            return false;
        }
    return true;
}

std::vector<SampledChannel> canonicalChannels (const Sample& sample)
{
    std::vector<SampledChannel> result;
    result.reserve(sample.tracks().size());
    for (const auto& track : sample.tracks())
        result.push_back({ { track.target(), track.channel() }, track.track(), &track.values() });
    std::sort(result.begin(), result.end(), [] (const auto& left, const auto& right)
    {
        return left.key < right.key;
    });
    return result;
}

bool mixLinear (const std::vector<float>& from, const std::vector<float>& to,
                double progress, std::vector<float>& output) noexcept
{
    if (from.size() != to.size())
        return false;
    output.resize(from.size());
    for (std::size_t index = 0; index < from.size(); ++index)
    {
        const auto value = static_cast<double>(from[index])
                         + (static_cast<double>(to[index]) - from[index]) * progress;
        if (! std::isfinite(value))
            return false;
        output[index] = static_cast<float>(value);
    }
    return true;
}

bool mixRotation (const std::vector<float>& from, const std::vector<float>& to,
                  double progress, std::vector<float>& output) noexcept
{
    if (from.size() != 4 || to.size() != 4)
        return false;
    double dot = 0.0;
    for (std::size_t component = 0; component < 4; ++component)
        dot += static_cast<double>(from[component]) * to[component];
    const auto hemisphere = dot < 0.0 ? -1.0 : 1.0;
    output.resize(4);
    double normSquared = 0.0;
    for (std::size_t component = 0; component < 4; ++component)
    {
        const auto value = static_cast<double>(from[component])
                         + (hemisphere * static_cast<double>(to[component]) - from[component])
                             * progress;
        if (! std::isfinite(value))
            return false;
        output[component] = static_cast<float>(value);
        normSquared += value * value;
    }
    if (! std::isfinite(normSquared)
        || normSquared <= std::numeric_limits<double>::min())
        return false;
    const auto inverseNorm = 1.0 / std::sqrt(normSquared);
    for (auto& value : output)
        value = static_cast<float>(value * inverseNorm);
    return true;
}
} // namespace

class CrossfadeBuilder
{
public:
    static std::shared_ptr<const CrossfadeSnapshot> build (
        const Clip& fromClip, const Clip& toClip,
        const CrossfadeRequest& request, const CrossfadeLimits& requestedLimits,
        std::string& error)
    {
        error.clear();
        if (! std::isfinite(request.progress)
            || request.progress < 0.0 || request.progress > 1.0)
        {
            error = "animation crossfade progress must be finite and in [0,1]";
            return {};
        }

        const auto limits = boundedLimits(requestedLimits);
        std::size_t scalarValues = 0;
        if (! validateClipShape(fromClip, limits, scalarValues, error)
            || ! validateClipShape(toClip, limits, scalarValues, error))
            return {};

        try
        {
            std::string sampleError;
            const auto fromSample = sample(fromClip, request.from, sampleError);
            if (! fromSample)
            {
                error = sampleError;
                return {};
            }
            const auto toSample = sample(toClip, request.to, sampleError);
            if (! toSample)
            {
                error = sampleError;
                return {};
            }

            const auto fromChannels = canonicalChannels(*fromSample);
            const auto toChannels = canonicalChannels(*toSample);
            if (fromChannels.size() != toChannels.size())
            {
                error = "animation crossfade clips have incompatible channel sets";
                return {};
            }

            std::shared_ptr<CrossfadeSnapshot> result(new CrossfadeSnapshot());
            result->fromClip_ = fromClip.id();
            result->toClip_ = toClip.id();
            result->progress_ = request.progress;
            result->fromSampleTimeSeconds_ = fromSample->sampleTimeSeconds();
            result->toSampleTimeSeconds_ = toSample->sampleTimeSeconds();
            result->channels_.reserve(fromChannels.size());

            for (std::size_t index = 0; index < fromChannels.size(); ++index)
            {
                const auto& from = fromChannels[index];
                const auto& to = toChannels[index];
                if (! (from.key == to.key)
                    || from.values == nullptr || to.values == nullptr
                    || from.values->size() != to.values->size())
                {
                    error = "animation crossfade clips have incompatible channel sets";
                    return {};
                }

                CrossfadedChannel channel;
                channel.target_ = from.key.target;
                channel.channel_ = from.key.channel;
                channel.fromTrack_ = from.track;
                channel.toTrack_ = to.track;
                const auto mixed = channel.channel_ == Channel::Rotation
                    ? mixRotation(*from.values, *to.values, request.progress, channel.values_)
                    : mixLinear(*from.values, *to.values, request.progress, channel.values_);
                if (! mixed)
                {
                    error = "animation crossfade produced an invalid channel value";
                    return {};
                }
                result->channels_.push_back(std::move(channel));
            }
            return result;
        }
        catch (const std::bad_alloc&)
        {
            error = "animation crossfade allocation failed within admitted limits";
            return {};
        }
    }
};

std::shared_ptr<const CrossfadeSnapshot> crossfade (
    const Clip& fromClip, const Clip& toClip,
    const CrossfadeRequest& request, const CrossfadeLimits& limits,
    std::string& error)
{
    return CrossfadeBuilder::build(fromClip, toClip, request, limits, error);
}
} // namespace visualanimation
