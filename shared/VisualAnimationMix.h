#pragma once

#include "VisualAnimationClip.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace visualanimation
{
struct CrossfadeLimits
{
    std::size_t maxChannels = 4096;
    std::size_t maxScalarValues = 1024u * 1024u;
};

struct CrossfadeRequest
{
    SampleRequest from;
    SampleRequest to;
    double progress = 0.0;
};

class CrossfadedChannel
{
public:
    TargetId target() const noexcept { return target_; }
    Channel channel() const noexcept { return channel_; }
    TrackId fromTrack() const noexcept { return fromTrack_; }
    TrackId toTrack() const noexcept { return toTrack_; }
    const std::vector<float>& values() const noexcept { return values_; }

private:
    friend class CrossfadeBuilder;

    TargetId target_;
    Channel channel_ = Channel::Translation;
    TrackId fromTrack_;
    TrackId toTrack_;
    std::vector<float> values_;
};

class CrossfadeSnapshot
{
public:
    ClipId fromClip() const noexcept { return fromClip_; }
    ClipId toClip() const noexcept { return toClip_; }
    double progress() const noexcept { return progress_; }
    double fromSampleTimeSeconds() const noexcept { return fromSampleTimeSeconds_; }
    double toSampleTimeSeconds() const noexcept { return toSampleTimeSeconds_; }
    const std::vector<CrossfadedChannel>& channels() const noexcept { return channels_; }

private:
    friend class CrossfadeBuilder;

    ClipId fromClip_;
    ClipId toClip_;
    double progress_ = 0.0;
    double fromSampleTimeSeconds_ = 0.0;
    double toSampleTimeSeconds_ = 0.0;
    std::vector<CrossfadedChannel> channels_;
};

std::shared_ptr<const CrossfadeSnapshot> crossfade (
    const Clip& fromClip,
    const Clip& toClip,
    const CrossfadeRequest& request,
    const CrossfadeLimits& limits,
    std::string& error);

// Retains the first clip's track identities and timing for existing consumers.
std::shared_ptr<const Sample> crossfadeSample (
    const Clip& fromClip, const Clip& toClip,
    const CrossfadeRequest& request, const CrossfadeLimits& limits,
    std::string& error);
} // namespace visualanimation
