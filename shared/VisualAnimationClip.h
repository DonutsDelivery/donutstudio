#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace visualanimation
{
enum class CombinationMode
{
    Replace,
    WeightedBlend,
    Add,
    Multiply,
    AddAfterImportedAnimation
};

template <typename Tag>
struct StableId
{
    std::uint64_t value = 0;

    constexpr bool isValid() const noexcept { return value != 0; }
    constexpr bool operator== (StableId other) const noexcept { return value == other.value; }
    constexpr bool operator!= (StableId other) const noexcept { return value != other.value; }
    constexpr bool operator< (StableId other) const noexcept { return value < other.value; }
};

struct ClipIdTag {};
struct TrackIdTag {};
struct TargetIdTag {};

using ClipId = StableId<ClipIdTag>;
using TrackId = StableId<TrackIdTag>;
using TargetId = StableId<TargetIdTag>;

enum class Channel : std::uint8_t
{
    Translation,
    Rotation,
    Scale,
    MorphWeights
};

enum class Interpolation : std::uint8_t
{
    Step,
    Linear
};

enum class Playback : std::uint8_t
{
    Clamp,
    Loop
};

struct Limits
{
    std::size_t maxTracks = 4096;
    std::size_t maxKeysPerTrack = 65536;
    std::size_t maxTotalKeys = 262144;
    std::size_t maxMorphWeightsPerTrack = 256;
    std::size_t maxTotalValues = 4u * 1024u * 1024u;
};

// Non-owning admission input. The caller retains ownership only until create()
// returns. Values are packed by key with 3 translation/scale components,
// 4 rotation components in x/y/z/w order, or morphWeightCount components.
struct TrackView
{
    TrackId id;
    TargetId target;
    Channel channel = Channel::Translation;
    Interpolation interpolation = Interpolation::Linear;
    const double* keyTimesSeconds = nullptr;
    const float* values = nullptr;
    std::size_t keyCount = 0;
    std::size_t valueCount = 0;
    std::size_t morphWeightCount = 0;
};

struct ClipView
{
    ClipId id;
    double durationSeconds = 0.0;
    const TrackView* tracks = nullptr;
    std::size_t trackCount = 0;
};

class Track
{
public:
    TrackId id() const noexcept { return id_; }
    TargetId target() const noexcept { return target_; }
    Channel channel() const noexcept { return channel_; }
    Interpolation interpolation() const noexcept { return interpolation_; }
    std::size_t keyCount() const noexcept { return keyTimesSeconds_.size(); }
    std::size_t valueWidth() const noexcept { return valueWidth_; }
    const std::vector<double>& keyTimesSeconds() const noexcept { return keyTimesSeconds_; }
    const std::vector<float>& values() const noexcept { return values_; }

private:
    friend class Clip;

    TrackId id_;
    TargetId target_;
    Channel channel_ = Channel::Translation;
    Interpolation interpolation_ = Interpolation::Linear;
    std::size_t valueWidth_ = 0;
    std::vector<double> keyTimesSeconds_;
    std::vector<float> values_;
};

class Clip
{
public:
    static std::shared_ptr<const Clip> create (const ClipView& source,
                                               const Limits& limits,
                                               std::string& error);

    ClipId id() const noexcept { return id_; }
    double durationSeconds() const noexcept { return durationSeconds_; }
    const std::vector<Track>& tracks() const noexcept { return tracks_; }

private:
    ClipId id_;
    double durationSeconds_ = 0.0;
    std::vector<Track> tracks_;
};

struct SampleRequest
{
    double timeSeconds = 0.0;
    Playback playback = Playback::Clamp;
    double rangeStartSeconds = 0.0;
    // Zero retains the historical full-clip range. Nonzero endpoints are
    // admitted against the immutable clip duration by sample().
    double rangeEndSeconds = 0.0;
};

struct SampleBuilder;

class TrackSample
{
public:
    TrackId track() const noexcept { return track_; }
    TargetId target() const noexcept { return target_; }
    Channel channel() const noexcept { return channel_; }
    const std::vector<float>& values() const noexcept { return values_; }

private:
    friend struct SampleBuilder;
    friend class CrossfadeBuilder;

    TrackId track_;
    TargetId target_;
    Channel channel_ = Channel::Translation;
    std::vector<float> values_;
};

class Sample
{
public:
    ClipId clip() const noexcept { return clip_; }
    double requestedTimeSeconds() const noexcept { return requestedTimeSeconds_; }
    double sampleTimeSeconds() const noexcept { return sampleTimeSeconds_; }
    Playback playback() const noexcept { return playback_; }
    const std::vector<TrackSample>& tracks() const noexcept { return tracks_; }

private:
    friend struct SampleBuilder;
    friend class CrossfadeBuilder;

    ClipId clip_;
    double requestedTimeSeconds_ = 0.0;
    double sampleTimeSeconds_ = 0.0;
    Playback playback_ = Playback::Clamp;
    std::vector<TrackSample> tracks_;
};

std::optional<Sample> sample (const Clip& clip,
                              const SampleRequest& request,
                              std::string& error);
} // namespace visualanimation
