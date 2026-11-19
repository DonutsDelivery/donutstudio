#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <string_view>

namespace visualtemporalsampling
{

inline constexpr std::string_view kTemporalAntiAliasingKind { "visual.temporal-aa" };
inline constexpr std::string_view kMotionBlurKind { "visual.motion-blur" };
// Samples run to the float-exact integer range of the node parameter; the
// shaders loop to the requested count.
inline constexpr std::uint32_t kMaximumSamples = 1u << 24;
inline constexpr std::uint32_t kTemporalAntiAliasingHistoryImages = 2;

enum class Mode : std::uint8_t
{
    temporalAntiAliasing,
    motionBlur
};

struct Payload
{
    Mode mode = Mode::temporalAntiAliasing;
    std::uint32_t sampleCount = 8;
    float historyWeight = 0.9f;
    float jitterSpread = 1.0f;
    float shutterAngleDegrees = 0.0f;
    float motionScale = 0.0f;
};

inline bool operator== (const Payload& left, const Payload& right) noexcept
{
    return left.mode == right.mode
        && left.sampleCount == right.sampleCount
        && left.historyWeight == right.historyWeight
        && left.jitterSpread == right.jitterSpread
        && left.shutterAngleDegrees == right.shutterAngleDegrees
        && left.motionScale == right.motionScale;
}

inline bool operator!= (const Payload& left, const Payload& right) noexcept
{
    return ! (left == right);
}

struct Requirements
{
    bool color = true;
    bool motionVectors = true;
    std::uint32_t retainedHistoryImages = 0;
    std::uint32_t transientImages = 1;
};

struct ResourceLimits
{
    std::uint32_t maximumImageDimension = 8192;
    std::uint64_t maximumHistoryBytes = 512ull * 1024ull * 1024ull;
    std::uint64_t maximumTransientBytes = 256ull * 1024ull * 1024ull;
};

struct ResourceFootprint
{
    std::uint64_t bytesPerColorImage = 0;
    std::uint64_t bytesPerMotionImage = 0;
    std::uint64_t historyBytes = 0;
    std::uint64_t transientBytes = 0;
};

struct SamplePoint
{
    float jitterX = 0.0f;
    float jitterY = 0.0f;
    float timeOffsetFrames = 0.0f;
};

enum class EvaluationMode : std::uint8_t
{
    preview,
    exportFrame
};

enum class TransitionAction : std::uint8_t
{
    resetAndAdvance,
    advance,
    hold,
    reject
};

enum class ResetCause : std::uint8_t
{
    none,
    firstEvaluation,
    structuralRevision,
    backwardSeek,
    imageExtent,
    helperRestart
};

enum class RejectionCause : std::uint8_t
{
    none,
    invalidOwner,
    invalidTime,
    invalidExtent,
    mixedEvaluationMode
};

struct EvaluationPoint
{
    std::int64_t ownerId = -1;
    std::uint64_t structuralRevision = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint64_t helperGeneration = 0;
    double timeSec = 0.0;
    EvaluationMode mode = EvaluationMode::preview;
    bool paused = false;
};

struct Transition
{
    TransitionAction action = TransitionAction::reject;
    ResetCause resetCause = ResetCause::none;
    RejectionCause rejectionCause = RejectionCause::none;
};

class LifecycleState final
{
public:
    Transition evaluate (const EvaluationPoint& point) noexcept
    {
        if (point.ownerId < 0)
            return reject (RejectionCause::invalidOwner);
        if (! std::isfinite (point.timeSec))
            return reject (RejectionCause::invalidTime);
        if (point.width == 0 || point.height == 0)
            return reject (RejectionCause::invalidExtent);
        if (! initialized_)
            return resetAndAdvance (point, ResetCause::firstEvaluation);
        if (point.ownerId != ownerId_)
            return reject (RejectionCause::invalidOwner);
        if (point.mode != mode_)
            return reject (RejectionCause::mixedEvaluationMode);
        if (point.helperGeneration != helperGeneration_)
            return resetAndAdvance (point, ResetCause::helperRestart);
        if (point.structuralRevision != structuralRevision_)
            return resetAndAdvance (point, ResetCause::structuralRevision);
        if (point.width != width_ || point.height != height_)
            return resetAndAdvance (point, ResetCause::imageExtent);
        if (point.timeSec < timeSec_)
            return resetAndAdvance (point, ResetCause::backwardSeek);
        if (point.paused || point.timeSec == timeSec_)
            return { TransitionAction::hold, ResetCause::none, RejectionCause::none };
        remember (point);
        return { TransitionAction::advance, ResetCause::none, RejectionCause::none };
    }

    void reset() noexcept { initialized_ = false; }
    bool initialized() const noexcept { return initialized_; }

private:
    static Transition reject (RejectionCause cause) noexcept
    {
        return { TransitionAction::reject, ResetCause::none, cause };
    }

    Transition resetAndAdvance (const EvaluationPoint& point, ResetCause cause) noexcept
    {
        remember (point);
        return { TransitionAction::resetAndAdvance, cause, RejectionCause::none };
    }

    void remember (const EvaluationPoint& point) noexcept
    {
        initialized_ = true;
        ownerId_ = point.ownerId;
        structuralRevision_ = point.structuralRevision;
        width_ = point.width;
        height_ = point.height;
        helperGeneration_ = point.helperGeneration;
        timeSec_ = point.timeSec;
        mode_ = point.mode;
    }

    bool initialized_ = false;
    std::int64_t ownerId_ = -1;
    std::uint64_t structuralRevision_ = 0;
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    std::uint64_t helperGeneration_ = 0;
    double timeSec_ = 0.0;
    EvaluationMode mode_ = EvaluationMode::preview;
};

inline Requirements requirements (Mode mode) noexcept
{
    Requirements value;
    value.retainedHistoryImages = mode == Mode::temporalAntiAliasing
        ? kTemporalAntiAliasingHistoryImages : 0u;
    return value;
}

inline std::string_view modeName (Mode mode) noexcept
{
    switch (mode)
    {
        case Mode::temporalAntiAliasing: return "temporal-aa";
        case Mode::motionBlur: return "motion-blur";
    }
    return {};
}

inline bool modeForKind (std::string_view kind, Mode& mode) noexcept
{
    if (kind == kTemporalAntiAliasingKind) mode = Mode::temporalAntiAliasing;
    else if (kind == kMotionBlurKind) mode = Mode::motionBlur;
    else return false;
    return true;
}

inline bool validate (const Payload& payload, std::string* error = nullptr)
{
    const auto fail = [&] (const char* reason)
    {
        if (error != nullptr) *error = reason;
        return false;
    };
    if (payload.sampleCount == 0 || payload.sampleCount > kMaximumSamples)
        return fail ("temporal sampling count is out of bounds");
    if (! std::isfinite (payload.historyWeight) || ! std::isfinite (payload.jitterSpread)
        || ! std::isfinite (payload.shutterAngleDegrees) || ! std::isfinite (payload.motionScale))
        return fail ("temporal sampling contains a non-finite parameter");

    switch (payload.mode)
    {
        case Mode::temporalAntiAliasing:
            if (payload.historyWeight < 0.0f || payload.historyWeight > 0.99f
                || payload.jitterSpread < 0.0f || payload.jitterSpread > 1.0f
                || payload.shutterAngleDegrees != 0.0f || payload.motionScale != 0.0f)
                return fail ("temporal anti-aliasing parameters are out of bounds");
            break;
        case Mode::motionBlur:
            if (payload.historyWeight != 0.0f || payload.jitterSpread != 0.0f
                || payload.shutterAngleDegrees < 0.0f || payload.shutterAngleDegrees > 360.0f
                || payload.motionScale < 0.0f || payload.motionScale > 4.0f)
                return fail ("motion blur parameters are out of bounds");
            break;
    }
    if (error != nullptr) error->clear();
    return true;
}

inline std::string serialize (const Payload& payload)
{
    if (! validate (payload)) return {};
    std::ostringstream stream;
    stream.imbue (std::locale::classic());
    stream << std::setprecision (std::numeric_limits<float>::max_digits10)
           << "temporal-sampling-v1|mode=" << modeName (payload.mode)
           << "|samples=" << payload.sampleCount
           << "|history-weight=" << payload.historyWeight
           << "|jitter=" << payload.jitterSpread
           << "|shutter-degrees=" << payload.shutterAngleDegrees
           << "|motion-scale=" << payload.motionScale;
    return stream.str();
}

namespace detail
{
inline bool parseUnsigned (std::string_view text, std::uint32_t& value) noexcept
{
    if (text.empty()) return false;
    std::uint64_t parsed = 0;
    for (const char character : text)
    {
        if (character < '0' || character > '9') return false;
        parsed = parsed * 10u + static_cast<std::uint32_t> (character - '0');
        if (parsed > std::numeric_limits<std::uint32_t>::max()) return false;
    }
    value = static_cast<std::uint32_t> (parsed);
    return true;
}

inline bool parseFloat (std::string_view text, float& value) noexcept
{
    if (text.empty() || text.size() >= 64) return false;
    std::array<char, 64> copy {};
    for (std::size_t index = 0; index < text.size(); ++index) copy[index] = text[index];
    char* end = nullptr;
    const auto parsed = std::strtof (copy.data(), &end);
    if (end != copy.data() + static_cast<std::ptrdiff_t> (text.size())
        || ! std::isfinite (parsed))
        return false;
    value = parsed;
    return true;
}

inline bool checkedMultiply (std::uint64_t left, std::uint64_t right,
                             std::uint64_t& result) noexcept
{
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left)
        return false;
    result = left * right;
    return true;
}

inline float radicalInverse (std::uint64_t index, std::uint32_t base) noexcept
{
    float result = 0.0f;
    float fraction = 1.0f / static_cast<float> (base);
    while (index != 0)
    {
        result += static_cast<float> (index % base) * fraction;
        index /= base;
        fraction /= static_cast<float> (base);
    }
    return result;
}
} // namespace detail

inline bool parse (std::string_view wire, Payload& payload, std::string* error = nullptr)
{
    const auto fail = [&] (const char* reason)
    {
        if (error != nullptr) *error = reason;
        return false;
    };
    std::array<std::string_view, 7> fields {};
    std::size_t count = 0;
    std::size_t begin = 0;
    while (begin <= wire.size() && count < fields.size())
    {
        const auto separator = wire.find ('|', begin);
        fields[count++] = wire.substr (begin, separator == std::string_view::npos
            ? wire.size() - begin : separator - begin);
        if (separator == std::string_view::npos) break;
        begin = separator + 1;
    }
    if (count != fields.size() || fields[0] != "temporal-sampling-v1"
        || std::count (wire.begin(), wire.end(), '|') != 6)
        return fail ("temporal sampling payload schema is malformed");

    const auto value = [&] (std::size_t index, std::string_view prefix,
                            std::string_view& result)
    {
        if (fields[index].substr (0, prefix.size()) != prefix) return false;
        result = fields[index].substr (prefix.size());
        return ! result.empty();
    };
    std::string_view mode, samples, historyWeight, jitter, shutter, motionScale;
    if (! value (1, "mode=", mode) || ! value (2, "samples=", samples)
        || ! value (3, "history-weight=", historyWeight)
        || ! value (4, "jitter=", jitter)
        || ! value (5, "shutter-degrees=", shutter)
        || ! value (6, "motion-scale=", motionScale))
        return fail ("temporal sampling payload schema is malformed");

    if (mode == "temporal-aa") payload.mode = Mode::temporalAntiAliasing;
    else if (mode == "motion-blur") payload.mode = Mode::motionBlur;
    else return fail ("temporal sampling mode is unsupported");
    if (! detail::parseUnsigned (samples, payload.sampleCount)
        || ! detail::parseFloat (historyWeight, payload.historyWeight)
        || ! detail::parseFloat (jitter, payload.jitterSpread)
        || ! detail::parseFloat (shutter, payload.shutterAngleDegrees)
        || ! detail::parseFloat (motionScale, payload.motionScale))
        return fail ("temporal sampling payload values are malformed");
    return validate (payload, error);
}

inline bool parseForKind (std::string_view kind, std::string_view wire, Payload& payload,
                          std::string* error = nullptr)
{
    Mode expected {};
    if (! modeForKind (kind, expected) || ! parse (wire, payload, error)) return false;
    if (payload.mode != expected)
    {
        if (error != nullptr) *error = "temporal sampling kind and payload mode disagree";
        return false;
    }
    return true;
}

inline SamplePoint samplePoint (const Payload& payload, std::int64_t frameIndex,
                                std::uint32_t sampleIndex) noexcept
{
    if (! validate (payload) || sampleIndex >= payload.sampleCount) return {};
    SamplePoint result;
    if (payload.mode == Mode::temporalAntiAliasing)
    {
        const auto nonnegativeFrame = frameIndex < 0
            ? static_cast<std::uint64_t> (-(frameIndex + 1)) + 1u
            : static_cast<std::uint64_t> (frameIndex);
        const auto sequence = nonnegativeFrame * payload.sampleCount + sampleIndex + 1u;
        result.jitterX = (detail::radicalInverse (sequence, 2) - 0.5f) * payload.jitterSpread;
        result.jitterY = (detail::radicalInverse (sequence, 3) - 0.5f) * payload.jitterSpread;
    }
    else
    {
        const float normalized = (static_cast<float> (sampleIndex) + 0.5f)
                               / static_cast<float> (payload.sampleCount) - 0.5f;
        result.timeOffsetFrames = normalized * (payload.shutterAngleDegrees / 360.0f);
    }
    return result;
}

inline bool admitResources (const Payload& payload, std::uint32_t width, std::uint32_t height,
                            const ResourceLimits& limits, ResourceFootprint& footprint,
                            std::string& error)
{
    footprint = {};
    error.clear();
    if (! validate (payload, &error)) return false;
    if (width == 0 || height == 0)
    {
        error = "temporal sampling requires a positive explicit image extent";
        return false;
    }
    if (limits.maximumImageDimension == 0 || width > limits.maximumImageDimension
        || height > limits.maximumImageDimension)
    {
        error = "temporal sampling image dimension capacity exceeded";
        return false;
    }

    std::uint64_t pixels = 0;
    if (! detail::checkedMultiply (width, height, pixels)
        || ! detail::checkedMultiply (pixels, 8u, footprint.bytesPerColorImage)
        || ! detail::checkedMultiply (pixels, 4u, footprint.bytesPerMotionImage))
    {
        footprint = {};
        error = "temporal sampling byte accounting overflow";
        return false;
    }
    const auto required = requirements (payload.mode);
    if (! detail::checkedMultiply (footprint.bytesPerColorImage,
                                   required.retainedHistoryImages, footprint.historyBytes)
        || ! detail::checkedMultiply (footprint.bytesPerColorImage,
                                      required.transientImages, footprint.transientBytes))
    {
        footprint = {};
        error = "temporal sampling byte accounting overflow";
        return false;
    }
    if (footprint.historyBytes > limits.maximumHistoryBytes)
    {
        footprint = {};
        error = "temporal sampling history byte capacity exceeded";
        return false;
    }
    if (footprint.transientBytes > limits.maximumTransientBytes)
    {
        footprint = {};
        error = "temporal sampling transient byte capacity exceeded";
        return false;
    }
    return true;
}

} // namespace visualtemporalsampling
