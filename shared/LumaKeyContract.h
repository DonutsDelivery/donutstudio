#pragma once

#include <cmath>
#include <cstdint>
#include <string_view>

namespace videowire::lumakey
{
// Bounds and parameter order of the native LumaKey pass
// (video-helper/src/effect_defs.h, EffectType::LumaKey). Shared by the plugin
// lowering and the helper executor so both sides refuse the same values.
inline constexpr std::uint32_t kSchemaVersion = 1;
inline constexpr double kDefaultLow = 0.0;
inline constexpr double kDefaultHigh = 1.0;
inline constexpr double kDefaultSoftness = 0.1;
inline constexpr double kDefaultInvert = 0.0;

struct Description final
{
    std::uint32_t schemaVersion = kSchemaVersion;
    double low = kDefaultLow;
    double high = kDefaultHigh;
    double softness = kDefaultSoftness;
    double invert = kDefaultInvert;
};

enum class AdmissionFailure : std::uint8_t
{
    none = 0,
    unsupportedSchema,
    nonFiniteParameter,
    parameterOutsideBounds,
    invalidRange
};

inline constexpr std::string_view token(AdmissionFailure value) noexcept
{
    switch (value)
    {
        case AdmissionFailure::none: return "none";
        case AdmissionFailure::unsupportedSchema: return "unsupported-schema";
        case AdmissionFailure::nonFiniteParameter: return "non-finite-parameter";
        case AdmissionFailure::parameterOutsideBounds: return "parameter-outside-bounds";
        case AdmissionFailure::invalidRange: return "invalid-range";
    }
    return "unknown";
}

// Refuses rather than clamps. Every value must already sit inside the pass
// bounds, low must not exceed high, and invert is a toggle (exactly 0 or 1).
inline bool admit(const Description& source, AdmissionFailure& failure) noexcept
{
    failure = AdmissionFailure::none;
    if (source.schemaVersion != kSchemaVersion)
    {
        failure = AdmissionFailure::unsupportedSchema;
        return false;
    }

    const double values[] { source.low, source.high, source.softness, source.invert };
    for (const double value : values)
        if (!std::isfinite(value))
        {
            failure = AdmissionFailure::nonFiniteParameter;
            return false;
        }

    const auto unit = [](double value) { return value >= 0.0 && value <= 1.0; };
    if (!unit(source.low) || !unit(source.high) || !unit(source.softness)
        || (source.invert != 0.0 && source.invert != 1.0))
    {
        failure = AdmissionFailure::parameterOutsideBounds;
        return false;
    }

    if (source.low > source.high)
    {
        failure = AdmissionFailure::invalidRange;
        return false;
    }
    return true;
}
} // namespace videowire::lumakey
