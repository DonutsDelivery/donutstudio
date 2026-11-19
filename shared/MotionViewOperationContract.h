#pragma once

#include "PortableFloatingPointChars.h"

#include <cmath>
#include <initializer_list>
#include <string>
#include <string_view>

namespace motionviewoperation
{
inline constexpr std::string_view kNodeKind = "visual.motion.view";
inline constexpr std::string_view kBackendCapability = "native-gpu";
inline constexpr const char* kDisplayRangeParameter = "displayRangePixels";
inline constexpr double kMaximumDisplayRangePixels = 65536.0;

// Optical Flow tracks the first image into the second in top-left image
// coordinates. Render Passes Motion describes current-minus-previous geometry
// in Y-up coordinates. A consumer must use the declared convention.
enum class Convention { OpticalFirstToSecondTopLeft, GeometricCurrentMinusPreviousYUp };

struct Payload final
{
    Convention convention = Convention::OpticalFirstToSecondTopLeft;
    // Zero preserves the legacy x/width, y/height display. Positive ranges use
    // the same pixel denominator for both axes, without changing the field.
    double displayRangePixels = 0.0;
};

inline bool validDisplayRange (double value) noexcept
{
    return std::isfinite (value)
        && (value == 0.0 || (value >= 1.0 && value <= kMaximumDisplayRangePixels));
}

inline bool parseDisplayRange (std::string_view text, double& destination) noexcept
{
    if (text.empty()) return false;
    double candidate = 0.0;
    const auto parsed = portablefloat::fromChars (text.data(), text.data() + text.size(), candidate);
    if (parsed.ec != std::errc {} || parsed.ptr != text.data() + text.size()
        || ! validDisplayRange (candidate)) return false;
    destination = candidate;
    return true;
}

inline bool valid (const Payload& payload) noexcept
{
    return validDisplayRange (payload.displayRangePixels)
        && (payload.convention == Convention::OpticalFirstToSecondTopLeft
            || (payload.convention == Convention::GeometricCurrentMinusPreviousYUp
                && payload.displayRangePixels == 0.0));
}

inline constexpr std::string_view token (Convention value) noexcept
{
    return value == Convention::OpticalFirstToSecondTopLeft
        ? "opticalFirstToSecondTopLeft" : "geometricCurrentMinusPreviousYUp";
}

inline std::string serialize (const Payload& payload)
{
    if (! valid (payload)) return {};
    std::string result = "<MotionView schemaVersion=\"1\" convention=\""
        + std::string (token (payload.convention)) + "\"";
    if (payload.displayRangePixels != 0.0)
    {
        char number[64];
        const auto encoded = portablefloat::toChars (number, number + sizeof (number),
                                                     payload.displayRangePixels);
        if (encoded.ec != std::errc {}) return {};
        result += " displayRangePixels=\"" + std::string (number, encoded.ptr) + "\"";
    }
    return result + "/>";
}

inline std::string serialize (Convention value)
{
    return serialize (Payload { value, 0.0 });
}

inline bool parse (std::string_view wire, Payload& destination)
{
    for (const auto value : { Convention::OpticalFirstToSecondTopLeft,
                              Convention::GeometricCurrentMinusPreviousYUp })
    {
        const auto legacy = serialize (value);
        if (wire == legacy)
        {
            destination = { value, 0.0 };
            return true;
        }
        const auto prefix = legacy.substr (0, legacy.size() - 2) + " displayRangePixels=\"";
        constexpr std::string_view suffix = "\"/>";
        if (wire.size() <= prefix.size() + suffix.size()
            || wire.substr (0, prefix.size()) != prefix
            || wire.substr (wire.size() - suffix.size()) != suffix) continue;
        Payload candidate { value, 0.0 };
        if (! parseDisplayRange (wire.substr (prefix.size(),
                wire.size() - prefix.size() - suffix.size()), candidate.displayRangePixels)
            || ! valid (candidate)) return false;
        destination = candidate;
        return true;
    }
    return false;
}

inline bool parse (std::string_view wire, Convention& destination)
{
    Payload payload;
    if (! parse (wire, payload)) return false;
    destination = payload.convention;
    return true;
}
} // namespace motionviewoperation
