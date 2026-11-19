#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <string_view>

namespace visualtemporaloperation
{

inline constexpr std::string_view kFrameDelayKind { "visual.frame-delay" };
inline constexpr std::string_view kFeedbackKind { "visual.feedback" };
inline constexpr std::string_view kEchoKind { "visual.echo" };
inline constexpr std::string_view kStutterKind { "visual.stutter" };
inline constexpr std::string_view kLongExposureKind { "visual.long-exposure" };
// History is a ring of textures sized from the payload, and hold is a frame
// count. Both run to the float-exact integer range of the node parameters.
inline constexpr uint32_t kMaximumHistoryLength = 1u << 24;
inline constexpr uint32_t kMaximumHoldFrames = 1u << 24;

enum class Mode : uint8_t
{
    frameDelay,
    feedback,
    echo,
    stutter,
    longExposure
};

struct Payload
{
    Mode mode = Mode::feedback;
    uint32_t historyLength = 2;
    uint32_t holdFrames = 1;
    float mix = 1.0f;
    float decay = 0.92f;
    float zoom = 0.99f;
    float swirl = 0.0f;
    float reset = 0.0f;
};

inline bool operator== (const Payload& left, const Payload& right) noexcept
{
    return left.mode == right.mode
        && left.historyLength == right.historyLength
        && left.holdFrames == right.holdFrames
        && left.mix == right.mix
        && left.decay == right.decay
        && left.zoom == right.zoom
        && left.swirl == right.swirl && left.reset == right.reset;
}

inline bool operator!= (const Payload& left, const Payload& right) noexcept
{
    return ! (left == right);
}

inline std::string_view modeName (Mode mode) noexcept
{
    switch (mode)
    {
        case Mode::frameDelay: return "frame-delay";
        case Mode::feedback: return "feedback";
        case Mode::echo: return "echo";
        case Mode::stutter: return "stutter";
        case Mode::longExposure: return "long-exposure";
    }
    return {};
}

inline bool modeForKind (std::string_view kind, Mode& mode) noexcept
{
    if (kind == kFrameDelayKind) mode = Mode::frameDelay;
    else if (kind == kFeedbackKind) mode = Mode::feedback;
    else if (kind == kEchoKind) mode = Mode::echo;
    else if (kind == kStutterKind) mode = Mode::stutter;
    else if (kind == kLongExposureKind) mode = Mode::longExposure;
    else return false;
    return true;
}

inline bool isTemporalKind (std::string_view kind) noexcept
{
    Mode ignored {};
    return modeForKind (kind, ignored);
}

inline bool validate (const Payload& payload, std::string* error = nullptr)
{
    const auto fail = [&] (const char* reason)
    {
        if (error != nullptr) *error = reason;
        return false;
    };
    if (! std::isfinite (payload.mix) || ! std::isfinite (payload.decay)
        || ! std::isfinite (payload.zoom) || ! std::isfinite (payload.swirl)
        || !std::isfinite(payload.reset) || payload.reset<0 || payload.reset>1)
        return fail ("temporal operation contains a non-finite parameter");
    if (payload.historyLength == 0 || payload.historyLength > kMaximumHistoryLength)
        return fail ("temporal operation history length is out of bounds");
    if (payload.holdFrames == 0 || payload.holdFrames > kMaximumHoldFrames)
        return fail ("temporal operation hold length is out of bounds");
    if (payload.mix < 0.0f || payload.mix > 1.0f)
        return fail ("temporal operation mix is out of bounds");

    const bool neutral = payload.decay == 0.0f && payload.zoom == 1.0f
        && payload.swirl == 0.0f;
    switch (payload.mode)
    {
        case Mode::frameDelay:
            if (payload.holdFrames != 1 || payload.mix != 1.0f || ! neutral)
                return fail ("frame delay contains unsupported parameters");
            break;
        case Mode::feedback:
            if (payload.historyLength != 2 || payload.holdFrames != 1 || payload.mix != 1.0f
                || payload.decay < 0.0f || payload.decay > 0.999f
                || payload.zoom < 0.9f || payload.zoom > 1.05f
                || payload.swirl < -0.1f || payload.swirl > 0.1f)
                return fail ("feedback parameters are out of bounds");
            break;
        case Mode::echo:
            if (payload.holdFrames != 1 || ! neutral)
                return fail ("echo contains unsupported parameters");
            break;
        case Mode::stutter:
            if (payload.historyLength != 1 || payload.mix != 1.0f || ! neutral)
                return fail ("stutter contains unsupported parameters");
            break;
        case Mode::longExposure:
            if (payload.holdFrames != 1 || payload.mix != 1.0f || ! neutral)
                return fail ("long exposure contains unsupported parameters");
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
           << (payload.reset==0 ? "temporal-v1|mode=":"temporal-v2|mode=") << modeName (payload.mode)
           << "|history=" << payload.historyLength
           << "|hold=" << payload.holdFrames
           << "|mix=" << payload.mix
           << "|decay=" << payload.decay
           << "|zoom=" << payload.zoom
           << "|swirl=" << payload.swirl;
    if (payload.reset!=0) stream << "|reset=" << payload.reset;
    return stream.str();
}

namespace detail
{
inline bool parseUnsigned (std::string_view text, uint32_t& value) noexcept
{
    if (text.empty()) return false;
    uint64_t parsed = 0;
    for (const char character : text)
    {
        if (character < '0' || character > '9') return false;
        parsed = parsed * 10u + static_cast<uint32_t> (character - '0');
        if (parsed > std::numeric_limits<uint32_t>::max()) return false;
    }
    value = static_cast<uint32_t> (parsed);
    return true;
}

inline bool parseFloat (std::string_view text, float& value) noexcept
{
    if (text.empty() || text.size() >= 64) return false;
    std::array<char, 64> copy {};
    for (size_t index = 0; index < text.size(); ++index) copy[index] = text[index];
    char* end = nullptr;
    const auto parsed = std::strtof (copy.data(), &end);
    if (end != copy.data() + static_cast<std::ptrdiff_t> (text.size()) || ! std::isfinite (parsed))
        return false;
    value = parsed;
    return true;
}
} // namespace detail

inline bool parse (std::string_view wire, Payload& payload, std::string* error = nullptr)
{
    const auto fail = [&] (const char* reason)
    {
        if (error != nullptr) *error = reason;
        return false;
    };
    std::array<std::string_view, 9> fields {};
    size_t count = 0;
    size_t begin = 0;
    while (begin <= wire.size() && count < fields.size())
    {
        const auto separator = wire.find ('|', begin);
        fields[count++] = wire.substr (begin, separator == std::string_view::npos
            ? wire.size() - begin : separator - begin);
        if (separator == std::string_view::npos) break;
        begin = separator + 1;
    }
    const bool extended=fields[0]=="temporal-v2";
    if (count != (extended ? 9u:8u) || (!extended && fields[0] != "temporal-v1")
        || std::count (wire.begin(), wire.end(), '|') != (extended ? 8:7))
        return fail ("temporal operation payload schema is malformed");

    const auto value = [&] (size_t index, std::string_view prefix,
                            std::string_view& result)
    {
        if (fields[index].substr (0, prefix.size()) != prefix) return false;
        result = fields[index].substr (prefix.size());
        return ! result.empty();
    };
    std::string_view mode, history, hold, mix, decay, zoom, swirl;
    if (! value (1, "mode=", mode) || ! value (2, "history=", history)
        || ! value (3, "hold=", hold) || ! value (4, "mix=", mix)
        || ! value (5, "decay=", decay) || ! value (6, "zoom=", zoom)
        || ! value (7, "swirl=", swirl))
        return fail ("temporal operation payload schema is malformed");

    if (mode == "frame-delay") payload.mode = Mode::frameDelay;
    else if (mode == "feedback") payload.mode = Mode::feedback;
    else if (mode == "echo") payload.mode = Mode::echo;
    else if (mode == "stutter") payload.mode = Mode::stutter;
    else if (mode == "long-exposure") payload.mode = Mode::longExposure;
    else return fail ("temporal operation mode is unsupported");

    if (! detail::parseUnsigned (history, payload.historyLength)
        || ! detail::parseUnsigned (hold, payload.holdFrames)
        || ! detail::parseFloat (mix, payload.mix)
        || ! detail::parseFloat (decay, payload.decay)
        || ! detail::parseFloat (zoom, payload.zoom)
        || ! detail::parseFloat (swirl, payload.swirl))
        return fail ("temporal operation payload values are malformed");
    payload.reset=0;
    if (extended)
    {
        std::string_view reset;
        if (!value(8,"reset=",reset) || !detail::parseFloat(reset,payload.reset))
            return fail("temporal reset value is malformed");
    }
    return validate (payload, error);
}

inline bool parseForKind (std::string_view kind, std::string_view wire, Payload& payload,
                          std::string* error = nullptr)
{
    Mode expected {};
    if (! modeForKind (kind, expected) || ! parse (wire, payload, error)) return false;
    if (payload.mode != expected)
    {
        if (error != nullptr) *error = "temporal operation kind and payload mode disagree";
        return false;
    }
    return true;
}

} // namespace visualtemporaloperation
