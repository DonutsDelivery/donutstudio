#pragma once

#include "RenderPassOutputContract.h"
#include "PortableFloatingPointChars.h"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>

namespace aovinspection
{
inline constexpr std::uint32_t kSchemaVersion = 1;
inline constexpr std::string_view kDepthOperationKind = "visual.depth.inspect";
inline constexpr std::string_view kNormalOperationKind = "visual.normal.inspect";
inline constexpr std::string_view kBackendCapability = "native-gpu";

enum class Source : std::uint8_t
{
    Depth = 0,
    Normal = 1
};

struct Payload final
{
    std::uint32_t schemaVersion = kSchemaVersion;
    Source source = Source::Depth;
    renderpassoutput::Extent extent;
    float depthNear = 0.0f;
    float depthFar = 1.0f;
    bool invertDepth = false;
};

inline constexpr std::string_view token (Source source) noexcept
{
    return source == Source::Depth ? "depth" : "normal";
}

inline constexpr renderpassoutput::Output output (Source source) noexcept
{
    return source == Source::Depth ? renderpassoutput::Output::Depth
                                   : renderpassoutput::Output::Normal;
}

inline constexpr std::string_view operationKind (Source source) noexcept
{
    return source == Source::Depth ? kDepthOperationKind : kNormalOperationKind;
}

inline bool finiteAndOrdered (const Payload& payload) noexcept
{
    return std::isfinite (payload.depthNear) && std::isfinite (payload.depthFar)
        && payload.depthNear >= 0.0f && payload.depthFar <= 1.0f
        && payload.depthNear < payload.depthFar;
}

inline renderpassoutput::Description description (const Payload& payload)
{
    renderpassoutput::Description result;
    result.extent = payload.extent;
    for (const auto passOutput : { renderpassoutput::Output::Color, output (payload.source) })
    {
        const auto requirement = renderpassoutput::requirements (passOutput);
        result.attachments.push_back ({ passOutput, requirement.format,
                                        requirement.colorSpace, payload.extent });
    }
    return result;
}

namespace detail
{
inline bool parseUnsigned (std::string_view text, std::uint32_t& value) noexcept
{
    if (text.empty()) return false;
    const auto result = std::from_chars (text.data(), text.data() + text.size(), value);
    return result.ec == std::errc {} && result.ptr == text.data() + text.size();
}

inline bool parseFloat (std::string_view text, float& value) noexcept
{
    if (text.empty()) return false;
    const auto result = portablefloat::fromChars (text.data(), text.data() + text.size(), value,
                                                  std::chars_format::general);
    return result.ec == std::errc {} && result.ptr == text.data() + text.size()
        && std::isfinite (value);
}

inline bool attribute (std::string_view xml, std::string_view name,
                       std::string_view& value) noexcept
{
    std::string key;
    key.reserve (name.size() + 2);
    key.append (name);
    key.append ("=\"");
    const auto begin = xml.find (key);
    if (begin == std::string_view::npos || xml.find (key, begin + key.size()) != std::string_view::npos)
        return false;
    const auto valueBegin = begin + key.size();
    const auto valueEnd = xml.find ('\"', valueBegin);
    if (valueEnd == std::string_view::npos) return false;
    value = xml.substr (valueBegin, valueEnd - valueBegin);
    return true;
}
} // namespace detail

inline std::string serialize (const Payload& payload)
{
    if (payload.schemaVersion != kSchemaVersion || payload.extent.width == 0
        || payload.extent.height == 0 || ! finiteAndOrdered (payload))
        return {};
    std::string result = "<AovInspection schemaVersion=\"1\" source=\"";
    result += token (payload.source);
    result += "\" width=\"" + std::to_string (payload.extent.width);
    result += "\" height=\"" + std::to_string (payload.extent.height);
    result += "\" depthNear=\"" + std::to_string (payload.depthNear);
    result += "\" depthFar=\"" + std::to_string (payload.depthFar);
    result += "\" invertDepth=\"" + std::string (payload.invertDepth ? "1" : "0") + "\"/>";
    return result;
}

inline bool parse (std::string_view xml, Payload& payload) noexcept
{
    constexpr std::string_view prefix = "<AovInspection ";
    constexpr std::string_view suffix = "/>";
    if (xml.size() < prefix.size() + suffix.size()
        || xml.substr (0, prefix.size()) != prefix
        || xml.substr (xml.size() - suffix.size()) != suffix
        || xml.find ('<', 1) != std::string_view::npos)
        return false;
    std::string_view schema, source, width, height, nearValue, farValue, invert;
    Payload candidate;
    std::uint32_t schemaVersion = 0;
    if (! detail::attribute (xml, "schemaVersion", schema)
        || ! detail::attribute (xml, "source", source)
        || ! detail::attribute (xml, "width", width)
        || ! detail::attribute (xml, "height", height)
        || ! detail::attribute (xml, "depthNear", nearValue)
        || ! detail::attribute (xml, "depthFar", farValue)
        || ! detail::attribute (xml, "invertDepth", invert)
        || ! detail::parseUnsigned (schema, schemaVersion)
        || schemaVersion != kSchemaVersion
        || ! detail::parseUnsigned (width, candidate.extent.width)
        || ! detail::parseUnsigned (height, candidate.extent.height)
        || ! detail::parseFloat (nearValue, candidate.depthNear)
        || ! detail::parseFloat (farValue, candidate.depthFar)
        || (invert != "0" && invert != "1"))
        return false;
    if (source == "depth") candidate.source = Source::Depth;
    else if (source == "normal") candidate.source = Source::Normal;
    else return false;
    candidate.invertDepth = invert == "1";
    if (! finiteAndOrdered (candidate)) return false;
    renderpassoutput::AdmissionFailure failure = renderpassoutput::AdmissionFailure::None;
    if (! renderpassoutput::admit (description (candidate), failure).has_value()) return false;
    payload = candidate;
    return true;
}
} // namespace aovinspection
