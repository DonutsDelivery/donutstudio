#pragma once

#include "RenderPassOutputContract.h"

#include <charconv>
#include <cstdint>
#include <string>
#include <string_view>

namespace coloraov
{
inline constexpr std::uint32_t kSchemaVersion = 1;
inline constexpr std::string_view kOperationKind = "visual.3d.render.passes";
inline constexpr std::string_view kBackendCapability = "native-gpu";

struct Payload final
{
    renderpassoutput::Extent extent;
};

// This payload is a canonical projection of the existing Render Passes node
// controls. It deliberately adds no clear-colour or backend fields: Color AOV v1
// has one fixed transparent-black clear semantic and native-GPU-only execution.
inline std::string serialize (const Payload& payload)
{
    return "<NodeParams width=\"" + std::to_string (payload.extent.width)
         + "\" height=\"" + std::to_string (payload.extent.height)
         + "\" color=\"1\"/>";
}

inline bool parseUnsigned (std::string_view value, std::uint32_t& result) noexcept
{
    if (value.empty()) return false;
    std::uint32_t parsed = 0;
    const auto conversion = std::from_chars (value.data(), value.data() + value.size(), parsed);
    if (conversion.ec != std::errc {} || conversion.ptr != value.data() + value.size())
        return false;
    result = parsed;
    return true;
}

inline bool parse (std::string_view wire, Payload& payload) noexcept
{
    constexpr std::string_view prefix = "<NodeParams width=\"";
    constexpr std::string_view between = "\" height=\"";
    constexpr std::string_view suffix = "\" color=\"1\"/>";
    if (wire.size() <= prefix.size() + between.size() + suffix.size()
        || wire.substr (0, prefix.size()) != prefix)
        return false;

    const auto widthEnd = wire.find (between, prefix.size());
    if (widthEnd == std::string_view::npos) return false;
    const auto heightBegin = widthEnd + between.size();
    if (wire.size() < suffix.size()
        || wire.substr (wire.size() - suffix.size()) != suffix
        || heightBegin > wire.size() - suffix.size())
        return false;

    Payload parsed;
    if (! parseUnsigned (wire.substr (prefix.size(), widthEnd - prefix.size()),
                         parsed.extent.width)
        || ! parseUnsigned (wire.substr (heightBegin,
                           wire.size() - suffix.size() - heightBegin),
                            parsed.extent.height))
        return false;

    renderpassoutput::Description description;
    description.extent = parsed.extent;
    const auto requirements = renderpassoutput::requirements (renderpassoutput::Output::Color);
    description.attachments.push_back ({ renderpassoutput::Output::Color,
                                         requirements.format,
                                         requirements.colorSpace,
                                         parsed.extent });
    renderpassoutput::AdmissionFailure failure = renderpassoutput::AdmissionFailure::None;
    if (! renderpassoutput::admit (description, failure).has_value()) return false;
    payload = parsed;
    return true;
}

inline renderpassoutput::Description description (const Payload& payload)
{
    renderpassoutput::Description result;
    result.extent = payload.extent;
    const auto requirements = renderpassoutput::requirements (renderpassoutput::Output::Color);
    result.attachments.push_back ({ renderpassoutput::Output::Color,
                                    requirements.format,
                                    requirements.colorSpace,
                                    payload.extent });
    return result;
}
} // namespace coloraov
