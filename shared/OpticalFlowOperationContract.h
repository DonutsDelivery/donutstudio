#pragma once

#include "RenderPassOutputContract.h"

#include <array>
#include <charconv>
#include <cstdint>
#include <string>
#include <string_view>

namespace opticalflowoperation
{
inline constexpr std::uint32_t kSchemaVersion = 1;
inline constexpr std::string_view kOperationKind = "visual.3d.render.passes";
inline constexpr std::string_view kBackendCapability = "native-gpu";
inline constexpr std::string_view kPairOperationKind = "visual.optical-flow";

struct SourceEndpoint final
{
    std::uint32_t nodeId = 0;
    std::uint32_t portId = 0;
    bool operator== (const SourceEndpoint& other) const noexcept
    {
        return nodeId == other.nodeId && portId == other.portId;
    }
};

struct SourcePair final
{
    std::array<SourceEndpoint, 2> images;
};

// A lowered source op names the saved media clip selected by its endpoint.
// Frame timestamps and GPU resources are resolved only by the export worker.
inline std::string serializeSourceClip (std::uint32_t clipId)
{
    return "<OpticalFlowSource schemaVersion=\"1\" clipId=\""
        + std::to_string (clipId) + "\"/>";
}

inline std::string serializeSourcePair (const SourcePair& pair)
{
    return "<OpticalFlowPair schemaVersion=\"1\" firstNode=\""
        + std::to_string (pair.images[0].nodeId) + "\" firstPort=\""
        + std::to_string (pair.images[0].portId) + "\" secondNode=\""
        + std::to_string (pair.images[1].nodeId) + "\" secondPort=\""
        + std::to_string (pair.images[1].portId) + "\"/>";
}

struct Payload final
{
    renderpassoutput::Extent extent;
};

// Canonical projection of the existing Render Passes fields for the first
// optical-flow product slice. No frame identity, resource handle, backend
// selection, or execution grant is authored into the graph; those remain owned
// by the renderer at evaluation time.
inline std::string serialize (const Payload& payload)
{
    return "<NodeParams width=\"" + std::to_string (payload.extent.width)
         + "\" height=\"" + std::to_string (payload.extent.height)
         + "\" color=\"0\" motion=\"1\"/>";
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

inline bool parseSourceClip (std::string_view wire, std::uint32_t& clipId)
{
    constexpr std::string_view prefix = "<OpticalFlowSource schemaVersion=\"1\" clipId=\"";
    constexpr std::string_view suffix = "\"/>";
    if (wire.size() <= prefix.size() + suffix.size()
        || wire.substr (0, prefix.size()) != prefix
        || wire.substr (wire.size() - suffix.size()) != suffix)
        return false;
    std::uint32_t parsed = 0;
    if (! parseUnsigned (wire.substr (prefix.size(), wire.size() - prefix.size() - suffix.size()), parsed)
        || parsed == 0 || parsed > 16777215u
        || serializeSourceClip (parsed) != std::string (wire))
        return false;
    clipId = parsed;
    return true;
}

inline bool parseSourcePair (std::string_view wire, SourcePair& destination)
{
    constexpr std::array<std::string_view, 5> parts {
        "<OpticalFlowPair schemaVersion=\"1\" firstNode=\"", "\" firstPort=\"",
        "\" secondNode=\"", "\" secondPort=\"", "\"/>"
    };
    SourcePair parsed;
    std::array<std::uint32_t*, 4> fields { &parsed.images[0].nodeId,
        &parsed.images[0].portId, &parsed.images[1].nodeId, &parsed.images[1].portId };
    if (wire.substr (0, parts[0].size()) != parts[0]) return false;
    std::size_t cursor = parts[0].size();
    for (std::size_t index = 0; index < fields.size(); ++index)
    {
        const auto end = wire.find (parts[index + 1], cursor);
        if (end == std::string_view::npos
            || ! parseUnsigned (wire.substr (cursor, end - cursor), *fields[index]))
            return false;
        cursor = end + parts[index + 1].size();
    }
    if (cursor != wire.size() || parsed.images[0].nodeId == 0
        || parsed.images[1].nodeId == 0
        || serializeSourcePair (parsed) != std::string (wire))
        return false;
    destination = parsed;
    return true;
}

inline bool parse (std::string_view wire, Payload& payload) noexcept
{
    constexpr std::string_view prefix = "<NodeParams width=\"";
    constexpr std::string_view between = "\" height=\"";
    constexpr std::string_view suffix = "\" color=\"0\" motion=\"1\"/>";
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
    const auto requirements = renderpassoutput::requirements (renderpassoutput::Output::Motion);
    description.attachments.push_back ({ renderpassoutput::Output::Motion,
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
    const auto requirements = renderpassoutput::requirements (renderpassoutput::Output::Motion);
    result.attachments.push_back ({ renderpassoutput::Output::Motion,
                                    requirements.format,
                                    requirements.colorSpace,
                                    payload.extent });
    return result;
}
} // namespace opticalflowoperation
