#pragma once

#include "GeometryCoreTransport.h"
#include "VisualNoteInstancingContract.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <string_view>

namespace visualharmonicgeometry
{
inline constexpr const char* kNodeKind = "visual.score.harmonic-link-geometry";
inline constexpr const char* kOperationKind = "geometry.harmonic-links.runtime";
// Id of the node's optional material input. Graph code wraps it in a PortId.
inline constexpr int kMaterialPort = 2;
// Ribbon mappings keep the version 1 wire form, so saved graphs lower to the same payload.
inline constexpr std::string_view kTransportHeader = "visual.harmonic-link-geometry.v1";
// Version 2 adds one line with the link shape, curvature percent and segment count.
inline constexpr std::string_view kShapeTransportHeader = "visual.harmonic-link-geometry.v2";
// Highest payload version this build decodes. Newer payloads are refused, and the refusal names the version.
inline constexpr std::uint32_t kLatestTransportVersion = 2;
// Longest version suffix that transportVersion reads. Longer suffixes are not versions.
inline constexpr std::size_t kMaximumVersionDigits = 4;
inline constexpr std::size_t kMaximumLinks = 256;
// Vertex and index budgets of the geometry contract. They admit kMaximumLinks ribbon links.
inline constexpr std::size_t kVertexBudget = kMaximumLinks * 4;
inline constexpr std::size_t kIndexBudget = kMaximumLinks * 6;
inline constexpr std::uint32_t kMaximumSegments = 8;
inline constexpr std::uint32_t kMaximumCurvaturePercent = 100;

enum class LinkShape : std::uint32_t
{
    ribbon = 0,
    beam = 1,
    curve = 2,
    spring = 3,
    strut = 4
};
inline constexpr std::uint32_t kLinkShapeCount = 5;

struct Mapping final
{
    videowire::geometry::StableId stableId = 0;
    visualnoteinstancing::Mapping notes;
    LinkShape shape = LinkShape::ribbon;
    // Curve bow and spring coil count, in whole percent of the unit range.
    std::uint32_t curvaturePercent = 50;
    // Curve segments, or spring segments per coil.
    std::uint32_t segments = 4;
};

inline bool valid(const Mapping& mapping) noexcept
{
    const visualnoteinstancing::Mapping defaults;
    return videowire::geometry::validStableId(mapping.stableId)
        && visualnoteinstancing::valid(mapping.notes)
        && (mapping.notes.schemaVersion == defaults.schemaVersion
            || mapping.notes.schemaVersion == visualnoteinstancing::kDirectAxesMappingSchemaVersion)
        && !mapping.notes.includeMuted && visualnoteinstancing::defaultSelection(mapping.notes)
        && mapping.notes.appearanceSource == defaults.appearanceSource
        && mapping.notes.appearanceLow == defaults.appearanceLow
        && mapping.notes.appearanceHigh == defaults.appearanceHigh
        && static_cast<std::uint32_t>(mapping.shape) < kLinkShapeCount
        && mapping.curvaturePercent <= kMaximumCurvaturePercent
        && mapping.segments >= 1 && mapping.segments <= kMaximumSegments;
}

inline const char* shapeName(LinkShape shape) noexcept
{
    switch (shape)
    {
    case LinkShape::ribbon: return "Ribbon";
    case LinkShape::beam: return "Beam";
    case LinkShape::curve: return "Curve";
    case LinkShape::spring: return "Spring";
    case LinkShape::strut: return "Strut";
    }
    return "Ribbon";
}

// Spring coils rise from one to six as curvature goes from zero to one hundred percent.
inline std::uint32_t springCoils(std::uint32_t curvaturePercent) noexcept
{
    return 1 + (curvaturePercent * 5 + 50) / 100;
}

// Number of polyline segments that a link samples, given spring coils and segments. Spring samples
// each coil in segments steps. Ribbon, beam and strut sample one segment.
inline std::size_t polylineSegments(LinkShape shape, std::uint32_t coils,
                                    std::uint32_t segments) noexcept
{
    switch (shape)
    {
    case LinkShape::curve: return segments;
    case LinkShape::spring: return static_cast<std::size_t>(coils) * segments;
    case LinkShape::ribbon:
    case LinkShape::beam:
    case LinkShape::strut: return 1;
    }
    return 1;
}

struct LinkLayout final
{
    std::size_t vertices = 0;
    std::size_t triangles = 0;
};

// Vertices and triangles emitted for one link, given spring coils and segments. Face attributes
// carry one element per triangle. Beam and strut have fixed layouts.
inline LinkLayout linkLayoutAt(LinkShape shape, std::uint32_t coils, std::uint32_t segments) noexcept
{
    switch (shape)
    {
    case LinkShape::beam: return { 8, 12 };
    case LinkShape::strut: return { 16, 12 };
    case LinkShape::ribbon:
    case LinkShape::curve:
    case LinkShape::spring:
    {
        const auto count = polylineSegments(shape, coils, segments);
        return { 2 * (count + 1), 2 * count };
    }
    }
    return {};
}

// Largest link count that the vertex and index budgets admit for one layout.
inline std::size_t linkCapacity(LinkLayout layout) noexcept
{
    if (layout.vertices == 0 || layout.triangles == 0) return 0;
    return std::min({ kMaximumLinks, kVertexBudget / layout.vertices,
                      kIndexBudget / (layout.triangles * 3) });
}

// Detail emitted for one link set. Every link of the set shares its segments, coils and layout.
struct LinkDetail final
{
    std::uint32_t segments = 1; // Curve segments, or spring segments per coil.
    std::uint32_t coils = 1;    // Spring coils. Other shapes report one.
    LinkLayout layout;          // Vertices and triangles emitted for each link.
    bool admitted = false;      // False when the link count exceeds the budget at minimum detail.
};

// True while the detail has a unit left to drop. Beam, strut and ribbon never have one.
inline bool canLowerDetail(LinkShape shape, const LinkDetail& detail) noexcept
{
    switch (shape)
    {
    case LinkShape::curve: return detail.segments > 1;
    case LinkShape::spring: return detail.segments > 1 || detail.coils > 1;
    case LinkShape::ribbon:
    case LinkShape::beam:
    case LinkShape::strut: return false;
    }
    return false;
}

// Drops one unit from the larger of spring coils and segments per coil, and from segments on a tie.
// Callers check canLowerDetail first.
inline void lowerDetail(LinkShape shape, LinkDetail& detail) noexcept
{
    if (shape == LinkShape::spring && detail.coils > detail.segments)
        --detail.coils;
    else
        --detail.segments;
}

// The level of detail rule. Emission and admission both use it, and its output depends only on the
// arguments. The requested segments and curvature hold while the budgets admit linkCount links.
// Otherwise the detail steps down one unit at a time until they do. The floor is one segment and
// one coil. When linkCount does not fit even at the floor, admitted is false.
inline LinkDetail effectiveLinkLayout(LinkShape shape, std::uint32_t curvaturePercent,
                                      std::uint32_t segments, std::size_t linkCount) noexcept
{
    LinkDetail effective;
    effective.segments = segments;
    effective.coils = shape == LinkShape::spring ? springCoils(curvaturePercent)
                                                 : std::uint32_t { 1 };
    effective.layout = linkLayoutAt(shape, effective.coils, effective.segments);
    while (linkCount > linkCapacity(effective.layout) && canLowerDetail(shape, effective))
    {
        lowerDetail(shape, effective);
        effective.layout = linkLayoutAt(shape, effective.coils, effective.segments);
    }
    effective.admitted = linkCount <= linkCapacity(effective.layout);
    return effective;
}

inline std::string encode(const Mapping& mapping)
{
    if (!valid(mapping)) return {};
    const auto identity = std::to_string(mapping.stableId) + '\n';
    if (mapping.shape == LinkShape::ribbon)
        return std::string(kTransportHeader) + '\n' + identity
            + visualnoteinstancing::encode(mapping.notes);
    return std::string(kShapeTransportHeader) + '\n' + identity
        + std::to_string(static_cast<std::uint32_t>(mapping.shape)) + ' '
        + std::to_string(mapping.curvaturePercent) + ' ' + std::to_string(mapping.segments)
        + '\n' + visualnoteinstancing::encode(mapping.notes);
}

inline bool decode(std::string_view encoded, Mapping& output)
{
    output = {};
    if (encoded.size() > visualnoteinstancing::kMaximumTransportBytes + 128) return false;
    const auto first = encoded.find('\n');
    if (first == std::string_view::npos) return false;
    const auto header = encoded.substr(0, first);
    const bool shaped = header == kShapeTransportHeader;
    if (!shaped && header != kTransportHeader) return false;
    const auto second = encoded.find('\n', first + 1);
    if (second == std::string_view::npos || second == first + 1) return false;
    Mapping candidate;
    for (auto index = first + 1; index < second; ++index)
    {
        const auto digit = encoded[index];
        if (digit < '0' || digit > '9'
            || candidate.stableId > videowire::geometry::kMaximumStableId / 10)
            return false;
        candidate.stableId = candidate.stableId * 10 + static_cast<unsigned>(digit - '0');
    }
    auto notesText = encoded.substr(second + 1);
    if (shaped)
    {
        // Version 1 payloads have no shape line and decode as Ribbon with the defaults.
        const auto third = encoded.find('\n', second + 1);
        if (third == std::string_view::npos) return false;
        std::istringstream line{ std::string(encoded.substr(second + 1, third - second - 1)) };
        std::uint32_t shape = 0, curvature = 0, segmentCount = 0;
        if (!(line >> shape >> curvature >> segmentCount)) return false;
        line >> std::ws;
        if (!line.eof() || shape >= kLinkShapeCount) return false;
        candidate.shape = static_cast<LinkShape>(shape);
        candidate.curvaturePercent = curvature;
        candidate.segments = segmentCount;
        notesText = encoded.substr(third + 1);
    }
    if (!visualnoteinstancing::decode(notesText, candidate.notes)
        || !valid(candidate) || encode(candidate) != encoded) return false;
    output = candidate;
    return true;
}

// Version named by the header line of an encoded payload, or 0 when the line is not a harmonic link
// header. Lets a caller tell a newer payload apart from a malformed one.
inline std::uint32_t transportVersion(std::string_view encoded) noexcept
{
    constexpr std::string_view prefix = "visual.harmonic-link-geometry.v";
    const auto header = encoded.substr(0, encoded.find('\n'));
    if (header.size() <= prefix.size() || header.substr(0, prefix.size()) != prefix) return 0;
    const auto digits = header.substr(prefix.size());
    if (digits.size() > kMaximumVersionDigits) return 0;
    std::uint32_t version = 0;
    for (const auto digit : digits)
    {
        if (digit < '0' || digit > '9') return 0;
        version = version * 10 + static_cast<std::uint32_t>(digit - '0');
    }
    return version;
}

inline videowire::geometry::PortContract geometryContract()
{
    using namespace videowire::geometry;
    PortContract contract;
    contract.carrier = CarrierKind::geometry3D;
    contract.overflow = OverflowPolicy::reject;
    contract.maxVertices = kVertexBudget;
    contract.maxIndices = kIndexBudget;
    contract.maxAttributes = 5;
    const char* names[] { "harmonicLinkId", "masterNoteId", "slaveNoteId",
                          "ratioNumerator", "ratioDenominator" };
    for (std::size_t index = 0; index < 5; ++index)
        contract.attributes.push_back({index + 1, names[index], ValueType::integer,
                                       Domain::face, Interpolation::constant});
    return contract;
}

// Largest link count that the budgets admit for a shape. Level of detail steps down until a layout
// fits, so the bound is the capacity at minimum detail.
inline std::size_t maximumLinks(LinkShape shape) noexcept
{
    return linkCapacity(linkLayoutAt(shape, 1, 1));
}

// Refuses a link count that the budgets do not admit at any level of detail.
inline bool admitLinkCount(LinkShape shape, std::uint32_t curvaturePercent,
                           std::uint32_t segments, std::size_t links, std::string& error)
{
    if (effectiveLinkLayout(shape, curvaturePercent, segments, links).admitted)
    {
        error.clear();
        return true;
    }
    error = std::string("Harmonic Link Geometry ") + shapeName(shape)
        + " exceeds the geometry budget: " + std::to_string(links) + " links requested, "
        + std::to_string(maximumLinks(shape)) + " admitted";
    return false;
}
} // namespace visualharmonicgeometry
