#pragma once

#include "../../shared/PathMaskContract.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace videohelper::pathmask
{
inline constexpr std::uint32_t kQuadraticLineSegments = 8;
inline constexpr std::uint32_t kCubicLineSegments = 16;
inline constexpr bool kAllowsCpuProductionImageRendering = false;
inline constexpr bool kAuthorizesGraphBypass = false;

using videowire::pathmask::Contour;
using videowire::pathmask::Description;
using videowire::pathmask::FillRule;
using videowire::pathmask::Limits;
using videowire::pathmask::Point;
using videowire::pathmask::Segment;
using videowire::pathmask::SegmentKind;

struct FlattenedLine final
{
    videowire::pathmask::ContourId contourId = 0;
    Point from {};
    Point to {};
    bool contourClosed = false;
    bool implicitClosingLine = false;
    bool contributesToFill = false;
};

struct MemoryFootprint final
{
    std::size_t logicalPointCount = 0;
    std::size_t flattenedLineCount = 0;
    std::size_t canonicalScalarBytes = 0;
    std::size_t admittedValueBytes = 0;
    std::size_t loweredLineBytes = 0;
};

enum class AdmissionFailure : std::uint8_t
{
    none = 0,
    unsupportedVersion,
    invalidLimits,
    invalidAssetIdentity,
    invalidRevision,
    invalidFillRule,
    invalidFeather,
    invalidChoke,
    contourLimitExceeded,
    segmentLimitExceeded,
    emptyAsset,
    invalidContourIdentity,
    duplicateContourIdentity,
    noncanonicalSegmentRange,
    emptyContour,
    closedContourTooShort,
    noClosedContour,
    invalidCoordinate,
    invalidSegmentKind,
    noncanonicalUnusedControl,
    explicitClosureForbidden,
    pointLimitExceeded,
    byteCountOverflow
};

inline constexpr std::string_view token (AdmissionFailure failure) noexcept
{
    switch (failure)
    {
        case AdmissionFailure::none: return "none";
        case AdmissionFailure::unsupportedVersion: return "unsupportedVersion";
        case AdmissionFailure::invalidLimits: return "invalidLimits";
        case AdmissionFailure::invalidAssetIdentity: return "invalidAssetIdentity";
        case AdmissionFailure::invalidRevision: return "invalidRevision";
        case AdmissionFailure::invalidFillRule: return "invalidFillRule";
        case AdmissionFailure::invalidFeather: return "invalidFeather";
        case AdmissionFailure::invalidChoke: return "invalidChoke";
        case AdmissionFailure::contourLimitExceeded: return "contourLimitExceeded";
        case AdmissionFailure::segmentLimitExceeded: return "segmentLimitExceeded";
        case AdmissionFailure::emptyAsset: return "emptyAsset";
        case AdmissionFailure::invalidContourIdentity: return "invalidContourIdentity";
        case AdmissionFailure::duplicateContourIdentity: return "duplicateContourIdentity";
        case AdmissionFailure::noncanonicalSegmentRange: return "noncanonicalSegmentRange";
        case AdmissionFailure::emptyContour: return "emptyContour";
        case AdmissionFailure::closedContourTooShort: return "closedContourTooShort";
        case AdmissionFailure::noClosedContour: return "noClosedContour";
        case AdmissionFailure::invalidCoordinate: return "invalidCoordinate";
        case AdmissionFailure::invalidSegmentKind: return "invalidSegmentKind";
        case AdmissionFailure::noncanonicalUnusedControl: return "noncanonicalUnusedControl";
        case AdmissionFailure::explicitClosureForbidden: return "explicitClosureForbidden";
        case AdmissionFailure::pointLimitExceeded: return "pointLimitExceeded";
        case AdmissionFailure::byteCountOverflow: return "byteCountOverflow";
    }
    return {};
}

namespace detail
{
inline constexpr bool checkedAdd (std::size_t left, std::size_t right,
                                  std::size_t& result) noexcept
{
    if (right > std::numeric_limits<std::size_t>::max() - left)
        return false;
    result = left + right;
    return true;
}

inline constexpr bool checkedMultiply (std::size_t left, std::size_t right,
                                       std::size_t& result) noexcept
{
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
        return false;
    result = left * right;
    return true;
}

inline bool limitsValid (const Limits& limits) noexcept
{
    return limits.maximumContours != 0
        && limits.maximumContours <= videowire::pathmask::kMaximumContours
        && limits.maximumSegments != 0
        && limits.maximumSegments <= videowire::pathmask::kMaximumSegments
        && limits.maximumLogicalPoints != 0
        && limits.maximumLogicalPoints <= videowire::pathmask::kMaximumLogicalPoints
        && std::isfinite (limits.maximumFeather)
        && limits.maximumFeather >= 0.0
        && limits.maximumFeather <= videowire::pathmask::kMaximumFeather
        && std::isfinite (limits.maximumAbsoluteChoke)
        && limits.maximumAbsoluteChoke >= 0.0
        && limits.maximumAbsoluteChoke <= videowire::pathmask::kMaximumAbsoluteChoke;
}

inline bool identityValid (std::uint64_t value) noexcept
{
    return value != 0 && value <= videowire::pathmask::kMaximumScalarIdentity;
}

inline bool normalized (const Point& point) noexcept
{
    return std::isfinite (point.x) && std::isfinite (point.y)
        && point.x >= 0.0 && point.x <= 1.0
        && point.y >= 0.0 && point.y <= 1.0;
}

inline bool zero (const Point& point) noexcept
{
    return point.x == 0.0 && point.y == 0.0;
}

inline Point canonicalPoint (Point point) noexcept
{
    if (point.x == 0.0) point.x = 0.0;
    if (point.y == 0.0) point.y = 0.0;
    return point;
}

inline bool samePoint (const Point& left, const Point& right) noexcept
{
    return left.x == right.x && left.y == right.y;
}

inline std::size_t pointCount (SegmentKind kind) noexcept
{
    switch (kind)
    {
        case SegmentKind::line: return 1;
        case SegmentKind::quadratic: return 2;
        case SegmentKind::cubic: return 3;
    }
    return 0;
}

inline std::size_t subdivisionCount (SegmentKind kind) noexcept
{
    switch (kind)
    {
        case SegmentKind::line: return 1;
        case SegmentKind::quadratic: return kQuadraticLineSegments;
        case SegmentKind::cubic: return kCubicLineSegments;
    }
    return 0;
}

inline Point evaluateSegment (Point start, const Segment& segment, double time) noexcept
{
    const auto oneMinus = 1.0 - time;
    switch (segment.kind)
    {
        case SegmentKind::line:
            return { start.x * oneMinus + segment.end.x * time,
                     start.y * oneMinus + segment.end.y * time };
        case SegmentKind::quadratic:
            return { oneMinus * oneMinus * start.x
                         + 2.0 * oneMinus * time * segment.control1.x
                         + time * time * segment.end.x,
                     oneMinus * oneMinus * start.y
                         + 2.0 * oneMinus * time * segment.control1.y
                         + time * time * segment.end.y };
        case SegmentKind::cubic:
            return { oneMinus * oneMinus * oneMinus * start.x
                         + 3.0 * oneMinus * oneMinus * time * segment.control1.x
                         + 3.0 * oneMinus * time * time * segment.control2.x
                         + time * time * time * segment.end.x,
                     oneMinus * oneMinus * oneMinus * start.y
                         + 3.0 * oneMinus * oneMinus * time * segment.control1.y
                         + 3.0 * oneMinus * time * time * segment.control2.y
                         + time * time * time * segment.end.y };
    }
    return start;
}

inline double distanceSquaredToLine (Point point, Point from, Point to) noexcept
{
    const auto dx = to.x - from.x;
    const auto dy = to.y - from.y;
    const auto lengthSquared = dx * dx + dy * dy;
    if (lengthSquared == 0.0)
    {
        const auto px = point.x - from.x;
        const auto py = point.y - from.y;
        return px * px + py * py;
    }
    const auto projection = std::clamp (
        ((point.x - from.x) * dx + (point.y - from.y) * dy) / lengthSquared,
        0.0, 1.0);
    const auto px = point.x - (from.x + projection * dx);
    const auto py = point.y - (from.y + projection * dy);
    return px * px + py * py;
}

inline int windingContribution (Point point, Point from, Point to) noexcept
{
    const auto cross = (to.x - from.x) * (point.y - from.y)
        - (point.x - from.x) * (to.y - from.y);
    if (from.y <= point.y)
        return to.y > point.y && cross > 0.0 ? 1 : 0;
    return to.y <= point.y && cross < 0.0 ? -1 : 0;
}
} // namespace detail

class AdmittedPathMask final
{
public:
    AdmittedPathMask (const AdmittedPathMask&) = default;
    AdmittedPathMask (AdmittedPathMask&&) noexcept = default;
    AdmittedPathMask& operator= (const AdmittedPathMask&) = delete;
    AdmittedPathMask& operator= (AdmittedPathMask&&) = delete;

    std::uint32_t schemaVersion() const noexcept { return videowire::pathmask::kSchemaVersion; }
    videowire::pathmask::AssetId assetId() const noexcept { return assetId_; }
    std::uint64_t revision() const noexcept { return revision_; }
    FillRule fillRule() const noexcept { return fillRule_; }
    double feather() const noexcept { return feather_; }
    double choke() const noexcept { return choke_; }
    const std::vector<Contour>& contours() const noexcept { return contours_; }
    const std::vector<Segment>& segments() const noexcept { return segments_; }

    MemoryFootprint footprint() const noexcept
    {
        MemoryFootprint result = footprint_;
        result.admittedValueBytes = sizeof (*this)
            + contours_.capacity() * sizeof (Contour)
            + segments_.capacity() * sizeof (Segment);
        return result;
    }

private:
    friend std::optional<AdmittedPathMask> admit (
        const Description&, const Limits&, AdmissionFailure&);

    AdmittedPathMask (const Description& source, std::vector<Contour> contours,
                      std::vector<Segment> segments, MemoryFootprint footprint)
        : assetId_ (source.assetId), revision_ (source.revision), fillRule_ (source.fillRule),
          feather_ (source.feather == 0.0 ? 0.0 : source.feather),
          choke_ (source.choke == 0.0 ? 0.0 : source.choke),
          contours_ (std::move (contours)), segments_ (std::move (segments)),
          footprint_ (footprint)
    {
    }

    videowire::pathmask::AssetId assetId_ = 0;
    std::uint64_t revision_ = 0;
    FillRule fillRule_ = FillRule::nonZero;
    double feather_ = 0.0;
    double choke_ = 0.0;
    std::vector<Contour> contours_;
    std::vector<Segment> segments_;
    MemoryFootprint footprint_;
};

inline std::optional<AdmittedPathMask> admit (const Description& source,
                                              const Limits& limits,
                                              AdmissionFailure& failure)
{
    failure = AdmissionFailure::none;
    const auto reject = [&] (AdmissionFailure reason) -> std::optional<AdmittedPathMask>
    {
        failure = reason;
        return std::nullopt;
    };

    if (source.schemaVersion != videowire::pathmask::kSchemaVersion)
        return reject (AdmissionFailure::unsupportedVersion);
    if (! detail::limitsValid (limits))
        return reject (AdmissionFailure::invalidLimits);
    if (! detail::identityValid (source.assetId))
        return reject (AdmissionFailure::invalidAssetIdentity);
    if (! detail::identityValid (source.revision))
        return reject (AdmissionFailure::invalidRevision);
    if (source.fillRule != FillRule::evenOdd && source.fillRule != FillRule::nonZero)
        return reject (AdmissionFailure::invalidFillRule);
    if (! std::isfinite (source.feather) || source.feather < 0.0
        || source.feather > limits.maximumFeather)
        return reject (AdmissionFailure::invalidFeather);
    if (! std::isfinite (source.choke) || std::abs (source.choke) > limits.maximumAbsoluteChoke)
        return reject (AdmissionFailure::invalidChoke);
    if (source.contours.empty() || source.segments.empty())
        return reject (AdmissionFailure::emptyAsset);
    if (source.contours.size() > limits.maximumContours)
        return reject (AdmissionFailure::contourLimitExceeded);
    if (source.segments.size() > limits.maximumSegments)
        return reject (AdmissionFailure::segmentLimitExceeded);

    std::vector<Contour> contours (source.contours.begin(), source.contours.end());
    std::vector<Segment> segments (source.segments.begin(), source.segments.end());
    std::size_t expectedFirstSegment = 0;
    std::size_t logicalPoints = 0;
    std::size_t flattenedLines = 0;
    bool hasClosedContour = false;

    for (std::size_t contourIndex = 0; contourIndex < contours.size(); ++contourIndex)
    {
        auto& contour = contours[contourIndex];
        if (! detail::identityValid (contour.id))
            return reject (AdmissionFailure::invalidContourIdentity);
        for (std::size_t previous = 0; previous < contourIndex; ++previous)
            if (contours[previous].id == contour.id)
                return reject (AdmissionFailure::duplicateContourIdentity);
        if (! detail::normalized (contour.start))
            return reject (AdmissionFailure::invalidCoordinate);
        contour.start = detail::canonicalPoint (contour.start);
        if (contour.firstSegment != expectedFirstSegment
            || contour.segmentCount > segments.size() - expectedFirstSegment)
            return reject (AdmissionFailure::noncanonicalSegmentRange);
        if (contour.segmentCount == 0)
            return reject (AdmissionFailure::emptyContour);
        if (contour.closed && contour.segmentCount < 2)
            return reject (AdmissionFailure::closedContourTooShort);
        hasClosedContour = hasClosedContour || contour.closed;

        if (! detail::checkedAdd (logicalPoints, 1, logicalPoints))
            return reject (AdmissionFailure::byteCountOverflow);
        Point previousPoint = contour.start;
        for (std::size_t offset = 0; offset < contour.segmentCount; ++offset)
        {
            auto& segment = segments[expectedFirstSegment + offset];
            const auto points = detail::pointCount (segment.kind);
            const auto subdivisions = detail::subdivisionCount (segment.kind);
            if (points == 0 || subdivisions == 0)
                return reject (AdmissionFailure::invalidSegmentKind);
            if (! detail::normalized (segment.end)
                || (segment.kind != SegmentKind::line && ! detail::normalized (segment.control1))
                || (segment.kind == SegmentKind::cubic && ! detail::normalized (segment.control2)))
                return reject (AdmissionFailure::invalidCoordinate);
            if ((segment.kind == SegmentKind::line
                 && (! detail::zero (segment.control1) || ! detail::zero (segment.control2)))
                || (segment.kind == SegmentKind::quadratic && ! detail::zero (segment.control2)))
                return reject (AdmissionFailure::noncanonicalUnusedControl);

            segment.end = detail::canonicalPoint (segment.end);
            segment.control1 = detail::canonicalPoint (segment.control1);
            segment.control2 = detail::canonicalPoint (segment.control2);
            previousPoint = segment.end;
            if (! detail::checkedAdd (logicalPoints, points, logicalPoints)
                || ! detail::checkedAdd (flattenedLines, subdivisions, flattenedLines))
                return reject (AdmissionFailure::byteCountOverflow);
        }
        if (contour.closed)
        {
            if (detail::samePoint (previousPoint, contour.start))
                return reject (AdmissionFailure::explicitClosureForbidden);
            if (! detail::checkedAdd (flattenedLines, 1, flattenedLines))
                return reject (AdmissionFailure::byteCountOverflow);
        }
        expectedFirstSegment += contour.segmentCount;
    }

    if (expectedFirstSegment != segments.size())
        return reject (AdmissionFailure::noncanonicalSegmentRange);
    if (! hasClosedContour)
        return reject (AdmissionFailure::noClosedContour);
    if (logicalPoints > limits.maximumLogicalPoints)
        return reject (AdmissionFailure::pointLimitExceeded);

    // Compact scalar encoding sizes are fixed by this schema and do not depend
    // on compiler padding: header 45, contour 33, segment 49 bytes.
    std::size_t contourBytes = 0;
    std::size_t segmentBytes = 0;
    std::size_t scalarBytes = 45;
    std::size_t loweredBytes = 0;
    if (! detail::checkedMultiply (contours.size(), 33, contourBytes)
        || ! detail::checkedMultiply (segments.size(), 49, segmentBytes)
        || ! detail::checkedAdd (scalarBytes, contourBytes, scalarBytes)
        || ! detail::checkedAdd (scalarBytes, segmentBytes, scalarBytes)
        || ! detail::checkedMultiply (flattenedLines, sizeof (FlattenedLine), loweredBytes))
        return reject (AdmissionFailure::byteCountOverflow);

    MemoryFootprint footprint;
    footprint.logicalPointCount = logicalPoints;
    footprint.flattenedLineCount = flattenedLines;
    footprint.canonicalScalarBytes = scalarBytes;
    footprint.loweredLineBytes = loweredBytes;
    return AdmittedPathMask (source, std::move (contours), std::move (segments), footprint);
}

template <typename Visitor>
void visitFlattenedLines (const AdmittedPathMask& asset, Visitor&& visitor)
{
    const auto& segments = asset.segments();
    for (const auto& contour : asset.contours())
    {
        Point previous = contour.start;
        for (std::size_t offset = 0; offset < contour.segmentCount; ++offset)
        {
            const auto& segment = segments[contour.firstSegment + offset];
            const auto segmentStart = previous;
            auto lineStart = segmentStart;
            const auto subdivisions = detail::subdivisionCount (segment.kind);
            for (std::size_t step = 1; step <= subdivisions; ++step)
            {
                const auto next = detail::evaluateSegment (
                    segmentStart, segment, static_cast<double> (step)
                        / static_cast<double> (subdivisions));
                visitor (FlattenedLine { contour.id, lineStart, next, contour.closed,
                                         false, contour.closed });
                lineStart = next;
            }
            previous = segment.end;
        }
        if (contour.closed)
            visitor (FlattenedLine { contour.id, previous, contour.start, true, true, true });
    }
}

struct GeometrySample final
{
    bool inside = false;
    bool onBoundary = false;
    int windingNumber = 0;
    double signedDistance = 0.0;
    double coverage = 0.0;
};

enum class EvaluationFailure : std::uint8_t
{
    none = 0,
    invalidCoordinate
};

inline constexpr std::string_view token (EvaluationFailure failure) noexcept
{
    switch (failure)
    {
        case EvaluationFailure::none: return "none";
        case EvaluationFailure::invalidCoordinate: return "invalidCoordinate";
    }
    return {};
}

// This scalar reference evaluator never creates an image. Curves use fixed
// subdivisions, closed contours receive one implicit closing line, and open
// contours are lowered without closure but never participate in fill coverage.
inline std::optional<GeometrySample> evaluate (const AdmittedPathMask& asset,
                                               Point coordinate,
                                               EvaluationFailure& failure) noexcept
{
    failure = EvaluationFailure::none;
    if (! detail::normalized (coordinate))
    {
        failure = EvaluationFailure::invalidCoordinate;
        return std::nullopt;
    }
    coordinate = detail::canonicalPoint (coordinate);

    constexpr double boundaryEpsilonSquared = 1.0e-24;
    auto minimumDistanceSquared = std::numeric_limits<double>::infinity();
    int winding = 0;
    int crossings = 0;
    bool boundary = false;
    visitFlattenedLines (asset, [&] (const FlattenedLine& line)
    {
        if (! line.contributesToFill)
            return;
        const auto distanceSquared = detail::distanceSquaredToLine (
            coordinate, line.from, line.to);
        minimumDistanceSquared = std::min (minimumDistanceSquared, distanceSquared);
        boundary = boundary || distanceSquared <= boundaryEpsilonSquared;
        const auto contribution = detail::windingContribution (coordinate, line.from, line.to);
        winding += contribution;
        if (contribution != 0)
            ++crossings;
    });

    const auto insideByRule = asset.fillRule() == FillRule::evenOdd
        ? (crossings % 2) != 0
        : winding != 0;
    const auto inside = boundary || insideByRule;
    const auto distance = std::sqrt (minimumDistanceSquared);
    const auto signedDistance = inside ? -distance : distance;
    const auto shiftedDistance = signedDistance + asset.choke();

    double coverage = 0.0;
    if (asset.feather() == 0.0)
    {
        coverage = shiftedDistance <= 0.0 ? 1.0 : 0.0;
    }
    else
    {
        // Feather is the full transition width centered on the choked boundary.
        const auto blend = std::clamp (
            0.5 - shiftedDistance / asset.feather(), 0.0, 1.0);
        coverage = blend * blend * (3.0 - 2.0 * blend);
    }

    return GeometrySample { inside, boundary, winding, signedDistance, coverage };
}
} // namespace videohelper::pathmask
