#pragma once

#include "../../shared/VisualHarmonicLinkGeometryContract.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace videohelper::harmonicgeometry
{
using LinkPoint = videowire::geometry::Vec3;

namespace detail
{
inline constexpr float kLinkPi = 3.14159265358979f;

inline LinkPoint offsetPoint(LinkPoint point, LinkPoint offset, float sign) noexcept
{
    return { point.x + sign * offset.x, point.y + sign * offset.y, point.z + sign * offset.z };
}

inline LinkPoint scalePoint(LinkPoint point, float scale) noexcept
{
    return { point.x * scale, point.y * scale, point.z * scale };
}

inline LinkPoint addPoint(LinkPoint a, LinkPoint b) noexcept
{
    return { a.x + b.x, a.y + b.y, a.z + b.z };
}

inline LinkPoint crossPoint(LinkPoint a, LinkPoint b) noexcept
{
    return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}

inline float lengthOf(LinkPoint point) noexcept
{
    return std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
}

// Corners of a square section around center, in counter-clockwise order seen along the axis.
inline void appendSectionCorners(videowire::geometry::GeometryData& mesh, LinkPoint center,
                                 LinkPoint u, LinkPoint v)
{
    mesh.positions.push_back(offsetPoint(offsetPoint(center, u, -1.0f), v, -1.0f));
    mesh.positions.push_back(offsetPoint(offsetPoint(center, u, 1.0f), v, -1.0f));
    mesh.positions.push_back(offsetPoint(offsetPoint(center, u, 1.0f), v, 1.0f));
    mesh.positions.push_back(offsetPoint(offsetPoint(center, u, -1.0f), v, 1.0f));
}

// Four side quads between the start section (first..first+3) and the end section (first+4..first+7).
inline void appendPrismSides(videowire::geometry::GeometryData& mesh, std::uint32_t first)
{
    for (std::uint32_t i = 0; i < 4; ++i)
    {
        const auto j = (i + 1) % 4;
        const auto startI = first + i, startJ = first + j;
        const auto endI = first + 4 + i, endJ = first + 4 + j;
        mesh.indices.insert(mesh.indices.end(), { startI, startJ, endI, startJ, endJ, endI });
    }
}
} // namespace detail

// Half width vector perpendicular to a non-zero link direction. Its length is half the mesh scale.
inline LinkPoint linkSideHalfWidth(LinkPoint direction, float meshScale)
{
    const auto length = detail::lengthOf(direction);
    LinkPoint side { -direction.y, direction.x, 0.0f };
    if (std::abs(direction.z) > length * 0.9f)
        side = { direction.z, 0.0f, -direction.x };
    const auto sideLength = detail::lengthOf(side);
    return detail::scalePoint(side, meshScale * 0.5f / sideLength);
}

// Appends one link as the requested shape at the detail in effective, as returned by
// visualharmonicgeometry::effectiveLinkLayout. side must be perpendicular to end - start with a
// length of half the ribbon width. Vertices and triangles match effective.layout.
// Vertex identities are not written here; the caller assigns them per emitted vertex.
inline void appendLinkShape(videowire::geometry::GeometryData& mesh,
                            visualharmonicgeometry::LinkShape shape,
                            std::uint32_t curvaturePercent,
                            const visualharmonicgeometry::LinkDetail& effective,
                            LinkPoint start, LinkPoint end, LinkPoint side)
{
    using visualharmonicgeometry::LinkShape;
    const auto first = static_cast<std::uint32_t>(mesh.positions.size());
    switch (shape)
    {
    case LinkShape::beam:
    {
        const LinkPoint axisVector { end.x - start.x, end.y - start.y, end.z - start.z };
        const auto length = detail::lengthOf(axisVector);
        const auto axis = detail::scalePoint(axisVector, 1.0f / length);
        // |across| equals the half width because side is perpendicular to the axis.
        const auto across = detail::crossPoint(axis, side);
        detail::appendSectionCorners(mesh, start, side, across);
        detail::appendSectionCorners(mesh, end, side, across);
        detail::appendPrismSides(mesh, first);
        // Start cap faces back along the axis, end cap faces forward. The prism is closed.
        mesh.indices.insert(mesh.indices.end(), {
            first, first + 2, first + 1, first, first + 3, first + 2,
            first + 4, first + 5, first + 6, first + 4, first + 6, first + 7 });
        break;
    }
    case LinkShape::strut:
    {
        const LinkPoint axisVector { end.x - start.x, end.y - start.y, end.z - start.z };
        const auto length = detail::lengthOf(axisVector);
        const auto axis = detail::scalePoint(axisVector, 1.0f / length);
        const auto across = detail::crossPoint(axis, side);
        // Thin tube first (0..7), then full-width end plates (8..15).
        const auto thinSide = detail::scalePoint(side, 0.5f);
        const auto thinAcross = detail::scalePoint(across, 0.5f);
        detail::appendSectionCorners(mesh, start, thinSide, thinAcross);
        detail::appendSectionCorners(mesh, end, thinSide, thinAcross);
        detail::appendPrismSides(mesh, first);
        detail::appendSectionCorners(mesh, start, side, across);
        detail::appendSectionCorners(mesh, end, side, across);
        mesh.indices.insert(mesh.indices.end(), {
            first + 8, first + 10, first + 9, first + 8, first + 11, first + 10,
            first + 12, first + 13, first + 14, first + 12, first + 14, first + 15 });
        break;
    }
    default:
    {
        // Ribbon, curve and spring are one ribbon along a polyline. Each sample carries a width
        // vector, and each segment is two triangles with the ribbon's winding.
        const auto count = visualharmonicgeometry::polylineSegments(shape, effective.coils,
                                                                    effective.segments);
        const auto count32 = static_cast<float>(count);
        const LinkPoint axisVector { end.x - start.x, end.y - start.y, end.z - start.z };
        const auto length = detail::lengthOf(axisVector);
        const auto axis = detail::scalePoint(axisVector, 1.0f / length);
        const auto halfWidth = detail::lengthOf(side);
        const auto sideUnit = detail::scalePoint(side, 1.0f / halfWidth);
        const auto curvature = static_cast<float>(curvaturePercent) / 100.0f;
        const auto bow = curvature * 0.5f * length;
        const auto coils = static_cast<float>(effective.coils);
        const auto helixAcross = detail::crossPoint(axis, sideUnit);
        const auto radius = length * 0.1f + halfWidth;
        for (std::size_t sample = 0; sample <= count; ++sample)
        {
            const auto t = static_cast<float>(sample) / count32;
            // The ribbon endpoints are exact link endpoints for every shape.
            const auto straight = detail::addPoint(start, detail::scalePoint(axisVector, t));
            LinkPoint center = sample == 0 ? start : (sample == count ? end : straight);
            LinkPoint width = side;
            if (shape == LinkShape::curve)
            {
                // Arc bowed along the side direction. The width stays normal to the tangent in that plane.
                const auto phase = detail::kLinkPi * t;
                if (sample != 0 && sample != count)
                    center = detail::offsetPoint(straight, sideUnit, bow * std::sin(phase));
                const auto normal = detail::addPoint(detail::scalePoint(sideUnit, length),
                    detail::scalePoint(axis, -bow * detail::kLinkPi * std::cos(phase)));
                width = detail::scalePoint(normal, halfWidth / detail::lengthOf(normal));
            }
            else if (shape == LinkShape::spring)
            {
                // Helix ribbon. The radial width is perpendicular to the helix tangent.
                const auto theta = 2.0f * detail::kLinkPi * coils * t;
                const auto radial = detail::addPoint(
                    detail::scalePoint(sideUnit, std::cos(theta)),
                    detail::scalePoint(helixAcross, std::sin(theta)));
                const auto offset = detail::addPoint(radial, detail::scalePoint(sideUnit, -1.0f));
                if (sample != 0 && sample != count)
                    center = detail::offsetPoint(straight, detail::scalePoint(offset, radius), 1.0f);
                width = detail::scalePoint(radial, halfWidth);
            }
            mesh.positions.push_back(detail::offsetPoint(center, width, -1.0f));
            mesh.positions.push_back(detail::offsetPoint(center, width, 1.0f));
        }
        for (std::size_t segment = 0; segment < count; ++segment)
        {
            const auto left = first + static_cast<std::uint32_t>(2 * segment);
            const auto right = left + 1;
            const auto nextLeft = left + 2;
            const auto nextRight = left + 3;
            mesh.indices.insert(mesh.indices.end(), { left, nextLeft, right, right, nextLeft, nextRight });
        }
        break;
    }
    }
}
} // namespace videohelper::harmonicgeometry
