#pragma once

// Realtime's planar mirror for a grating. A grating's orders depend on the direction it
// is seen from, so the screen-space reflection, which copies the grating as the camera
// sees it, showed a glossy floor the wrong colours: Path Trace's floor reflects the
// holographic card with its foil streak where the mirror image of the camera sees it,
// and Realtime's showed the card's grey body and the bottom of its top streak.
//
// For each flat glossy surface, the grating draws once more into a mirror target: each
// vertex is placed at its mirror image in the plane, and each fragment shades the real
// grating point as the camera's mirror image sees it (uViewerMirror). A fragment keeps
// only pixels whose visible surface lies on that plane and points on the camera's side
// of it. The surface's reflection phase reads the target where it has the grating and
// marches the screen as before elsewhere.

#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <vector>

namespace arbitgpu::realtimeplanarmirror
{
using Point = std::array<float, 3>;
// Unit normal in xyz and offset in w: a point p lies on it when dot(n, p) + w is 0.
using Plane = std::array<float, 4>;

inline float distance (const Plane& plane, const Point& p) noexcept
{
    return plane[0] * p[0] + plane[1] * p[1] + plane[2] * p[2] + plane[3];
}

// The plane all points lie on, or nothing when they do not span one or one point leaves
// it by more than 1e-5 of their extent. The normal faces the viewer.
inline std::optional<Plane> plane (const std::vector<Point>& points, const Point& viewer)
{
    if (points.size() < 3) return std::nullopt;
    const auto& a = points[0];
    float extent = 0.0f;
    std::size_t far = 0;
    for (std::size_t i = 1; i < points.size(); ++i)
    {
        const float dx = points[i][0] - a[0], dy = points[i][1] - a[1], dz = points[i][2] - a[2];
        const float d = std::sqrt (dx * dx + dy * dy + dz * dz);
        if (d > extent) { extent = d; far = i; }
    }
    if (! (extent > 0.0f)) return std::nullopt;
    const Point u { points[far][0] - a[0], points[far][1] - a[1], points[far][2] - a[2] };
    // The point farthest from the line a-far spans the plane with it.
    Point normal {};
    float best = 0.0f;
    for (const auto& p : points)
    {
        const Point v { p[0] - a[0], p[1] - a[1], p[2] - a[2] };
        const Point c { u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0] };
        const float length = std::sqrt (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
        if (length > best) { best = length; normal = { c[0] / length, c[1] / length, c[2] / length }; }
    }
    if (! (best > 1.0e-4f * extent * extent)) return std::nullopt;
    Plane result { normal[0], normal[1], normal[2],
                   -(normal[0] * a[0] + normal[1] * a[1] + normal[2] * a[2]) };
    const float tolerance = 1.0e-5f * extent + 1.0e-6f;
    for (const auto& p : points)
        if (std::abs (distance (result, p)) > tolerance) return std::nullopt;
    if (distance (result, viewer) < 0.0f)
        for (auto& value : result) value = -value;
    return result;
}

// True when two planes face the same way and lie within tolerance of each other.
inline bool same (const Plane& a, const Plane& b, float tolerance) noexcept
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] > 1.0f - 1.0e-5f && std::abs (a[3] - b[3]) <= tolerance;
}

// The mirror image of p in the plane.
inline Point reflect (const Plane& plane, const Point& p) noexcept
{
    const float d = 2.0f * distance (plane, p);
    return { p[0] - d * plane[0], p[1] - d * plane[1], p[2] - d * plane[2] };
}
} // namespace arbitgpu::realtimeplanarmirror
