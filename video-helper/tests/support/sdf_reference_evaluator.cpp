#include "sdf_reference_evaluator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace videohelper::sdf::reference
{
namespace
{
constexpr double pi = 3.141592653589793238462643383279502884;

Point3 operator+ (Point3 a, Point3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
Point3 operator- (Point3 a, Point3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
Point3 operator* (Point3 a, double b) { return { a.x * b, a.y * b, a.z * b }; }

double dot (Point3 a, Point3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
double length (Point3 value) { return std::sqrt (dot (value, value)); }
Point3 cross (Point3 a, Point3 b)
{
    return { a.y * b.z - a.z * b.y,
             a.z * b.x - a.x * b.z,
             a.x * b.y - a.y * b.x };
}
Point3 normalized (Point3 value) { return value * (1.0 / length (value)); }

double& component (Point3& value, int axis)
{
    if (axis == 0) return value.x;
    if (axis == 1) return value.y;
    return value.z;
}

Point3 rotate (Point3 value, Point3 axis, double radians)
{
    axis = normalized (axis);
    const auto cosine = std::cos (radians);
    const auto sine = std::sin (radians);
    return value * cosine + cross (axis, value) * sine
        + axis * (dot (axis, value) * (1.0 - cosine));
}

void rotatePair (double& first, double& second, double radians)
{
    const auto oldFirst = first;
    const auto oldSecond = second;
    const auto cosine = std::cos (radians);
    const auto sine = std::sin (radians);
    first = cosine * oldFirst - sine * oldSecond;
    second = sine * oldFirst + cosine * oldSecond;
}

double centeredModulo (double value, double period)
{
    return value - period * std::floor ((value + 0.5 * period) / period);
}

double boxDistance (Point3 point, Point3 halfExtent)
{
    const Point3 outside { std::abs (point.x) - halfExtent.x,
                           std::abs (point.y) - halfExtent.y,
                           std::abs (point.z) - halfExtent.z };
    const Point3 positive { std::max (outside.x, 0.0),
                            std::max (outside.y, 0.0),
                            std::max (outside.z, 0.0) };
    return length (positive)
        + std::min (std::max (outside.x, std::max (outside.y, outside.z)), 0.0);
}

double smoothUnion (double a, double b, double radius)
{
    const auto h = std::clamp (0.5 + 0.5 * (b - a) / radius, 0.0, 1.0);
    return b * (1.0 - h) + a * h - radius * h * (1.0 - h);
}

class Evaluator final
{
public:
    explicit Evaluator (const AdmittedSdfIr& geometry)
    {
        for (const auto& record : geometry.records()) records.emplace (record.stableId, &record);
    }

    double at (videowire::SdfStableId id, Point3 point) const
    {
        const auto found = records.find (id);
        if (found == records.end()) throw std::logic_error ("admitted SDF record is missing");
        const auto& record = *found->second;
        const auto& p = record.parameters;
        const auto child = [&] (std::size_t index, Point3 query)
        {
            return at (record.inputs[index], query);
        };

        switch (record.operation)
        {
            case videowire::SdfOperation::sphere:
                return length (point) - p[0];
            case videowire::SdfOperation::box:
                return boxDistance (point, { p[0], p[1], p[2] });
            case videowire::SdfOperation::roundedBox:
                return boxDistance (point, { p[0] - p[3], p[1] - p[3], p[2] - p[3] }) - p[3];
            case videowire::SdfOperation::plane:
                return dot (point, normalized ({ p[0], p[1], p[2] })) + p[3];
            case videowire::SdfOperation::torus:
                return std::hypot (std::hypot (point.x, point.z) - p[0], point.y) - p[1];
            case videowire::SdfOperation::capsule:
            {
                const Point3 start { p[0], p[1], p[2] };
                const Point3 segment { p[3] - p[0], p[4] - p[1], p[5] - p[2] };
                const auto relative = point - start;
                const auto along = std::clamp (dot (relative, segment) / dot (segment, segment), 0.0, 1.0);
                return length (relative - segment * along) - p[6];
            }
            case videowire::SdfOperation::cylinder:
            {
                const auto radial = std::hypot (point.x, point.z) - p[0];
                const auto axial = std::abs (point.y) - p[1];
                return std::min (std::max (radial, axial), 0.0)
                    + std::hypot (std::max (radial, 0.0), std::max (axial, 0.0));
            }
            case videowire::SdfOperation::cone:
            {
                const std::array<double, 2> q { std::hypot (point.x, point.z), point.y };
                const std::array<double, 2> base { p[0], -p[1] };
                const std::array<double, 2> side { p[0], -2.0 * p[1] };
                const std::array<double, 2> cap {
                    q[0] - std::min (q[0], q[1] < 0.0 ? p[0] : 0.0),
                    std::abs (q[1]) - p[1]
                };
                const auto projection = std::clamp (
                    ((base[0] - q[0]) * side[0] + (base[1] - q[1]) * side[1])
                        / (side[0] * side[0] + side[1] * side[1]),
                    0.0, 1.0);
                const std::array<double, 2> slope {
                    q[0] - base[0] + side[0] * projection,
                    q[1] - base[1] + side[1] * projection
                };
                const auto squaredCap = cap[0] * cap[0] + cap[1] * cap[1];
                const auto squaredSlope = slope[0] * slope[0] + slope[1] * slope[1];
                const auto sign = slope[0] < 0.0 && cap[1] < 0.0 ? -1.0 : 1.0;
                return sign * std::sqrt (std::min (squaredCap, squaredSlope));
            }
            case videowire::SdfOperation::gyroid:
            {
                const auto x = p[0] * point.x;
                const auto y = p[0] * point.y;
                const auto z = p[0] * point.z;
                const auto field = std::sin (x) * std::cos (z)
                    + std::sin (y) * std::cos (x)
                    + std::sin (z) * std::cos (y);
                return std::abs (field) / p[0] - p[1];
            }
            case videowire::SdfOperation::unionOp:
                return std::min (child (0, point), child (1, point));
            case videowire::SdfOperation::intersection:
                return std::max (child (0, point), child (1, point));
            case videowire::SdfOperation::subtraction:
                return std::max (child (0, point), -child (1, point));
            case videowire::SdfOperation::smoothUnion:
                return smoothUnion (child (0, point), child (1, point), p[0]);
            case videowire::SdfOperation::smoothIntersection:
                return -smoothUnion (-child (0, point), -child (1, point), p[0]);
            case videowire::SdfOperation::smoothSubtraction:
                return -smoothUnion (-child (0, point), child (1, point), p[0]);
            case videowire::SdfOperation::translate:
                return child (0, point - Point3 { p[0], p[1], p[2] });
            case videowire::SdfOperation::rotate:
                return child (0, rotate (point, { p[0], p[1], p[2] }, -p[3]));
            case videowire::SdfOperation::scale:
            {
                const Point3 query { point.x / p[0], point.y / p[1], point.z / p[2] };
                const auto distanceScale = std::min ({ std::abs (p[0]), std::abs (p[1]), std::abs (p[2]) });
                return child (0, query) * distanceScale;
            }
            case videowire::SdfOperation::repeat:
                return child (0, { centeredModulo (point.x, p[0]), centeredModulo (point.y, p[1]),
                                   centeredModulo (point.z, p[2]) });
            case videowire::SdfOperation::polarRepeat:
            {
                const auto axis = static_cast<int> (p[0]);
                const auto u = (axis + 1) % 3;
                const auto v = (axis + 2) % 3;
                const auto radius = std::hypot (component (point, u), component (point, v));
                const auto angle = std::atan2 (component (point, v), component (point, u));
                const auto sector = 2.0 * pi / p[1];
                const auto folded = centeredModulo (angle - p[2], sector);
                component (point, u) = radius * std::cos (folded);
                component (point, v) = radius * std::sin (folded);
                return child (0, point);
            }
            case videowire::SdfOperation::mirror:
                if (p[0] != 0.0) point.x = std::abs (point.x);
                if (p[1] != 0.0) point.y = std::abs (point.y);
                if (p[2] != 0.0) point.z = std::abs (point.z);
                return child (0, point);
            case videowire::SdfOperation::twist:
            {
                const auto axis = static_cast<int> (p[0]);
                const auto u = (axis + 1) % 3;
                const auto v = (axis + 2) % 3;
                rotatePair (component (point, u), component (point, v),
                            -p[1] * component (point, axis));
                return child (0, point);
            }
            case videowire::SdfOperation::bend:
            {
                const auto axis = static_cast<int> (p[0]);
                const auto u = (axis + 1) % 3;
                const auto angle = -p[1] * component (point, axis);
                rotatePair (component (point, u), component (point, axis), angle);
                return child (0, point);
            }
            case videowire::SdfOperation::taper:
            {
                const auto axis = static_cast<int> (p[0]);
                const auto factor = 1.0 - p[1] * component (point, axis);
                component (point, (axis + 1) % 3) *= factor;
                component (point, (axis + 2) % 3) *= factor;
                return child (0, point);
            }
            case videowire::SdfOperation::displacement:
                return child (0, point) + p[0] * std::sin (p[1] * point.x)
                    * std::sin (p[1] * point.y) * std::sin (p[1] * point.z);
            case videowire::SdfOperation::domainWarp:
                return child (0, { point.x + p[0] * std::sin (p[3] * point.y),
                                   point.y + p[1] * std::sin (p[4] * point.z),
                                   point.z + p[2] * std::sin (p[5] * point.x) });
        }
        throw std::logic_error ("admitted SDF operation is unknown");
    }

private:
    std::unordered_map<videowire::SdfStableId, const videowire::SdfRecord*> records;
};
} // namespace

double evaluatePoint (const AdmittedSdfIr& geometry, Point3 point)
{
    return Evaluator (geometry).at (geometry.rootId(), point);
}
} // namespace videohelper::sdf::reference
