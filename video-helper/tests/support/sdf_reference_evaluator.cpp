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

    double at (videowire::SdfStableId id, Point3 point,
               videowire::SdfStableId* contributor = nullptr) const
    {
        const auto found = records.find (id);
        if (found == records.end()) throw std::logic_error ("admitted SDF record is missing");
        const auto& record = *found->second;
        const auto& p = record.parameters;
        const auto child = [&] (std::size_t index, Point3 query)
        {
            return at (record.inputs[index], query, contributor);
        };
        if (contributor != nullptr && record.inputCount == 0) *contributor = id;

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
            case videowire::SdfOperation::intersection:
            case videowire::SdfOperation::subtraction:
            case videowire::SdfOperation::smoothUnion:
            case videowire::SdfOperation::smoothIntersection:
            case videowire::SdfOperation::smoothSubtraction:
            {
                videowire::SdfStableId firstId = 0, secondId = 0;
                const auto a = at (record.inputs[0], point, &firstId);
                const auto b = at (record.inputs[1], point, &secondId);
                const auto op = record.operation;
                const bool minimum = op == videowire::SdfOperation::unionOp
                    || op == videowire::SdfOperation::smoothUnion;
                const bool cut = op == videowire::SdfOperation::subtraction
                    || op == videowire::SdfOperation::smoothSubtraction;
                if (contributor != nullptr)
                {
                    if (op == videowire::SdfOperation::smoothUnion
                        || op == videowire::SdfOperation::smoothIntersection
                        || op == videowire::SdfOperation::smoothSubtraction)
                    {
                        // Calculate the smooth blend weight independently of
                        // the native evaluator's equivalent distance ordering.
                        const auto blendA = minimum ? a : -a;
                        const auto blendB = minimum || cut ? b : -b;
                        const auto firstWeight = std::clamp (
                            0.5 + 0.5 * (blendB - blendA) / p[0], 0.0, 1.0);
                        *contributor = firstWeight >= 0.5 ? firstId : secondId;
                    }
                    else
                        *contributor = (minimum ? a <= b : a >= (cut ? -b : b)) ? firstId : secondId;
                }
                if (op == videowire::SdfOperation::unionOp) return std::min (a, b);
                if (op == videowire::SdfOperation::intersection) return std::max (a, b);
                if (op == videowire::SdfOperation::subtraction) return std::max (a, -b);
                if (op == videowire::SdfOperation::smoothUnion) return smoothUnion (a, b, p[0]);
                if (op == videowire::SdfOperation::smoothIntersection) return -smoothUnion (-a, -b, p[0]);
                return -smoothUnion (-a, b, p[0]);
            }
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

void validateOutputControls (const NativeSdfRenderControls& controls)
{
    if (controls.normalQuality >= arbitgpu::NativeSdfQuality::count
        || controls.shadowQuality >= arbitgpu::NativeSdfQuality::count
        || controls.adaptiveQuality >= arbitgpu::NativeSdfQuality::count
        || controls.output >= arbitgpu::NativeSdfOutput::count
        || controls.maximumSteps == 0 || controls.maximumSteps > 512
        || ! std::isfinite (controls.epsilon) || controls.epsilon < 1.0e-6 || controls.epsilon > 0.1
        || ! std::isfinite (controls.maximumDistance)
        || controls.maximumDistance < controls.epsilon || controls.maximumDistance > 1000.0)
        throw std::invalid_argument ("SDF output oracle requires valid bounded controls");
}

class OutputEvaluator final
{
public:
    OutputEvaluator (const AdmittedSdfIr& geometry, const NativeSdfRenderControls& settings)
        : field (geometry), root (geometry.rootId()), controls (settings) {}

    double distance (Point3 p) const { return field.at (root, p); }
    static double qualityScale (arbitgpu::NativeSdfQuality q)
    {
        constexpr double scales[] { 4.0, 2.0, 1.0, 0.5 };
        return scales[static_cast<unsigned> (q)];
    }
    Point3 normal (Point3 p) const
    {
        const auto e = std::max (controls.epsilon * qualityScale (controls.normalQuality), 1.0e-6);
        Point3 gradient {
            distance (p + Point3 {e,0,0}) - distance (p - Point3 {e,0,0}),
            distance (p + Point3 {0,e,0}) - distance (p - Point3 {0,e,0}),
            distance (p + Point3 {0,0,e}) - distance (p - Point3 {0,0,e}) };
        const auto squared = dot (gradient, gradient);
        return squared > 0.0 && std::isfinite (squared) ? normalized (gradient) : Point3 {0,0,1};
    }
    double curvature (Point3 p) const
    {
        const auto h = std::max ({controls.epsilon * qualityScale (controls.normalQuality) * 4.0,
                                 0.002, length (p) * 0.0001});
        const auto divergence = normal (p + Point3 {h,0,0}).x - normal (p - Point3 {h,0,0}).x
            + normal (p + Point3 {0,h,0}).y - normal (p - Point3 {0,h,0}).y
            + normal (p + Point3 {0,0,h}).z - normal (p - Point3 {0,0,h}).z;
        const auto value = divergence / (4.0 * h);
        return std::isfinite (value) ? std::clamp (value, -1.0/h, 1.0/h) : 0.0;
    }
    double ambient (Point3 p, Point3 n) const
    {
        const auto count = 4u + 4u * static_cast<unsigned> (controls.normalQuality);
        const auto radius = std::min (controls.maximumDistance, std::max (0.5, 32.0*controls.epsilon));
        double occlusion = 0.0, total = 0.0, weight = 1.0;
        for (unsigned i = 0; i < count; ++i)
        {
            const auto reach = radius * (i+1) / count;
            const auto d = distance (p + n * reach);
            if (! std::isfinite (d)) return 0.0;
            occlusion += weight * std::clamp (1.0 - d/reach, 0.0, 1.0);
            total += weight;
            weight *= 0.75;
        }
        return std::clamp (1.0 - occlusion/total, 0.0, 1.0);
    }
    double shadow (Point3 p, Point3 n) const
    {
        const auto light = normalized ({-0.45,0.75,0.6});
        if (dot (n, light) <= 0.0) return 0.0;
        const auto origin = p + n * (controls.epsilon * 4.0);
        const auto limit = 8u << static_cast<unsigned> (controls.shadowQuality);
        double travel = controls.epsilon * 4.0, visibility = 1.0;
        for (unsigned step = 0; step < limit; ++step)
        {
            const auto d = distance (origin + light * travel);
            if (! std::isfinite (d) || d < controls.epsilon) return 0.0;
            visibility = std::min (visibility, 12.0*d/std::max (travel, controls.epsilon));
            travel += std::clamp (d, controls.epsilon * 2.0, 0.25);
            if (travel > std::min (controls.maximumDistance, 8.0)) break;
        }
        return std::clamp (visibility, 0.0, 1.0);
    }
    bool trace (Point3 direction, double& travel) const
    {
        travel = 0.0;
        for (unsigned step = 0; step < controls.maximumSteps && step < 512; ++step)
        {
            if (travel > controls.maximumDistance) break;
            const auto d = distance (Point3 {0,0,3} + direction * travel);
            if (! std::isfinite (d)) break;
            const auto threshold = std::max (controls.epsilon * qualityScale (controls.adaptiveQuality)
                * std::max (1.0, travel * 0.05), 1.0e-6);
            if (d <= threshold) return true;
            travel += d;
        }
        return false;
    }
    double edgeDistance (double u, double v, std::uint32_t height) const
    {
        constexpr std::array<std::array<double, 2>, 8> axes {{
            {{1,0}},{{-1,0}},{{0,1}},{{0,-1}},
            {{0.70710678,0.70710678}},{{-0.70710678,0.70710678}},
            {{0.70710678,-0.70710678}},{{-0.70710678,-0.70710678}} }};
        double nearest = 8.0;
        for (const auto& axis : axes)
        {
            const auto hitAt = [&] (double pixels)
            {
                double travel;
                return trace (normalized ({u + axis[0]*pixels*2.0/height,
                                            v + axis[1]*pixels*2.0/height, -1.8}), travel);
            };
            double low = 0.0, high = 8.0;
            if (hitAt (high)) continue;
            for (unsigned iteration = 0; iteration < 4; ++iteration)
            {
                const auto middle = (low + high) * 0.5;
                if (hitAt (middle)) low = middle;
                else high = middle;
            }
            nearest = std::min (nearest, high);
        }
        return nearest;
    }
    std::array<double, 4> pixel (std::uint32_t width, std::uint32_t height, double x, double y) const
    {
        const auto u = (2.0*(x+0.5) - width)/height;
        const auto v = (2.0*(y+0.5) - height)/height;
        const auto direction = normalized ({u,v,-1.8});
        double travel;
        const bool hit = trace (direction, travel);
        using Output = arbitgpu::NativeSdfOutput;
        const auto gray = [] (double value) { return std::array<double,4> {value,value,value,1.0}; };
        if (! hit)
        {
            if (controls.output >= Output::materialId) return gray (0.0);
            if (controls.output == Output::depth) return gray (1.0);
            return {0.02745,0.03922,0.07059,1.0};
        }
        const auto point = Point3 {0,0,3} + direction * travel;
        if (controls.output == Output::depth) return gray (std::clamp (travel/controls.maximumDistance,0.0,1.0));
        if (controls.output == Output::materialId)
        {
            // Independent 64-bit visualization oracle; no native program or
            // native color helper is consulted here.
            videowire::SdfStableId id = 0;
            field.at (root, point, &id);
            id = (id ^ (id >> 30u)) * UINT64_C(0xbf58476d1ce4e5b9);
            id = (id ^ (id >> 27u)) * UINT64_C(0x94d049bb133111eb);
            const auto color = static_cast<std::uint32_t> (id ^ (id >> 31u)) | 0x00202020u;
            return {(color & 255u)/255.0, ((color >> 8u) & 255u)/255.0,
                    ((color >> 16u) & 255u)/255.0, 1.0};
        }
        if (controls.output == Output::curvature)
        {
            const auto h = curvature (point);
            return gray (0.5 + 0.5*h/(1.0+std::abs (h)));
        }
        if (controls.output == Output::edgeDistance) return gray (edgeDistance (u,v,height)/8.0);
        const auto n = normal (point);
        if (controls.output == Output::normal) return {n.x*0.5+0.5,n.y*0.5+0.5,n.z*0.5+0.5,1.0};
        if (controls.output == Output::ambientOcclusion) return gray (ambient (point,n));
        if (controls.output == Output::softShadow) return gray (shadow (point,n));
        const auto diffuse = std::max (dot (n, normalized ({-0.45,0.75,0.6})),0.0);
        const auto rim = std::pow (1.0-std::max (dot (n,direction*-1.0),0.0),3.0);
        const auto shaded = 0.12+0.88*diffuse*shadow (point,n);
        return {0.12*shaded+0.18*rim,0.42*shaded+0.35*rim,0.88*shaded+0.65*rim,1.0};
    }
private:
    Evaluator field;
    videowire::SdfStableId root;
    const NativeSdfRenderControls& controls;
};
} // namespace

double evaluatePoint (const AdmittedSdfIr& geometry, Point3 point)
{
    return Evaluator (geometry).at (geometry.rootId(), point);
}

PointSample evaluateSample (const AdmittedSdfIr& geometry, Point3 point)
{
    PointSample result;
    result.distance = Evaluator (geometry).at (geometry.rootId(), point, &result.primitiveId);
    return result;
}

SurfaceSample evaluateSurface (const AdmittedSdfIr& geometry, Point3 point,
                               const NativeSdfRenderControls& controls)
{
    validateOutputControls (controls);
    const OutputEvaluator evaluator (geometry, controls);
    const auto normal = evaluator.normal (point);
    return {normal, evaluator.curvature (point), evaluator.ambient (point,normal),
            evaluator.shadow (point,normal)};
}

std::array<double, 4> evaluateOutputPixel (const AdmittedSdfIr& geometry,
                                        const NativeSdfRenderControls& controls,
                                        std::uint32_t width, std::uint32_t height,
                                        double x, double y)
{
    validateOutputControls (controls);
    if (width == 0 || height == 0 || ! std::isfinite (x) || ! std::isfinite (y))
        throw std::invalid_argument ("SDF output oracle requires valid controls and extent");
    return OutputEvaluator (geometry,controls).pixel (width,height,x,y);
}
} // namespace videohelper::sdf::reference
