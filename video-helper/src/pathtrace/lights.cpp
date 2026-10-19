#include "lights.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <new>
#include <utility>

namespace videohelper::pathtrace
{
namespace
{

using HarmonicMIDI::grid::SceneLightKind;
using HarmonicMIDI::grid::SceneLightRecord;
using HarmonicMIDI::grid::SceneVec3;

constexpr float kTwoPi = 6.28318530717958647692f;
constexpr float kFourPi = 12.5663706143591729539f;
constexpr float kDegrees = 0.01745329251994329577f;

Vec3 toVec3 (const SceneVec3& v) noexcept { return { v.x, v.y, v.z }; }

float average (Vec3 v) noexcept { return (v.x + v.y + v.z) * (1.0f / 3.0f); }

float maxComponent (Vec3 v) noexcept { return std::max (v.x, std::max (v.y, v.z)); }

Vec3 nonNegative (Vec3 v) noexcept
{
    return { std::max (v.x, 0.0f), std::max (v.y, 0.0f), std::max (v.z, 0.0f) };
}

float clampUnit (float x) noexcept { return std::min (1.0f, std::max (0.0f, x)); }

float finiteOrZero (float x) noexcept { return std::isfinite (x) ? x : 0.0f; }

bool normalizedRotation (const SceneLightRecord& record, Quat& q) noexcept
{
    const auto& r = record.transform.rotation;
    const float norm = std::sqrt (r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w);
    if (! std::isfinite (norm) || ! (norm > 0.0f))
        return false;
    q = { r.x / norm, r.y / norm, r.z / norm, r.w / norm };
    return true;
}

// GLSL smoothstep(cosOuter, cosInner, cosAngle), as the raster spot cone.
float spotCone (float cosAngle, float cosOuter, float cosInner) noexcept
{
    if (cosInner > cosOuter)
    {
        const float t = clampUnit ((cosAngle - cosOuter) / (cosInner - cosOuter));
        return t * t * (3.0f - 2.0f * t);
    }
    return cosAngle >= cosOuter ? 1.0f : 0.0f;
}

bool normalizedQuat (Quat in, Quat& q) noexcept
{
    const float norm = std::sqrt (in.x * in.x + in.y * in.y + in.z * in.z + in.w * in.w);
    if (! std::isfinite (norm) || ! (norm > 0.0f))
        return false;
    q = { in.x / norm, in.y / norm, in.z / norm, in.w / norm };
    return true;
}

// Distance falloff, range window and spot cone of a point or spot light, at
// squared distance d2 from its centre in unit direction fromLight.
float pointFalloff (const Light& light, float d2, Vec3 fromLight) noexcept
{
    float falloff = 1.0f / std::max (d2, 1.0f);
    if (light.range > 0.0f)
    {
        const float window = clampUnit (1.0f - std::sqrt (d2) / light.range);
        falloff *= window * window;
    }
    if (light.type == LightType::Spot)
        falloff *= spotCone (dot (light.direction, fromLight), light.cosOuter, light.cosInner);
    return falloff;
}

// 1 - cos of the half angle a sphere of radius r subtends at squared distance
// d2 > r^2, written so that small spheres keep their precision.
float sphereConeOneMinusCos (float r, float d2) noexcept
{
    const float sin2 = std::min (1.0f, r * r / d2);
    return sin2 / (1.0f + std::sqrt (std::max (0.0f, 1.0f - sin2)));
}

bool outsideSphere (const Light& light, float d2) noexcept
{
    return std::isfinite (d2) && d2 > light.radius * light.radius * (1.0f + 1.0e-4f);
}

// Radiance of a sphere light seen from squared distance d2 in unit direction
// fromLight (centre to viewer), chosen so it delivers the point light's
// irradiance.
Vec3 sphereRadiance (const Light& light, float d2, Vec3 fromLight) noexcept
{
    return light.strength * (pointFalloff (light, d2, fromLight) * d2 / (kPi * light.radius * light.radius));
}

// Planar lights emit on the side their normal points to, or both.
bool planarLightFaces (const Light& light, Vec3 p) noexcept
{
    const Vec3 onPlane = light.type == LightType::Rect ? light.vertex0 : light.position;
    return light.twoSided || dot (p - onPlane, light.normal) > 0.0f;
}

// Point on the unit disk by the concentric mapping (Shirley and Chiu 1997).
Vec2 concentricDisk (Vec2 u) noexcept
{
    const float ox = 2.0f * u.x - 1.0f;
    const float oy = 2.0f * u.y - 1.0f;
    if (ox == 0.0f && oy == 0.0f)
        return {};
    float r = 0.0f;
    float theta = 0.0f;
    if (std::fabs (ox) > std::fabs (oy))
    {
        r = ox;
        theta = 0.25f * kPi * (oy / ox);
    }
    else
    {
        r = oy;
        theta = 0.5f * kPi - 0.25f * kPi * (ox / oy);
    }
    return { r * std::cos (theta), r * std::sin (theta) };
}

// Below this solid angle a rect is sampled by area: the spherical
// construction loses its precision to cancellation.
constexpr float kRectMinSolidAngle = 1.0e-4f;

// A rect as seen from one point, in the rect's frame with the point at the
// origin and the rect in the plane z = z0 < 0 (Urena et al. 2013, section 3).
struct SphericalRect
{
    Vec3 origin {};
    Vec3 axisX {};
    Vec3 axisY {};
    Vec3 axisZ {};
    float x0 = 0.0f;
    float x1 = 0.0f;
    float y0 = 0.0f;
    float y1 = 0.0f;
    float z0 = 0.0f;
    float b0 = 0.0f;
    float b1 = 0.0f;
    float k = 0.0f;
    float solidAngle = 0.0f;
};

float angleBetweenPlanes (Vec3 a, Vec3 b) noexcept
{
    return std::acos (std::min (1.0f, std::max (-1.0f, -dot (a, b))));
}

// False when the point is in the rect's plane or the solid angle is too
// small to sample this way.
bool sphericalRectFrom (const Light& light, Vec3 p, SphericalRect& out) noexcept
{
    const float lengthX = length (light.edge1);
    const float lengthY = length (light.edge2);
    if (! (lengthX > 0.0f) || ! (lengthY > 0.0f))
        return false;
    out.origin = p;
    out.axisX = light.edge1 * (1.0f / lengthX);
    out.axisY = light.edge2 * (1.0f / lengthY);
    out.axisZ = cross (out.axisX, out.axisY);
    const Vec3 toCorner = light.vertex0 - p;
    out.z0 = dot (toCorner, out.axisZ);
    if (out.z0 > 0.0f)
    {
        out.axisZ = -out.axisZ;
        out.z0 = -out.z0;
    }
    if (! (out.z0 < 0.0f))
        return false;
    out.x0 = dot (toCorner, out.axisX);
    out.y0 = dot (toCorner, out.axisY);
    out.x1 = out.x0 + lengthX;
    out.y1 = out.y0 + lengthY;
    // Inward normals of the four great-circle edges of the projected rect.
    const Vec3 n0 = normalizeOrZero ({ 0.0f, out.z0, -out.y0 });
    const Vec3 n1 = normalizeOrZero ({ -out.z0, 0.0f, out.x1 });
    const Vec3 n2 = normalizeOrZero ({ 0.0f, -out.z0, out.y1 });
    const Vec3 n3 = normalizeOrZero ({ out.z0, 0.0f, -out.x0 });
    const float g0 = angleBetweenPlanes (n0, n1);
    const float g1 = angleBetweenPlanes (n1, n2);
    const float g2 = angleBetweenPlanes (n2, n3);
    const float g3 = angleBetweenPlanes (n3, n0);
    out.b0 = n0.z;
    out.b1 = n2.z;
    out.k = 2.0f * kPi - g2 - g3;
    out.solidAngle = g0 + g1 - out.k;
    return std::isfinite (out.solidAngle) && out.solidAngle >= kRectMinSolidAngle;
}

// Point on the rect for u, uniform in the solid angle it subtends.
bool sampleSphericalRect (const SphericalRect& rect, Vec2 u, Vec3& point) noexcept
{
    const float au = u.x * rect.solidAngle + rect.k;
    const float sinAu = std::sin (au);
    const float fu = (std::cos (au) * rect.b0 - rect.b1) / sinAu;
    float cu = (fu > 0.0f ? 1.0f : -1.0f) / std::sqrt (fu * fu + rect.b0 * rect.b0);
    if (! std::isfinite (cu))
        cu = 0.0f;
    cu = std::min (1.0f, std::max (-1.0f, cu));
    float xu = -(cu * rect.z0) / std::sqrt (std::max (1.0f - cu * cu, 1.0e-12f));
    xu = std::min (rect.x1, std::max (rect.x0, finiteOrZero (xu)));
    const float distance = std::sqrt (xu * xu + rect.z0 * rect.z0);
    const float h0 = rect.y0 / std::sqrt (distance * distance + rect.y0 * rect.y0);
    const float h1 = rect.y1 / std::sqrt (distance * distance + rect.y1 * rect.y1);
    const float hv = h0 + u.y * (h1 - h0);
    const float hv2 = hv * hv;
    const float yv = hv2 < 1.0f - 1.0e-6f ? hv * distance / std::sqrt (1.0f - hv2) : rect.y1;
    point = rect.origin + rect.axisX * xu + rect.axisY * std::min (rect.y1, std::max (rect.y0, yv)) + rect.axisZ * rect.z0;
    return isFinite (point);
}

bool sameAsset (const visualenvironmentimage::Identity& a, const visualenvironmentimage::Identity& b) noexcept
{
    return a.id == b.id && a.version == b.version && a.contentSha256 == b.contentSha256;
}

// The map for an image record, or null when the record, its pixels or the
// binding is unusable. The radiance bound is the raster's
// (fixture_environment_image.h validBinding), so both refuse the same scenes.
std::shared_ptr<const EnvironmentMap> bindEnvironmentImage (const SceneLightRecord& record,
                                                            const EnvironmentPixels* pixels)
{
    if (pixels == nullptr || ! pixels->valid() || ! HarmonicMIDI::grid::validSceneEnvironmentImage (record)
        || ! sameAsset (pixels->asset, record.environmentImage.asset))
        return {};
    try
    {
        auto map = std::make_shared<EnvironmentMap>();
        if (! map->build (*pixels, environmentImageTint (record), environmentImageOrientationRadians (record)))
            return {};
        const double scale = std::exp2 (static_cast<double> (record.environmentImage.exposureStops)) * record.intensity;
        const double tint[] { record.color.x * scale, record.color.y * scale, record.color.z * scale };
        const Vec3 peak = map->maximumTexel();
        for (int c = 0; c < 3; ++c)
            if (! std::isfinite (tint[c]) || tint[c] < 0.0
                || tint[c] * static_cast<double> (peak[c]) > visualenvironmentimage::kMaxRadiance)
                return {};
        return map;
    }
    catch (const std::bad_alloc&)
    {
        return {};
    }
}

} // namespace

bool EnvironmentField::active() const noexcept
{
    if (maxComponent (constant) > 0.0f || (image != nullptr && image->active()))
        return true;
    return analytic && (maxComponent (zenith) > 0.0f || maxComponent (horizon) > 0.0f
                        || maxComponent (ground) > 0.0f || maxComponent (sunRadiance) > 0.0f);
}

Vec3 EnvironmentField::radiance (Vec3 direction) const noexcept
{
    Vec3 result = constant;
    if (image != nullptr)
        return result + image->radiance (direction);
    if (! analytic)
        return result;
    const float y = direction.y;
    const Vec3 pole = y >= 0.0f ? zenith : ground;
    result = result + horizon + (pole - horizon) * std::fabs (y);
    const float capWidth = 1.0f - sunCosRadius;
    if (capWidth > 1.0e-7f && maxComponent (sunRadiance) > 0.0f)
    {
        const float c = clampUnit ((dot (direction, sunDirection) - sunCosRadius) / capWidth);
        result = result + sunRadiance * (c * c * (3.0f - 2.0f * c));
    }
    return result;
}

float EnvironmentField::pdf (Vec3 direction) const noexcept
{
    if (image != nullptr)
        return (1.0f - imageSampleProbability) / kFourPi + imageSampleProbability * image->pdf (direction);
    float density = (1.0f - sunSampleProbability) / kFourPi;
    if (sunSampleProbability > 0.0f && dot (direction, sunDirection) >= sunCosRadius)
        density += sunSampleProbability / (kTwoPi * (1.0f - sunCosRadius));
    return density;
}

Vec3 EnvironmentField::sample (Vec2 u, float& pdfOut) const noexcept
{
    Vec3 direction;
    if (image != nullptr && u.x < imageSampleProbability)
    {
        float imagePdf = 0.0f;
        direction = image->sample ({ u.x / imageSampleProbability, u.y }, imagePdf);
        if (! (imagePdf > 0.0f))
        {
            pdfOut = 0.0f;
            return {};
        }
    }
    else if (image != nullptr)
    {
        const float v = (u.x - imageSampleProbability) / (1.0f - imageSampleProbability);
        const float z = 1.0f - 2.0f * std::min (v, 1.0f);
        const float r = std::sqrt (std::max (0.0f, 1.0f - z * z));
        const float phi = kTwoPi * u.y;
        direction = { r * std::cos (phi), r * std::sin (phi), z };
    }
    else if (u.x < sunSampleProbability)
    {
        const float v = u.x / sunSampleProbability;
        const float cosTheta = 1.0f - v * (1.0f - sunCosRadius);
        const float sinTheta = std::sqrt (std::max (0.0f, 1.0f - cosTheta * cosTheta));
        const float phi = kTwoPi * u.y;
        direction = Frame::fromNormal (sunDirection)
                        .toWorld ({ sinTheta * std::cos (phi), sinTheta * std::sin (phi), cosTheta });
    }
    else
    {
        const float v = (u.x - sunSampleProbability) / (1.0f - sunSampleProbability);
        const float z = 1.0f - 2.0f * std::min (v, 1.0f);
        const float r = std::sqrt (std::max (0.0f, 1.0f - z * z));
        const float phi = kTwoPi * u.y;
        direction = { r * std::cos (phi), r * std::sin (phi), z };
    }
    direction = normalizeOrZero (direction);
    pdfOut = pdf (direction);
    return direction;
}

void LightSet::clear() noexcept
{
    lights_.clear();
    cdf_.clear();
    probability_.clear();
    triangleLight_.clear();
    analytic_.clear();
    environment_ = {};
    environmentLight_ = -1;
    stats_ = {};
}

void LightSet::build (const SceneLightRecord* records, std::size_t count, Vec3 ambient,
                      const IngestedScene& geometry, const std::vector<Material>& materials,
                      const Material& fallback, const EnvironmentPixels* environmentImage,
                      const std::vector<LightShape>* shapes, const std::vector<AreaLightRecord>* areaLights)
{
    clear();
    std::vector<double> weights;

    float radius = 1.0f;
    if (! geometry.bounds.empty())
    {
        const float half = 0.5f * length (geometry.bounds.extent());
        if (std::isfinite (half) && half > 1.0e-3f)
            radius = half;
    }
    const float sceneDisk = kPi * radius * radius;

    const auto push = [&] (const Light& light, double weight) {
        if (! (weight > 0.0) || ! std::isfinite (weight))
        {
            ++stats_.zeroPowerLights;
            return;
        }
        lights_.push_back (light);
        weights.push_back (weight);
    };

    bool haveAnalytic = false;
    std::shared_ptr<const EnvironmentMap> image;
    for (std::size_t i = 0; records != nullptr && i < count; ++i)
    {
        const SceneLightRecord& record = records[i];
        const Vec3 color = toVec3 (record.color);
        if (! isFinite (color) || ! std::isfinite (record.intensity))
        {
            ++stats_.invalidLights;
            continue;
        }
        const Vec3 strength = nonNegative (color * record.intensity);
        const LightShape shape = shapes != nullptr && i < shapes->size() ? (*shapes)[i] : LightShape {};
        Quat q;
        const bool rotationValid = normalizedRotation (record, q);
        Light light;
        switch (record.kind)
        {
            case SceneLightKind::Directional:
            {
                const Vec3 toLight = normalizeOrZero (rotate (q, { 0.0f, 0.0f, 1.0f }));
                if (! rotationValid || dot (toLight, toLight) == 0.0f)
                {
                    ++stats_.invalidLights;
                    break;
                }
                light.type = LightType::Directional;
                light.strength = strength;
                light.direction = toLight;
                if (std::isfinite (shape.angularRadius) && shape.angularRadius > 1.0e-4f)
                {
                    const float a = std::min (shape.angularRadius, 1.5f);
                    const float halfSine = std::sin (0.5f * a);
                    const float sine = std::sin (a);
                    light.coneCos = std::cos (a);
                    light.coneOneMinusCos = 2.0f * halfSine * halfSine;
                    light.coneRadianceScale = 1.0f / (kPi * sine * sine);
                    ++stats_.sizedLights;
                }
                ++stats_.directional;
                push (light, static_cast<double> (average (strength)) * sceneDisk);
                break;
            }
            case SceneLightKind::Point:
            case SceneLightKind::Spot:
            {
                const Vec3 position = toVec3 (record.transform.translation);
                const bool spot = record.kind == SceneLightKind::Spot;
                const Vec3 axis = normalizeOrZero (rotate (q, { 0.0f, 0.0f, -1.0f }));
                if (! isFinite (position) || ! std::isfinite (record.range)
                    || (spot && (! rotationValid || dot (axis, axis) == 0.0f
                                 || ! std::isfinite (record.innerConeAngle)
                                 || ! std::isfinite (record.outerConeAngle))))
                {
                    ++stats_.invalidLights;
                    break;
                }
                light.type = spot ? LightType::Spot : LightType::Point;
                light.strength = strength;
                light.position = position;
                light.range = std::max (record.range, 0.0f);
                if (std::isfinite (shape.radius) && shape.radius > 1.0e-6f)
                {
                    light.radius = shape.radius;
                    ++stats_.sizedLights;
                }
                double solidAngle = 4.0 * kPi;
                if (spot)
                {
                    light.direction = axis;
                    light.cosInner = std::cos (record.innerConeAngle);
                    light.cosOuter = std::cos (record.outerConeAngle);
                    solidAngle = std::max (2.0 * kPi * (1.0 - static_cast<double> (light.cosOuter)), 1.0e-4);
                    ++stats_.spot;
                }
                else
                {
                    ++stats_.point;
                }
                push (light, static_cast<double> (average (strength)) * solidAngle);
                break;
            }
            case SceneLightKind::Environment:
            {
                if (record.environmentImage.version != 0)
                {
                    std::shared_ptr<const EnvironmentMap> bound;
                    if (image == nullptr)
                        bound = bindEnvironmentImage (record, environmentImage);
                    if (bound == nullptr)
                    {
                        ++stats_.ignoredEnvironmentImages;
                        break;
                    }
                    image = std::move (bound);
                    ++stats_.environmentImages;
                    break;
                }
                if (record.sky.version == 0)
                {
                    environment_.constant = environment_.constant + strength;
                    ++stats_.environment;
                    break;
                }
                const auto& sky = record.sky;
                const Vec3 zenith = toVec3 (sky.zenith) * strength;
                const Vec3 horizon = toVec3 (sky.horizon) * strength;
                const Vec3 ground = toVec3 (sky.ground) * strength;
                const Vec3 sun = toVec3 (sky.sunColor) * strength * sky.sunIntensity;
                const float azimuth = (sky.sunAzimuthDegrees + sky.orientationDegrees) * kDegrees;
                const float elevation = sky.sunElevationDegrees * kDegrees;
                const Vec3 sunDirection = normalizeOrZero ({ std::sin (azimuth) * std::cos (elevation), std::sin (elevation),
                                                             std::cos (azimuth) * std::cos (elevation) });
                const float sunCos = std::cos (sky.sunRadiusDegrees * kDegrees);
                if (haveAnalytic || ! isFinite (zenith) || ! isFinite (horizon) || ! isFinite (ground)
                    || ! isFinite (sun) || dot (sunDirection, sunDirection) == 0.0f || ! std::isfinite (sunCos))
                {
                    ++stats_.invalidLights;
                    break;
                }
                haveAnalytic = true;
                environment_.analytic = true;
                environment_.zenith = nonNegative (zenith);
                environment_.horizon = nonNegative (horizon);
                environment_.ground = nonNegative (ground);
                environment_.sunRadiance = 1.0f - sunCos > 1.0e-7f ? nonNegative (sun) : Vec3 {};
                environment_.sunDirection = sunDirection;
                environment_.sunCosRadius = std::min (sunCos, 1.0f);
                ++stats_.environment;
                break;
            }
        }
    }

    // An image owns the field: the records' constant and analytic terms go.
    if (image != nullptr)
    {
        stats_.supersededEnvironments = stats_.environment;
        stats_.environment = 0;
        environment_ = {};
        environment_.image = image;
    }

    if (isFinite (ambient))
        environment_.constant = environment_.constant + nonNegative (ambient);

    if (environment_.active())
    {
        Vec3 meanSky = environment_.constant;
        if (environment_.analytic)
            meanSky = meanSky + (environment_.horizon * 2.0f + environment_.zenith + environment_.ground) * 0.25f;
        double skyIrradiance = kPi * static_cast<double> (average (meanSky));
        if (environment_.image != nullptr)
        {
            // Mean radiance over the sphere is the integral over 4 pi.
            const double imageIntegral = static_cast<double> (average (environment_.image->integratedRadiance()));
            const double constantIntegral = static_cast<double> (average (environment_.constant)) * kFourPi;
            skyIrradiance += kPi * imageIntegral / kFourPi;
            float share = 1.0f;
            if (constantIntegral > 0.0)
                share = static_cast<float> (std::min (0.95, std::max (0.05, imageIntegral / (imageIntegral + constantIntegral))));
            environment_.imageSampleProbability = share;
        }
        // The smooth cap averages about half its peak over its solid angle.
        const double sunIrradiance = static_cast<double> (average (environment_.sunRadiance))
                                   * kTwoPi * (1.0 - static_cast<double> (environment_.sunCosRadius)) * 0.5;
        if (sunIrradiance > 0.0)
        {
            const double share = sunIrradiance / (sunIrradiance + skyIrradiance);
            environment_.sunSampleProbability = static_cast<float> (std::min (0.95, std::max (0.05, share)));
        }
        Light light;
        light.type = LightType::Environment;
        const std::size_t before = lights_.size();
        push (light, (skyIrradiance + sunIrradiance) * sceneDisk);
        if (lights_.size() > before)
            environmentLight_ = static_cast<std::int32_t> (lights_.size() - 1);
    }

    const std::size_t triangleCount = geometry.triangles.size();
    triangleLight_.assign (triangleCount, -1);
    for (std::size_t t = 0; t < triangleCount && 3 * t + 2 < geometry.positions.size(); ++t)
    {
        const std::uint32_t materialIndex = geometry.triangles[t].materialIndex;
        const Material& material = materialIndex < materials.size() ? materials[materialIndex] : fallback;
        const Vec3 emitted = material.emission * materialCoverage (material);
        if (! isFinite (emitted))
        {
            ++stats_.invalidLights;
            continue;
        }
        if (! (maxComponent (emitted) > 0.0f))
            continue;
        const Vec3 p0 = geometry.positions[3 * t];
        const Vec3 e1 = geometry.positions[3 * t + 1] - p0;
        const Vec3 e2 = geometry.positions[3 * t + 2] - p0;
        const Vec3 n = cross (e1, e2);
        const float area = 0.5f * length (n);
        if (! std::isfinite (area) || ! (area > 0.0f))
            continue;
        Light light;
        light.type = LightType::Triangle;
        light.strength = nonNegative (emitted);
        light.triangle = static_cast<std::uint32_t> (t);
        light.vertex0 = p0;
        light.edge1 = e1;
        light.edge2 = e2;
        light.normal = normalizeOrZero (n);
        light.area = area;
        const std::size_t before = lights_.size();
        // Emission leaves both faces: pi * L * A per face.
        push (light, 2.0 * kPi * static_cast<double> (average (light.strength)) * area);
        if (lights_.size() > before)
        {
            triangleLight_[t] = static_cast<std::int32_t> (lights_.size() - 1);
            ++stats_.emissiveTriangles;
        }
    }

    for (std::size_t a = 0; areaLights != nullptr && a < areaLights->size(); ++a)
    {
        const AreaLightRecord& record = (*areaLights)[a];
        const Vec3 emitted = record.color * record.intensity;
        Quat q;
        if (! isFinite (emitted) || ! isFinite (record.position) || ! normalizedQuat (record.rotation, q)
            || ! std::isfinite (record.sizeX) || ! std::isfinite (record.sizeY) || ! (record.sizeX > 0.0f)
            || ! (record.sizeY > 0.0f))
        {
            ++stats_.invalidLights;
            continue;
        }
        const Vec3 axisX = normalizeOrZero (rotate (q, { 1.0f, 0.0f, 0.0f }));
        const Vec3 axisY = normalizeOrZero (rotate (q, { 0.0f, 1.0f, 0.0f }));
        Light light;
        light.strength = nonNegative (emitted);
        light.normal = normalizeOrZero (rotate (q, { 0.0f, 0.0f, -1.0f }));
        light.twoSided = record.twoSided;
        if (record.shape == AreaLightShape::Rect)
        {
            light.type = LightType::Rect;
            light.edge1 = axisX * record.sizeX;
            light.edge2 = axisY * record.sizeY;
            light.vertex0 = record.position - light.edge1 * 0.5f - light.edge2 * 0.5f;
            light.area = record.sizeX * record.sizeY;
        }
        else
        {
            light.type = LightType::Disk;
            light.position = record.position;
            light.edge1 = axisX * (0.5f * record.sizeX);
            light.edge2 = axisY * (0.5f * record.sizeY);
            light.area = 0.25f * kPi * record.sizeX * record.sizeY;
        }
        if (! std::isfinite (light.area) || ! (light.area > 0.0f))
        {
            ++stats_.invalidLights;
            continue;
        }
        const std::size_t before = lights_.size();
        // pi * L * A per emitting face.
        push (light, (light.twoSided ? 2.0 : 1.0) * kPi * static_cast<double> (average (light.strength))
                         * static_cast<double> (light.area));
        if (lights_.size() > before)
            ++stats_.areaLights;
    }

    double total = 0.0;
    for (const double w : weights)
        total += w;
    if (! (total > 0.0) || ! std::isfinite (total))
    {
        lights_.clear();
        triangleLight_.assign (triangleCount, -1);
        environmentLight_ = -1;
        return;
    }
    for (std::size_t i = 0; i < lights_.size(); ++i)
    {
        const Light& light = lights_[i];
        const bool sized = light.radius > 0.0f || light.coneOneMinusCos > 0.0f || light.type == LightType::Rect
                        || light.type == LightType::Disk;
        if (sized)
            analytic_.push_back (static_cast<std::uint32_t> (i));
    }
    probability_.resize (weights.size());
    cdf_.resize (weights.size());
    double running = 0.0;
    for (std::size_t i = 0; i < weights.size(); ++i)
    {
        probability_[i] = static_cast<float> (weights[i] / total);
        running += weights[i];
        cdf_[i] = static_cast<float> (running / total);
    }
    cdf_.back() = 1.0f;
}

float LightSet::selectionProbability (std::size_t light) const noexcept
{
    return light < probability_.size() ? probability_[light] : 0.0f;
}

bool LightSet::sample (Vec3 p, float uSelect, Vec2 u, LightSample& out) const noexcept
{
    out = {};
    if (lights_.empty())
        return false;
    std::size_t index = static_cast<std::size_t> (std::upper_bound (cdf_.begin(), cdf_.end(), uSelect) - cdf_.begin());
    index = std::min (index, lights_.size() - 1);
    const float selection = probability_[index];
    if (! (selection > 0.0f))
        return false;
    const Light& light = lights_[index];
    out.light = static_cast<std::uint32_t> (index);
    switch (light.type)
    {
        case LightType::Directional:
        {
            out.infinite = true;
            if (light.coneOneMinusCos > 0.0f)
            {
                // Uniform in the sun's cone.
                const float oneMinusCos = u.x * light.coneOneMinusCos;
                const float cosTheta = 1.0f - oneMinusCos;
                const float sinTheta = std::sqrt (std::max (0.0f, oneMinusCos * (2.0f - oneMinusCos)));
                const float phi = kTwoPi * u.y;
                out.wi = normalizeOrZero (Frame::fromNormal (light.direction)
                                              .toWorld ({ sinTheta * std::cos (phi), sinTheta * std::sin (phi), cosTheta }));
                out.radiance = light.strength * light.coneRadianceScale;
                out.pdf = selection / (kTwoPi * light.coneOneMinusCos);
                return dot (out.wi, out.wi) > 0.0f && std::isfinite (out.pdf);
            }
            out.wi = light.direction;
            out.radiance = light.strength;
            out.pdf = selection;
            out.delta = true;
            return true;
        }
        case LightType::Point:
        case LightType::Spot:
        {
            const Vec3 toLight = light.position - p;
            const float d2 = dot (toLight, toLight);
            if (! (d2 > 0.0f) || ! std::isfinite (d2))
                return false;
            const float d = std::sqrt (d2);
            const Vec3 axis = toLight * (1.0f / d);
            if (light.radius > 0.0f && outsideSphere (light, d2))
            {
                // Uniform in the cone the sphere subtends.
                const float coneWidth = sphereConeOneMinusCos (light.radius, d2);
                const float oneMinusCos = u.x * coneWidth;
                const float cosTheta = 1.0f - oneMinusCos;
                const float sinTheta = std::sqrt (std::max (0.0f, oneMinusCos * (2.0f - oneMinusCos)));
                const float phi = kTwoPi * u.y;
                out.wi = normalizeOrZero (Frame::fromNormal (axis)
                                              .toWorld ({ sinTheta * std::cos (phi), sinTheta * std::sin (phi), cosTheta }));
                // Nearest intersection with the sphere along wi.
                const float along = dot (out.wi, toLight);
                const float disc = light.radius * light.radius - std::max (0.0f, d2 - along * along);
                const float t = std::max (0.0f, along - std::sqrt (std::max (0.0f, disc)));
                out.radiance = sphereRadiance (light, d2, -axis);
                out.point = p + out.wi * t;
                out.distance = t;
                out.pdf = selection / (kTwoPi * coneWidth);
                return dot (out.wi, out.wi) > 0.0f && std::isfinite (out.pdf) && out.pdf > 0.0f;
            }
            // A point light, or a shading point inside the sphere.
            out.wi = axis;
            out.radiance = light.strength * pointFalloff (light, d2, -axis);
            out.point = light.position;
            out.distance = d;
            out.pdf = selection;
            out.delta = true;
            return true;
        }
        case LightType::Environment:
        {
            float directionPdf = 0.0f;
            out.wi = environment_.sample (u, directionPdf);
            if (! (directionPdf > 0.0f) || dot (out.wi, out.wi) == 0.0f)
                return false;
            out.radiance = environment_.radiance (out.wi);
            out.pdf = selection * directionPdf;
            out.infinite = true;
            return true;
        }
        case LightType::Triangle:
        {
            // Uniform point by area.
            const float root = std::sqrt (u.x);
            const float b1 = root * (1.0f - u.y);
            const float b2 = root * u.y;
            const Vec3 q = light.vertex0 + light.edge1 * b1 + light.edge2 * b2;
            const Vec3 toLight = q - p;
            const float d2 = dot (toLight, toLight);
            if (! (d2 > 0.0f) || ! std::isfinite (d2))
                return false;
            const float d = std::sqrt (d2);
            out.wi = toLight * (1.0f / d);
            const float cosAtLight = std::fabs (dot (light.normal, out.wi));
            if (! (cosAtLight > 1.0e-6f))
                return false;
            out.radiance = light.strength;
            out.point = q;
            out.lightNormal = light.normal;
            out.distance = d;
            out.pdf = selection * d2 / (cosAtLight * light.area);
            out.area = true;
            return std::isfinite (out.pdf) && out.pdf > 0.0f;
        }
        case LightType::Rect:
        case LightType::Disk:
        {
            if (! planarLightFaces (light, p))
                return false;
            Vec3 q;
            float directionPdf = 0.0f;
            SphericalRect rect;
            if (light.type == LightType::Rect && sphericalRectFrom (light, p, rect))
            {
                if (! sampleSphericalRect (rect, u, q))
                    return false;
                directionPdf = 1.0f / rect.solidAngle;
            }
            else if (light.type == LightType::Rect)
            {
                q = light.vertex0 + light.edge1 * u.x + light.edge2 * u.y;
            }
            else
            {
                const Vec2 disk = concentricDisk (u);
                q = light.position + light.edge1 * disk.x + light.edge2 * disk.y;
            }
            const Vec3 toLight = q - p;
            const float d2 = dot (toLight, toLight);
            if (! (d2 > 0.0f) || ! std::isfinite (d2))
                return false;
            const float d = std::sqrt (d2);
            out.wi = toLight * (1.0f / d);
            const float cosAtLight = std::fabs (dot (light.normal, out.wi));
            if (! (cosAtLight > 1.0e-6f))
                return false;
            if (! (directionPdf > 0.0f))
                directionPdf = d2 / (cosAtLight * light.area);
            out.radiance = light.strength;
            out.point = q;
            out.lightNormal = light.normal;
            out.distance = d;
            out.pdf = selection * directionPdf;
            return std::isfinite (out.pdf) && out.pdf > 0.0f;
        }
    }
    return false;
}

bool LightSet::analyticHit (std::size_t index, Vec3 origin, Vec3 direction, float tMax, Vec3& radiance,
                            float& pdf, float& distance) const noexcept
{
    radiance = {};
    pdf = 0.0f;
    distance = std::numeric_limits<float>::infinity();
    if (index >= analytic_.size())
        return false;
    const std::size_t slot = analytic_[index];
    const Light& light = lights_[slot];
    const float selection = probability_[slot];
    switch (light.type)
    {
        case LightType::Directional:
        {
            if (! std::isinf (tMax) || ! (light.coneOneMinusCos > 0.0f) || dot (direction, light.direction) < light.coneCos)
                return false;
            radiance = light.strength * light.coneRadianceScale;
            pdf = selection / (kTwoPi * light.coneOneMinusCos);
            return true;
        }
        case LightType::Point:
        case LightType::Spot:
        {
            const Vec3 toLight = light.position - origin;
            const float d2 = dot (toLight, toLight);
            // From inside, light sampling treats the sphere as a point.
            if (! (light.radius > 0.0f) || ! outsideSphere (light, d2))
                return false;
            const float along = dot (direction, toLight);
            const float disc = light.radius * light.radius - (d2 - along * along);
            if (! (disc >= 0.0f) || ! (along > 0.0f))
                return false;
            const float t = along - std::sqrt (disc);
            if (! (t > 0.0f) || ! (t < tMax))
                return false;
            radiance = sphereRadiance (light, d2, toLight * (-1.0f / std::sqrt (d2)));
            distance = t;
            pdf = selection / (kTwoPi * sphereConeOneMinusCos (light.radius, d2));
            return true;
        }
        case LightType::Rect:
        case LightType::Disk:
        {
            if (! planarLightFaces (light, origin))
                return false;
            const float denominator = dot (direction, light.normal);
            if (! (std::fabs (denominator) > 1.0e-8f))
                return false;
            const Vec3 onPlane = light.type == LightType::Rect ? light.vertex0 : light.position;
            const float t = dot (onPlane - origin, light.normal) / denominator;
            if (! (t > 0.0f) || ! (t < tMax))
                return false;
            const Vec3 offset = origin + direction * t - onPlane;
            const float a = dot (offset, light.edge1) / dot (light.edge1, light.edge1);
            const float b = dot (offset, light.edge2) / dot (light.edge2, light.edge2);
            const bool inside = light.type == LightType::Rect ? a >= 0.0f && a <= 1.0f && b >= 0.0f && b <= 1.0f
                                                              : a * a + b * b <= 1.0f;
            if (! inside)
                return false;
            float directionPdf = 0.0f;
            SphericalRect rect;
            if (light.type == LightType::Rect && sphericalRectFrom (light, origin, rect))
                directionPdf = 1.0f / rect.solidAngle;
            else
                directionPdf = t * t / (std::fabs (denominator) * light.area);
            radiance = light.strength;
            pdf = selection * directionPdf;
            distance = t;
            return std::isfinite (pdf);
        }
        case LightType::Environment:
        case LightType::Triangle:
            return false;
    }
    return false;
}

float LightSet::trianglePdf (std::uint32_t triangle, float distance, float cosAtLight) const noexcept
{
    if (triangle >= triangleLight_.size() || triangleLight_[triangle] < 0 || ! (cosAtLight > 1.0e-6f))
        return 0.0f;
    const auto index = static_cast<std::size_t> (triangleLight_[triangle]);
    return probability_[index] * distance * distance / (cosAtLight * lights_[index].area);
}

float LightSet::environmentPdf (Vec3 direction) const noexcept
{
    if (environmentLight_ < 0)
        return 0.0f;
    return probability_[static_cast<std::size_t> (environmentLight_)] * environment_.pdf (direction);
}

} // namespace videohelper::pathtrace
