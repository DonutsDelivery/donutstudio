#include "lights.h"
#include "bvh.h"
#include "sky_cloud_volume.h"

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

#if ! defined(PATHTRACE_DEVICE_METAL)
using HarmonicMIDI::grid::SceneLightKind;
using HarmonicMIDI::grid::SceneLightRecord;
using HarmonicMIDI::grid::SceneVec3;
#endif

constexpr float kLightTwoPi = 6.28318530717958647692f;
constexpr float kFourPi = 12.5663706143591729539f;
constexpr float kDegrees = 0.01745329251994329577f;

#if ! defined(PATHTRACE_DEVICE_METAL)
PT_SHARED_ENTRY Vec3 toVec3 (const SceneVec3& v) noexcept { return { v.x, v.y, v.z }; }
#endif

// A sum past a float divides first, so the brightest light still weighs a finite amount.
PT_SHARED_ENTRY float meanOf (Vec3 v) noexcept
{
    const float sum = v.x + v.y + v.z;
    return std::isfinite (sum) ? sum * (1.0f / 3.0f) : v.x * (1.0f / 3.0f) + v.y * (1.0f / 3.0f) + v.z * (1.0f / 3.0f);
}

PT_SHARED_ENTRY float maxComponent (Vec3 v) noexcept { return std::max (v.x, std::max (v.y, v.z)); }

PT_SHARED_ENTRY Vec3 nonNegative (Vec3 v) noexcept
{
    return { std::max (v.x, 0.0f), std::max (v.y, 0.0f), std::max (v.z, 0.0f) };
}

PT_SHARED_ENTRY float clampLightUnit (float x) noexcept { return std::min (1.0f, std::max (0.0f, x)); }

// The raster's analyticSkyCloudOpacity (fixture_analytic_sky.h), so one sky shows one cloud
// layer on every engine: five octaves of value noise on a flat layer above the camera.
PT_SHARED_ENTRY float cloudFract (float x) noexcept { return x - std::floor (x); }
PT_SHARED_ENTRY float cloudMix (float a, float b, float t) noexcept { return a + (b - a) * t; }
PT_SHARED_ENTRY float cloudSmoothstep (float low, float high, float x) noexcept
{
    const float t = clampLightUnit ((x - low) / (high - low));
    return t * t * (3.0f - 2.0f * t);
}
PT_SHARED_ENTRY float cloudHash (float x, float y) noexcept
{
    float a = cloudFract (x * 0.1031f), b = cloudFract (y * 0.1031f), c = a;
    const float shift = a * (b + 33.33f) + b * (c + 33.33f) + c * (a + 33.33f);
    a += shift;
    b += shift;
    c += shift;
    return cloudFract ((a + b) * c);
}
PT_SHARED_ENTRY float cloudNoise (float x, float y) noexcept
{
    const float cellX = std::floor (x), cellY = std::floor (y);
    float fx = x - cellX, fy = y - cellY;
    fx = fx * fx * (3.0f - 2.0f * fx);
    fy = fy * fy * (3.0f - 2.0f * fy);
    return cloudMix (cloudMix (cloudHash (cellX, cellY), cloudHash (cellX + 1.0f, cellY), fx),
                     cloudMix (cloudHash (cellX, cellY + 1.0f), cloudHash (cellX + 1.0f, cellY + 1.0f), fx), fy);
}
PT_SHARED_ENTRY float cloudOpacity (const EnvironmentField& field, Vec3 d) noexcept
{
    if (field.cloudCoverage <= 0.0f || d.y <= 0.0f)
        return 0.0f;
    const float alongX = std::sin (field.cloudHeadingRadians), alongZ = std::cos (field.cloudHeadingRadians);
    const float px = d.x / (d.y + 0.1f), pz = d.z / (d.y + 0.1f);
    float qx = (px * alongX + pz * alongZ - field.cloudDrift) * cloudMix (1.6f, 0.35f, field.cloudKind);
    float qy = (px * alongZ - pz * alongX) * cloudMix (1.6f, 2.8f, field.cloudKind);
    float value = 0.0f, amplitude = 0.5f;
    for (int octave = 0; octave < 5; ++octave)
    {
        value += amplitude * cloudNoise (qx, qy);
        qx = qx * 2.03f + 17.0f;
        qy = qy * 2.03f + 9.0f;
        amplitude *= 0.5f;
    }
    const float edge = cloudMix (0.12f, 0.3f, field.cloudKind);
    const float threshold = cloudMix (0.85f, -edge, field.cloudCoverage);
    return cloudSmoothstep (threshold, threshold + edge, value) * cloudMix (0.96f, 0.7f, field.cloudKind)
        * cloudSmoothstep (0.015f, 0.14f, d.y);
}

PT_SHARED_ENTRY float finiteOrZero (float x) noexcept { return std::isfinite (x) ? x : 0.0f; }

// Host only: the Metal kernel (device_metal.metal) leaves this out.
#if ! defined(PATHTRACE_DEVICE_METAL)
// A float holds any radiance short of overflow, the one bound authored light has. A product past
// it keeps its hue at the largest float instead of going infinite and dropping the light.
Vec3 finiteProduct (const Vec3& a, const Vec3& b, float gain) noexcept
{
    const double x = static_cast<double> (a.x) * b.x * gain, y = static_cast<double> (a.y) * b.y * gain,
                 z = static_cast<double> (a.z) * b.z * gain;
    const double peak = std::max ({ x, y, z }), limit = std::numeric_limits<float>::max();
    const double fit = peak > limit ? limit / peak : 1.0;
    return { static_cast<float> (x * fit), static_cast<float> (y * fit), static_cast<float> (z * fit) };
}
#endif

// Host only: the Metal kernel (device_metal.metal) leaves this out.
#if ! defined(PATHTRACE_DEVICE_METAL)
PT_SHARED_ENTRY bool normalizedRotation (const SceneLightRecord& record, Quat& q) noexcept
{
    const auto& r = record.transform.rotation;
    const float norm = std::sqrt (r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w);
    if (! std::isfinite (norm) || ! (norm > 0.0f))
        return false;
    q = { r.x / norm, r.y / norm, r.z / norm, r.w / norm };
    return true;
}
#endif

// GLSL smoothstep(cosOuter, cosInner, cosAngle), as the raster spot cone.
PT_SHARED_ENTRY float spotCone (float cosAngle, float cosOuter, float cosInner) noexcept
{
    if (cosInner > cosOuter)
    {
        const float t = clampLightUnit ((cosAngle - cosOuter) / (cosInner - cosOuter));
        return t * t * (3.0f - 2.0f * t);
    }
    return cosAngle >= cosOuter ? 1.0f : 0.0f;
}

PT_SHARED_ENTRY bool normalizedQuat (Quat in, Quat& q) noexcept
{
    const float norm = std::sqrt (in.x * in.x + in.y * in.y + in.z * in.z + in.w * in.w);
    if (! std::isfinite (norm) || ! (norm > 0.0f))
        return false;
    q = { in.x / norm, in.y / norm, in.z / norm, in.w / norm };
    return true;
}

// Distance falloff, range window and spot cone of a point or spot light, at
// squared distance d2 from its centre in unit direction fromLight.
PT_SHARED_ENTRY float pointFalloff (const PT_GLOBAL Light& light, float d2, Vec3 fromLight) noexcept
{
    float falloff = 1.0f / std::max (d2, 1.0f);
    if (light.range > 0.0f)
    {
        const float window = clampLightUnit (1.0f - std::sqrt (d2) / light.range);
        falloff *= window * window;
    }
    if (light.type == LightType::Spot)
        falloff *= spotCone (dot (light.direction, fromLight), light.cosOuter, light.cosInner);
    return falloff;
}

// 1 - cos of the half angle a sphere of radius r subtends at squared distance
// d2 > r^2, written so that small spheres keep their precision.
PT_SHARED_ENTRY float sphereConeOneMinusCos (float r, float d2) noexcept
{
    const float sin2 = std::min (1.0f, r * r / d2);
    return sin2 / (1.0f + std::sqrt (std::max (0.0f, 1.0f - sin2)));
}

PT_SHARED_ENTRY bool outsideSphere (const PT_GLOBAL Light& light, float d2) noexcept
{
    return std::isfinite (d2) && d2 > light.radius * light.radius * (1.0f + 1.0e-4f);
}

// Radiance of a sphere light seen from squared distance d2 in unit direction
// fromLight (centre to viewer), chosen so it delivers the point light's
// irradiance.
PT_SHARED_ENTRY Vec3 sphereRadiance (const PT_GLOBAL Light& light, float d2, Vec3 fromLight) noexcept
{
    return light.strength * (pointFalloff (light, d2, fromLight) * d2 / (kPi * light.radius * light.radius));
}

// Planar lights emit on the side their normal points to, or both.
PT_SHARED_ENTRY bool planarLightFaces (const PT_GLOBAL Light& light, Vec3 p) noexcept
{
    const Vec3 onPlane = light.type == LightType::Rect ? light.vertex0 : light.position;
    return light.twoSided || dot (p - onPlane, light.normal) > 0.0f;
}

// Point on the unit disk by the concentric mapping (Shirley and Chiu 1997).
PT_SHARED_ENTRY Vec2 concentricDisk (Vec2 u) noexcept
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

PT_SHARED_ENTRY float angleBetweenPlanes (Vec3 a, Vec3 b) noexcept
{
    return std::acos (std::min (1.0f, std::max (-1.0f, -dot (a, b))));
}

// False when the point is in the rect's plane or the solid angle is too
// small to sample this way.
PT_SHARED_ENTRY bool sphericalRectFrom (const PT_GLOBAL Light& light, Vec3 p, SphericalRect& out) noexcept
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
PT_SHARED_ENTRY bool sampleSphericalRect (const SphericalRect& rect, Vec2 u, Vec3& point) noexcept
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

// Host only: the Metal kernel (device_metal.metal) leaves this out.
#if ! defined(PATHTRACE_DEVICE_METAL)
bool sameAsset (const visualenvironmentimage::Identity& a, const visualenvironmentimage::Identity& b) noexcept
{
    return a.id == b.id && a.version == b.version && a.contentSha256 == b.contentSha256;
}

// The map for an image record, or null when the record, its pixels or the
// binding is unusable.
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
        return map;
    }
    catch (const std::bad_alloc&)
    {
        return {};
    }
}
#endif

} // namespace

PT_SHARED_ENTRY bool EnvironmentField::active() const noexcept
{
    if (maxComponent (uniformRadiance) > 0.0f || (image != nullptr && environmentMapAt (image).active()))
        return true;
    return analytic && (maxComponent (zenith) > 0.0f || maxComponent (horizon) > 0.0f
                        || maxComponent (ground) > 0.0f || maxComponent (sunRadiance) > 0.0f);
}

PT_SHARED_ENTRY Vec3 EnvironmentField::radiance (Vec3 direction) const noexcept
{
    Vec3 result = uniformRadiance;
    if (image != nullptr)
        return result + environmentMapAt (image).radiance (direction);
    if (! analytic)
        return result;
    const float y = direction.y;
    const Vec3 pole = y >= 0.0f ? zenith : ground;
    Vec3 sky = horizon + (pole - horizon) * std::fabs (y);
    // The clouds subtract this sky, so an infinite term here would turn into NaN.
    const float capWidth = 1.0f - sunCosRadius;
    if (sunHalo > 0.0f && capWidth > 1.0e-7f)
    {
        // The cap's irradiance pi*(1-cos radius)*peak in a Henyey-Greenstein lobe (g 0.8),
        // strongest at the horizon, as the raster's analyticSkyHalo.
        const float g = 0.8f;
        const float mu = dot (direction, sunDirection);
        const float t = std::max (1.0f + g * g - 2.0f * g * mu, 1.0e-6f);
        const float lobe = (1.0f - g * g) / (12.566370614359172f * t * std::sqrt (t));
        sky = saturatedRadiance (sky + sunRadiance * (3.141592653589793f * capWidth * sunHalo * lobe * (1.0f - 0.65f * std::fabs (y))));
    }
    if (capWidth > 1.0e-7f && maxComponent (sunRadiance) > 0.0f)
    {
        const float c = clampLightUnit ((dot (direction, sunDirection) - sunCosRadius) / capWidth);
        sky = saturatedRadiance (sky + sunRadiance * (c * c * (3.0f - 2.0f * c)));
    }
    if (cloudThickness > 0.0f && cloudCoverage > 0.0f)
    {
        // The volumetric layer dims the sky and sun behind it, so a sampled sun is shadowed.
        cloudvolume::Layer layer;
        layer.coverage = cloudCoverage;
        layer.kind = cloudKind;
        layer.headingRadians = cloudHeadingRadians;
        layer.drift = cloudDrift;
        layer.thickness = cloudThickness;
        const Vec3 irradiance = sunRadiance * (3.141592653589793f * capWidth);
        const float white = maxComponent (horizon * 0.5f + zenith * 0.5f);
        const auto volume = cloudvolume::march (layer, direction.x, direction.y, direction.z, sunDirection.x, sunDirection.y,
                                                sunDirection.z, irradiance.x, irradiance.y, irradiance.z, white);
        return saturatedRadiance (result + sky * volume.transmittance + Vec3 { volume.r, volume.g, volume.b });
    }
    const float cloud = cloudOpacity (*this, direction);
    if (cloud > 0.0f)
    {
        const float white = maxComponent (horizon * 0.5f + zenith * 0.5f);
        const Vec3 lit = saturatedRadiance (Vec3 { white, white, white } + sunRadiance * ((1.0f - sunCosRadius) * std::max (sunDirection.y, 0.0f)))
            * (1.0f - 0.3f * cloud * cloud);
        sky = sky + (lit - sky) * cloud;
    }
    return saturatedRadiance (result + sky);
}

PT_SHARED_ENTRY float EnvironmentField::pdf (Vec3 direction) const noexcept
{
    if (image != nullptr)
        return (1.0f - imageSampleProbability) / kFourPi + imageSampleProbability * environmentMapAt (image).pdf (direction);
    float density = (1.0f - sunSampleProbability) / kFourPi;
    if (sunSampleProbability > 0.0f && dot (direction, sunDirection) >= sunCosRadius)
        density += sunSampleProbability / (kLightTwoPi * (1.0f - sunCosRadius));
    return density;
}

PT_SHARED_ENTRY Vec3 EnvironmentField::sample (Vec2 u, float& pdfOut) const noexcept
{
    Vec3 direction;
    if (image != nullptr && u.x < imageSampleProbability)
    {
        float imagePdf = 0.0f;
        direction = environmentMapAt (image).sample ({ u.x / imageSampleProbability, u.y }, imagePdf);
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
        const float phi = kLightTwoPi * u.y;
        direction = { r * std::cos (phi), r * std::sin (phi), z };
    }
    else if (u.x < sunSampleProbability)
    {
        const float v = u.x / sunSampleProbability;
        const float cosTheta = 1.0f - v * (1.0f - sunCosRadius);
        const float sinTheta = std::sqrt (std::max (0.0f, 1.0f - cosTheta * cosTheta));
        const float phi = kLightTwoPi * u.y;
        direction = Frame::fromNormal (sunDirection)
                        .toWorld ({ sinTheta * std::cos (phi), sinTheta * std::sin (phi), cosTheta });
    }
    else
    {
        const float v = (u.x - sunSampleProbability) / (1.0f - sunSampleProbability);
        const float z = 1.0f - 2.0f * std::min (v, 1.0f);
        const float r = std::sqrt (std::max (0.0f, 1.0f - z * z));
        const float phi = kLightTwoPi * u.y;
        direction = { r * std::cos (phi), r * std::sin (phi), z };
    }
    direction = normalizeOrZero (direction);
    pdfOut = pdf (direction);
    return direction;
}

// Host only: the Metal kernel (device_metal.metal) leaves this out.
#if ! defined(PATHTRACE_DEVICE_METAL)
void LightSet::clear() noexcept
{
    ownedLights_.clear();
    ownedCdf_.clear();
    ownedProbability_.clear();
    ownedTriangleLight_.clear();
    ownedAnalytic_.clear();
    environment_ = {};
    environmentLight_ = -1;
    ownedImage_.reset();
    stats_ = {};
    refreshView();
}

void LightSet::refreshView() noexcept
{
    lights_ = { ownedLights_.data(), ownedLights_.size() };
    cdf_ = { ownedCdf_.data(), ownedCdf_.size() };
    probability_ = { ownedProbability_.data(), ownedProbability_.size() };
    triangleLight_ = { ownedTriangleLight_.data(), ownedTriangleLight_.size() };
    analytic_ = { ownedAnalytic_.data(), ownedAnalytic_.size() };
}

LightSet::LightSet (const LightSet& other)
    : LightSetView (other),
      ownedLights_ (other.ownedLights_),
      ownedCdf_ (other.ownedCdf_),
      ownedProbability_ (other.ownedProbability_),
      ownedTriangleLight_ (other.ownedTriangleLight_),
      ownedAnalytic_ (other.ownedAnalytic_),
      ownedImage_ (other.ownedImage_),
      stats_ (other.stats_)
{
    refreshView();
}

LightSet::LightSet (LightSet&& other) noexcept
    : LightSetView (other),
      ownedLights_ (std::move (other.ownedLights_)),
      ownedCdf_ (std::move (other.ownedCdf_)),
      ownedProbability_ (std::move (other.ownedProbability_)),
      ownedTriangleLight_ (std::move (other.ownedTriangleLight_)),
      ownedAnalytic_ (std::move (other.ownedAnalytic_)),
      ownedImage_ (std::move (other.ownedImage_)),
      stats_ (other.stats_)
{
    refreshView();
    other.clear();
}

LightSet& LightSet::operator= (const LightSet& other)
{
    if (this != &other)
    {
        LightSet copy (other);
        *this = std::move (copy);
    }
    return *this;
}

LightSet& LightSet::operator= (LightSet&& other) noexcept
{
    if (this != &other)
    {
        LightSetView::operator= (other);
        ownedLights_ = std::move (other.ownedLights_);
        ownedCdf_ = std::move (other.ownedCdf_);
        ownedProbability_ = std::move (other.ownedProbability_);
        ownedTriangleLight_ = std::move (other.ownedTriangleLight_);
        ownedAnalytic_ = std::move (other.ownedAnalytic_);
        ownedImage_ = std::move (other.ownedImage_);
        stats_ = other.stats_;
        refreshView();
        other.clear();
    }
    return *this;
}

PT_SHARED_ENTRY bool authoredSpectralSourceLight (const diffractionmaterial::LightingDescription& lighting,
                                                  SceneLightRecord& record, LightShape& shape) noexcept
{
    if (lighting.version != 3 || lighting.pathCount != 1
        || lighting.paths[0].kind != diffractionmaterial::LightingPathKind::Direct)
        return false;
    const auto& incident = lighting.paths[0].incident;
    const Vec3 toLight = normalizeOrZero ({ incident.direction[0], incident.direction[1], incident.direction[2] });
    float sum = 0.0f;
    for (const float band : incident.radiance)
        sum += band;
    const float mean = sum / static_cast<float> (incident.radiance.size());
    if (dot (toLight, toLight) == 0.0f || ! std::isfinite (mean) || ! (mean > 0.0f))
        return false;
    record = {};
    record.kind = SceneLightKind::Directional;
    record.color = { 1.0f, 1.0f, 1.0f };
    record.intensity = mean;
    // The shortest turn from local +Z, the record's way toward the light.
    const Vec3 axis = cross ({ 0.0f, 0.0f, 1.0f }, toLight);
    const float w = 1.0f + toLight.z;
    if (w > 1.0e-6f)
    {
        const float size = std::sqrt (dot (axis, axis) + w * w);
        record.transform.rotation = { axis.x / size, axis.y / size, axis.z / size, w / size };
    }
    else
    {
        record.transform.rotation = { 1.0f, 0.0f, 0.0f, 0.0f };
    }
    shape = {};
    shape.angularRadius = kGratingSourceAngularRadius;
    shape.spectralBands = true;
    shape.bands = incident.radiance;
    return true;
}

#endif

PT_SHARED_ENTRY float areaSpreadAttenuation (float tanHalfSpread, float normalization, float cosAtLight) noexcept
{
    if (! (normalization > 0.0f))
        return 1.0f;
    if (! (cosAtLight > 0.0f))
        return 0.0f;
    const float sinAtLight = std::sqrt (std::max (0.0f, 1.0f - cosAtLight * cosAtLight));
    return std::max (0.0f, (tanHalfSpread - sinAtLight / cosAtLight) * normalization);
}

#if ! defined(PATHTRACE_DEVICE_METAL)
void LightSet::build (const SceneLightRecord* records, std::size_t count, Vec3 ambient,
                      const IngestedScene& geometry, const std::vector<Material>& materials,
                      const Material& fallback, const EnvironmentPixels* environmentImage,
                      const std::vector<LightShape>* shapes, const std::vector<AreaLightRecord>* areaLights)
{
    clear();
    // Every exit, exceptions included, leaves the view on the current buffers.
    struct ViewRefresh
    {
        LightSet& set;
        ~ViewRefresh() { set.refreshView(); }
    } viewRefresh { *this };
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
        ownedLights_.push_back (light);
        weights.push_back (weight);
    };

    // Rects, disks and ellipses, from areaLights and from Area records.
    const auto addArea = [&] (const AreaLightRecord& record) {
        const Vec3 emitted = record.color * record.intensity;
        Quat q;
        if (! isFinite (emitted) || ! isFinite (record.position) || ! normalizedQuat (record.rotation, q)
            || ! std::isfinite (record.sizeX) || ! std::isfinite (record.sizeY) || ! (record.sizeX > 0.0f)
            || ! (record.sizeY > 0.0f))
        {
            ++stats_.invalidLights;
            return;
        }
        const Vec3 axisX = normalizeOrZero (rotate (q, { 1.0f, 0.0f, 0.0f }));
        const Vec3 axisY = normalizeOrZero (rotate (q, { 0.0f, 1.0f, 0.0f }));
        Light light;
        light.strength = nonNegative (emitted);
        light.normal = normalizeOrZero (rotate (q, { 0.0f, 0.0f, -1.0f }));
        light.twoSided = record.twoSided;
        light.visibility = record.visibility;
        const float spreadNormalization = HarmonicMIDI::grid::sceneAreaSpreadNormalization (record.spread);
        if (spreadNormalization > 0.0f)
        {
            light.tanHalfSpread = std::tan (0.5f * std::max (record.spread, 1.0e-4f));
            light.spreadNormalization = spreadNormalization;
        }
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
            return;
        }
        const std::size_t before = ownedLights_.size();
        // pi * L * A per emitting face.
        push (light, (light.twoSided ? 2.0 : 1.0) * kPi * static_cast<double> (meanOf (light.strength))
                         * static_cast<double> (light.area));
        if (ownedLights_.size() > before)
            ++stats_.areaLights;
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
        const Vec3 strength = nonNegative (finiteProduct (color, { 1.0f, 1.0f, 1.0f }, record.intensity));
        LightShape shape = shapes != nullptr && i < shapes->size() ? (*shapes)[i] : LightShape {};
        // A size the caller's shape sets was not authored as a visible emitter
        // (a grating's widened sun, a spectral plan's source): the camera does
        // not see it. Otherwise the record's own size and visibility apply.
        const bool callerSized = shape.radius > 0.0f || shape.angularRadius > 0.0f;
        if (! (shape.radius > 0.0f))
            shape.radius = record.radius;
        if (! (shape.angularRadius > 0.0f))
            shape.angularRadius = record.angularRadius;
        Quat q;
        const bool rotationValid = normalizedRotation (record, q);
        Light light;
        light.visibility = record.visibility;
        if (callerSized)
            light.visibility = static_cast<std::uint8_t> (light.visibility & ~HarmonicMIDI::grid::sceneRayVisibility::Camera);
        switch (record.kind)
        {
            case SceneLightKind::Area:
            {
                const HarmonicMIDI::grid::SceneVec2 extent = HarmonicMIDI::grid::sceneAreaLightExtent (record);
                AreaLightRecord area;
                area.shape = record.areaShape == HarmonicMIDI::grid::SceneAreaLightShape::Square
                                     || record.areaShape == HarmonicMIDI::grid::SceneAreaLightShape::Rectangle
                                 ? AreaLightShape::Rect
                                 : record.areaShape == HarmonicMIDI::grid::SceneAreaLightShape::Disk ? AreaLightShape::Disk
                                                                                                     : AreaLightShape::Ellipse;
                area.position = toVec3 (record.transform.translation);
                area.rotation = q;
                area.sizeX = extent.x;
                area.sizeY = extent.y;
                const float emittingArea = area.shape == AreaLightShape::Rect ? extent.x * extent.y
                                                                              : 0.25f * kPi * extent.x * extent.y;
                // intensity is the radiant intensity along the axis, as a point light's.
                area.color = color;
                area.intensity = record.intensity / emittingArea;
                area.spread = record.spread;
                area.visibility = light.visibility;
                if (! rotationValid || ! (emittingArea > 0.0f) || ! std::isfinite (area.intensity))
                {
                    ++stats_.invalidLights;
                    break;
                }
                addArea (area);
                break;
            }
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
                if (shape.spectralBands && meanOf (strength) > 0.0f)
                {
                    float sum = 0.0f;
                    for (const float band : shape.bands)
                        sum += band;
                    const float mean = sum / static_cast<float> (shape.bands.size());
                    light.spectralBands = std::isfinite (mean) && mean > 0.0f;
                    for (std::size_t b = 0; light.spectralBands && b < shape.bands.size(); ++b)
                        light.bandScale[b] = shape.bands[b] / mean;
                }
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
                push (light, static_cast<double> (meanOf (strength)) * sceneDisk);
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
                push (light, static_cast<double> (meanOf (strength)) * solidAngle);
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
                    environment_.uniformRadiance = environment_.uniformRadiance + strength;
                    ++stats_.environment;
                    break;
                }
                const auto& sky = record.sky;
                const Vec3 zenith = finiteProduct (toVec3 (sky.zenith), strength, 1.0f);
                const Vec3 horizon = finiteProduct (toVec3 (sky.horizon), strength, 1.0f);
                const Vec3 ground = finiteProduct (toVec3 (sky.ground), strength, 1.0f);
                const Vec3 sun = finiteProduct (toVec3 (sky.sunColor), strength, sky.sunIntensity);
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
                environment_.sunHalo = std::round (std::min (1.0f, std::max (0.0f, sky.sunHalo)) * 63.0f) / 63.0f;
                const float coverage = std::round (std::min (1.0f, std::max (0.0f, sky.cloudCoverage)) * 1023.0f);
                if (coverage > 0.0f)
                {
                    const float turns = (sky.cloudHeadingDegrees + sky.orientationDegrees) / 360.0f;
                    const float headingCode = std::fmod (std::round ((turns - std::floor (turns)) * 256.0f), 256.0f);
                    environment_.cloudCoverage = coverage / 1023.0f;
                    environment_.cloudKind = std::round (std::min (1.0f, std::max (0.0f, sky.cloudType)) * 63.0f) / 63.0f;
                    environment_.cloudHeadingRadians = headingCode * 6.283185307179586f / 256.0f;
                    environment_.cloudDrift = sky.cloudDrift;
                    environment_.cloudThickness = std::round (std::min (1.0f, std::max (0.0f, sky.cloudThickness)) * 63.0f) / 63.0f;
                }
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
        ownedImage_ = image;
        environment_.image = ownedImage_.get();
    }

    if (isFinite (ambient))
    {
        const Vec3 sum = environment_.uniformRadiance + nonNegative (ambient), top = Vec3 { 1.0f, 1.0f, 1.0f } * std::numeric_limits<float>::max();
        environment_.uniformRadiance = { std::min (sum.x, top.x), std::min (sum.y, top.y), std::min (sum.z, top.z) };
    }

    if (environment_.active())
    {
        Vec3 meanSky = environment_.uniformRadiance;
        if (environment_.analytic)
            meanSky = meanSky + (environment_.horizon * 2.0f + environment_.zenith + environment_.ground) * 0.25f;
        double skyIrradiance = kPi * static_cast<double> (meanOf (meanSky));
        if (! isFinite (meanSky))
        {
            // A sky past a float still weighs a finite amount in doubles.
            const auto channel = [&] (float constant, float horizon, float zenith, float ground) {
                const double sky = environment_.analytic ? (2.0 * horizon + zenith + ground) * 0.25 : 0.0;
                return constant + sky;
            };
            const double r = channel (environment_.uniformRadiance.x, environment_.horizon.x, environment_.zenith.x, environment_.ground.x);
            const double g = channel (environment_.uniformRadiance.y, environment_.horizon.y, environment_.zenith.y, environment_.ground.y);
            const double b = channel (environment_.uniformRadiance.z, environment_.horizon.z, environment_.zenith.z, environment_.ground.z);
            skyIrradiance = kPi * (r + g + b) / 3.0;
        }
        if (environment_.image != nullptr)
        {
            // Mean radiance over the sphere is the integral over 4 pi.
            const double imageIntegral = static_cast<double> (meanOf (environment_.image->integratedRadiance()));
            const double constantIntegral = static_cast<double> (meanOf (environment_.uniformRadiance)) * kFourPi;
            skyIrradiance += kPi * imageIntegral / kFourPi;
            float share = 1.0f;
            if (constantIntegral > 0.0)
                share = static_cast<float> (std::min (0.95, std::max (0.05, imageIntegral / (imageIntegral + constantIntegral))));
            environment_.imageSampleProbability = share;
        }
        // The smooth cap averages about half its peak over its solid angle.
        const double sunIrradiance = static_cast<double> (meanOf (environment_.sunRadiance))
                                   * kLightTwoPi * (1.0 - static_cast<double> (environment_.sunCosRadius)) * 0.5;
        if (sunIrradiance > 0.0)
        {
            const double share = sunIrradiance / (sunIrradiance + skyIrradiance);
            environment_.sunSampleProbability = static_cast<float> (std::min (0.95, std::max (0.05, share)));
        }
        Light light;
        light.type = LightType::Environment;
        const std::size_t before = ownedLights_.size();
        push (light, (skyIrradiance + sunIrradiance) * sceneDisk);
        if (ownedLights_.size() > before)
            environmentLight_ = static_cast<std::int32_t> (ownedLights_.size() - 1);
    }

    const std::size_t triangleCount = geometry.triangles.size();
    ownedTriangleLight_.assign (triangleCount, -1);
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
        const std::size_t before = ownedLights_.size();
        // Emission leaves both faces: pi * L * A per face.
        push (light, 2.0 * kPi * static_cast<double> (meanOf (light.strength)) * area);
        if (ownedLights_.size() > before)
        {
            ownedTriangleLight_[t] = static_cast<std::int32_t> (ownedLights_.size() - 1);
            ++stats_.emissiveTriangles;
        }
    }

    // Caller-given rects and disks keep their old rule: the camera does not see them.
    for (std::size_t a = 0; areaLights != nullptr && a < areaLights->size(); ++a)
    {
        AreaLightRecord area = (*areaLights)[a];
        area.visibility = static_cast<std::uint8_t> (area.visibility & ~HarmonicMIDI::grid::sceneRayVisibility::Camera);
        addArea (area);
    }

    double total = 0.0;
    for (const double w : weights)
        total += w;
    if (! (total > 0.0) || ! std::isfinite (total))
    {
        ownedLights_.clear();
        ownedTriangleLight_.assign (triangleCount, -1);
        environmentLight_ = -1;
        return;
    }
    for (std::size_t i = 0; i < ownedLights_.size(); ++i)
    {
        const Light& light = ownedLights_[i];
        const bool sized = light.radius > 0.0f || light.coneOneMinusCos > 0.0f || light.type == LightType::Rect
                        || light.type == LightType::Disk;
        if (sized)
            ownedAnalytic_.push_back (static_cast<std::uint32_t> (i));
    }
    ownedProbability_.resize (weights.size());
    ownedCdf_.resize (weights.size());
    double running = 0.0;
    for (std::size_t i = 0; i < weights.size(); ++i)
    {
        ownedProbability_[i] = static_cast<float> (weights[i] / total);
        running += weights[i];
        ownedCdf_[i] = static_cast<float> (running / total);
    }
    ownedCdf_.back() = 1.0f;
}
#endif

PT_SHARED_ENTRY float LightSetView::selectionProbability (std::size_t light) const noexcept
{
    return light < probability_.size() ? probability_[light] : 0.0f;
}

PT_SHARED_ENTRY bool LightSetView::sample (Vec3 p, float uSelect, Vec2 u, LightSample& out) const noexcept
{
    out = {};
    if (lights_.empty())
        return false;
    std::size_t index = upperBoundIndex (cdf_.data, cdf_.size(), uSelect);
    index = std::min (index, lights_.size() - 1);
    const float selection = probability_[index];
    if (! (selection > 0.0f))
        return false;
    const PT_GLOBAL Light& light = lights_[index];
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
                const float phi = kLightTwoPi * u.y;
                out.wi = normalizeOrZero (Frame::fromNormal (light.direction)
                                              .toWorld ({ sinTheta * std::cos (phi), sinTheta * std::sin (phi), cosTheta }));
                out.radiance = light.strength * light.coneRadianceScale;
                out.pdf = selection / (kLightTwoPi * light.coneOneMinusCos);
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
                const float phi = kLightTwoPi * u.y;
                out.wi = normalizeOrZero (Frame::fromNormal (axis)
                                              .toWorld ({ sinTheta * std::cos (phi), sinTheta * std::sin (phi), cosTheta }));
                // Nearest intersection with the sphere along wi.
                const float along = dot (out.wi, toLight);
                const float disc = light.radius * light.radius - std::max (0.0f, d2 - along * along);
                const float t = std::max (0.0f, along - std::sqrt (std::max (0.0f, disc)));
                out.radiance = sphereRadiance (light, d2, -axis);
                out.point = p + out.wi * t;
                out.distance = t;
                out.pdf = selection / (kLightTwoPi * coneWidth);
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
            out.radiance = light.strength * areaSpreadAttenuation (light.tanHalfSpread, light.spreadNormalization, cosAtLight);
            out.point = q;
            out.lightNormal = light.normal;
            out.distance = d;
            out.pdf = selection * directionPdf;
            return std::isfinite (out.pdf) && out.pdf > 0.0f;
        }
    }
    return false;
}

PT_SHARED_ENTRY float LightSetView::spectralBandScale (std::uint32_t index, float wavelengthNm) const noexcept
{
    if (index >= lights_.size() || ! lights_[index].spectralBands)
        return 1.0f;
    const PT_GLOBAL auto& scale = lights_[index].bandScale;
    // Copied, since device code cannot bind a reference to a host global.
    constexpr auto grid = diffractionmaterial::kProductionWavelengthsNanometres;
    if (wavelengthNm <= grid[0])
        return scale[0];
    for (std::size_t b = 1; b < grid.size(); ++b)
        if (wavelengthNm <= grid[b])
        {
            const float t = (wavelengthNm - grid[b - 1]) / (grid[b] - grid[b - 1]);
            return scale[b - 1] + (scale[b] - scale[b - 1]) * t;
        }
    return scale[scale.size() - 1];
}

PT_SHARED_ENTRY bool LightSetView::analyticHit (std::size_t index, Vec3 origin, Vec3 direction, float tMax, Vec3& radiance,
                            float& pdf, float& distance) const noexcept
{
    radiance = {};
    pdf = 0.0f;
    distance = std::numeric_limits<float>::infinity();
    if (index >= analytic_.size())
        return false;
    const std::size_t slot = analytic_[index];
    const PT_GLOBAL Light& light = lights_[slot];
    const float selection = probability_[slot];
    switch (light.type)
    {
        case LightType::Directional:
        {
            if (! std::isinf (tMax) || ! (light.coneOneMinusCos > 0.0f) || dot (direction, light.direction) < light.coneCos)
                return false;
            radiance = light.strength * light.coneRadianceScale;
            pdf = selection / (kLightTwoPi * light.coneOneMinusCos);
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
            pdf = selection / (kLightTwoPi * sphereConeOneMinusCos (light.radius, d2));
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
            radiance = light.strength
                     * areaSpreadAttenuation (light.tanHalfSpread, light.spreadNormalization, std::fabs (denominator));
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

namespace
{
// Cosine-weighted direction about unit n (Malley's method on the concentric disk).
PT_SHARED_ENTRY Vec3 cosineDirection (Vec3 n, Vec2 u) noexcept
{
    const Vec2 d = concentricDisk (u);
    const float z = std::sqrt (std::max (0.0f, 1.0f - d.x * d.x - d.y * d.y));
    return normalizeOrZero (Frame::fromNormal (n).toWorld ({ d.x, d.y, z }));
}

// Uniform direction in the cone of unit axis with 1 - cos of its half angle oneMinusCosMax.
PT_SHARED_ENTRY Vec3 coneDirection (Vec3 axis, float oneMinusCosMax, Vec2 u) noexcept
{
    const float oneMinusCos = u.x * oneMinusCosMax;
    const float cosTheta = 1.0f - oneMinusCos;
    const float sinTheta = std::sqrt (std::max (0.0f, oneMinusCos * (2.0f - oneMinusCos)));
    const float phi = kLightTwoPi * u.y;
    return normalizeOrZero (Frame::fromNormal (axis).toWorld ({ sinTheta * std::cos (phi), sinTheta * std::sin (phi), cosTheta }));
}

// Origin of a particle from infinitely far: a point of the disk of radius
// `radius` that faces `travel`, one radius behind the centre.
PT_SHARED_ENTRY Vec3 farOrigin (Vec3 centre, float radius, Vec3 travel, Vec2 u) noexcept
{
    const Vec2 d = concentricDisk (u);
    const Frame frame = Frame::fromNormal (travel);
    return centre - travel * radius + (frame.tangent * d.x + frame.bitangent * d.y) * radius;
}

// Picks a face of a two-sided emitter with u.x and rescales u.x for reuse.
PT_SHARED_ENTRY Vec3 pickFace (Vec3 normal, bool twoSided, Vec2& u) noexcept
{
    if (! twoSided)
        return normal;
    if (u.x < 0.5f)
    {
        u.x = std::min (u.x * 2.0f, std::nextafter (1.0f, 0.0f));
        return normal;
    }
    u.x = std::min ((u.x - 0.5f) * 2.0f, std::nextafter (1.0f, 0.0f));
    return -normal;
}
} // namespace

PT_SHARED_ENTRY bool LightSetView::sampleEmission (float uSelect, Vec2 uPosition, Vec2 uDirection, Vec3 sceneCentre, float sceneRadius,
                               EmissionSample& out) const noexcept
{
    out = {};
    if (lights_.empty())
        return false;
    std::size_t index = upperBoundIndex (cdf_.data, cdf_.size(), uSelect);
    index = std::min (index, lights_.size() - 1);
    const float selection = probability_[index];
    if (! (selection > 0.0f))
        return false;
    const PT_GLOBAL Light& light = lights_[index];
    out.light = static_cast<std::uint32_t> (index);
    out.type = light.type;
    const bool farLightsReachable = std::isfinite (sceneRadius) && sceneRadius > 0.0f && isFinite (sceneCentre);
    const float diskArea = kPi * sceneRadius * sceneRadius;
    switch (light.type)
    {
        case LightType::Directional:
        {
            if (! farLightsReachable)
                return false;
            if (light.coneOneMinusCos > 0.0f)
            {
                const Vec3 toLight = coneDirection (light.direction, light.coneOneMinusCos, uDirection);
                out.direction = -toLight;
                out.weight = light.strength * (light.coneRadianceScale * diskArea * kLightTwoPi * light.coneOneMinusCos / selection);
            }
            else
            {
                out.direction = -light.direction;
                out.weight = light.strength * (diskArea / selection);
            }
            out.origin = farOrigin (sceneCentre, sceneRadius, out.direction, uPosition);
            break;
        }
        case LightType::Point:
        case LightType::Spot:
        {
            // The weight leaves the light as intensity times solid angle;
            // emissionFalloff turns it into the raster's irradiance.
            out.receiverFalloff = true;
            if (light.radius > 0.0f)
            {
                // A sphere of radiance L(d) = strength g(d) / (pi r^2) over area
                // 4 pi r^2, cosine-weighted: strength 4 pi times g(d) / d^2 at d.
                const float z = 1.0f - 2.0f * uPosition.x;
                const float r = std::sqrt (std::max (0.0f, 1.0f - z * z));
                const float phi = kLightTwoPi * uPosition.y;
                const Vec3 outward { r * std::cos (phi), r * std::sin (phi), z };
                out.origin = light.position + outward * light.radius;
                out.direction = cosineDirection (outward, uDirection);
                out.weight = light.strength * (kFourPi / selection);
            }
            else if (light.type == LightType::Spot)
            {
                const float width = std::min (2.0f, std::max (1.0e-6f, 1.0f - light.cosOuter));
                out.origin = light.position;
                out.direction = coneDirection (light.direction, width, uDirection);
                out.weight = light.strength * (kLightTwoPi * width / selection);
            }
            else
            {
                const float z = 1.0f - 2.0f * uDirection.x;
                const float r = std::sqrt (std::max (0.0f, 1.0f - z * z));
                const float phi = kLightTwoPi * uDirection.y;
                out.origin = light.position;
                out.direction = { r * std::cos (phi), r * std::sin (phi), z };
                out.weight = light.strength * (kFourPi / selection);
            }
            break;
        }
        case LightType::Environment:
        {
            if (! farLightsReachable)
                return false;
            float directionPdf = 0.0f;
            const Vec3 toSky = environment_.sample (uDirection, directionPdf);
            if (! (directionPdf > 0.0f) || dot (toSky, toSky) == 0.0f)
                return false;
            out.direction = -toSky;
            out.origin = farOrigin (sceneCentre, sceneRadius, out.direction, uPosition);
            out.weight = environment_.radiance (toSky) * (diskArea / (directionPdf * selection));
            break;
        }
        case LightType::Triangle:
        case LightType::Rect:
        case LightType::Disk:
        {
            Vec2 u = uDirection;
            const bool twoSided = light.type == LightType::Triangle || light.twoSided;
            const Vec3 face = pickFace (light.normal, twoSided, u);
            if (light.type == LightType::Triangle)
            {
                const float root = std::sqrt (uPosition.x);
                out.origin = light.vertex0 + light.edge1 * (root * (1.0f - uPosition.y)) + light.edge2 * (root * uPosition.y);
                out.ignoreTriangle = light.triangle;
            }
            else if (light.type == LightType::Rect)
            {
                out.origin = light.vertex0 + light.edge1 * uPosition.x + light.edge2 * uPosition.y;
            }
            else
            {
                const Vec2 d = concentricDisk (uPosition);
                out.origin = light.position + light.edge1 * d.x + light.edge2 * d.y;
            }
            // Leave the surface on the emitting side.
            out.origin = offsetRayOrigin (out.origin, face);
            out.direction = cosineDirection (face, u);
            // Radiance L over area A and the cosine-weighted hemisphere:
            // L A pi, times 2 when a face was picked at random.
            out.weight = light.strength * (light.area * kPi * (twoSided ? 2.0f : 1.0f) / selection);
            if (light.type != LightType::Triangle)
                out.weight = out.weight
                           * areaSpreadAttenuation (light.tanHalfSpread, light.spreadNormalization, dot (out.direction, face));
            break;
        }
    }
    return dot (out.direction, out.direction) > 0.0f && isFinite (out.origin) && isFinite (out.weight)
        && (out.weight.x > 0.0f || out.weight.y > 0.0f || out.weight.z > 0.0f);
}

PT_SHARED_ENTRY float LightSetView::emissionFalloff (const EmissionSample& sample, Vec3 receiver) const noexcept
{
    if (! sample.receiverFalloff || sample.light >= lights_.size())
        return 1.0f;
    const PT_GLOBAL Light& light = lights_[sample.light];
    const Vec3 fromLight = receiver - light.position;
    const float d2 = dot (fromLight, fromLight);
    if (! (d2 > 0.0f) || ! std::isfinite (d2))
        return 0.0f;
    return d2 * pointFalloff (light, d2, fromLight * (1.0f / std::sqrt (d2)));
}

PT_SHARED_ENTRY float LightSetView::trianglePdf (std::uint32_t triangle, float distance, float cosAtLight) const noexcept
{
    if (triangle >= triangleLight_.size() || triangleLight_[triangle] < 0 || ! (cosAtLight > 1.0e-6f))
        return 0.0f;
    const auto index = static_cast<std::size_t> (triangleLight_[triangle]);
    return probability_[index] * distance * distance / (cosAtLight * lights_[index].area);
}

PT_SHARED_ENTRY float LightSetView::environmentPdf (Vec3 direction) const noexcept
{
    if (environmentLight_ < 0)
        return 0.0f;
    return probability_[static_cast<std::size_t> (environmentLight_)] * environment_.pdf (direction);
}

} // namespace videohelper::pathtrace
