#include "lights.h"

#include <algorithm>
#include <cmath>
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
    environment_ = {};
    environmentLight_ = -1;
    stats_ = {};
}

void LightSet::build (const SceneLightRecord* records, std::size_t count, Vec3 ambient,
                      const IngestedScene& geometry, const std::vector<Material>& materials,
                      const Material& fallback, const EnvironmentPixels* environmentImage)
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
            out.wi = light.direction;
            out.radiance = light.strength;
            out.pdf = selection;
            out.delta = true;
            out.infinite = true;
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
            out.wi = toLight * (1.0f / d);
            float falloff = 1.0f / std::max (d2, 1.0f);
            if (light.range > 0.0f)
            {
                const float window = clampUnit (1.0f - d / light.range);
                falloff *= window * window;
            }
            if (light.type == LightType::Spot)
                falloff *= spotCone (dot (light.direction, -out.wi), light.cosOuter, light.cosInner);
            out.radiance = light.strength * falloff;
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
