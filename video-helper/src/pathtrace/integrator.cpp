#include "integrator.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace videohelper::pathtrace
{
namespace
{

using HarmonicMIDI::grid::SceneAlphaMode;
using HarmonicMIDI::grid::SceneMaterialRecord;
using HarmonicMIDI::grid::Visual3DScene;

// Sample dimensions: the pixel filter takes dimension 0, then every path
// vertex owns a block of kDimensionsPerVertex numbers.
enum : std::uint32_t
{
    kSlotTransparency = 0,
    kSlotLightSelect = 1,
    kSlotLightPosition = 2,
    kSlotLobe = 3,
    kSlotDirection = 4,
    kSlotRoulette = 5,
    kDimensionsPerVertex = 8
};

std::uint32_t dimensionOf (std::uint32_t vertex, std::uint32_t slot) noexcept
{
    return 1u + vertex * kDimensionsPerVertex + slot;
}

constexpr std::uint32_t kMaxShadowCrossings = 16;

float maxComponent (Vec3 v) noexcept { return std::max (v.x, std::max (v.y, v.z)); }

bool anyPositive (Vec3 v) noexcept { return v.x > 0.0f || v.y > 0.0f || v.z > 0.0f; }

float finiteOr (float value, float fallback) noexcept { return std::isfinite (value) ? value : fallback; }

// Power heuristic with exponent 2, written as a ratio so huge pdfs cannot
// overflow into inf / inf.
float powerHeuristic (float chosen, float other) noexcept
{
    if (! (chosen > 0.0f))
        return 0.0f;
    if (! (other > 0.0f))
        return 1.0f;
    const float ratio = other / chosen;
    return 1.0f / (1.0f + ratio * ratio);
}

// Weight of an emitter reached by a BSDF ray. lightPdf is the density light
// sampling would have given the same direction, 0 when it cannot reach it.
float emitterHitWeight (LightStrategy strategy, float bsdfPdf, float lightPdf) noexcept
{
    if (! (lightPdf > 0.0f))
        return 1.0f;
    switch (strategy)
    {
        case LightStrategy::MultipleImportance:
            return powerHeuristic (bsdfPdf, lightPdf);
        case LightStrategy::LightOnly:
            return 0.0f;
        case LightStrategy::BsdfOnly:
            return 1.0f;
    }
    return 1.0f;
}

struct SurfacePoint
{
    Vec3 position {};
    Vec3 geometricNormal {};  // front face
    Vec3 shadingNormal {};    // same hemisphere as geometricNormal
};

SurfacePoint surfaceAt (const RenderScene& scene, const Hit& hit) noexcept
{
    const std::size_t base = static_cast<std::size_t> (hit.triangle) * 3;
    const TriangleAttributes& attributes = scene.geometry.triangles[hit.triangle];
    const float w = 1.0f - hit.u - hit.v;
    SurfacePoint point;
    point.position = scene.geometry.positions[base] * w + scene.geometry.positions[base + 1] * hit.u
                   + scene.geometry.positions[base + 2] * hit.v;
    point.geometricNormal = attributes.geometricNormal;
    point.shadingNormal = point.geometricNormal;
    if (attributes.hasShadingNormals)
    {
        const Vec3 n = normalizeOrZero (attributes.shadingNormals[0] * w + attributes.shadingNormals[1] * hit.u
                                        + attributes.shadingNormals[2] * hit.v);
        if (dot (n, n) > 0.0f)
            point.shadingNormal = dot (n, point.geometricNormal) < 0.0f ? -n : n;
    }
    return point;
}

// Fraction of light that crosses the segment of `ray`. Opaque and
// transmissive (glass) surfaces block it; partially covered ones attenuate.
float shadowTransmittance (const RenderScene& scene, Ray ray, PathCounters& counters) noexcept
{
    ++counters.shadowRays;
    if (! scene.anyPartialCoverage)
        return scene.bvh.occluded (ray) ? 0.0f : 1.0f;
    float transmittance = 1.0f;
    for (std::uint32_t crossing = 0; crossing <= kMaxShadowCrossings; ++crossing)
    {
        Hit hit;
        if (! scene.bvh.intersect (ray, hit))
            return transmittance;
        const Material& material = scene.materialOf (hit.triangle);
        const float coverage = materialCoverage (material);
        if (coverage >= 1.0f || material.transmission > 0.0f)
            return 0.0f;
        transmittance *= 1.0f - coverage;
        if (! (transmittance > 0.0f))
            return 0.0f;
        ray.tMin = hit.t;
        ray.ignoreTriangle = hit.triangle;
    }
    return 0.0f;
}

} // namespace

PathCounters& PathCounters::operator+= (const PathCounters& other) noexcept
{
    closestRays += other.closestRays;
    shadowRays += other.shadowRays;
    clampedContributions += other.clampedContributions;
    transparentLimitPaths += other.transparentLimitPaths;
    return *this;
}

void RenderScene::commit()
{
    bvh.build (geometry.positions);
    lights.build (lightRecords.data(), lightRecords.size(), ambient, geometry, materials, fallbackMaterial,
                  &environmentImage);
    anyPartialCoverage = materialCoverage (fallbackMaterial) < 1.0f;
    for (const Material& material : materials)
        anyPartialCoverage = anyPartialCoverage || materialCoverage (material) < 1.0f;
}

const Material& RenderScene::materialOf (std::uint32_t triangle) const noexcept
{
    if (triangle >= geometry.triangles.size())
        return fallbackMaterial;
    const std::uint32_t index = geometry.triangles[triangle].materialIndex;
    return index < materials.size() ? materials[index] : fallbackMaterial;
}

Material materialFromRecord (const SceneMaterialRecord& record) noexcept
{
    const Material defaults {};
    const auto unit = [] (float value, float fallback) {
        return std::isfinite (value) ? std::min (1.0f, std::max (0.0f, value)) : fallback;
    };
    const auto radiance = [] (float value) {
        return std::isfinite (value) && value > 0.0f ? value : 0.0f;
    };
    Material material;
    material.baseColor = { unit (record.baseColor.x, defaults.baseColor.x), unit (record.baseColor.y, defaults.baseColor.y),
                           unit (record.baseColor.z, defaults.baseColor.z) };
    material.metallic = unit (record.metallic, defaults.metallic);
    material.roughness = unit (record.roughness, defaults.roughness);
    material.emission = { radiance (record.emissive.x), radiance (record.emissive.y), radiance (record.emissive.z) };
    material.opacity = unit (record.opacity, defaults.opacity);
    material.alphaCutoff = finiteOr (record.alphaCutoff, defaults.alphaCutoff);
    switch (record.alphaMode)
    {
        case SceneAlphaMode::Opaque: material.alphaMode = MaterialAlphaMode::Opaque; break;
        case SceneAlphaMode::Mask: material.alphaMode = MaterialAlphaMode::Mask; break;
        case SceneAlphaMode::Blend: material.alphaMode = MaterialAlphaMode::Blend; break;
    }
    return material;
}

void prepareRenderScene (const Visual3DScene& scene, RenderScene& out, const PrepareOptions& options)
{
    ingestScene (scene, out.geometry, options.ingest);
    out.materials.clear();
    const std::size_t materialCount = std::min (scene.materialCount, Visual3DScene::kMaxMaterials);
    for (std::size_t i = 0; i < materialCount; ++i)
        out.materials.push_back (materialFromRecord (scene.materials[i]));
    out.fallbackMaterial = Material {};
    const std::size_t lightCount = std::min (scene.lightCount, Visual3DScene::kMaxLights);
    out.lightRecords.assign (scene.lights.begin(), scene.lights.begin() + static_cast<std::ptrdiff_t> (lightCount));
    out.ambient = options.ambientAsEnvironment
                    ? Vec3 { scene.ambientColor.x, scene.ambientColor.y, scene.ambientColor.z }
                    : Vec3 {};
    // The raster binds at most one image (validSceneAnalyticSkySet); resolve
    // the first record that names one.
    out.environmentImage = {};
    if (options.environmentImageSource)
        for (const auto& record : out.lightRecords)
            if (record.kind == HarmonicMIDI::grid::SceneLightKind::Environment && record.environmentImage.version != 0)
            {
                out.environmentImage = options.environmentImageSource (record.environmentImage);
                break;
            }
    out.commit();
}

bool tracePath (const RenderScene& scene, const IntegratorSettings& settings, const Ray& cameraRay,
                const PathSampler& sampler, PathCounters& counters, Vec3& radiance) noexcept
{
    const LightSet& lights = scene.lights;
    const bool lightSamplingForAreas = settings.strategy != LightStrategy::BsdfOnly;
    const bool clampIndirect = settings.indirectClamp > 0.0f && std::isfinite (settings.indirectClamp);

    Vec3 sum {};
    Vec3 throughput { 1.0f, 1.0f, 1.0f };
    bool previousDelta = true;  // the camera acts as a delta "lobe"
    float previousPdf = 0.0f;
    std::uint32_t depth = 0;            // bounces taken
    std::uint32_t vertex = 0;           // surface interactions, including alpha crossings
    std::uint32_t transparentHits = 0;
    bool finite = true;
    Ray ray = cameraRay;

    // bounces is the number of scattering events the contribution went through.
    const auto add = [&] (Vec3 contribution, std::uint32_t bounces) {
        if (! isFinite (contribution))
        {
            finite = false;
            return;
        }
        if (clampIndirect && bounces >= 2)
        {
            const float peak = maxComponent (contribution);
            if (peak > settings.indirectClamp)
            {
                contribution = contribution * (settings.indirectClamp / peak);
                ++counters.clampedContributions;
            }
        }
        sum = sum + contribution;
    };

    while (finite)
    {
        Hit hit;
        ++counters.closestRays;
        if (! scene.bvh.intersect (ray, hit))
        {
            if (lights.hasEnvironment() && (depth > 0 || settings.environmentVisibleToCamera))
            {
                const Vec3 direction = normalizeOrZero (ray.direction);
                const Vec3 emitted = lights.environmentRadiance (direction);
                const float weight = previousDelta
                                       ? 1.0f
                                       : emitterHitWeight (settings.strategy, previousPdf, lights.environmentPdf (direction));
                if (weight > 0.0f && anyPositive (emitted))
                    add (throughput * emitted * weight, depth);
            }
            break;
        }

        const SurfacePoint point = surfaceAt (scene, hit);
        const Material& material = scene.materialOf (hit.triangle);
        const float coverage = materialCoverage (material);
        const Vec3 direction = normalizeOrZero (ray.direction);

        // Emission from either face.
        if (coverage > 0.0f && (material.emission.x != 0.0f || material.emission.y != 0.0f || material.emission.z != 0.0f))
        {
            float weight = 1.0f;
            if (! previousDelta)
            {
                const float distance = hit.t * length (ray.direction);
                const float cosAtLight = std::fabs (dot (point.geometricNormal, direction));
                weight = emitterHitWeight (settings.strategy, previousPdf,
                                           lights.trianglePdf (hit.triangle, distance, cosAtLight));
            }
            if (weight > 0.0f)
                add (throughput * material.emission * (coverage * weight), depth);
        }

        // Alpha cut-outs: cross with probability 1 - coverage, keeping the
        // path state, so the expected result is the coverage blend.
        if (coverage < 1.0f)
        {
            const float u = sampler.get1D (dimensionOf (vertex, kSlotTransparency));
            if (u >= coverage)
            {
                if (++transparentHits > settings.maxTransparentHits)
                {
                    ++counters.transparentLimitPaths;
                    break;
                }
                ++vertex;
                ray.origin = spawnRayOrigin (point.position, point.geometricNormal, ray.direction);
                if (std::isfinite (ray.tMax))
                    ray.tMax = std::max (0.0f, ray.tMax - hit.t);
                ray.tMin = 0.0f;
                ray.ignoreTriangle = hit.triangle;
                continue;
            }
        }

        if (depth >= settings.maxDepth)
            break;

        const Vec3 wo = -direction;
        const bool front = dot (point.geometricNormal, wo) >= 0.0f;
        const Vec3 ng = front ? point.geometricNormal : -point.geometricNormal;
        Vec3 ns = front ? point.shadingNormal : -point.shadingNormal;
        if (dot (ns, wo) <= 1.0e-4f)
            ns = ng;
        Bsdf bsdf;
        buildMaterialBsdf (material, ns, ng, wo, front, bsdf);
        if (bsdf.lobeCount() == 0)
            break;

        // Next event estimation: one light sample.
        if (bsdf.hasNonDelta() && lights.size() > 0)
        {
            LightSample light;
            if (lights.sample (point.position, sampler.get1D (dimensionOf (vertex, kSlotLightSelect)),
                               sampler.get2D (dimensionOf (vertex, kSlotLightPosition)), light)
                && (light.delta || lightSamplingForAreas) && anyPositive (light.radiance))
            {
                const BsdfEval eval = bsdf.eval (wo, light.wi);
                if (anyPositive (eval.value))
                {
                    Ray shadow;
                    shadow.origin = spawnRayOrigin (point.position, ng, light.wi);
                    shadow.ignoreTriangle = hit.triangle;
                    shadow.tMin = 0.0f;
                    if (light.infinite)
                    {
                        shadow.direction = light.wi;
                        shadow.tMax = std::numeric_limits<float>::infinity();
                    }
                    else
                    {
                        Vec3 target = light.point;
                        if (light.area)
                        {
                            const Vec3 facing = dot (light.lightNormal, light.wi) < 0.0f ? light.lightNormal : -light.lightNormal;
                            target = offsetRayOrigin (light.point, facing);
                        }
                        shadow.direction = target - shadow.origin;
                        shadow.tMax = 1.0f;
                    }
                    const float transmittance = shadowTransmittance (scene, shadow, counters);
                    if (transmittance > 0.0f)
                    {
                        const float weight = light.delta || settings.strategy != LightStrategy::MultipleImportance
                                               ? 1.0f
                                               : powerHeuristic (light.pdf, eval.pdf);
                        add (throughput * eval.value * light.radiance * (transmittance * weight / light.pdf), depth + 1);
                    }
                }
            }
        }

        // Russian roulette on the path throughput.
        if (depth >= settings.rouletteStartDepth)
        {
            const float survive = std::min (1.0f, std::sqrt (std::max (0.0f, maxComponent (throughput))));
            if (! (survive > 0.0f))
                break;
            if (survive < 1.0f)
            {
                if (sampler.get1D (dimensionOf (vertex, kSlotRoulette)) >= survive)
                    break;
                throughput = throughput * (1.0f / survive);
            }
        }

        BsdfSample scattered;
        if (! bsdf.sample (wo, sampler.get1D (dimensionOf (vertex, kSlotLobe)),
                           sampler.get2D (dimensionOf (vertex, kSlotDirection)), scattered))
            break;
        throughput = throughput * scattered.weight;
        if (! isFinite (throughput))
        {
            finite = false;
            break;
        }
        if (! anyPositive (throughput))
            break;
        previousDelta = (scattered.flags & kBsdfDelta) != 0;
        previousPdf = scattered.pdf;

        ray = Ray {};
        ray.origin = spawnRayOrigin (point.position, ng, scattered.wi);
        ray.direction = scattered.wi;
        ray.tMin = 0.0f;
        ray.tMax = std::numeric_limits<float>::infinity();
        ray.ignoreTriangle = hit.triangle;
        ++depth;
        ++vertex;
    }

    radiance = sum;
    return finite && isFinite (sum);
}

PathTracer::PathTracer (unsigned threadCount) : pool_ (threadCount) {}

RenderStats PathTracer::renderProgressive (const RenderScene& scene, const PinholeCamera& camera,
                                           std::uint32_t width, std::uint32_t height, std::uint32_t samplesPerPass,
                                           std::uint64_t generation, const CancellationToken& cancel)
{
    if (width == 0 || height == 0 || camera.width != width || camera.height != height || samplesPerPass == 0)
    {
        RenderStats stats;
        stats.invalidRequest = true;
        stats.indirectClamp = settings.indirectClamp;
        return stats;
    }
    if (accumulator_.width() != width || accumulator_.height() != height)
    {
        accumulator_.resize (width, height);
        accumulator_.reset (generation);
        nextSample_ = 0;
    }
    else if (accumulator_.generation() != generation)
    {
        accumulator_.reset (generation);
        nextSample_ = 0;
    }
    const std::uint32_t first = nextSample_;
    // Advance even if the pass is cancelled, so no index is ever reused.
    nextSample_ += samplesPerPass;
    return runPass (scene, camera, first, samplesPerPass, cancel);
}

RenderStats PathTracer::renderFixed (const RenderScene& scene, const PinholeCamera& camera,
                                     std::uint32_t width, std::uint32_t height, std::uint32_t samplesPerPixel,
                                     std::uint64_t generation, const CancellationToken& cancel)
{
    if (width == 0 || height == 0 || camera.width != width || camera.height != height || samplesPerPixel == 0)
    {
        RenderStats stats;
        stats.invalidRequest = true;
        stats.indirectClamp = settings.indirectClamp;
        return stats;
    }
    accumulator_.resize (width, height);
    accumulator_.reset (generation);
    nextSample_ = samplesPerPixel;
    return runPass (scene, camera, 0, samplesPerPixel, cancel);
}

RenderStats PathTracer::runPass (const RenderScene& scene, const PinholeCamera& camera,
                                 std::uint32_t firstSample, std::uint32_t samples, const CancellationToken& cancel)
{
    RenderStats stats;
    stats.indirectClamp = settings.indirectClamp > 0.0f && std::isfinite (settings.indirectClamp) ? settings.indirectClamp : 0.0f;
    stats.samplesPerPixel = samples;
    stats.firstSampleIndex = firstSample;
    stats.lights = scene.lights.stats();

    struct ThreadTotals
    {
        PathCounters counters {};
        std::uint64_t samplesAdded = 0;
        std::uint64_t nonFinite = 0;
        bool stoppedEarly = false;
    };
    std::vector<ThreadTotals> totals (pool_.concurrency());
    const std::uint32_t width = accumulator_.width();
    const std::uint64_t generation = accumulator_.generation();
    const IntegratorSettings config = settings;
    const TileGrid grid (accumulator_.width(), accumulator_.height(), config.tileSize);

    const TilePassResult result = pool_.runTiles (grid, cancel, [&] (const Tile& tile, unsigned threadIndex) {
        ThreadTotals local;
        for (std::uint32_t y = tile.y0; y < tile.y1; ++y)
        {
            // Long fixed-count tiles still stop promptly.
            if (cancel.isCancelled())
            {
                local.stoppedEarly = true;
                break;
            }
            for (std::uint32_t x = tile.x0; x < tile.x1; ++x)
            {
                const std::uint32_t pixel = y * width + x;
                for (std::uint32_t s = 0; s < samples; ++s)
                {
                    const PathSampler sampler (pixel, firstSample + s, generation, config.pattern);
                    const Vec2 jitter = sampler.get2D (kPixelFilterDimension);
                    const Ray ray = camera.generateRay (static_cast<float> (x) + jitter.x,
                                                        static_cast<float> (y) + jitter.y);
                    Vec3 radiance;
                    if (! tracePath (scene, config, ray, sampler, local.counters, radiance))
                    {
                        ++local.nonFinite;
                        continue;
                    }
                    accumulator_.addSample (x, y, { radiance.x, radiance.y, radiance.z });
                    ++local.samplesAdded;
                }
            }
        }
        ThreadTotals& slot = totals[threadIndex];
        slot.counters += local.counters;
        slot.samplesAdded += local.samplesAdded;
        slot.nonFinite += local.nonFinite;
        slot.stoppedEarly = slot.stoppedEarly || local.stoppedEarly;
    });

    bool stoppedEarly = false;
    PathCounters counters;
    for (const ThreadTotals& slot : totals)
    {
        counters += slot.counters;
        stats.samplesAdded += slot.samplesAdded;
        stats.nonFiniteSamples += slot.nonFinite;
        stoppedEarly = stoppedEarly || slot.stoppedEarly;
    }
    stats.closestRays = counters.closestRays;
    stats.shadowRays = counters.shadowRays;
    stats.clampedContributions = counters.clampedContributions;
    stats.transparentLimitPaths = counters.transparentLimitPaths;
    stats.tileCount = result.tileCount;
    stats.tilesCompleted = result.tilesCompleted;
    stats.cancelled = result.cancelled || stoppedEarly;
    return stats;
}

} // namespace videohelper::pathtrace
