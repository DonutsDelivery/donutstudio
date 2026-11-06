#include "integrator.h"
#include "../diffraction_material_admission.h"
#include "devices/device_cpu.h"

#include <algorithm>
#include <limits>
#include <utility>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace videohelper::pathtrace
{
namespace
{

using HarmonicMIDI::grid::SceneAlphaMode;
using HarmonicMIDI::grid::SceneMaterialRecord;
using HarmonicMIDI::grid::SceneVolume;
using HarmonicMIDI::grid::SceneVolumeField;
using HarmonicMIDI::grid::sceneVolumeActive;
using HarmonicMIDI::grid::Visual3DScene;

float finiteOr (float value, float fallback) noexcept { return std::isfinite (value) ? value : fallback; }

} // namespace

PathCounters& PathCounters::operator+= (const PathCounters& other) noexcept
{
    closestRays += other.closestRays;
    shadowRays += other.shadowRays;
    clampedContributions += other.clampedContributions;
    bounceLimitPaths += other.bounceLimitPaths;
    transparentLimitPaths += other.transparentLimitPaths;
    lightPaths += other.lightPaths;
    lightPathRays += other.lightPathRays;
    causticSplats += other.causticSplats;
    for (std::size_t i = 0; i < causticSplatsByLight.size(); ++i)
        causticSplatsByLight[i] += other.causticSplatsByLight[i];
    return *this;
}

void RenderScene::commit()
{
    const auto bvhStart = PhaseClock::now();
    if (geometry.motionOpen.empty())
        bvh.build (geometry.positions);
    else
        bvh.buildMotion (geometry.motionOpen.data(), geometry.motionClose.data(), geometry.triangleCount());
    bvhBuildMs = millisecondsSince (bvhStart);
    // Every authored size lights the scene (LightSet::build reads the records).
    // Gratings have delta orders only, so an unsized sun gets a small disk in a
    // grating scene: see kGratingSourceAngularRadius.
    if (! gratings.empty())
    {
        if (lightShapes.size() < lightRecords.size())
            lightShapes.resize (lightRecords.size());
        for (std::size_t i = 0; i < lightRecords.size(); ++i)
            if (lightRecords[i].kind == HarmonicMIDI::grid::SceneLightKind::Directional
                && ! (lightShapes[i].angularRadius > 0.0f) && ! (lightRecords[i].angularRadius > 0.0f))
                lightShapes[i].angularRadius = kGratingSourceAngularRadius;
    }
    lights.build (lightRecords.data(), lightRecords.size(), ambient, geometry, materials, fallbackMaterial,
                  &environmentImage, &lightShapes, &areaLights);
    const auto note = [this] (const Material& material) {
        anyPartialCoverage = anyPartialCoverage || materialCoverage (material) < 1.0f;
        anyThinSheet = anyThinSheet || isThinSheet (material);
        anyMediumBoundary = anyMediumBoundary || isMediumBoundary (material);
        anyDispersive = anyDispersive || materialDispersive (material);
        anyVolume = anyVolume || materialHasVolume (material);
    };
    anyPartialCoverage = false;
    anyThinSheet = false;
    anyMediumBoundary = false;
    anyDispersive = false;
    anyVolume = false;
    note (fallbackMaterial);
    for (const Material& material : materials)
        note (material);
    anyHiddenObjects = false;
    for (const TriangleAttributes& triangle : geometry.triangles)
        anyHiddenObjects = anyHiddenObjects || triangle.visibility != HarmonicMIDI::grid::sceneRayVisibility::All;
    boundsCentre = {};
    boundsRadius = 0.0f;
    if (! geometry.positions.empty())
    {
        Vec3 lo = geometry.positions.front();
        Vec3 hi = lo;
        for (const Vec3& p : geometry.positions)
        {
            lo = { std::min (lo.x, p.x), std::min (lo.y, p.y), std::min (lo.z, p.z) };
            hi = { std::max (hi.x, p.x), std::max (hi.y, p.y), std::max (hi.z, p.z) };
        }
        boundsCentre = (lo + hi) * 0.5f;
        // A little margin so rays from the disk start outside every triangle.
        boundsRadius = length (hi - lo) * 0.5f * 1.01f + 1.0e-3f;
        if (! isFinite (boundsCentre) || ! std::isfinite (boundsRadius))
        {
            boundsCentre = {};
            boundsRadius = 0.0f;
        }
    }
    // The world volume needs a ball to fill: four scene radii unless given.
    worldVolumeExtent = std::isfinite (worldVolumeRadius) && worldVolumeRadius > 0.0f ? worldVolumeRadius : 4.0f * boundsRadius;
    hasWorldVolume = materialHasVolume (worldVolume) && worldVolumeExtent > 0.0f;
    if (hasWorldVolume)
    {
        // A medium of its own with no surface; a Noise field still shapes it.
        worldVolume.transmission = 0.0f;
        worldVolume.absorptionDistance = 0.0f;
    }
    anyVolume = anyVolume || hasWorldVolume;
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
    applySceneVolume (record.volume, material);
    return material;
}

void applySceneVolume (const SceneVolume& volume, Material& material) noexcept
{
    const auto unit = [] (float value) {
        return std::isfinite (value) ? std::min (1.0f, std::max (0.0f, value)) : 0.0f;
    };
    const auto radiance = [] (float value) {
        return std::isfinite (value) && value > 0.0f ? value : 0.0f;
    };
    material.volumeDensity = radiance (volume.density);
    material.volumeColor = { unit (volume.color.x), unit (volume.color.y), unit (volume.color.z) };
    material.volumeAbsorptionColor = { unit (volume.absorptionColor.x), unit (volume.absorptionColor.y),
                                       unit (volume.absorptionColor.z) };
    material.volumeAnisotropy = std::isfinite (volume.anisotropy)
        ? std::min (1.0f, std::max (-1.0f, volume.anisotropy)) : 0.0f;
    material.volumeEmission = { radiance (volume.emission.x), radiance (volume.emission.y), radiance (volume.emission.z) };
    material.volumeField = volume.field == SceneVolumeField::Noise ? VolumeDensityField::Noise : VolumeDensityField::Uniform;
    material.volumeFieldDetail = static_cast<std::uint8_t> (std::min<std::uint32_t> (8, std::max<std::uint32_t> (1, volume.fieldDetail)));
    material.volumeFieldScale = std::isfinite (volume.fieldScale) && volume.fieldScale > 0.0f ? volume.fieldScale : 1.0f;
    material.volumeFieldOffset = { finiteOr (volume.fieldOffset.x, 0.0f), finiteOr (volume.fieldOffset.y, 0.0f),
                                   finiteOr (volume.fieldOffset.z, 0.0f) };
    material.volumeOnly = volume.volumeOnly && material.volumeDensity > 0.0f;
}

void prepareRenderScene (const Visual3DScene& scene, RenderScene& out, const PrepareOptions& options)
{
    ingestScene (scene, out.geometry, options.ingest);
    out.materials.clear();
    const std::size_t materialCount = std::min (scene.materialCount, scene.materials.size());
    for (std::size_t i = 0; i < materialCount; ++i)
        out.materials.push_back (materialFromRecord (scene.materials[i]));
    out.fallbackMaterial = Material {};
    // An explicit world volume from the caller wins; otherwise the scene's fog.
    out.worldVolume = options.worldVolume;
    out.worldVolumeRadius = options.worldVolumeRadius;
    if (! materialHasVolume (out.worldVolume) && sceneVolumeActive (scene.worldVolume))
    {
        out.worldVolume = Material {};
        applySceneVolume (scene.worldVolume, out.worldVolume);
        out.worldVolumeRadius = std::isfinite (scene.worldVolume.radius) ? scene.worldVolume.radius : 0.0f;
    }
    // Object overrides append their materials after the record materials and
    // point that object's triangles at them.
    for (const ObjectMaterialOverride& entry : options.objectMaterials)
    {
        const auto index = static_cast<std::uint32_t> (out.materials.size());
        out.materials.push_back (entry.material);
        for (TriangleAttributes& triangle : out.geometry.triangles)
            if (triangle.objectId == entry.objectId)
                triangle.materialIndex = index;
    }
    // Gratings get a material slot of their own, which only the binding reads.
    out.gratingMaterials.clear();
    out.gratings.clear();
    for (const ObjectGratingBinding& entry : options.objectGratings)
    {
        if (entry.material == nullptr)
            continue;
        device::GratingBinding binding { 0u, entry.material->description() };
        if (entry.field.has_value())
        {
            binding.field = *entry.field;
            for (const auto& corner : binding.field)
            {
                binding.varied = binding.varied || corner[0] != binding.field[0][0] || corner[1] != binding.field[0][1]
                              || corner[2] != binding.field[0][2];
                binding.masked = binding.masked || corner[3] < 1.0f;
            }
        }
        out.gratingMaterials.push_back (entry.material);
        if (! binding.masked)
        {
            binding.materialIndex = static_cast<std::uint32_t> (out.materials.size());
            out.materials.push_back (Material {});
            for (TriangleAttributes& triangle : out.geometry.triangles)
                if (triangle.objectId == entry.objectId)
                    triangle.materialIndex = binding.materialIndex;
            out.gratings.push_back (binding);
            continue;
        }
        // Where the mask leaves the foil out the surface is the object's own material, so
        // each of the object's materials gets a slot that keeps it.
        std::vector<std::pair<std::uint32_t, std::uint32_t>> slots;
        for (TriangleAttributes& triangle : out.geometry.triangles)
        {
            if (triangle.objectId != entry.objectId)
                continue;
            const std::uint32_t own = triangle.materialIndex;
            const auto slot = std::find_if (slots.begin(), slots.end(), [own] (const auto& s) { return s.first == own; });
            if (slot != slots.end())
            {
                triangle.materialIndex = slot->second;
                continue;
            }
            const Material base = own < out.materials.size() ? out.materials[own] : out.fallbackMaterial;
            binding.materialIndex = static_cast<std::uint32_t> (out.materials.size());
            out.materials.push_back (base);
            out.gratings.push_back (binding);
            slots.push_back ({ own, binding.materialIndex });
            triangle.materialIndex = binding.materialIndex;
        }
    }
    out.frameBindings = options.objectFrames;
    const float gain = std::isfinite (options.emissionGain) && options.emissionGain > 0.0f ? options.emissionGain : 0.0f;
    if (gain != 1.0f)
    {
        // A gain past a float keeps the hue at the largest float instead of dropping the emitter.
        const auto gained = [gain] (Vec3 c) noexcept {
            const double x = static_cast<double> (c.x) * gain, y = static_cast<double> (c.y) * gain, z = static_cast<double> (c.z) * gain;
            const double peak = std::max ({ x, y, z }), limit = std::numeric_limits<float>::max();
            const double fit = peak > limit ? limit / peak : 1.0;
            return Vec3 { static_cast<float> (x * fit), static_cast<float> (y * fit), static_cast<float> (z * fit) };
        };
        for (Material& material : out.materials)
            material.emission = gained (material.emission);
        out.fallbackMaterial.emission = gained (out.fallbackMaterial.emission);
    }
    if (options.lightRecords != nullptr)
        out.lightRecords = *options.lightRecords;
    else
    {
        const std::size_t lightCount = std::min (scene.lightCount, scene.lights.size());
        out.lightRecords.assign (scene.lights.begin(), scene.lights.begin() + static_cast<std::ptrdiff_t> (lightCount));
    }
    out.lightShapes = options.lightShapes;
    out.areaLights = options.areaLights;
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

bool usesSpectralTransport (const RenderScene& scene, const IntegratorSettings& settings) noexcept
{
    switch (settings.spectral)
    {
        case SpectralMode::Off:
            return false;
        case SpectralMode::Automatic:
            return scene.anyDispersive || ! scene.gratings.empty();
        case SpectralMode::On:
            return true;
    }
    return false;
}

bool tracePath (const RenderScene& scene, const IntegratorSettings& settings, const Ray& cameraRay,
                const PathSampler& sampler, PathCounters& counters, Vec3& radiance) noexcept
{
    return device::cpu::tracePath (device::cpu::makeSceneView (scene), settings, cameraRay, sampler, counters, radiance);
}

PathTracer::PathTracer (unsigned threadCount)
    : ownedPool_ (threadCount == 0 ? nullptr : std::make_unique<ThreadPool> (threadCount)),
      pool_ (ownedPool_ != nullptr ? *ownedPool_ : sharedThreadPool())
{
}

RenderStats PathTracer::renderProgressive (const RenderScene& scene, const PinholeCamera& camera,
                                           std::uint32_t width, std::uint32_t height, std::uint32_t samplesPerPass,
                                           std::uint64_t generation, const CancellationToken& cancel,
                                           std::uint64_t samplerGeneration)
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
    if (nextSample_ == 0)
        frameLayers_.clear();
    const std::uint32_t first = nextSample_;
    // Advance even if the pass is cancelled, so no index is ever reused.
    nextSample_ += samplesPerPass;
    samplerGeneration_ = samplerGeneration != 0 ? samplerGeneration : generation;
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
    frameLayers_.clear();
    nextSample_ = samplesPerPixel;
    samplerGeneration_ = generation;
    return runPass (scene, camera, 0, samplesPerPixel, cancel);
}

TilePassResult PathTracer::forEachTile (std::uint32_t width, std::uint32_t height, const CancellationToken& cancel,
                                        const ThreadPool::TileFunction& fn)
{
    const TileGrid grid (width, height, settings.tileSize);
    return pool_.runTiles (grid, cancel, fn);
}

RenderStats PathTracer::runPass (const RenderScene& scene, const PinholeCamera& camera,
                                 std::uint32_t firstSample, std::uint32_t samples, const CancellationToken& cancel)
{
    RenderStats stats;
    stats.indirectClamp = settings.indirectClamp > 0.0f && std::isfinite (settings.indirectClamp) ? settings.indirectClamp : 0.0f;
    stats.samplesPerPixel = samples;
    stats.firstSampleIndex = firstSample;
    stats.lights = scene.lights.stats();
    stats.spectral = usesSpectralTransport (scene, settings);

    struct ThreadTotals
    {
        PathCounters counters {};
        std::uint64_t samplesAdded = 0;
        std::uint64_t nonFinite = 0;
        bool stoppedEarly = false;
    };
    std::vector<ThreadTotals> totals (pool_.concurrency());
    const std::uint32_t width = accumulator_.width();
    // The accumulation's own generation unless its pass seeded another one.
    const std::uint64_t generation = samplerGeneration_ != 0 ? samplerGeneration_ : accumulator_.generation();
    const IntegratorSettings config = settings;
    const TileGrid grid (accumulator_.width(), accumulator_.height(), config.tileSize);
    const device::cpu::SceneView view = device::cpu::makeSceneView (scene);

    std::size_t layerCount = 0;
    for (const ObjectFrameBinding& binding : scene.frameBindings)
        layerCount = std::max (layerCount, static_cast<std::size_t> (binding.layer) + 1u);
    if (frameLayers_.size() != layerCount)
    {
        frameLayers_.resize (layerCount);
        for (FrameLayerFilm& layer : frameLayers_)
            layer.resize (accumulator_.width(), accumulator_.height());
    }
    // A sample whose camera ray first hits a frame-tinted object goes to that
    // frame's layer. The accumulator counts it as a black sample, so the two
    // films sum to the pixel's mean.
    const auto addFrameSample = [&] (std::uint32_t x, std::uint32_t y, const Ray& ray, Vec3 radiance) {
        Hit hit;
        if (scene.bvh.intersect (ray, hit))
        {
            const TriangleAttributes& triangle = scene.geometry.triangles[hit.triangle];
            for (const ObjectFrameBinding& binding : scene.frameBindings)
                if (binding.objectId == triangle.objectId)
                {
                    const float w = 1.0f - hit.u - hit.v;
                    accumulator_.addSample (x, y, {});
                    frameLayers_[binding.layer].add (x, y, { radiance.x, radiance.y, radiance.z },
                        triangle.uvs[0].x * w + triangle.uvs[1].x * hit.u + triangle.uvs[2].x * hit.v,
                        triangle.uvs[0].y * w + triangle.uvs[1].y * hit.u + triangle.uvs[2].y * hit.v);
                    return;
                }
        }
        accumulator_.addSample (x, y, { radiance.x, radiance.y, radiance.z });
    };

    if (devicePass_ && layerCount != 0 && deviceSceneFallback_)
        stats.deviceSkipped = "because frame-tinted objects in this scene draw on the CPU only so far.";
    else if (devicePass_)
    {
        if (layerCount != 0)
        {
            deviceFailure_ = "frame-tinted objects have no device film yet, so this device cannot render this scene";
            stats.deviceFailed = true;
            return stats;
        }
        PathCounters counters;
        DevicePass pass { scene, camera, config, firstSample, samples, generation, cancel, accumulator_, stats, counters, deviceFailure_ };
        deviceFailure_.clear();
        if (! devicePass_ (pass))
        {
            stats.deviceFailed = true;
            return stats;
        }
        stats.closestRays = counters.closestRays;
        stats.shadowRays = counters.shadowRays;
        stats.clampedContributions = counters.clampedContributions;
        stats.bounceLimitPaths = counters.bounceLimitPaths;
        stats.transparentLimitPaths = counters.transparentLimitPaths;
        stats.limitEndedPaths = counters.bounceLimitPaths + counters.transparentLimitPaths;
        stats.lightPaths = counters.lightPaths;
        stats.lightPathRays = counters.lightPathRays;
        stats.causticSplats = counters.causticSplats;
        stats.causticSplatsByLight = counters.causticSplatsByLight;
        stats.tileCount = stats.tilesCompleted = 1;
        return stats;
    }

    // The CPU traces into host memory: it has no pipeline, upload or readback,
    // and its BVH is the scene's (the engine times that as AccelBuild).
    stats.phases.markAbsent (FramePhase::PipelineWait, "the CPU device has no kernel build");
    stats.phases.markAbsent (FramePhase::Upload, "the CPU device traces from host memory");
    stats.phases.markAbsent (FramePhase::Readback, "the CPU device accumulates into host memory");
    const auto traceStart = PhaseClock::now();
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
                for (std::uint32_t s = 0; s < samples; ++s)
                {
                    device::SampleOutcome outcome;
                    if (layerCount == 0)
                        outcome = device::cpu::renderSample (view, config, camera, x, y, width, firstSample + s, generation,
                                                             accumulator_, local.counters);
                    else
                    {
                        Ray ray;
                        Vec3 radiance;
                        outcome = device::cpu::traceCameraSample (view, config, camera, x, y, width, firstSample + s,
                                                                  generation, accumulator_, local.counters, ray, radiance);
                        if (outcome == device::SampleOutcome::Added)
                        {
                            addFrameSample (x, y, ray, radiance);
                            Vec3 albedo;
                            Vec3 normal;
                            device::cpu::firstHitGuides (view, ray, albedo, normal);
                            accumulator_.addGuide (x, y, { albedo.x, albedo.y, albedo.z }, { normal.x, normal.y, normal.z });
                        }
                    }
                    if (outcome == device::SampleOutcome::NonFinite)
                        ++local.nonFinite;
                    else
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
    stats.phases.set (FramePhase::Trace, millisecondsSince (traceStart));

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
    stats.bounceLimitPaths = counters.bounceLimitPaths;
    stats.transparentLimitPaths = counters.transparentLimitPaths;
    stats.limitEndedPaths = counters.bounceLimitPaths + counters.transparentLimitPaths;
    stats.lightPaths = counters.lightPaths;
    stats.lightPathRays = counters.lightPathRays;
    stats.causticSplats = counters.causticSplats;
    stats.causticSplatsByLight = counters.causticSplatsByLight;
    stats.tileCount = result.tileCount;
    stats.tilesCompleted = result.tilesCompleted;
    stats.cancelled = result.cancelled || stoppedEarly;
    return stats;
}

} // namespace videohelper::pathtrace
