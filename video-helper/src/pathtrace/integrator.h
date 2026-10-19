#pragma once

// Unidirectional path tracer over the PT-A1 core (ingest, BVH, tile pool,
// accumulator).
//
// Each vertex with a non-delta lobe takes one light sample (next event
// estimation, light chosen by power). Area and environment lights combine
// light sampling and BSDF sampling with the power heuristic; delta lights are
// reached by light sampling only and delta lobes by BSDF sampling only.
// Russian roulette starts after rouletteStartDepth bounces with continuation
// probability min(1, sqrt(max throughput)). Partially covered surfaces
// (Mask/Blend alpha) are crossed stochastically without counting a bounce, and
// shadow rays multiply their transmittance.
//
// Nothing is clamped by default. A sample whose radiance is NaN or infinite
// is dropped before it reaches the accumulator and counted in
// RenderStats::nonFiniteSamples. The optional indirect clamp is off unless
// IntegratorSettings::indirectClamp > 0, and RenderStats reports both its
// value and how many contributions it changed.
//
// Output is linear scene-referred RGB in the Accumulator. Pixel samples use a
// box filter over the pixel footprint.
//
// Transport channels (spectrum.h). In RGB mode a path carries linear sRGB. In
// spectral mode it carries four wavelengths, a hero and three companions,
// combined by hero wavelength MIS: the path is sampled with the hero's
// indices, the throughput holds each channel's value over the hero's density,
// and each contribution is divided by the mean ratio of the channels'
// densities to the hero's. A dispersive refraction (or a rough dispersive
// interface) leaves the companions at 0, after which only the hero counts,
// with weight 4 (Wilkie et al. 2014; Cycles path_state.h does the same after
// dispersion). SpectralMode::Automatic, the default, uses spectral transport
// only when some material is dispersive, so every other scene renders exactly
// as in RGB mode.
//
// Media (media.h). Refracting materials push and pop a medium stack. Real
// interfaces refract with the ratio of the indices on both sides, false ones
// (nested media with priorities) are crossed without scattering, and the
// owning medium absorbs by Beer-Lambert along every segment
// (Material::absorptionDistance). Thin sheets transmit straight, which camera,
// bounce and shadow rays all follow. Shadow rays still stop at real
// refracting interfaces: light seen through glass arrives by BSDF sampling
// only (caustics are PT-E).
//
// Lights with a size (lights.h) are invisible to camera rays. Bounce rays
// reach them, and MIS pairs those hits with light sampling. The MIS densities
// of an emitter are measured from the last scattering vertex, so alpha
// cut-outs, false interfaces and thin sheets in between do not change them.
//
// Behavioural reference (studied, not copied): Blender Cycles
// intern/cycles/kernel/integrator/shade_surface.h, shade_background.h,
// shade_light.h, intersect_closest.h, path_state.h, volume_stack.h and
// intern/cycles/kernel/light/sample.h, intern/cycles/kernel/sample/mis.h,
// intern/cycles/kernel/geom/shader_data.h (wavelength setup).

#include "accumulator.h"
#include "bsdf.h"
#include "bvh.h"
#include "environment_map.h"
#include "lights.h"
#include "media.h"
#include "sampler.h"
#include "spectrum.h"
#include "scene_ingest.h"
#include "thread_pool.h"
#include "../../../shared/Visual3DScene.h"

#include <cstdint>
#include <functional>
#include <vector>

namespace videohelper::pathtrace
{

enum class LightStrategy : std::uint8_t
{
    MultipleImportance = 0,  // light and BSDF sampling, power heuristic
    LightOnly = 1,           // emitters only through light samples (where possible)
    BsdfOnly = 2             // area and environment light only through BSDF rays
};

enum class SpectralMode : std::uint8_t
{
    Off = 0,        // RGB channels
    Automatic = 1,  // spectral when the scene has a dispersive material
    On = 2          // always spectral
};

struct IntegratorSettings
{
    std::uint32_t maxDepth = 12;            // bounces after the camera hit
    std::uint32_t rouletteStartDepth = 3;   // first bounce that may be terminated
    std::uint32_t maxTransparentHits = 16;  // alpha cut-outs crossed per path
    // 0 disables. Otherwise the largest channel of every indirect contribution
    // (two or more bounces) is limited to this value; RenderStats counts them.
    float indirectClamp = 0.0f;
    LightStrategy strategy = LightStrategy::MultipleImportance;
    SpectralMode spectral = SpectralMode::Automatic;
    SamplePattern pattern = SamplePattern::OwenSobol;
    bool environmentVisibleToCamera = true;
    std::uint32_t tileSize = TileGrid::kDefaultTileSize;
};

// Geometry, materials and lights the integrator reads. Fill the public
// members (or call prepareRenderScene) and then commit(); the integrator never
// writes to it, so any number of renders may share one instance.
struct RenderScene
{
    IngestedScene geometry;
    // Indexed by TriangleAttributes::materialIndex. Triangles without a
    // material, or with an index past the end, use fallbackMaterial.
    std::vector<Material> materials;
    Material fallbackMaterial {};
    std::vector<HarmonicMIDI::grid::SceneLightRecord> lightRecords;
    Vec3 ambient {};  // added to the environment radiance
    // Decoded pixels of the environment image record whose asset they name.
    // Empty when the scene has none or nothing resolved them.
    EnvironmentPixels environmentImage;
    // Size of lightRecords[i] at index i; shorter (or empty) means delta lights.
    std::vector<LightShape> lightShapes;
    // Rect and disk lights next to the records.
    std::vector<AreaLightRecord> areaLights;

    // Built by commit().
    Bvh bvh;
    LightSet lights;
    bool anyPartialCoverage = false;
    bool anyThinSheet = false;
    bool anyMediumBoundary = false;
    bool anyDispersive = false;

    // Rebuilds the BVH and the light set from the members above.
    void commit();
    const Material& materialOf (std::uint32_t triangle) const noexcept;
};

// A material for every triangle of one object, in place of its record
// material. The helper maps a surface material program's PBR block this way.
struct ObjectMaterialOverride
{
    std::uint32_t objectId = 0;  // SceneObjectId::value
    Material material {};
};

struct PrepareOptions
{
    IngestOptions ingest {};
    // Applied in order; a later entry for the same object wins.
    std::vector<ObjectMaterialOverride> objectMaterials;
    // When set, these records light the scene instead of scene.lights, e.g.
    // the records a runtime light override produced. Must outlive the call.
    const std::vector<HarmonicMIDI::grid::SceneLightRecord>* lightRecords = nullptr;
    // Sizes of the scene's (or lightRecords') lights, by record index, and
    // extra rect and disk lights. Empty keeps every light a delta light.
    std::vector<LightShape> lightShapes;
    std::vector<AreaLightRecord> areaLights;
    // Multiplies every material's emission, like the raster's emission gain.
    float emissionGain = 1.0f;
    // The raster adds Visual3DScene::ambientColor to its ambient term; the
    // tracer treats it as extra constant environment radiance.
    bool ambientAsEnvironment = true;
    // Returns the decoded pixels of an environment image record. Unset, image
    // records light nothing (LightSetStats::ignoredEnvironmentImages). The
    // helper passes residentEnvironmentPixels (environment_resident.h), which
    // shares the store's decoded pixels instead of decoding again.
    std::function<EnvironmentPixels (const HarmonicMIDI::grid::SceneEnvironmentImage&)> environmentImageSource;
};

Material materialFromRecord (const HarmonicMIDI::grid::SceneMaterialRecord& record) noexcept;
// Ingests the scene, converts its materials and lights, and commits.
void prepareRenderScene (const HarmonicMIDI::grid::Visual3DScene& scene, RenderScene& out,
                         const PrepareOptions& options = {});

// Per-thread path counters, summed into RenderStats.
struct PathCounters
{
    std::uint64_t closestRays = 0;
    std::uint64_t shadowRays = 0;
    std::uint64_t clampedContributions = 0;
    std::uint64_t transparentLimitPaths = 0;

    PathCounters& operator+= (const PathCounters& other) noexcept;
};

struct RenderStats
{
    std::uint64_t samplesAdded = 0;
    std::uint64_t nonFiniteSamples = 0;    // dropped, never accumulated
    std::uint64_t clampedContributions = 0;
    float indirectClamp = 0.0f;            // the setting in force; 0 is off
    std::uint64_t closestRays = 0;         // camera and bounce rays
    std::uint64_t shadowRays = 0;
    std::uint64_t transparentLimitPaths = 0;
    std::uint32_t samplesPerPixel = 0;     // requested by this call
    std::uint32_t firstSampleIndex = 0;
    std::uint32_t tileCount = 0;
    std::uint32_t tilesCompleted = 0;
    bool cancelled = false;
    bool invalidRequest = false;           // size mismatch or zero samples; nothing rendered
    bool spectral = false;                 // the pass used spectral transport
    LightSetStats lights {};
};

// Whether a render of scene with settings uses spectral transport.
bool usesSpectralTransport (const RenderScene& scene, const IntegratorSettings& settings) noexcept;

// Radiance along one camera ray, linear sRGB. Returns false (and leaves
// radiance unspecified) when the estimate is NaN or infinite.
bool tracePath (const RenderScene& scene, const IntegratorSettings& settings, const Ray& cameraRay,
                const PathSampler& sampler, PathCounters& counters, Vec3& radiance) noexcept;

// Sample dimension of the pixel filter jitter (2D).
constexpr std::uint32_t kPixelFilterDimension = 0;
// Sample dimension of the hero wavelength (1D, independent of the 2D one).
constexpr std::uint32_t kWavelengthDimension = 0;

class PathTracer
{
public:
    explicit PathTracer (unsigned threadCount = 0);

    IntegratorSettings settings {};

    // One progressive pass of samplesPerPass samples per pixel over every
    // tile. A new size or generation clears the accumulator first. Sample
    // indices continue across passes of one generation, so the low-discrepancy
    // sequence keeps its stratification.
    RenderStats renderProgressive (const RenderScene& scene, const PinholeCamera& camera,
                                   std::uint32_t width, std::uint32_t height, std::uint32_t samplesPerPass,
                                   std::uint64_t generation, const CancellationToken& cancel);

    // Clears the accumulator and renders exactly samplesPerPixel samples per
    // pixel (unless cancelled), for export.
    RenderStats renderFixed (const RenderScene& scene, const PinholeCamera& camera,
                             std::uint32_t width, std::uint32_t height, std::uint32_t samplesPerPixel,
                             std::uint64_t generation, const CancellationToken& cancel);

    const Accumulator& accumulator() const noexcept { return accumulator_; }
    unsigned concurrency() const noexcept { return pool_.concurrency(); }

    // Runs fn once per tile of a width x height image on this tracer's pool,
    // for per-pixel work that should share its threads (a depth pass).
    TilePassResult forEachTile (std::uint32_t width, std::uint32_t height, const CancellationToken& cancel,
                                const ThreadPool::TileFunction& fn);

private:
    RenderStats runPass (const RenderScene& scene, const PinholeCamera& camera,
                         std::uint32_t firstSample, std::uint32_t samples, const CancellationToken& cancel);

    ThreadPool pool_;
    Accumulator accumulator_;
    std::uint32_t nextSample_ = 0;
};

} // namespace videohelper::pathtrace
