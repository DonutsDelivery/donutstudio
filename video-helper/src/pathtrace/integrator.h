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
// Light-path limits, as in Cycles. maxDepth caps every scattering event; the
// diffuse, glossy and transmission limits cap the bounces of that kind (a
// delta reflection counts as glossy, any transmission as transmission), and
// maxTransparentHits caps alpha cut-outs, false interfaces and thin sheets
// crossed. A path that reaches a limit ends there and keeps the radiance it
// gathered; RenderStats counts it. No limit ever refuses a frame. The tracer
// has no volume scattering yet (media only absorb), so there is no volume
// bounce limit to reach.
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
// dispersion). One wavelength alone comes out as a nearly pure colour, so any
// other noise in the path would show as colour speckle. A path that meets
// dispersion is therefore traced four times with the same random numbers and
// each of its wavelengths as the hero in turn, and the estimates are averaged.
// Each wavelength follows its own direction, so the colour on screen comes
// from the geometry and not from the random numbers. Paths that never meet
// dispersion are traced once. SpectralMode::Automatic, the default, uses spectral transport
// only when some material is dispersive, so every other scene renders exactly
// as in RGB mode.
//
// Media (media.h). Refracting materials push and pop a medium stack. Real
// interfaces refract with the ratio of the indices on both sides, false ones
// (nested media with priorities) are crossed without scattering, and the
// owning medium absorbs by Beer-Lambert along every segment
// (Material::absorptionDistance). Thin sheets transmit straight, which camera,
// bounce and shadow rays all follow. Shadow rays still stop at real
// refracting interfaces: light seen through glass arrives by BSDF sampling.
//
// Caustics (light tracing). A camera path cannot reach a point or
// directional light through glass, a mirror chain or a grating, and finds
// small lights that way only by chance. When the scene has a refracting
// interface or a bound grating (causticLightPaths, on by default), every
// pixel sample also traces one light path: it leaves a light chosen by power
// (LightSet::sampleEmission, every light type including the environment and
// directional lights through a disk around the scene), follows delta lobes
// only (smooth refraction with its Fresnel split, total internal reflection,
// mirrors, dispersion per wavelength as above, diffraction orders through
// diffraction_order_bsdf), and at the first surface with a non-delta lobe
// after at least one delta vertex it connects to the camera and splats into
// the Accumulator. The camera path leaves out exactly those paths: an emitter
// reached after a first non-delta scatter on an opaque surface followed only
// by delta scatters. Neither method counts a path twice, so the sum is
// unbiased. Caustics seen through glass or a mirror (the camera's first
// scatter is delta) stay with the camera path.
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
#include "devices/device_interface.h"
#include "../../../shared/Visual3DScene.h"

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace diffractionmaterial
{
class AdmittedDiffractionMaterialIR;
}

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

// A per-kind bounce limit that never binds before maxDepth does.
constexpr std::uint32_t kUnlimitedBounces = 0xffffffffu;

struct IntegratorSettings
{
    std::uint32_t maxDepth = 12;            // bounces after the camera hit
    std::uint32_t maxDiffuseBounces = kUnlimitedBounces;
    std::uint32_t maxGlossyBounces = kUnlimitedBounces;
    std::uint32_t maxTransmissionBounces = kUnlimitedBounces;
    std::uint32_t rouletteStartDepth = 3;   // first bounce that may be terminated
    std::uint32_t maxTransparentHits = 16;  // alpha cut-outs crossed per path
    // 0 disables. Otherwise the largest channel of every indirect contribution
    // (two or more bounces) is limited to this value; RenderStats counts them.
    float indirectClamp = 0.0f;
    LightStrategy strategy = LightStrategy::MultipleImportance;
    SpectralMode spectral = SpectralMode::Automatic;
    SamplePattern pattern = SamplePattern::OwenSobol;
    bool environmentVisibleToCamera = true;
    // Light tracing for caustics (see the header comment). It runs only for
    // scenes with a refracting interface or a bound grating.
    bool causticLightPaths = true;
    std::uint32_t tileSize = TileGrid::kDefaultTileSize;
};

// An object tinted by a frame texture. Camera samples that first hit it go to
// frame layer `layer` instead of the accumulator (FrameLayerFilm).
struct ObjectFrameBinding
{
    std::uint32_t objectId = 0;  // SceneObjectId::value
    std::uint32_t layer = 0;
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
    // Materials whose surfaces scatter by diffraction orders. Each binding
    // holds a copy of its admitted material's description.
    std::vector<std::shared_ptr<const diffractionmaterial::AdmittedDiffractionMaterialIR>> gratingMaterials;
    std::vector<device::GratingBinding> gratings;
    std::vector<ObjectFrameBinding> frameBindings;

    // Built by commit().
    Bvh bvh;
    LightSet lights;
    bool anyPartialCoverage = false;
    bool anyThinSheet = false;
    bool anyMediumBoundary = false;
    bool anyDispersive = false;
    // Bounding sphere of the triangles; radius 0 for an empty scene.
    Vec3 boundsCentre {};
    float boundsRadius = 0.0f;

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

// An object whose triangles scatter by the diffraction orders of an admitted
// material (diffraction_order_bsdf.h) instead of a Material.
struct ObjectGratingBinding
{
    std::uint32_t objectId = 0;  // SceneObjectId::value
    std::shared_ptr<const diffractionmaterial::AdmittedDiffractionMaterialIR> material;
    // A spatial foil's corners (GratingBinding::field).
    std::optional<std::array<std::array<float, 4>, 4>> field {};
};

struct PrepareOptions
{
    IngestOptions ingest {};
    // Applied in order; a later entry for the same object wins.
    std::vector<ObjectMaterialOverride> objectMaterials;
    // Applied after objectMaterials; a null material is skipped.
    std::vector<ObjectGratingBinding> objectGratings;
    std::vector<ObjectFrameBinding> objectFrames;
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
    std::uint64_t bounceLimitPaths = 0;       // ended at maxDepth or a per-kind limit
    std::uint64_t transparentLimitPaths = 0;  // ended at maxTransparentHits
    // Caustic light tracing. lightPathRays counts the light paths' closest
    // and camera-visibility rays, which closestRays and shadowRays leave out.
    std::uint64_t lightPaths = 0;
    std::uint64_t lightPathRays = 0;
    std::uint64_t causticSplats = 0;
    std::array<std::uint64_t, 7> causticSplatsByLight {};  // by LightType

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
    // Paths that ended at a light-path limit. They still contribute, so these
    // describe the image; they never mean a pixel or frame failed.
    std::uint64_t bounceLimitPaths = 0;
    std::uint64_t transparentLimitPaths = 0;
    std::uint64_t limitEndedPaths = 0;     // bounceLimitPaths + transparentLimitPaths
    // Caustic light tracing (PathCounters).
    std::uint64_t lightPaths = 0;
    std::uint64_t lightPathRays = 0;
    std::uint64_t causticSplats = 0;
    std::array<std::uint64_t, 7> causticSplatsByLight {};  // by LightType
    std::uint32_t samplesPerPixel = 0;     // requested by this call
    std::uint32_t firstSampleIndex = 0;
    std::uint32_t tileCount = 0;
    std::uint32_t tilesCompleted = 0;
    bool cancelled = false;
    bool invalidRequest = false;           // size mismatch or zero samples; nothing rendered
    bool spectral = false;                 // the pass used spectral transport
    // This pass ran on the CPU because the device cannot draw the scene, and why.
    // Null otherwise. See PathTracer::setDeviceSceneFallback.
    const char* deviceSkipped = nullptr;
    bool deviceFailed = false;             // PathTracer::deviceFailure(); the frame must not show
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
    // threadCount 0 renders on sharedThreadPool(). Any other count gets a private
    // pool of that size.
    explicit PathTracer (unsigned threadCount = 0);

    IntegratorSettings settings {};

    // One progressive pass of samplesPerPass samples per pixel over every
    // tile. A new size or generation clears the accumulator first. Sample
    // indices continue across passes of one generation, so the low-discrepancy
    // sequence keeps its stratification. samplerGeneration seeds the sample
    // streams; zero means generation itself. An accumulation seeded with the
    // export generation draws the same samples as renderFixed, so after N
    // samples it holds the export image at N samples bit for bit.
    RenderStats renderProgressive (const RenderScene& scene, const PinholeCamera& camera,
                                   std::uint32_t width, std::uint32_t height, std::uint32_t samplesPerPass,
                                   std::uint64_t generation, const CancellationToken& cancel,
                                   std::uint64_t samplerGeneration = 0);

    // Clears the accumulator and renders exactly samplesPerPixel samples per
    // pixel (unless cancelled), for export.
    RenderStats renderFixed (const RenderScene& scene, const PinholeCamera& camera,
                             std::uint32_t width, std::uint32_t height, std::uint32_t samplesPerPixel,
                             std::uint64_t generation, const CancellationToken& cancel);

    // One pass on another device. It adds to film and stats, or fills error.
    struct DevicePass
    {
        const RenderScene& scene;
        const PinholeCamera& camera;
        const IntegratorSettings& settings;
        std::uint32_t firstSample;
        std::uint32_t samples;
        std::uint64_t generation;
        const CancellationToken& cancel;
        Accumulator& film;
        RenderStats& stats;
        PathCounters& counters;
        std::string& error;
    };
    using DevicePassFunction = std::function<bool (DevicePass&)>;
    // Set, every pass runs there; a failure sets RenderStats::deviceFailed and
    // the CPU traces nothing. Not set, the CPU device renders.
    void setDevicePass (DevicePassFunction pass) { devicePass_ = std::move (pass); }
    const std::string& deviceFailure() const noexcept { return deviceFailure_; }
    // A scene the device cannot draw goes to the CPU for that scene's passes, and
    // the device stays set for the next scene. Off, such a scene fails instead.
    void setDeviceSceneFallback (bool allowed) noexcept { deviceSceneFallback_ = allowed; }

    const Accumulator& accumulator() const noexcept { return accumulator_; }
    // One film per frame layer of the scene last rendered. Divide by the
    // accumulator's sample count, which counts these samples too.
    const std::vector<FrameLayerFilm>& frameLayers() const noexcept { return frameLayers_; }
    unsigned concurrency() const noexcept { return pool_.concurrency(); }

    // Runs fn once per tile of a width x height image on this tracer's pool,
    // for per-pixel work that should share its threads (a depth pass).
    TilePassResult forEachTile (std::uint32_t width, std::uint32_t height, const CancellationToken& cancel,
                                const ThreadPool::TileFunction& fn);

private:
    RenderStats runPass (const RenderScene& scene, const PinholeCamera& camera,
                         std::uint32_t firstSample, std::uint32_t samples, const CancellationToken& cancel);

    std::unique_ptr<ThreadPool> ownedPool_;
    ThreadPool& pool_;
    DevicePassFunction devicePass_;
    bool deviceSceneFallback_ = false;
    std::string deviceFailure_;
    Accumulator accumulator_;
    std::vector<FrameLayerFilm> frameLayers_;
    std::uint32_t nextSample_ = 0;
    // Seeds the sample streams of the live accumulation (see renderProgressive).
    std::uint64_t samplerGeneration_ = 0;
};

} // namespace videohelper::pathtrace
