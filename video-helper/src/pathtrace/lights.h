#pragma once

// Light sources of the CPU path tracer and their sampling.
//
// Units follow the raster backends (backend_opengl.cpp, fixture shader light
// loop), so a scene lit by the tracer and by the raster agrees in scale:
// - Directional: color * intensity is the irradiance on a surface facing the
//   light. The light travels along the record's rotated local -Z.
// - Point and Spot: color * intensity / max(d^2, 1) is the irradiance at
//   distance d, times (1 - d / range)^2 clamped to [0, 1] when range > 0. A
//   Spot also multiplies smoothstep(cos(outer), cos(inner), cos(angle)) with
//   the axis along rotated local -Z.
// - Environment (constant): color * intensity is the radiance from every
//   direction. The scene's ambientColor adds to it when requested, as the
//   raster adds it to its ambient term. Version-1 analytic skies use the
//   raster's zenith, horizon and ground gradient and its smooth sun cap.
//   An environment image (version 1) with matching decoded pixels replaces
//   the constant and analytic fields of the records (environment_map.h has
//   its lookup convention); the scene ambient still adds to it. An image
//   record without usable pixels is counted in
//   LightSetStats::ignoredEnvironmentImages and lights nothing.
// - Emissive triangles: Material::emission times coverage is the radiance
//   leaving both faces, as the raster adds emissive to the shaded colour.
//
// Sizes. SceneLightRecord::radius and angularRadius size a record's light; a
// caller may also pass one LightShape per record, which wins where it is set
// (all 0 by default, the delta lights above):
// - Point and Spot with radius r > 0 become uniformly emitting spheres. Their
//   radiance is chosen so that a surface at distance d receives the same
//   irradiance as from the point light, falloff, range window and spot cone
//   included (all evaluated from the sphere centre): L = color * intensity *
//   d^2 / max(d^2, 1) / (pi r^2). Inside the sphere the point light is used.
// - Directional with angular radius a > 0 becomes a uniform sun disk of
//   radiance E / (pi sin^2 a), which gives the same irradiance E at normal
//   incidence.
// AreaLightRecord adds rect, disk and ellipse lights. Their radiance is
// color * intensity, along rotated local -Z (one-sided by default, as the spot
// axis). A scene record of kind Area becomes one of them with radiance
// color * intensity / area, so intensity is the radiant intensity along the
// axis, as a point light's is. Spread below pi narrows the emission as a
// softbox grid does: radiance times max(0, tan(s/2) - tan(theta)) * N, with N
// chosen so the emitted power stays pi L A (N tends to 3 / (s/2)^3 for a
// narrow spread). Sampling still covers the whole light; the attenuation is in
// the radiance, so sample() and pdf stay consistent.
//
// Ray visibility (SceneLightRecord::visibility) is copied to Light::visibility.
// The integrator reads it: Camera lets camera rays see a sized light (suns
// never: an infinitely far disk stays off the backdrop, as in Cycles), Shadow
// off lets the light through every blocker, and Diffuse, Glossy and
// Transmission mask what a scatter of that kind receives from the light.
//
// Sized lights have no geometry: they never block rays. Bounce rays reach them through analyticHit and combine with light
// sampling by MIS. Spheres and suns are sampled uniformly in the cone they
// subtend, rects by solid angle (Urena, Fajardo and King, "An Area-Preserving
// Parametrization for Spherical Rectangles", EGSR 2013) with an area-sampling
// fallback for tiny solid angles, and disks uniformly by area.
//
// One light is chosen per shading point with probability proportional to an
// estimate of the power it delivers to the scene. Directional, Point and Spot
// lights are delta lights. Emissive triangles are sampled uniformly by area
// and converted to a solid-angle pdf; the environment is sampled uniformly over
// the sphere, mixed with uniform cone sampling of an analytic sun cap, or with
// the image's power distribution when an image is present. An image's
// selection power is its integrated radiance.
//
// Behavioural reference (studied, not copied): Blender Cycles
// intern/cycles/kernel/light/distribution.h, light/triangle.h,
// light/background.h, light/point.h, light/spot.h, light/sun.h and
// light/area.h.

#include "bsdf.h"
#include "environment_map.h"
#include "scene_ingest.h"
#include "vec_math.h"
#if ! defined(PATHTRACE_DEVICE_METAL)
#include "../../../shared/DiffractionLightingPlan.h"
#endif
#include "devices/device_interface.h"
#if ! defined(PATHTRACE_DEVICE_METAL)
#include "../../../shared/Visual3DScene.h"
#endif

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace videohelper::pathtrace
{

enum class LightType : std::uint8_t
{
    Directional = 0,
    Point = 1,
    Spot = 2,
    Environment = 3,
    Triangle = 4,
    Rect = 5,
    Disk = 6
};

// Size of the emitter behind one scene light record. 0 keeps the delta light.
struct LightShape
{
    // Point and Spot: sphere radius in scene units.
    float radius = 0.0f;
    // Directional: angular radius of the sun disk in radians, at most 1.5.
    float angularRadius = 0.0f;
    // Directional: one authored value per production wavelength. The record's
    // color * intensity is then their mean, and each band scales it.
    bool spectralBands = false;
    std::array<float, diffractionmaterial::kMaximumSpectralSamples> bands {};
};

// A grating scatters into delta orders, and a delta order never meets a delta
// light, so an unsized Directional Light gives a grating nothing. RenderScene
// gives each one in a grating scene this angular radius. Its radiance is
// E / (pi sin^2 a), so the irradiance E stays what was authored. At 1400 nm,
// first order disperses about 0.7 mrad per nm, so 3 degrees lights a band about
// 150 nm wide: narrow enough to stay coloured.
inline constexpr float kGratingSourceAngularRadius = 0.05235988f;

// Host only: the Metal kernel (device_metal.metal) leaves this out.
#if ! defined(PATHTRACE_DEVICE_METAL)
// The Direct path of an authored spectral lighting plan (version 3) as a
// Directional record and its shape. Each band is a beam integral in W / (m^2 nm),
// the unit a Directional Light's irradiance already has. False for other plans
// and for a source with no energy.
PT_SHARED_ENTRY bool authoredSpectralSourceLight (const diffractionmaterial::LightingDescription& lighting,
                                                  HarmonicMIDI::grid::SceneLightRecord& record, LightShape& shape) noexcept;
#endif

enum class AreaLightShape : std::uint8_t
{
    Rect = 0,
    Disk = 1,
    // Disk and Ellipse both read sizeX and sizeY as diameters and sample the same
    // way; the scene's Disk shape passes equal sizes (sceneAreaLightExtent).
    Ellipse = 2
};

// Spread attenuation of a planar light, see above (N is
// HarmonicMIDI::grid::sceneAreaSpreadNormalization). 1 when N is 0.
PT_SHARED_ENTRY float areaSpreadAttenuation (float tanHalfSpread, float normalization, float cosAtLight) noexcept;

// A planar light that has no scene record kind.
struct AreaLightRecord
{
    AreaLightShape shape = AreaLightShape::Rect;
    Vec3 position {};  // centre
    // Local X and Y span the light. It emits along rotated local -Z.
    Quat rotation {};
    // Rect: full width along X and height along Y. Disk and Ellipse: diameters.
    float sizeX = 1.0f;
    float sizeY = 1.0f;
    Vec3 color { 1.0f, 1.0f, 1.0f };
    float intensity = 1.0f;  // color * intensity is the emitted radiance
    bool twoSided = false;
    // Full spread angle in radians; pi emits as a Lambertian surface.
    float spread = 3.14159265f;
    // HarmonicMIDI::grid::sceneRayVisibility bits. LightSet::build clears the
    // camera bit of records passed in its areaLights list.
    std::uint8_t visibility = 0x3fu;
};

struct Light
{
    LightType type = LightType::Point;
    // Directional: irradiance. Point and Spot: color * intensity. Triangle:
    // emitted radiance. Environment: unused (see EnvironmentField).
    Vec3 strength {};
    Vec3 position {};             // Point, Spot
    Vec3 direction {};            // Directional: toward the light. Spot: propagation axis.
    float range = 0.0f;           // Point, Spot; 0 is unlimited
    float cosInner = 1.0f;        // Spot
    float cosOuter = 0.0f;        // Spot
    std::uint32_t triangle = 0;   // Triangle: index into the ingested triangle list
    Vec3 vertex0 {};              // Triangle corners: vertex0, vertex0 + edge1, vertex0 + edge2
    Vec3 edge1 {};
    Vec3 edge2 {};
    Vec3 normal {};               // Triangle: unit front-face normal. Rect, Disk: emission side.
    float area = 0.0f;            // Triangle, Rect, Disk
    // Point and Spot: sphere radius, 0 for a point.
    float radius = 0.0f;
    // Directional with a disk: cos and 1 - cos of the angular radius, and
    // radiance over irradiance, 1 / (pi sin^2). coneOneMinusCos 0 is delta.
    float coneCos = 1.0f;
    float coneOneMinusCos = 0.0f;
    float coneRadianceScale = 0.0f;
    // Directional from an authored spectrum: each band over strength.
    bool spectralBands = false;
    std::array<float, diffractionmaterial::kMaximumSpectralSamples> bandScale {};
    // Rect: vertex0 is a corner, edge1 and edge2 the full sides. Disk:
    // position is the centre, edge1 and edge2 the half axes.
    bool twoSided = false;
    // Rect and Disk: tan of half the spread and its normalization N; N = 0
    // leaves the emission Lambertian.
    float tanHalfSpread = 0.0f;
    float spreadNormalization = 0.0f;
    // HarmonicMIDI::grid::sceneRayVisibility bits.
    std::uint8_t visibility = 0x3fu;
};

// Radiance arriving from infinitely far away, by direction.
struct EnvironmentField
{
    Vec3 uniformRadiance {};
    bool analytic = false;
    Vec3 zenith {};
    Vec3 horizon {};
    Vec3 ground {};
    Vec3 sunRadiance {};          // peak radiance at the cap centre
    Vec3 sunDirection { 0.0f, 1.0f, 0.0f };
    float sunCosRadius = 1.0f;
    float sunSampleProbability = 0.0f;  // share of samples drawn inside the sun cap
    // The analytic cloud layer, as the raster's analyticSkyCloudOpacity reads it: coverage in
    // 1023rds, type in 63rds and heading in 256ths of a turn. Coverage 0 means no clouds.
    float cloudCoverage = 0.0f;
    float cloudKind = 0.0f;
    float cloudHeadingRadians = 0.0f;
    float cloudDrift = 0.0f;
    // The sun halo share and the volumetric cloud thickness, in 63rds as the raster
    // reads them. Zero draws no halo and keeps the flat cloud layer.
    float sunHalo = 0.0f;
    float cloudThickness = 0.0f;
    // When set, the image replaces the analytic gradient and sun, and
    // constant still adds to it.
    // Owned by the LightSet that built this field.
    const PT_GLOBAL EnvironmentMapView* image = nullptr;
    float imageSampleProbability = 0.0f;  // share of samples drawn from the image

    PT_SHARED_ENTRY bool active() const noexcept;
    PT_SHARED_ENTRY Vec3 radiance (Vec3 direction) const noexcept;
    // Solid-angle density of sample() for a unit direction.
    PT_SHARED_ENTRY float pdf (Vec3 direction) const noexcept;
    PT_SHARED_ENTRY Vec3 sample (Vec2 u, float& pdfOut) const noexcept;
};

// The map an EnvironmentField points at. Metal calls its member functions on a
// copy in the kernel's own memory; everywhere else this is the map itself.
#if defined(PATHTRACE_DEVICE_METAL)
PT_SHARED_FN EnvironmentMapView environmentMapAt (const PT_GLOBAL EnvironmentMapView* image) noexcept { return *image; }
#else
PT_SHARED_FN const EnvironmentMapView& environmentMapAt (const EnvironmentMapView* image) noexcept { return *image; }
#endif

struct LightSample
{
    Vec3 wi {};               // unit, toward the light
    Vec3 radiance {};         // area and environment: incident radiance; delta lights: irradiance
    Vec3 point {};            // Point, Spot, Triangle: the light position
    Vec3 lightNormal {};      // Triangle: front normal
    float distance = std::numeric_limits<float>::infinity();
    // Non-delta: solid-angle density including the selection probability.
    // Delta: the selection probability.
    float pdf = 0.0f;
    bool delta = false;
    bool infinite = false;    // Directional and Environment
    bool area = false;        // Triangle
    std::uint32_t light = 0;
};

// One particle leaving a light, for light tracing (integrator.h, caustic
// light paths). weight is the flux it carries: emitted radiance times the
// area and solid angle it stands for, over their densities and the selection
// probability, so the expected weight is the light set's emitted power in the
// units above. Lights at infinity (Directional, Environment) emit through a
// disk of the scene bounding sphere's radius, facing the particle, so every
// surface inside the sphere is lit as from infinitely far.
// Point and Spot lights, sized or not, follow the raster's receiver-dependent
// falloff: multiply the weight by emissionFalloff() at the first surface the
// particle reaches.
struct EmissionSample
{
    Vec3 origin {};
    Vec3 direction {};          // unit, the way the light travels
    Vec3 weight {};
    std::uint32_t light = 0;    // index into lights()
    LightType type = LightType::Point;
    // Emissive triangle the particle leaves, which its first ray must skip;
    // 0xffffffff for other lights.
    std::uint32_t ignoreTriangle = 0xffffffffu;
    bool receiverFalloff = false;
};

struct LightSetStats
{
    std::uint32_t directional = 0;
    std::uint32_t point = 0;
    std::uint32_t spot = 0;
    std::uint32_t environment = 0;            // constant and analytic records merged into the field
    std::uint32_t environmentImages = 0;      // image records lighting the scene (at most 1)
    std::uint32_t supersededEnvironments = 0; // constant and analytic records an image replaced
    std::uint32_t emissiveTriangles = 0;
    std::uint32_t sizedLights = 0;            // spheres and sun disks
    std::uint32_t areaLights = 0;             // rects, disks and ellipses
    // Image records without matching valid pixels, with an invalid binding, a
    // black image, or after the first usable image. They light nothing.
    std::uint32_t ignoredEnvironmentImages = 0;
    std::uint32_t invalidLights = 0;          // non-finite records or emitters, skipped
    std::uint32_t zeroPowerLights = 0;        // valid but black, skipped
};

// The read side of a built light set: plain data over borrowed buffers, so the
// same sampling code runs on a device once the buffers are uploaded. A LightSet
// owns the buffers its view reads.
class LightSetView
{
public:
    PT_SHARED_FN std::size_t size() const noexcept { return lights_.size(); }
    PT_SHARED_FN const EnvironmentField& environment() const noexcept { return environment_; }
    PT_SHARED_FN bool hasEnvironment() const noexcept { return environment_.active(); }
    PT_SHARED_ENTRY float selectionProbability (std::size_t light) const noexcept;

    // One light sample toward shading point p.
    PT_SHARED_ENTRY bool sample (Vec3 p, float uSelect, Vec2 u, LightSample& out) const noexcept;

    // Density sample() gives to the direction of a ray from a shading point
    // that hit emissive triangle `triangle` at distance `distance` with
    // |cos| = cosAtLight at the light. 0 when the triangle is not a light.
    PT_SHARED_ENTRY float trianglePdf (std::uint32_t triangle, float distance, float cosAtLight) const noexcept;
    // Density sample() gives to an escaping unit direction.
    PT_SHARED_ENTRY float environmentPdf (Vec3 direction) const noexcept;

    // One emitted particle: a light by selection probability, then a point
    // and direction on it. sceneCentre and sceneRadius bound the geometry;
    // lights at infinity emit nothing without them (sceneRadius 0).
    PT_SHARED_ENTRY bool sampleEmission (float uSelect, Vec2 uPosition, Vec2 uDirection, Vec3 sceneCentre, float sceneRadius,
                         EmissionSample& out) const noexcept;
    // Point and Spot: d^2 times the falloff, range window and spot cone at
    // squared distance d^2 from the light centre, which turns the particle's
    // weight into the irradiance the raster convention gives at receiver. 1
    // for every other particle.
    PT_SHARED_ENTRY float emissionFalloff (const EmissionSample& sample, Vec3 receiver) const noexcept;
    PT_SHARED_FN Vec3 environmentRadiance (Vec3 direction) const noexcept { return environment_.radiance (direction); }

    // Sized lights without geometry (spheres, sun disks, rects, disks).
    PT_SHARED_FN std::size_t analyticCount() const noexcept { return analytic_.size(); }
    // The index into lights_ of analytic light `index`.
    PT_SHARED_FN std::uint32_t analyticLight (std::size_t index) const noexcept { return analytic_[index]; }
    // The authored spectrum of a light at a wavelength, relative to its
    // strength: 1 for a light with no authored spectrum.
    PT_SHARED_ENTRY float spectralBandScale (std::uint32_t light, float wavelengthNm) const noexcept;
    // Whether the ray origin + t direction, 0 < t < tMax (direction unit),
    // reaches analytic light `index`. On a hit, radiance is what the ray
    // receives from it, pdf the density sample() gives that direction from
    // origin, selection included, and distance the t of the hit (infinite
    // for a sun).
    PT_SHARED_ENTRY bool analyticHit (std::size_t index, Vec3 origin, Vec3 direction, float tMax, Vec3& radiance,
                      float& pdf, float& distance) const noexcept;
    // The ray visibility bits of light `light` (an index into lights()).
    PT_SHARED_FN std::uint8_t visibility (std::uint32_t light) const noexcept
    {
        return light < lights_.size() ? lights_[light].visibility : std::uint8_t { 0x3fu };
    }
    // Whether light `light` lies at infinity with a size: the environment, or a sun
    // with a disk. A light sample reaches it by density, not as a delta.
    PT_SHARED_FN bool distantSized (std::uint32_t light) const noexcept
    {
        if (light >= lights_.size())
            return false;
        const PT_GLOBAL Light& record = lights_[light];
        return record.type == LightType::Environment || (record.type == LightType::Directional && record.coneOneMinusCos > 0.0f);
    }
    PT_SHARED_FN bool isEnvironment (std::uint32_t light) const noexcept
    {
        return light < lights_.size() && lights_[light].type == LightType::Environment;
    }
    // Whether a ray of kind rayKind (one sceneRayVisibility bit) sees analytic
    // light `index`. A sun is never in the camera's backdrop.
    PT_SHARED_FN bool analyticSeenBy (std::size_t index, std::uint8_t rayKind) const noexcept
    {
        const PT_GLOBAL Light& light = lights_[analytic_[index]];
        return (light.visibility & rayKind) != 0 && ! (rayKind == 1u && light.type == LightType::Directional);
    }

#if ! defined(__CUDACC__)
    // This view with each buffer replaced by what upload returns for it.
    template <typename Upload>
    LightSetView deviceCopy (Upload&& upload) const
    {
        LightSetView copy = *this;
        copy.lights_ = upload (lights_);
        copy.cdf_ = upload (cdf_);
        copy.probability_ = upload (probability_);
        copy.triangleLight_ = upload (triangleLight_);
        copy.analytic_ = upload (analytic_);
        if (environment_.image != nullptr)
        {
            const EnvironmentMapView image = environment_.image->deviceCopy (upload);
            copy.environment_.image = upload (device::BufferView<EnvironmentMapView> { &image, 1 }).data;
        }
        return copy;
    }
#endif

protected:
    device::BufferView<Light> lights_;
    device::BufferView<float> cdf_;                 // inclusive, normalized, last entry 1
    device::BufferView<float> probability_;
    device::BufferView<std::int32_t> triangleLight_;
    device::BufferView<std::uint32_t> analytic_;    // indices into lights_
    EnvironmentField environment_ {};
    std::int32_t environmentLight_ = -1;
};

// Host only: the Metal kernel (device_metal.metal) leaves this out.
#if ! defined(PATHTRACE_DEVICE_METAL)
class LightSet : public LightSetView
{
public:
    LightSet() = default;
    LightSet (const LightSet& other);
    LightSet (LightSet&& other) noexcept;
    LightSet& operator= (const LightSet& other);
    LightSet& operator= (LightSet&& other) noexcept;

    // records may be null when count is 0. materials is indexed by
    // TriangleAttributes::materialIndex; triangles outside it use fallback.
    // environmentImage holds the decoded pixels for the image record whose
    // asset identity it names; null or empty means no image is available.
    // shapes, when given, holds the size of records[i] at index i; missing
    // entries are delta lights. areaLights adds rects and disks.
    void build (const HarmonicMIDI::grid::SceneLightRecord* records, std::size_t count, Vec3 ambient,
                const IngestedScene& geometry, const std::vector<Material>& materials,
                const Material& fallback, const EnvironmentPixels* environmentImage = nullptr,
                const std::vector<LightShape>* shapes = nullptr,
                const std::vector<AreaLightRecord>* areaLights = nullptr);
    void clear() noexcept;

    const std::vector<Light>& lights() const noexcept { return ownedLights_; }
    const LightSetStats& stats() const noexcept { return stats_; }

private:
    // Points the view at the buffers below. Runs after every change to them.
    void refreshView() noexcept;

    std::vector<Light> ownedLights_;
    std::vector<float> ownedCdf_;
    std::vector<float> ownedProbability_;
    std::vector<std::int32_t> ownedTriangleLight_;
    std::vector<std::uint32_t> ownedAnalytic_;
    std::shared_ptr<const EnvironmentMap> ownedImage_;
    LightSetStats stats_ {};
};
#endif

} // namespace videohelper::pathtrace
