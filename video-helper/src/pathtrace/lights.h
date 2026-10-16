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
// light/background.h, light/point.h, light/spot.h and light/sun.h.

#include "bsdf.h"
#include "environment_map.h"
#include "scene_ingest.h"
#include "vec_math.h"
#include "../../../shared/Visual3DScene.h"

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
    Triangle = 4
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
    Vec3 normal {};               // Triangle: unit front-face normal
    float area = 0.0f;            // Triangle
};

// Radiance arriving from infinitely far away, by direction.
struct EnvironmentField
{
    Vec3 constant {};
    bool analytic = false;
    Vec3 zenith {};
    Vec3 horizon {};
    Vec3 ground {};
    Vec3 sunRadiance {};          // peak radiance at the cap centre
    Vec3 sunDirection { 0.0f, 1.0f, 0.0f };
    float sunCosRadius = 1.0f;
    float sunSampleProbability = 0.0f;  // share of samples drawn inside the sun cap
    // When set, the image replaces the analytic gradient and sun, and
    // constant still adds to it.
    std::shared_ptr<const EnvironmentMap> image;
    float imageSampleProbability = 0.0f;  // share of samples drawn from the image

    bool active() const noexcept;
    Vec3 radiance (Vec3 direction) const noexcept;
    // Solid-angle density of sample() for a unit direction.
    float pdf (Vec3 direction) const noexcept;
    Vec3 sample (Vec2 u, float& pdfOut) const noexcept;
};

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

struct LightSetStats
{
    std::uint32_t directional = 0;
    std::uint32_t point = 0;
    std::uint32_t spot = 0;
    std::uint32_t environment = 0;            // constant and analytic records merged into the field
    std::uint32_t environmentImages = 0;      // image records lighting the scene (at most 1)
    std::uint32_t supersededEnvironments = 0; // constant and analytic records an image replaced
    std::uint32_t emissiveTriangles = 0;
    // Image records without matching valid pixels, with an invalid binding, a
    // black image, or after the first usable image. They light nothing.
    std::uint32_t ignoredEnvironmentImages = 0;
    std::uint32_t invalidLights = 0;          // non-finite records or emitters, skipped
    std::uint32_t zeroPowerLights = 0;        // valid but black, skipped
};

class LightSet
{
public:
    // records may be null when count is 0. materials is indexed by
    // TriangleAttributes::materialIndex; triangles outside it use fallback.
    // environmentImage holds the decoded pixels for the image record whose
    // asset identity it names; null or empty means no image is available.
    void build (const HarmonicMIDI::grid::SceneLightRecord* records, std::size_t count, Vec3 ambient,
                const IngestedScene& geometry, const std::vector<Material>& materials,
                const Material& fallback, const EnvironmentPixels* environmentImage = nullptr);
    void clear() noexcept;

    std::size_t size() const noexcept { return lights_.size(); }
    const std::vector<Light>& lights() const noexcept { return lights_; }
    const LightSetStats& stats() const noexcept { return stats_; }
    const EnvironmentField& environment() const noexcept { return environment_; }
    bool hasEnvironment() const noexcept { return environment_.active(); }
    float selectionProbability (std::size_t light) const noexcept;

    // One light sample toward shading point p.
    bool sample (Vec3 p, float uSelect, Vec2 u, LightSample& out) const noexcept;

    // Density sample() gives to the direction of a ray from a shading point
    // that hit emissive triangle `triangle` at distance `distance` with
    // |cos| = cosAtLight at the light. 0 when the triangle is not a light.
    float trianglePdf (std::uint32_t triangle, float distance, float cosAtLight) const noexcept;
    // Density sample() gives to an escaping unit direction.
    float environmentPdf (Vec3 direction) const noexcept;
    Vec3 environmentRadiance (Vec3 direction) const noexcept { return environment_.radiance (direction); }

private:
    std::vector<Light> lights_;
    std::vector<float> cdf_;                 // inclusive, normalized, back() == 1
    std::vector<float> probability_;
    std::vector<std::int32_t> triangleLight_;
    EnvironmentField environment_ {};
    std::int32_t environmentLight_ = -1;
    LightSetStats stats_ {};
};

} // namespace videohelper::pathtrace
