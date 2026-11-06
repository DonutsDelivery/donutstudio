#pragma once

// The CPU device: the first device the integrator body compiles for, and the
// reference the GPU devices are checked against. PathTracer renders through it.

#include "scene_view.h"

namespace videohelper::pathtrace::device
{

struct CpuDevice
{
    using Accel = const Bvh*;
    using Film = Accumulator;
    using Sampler = PathSampler;

    static Sampler makeSampler (std::uint32_t pixel, std::uint32_t sampleIndex, std::uint64_t generation,
                                SamplePattern pattern) noexcept
    {
        return Sampler (pixel, sampleIndex, generation, pattern);
    }
    static bool intersect (const Accel& accel, const Ray& ray, Hit& hit) noexcept { return accel->intersect (ray, hit); }
    static bool occluded (const Accel& accel, const Ray& ray) noexcept { return accel->occluded (ray); }
    // Tiles own disjoint pixels, so no two threads add to one pixel.
    static void filmAdd (Film& film, std::uint32_t x, std::uint32_t y, Vec3 rgb) noexcept
    {
        film.addSample (x, y, { rgb.x, rgb.y, rgb.z });
    }
    // Fixed-point atomics: the sum does not depend on thread order.
    static void filmSplat (Film& film, std::uint32_t x, std::uint32_t y, Vec3 rgb) noexcept
    {
        film.addSplat (x, y, { rgb.x, rgb.y, rgb.z });
    }
    static void filmCountLightPath (Film& film) noexcept { film.addLightPath(); }
    // Tiles own disjoint pixels, as for filmAdd.
    static void filmGuide (Film& film, std::uint32_t x, std::uint32_t y, Vec3 albedo, Vec3 normal) noexcept
    {
        film.addGuide (x, y, { albedo.x, albedo.y, albedo.z }, { normal.x, normal.y, normal.z });
    }
};

namespace cpu
{

using SceneView = device::SceneView<CpuDevice::Accel>;

// Points a view at scene's buffers. The view borrows scene and gratings;
// empty gratings uses the scene's own (RenderScene::gratings).
SceneView makeSceneView (const RenderScene& scene, BufferView<GratingBinding> gratings = {}) noexcept;

// Whether a render of view uses spectral transport. Bound gratings count as
// dispersive under SpectralMode::Automatic.
bool usesSpectralTransport (const SceneView& view, const IntegratorSettings& settings) noexcept;

// Whether renderSample traces a light path next to each camera path
// (IntegratorSettings::causticLightPaths).
bool causticLightTracingActive (const SceneView& view, const IntegratorSettings& settings) noexcept;

// Radiance along one camera ray, linear sRGB; false when the estimate is not
// finite (see tracePath in integrator.h). excludeCausticPaths leaves out the
// paths the light tracer carries (renderSample sets it when it is active). camera,
// the one the ray left at its time, lets a light sample share those paths with the
// light tracer (gratingShare in integrator_body.inl); without it they stay the
// light tracer's.
bool tracePath (const SceneView& view, const IntegratorSettings& settings, const Ray& cameraRay,
                const PathSampler& sampler, PathCounters& counters, Vec3& radiance,
                bool excludeCausticPaths = false, const PinholeCamera* camera = nullptr) noexcept;

// renderSample up to the camera path's film write: returns the jittered camera
// ray and its radiance, and leaves adding that radiance to the caller. Light paths
// still splat to film.
SampleOutcome traceCameraSample (const SceneView& view, const IntegratorSettings& settings, const PinholeCamera& camera,
                                 std::uint32_t x, std::uint32_t y, std::uint32_t width, std::uint32_t sampleIndex,
                                 std::uint64_t generation, Accumulator& film, PathCounters& counters,
                                 Ray& ray, Vec3& radiance) noexcept;

// The first-hit albedo and normal guides of a camera ray (Plan 3, Denoising).
void firstHitGuides (const SceneView& view, const Ray& cameraRay, Vec3& albedo, Vec3& normal) noexcept;
// The unflipped world shading normal at the first hit, for the Normal output.
Vec3 firstHitWorldNormal (const SceneView& view, const Ray& cameraRay) noexcept;

// One sample of pixel (x, y) of a width-wide image: draws the sampler, jitters
// the camera ray, traces it and adds the result to film. When caustic light
// tracing is active it also traces one light path and splats it. An added sample
// also adds its first-hit guides.
SampleOutcome renderSample (const SceneView& view, const IntegratorSettings& settings, const PinholeCamera& camera,
                            std::uint32_t x, std::uint32_t y, std::uint32_t width, std::uint32_t sampleIndex,
                            std::uint64_t generation, Accumulator& film, PathCounters& counters) noexcept;

} // namespace cpu
} // namespace videohelper::pathtrace::device
