// OptiX device driver TU: compiles the single-source integrator body for
// OptiX (Plan 3, Tier V, OptiX first). Built only with -DARBIT_OPTIX_KERNEL=ON
// on a machine where configure finds the SDK and a CUDA compiler.
//
// The launch that runs it, with the film and counters it fills, is in
// device_optix_host.cpp. The build embeds the PTX (cmake/OptiXEmbedKernel.cmake).

#define PATHTRACE_DEVICE_OPTIX 1

#if ! defined(__CUDACC_RELAXED_CONSTEXPR__)
#error "Compile this kernel with --expt-relaxed-constexpr"
#endif

#include <optix.h>

#include "device_interface.h"
#include "device_optix_params.h"
#include "scene_view.h"
#include "../bvh.h"
#include "../sampler.h"

namespace videohelper::pathtrace::device
{

using OptixFilm = optix::OptixFilm;

struct OptixDevice
{
    using Accel = OptixTraversableHandle;
    using Film = OptixFilm;
    using Sampler = PathSampler;

    static constexpr unsigned kRayTypeClosest = 0;
    static constexpr unsigned kRayTypeShadow = 1;
    static constexpr unsigned kRayTypeCount = 2;

    PT_DEVICE_FN static Sampler makeSampler (std::uint32_t pixel, std::uint32_t sampleIndex, std::uint64_t generation,
                                             SamplePattern pattern) noexcept
    {
        return Sampler (pixel, sampleIndex, generation, pattern);
    }

    // Payload: t, u, v, triangle, ignored triangle. The closest-hit program
    // writes the first four; the miss program leaves triangle at kNoTriangle.
    PT_DEVICE_FN static bool intersect (const Accel& accel, const Ray& ray, Hit& hit) noexcept
    {
        unsigned p0 = __float_as_uint (ray.tMax), p1 = 0, p2 = 0, p3 = kNoTriangle, p4 = ray.ignoreTriangle;
        optixTrace (accel, make_float3 (ray.origin.x, ray.origin.y, ray.origin.z),
                    make_float3 (ray.direction.x, ray.direction.y, ray.direction.z), ray.tMin, ray.tMax, 0.0f,
                    OptixVisibilityMask (255), OPTIX_RAY_FLAG_NONE, kRayTypeClosest, kRayTypeCount, kRayTypeClosest,
                    p0, p1, p2, p3, p4);
        if (p3 == kNoTriangle)
            return false;
        hit.t = __uint_as_float (p0);
        hit.u = __uint_as_float (p1);
        hit.v = __uint_as_float (p2);
        hit.triangle = p3;
        return true;
    }

    PT_DEVICE_FN static bool occluded (const Accel& accel, const Ray& ray) noexcept
    {
        unsigned blocked = 1;  // the shadow miss program clears it
        unsigned ignored = ray.ignoreTriangle;
        optixTrace (accel, make_float3 (ray.origin.x, ray.origin.y, ray.origin.z),
                    make_float3 (ray.direction.x, ray.direction.y, ray.direction.z), ray.tMin, ray.tMax, 0.0f,
                    OptixVisibilityMask (255),
                    OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, kRayTypeShadow,
                    kRayTypeCount, kRayTypeShadow, blocked, ignored);
        return blocked != 0;
    }

    PT_DEVICE_FN static void filmAdd (Film& film, std::uint32_t x, std::uint32_t y, Vec3 rgb) noexcept
    {
        optix::FilmPixel& pixel = film.pixels[y * film.width + x];
        // A non-finite channel counts as 0, as Accumulator::addSample has it.
        atomicAdd (&pixel.x, std::isfinite (rgb.x) ? rgb.x : 0.0f);
        atomicAdd (&pixel.y, std::isfinite (rgb.y) ? rgb.y : 0.0f);
        atomicAdd (&pixel.z, std::isfinite (rgb.z) ? rgb.z : 0.0f);
        atomicAdd (&pixel.w, 1.0f);
    }

    PT_DEVICE_FN static void filmSplat (Film& film, std::uint32_t x, std::uint32_t y, Vec3 rgb) noexcept
    {
        optix::FilmPixel& splat = film.splats[y * film.width + x];
        atomicAdd (&splat.x, std::isfinite (rgb.x) ? rgb.x : 0.0f);
        atomicAdd (&splat.y, std::isfinite (rgb.y) ? rgb.y : 0.0f);
        atomicAdd (&splat.z, std::isfinite (rgb.z) ? rgb.z : 0.0f);
    }

    PT_DEVICE_FN static void filmCountLightPath (Film& film) noexcept { atomicAdd (film.lightPaths, 1ull); }
};

} // namespace videohelper::pathtrace::device

extern "C" __constant__ ::videohelper::pathtrace::device::optix::LaunchParams params;

namespace videohelper::pathtrace::device
{
PT_SHARED_ENTRY const DeviceLookupTables& deviceLookupTables() noexcept
{
    return params.tables;
}
} // namespace videohelper::pathtrace::device

// OptiX compiles one TU, so the out-of-line shared code comes in as source. The
// parts of these files only the host uses are dropped with the host pass.
#include "../spectrum.cpp"
#include "../bsdf.cpp"
#include "../environment_map.cpp"
#include "../lights.cpp"

#define PT_DEVICE ::videohelper::pathtrace::device::OptixDevice
#define PT_DEVICE_NAMESPACE optix
#include "integrator_body.inl"

// One thread per pixel, the same per-sample kernel as the CPU device.
extern "C" __global__ void __raygen__pathtrace()
{
    namespace optix = ::videohelper::pathtrace::device::optix;
    const uint3 index = optixGetLaunchIndex();
    ::videohelper::pathtrace::device::OptixFilm film = params.film;
    ::videohelper::pathtrace::PathCounters counters;
    unsigned long long added = 0, nonFinite = 0;
    for (std::uint32_t s = 0; s < params.samples; ++s)
    {
        const auto outcome = optix::renderSample (params.view, params.settings, params.camera, index.x, index.y, params.film.width,
                                                  params.firstSample + s, params.generation, film, counters);
        if (outcome == ::videohelper::pathtrace::device::SampleOutcome::NonFinite)
            ++nonFinite;
        else
            ++added;
    }
    const auto* words = reinterpret_cast<const unsigned long long*> (&counters);
    for (std::size_t i = 0; i < optix::kPathCounterWords; ++i)
        atomicAdd (&params.totals->counters[i], words[i]);
    atomicAdd (&params.totals->samplesAdded, added);
    atomicAdd (&params.totals->nonFinite, nonFinite);
}

// Ray.ignoreTriangle, as the CPU BVH honours it. The GAS must leave any-hit
// enabled on these triangles (no OPTIX_GEOM_FLAG_DISABLE_ANYHIT).
extern "C" __global__ void __anyhit__pathtrace()
{
    if (optixGetPrimitiveIndex() == optixGetPayload_4())
        optixIgnoreIntersection();
}

extern "C" __global__ void __anyhit__shadow()
{
    if (optixGetPrimitiveIndex() == optixGetPayload_1())
        optixIgnoreIntersection();
}

extern "C" __global__ void __closesthit__pathtrace()
{
    const float2 barycentrics = optixGetTriangleBarycentrics();
    optixSetPayload_0 (__float_as_uint (optixGetRayTmax()));
    optixSetPayload_1 (__float_as_uint (barycentrics.x));
    optixSetPayload_2 (__float_as_uint (barycentrics.y));
    optixSetPayload_3 (optixGetPrimitiveIndex());
}

extern "C" __global__ void __miss__pathtrace() {}

extern "C" __global__ void __miss__shadow()
{
    optixSetPayload_0 (0u);
}
