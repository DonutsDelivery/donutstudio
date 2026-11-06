#pragma once

// The device struct of the CUDA, HIP and oneAPI kernels (Plan 3, Render engines
// and devices): the integrator body's device members over the CPU build of the
// BVH and the GPU film. device_compute_kernel.inl (CUDA, HIP) and
// device_oneapi.cpp (SYCL) compile the body with it.

#if ! defined(__CUDACC__)
#error "device_compute_device.h is device code: include it from a GPU kernel TU"
#endif

#include "device_interface.h"
#include "gpu_film.h"
#include "scene_view.h"
#include "../bvh_traversal.h"
#include "../sampler.h"

namespace videohelper::pathtrace::device
{

struct ComputeDevice
{
    using Accel = BvhView;
    using Film = gpu::Film;
    using Sampler = PathSampler;

    PT_DEVICE_FN static Sampler makeSampler (std::uint32_t pixel, std::uint32_t sampleIndex, std::uint64_t generation,
                                             SamplePattern pattern) noexcept
    {
        return Sampler (pixel, sampleIndex, generation, pattern);
    }
    PT_DEVICE_FN static bool intersect (const Accel& accel, const Ray& ray, Hit& hit) noexcept
    {
        return traverseClosest (accel, ray, hit);
    }
    PT_DEVICE_FN static bool occluded (const Accel& accel, const Ray& ray) noexcept { return traverseAny (accel, ray); }
    PT_DEVICE_FN static void filmAdd (Film& film, std::uint32_t x, std::uint32_t y, Vec3 rgb) noexcept
    {
        gpu::filmAdd (film, x, y, rgb);
    }
    PT_DEVICE_FN static void filmSplat (Film& film, std::uint32_t x, std::uint32_t y, Vec3 rgb) noexcept
    {
        gpu::filmSplat (film, x, y, rgb);
    }
    PT_DEVICE_FN static void filmCountLightPath (Film& film) noexcept { gpu::filmCountLightPath (film); }
    PT_DEVICE_FN static void filmGuide (Film& film, std::uint32_t x, std::uint32_t y, Vec3 albedo, Vec3 normal) noexcept
    {
        gpu::filmGuide (film, x, y, albedo, normal);
    }
};

} // namespace videohelper::pathtrace::device
