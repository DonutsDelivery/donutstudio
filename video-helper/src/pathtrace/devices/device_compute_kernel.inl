// The compute kernel of the CUDA and HIP devices (Plan 3, Render engines and
// devices). device_cuda.cu compiles it with nvcc and device_hip.hip with hipcc.
// It runs the single-source integrator body like the CPU and OptiX devices, and
// traverses the CPU build of the BVH, uploaded as it is, with the traversal the
// CPU uses (bvh_traversal.h). Only OptiX traverses its own GAS.

#if ! defined(__CUDACC__)
#error "device_compute_kernel.inl is device code: include it from device_cuda.cu or device_hip.hip"
#endif

#include "device_compute_device.h"
#include "device_compute_params.h"

extern "C" __constant__ ::videohelper::pathtrace::device::compute::LaunchParams arbitComputeParams;

namespace videohelper::pathtrace::device
{
PT_SHARED_ENTRY const DeviceLookupTables& deviceLookupTables() noexcept
{
    return arbitComputeParams.tables;
}
} // namespace videohelper::pathtrace::device

// One TU, so the out-of-line shared code comes in as source, as for OptiX.
#include "../spectrum.cpp"
#include "../bsdf.cpp"
#include "../environment_map.cpp"
#include "../lights.cpp"

#define PT_DEVICE ::videohelper::pathtrace::device::ComputeDevice
#define PT_DEVICE_NAMESPACE compute
#include "integrator_body.inl"

// One thread per pixel, the same per-sample kernel as the CPU and OptiX devices.
extern "C" __global__ void arbitPathTraceCompute()
{
    namespace pt = ::videohelper::pathtrace;
    namespace compute = pt::device::compute;
    const compute::LaunchParams& params = arbitComputeParams;
    const std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.film.width || y >= params.height)
        return;
    pt::device::gpu::Film film = params.film;
    pt::PathCounters counters;
    unsigned long long added = 0, nonFinite = 0;
    for (std::uint32_t s = 0; s < params.samples; ++s)
    {
        const auto outcome = compute::renderSample (params.view, params.settings, params.camera, x, y, params.film.width,
                                                    params.firstSample + s, params.generation, film, counters);
        if (outcome == pt::device::SampleOutcome::NonFinite)
            ++nonFinite;
        else
            ++added;
    }
    pt::device::gpu::addTotals (params.totals, counters, added, nonFinite);
}
