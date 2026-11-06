#pragma once

// What the CUDA and HIP host code hands the compute kernel, shared by
// device_compute_kernel.inl and device_compute_host.cpp. Every pointer in here
// is a device address. The two devices run one kernel source and one host
// class, so they agree on this layout by construction.

#include "device_tables.h"
#include "gpu_film.h"
#include "scene_view.h"
#include "../bvh.h"
#include "../scene_ingest.h"

#include <cstdint>

namespace videohelper::pathtrace::device::compute
{

// The scene with the CPU build of the BVH, copied to device memory.
using SceneViewForLaunch = device::SceneView<BvhView>;

struct LaunchParams
{
    SceneViewForLaunch view;
    IntegratorSettings settings;
    PinholeCamera camera;
    gpu::Film film;
    gpu::LaunchTotals* totals = nullptr;
    DeviceLookupTables tables;
    std::uint32_t height = 0;
    std::uint32_t firstSample = 0;
    std::uint32_t samples = 0;
    std::uint64_t generation = 0;
};

// The kernel's entry point and the constant it reads its parameters from, by the
// names the host looks them up with.
inline constexpr const char* kKernelName = "arbitPathTraceCompute";
inline constexpr const char* kParamsName = "arbitComputeParams";
// Threads per block side: 8 x 8 pixels per block.
inline constexpr unsigned kBlockSide = 8;

} // namespace videohelper::pathtrace::device::compute
