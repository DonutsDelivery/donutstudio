#pragma once

// What the Metal device's host hands its kernel (Plan 3, Render engines and
// devices). device_metal_host.mm fills it and device_metal.metal reads it, so both
// include this file; the Metal compiler sees it through the kernel source the
// build embeds. Every pointer in here is a buffer's gpuAddress.
//
// The scene is the CUDA kernel's, without the BVH: Metal traces against its own
// primitive acceleration structure, bound beside these parameters with the
// intersection function table that skips the ray's ignored triangle.

#include "gpu_film.h"
#include "scene_view.h"
#include "../scene_ingest.h"

#include <cstdint>

namespace videohelper::pathtrace::device::metal
{

// The scene's slot for the acceleration structure. The kernel binds the structure
// itself, so the host leaves this zero.
struct AccelSlot
{
    std::uint64_t unused = 0;
};

struct LaunchParams
{
    // view.lights is the uploaded light set; the kernel copies it.
    SceneView<AccelSlot, const PT_GLOBAL LightSetView*> view;
    IntegratorSettings settings;
    PinholeCamera camera;
    gpu::Film film;
    PT_GLOBAL gpu::LaunchTotals* totals = nullptr;
    std::uint32_t height = 0;
    std::uint32_t firstSample = 0;
    std::uint32_t samples = 0;
    std::uint32_t padding = 0;
    std::uint64_t generation = 0;
};

// The kernel and its intersection function, by the names the host looks up, and
// the buffer indices the kernel binds. The names are the host's: Metal has no
// program-scope pointer to a string.
#if ! defined(PATHTRACE_DEVICE_METAL)
inline constexpr const char* kKernelName = "arbitPathTraceMetal";
inline constexpr const char* kSkipTriangleName = "arbitSkipIgnoredTriangle";
#endif
inline constexpr unsigned kParamsIndex = 0;
inline constexpr unsigned kAccelIndex = 1;
inline constexpr unsigned kFunctionTableIndex = 2;
// Threads per threadgroup side, as the CUDA kernel's blocks.
inline constexpr unsigned kGroupSide = 8;
// The line in the kernel source the host replaces with the lookup tables as
// constant arrays: Metal has no writable program-scope data to point at them.
#if ! defined(PATHTRACE_DEVICE_METAL)
inline constexpr const char* kTablesMarker = "// ARBIT_METAL_LOOKUP_TABLES";
#endif

} // namespace videohelper::pathtrace::device::metal
