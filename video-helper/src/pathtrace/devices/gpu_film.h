#pragma once

// The film and launch tallies every GPU device's kernel fills: OptiX, CUDA, HIP
// and oneAPI. The host reads them back into the Accumulator after a pass. Every pointer
// in here is a device address (PT_GLOBAL).

#include "device_annotations.h"
#include "../integrator.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace videohelper::pathtrace::device::gpu
{

struct FilmPixel
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 0.0f;
};

// Film: one FilmPixel per pixel (rgb sum, sample count), added with atomics
// because a launch may give several threads the same pixel. Light paths add
// to splats (rgb sum) and count themselves in lightPaths; the resolve divides
// the splat sum by that count. Float atomics make the splat sum depend on
// thread order, unlike the CPU film's fixed point. Each added sample adds its
// first-hit guides too.
struct Film
{
    PT_GLOBAL FilmPixel* pixels = nullptr;
    PT_GLOBAL FilmPixel* splats = nullptr;
    PT_GLOBAL PtCounterWord* lightPaths = nullptr;
    // The denoiser's first-hit guides, summed: albedo with the sample count in w,
    // and the normal.
    PT_GLOBAL FilmPixel* guideAlbedo = nullptr;
    PT_GLOBAL FilmPixel* guideNormal = nullptr;
    std::uint32_t width = 0;
};

// One launch's tallies, summed over its threads with atomics.
inline constexpr std::size_t kPathCounterWords = sizeof (PathCounters) / sizeof (std::uint64_t);
static_assert (sizeof (PathCounters) % sizeof (std::uint64_t) == 0);

struct LaunchTotals
{
    PtCounterWord counters[kPathCounterWords] {};
    PtCounterWord samplesAdded = 0;
    PtCounterWord nonFinite = 0;
};

#if defined(__CUDACC__)
// The device shims' film writes. A non-finite channel counts as 0, as
// Accumulator::addSample has it.
__device__ __forceinline__ float finiteOrZero (float value) { return isfinite (value) ? value : 0.0f; }

__device__ __forceinline__ void filmAdd (Film& film, std::uint32_t x, std::uint32_t y, Vec3 rgb)
{
    PT_GLOBAL FilmPixel& pixel = film.pixels[y * film.width + x];
    atomicAdd (&pixel.x, finiteOrZero (rgb.x));
    atomicAdd (&pixel.y, finiteOrZero (rgb.y));
    atomicAdd (&pixel.z, finiteOrZero (rgb.z));
    atomicAdd (&pixel.w, 1.0f);
}

__device__ __forceinline__ void filmSplat (Film& film, std::uint32_t x, std::uint32_t y, Vec3 rgb)
{
    PT_GLOBAL FilmPixel& splat = film.splats[y * film.width + x];
    atomicAdd (&splat.x, finiteOrZero (rgb.x));
    atomicAdd (&splat.y, finiteOrZero (rgb.y));
    atomicAdd (&splat.z, finiteOrZero (rgb.z));
}

__device__ __forceinline__ void filmCountLightPath (Film& film) { atomicAdd (film.lightPaths, PtCounterWord (1)); }

__device__ __forceinline__ void filmGuide (Film& film, std::uint32_t x, std::uint32_t y, Vec3 albedo, Vec3 normal)
{
    PT_GLOBAL FilmPixel& a = film.guideAlbedo[y * film.width + x];
    PT_GLOBAL FilmPixel& n = film.guideNormal[y * film.width + x];
    atomicAdd (&a.x, finiteOrZero (albedo.x));
    atomicAdd (&a.y, finiteOrZero (albedo.y));
    atomicAdd (&a.z, finiteOrZero (albedo.z));
    atomicAdd (&a.w, 1.0f);
    atomicAdd (&n.x, finiteOrZero (normal.x));
    atomicAdd (&n.y, finiteOrZero (normal.y));
    atomicAdd (&n.z, finiteOrZero (normal.z));
}

// A thread's tallies into the launch totals.
__device__ __forceinline__ void addTotals (PT_GLOBAL LaunchTotals* totals, const PathCounters& counters,
                                           PtCounterWord added, PtCounterWord nonFinite)
{
    const auto* words = reinterpret_cast<const PtCounterWord*> (&counters);
    for (std::size_t i = 0; i < kPathCounterWords; ++i)
        atomicAdd (&totals->counters[i], words[i]);
    atomicAdd (&totals->samplesAdded, added);
    atomicAdd (&totals->nonFinite, nonFinite);
}
#endif

} // namespace videohelper::pathtrace::device::gpu
