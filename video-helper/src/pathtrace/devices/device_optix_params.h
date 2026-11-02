#pragma once

// What the OptiX host code hands the kernel, shared by device_optix.cu and
// device_optix_host.cpp. Every pointer in here is a device address.

#include "device_tables.h"
#include "scene_view.h"
#include "../scene_ingest.h"

#include <optix_types.h>

#include <cstdint>

namespace videohelper::pathtrace::device::optix
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
// thread order, unlike the CPU film's fixed point.
struct OptixFilm
{
    FilmPixel* pixels = nullptr;
    FilmPixel* splats = nullptr;
    unsigned long long* lightPaths = nullptr;
    std::uint32_t width = 0;
};

// One launch's tallies, summed over its threads with atomics.
inline constexpr std::size_t kPathCounterWords = sizeof (PathCounters) / sizeof (std::uint64_t);
static_assert (sizeof (PathCounters) % sizeof (std::uint64_t) == 0);

struct LaunchTotals
{
    unsigned long long counters[kPathCounterWords] {};
    unsigned long long samplesAdded = 0;
    unsigned long long nonFinite = 0;
};

using SceneViewForLaunch = device::SceneView<OptixTraversableHandle>;

struct LaunchParams
{
    SceneViewForLaunch view;
    IntegratorSettings settings;
    PinholeCamera camera;
    OptixFilm film;
    LaunchTotals* totals = nullptr;
    DeviceLookupTables tables;
    std::uint32_t firstSample = 0;
    std::uint32_t samples = 0;
    std::uint64_t generation = 0;
};

} // namespace videohelper::pathtrace::device::optix
