#pragma once

// What the OptiX host code hands the kernel, shared by device_optix.cu and
// device_optix_host.cpp. Every pointer in here is a device address.

#include "device_tables.h"
#include "gpu_film.h"
#include "scene_view.h"
#include "../scene_ingest.h"

#include <optix_types.h>

#include <cstdint>

namespace videohelper::pathtrace::device::optix
{

// The film and tallies every GPU device fills (gpu_film.h).
using FilmPixel = gpu::FilmPixel;
using OptixFilm = gpu::Film;
using LaunchTotals = gpu::LaunchTotals;
using gpu::kPathCounterWords;

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
