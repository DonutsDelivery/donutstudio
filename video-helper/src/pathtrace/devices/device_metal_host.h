#pragma once

// The Metal device's render pass (Plan 3, Render engines and devices), without
// Objective-C types, so the fixture engine can call it. Exists only in a macOS
// build with ARBIT_HAVE_METAL_PATHTRACE (device_metal_host.mm).
//
// The pass uploads the scene into shared-storage buffers, builds a Metal
// primitive acceleration structure over its triangles, and runs the integrator
// body as an MSL kernel (device_metal.metal) that traces with Metal ray tracing.
// An intersection function skips the triangle a ray leaves from.

#include "../integrator.h"

#include <cstdint>
#include <functional>
#include <string>

namespace videohelper::pathtrace::device
{

// A device pass for PathTracer::setDevicePass, as makeCudaDevicePass: every pass
// shares one device and pipeline, and keeps its own scene, structure and film.
PathTracer::DevicePassFunction makeMetalDevicePass (std::function<std::uint64_t()> sceneBuild);

// The kernel compiles on its own thread, once per process, from the embedded MSL.
// True when a pass can run now, or when the compile failed and the pass will say so.
bool metalPipelineReady();
void startMetalPipelineBuild();
// The user's Cancel and Start again, as for OptiX.
bool stopMetalPipelineBuildByUser();
void startMetalPipelineBuildByUser();
// Waits for a running compile to end. For process exit.
void cancelMetalPipelineBuild();

// Empty when this Mac has a Metal GPU with ray tracing, otherwise why not. The
// registry's probe in a build with the Metal device.
std::string metalDeviceProblem();

// The kernel source the pipeline compiles: the embedded text with the lookup
// tables in place of kTablesMarker. Portable (device_metal_source.cpp), so the
// tests check it on every platform.
std::string metalKernelSource();
// The lookup tables as MSL constant arrays, as exportSpectrumTables and
// exportGgxTables fill them on the host.
std::string metalLookupTableSource();

} // namespace videohelper::pathtrace::device
