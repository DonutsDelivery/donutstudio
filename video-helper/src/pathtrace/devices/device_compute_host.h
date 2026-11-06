#pragma once

// The CUDA, HIP and oneAPI devices' render passes (Plan 3, Render engines and
// devices). All three run the integrator body over the CPU build of the BVH
// (device_compute_kernel.inl for CUDA and HIP, device_oneapi.cpp for SYCL) and
// share one host class over driver APIs that load at run time: libcuda,
// libamdhip64, and the oneAPI device library beside the helper, which brings the
// SYCL and Level Zero runtimes. A build without a device's kernel still links
// this file and reports the missing kernel by name.

#include "device_registry.h"
#include "../integrator.h"

#include <cstdint>
#include <functional>
#include <string>

namespace videohelper::pathtrace::device
{

// A device pass for PathTracer::setDevicePass, as makeOptixDevicePass: every pass
// shares one context and module per device, and keeps its own scene and film.
PathTracer::DevicePassFunction makeCudaDevicePass (std::function<std::uint64_t()> sceneBuild);
PathTracer::DevicePassFunction makeHipDevicePass (std::function<std::uint64_t()> sceneBuild);
PathTracer::DevicePassFunction makeOneApiDevicePass (std::function<std::uint64_t()> sceneBuild);

// CUDA, HIP or oneAPI: the devices this file drives.
bool isComputeDevice (DeviceKind kind) noexcept;

// The module loads on its own thread, once per process: the driver compiles the
// embedded kernel on first use and caches it. True when a pass can run now, or when
// the load failed and the pass will say so.
bool cudaModuleReady();
bool hipModuleReady();
bool oneApiModuleReady();
void startComputeModuleBuild (DeviceKind kind);
// The user's Cancel and Start again, as for OptiX. A load already inside the driver
// runs to its end; Cancel holds the preview until Start again.
bool stopComputeModuleBuildByUser (DeviceKind kind);
void startComputeModuleBuildByUser (DeviceKind kind);
// Waits for a running load to end. For process exit.
void cancelComputeModuleBuild (DeviceKind kind);

// Empty when the oneAPI device library loads and finds an Intel GPU, otherwise
// why not: the library or the SYCL runtime it needs did not load, or the runtime
// found no GPU. Asked once per process; the registry's probe for this build.
std::string oneApiDeviceProblem();

} // namespace videohelper::pathtrace::device
