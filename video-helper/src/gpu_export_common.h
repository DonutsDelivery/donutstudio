// gpu_export_common.h — selects the platform's zero-copy GPU exporter for
// the shared docked viewport (SharedGpuSurfaceProtocol.h) and exposes it
// under one name, `gpuexp::Exporter`, so viewport.cpp's shared-mode code is
// platform-agnostic. Exactly one backend compiles per platform:
//
//   ARBIT_HAVE_DMABUF    (Linux)   — gpu_export.h, EGL dmabuf export
//   ARBIT_HAVE_IOSURFACE (macOS)   — gpu_export_iosurface.h, IOSurface + CGL
//   ARBIT_HAVE_D3D       (Windows) — gpu_export_d3d.h, D3D11 shared textures
//                                    bridged into GL via WGL_NV_DX_interop2
//
// Every backend defines the same `gpuexp::ExportedBuffer` field set used by
// the generic BUFFERS announcement (width/height/fourcc/modifier/planeCount/
// fds/strides/offsets/busy) and the same Exporter method shape (compile-time
// duck typing — see gpu_export.h for the canonical API doc). Backends
// without fd-based handles report planeCount = 0 and carry their handle in
// `modifier` (IOSurfaceID / DXGI shared HANDLE).
//
// ARBIT_HAVE_GPU_SHARED is 1 when any backend is compiled in — viewport.cpp
// gates all shared-mode code on it.
#pragma once

#if ARBIT_HAVE_VKFD

// Linux, Vulkan external memory (OPAQUE_FD) → GL_EXT_memory_object_fd. Preferred
// over the EGL dmabuf backend because it works on the NVIDIA proprietary driver
// under GLX (where EGL_MESA dmabuf export is unavailable) and is windowing-
// system agnostic. Vulkan-capable Mesa GPUs use it too. Falls back to shm-docked
// at runtime if the Vulkan device/extension probe fails (see viewport.cpp).
#include "gpu_export_vk.h"
#define ARBIT_HAVE_GPU_SHARED 1
namespace gpuexp
{
using Exporter = VkFdExporter;
constexpr const char* kSharedPathName = "vkfd-docked";
constexpr const char* kSharedGpuTag = "vkfd";
} // namespace gpuexp

#elif ARBIT_HAVE_DMABUF

#include "gpu_export.h"
#define ARBIT_HAVE_GPU_SHARED 1
namespace gpuexp
{
using Exporter = DmabufExporter;
constexpr const char* kSharedPathName = "dmabuf-docked";
constexpr const char* kSharedGpuTag = "dmabuf";
} // namespace gpuexp

#elif ARBIT_HAVE_IOSURFACE

#include "gpu_export_iosurface.h"
#define ARBIT_HAVE_GPU_SHARED 1
namespace gpuexp
{
using Exporter = IOSurfaceExporter;
constexpr const char* kSharedPathName = "iosurface-docked";
constexpr const char* kSharedGpuTag = "iosurface";
} // namespace gpuexp

#elif ARBIT_HAVE_D3D

#include "gpu_export_d3d.h"
#define ARBIT_HAVE_GPU_SHARED 1
namespace gpuexp
{
using Exporter = D3DInteropExporter;
constexpr const char* kSharedPathName = "d3d11-docked";
constexpr const char* kSharedGpuTag = "d3d11";
} // namespace gpuexp

#else

#define ARBIT_HAVE_GPU_SHARED 0

#endif
