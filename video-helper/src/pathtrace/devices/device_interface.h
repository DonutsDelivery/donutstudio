#pragma once

// Device interface of the single-source path integrator (Plan 3, Tier V).
//
// integrator_body.inl holds the path loop once. Each device type compiles it
// in its own driver TU after defining a backend macro and a device struct:
//
//   device_cpu.cpp    PATHTRACE_DEVICE_CPU    CpuDevice    (always built)
//   device_optix.cu   PATHTRACE_DEVICE_OPTIX  OptixDevice  (ARBIT_HAVE_OPTIX_KERNEL, when configure finds the SDK and nvcc)
//   device_cuda.cu    PATHTRACE_DEVICE_CUDA   ComputeDevice (ARBIT_HAVE_CUDA_KERNEL, nvcc)
//   device_hip.hip    PATHTRACE_DEVICE_HIP    ComputeDevice (ARBIT_HAVE_HIP_KERNEL, hipcc)
//   device_oneapi.cpp PATHTRACE_DEVICE_ONEAPI ComputeDevice (ARBIT_HAVE_ONEAPI_DEVICE, icpx -fsycl, its own library)
//
// device_registry.h lists the devices and selects one.
//
// The body reaches the device only through the members below. Everything else
// it calls (light sampling, environment importance sampling, spectral
// transport, the BSDF lobes, diffraction orders) is the shared algorithm code
// in this directory and in diffraction_order_bsdf.cpp, never a per-device copy.
//
// A device struct supplies:
//
//   using Accel   = ...;  // acceleration structure handle in SceneView
//   using Film    = ...;  // film the kernel accumulates into
//   using Sampler = ...;  // RNG: get1D (dimension), get2D (dimension), as PathSampler
//   static Sampler makeSampler (pixel, sampleIndex, generation, pattern);
//   static bool intersect (const Accel&, const Ray&, Hit&);   // closest hit
//   static bool occluded (const Accel&, const Ray&);          // any hit
//   static void filmAdd (Film&, x, y, Vec3 rgb);              // must be safe for
//                                                             // concurrent pixels
//   static void filmSplat (Film&, x, y, Vec3 rgb);            // a light path's
//                                                             // contribution; any
//                                                             // thread, any pixel
//   static void filmCountLightPath (Film&);                   // one per traced
//                                                             // light path
//   static void filmGuide (Film&, x, y, Vec3 albedo,          // first-hit guides
//                          Vec3 normal);                      // of an added sample
//
// Each device keeps its own conformance test against the CPU device on the
// shared cases (tests/pathtrace_device_tests.cpp for the CPU device itself).
//
// device_annotations.h marks the shared code for both compilers.
// tests/pathtrace_device_portability_source_contract.py lists every referenced
// symbol and keeps the list of host-only ones from growing.

#include "device_annotations.h"
#include "../../../../shared/DiffractionMaterialIR.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace videohelper::pathtrace::device
{

// Read-only view of one device buffer.
template <typename T>
struct BufferView
{
    const PT_GLOBAL T* data = nullptr;
    std::size_t count = 0;

    PT_SHARED_FN const PT_GLOBAL T& operator[] (std::size_t index) const noexcept { return data[index]; }
    PT_SHARED_FN bool empty() const noexcept { return count == 0; }
    PT_SHARED_FN std::size_t size() const noexcept { return count; }
};

// A material index whose surfaces scatter by diffraction orders instead of
// their Material lobes. RenderScene::gratings holds the bindings of a
// prepared scene (PrepareOptions::objectGratings); the helper binds the
// admitted diffraction material of a fixture's grating object this way. It holds
// the admitted description by value and no pointers, so a device upload copies
// the bindings as they are.
struct GratingBinding
{
    std::uint32_t materialIndex = 0;
    diffractionmaterial::Description description {};
    // A spatial foil's corners in the order (0,0), (1,0), (0,1), (1,1) of uv, each
    // reciprocal direction u, v, period in nm and diffraction coverage.
    std::array<std::array<float, 4>, 4> field {};
    bool varied = false;  // the corners differ in direction or period
    bool masked = false;  // a corner's coverage is below 1
};
static_assert (std::is_trivially_copyable_v<GratingBinding>);

// Outcome of one pixel sample. A non-finite estimate is dropped and counted;
// it never ends the frame.
enum class SampleOutcome : std::uint8_t
{
    Added = 0,
    NonFinite = 1
};

} // namespace videohelper::pathtrace::device
