// CPU device driver TU: compiles the single-source integrator body for the
// CPU device (device_cpu.h).

#define PATHTRACE_DEVICE_CPU 1

#include "device_cpu.h"

namespace videohelper::pathtrace::device::cpu
{

SceneView makeSceneView (const RenderScene& scene, BufferView<GratingBinding> gratings) noexcept
{
    SceneView view;
    view.positions = { scene.geometry.positions.data(), static_cast<std::uint32_t> (scene.geometry.positions.size()) };
    view.triangles = { scene.geometry.triangles.data(), static_cast<std::uint32_t> (scene.geometry.triangles.size()) };
    view.materials = { scene.materials.data(), static_cast<std::uint32_t> (scene.materials.size()) };
    view.fallbackMaterial = &scene.fallbackMaterial;
    view.gratings = gratings.empty()
                      ? BufferView<GratingBinding> { scene.gratings.data(), static_cast<std::uint32_t> (scene.gratings.size()) }
                      : gratings;
    view.accel = &scene.bvh;
    view.lights = &scene.lights;
    view.anyPartialCoverage = scene.anyPartialCoverage;
    view.anyThinSheet = scene.anyThinSheet;
    view.anyDispersive = scene.anyDispersive;
    view.anyMediumBoundary = scene.anyMediumBoundary;
    view.boundsCentre = scene.boundsCentre;
    view.boundsRadius = scene.boundsRadius;
    return view;
}

} // namespace videohelper::pathtrace::device::cpu

#define PT_DEVICE ::videohelper::pathtrace::device::CpuDevice
#define PT_DEVICE_NAMESPACE cpu
#include "integrator_body.inl"
