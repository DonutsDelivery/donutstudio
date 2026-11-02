#pragma once

// The scene as the integrator body sees it: buffer views plus the device's
// acceleration handle. A device fills one from its own uploaded buffers; the
// CPU device points it at a RenderScene (device_cpu.h).

#include "device_interface.h"
#include "../integrator.h"

namespace videohelper::pathtrace::device
{

template <typename Accel>
struct SceneView
{
    BufferView<Vec3> positions;                   // three per triangle
    BufferView<TriangleAttributes> triangles;
    BufferView<Material> materials;               // by TriangleAttributes::materialIndex
    const Material* fallbackMaterial = nullptr;
    BufferView<GratingBinding> gratings;
    Accel accel {};
    const LightSetView* lights = nullptr;
    bool anyPartialCoverage = false;
    bool anyThinSheet = false;
    bool anyDispersive = false;
    bool anyMediumBoundary = false;
    // Sphere around every triangle: light from infinitely far (directional,
    // environment) enters the light tracer through its disk.
    Vec3 boundsCentre {};
    float boundsRadius = 0.0f;
};

} // namespace videohelper::pathtrace::device
