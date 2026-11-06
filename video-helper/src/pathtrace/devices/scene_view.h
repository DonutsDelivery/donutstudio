#pragma once

// The scene as the integrator body sees it: buffer views plus the device's
// acceleration handle. A device fills one from its own uploaded buffers; the
// CPU device points it at a RenderScene (device_cpu.h).

#include "device_interface.h"
#include "../integrator.h"

namespace videohelper::pathtrace::device
{

// LightSetPointer is the Metal kernel's: its launch parameters hold the light
// set's device address, and the kernel's view points at its own copy.
template <typename Accel, typename LightSetPointer = const PT_VIEW_SPACE LightSetView*>
struct SceneView
{
    BufferView<Vec3> positions;                   // three per triangle
    // Motion blur: the same vertices at shutter open and close, or both empty.
    BufferView<Vec3> motionOpen;
    BufferView<Vec3> motionClose;
    BufferView<TriangleAttributes> triangles;
    BufferView<Material> materials;               // by TriangleAttributes::materialIndex
    const PT_GLOBAL Material* fallbackMaterial = nullptr;
    BufferView<GratingBinding> gratings;
    Accel accel {};
    LightSetPointer lights = nullptr;
    bool anyPartialCoverage = false;
    bool anyThinSheet = false;
    bool anyDispersive = false;
    bool anyMediumBoundary = false;
    bool anyVolume = false;
    // The world volume (RenderScene::worldVolume), or null; it fills the ball
    // of worldVolumeRadius around boundsCentre.
    const PT_GLOBAL Material* worldMedium = nullptr;
    float worldVolumeRadius = 0.0f;
    bool anyHiddenObjects = false;
    // Sphere around every triangle: light from infinitely far (directional,
    // environment) enters the light tracer through its disk.
    Vec3 boundsCentre {};
    float boundsRadius = 0.0f;
};

} // namespace videohelper::pathtrace::device
