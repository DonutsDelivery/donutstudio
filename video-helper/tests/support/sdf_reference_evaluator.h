#pragma once

#include "../../src/sdf_ir_admission.h"
#include "../../src/sdf_native_render_admission.h"

#include <array>

namespace videohelper::sdf::reference
{
struct Point3 final
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

// Test-only scalar oracle for the normative SdfIr point-distance contract.
// Production renderers must not call this CPU implementation.
double evaluatePoint (const AdmittedSdfIr& geometry, Point3 point);

struct PointSample final
{
    double distance = 0.0;
    videowire::SdfStableId primitiveId = 0;
};

struct SurfaceSample final
{
    Point3 normal;
    double meanCurvature = 0.0; // Inverse scene units, positive on a convex sphere.
    double ambientVisibility = 0.0;
    double lightVisibility = 0.0;
};

PointSample evaluateSample (const AdmittedSdfIr& geometry, Point3 point);
SurfaceSample evaluateSurface (const AdmittedSdfIr& geometry, Point3 point,
                               const NativeSdfRenderControls& controls);
// Pixel centers and camera match the existing native SDF ray convention.
// This independently evaluates the scalar IR, without the GPU compiled records.
std::array<double, 4> evaluateOutputPixel (const AdmittedSdfIr& geometry,
                                        const NativeSdfRenderControls& controls,
                                        std::uint32_t width, std::uint32_t height,
                                        double x, double y);
} // namespace videohelper::sdf::reference
