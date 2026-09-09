#pragma once

#include "../../src/sdf_ir_admission.h"

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
} // namespace videohelper::sdf::reference
