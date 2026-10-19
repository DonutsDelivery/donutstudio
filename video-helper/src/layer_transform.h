// layer_transform.h - the clip layer transform shared by the OpenGL compositor
// (renderer.cpp, which serves both the live viewport and the exporter) and the
// Metal compositor (gpu_backend/backend_sokol.mm).
//
// The quad lives in canvas NDC, where x and y span the same -1..1 range but not
// the same number of pixels unless the canvas is square. A rotation applied
// straight in NDC therefore shears a 16:9 frame into a parallelogram. The
// rotation here runs in aspect-corrected space instead:
//
//   M = T(tx,ty) * A^-1 * R(deg) * A * S(sx,sy),   A = diag(aspect, 1)
//
// with aspect = canvas width / canvas height. A turns NDC into canvas-pixel
// proportions, R rotates there, and A^-1 maps back. Corners stay at right angles
// and edge lengths are the same as at 0 degrees. Position (tx,ty) and scale
// (sx,sy) keep their old NDC meaning, so existing projects move exactly as before
// at 0 degrees. +rotation turns clockwise on screen, +ty moves down.
//
// Column-major, ready for a mat4 uniform or a float4x4.

#pragma once

#include <cmath>
#include <cstring>

namespace videorender
{

inline void buildLayerTransform (float tx, float ty, float rotationDeg,
                                 float sx, float sy, float aspect, float out[16])
{
    if (! (aspect > 0.0f) || ! std::isfinite (aspect))
        aspect = 1.0f;
    const float a = rotationDeg * 3.14159265358979323846f / 180.0f;
    const float c = std::cos (a);
    const float s = std::sin (a);
    const float m[16] = {
        sx * c,           sx * s * aspect,  0.0f, 0.0f,
       -sy * s / aspect,  sy * c,           0.0f, 0.0f,
        0.0f,             0.0f,             1.0f, 0.0f,
        tx,               ty,               0.0f, 1.0f,
    };
    std::memcpy (out, m, sizeof (m));
}

} // namespace videorender
