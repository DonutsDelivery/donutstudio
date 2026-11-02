#pragma once

// Small value types for the CPU path tracer. Float precision, no SIMD. The
// matrix layout is column-major with 16 floats, the same layout the native
// raster backends upload, so a matrix built here matches theirs entry for entry.

#include "devices/device_annotations.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace videohelper::pathtrace
{

struct Vec2
{
    float x = 0.0f;
    float y = 0.0f;
};

struct Vec3
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    PT_SHARED_FN float operator[] (int axis) const noexcept { return axis == 0 ? x : axis == 1 ? y : z; }
    PT_SHARED_FN float& operator[] (int axis) noexcept { return axis == 0 ? x : axis == 1 ? y : z; }
};

PT_SHARED_FN Vec3 operator+ (Vec3 a, Vec3 b) noexcept { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
PT_SHARED_FN Vec3 operator- (Vec3 a, Vec3 b) noexcept { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
PT_SHARED_FN Vec3 operator- (Vec3 a) noexcept { return { -a.x, -a.y, -a.z }; }
PT_SHARED_FN Vec3 operator* (Vec3 a, float s) noexcept { return { a.x * s, a.y * s, a.z * s }; }
PT_SHARED_FN Vec3 operator* (float s, Vec3 a) noexcept { return a * s; }
PT_SHARED_FN Vec3 operator* (Vec3 a, Vec3 b) noexcept { return { a.x * b.x, a.y * b.y, a.z * b.z }; }
PT_SHARED_FN float dot (Vec3 a, Vec3 b) noexcept { return a.x * b.x + a.y * b.y + a.z * b.z; }
PT_SHARED_FN Vec3 cross (Vec3 a, Vec3 b) noexcept
{
    return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}
PT_SHARED_FN float length (Vec3 a) noexcept { return std::sqrt (dot (a, a)); }
PT_SHARED_FN Vec3 vmin (Vec3 a, Vec3 b) noexcept
{
    return { std::min (a.x, b.x), std::min (a.y, b.y), std::min (a.z, b.z) };
}
PT_SHARED_FN Vec3 vmax (Vec3 a, Vec3 b) noexcept
{
    return { std::max (a.x, b.x), std::max (a.y, b.y), std::max (a.z, b.z) };
}
PT_SHARED_FN bool isFinite (Vec3 a) noexcept
{
    return std::isfinite (a.x) && std::isfinite (a.y) && std::isfinite (a.z);
}

// Light past a float holds the largest float. A NaN stays, so isFinite still catches it.
PT_SHARED_FN Vec3 saturatedRadiance (Vec3 a) noexcept
{
    const float top = std::numeric_limits<float>::max();
    return { a.x > top ? top : a.x, a.y > top ? top : a.y, a.z > top ? top : a.z };
}

// Returns the zero vector when the input has no finite direction.
PT_SHARED_FN Vec3 normalizeOrZero (Vec3 a) noexcept
{
    const float len = length (a);
    if (! std::isfinite (len) || len <= 0.0f)
        return {};
    return a * (1.0f / len);
}

// Unit quaternion in glTF x, y, z, w order.
struct Quat
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 1.0f;
};

// Same formula as the raster shaders' rotateQ: v + 2 q.xyz x (q.xyz x v + w v).
PT_SHARED_FN Vec3 rotate (Quat q, Vec3 v) noexcept
{
    const Vec3 u { q.x, q.y, q.z };
    return v + 2.0f * cross (u, cross (u, v) + q.w * v);
}

PT_SHARED_FN Quat conjugate (Quat q) noexcept { return { -q.x, -q.y, -q.z, q.w }; }

// Column-major 4x4. Element (row, column) is m[column * 4 + row].
struct Mat4
{
    std::array<float, 16> m { 1.0f, 0.0f, 0.0f, 0.0f,
                              0.0f, 1.0f, 0.0f, 0.0f,
                              0.0f, 0.0f, 1.0f, 0.0f,
                              0.0f, 0.0f, 0.0f, 1.0f };
};

PT_SHARED_FN Mat4 operator* (const Mat4& left, const Mat4& right) noexcept
{
    Mat4 result;
    for (float& entry : result.m)
        entry = 0.0f;
    for (int column = 0; column < 4; ++column)
        for (int row = 0; row < 4; ++row)
            for (int inner = 0; inner < 4; ++inner)
                result.m[static_cast<std::size_t> (column * 4 + row)]
                    += left.m[static_cast<std::size_t> (inner * 4 + row)]
                     * right.m[static_cast<std::size_t> (column * 4 + inner)];
    return result;
}

PT_SHARED_FN Vec3 transformPoint (const Mat4& t, Vec3 p) noexcept
{
    const auto& m = t.m;
    return { m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12],
             m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13],
             m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14] };
}

PT_SHARED_FN Vec3 transformVector (const Mat4& t, Vec3 v) noexcept
{
    const auto& m = t.m;
    return { m[0] * v.x + m[4] * v.y + m[8] * v.z,
             m[1] * v.x + m[5] * v.y + m[9] * v.z,
             m[2] * v.x + m[6] * v.y + m[10] * v.z };
}

PT_SHARED_FN float determinant3x3 (const Mat4& t) noexcept
{
    const auto& m = t.m;
    return m[0] * (m[5] * m[10] - m[9] * m[6])
         - m[4] * (m[1] * m[10] - m[9] * m[2])
         + m[8] * (m[1] * m[6] - m[5] * m[2]);
}

// Row-major 3x3 used to carry normals: the inverse transpose of a Mat4's
// linear part. A singular input yields the zero matrix, so its normals fall
// back to the geometric normal.
struct NormalMatrix
{
    std::array<float, 9> r {};

    PT_SHARED_FN Vec3 apply (Vec3 n) const noexcept
    {
        return { r[0] * n.x + r[1] * n.y + r[2] * n.z,
                 r[3] * n.x + r[4] * n.y + r[5] * n.z,
                 r[6] * n.x + r[7] * n.y + r[8] * n.z };
    }
};

PT_SHARED_FN NormalMatrix normalMatrixOf (const Mat4& t) noexcept
{
    // a(row, col) of the linear part.
    const auto a = [&t] (int row, int col) { return t.m[static_cast<std::size_t> (col * 4 + row)]; };
    const float det = determinant3x3 (t);
    NormalMatrix out;
    if (! std::isfinite (det) || det == 0.0f)
        return out;
    const float inv = 1.0f / det;
    // The inverse transpose equals the cofactor matrix divided by the determinant.
    out.r[0] = (a (1, 1) * a (2, 2) - a (1, 2) * a (2, 1)) * inv;
    out.r[1] = (a (1, 2) * a (2, 0) - a (1, 0) * a (2, 2)) * inv;
    out.r[2] = (a (1, 0) * a (2, 1) - a (1, 1) * a (2, 0)) * inv;
    out.r[3] = (a (0, 2) * a (2, 1) - a (0, 1) * a (2, 2)) * inv;
    out.r[4] = (a (0, 0) * a (2, 2) - a (0, 2) * a (2, 0)) * inv;
    out.r[5] = (a (0, 1) * a (2, 0) - a (0, 0) * a (2, 1)) * inv;
    out.r[6] = (a (0, 1) * a (1, 2) - a (0, 2) * a (1, 1)) * inv;
    out.r[7] = (a (0, 2) * a (1, 0) - a (0, 0) * a (1, 2)) * inv;
    out.r[8] = (a (0, 0) * a (1, 1) - a (0, 1) * a (1, 0)) * inv;
    return out;
}

// Axis-aligned bounds. A default instance is empty and grows to fit.
struct Bounds
{
    Vec3 lo { std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
              std::numeric_limits<float>::infinity() };
    Vec3 hi { -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
              -std::numeric_limits<float>::infinity() };

    PT_SHARED_FN void grow (Vec3 p) noexcept { lo = vmin (lo, p); hi = vmax (hi, p); }
    PT_SHARED_FN void grow (const Bounds& b) noexcept { lo = vmin (lo, b.lo); hi = vmax (hi, b.hi); }
    PT_SHARED_FN bool empty() const noexcept { return lo.x > hi.x || lo.y > hi.y || lo.z > hi.z; }
    PT_SHARED_FN Vec3 extent() const noexcept { return empty() ? Vec3 {} : hi - lo; }
    PT_SHARED_FN Vec3 centre() const noexcept { return (lo + hi) * 0.5f; }
    PT_SHARED_FN float halfArea() const noexcept
    {
        const Vec3 e = extent();
        return e.x * e.y + e.y * e.z + e.z * e.x;
    }
};

PT_SHARED_FN std::uint32_t floatBits (float f) noexcept
{
#if defined(__CUDACC__)
    return __float_as_uint (f);
#else
    std::uint32_t bits = 0;
    std::memcpy (&bits, &f, sizeof bits);
    return bits;
#endif
}

PT_SHARED_FN float bitsToFloat (std::uint32_t bits) noexcept
{
#if defined(__CUDACC__)
    return __uint_as_float (bits);
#else
    float f = 0.0f;
    std::memcpy (&f, &bits, sizeof f);
    return f;
#endif
}

// std::upper_bound over floats: the index of the first value greater than
// `value`, or count. Device code cannot call the standard one under C++17.
PT_SHARED_FN std::size_t upperBoundIndex (const float* values, std::size_t count, float value) noexcept
{
    std::size_t first = 0;
    while (count > 0)
    {
        const std::size_t half = count / 2;
        if (value < values[first + half])
            count = half;
        else
        {
            first += half + 1;
            count -= half + 1;
        }
    }
    return first;
}

} // namespace videohelper::pathtrace
