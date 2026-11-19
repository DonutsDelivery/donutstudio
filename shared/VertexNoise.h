#pragma once

#include <cmath>
#include <cstdint>
#include <string>

// Value noise for the Vertex Noise 3D nodes. The C++ function runs Path Trace's
// pre-BVH displacement, and source() writes the same steps as OpenGL or Metal
// statements for Raster: the same integer hash, the same 32-bit float corner
// values and smoothstep weights, summed over the eight lattice corners in the
// same order. The result lies in [-1, 1]. A vector noise draws its three lanes
// with seeds seed, seed + 1 and seed + 2.
namespace videowire::vertexnoise
{
inline std::uint32_t hash (std::uint32_t x) noexcept
{
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

inline float corner (std::int32_t x, std::int32_t y, std::int32_t z, std::uint32_t seed) noexcept
{
    auto h = hash ((static_cast<std::uint32_t> (z) * 0xcb1ab31fU) ^ seed);
    h = hash ((static_cast<std::uint32_t> (y) * 0xd8163841U) ^ h);
    h = hash ((static_cast<std::uint32_t> (x) * 0x8da6b343U) ^ h);
    return static_cast<float> (h >> 8) * (2.0f / 16777215.0f) - 1.0f;
}

inline float noise (float x, float y, float z, float frequency, std::uint32_t seed) noexcept
{
    const float q[3] { x * frequency, y * frequency, z * frequency };
    std::int32_t cell[3] {};
    float weight[3] {};
    for (int axis = 0; axis < 3; ++axis)
    {
        const float floor = std::floor (q[axis]);
        const float t = q[axis] - floor;
        cell[axis] = static_cast<std::int32_t> (floor);
        weight[axis] = t * t * (3.0f - 2.0f * t);
    }
    float sum = 0.0f;
    for (int c = 0; c < 8; ++c)
    {
        const int dx = c & 1, dy = (c >> 1) & 1, dz = (c >> 2) & 1;
        const float w = (dx ? weight[0] : 1.0f - weight[0]) * (dy ? weight[1] : 1.0f - weight[1])
            * (dz ? weight[2] : 1.0f - weight[2]);
        sum += w * corner (cell[0] + dx, cell[1] + dy, cell[2] + dz, seed);
    }
    return sum;
}

// Statements that declare `float <name>` holding noise(position, frequency,
// seed). `position` is a vec3/float3 expression, `frequency` a float literal.
inline std::string source (bool metal, const std::string& name, const std::string& position,
                           const std::string& frequency, std::uint32_t seed)
{
    const std::string f3 = metal ? "float3" : "vec3", i3 = metal ? "int3" : "ivec3";
    const std::string u = "u";
    const auto hashStep = [&] (const std::string& lattice, const std::string& constant, const std::string& previous)
    {
        return "h = (uint(" + lattice + ") * " + constant + u + ") ^ " + previous + ";\n"
            "h ^= h >> 16; h *= 0x7feb352d" + u + "; h ^= h >> 15; h *= 0x846ca68b" + u + "; h ^= h >> 16;\n";
    };
    return "float " + name + " = 0.0;\n{\n"
        + f3 + " q = (" + position + ") * " + frequency + ";\n"
        + f3 + " fl = floor(q);\n"
        + f3 + " t = q - fl;\n"
        + f3 + " w = t * t * (3.0 - 2.0 * t);\n"
        + i3 + " cell = " + i3 + "(fl);\n"
        + "for (int c = 0; c < 8; ++c) {\n"
        + "int dx = c & 1; int dy = (c >> 1) & 1; int dz = (c >> 2) & 1;\n"
        + "float k = (dx != 0 ? w.x : 1.0 - w.x) * (dy != 0 ? w.y : 1.0 - w.y) * (dz != 0 ? w.z : 1.0 - w.z);\n"
        + "uint h = " + std::to_string (seed) + u + ";\n"
        + hashStep ("cell.z + dz", "0xcb1ab31f", "h")
        + hashStep ("cell.y + dy", "0xd8163841", "h")
        + hashStep ("cell.x + dx", "0x8da6b343", "h")
        + name + " += k * (float(h >> 8" + u + ") * (2.0 / 16777215.0) - 1.0);\n}\n}\n";
}
} // namespace videowire::vertexnoise
