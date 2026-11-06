#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace arbitgpu::fixturedepth
{
// The fixture vertex shader maps camera distance to clip depth over this range.
// The camera's authored near and far stay in uProjection: linear depth, ray
// ranges and offsets read them. Only the depth test reads this range, so a far
// camera with a small near plane keeps its precision on the scene it frames.
// The Grooved Disc sat 80 units away with near 0.1 and far 400, where one
// 24-bit depth step was about 4 mm, and its 1 mm grating film lost to the body
// beneath it in stripes.
struct Range
{
    float nearest = std::numeric_limits<float>::max();
    float farthest = std::numeric_limits<float>::lowest();
    bool empty() const noexcept { return !(nearest <= farthest); }
};

using Matrix = std::array<float, 16>;

inline std::array<float, 3> rotate(const std::array<float, 4>& q, const std::array<float, 3>& v) noexcept
{
    // v + 2 q.xyz x (q.xyz x v + w v), the shader's rotateQ.
    const std::array<float, 3> inner {
        q[1]*v[2] - q[2]*v[1] + q[3]*v[0],
        q[2]*v[0] - q[0]*v[2] + q[3]*v[1],
        q[0]*v[1] - q[1]*v[0] + q[3]*v[2]};
    return {v[0] + 2.0f*(q[1]*inner[2] - q[2]*inner[1]),
            v[1] + 2.0f*(q[2]*inner[0] - q[0]*inner[2]),
            v[2] + 2.0f*(q[0]*inner[1] - q[1]*inner[0])};
}

// Camera distance (-camera.z) of a world point, as the vertex shader computes it.
inline float cameraDistance(const std::array<float, 3>& world, const std::array<float, 4>& cameraRotation,
                            const std::array<float, 3>& cameraTranslation) noexcept
{
    const std::array<float, 4> inverse {-cameraRotation[0], -cameraRotation[1], -cameraRotation[2], cameraRotation[3]};
    return -rotate(inverse, {world[0] - cameraTranslation[0], world[1] - cameraTranslation[1],
                             world[2] - cameraTranslation[2]})[2];
}

// Adds the eight corners of an object-space box under a column-major world matrix.
inline void includeBox(Range& range, const Matrix& world, const std::array<float, 3>& low,
                       const std::array<float, 3>& high, const std::array<float, 4>& cameraRotation,
                       const std::array<float, 3>& cameraTranslation) noexcept
{
    for (int corner = 0; corner < 8; ++corner) {
        const std::array<float, 3> p {(corner & 1) ? high[0] : low[0], (corner & 2) ? high[1] : low[1],
                                      (corner & 4) ? high[2] : low[2]};
        const std::array<float, 3> w {world[0]*p[0] + world[4]*p[1] + world[8]*p[2] + world[12],
                                      world[1]*p[0] + world[5]*p[1] + world[9]*p[2] + world[13],
                                      world[2]*p[0] + world[6]*p[1] + world[10]*p[2] + world[14]};
        const float distance = cameraDistance(w, cameraRotation, cameraTranslation);
        if (!std::isfinite(distance)) { range.nearest = std::numeric_limits<float>::lowest();
            range.farthest = std::numeric_limits<float>::max(); continue; }
        range.nearest = std::min(range.nearest, distance);
        range.farthest = std::max(range.farthest, distance);
    }
}

// The clip depth range: the scene's distances padded by a thousandth, inside the
// authored planes. A scene that reaches past either plane keeps that plane, and
// an empty or degenerate range keeps the authored one.
inline std::array<float, 2> fitted(const Range& scene, float authoredNear, float authoredFar) noexcept
{
    if (scene.empty() || !(authoredNear < authoredFar)) return {authoredNear, authoredFar};
    const float pad = 1.0e-3f * std::max({1.0f, std::fabs(scene.nearest), std::fabs(scene.farthest)});
    const float nearest = std::max(authoredNear, scene.nearest - pad);
    const float farthest = std::min(authoredFar, scene.farthest + pad);
    if (!(nearest < farthest)) return {authoredNear, authoredFar};
    return {nearest, farthest};
}

// The perspective vertex shader's window depth in [0, 1] for a camera distance.
inline double perspectiveWindowDepth(double distance, double nearPlane, double farPlane) noexcept
{
    const double clipZ = distance*(farPlane + nearPlane)/(farPlane - nearPlane)
        - 2.0*nearPlane*farPlane/(farPlane - nearPlane);
    return 0.5*(clipZ/distance) + 0.5;
}

// The value a fixed-point depth attachment of the given bits stores.
inline std::uint64_t quantizedDepth(double windowDepth, int bits) noexcept
{
    const double scale = double((std::uint64_t(1) << bits) - 1);
    return std::uint64_t(std::llround(std::clamp(windowDepth, 0.0, 1.0)*scale));
}
} // namespace arbitgpu::fixturedepth
