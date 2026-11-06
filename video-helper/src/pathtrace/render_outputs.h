#pragma once

// Render outputs of Path Trace (Plan 3, Compositing): Depth, Normal, Motion,
// Emission, Mask, Material ID and Object ID next to the Image, in the raster's
// raw pass formats (backend.h rawPassFormatMatches), so the Render node's
// output ports, its Image views, pass composites and data exports read the
// same values from every engine.
//
// The engine writes them at the centre ray's first hit, once per accumulation.
// Motion is the raster's: current minus previous pixel position, positive Y
// upwards, from the scene's previous object and instance transforms and the
// previous camera. A scene without them has zero motion, as the raster has
// without a previous sample.

#include "scene_ingest.h"
#include "vec_math.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace videohelper::pathtrace
{

// The inverse of an affine column-major matrix. A singular one gives identity,
// so a collapsed object reads as not moving.
inline Mat4 affineInverse (const Mat4& t) noexcept
{
    const auto& m = t.m;
    const float det = determinant3x3 (t);
    if (! (std::abs (det) > 0.0f) || ! std::isfinite (det))
        return {};
    const float inv = 1.0f / det;
    Mat4 r;
    // Element (row, column) is m[column * 4 + row].
    const auto at = [&] (int row, int column) { return m[static_cast<std::size_t> (column * 4 + row)]; };
    auto set = [&] (int row, int column, float value) { r.m[static_cast<std::size_t> (column * 4 + row)] = value; };
    set (0, 0, (at (1, 1) * at (2, 2) - at (1, 2) * at (2, 1)) * inv);
    set (0, 1, (at (0, 2) * at (2, 1) - at (0, 1) * at (2, 2)) * inv);
    set (0, 2, (at (0, 1) * at (1, 2) - at (0, 2) * at (1, 1)) * inv);
    set (1, 0, (at (1, 2) * at (2, 0) - at (1, 0) * at (2, 2)) * inv);
    set (1, 1, (at (0, 0) * at (2, 2) - at (0, 2) * at (2, 0)) * inv);
    set (1, 2, (at (0, 2) * at (1, 0) - at (0, 0) * at (1, 2)) * inv);
    set (2, 0, (at (1, 0) * at (2, 1) - at (1, 1) * at (2, 0)) * inv);
    set (2, 1, (at (0, 1) * at (2, 0) - at (0, 0) * at (2, 1)) * inv);
    set (2, 2, (at (0, 0) * at (1, 1) - at (0, 1) * at (1, 0)) * inv);
    for (int row = 0; row < 3; ++row)
        set (row, 3, -(r.m[static_cast<std::size_t> (row)] * at (0, 3) + r.m[static_cast<std::size_t> (4 + row)] * at (1, 3)
                       + r.m[static_cast<std::size_t> (8 + row)] * at (2, 3)));
    return r;
}

inline bool sceneHasMotionHistory (const HarmonicMIDI::grid::Visual3DScene& scene) noexcept
{
    return scene.previousObjectTransformCount > 0 || scene.previousObjectInstanceCount > 0;
}

// The world of object objectIndex one frame earlier. current is the world the
// tracer drew, a runtime override or the authored parent chain. The previous
// chain takes each node's previous transform where the scene has one, and the
// runtime offset current * inverse (authored) carries over unchanged. A chain
// whose previous transforms all equal the current ones returns current itself,
// so a still object has exactly zero motion.
inline Mat4 previousObjectWorld (const HarmonicMIDI::grid::Visual3DScene& scene, std::size_t objectIndex,
                                 const Mat4& current) noexcept
{
    const std::size_t objectCount = std::min (scene.objectCount, scene.objects.size());
    if (objectIndex >= objectCount || scene.previousObjectTransformCount == 0)
        return current;
    const std::size_t previousCount = std::min (scene.previousObjectTransformCount, scene.previousObjectTransforms.size());
    const auto previousOf = [&] (std::size_t slot) {
        return transformMatrix (slot < previousCount ? scene.previousObjectTransforms[slot] : scene.objects[slot].transform);
    };
    Mat4 authored = transformMatrix (scene.objects[objectIndex].transform);
    Mat4 previous = previousOf (objectIndex);
    bool moved = previous.m != authored.m;
    auto parent = scene.objects[objectIndex].parent;
    for (std::size_t depth = 0; parent.isValid() && depth < objectCount; ++depth)
    {
        std::size_t slot = objectCount;
        for (std::size_t i = 0; i < objectCount; ++i)
            if (scene.objects[i].id == parent)
            {
                slot = i;
                break;
            }
        if (slot >= objectCount)
            break;
        const Mat4 node = transformMatrix (scene.objects[slot].transform), nodePrevious = previousOf (slot);
        moved = moved || nodePrevious.m != node.m;
        authored = node * authored;
        previous = nodePrevious * previous;
        parent = scene.objects[slot].parent;
    }
    return moved ? current * affineInverse (authored) * previous : current;
}

// The previous transform of instance `instance` of the instanced object, or its
// current one when the scene has no previous instance there.
inline Mat4 previousInstanceMatrix (const HarmonicMIDI::grid::Visual3DScene& scene, std::size_t instance) noexcept
{
    const std::size_t previousCount = std::min (scene.previousObjectInstanceCount, scene.previousObjectInstances.size());
    if (instance < previousCount)
        return transformMatrix (scene.previousObjectInstances[instance]);
    return instance < std::min (scene.objectInstanceCount, scene.objectInstances.size())
        ? transformMatrix (scene.objectInstances[instance]) : Mat4 {};
}

inline bool sameView (const PinholeCamera& a, const PinholeCamera& b) noexcept
{
    return a.position.x == b.position.x && a.position.y == b.position.y && a.position.z == b.position.z
        && a.rotation.x == b.rotation.x && a.rotation.y == b.rotation.y && a.rotation.z == b.rotation.z
        && a.rotation.w == b.rotation.w && a.tanHalfFov == b.tanHalfFov && a.aspect == b.aspect;
}

// The raster's RG16F motion of a surface point: its pixel position now minus
// where previousWorld * inverse (currentWorld) put it a frame earlier, seen
// through the previous camera, Y up. Zero when neither the worlds nor the
// camera moved, or when either position does not project (behind the eye).
inline std::array<float, 2> pixelMotion (const PinholeCamera& camera, const PinholeCamera& previousCamera, Vec3 world,
                                         const Mat4& currentWorld, const Mat4& previousWorld) noexcept
{
    const bool still = previousWorld.m == currentWorld.m;
    if (still && sameView (camera, previousCamera))
        return { 0.0f, 0.0f };
    const Vec3 previousPoint = still ? world
        : transformPoint (previousWorld, transformPoint (affineInverse (currentWorld), world));
    float x = 0.0f, y = 0.0f, px = 0.0f, py = 0.0f, depth = 0.0f;
    if (! camera.project (world, x, y, depth) || ! previousCamera.project (previousPoint, px, py, depth))
        return { 0.0f, 0.0f };
    // Pixel rows run downwards; the raster's motion runs upwards.
    return { x - px, py - y };
}

inline std::array<float, 2> pixelMotion (const PinholeCamera& camera, Vec3 world, const Mat4& currentWorld,
                                         const Mat4& previousWorld) noexcept
{
    return pixelMotion (camera, camera, world, currentWorld, previousWorld);
}

// The raw pass channels of one pixel, as the raster's fragment writes them.
struct FirstHitOutput
{
    float depth = 1.0f;                    // (viewDepth - near) / (far - near), 1 where nothing was hit
    std::array<float, 3> normal {};        // unit world shading normal, unflipped; zero where nothing was hit
    std::array<float, 3> emission {};      // emitted radiance times the emission gain
    std::uint8_t mask = 0;                 // 255 where the centre ray hit
    std::uint32_t materialId = 0;          // SceneMaterialId::value
    std::uint32_t objectId = 0;            // SceneObjectId::value
    std::array<float, 2> motion {};        // pixels, Y up
};

} // namespace videohelper::pathtrace
