#pragma once

// Lowers a Visual3DScene snapshot into the flat world-space triangle list the
// path tracer's BVH consumes, and builds a pinhole camera that frames the
// image exactly like the native raster backends.
//
// Conventions shared with the raster path (backend_opengl.cpp):
// - An object's world matrix is parent-chain * local, where local applies
//   scale, then rotation, then translation.
// - The scene's instanced object draws once per objectInstances entry with
//   matrix objectWorld * instanceLocal, and never at its bare world matrix.
// - A mirrored world matrix (negative determinant) keeps the authored front
//   face. Ingest swaps the second and third vertex of such triangles so that
//   counter-clockwise world winding is always the front face.
// - The camera uses its translation and rotation only (no parent, no scale)
//   and looks down local -Z with +Y up.

#include "devices/device_annotations.h"
#include "vec_math.h"
#if ! defined(PATHTRACE_DEVICE_METAL)
#include "bvh.h"
#endif
#if ! defined(PATHTRACE_DEVICE_METAL)
#include "../../../shared/Visual3DScene.h"
#endif

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace videohelper::pathtrace
{

constexpr std::uint32_t kNoMaterial = 0xffffffffu;
constexpr std::uint32_t kNoInstance = 0xffffffffu;

struct TriangleAttributes
{
    std::uint32_t objectIndex = 0;              // into Visual3DScene::objects
    std::uint32_t objectId = 0;                 // SceneObjectId::value
    std::uint32_t materialIndex = kNoMaterial;  // into Visual3DScene::materials
    std::uint32_t materialId = 0;               // SceneMaterialId::value, 0 when unset
    std::uint32_t instanceIndex = kNoInstance;  // into objectInstances, or kNoInstance
    std::uint32_t primitiveIndex = 0;           // triangle number within the object's index range
    bool mirrored = false;                      // world determinant < 0; vertices 1 and 2 swapped
    bool hasShadingNormals = false;             // false: shadingNormals repeat geometricNormal
    bool authoredShading = false;               // Shade Mesh normals: curved shadow origins, no side rejection
    std::uint8_t visibility = 0x3fu;            // the object's HarmonicMIDI::grid::sceneRayVisibility bits
    Vec3 geometricNormal {};                    // unit, front face by counter-clockwise winding
    std::array<Vec3, 3> shadingNormals {};      // unit, world space, in stored vertex order
    std::array<Vec2, 3> uvs {};
    // World tangent xyz (unit) and bitangent sign w, already corrected for mirroring.
    std::array<std::array<float, 4>, 3> tangents {};
    // World dP/du at each corner, smoothed across the triangles that share the
    // vertex and kept at its length, so it interpolates linearly over a triangle.
    // A polar cap's circle tangent is linear in position, so the interpolated
    // value is that tangent everywhere. Zero where the vertex has no single u
    // direction (a cap's hub) or the mesh has no uv.
    std::array<Vec3, 3> grooveTangents {};
    // The vertex colour and roughness lane at each corner, which Raster also
    // reads: colour multiplies the base colour, and roughness (value, weight)
    // replaces the material's by its weight. See vertexShaded.
    std::array<std::array<float, 4>, 3> colors {{ { 1.0f, 1.0f, 1.0f, 1.0f }, { 1.0f, 1.0f, 1.0f, 1.0f },
                                                  { 1.0f, 1.0f, 1.0f, 1.0f } }};
    std::array<std::array<float, 2>, 3> roughness {};
};

// Host only: the Metal kernel (device_metal.metal) leaves this out.
#if ! defined(PATHTRACE_DEVICE_METAL)
struct IngestStats
{
    std::uint32_t objectsWithGeometry = 0;
    std::uint32_t instancesExpanded = 0;
    std::uint32_t invalidObjectRanges = 0;  // vertex or index range outside the scene arrays
    std::uint32_t invalidIndices = 0;       // triangles naming a vertex outside the object range
    std::uint32_t nonFiniteTriangles = 0;
    std::uint32_t degenerateTriangles = 0;  // zero world-space area
};

struct IngestedScene
{
    // Three world-space vertices per triangle, in the order Bvh::build expects.
    std::vector<Vec3> positions;
    // Motion blur: the same vertices at shutter open and shutter close. Both
    // are empty when nothing moves over the shutter, or the same size as
    // positions. A vertex at ray time t lies at motionLerp (open, close, t).
    std::vector<Vec3> motionOpen;
    std::vector<Vec3> motionClose;
    std::vector<TriangleAttributes> triangles;
    Bounds bounds;
    IngestStats stats;
    // Empty when ingested. Otherwise the scene counts more records than its
    // buffers hold, nothing was ingested, and this names the buffer and both
    // numbers.
    std::string refusal;

    std::size_t triangleCount() const noexcept { return triangles.size(); }
    void clear() noexcept;
};

struct IngestOptions
{
    // Optional per-object world matrices, used as-is in place of the authored
    // parent chain. They let a caller apply the runtime object transforms the
    // raster path applies. Ignored unless objectWorldCount == scene.objectCount.
    const Mat4* objectWorlds = nullptr;
    std::size_t objectWorldCount = 0;

    // Motion blur over the shutter [shutterOpen, shutterClose], in frames from
    // the current frame (negative is earlier). Each vertex moves on the line
    // from where it was one frame earlier to where it is now, extended past
    // the current frame, so linear motion through the frame is exact. The
    // earlier frame comes from:
    // - previousObjectWorlds, one per object like objectWorlds (else the
    //   current world),
    // - previousInstances, one per objectInstances entry (else the current one),
    // - previousPositions, the scene's vertex positions, a prefix of
    //   scene.vertices (vertices past it keep their current position).
    // A closed shutter (close <= open) ingests no motion.
    const Mat4* previousObjectWorlds = nullptr;
    std::size_t previousObjectWorldCount = 0;
    const HarmonicMIDI::grid::SceneTransform3D* previousInstances = nullptr;
    std::size_t previousInstanceCount = 0;
    const HarmonicMIDI::grid::SceneVec3* previousPositions = nullptr;
    std::size_t previousPositionCount = 0;
    float shutterOpen = 0.0f;
    float shutterClose = 0.0f;
};

Mat4 transformMatrix (const HarmonicMIDI::grid::SceneTransform3D& transform) noexcept;
// Parent chain times local, with the raster's cycle guard (depth < objectCount).
Mat4 objectWorldMatrix (const HarmonicMIDI::grid::Visual3DScene& scene, std::size_t objectIndex) noexcept;

// Replaces out. Skipped geometry is counted in out.stats. A scene whose
// counts exceed its allocated buffers is refused whole, see out.refusal.
void ingestScene (const HarmonicMIDI::grid::Visual3DScene& scene, IngestedScene& out,
                  const IngestOptions& options = {});
#endif

// Where a point is at `frame` frames from now, given where it was one frame
// earlier and where it is now: on the line through both, extended both ways.
PT_SHARED_FN Vec3 pointAtFrame (Vec3 previous, Vec3 current, float frame) noexcept
{
    return previous + (current - previous) * (1.0f + frame);
}

// Rotation between a and b at fraction t along the shorter arc: a normalized
// lerp, which is what the shutter needs for the small turn of one frame.
PT_SHARED_FN Quat nlerpRotation (Quat a, Quat b, float t) noexcept
{
    const float sign = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w < 0.0f ? -1.0f : 1.0f;
    Quat q { a.x + (sign * b.x - a.x) * t, a.y + (sign * b.y - a.y) * t,
             a.z + (sign * b.z - a.z) * t, a.w + (sign * b.w - a.w) * t };
    const float n = std::sqrt (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    if (! (n > 0.0f))
        return a;
    return { q.x / n, q.y / n, q.z / n, q.w / n };
}

// A uniform point on the unit aperture from u in [0,1)^2: the disc for fewer
// than 3 blades, otherwise the regular polygon inscribed in it with a corner at
// angle rotation. u.x picks one of the polygon's triangles about the centre.
PT_SHARED_FN Vec2 sampleAperture (Vec2 u, std::uint32_t blades, float rotation) noexcept
{
    constexpr float kTwoPi = 6.28318530718f;
    if (blades < 3)
    {
        const float r = std::sqrt (u.x);
        const float phi = kTwoPi * u.y;
        return { r * std::cos (phi), r * std::sin (phi) };
    }
    const float n = static_cast<float> (blades);
    const float scaled = u.x * n;
    const float wedge = std::fmin (std::floor (scaled), n - 1.0f);
    // Uniform in the triangle (centre, corner k, corner k + 1).
    const float s = std::sqrt (std::fmin (scaled - wedge, 1.0f));
    const float a = s * (1.0f - u.y);
    const float b = s * u.y;
    const float angle0 = rotation + wedge * (kTwoPi / n);
    const float angle1 = angle0 + kTwoPi / n;
    return { a * std::cos (angle0) + b * std::cos (angle1), a * std::sin (angle0) + b * std::sin (angle1) };
}

// Pinhole camera matching the raster projection. Pixel coordinates are
// continuous with (0, 0) at the top-left image corner; pixel (i, j) covers
// [i, i + 1) x [j, j + 1), so its centre is (i + 0.5, j + 0.5).
struct PinholeCamera
{
    Vec3 position {};
    Quat rotation {};
    float tanHalfFov = 0.57735027f;
    float aspect = 1.0f;
    float nearPlane = 0.1f;
    float farPlane = 1000.0f;
    std::uint32_t width = 1;
    std::uint32_t height = 1;
    // Above zero the camera is orthographic: half the view height in world units.
    // Rays then start on the view plane and all run along the view axis. An orthographic
    // camera has no lens: its record keeps fStop 0.
    float orthographicHalfHeight = 0.0f;
    // Thin lens: apertureRadius 0 is a pinhole. The aperture is apertureRadius
    // high and apertureRatio times that wide, a disc or a polygon of
    // apertureBlades, and the plane at view depth focusDistance is sharp.
    float apertureRadius = 0.0f;
    float apertureRatio = 1.0f;
    float focusDistance = 1.0f;
    std::uint32_t apertureBlades = 0;
    float bladeRotation = 0.0f;
    // Motion blur: when moving, position and rotation above are the camera at
    // shutter open and these at shutter close. A ray at time t uses at (t).
    bool moving = false;
    Vec3 closePosition {};
    Quat closeRotation {};

    PT_SHARED_FN bool hasLens() const noexcept { return apertureRadius > 0.0f; }
    // The camera at ray time t in [0, 1] over the shutter, as a still camera.
    PT_SHARED_FN PinholeCamera at (float t) const noexcept
    {
        PinholeCamera still = *this;
        if (! moving)
            return still;
        still.position = motionLerp (position, closePosition, t);
        still.rotation = nlerpRotation (rotation, closeRotation, t);
        still.moving = false;
        return still;
    }
    // A point on the lens in camera space (z = 0), from u in [0,1)^2.
    PT_SHARED_FN Vec3 lensPoint (Vec2 u) const noexcept
    {
        if (! hasLens())
            return {};
        const Vec2 unit = sampleAperture (u, apertureBlades, bladeRotation);
        return { unit.x * apertureRadius * apertureRatio, unit.y * apertureRadius, 0.0f };
    }
    // The ray through pixel (pixelX, pixelY) from the lens point lens gives.
    // Every lens point of a pixel meets at its point on the focus plane, and
    // tMin and tMax still bound view depth to the near and far planes.
    PT_SHARED_FN Ray generateRay (float pixelX, float pixelY, Vec2 lens) const noexcept
    {
        Ray ray = generateRay (pixelX, pixelY);
        if (! hasLens())
            return ray;
        const float ndcX = 2.0f * pixelX / static_cast<float> (width) - 1.0f;
        const float ndcY = 1.0f - 2.0f * pixelY / static_cast<float> (height);
        const Vec3 offset = lensPoint (lens);
        const Vec3 local = Vec3 { ndcX * tanHalfFov * aspect, ndcY * tanHalfFov, -1.0f } * focusDistance - offset;
        const float localLength = length (local);
        ray.origin = position + rotate (rotation, offset);
        ray.direction = rotate (rotation, local) * (1.0f / localLength);
        ray.tMin = nearPlane * localLength / focusDistance;
        ray.tMax = farPlane * localLength / focusDistance;
        return ray;
    }
    // The lens counterpart of project: the pixel whose rays from lens point
    // offset (camera space) pass through world. That is the pinhole pixel of
    // where the line from the lens point through world crosses the focus plane.
    PT_SHARED_FN bool projectThroughLens (Vec3 world, Vec3 offset, float& pixelX, float& pixelY,
                                          float& viewDepth) const noexcept
    {
        const Vec3 view = rotate (conjugate (rotation), world - position);
        viewDepth = -view.z;
        if (! (viewDepth > 0.0f))
            return false;
        const Vec3 focus = offset + (view - offset) * (focusDistance / viewDepth);
        const float ndcX = focus.x / (tanHalfFov * aspect * focusDistance);
        const float ndcY = focus.y / (tanHalfFov * focusDistance);
        pixelX = (ndcX + 1.0f) * 0.5f * static_cast<float> (width);
        pixelY = (1.0f - ndcY) * 0.5f * static_cast<float> (height);
        return std::isfinite (pixelX) && std::isfinite (pixelY);
    }

    // Unit direction. tMin and tMax are the near and far planes measured along
    // this ray, so the traced depth range equals the raster clip range.
    PT_SHARED_FN Ray generateRay (float pixelX, float pixelY) const noexcept
    {
        const float ndcX = 2.0f * pixelX / static_cast<float> (width) - 1.0f;
        const float ndcY = 1.0f - 2.0f * pixelY / static_cast<float> (height);
        if (orthographicHalfHeight > 0.0f)
        {
            Ray ray;
            ray.origin = position + rotate (rotation, Vec3 { ndcX * orthographicHalfHeight * aspect,
                                                             ndcY * orthographicHalfHeight, 0.0f });
            ray.direction = rotate (rotation, Vec3 { 0.0f, 0.0f, -1.0f });
            ray.tMin = nearPlane;
            ray.tMax = farPlane;
            return ray;
        }
        const Vec3 local { ndcX * tanHalfFov * aspect, ndcY * tanHalfFov, -1.0f };
        const float localLength = length (local);
        Ray ray;
        ray.origin = position;
        ray.direction = rotate (rotation, local) * (1.0f / localLength);
        ray.tMin = nearPlane * localLength;
        ray.tMax = farPlane * localLength;
        return ray;
    }
    // The raster vertex transform: false for points at or behind the eye.
    PT_SHARED_FN bool project (Vec3 world, float& pixelX, float& pixelY, float& viewDepth) const noexcept
    {
        const Vec3 view = rotate (conjugate (rotation), world - position);
        viewDepth = -view.z;
        if (! (viewDepth > 0.0f))
            return false;
        const float scale = orthographicHalfHeight > 0.0f ? orthographicHalfHeight : tanHalfFov * viewDepth;
        const float ndcX = view.x / (aspect * scale);
        const float ndcY = view.y / scale;
        pixelX = (ndcX + 1.0f) * 0.5f * static_cast<float> (width);
        pixelY = (1.0f - ndcY) * 0.5f * static_cast<float> (height);
        return std::isfinite (pixelX) && std::isfinite (pixelY);
    }
};

#if ! defined(PATHTRACE_DEVICE_METAL)
// False for a non-finite record, fov outside (0, pi), near <= 0, far <= near,
// a zero quaternion or a zero-sized image. The quaternion is normalized.
bool makePinholeCamera (const HarmonicMIDI::grid::SceneCameraRecord& record,
                        std::uint32_t width, std::uint32_t height, PinholeCamera& camera) noexcept;
// Moves camera (made from the current record) over the shutter [open, close]
// in frames, from previous, the record one frame earlier: position on the line
// through both, rotation by normalized lerp. Lens and projection stay the
// current record's. A closed shutter, an equal or an invalid previous record
// leaves the camera still.
void setCameraShutter (PinholeCamera& camera, const HarmonicMIDI::grid::SceneCameraRecord& previous,
                       float open, float close) noexcept;
// Uses scene.activeCamera. False when it names no camera in the scene.
bool resolveActiveCamera (const HarmonicMIDI::grid::Visual3DScene& scene,
                          std::uint32_t width, std::uint32_t height, PinholeCamera& camera) noexcept;
#endif

} // namespace videohelper::pathtrace
