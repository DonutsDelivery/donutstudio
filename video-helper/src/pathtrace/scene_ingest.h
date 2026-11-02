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
#include "bvh.h"
#include "../../../shared/Visual3DScene.h"

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
};

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
};

Mat4 transformMatrix (const HarmonicMIDI::grid::SceneTransform3D& transform) noexcept;
// Parent chain times local, with the raster's cycle guard (depth < objectCount).
Mat4 objectWorldMatrix (const HarmonicMIDI::grid::Visual3DScene& scene, std::size_t objectIndex) noexcept;

// Replaces out. Skipped geometry is counted in out.stats. A scene whose
// counts exceed its allocated buffers is refused whole, see out.refusal.
void ingestScene (const HarmonicMIDI::grid::Visual3DScene& scene, IngestedScene& out,
                  const IngestOptions& options = {});

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

    // Unit direction. tMin and tMax are the near and far planes measured along
    // this ray, so the traced depth range equals the raster clip range.
    PT_SHARED_FN Ray generateRay (float pixelX, float pixelY) const noexcept
    {
        const float ndcX = 2.0f * pixelX / static_cast<float> (width) - 1.0f;
        const float ndcY = 1.0f - 2.0f * pixelY / static_cast<float> (height);
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
        const float ndcX = view.x / (tanHalfFov * aspect * viewDepth);
        const float ndcY = view.y / (tanHalfFov * viewDepth);
        pixelX = (ndcX + 1.0f) * 0.5f * static_cast<float> (width);
        pixelY = (1.0f - ndcY) * 0.5f * static_cast<float> (height);
        return std::isfinite (pixelX) && std::isfinite (pixelY);
    }
};

// False for a non-finite record, fov outside (0, pi), near <= 0, far <= near,
// a zero quaternion or a zero-sized image. The quaternion is normalized.
bool makePinholeCamera (const HarmonicMIDI::grid::SceneCameraRecord& record,
                        std::uint32_t width, std::uint32_t height, PinholeCamera& camera) noexcept;
// Uses scene.activeCamera. False when it names no camera in the scene.
bool resolveActiveCamera (const HarmonicMIDI::grid::Visual3DScene& scene,
                          std::uint32_t width, std::uint32_t height, PinholeCamera& camera) noexcept;

} // namespace videohelper::pathtrace
