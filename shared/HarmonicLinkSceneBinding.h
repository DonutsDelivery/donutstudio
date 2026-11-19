#pragma once

#include "GeometryCoreTransport.h"
#include "Visual3DScene.h"
#include "VisualHarmonicLinkGeometryContract.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace videowire::geometry {
// One Harmonic Link Geometry node in a retained scene. The compiled scene holds
// a placeholder object for its transform and material. Every frame replaces the
// mesh, whose size follows the score, so this is not a timeline binding.
struct HarmonicLinkSceneBinding final {
  HarmonicMIDI::grid::SceneObjectId object;
  visualharmonicgeometry::Mapping mapping;
  // True when the node feeds Render 3D directly and has no authored camera to
  // frame it. A composed scene keeps the score coordinates.
  bool fitToView = false;
};

// Fits the links in the view of a camera 10 units back with a 45 degree vertical
// field, centred. The musical mapping stays in the link positions' proportions.
inline void fitHarmonicLinksToView(std::vector<Vec3>& positions, int width, int height) {
  if (positions.empty() || width <= 0 || height <= 0) return;
  auto minimum = positions.front();
  auto maximum = minimum;
  for (const auto& position : positions) {
    minimum.x = std::min(minimum.x, position.x);
    minimum.y = std::min(minimum.y, position.y);
    minimum.z = std::min(minimum.z, position.z);
    maximum.x = std::max(maximum.x, position.x);
    maximum.y = std::max(maximum.y, position.y);
    maximum.z = std::max(maximum.z, position.z);
  }
  const auto visibleHeight = 2.0f * 10.0f * std::tan(0.7853981634f * 0.5f);
  const auto visibleWidth = visibleHeight * static_cast<float>(width) / static_cast<float>(height);
  // The depth extent counts too, so the links stay in front of the camera. Any
  // spread fills the same share of the view: tiny links grow and huge ones shrink.
  const auto largest = std::max({(maximum.x - minimum.x) / (visibleWidth * 0.65f),
                                 (maximum.y - minimum.y) / (visibleHeight * 0.65f),
                                 (maximum.z - minimum.z) / (10.0f * 0.65f)});
  const auto displayScale = largest > 0.0f ? 1.0f / largest : 1.0f;
  const Vec3 center {(minimum.x + maximum.x) * 0.5f, (minimum.y + maximum.y) * 0.5f,
                     (minimum.z + maximum.z) * 0.5f};
  for (auto& position : positions) {
    position.x = (position.x - center.x) * displayScale;
    position.y = (position.y - center.y) * displayScale;
    position.z = (position.z - center.z) * displayScale;
  }
}

// Points the object at this frame's mesh. The mesh goes at the end of the scene
// buffers, which grow to fit, and the placeholder's records stay unreferenced.
inline bool replaceHarmonicLinkSceneMesh(const HarmonicLinkSceneBinding& binding, const GeometryData& mesh,
    HarmonicMIDI::grid::Visual3DScene& scene, std::string& error) {
  using namespace HarmonicMIDI::grid;
  if (mesh.positions.empty() || mesh.indices.empty()) {
    error = "Harmonic Link Geometry requires a mesh with vertices and indices";
    return false;
  }
  for (const auto index : mesh.indices)
    if (index >= mesh.positions.size()) {
      error = "Harmonic Link Geometry has an out-of-range mesh index";
      return false;
    }
  constexpr auto kRecordRange = static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max());
  if (scene.vertexCount > kRecordRange || mesh.positions.size() > kRecordRange - scene.vertexCount
      || scene.indexCount > kRecordRange || mesh.indices.size() > kRecordRange - scene.indexCount) {
    error = "Harmonic Link Geometry does not fit the 32-bit scene object range";
    return false;
  }
  Visual3DSceneGrowth added;
  added.vertices = mesh.positions.size();
  added.indices = mesh.indices.size();
  if (!growVisual3DScene(scene, added, error)) return false;
  SceneObjectRecord* object = nullptr;
  for (std::size_t index = 0; index < scene.objectCount; ++index)
    if (scene.objects[index].id == binding.object) object = &scene.objects[index];
  if (object == nullptr) {
    error = "Harmonic Link Geometry lost its scene object";
    return false;
  }
  const auto vertexBase = scene.vertexCount;
  const auto indexBase = scene.indexCount;
  for (std::size_t index = 0; index < mesh.positions.size(); ++index) {
    const auto& position = mesh.positions[index];
    scene.vertices[vertexBase + index] = {};
    scene.vertices[vertexBase + index].position = {position.x, position.y, position.z};
  }
  for (std::size_t index = 0; index < mesh.indices.size(); ++index)
    scene.indices[indexBase + index] = mesh.indices[index];
  for (std::size_t index = 0; index + 2 < mesh.indices.size(); index += 3) {
    const auto a = mesh.indices[index], b = mesh.indices[index + 1], c = mesh.indices[index + 2];
    const auto& p = mesh.positions[a];
    const auto& q = mesh.positions[b];
    const auto& r = mesh.positions[c];
    const Vec3 normal {(q.y-p.y)*(r.z-p.z)-(q.z-p.z)*(r.y-p.y),
                       (q.z-p.z)*(r.x-p.x)-(q.x-p.x)*(r.z-p.z),
                       (q.x-p.x)*(r.y-p.y)-(q.y-p.y)*(r.x-p.x)};
    for (const auto vertex : {a, b, c}) {
      auto& n = scene.vertices[vertexBase + vertex].normal;
      n.x += normal.x; n.y += normal.y; n.z += normal.z;
    }
  }
  for (std::size_t index = 0; index < mesh.positions.size(); ++index) {
    auto& n = scene.vertices[vertexBase + index].normal;
    const auto length = std::sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
    if (length > 0) { n.x /= length; n.y /= length; n.z /= length; }
    else n = {0, 0, 1};
  }
  scene.vertexCount += mesh.positions.size();
  scene.indexCount += mesh.indices.size();
  object->firstVertex = static_cast<std::uint32_t>(vertexBase);
  object->vertexCount = static_cast<std::uint32_t>(mesh.positions.size());
  object->firstIndex = static_cast<std::uint32_t>(indexBase);
  object->indexCount = static_cast<std::uint32_t>(mesh.indices.size());
  return true;
}
} // namespace videowire::geometry
