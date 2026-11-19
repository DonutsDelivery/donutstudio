#pragma once

#include "GeometryCore.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace videowire::geometry {

// Half extents of the Solid 3D oriented-box contact proxy for a static body
// mesh. The renderer centers the mesh, divides by its enclosing radius, then
// scales by bodyRadius, so the proxy uses that same bound. Returns false when
// the mesh is empty or too large for the proxy, or when any dimension is flat
// or degenerate. The helper admission and the plugin lowering gate both call
// this function, so they refuse the same meshes.
inline bool particleSolidBoxHalfExtents(const GeometryData &mesh, float bodyRadius,
                                        std::array<float, 3> &half) {
  if (mesh.positions.empty() || mesh.positions.size() > 256 ||
      mesh.indices.empty() || mesh.indices.size() > 384)
    return false;
  auto low = mesh.positions.front(), high = low;
  for (const auto &v : mesh.positions) {
    low.x = std::min(low.x, v.x);
    low.y = std::min(low.y, v.y);
    low.z = std::min(low.z, v.z);
    high.x = std::max(high.x, v.x);
    high.y = std::max(high.y, v.y);
    high.z = std::max(high.z, v.z);
  }
  const std::array<float, 3> center{(low.x + high.x) * 0.5f,
                                    (low.y + high.y) * 0.5f,
                                    (low.z + high.z) * 0.5f};
  float radius = 0;
  for (const auto &v : mesh.positions)
    radius = std::max(radius, std::hypot(std::hypot(v.x - center[0], v.y - center[1]),
                                         v.z - center[2]));
  if (!std::isfinite(radius) || radius <= 1.0e-7f)
    return false;
  half = {bodyRadius * (high.x - low.x) * 0.5f / radius,
          bodyRadius * (high.y - low.y) * 0.5f / radius,
          bodyRadius * (high.z - low.z) * 0.5f / radius};
  return std::all_of(half.begin(), half.end(),
                     [](float v) { return std::isfinite(v) && v > 1.0e-6f; });
}

}  // namespace videowire::geometry
