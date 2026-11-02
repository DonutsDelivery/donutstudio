#include "../../shared/GeometryCore.h"
#include "../../shared/GeometryCoreTransport.h"
#include "../src/pathtrace/bvh.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <variant>
#include <vector>

// Catmull-Clark vertex counts, crease behaviour, a mesh far past the scene record's
// 12,288 indices, and overflow that reports counts and bytes.
namespace {
using namespace videowire::geometry;
namespace pt = videohelper::pathtrace;

int failures = 0;

void check(bool condition, const std::string &message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    ++failures;
  }
}

bool nearly(double actual, double expected) {
  return std::fabs(actual - expected) <= 1e-5;
}

PortContract meshContract() {
  PortContract contract;
  contract.carrier = CarrierKind::geometry3D;
  contract.overflow = OverflowPolicy::reject;
  contract.maxVertices = kMaximumVertices;
  contract.maxIndices = kMaximumIndices;
  contract.maxAttributes = 32;
  return contract;
}

GeometryData unitCube() {
  std::string error;
  const auto cube = lowerMeshGenerator(11, 11, 1, OperationCode::cube, {1, 1, 1, 0}, meshContract(), error);
  check(cube.has_value(), "the unit cube lowers: " + error);
  return cube ? std::get<GeometryData>(cube->data) : GeometryData{};
}

StableId idAt(const GeometryData &mesh, float x, float y, float z) {
  for (std::size_t i = 0; i < mesh.positions.size(); ++i) {
    const auto &p = mesh.positions[i];
    if (p.x == x && p.y == y && p.z == z) return mesh.vertexIds[i];
  }
  check(false, "the cube has the corner the test asks for");
  return 0;
}

bool hasPoint(const GeometryData &mesh, double x, double y, double z) {
  for (const auto &p : mesh.positions)
    if (nearly(p.x, x) && nearly(p.y, y) && nearly(p.z, z)) return true;
  return false;
}

GeometryData subdivide(const GeometryData &source, const std::vector<SubdivisionCrease> &creases, std::uint32_t level) {
  GeometryData result;
  std::vector<AttributeData> attributes;
  std::string error;
  check(materializeSubdivisionSurface(source, {}, creases, level, result, attributes, error),
        "level " + std::to_string(level) + " subdivides: " + error);
  return result;
}

void cubeLevelsHaveCatmullClarkCounts() {
  const auto cube = unitCube();
  const auto first = subdivide(cube, {}, 1);
  check(first.positions.size() == 26 && first.indices.size() == 144,
        "level 1 of a cube has 8+12+6 vertices and 24 quads");
  const auto second = subdivide(cube, {}, 2);
  check(second.positions.size() == 98 && second.indices.size() == 576,
        "level 2 of a cube has 98 vertices and 96 quads");
  check(!hasPoint(first, 0.5, 0.5, 0.5), "an uncreased cube corner moves inward");
  const auto copy = subdivide(cube, {}, 0);
  check(copy.positions.size() == cube.positions.size() && copy.indices == cube.indices, "level 0 copies the mesh");
}

void creasesKeepMarkedEdges() {
  const auto cube = unitCube();
  const auto corner = idAt(cube, 0.5f, 0.5f, 0.5f);
  const auto crease = [&](StableId other) {
    return SubdivisionCrease{std::min(corner, other), std::max(corner, other), 1.0f};
  };
  std::vector<SubdivisionCrease> creases{crease(idAt(cube, -0.5f, 0.5f, 0.5f)), crease(idAt(cube, 0.5f, -0.5f, 0.5f)),
                                         crease(idAt(cube, 0.5f, 0.5f, -0.5f))};
  std::sort(creases.begin(), creases.end(), [](const SubdivisionCrease &a, const SubdivisionCrease &b) {
    return std::make_pair(a.vertexA, a.vertexB) < std::make_pair(b.vertexA, b.vertexB);
  });
  const auto kept = subdivide(cube, creases, 2);
  check(hasPoint(kept, 0.5, 0.5, 0.5), "three fully creased edges keep their corner exactly at level 2");
  check(kept.positions.size() == 98, "creases do not change the vertex count");
  const auto single = subdivide(cube, {crease(idAt(cube, 0.5f, 0.5f, -0.5f))}, 1);
  check(hasPoint(single, 0.5, 0.5, 0.0), "a fully creased edge keeps its midpoint on the original edge");
  check(!hasPoint(subdivide(cube, {}, 1), 0.5, 0.5, 0.0), "the same edge midpoint moves without a crease");
}

void uvInterpolatesAcrossAQuad() {
  GeometryData quad;
  quad.positions = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
  quad.vertexIds = {1, 2, 3, 4};
  quad.indices = {0, 1, 2, 0, 2, 3};
  quad.uv = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
  const auto result = subdivide(quad, {}, 1);
  check(result.positions.size() == 9 && result.indices.size() == 24, "one quad becomes 9 vertices and 4 quads");
  bool center = false;
  for (std::size_t i = 0; i < result.positions.size(); ++i)
    if (nearly(result.positions[i].x, 0.5) && nearly(result.positions[i].y, 0.5))
      center = nearly(result.uv[i].x, 0.5) && nearly(result.uv[i].y, 0.5);
  check(center, "the face point carries the face's mean uv");
  check(result.uv.size() == 9, "matching uv values do not split a vertex");
}

void largeMeshSubdividesAndTraces() {
  // 300 x 300 quads: 540,000 indices going in, 8,640,000 coming out.
  constexpr std::uint32_t cells = 300, side = cells + 1;
  GeometryData grid;
  grid.positions.reserve(side * side);
  grid.vertexIds.reserve(side * side);
  for (std::uint32_t y = 0; y < side; ++y)
    for (std::uint32_t x = 0; x < side; ++x) {
      grid.positions.push_back({static_cast<float>(x), static_cast<float>(y),
                                0.02f * std::sin(static_cast<float>(x) * 0.1f)});
      grid.vertexIds.push_back(y * side + x + 1);
    }
  grid.indices.reserve(cells * cells * 6);
  for (std::uint32_t y = 0; y < cells; ++y)
    for (std::uint32_t x = 0; x < cells; ++x) {
      const auto a = y * side + x, b = a + 1, c = a + side, d = c + 1;
      for (const auto index : {a, b, c, b, d, c}) grid.indices.push_back(index);
    }
  const auto result = subdivide(grid, {}, 2);
  check(result.positions.size() == 1442401, "300 x 300 quads reach 1,442,401 vertices at level 2, got "
        + std::to_string(result.positions.size()));
  check(result.indices.size() == 8640000, "300 x 300 quads reach 8,640,000 indices at level 2, got "
        + std::to_string(result.indices.size()));
  std::vector<pt::Vec3> soup;
  soup.reserve(result.indices.size());
  for (const auto index : result.indices) {
    const auto &p = result.positions[index];
    soup.push_back({p.x, p.y, p.z});
  }
  pt::Bvh bvh;
  bvh.build(soup);
  std::size_t hits = 0;
  constexpr std::size_t samples = 64;
  for (std::size_t i = 0; i < samples; ++i) {
    pt::Ray ray;
    const auto at = 2.0f + static_cast<float>(i) * 4.5f;
    ray.origin = {at, at, 10.0f};
    ray.direction = {0.0f, 0.0f, -1.0f};
    pt::Hit hit;
    if (bvh.intersect(ray, hit) && std::fabs(hit.t - 10.0f) < 0.1f) ++hits;
  }
  check(hits == samples, "every primary ray hits the subdivided large mesh, got " + std::to_string(hits));
}

void loweredPlanCarriesLevelsAndRoundTrips() {
  std::string error;
  const auto grid = lowerGrid(21, 21, 1, 65, 65, 1, meshContract(), error);
  check(grid.has_value(), "the grid lowers: " + error);
  if (!grid) return;
  const auto inputIndices = std::get<GeometryData>(grid->data).indices.size();
  check(inputIndices > 12288, "the lowered grid is past the scene record's 12,288 indices");
  const auto absent = lowerSubdivisionSurface(*grid, 22, 0, 0, {}, false, meshContract(), error);
  const auto bytesOf = [&](const ValueDescriptor &value) {
    const auto admitted = admitValue(value, meshContract(), ResourceLimits{}, {}, error);
    return admitted ? encodeRuntimeValue(*admitted) : std::vector<std::uint8_t>{};
  };
  check(absent && !bytesOf(*absent).empty() && bytesOf(*absent) == bytesOf(*grid),
        "absent levels lower to the input's exact bytes");
  const auto viewport = lowerSubdivisionSurface(*grid, 22, 1, 2, {}, false, meshContract(), error);
  check(viewport.has_value(), "viewport level lowers: " + error);
  if (!viewport) return;
  check(std::get<GeometryData>(viewport->data).indices.size() == inputIndices * 4, "preview lowers level 1");
  auto admitted = admitValue(*viewport, meshContract(), ResourceLimits{}, {}, error);
  check(admitted.has_value(), "the subdivided value admits: " + error);
  if (!admitted) return;
  const auto bytes = encodeRuntimeValue(*admitted);
  check(bytes[4] == kSubdivisionTransportVersion && bytes[5] == 0, "subdivision bytes use the subdivision version");
  check(decodeRuntimeValue(bytes.data(), bytes.size(), meshContract(), ResourceLimits{}, {}, error).has_value(),
        "the subdivided value round-trips: " + error);
  auto selected = *viewport;
  check(selectSubdivisionSurfaceUse(selected, true, {}, error), "export selection replays: " + error);
  check(std::get<GeometryData>(selected.data).indices.size() == inputIndices * 16, "export selects level 2");
  check(selectSubdivisionSurfaceUse(selected, false, {}, error)
            && std::get<GeometryData>(selected.data).indices.size() == inputIndices * 4,
        "preview selection returns to level 1");
}

void invalidCreasesAreTypedErrors() {
  std::string error;
  const auto cube = lowerMeshGenerator(31, 31, 1, OperationCode::cube, {1, 1, 1, 0}, meshContract(), error);
  check(cube.has_value(), "the cube lowers");
  if (!cube) return;
  check(!lowerSubdivisionSurface(*cube, 32, 1, 1, {{1, 2, 2.0f}}, false, meshContract(), error),
        "a crease weight above 1 is a typed error");
  check(!lowerSubdivisionSurface(*cube, 32, 1, 1, {{1, 2, 0.5f}, {2, 1, 0.5f}}, false, meshContract(), error),
        "a crease edge whose ids are not ascending is a typed error");
}

void overflowReportsCountsAndBytes() {
  GeometryData result;
  std::vector<AttributeData> attributes;
  std::string error;
  check(!materializeSubdivisionSurface(unitCube(), {}, {}, 1000, result, attributes, error),
        "level 1000 of a cube overflows integer counts");
  check(error.find("vertices") != std::string::npos && error.find("corners") != std::string::npos
            && error.find("bytes") != std::string::npos,
        "the overflow names counts and bytes: " + error);
}

void adaptiveLevelTracksScreenSize() {
  // A 1 unit edge at distance 1 with tan(fov/2)=1 on a 1080 line view spans 540 px.
  check(adaptiveSubdivisionLevel(1, 1, 1, 1080, 8, 0) == 7, "540 px halves seven times to reach 8 px");
  check(adaptiveSubdivisionLevel(1, 16, 1, 1080, 8, 0) == 3, "sixteen times farther needs four fewer levels");
  check(adaptiveSubdivisionLevel(1, 1000, 1, 1080, 8, 2) == 2, "the authored level is the minimum");
  check(adaptiveSubdivisionLevel(0, 1, 1, 1080, 8, 1) == 1, "a degenerate view keeps the authored level");
}
} // namespace

int main() {
  cubeLevelsHaveCatmullClarkCounts();
  creasesKeepMarkedEdges();
  uvInterpolatesAcrossAQuad();
  largeMeshSubdividesAndTraces();
  loweredPlanCarriesLevelsAndRoundTrips();
  invalidCreasesAreTypedErrors();
  overflowReportsCountsAndBytes();
  adaptiveLevelTracksScreenSize();
  if (failures != 0) {
    std::fprintf(stderr, "%d subdivision surface checks failed\n", failures);
    return 1;
  }
  std::puts("Subdivision Surface tests passed");
  return 0;
}
