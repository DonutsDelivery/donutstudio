#include "../../shared/GeometryEdgeSelection.h"

#include <cstdio>
#include <initializer_list>
#include <string>
#include <vector>

// Selection-field sharp edges and creases: edge selection from vertices, merging
// with authored entries, and lowering through Shade Mesh and Subdivision Surface.
namespace {
using namespace videowire::geometry;

int failures = 0;

void check(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
  }
}

FieldData selection(std::initializer_list<int> flags) {
  FieldData field;
  for (const auto flag : flags) field.elements.push_back({{static_cast<double>(flag), 0, 0, 0}});
  return field;
}

void edges() {
  PortContract contract; contract.carrier = CarrierKind::geometry3D; contract.overflow = OverflowPolicy::reject;
  contract.maxVertices = 256; contract.maxIndices = 1536; contract.maxAttributes = 8;
  std::string error;
  // A 3 by 2 grid: vertices 0 1 2 on the first row, 3 4 5 on the second.
  auto grid = lowerGrid(30101, 30101, 1, 3, 2, 1.0f, contract, error);
  check(bool(grid), "edge selection grid lowers");
  if (!grid) return;
  const auto& mesh = std::get<GeometryData>(grid->data);
  std::vector<std::array<std::uint32_t, 2>> selected;
  check(selectedTriangleEdges(mesh, selection({1, 1, 1, 0, 0, 0}), selected, error)
            && selected == std::vector<std::array<std::uint32_t, 2>>{{0, 1}, {1, 2}},
        "a selected row gives the edges between its vertices, once each, sorted");
  check(selectedTriangleEdges(mesh, selection({0, 0, 0, 0, 0, 0}), selected, error) && selected.empty(),
        "an empty selection selects no edge");
  check(!selectedTriangleEdges(mesh, selection({1, 1}), selected, error)
            && error.find("2 values for 6 vertices") != std::string::npos,
        "a field of the wrong length names both counts");

  const auto merged = mergeSharpEdges({{2, 1}, {4, 5}}, {{0, 1}, {1, 2}});
  check(merged == std::vector<std::array<std::uint32_t, 2>>{{0, 1}, {1, 2}, {4, 5}},
        "authored and selected sharp edges merge sorted and unique");

  std::vector<SubdivisionCrease> creases{{mesh.vertexIds[2], mesh.vertexIds[1], 0.25f}};
  check(mergeSelectedCreases(mesh, {{0, 1}, {1, 2}}, 0.75f, creases, error) && creases.size() == 2,
        "selected edges become creases keyed by stable vertex ids");
  bool authoredKept = false, selectedWeighted = false;
  for (const auto& crease : creases) {
    check(crease.vertexA < crease.vertexB, "crease ids are ordered");
    const auto edge = std::make_pair(crease.vertexA, crease.vertexB);
    const auto ab = std::minmax(mesh.vertexIds[1], mesh.vertexIds[2]);
    const auto cd = std::minmax(mesh.vertexIds[0], mesh.vertexIds[1]);
    if (edge == std::make_pair(ab.first, ab.second)) authoredKept = crease.weight == 0.25f;
    if (edge == std::make_pair(cd.first, cd.second)) selectedWeighted = crease.weight == 0.75f;
  }
  check(authoredKept && selectedWeighted, "an authored crease keeps its weight; a selected edge takes the crease weight");
  check(!mergeSelectedCreases(mesh, {}, 1.5f, creases, error), "a crease weight above 1 is refused");

  auto subdivided = lowerSubdivisionSurface(*grid, 30102, 1, 1, creases, false, contract, error);
  check(bool(subdivided), ("subdivision lowers with selected creases: " + error).c_str());
  if (subdivided) {
    const auto& record = subdivided->operations.back().subdivision;
    check(record && record->creases.size() == 2, "the subdivision record carries the merged creases");
  }
  auto shaded = lowerShadeEdit(*grid, 30103, 1, 30, merged, error);
  check(bool(shaded), ("shade lowers with selected sharp edges: " + error).c_str());
  if (shaded) check(shaded->operations.back().sharpEdges.size() == 3, "the shade record carries the merged sharp edges");
}
} // namespace

int main() {
  edges();
  if (failures != 0) return 1;
  std::puts("geometry edge selection tests passed");
  return 0;
}
