#include "../../shared/GeometryCore.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

// A camera-graded Grid: cells widen with distance from the centre, replay
// rebuilds the same mesh, grade 0 is the old regular grid bit for bit, the
// Ocean lowers and replays on it, and Water still needs a regular grid.
namespace {
using namespace videowire::geometry;

int failures = 0;

void check(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
  }
}

bool samePositions(const std::vector<Vec3>& a, const std::vector<Vec3>& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (a[i].x != b[i].x || a[i].y != b[i].y || a[i].z != b[i].z) return false;
  return true;
}

PortContract meshContract() {
  PortContract contract; contract.carrier = CarrierKind::geometry3D; contract.overflow = OverflowPolicy::reject;
  contract.maxVertices = 4096; contract.maxIndices = 24576; contract.maxAttributes = 8;
  return contract;
}

void grading() {
  std::string error;
  const auto contract = meshContract();
  check(!lowerGrid(70101, 70101, 1, 8, 8, 0.25f, contract, error, -0.1f), "a negative grade is refused");
  check(!lowerGrid(70101, 70101, 1, 8, 8, 0.25f, contract, error, kMaximumGridGrade * 2), "a grade above its bound is refused");

  auto regular = lowerGrid(70101, 70101, 1, 9, 9, 0.25f, contract, error);
  auto flat = lowerGrid(70101, 70101, 1, 9, 9, 0.25f, contract, error, 0.0f);
  check(regular && flat && samePositions(std::get<GeometryData>(regular->data).positions, std::get<GeometryData>(flat->data).positions),
        "grade 0 is the regular grid");
  if (regular) {
    const auto& p = std::get<GeometryData>(regular->data).positions;
    check(p[0].x == (0.0f - 4.0f) * 0.25f && p[8].x == (8.0f - 4.0f) * 0.25f, "the regular grid keeps its old coordinates");
  }

  auto graded = lowerGrid(70102, 70102, 1, 9, 9, 0.25f, contract, error, 0.5f);
  check(bool(graded), ("a graded grid lowers: " + error).c_str());
  if (!graded) return;
  const auto& p = std::get<GeometryData>(graded->data).positions;
  const double centre = p[5].x - p[4].x, outer = p[8].x - p[7].x;
  check(std::abs(centre - 0.25 * 1.5) < 1e-6, "the centre cell is spacing * (1 + grade)");
  check(std::abs(outer - 0.25 * (1 + 0.5 * 7)) < 1e-6, "the outer cell is spacing * (1 + 7 grade)");
  check(p[0].x == -p[8].x && p[0].y == p[0].x, "the grading is symmetric on both axes");

  MaterializedInstancePlan plan;
  const RuntimeFieldEvaluation frame;
  check(validateOperationPlan(*graded, {}, &plan, error, nullptr, &frame), ("the graded grid replays: " + error).c_str());
  const auto replayed = plan.geometries.find(graded->stableId);
  check(replayed != plan.geometries.end() && samePositions(replayed->second.positions, p), "replay rebuilds the same graded mesh");
}

void ocean() {
  std::string error;
  const auto contract = meshContract();
  auto graded = lowerGrid(70201, 70201, 1, 16, 16, 0.25f, contract, error, 0.25f);
  check(bool(graded), "the ocean grid lowers");
  if (!graded) return;
  WaterSurfaceGrid grid;
  check(!validateWaterSurfaceGrid(std::get<GeometryData>(graded->data), grid, error),
        "Water still needs a regular grid");
  check(validateWaterSurfaceGrid(std::get<GeometryData>(graded->data), grid, error, true)
            && std::abs(grid.spacingX - 0.25 * 1.125) < 1e-6,
        "a graded grid admits with its narrowest (centre) cell as spacing");

  OceanSettings o; o.size = 20; o.windSpeed = 9;
  const auto firstField = (StableId{1} << 42) + 70202 * 16;
  auto surface = lowerOceanSurface(*graded, 70202, firstField, o, 1, 0.5, contract, error);
  check(bool(surface), ("the Ocean lowers on a graded grid: " + error).c_str());
  if (!surface) return;
  RuntimeFieldEvaluation runtime; runtime.timelineSeconds = 2;
  MaterializedInstancePlan plan; ValueDescriptor replayed;
  check(validateOperationPlan(*surface, {}, &plan, error, &replayed, &runtime),
        ("the graded Ocean replays: " + error).c_str());
  if (!std::holds_alternative<GeometryData>(replayed.data)) return;
  const auto& rest = std::get<GeometryData>(graded->data).positions;
  const auto& moved = std::get<GeometryData>(replayed.data).positions;
  bool waves = false, finiteMesh = true;
  for (std::size_t i = 0; i < rest.size(); ++i) {
    waves = waves || moved[i].z != rest[i].z;
    finiteMesh = finiteMesh && std::isfinite(moved[i].x) && std::isfinite(moved[i].z);
  }
  check(waves && finiteMesh, "the graded Ocean carries finite waves");
}
} // namespace

int main() {
  grading();
  ocean();
  if (failures != 0) return 1;
  std::puts("geometry graded grid tests passed");
  return 0;
}
