#include "../../shared/GeometryCore.h"

#include <cmath>
#include <cstdio>
#include <string>

// Render-time adaptive subdivision: the level a cage needs from a view and a
// dicing rate in pixels, the lowered rate, and frame replay that picks the
// level from the view while admission keeps the authored level.
namespace {
using namespace videowire::geometry;

int failures = 0;

void check(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
  }
}

PortContract meshContract() {
  PortContract contract; contract.carrier = CarrierKind::geometry3D; contract.overflow = OverflowPolicy::reject;
  contract.maxVertices = 200000; contract.maxIndices = 1200000; contract.maxAttributes = 8;
  return contract;
}

// A camera on the +Z axis looking down -Z at the cage on the z = 0 plane.
std::array<double, 12> at(double distance) { return {{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -distance}}; }

void levels(const GeometryData& cage) {
  const auto tanHalf = std::tan(0.7853981634 / 2.0);
  // A 3 by 3 grid of 1 m cells: its longest edge is a cell diagonal, about 184 px
  // tall on a 540 px view from 5 m, so a 16 px rate needs four halvings.
  check(adaptiveCageLevel(cage, 16, at(5), tanHalf, 540, 0) == 4, "a near cage dices to four levels");
  check(adaptiveCageLevel(cage, 16, at(10), tanHalf, 540, 0) == 3, "twice as far needs one level less");
  check(adaptiveCageLevel(cage, 32, at(5), tanHalf, 540, 0) == 3, "a coarser rate needs one level less");
  check(adaptiveCageLevel(cage, 16, at(5), tanHalf, 540, 6) == 6, "the authored level is the floor");
  check(adaptiveCageLevel(cage, 16, at(0), tanHalf, 540, 0) > 4, "a camera inside the bounds dices finest");
  check(adaptiveCageLevel(cage, 16, at(5), tanHalf, 0, 2) == 2, "no viewport keeps the authored level");
  // The cage is measured in camera space: an object scaled by 2 seen from 10 m
  // covers the same pixels as the unscaled cage from 5 m.
  check(adaptiveCageLevel(cage, 16, {{2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, -10}}, tanHalf, 540, 0) == 4,
        "object scale counts in the camera-space measure");
}

void replay() {
  std::string error;
  const auto contract = meshContract();
  auto grid = lowerGrid(60101, 60101, 1, 3, 3, 1.0f, contract, error);
  check(bool(grid), "subdivision cage lowers");
  if (!grid) return;
  levels(std::get<GeometryData>(grid->data));

  check(!lowerSubdivisionSurface(*grid, 60102, 1, 1, {}, false, contract, error, -1.0f)
            && error.find("dicing rate") != std::string::npos,
        "a negative dicing rate is refused by name");
  auto plain = lowerSubdivisionSurface(*grid, 60102, 0, 0, {}, false, contract, error);
  check(plain && plain->operations.size() == grid->operations.size(), "no levels and no rate leave the cage alone");
  auto authored = lowerSubdivisionSurface(*grid, 60102, 1, 1, {}, false, contract, error);
  check(authored && !hasAdaptiveSubdivision(*authored) && authored->operations.back().parameters[3] == 0.0f,
        "a zero rate keeps the operation as before");

  auto adaptive = lowerSubdivisionSurface(*grid, 60102, 1, 1, {}, false, contract, error, 16.0f);
  check(bool(adaptive), ("a dicing rate lowers: " + error).c_str());
  auto fine = lowerSubdivisionSurface(*grid, 60103, 4, 4, {}, false, contract, error);
  check(bool(fine), "the four-level reference lowers");
  if (!adaptive || !fine || !authored) return;
  check(hasAdaptiveSubdivision(*adaptive) && adaptive->operations.back().parameters[3] == 16.0f,
        "the lowered operation carries the rate");
  const auto admittedTriangles = std::get<GeometryData>(adaptive->data).indices.size() / 3;
  check(admittedTriangles == std::get<GeometryData>(authored->data).indices.size() / 3,
        "admission builds the authored level");

  RuntimeFieldEvaluation view;
  view.subdivisionLocalToCamera = at(5);
  view.subdivisionTanHalfFov = std::tan(0.7853981634 / 2.0);
  view.subdivisionViewportHeight = 540;
  const auto replayed = [&](const RuntimeFieldEvaluation& runtime) -> std::size_t {
    MaterializedInstancePlan plan;
    if (!validateOperationPlan(*adaptive, {}, &plan, error, nullptr, &runtime)) return 0;
    const auto terminal = plan.geometries.find(adaptive->stableId);
    return terminal == plan.geometries.end() ? 0 : terminal->second.indices.size() / 3;
  };
  check(replayed(view) == std::get<GeometryData>(fine->data).indices.size() / 3,
        ("the view dices the cage to the four-level mesh: " + error).c_str());
  check(replayed(RuntimeFieldEvaluation{}) == admittedTriangles, "without a view the authored level stays");
}
} // namespace

int main() {
  replay();
  if (failures != 0) return 1;
  std::puts("geometry adaptive subdivision tests passed");
  return 0;
}
