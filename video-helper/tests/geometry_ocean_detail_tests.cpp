#include "../../shared/GeometryCore.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

// Ocean wave detail that follows wavelength: the fade weight, per-vertex
// footprints on uniform and graded grids, and the lowered Ocean with a detail
// falloff, which must match the filtered sum at replay and keep the old chain
// when the falloff is 0.
namespace {
using namespace videowire::geometry;

int failures = 0;

void check(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
  }
}

constexpr double kTwoPi = 6.283185307179586;

void weight() {
  const auto k = [](double wavelength) { return kTwoPi / wavelength; };
  check(oceanDetailWeight(k(0.01), 0) == 1, "a zero footprint keeps every wave");
  check(std::abs(oceanDetailWeight(k(4), 1) - 1) < 1e-12, "a wave four footprints long keeps full height");
  check(oceanDetailWeight(k(2), 1) < 1e-12, "a wave two footprints long is gone");
  check(std::abs(oceanDetailWeight(k(3), 1) - 0.5) < 1e-12, "the fade is halfway at three footprints");
  check(oceanDetailWeight(k(2.5), 1) < oceanDetailWeight(k(3.5), 1), "longer waves keep more height");
}

void footprints() {
  // A graded 4 by 3 grid: columns at 0, 1, 3, 7 and rows at 0, 0.5, 2.
  const std::vector<double> xs{0, 1, 3, 7}, ys{0, 0.5, 2};
  std::vector<FieldElement> positions;
  for (const auto y : ys) for (const auto x : xs) positions.push_back({{x, y, 0, 0}});
  std::vector<double> out;
  std::string error;
  check(oceanFootprints(positions, {FieldElement{{0, 0, 0, 0}}}, out, error) && out.size() == 12,
        "a graded grid has a footprint per vertex");
  if (out.size() == 12) {
    check(out[0] == 1, "the dense corner takes its widest neighbouring cell");
    check(out[1] == 2 && out[3] == 4, "the cell widens with the grading");
    check(out[8] == 1.5 && out[11] == 4, "the far row takes its tall cell or wider column");
  }
  std::vector<FieldElement> authored(12, FieldElement{{0, 0, 0, 0}});
  authored[0].components[0] = 9;
  check(oceanFootprints(positions, authored, out, error) && out[0] == 9, "an authored footprint wider than the cell wins");
  authored[0].components[0] = -1;
  check(!oceanFootprints(positions, authored, out, error), "a negative footprint is refused");
  auto scattered = positions;
  std::swap(scattered[1], scattered[6]);
  check(!oceanFootprints(scattered, {FieldElement{{0, 0, 0, 0}}}, out, error)
            && error.find("regular XY grid") != std::string::npos,
        "a scattered mesh is refused by name");
}

void lowering() {
  PortContract contract; contract.carrier = CarrierKind::geometry3D; contract.overflow = OverflowPolicy::reject;
  contract.maxVertices = 4096; contract.maxIndices = 24576; contract.maxAttributes = 8;
  std::string error;
  auto grid = lowerGrid(50101, 50101, 1, 24, 20, 0.25f, contract, error);
  check(bool(grid), "detail grid lowers");
  if (!grid) return;
  // Waves down to 0.1 m on a 0.25 m grid: the shortest ones cannot be carried.
  OceanSettings o; o.smallestWave = 0.1; o.size = 6; o.windSpeed = 9;
  const auto firstField = (StableId{1} << 42) + 50102 * 16;
  auto plain = lowerOceanSurface(*grid, 50102, firstField, o, 1, 0.5, contract, error);
  check(bool(plain), ("an ocean without detail falloff lowers: " + error).c_str());
  if (plain) {
    const auto noise = std::find_if(plain->operations.begin(), plain->operations.end(), [](const auto& op) {
      return op.code == OperationCode::fieldNoise && op.field.mode == kNoiseFieldOceanMode;
    });
    check(noise != plain->operations.end() && noise->field.auxiliaryInputStableId == 0,
          "a zero falloff keeps the ocean without a footprint input");
  }
  const double falloff = 0.2, focusX = 0, focusY = 0;
  auto detailed = lowerOceanSurface(*grid, 50102, firstField, o, 1, 0.5, contract, error, falloff, focusX, focusY);
  check(bool(detailed), ("an ocean with detail falloff lowers: " + error).c_str());
  check(!lowerOceanSurface(*grid, 50102, firstField, o, 1, 0.5, contract, error, 2)
            && error.find("detail falloff") != std::string::npos,
        "a falloff above its bound is refused by name");
  if (!detailed) return;
  const auto noise = std::find_if(detailed->operations.begin(), detailed->operations.end(), [](const auto& op) {
    return op.code == OperationCode::fieldNoise && op.field.mode == kNoiseFieldOceanMode;
  });
  check(noise != detailed->operations.end() && noise->field.auxiliaryInputStableId == firstField + 8,
        "the ocean reads the footprint field");
  if (noise == detailed->operations.end()) return;
  OceanSettings lowered;
  std::vector<WindSpectrumWave> waves;
  check(oceanSettingsFromValues(noise->field.values, lowered, error) && oceanWaves(lowered, waves, error),
        "the lowered settings rebuild the waves");

  RuntimeFieldEvaluation runtime; runtime.timelineSeconds = 1.5;
  MaterializedInstancePlan plan; ValueDescriptor replayed, unfiltered;
  check(validateOperationPlan(*detailed, {}, &plan, error, &replayed, &runtime),
        ("the detailed ocean replays: " + error).c_str());
  MaterializedInstancePlan plainPlan;
  check(plain && validateOperationPlan(*plain, {}, &plainPlan, error, &unfiltered, &runtime), "the plain ocean replays");
  if (!std::holds_alternative<GeometryData>(replayed.data) || !std::holds_alternative<GeometryData>(unfiltered.data)) return;
  const auto& rest = std::get<GeometryData>(grid->data).positions;
  const auto& moved = std::get<GeometryData>(replayed.data).positions;
  const auto& full = std::get<GeometryData>(unfiltered.data).positions;
  double worst = 0, nearChange = 0, farChange = 0;
  for (std::size_t i = 0; i < rest.size(); ++i) {
    const double dx = rest[i].x - focusX, dy = rest[i].y - focusY, dz = rest[i].z;
    const auto distance = std::sqrt(dx * dx + dy * dy + dz * dz);
    const auto footprint = std::max(0.25, falloff * distance);
    const auto d = oceanDisplacement(waves, rest[i].x, rest[i].y, 1.5, footprint);
    worst = std::max(worst, std::abs(moved[i].z - (rest[i].z + d[2])));
    worst = std::max(worst, std::abs(moved[i].x - (rest[i].x + d[0])));
    const auto change = std::abs(static_cast<double>(moved[i].z - full[i].z));
    if (distance < 1) nearChange = std::max(nearChange, change);
    else if (distance > 3) farChange = std::max(farChange, change);
  }
  check(worst < 2e-4, "each vertex moves by the sum filtered at its own footprint");
  check(nearChange > 0, "waves shorter than the grid fade even at the focus");
  check(farChange >= nearChange, "far vertices lose at least as much detail as near ones");
}
} // namespace

int main() {
  weight();
  footprints();
  lowering();
  if (failures != 0) return 1;
  std::puts("geometry ocean detail tests passed");
  return 0;
}
