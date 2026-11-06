#include "../../shared/GeometryFieldLowering.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {
using namespace videowire::geometry;

int failures = 0;

void check(bool condition, const std::string& message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    ++failures;
  }
}

// Vec3 has no operator==, so compare the components here.
bool samePositions(const std::vector<Vec3>& a, const std::vector<Vec3>& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (a[i].x != b[i].x || a[i].y != b[i].y || a[i].z != b[i].z) return false;
  return true;
}

constexpr double tau = 6.283185307179586476925286766559;

bool same(const std::vector<WindSpectrumWave>& a, const std::vector<WindSpectrumWave>& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (a[i].kx != b[i].kx || a[i].ky != b[i].ky || a[i].amplitude != b[i].amplitude
        || a[i].omega != b[i].omega || a[i].phase != b[i].phase) return false;
  return true;
}

// Share of the sea's energy carried by waves more than 45 degrees off the wind.
double crossShare(const std::vector<WindSpectrumWave>& waves, double heading) {
  double cross = 0, total = 0;
  for (const auto& wave : waves) {
    const auto k = std::hypot(wave.kx, wave.ky);
    const auto along = (wave.kx * std::cos(heading) + wave.ky * std::sin(heading)) / k;
    const auto energy = wave.amplitude * wave.amplitude;
    total += energy;
    if (along < std::cos(tau / 8)) cross += energy;
  }
  return total > 0 ? cross / total : 0;
}

void zeroShapeKeepsTheOriginalSea() {
  std::string error;
  std::vector<WindSpectrumWave> original, zero;
  check(windSpectrumWaves(10, 30, 0.2, 4, 63, original, error), "the original sea synthesizes: " + error);
  check(windSpectrumWaves(10, 30, 0.2, 4, 63, zero, error, {0, 0, 0}), "a zero shape synthesizes: " + error);
  check(same(original, zero), "a zero shape is bit-identical to a sea saved before the shape controls");
}

void alignmentNarrowsTheSpread() {
  std::string error;
  std::vector<WindSpectrumWave> open, aligned;
  check(windSpectrumWaves(10, 0, 0.2, 4, 63, open, error) && windSpectrumWaves(10, 0, 0.2, 4, 63, aligned, error, {4, 0, 0}),
      "aligned and open seas synthesize: " + error);
  check(crossShare(aligned, 0) < crossShare(open, 0) / 2,
      "wave alignment moves energy from crossing waves onto the wind: " + std::to_string(crossShare(open, 0))
          + " -> " + std::to_string(crossShare(aligned, 0)));
  double openEnergy = 0, alignedEnergy = 0;
  for (const auto& wave : open) openEnergy += wave.amplitude * wave.amplitude;
  for (const auto& wave : aligned) alignedEnergy += wave.amplitude * wave.amplitude;
  check(std::abs(openEnergy - alignedEnergy) <= 1e-12 * openEnergy, "alignment keeps the authored root mean square height");
}

void depthSlowsLongWaves() {
  std::string error;
  std::vector<WindSpectrumWave> deep, shallow;
  check(windSpectrumWaves(10, 0, 0.2, 4, 63, deep, error) && windSpectrumWaves(10, 0, 0.2, 4, 63, shallow, error, {0, 2, 0}),
      "deep and shallow seas synthesize: " + error);
  bool dispersion = deep.size() == shallow.size();
  for (std::size_t i = 0; dispersion && i < deep.size(); ++i) {
    const auto k = std::hypot(shallow[i].kx, shallow[i].ky);
    const auto expected = std::sqrt(kWindSpectrumGravity * k * std::tanh(k * 2));
    dispersion = std::abs(shallow[i].omega - expected) <= 1e-12 * expected && shallow[i].omega < deep[i].omega
        && shallow[i].kx == deep[i].kx && shallow[i].amplitude == deep[i].amplitude;
  }
  check(dispersion, "two metres of water slow every wave by sqrt(g k tanh(k h)) and keep its shape");
}

void loopPeriodRepeatsSeamlessly() {
  std::string error;
  std::vector<WindSpectrumWave> looped;
  constexpr double period = 12;
  check(windSpectrumWaves(10, 20, 0.3, 4, 63, looped, error, {0, 0, period}), "a looped sea synthesizes: " + error);
  bool whole = !looped.empty();
  for (const auto& wave : looped) {
    const auto cycles = wave.omega * period / tau;
    whole = whole && cycles >= 1 && std::abs(cycles - std::round(cycles)) <= 1e-9;
  }
  check(whole, "every looped wave completes a whole number of cycles in the period");
  double drift = 0;
  for (const double x : {-7.0, 0.0, 3.5}) for (const double y : {-2.0, 5.0}) for (const double t : {0.0, 4.25})
    drift = std::max(drift, std::abs(windSpectrumHeight(looped, x, y, t) - windSpectrumHeight(looped, x, y, t + period)));
  check(drift <= 1e-9, "heights one period apart match: " + std::to_string(drift));
}

void invalidShapesAreRefused() {
  std::string error;
  std::vector<WindSpectrumWave> waves;
  check(!windSpectrumWaves(10, 0, 0.2, 4, 63, waves, error, {-1, 0, 0}) && error.find("alignment") != std::string::npos,
      "negative alignment is refused by name");
  check(!windSpectrumWaves(10, 0, 0.2, 4, 63, waves, error, {0, -1, 0}) && error.find("depth") != std::string::npos,
      "negative depth is refused by name");
  check(!windSpectrumWaves(10, 0, 0.2, 4, 63, waves, error, {0, 0, NAN}) && error.find("loop") != std::string::npos,
      "a non-finite loop period is refused by name");
}

// The lowering stores the shape in the wind field's last three settings, so
// replay, plan comparison and saved projects all carry it.
void loweringCarriesTheShape() {
  std::string error;
  PortContract contract;
  contract.carrier = CarrierKind::geometry3D; contract.maxVertices = 4096; contract.maxIndices = 24576;
  const auto grid = lowerGrid(10180, 10180, 1, 17, 17, 8, contract, error);
  const auto plain = grid ? lowerWindSpectrumSurfaceWave(*grid, 10181, 10182, 0.2, 10, 0, contract, error) : std::nullopt;
  const auto shaped = grid ? lowerWindSpectrumSurfaceWave(*grid, 10181, 10182, 0.2, 10, 0, contract, error, {2, 30, 20})
                           : std::nullopt;
  check(plain && shaped, "plain and shaped wind seas lower: " + error);
  if (!plain || !shaped) return;
  const auto wind = [](const ValueDescriptor& value) -> const OperationRecord* {
    for (const auto& op : value.operations)
      if (op.code == OperationCode::fieldNoise && op.field.mode == kNoiseFieldWindSpectrumMode) return &op;
    return nullptr;
  };
  const auto* plainWind = wind(*plain);
  const auto* shapedWind = wind(*shaped);
  check(plainWind && plainWind->field.values[5] == 0 && plainWind->field.values[6] == 0 && plainWind->field.values[7] == 0,
      "a sea without shape controls lowers exactly as before");
  check(shapedWind && shapedWind->field.values[5] == 2 && shapedWind->field.values[6] == 30
          && shapedWind->field.values[7] == 20,
      "alignment, depth and loop period reach the wind field");
  check(!lowerWindSpectrumSurfaceWave(*grid, 10181, 10182, 0.2, 10, 0, contract, error, {0, -3, 0}),
      "the lowering refuses a negative depth");
  check(!samePositions(std::get<GeometryData>(plain->data).positions, std::get<GeometryData>(shaped->data).positions),
      "the shaped sea's admitted surface differs from the open deep sea");
}
}  // namespace

int main() {
  zeroShapeKeepsTheOriginalSea();
  alignmentNarrowsTheSpread();
  depthSlowsLongWaves();
  loopPeriodRepeatsSeamlessly();
  invalidShapesAreRefused();
  loweringCarriesTheShape();
  if (failures != 0) return 1;
  std::puts("wind spectrum shape tests passed");
  return 0;
}
