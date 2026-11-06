#include "../../shared/GeometryCore.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

// The Ocean node: settings packing, spectrum shape, loop period, choppy motion,
// foam, the grid fast path, and lowering plus timeline replay of a grid.
namespace {
using namespace videowire::geometry;

int failures = 0;

void check(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
  }
}

double energy(const std::vector<WindSpectrumWave>& waves) {
  double total = 0;
  for (const auto& w : waves) total += w.amplitude * w.amplitude / 2;
  return total;
}

void settingsPacking() {
  OceanSettings o;
  o.windSpeed = 17.5; o.headingDegrees = -42; o.scale = 2.5; o.smallestWave = 0.3; o.size = 120;
  o.depth = 8; o.loopPeriod = 12; o.alignment = 3.47; o.damping = 0.83;
  o.spectrum = OceanSpectrumShape::tma; o.resolution = 32; o.seed = kOceanMaximumSeed;
  std::string error;
  OceanSettings back;
  check(oceanSettingsFromValues(oceanFieldValues(o), back, error), "packed ocean settings decode");
  check(back.windSpeed == o.windSpeed && back.headingDegrees == o.headingDegrees && back.scale == o.scale
            && back.smallestWave == o.smallestWave && back.size == o.size && back.depth == o.depth
            && back.loopPeriod == o.loopPeriod && back.spectrum == o.spectrum && back.resolution == o.resolution
            && back.seed == o.seed,
        "every exact ocean setting survives the field values");
  check(std::abs(back.alignment - 3.47) < 1e-12 && std::abs(back.damping - 0.83) < 1e-12,
        "alignment and damping travel as hundredths");
  auto bad = o; bad.damping = 1.5;
  check(!validOceanSettings(bad, error) && error.find("damping") != std::string::npos,
        "a damping refusal names the control");
  bad = o; bad.size = o.smallestWave;
  check(!validOceanSettings(bad, error) && error.find("size") != std::string::npos,
        "the size must exceed the smallest wave");
  auto values = oceanFieldValues(o); values[7] += 0.5;
  check(!oceanSettingsFromValues(values, back, error), "a fractional packed value is refused");
}

void spectrumShape() {
  std::string error;
  OceanSettings o; o.smallestWave = 0.5; o.size = 200;
  std::vector<WindSpectrumWave> phillips, jonswap, aligned, damped, deep, shallow;
  check(oceanWaves(o, phillips, error), "Phillips ocean synthesizes");
  check(phillips.size() == std::size_t{oceanRings(4)} * oceanDirections(4), "resolution sets the wave count");
  // 10 m/s Phillips over the full band holds about 0.2 square metres of variance.
  check(energy(phillips) > 0.05 && energy(phillips) < 0.5, "Phillips variance follows the absolute spectrum");
  auto j = o; j.spectrum = OceanSpectrumShape::jonswap;
  check(oceanWaves(j, jonswap, error), "JONSWAP ocean synthesizes");
  check(std::abs(energy(jonswap) - energy(phillips)) > 1e-6, "JONSWAP is a different spectrum");
  auto scaled = o; scaled.scale = 2;
  std::vector<WindSpectrumWave> twice;
  check(oceanWaves(scaled, twice, error) && std::abs(energy(twice) - 4 * energy(phillips)) < 1e-9 * energy(twice),
        "height scale multiplies every amplitude");
  auto a = o; a.alignment = 6; a.damping = 1;
  check(oceanWaves(a, aligned, error), "aligned ocean synthesizes");
  double along = 0, against = 0;
  for (const auto& w : aligned) (w.kx > 0 ? along : against) += w.amplitude * w.amplitude;
  check(against == 0 && along > 0, "full damping removes every wave running against the wind");
  auto d = o; d.damping = 0;
  check(oceanWaves(d, damped, error), "undamped ocean synthesizes");
  double backward = 0;
  for (const auto& w : damped) if (w.kx < 0) backward += w.amplitude * w.amplitude;
  check(backward > 0, "zero damping keeps waves running against the wind");
  auto s = o; s.depth = 2;
  check(oceanWaves(o, deep, error) && oceanWaves(s, shallow, error), "deep and shallow oceans synthesize");
  // Rings run from the smallest wave up, so the back of the set is the longest wave.
  check(shallow.back().omega < deep.back().omega, "shallow water slows the longest waves");
  check(std::abs(shallow.front().omega - deep.front().omega) < 1e-3 * deep.front().omega,
        "short waves do not feel a two metre depth");
  auto seeded = o; seeded.seed = 7;
  std::vector<WindSpectrumWave> other;
  check(oceanWaves(seeded, other, error) && other.front().phase != phillips.front().phase, "seed changes the phases");
  // Smallest Wave bounds the band: no synthesized wavelength is shorter.
  auto coarse = o; coarse.smallestWave = 4;
  std::vector<WindSpectrumWave> band;
  double shortest = 1e300;
  if (oceanWaves(coarse, band, error))
    for (const auto& wave : band) shortest = std::min(shortest, 2 * 3.14159265358979 / std::hypot(wave.kx, wave.ky));
  check(!band.empty() && shortest >= 4 * (1 - 1e-9), "smallest wave bounds the shortest wavelength");
}

void loopAndMotion() {
  std::string error;
  OceanSettings o; o.smallestWave = 0.5; o.size = 60; o.loopPeriod = 7.5;
  std::vector<WindSpectrumWave> waves;
  check(oceanWaves(o, waves, error), "looping ocean synthesizes");
  double drift = 0, change = 0;
  for (double x : {0.0, 3.1, 17.0}) {
    const auto start = oceanDisplacement(waves, x, 2.0, 1.25);
    const auto later = oceanDisplacement(waves, x, 2.0, 1.25 + o.loopPeriod);
    const auto half = oceanDisplacement(waves, x, 2.0, 1.25 + o.loopPeriod / 2);
    for (int i = 0; i < 3; ++i) {
      drift = std::max(drift, std::abs(later[i] - start[i]));
      change = std::max(change, std::abs(half[i] - start[i]));
    }
  }
  check(drift < 1e-9, "every wave repeats exactly after the loop period");
  check(change > 1e-3, "the ocean moves inside the loop");
  check(std::abs(oceanSeconds(o, 0.5) - 3.75) < 1e-12, "a looping ocean reads the loop phase");

  // One wave along +X: points move toward its crest at phase pi/2.
  const std::vector<WindSpectrumWave> single{{1, 0, 0.5, 0, 0}};
  const auto before = oceanDisplacement(single, 1.4, 0, 0);
  const auto after = oceanDisplacement(single, 1.8, 0, 0);
  check(before[0] > 0 && after[0] < 0, "choppy motion pulls points toward the crest");
  check(std::abs(oceanDisplacement(single, 3.14159265358979 / 2, 0, 0)[2] - 0.5) < 1e-12,
        "height is the summed sine");
}

void gridAndFoam() {
  std::string error;
  OceanSettings o; o.smallestWave = 0.4; o.size = 16;
  std::vector<WindSpectrumWave> waves;
  check(oceanWaves(o, waves, error), "grid ocean synthesizes");
  std::vector<FieldElement> positions;
  for (int row = 0; row < 9; ++row)
    for (int column = 0; column < 11; ++column) positions.push_back({{column * 0.25 - 1, row * 0.25 - 1, 0, 0}});
  std::vector<std::array<double, 3>> grid;
  check(oceanGridDisplacements(waves, o, positions, {FieldElement{{2.5, 0, 0, 0}}}, grid), "grid fast path accepts a grid");
  double worst = 0;
  for (std::size_t i = 0; i < positions.size(); ++i) {
    const auto direct = oceanDisplacement(waves, positions[i].components[0], positions[i].components[1], 2.5);
    for (int j = 0; j < 3; ++j) worst = std::max(worst, std::abs(direct[j] - grid[i][j]));
  }
  check(worst < 1e-9, "grid fast path matches the per-vertex formula");
  auto scattered = positions; std::swap(scattered[3], scattered[40]);
  check(!oceanGridDisplacements(waves, o, scattered, {FieldElement{{2.5, 0, 0, 0}}}, grid) && grid.empty(),
        "a scattered layout leaves the formula to the caller");

  std::vector<FieldElement> flat(positions.size()), folded(positions.size());
  for (std::size_t i = 0; i < positions.size(); ++i)
    folded[i].components[0] = -0.4 * std::sin(positions[i].components[0] * 4);
  std::vector<double> foam;
  check(oceanFoamValues(positions, flat, 0.5, foam, error)
            && std::all_of(foam.begin(), foam.end(), [](double f) { return f == 0; }),
        "a flat sea has no foam");
  check(oceanFoamValues(positions, folded, 0.5, foam, error)
            && *std::max_element(foam.begin(), foam.end()) > 0.5
            && *std::min_element(foam.begin(), foam.end()) == 0,
        "foam appears where the surface compresses and stays clear where it stretches");
  check(!oceanFoamValues(positions, flat, 3, foam, error) && error.find("foam coverage") != std::string::npos,
        "a foam coverage refusal names the control");
}

void lowering() {
  PortContract contract; contract.carrier = CarrierKind::geometry3D; contract.overflow = OverflowPolicy::reject;
  contract.maxVertices = 4096; contract.maxIndices = 24576; contract.maxAttributes = 8;
  std::string error;
  auto grid = lowerGrid(20101, 20101, 1, 24, 20, 0.25f, contract, error);
  check(bool(grid), "ocean rest grid lowers");
  if (!grid) return;
  OceanSettings o; o.smallestWave = 0; o.size = 0; o.loopPeriod = 6; o.windSpeed = 8;
  const auto firstField = (StableId{1} << 42) + 20102 * 8;
  auto ocean = lowerOceanSurface(*grid, 20102, firstField, o, 1.5, 0.8, contract, error);
  check(bool(ocean), ("ocean lowers a regular grid: " + error).c_str());
  if (!ocean) return;
  check(bool(admitValue(*ocean, withAttributeContract(contract, *ocean), error)), "ocean admits");
  const auto* foam = findNamedAttribute(ocean->attributes, "foam");
  check(foam && foam->descriptor.domain == Domain::vertex && foam->descriptor.valueType == ValueType::floatValue
            && foam->elements.size() == std::get<GeometryData>(grid->data).positions.size(),
        "ocean stores a vertex Float foam attribute");
  const auto noise = std::find_if(ocean->operations.begin(), ocean->operations.end(), [](const auto& op) {
    return op.code == OperationCode::fieldNoise && op.field.mode == kNoiseFieldOceanMode;
  });
  OceanSettings lowered;
  check(noise != ocean->operations.end() && oceanSettingsFromValues(noise->field.values, lowered, error)
            && std::abs(lowered.smallestWave - 1.0) < 1e-6 && std::abs(lowered.size - 5.75) < 1e-6,
        "zero smallest wave and size follow the grid: four cells to its full extent");

  const auto replayAt = [&](double seconds, ValueDescriptor& out) {
    RuntimeFieldEvaluation runtime; runtime.timelineSeconds = seconds;
    MaterializedInstancePlan plan;
    return validateOperationPlan(*ocean, {}, &plan, error, &out, &runtime);
  };
  ValueDescriptor first, second, looped;
  check(replayAt(1.0, first) && replayAt(2.3, second) && replayAt(7.0, looped), "ocean replays at timeline times");
  if (!std::holds_alternative<GeometryData>(first.data) || !std::holds_alternative<GeometryData>(second.data)
      || !std::holds_alternative<GeometryData>(looped.data)) return;
  const auto& a = std::get<GeometryData>(first.data).positions;
  const auto& b = std::get<GeometryData>(second.data).positions;
  const auto& c = std::get<GeometryData>(looped.data).positions;
  const auto& rest = std::get<GeometryData>(grid->data).positions;
  double moved = 0, horizontal = 0, loopDrift = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    moved = std::max(moved, static_cast<double>(std::abs(a[i].z - b[i].z)));
    horizontal = std::max(horizontal, static_cast<double>(std::abs(a[i].x - rest[i].x)));
    loopDrift = std::max(loopDrift, static_cast<double>(std::abs(a[i].z - c[i].z) + std::abs(a[i].x - c[i].x)));
  }
  check(moved > 1e-4, "the surface moves between two timeline times");
  check(horizontal > 1e-5, "choppiness moves vertices horizontally");
  check(loopDrift < 1e-4, "replay one loop period later matches");
  check(findNamedAttribute(first.attributes, "foam") != nullptr, "replay carries the foam attribute");

  check(!lowerOceanSurface(*grid, 20102, firstField, o, 11, 0.8, contract, error)
            && error.find("choppiness") != std::string::npos,
        "a choppiness refusal names the control");
  auto points = *grid;
  std::swap(std::get<GeometryData>(points.data).positions[1], std::get<GeometryData>(points.data).positions[30]);
  check(!lowerOceanSurface(points, 20102, firstField, o, 1, 0.8, contract, error), "ocean refuses a scattered mesh");
}
} // namespace

int main() {
  settingsPacking();
  spectrumShape();
  loopAndMotion();
  gridAndFoam();
  lowering();
  if (failures != 0) return 1;
  std::puts("geometry ocean tests passed");
  return 0;
}
