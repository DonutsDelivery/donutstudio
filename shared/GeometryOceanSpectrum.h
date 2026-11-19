#pragma once

// Included by GeometryFieldOperations.h after the wind spectrum. The Ocean node
// sums a wind-driven wave spectrum and moves each vertex along all three axes:
// heights from the spectrum, and choppy horizontal motion toward the crests.
namespace videowire::geometry {
// kNoiseFieldOceanMode evaluates oceanDisplacement with choppiness 1. Its
// timeline input is seconds, or the loop phase 0 to 1 when the ocean loops.

enum class OceanSpectrumShape : std::uint8_t { phillips = 0, jonswap = 1, tma = 2 };

inline constexpr std::uint32_t kOceanMaximumResolution = 32;
inline constexpr std::uint32_t kOceanMaximumSeed = 1048575;
inline constexpr double kOceanMaximumAlignment = 10;
inline constexpr double kOceanMaximumDepth = 10000;
inline constexpr double kOceanMaximumSize = 100000;
inline constexpr double kOceanMaximumScale = 1000;
inline constexpr double kOceanMaximumLoopPeriod = 86400;
inline constexpr double kOceanMaximumChoppiness = 10;
inline constexpr double kOceanMaximumFoamCoverage = 2;
// Alignment and damping travel as hundredths inside one packed field value.
inline constexpr double kOceanAlignmentSteps = 1001;
inline constexpr double kOceanDampingSteps = 101;

struct OceanSettings final {
  double windSpeed = 10, headingDegrees = 0, scale = 1;
  // Wavelength band in metres. The spectrum has no waves outside it.
  double smallestWave = 0.5, size = 50;
  // Zero is deep water. A depth slows long waves and, with TMA, flattens them.
  double depth = 0;
  // Zero runs forever. Otherwise every wave completes a whole number of cycles.
  double loopPeriod = 0;
  // Directional spread is |cos|^(2 alignment). Damping removes that fraction of
  // the waves running against the wind.
  double alignment = 1, damping = 0.5;
  OceanSpectrumShape spectrum = OceanSpectrumShape::phillips;
  std::uint32_t resolution = 4, seed = 0;
};

inline std::uint32_t oceanRings(std::uint32_t resolution) noexcept { return 4 * resolution; }
inline std::uint32_t oceanDirections(std::uint32_t resolution) noexcept { return 2 * resolution + 4; }

inline bool validOceanSettings(const OceanSettings& o, std::string& error) {
  const auto refuse = [&](const char* control, double value, double low, double high) {
    error = std::string("Ocean refused: ") + control + " " + std::to_string(value) + " is outside "
        + std::to_string(low) + " to " + std::to_string(high);
    return false;
  };
  if (!validWindSpectrumSettings(o.windSpeed, o.headingDegrees, error)) return false;
  if (!(o.scale >= 0 && o.scale <= kOceanMaximumScale)) return refuse("height scale", o.scale, 0, kOceanMaximumScale);
  if (!(o.smallestWave > 0 && o.smallestWave <= kOceanMaximumSize))
    return refuse("smallest wave", o.smallestWave, 0, kOceanMaximumSize);
  if (!(o.size > o.smallestWave && o.size <= kOceanMaximumSize)) {
    error = "Ocean refused: size " + std::to_string(o.size) + " m must exceed the smallest wave "
        + std::to_string(o.smallestWave) + " m and stay at most " + std::to_string(kOceanMaximumSize) + " m";
    return false;
  }
  if (!(o.depth >= 0 && o.depth <= kOceanMaximumDepth)) return refuse("depth", o.depth, 0, kOceanMaximumDepth);
  if (!(o.loopPeriod == 0 || (o.loopPeriod >= 0.001 && o.loopPeriod <= kOceanMaximumLoopPeriod)))
    return refuse("loop period", o.loopPeriod, 0.001, kOceanMaximumLoopPeriod);
  if (!(o.alignment >= 0 && o.alignment <= kOceanMaximumAlignment))
    return refuse("wave alignment", o.alignment, 0, kOceanMaximumAlignment);
  if (!(o.damping >= 0 && o.damping <= 1)) return refuse("damping", o.damping, 0, 1);
  if (o.spectrum > OceanSpectrumShape::tma) { error = "Ocean refused: unknown spectrum"; return false; }
  if (o.resolution < 1 || o.resolution > kOceanMaximumResolution)
    return refuse("resolution", o.resolution, 1, kOceanMaximumResolution);
  if (o.seed > kOceanMaximumSeed) return refuse("seed", o.seed, 0, kOceanMaximumSeed);
  error.clear();
  return true;
}

// Field operations carry eight doubles. The integer settings and the two
// hundredths-quantised spreads share the last one; every packed value stays
// below 2^53, so the round trip is exact.
inline std::array<double, 8> oceanFieldValues(const OceanSettings& o) noexcept {
  const auto alignment = std::round(o.alignment * 100);
  const auto damping = std::round(o.damping * 100);
  const auto packed = static_cast<double>(o.spectrum) + 3.0 * (o.resolution - 1)
      + 96.0 * (alignment + kOceanAlignmentSteps * (damping + kOceanDampingSteps * o.seed));
  return {o.windSpeed, o.headingDegrees, o.scale, o.smallestWave, o.size, o.depth, o.loopPeriod, packed};
}

inline bool oceanSettingsFromValues(const std::array<double, 8>& p, OceanSettings& o, std::string& error) {
  const auto packed = p[7];
  if (!(packed >= 0 && packed <= 9007199254740991.0) || std::floor(packed) != packed) {
    error = "Ocean field settings are not a packed integer"; return false;
  }
  auto rest = static_cast<std::uint64_t>(packed);
  o = {};
  o.windSpeed = p[0]; o.headingDegrees = p[1]; o.scale = p[2]; o.smallestWave = p[3];
  o.size = p[4]; o.depth = p[5]; o.loopPeriod = p[6];
  o.spectrum = static_cast<OceanSpectrumShape>(rest % 3); rest /= 3;
  o.resolution = static_cast<std::uint32_t>(rest % 32) + 1; rest /= 32;
  o.alignment = static_cast<double>(rest % 1001) / 100; rest /= 1001;
  o.damping = static_cast<double>(rest % 101) / 100; rest /= 101;
  if (rest > kOceanMaximumSeed) { error = "Ocean field seed exceeds its bound"; return false; }
  o.seed = static_cast<std::uint32_t>(rest);
  return validOceanSettings(o, error);
}

// Angular frequency for wavenumber k at depth (zero is deep water).
inline double oceanAngularFrequency(double k, double depth) noexcept {
  const auto shallow = depth > 0 ? std::tanh(k * depth) : 1.0;
  return std::sqrt(kWindSpectrumGravity * k * shallow);
}

// Variance density per unit wavenumber, before direction, in square metres per (rad/m).
//   Phillips: alpha/2 k^-3 exp(-1/(k L)^2), L = U^2/g.
//   JONSWAP: alpha g^2 w^-5 exp(-5/4 (wp/w)^4) gamma^r, wp = 0.877 g/U, gamma = 3.3,
//            converted to wavenumber through dw/dk.
//   TMA: JONSWAP times the Kitaigorodskii depth factor.
inline double oceanWavenumberSpectrum(const OceanSettings& o, double k) noexcept {
  constexpr double alpha = 0.0081;
  const auto g = kWindSpectrumGravity;
  if (o.spectrum == OceanSpectrumShape::phillips) {
    const auto L = o.windSpeed * o.windSpeed / g;
    return 0.5 * alpha / (k * k * k) * std::exp(-1.0 / ((k * L) * (k * L)));
  }
  const auto w = oceanAngularFrequency(k, o.depth);
  const auto kh = k * o.depth;
  const auto slope = o.depth > 0
      ? g * (std::tanh(kh) + kh / (std::cosh(kh) * std::cosh(kh))) / (2 * w)
      : g / (2 * w);
  const auto peak = 0.877 * g / o.windSpeed;
  const auto sigma = w <= peak ? 0.07 : 0.09;
  const auto r = std::exp(-(w - peak) * (w - peak) / (2 * sigma * sigma * peak * peak));
  auto s = alpha * g * g / std::pow(w, 5) * std::exp(-1.25 * std::pow(peak / w, 4)) * std::pow(3.3, r);
  if (o.spectrum == OceanSpectrumShape::tma && o.depth > 0) {
    const auto wh = w * std::sqrt(o.depth / g);
    s *= wh <= 1 ? 0.5 * wh * wh : wh < 2 ? 1 - 0.5 * (2 - wh) * (2 - wh) : 1.0;
  }
  return s * slope;
}

// Log-spaced rings between the smallest wave and the size, each split into
// directions around the full circle. A component's amplitude is
// sqrt(2 F(k) D(angle) dk dangle) times the height scale, so the surface
// variance matches the spectrum over the band. Phases and direction jitter
// follow the seed; replay at a given time is exact.
inline bool oceanWaves(const OceanSettings& o, std::vector<WindSpectrumWave>& waves, std::string& error) {
  constexpr double tau = 6.283185307179586476925286766559;
  constexpr double pi = tau / 2;
  if (!validOceanSettings(o, error)) return false;
  const auto rings = oceanRings(o.resolution), directions = oceanDirections(o.resolution);
  const auto heading = o.headingDegrees * tau / 360;
  const auto logStep = std::log(o.size / o.smallestWave) / rings;
  const auto angleStep = tau / directions;
  // Normalises |cos|^(2a) over the full circle: 2 sqrt(pi) Gamma(a + 1/2) / Gamma(a + 1).
  const auto spread = 2 * std::sqrt(pi) * std::tgamma(o.alignment + 0.5) / std::tgamma(o.alignment + 1);
  const std::uint64_t jitterSeed = 0x6f6365616e000000ull + 2ull * o.seed;
  waves.clear();
  waves.reserve(static_cast<std::size_t>(rings) * directions);
  for (std::uint32_t ring = 0; ring < rings; ++ring) {
    const auto wavelength = o.smallestWave * std::exp(logStep * (ring + 0.5));
    const auto k = tau / wavelength;
    const auto variance = oceanWavenumberSpectrum(o, k) * k * logStep;
    auto omega = oceanAngularFrequency(k, o.depth);
    if (o.loopPeriod > 0)
      omega = tau / o.loopPeriod * std::max(1.0, std::round(omega * o.loopPeriod / tau));
    for (std::uint32_t direction = 0; direction < directions; ++direction) {
      const std::uint64_t id = static_cast<std::uint64_t>(ring) * directions + direction;
      const auto offset = (direction + 0.5 + 0.8 * (stableRandom(id, jitterSeed) - 0.5)) * angleStep - pi;
      const auto along = std::cos(offset);
      auto weight = std::pow(std::abs(along), 2 * o.alignment) / spread;
      if (along < 0) weight *= 1 - o.damping;
      const auto amplitude = o.scale * std::sqrt(std::max(0.0, 2 * variance * weight * angleStep));
      const auto angle = heading + offset;
      waves.push_back({k * std::cos(angle), k * std::sin(angle), amplitude, omega,
                       tau * stableRandom(id, jitterSeed + 1)});
    }
  }
  if (!std::all_of(waves.begin(), waves.end(), [](const auto& w) { return finite(w.amplitude) && finite(w.omega); })) {
    error = "Ocean spectrum produced a non-finite wave; lower the height scale or raise the smallest wave";
    return false;
  }
  error.clear();
  return true;
}

// The timeline input is seconds, or the loop phase 0 to 1 when the ocean loops,
// which keeps the phase exact however long the timeline runs.
inline double oceanSeconds(const OceanSettings& o, double timeline) noexcept {
  return o.loopPeriod > 0 ? timeline * o.loopPeriod : timeline;
}

// Wave detail follows the sample footprint in metres. A wave shorter than two
// footprints cannot be sampled and shimmers, so its amplitude fades smoothly from
// full at four footprints to zero at two. A footprint of 0 keeps every wave.
inline double oceanDetailWeight(double k, double footprint) noexcept {
  if (!(footprint > 0)) return 1;
  const auto t = std::clamp((6.283185307179586 / (k * footprint) - 2) / 2, 0.0, 1.0);
  return t * t * (3 - 2 * t);
}

// Height is sum a sin(phi). Horizontal motion is sum (k/|k|) a cos(phi), the
// Gerstner form, which pulls points toward each crest.
inline std::array<double, 3> oceanDisplacement(const std::vector<WindSpectrumWave>& waves,
                                               double x, double y, double seconds,
                                               double footprint = 0) noexcept {
  std::array<double, 3> d{};
  for (const auto& w : waves) {
    const auto phi = w.kx * x + w.ky * y - w.omega * seconds + w.phase;
    const auto k = std::hypot(w.kx, w.ky);
    const auto c = w.amplitude * oceanDetailWeight(k, footprint) * std::cos(phi);
    d[0] += w.kx / k * c; d[1] += w.ky / k * c;
    d[2] += w.amplitude * oceanDetailWeight(k, footprint) * std::sin(phi);
  }
  return d;
}

// Row-major XY grid layout of a position field, or false for any other layout.
inline bool oceanGridLayout(const std::vector<FieldElement>& positions, std::size_t& columns, std::size_t& rows) {
  const auto count = positions.size();
  columns = 1; rows = 0;
  if (count < 4) return false;
  while (columns < count && positions[columns].components[1] == positions[0].components[1]) ++columns;
  rows = count / columns;
  if (columns < 2 || rows < 2 || columns * rows != count) return false;
  for (std::size_t i = 0; i < count; ++i) {
    const auto column = i % columns;
    if (positions[i].components[0] != positions[column].components[0]
        || positions[i].components[1] != positions[i - column].components[1]) return false;
  }
  return true;
}

// Each grid vertex's footprint: the larger of its widest neighbouring cell and the
// authored footprint, so a coarse or graded grid filters what it cannot carry.
inline bool oceanFootprints(const std::vector<FieldElement>& positions,
                            const std::vector<FieldElement>& authored,
                            std::vector<double>& footprints, std::string& error) {
  footprints.clear();
  std::size_t columns = 0, rows = 0;
  if (!oceanGridLayout(positions, columns, rows)
      || (authored.size() != 1 && authored.size() != positions.size())) {
    error = "Ocean wave detail requires a regular XY grid and one footprint per vertex"; return false;
  }
  footprints.resize(positions.size());
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t column = 0; column < columns; ++column) {
      const auto i = row * columns + column;
      const auto x = [&](std::size_t c) { return positions[c].components[0]; };
      const auto y = [&](std::size_t r) { return positions[r * columns].components[1]; };
      double cell = 0;
      if (column > 0) cell = std::max(cell, std::abs(x(column) - x(column - 1)));
      if (column + 1 < columns) cell = std::max(cell, std::abs(x(column + 1) - x(column)));
      if (row > 0) cell = std::max(cell, std::abs(y(row) - y(row - 1)));
      if (row + 1 < rows) cell = std::max(cell, std::abs(y(row + 1) - y(row)));
      const auto own = authored[authored.size() == 1 ? 0 : i].components[0];
      if (!(std::isfinite(own) && own >= 0)) { error = "Ocean wave footprint must be finite and nonnegative"; return false; }
      footprints[i] = std::max(cell, own);
    }
  error.clear();
  return true;
}

// The grid form of oceanDisplacement: one sine table per column and one sine pair
// per row for each wave. False leaves out empty, and the caller uses the formula.
// Footprints, when given, hold one per vertex and fade the waves each vertex cannot carry.
inline bool oceanGridDisplacements(const std::vector<WindSpectrumWave>& waves, const OceanSettings& o,
                                   const std::vector<FieldElement>& positions,
                                   const std::vector<FieldElement>& times,
                                   std::vector<std::array<double, 3>>& out,
                                   const std::vector<double>* footprints = nullptr) {
  out.clear();
  std::size_t columns = 0, rows = 0;
  if (times.empty() || (times.size() != 1 && times.size() != positions.size())) return false;
  const auto timeline = times[0].components[0];
  if (!std::all_of(times.begin(), times.end(), [&](const auto& t) { return t.components[0] == timeline; })
      || !oceanGridLayout(positions, columns, rows)
      || (footprints && footprints->size() != positions.size())) return false;
  const auto seconds = oceanSeconds(o, timeline);
  out.assign(positions.size(), {});
  std::vector<double> columnSin(columns), columnCos(columns);
  for (const auto& w : waves) {
    const auto k = std::hypot(w.kx, w.ky);
    const auto ux = w.kx / k, uy = w.ky / k;
    for (std::size_t column = 0; column < columns; ++column) {
      const auto angle = w.kx * positions[column].components[0];
      columnSin[column] = std::sin(angle); columnCos[column] = std::cos(angle);
    }
    for (std::size_t row = 0; row < rows; ++row) {
      const auto rowAngle = w.ky * positions[row * columns].components[1] - w.omega * seconds + w.phase;
      const auto rowSin = w.amplitude * std::sin(rowAngle), rowCos = w.amplitude * std::cos(rowAngle);
      auto* cell = out.data() + row * columns;
      for (std::size_t column = 0; column < columns; ++column) {
        const auto g = footprints ? oceanDetailWeight(k, (*footprints)[row * columns + column]) : 1.0;
        const auto s = g * (columnSin[column] * rowCos + columnCos[column] * rowSin);
        const auto c = g * (columnCos[column] * rowCos - columnSin[column] * rowSin);
        cell[column][0] += ux * c; cell[column][1] += uy * c; cell[column][2] += s;
      }
    }
  }
  return true;
}

// Foam marks where the displaced surface folds or compresses. The horizontal
// Jacobian [[1 + dDx/dx, dDx/dy], [dDy/dx, 1 + dDy/dy]] comes from central
// differences on the grid (one-sided at its edges); foam is
// clamp(coverage - smaller eigenvalue, 0, 1). A flat sea has eigenvalue 1, so
// coverage 1 starts foam at any compression and 0 only where the surface folds.
inline bool oceanFoamValues(const std::vector<FieldElement>& positions,
                            const std::vector<FieldElement>& displacement,
                            double coverage, std::vector<double>& foam, std::string& error) {
  foam.clear();
  std::size_t columns = 0, rows = 0;
  if (positions.size() != displacement.size() || !oceanGridLayout(positions, columns, rows)) {
    error = "Ocean foam requires a regular XY grid and one displacement per vertex"; return false;
  }
  if (!(coverage >= 0 && coverage <= kOceanMaximumFoamCoverage)) {
    error = "Ocean refused: foam coverage " + std::to_string(coverage) + " is outside 0 to "
        + std::to_string(kOceanMaximumFoamCoverage);
    return false;
  }
  foam.resize(positions.size());
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t column = 0; column < columns; ++column) {
      const auto i = row * columns + column;
      const auto left = row * columns + (column > 0 ? column - 1 : column);
      const auto right = row * columns + (column + 1 < columns ? column + 1 : column);
      const auto down = (row > 0 ? row - 1 : row) * columns + column;
      const auto up = (row + 1 < rows ? row + 1 : row) * columns + column;
      const auto dx = positions[right].components[0] - positions[left].components[0];
      const auto dy = positions[up].components[1] - positions[down].components[1];
      if (dx == 0 || dy == 0) { error = "Ocean foam requires distinct grid coordinates"; return false; }
      const auto& r = displacement[right].components; const auto& l = displacement[left].components;
      const auto& u = displacement[up].components; const auto& d = displacement[down].components;
      const auto a = 1 + (r[0] - l[0]) / dx, b = (u[0] - d[0]) / dy;
      const auto c = (r[1] - l[1]) / dx, e = 1 + (u[1] - d[1]) / dy;
      const auto half = (a + e) / 2, det = a * e - b * c;
      const auto smaller = half - std::sqrt(std::max(0.0, half * half - det));
      foam[i] = std::clamp(coverage - smaller, 0.0, 1.0);
    }
  error.clear();
  return true;
}
} // namespace videowire::geometry
