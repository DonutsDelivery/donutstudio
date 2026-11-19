#pragma once

// The open sea's chain (Grid, Wind Spectrum, Shade Smooth) as a wave table, so a GPU
// can move its vertices each frame instead of replaying the whole mesh on the CPU.
// Only the exact chain qualifies: a flat row-major grid displaced along Z by the
// wind spectrum at timeline time, then smooth shaded with no sharp edges. Any other
// operation, field or binding returns nothing and the CPU replay stays in use.
//
// The GPU reproduces what the replay computes for each vertex: the height is the
// sum of amplitude * sin(kx * x + ky * y + phase(t)), and the normal is the
// angle-weighted sum of the six grid triangles around the vertex (Shade Smooth's
// weights over the grid's split a-b-c, b-d-c), normalized.

#include "GeometryCore.h"
#include "GeometryFieldOperations.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

namespace videowire::geometry {

struct SurfaceWaveProgram final {
  std::uint32_t columns = 0, rows = 0;
  // The displacement strength, applied to every amplitude.
  float strength = 1;
  // Timeline seconds T give wave time (T - timeOrigin) * timeGain + timeBias.
  double timeOrigin = 0, timeGain = 1, timeBias = 0;
  std::vector<WindSpectrumWave> waves;
};

inline std::optional<SurfaceWaveProgram> recognizeSurfaceWaveProgram(const ValueDescriptor& value) {
  if (value.carrier != CarrierKind::geometry3D || !value.spectrumFields.empty() || !value.scoreFields.empty()
      || value.operations.size() != 8)
    return std::nullopt;
  const OperationRecord* found[8] {};
  const OperationCode order[8] {OperationCode::grid, OperationCode::positionField, OperationCode::fieldVectorMeasure,
      OperationCode::fieldTimeline, OperationCode::fieldNoise, OperationCode::fieldComposeVector,
      OperationCode::vectorDisplacement, OperationCode::shadeMesh};
  for (const auto& op : value.operations) {
    std::size_t slot = 8;
    for (std::size_t i = 0; i < 8; ++i) if (op.code == order[i]) slot = i;
    if (slot == 8 || found[slot]) return std::nullopt;
    found[slot] = &op;
  }
  const auto& grid = *found[0]; const auto& positions = *found[1]; const auto& zero = *found[2];
  const auto& time = *found[3]; const auto& noise = *found[4]; const auto& compose = *found[5];
  const auto& displace = *found[6]; const auto& shade = *found[7];
  // The chain's links, by output identity.
  if (positions.inputStableId != grid.outputStableId || zero.inputStableId != positions.outputStableId
      || time.inputStableId != zero.outputStableId || noise.inputStableId != positions.outputStableId
      || noise.secondaryInputStableId != time.outputStableId || compose.inputStableId != noise.outputStableId
      || compose.secondaryInputStableId != 0 || displace.inputStableId != grid.outputStableId
      || displace.secondaryInputStableId != compose.outputStableId || shade.inputStableId != displace.outputStableId
      || value.stableId != shade.outputStableId)
    return std::nullopt;
  SurfaceWaveProgram program;
  // A graded grid still has its rest X and Y in the vertex buffer, and the GPU
  // reads them from there, so only the counts matter here.
  if (!exactIntegerParameter(grid.parameters[0], 2, program.columns)
      || !exactIntegerParameter(grid.parameters[1], 2, program.rows))
    return std::nullopt;
  const auto& t = time.field;
  if (t.mode != 0 || t.domain != Domain::vertex || t.values[0] <= 0 || t.values[6] != 0 || t.values[7] != 0)
    return std::nullopt;
  program.timeOrigin = t.values[1]; program.timeGain = t.values[4]; program.timeBias = t.values[5];
  const auto& n = noise.field;
  if (n.mode != kNoiseFieldWindSpectrumMode || n.domain != Domain::vertex || n.valueType != ValueType::floatValue)
    return std::nullopt;
  std::string error;
  if (!windSpectrumWaves(n.values[0], n.values[1], n.values[2], n.values[3], n.values[4], program.waves, error,
          {n.values[5], n.values[6], n.values[7]}))
    return std::nullopt;
  const auto& c = compose.field;
  if (c.mode != 1 || c.valueType != ValueType::vector || c.values != std::array<double, 8>{0, 0, 1, 0, 0, 0, 0, 0})
    return std::nullopt;
  program.strength = displace.parameters[0];
  if (!finite(program.strength) || displace.parameters[1] != 0 || displace.parameters[2] != 0
      || displace.parameters[3] != 0)
    return std::nullopt;
  if (shade.parameters != std::array<float, 4>{kShadeSmooth, shade.parameters[1], 0, 0} || !shade.sharpEdges.empty())
    return std::nullopt;
  return program;
}

// One row per wave for one frame: kx, ky, amplitude times strength, and the phase
// at this time reduced to [0, 2 pi) in double, so the GPU adds only kx * x + ky * y.
inline std::vector<std::array<float, 4>> surfaceWaveTable(const SurfaceWaveProgram& program, double timelineSeconds) {
  constexpr double tau = 6.283185307179586476925286766559;
  const auto seconds = (timelineSeconds - program.timeOrigin) * program.timeGain + program.timeBias;
  std::vector<std::array<float, 4>> table;
  table.reserve(program.waves.size());
  for (const auto& wave : program.waves) {
    auto phase = std::fmod(wave.phase - wave.omega * seconds, tau);
    if (phase < 0) phase += tau;
    table.push_back({float(wave.kx), float(wave.ky), float(wave.amplitude * program.strength), float(phase)});
  }
  return table;
}

} // namespace videowire::geometry
