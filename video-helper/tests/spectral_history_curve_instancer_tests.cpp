#include "../../shared/GeometryCore.h"

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <string>
#include <variant>
#include <vector>

namespace geo = videowire::geometry;

namespace {

int failures = 0;

void check(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

// Known analysis. Each (frame, band) pair has its own value, so a row that reads
// the wrong frame or the wrong band cannot match the oracle by accident. Frame k
// is the analysis time k / 4 seconds.
double curveHistoryValue(long frame, std::size_t band) {
  return 0.01 * (10.0 * static_cast<double>(frame)
                 + 3.0 * (static_cast<double>(band) - 8.0) + 1.0);
}

geo::spectrum::Bands curveHistorySpectrum(double seconds) {
  geo::spectrum::Bands bands {};
  const long frame = std::lround(seconds * 4.0);
  for (std::size_t band = 8; band <= 11; ++band)
    bands[band] = static_cast<float>(curveHistoryValue(frame, band));
  return bands;
}

// Mirrors the backend instancer path. Rows are evaluated once per frame, then
// each instance samples the row chosen by its history coordinate.
bool curveHistoryReplay(const geo::spectrum::Binding& binding, double seconds,
                        geo::spectrum::Follower& follower,
                        std::vector<geo::spectrum::Bands>& rows,
                        std::vector<float>& amounts, std::string& error) {
  if (!geo::spectrum::evaluateHistory(binding, {}, seconds, curveHistorySpectrum,
                                      false, false, follower, rows, error))
    return false;
  amounts.resize(binding.coordinates.size());
  for (std::size_t instance = 0; instance < amounts.size(); ++instance)
    amounts[instance] = geo::spectrum::sampleHistory(binding, rows, instance);
  return true;
}

// Frame index of the analysis that instance i shows at analysis frame `frame`.
// Instance i reads row i, so it shows the band of frame - i.
float curveHistoryExpectedAmount(long frame, std::size_t instance) {
  return static_cast<float>(curveHistoryValue(4 + frame - static_cast<long>(instance),
                                              8 + instance));
}

void spectralHistoryDrivesCurveSpectrumInstancer() {
  std::string error;

  geo::PortContract meshContract;
  meshContract.carrier = geo::CarrierKind::geometry3D;
  meshContract.overflow = geo::OverflowPolicy::reject;
  meshContract.maxVertices = 64;
  meshContract.maxIndices = 256;
  const auto prototype = geo::lowerMeshGenerator(9850, 9850, 1, geo::OperationCode::plane,
                                                 {0.1f, 0.1f, 0.0f, 0.0f}, meshContract, error);

  // Three points run from x = -1 to x = 1. Four instances sit at arc positions
  // 2 i / 3, so their x coordinates are -1 + 2 i / 3.
  geo::PortContract curveContract;
  curveContract.carrier = geo::CarrierKind::curves3D;
  curveContract.maxCurvePoints = 8;
  curveContract.maxSplines = 1;
  const auto curve = geo::lowerCurveLine(9851, 9851, 1, 3, 2.0f, curveContract, error);
  check(prototype.has_value() && curve.has_value(), "history curve placement fixtures lower");
  if (!prototype || !curve) {
    std::cerr << error << '\n';
    ++failures;
    return;
  }

  geo::PortContract instanceContract;
  instanceContract.carrier = geo::CarrierKind::instances3D;
  instanceContract.overflow = geo::OverflowPolicy::reject;
  instanceContract.maxInstances = 64;

  geo::spectrum::Binding binding;
  binding.firstBand = 8;
  binding.lastBand = 11;
  binding.direction = {0.0f, 1.0f, 0.0f};
  binding.history.enabled = true;
  binding.history.depth = 4;
  binding.history.spacingSeconds = 0.25f;
  const geo::spectrum::HistoryAxis byIndex {0, -1.0, 1.0};
  const auto layout = geo::lowerSpectrumInstancer(*prototype, 9852, binding,
                                                  {3.0f, 2.0f, 2.0f, 2.0f},
                                                  instanceContract, error, &*curve, &byIndex);
  check(layout.has_value(), "curve Spectrum Instancer lowers with Spectral History rows");
  if (!layout) {
    std::cerr << error << '\n';
    ++failures;
    return;
  }
  const auto& instances = std::get<geo::InstancesData>(layout->data).instances;
  const auto& instancer = layout->spectrumFields.back();
  check(instances.size() == 4 && instancer.coordinates.size() == 4
        && instancer.history.coordinates.size() == 4,
        "curve instancer has one instance and one history row per band");
  if (instances.size() != 4 || instancer.history.coordinates.size() != 4) {
    std::cerr << error << '\n';
    ++failures;
    return;
  }

  // Assertions 1 and 2 use the same five analysis frames at 1.00, 1.25, ... 2.00 s.
  geo::spectrum::Follower follower;
  std::vector<geo::spectrum::Bands> rows;
  std::vector<float> amounts;
  bool placementStable = true, countStable = true, rowsFollowHistory = true;
  for (long frame = 0; frame < 5; ++frame) {
    const double seconds = 1.0 + 0.25 * static_cast<double>(frame);
    if (!curveHistoryReplay(instancer, seconds, follower, rows, amounts, error)) {
      std::cerr << error << '\n';
      ++failures;
      return;
    }
    countStable = countStable && amounts.size() == instances.size() && rows.size() == 4;

    // Assertion 1: each instance stays at its arc-length slot. Only the displacement
    // along the binding direction (Y here) changes with the frame.
    for (std::size_t i = 0; i < instances.size(); ++i) {
      const auto& base = instances[i].transform.translation;
      const float arc = -1.0f + 2.0f * static_cast<float>(i) / 3.0f;
      const float x = base.x + amounts[i] * instancer.direction[0];
      const float y = base.y + amounts[i] * instancer.direction[1];
      const float z = base.z + amounts[i] * instancer.direction[2];
      placementStable = placementStable && std::abs(base.x - arc) < 1.0e-5f
          && std::abs(base.y) < 1.0e-6f && std::abs(base.z) < 1.0e-6f
          && std::abs(x - arc) < 1.0e-5f && std::abs(y - amounts[i]) < 1.0e-6f
          && std::abs(z) < 1.0e-6f;
    }

    // Assertion 2: row k at this frame holds the band values of frame - k, and
    // instance i reads row i.
    for (std::size_t row = 0; row < 4; ++row)
      for (std::size_t band = 8; band <= 11; ++band) {
        const float expected = static_cast<float>(
            curveHistoryValue(4 + frame - static_cast<long>(row), band));
        rowsFollowHistory = rowsFollowHistory && std::abs(rows[row][band] - expected) < 1.0e-6f;
      }
    for (std::size_t i = 0; i < instances.size(); ++i)
      rowsFollowHistory = rowsFollowHistory
          && std::abs(amounts[i] - curveHistoryExpectedAmount(frame, i)) < 1.0e-6f;
  }
  check(countStable, "curve instancer keeps its instance count across analysis frames");
  check(placementStable, "curve instancer keeps arc-length placement across analysis frames");
  check(rowsFollowHistory, "curve instancer row k shows the band of frame minus k");

  // Assertion 3, baked history: a backward time jump carries no state. Going back
  // to 1.00 s after 1.50 s must match a fresh evaluation at 1.00 s.
  geo::spectrum::Follower jumped, fresh;
  std::vector<geo::spectrum::Bands> jumpedRows, freshRows;
  std::vector<float> jumpedAmounts, freshAmounts;
  const bool jumpOk = curveHistoryReplay(instancer, 1.5, jumped, rows, amounts, error)
      && curveHistoryReplay(instancer, 1.0, jumped, jumpedRows, jumpedAmounts, error)
      && curveHistoryReplay(instancer, 1.0, fresh, freshRows, freshAmounts, error);
  bool jumpMatchesOracle = true;
  for (std::size_t i = 0; i < instances.size(); ++i)
    jumpMatchesOracle = jumpMatchesOracle
        && std::abs(jumpedAmounts[i] - curveHistoryExpectedAmount(0, i)) < 1.0e-6f;
  check(jumpOk && jumpedRows == freshRows && jumpedAmounts == freshAmounts && jumpMatchesOracle,
        "baked curve history after a backward jump equals a fresh evaluation");

  // Assertion 3, live history: a backward jump clears the recorded rows. Only the
  // newly received frame remains, so row 0 holds it and rows 1 to 3 are empty.
  geo::spectrum::Binding live = instancer;
  live.history.live = true;
  geo::spectrum::Follower liveFollower;
  std::vector<geo::spectrum::Bands> liveRows;
  bool liveOk = true;
  for (const double seconds : {1.0, 1.25, 1.5})
    liveOk = liveOk && geo::spectrum::evaluateHistory(live, curveHistorySpectrum(seconds), seconds,
                                                      {}, true, false, liveFollower, liveRows, error);
  const double backward = 0.5;
  liveOk = liveOk && geo::spectrum::evaluateHistory(live, curveHistorySpectrum(backward), backward,
                                                    {}, true, false, liveFollower, liveRows, error);
  bool cleared = liveRows.size() == 4;
  for (std::size_t row = 1; cleared && row < liveRows.size(); ++row)
    for (std::size_t band = 8; band <= 11; ++band)
      cleared = cleared && liveRows[row][band] == 0.0f;
  check(liveOk && cleared, "live curve history clears its rows on a backward jump");
  check(liveOk && liveRows.size() == 4
        && std::abs(liveRows[0][8] - static_cast<float>(curveHistoryValue(2, 8))) < 1.0e-6f,
        "live curve history keeps the newly received frame in row 0");
  if (!liveOk) {
    std::cerr << error << '\n';
    ++failures;
  }
}

}  // namespace

int main() {
  spectralHistoryDrivesCurveSpectrumInstancer();
  if (failures == 0)
    std::cout << "spectral history curve instancer: all checks passed\n";
  return failures == 0 ? 0 : 1;
}
