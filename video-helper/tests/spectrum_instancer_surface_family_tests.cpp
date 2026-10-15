#include "../../shared/GeometryCore.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <optional>
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

void report(const std::string& error) {
  if (!error.empty()) std::cerr << error << '\n';
}

// Four bands (8 to 11) give four instances. Band b keeps identity b + 1, so the
// instance ids are 9 to 12 and every instance cites the prototype as its source.
using Expected = std::array<std::array<double, 3>, 4>;
using Rows = std::array<double, 4>;

geo::PortContract meshContract() {
  geo::PortContract contract;
  contract.carrier = geo::CarrierKind::geometry3D;
  contract.overflow = geo::OverflowPolicy::reject;
  contract.maxVertices = 64;
  contract.maxIndices = 256;
  return contract;
}

geo::PortContract instanceContract() {
  geo::PortContract contract;
  contract.carrier = geo::CarrierKind::instances3D;
  contract.overflow = geo::OverflowPolicy::reject;
  contract.maxInstances = 64;
  return contract;
}

geo::spectrum::Binding bandBinding() {
  geo::spectrum::Binding binding;
  binding.firstBand = 8;
  binding.lastBand = 11;
  binding.direction = {0.0f, 1.0f, 0.0f};
  return binding;
}

geo::spectrum::Binding historyBinding() {
  auto binding = bandBinding();
  binding.history.enabled = true;
  binding.history.depth = 4;
  binding.history.spacingSeconds = 0.25f;
  return binding;
}

// Exact per-band check of instance count, identity, source and static placement.
bool expectPlacements(const std::optional<geo::ValueDescriptor>& layout,
                      geo::StableId source, const Expected& expected) {
  if (!layout) return false;
  const auto& instances = std::get<geo::InstancesData>(layout->data).instances;
  if (instances.size() != 4) return false;
  for (std::size_t i = 0; i < 4; ++i) {
    const auto& instance = instances[i];
    const auto& t = instance.transform.translation;
    if (instance.stableId != static_cast<geo::StableId>(9 + i) || instance.sourceStableId != source)
      return false;
    if (std::abs(t.x - expected[i][0]) > 1.0e-5 || std::abs(t.y - expected[i][1]) > 1.0e-5
        || std::abs(t.z - expected[i][2]) > 1.0e-5)
      return false;
  }
  return true;
}

// Exact per-row check of the Spectral History coordinate that each instance uses.
bool expectRows(const std::optional<geo::ValueDescriptor>& layout, const Rows& expected) {
  if (!layout) return false;
  const auto& coordinates = layout->spectrumFields.back().history.coordinates;
  if (coordinates.size() != 4) return false;
  for (std::size_t i = 0; i < 4; ++i)
    if (std::abs(static_cast<double>(coordinates[i]) - expected[i]) > 1.0e-5) return false;
  return true;
}

// Line, ring and grid arrangements read the prototype and need no placement source.
void lineRingGridFamilies(const geo::ValueDescriptor& proto, std::string& error) {
  const auto source = proto.stableId;

  // Line of width 2: x runs from -1 to 1 in steps of 2/3 across the four bands.
  Expected line {};
  for (std::size_t i = 0; i < 4; ++i)
    line[i] = {(static_cast<double>(i) / 3.0 - 0.5) * 2.0, 0.0, 0.0};
  const auto lineLayout = geo::lowerSpectrumInstancer(proto, 9860, bandBinding(),
      {0.0f, 2.0f, 2.0f, 2.0f}, instanceContract(), error);
  check(expectPlacements(lineLayout, source, line),
        "line Spectrum Instancer places bands 8 to 11 at exact x slots");
  if (!lineLayout) report(error);

  // Line history rows follow the instance index, so instance i reads row i at age i / 3.
  const geo::spectrum::HistoryAxis byIndex {0, -1.0, 1.0};
  const auto lineRows = geo::lowerSpectrumInstancer(proto, 9861, historyBinding(),
      {0.0f, 2.0f, 2.0f, 2.0f}, instanceContract(), error, nullptr, &byIndex);
  check(expectPlacements(lineRows, source, line)
        && expectRows(lineRows, {0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0}),
        "line history rows follow the instance index without moving the slots");
  if (!lineRows) report(error);

  // Ring with width 2 and height 4: an ellipse with semi-axes 1 and 2, sampled at
  // quarter turns starting at angle zero.
  const Expected ring {{{1.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {-1.0, 0.0, 0.0}, {0.0, -2.0, 0.0}}};
  const auto ringLayout = geo::lowerSpectrumInstancer(proto, 9862, bandBinding(),
      {1.0f, 2.0f, 4.0f, 2.0f}, instanceContract(), error);
  check(expectPlacements(ringLayout, source, ring),
        "ring Spectrum Instancer places bands 8 to 11 at exact ellipse quarter turns");
  if (!ringLayout) report(error);

  // Ring history by X: the rows are the normalized X coordinates 1, 0, -1 and 0.
  const geo::spectrum::HistoryAxis byX {1, -1.0, 1.0};
  const auto ringRows = geo::lowerSpectrumInstancer(proto, 9863, historyBinding(),
      {1.0f, 2.0f, 4.0f, 2.0f}, instanceContract(), error, nullptr, &byX);
  check(expectPlacements(ringRows, source, ring)
        && expectRows(ringRows, {1.0, 0.5, 0.0, 0.5}),
        "ring history rows follow normalized X without moving the ellipse");
  if (!ringRows) report(error);

  // Grid of two columns over a 2 by 2 extent: rows run in Y, columns run in X.
  const Expected grid {{{-1.0, -1.0, 0.0}, {1.0, -1.0, 0.0}, {-1.0, 1.0, 0.0}, {1.0, 1.0, 0.0}}};
  const auto gridLayout = geo::lowerSpectrumInstancer(proto, 9864, bandBinding(),
      {2.0f, 2.0f, 2.0f, 2.0f}, instanceContract(), error);
  check(expectPlacements(gridLayout, source, grid),
        "grid Spectrum Instancer places bands 8 to 11 at exact cell corners");
  if (!gridLayout) report(error);

  // Grid history by Y: rows are the normalized Y coordinates 0, 0, 1 and 1.
  const geo::spectrum::HistoryAxis byY {2, -1.0, 1.0};
  const auto gridRows = geo::lowerSpectrumInstancer(proto, 9865, historyBinding(),
      {2.0f, 2.0f, 2.0f, 2.0f}, instanceContract(), error, nullptr, &byY);
  check(expectPlacements(gridRows, source, grid)
        && expectRows(gridRows, {0.0, 0.0, 1.0, 1.0}),
        "grid history rows follow normalized Y without moving the cells");
  if (!gridRows) report(error);
}

// Points arrangement: placement is the point set, and band k takes the k-th point
// in stable identity order.
void pointFamily(const geo::ValueDescriptor& proto, std::string& error) {
  const auto source = proto.stableId;
  const auto grid = geo::lowerGrid(9871, 9871, 1, 2, 2, 2.0f, meshContract(), error);
  check(grid.has_value(), "point family grid lowers");
  geo::PortContract pointsContract;
  pointsContract.carrier = geo::CarrierKind::points3D;
  pointsContract.maxPoints = 4;
  const auto cloud = grid ? geo::lowerPointsFromVertices(*grid, 9872, pointsContract, error)
                          : std::nullopt;
  check(cloud.has_value(), "point family lowers a four point Points3D placement");
  if (!cloud) {
    report(error);
    return;
  }
  auto sorted = std::get<geo::PointsData>(cloud->data).points;
  if (sorted.size() != 4) {
    check(false, "point family placement has four points");
    return;
  }
  std::sort(sorted.begin(), sorted.end(), [](const geo::PointRecord& a, const geo::PointRecord& b) {
    return a.stableId < b.stableId;
  });
  Expected expected {};
  Rows rowsByX {};
  for (std::size_t i = 0; i < 4; ++i) {
    expected[i] = {sorted[i].position.x, sorted[i].position.y, sorted[i].position.z};
    rowsByX[i] = (static_cast<double>(sorted[i].position.x) + 1.0) / 2.0;
  }
  const auto layout = geo::lowerSpectrumInstancer(proto, 9873, bandBinding(),
      {4.0f, 2.0f, 2.0f, 2.0f}, instanceContract(), error, &*cloud);
  check(expectPlacements(layout, source, expected),
        "point Spectrum Instancer places band k at the k-th point by stable identity");
  if (!layout) report(error);

  // Point history by X: each row is the normalized X of the point the instance uses.
  const geo::spectrum::HistoryAxis byX {1, -1.0, 1.0};
  const auto rows = geo::lowerSpectrumInstancer(proto, 9874, historyBinding(),
      {4.0f, 2.0f, 2.0f, 2.0f}, instanceContract(), error, &*cloud, &byX);
  check(expectPlacements(rows, source, expected) && expectRows(rows, rowsByX),
        "point history rows follow normalized X of the selected point");
  if (!rows) report(error);
}

// Mesh Surface arrangement: distributed points lie on the placement triangles.
// Exact identities and history rows are checked. Exact point positions depend on
// the retained area distribution, so they are checked against the square bounds.
void meshSurfaceFamily(const geo::ValueDescriptor& proto, std::string& error) {
  const auto source = proto.stableId;
  const auto surface = geo::lowerGrid(9881, 9881, 1, 2, 2, 2.0f, meshContract(), error);
  check(surface.has_value(), "mesh family placement surface lowers");
  if (!surface) {
    report(error);
    return;
  }
  const auto layout = geo::lowerSpectrumInstancer(proto, 9882, bandBinding(),
      {5.0f, 2.0f, 2.0f, 2.0f}, instanceContract(), error, &*surface);
  check(layout.has_value(), "mesh Spectrum Instancer lowers on a retained triangle surface");
  if (!layout) {
    report(error);
    return;
  }
  const auto& instances = std::get<geo::InstancesData>(layout->data).instances;
  bool onSurface = instances.size() == 4;
  for (std::size_t i = 0; onSurface && i < instances.size(); ++i) {
    const auto& instance = instances[i];
    const auto& t = instance.transform.translation;
    onSurface = instance.stableId == static_cast<geo::StableId>(9 + i)
        && instance.sourceStableId == source
        && std::abs(t.z) < 1.0e-6 && std::abs(t.x) <= 1.0 + 1.0e-5
        && std::abs(t.y) <= 1.0 + 1.0e-5;
  }
  check(onSurface, "mesh Spectrum Instancer keeps band identities on the square surface");

  // The retained area distribution is deterministic: a second lowering repeats every point.
  const auto again = geo::lowerSpectrumInstancer(proto, 9882, bandBinding(),
      {5.0f, 2.0f, 2.0f, 2.0f}, instanceContract(), error, &*surface);
  bool repeatable = again.has_value();
  if (repeatable) {
    const auto& repeat = std::get<geo::InstancesData>(again->data).instances;
    repeatable = repeat.size() == instances.size();
    for (std::size_t i = 0; repeatable && i < instances.size(); ++i) {
      const auto& a = instances[i].transform.translation;
      const auto& b = repeat[i].transform.translation;
      repeatable = a.x == b.x && a.y == b.y && a.z == b.z;
    }
  }
  check(repeatable, "mesh Spectrum Instancer repeats the same surface points on relowering");

  // Mesh history by Y: each row is the normalized Y of the surface point it follows.
  Rows rowsByY {};
  for (std::size_t i = 0; i < instances.size() && i < 4; ++i)
    rowsByY[i] = (static_cast<double>(instances[i].transform.translation.y) + 1.0) / 2.0;
  const geo::spectrum::HistoryAxis byY {2, -1.0, 1.0};
  const auto rows = geo::lowerSpectrumInstancer(proto, 9883, historyBinding(),
      {5.0f, 2.0f, 2.0f, 2.0f}, instanceContract(), error, &*surface, &byY);
  check(rows.has_value() && expectRows(rows, rowsByY),
        "mesh history rows follow normalized Y of each surface point");
  if (!rows) report(error);
}

// SDF Surface output is a Geometry3D triangle surface, so it is admitted as a Mesh
// Surface placement. The arrangement index 6 stays refused because it names no layout.
void sdfSurfaceFamily(const geo::ValueDescriptor& proto, std::string& error) {
  videowire::sdfsurface::Request request;
  request.cells = {10, 10, 10};
  request.maximumResidual = 0.06;
  videowire::SdfRecord sphere;
  sphere.stableId = 9007199254740001ull;
  sphere.operation = videowire::SdfOperation::sphere;
  sphere.parameterCount = 1;
  sphere.parameters[0] = 1.013;
  request.geometry.rootId = sphere.stableId;
  request.geometry.records = {sphere};

  geo::PortContract sdfContract;
  sdfContract.carrier = geo::CarrierKind::geometry3D;
  sdfContract.overflow = geo::OverflowPolicy::reject;
  sdfContract.maxVertices = 100000;
  sdfContract.maxIndices = 600000;
  sdfContract.maxAttributes = 3;
  sdfContract.attributes = geo::sdfSurfaceAttributes(9700);
  const auto sdf = geo::lowerSdfSurface(9700, 9701, 1, request, sdfContract, error);
  check(sdf.has_value(), "SDF sphere lowers as a Geometry3D placement surface");
  if (!sdf) {
    report(error);
    return;
  }

  const auto layout = geo::lowerSpectrumInstancer(proto, 9891, bandBinding(),
      {5.0f, 2.0f, 2.0f, 2.0f}, instanceContract(), error, &*sdf);
  check(layout.has_value(), "SDF surface lowers as a Mesh Surface Spectrum Instancer placement");
  if (!layout) {
    report(error);
    return;
  }
  const auto& instances = std::get<geo::InstancesData>(layout->data).instances;
  bool onSphere = instances.size() == 4;
  for (std::size_t i = 0; onSphere && i < instances.size(); ++i) {
    const auto& t = instances[i].transform.translation;
    const double radius = std::sqrt(static_cast<double>(t.x) * t.x + static_cast<double>(t.y) * t.y
        + static_cast<double>(t.z) * t.z);
    onSphere = instances[i].stableId == static_cast<geo::StableId>(9 + i)
        && instances[i].sourceStableId == proto.stableId && std::abs(radius - 1.013) < 0.1;
  }
  check(onSphere, "SDF Mesh Surface instances keep band identities on the sphere residual");

  const auto asCurve = geo::lowerSpectrumInstancer(proto, 9892, bandBinding(),
      {3.0f, 2.0f, 2.0f, 2.0f}, instanceContract(), error, &*sdf);
  check(!asCurve.has_value(), "SDF surface refuses as a curve placement");
  const auto asPoints = geo::lowerSpectrumInstancer(proto, 9893, bandBinding(),
      {4.0f, 2.0f, 2.0f, 2.0f}, instanceContract(), error, &*sdf);
  check(!asPoints.has_value(), "SDF surface refuses as a point placement");
  const auto asLayout6 = geo::lowerSpectrumInstancer(proto, 9894, bandBinding(),
      {6.0f, 2.0f, 2.0f, 2.0f}, instanceContract(), error, &*sdf);
  check(!asLayout6.has_value(), "arrangement 6 names no layout and refuses");
}

void spectrumInstancerSurfaceFamilies() {
  std::string error;
  const auto proto = geo::lowerMeshGenerator(9850, 9850, 1, geo::OperationCode::plane,
      {0.1f, 0.1f, 0.0f, 0.0f}, meshContract(), error);
  check(proto.has_value(), "surface family prototype lowers");
  if (!proto) {
    report(error);
    return;
  }
  lineRingGridFamilies(*proto, error);
  pointFamily(*proto, error);
  meshSurfaceFamily(*proto, error);
  sdfSurfaceFamily(*proto, error);
}

}  // namespace

int main() {
  spectrumInstancerSurfaceFamilies();
  if (failures == 0)
    std::cout << "spectrum instancer surface families: all checks passed\n";
  return failures == 0 ? 0 : 1;
}
