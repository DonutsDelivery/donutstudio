#include "../../shared/GeometryCore.h"
#include "../../shared/GeometryCoreTransport.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
#include <string>
#include <variant>
#include <vector>

// Density distribution on faces: the realized count is floor(area * density), capped by
// a maximum that overflow rejects, and the placement is seeded and area-weighted. The
// density operation is written at transport version 18 and an older header refuses it.
namespace {
using namespace videowire::geometry;

int failures = 0;

void check(bool condition, const char *message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
  }
}

PortContract meshContract() {
  PortContract contract;
  contract.carrier = CarrierKind::geometry3D;
  contract.overflow = OverflowPolicy::reject;
  contract.maxVertices = 64;
  contract.maxIndices = 256;
  contract.maxAttributes = 32;
  return contract;
}

PortContract pointsContract() {
  PortContract contract;
  contract.carrier = CarrierKind::points3D;
  contract.overflow = OverflowPolicy::reject;
  contract.maxPoints = 64;
  contract.maxVertices = 64;
  contract.maxIndices = 256;
  contract.maxAttributes = 32;
  return contract;
}

// Two triangles covering a 2 x 2 square in the z=0 plane. Its area is exactly 4.
GeometryData squareMesh() {
  GeometryData mesh;
  mesh.positions = {{0.0f, 0.0f, 0.0f}, {2.0f, 0.0f, 0.0f}, {2.0f, 2.0f, 0.0f}, {0.0f, 2.0f, 0.0f}};
  mesh.vertexIds = {1, 2, 3, 4};
  mesh.indices = {0, 1, 2, 0, 2, 3};
  return mesh;
}

// Surface area summed from triangle cross products, with the half factor applied.
double meshArea(const GeometryData &mesh) {
  double total = 0;
  for (std::size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
    const auto &a = mesh.positions[mesh.indices[i]];
    const auto &b = mesh.positions[mesh.indices[i + 1]];
    const auto &c = mesh.positions[mesh.indices[i + 2]];
    const double ux = double(b.x) - a.x, uy = double(b.y) - a.y, uz = double(b.z) - a.z;
    const double vx = double(c.x) - a.x, vy = double(c.y) - a.y, vz = double(c.z) - a.z;
    const double x = uy * vz - uz * vy, y = uz * vx - ux * vz, z = ux * vy - uy * vx;
    total += 0.5 * std::sqrt(x * x + y * y + z * z);
  }
  return total;
}

// A density operation over the square, with the parameter array as the caller writes it.
OperationRecord densityOp(const std::array<float, 4> &parameters) {
  return OperationRecord{1, OperationCode::pointsOnFacesDensity, 1, 2, parameters};
}

bool materializes(const std::array<float, 4> &parameters, std::size_t expectedCount) {
  PointsData points;
  std::string error;
  const bool ok = materializeDistributedPoints(squareMesh(), densityOp(parameters), points, error);
  return ok && error.empty() && points.points.size() == expectedCount;
}

bool refuses(const std::array<float, 4> &parameters) {
  PointsData points;
  std::string error;
  return !materializeDistributedPoints(squareMesh(), densityOp(parameters), points, error)
      && !error.empty();
}

// Area 4 times density: 2 gives 8, 0.5 gives 2, and 0.3 gives floor(1.2) = 1.
void densityCountIsFloorOfAreaTimesDensity() {
  check(materializes({16, 7, 2, 0}, 8), "density 2 on area 4 yields 8 points");
  check(materializes({16, 7, 0.5f, 0}, 2), "density 0.5 on area 4 yields 2 points");
  check(materializes({16, 7, 0.3f, 0}, 1), "density 0.3 on area 4 truncates to 1 point");
  check(materializes({64, 7, 16, 0}, 64), "density 16 on area 4 yields exactly the 64 point maximum");
}

// A count above the maximum is rejected rather than truncated, and so is a count of zero.
void densityRejectsOverflowAndEmptyCounts() {
  check(refuses({16, 7, 17, 0}), "density 17 needs 68 points and overflows a maximum of 16");
  check(refuses({64, 7, 17, 0}), "density 17 needs 68 points and overflows the 64 point maximum");
  check(refuses({16, 7, 0.2f, 0}), "density 0.2 yields zero points and is rejected");
  check(refuses({0, 7, 2, 0}), "a zero maximum is rejected");
}

// Density, seed and reserved parameters must be finite, positive and in range.
void densityRejectsInvalidParameters() {
  check(refuses({16, 7, 0, 0}), "a zero density is rejected");
  check(refuses({16, 7, -1, 0}), "a negative density is rejected");
  check(refuses({16, 7, std::numeric_limits<float>::quiet_NaN(), 0}), "a NaN density is rejected");
  check(refuses({16, 7, std::numeric_limits<float>::infinity(), 0}), "an infinite density is rejected");
  check(refuses({16, 7, 2, 1}), "a nonzero reserved parameter is rejected");
  check(refuses({16, 7.5f, 2, 0}), "a fractional seed is rejected");
}

// The count parameter keeps its old meaning: a density value on Distribute Points on Faces is refused.
void countOperationIsUnchanged() {
  PointsData points;
  std::string error;
  const OperationRecord count{1, OperationCode::pointsOnFaces, 1, 2, {8, 7, 0, 0}};
  check(materializeDistributedPoints(squareMesh(), count, points, error)
            && points.points.size() == 8,
        "Distribute Points on Faces still yields its count");
  const OperationRecord forged{1, OperationCode::pointsOnFaces, 1, 2, {8, 7, 1, 0}};
  check(!materializeDistributedPoints(squareMesh(), forged, points, error),
        "Distribute Points on Faces refuses a density value in the count operation");
}

// Same seed, same points. A different seed moves at least one point. Every point lies on the square.
void densityIsSeededAndInsideSurface() {
  PointsData first, again, other;
  std::string error;
  const auto seeded = densityOp({16, 7, 2, 0});
  check(materializeDistributedPoints(squareMesh(), seeded, first, error), "seeded density distribution materializes");
  check(materializeDistributedPoints(squareMesh(), seeded, again, error), "repeated density distribution materializes");
  bool identical = first.points.size() == again.points.size();
  for (std::size_t i = 0; identical && i < first.points.size(); ++i)
    identical = first.points[i].position.x == again.points[i].position.x
        && first.points[i].position.y == again.points[i].position.y
        && first.points[i].position.z == again.points[i].position.z
        && first.points[i].stableId == again.points[i].stableId;
  check(identical, "the same seed reproduces the same density points exactly");

  check(materializeDistributedPoints(squareMesh(), densityOp({16, 8, 2, 0}), other, error),
        "a different seed materializes");
  bool moved = false;
  for (std::size_t i = 0; i < first.points.size() && i < other.points.size(); ++i)
    moved = moved || first.points[i].position.x != other.points[i].position.x
        || first.points[i].position.y != other.points[i].position.y;
  check(moved, "a different seed moves at least one density point");

  bool inside = first.points.size() == 8;
  for (std::size_t i = 0; inside && i < first.points.size(); ++i) {
    const auto &p = first.points[i].position;
    inside = std::fabs(p.z) <= 1e-6f && p.x >= -1e-5f && p.x <= 2.0f + 1e-5f
        && p.y >= -1e-5f && p.y <= 2.0f + 1e-5f && first.points[i].stableId == i + 1;
  }
  check(inside, "density points lie on the square with stable identities 1 to 8");
}

// Lowering on a cube: the count is floor(area times density), and the operation is recorded.
void densityLowersWithExactCount() {
  std::string error;
  const auto cube = lowerMeshGenerator(7301, 7301, 1, OperationCode::cube, {2, 2, 1, 0}, meshContract(), error);
  check(cube.has_value(), "a cube lowers for the density lowering check");
  if (!cube) return;
  const double area = meshArea(std::get<GeometryData>(cube->data));
  const auto expected = static_cast<std::size_t>(std::floor(area * 0.25));
  check(expected >= 1, "the cube yields at least one point at density 0.25");

  auto points = lowerPointDensityDistribution(*cube, 7302, 64, 3, 0.25f, pointsContract(), error);
  check(points.has_value(), "a density distribution lowers on a cube");
  if (!points) return;
  check(std::get<PointsData>(points->data).points.size() == expected,
        "the lowered density count equals floor(area times density)");
  check(points->operations.back().code == OperationCode::pointsOnFacesDensity
            && points->operations.back().parameters[2] == 0.25f,
        "the lowered operation records the density");

  error.clear();
  check(!lowerPointDensityDistribution(*cube, 7303, 64, 3, 1000.0f, pointsContract(), error).has_value()
            && !error.empty(),
        "a density that overflows the maximum is rejected at lowering");
}

// A geometryDeformation (Audio Deformer) binding is accepted by the count operation but refused by
// density, whose realized count changes with the per-frame area.
void densityRefusesAudioDeformerBindings() {
  std::string error;
  const auto cube = lowerMeshGenerator(7321, 7321, 1, OperationCode::cube, {2, 2, 1, 0}, meshContract(), error);
  check(cube.has_value(), "a cube lowers for the binding refusal check");
  if (!cube) return;
  spectrum::Binding binding;
  binding.fieldStableId = 7322; binding.sourceGeometryStableId = cube->stableId;
  binding.consumerGeometryStableId = 7323;
  binding.target = spectrum::Target::geometryDeformation;
  binding.coordinates.resize(std::get<GeometryData>(cube->data).positions.size(), 0.0f);
  auto bound = *cube;
  bound.spectrumFields = {binding};

  check(lowerPointDistribution(bound, 7324, OperationCode::pointsOnFaces, 8, 7, pointsContract(), error).has_value(),
        "the count operation still lowers under an Audio Deformer binding");
  error.clear();
  check(!lowerPointDensityDistribution(bound, 7325, 64, 3, 0.25f, pointsContract(), error).has_value()
            && error == "Density point distribution does not support Audio Deformer bindings",
        "the density operation refuses an Audio Deformer binding by name");
}

// The density operation is written at version 18, roundtrips, and a version 17 header refuses it.
void densityTransportVersionAndRoundTrip() {
  std::string error;
  const auto cube = lowerMeshGenerator(7311, 7311, 1, OperationCode::cube, {2, 2, 1, 0}, meshContract(), error);
  check(cube.has_value(), "a cube lowers for the transport check");
  if (!cube) return;
  auto points = lowerPointDensityDistribution(*cube, 7312, 64, 3, 0.25f, pointsContract(), error);
  check(points.has_value(), "a density distribution lowers for the transport check");
  if (!points) return;
  auto admitted = admitValue(*points, pointsContract(), error);
  check(admitted.has_value(), "a density distribution admits");
  if (!admitted) return;
  const auto bytes = encodeRuntimeValue(*admitted);
  check(bytes.size() > 5 && bytes[4] == kPointDensityTransportVersion && bytes[5] == 0,
        "a density distribution is written at transport version 18");
  check(kPointDensityTransportVersion == kTransportVersion
            && kPointDensityTransportVersion > kCullTransportVersion,
        "the density version is the current transport version and follows the cull version");
  auto reopened = decodeRuntimeValue(bytes, pointsContract(), {}, {}, error);
  check(reopened.has_value() && encodeRuntimeValue(*reopened) == bytes,
        "a version 18 density distribution roundtrips");

  auto forged = bytes;
  forged[4] = 17;
  forged[5] = 0;
  error.clear();
  check(!decodeRuntimeValue(forged, pointsContract(), {}, {}, error).has_value()
            && error.find("unknown Geometry Core operation") != std::string::npos,
        "a version 17 header refuses the density operation");

  // The count operation keeps the version it had before this change.
  auto legacy = lowerPointDistribution(*cube, 7313, OperationCode::pointsOnFaces, 8, 7, pointsContract(), error);
  check(legacy.has_value(), "a count distribution lowers for the version check");
  if (!legacy) return;
  auto legacyAdmitted = admitValue(*legacy, pointsContract(), error);
  check(legacyAdmitted.has_value() && encodeRuntimeValue(*legacyAdmitted)[4] == kNonTextTransportVersion,
        "a count distribution keeps the pre-17 non-text transport version");
}
}  // namespace

int main() {
  densityCountIsFloorOfAreaTimesDensity();
  densityRejectsOverflowAndEmptyCounts();
  densityRejectsInvalidParameters();
  countOperationIsUnchanged();
  densityIsSeededAndInsideSurface();
  densityLowersWithExactCount();
  densityTransportVersionAndRoundTrip();
  densityRefusesAudioDeformerBindings();
  if (failures != 0) {
    std::fprintf(stderr, "%d geometry core distribution check(s) failed\n", failures);
    return 1;
  }
  std::printf("geometry core distribution tests passed\n");
  return 0;
}
