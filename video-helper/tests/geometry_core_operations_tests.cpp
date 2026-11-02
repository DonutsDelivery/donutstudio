#include "../../shared/GeometryCore.h"
#include "../../shared/GeometryCoreTransport.h"
#include "../../shared/SdfSurfaceOperationContract.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <set>
#include <string>
#include <variant>
#include <vector>

// Exact vertex, index, bounds and normal values for the Geometry Core operations
// that move, join, measure and displace geometry: Transform, Join, Bounding Box,
// bounds fields, Vector Displacement and normals recomputed by Normal Field and Set Normal.
namespace {
using namespace videowire::geometry;

int failures = 0;

void check(bool condition, const char *message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
  }
}

bool nearly(double actual, double expected) {
  return std::fabs(actual - expected) <= 1e-4;
}

bool nearlyVec(const Vec3 &actual, double x, double y, double z) {
  return nearly(actual.x, x) && nearly(actual.y, y) && nearly(actual.z, z);
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

// A contract at the format bounds, for the generator tests that run past the old
// node limits. The caps here are the 32-bit index limits, so only the format stops a count.
PortContract contractFor(CarrierKind carrier) {
  PortContract contract;
  contract.carrier = carrier;
  contract.overflow = OverflowPolicy::reject;
  contract.maxVertices = kMaximumVertices;
  contract.maxIndices = kMaximumIndices;
  contract.maxPoints = kMaximumPoints;
  contract.maxCurvePoints = kMaximumCurvePoints;
  contract.maxSplines = kMaximumSplines;
  contract.maxInstances = kMaximumInstances;
  contract.maxAttributes = kMaximumAttributes;
  return contract;
}

PortContract vectorFieldContract() {
  PortContract contract;
  contract.carrier = CarrierKind::field;
  contract.overflow = OverflowPolicy::reject;
  contract.fieldDomain = Domain::vertex;
  contract.fieldValueType = ValueType::vector;
  contract.fieldInterpolation = Interpolation::constant;
  contract.maxFieldElements = 64;
  return contract;
}

// Admits the value, then checks that its runtime transport roundtrip is byte-stable.
bool admitsAndRoundTrips(const ValueDescriptor &value, const PortContract &contract) {
  std::string error;
  auto admitted = admitValue(value, contract, error);
  if (!admitted) return false;
  const auto bytes = encodeRuntimeValue(*admitted);
  auto decoded = decodeRuntimeValue(bytes, contract, {}, {}, error);
  return decoded.has_value() && encodeRuntimeValue(*decoded) == bytes;
}

// Axis-aligned cube centered on the origin with 24 vertices and 12 triangles.
ValueDescriptor unitCube(StableId id) {
  std::string error;
  auto cube = lowerMeshGenerator(id, id, 1, OperationCode::cube, {1, 1, 1, 0}, meshContract(), error);
  check(cube.has_value(), "unit cube lowers");
  if (!cube) return ValueDescriptor{};
  return std::move(*cube);
}

// The unit cube after scale (2, -3, 0.5), a quarter turn about Z, and translation (2, 3, 4).
// The cube keeps `id` and the transform output is `id + 1`.
std::optional<ValueDescriptor> movedCube(StableId id) {
  std::string error;
  Transform transform;
  transform.translation = {2, 3, 4};
  transform.scale = {2, -3, 0.5f};
  transform.rotation = {0, 0, 0.7071067812f, 0.7071067812f};
  return lowerTransformGeometry(unitCube(id), id + 1, transform, meshContract(), error);
}

void transformMovesTheUnitCubeExactly() {
  std::string error;
  const auto source = unitCube(2001);
  const auto moved = movedCube(2001);
  check(moved.has_value(), "TRS transform lowers a unit cube");
  if (!moved) return;
  const auto &mesh = std::get<GeometryData>(moved->data);
  const auto &original = std::get<GeometryData>(source.data);
  check(mesh.positions.size() == 24 && mesh.indices == original.indices
            && mesh.vertexIds == original.vertexIds,
        "transform keeps the cube's topology and vertex identities");
  // Vertex 0 is the corner (-0.5, -0.5, -0.5). Scale gives (-1, 1.5, -0.25), the quarter
  // turn gives (-1.5, -1, -0.25), and the translation gives (0.5, 2, 3.75).
  check(nearlyVec(mesh.positions[0], 0.5, 2, 3.75),
        "transform applies scale, then rotation, then translation");
  Vec3 minimum, maximum;
  check(geometryBounds(mesh, minimum, maximum, error) && nearlyVec(minimum, 0.5, 2, 3.75)
            && nearlyVec(maximum, 3.5, 4, 4.25),
        "transformed cube bounds are (0.5, 2, 3.75) to (3.5, 4, 4.25)");
  check(moved->operations.back().code == OperationCode::transformTRS,
        "transform records one TRS operation");
  check(admitsAndRoundTrips(*moved, meshContract()), "transformed cube admits and roundtrips");
  MaterializedInstancePlan replay;
  check(validateOperationPlan(*moved, {}, &replay, error) && equal(replay.geometries.at(2002), mesh),
        "replaying the transform reproduces the materialized vertices");
}

void transformRefusesZeroOrientation() {
  std::string error;
  const auto cube = unitCube(2011);
  Transform transform;
  transform.rotation = {0, 0, 0, 0};
  check(!lowerTransformGeometry(cube, 2012, transform, meshContract(), error),
        "transform refuses a zero orientation before publication");
}

void boundingBoxIsTheAxisAlignedBoxOfTheMesh() {
  std::string error;
  const auto moved = movedCube(2021);
  check(moved.has_value(), "bounding box fixture transforms a cube");
  if (!moved) return;
  auto box = lowerBoundingBox(*moved, 2023, meshContract(), error);
  check(box.has_value(), "bounding box lowers");
  if (!box) return;
  const auto &mesh = std::get<GeometryData>(box->data);
  check(mesh.positions.size() == 8 && mesh.indices.size() == 36
            && mesh.vertexIds == std::vector<StableId>{1, 2, 3, 4, 5, 6, 7, 8},
        "bounding box is an eight-vertex, twelve-triangle box with fixed identities");
  check(nearlyVec(mesh.positions[0], 0.5, 2, 3.75) && nearlyVec(mesh.positions[6], 3.5, 4, 4.25),
        "bounding box corners 0 and 6 are the minimum and maximum of the source");
  bool onCorners = true;
  for (const auto &p : mesh.positions)
    onCorners = onCorners && (nearly(p.x, 0.5) || nearly(p.x, 3.5))
        && (nearly(p.y, 2) || nearly(p.y, 4)) && (nearly(p.z, 3.75) || nearly(p.z, 4.25));
  check(onCorners, "every bounding box vertex sits on a corner of the source bounds");
  check(box->operations.back().code == OperationCode::boundingBox
            && admitsAndRoundTrips(*box, meshContract()),
        "bounding box records its operation and admits and roundtrips");
  auto small = meshContract();
  small.maxVertices = 7;
  check(!lowerBoundingBox(*moved, 2024, small, error),
        "bounding box refuses a contract without room for eight vertices");
}

void boundsFieldsReportCenterSizeAndMaximum() {
  std::string error;
  const auto moved = movedCube(2031);
  check(moved.has_value(), "bounds field fixture transforms a cube");
  if (!moved) return;
  const auto center = lowerElementField(*moved, 2033, OperationCode::boundsCenterField,
                                        vectorFieldContract(), error);
  const auto size = lowerElementField(*moved, 2034, OperationCode::boundsSizeField,
                                      vectorFieldContract(), error);
  const auto maximum = lowerElementField(*moved, 2035, OperationCode::boundsMaximumField,
                                         vectorFieldContract(), error);
  check(center.has_value() && size.has_value() && maximum.has_value(),
        "bounds center, size and maximum fields lower for a transformed cube");
  if (!center || !size || !maximum) return;
  const auto &c = std::get<FieldData>(center->data).elements;
  const auto &s = std::get<FieldData>(size->data).elements;
  const auto &m = std::get<FieldData>(maximum->data).elements;
  check(c.size() == 1 && nearly(c[0].components[0], 2) && nearly(c[0].components[1], 3)
            && nearly(c[0].components[2], 4),
        "bounds center is (2, 3, 4)");
  check(s.size() == 1 && nearly(s[0].components[0], 3) && nearly(s[0].components[1], 2)
            && nearly(s[0].components[2], 0.5),
        "bounds size is (3, 2, 0.5)");
  check(m.size() == 1 && nearly(m[0].components[0], 3.5) && nearly(m[0].components[1], 4)
            && nearly(m[0].components[2], 4.25),
        "bounds maximum is (3.5, 4, 4.25)");
}

void joinAppendsTheSecondMeshWithOffsetIndices() {
  std::string error;
  const auto left = unitCube(2041);
  Transform shift;
  shift.translation = {3, 0, 0};
  auto right = lowerTransformGeometry(unitCube(2043), 2044, shift, meshContract(), error);
  check(right.has_value(), "second cube translates for the join");
  if (!right) return;
  const auto &a = std::get<GeometryData>(left.data);
  const auto &b = std::get<GeometryData>(right->data);
  auto joined = lowerJoinGeometry(left, *right, 2045, meshContract(), error);
  check(joined.has_value(), "join lowers two cubes with matching materials");
  if (!joined) return;
  const auto &mesh = std::get<GeometryData>(joined->data);
  check(mesh.positions.size() == 48 && mesh.indices.size() == 72,
        "join concatenates 48 vertices and 72 indices");
  bool sourceOrder = a.positions.size() == 24 && b.positions.size() == 24;
  for (std::size_t i = 0; sourceOrder && i < 24; ++i)
    sourceOrder = equal(mesh.positions[i], a.positions[i])
        && equal(mesh.positions[24 + i], b.positions[i]);
  check(sourceOrder, "join keeps the first mesh's vertices, then the second mesh's, in order");
  bool offsets = a.indices.size() == 36 && b.indices.size() == 36;
  for (std::size_t i = 0; offsets && i < 36; ++i)
    offsets = mesh.indices[i] == a.indices[i] && mesh.indices[36 + i] == b.indices[i] + 24;
  check(offsets, "join offsets the second mesh's triangle indices by the first vertex count");
  const std::set<StableId> ids(mesh.vertexIds.begin(), mesh.vertexIds.end());
  check(ids.size() == 48, "join gives every joined vertex a distinct identity");
  auto repeated = lowerJoinGeometry(left, *right, 2045, meshContract(), error);
  check(repeated.has_value() && equal(mesh, std::get<GeometryData>(repeated->data)),
        "join is deterministic for the same inputs");
  check(admitsAndRoundTrips(*joined, meshContract()), "joined cubes admit and roundtrip");
  MaterializedInstancePlan replay;
  check(validateOperationPlan(*joined, {}, &replay, error) && equal(replay.geometries.at(2045), mesh),
        "replaying the join reproduces its vertices and indices");
  auto small = meshContract();
  small.maxVertices = 47;
  check(!lowerJoinGeometry(left, *right, 2046, small, error),
        "join refuses a result beyond its vertex capacity");
}

void displacementMovesEachVertexByTheScaledField() {
  std::string error;
  // A 2 by 2 plane at z = 0, lifted to z = 2, so its maximum corner is (1, 1, 2).
  auto plane = lowerMeshGenerator(2051, 2051, 1, OperationCode::plane, {2, 2, 0, 0},
                                  meshContract(), error);
  check(plane.has_value(), "plane fixture lowers");
  if (!plane) return;
  Transform lift;
  lift.translation = {0, 0, 2};
  auto raised = lowerTransformGeometry(*plane, 2052, lift, meshContract(), error);
  check(raised.has_value(), "plane lifts to z equals two");
  if (!raised) return;
  auto maximum = lowerElementField(*raised, 2053, OperationCode::boundsMaximumField,
                                   vectorFieldContract(), error);
  check(maximum.has_value(), "bounds maximum field lowers for the lifted plane");
  if (!maximum) return;
  auto displaced = lowerVectorDisplacement(*raised, *maximum, 2054, 0.5f, meshContract(), error);
  check(displaced.has_value(), "bounds vector displaces the plane");
  if (!displaced) return;
  const auto &mesh = std::get<GeometryData>(displaced->data);
  // Each vertex moves by 0.5 times (1, 1, 2). X and Y shift by 0.5 and Z goes from 2 to 3.
  check(mesh.positions.size() == 4 && nearlyVec(mesh.positions[0], -0.5, -0.5, 3)
            && nearlyVec(mesh.positions[1], 1.5, -0.5, 3)
            && nearlyVec(mesh.positions[2], -0.5, 1.5, 3)
            && nearlyVec(mesh.positions[3], 1.5, 1.5, 3),
        "displacement adds strength times the broadcast field to every vertex");
  check(displaced->operations.back().code == OperationCode::vectorDisplacement
            && nearly(displaced->operations.back().parameters[0], 0.5),
        "displacement records its strength");
  auto box = lowerBoundingBox(*displaced, 2055, meshContract(), error);
  check(box.has_value(), "bounding box lowers for the displaced plane");
  if (box) {
    const auto &boxMesh = std::get<GeometryData>(box->data);
    check(nearlyVec(boxMesh.positions[0], -0.5, -0.5, 3) && nearlyVec(boxMesh.positions[6], 1.5, 1.5, 3),
          "bounding box of the displaced plane spans its displaced corners");
  }
}

void recomputedNormalsAreTheNormalizedFaceSums() {
  std::string error;
  // Two triangles, (0, 1, 2) and (1, 3, 2), each with area-weighted normal (0, 0, 4).
  auto plane = lowerMeshGenerator(2061, 2061, 1, OperationCode::plane, {2, 2, 0, 0},
                                  meshContract(), error);
  check(plane.has_value(), "plane fixture for normals lowers");
  if (!plane) return;
  auto normals = lowerElementField(*plane, 2062, OperationCode::normalField,
                                   vectorFieldContract(), error);
  check(normals.has_value(), "normal field lowers");
  if (!normals) return;
  // The field sums unnormalized face normals per vertex. Vertices 1 and 2 touch both triangles.
  const auto &sums = std::get<FieldData>(normals->data).elements;
  const std::array<double, 4> expectedZ{4, 8, 8, 4};
  bool sumsMatch = sums.size() == 4;
  for (std::size_t i = 0; sumsMatch && i < 4; ++i)
    sumsMatch = nearly(sums[i].components[0], 0) && nearly(sums[i].components[1], 0)
        && nearly(sums[i].components[2], expectedZ[i]);
  check(sumsMatch, "normal field sums the face normals at each vertex");
  auto unit = lowerNamedAttribute(*plane, &*normals, 2063, OperationCode::setNormal, "normal",
                                  ValueType::vector, meshContract(), error);
  check(unit.has_value() && unit->attributes.size() == 1 && unit->attributes[0].elements.size() == 4,
        "normal field feeds Set Normal");
  if (!unit) return;
  bool normalized = true;
  for (const auto &element : unit->attributes[0].elements)
    normalized = normalized && nearly(element.components[0], 0) && nearly(element.components[1], 0)
        && nearly(element.components[2], 1);
  check(normalized, "Set Normal with the default toggle makes the recomputed normals unit length");
}

// Counts past the old node limits build the exact mesh, and only a count that
// 32-bit indices can't address stops, before anything is allocated.
void meshGeneratorCountsStopOnlyAtTheirFormat() {
  const auto contract = contractFor(CarrierKind::geometry3D);
  std::string error;
  struct Fixture { OperationCode code; std::array<float, 4> parameters; std::size_t vertices, indices; };
  for (const auto &fixture : std::array<Fixture, 4>{{
      {OperationCode::sphere, {1, 128, 64, 0}, 8066, 48384},
      {OperationCode::icosphere, {1, 9, 0, 0}, 2621442, 15728640},
      {OperationCode::cylinder, {1, 2, 1024, 0}, 4098, 12288},
      {OperationCode::cylinder, {1, 2, 1024, 0.5f}, 8192, 24576}}}) {
    const auto value = lowerMeshGenerator(501, 501, 1, fixture.code, fixture.parameters, contract, error);
    check(value.has_value(), "a mesh generator lowers past the old node limits");
    if (!value) continue;
    const auto &mesh = std::get<GeometryData>(value->data);
    check(mesh.positions.size() == fixture.vertices && mesh.indices.size() == fixture.indices
              && *std::max_element(mesh.indices.begin(), mesh.indices.end()) == fixture.vertices - 1,
          "a large generated mesh has its exact topology and every index in range");
  }
  for (const auto &fixture : std::array<Fixture, 6>{{
      {OperationCode::sphere, {1, 65536, 65537, 0}, 0, 0},
      {OperationCode::sphere, {1, 4294967040.0f, 4294967040.0f, 0}, 0, 0},
      {OperationCode::sphere, {1, 4294967296.0f, 2, 0}, 0, 0},
      {OperationCode::icosphere, {1, 4294967040.0f, 0, 0}, 0, 0},
      {OperationCode::cylinder, {1, 2, 1073741824.0f, 0}, 0, 0},
      {OperationCode::cylinder, {1, 2, 536870912.0f, 0.5f}, 0, 0}}})
    check(!lowerMeshGenerator(501, 501, 1, fixture.code, fixture.parameters, contract, error),
          "a count past the 32-bit vertex index refuses before allocation");
  check(!lowerGrid(601, 601, 1, 65536, 65537, 1, contract, error),
        "a grid past the 32-bit vertex index refuses before allocation");
  const auto curves = contractFor(CarrierKind::curves3D);
  const auto circle = lowerCurveCircle(602, 602, 1, 65537, 1, curves, error);
  const auto line = lowerCurveLine(603, 603, 1, 2049, 1, curves, error);
  check(circle && std::get<CurvesData>(circle->data).points.size() == 65537
            && line && std::get<CurvesData>(line->data).points.size() == 2049,
        "curve generators lower past the old node limits");
  const auto solid = lowerMeshGenerator(604, 604, 1, OperationCode::cube, {1, 1, 1, 0}, contract, error);
  const auto scattered = lowerPointDistribution(*solid, 605, OperationCode::pointsOnFaces, 1024, 0,
                                                contractFor(CarrierKind::points3D), error);
  check(scattered && std::get<PointsData>(scattered->data).points.size() == 1024,
        "point distribution lowers past the old node limit");
}

} // namespace

// The grid stopped at 64 cells per axis until the limit came off. 72 must mesh, finer than 64,
// on the sphere's surface, without any other budget refusing it.
void sdfGridAboveTheOldLimitMeshes() {
  namespace surface = videowire::sdfsurface;
  const auto mesh = [](std::uint32_t cells, surface::Mesh& out, std::string& error) {
    surface::Request request;
    request.cells = {cells, cells, cells};
    videowire::SdfRecord sphere;
    sphere.stableId = 9007199254740001ull;
    sphere.operation = videowire::SdfOperation::sphere;
    sphere.parameterCount = 1;
    sphere.parameters[0] = 1.013;
    request.geometry.rootId = sphere.stableId;
    request.geometry.records = {sphere};
    return surface::materialize(request, 9700, 16777216, 100663296, out, error);
  };
  surface::Mesh old, finer;
  std::string error;
  check(mesh(64, old, error), error.c_str());
  check(mesh(72, finer, error), error.c_str());
  check(finer.indices.size() > old.indices.size(), "72 cells mesh finer than 64");
  check(!finer.positions.empty(), "72 cells produce vertices");
  for (const auto& position : finer.positions) {
    const auto radius = std::sqrt(double(position[0]) * position[0] + double(position[1]) * position[1]
                                  + double(position[2]) * position[2]);
    if (std::fabs(radius - 1.013) > 0.011) {
      check(false, "a 72 cell vertex lies on the sphere");
      break;
    }
  }
}

// Depth bounds real recursion and reports its refusal; the record count is only the wire's.
void sdfRecordsAndDepthLimits() {
  namespace surface = videowire::sdfsurface;
  std::uint64_t nextId = 9007199254000000ull;
  const auto sphere = [&](std::size_t index) {
    videowire::SdfRecord record;
    record.stableId = nextId++;
    record.operation = videowire::SdfOperation::sphere;
    record.parameterCount = 1;
    record.parameters[0] = 1.013f + float(index) * 1.0e-4f;
    return record;
  };
  const auto join = [&](std::uint64_t a, std::uint64_t b) {
    videowire::SdfRecord record;
    record.stableId = nextId++;
    record.operation = videowire::SdfOperation::unionOp;
    record.inputCount = 2;
    record.inputs[0] = a;
    record.inputs[1] = b;
    return record;
  };
  const auto chain = [&](std::size_t depth) {
    surface::Request request;
    request.cells = {8, 8, 8};
    auto root = sphere(0);
    request.geometry.records.push_back(root);
    for (std::size_t level = 1; level < depth; ++level) {
      const auto side = sphere(level);
      const auto next = join(root.stableId, side.stableId);
      request.geometry.records.push_back(side);
      request.geometry.records.push_back(next);
      root = next;
    }
    request.geometry.rootId = root.stableId;
    return request;
  };
  surface::Budget budget;
  std::string error;
  check(surface::inspect(chain(64), budget, error), error.c_str());
  error.clear();
  check(!surface::inspect(chain(65), budget, error), "depth 65 is refused");
  check(error.find("depth") != std::string::npos, "the depth refusal names depth");

  surface::Request wide;
  wide.cells = {8, 8, 8};
  std::vector<std::uint64_t> level;
  for (std::size_t index = 0; index < 300; ++index) {
    const auto leaf = sphere(index);
    wide.geometry.records.push_back(leaf);
    level.push_back(leaf.stableId);
  }
  while (level.size() > 1) {
    std::vector<std::uint64_t> next;
    for (std::size_t index = 0; index < level.size(); index += 2) {
      if (index + 1 == level.size()) {
        next.push_back(level[index]);
        continue;
      }
      const auto parent = join(level[index], level[index + 1]);
      wide.geometry.records.push_back(parent);
      next.push_back(parent.stableId);
    }
    level = std::move(next);
  }
  wide.geometry.rootId = level.front();
  check(wide.geometry.records.size() > 256, "the wide shape has more than the old 256 records");
  surface::Mesh mesh;
  error.clear();
  check(surface::materialize(wide, 9701, 16777216, 100663296, mesh, error), error.c_str());
  check(!mesh.indices.empty(), "the wide shape meshes");
}

int main() {
  transformMovesTheUnitCubeExactly();
  transformRefusesZeroOrientation();
  boundingBoxIsTheAxisAlignedBoxOfTheMesh();
  boundsFieldsReportCenterSizeAndMaximum();
  joinAppendsTheSecondMeshWithOffsetIndices();
  displacementMovesEachVertexByTheScaledField();
  recomputedNormalsAreTheNormalizedFaceSums();
  meshGeneratorCountsStopOnlyAtTheirFormat();
  sdfGridAboveTheOldLimitMeshes();
  sdfRecordsAndDepthLimits();
  if (failures != 0) {
    std::fprintf(stderr, "%d geometry operation check(s) failed\n", failures);
    return 1;
  }
  std::printf("geometry core operation tests passed\n");
  return 0;
}
