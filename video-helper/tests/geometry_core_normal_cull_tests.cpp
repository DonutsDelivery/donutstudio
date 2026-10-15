#include "../../shared/GeometryCore.h"
#include "../../shared/GeometryCoreTransport.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <variant>
#include <vector>

// Cull Geometry, Cull Points and the Set Normal toggle: range selection, index and
// attribute compaction, normalization, admission replay and transport version gating.
namespace {
using namespace videowire::geometry;

int failures = 0;

void check(bool condition, const char *message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
  }
}

PortContract contractFor(CarrierKind carrier) {
  PortContract contract;
  contract.carrier = carrier;
  contract.overflow = OverflowPolicy::reject;
  contract.maxVertices = 16;
  contract.maxIndices = 48;
  contract.maxPoints = 16;
  contract.maxAttributes = 32;
  return contract;
}

// Retained through a copy operation, as the admission tests do.
ValueDescriptor baseValue(CarrierKind carrier, StableId id) {
  ValueDescriptor value;
  value.carrier = carrier;
  value.stableId = id;
  value.sourceStableId = 77;
  value.sourceRevision = 9;
  value.operations = {{id, OperationCode::copy, 0, id, {0, 0, 0, 0}}};
  value.dispatchCount = 1;
  return value;
}

// Four triangles over six vertices laid out as a 3 by 2 strip.
ValueDescriptor stripMesh() {
  auto value = baseValue(CarrierKind::geometry3D, 501);
  GeometryData mesh;
  mesh.positions = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {0, 1, 0}, {1, 1, 0}, {2, 1, 0}};
  mesh.vertexIds = {11, 12, 13, 14, 15, 16};
  mesh.indices = {0, 1, 3, 1, 4, 3, 1, 2, 4, 2, 5, 4};
  value.data = std::move(mesh);
  return value;
}

ValueDescriptor pointCloud() {
  auto value = baseValue(CarrierKind::points3D, 601);
  PointsData points;
  points.points = {{{0, 0, 0}, 1, 1, {0, 0, 0, 1}}, {{1, 0, 0}, 2, 1, {0, 0, 0, 1}},
                   {{2, 0, 0}, 3, 1, {0, 0, 0, 1}}, {{3, 0, 0}, 4, 1, {0, 0, 0, 1}}};
  value.data = std::move(points);
  return value;
}

// Admits the lowered value, then checks the transport roundtrip and that a
// version 16 header refuses the new operation codes.
bool admitsAndRoundTrips(const ValueDescriptor &value, CarrierKind carrier) {
  std::string error;
  const auto contract = contractFor(carrier);
  auto admitted = admitValue(value, contract, error);
  if (!admitted) {
    std::fprintf(stderr, "admission error: %s\n", error.c_str());
    return false;
  }
  const auto bytes = encodeRuntimeValue(*admitted);
  auto reopened = decodeRuntimeValue(bytes, contract, {}, {}, error);
  if (!reopened || encodeRuntimeValue(*reopened) != bytes) return false;
  auto forged = bytes;
  forged[4] = 16;
  return !decodeRuntimeValue(forged, contract, {}, {}, error);
}

std::optional<ValueDescriptor> lowerCull(const ValueDescriptor &input, OperationCode code,
                                         const std::array<float, 4> &range, std::string &error) {
  return lowerCullEdit(input, 900, code, range, contractFor(input.carrier), error);
}

void meshCullKeepsOnlyReferencedVertices() {
  std::string error;
  const auto mesh = stripMesh();
  // Remove faces 0 to 2. Face 3 references vertices 2, 4 and 5, which renumber to 0, 1 and 2.
  auto culled = lowerCull(mesh, OperationCode::cullGeometry, {0, 2, 1, 0}, error);
  check(culled.has_value(), "mesh cull keeps a nonempty selection");
  if (!culled) return;
  const auto &geometry = std::get<GeometryData>(culled->data);
  check(geometry.indices == std::vector<std::uint32_t>({0, 2, 1}),
        "mesh cull renumbers the kept triangle");
  check(geometry.positions.size() == 3 && geometry.positions[0].x == 2
            && geometry.positions[1].x == 1 && geometry.positions[1].y == 1
            && geometry.positions[2].x == 2 && geometry.positions[2].y == 1,
        "mesh cull keeps only referenced vertices in their original order");
  check(geometry.vertexIds == std::vector<StableId>({13, 15, 16}),
        "mesh cull keeps vertex identities with their positions");
  check(culled->operations.size() == 2 && culled->dispatchCount == 2,
        "mesh cull records one operation and one dispatch");
  check(admitsAndRoundTrips(*culled, CarrierKind::geometry3D),
        "mesh cull admits, roundtrips and refuses version 16");
}

void meshCullStrideRemovesEveryOtherFace() {
  std::string error;
  auto culled = lowerCull(stripMesh(), OperationCode::cullGeometry, {0, 3, 2, 0}, error);
  check(culled.has_value(), "strided mesh cull succeeds");
  if (!culled) return;
  const auto &geometry = std::get<GeometryData>(culled->data);
  // Faces 1 and 3 remain. Vertex 0 is unreferenced and dropped.
  check(geometry.indices == std::vector<std::uint32_t>({0, 3, 2, 1, 4, 3}),
        "strided mesh cull keeps faces 1 and 3 with renumbered indices");
  check(geometry.positions.size() == 5 && geometry.vertexIds.size() == 5,
        "strided mesh cull keeps five vertices and their identities");
  check(admitsAndRoundTrips(*culled, CarrierKind::geometry3D),
        "strided mesh cull admits and roundtrips");
}

void meshCullKeepSelectedKeepsOnlyTheRange() {
  std::string error;
  auto culled = lowerCull(stripMesh(), OperationCode::cullGeometry, {0, 0, 1, 1}, error);
  check(culled.has_value(), "keep-selected mesh cull succeeds");
  if (!culled) return;
  const auto &geometry = std::get<GeometryData>(culled->data);
  check(geometry.indices == std::vector<std::uint32_t>({0, 1, 2}) && geometry.positions.size() == 3,
        "keep-selected mesh cull keeps only face 0");
  check(admitsAndRoundTrips(*culled, CarrierKind::geometry3D),
        "keep-selected mesh cull admits and roundtrips");
}

void emptyRangeCullsNothing() {
  std::string error;
  const auto mesh = stripMesh();
  auto culled = lowerCull(mesh, OperationCode::cullGeometry, {10, 20, 1, 0}, error);
  check(culled.has_value(), "a range past the last face is an empty selection");
  if (!culled) return;
  const auto &geometry = std::get<GeometryData>(culled->data);
  check(geometry.indices == std::get<GeometryData>(mesh.data).indices
            && geometry.positions.size() == 6,
        "an empty selection keeps every face and vertex");
  check(admitsAndRoundTrips(*culled, CarrierKind::geometry3D),
        "an empty selection admits and roundtrips");
}

void fullRangeRefusesEmptyResult() {
  std::string error;
  check(!lowerCull(stripMesh(), OperationCode::cullGeometry, {0, 3, 1, 0}, error).has_value()
            && error.find("no renderable faces") != std::string::npos,
        "culling every face is refused by name");
  error.clear();
  check(!lowerCull(pointCloud(), OperationCode::cullPoints, {0, 3, 1, 0}, error).has_value()
            && error.find("leaves no points") != std::string::npos,
        "culling every point is refused by name");
}

void malformedRangesAreRefused() {
  const std::array<std::array<float, 4>, 4> malformed{{
      {2, 1, 1, 0},   // last before first
      {0, 3, 0, 0},   // stride below one
      {0, 3, 1, 2},   // keep flag other than 0 or 1
      {0.5f, 3, 1, 0} // fractional first index
  }};
  for (const auto &range : malformed) {
    std::string error;
    check(!lowerCull(stripMesh(), OperationCode::cullGeometry, range, error).has_value()
              && error.find("Cull requires") != std::string::npos,
          "malformed mesh cull range is refused by name");
    error.clear();
    check(!lowerCull(pointCloud(), OperationCode::cullPoints, range, error).has_value()
              && error.find("Cull requires") != std::string::npos,
          "malformed point cull range is refused by name");
  }
}

void pointCullRemovesTheRange() {
  std::string error;
  auto culled = lowerCull(pointCloud(), OperationCode::cullPoints, {1, 2, 1, 0}, error);
  check(culled.has_value(), "point cull succeeds");
  if (!culled) return;
  const auto &points = std::get<PointsData>(culled->data).points;
  check(points.size() == 2 && points[0].stableId == 1 && points[1].stableId == 4,
        "point cull removes points 2 and 3 and keeps identities");
  check(points.size() == 2 && points[1].position.x == 3,
        "point cull keeps the positions of surviving points");
  check(admitsAndRoundTrips(*culled, CarrierKind::points3D),
        "point cull admits and roundtrips");

  auto emptied = lowerCull(pointCloud(), OperationCode::cullPoints, {10, 20, 1, 0}, error);
  check(emptied.has_value() && std::get<PointsData>(emptied->data).points.size() == 4,
        "an empty point selection keeps every point");

  auto kept = lowerCull(pointCloud(), OperationCode::cullPoints, {1, 1, 1, 1}, error);
  check(kept.has_value() && std::get<PointsData>(kept->data).points.size() == 1
            && std::get<PointsData>(kept->data).points[0].stableId == 2,
        "keep-selected point cull keeps only the ranged point");
}

// Set Normal writes the vertex "normal" attribute from a vector field. Its
// normalize toggle maps to preserveLength = 0, which is the stored default.
void setNormalNormalizesByDefaultAndHonorsTheToggle() {
  std::string error;
  PortContract fieldContract;
  fieldContract.carrier = CarrierKind::field;
  fieldContract.fieldDomain = Domain::vertex;
  fieldContract.fieldValueType = ValueType::vector;
  fieldContract.fieldInterpolation = Interpolation::constant;
  fieldContract.maxFieldElements = 64;
  fieldContract.maxAttributes = 32;
  FieldOperationSettings settings;
  settings.domain = Domain::vertex;
  settings.valueType = ValueType::vector;
  settings.values = {0, 0, 2, 0, 1, 0, 0, 0};
  auto direction = lowerConstructedField(910, OperationCode::fieldConstant, settings, nullptr,
                                         nullptr, nullptr, fieldContract, error);
  check(direction.has_value(), "a constant vector field lowers for Set Normal");
  if (!direction) return;

  const auto mesh = stripMesh();
  const auto meshContract = contractFor(CarrierKind::geometry3D);
  auto normalized = lowerNamedAttribute(mesh, &*direction, 911, OperationCode::setNormal, "normal",
                                        ValueType::vector, meshContract, error);
  check(normalized.has_value(), "Set Normal with the default toggle lowers");
  if (normalized) {
    check(normalized->attributes.size() == 1 && normalized->attributes[0].elements.size() == 6
              && normalized->attributes[0].elements[0].components[2] == 1
              && normalized->attributes[0].elements[5].components[2] == 1,
          "Set Normal normalizes each stored vector by default");
    check(normalized->operations.back().parameters[0] == 0,
          "the default toggle stores the normalize flag as zero");
  }

  auto authored = lowerNamedAttribute(mesh, &*direction, 912, OperationCode::setNormal, "normal",
                                      ValueType::vector, meshContract, error, AdmissionContext{}, 1.0f);
  check(authored.has_value() && authored->attributes.size() == 1
            && authored->attributes[0].elements[0].components[2] == 2,
        "turning the toggle off keeps the authored length");

  FieldOperationSettings zero = settings;
  zero.values = {};
  auto empty = lowerConstructedField(913, OperationCode::fieldConstant, zero, nullptr, nullptr,
                                     nullptr, fieldContract, error);
  error.clear();
  check(empty.has_value()
            && !lowerNamedAttribute(mesh, &*empty, 914, OperationCode::setNormal, "normal",
                                    ValueType::vector, meshContract, error).has_value()
            && error.find("nonzero") != std::string::npos,
        "normalizing a zero vector is refused by name");
  error.clear();
  check(lowerNamedAttribute(mesh, &*empty, 915, OperationCode::setNormal, "normal",
                            ValueType::vector, meshContract, error, AdmissionContext{}, 1.0f).has_value(),
        "a zero vector is accepted when the toggle is off");
}

// The stripMesh fixture plus two vertices that no face references.
ValueDescriptor stripMeshWithStrayVertices() {
  auto value = stripMesh();
  auto &mesh = std::get<GeometryData>(value.data);
  mesh.positions.push_back({9, 9, 0});
  mesh.positions.push_back({8, 8, 0});
  mesh.vertexIds.push_back(21);
  mesh.vertexIds.push_back(22);
  return value;
}

void emptyRangeKeepsStrayVertices() {
  std::string error;
  const auto mesh = stripMeshWithStrayVertices();
  const auto &source = std::get<GeometryData>(mesh.data);
  auto culled = lowerCull(mesh, OperationCode::cullGeometry, {10, 20, 1, 0}, error);
  check(culled.has_value(), "an empty mesh selection lowers");
  if (!culled) return;
  const auto &geometry = std::get<GeometryData>(culled->data);
  check(geometry.positions.size() == 8 && geometry.positions[6].x == 9
            && geometry.positions[7].x == 8,
        "an empty mesh selection keeps the unreferenced vertices and their positions");
  check(geometry.vertexIds == source.vertexIds && geometry.indices == source.indices,
        "an empty mesh selection keeps every vertex identity and index");
  check(admitsAndRoundTrips(*culled, CarrierKind::geometry3D),
        "an empty mesh selection with stray vertices admits and roundtrips");

  // A cull that removes faces still drops the vertices no kept face references.
  auto dropped = lowerCull(mesh, OperationCode::cullGeometry, {0, 2, 1, 0}, error);
  check(dropped.has_value() && std::get<GeometryData>(dropped->data).positions.size() == 3,
        "a mesh cull that removes faces still drops unreferenced vertices");

  error.clear();
  ValueDescriptor empty = baseValue(CarrierKind::geometry3D, 502);
  empty.data = GeometryData{};
  check(!lowerCull(empty, OperationCode::cullGeometry, {0, 3, 1, 0}, error).has_value()
            && error.find("Cull Geometry needs a non-empty mesh") != std::string::npos,
        "an empty mesh is refused by name");
}

// Lowers a Set Normal with the given preserve-length flag on the strip mesh.
std::optional<ValueDescriptor> lowerSetNormalValue(float preserveLength, std::string &error) {
  PortContract fieldContract;
  fieldContract.carrier = CarrierKind::field;
  fieldContract.fieldDomain = Domain::vertex;
  fieldContract.fieldValueType = ValueType::vector;
  fieldContract.fieldInterpolation = Interpolation::constant;
  fieldContract.maxFieldElements = 64;
  fieldContract.maxAttributes = 32;
  FieldOperationSettings settings;
  settings.domain = Domain::vertex;
  settings.valueType = ValueType::vector;
  settings.values = {0, 0, 2, 0, 1, 0, 0, 0};
  auto direction = lowerConstructedField(920, OperationCode::fieldConstant, settings, nullptr,
                                         nullptr, nullptr, fieldContract, error);
  if (!direction) return std::nullopt;
  return lowerNamedAttribute(stripMesh(), &*direction, 921, OperationCode::setNormal, "normal",
                             ValueType::vector, contractFor(CarrierKind::geometry3D), error,
                             AdmissionContext{}, preserveLength);
}

// Transport version from the little-endian header that follows the four magic bytes.
std::uint16_t headerVersion(const std::vector<std::uint8_t> &bytes) {
  return bytes.size() < 6 ? 0 : static_cast<std::uint16_t>(bytes[4] | (bytes[5] << 8));
}

void preservedSetNormalNeedsVersionSeventeen() {
  std::string error;
  const auto contract = contractFor(CarrierKind::geometry3D);
  auto preserved = lowerSetNormalValue(1.0f, error);
  check(preserved.has_value(), "a preserved Set Normal lowers");
  if (!preserved) return;
  check(preserved->operations.back().parameters[0] == 1,
        "the preserved Set Normal stores its flag as one");
  auto admitted = admitValue(*preserved, contract, error);
  check(admitted.has_value(), "a preserved Set Normal admits");
  if (!admitted) {
    std::fprintf(stderr, "admission error: %s\n", error.c_str());
    return;
  }
  const auto bytes = encodeRuntimeValue(*admitted);
  check(headerVersion(bytes) == kSetNormalFlagTransportVersion && kSetNormalFlagTransportVersion < kTransportVersion,
        "a preserved Set Normal emits transport version 17");
  auto reopened = decodeRuntimeValue(bytes, contract, {}, {}, error);
  check(reopened.has_value() && encodeRuntimeValue(*reopened) == bytes,
        "a preserved Set Normal roundtrips");

  // A version 16 header cannot carry the flag and is refused by name.
  auto forged = bytes;
  forged[4] = 16;
  forged[5] = 0;
  error.clear();
  check(!decodeRuntimeValue(forged, contract, {}, {}, error).has_value()
            && error.find("transport version 17") != std::string::npos,
        "a version 16 header refuses a preserved Set Normal by name");
}

void defaultSetNormalKeepsTheOldVersion() {
  std::string error;
  const auto contract = contractFor(CarrierKind::geometry3D);
  auto normalized = lowerSetNormalValue(0.0f, error);
  check(normalized.has_value(), "a normalizing Set Normal lowers");
  if (!normalized) return;
  auto admitted = admitValue(*normalized, contract, error);
  check(admitted.has_value(), "a normalizing Set Normal admits");
  if (!admitted) return;
  const auto bytes = encodeRuntimeValue(*admitted);
  // Without a cull or a preserved normal, the encoder keeps the version it used before the flag.
  check(headerVersion(bytes) == kNonTextTransportVersion && headerVersion(bytes) != kTransportVersion,
        "a normalizing Set Normal without a cull keeps the pre-17 transport version");
}

void invalidSetNormalFlagIsRefused() {
  std::string error;
  check(!lowerSetNormalValue(2.0f, error).has_value()
            && error.find("normalize flag must be 0 or 1") != std::string::npos,
        "a Set Normal flag other than 0 or 1 is refused by name");
}

void attributesFollowTheCull() {
  std::string error;
  auto mesh = stripMesh();
  AttributeData vertexValues{AttributeDescriptor{0, "value", ValueType::floatValue, Domain::vertex,
                                                 Interpolation::linear}, {}};
  for (int i = 0; i < 6; ++i) {
    FieldElement element;
    element.components[0] = i;
    vertexValues.elements.push_back(element);
  }
  AttributeData faceValues{AttributeDescriptor{0, "face", ValueType::floatValue, Domain::face,
                                               Interpolation::nearest}, {}};
  for (int i = 0; i < 4; ++i) {
    FieldElement element;
    element.components[0] = 10 * i;
    faceValues.elements.push_back(element);
  }
  mesh.attributes = {vertexValues, faceValues};
  auto culled = lowerCull(mesh, OperationCode::cullGeometry, {0, 2, 1, 0}, error);
  check(culled.has_value(), "mesh cull with attributes succeeds");
  if (!culled) return;
  check(culled->attributes.size() == 2 && culled->attributes[0].elements.size() == 3
            && culled->attributes[0].elements[0].components[0] == 2
            && culled->attributes[0].elements[2].components[0] == 5,
        "vertex attributes follow the kept vertices");
  check(culled->attributes[1].elements.size() == 1
            && culled->attributes[1].elements[0].components[0] == 30,
        "face attributes follow the kept faces");

  auto unsupported = stripMesh();
  AttributeData splineValues{AttributeDescriptor{0, "spline", ValueType::floatValue, Domain::spline,
                                                 Interpolation::constant}, {}};
  splineValues.elements.resize(6);
  unsupported.attributes = {splineValues};
  check(!lowerCull(unsupported, OperationCode::cullGeometry, {0, 2, 1, 0}, error).has_value(),
        "attributes in a domain the cull cannot renumber are refused");
}
} // namespace

int main() {
  meshCullKeepsOnlyReferencedVertices();
  meshCullStrideRemovesEveryOtherFace();
  meshCullKeepSelectedKeepsOnlyTheRange();
  emptyRangeCullsNothing();
  fullRangeRefusesEmptyResult();
  malformedRangesAreRefused();
  pointCullRemovesTheRange();
  attributesFollowTheCull();
  setNormalNormalizesByDefaultAndHonorsTheToggle();
  emptyRangeKeepsStrayVertices();
  preservedSetNormalNeedsVersionSeventeen();
  defaultSetNormalKeepsTheOldVersion();
  invalidSetNormalFlagIsRefused();
  if (failures != 0) {
    std::fprintf(stderr, "%d geometry cull check(s) failed\n", failures);
    return 1;
  }
  std::printf("geometry core cull tests passed\n");
  return 0;
}
