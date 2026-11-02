#include "../../shared/GeometryCore.h"
#include "../../shared/GeometryCoreTransport.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <set>
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
  if (carrier == CarrierKind::geometry3D) {
    contract.maxVertices = 16;
    contract.maxIndices = 48;
  } else {
    contract.maxPoints = 16;
  }
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

// Admission replays a retained mesh. A copy operation carries no geometry to replay.
ValueDescriptor retainedMeshValue(StableId id, GeometryData mesh) {
  auto value = baseValue(CarrierKind::geometry3D, id);
  auto &operation = value.operations.front();
  operation.code = OperationCode::retainedMesh;
  operation.inputStableId = value.sourceStableId;
  operation.retainedMesh = std::make_shared<RetainedMeshData>(RetainedMeshData{mesh, {}});
  value.data = std::move(mesh);
  return value;
}

// Four triangles over six vertices laid out as a 3 by 2 strip.
ValueDescriptor stripMesh() {
  GeometryData mesh;
  mesh.positions = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {0, 1, 0}, {1, 1, 0}, {2, 1, 0}};
  mesh.vertexIds = {11, 12, 13, 14, 15, 16};
  mesh.indices = {0, 1, 3, 1, 4, 3, 1, 2, 4, 2, 5, 4};
  return retainedMeshValue(501, std::move(mesh));
}

// Four points, retained as the vertices of a mesh.
ValueDescriptor pointCloud() {
  GeometryData mesh;
  mesh.positions = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {3, 0, 0}};
  mesh.vertexIds = {1, 2, 3, 4};
  mesh.indices = {0, 1, 2, 0, 2, 3};
  std::string error;
  auto points = lowerPointsFromVertices(retainedMeshValue(600, std::move(mesh)), 601, contractFor(CarrierKind::points3D), error);
  if (!points) std::fprintf(stderr, "point cloud: %s\n", error.c_str());
  return *points;
}

// Admits the lowered value, then checks the transport roundtrip and that a
// version 16 header refuses the new operation codes.
bool admitsAndRoundTrips(const ValueDescriptor &value, CarrierKind carrier) {
  std::string error;
  const auto contract = withAttributeContract(contractFor(carrier), value);
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
  if (!normalized) std::fprintf(stderr, "Set Normal: %s\n", error.c_str());
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
  zero.values = {0, 0, 0, 0, 1, 0, 0, 0};
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
  auto mesh = std::get<GeometryData>(stripMesh().data);
  mesh.positions.push_back({9, 9, 0});
  mesh.positions.push_back({8, 8, 0});
  mesh.vertexIds.push_back(21);
  mesh.vertexIds.push_back(22);
  return retainedMeshValue(501, std::move(mesh));
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
  auto preserved = lowerSetNormalValue(1.0f, error);
  if (!preserved) std::fprintf(stderr, "preserved Set Normal: %s\n", error.c_str());
  check(preserved.has_value(), "a preserved Set Normal lowers");
  if (!preserved) return;
  check(preserved->operations.back().parameters[0] == 1,
        "the preserved Set Normal stores its flag as one");
  const auto contract = withAttributeContract(contractFor(CarrierKind::geometry3D), *preserved);
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
  auto admitted = admitValue(*normalized, withAttributeContract(contract, *normalized), error);
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

// Two faces over a shared edge 0-1 with normals +z and +y, a 90 degree dihedral.
ValueDescriptor foldMesh() {
  GeometryData mesh;
  mesh.positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  mesh.vertexIds = {21, 22, 23, 24};
  mesh.indices = {0, 1, 2, 1, 0, 3};
  return retainedMeshValue(701, std::move(mesh));
}

// Eight vertices shared by twelve outward triangles.
ValueDescriptor sharedCube() {
  GeometryData mesh;
  mesh.positions = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
  mesh.indices = {0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4,
                  3, 7, 6, 3, 6, 2, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5};
  return retainedMeshValue(702, std::move(mesh));
}

// An open cylinder with split seam columns, the shape of a coarse generator mesh.
ValueDescriptor splitCylinder(std::uint32_t segments) {
  GeometryData mesh;
  for (std::uint32_t s = 0; s < segments; ++s) {
    const auto a = 6.283185307179586 * s / segments;
    const auto b = 6.283185307179586 * ((s + 1) % segments) / segments;
    const auto ax = static_cast<float>(std::sin(a)), az = static_cast<float>(std::cos(a));
    const auto bx = static_cast<float>(std::sin(b)), bz = static_cast<float>(std::cos(b));
    const auto base = static_cast<std::uint32_t>(mesh.positions.size());
    mesh.positions.insert(mesh.positions.end(), {{ax, 0, az}, {bx, 0, bz}, {bx, 2, bz}, {ax, 2, az}});
    for (std::uint32_t k : {0u, 1u, 2u, 0u, 2u, 3u}) mesh.indices.push_back(base + k);
  }
  return retainedMeshValue(703, std::move(mesh));
}

std::optional<ValueDescriptor> shade(const ValueDescriptor &value, float mode, float angle = 30,
                                     std::vector<std::array<std::uint32_t, 2>> sharp = {}) {
  std::string error;
  auto shaded = lowerShadeEdit(value, 950, mode, angle, std::move(sharp), error);
  if (!shaded) std::fprintf(stderr, "shade error: %s\n", error.c_str());
  return shaded;
}

const GeometryData &meshOf(const ValueDescriptor &value) { return std::get<GeometryData>(value.data); }

const AttributeData *normalsOf(const ValueDescriptor &value) {
  for (const auto &attribute : value.attributes)
    if (attribute.descriptor.name == "normal") return &attribute;
  return nullptr;
}

bool sameNormal(const FieldElement &a, const FieldElement &b, double tolerance = 1.0e-6) {
  for (std::size_t axis = 0; axis < 3; ++axis)
    if (std::abs(static_cast<double>(a.components[axis]) - b.components[axis]) > tolerance) return false;
  return true;
}

bool nearVector(const FieldElement &a, double x, double y, double z) {
  return std::abs(a.components[0] - x) < 1.0e-5 && std::abs(a.components[1] - y) < 1.0e-5
      && std::abs(a.components[2] - z) < 1.0e-5;
}

void shadeSmoothBlendsAcrossTheEdge() {
  const auto shaded = shade(foldMesh(), kShadeSmooth);
  check(shaded.has_value(), "Shade Smooth lowers");
  if (!shaded) return;
  const auto *normals = normalsOf(*shaded);
  check(meshOf(*shaded).positions.size() == 4 && normals != nullptr && normals->elements.size() == 4,
        "Shade Smooth keeps a shared edge welded");
  if (normals == nullptr || normals->elements.size() != 4) return;
  const auto half = std::sqrt(0.5);
  check(nearVector(normals->elements[0], 0, half, half) && nearVector(normals->elements[1], 0, half, half),
        "the edge normal blends both faces");
  check(nearVector(normals->elements[2], 0, 0, 1) && nearVector(normals->elements[3], 0, 1, 0),
        "a vertex on one face takes that face's normal");
  check(normals->descriptor.stableId == namedAttributeIdentity("normal", Domain::vertex, ValueType::vector)
            && normals->descriptor.interpolation == Interpolation::nearest,
        "the normal attribute has the Set Normal descriptor");
  check(admitsAndRoundTrips(*shaded, CarrierKind::geometry3D), "a shaded mesh admits and roundtrips");
}

void shadeFlatIsConstantPerFace() {
  const auto shaded = shade(foldMesh(), kShadeFlat);
  check(shaded.has_value(), "Shade Flat lowers");
  if (!shaded) return;
  const auto &mesh = meshOf(*shaded);
  const auto *normals = normalsOf(*shaded);
  check(mesh.positions.size() == 6 && normals != nullptr && normals->elements.size() == 6,
        "Shade Flat gives each face corner its own vertex");
  if (normals == nullptr || normals->elements.size() != 6) return;
  for (std::size_t corner = 0; corner < 3; ++corner) {
    check(nearVector(normals->elements[mesh.indices[corner]], 0, 0, 1), "every corner of the first face is +z");
    check(nearVector(normals->elements[mesh.indices[corner + 3]], 0, 1, 0), "every corner of the second face is +y");
  }
  check(admitsAndRoundTrips(*shaded, CarrierKind::geometry3D), "a flat mesh admits and roundtrips");
}

void smoothByAngleSplitsExactlyTheEdgesOverTheAngle() {
  const auto vertexCount = [](const ValueDescriptor &value, float angle) {
    const auto shaded = shade(value, kShadeByAngle, angle);
    return shaded ? meshOf(*shaded).positions.size() : std::size_t{0};
  };
  check(vertexCount(foldMesh(), 89) == 6, "a 90 degree edge splits under an 89 degree threshold");
  check(vertexCount(foldMesh(), 91) == 4, "a 90 degree edge stays smooth under a 91 degree threshold");
  check(vertexCount(foldMesh(), 90) == 4, "an edge exactly at the threshold stays smooth");
  check(vertexCount(sharedCube(), 30) == 24, "every cube edge splits under 30 degrees and coplanar faces stay joined");
  check(vertexCount(sharedCube(), 100) == 8, "no cube edge splits under 100 degrees");
}

void markedSharpEdgesSplitInAnyMode() {
  const auto marked = shade(foldMesh(), kShadeSmooth, 30, {{1, 0}});
  check(marked && meshOf(*marked).positions.size() == 6, "a marked edge splits under Shade Smooth, in either order");
  const auto byAngle = shade(foldMesh(), kShadeByAngle, 120, {{0, 1}});
  check(byAngle && meshOf(*byAngle).positions.size() == 6, "a marked edge splits under the angle threshold");
  const auto outside = shade(foldMesh(), kShadeSmooth, 30, {{0, 99}});
  check(outside && meshOf(*outside).positions.size() == 4, "a mark outside the mesh marks nothing");
  if (!marked) return;
  check(marked->operations.back().sharpEdges.size() == 1, "the operation carries the marks");
  check(admitsAndRoundTrips(*marked, CarrierKind::geometry3D), "marked edges survive admission and transport");
  const auto &ids = meshOf(*marked).vertexIds;
  const std::set<StableId> unique(ids.begin(), ids.end());
  check(ids.size() == 6 && unique.size() == 6, "split vertices get new stable identities");
  std::string error;
  auto wrong = foldMesh();
  wrong.operations.back().sharpEdges = {{0, 1}};
  check(!admitValue(wrong, contractFor(CarrierKind::geometry3D), error).has_value(),
        "sharp edges on another operation are refused");
}

void shadingKeepsAttributesAlignedAndReplacesImportedNormals() {
  auto value = foldMesh();
  AttributeData imported;
  imported.descriptor = {namedAttributeIdentity("normal", Domain::vertex, ValueType::vector), "normal",
                         ValueType::vector, Domain::vertex, Interpolation::nearest};
  imported.elements.resize(4);
  for (auto &element : imported.elements) element.components = {1, 0, 0, 0};
  AttributeData weight;
  weight.descriptor = {0, "weight", ValueType::floatValue, Domain::vertex, Interpolation::linear};
  weight.elements.resize(4);
  for (std::size_t v = 0; v < 4; ++v) weight.elements[v].components = {static_cast<float>(v), 0, 0, 0};
  AttributeData faceTag;
  faceTag.descriptor = {0, "tag", ValueType::floatValue, Domain::face, Interpolation::nearest};
  faceTag.elements.resize(2);
  value.attributes = {imported, weight, faceTag};

  check(normalsOf(value)->elements[0].components[0] == 1, "an imported normal stays without a shade operation");
  const auto shaded = shade(value, kShadeFlat);
  check(shaded.has_value(), "imported normals can be overridden");
  if (!shaded) return;
  std::size_t normalAttributes = 0;
  for (const auto &attribute : shaded->attributes) normalAttributes += attribute.descriptor.name == "normal" ? 1u : 0u;
  check(normalAttributes == 1 && nearVector(normalsOf(*shaded)->elements[meshOf(*shaded).indices[0]], 0, 0, 1),
        "the override replaces the imported normals");
  const auto &mesh = meshOf(*shaded);
  const auto &source = meshOf(value);
  for (const auto &attribute : shaded->attributes) {
    if (attribute.descriptor.name == "weight") {
      check(attribute.elements.size() == mesh.positions.size(), "a vertex attribute follows the split");
      for (std::size_t corner = 0; corner < mesh.indices.size(); ++corner)
        check(attribute.elements[mesh.indices[corner]].components[0] == static_cast<float>(source.indices[corner]),
              "a split vertex keeps its source's attribute value");
    }
    if (attribute.descriptor.name == "tag") check(attribute.elements.size() == 2, "a face attribute is unchanged");
  }
}

void shadeNeedsItsTransportVersion() {
  std::string error;
  const auto shaded = shade(foldMesh(), kShadeByAngle, 45, {{0, 1}});
  check(shaded.has_value(), "the transport fixture lowers");
  if (!shaded) return;
  // Shading adds the normal attribute, so the shaded value's contract declares it.
  const auto contract = withAttributeContract(contractFor(CarrierKind::geometry3D), *shaded);
  auto admitted = admitValue(*shaded, contract, error);
  check(admitted.has_value(), "the transport fixture admits");
  if (!admitted) return;
  const auto bytes = encodeRuntimeValue(*admitted);
  check(headerVersion(bytes) == kShadeTransportVersion, "a shaded value is written at the shade version");
  auto reopened = decodeRuntimeValue(bytes, contract, {}, {}, error);
  check(reopened && reopened->descriptor().operations.back().sharpEdges == shaded->operations.back().sharpEdges,
        "sharp edges roundtrip");
  auto forged = bytes;
  forged[4] = static_cast<std::uint8_t>(kShadeTransportVersion - 1);
  check(!decodeRuntimeValue(forged, contract, {}, {}, error).has_value(),
        "a shade operation at the previous version is refused");
  auto plain = admitValue(foldMesh(), contractFor(CarrierKind::geometry3D), error);
  check(plain && headerVersion(encodeRuntimeValue(*plain)) < kShadeTransportVersion,
        "a value without shading keeps its older version and bytes");
}

// A flat fan around vertex 0, closed by a last face from vertex 4 to vertex 1 through
// vertex 0. Raising vertex 0 by height turns that face into a sliver facing sideways.
GeometryData fanWithSliver(float height) {
  GeometryData mesh;
  mesh.positions = {{0.5f, 0, height}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 0}};
  mesh.indices = {0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1};
  return mesh;
}

// A face with no area has no direction, so it adds nothing to a smooth normal. A
// sliver one float step from that has a direction made of rounding, so neither may it.
void degenerateFacesAddNothingToASmoothNormal() {
  const auto normalAtZero = [](GeometryData mesh, FieldElement &normal) {
    const auto shaded = shade(retainedMeshValue(704, std::move(mesh)), kShadeSmooth);
    const auto *normals = shaded ? normalsOf(*shaded) : nullptr;
    if (normals == nullptr || normals->elements.empty()) return false;
    normal = normals->elements[0];
    return true;
  };
  FieldElement normal;
  check(normalAtZero(fanWithSliver(0), normal) && nearVector(normal, 0, 0, 1),
        "a zero-area face leaves the normal on the surface");
  check(normalAtZero(fanWithSliver(1.0e-7f), normal) && nearVector(normal, 0, 0, 1),
        "a rounding sliver leaves the normal on the surface");

  auto repeated = fanWithSliver(0);
  repeated.indices.insert(repeated.indices.end(), {0, 0, 2, 2, 2, 2});
  check(normalAtZero(repeated, normal) && nearVector(normal, 0, 0, 1),
        "faces with repeated vertices leave the normal on the surface");

  // A vertex with only a degenerate face takes the normal of its position.
  auto stray = fanWithSliver(0);
  stray.positions.push_back(stray.positions[2]);
  stray.indices.insert(stray.indices.end(), {5, 5, 5});
  const auto shadedStray = shade(retainedMeshValue(705, std::move(stray)), kShadeSmooth);
  const auto *strayNormals = shadedStray ? normalsOf(*shadedStray) : nullptr;
  check(strayNormals != nullptr && strayNormals->elements.size() > 5 && nearVector(strayNormals->elements[5], 0, 0, 1),
        "a vertex with only a degenerate face takes the welded normal");

  // With no surface at all there is nothing to read, and the normal is still a unit vector.
  GeometryData line;
  line.positions = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}};
  line.indices = {0, 1, 2};
  check(normalAtZero(std::move(line), normal) && nearVector(normal, 0, 0, 1),
        "a mesh of only zero-area faces falls back to a unit normal");
}

// Measurable form of "no faceted banding" at the normal level: on an 8 segment
// cylinder with split seams, all normals at one position agree within 1e-6, and
// each equals the radial direction within 1e-5.
void coarseCylinderHasOneNormalPerPosition() {
  const auto cylinder = splitCylinder(8);
  const auto shaded = shade(cylinder, kShadeSmooth);
  check(shaded.has_value(), "the coarse cylinder shades");
  if (!shaded) return;
  const auto &mesh = meshOf(*shaded);
  const auto *normals = normalsOf(*shaded);
  check(normals != nullptr && mesh.positions.size() == meshOf(cylinder).positions.size(),
        "split seams weld without adding vertices");
  if (normals == nullptr) return;
  for (std::size_t a = 0; a < mesh.positions.size(); ++a) {
    const auto &p = mesh.positions[a];
    const auto length = std::sqrt(double(p.x) * p.x + double(p.z) * p.z);
    check(nearVector(normals->elements[a], p.x / length, 0, p.z / length), "a smooth cylinder normal is radial");
    for (std::size_t b = a + 1; b < mesh.positions.size(); ++b) {
      const auto &q = mesh.positions[b];
      if (p.x == q.x && p.y == q.y && p.z == q.z)
        check(sameNormal(normals->elements[a], normals->elements[b]), "normals agree across a split seam");
    }
  }
  const auto flat = shade(cylinder, kShadeFlat);
  check(flat && meshOf(*flat).positions.size() == mesh.indices.size(), "flat shades every coarse face alone");
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
  shadeSmoothBlendsAcrossTheEdge();
  shadeFlatIsConstantPerFace();
  smoothByAngleSplitsExactlyTheEdgesOverTheAngle();
  markedSharpEdgesSplitInAnyMode();
  shadingKeepsAttributesAlignedAndReplacesImportedNormals();
  shadeNeedsItsTransportVersion();
  coarseCylinderHasOneNormalPerPosition();
  degenerateFacesAddNothingToASmoothNormal();
  if (failures != 0) {
    std::fprintf(stderr, "%d geometry cull check(s) failed\n", failures);
    return 1;
  }
  std::printf("geometry core cull tests passed\n");
  return 0;
}
