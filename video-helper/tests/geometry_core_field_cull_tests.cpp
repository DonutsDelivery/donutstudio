#include "../../shared/GeometryCore.h"
#include "../../shared/GeometryCoreTransport.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

// Exact values for the position field of a transformed mesh, the normal field of a
// cube, and culls of mesh triangles, points and instances. Each cull's surviving
// stable identities are checked, then the cull is replayed from its retained graph.
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

// Field elements are stored as (x, y, z, 0) for position and normal fields.
bool nearlyElement(const FieldElement &element, double x, double y, double z) {
  return nearly(element.components[0], x) && nearly(element.components[1], y)
      && nearly(element.components[2], z) && nearly(element.components[3], 0);
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
  contract.maxAttributes = 32;
  return contract;
}

PortContract instancesContract() {
  PortContract contract;
  contract.carrier = CarrierKind::instances3D;
  contract.overflow = OverflowPolicy::reject;
  contract.maxInstances = 64;
  contract.maxAttributes = 32;
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
bool roundTrips(const ValueDescriptor &value, const PortContract &contract) {
  std::string error;
  auto admitted = admitValue(value, contract, error);
  if (!admitted) {
    std::fprintf(stderr, "admission error: %s\n", error.c_str());
    return false;
  }
  const auto bytes = encodeRuntimeValue(*admitted);
  auto decoded = decodeRuntimeValue(bytes, contract, {}, {}, error);
  return decoded.has_value() && encodeRuntimeValue(*decoded) == bytes;
}

// Axis-aligned cube centered on the origin. lowerMeshGenerator emits four vertices per
// face in face order (-Z, +Z, -Y, +Y, -X, +X), so vertex 4f + k belongs to face f.
// Vertex i carries stable identity i + 1. Triangle 2f is (4f, 4f+1, 4f+2) and triangle
// 2f+1 is (4f, 4f+2, 4f+3).
ValueDescriptor cube(StableId id) {
  std::string error;
  auto lowered = lowerMeshGenerator(id, id, 1, OperationCode::cube, {1, 1, 1, 0}, meshContract(), error);
  check(lowered.has_value(), "unit cube lowers");
  return lowered ? std::move(*lowered) : ValueDescriptor{};
}

// Each face's triangles have cross products of length one along the outward axis, so
// the accumulated vertex normal points along the outward axis and normalizes to it.
void cubeNormalFieldIsTheOutwardFaceAxis() {
  std::string error;
  const auto source = cube(3011);
  auto normals = lowerElementField(source, 3012, OperationCode::normalField, vectorFieldContract(), error);
  check(normals.has_value(), "normal field lowers on the cube");
  if (!normals) return;
  const auto &elements = std::get<FieldData>(normals->data).elements;
  check(elements.size() == 24 && normals->fieldDomainCardinality == 24,
        "normal field has one element per cube vertex");
  if (elements.size() != 24) return;
  const std::array<std::array<double, 3>, 6> axes{{{0.0, 0.0, -1.0}, {0.0, 0.0, 1.0},
      {0.0, -1.0, 0.0}, {0.0, 1.0, 0.0}, {-1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}}};
  bool outward = true;
  for (std::size_t face = 0; outward && face < 6; ++face)
    for (std::size_t corner = 0; outward && corner < 4; ++corner)
      outward = nearlyElement(elements[face * 4 + corner], axes[face][0], axes[face][1], axes[face][2]);
  check(outward, "each cube vertex normal is the unit outward normal of its face");
  check(normals->operations.back().code == OperationCode::normalField,
        "normal field records its operation");
}

// A TRS transform applied to the unit cube: scale (2, -3, 0.5), a quarter turn about Z,
// then translation (2, 3, 4). The quarter turn maps (x, y, z) to (-y, x, z).
void transformedCubePositionFieldMatchesStoredVertices() {
  std::string error;
  Transform transform;
  transform.translation = {2, 3, 4};
  transform.scale = {2, -3, 0.5f};
  transform.rotation = {0, 0, 0.7071067812f, 0.7071067812f};
  auto moved = lowerTransformGeometry(cube(3001), 3002, transform, meshContract(), error);
  check(moved.has_value(), "TRS transform lowers the unit cube for the position field");
  if (!moved) return;
  auto field = lowerElementField(*moved, 3003, OperationCode::positionField, vectorFieldContract(), error);
  check(field.has_value(), "position field lowers on the transformed cube");
  if (!field) return;
  const auto &elements = std::get<FieldData>(field->data).elements;
  const auto &positions = std::get<GeometryData>(moved->data).positions;
  check(elements.size() == 24 && positions.size() == 24 && field->fieldDomainCardinality == 24,
        "position field has one element per transformed vertex");
  if (elements.size() != 24 || positions.size() != 24) return;
  bool stored = true;
  for (std::size_t i = 0; stored && i < 24; ++i)
    stored = nearlyElement(elements[i], positions[i].x, positions[i].y, positions[i].z);
  check(stored, "position field equals the stored transformed positions exactly");
  // Vertex 0 is corner (-,-,-), vertex 1 is (-,+,-), vertex 3 is (+,-,-), vertex 6 is (+,+,+).
  // Scale, quarter turn and translation give (0.5, 2, 3.75), (3.5, 2, 3.75), (0.5, 4, 3.75)
  // and (3.5, 4, 4.25).
  check(nearlyElement(elements[0], 0.5, 2, 3.75) && nearlyElement(elements[1], 3.5, 2, 3.75)
            && nearlyElement(elements[3], 0.5, 4, 3.75) && nearlyElement(elements[6], 3.5, 4, 4.25),
        "position field values after scale, quarter turn and translation");
  check(field->operations.back().code == OperationCode::positionField,
        "position field records its operation");
}

// Cull Geometry with first 0, last 11, stride 2 and keep flag 0 removes the even
// triangles 0, 2, ..., 10. The odd triangles are the second triangle of each face,
// which use original vertices 4f, 4f+2 and 4f+3. Those become 3f, 3f+1 and 3f+2.
void cubeMeshCullKeepsSecondTrianglesAndReplays() {
  std::string error;
  const auto source = cube(3021);
  auto culled = lowerCullEdit(source, 3022, OperationCode::cullGeometry, {0, 11, 2, 0}, meshContract(), error);
  check(culled.has_value(), "stride cull of the cube lowers");
  if (!culled) return;
  const auto &mesh = std::get<GeometryData>(culled->data);
  std::vector<std::uint32_t> identity;
  for (std::uint32_t i = 0; i < 18; ++i) identity.push_back(i);
  check(mesh.indices == identity, "cube cull keeps six triangles renumbered 0 to 17");
  check(mesh.vertexIds == std::vector<StableId>({1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16,
                                                  17, 19, 20, 21, 23, 24}),
        "cube cull keeps the stable identities of the corners of the kept triangles");
  check(mesh.positions.size() == 18 && nearlyVec(mesh.positions[0], -0.5, -0.5, -0.5)
            && nearlyVec(mesh.positions[1], 0.5, 0.5, -0.5)
            && nearlyVec(mesh.positions[2], 0.5, -0.5, -0.5),
        "cube cull keeps the positions of the first face's kept corners");
  check(culled->operations.back().code == OperationCode::cullGeometry
            && culled->operations.back().parameters == std::array<float, 4>{0, 11, 2, 0},
        "cube cull records its selection on the operation");
  MaterializedInstancePlan replay;
  check(validateOperationPlan(*culled, {}, &replay, error) && replay.geometries.count(3022) == 1
            && equal(replay.geometries.at(3022), mesh),
        "replaying the cube cull reproduces its triangles and vertex identities");
  check(roundTrips(*culled, meshContract()), "cube cull admits and roundtrips");
}

// A cull after an Audio Deformer keeps the binding and re-runs on each frame's deformed
// mesh: the index range keeps the admitted layout, and the positions are this frame's.
void deformedCubeCullFollowsEachFrame() {
  std::string error;
  auto deformed = lowerAudioDeformer(cube(3041), 3042, spectrum::Binding{}, meshContract(), error);
  check(deformed.has_value(), "the cube's audio deformer lowers");
  if (!deformed) return;
  auto culled = lowerCullEdit(*deformed, 3043, OperationCode::cullGeometry, {0, 11, 2, 0}, meshContract(), error);
  if (!culled) std::fprintf(stderr, "deformed cull: %s\n", error.c_str());
  check(culled.has_value(), "a cull of deformed geometry lowers");
  if (!culled) return;
  check(culled->spectrumFields.size() == 1, "the cull keeps the deformer binding");
  const std::function<bool(GeometryData&, const spectrum::Binding&, std::string&)> lift =
      [](GeometryData& mesh, const spectrum::Binding&, std::string&) {
        for (auto& position : mesh.positions) position.z += 1;
        return true;
      };
  MaterializedInstancePlan frame;
  check(validateOperationPlan(*culled, {}, &frame, error, nullptr, nullptr, lift)
            && frame.geometries.count(3043) == 1,
        "the deformed cull replays with a frame deformer");
  if (frame.geometries.count(3043) != 1) return;
  const auto& evaluated = frame.geometries.at(3043);
  const auto& rest = std::get<GeometryData>(culled->data);
  check(evaluated.indices == rest.indices && evaluated.vertexIds == rest.vertexIds,
        "each frame keeps the admitted layout");
  check(evaluated.positions.size() == rest.positions.size()
            && nearly(evaluated.positions[0].z, rest.positions[0].z + 1),
        "each frame culls the deformed positions");
  check(roundTrips(*culled, meshContract()), "the deformed cull admits and roundtrips");
}

// A Bounding Box after an Audio Deformer boxes each frame's deformed mesh.
void deformedCubeBoundsFollowEachFrame() {
  std::string error;
  auto deformed = lowerAudioDeformer(cube(3051), 3052, spectrum::Binding{}, meshContract(), error);
  auto box = deformed ? lowerBoundingBox(*deformed, 3053, meshContract(), error) : std::nullopt;
  if (!box) std::fprintf(stderr, "deformed bounds: %s\n", error.c_str());
  check(box.has_value(), "a bounding box of deformed geometry lowers");
  if (!box) return;
  const std::function<bool(GeometryData&, const spectrum::Binding&, std::string&)> lift =
      [](GeometryData& mesh, const spectrum::Binding&, std::string&) {
        for (auto& position : mesh.positions) position.z += 1;
        return true;
      };
  MaterializedInstancePlan frame;
  check(validateOperationPlan(*box, {}, &frame, error, nullptr, nullptr, lift)
            && frame.geometries.count(3053) == 1,
        "the deformed box replays with a frame deformer");
  if (frame.geometries.count(3053) != 1) return;
  const auto& evaluated = frame.geometries.at(3053);
  const auto& rest = std::get<GeometryData>(box->data);
  check(evaluated.positions.size() == 8 && rest.positions.size() == 8, "each frame's box has eight vertices");
  bool lifted = evaluated.positions.size() == rest.positions.size();
  for (std::size_t i = 0; lifted && i < rest.positions.size(); ++i)
    lifted = nearly(evaluated.positions[i].z, rest.positions[i].z + 1);
  check(lifted, "each frame's box follows the deformed positions");
}

// Density points on an Audio Deformer cube follow each frame's deformed area. The unit
// cube's area 6 at density 2 admits 12 points; doubling its size gives area 24 and 48,
// or the node's maximum when that is lower.
void deformedCubeDensityFollowsEachFrame() {
  std::string error;
  auto deformed = lowerAudioDeformer(cube(3061), 3062, spectrum::Binding{}, meshContract(), error);
  auto points = deformed ? lowerPointDensityDistribution(*deformed, 3063, 64, 7, 2, pointsContract(), error)
                         : std::nullopt;
  if (!points) std::fprintf(stderr, "deformed density: %s\n", error.c_str());
  check(points.has_value(), "density points on deformed geometry lower");
  if (!points) return;
  check(std::get<PointsData>(points->data).points.size() == 12, "the rest area admits 12 points");
  const std::function<bool(GeometryData&, const spectrum::Binding&, std::string&)> grow =
      [](GeometryData& mesh, const spectrum::Binding&, std::string&) {
        for (auto& position : mesh.positions) position = {position.x * 2, position.y * 2, position.z * 2};
        return true;
      };
  MaterializedInstancePlan frame;
  const bool replayed = validateOperationPlan(*points, {}, &frame, error, nullptr, nullptr, grow);
  if (!replayed) std::fprintf(stderr, "deformed density replay: %s\n", error.c_str());
  check(replayed && frame.points.count(3063) == 1, "the density points replay with a frame deformer");
  if (frame.points.count(3063) == 1)
    check(frame.points.at(3063).points.size() == 48, "each frame samples the deformed area");
  // A frame past the node's maximum keeps the maximum: the node's own bound is the only one.
  auto capped = lowerPointDensityDistribution(*deformed, 3064, 30, 7, 2, pointsContract(), error);
  MaterializedInstancePlan cappedFrame;
  check(capped && validateOperationPlan(*capped, {}, &cappedFrame, error, nullptr, nullptr, grow)
            && cappedFrame.points.count(3064) == 1 && cappedFrame.points.at(3064).points.size() == 30,
        "a frame past the node's maximum samples the maximum");
}

// A Set Position spectrum displacement that a cull follows runs at the Set Position
// output during the replay, so the cull keeps the displaced vertices. Without the cull
// the displacement stays with the scene adapter.
void setPositionDisplacementCullFollowsEachFrame() {
  std::string error;
  auto moved = lowerTransformGeometry(cube(3071), 3072, Transform{}, meshContract(), error);
  check(moved.has_value(), "the Set Position stand-in lowers");
  if (!moved) return;
  spectrum::Binding binding;
  binding.target = spectrum::Target::vertexDisplacement;
  binding.fieldStableId = 3079; binding.sourceGeometryStableId = 3071;
  binding.consumerGeometryStableId = 3072;
  binding.coordinates.assign(std::get<GeometryData>(moved->data).positions.size(), 0.0f);
  auto bound = *moved;
  bound.spectrumFields = {binding};
  check(!displacedInReplay(bound, 3072), "a terminal Set Position keeps the adapter displacement");
  auto culled = lowerCullEdit(bound, 3073, OperationCode::cullGeometry, {0, 11, 2, 0}, meshContract(), error);
  if (!culled) std::fprintf(stderr, "set position cull: %s\n", error.c_str());
  check(culled.has_value(), "a cull after a Set Position displacement lowers");
  if (!culled) return;
  check(displacedInReplay(*culled, 3072) && hasReplayDisplacement(*culled),
        "the following cull moves the displacement into the replay");
  std::size_t calls = 0;
  RuntimeFieldEvaluation runtime;
  runtime.displaceConsumer = [&calls](GeometryData& mesh, StableId consumer, std::string&) {
    if (consumer != 3072) return false;
    ++calls;
    for (auto& position : mesh.positions) position.z += 1;
    return true;
  };
  MaterializedInstancePlan frame;
  const bool replayed = validateOperationPlan(*culled, {}, &frame, error, nullptr, &runtime);
  if (!replayed) std::fprintf(stderr, "set position cull replay: %s\n", error.c_str());
  check(replayed && calls == 1 && frame.geometries.count(3073) == 1,
        "the replay displaces the Set Position output once");
  if (frame.geometries.count(3073) != 1) return;
  const auto& evaluated = frame.geometries.at(3073);
  const auto& rest = std::get<GeometryData>(culled->data);
  check(evaluated.vertexIds == rest.vertexIds && evaluated.indices == rest.indices,
        "the displaced cull keeps the admitted layout");
  bool lifted = evaluated.positions.size() == rest.positions.size();
  for (std::size_t i = 0; lifted && i < rest.positions.size(); ++i)
    lifted = nearly(evaluated.positions[i].z, rest.positions[i].z + 1);
  check(lifted, "the cull keeps the displaced positions");
  RuntimeFieldEvaluation missing;
  MaterializedInstancePlan refused;
  check(!validateOperationPlan(*culled, {}, &refused, error, nullptr, &missing),
        "a replay without the frame owner refuses the displacement");
}

// In a composed scene, a Set Position displacement that a transform follows lands on the
// transformed terminal mesh, as the image path's adapter adds it after the replay.
void composedSetPositionDisplacementFollowsLaterTransform() {
  std::string error;
  auto moved = lowerTransformGeometry(cube(3081), 3082, Transform{}, meshContract(), error);
  check(moved.has_value(), "the Set Position stand-in lowers for the order check");
  if (!moved) return;
  spectrum::Binding binding;
  binding.target = spectrum::Target::vertexDisplacement;
  binding.fieldStableId = 3089; binding.sourceGeometryStableId = 3081;
  binding.consumerGeometryStableId = 3082; binding.direction = {0, 0, 1};
  binding.coordinates.assign(std::get<GeometryData>(moved->data).positions.size(), 0.0f);
  auto bound = *moved;
  bound.spectrumFields = {binding};
  Transform doubled;
  doubled.scale = {2, 2, 2};
  auto scaled = lowerTransformGeometry(bound, 3083, doubled, meshContract(), error);
  if (!scaled) std::fprintf(stderr, "order transform: %s\n", error.c_str());
  check(scaled.has_value(), "a transform after the Set Position displacement lowers");
  if (!scaled) return;
  check(!displacedInReplay(*scaled, 3082), "a transform keeps the displacement after the replay");
  const auto& rest = std::get<GeometryData>(scaled->data);
  std::size_t calls = 0;
  bool sawTerminal = false;
  RuntimeFieldEvaluation runtime;
  runtime.displaceAdapters = true;
  runtime.displaceConsumer = [&](GeometryData& mesh, StableId consumer, std::string&) {
    if (consumer != 3082) return false;
    ++calls;
    sawTerminal = mesh.positions.size() == rest.positions.size();
    for (std::size_t i = 0; sawTerminal && i < rest.positions.size(); ++i)
      sawTerminal = nearlyVec(mesh.positions[i], rest.positions[i].x, rest.positions[i].y, rest.positions[i].z);
    for (auto& position : mesh.positions) position.z += 1;
    return true;
  };
  MaterializedInstancePlan frame;
  const bool replayed = validateOperationPlan(*scaled, {}, &frame, error, nullptr, &runtime);
  if (!replayed) std::fprintf(stderr, "order replay: %s\n", error.c_str());
  check(replayed && calls == 1, "the composed replay displaces once");
  check(sawTerminal, "the displacement sees the transformed terminal mesh");
  if (frame.geometries.count(3083) != 1) return;
  const auto& evaluated = frame.geometries.at(3083);
  bool lifted = evaluated.positions.size() == rest.positions.size();
  for (std::size_t i = 0; lifted && i < rest.positions.size(); ++i)
    lifted = nearly(evaluated.positions[i].z, rest.positions[i].z + 1);
  check(lifted, "the lift is not scaled by the later transform");
}

// Join keeps the Audio Deformer of either input and joins each frame's deformed meshes.
void deformedCubeJoinFollowsEachFrame() {
  std::string error;
  auto deformed = lowerAudioDeformer(cube(3061), 3062, spectrum::Binding{}, meshContract(), error);
  const auto plain = cube(3063);
  auto joined = deformed ? lowerJoinGeometry(*deformed, plain, 3064, meshContract(), error) : std::nullopt;
  if (!joined) std::fprintf(stderr, "deformed join: %s\n", error.c_str());
  check(joined.has_value(), "a join of deformed and static geometry lowers");
  if (!joined) return;
  check(joined->spectrumFields.size() == 1, "the join keeps the deformer binding");
  const std::function<bool(GeometryData&, const spectrum::Binding&, std::string&)> lift =
      [](GeometryData& mesh, const spectrum::Binding&, std::string&) {
        for (auto& position : mesh.positions) position.z += 1;
        return true;
      };
  MaterializedInstancePlan frame;
  check(validateOperationPlan(*joined, {}, &frame, error, nullptr, nullptr, lift)
            && frame.geometries.count(3064) == 1,
        "the deformed join replays with a frame deformer");
  if (frame.geometries.count(3064) != 1) return;
  const auto& evaluated = frame.geometries.at(3064);
  const auto& rest = std::get<GeometryData>(joined->data);
  check(evaluated.indices == rest.indices && evaluated.positions.size() == rest.positions.size(),
        "each frame keeps the joined layout");
  std::size_t lifted = 0;
  for (std::size_t i = 0; i < rest.positions.size() && i < evaluated.positions.size(); ++i)
    if (nearly(evaluated.positions[i].z, rest.positions[i].z + 1)) ++lifted;
  check(lifted == 24, "only the deformed input's 24 vertices move");
  check(roundTrips(*joined, meshContract()), "the deformed join admits and roundtrips");
}

// Points from the cube's 24 vertices carry identities 1 to 24 in vertex order. Keep-selected
// with stride 2 keeps the even indices 0, 2, ..., 22, which have identities 1, 3, ..., 23.
void cubePointCullKeepsEvenIdentitiesAndReplays() {
  std::string error;
  const auto source = cube(3031);
  auto points = lowerPointsFromVertices(source, 3032, pointsContract(), error);
  check(points.has_value(), "points from cube vertices lower");
  if (!points) return;
  check(std::get<PointsData>(points->data).points.size() == 24,
        "points from vertices keep every cube vertex");
  auto culled = lowerCullEdit(*points, 3033, OperationCode::cullPoints, {0, 23, 2, 1}, pointsContract(), error);
  check(culled.has_value(), "keep-selected point cull lowers");
  if (!culled) return;
  const auto &survivors = std::get<PointsData>(culled->data).points;
  std::vector<StableId> identities;
  for (const auto &point : survivors) identities.push_back(point.stableId);
  std::vector<StableId> odd;
  for (StableId i = 0; i < 12; ++i) odd.push_back(2 * i + 1);
  check(identities == odd, "point cull keeps the even indices with identities 1, 3, ..., 23");
  check(survivors.size() == 12 && nearlyVec(survivors[1].position, 0.5, 0.5, -0.5)
            && nearlyVec(survivors[2].position, -0.5, -0.5, 0.5),
        "point cull keeps the positions of index 2 and index 4");
  MaterializedInstancePlan replay;
  check(validateOperationPlan(*culled, {}, &replay, error) && replay.points.count(3033) == 1
            && equal(replay.points.at(3033), std::get<PointsData>(culled->data)),
        "replaying the point cull reproduces its surviving points and identities");
  check(roundTrips(*culled, pointsContract()), "point cull admits and roundtrips");
}

// Instances on the cube's points take each point's identity and position. Cull Instances
// removes indices 2 to 9, so identities 3 to 10 are gone and 16 instances remain.
void cubeInstanceCullKeepsStableIdentitiesAndReplays() {
  std::string error;
  const auto source = cube(3041);
  auto points = lowerPointsFromVertices(source, 3042, pointsContract(), error);
  auto instances = points ? lowerInstanceOnPoints(*points, source, 3043, instancesContract(), error)
                          : std::nullopt;
  check(instances.has_value(), "instances on cube vertices lower");
  if (!instances) return;
  check(std::get<InstancesData>(instances->data).instances.size() == 24,
        "instances on points keep every cube vertex");
  auto culled = lowerInstanceEdit(*instances, 3044, OperationCode::cullInstances, {2, 9, 1, 0},
                                  Transform{}, instancesContract(), error);
  check(culled.has_value(), "range instance cull lowers");
  if (!culled) return;
  const auto &survivors = std::get<InstancesData>(culled->data).instances;
  std::vector<StableId> identities{1, 2};
  for (StableId id = 11; id <= 24; ++id) identities.push_back(id);
  std::vector<StableId> kept;
  for (const auto &instance : survivors) kept.push_back(instance.stableId);
  check(kept == identities, "instance cull keeps identities 1, 2 and 11 to 24");
  bool sourced = true;
  for (const auto &instance : survivors) sourced = sourced && instance.sourceStableId == 3041;
  check(sourced, "culled instances keep their source geometry identity");
  // Identity 11 is index 10, which is corner 5 (+, -, +) of the cube.
  check(survivors.size() == 16 && nearlyVec(survivors[2].transform.translation, 0.5, -0.5, 0.5),
        "culled instances keep the transform of their stable identity");
  MaterializedInstancePlan replay;
  check(validateOperationPlan(*culled, {}, &replay, error)
            && equal(replay.instances, std::get<InstancesData>(culled->data)),
        "replaying the instance cull reproduces its surviving instances");
  check(roundTrips(*culled, instancesContract()), "instance cull admits and roundtrips");
}

} // namespace

int main() {
  cubeNormalFieldIsTheOutwardFaceAxis();
  transformedCubePositionFieldMatchesStoredVertices();
  cubeMeshCullKeepsSecondTrianglesAndReplays();
  cubePointCullKeepsEvenIdentitiesAndReplays();
  deformedCubeCullFollowsEachFrame();
  deformedCubeBoundsFollowEachFrame();
  deformedCubeJoinFollowsEachFrame();
  deformedCubeDensityFollowsEachFrame();
  setPositionDisplacementCullFollowsEachFrame();
  composedSetPositionDisplacementFollowsLaterTransform();
  cubeInstanceCullKeepsStableIdentitiesAndReplays();
  if (failures != 0) {
    std::fprintf(stderr, "%d geometry field and cull check(s) failed\n", failures);
    return 1;
  }
  std::printf("geometry core field and cull tests passed\n");
  return 0;
}
