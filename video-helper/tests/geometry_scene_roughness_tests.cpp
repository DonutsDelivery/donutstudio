#include "../../shared/GeometryCoreScene.h"

#include <cstdio>
#include <memory>
#include <string>

// A vertex Float attribute named "roughness" (Ocean foam through Store
// Attribute, say) fills each scene vertex's roughness lane at full weight,
// clamped to [0, 1]. A mesh without it keeps weight 0, so the material
// roughness stands, and a "roughness" of another type is refused by name.
namespace {
using namespace videowire::geometry;
using namespace HarmonicMIDI::grid;

int failures = 0;

void check(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
  }
}

PortContract meshContract() {
  PortContract contract; contract.carrier = CarrierKind::geometry3D; contract.overflow = OverflowPolicy::reject;
  contract.maxVertices = 1000; contract.maxIndices = 6000; contract.maxAttributes = 8;
  return contract;
}

std::optional<ValueDescriptor> constant(StableId id, ValueType type, std::array<double, 4> value, std::string& error) {
  PortContract field;
  field.carrier = CarrierKind::field; field.fieldDomain = Domain::vertex; field.fieldValueType = type;
  field.fieldInterpolation = Interpolation::constant; field.maxFieldElements = 64; field.maxAttributes = 8;
  FieldOperationSettings settings;
  settings.domain = Domain::vertex; settings.valueType = type;
  settings.values = {value[0], value[1], value[2], value[3], 1, 0, 0, 0};
  return lowerConstructedField(id, OperationCode::fieldConstant, settings, nullptr, nullptr, nullptr, field, error);
}

// Composes one mesh into a fresh scene; returns false with the error on refusal.
bool compose(const ValueDescriptor& mesh, Visual3DScene& scene, std::string& error) {
  const auto contract = withAttributeContract(meshContract(), mesh);
  auto admitted = admitValue(ValueDescriptor(mesh), contract, {}, error);
  if (!admitted) return false;
  SceneObjectRecord object; object.id = {1}; object.material = {1};
  SceneMaterialRecord material; material.id = {1};
  return appendGeometryToScene(*admitted, object, material, scene, error);
}

std::optional<ValueDescriptor> stored(const ValueDescriptor& grid, StableId fieldId, ValueType type,
                                      std::array<double, 4> value, std::string& error) {
  auto field = constant(fieldId, type, value, error);
  if (!field) return std::nullopt;
  return lowerNamedAttribute(grid, &*field, fieldId + 1, OperationCode::storeNamedAttribute, "roughness", type,
                             meshContract(), error);
}

void mapping() {
  std::string error;
  auto grid = lowerGrid(70101, 70101, 1, 3, 3, 1.0f, meshContract(), error);
  check(bool(grid), ("the grid lowers: " + error).c_str());
  if (!grid) return;

  auto plain = std::make_unique<Visual3DScene>();
  check(compose(*grid, *plain, error) && plain->vertexCount == 9, ("a plain grid composes: " + error).c_str());
  bool unweighted = true;
  for (std::size_t i = 0; i < plain->vertexCount; ++i) unweighted = unweighted && plain->vertices[i].roughness[1] == 0;
  check(unweighted, "a mesh without roughness leaves the lane at weight 0");

  auto rough = stored(*grid, 70110, ValueType::floatValue, {0.3, 0, 0, 0}, error);
  check(bool(rough), ("a Float roughness stores: " + error).c_str());
  auto scene = std::make_unique<Visual3DScene>();
  const bool composed = rough && compose(*rough, *scene, error);
  check(composed, ("the rough grid composes: " + error).c_str());
  if (composed) {
    bool lane = scene->vertexCount == 9;
    for (std::size_t i = 0; lane && i < scene->vertexCount; ++i)
      lane = scene->vertices[i].roughness[0] == 0.3f && scene->vertices[i].roughness[1] == 1.0f;
    check(lane, "each vertex carries the stored roughness at full weight");
  }

  auto high = stored(*grid, 70120, ValueType::floatValue, {1.7, 0, 0, 0}, error);
  auto clamped = std::make_unique<Visual3DScene>();
  check(high && compose(*high, *clamped, error) && clamped->vertices[0].roughness[0] == 1.0f,
        "a roughness above 1 clamps to 1");

  auto vector = stored(*grid, 70130, ValueType::vector, {0.2, 0.2, 0.2, 0}, error);
  auto refused = std::make_unique<Visual3DScene>();
  error.clear();
  check(vector && !compose(*vector, *refused, error) && error.find("roughness") != std::string::npos,
        "a vector roughness is refused by name");
}
} // namespace

int main() {
  mapping();
  if (failures != 0) return 1;
  std::puts("geometry scene roughness tests passed");
  return 0;
}
