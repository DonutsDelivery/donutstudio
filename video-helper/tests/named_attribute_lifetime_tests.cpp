// Named attribute lifetime: Store Attribute and Read Attribute as implemented.
// The written contract is the block at the top of
// video-helper/src/geometry_core_execution_cache.h. These checks use only the
// static (non-runtime) path, so no GPU, audio or clock input is needed.
// Not covered here: runtime-field seek and the GL/Metal frame cache hit.
#include "../src/geometry_core_admission.h"

#include <array>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {
using namespace videowire::geometry;
int failures = 0;

void check(bool condition, const char *message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

// Same port shapes as constructedFieldMeshContract and constructedFieldContract
// in constructed_field_fixture.h, kept local so this file needs no fixture.
PortContract attributeMeshContract() {
  PortContract contract;
  contract.carrier = CarrierKind::geometry3D;
  contract.maxVertices = 1024;
  contract.maxIndices = 6144;
  contract.maxAttributes = 32;
  return contract;
}

PortContract attributeFieldContract(Domain domain, ValueType type) {
  PortContract contract;
  contract.carrier = CarrierKind::field;
  contract.fieldDomain = domain;
  contract.fieldValueType = type;
  contract.fieldInterpolation = Interpolation::constant;
  contract.maxFieldElements = 1024;
  contract.maxAttributes = 32;
  return contract;
}

bool sameAttributes(const std::vector<AttributeData> &a,
                    const std::vector<AttributeData> &b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (!(a[i].descriptor == b[i].descriptor) ||
        a[i].elements.size() != b[i].elements.size())
      return false;
    for (std::size_t j = 0; j < a[i].elements.size(); ++j)
      if (a[i].elements[j].components != b[i].elements[j].components)
        return false;
  }
  return true;
}

std::optional<ValueDescriptor> constantField(StableId valueId, Domain domain,
                                             ValueType type, std::string &error) {
  FieldOperationSettings settings;
  settings.domain = domain;
  settings.valueType = type;
  settings.values = type == ValueType::vector
      ? std::array<double, 8>{0.5, 0.25, 0.75, 0, 1, 0, 0, 0}
      : std::array<double, 8>{0.5, 0, 0, 0, 1, 0, 0, 0};
  return lowerConstructedField(valueId, OperationCode::fieldConstant, settings,
                               nullptr, nullptr, nullptr,
                               attributeFieldContract(domain, type), error);
}

// Where a store's value is visible, and whether replay of the same plan agrees.
void transientVisibility(const ValueDescriptor &mesh, const ValueDescriptor &weight,
                         std::string &error) {
  const auto contract = attributeMeshContract();
  const auto stored = lowerNamedAttribute(mesh, &weight, 9920,
                                          OperationCode::storeNamedAttribute, "weight",
                                          ValueType::floatValue, contract, error);
  check(stored.has_value(), "a float store lowers onto the grid");
  if (!stored) return;
  check(findNamedAttribute(stored->attributes, "weight") != nullptr,
        "a store is visible on its own output");
  check(findNamedAttribute(mesh.attributes, "weight") == nullptr,
        "a store never mutates its source value");
  check(!lowerNamedAttribute(mesh, nullptr, 9921, OperationCode::readNamedAttribute,
                             "weight", ValueType::floatValue,
                             attributeFieldContract(Domain::vertex, ValueType::floatValue),
                             error).has_value(),
        "an upstream value cannot read a store made downstream of it");

  ValueDescriptor first, second, upstream;
  const bool replays = validateOperationPlan(*stored, {}, nullptr, error, &first) &&
                       validateOperationPlan(*stored, {}, nullptr, error, &second);
  check(replays && sameAttributes(first.attributes, stored->attributes),
        "replay reproduces the attributes the store lowered to");
  check(replays && sameAttributes(first.attributes, second.attributes),
        "a repeated replay of one plan is identical");
  check(validateOperationPlan(mesh, {}, nullptr, error, &upstream) &&
        upstream.attributes.empty(),
        "replaying the source alone yields no stored attribute");

  // A sibling plan replayed between two calls must not leak into the first plan.
  const auto sibling = lowerNamedAttribute(mesh, &weight, 9922,
                                           OperationCode::storeNamedAttribute, "other",
                                           ValueType::floatValue, contract, error);
  ValueDescriptor siblingReplay, third;
  check(sibling.has_value() && validateOperationPlan(*sibling, {}, nullptr, error, &siblingReplay) &&
        validateOperationPlan(*stored, {}, nullptr, error, &third) &&
        sameAttributes(third.attributes, first.attributes),
        "an interleaved sibling plan does not leak into a later replay");
}

// A chain with no runtime fields is eligible for frame-cache replay.
void staticChainIsCacheable(const ValueDescriptor &stored) {
  check(!hasRuntimeFields(stored),
        "a constant store chain is not dynamic, so a frame-cache hit may stand in for it");
}

// Reading a name nothing stored refuses. It never yields a default or zero.
void missingNameRefuses(const ValueDescriptor &stored, std::string &error) {
  const auto contract = attributeFieldContract(Domain::vertex, ValueType::floatValue);
  error.clear();
  check(!lowerNamedAttribute(stored, nullptr, 9930, OperationCode::readNamedAttribute,
                             "absent", ValueType::floatValue, contract, error).has_value(),
        "reading a name no store wrote refuses instead of returning a default");
  check(error.find("missing") != std::string::npos,
        "the missing-name refusal says why");

  const auto readBack = lowerNamedAttribute(stored, nullptr, 9931,
                                            OperationCode::readNamedAttribute, "weight",
                                            ValueType::floatValue, contract, error);
  check(readBack.has_value(), "the stored name reads back");
  if (!readBack) return;
  auto forged = *readBack;
  forged.operations.back().field.attributeName = "absent";
  ValueDescriptor replayed;
  check(!validateOperationPlan(forged, {}, nullptr, error, &replayed),
        "replay refuses a read of a name the plan never stored");
}

// A same-name store replaces the attribute, including its type. It is not a conflict.
void sameNameStoreReplaces(const ValueDescriptor &mesh, const ValueDescriptor &weight,
                           const ValueDescriptor &vectorWeight, std::string &error) {
  const auto contract = attributeMeshContract();
  const auto floatStore = lowerNamedAttribute(mesh, &weight, 9940,
                                              OperationCode::storeNamedAttribute, "weight",
                                              ValueType::floatValue, contract, error);
  check(floatStore.has_value(), "the first float store lowers");
  if (!floatStore) return;

  check(!lowerNamedAttribute(*floatStore, &vectorWeight, 9941,
                             OperationCode::storeNamedAttribute, "weight",
                             ValueType::floatValue, contract, error).has_value(),
        "a store refuses a field whose value type differs from the store");

  const auto vectorStore = lowerNamedAttribute(*floatStore, &vectorWeight, 9942,
                                               OperationCode::storeNamedAttribute, "weight",
                                               ValueType::vector, contract, error);
  check(vectorStore.has_value(), "a same-name store with a new type replaces the attribute");
  if (!vectorStore) return;

  std::size_t count = 0;
  for (const auto &attribute : vectorStore->attributes)
    if (attribute.descriptor.name == "weight") ++count;
  const auto *replaced = findNamedAttribute(vectorStore->attributes, "weight");
  check(count == 1 && replaced != nullptr &&
        replaced->descriptor.valueType == ValueType::vector &&
        replaced->descriptor.domain == Domain::vertex,
        "one vertex vector attribute survives the same-name store");

  ValueDescriptor replayed;
  check(validateOperationPlan(*vectorStore, {}, nullptr, error, &replayed) &&
        sameAttributes(replayed.attributes, vectorStore->attributes),
        "replay agrees with the replaced attribute");

  check(!lowerNamedAttribute(*vectorStore, nullptr, 9943,
                             OperationCode::readNamedAttribute, "weight",
                             ValueType::floatValue,
                             attributeFieldContract(Domain::vertex, ValueType::floatValue),
                             error).has_value(),
        "the replaced type no longer reads as the old type");
  check(lowerNamedAttribute(*vectorStore, nullptr, 9944,
                            OperationCode::readNamedAttribute, "weight",
                            ValueType::vector,
                            attributeFieldContract(Domain::vertex, ValueType::vector),
                            error).has_value(),
        "the replacement reads as its new type");
  check(!lowerNamedAttribute(*vectorStore, nullptr, 9945,
                             OperationCode::readNamedAttribute, "weight",
                             ValueType::vector,
                             attributeFieldContract(Domain::face, ValueType::vector),
                             error).has_value(),
        "a face read refuses a vertex attribute of the same name");
}

// Nothing in replay keeps state between calls, so lowering the same plan again
// and replaying it gives the same attributes. This is the restart case.
void restartReproduces(const ValueDescriptor &mesh, const ValueDescriptor &weight,
                       std::string &error) {
  const auto contract = attributeMeshContract();
  const auto before = lowerNamedAttribute(mesh, &weight, 9950,
                                          OperationCode::storeNamedAttribute, "weight",
                                          ValueType::floatValue, contract, error);
  const auto after = lowerNamedAttribute(mesh, &weight, 9950,
                                         OperationCode::storeNamedAttribute, "weight",
                                         ValueType::floatValue, contract, error);
  check(before.has_value() && after.has_value(), "the same store lowers twice");
  if (!before || !after) return;
  ValueDescriptor a, b;
  check(validateOperationPlan(*before, {}, nullptr, error, &a) &&
        validateOperationPlan(*after, {}, nullptr, error, &b) &&
        sameAttributes(a.attributes, b.attributes),
        "a plan lowered again after restart replays identical attributes");
}
} // namespace

int main() {
  std::string error;
  const auto mesh = lowerGrid(9901, 9901, 1, 2, 2, 1.0f, attributeMeshContract(), error);
  const auto weight = constantField(9902, Domain::vertex, ValueType::floatValue, error);
  const auto vectorWeight = constantField(9903, Domain::vertex, ValueType::vector, error);
  check(mesh.has_value() && weight.has_value() && vectorWeight.has_value(),
        "the grid and constant fields lower");
  if (mesh && weight && vectorWeight) {
    error.clear();
    const auto stored = lowerNamedAttribute(*mesh, &*weight, 9920,
                                            OperationCode::storeNamedAttribute, "weight",
                                            ValueType::floatValue,
                                            attributeMeshContract(), error);
    check(stored.has_value(), "the reference store lowers");
    transientVisibility(*mesh, *weight, error);
    if (stored) {
      staticChainIsCacheable(*stored);
      missingNameRefuses(*stored, error);
    }
    sameNameStoreReplaces(*mesh, *weight, *vectorWeight, error);
    restartReproduces(*mesh, *weight, error);
  }
  if (failures)
    std::cerr << failures << " named attribute lifetime checks failed\n";
  return failures ? 1 : 0;
}
