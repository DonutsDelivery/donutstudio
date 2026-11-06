#include "../../shared/GeometryCore.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <optional>
#include <string>

// Ocean foam as a material colour: the Named Attribute read of "foam", mixed
// between two colours and stored as the vertex "color" that the native draw
// multiplies into the material's base colour, at lowering and at replay.
namespace {
using namespace videowire::geometry;

int failures = 0;

void check(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
  }
}

PortContract fieldContract(ValueType type) {
  PortContract contract;
  contract.carrier = CarrierKind::field; contract.fieldDomain = Domain::vertex; contract.fieldValueType = type;
  contract.fieldInterpolation = Interpolation::constant; contract.overflow = OverflowPolicy::reject;
  contract.maxFieldElements = 4096; contract.maxAttributes = 8;
  return contract;
}

std::optional<ValueDescriptor> colorConstant(StableId id, std::size_t count, double r, double g, double b,
                                             std::string& error) {
  FieldOperationSettings settings; settings.domain = Domain::vertex; settings.valueType = ValueType::color;
  settings.values = {r, g, b, 1, static_cast<double>(count), 0, 0, 0};
  return lowerConstructedField(id, OperationCode::fieldConstant, settings, nullptr, nullptr, nullptr,
                               fieldContract(ValueType::color), error);
}

void foamColor() {
  PortContract contract; contract.carrier = CarrierKind::geometry3D; contract.overflow = OverflowPolicy::reject;
  contract.maxVertices = 4096; contract.maxIndices = 24576; contract.maxAttributes = 8;
  std::string error;
  auto grid = lowerGrid(40101, 40101, 1, 24, 20, 0.25f, contract, error);
  check(bool(grid), "foam colour grid lowers");
  if (!grid) return;
  OceanSettings o; o.smallestWave = 0; o.size = 0; o.loopPeriod = 6; o.windSpeed = 14;
  auto ocean = lowerOceanSurface(*grid, 40102, (StableId{1} << 42) + 40102 * 8, o, 2.0, 1.2, contract, error);
  check(bool(ocean), ("foam colour ocean lowers: " + error).c_str());
  if (!ocean) return;
  const auto count = std::get<GeometryData>(ocean->data).positions.size();

  auto foam = lowerNamedAttribute(*ocean, nullptr, 40103, OperationCode::readNamedAttribute, "foam",
                                  ValueType::floatValue, fieldContract(ValueType::floatValue), error);
  check(bool(foam), ("the Named Attribute read finds the ocean foam: " + error).c_str());
  auto water = colorConstant(40104, count, 0.1, 0.3, 0.5, error);
  auto white = colorConstant(40105, count, 1, 1, 1, error);
  check(water && white, "colour constants lower");
  if (!foam || !water || !white) return;
  FieldOperationSettings mix; mix.domain = Domain::vertex; mix.valueType = ValueType::color;
  auto tint = lowerConstructedField(40106, OperationCode::fieldMix, mix, &*water, &*white, &*foam,
                                    fieldContract(ValueType::color), error);
  check(bool(tint), ("foam mixes water and white: " + error).c_str());
  if (!tint) return;
  auto colored = lowerNamedAttribute(*ocean, &*tint, 40107, OperationCode::storeNamedAttribute, "color",
                                     ValueType::color, contract, error);
  check(bool(colored), ("the mix stores as the vertex colour: " + error).c_str());
  if (!colored) return;

  const auto matches = [&](const ValueDescriptor& value, const char* message) {
    const auto* stored = findNamedAttribute(value.attributes, "color");
    const auto* amount = findNamedAttribute(value.attributes, "foam");
    if (!stored || !amount || stored->elements.size() != count || amount->elements.size() != count) {
      check(false, message); return 0.0;
    }
    double worst = 0, peak = 0;
    for (std::size_t i = 0; i < count; ++i) {
      const auto t = std::clamp(amount->elements[i].components[0], 0.0, 1.0);
      peak = std::max(peak, t);
      worst = std::max(worst, std::abs(stored->elements[i].components[0] - (0.1 + 0.9 * t)));
      worst = std::max(worst, std::abs(stored->elements[i].components[2] - (0.5 + 0.5 * t)));
    }
    check(worst < 1e-9, message);
    return peak;
  };
  check(matches(*colored, "each vertex colour follows its foam") > 0, "a windy ocean has some foam");

  RuntimeFieldEvaluation runtime; runtime.timelineSeconds = 2.25;
  MaterializedInstancePlan plan; ValueDescriptor replayed;
  check(validateOperationPlan(*colored, {}, &plan, error, &replayed, &runtime),
        ("the foam colour replays at a timeline time: " + error).c_str());
  matches(replayed, "replayed vertex colours follow the replayed foam");
}
} // namespace

int main() {
  foamColor();
  if (failures != 0) return 1;
  std::puts("geometry foam colour tests passed");
  return 0;
}
