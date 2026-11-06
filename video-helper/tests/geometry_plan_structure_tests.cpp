#include "../../shared/GeometryCore.h"
#include "../../shared/GeometryCoreTransport.h"
#include "../../shared/GeometryPlanStructure.h"

#include <cstdio>
#include <cstring>
#include <optional>
#include <string>

// The structure key holds through a value edit and moves on a topology edit, on the same
// lowering route the Ocean Waves node takes.
namespace {
using namespace videowire::geometry;

int failures = 0;

void check(bool condition, const char *message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
  }
}

PortContract seaContract() {
  PortContract contract;
  contract.carrier = CarrierKind::geometry3D;
  contract.overflow = OverflowPolicy::reject;
  contract.maxVertices = kMaximumVertices;
  contract.maxIndices = kMaximumIndices;
  contract.maxAttributes = kMaximumAttributes;
  return contract;
}

struct Sea final {
  std::uint32_t columns = 16, rows = 16;
  float spacing = 0.25f;
  double amplitude = 3.5, windSpeed = 1.5, heading = 110;
};

ValueDescriptor lowerSea(const Sea &sea) {
  std::string error;
  const auto contract = seaContract();
  const StableId gridId = 7, waveId = 9;
  const auto grid = lowerGrid(gridId, gridId, 1, sea.columns, sea.rows, sea.spacing, contract, error);
  check(grid.has_value(), "the sea grid lowers");
  if (!grid) return {};
  const auto wave = lowerWindSpectrumSurfaceWave(*grid, waveId, (StableId{1} << 40) + waveId * 16,
      sea.amplitude, sea.windSpeed, sea.heading, contract, error);
  if (!wave) std::fprintf(stderr, "sea wave: %s\n", error.c_str());
  check(wave.has_value(), "the sea wave lowers");
  return wave ? *wave : ValueDescriptor{};
}

std::uint64_t keyOf(const ValueDescriptor &value) {
  const auto split = splitPlanStructure(value);
  check(split.has_value(), "the plan splits");
  return split ? split->key : 0;
}

void testValueEditKeepsKey() {
  const auto base = lowerSea({});
  Sea edited; edited.amplitude = 3.85;
  const auto amplitude = lowerSea(edited);
  check(keyOf(base) != 0, "the key is set");
  check(keyOf(base) == keyOf(amplitude), "an amplitude edit keeps the key");
  check(splitPlanStructure(base)->values != splitPlanStructure(amplitude)->values,
        "an amplitude edit changes the values");
  check(splitPlanStructure(base)->pointwise, "the sea is generator and pointwise operations only");
  edited = {}; edited.windSpeed = 2.5;
  check(keyOf(base) == keyOf(lowerSea(edited)), "a wind speed edit keeps the key");
  edited = {}; edited.heading = 40;
  check(keyOf(base) == keyOf(lowerSea(edited)), "a wind heading edit keeps the key");
  edited = {}; edited.spacing = 0.5f;
  check(keyOf(base) == keyOf(lowerSea(edited)), "a grid spacing edit keeps the key");
}

// A forged collision: one hash over different bytes is not the same structure.
void testCollisionIsNotTheSameStructure() {
  Sea edited; edited.amplitude = 3.85;
  const auto a = *splitPlanStructure(lowerSea({}));
  const auto same = *splitPlanStructure(lowerSea(edited));
  check(samePlanStructure(a, same), "a value edit has the same structure");
  check(a.keyBytes == same.keyBytes, "a value edit keeps every key byte");
  edited = {}; edited.columns = 17;
  auto forged = *splitPlanStructure(lowerSea(edited));
  check(a.keyBytes != forged.keyBytes, "a topology edit changes the key bytes");
  forged.key = a.key;
  check(!samePlanStructure(a, forged), "an equal hash over different bytes is refused");
}

// The receipt digest of an SDF request is the digest of its canonical bytes.
void testSdfDigestMatchesItsBytes() {
  videowire::sdfsurface::Request request;
  request.gradientStep = 0.125;
  std::uint64_t h = 1469598103934665603ull, expected = h;
  digestSdfSurfaceRequest(h, request);
  const auto real = [&](double x) { std::uint64_t bits = 0; std::memcpy(&bits, &x, sizeof(bits)); digestValue(expected, bits); };
  digestValue(expected, request.algorithmVersion);
  for (auto x : request.minimum) real(x);
  for (auto x : request.maximum) real(x);
  for (auto x : request.cells) digestValue(expected, x);
  real(request.gradientStep);
  real(request.maximumResidual);
  digestValue(expected, request.geometry.schemaVersion);
  digestValue(expected, request.geometry.rootId);
  check(request.geometry.records.empty(), "the digest case has no records");
  check(h == expected, "the SDF digest keeps its value");
}

void testTopologyEditMovesKey() {
  const auto base = lowerSea({});
  Sea edited; edited.columns = 17;
  check(keyOf(base) != keyOf(lowerSea(edited)), "a column edit moves the key");
  edited = {}; edited.rows = 15;
  check(keyOf(base) != keyOf(lowerSea(edited)), "a row edit moves the key");
}

// The values of an edited plan, written onto the old plan, give exactly the edited operations.
void testValuesReplayTheEdit() {
  auto operations = lowerSea({}).operations;
  Sea edited; edited.amplitude = 3.85;
  const auto target = lowerSea(edited);
  check(applyPlanValues(operations, splitPlanStructure(target)->values), "the values apply");
  check(operations.size() == target.operations.size(), "the operation count holds");
  for (std::size_t i = 0; i < operations.size() && i < target.operations.size(); ++i)
    check(operations[i].parameters == target.operations[i].parameters
              && operations[i].field == target.operations[i].field,
          "each operation matches the lowered edit");
  check(!applyPlanValues(operations, {1.0}), "a wrong value count is refused");
}

// Evaluated vertices are not part of the key, so a plan with them removed keeps it.
void testEvaluatedDataIsNotStructure() {
  const auto base = lowerSea({});
  auto stripped = base;
  stripped.data = GeometryData{};
  const auto a = splitPlanStructure(base), b = splitPlanStructure(stripped);
  check(a->key == b->key && a->values == b->values, "the split ignores evaluated vertices");
}

void testTransportKeepsTheSplit() {
  const auto base = lowerSea({});
  std::string error;
  const auto contract = seaContract();
  const auto admitted = admitValue(base, contract, error);
  check(admitted.has_value(), "the sea admits");
  if (!admitted) return;
  const auto bytes = encodeRuntimeValue(*admitted);
  const auto decoded = decodeRuntimeValue(bytes, contract, {}, {}, error);
  check(decoded.has_value(), "the sea decodes");
  if (!decoded) return;
  const auto a = splitPlanStructure(base), b = splitPlanStructure(decoded->descriptor());
  check(a->key == b->key && a->values == b->values, "the wire keeps the key and the values");
}

OperationRecord mesh(OperationCode code, std::array<float, 4> parameters) {
  OperationRecord op;
  op.stableId = op.inputStableId = op.outputStableId = 1;
  op.code = code;
  op.parameters = parameters;
  return op;
}

ValueDescriptor single(OperationRecord op) {
  ValueDescriptor value;
  value.carrier = CarrierKind::geometry3D;
  value.stableId = value.sourceStableId = 1;
  value.operations = {std::move(op)};
  return value;
}

void testGeneratorSizesAreValues() {
  const auto sphere = [](float radius, float segments) {
    return keyOf(single(mesh(OperationCode::sphere, {radius, segments, 8, 0})));
  };
  check(sphere(1, 16) == sphere(2, 16), "a sphere radius edit keeps the key");
  check(sphere(1, 16) != sphere(1, 24), "a sphere segment edit moves the key");
  const auto cylinder = [](float height, float hole) {
    return keyOf(single(mesh(OperationCode::cylinder, {2, height, 16, hole})));
  };
  check(cylinder(1, 0) == cylinder(3, 0), "a cylinder height edit keeps the key");
  check(cylinder(1, 0) != cylinder(1, 0.5f), "opening a cylinder's hole moves the key");
  const auto constant = [](double value, double count) {
    OperationRecord op; op.stableId = op.inputStableId = op.outputStableId = 1;
    op.code = OperationCode::fieldConstant;
    op.field.values = {value, 0, 0, 0, count, 0, 0, 0};
    return keyOf(single(op));
  };
  check(constant(1, 4) == constant(2, 4), "a constant field's value keeps the key");
  check(constant(1, 4) != constant(1, 5), "a constant field's count moves the key");
}

// A displaced grid, optionally culled by a field whose threshold is one number upstream.
ValueDescriptor culled(double threshold, bool cull) {
  ValueDescriptor value;
  value.carrier = CarrierKind::geometry3D;
  value.stableId = value.sourceStableId = 1;
  auto grid = mesh(OperationCode::grid, {8, 8, 1, 0});
  OperationRecord field; field.stableId = field.inputStableId = field.outputStableId = 2;
  field.code = OperationCode::fieldConstant;
  field.field.values = {threshold, 0, 0, 0, 64, 0, 0, 0};
  value.operations = {grid, field};
  if (cull) {
    OperationRecord op; op.stableId = op.outputStableId = 3; op.inputStableId = 1;
    op.secondaryInputStableId = 2; op.code = OperationCode::cullGeometry;
    value.operations.push_back(op);
  }
  return value;
}

// Cull's output counts follow its field, so that field's value is structure there.
void testValueDependentTopologyTakesItsInputs() {
  check(keyOf(culled(0.25, false)) == keyOf(culled(0.75, false)),
        "without a cull the field value keeps the key");
  check(keyOf(culled(0.25, true)) != keyOf(culled(0.75, true)),
        "a value feeding a cull moves the key");
  const auto split = splitPlanStructure(culled(0.25, true));
  check(!split->pointwise, "a cull is not pointwise");
  check(split->values.empty(), "every number upstream of a cull is structure");
  auto operations = culled(0.25, true).operations;
  check(applyPlanValues(operations, {}), "a culled plan takes an empty value list");
}

void testRetainedMeshContentIsStructure() {
  const auto retained = [](std::uint32_t lastIndex) {
    auto data = std::make_shared<RetainedMeshData>();
    data->geometry.positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}};
    data->geometry.vertexIds = {1, 2, 3, 4};
    data->geometry.indices = {0, 1, lastIndex};
    auto op = mesh(OperationCode::retainedMesh, {});
    op.retainedMesh = data;
    return keyOf(single(op));
  };
  check(retained(2) == retained(2), "the same mesh keeps the key");
  check(retained(2) != retained(3), "changed connectivity moves the key");
}

// A sea bound to the spectrum and the score. A binding's gain or shape is a value, so a
// music-reactive plan takes the same value path as a plain edit.
ValueDescriptor boundSea(float gain, std::uint32_t lastBand, float scoreGain) {
  auto value = lowerSea({});
  spectrum::Binding audio;
  audio.fieldStableId = 21; audio.sourceGeometryStableId = 7; audio.consumerGeometryStableId = 9;
  audio.gain = gain; audio.lastBand = lastBand;
  audio.target = spectrum::Target::geometryDeformation;
  value.spectrumFields.push_back(audio);
  scorefield::Binding score;
  score.fieldStableId = 22; score.sourceGeometryStableId = 7; score.scoreSourceStableId = 5;
  score.gain = scoreGain;
  value.scoreFields.push_back(score);
  return value;
}

void testBoundPlansTakeTheValuePath() {
  const auto base = splitPlanStructure(boundSea(1, 63, 1));
  check(base.has_value(), "a bound plan splits");
  if (!base) return;
  check(!base->pointwise, "a bound plan is evaluated per frame, not pointwise");
  const auto gain = splitPlanStructure(boundSea(2, 63, 1));
  check(samePlanStructure(*base, *gain), "a spectrum gain edit keeps the structure");
  check(base->values != gain->values, "a spectrum gain edit changes the values");
  check(samePlanStructure(*base, *splitPlanStructure(boundSea(1, 63, 3))),
        "a score gain edit keeps the structure");
  check(!samePlanStructure(*base, *splitPlanStructure(boundSea(1, 31, 1))),
        "a band range edit moves the key");
  check(!samePlanStructure(*base, *splitPlanStructure(lowerSea({}))),
        "removing the bindings moves the key");
  auto value = boundSea(1, 63, 1);
  const auto target = boundSea(2, 63, 3);
  check(applyPlanValues(value, splitPlanStructure(target)->values), "bound values apply");
  check(value.spectrumFields[0].gain == 2 && value.scoreFields[0].gain == 3,
        "the bound values reach their bindings");
  check(splitPlanStructure(value)->values == splitPlanStructure(target)->values,
        "the applied plan has the edited values");
  check(!applyPlanValues(value, {1.0}), "a wrong bound value count is refused");
}

// Under a topologyFollowsValues operation a binding's numbers move elements that decide
// counts, so they stay in the key.
void testBindingsUnderTopologyAreStructure() {
  const auto bound = [](float gain) {
    auto value = culled(0.25, true);
    spectrum::Binding audio; audio.fieldStableId = 21; audio.gain = gain;
    value.spectrumFields.push_back(audio);
    return splitPlanStructure(value);
  };
  check(!samePlanStructure(*bound(1), *bound(2)), "a gain edit under a cull moves the key");
  check(bound(1)->values.empty(), "a binding under a cull has no values");
}
} // namespace

int main() {
  testValueEditKeepsKey();
  testCollisionIsNotTheSameStructure();
  testSdfDigestMatchesItsBytes();
  testTopologyEditMovesKey();
  testValuesReplayTheEdit();
  testEvaluatedDataIsNotStructure();
  testTransportKeepsTheSplit();
  testGeneratorSizesAreValues();
  testValueDependentTopologyTakesItsInputs();
  testRetainedMeshContentIsStructure();
  testBoundPlansTakeTheValuePath();
  testBindingsUnderTopologyAreStructure();
  if (failures != 0) {
    std::fprintf(stderr, "%d geometry plan structure check(s) failed\n", failures);
    return 1;
  }
  std::puts("geometry plan structure tests passed");
  return 0;
}
