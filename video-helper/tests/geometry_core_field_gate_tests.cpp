#include "../../shared/GeometryCore.h"
#include "../../shared/GeometryCoreTransport.h"

#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// Sample Index (operation 137): the writer stamps the version that introduced it, and
// replay refuses a forged domain or index values that fall outside the source attribute.
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
  contract.maxVertices = 1024;
  contract.maxIndices = 6144;
  contract.maxAttributes = 32;
  return contract;
}

PortContract sampleContract() {
  PortContract contract;
  contract.carrier = CarrierKind::field;
  contract.fieldDomain = Domain::vertex;
  contract.fieldValueType = ValueType::integer;
  contract.fieldInterpolation = Interpolation::constant;
  contract.maxFieldElements = 1024;
  contract.maxAttributes = 32;
  return contract;
}

// A retained vertex attribute holding each vertex's own index, plus its vertex count.
struct SampleSource {
  ValueDescriptor stored;
  std::size_t vertexCount = 0;
};

std::optional<SampleSource> lowerSampleSource(std::string &error) {
  auto mesh = lowerGrid(8820, 8820, 1, 2, 2, 1, meshContract(), error);
  if (!mesh) return std::nullopt;
  SampleSource source;
  source.vertexCount = std::get<GeometryData>(mesh->data).positions.size();
  FieldOperationSettings nearest;
  nearest.domain = Domain::vertex;
  nearest.valueType = ValueType::integer;
  auto identity = lowerNearestField(*mesh, *mesh, 8821, OperationCode::fieldNearestIndex, nearest,
                                    sampleContract(), error);
  if (!identity) return std::nullopt;
  auto stored = lowerNamedAttribute(*mesh, &*identity, 8822, OperationCode::storeNamedAttribute,
                                    "value", ValueType::integer, meshContract(), error);
  if (!stored) return std::nullopt;
  source.stored = std::move(*stored);
  return source;
}

// A constant integer index field with the given first value and the source's element count.
std::optional<ValueDescriptor> lowerIndex(std::size_t count, double value, std::string &error) {
  FieldOperationSettings constant;
  constant.domain = Domain::vertex;
  constant.valueType = ValueType::integer;
  constant.values = {value, 0, 0, 0, static_cast<double>(count), 0, 0, 0};
  return lowerConstructedField(8823, OperationCode::fieldConstant, constant, nullptr, nullptr,
                               nullptr, sampleContract(), error);
}

std::optional<ValueDescriptor> lowerSample(const SampleSource &source, const ValueDescriptor &index,
                                           std::string &error) {
  FieldOperationSettings sample;
  sample.domain = Domain::vertex;
  sample.valueType = ValueType::integer;
  sample.attributeName = "value";
  return lowerSampleIndexField(source.stored, index, 8824, sample, sampleContract(), error);
}

// A sampled value whose index is the last source element, admitted and ready to forge.
std::optional<ValueDescriptor> lowerValidSample(SampleSource &source, std::string &error) {
  auto sourceOpt = lowerSampleSource(error);
  if (!sourceOpt) return std::nullopt;
  source = std::move(*sourceOpt);
  auto index = lowerIndex(source.vertexCount, 2, error);
  if (!index) return std::nullopt;
  return lowerSample(source, *index, error);
}

// Forges the constant index field's first element in an already lowered sample and replays it.
// Returns true when the forged value is found and admission refuses it, with the refusal in error.
bool replayRefusesForgedIndex(const ValueDescriptor &sampled, double index, std::string &error) {
  auto forged = sampled;
  bool found = false;
  for (auto &op : forged.operations) {
    if (op.code == OperationCode::fieldConstant) {
      op.field.values[0] = index;
      found = true;
    }
  }
  error.clear();
  return found && !admitValue(forged, sampleContract(), error);
}

// The writer stamps the version that introduced operation 137, not the current version.
void sampleIndexWriterStampsIntroducingVersion() {
  std::string error;
  SampleSource source;
  auto sampled = lowerValidSample(source, error);
  check(sampled.has_value(), "sample index lowers for the version stamp check");
  if (!sampled) {
    std::fprintf(stderr, "lowering error: %s\n", error.c_str());
    return;
  }
  auto admitted = admitValue(*sampled, sampleContract(), error);
  check(admitted.has_value(), "sample index admits for the version stamp check");
  if (!admitted) return;
  const auto bytes = encodeRuntimeValue(*admitted);
  check(bytes.size() > 5 && bytes[4] == kSampleIndexTransportVersion && bytes[5] == 0,
        "a sample index only graph is written at transport version 15");
  check(kSampleIndexTransportVersion < kTransportVersion,
        "sample index does not bump the current transport version");
  error.clear();
  auto reopened = decodeRuntimeValue(bytes, sampleContract(), {}, {}, error);
  check(reopened.has_value() && encodeRuntimeValue(*reopened) == bytes,
        "a version 15 sample index roundtrips");
}

// The sample operation's own domain must match its source attribute; a forged point domain is refused.
void sampleIndexReplayRefusesForgedDomain() {
  std::string error;
  SampleSource source;
  auto sampled = lowerValidSample(source, error);
  check(sampled.has_value(), "sample index lowers for the domain forgery check");
  if (!sampled) return;
  auto forged = *sampled;
  bool found = false;
  for (auto &op : forged.operations) {
    if (op.code == OperationCode::fieldSampleIndex) {
      op.field.domain = Domain::point;
      found = true;
    }
  }
  check(found, "sample index operation is present to forge");
  error.clear();
  const bool refused = !admitValue(forged, sampleContract(), error);
  check(refused && error == "Sample index requires a retained source attribute and an Integer index field",
        "sample index replay refuses an operation domain that differs from its source attribute");
}

// Forged index values that fall outside the source attribute are refused by replay, not read.
void sampleIndexReplayRefusesForgedOutOfRangeIndices() {
  std::string error;
  SampleSource source;
  auto sampled = lowerValidSample(source, error);
  check(sampled.has_value(), "sample index lowers for the out-of-range forgery check");
  if (!sampled) return;
  const auto count = static_cast<double>(source.vertexCount);

  error.clear();
  check(replayRefusesForgedIndex(*sampled, count, error)
            && error == "Sample index is outside the source attribute",
        "sample index replay refuses an index equal to the source element count");
  error.clear();
  check(replayRefusesForgedIndex(*sampled, -1, error)
            && error == "Sample index is outside the source attribute",
        "sample index replay refuses a negative index");
  error.clear();
  check(replayRefusesForgedIndex(*sampled, 1.5, error)
            && error == "Sample index is outside the source attribute",
        "sample index replay refuses a fractional index");
  // A non-finite index is refused earlier, by the constant field's own validation.
  error.clear();
  check(replayRefusesForgedIndex(*sampled, std::numeric_limits<double>::quiet_NaN(), error)
            && !error.empty(),
        "sample index replay refuses a non-finite index");

  // The untouched value still admits, so the refusals above come from the forged values alone.
  error.clear();
  check(admitValue(*sampled, sampleContract(), error).has_value(),
        "the unforged sample index value still admits");
}

// Lowering refuses the same out-of-range indices before they can reach a transport value.
void sampleIndexLoweringRefusesOutOfRangeIndices() {
  std::string error;
  SampleSource source;
  auto sourceOpt = lowerSampleSource(error);
  check(sourceOpt.has_value(), "sample index source lowers for the lowering refusal check");
  if (!sourceOpt) return;
  source = std::move(*sourceOpt);

  for (const double value : {static_cast<double>(source.vertexCount), -1.0}) {
    error.clear();
    auto index = lowerIndex(source.vertexCount, value, error);
    check(index.has_value(), "an out-of-range constant index still lowers as a field");
    if (!index) continue;
    error.clear();
    check(!lowerSample(source, *index, error).has_value()
              && error == "Sample index is outside the source attribute",
          "lowering refuses a sample index outside the source attribute");
  }
}
}  // namespace

int main() {
  sampleIndexWriterStampsIntroducingVersion();
  sampleIndexReplayRefusesForgedDomain();
  sampleIndexReplayRefusesForgedOutOfRangeIndices();
  sampleIndexLoweringRefusesOutOfRangeIndices();
  if (failures != 0) {
    std::fprintf(stderr, "%d geometry field gate check(s) failed\n", failures);
    return 1;
  }
  std::printf("geometry core field gate tests passed\n");
  return 0;
}
