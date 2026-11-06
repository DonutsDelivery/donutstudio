// A paired particle body may carry its whole-object medium (slot 0). Its plan
// shape admits the medium, its wire must equal the admitted schedule lowered
// with that medium, and slot 0 needs geometry without a material table.

#include "../../shared/GeometryCoreSlotMedia.h"

#include <cstdio>
#include <string>

namespace {
int failures = 0;
void check(bool condition, const char *what) {
  if (condition) return;
  ++failures;
  std::fprintf(stderr, "FAILED: %s\n", what);
}

HarmonicMIDI::grid::SceneVolume glassTint() {
  HarmonicMIDI::grid::SceneVolume volume;
  volume.density = 1.5f;
  volume.color = {0.2f, 0.6f, 0.9f};
  volume.absorptionColor = {0.0f, 0.1f, 0.3f};
  return volume;
}

std::string text(const std::vector<std::uint8_t> &bytes) { return {bytes.begin(), bytes.end()}; }

void planShapeAdmitsMedia() {
  using namespace videowire::geometry;
  LoweredRuntimePlan body;
  body.contract.carrier = CarrierKind::instances3D;
  body.runtimeValue.assign(21, 0);
  check(pairedBodyPlanShape(body), "an instances-only body admits");
  body.slotMedia = text(encodeSlotMedia({{0, glassTint()}}));
  check(pairedBodyPlanShape(body), "a body with its whole-object medium admits");

  auto surfaced = body;
  surfaced.surfaceMaterial = "surface";
  check(!pairedBodyPlanShape(surfaced), "a body with a Surface binding is refused");
  auto meshed = body;
  meshed.contract.carrier = CarrierKind::geometry3D;
  check(!pairedBodyPlanShape(meshed), "a mesh plan is not a paired body");
  auto shortValue = body;
  shortValue.runtimeValue.resize(20);
  check(!pairedBodyPlanShape(shortValue), "a value shorter than its schedule header is refused");
}

void wireMatchesWithMedium() {
  using namespace videowire::geometry;
  std::string error;
  PortContract contract;
  contract.carrier = CarrierKind::geometry3D;
  contract.maxVertices = 64;
  contract.maxIndices = 192;
  contract.maxAttributes = 4;
  const auto grid = lowerGrid(11, 12, 1, 3, 3, 0.5f, contract, error);
  const auto value = grid ? admitValue(*grid, contract, error) : std::nullopt;
  check(value.has_value(), "the grid admits");
  if (!value) { std::fprintf(stderr, "%s\n", error.c_str()); return; }

  const auto media = text(encodeSlotMedia({{0, glassTint()}}));
  const auto bytes = lowerRuntimePlan(contract, *value, {}, media);
  const auto plan = decodeLoweredRuntimePlan(bytes, error);
  check(plan && plan->slotMedia == media, "the lowered plan keeps the body's medium");
  if (!plan) return;
  check(pairedBodyWireExact(contract, *value, *plan, bytes), "the admitted schedule with its medium matches exactly");
  check(!pairedBodyWireExact(contract, *value, *plan, lowerRuntimePlan(contract, *value)),
        "a wire without the medium differs");
  auto other = *plan;
  auto denser = glassTint();
  denser.density = 3.0f;
  other.slotMedia = text(encodeSlotMedia({{0, denser}}));
  check(!pairedBodyWireExact(contract, *value, other, bytes), "a different medium differs");

  // A body without media keeps its released version 1 bytes.
  const auto plain = lowerRuntimePlan(contract, *value);
  const auto plainPlan = decodeLoweredRuntimePlan(plain, error);
  check(plainPlan && plainPlan->slotMedia.empty() && plain[4] == 1
            && pairedBodyWireExact(contract, *value, *plainPlan, plain),
        "a body without media keeps version 1 and matches");
}

void slotZeroNeedsNoTable() {
  using namespace videowire::geometry;
  OperationRecord table;
  table.code = OperationCode::materialTableSlot;
  table.secondaryInputStableId = 7;
  const std::vector<OperationRecord> withTable{table}, withoutTable{};
  const SlotMedia whole{{0, glassTint()}}, slot{{7, glassTint()}}, stray{{8, glassTint()}};
  check(slotMediaOwned(whole, withoutTable), "slot 0 is owned by geometry without a table");
  check(!slotMediaOwned(whole, withTable), "slot 0 is refused on geometry with a table");
  check(slotMediaOwned(slot, withTable), "a table slot owns its medium");
  check(!slotMediaOwned(stray, withTable), "a medium without its table slot is refused");
  check(!slotMediaOwned(slot, withoutTable), "a slot medium without a table is refused");
  check(slotMediaOwned({}, withTable), "no media need no owner");
}
} // namespace

int main() {
  planShapeAdmitsMedia();
  wireMatchesWithMedium();
  slotZeroNeedsNoTable();
  if (failures) return 1;
  std::puts("geometry core paired media tests passed");
  return 0;
}
