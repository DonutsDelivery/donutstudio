// The lowered Geometry Core plan carries its material table slots' media to the
// reactive helper path. Plans without media keep their version 1 and 2 bytes,
// and the slot media wire refuses inactive, unordered or trailing entries.

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

HarmonicMIDI::grid::SceneVolume fog(float density) {
  HarmonicMIDI::grid::SceneVolume volume;
  volume.density = density;
  volume.color = {0.9f, 0.6f, 0.3f};
  volume.absorptionColor = {0.1f, 0.2f, 0.0f};
  volume.anisotropy = 0.4f;
  volume.emission = {0.0f, 0.5f, 2.0f};
  volume.field = HarmonicMIDI::grid::SceneVolumeField::Noise;
  volume.fieldDetail = 5;
  volume.fieldScale = 2.5f;
  volume.fieldOffset = {1.0f, -2.0f, 3.0f};
  volume.volumeOnly = true;
  return volume;
}

std::string text(const std::vector<std::uint8_t> &bytes) { return {bytes.begin(), bytes.end()}; }

void slotMediaRoundTrips() {
  using namespace videowire::geometry;
  std::string error;
  SlotMedia media{{3, fog(0.8f)}, {9, HarmonicMIDI::grid::SceneVolume{}}};
  media[9].density = 4.0f;
  const auto bytes = encodeSlotMedia(media);
  const auto decoded = decodeSlotMedia(text(bytes), error);
  check(decoded && decoded->size() == 2, "two slot media decode");
  if (decoded)
    for (const auto &[slot, volume] : media)
      check(decoded->count(slot) && HarmonicMIDI::grid::sameSceneVolume(decoded->at(slot), volume),
            "each slot medium decodes bit for bit");
  check(encodeSlotMedia({}).empty(), "no media encodes nothing");
  check(decodeSlotMedia(std::string{}, error) && decodeSlotMedia(std::string{}, error)->empty(),
        "no bytes decode as no media");

  auto trailing = text(bytes);
  trailing.push_back('\0');
  check(!decodeSlotMedia(trailing, error), "trailing bytes are refused");
  check(!decodeSlotMedia(text(bytes).substr(0, bytes.size() - 1), error), "a truncated medium is refused");
  check(!decodeSlotMedia(text(encodeSlotMedia({{3, HarmonicMIDI::grid::SceneVolume{}}})), error),
        "a medium at density 0 is refused");
  // Slot 0 is a Surface override's whole-object medium, as a paired body carries.
  const auto whole = decodeSlotMedia(text(encodeSlotMedia({{0, fog(1.0f)}, {4, fog(2.0f)}})), error);
  check(whole && whole->count(0) && whole->count(4), "slot 0 carries a whole-object medium");
  // Swap the two entries' slot ids so they arrive out of order.
  auto swapped = bytes;
  const std::size_t entry = (bytes.size() - 8) / 2;
  for (std::size_t i = 0; i < 8; ++i) std::swap(swapped[8 + i], swapped[8 + entry + i]);
  check(!decodeSlotMedia(text(swapped), error), "slots out of order are refused");
}

void loweredPlanCarriesMedia() {
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

  const auto plain = lowerRuntimePlan(contract, *value);
  const auto surfaced = lowerRuntimePlan(contract, *value, "surface");
  check(plain == lowerRuntimePlan(contract, *value, {}, {}), "a plan without media keeps its version 1 bytes");
  check(surfaced == lowerRuntimePlan(contract, *value, "surface", {}), "a Surface plan keeps its version 2 bytes");
  check(plain[4] == 1 && surfaced[4] == 2, "released plans keep versions 1 and 2");

  const auto media = text(encodeSlotMedia({{7, fog(1.5f)}}));
  for (const std::string &surface : {std::string{}, std::string{"surface"}}) {
    const auto bytes = lowerRuntimePlan(contract, *value, surface, media);
    check(bytes[4] == 3, "a plan with media is version 3");
    const auto decoded = decodeLoweredRuntimePlan(bytes, error);
    check(decoded && decoded->surfaceMaterial == surface && decoded->slotMedia == media,
          "version 3 keeps its Surface binding and its slot media");
    check(decoded && decoded->runtimeValue == encodeRuntimeValue(*value), "version 3 keeps the runtime value");
    auto cut = bytes;
    cut.pop_back();
    check(!decodeLoweredRuntimePlan(cut, error), "a truncated version 3 plan is refused");
  }
  const auto old = decodeLoweredRuntimePlan(surfaced, error);
  check(old && old->surfaceMaterial == "surface" && old->slotMedia.empty(), "version 2 decodes without media");
}
} // namespace

int main() {
  slotMediaRoundTrips();
  loweredPlanCarriesMedia();
  if (failures) return 1;
  std::puts("geometry core slot media tests passed");
  return 0;
}
