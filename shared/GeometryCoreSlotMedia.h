#pragma once

// Media of a Geometry Core material table's slots, as the lowered plan carries
// them to the reactive helper path. Each entry names its slot's source material
// and holds the Volume Material or the Surface Material glass or water medium
// the scene path gives the same slot. Slot 0 is the medium of a Surface override
// on geometry without a table, which fills every draw. Entries are in ascending
// slot order and every one has density above 0, so a plan without media
// encodes nothing and keeps its released bytes.

#include "GeometryCoreTransport.h"
#include "Visual3DScene.h"

#include <algorithm>
#include <map>

namespace videowire::geometry {

using SlotMedia = std::map<StableId, HarmonicMIDI::grid::SceneVolume>;

inline std::vector<std::uint8_t> encodeSlotMedia(const SlotMedia &media) {
  transport_detail::Writer w;
  if (media.empty())
    return {};
  w.integer<std::uint64_t>(media.size());
  const auto vec3 = [&](const HarmonicMIDI::grid::SceneVec3 &v) {
    w.floating(v.x); w.floating(v.y); w.floating(v.z);
  };
  for (const auto &[slot, volume] : media) {
    w.integer<std::uint64_t>(slot);
    w.floating(volume.density);
    vec3(volume.color);
    vec3(volume.absorptionColor);
    w.floating(volume.anisotropy);
    vec3(volume.emission);
    w.integer(static_cast<std::uint8_t>(volume.field));
    w.integer<std::uint32_t>(volume.fieldDetail);
    w.floating(volume.fieldScale);
    vec3(volume.fieldOffset);
    w.integer<std::uint8_t>(volume.volumeOnly ? 1 : 0);
    w.floating(volume.radius);
  }
  return std::move(w.bytes);
}

inline std::optional<SlotMedia>
decodeSlotMedia(const std::uint8_t *data, std::size_t size, std::string &error) {
  // 8 slot + 4 density + 36 colours + 4 anisotropy + 1 field + 4 detail
  // + 4 scale + 12 offset + 1 volume-only + 4 radius.
  static constexpr std::size_t entryBytes = 8 + 4 + 36 + 4 + 1 + 4 + 4 + 12 + 1 + 4;
  SlotMedia media;
  if (size == 0)
    return media;
  transport_detail::Reader r(data, size);
  std::size_t count = 0;
  if (!r.count(count, size / entryBytes) || count == 0 || !r.arrayBytes(count, entryBytes)) {
    error = "Geometry Core slot media count is invalid";
    return std::nullopt;
  }
  StableId previous = 0;
  for (std::size_t i = 0; i < count; ++i) {
    StableId slot = 0;
    HarmonicMIDI::grid::SceneVolume volume;
    std::uint8_t field = 0, volumeOnly = 0;
    const auto vec3 = [&](HarmonicMIDI::grid::SceneVec3 &v) {
      return r.floating(v.x) && r.floating(v.y) && r.floating(v.z);
    };
    if (!r.integer(slot) || !r.floating(volume.density) || !vec3(volume.color)
        || !vec3(volume.absorptionColor) || !r.floating(volume.anisotropy)
        || !vec3(volume.emission) || !r.integer(field) || !r.integer(volume.fieldDetail)
        || !r.floating(volume.fieldScale) || !vec3(volume.fieldOffset)
        || !r.integer(volumeOnly) || !r.floating(volume.radius)) {
      error = r.error;
      return std::nullopt;
    }
    volume.field = static_cast<HarmonicMIDI::grid::SceneVolumeField>(field);
    volume.volumeOnly = volumeOnly == 1;
    if ((i > 0 && slot <= previous) || volumeOnly > 1
        || field > static_cast<std::uint8_t>(HarmonicMIDI::grid::SceneVolumeField::Noise)
        || !HarmonicMIDI::grid::visual3d_detail::validVolume(volume)
        || !HarmonicMIDI::grid::sceneVolumeActive(volume)) {
      error = "Geometry Core slot medium is invalid or out of slot order";
      return std::nullopt;
    }
    previous = slot;
    media.emplace(slot, volume);
  }
  if (r.remaining != 0) {
    error = "Geometry Core slot media has trailing bytes";
    return std::nullopt;
  }
  return media;
}

inline std::optional<SlotMedia> decodeSlotMedia(const std::string &bytes, std::string &error) {
  return decodeSlotMedia(reinterpret_cast<const std::uint8_t *>(bytes.data()), bytes.size(), error);
}

// Every medium has an exact owner: the table slot that names its source
// material, or for slot 0, a Surface override's medium, geometry without a table.
inline bool slotMediaOwned(const SlotMedia &media, const std::vector<OperationRecord> &operations) {
  const auto owns = [&](StableId slot) {
    return std::any_of(operations.begin(), operations.end(), [&](const auto &op) {
      return op.code == OperationCode::materialTableSlot && (slot == 0 || op.secondaryInputStableId == slot);
    });
  };
  for (const auto &medium : media)
    if (medium.first == 0 ? owns(0) : !owns(medium.first)) return false;
  return true;
}

// A paired particle body's lowered plan is an instances-only schedule with no
// Surface binding or attributes. It may carry its whole-object medium (slot 0).
inline bool pairedBodyPlanShape(const LoweredRuntimePlan &plan) {
  return plan.surfaceMaterial.empty() && plan.contract.attributes.empty()
      && plan.contract.carrier == CarrierKind::instances3D && plan.runtimeValue.size() >= 21;
}

// The schedule never names the medium, so the body's exact bytes are its
// admitted schedule lowered with the plan's own slot media and Render 3D settings.
inline bool pairedBodyWireExact(const PortContract &contract, const AdmittedValue &admitted,
                                const LoweredRuntimePlan &plan, const std::vector<std::uint8_t> &bytes) {
  return lowerRuntimePlan(contract, admitted, {}, plan.slotMedia, false, plan.renderSettings) == bytes;
}

} // namespace videowire::geometry
