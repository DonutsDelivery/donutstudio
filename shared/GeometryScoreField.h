#pragma once

#include "VisualNoteInstancingContract.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <new>
#include <set>
#include <string>
#include <vector>

namespace videowire::geometry::scorefield {
inline constexpr const char* kNodeKind = "visual.geometry.score-field";
inline constexpr const char* kInstanceNodeKind = "visual.geometry.score-field.instance.vector";
enum class Mode : std::uint8_t {
  closestNoteDistance, velocityInfluence, pitchSpacing, primeLattice, harmonicLinkDistance,
  pitchBendSemitones, noteAgeBeats, remainingDurationBeats, rootFlag, midiNoteNumber,
  resolvedFrequencyHz, harmonicLinkRatio,
  // The track lane the nearest note sits on (its stable track id).
  midiLane
};

// Score Sample Field operations store choice numbers, not Mode values. Choices
// 0 through 3 are released and keep their meaning, so choice 3 is Harmonic Link
// Distance rather than Prime Lattice. Choices 11 through 13 are the X, Y and Z
// Prime Lattice components at the vector mode's default primes 2, 3 and 5.
// Choice 14 is MIDI Lane.
inline constexpr std::uint8_t kFirstSamplePrimeLatticeChoice = 11;
inline constexpr std::uint8_t kSamplePrimeLatticeChoiceEnd = 14;
inline constexpr std::uint8_t kSampleFieldModeCount = 15;
inline bool sampleFieldMode(std::uint8_t choice, Mode& mode) noexcept {
  static constexpr std::array<Mode, kSampleFieldModeCount> measures {
    Mode::closestNoteDistance, Mode::velocityInfluence, Mode::pitchSpacing, Mode::harmonicLinkDistance,
    Mode::pitchBendSemitones, Mode::noteAgeBeats, Mode::remainingDurationBeats, Mode::rootFlag,
    Mode::midiNoteNumber, Mode::resolvedFrequencyHz, Mode::harmonicLinkRatio,
    Mode::primeLattice, Mode::primeLattice, Mode::primeLattice, Mode::midiLane
  };
  if (choice >= kSampleFieldModeCount) return false;
  mode = measures[choice];
  return true;
}
// Prime Lattice choices read one Binding::primes slot, the slot the vector mode
// reads for that axis. Slot 0 is X, slot 1 is Y and slot 2 is Z.
inline bool samplePrimeLatticeSlot(std::uint8_t choice, std::uint8_t& slot) noexcept {
  if (choice < kFirstSamplePrimeLatticeChoice || choice >= kSamplePrimeLatticeChoiceEnd) return false;
  slot = static_cast<std::uint8_t>(choice - kFirstSamplePrimeLatticeChoice);
  return true;
}

// Only authored controls and immutable geometry samples travel with the plan.
// Notes and links come from the renderer's canonical Block C frame.
struct Binding final {
  std::uint64_t fieldStableId = 0, sourceGeometryStableId = 0;
  std::uint64_t consumerGeometryStableId = 0, scoreSourceStableId = 0;
  Mode mode = Mode::velocityInfluence;
  visualnoteinstancing::Mapping mapping;
  float radius = 1.0f, gain = 1.0f, bias = 0.0f;
  std::array<float, 3> direction {0.0f, 0.0f, 1.0f};
  std::array<std::uint8_t, 3> primes {0, 1, 2};
  std::vector<std::array<float, 3>> positions;
  std::vector<std::uint64_t> elementIds;
};

// Sample buffers are sized from the input geometry. The only refusal is a real
// allocation failure, and the error names the node that asked for the memory.
inline bool reserveSamples(Binding& binding, std::size_t count, const char* node, std::string& error) {
  try {
    binding.positions.reserve(count); binding.elementIds.reserve(count);
  } catch (const std::bad_alloc&) {
    error = std::string(node) + " could not allocate memory for " + std::to_string(count) + " geometry samples";
    return false;
  }
  return true;
}

inline bool validate(const Binding& binding, std::size_t cardinality, std::string& error) {
  const visualnoteinstancing::Mapping defaults;
  if (binding.fieldStableId == 0 || binding.sourceGeometryStableId == 0
      || binding.scoreSourceStableId == 0 || !visualnoteinstancing::valid(binding.mapping)
      || binding.mapping.schemaVersion != defaults.schemaVersion || binding.mapping.includeMuted
      || binding.mapping.meshScale != defaults.meshScale
      || binding.mapping.appearanceSource != defaults.appearanceSource
      || binding.mapping.appearanceLow != defaults.appearanceLow
      || binding.mapping.appearanceHigh != defaults.appearanceHigh
      || static_cast<unsigned>(binding.mode) > static_cast<unsigned>(Mode::midiLane)
      || !std::isfinite(binding.radius) || binding.radius <= 0 || binding.radius > 10000
      || !std::isfinite(binding.gain) || std::abs(binding.gain) > 1000
      || !std::isfinite(binding.bias) || std::abs(binding.bias) > 1000
      || cardinality == 0
      || binding.positions.size() != cardinality || binding.elementIds.size() != cardinality) {
    error = "Score Field has invalid controls or no geometry samples";
    return false;
  }
  for (auto prime : binding.primes)
    if (prime >= 6) { error = "Score Field prime must be one of 2, 3, 5, 7, 11 or 13"; return false; }
  for (auto component : binding.direction)
    if (!std::isfinite(component) || std::abs(component) > 1000) {
      error = "Score Field direction exceeds its finite bounds"; return false;
    }
  std::set<std::uint64_t> ids;
  for (std::size_t index = 0; index < cardinality; ++index) {
    if (binding.elementIds[index] == 0 || !ids.insert(binding.elementIds[index]).second) {
      error = "Score Field requires unique stable geometry element identities"; return false;
    }
    for (auto component : binding.positions[index])
      if (!std::isfinite(component) || std::abs(component) > 1000000) {
        error = "Score Field sample position exceeds its finite bounds"; return false;
      }
  }
  error.clear();
  return true;
}
inline bool sameBinding(const Binding& a, const Binding& b) {
  return a.fieldStableId==b.fieldStableId && a.sourceGeometryStableId==b.sourceGeometryStableId
      && a.consumerGeometryStableId==b.consumerGeometryStableId && a.scoreSourceStableId==b.scoreSourceStableId
      && a.mode==b.mode && visualnoteinstancing::encode(a.mapping)==visualnoteinstancing::encode(b.mapping)
      && a.radius==b.radius && a.gain==b.gain && a.bias==b.bias && a.direction==b.direction
      && a.primes==b.primes && a.positions==b.positions && a.elementIds==b.elementIds;
}
} // namespace videowire::geometry::scorefield
