#include "../../shared/GeometryCore.h"
#include "../../shared/GeometryCoreTransport.h"
#include "../src/beat_timeline.h"

#include <cmath>
#include <cstdio>
#include <optional>
#include <string>

namespace {
using namespace videowire::geometry;

int failures = 0;

void check(bool condition, const std::string& message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    ++failures;
  }
}

// Band zero rises one tenth per second, so a read shows the time it came from.
const spectrum::FeaturesAt ramp = [](double seconds) {
  spectrum::Bands bands {};
  bands[0] = static_cast<float>(seconds / 10);
  return bands;
};

spectrum::BeatDelay delayThrough(const videotime::BeatTimeline& timeline) {
  return [timeline](double seconds, double beats) { return timeline.secondsBeforeBeats(seconds, beats); };
}

spectrum::Binding effector() {
  spectrum::Binding binding;
  binding.target = spectrum::Target::geometryDeformation;
  binding.firstBand = 0; binding.lastBand = 0;
  return binding;
}

void tempoMapMovesBeatsAcrossTempoChanges() {
  videotime::BeatTimeline timeline;
  timeline.set({{0, 120, true}, {4, 60, true}}, {{0, 4, 4}});
  check(std::abs(timeline.secondsBeforeBeats(3, 2) - 2) < 1e-12, "two beats at 120 BPM are one second");
  check(std::abs(timeline.secondsBeforeBeats(4, 2) - 2) < 1e-12,
      "two beats at 60 BPM after the tempo change are two seconds");
  check(std::abs(timeline.secondsBeforeBeats(2.5, 2) - 1.5) < 1e-12,
      "a delay across the tempo change takes one beat from each tempo");
  check(timeline.secondsBeforeBeats(0.5, 2) < 0, "a delay before beat zero has no source time");
}

void effectorReadsTheDelayedBeat() {
  videotime::BeatTimeline timeline;
  timeline.reset(120, 4);
  auto binding = effector();
  binding.deformer.delayBeats = 2;
  spectrum::ScalarInput scalars;
  scalars.beatDelay = delayThrough(timeline);
  spectrum::EffectorFrame sample;
  check(spectrum::readEffectorFrame(binding, {}, ramp, scalars, 3, sample) && std::abs(sample.bands[0] - 0.2f) < 1e-6f,
      "two beats at 120 BPM read the analysis one second earlier");
  check(spectrum::readEffectorFrame(binding, {}, ramp, scalars, 0.5, sample) && sample.bands[0] == 0,
      "before two beats have passed the effector reads silence");
  scalars.beatDelay = {};
  check(spectrum::readEffectorFrame(binding, {}, ramp, scalars, 3, sample) && sample.bands[0] == 0,
      "without a tempo map a beat delay never reads undelayed analysis");
}

void beatsMatchTheSameDelayInSeconds() {
  videotime::BeatTimeline timeline;
  timeline.reset(90, 4);
  auto beats = effector();
  beats.deformer.delayBeats = 3;
  auto seconds = effector();
  seconds.deformer.delaySeconds = 2;
  spectrum::ScalarInput scalars;
  scalars.beatDelay = delayThrough(timeline);
  spectrum::Follower beatState, secondState;
  spectrum::Bands beatResult {}, secondResult {};
  std::string error;
  bool equal = true;
  for (const double time : {0.5, 1.9, 2.0, 3.25, 7.0}) {
    const bool a = spectrum::evaluate(beats, ramp(time), time, ramp, beatState, beatResult, error, scalars);
    const bool b = spectrum::evaluate(seconds, ramp(time), time, ramp, secondState, secondResult, error, scalars);
    for (std::size_t band = 0; band < beatResult.size(); ++band)
      equal = equal && std::abs(beatResult[band] - secondResult[band]) <= 1e-6f;
    equal = equal && a && b;
  }
  check(equal, "three beats at 90 BPM drive the effector like two seconds: " + error);
}

void evaluationRefusesAMissingTempoMap() {
  auto binding = effector();
  binding.deformer.delayBeats = 1;
  spectrum::Follower state;
  spectrum::Bands result {};
  std::string error;
  check(!spectrum::evaluate(binding, ramp(2), 2, ramp, state, result, error)
          && error == spectrum::kBeatDelayUnavailable,
      "a beat delay without the project tempo map is refused by name");
  videotime::BeatTimeline timeline;
  spectrum::ScalarInput scalars;
  scalars.beatDelay = delayThrough(timeline);
  error.clear();
  check(!spectrum::evaluate(binding, ramp(2), 2, {}, state, result, error, scalars)
          && error == spectrum::kBeatDelayUnavailable,
      "a beat delay without baked analysis history is refused by name");
}

PortContract meshContract() {
  PortContract contract;
  contract.carrier = CarrierKind::geometry3D; contract.overflow = OverflowPolicy::reject;
  contract.maxVertices = 100; contract.maxIndices = 600;
  return contract;
}

std::optional<ValueDescriptor> deformed(float delaySeconds, float delayBeats, std::string& error) {
  const auto source = lowerGrid(9410, 9410, 1, 9, 9, 0.25f, meshContract(), error);
  if (!source) return std::nullopt;
  spectrum::Binding binding;
  binding.firstBand = 0; binding.lastBand = 23;
  binding.deformer.delaySeconds = delaySeconds;
  binding.deformer.delayBeats = delayBeats;
  return lowerAudioDeformer(*source, 9411, binding, meshContract(), error);
}

std::uint16_t headerVersion(const std::vector<std::uint8_t>& bytes) {
  return bytes.size() > 5 ? static_cast<std::uint16_t>(bytes[4] | (bytes[5] << 8)) : 0;
}

void transportCarriesBeatsAndKeepsOldBytes() {
  std::string error;
  const auto plain = deformed(0.5f, 0, error);
  const auto beats = deformed(0, 2.5f, error);
  check(plain && beats, "seconds and beats deformers lower: " + error);
  if (!plain || !beats) return;
  const auto admittedPlain = admitValue(*plain, meshContract(), error);
  const auto admittedBeats = admitValue(*beats, meshContract(), error);
  check(admittedPlain && admittedBeats, "seconds and beats deformers admit: " + error);
  if (!admittedPlain || !admittedBeats) return;
  const auto plainBytes = encodeRuntimeValue(*admittedPlain);
  const auto beatBytes = encodeRuntimeValue(*admittedBeats);
  check(headerVersion(plainBytes) < kEffectorBeatDelayTransportVersion,
      "a delay in seconds keeps the transport version it had before beats");
  check(headerVersion(beatBytes) == kEffectorBeatDelayTransportVersion
          && kEffectorBeatDelayTransportVersion == 20 && kEffectorBeatDelayTransportVersion < kTransportVersion,
      "a delay in beats is written at transport version 20");
  const auto reopened = decodeRuntimeValue(beatBytes, meshContract(), {}, {}, error);
  check(reopened && reopened->descriptor().spectrumFields.size() == 1
          && reopened->descriptor().spectrumFields.front().deformer.delayBeats == 2.5f
          && reopened->descriptor().spectrumFields.front().deformer.delaySeconds == 0
          && encodeRuntimeValue(*reopened) == beatBytes,
      "the beat delay survives a transport round trip: " + error);
}

void loweringRefusesBadBeatDelays() {
  std::string error;
  check(!deformed(0.5f, 1, error), "a delay in both seconds and beats is refused");
  check(!deformed(0, 65, error), "a delay past 64 beats is refused");
  check(!deformed(0, -1, error), "a negative beat delay is refused");
}
}  // namespace

int main() {
  tempoMapMovesBeatsAcrossTempoChanges();
  effectorReadsTheDelayedBeat();
  beatsMatchTheSameDelayInSeconds();
  evaluationRefusesAMissingTempoMap();
  transportCarriesBeatsAndKeepsOldBytes();
  loweringRefusesBadBeatDelays();
  if (failures != 0) return 1;
  std::puts("effector beat delay tests passed");
  return 0;
}
