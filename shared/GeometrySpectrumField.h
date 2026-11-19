#pragma once

#include "../video-helper/src/block_b_defs.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <iterator>
#include <string>
#include <tuple>
#include <vector>

namespace videowire::geometry::spectrum {
inline constexpr std::size_t kBands = arbitblockb::kBandCount;
using Bands = std::array<float, kBands>;
enum class Source : std::uint8_t { master, track, group };
inline constexpr std::size_t kMaximumAnalysisSources = 8;
using SourceFeaturesAt = std::function<bool(Source, std::int32_t, double, Bands&)>;
enum class Target : std::uint8_t { vertexDisplacement, instanceTransform, geometryDeformation };
enum class Deformation : std::uint8_t { normal, vector, bulge, ripple, wave, twist, bend, taper, squash, turbulence, attraction };
// History rows and span follow the node. Depth runs to the float-exact
// integer range of its parameter, and the span to the project time range.
inline constexpr std::uint32_t kMaximumHistoryRows = 1u << 24;
inline constexpr double kMaximumHistorySeconds = 1.0e9;

struct History final {
  bool enabled = false, live = false, useTimeline = true;
  double currentTimeSeconds = 0.0;
  std::uint32_t depth = 32;
  float spacingSeconds = 0.1f;
  std::vector<float> coordinates;
};

// Live frames the ring keeps for one history: its span at 60 Hz, plus the
// frame before the oldest row for interpolation and the current frame.
inline std::size_t liveHistoryFrames(const History& history) noexcept {
  const double span = history.depth > 0 ? (history.depth - 1) * static_cast<double>(history.spacingSeconds) : 0.0;
  return static_cast<std::size_t>(std::ceil(std::max(0.0, span) * 60.0)) + 2;
}

// Lowering-time choice of the retained element coordinate that selects a
// history row. by 0 uses element index, 1-3 use position X, Y or Z mapped
// from [minimum, maximum] onto [0, 1].
struct HistoryAxis final {
  std::uint32_t by = 0;
  double minimum = -1.0, maximum = 1.0;
};

struct Deformer final {
  Deformation mode = Deformation::normal;
  std::array<float, 3> center {};
  float frequency = 4.0f, speed = 1.0f, phase = 0.0f;
  float falloffRadius = 0.0f, falloffPower = 1.0f;
  float delaySeconds = 0.0f, threshold = 0.0f, hysteresis = 0.0f;
  float peakHoldSeconds = 0.0f, quantizeSteps = 0.0f, envelopePower = 1.0f;
  // A delay in beats follows the project tempo map. It replaces delaySeconds,
  // so at most one of the two is nonzero. Zero keeps a delay in seconds.
  float delayBeats = 0.0f;
};

// What the Audio Effector gate reads each frame. bandRange is the mean of the
// selected bands. The others are analyzer scalars, which replace that mean only.
enum class EffectorInput : std::uint8_t { bandRange, rms, peak, onset };

// Per-frame analyzer scalars, each in zero to one.
struct Scalars final { float rms = 0.0f, peak = 0.0f, onset = 0.0f; };
using SourceScalarsAt = std::function<bool(Source, std::int32_t, double, Scalars&)>;

// These are controls and sampling coordinates, not another audio snapshot.
// Band values are supplied by the existing Block B analysis at evaluation time.
struct Binding final {
  Source source = Source::master;
  std::int32_t sourceTrackId = -1;
  std::uint64_t fieldStableId = 0;
  std::uint64_t sourceGeometryStableId = 0;
  std::uint64_t consumerGeometryStableId = 0;
  std::uint32_t firstBand = 0, lastBand = 63;
  bool interpolate = true, normalize = false;
  float gain = 1.0f, bias = 0.0f;
  float attackSeconds = 0.0f, releaseSeconds = 0.0f;
  std::array<float, 3> direction {0.0f, 0.0f, 1.0f};
  std::vector<float> coordinates;
  Target target = Target::vertexDisplacement;
  std::array<float, 3> baseScale {0.1f, 0.1f, 0.1f};
  std::array<float, 3> scaleResponse {0.0f, 1.0f, 0.0f};
  Deformer deformer;
  EffectorInput input = EffectorInput::bandRange;
  History history;
};

inline bool sameBinding(const Binding& a, const Binding& b) {
  const auto controls=[](const Binding& v) {
    const auto& d=v.deformer; const auto& h=v.history;
    return std::tie(v.source,v.sourceTrackId,v.fieldStableId,v.sourceGeometryStableId,v.consumerGeometryStableId,
        v.firstBand,v.lastBand,v.input,v.interpolate,v.normalize,v.gain,v.bias,v.attackSeconds,v.releaseSeconds,
        v.direction,v.coordinates,v.target,v.baseScale,v.scaleResponse,d.mode,d.center,d.frequency,
        d.speed,d.phase,d.falloffRadius,d.falloffPower,d.delaySeconds,d.threshold,d.hysteresis,
        d.peakHoldSeconds,d.quantizeSteps,d.envelopePower,d.delayBeats,h.enabled,h.live,h.useTimeline,
        h.currentTimeSeconds,h.depth,h.spacingSeconds,h.coordinates);
  };
  return controls(a)==controls(b);
}

inline bool appendOrderedBindings(std::vector<Binding>& bindings,
                                  const std::vector<Binding>& additional, std::string& error) {
  for (const auto& binding : additional) {
    if (binding.target!=Target::geometryDeformation) {
      error="Ordered geometry replay supports Audio Deformer bindings only"; return false;
    }
    const auto existing=std::find_if(bindings.begin(),bindings.end(),[&](const auto& candidate) {
      return candidate.fieldStableId==binding.fieldStableId;
    });
    if (existing!=bindings.end()) {
      if (!sameBinding(*existing,binding)) {
        error="Geometry branches disagree on their Audio Deformer binding"; return false;
      }
    } else {
      bindings.push_back(binding);
    }
  }
  return true;
}

// Attack, release, peak hold and hysteresis need a time-ordered envelope. Delay
// alone reads delayed input without memory, as it does on the current-frame path.
inline bool deformerHasEnvelopeMemory(const Binding& binding) noexcept {
  const auto& d = binding.deformer;
  return d.hysteresis > 0 || d.peakHoldSeconds > 0
      || binding.attackSeconds > 0 || binding.releaseSeconds > 0;
}

inline bool validate(const Binding& binding, std::size_t cardinality,
                     std::string& error) {
  if (binding.fieldStableId == 0 || binding.sourceGeometryStableId == 0
      || binding.source > Source::group
      || (binding.source == Source::master ? binding.sourceTrackId != -1 : binding.sourceTrackId < 0)
      || binding.firstBand > binding.lastBand || binding.lastBand >= kBands
      || !std::isfinite(binding.gain) || std::abs(binding.gain) > 1000.0f
      || !std::isfinite(binding.bias) || std::abs(binding.bias) > 1000.0f
      || !std::isfinite(binding.attackSeconds) || binding.attackSeconds < 0.0f
      || binding.attackSeconds > 2.0f || !std::isfinite(binding.releaseSeconds)
      || binding.releaseSeconds < 0.0f || binding.releaseSeconds > 2.0f
      || cardinality == 0 || binding.coordinates.size() != cardinality) {
    error = "Spectrum binding has invalid controls or element cardinality";
    return false;
  }
  for (const auto component : binding.direction)
    if (!std::isfinite(component) || std::abs(component) > 1000000.0f) {
      error = "Audio Spectrum Field displacement direction is invalid";
      return false;
    }
  for (const auto coordinate : binding.coordinates)
    if (!std::isfinite(coordinate) || coordinate < 0.0f || coordinate > 1.0f) {
      error = "Audio Spectrum Field sample coordinate is outside zero to one";
      return false;
    }
  if (binding.target != Target::vertexDisplacement && binding.target != Target::instanceTransform
      && binding.target != Target::geometryDeformation) {
    error = "Spectrum binding has an unknown target";
    return false;
  }
  const auto& history = binding.history;
  if (history.enabled && history.live && binding.source != Source::master) {
    error = "Track and group spectra use project baked analysis; live history is available only for Master";
    return false;
  }
  if (history.enabled) {
    if (history.depth < 2
        || history.depth > kMaximumHistoryRows || !std::isfinite(history.spacingSeconds)
        || history.spacingSeconds < 1.0f / 60.0f
        || (history.depth - 1) * static_cast<double>(history.spacingSeconds) > kMaximumHistorySeconds
        || !std::isfinite(history.currentTimeSeconds) || history.currentTimeSeconds < 0
        || history.currentTimeSeconds > 1.0e9 || (history.live && !history.useTimeline)
        || history.coordinates.size() != cardinality
        || !std::all_of(history.coordinates.begin(), history.coordinates.end(), [](float coordinate) {
             return std::isfinite(coordinate) && coordinate >= 0 && coordinate <= 1;
           })) {
      error = "Spectral History Field requires bounded history rows, finite coordinates and timeline time for live input";
      return false;
    }
  } else if (!history.coordinates.empty()) {
    error = "Inactive Spectral History Field cannot retain history coordinates";
    return false;
  }
  if (static_cast<std::uint8_t>(binding.input) > static_cast<std::uint8_t>(EffectorInput::onset)) {
    error = "Audio Effector input is unknown";
    return false;
  }
  if (binding.input != EffectorInput::bandRange) {
    if (binding.target != Target::geometryDeformation) {
      error = "Only Audio Effector stages choose an analyzer scalar input";
      return false;
    }
    if (binding.history.enabled && binding.history.live) {
      error = "Live Spectral History cannot use Audio Effector RMS, Peak or Onset input. Select Project Baked Analysis";
      return false;
    }
  }
  if (binding.target == Target::geometryDeformation) {
    const auto& d = binding.deformer;
    const auto bounded = [](float v, float lo, float hi) { return std::isfinite(v) && v >= lo && v <= hi; };
    if (d.mode > Deformation::attraction || !bounded(d.frequency, 0, 100)
        || !bounded(d.speed, -100, 100) || !bounded(d.phase, -100, 100)
        || !bounded(d.falloffRadius, 0, 1000000) || !bounded(d.falloffPower, 0.01f, 16)
        || !bounded(d.delaySeconds, 0, 10) || !bounded(d.delayBeats, 0, 64)
        || (d.delaySeconds > 0 && d.delayBeats > 0) || !bounded(d.threshold, 0, 1)
        || !bounded(d.hysteresis, 0, 1) || d.hysteresis > d.threshold
        || !bounded(d.peakHoldSeconds, 0, 2) || !bounded(d.quantizeSteps, 0, 128)
        || std::floor(d.quantizeSteps) != d.quantizeSteps || !bounded(d.envelopePower, 0.01f, 16)
        || !std::all_of(d.center.begin(), d.center.end(), [&](float v) { return bounded(v, -1000000, 1000000); })) {
      error = "Audio Deformer or Audio Effector controls are invalid";
      return false;
    }
    // A live window keeps only received frames, which cannot supply the 20 second
    // envelope warm-up or delayed input. Live attack and release stay as spectrum
    // smoothing, as they were before, so only these three are refused, explicitly.
    if (binding.history.enabled && binding.history.live
        && (d.peakHoldSeconds > 0 || d.hysteresis > 0 || d.delaySeconds > 0 || d.delayBeats > 0)) {
      error = "Live Spectral History cannot use Audio Deformer peak hold, hysteresis or delay. Select Project Baked Analysis";
      return false;
    }
  }
  if (binding.target == Target::instanceTransform) {
    if (cardinality != binding.lastBand - binding.firstBand + 1) {
      error = "Spectrum Instancer requires one instance per selected band";
      return false;
    }
    for (const auto component : binding.baseScale)
      if (!std::isfinite(component) || component < 0.000001f || component > 1000.0f) {
        error = "Spectrum Instancer base scale is invalid";
        return false;
      }
    for (const auto component : binding.scaleResponse)
      if (!std::isfinite(component) || std::abs(component) > 1000.0f) {
        error = "Spectrum Instancer scale response is invalid";
        return false;
      }
  }
  error.clear();
  return true;
}

inline Bands normalizeBands(Bands bands, const Binding& binding) noexcept {
  float peak = 0.0f;
  for (std::size_t i = 0; i < bands.size(); ++i) {
    bands[i] = std::isfinite(bands[i]) ? std::clamp(bands[i], 0.0f, 1.0f) : 0.0f;
    if (i >= binding.firstBand && i <= binding.lastBand)
      peak = std::max(peak, bands[i]);
  }
  if (binding.normalize && peak > 0.0f)
    for (auto& band : bands) band /= peak;
  return bands;
}

inline float sample(const Binding& binding, const Bands& bands, float coordinate) noexcept {
  const float bin = binding.firstBand + coordinate * (binding.lastBand - binding.firstBand);
  const auto low = static_cast<std::size_t>(bin);
  const auto high = std::min(low + 1, static_cast<std::size_t>(binding.lastBand));
  const float value = binding.interpolate
      ? bands[low] + (bands[high] - bands[low]) * (bin - static_cast<float>(low))
      : bands[static_cast<std::size_t>(std::round(bin))];
  return value * binding.gain + binding.bias;
}

inline void advance(Bands& followed, const Bands& target, const Binding& binding,
                    double elapsedSeconds) noexcept {
  for (std::size_t band = 0; band < followed.size(); ++band) {
    const double seconds = target[band] > followed[band]
        ? binding.attackSeconds : binding.releaseSeconds;
    const float weight = seconds <= 0.0 ? 1.0f
        : static_cast<float>(-std::expm1(-elapsedSeconds / seconds));
    followed[band] += (target[band] - followed[band]) * weight;
  }
}

// Scalar input has its own time source. A caller that has no scalar reader at
// all leaves ScalarInput empty, and a scalar choice then fails closed. A reader
// that exists but has no frame for a time returns zeros, the same as the band
// fallback when a range export has no project history. Band range never reads
// ScalarInput.
using ScalarsAt = std::function<bool(double, Scalars&)>;
// Maps a project time and a delay in beats to the earlier project time the
// tempo map gives. A negative result means the delayed time is before beat zero.
using BeatDelay = std::function<double(double seconds, double beats)>;

struct ScalarInput final {
  const Scalars* current = nullptr;
  ScalarsAt at;
  // The project tempo map for an Audio Effector delay in beats.
  BeatDelay beatDelay;
};

inline float clampedFeature(float value) noexcept {
  return std::isfinite(value) ? std::clamp(value, 0.0f, 1.0f) : 0.0f;
}

// The Audio Effector input before its gate. Band range keeps the mean of the
// selected normalized bands. A scalar choice returns its feature alone, so the
// threshold, hysteresis, peak hold, attack and decay see the same stages.
inline float effectorReduction(const Binding& binding, const Bands& raw, const Scalars& scalars) noexcept {
  switch (binding.input) {
    case EffectorInput::rms: return clampedFeature(scalars.rms);
    case EffectorInput::peak: return clampedFeature(scalars.peak);
    case EffectorInput::onset: return clampedFeature(scalars.onset);
    case EffectorInput::bandRange: break;
  }
  const auto bands = normalizeBands(raw, binding);
  float value = 0;
  for (auto i = binding.firstBand; i <= binding.lastBand; ++i) value += bands[i];
  value /= static_cast<float>(binding.lastBand - binding.firstBand + 1);
  return value;
}

// A baked analysis stream gives random-access preview and export the same
// finite, 20-second following window at 60 Hz. The maximum time constant is two
// seconds, so the window covers ten time constants and at most 1201 samples.
// Live input follows successive presentation times and resets on a seek.
struct Follower final {
  struct HistoryFrame final { double timeSeconds; Bands bands; };
  Bands followed {};
  double timeSeconds = -1.0;
  float envelope = 0.0f, heldPeak = 0.0f;
  double peakUntil = -1.0;
  bool gate = false;
  std::uint64_t seekGeneration = 0, loopGeneration = 0;
  std::deque<HistoryFrame> history;
  bool historySourceSet = false;
  Source historySource = Source::master;
  std::int32_t historyTrackId = -1;
  std::uint64_t historyFieldId = 0, historyGeometryId = 0;
};
using FeaturesAt = std::function<Bands(double)>;

struct EffectorFrame final { Bands bands {}; Scalars scalars {}; };
inline constexpr const char* kBeatDelayUnavailable =
    "Audio Effector delay in beats requires the project's baked analysis history and tempo map";
inline constexpr const char* kScalarInputUnavailable =
    "Audio Effector RMS, Peak and Onset inputs require the shared audio feature frame";

// Reads the Audio Effector input at one project time. A delayed read uses the
// same source as an undelayed one. Returns false when a scalar input has no source.
inline bool readEffectorFrame(const Binding& binding, const Bands& current, const FeaturesAt& featuresAt,
                              const ScalarInput& scalars, double time, EffectorFrame& sample) {
  sample = {};
  const auto& d = binding.deformer;
  if (time < d.delaySeconds) return true;
  double source = time - d.delaySeconds;
  if (d.delayBeats > 0) {
    source = scalars.beatDelay ? scalars.beatDelay(time, d.delayBeats) : -1.0;
    if (!(source >= 0)) return true;
  }
  sample.bands = featuresAt ? featuresAt(source) : current;
  if (binding.input == EffectorInput::bandRange) return true;
  if (featuresAt) return scalars.at && scalars.at(source, sample.scalars);
  if (!scalars.current) return false;
  sample.scalars = *scalars.current;
  return true;
}

inline void advanceEffector(const Binding& binding, const Bands& raw, double seconds,
                            double elapsed, Follower& state, const Scalars& scalars = {}) noexcept {
  float value = effectorReduction(binding, raw, scalars);
  const auto& d = binding.deformer;
  state.gate = state.gate ? value >= d.threshold - d.hysteresis : value >= d.threshold;
  if (!state.gate) value = 0;
  if (value >= state.heldPeak || seconds > state.peakUntil) {
    state.heldPeak = value;
    state.peakUntil = seconds + d.peakHoldSeconds;
  }
  value = std::max(value, state.heldPeak);
  const double tau = value > state.envelope ? binding.attackSeconds : binding.releaseSeconds;
  const auto weight = tau <= 0 ? 1.0f : static_cast<float>(-std::expm1(-elapsed / tau));
  state.envelope += (value - state.envelope) * weight;
}

inline bool evaluate(const Binding& binding, const Bands& current, double seconds,
                     const FeaturesAt& featuresAt, Follower& state,
                     Bands& result, std::string& error, const ScalarInput& scalars = {}) {
  if (!std::isfinite(seconds) || seconds < 0.0 || seconds > 1.0e9) {
    error = "Audio Spectrum Field requires a finite non-negative timeline time";
    return false;
  }
  if (binding.target == Target::geometryDeformation) {
    const auto& d = binding.deformer;
    if (d.delaySeconds > 0 && !featuresAt) {
      error = "Audio Effector delay requires the project's baked analysis history";
      return false;
    }
    if (d.delayBeats > 0 && (!featuresAt || !scalars.beatDelay)) {
      error = kBeatDelayUnavailable;
      return false;
    }
    const auto at = [&](double time, EffectorFrame& sample) {
      return readEffectorFrame(binding, current, featuresAt, scalars, time, sample);
    };
    const auto unavailable = [&] {
      state = {};
      error = kScalarInputUnavailable;
      return false;
    };
    const bool hasMemory = d.hysteresis > 0 || d.peakHoldSeconds > 0
        || binding.attackSeconds > 0 || binding.releaseSeconds > 0;
    EffectorFrame sample;
    if (!hasMemory) {
      state = {};
      if (!at(seconds, sample)) return unavailable();
      advanceEffector(binding, sample.bands, seconds, 1.0/60.0, state, sample.scalars);
    } else if (featuresAt) {
      state = {};
      const auto last = static_cast<std::int64_t>(std::floor(seconds * 60.0));
      const auto first = std::max<std::int64_t>(0, last - 1200);
      for (auto frame = first; frame <= last; ++frame) {
        const double time = static_cast<double>(frame) / 60.0;
        if (!at(time, sample)) return unavailable();
        advanceEffector(binding, sample.bands, time, 1.0 / 60.0, state, sample.scalars);
      }
      const auto remainder = seconds - static_cast<double>(last) / 60.0;
      if (remainder > 0) {
        if (!at(seconds, sample)) return unavailable();
        advanceEffector(binding, sample.bands, seconds, remainder, state, sample.scalars);
      }
    } else {
      if (state.timeSeconds < 0 || seconds < state.timeSeconds || seconds - state.timeSeconds > 0.25) {
        state = {};
        state.timeSeconds = seconds - 1.0 / 60.0;
      }
      if (!at(seconds, sample)) return unavailable();
      advanceEffector(binding, sample.bands, seconds, seconds - state.timeSeconds, state, sample.scalars);
    }
    state.timeSeconds = seconds;
    auto shaped = std::pow(std::clamp(state.envelope, 0.0f, 1.0f), d.envelopePower);
    if (d.quantizeSteps > 0) shaped = std::round(shaped * d.quantizeSteps) / d.quantizeSteps;
    result.fill(shaped);
    error.clear();
    return true;
  }
  if (binding.attackSeconds == 0.0f && binding.releaseSeconds == 0.0f) {
    result = normalizeBands(current, binding);
  } else if (featuresAt) {
    result = {};
    constexpr double rate = 60.0;
    const auto last = static_cast<std::int64_t>(std::floor(seconds * rate));
    const auto first = std::max<std::int64_t>(0, last - 1200);
    for (auto frame = first; frame <= last; ++frame)
      advance(result, normalizeBands(featuresAt(static_cast<double>(frame) / rate), binding),
              binding, 1.0 / rate);
    const auto remainder = seconds - static_cast<double>(last) / rate;
    if (remainder > 0.0)
      advance(result, normalizeBands(current, binding), binding, remainder);
  } else {
    if (state.timeSeconds < 0.0 || seconds < state.timeSeconds
        || seconds - state.timeSeconds > 0.25) {
      state.followed = {};
      state.timeSeconds = seconds - 1.0 / 60.0;
    }
    advance(state.followed, normalizeBands(current, binding), binding,
            seconds - state.timeSeconds);
    state.timeSeconds = seconds;
    result = state.followed;
  }
  error.clear();
  return true;
}

// Envelope of one stateless deformer row: the mean of the selected bands after
// normalization, gated by threshold. Shaping happens in deformerHistoryResponse.
inline float deformerStatelessEnvelope(const Binding& binding, const Bands& bands,
                                       const Scalars& scalars = {}) noexcept {
  const float value = effectorReduction(binding, bands, scalars);
  return value < binding.deformer.threshold ? 0.0f : value;
}

// Baked Audio Deformer history rows hold one envelope value in every band. Row r
// is the envelope the current-frame deformer produces at its row time, so delay,
// peak hold, attack, release and hysteresis mean the same with and without history.
// Live history keeps spectrum smoothing for attack and release instead.
// Baked rows share one replay that starts 20 seconds before the oldest row, the
// warm-up the current-frame path uses. The start depends only on project time,
// so seek, loop and export order cannot change a row. The replay starts from zero,
// so a row differs from an unbounded replay by at most exp(-20 / 2), about 4.5e-5,
// when release is at its 2 second maximum. Shorter releases decay faster.
// A peak hold lasts at most 2 seconds, so a hold that began before the warm-up
// start has expired by the oldest row.
inline bool evaluateDeformerHistory(const Binding& binding, double seconds,
                                    const FeaturesAt& featuresAt, const ScalarInput& scalars,
                                    std::vector<Bands>& rows, std::string& error) {
  constexpr std::int64_t warmupFrames = 1200;
  if (binding.deformer.delayBeats > 0 && !scalars.beatDelay) { error = kBeatDelayUnavailable; return false; }
  const double spacing = static_cast<double>(binding.history.spacingSeconds);
  const Bands none {};
  EffectorFrame sample;
  const auto at = [&](double time) {
    return readEffectorFrame(binding, none, featuresAt, scalars, time, sample);
  };
  if (!deformerHasEnvelopeMemory(binding)) {
    for (std::size_t row = 0; row < rows.size(); ++row) {
      const double time = seconds - row * spacing;
      if (time < 0) continue;
      if (!at(time)) { error = kScalarInputUnavailable; return false; }
      rows[row].fill(deformerStatelessEnvelope(binding, sample.bands, sample.scalars));
    }
    return true;
  }
  const double oldest = seconds - static_cast<double>(rows.size() - 1) * spacing;
  auto frame = std::max<std::int64_t>(0, static_cast<std::int64_t>(std::floor(oldest * 60.0)) - warmupFrames);
  Follower state;
  // Rows are visited oldest first, so one shared replay serves all of them.
  for (std::size_t row = rows.size(); row-- > 0;) {
    const double time = seconds - row * spacing;
    if (time < 0) continue;
    const auto last = static_cast<std::int64_t>(std::floor(time * 60.0));
    for (; frame <= last; ++frame) {
      const double frameTime = static_cast<double>(frame) / 60.0;
      if (!at(frameTime)) { error = kScalarInputUnavailable; return false; }
      advanceEffector(binding, sample.bands, frameTime, 1.0 / 60.0, state, sample.scalars);
    }
    // The partial frame up to the exact row time runs on a copy, so the shared
    // replay stays on frame boundaries.
    Follower probe = state;
    const auto remainder = time - static_cast<double>(last) / 60.0;
    if (remainder > 0) {
      if (!at(time)) { error = kScalarInputUnavailable; return false; }
      advanceEffector(binding, sample.bands, time, remainder, probe, sample.scalars);
    }
    rows[row].fill(probe.envelope);
  }
  return true;
}

// Baked rows share one project-time 60 Hz pass, bounded by the history span
// plus the existing 20-second follower window. No rendering order enters it.
// Live rows contain only received frames. Seeking, looping or a gap longer
// than 250 ms discards them; absent history reads as zero.
inline bool evaluateHistory(const Binding& binding, const Bands& current, double timelineSeconds,
                            const FeaturesAt& featuresAt, bool liveFrameAvailable, bool exporting,
                            Follower& state, std::vector<Bands>& rows, std::string& error,
                            std::uint64_t seekGeneration = 0, std::uint64_t loopGeneration = 0,
                            const ScalarInput& scalars = {}) {
  if (!binding.history.enabled || !validate(binding, binding.coordinates.size(), error)) { state={}; return false; }
  if (!state.historySourceSet || state.historySource!=binding.source || state.historyTrackId!=binding.sourceTrackId
      || state.historyFieldId!=binding.fieldStableId || state.historyGeometryId!=binding.sourceGeometryStableId)
    state={};
  const auto& history = binding.history;
  const double seconds = history.useTimeline ? timelineSeconds : history.currentTimeSeconds;
  if (!std::isfinite(seconds) || seconds < 0 || seconds > 1.0e9) {
    error = "Spectral History Field requires a finite non-negative current time"; state={}; return false;
  }
  if (history.live && exporting) {
    error = "Live Spectral History cannot be exported. Select Project Baked Analysis and bake the project mix";
    state={}; return false;
  }
  rows.assign(history.depth, {});
  if (!history.live) {
    state={};
    if (!featuresAt) {
      error = "Spectral History Field requires the project's baked audio analysis. Open the Video Editor and stop playback to bake the mix";
      state={}; return false;
    }
    if (binding.target==Target::geometryDeformation) {
      if (!evaluateDeformerHistory(binding,seconds,featuresAt,scalars,rows,error)) { state={}; return false; }
      error.clear(); return true;
    }
    const auto oldest = seconds - (history.depth - 1) * static_cast<double>(history.spacingSeconds);
    auto nextFrame = std::max<std::int64_t>(0, static_cast<std::int64_t>(std::floor(oldest * 60.0)) - 1200);
    Bands followed {};
    for (std::size_t row = history.depth; row-- > 0;) {
      const double time = seconds - row * static_cast<double>(history.spacingSeconds);
      if (time < 0) continue;
      const auto lastFrame = static_cast<std::int64_t>(std::floor(time * 60.0));
      if (binding.attackSeconds == 0 && binding.releaseSeconds == 0) {
        rows[row] = normalizeBands(featuresAt(time), binding);
        continue;
      }
      for (; nextFrame <= lastFrame; ++nextFrame)
        advance(followed, normalizeBands(featuresAt(static_cast<double>(nextFrame) / 60.0), binding),
                binding, 1.0 / 60.0);
      rows[row] = followed;
      const auto remainder = time - static_cast<double>(lastFrame) / 60.0;
      if (remainder > 0)
        advance(rows[row], normalizeBands(featuresAt(time), binding), binding, remainder);
    }
  } else {
    if (!liveFrameAvailable) {
      error = "Live Spectral History requires the shared live master audio frame"; state={}; return false;
    }
    if (state.timeSeconds < 0 || seconds < state.timeSeconds || seconds - state.timeSeconds > 0.25
        || state.seekGeneration != seekGeneration || state.loopGeneration != loopGeneration) {
      state = {};
      state.timeSeconds = seconds - 1.0 / 60.0;
    }
    state.historySourceSet=true; state.historySource=binding.source; state.historyTrackId=binding.sourceTrackId;
    state.historyFieldId=binding.fieldStableId; state.historyGeometryId=binding.sourceGeometryStableId;
    state.seekGeneration=seekGeneration; state.loopGeneration=loopGeneration;
    advance(state.followed, normalizeBands(current, binding), binding, seconds - state.timeSeconds);
    state.timeSeconds = seconds;
    // Coalesce presentation frames in the same 60 Hz bucket.
    if (!state.history.empty() && std::floor(state.history.back().timeSeconds * 60.0) == std::floor(seconds * 60.0))
      state.history.back() = {seconds, state.followed};
    else state.history.push_back({seconds, state.followed});
    // Keep the binding's own span, plus one older frame to interpolate the oldest row.
    const double span = (history.depth - 1) * static_cast<double>(history.spacingSeconds);
    while (state.history.size() > 2 && state.history[1].timeSeconds < seconds - span)
      state.history.pop_front();
    for (std::size_t row = 0; row < history.depth; ++row) {
      const auto time = seconds - row * static_cast<double>(history.spacingSeconds);
      if (time < state.history.front().timeSeconds || time > state.history.back().timeSeconds) continue;
      const auto upper = std::lower_bound(state.history.begin(), state.history.end(), time,
          [](const auto& frame, double at) { return frame.timeSeconds < at; });
      if (upper == state.history.begin() || upper->timeSeconds == time) rows[row] = upper->bands;
      else {
        const auto lower = std::prev(upper);
        const auto weight = static_cast<float>((time - lower->timeSeconds) / (upper->timeSeconds - lower->timeSeconds));
        for (std::size_t band = 0; band < kBands; ++band)
          rows[row][band] = lower->bands[band] + weight * (upper->bands[band] - lower->bands[band]);
      }
    }
  }
  // Live Audio Deformer rows keep the previous semantics. Attack and release
  // smooth the band spectrum in the follower above, and each row then reduces to
  // its gated mean, the same value the previous response computed. Baked history
  // uses deformer-envelope semantics instead, handled by evaluateDeformerHistory.
  if (binding.target==Target::geometryDeformation)
    for (auto& row : rows) row.fill(deformerStatelessEnvelope(binding,row));
  error.clear(); return true;
}

inline float sampleHistory(const Binding& binding, const std::vector<Bands>& rows, std::size_t vertex) noexcept {
  const auto row = static_cast<std::size_t>(std::round(binding.history.coordinates[vertex] * (rows.size() - 1)));
  return sample(binding, rows[row], binding.coordinates[vertex]);
}

// Each row holds the deformer envelope, so the response only shapes and
// quantizes it, then applies the same gain and bias as the current-frame deformer.
inline float deformerHistoryResponse(const Binding& binding, const std::vector<Bands>& rows,
                                     std::size_t vertex) noexcept {
  const auto row = static_cast<std::size_t>(std::round(binding.history.coordinates[vertex] * (rows.size() - 1)));
  const auto& d = binding.deformer;
  auto shaped = std::pow(std::clamp(rows[row][binding.firstBand], 0.0f, 1.0f), d.envelopePower);
  if (d.quantizeSteps > 0) shaped = std::round(shaped * d.quantizeSteps) / d.quantizeSteps;
  return shaped * binding.gain + binding.bias;
}
} // namespace videowire::geometry::spectrum
