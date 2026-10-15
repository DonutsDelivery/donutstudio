#include "VisualAnimationPlaybackControl.h"
#include "VisualImportedAnimationOperationContract.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace
{
using namespace visualanimation;

int failures = 0;

void check (bool condition, const char* message)
{
    if (condition)
        return;
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", message);
}

bool near (double actual, double expected, double tolerance = 1.0e-9)
{
    return std::abs(actual - expected) <= tolerance;
}
} // namespace

int main()
{
    const std::array<double, 2> times { 0.0, 2.0 };
    const std::array<float, 6> values { 0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f };
    const TrackView track {
        TrackId { 1 }, TargetId { 1 }, Channel::Translation, Interpolation::Linear,
        times.data(), values.data(), times.size(), values.size(), 0
    };
    const ClipView view { ClipId { 1 }, 2.0, &track, 1 };
    std::string error;
    const auto clip = Clip::create(view, {}, error);
    check(clip && error.empty(), "playback-control fixture is admitted");
    if (! clip)
        return 1;

    PlaybackControlRequest timeline;
    timeline.timelineSeconds = 3.0;
    timeline.clipStartTimelineSeconds = 1.0;
    timeline.speed = 2.0;
    timeline.offsetSeconds = 0.25;
    timeline.weight = 0.75;
    timeline.playback = Playback::Clamp;
    const auto mappedTimeline = resolvePlaybackControl(*clip, timeline, {}, error);
    const auto timelineSample = mappedTimeline
        ? sample(*clip, mappedTimeline->sample, error) : std::nullopt;
    check(mappedTimeline && near(mappedTimeline->sample.timeSeconds, 4.25)
          && near(mappedTimeline->weight, 0.75) && timelineSample
          && near(timelineSample->sampleTimeSeconds(), 2.0),
          "timeline sync applies clip start, speed, offset, weight, and clamp explicitly");

    PlaybackControlRequest beat;
    beat.timeSource = TimeSource::BeatSync;
    beat.timelineBeat = 3.0;
    beat.clipStartBeat = 1.0;
    beat.beatsPerLoop = 4.0;
    beat.playback = Playback::Loop;
    const auto mappedBeat = resolvePlaybackControl(*clip, beat, {}, error);
    const auto beatSample = mappedBeat ? sample(*clip, mappedBeat->sample, error) : std::nullopt;
    check(mappedBeat && near(mappedBeat->sample.timeSeconds, 1.0)
          && beatSample && near(beatSample->sampleTimeSeconds(), 1.0),
          "beat sync maps an authored beat span onto one clip duration");

    beat.timelineBeat = 5.0;
    const auto loopBoundary = resolvePlaybackControl(*clip, beat, {}, error);
    const auto boundarySample = loopBoundary
        ? sample(*clip, loopBoundary->sample, error) : std::nullopt;
    check(boundarySample && near(boundarySample->sampleTimeSeconds(), 0.0),
          "beat-sync loop endpoint maps to the first clip frame");

    beat.speed = -1.0;
    beat.timelineBeat = 3.0;
    beat.offsetSeconds = 2.0;
    const auto reverse = resolvePlaybackControl(*clip, beat, {}, error);
    const auto repeated = resolvePlaybackControl(*clip, beat, {}, error);
    check(reverse && repeated && near(reverse->sample.timeSeconds, 1.0)
          && reverse->sample.timeSeconds == repeated->sample.timeSeconds
          && reverse->weight == repeated->weight,
          "reverse playback and equal-time reevaluation are deterministic");

    auto trimmed = timeline;
    trimmed.timelineSeconds = 1.75;
    trimmed.clipStartTimelineSeconds = 1.0;
    trimmed.speed = 1.0;
    trimmed.offsetSeconds = 0.0;
    trimmed.trimStartSeconds = 0.5;
    trimmed.trimEndSeconds = 1.5;
    trimmed.playback = Playback::Loop;
    const auto mappedTrim = resolvePlaybackControl(*clip, trimmed, {}, error);
    const auto trimSample = mappedTrim
        ? sample(*clip, mappedTrim->sample, error) : std::nullopt;
    check(mappedTrim && near(mappedTrim->sample.timeSeconds, 1.25)
          && near(mappedTrim->sample.rangeStartSeconds, 0.5)
          && near(mappedTrim->sample.rangeEndSeconds, 1.5)
          && trimSample && near(trimSample->sampleTimeSeconds(), 1.25),
          "trimmed playback offsets into the admitted clip range");
    trimmed.timelineSeconds = 2.25;
    const auto trimLoop = resolvePlaybackControl(*clip, trimmed, {}, error);
    const auto trimLoopSample = trimLoop
        ? sample(*clip, trimLoop->sample, error) : std::nullopt;
    check(trimLoopSample && near(trimLoopSample->sampleTimeSeconds(), 0.75),
          "trimmed loop playback wraps inside the selected range");
    trimmed.timeSource = TimeSource::BeatSync;
    trimmed.timelineBeat = 2.0;
    trimmed.clipStartBeat = 0.0;
    trimmed.beatsPerLoop = 4.0;
    const auto trimBeat = resolvePlaybackControl(*clip, trimmed, {}, error);
    const auto trimBeatSample = trimBeat
        ? sample(*clip, trimBeat->sample, error) : std::nullopt;
    check(trimBeatSample && near(trimBeatSample->sampleTimeSeconds(), 1.0),
          "beat sync maps one beat span onto the trimmed duration");

    auto invalid = timeline;
    invalid.speed = std::numeric_limits<double>::infinity();
    check(! resolvePlaybackControl(*clip, invalid, {}, error)
          && error == "animation playback speed is nonfinite or outside the admitted range",
          "nonfinite playback speed is rejected");
    invalid = timeline;
    invalid.weight = -0.1;
    check(! resolvePlaybackControl(*clip, invalid, {}, error)
          && error == "animation playback weight must be finite and in [0,1]",
          "out-of-range blend weight is rejected");
    invalid = beat;
    invalid.beatsPerLoop = 0.0;
    check(! resolvePlaybackControl(*clip, invalid, {}, error)
          && error == "animation beat-sync length is outside the admitted range",
          "zero beat-sync duration is rejected");
    invalid = timeline;
    invalid.trimStartSeconds = 1.5;
    invalid.trimEndSeconds = 1.0;
    check(! resolvePlaybackControl(*clip, invalid, {}, error)
          && error == "animation playback trim range is invalid or outside the admitted clip",
          "reversed trim ranges are rejected");
    invalid = timeline;
    invalid.timeSource = static_cast<TimeSource>(99);
    check(! resolvePlaybackControl(*clip, invalid, {}, error)
          && error == "animation time source is unsupported",
          "unknown time-source values fail closed");

    PlaybackControlLimits narrow;
    narrow.maxAbsoluteSpeed = 1.0;
    check(! resolvePlaybackControl(*clip, timeline, narrow, error)
          && error == "animation playback speed is nonfinite or outside the admitted range",
          "caller playback bounds are enforced");

    visualanimationimport::Request operation;
    operation.sourceStableId = 1;
    operation.deformationStableId = 2;
    operation.schedule = { 1, 2 };
    operation.asset = { "trimmed-model", 3, std::string(64, 'a'),
                        "model/gltf-binary", 4096 };
    operation.animationClipStableId = 4;
    operation.meshStableId = 5;
    operation.clipName = "Walk";
    operation.playback.trimStartSeconds = 0.5;
    operation.playback.trimEndSeconds = 1.5;
    const auto encoded = visualanimationoperation::encode(operation);
    visualanimationimport::Request decoded;
    check(encoded.rfind("visual.imported-animation.deformation.v5\n", 0) == 0
          && visualanimationoperation::decode(encoded, decoded, error)
          && decoded.animationClipStableId == operation.animationClipStableId
          && decoded.meshStableId == operation.meshStableId
          && near(decoded.playback.trimStartSeconds, 0.5)
          && near(decoded.playback.trimEndSeconds, 1.5),
          "operation V5 preserves typed selectors and the admitted trim range");

    for (int version=1;version<=7;++version)
    {
        auto legacy=operation;
        legacy.playback.trimStartSeconds=legacy.playback.trimEndSeconds=0;
        legacy.meshStableId=0; legacy.animationClipStableId=0;
        if (version>=2) legacy.animationClipStableId=4;
        if (version==3) legacy.combinationMode=CombinationMode::WeightedBlend;
        if (version>=4) legacy.meshStableId=5;
        if (version==5) legacy.playback.trimStartSeconds=0.5;
        if (version==6) { legacy.pose.meshStableId=5; legacy.pose.boneEnabled=true; legacy.pose.boneStableId=2; }
        if (version==7) { legacy.pose.meshStableId=5; legacy.pose.nodeStableId=3; }
        const auto oldBytes=visualanimationoperation::encode(legacy);
        check(oldBytes.rfind("visual.imported-animation.deformation.v"+std::to_string(version)+"\n",0)==0
            && visualanimationoperation::decode(oldBytes,decoded,error)
            && decoded.crossfadeClipStableId==0 && decoded.crossfadeClipName.empty()
            && decoded.crossfadeProgress==0 && visualanimationoperation::encode(decoded)==oldBytes,
            "every released v1-v7 payload roundtrips unchanged with single-action defaults");
    }

    auto crossfadeOperation=operation;
    crossfadeOperation.crossfadeClipStableId=6;
    crossfadeOperation.crossfadeClipName="Reach";
    crossfadeOperation.crossfadeProgress=0.375;
    const auto crossfadeBytes=visualanimationoperation::encode(crossfadeOperation);
    check(crossfadeBytes.rfind("visual.imported-animation.deformation.v8\n",0)==0
        && visualanimationoperation::decode(crossfadeBytes,decoded,error)
        && decoded.crossfadeClipStableId==6 && decoded.crossfadeClipName=="Reach"
        && decoded.crossfadeProgress==0.375 && decoded.playback.trimStartSeconds==0.5,
        "v8 retains two exact clip selectors, crossfade progress and old playback controls");
    check(visualanimationoperation::decode(encoded,decoded,error)
        && decoded.crossfadeClipStableId==0 && decoded.crossfadeProgress==0.0,
        "old v5 payloads remain single-action requests after decoding into a reused destination");
    auto invalidCrossfade=crossfadeOperation;
    invalidCrossfade.crossfadeProgress=1.01;
    check(visualanimationoperation::encode(invalidCrossfade).empty(),
        "crossfade progress outside the authored range is refused at the wire boundary");
    invalidCrossfade=crossfadeOperation; invalidCrossfade.crossfadeClipName="Reach\n";
    check(visualanimationoperation::encode(invalidCrossfade).empty(),
        "crossfade names retain the released bounded text grammar");
    check(!visualanimationoperation::decode(crossfadeBytes+"extra",decoded,error),
        "crossfade wire payloads refuse trailing data");

    // Morph weight deltas: a V9 payload appends the selected mesh's keys as one
    // trailing block. These checks pin its grammar, its bounds and its mesh binding.
    const auto withDeltas = [&operation] (
        std::vector<std::pair<std::uint32_t, double>> entries, std::uint64_t meshStableId = 5)
    {
        auto candidate = operation;
        candidate.pose.meshStableId = meshStableId;
        candidate.pose.morphWeightDeltas = std::move(entries);
        return candidate;
    };
    const auto keyed = withDeltas({ { 1u, 0.5 }, { 3u, -2.25 } });
    const auto keyedBytes = visualanimationoperation::encode(keyed);
    visualanimationimport::Request keyedDecoded;
    check(keyedBytes.rfind("visual.imported-animation.deformation.v9\n", 0) == 0
        && visualanimationoperation::decode(keyedBytes, keyedDecoded, error)
        && keyedDecoded.pose.morphWeightDeltas == keyed.pose.morphWeightDeltas
        && keyedDecoded.crossfadeClipStableId == 0 && keyedDecoded.crossfadeProgress == 0.0
        && keyedDecoded.playback.trimStartSeconds == 0.5
        && visualanimationoperation::encode(keyedDecoded) == keyedBytes,
        "v9 carries two morph weight deltas and re-encodes identically");
    // The wire bytes identify the operation that the parent binding and the consumer compare,
    // so changing one delta changes that identity.
    const auto nudged = withDeltas({ { 1u, 0.5 }, { 3u, -2.5 } });
    check(!visualanimationoperation::encode(nudged).empty()
        && visualanimationoperation::encode(nudged) != keyedBytes,
        "one changed morph weight delta changes the encoded operation identity");

    auto crossfadeKeyed = crossfadeOperation;
    crossfadeKeyed.pose.meshStableId = 5;
    crossfadeKeyed.pose.morphWeightDeltas = { { 2u, 1.0 } };
    const auto crossfadeKeyedBytes = visualanimationoperation::encode(crossfadeKeyed);
    check(crossfadeKeyedBytes.rfind("visual.imported-animation.deformation.v9\n", 0) == 0
        && visualanimationoperation::decode(crossfadeKeyedBytes, decoded, error)
        && decoded.crossfadeClipStableId == 6 && decoded.crossfadeClipName == "Reach"
        && decoded.crossfadeProgress == 0.375
        && decoded.pose.morphWeightDeltas.size() == 1
        && decoded.pose.morphWeightDeltas[0].first == 2u
        && decoded.pose.morphWeightDeltas[0].second == 1.0,
        "v9 keeps the crossfade selection and the morph weight deltas together");

    check(visualanimationoperation::encode(withDeltas({ { 1u, 0.5 } }, 0)).empty(),
        "morph weight deltas without a selected mesh are refused");
    check(visualanimationoperation::encode(withDeltas({ { 1u, 0.5 } }, 6)).empty(),
        "morph weight deltas must name the request's selected mesh");
    check(visualanimationoperation::encode(withDeltas({ { 1u, 0.5 }, { 1u, 0.25 } })).empty(),
        "duplicate per-key morph indices are refused");
    check(visualanimationoperation::encode(withDeltas({ { 65536u, 0.5 } })).empty(),
        "per-key morph indices at or above 65536 are refused");
    check(visualanimationoperation::encode(
              withDeltas({ { 1u, std::numeric_limits<double>::quiet_NaN() } })).empty()
        && visualanimationoperation::encode(
              withDeltas({ { 1u, std::numeric_limits<double>::infinity() } })).empty(),
        "nonfinite per-key morph weights are refused");
    check(visualanimationoperation::encode(withDeltas({ { 1u, 16.5 } })).empty()
        && visualanimationoperation::encode(withDeltas({ { 1u, -16.5 } })).empty(),
        "per-key morph weights outside [-16,16] are refused");
    check(!visualanimationoperation::encode(withDeltas({ { 1u, 16.0 }, { 2u, -16.0 } })).empty(),
        "per-key morph weights at the 16 limit are admitted");

    std::vector<std::pair<std::uint32_t, double>> full;
    for (std::uint32_t index = 0; index < visualanimation::kMaximumMorphWeightDeltas; ++index)
        full.emplace_back(index * 1000u + 7u, -(static_cast<double>(index) + 1.0) / 5.0);
    const auto fullBytes = visualanimationoperation::encode(withDeltas(full));
    visualanimationimport::Request fullDecoded;
    check(!fullBytes.empty() && visualanimationoperation::decode(fullBytes, fullDecoded, error)
        && fullDecoded.pose.morphWeightDeltas == full,
        "64 morph weight deltas are admitted and roundtrip");
    auto overfull = full;
    overfull.emplace_back(65000u, 0.0);
    check(visualanimationoperation::encode(withDeltas(overfull)).empty(),
        "65 morph weight deltas are refused");

    std::vector<std::pair<std::uint32_t, double>> widest;
    for (std::uint32_t index = 0; index < visualanimation::kMaximumMorphWeightDeltas; ++index)
        widest.emplace_back(65535u - index, -(15.0 + 1.0 / 3.0));
    auto widestCase = withDeltas(widest);
    widestCase.clipName.assign(256, 'w');
    widestCase.crossfadeClipStableId = 6;
    widestCase.crossfadeClipName.assign(256, 'c');
    widestCase.crossfadeProgress = 0.375;
    const auto widestBytes = visualanimationoperation::encode(widestCase);
    visualanimationimport::Request widestDecoded;
    check(!widestBytes.empty() && widestBytes.size() <= visualanimationoperation::kMaximumEncodedBytes
        && visualanimationoperation::decode(widestBytes, widestDecoded, error)
        && widestDecoded.pose.morphWeightDeltas == widest
        && widestDecoded.clipName.size() == 256 && widestDecoded.crossfadeClipName.size() == 256,
        "64 widest morph weight deltas with 256-byte names fit the encoded cap");

    // The last line of a V9 payload is its delta block. Each tampered variant replaces it.
    const auto keyedHead = keyedBytes.substr(0, keyedBytes.rfind('\n', keyedBytes.size() - 2) + 1);
    const auto refusesWire = [&error] (const std::string& wire)
    {
        visualanimationimport::Request sink;
        return !visualanimationoperation::decode(wire, sink, error);
    };
    check(refusesWire(keyedHead + "3 1 0.5 3 -2.25\n"),
        "an delta count larger than its listed pairs is refused");
    check(refusesWire(keyedHead + "two 1 0.5 3 -2.25\n"),
        "a non-numeric delta count is refused");
    check(refusesWire(keyedHead + "2 1 0.5 1 -2.25\n"),
        "a duplicate delta index on the wire is refused");
    check(refusesWire(keyedHead + "1 65536 0.5\n"),
        "a wire delta index at or above 65536 is refused");
    check(refusesWire(keyedHead + "1 1 nan\n"),
        "a nonfinite wire delta weight is refused");
    std::string oversizedBlock = "65";
    for (std::uint32_t index = 0; index < 65; ++index)
        oversizedBlock += " " + std::to_string(index) + " 0.5";
    check(refusesWire(keyedHead + oversizedBlock + "\n"),
        "a declared delta count above 64 is refused");
    check(refusesWire(keyedBytes + "extra"),
        "v9 payloads refuse trailing data");

    std::fprintf(stderr, failures == 0
        ? "Animation playback-control checks passed\n"
        : "%d animation playback-control checks failed\n", failures);
    return failures == 0 ? 0 : 1;
}
