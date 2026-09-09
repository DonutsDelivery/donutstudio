#include "canonical_block_c_frame.h"
#include "gpu_backend/backend.h"
#include "score_json_parser.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>

namespace
{
using canonicalblockc::CanonicalBlockCFrame;
using canonicalblockc::FrameKey;
using canonicalblockc::FrameProducer;

[[noreturn]] void fail(const char* message)
{
    std::cerr << message << '\n';
    std::exit(1);
}

void require(bool condition, const char* message)
{
    if (!condition) fail(message);
}

FrameKey key()
{
    FrameKey result;
    result.projectGeneration = 1;
    result.sourceGeneration = 2;
    result.helperGeneration = 3;
    result.backendGeneration = 4;
    result.deviceGeneration = 5;
    result.scoreGeneration = 6;
    result.beatMapGeneration = 7;
    result.fpsGeneration = 8;
    result.loopGeneration = 9;
    result.seekGeneration = 10;
    result.frame = 0;
    result.beat = 0.0;
    result.fps = 60.0;
    return result;
}

arbitmod::Note note(int id, float startBeat, float midiNote, float frequency)
{
    arbitmod::Note result;
    result.id = id;
    result.durableKind = 2;
    result.durableNoteId = -1;
    result.clipId = 41;
    result.clipNoteId = id == -1 ? 11 : 12;
    result.repeatIndex = 3;
    result.trackId = 1;
    result.startBeat = startBeat;
    result.lengthBeats = 8.0f;
    result.midiNote = midiNote;
    result.velocity = 96.0f;
    result.freqHz = frequency;
    result.durationSeconds = 4.0f;
    result.pitchAnchors.push_back({-701 + id, 0.0f, frequency});
    return result;
}

std::shared_ptr<arbitmod::Score> projectedLoopScore()
{
    auto score = std::make_shared<arbitmod::Score>();
    score->notationVersion = 1;
    score->scoreRevision = 77;
    score->edoStepsPerOctave = 12;
    score->rootFreq = 440.0f;
    score->historyBeats = 8.0f;
    score->lookaheadBeats = 16.0f;
    score->notes.push_back(note(-1, -1.0f, 60.0f, 261.6256f));
    score->notes.push_back(note(-102, 0.0f, 67.0f, 391.9954f));
    score->notes[1].linkMasterId = -1;
    score->links.push_back({-301, -102, -1, 3, 2, 0});
    return score;
}

void requireProjectedIdentities(const std::shared_ptr<const CanonicalBlockCFrame>& frame)
{
    require(canonicalblockc::valid(frame), "projected loop frame was not valid");
    require(frame->noteRowCount == 2, "projected notes were not packed");
    require(frame->linkRowCount == 1, "projected harmonic link was not packed");
    const auto hasNote = [&](std::int64_t identity)
    {
        return std::find(frame->noteIdentities.begin(), frame->noteIdentities.end(), identity)
            != frame->noteIdentities.end();
    };
    require(hasNote(-1) && hasNote(-102), "negative projected note identity changed");
    require(frame->linkIdentities[0] == -301, "negative projected link identity changed");
    const auto rowOf = [&](std::int64_t identity)
    {
        const auto found = std::find(frame->noteIdentities.begin(), frame->noteIdentities.end(), identity);
        return found == frame->noteIdentities.end()
            ? -1 : static_cast<int>(std::distance(frame->noteIdentities.begin(), found));
    };
    const auto packedMasterRow = [&](int noteRow)
    {
        constexpr std::size_t valuesPerNote = arbitblockc::kTexelsPerNote * 4u;
        return static_cast<int>(frame->noteTextureValues[
            static_cast<std::size_t>(noteRow) * valuesPerNote + 3u * 4u + 2u]);
    };
    const int masterRow = rowOf(-1);
    const int slaveRow = rowOf(-102);
    require(masterRow >= 0 && slaveRow >= 0, "projected note row was absent");
    require(packedMasterRow(slaveRow) == masterRow,
            "projected master identity -1 did not resolve to its resident row");
    require(packedMasterRow(masterRow) == -1,
            "zero no-master sentinel did not pack as no master");
    for (int row = 0; row < frame->noteRowCount; ++row)
        if (frame->noteIdentities[static_cast<std::size_t>(row)] != 0)
            require(frame->notationRows[static_cast<std::size_t>(row)].identity
                        == frame->noteIdentities[static_cast<std::size_t>(row)],
                    "notation identity differs from packed identity");
}
}

int main()
{
    const auto wireScore = [](const nlohmann::json& version) {
        nlohmann::json score = {
            {"scoreRevision", 77},
            {"notes", nlohmann::json::array({
                {{"id", -1}, {"linkMasterId", 0}},
                {{"id", -102}, {"linkMasterId", -1}}
            })},
            {"links", nlohmann::json::array({
                {{"id", -301}, {"slaveNoteId", -102}, {"masterNoteId", -1}}
            })}
        };
        if (!version.is_null()) score["schemaVersion"] = version;
        return score;
    };
    const auto parseWire = [&](const nlohmann::json& wire, arbitmod::Score& parsed,
                               const char* message) {
        std::string error;
        require(videohelper::scorejson::parseScoreJson(wire, parsed, error), message);
        require(error.empty(), "successful score parse retained an error");
    };

    arbitmod::Score missingVersion;
    parseWire(wireScore(nullptr), missingVersion, "missing schema version was rejected");
    require(missingVersion.notes[1].linkMasterId == 0,
            "missing schema version did not migrate legacy -1 to zero");

    arbitmod::Score legacyVersion;
    parseWire(wireScore(1), legacyVersion, "explicit v1 score was rejected");
    require(legacyVersion.notes[1].linkMasterId == 0,
            "explicit v1 did not migrate legacy -1 to zero");

    arbitmod::Score currentVersion;
    parseWire(wireScore(2), currentVersion, "explicit v2 score was rejected");
    require(currentVersion.notes[0].linkMasterId == 0,
            "v2 zero absence changed during parsing");
    require(currentVersion.notes[1].linkMasterId == -1,
            "v2 projected -1 master changed during parsing");

    for (const auto& malformed : { nlohmann::json("2"), nlohmann::json(2.0),
                                   nlohmann::json(true), nlohmann::json::object(),
                                   nlohmann::json(std::numeric_limits<std::uint64_t>::max()) })
    {
        arbitmod::Score rejected;
        std::string error;
        require(!videohelper::scorejson::parseScoreJson(wireScore(malformed), rejected, error),
                "malformed schema version did not fail closed");
        require(!error.empty(), "malformed schema version lacked an error");
    }
    for (const int unknown : { -1, 0, 3, 99 })
    {
        arbitmod::Score rejected;
        std::string error;
        require(!videohelper::scorejson::parseScoreJson(wireScore(unknown), rejected, error),
                "unknown schema version did not fail closed");
        require(!error.empty(), "unknown schema version lacked an error");
    }

    FrameProducer legacyProducer;
    const auto legacyFrame = legacyProducer.evaluate(
        key(), std::make_shared<arbitmod::Score>(legacyVersion), 0.0f);
    require(canonicalblockc::valid(legacyFrame),
            "legacy score beside projected -1 was rejected");
    const auto legacySlave = std::find(legacyFrame->noteIdentities.begin(),
                                      legacyFrame->noteIdentities.end(), -102);
    require(legacySlave != legacyFrame->noteIdentities.end(),
            "legacy slave row was absent");
    constexpr std::size_t valuesPerNote = arbitblockc::kTexelsPerNote * 4u;
    const auto legacySlaveRow = static_cast<std::size_t>(
        std::distance(legacyFrame->noteIdentities.begin(), legacySlave));
    require(legacyFrame->noteTextureValues[
                legacySlaveRow * valuesPerNote + 3u * 4u + 2u] == -1.0f,
            "legacy -1 absence acquired the adjacent projected -1 master");

    const auto score = projectedLoopScore();
    auto frameKey = key();
    require(FrameProducer::admissible(frameKey, score, 0.0f),
            "negative projected identities were rejected at admission");

    FrameProducer producer;
    auto first = producer.evaluate(frameKey, score, 0.0f);
    requireProjectedIdentities(first);
    arbitgpu::NativeImportedSceneRuntimeInputs runtimeInputs;
    runtimeInputs.canonicalBlockCFrame = first;
    visualnoteinstancing::Mapping mapping;
    mapping.x = visualnoteinstancing::MappingAxis::Onset;
    mapping.y = visualnoteinstancing::MappingAxis::LogFrequency;
    mapping.z = visualnoteinstancing::MappingAxis::Track;
    mapping.xScale = 2.0f;
    mapping.yScale = 3.0f;
    mapping.zScale = 4.0f;
    mapping.meshScale = 0.25f;
    runtimeInputs.noteInstanceMapping = mapping;
    require(arbitgpu::validFixtureRuntimeInputs(runtimeInputs),
            "canonical imported-scene runtime inputs were rejected");
    const auto instances = arbitgpu::prepareNativeNoteInstances(runtimeInputs);
    require(instances.count == 2,
            "native note preparation did not preserve the admitted Block C rows");
    require(instances.identities[0] < 0 && instances.identities[1] < 0
                && instances.identities[0] != instances.identities[1],
            "native note preparation changed projected negative identities");
    for (std::size_t index = 0; index < instances.count; ++index)
    {
        const auto row = instances.canonicalRows[index];
        require(first->noteIdentities[row] == instances.identities[index],
                "native note preparation lost canonical row identity");
        require(instances.transforms[index][3] == mapping.meshScale,
                "native note preparation changed the bounded mesh scale");
    }
    auto incompleteInputs = runtimeInputs;
    incompleteInputs.noteInstanceMapping.reset();
    require(!arbitgpu::validFixtureRuntimeInputs(incompleteInputs),
            "a Block C frame without its exact mapping was admitted");
    auto malformedInputs = runtimeInputs;
    malformedInputs.noteInstanceMapping->xScale
        = std::numeric_limits<float>::infinity();
    require(!arbitgpu::validFixtureRuntimeInputs(malformedInputs),
            "a malformed native note mapping was admitted");
    const auto packsAfterFirst = producer.packCount();
    auto cached = producer.evaluate(frameKey, score, 0.0f);
    require(cached == first, "same-key evaluation did not return the cached frame");
    require(producer.packCount() == packsAfterFirst, "same-key cache repacked Block C");

    ++frameKey.seekGeneration;
    auto afterSeek = producer.evaluate(frameKey, score, 0.0f);
    requireProjectedIdentities(afterSeek);
    require(afterSeek != first, "seek generation reused the old frame");

    FrameProducer warmedProducer;
    std::int64_t packedFrame = -1;
    std::shared_ptr<const CanonicalBlockCFrame> warmed;
    auto warmKey = key();
    const auto beatAtFrame = [](std::int64_t frame) { return static_cast<double>(frame) * 0.25; };
    warmed = canonicalblockc::warmFrameProducerTo(
        warmedProducer, packedFrame, warmed, warmKey, score, 2, beatAtFrame);
    requireProjectedIdentities(warmed);
    require(packedFrame == 2, "warm path did not reach the requested frame");
    const auto packsAfterWarm = warmedProducer.packCount();
    const auto warmCached = canonicalblockc::warmFrameProducerTo(
        warmedProducer, packedFrame, warmed, warmKey, score, 2, beatAtFrame);
    require(warmCached == warmed, "warm path did not retain the cached frame");
    require(warmedProducer.packCount() == packsAfterWarm, "warm cache repacked the same frame");

    ++warmKey.scoreGeneration;
    const auto afterGeneration = canonicalblockc::warmFrameProducerTo(
        warmedProducer, packedFrame, warmed, warmKey, score, 2, beatAtFrame);
    requireProjectedIdentities(afterGeneration);
    require(afterGeneration != warmCached, "score generation reused the old frame");

    const auto requireRejected = [&](auto mutate, const char* message)
    {
        auto invalid = std::make_shared<arbitmod::Score>(*score);
        mutate(*invalid);
        require(!FrameProducer::admissible(key(), invalid, 0.0f), message);
        FrameProducer rejectingProducer;
        require(rejectingProducer.evaluate(key(), invalid, 0.0f) == nullptr, message);
    };
    requireRejected([](auto& invalid) { invalid.notes[0].id = 0; },
                    "zero note identity was admitted");
    requireRejected([](auto& invalid) { invalid.notes[0].pitchAnchors[0].id = 0; },
                    "zero anchor identity was admitted");
    requireRejected([](auto& invalid) { invalid.links[0].id = 0; },
                    "zero link identity was admitted");
    requireRejected([](auto& invalid) { invalid.links[0].slaveNoteId = 0; },
                    "zero link endpoint identity was admitted");
    requireRejected([](auto& invalid) { invalid.notes[1].linkMasterId = -77; },
                    "dangling nonzero master identity was admitted");

    auto invalid = std::make_shared<arbitmod::Score>(*score);
    invalid->links[0].masterNoteId = 0;
    packedFrame = 2;
    warmed = afterGeneration;
    require(canonicalblockc::warmFrameProducerTo(
                warmedProducer, packedFrame, warmed, key(), invalid, 3, beatAtFrame) == nullptr,
            "invalid warm request returned a frame");
    require(packedFrame == -1 && warmed == nullptr,
            "invalid warm request did not clear cached frame state");

    std::cout << "canonical Block C projected identity semantics passed\n";
    return 0;
}
