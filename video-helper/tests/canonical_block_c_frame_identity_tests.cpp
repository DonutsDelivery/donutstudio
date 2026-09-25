#include "canonical_block_c_frame.h"
#include "block_c_frame_owner.h"
#include "gpu_backend/backend.h"
#include "harmonic_link_geometry.h"
#include "particle_parameters.h"
#include "score_json_parser.h"
#include "score_field_fixture.h"

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

void requireResidentReferences(const CanonicalBlockCFrame& frame, const arbitmod::Score& score)
{
    require(frame.noteRowCount <= arbitblockc::kMaxNotes
                && frame.linkRowCount <= arbitblockc::kMaxLinks,
            "long score exceeded resident texture capacity");
    for (int row = 0; row < frame.noteRowCount; ++row)
    {
        const auto id = frame.noteIdentities[static_cast<std::size_t>(row)];
        if (id == 0) continue;
        const auto* source = score.noteById(static_cast<int>(id));
        require(source != nullptr, "resident row invented a note identity");
        require(frame.notationRows[static_cast<std::size_t>(row)].identity == id,
                "resident notation lost its note identity");
        const auto master = std::find(frame.noteIdentities.begin(), frame.noteIdentities.end(),
                                      source->linkMasterId);
        const int expectedMaster = source->linkMasterId == 0 || master == frame.noteIdentities.end()
            ? -1 : static_cast<int>(master - frame.noteIdentities.begin());
        require(frame.noteTextureValues[static_cast<std::size_t>(row) * 16u + 14u] == expectedMaster,
                "note master did not resolve to its current resident row or absence");
    }
    for (int row = 0; row < frame.linkRowCount; ++row)
    {
        const auto id = frame.linkIdentities[static_cast<std::size_t>(row)];
        const auto source = std::find_if(score.links.begin(), score.links.end(),
                                         [id](const auto& link) { return link.id == id; });
        require(source != score.links.end(), "resident row invented a link identity");
        const auto offset = static_cast<std::size_t>(row) * 4u;
        for (std::size_t endpoint = 0; endpoint < 2; ++endpoint)
        {
            const float value = frame.linkTextureValues[offset + endpoint];
            const int noteRow = static_cast<int>(value);
            require(value == noteRow && noteRow >= 0 && noteRow < frame.noteRowCount,
                    "resident link endpoint escaped the note texture");
            require(frame.noteIdentities[static_cast<std::size_t>(noteRow)]
                        == (endpoint == 0 ? source->slaveNoteId : source->masterNoteId),
                    "resident link changed its source endpoint identity");
        }
        require(frame.linkTextureValues[offset + 2] == source->slaveHarmonic
                    && frame.linkTextureValues[offset + 3] == source->masterHarmonic,
                "resident link changed its harmonic ratio");
    }
}

void testLongScoreResidency()
{
    auto score = projectedLoopScore();
    score->notes.clear();
    score->links.clear();
    score->historyBeats = score->lookaheadBeats = 0;
    // Eighty separated four-note chords. The last chord and its six edges are
    // beyond both texture capacities in the complete source vectors.
    for (int chord = 0; chord < 80; ++chord)
    {
        const int master = -(chord * 4 + 1);
        for (int voice = 0; voice < 4; ++voice)
        {
            auto value = note(master - voice, static_cast<float>(chord * 4), 60.0f + voice, 220.0f);
            value.lengthBeats = 3;
            value.linkMasterId = voice == 0 ? 0 : master;
            score->notes.push_back(value);
            for (int other = 0; other < voice; ++other)
                score->links.push_back({-1000 - static_cast<int>(score->links.size()),
                                        value.id, master - other, voice + 1, other + 1, 0});
        }
    }
    require(score->notes.size() == 320 && score->links.size() == 480,
            "long-score fixture does not exceed both source capacities");
    std::string diagnostic;
    require(FrameProducer::admissible(key(), score, 0, &diagnostic) && diagnostic.empty(),
            "complete long score was rejected before resident selection");
    canonicalblockc::OwnerIdentity identity {1,2,3,4,5,6,7,8,9,10};
    canonicalblockc::FrameOwner preview, exported;
    const auto beatAtFrame = [](std::int64_t frame) { return frame == 2 ? 316.0 : frame * 4.0; };
    std::shared_ptr<const CanonicalBlockCFrame> last;
    for (std::int64_t frame = 0; frame < 3; ++frame)
    {
        const auto actual = preview.frameAt(
            canonicalblockc::PreviewFrameRequest {identity,frame,60.0}, score, beatAtFrame);
        const auto expected = exported.frameAt(
            canonicalblockc::ExportFrameRequest {identity,frame,60.0}, score, beatAtFrame);
        require(canonicalblockc::valid(actual) && canonicalblockc::valid(expected),
                "long score did not produce valid preview/export frames");
        require(actual->noteRowCount == 4 && actual->linkRowCount == 6,
                "sparse long score did not select just the current chord");
        requireResidentReferences(*actual, *score);
        require(actual->noteTextureValues == expected->noteTextureValues
                    && actual->linkTextureValues == expected->linkTextureValues
                    && actual->noteIdentities == expected->noteIdentities
                    && actual->linkIdentities == expected->linkIdentities,
                "preview/export warming changed long-score resident rows");
        last = actual;
    }
    require(std::find(last->noteIdentities.begin(), last->noteIdentities.end(), -320)
                != last->noteIdentities.end() && last->linkIdentities[0] == -1474,
            "late score content was truncated to a prefix instead of selected by time");
    const auto packs = preview.packCount();
    require(preview.frameAt(canonicalblockc::PreviewFrameRequest {identity,2,60.0}, score, beatAtFrame) == last
                && preview.packCount() == packs,
            "held long-score frame repacked or changed identity");
    ++identity.seekGeneration;
    const auto replay = preview.frameAt(canonicalblockc::PreviewFrameRequest {identity,2,60.0}, score, beatAtFrame);
    require(canonicalblockc::valid(replay) && replay != last
                && replay->noteIdentities == last->noteIdentities
                && replay->linkTextureValues == last->linkTextureValues,
            "seek lifecycle did not reset and deterministically rebuild long-score rows");

    const auto rejectLate = [&](auto mutate, const char* reason)
    {
        auto malformed = std::make_shared<arbitmod::Score>(*score);
        mutate(*malformed);
        require(!FrameProducer::admissible(key(), malformed, 0, &diagnostic) && diagnostic == reason,
                "nonresident malformed source was missed or masked by a source-capacity rejection");
        FrameProducer producer;
        const auto validFrame = producer.evaluate(key(), score, 0);
        require(validFrame != nullptr && producer.evaluate(key(), malformed, 0) == nullptr,
                "invalid source reused a cached valid long-score frame");
        const auto recovered = producer.evaluate(key(), score, 0);
        require(canonicalblockc::valid(recovered) && recovered != validFrame,
                "rejected long score did not clear the resident cache");
    };
    rejectLate([](auto& s) { s.notes.back().id = s.notes.front().id; }, "Block C note identity is duplicated");
    rejectLate([](auto& s) { s.notes.back().linkMasterId = -99999; }, "Block C note link master is missing");
    rejectLate([](auto& s) { s.notes.back().freqHz = std::numeric_limits<float>::quiet_NaN(); },
               "Block C note identity, timing, pitch, ratio, or notation is invalid");
    rejectLate([](auto& s) { s.links.back().id = s.links.front().id; }, "Block C harmonic link identity is duplicated");
    rejectLate([](auto& s) { s.links.back().masterNoteId = -99999; }, "Block C harmonic link endpoint is missing");
    rejectLate([](auto& s) { s.links.back().slaveHarmonic = 0; }, "Block C harmonic link is invalid");
    auto staleKey = key();
    staleKey.sourceGeneration = 0;
    require(!FrameProducer::admissible(staleKey, score, 0), "long-score admission weakened lifecycle validation");
    require(score->notes.size() == 320 && score->links.size() == 480,
            "packing or validation modified the complete source score");

    // Saturate the resident textures too. Nonresident links come first, so a
    // source-vector prefix cap would discard every eligible resident link.
    auto dense = projectedLoopScore();
    dense->notes.clear(); dense->links.clear();
    for (int id = 1; id <= 160; ++id)
        dense->notes.push_back(note(id, 0, 60, 220));
    dense->notes[0].linkMasterId = 160; // Valid source master outside the selected set.
    for (int edge = 0; edge < 260; ++edge)
        dense->links.push_back({1000 + edge, 160, 159, 3, 2, 0});
    for (int edge = 0; edge < 300; ++edge)
        dense->links.push_back({2000 + edge, 2 + edge % 127, 1, 3, 2, 0});
    FrameProducer denseProducer;
    const auto denseFrame = denseProducer.evaluate(key(), dense, 0);
    require(canonicalblockc::valid(denseFrame) && denseFrame->noteRowCount == 128
                && denseFrame->linkRowCount == 256,
            "dense source did not retain the fixed resident capacities");
    require(denseFrame->noteIdentities[0] == 1 && denseFrame->noteIdentities[127] == 128
                && denseFrame->linkIdentities[0] == 2000 && denseFrame->linkIdentities[255] == 2255,
            "dense source changed deterministic resident priority or eligible link order");
    requireResidentReferences(*denseFrame, *dense);
    auto reversed = std::make_shared<arbitmod::Score>(*dense);
    std::reverse(reversed->notes.begin(), reversed->notes.end());
    FrameProducer reversedProducer;
    const auto reordered = reversedProducer.evaluate(key(), reversed, 0);
    require(canonicalblockc::valid(reordered) && reordered->noteTextureValues == denseFrame->noteTextureValues
                && reordered->linkTextureValues == denseFrame->linkTextureValues,
            "source note storage order changed dense resident rows");
    require(dense->notes.size() == 160 && dense->links.size() == 560,
            "dense packing truncated the source score");
}
}

int main()
{
    testLongScoreResidency();
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
    mapping.appearanceLow = { 0.125f, 0.25f, 1.0f, 0.0f };
    mapping.appearanceHigh = { 1.0f, 0.5f, 0.0f, 4.0f };
    visualnoteinstancing::Mapping decodedMapping;
    const auto encodedMapping = visualnoteinstancing::encode(mapping);
    require(visualnoteinstancing::decode(encodedMapping, decodedMapping)
                && visualnoteinstancing::encode(decodedMapping) == encodedMapping,
            "note appearance mapping did not round trip exactly");
    auto legacyMapping = visualnoteinstancing::Mapping{};
    legacyMapping.schemaVersion = 1;
    const auto legacyMappingBytes = visualnoteinstancing::encode(legacyMapping);
    require(visualnoteinstancing::decode(legacyMappingBytes, decodedMapping)
                && decodedMapping.schemaVersion == 1
                && decodedMapping.appearanceLow == legacyMapping.appearanceLow
                && visualnoteinstancing::encode(decodedMapping) == legacyMappingBytes,
            "legacy note mapping changed its neutral appearance");
    require(!visualnoteinstancing::decode(encodedMapping + "0\n", decodedMapping),
            "note appearance transport accepted trailing values");
    auto invalidAppearance = mapping;
    invalidAppearance.appearanceHigh[3] = 65.0f;
    require(visualnoteinstancing::encode(invalidAppearance).empty(),
            "note emission exceeded its bound");
    invalidAppearance = mapping;
    invalidAppearance.appearanceLow[0] = std::numeric_limits<float>::quiet_NaN();
    require(visualnoteinstancing::encode(invalidAppearance).empty(),
            "note tint accepted a non-finite value");
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
        const auto velocity = first->noteTextureValues[row * valuesPerNote + 1];
        require(instances.appearanceValues[index] == velocity
                    && instances.appearanceLow == mapping.appearanceLow
                    && instances.appearanceHigh == mapping.appearanceHigh,
                "note appearance lost its canonical velocity row or endpoints");
    }
    auto identityInputs = runtimeInputs;
    identityInputs.noteInstanceMapping->appearanceSource
        = visualnoteinstancing::AppearanceSource::StableNoteIdentity;
    const auto identityBatch = arbitgpu::prepareNativeNoteInstances(identityInputs);
    require(identityBatch.appearanceValues[0] != identityBatch.appearanceValues[1],
            "distinct projected notes received the same palette coordinate in this fixture");
    const auto compareAppearance = [&](std::shared_ptr<const CanonicalBlockCFrame> frame)
    {
        identityInputs.canonicalBlockCFrame = std::move(frame);
        const auto next = arbitgpu::prepareNativeNoteInstances(identityInputs);
        require(next.count == identityBatch.count, "note appearance lost an instance");
        for (std::size_t index = 0; index < next.count; ++index)
        {
            const auto previous = std::find(identityBatch.identities.begin(),
                identityBatch.identities.begin() + identityBatch.count, next.identities[index]);
            require(previous != identityBatch.identities.begin() + identityBatch.count,
                    "note appearance changed canonical identity");
            require(next.appearanceValues[index] == identityBatch.appearanceValues[
                        static_cast<std::size_t>(previous - identityBatch.identities.begin())],
                    "note palette changed after reordering, seeking, looping or export warming");
        }
    };
    auto reorderedScore = std::make_shared<arbitmod::Score>(*score);
    std::reverse(reorderedScore->notes.begin(), reorderedScore->notes.end());
    FrameProducer reorderedProducer;
    compareAppearance(reorderedProducer.evaluate(frameKey, reorderedScore, 0.0f));
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
    compareAppearance(afterSeek);
    auto loopKey = frameKey;
    ++loopKey.loopGeneration;
    compareAppearance(producer.evaluate(loopKey, score, 0.0f));

    FrameProducer warmedProducer;
    std::int64_t packedFrame = -1;
    std::shared_ptr<const CanonicalBlockCFrame> warmed;
    auto warmKey = key();
    const auto beatAtFrame = [](std::int64_t frame) { return static_cast<double>(frame) * 0.25; };
    warmed = canonicalblockc::warmFrameProducerTo(
        warmedProducer, packedFrame, warmed, warmKey, score, 2, beatAtFrame);
    requireProjectedIdentities(warmed);
    compareAppearance(warmed);
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

    {
        using namespace videowire::geometry;
        visualharmonicgeometry::Mapping mapping;
        mapping.stableId = 81;
        mapping.notes.zScale = 0.0f;
        mapping.notes.meshScale = 0.2f;
        visualharmonicgeometry::Mapping decoded;
        const auto encoded = visualharmonicgeometry::encode(mapping);
        require(visualharmonicgeometry::decode(encoded, decoded), "harmonic mapping did not reopen");
        require(!visualharmonicgeometry::decode(encoded + "junk", decoded),
                "harmonic mapping accepted trailing bytes");
        auto malformed = mapping;
        malformed.notes.meshScale = 0;
        require(visualharmonicgeometry::encode(malformed).empty(), "zero ribbon width was admitted");
        malformed = mapping;
        malformed.notes.x = static_cast<visualnoteinstancing::MappingAxis>(99);
        require(visualharmonicgeometry::encode(malformed).empty(), "unknown note axis was admitted");
        std::string error;
        const auto geometry = videohelper::harmonicgeometry::lower(mapping, first, error);
        require(geometry.has_value(), "canonical harmonic ribbon was not lowered");
        const auto& mesh = std::get<GeometryData>(geometry->data);
        require(mesh.positions.size() == 4 && mesh.indices.size() == 6, "link did not produce one ribbon");
        require(geometry->attributes[0].elements[0].components[0] == -301
            && geometry->attributes[1].elements[0].components[0] == -1
            && geometry->attributes[2].elements[0].components[0] == -102
            && geometry->attributes[3].elements[0].components[0] == 3
            && geometry->attributes[4].elements[0].components[0] == 2,
            "ribbon lost canonical signed link/note identities or ratio");
        arbitgpu::NativeFixtureSceneRuntimeInputs noteInputs;
        noteInputs.canonicalBlockCFrame = first;
        noteInputs.noteInstanceMapping = mapping.notes;
        const auto noteBatch = arbitgpu::prepareNativeNoteInstances(noteInputs);
        const auto notePosition = [&](std::int64_t id) {
            const auto found = std::find(noteBatch.identities.begin(),
                noteBatch.identities.begin() + static_cast<std::ptrdiff_t>(noteBatch.count), id);
            require(found != noteBatch.identities.begin() + static_cast<std::ptrdiff_t>(noteBatch.count),
                    "ribbon endpoint has no matching note instance");
            return noteBatch.transforms[static_cast<std::size_t>(found - noteBatch.identities.begin())];
        };
        const auto masterPosition = notePosition(-1), slavePosition = notePosition(-102);
        require(std::abs((mesh.positions[0].x + mesh.positions[1].x) * 0.5f - masterPosition[0]) < 0.00001f
            && std::abs((mesh.positions[2].y + mesh.positions[3].y) * 0.5f - slavePosition[1]) < 0.00001f,
            "ribbon endpoints disagree with Note Instanced Mesh mapping");
        const auto contract = visualharmonicgeometry::geometryContract();
        const auto admitted = admitValue(*geometry, contract, {}, error);
        require(admitted.has_value(), "harmonic geometry failed immutable Geometry Core admission");
        const auto bytes = lowerRuntimePlan(contract, *admitted);
        const auto wire = decodeLoweredRuntimePlan(bytes, error);
        require(wire.has_value() && decodeRuntimeValue(wire->runtimeValue, wire->contract, {}, {}, error).has_value(),
                "harmonic geometry transport lost its attributes");
        FrameProducer exportProducer;
        const auto exported = videohelper::harmonicgeometry::lower(mapping,
            exportProducer.evaluate(key(), score, 0.0f), error);
        const auto exportAdmission = exported ? admitValue(*exported, contract, {}, error) : std::nullopt;
        require(exportAdmission.has_value() && lowerRuntimePlan(contract, *exportAdmission) == bytes,
                "independent preview/export frames produced different ribbons");
        FrameProducer reorderedLinksProducer;
        const auto reordered = videohelper::harmonicgeometry::lower(mapping,
            reorderedLinksProducer.evaluate(key(), reorderedScore, 0.0f), error);
        require(reordered.has_value() && std::get<GeometryData>(reordered->data).vertexIds == mesh.vertexIds,
                "note array reordering changed ribbon corner identities");
        auto coincident = mapping;
        coincident.notes.xScale = coincident.notes.yScale = coincident.notes.zScale = 0;
        const auto empty = videohelper::harmonicgeometry::lower(coincident, first, error);
        require(empty.has_value() && std::get<GeometryData>(empty->data).positions.empty(),
                "coincident notes produced invalid geometry");
        const auto noScore = videohelper::harmonicgeometry::lower(mapping, {}, error);
        require(noScore.has_value() && std::get<GeometryData>(noScore->data).positions.empty(),
                "an empty score did not produce an empty ribbon set");
        auto fullScore = std::make_shared<arbitmod::Score>(*score);
        fullScore->links.clear();
        for (std::size_t index = 0; index < visualharmonicgeometry::kMaximumLinks; ++index)
            fullScore->links.push_back({-static_cast<int>(index) - 1, -102, -1, 3, 2, 0});
        FrameProducer fullProducer;
        const auto fullGeometry = videohelper::harmonicgeometry::lower(mapping,
            fullProducer.evaluate(key(), fullScore, 0), error);
        require(fullGeometry.has_value()
            && std::get<GeometryData>(fullGeometry->data).positions.size() == 1024
            && admitValue(*fullGeometry, contract, {}, error).has_value(),
            "the canonical 256-link bound did not produce an admitted ribbon set");
        fullScore->links.push_back({-257, -102, -1, 3, 2, 0});
        FrameProducer overflowProducer;
        const auto overflow = overflowProducer.evaluate(key(), fullScore, 0);
        require(canonicalblockc::valid(overflow) && overflow->linkRowCount == 256
                    && overflow->linkIdentities[255] == -256 && fullScore->links.size() == 257,
                "source links beyond resident capacity rejected or changed the bounded frame");
    }
    {
        using namespace videowire::geometry;
        scorefield::Binding binding;
        binding.fieldStableId=11; binding.sourceGeometryStableId=12; binding.scoreSourceStableId=13;
        binding.positions={{{0,1,0}}}; binding.elementIds={7}; binding.direction={1,1,1};
        const auto frame=scoreFieldFrame();
        videohelper::scorefield::Evaluation evaluated;
        std::string error;
        for (unsigned mode=0;mode<5;++mode) {
            binding.mode=static_cast<scorefield::Mode>(mode); binding.radius=2;
            require(videohelper::scorefield::evaluate(binding,frame,evaluated,error),error.c_str());
            require(evaluated.closestNoteIds==std::vector<std::int64_t>{-1},
                    "Score Field did not retain the closest signed projected note ID");
            const auto result=evaluated;
            require(videohelper::scorefield::evaluate(binding,scoreFieldFrame(true),evaluated,error),error.c_str());
            require(result.offsets==evaluated.offsets && result.closestNoteIds==evaluated.closestNoteIds
                    && result.closestLinkIds==evaluated.closestLinkIds,
                    "Score Field changed after canonical input reordering");
            if (mode==0) require(std::abs(evaluated.offsets[0][0]-1)<1.0e-6,
                                 "closest-note distance is wrong");
            if (mode==1) require(std::abs(evaluated.offsets[0][0]-0.5f)<1.0e-6,
                                 "velocity-weighted linear-radius influence is wrong");
            if (mode==2) require(std::abs(evaluated.offsets[0][0]-12)<1.0e-6,
                                 "frame-resolved nearest distinct pitch spacing is wrong");
            if (mode==3) require(evaluated.offsets[0]==std::array<float,3>{1,2,3},
                                 "canonical prime exponents did not reach the vector field");
            if (mode==4) {
                require(evaluated.closestLinkIds==std::vector<std::int64_t>{-301}
                        && evaluated.linkRatios[0]==std::array<float,2>{2,1},
                        "link-distance field lost canonical link identity or ratio");
                require(std::abs(evaluated.offsets[0][0]-std::sqrt(0.8f))<1.0e-6,
                        "link-distance field did not measure the closest point on the segment");
            }
        }
        require(!videohelper::scorefield::evaluate(binding,{},evaluated,error)
                && error.find("canonical score frame")!=std::string::npos,
                "Score Field silently accepted a missing frame");
        require(!videohelper::scorefield::evaluate(binding,scoreFieldFrame(false,true),evaluated,error)
                && error.find("no resident unmuted notes")!=std::string::npos,
                "Score Field silently accepted an empty score");
        require(!videohelper::scorefield::evaluate(binding,scoreFieldFrame(false,false,false),evaluated,error)
                && error.find("no resident harmonic links")!=std::string::npos,
                "link-distance field silently accepted absent links");
        binding.mode=scorefield::Mode::velocityInfluence;
        binding.positions.resize(scorefield::kMaximumElements+1);
        binding.elementIds.resize(binding.positions.size());
        require(!scorefield::validate(binding,binding.positions.size(),error),
                "Score Field sample bound was not enforced");
    }
    {
        auto springScore = projectedLoopScore();
        FrameProducer springProducer;
        const auto frame = springProducer.evaluate(key(), springScore, 0);
        const auto original = frame->noteTextureValues;
        const auto springs = videorender::particleNoteUpload(frame.get(), true);
        require(frame->noteTextureValues == original,
                "particle springs changed the canonical Block C frame");
        require(videorender::particleNoteUpload(frame.get(), false) == original,
                "particle springs changed the accumulated-motion upload");
        // Signed ID -102 sorts before -1. Their 3:2 link pulls toward the other
        // endpoint with symmetric stiffness, despite different note row order.
        const float distance = (67.0f - 60.0f) / 60.0f * 0.8f;
        require(springs[0] == 67.0f && springs[16] == 60.0f
                && std::abs(springs[8] + distance * 1.5f) < 1.0e-6f
                && std::abs(springs[24] - distance * 1.5f) < 1.0e-6f
                && springs[10] == 1.5f && springs[26] == 1.5f,
                "harmonic particle springs lost signed endpoints or 3:2 weighting");
        const auto unweighted = videorender::particleNoteUpload(frame.get(), true, 0);
        require(unweighted[10] == 1 && unweighted[26] == 1
                && std::abs(unweighted[8] + distance) < 1.0e-6f,
                "link ratio weighting did not change spring stiffness");
        springScore->links.push_back({-99, -102, -1, 7, 4, 0});
        springScore->links.push_back({-701, -102, -1, 5, 3, 0});
        FrameProducer orderedProducer, reversedProducer;
        const auto ordered = videorender::particleNoteUpload(
            orderedProducer.evaluate(key(), springScore, 0).get(), true);
        std::reverse(springScore->notes.begin(), springScore->notes.end());
        std::reverse(springScore->links.begin(), springScore->links.end());
        const auto reversed = videorender::particleNoteUpload(
            reversedProducer.evaluate(key(), springScore, 0).get(), true);
        require(ordered == reversed, "canonical link order changed spring summation");
        auto priorScore = std::make_shared<arbitmod::Score>(*springScore);
        priorScore->notes.erase(std::remove_if(priorScore->notes.begin(), priorScore->notes.end(),
            [](const auto& note) { return note.id == -1; }), priorScore->notes.end());
        priorScore->notes[0].linkMasterId = 0;
        priorScore->links.clear();
        FrameProducer residentProducer;
        require(residentProducer.evaluate(key(), priorScore, 0) != nullptr,
                "spring residency fixture failed");
        auto next = key(); next.frame = 1;
        const auto resident = videorender::particleNoteUpload(
            residentProducer.evaluate(next, springScore, 0).get(), true);
        for (int row = 0; row < 2; ++row)
            for (int channel = 8; channel < 12; ++channel)
                require(resident[static_cast<std::size_t>(row * 16 + channel)]
                        == ordered[static_cast<std::size_t>(row * 16 + channel)],
                        "canonical residency row reuse changed particle endpoints");
        springScore->links.clear();
        for (int link = 0; link < arbitblockc::kMaxLinks; ++link)
            springScore->links.push_back({-link - 1, -102, -1, 100, 1, 0});
        FrameProducer capacityProducer;
        const auto capacity = videorender::particleNoteUpload(
            capacityProducer.evaluate(key(), springScore, 0).get(), true);
        require(capacity[10] == 8 && capacity[26] == 8
                && capacity[11] == 256 && capacity[27] == 256,
                "dense harmonic links exceeded the bounded spring stiffness");
        springScore->notes[0].startBeat = 1;
        FrameProducer upcomingProducer;
        const auto upcoming = videorender::particleNoteUpload(
            upcomingProducer.evaluate(key(), springScore, 0).get(), true);
        require(upcoming[10] == 0 && upcoming[26] == 0,
                "a nonsounding harmonic endpoint attracted particles");
    }
    std::cout << "canonical Block C projected identity, harmonic geometry, score fields and particle springs passed\n";
    return 0;
}
