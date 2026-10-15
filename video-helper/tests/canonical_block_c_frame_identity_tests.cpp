#include "canonical_block_c_frame.h"
#include "block_c_frame_owner.h"
#include "gpu_backend/backend.h"
#include "harmonic_link_geometry.h"
#include "particle_parameters.h"
#include "score_json_parser.h"
#include "score_field_fixture.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <vector>

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

void testDirectNoteInstanceAxes()
{
    using Axis = visualnoteinstancing::MappingAxis;
    constexpr auto directVersion = visualnoteinstancing::kDirectAxesMappingSchemaVersion;
    const std::string legacyGolden =
        "visual.note-instances.mapping.v1\n1 0 1 4 1065353216 1065353216 1065353216 1036831949 0\n";
    const std::string defaultGolden =
        "visual.note-instances.mapping.v2\n2 0 1 4 1065353216 1065353216 1065353216 1036831949 0\n"
        "0 1065353216 1065353216 1065353216 0 1065353216 1065353216 1065353216 0\n";
    const std::string directGolden =
        "visual.note-instances.mapping.v3\n3 5 6 16 1065353216 1065353216 1065353216 1036831949 0\n"
        "0 1065353216 1065353216 1065353216 0 1065353216 1065353216 1065353216 0\n";
    const std::string velocityGolden =
        "visual.note-instances.mapping.v4\n4 5 6 16 1065353216 1065353216 1065353216 1036831949 0\n"
        "0 1065353216 1065353216 1065353216 0 1065353216 1065353216 1065353216 0\n"
        "1056964608 1073741824\n";
    visualnoteinstancing::Mapping mapping, decoded;
    require(mapping.schemaVersion == 2 && visualnoteinstancing::encode(mapping) == defaultGolden,
            "default note mapping migrated from its exact v2 bytes");
    mapping.schemaVersion = 1;
    require(visualnoteinstancing::encode(mapping) == legacyGolden,
            "legacy note mapping changed its exact v1 bytes");
    for (const auto& golden : {legacyGolden, defaultGolden, directGolden})
        require(visualnoteinstancing::decode(golden, decoded)
                    && decoded.velocityScaleLow == 1.0f && decoded.velocityScaleHigh == 1.0f
                    && visualnoteinstancing::encode(decoded) == golden,
                "note mapping golden failed canonical round trip or changed neutral scale");
    require(decoded.x == Axis::PitchBendSemitones && decoded.y == Axis::NoteAgeBeats
                && decoded.z == Axis::Prime13Exponent,
            "v3 golden changed appended axis ordinals");
    require(visualnoteinstancing::decode(velocityGolden, decoded)
                && decoded.schemaVersion == visualnoteinstancing::kVelocityScaleMappingSchemaVersion
                && decoded.velocityScaleLow == 0.5f && decoded.velocityScaleHigh == 2.0f
                && visualnoteinstancing::encode(decoded) == velocityGolden,
            "velocity scale mapping changed its exact v4 bytes");
    const auto velocityMapping = decoded;
    for (const float invalid : {0.0f, -1.0f, 0.0000001f, 10001.0f,
                               std::numeric_limits<float>::infinity(),
                               std::numeric_limits<float>::quiet_NaN()})
        for (const bool high : {false, true})
        {
            auto candidate = velocityMapping;
            (high ? candidate.velocityScaleHigh : candidate.velocityScaleLow) = invalid;
            require(!visualnoteinstancing::valid(candidate) && visualnoteinstancing::encode(candidate).empty(),
                    "velocity response accepted a non-finite or out-of-range endpoint");
            auto wire = velocityGolden.substr(0, velocityGolden.rfind('\n', velocityGolden.size() - 2) + 1);
            wire += std::to_string(visualnoteinstancing::floatBits(candidate.velocityScaleLow)) + " "
                + std::to_string(visualnoteinstancing::floatBits(candidate.velocityScaleHigh)) + "\n";
            require(!visualnoteinstancing::decode(wire, decoded),
                    "velocity response decoder accepted an invalid endpoint");
        }
    auto excessive = velocityMapping; excessive.meshScale = 10000.0f;
    require(!visualnoteinstancing::valid(excessive), "effective note scale exceeded its existing cap");
    excessive = velocityMapping; excessive.meshScale = std::numeric_limits<float>::denorm_min();
    excessive.velocityScaleLow = 0.000001f;
    require(!visualnoteinstancing::valid(excessive), "effective note scale underflowed to zero");
    for (const auto& malformed : {velocityGolden.substr(0, velocityGolden.rfind('\n', velocityGolden.size() - 2) + 1),
                                  velocityGolden + "0\n"})
        require(!visualnoteinstancing::decode(malformed, decoded),
                "v4 accepted missing response endpoints or trailing transport");
    for (std::uint32_t version : {1u, 2u, 3u})
    {
        auto old = visualnoteinstancing::Mapping{}; old.schemaVersion = version;
        old.velocityScaleHigh = 2.0f;
        require(!visualnoteinstancing::valid(old) && visualnoteinstancing::encode(old).empty(),
                "an old mapping silently discarded a velocity response");
    }
    for (std::uint32_t version : {1u, 2u})
        for (unsigned axis = 5; axis <= 16; ++axis)
            for (unsigned coordinate = 0; coordinate < 3; ++coordinate)
            {
                mapping = {};
                mapping.schemaVersion = version;
                Axis* axes[] { &mapping.x, &mapping.y, &mapping.z };
                *axes[coordinate] = static_cast<Axis>(axis);
                require(!visualnoteinstancing::valid(mapping)
                            && visualnoteinstancing::encode(mapping).empty(),
                        "old note schema accepted an appended axis");
            }
    for (const auto& golden : {legacyGolden, defaultGolden})
    {
        auto newAxisInOldSchema = golden;
        const auto axisStart = newAxisInOldSchema.find(' ', newAxisInOldSchema.find('\n')) + 1;
        newAxisInOldSchema.replace(axisStart, 1, "5");
        require(!visualnoteinstancing::decode(newAxisInOldSchema, decoded),
                "old note schema decoded an appended axis ordinal");
    }
    mapping = {}; mapping.schemaVersion = 1; mapping.appearanceLow[0] = 0.5f;
    require(visualnoteinstancing::encode(mapping).empty(), "legacy v1 lost its neutral appearance restriction");
    for (const auto& header : {"visual.note-instances.mapping.v1",
                              "visual.note-instances.mapping.v2",
                              "visual.note-instances.mapping.v3",
                              "visual.note-instances.mapping.v4"})
        for (const auto& golden : {legacyGolden, defaultGolden, directGolden, velocityGolden})
        {
            const auto changed = std::string(header) + golden.substr(golden.find('\n'));
            if (changed != golden)
                require(!visualnoteinstancing::decode(changed, decoded),
                        "note mapping accepted a mismatched header and schema");
        }
    for (const auto& malformed : {
            std::string("visual.note-instances.mapping.v4\n") + directGolden.substr(directGolden.find('\n') + 1),
            directGolden + "0\n", std::string(1025, '0'),
            std::string("visual.note-instances.mapping.v3\n3 17 6 16 1065353216 1065353216 1065353216 1036831949 0\n")
                + directGolden.substr(directGolden.find('\n', directGolden.find('\n') + 1) + 1),
            std::string("visual.note-instances.mapping.v3\n3 5 6 16 2139095040 1065353216 1065353216 1036831949 0\n")
                + directGolden.substr(directGolden.find('\n', directGolden.find('\n') + 1) + 1),
            std::string("visual.note-instances.mapping.v3\n3 5.5 6 16 1065353216 1065353216 1065353216 1036831949 0\n")
                + directGolden.substr(directGolden.find('\n', directGolden.find('\n') + 1) + 1)})
        require(!visualnoteinstancing::decode(malformed, decoded),
                "note mapping accepted malformed or out-of-range transport");
    auto noncanonical = directGolden;
    noncanonical.insert(noncanonical.find('\n') + 1, "0");
    require(!visualnoteinstancing::decode(noncanonical, decoded),
            "note mapping accepted a noncanonical schema spelling");
    auto carriageReturn = directGolden;
    carriageReturn.insert(carriageReturn.find('\n'), "\r");
    require(!visualnoteinstancing::decode(carriageReturn, decoded),
            "note mapping accepted carriage returns");

    auto score = projectedLoopScore();
    score->rootFreq = 220.0f;
    score->notes[0].freqHz = 220.0f;
    score->notes[0].midiNote = 60.5f;
    score->notes[0].velocity = 127.0f;
    score->notes[0].trackId = 3;
    score->notes[0].isRoot = true;
    score->notes[1].freqHz = 440.0f;
    score->notes[1].lengthBeats = 5.0f;
    score->notes[1].velocity = 0.0f;
    score->notes[1].trackId = 5;
    const float firstPrimes[] {2, -3, 5, -7, 11, -13};
    const float secondPrimes[] {-2, 3, -5, 7, -11, 13};
    for (std::size_t prime = 0; prime < 6; ++prime)
    {
        score->notes[0].primes[prime] = firstPrimes[prime];
        score->notes[1].primes[prime] = secondPrimes[prime];
    }
    for (auto& source : score->notes) source.pitchAnchors.clear();
    score->notes[0].pitchBendPoints.push_back({0.0f, 12.0f});
    score->notes[1].pitchBendPoints.push_back({0.0f, -12.0f});
    auto frameKey = key(); frameKey.frame = 4; frameKey.beat = 1.0;
    FrameProducer producer;
    const auto frame = producer.evaluate(frameKey, score, 1.0f);
    require(canonicalblockc::valid(frame), "direct axis producer fixture was rejected");
    const std::array<std::array<float, 16>, 2> literalTexels {{
        {{60.5f, 1, 2, 6, 440, 1200, 3, 1, 2, -3, 5, -7, 11, -13, -1, 12}},
        {{67, 0, 1, 4, 220, 0, 5, 0, -2, 3, -5, 7, -11, 13, 0, -12}}
    }};
    struct ExpectedAxis { Axis axis; std::size_t texelOffset; float first, second; };
    const ExpectedAxis expected[] {
        {Axis::PitchBendSemitones, 15, 12, -12}, {Axis::NoteAgeBeats, 2, 2, 1},
        {Axis::RemainingDurationBeats, 3, 6, 4}, {Axis::RootFlag, 7, 1, 0},
        {Axis::MidiNote, 0, 60.5f, 67}, {Axis::ResolvedFrequencyHz, 4, 440, 220},
        {Axis::Prime2Exponent, 8, 2, -2}, {Axis::Prime3Exponent, 9, -3, 3},
        {Axis::Prime5Exponent, 10, 5, -5}, {Axis::Prime7Exponent, 11, -7, 7},
        {Axis::Prime11Exponent, 12, 11, -11}, {Axis::Prime13Exponent, 13, -13, 13}
    };
    const auto requireLiteralBatch = [&](const auto& current)
    {
        require(canonicalblockc::valid(current) && current->noteRows() == 2
                    && current->noteIdentities[0] == -1 && current->noteIdentities[1] == -102,
                "direct axes changed canonical projected identity or order");
        for (std::size_t row = 0; row < 2; ++row)
            for (std::size_t offset = 0; offset < 16; ++offset)
                require(std::abs(current->noteTextureValues[row * 16 + offset]
                                     - literalTexels[row][offset]) < 0.0001f,
                        "canonical producer changed independently specified direct-axis texels");
        for (const auto& value : expected)
        {
            mapping = {}; mapping.schemaVersion = directVersion;
            mapping.x = mapping.y = mapping.z = value.axis;
            mapping.xScale = 2; mapping.yScale = -3; mapping.zScale = 0.25f;
            mapping.meshScale = 0.5f;
            const auto encoded = visualnoteinstancing::encode(mapping);
            require(visualnoteinstancing::decode(encoded, decoded), "direct axis mapping did not reopen");
            arbitgpu::NativeImportedSceneRuntimeInputs inputs;
            inputs.canonicalBlockCFrame = current; inputs.noteInstanceMapping = decoded;
            require(arbitgpu::validFixtureRuntimeInputs(inputs), "finite direct-axis inputs were rejected");
            const auto batch = arbitgpu::prepareNativeNoteInstances(inputs);
            require(batch.admitted && batch.count == 2, "direct axes lost admitted instances");
            for (std::size_t row = 0; row < 2; ++row)
            {
                const auto literal = row == 0 ? value.first : value.second;
                require(current->noteTextureValues[row * 16 + value.texelOffset] == literal,
                        "direct-axis texel differs from its source literal");
                require(batch.identities[row] == (row == 0 ? -1 : -102)
                            && batch.canonicalRows[row] == row
                            && batch.transforms[row] == std::array<float, 4> {
                                literal * 2, literal * -3, literal * 0.25f, 0.5f}
                            && batch.appearanceValues[row] == (row == 0 ? 1.0f : 0.0f),
                        "direct axis value, scale, appearance or durable identity changed");
            }
        }
        mapping = {};
        mapping.xScale = 2; mapping.yScale = -3; mapping.zScale = 0.25f;
        arbitgpu::NativeImportedSceneRuntimeInputs legacyInputs;
        legacyInputs.canonicalBlockCFrame = current; legacyInputs.noteInstanceMapping = mapping;
        const auto legacyBatch = arbitgpu::prepareNativeNoteInstances(legacyInputs);
        require(legacyBatch.transforms[0] == std::array<float, 4> {-4, -3, 0.75f, 0.1f}
                    && legacyBatch.transforms[1] == std::array<float, 4> {-2, 0, 1.25f, 0.1f},
                "legacy onset, log frequency or track value changed");
        for (std::uint32_t version : {1u, 2u})
        {
            legacyInputs.noteInstanceMapping->schemaVersion = version;
            legacyInputs.noteInstanceMapping->x = Axis::Duration;
            legacyInputs.noteInstanceMapping->y = Axis::Velocity;
            const auto old = arbitgpu::prepareNativeNoteInstances(legacyInputs);
            require(old.transforms[0] == std::array<float, 4> {16, -3, 0.75f, 0.1f}
                        && old.transforms[1] == std::array<float, 4> {10, 0, 1.25f, 0.1f},
                    "legacy duration or velocity value changed");
        }
        arbitgpu::NativeImportedSceneRuntimeInputs scaleInputs;
        scaleInputs.canonicalBlockCFrame = current;
        mapping = velocityMapping; mapping.x = Axis::Onset; mapping.y = Axis::LogFrequency;
        mapping.z = Axis::Track; mapping.xScale = 2; mapping.yScale = -3; mapping.zScale = 0.25f;
        scaleInputs.noteInstanceMapping = mapping;
        require(!arbitgpu::validFixtureRuntimeInputs(scaleInputs),
                "unqualified native velocity-scale transport was admitted");
        const auto scaled = arbitgpu::prepareNativeNoteInstances(scaleInputs);
        require(scaled.admitted && scaled.count == 2 && scaled.identities == legacyBatch.identities
                    && scaled.canonicalRows == legacyBatch.canonicalRows
                    && scaled.transforms[0] == std::array<float, 4> {-4, -3, 0.75f, 0.2f}
                    && scaled.transforms[1] == std::array<float, 4> {-2, 0, 1.25f, 0.05f}
                    && scaled.appearanceValues == legacyBatch.appearanceValues,
                "velocity endpoint scales changed positions, canonical IDs, rows or appearance");
        const auto prior = arbitgpu::sceneNoteMotionTransforms(scaleInputs);
        require(prior && prior->find(-1) && prior->find(-102)
                    && *prior->find(-1) == scaled.transforms[0] && *prior->find(-102) == scaled.transforms[1],
                "stable-ID temporal transforms lost effective per-note scale");
        scaleInputs.noteInstanceMapping->appearanceSource = visualnoteinstancing::AppearanceSource::StableNoteIdentity;
        const auto palette = arbitgpu::prepareNativeNoteInstances(scaleInputs);
        require(palette.transforms == scaled.transforms && palette.identities == scaled.identities
                    && palette.appearanceValues != scaled.appearanceValues,
                "stable-ID appearance aliased the velocity scale driver");
        scaleInputs.noteInstanceMapping->velocityScaleLow = 2.0f;
        scaleInputs.noteInstanceMapping->velocityScaleHigh = 0.5f;
        const auto descending = arbitgpu::prepareNativeNoteInstances(scaleInputs);
        require(descending.transforms[0][3] == 0.05f && descending.transforms[1][3] == 0.2f,
                "descending velocity response reversed canonical row ownership");
        scaleInputs.noteInstanceMapping->velocityScaleLow = 10000.0f;
        scaleInputs.noteInstanceMapping->velocityScaleHigh = 0.000001f;
        require(arbitgpu::prepareNativeNoteInstances(scaleInputs).transforms[0][3] > 0.0f,
                "descending response cancellation collapsed the high-velocity note");
    };
    requireLiteralBatch(frame);
    auto reversed = std::make_shared<arbitmod::Score>(*score);
    std::reverse(reversed->notes.begin(), reversed->notes.end());
    FrameProducer reverseProducer;
    requireLiteralBatch(reverseProducer.evaluate(frameKey, reversed, 1.0f));
    ++frameKey.seekGeneration;
    requireLiteralBatch(producer.evaluate(frameKey, score, 1.0f));
    ++frameKey.loopGeneration;
    requireLiteralBatch(producer.evaluate(frameKey, score, 1.0f));
    FrameProducer warmProducer;
    std::int64_t packedFrame = -1;
    std::shared_ptr<const CanonicalBlockCFrame> warmed;
    requireLiteralBatch(canonicalblockc::warmFrameProducerTo(warmProducer, packedFrame, warmed,
        frameKey, score, 4, [](std::int64_t value) { return static_cast<double>(value) * 0.25; }));

    for (const auto beat : {-2.0f, 8.0f})
    {
        auto temporalKey = frameKey; temporalKey.beat = beat;
        FrameProducer temporalProducer;
        arbitgpu::NativeImportedSceneRuntimeInputs temporalInputs;
        temporalInputs.canonicalBlockCFrame = temporalProducer.evaluate(temporalKey, score, beat);
        mapping = {}; mapping.schemaVersion = directVersion;
        mapping.x = Axis::NoteAgeBeats; mapping.y = Axis::RemainingDurationBeats;
        mapping.z = Axis::PitchBendSemitones;
        temporalInputs.noteInstanceMapping = mapping;
        require(arbitgpu::validFixtureRuntimeInputs(temporalInputs), "temporal direct axes were rejected");
        const auto temporal = arbitgpu::prepareNativeNoteInstances(temporalInputs);
        const std::array<float, 4> first = beat == -2.0f
            ? std::array<float, 4> {-1, 9, 12, 0.1f} : std::array<float, 4> {9, -1, 12, 0.1f};
        const std::array<float, 4> second = beat == -2.0f
            ? std::array<float, 4> {-2, 7, -12, 0.1f} : std::array<float, 4> {8, -3, -12, 0.1f};
        require(temporal.admitted && temporal.count == 2 && temporal.identities[0] == -1
                    && temporal.identities[1] == -102 && temporal.transforms[0] == first
                    && temporal.transforms[1] == second,
                "upcoming age or ended remaining duration lost its signed literal value or identity");
    }

    auto zeroScore = std::make_shared<arbitmod::Score>(*score);
    zeroScore->notes[0].pitchBendPoints[0].semitones = -2400.0f;
    FrameProducer zeroProducer;
    const auto zeroFrame = zeroProducer.evaluate(frameKey, zeroScore, 1.0f);
    require(canonicalblockc::valid(zeroFrame) && zeroFrame->noteTextureValues[4] == 0.0f,
            "valid packed zero resolved Hz changed producer eligibility");
    arbitgpu::NativeImportedSceneRuntimeInputs zeroInputs;
    zeroInputs.canonicalBlockCFrame = zeroFrame;
    zeroInputs.noteInstanceMapping = visualnoteinstancing::Mapping{};
    require(arbitgpu::validFixtureRuntimeInputs(zeroInputs), "packed zero Hz rejected legacy mapping");
    zeroInputs.noteInstanceMapping->schemaVersion = directVersion;
    zeroInputs.noteInstanceMapping->x = Axis::ResolvedFrequencyHz;
    require(arbitgpu::validFixtureRuntimeInputs(zeroInputs)
                && arbitgpu::prepareNativeNoteInstances(zeroInputs).transforms[0][0] == 0.0f,
            "direct resolved Hz replaced a packed zero with a positive floor");

    visualharmonicgeometry::Mapping links; links.stableId = 81;
    const auto oldLinks = visualharmonicgeometry::encode(links);
    require(oldLinks == "visual.harmonic-link-geometry.v1\n81\n" + defaultGolden,
            "harmonic geometry changed its old outer or nested bytes");
    links.notes.schemaVersion = directVersion; links.notes.x = Axis::MidiNote;
    links.notes.y = Axis::Prime13Exponent;
    visualharmonicgeometry::Mapping decodedLinks;
    const auto directLinks = visualharmonicgeometry::encode(links);
    require(visualharmonicgeometry::decode(directLinks, decodedLinks)
                && visualharmonicgeometry::encode(decodedLinks) == directLinks,
            "neutral v3 harmonic mapping did not canonically reopen");
    std::string error;
    const auto linkValue = videohelper::harmonicgeometry::lower(decodedLinks, frame, error);
    require(linkValue && std::get<videowire::geometry::GeometryData>(linkValue->data).positions.size() == 4
                && linkValue->attributes[0].elements[0].components[0] == -301
                && linkValue->attributes[1].elements[0].components[0] == -1
                && linkValue->attributes[2].elements[0].components[0] == -102,
            "v3 harmonic endpoints lost exact signed canonical identities");
    for (unsigned refusal = 0; refusal < 5; ++refusal)
    {
        auto invalidLinks = links;
        if (refusal == 0) invalidLinks.notes.schemaVersion = 1;
        if (refusal == 1) invalidLinks.notes.includeMuted = true;
        if (refusal == 2) invalidLinks.notes.appearanceSource = visualnoteinstancing::AppearanceSource::StableNoteIdentity;
        if (refusal == 3) invalidLinks.notes.appearanceHigh[3] = 1;
        if (refusal == 4) invalidLinks.notes = velocityMapping;
        require(visualharmonicgeometry::encode(invalidLinks).empty(),
                "harmonic geometry relaxed an unrelated nested restriction");
    }

    const auto requireAtomicRefusal = [&](const auto& refusedFrame, Axis axis, float scale)
    {
        arbitgpu::NativeImportedSceneRuntimeInputs inputs;
        inputs.canonicalBlockCFrame = refusedFrame;
        mapping = {}; mapping.schemaVersion = directVersion; mapping.x = axis; mapping.xScale = scale;
        inputs.noteInstanceMapping = mapping;
        require(canonicalblockc::valid(refusedFrame) && visualnoteinstancing::valid(mapping),
                "coordinate refusal fixture lacks authentic admitted finite source data");
        require(!arbitgpu::validFixtureRuntimeInputs(inputs), "out-of-bounds v3 runtime inputs were admitted");
        const auto refused = arbitgpu::prepareNativeNoteInstances(inputs);
        const arbitgpu::NativeNoteInstanceBatch empty;
        require(!refused.admitted && refused.count == 0 && refused.transforms == empty.transforms
                    && refused.identities == empty.identities && refused.canonicalRows == empty.canonicalRows
                    && refused.appearanceValues == empty.appearanceValues
                    && refused.appearanceLow == empty.appearanceLow && refused.appearanceHigh == empty.appearanceHigh
                    && refused.meshScale == empty.meshScale,
                "v3 coordinate refusal retained a partial instance or identity");
        links.notes = mapping;
        require(!videohelper::harmonicgeometry::lower(links, refusedFrame, error)
                    && error == "Harmonic link endpoint coordinates exceed the geometry bounds",
                "v3 harmonic coordinate refusal became an empty successful geometry");
        inputs.noteInstanceMapping->schemaVersion = 2;
        inputs.noteInstanceMapping->x = Axis::Velocity;
        require(arbitgpu::validFixtureRuntimeInputs(inputs), "v3 guard altered legacy runtime eligibility");
    };
    auto hugeScore = std::make_shared<arbitmod::Score>(*score);
    hugeScore->notes[1].pitchBendPoints.clear();
    hugeScore->notes[1].freqHz = std::numeric_limits<float>::max() / 2.0f;
    FrameProducer hugeProducer;
    const auto hugeFrame = hugeProducer.evaluate(frameKey, hugeScore, 1.0f);
    requireAtomicRefusal(hugeFrame, Axis::ResolvedFrequencyHz, 4.0f);
    auto boundedScore = std::make_shared<arbitmod::Score>(*score);
    boundedScore->notes[1].primes[0] = 100.0f;
    FrameProducer boundedProducer;
    const auto boundedFrame = boundedProducer.evaluate(frameKey, boundedScore, 1.0f);
    mapping = {}; mapping.schemaVersion = directVersion; mapping.x = Axis::Prime2Exponent; mapping.xScale = 10000;
    arbitgpu::NativeImportedSceneRuntimeInputs boundInputs;
    boundInputs.canonicalBlockCFrame = boundedFrame; boundInputs.noteInstanceMapping = mapping;
    require(arbitgpu::validFixtureRuntimeInputs(boundInputs)
                && arbitgpu::prepareNativeNoteInstances(boundInputs).transforms[1][0] == 1000000.0f,
            "v3 coordinate ceiling rejected its exact boundary");
    boundedScore->notes[1].primes[0] = 101.0f;
    FrameProducer outsideProducer;
    requireAtomicRefusal(outsideProducer.evaluate(frameKey, boundedScore, 1.0f), Axis::Prime2Exponent, 10000.0f);
    auto emptyScore = std::make_shared<arbitmod::Score>(*score);
    emptyScore->notes.clear(); emptyScore->links.clear();
    FrameProducer emptyProducer;
    boundInputs.canonicalBlockCFrame = emptyProducer.evaluate(frameKey, emptyScore, 1.0f);
    require(arbitgpu::validFixtureRuntimeInputs(boundInputs), "empty v3 score lost successful admission");
    const auto emptyBatch = arbitgpu::prepareNativeNoteInstances(boundInputs);
    require(emptyBatch.admitted && emptyBatch.count == 0, "empty v3 score became an atomic refusal");
    links.notes = mapping;
    const auto emptyLinks = videohelper::harmonicgeometry::lower(links, boundInputs.canonicalBlockCFrame, error);
    require(emptyLinks && std::get<videowire::geometry::GeometryData>(emptyLinks->data).positions.empty()
                && error.empty(), "empty v3 harmonic score became a coordinate refusal");
}

// Per-note identity checks. Every assertion is keyed by note id, never by the
// texture row that happens to hold the note in a given frame.
struct NoteTexels final
{
    std::array<float, 16> values {};
    std::int64_t masterIdentity = 0;

    bool operator== (const NoteTexels& other) const noexcept
    {
        return values == other.values && masterIdentity == other.masterIdentity;
    }
};

struct NoteSnapshot final
{
    std::map<std::int64_t, NoteTexels> texels;
    std::map<std::int64_t, float> appearance;
};

std::map<std::int64_t, NoteTexels> identityTexels(const CanonicalBlockCFrame& frame)
{
    constexpr std::size_t valuesPerNote = arbitblockc::kTexelsPerNote * 4u;
    // texel 3, channel 2 carries the link master's row.
    constexpr std::size_t masterChannel = 3u * 4u + 2u;
    std::map<std::int64_t, NoteTexels> result;
    for (int row = 0; row < frame.noteRows(); ++row)
    {
        const auto identity = frame.noteIdentities[static_cast<std::size_t>(row)];
        if (identity == 0) continue;
        NoteTexels texels;
        const auto offset = static_cast<std::size_t>(row) * valuesPerNote;
        for (std::size_t channel = 0; channel < texels.values.size(); ++channel)
            texels.values[channel] = frame.noteTextureValues[offset + channel];
        const float masterRow = texels.values[masterChannel];
        if (masterRow != -1.0f)
        {
            const auto master = static_cast<int>(masterRow);
            require(master >= 0 && master < frame.noteRows()
                        && static_cast<float>(master) == masterRow,
                    "resident note master escaped the resident rows");
            texels.masterIdentity = frame.noteIdentities[static_cast<std::size_t>(master)];
        }
        texels.values[masterChannel] = 0.0f;
        require(result.emplace(identity, texels).second, "one note identity occupied two rows");
    }
    return result;
}

std::map<std::int64_t, float> appearanceById(const std::shared_ptr<const CanonicalBlockCFrame>& frame)
{
    arbitgpu::NativeImportedSceneRuntimeInputs inputs;
    inputs.canonicalBlockCFrame = frame;
    visualnoteinstancing::Mapping mapping;
    mapping.appearanceSource = visualnoteinstancing::AppearanceSource::StableNoteIdentity;
    inputs.noteInstanceMapping = mapping;
    const auto batch = arbitgpu::prepareNativeNoteInstances(inputs);
    require(batch.admitted && batch.count <= visualnoteinstancing::kMaximumInstances,
            "note appearance batch was not admitted");
    std::map<std::int64_t, float> result;
    for (std::size_t index = 0; index < batch.count; ++index)
        require(result.emplace(batch.identities[index], batch.appearanceValues[index]).second,
                "one note identity received two appearance instances");
    return result;
}

NoteSnapshot snapshotOf(const std::shared_ptr<const CanonicalBlockCFrame>& frame)
{
    require(canonicalblockc::valid(frame), "note identity frame was not valid");
    return NoteSnapshot {identityTexels(*frame), appearanceById(frame)};
}

// A fresh producer has no row history, so its per-note values are the reference.
NoteSnapshot coldSnapshot(const std::shared_ptr<const arbitmod::Score>& score,
                          const canonicalblockc::OwnerIdentity& identity,
                          std::int64_t frame, float beat)
{
    auto frameKey = canonicalblockc::makeFrameKey(identity);
    frameKey.fps = 60.0;
    frameKey.frame = frame;
    frameKey.beat = static_cast<double>(beat);
    FrameProducer producer;
    return snapshotOf(producer.evaluate(frameKey, score, beat));
}

// Eviction and re-entry, loop wrap, seek and preview/export through one owner
// sequence. The score holds 200 notes, so the 128 resident rows cannot hold all.
void testNoteIdentityAcrossEvictionLoopSeekAndExport()
{
    auto score = projectedLoopScore();
    score->notes.clear();
    score->links.clear();
    score->historyBeats = 1000.0f;
    score->lookaheadBeats = 0.0f;
    for (int index = 0; index < 200; ++index)
    {
        auto value = note(-(index + 1), static_cast<float>(index) * 0.25f,
                          60.0f + static_cast<float>(index % 12), 261.6256f);
        value.lengthBeats = 0.25f;
        score->notes.push_back(value);
    }
    require(score->notes.size() > static_cast<std::size_t>(arbitblockc::kMaxNotes),
            "per-note fixture no longer exceeds resident capacity");

    struct Request final
    {
        canonicalblockc::OwnerIdentity identity;
        std::int64_t frame;
    };
    const canonicalblockc::OwnerIdentity base {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto looped = base;
    ++looped.loopGeneration;
    auto seeked = looped;
    ++seeked.seekGeneration;
    // Beat is 10 on even frames and 60 on odd frames. Request 2 keeps the frame
    // index increasing while the beat drops back to 10, so the packer keeps its
    // rows across that jump. Request 4 decreases the frame index, request 5 is a
    // loop wrap, and request 6 is a seek.
    const std::array<Request, 7> requests {{
        {base, 0}, {base, 1}, {base, 2}, {base, 3}, {base, 1}, {looped, 2}, {seeked, 3}}};
    const auto beatOf = [](std::int64_t frame) { return frame % 2 == 1 ? 60.0 : 10.0; };

    canonicalblockc::FrameOwner previewOwner, exportOwner;
    std::vector<NoteSnapshot> snapshots;
    for (const auto& request : requests)
    {
        const auto beat = static_cast<float>(beatOf(request.frame));
        const auto previewed = snapshotOf(previewOwner.frameAt(
            canonicalblockc::PreviewFrameRequest {request.identity, request.frame, 60.0},
            score, beatOf));
        const auto exported = snapshotOf(exportOwner.frameAt(
            canonicalblockc::ExportFrameRequest {request.identity, request.frame, 60.0},
            score, beatOf));
        const auto expected = coldSnapshot(score, request.identity, request.frame, beat);
        require(previewed.texels == expected.texels && previewed.appearance == expected.appearance,
                "preview frame changed a note's values relative to a fresh evaluation");
        require(exported.texels == expected.texels && exported.appearance == expected.appearance,
                "export frame changed a note's values relative to a fresh evaluation");
        require(previewed.texels == exported.texels && previewed.appearance == exported.appearance,
                "preview and export disagree per note id");
        snapshots.push_back(previewed);
    }

    const auto& atTen = snapshots[0];
    const auto& evicted = snapshots[1];
    const auto& reentered = snapshots[2];
    require(evicted.texels.size() == static_cast<std::size_t>(arbitblockc::kMaxNotes),
            "beat 60 did not fill the resident rows");
    require(atTen.texels.count(-6) == 1 && evicted.texels.count(-6) == 0
                && reentered.texels.count(-6) == 1,
            "note -6 was not evicted at beat 60 and re-admitted at beat 10");
    require(reentered.appearance.at(-6) == atTen.appearance.at(-6)
                && reentered.texels.at(-6) == atTen.texels.at(-6),
            "an evicted note that re-entered changed its appearance or note values");
    require(snapshots[4].appearance == evicted.appearance && snapshots[4].texels == evicted.texels,
            "a decreasing frame index changed the per-note mapping");
    require(snapshots[5].appearance == atTen.appearance && snapshots[5].texels == atTen.texels,
            "a loop wrap changed the per-note mapping");
    require(snapshots[6].appearance == snapshots[3].appearance
                && snapshots[6].texels == snapshots[3].texels,
            "a seek bump changed the per-note mapping");
}

// A released note in its history tail keeps its id and appearance while
// historyBeats keeps it resident.
void testReleasedNoteTailKeepsIdentity()
{
    auto score = projectedLoopScore();
    score->notes.clear();
    score->links.clear();
    score->lookaheadBeats = 0.0f;
    score->historyBeats = 2.0f;
    for (int index = 0; index < 8; ++index)
    {
        auto value = note(-(index + 1), static_cast<float>(index) * 0.25f,
                          60.0f + static_cast<float>(index), 261.6256f);
        value.lengthBeats = 0.25f;
        score->notes.push_back(value);
    }
    const canonicalblockc::OwnerIdentity base {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    const auto sounding = coldSnapshot(score, base, 0, 0.1f);
    const auto tail = coldSnapshot(score, base, 1, 1.0f);
    require(sounding.texels.count(-1) == 1 && tail.texels.count(-1) == 1,
            "released note left the history window before its tail ended");
    require(tail.appearance.at(-1) == sounding.appearance.at(-1),
            "released tail note changed its appearance");
    require(tail.texels.at(-1).values[3] < 0.0f,
            "tail fixture did not place the note after its release");
    score->historyBeats = 0.0f;
    require(coldSnapshot(score, base, 1, 1.0f).texels.count(-1) == 0,
            "zero history kept the released note, so the tail check does not discriminate");
}
}

int main()
{
    testLongScoreResidency();
    testNoteIdentityAcrossEvictionLoopSeekAndExport();
    testReleasedNoteTailKeepsIdentity();
    testDirectNoteInstanceAxes();
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
        videohelper::harmonicprobe::Receipt receipt;
        const auto geometry = videohelper::harmonicgeometry::lower(mapping, first, error, &receipt);
        require(geometry.has_value(), "canonical harmonic ribbon was not lowered");
        const auto& mesh = std::get<GeometryData>(geometry->data);
        require(mesh.positions.size() == 4 && mesh.indices.size() == 6, "link did not produce one ribbon");
        require(first->noteRows() == 2 && first->linkRows() == 1
            && receipt.lowerCallCount == 1 && receipt.emittedVertexCount == 4
            && receipt.emittedIndexCount == 6,
            "two-note one-link probe did not report its emitted ribbon");
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
        videohelper::harmonicprobe::Receipt emptyReceipt;
        const auto noScore = videohelper::harmonicgeometry::lower(mapping, {}, error, &emptyReceipt);
        require(noScore.has_value() && std::get<GeometryData>(noScore->data).positions.empty(),
                "an empty score did not produce an empty ribbon set");
        require(emptyReceipt.lowerCallCount == 1 && emptyReceipt.emittedVertexCount == 0
            && emptyReceipt.emittedIndexCount == 0,
            "an empty score reported harmonic geometry that was not emitted");
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
        OperationRecord sample; sample.stableId=11; sample.inputStableId=12;
        sample.field.mode=3; sample.field.values={2,1,0,1,13,0,0,0};
        FieldData sampled;
        require(videohelper::scorefield::evaluateSampleField(sample,FieldData{{FieldElement{{0,1,0,0}}}},
                    frame,sampled,error) && std::abs(sampled.elements[0].components[0]-std::sqrt(0.8))<1e-6,
                "generic Score Sample Field mode 3 lost its separate harmonic-distance alias");
        {
            // Choices 11 through 13 must equal the vector X, Y and Z components on the same note.
            scorefield::Binding lattice=binding;
            lattice.mode=scorefield::Mode::primeLattice; lattice.radius=2; lattice.gain=1; lattice.bias=0;
            lattice.direction={1,1,1}; lattice.mapping.xScale=lattice.mapping.yScale=lattice.mapping.zScale=1;
            videohelper::scorefield::Evaluation reference;
            require(videohelper::scorefield::evaluate(lattice,frame,reference,error),error.c_str());
            require(reference.offsets[0]==std::array<float,3>{1,2,3},
                    "vector Prime Lattice fixture no longer has its default prime components");
            for (std::uint8_t choice=scorefield::kFirstSamplePrimeLatticeChoice;
                 choice<scorefield::kSampleFieldModeCount;++choice) {
                OperationRecord latticeSample; latticeSample.stableId=11; latticeSample.inputStableId=12;
                latticeSample.field.mode=choice; latticeSample.field.values={2,1,0,1,13,0,0,0};
                FieldData latticeSampled;
                require(videohelper::scorefield::evaluateSampleField(latticeSample,FieldData{{FieldElement{{0,1,0,0}}}},
                            frame,latticeSampled,error) && latticeSampled.elements.size()==1,
                        "Prime Lattice Score Sample choice did not evaluate");
                require(latticeSampled.elements[0].components[0]
                            ==reference.offsets[0][choice-scorefield::kFirstSamplePrimeLatticeChoice],
                        "Prime Lattice Score Sample choice differs from its vector component");
            }
            {
                // Gain 2.5 and bias 0.75 apply after the lattice slot is read, on both paths.
                scorefield::Binding scaled=lattice; scaled.gain=2.5f; scaled.bias=0.75f;
                videohelper::scorefield::Evaluation scaledReference;
                require(videohelper::scorefield::evaluate(scaled,frame,scaledReference,error),error.c_str());
                require(scaledReference.offsets[0]==std::array<float,3>{3.25f,5.75f,8.25f},
                        "scaled vector Prime Lattice components did not apply gain and bias");
                for (std::uint8_t choice=scorefield::kFirstSamplePrimeLatticeChoice;
                     choice<scorefield::kSampleFieldModeCount;++choice) {
                    OperationRecord scaledSample; scaledSample.stableId=11; scaledSample.inputStableId=12;
                    scaledSample.field.mode=choice; scaledSample.field.values={2,2.5,0.75,1,13,0,0,0};
                    FieldData scaledSampled;
                    require(videohelper::scorefield::evaluateSampleField(scaledSample,FieldData{{FieldElement{{0,1,0,0}}}},
                                frame,scaledSampled,error) && scaledSampled.elements.size()==1,
                            "scaled Prime Lattice Score Sample choice did not evaluate");
                    require(scaledSampled.elements[0].components[0]
                                ==scaledReference.offsets[0][choice-scorefield::kFirstSamplePrimeLatticeChoice],
                            "scaled Prime Lattice Score Sample choice differs from its vector component");
                }
            }
            OperationRecord refused=sample; refused.field.mode=scorefield::kSampleFieldModeCount;
            FieldData refusedOut;
            require(!videohelper::scorefield::evaluateSampleField(refused,FieldData{{FieldElement{{0,1,0,0}}}},
                        frame,refusedOut,error) && error.find("unknown measurement")!=std::string::npos,
                    "Score Sample Field accepted a choice one past the appended Prime Lattice range");
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
        using namespace videowire::geometry;
        auto score=scoreFieldMeasureScore();
        for (auto& note : score->notes) {
            note.startBeat=0;
            note.pitchBendPoints={{0,-2400},{1,-2400}};
            for (std::size_t prime=0;prime<6;++prime) note.primes[prime]=static_cast<float>(prime+1);
        }
        FrameProducer producer;
        const auto frame=producer.evaluate(scoreFieldMeasureKey(0),score,0);
        require(canonicalblockc::valid(frame) && frame->noteRows()==2 && frame->linkRows()==1,
                "finite extreme bend did not produce a canonical frame");
        for (int row=0;row<frame->noteRows();++row)
            require(frame->noteTextureValues[static_cast<std::size_t>(row)*16+4]==0,
                    "finite extreme bend did not underflow to packed zero frequency");
        scorefield::Binding binding;
        binding.fieldStableId=11; binding.sourceGeometryStableId=12; binding.scoreSourceStableId=13;
        binding.positions={{{0,0,0}}}; binding.elementIds={7};
        binding.mapping.xScale=binding.mapping.yScale=binding.mapping.zScale=1;
        binding.radius=100; binding.direction={1,1,1};
        videohelper::scorefield::Evaluation evaluated;
        std::string error;
        // Both notes map to log2(0.000001 / 440), retaining the legacy frequency clamp.
        const std::array<std::array<float,3>,5> expected {{
            {28.712928f,28.712928f,28.712928f},
            {1.4257414f,1.4257414f,1.4257414f},
            {0,0,0}, {1,2,3}, {28.712928f,28.712928f,28.712928f}}};
        for (unsigned mode=0;mode<5;++mode) {
            binding.mode=static_cast<scorefield::Mode>(mode);
            require(videohelper::scorefield::evaluate(binding,frame,evaluated,error),error.c_str());
            require(evaluated.closestNoteIds==std::vector<std::int64_t>{-102},
                    "zero-frequency legacy field lost signed note tie identity");
            for (std::size_t axis=0;axis<3;++axis)
                require(std::isfinite(evaluated.offsets[0][axis])
                            && std::abs(evaluated.offsets[0][axis]-expected[mode][axis])<1e-5f,
                        "zero-frequency legacy field changed its literal finite behavior");
            if (mode==4)
                require(evaluated.closestLinkIds==std::vector<std::int64_t>{-301}
                            && evaluated.linkRatios[0]==std::array<float,2>{3,2},
                        "zero-frequency legacy field lost harmonic link identity or ratio");
        }
        OperationRecord sample; sample.stableId=11; sample.inputStableId=12;
        sample.field.mode=3; sample.field.values={100,1,0,1,13,0,0,0};
        FieldData sampled;
        require(videohelper::scorefield::evaluateSampleField(sample,FieldData{{FieldElement{{0,0,0,0}}}},
                    frame,sampled,error)
                    && std::isfinite(sampled.elements[0].components[0])
                    && std::abs(sampled.elements[0].components[0]-28.712928)<1e-5,
                "zero-frequency generic mode 3 lost its legacy harmonic-distance behavior");
        binding.mode=scorefield::Mode::resolvedFrequencyHz;
        require(videohelper::scorefield::evaluate(binding,frame,evaluated,error)
                    && evaluated.closestNoteIds==std::vector<std::int64_t>{-102}
                    && evaluated.offsets[0]==std::array<float,3>{0,0,0},
                "resolved frequency did not preserve the canonical packed zero");
    }
    {
        using namespace videowire::geometry;
        const std::array<scorefield::Mode,7> modes {
            scorefield::Mode::pitchBendSemitones,scorefield::Mode::noteAgeBeats,
            scorefield::Mode::remainingDurationBeats,scorefield::Mode::rootFlag,
            scorefield::Mode::midiNoteNumber,scorefield::Mode::resolvedFrequencyHz,
            scorefield::Mode::harmonicLinkRatio};
        const std::array<double,7> expected {-12,2,6,1,69,440,1.5};
        auto score=scoreFieldMeasureScore();
        FrameProducer producer;
        const auto frame=producer.evaluate(scoreFieldMeasureKey(),score,2);
        require(canonicalblockc::valid(frame),"direct Score Field fixture did not produce a canonical frame");
        scorefield::Binding binding;
        binding.fieldStableId=11; binding.sourceGeometryStableId=12; binding.scoreSourceStableId=13;
        binding.positions={{{-2,0,0}}}; binding.elementIds={7};
        binding.gain=0.25f; binding.bias=2; binding.direction={1,-2,0.5f};
        videohelper::scorefield::Evaluation evaluated;
        std::string error;
        auto reversedScore=std::make_shared<arbitmod::Score>(*score);
        std::reverse(reversedScore->notes.begin(),reversedScore->notes.end());
        std::reverse(reversedScore->links.begin(),reversedScore->links.end());
        FrameProducer reversedProducer;
        const auto reversed=reversedProducer.evaluate(scoreFieldMeasureKey(),reversedScore,2);
        for (std::size_t i=0;i<modes.size();++i) {
            binding.mode=modes[i];
            require(videohelper::scorefield::evaluate(binding,frame,evaluated,error),error.c_str());
            require(evaluated.closestNoteIds==std::vector<std::int64_t>{-1},
                    "direct measurement did not select the nearest signed note identity");
            for (std::size_t axis=0;axis<3;++axis)
                require(std::abs(evaluated.offsets[0][axis]-(expected[i]*0.25+2)*binding.direction[axis])<1e-6,
                        "direct measurement lost its literal packed value, gain, bias or direction");
            if (modes[i]==scorefield::Mode::harmonicLinkRatio)
                require(evaluated.closestLinkIds==std::vector<std::int64_t>{-301}
                        && evaluated.linkRatios[0]==std::array<float,2>{3,2},
                        "direct harmonic ratio lost its exact numerator and denominator");
            const auto first=evaluated;
            require(videohelper::scorefield::evaluate(binding,reversed,evaluated,error),error.c_str());
            require(first.offsets==evaluated.offsets && first.closestNoteIds==evaluated.closestNoteIds
                    && first.closestLinkIds==evaluated.closestLinkIds && first.linkRatios==evaluated.linkRatios,
                    "direct measurements depended on source storage order");
        }
        binding.gain=1; binding.bias=0; binding.direction={1,0,0};
        // A fresh history frame assigns the sounding note before the ended root.
        // A continuous producer keeps the opposite row order at the same beat.
        const auto historyKey=scoreFieldMeasureKey(9);
        const auto continuous=producer.evaluate(historyKey,score,9);
        FrameProducer historyProducer;
        const auto fresh=historyProducer.evaluate(historyKey,score,9);
        require(canonicalblockc::valid(continuous) && canonicalblockc::valid(fresh),
                "direct measurement history fixtures did not produce canonical frames");
        require(continuous->noteIdentities[0]!=fresh->noteIdentities[0],
                "direct measurement fixture did not exercise actual canonical row replacement");
        binding.positions[0]={-9,0,0};
        for (const auto mode : modes) {
            binding.mode=mode;
            require(videohelper::scorefield::evaluate(binding,continuous,evaluated,error),error.c_str());
            const auto first=evaluated;
            require(videohelper::scorefield::evaluate(binding,fresh,evaluated,error),error.c_str());
            require(first.offsets==evaluated.offsets && first.closestNoteIds==evaluated.closestNoteIds
                    && first.closestLinkIds==evaluated.closestLinkIds && first.linkRatios==evaluated.linkRatios,
                    "direct measurements reused values from another canonical row");
            if (mode==scorefield::Mode::noteAgeBeats) require(evaluated.offsets[0][0]==9,"note age did not advance");
            if (mode==scorefield::Mode::remainingDurationBeats)
                require(evaluated.offsets[0][0]==-1,"ended note lost its negative remaining duration");
            binding.positions[0]={-2,0,0};
            require(videohelper::scorefield::evaluate(binding,frame,evaluated,error),error.c_str());
            const auto i=static_cast<unsigned>(mode)-5;
            require(std::abs(evaluated.offsets[0][0]-expected[i])<1e-6,
                    "returning to an earlier frame retained a later musical value");
            binding.positions[0]={-9,0,0};
        }
        auto replacement=std::make_shared<arbitmod::Score>(*score);
        replacement->notes[0].id=-7; replacement->notes[1].linkMasterId=-7;
        replacement->links[0].masterNoteId=-7;
        replacement->notes[0].pitchBendPoints={{0,12},{1,12}};
        replacement->notes[0].midiNote=72; replacement->notes[0].isRoot=false;
        auto replacementKey=scoreFieldMeasureKey(); ++replacementKey.sourceGeneration; ++replacementKey.seekGeneration;
        const auto replaced=producer.evaluate(replacementKey,replacement,2);
        binding.positions[0]={-2,2,0};
        const std::array<double,7> changed {12,2,6,0,72,1760,1.5};
        for (std::size_t i=0;i<modes.size();++i) {
            binding.mode=modes[i];
            require(videohelper::scorefield::evaluate(binding,replaced,evaluated,error),error.c_str());
            require(evaluated.closestNoteIds==std::vector<std::int64_t>{-7}
                    && std::abs(evaluated.offsets[0][0]-changed[i])<1e-6,
                    "lifecycle replacement lost bend, root, MIDI number or resolved frequency causality");
        }
        auto upcoming=std::make_shared<arbitmod::Score>(*score);
        upcoming->notes[0].startBeat=4;
        FrameProducer upcomingProducer;
        const auto upcomingFrame=upcomingProducer.evaluate(scoreFieldMeasureKey(),upcoming,2);
        binding.positions[0]={2,0,0}; binding.mode=scorefield::Mode::noteAgeBeats;
        require(videohelper::scorefield::evaluate(binding,upcomingFrame,evaluated,error)
                && evaluated.offsets[0][0]==-2,"lookahead note lost its negative age");
        binding.mode=scorefield::Mode::remainingDurationBeats;
        require(videohelper::scorefield::evaluate(binding,upcomingFrame,evaluated,error)
                && evaluated.offsets[0][0]==10,"lookahead note remaining duration is wrong");
        binding.mapping.xScale=binding.mapping.yScale=binding.mapping.zScale=0;
        binding.positions[0]={0,0,0};
        const std::array<double,7> tied {0,0,8,0,81,880,1.5};
        for (std::size_t i=0;i<modes.size();++i) {
            binding.mode=modes[i];
            require(videohelper::scorefield::evaluate(binding,frame,evaluated,error),error.c_str());
            require(evaluated.closestNoteIds==std::vector<std::int64_t>{-102}
                    && std::abs(evaluated.offsets[0][0]-tied[i])<1e-6,
                    "direct measurement tie did not select the lowest signed note identity");
        }
        auto nearestScore=std::make_shared<arbitmod::Score>(*score);
        auto third=nearestScore->notes[1]; third.id=-205; third.freqHz=440; third.midiNote=65;
        nearestScore->notes.push_back(third); nearestScore->links.push_back({-1001,-205,-102,5,4,0});
        FrameProducer nearestProducer;
        const auto nearestFrame=nearestProducer.evaluate(scoreFieldMeasureKey(),nearestScore,2);
        binding.mapping.xScale=binding.mapping.yScale=binding.mapping.zScale=1;
        binding.positions[0]={-2,0,0}; binding.mode=scorefield::Mode::harmonicLinkRatio;
        require(videohelper::scorefield::evaluate(binding,nearestFrame,evaluated,error)
                && evaluated.closestLinkIds==std::vector<std::int64_t>{-301}
                && evaluated.offsets[0][0]==1.5f,
                "harmonic ratio selected a distant lower-ID segment instead of the nearest link");
        binding.mapping.xScale=binding.mapping.yScale=binding.mapping.zScale=0;
        binding.positions[0]={0,0,0};
        score->links.push_back({-999,-1,-102,2,3,0});
        FrameProducer linkProducer;
        const auto linkFrame=linkProducer.evaluate(scoreFieldMeasureKey(),score,2);
        binding.mode=scorefield::Mode::harmonicLinkRatio;
        require(videohelper::scorefield::evaluate(binding,linkFrame,evaluated,error)
                && evaluated.closestLinkIds==std::vector<std::int64_t>{-999}
                && evaluated.linkRatios[0]==std::array<float,2>{2,3}
                && std::abs(evaluated.offsets[0][0]-2.0/3.0)<1e-6,
                "harmonic ratio lost slave/master direction or signed link tie ordering");
        std::reverse(score->links.begin(),score->links.end());
        FrameProducer reorderedLinks;
        const auto reordered=reorderedLinks.evaluate(scoreFieldMeasureKey(),score,2);
        require(videohelper::scorefield::evaluate(binding,reordered,evaluated,error)
                && evaluated.closestLinkIds==std::vector<std::int64_t>{-999}
                && evaluated.linkRatios[0]==std::array<float,2>{2,3},
                "harmonic ratio tie depended on canonical link storage order");
        binding.mode=scorefield::Mode::harmonicLinkDistance;
        require(videohelper::scorefield::evaluate(binding,reordered,evaluated,error)
                && evaluated.closestLinkIds==std::vector<std::int64_t>{-999},
                "distance and ratio did not share signed link selection");
        binding.mode=scorefield::Mode::harmonicLinkRatio;
        require(!videohelper::scorefield::evaluate(binding,scoreFieldFrame(false,false,false),evaluated,error)
                && error.find("no resident harmonic links")!=std::string::npos,
                "direct harmonic ratio accepted a frame without links");
        const auto rejectSource=[&](auto mutate) {
            auto invalid=scoreFieldMeasureScore(); mutate(*invalid);
            FrameProducer rejectingProducer;
            const auto rejected=rejectingProducer.evaluate(scoreFieldMeasureKey(),invalid,2);
            binding.mode=scorefield::Mode::harmonicLinkRatio;
            require(!rejected && !videohelper::scorefield::evaluate(binding,rejected,evaluated,error),
                    "malformed musical data bypassed canonical refusal");
        };
        rejectSource([](auto& invalid) { invalid.links[0].masterHarmonic=0; });
        rejectSource([](auto& invalid) { invalid.links[0].slaveHarmonic=-1; });
        rejectSource([](auto& invalid) { invalid.notes[0].midiNote=std::numeric_limits<float>::quiet_NaN(); });
        rejectSource([](auto& invalid) { invalid.notes[0].pitchBendPoints[0].semitones=std::numeric_limits<float>::infinity(); });
        rejectSource([](auto& invalid) { invalid.notes[0].lengthBeats=std::numeric_limits<float>::infinity(); });
        rejectSource([](auto& invalid) { invalid.notes[0].freqHz=0; });
        binding.mode=static_cast<scorefield::Mode>(12);
        require(!scorefield::validate(binding,1,error),"Score Field accepted an unknown appended ordinal");
        binding.mode=scorefield::Mode::resolvedFrequencyHz; binding.gain=1000; binding.direction={1000,0,0};
        require(!videohelper::scorefield::evaluate(binding,frame,evaluated,error)
                && error.find("displacement")!=std::string::npos,
                "direct frequency bypassed finite displacement limits");
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
