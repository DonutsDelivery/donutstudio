#include "canonical_block_c_frame.h"
#include "block_c_frame_owner.h"
#include "gpu_backend/backend.h"
#include "harmonic_link_geometry.h"
#include "particle_parameters.h"
#include "score_json_parser.h"
#include "score_field_fixture.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

// Row (voice slot) reassignment must never move a note's identity or its
// per-identity data. The packer keeps a resident note's row while it stays
// selected, and a reset or fresh packer reassigns rows by sort order. Each
// check below compares identity-keyed output against a fresh evaluation, and
// asserts that the row really moved so the comparison can fail.
namespace
{
using canonicalblockc::CanonicalBlockCFrame;
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

arbitmod::Note timedNote(int id, float startBeat, float lengthBeats, float midiNote)
{
    arbitmod::Note result;
    result.id = id;
    result.durableKind = 2;
    result.durableNoteId = -1;
    result.clipId = 41;
    result.clipNoteId = 12;
    result.repeatIndex = 3;
    result.trackId = 1;
    result.startBeat = startBeat;
    result.lengthBeats = lengthBeats;
    result.midiNote = midiNote;
    result.velocity = 96.0f;
    result.freqHz = 261.6256f;
    result.durationSeconds = 4.0f;
    result.pitchAnchors.push_back({-701 + id, 0.0f, 261.6256f});
    return result;
}

std::shared_ptr<arbitmod::Score> makeScore(float historyBeats)
{
    auto score = std::make_shared<arbitmod::Score>();
    score->notationVersion = 1;
    score->scoreRevision = 77;
    score->edoStepsPerOctave = 12;
    score->rootFreq = 440.0f;
    score->historyBeats = historyBeats;
    score->lookaheadBeats = 0.0f;
    return score;
}

canonicalblockc::FrameKey frameKeyAt(const canonicalblockc::OwnerIdentity& identity,
                                     std::int64_t frame, float beat)
{
    auto frameKey = canonicalblockc::makeFrameKey(identity);
    frameKey.fps = 60.0;
    frameKey.frame = frame;
    frameKey.beat = static_cast<double>(beat);
    return frameKey;
}

// Fresh producer: no row history, so its rows are the sort-order reference.
std::shared_ptr<const CanonicalBlockCFrame> coldFrame(
    const std::shared_ptr<const arbitmod::Score>& score,
    const canonicalblockc::OwnerIdentity& identity, std::int64_t frame, float beat)
{
    FrameProducer producer;
    return producer.evaluate(frameKeyAt(identity, frame, beat), score, beat);
}

int rowOf(const CanonicalBlockCFrame& frame, std::int64_t identity)
{
    for (int row = 0; row < frame.noteRows(); ++row)
        if (frame.noteIdentities[static_cast<std::size_t>(row)] == identity) return row;
    return -1;
}

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
    std::map<std::int64_t, std::array<float, 4>> transforms;

    bool operator== (const NoteSnapshot& other) const noexcept
    {
        return texels == other.texels && appearance == other.appearance
            && transforms == other.transforms;
    }
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

NoteSnapshot snapshotOf(const std::shared_ptr<const CanonicalBlockCFrame>& frame)
{
    require(canonicalblockc::valid(frame), "note identity frame was not valid");
    arbitgpu::NativeImportedSceneRuntimeInputs inputs;
    inputs.canonicalBlockCFrame = frame;
    visualnoteinstancing::Mapping mapping;
    mapping.appearanceSource = visualnoteinstancing::AppearanceSource::StableNoteIdentity;
    inputs.noteInstanceMapping = mapping;
    const auto batch = arbitgpu::prepareNativeNoteInstances(inputs);
    require(batch.admitted && batch.count <= visualnoteinstancing::kMaximumInstances,
            "note instance batch was not admitted");
    NoteSnapshot result {identityTexels(*frame), {}, {}};
    for (std::size_t index = 0; index < batch.count; ++index)
    {
        const auto identity = batch.identities[index];
        require(result.appearance.emplace(identity, batch.appearanceValues[index]).second,
                "one note identity received two appearance instances");
        require(result.transforms.emplace(identity, batch.transforms[index]).second,
                "one note identity received two motion transforms");
    }
    return result;
}

void requireMidi(const NoteSnapshot& snapshot, std::int64_t identity, float midiNote,
                 const char* message)
{
    const auto found = snapshot.texels.find(identity);
    require(found != snapshot.texels.end() && found->second.values[0] == midiNote, message);
}

// A held note keeps its identity and per-identity data when other notes start
// before it and end, which moves its row in a fresh allocation.
void testHeldNoteRowChangeKeepsIdentity()
{
    constexpr int endingNote = -21;
    constexpr int startingNote = -22;
    constexpr int heldNote = -23;
    auto score = makeScore(4.0f);
    score->notes.push_back(timedNote(endingNote, 0.0f, 1.2f, 60.0f));
    score->notes.push_back(timedNote(startingNote, 0.5f, 0.5f, 62.0f));
    score->notes.push_back(timedNote(heldNote, 1.0f, 6.0f, 64.0f));
    const canonicalblockc::OwnerIdentity identity {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    const std::array<float, 4> beats {0.25f, 0.75f, 1.1f, 1.5f};

    FrameProducer warm;
    std::array<int, 4> warmRow {};
    std::array<int, 4> coldRow {};
    NoteSnapshot firstHeld;
    for (std::size_t frame = 0; frame < beats.size(); ++frame)
    {
        const auto warmFrame = warm.evaluate(
            frameKeyAt(identity, static_cast<std::int64_t>(frame), beats[frame]), score, beats[frame]);
        require(canonicalblockc::valid(warmFrame), "held-note warm frame was not valid");
        const auto cold = coldFrame(score, identity, static_cast<std::int64_t>(frame), beats[frame]);
        require(canonicalblockc::valid(cold), "held-note cold frame was not valid");
        const auto warmSnapshot = snapshotOf(warmFrame);
        const auto coldSnapshot = snapshotOf(cold);
        require(warmSnapshot == coldSnapshot,
                "held note's identity-keyed values or motion changed with its voice slot");
        requireMidi(warmSnapshot, heldNote, 64.0f, "held note identity does not carry its own pitch");
        requireMidi(warmSnapshot, startingNote, 62.0f, "starting note identity does not carry its own pitch");
        warmRow[frame] = rowOf(*warmFrame, heldNote);
        coldRow[frame] = rowOf(*cold, heldNote);
        require(warmRow[frame] >= 0 && coldRow[frame] >= 0, "held note left the resident rows");
        if (frame == 0) firstHeld = warmSnapshot;
        else require(warmSnapshot.appearance.at(heldNote) == firstHeld.appearance.at(heldNote),
                     "held note appearance changed across frames");
    }
    // Frames 2 and 3: the warm allocator keeps the held note's row, while a
    // fresh allocation moves it. The identity check above is only meaningful
    // when that row difference exists.
    require(warmRow[2] == warmRow[3], "packer moved a held note's row while it stayed resident");
    require(coldRow[2] != warmRow[2] && coldRow[3] != warmRow[3],
            "fixture did not reallocate the held note's row");
}

// A released note keeps its identity and per-identity data when it re-enters
// after a seek, even though the fresh allocation gives it a different row.
void testReleasedNoteReentryKeepsIdentity()
{
    constexpr int earlyNote = -35;
    constexpr int releasedNote = -31;
    auto score = makeScore(2.0f);
    score->notes.push_back(timedNote(earlyNote, 0.0f, 0.5f, 65.0f));
    score->notes.push_back(timedNote(releasedNote, 0.25f, 1.0f, 67.0f));
    canonicalblockc::OwnerIdentity identity {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    FrameProducer producer;
    const auto sounding = producer.evaluate(frameKeyAt(identity, 0, 1.0f), score, 1.0f);
    require(canonicalblockc::valid(sounding), "released-note sounding frame was not valid");
    const auto soundingSnapshot = snapshotOf(sounding);
    const int soundingRow = rowOf(*sounding, releasedNote);
    require(soundingRow == 0, "sounding note did not take the first row");

    // The note is released into history, then a seek re-enters the note.
    const auto released = producer.evaluate(frameKeyAt(identity, 1, 1.5f), score, 1.5f);
    require(canonicalblockc::valid(released) && rowOf(*released, releasedNote) == soundingRow,
            "released note did not keep its row while resident in history");

    ++identity.seekGeneration;
    const auto reentered = producer.evaluate(frameKeyAt(identity, 2, 0.4f), score, 0.4f);
    require(canonicalblockc::valid(reentered), "re-entered frame was not valid");
    const int reenteredRow = rowOf(*reentered, releasedNote);
    require(reenteredRow >= 0 && reenteredRow != soundingRow,
            "fixture did not reassign the re-entered note to a different row");

    const auto reenteredSnapshot = snapshotOf(reentered);
    const auto cold = snapshotOf(coldFrame(score, identity, 2, 0.4f));
    require(reenteredSnapshot == cold,
            "re-entered note's identity-keyed values differ from a fresh evaluation");
    requireMidi(reenteredSnapshot, releasedNote, 67.0f, "re-entered identity does not carry its own pitch");
    requireMidi(reenteredSnapshot, earlyNote, 65.0f, "early note identity does not carry its own pitch");
    require(reenteredSnapshot.appearance.at(releasedNote) == soundingSnapshot.appearance.at(releasedNote),
            "re-entered note changed its appearance across its release");
}

// Two notes exchange voice order between frames. The warm allocator keeps each
// note's row, and the fresh allocation swaps the rows. Identity must follow the
// note, not the row.
void testSwappedVoiceOrderKeepsIdentity()
{
    constexpr int firstNote = -41;
    constexpr int secondNote = -42;
    auto score = makeScore(2.0f);
    score->notes.push_back(timedNote(firstNote, 0.0f, 1.0f, 60.0f));
    score->notes.push_back(timedNote(secondNote, 0.5f, 4.0f, 67.0f));
    const canonicalblockc::OwnerIdentity identity {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    const std::array<float, 3> beats {0.25f, 0.75f, 1.5f};

    FrameProducer warm;
    std::shared_ptr<const CanonicalBlockCFrame> last;
    for (std::size_t frame = 0; frame < beats.size(); ++frame)
    {
        last = warm.evaluate(frameKeyAt(identity, static_cast<std::int64_t>(frame), beats[frame]),
                             score, beats[frame]);
        require(canonicalblockc::valid(last), "swapped-order warm frame was not valid");
        const auto cold = coldFrame(score, identity, static_cast<std::int64_t>(frame), beats[frame]);
        require(snapshotOf(last) == snapshotOf(cold),
                "swapped voice order changed a note's identity-keyed values");
    }
    require(rowOf(*last, firstNote) == 0 && rowOf(*last, secondNote) == 1,
            "packer swapped resident voice rows that should keep their slots");
    const auto cold = coldFrame(score, identity, 2, beats[2]);
    require(rowOf(*cold, secondNote) == 0 && rowOf(*cold, firstNote) == 1,
            "fixture did not swap the fresh voice order");
    const auto snapshot = snapshotOf(last);
    requireMidi(snapshot, firstNote, 60.0f, "first note identity took the second note's pitch");
    requireMidi(snapshot, secondNote, 67.0f, "second note identity took the first note's pitch");
}

// A note keeps one identity from note-on through its release tail until the
// history window frees its voice. remain = endBeat - beat turns negative in the
// tail, so the check fails if the tail is not really the released state.
void testNoteIdentityThroughReleaseTailUntilVoiceFreed()
{
    constexpr int releasedNote = -51;
    auto score = makeScore(2.0f);
    score->notes.push_back(timedNote(releasedNote, 0.0f, 1.0f, 64.0f));
    const canonicalblockc::OwnerIdentity identity {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    // Sounding, two tail frames inside historyBeats, then past the window.
    const std::array<float, 4> beats {0.5f, 1.5f, 2.5f, 3.25f};

    FrameProducer warm;
    NoteSnapshot sounding;
    for (std::size_t frame = 0; frame < beats.size(); ++frame)
    {
        const auto frameIndex = static_cast<std::int64_t>(frame);
        const auto warmFrame = warm.evaluate(frameKeyAt(identity, frameIndex, beats[frame]), score,
                                             beats[frame]);
        require(canonicalblockc::valid(warmFrame), "release-tail warm frame was not valid");
        const auto warmSnapshot = snapshotOf(warmFrame);
        const auto cold = snapshotOf(coldFrame(score, identity, frameIndex, beats[frame]));
        require(warmSnapshot == cold,
                "released note's identity-keyed values differ from a fresh evaluation");

        if (frame + 1 == beats.size())
        {
            require(rowOf(*warmFrame, releasedNote) < 0 && warmSnapshot.texels.count(releasedNote) == 0,
                    "released note kept its voice after the history window closed");
            continue;
        }
        require(rowOf(*warmFrame, releasedNote) >= 0,
                "released note left the resident rows before its voice was freed");
        requireMidi(warmSnapshot, releasedNote, 64.0f, "released note identity does not carry its own pitch");
        require(warmSnapshot.appearance.count(releasedNote) == 1,
                "released note lost its appearance instance");
        if (frame == 0)
        {
            require(warmSnapshot.texels.at(releasedNote).values[3] > 0.0f,
                    "fixture did not start the note sounding");
            sounding = warmSnapshot;
            continue;
        }
        require(warmSnapshot.texels.at(releasedNote).values[3] < 0.0f,
                "fixture did not place the note in its release tail");
        require(warmSnapshot.appearance.at(releasedNote) == sounding.appearance.at(releasedNote),
                "released note changed its appearance between note-on and its release tail");
    }
}

// A note struck again on the same pitch is a different note. It gets its own
// identity and voice while the earlier note's release tail keeps its identity.
void testRestruckSamePitchGetsNewIdentityDuringOldTail()
{
    constexpr int releasedNote = -71;
    constexpr int restruckNote = -72;
    auto score = makeScore(2.0f);
    score->lookaheadBeats = 1.0f;
    score->notes.push_back(timedNote(releasedNote, 0.0f, 1.0f, 64.0f));
    score->notes.push_back(timedNote(restruckNote, 1.5f, 1.0f, 64.0f));
    const canonicalblockc::OwnerIdentity identity {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    // Sounding first note; its tail with the second note upcoming; the second
    // note sounding while the first is still in its tail.
    const std::array<float, 3> beats {0.5f, 1.25f, 1.75f};

    FrameProducer warm;
    NoteSnapshot sounding;
    for (std::size_t frame = 0; frame < beats.size(); ++frame)
    {
        const auto frameIndex = static_cast<std::int64_t>(frame);
        const auto warmFrame = warm.evaluate(frameKeyAt(identity, frameIndex, beats[frame]), score,
                                             beats[frame]);
        require(canonicalblockc::valid(warmFrame), "re-strike warm frame was not valid");
        const auto warmSnapshot = snapshotOf(warmFrame);
        require(warmSnapshot == snapshotOf(coldFrame(score, identity, frameIndex, beats[frame])),
                "re-struck note's identity-keyed values differ from a fresh evaluation");
        require(rowOf(*warmFrame, releasedNote) >= 0, "released note left the resident rows during its tail");
        if (frame == 0)
        {
            sounding = warmSnapshot;
            continue;
        }
        require(warmSnapshot.texels.count(releasedNote) == 1
                    && warmSnapshot.texels.count(restruckNote) == 1,
                "re-struck note and its predecessor's tail did not both stay resident");
        require(rowOf(*warmFrame, releasedNote) != rowOf(*warmFrame, restruckNote),
                "re-struck note shared the released note's voice");
        require(warmSnapshot.appearance.at(releasedNote) == sounding.appearance.at(releasedNote),
                "release tail changed the released note's appearance after a re-strike");
        requireMidi(warmSnapshot, restruckNote, 64.0f, "re-struck identity does not carry its own pitch");
        requireMidi(warmSnapshot, releasedNote, 64.0f, "released identity lost its own pitch");
        if (frame == 1)
        {
            require(warmSnapshot.texels.at(releasedNote).values[3] < 0.0f,
                    "fixture did not place the released note in its tail");
            require(warmSnapshot.texels.at(restruckNote).values[2] < 0.0f,
                    "fixture did not keep the re-struck note upcoming");
        }
        else
        {
            require(warmSnapshot.texels.at(restruckNote).values[2] >= 0.0f,
                    "fixture did not start the re-struck note");
        }
    }
}

// A seek bumps the seek generation, which re-derives the rows from a fresh
// allocation. A note whose seek lands inside its release tail keeps its identity
// and the identity-keyed data it had before the seek.
void testReleaseTailSeekKeepsIdentity()
{
    constexpr int releasedNote = -81;
    auto score = makeScore(2.0f);
    score->notes.push_back(timedNote(releasedNote, 0.25f, 1.0f, 67.0f));
    canonicalblockc::OwnerIdentity identity {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    FrameProducer producer;
    const auto sounding = producer.evaluate(frameKeyAt(identity, 0, 1.0f), score, 1.0f);
    require(canonicalblockc::valid(sounding), "release-seek sounding frame was not valid");
    const auto soundingSnapshot = snapshotOf(sounding);

    const auto tail = producer.evaluate(frameKeyAt(identity, 1, 1.5f), score, 1.5f);
    require(canonicalblockc::valid(tail), "release-seek tail frame was not valid");
    const auto tailSnapshot = snapshotOf(tail);
    require(tailSnapshot.texels.at(releasedNote).values[3] < 0.0f,
            "fixture did not place the note in its release tail before the seek");

    // The seek lands inside the tail at the same beat as the pre-seek tail frame.
    ++identity.seekGeneration;
    const auto seeked = producer.evaluate(frameKeyAt(identity, 2, 1.5f), score, 1.5f);
    require(canonicalblockc::valid(seeked), "seek-into-tail frame was not valid");
    const auto seekedSnapshot = snapshotOf(seeked);
    const auto cold = coldFrame(score, identity, 2, 1.5f);
    require(seekedSnapshot == snapshotOf(cold),
            "seek inside the release tail changed the note's identity-keyed values");
    require(rowOf(*seeked, releasedNote) == rowOf(*cold, releasedNote) && rowOf(*seeked, releasedNote) >= 0,
            "seek did not re-derive the released note's row from a fresh allocation");
    require(seekedSnapshot.texels == tailSnapshot.texels && seekedSnapshot.appearance == tailSnapshot.appearance,
            "seek inside the release tail changed the released note's values at the same beat");
    require(seekedSnapshot.appearance.at(releasedNote) == soundingSnapshot.appearance.at(releasedNote),
            "seek inside the release tail changed the released note's appearance");
}
}

int main()
{
    testHeldNoteRowChangeKeepsIdentity();
    testReleasedNoteReentryKeepsIdentity();
    testSwappedVoiceOrderKeepsIdentity();
    testNoteIdentityThroughReleaseTailUntilVoiceFreed();
    testRestruckSamePitchGetsNewIdentityDuringOldTail();
    testReleaseTailSeekKeepsIdentity();
    return 0;
}
