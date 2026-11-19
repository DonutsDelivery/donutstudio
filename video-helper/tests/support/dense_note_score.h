#pragma once

// A dense multi-track score whose sounding notes sit in high uNotes rows, the
// way a real project packs: at beat 9.5 the 30 notes that sounded from beat 0
// still hold rows 0-29 as history, and the 40 sounding notes hold the rows
// after them. Half the sounding notes are clip notes with negative projected
// ids, spread over three tracks. One resident note has velocity 0 and one is
// muted, so neither counts as sounding.

#include "canonical_block_c_frame.h"

#include <cmath>
#include <memory>

namespace densenotescore
{
constexpr int kHistoryNotes = 30;
constexpr int kSoundingNotes = 40;
constexpr float kBeat = 9.5f;

inline bool soundingId (int id) noexcept
{
    return (id <= -1000 && id > -1000 - kSoundingNotes) || (id >= 500 && id < 500 + kSoundingNotes);
}

inline int soundingNoteId (int index) noexcept
{
    return index % 2 == 0 ? -(1000 + index) : 500 + index;
}

// silencedId: a sounding note given velocity 0, which keeps its row but stops
// sounding. Zero silences nothing; -1 silences every sounding note.
inline std::shared_ptr<const arbitmod::Score> score (int silencedId = 0)
{
    auto result = std::make_shared<arbitmod::Score>();
    result->notationVersion = 1;
    result->scoreRevision = 1;
    result->rootFreq = 261.625565f;
    result->historyBeats = 4.0f;
    result->lookaheadBeats = 12.0f;
    const auto add = [&] (int id, int track, float start, float length, float midi, float velocity)
    {
        arbitmod::Note note;
        note.id = id;
        note.trackId = track;
        note.startBeat = start;
        note.lengthBeats = length;
        note.midiNote = midi;
        note.velocity = velocity;
        note.freqHz = 440.0f * std::pow (2.0f, (midi - 69.0f) / 12.0f);
        note.durationSeconds = length * 0.5f;
        result->notes.push_back (note);
        return &result->notes.back();
    };
    for (int index = 0; index < kHistoryNotes; ++index)
        add (100 + index, index % 3 + 1, 0.0f, 8.0f, 40.0f + static_cast<float> (index), 90.0f);
    for (int index = 0; index < kSoundingNotes; ++index)
    {
        const int id = soundingNoteId (index);
        const bool silenced = silencedId == -1 || silencedId == id;
        add (id, index % 3 + 1, 9.0f, 4.0f, 30.0f + static_cast<float> (index) * 1.7f,
             silenced ? 0.0f : 64.0f + static_cast<float> (index % 8) * 8.0f);
    }
    for (int index = 0; index + 1 < kSoundingNotes; index += 4)
    {
        arbitmod::Link link;
        link.id = 1 + index;
        link.slaveNoteId = soundingNoteId (index + 1);
        link.masterNoteId = soundingNoteId (index);
        link.slaveHarmonic = 3;
        link.masterHarmonic = 2;
        result->links.push_back (link);
    }
    add (900, 2, 9.0f, 4.0f, 77.0f, 0.0f);
    add (901, 3, 9.0f, 4.0f, 79.0f, 100.0f)->muted = true;
    return result;
}

// Packs at beat 0, then at kBeat with the same producer, so rows stay where
// the sticky allocator put them.
inline std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> frame (int silencedId = 0)
{
    const auto notes = score (silencedId);
    canonicalblockc::FrameKey key;
    key.projectGeneration = key.sourceGeneration = key.helperGeneration = 1;
    key.backendGeneration = key.deviceGeneration = key.scoreGeneration = 1;
    key.beatMapGeneration = key.fpsGeneration = key.loopGeneration = 1;
    key.seekGeneration = 1;
    key.fps = 2.0;
    canonicalblockc::FrameProducer producer;
    key.frame = 0;
    key.beat = 0.0;
    if (! canonicalblockc::valid (producer.evaluate (key, notes, 0.0f))) return {};
    key.frame = 19;
    key.beat = static_cast<double> (kBeat);
    return producer.evaluate (key, notes, kBeat);
}
}
