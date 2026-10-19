// harmony_signals.h: per-frame harmony and structure signals for visuals.
//
// The plugin publishes a chord timeline, a key timeline and the phrase length
// in the score payload (`score.harmony`, see PROTOCOL.md). This header holds
// the parsed timeline and evaluates it at a beat. The shader uniforms (uChord,
// uKey, uPhrase, uLastOnsetBeat) and the `score.harmony.signal` control node
// both read evaluateHarmony(), so export and the live viewport agree.
//
// Plain C++17 with no JSON, GL or JUCE dependency. harmony_signals_json.h
// parses the wire object.

#pragma once

#include "beat_timeline.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace arbitharmony
{

// Chord quality classes. 0 means no chord is known at this beat.
enum QualityClass : int
{
    QualityNone = 0,
    QualityMajor = 1,
    QualityMinor = 2,
    QualityDiminished = 3,
    QualityAugmented = 4,
    QualitySuspended = 5,
    QualityDominant7 = 6,
    QualityMajor7 = 7,
    QualityMinor7 = 8,
    QualityOther = 9
};
inline constexpr int kQualityClassCount = 9;

// Modes follow the plugin's Tonal::Mode order on the wire (0..8). The uniform
// adds one so that 0 means no key is known.
inline constexpr int kModeCount = 9;
inline constexpr int kDefaultPhraseBars = 4;
inline constexpr std::size_t kMaxChordSegments = 8192;
inline constexpr std::size_t kMaxKeySegments = 1024;

// Mode brightness: Lydian 1 down to Locrian 0, counted by raised degrees.
// Harmonic minor sits between Aeolian and Dorian, melodic minor between
// Dorian and Mixolydian.
inline float modeBrightness (int mode) noexcept
{
    static constexpr float kBrightness[kModeCount] = {
        5.0f / 6.0f,   // Ionian
        3.0f / 6.0f,   // Dorian
        1.0f / 6.0f,   // Phrygian
        1.0f,          // Lydian
        4.0f / 6.0f,   // Mixolydian
        2.0f / 6.0f,   // Aeolian
        0.0f,          // Locrian
        2.5f / 6.0f,   // Harmonic minor
        3.5f / 6.0f    // Melodic minor
    };
    return mode >= 0 && mode < kModeCount ? kBrightness[mode] : 0.0f;
}

struct ChordSegment
{
    double startBeat = 0.0;
    double endBeat = 0.0;
    int rootPitchClass = 0;     // nearest 12-TET pitch class, 0 = C
    double rootPosition = 0.0;  // continuous root pitch class / 12, in [0,1)
    double rootCents = 0.0;     // root above the project root, folded to [0,1200)
    int qualityClass = QualityNone;
    double tension = 0.0;       // 0 = consonant, 1 = harsh
};

struct KeySegment
{
    double startBeat = 0.0;
    double endBeat = 0.0;
    int tonicPitchClass = 0;
    int mode = 0;               // Tonal::Mode order, 0 = Ionian
};

struct HarmonyTimeline
{
    // False for payloads written before the harmony feed. Every harmony
    // signal then stays zero; uLastOnsetBeat still reads the notes.
    bool present = false;
    int phraseBars = kDefaultPhraseBars;
    double songEndBeat = 0.0;
    std::vector<ChordSegment> chords;   // sorted by startBeat
    std::vector<KeySegment> keys;       // sorted by startBeat
    std::vector<double> onsetBeats;     // sorted note onsets, muted notes excluded
};

struct HarmonySignals
{
    float chord[4] = { 0.0f, 0.0f, 0.0f, 0.0f };   // root, quality, tension, beats since change
    float key[4] = { 0.0f, 0.0f, 0.0f, 0.0f };     // tonic, mode + 1, beats since change, brightness
    float phrase[4] = { 0.0f, 0.0f, 0.0f, 0.0f };  // bar index, phrase index, phrase phase, song progress
    float lastOnsetBeat = 0.0f;
    float barPhase = 0.0f;
    int chordIndex = -1;
    int keyIndex = -1;
};

// Index of the last segment that starts at or before `beat`, or -1. A segment
// holds through the silence after it, so a rest never reads as a change.
template <typename Segment>
inline int segmentAtBeat (const std::vector<Segment>& segments, double beat) noexcept
{
    const auto next = std::upper_bound (segments.begin(), segments.end(), beat,
        [] (double value, const Segment& segment) { return value < segment.startBeat; });
    return next == segments.begin() ? -1 : static_cast<int> (next - segments.begin()) - 1;
}

// Whole bars before `beat`, counted across meter changes. A meter change that
// lands mid-bar starts a new bar. Beats before zero count as bar 0.
inline std::int64_t barIndexAtBeat (const videotime::BeatTimeline& timeline, double beat,
                                    double& barPhase) noexcept
{
    barPhase = 0.0;
    if (! std::isfinite (beat) || beat < 0.0)
        beat = 0.0;
    const auto& meters = timeline.meterPoints();
    const auto beatsPerBar = [] (const videotime::MeterPoint& point)
    {
        const double value = static_cast<double> (std::max (1, point.numerator)) * 4.0
                           / static_cast<double> (std::max (1, point.denominator));
        return value > 1.0e-6 ? value : 4.0;
    };
    double bars = 0.0;
    double segmentStart = 0.0;
    double length = 4.0;
    if (! meters.empty())
    {
        std::size_t current = 0;
        while (current + 1 < meters.size() && meters[current + 1].beat <= beat)
        {
            const double span = meters[current + 1].beat - meters[current].beat;
            bars += std::ceil (span / beatsPerBar (meters[current]) - 1.0e-9);
            ++current;
        }
        segmentStart = std::min (meters[current].beat, beat);
        length = beatsPerBar (meters[current]);
    }
    const double local = std::max (0.0, beat - segmentStart) / length;
    const double whole = std::floor (local);
    barPhase = local - whole;
    const double index = bars + whole;
    return index < 9.0e15 ? static_cast<std::int64_t> (index) : 0;
}

// Evaluate every harmony signal at an absolute timeline beat. Pure function of
// (timeline, beat timeline, beat): no history, so a seek or a mid-range export
// gives the same values as playing from the top.
inline HarmonySignals evaluateHarmony (const HarmonyTimeline& harmony,
                                       const videotime::BeatTimeline& timeline,
                                       double beat) noexcept
{
    HarmonySignals out;
    if (! std::isfinite (beat))
        return out;

    double barPhase = 0.0;
    const std::int64_t bar = barIndexAtBeat (timeline, beat, barPhase);
    out.barPhase = static_cast<float> (barPhase);

    const auto onset = std::upper_bound (harmony.onsetBeats.begin(), harmony.onsetBeats.end(), beat);
    if (onset != harmony.onsetBeats.begin())
        out.lastOnsetBeat = static_cast<float> (*(onset - 1));

    if (! harmony.present)
        return out;

    out.chordIndex = segmentAtBeat (harmony.chords, beat);
    if (out.chordIndex >= 0)
    {
        const auto& chord = harmony.chords[static_cast<std::size_t> (out.chordIndex)];
        out.chord[0] = static_cast<float> (chord.rootPosition);
        out.chord[1] = static_cast<float> (chord.qualityClass);
        out.chord[2] = static_cast<float> (chord.tension);
        out.chord[3] = static_cast<float> (std::max (0.0, beat - chord.startBeat));
    }

    out.keyIndex = segmentAtBeat (harmony.keys, beat);
    if (out.keyIndex >= 0)
    {
        const auto& key = harmony.keys[static_cast<std::size_t> (out.keyIndex)];
        out.key[0] = static_cast<float> (key.tonicPitchClass) / 12.0f;
        out.key[1] = static_cast<float> (key.mode + 1);
        out.key[2] = static_cast<float> (std::max (0.0, beat - key.startBeat));
        out.key[3] = modeBrightness (key.mode);
    }

    const int phraseBars = std::max (1, harmony.phraseBars);
    const std::int64_t phrase = bar / phraseBars;
    const auto barInPhrase = static_cast<double> (bar - phrase * phraseBars);
    out.phrase[0] = static_cast<float> (bar);
    out.phrase[1] = static_cast<float> (phrase);
    out.phrase[2] = static_cast<float> ((barInPhrase + barPhase) / static_cast<double> (phraseBars));
    out.phrase[3] = harmony.songEndBeat > 1.0e-9
        ? static_cast<float> (std::clamp (beat / harmony.songEndBeat, 0.0, 1.0)) : 0.0f;
    return out;
}

// Selector values of the `score.harmony.signal` control node (param 0).
enum ControlSignal : int
{
    SignalChordRoot = 0,
    SignalChordQuality = 1,
    SignalChordTension = 2,
    SignalChordChange = 3,
    SignalKeyTonic = 4,
    SignalKeyMode = 5,
    SignalKeyBrightness = 6,
    SignalBarPhase = 7,
    SignalPhrasePhase = 8,
    SignalSongProgress = 9
};
inline constexpr int kControlSignalCount = 10;

// Chord index of the previous evaluated frame when there is none (first
// frame after bind, reset or a missing context).
inline constexpr int kNoPreviousChord = -2;

// One 0..1 control value for the control node. Every selector except the
// chord-change trigger is a pure function of the beat. The trigger compares
// chord indices between consecutive evaluated frames, because comparing a
// float beat age against the frame step misfires on rounding. It is 1 on the
// first frame inside a new chord. Callers pass the previous timeline frame's
// chord index; with kNoPreviousChord (frame 0, or no frame number to look
// back from) it fires only on the chord's first beat. The control-plan
// executor recomputes the previous frame's chord after a seek or loop jump.
inline float harmonyControlValue (const HarmonySignals& signals, int selector,
                                  int previousChordIndex) noexcept
{
    const auto clamp01 = [] (float value) { return std::isfinite (value) ? std::clamp (value, 0.0f, 1.0f) : 0.0f; };
    switch (selector)
    {
        case SignalChordRoot:     return clamp01 (signals.chord[0]);
        case SignalChordQuality:  return clamp01 (signals.chord[1] / static_cast<float> (kQualityClassCount));
        case SignalChordTension:  return clamp01 (signals.chord[2]);
        case SignalChordChange:
            if (signals.chordIndex < 0) return 0.0f;
            if (previousChordIndex == kNoPreviousChord) return signals.chord[3] < 1.0e-6f ? 1.0f : 0.0f;
            return signals.chordIndex != previousChordIndex ? 1.0f : 0.0f;
        case SignalKeyTonic:      return clamp01 (signals.key[0]);
        case SignalKeyMode:       return clamp01 (signals.key[1] / static_cast<float> (kModeCount));
        case SignalKeyBrightness: return signals.keyIndex >= 0 ? clamp01 (signals.key[3]) : 0.0f;
        case SignalBarPhase:      return clamp01 (signals.barPhase);
        case SignalPhrasePhase:   return clamp01 (signals.phrase[2]);
        case SignalSongProgress:  return clamp01 (signals.phrase[3]);
        default:                  return 0.0f;
    }
}

} // namespace arbitharmony
