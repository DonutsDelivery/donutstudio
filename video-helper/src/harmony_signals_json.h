// harmony_signals_json.h: parse `score.harmony` into an arbitharmony timeline.
//
// The harmony object is optional. A payload without it (every payload written
// before the harmony feed) parses to `present == false`, and the score still
// loads. Malformed entries are skipped one by one instead of failing the
// score, because the notes stay valid without them.

#pragma once

#include "harmony_signals.h"
#include "mod_defs.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>

namespace arbitharmony
{

namespace detail
{
inline bool finiteNumber (const nlohmann::json& object, const char* key, double& value)
{
    const auto found = object.find (key);
    if (found == object.end() || ! found->is_number())
        return false;
    value = found->get<double>();
    return std::isfinite (value);
}

inline int clampedInt (const nlohmann::json& object, const char* key, int low, int high, int fallback)
{
    double value = 0.0;
    if (! finiteNumber (object, key, value))
        return fallback;
    const double clamped = std::clamp (value, static_cast<double> (low), static_cast<double> (high));
    return static_cast<int> (std::lround (clamped));
}

template <typename Segment>
inline void sortSegments (std::vector<Segment>& segments)
{
    std::stable_sort (segments.begin(), segments.end(),
        [] (const Segment& a, const Segment& b) { return a.startBeat < b.startBeat; });
}
} // namespace detail

// Onsets come from the parsed notes so uLastOnsetBeat works for old payloads.
inline void collectOnsets (const arbitmod::Score& score, HarmonyTimeline& timeline)
{
    timeline.onsetBeats.clear();
    for (const auto& note : score.notes)
    {
        if (note.muted || note.velocity <= 0.0f || ! std::isfinite (note.startBeat))
            continue;
        timeline.onsetBeats.push_back (static_cast<double> (note.startBeat));
    }
    // No cap: one double per parsed note, far smaller than the parsed notes
    // themselves, so uLastOnsetBeat stays right for any score length.
    std::sort (timeline.onsetBeats.begin(), timeline.onsetBeats.end());
    timeline.onsetBeats.erase (std::unique (timeline.onsetBeats.begin(), timeline.onsetBeats.end()),
                               timeline.onsetBeats.end());
}

// `scoreObject` is the score wire object (the one parseScoreJson read).
inline HarmonyTimeline parseHarmonyJson (const nlohmann::json& scoreObject, const arbitmod::Score& score)
{
    HarmonyTimeline timeline;
    collectOnsets (score, timeline);
    if (! scoreObject.is_object())
        return timeline;
    const auto found = scoreObject.find ("harmony");
    if (found == scoreObject.end() || ! found->is_object())
        return timeline;
    const auto& harmony = *found;

    timeline.present = true;
    timeline.phraseBars = detail::clampedInt (harmony, "phraseBars", 1, 64, kDefaultPhraseBars);
    double songEnd = 0.0;
    if (detail::finiteNumber (harmony, "songEndBeat", songEnd) && songEnd > 0.0)
        timeline.songEndBeat = songEnd;

    const auto chords = harmony.find ("chords");
    if (chords != harmony.end() && chords->is_array())
        for (const auto& entry : *chords)
        {
            if (timeline.chords.size() >= kMaxChordSegments)
                break;
            if (! entry.is_object())
                continue;
            ChordSegment segment;
            if (! detail::finiteNumber (entry, "startBeat", segment.startBeat)
                || ! detail::finiteNumber (entry, "endBeat", segment.endBeat)
                || segment.endBeat < segment.startBeat)
                continue;
            segment.rootPitchClass = detail::clampedInt (entry, "rootPc", 0, 11, 0);
            double position = static_cast<double> (segment.rootPitchClass) / 12.0;
            if (detail::finiteNumber (entry, "rootPos", position))
                position -= std::floor (position);
            segment.rootPosition = std::clamp (position, 0.0, 1.0);
            double cents = 0.0;
            if (detail::finiteNumber (entry, "rootCents", cents))
                segment.rootCents = cents - 1200.0 * std::floor (cents / 1200.0);
            segment.qualityClass = detail::clampedInt (entry, "qualityClass", 0, kQualityClassCount, QualityOther);
            double tension = 0.0;
            if (detail::finiteNumber (entry, "tension", tension))
                segment.tension = std::clamp (tension, 0.0, 1.0);
            timeline.chords.push_back (segment);
        }

    const auto keys = harmony.find ("keys");
    if (keys != harmony.end() && keys->is_array())
        for (const auto& entry : *keys)
        {
            if (timeline.keys.size() >= kMaxKeySegments)
                break;
            if (! entry.is_object())
                continue;
            KeySegment segment;
            if (! detail::finiteNumber (entry, "startBeat", segment.startBeat)
                || ! detail::finiteNumber (entry, "endBeat", segment.endBeat)
                || segment.endBeat < segment.startBeat)
                continue;
            segment.tonicPitchClass = detail::clampedInt (entry, "tonicPc", 0, 11, 0);
            segment.mode = detail::clampedInt (entry, "mode", 0, kModeCount - 1, 0);
            timeline.keys.push_back (segment);
        }

    detail::sortSegments (timeline.chords);
    detail::sortSegments (timeline.keys);
    return timeline;
}

} // namespace arbitharmony
