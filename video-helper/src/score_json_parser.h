// Production parser for the Block C score wire object.
#pragma once

#include "canonical_block_c_frame.h"
#include "mod_defs.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>

namespace videohelper::scorejson
{
inline bool parseScoreJson (const nlohmann::json& sc, arbitmod::Score& score,
                            std::string& error)
{
    score = {};
    int schemaVersion = canonicalblockc::kLegacyScoreWireSchemaVersion;
    if (sc.contains("schemaVersion"))
    {
        const auto& wireVersion = sc["schemaVersion"];
        if (!wireVersion.is_number_integer()
            || (wireVersion.is_number_unsigned()
                && wireVersion.get<std::uint64_t>()
                    > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
            || (!wireVersion.is_number_unsigned()
                && (wireVersion.get<std::int64_t>() < std::numeric_limits<int>::min()
                    || wireVersion.get<std::int64_t>() > std::numeric_limits<int>::max())))
        {
            error = "score.schemaVersion must be an integer";
            return false;
        }
        schemaVersion = wireVersion.get<int>();
    }
    int ignoredIdentity = 0;
    if (!canonicalblockc::normalizeLinkMasterIdentity(
            schemaVersion, 0, ignoredIdentity))
    {
        error = "unsupported score.schemaVersion";
        return false;
    }
    score.notationVersion = sc.value("notationVersion", 1);
    score.scoreRevision = sc.value("scoreRevision", uint64_t { 0 });
    score.edoStepsPerOctave = std::max(1, sc.value("edoStepsPerOctave", 12));
    score.rootFreq = sc.value ("rootFreq", 261.625565f);
    score.historyBeats = std::max (0.0f, sc.value (
        "historyBeats", arbitmod::kDefaultScoreHistoryBeats));
    score.lookaheadBeats = std::max (0.0f, sc.value (
        "lookaheadBeats", arbitmod::kDefaultScoreLookaheadBeats));
    if (sc.contains ("notes"))
        for (const auto& n : sc["notes"])
        {
            arbitmod::Note nt;
            nt.id           = n.value ("id", 0);
            nt.durableKind  = n.value ("durableKind", 0);
            nt.durableNoteId = n.value ("durableNoteId", -1);
            nt.clipId       = n.value ("clipId", -1);
            nt.clipNoteId   = n.value ("clipNoteId", -1);
            nt.repeatIndex  = n.value ("repeatIndex", -1);
            nt.trackId      = n.value ("trackId", 0);
            nt.startBeat    = n.value ("startBeat", 0.0f);
            nt.lengthBeats  = n.value ("lengthBeats", 1.0f);
            nt.midiNote     = n.value ("midiNote", 60.0f);
            nt.velocity     = n.value ("velocity", 100.0f);
            nt.freqHz       = n.value ("freqHz", 261.625565f);
            nt.durationSeconds = n.value ("durationSeconds", 0.0f);
            if (n.contains ("pitchBendPoints") && n["pitchBendPoints"].is_array())
                for (const auto& point : n["pitchBendPoints"])
                {
                    arbitmod::PitchBendPoint bend;
                    bend.position = point.value ("position", 0.0f);
                    bend.semitones = point.value ("semitones", 0.0f);
                    bend.tension = point.value ("tension", 0.0f);
                    bend.sCurve = point.value ("sCurve", 0.0f);
                    bend.vibratoDepthCents = point.value ("vibratoDepthCents", 0.0f);
                    bend.vibratoRateHz = point.value ("vibratoRateHz", 0.0f);
                    bend.vibratoWaveform = point.value ("vibratoWaveform", 0);
                    bend.vibratoFadeIn = point.value ("vibratoFadeIn", 0.0f);
                    bend.vibratoFadeOut = point.value ("vibratoFadeOut", 0.0f);
                    nt.pitchBendPoints.push_back (bend);
                }
            if (n.contains ("pitchAnchors") && n["pitchAnchors"].is_array())
                for (const auto& anchor : n["pitchAnchors"])
                    nt.pitchAnchors.push_back ({
                        anchor.value ("id", 0),
                        anchor.value ("position", 0.0f),
                        anchor.value ("frequency", 0.0f) });
            nt.ratioNum     = n.value ("ratioNum", 1);
            nt.ratioDen     = n.value ("ratioDen", 1);
            const int wireMasterId = n.value ("linkMasterId", 0);
            if (!canonicalblockc::normalizeLinkMasterIdentity(
                    schemaVersion, wireMasterId, nt.linkMasterId))
            {
                error = "unsupported score.schemaVersion";
                return false;
            }
            nt.isRoot       = n.value ("isRoot", false);
            nt.centsOffset = n.value("centsOffset", 0.0f);
            nt.edoStep = n.value("edoStep", -1);
            nt.muted = n.value("muted", false);
            nt.notationVisible = n.value("notationVisible", ! nt.muted);
            nt.diatonicIndex = n.value("diatonicIndex", 28);
            nt.baseAccidental = n.value("baseAccidental", 0);
            nt.linked = n.value("linked", false);
            nt.hasUnmappedPrime = n.value("hasUnmappedPrime", false);
            nt.edoActive = n.value("edoActive", false);
            nt.edoInflection = n.value("edoInflection", 0);
            nt.edoDegree = n.value("edoDegree", 0);
            if (n.contains("commas") && n["commas"].is_array())
                for (const auto& comma : n["commas"])
                {
                    if (nt.commaCount >= static_cast<int>(nt.commas.size())) break;
                    nt.commas[static_cast<size_t>(nt.commaCount++)] = {
                        comma.value("prime", 0), comma.value("exponent", 0) };
                }
            if (n.contains ("primes") && n["primes"].is_array())
                for (size_t i = 0; i < n["primes"].size() && i < 6; ++i)
                    if (n["primes"][i].is_number())
                        nt.primes[i] = n["primes"][i].get<float>();
            score.notes.push_back (nt);
        }
    if (sc.contains ("links"))
        for (const auto& l : sc["links"])
        {
            arbitmod::Link lk;
            lk.id              = l.value ("id", 0);
            lk.slaveNoteId     = l.value ("slaveNoteId", 0);
            lk.masterNoteId    = l.value ("masterNoteId", 0);
            lk.slaveHarmonic   = l.value ("slaveHarmonic", 1);
            lk.masterHarmonic  = l.value ("masterHarmonic", 1);
            lk.octaveTranspose = l.value ("octaveTranspose", 0);
            score.links.push_back (lk);
        }
    return true;
}

} // namespace videohelper::scorejson
