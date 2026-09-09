// Immutable, versioned Block C frame shared by every score consumer.
#pragma once

#include "block_c_packer.h"

#include <algorithm>
#include <array>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>


namespace canonicalblockc
{

inline constexpr std::uint32_t kCanonicalBlockCFrameVersion = 2;
inline constexpr int kLegacyScoreWireSchemaVersion = 1;
inline constexpr int kCurrentScoreWireSchemaVersion = 2;

inline bool normalizeLinkMasterIdentity(int schemaVersion, int wireIdentity,
                                        int& normalizedIdentity) noexcept
{
    if (schemaVersion == kLegacyScoreWireSchemaVersion)
    {
        normalizedIdentity = wireIdentity == -1 ? 0 : wireIdentity;
        return true;
    }
    if (schemaVersion == kCurrentScoreWireSchemaVersion)
    {
        normalizedIdentity = wireIdentity;
        return true;
    }
    normalizedIdentity = 0;
    return false;
}
inline constexpr std::size_t kNoteTextureValues
    = static_cast<std::size_t>(arbitblockc::kMaxNotes * arbitblockc::kTexelsPerNote * 4);
inline constexpr std::size_t kLinkTextureValues
    = static_cast<std::size_t>(arbitblockc::kMaxLinks * 4);

struct FrameKey final
{
    std::uint32_t carrierVersion = kCanonicalBlockCFrameVersion;
    std::uint64_t projectGeneration = 0;
    std::uint64_t sourceGeneration = 0;
    std::uint64_t helperGeneration = 0;
    std::uint64_t backendGeneration = 0;
    std::uint64_t deviceGeneration = 0;
    std::uint64_t scoreGeneration = 0;
    std::uint64_t beatMapGeneration = 0;
    std::uint64_t fpsGeneration = 0;
    std::uint64_t loopGeneration = 0;
    std::uint64_t seekGeneration = 0;
    std::int64_t frame = 0;
    double beat = 0.0;
    double fps = 0.0;

    bool operator== (const FrameKey& other) const noexcept
    {
        return carrierVersion == other.carrierVersion
            && projectGeneration == other.projectGeneration
            && sourceGeneration == other.sourceGeneration
            && helperGeneration == other.helperGeneration
            && backendGeneration == other.backendGeneration
            && deviceGeneration == other.deviceGeneration
            && scoreGeneration == other.scoreGeneration
            && beatMapGeneration == other.beatMapGeneration
            && fpsGeneration == other.fpsGeneration && loopGeneration == other.loopGeneration
            && seekGeneration == other.seekGeneration && frame == other.frame
            && beat == other.beat && fps == other.fps;
    }
};

struct FrozenNotationNote final
{
    struct NotationComma final { int prime = 0; int exponent = 0; };

    std::int64_t identity = 0;
    float startBeat = 0.0f;
    float lengthBeats = 0.0f;
    int diatonicIndex = 0;
    int baseAccidental = 0;
    std::array<NotationComma, 9> commas {};
    int commaCount = 0;
    int edoInflection = 0;
    int trackId = 0;
    bool notationVisible = false;
    bool muted = false;
    bool isRoot = false;
    bool edoActive = false;

    float endBeat() const noexcept { return startBeat + lengthBeats; }
};

class FrameProducer;

class CanonicalBlockCFrame final
{
public:
    const std::uint32_t version;
    const FrameKey key;
    const std::array<float, kNoteTextureValues> noteTextureValues;
    const std::array<float, kLinkTextureValues> linkTextureValues;
    const std::array<std::int64_t, arbitblockc::kMaxNotes> noteIdentities;
    const std::array<std::int64_t, arbitblockc::kMaxLinks> linkIdentities;
    const std::array<FrozenNotationNote, arbitblockc::kMaxNotes> notationRows;
    const int noteRowCount;
    const int linkRowCount;
    const float packedBeat;
    const float rootFrequencyHz;
    const float scoreHistoryBeats;
    const float scoreLookaheadBeats;
    const std::array<float, 6> primeBasis;

    const auto& noteTexture() const noexcept { return noteTextureValues; }
    const auto& linkTexture() const noexcept { return linkTextureValues; }
    int noteRows() const noexcept { return noteRowCount; }
    int linkRows() const noexcept { return linkRowCount; }
    float historyBeats() const noexcept { return scoreHistoryBeats; }
    float lookaheadBeats() const noexcept { return scoreLookaheadBeats; }
    const FrozenNotationNote* noteAtRow (int row) const noexcept
    {
        if (row < 0 || row >= noteRowCount) return nullptr;
        const auto& note = notationRows[static_cast<std::size_t>(row)];
        return note.identity == 0 ? nullptr : &note;
    }

private:
    friend class FrameProducer;
    CanonicalBlockCFrame (
        FrameKey keyIn, int noteCount, int linkCount, float beat,
        float rootFrequency, float history, float lookahead,
        std::array<float, kNoteTextureValues> notes,
        std::array<float, kLinkTextureValues> links,
        std::array<std::int64_t, arbitblockc::kMaxNotes> noteIds,
        std::array<std::int64_t, arbitblockc::kMaxLinks> linkIds,
        std::array<FrozenNotationNote, arbitblockc::kMaxNotes> notation)
        : version(kCanonicalBlockCFrameVersion), key(keyIn),
          noteTextureValues(std::move(notes)), linkTextureValues(std::move(links)),
          noteIdentities(std::move(noteIds)), linkIdentities(std::move(linkIds)),
          notationRows(std::move(notation)), noteRowCount(noteCount),
          linkRowCount(linkCount), packedBeat(beat), rootFrequencyHz(rootFrequency),
          scoreHistoryBeats(history), scoreLookaheadBeats(lookahead),
          primeBasis{{2, 3, 5, 7, 11, 13}}
    {}
};

inline bool valid(const std::shared_ptr<const CanonicalBlockCFrame>& frame) noexcept;

class FrameProducer final
{
public:
    void reset() noexcept
    {
        packer_.reset();
        lastFrame_ = std::numeric_limits<std::int64_t>::min();
        last_.reset();
        ++generation_;
    }

    std::uint64_t packCount() const noexcept { return evaluatedPackCount_; }
    std::uint64_t generation() const noexcept { return generation_; }

    void rejectRequest() noexcept { reject(); }

    static bool admissible(const FrameKey& key,
                           const std::shared_ptr<const arbitmod::Score>& score,
                           float beat) noexcept
    {
        if (score == nullptr || key.carrierVersion != kCanonicalBlockCFrameVersion
            || key.projectGeneration == 0 || key.sourceGeneration == 0
            || key.helperGeneration == 0 || key.backendGeneration == 0
            || key.deviceGeneration == 0 || key.scoreGeneration == 0
            || key.beatMapGeneration == 0 || key.fpsGeneration == 0
            || key.loopGeneration == 0 || key.seekGeneration == 0 || key.frame < 0
            || !std::isfinite(key.fps) || key.fps <= 0.0 || !std::isfinite(key.beat)
            || !std::isfinite(beat) || key.beat != static_cast<double>(beat)
            || score->notes.size() > arbitblockc::kMaxNotes
            || score->links.size() > arbitblockc::kMaxLinks
            || score->notationVersion <= 0 || score->scoreRevision == 0
            || score->edoStepsPerOctave <= 0
            || !std::isfinite(score->rootFreq) || score->rootFreq <= 0.0f
            || !std::isfinite(score->historyBeats) || score->historyBeats < 0.0f
            || !std::isfinite(score->lookaheadBeats) || score->lookaheadBeats < 0.0f)
            return false;
        for (std::size_t noteIndex = 0; noteIndex < score->notes.size(); ++noteIndex)
        {
            const auto& note = score->notes[noteIndex];
            if (note.id == 0 || note.durableKind < 0 || note.durableKind > 2
                || note.trackId < 0 || !std::isfinite(note.startBeat)
                || !std::isfinite(note.lengthBeats) || note.lengthBeats < 0.0f
                || !std::isfinite(note.midiNote) || !std::isfinite(note.velocity)
                || !std::isfinite(note.freqHz) || note.freqHz <= 0.0f
                || !std::isfinite(note.durationSeconds) || note.durationSeconds < 0.0f
                || note.ratioNum <= 0 || note.ratioDen <= 0
                || !std::isfinite(note.centsOffset) || note.commaCount < 0
                || note.commaCount > static_cast<int>(note.commas.size())) return false;
            for (std::size_t prior = 0; prior < noteIndex; ++prior)
                if (score->notes[prior].id == note.id) return false;
            if (note.linkMasterId != 0)
            {
                const bool haveMaster = std::any_of (
                    score->notes.begin(), score->notes.end(), [&note] (const auto& candidate)
                    { return candidate.id == note.linkMasterId; });
                if (!haveMaster) return false;
            }
            for (const auto prime : note.primes) if (!std::isfinite(prime)) return false;
            for (const auto& comma : note.commas)
                if (comma.prime < 0) return false;
            for (const auto& point : note.pitchBendPoints)
                if (!std::isfinite(point.position) || !std::isfinite(point.semitones)
                    || !std::isfinite(point.tension) || !std::isfinite(point.sCurve)
                    || !std::isfinite(point.vibratoDepthCents)
                    || !std::isfinite(point.vibratoRateHz)
                    || !std::isfinite(point.vibratoFadeIn)
                    || !std::isfinite(point.vibratoFadeOut)
                    || point.vibratoWaveform < 0) return false;
            for (std::size_t anchorIndex = 0; anchorIndex < note.pitchAnchors.size(); ++anchorIndex)
            {
                const auto& anchor = note.pitchAnchors[anchorIndex];
                if (anchor.id == 0 || !std::isfinite(anchor.position)
                    || !std::isfinite(anchor.frequency) || anchor.frequency <= 0.0f) return false;
                for (std::size_t prior = 0; prior < anchorIndex; ++prior)
                    if (note.pitchAnchors[prior].id == anchor.id) return false;
            }
        }
        for (std::size_t linkIndex = 0; linkIndex < score->links.size(); ++linkIndex)
        {
            const auto& link = score->links[linkIndex];
            if (link.id == 0 || link.slaveNoteId == 0 || link.masterNoteId == 0
                || link.slaveHarmonic <= 0 || link.masterHarmonic <= 0) return false;
            for (std::size_t prior = 0; prior < linkIndex; ++prior)
                if (score->links[prior].id == link.id) return false;
            bool haveSlave = false, haveMaster = false;
            for (const auto& note : score->notes)
            {
                haveSlave = haveSlave || note.id == link.slaveNoteId;
                haveMaster = haveMaster || note.id == link.masterNoteId;
            }
            if (!haveSlave || !haveMaster) return false;
        }
        return true;
    }

    std::shared_ptr<const CanonicalBlockCFrame> evaluate(
        const FrameKey& key, const std::shared_ptr<const arbitmod::Score>& score, float beat)
    {
        if (!admissible(key, score, beat)) return reject();
        if (last_ != nullptr && key == last_->key && key.beat == static_cast<double>(beat))
            return last_;

        if (last_ == nullptr || !sameLifecycle(key, last_->key) || key.frame < lastFrame_)
            packer_.reset();

        constexpr std::size_t expectedNoteValues = kNoteTextureValues;
        constexpr std::size_t expectedLinkValues = kLinkTextureValues;
        const auto packed = packer_.pack(*score, beat);
        ++evaluatedPackCount_;

        if (packed.noteCount < 0 || packed.noteCount > arbitblockc::kMaxNotes
            || packed.linkCount < 0 || packed.linkCount > arbitblockc::kMaxLinks
            || packed.notesTex.size() != expectedNoteValues
            || packed.linksTex.size() != expectedLinkValues)
            return reject();
        const auto usedNoteValues = static_cast<std::size_t>(packed.noteCount)
            * static_cast<std::size_t>(arbitblockc::kTexelsPerNote) * 4u;
        const auto usedLinkValues = static_cast<std::size_t>(packed.linkCount) * 4u;
        const auto trailingNoteValues = expectedNoteValues - usedNoteValues;
        const auto trailingLinkValues = expectedLinkValues - usedLinkValues;
        if (usedNoteValues + trailingNoteValues != expectedNoteValues
            || usedLinkValues + trailingLinkValues != expectedLinkValues)
            return reject();

        std::array<float, kNoteTextureValues> notes {};
        std::array<float, kLinkTextureValues> links {};
        std::copy(packed.notesTex.begin(), packed.notesTex.end(), notes.begin());
        std::copy(packed.linksTex.begin(), packed.linksTex.end(), links.begin());
        std::array<FrozenNotationNote, arbitblockc::kMaxNotes> notation {};
        for (int row = 0; row < packed.noteCount; ++row)
        {
            const auto identity = packed.noteIds[static_cast<std::size_t>(row)];
            if (identity == 0) continue;
            const auto* source = score->noteById(static_cast<int>(identity));
            if (source == nullptr) return reject();
            auto& frozen = notation[static_cast<std::size_t>(row)];
            frozen.identity = source->id;
            frozen.startBeat = source->startBeat;
            frozen.lengthBeats = source->lengthBeats;
            frozen.diatonicIndex = source->diatonicIndex;
            frozen.baseAccidental = source->baseAccidental;
            for (int index = 0; index < source->commaCount; ++index)
            {
                frozen.commas[static_cast<std::size_t>(index)].prime
                    = source->commas[static_cast<std::size_t>(index)].prime;
                frozen.commas[static_cast<std::size_t>(index)].exponent
                    = source->commas[static_cast<std::size_t>(index)].exponent;
            }
            frozen.commaCount = source->commaCount;
            frozen.edoInflection = source->edoInflection;
            frozen.trackId = source->trackId;
            frozen.notationVisible = source->notationVisible;
            frozen.muted = source->muted;
            frozen.isRoot = source->isRoot;
            frozen.edoActive = source->edoActive;
        }
        lastFrame_ = key.frame;
        last_ = std::shared_ptr<const CanonicalBlockCFrame>(new CanonicalBlockCFrame(
            key, packed.noteCount, packed.linkCount, packed.beat, score->rootFreq,
            score->historyBeats, score->lookaheadBeats, std::move(notes), std::move(links),
            packed.noteIds, packed.linkIds, std::move(notation)));
        return valid(last_) ? last_ : reject();
    }

private:
    static bool sameLifecycle(const FrameKey& left, const FrameKey& right) noexcept
    {
        return left.carrierVersion == right.carrierVersion
            && left.projectGeneration == right.projectGeneration
            && left.sourceGeneration == right.sourceGeneration
            && left.helperGeneration == right.helperGeneration
            && left.backendGeneration == right.backendGeneration
            && left.deviceGeneration == right.deviceGeneration
            && left.scoreGeneration == right.scoreGeneration
            && left.beatMapGeneration == right.beatMapGeneration
            && left.fpsGeneration == right.fpsGeneration && left.fps == right.fps
            && left.loopGeneration == right.loopGeneration
            && left.seekGeneration == right.seekGeneration;
    }


    std::shared_ptr<const CanonicalBlockCFrame> reject() noexcept
    {
        last_.reset();
        lastFrame_ = std::numeric_limits<std::int64_t>::min();
        packer_.reset();
        return {};
    }

    arbitblockc::BlockCPacker packer_;
    std::shared_ptr<const CanonicalBlockCFrame> last_;
    std::int64_t lastFrame_ = std::numeric_limits<std::int64_t>::min();
    std::uint64_t generation_ = 1;
    std::uint64_t evaluatedPackCount_ = 0;
};

template <typename BeatAtFrame>
std::shared_ptr<const CanonicalBlockCFrame> warmFrameProducerTo(
    FrameProducer& producer, std::int64_t& packedFrame,
    std::shared_ptr<const CanonicalBlockCFrame>& cachedFrame,
    FrameKey key, const std::shared_ptr<const arbitmod::Score>& score,
    std::int64_t targetFrame, BeatAtFrame&& beatAtFrame)
{
    const auto clearRejectedRequest = [&]()
    {
        producer.rejectRequest();
        packedFrame = -1;
        cachedFrame.reset();
        return std::shared_ptr<const CanonicalBlockCFrame> {};
    };
    if (targetFrame < 0 || !std::isfinite(key.fps) || key.fps <= 0.0)
        return clearRejectedRequest();

    const float targetBeat = static_cast<float>(beatAtFrame(targetFrame));
    key.frame = targetFrame;
    key.beat = static_cast<double>(targetBeat);
    if (!FrameProducer::admissible(key, score, targetBeat))
        return clearRejectedRequest();

    if (cachedFrame != nullptr)
    {
        const auto& previous = cachedFrame->key;
        const bool sameLifecycle = key.carrierVersion == previous.carrierVersion
            && key.projectGeneration == previous.projectGeneration
            && key.sourceGeneration == previous.sourceGeneration
            && key.helperGeneration == previous.helperGeneration
            && key.backendGeneration == previous.backendGeneration
            && key.deviceGeneration == previous.deviceGeneration
            && key.scoreGeneration == previous.scoreGeneration
            && key.beatMapGeneration == previous.beatMapGeneration
            && key.fpsGeneration == previous.fpsGeneration && key.fps == previous.fps
            && key.loopGeneration == previous.loopGeneration
            && key.seekGeneration == previous.seekGeneration;
        if (!sameLifecycle || targetFrame < packedFrame)
        {
            producer.reset();
            packedFrame = -1;
            cachedFrame.reset();
        }
        else if (targetFrame == packedFrame)
        {
            cachedFrame = producer.evaluate(key, score, targetBeat);
            if (cachedFrame == nullptr) return clearRejectedRequest();
            return cachedFrame;
        }
    }
    for (std::int64_t frame = packedFrame + 1; frame <= targetFrame; ++frame)
    {
        const float beat = frame == targetFrame
            ? targetBeat : static_cast<float>(beatAtFrame(frame));
        key.frame = frame;
        key.beat = static_cast<double>(beat);
        if (!FrameProducer::admissible(key, score, beat))
            return clearRejectedRequest();
        cachedFrame = producer.evaluate(key, score, beat);
        if (cachedFrame == nullptr) return clearRejectedRequest();
        packedFrame = frame;
    }
    return cachedFrame;
}

inline bool valid(const std::shared_ptr<const CanonicalBlockCFrame>& frame) noexcept
{
    if (frame == nullptr || frame->version != kCanonicalBlockCFrameVersion
        || frame->key.carrierVersion != frame->version || frame->key.projectGeneration == 0
        || frame->key.sourceGeneration == 0 || frame->key.helperGeneration == 0
        || frame->key.backendGeneration == 0 || frame->key.deviceGeneration == 0
        || frame->key.scoreGeneration == 0 || frame->key.beatMapGeneration == 0
        || frame->key.fpsGeneration == 0 || frame->key.loopGeneration == 0
        || frame->key.seekGeneration == 0 || frame->key.frame < 0
        || !std::isfinite(frame->key.beat) || !std::isfinite(frame->key.fps)
        || frame->key.fps <= 0.0 || frame->packedBeat != static_cast<float>(frame->key.beat)
        || frame->noteRowCount < 0 || frame->noteRowCount > arbitblockc::kMaxNotes
        || frame->linkRowCount < 0 || frame->linkRowCount > arbitblockc::kMaxLinks
        || !std::isfinite(frame->rootFrequencyHz) || frame->rootFrequencyHz <= 0.0f
        || !std::isfinite(frame->scoreHistoryBeats) || frame->scoreHistoryBeats < 0.0f
        || !std::isfinite(frame->scoreLookaheadBeats) || frame->scoreLookaheadBeats < 0.0f)
        return false;
    if (!std::all_of(frame->noteTextureValues.begin(), frame->noteTextureValues.end(),
                     [](float value) { return std::isfinite(value); })
        || !std::all_of(frame->linkTextureValues.begin(), frame->linkTextureValues.end(),
                        [](float value) { return std::isfinite(value); })) return false;
    for (int row = 0; row < arbitblockc::kMaxNotes; ++row)
    {
        const auto identity = frame->noteIdentities[static_cast<std::size_t>(row)];
        const bool hole = identity == 0;
        if ((row >= frame->noteRowCount && !hole)
            || frame->notationRows[static_cast<std::size_t>(row)].identity != identity) return false;

    }
    for (int row = frame->linkRowCount; row < arbitblockc::kMaxLinks; ++row)
    {
        if (frame->linkIdentities[static_cast<std::size_t>(row)] != 0) return false;

    }
    return true;
}
} // namespace canonicalblockc
