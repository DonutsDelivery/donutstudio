// block_c_packer.h — Block C voice allocator + texture packer (pre-work A2)
//
// Turns the note/link timeline into the two RGBA32F textures the shader contract
// (§4.3) reads: uNotes (one row of 4 texels per resident note) and uLinks
// (one texel per link, kLinkTextureWidth texels per texture row). Both are
// sized from the score; the row counts reach shaders as uNoteCount/uLinkCount.
//
// The differentiator vs. an FFT visualiser is that row assignment is a STATEFUL
// VOICE ALLOCATOR, not a per-frame sort: a note gets a row when it enters the
// past/future timeline window and keeps that row while it remains selected;
// rows free on exit and are reused. masterRow and the uLinks
// row indices therefore stay valid for a note's entire residency, so age- and
// row-keyed shader effects don't jump when an unrelated note enters or leaves.
//
// The shader-workbench's packScoreTextures() re-rows every frame (contiguous,
// stateless) — fine for a single still, wrong for motion. This packer is the
// real contract: it is what the helper must implement, and the row-stability
// unit tests pin the behaviour.
//
// Dependency-free C++17; reuses the arbitmod note/link model (A1) so plugin and
// helper share ONE definition. Destined for video-helper/src alongside
// mod_defs.h. No GPL code.
//
// uNoteCount contract note: with a sticky allocator, freeing a middle row leaves
// a hole, so uNoteCount is the *upper bound* (highest occupied row + 1), NOT the
// count of active notes. Shaders loop `i < uNoteCount` and read row i; holes are
// written zeroed (velocity 0) and gate to nothing. (This is the one place the
// stable allocator diverges from the workbench's contiguous packer — pinned by
// the noteCountCoversHoles test.)

#pragma once

#include "mod_defs.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace arbitblockc
{

using arbitmod::Link;
using arbitmod::Note;
using arbitmod::Score;

constexpr int kTexelsPerNote = 4;
// Layout width of the links texture, not a limit: link i sits at texel
// (i % kLinkTextureWidth, i / kLinkTextureWidth). Shaders read the width back
// with textureSize, so any layout width works.
constexpr int kLinkTextureWidth = 1024;

// Texture heights for a row count. A texture never has zero rows, so an empty
// score still binds a valid 1-row texture.
inline int noteTextureRows (int noteRows) noexcept { return std::max (1, noteRows); }
inline int linkTextureRows (int linkRows) noexcept
{
    return std::max (1, (std::max (0, linkRows) + kLinkTextureWidth - 1) / kLinkTextureWidth);
}
inline std::size_t noteTextureValues (int noteRows) noexcept
{
    return static_cast<std::size_t> (noteTextureRows (noteRows)) * kTexelsPerNote * 4u;
}
inline std::size_t linkTextureValues (int linkRows) noexcept
{
    return static_cast<std::size_t> (linkTextureRows (linkRows)) * kLinkTextureWidth * 4u;
}
// Upper bound on both textures' float count for a source of the given size,
// for work estimates that run before packing. Saturates instead of wrapping.
inline std::uint64_t textureValueBound (std::uint64_t notes, std::uint64_t links) noexcept
{
    constexpr auto cap = std::numeric_limits<std::uint64_t>::max() / 64u;
    notes = std::min (std::max<std::uint64_t> (1, notes), cap);
    links = std::min (links, cap);
    const auto linkRows = std::max<std::uint64_t> (1, (links + kLinkTextureWidth - 1) / kLinkTextureWidth);
    return notes * kTexelsPerNote * 4u + linkRows * kLinkTextureWidth * 4u;
}

struct PackResult
{
    // noteTextureRows(noteCount) * 4 texels * 4 channels, row-major (row, texel, channel).
    std::vector<float> notesTex;
    // linkTextureRows(linkCount) * kLinkTextureWidth texels * 4 channels.
    std::vector<float> linksTex;
    // Stable source identities for occupied texture rows. Zero marks a hole.
    // noteIds has noteCount entries; linkIds has linkCount entries.
    std::vector<std::int64_t> noteIds;
    std::vector<std::int64_t> linkIds;
    std::uint64_t scoreRevision = 0;
    float beat = std::numeric_limits<float>::quiet_NaN();
    int noteCount = 0;   // highest occupied row + 1 (loop bound, see header note)
    int linkCount = 0;   // edges with both endpoints resident
    // The textures grow with the score, so the packer leaves nothing out. Kept
    // so export progress and the HUD keep their fields; always zero.
    int droppedNotes = 0;
    int droppedLinks = 0;
};

// Empty unless a producer reports dropped rows; the content-sized packer never does.
inline std::string scoreDropWarning (int droppedNotes, int droppedLinks)
{
    if (droppedNotes <= 0 && droppedLinks <= 0)
        return {};
    std::string left;
    if (droppedNotes > 0)
        left = std::to_string (droppedNotes) + (droppedNotes == 1 ? " note" : " notes");
    if (droppedLinks > 0)
        left += (left.empty() ? "" : " and ") + std::to_string (droppedLinks)
              + (droppedLinks == 1 ? " link" : " links");
    return "Score visuals left out " + left + ".";
}

class BlockCPacker
{
public:
    BlockCPacker() { reset(); }

    void reset()
    {
        // Projected clip-loop identities are negative. Zero is the only invalid
        // projected identity and is therefore the allocator's hole sentinel.
        rowToNote_.clear();
        noteToRow_.clear();
        packCount_ = 0;
    }

    std::uint64_t packCount() const noexcept { return packCount_; }

    // Priority: sounding > upcoming > recently ended. Selection applies this
    // before row assignment so lower-priority history cannot starve new notes.
    static int residency (const Note& n, float beat,
                          float historyBeats, float lookaheadBeats)
    {
        return arbitmod::timelineResidency (n, beat, historyBeats, lookaheadBeats);
    }

    // One note's texels at a beat, in row layout:
    // texel0: midiNote, velocity/127, ageBeats, remainBeats
    // texel1: frame-resolved freqHz, centsFromRoot, trackId, isRoot
    // texel2: e2,e3,e5,e7
    // texel3: e11,e13, masterRow, pitch bend in semitones
    static std::array<float, kTexelsPerNote * 4> noteTexels (const Note& n, const Score& score,
                                                             float beat, int masterRow)
    {
        const float bentFrequency = arbitmod::pitchFrequencyAtBeat (n, beat);
        return { static_cast<float> (n.midiNote), arbitmod::clamp01 (n.velocity / 127.0f),
                 beat - n.startBeat, // negative for lookahead notes
                 n.endBeat() - beat,
                 bentFrequency, arbitmod::centsFromRoot (bentFrequency, score.rootFreq),
                 static_cast<float> (n.trackId), n.isRoot ? 1.0f : 0.0f,
                 n.primes[0], n.primes[1], n.primes[2], n.primes[3],
                 n.primes[4], n.primes[5], static_cast<float> (masterRow),
                 n.freqHz > 0.0f && bentFrequency > 0.0f
                     ? 12.0f * std::log2 (bentFrequency / n.freqHz) : 0.0f };
    }

    int rowOf (int noteId) const
    {
        auto it = noteToRow_.find (noteId);
        return it == noteToRow_.end() ? -1 : it->second;
    }

    PackResult pack (const Score& score, float beat)
    {
        return pack (score, beat, score.historyBeats, score.lookaheadBeats);
    }

    PackResult pack (const Score& score, float beat,
                     float historyBeats, float lookaheadBeats)
    {
        ++packCount_;
        // 1. Select the best bounded resident set. Within a class, sounding and
        // upcoming notes sort by onset; history sorts newest end first.
        struct Candidate { const Note* note; int priority; };
        std::vector<Candidate> candidates;
        candidates.reserve (score.notes.size());
        for (const auto& note : score.notes)
        {
            if (note.muted)
                continue;
            const int priority = residency (note, beat, historyBeats, lookaheadBeats);
            if (priority > 0)
                candidates.push_back ({ &note, priority });
        }
        std::sort (candidates.begin(), candidates.end(),
                   [] (const Candidate& a, const Candidate& b)
                   {
                       if (a.priority != b.priority) return a.priority > b.priority;
                       if (a.priority == 1 && a.note->endBeat() != b.note->endBeat())
                           return a.note->endBeat() > b.note->endBeat();
                       if (a.note->startBeat != b.note->startBeat)
                           return a.note->startBeat < b.note->startBeat;
                       return a.note->id < b.note->id;
                   });
        std::unordered_set<int> selectedIds;
        selectedIds.reserve (candidates.size());
        for (const auto& candidate : candidates)
            selectedIds.insert (candidate.note->id);

        // 2. Free rows outside the selected set. This is the capacity-aware
        // eviction step: active/upcoming entrants can displace older history.
        for (auto& id : rowToNote_)
        {
            if (id == 0) continue;
            if (selectedIds.find (id) == selectedIds.end())
            {
                noteToRow_.erase (id);
                id = 0;
            }
        }

        // 3. Assign the selected entrants to the lowest free rows. Notes that
        // survived selection keep their old rows, preserving motion continuity.
        std::size_t nextFree = 0;
        for (const auto& candidate : candidates)
        {
            if (noteToRow_.find (candidate.note->id) != noteToRow_.end())
                continue;
            // Lowest free row, or a new row past the end. The search resumes
            // where the last one stopped, since earlier rows are now taken.
            while (nextFree < rowToNote_.size() && rowToNote_[nextFree] != 0) ++nextFree;
            if (nextFree == rowToNote_.size()) rowToNote_.push_back (0);
            const int freeRow = static_cast<int> (nextFree);
            rowToNote_[nextFree] = candidate.note->id;
            noteToRow_[candidate.note->id] = freeRow;
        }

        // 4. Emit textures.
        PackResult out;
        out.scoreRevision = score.scoreRevision;
        out.beat = beat;
        // Trailing free rows go, so the row count is the highest occupied row + 1.
        while (! rowToNote_.empty() && rowToNote_.back() == 0) rowToNote_.pop_back();
        out.noteCount = static_cast<int> (rowToNote_.size());
        out.notesTex.assign (noteTextureValues (out.noteCount), 0.0f);
        out.noteIds.assign (static_cast<std::size_t> (out.noteCount), 0);
        for (int row = 0; row < out.noteCount; ++row)
        {
            const int id = rowToNote_[static_cast<std::size_t> (row)];
            if (id == 0) continue;
            const Note* n = score.noteById (id);
            if (n == nullptr) continue;
            out.noteIds[static_cast<std::size_t> (row)] = n->id;
            writeNote (out.notesTex, row, *n, score, beat);
        }
        // A selected note may no longer be in the score; its row reads as a hole.
        while (out.noteCount > 0 && out.noteIds[static_cast<std::size_t> (out.noteCount - 1)] == 0)
            --out.noteCount;
        out.noteIds.resize (static_cast<std::size_t> (out.noteCount));
        out.notesTex.resize (noteTextureValues (out.noteCount));

        // 5. Link edges: both endpoints must be resident.
        out.linkIds.reserve (score.links.size());
        for (const auto& l : score.links)
        {
            const int sRow = rowOf (l.slaveNoteId);
            const int mRow = rowOf (l.masterNoteId);
            if (sRow < 0 || mRow < 0 || sRow >= out.noteCount || mRow >= out.noteCount)
                continue;
            out.linkIds.push_back (l.id);
        }
        out.linkCount = static_cast<int> (out.linkIds.size());
        out.linksTex.assign (linkTextureValues (out.linkCount), 0.0f);
        int edge = 0;
        for (const auto& l : score.links)
        {
            const int sRow = rowOf (l.slaveNoteId);
            const int mRow = rowOf (l.masterNoteId);
            if (sRow < 0 || mRow < 0 || sRow >= out.noteCount || mRow >= out.noteCount)
                continue;
            const size_t off = static_cast<size_t> (edge++) * 4u;
            out.linksTex[off + 0] = static_cast<float> (sRow);
            out.linksTex[off + 1] = static_cast<float> (mRow);
            out.linksTex[off + 2] = static_cast<float> (l.slaveHarmonic);   // num
            out.linksTex[off + 3] = static_cast<float> (l.masterHarmonic);  // den
        }
        return out;
    }

private:
    void writeNote (std::vector<float>& tex, int row, const Note& n,
                    const Score& score, float beat) const
    {
        const int masterRow = n.linkMasterId != 0 ? rowOf (n.linkMasterId) : -1;
        const auto texels = noteTexels (n, score, beat, masterRow);
        std::copy (texels.begin(), texels.end(),
                   tex.begin() + static_cast<std::ptrdiff_t> (row) * kTexelsPerNote * 4);
    }

    std::vector<int> rowToNote_;
    std::unordered_map<int, int> noteToRow_;
    std::uint64_t packCount_ = 0;
};

// Convenience accessors for tests / consumers reading PackResult texels.
inline float noteTexel (const PackResult& r, int row, int texel, int ch)
{
    return r.notesTex[(static_cast<size_t> (row) * kTexelsPerNote + static_cast<size_t> (texel)) * 4u
                      + static_cast<size_t> (ch)];
}
inline float linkTexel (const PackResult& r, int edge, int ch)
{
    return r.linksTex[static_cast<size_t> (edge) * 4u + static_cast<size_t> (ch)];
}

} // namespace arbitblockc
