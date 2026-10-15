#pragma once

#include "water_surface_replay.h"

#include <array>
#include <cstddef>

namespace videorender::watersourcebudget
{
// Counts are source visits or logical payload bytes, never CPU instructions,
// allocator bytes or an evaluated replay receipt. A recipe grants no frame reuse.
enum class Work : std::size_t
{
    censusNotes, scoreNotes, scoreMasterNotes, scoreLinks, primes, commas,
    bends, anchors, priorAnchorComparisons, scoreIdentityCalls,
    tempoPoints, meterPoints, secondsToBeatCalls, beatToSecondsCalls,
    tempoSegmentVisits, tempoPointReads, scoreOnsetScan,
    audioValidationFrames, audioValidationScalars, audioValidationBands, audioOnsetScan,
    canonicalTextureValues, canonicalNoteRows, canonicalLinkRows,
    gridColumnScan, gridVertices, gridIndices, gridIdentityCalls, settingsValues,
    impulseCandidates, sortElements, waveInitializationElements,
    velocityCopyElements, eventCursorChecks, eventDeposits, stencilVertices,
    stencilHeightReads, publicationTransfers, count
};
inline constexpr std::array<const char*, static_cast<std::size_t>(Work::count)> workNames {{
    "note census", "score notes", "score master notes", "score links", "primes", "commas",
    "pitch bends", "pitch anchors", "prior anchor comparisons", "score identity calls",
    "tempo source points", "meter source points", "secondsToBeat calls", "beatToSeconds calls",
    "tempo segment visits", "tempo point reads", "score onset scan",
    "audio validation frames", "audio validation scalars", "audio validation bands", "audio onset scan",
    "canonical texture values", "canonical note rows", "canonical link rows",
    "grid column scan", "grid vertices", "grid indices", "grid identity calls", "settings values",
    "impulse candidates", "sort elements", "wave initialization elements",
    "velocity copy elements", "event cursor checks", "event deposits", "stencil vertices",
    "stencil height reads", "publication transfers"
}};
enum class Storage : std::size_t { noteIdentities, linkIdentities, gridIdentities, impulses, waveArrays, count };
inline constexpr std::array<const char*, static_cast<std::size_t>(Storage::count)> storageNames {{
    "note identity payload", "link identity payload", "grid identity payload", "impulse payload", "four wave arrays"
}};

struct Limits final
{
    // Explicit independent refusal thresholds. These do not bound library cost.
    std::array<std::uint64_t, static_cast<std::size_t>(Work::count)> work;
    std::array<std::uint64_t, static_cast<std::size_t>(Storage::count)> logicalBytes;
    Limits() { work.fill(10000000); logicalBytes.fill(256ull * 1024 * 1024); }
};

struct Recipe final
{
    std::array<std::uint64_t, static_cast<std::size_t>(Work::count)> work {};
    std::array<std::uint64_t, static_cast<std::size_t>(Storage::count)> logicalBytes {};
    std::uint64_t sourceNotes = 0, sourceLinks = 0, sourceTempoPoints = 0, sourceMeterPoints = 0;
    std::uint64_t sourceAudioFrames = 0, audioPrefix = 0, eligibleScoreOnsets = 0;
    std::uint64_t sourceRestPositions = 0, sourceRestVertexIds = 0, sourceRestIndices = 0;
    std::uint64_t steps = 0, impulseCapacity = 0;
    double projectQuerySeconds = 0;
    std::uint64_t at(Work category) const { return work[static_cast<std::size_t>(category)]; }
    std::uint64_t bytes(Storage category) const { return logicalBytes[static_cast<std::size_t>(category)]; }
    // std::sort comparisons/moves, unordered_set buckets/collisions, std::set
    // tree operations, vector growth/capacity, allocator, libm and driver costs
    // remain unknown. Borrowed score/audio/mesh/frame storage is not owned here.
    // Publication moves one of the four arrays, with no fifth payload copy.
    // Prior/current/escaped consumer ownership requires its own later ledger.
};

inline bool checkedAdd(std::uint64_t a, std::uint64_t b, std::uint64_t& out)
{
    if (b > std::numeric_limits<std::uint64_t>::max() - a) return false;
    out = a + b; return true;
}
inline bool checkedMultiply(std::uint64_t a, std::uint64_t b, std::uint64_t& out)
{
    if (a && b > std::numeric_limits<std::uint64_t>::max() / a) return false;
    out = a * b; return true;
}
inline bool checkedPairs(std::uint64_t count, std::uint64_t& out)
{
    if (count < 2) { out = 0; return true; }
    return count % 2 == 0 ? checkedMultiply(count / 2, count - 1, out)
                          : checkedMultiply(count, (count - 1) / 2, out);
}

// Called anew for EVERY replay request. Only the note census runs here;
// it has its own allowance before the first note is read. No source validation,
// tempo evaluation, audio walk, source copy, impulse sort or wave work runs here.
// The later consumer must use the actual complete sources and exact anchors
// synchronously, without mutation or retaining borrowed references.
// The recipe alone is deliberately not an authentication/admissibility token.
// Before any replay, that consumer must run canonicalblockc::valid, the exact
// current key/rounded beat match, validateWaterSurfaceGrid/Settings and complete
// FrameProducer::admissible/audio predicates through the released evaluator.
// This covers one callback replay only. Mesh materialization, memo verification,
// family retention and body coupling have separate schedules and ownership.
inline bool preflight(const ParticleHistorySource& source,
    const videowire::geometry::OperationRecord& operation,
    const videowire::geometry::GeometryData& restMesh,
    const videowire::geometry::WaterSurfaceGrid&,
    const std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame>& frame,
    double projectSeconds, double sceneSeconds, double requestedSeconds,
    const Limits& limits, Recipe& result, std::string& error)
{
    const auto refuse = [&](const std::string& reason) { error = reason; return false; };
    Recipe recipe;
    const auto put = [&](Work category, std::uint64_t count) {
        const auto index = static_cast<std::size_t>(category);
        if (count > limits.work[index]) return refuse(std::string("Water source budget exceeds ") + workNames[index]);
        recipe.work[index] = count; return true;
    };
    const auto bytes = [&](Storage category, std::uint64_t count, std::uint64_t width) {
        const auto index = static_cast<std::size_t>(category);
        std::uint64_t size = 0;
        if (!checkedMultiply(count, width, size)) return refuse("Water source logical payload overflow");
        if (size > limits.logicalBytes[index]) return refuse(std::string("Water source budget exceeds ") + storageNames[index]);
        recipe.logicalBytes[index] = size; return true;
    };
    if (!std::isfinite(projectSeconds) || !std::isfinite(sceneSeconds) || !std::isfinite(requestedSeconds))
        return refuse("Water source budget requires finite time anchors");
    recipe.projectQuerySeconds = projectSeconds + (requestedSeconds - sceneSeconds);
    if (!std::isfinite(recipe.projectQuerySeconds) || recipe.projectQuerySeconds < 0
        || recipe.projectQuerySeconds > waterreplay::kMaximumSeconds)
        return refuse("Water source budget requires a project query from zero through 10 seconds");
    // The callback's canonical clock uses the global anchor, not the query.
    if (!frame || !source.score || !std::isfinite(frame->key.fps) || frame->key.fps <= 0)
        return refuse("Water source budget requires actual score and canonical clock sources");
    if (operation.code != videowire::geometry::OperationCode::fieldWaterHeight || operation.field.values[4] != 0)
        return refuse("Water source budget covers only a fresh uncoupled Water replay");
    const auto clock = projectSeconds * frame->key.fps;
    if (!std::isfinite(clock) || clock < 0 || clock >= double(std::numeric_limits<std::int64_t>::max()))
        return refuse("Water source canonical clock overflow");
    recipe.steps = static_cast<std::uint64_t>(std::llround(recipe.projectQuerySeconds * kParticleReplayTicksPerSecond)
        / kParticleReplayStepTicks);
    recipe.sourceNotes = source.score->notes.size(); recipe.sourceLinks = source.score->links.size();
    recipe.sourceTempoPoints = source.timeline.tempoPoints().size();
    recipe.sourceMeterPoints = source.timeline.meterPoints().size();
    recipe.sourceAudioFrames = source.audioFrames ? source.audioFrames->size() : 0;
    recipe.sourceRestPositions = restMesh.positions.size(); recipe.sourceRestVertexIds = restMesh.vertexIds.size();
    recipe.sourceRestIndices = restMesh.indices.size();
    if (!put(Work::censusNotes, recipe.sourceNotes)) return false;
    std::uint64_t bends = 0, anchors = 0, pairs = 0;
    for (const auto& note : source.score->notes)
    {
        std::uint64_t notePairs = 0;
        if (!checkedAdd(bends, note.pitchBendPoints.size(), bends)
            || !checkedAdd(anchors, note.pitchAnchors.size(), anchors)
            || !checkedPairs(note.pitchAnchors.size(), notePairs)
            || !checkedAdd(pairs, notePairs, pairs)) return refuse("Water nested source census overflow");
        if (operation.field.values[2] > 0 && !note.muted && !(note.startBeat < 0)) ++recipe.eligibleScoreOnsets;
    }
    if (operation.field.values[3] > 0)
    {
        if (!source.audioFrames || source.audioFrames->empty() || !std::isfinite(source.audioSampleRate)
            || source.audioSampleRate <= 0) return refuse("Water source budget requires master audio coverage");
        // Match prepare's floor then endpoint comparison. Guard every floating
        // conversion first, including the rounded size_t boundary on 64-bit hosts.
        const auto product = recipe.projectQuerySeconds * source.audioSampleRate;
        const auto hops = std::floor(product / arbitblockb::kHop);
        if (!std::isfinite(product) || !std::isfinite(hops) || hops < 0
            || hops >= double(std::numeric_limits<std::size_t>::max())
            || hops > double(recipe.sourceAudioFrames)) return refuse("Water audio prefix overflow or unavailable coverage");
        auto count = static_cast<std::size_t>(hops);
        if (double(count) * arbitblockb::kHop / source.audioSampleRate < recipe.projectQuerySeconds)
        {
            if (count == std::numeric_limits<std::size_t>::max()) return refuse("Water audio prefix overflow");
            ++count;
        }
        if (count > recipe.sourceAudioFrames
            || static_cast<std::uint64_t>(count) > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max() / arbitblockb::kHop))
            return refuse("Water audio prefix overflow or unavailable coverage");
        recipe.audioPrefix = count;
    }
    const std::uint64_t n = recipe.sourceNotes, l = recipe.sourceLinks, v = restMesh.positions.size();
    std::uint64_t primes = 0, commas = 0, identityCalls = 0, linkCalls = 0, candidates = 0;
    std::uint64_t segments = 0, calls = 0, pointReads = 0, twiceSegments = 0;
    std::uint64_t wave = 0, copied = 0, checks = 0, deposits = 0, heightReads = 0, audioScalars = 0, bands = 0;
    if (!checkedMultiply(n, 6, primes) || !checkedMultiply(n, 9, commas)
        || !checkedMultiply(n, 2, identityCalls) || !checkedMultiply(l, 3, linkCalls)
        || !checkedAdd(identityCalls, linkCalls, identityCalls)
        || !checkedAdd(recipe.eligibleScoreOnsets, recipe.audioPrefix, candidates)
        || !checkedAdd(recipe.eligibleScoreOnsets, 1, calls)
        || !checkedMultiply(calls, recipe.sourceTempoPoints ? recipe.sourceTempoPoints - 1 : 0, segments)
        || !checkedMultiply(segments, 2, twiceSegments) || !checkedAdd(twiceSegments, calls, pointReads)
        || !checkedMultiply(v, 4, wave) || !checkedMultiply(v, recipe.steps, copied)
        || !checkedMultiply(recipe.audioPrefix, 5, audioScalars) || !checkedMultiply(recipe.audioPrefix, 64, bands))
        return refuse("Water source schedule overflow");
    recipe.impulseCapacity = std::min<std::uint64_t>(kParticleReplayMaxEvents, candidates);
    if (!checkedAdd(recipe.steps, recipe.impulseCapacity, checks)
        || !checkedMultiply(recipe.impulseCapacity, 4, deposits) || !checkedMultiply(copied, 7, heightReads))
        return refuse("Water step schedule overflow");
    // Conservative full-call bounds include early refusals and future onsets.
    // sortElements bounds input size only; it promises no comparison count.
    const std::array<std::pair<Work, std::uint64_t>, 37> counts {{
        {Work::scoreNotes,n},{Work::scoreMasterNotes,n},{Work::scoreLinks,l},{Work::primes,primes},{Work::commas,commas},
        {Work::bends,bends},{Work::anchors,anchors},{Work::priorAnchorComparisons,pairs},{Work::scoreIdentityCalls,identityCalls},
        {Work::tempoPoints,recipe.sourceTempoPoints},{Work::meterPoints,recipe.sourceMeterPoints},
        {Work::secondsToBeatCalls,1},{Work::beatToSecondsCalls,recipe.eligibleScoreOnsets},
        {Work::tempoSegmentVisits,segments},{Work::tempoPointReads,pointReads},
        {Work::scoreOnsetScan,operation.field.values[2]>0 ? n : 0},
        {Work::audioValidationFrames,recipe.audioPrefix},{Work::audioValidationScalars,audioScalars},
        {Work::audioValidationBands,bands},{Work::audioOnsetScan,recipe.audioPrefix},
        {Work::canonicalTextureValues,canonicalblockc::kNoteTextureValues+canonicalblockc::kLinkTextureValues},
        {Work::canonicalNoteRows,arbitblockc::kMaxNotes},{Work::canonicalLinkRows,arbitblockc::kMaxLinks},
        {Work::gridColumnScan,v},{Work::gridVertices,v},{Work::gridIndices,restMesh.indices.size()},
        {Work::gridIdentityCalls,v},{Work::settingsValues,12},
        {Work::impulseCandidates,candidates},{Work::sortElements,recipe.impulseCapacity},
        {Work::waveInitializationElements,wave},{Work::velocityCopyElements,copied},
        {Work::eventCursorChecks,checks},{Work::eventDeposits,deposits},{Work::stencilVertices,copied},
        {Work::stencilHeightReads,heightReads},{Work::publicationTransfers,1}
    }};
    for (const auto& count : counts) if (!put(count.first, count.second)) return false;
    if (!bytes(Storage::noteIdentities,n,sizeof(int)) || !bytes(Storage::linkIdentities,l,sizeof(int))
        || !bytes(Storage::gridIdentities,v,sizeof(videowire::geometry::StableId))
        || !bytes(Storage::impulses,recipe.impulseCapacity,sizeof(waterreplay::Impulse))
        || !bytes(Storage::waveArrays,wave,sizeof(double))) return false;
    result = recipe; error.clear(); return true;
}

// Recipe-only preflight permits a consumer to admit every sibling before any
// replay allocation. It still runs its note census once and grants no identity.
// withBudget preserves the original single synchronous consumer contract.
template <typename Consumer>
bool withBudget(const ParticleHistorySource& source,
    const videowire::geometry::OperationRecord& operation,
    const videowire::geometry::GeometryData& restMesh,
    const videowire::geometry::WaterSurfaceGrid& grid,
    const std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame>& frame,
    double projectSeconds, double sceneSeconds, double requestedSeconds,
    const Limits& limits, Consumer&& consume, std::string& error)
{
    Recipe recipe;
    if (!preflight(source,operation,restMesh,grid,frame,projectSeconds,sceneSeconds,
        requestedSeconds,limits,recipe,error)) return false;
    return consume(source, operation, restMesh, grid, frame, projectSeconds, sceneSeconds,
        requestedSeconds, recipe, error);
}
}
