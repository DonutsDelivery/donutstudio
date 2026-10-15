#pragma once

#include "particle_body_replay.h"
#include "water_replay_source_budget.h"

namespace videohelper::geometry::coupledwaterbudget {
using namespace videorender;
using namespace watersourcebudget;
using videowire::geometry::OperationRecord;
using videowire::geometry::GeometryData;
using videowire::geometry::WaterSurfaceGrid;

// A separate recipe: the mass-zero source preflight remains unchanged.
// Counts are logical source/copy visits. Callback internals, STL algorithms,
// allocator/control blocks, libm and driver costs are outside this accounting.
inline bool preflight(const ParticleHistorySource& source,
    const videowire::geometry::OperationRecord& operation,
    const videowire::geometry::GeometryData& restMesh,
    const videowire::geometry::WaterSurfaceGrid& /* grid */,
    const std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame>& frame,
    double projectSeconds, double sceneSeconds, double requestedSeconds,
    const watersourcebudget::Limits& limits, watersourcebudget::Recipe& result, std::string& error)
{
    const auto refuse = [&](const std::string& reason) { error = reason; return false; };
    watersourcebudget::Recipe recipe;
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
    if (operation.code != videowire::geometry::OperationCode::fieldWaterHeight || !(operation.field.values[4] > 0))
        return refuse("Water source budget requires a fresh positive-mass coupled Water replay");
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
    std::uint64_t censusVisits=0;
    if (!checkedMultiply(recipe.sourceNotes,2,censusVisits) || !put(Work::censusNotes,censusVisits)) return false;
    std::uint64_t bends = 0, anchors = 0, pairs = 0;
    for (const auto& note : source.score->notes)
    {
        std::uint64_t notePairs = 0;
        if (!watersourcebudget::checkedAdd(bends, note.pitchBendPoints.size(), bends)
            || !watersourcebudget::checkedAdd(anchors, note.pitchAnchors.size(), anchors)
            || !checkedPairs(note.pitchAnchors.size(), notePairs)
            || !watersourcebudget::checkedAdd(pairs, notePairs, pairs)) return refuse("Water nested source census overflow");
        if (operation.field.values[2] > 0 && !note.muted && !(note.startBeat < 0)
            && !watersourcebudget::checkedAdd(recipe.eligibleScoreOnsets,1,recipe.eligibleScoreOnsets))
            return refuse("Water eligible source census overflow");
    }
    if (source.audioFrames && !source.audioFrames->empty())
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
        || !watersourcebudget::checkedAdd(identityCalls, linkCalls, identityCalls)
        || !watersourcebudget::checkedAdd(recipe.eligibleScoreOnsets, recipe.audioPrefix, candidates)
        || !watersourcebudget::checkedAdd(recipe.eligibleScoreOnsets, 1, calls)
        || !checkedMultiply(calls, recipe.sourceTempoPoints ? recipe.sourceTempoPoints - 1 : 0, segments)
        || !checkedMultiply(segments, 2, twiceSegments) || !watersourcebudget::checkedAdd(twiceSegments, calls, pointReads)
        || !checkedMultiply(v, 4, wave) || !checkedMultiply(v, recipe.steps, copied)
        || !checkedMultiply(recipe.audioPrefix, 5, audioScalars) || !checkedMultiply(recipe.audioPrefix, 64, bands))
        return refuse("Water source schedule overflow");
    recipe.impulseCapacity = std::min<std::uint64_t>(kParticleReplayMaxEvents, candidates);
    if (!watersourcebudget::checkedAdd(recipe.steps, recipe.impulseCapacity, checks)
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

enum class Work : std::size_t {
  historyCalls, parameterCallbacks, parameterCopies, bodyPreparationCopies, bodyPairVisits, boundaryRecords,
  eventScans, canonicalEvaluations,
  geometrySamples, geometryReplayVisits, sourceCopies, descriptorComparisons,
  frameComparisons, parameterComparisons, familyVisits, massInitializations,
  massTriangles, massPositionReads, massDeposits, previousHeightCopies,
  velocityApplications, impulseClears, contacts, contactDuplicateComparisons,
  contactCorners, finalColliderSamples, outputCopies, solidAppendRecords, solidAppendCopies, solidInspectorVisits,
  canonicalCandidateScans, canonicalSortElements, canonicalSelectionVisits, canonicalRowVisits,
  canonicalRowAssignmentSearches, canonicalNoteSearchVisits, canonicalLinkVisits, canonicalIdentityCalls,
  canonicalPitchBendVisits, canonicalPitchAnchorVisits, canonicalNotationCommaCopies, canonicalProducerCopyBytes,
  sourceSnapshotCopyBytes, bodyPreservationSearches, bodyArrayCopyElements, colliderCopyBytes,
  bodyRowVisits, bodySortElements, bodyLinkSearchVisits, bodyLinkPassVisits, bodyIntegrationVisits, boundarySortElements, count
};
inline constexpr std::array<const char*,static_cast<std::size_t>(Work::count)> names{{
  "history calls","parameter callbacks","parameter copies","body preparation copies","body pair visits","boundary records",
  "event scans","canonical evaluations",
  "geometry samples","geometry replay visits","source copies","descriptor comparisons",
  "frame comparisons","parameter comparisons","family visits","mass initializations",
  "mass triangles","mass position reads","mass deposits","previous height copies",
  "velocity applications","impulse clears","contacts","contact duplicate comparisons",
  "contact corners","final collider samples","output copies","Solid append records","Solid append copies","Solid inspector visits",
  "canonical candidate scans","canonical sort elements","canonical selection visits","canonical row visits",
  "canonical row assignment searches","canonical note search visits","canonical link visits","canonical identity calls",
  "canonical pitch bend visits","canonical pitch anchor visits","canonical notation comma copies","canonical producer copy bytes",
  "source snapshot copy bytes","body preservation searches","body array copy elements","collider copy bytes",
  "body row visits","body sort elements","body link search visits","body link pass visits","body integration visits","boundary sort elements"
}};
struct Limits final {
  watersourcebudget::Limits source;
  std::size_t maxRetainedBytes=256u*1024u*1024u, maxScratchBytes=256u*1024u*1024u, maxComparisonBytes=256u*1024u*1024u;
  std::array<std::uint64_t,static_cast<std::size_t>(Work::count)> work;
  Limits() { work.fill(1000000000000ULL); }
};
struct Recipe final {
  watersourcebudget::Recipe source;
  std::array<std::uint64_t,static_cast<std::size_t>(Work::count)> work{};
  std::size_t retainedBytes=0,scratchBytes=0,comparisonBytes=0,totalWork=0;
  std::uint64_t intervals=0,events=0,geometryBytes=0,geometryWork=0;
  std::uint64_t at(Work category) const { return work[static_cast<std::size_t>(category)]; }
};

inline bool descriptorBytes(const videowire::geometry::ValueDescriptor& value,std::size_t& bytes) {
  using namespace videowire::geometry;
  if (!videowire::geometry::checkedAdd(bytes,sizeof(ValueDescriptor),bytes)
      || !addArrayBytes(value.operations.capacity(),sizeof(OperationRecord),bytes)) return false;
  for (const auto& operation:value.operations)
    if (!videowire::geometry::checkedAdd(bytes,operation.field.attributeName.capacity(),bytes)
        || !videowire::geometry::checkedAdd(bytes,1,bytes)) return false;
  if (const auto* mesh=std::get_if<GeometryData>(&value.data))
    return addArrayBytes(mesh->positions.capacity(),sizeof(Vec3),bytes)
        && addArrayBytes(mesh->vertexIds.capacity(),sizeof(StableId),bytes)
        && addArrayBytes(mesh->indices.capacity(),sizeof(std::uint32_t),bytes);
  if (const auto* points=std::get_if<PointsData>(&value.data))
    return addArrayBytes(points->points.capacity(),sizeof(PointRecord),bytes);
  return false;
}

// This first source domain permits literal static emitter/attractor points.
// Their operation-free descriptors have no borrowed evaluator or nested source.
inline bool pointSourceBytes(const std::shared_ptr<const videowire::geometry::ValueDescriptor>& source,
    std::size_t& bytes,std::string& error) {
  using namespace videowire::geometry;
  if (!source) return true;
  if (source.use_count()==0 || source->carrier!=CarrierKind::points3D || !std::holds_alternative<PointsData>(source->data)
      || !source->operations.empty() || !source->attributes.empty()
      || !source->spectrumFields.empty() || !source->scoreFields.empty()
      || std::get<PointsData>(source->data).points.size()>63) {
    error="Coupled family requires bounded literal static emitter/attractor points"; return false;
  }
  return descriptorBytes(*source,bytes);
}

inline bool complete(const ParticleParams& p,const OperationRecord& operation,const GeometryData& rest,
    const WaterSurfaceGrid& grid,const std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame>& frame,
    double seconds,const Limits& limits,std::size_t familyWork,std::size_t geometryWork,
    std::size_t geometryScratch,std::size_t descriptorBytes,std::size_t appendWork,std::size_t appendBytes,Recipe& result,std::string& error) {
  using namespace videowire::geometry;
  Recipe r;
  if (!p.history || p.history.use_count()==0 || !p.history->score || p.history->score.use_count()==0
      || (p.history->analysisSources && p.history->analysisSources.use_count()==0)
      || !frame || frame.use_count()==0 || !p.geometryBinding || p.geometryBinding.use_count()==0
      || !p.geometryBinding->mesh || p.geometryBinding->mesh.use_count()==0
      || p.geometryBinding->bodySource || p.geometryBinding->body || p.geometryBinding->points
      || p.preparedBodies || p.preparedSolids || p.preparedBodyReplay || p.preparedImportedSolids || p.bodyReplayContext
      || p.motionMode!=2 || !p.historicalReplay || p.simulationSpace!=1 || p.bodyShape!=0
      || p.resetMode!=1 || p.resetTime!=0 || p.reset!=0 || p.count<1 || p.count>63
      || !std::isfinite(p.lifetime) || !std::isfinite(p.bodyRadius) || p.bodyRadius<=0 || p.bodyRadius>.1f
      || !std::isfinite(p.history->parameterFrameSeconds)
      || !std::isfinite(p.geometryBinding->centerX) || !std::isfinite(p.geometryBinding->centerY)
      || !std::isfinite(p.geometryBinding->centerZ)
      || seconds!=p.historyProjectSeconds || !std::isfinite(seconds) || seconds<0 || seconds>10
      || !std::isfinite(p.geometryBinding->scale) || p.geometryBinding->scale<=0 || p.geometryBinding->projection!=0
      || grid.columns!=grid.rows || (grid.columns!=8 && grid.columns!=9)
      || rest.positions.size()>256 || rest.indices.size()>384) {
    error="Coupled family requires one fresh released project-zero static Solid sphere request"; return false;
  }
  // The full source census above is admitted by the caller before it reads notes.
  if (!coupledwaterbudget::preflight(*p.history,operation,rest,grid,frame,seconds,seconds,seconds,limits.source,r.source,error)) return false;
  const auto window=particleReplayWindow(p,seconds);
  const auto ticks=std::llround(seconds*kParticleReplayTicksPerSecond);
  if (window.start!=0 || window.age!=ticks) {
    error="Coupled family body and Water clocks do not share the exact uncycled rational window"; return false;
  }
  const auto V=rest.positions.size(),I=rest.indices.size();
  std::uint64_t eventCandidates=0;
  if (!checkedMultiply(p.geometryCount==0 ? r.source.sourceNotes:0,2,eventCandidates)
      || !watersourcebudget::checkedAdd(eventCandidates,p.history->hasAudio() ? r.source.sourceAudioFrames:0,eventCandidates)) return false;
  r.events=std::min<std::uint64_t>(kParticleReplayMaxEvents,eventCandidates);
  if (!watersourcebudget::checkedAdd(r.source.steps,r.events,r.intervals) || !watersourcebudget::checkedAdd(r.intervals,1,r.intervals)) return false;
  const auto K=r.intervals;
  std::uint64_t H=0,G=0,canonicalChecks=0,sourceChecks=0,boundaries=0,parameterCopies=0,sourceCopies=0,previousHeightPasses=0;
  if (!watersourcebudget::checkedAdd(K,2,H) || !checkedMultiply(K,2,G)
      || !watersourcebudget::checkedAdd(G,5,G) || !watersourcebudget::checkedAdd(H,3,canonicalChecks)
      || !watersourcebudget::checkedAdd(H,5,sourceChecks) || !watersourcebudget::checkedAdd(K,1,boundaries)
      || !watersourcebudget::checkedAdd(H,12,parameterCopies) || !watersourcebudget::checkedAdd(G,3,sourceCopies)
      || !watersourcebudget::checkedAdd(r.source.steps,3,previousHeightPasses)) {
    error="Coupled family repeated call schedule overflow"; return false;
  }
  const auto put=[&](Work category,std::uint64_t a,std::uint64_t b=1) {
    std::uint64_t n=0; const auto i=static_cast<std::size_t>(category);
    if (!checkedMultiply(a,b,n) || n>limits.work[i]) {
      error=std::string("Coupled family budget exceeds ")+names[i]; return false;
    }
    r.work[i]=n; return true;
  };
  std::size_t sourceBytes=sizeof(ParticleGeometryBinding);
  if (!pointSourceBytes(p.geometryBinding->emitters,sourceBytes,error)
      || !pointSourceBytes(p.geometryBinding->attractors,sourceBytes,error)
      || !videowire::geometry::checkedAdd(sourceBytes,descriptorBytes,sourceBytes)) return false;
  // Retained actual sources are charged even when the source caller also owns them.
  std::size_t historyBytes=sizeof(ParticleHistorySource)+sizeof(arbitmod::Score);
  const auto& score=*p.history->score;
  if (!addArrayBytes(score.notes.capacity(),sizeof(decltype(score.notes)::value_type),historyBytes)
      || !addArrayBytes(score.links.capacity(),sizeof(decltype(score.links)::value_type),historyBytes)) return false;
  for (const auto& note:score.notes)
    if (!addArrayBytes(note.pitchBendPoints.capacity(),sizeof(decltype(note.pitchBendPoints)::value_type),historyBytes)
        || !addArrayBytes(note.pitchAnchors.capacity(),sizeof(decltype(note.pitchAnchors)::value_type),historyBytes)) return false;
  if (!addArrayBytes(p.history->timeline.tempoPoints().capacity(),2*sizeof(videotime::TempoPoint),historyBytes)
      || !addArrayBytes(p.history->timeline.meterPoints().capacity(),2*sizeof(videotime::MeterPoint),historyBytes)) return false;
  const auto sourceBends=r.source.at(watersourcebudget::Work::bends);
  const auto sourceAnchors=r.source.at(watersourcebudget::Work::anchors);
  // Every actual canonical evaluation validates the full score and publishes a
  // complete frame. Readiness adds two full admissible/event scans. Water prepare
  // is the independent source recipe; no dynamic-body receipt pays these visits.
  for (std::size_t i=1;i<r.source.work.size();++i) {
    const auto category=static_cast<watersourcebudget::Work>(i);
    std::uint64_t multiplier=1;
    if (category>=watersourcebudget::Work::scoreNotes && category<=watersourcebudget::Work::tempoPointReads)
      multiplier=sourceChecks;
    if (category>=watersourcebudget::Work::canonicalTextureValues && category<=watersourcebudget::Work::canonicalLinkRows)
      multiplier=canonicalChecks;
    if (category>=watersourcebudget::Work::audioValidationFrames && category<=watersourcebudget::Work::audioValidationBands)
      multiplier=3;
    std::uint64_t n=0;
    if (!checkedMultiply(r.source.work[i],multiplier,n) || n>limits.source.work[i]) {
      error=std::string("Coupled family source budget exceeds ")+workNames[i]; return false;
    }
    r.source.work[i]=n;
  }
  // Body readiness and schedule insertion query both endpoints of every
  // authored note, even when Water's score impulse gain is zero.
  std::uint64_t beatCalls=0,clockCalls=0,segments=0,reads=0,secondsCalls=0;
  if (!watersourcebudget::checkedAdd(H,4,secondsCalls)) return false;
  if (!checkedMultiply(r.source.sourceNotes,6,beatCalls)
      || !watersourcebudget::checkedAdd(beatCalls,r.source.eligibleScoreOnsets,beatCalls)
      || !watersourcebudget::checkedAdd(beatCalls,secondsCalls,clockCalls)
      || !checkedMultiply(clockCalls,r.source.sourceTempoPoints ? r.source.sourceTempoPoints-1:0,segments)
      || !checkedMultiply(segments,2,reads) || !watersourcebudget::checkedAdd(reads,clockCalls,reads)) return false;
  const std::array<std::pair<watersourcebudget::Work,std::uint64_t>,4> clockCounts{{
    {watersourcebudget::Work::beatToSecondsCalls,beatCalls},{watersourcebudget::Work::secondsToBeatCalls,secondsCalls},
    {watersourcebudget::Work::tempoSegmentVisits,segments},{watersourcebudget::Work::tempoPointReads,reads}
  }};
  for (const auto& count:clockCounts) {
    const auto i=static_cast<std::size_t>(count.first);
    r.source.work[i]=std::max(r.source.work[i],count.second);
    if (r.source.work[i]>limits.source.work[i]) { error="Coupled family complete tempo query allowance is short"; return false; }
  }
  std::uint64_t eventVisits=0;
  if (!checkedMultiply(r.source.sourceNotes,2,eventVisits)
      || !watersourcebudget::checkedAdd(eventVisits,p.history->hasAudio() ? r.source.sourceAudioFrames:0,eventVisits)) return false;
  // FrameProducer::evaluate may pack each history sample. Its two noteById
  // calls per occupied row each walk the complete source vector. Candidate
  // sorting is charged by input elements, not STL comparison complexity.
  // Four full frame byte-slot copies cover texture/identity/notation staging;
  // candidate pushes also scale with source notes and include record padding.
  const auto resident=std::min<std::uint64_t>(r.source.sourceNotes,arbitblockc::kMaxNotes);
  const auto B=static_cast<std::uint64_t>(p.count);
  std::uint64_t noteSearches=0,rowSearches=0,identityCalls=0,pitchBends=0,bodyPairs=0;
  std::uint64_t preservation=0,bodyCopies=0,colliderCopies=0,preservationCopies=0,linkSearches=0,linkPasses=0,callbacks=0;
  std::uint64_t producerBytes=0,candidateBytes=0,outputBytes=0,heightBytes=0,notationCopies=0;
  if (!checkedMultiply(resident,r.source.sourceNotes,noteSearches) || !checkedMultiply(noteSearches,2,noteSearches)
      || !checkedMultiply(resident,arbitblockc::kMaxNotes,rowSearches)
      || !checkedMultiply(resident,5,identityCalls)
      || !watersourcebudget::checkedAdd(identityCalls,2*arbitblockc::kMaxNotes,identityCalls)
      || !checkedMultiply(r.source.sourceLinks,2,linkSearches)
      || !watersourcebudget::checkedAdd(identityCalls,linkSearches,identityCalls)
      || !checkedMultiply(sourceBends,2,pitchBends) || !checkedPairs(B,bodyPairs)
      || !checkedMultiply(bodyPairs,4,bodyPairs) || !checkedMultiply(B,B,preservation)
      || !checkedMultiply(K,5,bodyCopies) || !checkedMultiply(H,3,preservationCopies)
      || !watersourcebudget::checkedAdd(bodyCopies,preservationCopies,bodyCopies)
      || !checkedMultiply(bodyCopies,videorender::kParticleSolidMaxBodies,bodyCopies)
      || !watersourcebudget::checkedAdd(K,3,colliderCopies)
      || !checkedMultiply(arbitblockc::kMaxLinks,B,linkSearches)
      || !checkedMultiply(arbitblockc::kMaxLinks,5,linkPasses)
      || !checkedMultiply(secondsCalls,visualsimulation::parameters.size(),callbacks)
      || !watersourcebudget::checkedAdd(callbacks,3,callbacks)
      || !checkedMultiply(resident,9,notationCopies)
      || !checkedMultiply(sizeof(canonicalblockc::CanonicalBlockCFrame),4,producerBytes)
      || !checkedMultiply(r.source.sourceNotes,sizeof(const arbitmod::Note*)+sizeof(int)+alignof(const arbitmod::Note*),candidateBytes)
      || !watersourcebudget::checkedAdd(producerBytes,candidateBytes,producerBytes)
      || !watersourcebudget::checkedAdd(sizeof(ParticleBodyState),sizeof(ParticleSolidState),outputBytes)
      || !watersourcebudget::checkedAdd(outputBytes,sizeof(WaterCoupledFrame),outputBytes)
      || !watersourcebudget::checkedAdd(outputBytes,descriptorBytes,outputBytes)
      || !checkedMultiply(V,sizeof(double),heightBytes)
      || !watersourcebudget::checkedAdd(outputBytes,heightBytes,outputBytes)) {
    error="Coupled family producer/body schedule overflow"; return false;
  }
  // updateInputs copies every body slot (kParticleSolidMaxBodies, 128, in both
  // modes because the replay arrays are sized for Solid) plus birth/lifetime
  // arrays, then searches the previous active bodies at every preserving history sample.
  // advanceInterval copies bodies once plus once in each of its four passes,
  // and copies the entire collider. Triangle query work has its own ledger.
  if (!put(Work::canonicalCandidateScans,r.source.sourceNotes,H)
      || !put(Work::canonicalSortElements,r.source.sourceNotes,H)
      || !put(Work::canonicalSelectionVisits,resident,H)
      || !put(Work::canonicalRowVisits,4*arbitblockc::kMaxNotes,H)
      || !put(Work::canonicalRowAssignmentSearches,rowSearches,H)
      || !put(Work::canonicalNoteSearchVisits,noteSearches,H)
      || !put(Work::canonicalLinkVisits,r.source.sourceLinks,H)
      || !put(Work::canonicalIdentityCalls,identityCalls,H)
      || !put(Work::canonicalPitchBendVisits,pitchBends,H)
      || !put(Work::canonicalPitchAnchorVisits,sourceAnchors,H)
      || !put(Work::canonicalNotationCommaCopies,notationCopies,H)
      || !put(Work::canonicalProducerCopyBytes,producerBytes,H)
      || !put(Work::sourceSnapshotCopyBytes,historyBytes,2)
      || !put(Work::bodyPreservationSearches,preservation,H-1)
      || !put(Work::bodyArrayCopyElements,bodyCopies)
      || !put(Work::colliderCopyBytes,sizeof(ParticleSolidState),colliderCopies)
      || !put(Work::bodyRowVisits,arbitblockc::kMaxNotes,H)
      || !put(Work::bodySortElements,arbitblockc::kMaxNotes+arbitblockc::kMaxLinks,H)
      || !put(Work::bodyLinkSearchVisits,linkSearches,H)
      || !put(Work::bodyLinkPassVisits,linkPasses,K)
      || !put(Work::bodyIntegrationVisits,B,K)
      || !put(Work::boundarySortElements,boundaries)
      || !put(Work::historyCalls,H) || !put(Work::parameterCallbacks,
          p.history->parameterAt && p.historyNodeId>0 ? callbacks:0)
      || !put(Work::parameterCopies,sizeof(ParticleParams),parameterCopies)
      || !put(Work::bodyPreparationCopies,ParticleBodyReplayState::callerPrepareWork())
      || !put(Work::bodyPairVisits,bodyPairs,K) || !put(Work::boundaryRecords,boundaries)
      || !put(Work::eventScans,eventVisits,3) || !put(Work::canonicalEvaluations,canonicalChecks)
      || !put(Work::geometrySamples,G) || !put(Work::geometryReplayVisits,geometryWork,G)
      || !put(Work::sourceCopies,sourceBytes,sourceCopies) || !put(Work::descriptorComparisons,descriptorBytes,3)
      || !put(Work::frameComparisons,sizeof(canonicalblockc::CanonicalBlockCFrame),3)
      || !put(Work::parameterComparisons,sizeof(ParticleParams),8)
      || !put(Work::familyVisits,familyWork) || !put(Work::massInitializations,V,3)
      || !put(Work::massTriangles,I/3) || !put(Work::massPositionReads,I) || !put(Work::massDeposits,I)
      || !put(Work::previousHeightCopies,V,previousHeightPasses)
      || !put(Work::velocityApplications,V,boundaries) || !put(Work::impulseClears,V,boundaries)
      || !put(Work::contacts,63,K) || !put(Work::contactDuplicateComparisons,63*63,K)
      || !put(Work::contactCorners,6*63,K) || !put(Work::finalColliderSamples,1)
      || !put(Work::outputCopies,outputBytes,3)
      || !put(Work::solidAppendRecords,appendWork) || !put(Work::solidAppendCopies,appendBytes,3)
      || !put(Work::solidInspectorVisits,appendWork,2)) return false;
  r.geometryBytes=sourceBytes; r.geometryWork=geometryWork;
  if (p.history->analysisSources) {
    if (p.history->analysisSources->size()>videowire::geometry::spectrum::kMaximumAnalysisSources
        || !addArrayBytes(p.history->analysisSources->capacity(),sizeof(videohelper::AnalysisSourceStream),historyBytes)) return false;
    for (const auto& stream:*p.history->analysisSources)
      if (!addArrayBytes(stream.frames.capacity(),sizeof(arbitblockb::FeatureFrame),historyBytes)) return false;
  }
  r.retainedBytes=historyBytes;
  if (!addArrayBytes(sourceBytes,2,r.retainedBytes)
      || !videowire::geometry::checkedAdd(r.retainedBytes,sizeof(canonicalblockc::CanonicalBlockCFrame),r.retainedBytes)
      || !addArrayBytes(V,2*sizeof(Vec3)+2*sizeof(StableId)+sizeof(double),r.retainedBytes)
      || !addArrayBytes(I,2*sizeof(std::uint32_t),r.retainedBytes)) return false;
  r.scratchBytes=ParticleBodyReplayState::callerScratchLogicalBytes();
  if (candidateBytes>std::numeric_limits<std::size_t>::max()
      || !videowire::geometry::checkedAdd(r.scratchBytes,static_cast<std::size_t>(candidateBytes),r.scratchBytes)
      || !videowire::geometry::checkedAdd(r.scratchBytes,geometryScratch,r.scratchBytes)
      || !addArrayBytes(V,4*sizeof(double),r.scratchBytes)
      || !addArrayBytes(sourceBytes,3,r.scratchBytes)) return false;
  for (const auto bytes:r.source.logicalBytes)
    if (bytes>std::numeric_limits<std::size_t>::max()
        || !videowire::geometry::checkedAdd(r.scratchBytes,static_cast<std::size_t>(bytes),r.scratchBytes)) return false;
  r.comparisonBytes=3*sizeof(ParticleParams)+2*sizeof(ValueDescriptor);
  if (!videowire::geometry::checkedAdd(r.comparisonBytes,sourceBytes,r.comparisonBytes)) return false;
  for (const auto n:r.source.work)
    if (n>std::numeric_limits<std::size_t>::max() || !videowire::geometry::checkedAdd(r.totalWork,static_cast<std::size_t>(n),r.totalWork)) return false;
  for (const auto n:r.work)
    if (n>std::numeric_limits<std::size_t>::max() || !videowire::geometry::checkedAdd(r.totalWork,static_cast<std::size_t>(n),r.totalWork)) return false;
  if (r.retainedBytes>limits.maxRetainedBytes || r.scratchBytes>limits.maxScratchBytes || r.comparisonBytes>limits.maxComparisonBytes) {
    error="Coupled family complete retained/scratch/comparison source envelope is short"; return false;
  }
  result=r; error.clear(); return true;
}
} // namespace videohelper::geometry::coupledwaterbudget
