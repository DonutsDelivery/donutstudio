#pragma once

#include "timeline_coupled_water_budget.h"

namespace videohelper::geometry {
class TimelineSceneFamilyEvaluator;

namespace coupledwaterdetail {
using namespace videowire::geometry;
inline bool sameOperation(const OperationRecord& a,const OperationRecord& b) {
  return a.code==b.code && a.stableId==b.stableId && a.inputStableId==b.inputStableId
      && a.outputStableId==b.outputStableId && a.secondaryInputStableId==b.secondaryInputStableId
      && a.parameters==b.parameters && a.field==b.field && equal(a.transform,b.transform)
      && a.retainedMesh==b.retainedMesh && a.sdfSurface==b.sdfSurface;
}
// Complete comparison for the admitted four-operation Water and literal points
// domains. No digest or stable ID can stand in for this source equality.
inline bool sameDescriptor(const ValueDescriptor& a,const ValueDescriptor& b) {
  if (a.schemaVersion!=b.schemaVersion || a.carrier!=b.carrier || a.stableId!=b.stableId
      || a.sourceStableId!=b.sourceStableId || a.sourceRevision!=b.sourceRevision || a.overflow!=b.overflow
      || a.fieldDomain!=b.fieldDomain || a.fieldValueType!=b.fieldValueType
      || a.fieldInterpolation!=b.fieldInterpolation || a.fieldDomainCardinality!=b.fieldDomainCardinality
      || a.dispatchCount!=b.dispatchCount || a.data.index()!=b.data.index()
      || !a.attributes.empty() || !b.attributes.empty() || !a.spectrumFields.empty() || !b.spectrumFields.empty()
      || !a.scoreFields.empty() || !b.scoreFields.empty() || a.operations.size()!=b.operations.size()) return false;
  for (std::size_t i=0;i<a.operations.size();++i) if (!sameOperation(a.operations[i],b.operations[i])) return false;
  if (std::holds_alternative<GeometryData>(a.data)) return equal(std::get<GeometryData>(a.data),std::get<GeometryData>(b.data));
  if (!std::holds_alternative<PointsData>(a.data)) return false;
  const auto& x=std::get<PointsData>(a.data).points; const auto& y=std::get<PointsData>(b.data).points;
  if (x.size()!=y.size()) return false;
  for (std::size_t i=0;i<x.size();++i)
    if (!equal(x[i].position,y[i].position) || x[i].stableId!=y[i].stableId || x[i].radius!=y[i].radius
        || x[i].rotation.x!=y[i].rotation.x || x[i].rotation.y!=y[i].rotation.y
        || x[i].rotation.z!=y[i].rotation.z || x[i].rotation.w!=y[i].rotation.w) return false;
  return true;
}
inline bool sameParameters(const videorender::ParticleParams& a,const videorender::ParticleParams& b) {
  auto x=a,y=b; const auto xf=videorender::particleAutomationFields(x),yf=videorender::particleAutomationFields(y);
  for (std::size_t i=0;i<xf.size();++i) if (*xf[i]!=*yf[i]) return false;
  return a.count==b.count && a.spawnTrack==b.spawnTrack && a.seed==b.seed && a.size==b.size
      && a.lifetime==b.lifetime && a.red==b.red && a.green==b.green && a.blue==b.blue && a.alpha==b.alpha
      && a.motionMode==b.motionMode && a.collisionMode==b.collisionMode && a.bodyShape==b.bodyShape
      && a.linkConstraint==b.linkConstraint && a.resetMode==b.resetMode && a.simulationSpace==b.simulationSpace
      && a.spawnZ==b.spawnZ && a.orientationDegrees==b.orientationDegrees && a.angularVelocity==b.angularVelocity
      && a.historicalReplay==b.historicalReplay && a.colliderThickness==b.colliderThickness
      && a.geometryCount==b.geometryCount && a.geometryAnchors==b.geometryAnchors
      && a.geometryIdentities==b.geometryIdentities && a.geometryDepths==b.geometryDepths
      && a.rms==b.rms && a.onset==b.onset && a.onsetAge==b.onsetAge
      && a.historyClipId==b.historyClipId && a.historyNodeId==b.historyNodeId && a.geometryRevision==b.geometryRevision
      && a.historyProjectSeconds==b.historyProjectSeconds;
}
inline videorender::ParticleParams scalarMetadata(videorender::ParticleParams p) {
  p.history.reset(); p.geometryBinding.reset(); p.preparedBodies.reset(); p.preparedSolids.reset();
  p.preparedBodyReplay.reset(); p.preparedImportedSolids.reset(); p.bodyReplayContext.reset(); return p;
}
inline bool currentAudio(const videorender::ParticleHistorySource& history,double seconds,std::string& error) {
  if (!history.audioFrames || history.audioFrames->empty()) return true;
  if (!std::isfinite(history.audioSampleRate) || history.audioSampleRate<=0) {
    error="Coupled family audio source has no finite positive sample rate"; return false;
  }
  const auto hops=std::floor(seconds*history.audioSampleRate/arbitblockb::kHop);
  if (!std::isfinite(hops) || hops<0 || hops>double(history.audioFrames->size())) {
    error="Coupled family audio source does not cover its exact project time"; return false;
  }
  auto count=static_cast<std::size_t>(hops);
  if (double(count)*arbitblockb::kHop/history.audioSampleRate<seconds) ++count;
  if (count>history.audioFrames->size()) { error="Coupled family audio coverage is short"; return false; }
  for (std::size_t i=0;i<count;++i) {
    const auto& frame=(*history.audioFrames)[i];
    if (i>static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()/arbitblockb::kHop-1)
        || frame.sampleIndex!=static_cast<std::int64_t>(i+1)*arbitblockb::kHop || frame.frameIndex!=static_cast<std::int64_t>(i)
        || !videorender::waterreplay::validAudioFrame(frame)) {
      error="Coupled family requested audio history is malformed or not contiguous"; return false;
    }
  }
  return true;
}
inline bool sameTimeline(const videotime::BeatTimeline& a,const videotime::BeatTimeline& b) {
  const auto& x=a.tempoPoints(); const auto& y=b.tempoPoints();
  const auto& m=a.meterPoints(); const auto& n=b.meterPoints();
  if (x.size()!=y.size() || m.size()!=n.size()) return false;
  for (std::size_t i=0;i<x.size();++i) if (x[i].beat!=y[i].beat || x[i].bpm!=y[i].bpm || x[i].step!=y[i].step) return false;
  for (std::size_t i=0;i<m.size();++i)
    if (m[i].beat!=n[i].beat || m[i].numerator!=n[i].numerator || m[i].denominator!=n[i].denominator) return false;
  return true;
}
} // namespace coupledwaterdetail

// Only the family coordinator can issue this pair, after every child's complete
// prospective schedule. Public request values never confer completed authority.
class TimelineCoupledWaterReplay final : public std::enable_shared_from_this<TimelineCoupledWaterReplay> {
public:
  TimelineCoupledWaterReplay(const TimelineCoupledWaterReplay&)=delete;
  TimelineCoupledWaterReplay& operator=(const TimelineCoupledWaterReplay&)=delete;
  std::shared_ptr<const videorender::WaterCoupledFrame> water() const {
    return {shared_from_this(),&water_};
  }
  std::shared_ptr<const videorender::ParticleBodyState> bodies() const {
    return {shared_from_this(),&bodies_};
  }
  std::shared_ptr<const videorender::ParticleSolidState> solids() const {
    return {shared_from_this(),&solids_};
  }
  std::int64_t terminalTick() const noexcept { return terminalTick_; }
  const coupledwaterbudget::Recipe& recipe() const noexcept { return recipe_; }

  // Inspection of escaped payloads does not consult borrowed history/audio.
  // Fresh materialization must independently authenticate its live request.
  bool matches(GeometryCorePlanRuntime& runtime,const PlanOwnerIdentity& owner,
      const std::shared_ptr<const AdmittedTimelineScenePlans>& family,std::size_t childIndex,
      const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& authored,
      const videorender::ParticleParams& p,double sourceSeconds,double bodySeconds,double projectSeconds,float aspect,
      const std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame>& frame,std::string& error) const {
    const auto comparison=callerLease_->ledger->reserve(recipe_.comparisonBytes,recipe_.totalWork);
    if (!comparison) { error="Coupled family independent comparison admission is short"; return false; }
    try {
      return matchesReserved(runtime,owner,family,childIndex,authored,p,sourceSeconds,bodySeconds,projectSeconds,aspect,frame,comparison,error);
    } catch (const std::bad_alloc&) {
      error="Coupled family independent comparison allocation failed"; return false;
    }
  }
private:
  bool matchesReserved(GeometryCorePlanRuntime& runtime,const PlanOwnerIdentity& owner,
      const std::shared_ptr<const AdmittedTimelineScenePlans>& family,std::size_t childIndex,
      const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& authored,
      const videorender::ParticleParams& p,double sourceSeconds,double bodySeconds,double projectSeconds,float aspect,
      const std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame>& frame,
      const std::shared_ptr<const videorender::ParticleBodyCallerLease>& comparison,std::string& error) const {
    using namespace coupledwaterdetail;
    if (!comparison || comparison->ledger!=callerLease_->ledger
        || comparison->bytes!=recipe_.comparisonBytes+sizeof(videorender::ParticleBodyCallerLease)) {
      error="Coupled family independent comparison has no exact prospective lease"; return false;
    }
    const auto live=history_.lock();
    if (!live || !p.history || live.get()!=p.history.get() || live.owner_before(p.history) || p.history.owner_before(live)
        || family!=family_ || family.owner_before(family_) || family_.owner_before(family) || !(owner==owner_) || !family || childIndex!=childIndex_
        || childIndex>=family->children().size() || &family->children()[childIndex]!=child_
        || !(child_->owner()==owner) || frame!=frame_ || !frame || frame.owner_before(frame_) || frame_.owner_before(frame) || !(frame->key==frameKey_)
        || !authored || !timelinefamily::sameScene(*authored,family->scene())
        || sourceSeconds!=seconds_ || bodySeconds!=seconds_ || projectSeconds!=seconds_ || aspect!=aspect_
        || !sameParameters(p,parameters_) || !p.geometryBinding || p.geometryBinding.get()!=bindingSource_.get()
        || p.geometryBinding.owner_before(bindingSource_) || bindingSource_.owner_before(p.geometryBinding)
        || p.preparedBodies || p.preparedSolids || p.preparedBodyReplay || p.preparedImportedSolids || p.bodyReplayContext
        || !(live->key==historyKey_) || live->score!=score_ || live->score.owner_before(score_) || score_.owner_before(live->score)
        || live->analysisSources!=analysis_ || live->analysisSources.owner_before(analysis_) || analysis_.owner_before(live->analysisSources)
        || live->audioFrames!=audio_ || live->audioSampleRate!=audioRate_ || live->parameterFrameSeconds!=parameterFrameSeconds_
        || bool(live->parameterAt)!=parameterCallback_ || bool(live->importedGeometry)!=importedCallback_
        || !sameTimeline(live->timeline,timeline_)) {
      error="Coupled family pair differs from its live family/child/owner/frame/history/time request"; return false;
    }
    const auto& b=*p.geometryBinding;
    if (b.mesh!=meshSource_ || b.mesh.owner_before(meshSource_) || meshSource_.owner_before(b.mesh)
        || b.emitters!=emitterSource_ || b.emitters.owner_before(emitterSource_) || emitterSource_.owner_before(b.emitters)
        || b.attractors!=attractorSource_ || b.attractors.owner_before(attractorSource_) || attractorSource_.owner_before(b.attractors)
        || b.body || b.bodySource || b.points || b.scale!=binding_.scale || b.centerX!=binding_.centerX
        || b.centerY!=binding_.centerY || b.centerZ!=binding_.centerZ || b.projection!=binding_.projection
        || !b.mesh || !sameDescriptor(*b.mesh,child_->plan().value().descriptor())
        || !sameDescriptor(*b.mesh,*binding_.mesh)
        || bool(b.emitters)!=bool(binding_.emitters) || bool(b.attractors)!=bool(binding_.attractors)
        || (b.emitters && !sameDescriptor(*b.emitters,*binding_.emitters))
        || (b.attractors && !sameDescriptor(*b.attractors,*binding_.attractors))) {
      error="Coupled family pair lost its exact frozen geometry binding"; return false;
    }
    // The caller keeps the live owner through this synchronous validation. No
    // retained callback or stale raw pointer is invoked from the issued pair.
    if (!canonicalblockc::valid(frame) || !currentAudio(*live,bodySeconds,error)
        || !videorender::particleHistoryReady(p,error,bodySeconds)) return false;
    auto expected=live->key; expected.fps=frame->key.fps;
    expected.frame=std::llround(projectSeconds*expected.fps);
    expected.beat=double(float(live->timeline.secondsToBeat(double(expected.frame)/expected.fps)));
    if (!(expected==frame->key) || terminalTick_!=std::llround(projectSeconds*videorender::kParticleReplayTicksPerSecond)) {
      error="Coupled family pair lost its complete canonical/rational clock"; return false;
    }
    auto sampled=p;
    if (!videorender::sampleParticleParameters(p,sampled,bodySeconds,error) || !sameParameters(sampled,sampledParameters_)) {
      if (error.empty()) error="Coupled family current parameter callback differs from issued metadata";
      return false;
    }
    std::vector<videowire::geometry::TimelineSceneBinding> bindings;
    bindings.reserve(family->children().size());
    for (const auto& child:family->children()) bindings.push_back(child.binding());
    const auto& limits=family->children().front().plan().receipt().effectiveLimits;
    const auto current=owner.use==PlanUse::preview
        ? runtime.admitTimelineScenePreview(owner,*authored,bindings,limits,error)
        : runtime.admitTimelineSceneExport(owner,*authored,bindings,limits,error);
    if (current!=family || current.owner_before(family) || family.owner_before(current)) { if (error.empty()) error="Coupled family independent consumer has no exact current runtime admission"; return false; }
    error.clear(); return true;
  }
private:
  TimelineCoupledWaterReplay(std::shared_ptr<const videorender::ParticleBodyCallerLease> lease,
      std::shared_ptr<const void> familyLease,const PlanOwnerIdentity& owner,
      std::shared_ptr<const AdmittedTimelineScenePlans> family,std::size_t index,
      const videorender::ParticleParams& p,float aspect,
      std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> frame,coupledwaterbudget::Recipe recipe)
      : callerLease_(std::move(lease)),familyLease_(std::move(familyLease)),family_(std::move(family)),
        frame_(std::move(frame)),bindingSource_(p.geometryBinding),meshSource_(p.geometryBinding->mesh),
        emitterSource_(p.geometryBinding->emitters),attractorSource_(p.geometryBinding->attractors),history_(p.history),score_(p.history->score),
        analysis_(p.history->analysisSources),timeline_(p.history->timeline),binding_(*p.geometryBinding),
        parameters_(coupledwaterdetail::scalarMetadata(p)),owner_(owner),childIndex_(index),child_(&family_->children()[index]),
        historyKey_(p.history->key),frameKey_(frame_->key),audio_(p.history->audioFrames),audioRate_(p.history->audioSampleRate),
        parameterFrameSeconds_(p.history->parameterFrameSeconds),parameterCallback_(bool(p.history->parameterAt)),
        importedCallback_(bool(p.history->importedGeometry)),seconds_(p.historyProjectSeconds),aspect_(aspect),
        terminalTick_(std::llround(seconds_*videorender::kParticleReplayTicksPerSecond)),recipe_(recipe) {
    // Private copies freeze supported descriptors without retaining history readers.
    binding_.mesh=std::make_shared<const videowire::geometry::ValueDescriptor>(*binding_.mesh);
    if (binding_.emitters) binding_.emitters=std::make_shared<const videowire::geometry::ValueDescriptor>(*binding_.emitters);
    if (binding_.attractors) binding_.attractors=std::make_shared<const videowire::geometry::ValueDescriptor>(*binding_.attractors);
  }
  static std::shared_ptr<const TimelineCoupledWaterReplay> issue(
      std::shared_ptr<const videorender::ParticleBodyCallerLease> retained,
      std::shared_ptr<const void> familyLease,const PlanOwnerIdentity& owner,
      const std::shared_ptr<const AdmittedTimelineScenePlans>& family,std::size_t index,
      const videorender::ParticleParams& p,float aspect,
      const std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame>& frame,
      const coupledwaterbudget::Recipe& recipe,std::string& error) {
    auto result=std::shared_ptr<TimelineCoupledWaterReplay>(new TimelineCoupledWaterReplay(
        std::move(retained),std::move(familyLease),owner,family,index,p,aspect,frame,recipe));
    if (!coupledwaterdetail::currentAudio(*p.history,p.historyProjectSeconds,error)) return {};
    auto sampled=p;
    if (!videorender::sampleParticleParameters(p,sampled,p.historyProjectSeconds,error)
        || sampled.resetTime!=0 || sampled.reset!=0 || sampled.bodyRadius<=0 || sampled.bodyRadius>.1f) {
      if (error.empty()) error="Coupled family current sphere/reset parameters are outside the exact append domain";
      return {};
    }
    result->sampledParameters_=coupledwaterdetail::scalarMetadata(sampled);
    // Exactly one solver invocation owns both sides of momentum transfer.
    if (!videorender::replayWaterCoupledBodies(p,p.historyProjectSeconds,aspect,frame.get(),
        result->bodies_,result->solids_,nullptr,error,nullptr,&result->water_)) return {};
    sampled=p;
    if (!videorender::sampleParticleParameters(p,sampled,p.historyProjectSeconds,error)
        || !coupledwaterdetail::sameParameters(sampled,result->sampledParameters_)) {
      if (error.empty()) error="Coupled family current parameters changed during actual paired replay";
      return {};
    }
    // Accept actual copied application capacities only within the prospective
    // retained envelope. Library allocation overhead is not claimed here.
    std::size_t copied=0;
    if (!coupledwaterbudget::descriptorBytes(*result->binding_.mesh,copied)
        || (result->binding_.emitters && !coupledwaterbudget::descriptorBytes(*result->binding_.emitters,copied))
        || (result->binding_.attractors && !coupledwaterbudget::descriptorBytes(*result->binding_.attractors,copied))
        || !videowire::geometry::addArrayBytes(result->water_.restMesh.positions.capacity(),sizeof(videowire::geometry::Vec3),copied)
        || !videowire::geometry::addArrayBytes(result->water_.restMesh.vertexIds.capacity(),sizeof(videowire::geometry::StableId),copied)
        || !videowire::geometry::addArrayBytes(result->water_.restMesh.indices.capacity(),sizeof(std::uint32_t),copied)
        || !videowire::geometry::addArrayBytes(result->water_.heights.capacity(),sizeof(double),copied)
        || !videowire::geometry::addArrayBytes(result->timeline_.tempoPoints().capacity(),sizeof(videotime::TempoPoint),copied)
        || !videowire::geometry::addArrayBytes(result->timeline_.meterPoints().capacity(),sizeof(videotime::MeterPoint),copied)
        || copied>recipe.retainedBytes) {
      error="Coupled family actual copied source/Water capacities exceed their prospective envelope"; return {};
    }
    return result;
  }
  // Payloads retire before either ledger lease. Alias access retains this owner.
  const std::shared_ptr<const videorender::ParticleBodyCallerLease> callerLease_;
  const std::shared_ptr<const void> familyLease_;
  const std::shared_ptr<const AdmittedTimelineScenePlans> family_;
  const std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> frame_;
  const std::shared_ptr<const videorender::ParticleGeometryBinding> bindingSource_;
  const std::shared_ptr<const videowire::geometry::ValueDescriptor> meshSource_,emitterSource_,attractorSource_;
  const std::weak_ptr<const videorender::ParticleHistorySource> history_;
  const std::shared_ptr<const arbitmod::Score> score_;
  const std::shared_ptr<const videohelper::AnalysisSources> analysis_;
  const videotime::BeatTimeline timeline_;
  videorender::ParticleGeometryBinding binding_;
  const videorender::ParticleParams parameters_;
  videorender::ParticleParams sampledParameters_;
  const PlanOwnerIdentity owner_;
  const std::size_t childIndex_;
  const TimelineSceneChildAdmission* const child_;
  const canonicalblockc::FrameKey historyKey_,frameKey_;
  const std::vector<arbitblockb::FeatureFrame>* const audio_;
  const double audioRate_,parameterFrameSeconds_;
  const bool parameterCallback_,importedCallback_;
  const double seconds_;
  const float aspect_;
  const std::int64_t terminalTick_;
  const coupledwaterbudget::Recipe recipe_;
  videorender::WaterCoupledFrame water_;
  videorender::ParticleBodyState bodies_{};
  videorender::ParticleSolidState solids_;
  friend class TimelineSceneFamilyEvaluator;
};
} // namespace videohelper::geometry
