#pragma once

#include "imported_geometry_fixture.h"
#include "instance_appearance_fixture.h"
#include "particle_points_fixture.h"

inline videowire::geometry::FieldOperationSettings spectrumDomainSettings(
    videowire::geometry::Domain domain, videowire::geometry::ValueType type, unsigned mode) {
  videowire::geometry::FieldOperationSettings settings;
  settings.domain=domain; settings.valueType=type; settings.mode=static_cast<std::uint8_t>(mode);
  settings.values={0,1,arbitblockb::kBandLoHz,arbitblockb::kBandHiHz,0,0,0,0};
  return settings;
}

inline videowire::geometry::spectrum::Binding spectrumDomainBinding() {
  videowire::geometry::spectrum::Binding binding;
  binding.firstBand=2; binding.lastBand=5; binding.direction={0.35f,0,0};
  return binding;
}

inline videowire::geometry::spectrum::Bands spectrumDomainBands(bool changed=false) {
  videowire::geometry::spectrum::Bands bands{};
  bands[2]=changed ? 0.8f : 0.1f; bands[3]=changed ? 0.6f : 0.3f;
  bands[4]=changed ? 0.2f : 0.7f; bands[5]=changed ? 0.05f : 0.9f;
  return bands;
}

// All four fixtures end in ordinary retained mesh/instance rendering. Native
// submissions below are hooks for same-input helper parity, not editor evidence.
inline std::optional<videowire::geometry::ValueDescriptor> spectrumDomainFixture(
    unsigned domain, std::string& error, bool history = false) {
  using namespace videowire::geometry;
  const auto meshContract=constructedFieldMeshContract();
  const auto instancesContract=appearanceInstanceContract();
  std::optional<ValueDescriptor> source;
  if (domain==0) {
    const auto admitted=videohelper::gltf::adaptGlbMeshToGeometryCore(importedGeometryFixture(),1,9601,9600,1,error);
    if (!admitted) return std::nullopt;
    source=admitted->descriptor();
  }
  else if (domain==3) source=instanceAppearanceFixture(0,error);
  else if (domain==2) {
    PortContract curves; curves.carrier=CarrierKind::curves3D; curves.maxCurvePoints=8; curves.maxSplines=1;
    source=lowerCurveCircle(9601,9601,1,8,0.7f,curves,error);
  } else {
    auto grid=lowerGrid(9601,9601,1,2,2,1.2f,meshContract,error);
    PortContract points; points.carrier=CarrierKind::points3D; points.maxPoints=4;
    source=grid ? lowerPointsFromVertices(*grid,9602,points,error) : std::nullopt;
  }
  if (!source) return std::nullopt;
  auto binding=spectrumDomainBinding();
  const auto type=domain==3 ? ValueType::color : ValueType::vector;
  if (domain==3) binding.direction={0.2f,1,0.1f};
  auto settings=spectrumDomainSettings(elementDomain(source->carrier),type,domain==0 ? 5 : domain==3 ? 7 : 0);
  if (history) {
    binding.history.enabled=true; binding.history.depth=4; binding.history.spacingSeconds=0.25f;
    settings.values[4]=0; settings.values[5]=-1; settings.values[6]=1;
  }
  auto field=lowerAudioSpectrumField(*source,9610,binding,settings,
      constructedFieldContract(settings.domain,type),error);
  if (!field) return std::nullopt;
  if (domain==3) return lowerInstanceAppearance(*source,*field,nullptr,9611,"emission",instancesContract,error);
  auto carrierContract=domain==0 ? importedGeometryContract() : PortContract{};
  carrierContract.carrier=source->carrier; carrierContract.maxAttributes=kMaximumAttributes;
  if (domain==1) carrierContract.maxPoints=4;
  if (domain==2) { carrierContract.maxCurvePoints=8; carrierContract.maxSplines=1; }
  auto displaced=lowerVectorDisplacement(*source,*field,9611,1,
      withAttributeContract(carrierContract,*source),error);
  if (!displaced || domain==0) return displaced;
  if (domain==2) return lowerCurveToMesh(*displaced,9612,meshContract,error,0.15f);
  auto prototype=lowerMeshGenerator(9612,9612,1,OperationCode::plane,{0.3f,0.3f,0,0},meshContract,error);
  auto instances=prototype ? lowerInstanceOnPoints(*displaced,prototype->stableId,9613,instancesContract,error) : std::nullopt;
  if (!instances || !appendUniqueOperations(instances->operations,prototype->operations,error)) return std::nullopt;
  std::rotate(instances->operations.begin(),instances->operations.end()-1,instances->operations.end());
  instances->dispatchCount=instances->operations.size();
  return instances;
}

inline videowire::geometry::PortContract spectrumDomainTerminalContract(
    const videowire::geometry::ValueDescriptor& value) {
  using namespace videowire::geometry;
  return withAttributeContract(value.carrier==CarrierKind::instances3D
      ? appearanceInstanceContract() : importedGeometryContract(),value);
}

inline std::optional<videowire::geometry::ValueDescriptor> deferredParticleSpectrumFixture(
    std::string& error, unsigned mode = 0, bool color = false) {
  using namespace videowire::geometry;
  const auto original=particlePointsInstanceFixture(error);
  if (!original) return std::nullopt;
  ValueDescriptor points;
  points.carrier=CarrierKind::points3D; points.stableId=points.sourceStableId=801;
  points.sourceRevision=3; points.data=PointsData{};
  points.operations={{801,OperationCode::copy,0,801,{}}}; points.dispatchCount=1;
  const auto context=particlePointsFixtureContext();
  auto binding=spectrumDomainBinding(); binding.source=spectrum::Source::group; binding.sourceTrackId=17;
  const auto type=color ? ValueType::color : ValueType::vector;
  if (color) binding.direction={0.2f,1,0.1f};
  const auto settings=spectrumDomainSettings(Domain::point,type,mode);
  auto field=lowerAudioSpectrumField(points,9630,binding,settings,
      constructedFieldContract(Domain::point,type),error,context);
  PortContract pointContract; pointContract.carrier=CarrierKind::points3D; pointContract.maxPoints=2;
  // The stored color attribute needs one attribute slot. Admission rejects it at zero.
  if (color) pointContract.maxAttributes=1;
  auto modified=field ? (color
      ? lowerNamedAttribute(points,&*field,9631,OperationCode::storeNamedAttribute,"color",type,pointContract,error,context)
      : lowerVectorDisplacement(points,*field,9631,1,pointContract,error,context)) : std::nullopt;
  auto instances=modified ? lowerInstanceOnPoints(*modified,802,9632,particlePointsInstanceContract(),error,context) : std::nullopt;
  if (!instances) return std::nullopt;
  auto operations=original->operations;
  operations.pop_back();
  if (!appendUniqueOperations(operations,instances->operations,error)) return std::nullopt;
  instances->operations=std::move(operations); instances->dispatchCount=instances->operations.size();
  return instances;
}

template <typename ReadPixels>
bool verifyNativeDeferredParticleSpectrum(arbitgpu::NativeFixtureSceneBackend& backend,
    ReadPixels readPixels, std::string& error, bool color = false) {
  using namespace videowire::geometry;
  using namespace videohelper::geometry;
  const auto value=deferredParticleSpectrumFixture(error,color ? 7 : 0,color);
  const auto context=particlePointsFixtureContext();
  const auto contract=value ? withAttributeContract(particlePointsInstanceContract(),*value) : particlePointsInstanceContract();
  const auto admitted=value ? admitValue(*value,contract,{},context,error) : std::nullopt;
  if (!admitted) return false;
  NativeGeometryCoreCapabilitySource capabilities(backend);
  GeometryCorePlanRuntime runtime(capabilities);
  const auto bytes=lowerRuntimePlan(contract,*admitted);
  const auto preview=runtime.admitPreview({1,1,1,4,3,PlanUse::preview},bytes,{},context,error);
  const auto exported=runtime.admitExport({1,1,1,4,3,PlanUse::exportRender},bytes,{},context,error);
  if (!preview || !exported) return false;
  double seconds=.5;
  auto particles=particlePointsFixture();
  const PointsEvaluator evaluator=[&](auto id,const auto& source,const auto& owner,auto& result,auto& diagnostic) {
    if (id!=801 || !(source==context.deferredPoints.at(801)) || owner.planGeneration!=3
        || owner.projectGeneration!=1 || owner.helperGeneration!=1 || owner.deviceGeneration!=1 || owner.clipIdentity!=4) {
      diagnostic="Particle spectrum frame received a stale compiled owner"; return false;
    }
    return sampleParticlePointsFixture(particles,seconds,result,diagnostic);
  };
  const auto draw=[&](const RuntimeAdmission& plan,bool changed) {
    SpectrumEvaluation spectrum; spectrum.timeSeconds=seconds;
    // Master deliberately disagrees. Only the selected group may move points.
    spectrum.bands.fill(1);
    spectrum.sourceFeaturesAt=[changed](spectrum::Source source,std::int32_t id,double,auto& bands) {
      if (source!=spectrum::Source::group || id!=17) return false;
      bands=spectrumDomainBands(changed); return true;
    };
    return executeNativeGeometry(backend,plan,64,64,error,false,{},&spectrum,{},seconds,evaluator);
  };
  const auto first=draw(*preview,false), changed=draw(*preview,true), offline=draw(*exported,true);
  if (!first || !changed || !offline) return false;
  const auto firstPixels=readPixels(first->frame), changedPixels=readPixels(changed->frame);
  const std::vector<StableId> both{13,22};
  if (firstPixels.empty() || firstPixels==changedPixels || changedPixels!=readPixels(offline->frame)
      || first->drawnInstanceIds!=both || changed->drawnInstanceIds!=both) {
    error="Selected particle spectrum must change native pixels and preserve preview/export and stable IDs"; return false;
  }
  for (const auto at : {1.1,1.3,.5}) {
    seconds=at;
    const auto sought=draw(*preview,false), soughtExport=draw(*exported,false);
    if (!sought || !soughtExport) return false;
    const std::vector<StableId> expected=at==.5 ? both : at==1.1 ? std::vector<StableId>{22} : std::vector<StableId>{};
    const auto pixels=readPixels(sought->frame);
    if (pixels!=readPixels(soughtExport->frame) || sought->drawnInstanceIds!=expected
        || (at==.5 && pixels!=firstPixels)) {
      error="Particle spectrum native seek/expiry changed ownership or preview/export pixels"; return false;
    }
  }
  particles.resetTime=.375f;
  const auto reset=draw(*preview,false), resetExport=draw(*exported,false);
  particles=particlePointsFixture();
  const auto restored=draw(*preview,false);
  if (!reset || !resetExport || !restored || readPixels(reset->frame)!=readPixels(resetExport->frame)
      || readPixels(reset->frame)==firstPixels || readPixels(restored->frame)!=firstPixels) {
    error="Particle spectrum reset and restored replay must reproduce exact native frames"; return false;
  }
  SpectrumEvaluation missing; missing.timeSeconds=seconds;
  if (executeNativeGeometry(backend,*preview,64,64,error,false,{},&missing,{},seconds,evaluator)
      || executeNativeGeometry(backend,*preview,64,64,error)) {
    error="Deferred particle spectrum must refuse missing selected analysis or point provider"; return false;
  }
  auto stale=*preview; stale.owner.planGeneration=4;
  if (executeNativeGeometry(backend,stale,64,64,error,false,{},&missing,{},seconds,evaluator)) {
    error="Deferred particle spectrum admitted a stale native owner"; return false;
  }
  error.clear(); return true;
}

template <typename ReadPixels>
bool verifyNativeSpectralHistoryDomains(arbitgpu::NativeFixtureSceneBackend& backend,
    ReadPixels readPixels, std::string& error) {
  using namespace videowire::geometry;
  using namespace videohelper::geometry;
  NativeGeometryCoreCapabilitySource capabilities(backend);
  for (unsigned domain=1;domain<4;++domain) {
    const auto value=spectrumDomainFixture(domain,error,true);
    const auto contract=value ? spectrumDomainTerminalContract(*value) : PortContract{};
    const auto admitted=value ? admitValue(*value,contract,error) : std::nullopt;
    if (!admitted) return false;
    GeometryCorePlanRuntime runtime(capabilities);
    const auto bytes=lowerRuntimePlan(contract,*admitted);
    const auto preview=runtime.admitPreview({1,1,1,9611,1,PlanUse::preview},bytes,{}, {},error);
    const auto exported=runtime.admitExport({1,1,1,9611,1,PlanUse::exportRender},bytes,{}, {},error);
    if (!preview || !exported) return false;
    SpectrumEvaluation input; input.timeSeconds=1;
    input.historyFeaturesAt=[](double time) { return time>=0.2 && time<0.8 ? spectrumDomainBands(true) : spectrum::Bands{}; };
    const auto history=executeNativeGeometry(backend,*preview,64,64,error,false,{},&input);
    const auto offline=executeNativeGeometry(backend,*exported,64,64,error,false,{},&input);
    if (!history || !offline) return false;
    const auto historyPixels=readPixels(history->frame);
    input.timeSeconds=0.1; ++input.seekGeneration;
    const auto before=executeNativeGeometry(backend,*preview,64,64,error,false,{},&input);
    input.timeSeconds=1; ++input.loopGeneration;
    const auto restored=executeNativeGeometry(backend,*preview,64,64,error,false,{},&input);
    if (!before || !restored || historyPixels.empty() || historyPixels==readPixels(before->frame)
        || historyPixels!=readPixels(offline->frame) || historyPixels!=readPixels(restored->frame)) {
      error="Retained history domains must show past audio with silent current bands and replay exact native pixels"; return false;
    }
    input.historyFeaturesAt={};
    if (executeNativeGeometry(backend,*preview,64,64,error,false,{},&input)) {
      error="Retained history domains must refuse missing project-time analysis"; return false;
    }
    auto selected=*value;
    selected.spectrumFields.back().source=spectrum::Source::group; selected.spectrumFields.back().sourceTrackId=17;
    selected.spectrumFields.back().history.useTimeline=false; selected.spectrumFields.back().history.currentTimeSeconds=1;
    const auto selectedValue=admitValue(selected,contract,error);
    const auto selectedPlan=selectedValue ? runtime.admitPreview({1,1,2,9611,1,PlanUse::preview},
        lowerRuntimePlan(contract,*selectedValue),{}, {},error) : std::nullopt;
    if (!selectedPlan) return false;
    input.bands.fill(1); input.timeSeconds=100;
    input.sourceFeaturesAt=[](spectrum::Source source,std::int32_t id,double time,auto& bands) {
      if (source!=spectrum::Source::group || id!=17 || time>1) return false;
      bands=time>=0.2 && time<0.8 ? spectrumDomainBands(true) : spectrum::Bands{}; return true;
    };
    const auto group=executeNativeGeometry(backend,*selectedPlan,64,64,error,false,{},&input);
    if (!group || readPixels(group->frame)!=historyPixels) { error="Historical group source lost its exact selected identity"; return false; }
    input.sourceFeaturesAt=[](spectrum::Source,std::int32_t,double time,auto& bands) {
      bands={}; return time==1;
    };
    if (executeNativeGeometry(backend,*selectedPlan,64,64,error,false,{},&input)) {
      error="Historical group source substituted missing past samples"; return false;
    }
  }
  error.clear(); return true;
}

template <typename ReadPixels>
bool verifyNativeSpectrumDomains(arbitgpu::NativeFixtureSceneBackend& backend,
    ReadPixels readPixels, std::string& error) {
  using namespace videowire::geometry;
  using namespace videohelper::geometry;
  NativeGeometryCoreCapabilitySource capabilities(backend);
  for (unsigned domain=0;domain<4;++domain) {
    const auto value=spectrumDomainFixture(domain,error);
    if (!value) return false;
    const auto contract=spectrumDomainTerminalContract(*value);
    const auto admitted=admitValue(*value,contract,error);
    if (!admitted) return false;
    GeometryCorePlanRuntime runtime(capabilities);
    const auto bytes=lowerRuntimePlan(contract,*admitted);
    const auto preview=runtime.admitPreview({1,1,1,9611,1,PlanUse::preview},bytes,{}, {},error);
    const auto exported=runtime.admitExport({1,1,1,9611,1,PlanUse::exportRender},bytes,{}, {},error);
    if (!preview || !exported) return false;
    const auto draw=[&](const RuntimeAdmission& plan,bool changed) {
      SpectrumEvaluation spectrum; spectrum.bands=spectrumDomainBands(changed); spectrum.timeSeconds=3.25;
      spectrum.featuresAt=[changed](double){return spectrumDomainBands(changed);};
      return executeNativeGeometry(backend,plan,64,64,error,false,{},&spectrum,{},3.25);
    };
    const auto first=draw(*preview,false), changed=draw(*preview,true), offline=draw(*exported,true), sought=draw(*preview,false);
    if (!first || !changed || !offline || !sought) return false;
    const auto a=readPixels(first->frame), b=readPixels(changed->frame);
    if (a.empty() || a==b || b!=readPixels(offline->frame) || a!=readPixels(sought->frame)
        || first->drawnInstanceIds!=changed->drawnInstanceIds) {
      error="Typed spectrum domains must change native pixels causally and reproduce preview, export and seeks"; return false;
    }
    SpectrumEvaluation missing; missing.liveFrameAvailable=false;
    if (executeNativeGeometry(backend,*preview,64,64,error,false,{},&missing)) {
      error="Typed spectrum geometry admitted missing shared analysis"; return false;
    }
    auto selected=*value;
    selected.spectrumFields.back().source=spectrum::Source::group; selected.spectrumFields.back().sourceTrackId=17;
    auto selectedAdmitted=admitValue(selected,contract,error);
    const auto selectedPlan=selectedAdmitted ? runtime.admitPreview({1,1,2,9611,1,PlanUse::preview},
        lowerRuntimePlan(contract,*selectedAdmitted),{}, {},error) : std::nullopt;
    if (!selectedPlan) return false;
    SpectrumEvaluation group; group.timeSeconds=3.25;
    group.sourceFeaturesAt=[](spectrum::Source source,std::int32_t id,double,auto& bands) {
      if (source!=spectrum::Source::group || id!=17) return false;
      bands=spectrumDomainBands(true); return true;
    };
    const auto groupDraw=executeNativeGeometry(backend,*selectedPlan,64,64,error,false,{},&group,{},3.25);
    if (!groupDraw || readPixels(groupDraw->frame)!=b) { error="Selected group bands lost source identity"; return false; }
    selected.spectrumFields.back().attackSeconds=0.1f;
    const auto followedAdmitted=admitValue(selected,contract,error);
    const auto followedPlan=followedAdmitted ? runtime.admitPreview({1,1,3,9611,1,PlanUse::preview},
        lowerRuntimePlan(contract,*followedAdmitted),{}, {},error) : std::nullopt;
    if (!followedPlan) return false;
    group.sourceFeaturesAt=[](spectrum::Source,std::int32_t,double time,auto& bands) {
      bands=spectrumDomainBands(true); return time>=3;
    };
    if (executeNativeGeometry(backend,*followedPlan,64,64,error,false,{},&group,{},3.25)) {
      error="Selected group bands admitted a missing follower history sample"; return false;
    }
  }
  return verifyNativeSpectralHistoryDomains(backend,readPixels,error)
      && verifyNativeDeferredParticleSpectrum(backend,readPixels,error)
      && verifyNativeDeferredParticleSpectrum(backend,readPixels,error,true);
}

// Band 8 through 11 sound from 0.4 s to 0.6 s. With four rows spaced 0.25 s,
// a read at 1.0 s sees the sounding row at 0.5 s and silence in the others.
inline videowire::geometry::spectrum::Bands spectrumHistoryRowsPulse(double seconds) {
  videowire::geometry::spectrum::Bands bands {};
  if (seconds>=0.4 && seconds<0.6) for (std::size_t band=8;band<=11;++band) bands[band]=0.8f;
  return bands;
}

// One Spectral History deformer over a 4x4 grid. Vertex i reads history row round(i/5).
inline std::optional<videowire::geometry::ValueDescriptor> spectrumHistoryDeformerFixture(std::string& error) {
  using namespace videowire::geometry;
  PortContract contract;
  contract.carrier=CarrierKind::geometry3D; contract.overflow=OverflowPolicy::reject;
  contract.maxVertices=16; contract.maxIndices=54;
  const auto grid=lowerGrid(9861,9861,1,4,4,0.5f,contract,error);
  if (!grid) return std::nullopt;
  spectrum::Binding binding;
  binding.firstBand=8; binding.lastBand=8; binding.direction={0,0.6f,1};
  binding.history.enabled=true; binding.history.depth=4; binding.history.spacingSeconds=0.25f;
  const spectrum::HistoryAxis byIndex{0,-1,1};
  return lowerAudioDeformer(*grid,9862,binding,contract,error,&byIndex);
}

// One Spectral History instancer with four instances. Instance i reads history row i.
inline std::optional<videowire::geometry::ValueDescriptor> spectrumHistoryInstancerFixture(std::string& error) {
  using namespace videowire::geometry;
  PortContract meshContract;
  meshContract.carrier=CarrierKind::geometry3D; meshContract.overflow=OverflowPolicy::reject;
  meshContract.maxVertices=4; meshContract.maxIndices=6;
  const auto prototype=lowerGrid(9863,9863,1,2,2,1,meshContract,error);
  if (!prototype) return std::nullopt;
  PortContract contract;
  contract.carrier=CarrierKind::instances3D; contract.overflow=OverflowPolicy::reject;
  contract.maxInstances=64;
  spectrum::Binding binding;
  binding.firstBand=8; binding.lastBand=11; binding.baseScale={0.35f,0.35f,0.35f};
  binding.scaleResponse={0,3,0}; binding.direction={0,0.5f,0};
  binding.history.enabled=true; binding.history.depth=4; binding.history.spacingSeconds=0.25f;
  const spectrum::HistoryAxis byIndex{0,-1,1};
  return lowerSpectrumInstancer(*prototype,9864,binding,{0,2,2,2},contract,error,nullptr,&byIndex);
}

// Replays history-driven Audio Deformer and Spectrum Instancer rows through the native
// backend. The native result exposes frames, drawn IDs and stats, not transforms, so
// per-vertex and per-instance motion is checked through sounding versus silent pixels.
template <typename ReadPixels>
bool verifyNativeSpectralHistoryRows(arbitgpu::NativeFixtureSceneBackend& backend,
    ReadPixels readPixels, std::string& error) {
  using namespace videowire::geometry;
  using namespace videohelper::geometry;
  NativeGeometryCoreCapabilitySource capabilities(backend);
  const auto deformer=spectrumHistoryDeformerFixture(error);
  const auto instancer=spectrumHistoryInstancerFixture(error);
  if (!deformer || !instancer) return false;
  PortContract deformerContract;
  deformerContract.carrier=CarrierKind::geometry3D; deformerContract.overflow=OverflowPolicy::reject;
  deformerContract.maxVertices=16; deformerContract.maxIndices=54;
  PortContract instancerContract;
  instancerContract.carrier=CarrierKind::instances3D; instancerContract.overflow=OverflowPolicy::reject;
  instancerContract.maxInstances=64;
  const auto draw=[&](const auto& plan,double seconds) {
    SpectrumEvaluation input;
    input.timeSeconds=seconds; input.bands=spectrumHistoryRowsPulse(seconds);
    input.historyFeaturesAt=spectrumHistoryRowsPulse;
    return executeNativeGeometry(backend,plan,64,64,error,false,{},&input);
  };
  const auto replay=[&](const ValueDescriptor& value,const PortContract& contract,StableId id,
                        std::size_t instanceCount,const char* label) {
    const auto admitted=admitValue(value,contract,error);
    if (!admitted) return false;
    GeometryCorePlanRuntime runtime(capabilities);
    const auto bytes=lowerRuntimePlan(contract,*admitted);
    const auto preview=runtime.admitPreview({1,1,1,id,1,PlanUse::preview},bytes,{}, {},error);
    const auto exported=runtime.admitExport({1,1,1,id,1,PlanUse::exportRender},bytes,{}, {},error);
    if (!preview || !exported) return false;
    const auto silent=draw(*preview,0.2), sounding=draw(*preview,1.0), offline=draw(*exported,1.0);
    if (!silent || !sounding || !offline) return false;
    const auto quietPixels=readPixels(silent->frame), activePixels=readPixels(sounding->frame);
    if (quietPixels.empty() || activePixels==quietPixels || activePixels!=readPixels(offline->frame)
        || sounding->drawnInstanceIds!=silent->drawnInstanceIds
        || (instanceCount!=0 && sounding->drawnInstanceIds.size()!=instanceCount)) {
      error=std::string(label)+" history rows must move only sounding elements and match export pixels";
      return false;
    }
    const auto sought=draw(*preview,0.2), restored=draw(*preview,1.0);
    if (!sought || !restored || readPixels(sought->frame)!=quietPixels || readPixels(restored->frame)!=activePixels) {
      error=std::string(label)+" history seek and loop did not restore the same native frame";
      return false;
    }
    SpectrumEvaluation missing; missing.timeSeconds=1.0;
    if (executeNativeGeometry(backend,*preview,64,64,error,false,{},&missing)) {
      error=std::string(label)+" history admitted missing project-time analysis"; return false;
    }
    return true;
  };
  const bool ok=replay(*deformer,deformerContract,9862,0,"Audio Deformer")
      && replay(*instancer,instancerContract,9864,4,"Spectrum Instancer");
  if (ok) error.clear();
  return ok;
}
