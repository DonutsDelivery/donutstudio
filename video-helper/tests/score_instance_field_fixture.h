#pragma once

#include "score_field_fixture.h"
#include "instance_appearance_fixture.h"

inline std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> instanceScoreFrameAt(
    std::int64_t frameNumber, bool reverse=false) {
  auto score=std::make_shared<arbitmod::Score>(); score->notationVersion=1; score->scoreRevision=1;
  score->rootFreq=440; score->historyBeats=score->lookaheadBeats=8;
  for (int i=0;i<2;++i) {
    arbitmod::Note note; note.id=i==0 ? -1 : -102; note.startBeat=static_cast<float>(i*2);
    note.lengthBeats=8; note.velocity=127; note.freqHz=i==0 ? 440.0f : 880.0f; note.durationSeconds=4; note.trackId=0;
    for (std::size_t prime=0;prime<6;++prime) note.primes[prime]=static_cast<float>(prime+1);
    if (i==1) note.linkMasterId=-1;
    score->notes.push_back(note);
  }
  if (reverse) std::reverse(score->notes.begin(),score->notes.end());
  score->links.push_back({-301,-102,-1,2,1,0});
  auto key=scoreFieldFrame()->key; key.frame=frameNumber;
  canonicalblockc::FrameProducer producer;
  return producer.evaluate(key,score,static_cast<float>(double(frameNumber)/60*2));
}

inline std::optional<videowire::geometry::ValueDescriptor> instanceScoreFieldFixture(
    unsigned mode, const std::string& property, std::string& error) {
  using namespace videowire::geometry;
  const auto source=instanceAppearanceFixture(0,error);
  if (!source) return std::nullopt;
  scorefield::Binding binding; binding.scoreSourceStableId=9910;
  binding.mode=static_cast<scorefield::Mode>(mode); binding.radius=2;
  binding.gain=mode==2 ? 0.025f : mode==3 ? 0.1f : 0.4f;
  binding.direction={1,0.5f,0.25f};
  const auto vectorContract=constructedFieldContract(Domain::instance,ValueType::vector);
  const auto scalarContract=constructedFieldContract(Domain::instance,ValueType::floatValue);
  const auto colorContract=constructedFieldContract(Domain::instance,ValueType::color);
  const auto field=lowerInstanceScoreField(*source,9911,binding,vectorContract,error);
  FieldOperationSettings settings; settings.domain=Domain::instance; settings.valueType=ValueType::floatValue;
  settings.mode=3;
  const auto component=field ? lowerConstructedField(9912,OperationCode::fieldVectorMeasure,settings,
      &*field,nullptr,nullptr,scalarContract,error) : std::nullopt;
  settings.mode=0; settings.values={0,1,0,0,0,0,0,0};
  const auto scalar=component ? lowerConstructedField(9913,OperationCode::fieldClamp,settings,
      &*component,nullptr,nullptr,scalarContract,error) : std::nullopt;
  settings.valueType=ValueType::boolean; settings.mode=2; settings.values={0.1,0,0,0,0,0,0,0};
  const auto selected=scalar ? lowerConstructedField(9914,OperationCode::fieldCompare,settings,
      &*scalar,nullptr,nullptr,constructedFieldContract(Domain::instance,ValueType::boolean),error) : std::nullopt;
  if (!scalar || !selected) return std::nullopt;
  if (property=="opacity" || property=="metallic" || property=="roughness")
    return lowerInstanceAppearance(*source,*scalar,&*selected,9920,property,appearanceInstanceContract(),error);
  settings.valueType=ValueType::color; settings.mode=0; settings.values={0.1,0.2,0.3,1,1,0,0,0};
  const auto low=lowerConstructedField(9915,OperationCode::fieldConstant,settings,nullptr,nullptr,nullptr,colorContract,error);
  settings.values={0.8,0.3,0.1,1,1,0,0,0};
  const auto high=lowerConstructedField(9916,OperationCode::fieldConstant,settings,nullptr,nullptr,nullptr,colorContract,error);
  settings.values={};
  const auto color=low && high ? lowerConstructedField(9917,OperationCode::fieldMix,settings,&*low,&*high,&*scalar,colorContract,error) : std::nullopt;
  return color ? lowerInstanceAppearance(*source,*color,&*selected,9920,property,appearanceInstanceContract(),error) : std::nullopt;
}

inline std::optional<videowire::geometry::ValueDescriptor> fixedPlacementScoreFixture(
    const videowire::geometry::ValueDescriptor& source, videowire::geometry::StableId id, std::string& error) {
  using namespace videowire::geometry;
  scorefield::Binding binding; binding.scoreSourceStableId=9910;
  binding.mode=scorefield::Mode::closestNoteDistance; binding.gain=0.2f; binding.bias=0.1f;
  binding.direction={1,0,0};
  binding.mapping.x=visualnoteinstancing::MappingAxis::LogFrequency;
  binding.mapping.xScale=binding.mapping.yScale=12;
  const auto vector=lowerInstanceScoreField(source,id,binding,
      constructedFieldContract(Domain::instance,ValueType::vector),error);
  FieldOperationSettings settings; settings.domain=Domain::instance; settings.valueType=ValueType::floatValue; settings.mode=3;
  const auto scalar=vector ? lowerConstructedField(id+1,OperationCode::fieldVectorMeasure,settings,&*vector,nullptr,nullptr,
      constructedFieldContract(Domain::instance,ValueType::floatValue),error) : std::nullopt;
  return scalar ? lowerInstanceAppearance(source,*scalar,nullptr,id+2,"opacity",appearanceInstanceContract(),error) : std::nullopt;
}

// A retained request and a synthetic topology-stable frame provider isolate the
// placement/Score Field ordering. This fixture does not prove GLB asset loading.
inline std::optional<videowire::geometry::ValueDescriptor> animatedScorePlacementFixture(std::string& error) {
  using namespace videowire::geometry;
  auto placement=lowerGrid(9970,9970,1,2,2,1.2f,constructedFieldMeshContract(),error);
  const auto prototype=lowerMeshGenerator(9972,9972,1,OperationCode::plane,{0.3f,0.3f,0,0},constructedFieldMeshContract(),error);
  if (!placement || !prototype) return std::nullopt;
  visualanimationimport::Request request; request.sourceStableId=9970; request.deformationStableId=9974;
  request.schedule={9970,9974}; request.meshStableId=1; request.animationClipStableId=1; request.clipName="motion";
  request.asset={"score-field-placement",1,std::string(64,'a'),"model/gltf-binary",16};
  auto retained=std::make_shared<RetainedMeshData>(); retained->geometry=std::get<GeometryData>(placement->data);
  retained->importedAnimation=visualanimationoperation::encode(request);
  if (retained->importedAnimation.empty()) { error="Animation placement fixture request is invalid"; return std::nullopt; }
  auto& operation=placement->operations.front(); operation.code=OperationCode::retainedMesh;
  operation.parameters={}; operation.retainedMesh=std::move(retained);
  PortContract pointsContract; pointsContract.carrier=CarrierKind::points3D; pointsContract.maxPoints=4;
  const auto points=lowerPointsFromVertices(*placement,9971,pointsContract,error);
  auto instances=points ? lowerInstanceOnPoints(*points,9972,9973,appearanceInstanceContract(),error) : std::nullopt;
  if (!instances) return std::nullopt;
  instances->operations.insert(instances->operations.begin(),prototype->operations.begin(),prototype->operations.end());
  instances->dispatchCount=instances->operations.size();
  return fixedPlacementScoreFixture(*instances,9975,error);
}

template <typename ReadPixels>
bool verifyNativeFixedInstanceScorePlacements(arbitgpu::NativeFixtureSceneBackend& backend,
    ReadPixels readPixels,std::string& error) {
  using namespace videowire::geometry;
  using namespace videohelper::geometry;
  NativeGeometryCoreCapabilitySource capabilities(backend); GeometryCorePlanRuntime runtime(capabilities);
  const auto prototype=lowerMeshGenerator(10010,10010,1,OperationCode::plane,{0.5f,0.5f,0,0},constructedFieldMeshContract(),error);
  spectrum::Binding spectrumBinding; spectrumBinding.firstBand=spectrumBinding.lastBand=2;
  spectrumBinding.direction={0.4f,0,0}; spectrumBinding.scaleResponse={0.5f,0.25f,0};
  const auto source=prototype ? lowerSpectrumInstancer(*prototype,10011,spectrumBinding,{0,1,1,1},appearanceInstanceContract(),error) : std::nullopt;
  const auto value=source ? fixedPlacementScoreFixture(*source,10012,error) : std::nullopt;
  if (!value) return false;
  const auto admit=[&](const ValueDescriptor& descriptor,PlanUse use)->std::optional<RuntimeAdmission> {
    const auto contract=withAttributeContract(appearanceInstanceContract(),descriptor);
    const auto admitted=admitValue(descriptor,contract,error);
    if (!admitted) return std::nullopt;
    const auto bytes=lowerRuntimePlan(contract,*admitted);
    const PlanOwnerIdentity owner{1,1,1,descriptor.stableId,1,use};
    return use==PlanUse::preview ? runtime.admitPreview(owner,bytes,{}, {},error)
        : runtime.admitExport(owner,bytes,{}, {},error);
  };
  const auto preview=admit(*value,PlanUse::preview),exported=admit(*value,PlanUse::exportRender);
  if (!preview || !exported) return false;
  SpectrumEvaluation spectrumInput; spectrumInput.bands[2]=0.6f;
  arbitgpu::NativeFixtureSceneRuntimeInputs inputs; inputs.canonicalBlockCFrame=instanceScoreFrameAt(0);
  const auto first=executeNativeGeometry(backend,*preview,64,64,error,false,inputs,&spectrumInput);
  // The legacy Spectrum path applies one post-replay transform. Give it the
  // independently predicted opacity to detect a second transform in the new path.
  FieldOperationSettings constant; constant.domain=Domain::instance; constant.valueType=ValueType::floatValue;
  constant.values={double(0.6f*0.4f)*0.2+0.1,0,0,0,1,0,0,0};
  const auto field=lowerConstructedField(10015,OperationCode::fieldConstant,constant,nullptr,nullptr,nullptr,
      constructedFieldContract(Domain::instance,ValueType::floatValue),error);
  const auto reference=field ? lowerInstanceAppearance(*source,*field,nullptr,10016,"opacity",appearanceInstanceContract(),error) : std::nullopt;
  const auto referencePlan=reference ? admit(*reference,PlanUse::preview) : std::nullopt;
  auto exportSpectrum=spectrumInput; exportSpectrum.followers.clear();
  const auto offline=executeNativeGeometry(backend,*exported,64,64,error,false,inputs,&exportSpectrum);
  const auto expected=referencePlan ? executeNativeGeometry(backend,*referencePlan,64,64,error,false,inputs,&exportSpectrum) : std::nullopt;
  if (!first || !offline || !expected || readPixels(first->frame)!=readPixels(offline->frame)
      || readPixels(first->frame)!=readPixels(expected->frame) || first->drawnInstanceIds!=std::vector<StableId>{3}) {
    error="Current instance Score Fields disagree with independent single-application Spectrum rendering"; return false;
  }
  const auto animation=animatedScorePlacementFixture(error);
  const auto animationPlan=animation ? admit(*animation,PlanUse::preview) : std::nullopt;
  const auto animationExport=animation ? admit(*animation,PlanUse::exportRender) : std::nullopt;
  if (!animationPlan || !animationExport) return false;
  const auto moved=[](auto& mesh,std::string&) {
    for (auto& p : mesh.geometry.positions) p.x+=0.3f;
    return true;
  };
  const auto base=[](auto&,std::string&) { return true; };
  const auto a=executeNativeGeometry(backend,*animationPlan,64,64,error,false,inputs,nullptr,base,0);
  inputs.canonicalBlockCFrame=instanceScoreFrameAt(60);
  const auto b=executeNativeGeometry(backend,*animationPlan,64,64,error,false,inputs,nullptr,moved,1);
  const auto c=executeNativeGeometry(backend,*animationExport,64,64,error,false,inputs,nullptr,moved,1);
  inputs.canonicalBlockCFrame=instanceScoreFrameAt(0);
  const auto reset=executeNativeGeometry(backend,*animationPlan,64,64,error,false,inputs,nullptr,base,0);
  if (!a || !b || !c || !reset || readPixels(a->frame)==readPixels(b->frame)
      || readPixels(b->frame)!=readPixels(c->frame) || readPixels(a->frame)!=readPixels(reset->frame)
      || a->drawnInstanceIds!=std::vector<StableId>{1,2,3,4} || b->drawnInstanceIds!=a->drawnInstanceIds) {
    error="Animated instance Score Fields lost current-position causality, exact IDs or export/reset parity"; return false;
  }
  inputs.canonicalBlockCFrame=instanceScoreFrameAt(60);
  for (unsigned invalid=0;invalid<4;++invalid) {
    const auto hostile=[invalid](auto& mesh,std::string&) {
      if (invalid==0) mesh.geometry.positions.pop_back();
      if (invalid==1) mesh.geometry.vertexIds.front()=123;
      if (invalid==2) std::swap(mesh.geometry.indices[0],mesh.geometry.indices[1]);
      if (invalid==3) mesh.importedAnimation.clear();
      return true;
    };
    if (executeNativeGeometry(backend,*animationPlan,64,64,error,false,inputs,nullptr,hostile,1)) {
      error="Animation provider changed admitted count, indices, ordered IDs or source request"; return false;
    }
  }
  if (executeNativeGeometry(backend,*animationPlan,64,64,error,false,inputs)) {
    error="Animated instance Score Fields rendered without the exact frame provider"; return false;
  }
  error.clear(); return true;
}

// Wire this hook into the existing GL/Metal geometry fixture entries after the
// covering batch builds. Editor preview and decoded export need separate proof.
template <typename ReadPixels>
bool verifyNativeInstanceScoreFields(arbitgpu::NativeFixtureSceneBackend& backend,
    ReadPixels readPixels,std::string& error) {
  using namespace videowire::geometry;
  using namespace videohelper::geometry;
  NativeGeometryCoreCapabilitySource capabilities(backend);
  for (const auto* property : {"color","emission","opacity","metallic","roughness"}) {
    const auto value=instanceScoreFieldFixture(1,property,error);
    if (!value) return false;
    const auto contract=withAttributeContract(appearanceInstanceContract(),*value);
    const auto admitted=admitValue(*value,contract,error);
    if (!admitted) return false;
    GeometryCorePlanRuntime runtime(capabilities);
    const auto bytes=lowerRuntimePlan(contract,*admitted);
    const auto preview=runtime.admitPreview({1,1,1,9920,1,PlanUse::preview},bytes,{}, {},error);
    const auto exported=runtime.admitExport({1,1,1,9920,1,PlanUse::exportRender},bytes,{}, {},error);
    if (!preview || !exported) return false;
    arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
    if (std::string(property)=="emission") inputs.imageOutput=renderpassoutput::Output::Emission;
    inputs.canonicalBlockCFrame=scoreFieldFrame();
    const auto first=executeNativeGeometry(backend,*preview,64,64,error,false,inputs);
    inputs.canonicalBlockCFrame=scoreFieldFrame(true);
    const auto offline=executeNativeGeometry(backend,*exported,64,64,error,false,inputs);
    inputs.canonicalBlockCFrame=scoreFieldFrame(false,false,true,32);
    const auto softer=executeNativeGeometry(backend,*preview,64,64,error,false,inputs);
    inputs.canonicalBlockCFrame=scoreFieldFrame();
    const auto sought=executeNativeGeometry(backend,*preview,64,64,error,false,inputs);
    if (!first || !offline || !softer || !sought) return false;
    const auto pixels=readPixels(first->frame);
    if (pixels.empty() || pixels!=readPixels(offline->frame) || pixels!=readPixels(sought->frame)
        || pixels==readPixels(softer->frame) || first->drawnInstanceIds!=std::vector<StableId>{1,2,3,4}
        || offline->drawnInstanceIds!=first->drawnInstanceIds || sought->drawnInstanceIds!=first->drawnInstanceIds
        || first->stats.noteInstanceDrawCount!=0) {
      error="Instance Score Fields lost causal appearance, exact IDs or equal-frame preview/export/seek parity"; return false;
    }
    inputs.canonicalBlockCFrame={};
    if (executeNativeGeometry(backend,*preview,64,64,error,false,inputs)) {
      error="Instance Score Field rendered without a canonical score frame"; return false;
    }
    inputs.canonicalBlockCFrame=scoreFieldFrame(false,true);
    if (executeNativeGeometry(backend,*preview,64,64,error,false,inputs)) {
      error="Instance Score Field latched appearance through an empty score window"; return false;
    }
  }
  return verifyNativeFixedInstanceScorePlacements(backend,readPixels,error);
}
