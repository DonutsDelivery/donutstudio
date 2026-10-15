#pragma once

#include "spectrum_field_domains_fixture.h"
#include "../../shared/HdrImageOutputContract.h"
#include <cstring>

inline std::optional<videowire::geometry::ValueDescriptor> spectrumInstancerLayoutFixture(
    unsigned mode, std::string& error, bool movedPlacement=false) {
  using namespace videowire::geometry;
  const auto meshContract=constructedFieldMeshContract();
  auto prototype=lowerMeshGenerator(9801,9801,1,OperationCode::plane,{0.5f,0.5f,0,0},meshContract,error);
  prototype=prototype ? lowerMaterialTableSlot(*prototype,9802,9803,0,
      {0.3,0.7,1,0.2,0.4,0.8,0,0.4},1,meshContract,error) : std::nullopt;
  std::optional<ValueDescriptor> placement;
  if (mode==3) {
    PortContract curves; curves.carrier=CarrierKind::curves3D; curves.maxCurvePoints=8; curves.maxSplines=1;
    placement=lowerCurveCircle(9810,9810,1,8,movedPlacement ? 0.7f : 0.5f,curves,error);
  } else {
    placement=lowerGrid(9810,9810,1,2,2,movedPlacement ? 1.1f : 0.8f,meshContract,error);
    if (placement && mode==4) {
      PortContract points; points.carrier=CarrierKind::points3D; points.maxPoints=4;
      placement=lowerPointsFromVertices(*placement,9811,points,error);
    }
  }
  if (!prototype || !placement || mode<3 || mode>5) return std::nullopt;
  auto binding=spectrumDomainBinding(); binding.baseScale={0.5f,0.5f,0.5f};
  binding.scaleResponse={1,1,0}; binding.direction={0.22f,0.1f,0};
  return lowerSpectrumInstancer(*prototype,9820,binding,{static_cast<float>(mode),2,2,2},
      appearanceInstanceContract(),error,&*placement);
}

// These execute the existing native instance path. They are helper parity hooks;
// real editor preview and decoded export still need their own acceptance rows.
template <typename ReadPixels>
bool verifyNativeSpectrumInstancerLayouts(arbitgpu::NativeFixtureSceneBackend& backend,
    ReadPixels readPixels, std::string& error) {
  using namespace videowire::geometry;
  using namespace videohelper::geometry;
  NativeGeometryCoreCapabilitySource capabilities(backend);
  for (unsigned mode=3;mode<=5;++mode) {
    const auto value=spectrumInstancerLayoutFixture(mode,error);
    const auto contract=appearanceInstanceContract();
    const auto admitted=value ? admitValue(*value,contract,error) : std::nullopt;
    if (!admitted) return false;
    GeometryCorePlanRuntime runtime(capabilities);
    const auto bytes=lowerRuntimePlan(contract,*admitted);
    const auto preview=runtime.admitPreview({1,1,1,9820,1,PlanUse::preview},bytes,{}, {},error);
    const auto exported=runtime.admitExport({1,1,1,9820,1,PlanUse::exportRender},bytes,{}, {},error);
    if (!preview || !exported) return false;
    const auto draw=[&](const RuntimeAdmission& plan,bool changed,double time) {
      SpectrumEvaluation spectrum; spectrum.bands=spectrumDomainBands(changed); spectrum.timeSeconds=time;
      spectrum.featuresAt=[changed](double){return spectrumDomainBands(changed);};
      return executeNativeGeometry(backend,plan,64,64,error,false,{},&spectrum,{},time);
    };
    const auto first=draw(*preview,false,3.25), changed=draw(*preview,true,3.25);
    const auto offline=draw(*exported,true,3.25), sought=draw(*preview,false,1.0);
    if (!first || !changed || !offline || !sought) return false;
    const auto a=readPixels(first->frame), b=readPixels(changed->frame);
    if (a.empty() || a==b || b!=readPixels(offline->frame) || a!=readPixels(sought->frame)
        || first->drawnInstanceIds!=std::vector<StableId>{3,4,5,6}
        || first->drawnInstanceIds!=changed->drawnInstanceIds) {
      error="Spectrum Instancer layouts lost causal bands, absolute IDs or native preview/export/seek parity"; return false;
    }
    const auto moved=spectrumInstancerLayoutFixture(mode,error,true);
    const auto movedAdmitted=moved ? admitValue(*moved,contract,error) : std::nullopt;
    const auto movedPlan=movedAdmitted ? runtime.admitPreview({1,1,2,9820,1,PlanUse::preview},
        lowerRuntimePlan(contract,*movedAdmitted),{}, {},error) : std::nullopt;
    const auto movedDraw=movedPlan ? draw(*movedPlan,false,3.25) : std::nullopt;
    if (!movedDraw || readPixels(movedDraw->frame)==a || movedDraw->drawnInstanceIds!=first->drawnInstanceIds) {
      error="Spectrum Instancer must consume authored placement changes with stable band identities"; return false;
    }
    SpectrumEvaluation missing; missing.liveFrameAvailable=false;
    if (executeNativeGeometry(backend,*preview,64,64,error,false,{},&missing)) {
      error="Spectrum Instancer admitted missing shared analysis"; return false;
    }
    auto selected=*value; selected.spectrumFields[0].source=spectrum::Source::group;
    selected.spectrumFields[0].sourceTrackId=17;
    auto selectedAdmitted=admitValue(selected,contract,error);
    const auto selectedPlan=selectedAdmitted ? runtime.admitPreview({1,1,3,9820,1,PlanUse::preview},
        lowerRuntimePlan(contract,*selectedAdmitted),{}, {},error) : std::nullopt;
    if (!selectedPlan) return false;
    SpectrumEvaluation group; group.timeSeconds=3.25;
    group.sourceFeaturesAt=[](spectrum::Source source,std::int32_t id,double,auto& bands) {
      if (source!=spectrum::Source::group || id!=17) return false;
      bands=spectrumDomainBands(true); return true;
    };
    const auto groupDraw=executeNativeGeometry(backend,*selectedPlan,64,64,error,false,{},&group,{},3.25);
    if (!groupDraw || readPixels(groupDraw->frame)!=b) {
      error="Spectrum Instancer lost selected group identity"; return false;
    }
    group.sourceFeaturesAt={};
    if (executeNativeGeometry(backend,*selectedPlan,64,64,error,false,{},&group)) {
      error="Spectrum Instancer admitted absent selected source"; return false;
    }
    selected.spectrumFields[0].attackSeconds=0.1f;
    const auto followedAdmitted=admitValue(selected,contract,error);
    const auto followedPlan=followedAdmitted ? runtime.admitPreview({1,1,4,9820,1,PlanUse::preview},
        lowerRuntimePlan(contract,*followedAdmitted),{}, {},error) : std::nullopt;
    if (!followedPlan) return false;
    group.sourceFeaturesAt=[](spectrum::Source,std::int32_t,double time,auto& bands) {
      bands=spectrumDomainBands(true); return time>=3;
    };
    if (executeNativeGeometry(backend,*followedPlan,64,64,error,false,{},&group)) {
      error="Spectrum Instancer admitted missing selected-source follower history"; return false;
    }
  }
  error.clear(); return true;
}

inline std::optional<videowire::geometry::ValueDescriptor> spectrumInstancerAppearanceFixture(
    std::string& error, bool fixedTransforms=false, double scalarMultiplier=0.5,
    double mixFactor=0.25, double threshold=0.25) {
  using namespace videowire::geometry;
  auto source=spectrumInstancerLayoutFixture(5,error);
  if (!source) return std::nullopt;
  if (fixedTransforms) {
    source->spectrumFields.front().direction={0,0,0};
    source->spectrumFields.front().scaleResponse={0,0,0};
  }
  const auto scalarContract=constructedFieldContract(Domain::instance,ValueType::floatValue);
  const auto colorContract=constructedFieldContract(Domain::instance,ValueType::color);
  const auto booleanContract=constructedFieldContract(Domain::instance,ValueType::boolean);
  auto binding=spectrumDomainBinding(); binding.direction={0.2f,1,0.1f};
  const auto settings=spectrumDomainSettings(Domain::instance,ValueType::floatValue,7);
  const auto scalar=lowerAudioSpectrumField(*source,9830,binding,settings,scalarContract,error);
  auto colorSettings=settings; colorSettings.valueType=ValueType::color;
  const auto color=lowerAudioSpectrumField(*source,9831,binding,colorSettings,colorContract,error);
  if (!scalar || !color) return std::nullopt;
  FieldOperationSettings math; math.domain=Domain::instance; math.valueType=ValueType::floatValue;
  math.mode=2; math.values[0]=scalarMultiplier;
  const auto scaled=lowerConstructedField(9832,OperationCode::fieldMath,math,
      &*scalar,nullptr,nullptr,scalarContract,error);
  math.valueType=ValueType::boolean; math.values[0]=threshold;
  const auto selection=lowerConstructedField(9833,OperationCode::fieldCompare,math,
      &*scalar,nullptr,nullptr,booleanContract,error);
  FieldOperationSettings mix; mix.domain=Domain::instance; mix.valueType=ValueType::color;
  mix.values={0.1,0.2,0.3,1,mixFactor,0,0,0};
  const auto mixed=lowerConstructedField(9834,OperationCode::fieldMix,mix,
      &*color,nullptr,nullptr,colorContract,error);
  if (!scaled || !selection || !mixed) return std::nullopt;
  auto terminal=*source;
  StableId id=9840;
  for (const auto* property : {"color","emission","opacity","metallic","roughness"}) {
    const auto& field=std::string(property)=="color" || std::string(property)=="emission" ? *mixed : *scaled;
    const auto next=lowerInstanceAppearance(terminal,field,&*selection,id++,property,
        appearanceInstanceContract(),error);
    if (!next) return std::nullopt;
    terminal=*next;
  }
  return terminal;
}

// The expected values below come from the authored multiply, mix and comparison,
// without calling the production field or material evaluators for the oracle.
template <typename ReadPixels>
bool verifyNativeSpectrumInstancerAppearance(arbitgpu::NativeFixtureSceneBackend& backend,
    ReadPixels readPixels, std::string& error) {
  using namespace videowire::geometry;
  using namespace videohelper::geometry;
  using Output=renderpassoutput::Output;
  using Format=arbitgpu::NativeTexturePixelFormat;
  constexpr unsigned extent=96;
  const auto reject=[&](const char* message) { error=message; return false; };
  const auto bandsFor=[](unsigned variant) {
    return variant==2 ? spectrum::Bands{} : spectrumDomainBands(variant==1);
  };
  const auto expectedColor=[](double band) {
    return band>0.25 ? std::array<double,4>{0.15*band+0.025,0.75*band+0.05,0.075*band+0.075,1}
                    : std::array<double,4>{1,1,1,1};
  };
  const auto expectedEmission=[&](double band) {
    return band>0.25 ? expectedColor(band) : std::array<double,4>{0,0,0,1};
  };
  const auto checkReplay=[&](const ValueDescriptor& value,unsigned variant) {
    RuntimeFieldEvaluation evaluation;
    evaluation.spectrumAt=[&](const auto&,auto& bands,std::string&) { bands=bandsFor(variant); return true; };
    MaterializedInstancePlan retained;
    ValueDescriptor replay;
    if (!validateOperationPlan(value,{},&retained,error,&replay,&evaluation)) return false;
    const auto& instances=std::get<InstancesData>(replay.data).instances;
    if (instances.size()!=4 || value.spectrumFields.size()!=3
        || value.spectrumFields[0].target!=spectrum::Target::instanceTransform
        || value.spectrumFields[1].sourceGeometryStableId!=9820
        || value.spectrumFields[2].sourceGeometryStableId!=9820
        || retained.materialTables.count(value.stableId)!=1
        || retained.materialTables.at(value.stableId).count(0)!=1
        || retained.materialTables.at(value.stableId).at(0).secondaryInputStableId!=9803)
      return reject("Composed spectrum appearance lost its exact instancer, prototype or authored material source");
    const auto bands=bandsFor(variant);
    for (std::size_t i=0;i<4;++i) {
      if (instances[i].stableId!=i+3 || instances[i].sourceStableId!=9802)
        return reject("Composed spectrum appearance changed absolute band IDs or the retained prototype");
      const double band=bands[i+2];
      for (const auto* property : {"color","emission","opacity","metallic","roughness"}) {
        const auto* attribute=findNamedAttribute(replay.attributes,property);
        if (!attribute || attribute->elements.size()!=4)
          return reject("Composed spectrum appearance omitted an instance property");
        std::array<double,4> expected{};
        const bool color=std::string(property)=="color", emission=std::string(property)=="emission";
        if (color) expected=expectedColor(band);
        else if (emission) expected=expectedEmission(band);
        else expected[0]=band>0.25 ? 0.5*band : std::string(property)=="metallic" ? 0 : 1;
        for (std::size_t channel=0;channel<(color || emission ? 4u : 1u);++channel)
          if (std::abs(attribute->elements[i].components[channel]-expected[channel])>1e-6)
            return reject("Composed spectrum appearance disagrees with independent math, mix or selection values");
      }
      for (const auto* property : {"metallicSelected","roughnessSelected"}) {
        const auto* selected=findNamedAttribute(replay.attributes,property);
        if (!selected || selected->elements.size()!=4
            || selected->elements[i].components[0]!=(band>0.25 ? 1 : 0))
          return reject("Unselected spectrum instances did not inherit authored material scalars");
      }
    }
    return true;
  };
  NativeGeometryCoreCapabilitySource capabilities(backend);
  GeometryCorePlanRuntime runtime(capabilities);
  const auto planFor=[&](const ValueDescriptor& value,std::uint64_t revision,bool exportUse=false)
      -> std::optional<RuntimeAdmission> {
    const auto contract=withAttributeContract(appearanceInstanceContract(),value);
    const auto admitted=admitValue(value,contract,error);
    if (!admitted) return std::nullopt;
    const auto bytes=lowerRuntimePlan(contract,*admitted);
    const PlanOwnerIdentity owner{1,1,1,value.stableId,revision,
        exportUse ? PlanUse::exportRender : PlanUse::preview};
    return exportUse ? runtime.admitExport(owner,bytes,{}, {},error)
                     : runtime.admitPreview(owner,bytes,{}, {},error);
  };
  const auto draw=[&](const RuntimeAdmission& plan,unsigned variant,double time=3.25,
      bool diagnostic=false,Output output=Output::Color) {
    SpectrumEvaluation spectrum; spectrum.bands=bandsFor(variant); spectrum.timeSeconds=time;
    spectrum.seekGeneration=time==3.25 ? 1 : 2;
    spectrum.featuresAt=[&](double) { return bandsFor(variant); };
    arbitgpu::NativeFixtureSceneRuntimeInputs inputs; inputs.imageOutput=output;
    return executeNativeGeometry(backend,plan,extent,extent,error,diagnostic,inputs,&spectrum,{},time);
  };
  const auto raw=[&](const NativeGeometryExecution& execution,Output output,Format format,
      arbitgpu::NativeRawPassPixels& pixels) {
    if (!execution.frame || !execution.frame->readRawPass(output,pixels,error)) return false;
    if (pixels.format!=format || pixels.width!=extent || pixels.height!=extent
        || pixels.bytes.size()!=extent*extent*arbitgpu::rawPassChannels(format)
            *arbitgpu::rawPassScalarBytes(format))
      return reject("Composed spectrum appearance returned the wrong raw attachment layout");
    return true;
  };
  const auto idAt=[](const arbitgpu::NativeRawPassPixels& pixels,std::size_t i) {
    std::uint32_t id=0; std::memcpy(&id,pixels.bytes.data()+i*4,4); return id;
  };
  const auto checkRaw=[&](const NativeGeometryExecution& execution,unsigned variant) {
    arbitgpu::NativeRawPassPixels ids,materials,emission,mask;
    if (!raw(execution,Output::ObjectId,Format::R32Uint,ids)
        || !raw(execution,Output::MaterialId,Format::R32Uint,materials)
        || !raw(execution,Output::Emission,Format::Rgba16Float,emission)
        || !raw(execution,Output::Mask,Format::R8Unorm,mask)) return false;
    const std::array<std::uint32_t,4> expectedMaterials=variant==0 ? std::array<std::uint32_t,4>{1,2,3,4}
        : variant==1 ? std::array<std::uint32_t,4>{1,2,3,3} : std::array<std::uint32_t,4>{1,1,1,1};
    const auto bands=bandsFor(variant);
    const std::array<double,3> authoredEmission{0.2,0.4,0.8};
    std::set<std::uint32_t> visible;
    bool selectedSeen=false,unselectedSeen=false;
    for (std::size_t i=0;i<extent*extent;++i) {
      const auto id=idAt(ids,i), material=idAt(materials,i);
      if (!id) {
        if (material || mask.bytes[i]) return reject("Composed spectrum background has a raw scene identity or coverage");
        continue;
      }
      if (id<3 || id>6 || material!=expectedMaterials[id-3])
        return reject("Composed spectrum raw IDs differ from absolute bands and canonical material variants");
      visible.insert(id);
      const double band=bands[id-1];
      selectedSeen=selectedSeen || band>0.25; unselectedSeen=unselectedSeen || band<=0.25;
      const auto expected=expectedEmission(band);
      for (std::size_t channel=0;channel<3;++channel) {
        std::uint16_t half=0; std::memcpy(&half,emission.bytes.data()+i*8+channel*2,2);
        const double actual=hdrimage::halfToFloat(half);
        if (!std::isfinite(actual) || std::abs(actual-authoredEmission[channel]-expected[channel])>0.002)
          return reject("Native spectrum emission disagrees with the authored table plus independent field mix");
      }
      const double opacity=band>0.25 ? 0.5*band : 1;
      if (std::abs(double(mask.bytes[i])/255-opacity)>1.0/255+1e-6)
        return reject("Native spectrum coverage disagrees with independent opacity and selection");
    }
    if (visible.size()<2 || !unselectedSeen || (variant!=2 && !selectedSeen))
      return reject("Composed spectrum fixture did not render selected and inherited material instances");
    return true;
  };
  std::vector<std::uint8_t> fixedPixels;
  for (const bool fixed : {false,true}) {
    const auto value=spectrumInstancerAppearanceFixture(error,fixed);
    if (!value) return false;
    for (unsigned variant=0;variant<3;++variant)
      if (!checkReplay(*value,variant)) return false;
    const auto preview=planFor(*value,fixed ? 2 : 1), exported=planFor(*value,fixed ? 2 : 1,true);
    if (!preview || !exported) return false;
    const auto first=draw(*preview,0),changed=draw(*preview,1),silent=draw(*preview,2);
    const auto offline=draw(*exported,1),sought=draw(*preview,0,1.0);
    const auto diagnostic=draw(*preview,0,3.25,true,Output::ObjectId);
    const auto firstEmission=draw(*preview,0,3.25,false,Output::Emission);
    const auto changedEmission=draw(*preview,1,3.25,false,Output::Emission);
    if (!first || !changed || !silent || !offline || !sought || !diagnostic
        || !firstEmission || !changedEmission) return false;
    const auto a=readPixels(first->frame),b=readPixels(changed->frame);
    if (a.size()!=extent*extent*4 || a==b || a==readPixels(silent->frame)
        || b!=readPixels(offline->frame) || a!=readPixels(sought->frame)
        || readPixels(firstEmission->frame)==readPixels(changedEmission->frame)
        || first->drawnInstanceIds!=std::vector<StableId>{3,4,5,6}
        || first->drawnInstanceIds!=changed->drawnInstanceIds
        || first->drawnInstanceIds!=silent->drawnInstanceIds
        || first->drawnInstanceIds!=offline->drawnInstanceIds
        || first->drawnInstanceIds!=sought->drawnInstanceIds
        || first->stats.vertexBytes!=4*sizeof(HarmonicMIDI::grid::SceneVertex)
        || first->stats.ordinaryDrawCount!=4 || first->stats.instancedDrawCount!=0)
      return reject("Composed spectrum appearance lost causal Color/Emission, stable bands, shared mesh or preview/export/seek parity");
    if (!checkRaw(*first,0) || !checkRaw(*changed,1) || !checkRaw(*silent,2)) return false;
    for (const auto output : {Output::ObjectId,Output::MaterialId,Output::Emission,Output::Mask}) {
      arbitgpu::NativeRawPassPixels before,after,exportRaw,seekRaw,diagnosticRaw;
      if (!first->frame->readRawPass(output,before,error)
          || !changed->frame->readRawPass(output,after,error)
          || !offline->frame->readRawPass(output,exportRaw,error)
          || !sought->frame->readRawPass(output,seekRaw,error)
          || !diagnostic->frame->readRawPass(output,diagnosticRaw,error)) return false;
      if (after.format!=exportRaw.format || after.bytes!=exportRaw.bytes
          || before.format!=seekRaw.format || before.bytes!=seekRaw.bytes
          || before.format!=diagnosticRaw.format || before.bytes!=diagnosticRaw.bytes
          || (fixed && output==Output::ObjectId && before.bytes!=after.bytes))
        return reject("Spectrum appearance raw attachments lost parity or were replaced by RGB identity diagnostics");
    }
    if (fixed) fixedPixels=a;
    SpectrumEvaluation missing; missing.liveFrameAvailable=false;
    if (executeNativeGeometry(backend,*preview,extent,extent,error,false,{},&missing))
      return reject("Composed spectrum appearance admitted missing shared frame analysis");
    if (!fixed) continue;
    auto selected=*value;
    for (auto& binding : selected.spectrumFields) {
      binding.source=spectrum::Source::group; binding.sourceTrackId=17;
    }
    const auto groupPlan=planFor(selected,3);
    if (!groupPlan) return false;
    SpectrumEvaluation group; group.timeSeconds=3.25; group.bands=spectrumDomainBands();
    group.featuresAt=[](double) { return spectrumDomainBands(); };
    group.sourceFeaturesAt=[](spectrum::Source source,std::int32_t id,double,auto& bands) {
      if (source!=spectrum::Source::group || id!=17) return false;
      bands=spectrumDomainBands(true); return true;
    };
    const auto groupDraw=executeNativeGeometry(backend,*groupPlan,extent,extent,error,false,{},&group,{},3.25);
    if (!groupDraw || readPixels(groupDraw->frame)!=b)
      return reject("Composed spectrum appearance substituted master analysis for the selected group");
    group.sourceFeaturesAt={};
    if (executeNativeGeometry(backend,*groupPlan,extent,extent,error,false,{},&group,{},3.25))
      return reject("Composed spectrum appearance admitted absent selected-source analysis");
    selected.spectrumFields.back().attackSeconds=0.1f;
    const auto followed=planFor(selected,4);
    if (!followed) return false;
    group.sourceFeaturesAt=[](spectrum::Source source,std::int32_t id,double time,auto& bands) {
      if (source!=spectrum::Source::group || id!=17 || time<3) return false;
      bands=spectrumDomainBands(true); return true;
    };
    if (executeNativeGeometry(backend,*followed,extent,extent,error,false,{},&group,{},3.25))
      return reject("Composed spectrum appearance admitted missing selected-source follower history");
  }
  std::uint64_t revision=5;
  for (const auto controls : {std::array<double,3>{0.8,0.25,0.25},
       std::array<double,3>{0.5,0.75,0.25},std::array<double,3>{0.5,0.25,0.8}}) {
    const auto edited=spectrumInstancerAppearanceFixture(error,true,controls[0],controls[1],controls[2]);
    const auto plan=edited ? planFor(*edited,revision++) : std::nullopt;
    const auto rendered=plan ? draw(*plan,0) : std::nullopt;
    if (!rendered || readPixels(rendered->frame)==fixedPixels)
      return reject("Authored spectrum multiply, mix or comparison edits did not change native appearance");
  }
  const auto value=spectrumInstancerAppearanceFixture(error,true);
  if (!value) return false;
  const auto contract=withAttributeContract(appearanceInstanceContract(),*value);
  ResourceLimits limits; limits.maxBufferBytes=1;
  if (admitValue(*value,contract,limits,error))
    return reject("Composed spectrum appearance exceeded its memory admission bound");
  auto hostile=*value; hostile.spectrumFields[1].gain=1001;
  if (admitValue(hostile,contract,error)) return reject("Composed spectrum appearance admitted an out-of-bound gain");
  hostile=*value; hostile.spectrumFields[1].firstBand=64;
  if (admitValue(hostile,contract,error)) return reject("Composed spectrum appearance admitted an out-of-bound band");
  hostile=*value; hostile.spectrumFields[1].attackSeconds=2.1f;
  if (admitValue(hostile,contract,error)) return reject("Composed spectrum appearance admitted an out-of-bound follower time");
  hostile=*value; hostile.spectrumFields[1].coordinates.pop_back();
  if (admitValue(hostile,contract,error)) return reject("Composed spectrum appearance admitted a truncated field cardinality");
  hostile=*value; hostile.spectrumFields[1].sourceGeometryStableId=9810;
  if (admitValue(hostile,contract,error)) return reject("Composed spectrum appearance admitted a foreign instance field source");
  const auto excessive=spectrumInstancerAppearanceFixture(error,true,2);
  const auto excessivePlan=excessive ? planFor(*excessive,revision) : std::nullopt;
  if (!excessivePlan) return false;
  if (draw(*excessivePlan,0)) return reject("Composed spectrum appearance published opacity or material scalars above one");
  error.clear(); return true;
}
