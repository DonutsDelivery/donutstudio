#pragma once

#include "instance_material_fields_fixture.h"
#include "spectrum_instancer_fixture.h"
#include "../../shared/generated/SurfaceMaterialStarterPrograms.h"
#include "../../shared/GeometrySurfaceMaterial.h"
#include "../src/fixture_scene_renderer.h"
#include "surface_domain_field_fixture.h"
#include "geometry_scene_composition_fixture.h"

inline surfacematerialbinding::ImportedSceneMaterialRequest reactiveSurfaceRequest(bool time = false) {
  using namespace surfacematerial;
  surfacematerialbinding::ImportedSceneMaterialRequest request;
  request.scene = {1}; request.binding.object = {1};
  request.sceneRevision = request.structuralRevision = request.evaluationRevision = request.programRevision = 1;
  request.binding.targetKind = surfacematerialbinding::BindingTargetKind::ObjectOverride;
  request.binding.surfaceMaterialRevision = 1;
  request.program = surfacematerialstarterfixture::kPrograms[0].program();
  const auto output = [&](OutputSemantic semantic) -> Operation& {
    const auto id = request.program.outputs[static_cast<std::size_t>(semantic)];
    return *std::find_if(request.program.operations.begin(), request.program.operations.end(),
        [&](const auto& operation) { return operation.id == id; });
  };
  output(OutputSemantic::MaterialId).unsignedLiteral = 1;
  output(OutputSemantic::BaseColor).literal = {0.5f, 0.4f, 0.3f, 0};
  output(OutputSemantic::Transmission).literal = {0.2f, 0, 0, 0};
  output(OutputSemantic::Ior).literal = {1.7f, 0, 0, 0};
  output(OutputSemantic::Clearcoat).literal = {0.3f, 0, 0, 0};
  if (time) {
    const auto append = [&](OperationKind kind, ValueType type, std::initializer_list<ValueId> inputs,
                            std::array<float, 4> literal = {}, InputSemantic semantic = InputSemantic::Invalid) {
      Operation operation;
      operation.id = static_cast<ValueId>(request.program.operations.size() + 1);
      operation.kind = kind; operation.resultType = type;
      operation.inputCount = static_cast<std::uint8_t>(inputs.size());
      std::copy(inputs.begin(), inputs.end(), operation.inputs.begin());
      operation.literal = literal; operation.semantic = semantic;
      request.program.operations.push_back(operation); return operation.id;
    };
    const auto end = append(OperationKind::FloatConstant, ValueType::Vec3, {}, {0.05f, 0.8f, 0.1f, 0});
    const auto lower = append(OperationKind::FloatConstant, ValueType::Scalar, {}, {});
    const auto upper = append(OperationKind::FloatConstant, ValueType::Scalar, {}, {1,0,0,0});
    const auto clock = append(OperationKind::Input, ValueType::Scalar, {}, {}, InputSemantic::Time);
    const auto clamp = append(OperationKind::Clamp, ValueType::Scalar, {clock,lower,upper});
    const auto mix = append(OperationKind::Mix, ValueType::Vec3, {request.program.outputs[0],end,clamp});
    request.program.outputs[0] = mix;
  }
  std::string error;
  const auto admitted = admit(request.program, error);
  if (admitted) request.binding.surfaceMaterialDigest = admitted->structuralDigest();
  return request;
}

inline std::optional<surfacematerialfield::Binding> reactiveSurfaceField(std::uint32_t target, std::string& error, bool score = false) {
  using namespace videowire::geometry;
  const auto contract=constructedFieldContract(Domain::vertex,ValueType::floatValue);
  FieldOperationSettings settings; settings.domain=Domain::vertex; settings.valueType=score ? ValueType::vector : ValueType::floatValue;
  settings.values={0,0,0,0,4,0,0,0};
  const auto coordinates=lowerConstructedField(9701,OperationCode::fieldConstant,settings,nullptr,nullptr,nullptr,
      constructedFieldContract(Domain::vertex,settings.valueType),error);
  settings.valueType=ValueType::floatValue;
  settings.mode=0; settings.values={4,0,0,0,0.15,0.1,0,0};
  if (score) { settings.mode=1; settings.values={4,0.7,0.1,0.25,9798,0,0,0}; }
  const auto field=coordinates ? lowerConstructedField(9702,score ? OperationCode::fieldScoreSample : OperationCode::fieldTimeline,settings,
      &*coordinates,nullptr,nullptr,contract,error) : std::nullopt;
  const auto admitted=field ? admitValue(*field,contract,error) : std::nullopt;
  if (!admitted) return std::nullopt;
  surfacematerialfield::Binding binding;
  binding.sample.materialNode=9799; binding.sample.target=target;
  binding.sample.gain=target==8 ? -1 : 1;
  binding.sample.fieldPlan=encodeLoweredPlanText(lowerRuntimePlan(contract,*admitted));
  if (!surfacematerialfield::valid(binding,error)) return std::nullopt;
  return binding;
}

inline surfacematerialbinding::ImportedSceneMaterialRequest tableSurfaceRequest(std::uint32_t id, bool green) {
  auto request=reactiveSurfaceRequest();
  for (auto& operation : request.program.operations) {
    if (operation.id==request.program.outputs[static_cast<std::size_t>(surfacematerial::OutputSemantic::MaterialId)])
      operation.unsignedLiteral=id;
    if (operation.id==request.program.outputs[static_cast<std::size_t>(surfacematerial::OutputSemantic::BaseColor)])
      operation.literal=green ? std::array<float,4>{0.05f,0.8f,0.05f,0} : std::array<float,4>{0.8f,0.05f,0.05f,0};
  }
  std::string error;
  const auto admitted=surfacematerial::admit(request.program,error);
  request.binding.surfaceMaterialDigest=admitted ? admitted->structuralDigest() : std::string{};
  return request;
}

inline std::optional<visualimportedscenerender::Request> retainedSurfaceFixture(std::string& error) {
  const auto scene=geometryCompositionFixture(error);
  const auto index=surfaceDomainFieldFixture(videowire::geometry::Domain::instance,error);
  const auto time=reactiveSurfaceField(6,error);
  if (!scene || !index || !time) return std::nullopt;
  visualimportedscenerender::Request request;
  request.sceneSnapshot=scene; request.sourceStableId=scene->id.value; request.renderStableId=99;
  request.structuralRevision=request.evaluationRevision=1;
  for (std::size_t i=0;i<scene->objectCount;++i) {
    const auto& object=scene->objects[i];
    if (object.indexCount==0) continue;
    geometrysurfacematerial::ObjectProgram entry;
    entry.program.sourceMaterial=i==0 ? 9901 : 9902;
    entry.program.material=tableSurfaceRequest(object.material.value,i!=0);
    entry.program.material.scene=scene->id; entry.program.material.sceneSnapshot=scene;
    entry.program.material.binding.object=object.id;
    entry.program.fields={i==0 ? *time : *index};
    if (i!=0) {
      entry.program.fields[0].objectIndex=true;
      entry.program.fields[0].sample.reduction=materialfield::Reduction::index;
      entry.instanceIndex=static_cast<std::uint32_t>(i-2);
      entry.appearance.color[3]=0.8f;
    }
    request.surfacePrograms.push_back(std::move(entry));
  }
  std::sort(request.surfacePrograms.begin(),request.surfacePrograms.end(),[](const auto& a,const auto& b) {
    return a.program.material.binding.object.value<b.program.material.binding.object.value;
  });
  if (!visualimportedscenerender::valid(request)) { error="Retained Surface fixture identity is invalid"; return std::nullopt; }
  return request;
}

template <typename ReadPixels>
bool verifyNativeRetainedSurfacePrograms(arbitgpu::NativeFixtureSceneBackend& backend,
    ReadPixels readPixels, std::string& error) {
  const auto request=retainedSurfaceFixture(error);
  if (!request) return false;
  visualimportedscenerender::Request reopened;
  const auto wire=visualimportedscenerender::encode(*request);
  if (!visualimportedscenerender::decode(wire,reopened)) { error="Retained Surface reopen failed"; return false; }
  const auto target=backend.info().backend=="metal" ? videohelper::materialprogram::BackendTarget::Metal
      : videohelper::materialprogram::BackendTarget::OpenGl;
  videorender::fixture3d::FixtureSceneRenderer renderer(backend);
  videowire::geometry::RuntimeFieldEvaluation evaluation;
  const auto first=videorender::fixture3d::admitSurfaceMaterialCollection(reopened.sceneSnapshot,reopened.surfacePrograms,evaluation,target,error);
  evaluation.timelineSeconds=3;
  const auto later=videorender::fixture3d::admitSurfaceMaterialCollection(reopened.sceneSnapshot,reopened.surfacePrograms,evaluation,target,error);
  evaluation.timelineSeconds=0;
  const auto sought=videorender::fixture3d::admitSurfaceMaterialCollection(reopened.sceneSnapshot,reopened.surfacePrograms,evaluation,target,error);
  if (!first || !later || !sought || first->bindingDigest()==later->bindingDigest()
      || first->bindingDigest()!=sought->bindingDigest()) { error="Retained Surface sampled cache identity is not deterministic"; return false; }
  videorender::fixture3d::RenderedFrame a,b,c,d,cached;
  if (!renderer.renderPreview(reopened.sceneSnapshot,first,{128,128},"native-gpu",a,error)
      || !renderer.renderPreview(reopened.sceneSnapshot,first,{128,128},"native-gpu",cached,error)
      || !cached.stats.reusedMaterialProgram
      || !renderer.renderPreview(reopened.sceneSnapshot,later,{128,128},"native-gpu",b,error)
      || !renderer.renderExport(reopened.sceneSnapshot,later,{128,128},"native-gpu",c,error)
      || !renderer.renderPreview(reopened.sceneSnapshot,sought,{128,128},"native-gpu",d,error)) return false;
  if (readPixels(a.nativeFrame).empty() || readPixels(a.nativeFrame)==readPixels(b.nativeFrame)
      || readPixels(b.nativeFrame)!=readPixels(c.nativeFrame) || readPixels(a.nativeFrame)!=readPixels(d.nativeFrame)
      || a.stats.ordinaryDrawCount!=5) { error="Retained object Surfaces lost draw selection, Field replay, reopen or preview/export parity"; return false; }
  using videowire::geometry::Domain;
  for (const auto domain : {Domain::vertex,Domain::face,Domain::point,Domain::curvePoint,Domain::spline,Domain::instance}) {
    auto field=surfaceDomainFieldFixture(domain,error);
    if (!field) return false;
    field->sample.target=7;
    auto programs=reopened.surfacePrograms; programs.front().program.fields={*field};
    const auto binding=videorender::fixture3d::admitSurfaceMaterialCollection(reopened.sceneSnapshot,programs,evaluation,target,error);
    if (!binding || !renderer.renderPreview(reopened.sceneSnapshot,binding,{128,128},"native-gpu",b,error)
        || !renderer.renderExport(reopened.sceneSnapshot,binding,{128,128},"native-gpu",c,error)) return false;
    if (readPixels(b.nativeFrame)==readPixels(a.nativeFrame) || readPixels(b.nativeFrame)!=readPixels(c.nativeFrame)) {
      error="Geometry-backed Material Field domain did not reach native preview/export pixels"; return false;
    }
  }
  // A collection may start at a non-first draw. Unbound objects keep PBR.
  auto mixed=reopened.surfacePrograms;
  mixed.erase(mixed.begin());
  const auto selected=videorender::fixture3d::admitSurfaceMaterialCollection(reopened.sceneSnapshot,mixed,evaluation,target,error);
  if (!selected || !renderer.renderPreview(reopened.sceneSnapshot,selected,{128,128},"native-gpu",cached,error)
      || readPixels(cached.nativeFrame)==readPixels(a.nativeFrame)) {
    error="Retained Surface collection leaked onto an unbound ordinary material"; return false;
  }
  auto invalid=reopened.surfacePrograms;
  invalid.front().program.material.binding.object={11};
  if (videorender::fixture3d::admitSurfaceMaterialCollection(reopened.sceneSnapshot,invalid,evaluation,target,error)) {
    error="Transform-only parent admitted a Surface draw binding"; return false;
  }
  error.clear(); return true;
}

template <typename ReadPixels>
bool verifyNativeSurfaceTableAndFields(arbitgpu::NativeFixtureSceneBackend& backend,
    ReadPixels readPixels, std::string& error) {
  using namespace videowire::geometry;
  using namespace videohelper::geometry;
  if (!verifyNativeRetainedSurfacePrograms(backend,readPixels,error)) return false;
  NativeGeometryCoreCapabilitySource capabilities(backend);
  for (std::size_t target=0;target<=std::size(surfacematerialfield::kTargets);++target) {
    const bool table=target==std::size(surfacematerialfield::kTargets);
    const auto value=table ? instanceMaterialFieldsFixture("materialIndex",true,false,error) : instanceAppearanceFixture(0,error);
    if (!value) return false;
    const auto contract=withAttributeContract(appearanceInstanceContract(),*value);
    const auto admitted=admitValue(*value,contract,error);
    if (!admitted) return false;
    geometrysurfacematerial::ProgramSet programs;
    if (table) programs={{9521,tableSurfaceRequest(17,false),{}},{9522,tableSurfaceRequest(29,true),{}}};
    else {
      const auto field=reactiveSurfaceField(target,error);
      if (!field) return false;
      programs={{0,reactiveSurfaceRequest(),{*field}}};
    }
    const auto wire=geometrysurfacematerial::encodeSet(programs);
    if (wire.empty()) { error="Surface table/Field fixture did not encode"; return false; }
    GeometryCorePlanRuntime runtime(capabilities);
    const auto bytes=lowerRuntimePlan(contract,*admitted,wire);
    const auto preview=runtime.admitPreview({1,1,1,9700,1,PlanUse::preview},bytes,{}, {},error);
    const auto exported=runtime.admitExport({1,1,1,9700,1,PlanUse::exportRender},bytes,{}, {},error);
    if (!preview || !exported) return false;
    const auto draw=[&](const RuntimeAdmission& plan,double seconds) {
      return executeNativeGeometry(backend,plan,64,64,error,false,{},nullptr,{},seconds);
    };
    const auto first=draw(*preview,0),later=draw(*preview,3),offline=draw(*exported,3),sought=draw(*preview,0);
    if (!first || !later || !offline || !sought) return false;
    const auto a=readPixels(first->frame),b=readPixels(later->frame);
    if (a.empty() || a==b || b!=readPixels(offline->frame) || a!=readPixels(sought->frame)
        || first->drawnInstanceIds!=later->drawnInstanceIds || first->stats.materialProgramUploadCount==0
        || first->stats.vertexBytes!=4*sizeof(HarmonicMIDI::grid::SceneVertex)) {
      error=std::string("Native Surface ")+(table ? "table selection" : surfacematerialfield::kTargets[target])
          +" must change pixels and reproduce export/seeks on shared vertices"; return false;
    }
    if (table) {
      const auto mixed=draw(*preview,1.5);
      if (!mixed || readPixels(mixed->frame)==a || readPixels(mixed->frame)==b) {
        error="Surface table must select distinct programs for the exact mixed instance batch"; return false;
      }
    }
  }
  const auto field=reactiveSurfaceField(5,error,true);
  const auto value=instanceAppearanceFixture(0,error);
  if (!field || !value) return false;
  const auto contract=appearanceInstanceContract();
  const auto admitted=admitValue(*value,contract,error);
  if (!admitted) return false;
  const geometrysurfacematerial::ProgramSet programs{{0,reactiveSurfaceRequest(),{*field}}};
  if (!geometrysurfacematerial::usesScore(programs)) { error="Surface Score Field did not retain its canonical score dependency"; return false; }
  GeometryCorePlanRuntime runtime(capabilities);
  const auto bytes=lowerRuntimePlan(contract,*admitted,geometrysurfacematerial::encodeSet(programs));
  const auto preview=runtime.admitPreview({1,1,1,9703,1,PlanUse::preview},bytes,{}, {},error);
  const auto exported=runtime.admitExport({1,1,1,9703,1,PlanUse::exportRender},bytes,{}, {},error);
  if (!preview || !exported) return false;
  arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
  inputs.canonicalBlockCFrame=scoreFieldFrame();
  const auto notes=executeNativeGeometry(backend,*preview,64,64,error,false,inputs);
  inputs.canonicalBlockCFrame=scoreFieldFrame(true);
  const auto offline=executeNativeGeometry(backend,*exported,64,64,error,false,inputs);
  inputs.canonicalBlockCFrame=scoreFieldFrame(false,true);
  const auto empty=executeNativeGeometry(backend,*preview,64,64,error,false,inputs);
  if (!notes || !offline || !empty || readPixels(notes->frame)!=readPixels(offline->frame)
      || readPixels(notes->frame)==readPixels(empty->frame)) {
    if (error.empty()) error="Surface Score Field must use canonical note frames in preview and export";
    return false;
  }
  return true;
}

template <typename ReadPixels>
bool verifyNativeReactiveSurface(arbitgpu::NativeFixtureSceneBackend& backend,
    ReadPixels readPixels, std::string& error) {
  using namespace videowire::geometry;
  using namespace videohelper::geometry;
  if (!verifyNativeSurfaceTableAndFields(backend,readPixels,error)) return false;
  NativeGeometryCoreCapabilitySource capabilities(backend);
  for (const auto* property : {"color", "emission", "opacity", "metallic", "roughness", "time"}) {
    const bool time = std::string(property) == "time";
    const auto value = time ? instanceAppearanceFixture(0,error)
        : instanceMaterialFieldsFixture(property,false,false,error);
    if (!value) return false;
    const auto contract = withAttributeContract(appearanceInstanceContract(), *value);
    const auto admitted = admitValue(*value, contract, error);
    if (!admitted) return false;
    const auto surface = geometrysurfacematerial::encode(reactiveSurfaceRequest(time));
    if (surface.empty()) { error = "Reactive Surface fixture failed canonical IR admission"; return false; }
    GeometryCorePlanRuntime runtime(capabilities);
    const auto bytes = lowerRuntimePlan(contract, *admitted, surface);
    const auto preview = runtime.admitPreview({1,1,1,9601,1,PlanUse::preview}, bytes, {}, {}, error);
    const auto exported = runtime.admitExport({1,1,1,9601,1,PlanUse::exportRender}, bytes, {}, {}, error);
    if (!preview || !exported) return false;
    const auto draw = [&](const RuntimeAdmission& plan, double seconds) {
      return executeNativeGeometry(backend,plan,64,64,error,false,{},nullptr,{},seconds);
    };
    const auto first = draw(*preview,0), later = draw(*preview,3), offline = draw(*exported,3), sought = draw(*preview,0);
    if (!first || !later || !offline || !sought) return false;
    const auto a = readPixels(first->frame), b = readPixels(later->frame);
    if (a.empty() || a == b || b != readPixels(offline->frame) || a != readPixels(sought->frame)
        || first->drawnInstanceIds != later->drawnInstanceIds || first->stats.materialProgramUploadCount == 0
        || first->stats.vertexBytes != 4 * sizeof(HarmonicMIDI::grid::SceneVertex)) {
      error = std::string("Surface ") + property + " must replay exact independent instance pixels in preview, export and seeks";
      return false;
    }
  }
  const auto score = instanceMaterialFieldsFixture("emission",false,true,error);
  if (!score) return false;
  const auto contract = withAttributeContract(appearanceInstanceContract(), *score);
  const auto admitted = admitValue(*score,contract,error);
  if (!admitted) return false;
  GeometryCorePlanRuntime runtime(capabilities);
  const auto bytes = lowerRuntimePlan(contract,*admitted,geometrysurfacematerial::encode(reactiveSurfaceRequest()));
  const auto preview = runtime.admitPreview({1,1,1,9602,1,PlanUse::preview},bytes,{}, {},error);
  const auto exported = runtime.admitExport({1,1,1,9602,1,PlanUse::exportRender},bytes,{}, {},error);
  if (!preview || !exported) return false;
  arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
  inputs.canonicalBlockCFrame = scoreFieldFrame();
  const auto notes = executeNativeGeometry(backend,*preview,64,64,error,false,inputs);
  inputs.canonicalBlockCFrame = scoreFieldFrame(true);
  const auto offline = executeNativeGeometry(backend,*exported,64,64,error,false,inputs);
  inputs.canonicalBlockCFrame = scoreFieldFrame(false,true);
  const auto empty = executeNativeGeometry(backend,*preview,64,64,error,false,inputs);
  if (!notes || !offline || !empty) return false;
  if (readPixels(notes->frame) != readPixels(offline->frame) || readPixels(notes->frame) == readPixels(empty->frame)) {
    error = "Surface instance emission must consume the canonical score frame in preview and export"; return false;
  }
  return verifyNativeSpectrumInstancer(backend,readPixels,error,
      geometrysurfacematerial::encode(reactiveSurfaceRequest()));
}

template <typename ReadPixels>
bool verifyNativeObjectSurfacePrograms(arbitgpu::NativeFixtureSceneBackend& backend, ReadPixels readPixels,
    const std::shared_ptr<HarmonicMIDI::grid::Visual3DScene>& scene, std::string& error) {
  using namespace arbitgpu;
  if (!scene || scene->objectCount != 2) { error = "Surface object fixture requires two ordinary objects"; return false; }
  const auto target = backend.info().backend == "metal" ? videohelper::materialprogram::BackendTarget::Metal
      : videohelper::materialprogram::BackendTarget::OpenGl;
  const auto bind = [&](std::size_t index) {
    auto request = reactiveSurfaceRequest(index == 1);
    request.scene = scene->id; request.sceneSnapshot = scene; request.binding.object = scene->objects[index].id;
    return videorender::fixture3d::admitSurfaceMaterialBinding(scene,request,target,error);
  };
  const auto first = bind(0), second = bind(1);
  if (!first || !second) return false;
  auto program = std::make_shared<NativeFixtureSurfaceMaterialProgram>(*first->nativeProgram());
  program->objectPrograms.push_back(second->nativeProgram());
  auto prepared = backend.prepare(scene,program);
  if (!prepared.prepared) { error = prepared.error; return false; }
  NativeFixtureSceneRuntimeInputs inputs; inputs.timeSeconds = 1;
  const auto both = backend.render(scene,prepared.resources,64,64,inputs);
  const auto secondOnly = backend.prepare(scene,second->nativeProgram());
  const auto selected = secondOnly.prepared ? backend.render(scene,secondOnly.resources,64,64,inputs) : NativeFixtureSceneSubmission{};
  if (!both.rendered || !selected.rendered || readPixels(both.frame) == readPixels(selected.frame)) {
    error = "Exact object Surface programs must coexist with ordinary materials on unbound objects"; return false;
  }
  auto reordered = std::make_shared<HarmonicMIDI::grid::Visual3DScene>(*scene);
  std::swap(reordered->objects[0],reordered->objects[1]);
  const auto reorderedResources = backend.prepare(reordered,program);
  const auto reorderedDraw = reorderedResources.prepared
      ? backend.render(reordered,reorderedResources.resources,64,64,inputs) : NativeFixtureSceneSubmission{};
  if (!reorderedDraw.rendered || readPixels(reorderedDraw.frame) != readPixels(both.frame)) {
    error = "Surface program selection must follow object identity, not draw-array position"; return false;
  }
  inputs.timeSeconds = std::numeric_limits<float>::quiet_NaN();
  if (backend.render(scene,prepared.resources,64,64,inputs).rendered) {
    error = "A time-dependent secondary Surface program requires finite frame time"; return false;
  }
  program->objectPrograms.push_back(second->nativeProgram());
  if (backend.prepare(scene,program).prepared) {
    error = "Duplicate Surface object bindings must fail native admission"; return false;
  }
  program->objectPrograms.clear();
  program->parameters.baseColorMetallic[0] = 0;
  inputs.timeSeconds = 1;
  const auto immutable = backend.render(scene,prepared.resources,64,64,inputs);
  if (!immutable.rendered || readPixels(immutable.frame) != readPixels(both.frame)) {
    error = "Prepared Surface collections must own immutable native program snapshots"; return false;
  }
  return true;
}
