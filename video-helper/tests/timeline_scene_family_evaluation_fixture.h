#pragma once

#include "../src/composed_timeline_geometry.h"
#include "water_surface_replay_fixture.h"

namespace timelinefamilyfixture {
using namespace videowire::geometry;
using namespace videohelper::geometry;
using namespace HarmonicMIDI::grid;

class Backend final : public BackendCapabilitySource {
public:
  BackendCapabilities geometryCoreCapabilities() const override {
    BackendCapabilities result;
    result.backendIdentity="family-source-fixture"; result.deviceIdentity="owned-fixture-device";
    result.nativeGpuAvailable=result.geometryCoreExecution=result.immutableSourceBuffers=true;
    result.stableElementIds=result.typedFieldEvaluation=true;
    result.supportedCarriers=carrierBit(CarrierKind::geometry3D);
    return result;
  }
};

struct Source final {
  std::shared_ptr<Visual3DScene> scene;
  std::vector<TimelineSceneBinding> bindings;
};

inline std::optional<Source> source(std::string& error,double secondGain=0.5,bool water=false,
    double scoreScale=1,double secondAudioGain=0,double mass=0) {
  PortContract geometry; geometry.carrier=CarrierKind::geometry3D;
  geometry.maxVertices=4096; geometry.maxIndices=12288;
  PortContract vector; vector.carrier=CarrierKind::field; vector.fieldDomain=Domain::vertex;
  vector.fieldValueType=ValueType::vector; vector.fieldInterpolation=Interpolation::constant;
  vector.maxFieldElements=4096;
  auto scalar=vector; scalar.fieldValueType=ValueType::floatValue;
  const auto mesh=[&](StableId id,bool first)->std::optional<AdmittedValue> {
    auto grid=lowerGrid(id,id,1,2,2,water && !first ? 2 : 1,geometry,error);
    if (!grid) return std::nullopt;
    if (water) {
      auto lowered=lowerWaterSurfaceSimulation(*grid,id+1,id+2,1,2,(first ? 0.2 : 0.4)*scoreScale,
          first ? 0 : secondAudioGain,geometry,error,mass);
      return lowered ? admitValue(*lowered,geometry,error) : std::nullopt;
    }
    FieldOperationSettings settings; settings.domain=Domain::vertex; settings.valueType=ValueType::floatValue;
    std::optional<ValueDescriptor> seed;
    if (first) {
      auto position=lowerElementField(*grid,id+1,OperationCode::positionField,vector,error);
      settings.mode=3;
      seed=position ? lowerConstructedField(id+2,OperationCode::fieldVectorMeasure,settings,&*position,
          nullptr,nullptr,scalar,error) : std::nullopt;
      settings.mode=2; settings.values={4,0,0.25,0,0.25,0,0,0};
    } else {
      settings.values={0,0,0,0,4,0,0,0};
      seed=lowerConstructedField(id+1,OperationCode::fieldConstant,settings,nullptr,nullptr,nullptr,scalar,error);
      settings.mode=0; settings.values={3,0,0,0,secondGain,0,0,0};
    }
    auto time=seed ? lowerConstructedField(id+3,OperationCode::fieldTimeline,settings,&*seed,
        nullptr,nullptr,scalar,error) : std::nullopt;
    settings.valueType=ValueType::vector; settings.mode=1;
    settings.values=first ? std::array<double,8>{0,0,1,0,0,0,0,0} : std::array<double,8>{1,0,0,0,0,0,0,0};
    auto offset=time ? lowerConstructedField(id+4,OperationCode::fieldComposeVector,settings,&*time,
        nullptr,nullptr,vector,error) : std::nullopt;
    auto lowered=offset ? lowerVectorDisplacement(*grid,*offset,id+5,1,geometry,error) : std::nullopt;
    return lowered ? admitValue(*lowered,geometry,error) : std::nullopt;
  };
  auto first=mesh(17000,true),second=mesh(17100,false);
  if (!first || !second) return std::nullopt;
  Source result; result.scene=std::make_shared<Visual3DScene>();
  auto& scene=*result.scene;
  scene.id={7}; scene.activeCamera={50}; scene.ambientColor={0.01f,0.02f,0.03f};
  scene.cameras[0].id={50}; scene.cameras[0].transform.translation.z=4; scene.cameraCount=1;
  scene.lights[0].id={40}; scene.lights[0].intensity=1.5f; scene.lightCount=1;
  SceneObjectRecord object; object.id={10}; object.parent={100}; object.transform.translation.x=-1;
  SceneMaterialRecord material; material.id={20}; material.baseColor={1,0.03f,0.02f};
  if (!appendGeometryToScene(*first,object,material,scene,error)) return std::nullopt;
  object.id={11}; object.transform.translation.x=1;
  material.id={21}; material.baseColor={0.02f,0.03f,1};
  if (!appendGeometryToScene(*second,object,material,scene,error)) return std::nullopt;
  auto cube=lowerMeshGenerator(17200,17200,1,OperationCode::cube,{0.7f,0.7f,0.7f,0},geometry,error);
  auto cubeValue=cube ? admitValue(*cube,geometry,error) : std::nullopt;
  object.id={12}; object.transform.translation.y=1;
  material.id={22}; material.baseColor={0.1f,0.8f,0.2f};
  if (!cubeValue || !appendGeometryToScene(*cubeValue,object,material,scene,error)) return std::nullopt;
  scene.objects[scene.objectCount].id={100};
  scene.objects[scene.objectCount++].transform.translation.y=0.2f;
  scene.textures[0].id={88}; scene.textures[0].width=scene.textures[0].height=1;
  scene.textureCount=1; scene.textureTexels.push_back({17,23,41,255}); scene.textureTexelCount=1;
  scene.materials[0].baseColorTexture={88};
  // Include declared fields outside active ranges in the exact frozen anchor.
  scene.cameras.back().farPlane=77;
  if (!validateVisual3DScene(scene).valid()) return std::nullopt;
  result.bindings={{{10},encodeLoweredPlanText(lowerRuntimePlan(geometry,*first))},
                   {{11},encodeLoweredPlanText(lowerRuntimePlan(geometry,*second))}};
  return result;
}

// Literal positions plus independent triangle normal slope. The untouched
// authored records are compared with the complete bit-exact scene comparator.
inline bool literal(const Visual3DScene& frame,const Visual3DScene& authored,
    double leftZ,double rightZ,double secondX) {
  if (frame.vertexCount!=32 || frame.indexCount!=48 || frame.objectCount!=4) return false;
  const std::array<float,4> xs{-0.5f,0.5f,-0.5f,0.5f},ys{-0.5f,-0.5f,0.5f,0.5f};
  const auto dz=double(static_cast<float>(rightZ))-double(static_cast<float>(leftZ));
  const auto length=std::sqrt(dz*dz+1);
  auto expected=std::make_shared<Visual3DScene>(authored);
  for (std::size_t i=0;i<4;++i) {
    const auto& a=frame.vertices[i]; const auto& b=frame.vertices[i+4];
    if (a.position.x!=xs[i] || a.position.y!=ys[i]
        || std::abs(a.position.z-(i%2 ? rightZ : leftZ))>0.000001
        || std::abs(a.normal.x+dz/length)>0.000001 || a.normal.y!=0
        || std::abs(a.normal.z-1/length)>0.000001
        || std::abs(b.position.x-(xs[i]+secondX))>0.000001 || b.position.y!=ys[i] || b.position.z!=0
        || b.normal.x!=0 || b.normal.y!=0 || b.normal.z!=1) return false;
    for (const auto index : {i,i+4}) {
      const auto& actual=frame.vertices[index]; const auto& base=authored.vertices[index];
      if (actual.uv.x!=base.uv.x || actual.uv.y!=base.uv.y
          || actual.color!=base.color || actual.tangent!=base.tangent) return false;
    }
    expected->vertices[i]=a; expected->vertices[i+4]=b;
  }
  const std::array<std::uint32_t,6> indices{0,1,2,1,3,2};
  for (std::size_t i=0;i<indices.size();++i)
    if (frame.indices[i]!=indices[i] || frame.indices[i+6]!=indices[i]) return false;
  return timelinefamily::sameScene(*expected,frame);
}

// One note at zero has unit MIDI velocity, x=1/2 and the literal
// SplitMix64 identity-1 coordinate y=0.566561575172281. At one completed
// step the Laplacian is zero. At two steps each row has only its opposite
// row as a nonzero-gradient neighbor. These are independent closed forms.
inline bool waterLiteral(const Visual3DScene& frame,const Visual3DScene& authored,int steps) {
  if (frame.vertexCount!=32 || frame.indexCount!=48 || frame.objectCount!=4 || steps<0 || steps>2) return false;
  constexpr double dt=1.0/120,r=59.0/60,y=0.566561575172281;
  auto expected=std::make_shared<Visual3DScene>(authored);
  for (std::size_t child=0;child<2;++child) {
    const double spacing=child ? 2 : 1,gain=child ? 0.4 : 0.2;
    std::array<float,2> z{};
    for (std::size_t row=0;row<2;++row) {
      const double weight=(row ? y : 1-y)/2,other=(row ? 1-y : y)/2;
      const double height=steps==0 ? 0 : steps==1 ? gain*dt*r*weight
          : gain*dt*(r*weight+r*r*weight+r*r*dt*dt*(other-weight)/(spacing*spacing));
      z[row]=static_cast<float>(height);
    }
    const double slope=(double(z[1])-z[0])/spacing,length=std::sqrt(1+slope*slope);
    for (std::size_t i=0;i<4;++i) {
      const auto index=child*4+i;
      const auto& actual=frame.vertices[index]; const auto& base=authored.vertices[index];
      if (actual.position.x!=base.position.x || actual.position.y!=base.position.y
          || std::abs(actual.position.z-z[i/2])>0.000001 || actual.normal.x!=0
          || std::abs(actual.normal.y+slope/length)>0.000001
          || std::abs(actual.normal.z-1/length)>0.000001
          || actual.uv.x!=base.uv.x || actual.uv.y!=base.uv.y
          || actual.color!=base.color || actual.tangent!=base.tangent) return false;
      expected->vertices[index]=actual;
    }
  }
  return timelinefamily::sameScene(*expected,frame);
}

// The same released 9x9 one-binding impact source, with a separately authored
// timed child and an untouched cube/camera/light/texture anchor.
inline std::optional<Source> coupledSource(std::string& error,double siblingGain=0.5,double siblingAxisGain=1) {
  auto baseline=source(error,siblingGain); if (!baseline) return std::nullopt;
  PortContract geometry; geometry.carrier=CarrierKind::geometry3D;
  geometry.maxVertices=256; geometry.maxIndices=384;
  auto grid=lowerGrid(18000,18000,1,9,9,.25f,geometry,error);
  auto water=grid ? lowerWaterSurfaceSimulation(*grid,18001,18002,0,0,0,0,geometry,error,100):std::nullopt;
  auto admitted=water ? admitValue(*water,geometry,error):std::nullopt;
  auto sibling=decodeTimelineScenePlan(baseline->bindings[1],error);
  if (sibling && siblingAxisGain!=1) {
    auto modified=sibling->descriptor();
    for (auto& operation:modified.operations)
      if (operation.code==OperationCode::fieldComposeVector) operation.field.values[0]=siblingAxisGain;
    auto admittedModified=admitValue(modified,geometry,error);
    sibling.reset();
    if (admittedModified) sibling.emplace(std::move(*admittedModified));
    if (sibling) baseline->bindings[1].loweredPlan=encodeLoweredPlanText(lowerRuntimePlan(geometry,*sibling));
  }
  auto cube=lowerMeshGenerator(17200,17200,1,OperationCode::cube,{.7f,.7f,.7f,0},geometry,error);
  auto cubeValue=cube ? admitValue(*cube,geometry,error):std::nullopt;
  if (!admitted || !sibling || !cubeValue) return std::nullopt;
  auto scene=std::make_shared<Visual3DScene>(*baseline->scene);
  scene->objectCount=scene->materialCount=scene->vertexCount=scene->indexCount=0;
  if (!appendGeometryToScene(*admitted,baseline->scene->objects[0],baseline->scene->materials[0],*scene,error)
      || !appendGeometryToScene(*sibling,baseline->scene->objects[1],baseline->scene->materials[1],*scene,error)
      || !appendGeometryToScene(*cubeValue,baseline->scene->objects[2],baseline->scene->materials[2],*scene,error)) return std::nullopt;
  scene->objects[scene->objectCount++]=baseline->scene->objects[3];
  if (!validateVisual3DScene(*scene).valid()) return std::nullopt;
  return Source{scene,{{{10},encodeLoweredPlanText(lowerRuntimePlan(geometry,*admitted))},baseline->bindings[1]}};
}
inline videorender::ParticleParams coupledParameters(const Source& source,
    std::shared_ptr<const videorender::ParticleHistorySource> history,std::string& error) {
  videorender::ParticleParams p;
  p.count=1; p.force=0; p.motionMode=2; p.simulationSpace=1; p.historicalReplay=true;
  p.lifetime=1; p.resetMode=1; p.impulseZ=-4; p.restitution=1; p.colliderThickness=0;
  p.historyClipId=35; p.historyNodeId=90; p.history=std::move(history);
  auto value=decodeTimelineScenePlan(source.bindings[0],error);
  auto binding=std::make_shared<videorender::ParticleGeometryBinding>();
  if (value) binding->mesh=std::make_shared<const ValueDescriptor>(value->descriptor());
  ValueDescriptor emitter; emitter.carrier=CarrierKind::points3D; emitter.stableId=18100;
  emitter.sourceStableId=18100; emitter.sourceRevision=1; emitter.data=PointsData{};
  std::get<PointsData>(emitter.data).points.push_back({{-.4f,-.2f,.12f},18101,1,{0,0,0,1}});
  binding->emitters=std::make_shared<const ValueDescriptor>(std::move(emitter));
  p.geometryBinding=std::move(binding); return p;
}

template <typename Check>
void verifyCoupled(Check check) {
  using namespace videorender;
  std::string error; auto authored=coupledSource(error);
  check(bool(authored),"coupled family fixture authors genuine 9x9 Water and an independent timed sibling: "+error);
  if (!authored) return;
  Backend backend; GeometryCorePlanRuntime runtime(backend,16);
  const PlanOwnerIdentity owner{31,32,33,34,35,PlanUse::preview};
  runtime.reconcileGeneration(31,32,33,error);
  auto family=runtime.admitTimelineScenePreview(owner,*authored->scene,authored->bindings,{},error);
  check(bool(family),"coupled family admits every exclusive child at authored rest: "+error);
  if (!family) return;
  auto audio=waterreplayfixture::audio(2);
  auto history=std::make_shared<ParticleHistorySource>(waterreplayfixture::source());
  auto score=std::make_shared<arbitmod::Score>(*history->score);
  score->notes.push_back(waterreplayfixture::note(81,100)); history->score=score;
  history->audioFrames=&audio; history->audioSampleRate=61440;
  auto parameters=coupledParameters(*authored,history,error);
  constexpr double dt=1.0/120; constexpr float aspect=640.0f/360;
  TimelineSceneFamilyEvaluator evaluator(runtime);
  const auto sample=[&](double seconds,PlanOwnerIdentity requestedOwner={31,32,33,34,35,PlanUse::preview},
      std::shared_ptr<const AdmittedTimelineScenePlans> requestedFamily=nullptr) {
    auto p=parameters; p.historyProjectSeconds=seconds;
    auto frame=waterreplayfixture::frame(*history,seconds,30,requestedOwner.use==PlanUse::exportRender);
    return evaluator.evaluateCoupled(requestedOwner,requestedFamily ? requestedFamily:family,authored->scene,p,
        seconds,seconds,seconds,aspect,frame,{},{},error);
  };
  auto impact=sample(dt),following=sample(2*dt);
  check(impact.scene && impact.replay && following.scene && following.replay
      && impact.replay->solids()->bodies[0].vz>0
      && std::any_of(following.replay->water()->heights.begin(),following.replay->water()->heights.end(),[](double h) { return h<0; }),
      "actual single solver pair rebounds at impact and transfers negative Water displacement on the following step: "+error);
  if (!impact.scene || !following.scene) return;
  const auto literalSibling=[&](const Visual3DScene& scene,double seconds) {
    const auto first=family->children()[1].object().firstVertex;
    for (std::size_t i=0;i<4;++i) {
      const auto& actual=scene.vertices[first+i]; const auto& rest=authored->scene->vertices[first+i];
      if (std::abs(actual.position.x-(rest.position.x+float(.5*seconds)))>1e-6
          || actual.position.y!=rest.position.y || actual.position.z!=rest.position.z) return false;
    }
    return true;
  };
  check(literalSibling(*following.scene,2*dt),"independent authored sibling uses its own timed geometry outside the coupled collider domain");
  auto p=parameters; p.historyProjectSeconds=2*dt;
  auto frame=waterreplayfixture::frame(*history,2*dt,30);
  ParticleBodyState oracle; ParticleSolidState oracleSolids; WaterCoupledFrame oracleWater;
  const auto genuine=replayWaterCoupledBodies(p,2*dt,aspect,frame.get(),oracle,oracleSolids,nullptr,error,nullptr,&oracleWater);
  check(genuine && oracle==*following.replay->bodies() && oracleWater.heights==following.replay->water()->heights
      && oracleSolids.colliders.meshVertices==following.replay->solids()->colliders.meshVertices,
      "paired family output equals the actual one-binding solver oracle, including final collider resampling");
  const auto& collider=following.replay->solids()->colliders;
  bool sameCollider=true;
  for (std::size_t i=0;i<81;++i) {
    const auto& v=following.scene->vertices[i].position; const auto& b=*p.geometryBinding;
    sameCollider=sameCollider && collider.meshVertices[i]==SolidVector{b.centerX+b.scale*v.x,b.centerY+b.scale*v.y,b.centerZ+b.scale*v.z};
  }
  check(sameCollider,"family Water geometry and paired Solid terminal collider share exactly the same completed wave");
  auto repeat=sample(2*dt),reset=sample(0),seek=sample(2*dt);
  check(repeat.scene && reset.scene && seek.scene && timelinefamily::sameScene(*following.scene,*repeat.scene)
      && timelinefamily::sameScene(*following.scene,*seek.scene)
      && std::all_of(reset.replay->water()->heights.begin(),reset.replay->water()->heights.end(),[](double h) { return h==0; }),
      "repeat, backward reset and seek independently replay authored rest without frame-order state");
  auto exportOwner=owner; exportOwner.use=PlanUse::exportRender;
  auto exportFamily=runtime.admitTimelineSceneExport(exportOwner,*authored->scene,authored->bindings,{},error);
  auto exported=sample(2*dt,exportOwner,exportFamily);
  check(exported.scene && timelinefamily::sameScene(*exported.scene,*following.scene)
      && *exported.replay->bodies()==*following.replay->bodies(),"Export retains its own admitted use and equals the actual Preview pair");
  TimelineSceneFamilyEvaluator matching(runtime);
  auto accepted=matching.evaluateCoupled(owner,family,authored->scene,p,2*dt,2*dt,2*dt,aspect,frame,{},{},error);
  check(accepted.replay && accepted.replay->matches(runtime,owner,family,0,authored->scene,p,2*dt,2*dt,2*dt,aspect,frame,error),
      "private pair independently consumes the exact current live owner and all canonical/source metadata");
  if (!accepted.replay) return;
  const auto matches=[&](const ParticleParams& request,const auto& f,PlanOwnerIdentity o={31,32,33,34,35,PlanUse::preview},std::size_t child=0) {
    return accepted.replay->matches(runtime,o,family,child,authored->scene,request,2*dt,2*dt,2*dt,aspect,f,error);
  };
  auto foreignHistory=std::make_shared<ParticleHistorySource>(*history); auto changed=p; changed.history=foreignHistory;
  auto sameKeyFrame=waterreplayfixture::frame(*history,2*dt,30);
  auto foreignBinding=std::make_shared<ParticleGeometryBinding>(*p.geometryBinding); changed=p; changed.geometryBinding=foreignBinding;
  check(!matches(changed,frame) && !matches(p,sameKeyFrame) && !matches(p,frame,exportOwner) && !matches(p,frame,owner,1),
      "equal key/frame payload, copied binding, use and foreign child substitutions cannot consume the pair");
  changed=p; changed.history=std::shared_ptr<const ParticleHistorySource>(std::shared_ptr<const ParticleHistorySource>{},history.get());
  const std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> borrowedFrame(
      std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame>{},frame.get());
  check(!matches(changed,frame) && !matches(p,borrowedFrame),"same raw history/frame addresses with substituted control blocks refuse");
  changed=p; changed.history=foreignHistory;
  check(!matches(changed,frame),"equal-key history owner/control-block substitution refuses");
  changed=p; changed.preparedBodies=accepted.replay->bodies(); changed.preparedSolids=accepted.replay->solids();
  check(!matching.evaluateCoupled(owner,family,authored->scene,changed,2*dt,2*dt,2*dt,aspect,frame,{},{},error).scene,
      "caller prepared bodies and Solid state cannot mint paired authority");
  changed=p; foreignBinding=std::make_shared<ParticleGeometryBinding>(*p.geometryBinding);
  foreignBinding->bodySource=std::make_shared<const AdmittedValue>(family->children()[0].plan().value());
  changed.geometryBinding=foreignBinding;
  check(!matching.evaluateCoupled(owner,family,authored->scene,changed,2*dt,2*dt,2*dt,aspect,frame,{},{},error).scene,
      "dynamic body source authority remains refused in the released static sphere coupled lane");
  foreignBinding->bodySource.reset(); foreignBinding->points=foreignBinding->emitters;
  check(!matching.evaluateCoupled(owner,family,authored->scene,changed,2*dt,2*dt,2*dt,aspect,frame,{},{},error).scene,
      "additional Points collider cannot bypass the single Water mesh solver domain");
  changed=p; changed.impulseZ=-3;
  check(!matches(changed,frame),"authored scalar parameter substitution cannot reuse a completed pair");
  changed=p; auto changedMesh=std::make_shared<ValueDescriptor>(*p.geometryBinding->mesh);
  changedMesh->operations[1].field.values[4]=99; foreignBinding=std::make_shared<ParticleGeometryBinding>(*p.geometryBinding);
  foreignBinding->mesh=changedMesh; changed.geometryBinding=foreignBinding;
  check(!matching.evaluateCoupled(owner,family,authored->scene,changed,2*dt,2*dt,2*dt,aspect,frame,{},{},error).scene,
      "same stable IDs with different complete Water operation bytes refuse before replay");
  check(!matching.evaluateCoupled(owner,family,authored->scene,p,dt,2*dt,2*dt,aspect,frame,{},{},error).scene
      && !accepted.replay->matches(runtime,owner,family,0,authored->scene,p,2*dt,dt,2*dt,aspect,frame,error),
      "separate source/body/project clocks remain refused only in the new coupled domain");
  auto differentScore=std::make_shared<arbitmod::Score>(*history->score);
  differentScore->notes.push_back(waterreplayfixture::note(82,0,65));
  auto alteredHistory=*history; alteredHistory.score=differentScore;
  auto nonKeyPayload=waterreplayfixture::frame(alteredHistory,2*dt,30);
  check(nonKeyPayload && nonKeyPayload->key==frame->key && !matches(p,nonKeyPayload),
      "a changed non-key canonical score payload cannot substitute the retained full frame owner");
  auto foreignScene=std::make_shared<Visual3DScene>(*authored->scene); foreignScene->cameras.back().farPlane=78;
  check(!matching.evaluateCoupled(owner,family,foreignScene,p,2*dt,2*dt,2*dt,aspect,frame,{},{},error).scene,
      "inactive authored camera bytes remain part of the complete frozen scene association");
  // A loop-generation change alone must invalidate the issued pair, and a fresh
  // replay after that invalidation must reproduce the following-step output.
  const auto loopOnlyKey=history->key; ++history->key.loopGeneration;
  auto loopOnlyRegenerated=sample(2*dt);
  check(loopOnlyRegenerated.scene && timelinefamily::sameScene(*loopOnlyRegenerated.scene,*following.scene) && !matches(p,frame),
      "a loop-generation change alone refuses the issued coupled pair and its fresh replay matches the following-step output");
  // Loop back to an earlier time than the cached 2*dt pair: the fresh replay
  // must reproduce the first-step output, and the cached later pair must refuse.
  auto loopBackRegenerated=sample(dt);
  check(loopBackRegenerated.scene && timelinefamily::sameScene(*loopBackRegenerated.scene,*impact.scene) && !matches(p,frame),
      "looping back to an earlier time after a loop-generation change regenerates the first-step output and refuses the cached later pair");
  history->key=loopOnlyKey;
  const auto savedKey=history->key; ++history->key.seekGeneration; ++history->key.loopGeneration;
  auto regenerated=sample(2*dt);
  check(regenerated.scene && timelinefamily::sameScene(*regenerated.scene,*following.scene) && !matches(p,frame),
      "fresh actual seek/loop lifecycle independently reproduces following-step output while old live association refuses");
  history->key=savedKey;
  GeometryCorePlanRuntime foreignRuntime(backend,8); foreignRuntime.reconcileGeneration(31,32,33,error);
  auto foreignFamily=foreignRuntime.admitTimelineScenePreview(owner,*authored->scene,authored->bindings,{},error);
  check(foreignFamily && !accepted.replay->matches(foreignRuntime,owner,foreignFamily,0,authored->scene,p,
      2*dt,2*dt,2*dt,aspect,frame,error),"equal scene/plan family from a different runtime cannot substitute admission ownership");
  auto ephemeralHistory=std::make_shared<ParticleHistorySource>(*history); auto ephemeralParams=p;
  ephemeralParams.history=ephemeralHistory;
  std::weak_ptr<const ParticleHistorySource> weakHistory=ephemeralHistory;
  auto ephemeral=evaluator.evaluateCoupled(owner,family,authored->scene,ephemeralParams,2*dt,2*dt,2*dt,aspect,
      frame,{},{},error);
  ephemeralParams.history.reset(); ephemeralHistory.reset();
  check(ephemeral.replay && weakHistory.expired() && ephemeral.replay->water()->heights.size()==81
      && !ephemeral.replay->matches(runtime,owner,family,0,authored->scene,p,2*dt,2*dt,2*dt,aspect,frame,error),
      "escaped inspection retains immutable pair payloads without retaining or consulting old history callbacks");
  timelineevaluation::Envelope envelope; coupledwaterbudget::Recipe recipe;
  TimelineSceneFamilyEvaluator measured(runtime);
  auto witness=measured.evaluateCoupled(owner,family,authored->scene,p,2*dt,2*dt,2*dt,aspect,frame,{},{},error,&envelope,&recipe);
  check(witness.scene && recipe.intervals>=2 && recipe.at(coupledwaterbudget::Work::contactDuplicateComparisons)==63*63*recipe.intervals
      && recipe.at(coupledwaterbudget::Work::massPositionReads)==384
      && recipe.at(coupledwaterbudget::Work::previousHeightCopies)==5*81,
      "full coupled recipe names actual source, mass, historical calls, contact comparisons and copies");
  witness={};
  std::size_t residentCandidateScratch=0;
  const auto boundaryCase=[&](int bodies,bool populated,int sourceNotes=128) {
    auto residentHistory=std::make_shared<ParticleHistorySource>(waterreplayfixture::source());
    auto residentScore=std::make_shared<arbitmod::Score>(*residentHistory->score);
    if (populated) {
      // Reverse source order makes row IDs independent of vector positions.
      for (int id=sourceNotes;id>0;--id) {
        auto note=waterreplayfixture::note(id,0,60); note.lengthBeats=1;
        note.pitchBendPoints.push_back({});
        arbitmod::PitchAnchor anchor; anchor.id=id; anchor.frequency=440;
        note.pitchAnchors.push_back(anchor);
        residentScore->notes.push_back(note);
      }
    }
    residentHistory->score=residentScore;
    auto request=coupledParameters(*authored,residentHistory,error);
    auto binding=std::make_shared<ParticleGeometryBinding>(*request.geometryBinding);
    auto emitters=std::make_shared<ValueDescriptor>(*binding->emitters);
    auto& points=std::get<PointsData>(emitters->data).points; points.clear();
    for (int i=0;i<bodies;++i)
      points.push_back({{float(i%10)*.05f,float(i/10)*.05f,1},std::uint64_t(18200+i),1,{0,0,0,1}});
    binding->emitters=emitters; request.geometryBinding=binding;
    request.count=bodies; request.collisionMode=2; request.impulseZ=0; request.historyProjectSeconds=2*dt;
    auto current=waterreplayfixture::frame(*residentHistory,2*dt,30);
    TimelineSceneFamilyEvaluator measuredBoundary(runtime);
    timelineevaluation::Envelope cost; coupledwaterbudget::Recipe schedule;
    auto result=measuredBoundary.evaluateCoupled(owner,family,authored->scene,request,2*dt,2*dt,2*dt,
        aspect,current,{},{},error,&cost,&schedule);
    check(result.scene && result.replay && result.replay->solids()->count==bodies
        && current && current->noteRows()==(populated ? 128:0),
        "historical boundary witness actually publishes its requested bodies and resident canonical rows: "+error);
    if (!result.scene || !result.replay || !current) return;
    // Independent visits from the two executed intervals, their four collision
    // passes, and the linear successful preservation searches at both starts
    // and the terminal sample. The extra prospective interval is conservative.
    std::uint64_t collisionVisits=0,preservationVisits=0;
    for (int interval=0;interval<2;++interval)
      for (int pass=0;pass<4;++pass)
        for (int a=0;a<bodies;++a)
          for (int b=a+1;b<bodies;++b) ++collisionVisits;
    for (int sample=0;sample<3;++sample)
      for (int body=0;body<bodies;++body)
        for (int old=0;old<=body;++old) ++preservationVisits;
    check(schedule.at(coupledwaterbudget::Work::bodyPairVisits)>=collisionVisits
        && schedule.at(coupledwaterbudget::Work::bodyPreservationSearches)>=preservationVisits
        && (populated || collisionVisits==14160),
        "complete recipe covers actual four-pass 60-body visits and successful linear preservation searches");
    if (populated) {
      // Evaluate the actual producer at every distinct history tick, then walk
      // each occupied identity using Score::noteById's source order twice,
      // once for packing and once for freezing the canonical notation row.
      canonicalblockc::FrameProducer producer;
      std::uint64_t searched=0,packed=0;
      for (const auto tick:{std::int64_t(0),kParticleReplayStepTicks,2*kParticleReplayStepTicks}) {
        auto key=residentHistory->key; key.frame=tick; key.fps=kParticleReplayTicksPerSecond;
        const auto beat=float(residentHistory->timeline.secondsToBeat(double(tick)/kParticleReplayTicksPerSecond));
        key.beat=beat; auto historical=producer.evaluate(key,residentScore,beat);
        check(historical && historical->noteRows()==128,"actual historical producer retains all 128 resident rows");
        if (!historical) return;
        ++packed;
        for (int row=0;row<historical->noteRows();++row)
          for (int search=0;search<2;++search)
            for (const auto& note:residentScore->notes) {
              ++searched; if (note.id==historical->noteIdentities[std::size_t(row)]) break;
            }
      }
      check(packed==3 && (sourceNotes!=128 || searched==49536) && producer.packCount()==packed
          && schedule.at(coupledwaterbudget::Work::canonicalNoteSearchVisits)>=searched
          && schedule.at(coupledwaterbudget::Work::canonicalCandidateScans)>=packed*residentScore->notes.size()
          && (sourceNotes!=128 || schedule.at(coupledwaterbudget::Work::canonicalRowAssignmentSearches)>=searched/2)
          && schedule.at(coupledwaterbudget::Work::canonicalRowAssignmentSearches)>=packed*128*128,
          "canonical search bound covers independently walked packer and frozen-row searches across actual history");
    }
    if (populated) {
      if (sourceNotes==128) residentCandidateScratch=schedule.scratchBytes;
      else {
        const auto extraNotes=std::size_t(sourceNotes-128);
        const auto candidateWidth=sizeof(const arbitmod::Note*)+sizeof(int)+alignof(const arbitmod::Note*);
        check(residentScore->notes.size()==std::size_t(sourceNotes) && current->noteRows()==128
            && residentCandidateScratch>0
            && schedule.scratchBytes==residentCandidateScratch+extraNotes*(sizeof(int)+candidateWidth),
            "full-source candidate backing grows beyond 128 resident rows separately from note identity storage");
      }
    }
    coupledwaterbudget::Limits exact;
    exact.work=schedule.work; exact.source.work=schedule.source.work; exact.source.logicalBytes=schedule.source.logicalBytes;
    exact.maxRetainedBytes=cost.retainedBytes; exact.maxScratchBytes=cost.scratchBytes; exact.maxComparisonBytes=schedule.comparisonBytes;
    TimelineSceneFamilyEvaluator exactEvaluator(runtime);
    auto admitted=exactEvaluator.evaluateCoupled(owner,family,authored->scene,request,2*dt,2*dt,2*dt,
        aspect,current,{},exact,error);
    check(admitted.scene && admitted.replay && *admitted.replay->bodies()==*result.replay->bodies()
        && timelinefamily::sameScene(*admitted.scene,*result.scene),
        "exact complete prospective work and simultaneous storage limits admit the historical boundary");
    auto shortScratch=exact; --shortScratch.maxScratchBytes;
    TimelineSceneFamilyEvaluator scratchEvaluator(runtime);
    auto scratchCaller=particleBodyRenderThreadCaller(error); const auto beforeScratch=scratchCaller->usage();
    auto scratchRefused=scratchEvaluator.evaluateCoupled(owner,family,authored->scene,request,2*dt,2*dt,2*dt,
        aspect,current,{},shortScratch,error);
    const auto afterScratch=scratchCaller->usage();
    check(!scratchRefused.scene && !scratchRefused.replay && afterScratch.work>beforeScratch.work
        && afterScratch.logicalBytes==beforeScratch.logicalBytes,
        "one-byte-short full-source candidate scratch refuses permanently charged work and releases storage");
    for (std::size_t i=0;i<schedule.work.size();++i) {
      if (!schedule.work[i]) continue;
      auto shortLimit=exact; --shortLimit.work[i]; TimelineSceneFamilyEvaluator shortEvaluator(runtime);
      const auto refused=shortEvaluator.evaluateCoupled(owner,family,authored->scene,request,2*dt,2*dt,2*dt,
          aspect,current,{},shortLimit,error);
      check(!refused.scene && !refused.replay && result.replay->solids()->count==bodies,
          std::string("one-short boundary refuses before solver publication: ")+coupledwaterbudget::names[i]);
    }
    for (std::size_t i=0;i<schedule.source.work.size();++i) {
      if (!schedule.source.work[i]) continue;
      auto shortLimit=exact; --shortLimit.source.work[i]; TimelineSceneFamilyEvaluator shortEvaluator(runtime);
      auto refused=shortEvaluator.evaluateCoupled(owner,family,authored->scene,request,2*dt,2*dt,2*dt,
          aspect,current,{},shortLimit,error);
      check(!refused.scene && !refused.replay,"one-short populated/empty full source schedule refuses before solver publication");
    }
    auto failingHistory=std::make_shared<ParticleHistorySource>(*residentHistory); bool fail=true;
    failingHistory->parameterAt=[&](const std::string&,double,double&,std::string& diagnostic) {
      if (fail) { diagnostic="boundary callback refusal"; return false; } return true;
    };
    auto failingRequest=request; failingRequest.history=failingHistory;
    TimelineSceneFamilyEvaluator failedEvaluator(runtime); auto stable=particleBodyRenderThreadCaller(error);
    const auto before=stable->usage();
    auto failed=failedEvaluator.evaluateCoupled(owner,family,authored->scene,failingRequest,2*dt,2*dt,2*dt,
        aspect,current,{},{},error);
    const auto once=stable->usage();
    auto failedAgain=failedEvaluator.evaluateCoupled(owner,family,authored->scene,failingRequest,2*dt,2*dt,2*dt,
        aspect,current,{},{},error);
    const auto twice=stable->usage(); fail=false;
    auto retried=failedEvaluator.evaluateCoupled(owner,family,authored->scene,failingRequest,2*dt,2*dt,2*dt,
        aspect,current,{},{},error);
    check(!failed.scene && !failedAgain.scene && retried.scene && once.work>before.work && twice.work>once.work
        && once.logicalBytes==before.logicalBytes && twice.logicalBytes==before.logicalBytes,
        "repeated historical failures retain attempted work while payload storage retires for retry");
    coupledwaterbudget::Limits spent;
    const auto category=populated ? coupledwaterbudget::Work::canonicalNoteSearchVisits
        : coupledwaterbudget::Work::bodyPreservationSearches;
    spent.work[std::size_t(category)]=3*schedule.at(category);
    check(!failedEvaluator.evaluateCoupled(owner,family,authored->scene,request,2*dt,2*dt,2*dt,
        aspect,current,{},spent,error).scene && error.find(coupledwaterbudget::names[std::size_t(category)])!=std::string::npos,
        "the new producer and preservation categories retain both failed attempts and the successful retry");
    TimelineSceneFamilyEvaluator storageEvaluator(runtime); timelineevaluation::Envelope storageCost;
    auto incumbent=storageEvaluator.evaluateCoupled(owner,family,authored->scene,request,2*dt,2*dt,2*dt,
        aspect,current,{},{},error,&storageCost);
    timelineevaluation::Limits storageLimit; storageLimit.maxAdditionalBytes=storageCost.concurrentBytes;
    auto shortage=storageEvaluator.evaluateCoupled(owner,family,authored->scene,request,2*dt,2*dt,2*dt,
        aspect,current,storageLimit,{},error);
    check(incumbent.scene && !shortage.scene && timelinefamily::sameScene(*incumbent.scene,*result.scene),
        "historical boundary storage refusal preserves the escaped incumbent scene and pair");
    incumbent={};
    auto recovered=storageEvaluator.evaluateCoupled(owner,family,authored->scene,request,2*dt,2*dt,2*dt,
        aspect,current,storageLimit,{},error);
    check(bool(recovered.scene),"historical boundary storage retry succeeds after final incumbent payload release");
  };
  boundaryCase(60,false); boundaryCase(1,true); boundaryCase(1,true,512);
  for (std::size_t i=0;i<recipe.work.size();++i) {
    if (!recipe.work[i]) continue;
    coupledwaterbudget::Limits limits; limits.work[i]=recipe.work[i]-1;
    TimelineSceneFamilyEvaluator shortEvaluator(runtime);
    check(!shortEvaluator.evaluateCoupled(owner,family,authored->scene,p,2*dt,2*dt,2*dt,aspect,frame,{},limits,error).scene,
        "each one-short named complete coupled work category refuses before solver publication");
  }
  for (std::size_t i=0;i<recipe.source.work.size();++i) {
    if (!recipe.source.work[i]) continue;
    coupledwaterbudget::Limits limits; limits.source.work[i]=recipe.source.work[i]-1;
    TimelineSceneFamilyEvaluator shortEvaluator(runtime);
    check(!shortEvaluator.evaluateCoupled(owner,family,authored->scene,p,2*dt,2*dt,2*dt,aspect,frame,{},limits,error).scene,
        "each one-short complete source/canonical/score schedule refuses prospectively");
  }
  for (std::size_t i=0;i<recipe.source.logicalBytes.size();++i) {
    if (!recipe.source.logicalBytes[i]) continue;
    coupledwaterbudget::Limits limits; limits.source.logicalBytes[i]=recipe.source.logicalBytes[i]-1;
    TimelineSceneFamilyEvaluator shortEvaluator(runtime);
    check(!shortEvaluator.evaluateCoupled(owner,family,authored->scene,p,2*dt,2*dt,2*dt,aspect,frame,{},limits,error).scene,
        "each one-short separate source/wave logical storage category refuses before replay");
  }
  const std::pair<std::size_t coupledwaterbudget::Limits::*,std::size_t> storageCaps[] = {
    {&coupledwaterbudget::Limits::maxRetainedBytes,envelope.retainedBytes},
    {&coupledwaterbudget::Limits::maxScratchBytes,envelope.scratchBytes},
    {&coupledwaterbudget::Limits::maxComparisonBytes,recipe.comparisonBytes}
  };
  for (const auto& cap:storageCaps) {
    coupledwaterbudget::Limits limits; limits.*cap.first=cap.second-1;
    TimelineSceneFamilyEvaluator shortEvaluator(runtime);
    check(!shortEvaluator.evaluateCoupled(owner,family,authored->scene,p,2*dt,2*dt,2*dt,aspect,frame,{},limits,error).scene,
        "one-short simultaneous retained, scratch and independent-comparison envelopes refuse before publication");
  }
  const auto escaped=[&](int alias) {
    TimelineSceneFamilyEvaluator limited(runtime); timelineevaluation::Envelope cost;
    auto result=limited.evaluateCoupled(owner,family,authored->scene,p,2*dt,2*dt,2*dt,aspect,frame,{},{},error,&cost);
    if (!result.scene || !result.replay) { check(false,"escaped alias fixture requires a genuine issued pair: "+error); return; }
    std::shared_ptr<const Visual3DScene> scene=alias==0 ? result.scene:nullptr;
    std::shared_ptr<const WaterCoupledFrame> water=alias==1 ? result.replay->water():nullptr;
    std::shared_ptr<const ParticleSolidState> solids=alias==2 ? result.replay->solids():nullptr;
    result={};
    timelineevaluation::Limits shortStorage; shortStorage.maxAdditionalBytes=cost.concurrentBytes;
    auto failed=limited.evaluateCoupled(owner,family,authored->scene,p,2*dt,2*dt,2*dt,aspect,frame,shortStorage,{},error);
    check(!failed.scene && (scene || water || solids),"scene-only, Water-only and Solid-only aliases retain all pair/family storage debits");
    scene.reset(); water.reset(); solids.reset();
    auto retry=limited.evaluateCoupled(owner,family,authored->scene,p,2*dt,2*dt,2*dt,aspect,frame,shortStorage,{},error);
    check(bool(retry.scene),"only final alias payload release restores the simultaneous storage envelope for retry");
  };
  escaped(0); escaped(1); escaped(2);
  timelineevaluation::Limits shortStorage; shortStorage.maxAdditionalBytes=envelope.concurrentBytes-1;
  TimelineSceneFamilyEvaluator storageLimited(runtime);
  check(!storageLimited.evaluateCoupled(owner,family,authored->scene,p,2*dt,2*dt,2*dt,aspect,frame,shortStorage,{},error).scene
      && timelinefamily::sameScene(*accepted.scene,*following.scene),"one-byte-short full retained/scratch admission preserves escaped incumbent");
  const auto stableCaller=particleBodyRenderThreadCaller(error); const auto beforeMalformed=stableCaller->usage();
  audio[0].bands[17]=std::numeric_limits<float>::quiet_NaN();
  TimelineSceneFamilyEvaluator malformedEvaluator(runtime);
  auto malformed=malformedEvaluator.evaluateCoupled(owner,family,authored->scene,p,2*dt,2*dt,2*dt,aspect,frame,{},{},error);
  const auto afterMalformed=stableCaller->usage(); audio[0].bands[17]=0;
  check(!malformed.scene && afterMalformed.work>beforeMalformed.work && afterMalformed.logicalBytes==beforeMalformed.logicalBytes
      && timelinefamily::sameScene(*accepted.scene,*following.scene),
      "malformed requested audio with zero Water impulse gain still refuses with permanent work and atomic storage retirement");
  auto failHistory=std::make_shared<ParticleHistorySource>(*history); bool failing=true;
  failHistory->parameterAt=[&](const std::string&,double,double&,std::string& diagnostic) {
    if (failing) { diagnostic="injected historical parameter refusal"; return false; } return true;
  };
  auto failingParams=p; failingParams.history=failHistory;
  TimelineSceneFamilyEvaluator failingEvaluator(runtime); auto caller=particleBodyRenderThreadCaller(error);
  const auto before=caller->usage();
  auto failed=failingEvaluator.evaluateCoupled(owner,family,authored->scene,failingParams,2*dt,2*dt,2*dt,aspect,frame,{},{},error);
  const auto after=caller->usage(); failing=false;
  auto retry=failingEvaluator.evaluateCoupled(owner,family,authored->scene,failingParams,2*dt,2*dt,2*dt,aspect,frame,{},{},error);
  check(!failed.scene && retry.scene && after.work>before.work && after.logicalBytes==before.logicalBytes,
      "failed actual callbacks spend permanent work, retire candidate payloads, preserve incumbent and allow storage retry");
  coupledwaterbudget::Limits spent;
  spent.work[static_cast<std::size_t>(coupledwaterbudget::Work::historyCalls)]
      =2*recipe.at(coupledwaterbudget::Work::historyCalls);
  check(!failingEvaluator.evaluateCoupled(owner,family,authored->scene,failingParams,2*dt,2*dt,2*dt,aspect,frame,{},spent,error).scene,
      "failed historical attempt and retry retain the complete category epoch without refunded work credit");
  auto explosive=coupledSource(error,.5,1e41); auto explosiveOwner=owner; explosiveOwner.planGeneration+=20;
  auto explosiveFamily=explosive ? runtime.admitTimelineScenePreview(explosiveOwner,*explosive->scene,explosive->bindings,{},error):nullptr;
  check(bool(explosiveFamily),"later-sibling overflow fixture admits its finite authored-rest anchor: "+error);
  if (explosiveFamily) {
    auto explosiveParams=coupledParameters(*explosive,history,error); explosiveParams.historyProjectSeconds=2*dt;
    const auto old=caller->usage();
    auto rejected=evaluator.evaluateCoupled(explosiveOwner,explosiveFamily,explosive->scene,explosiveParams,2*dt,2*dt,2*dt,
        aspect,frame,{},{},error);
    const auto attempted=caller->usage(); explosiveParams.historyProjectSeconds=0;
    auto recovered=evaluator.evaluateCoupled(explosiveOwner,explosiveFamily,explosive->scene,explosiveParams,0,0,0,
        aspect,waterreplayfixture::frame(*history,0,30),{},{},error);
    check(!rejected.scene && !rejected.replay && recovered.scene && attempted.work>old.work
        && attempted.logicalBytes==old.logicalBytes && timelinefamily::sameScene(*accepted.scene,*following.scene),
        "later authored sibling failure retires the candidate pair and scene atomically, spends work and preserves incumbent");
  }
  auto duplicate=source(error,.5,true,1,0,100); auto duplicateOwner=owner; duplicateOwner.planGeneration+=10;
  auto duplicateFamily=duplicate ? runtime.admitTimelineScenePreview(duplicateOwner,*duplicate->scene,duplicate->bindings,{},error):nullptr;
  auto duplicateParams=duplicate ? coupledParameters(*duplicate,history,error):p; duplicateParams.historyProjectSeconds=2*dt;
  check(duplicateFamily && !evaluator.evaluateCoupled(duplicateOwner,duplicateFamily,duplicate->scene,duplicateParams,
      2*dt,2*dt,2*dt,aspect,frame,{},{},error).scene,"multiple positive-mass children cannot publish independent body trajectories");
  // No native renderer is invoked here; prepared fixture source is not native PASS.
}

template <typename Check>
void verify(Check check) {
  verifyCoupled(check);
  std::string error;
  auto authored=source(error);
  check(bool(authored),"family evaluation fixture lowers two different authored field plans: "+error);
  if (!authored) return;
  const auto savedAuthored=std::make_shared<Visual3DScene>(*authored->scene);
  Backend backend; GeometryCorePlanRuntime runtime(backend,8);
  const PlanOwnerIdentity owner{31,32,33,34,35,PlanUse::preview};
  check(runtime.reconcileGeneration(31,32,33,error),"family evaluator caller establishes initial generation");
  auto family=runtime.admitTimelineScenePreview(owner,*authored->scene,authored->bindings,{},error);
  check(bool(family),"two exact typed field children admit against the full authored scene: "+error);
  if (!family) return;
  TimelineSceneFamilyEvaluator evaluator(runtime);
  timelineevaluation::Envelope receipt;
  const auto evaluate=[&](double seconds) {
    return evaluator.evaluate(owner,family,authored->scene,seconds,{},error,nullptr,9.0);
  };
  auto half=evaluator.evaluate(owner,family,authored->scene,0.5,{},error,&receipt,9.0);
  check(half && literal(*half,*authored->scene,0,0.25,0.25),
      "two-child half-second literal preserves static cube, hierarchy, materials, texture, camera and light: "+error);
  check(receipt.sceneValidationCalls==4 && receipt.replayOperationVisits==33
      && receipt.gridVertexVisits==24 && receipt.gridCellVisits==6
      && receipt.displacementVisits==48 && receipt.replacementVertexVisits==8
      && receipt.rangeComparisons==8 && receipt.matrixMultiplyTerms==4*4*4*64
      && receipt.concurrentBytes>receipt.retainedBytes+receipt.scratchBytes,
      "incremental family receipt counts actual source loops separately from accepted admission");
  auto loop=evaluate(4.5),back=evaluate(0.25),reset=evaluate(0),negative=evaluate(-3.5);
  check(loop && literal(*loop,*authored->scene,0,0.25,2.25),"family loop retains distinct linear child source seconds");
  check(back && literal(*back,*authored->scene,-0.09567085809127245,0.23096988312782168,0.125),
      "backward seek starts independently from both authored meshes");
  check(reset && literal(*reset,*authored->scene,-0.17677669529663688,0.17677669529663688,0),
      "family reset has literal time-zero positions");
  check(negative && literal(*negative,*authored->scene,0,0.25,-1.75),"negative source-time overlay keeps the explicit global project anchor");
  auto exportedOwner=owner; exportedOwner.use=PlanUse::exportRender;
  auto exportedFamily=runtime.admitTimelineSceneExport(exportedOwner,*authored->scene,authored->bindings,{},error);
  auto exported=evaluator.evaluate(exportedOwner,exportedFamily,authored->scene,0.5,{},error);
  check(exported && literal(*exported,*authored->scene,0,0.25,0.25),"export independently evaluates both literal child fields");
  GeometryCorePlanRuntime reopened(backend); reopened.reconcileGeneration(31,32,33,error);
  auto reopenedFamily=reopened.admitTimelineScenePreview(owner,*authored->scene,authored->bindings,{},error);
  TimelineSceneFamilyEvaluator reopenedEvaluator(reopened);
  auto fresh=reopenedEvaluator.evaluate(owner,reopenedFamily,authored->scene,0.5,{},error);
  check(fresh && literal(*fresh,*authored->scene,0,0.25,0.25),"fresh reopen request satisfies literals without comparison to a prior evaluation");
  check(!evaluator.evaluate(owner,reopenedFamily,authored->scene,0.5,{},error),"equal family from another runtime cannot substitute admission ownership");
  auto wrongUse=owner; wrongUse.use=PlanUse::exportRender;
  check(!evaluator.evaluate(wrongUse,family,authored->scene,0.5,{},error),"preview child family cannot borrow export ownership");
  auto wrongPlan=owner; ++wrongPlan.planGeneration;
  check(!evaluator.evaluate(wrongPlan,family,authored->scene,0.5,{},error),"substituted parent plan generation refuses");
  auto wrongClip=owner; ++wrongClip.clipIdentity;
  check(!evaluator.evaluate(wrongClip,family,authored->scene,0.5,{},error),"substituted clip identity refuses");
  auto foreign=std::make_shared<Visual3DScene>(*authored->scene); foreign->cameras.back().farPlane=78;
  check(!evaluator.evaluate(owner,family,foreign,0.5,{},error),"unused camera record substitution refuses the exact frozen authored anchor");
  auto changed=authored->bindings; changed[1].loweredPlan[0]=changed[1].loweredPlan[0]=='0' ? '1' : '0';
  check(!runtime.admitTimelineScenePreview(owner,*authored->scene,changed,{},error),"second exact source bytes cannot replace an incumbent child");
  // This evaluator has one live scene. Obtain its prospective two-scene cost,
  // then refuse one byte below it and release the incumbent to recover space.
  timelineevaluation::Envelope liveReceipt;
  auto second=reopenedEvaluator.evaluate(owner,reopenedFamily,authored->scene,0.5,{},error,&liveReceipt);
  check(second && liveReceipt.concurrentBytes>receipt.concurrentBytes,"escaped scene owner remains charged to its evaluator");
  second.reset();
  timelineevaluation::Limits shortLimit{liveReceipt.concurrentBytes-1};
  check(!reopenedEvaluator.evaluate(owner,reopenedFamily,authored->scene,0.5,shortLimit,error)
      && fresh && literal(*fresh,*authored->scene,0,0.25,0.25),"one-byte short budget refuses before replay and preserves incumbent");
  fresh.reset();
  auto recovered=reopenedEvaluator.evaluate(owner,reopenedFamily,authored->scene,0.5,shortLimit,error);
  check(recovered && literal(*recovered,*authored->scene,0,0.25,0.25),"releasing escaped evaluation returns its exact reservation");
  auto explosive=source(error,1000);
  auto failingOwner=owner; ++failingOwner.planGeneration;
  auto failingFamily=explosive ? runtime.admitTimelineScenePreview(failingOwner,*explosive->scene,explosive->bindings,{},error) : nullptr;
  check(bool(failingFamily),"later-child rollback case admits its authored finite time-zero source");
  if (failingFamily) {
    timelineevaluation::Envelope beforeFailure,afterFailure;
    auto baseline=evaluator.evaluate(failingOwner,failingFamily,explosive->scene,0.00025,{},error,&beforeFailure);
    check(bool(baseline),"rollback reservation witness has a successful finite sibling request");
    baseline.reset();
    auto failed=evaluator.evaluate(failingOwner,failingFamily,explosive->scene,1e308,{},error);
    check(!failed && error.find("non-finite")!=std::string::npos && half
        && literal(*half,*authored->scene,0,0.25,0.25),"second child fails after first replacement without publishing partial geometry");
    auto retry=evaluator.evaluate(failingOwner,failingFamily,explosive->scene,0.00025,{},error,&afterFailure);
    check(retry && std::abs(retry->vertices[4].position.x+0.25f)<0.000001
        && beforeFailure.concurrentBytes==afterFailure.concurrentBytes,
        "failed sibling releases candidate accounting and retains the admitted authored source");
  }
  auto wet=source(error,0.5,true); auto wetOwner=owner; wetOwner.planGeneration+=2;
  auto wetFamily=wet ? runtime.admitTimelineScenePreview(wetOwner,*wet->scene,wet->bindings,{},error) : nullptr;
  check(bool(wetFamily),"two independent Water children actually admit before authenticated replay: "+error);
  if (wetFamily) {
    const auto& a=wetFamily->children()[0].plan().value().descriptor();
    const auto& b=wetFamily->children()[1].plan().value().descriptor();
    check(a.operations[1].stableId!=b.operations[1].stableId && a.operations[1].field.values[2]==0.2
        && b.operations[1].field.values[2]==0.4 && !equal(std::get<GeometryData>(a.data),std::get<GeometryData>(b.data)),
        "Water children own genuinely different plans, rest meshes, operation IDs and gains");
    auto history=waterreplayfixture::source();
    auto score=std::make_shared<arbitmod::Score>(*history.score);
    score->notes={waterreplayfixture::note(1,0,63.5f)}; history.score=score;
    constexpr double dt=1.0/120,local=23.5;
    auto current=waterreplayfixture::frame(history,dt);
    TimelineSceneFamilyEvaluator waterEvaluator(runtime);
    const auto sample=[&](double project,double sourceTime,const auto& frame) {
      return waterEvaluator.evaluate(wetOwner,wetFamily,wet->scene,sourceTime,{},error,nullptr,project,nullptr,&history,frame);
    };
    timelineevaluation::Envelope waterReceipt;
    auto first=waterEvaluator.evaluate(wetOwner,wetFamily,wet->scene,local,{},error,&waterReceipt,dt,nullptr,&history,current);
    check(first && waterLiteral(*first,*wet->scene,1),"independent literal one-step child heights/normals preserve the entire static scene: "+error);
    check(waterReceipt.waterChildren==2 && waterReceipt.waterHeightBytes==8*sizeof(double)
        && waterReceipt.water.at(videorender::watersourcebudget::Work::censusNotes)==2
        && waterReceipt.water.at(videorender::watersourcebudget::Work::scoreNotes)==2
        && waterReceipt.water.at(videorender::watersourcebudget::Work::stencilVertices)==8
        && waterReceipt.water.bytes(videorender::watersourcebudget::Storage::waveArrays)==32*sizeof(double)
        && waterReceipt.waterMemoBytes>waterReceipt.waterHeightBytes
        && waterReceipt.waterIdentityVisits==5*(8+12)
        && waterReceipt.waterRestPositionVisits==2*8
        && waterReceipt.waterMemoCopyElements==2*8+12
        && waterReceipt.waterColumnDiscoveryVisits==3*8,
        "all sibling source, retained heights and copied memo payloads have prospective independent counts");
    auto repeated=sample(dt,local,current),differentLocal=sample(dt,-7,current);
    auto later=sample(2*dt,local,waterreplayfixture::frame(history,2*dt));
    auto sought=sample(dt,local,current),zero=sample(0,local,waterreplayfixture::frame(history,0));
    check(repeated && differentLocal && sought && later && zero
        && waterLiteral(*repeated,*wet->scene,1) && waterLiteral(*differentLocal,*wet->scene,1)
        && waterLiteral(*sought,*wet->scene,1) && waterLiteral(*later,*wet->scene,2)
        && waterLiteral(*zero,*wet->scene,0),
        "repeat, independent local anchors, backward seek and reset use the actual project-zero Water time");
    auto retainedFrame=waterreplayfixture::frame(history,dt);
    std::weak_ptr<const canonicalblockc::CanonicalBlockCFrame> weakFrame=retainedFrame;
    auto frameOwner=sample(dt,local,retainedFrame); retainedFrame.reset();
    check(frameOwner && !weakFrame.expired(),"escaped Water scene retains the actual immutable canonical frame owner");
    frameOwner.reset();
    check(weakFrame.expired(),"only payload release retires that independent canonical frame owner");
    auto waterExportOwner=wetOwner; waterExportOwner.use=PlanUse::exportRender;
    auto waterExportFamily=runtime.admitTimelineSceneExport(waterExportOwner,*wet->scene,wet->bindings,{},error);
    auto waterExport=waterEvaluator.evaluate(waterExportOwner,waterExportFamily,wet->scene,local,{},error,nullptr,
        dt,nullptr,&history,waterreplayfixture::frame(history,dt,120,true));
    check(waterExport && waterLiteral(*waterExport,*wet->scene,1),"Export consumes the same authentic source/time request and literal child outputs");
    auto foreignHistory=history; ++foreignHistory.key.seekGeneration;
    check(!waterEvaluator.evaluate(wetOwner,wetFamily,wet->scene,local,{},error,nullptr,dt,nullptr,&foreignHistory,current)
        && !sample(2*dt,local,current)
        && !sample(dt,local,waterreplayfixture::frame(history,2*dt))
        && first && waterLiteral(*first,*wet->scene,1),
        "source lifecycle, global time and current-frame substitutions refuse without replacing escaped output");
    auto changedTempo=history; changedTempo.timeline.reset(90,4);
    check(!sample(dt,local,waterreplayfixture::frame(changedTempo,dt)),
        "a different actual tempo map's rounded canonical beat cannot substitute the current history clock");
    auto changedChildren=wet->bindings; std::swap(changedChildren[0].loweredPlan,changedChildren[1].loweredPlan);
    auto changedOwner=wetOwner; changedOwner.planGeneration+=10;
    check(!runtime.admitTimelineScenePreview(changedOwner,*wet->scene,changedChildren,{},error),
        "distinct child plan/rest identities cannot be swapped between authored object ranges");
    const auto shortage=[&](timelineevaluation::Limits limits,const char* message) {
      TimelineSceneFamilyEvaluator limited(runtime); timelineevaluation::Envelope refused;
      auto candidate=limited.evaluate(wetOwner,wetFamily,wet->scene,local,limits,error,&refused,dt,nullptr,&history,current);
      bool noReplay=true;
      for (std::size_t i=1;i<refused.attemptedWaterWork.size();++i) noReplay=noReplay && refused.attemptedWaterWork[i]==0;
      check(!candidate && noReplay && first && waterLiteral(*first,*wet->scene,1),message);
    };
    const std::pair<std::size_t timelineevaluation::Limits::*,std::size_t> waterWorkCaps[] = {
      {&timelineevaluation::Limits::maxWaterIdentityVisits,waterReceipt.waterIdentityVisits},
      {&timelineevaluation::Limits::maxWaterRestPositionVisits,waterReceipt.waterRestPositionVisits},
      {&timelineevaluation::Limits::maxWaterMemoCopyElements,waterReceipt.waterMemoCopyElements},
      {&timelineevaluation::Limits::maxWaterColumnDiscoveryVisits,waterReceipt.waterColumnDiscoveryVisits}
    };
    for (const auto& cap : waterWorkCaps) {
      timelineevaluation::Limits limits; limits.*cap.first=cap.second-1;
      shortage(limits,"each Water rest comparison, memo copy and column discovery work shortage refuses before replay");
    }
    for (std::size_t i=0;i<waterReceipt.water.work.size();++i) {
      if (!waterReceipt.water.work[i]) continue;
      timelineevaluation::Limits limits; limits.waterSource.work[i]=waterReceipt.water.work[i]-1;
      shortage(limits,"each all-sibling source work shortage refuses before replay, including the second child's allowance");
    }
    for (std::size_t i=0;i<waterReceipt.water.logicalBytes.size();++i) {
      if (!waterReceipt.water.logicalBytes[i]) continue;
      timelineevaluation::Limits limits; limits.waterSource.logicalBytes[i]=waterReceipt.water.logicalBytes[i]-1;
      shortage(limits,"each all-sibling source payload shortage refuses before replay allocation");
    }
    timelineevaluation::Envelope retainedReceipt;
    auto escaped=waterEvaluator.evaluate(wetOwner,wetFamily,wet->scene,local,{},error,&retainedReceipt,dt,nullptr,&history,current);
    escaped.reset();
    timelineevaluation::Limits retainedShort; retainedShort.maxAdditionalBytes=retainedReceipt.concurrentBytes-1;
    check(!waterEvaluator.evaluate(wetOwner,wetFamily,wet->scene,local,retainedShort,error,nullptr,dt,nullptr,&history,current)
        && first && waterLiteral(*first,*wet->scene,1),"escaped Water scenes and immutable child heights stay charged at one-byte-short prospective capacity");
    auto quiet=source(error,0.5,true,0); auto quietOwner=wetOwner; quietOwner.planGeneration+=20;
    auto quietFamily=quiet ? runtime.admitTimelineScenePreview(quietOwner,*quiet->scene,quiet->bindings,{},error) : nullptr;
    auto quietResult=quietFamily ? waterEvaluator.evaluate(quietOwner,quietFamily,quiet->scene,local,{},error,nullptr,dt,nullptr,&history,current) : nullptr;
    auto absent=history; absent.score.reset();
    check(quietResult && waterLiteral(*quietResult,*quiet->scene,0)
        && !waterEvaluator.evaluate(quietOwner,quietFamily,quiet->scene,local,{},error,nullptr,dt,nullptr,&absent,current)
        && !waterEvaluator.evaluate(quietOwner,quietFamily,quiet->scene,local,{},error,nullptr,dt,nullptr,nullptr,current),
        "zero impulse gains still require actual canonical score/history ownership");
    auto invalidHistory=history;
    auto invalidScore=std::make_shared<arbitmod::Score>(*score);
    invalidScore->notes.push_back(waterreplayfixture::note(2,100));
    invalidScore->notes.back().velocity=std::numeric_limits<float>::quiet_NaN(); invalidHistory.score=invalidScore;
    check(!waterEvaluator.evaluate(quietOwner,quietFamily,quiet->scene,local,{},error,nullptr,dt,nullptr,&invalidHistory,current)
        && quietResult && waterLiteral(*quietResult,*quiet->scene,0),
        "zero gains still validate the complete actual score, including malformed future notes");
    auto audioWet=source(error,0.5,true,1,1); auto audioOwner=wetOwner; audioOwner.planGeneration+=30;
    auto audioFamily=audioWet ? runtime.admitTimelineScenePreview(audioOwner,*audioWet->scene,audioWet->bindings,{},error) : nullptr;
    check(bool(audioFamily),"second-child positive-audio family admits its exact authored sources");
    if (audioFamily) {
      TimelineSceneFamilyEvaluator audioEvaluator(runtime);
      timelineevaluation::Envelope missingAudio;
      check(!audioEvaluator.evaluate(audioOwner,audioFamily,audioWet->scene,local,{},error,&missingAudio,dt,nullptr,&history,current)
          && missingAudio.attemptedWaterWork[static_cast<std::size_t>(videorender::watersourcebudget::Work::scoreNotes)]==0,
          "second positive gain requires actual audio coverage in all-child preflight before first replay");
      auto audio=waterreplayfixture::audio(2); history.audioFrames=&audio; history.audioSampleRate=61440;
      auto good=audioEvaluator.evaluate(audioOwner,audioFamily,audioWet->scene,local,{},error,nullptr,dt,nullptr,&history,current);
      check(good && waterLiteral(*good,*audioWet->scene,1),"positive audio gain consumes contiguous genuine zero-onset coverage");
      timelineevaluation::Envelope beforeFailure,failedReceipt,afterFailure;
      auto before=audioEvaluator.evaluate(audioOwner,audioFamily,audioWet->scene,local,{},error,&beforeFailure,dt,nullptr,&history,current);
      before.reset();
      audio[0].bands[17]=std::numeric_limits<float>::quiet_NaN();
      auto failed=audioEvaluator.evaluate(audioOwner,audioFamily,audioWet->scene,local,{},error,&failedReceipt,dt,nullptr,&history,current);
      check(!failed && error.find("malformed")!=std::string::npos && good && waterLiteral(*good,*audioWet->scene,1)
          && failedReceipt.attemptedWaterWork[static_cast<std::size_t>(videorender::watersourcebudget::Work::scoreNotes)]
              ==beforeFailure.attemptedWaterWork[static_cast<std::size_t>(videorender::watersourcebudget::Work::scoreNotes)]+2,
          "malformed second-child audio refuses after first replay without partial scene publication or work refund");
      audio[0].bands[17]=0;
      auto retry=audioEvaluator.evaluate(audioOwner,audioFamily,audioWet->scene,local,{},error,&afterFailure,dt,nullptr,&history,current);
      check(retry && waterLiteral(*retry,*audioWet->scene,1) && beforeFailure.concurrentBytes==afterFailure.concurrentBytes,
          "failed sibling destroys candidate payloads before releasing storage while escaped first output survives");
      timelineevaluation::Limits spent;
      spent.waterSource.work[static_cast<std::size_t>(videorender::watersourcebudget::Work::scoreNotes)]
          =afterFailure.attemptedWaterWork[static_cast<std::size_t>(videorender::watersourcebudget::Work::scoreNotes)]+1;
      check(!audioEvaluator.evaluate(audioOwner,audioFamily,audioWet->scene,local,spent,error,nullptr,dt,nullptr,&history,current),
          "cumulative failed and successful attempts cannot borrow a refunded second-child source allowance");
    }
    history.audioFrames=nullptr;
    auto coupledSource=source(error,0.5,true,1,0,1); auto coupledOwner=wetOwner; coupledOwner.planGeneration+=40;
    auto coupledFamily=coupledSource ? runtime.admitTimelineScenePreview(coupledOwner,*coupledSource->scene,coupledSource->bindings,{},error) : nullptr;
    check(coupledFamily && !waterEvaluator.evaluate(coupledOwner,coupledFamily,coupledSource->scene,local,{},error,nullptr,dt,nullptr,&history,current)
        && error.find("independently owned child frames")!=std::string::npos,"positive-mass Water children still refuse without independent coupled ownership");
    history.key.seekGeneration++; history.key.loopGeneration++;
    auto regenerated=sample(dt,local,waterreplayfixture::frame(history,dt));
    check(regenerated && waterLiteral(*regenerated,*wet->scene,1),"fresh canonical seek/loop lifecycle retains deterministic authored-rest replay");
  }
  videorender::WaterCoupledFrame coupled;
  check(!evaluator.evaluate(owner,family,authored->scene,0.5,{},error,nullptr,0.5,&coupled)
      && error.find("independently owned child frames")!=std::string::npos,"one coupled Water frame cannot be lent to a family");
  check(runtime.reconcileGeneration(31,32,34,error),"device restart establishes a new actual active generation");
  check(!evaluator.evaluate(owner,family,authored->scene,0.5,{},error)
      && half && literal(*half,*authored->scene,0,0.25,0.25),"stale escaped family refuses runtime re-admission while its completed snapshot survives");
  if (wetFamily) {
    auto history=waterreplayfixture::source();
    check(!evaluator.evaluate(wetOwner,wetFamily,wet->scene,23.5,{},error,nullptr,1.0/120,nullptr,&history,
        waterreplayfixture::frame(history,1.0/120)),"Water history/frame cannot revive a stale exact family admission");
  }
  auto wrongProject=owner; ++wrongProject.projectGeneration;
  check(!evaluator.evaluate(wrongProject,family,authored->scene,0.5,{},error),"substituted project generation refuses");
  auto wrongHelper=owner; ++wrongHelper.helperGeneration;
  check(!evaluator.evaluate(wrongHelper,family,authored->scene,0.5,{},error),"substituted helper generation refuses");
  family.reset();
  check(half && literal(*half,*authored->scene,0,0.25,0.25),"returned alias retains all independent family children after caller release and reconciliation");
  GeometryCorePlanRuntime pressure(backend,1); pressure.reconcileGeneration(31,32,33,error);
  auto heldEvaluator=std::make_unique<TimelineSceneFamilyEvaluator>(pressure);
  auto heldFamily=pressure.admitTimelineScenePreview(owner,*authored->scene,authored->bindings,{},error);
  auto sentinelA=std::make_shared<const int>(17),sentinelB=std::make_shared<const int>(23);
  std::weak_ptr<const int> weakA=sentinelA,weakB=sentinelB;
  if (heldFamily) {
    heldFamily->children()[0].nativeResources().value=sentinelA;
    heldFamily->children()[1].nativeResources().value=sentinelB;
  }
  auto held=heldEvaluator->evaluate(owner,heldFamily,authored->scene,0.5,{},error);
  heldEvaluator.reset();
  heldFamily.reset(); sentinelA.reset(); sentinelB.reset();
  auto nextDevice=owner; ++nextDevice.deviceGeneration;
  pressure.reconcileGeneration(31,32,34,error);
  check(held && !weakA.expired() && !weakB.expired()
      && !pressure.admitTimelineScenePreview(nextDevice,*authored->scene,authored->bindings,{},error),
      "one returned scene independently retains both native resources and blocks old-generation eviction");
  held.reset();
  auto nextFamily=pressure.admitTimelineScenePreview(nextDevice,*authored->scene,authored->bindings,{},error);
  check(nextFamily && weakA.expired() && weakB.expired(),
      "only releasing the final evaluated scene permits both child leases/resources to retire");
  check(timelinefamily::sameScene(*savedAuthored,*authored->scene),
      "every successful and refused request conserves the complete caller-owned authored scene");
}
} // namespace timelinefamilyfixture
