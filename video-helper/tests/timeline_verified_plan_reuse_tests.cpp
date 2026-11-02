// A timeline mesh is checked in full on the first frame of its authored scene. Later
// frames reuse that verdict, and must render exactly what the full checks would.
#include "composed_timeline_geometry_fixture.h"

#include <cstdio>
#include <memory>
#include <set>

namespace {
int failures=0;
void check(bool condition,const char* what) {
  if (condition) return;
  ++failures; std::fprintf(stderr,"FAILED: %s\n",what);
}
class Capabilities final : public videohelper::geometry::BackendCapabilitySource {
public:
  videohelper::geometry::BackendCapabilities geometryCoreCapabilities() const override {
    using namespace videohelper::geometry;
    using namespace videowire::geometry;
    BackendCapabilities c; c.backendIdentity="verified-plan-fixture"; c.deviceIdentity="fixture-device";
    c.nativeGpuAvailable=c.geometryCoreExecution=c.immutableSourceBuffers=c.stableElementIds=c.typedFieldEvaluation=true;
    c.supportedCarriers=carrierBit(CarrierKind::geometry3D);
    return c;
  }
};
// By Angle follows a moving mesh from frame to frame, and a reused verdict skips the
// vertex identity check. So at every time a reused runtime must agree with a fresh one.
void byAngleAgreesWithTheFullChecks() {
  using namespace videohelper::geometry;
  using namespace videowire::geometry;
  using namespace HarmonicMIDI::grid;
  std::string error;
  PortContract geometry; geometry.carrier=CarrierKind::geometry3D;
  geometry.maxVertices=4096; geometry.maxIndices=12288; geometry.maxAttributes=8;
  auto grid=lowerGrid(601,601,1,9,9,0.35f,geometry,error);
  auto wave=grid ? lowerDirectionalSurfaceWave(*grid,606,10000,0.35,2,30,0.5,0.3,4,geometry,error) : std::nullopt;
  auto shaded=wave ? lowerShadeEdit(*wave,620,kShadeByAngle,25,{},error) : std::nullopt;
  const auto shadedContract=shaded ? withAttributeContract(geometry,*shaded) : geometry;
  auto value=shaded ? admitValue(*shaded,shadedContract,error) : std::nullopt;
  check(value.has_value(),"the By Angle plan admits");
  if (!value) { std::fprintf(stderr,"%s\n",error.c_str()); return; }
  auto scene=std::make_shared<Visual3DScene>();
  scene->id={7}; scene->activeCamera={50};
  scene->cameras[0].id={50}; scene->cameras[0].transform.translation.z=4; scene->cameraCount=1;
  scene->lights[0].id={40}; scene->lights[0].kind=SceneLightKind::Directional; scene->lightCount=1;
  SceneObjectRecord object; object.id={10};
  SceneMaterialRecord material; material.id={20};
  check(appendGeometryToScene(*value,object,material,*scene,error) && validateVisual3DScene(*scene).valid(),"the By Angle scene is valid");
  const std::shared_ptr<const Visual3DScene> authored=scene;
  const TimelineSceneBinding binding{{10},encodeLoweredPlanText(lowerRuntimePlan(shadedContract,*value))};
  const PlanOwnerIdentity owner {3,4,5,8,9,PlanUse::preview};
  Capabilities capabilities;
  GeometryCorePlanRuntime reused(capabilities);
  std::string reusedError;
  check(evaluateComposedTimelineGeometry(reused,owner,binding,authored,0.0,reusedError)!=nullptr,"the authored frame verifies");
  int same=0;
  std::set<std::size_t> creases;
  for (int step=1; step<=48; ++step) {
    const double seconds=step/12.0;
    Capabilities freshCapabilities;
    GeometryCorePlanRuntime fresh(freshCapabilities);
    std::string freshError;
    reusedError.clear();
    const auto full=evaluateComposedTimelineGeometry(fresh,owner,binding,authored,seconds,freshError);
    const auto frame=evaluateComposedTimelineGeometry(reused,owner,binding,authored,seconds,reusedError);
    check(full!=nullptr,"the full checks accept every frame of the moving wave");
    check(frame!=nullptr,"reuse accepts every frame of the moving wave");
    if (!full || !frame) { std::fprintf(stderr,"%.3f s: %s | %s\n",seconds,freshError.c_str(),reusedError.c_str()); continue; }
    ++same;
    check(sceneaov::detail::encodeScene(*full)==sceneaov::detail::encodeScene(*frame),"reuse renders what the full checks render");
    std::size_t sharp=0;
    for (std::size_t a=0; a<frame->vertexCount; ++a)
      for (std::size_t b=a+1; b<frame->vertexCount; ++b) {
        const auto& p=frame->vertices[a]; const auto& q=frame->vertices[b];
        if (p.position.x==q.position.x && p.position.y==q.position.y && p.position.z==q.position.z
            && (p.normal.x!=q.normal.x || p.normal.y!=q.normal.y || p.normal.z!=q.normal.z)) ++sharp;
      }
    creases.insert(sharp);
  }
  std::printf("by angle: %d frames equal, %zu different crease counts, %llu reuses\n",same,creases.size(),
      static_cast<unsigned long long>(reused.verifiedTimelineReuses()));
  check(same==48,"every frame of the sweep renders");
  check(creases.size()>1,"the crease follows the moving wave");
  check(reused.verifiedTimelineReuses()>0,"the sweep reuses the verdict");
}

// A converging preview evaluates one frame once per sample. Only the first may do the work,
// and any change to what the mesh reads must evaluate and validate again.
// A lowered plan is hashed once per publication. A new publication hashes again, even at a
// freed address, and a changed plan in it is refused.
void publishedPlanHashesOnce(const videowire::geometry::TimelineSceneBinding& binding,
    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& authored,
    const videowire::geometry::TimelineSceneBinding& otherPlan) {
  using namespace videohelper::geometry;
  std::string error;
  const PlanOwnerIdentity owner {3,4,5,8,9,PlanUse::preview};
  Capabilities capabilities;
  GeometryCorePlanRuntime runtime(capabilities);
  check(evaluateComposedTimelineGeometry(runtime,owner,binding,authored,0.0,error)!=nullptr,"the first frame admits");
  auto publication=std::make_shared<int>(1);
  const auto reuse=[&](const auto& plan,const std::shared_ptr<int>& lease) {
    return runtime.reuseVerifiedTimelinePlan(owner,plan,authored,lease).has_value();
  };
  const auto before=runtime.timelinePlanDigests();
  check(reuse(binding,publication) && runtime.timelinePlanDigests()==before+1,"the first reuse in a publication hashes");
  check(reuse(binding,publication) && reuse(binding,publication) && runtime.timelinePlanDigests()==before+1,
      "later samples in that publication do not hash");
  publication=std::make_shared<int>(2);
  check(reuse(binding,publication) && runtime.timelinePlanDigests()==before+2,"a new publication hashes again");
  check(!reuse(otherPlan,publication) && runtime.timelinePlanDigests()==before+3,"a changed plan object is hashed and refused");
  check(runtime.reuseVerifiedTimelinePlan(owner,binding,authored).has_value()
      && runtime.timelinePlanDigests()==before+4,"a caller with no publication always hashes");
}

void repeatedSamplesEvaluateOnce(const videowire::geometry::TimelineSceneBinding& binding,
    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& authored) {
  using namespace videohelper::geometry;
  using Scene = HarmonicMIDI::grid::Visual3DScene;
  std::string error;
  const PlanOwnerIdentity owner {3,4,5,8,9,PlanUse::preview};
  Capabilities capabilities;
  GeometryCorePlanRuntime runtime(capabilities);
  const auto evaluate=[&](const std::shared_ptr<const Scene>& scene,double seconds) {
    return evaluateComposedTimelineGeometry(runtime,owner,binding,scene,seconds,error);
  };
  const auto first=evaluate(authored,0.5);
  check(first!=nullptr,"the first sample evaluates");
  for (int sample=0;sample<16;++sample)
    check(evaluate(authored,0.5)==first,"a later sample gets the same immutable snapshot");
  check(runtime.timelineSceneEvaluations()==1,"sixteen more samples evaluate and validate nothing");
  check(runtime.evaluatedTimelineReuses()==16,"each later sample is counted as a reuse");

  const auto moved=evaluate(authored,0.75);
  check(moved!=nullptr && moved!=first,"a new time is a new snapshot");
  check(runtime.timelineSceneEvaluations()==2,"a new time evaluates again");
  check(evaluate(authored,0.75)==moved,"the new time is then held");

  // An edit arrives as a new authored scene, which has to pass the full checks.
  const auto edited=std::make_shared<const Scene>(*authored);
  const auto afterEdit=evaluate(edited,0.75);
  check(afterEdit!=nullptr && afterEdit!=moved,"an edited scene gets its own snapshot");
  check(runtime.timelineSceneEvaluations()==3,"an edited scene evaluates again");
  auto broken=std::make_shared<Scene>(*authored);
  broken->vertices[broken->objects[0].firstVertex].position.x+=1.0f;
  const std::shared_ptr<const Scene> brokenScene=broken;
  check(evaluate(brokenScene,0.75)==nullptr,"an invalid edit is refused, not served from the last sample");
  check(runtime.timelineSceneEvaluations()==3,"a refused edit publishes nothing");
  error.clear();

  // Nothing is retained here: once the owner drops the snapshot it is evaluated afresh.
  const auto beforeDrop=runtime.timelineSceneEvaluations();
  { const auto scene=evaluate(authored,2.0); check(scene!=nullptr,"a held frame evaluates"); }
  check(evaluate(authored,2.0)!=nullptr,"a dropped snapshot evaluates again");
  check(runtime.timelineSceneEvaluations()==beforeDrop+2,"a dropped snapshot is not kept alive");

  // A score or beat map edit changes the frame key at an unchanged time.
  canonicalblockc::FrameKey frame;
  frame.scoreGeneration=1;
  const auto before=timelineSceneEvaluationInputs(0.5,0.5,true,&frame);
  check(before==timelineSceneEvaluationInputs(0.5,0.5,true,&frame),"equal inputs compare equal");
  frame.scoreGeneration=2;
  check(before!=timelineSceneEvaluationInputs(0.5,0.5,true,&frame),"a score edit changes the inputs");
  frame.scoreGeneration=1; frame.beatMapGeneration=7;
  check(before!=timelineSceneEvaluationInputs(0.5,0.5,true,&frame),"a beat map edit changes the inputs");
  check(before!=timelineSceneEvaluationInputs(0.5,0.25,true,nullptr),"the project time and frame are inputs");
}
}

int main() {
  using namespace videohelper::geometry;
  std::string error;
  const DirectionalWaveFixtureSettings wave;
  const auto request=composedTimelineGeometryFixture(error,&wave);
  check(request.has_value(),"fixture lowers");
  if (!request) { std::fprintf(stderr,"%s\n",error.c_str()); return 1; }
  const auto& binding=*request->timelineGeometry;
  const auto authored=request->sceneSnapshot;
  const PlanOwnerIdentity owner {3,4,5,8,9,PlanUse::preview};
  Capabilities capabilities;
  GeometryCorePlanRuntime runtime(capabilities);
  const auto evaluate=[&](GeometryCorePlanRuntime& on,const auto& scene,double seconds) {
    return evaluateComposedTimelineGeometry(on,owner,binding,scene,seconds,error);
  };
  const auto bytes=[](const auto& scene) { return sceneaov::detail::encodeScene(*scene); };

  const auto first=evaluate(runtime,authored,0.0);
  check(first!=nullptr,"first frame runs the full checks");
  check(runtime.verifiedTimelineReuses()==0,"first frame reuses nothing");
  const auto reused=evaluate(runtime,authored,0.5);
  check(reused!=nullptr,"second frame evaluates");
  check(runtime.verifiedTimelineReuses()==1,"second frame reuses the verdict");

  // Every frame after the first must match a runtime that ran all the checks.
  Capabilities freshCapabilities;
  GeometryCorePlanRuntime fresh(freshCapabilities);
  const auto full=evaluate(fresh,authored,0.5);
  check(full && reused && bytes(full)==bytes(reused),"a reused frame equals a fully checked frame");
  check(first && reused && bytes(first)!=bytes(reused),"a reused frame still moves with time");

  // A new scene object has not been checked, even with the same contents.
  const auto copy=std::make_shared<const HarmonicMIDI::grid::Visual3DScene>(*authored);
  check(evaluate(runtime,copy,0.5)!=nullptr,"a copied scene passes the full checks");
  check(runtime.verifiedTimelineReuses()==1,"a copied scene does not reuse the verdict");

  // A changed mesh in a new scene is still refused.
  auto changed=std::make_shared<HarmonicMIDI::grid::Visual3DScene>(*authored);
  changed->vertices[changed->objects[0].firstVertex].position.x+=1.0f;
  const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> changedScene=changed;
  check(evaluate(runtime,changedScene,0.5)==nullptr,"a changed mesh is refused");
  check(runtime.verifiedTimelineReuses()==1,"a changed mesh does not reuse the verdict");
  error.clear();
  check(evaluate(runtime,copy,1.0)!=nullptr,"the last verified scene still evaluates after a refusal");
  check(runtime.verifiedTimelineReuses()==2,"the last verified scene reuses its verdict");

  // A changed plan on the verified scene goes back through full admission.
  DirectionalWaveFixtureSettings otherWave; otherWave.amplitude=0.35;
  const auto other=composedTimelineGeometryFixture(error,&otherWave);
  check(other.has_value(),"the second plan lowers");
  if (!other) { std::fprintf(stderr,"%s\n",error.c_str()); return 1; }
  auto changedPlan=binding; changedPlan.loweredPlan=other->timelineGeometry->loweredPlan;
  check(changedPlan.loweredPlan!=binding.loweredPlan,"the second plan differs");
  const auto reusesBeforePlan=runtime.verifiedTimelineReuses();
  check(!runtime.reuseVerifiedTimelinePlan(owner,changedPlan,copy),"a changed plan finds no verdict");
  const auto otherAuthored=other->sceneSnapshot;
  const auto replanned=evaluateComposedTimelineGeometry(runtime,owner,changedPlan,otherAuthored,0.5,error);
  std::printf("changed plan: %s [%s] admissions %llu\n",replanned?"admitted":"refused",error.c_str(),(unsigned long long)runtime.admissionsInGeneration());
  check(runtime.verifiedTimelineReuses()==reusesBeforePlan,"a changed plan does not reuse the verdict");
  check(!runtime.reuseVerifiedTimelinePlan(owner,binding,authored),"the old plan's verdict is gone once replaced");
  error.clear();

  // Generations the runtime is not running on never reuse a verdict.
  check(evaluate(runtime,authored,0.5)!=nullptr,"the first plan verifies again");
  const auto reusesBeforeGenerations=runtime.verifiedTimelineReuses();
  auto wrongProject=owner; ++wrongProject.projectGeneration;
  auto wrongHelper=owner; ++wrongHelper.helperGeneration;
  auto wrongDevice=owner; ++wrongDevice.deviceGeneration;
  for (const auto& wrong : {wrongProject,wrongHelper,wrongDevice})
    check(!runtime.reuseVerifiedTimelinePlan(wrong,binding,authored),"a wrong generation is refused");
  auto incomplete=owner; incomplete.deviceGeneration=0;
  check(evaluateComposedTimelineGeometry(runtime,incomplete,binding,authored,0.5,error)==nullptr,"an incomplete generation is refused");
  check(runtime.verifiedTimelineReuses()==reusesBeforeGenerations,"refused generations reuse nothing");
  error.clear();
  // Moving on to a new device retires the old owner's verdict for good.
  check(evaluateComposedTimelineGeometry(runtime,wrongDevice,binding,authored,0.5,error)!=nullptr,"a new device passes the full checks");
  check(runtime.verifiedTimelineReuses()==reusesBeforeGenerations,"a new device starts without a verdict");
  check(!runtime.reuseVerifiedTimelinePlan(owner,binding,authored),"the old device's owner is refused");

  byAngleAgreesWithTheFullChecks();
  repeatedSamplesEvaluateOnce(binding,authored);
  publishedPlanHashesOnce(binding,authored,changedPlan);
  if (failures==0) std::puts("timeline verified plan reuse: ok");
  return failures==0 ? 0 : 1;
}
