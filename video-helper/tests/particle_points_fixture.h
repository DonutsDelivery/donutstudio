#pragma once

#include "../src/particle_body_replay.h"
#include "../src/geometry_core_backend.h"

inline videorender::ParticleParams particlePointsFixture()
{
    using namespace videorender;
    ParticleParams p;
    p.motionMode=2; p.historicalReplay=true; p.count=2; p.force=0;
    p.lifetime=2; p.resetMode=1; p.impulseX=.4f; p.impulseY=.2f; p.bodyRadius=.08f;
    auto score=std::make_shared<arbitmod::Score>(); score->scoreRevision=1;
    arbitmod::Note first; first.id=-7; first.trackId=0; first.startBeat=0;
    first.lengthBeats=2; first.midiNote=60; first.velocity=127;
    auto second=first; second.id=11; second.startBeat=.5f; second.midiNote=72; second.freqHz=first.freqHz*2;
    score->notes={second,first};
    auto history=std::make_shared<ParticleHistorySource>();
    history->score=score; history->timeline.reset(120,4);
    auto& key=history->key;
    key.projectGeneration=key.sourceGeneration=key.helperGeneration=key.backendGeneration=key.deviceGeneration=1;
    key.scoreGeneration=key.beatMapGeneration=key.fpsGeneration=key.loopGeneration=key.seekGeneration=1;
    key.fps=30; p.history=history;
    return p;
}

inline videowire::geometry::AdmissionContext particlePointsFixtureContext()
{
    videowire::geometry::AdmissionContext context;
    context.deferredPoints.emplace(801,videowire::geometry::DeferredPointsSource{801,3,2});
    return context;
}

inline videowire::geometry::PortContract particlePointsInstanceContract()
{
    videowire::geometry::PortContract contract;
    contract.carrier=videowire::geometry::CarrierKind::instances3D;
    contract.maxInstances=64; contract.maxAttributes=32; return contract;
}

inline std::optional<videowire::geometry::ValueDescriptor> particlePointsInstanceFixture(std::string& error)
{
    using namespace videowire::geometry;
    PortContract meshContract;
    meshContract.carrier=CarrierKind::geometry3D; meshContract.maxVertices=24; meshContract.maxIndices=36;
    const auto prototype=lowerMeshGenerator(802,802,1,OperationCode::cube,{2,2,2,0},meshContract,error);
    ValueDescriptor points;
    points.carrier=CarrierKind::points3D; points.stableId=points.sourceStableId=801;
    points.sourceRevision=3; points.data=PointsData{};
    points.operations={{801,OperationCode::copy,0,801,{}}}; points.dispatchCount=1;
    auto value=prototype ? lowerInstanceOnPoints(points,802,803,particlePointsInstanceContract(),error) : std::nullopt;
    if (value) {
        value->operations.insert(value->operations.begin(),prototype->operations.begin(),prototype->operations.end());
        value->dispatchCount=value->operations.size();
    }
    return value;
}

inline bool sampleParticlePointsFixture(videorender::ParticleParams p,double seconds,
    videowire::geometry::ValueDescriptor& value,std::string& error)
{
    p.historyProjectSeconds=seconds;
    return videorender::replayParticlePoints(p,seconds,1,nullptr,801,{801,3,2},value,error);
}

inline bool sameParticlePoints(const videowire::geometry::ValueDescriptor& a,
                              const videowire::geometry::ValueDescriptor& b)
{
    using namespace videowire::geometry;
    return equal(std::get<PointsData>(a.data),std::get<PointsData>(b.data)) && equalAttributes(a.attributes,b.attributes);
}

inline bool verifyParticlePointsOracles(std::string& error)
{
    using namespace videowire::geometry;
    auto p=particlePointsFixture();
    ValueDescriptor direct,again;
    if (!sampleParticlePointsFixture(p,.5,direct,error)) return false;
    const auto& points=std::get<PointsData>(direct.data).points;
    if (points.size()!=2 || points[0].stableId!=13 || points[1].stableId!=22) {
        error="Signed note identities must map reversibly independent of source order"; return false;
    }
    // Independent linear solution: force, gravity, drag and collisions are zero.
    // Notes begin at 0 and .25 seconds, with pitches 60 and 72 at 120 BPM.
    for (std::size_t i=0;i<2;++i) {
        const double age=i==0 ? .5:.25;
        const double spawnX=i==0 ? .42:.58;
        const auto& velocity=direct.attributes[0].elements[i].components;
        if (std::abs(points[i].position.x-(spawnX+.4*age))>2.e-6
            || std::abs(points[i].position.y-(.12+.2*age))>2.e-6
            || points[i].position.z!=0 || std::abs(velocity[0]-.4)>1.e-6 || std::abs(velocity[1]-.2)>1.e-6
            || velocity[2]!=0 || std::abs(direct.attributes[1].elements[i].components[0]-age)>1.e-7
            || std::abs(direct.attributes[2].elements[i].components[0]-1)>1.e-7) {
            error="Particle Points3D disagrees with the authored linear-motion, age or lifetime oracle"; return false;
        }
    }
    for (int frame=0;frame<=15;++frame)
        if (!sampleParticlePointsFixture(p,double(frame)/30,again,error)) return false;
    if (!sameParticlePoints(direct,again) || !sampleParticlePointsFixture(p,.9,again,error)
        || !sampleParticlePointsFixture(p,.125,again,error) || !sampleParticlePointsFixture(p,.5,again,error)
        || !sameParticlePoints(direct,again)) { error="Sequential and direct-seek points differ"; return false; }
    auto reset=p; reset.resetTime=.375f;
    if (!sampleParticlePointsFixture(reset,.5,again,error)) return false;
    for (std::size_t i=0;i<2;++i)
        if (std::abs(again.attributes[1].elements[i].components[0]-.125)>1.e-7
            || std::abs(again.attributes[2].elements[i].components[0]-(i==0 ? .625:.875))>1.e-7
            || std::abs(std::get<PointsData>(again.data).points[i].position.x-((i==0 ? .42:.58)+.05))>2.e-6) {
            error="Reset must restart the replay-owned birth and trajectory"; return false;
        }
    if (!sampleParticlePointsFixture(p,1.1,again,error) || std::get<PointsData>(again.data).points.size()!=1
        || std::get<PointsData>(again.data).points[0].stableId!=22
        || !sampleParticlePointsFixture(p,1.3,again,error) || !std::get<PointsData>(again.data).points.empty()) {
        error="Note expiry must change cardinality without inventing replacement identities"; return false;
    }
    auto changed=std::make_shared<videorender::ParticleHistorySource>(*p.history);
    auto score=std::make_shared<arbitmod::Score>(*changed->score);
    score->notes[0].midiNote=84; score->notes[0].freqHz*=2; score->scoreRevision++; changed->key.scoreGeneration++; changed->score=score;
    auto edited=p; edited.history=changed;
    if (!sampleParticlePointsFixture(edited,.5,again,error)
        || std::get<PointsData>(again.data).points[1].stableId!=22
        || std::abs(std::get<PointsData>(again.data).points[1].position.x-.84)>2.e-6
        || !sampleParticlePointsFixture(p,.5,again,error) || !sameParticlePoints(direct,again)) {
        error="Source edit and restored immutable score must retain identities and restore poses"; return false;
    }
    auto declaration=particlePointsFixtureContext();
    auto value=particlePointsInstanceFixture(error);
    const auto admitted=value ? admitValue(*value,particlePointsInstanceContract(),{},declaration,error) : std::nullopt;
    if (!admitted) return false;
    const auto wire=encodeRuntimeValue(*admitted);
    // Non-text, non-body values emit GCRV 13 (kNonTextTransportVersion).
    if (wire[4]!=13 || decodeRuntimeValue(wire,particlePointsInstanceContract(),{}, {},error)) {
        error="GCRV v13 must retain its format and refuse an undeclared external Points source"; return false;
    }
    declaration.evaluatedPoints.emplace(801,direct);
    ValueDescriptor replayed;
    if (!validateOperationPlan(*value,declaration,nullptr,error,&replayed)
        || std::get<InstancesData>(replayed.data).instances.size()!=2 || replayed.attributes.size()!=3) return false;
    const auto& instances=std::get<InstancesData>(replayed.data).instances;
    for (std::size_t i=0;i<2;++i)
        if (instances[i].stableId!=points[i].stableId || !equal(instances[i].transform.translation,points[i].position)
            || instances[i].sourceStableId!=802 || replayed.attributes[i].descriptor.domain!=Domain::instance) {
            error="Instance on Points must consume the exact evaluated points and source mesh"; return false;
        }
    auto hostile=declaration;
    hostile.evaluatedPoints[801].sourceRevision=4;
    if (validateOperationPlan(*value,hostile,nullptr,error,&replayed)) { error="Stale Points owner admitted"; return false; }
    hostile=declaration;
    std::get<PointsData>(hostile.evaluatedPoints[801].data).points[1].stableId=13;
    if (validateOperationPlan(*value,hostile,nullptr,error,&replayed)) { error="Duplicate Points identity admitted"; return false; }
    hostile=declaration; hostile.deferredPoints[801].capacity=1;
    if (validateOperationPlan(*value,hostile,nullptr,error,&replayed)) { error="Over-cap Points source admitted"; return false; }
    hostile=declaration; hostile.evaluatedPoints[801].dispatchCount=1;
    if (validateOperationPlan(*value,hostile,nullptr,error,&replayed)) { error="Point samples must not carry dispatches"; return false; }
    hostile=declaration; hostile.evaluatedPoints[801].fieldDomain=Domain::point;
    if (validateOperationPlan(*value,hostile,nullptr,error,&replayed)) { error="Point samples must not carry field headers"; return false; }
    hostile=declaration; hostile.deferredPoints.emplace(804,DeferredPointsSource{801,3,2});
    if (validateOperationPlan(*value,hostile,nullptr,error,&replayed)) { error="Disconnected deferred copy declaration admitted"; return false; }
    auto substituted=*value;
    substituted.operations[1].code=OperationCode::pointsFromVertices;
    substituted.operations[1].inputStableId=802;
    if (validateOperationPlan(substituted,declaration,nullptr,error,&replayed)) { error="Deferred source bound to an ordinary point operation"; return false; }
    for (int mode : {0,1}) {
        auto refused=p; refused.motionMode=mode;
        if (sampleParticlePointsFixture(refused,.5,again,error)) { error="Unimplemented particle mode published typed state"; return false; }
    }
    // Metadata collection does not change the existing Image replay body states.
    p.historyProjectSeconds=.5;
    const auto image=videorender::replayParticleBodies(p,.5,1,nullptr,&error);
    if (!error.empty() || std::abs(image[0][0]-.62f)>2.e-6 || std::abs(image[1][0]-.68f)>2.e-6) return false;
    error.clear(); return true;
}

class ParticlePointsSceneRecordingBackend final : public arbitgpu::NativeFixtureSceneBackend {
public:
    explicit ParticlePointsSceneRecordingBackend(arbitgpu::NativeFixtureSceneBackend& backend) : backend_(backend) {}
    arbitgpu::BackendInfo info() const override { return backend_.info(); }
    arbitgpu::GeometryCoreExecutionCapabilities geometryCoreCapabilities() const override {
        return backend_.geometryCoreCapabilities();
    }
    arbitgpu::NativeFixtureScenePreparation prepare(
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
        const std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram>& material = {},
        arbitgpu::NativeFixturePrepareUse use = arbitgpu::NativeFixturePrepareUse::Blocking) override {
        if (scene && scene->objectCount==1 && scene->objects[0].vertexCount==0 && scene->objects[0].indexCount==0) {
            const auto& root=scene->objects[0];
            // Check the actual produced scene before submitting it to GL/Metal.
            // Shared scene validation permits material-less transform roots;
            // this adapter also retains the exact existing material association.
            if (!HarmonicMIDI::grid::validateVisual3DScene(*scene).valid()
                || root.id.value!=1 || root.parent.isValid() || root.firstVertex!=0 || root.firstIndex!=0
                || scene->materialCount!=1 || !root.material.isValid() || root.material!=scene->materials[0].id) {
                arbitgpu::NativeFixtureScenePreparation failed;
                failed.error="Expired particle scene must have a valid non-drawing root and exact material owner";
                return failed;
            }
            ++blankPreparations;
        }
        return backend_.prepare(scene,material,use);
    }
    arbitgpu::NativeFixtureScenePreparation prepareGeometryInstances(
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
        const std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram>& material,
        const std::shared_ptr<const videohelper::geometry::AdmittedPlanValue>& admission,
        bool diagnostic = false, const std::vector<videowire::geometry::AttributeData>* attributes = nullptr,
        arbitgpu::NativeFixturePrepareUse use = arbitgpu::NativeFixturePrepareUse::Blocking) override {
        ++instancePreparations;
        lastInstanceScene=scene;
        return backend_.prepareGeometryInstances(scene,material,admission,diagnostic,attributes,use);
    }
    arbitgpu::NativeFixtureSceneSubmission render(
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
        const std::shared_ptr<const arbitgpu::NativeFixtureSceneResources>& resources,
        std::uint32_t width, std::uint32_t height, arbitgpu::NativeFixtureSceneRuntimeInputs inputs = {}) override {
        return backend_.render(scene,resources,width,height,std::move(inputs));
    }
    std::size_t blankPreparations=0,instancePreparations=0;
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> lastInstanceScene;
private:
    arbitgpu::NativeFixtureSceneBackend& backend_;
};

template <typename ReadPixels>
bool verifyNativeParticlePoints(arbitgpu::NativeFixtureSceneBackend& backend,ReadPixels readPixels,std::string& error)
{
    using namespace videowire::geometry;
    using namespace videohelper::geometry;
    if (!verifyParticlePointsOracles(error)) return false;
    auto value=particlePointsInstanceFixture(error);
    const auto context=particlePointsFixtureContext();
    const auto admitted=value ? admitValue(*value,particlePointsInstanceContract(),{},context,error) : std::nullopt;
    if (!admitted) return false;
    ParticlePointsSceneRecordingBackend recorded(backend);
    NativeGeometryCoreCapabilitySource capabilities(recorded);
    GeometryCorePlanRuntime runtime(capabilities);
    const auto bytes=lowerRuntimePlan(particlePointsInstanceContract(),*admitted);
    const auto preview=runtime.admitPreview({1,1,1,4,3,PlanUse::preview},bytes,{},context,error);
    const auto exported=runtime.admitExport({1,1,1,4,3,PlanUse::exportRender},bytes,{},context,error);
    if (!preview || !exported) return false;
    double seconds=.5;
    std::size_t evaluations=0;
    const PointsEvaluator evaluator=[&](auto id,const auto& binding,const auto& owner,auto& result,auto& diagnostic) {
        if (id!=801 || binding.sourceStableId!=801 || binding.sourceRevision!=3 || owner.planGeneration!=3
            || owner.projectGeneration!=1 || owner.helperGeneration!=1 || owner.deviceGeneration!=1
            || owner.clipIdentity!=4) { diagnostic="Native point oracle received a stale owner"; return false; }
        ++evaluations;
        return sampleParticlePointsFixture(particlePointsFixture(),seconds,result,diagnostic);
    };
    std::vector<std::uint8_t> firstPixels;
    for (double at : {.5,.8,1.1,1.3,.5}) {
        seconds=at;
        const auto a=executeNativeGeometry(recorded,*preview,64,64,error,false,{},nullptr,{},at,evaluator);
        const auto b=executeNativeGeometry(recorded,*exported,64,64,error,false,{},nullptr,{},at,evaluator);
        if (!a || !b) return false;
        const auto pixels=readPixels(a->frame);
        const std::vector<StableId> expected=at<1 ? std::vector<StableId>{13,22}
            : at<1.25 ? std::vector<StableId>{22} : std::vector<StableId>{};
        if (pixels.empty() || pixels!=readPixels(b->frame) || a->drawnInstanceIds!=expected || b->drawnInstanceIds!=expected) {
            error="Native particle instances must retain preview/export pixels and lifecycle IDs"; return false;
        }
        if (expected.empty() && (recorded.blankPreparations!=2 || a->stats.drawCount!=0 || b->stats.drawCount!=0
            || a->stats.ordinaryDrawCount!=0 || b->stats.ordinaryDrawCount!=0
            || a->stats.instancedDrawCount!=0 || b->stats.instancedDrawCount!=0
            || a->stats.submittedInstanceCount!=0 || b->stats.submittedInstanceCount!=0)) {
            error="Expired particle preview/export must prepare non-drawing scenes and submit no native draw"; return false;
        }
        if (at==.5) {
            if (firstPixels.empty()) firstPixels=pixels;
            else if (firstPixels!=pixels) { error="Native backward seek retained stale particle pixels"; return false; }
        } else if (pixels==firstPixels) { error="Native particle pixels did not follow motion or expiry"; return false; }
    }
    if (evaluations!=10) { error="Each preview and export frame must evaluate its point provider"; return false; }
    auto stale=*preview; stale.owner.planGeneration=4;
    if (executeNativeGeometry(backend,stale,64,64,error,false,{},nullptr,{},seconds,evaluator)) {
        error="Native deferred points accepted a stale request owner"; return false;
    }
    if (executeNativeGeometry(backend,*preview,64,64,error)) {
        error="Native deferred points rendered without their provider"; return false;
    }
    error.clear(); return true;
}

#include "particle_body_geometry_admission_fixture.h"
#include "../src/particle_points_body_frame.h"
#include "../src/visual_plan_executor.h"

static_assert(!std::is_default_constructible_v<videorender::ParticlePointsBodySourceAdmission>
    && !std::is_copy_constructible_v<videorender::ParticlePointsBodySourceAdmission>
    && !std::is_default_constructible_v<videorender::ParticlePointsBodyRequest>
    && !std::is_default_constructible_v<videorender::ParticlePointsBodyFrame>
    && !std::is_copy_constructible_v<videorender::ParticlePointsBodyFrame>,
    "Paired source/request/completed-frame authority has private issuance and immutable shared owners");

inline std::optional<videowire::geometry::ValueDescriptor> particlePointsTipBodyDescriptor(std::string& error)
{
    using namespace videowire::geometry;
    const auto mesh=particleBodyPyramidDescriptor(-.375f);
    PortContract vector; vector.carrier=CarrierKind::field; vector.fieldDomain=Domain::vertex;
    vector.fieldValueType=ValueType::vector; vector.fieldInterpolation=Interpolation::constant; vector.maxFieldElements=5;
    auto scalar=vector; scalar.fieldValueType=ValueType::floatValue;
    auto position=lowerElementField(mesh,84002,OperationCode::positionField,vector,error);
    FieldOperationSettings settings; settings.domain=Domain::vertex; settings.valueType=ValueType::floatValue; settings.mode=5;
    auto coordinate=position ? lowerConstructedField(84003,OperationCode::fieldVectorMeasure,settings,&*position,nullptr,nullptr,scalar,error):std::nullopt;
    settings.mode=1; settings.values={1,0,.375,.375,1,0,0,0};
    auto phase=coordinate ? lowerConstructedField(84004,OperationCode::fieldTimeline,settings,&*coordinate,nullptr,nullptr,scalar,error):std::nullopt;
    settings.values={2.0/3.0,0,.5,0,1,0,0,0};
    auto time=phase ? lowerConstructedField(84005,OperationCode::fieldTimeline,settings,&*phase,nullptr,nullptr,scalar,error):std::nullopt;
    settings.mode=1; settings.valueType=ValueType::vector; settings.values={1,0,0,0,0,0,0,0};
    auto offset=time ? lowerConstructedField(84006,OperationCode::fieldComposeVector,settings,&*time,nullptr,nullptr,vector,error):std::nullopt;
    return offset ? lowerVectorDisplacement(mesh,*offset,84007,1,particleBodyGeometryFixtureContract(),error):std::nullopt;
}

inline std::shared_ptr<const videorender::ParticlePointsBodySourceAdmission> particlePointsBodySourceFixture(
    const std::shared_ptr<videorender::ParticleBodyCallerLedger>& caller,std::string& error,
    videowire::geometry::DeferredBodyGeometrySource* compiledOut=nullptr,std::string* payloadOut=nullptr,std::string* bodyOut=nullptr)
{
    using namespace videowire::geometry;
    const auto descriptor=particlePointsTipBodyDescriptor(error);
    const auto token=descriptor ? particleBodyFixtureAdmitted(*descriptor,error):nullptr;
    if (!token) return {};
    DeferredBodyGeometrySource compiled;
    compiled.outputStableId=802; compiled.pointsOutputStableId=801; compiled.points={801,3,2};
    compiled.terminalStableId=token->descriptor().stableId; compiled.sourceStableId=token->descriptor().sourceStableId;
    compiled.sourceRevision=token->descriptor().sourceRevision; compiled.structuralRevision=3;
    compiled.source=token; compiled.reference=std::shared_ptr<const GeometryData>(token,&std::get<GeometryData>(token->descriptor().data));
    compiled.maxVertices=256; compiled.maxIndices=384;
    const auto body=encodeLoweredPlanText(lowerRuntimePlan(particleBodyGeometryFixtureContract(),*token));
    const auto payload="<NodeParams motionMode=\"2\" historicalReplay=\"1\" simulationSpace=\"1\" count=\"2\" "
        "lifetime=\"2\" resetMode=\"1\" speed=\"0\" bodyRadius=\"0.08\" impulseX=\"0.4\" impulseY=\"0.2\" bodyGeometryPlan=\""+body+"\"/>";
    if (compiledOut) *compiledOut=compiled;
    if (payloadOut) *payloadOut=payload;
    if (bodyOut) *bodyOut=body;
    return videowire::admitParticlePointsBodySource(payload,body,compiled,caller,error);
}

inline std::shared_ptr<const videorender::ParticlePointsBodyRequest> particlePointsBodyRequestFixture(
    const std::shared_ptr<const videorender::ParticlePointsBodySourceAdmission>& source,double seconds,std::string& error,
    videohelper::geometry::PlanUse use=videohelper::geometry::PlanUse::preview,int width=64,int height=64,double projectSeconds=-1)
{
    using namespace videorender;
    auto history=std::make_shared<ParticleHistorySource>(*particlePointsFixture().history);
    const auto project=projectSeconds<0 ? seconds:projectSeconds;
    auto& key=history->key; key.frame=static_cast<std::int64_t>(project*30); key.beat=project*2;
    canonicalblockc::FrameProducer producer;
    const auto frame=producer.evaluate(key,history->score,static_cast<float>(key.beat));
    return ParticlePointsBodyRequest::create(source,source->caller(),{1,1,1,4,3,use},800,width,height,history,frame,
        seconds,projectSeconds<0 ? seconds:projectSeconds,error);
}

inline std::optional<videowire::geometry::ValueDescriptor> particlePointsBodyCoreFixture(
    const videorender::ParticlePointsBodySourceAdmission& source,std::string& error)
{
    using namespace videowire::geometry;
    const auto& declaration=source.declaration();
    ValueDescriptor value; value.carrier=CarrierKind::instances3D; value.stableId=803;
    value.sourceStableId=801; value.sourceRevision=3; value.data=InstancesData{}; value.dispatchCount=3;
    auto retained=std::make_shared<RetainedMeshData>(); retained->geometry=*declaration.reference;
    OperationRecord body{802,OperationCode::retainedMesh,declaration.terminalStableId,802,{}};
    body.retainedMesh=std::move(retained);
    value.operations={std::move(body),{801,OperationCode::copy,0,801,{}},
        {803,OperationCode::instanceOnPoints,801,803,{},802}};
    AdmissionContext context; context.deferredPoints.emplace(801,declaration.points);
    context.deferredBodyGeometry.emplace(802,declaration);
    return admitValue(value,particlePointsInstanceContract(),{},context,error) ? std::optional<ValueDescriptor>(value):std::nullopt;
}

inline bool verifyParticlePointsBodyAuthority(std::string& error)
{
    using namespace videowire::geometry;
    using namespace videorender;
    const auto fail=[&](const char* reason) { error=reason; return false; };
    auto caller=makeParticleBodyCallerLedger({},error);
    DeferredBodyGeometrySource compiled; std::string payload,body;
    const auto source=particlePointsBodySourceFixture(caller,error,&compiled,&payload,&body);
    if (!source || source->declaration().source==compiled.source || source->declaration().sourceRevision!=7
        || source->declaration().structuralRevision!=3) return fail("Paired source must retain the genuine new decoder token and separate revisions");
    if (videowire::admitParticlePointsBodySource(payload,body+"00",compiled,caller,error))
        return fail("Paired source admitted substituted serialized body bytes");
    auto hostile=compiled; hostile.sourceRevision=8;
    if (videowire::admitParticlePointsBodySource(payload,body,hostile,caller,error))
        return fail("Paired source admitted a substituted source revision");
    ParticlePointsBodyEnvelope issuance;
    if (!ParticlePointsBodySourceAdmission::issuanceEnvelope(compiled,payload,body,issuance)) return fail("Fixture source schedule missing");
    const auto shortCaller=makeParticleBodyCallerLedger({sizeof(ParticleBodyCallerLedger)+sizeof(ParticleBodyCallerLease)+issuance.bytes-1,
        (issuance.work+particlePointsBodyIssuanceCensusWork)*2},error);
    if (!shortCaller || videowire::admitParticlePointsBodySource(payload,body,compiled,shortCaller,error)
        || shortCaller->usage().work!=issuance.work+particlePointsBodyIssuanceCensusWork || shortCaller->usage().logicalBytes!=sizeof(ParticleBodyCallerLedger))
        return fail("One-byte-short source refusal must precede decode and preserve attempted work");
    if (videowire::admitParticlePointsBodySource(payload,body,compiled,shortCaller,error)
        || shortCaller->usage().work!=(issuance.work+particlePointsBodyIssuanceCensusWork)*2
        || videowire::admitParticlePointsBodySource(payload,body,compiled,shortCaller,error))
        return fail("Repeated source shortages must exhaust the persistent caller work epoch");
    const auto shortWork=makeParticleBodyCallerLedger({256u*1024u*1024u,
        issuance.work+particlePointsBodyIssuanceCensusWork-1},error);
    if (!shortWork || videowire::admitParticlePointsBodySource(payload,body,compiled,shortWork,error)
        || shortWork->usage().work!=issuance.work+particlePointsBodyIssuanceCensusWork-1
        || shortWork->usage().logicalBytes!=sizeof(ParticleBodyCallerLedger))
        return fail("One-work-unit-short source admission must exhaust before decode");
    const auto request=particlePointsBodyRequestFixture(source,.5,error);
    const auto completed=request ? ParticlePointsBodyFrame::replay(request,error):nullptr;
    if (!completed || !completed->matches(*request,error) || completed->receipt()->sampler()->admission()!=completed->sample()->admission()
        || completed->sample()->absoluteTick()!=14112000 || completed->sample()->source()!=source->declaration().source)
        return fail("Paired Points and body must share one genuine complete terminal replay");
    const auto& mesh=completed->sample()->prototype();
    const float inverseRootThree=.5773502691896257645f;
    if (mesh.vertexIds!=std::vector<StableId>{11,19,23,41,67}
        || mesh.indices!=std::vector<std::uint32_t>{0,2,1,0,3,2,0,1,4,1,2,4,2,3,4,3,0,4}
        || std::abs(mesh.positions[4].x-.5f*inverseRootThree)>2.e-6f
        || !equal(completed->sample()->center(),Vec3{})
        || std::abs(completed->sample()->radius()-1.7320508075688772935f)>2.e-6f)
        return fail("Tip-only source must move x by one half with unchanged center, radius and ordered topology");
    // The existing Points conversion is the independent legacy producer oracle.
    auto legacy=request->parameters(); ValueDescriptor oldPoints;
    if (!replayParticlePoints(legacy,.5,1,request->frame(),801,{801,3,2},oldPoints,error)
        || !sameParticlePoints(oldPoints,completed->points())) return fail("Paired conversion changed legacy Points fields");
    const auto other=particlePointsBodyRequestFixture(source,.5,error);
    if (!other || completed->matches(*other,error)) return fail("A different current frame/history/context must refuse an old completed bundle");
    for (const auto& pair:std::array<std::pair<double,videohelper::geometry::PlanUse>,3>{{
        {.125,videohelper::geometry::PlanUse::preview},{.5,videohelper::geometry::PlanUse::exportRender},{.5,videohelper::geometry::PlanUse::preview}}}) {
        const auto fresh=particlePointsBodyRequestFixture(source,pair.first,error,pair.second);
        const auto result=fresh ? ParticlePointsBodyFrame::replay(fresh,error):nullptr;
        if (!result || !result->matches(*fresh,error) || result->sample()->absoluteTick()!=result->receipt()->terminalTick())
            return fail("Seek and separate Preview/Export requests must use their actual terminal replay tick");
        if (pair.first==.5 && (!equal(result->sample()->prototype(),mesh) || !sameParticlePoints(result->points(),completed->points())))
            return fail("Fresh seek and Preview/Export must preserve exact paired appearance and Points");
    }
    const auto shifted=particlePointsBodyRequestFixture(source,.25,error,videohelper::geometry::PlanUse::preview,64,64,.5);
    const auto shiftedFrame=shifted ? ParticlePointsBodyFrame::replay(shifted,error):nullptr;
    if (!shiftedFrame || !shiftedFrame->matches(*shifted,error) || shiftedFrame->sample()->absoluteTick()!=14112000
        || completed->matches(*shifted,error))
        return fail("Separate clip/project clocks must bind the actual scheduled terminal tick and refuse substituted clocks");
    auto value=particlePointsBodyCoreFixture(*source,error);
    AdmissionContext context; context.deferredPoints.emplace(801,source->declaration().points);
    context.deferredBodyGeometry.emplace(802,source->declaration());
    if (!value || !validateOperationPlan(*value,context,nullptr,error)) return fail("Immutable body declaration must validate through retainedMesh");
    context.evaluatedPoints.emplace(801,completed->points()); context.evaluatedBodyGeometry.emplace(802,completed->geometry());
    MaterializedInstancePlan materialized; ValueDescriptor replayed;
    if (!validateOperationPlan(*value,context,&materialized,error,&replayed)
        || !equal(materialized.geometries.at(802),mesh) || materialized.instances.instances.size()!=2)
        return fail("Core must materialize the sampled body before existing InstanceOnPoints");
    auto corrupt=context; corrupt.evaluatedBodyGeometry[802].source=compiled.source;
    if (validateOperationPlan(*value,corrupt,nullptr,error,&replayed)) return fail("Core accepted a substituted body token");
    corrupt=context; corrupt.evaluatedBodyGeometry[802].structuralRevision=7;
    if (validateOperationPlan(*value,corrupt,nullptr,error,&replayed)) return fail("Core conflated body source and structural revisions");
    corrupt=context; auto topology=std::make_shared<GeometryData>(mesh); topology->vertexIds[4]=99;
    corrupt.evaluatedBodyGeometry[802].geometry=topology;
    if (validateOperationPlan(*value,corrupt,nullptr,error,&replayed)) return fail("Core accepted changed body topology");
    corrupt=context; corrupt.deferredBodyGeometry.clear();
    if (validateOperationPlan(*value,corrupt,nullptr,error,&replayed)) return fail("Core accepted an orphan evaluated body sample");
    auto imported=*value; auto retained=std::make_shared<RetainedMeshData>(*imported.operations[0].retainedMesh);
    retained->importedAnimation="not-a-body-request"; imported.operations[0].retainedMesh=retained;
    if (validateOperationPlan(imported,context,nullptr,error,&replayed)) return fail("Body declaration collided with imported animation authority");
    {
        auto escapeCaller=makeParticleBodyCallerLedger({},error);
        auto escapeSource=particlePointsBodySourceFixture(escapeCaller,error);
        auto escapeRequest=escapeSource ? particlePointsBodyRequestFixture(escapeSource,.5,error):nullptr;
        auto escapeFrame=escapeRequest ? ParticlePointsBodyFrame::replay(escapeRequest,error):nullptr;
        if (!escapeFrame) return false;
        std::weak_ptr<const ParticlePointsBodySourceAdmission> sourceOwner=escapeSource;
        auto sampleOnly=escapeFrame->sample();
        escapeFrame.reset(); escapeRequest.reset(); escapeSource.reset();
        if (sourceOwner.expired() || escapeCaller->usage().logicalBytes<=sizeof(ParticleBodyCallerLedger))
            return fail("Sample-only escape must retain the already leased shared source closure");
        const auto work=escapeCaller->usage().work;
        sampleOnly.reset();
        if (!sourceOwner.expired() || escapeCaller->usage().logicalBytes!=sizeof(ParticleBodyCallerLedger)
            || escapeCaller->usage().work!=work)
            return fail("Final sample-only retirement must release source and conversion storage exactly once");
    }
    error.clear(); return true;
}

template <typename ReadPixels>
bool verifyNativeParticlePointsRuntimeResidency(arbitgpu::NativeFixtureSceneBackend& backend,
    ReadPixels readPixels,std::string& error)
{
    using namespace videowire::geometry;
    using namespace videohelper::geometry;
    using namespace videorender;
    using namespace HarmonicMIDI::grid;
    const auto fail=[&](const char* reason) { error=reason; return false; };
    PortContract cubeContract; cubeContract.carrier=CarrierKind::geometry3D;
    cubeContract.maxVertices=24; cubeContract.maxIndices=36;
    const auto cube=lowerMeshGenerator(910,910,1,OperationCode::cube,{1,1,1,0},cubeContract,error);
    const auto admittedCube=cube ? admitValue(*cube,cubeContract,error):std::nullopt;
    if (!admittedCube) return false;
    const auto cubeWire=lowerRuntimePlan(cubeContract,*admittedCube);
    const PlanOwnerIdentity ordinaryOwner{1,1,1,90,1,PlanUse::preview};
    auto familyScene=std::make_shared<Visual3DScene>();
    familyScene->id={7}; familyScene->activeCamera={50}; familyScene->cameraCount=1;
    familyScene->cameras[0].id={50}; familyScene->cameras[0].transform.translation.z=4;
    SceneObjectRecord object; object.id={10}; SceneMaterialRecord material; material.id={20};
    if (!appendGeometryToScene(*admittedCube,object,material,*familyScene,error)) return false;
    const std::vector<TimelineSceneBinding> bindings{{object.id,encodeLoweredPlanText(cubeWire)}};
    const PlanOwnerIdentity familyOwner{1,1,1,91,1,PlanUse::preview};
    ResourceLimits familyLimits;
    familyLimits.maxVertices=24; familyLimits.maxIndices=36; familyLimits.maxFieldElements=24;
    familyLimits.maxAttributes=1; familyLimits.maxOperations=1; familyLimits.maxDispatches=1;
    ParticlePointsSceneRecordingBackend recorded(backend);
    NativeGeometryCoreCapabilitySource capabilities(recorded);
    // Use fresh callers for independent cap domains, not for failed retries.
    for (int order=0;order<4;++order) {
        auto caller=makeParticleBodyCallerLedger({},error);
        const auto source=particlePointsBodySourceFixture(caller,error);
        const auto request=source ? particlePointsBodyRequestFixture(source,.5,error):nullptr;
        const auto value=source ? particlePointsBodyCoreFixture(*source,error):std::nullopt;
        AdmissionContext context;
        if (!source || !request || !value) return false;
        context.deferredPoints.emplace(801,source->declaration().points);
        context.deferredBodyGeometry.emplace(802,source->declaration());
        const auto bodyValue=admitValue(*value,particlePointsInstanceContract(),{},context,error);
        if (!bodyValue) return false;
        const auto wire=lowerRuntimePlan(particlePointsInstanceContract(),*bodyValue);
        ParticlePointsBodyEnvelope sourceEnvelope,frameEnvelope,conversion;
        std::size_t replayPeak=0;
        if (!particlePointsBodyCoreEnvelope(source->declaration(),wire.size(),false,sourceEnvelope)
            || !particlePointsBodyCoreEnvelope(source->declaration(),0,true,frameEnvelope)
            || !ParticlePointsBodyFrame::replayEnvelope(*request,conversion,replayPeak,error)) return false;
        ResourceLimits tight;
        tight.maxBufferBytes=caller->usage().logicalBytes+sourceEnvelope.bytes+frameEnvelope.bytes
            +2*sizeof(ParticleBodyCallerLease)+replayPeak;
        GeometryCorePlanRuntime runtime(capabilities,4);
        std::optional<RuntimeAdmission> ordinary;
        std::shared_ptr<const AdmittedTimelineScenePlans> family;
        const auto admitOther=[&] {
            if (order<2) ordinary=runtime.admitPreview(ordinaryOwner,cubeWire,{}, {},error);
            else family=runtime.admitTimelineScenePreview(familyOwner,*familyScene,bindings,familyLimits,error);
        };
        if (order%2==0) admitOther();
        auto paired=runtime.admitPointsBody(request->owner(),wire,tight,source,error);
        if (order%2==1) admitOther();
        if (!(ordinary || family) || !paired || runtime.admissionsInGeneration()!=2)
            return fail("Both real owners must coexist before mixed exact-cap frame admission in either order");
        arbitgpu::NativeFixtureSceneRuntimeInputs inputs; inputs.canonicalBlockCFrame=request->parameters().bodyReplayContext->frame;
        std::size_t callbacks=0;
        const PointsBodyEvaluator evaluator=[&](const auto& current,auto& diagnostic) {
            ++callbacks; return ParticlePointsBodyFrame::replay(current,diagnostic);
        };
        const auto resident=caller->usage().logicalBytes;
        const auto preparation=recorded.instancePreparations;
        const auto incumbent=paired->nativeResources->value;
        for (int retry=0;retry<2;++retry) {
            const auto work=caller->usage().work;
            if (executeNativeGeometry(recorded,*paired,64,64,error,false,inputs,nullptr,{},.5,{},nullptr,
                std::numeric_limits<double>::quiet_NaN(),nullptr,request,evaluator) || callbacks
                || recorded.instancePreparations!=preparation || paired->nativeResources->value!=incumbent
                || caller->usage().logicalBytes!=resident || caller->usage().work<=work)
                return fail("Current mixed owner census must refuse before callback/preparation and retain failed retry work");
        }
        const auto failedWork=caller->usage().work;
        ordinary.reset(); family.reset();
        if (!runtime.reconcileGeneration(2,2,2,error) || caller->usage().logicalBytes>=resident)
            return fail("Final mixed owner retirement did not release its one census registration");
        auto draw=executeNativeGeometry(recorded,*paired,64,64,error,false,inputs,nullptr,{},.5,{},nullptr,
            std::numeric_limits<double>::quiet_NaN(),nullptr,request,evaluator);
        if (!draw || callbacks!=1 || readPixels(draw->frame).empty() || caller->usage().work<=failedWork)
            return fail("Retired mixed owner must restore the paired-only exact boundary without refunding failed work");
    }
    // Sufficient mixed capacity still renders the original genuine body oracle.
    auto caller=makeParticleBodyCallerLedger({},error);
    const auto source=particlePointsBodySourceFixture(caller,error);
    const auto request=source ? particlePointsBodyRequestFixture(source,.5,error):nullptr;
    const auto value=source ? particlePointsBodyCoreFixture(*source,error):std::nullopt;
    if (!source || !request || !value) return false;
    AdmissionContext context; context.deferredPoints.emplace(801,source->declaration().points);
    context.deferredBodyGeometry.emplace(802,source->declaration());
    const auto bodyValue=admitValue(*value,particlePointsInstanceContract(),{},context,error);
    if (!bodyValue) return false;
    const auto wire=lowerRuntimePlan(particlePointsInstanceContract(),*bodyValue);
    arbitgpu::NativeFixtureSceneRuntimeInputs inputs; inputs.canonicalBlockCFrame=request->parameters().bodyReplayContext->frame;
    GeometryCorePlanRuntime runtime(capabilities,4);
    auto paired=runtime.admitPointsBody(request->owner(),wire,{},source,error);
    auto ordinary=runtime.admitPreview(ordinaryOwner,cubeWire,{}, {},error);
    auto family=runtime.admitTimelineScenePreview(familyOwner,*familyScene,bindings,familyLimits,error);
    if (!paired || !ordinary || !family) return false;
    const auto originalPlan=ordinary->plan.get();
    auto reused=runtime.admitPreview(ordinaryOwner,cubeWire,{}, {},error);
    if (!reused || reused->plan.get()!=originalPlan)
        return fail("Residency attachment must preserve genuine immutable plan identity and owner reuse");
    reused.reset();
    {
        auto substituted=*paired; substituted.plan=ordinary->plan;
        const auto substitutedWork=caller->usage().work;
        const auto substitutedPreparation=recorded.instancePreparations;
        if (executeNativeGeometry(recorded,substituted,64,64,error,false,inputs,nullptr,{},.5,{},nullptr,
            std::numeric_limits<double>::quiet_NaN(),nullptr,request,{})
            || caller->usage().work<=substitutedWork || recorded.instancePreparations!=substitutedPreparation)
            return fail("Mutating a runtime-issued ownership field must not replace its private census association");
    }
    bool ordinaryRefused=false,familyRefused=false,generationRefused=false,nativeRefused=false;
    const PointsBodyEvaluator reentrant=[&](const auto& current,auto& diagnostic) {
        const auto count=runtime.admissionsInGeneration();
        const auto bytes=caller->usage().logicalBytes;
        const auto work=caller->usage().work;
        auto nextOrdinary=ordinaryOwner; ++nextOrdinary.clipIdentity;
        auto nextFamily=familyOwner; nextFamily.clipIdentity+=2;
        ordinaryRefused=!runtime.admitPreview(nextOrdinary,cubeWire,{}, {},diagnostic);
        familyRefused=!runtime.admitTimelineScenePreview(nextFamily,*familyScene,bindings,familyLimits,diagnostic);
        generationRefused=!runtime.reconcileGeneration(2,2,2,diagnostic);
        nativeRefused=!executeNativeGeometry(recorded,*ordinary,64,64,diagnostic);
        if (runtime.admissionsInGeneration()!=count || caller->usage().logicalBytes!=bytes
            || caller->usage().work<=work) return std::shared_ptr<const ParticlePointsBodyFrame>{};
        diagnostic.clear(); return ParticlePointsBodyFrame::replay(current,diagnostic);
    };
    auto draw=executeNativeGeometry(recorded,*paired,64,64,error,false,inputs,nullptr,{},.5,{},nullptr,
        std::numeric_limits<double>::quiet_NaN(),nullptr,request,reentrant);
    if (!draw || !ordinaryRefused || !familyRefused || !generationRefused || !nativeRefused
        || draw->drawnInstanceIds!=std::vector<StableId>{13,22} || readPixels(draw->frame).empty()
        || !recorded.lastInstanceScene || recorded.lastInstanceScene->vertexCount!=5
        || std::abs(recorded.lastInstanceScene->vertices[4].position.x-.5f*.5773502691896257645f)>2.e-6f)
        return fail("Sufficient mixed cap must preserve the body oracle while refusing unmodelled reentrant mutations");
    draw.reset();
    {
        ResourceLimits shortOrdinary; shortOrdinary.maxBufferBytes=65536;
        auto refusedOwner=ordinaryOwner; refusedOwner.clipIdentity+=8;
        const auto incumbent=paired->nativeResources->value;
        const auto bytes=caller->usage().logicalBytes;
        const auto count=runtime.admissionsInGeneration();
        for (int retry=0;retry<2;++retry) {
            const auto work=caller->usage().work;
            if (runtime.admitPreview(refusedOwner,cubeWire,shortOrdinary,{},error)
                || caller->usage().logicalBytes!=bytes || caller->usage().work<=work
                || runtime.admissionsInGeneration()!=count || paired->nativeResources->value!=incumbent)
                return fail("Short ordinary live aggregate must preserve incumbent bytes/publication and cumulative retry work");
        }
    }
    // Retain each ordinary owning field alone while generation reconciliation
    // retires the cache entry. Owner-only keeps its original eviction policy.
    ordinary.reset(); family.reset();
    if (!runtime.reconcileGeneration(2,2,2,error) || !runtime.reconcileGeneration(1,1,1,error)) return false;
    for (int kind=0;kind<3;++kind) {
        auto owner=ordinaryOwner; owner.clipIdentity+=10+kind;
        auto admitted=runtime.admitPreview(owner,cubeWire,{}, {},error);
        if (!admitted) return false;
        std::shared_ptr<const void> escaped;
        if (kind==0) escaped=admitted->plan;
        else if (kind==1) escaped=admitted->ownerLease;
        else escaped=admitted->nativeResources;
        admitted.reset();
        const auto resident=caller->usage().logicalBytes;
        const auto work=caller->usage().work;
        if (!runtime.reconcileGeneration(2,2,2,error) || caller->usage().logicalBytes<resident)
            return fail("Generation replacement refunded an escaped ordinary owning alias");
        // Owner-only cannot be evicted until its final alias retires.
        escaped.reset();
        if (!runtime.reconcileGeneration(2,2,2,error) || caller->usage().logicalBytes>=resident
            || caller->usage().work<=work)
            return fail("Final ordinary alias retirement must refund its one registration without restoring work");
        if (!runtime.reconcileGeneration(1,1,1,error)) return false;
    }
    for (int kind=0;kind<2;++kind) {
        auto owner=ordinaryOwner; owner.clipIdentity+=30+kind;
        auto admitted=runtime.admitPreview(owner,cubeWire,{}, {},error);
        auto native=admitted ? executeNativeGeometry(recorded,*admitted,64,64,error):std::nullopt;
        if (!native) return false;
        std::shared_ptr<const void> escaped=kind==0
            ? std::shared_ptr<const void>(native->frame):std::shared_ptr<const void>(native->resources);
        native.reset(); admitted.reset();
        const auto resident=caller->usage().logicalBytes;
        const auto work=caller->usage().work;
        if (!runtime.reconcileGeneration(2,2,2,error) || caller->usage().logicalBytes<resident)
            return fail("Native frame/resource-only escape lost the ordinary runtime residency registration");
        escaped.reset();
        if (!runtime.reconcileGeneration(2,2,2,error) || caller->usage().logicalBytes>=resident
            || caller->usage().work<=work)
            return fail("Final ordinary native alias must retire one owner/frame debit with cumulative work intact");
        if (!runtime.reconcileGeneration(1,1,1,error)) return false;
    }
    {
        GeometryCorePlanRuntime evictionRuntime(capabilities,2);
        auto body=evictionRuntime.admitPointsBody(request->owner(),wire,{},source,error);
        auto old=evictionRuntime.admitPreview(ordinaryOwner,cubeWire,{}, {},error);
        if (!body || !old) return false;
        auto escapedPlan=old->plan;
        const auto identity=escapedPlan.get();
        old.reset();
        const auto bytes=caller->usage().logicalBytes;
        auto replacementOwner=ordinaryOwner; ++replacementOwner.clipIdentity;
        auto replacement=evictionRuntime.admitPreview(replacementOwner,cubeWire,{}, {},error);
        if (!replacement || escapedPlan.get()!=identity || caller->usage().logicalBytes<=bytes
            || evictionRuntime.admissionsInGeneration()!=3)
            return fail("LRU eviction must conserve the escaped plan's real identity and census debit");
        const auto escapedBytes=caller->usage().logicalBytes;
        const auto work=caller->usage().work;
        escapedPlan.reset();
        if (caller->usage().logicalBytes>=escapedBytes || caller->usage().work!=work)
            return fail("Final evicted plan retirement must refund its census registration once");
        const auto retired=caller->usage().logicalBytes;
        escapedPlan.reset();
        if (caller->usage().logicalBytes!=retired || caller->usage().work!=work)
            return fail("Evicted plan storage or work was refunded twice");
    }
    {
        std::shared_ptr<const AdmittedTimelineScenePlans> escapedFamily;
        std::optional<RuntimeAdmission> body;
        {
            GeometryCorePlanRuntime familyRuntime(capabilities,4);
            body=familyRuntime.admitPointsBody(request->owner(),wire,{},source,error);
            escapedFamily=familyRuntime.admitTimelineScenePreview(familyOwner,*familyScene,bindings,familyLimits,error);
            if (!body || !escapedFamily) return false;
        }
        const auto bytes=caller->usage().logicalBytes;
        const auto work=caller->usage().work;
        if (!(escapedFamily->owner()==familyOwner) || escapedFamily->children().size()!=1)
            return fail("Family-only escape lost the genuine owner or child table");
        escapedFamily.reset();
        if (caller->usage().logicalBytes>=bytes || caller->usage().work!=work)
            return fail("Final family-only retirement must refund storage once without restoring attempted work");
    }
    // A child-only copy survives retirement of both family parent and runtime.
    std::optional<TimelineSceneChildAdmission> escapedChild;
    std::optional<RuntimeAdmission> escapedPaired;
    {
        GeometryCorePlanRuntime familyRuntime(capabilities,4);
        escapedPaired=familyRuntime.admitPointsBody(request->owner(),wire,{},source,error);
        auto admitted=familyRuntime.admitTimelineScenePreview(familyOwner,*familyScene,bindings,familyLimits,error);
        if (!escapedPaired || !admitted) return false;
        escapedChild.emplace(admitted->children().front());
    }
    const auto childBytes=caller->usage().logicalBytes;
    const auto childWork=caller->usage().work;
    if (!escapedChild->plan().value().descriptor().stableId || !escapedChild->scene().objectCount)
        return fail("Escaped family child must preserve its genuine plan and frozen scene");
    escapedChild.reset();
    if (caller->usage().logicalBytes>=childBytes || caller->usage().work!=childWork)
        return fail("Family child final retirement must refund its registration once and preserve attempted work");
    const auto retiredBytes=caller->usage().logicalBytes;
    escapedChild.reset();
    if (caller->usage().logicalBytes!=retiredBytes || caller->usage().work!=childWork)
        return fail("Repeated family alias retirement refunded storage or failed work twice");
    error.clear(); return true;
}

template <typename ReadPixels>
bool verifyNativeParticlePointsBody(arbitgpu::NativeFixtureSceneBackend& backend,ReadPixels readPixels,std::string& error)
{
    using namespace videowire::geometry;
    using namespace videohelper::geometry;
    using namespace videorender;
    if (!verifyParticlePointsBodyAuthority(error)) return false;
    const auto fail=[&](const char* reason) { error=reason; return false; };
    auto caller=makeParticleBodyCallerLedger({},error);
    const auto source=particlePointsBodySourceFixture(caller,error);
    if (!source) return false;
    const auto sourceBytes=caller->usage().logicalBytes;
    auto request=particlePointsBodyRequestFixture(source,.5,error);
    const auto value=particlePointsBodyCoreFixture(*source,error);
    AdmissionContext context; context.deferredPoints.emplace(801,source->declaration().points);
    context.deferredBodyGeometry.emplace(802,source->declaration());
    const auto admitted=value ? admitValue(*value,particlePointsInstanceContract(),{},context,error):std::nullopt;
    if (!request || !admitted) return false;
    const auto wire=lowerRuntimePlan(particlePointsInstanceContract(),*admitted);
    ParticlePointsSceneRecordingBackend recorded(backend);
    NativeGeometryCoreCapabilitySource capabilities(recorded);
    std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> escapedFrame;
    std::shared_ptr<const arbitgpu::NativeFixtureSceneResources> escapedResources;
    std::weak_ptr<const ParticlePointsBodyFrame> completedWeak;
    std::weak_ptr<const ParticleBodyGeometrySample> sampleWeak;
    std::weak_ptr<const ParticlePointsBodyFrame> resourceCompletedWeak;
    std::weak_ptr<const ParticleBodyGeometrySample> resourceSampleWeak;
    std::size_t callbacks=0, legacyCallbacks=0;
    const PointsBodyEvaluator paired=[&](const auto& current,auto& diagnostic) {
        ++callbacks;
        auto result=ParticlePointsBodyFrame::replay(current,diagnostic);
        if (result) { completedWeak=result; sampleWeak=result->sample(); }
        return result;
    };
    const PointsEvaluator legacy=[&](auto,const auto&,const auto&,auto&,auto&) { ++legacyCallbacks; return false; };
    {
        GeometryCorePlanRuntime runtime(capabilities,1);
        auto preview=runtime.admitPointsBody(request->owner(),wire,{},source,error);
        if (!preview) return false;
        if (runtime.admitPreview(request->owner(),wire,{},context,error))
            return fail("Public Core body declaration conferred private immutable admission");
        if (executeNativeGeometry(recorded,*preview,64,64,error,false,{},nullptr,{},.5,legacy))
            return fail("Native body declaration used reference geometry without its completed provider");
        arbitgpu::NativeFixtureSceneRuntimeInputs inputs; inputs.canonicalBlockCFrame=request->parameters().bodyReplayContext->frame;
        const auto draw=executeNativeGeometry(recorded,*preview,64,64,error,false,inputs,nullptr,{},.5,legacy,nullptr,
            std::numeric_limits<double>::quiet_NaN(),nullptr,request,paired);
        if (!draw || callbacks!=1 || legacyCallbacks!=0 || draw->drawnInstanceIds!=std::vector<StableId>{13,22}
            || draw->stats.instancedDrawCount==0 || !recorded.lastInstanceScene)
            return fail("Native paired provider must replay once, skip legacy callback and retain canonical instance IDs");
        const auto& scene=*recorded.lastInstanceScene;
        if (scene.vertexCount!=5 || scene.indexCount!=18
            || std::abs(scene.vertices[4].position.x-.5f*.5773502691896257645f)>2.e-6f)
            return fail("Actual native prepared scene did not consume the terminal tip-only normalized body");
        const auto pixels=readPixels(draw->frame);
        if (pixels.empty()) return fail("Native paired draw produced no readable frame");
        auto exporting=particlePointsBodyRequestFixture(source,.5,error,PlanUse::exportRender);
        auto exported=runtime.admitPointsBody(exporting->owner(),wire,{},source,error);
        // Capacity one keeps the live Preview lease. Refusal must preserve it.
        if (exported || !preview->plan) return fail("Live paired Preview owner was evicted for Export");
        auto wrong=particlePointsBodyRequestFixture(source,.5,error,PlanUse::exportRender);
        if (executeNativeGeometry(recorded,*preview,64,64,error,false,inputs,nullptr,{},.5,legacy,nullptr,
            std::numeric_limits<double>::quiet_NaN(),nullptr,wrong,paired) || callbacks!=1)
            return fail("Wrong current use must refuse before invoking the paired callback");
        auto substitute=*preview; substitute.sourceContext.deferredBodyGeometry[802].sourceRevision=8;
        if (executeNativeGeometry(recorded,substitute,64,64,error,false,inputs,nullptr,{},.5,legacy,nullptr,
            std::numeric_limits<double>::quiet_NaN(),nullptr,request,paired) || callbacks!=1)
            return fail("Substituted native declaration must refuse before invoking the paired callback");
        if (executeNativeGeometry(recorded,*preview,63,64,error,false,inputs,nullptr,{},.5,legacy,nullptr,
            std::numeric_limits<double>::quiet_NaN(),nullptr,request,paired) || callbacks!=1)
            return fail("Wrong native extent must refuse before replay");
        const auto newDecodedSource=particlePointsBodySourceFixture(caller,error);
        if (!newDecodedSource || runtime.admitPointsBody(request->owner(),wire,{},newDecodedSource,error))
            return fail("Owner cache silently reused an old token for a newly decoded source capability");
        ResourceLimits shortCore; shortCore.maxBufferBytes=1;
        if (runtime.admitPointsBody(request->owner(),wire,shortCore,source,error))
            return fail("Paired Core prospective admission accepted a short storage cap");
        ParticlePointsBodyEnvelope admissionEnvelope,frameEnvelope;
        if (!particlePointsBodyCoreEnvelope(source->declaration(),wire.size(),false,admissionEnvelope)
            || !particlePointsBodyCoreEnvelope(source->declaration(),0,true,frameEnvelope))
            return fail("Paired native cost recipes did not derive");
        ParticlePointsBodyEnvelope conversionEnvelope;
        std::size_t replayPeak=0;
        if (!ParticlePointsBodyFrame::replayEnvelope(*request,conversionEnvelope,replayPeak,error)) return false;
        const auto leaseHeader=sizeof(ParticleBodyCallerLease);
        const auto baselineBytes=caller->usage().logicalBytes;
        const auto incumbent=preview->nativeResources->value;
        const auto preparations=recorded.instancePreparations;
        const auto sourceCap=baselineBytes+admissionEnvelope.bytes+leaseHeader;
        {
            ResourceLimits sourceShort; sourceShort.maxBufferBytes=sourceCap-1;
            GeometryCorePlanRuntime bounded(capabilities);
            if (bounded.admitPointsBody(request->owner(),wire,sourceShort,source,error)
                || caller->usage().logicalBytes!=baselineBytes)
                return fail("Source aggregate one byte below its lease header must refuse without new residency");
            sourceShort.maxBufferBytes=sourceCap;
            if (!bounded.admitPointsBody(request->owner(),wire,sourceShort,source,error))
                return fail("Source aggregate exact cap including its lease header must admit");
        }
        const auto exactCap=sourceCap+frameEnvelope.bytes+leaseHeader+replayPeak;
        const auto refuseAlteredSources=[&](RuntimeAdmission& issued,const auto& current,const auto& currentInputs) {
            if (!issued.sourceContext.admittedGeometryOrSceneSources.empty())
                return fail("Genuine paired issuer must start with an empty admitted source set");
            const auto plan=issued.plan;
            const auto ownerLease=issued.ownerLease;
            const auto resources=issued.nativeResources;
            const auto publication=resources->value;
            const auto resident=caller->usage().logicalBytes;
            const auto prepared=recorded.instancePreparations;
            const auto legacyCalls=legacyCallbacks;
            std::size_t refusedCallbacks=0;
            const PointsBodyEvaluator forbidden=[&](const auto& candidate,auto& diagnostic) {
                ++refusedCallbacks; return ParticlePointsBodyFrame::replay(candidate,diagnostic);
            };
            for (const auto count:std::array<std::size_t,2>{1,4096}) {
                for (std::size_t i=0;i<count;++i)
                    issued.sourceContext.admittedGeometryOrSceneSources.insert(static_cast<StableId>(10000+i));
                for (int retry=0;retry<2;++retry) {
                    const auto work=caller->usage().work;
                    if (executeNativeGeometry(recorded,issued,64,64,error,false,currentInputs,nullptr,{},.5,legacy,nullptr,
                        std::numeric_limits<double>::quiet_NaN(),nullptr,current,forbidden)
                        || refusedCallbacks!=0 || legacyCallbacks!=legacyCalls || recorded.instancePreparations!=prepared
                        || issued.plan!=plan || issued.ownerLease!=ownerLease || issued.nativeResources!=resources
                        || resources->value!=publication || preview->nativeResources->value!=incumbent
                        || caller->usage().logicalBytes!=resident || caller->usage().work<=work
                        || issued.sourceContext.admittedGeometryOrSceneSources.size()!=count)
                        return fail("Altered issued paired source set must refuse before callbacks/preparation and preserve owners, storage and failed work");
                }
                issued.sourceContext.admittedGeometryOrSceneSources.clear();
            }
            return true;
        };
        // This cap fits the former source/frame-only envelope, including both
        // native lease headers, but cannot fit the fresh authenticated replay.
        for (const auto cap:std::array<std::size_t,2>{sourceCap+frameEnvelope.bytes+leaseHeader,exactCap-1}) {
            ResourceLimits shortage; shortage.maxBufferBytes=cap;
            GeometryCorePlanRuntime bounded(capabilities);
            auto shortFrame=bounded.admitPointsBody(request->owner(),wire,shortage,source,error);
            if (!shortFrame) return fail("Short replay interval must still admit its native source envelope");
            const auto resident=caller->usage().logicalBytes;
            const auto work=caller->usage().work;
            if (executeNativeGeometry(recorded,*shortFrame,64,64,error,false,inputs,nullptr,{},.5,legacy,nullptr,
                std::numeric_limits<double>::quiet_NaN(),nullptr,request,paired) || callbacks!=1
                || recorded.instancePreparations!=preparations || preview->nativeResources->value!=incumbent
                || caller->usage().logicalBytes!=resident || caller->usage().work<=work)
                return fail("Complete replay shortage must precede callback/preparation and preserve incumbent publication");
            const auto attempted=caller->usage().work;
            if (executeNativeGeometry(recorded,*shortFrame,64,64,error,false,inputs,nullptr,{},.5,legacy,nullptr,
                std::numeric_limits<double>::quiet_NaN(),nullptr,request,paired) || callbacks!=1
                || recorded.instancePreparations!=preparations || preview->nativeResources->value!=incumbent
                || caller->usage().logicalBytes!=resident || caller->usage().work<=attempted)
                return fail("Failed native frame retries must retain work without callback, preparation or publication");
        }
        {
            ResourceLimits exact; exact.maxBufferBytes=exactCap;
            GeometryCorePlanRuntime bounded(capabilities);
            auto exactFrame=bounded.admitPointsBody(request->owner(),wire,exact,source,error);
            if (!exactFrame || !refuseAlteredSources(*exactFrame,request,inputs)) return false;
            std::size_t freshCallbacks=0;
            const PointsBodyEvaluator fresh=[&](const auto& current,auto& diagnostic) {
                ++freshCallbacks; return ParticlePointsBodyFrame::replay(current,diagnostic);
            };
            const auto exactDraw=exactFrame ? executeNativeGeometry(recorded,*exactFrame,64,64,error,false,inputs,nullptr,{},.5,
                legacy,nullptr,std::numeric_limits<double>::quiet_NaN(),nullptr,request,fresh):std::nullopt;
            if (!exactDraw || freshCallbacks!=1 || recorded.instancePreparations!=preparations+1
                || readPixels(exactDraw->frame)!=pixels || preview->nativeResources->value!=incumbent)
                return fail("Exact complete aggregate cap must execute a genuine fresh replay and preserve other owners");
        }
        {
            ResourceLimits exact; exact.maxBufferBytes=exactCap;
            GeometryCorePlanRuntime bounded(capabilities);
            auto attackFrame=bounded.admitPointsBody(request->owner(),wire,exact,source,error);
            if (!attackFrame) return false;
            const auto resident=caller->usage().logicalBytes;
            const auto work=caller->usage().work;
            const auto prepared=recorded.instancePreparations;
            std::shared_ptr<const ParticlePointsBodyFrame> maliciousEscape;
            bool secondRefused=false;
            const PointsBodyEvaluator doubleReplay=[&](const auto& current,auto& diagnostic) {
                maliciousEscape=ParticlePointsBodyFrame::replay(current,diagnostic);
                if (!maliciousEscape) return std::shared_ptr<const ParticlePointsBodyFrame>{};
                auto second=ParticlePointsBodyFrame::replay(current,diagnostic);
                secondRefused=!second;
                return second;
            };
            if (executeNativeGeometry(recorded,*attackFrame,64,64,error,false,inputs,nullptr,{},.5,legacy,nullptr,
                std::numeric_limits<double>::quiet_NaN(),nullptr,request,doubleReplay) || !maliciousEscape || !secondRefused
                || recorded.instancePreparations!=prepared || preview->nativeResources->value!=incumbent
                || caller->usage().logicalBytes<=resident || caller->usage().logicalBytes>exactCap
                || caller->usage().work<=work)
                return fail("Scoped Core ceiling must refuse a retained second genuine replay before native publication");
            const auto escapedWork=caller->usage().work;
            maliciousEscape.reset();
            if (caller->usage().logicalBytes!=resident || caller->usage().work!=escapedWork)
                return fail("Escaped callback retirement must release only storage and preserve failed work");
        }
        // Isolate each public geometry escape. In particular Admission is
        // copyable, so retaining only a copy must keep the exact same debit.
        for (int escapeKind=0;escapeKind<3;++escapeKind) {
            ResourceLimits exact; exact.maxBufferBytes=exactCap;
            GeometryCorePlanRuntime bounded(capabilities);
            auto attackFrame=bounded.admitPointsBody(request->owner(),wire,exact,source,error);
            if (!attackFrame) return false;
            const auto resident=caller->usage().logicalBytes;
            const auto prepared=recorded.instancePreparations;
            std::shared_ptr<const ParticleBodyGeometrySample> escapedSample;
            std::shared_ptr<const ParticleBodyGeometrySampler> escapedSampler;
            std::shared_ptr<const ParticleBodyGeometryAdmission> escapedAdmissionCopy;
            std::size_t attempts=0;
            bool refused=false;
            const PointsBodyEvaluator escapeGeometry=[&](const auto& current,auto& diagnostic) {
                auto first=ParticlePointsBodyFrame::replay(current,diagnostic);
                if (!first) return std::shared_ptr<const ParticlePointsBodyFrame>{};
                if (escapeKind==0) escapedSample=first->sample();
                else if (escapeKind==1) escapedSampler=first->receipt()->sampler();
                else escapedAdmissionCopy=std::make_shared<const ParticleBodyGeometryAdmission>(*first->sample()->admission());
                first.reset();
                for (int retry=0;retry<2;++retry) {
                    const auto work=caller->usage().work;
                    auto second=ParticlePointsBodyFrame::replay(current,diagnostic);
                    ++attempts;
                    if (second || caller->usage().work<=work) return second;
                }
                refused=true;
                return std::shared_ptr<const ParticlePointsBodyFrame>{};
            };
            if (executeNativeGeometry(recorded,*attackFrame,64,64,error,false,inputs,nullptr,{},.5,legacy,nullptr,
                std::numeric_limits<double>::quiet_NaN(),nullptr,request,escapeGeometry) || !refused || attempts!=2
                || !(escapedSample || escapedSampler || escapedAdmissionCopy)
                || recorded.instancePreparations!=prepared || preview->nativeResources->value!=incumbent
                || caller->usage().logicalBytes<=resident || caller->usage().logicalBytes>exactCap)
                return fail("Sample, sampler or copied admission escape must keep the shared debit and refuse repeated fresh replay");
            const auto failedWork=caller->usage().work;
            escapedSample.reset(); escapedSampler.reset(); escapedAdmissionCopy.reset();
            if (caller->usage().logicalBytes!=resident || caller->usage().work!=failedWork)
                return fail("Geometry-only escape retirement must release storage once without refunding failed work");
        }
        escapedResources=draw->resources;
        resourceCompletedWeak=completedWeak; resourceSampleWeak=sampleWeak;
        {
            GeometryCorePlanRuntime separateExport(capabilities);
            auto exportRequest=particlePointsBodyRequestFixture(source,.5,error,PlanUse::exportRender);
            if (!exportRequest) return false;
            ParticlePointsBodyEnvelope exportConversionEnvelope;
            std::size_t exportReplayPeak=0;
            if (!ParticlePointsBodyFrame::replayEnvelope(*exportRequest,exportConversionEnvelope,exportReplayPeak,error))
                return false;
            auto exportExactCap=caller->usage().logicalBytes;
            if (!checkedAdd(exportExactCap,admissionEnvelope.bytes,exportExactCap)
                || !checkedAdd(exportExactCap,leaseHeader,exportExactCap)
                || !checkedAdd(exportExactCap,frameEnvelope.bytes,exportExactCap)
                || !checkedAdd(exportExactCap,leaseHeader,exportExactCap)
                || !checkedAdd(exportExactCap,exportReplayPeak,exportExactCap))
                return fail("Separate native Export exact cap arithmetic overflow");
            ResourceLimits exact; exact.maxBufferBytes=exportExactCap;
            auto exportPlan=separateExport.admitPointsBody(exportRequest->owner(),wire,exact,source,error);
            arbitgpu::NativeFixtureSceneRuntimeInputs exportInputs;
            exportInputs.canonicalBlockCFrame=exportRequest->parameters().bodyReplayContext->frame;
            if (!exportPlan || !refuseAlteredSources(*exportPlan,exportRequest,exportInputs)) return false;
            const auto exportDraw=exportPlan ? executeNativeGeometry(recorded,*exportPlan,64,64,error,false,exportInputs,
                nullptr,{},.5,legacy,nullptr,std::numeric_limits<double>::quiet_NaN(),nullptr,exportRequest,paired):std::nullopt;
            if (!exportDraw || callbacks!=2 || readPixels(exportDraw->frame)!=pixels
                || exportDraw->drawnInstanceIds!=draw->drawnInstanceIds)
                return fail("Separate current native Export must preserve Preview paired pixels and identities");
        }
        auto replacement=executeNativeGeometry(recorded,*preview,64,64,error,false,inputs,nullptr,{},.5,legacy,nullptr,
            std::numeric_limits<double>::quiet_NaN(),nullptr,request,paired);
        if (!replacement || callbacks!=3 || readPixels(replacement->frame)!=pixels)
            return fail("Repeated current replay changed native paired pixels");
        escapedFrame=replacement->frame;
        if (preview->nativeResources->value.get()!=replacement->resources.get())
            return fail("Native resource cache did not publish the owner-preserving alias");
    }
    recorded.lastInstanceScene.reset(); request.reset();
    if (completedWeak.expired() || sampleWeak.expired() || resourceCompletedWeak.expired() || resourceSampleWeak.expired()
        || caller->usage().logicalBytes<=sourceBytes)
        return fail("Escaped native frame/resources lost the completed replay or refunded its storage early");
    const auto alive=caller->usage().logicalBytes;
    escapedFrame.reset();
    if (!completedWeak.expired() || !sampleWeak.expired() || resourceCompletedWeak.expired() || resourceSampleWeak.expired()
        || caller->usage().logicalBytes>=alive)
        return fail("Frame-only retirement or resource-only charged body retention was incorrect");
    const auto resourceAlive=caller->usage().logicalBytes;
    escapedResources.reset();
    if (!resourceCompletedWeak.expired() || !resourceSampleWeak.expired() || caller->usage().logicalBytes>=resourceAlive)
        return fail("Final native alias did not retire completed caller/sample storage");
    return verifyNativeParticlePointsRuntimeResidency(backend,readPixels,error);
}

namespace particlebodyretainedfixture {
struct DisposalState final { unsigned payload=0,resource=0; bool payloadHadResource=false; };
struct Resource final {
    DisposalState* state;
    ~Resource() { state->resource=++state->payload; }
};
struct Payload final {
    DisposalState* state;
    std::weak_ptr<const void> resource;
    Payload(DisposalState* s,std::weak_ptr<const void> r) : state(s),resource(std::move(r)) {}
    ~Payload() { state->payloadHadResource=!resource.expired(); ++state->payload; }
};
struct ThrowingPayload final {
    explicit ThrowingPayload(bool fail) { if (fail) throw std::bad_alloc(); }
};
template <typename T> struct RefusingAllocator final {
    using value_type=T;
    RefusingAllocator()=default;
    template <typename U> RefusingAllocator(const RefusingAllocator<U>&) {}
    T* allocate(std::size_t) { throw std::bad_alloc(); }
    void deallocate(T*,std::size_t) noexcept {}
};
} // namespace particlebodyretainedfixture

inline bool verifyParticleBodyRetainedResourceOracles(std::string& error)
{
    using namespace videowire::geometry;
    using namespace videorender;
    const auto fail=[&](const char* reason) { error=reason; return false; };
    // These generic holder checks exercise constructor and control-block disposal
    // without minting any private geometry authority in a callback.
    for (bool reverse : {false,true}) {
        particlebodyretainedfixture::DisposalState state;
        auto resource=std::make_shared<particlebodyretainedfixture::Resource>(); resource->state=&state;
        auto first=makeRetainedResourcePayload<particlebodyretainedfixture::Payload>(resource,&state,resource);
        auto second=first;
        resource.reset();
        if (reverse) second.reset(); else first.reset();
        if (state.payload || state.resource) return fail("First embedded alias refunded a live dependency");
        if (reverse) first.reset(); else second.reset();
        if (!state.payloadHadResource || state.payload!=2 || state.resource!=2)
            return fail("Embedded payload must retire before its resource dependency");
    }
    {
        auto resource=std::make_shared<int>(1); std::weak_ptr<int> weak=resource;
        bool threw=false;
        try { (void)makeRetainedResourcePayload<particlebodyretainedfixture::ThrowingPayload>(resource,true); }
        catch (const std::bad_alloc&) { threw=true; }
        resource.reset();
        if (!threw || !weak.expired()) return fail("Failed embedded construction retained its dependency");
    }
    {
        particlebodyretainedfixture::DisposalState state;
        auto resource=std::make_shared<particlebodyretainedfixture::Resource>(); resource->state=&state;
        using Holder=RetainedResourcePayload<particlebodyretainedfixture::Payload>;
        auto* raw=new Holder(resource,&state,resource);
        resource.reset(); bool threw=false;
        try { (void)std::shared_ptr<Holder>(raw,std::default_delete<Holder>{},particlebodyretainedfixture::RefusingAllocator<Holder>{}); }
        catch (const std::bad_alloc&) { threw=true; }
        if (!threw || !state.payloadHadResource || state.payload!=2 || state.resource!=2)
            return fail("Control-block refusal did not dispose embedded payload before dependency");
    }
    const auto descriptor=particlePointsTipBodyDescriptor(error);
    const auto prepared=descriptor ? particleBodyFixtureAdmitted(*descriptor,error):nullptr;
    if (!prepared) return false;
    const auto wire=encodeRuntimeValue(*prepared);
    const auto contract=particleBodyGeometryFixtureContract();
    auto legacy=decodeRuntimeValue(wire,contract,{}, {},error);
    auto empty=decodeRuntimeValueWithRetainedResources(wire,contract,{}, {},error,{});
    if (!legacy || !empty || encodeRuntimeValue(*legacy)!=wire || encodeRuntimeValue(*empty)!=wire
        || legacy->receipt().deterministicDigest!=empty->receipt().deterministicDigest)
        return fail("Empty retained dependencies changed legacy transport or receipt bytes");
    auto oldToken=std::make_shared<const AdmittedValue>(*legacy);
    ParticleBodyGeometryAttempt oldAttempt,emptyAttempt;
    auto oldBody=admitParticleBodyGeometry(oldToken,ParticleBodyGeometryProxy::sphere,3,{},oldAttempt,error);
    auto emptyBody=admitParticleBodyGeometryWithRetainedResources(oldToken,ParticleBodyGeometryProxy::sphere,3,{},emptyAttempt,error,{},{});
    if (!oldBody || !emptyBody || oldBody->recipe().factoryAttemptLogicalBytes!=emptyBody->recipe().factoryAttemptLogicalBytes
        || !equal(oldBody->referencePrototype(),emptyBody->referencePrototype())
        || oldBody->ownedDescriptor().operations.front().retainedMesh==oldToken->descriptor().operations.front().retainedMesh)
        return fail("Empty snapshot forwarding changed released factory behavior or independent copies");

    ParticleBodyGeometryScalarCensus scalar;
    scalar.operationCount=prepared->descriptor().operations.size();
    const auto& terminal=std::get<GeometryData>(prepared->descriptor().data);
    scalar.terminalVertices=terminal.positions.size(); scalar.terminalIndices=terminal.indices.size();
    scalar.terminalId=prepared->descriptor().stableId;
    for (std::size_t i=0;i<scalar.operationCount;++i) {
        const auto& op=prepared->descriptor().operations[i]; auto& record=scalar.operations[i];
        record.code=op.code; record.operationId=op.stableId; record.inputId=op.inputStableId;
        record.outputId=op.outputStableId; record.secondaryId=op.secondaryInputStableId; record.auxiliaryId=op.field.auxiliaryInputStableId;
        record.emptyPayloads=true;
        if (op.code==OperationCode::retainedMesh || op.code==OperationCode::vectorDisplacement) {
            record.vertices=5; record.indices=18;
        } else record.fields=5;
        if (op.retainedMesh) { record.retainedVertices=5; record.retainedIndices=18; }
    }
    ParticleBodyGeometryScalarCost affine,numerical;
    ParticleBodyGeometryCostRefusal refusal;
    ParticleBodyGeometryOwnedCapacityCensus capacity;
    if (!ParticleBodyGeometryAdmission::deriveScalarCost(scalar,3,ParticleBodyGeometryProxy::sphere,affine,refusal)
        || !ParticleBodyGeometryAdmission::observeOwnedCapacities(*oldToken,capacity,refusal)
        || !ParticleBodyGeometryAdmission::instantiateScalarCost(*oldToken,affine,capacity,numerical,refusal))
        return fail("Scalar and genuine-capacity factory cost derivation refused the Timeline pyramid");
    const auto& released=oldBody->recipe();
    const auto sameWork=[](const ParticleBodyGeometryWork& a,const ParticleBodyGeometryWork& b) {
        return a.operationRecords==b.operationRecords && a.shapeProbes==b.shapeProbes && a.dependencyEdges==b.dependencyEdges
            && a.positionElements==b.positionElements && a.identityElements==b.identityElements && a.indexElements==b.indexElements
            && a.fieldElements==b.fieldElements && a.fieldComponents==b.fieldComponents && a.stringBytes==b.stringBytes;
    };
    if (numerical.sourceLogicalBytes!=released.sourceLogicalBytes || numerical.retainedLogicalBytes!=released.retainedLogicalBytes
        || numerical.factoryAttemptLogicalBytes!=released.factoryAttemptLogicalBytes
        || numerical.descriptorCopyLogicalBytes!=released.descriptorCopyLogicalBytes
        || numerical.retainedSourceSnapshotLogicalBytes!=released.retainedSourceSnapshotLogicalBytes
        || numerical.sampleLogicalBytes!=released.sampleLogicalBytes || numerical.censusLogicalBytes!=released.censusLogicalBytes
        || numerical.retainedSourceCoefficient!=1 || numerical.factorySourceCoefficient!=2
        || numerical.nativeDataBytes==0 || numerical.nativeBodyBytes!=sizeof(ParticleBodyGeometryAdmission)
        || !sameWork(numerical.census,released.census) || !sameWork(numerical.referenceValidation,released.referenceValidation)
        || !sameWork(numerical.referenceNormalization,released.referenceNormalization) || !sameWork(numerical.publication,released.publication)
        || numerical.owningFactoryAttemptLogicalBytes!=released.factoryAttemptLogicalBytes
            +numerical.factoryDependencyLogicalBytes+sizeof(RetainedResourcePayload<RetainedMeshData>)-sizeof(RetainedMeshData))
        return fail("Scalar factory recipe changed the released capacity envelope or named factory work");
    if (capacity.positions!=std::get<GeometryData>(oldToken->descriptor().data).positions.capacity()
        || capacity.records[0].attributeName!=oldToken->descriptor().operations[0].field.attributeName.capacity()
        || capacity.records[0].importedAnimation!=oldToken->descriptor().operations[0].retainedMesh->importedAnimation.capacity()
        || ParticleBodyGeometryAdmission::instantiateScalarCost(*prepared,affine,capacity,numerical,refusal))
        return fail("Capacity census did not observe exact containers or reject source substitution");
    auto excess=scalar; excess.operationCount=33;
    if (ParticleBodyGeometryAdmission::deriveScalarCost(excess,3,ParticleBodyGeometryProxy::sphere,numerical,refusal))
        return fail("Scalar factory census accepted more than 32 records");
    excess=scalar; excess.operations[0].retainedVertices=std::numeric_limits<std::size_t>::max();
    if (ParticleBodyGeometryAdmission::deriveScalarCost(excess,3,ParticleBodyGeometryProxy::sphere,numerical,refusal))
        return fail("Scalar factory census accepted hostile retained cardinalities");

    // A consumer refusal occurs after real parsing. Capturing its parsed alias
    // observes authority input only; the callback never constructs a token.
    for (bool rejectContract : {false,true}) {
        auto caller=makeParticleBodyCallerLedger({},error); if (!caller) return false;
        const auto base=caller->usage().logicalBytes;
        auto lease=caller->reserve(1024*1024,17); if (!lease) return false;
        const auto resident=caller->usage().logicalBytes;
        std::shared_ptr<const RetainedMeshData> parsed;
        auto bad=contract; if (rejectContract) bad.maxVertices=1;
        auto rejected=decodeRuntimeValueWithRetainedResources(wire,bad,{}, {},error,lease,
            [&](const OperationRecord& op,std::string&) { if (op.retainedMesh) parsed=op.retainedMesh; return rejectContract; });
        lease.reset();
        if (rejected || !parsed || caller->usage().logicalBytes!=resident
            || !equal(parsed->geometry,prepared->descriptor().operations.front().retainedMesh->geometry))
            return fail("Refused decoder or Core attempt refunded an escaping parsed owner");
        parsed.reset();
        if (caller->usage().logicalBytes!=base || caller->usage().work!=17)
            return fail("Refused parsed owner retirement refunded work or leaked its lease");
    }

    // Each token/reference/compatibility/binding handle independently owns P1.
    for (unsigned escape=0;escape<4;++escape) {
        auto caller=makeParticleBodyCallerLedger({},error); if (!caller) return false;
        const auto base=caller->usage().logicalBytes;
        auto lease=caller->reserve(1024*1024,19); if (!lease) return false;
        auto decoded=decodeRuntimeValueWithRetainedResources(wire,contract,{}, {},error,lease); if (!decoded) return false;
        auto holder=retainParticleBodyCallerPayload<std::optional<AdmittedValue>>(lease,std::move(decoded));
        auto token=std::shared_ptr<const AdmittedValue>(holder,&holder->value());
        auto reference=std::shared_ptr<const GeometryData>(holder,&std::get<GeometryData>(holder->value().descriptor().data));
        auto compatibility=retainParticleBodyCallerPayload<ValueDescriptor>(lease,token->descriptor());
        auto binding=retainParticleBodyCallerPayload<ParticleGeometryBinding>(lease);
        binding->body=compatibility; binding->bodySource=token;
        holder.reset(); decoded.reset(); lease.reset();
        if (escape!=0) token.reset();
        if (escape!=1) reference.reset();
        if (escape!=2) compatibility.reset();
        if (escape!=3) binding.reset();
        if (caller->usage().logicalBytes<=base) return fail("An independent P1 outer alias lost its lease");
        token.reset(); reference.reset(); compatibility.reset(); binding.reset();
        if (caller->usage().logicalBytes!=base || caller->usage().work!=19)
            return fail("P1 outer alias retirement did not refund exactly once");
    }

    // Distinct parsed/clone/snapshot owners and both alias death orders preserve
    // P1 and P2 independently, including a snapshot surviving its whole body.
    for (unsigned order=0;order<4;++order) {
        auto caller=makeParticleBodyCallerLedger({},error); if (!caller) return false;
        const auto base=caller->usage().logicalBytes;
        auto p1=caller->reserve(1024*1024,23); if (!p1) return false;
        const auto p1Bytes=caller->usage().logicalBytes;
        std::shared_ptr<const RetainedMeshData> parsed;
        auto decoded=decodeRuntimeValueWithRetainedResources(wire,contract,{}, {},error,p1,
            [&](const OperationRecord& op,std::string&) { if (op.retainedMesh) parsed=op.retainedMesh; return true; });
        if (!decoded || !parsed) return false;
        auto holder=retainParticleBodyCallerPayload<std::optional<AdmittedValue>>(p1,std::move(decoded));
        auto token=std::shared_ptr<const AdmittedValue>(holder,&holder->value());
        auto clone=token->descriptor().operations.front().retainedMesh;
        if (clone==parsed || !equal(clone->geometry,parsed->geometry)) return fail("Core clone did not independently retain exact parsed contents");
        std::weak_ptr<const RetainedMeshData> parsedWeak=parsed;
        auto parsedCopy=parsed; parsed.reset();
        if (caller->usage().logicalBytes!=p1Bytes) return fail("First parsed alias retirement refunded P1");
        parsedCopy.reset();
        if (!parsedWeak.expired()) return fail("Parsed allocation survived replacement without an actual owner");
        if (!ParticleBodyGeometryAdmission::observeOwnedCapacities(*token,capacity,refusal)
            || !ParticleBodyGeometryAdmission::instantiateScalarCost(*token,affine,capacity,numerical,refusal)) return false;
        auto p2=caller->reserve(numerical.owningFactoryAttemptLogicalBytes,29); if (!p2) return false;
        std::weak_ptr<const ParticleBodyCallerLease> p2Weak=p2;
        auto dependency=ParticleBodyGeometryAdmission::makeSnapshotResources(token,p2);
        ParticleBodyGeometryAttempt attempt;
        auto body=admitParticleBodyGeometryWithRetainedResources(token,ParticleBodyGeometryProxy::sphere,3,{},attempt,error,p2,dependency);
        if (!body) return false;
        auto bodyCopy=std::make_shared<const ParticleBodyGeometryAdmission>(*body);
        const auto* snapshotDescriptor=&body->ownedDescriptor();
        body.reset();
        if (&bodyCopy->ownedDescriptor()!=snapshotDescriptor || bodyCopy->source()!=token || p2Weak.expired())
            return fail("Copied body admission did not share its exact P1/P2-owned Data");
        body=bodyCopy; bodyCopy.reset();
        auto snapshot=body->ownedDescriptor().operations.front().retainedMesh;
        if (snapshot==clone || !equal(snapshot->geometry,clone->geometry)
            || !body->matchesExactSource(token,100000)) return fail("Owner-aware snapshot changed independent source comparison");
        auto cloneCopy=clone; auto snapshotCopy=snapshot;
        std::weak_ptr<const AdmittedValue> tokenWeak=token;
        holder.reset(); decoded.reset(); token.reset(); p1.reset(); p2.reset(); dependency.reset(); body.reset();
        const auto both=caller->usage().logicalBytes;
        if (order&1) cloneCopy.reset(); else clone.reset();
        if (order&2) snapshotCopy.reset(); else snapshot.reset();
        if (caller->usage().logicalBytes!=both || p2Weak.expired()) return fail("First clone or snapshot alias refunded a shared lease");
        if (order<2) {
            snapshot.reset(); snapshotCopy.reset();
            if (!p2Weak.expired() || !tokenWeak.expired() || caller->usage().logicalBytes!=p1Bytes)
                return fail("Snapshot-first retirement did not release only P2 and its source token");
            clone.reset(); cloneCopy.reset();
        } else {
            clone.reset(); cloneCopy.reset();
            if (p2Weak.expired() || tokenWeak.expired() || caller->usage().logicalBytes!=both)
                return fail("Clone-first retirement lost the snapshot's original P1 token");
            snapshot.reset(); snapshotCopy.reset();
        }
        if (!p2Weak.expired() || caller->usage().logicalBytes!=base || caller->usage().work!=52)
            return fail("Final retained alias retirement leaked storage or refunded work");
    }
    error.clear();
    return true;
}

inline bool verifyParticleBodyOwnedCapacityResetOracles(std::string& error)
{
    using namespace videowire::geometry;
    using namespace videorender;
    const auto fail=[&](const char* reason) { error=reason; return false; };
    const auto descriptor=particlePointsTipBodyDescriptor(error);
    const auto source=descriptor ? particleBodyFixtureAdmitted(*descriptor,error):nullptr;
    if (!source) return false;
    ParticleBodyGeometryOwnedCapacityCensus capacity;
    const auto poison=[&] {
        capacity.observedSource=source.get();
        capacity.operationCount=std::numeric_limits<std::size_t>::max();
        capacity.terminalVertices=std::numeric_limits<std::size_t>::max();
        capacity.terminalIndices=std::numeric_limits<std::size_t>::max();
        capacity.positions=std::numeric_limits<std::size_t>::max();
        capacity.identities=std::numeric_limits<std::size_t>::max();
        capacity.indices=std::numeric_limits<std::size_t>::max();
        capacity.operations=std::numeric_limits<std::size_t>::max();
        capacity.attributes=std::numeric_limits<std::size_t>::max();
        capacity.spectrumFields=std::numeric_limits<std::size_t>::max();
        capacity.scoreFields=std::numeric_limits<std::size_t>::max();
        for (auto& record:capacity.records) {
            record.attributeName=std::numeric_limits<std::size_t>::max();
            record.retained=true;
            record.positions=std::numeric_limits<std::size_t>::max();
            record.identities=std::numeric_limits<std::size_t>::max();
            record.indices=std::numeric_limits<std::size_t>::max();
            record.attributes=std::numeric_limits<std::size_t>::max();
            record.importedAnimation=std::numeric_limits<std::size_t>::max();
        }
    };
    const auto zeroRecord=[](const ParticleBodyGeometryOwnedCapacityRecord& record) {
        return record.attributeName==0 && !record.retained && record.positions==0
            && record.identities==0 && record.indices==0 && record.attributes==0 && record.importedAnimation==0;
    };
    if (particlebodycapacityreset::globalStores.size()!=11 || particlebodycapacityreset::recordStores.size()!=7
        || particlebodycapacityreset::recordLoopVisits!=capacity.records.size()
        || particlebodycapacityreset::scalarStores!=235
        || particlebodycapacityreset::directNativeByteSlots!=sizeof(capacity.observedSource)
            +(10+6*capacity.records.size())*sizeof(std::size_t)+capacity.records.size()*sizeof(bool))
        return fail("Capacity reset metadata does not describe exact native field stores and loop visits");
    poison();
    ParticleBodyGeometryCostRefusal refusal=ParticleBodyGeometryCostRefusal::sourceMismatch;
    if (!ParticleBodyGeometryAdmission::observeOwnedCapacities(*source,capacity,refusal)
        || refusal!=ParticleBodyGeometryCostRefusal::none)
        return fail("Poisoned capacity observer refused a genuine supported source");
    const auto& value=source->descriptor(); const auto& mesh=std::get<GeometryData>(value.data);
    if (capacity.observedSource!=source.get() || capacity.operationCount!=value.operations.size()
        || capacity.terminalVertices!=mesh.positions.size() || capacity.terminalIndices!=mesh.indices.size()
        || capacity.positions!=mesh.positions.capacity() || capacity.identities!=mesh.vertexIds.capacity()
        || capacity.indices!=mesh.indices.capacity() || capacity.operations!=value.operations.capacity()
        || capacity.attributes!=value.attributes.capacity() || capacity.spectrumFields!=value.spectrumFields.capacity()
        || capacity.scoreFields!=value.scoreFields.capacity())
        return fail("Capacity reset changed genuine source identity, cardinalities or global capacities");
    for (std::size_t i=0;i<capacity.records.size();++i) {
        const auto& record=capacity.records[i];
        if (i>=value.operations.size()) {
            if (!zeroRecord(record)) return fail("Unused capacity record retained poison after successful observation");
            continue;
        }
        const auto& op=value.operations[i];
        if (record.attributeName!=op.field.attributeName.capacity() || record.retained!=bool(op.retainedMesh))
            return fail("Observed operation name capacity or retained presence changed");
        if (op.retainedMesh) {
            const auto& retained=*op.retainedMesh;
            if (record.positions!=retained.geometry.positions.capacity() || record.identities!=retained.geometry.vertexIds.capacity()
                || record.indices!=retained.geometry.indices.capacity() || record.attributes!=retained.attributes.capacity()
                || record.importedAnimation!=retained.importedAnimation.capacity())
                return fail("Observed retained mesh capacities changed");
        } else if (record.positions || record.identities || record.indices || record.attributes || record.importedAnimation)
            return fail("Non-retained operation kept poisoned mesh capacities");
    }
    // Existing instance admission and transport issue a genuine non-Geometry3D
    // token, rejected by the observer's unchanged first refusal branch.
    const auto unsupported=particlePointsInstanceFixture(error);
    const auto context=particlePointsFixtureContext();
    const auto admitted=unsupported ? admitValue(*unsupported,particlePointsInstanceContract(),{},context,error):std::nullopt;
    const auto refusedSource=admitted ? decodeRuntimeValue(encodeRuntimeValue(*admitted),
        particlePointsInstanceContract(),{},context,error):std::nullopt;
    if (!refusedSource) return fail("Refused-source setup did not pass released instance admission and transport");
    poison(); refusal=ParticleBodyGeometryCostRefusal::none;
    if (ParticleBodyGeometryAdmission::observeOwnedCapacities(*refusedSource,capacity,refusal)
        || refusal!=ParticleBodyGeometryCostRefusal::unsupported || capacity.observedSource
        || capacity.operationCount || capacity.terminalVertices || capacity.terminalIndices
        || capacity.positions || capacity.identities || capacity.indices || capacity.operations
        || capacity.attributes || capacity.spectrumFields || capacity.scoreFields)
        return fail("Early genuine-source refusal did not clear every global capacity field");
    for (const auto& record:capacity.records)
        if (!zeroRecord(record)) return fail("Early genuine-source refusal did not clear all 32 capacity records");
    return true;
}

inline bool verifyParticleBodyOwnerNativeProofOracles(std::string& error)
{
    using namespace videowire::geometry;
    using namespace videorender;
    using particlebodygeometrydetail::PositiveReferenceHalf;
    const auto fail=[&](const char* reason) { error=reason; return false; };
    if (!PositiveReferenceHalf{}(1.0f) || PositiveReferenceHalf{}(0.0f)
        || PositiveReferenceHalf{}(-1.0f) || PositiveReferenceHalf{}(std::numeric_limits<float>::infinity())
        || PositiveReferenceHalf{}(-std::numeric_limits<float>::infinity())
        || PositiveReferenceHalf{}(std::numeric_limits<float>::quiet_NaN()))
        return fail("Actual reference-half predicate changed finite positive admission");
    static_assert(std::is_empty_v<PositiveReferenceHalf> && !noexcept(PositiveReferenceHalf{}(1.0f)),
        "Actual half predicate stays memberless with its original exception specification");
    // This is the one actual argument initialization term. It is additive to
    // each accepted W2, with no operator-body, J, O or library-copy duplication.
    const std::size_t positiveHalfCopySlots=sizeof(PositiveReferenceHalf);
    std::size_t twoFactoryInitializationSlots=0;
    if (!checkedAdd(positiveHalfCopySlots,positiveHalfCopySlots,twoFactoryInitializationSlots)
        || twoFactoryInitializationSlots/2!=sizeof(PositiveReferenceHalf))
        return fail("Two prospective factories lost the actual predicate initialization identity");

    const auto contract=particleBodyGeometryFixtureContract();
    const auto pyramid=particlePointsTipBodyDescriptor(error);
    const auto retained=pyramid ? particleBodyFixtureAdmitted(*pyramid,error):nullptr;
    const auto grid=lowerGrid(85001,85000,7,3,2,1.0f,contract,error);
    PortContract vector; vector.carrier=CarrierKind::field; vector.fieldDomain=Domain::vertex;
    vector.fieldValueType=ValueType::vector; vector.fieldInterpolation=Interpolation::constant; vector.maxFieldElements=6;
    auto scalarContract=vector; scalarContract.fieldValueType=ValueType::floatValue;
    const auto position=grid ? lowerElementField(*grid,85002,OperationCode::positionField,vector,error):std::nullopt;
    FieldOperationSettings settings; settings.domain=Domain::vertex; settings.valueType=ValueType::floatValue; settings.mode=5;
    const auto coordinate=position ? lowerConstructedField(85003,OperationCode::fieldVectorMeasure,settings,
        &*position,nullptr,nullptr,scalarContract,error):std::nullopt;
    settings.mode=1; settings.values={1,0,.375,.375,1,0,0,0};
    const auto timeline=coordinate ? lowerConstructedField(85004,OperationCode::fieldTimeline,settings,
        &*coordinate,nullptr,nullptr,scalarContract,error):std::nullopt;
    settings.valueType=ValueType::vector; settings.values={1,0,0,0,0,0,0,0};
    const auto offset=timeline ? lowerConstructedField(85005,OperationCode::fieldComposeVector,settings,
        &*timeline,nullptr,nullptr,vector,error):std::nullopt;
    const auto displaced=offset ? lowerVectorDisplacement(*grid,*offset,85006,1,contract,error):std::nullopt;
    const auto planar=displaced ? particleBodyFixtureAdmitted(*displaced,error):nullptr;
    if (!retained || !planar) return fail("Owner proof setup did not reach genuine pyramid and Grid Timeline admission");
    const auto sameWork=[](const ParticleBodyGeometryWork& a,const ParticleBodyGeometryWork& b) {
        return a.operationRecords==b.operationRecords && a.shapeProbes==b.shapeProbes && a.dependencyEdges==b.dependencyEdges
            && a.positionElements==b.positionElements && a.identityElements==b.identityElements && a.indexElements==b.indexElements
            && a.fieldElements==b.fieldElements && a.fieldComponents==b.fieldComponents && a.stringBytes==b.stringBytes;
    };
    for (const auto& token : {retained,planar}) {
        const auto& value=token->descriptor(); const auto& terminal=std::get<GeometryData>(value.data);
        ParticleBodyGeometryScalarCensus census;
        census.operationCount=value.operations.size(); census.terminalId=value.stableId;
        census.terminalVertices=terminal.positions.size(); census.terminalIndices=terminal.indices.size();
        for (std::size_t i=0;i<value.operations.size();++i) {
            const auto& op=value.operations[i]; auto& record=census.operations[i];
            record.code=op.code; record.operationId=op.stableId; record.inputId=op.inputStableId;
            record.outputId=op.outputStableId; record.secondaryId=op.secondaryInputStableId;
            record.auxiliaryId=op.field.auxiliaryInputStableId; record.parameters=op.parameters;
            record.fieldDomain=op.field.domain; record.fieldType=op.field.valueType;
            record.fieldMode=op.field.mode; record.fieldValues=op.field.values; record.emptyPayloads=true;
            if (op.code==OperationCode::retainedMesh || op.code==OperationCode::grid || op.code==OperationCode::vectorDisplacement) {
                record.vertices=terminal.positions.size(); record.indices=terminal.indices.size();
            } else record.fields=terminal.positions.size();
            if (op.retainedMesh) {
                record.retainedVertices=op.retainedMesh->geometry.positions.size();
                record.retainedIndices=op.retainedMesh->geometry.indices.size();
            }
        }
        ParticleBodyGeometryOwnedCapacityCensus capacity;
        ParticleBodyGeometryCostRefusal refusal;
        if (!ParticleBodyGeometryAdmission::observeOwnedCapacities(*token,capacity,refusal)
            || capacity.observedSource!=token.get() || capacity.positions!=terminal.positions.capacity()
            || capacity.identities!=terminal.vertexIds.capacity() || capacity.indices!=terminal.indices.capacity()
            || capacity.operations!=value.operations.capacity())
            return fail("Owner proof capacity facts lost their exact genuine token/container identities");
        for (const auto proxy : {ParticleBodyGeometryProxy::sphere,ParticleBodyGeometryProxy::fixedBox}) {
            ParticleBodyGeometryScalarCost affine,numerical;
            if (!ParticleBodyGeometryAdmission::deriveScalarCost(census,3,proxy,affine,refusal)
                || !ParticleBodyGeometryAdmission::instantiateScalarCost(*token,affine,capacity,numerical,refusal))
                return fail("Owner proof scalar/capacity recipe refused genuine supported metadata");
            std::size_t floor=0;
            for (const auto term : {sizeof(ParticleBodyGeometryAdmission),numerical.nativeDataBytes,sizeof(AdmittedValue),
                    sizeof(ValueDescriptor),sizeof(ValueDescriptor),sizeof(OperationRecord),sizeof(OperationRecord),
                    sizeof(OperationRecord),std::size_t{1},std::size_t{1},std::size_t{1},sizeof(GeometryData),
                    sizeof(std::array<particlebodygeometrydetail::Shape,32>)})
                if (!checkedAdd(floor,term,floor)) return fail("Owner proof fixture native floor overflowed");
            if (numerical.factoryAttemptLogicalBytes<floor || numerical.retainedSourceCoefficient!=1
                || numerical.factorySourceCoefficient!=2 || numerical.nativeBodyBytes!=sizeof(ParticleBodyGeometryAdmission)
                || numerical.censusLogicalBytes!=sizeof(std::array<particlebodygeometrydetail::Shape,32>))
                return fail("Owner proof floor or unchanged source/native coefficients disagree with genuine cost facts");
            // Existing empty objects are supplied explicitly on these two paths.
            // The third path reaches the actual omitted/default argument. These
            // are ordinary public factories, not synthetic proof/token issuers.
            const std::shared_ptr<const ParticleBodyCallerLease> callerResources;
            const RetainedResourceOwner snapshotResources;
            for (unsigned route=0;route<3;++route) {
                ParticleBodyGeometryAttempt attempt;
                const auto body=route==0 ? admitParticleBodyGeometryWithRetainedResources(token,proxy,3,{},attempt,error,
                        callerResources,snapshotResources)
                    : route==1 ? admitParticleBodyGeometry(token,proxy,3,{},attempt,error,callerResources)
                    : admitParticleBodyGeometry(token,proxy,3,{},attempt,error);
                const auto& recipe=attempt.recipe;
                if (recipe.sourceLogicalBytes!=numerical.sourceLogicalBytes
                    || recipe.retainedLogicalBytes!=numerical.retainedLogicalBytes
                    || recipe.factoryAttemptLogicalBytes!=numerical.factoryAttemptLogicalBytes
                    || recipe.descriptorCopyLogicalBytes!=numerical.descriptorCopyLogicalBytes
                    || recipe.retainedSourceSnapshotLogicalBytes!=numerical.retainedSourceSnapshotLogicalBytes
                    || recipe.sampleLogicalBytes!=numerical.sampleLogicalBytes
                    || !sameWork(recipe.census,numerical.census) || !sameWork(recipe.referenceValidation,numerical.referenceValidation)
                    || !sameWork(recipe.referenceNormalization,numerical.referenceNormalization)
                    || !sameWork(recipe.publication,numerical.publication))
                    return fail("Direct/explicit/default factory changed released capacities, recipe or named work");
                if (token==planar && proxy==ParticleBodyGeometryProxy::fixedBox) {
                    if (body || !attempt.referenceReserved || attempt.publicationReserved || attempt.published
                        || error!="Body geometry fixed box requires three positive reference half ratios")
                        return fail("Planar Grid fixedBox changed its real positive-half refusal boundary");
                } else if (!body || !attempt.published || body->source()!=token
                    || body->proxy()!=proxy || body->structuralRevision()!=value.sourceRevision
                    || body->referencePrototype().vertexIds!=terminal.vertexIds || body->referencePrototype().indices!=terminal.indices
                    || !body->matchesExactSource(token,100000))
                    return fail("Owner proof genuine factory lost exact source, topology, revision or publication");
                if (body && token==retained && (body->ownedDescriptor().operations.front().retainedMesh
                    ==value.operations.front().retainedMesh || !equal(body->ownedDescriptor().operations.front().retainedMesh->geometry,
                        value.operations.front().retainedMesh->geometry)))
                    return fail("Owner proof retained factory replaced its independent snapshot with a source alias");
            }
        }
    }
    // Existing genuine replay oracles retain the ordered tip mesh, all legacy
    // Points fields and the independent intrinsic7/structural3 revisions.
    return verifyParticlePointsBodyAuthority(error);
}

inline bool verifyParticleBodyMinimumNativeProofOracles(std::string& error)
{
    using namespace videowire::geometry;
    using namespace videorender;
    const auto fail=[&](const char* reason) { error=reason; return false; };
    const auto contract=particleBodyGeometryFixtureContract();
    // All lowering, admission and serialization precede the factory attempts.
    const auto gridValue=lowerGrid(86001,86000,7,2,2,1.0f,contract,error);
    const auto grid=gridValue ? particleBodyFixtureAdmitted(*gridValue,error):nullptr;
    const auto retainedValue=[](bool single) {
        GeometryData mesh;
        mesh.positions={{-1,-2,-3}}; mesh.vertexIds={101}; mesh.indices={0,0,0};
        if (!single) {
            mesh.positions.push_back({1,2,3}); mesh.vertexIds.push_back(103); mesh.indices={0,1,0};
        }
        auto retained=std::make_shared<RetainedMeshData>(); retained->geometry=mesh;
        ValueDescriptor value; value.carrier=CarrierKind::geometry3D;
        value.stableId=single ? 86003:86002; value.sourceStableId=86000; value.sourceRevision=7;
        value.data=mesh; value.dispatchCount=1;
        OperationRecord op{value.stableId,OperationCode::retainedMesh,value.sourceStableId,value.stableId,{}};
        op.retainedMesh=std::move(retained); value.operations={std::move(op)};
        return value;
    };
    const auto diagonal=particleBodyFixtureAdmitted(retainedValue(false),error);
    const auto single=particleBodyFixtureAdmitted(retainedValue(true),error);
    if (!grid || !diagonal || !single)
        return fail("Minimum sources did not pass genuine Core admission and transport");
    for (const auto& token : {grid,diagonal,single}) {
        const auto& value=token->descriptor(); const auto& mesh=std::get<GeometryData>(value.data);
        const auto vertices=token==grid ? std::size_t{4}:token==diagonal ? std::size_t{2}:std::size_t{1};
        const auto indices=token==grid ? std::size_t{6}:std::size_t{3};
        if (value.operations.size()!=1 || value.dispatchCount!=1 || value.sourceStableId!=86000
            || value.sourceRevision!=7 || mesh.positions.size()!=vertices || mesh.vertexIds.size()!=vertices
            || mesh.indices.size()!=indices || !value.attributes.empty() || !value.spectrumFields.empty()
            || !value.scoreFields.empty())
            return fail("Minimum source changed its exact one-operation cardinalities or exclusions");
        const auto& op=value.operations.front();
        if (op.outputStableId!=value.stableId || op.inputStableId!=value.sourceStableId
            || !op.field.attributeName.empty() || op.sdfSurface
            || (token==grid && (op.code!=OperationCode::grid || op.retainedMesh
                || op.parameters!=std::array<float,4>{2,2,1,0}))
            || (token!=grid && (op.code!=OperationCode::retainedMesh || !op.retainedMesh
                || !equal(op.retainedMesh->geometry,mesh) || !op.retainedMesh->attributes.empty()
                || !op.retainedMesh->importedAnimation.empty())))
            return fail("Minimum source changed its genuine root or retained payload");
        if ((token==diagonal && (mesh.vertexIds!=std::vector<StableId>{101,103}
                || mesh.indices!=std::vector<std::uint32_t>{0,1,0}
                || !equal(mesh.positions[0],Vec3{-1,-2,-3}) || !equal(mesh.positions[1],Vec3{1,2,3})))
            || (token==single && (mesh.vertexIds!=std::vector<StableId>{101}
                || mesh.indices!=std::vector<std::uint32_t>{0,0,0})))
            return fail("Retained minimum lost distinct diagonal vertices or repeated triangle corners");
        ParticleBodyGeometryOwnedCapacityCensus capacity;
        ParticleBodyGeometryCostRefusal refusal;
        if (!ParticleBodyGeometryAdmission::observeOwnedCapacities(*token,capacity,refusal)
            || refusal!=ParticleBodyGeometryCostRefusal::none || capacity.observedSource!=token.get()
            || capacity.operationCount!=1 || capacity.terminalVertices!=vertices || capacity.terminalIndices!=indices
            || capacity.positions!=mesh.positions.capacity() || capacity.identities!=mesh.vertexIds.capacity()
            || capacity.indices!=mesh.indices.capacity() || capacity.operations!=value.operations.capacity()
            || capacity.attributes!=value.attributes.capacity() || capacity.spectrumFields!=value.spectrumFields.capacity()
            || capacity.scoreFields!=value.scoreFields.capacity()
            || capacity.records[0].attributeName!=op.field.attributeName.capacity()
            || capacity.records[0].retained!=bool(op.retainedMesh))
            return fail("Minimum capacity facts lost exact genuine source identity or containers");
        const auto& record=capacity.records[0];
        if (op.retainedMesh && (record.positions!=op.retainedMesh->geometry.positions.capacity()
            || record.identities!=op.retainedMesh->geometry.vertexIds.capacity()
            || record.indices!=op.retainedMesh->geometry.indices.capacity()
            || record.attributes!=op.retainedMesh->attributes.capacity()
            || record.importedAnimation!=op.retainedMesh->importedAnimation.capacity()))
            return fail("Minimum retained capacity facts differ from the genuine clone");
        // Use actual capacities for SC and sizes for X/A/R. Q is observed
        // nonnegative excess; neither capacity==size nor Q==0 is assumed.
        std::size_t sourceBytes=sizeof(ValueDescriptor),terminalBytes=0,retainedBytes=0,descriptorBytes=sizeof(ValueDescriptor);
        if (!addArrayBytes(capacity.positions,sizeof(Vec3),sourceBytes)
            || !addArrayBytes(capacity.identities,sizeof(StableId),sourceBytes)
            || !addArrayBytes(capacity.indices,sizeof(std::uint32_t),sourceBytes)
            || !addArrayBytes(capacity.operations,sizeof(OperationRecord),sourceBytes)
            || !addArrayBytes(capacity.attributes,sizeof(AttributeData),sourceBytes)
            || !addArrayBytes(capacity.spectrumFields,sizeof(spectrum::Binding),sourceBytes)
            || !addArrayBytes(capacity.scoreFields,sizeof(scorefield::Binding),sourceBytes)
            || !checkedAdd(sourceBytes,record.attributeName,sourceBytes) || !checkedAdd(sourceBytes,1,sourceBytes)
            || !addArrayBytes(vertices,sizeof(Vec3)+sizeof(StableId),terminalBytes)
            || !addArrayBytes(indices,sizeof(std::uint32_t),terminalBytes)
            || !checkedAdd(descriptorBytes,terminalBytes,descriptorBytes)
            || !checkedAdd(descriptorBytes,sizeof(OperationRecord),descriptorBytes)
            || !checkedAdd(descriptorBytes,1,descriptorBytes))
            return fail("Minimum source or descriptor byte calculation overflowed");
        if (op.retainedMesh) {
            retainedBytes=sizeof(RetainedMeshData);
            if (!checkedAdd(retainedBytes,terminalBytes,retainedBytes) || !checkedAdd(retainedBytes,1,retainedBytes)
                || !checkedAdd(sourceBytes,sizeof(RetainedMeshData),sourceBytes)
                || !addArrayBytes(record.positions,sizeof(Vec3),sourceBytes)
                || !addArrayBytes(record.identities,sizeof(StableId),sourceBytes)
                || !addArrayBytes(record.indices,sizeof(std::uint32_t),sourceBytes)
                || !addArrayBytes(record.attributes,sizeof(AttributeData),sourceBytes)
                || !checkedAdd(sourceBytes,record.importedAnimation,sourceBytes) || !checkedAdd(sourceBytes,1,sourceBytes))
                return fail("Minimum retained byte calculation overflowed");
        }
        std::size_t sourceMinimum=descriptorBytes,sampleBytes=sizeof(GeometryData);
        if (!checkedAdd(sourceMinimum,retainedBytes,sourceMinimum) || sourceBytes<sourceMinimum
            || !checkedAdd(sampleBytes,terminalBytes,sampleBytes))
            return fail("Minimum source capacities lost their nonnegative payload excess");
        const std::shared_ptr<const ParticleBodyCallerLease> callerResources;
        const RetainedResourceOwner snapshotResources;
        for (const auto proxy : {ParticleBodyGeometryProxy::sphere,ParticleBodyGeometryProxy::fixedBox}) {
            for (unsigned route=0;route<3;++route) {
                ParticleBodyGeometryAttempt attempt;
                const auto body=route==0 ? admitParticleBodyGeometryWithRetainedResources(token,proxy,1,{},attempt,error,
                        callerResources,snapshotResources)
                    : route==1 ? admitParticleBodyGeometry(token,proxy,1,{},attempt,error,callerResources)
                    : admitParticleBodyGeometry(token,proxy,1,{},attempt,error);
                const auto& recipe=attempt.recipe;
                std::size_t envelope=recipe.retainedLogicalBytes;
                if (recipe.sourceLogicalBytes!=sourceBytes || recipe.descriptorCopyLogicalBytes!=descriptorBytes
                    || recipe.retainedSourceSnapshotLogicalBytes!=retainedBytes || recipe.sampleLogicalBytes!=sampleBytes
                    || recipe.censusLogicalBytes!=particlebodygeometrydetail::censusLogicalBytes
                    || recipe.census.operationRecords!=1 || recipe.census.shapeProbes!=1
                    || recipe.referenceValidation.positionElements!=vertices || recipe.referenceValidation.identityElements!=vertices
                    || recipe.referenceValidation.indexElements!=indices || recipe.referenceNormalization.positionElements!=2*vertices
                    || recipe.referenceNormalization.fieldComponents!=(proxy==ParticleBodyGeometryProxy::fixedBox ? 3u:0u)
                    || !checkedAdd(envelope,sourceBytes,envelope) || !checkedAdd(envelope,sampleBytes,envelope)
                    || !checkedAdd(envelope,recipe.censusLogicalBytes,envelope) || envelope!=recipe.factoryAttemptLogicalBytes
                    || !attempt.censusReserved || !attempt.referenceReserved)
                    return fail("Minimum direct/explicit/default recipe or reservation staging changed");
                if (token==single || (token==grid && proxy==ParticleBodyGeometryProxy::fixedBox)) {
                    const auto reason=token==single ? "Body geometry reference has no finite enclosing radius"
                        : "Body geometry fixed box requires three positive reference half ratios";
                    if (body || attempt.publicationReserved || attempt.published || error!=reason)
                        return fail("Minimum radius or planar fixedBox refusal changed its real boundary");
                    continue;
                }
                if (!body || !attempt.publicationReserved || !attempt.published || !error.empty()
                    || body->source()!=token || body->sourceReceipt().sourceRevision!=token->receipt().sourceRevision
                    || body->proxy()!=proxy || body->structuralRevision()!=7 || !body->matchesExactSource(token,100000)
                    || body->referenceRadius()<=1.0e-7f || body->referencePrototype().positions.size()!=vertices
                    || body->referencePrototype().vertexIds!=mesh.vertexIds || body->referencePrototype().indices!=mesh.indices
                    || !equal(std::get<GeometryData>(body->ownedDescriptor().data),mesh)
                    || body->ownedDescriptor().operations.data()==value.operations.data()
                    || std::get<GeometryData>(body->ownedDescriptor().data).positions.data()==mesh.positions.data())
                    return fail("Minimum publication lost genuine source, topology or independent snapshot");
                if (op.retainedMesh && (body->ownedDescriptor().operations.front().retainedMesh==op.retainedMesh
                    || !equal(body->ownedDescriptor().operations.front().retainedMesh->geometry,mesh)))
                    return fail("Minimum retained publication aliased its independent source snapshot");
                if (token==diagonal && (!equal(body->referenceCenter(),Vec3{})
                    || !std::all_of(body->referenceBoxHalfRatios().begin(),body->referenceBoxHalfRatios().end(),
                        particlebodygeometrydetail::PositiveReferenceHalf{})))
                    return fail("Two diagonal retained vertices lost positive radius and box half ratios");
            }
        }
    }
    return verifyParticleBodyOwnerNativeProofOracles(error);
}
