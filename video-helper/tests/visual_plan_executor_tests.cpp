#include "../src/visual_plan_executor.h"
#include "../src/visual_plan_telemetry_json.h"
#include "../src/particle_body_replay.h"
#include "../src/particle_rotating_exact_scalar.h"
#include "../src/particle_rotating_plane_certificate.h"
#include "../src/particle_rotating_plane_response.h"
#include "particle_geometry_fixture.h"
#include "particle_body_geometry_admission_fixture.h"
#include "particle_body_geometry_sample_fixture.h"
#include "solid_body_fixture.h"
#include "particle_points_fixture.h"
#include "water_surface_replay_fixture.h"
#include "../src/visual_parameter_history.h"
#include "../src/shader_generator.h"
#include "support/generated_surface_scene.h"
#include "../../shared/generated/SurfaceMaterialStarterPrograms.h"
#include "../../shared/Visual3DSceneComposition.h"

#include <filesystem>
#include <fstream>

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <new>
#include <string>
#include <thread>

static_assert(sizeof(videowire::VisualPlanExecutionState) < 4096u,
              "visual plan execution state must keep its plan table off the thread stack");

namespace
{
std::atomic<uint64_t> allocations { 0 };
}

void* operator new (std::size_t size)
{
    allocations.fetch_add(1, std::memory_order_relaxed);
    if (void* value = std::malloc(size)) return value;
    throw std::bad_alloc();
}
void operator delete (void* value) noexcept { std::free(value); }
void operator delete (void* value, std::size_t) noexcept { std::free(value); }

namespace
{
int failures = 0;
void check (bool condition, const char* message)
{
    if (! condition) { ++failures; std::fprintf (stderr, "FAIL: %s\n", message); }
}

}

#include "rotating_plane_anchor_fixture.h"

namespace videorender
{
// The test-only friend declared by SolidRotationLedger. For one body it does
// what ParticleBodyReplayState does in one interval: begin the interval,
// record the actual solidRotate step, then issue the pass-zero view against
// the previous and scheduled snapshots.
struct SolidRotationTestIssuer
{
    SolidRotationLedger ledger;
    ParticleRigidBody previous {},scheduled {};
    std::int64_t tick=0;
    const SolidRotationView* schedule(const ParticleRigidBody& before,ParticleRigidBody& body,float dt)
    {
        ledger.begin(tick,tick+1); ++tick;
        ledger.rotate(body,0,dt,0);
        previous=before; scheduled=body;
        return ledger.issue(0,previous,scheduled);
    }
    void retire() { ledger.retire(); }
};
}

namespace
{
// Tests own replay states the way production does: on the heap, through
// makeParticleBodyReplayState, since the inline rotating storage is large.
std::unique_ptr<videorender::ParticleBodyReplayState> heapReplayState()
{
    std::string error;
    auto state=videorender::makeParticleBodyReplayState(error);
    if (!state) { std::fprintf(stderr,"FAIL: %s\n",error.c_str()); std::abort(); }
    return state;
}

void checkMaterialFrameResolverBounds()
{
    std::string error;std::uint64_t bytes=0;
    check(videowire::materialFrameImageBytes(4096,4096,false,colortransform::kMaximumResidentBytes,bytes,error)
        && bytes==67108864,"existing maximum SDR dependency has an exact checked logical bill");
    check(videowire::materialFrameImageBytes(4096,4096,true,colortransform::kMaximumResidentBytes,bytes,error)
        && bytes==134217728,"already-admitted decoded linear dependency uses its existing eight-byte bill");
    for(auto extent:{std::pair<std::uint64_t,std::uint64_t>{0,1},{4097,1},{8192,8192},
        {std::numeric_limits<std::uint64_t>::max(),std::numeric_limits<std::uint64_t>::max()}})
        check(!videowire::materialFrameImageBytes(extent.first,extent.second,false,colortransform::kMaximumResidentBytes,bytes,error),
            "invalid/oversized/overflowing dependency target refuses instead of using the scene output allowance");
    check(!videowire::materialFrameImageBytes(4096,4096,false,67108863,bytes,error),
        "reservation refuses a one-byte budget deficit before a lease");
    check(videowire::materialframe::maximumOperations==32 && videowire::materialframe::maximumEvaluations==256
        && colortransform::kMaximumResidentBytes==2147483648ULL,
        "GF2 preserves schedule/evaluation and logical resident-image limits");
}

void checkImportedMaterialFrameDag()
{
    using namespace videowire;
    using namespace HarmonicMIDI::grid;
    // __FILE__ is relative to the compiler's cwd ("../tests/..." from video-helper/build),
    // so lexical parent steps lose the repository root. Canonicalize when it resolves.
    std::filesystem::path testFile(__FILE__);
    std::error_code fileError;
    if (std::filesystem::exists(testFile, fileError)) testFile = std::filesystem::canonical(testFile, fileError);
    const auto sourceRoot = testFile.parent_path().parent_path().parent_path() / "content/shader-packs";
    const auto read = [&](const char* path) {
        std::ifstream input(sourceRoot / path,std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(input),std::istreambuf_iterator<char>());
    };
    const auto generatorSource = read("arbit-isf-starters/shaders/plasma.fs");
    const auto filterSource = read("vidvox-isf/ISF/RGB Invert.fs");
    const auto transitionSource = read("vidvox-isf/ISF/Fade.fs");
    check(!generatorSource.empty() && !filterSource.empty() && !transitionSource.empty(),"imported Frame prepared admission loads actual bundled shader sources");
    if (generatorSource.empty() || filterSource.empty() || transitionSource.empty()) return;
    const auto escape = [](const std::string& value) {
        std::string result;
        for (const char c : value) switch(c) {
            case '&': result += "&amp;"; break; case '<': result += "&lt;"; break;
            case '>': result += "&gt;"; break; case '\"': result += "&quot;"; break;
            case '\'': result += "&apos;"; break; default: result += c; break;
        }
        return result;
    };
    const auto shaderPayload = [&](const char* role,const char* pack,const char* program,const std::string& source) {
        const auto* entry = shadercatalog::find(pack,program);
        check(entry != nullptr,"imported Frame shader identities exist in the current catalog");
        return "<FlatShaderBridge schemaVersion=\"2\" role=\"" + std::string(role)
            + "\" language=\"isf\" source=\"" + escape(source) + "\" sourceSha256=\"" + videohelper::sha256Text(source)
            + "\" catalogPackId=\"" + pack + "\" catalogProgramId=\"" + program + "\" parameters=\""
            + escape(entry ? shadercatalog::defaultParameterWire(*entry) : std::string{}) + "\"/>";
    };
    const auto generator = shaderPayload("generator","arbit-isf-starters-v1","plasma",generatorSource);
    const auto filter = shaderPayload("filter","vidvox-isf","rgb-invert",filterSource);
    std::string error;
    shadertransition::Payload transition;
    transition.catalogPackId = "vidvox-isf"; transition.catalogProgramId = "fade";
    transition.source = transitionSource; transition.sourceSha256 = videohelper::sha256Text(transitionSource); transition.progress = .25;
    const auto* transitionEntry = shadercatalog::find("vidvox-isf","fade");
    if (!transitionEntry) { check(false,"imported Frame transition catalog is available"); return; }
    transition.parameters = shadertransition::parametersForProgress(*transitionEntry,transition.progress,error);
    const auto transitionPayload = shadertransition::serialize(transition,error);
    for (const bool retained : {false,true})
    for (int mode = 0; mode < 7; ++mode)
    {
        CompiledVisualLayerPlan plan; plan.clipId = 93; plan.structuralRevision = 9; plan.producerValidated = true;
        visualimportedscenerender::Request request; request.structuralRevision = request.evaluationRevision = 9;
        request.renderStableId = 12;
        const int source = retained ? 6 : 70, render = 11, terminal = 82;
        request.sourceStableId = static_cast<std::uint64_t>(source) + 1;
        if (retained)
        {
            Visual3DScene lighting; lighting.id = {7}; lighting.objectCount = lighting.materialCount = lighting.cameraCount = 1;
            lighting.objects[0].id = {1}; lighting.objects[0].material = {1}; lighting.materials[0].id = {1};
            lighting.activeCamera = {50}; lighting.cameras[0].id = {50}; lighting.cameras[0].transform.translation.z = 4;
            const auto scene = videohelper::tests::generatedSurfaceScene(lighting,error);
            check(scene.has_value(),"retained imported admission fixture uses actual admitted Geometry Core scene transport");
            if (!scene) return;
            request.sceneSnapshot = std::make_shared<const Visual3DScene>(*scene);
        }
        else
        {
            request.asset = {"model-asset-1",3,std::string(64,'a'),"model/gltf-binary",4096}; request.sceneIndex = 0;
        }
        surfacematerialbinding::ImportedSceneMaterialRequest material;
        material.scene = retained ? request.sceneSnapshot->id : Scene3DId{1}; material.sceneSnapshot = request.sceneSnapshot;
        material.sceneRevision = material.structuralRevision = material.evaluationRevision = material.programRevision = 9;
        material.binding.targetKind = surfacematerialbinding::BindingTargetKind::ObjectOverride; material.binding.object = {1};
        material.binding.surfaceMaterialRevision = 9; material.version = surfacematerialbinding::kGraphFrameWireVersion;
        material.program = surfacematerialstarterfixture::kPrograms[0].program(); material.program.textureSlotCount = 1;
        for (auto& op : material.program.operations) if (op.id == material.program.outputs[9]) op.unsignedLiteral = 1;
        const auto admitted = surfacematerial::admit(material.program,error);
        if (!admitted) { check(false,"imported Frame material fixture has admitted Surface IR"); return; }
        material.binding.surfaceMaterialDigest = admitted->structuralDigest();
        const materialframe::Endpoint endpoint {mode == 6 ? 94 : mode == 2 || mode == 3 || mode == 4 ? 92 : 91,
            mode == 3 ? 2 : mode == 2 || mode == 4 || mode == 6 ? 1 : 0};
        surfacematerialbinding::TextureSlotBinding texture; texture.source = surfacematerialbinding::TextureSourceKind::GraphFrame;
        texture.graphFrame = endpoint; material.binding.textures = {texture}; request.material = material;
        const auto addOperation = [&](int id,const std::string& kind,const char* capability,const std::string& payload = {}) {
            plan.nodeIds.push_back(id); plan.nodeKinds.push_back(kind); plan.operations.push_back({id,kind,capability,payload});
        };
        const auto imagePort = [&](int id,int port,const char* direction) {
            plan.ports.push_back({id,port,1,direction,"frame","image","rgba8","sRGB","sRGB","straight"});
        };
        const auto controlPort = [&](int id,int port,int channels,const char* direction,const std::string& type) {
            plan.ports.push_back({id,port,channels,direction,"control",type,"unspecified","unspecified"});
        };
        addOperation(source,retained ? "visual.3d.scene.retained" : "visual.3d.imported-animation","control-eval");
        if (!retained) plan.operations.back().backendCapability = visualimportedscenerender::kSourceBackendCapability;
        if (retained) controlPort(source,0,1,"out","scene3D");
        else
        {
            const char* types[] = {"mesh","skeleton","morphTargets","animationClip","scene3D"};
            for (int port = 0; port < 5; ++port) controlPort(source,port,1,"out",types[port]);
            for (std::size_t index = 0; index < material.program.operations.size(); ++index)
            {
                const auto& op = material.program.operations[index]; const int id = 100 + static_cast<int>(index);
                const auto token = surfacematerial::token(op.resultType);
                addOperation(id,"visual.surface.constant." + std::string(token),"control-eval");
                const int channels = op.resultType == surfacematerial::ValueType::Vec3 ? 3 : 1;
                controlPort(id,0,channels,"out",std::string(op.resultType == surfacematerial::ValueType::UInt ? "integer" : token));
                controlPort(terminal,static_cast<int>(index),channels,"in",std::string(op.resultType == surfacematerial::ValueType::UInt ? "integer" : token));
                plan.edges.push_back({id,0,terminal,static_cast<int>(index)});
            }
            addOperation(terminal,"visual.surface.material","control-eval"); controlPort(terminal,10,1,"out","material");
            controlPort(terminal,11,3,"in","vec3"); controlPort(render,2,1,"in","material");
            plan.edges.push_back({terminal,10,render,2});
        }
        addOperation(91,mode == 1 || mode == 6 ? "visual.shader.generator" : mode == 5 ? "video.layer.source" : "video.source",
            mode == 1 || mode == 6 ? "native-gpu" : "source-decode",
            mode == 1 || mode == 6 ? generator : mode == 5 ? materialframesource::encode(94) : videosourcecolor::encode(videosourcecolor::Policy{4000.0}));
        imagePort(91,0,"out");
        if (mode == 2 || mode == 3 || mode == 4 || mode == 6)
        {
            const auto kind = mode == 3 ? shadertransition::operationKind : mode == 4 ? "visual.frame-delay" : "visual.shader.filter";
            visualtemporaloperation::Payload temporal; temporal.mode = visualtemporaloperation::Mode::frameDelay;
            temporal.historyLength = 3; temporal.decay = 0; temporal.zoom = 1;
            addOperation(92,kind,"native-gpu",mode == 3 ? transitionPayload : mode == 4 ? visualtemporaloperation::serialize(temporal) : filter);
            imagePort(92,0,"in"); imagePort(92,mode == 3 ? 2 : 1,"out"); plan.edges.push_back({91,0,92,0});
            if (mode == 3) {
                addOperation(93,"video.layer.source","source-decode",materialframesource::encode(94)); imagePort(93,0,"out");
                imagePort(92,1,"in"); plan.edges.push_back({93,0,92,1});
            }
            if (mode == 6) {
                addOperation(94,"visual.shader.filter","native-gpu",filter); imagePort(94,0,"in"); imagePort(94,1,"out");
                plan.edges.push_back({92,1,94,0});
            }
        }
        addOperation(render,"visual.3d.render","native-gpu",visualimportedscenerender::encode(request));
        controlPort(render,0,1,"in","scene3D"); imagePort(render,1,"out"); imagePort(retained ? render : terminal,12,"in");
        plan.edges.insert(plan.edges.begin(),{source,retained ? 0 : 4,render,0});
        plan.edges.push_back({endpoint.node,endpoint.port,retained ? render : terminal,12});
        plan.descriptorCount = static_cast<int>(plan.ports.size()); plan.operationCount = static_cast<int>(plan.operations.size());
        plan.frameOutputCount = plan.peakLiveFrameCount = plan.allocatedFrameSlotCount = 5;
        VisualLayerExecution execution;
        const bool prepared = compileVisualLayerExecution(plan,execution,error);
        if (!prepared) std::fprintf(stderr,"imported Frame prepared admission retained=%d mode=%d: %s\n",retained,mode,error.c_str());
        check(prepared && execution.importedSceneRender && execution.geometryFrameSchedule
            && execution.geometryFrameSchedule->outputs == std::vector<materialframe::Endpoint>{endpoint},
            "ordinary and retained imported Surface prepare the exact bounded Frame DAG");
        videosourcecolor::Policy policy;
        check(materialframe::sourcePolicy(plan,{91,0},policy,error) == (mode != 1 && mode != 6)
            && (mode == 1 || mode == 5 || mode == 6 || policy.hlgNominalDisplayPeakNits == 4000.0),
            "prepared decoded source policies preserve the current clip HLG interpretation");
        check(materialframe::sourcePolicy(plan,endpoint,policy,error) == (mode == 0 || mode == 5),
            "generated, filtered, transition and temporal endpoints retain the decoder-only resolver refusal until GF2");
        const auto reject = [&](CompiledVisualLayerPlan invalid,const char* message) {
            VisualLayerExecution refused; check(!compileVisualLayerExecution(invalid,refused,error),message);
        };
        auto invalid = plan;
        auto frameSource = std::find_if(invalid.operations.begin(),invalid.operations.end(),[](const auto& op){return op.nodeId == 91;});
        frameSource->kind = "visual.shader.generator.future";
        const auto identity = std::find(invalid.nodeIds.begin(),invalid.nodeIds.end(),91);
        invalid.nodeKinds[static_cast<std::size_t>(std::distance(invalid.nodeIds.begin(),identity))] = frameSource->kind;
        reject(invalid,"imported Frame DAG rejects unknown producers without a kind-only fallback");
        invalid = plan; invalid.ports.push_back(plan.ports.back()); reject(invalid,"imported Frame DAG rejects duplicate ports");
        invalid = plan; invalid.edges.push_back(plan.edges.back()); reject(invalid,"imported Frame DAG rejects duplicate edges");
        invalid = plan; invalid.operations.push_back(plan.operations[retained ? 1 : 12]);
        invalid.nodeIds.push_back(invalid.operations.back().nodeId); invalid.nodeKinds.push_back(invalid.operations.back().kind);
        reject(invalid,"imported Frame DAG rejects duplicate identities");
        invalid = plan; invalid.nodeKinds.back() = "visual.3d.render.future"; reject(invalid,"imported Frame DAG rejects mismatched identity kinds");
        invalid = plan; invalid.edges.back().fromPort = 7; reject(invalid,"imported Frame DAG rejects mismatched authored endpoint ports");
        invalid = plan; invalid.edges.push_back({endpoint.node,endpoint.port,91,0}); reject(invalid,"imported Frame DAG rejects extra cyclic edges");
        invalid = plan; invalid.ports.back().pixelFormat = "rgba16f"; reject(invalid,"imported Frame DAG rejects scene-linear descriptor widening");
        invalid = plan; invalid.ports.back().transfer = "pq"; reject(invalid,"imported Frame DAG retains PQ/HLG refusal");
        invalid = plan; invalid.nodeIds.insert(invalid.nodeIds.end()-1,95); invalid.nodeKinds.insert(invalid.nodeKinds.end()-1,"video.source");
        invalid.operations.insert(invalid.operations.end()-1,{95,"video.source","source-decode",{}});
        invalid.ports.push_back({95,0,1,"out","frame","image","rgba8","sRGB"}); reject(invalid,"imported Frame DAG rejects unused producers");
        invalid = plan;
        for (int id = 200; id < 233; ++id) {
            invalid.nodeIds.insert(invalid.nodeIds.end()-1,id); invalid.nodeKinds.insert(invalid.nodeKinds.end()-1,"video.source");
            invalid.operations.insert(invalid.operations.end()-1,{id,"video.source","source-decode",{}});
            invalid.ports.push_back({id,0,1,"out","frame","image","rgba8","sRGB"});
        }
        reject(invalid,"imported Frame DAG retains the existing 32-operation admission cap");
        if (mode == 4) {
            invalid = plan;
            auto& payload = std::find_if(invalid.operations.begin(),invalid.operations.end(),[](const auto& op){return op.nodeId == 92;})->payloadXml;
            visualtemporaloperation::Payload temporal; visualtemporaloperation::parseForKind("visual.frame-delay",payload,temporal,&error);
            temporal.reset = 1; payload = visualtemporaloperation::serialize(temporal);
            reject(invalid,"imported Frame temporal reset refuses without the shared reset-history scheduler");
        }
    }
}

void checkParticlePointsCompiledProvider()
{
    using namespace videowire;
    using namespace videowire::geometry;
    std::string error;
    const auto value=particlePointsInstanceFixture(error);
    const auto context=particlePointsFixtureContext();
    const auto admitted=value ? admitValue(*value,particlePointsInstanceContract(),{},context,error) : std::nullopt;
    check(admitted.has_value(),"deferred source fixture admits with its compiled declaration");
    if (!admitted) return;
    CompiledVisualLayerPlan plan;
    plan.clipId=4; plan.structuralRevision=3; plan.producerValidated=true;
    plan.nodeIds={804,800}; plan.nodeKinds={"geometry.core.runtime","visual.particles.points"};
    plan.descriptorCount=plan.operationCount=2;
    plan.frameOutputCount=plan.peakLiveFrameCount=plan.allocatedFrameSlotCount=1;
    plan.ports={{804,1,1,"out","frame","image","rgba8","sRGB","sRGB","straight"},
                {800,7,1,"out","control","points3D","unspecified","unspecified"},
                {804,0,1,"in","control","points3D","unspecified","unspecified"}};
    plan.edges={{800,7,804,0}};
    plan.operations={{804,"geometry.core.runtime","native-gpu",encodeLoweredPlanText(lowerRuntimePlan(particlePointsInstanceContract(),*admitted))},
        {800,"visual.particles.points","control-eval","<NodeParams motionMode=\"2\" historicalReplay=\"1\" count=\"2\" pointsRevision=\"3\"/>"}};
    VisualLayerExecution execution;
    check(validateCompiledVisualLayerPlans({}, {plan},true,error)
        && compileVisualLayerExecution(plan,execution,error) && execution.particlePoints
        && !execution.particles && execution.particleNodeId==800,
        "exported compiled Graph admits the exact Points provider edge without changing Image source semantics");
    VisualPlanResourceUsage usage;
    check(accountVisualPlanResources(plan,64,64,usage,error)
        && usage.descriptors==2 && usage.operations==2 && usage.frameOutputs==1
        && usage.peakLiveFrames==1 && usage.frameSlots==1,
        "external Points provider retains the single native Frame allocation budget");
    auto disconnected=plan; disconnected.edges.clear();
    check(!validateCompiledVisualLayerPlans({}, {disconnected},true,error)
        && !compileVisualLayerExecution(disconnected,execution,error),
        "disconnected point providers refuse compiled Graph and execution admission");
    auto stale=plan;
    stale.operations[1].payloadXml="<NodeParams motionMode=\"2\" historicalReplay=\"1\" count=\"2\" pointsRevision=\"2\"/>";
    check(!compileVisualLayerExecution(stale,execution,error),"stale provider revision cannot bind the deferred point copy");
    auto fractional=plan;
    fractional.operations[1].payloadXml="<NodeParams motionMode=\"2\" historicalReplay=\"1\" count=\"2.5\" pointsRevision=\"3\"/>";
    check(!compileVisualLayerExecution(fractional,execution,error),"point capacity must not inherit Image count truncation");
    auto solidExcess=plan;
    solidExcess.operations[1].payloadXml="<NodeParams motionMode=\"2\" historicalReplay=\"1\" simulationSpace=\"1\" count=\"129\" pointsRevision=\"3\"/>";
    check(!compileVisualLayerExecution(solidExcess,execution,error)
        && error.find("capacity from 1 to 128")!=std::string::npos,
        "Solid 3D point capacity refuses 129 bodies with the exact bound");
    auto planarExcess=plan;
    planarExcess.operations[1].payloadXml="<NodeParams motionMode=\"2\" historicalReplay=\"1\" count=\"65\" pointsRevision=\"3\"/>";
    check(!compileVisualLayerExecution(planarExcess,execution,error)
        && error.find("1 to 64 bodies")!=std::string::npos,
        "planar point capacity keeps its 64-body bound");
    auto missing=plan; missing.operations.pop_back(); missing.nodeIds.pop_back(); missing.nodeKinds.pop_back();
    missing.ports.resize(1); missing.edges.clear();
    check(!compileVisualLayerExecution(missing,execution,error),"standalone GCRV cannot recreate an undeclared particle provider");
    auto wrongId=plan; wrongId.nodeIds[1]=799; wrongId.operations[1].nodeId=799;
    wrongId.ports[1].nodeId=799; wrongId.edges[0].fromNodeId=799;
    check(!compileVisualLayerExecution(wrongId,execution,error),"provider node identity must match the exact GCRV point copy");
}

void checkVisualParameterHistory()
{
    using namespace videorender;
    std::string error;
    auto score=std::make_shared<const arbitmod::Score>();
    videotime::BeatTimeline timeline;
    timeline.reset(120,4);
    arbitmod::Routing route;
    route.destination="clip4/visual7/gravityZ"; route.source.type=arbitmod::SourceType::Lfo;
    route.source.lfo.periodBeats=4; route.smoothingBeats=.2f; route.mode=arbitmod::Mode::Replace;
    videocontrol::Plan plan; plan.numSlots=1;
    videocontrol::Operation constant; constant.nodeId=1; constant.kind="control.const";
    constant.outputSlots={0}; constant.params={0.75f}; plan.operations.push_back(constant);
    videocontrol::Operation sink; sink.nodeId=2; sink.kind="sink"; sink.inputs={{0}};
    sink.destination="clip4/visual7/drag"; sink.mode=arbitmod::Mode::Replace;
    plan.operations.push_back(sink);
    VisualParameterHistory history;
    check(history.bind({route},plan,score,timeline,30,{route.destination,sink.destination},error),
          "history binds existing routing and Patch Bay control identities");
    double gravity=0,drag=0;
    check(history.sample(route.destination,1,gravity,{},error)
        && history.sample(sink.destination,1,drag,{},error) && drag==0.75,
        "history evaluates routed and graph-authored controls without live UI state");
    arbitmod::RoutingState state;
    for (int frame=0;frame<=30;++frame)
    {
        const double seconds=double(frame)/30;
        arbitmod::Clock clock {static_cast<float>(timeline.secondsToBeat(seconds)),120,4};
        arbitmod::evaluateRouting(route,state,0,*score,clock,{},2.0f/30);
    }
    check(std::abs(gravity-state.smoothed)<1.0e-6,
          "historical smoothing uses the canonical project value-grid evaluator");
    double later=0,again=0,capturedDrag=0;
    route.depth=0; plan.operations[0].params[0]=0.25f;
    check(history.sample(route.destination,2,later,{},error)
        && history.sample(route.destination,1,again,{},error) && again==gravity
        && history.sample(sink.destination,2,capturedDrag,{},error) && capturedDrag==drag,
        "captured history survives backward seek and mutation of its authoring inputs");
    check(!history.sample(route.destination,100000,later,{},error),
          "over-cap historical warmup fails closed");
    error.clear();
    auto values=std::make_shared<VisualParameterTimeline>();
    check(values->bind({{"clip4/visual7/gravityZ",0,.4},{"clip4/visual7/gravityZ",.25,-.2}},error),
          "bounded automation samples admit with exact destination identities");
    double held=0; values->sample("clip4/visual7/gravityZ",.249,held);
    check(held==.4,"fixed-step automation does not smear a discrete knot into earlier history");
    auto p=solidBodyFixture(); p.historyClipId=4; p.historyNodeId=7;
    auto source=std::make_shared<ParticleHistorySource>(*p.history);
    source->parameterAt=[values](const auto& destination,double time,double& value,std::string&)
    { values->sample(destination,time,value); return true; };
    p.history=source;
    const auto expected=sampleSolidBodyFixture(p,.5,error);
    sampleSolidBodyFixture(p,.9,error); sampleSolidBodyFixture(p,.125,error);
    check(error.empty() && sameSolidBodies(expected,sampleSolidBodyFixture(p,.5,error)),
          "automated forces replay identically for direct export, sequential preview and backward seeks");
    auto unmodulated=solidBodyFixture();
    check(!sameSolidBodies(expected,sampleSolidBodyFixture(unmodulated,.5,error)),
          "historical force changes affect the solid trajectory");
    source->parameterAt=[](const std::string& destination,double time,double& value,std::string&)
    {
        if (destination=="clip4/visual7/emissionRate") value=4;
        if (destination=="clip4/visual7/reset") value=time>=.4 && time<.41 ? 1:0;
        return true;
    };
    check(sampleSolidBodyFixture(p,.26,error).count==1
        && sampleSolidBodyFixture(p,.42,error).count==0
        && sampleSolidBodyFixture(p,.7,error).count==1,
        "historical rate and reset edges govern bounded source births");
    check(!values->bind({{"same",0,1},{"same",0,2}},error),
          "duplicate historical sample times cannot create an ambiguous snapshot");
    visualtemporaloperation::Payload resetPayload,decoded;
    const auto legacy=visualtemporaloperation::serialize(resetPayload);
    resetPayload.reset=1;
    const auto extended=visualtemporaloperation::serialize(resetPayload);
    check(legacy.rfind("temporal-v1",0)==0 && extended.rfind("temporal-v2",0)==0
        && visualtemporaloperation::parse(extended,decoded,&error) && decoded.reset==1
        && visualtemporaloperation::parse(legacy,decoded,&error) && decoded.reset==0,
        "temporal reset persistence preserves legacy payload bytes and clears absent reset state");
}

void checkTrackGroupSimulationHistory()
{
    using namespace videorender;
    auto sources=std::make_shared<videohelper::AnalysisSources>();
    // Independently authored Block B frames on an exactly representable 128 Hz grid.
    for (const auto group:{false,true})
    {
        videohelper::AnalysisSourceStream stream;
        stream.trackId=group ? 29:13; stream.group=group;
        stream.sampleRate=double(arbitblockb::kHop)*128;
        for (int i=0;i<121;++i)
        {
            arbitblockb::FeatureFrame frame;
            frame.sampleIndex=(i+1)*arbitblockb::kHop;
            frame.rms=group ? .75f:.25f; frame.peak=group ? .875f:.5f;
            frame.onset=i==(group ? 40:20) ? (group ? .625f:.125f):0;
            frame.onsetAge=frame.onset>0 ? 0:arbitblockb::kOnsetAgeMax;
            frame.bands[0]=group ? .375f:.0625f;
            stream.frames.push_back(frame);
        }
        sources->push_back(std::move(stream));
    }
    std::reverse(sources->begin(),sources->end());
    auto score=std::make_shared<const arbitmod::Score>();
    videotime::BeatTimeline timeline; timeline.reset(120,4);
    auto audio=std::make_shared<ParticleHistorySource>();
    audio->score=score; audio->analysisSources=sources;
    const VisualParameterHistory::AudioReader reader=[audio](double time,const auto& selected,
        bool master,arbitmod::Audio& value,std::string& error)
    { return particleHistoryModAudio(*audio,time,selected,master,value,error); };
    std::vector<arbitmod::Routing> routes;
    std::set<std::string> destinations;
    const std::array<arbitmod::SourceType,3> types{{arbitmod::SourceType::AudioRms,
        arbitmod::SourceType::AudioPeak,arbitmod::SourceType::AudioOnset}};
    for (int i=0;i<6;++i)
    {
        arbitmod::Routing route; route.source.type=types[i%3]; route.source.trackId=i<3 ? 13:29;
        route.mode=arbitmod::Mode::Replace; route.destination="scalar"+std::to_string(i);
        destinations.insert(route.destination); routes.push_back(route);
    }
    videocontrol::Plan plan; plan.numSlots=2;
    videocontrol::Operation source; source.kind="source"; source.nodeId=1;
    source.source.type=arbitmod::SourceType::AudioPeak; source.source.trackId=29; source.outputSlots={0};
    plan.operations.push_back(source);
    videocontrol::Operation sink; sink.kind="sink"; sink.nodeId=2; sink.inputs={{0}};
    sink.destination="graphPeak"; sink.mode=arbitmod::Mode::Replace;
    plan.operations.push_back(sink); destinations.insert(sink.destination);
    source.nodeId=3; source.source.trackId=99; source.outputSlots={1}; plan.operations.push_back(source);
    sink.nodeId=4; sink.inputs={{1}}; sink.destination="unrelated"; plan.operations.push_back(sink);
    VisualParameterHistory history; std::string error;
    check(history.bind(routes,plan,score,timeline,128,destinations,error),
          "track/group scalar history binds canonical routes and retained Control dependencies");
    const std::array<double,6> expected{{.25,.5,.125,.75,.875,.625}};
    for (int i=0;i<6;++i)
    {
        double value=-1;
        const double time=double(i<3 ? 21:41)/128;
        check(history.sample(routes[i].destination,time,value,reader,error) && value==expected[i],
              "historical RMS/peak/onset chooses the exact selected source and absolute hop");
    }
    double graphValue=0,earlyOnset=1,lateOnset=1;
    check(history.sample("graphPeak",.5,graphValue,reader,error) && graphValue==.875
        && history.sample("scalar2",20.0/128,earlyOnset,reader,error) && earlyOnset==0
        && history.sample("scalar5",42.0/128,lateOnset,reader,error) && lateOnset==0,
        "Control sources share exact group scalars while one-hop onsets keep their own project time");
    arbitmod::Audio scalarAudio;
    check(particleHistoryModAudio(*audio,.25,{13,29},false,scalarAudio,error)
        && scalarAudio.sources[0].tapPoint==1 && scalarAudio.sources[1].tapPoint==2,
        "scalar history retains track and group post-FX/pre-fader tap identities");
    auto spectrum=videohelper::sourceSpectrumReader(sources);
    videowire::geometry::spectrum::Bands bands;
    check(spectrum(videowire::geometry::spectrum::Source::track,13,.25,bands) && bands[0]==.0625f
        && !spectrum(videowire::geometry::spectrum::Source::group,13,.25,bands),
        "scalar history reuses spectrum streams without weakening track/group type admission");

    arbitmod::Routing force=routes[0]; force.destination="clip4/visual7/gravityX";
    auto knots=std::make_shared<VisualParameterTimeline>();
    check(knots->bind({{"clip4/visual7/reset",0,0},{"clip4/visual7/reset",.25,1},{"clip4/visual7/reset",31.0/120,0}},error),
        "reset edge fixture binds on the fixed 120 Hz project grid");
    const auto makeHistory=[&]
    {
        auto routed=std::make_shared<VisualParameterHistory>();
        check(routed->bind({force},{},score,timeline,120,{force.destination},error),
              "selected source force binds without an unrelated master analysis");
        auto owned=std::make_shared<ParticleHistorySource>(*audio);
        owned->parameterAt=[routed,knots,reader](const auto& destination,double time,double& value,std::string& error)
        { knots->sample(destination,time,value); return routed->sample(destination,time,value,reader,error); };
        return owned;
    };
    ParticleParams authored;
    authored.motionMode=2; authored.simulationSpace=1; authored.historicalReplay=true;
    authored.count=authored.geometryCount=1; authored.force=0; authored.lifetime=1;
    authored.historyClipId=4; authored.historyNodeId=7; authored.history=makeHistory();
    authored.geometryAnchors[0]={.35f,.45f,.35f,.45f}; authored.geometryDepths[0]={.5f,.5f};
    authored.geometryIdentities[0]=12345;
    std::array<unsigned char,sizeof(ParticleParams)> originalBytes;
    std::memcpy(originalBytes.data(),&authored,sizeof authored);
    const auto sample=[&](const ParticleParams& input,double time)
    {
        auto p=input; p.historyProjectSeconds=time;
        ParticleSolidState out; replayParticleBodies(p,time,1,nullptr,&error,&out); return out;
    };
    const auto before=sample(authored,.125),reset=sample(authored,.25),direct=sample(authored,.5);
    // Symplectic Euler: a*dt^2*N*(N+1)/2, with zero initial velocity.
    const double start=double(.35f),step=1.0/120,acceleration=.25;
    check(error.empty() && before.count==1 && reset.count==1 && direct.count==1
        && std::abs(before.bodies[0].x-(start+acceleration*step*step*15*16/2))<2.0e-6
        && reset.bodies[0].x==.35f && reset.bodies[0].vx==0
        && std::abs(direct.bodies[0].x-(start+acceleration*step*step*30*31/2))<2.0e-6,
        "one selected RMS force and reset edge match an independent fixed-step displacement oracle");
    auto whole=authored; whole.history=makeHistory();
    for (int frame=0;frame<=60;++frame) sample(whole,double(frame)/120);
    auto partitioned=authored; partitioned.history=makeHistory();
    for (const double time:{0.0,.125,.25,.375,.5}) sample(partitioned,time);
    sample(authored,.9); sample(authored,.1); sample(authored,.75);
    check(error.empty() && sameSolidBodies(direct,sample(whole,.5))
        && sameSolidBodies(direct,sample(partitioned,.5)) && sameSolidBodies(direct,sample(authored,.5))
        && std::memcmp(originalBytes.data(),&authored,sizeof authored)==0,
        "whole, partitioned, direct and backward replay agree while authored parameter bytes stay unchanged");

    error.clear();
    check(!particleHistoryModAudio(*audio,.25,{99},false,scalarAudio,error),
          "missing selected scalar source refuses instead of substituting master or silence");
    error.clear();
    check(!particleHistoryModAudio(*audio,.25,{13},true,scalarAudio,error),
          "a master route still requires its own baked project analysis");
    auto duplicate=std::make_shared<videohelper::AnalysisSources>(*sources);
    duplicate->push_back(duplicate->back()); duplicate->back().group=true;
    error.clear();
    check(!videohelper::sourceScalarAudioAt(duplicate.get(),{13},.25,scalarAudio,error),
          "conflicting track/group streams cannot alias the same stable source identity");
    auto overStreams=*sources;
    while (overStreams.size()<9) { auto stream=overStreams.front(); stream.trackId=100+int(overStreams.size()); overStreams.push_back(stream); }
    error.clear();
    check(!videohelper::sourceScalarAudioAt(&overStreams,{13},.25,scalarAudio,error),
          "scalar reads reject an over-cap canonical source collection");
    auto unavailable=*sources; unavailable.back().frames.clear(); error.clear();
    check(!videohelper::sourceScalarAudioAt(&unavailable,{13},.25,scalarAudio,error),
          "an empty selected stream does not become an admitted silent source");
    unavailable=*sources; unavailable.back().sampleRate=std::numeric_limits<double>::quiet_NaN(); error.clear();
    check(!videohelper::sourceScalarAudioAt(&unavailable,{13},.25,scalarAudio,error),
          "invalid selected-stream time identity refuses before indexing frames");
    auto missing=std::make_shared<ParticleHistorySource>(*audio); missing->analysisSources.reset();
    VisualParameterHistory absent; error.clear();
    check(absent.bind({force},{},score,timeline,120,{force.destination},error),"missing-source fixture binds routing");
    double value=0;
    check(!absent.sample(force.destination,.25,value,[missing](double time,const auto& selected,
        bool master,arbitmod::Audio& audio,std::string& error)
        { return particleHistoryModAudio(*missing,time,selected,master,audio,error); },error),
        "missing source propagates through the history evaluator before body replay");
    error.clear();
    check(!absent.sample(force.destination,.25,value,[](double,const auto&,bool,arbitmod::Audio&,std::string&)
        { return true; },error),"a reader cannot omit the selected scalar identity and report success");
    error.clear();
    check(!history.sample("scalar0",2000,value,reader,error),
          "selected-source history preserves the 216000-project-frame bound");
    force.source.type=arbitmod::SourceType::AudioBand; force.source.band=64; error.clear();
    check(!absent.bind({force},{},score,timeline,120,{force.destination},error),
          "an out-of-range selected AudioBand refuses before history binding");
    force.source.type=arbitmod::SourceType::AudioRms;
    std::vector<arbitmod::Routing> overSources;
    for (int i=0;i<9;++i) { force.source.trackId=100+i; overSources.push_back(force); }
    error.clear();
    check(!absent.bind(overSources,{},score,timeline,120,{force.destination},error),
          "nine selected scalar identities exceed the shared eight-source bound");
    error.clear();
    check(!absent.bind(std::vector<arbitmod::Routing>(129,force),{},score,timeline,120,{force.destination},error),
          "scalar history preserves the 128-route bound");
    force.source.trackId=13; error.clear();
    check(absent.bind(std::vector<arbitmod::Routing>(128,force),{},score,timeline,120,{force.destination},error)
        && !absent.sample(force.destination,400,value,reader,error),
        "scalar history preserves the retained-value bound before warming rows");
}

void checkTrackGroupBandSimulationHistory()
{
    using namespace videorender;
    auto streams=std::make_shared<videohelper::AnalysisSources>();
    for (const bool group:{false,true})
    {
        videohelper::AnalysisSourceStream stream;
        stream.trackId=group ? 29:13; stream.group=group;
        stream.sampleRate=double(arbitblockb::kHop)*128;
        for (int index=0;index<256;++index)
        {
            arbitblockb::FeatureFrame frame;
            frame.sampleIndex=(index+1)*arbitblockb::kHop;
            frame.rms=.875f; frame.peak=.9375f;
            frame.bands[0]=group ? .5f:.25f;
            frame.bands[63]=group ? .75f:(index<16 ? 0:.375f);
            stream.frames.push_back(frame);
        }
        streams->push_back(std::move(stream));
    }
    auto source=std::make_shared<ParticleHistorySource>();
    source->score=std::make_shared<const arbitmod::Score>(); source->analysisSources=streams;
    videotime::BeatTimeline timeline; timeline.reset(120,4);
    const VisualParameterHistory::AudioReader reader=[source](double time,const auto& selected,
        bool master,arbitmod::Audio& audio,std::string& error)
    { return particleHistoryModAudio(*source,time,selected,master,audio,error); };
    arbitmod::Routing force;
    force.source.type=arbitmod::SourceType::AudioBand; force.source.trackId=13; force.source.band=63;
    force.mode=arbitmod::Mode::Replace; force.destination="clip4/visual7/gravityX";
    videocontrol::Plan plan; plan.numSlots=1;
    videocontrol::Operation band; band.kind="source"; band.nodeId=1;
    band.source=force.source; band.source.trackId=29; band.outputSlots={0}; plan.operations.push_back(band);
    videocontrol::Operation sink; sink.kind="sink"; sink.nodeId=2; sink.inputs={{0}};
    sink.destination="clip4/visual7/gravityZ"; sink.mode=arbitmod::Mode::Replace; plan.operations.push_back(sink);
    VisualParameterHistory history; std::string error;
    check(history.bind({force},plan,source->score,timeline,128,{force.destination,sink.destination},error),
        "selected track and group AudioBand history bind flat and Control-plan routes");
    double early=-1,track=-1,group=-1;
    check(history.sample(force.destination,16.0/128,early,reader,error) && early==0
        && history.sample(force.destination,17.0/128,track,reader,error) && track==.375
        && history.sample(sink.destination,17.0/128,group,reader,error) && group==.75,
        "band 63 retains legitimate zero and distinct exact-source values at the authored Block B hop");
    double backward=-1;
    check(history.sample(force.destination,1,track,reader,error)
        && history.sample(force.destination,16.0/128,backward,reader,error) && backward==0,
        "selected bands replay on project time after a backward seek");
    arbitmod::Audio audio;
    check(particleHistoryModAudio(*source,.25,{29,13},false,audio,error)
        && audio.sourceCount==2 && audio.sources[0].tapPoint==2 && audio.sources[1].tapPoint==1
        && audio.sources[1].bandsReady && audio.sources[1].bands[63]==.375f
        && audio.sources[1].rms==.875f && audio.bands[63]==0,
        "selected band reads retain scalar and canonical group tap identity without an implicit master");
    auto masterFrames=streams->front().frames;
    masterFrames[31].bands[63]=.125f;
    source->audioFrames=&masterFrames; source->audioSampleRate=streams->front().sampleRate;
    auto master=force; master.source.trackId=-1; master.destination="master";
    VisualParameterHistory masterHistory;
    check(masterHistory.bind({master},{},source->score,timeline,128,{master.destination},error),
        "default master band route retains its existing history admission");
    double masterValue=-1;
    check(masterHistory.sample(master.destination,.25,masterValue,reader,error) && masterValue==.125,
        "master and selected bands remain independent at the same authored time");
    source->audioFrames=nullptr; source->audioSampleRate=0;
    videowire::geometry::spectrum::Bands exactBands;
    const auto spectrum=videohelper::sourceSpectrumReader(streams);
    check(spectrum(videowire::geometry::spectrum::Source::group,29,.25,exactBands)
        && exactBands[63]==.75f
        && !spectrum(videowire::geometry::spectrum::Source::track,29,.25,exactBands),
        "group band analysis retains the existing exact source-type admission");

    force.source.band=0;
    auto reset=std::make_shared<VisualParameterTimeline>();
    check(reset->bind({{"clip4/visual7/reset",0,0},{"clip4/visual7/reset",.25,1},
        {"clip4/visual7/reset",31.0/120,0}},error),"band-force reset fixture binds");
    const auto makeHistory=[&](bool withReset)
    {
        auto routed=std::make_shared<VisualParameterHistory>();
        check(routed->bind({force},{},source->score,timeline,120,{force.destination},error),
            "selected band force binds existing simulation automation");
        auto owned=std::make_shared<ParticleHistorySource>(*source);
        owned->parameterAt=[routed,reset,reader,withReset](const auto& destination,double time,
            double& value,std::string& error)
        {
            if (withReset) reset->sample(destination,time,value);
            return routed->sample(destination,time,value,reader,error);
        };
        return owned;
    };
    ParticleParams p;
    p.motionMode=2; p.simulationSpace=1; p.historicalReplay=true;
    p.count=p.geometryCount=1; p.force=0; p.lifetime=1;
    p.historyClipId=4; p.historyNodeId=7; p.history=makeHistory(true);
    p.geometryAnchors[0]={.35f,.45f,.35f,.45f}; p.geometryDepths[0]={.5f,.5f};
    p.geometryIdentities[0]=12345;
    const auto sample=[&](const ParticleParams& authored,double time)
    {
        auto value=authored; value.historyProjectSeconds=time;
        ParticleSolidState out; replayParticleBodies(value,time,1,nullptr,&error,&out); return out;
    };
    const auto direct=sample(p,.5),atReset=sample(p,.25);
    const double step=1.0/120,expected=double(.35f)+.25*step*step*30*31/2;
    check(error.empty() && direct.count==1 && atReset.count==1
        && atReset.bodies[0].vx==0 && atReset.bodies[0].x==.35f
        && std::abs(direct.bodies[0].x-expected)<2.0e-6,
        "selected band acceleration and reset match an independent fixed-step displacement oracle");
    auto sequential=p; sequential.history=makeHistory(true);
    for (int frame=0;frame<=60;++frame) sample(sequential,double(frame)/120);
    sample(p,.9); sample(p,.125);
    check(error.empty() && sameSolidBodies(direct,sample(sequential,.5))
        && sameSolidBodies(direct,sample(p,.5)),
        "band-driven direct export, sequential preview and backward seek reconstruct identical solid bodies");
    p.history=makeHistory(false);
    const auto firstLoop=sample(p,.5);
    check(error.empty() && sameSolidBodies(firstLoop,sample(p,1.5)),
        "repeated lifetime windows sample the band at their authored absolute project time");
    force.source.trackId=29; p.history=makeHistory(false);
    check(!sameSolidBodies(firstLoop,sample(p,.5)),"a different canonical group band changes the trajectory");

    // Refusals leave outputs and previously bound history intact.
    force.source.trackId=13; force.source.band=64;
    error.clear();
    check(!history.bind({force},{},source->score,timeline,128,{force.destination},error),
        "invalid band indices refuse before replacing a history snapshot");
    error.clear(); track=-1;
    check(history.sample(force.destination,17.0/128,track,reader,error) && track==.375,
        "a rejected rebind preserves the original selected band snapshot");
    force.source.band=63;
    const auto refuses=[&](std::shared_ptr<const videohelper::AnalysisSources> invalid,double seconds)
    {
        auto owned=std::make_shared<ParticleHistorySource>(*source); owned->analysisSources=std::move(invalid);
        VisualParameterHistory bound; std::string failure;
        if (!bound.bind({force},{},source->score,timeline,128,{force.destination},failure)) return false;
        double output=.8125;
        const bool rejected=!bound.sample(force.destination,seconds,output,
            [owned](double time,const auto& selected,bool master,arbitmod::Audio& audio,std::string& error)
            { return particleHistoryModAudio(*owned,time,selected,master,audio,error); },failure);
        return rejected && output==.8125 && !failure.empty();
    };
    check(refuses({},.25),"missing selected band refuses without substituting the master");
    auto invalid=std::make_shared<videohelper::AnalysisSources>(*streams);
    invalid->push_back(invalid->front()); invalid->back().group=true;
    check(refuses(invalid,.25),"duplicate track/group stable identities refuse selected bands atomically");
    invalid=std::make_shared<videohelper::AnalysisSources>(*streams); invalid->front().trackId=99;
    check(refuses(invalid,.25),"a foreign selected-source identity cannot replace the requested track");
    invalid=std::make_shared<videohelper::AnalysisSources>(*streams); invalid->front().frames[31].sampleIndex+=1;
    check(refuses(invalid,.25),"a stale selected band grid identity refuses at the requested hop");
    error.clear();
    check(videohelper::sourceScalarAudioAt(invalid.get(),{13},.25,audio,error)
        && audio.sources[0].rms==.875f && !audio.sources[0].bandsReady,
        "a stale band identity does not change the existing selected scalar route");
    invalid=std::make_shared<videohelper::AnalysisSources>(*streams);
    invalid->front().frames[31].bands[63]=std::numeric_limits<float>::quiet_NaN();
    check(refuses(invalid,.25),"nonfinite requested band history refuses before publishing parameter values");
    invalid=std::make_shared<videohelper::AnalysisSources>(*streams); invalid->front().frames.resize(16);
    check(refuses(invalid,.25),"missing retained selected band frames refuse rather than becoming silent history");
    invalid=std::make_shared<videohelper::AnalysisSources>(*streams);
    while (invalid->size()<9) { auto extra=invalid->front(); extra.trackId=100+int(invalid->size()); invalid->push_back(extra); }
    check(refuses(invalid,.25),"selected band history retains the eight-source collection bound");
    audio.sourceCount=0; audio.rms=.8125f; error.clear();
    check(!particleHistoryModAudio(*source,.25,{13,13},false,audio,error) && audio.rms==.8125f
        && audio.sourceCount==0,"a duplicate source request leaves the caller audio snapshot unchanged");
    VisualParameterHistory missingBands; error.clear();
    check(missingBands.bind({force},{},source->score,timeline,128,{force.destination},error),
        "band readiness refusal fixture binds");
    double unchanged=.8125;
    check(!missingBands.sample(force.destination,.25,unchanged,
        [](double,const auto& selected,bool,arbitmod::Audio& audio,std::string&)
        {
            audio.sourceCount=1; audio.sources[0]={selected[0],1,.25f,.5f,0}; return true;
        },error) && unchanged==.8125,"scalar-only readers cannot claim selected band availability");
}

void checkSolidBodyReplay()
{
    using namespace videorender;
    std::string error;
    auto p=solidBodyFixture();
    const auto first=sampleSolidBodyFixture(p,0,error),golden=sampleSolidBodyFixture(p,0.5,error);
    check(error.empty() && golden.count==2 && golden.bodies[0].z>first.bodies[0].z
        && golden.bodies[0].orientation!=first.bodies[0].orientation
        && golden.bodies[0].geometryIdentity==9007199254740001ull,
        "solid replay retains exact source IDs and integrates depth and quaternion orientation");
    const auto& q=golden.bodies[0].orientation;
    check(std::abs(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]-1)<1.0e-6f,
        "solid orientation remains a normalized quaternion");
    for (int frame=0;frame<=30;++frame) sampleSolidBodyFixture(p,double(frame)/60,error);
    sampleSolidBodyFixture(p,10000,error); sampleSolidBodyFixture(p,0.1,error);
    check(sameSolidBodies(golden,sampleSolidBodyFixture(p,0.5,error))
        && sameSolidBodies(golden,sampleSolidBodyFixture(p,2.5,error)),
        "solid sequential playback, direct seek and lifetime loop replay the same state");
    p.resetTime=0.25f;
    check(sameSolidBodies(golden,sampleSolidBodyFixture(p,0.75,error))
        && sampleSolidBodyFixture(p,0.1,error).count==0,"solid reset shifts replay and clears pre-reset bodies");
    p.resetTime=0; p.resetMode=1;
    const auto held=sampleSolidBodyFixture(p,2,error);
    check(sameSolidBodies(held,sampleSolidBodyFixture(p,10000,error)),"solid hold freezes after lifetime");
    auto fixedPose=solidBodyFixture(); fixedPose.angularVelocity={};
    fixedPose.orientationDegrees={17,29,43}; fixedPose.impulseZ=0;
    const auto authoredPose=sampleSolidBodyFixture(fixedPose,0,error);
    const auto fixedState=sampleSolidBodyFixture(fixedPose,.5,error);
    check(error.empty() && fixedState.bodies[0].orientation==authoredPose.bodies[0].orientation,
        "zero-spin authored Euler pose stays bit-identical during sphere replay");
    for (int frame=0;frame<=30;++frame) sampleSolidBodyFixture(fixedPose,double(frame)/60,error);
    sampleSolidBodyFixture(fixedPose,1.2,error); sampleSolidBodyFixture(fixedPose,.1,error);
    check(error.empty() && sameSolidBodies(fixedState,sampleSolidBodyFixture(fixedPose,.5,error))
        && sameSolidBodies(fixedState,sampleSolidBodyFixture(fixedPose,2.5,error)),
        "zero-spin Euler replay retains exact sequential, backward, seek and loop state");
    fixedPose.resetTime=.25f;
    check(sameSolidBodies(fixedState,sampleSolidBodyFixture(fixedPose,.75,error))
        && sampleSolidBodyFixture(fixedPose,.1,error).count==0,
        "zero-spin Euler replay retains exact reset state");
    fixedPose.resetTime=0;
    sampleSolidBodyFixture(fixedPose,.5,error);
    const auto fixedScene=particleSolidScene(fixedPose,fixedState,9,error);
    check(fixedScene && error.empty(),"zero-spin Euler sphere pose satisfies the existing scene quaternion contract");
    videowire::geometry::DeferredPointsSource pointsBinding;
    pointsBinding.sourceStableId=9001; pointsBinding.sourceRevision=1; pointsBinding.capacity=2;
    videowire::geometry::ValueDescriptor pointsValue;
    check(replayParticlePoints(fixedPose,.5,1,nullptr,9002,pointsBinding,pointsValue,error),
        "zero-spin Euler Points3D replay uses the shared pose authority");
    if (error.empty() && fixedScene) {
        const auto& rotation=fixedScene->objectInstances[0].rotation;
        const auto& point=std::get<videowire::geometry::PointsData>(pointsValue.data).points[0];
        const auto& pose=authoredPose.bodies[0].orientation;
        check(rotation.x==pose[0] && rotation.y==pose[1] && rotation.z==pose[2] && rotation.w==pose[3]
            && point.rotation.x==pose[0] && point.rotation.y==pose[1]
            && point.rotation.z==pose[2] && point.rotation.w==pose[3],
            "rendered sphere and Points3D preserve the exact authored quaternion");
    }
    ParticleRigidBody posedSphere=authoredPose.bodies[0],identitySphere=posedSphere;
    posedSphere.vx=identitySphere.vx=1; identitySphere.orientation={0,0,0,1};
    const SolidVector spherePoint{posedSphere.x+fixedPose.bodyRadius,posedSphere.y,posedSphere.z};
    solidSphereContact(posedSphere,nullptr,spherePoint,.01f,{},fixedPose);
    solidSphereContact(identitySphere,nullptr,spherePoint,.01f,{},fixedPose);
    check(solidPosition(posedSphere)==solidPosition(identitySphere)
        && solidVelocity(posedSphere)==solidVelocity(identitySphere) && posedSphere.angular==identitySphere.angular,
        "authored Euler pose keeps the original sphere collision geometry");
    ParticleRigidBody spun=authoredPose.bodies[0]; spun.angular={0,1,0};
    solidRotate(spun,1.0f/120,0);
    check(spun.orientation!=authoredPose.bodies[0].orientation,
        "current contact-style spin rotates a body whose authored angular velocity is zero");
    p=solidBodyFixture(); p.impulseZ=0; p.gravityZ=1; p.gravityX=0.3f; p.angularDrag=2;
    const auto gravity=sampleSolidBodyFixture(p,0.5,error);
    check(gravity.bodies[0].z>first.bodies[0].z && gravity.bodies[0].x>first.bodies[0].x
        && gravity.bodies[0].angular[1]<first.bodies[0].angular[1],"solid vector gravity and angular drag are effective");
    p=solidBodyFixture(); p.impulseZ=0; p.attraction=3; p.geometryDepths[0][1]=0.9f;
    check(sampleSolidBodyFixture(p,0.5,error).bodies[0].z>first.bodies[0].z,"solid point attraction acts in depth");
    p=solidBodyFixture(); p.force=0.5f;
    const auto square=sampleSolidBodyFixture(p,0.5,error);
    p.preparedSolids.reset(); p.preparedBodies.reset();
    ParticleSolidState wide;
    replayParticleBodies(p,0.5,2,nullptr,&error,&wide);
    check(sameSolidBodies(square,wide),"solid world-space dynamics do not depend on preview/export aspect ratio");

    auto score=std::make_shared<arbitmod::Score>(); score->scoreRevision=1;
    arbitmod::Note note; note.id=-101; note.midiNote=60; note.freqHz=261.625565f;
    note.velocity=127; note.startBeat=0; note.lengthBeats=8; score->notes.push_back(note);
    note.id=202; note.midiNote=72; note.freqHz*=2; note.startBeat=0.2f; note.lengthBeats=0.01f; score->notes.push_back(note);
    score->links.push_back({-301,-101,202,3,2,0});
    auto history=std::make_shared<ParticleHistorySource>(); history->score=score;
    auto& key=history->key;
    key.projectGeneration=key.sourceGeneration=key.helperGeneration=key.backendGeneration=key.deviceGeneration=1;
    key.scoreGeneration=key.beatMapGeneration=key.fpsGeneration=key.loopGeneration=key.seekGeneration=1;
    p=solidBodyFixture(); p.geometryCount=0; p.history=history; p.linkSpring=5; p.linkRestScale=0.5f;
    const auto linked=sampleSolidBodyFixture(p,0.5,error);
    p.linkSpring=0;
    const auto unlinked=sampleSolidBodyFixture(p,0.5,error);
    check(error.empty() && linked.count==1 && linked.bodies[0].identity==-101
        && linked.bodies[0].x!=unlinked.bodies[0].x,
        "solid replay resolves canonical signed link endpoints and short-lived note history");
    std::vector<arbitblockb::FeatureFrame> audio(50);
    for (std::size_t i=0;i<audio.size();++i) {
        audio[i].sampleIndex=static_cast<std::int64_t>(i+1)*arbitblockb::kHop;
        audio[i].rms=0.2f;
        if (i>=10) { audio[i].onset=1; audio[i].onsetAge=float(i-10)*arbitblockb::kHop/48000; }
    }
    history->audioFrames=&audio; history->audioSampleRate=48000;
    p.rmsGain=1; p.onsetGain=0.3f;
    const auto excited=sampleSolidBodyFixture(p,0.5,error);
    check(excited.bodies[0].y>unlinked.bodies[0].y
        && sameSolidBodies(excited,sampleSolidBodyFixture(p,0.5,error)),
        "solid audio RMS and onset impulses replay from canonical baked project history");

    ParticleRigidBody a,b; p=solidBodyFixture(); p.restitution=1; p.friction=0.5f;
    a.x=0.5f; a.y=0.5f; a.z=0.48f; a.vz=1; a.vx=0.2f;
    b.x=0.5f; b.y=0.5f; b.z=0.52f; b.vz=-1;
    solidSphereContact(a,&b,solidPosition(b),p.bodyRadius,{},p);
    check(b.z-a.z>=0.1799f && a.vz<0 && b.vz>0 && a.angular[1]!=0,
        "solid sphere contacts resolve in Z, bounce and transfer tangential impulse to spin");
    a={}; b={}; a.z=1; a.vz=32; b.z=0; b.vz=-32;
    solidSphereSweep(a,&b,{0,0,0},{0,0,1},{0,0,0},p.bodyRadius,{},p);
    check(a.z<b.z && a.vz<0 && b.vz>0,"swept solid sphere contacts prevent high-speed tunnelling");
    const SolidVector boxHalf {0.1f,0.02f,0.02f};
    a={}; b={}; a.x=0.5f; b.x=0.68f;
    solidBoxContact(a,b,boxHalf,p);
    check(b.x-a.x>=0.1999f,"oriented box proxy separates overlapping long axes");
    a={}; b={}; a.x=0.5f; b.x=0.68f;
    b.orientation={0,0,0.70710678f,0.70710678f};
    solidBoxContact(a,b,boxHalf,p);
    check(std::abs(b.x-a.x-0.18f)<1.0e-6f,
        "rotating the same box changes the separating axis and contact outcome");
    a={}; b={}; a.x=0.5f; b.x=0.68f; a.vx=1; b.vx=-1; a.vy=0.5f;
    auto noFrictionA=a,noFrictionB=b;
    p.friction=0; solidBoxContact(noFrictionA,noFrictionB,boxHalf,p);
    p.friction=1; solidBoxContact(a,b,boxHalf,p);
    check(a.vx<1 && b.vx>-1 && b.vx-a.vx>0,
        "oriented box contact applies restitution to closing bodies");
    check(std::abs(a.vy-b.vy)<std::abs(noFrictionA.vy-noFrictionB.vy),
        "oriented box contact reduces tangential slip with friction");
    a={}; b={}; a.x=0.8f; b.x=0.2f; a.y=b.y=a.z=b.z=0.5f;
    a.vx=32; b.vx=-32;
    b.orientation={0,0,0.70710678f,0.70710678f};
    auto beforeA=a,beforeB=b; beforeA.x=0.2f; beforeB.x=0.8f;
    solidBoxSweep(a,b,beforeA,beforeB,boxHalf,p);
    check(a.x<b.x && a.vx<0 && b.vx>0,
        "fast oriented boxes collide during a step instead of swapping sides");
    a=beforeA; b=beforeB; a.x=0.8f; b.x=0.2f; a.y=0.65f;
    beforeA.y=0.65f;
    solidBoxSweep(a,b,beforeA,beforeB,boxHalf,p);
    check(a.x==0.8f && b.x==0.2f,
        "swept oriented boxes keep a separated cross-axis near miss");
    const SolidVector touchingHalf {0.125f,0.02f,0.02f};
    a={}; b={}; a.x=0.125f; b.x=0.625f; a.vx=-1; b.vx=1;
    beforeA=a; beforeB=b; beforeA.x=0.25f; beforeB.x=0.5f;
    check(!solidBoxSweep(a,b,beforeA,beforeB,touchingHalf,p)
        && a.x==0.125f && b.x==0.625f && a.vx==-1 && b.vx==1,
        "boxes touching at step start can separate without rewinding");
    a=beforeA; b=beforeB; a.x=0.375f; b.x=0.375f; a.vx=1; b.vx=-1;
    check(solidBoxSweep(a,b,beforeA,beforeB,touchingHalf,p)
        && a.x<b.x && a.vx<0 && b.vx>0,
        "boxes touching at step start still collide when closing");
    a={}; a.x=a.y=a.z=0.5f; a.vy=1;
    const SolidVector pointCenter{0.5f,0.605f,0.5f};
    auto unrotated=a;
    solidBoxSphereContact(unrotated,boxHalf,pointCenter,0.01f,{},p);
    a.orientation={0,0,0.70710678f,0.70710678f};
    solidBoxSphereContact(a,boxHalf,pointCenter,0.01f,{},p);
    check(unrotated.y==0.5f && unrotated.vy==1 && a.y<0.5f && a.vy<0,
        "point sphere contacts use rotated box faces and change closing velocity");
    a={}; a.x=a.y=a.z=0.5f;
    solidBoxSphereContact(a,boxHalf,{0.5f,0.529f,0.529f},0.01f,{},p);
    check(a.y==0.5f && a.z==0.5f,
        "point sphere outside a rounded box corner does not make a false contact");
    ParticleGeometryFrame collider; collider.meshIdentity=9007199254740011ull;
    collider.minimum={0.2f,0.2f,0.45f}; collider.maximum={0.8f,0.8f,0.55f};
    collider.meshVertexCount=3; collider.meshTriangleCount=1;
    collider.meshVertices[0]={.2f,.2f,.45f}; collider.meshVertices[1]={.8f,.2f,.45f};
    collider.meshVertices[2]={.5f,.8f,.45f}; collider.meshVertexIdentities[0]=1;
    collider.meshVertexIdentities[1]=2; collider.meshVertexIdentities[2]=3;
    collider.meshTriangles[0]={0,1,2};
    a={}; a.x=a.y=0.5f; a.z=0.8f; a.vz=32;
    auto before=a; before.z=0.1f;
    check(solidGeometryContact(a,before,collider,collider,1.0f/120,p)
        && a.z<=0.36001f && a.vz<0,"solid Geometry triangle surface contact is swept in depth");
    ParticleSolidState state=golden; state.colliders=collider;
    const auto scene=particleSolidScene(p,state,9,error);
    check(scene && scene->objectCount==2 && scene->instancedObject==scene->objects[0].id
        && scene->objectInstanceCount==2 && scene->objectInstances[0].rotation.y!=0
        && scene->objects[1].vertexCount==3 && scene->objects[1].indexCount==3
        && scene->vertices[scene->objects[1].firstVertex].position.z==.45f
        && scene->materials[1].doubleSided,
        "native solid scene retains pose-driven body instances and the authored two-sided triangle");
    state.count=129;
    check(!particleSolidScene(p,state,9,error) && error.find("128")!=std::string::npos,
        "solid native body capacity fails closed before allocation");

    ParticleParams admitted;
    const std::string payload="<NodeParams motionMode=\"2\" historicalReplay=\"1\" count=\"8\" simulationSpace=\"1\" "
        "spawnZ=\"0.3\" impulseZ=\"0.4\" gravityX=\"0.1\" gravityZ=\"-0.2\" orientationY=\"45\" angularX=\"2\" angularDrag=\"0.7\"/>";
    check(videowire::admitParticleParameters(payload,admitted,error) && admitted.simulationSpace==1
        && admitted.orientationDegrees[1]==45 && admitted.angularVelocity[0]==2 && admitted.impulseZ==0.4f,
        "solid graph controls admit into the existing particle transport");
    for (const auto* bad:{"<NodeParams simulationSpace=\"1\"/>",
        "<NodeParams motionMode=\"2\" count=\"8\" simulationSpace=\"1\"/>",
        "<NodeParams motionMode=\"2\" historicalReplay=\"1\" count=\"8\" simulationSpace=\"1\" bodyShape=\"1\"/>",
        "<NodeParams motionMode=\"2\" historicalReplay=\"1\" count=\"8\" simulationSpace=\"1.5\"/>",
        "<NodeParams motionMode=\"2\" historicalReplay=\"1\" count=\"8\" simulationSpace=\"1\" angularZ=\"nan\"/>"})
        check(!videowire::admitParticleParameters(bad,admitted,error),"invalid solid transport fails closed");
    check(!videowire::admitParticleParameters(
        "<NodeParams motionMode=\"2\" historicalReplay=\"1\" count=\"2\" simulationSpace=\"1\" collisionMode=\"2\" bodyShape=\"2\"/>",
        admitted,error) && error.find("geometry")!=std::string::npos,
        "oriented box mode requires an explicit static body Geometry input");
    check(videowire::admitParticleParameters("<NodeParams/>",admitted,error) && admitted.simulationSpace==0,
        "legacy particle payloads retain planar defaults");

    using namespace videowire::geometry;
    PortContract contract; contract.carrier=CarrierKind::geometry3D; contract.maxVertices=256; contract.maxIndices=384;
    auto mesh=lowerGrid(4400,4400,1,2,2,0.5f,contract,error);
    auto retained=mesh ? admitValue(*mesh,contract,error):std::nullopt;
    check(retained.has_value(),"solid generated geometry fixture admits");
    if (retained) {
        auto boxDescriptor=retained->descriptor();
        std::get<GeometryData>(boxDescriptor.data).positions[0].z+=0.4f;
        auto boxReplay=solidBodyFixture();
        auto boxBinding=std::make_shared<ParticleGeometryBinding>();
        boxBinding->body=std::make_shared<const ValueDescriptor>(boxDescriptor);
        boxReplay.geometryBinding=boxBinding; boxReplay.bodyShape=2; boxReplay.collisionMode=2;
        SolidVector admittedHalf {};
        check(solidBodyBoxHalfExtents(boxReplay,admittedHalf) && admittedHalf[2]>0,
            "oriented contact extents derive from static body mesh bounds");
        boxReplay.geometryAnchors[0]={0.48f,0.5f,0.48f,0.5f};
        boxReplay.geometryAnchors[1]={0.52f,0.5f,0.52f,0.5f};
        boxReplay.geometryDepths[0]={0.5f,0.5f}; boxReplay.geometryDepths[1]={0.5f,0.5f};
        const auto boxState=sampleSolidBodyFixture(boxReplay,0.5,error);
        sampleSolidBodyFixture(boxReplay,1.3,error); sampleSolidBodyFixture(boxReplay,0.125,error);
        check(error.empty() && sameSolidBodies(boxState,sampleSolidBodyFixture(boxReplay,0.5,error))
            && sameSolidBodies(boxState,sampleSolidBodyFixture(boxReplay,2.5,error)),
            "oriented box replay is stable across seek and lifetime loop");
        boxReplay.resetTime=0.25f;
        check(sameSolidBodies(boxState,sampleSolidBodyFixture(boxReplay,0.75,error))
            && sampleSolidBodyFixture(boxReplay,0.1,error).count==0,
            "oriented box reset uses the existing historical replay clock");
        auto crossing=boxReplay;
        crossing.resetTime=0; crossing.bodyRadius=0.02f; crossing.attraction=1000;
        crossing.impulseZ=0; crossing.angularVelocity={};
        crossing.geometryAnchors[0]={0.45f,0.5f,0.9f,0.5f};
        crossing.geometryAnchors[1]={0.55f,0.5f,0.1f,0.5f};
        crossing.geometryDepths[0]={0.5f,0.5f}; crossing.geometryDepths[1]={0.5f,0.5f};
        const auto firstCrossing=sampleSolidBodyFixture(crossing,1.0/120,error);
        check(error.empty() && firstCrossing.count==2
            && firstCrossing.bodies[0].x<firstCrossing.bodies[1].x
            && firstCrossing.bodies[0].vx<0 && firstCrossing.bodies[1].vx>0,
            "fast boxes meet during the first replay step without tunnelling");
        const auto crossingSeek=sampleSolidBodyFixture(crossing,0.5,error);
        for (int frame=0;frame<=30;++frame)
            sampleSolidBodyFixture(crossing,double(frame)/60,error);
        check(error.empty() && sameSolidBodies(crossingSeek,sampleSolidBodyFixture(crossing,0.5,error))
            && sameSolidBodies(crossingSeek,sampleSolidBodyFixture(crossing,2.5,error)),
            "swept box contacts replay identically after sequential playback and lifetime loop");
        crossing.resetTime=0.25f;
        check(sameSolidBodies(crossingSeek,sampleSolidBodyFixture(crossing,0.75,error))
            && sampleSolidBodyFixture(crossing,0.1,error).count==0,
            "swept box contacts follow historical reset");
        const auto encoded=encodeLoweredPlanText(lowerRuntimePlan(contract,*retained));
        auto unsupported=payload;
        unsupported.insert(unsupported.size()-2,
            " bodyShape=\"2\" collisionMode=\"2\" bodyGeometryPlan=\""+encoded
                +"\" colliderGeometryPlan=\""+encoded+"\"");
        check(!videowire::admitParticleParameters(unsupported,admitted,error)
            && error.find("nonzero bounding")!=std::string::npos,
            "oriented body contacts still reject a planar body proxy with static mesh colliders");
        PortContract pointsContract; pointsContract.carrier=CarrierKind::points3D; pointsContract.maxPoints=64;
        auto pointSource=lowerPointsFromVertices(retained->descriptor(),4401,pointsContract,error);
        auto retainedPoints=pointSource ? admitValue(*pointSource,pointsContract,error):std::nullopt;
        auto cuboid=lowerMeshGenerator(4402,4402,1,OperationCode::cube,{2,0.4f,0.4f,0},contract,error);
        auto admittedBody=cuboid ? admitValue(*cuboid,contract,error):std::nullopt;
        check(retainedPoints.has_value() && admittedBody.has_value(),
            "oriented body and point collider fixtures admit");
        if (retainedPoints && admittedBody) {
            const auto bodyPlan=encodeLoweredPlanText(lowerRuntimePlan(contract,*admittedBody));
            const auto pointPlan=encodeLoweredPlanText(lowerRuntimePlan(pointsContract,*retainedPoints));
            const auto pointPayload="<NodeParams motionMode=\"2\" historicalReplay=\"1\" count=\"1\" "
                "simulationSpace=\"1\" collisionMode=\"2\" bodyShape=\"2\" "
                "bodyGeometryPlan=\""+bodyPlan+"\" colliderPointsPlan=\""+pointPlan+"\"/>";
            check(videowire::admitParticleParameters(pointPayload,admitted,error)
                && admitted.geometryBinding && admitted.geometryBinding->points,
                "oriented body admission retains typed point sphere colliders");
            const auto meshPayload="<NodeParams motionMode=\"2\" historicalReplay=\"1\" count=\"1\" "
                "simulationSpace=\"1\" collisionMode=\"2\" bodyShape=\"2\" "
                "bodyGeometryPlan=\""+bodyPlan+"\" colliderGeometryPlan=\""+encoded+"\"/>";
            check(videowire::admitParticleParameters(meshPayload,admitted,error)
                && admitted.geometryBinding && admitted.geometryBinding->mesh && admitted.geometryBinding->body,
                "oriented body admission retains an exact static authored mesh barrier");
            auto meshReplay=admitted; meshReplay.history=boxReplay.history;
            meshReplay.count=meshReplay.geometryCount=1; meshReplay.lifetime=2;
            meshReplay.geometryAnchors[0]={.5f,.5f,.5f,.5f}; meshReplay.geometryDepths[0]={.3f,.3f};
            meshReplay.geometryIdentities[0]=9007199254740001ull;
            auto meshBinding=std::make_shared<ParticleGeometryBinding>(*meshReplay.geometryBinding);
            meshBinding->scale=.5f; meshBinding->centerX=.5f; meshBinding->centerY=.5f; meshBinding->centerZ=.6f;
            meshReplay.geometryBinding=meshBinding;
            meshReplay.bodyRadius=.09f; meshReplay.colliderThickness=.005f;
            meshReplay.gravity=meshReplay.gravityX=meshReplay.gravityZ=0;
            meshReplay.force=meshReplay.attraction=meshReplay.drag=meshReplay.friction=0;
            meshReplay.impulseX=meshReplay.impulseY=0; meshReplay.impulseZ=.2f; meshReplay.angularVelocity={};
            meshReplay.orientationDegrees={0,0,7};
            ParticleGeometryFrame world; error.clear();
            check(sampleParticleGeometry(meshReplay,0,world,error) && world.meshIdentity==4400
                && world.meshVertices[0]==SolidVector{.375f,.375f,.6f}
                && world.meshVertexIdentities[0]==std::get<GeometryData>(retained->descriptor().data).vertexIds[0],
                "static oriented mesh source retains center plus scale world transform and exact source vertices");
            const auto meshAuthored=sampleSolidBodyFixture(meshReplay,0,error);
            const auto meshState=sampleSolidBodyFixture(meshReplay,1.5,error);
            check(error.empty() && meshState.count==1 && meshState.bodies[0].z<.6f && meshState.bodies[0].vz<0
                && meshState.bodies[0].angular==SolidVector{}
                && meshState.bodies[0].orientation==meshAuthored.bodies[0].orientation,
                "static authored mesh reflects the fixed box during historical replay");
            for (int frame=0;frame<=90;++frame) sampleSolidBodyFixture(meshReplay,double(frame)/60,error);
            sampleSolidBodyFixture(meshReplay,.25,error); sampleSolidBodyFixture(meshReplay,1.8,error);
            check(error.empty() && sameSolidBodies(meshState,sampleSolidBodyFixture(meshReplay,1.5,error))
                && sameSolidBodies(meshState,sampleSolidBodyFixture(meshReplay,3.5,error)),
                "static box mesh contacts replay identically after seek and lifetime loop");
            meshReplay.resetTime=.25f;
            check(sameSolidBodies(meshState,sampleSolidBodyFixture(meshReplay,1.75,error))
                && sampleSolidBodyFixture(meshReplay,.1,error).count==0,
                "static box mesh contacts retain the historical reset clock");
            meshReplay.resetTime=0; meshReplay.angularVelocity={0,1,0};
            meshReplay.preparedBodies.reset(); meshReplay.preparedSolids.reset(); meshReplay.historyProjectSeconds=.5;
            const auto refusedOwned=heapReplayState(); auto& refusedReplay=*refusedOwned; error.clear();
            const bool prepared=refusedReplay.prepare(meshReplay,.5,1,nullptr,error,true);
            const auto retainedState=meshState; auto unpublished=retainedState;
            ParticleBodyState planarBefore{}; planarBefore[0]={7,8,9,10}; auto planar=planarBefore;
            std::int64_t start=123,end=456;
            check(prepared && !refusedReplay.advanceOne(error) && error.find("Rotating")!=std::string::npos
                && !refusedReplay.completedIntervalTicks(start,end) && start==123 && end==456
                && !refusedReplay.publish(planar,error,&unpublished) && planar==planarBefore
                && sameSolidBodies(unpublished,retainedState),
                "unsupported rotational mesh sweep cannot publish bodies or a completed history interval");
            // A static-body replay has no caller lease to pay the capture copy
            // work, so it never admits an exact width. Long and short normal
            // spins both keep the released refusal before any pass-zero
            // operation. The authenticated route is checked in
            // checkRotatingPlaneSweepCaller.
            const std::string rotatingRefusal="Rotating box triangle sweeps are unsupported; the replay interval is refused";
            meshReplay.angularVelocity={0,0,1};
            for (const double seconds:{1.5,1.0/60}) {
                meshReplay.historyProjectSeconds=seconds;
                const auto spinOwned=heapReplayState(); auto& spinReplay=*spinOwned; error.clear();
                auto spinPlanar=planarBefore; auto spinUnpublished=retainedState;
                check(spinReplay.prepare(meshReplay,seconds,1,nullptr,error,true) && !spinReplay.advanceOne(error)
                    && error==rotatingRefusal && !spinReplay.completedIntervalTicks(start,end) && start==123 && end==456
                    && !spinReplay.publish(spinPlanar,error,&spinUnpublished) && spinPlanar==planarBefore
                    && sameSolidBodies(spinUnpublished,retainedState),
                    "a static-body normal spin admits no exact width and refuses its first interval with outputs unchanged");
            }
            auto overCapacity=pointPayload;
            // Solid 3D bodies cap at 128 independently of collider objects (65ed66be7e).
            overCapacity.replace(overCapacity.find("count=\"1\""),9,"count=\"129\"");
            check(!videowire::admitParticleParameters(overCapacity,admitted,error)
                && error.find("128")!=std::string::npos,
                "oriented box bodies are capped at 128 independently of point colliders");
            auto pointBinding=std::make_shared<ParticleGeometryBinding>(*boxBinding);
            pointBinding->body=std::make_shared<const ValueDescriptor>(admittedBody->descriptor());
            pointBinding->points=std::make_shared<const ValueDescriptor>(retainedPoints->descriptor());
            pointBinding->scale=0.5f;
            boxReplay.geometryBinding=pointBinding;
            SolidVector pointHalf {};
            check(solidBodyBoxHalfExtents(boxReplay,pointHalf),"point fixture retains valid box extents");
            pointBinding->centerX=0.5f+pointHalf[0]+0.002f-0.25f;
            pointBinding->centerY=0.25f; pointBinding->centerZ=0.5f;
            boxReplay.count=boxReplay.geometryCount=1;
            boxReplay.geometryAnchors[0]={0.5f,0.5f,0.5f,0.5f};
            boxReplay.geometryDepths[0]={0.5f,0.5f};
            boxReplay.impulseZ=0; boxReplay.angularVelocity={}; boxReplay.resetTime=0;
            auto withoutPoints=boxReplay;
            auto withoutPointBinding=std::make_shared<ParticleGeometryBinding>(*pointBinding);
            withoutPointBinding->points.reset();
            withoutPoints.geometryBinding=withoutPointBinding;
            const auto baseline=sampleSolidBodyFixture(withoutPoints,0.5,error);
            const auto pointState=sampleSolidBodyFixture(boxReplay,0.5,error);
            sampleSolidBodyFixture(boxReplay,1.2,error); sampleSolidBodyFixture(boxReplay,0.125,error);
            check(error.empty() && pointState.bodies[0].x<baseline.bodies[0].x
                && sameSolidBodies(pointState,sampleSolidBodyFixture(boxReplay,0.5,error))
                && sameSolidBodies(pointState,sampleSolidBodyFixture(boxReplay,2.5,error)),
                "oriented box point contacts replay identically across seek and loop");
            boxReplay.resetTime=0.25f;
            check(sameSolidBodies(pointState,sampleSolidBodyFixture(boxReplay,0.75,error))
                && sampleSolidBodyFixture(boxReplay,0.1,error).count==0,
                "oriented box point contacts follow historical reset");
        }
        auto geometryPayload=payload;
        geometryPayload.insert(geometryPayload.size()-2," bodyGeometryPlan=\""+encoded+"\" colliderGeometryPlan=\""+encoded+"\"");
        check(videowire::admitParticleParameters(geometryPayload,admitted,error) && admitted.geometryBinding
            && admitted.geometryBinding->body && admitted.geometryBinding->mesh,
            "generated or retained imported Geometry uses the same bounded body and collider transport");
        admitted.history=p.history;
        const auto sampled=sampleSolidBodyFixture(admitted,0.5,error);
        check(error.empty() && sampled.colliders.meshIdentity==4400,
            "solid mesh sampling retains exact Geometry identity");
        auto animated=particleGeometryFixture(true,error);
        const auto moving=animated ? admitValue(*animated->geometryBinding->mesh,contract,error):std::nullopt;
        check(moving.has_value(),"animated solid shape fixture admits its Geometry plan");
        if (moving) {
            if (admittedBody) {
                const auto movingMesh="<NodeParams motionMode=\"2\" historicalReplay=\"1\" count=\"1\" "
                    "simulationSpace=\"1\" collisionMode=\"2\" bodyShape=\"2\" bodyGeometryPlan=\""
                    +encodeLoweredPlanText(lowerRuntimePlan(contract,*admittedBody))+"\" colliderGeometryPlan=\""
                    +encodeLoweredPlanText(lowerRuntimePlan(contract,*moving))+"\"/>";
                check(videowire::admitParticleParameters(movingMesh,admitted,error)
                    && admitted.geometryBinding && admitted.geometryBinding->mesh && admitted.geometryBinding->body
                    && particleGeometryRequiresHistory(*admitted.geometryBinding->mesh),
                    "animated mesh barriers admit with a static oriented body proxy");
                const auto bodyLessMesh="<NodeParams motionMode=\"2\" historicalReplay=\"1\" count=\"1\" "
                    "simulationSpace=\"1\" collisionMode=\"2\" bodyShape=\"2\" colliderGeometryPlan=\""
                    +encodeLoweredPlanText(lowerRuntimePlan(contract,*moving))+"\"/>";
                check(!videowire::admitParticleParameters(bodyLessMesh,admitted,error),
                    "animated mesh barriers still require static oriented body geometry");
            }
            const auto animatedBody="<NodeParams motionMode=\"2\" historicalReplay=\"1\" count=\"1\" "
                "simulationSpace=\"1\" collisionMode=\"2\" bodyShape=\"2\" bodyGeometryPlan=\""
                +encodeLoweredPlanText(lowerRuntimePlan(contract,*moving))+"\" colliderGeometryPlan=\""+encoded+"\"/>";
            check(!videowire::admitParticleParameters(animatedBody,admitted,error)
                && error.find("static body")!=std::string::npos,
                "static mesh contacts refuse timeline box body geometry before dynamic body admission");
            auto rejected=payload;
            rejected.insert(rejected.size()-2," bodyGeometryPlan=\""+encodeLoweredPlanText(lowerRuntimePlan(contract,*moving))+"\"");
            check(videowire::admitParticleParameters(rejected,admitted,error) && admitted.geometryBinding->bodySource,
                "scheduled timeline body appearance retains its decoder-issued token");
        }
    }
}

static_assert(!std::is_default_constructible_v<videorender::ParticleBodyReplayContext>
    && !std::is_aggregate_v<videorender::ParticleBodyReplayContext>
    && !std::is_copy_constructible_v<videorender::ParticleBodyReplayContext>
    && !std::is_move_constructible_v<videorender::ParticleBodyReplayContext>
    && !std::is_copy_assignable_v<videorender::ParticleBodyReplayContext>
    && !std::is_move_assignable_v<videorender::ParticleBodyReplayContext>
    && !std::is_constructible_v<videorender::ParticleBodyReplayContext,
        std::shared_ptr<const videorender::ParticleBodyCallerLease>>
    && !std::is_assignable_v<decltype((std::declval<videorender::ParticleBodyReplayContext&>().caller)),
        std::shared_ptr<videorender::ParticleBodyCallerLedger>>,
    "Body replay context construction and reservation identity belong to its factory");

static_assert(!std::is_default_constructible_v<videorender::ParticleBodyReplayReceipt>
    && !std::is_copy_constructible_v<videorender::ParticleBodyReplayReceipt>
    && !std::is_default_constructible_v<videorender::ParticleSolidFrame>
    && !std::is_copy_constructible_v<videorender::ParticleSolidFrame>,
    "Completed dynamic body replay and native frames have private producers");

// Solid 3D bodies draw as one instanced scene object, so their bound follows
// the score's 128 note rows instead of the former 64 draw slots.
void checkSolidBodyInstancingBound()
{
    using namespace videorender;
    using HarmonicMIDI::grid::Visual3DSceneValidationCode;
    std::string error;
    auto p=manySolidBodyFixture(100);
    const auto direct=sampleSolidBodyFixture(p,0.5,error);
    check(error.empty() && direct.count==100 && direct.bodies[0].identity==1001 && direct.bodies[99].identity==1100
        && !sameSolidBodies(direct,sampleSolidBodyFixture(p,0.25,error)),
        "100 sounding notes replay 100 moving Solid 3D bodies in note identity order");
    for (int frame=0;frame<=30;++frame) sampleSolidBodyFixture(p,double(frame)/60,error);
    check(error.empty() && sameSolidBodies(direct,sampleSolidBodyFixture(p,0.5,error)),
        "100-body sequential replay matches the direct sample");
    sampleSolidBodyFixture(p,1.9,error); sampleSolidBodyFixture(p,0.125,error);
    const auto sought=sampleSolidBodyFixture(p,0.5,error);
    check(error.empty() && sameSolidBodies(direct,sought),"100-body forward and backward seeks match the direct sample");
    auto looped=std::make_shared<ParticleHistorySource>(*p.history);
    looped->key.loopGeneration++; p.history=looped;
    sampleSolidBodyFixture(p,1.9,error);
    check(error.empty() && sameSolidBodies(direct,sampleSolidBodyFixture(p,0.5,error)),
        "100-body replay after a loop wrap matches the direct sample");
    p.preparedSolids.reset(); p.preparedBodies.reset();
    auto wide=std::make_unique<ParticleSolidState>();
    replayParticleBodies(p,0.5,2,nullptr,&error,wide.get());
    check(error.empty() && sameSolidBodies(direct,*wide),"100-body replay does not depend on preview/export aspect ratio");
    const auto previewScene=particleSolidScene(p,direct,9,error);
    const auto exportScene=particleSolidScene(p,sought,9,error);
    bool sameInstances=previewScene && exportScene && previewScene->objectCount==1
        && previewScene->instancedObject==previewScene->objects[0].id
        && previewScene->objectInstanceCount==100 && exportScene->objectInstanceCount==100;
    for (std::size_t i=0;sameInstances && i<100;++i) {
        const auto& a=previewScene->objectInstances[i]; const auto& b=exportScene->objectInstances[i];
        sameInstances=a.translation.x==b.translation.x && a.translation.y==b.translation.y
            && a.translation.z==b.translation.z && a.rotation.x==b.rotation.x && a.rotation.y==b.rotation.y
            && a.rotation.z==b.rotation.z && a.rotation.w==b.rotation.w && a.scale.x==b.scale.x
            && a.scale.x==p.bodyRadius;
    }
    check(sameInstances,"preview and export scenes carry identical transforms for all 100 body instances");
    p.resetTime=0.25f;
    const auto reset=sampleSolidBodyFixture(p,0.75,error);
    sampleSolidBodyFixture(p,1.2,error); sampleSolidBodyFixture(p,0.3,error);
    check(error.empty() && reset.count==100 && sameSolidBodies(reset,sampleSolidBodyFixture(p,0.75,error)),
        "100 bodies respawn after a reset and replay identically across seeks");

    auto full=manySolidBodyFixture(128);
    const auto fullState=sampleSolidBodyFixture(full,0.5,error);
    const auto fullScene=particleSolidScene(full,fullState,9,error);
    check(error.empty() && fullState.count==128 && fullScene && fullScene->objectCount==1
        && fullScene->objectInstanceCount==128,"128 sounding notes replay and draw 128 Solid 3D bodies");
    auto excess=manySolidBodyFixture(129);
    check(sampleSolidBodyFixture(excess,0.5,error).count==128 && error.empty(),
        "a 129th sounding note stays within the 128-body replay bound");

    ParticleParams admitted;
    check(videowire::admitParticleParameters(
            "<NodeParams motionMode=\"2\" historicalReplay=\"1\" count=\"128\" simulationSpace=\"1\"/>",admitted,error)
        && admitted.count==128,"Solid 3D admission accepts 128 bodies");
    check(!videowire::admitParticleParameters(
            "<NodeParams motionMode=\"2\" historicalReplay=\"1\" count=\"129\" simulationSpace=\"1\"/>",admitted,error)
        && error=="Solid 3D coupled replay supports 1 to 128 bodies; set Count to 128 or less",
        "Solid 3D admission refuses 129 bodies with the exact bound");
    check(!videowire::admitParticleParameters(
            "<NodeParams motionMode=\"2\" historicalReplay=\"1\" count=\"65\"/>",admitted,error)
        && error=="coupled replay supports 1 to 64 bodies; set Count to 64 or less",
        "planar coupled bodies keep their 64-body bound");

    if (!fullScene) return;
    auto broken=std::make_shared<HarmonicMIDI::grid::Visual3DScene>(*fullScene);
    check(HarmonicMIDI::grid::validateVisual3DScene(*broken).valid(),"the instanced solid scene validates");
    broken->objectInstanceCount=0;
    check(HarmonicMIDI::grid::validateVisual3DScene(*broken).code==Visual3DSceneValidationCode::InvalidObjectInstances,
        "an instanced object with no instances is invalid");
    broken->objectInstanceCount=HarmonicMIDI::grid::Visual3DScene::kMaxObjectInstances+1;
    check(HarmonicMIDI::grid::validateVisual3DScene(*broken).code==Visual3DSceneValidationCode::InvalidObjectInstances,
        "more than 128 object instances are invalid");
    broken->objectInstanceCount=128; broken->objectInstances[100].scale.y=0;
    check(HarmonicMIDI::grid::validateVisual3DScene(*broken).code==Visual3DSceneValidationCode::InvalidObjectInstances,
        "a zero-scale object instance is invalid");
    broken->objectInstances[100]=fullScene->objectInstances[100]; broken->instancedObject={};
    check(HarmonicMIDI::grid::validateVisual3DScene(*broken).code==Visual3DSceneValidationCode::InvalidObjectInstances,
        "instances without an instanced object are invalid");
    auto composed=std::make_shared<HarmonicMIDI::grid::Visual3DScene>();
    check(!HarmonicMIDI::grid::scenecomposition::appendScene(*fullScene,*composed,true,error)
        && error=="Scene 3D composition does not carry instanced objects",
        "scene composition refuses instanced objects instead of dropping their instances");
    check(!arbitgpu::nativeObjectInstancesSupported(*fullScene,nullptr,true,error)
        && error=="Instanced scene objects require the plain scene material without a graph material program or Geometry instances",
        "instanced bodies refuse Geometry instances");
    error.clear();
    arbitgpu::NativeFixtureSceneRuntimeInputs motion; motion.imageOutput=renderpassoutput::Output::Motion;
    check(arbitgpu::nativeObjectInstanceRuntimeSupported(*fullScene,motion,error) && error.empty(),
        "instanced bodies accept Motion on a first frame, which has zero Motion");
    motion.previousMotion=arbitgpu::sceneMotionSample(motion);
    check(!arbitgpu::nativeObjectInstanceRuntimeSupported(*fullScene,motion,error)
        && error=="Motion output for instanced scene objects requires each instance's prior transform",
        "instanced bodies without prior transforms refuse Motion after a predecessor frame");
    error.clear();
    check(arbitgpu::nativeObjectInstanceRuntimeSupported(*fullScene,{},error) && error.empty(),
        "instanced bodies accept the Color output");
}

// Motion for Solid 3D bodies reads each body's transform one frame earlier.
// The prior comes from replaying the previous frame, so it does not depend on
// preview, export, seek or loop history.
void checkSolidBodyMotionPriors()
{
    using namespace videorender;
    using HarmonicMIDI::grid::Visual3DSceneValidationCode;
    std::string error;
    constexpr double step=1.0/120;
    const auto sameTransform=[](const HarmonicMIDI::grid::SceneTransform3D& a,const HarmonicMIDI::grid::SceneTransform3D& b) {
        return a.translation.x==b.translation.x && a.translation.y==b.translation.y && a.translation.z==b.translation.z
            && a.rotation.x==b.rotation.x && a.rotation.y==b.rotation.y && a.rotation.z==b.rotation.z
            && a.rotation.w==b.rotation.w && a.scale.x==b.scale.x && a.scale.y==b.scale.y && a.scale.z==b.scale.z;
    };
    // Counts the bodies whose prior is their own earlier pose. Every other
    // body must repeat its current transform.
    const auto priorsMatch=[&](const SolidBodyMotionSample& before,const SolidBodyMotionSample& now,
        const HarmonicMIDI::grid::Visual3DScene& scene,float radius,int& kept) {
        kept=0;
        if (scene.objectInstanceCount!=std::size_t(now.state.count)
            || scene.previousObjectInstanceCount!=scene.objectInstanceCount) return false;
        for (int i=0;i<now.state.count;++i) {
            const auto& body=now.state.bodies[std::size_t(i)];
            auto expected=scene.objectInstances[std::size_t(i)];
            for (int j=0;j<before.state.count;++j) {
                const auto& old=before.state.bodies[std::size_t(j)];
                if (old.identity!=body.identity || old.geometryIdentity!=body.geometryIdentity
                    || before.births[std::size_t(j)]!=now.births[std::size_t(i)]) continue;
                const auto& q=old.orientation;
                expected.translation={old.x-0.5f,old.y-0.5f,old.z-0.5f};
                expected.scale={radius,radius,radius}; expected.rotation={q[0],q[1],q[2],q[3]};
                ++kept; break;
            }
            if (!sameTransform(scene.previousObjectInstances[std::size_t(i)],expected)) return false;
        }
        return true;
    };
    const auto motionScene=[&](const ParticleParams& p,const SolidBodyMotionSample& before,const SolidBodyMotionSample& now) {
        const ParticleSolidMotionPrior prior{before.state,before.births,now.births,p.bodyRadius};
        return particleSolidScene(p,now.state,9,error,&prior);
    };

    // Mid-replay every body survives, so each prior is that body's pose one
    // frame earlier and at least one body moved.
    auto p=manySolidBodyFixture(100);
    const auto before=sampleSolidBodyMotion(p,0.5-step,error);
    const auto now=sampleSolidBodyMotion(p,0.5,error);
    const auto scene=motionScene(p,before,now);
    int kept=0; bool moved=false;
    check(error.empty() && scene && now.state.count==100 && priorsMatch(before,now,*scene,p.bodyRadius,kept) && kept==100
        && HarmonicMIDI::grid::validateVisual3DScene(*scene).valid(),
        "each of 100 surviving Solid 3D bodies carries its own pose one frame earlier");
    for (std::size_t i=0;scene && i<100;++i)
        moved=moved || !sameTransform(scene->previousObjectInstances[i],scene->objectInstances[i]);
    check(moved,"moving Solid 3D bodies carry prior transforms that differ from their current ones");
    const auto noPrior=particleSolidScene(p,now.state,9,error);
    check(noPrior && noPrior->previousObjectInstanceCount==0,"a scene built without a prior frame carries no prior transforms");

    // Preview, export, sequential playback, seeks, loop wraps and aspect give
    // byte-identical priors.
    const auto samePriors=[&](const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& other) {
        return scene && other && other->previousObjectInstanceCount==scene->previousObjectInstanceCount
            && std::memcmp(other->previousObjectInstances.data(),scene->previousObjectInstances.data(),
                   sizeof(HarmonicMIDI::grid::SceneTransform3D)*scene->previousObjectInstanceCount)==0
            && std::memcmp(other->objectInstances.data(),scene->objectInstances.data(),
                   sizeof(HarmonicMIDI::grid::SceneTransform3D)*scene->objectInstanceCount)==0;
    };
    for (int frame=0;frame<=30;++frame) sampleSolidBodyFixture(p,double(frame)/60,error);
    check(error.empty() && samePriors(motionScene(p,sampleSolidBodyMotion(p,0.5-step,error),sampleSolidBodyMotion(p,0.5,error))),
        "Solid 3D Motion priors after sequential playback match the direct sample");
    sampleSolidBodyFixture(p,1.9,error); sampleSolidBodyFixture(p,0.125,error);
    check(error.empty() && samePriors(motionScene(p,sampleSolidBodyMotion(p,0.5-step,error),sampleSolidBodyMotion(p,0.5,error))),
        "Solid 3D Motion priors after forward and backward seeks match the direct sample");
    auto looped=std::make_shared<ParticleHistorySource>(*p.history);
    looped->key.loopGeneration++; p.history=looped;
    sampleSolidBodyFixture(p,1.9,error);
    check(error.empty() && samePriors(motionScene(p,sampleSolidBodyMotion(p,0.5-step,error),sampleSolidBodyMotion(p,0.5,error))),
        "Solid 3D Motion priors after a loop wrap match the direct sample");

    // A replay window wrap respawns every body with its old identities but a
    // new birth time. Each counts as born this frame and has zero Motion.
    const auto beforeWrap=sampleSolidBodyMotion(p,2-step,error);
    const auto afterWrap=sampleSolidBodyMotion(p,2+step,error);
    const auto wrapped=motionScene(p,beforeWrap,afterWrap);
    bool allBorn=wrapped && afterWrap.state.count>0;
    for (int i=0;allBorn && i<afterWrap.state.count;++i)
        allBorn=sameTransform(wrapped->previousObjectInstances[std::size_t(i)],wrapped->objectInstances[std::size_t(i)]);
    check(error.empty() && allBorn && beforeWrap.state.count==afterWrap.state.count && priorsMatch(beforeWrap,afterWrap,*wrapped,p.bodyRadius,kept) && kept==0,
        "bodies respawned by a replay wrap repeat their current transforms, so their Motion is zero");
    auto reset=manySolidBodyFixture(100); reset.resetTime=0.25f;
    const auto beforeReset=sampleSolidBodyMotion(reset,0.25-step,error);
    const auto afterReset=sampleSolidBodyMotion(reset,0.25+step,error);
    const auto resetScene=motionScene(reset,beforeReset,afterReset);
    check(error.empty() && resetScene && afterReset.state.count==100
        && priorsMatch(beforeReset,afterReset,*resetScene,reset.bodyRadius,kept) && kept==0,
        "bodies born at a reset have zero Motion");

    // The radius is automatable, so the prior uses the prior frame's radius.
    ParticleSolidMotionPrior grown{before.state,before.births,now.births,p.bodyRadius*2};
    const auto radiusScene=particleSolidScene(p,now.state,9,error,&grown);
    check(radiusScene && radiusScene->previousObjectInstances[0].scale.x==p.bodyRadius*2
        && radiusScene->objectInstances[0].scale.x==p.bodyRadius,"Solid 3D Motion priors use the prior frame's body radius");
    grown.previousBodyRadius=0;
    check(!particleSolidScene(p,now.state,9,error,&grown)
        && error=="Solid 3D Motion prior frame has an invalid body count or radius",
        "a nonpositive prior body radius is refused");

    // Colliders draw without prior positions, so they must not move.
    auto collided=std::make_unique<SolidBodyMotionSample>(now);
    collided->state.colliders.pointCount=1; collided->state.colliders.solidPoints[0]={0.1f,0.2f,0.3f};
    auto still=std::make_unique<SolidBodyMotionSample>(before);
    still->state.colliders=collided->state.colliders;
    const auto steady=motionScene(p,*still,*collided);
    check(steady && steady->previousObjectInstanceCount==100 && steady->objectCount==2,
        "Solid 3D Motion accepts colliders that did not move");
    still->state.colliders.solidPoints[0][2]=0.31f;
    check(!motionScene(p,*still,*collided)
        && error=="Motion output for Solid 3D bodies over moving colliders is unavailable: prior collider positions are not retained",
        "Solid 3D Motion refuses moving colliders with the precise reason");

    // The scene contract keeps prior transforms paired with instances.
    if (!scene) return;
    auto broken=std::make_shared<HarmonicMIDI::grid::Visual3DScene>(*scene);
    broken->previousObjectInstanceCount=50;
    check(HarmonicMIDI::grid::validateVisual3DScene(*broken).code==Visual3DSceneValidationCode::InvalidObjectInstances,
        "prior instance transforms must match the instance count");
    broken->previousObjectInstanceCount=100; broken->previousObjectInstances[7].scale.z=0;
    check(HarmonicMIDI::grid::validateVisual3DScene(*broken).code==Visual3DSceneValidationCode::InvalidObjectInstances,
        "a zero-scale prior instance transform is invalid");
    broken->previousObjectInstances[7]=scene->previousObjectInstances[7];
    broken->instancedObject={}; broken->objectInstanceCount=0;
    check(HarmonicMIDI::grid::validateVisual3DScene(*broken).code==Visual3DSceneValidationCode::InvalidObjectInstances,
        "prior instance transforms without an instanced object are invalid");
    arbitgpu::NativeFixtureSceneRuntimeInputs motion; motion.imageOutput=renderpassoutput::Output::Motion;
    motion.previousMotion=arbitgpu::sceneMotionSample(motion);
    error.clear();
    check(arbitgpu::nativeObjectInstanceRuntimeSupported(*scene,motion,error) && error.empty(),
        "instanced bodies with prior transforms accept Motion after a predecessor frame");
}

// The production route asks for Solid 3D Motion through the Particles payload
// and builds the prior by replaying one project frame earlier. The draw is
// the same on preview and export and after seeks, and a Color draw is
// unchanged. The executor test target has no GL context, so the native
// pixels and Motion readback live in solid_body_fixture.h.
void checkSolidBodyMotionProduction()
{
    using namespace videorender;
    std::string error;
    constexpr double step=1.0/120;

    ParticleParams admitted;
    const std::string solid="<NodeParams motionMode=\"2\" historicalReplay=\"1\" count=\"8\" simulationSpace=\"1\"";
    check(videowire::admitParticleParameters(solid+"/>",admitted,error) && !admitted.solidMotionView && !admitted.solidExportMotion,
        "Solid 3D payloads without Motion requests draw Color");
    check(videowire::admitParticleParameters(solid+" imageOutput=\"1\" exportMotion=\"1\"/>",admitted,error)
        && admitted.solidMotionView && admitted.solidExportMotion,
        "Solid 3D payloads admit Motion View and Export Motion Data");
    for (const auto* planar:{"<NodeParams imageOutput=\"1\"/>","<NodeParams exportMotion=\"1\"/>"})
        check(!videowire::admitParticleParameters(planar,admitted,error) && error=="Particle Motion output requires Solid 3D",
            "planar particles refuse Motion requests with the precise reason");
    for (const auto* bad:{" imageOutput=\"2\"/>"," imageOutput=\"0.5\"/>"," exportMotion=\"nan\"/>"})
        check(!videowire::admitParticleParameters(solid+bad,admitted,error)
            && error=="Particle Image Output and Export Motion Data must be whole selections",
            "fractional or out-of-range Motion selections fail closed");

    // The production prior is the replay at clip time minus one project frame.
    auto p=manySolidBodyFixture(100); p.solidMotionView=true;
    const auto before=sampleSolidBodyMotion(p,0.5-step,error);
    const auto now=sampleSolidBodyMotion(p,0.5,error);
    ParticlePointReplayTimes times; times.birthSeconds=now.births;
    const auto replayPrior=[&](ParticleParams q) {
        q.historyProjectSeconds=0.5;
        return replayParticleSolidMotionPrior(q,0.5,step,1,nullptr,now.state,times,q.bodyRadius,error);
    };
    const auto prior=replayPrior(p);
    check(error.empty() && prior && now.state.count==100 && sameSolidBodies(prior->previous,before.state)
        && prior->previousBirths==before.births && prior->currentBirths==now.births
        && prior->previousBodyRadius==p.bodyRadius,
        "the production Solid 3D Motion prior is the replay one project frame earlier");
    auto sought=p;
    sampleSolidBodyFixture(sought,1.9,error); sampleSolidBodyFixture(sought,0.125,error);
    const auto again=replayPrior(sought);
    check(error.empty() && prior && again && sameSolidBodies(again->previous,prior->previous)
        && again->previousBirths==prior->previousBirths && again->previousBodyRadius==prior->previousBodyRadius,
        "the production Solid 3D Motion prior is unchanged by earlier seeks and repeated replays");

    // With a request, the drawn scene carries the reviewed per-body priors.
    auto drawn=p;
    drawn.preparedSolids=std::make_shared<const ParticleSolidState>(now.state); drawn.preparedSolidMotion=prior;
    const auto instances=[](const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& a,
                            const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& b,bool previous) {
        if (!a || !b || a->objectInstanceCount!=b->objectInstanceCount) return false;
        const auto& x=previous ? a->previousObjectInstances:a->objectInstances;
        const auto& y=previous ? b->previousObjectInstances:b->objectInstances;
        return std::memcmp(x.data(),y.data(),sizeof(HarmonicMIDI::grid::SceneTransform3D)*a->objectInstanceCount)==0;
    };
    // Raw passes key frames by stable id (node value + 1). Solid node 5 keys
    // by 6 and Render 3D node 4 keys by 5, so the two never share a frame key.
    check(render3dimage::renderStableIdForNodeValue(5)==6 && render3dimage::renderStableIdForNodeValue(4)==5,
        "Solid 3D raw passes use the Render 3D stable id rule");
    const auto preview=particleSolidRenderScene(drawn,9,false,error);
    const auto exported=particleSolidRenderScene(drawn,9,true,error);
    const ParticleSolidMotionPrior reviewed{before.state,before.births,now.births,p.bodyRadius};
    const auto direct=particleSolidScene(p,now.state,9,error,&reviewed);
    bool moved=false;
    for (std::size_t i=0;preview && i<preview->objectInstanceCount;++i) {
        const auto& a=preview->previousObjectInstances[i].translation; const auto& b=preview->objectInstances[i].translation;
        moved=moved || a.x!=b.x || a.y!=b.y || a.z!=b.z;
    }
    check(preview && preview->previousObjectInstanceCount==100 && moved
        && instances(preview,direct,true) && instances(preview,direct,false)
        && instances(preview,exported,true) && instances(preview,exported,false),
        "Motion View draws moving Solid 3D bodies with the same priors on preview and export");

    // Without a request the scene and native inputs are the Color draw.
    auto color=drawn; color.solidMotionView=false;
    const auto colorScene=particleSolidRenderScene(color,9,true,error);
    check(colorScene && colorScene->previousObjectInstanceCount==0 && instances(colorScene,preview,false),
        "a Solid 3D draw without a Motion request carries no prior transforms");
    arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
    check(particleSolidRuntimeInputs(color,true,false,inputs,error) && inputs.imageOutput==renderpassoutput::Output::Color
        && inputs.rawExportMask==0 && !inputs.previousMotion,
        "a Solid 3D draw without a Motion request keeps the default native inputs");
    check(particleSolidRuntimeInputs(drawn,false,false,inputs,error) && inputs.imageOutput==renderpassoutput::Output::Motion
        && inputs.rawExportMask==0 && inputs.previousMotion,
        "Motion View requests the Motion image with a predecessor sample");
    check(!particleSolidRuntimeInputs(drawn,false,true,inputs,error)
        && error=="HDR Solid 3D requires the Color image, not the Motion View",
        "HDR Solid 3D refuses the Motion View with the precise reason");

    // Export Motion Data alone keeps preview on the Color draw.
    auto exportOnly=color; exportOnly.solidExportMotion=true;
    check(!particleSolidMotionRequested(exportOnly,false) && particleSolidMotionRequested(exportOnly,true),
        "Export Motion Data needs the prior on export renders only");
    const auto exportPreview=particleSolidRenderScene(exportOnly,9,false,error);
    const auto exportRender=particleSolidRenderScene(exportOnly,9,true,error);
    check(exportPreview && exportPreview->previousObjectInstanceCount==0 && exportRender
        && exportRender->previousObjectInstanceCount==100 && instances(exportRender,preview,true),
        "Export Motion Data previews as Color and exports the same priors");
    check(particleSolidRuntimeInputs(exportOnly,false,false,inputs,error) && inputs.rawExportMask==0 && !inputs.previousMotion,
        "Export Motion Data previews with the default native inputs");
    check(particleSolidRuntimeInputs(exportOnly,true,false,inputs,error) && inputs.imageOutput==renderpassoutput::Output::Color
        && inputs.rawExportMask==(1u<<static_cast<unsigned>(renderpassoutput::Output::Motion)) && inputs.previousMotion,
        "Export Motion Data exports the raw Motion pass beside the Color image");

    auto missing=drawn; missing.preparedSolidMotion.reset();
    check(!particleSolidRenderScene(missing,9,false,error) && error=="Solid 3D Motion requires the replay one project frame earlier",
        "a Motion request without its prior replay is refused");

    // At clip start the prior frame is before the clip: every body is born
    // this frame, its prior is its current transform and static colliders
    // stand in for the prior ones.
    auto start=manySolidBodyFixture(100); start.solidMotionView=true;
    auto first=sampleSolidBodyMotion(start,0,error);
    first.state.colliders.pointCount=1; first.state.colliders.solidPoints[0]={0.1f,0.2f,0.3f};
    ParticlePointReplayTimes firstTimes; firstTimes.birthSeconds=first.births;
    start.historyProjectSeconds=0;
    const auto empty=replayParticleSolidMotionPrior(start,0,step,1,nullptr,first.state,firstTimes,start.bodyRadius,error);
    start.preparedSolids=std::make_shared<const ParticleSolidState>(first.state); start.preparedSolidMotion=empty;
    const auto firstScene=particleSolidRenderScene(start,9,false,error);
    check(empty && empty->previous.count==0 && firstScene
        && firstScene->previousObjectInstanceCount==firstScene->objectInstanceCount
        && std::memcmp(firstScene->previousObjectInstances.data(),firstScene->objectInstances.data(),
               sizeof(HarmonicMIDI::grid::SceneTransform3D)*firstScene->objectInstanceCount)==0,
        "the first clip frame reports zero Solid 3D Motion and keeps its static colliders");
    check(!replayParticleSolidMotionPrior(start,0.5,0,1,nullptr,first.state,firstTimes,start.bodyRadius,error)
        && error=="Solid 3D Motion requires a finite clip time, a project frame of at most 1 s and a positive body radius",
        "a Solid 3D Motion prior needs a positive project frame");

    // The prior replay cannot represent a timeline body mesh or Water.
    auto planar=p; planar.simulationSpace=0;
    check(!admitParticleSolidMotion(planar,error) && error=="Particle Motion output requires Solid 3D",
        "planar particles refuse the Solid 3D Motion prior");
    const auto descriptor=particleBodyTimelinePyramidDescriptor(error);
    auto timeline=p; auto body=std::make_shared<ParticleGeometryBinding>();
    if (descriptor) body->body=std::make_shared<const videowire::geometry::ValueDescriptor>(*descriptor);
    timeline.geometryBinding=body;
    check(descriptor && !admitParticleSolidMotion(timeline,error)
        && error=="Motion output for Solid 3D bodies with timeline body appearance is unavailable: the prior body mesh is not retained",
        "timeline body appearance refuses Solid 3D Motion before replay");
    auto water=std::make_shared<videowire::geometry::ValueDescriptor>();
    auto surface=waterreplayfixture::operation(1,0,0,0); surface.field.values[4]=1;
    water->operations.push_back(surface);
    auto coupled=p; auto mesh=std::make_shared<ParticleGeometryBinding>(); mesh->mesh=water;
    coupled.geometryBinding=mesh;
    check(!admitParticleSolidMotion(coupled,error)
        && error=="Motion output for Solid 3D bodies over moving colliders is unavailable: prior collider positions are not retained",
        "coupled Water refuses Solid 3D Motion before replay");
    check(!replayParticleSolidMotionPrior(coupled,0.5,step,1,nullptr,now.state,times,p.bodyRadius,error)
        && error=="Motion output for Solid 3D bodies over moving colliders is unavailable: prior collider positions are not retained",
        "the prior replay applies the same refusals");
    error.clear();
}

// This exercises the GraphDoc producer's six-operation Grid recipe through the
// real lowered-token protocol. GraphDoc compilation is verified in plugin tests;
// neither fixture is a cross-process or native-render acceptance receipt.
void checkGraphAuthoredTimelineBodyProtocol()
{
    using namespace videowire::geometry;
    using namespace videorender;
    std::string error;
    PortContract meshContract; meshContract.carrier=CarrierKind::geometry3D;
    meshContract.maxVertices=kMaximumVertices; meshContract.maxIndices=kMaximumIndices;
    PortContract vectorContract; vectorContract.carrier=CarrierKind::field;
    vectorContract.fieldDomain=Domain::vertex; vectorContract.fieldValueType=ValueType::vector;
    vectorContract.fieldInterpolation=Interpolation::constant; vectorContract.maxFieldElements=kMaximumFieldElements;
    auto scalarContract=vectorContract; scalarContract.fieldValueType=ValueType::floatValue;
    const auto mesh=lowerGrid(3,3,1,2,2,1,meshContract,error);
    const auto position=mesh ? lowerElementField(*mesh,5,OperationCode::positionField,vectorContract,error):std::nullopt;
    FieldOperationSettings settings; settings.domain=Domain::vertex; settings.valueType=ValueType::floatValue;
    settings.mode=3;
    const auto measure=position ? lowerConstructedField(6,OperationCode::fieldVectorMeasure,settings,
        &*position,nullptr,nullptr,scalarContract,error):std::nullopt;
    settings.mode=1; settings.values={4,0,0.25,0,1,0,0,0};
    const auto timeline=measure ? lowerConstructedField(7,OperationCode::fieldTimeline,settings,
        &*measure,nullptr,nullptr,scalarContract,error):std::nullopt;
    settings.valueType=ValueType::vector; settings.values={0,0,1,0,0,0,0,0};
    const auto vector=timeline ? lowerConstructedField(8,OperationCode::fieldComposeVector,settings,
        &*timeline,nullptr,nullptr,vectorContract,error):std::nullopt;
    const auto descriptor=vector ? lowerVectorDisplacement(*mesh,*vector,9,1,meshContract,error):std::nullopt;
    const auto admitted=descriptor ? admitValue(*descriptor,meshContract,error):std::nullopt;
    check(admitted && particleBodyGeometrySupportsTimelineReplay(admitted->descriptor()),
        "Graph-authored Grid recipe retains the supported six operations");
    if (!admitted) return;
    const auto token=encodeLoweredPlanText(lowerRuntimePlan(meshContract,*admitted));
    const auto payload="<NodeParams motionMode=\"2\" historicalReplay=\"1\" simulationSpace=\"1\" bodyShape=\"0\" "
        "count=\"4\" lifetime=\"2\" speed=\"0\" bodyRadius=\"0.03125\" bodyMass=\"2\" bodyGeometryPlan=\""+token+"\"/>";
    ParticleParams authored;
    check(videowire::admitParticleParameters(payload,authored,error),error.c_str());
    if (!error.empty() || !authored.geometryBinding || !authored.geometryBinding->bodySource) return;
    const auto source=authored.geometryBinding->bodySource;
    check(source->descriptor().sourceStableId==3 && source->descriptor().sourceRevision==1
        && source->descriptor().stableId==9 && source->receipt().deterministicDigest==admitted->receipt().deterministicDigest
        && source->descriptor().operations.size()==6,
        "actual decoder preserves Grid source identity, recipe and admitted digest");
    auto score=std::make_shared<arbitmod::Score>(); score->scoreRevision=1;
    auto history=std::make_shared<ParticleHistorySource>(); history->score=score;
    auto& key=history->key;
    key.projectGeneration=1; key.helperGeneration=2; key.deviceGeneration=3;
    key.sourceGeneration=key.backendGeneration=key.scoreGeneration=key.beatMapGeneration=1;
    key.fpsGeneration=key.loopGeneration=key.seekGeneration=1; key.fps=60; key.frame=60; key.beat=2;
    canonicalblockc::FrameProducer producer;
    const auto caller=makeParticleBodyCallerLedger({},error);
    auto context=makeParticleBodyReplayContext(caller,error);
    check(context!=nullptr,"Grid handoff uses a factory-admitted current replay context");
    if (!context) return;
    context->owner={1,2,3,4,17,videohelper::geometry::PlanUse::preview}; context->width=context->height=96;
    context->frame=producer.evaluate(key,score,2);
    authored.bodyReplayContext=context; authored.history=history;
    authored.historyClipId=4; authored.historyNodeId=1; authored.geometryRevision=17;
    const auto requestAt=[&](double seconds,videohelper::geometry::PlanUse use) {
        auto p=authored;
        auto currentHistory=std::make_shared<ParticleHistorySource>(*history);
        currentHistory->key.frame=static_cast<std::int64_t>(std::llround(seconds*60));
        currentHistory->key.beat=currentHistory->timeline.secondsToBeat(seconds);
        auto currentContext=makeParticleBodyReplayContext(caller,error);
        if (!currentContext) { p.bodyReplayContext.reset(); return p; }
        currentContext->owner=context->owner; currentContext->owner.use=use;
        currentContext->width=context->width; currentContext->height=context->height;
        currentContext->frame=producer.evaluate(currentHistory->key,score,currentHistory->key.beat);
        p.history=currentHistory; p.bodyReplayContext=currentContext;
        return p;
    };
    const auto run=[&](ParticleParams p,double seconds) {
        if (!p.bodyReplayContext) return std::shared_ptr<const ParticleBodyReplayReceipt>{};
        p.historyProjectSeconds=seconds;
        std::shared_ptr<const ParticleBodyReplayReceipt> receipt;
        ParticleSolidState solids;
        replayParticleBodies(p,seconds,1,p.bodyReplayContext->frame.get(),&error,&solids,nullptr,&receipt);
        return receipt;
    };
    const auto zeroRequest=requestAt(0,videohelper::geometry::PlanUse::preview);
    const auto oneRequest=requestAt(1,videohelper::geometry::PlanUse::preview);
    const auto zero=run(zeroRequest,0),one=run(oneRequest,1);
    check(error.empty() && zero && one && zero->solids()->terminalBodySample && one->solids()->terminalBodySample,
        "decoded Grid body reaches genuine completed replay and terminal sample receipts");
    if (!zero || !one || !zero->solids()->terminalBodySample || !one->solids()->terminalBodySample) return;
    const auto a=zero->solids()->terminalBodySample,b=one->solids()->terminalBodySample;
    check(a->source()==source && b->source()==source && a->absoluteTick()==0 && b->absoluteTick()==28224000
        && b->structuralRevision()==1 && authored.geometryRevision==17 && one->sampler()->usage().attempts==121,
        "Grid replay binds exact source revision separately from owner revision and absolute project ticks");
    const std::vector<StableId> ids{1,2,3,4};
    const std::vector<std::uint32_t> indices{0,1,2,1,3,2};
    check(a->prototype().vertexIds==ids && b->prototype().vertexIds==ids
        && a->prototype().indices==indices && b->prototype().indices==indices,
        "Grid Timeline sampling retains literal topology and stable identities");
    check(a->prototype().positions.size()==4 && b->prototype().positions.size()==4,
        "Grid body samples retain four normalized positions");
    if (a->prototype().positions.size()!=4 || b->prototype().positions.size()!=4) return;
    // Independent normalized coordinates: centers z=1/2 and 1/4,
    // enclosing radii sqrt(41)/8 and sqrt(33)/8 respectively.
    const auto close=[](float actual,double expected) { return std::abs(double(actual)-expected)<2.e-6; };
    check(close(a->prototype().positions[0].x,-4/std::sqrt(41.0))
        && close(a->prototype().positions[0].z,3/std::sqrt(41.0))
        && close(b->prototype().positions[0].x,-4/std::sqrt(33.0))
        && close(b->prototype().positions[0].z,-1/std::sqrt(33.0))
        && close(a->center().z,.5) && close(b->center().z,.25),
        "time zero and one produce independently known different normalized Grid appearances");
    const auto sought=run(zeroRequest,0),repeated=run(oneRequest,1);
    check(error.empty() && sought && repeated
        && equal(sought->solids()->terminalBodySample->prototype(),a->prototype())
        && equal(repeated->solids()->terminalBodySample->prototype(),b->prototype())
        && sameSolidBodies(*repeated->solids(),*one->solids()),
        "seeking and repeated replay preserve Grid appearance and body state");
    const auto exporting=requestAt(1,videohelper::geometry::PlanUse::exportRender);
    const auto exported=run(exporting,1);
    check(error.empty() && exported && sameSolidBodies(*exported->solids(),*one->solids())
        && equal(exported->solids()->terminalBodySample->prototype(),b->prototype()),
        "separate current Export owner retains Grid replay parity");
    auto stale=oneRequest; auto changed=std::make_shared<ParticleHistorySource>(*oneRequest.history);
    changed->key.seekGeneration++; stale.history=changed;
    check(!run(stale,1) && !error.empty(),"Grid replay refuses a history key that differs from its canonical context");
    stale=oneRequest; stale.history.reset();
    check(!run(stale,1) && !error.empty(),"Grid Timeline body refuses missing genuine project history");
    check(authored.bodyRadius==.03125f && authored.bodyMass==2 && authored.bodyShape==0,
        "time-dependent appearance conserves authored sphere proxy radius and mass");
}

void checkDynamicBodyProductionJoin()
{
    using namespace videorender;
    using namespace videowire::geometry;
    std::string error;
    const auto descriptor=particleBodyTimelinePyramidDescriptor(error);
    const auto body=descriptor ? admitValue(*descriptor,particleBodyGeometryFixtureContract(),error):std::nullopt;
    const auto rest=particleBodyPyramidDescriptor();
    PortContract pointContract; pointContract.carrier=CarrierKind::points3D; pointContract.maxPoints=64;
    const auto points=lowerPointsFromVertices(rest,85001,pointContract,error);
    const auto emitters=points ? admitValue(*points,pointContract,error):std::nullopt;
    check(body.has_value() && emitters.has_value(),"production body and emitter sources admit");
    if (!body || !emitters) return;
    const auto payload="<NodeParams motionMode=\"2\" historicalReplay=\"1\" simulationSpace=\"1\" count=\"5\" "
        "lifetime=\"2\" speed=\"0\" bodyMass=\"2\" bodyRadius=\"0.015\" geometryScale=\"0.1\" "
        "bodyGeometryPlan=\""+encodeLoweredPlanText(lowerRuntimePlan(particleBodyGeometryFixtureContract(),*body))
        +"\" emitterGeometryPlan=\""+encodeLoweredPlanText(lowerRuntimePlan(pointContract,*emitters))+"\"/>";
    ParticleParams authored;
    check(videowire::admitParticleParameters(payload,authored,error),error.c_str());
    if (!error.empty() || !authored.geometryBinding || !authored.geometryBinding->bodySource) return;
    auto history=std::make_shared<ParticleHistorySource>(); history->score=std::make_shared<const arbitmod::Score>();
    history->key.projectGeneration=1; history->key.helperGeneration=2; history->key.deviceGeneration=3;
    authored.history=history; authored.historyClipId=4; authored.historyNodeId=7; authored.geometryRevision=5;
    const auto caller=makeParticleBodyCallerLedger({},error);
    auto context=makeParticleBodyReplayContext(caller,error);
    check(context!=nullptr,"production replay admits its persistent caller context before allocation");
    if (!context) return;
    context->owner={1,2,3,4,5,videohelper::geometry::PlanUse::preview}; context->width=context->height=96;
    authored.bodyReplayContext=context;
    const auto makeConfiguredContext=[&](const ParticleBodyReplayContext& source) {
        auto admitted=makeParticleBodyReplayContext(source.caller,error);
        check(admitted!=nullptr,"independent replay context reserves its own payload before configuration");
        if (admitted) {
            admitted->owner=source.owner; admitted->width=source.width; admitted->height=source.height;
            admitted->frame=source.frame; admitted->admissionLimits=source.admissionLimits;
            admitted->sampleLimits=source.sampleLimits;
        }
        return admitted;
    };
    const auto run=[&](ParticleParams p,double clipSeconds,double projectSeconds) {
        p.historyProjectSeconds=projectSeconds;
        p.preparedBodies.reset(); p.preparedSolids.reset(); p.preparedBodyReplay.reset();
        std::shared_ptr<const ParticleBodyReplayReceipt> receipt;
        ParticleSolidState solids;
        replayParticleBodies(p,clipSeconds,1,p.bodyReplayContext->frame.get(),&error,&solids,nullptr,&receipt);
        if (receipt && error.empty()) {
            p.preparedBodyReplay=receipt; p.preparedBodies=receipt->bodies(); p.preparedSolids=receipt->solids();
        }
        return p;
    };
    auto current=run(authored,.75,.75);
    check(error.empty() && current.preparedBodyReplay && current.preparedSolids->count==5,
        "authentic decoded pyramid flows through production replay");
    if (!error.empty() || !current.preparedBodyReplay) return;
    const auto receipt=current.preparedBodyReplay;
    const auto sample=current.preparedSolids->terminalBodySample;
    check(sample && sample->absoluteTick()==21168000 && sample->structuralRevision()==7
        && sample->source()==authored.geometryBinding->bodySource && current.geometryRevision==5
        && receipt->sampler()->usage().attempts==91 && sample->admission()->recipe().replayBoundaries==91,
        "terminal absolute tick, distinct revisions and one attempt per unique boundary survive replay");
    auto scene=particleSolidScene(current,*current.preparedSolids,7,error);
    check(scene && scene->objectCount==1 && scene->objectInstanceCount==5
        && scene->instancedObject==scene->objects[0].id && scene->vertexCount==11 && scene->indexCount==42,
        "five dynamic bodies are instances of one five-vertex eighteen-index prototype");
    if (!scene) return;
    {
        const std::array<double,kParticleSolidMaxBodies> births {};
        const ParticleSolidMotionPrior prior{*current.preparedSolids,births,births,current.bodyRadius};
        check(!particleSolidScene(current,*current.preparedSolids,7,error,&prior)
            && error=="Motion output for Solid 3D bodies with timeline body appearance is unavailable: the prior body mesh is not retained",
            "dynamic body appearance refuses Solid 3D Motion with the precise reason");
        error.clear();
    }
    const std::array<HarmonicMIDI::grid::SceneVec3,5> literal{{
        {-.2425356250363329735f,-.4850712500726659470f,-.4850712500726659470f},
        {.7276068751089989205f,-.4850712500726659470f,-.4850712500726659470f},
        {.7276068751089989205f,.4850712500726659470f,-.4850712500726659470f},
        {-.2425356250363329735f,.4850712500726659470f,-.4850712500726659470f},
        {-.7276068751089989205f,0,.4850712500726659470f}}};
    for (std::size_t i=0;i<literal.size();++i) {
        const auto& actual=scene->vertices[6+i].position;
        check(std::abs(actual.x-literal[i].x)<2.e-6f && std::abs(actual.y-literal[i].y)<2.e-6f
            && std::abs(actual.z-literal[i].z)<2.e-6f,"production terminal appearance matches independent literal pyramid");
        const auto& object=scene->objects[0];
        check(object.firstVertex==6 && object.vertexCount==5 && object.firstIndex==24 && object.indexCount==18
            && scene->objectInstances[i].scale.x==authored.bodyRadius,"body instances share the prototype and retain authored radius");
    }
    auto staticParameters=authored;
    auto staticBinding=std::make_shared<ParticleGeometryBinding>(*authored.geometryBinding);
    staticBinding->body=std::make_shared<const ValueDescriptor>(rest); staticBinding->bodySource.reset();
    staticParameters.geometryBinding=staticBinding;
    staticParameters.historyProjectSeconds=.75;
    // Static replay retains its existing scalar publication interface.
    ParticleSolidState staticState;
    replayParticleBodies(staticParameters,.75,1,nullptr,&error,&staticState);
    check(error.empty() && sameSolidBodies(*current.preparedSolids,staticState)
        && current.bodyRadius==.015f && current.bodyMass==2 && current.collisionMode==0,
        "dynamic appearance preserves positions, authored mass, radius and sphere contacts");
    auto repeated=run(authored,.75,.75);
    check(error.empty() && repeated.preparedSolids && sameSolidBodies(*current.preparedSolids,*repeated.preparedSolids)
        && equal(sample->prototype(),repeated.preparedSolids->terminalBodySample->prototype()),
        "repeat and seek reproduce terminal appearance and body positions");
    auto cycled=run(authored,2.75,2.75);
    check(error.empty() && cycled.preparedBodyReplay && cycled.preparedSolids->terminalBodySample->absoluteTick()==77616000
        && sameSolidBodies(*current.preparedSolids,*cycled.preparedSolids)
        && equal(sample->prototype(),cycled.preparedSolids->terminalBodySample->prototype()),
        "cycle reset uses the absolute cycle origin while preserving body identity and terminal appearance");
    auto fractional=run(authored,1.0/240,1.0/240);
    check(error.empty() && fractional.preparedBodyReplay && fractional.preparedSolids->terminalBodySample->absoluteTick()==117600
        && fractional.preparedBodyReplay->sampler()->usage().attempts==2,"fractional terminal adds one unique boundary");
    auto zero=run(authored,0,0);
    check(error.empty() && zero.preparedBodyReplay && zero.preparedBodyReplay->sampler()->usage().attempts==1,"zero age evaluates one sample");
    auto heldAuthored=authored; heldAuthored.resetMode=1; heldAuthored.lifetime=.75f;
    auto held=run(heldAuthored,1.5,1.5);
    check(error.empty() && held.preparedBodyReplay && held.preparedSolids->terminalBodySample->absoluteTick()==21168000
        && equal(sample->prototype(),held.preparedSolids->terminalBodySample->prototype()),"held age retains the actual terminal appearance tick");
    auto resetAuthored=authored; resetAuthored.resetTime=.25f;
    auto reset=run(resetAuthored,1,1);
    check(error.empty() && reset.preparedBodyReplay && reset.preparedSolids->terminalBodySample->absoluteTick()==28224000,
        "reset origin shifts the complete absolute sample schedule");
    auto before=run(resetAuthored,.1,.1);
    check(error.empty() && before.preparedBodyReplay && before.preparedSolids->count==0 && !before.preparedSolids->terminalBodySample
        && !before.preparedBodyReplay->sampler(),"before reset publishes empty state without evaluating appearance");
    const auto refuses=[&](const ParticleParams& p,double time=.75) {
        return !receipt->matchesPrepared(p,time,1,p.bodyReplayContext ? p.bodyReplayContext->frame.get():nullptr);
    };
    auto stale=current; stale.historyProjectSeconds=.5;
    check(refuses(stale),"changed project clock cannot reuse dynamic prepared state");
    check(refuses(current,.5),"changed clip clock cannot reuse dynamic prepared state");
    const auto savedHistoryKey=history->key;
    history->key.seekGeneration++;
    check(refuses(current),"full history frame key is frozen even behind the same source pointer");
    history->key=savedHistoryKey;
    stale=current; auto soughtHistory=std::make_shared<ParticleHistorySource>(*history);
    soughtHistory->key.seekGeneration++; stale.history=soughtHistory;
    check(refuses(stale),"seek lifecycle cannot reuse dynamic prepared state");
    stale=current; auto replacedBinding=std::make_shared<ParticleGeometryBinding>(*authored.geometryBinding);
    replacedBinding->bodySource=particleBodyFixtureAdmitted(*descriptor,error); stale.geometryBinding=replacedBinding;
    check(refuses(stale),"equal source bytes in another decoded token cannot reuse prepared state");
    stale=current; stale.preparedSolids=std::make_shared<const ParticleSolidState>(*current.preparedSolids);
    check(refuses(stale) && !particleSolidScene(stale,*stale.preparedSolids,7,error),
        "copied prepared solid state cannot substitute the privately issued pair");
    stale=current; stale.preparedBodies=std::make_shared<const ParticleBodyState>(*current.preparedBodies);
    check(refuses(stale),"copied prepared body state cannot substitute the privately issued pair");
    ParticleSolidState rejectedPrepared;
    replayParticleBodies(stale,.75,1,nullptr,&error,&rejectedPrepared);
    check(!error.empty() && rejectedPrepared.count==0,"production prepared fastpath refuses an equal copied body array");
    std::string cachedError;
    ParticleSolidState cachedState;
    const auto cached=replayParticleBodies(current,.75,1,nullptr,&cachedError,&cachedState);
    check(cachedError.empty() && cached==*current.preparedBodies && cachedState.terminalBodySample==sample
        && receipt->sampler()->usage().attempts==91,"exact prepared replay reuses its retained terminal lease without another attempt");
    const auto unavailableTimesOwned=heapReplayState(); auto& unavailableTimes=*unavailableTimesOwned;
    check(!unavailableTimes.prepare(current,.75,1,nullptr,error,true,true),
        "prepared direct receipt does not call borrowed history again for unretained point-time outputs");
    stale=current; auto otherContext=makeConfiguredContext(*context);
    otherContext->owner.use=videohelper::geometry::PlanUse::exportRender; stale.bodyReplayContext=otherContext;
    check(refuses(stale),"preview receipt cannot substitute export ownership");
    for (int generation=0;generation<5;++generation) {
        otherContext=makeConfiguredContext(*context);
        auto& owner=otherContext->owner;
        if (generation==0) owner.projectGeneration++;
        else if (generation==1) owner.helperGeneration++;
        else if (generation==2) owner.deviceGeneration++;
        else if (generation==3) owner.clipIdentity++;
        else owner.planGeneration++;
        stale=current; stale.bodyReplayContext=otherContext;
        check(refuses(stale),"each replay owner generation is bound to the actual completed request");
    }
    stale=current; stale.historyNodeId++;
    check(refuses(stale),"another particle node cannot claim the completed replay");
    otherContext=makeConfiguredContext(*context); otherContext->width=97;
    stale=current; stale.bodyReplayContext=otherContext;
    check(refuses(stale) && !receipt->matchesRender(current,context->owner,7,97,96),"extent substitution refuses before native rendering");
    auto framed=authored; auto frameHistory=std::make_shared<ParticleHistorySource>(*history);
    auto score=std::make_shared<arbitmod::Score>(); score->scoreRevision=1; frameHistory->score=score;
    auto& key=frameHistory->key;
    key.sourceGeneration=key.backendGeneration=key.scoreGeneration=key.beatMapGeneration=1;
    key.fpsGeneration=key.loopGeneration=key.seekGeneration=1; key.fps=60; key.frame=45; key.beat=1.5;
    canonicalblockc::FrameProducer frameProducer;
    auto framedContext=makeConfiguredContext(*context);
    framedContext->frame=frameProducer.evaluate(key,score,1.5f); framed.history=frameHistory; framed.bodyReplayContext=framedContext;
    auto framedResult=run(framed,.75,.75);
    check(error.empty() && framedResult.preparedBodyReplay && framedContext->frame,"dynamic replay retains a genuine full-key canonical frame");
    if (framedResult.preparedBodyReplay) {
        auto frameReplacement=makeConfiguredContext(*framedContext);
        auto changedKey=key; changedKey.frame++; frameReplacement->frame=frameProducer.evaluate(changedKey,score,1.5f);
        framedResult.bodyReplayContext=frameReplacement;
        check(!framedResult.preparedBodyReplay->matchesPrepared(framedResult,.75,1,frameReplacement->frame.get()),
            "another canonical frame cannot replace the retained exact frame");
    }
    std::vector<arbitblockb::FeatureFrame> events(2);
    events[0].sampleIndex=200; events[1].sampleIndex=400;
    for (auto& event:events) { event.onset=1; event.onsetAge=0; }
    auto eventHistory=std::make_shared<ParticleHistorySource>(*history);
    eventHistory->audioFrames=&events; eventHistory->audioSampleRate=48000;
    auto eventAuthored=authored; eventAuthored.history=eventHistory;
    auto eventResult=run(eventAuthored,.75,.75);
    check(error.empty() && eventResult.preparedBodyReplay && eventResult.preparedBodyReplay->sampler()->usage().attempts==92
        && eventResult.preparedSolids->terminalBodySample->admission()->recipe().replayBoundaries==92,
        "actual onset event boundaries are sorted and deduplicated against regular steps");
    std::vector<arbitblockb::FeatureFrame> crowded(513);
    for (std::size_t i=0;i<crowded.size();++i) { crowded[i].sampleIndex=i+1; crowded[i].onset=1; crowded[i].onsetAge=0; }
    auto crowdedHistory=std::make_shared<ParticleHistorySource>(*history);
    crowdedHistory->audioFrames=&crowded; crowdedHistory->audioSampleRate=48000;
    eventAuthored.history=crowdedHistory;
    auto crowdedResult=run(eventAuthored,.75,.75);
    check(!error.empty() && !crowdedResult.preparedBodyReplay,"event overflow refuses before body sampling");
    auto shortSchedule=makeConfiguredContext(*context);
    shortSchedule->admissionLimits.maxReplayBoundaries=90;
    auto shortAuthored=authored; shortAuthored.bodyReplayContext=shortSchedule;
    const auto shortResult=run(shortAuthored,.75,.75);
    check(!error.empty() && !shortResult.preparedBodyReplay,"actual ninety-one-boundary schedule exceeds a ninety-boundary admission");
    auto shortWork=makeConfiguredContext(*context);
    shortWork->sampleLimits.maxTotalVisits=receipt->sampler()->recipe().totalVisits-1;
    shortAuthored.bodyReplayContext=shortWork;
    const auto workResult=run(shortAuthored,.75,.75);
    check(!error.empty() && !workResult.preparedBodyReplay,"whole actual schedule work shortage refuses before sampling");
    auto shortContext=makeConfiguredContext(*context);
    const auto& recipe=receipt->sampler()->recipe();
    shortContext->sampleLimits.maxRetainedLogicalBytes=recipe.retainedBaseLogicalBytes+recipe.retainedSampleLogicalBytes;
    auto shortRequest=authored; shortRequest.historyProjectSeconds=.75; shortRequest.bodyReplayContext=shortContext;
    const auto failedOwned=heapReplayState(); auto& failed=*failedOwned;
    check(failed.prepare(shortRequest,.75,1,nullptr,error,true) && failed.advanceOne(error) && !failed.advanceOne(error),
        "incumbent plus replacement sample storage is charged before the second evaluation");
    check(failed.geometrySampler() && failed.geometrySampler()->usage().attempts==1,
        "storage refusal does not re-evaluate the initial cached tick or consume an unreserved attempt");
    ParticleBodyState conserved=*current.preparedBodies; ParticleSolidState conservedSolid=*current.preparedSolids;
    const auto escaped=conservedSolid.terminalBodySample;
    check(!failed.publish(conserved,error,&conservedSolid) && conserved==*current.preparedBodies
        && conservedSolid.terminalBodySample==escaped,"failed later sample preserves incumbent caller outputs");
    auto escaping=run(authored,.75,.75);
    check(error.empty() && escaping.preparedBodyReplay,"independent escaping terminal replay completes");
    if (escaping.preparedBodyReplay) {
        const auto escapedSampler=escaping.preparedBodyReplay->sampler();
        auto escapedSample=escaping.preparedSolids->terminalBodySample;
        const auto retainedBefore=escapedSampler->usage().retainedLogicalBytes;
        escaping={};
        check(escapedSample && equal(escapedSample->prototype(),sample->prototype())
            && escapedSampler->usage().retainedLogicalBytes==retainedBefore,
            "escaped terminal lease remains charged after its completed pair and receipt are released");
        escapedSample.reset();
        check(escapedSampler->usage().retainedLogicalBytes==escapedSampler->recipe().retainedBaseLogicalBytes,
            "only releasing the final escaped sample releases its retained sample storage");
    }
    // Caller accounting is independent of the exact sampler schedule budget.
    // These are genuine decoded production requests, not hand-made receipts.
    const auto leaseBytes=sizeof(ParticleBodyCallerLease);
    const auto contextBytes=particleBodyContextLogicalBytes+leaseBytes;
    const auto contextWork=sizeof(ParticleBodyReplayContext)+3*sizeof(ParticleParams);
    ParticleBodyCallerLimits contextStorageLimits;
    contextStorageLimits.maxLogicalBytes=sizeof(ParticleBodyCallerLedger)+contextBytes-1;
    auto contextStorageCaller=makeParticleBodyCallerLedger(contextStorageLimits,error);
    auto missingContext=authored; missingContext.historyProjectSeconds=.75;
    missingContext.bodyReplayContext=makeParticleBodyReplayContext(contextStorageCaller,error);
    const auto missingContextReplayOwned=heapReplayState(); auto& missingContextReplay=*missingContextReplayOwned;
    check(!missingContext.bodyReplayContext && !missingContextReplay.prepare(missingContext,.75,1,nullptr,error,true)
        && !missingContextReplay.geometrySampler() && !missingContextReplay.receipt()
        && contextStorageCaller->usage().logicalBytes==sizeof(ParticleBodyCallerLedger)
        && contextStorageCaller->usage().work==contextWork,
        "direct replay context one byte short refuses before payload allocation, parameter copies or sampling");
    auto arbitraryLease=contextStorageCaller->reserve(1,0);
    missingContext.bodyReplayContext=makeParticleBodyReplayContext(contextStorageCaller,error);
    check(arbitraryLease && !missingContext.bodyReplayContext
        && !missingContextReplay.prepare(missingContext,.75,1,nullptr,error,true)
        && !missingContextReplay.receipt()
        && contextStorageCaller->usage().logicalBytes==sizeof(ParticleBodyCallerLedger)+1+leaseBytes
        && contextStorageCaller->usage().work==2*contextWork,
        "an arbitrary same-ledger reservation cannot replace the factory context envelope or reset failed work");
    arbitraryLease.reset();
    ParticleBodyCallerLimits contextWorkLimits; contextWorkLimits.maxWork=contextWork-1;
    auto contextWorkCaller=makeParticleBodyCallerLedger(contextWorkLimits,error);
    check(!makeParticleBodyReplayContext(contextWorkCaller,error)
        && contextWorkCaller->usage().logicalBytes==sizeof(ParticleBodyCallerLedger)
        && contextWorkCaller->usage().work==contextWorkLimits.maxWork
        && !makeParticleBodyReplayContext(contextWorkCaller,error)
        && contextWorkCaller->usage().work==contextWorkLimits.maxWork,
        "context construction work one short refuses before allocation and stays exhausted on retry");
    const auto beforeIndependent=caller->usage();
    auto independentContext=makeConfiguredContext(*context);
    check(independentContext && independentContext!=context && independentContext->caller==context->caller
        && caller->usage().logicalBytes==beforeIndependent.logicalBytes+contextBytes
        && caller->usage().work==beforeIndependent.work+contextWork,
        "same-config factory contexts on one ledger each retain the exact context envelope and construction work");
    if (!independentContext) return;
    auto substituted=current; substituted.bodyReplayContext=independentContext;
    const auto substitutedReplayOwned=heapReplayState(); auto& substitutedReplay=*substitutedReplayOwned;
    check(refuses(substituted) && !substitutedReplay.prepare(substituted,.75,1,nullptr,error,true)
        && !substitutedReplay.receipt(),
        "another admitted same-ledger context cannot substitute the completed context in direct replay");
    auto independentRequest=authored; independentRequest.historyProjectSeconds=.75;
    independentRequest.bodyReplayContext=independentContext;
    const auto independentReplayOwned=heapReplayState(); auto& independentReplay=*independentReplayOwned;
    const auto independentlyPrepared=independentReplay.prepare(independentRequest,.75,1,nullptr,error,true);
    if (independentlyPrepared) while (!independentReplay.complete() && independentReplay.advanceOne(error)) {}
    check(independentlyPrepared && independentReplay.complete() && independentReplay.receipt()
        && independentReplay.receipt()!=receipt,
        "an independently admitted factory context produces its own genuine direct-entry completed receipt");
    auto foreignCaller=makeParticleBodyCallerLedger({},error);
    auto foreignContext=makeParticleBodyReplayContext(foreignCaller,error);
    check(foreignContext!=nullptr,"foreign ledger independently admits its context");
    if (!foreignContext) return;
    foreignContext->owner=context->owner; foreignContext->width=context->width; foreignContext->height=context->height;
    foreignContext->frame=context->frame; foreignContext->admissionLimits=context->admissionLimits;
    foreignContext->sampleLimits=context->sampleLimits;
    substituted=current; substituted.bodyReplayContext=foreignContext;
    check(refuses(substituted) && !substitutedReplay.prepare(substituted,.75,1,nullptr,error,true)
        && !substitutedReplay.receipt(),
        "another ledger's admitted context cannot substitute the completed context in direct replay");
    const auto scratchBytes=ParticleBodyReplayState::callerScratchLogicalBytes()+leaseBytes;
    const auto retainedBytes=ParticleBodyReplayReceipt::retainedLogicalBytes()+leaseBytes;
    const auto outputBytes=ParticleBodyReplayState::callerOutputLogicalBytes(true,false)+leaseBytes;
    const auto exactPeak=sizeof(ParticleBodyCallerLedger)+contextBytes+scratchBytes+retainedBytes+outputBytes;
    const auto makeBoundedRequest=[&](const std::shared_ptr<ParticleBodyCallerLedger>& ledger) {
        auto request=authored;
        auto admittedContext=makeParticleBodyReplayContext(ledger,error);
        if (!admittedContext) { request.bodyReplayContext.reset(); return request; }
        admittedContext->owner=context->owner; admittedContext->width=admittedContext->height=96;
        admittedContext->admissionLimits.maxReplayBoundaries=91;
        admittedContext->sampleLimits.maxTotalVisits=recipe.totalVisits;
        admittedContext->sampleLimits.maxRetainedLogicalBytes=recipe.retainedBaseLogicalBytes+2*recipe.retainedSampleLogicalBytes;
        admittedContext->sampleLimits.maxAttemptLogicalBytes=recipe.retainedBaseLogicalBytes
            +std::max(recipe.derivationLogicalBytes,recipe.retainedSampleLogicalBytes+recipe.attemptLogicalBytes);
        request.bodyReplayContext=std::move(admittedContext); request.historyProjectSeconds=.75; return request;
    };
    ParticleBodyCallerLimits exactLimits; exactLimits.maxLogicalBytes=exactPeak;
    auto exactCaller=makeParticleBodyCallerLedger(exactLimits,error);
    auto exactRequest=makeBoundedRequest(exactCaller);
    auto exactResult=run(exactRequest,.75,.75);
    check(error.empty() && exactResult.preparedBodyReplay && exactResult.preparedBodyReplay->sampler()->usage().attempts==91
        && exactCaller->usage().peakLogicalBytes==exactPeak,
        "exact caller peak and exact genuine ninety-one-sample budgets admit together");
    if (!exactResult.preparedBodyReplay) return;
    auto shortageLimits=exactLimits; --shortageLimits.maxLogicalBytes;
    auto shortageCaller=makeParticleBodyCallerLedger(shortageLimits,error);
    auto shortageRequest=makeBoundedRequest(shortageCaller);
    const auto callerShortageOwned=heapReplayState(); auto& callerShortage=*callerShortageOwned;
    check(!callerShortage.prepare(shortageRequest,.75,1,nullptr,error,true)
        && !callerShortage.geometrySampler() && !callerShortage.receipt()
        && shortageCaller->usage().logicalBytes==sizeof(ParticleBodyCallerLedger)+contextBytes,
        "one byte below the simultaneous caller envelope refuses before parameter and receipt copies or sampling");
    const auto failedCallerWork=shortageCaller->usage().work;
    check(!callerShortage.prepare(shortageRequest,.75,1,nullptr,error,true)
        && shortageCaller->usage().work>failedCallerWork,
        "retrying caller storage refusal does not reset attempted work");
    const auto validationBytes=ParticleBodyReplayReceipt::validationLogicalBytes()+leaseBytes;
    const auto exactOccupied=exactCaller->usage().logicalBytes;
    auto comparisonStorage=exactCaller->reserve(exactPeak-exactOccupied-validationBytes+1-leaseBytes,1);
    check(comparisonStorage && !exactResult.preparedBodyReplay->matchesPrepared(exactResult,.75,1,nullptr)
        && exactCaller->usage().logicalBytes==exactPeak-validationBytes+1,
        "one byte short reserves no comparison scratch and performs no full source or parameter comparison");
    const auto storageRefusalWork=exactCaller->usage().work;
    comparisonStorage.reset();
    check(exactCaller->usage().logicalBytes==exactOccupied && exactCaller->usage().work==storageRefusalWork
        && exactResult.preparedBodyReplay->matchesPrepared(exactResult,.75,1,nullptr),
        "comparison storage refund permits retry while all failed attempted work remains charged");
    ParticleBodyCallerLimits prepareCopyLimits;
    prepareCopyLimits.maxWork=sizeof(ParticleBodyReplayContext)+3*sizeof(ParticleParams)
        +ParticleBodyReplayState::callerPrepareWork()-1;
    auto prepareCopyCaller=makeParticleBodyCallerLedger(prepareCopyLimits,error);
    auto prepareCopyRequest=makeBoundedRequest(prepareCopyCaller);
    const auto prepareCopyShortageOwned=heapReplayState(); auto& prepareCopyShortage=*prepareCopyShortageOwned;
    check(!prepareCopyShortage.prepare(prepareCopyRequest,.75,1,nullptr,error,true)
        && !prepareCopyShortage.geometrySampler() && prepareCopyCaller->usage().work==prepareCopyLimits.maxWork,
        "initial parameter-copy allowance one short refuses before caller parameter copies");
    const auto initialWork=sizeof(ParticleBodyReplayContext)+3*sizeof(ParticleParams)
        +ParticleBodyReplayState::callerPrepareWork()+ParticleBodyReplayState::callerReceiptWork()
        +ParticleBodyReplayState::callerOutputLogicalBytes(true,false);
    ParticleBodyCallerLimits parameterLimits;
    parameterLimits.maxWork=initialWork+ParticleBodyReplayState::callerScheduleCopyWork(91)-1;
    auto parameterCaller=makeParticleBodyCallerLedger(parameterLimits,error);
    auto parameterRequest=makeBoundedRequest(parameterCaller);
    const auto parameterShortageOwned=heapReplayState(); auto& parameterShortage=*parameterShortageOwned;
    check(!parameterShortage.prepare(parameterRequest,.75,1,nullptr,error,true)
        && !parameterShortage.geometrySampler() && parameterCaller->usage().work==parameterLimits.maxWork,
        "one work unit short refuses the complete scheduled parameter-copy allowance before the first sample");
    check(!parameterShortage.prepare(parameterRequest,.75,1,nullptr,error,true)
        && parameterCaller->usage().work==parameterLimits.maxWork,
        "exhausted parameter-copy work remains exhausted on retry");

    const auto completedWork=initialWork+ParticleBodyReplayState::callerScheduleCopyWork(91);
    ParticleBodyCallerLimits validationLimits;
    validationLimits.maxWork=completedWork+receipt->validationWork()-1;
    auto validationCaller=makeParticleBodyCallerLedger(validationLimits,error);
    auto validationRequest=makeBoundedRequest(validationCaller);
    auto validationResult=run(validationRequest,.75,.75);
    check(error.empty() && validationResult.preparedBodyReplay,
        "bounded caller completes the genuine request before validation shortage");
    if (validationResult.preparedBodyReplay) {
        auto invalid=validationResult; invalid.bodyRadius=.02f;
        const auto storageBefore=validationCaller->usage().logicalBytes;
        check(!validationResult.preparedBodyReplay->matchesPrepared(invalid,.75,1,nullptr)
            && validationCaller->usage().work==validationLimits.maxWork
            && validationCaller->usage().logicalBytes==storageBefore,
            "one comparison-copy work unit short refuses before full validation and refunds no live payload");
        check(!validationResult.preparedBodyReplay->matchesPrepared(validationResult,.75,1,nullptr)
            && validationCaller->usage().work==validationLimits.maxWork,
            "repeated exact validation cannot refresh failed comparison work");
    }
    struct CallerRenderLayer {
        float opacity=1; bool particleSource=true; unsigned texture=0;
        int texWidth=0,texHeight=0; std::string nativeTextureBackend;
        std::uintptr_t nativeTextureView=0;
        arbitgpu::NativeTextureViewDescriptor nativeTextureDescriptor;
        std::shared_ptr<const void> nativeTextureOwner;
    };
    ParticleBodyCallerLimits renderValidationLimits;
    renderValidationLimits.maxWork=completedWork+2*receipt->validationWork()-1;
    auto renderCaller=makeParticleBodyCallerLedger(renderValidationLimits,error);
    auto renderRequest=makeBoundedRequest(renderCaller);
    auto renderResult=run(renderRequest,.75,.75);
    check(error.empty() && renderResult.preparedBodyReplay,"render caller admits genuine production replay");
    if (renderResult.preparedBodyReplay) {
        CallerRenderLayer layer;
        check(!renderParticleSolids(renderResult,context->owner,7,96,96,layer,error)
            && error.find("completed terminal replay")!=std::string::npos && !layer.nativeTextureOwner
            && renderCaller->usage().work==renderValidationLimits.maxWork,
            "actual render path charges both full validations and refuses before scene allocation or native submission");
        check(!renderParticleSolids(renderResult,context->owner,7,96,96,layer,error)
            && renderCaller->usage().work==renderValidationLimits.maxWork && !layer.nativeTextureOwner,
            "repeated actual render validation has no fresh work allowance");
    }

    const auto sceneBytes=sizeof(ParticleBodyCallerPayload<HarmonicMIDI::grid::Visual3DScene>)+leaseBytes;
    ParticleBodyCallerLimits overlapLimits;
    // The incumbent and the replacement context each hold one envelope while
    // the replacement is refused for overlapping storage.
    overlapLimits.maxLogicalBytes=std::max(exactPeak,sizeof(ParticleBodyCallerLedger)+2*contextBytes+retainedBytes+sceneBytes);
    auto overlapCaller=makeParticleBodyCallerLedger(overlapLimits,error);
    auto incumbentRequest=makeBoundedRequest(overlapCaller);
    auto incumbent=run(incumbentRequest,.75,.75);
    auto incumbentScene=incumbent.preparedSolids ? particleSolidScene(incumbent,*incumbent.preparedSolids,7,error):nullptr;
    check(error.empty() && incumbentScene,"bounded incumbent receipt and scene escape together");
    if (incumbentScene) {
        auto escapedBodies=incumbent.preparedBodies;
        auto escapedSolids=incumbent.preparedSolids;
        auto replacement=makeBoundedRequest(overlapCaller);
        check(replacement.bodyReplayContext!=nullptr,"replacement context is admitted to the incumbent ledger");
        if (!replacement.bodyReplayContext) return;
        auto replacementBinding=std::make_shared<ParticleGeometryBinding>(*authored.geometryBinding);
        replacementBinding->bodySource=particleBodyFixtureAdmitted(*descriptor,error);
        replacement.geometryBinding=replacementBinding;
        replacement.geometryRevision=6;
        auto replacementContext=std::const_pointer_cast<ParticleBodyReplayContext>(replacement.bodyReplayContext);
        replacementContext->owner.planGeneration=6;
        auto replacedHistory=std::make_shared<ParticleHistorySource>(*history);
        replacedHistory->key.seekGeneration++; replacement.history=replacedHistory;
        const auto blocked=run(replacement,.75,.75);
        check(!error.empty() && !blocked.preparedBodyReplay && replacement.bodyReplayContext->caller==overlapCaller,
            "source, snapshot and plan replacement shares the incumbent ledger and refuses overlapping caller storage");
        const auto occupied=overlapCaller->usage().logicalBytes;
        incumbent={}; incumbentRequest={};
        check(overlapCaller->usage().logicalBytes==occupied-contextBytes && escapedBodies && escapedSolids,
            "escaped body and solid aliases keep all receipt storage charged after request retirement");
        escapedBodies.reset();
        check(overlapCaller->usage().logicalBytes==occupied-contextBytes,
            "one escaped pair owner cannot refund the other live payload");
        escapedSolids.reset();
        check(overlapCaller->usage().logicalBytes==sizeof(ParticleBodyCallerLedger)+contextBytes+sceneBytes,
            "final escaped solid release refunds receipt storage while escaped scene remains charged");
        const auto sceneBlocked=run(replacement,.75,.75);
        check(!error.empty() && !sceneBlocked.preparedBodyReplay,
            "escaped scene alone still charges overlap and prevents the candidate request");
        incumbentScene.reset();
        const auto recovered=run(replacement,.75,.75);
        check(error.empty() && recovered.preparedBodyReplay && recovered.preparedBodyReplay->sampler()->usage().attempts==91,
            "only final incumbent payload release permits the replacement on the same caller ledger");
    }
    ParticleBodyCallerLimits frameMetadataLimits;
    const auto frameMetadataBytes=ParticleSolidFrame::callerLogicalBytes()+leaseBytes;
    frameMetadataLimits.maxLogicalBytes=sizeof(ParticleBodyCallerLedger)+contextBytes+retainedBytes+sceneBytes+frameMetadataBytes-1;
    auto frameMetadataCaller=makeParticleBodyCallerLedger(frameMetadataLimits,error);
    auto frameMetadataRequest=makeBoundedRequest(frameMetadataCaller);
    auto frameMetadataResult=run(frameMetadataRequest,.75,.75);
    check(error.empty() && frameMetadataResult.preparedBodyReplay,"frame metadata caller admits genuine production replay");
    if (frameMetadataResult.preparedBodyReplay) {
        CallerRenderLayer layer;
        const auto frameStorageBefore=frameMetadataCaller->usage().logicalBytes;
        check(!renderParticleSolids(frameMetadataResult,context->owner,7,96,96,layer,error)
            && error.find("frame prospective caller admission")!=std::string::npos
            && !layer.nativeTextureOwner && frameMetadataCaller->usage().logicalBytes==frameStorageBefore,
            "actual renderer frame metadata one short refuses before native prepare/render and releases the candidate scene");
    }
    // The same wrapper used by scenes verifies refund ordering without forging
    // a native frame or reporting a native render PASS.
    bool destroyedWhileCharged=false;
    struct PayloadDestructionProbe {
        std::shared_ptr<ParticleBodyCallerLedger> ledger; std::size_t charged=0; bool* observed=nullptr;
        ~PayloadDestructionProbe() { if (observed) *observed=ledger->usage().logicalBytes>=charged; }
    };
    auto destructionCaller=makeParticleBodyCallerLedger({},error);
    auto destructionLease=destructionCaller->reserve(ParticleSolidFrame::callerLogicalBytes(),1);
    const auto destructionCharge=destructionCaller->usage().logicalBytes;
    auto destructionPayload=retainParticleBodyCallerPayload<PayloadDestructionProbe>(destructionLease);
    destructionPayload->ledger=destructionCaller; destructionPayload->charged=destructionCharge;
    destructionPayload->observed=&destroyedWhileCharged; destructionLease.reset(); destructionPayload.reset();
    check(destroyedWhileCharged && destructionCaller->usage().logicalBytes==sizeof(ParticleBodyCallerLedger),
        "logical frame-sized lease refunds only after payload destruction, with native evidence explicitly absent");
    const auto renderThreadCaller=particleBodyRenderThreadCaller(error);
    check(renderThreadCaller && renderThreadCaller==particleBodyRenderThreadCaller(error),
        "actual executing render thread retains one caller epoch across request replacements");

    auto box=authored; box.bodyShape=2; box.collisionMode=2;
    box.historyProjectSeconds=.75;
    const auto boxReplayOwned=heapReplayState(); auto& boxReplay=*boxReplayOwned;
    const bool boxPrepared=boxReplay.prepare(box,.75,1,nullptr,error,true);
    if (boxPrepared) while (!boxReplay.complete() && boxReplay.advanceOne(error)) {}
    check(boxPrepared && !boxReplay.complete() && error.find("half ratios")!=std::string::npos
        && boxReplay.geometrySampler() && boxReplay.geometrySampler()->usage().attempts>1
        && boxReplay.geometrySampler()->usage().visits>=boxReplay.geometrySampler()->usage().attempts
            *boxReplay.geometrySampler()->recipe().perAttemptVisits,
        "failed fixed-box sample keeps permanent attempted work charged without publishing completed state");
}

void checkBoxTriangleSurfaceContacts()
{
    using namespace videorender;
    ParticleGeometryFrame surface; surface.meshIdentity=4100;
    surface.meshVertexCount=3; surface.meshTriangleCount=1;
    surface.meshVertices[0]={0,0,0}; surface.meshVertices[1]={1,0,0}; surface.meshVertices[2]={0,1,0};
    surface.meshVertexIdentities[0]=11; surface.meshVertexIdentities[1]=22; surface.meshVertexIdentities[2]=33;
    surface.meshTriangles[0]={0,1,2};
    const std::array<SolidVector,3> triangle{surface.meshVertices[0],surface.meshVertices[1],surface.meshVertices[2]};
    const SolidVector half{.03125f,.03125f,.03125f};
    ParticleRigidBody body;
    struct Oracle { SolidVector center; double squared; };
    for (const auto& oracle:std::array<Oracle,3>{{{{.25f,.25f,.0625f},1.0/1024},
        {{.5f,-.0625f,.0625f},2.0/1024},{{-.0625f,-.0625f,.0625f},3.0/1024}}}) {
        body.x=oracle.center[0]; body.y=oracle.center[1]; body.z=oracle.center[2];
        solidtriangle::Query distance; SolidTriangleWorkBudget work;
        const double expected=std::sqrt(oracle.squared);
        check(solidtriangle::boxQuery(oracle.center,oracle.center,solidBox(body,half),triangle,0,distance,work)
            && distance.lower<=expected && distance.upper>=expected
            && std::abs(distance.lower-expected)<1.0e-12 && std::abs(distance.upper-expected)<1.0e-12,
            "box face, edge and vertex interval distances enclose independent double oracles");
    }
    // Every triangle edge has nonzero components and its normal has three
    // nonzero components, so all 96 candidate feature branches are charged.
    const std::array<SolidVector,3> featureTriangle{{{0,0,0},{1,2,3},{4,5,7}}};
    body={}; body.x=body.y=body.z=.25f;
    solidtriangle::Query featureQuery; SolidTriangleWorkBudget featureWork; featureWork.limit=96;
    check(solidtriangle::boxQuery(solidPosition(body),solidPosition(body),solidBox(body,half),
        featureTriangle,0,featureQuery,featureWork) && featureWork.used==96,
        "full box triangle feature query consumes the proven 96 charges");
    featureWork.used=0; featureWork.limit=95;
    check(!solidtriangle::boxQuery(solidPosition(body),solidPosition(body),solidBox(body,half),
        featureTriangle,0,featureQuery,featureWork) && featureWork.used==95,
        "full box triangle feature query refuses at 95 charges");
    auto p=solidBodyFixture(); p.bodyRadius=.0625f; p.colliderThickness=.03125f; p.collisionMode=0; p.restitution=1; p.friction=0;
    std::string error;
    const auto contact=[&](ParticleRigidBody& current,const ParticleRigidBody& previous,
        const ParticleGeometryFrame& frame,const ParticleGeometryFrame& old,SolidTriangleWorkBudget* work=nullptr) {
        error.clear(); return solidGeometryContact(current,previous,frame,old,1.0f/120,p,true,&half,work,&error);
    };
    // Independent unit-mass cube oracle: h=1/32, inverse inertia=1536,
    // face arm=+/-h and tangential inverse effective mass=1+1536*h*h=2.5.
    // Initial spin 4 adds +/-1/8 to unit tangential speed. The normal
    // impulse is 4, so friction .5 leaves the Coulomb cap nonbinding.
    auto frictionParams=p; frictionParams.bodyMass=1;
    for (float thickness:{0.0f,.03125f}) for (float approach:{-1.0f,1.0f}) {
        frictionParams.colliderThickness=thickness; frictionParams.friction=.5f;
        body={}; body.identity=7001; body.geometryIdentity=7002;
        body.x=body.y=.25f; body.z=approach*.5f; body.vx=1; body.vz=approach*2;
        body.angular={0,4,0};
        auto before=body; before.z=-approach*.5f;
        SolidTriangleContactImpulse receipt; error.clear();
        const bool resolved=solidGeometryContact(body,before,surface,surface,1.0f/120,frictionParams,true,
            &half,nullptr,&error,false,&receipt,surface.meshIdentity);
        const double tangentImpulse=(1+double(approach)/8)/2.5;
        const double expectedSpin=4-double(approach)*48*tangentImpulse;
        check(resolved && error.empty() && std::abs(double(body.vx)-(1-tangentImpulse))<2.0e-6
            && body.vy==0 && body.vz==-approach*2 && body.angular[0]==0 && body.angular[2]==0
            && std::abs(double(body.angular[1])-expectedSpin)<5.0e-5,
            "positive and zero collider thickness retain actual box face friction velocity and spin on both sides");
        check(resolved && receipt.valid && receipt.bodyIdentity==7001 && receipt.bodyGeometryIdentity==7002
            && receipt.meshIdentity==surface.meshIdentity && receipt.triangle==0
            && receipt.vertexIdentities==std::array<std::uint64_t,3>{11,22,33}
            && std::abs(receipt.point[0]-.25)<1.0e-12 && std::abs(receipt.point[1]-.25)<1.0e-12
            && receipt.point[2]==0 && std::abs(receipt.weights[0]-.5)<1.0e-12
            && std::abs(receipt.weights[1]-.25)<1.0e-12 && std::abs(receipt.weights[2]-.25)<1.0e-12
            && std::abs(receipt.impulse[0]+tangentImpulse)<2.0e-6 && receipt.impulse[1]==0
            && receipt.impulse[2]==-double(approach)*4 && receipt.time>0 && receipt.time<1,
            "box friction receipts retain the authored triangle point, weights, identities and actual velocity impulse");
        frictionParams.friction=0; body={}; body.x=body.y=.25f; body.z=approach*.5f;
        body.vx=1; body.vz=approach*2; before=body; before.z=-approach*.5f;
        check(solidGeometryContact(body,before,surface,surface,1.0f/120,frictionParams,true,&half,nullptr,&error)
            && body.vx==1 && body.vy==0 && body.vz==-approach*2
            && body.angular==SolidVector{},
            "centered zero-friction box face reflection retains zero spin with positive and zero thickness");
    }
    auto contactSpinParams=p; contactSpinParams.friction=.5f;
    body={}; body.x=body.y=.25f; body.z=.5f; body.vx=1; body.vz=2;
    auto spinBefore=body; spinBefore.z=-.5f;
    check(solidGeometryContact(body,spinBefore,surface,surface,1.0f/120,contactSpinParams,true,&half,nullptr,&error)
        && body.angular!=SolidVector{},"mesh friction can generate current spin from an initially zero-spin body");
    const auto contactPose=body.orientation;
    solidRotate(body,1.0f/120,0);
    check(body.orientation!=contactPose,"contact-generated spin still integrates after the zero-spin pose fix");
    for (const auto& xy:std::array<std::array<float,2>,3>{{{.25f,.25f},{.5f,-.046875f},{-.046875f,-.046875f}}}) {
        body={}; body.x=xy[0]; body.y=xy[1]; body.z=.5f; body.vz=32;
        auto previous=body; previous.z=-.5f;
        check(contact(body,previous,surface,surface) && body.z<0 && body.vz<32,
            "fixed box translation resolves triangle face, rounded edge and rounded vertex contacts");
    }
    p.colliderThickness=0;
    body={}; body.x=body.y=.25f; body.z=.5f; body.vz=32;
    auto previous=body; previous.z=-.5f;
    check(contact(body,previous,surface,surface) && std::abs(body.z+.03125f)<5.0e-6f && body.vz<0,
        "a fast fixed box cannot tunnel through a zero-thickness static triangle");
    body={}; body.x=body.y=.25f; body.z=-.5f; body.vz=-32; previous=body; previous.z=.5f;
    check(contact(body,previous,surface,surface) && body.z>0 && body.vz>0,
        "static box triangle contact normals oppose approach on either side");
    body={}; body.x=body.y=.9f; body.z=.5f; body.vz=32; previous=body; previous.z=-.5f;
    check(contact(body,previous,surface,surface) && body.z==.5f && body.vz==32,
        "a box crossing the empty triangle AABB corner remains free");
    body={}; body.x=body.y=.25f; body.z=.5f; body.vz=32;
    body.orientation={0,.3826834324f,0,.9238795325f}; previous=body; previous.z=-.5f;
    check(contact(body,previous,surface,surface)
        && std::abs(body.z+std::sqrt(.5f)*.0625f)<5.0e-6f && body.vz<32,
        "a fixed rotated box uses its oriented support radius against the triangle face");
    body={}; body.x=body.y=.25f; body.z=.5f; body.vz=32; previous=body; previous.z=-.5f;
    for (auto& component:previous.orientation) component=-component;
    check(contact(body,previous,surface,surface) && body.z<0,
        "equivalent opposite-sign quaternions preserve the fixed box mesh pose");
    const auto unchanged=[](const ParticleRigidBody& a,const ParticleRigidBody& b) {
        ParticleSolidState x,y; x.count=y.count=1; x.bodies[0]=a; y.bodies[0]=b;
        return sameSolidBodies(x,y);
    };
    body={}; body.x=body.y=.25f; body.z=.5f; body.vz=32; previous=body; previous.z=-.5f;
    const auto golden=body;
    previous.orientation={0,.3826834324f,0,.9238795325f};
    check(!contact(body,previous,surface,surface) && error.find("Rotating")!=std::string::npos && unchanged(body,golden),
        "a rotating box crossing a thin barrier refuses before changing caller body state");
    // Independent rising-plane oracle: the mesh rises from z=-1/8 to 0 under a
    // resting box with h=1/32, so the face meets the bottom at t=3/4. The
    // surface speed is (1/8)/dt; restitution 1 doubles it and the remaining
    // quarter interval carries the box to z=2h with the plane back at 0.
    {
        auto risingParams=p; risingParams.colliderThickness=0; risingParams.friction=0; risingParams.restitution=1;
        auto lowered=surface;
        for (std::uint32_t vertex=0;vertex<3;++vertex) lowered.meshVertices[vertex][2]=-.125f;
        for (const float dt:{1.0f/120,1.0f/64}) {
            body={}; body.identity=7101; body.geometryIdentity=7102; body.x=body.y=.25f;
            const auto start=body;
            SolidTriangleContactImpulse receipt; error.clear();
            const bool resolved=solidGeometryContact(body,start,surface,lowered,dt,risingParams,true,&half,nullptr,&error,
                false,&receipt,surface.meshIdentity);
            const double expectedSpeed=2*.125/double(dt);
            check(resolved && error.empty() && std::abs(double(body.vz)-expectedSpeed)<1.0e-3
                && std::abs(double(body.z)-.0625)<1.0e-4 && body.x==.25f && body.y==.25f
                && std::abs(body.vx)<1.0e-5f && std::abs(body.vy)<1.0e-5f
                && std::all_of(body.angular.begin(),body.angular.end(),[](float v) { return std::abs(v)<1.0e-3f; }),
                "a rising authored plane lifts a resting fixed box with the oracle surface velocity and remaining motion");
            check(resolved && receipt.valid && receipt.bodyIdentity==7101 && receipt.bodyGeometryIdentity==7102
                && receipt.meshIdentity==surface.meshIdentity && receipt.triangle==0
                && receipt.vertexIdentities==std::array<std::uint64_t,3>{11,22,33}
                && std::abs(receipt.time-.75)<1.0e-4
                && std::abs(receipt.impulse[2]-double(risingParams.bodyMass)*double(body.vz))<1.0e-6,
                "moving box receipts retain the impact time, authored identities and actual velocity impulse");
        }
        body={}; body.x=body.y=.25f; body.z=.0625f; auto resting=body;
        check(solidGeometryContact(body,resting,lowered,surface,1.0f/120,risingParams,true,&half,nullptr,&error)
            && unchanged(body,resting) && body.z==.0625f,
            "a falling authored plane departs from a resting box without a false contact");
        body={}; body.x=body.y=.9f; resting=body;
        check(solidGeometryContact(body,resting,surface,lowered,1.0f/120,risingParams,true,&half,nullptr,&error)
            && unchanged(body,resting),"a rising plane outside the box footprint leaves the box free");
        body={}; body.x=body.y=.25f; body.orientation={0,.3826834324f,0,.9238795325f}; resting=body;
        auto spunBefore=resting; spunBefore.orientation={0,0,0,1};
        check(!solidGeometryContact(body,spunBefore,surface,lowered,1.0f/120,risingParams,true,&half,nullptr,&error)
            && error.find("moving triangle surface")!=std::string::npos && unchanged(body,resting),
            "a rotating box against a moving triangle refuses before changing caller body state");
        auto collapsing=lowered; collapsing.meshVertices[1]={-1,0,-.125f};
        body={}; body.x=body.y=.25f; resting=body;
        check(!solidGeometryContact(body,resting,surface,collapsing,1.0f/120,risingParams,true,&half,nullptr,&error)
            && error.find("degenerate")!=std::string::npos && unchanged(body,resting),
            "a moving triangle that collapses inside its interval refuses transactionally for fixed boxes");
        SolidTriangleWorkBudget movingWork; movingWork.limit=50;
        check(!solidGeometryContact(body,resting,surface,lowered,1.0f/120,risingParams,true,&half,&movingWork,&error)
            && error.find("work units")!=std::string::npos && movingWork.used==movingWork.limit && unchanged(body,resting),
            "moving box work exhaustion remains permanently charged with unchanged caller state");
        // Correction passes check the current mesh pose without a motion sweep.
        body={}; body.x=body.y=.25f; body.z=.0625f; resting=body;
        check(solidGeometryContact(body,resting,surface,lowered,1.0f/120,risingParams,false,&half,nullptr,&error,true)
            && unchanged(body,resting),"correction passes against a moving mesh certify the current pose only");
        // Straddle limit: a flat square split along its (0,0)-(1,1) diagonal
        // rises under a box centered just above the diagonal. Both triangles
        // meet the face at the same time, the strict earliest-time comparison
        // keeps triangle 0, and its closest feature is the diagonal point
        // (.51,.51). The off-center arm adds spin, so the next interval against
        // the still-rising square is a clean rotating-box refusal.
        ParticleGeometryFrame square; square.meshIdentity=4200;
        square.meshVertexCount=4; square.meshTriangleCount=2;
        square.meshVertices[0]={0,0,0}; square.meshVertices[1]={1,0,0};
        square.meshVertices[2]={1,1,0}; square.meshVertices[3]={0,1,0};
        for (std::uint32_t vertex=0;vertex<4;++vertex) square.meshVertexIdentities[vertex]=41+vertex;
        square.meshTriangles[0]={0,1,2}; square.meshTriangles[1]={0,2,3};
        auto squareLow=square,squareHigh=square;
        for (std::uint32_t vertex=0;vertex<4;++vertex) {
            squareLow.meshVertices[vertex][2]=-.125f; squareHigh.meshVertices[vertex][2]=.125f;
        }
        body={}; body.identity=7201; body.geometryIdentity=7202; body.x=.5f; body.y=.52f;
        auto straddleStart=body;
        SolidTriangleContactImpulse straddleReceipt; error.clear();
        const bool straddled=solidGeometryContact(body,straddleStart,square,squareLow,1.0f/120,risingParams,true,&half,
            nullptr,&error,false,&straddleReceipt,square.meshIdentity);
        check(straddled && error.empty() && body.vz>0 && straddleReceipt.valid && straddleReceipt.triangle==0
            && body.angular[0]!=0 && body.angular[1]!=0 && std::abs(body.angular[2])<1.0e-3f,
            "a coplanar straddle landing resolves on the earlier triangle's diagonal feature and adds spin");
        auto turned=body;
        solidRotate(turned,1.0f/120,0);
        const auto turnedBefore=turned;
        check(straddled && !solidGeometryContact(turned,body,squareHigh,square,1.0f/120,risingParams,true,&half,nullptr,&error)
            && error.find("moving triangle surface")!=std::string::npos && unchanged(turned,turnedBefore),
            "the spun straddle box refuses the next rising interval before changing caller body state");
    }
    body=golden; previous=golden; previous.z=-.5f;
    auto invalid=surface; invalid.meshTriangles[0][2]=3;
    check(!contact(body,previous,invalid,invalid) && unchanged(body,golden),
        "invalid static box mesh indices cannot mutate the caller body");
    invalid=surface; invalid.meshVertexIdentities[1]=invalid.meshVertexIdentities[0];
    check(!contact(body,previous,invalid,invalid) && unchanged(body,golden),
        "invalid static box mesh identities cannot mutate the caller body");
    invalid=surface; invalid.meshVertices[2]=invalid.meshVertices[1];
    check(!contact(body,previous,invalid,invalid) && unchanged(body,golden),
        "degenerate static box triangles cannot mutate the caller body");
    invalid=surface; invalid.meshVertexIdentities[2]=44;
    check(!contact(body,previous,invalid,surface) && error.find("topology")!=std::string::npos && unchanged(body,golden),
        "static box triangle topology changes remain transactional refusals");
    SolidTriangleWorkBudget shortWork; shortWork.limit=9;
    check(!contact(body,previous,surface,surface,&shortWork) && error.find("work units")!=std::string::npos
        && shortWork.used==shortWork.limit && unchanged(body,golden),
        "box feature work exhaustion remains permanently charged with unchanged caller state");
    body={}; body.x=body.y=.25f; body.z=.5f; body.vz=1; previous=body; previous.z=.03125f;
    check(contact(body,previous,surface,surface) && body.z==.5f && body.vz==1,
        "a certified touching box can depart without a false contact");
    auto wallSurface=surface;
    wallSurface.meshVertices[0]={.03125f,0,0}; wallSurface.meshVertices[1]={.03125f,1,0};
    wallSurface.meshVertices[2]={.03125f,0,1};
    body={}; body.x=-.25f; body.y=body.z=.25f; previous=body; const auto beforeWall=body;
    p.collisionMode=2;
    check(!contact(body,previous,wallSurface,wallSurface) && error.find("boundary correction")!=std::string::npos
        && unchanged(body,beforeWall),"a box boundary correction cannot cross an authored triangle transactionally");

}

// Authenticated replay route: a dynamic box (the Timeline pyramid as a fixed
// box, half extents bodyRadius*ratio) spins about the normal of the static
// z=0.6 Grid plane. Parameter admission still refuses a dynamic box with a
// collider mesh in one payload, so the mesh binding comes from its own
// admitted payload. Two 1/120 s intervals: the first clears the plane, the
// second ends past the contact threshold, so the exact kernel must run.
void checkRotatingPlaneReplayRoute()
{
    using namespace videorender;
    using namespace videowire::geometry;
    std::string error;
    const auto descriptor=particleBodyTimelinePyramidDescriptor(error);
    const auto body=descriptor ? admitValue(*descriptor,particleBodyGeometryFixtureContract(),error):std::nullopt;
    PortContract meshContract; meshContract.carrier=CarrierKind::geometry3D;
    meshContract.maxVertices=256; meshContract.maxIndices=384;
    const auto grid=lowerGrid(4400,4400,1,2,2,0.5f,meshContract,error);
    const auto plane=grid ? admitValue(*grid,meshContract,error):std::nullopt;
    check(body.has_value() && plane.has_value(),"rotating replay body and plane sources admit");
    if (!body || !plane) return;
    const auto bodyPayload="<NodeParams motionMode=\"2\" historicalReplay=\"1\" simulationSpace=\"1\" count=\"1\" "
        "collisionMode=\"2\" bodyShape=\"2\" lifetime=\"2\" speed=\"0\" bodyMass=\"1\" bodyGeometryPlan=\""
        +encodeLoweredPlanText(lowerRuntimePlan(particleBodyGeometryFixtureContract(),*body))+"\"/>";
    const auto meshPayload="<NodeParams motionMode=\"2\" historicalReplay=\"1\" simulationSpace=\"1\" count=\"1\" "
        "colliderGeometryPlan=\""+encodeLoweredPlanText(lowerRuntimePlan(meshContract,*plane))+"\"/>";
    ParticleParams authored,meshSource;
    check(videowire::admitParticleParameters(bodyPayload,authored,error) && authored.geometryBinding
        && authored.geometryBinding->bodySource && videowire::admitParticleParameters(meshPayload,meshSource,error)
        && meshSource.geometryBinding && meshSource.geometryBinding->mesh,error.c_str());
    if (!authored.geometryBinding || !authored.geometryBinding->bodySource
        || !meshSource.geometryBinding || !meshSource.geometryBinding->mesh) return;
    auto binding=std::make_shared<ParticleGeometryBinding>(*authored.geometryBinding);
    binding->mesh=meshSource.geometryBinding->mesh;
    binding->scale=.5f; binding->centerX=.5f; binding->centerY=.5f; binding->centerZ=.6f;
    authored.geometryBinding=binding;
    // Contract source-fit radius 0x3d5db3d8 times the 1/sqrt(3) ratio gives
    // half exactly 1/32 (0.03125). The face at (.5,.42), rotated 7 degrees, stays inside one
    // Grid triangle for either diagonal. Contact starts at center z .56375.
    authored.count=authored.geometryCount=1;
    authored.geometryAnchors[0]={.5f,.42f,.5f,.42f}; authored.geometryDepths[0]={.545f,.545f};
    authored.geometryIdentities[0]=9007199254740001ull;
    authored.bodyRadius=rotatingexact::floatFromBits(0x3d5db3d8U); authored.colliderThickness=.005f;
    authored.gravity=authored.gravityX=authored.gravityZ=0;
    authored.force=authored.attraction=authored.drag=authored.friction=0;
    authored.impulseX=authored.impulseY=0; authored.impulseZ=1.2f;
    authored.orientationDegrees={0,0,7}; authored.angularVelocity={0,0,1};
    auto history=std::make_shared<ParticleHistorySource>(); history->score=std::make_shared<const arbitmod::Score>();
    history->key.projectGeneration=1; history->key.helperGeneration=2; history->key.deviceGeneration=3;
    authored.history=history; authored.historyClipId=4; authored.historyNodeId=7; authored.geometryRevision=5;
    const auto caller=makeParticleBodyCallerLedger({},error);
    auto context=makeParticleBodyReplayContext(caller,error);
    check(context!=nullptr,"rotating replay admits its caller context");
    if (!context) return;
    context->owner={1,2,3,4,5,videohelper::geometry::PlanUse::preview}; context->width=context->height=96;
    authored.bodyReplayContext=context;
    ParticleSolidState solids;
    const auto run=[&](ParticleParams request,double seconds) {
        request.historyProjectSeconds=seconds;
        request.preparedBodies.reset(); request.preparedSolids.reset(); request.preparedBodyReplay.reset();
        error.clear(); solids={};
        std::shared_ptr<const ParticleBodyReplayReceipt> receipt;
        replayParticleBodies(request,seconds,1,nullptr,&error,&solids,nullptr,&receipt);
        return receipt;
    };
    const std::string rotatingRefusal="Rotating box triangle sweeps are unsupported; the replay interval is refused";
    // An actual certified impact, not a refusal. Without contact the second
    // interval ends at center z .565 with vz 1.2. The kernel projects the
    // center outward to the contact threshold z ~.56375 (plane .6, half
    // 1/32, reach .005), reached at 15/16 (0.9375) of the interval. The
    // remaining 0.00104 s runs at the rebound speed restitution*1.2, which
    // moves the center about .00125*restitution further outward. The window
    // below .5635 therefore needs restitution above about 0.2; this test uses
    // the 0.7 default. A completed replay with vz<=0 and z in the window is
    // the kernel's contact result. The published pose stays a normal-axis spin.
    const auto impacted=[&](const std::shared_ptr<const ParticleBodyReplayReceipt>& receipt) {
        return error.empty() && receipt && solids.count>=1 && solids.bodies[0].vz<=0
            && solids.bodies[0].z<.5635f && solids.bodies[0].z>.55f
            && solids.bodies[0].orientation[0]==0 && solids.bodies[0].orientation[1]==0;
    };
    const auto clear=run(authored,1.0/120);
    check(error.empty() && clear && solids.count==1 && solids.bodies[0].vz==1.2f
        && solids.bodies[0].orientation[0]==0 && solids.bodies[0].orientation[1]==0
        && solids.bodies[0].angular==SolidVector{0,0,1},
        "an authenticated normal spin admits a width and certifies its clear first interval");
    const auto reached=run(authored,1.0/60);
    check(impacted(reached),"an authenticated normal spin reaching the plane gets the exact kernel's certified impact");
    auto tilted=authored; tilted.angularVelocity={0,1,0};
    run(tilted,1.0/60);
    check(error=="Rotating box triangle sweeps require a spin about one coordinate-plane mesh normal; the replay interval is refused",
        "an authenticated tilted spin consumes its issued capture and refuses the tilt");
    // R1 on the route that admits a width: body pairs refuse the whole step
    // before pass zero; the same pair without body pairs reaches the kernel.
    auto pair=authored; pair.count=pair.geometryCount=2;
    pair.geometryAnchors[1]={.3f,.3f,.3f,.3f}; pair.geometryDepths[1]={.3f,.3f};
    pair.geometryIdentities[1]=9007199254740002ull;
    const auto refusedPair=run(pair,1.0/60);
    check(error==rotatingRefusal && !refusedPair,"R1 refuses an authenticated two-body collisionMode 2 rotating step");
    pair.collisionMode=0;
    const auto freePair=run(pair,1.0/60);
    // The far body has no lateral velocity, so it stays at (.3,.3) and rises
    // freely from depth .3 by 1.2/60 to z .32.
    check(impacted(freePair) && solids.count==2 && solids.bodies[1].vz==1.2f
        && std::abs(solids.bodies[1].x-.3f)<1.0e-6f && std::abs(solids.bodies[1].y-.3f)<1.0e-6f
        && std::abs(solids.bodies[1].z-.32f)<1.0e-5f,
        "the same authenticated pair without body pairs gets the kernel's impact for the near body only");
}

// Only the replay can mint a rotating capture, so direct callers keep the
// released refusal. Caller outcomes use the test-only issuer, and the
// authenticated replay route is checkRotatingPlaneReplayRoute.
void checkRotatingPlaneSweepCaller()
{
    using namespace videorender;
    namespace exact=videorender::rotatingexact;
    static_assert(!std::is_default_constructible_v<SolidRotationCapture>
        && !std::is_copy_constructible_v<SolidRotationCapture> && !std::is_copy_assignable_v<SolidRotationCapture>
        && !std::is_default_constructible_v<SolidRotationView> && !std::is_copy_constructible_v<SolidRotationView>
        && !std::is_default_constructible_v<SolidRotationLedger>,
        "callers outside the replay cannot construct a rotating capture, view or ledger");
    // Contract W(n,V,T,P) and the largest admitted width. Twelve intervals of
    // one body against one triangle add 12*W(n) to the fixed-axis 1,783,732.
    static_assert(rotatingplane::ResponseCost::interval(22,3,1,0)==2362486
        && rotatingplane::ResponseCost::interval(35,3,1,0)==4624460
        && rotatingplane::ResponseCost::interval(39,3,1,0)==5452168
        && rotatingplane::ResponseCost::interval(87,3,1,0)==20219416
        && rotatingplane::Cost::grid(50)==65);
    SolidTriangleWorkBudget admission;
    check(admission.rotatingWidth(12,1,3,1,0)==50 && admission.rotatingWidth(4,1,3,1,0)==87
        && admission.rotatingWidth(40,1,3,1,0)==0 && admission.preflight(40,1,3,1,0,96),
        "rotating preflight picks the largest width that fits 100M and leaves fixed-axis admission alone");
    admission.limit=30133564;
    const bool smallest=admission.rotatingWidth(12,1,3,1,0)==22;
    admission.limit=30133563;
    check(smallest && admission.rotatingWidth(12,1,3,1,0)==0 && admission.preflight(12,1,3,1,0,96),
        "one checked unit below the narrowest rotating bound records the class unavailable");

    ParticleGeometryFrame surface; surface.meshIdentity=4700;
    surface.meshVertexCount=3; surface.meshTriangleCount=1;
    surface.meshVertices[0]={.1875f,exact::floatFromBits(0x2b800000U),0};
    surface.meshVertices[1]={.265625f,.1875f,0}; surface.meshVertices[2]={.1875f,.265625f,0};
    surface.meshVertexIdentities[0]=71; surface.meshVertexIdentities[1]=72; surface.meshVertexIdentities[2]=73;
    surface.meshTriangles[0]={0,1,2};
    const SolidVector half{.03125f,.03125f,.03125f};
    auto p=solidBodyFixture(); p.bodyMass=1; p.restitution=1; p.friction=.5f;
    p.colliderThickness=.03125f; p.collisionMode=0;
    // Endpoint corner: C0=(1/4,1/4,9/128), v=(0,0,-1), dt=1/128 and normal
    // spin 4. solidRotate reports its executed raw step (0,0,1/64,1).
    ParticleRigidBody previous; previous.identity=7101; previous.geometryIdentity=7102;
    previous.x=previous.y=.25f; previous.z=.0703125f; previous.vz=-1; previous.angular={0,0,4};
    auto body=previous; body.z=.0625f;
    std::array<float,4> raw {};
    check(solidRotate(body,1.0f/128,0,&raw) && raw==std::array<float,4>{0,0,.015625f,1}
        && body.orientation[0]==0 && body.orientation[1]==0
        && exact::floatBits(body.orientation[2])==0x3c7ff800U && exact::floatBits(body.orientation[3])==0x3f7ff800U,
        "solidRotate reports its pre-normalization step before the existing normalization");
    auto still=previous; still.angular={}; raw={9,9,9,9};
    check(!solidRotate(still,1.0f/128,0,&raw) && raw==std::array<float,4>{9,9,9,9}
        && still.orientation==previous.orientation,
        "a zero spin performs no rotating step and writes no raw capture");
    const auto workspaceOwned=std::make_unique<rotatingplane::Workspace>(); auto& workspace=*workspaceOwned;
    SolidTriangleWorkBudget work; SolidTriangleContactImpulse receipt; std::string error;
    const auto unchanged=[](const ParticleRigidBody& a,const ParticleRigidBody& b) {
        ParticleSolidState x,y; x.count=y.count=1; x.bodies[0]=a; y.bodies[0]=b;
        return sameSolidBodies(x,y);
    };
    // The refusal itself charges nothing, as released. The 8 units are the
    // unchanged validation of frame and old frame, 2(V+T) with V=3, T=1.
    auto golden=body;
    check(!solidGeometryContact(body,previous,surface,surface,1.0f/128,p,true,&half,&work,&error,false,
            &receipt,surface.meshIdentity) && error=="Rotating box triangle sweeps are unsupported; the replay interval is refused"
        && !receipt.valid && work.used==8 && unchanged(body,golden),
        "a direct rotating contact without an issued view keeps the released refusal after frame validation only");
    const SolidRotatingSweep forged{nullptr,&workspace,87};
    work={}; error.clear();
    check(!solidGeometryContact(body,previous,surface,surface,1.0f/128,p,true,&half,&work,&error,false,
            &receipt,surface.meshIdentity,&forged) && error=="Rotating box triangle sweeps are unsupported; the replay interval is refused"
        && !receipt.valid && work.used==8 && unchanged(body,golden),
        "a width and workspace without a replay-issued view confer no rotating authority");

    // Caller outcomes with a ledger-issued view, through the test-only issuer.
    // Width 39 (grid 40) holds every coefficient here, so the exact phases
    // charge the reviewed literals after the 8 validation units.
    // Heap-owned like replay states: the ledger alone is 9,272 bytes.
    const auto issuerOwned=std::make_unique<SolidRotationTestIssuer>(); auto& issuer=*issuerOwned;
    const SolidRotationView* view=nullptr;
    const auto scheduled=[&](const ParticleRigidBody& before,float z,float dt) {
        auto next=before; next.z=z;
        view=issuer.schedule(before,next,dt);
        return next;
    };
    const auto contact=[&](ParticleRigidBody& current,const ParticleRigidBody& before,float dt) {
        error.clear();
        const SolidRotatingSweep rotating{view,&workspace,39};
        const bool contacted=solidGeometryContact(current,before,surface,surface,dt,p,true,&half,&work,&error,false,
            &receipt,surface.meshIdentity,&rotating);
        issuer.retire(); view=nullptr;
        return contacted;
    };
    const auto doubleBits=[](double value) {
        std::uint64_t bits; std::memcpy(&bits,&value,8); return bits;
    };
    body=scheduled(previous,.0625f,1.0f/128);
    const auto pose=body.orientation;
    work={};
    check(contact(body,previous,1.0f/128) && error.empty() && work.used==8+5440816
        && exact::floatBits(body.x)==0x3e800000U && exact::floatBits(body.y)==0x3e800000U
        && exact::floatBits(body.z)==0x3d800001U
        && exact::floatBits(body.vx)==0x3e0e3978U && exact::floatBits(body.vy)==0x3e54db71U
        && exact::floatBits(body.vz)==0xbf000000U
        && exact::floatBits(body.angular[0])==0xc16c2aafU && exact::floatBits(body.angular[1])==0x418492d5U
        && exact::floatBits(body.angular[2])==0x3f9af8e6U && body.orientation==pose,
        "an issued normal spin certifies its exact endpoint corner contact and keeps the executed pose");
    check(receipt.valid && receipt.bodyIdentity==7101 && receipt.bodyGeometryIdentity==7102
        && receipt.meshIdentity==4700 && receipt.triangle==0
        && receipt.vertexIdentities==std::array<std::uint64_t,3>{71,72,73} && receipt.time==1
        && doubleBits(receipt.weights[0])==0x3fae0f100000712aULL
        && doubleBits(receipt.weights[1])==0x3fda698c9999999aULL
        && doubleBits(receipt.weights[2])==0x3fe0ea48b3332c21ULL
        && doubleBits(receipt.point[0])==0x3fcc207df8000000ULL
        && doubleBits(receipt.point[1])==0x3fcbe081f8000000ULL
        && receipt.point[2]==0 && receipt.impulse[0]!=0 && receipt.impulse[2]==.5,
        "rotating contact receipts retain the exact witness, identities and actual velocity impulse");
    // Strict clearance at both linear center endpoints is a certified no-hit
    // that runs no exact phase.
    auto high=previous; high.z=.5f;
    body=scheduled(high,.4375f,1.0f/128);
    golden=body; work={};
    check(contact(body,high,1.0f/128) && error.empty() && !receipt.valid && work.used==8 && unchanged(body,golden),
        "an issued normal spin with strict plane clearance at both endpoints is a certified no-hit");
    // Quarter-time corner: impulse 15 leaves an exact negative remainder.
    auto quarter=previous; quarter.z=.125f; quarter.vz=-30; p.friction=0;
    body=scheduled(quarter,-.125f,1.0f/120);
    golden=body; work={};
    check(!contact(body,quarter,1.0f/120)
        && error=="Rotating box triangle sweep cannot certify its exact plane contact; the replay interval is refused"
        && !receipt.valid && work.used==8+5440816 && unchanged(body,golden),
        "a rotating sweep whose remainder exceeds the certified clearance refuses with both phases charged");
    p.friction=.5f;
    auto tilted=previous; tilted.angular={4,0,0};
    body=scheduled(tilted,.0625f,1.0f/128);
    golden=body; work={};
    check(!contact(body,tilted,1.0f/128)
        && error=="Rotating box triangle sweeps require a spin about one coordinate-plane mesh normal; the replay interval is refused"
        && !receipt.valid && work.used==8 && unchanged(body,golden),
        "an issued tilted spin outside the normal-axis class keeps its transactional refusal");
    body=scheduled(previous,.0625f,1.0f/128);
    body.x=.2578125f; golden=body; work={};
    check(!contact(body,previous,1.0f/128)
        && error=="Rotating box triangle sweep lost its replay rotation provenance; the replay interval is refused"
        && !receipt.valid && work.used==8 && unchanged(body,golden),
        "a center moved after the issued solidRotate no longer matches its capture and refuses");
    body=scheduled(previous,.0625f,1.0f/128);
    golden=body; work={}; work.limit=8+4320677;
    check(!contact(body,previous,1.0f/128) && error=="Triangle contact replay exceeds 100M checked work units"
        && work.used==8+160696 && !receipt.valid && unchanged(body,golden),
        "a rotating sweep short of its rate phase keeps the certificate charge and the caller body");
    checkRotatingPlaneReplayRoute();
}

void checkTriangleSurfaceContacts()
{
    using namespace videorender;
    ParticleGeometryFrame surface; surface.meshIdentity=4100;
    surface.meshVertexCount=3; surface.meshTriangleCount=1;
    surface.meshVertices[0]={0,0,0}; surface.meshVertices[1]={1,0,0}; surface.meshVertices[2]={0,1,0};
    surface.meshVertexIdentities[0]=11; surface.meshVertexIdentities[1]=22; surface.meshVertexIdentities[2]=33;
    surface.meshTriangles[0]={0,1,2};
    const std::array<SolidVector,3> triangle{surface.meshVertices[0],surface.meshVertices[1],surface.meshVertices[2]};
    struct Oracle { SolidVector point,closest; double squared; };
    const std::array<Oracle,3> oracles{{{{.25f,.25f,.125f},{.25f,.25f,0},1.0/64},
        {{.5f,-.125f,.125f},{.5f,0,0},1.0/32},{{-.125f,-.125f,.125f},{0,0,0},3.0/64}}};
    for (const auto& oracle:oracles) {
        solidtriangle::Query query;
        const double distance=std::sqrt(oracle.squared);
        check(solidtriangle::query(oracle.point,oracle.point,triangle,triangle,0,query)
            && query.lower<=distance && query.upper>=distance
            && std::abs(query.lower-distance)<1.0e-12 && std::abs(query.upper-distance)<1.0e-12
            && std::abs(query.closest[0]-oracle.closest[0])<1.0e-12
            && std::abs(query.closest[1]-oracle.closest[1])<1.0e-12
            && std::abs(query.closest[2]-oracle.closest[2])<1.0e-12,
            "face, edge and vertex interval distances enclose independent double oracles");
    }
    std::string error;
    auto p=solidBodyFixture(); p.bodyRadius=.0625f; p.colliderThickness=0;
    p.collisionMode=0; p.restitution=1; p.friction=0;
    const auto contact=[&](ParticleRigidBody& body,const ParticleRigidBody& previous,
        const ParticleGeometryFrame& current,const ParticleGeometryFrame& old,SolidTriangleWorkBudget* work=nullptr)
    { error.clear(); return solidGeometryContact(body,previous,current,old,1.0f/120,p,true,nullptr,work,&error); };
    ParticleRigidBody body; body.x=body.y=.9f; body.z=.5f; body.vz=32;
    auto previous=body; previous.z=-.5f;
    check(contact(body,previous,surface,surface) && body.z==.5f && body.vz==32,
        "crossing an empty AABB corner does not collide with the triangle surface");
    for (const auto& xy:std::array<std::array<float,2>,3>{{{.25f,.25f},{.5f,-.03125f},{-.03125f,-.03125f}}}) {
        body={}; body.x=xy[0]; body.y=xy[1]; body.z=.5f; body.vz=32;
        previous=body; previous.z=-.5f;
        check(contact(body,previous,surface,surface) && body.z<0 && body.vz<0,
            "fast static sweeps resolve face, rounded edge and rounded vertex contacts");
    }
    body={}; body.x=body.y=.25f; body.z=-.5f; body.vz=-32;
    previous=body; previous.z=.5f;
    check(contact(body,previous,surface,surface) && body.z>0 && body.vz>0,
        "authored triangle shells collide on both sides");
    previous=body; previous.z=p.bodyRadius; body.z=.5f; body.vz=1;
    check(contact(body,previous,surface,surface) && body.z==.5f && body.vz==1,
        "a certified initial touching surface can separate without rewinding");
    previous.z=p.bodyRadius; body.z=-.5f; body.vz=-1;
    check(contact(body,previous,surface,surface) && body.z>0 && body.vz>0,
        "initial touching contact still resolves closing motion");
    auto old=surface,moving=surface;
    // Independent translating-plane oracle: gap .125, radius .0625, contact
    // at normalized time .5; surface speed 15, restitution 1 yields speed 30
    // and final body z .125 after the remaining half of a 120 Hz interval.
    for (int vertex=0;vertex<3;++vertex) old.meshVertices[vertex][2]=-.125f;
    body={}; body.x=body.y=.25f; previous=body;
    SolidTriangleWorkBudget movingWork;
    check(contact(body,previous,moving,old,&movingWork) && std::abs(body.z-.125f)<8.0e-6f
        && std::abs(body.vz-30)<1.0e-5f && body.vx==0 && body.vy==0,
        "a translating face matches independent collision-time, impulse and remaining-motion oracles");
    const auto moved=body;
    // The same normalized collision with a different event-split dt gives
    // speed 16 and the same final position; dt is not hardcoded to 120 Hz.
    body=previous; error.clear();
    check(solidGeometryContact(body,previous,moving,old,1.0f/64,p,true,nullptr,nullptr,&error)
        && std::abs(body.vz-16)<1.0e-5f && std::abs(body.z-.125f)<8.0e-6f,
        "moving surface velocity and remaining integration use the actual interval duration");
    const auto unchanged=previous;
    // Edge and vertex oracles use the circle equation in the fixed lateral
    // plane; their normal reflection gives 2*vSurface*nz*n at first contact.
    for (const auto& xy:std::array<std::array<float,2>,2>{{{.5f,-.03125f},{-.03125f,-.03125f}}}) {
        body={}; body.x=xy[0]; body.y=xy[1]; previous=body;
        const double lateral=double(xy[1])*xy[1]+(xy[0]<0 ? double(xy[0])*xy[0]:0);
        const double height=std::sqrt(double(p.bodyRadius)*p.bodyRadius-lateral);
        const double impact=(.125-height)/.125, nz=height/p.bodyRadius;
        const double speed=double(.125f)/(1.0f/120);
        const double expectedVz=2*speed*nz*nz;
        const double expectedZ=(1-impact)*(1.0f/120)*expectedVz;
        check(contact(body,previous,moving,old) && std::abs(body.vz-expectedVz)<.002
            && std::abs(body.z-expectedZ)<2.0e-5,
            "moving rounded edge and vertex responses match independent circle and reflection oracles");
    }
    // Deform within the face plane while translating in z. The contact at
    // (.25,.25) has weights (5/8,1/8,1/4) on the impact triangle, so its
    // tangential surface speed is 30 rather than the bounds-centre speed.
    moving.meshVertices[1][0]=3;
    body=unchanged; previous=body; p.friction=1;
    check(contact(body,previous,moving,old) && std::abs(body.vx-30.0f/3.5f)<.002f
        && std::abs(body.angular[1])>0 && std::abs(body.vz-30)<1.0e-5f,
        "deforming face response uses contact barycentrics for surface friction and sphere torque");
    p.friction=0; moving=surface;
    auto momentumParams=p; momentumParams.bodyMass=2.5f;
    body=unchanged; body.identity=77; body.geometryIdentity=88; body.vz=-32; previous=body;
    SolidTriangleContactImpulse observed;
    check(solidGeometryContact(body,previous,moving,old,1.0f/120,momentumParams,true,nullptr,nullptr,
        &error,false,&observed,surface.meshIdentity) && observed.valid && body.vz==32
        && observed.bodyIdentity==77 && observed.bodyGeometryIdentity==88
        && observed.meshIdentity==surface.meshIdentity && observed.triangle==0
        && observed.vertexIdentities==std::array<std::uint64_t,3>{11,22,33}
        && std::abs(observed.weights[0]+observed.weights[1]+observed.weights[2]-1)<1e-12
        && observed.impulse[2]==160,
        "contact observation retains exact identities and actual clamped mass times delta velocity");
    body=unchanged; previous=body; body.x=previous.x=4;
    check(contact(body,previous,moving,old) && body.x==4 && body.vz==0,
        "linear moving triangles can admit a certified full-interval miss");
    body=unchanged; previous=body; body.x=previous.x=.9f; body.y=previous.y=.9f;
    check(contact(body,previous,moving,old) && body.z==0 && body.vz==0,
        "a moving AABB corner miss retains the exact body state");
    // A 120-unit/s surface cannot be followed by the existing 32-unit/s
    // body bound. Refuse its remaining crossing after the provisional impulse.
    for (int vertex=0;vertex<3;++vertex) { old.meshVertices[vertex][2]=-.5f; moving.meshVertices[vertex][2]=.5f; }
    body=unchanged; previous=body;
    check(!contact(body,previous,moving,old) && !error.empty()
        && solidPosition(body)==solidPosition(unchanged) && solidVelocity(body)==solidVelocity(unchanged),
        "a clamp-limited moving response refuses without publishing a provisional impulse");
    old=surface; moving=surface;
    moving.meshVertices[1]={-1,0,0}; moving.meshVertices[2]={0,-1,0};
    body=unchanged; previous=body;
    check(!contact(body,previous,moving,old) && error.find("degenerate")!=std::string::npos
        && solidPosition(body)==solidPosition(unchanged),
        "nondegenerate endpoint triangles that collapse mid-interval refuse transactionally");
    old=surface; moving=surface;
    for (int vertex=0;vertex<3;++vertex) old.meshVertices[vertex][2]=-.125f;
    auto multiple=moving,previousMultiple=old;
    multiple.meshTriangleCount=previousMultiple.meshTriangleCount=2;
    multiple.meshTriangles[1]=previousMultiple.meshTriangles[1]={0,1,2};
    body=unchanged; previous=body;
    check(!contact(body,previous,multiple,previousMultiple) && error.find("another surface")!=std::string::npos
        && solidPosition(body)==solidPosition(unchanged) && solidVelocity(body)==solidVelocity(unchanged),
        "simultaneous moving multiface contact refuses rather than choosing an arbitrary impulse");
    // A distinct face is reached only by the reflected suffix, although the
    // original stationary trajectory misses it. This checks every suffix face.
    multiple=moving; previousMultiple=old;
    multiple.meshVertexCount=previousMultiple.meshVertexCount=6;
    multiple.meshTriangleCount=previousMultiple.meshTriangleCount=2;
    multiple.meshTriangles[1]=previousMultiple.meshTriangles[1]={3,4,5};
    for (int vertex=0;vertex<3;++vertex) {
        multiple.meshVertexIdentities[vertex+3]=previousMultiple.meshVertexIdentities[vertex+3]=44+vertex;
        multiple.meshVertices[vertex+3]=previousMultiple.meshVertices[vertex+3]=surface.meshVertices[vertex];
        multiple.meshVertices[vertex+3][2]=previousMultiple.meshVertices[vertex+3][2]=.15625f;
    }
    body=unchanged; previous=body;
    check(!contact(body,previous,multiple,previousMultiple) && !error.empty()
        && solidPosition(body)==solidPosition(unchanged) && solidVelocity(body)==solidVelocity(unchanged),
        "a reflected remaining path into a distinct face refuses the entire contact transaction");
    SolidTriangleWorkBudget suffixExhausted; suffixExhausted.limit=movingWork.used-1;
    body=unchanged; previous=body;
    check(!contact(body,previous,moving,old,&suffixExhausted) && error.find("work units")!=std::string::npos
        && solidPosition(body)==solidPosition(unchanged) && solidVelocity(body)==solidVelocity(unchanged),
        "exhausting work during remaining-response checks preserves the caller body");
    check(moved.z>p.bodyRadius && movingWork.used<=SolidTriangleWorkBudget::maximum,
        "a certified moving response clears the final shell within the existing hard work limit");
    auto invalid=surface; invalid.meshVertices[2]={2,0,0};
    body=unchanged; previous=body;
    check(!contact(body,previous,invalid,invalid) && error.find("Degenerate")!=std::string::npos
        && solidPosition(body)==solidPosition(unchanged),"degenerate triangles refuse without mutating a body");
    invalid=surface; invalid.meshTriangles[0][2]=3;
    check(!contact(body,previous,invalid,invalid) && error.find("index")!=std::string::npos,
        "out-of-range triangle indices refuse before access");
    invalid=surface; invalid.meshTriangles[0]={1,2,0};
    check(!contact(body,previous,invalid,surface) && error.find("topology")!=std::string::npos,
        "ordered triangle identity changes refuse despite identical geometric coverage");
    invalid=surface; invalid.meshVertexIdentities[2]=22;
    check(!contact(body,previous,invalid,invalid) && error.find("identities")!=std::string::npos,
        "duplicate stable vertex identities refuse");
    auto wide=surface;
    wide.meshVertexIdentities[0]=1+(std::uint64_t{1}<<63);
    wide.meshVertexIdentities[1]=1;
    wide.meshVertexIdentities[2]=1+(std::uint64_t{1}<<32);
    const auto sourceOrder=wide.meshVertexIdentities;
    SolidTriangleWorkBudget identityWork;
    error.clear(); error.reserve(256);
    const auto beforeIdentityAllocations=allocations.load(std::memory_order_relaxed);
    check(validateSolidTriangleFrame(wide,identityWork,error)
        && allocations.load(std::memory_order_relaxed)==beforeIdentityAllocations
        && wide.meshVertexIdentities==sourceOrder && identityWork.used==4,
        "full-width identities validate without heap allocation or changing source order/work units");
    auto maximumIdentities=surface; maximumIdentities.meshVertexCount=256;
    for (std::uint32_t vertex=0;vertex<256;++vertex)
        maximumIdentities.meshVertexIdentities[vertex]=(std::uint64_t{1}<<63)+256-vertex;
    const auto maximumOrder=maximumIdentities.meshVertexIdentities;
    SolidTriangleWorkBudget maximumIdentityWork;
    const auto beforeMaximumAllocations=allocations.load(std::memory_order_relaxed);
    check(validateSolidTriangleFrame(maximumIdentities,maximumIdentityWork,error)
        && allocations.load(std::memory_order_relaxed)==beforeMaximumAllocations
        && maximumIdentities.meshVertexIdentities==maximumOrder && maximumIdentityWork.used==257,
        "maximum descending full-width identities retain bounded stack validation and exact frame work");
    wide.meshVertexIdentities[2]=wide.meshVertexIdentities[0];
    check(!contact(body,previous,wide,wide) && error.find("identities")!=std::string::npos,
        "duplicate identities above the signed and float ranges still refuse");
    wide=surface; wide.meshVertexIdentities[1]=0;
    check(!contact(body,previous,wide,wide) && error.find("identities")!=std::string::npos,
        "zero vertex identities retain their explicit refusal");
    invalid=surface; invalid.meshVertices[0][2]=std::numeric_limits<float>::denorm_min();
    check(!contact(body,previous,invalid,invalid),"subnormal native inputs refuse the triangle arithmetic contract");
    SolidTriangleWorkBudget exhausted; exhausted.limit=0;
    check(!contact(body,previous,surface,surface,&exhausted) && error.find("work units")!=std::string::npos,
        "the hard triangle work limit refuses before a partial response");
    SolidTriangleWorkBudget admitted;
    check(admitted.preflight(120,1,3,1) && !admitted.preflight(1712,63,256,128),
        "whole replay preflight includes all four passes and every bounded sweep branch");
    // Four passes: two validations, six sweep branches, a motion guard and
    // impact/final queries per triangle. Include initial and interval samples.
    admitted.limit=121*4+120*4*(2*4+6*64+3);
    check(admitted.preflight(120,1,3,1,0),"moving response worst-case work admits at its exact bound");
    --admitted.limit;
    check(!admitted.preflight(120,1,3,1,0),"moving response worst-case work refuses one unit below its bound");
    admitted.limit=121*4+120*4*(2*4+(6*64+3)*96);
    check(admitted.preflight(120,1,3,1,0,96),"box96 replay admits at its exact weighted work bound");
    --admitted.limit;
    check(!admitted.preflight(120,1,3,1,0,96),"box96 replay refuses one unit below its weighted work bound");
    admitted.limit=SolidTriangleWorkBudget::maximum;
    check(admitted.preflight(120,1,3,1,64,1) && !admitted.preflight(120,1,3,1,64,96),
        "box feature weighting refuses a workload that unit sphere queries admit");
    check(!admitted.preflight(120,1,3,1,0,0) && !admitted.preflight(120,1,3,1,0,97)
        && !admitted.preflight(120,1,3,1,0,std::numeric_limits<std::uint64_t>::max())
        && !admitted.preflight(std::numeric_limits<std::uint64_t>::max(),1,3,1,0,96)
        && !admitted.preflight(120,std::numeric_limits<std::uint64_t>::max(),3,1,0,96)
        && !admitted.preflight(120,1,std::numeric_limits<std::uint64_t>::max(),1,0,96)
        && !admitted.preflight(120,1,3,std::numeric_limits<std::uint64_t>::max(),0,96)
        && !admitted.preflight(120,1,3,1,std::numeric_limits<std::uint64_t>::max(),96),
        "invalid costs and overflow-sized preflight inputs refuse without wrapping");
    SolidTriangleWorkBudget tangentWork; solidtriangle::Query tangent; bool hit=false;
    error.clear();
    check(!solidtriangle::sweep({-1,-p.bodyRadius,0},{1,-p.bodyRadius,0},triangle,triangle,
        p.bodyRadius,tangentWork,tangent,hit,error) && error.find("64")!=std::string::npos,
        "grazing motion without a finite convergence certificate refuses at the exact query bound");
    auto wallSurface=surface;
    wallSurface.meshVertices[0]={.03125f,0,0}; wallSurface.meshVertices[1]={.03125f,1,0};
    wallSurface.meshVertices[2]={.03125f,0,1};
    body={}; body.x=-.25f; body.y=body.z=.25f; previous=body;
    p.collisionMode=1;
    check(!contact(body,previous,wallSurface,wallSurface) && error.find("boundary correction")!=std::string::npos
        && body.x==-.25f,"a boundary correction that crosses a surface refuses transactionally");
    p.collisionMode=0;
    if (solidtriangle::arithmeticSupported()) {
        const int rounding=std::fegetround(); std::fesetround(FE_UPWARD);
        SolidTriangleWorkBudget work;
        check(!validateSolidTriangleFrame(surface,work,error),"non-nearest rounding refuses certified triangle contacts");
        std::fesetround(rounding);
    }

    using namespace videowire::geometry;
    PortContract contract; contract.carrier=CarrierKind::geometry3D; contract.maxVertices=256; contract.maxIndices=384;
    const auto grid=lowerGrid(4200,4200,1,2,2,2,contract,error);
    check(grid.has_value(),"triangle replay fixture uses existing grid authority"); if (!grid) return;
    auto binding=std::make_shared<ParticleGeometryBinding>(); binding->scale=.25f;
    binding->mesh=std::make_shared<const ValueDescriptor>(*grid);
    p=solidBodyFixture(); p.geometryBinding=binding; p.geometryRevision=1;
    p.count=p.geometryCount=1; p.collisionMode=0; p.impulseZ=1; p.attraction=0;
    p.geometryAnchors[0]={.4f,.45f,.4f,.45f}; p.geometryDepths[0]={.2f,.2f};
    ParticleGeometryFrame sampled; error.clear();
    check(sampleParticleGeometry(p,0,sampled,error) && sampled.meshVertexCount==4 && sampled.meshTriangleCount==2
        && sampled.meshVertices[0]==SolidVector{.25f,.25f,.5f}
        && sampled.meshVertexIdentities[0]==1 && sampled.meshTriangles[0]==std::array<std::uint32_t,3>{0,1,2},
        "sampler retains evaluated world vertices and exact stable indexed triangle order");
    error.clear(); const auto direct=sampleSolidBodyFixture(p,.5,error);
    check(error.empty() && direct.count==1 && direct.bodies[0].z<.5f && direct.bodies[0].vz<0,
        "static authored surfaces alter historical solid replay");
    auto preview=p,exported=p;
    for (int frame=0;frame<=30;++frame) sampleSolidBodyFixture(preview,double(frame)/60,error);
    sampleSolidBodyFixture(preview,1.75,error); sampleSolidBodyFixture(preview,.125,error);
    check(error.empty() && sameSolidBodies(direct,sampleSolidBodyFixture(preview,.5,error))
        && sameSolidBodies(direct,sampleSolidBodyFixture(exported,.5,error))
        && sameSolidBodies(direct,sampleSolidBodyFixture(p,2.5,error)),
        "triangle direct, sequential, backward, loop and same-time preview/export replay agree");
    p.resetTime=.25f;
    check(sameSolidBodies(direct,sampleSolidBodyFixture(p,.75,error))
        && sampleSolidBodyFixture(p,.1,error).count==0,"triangle replay retains reset clock semantics");
    auto duplicate=*grid; std::get<GeometryData>(duplicate.data).indices={0,0,0};
    // Runtime lowering must reject the disagreement with its source plan.
    auto rejectedBinding=std::make_shared<ParticleGeometryBinding>(*binding);
    rejectedBinding->mesh=std::make_shared<const ValueDescriptor>(duplicate);
    p.resetTime=0; p.geometryBinding=rejectedBinding; p.preparedBodies.reset(); p.preparedSolids.reset(); error.clear();
    ParticleSolidState failed=direct;
    const auto failedPoints=replayParticleBodies(p,.5,1,nullptr,&error,&failed);
    check(!error.empty() && failed.count==0 && failedPoints[0][2]==0,
        "failed triangle sampling clears complete solid and Points publication");
    // Author the moving shell through existing typed fields and history. Its
    // two-second phase matches the replay lifetime, while keeping project time
    // authoritative through seek, loop and reset.
    error.clear();
    PortContract vector; vector.carrier=CarrierKind::field; vector.fieldDomain=Domain::vertex;
    vector.fieldValueType=ValueType::vector; vector.fieldInterpolation=Interpolation::constant; vector.maxFieldElements=4;
    auto scalar=vector; scalar.fieldValueType=ValueType::floatValue;
    auto position=lowerElementField(*grid,4251,OperationCode::positionField,vector,error);
    FieldOperationSettings settings; settings.domain=Domain::vertex; settings.valueType=ValueType::floatValue; settings.mode=3;
    auto coordinate=position ? lowerConstructedField(4252,OperationCode::fieldVectorMeasure,settings,
        &*position,nullptr,nullptr,scalar,error):std::nullopt;
    settings.mode=1; settings.values={2,0,0,0,.5,0,0,0};
    auto time=coordinate ? lowerConstructedField(4253,OperationCode::fieldTimeline,settings,
        &*coordinate,nullptr,nullptr,scalar,error):std::nullopt;
    settings.mode=1; settings.valueType=ValueType::vector; settings.values={0,0,1,0,0,0,0,0};
    auto offset=time ? lowerConstructedField(4254,OperationCode::fieldComposeVector,settings,
        &*time,nullptr,nullptr,vector,error):std::nullopt;
    auto movingGrid=offset ? lowerVectorDisplacement(*grid,*offset,4255,1,contract,error):std::nullopt;
    check(movingGrid.has_value(),"moving shell replay uses the existing typed timeline and displacement authority");
    if (movingGrid) {
        auto movingBinding=std::make_shared<ParticleGeometryBinding>(*binding); movingBinding->centerZ=.25f;
        movingBinding->mesh=std::make_shared<const ValueDescriptor>(*movingGrid);
        auto movingReplay=solidBodyFixture(); movingReplay.geometryBinding=movingBinding; movingReplay.geometryRevision=1;
        movingReplay.count=movingReplay.geometryCount=1; movingReplay.collisionMode=0;
        movingReplay.bodyRadius=.015f; movingReplay.colliderThickness=0; movingReplay.restitution=1;
        movingReplay.impulseZ=0; movingReplay.angularVelocity={};
        movingReplay.geometryAnchors[0]={.4f,.45f,.4f,.45f}; movingReplay.geometryDepths[0]={.28f,.28f};
        error.clear(); const auto movingDirect=sampleSolidBodyFixture(movingReplay,.5,error);
        check(error.empty() && movingDirect.count==1 && std::abs(movingDirect.bodies[0].z-.3125f)<.0001f
            && std::abs(movingDirect.bodies[0].vz-.125f)<.0001f
            && movingDirect.bodies[0].geometryIdentity==movingReplay.geometryIdentities[0],
            "project-history moving face replay matches an independent constant-speed plane response");
        auto movingPreview=movingReplay,movingExport=movingReplay;
        for (int frame=0;frame<=30;++frame) sampleSolidBodyFixture(movingPreview,double(frame)/60,error);
        sampleSolidBodyFixture(movingPreview,1.75,error); sampleSolidBodyFixture(movingPreview,.125,error);
        check(error.empty() && sameSolidBodies(movingDirect,sampleSolidBodyFixture(movingPreview,.5,error))
            && sameSolidBodies(movingDirect,sampleSolidBodyFixture(movingExport,.5,error))
            && sameSolidBodies(movingDirect,sampleSolidBodyFixture(movingReplay,2.5,error)),
            "moving shell sequential, backward, direct, phase loop and same-time preview/export replay agree");
        movingReplay.resetTime=.25f;
        const auto resetDirect=sampleSolidBodyFixture(movingReplay,.75,error);
        sampleSolidBodyFixture(movingReplay,1.25,error); sampleSolidBodyFixture(movingReplay,.3,error);
        check(error.empty() && resetDirect.count==1
            && sameSolidBodies(resetDirect,sampleSolidBodyFixture(movingReplay,.75,error))
            && sampleSolidBodyFixture(movingReplay,.1,error).count==0,
            "moving shell reset reconstructs on authored project time and suppresses pre-reset bodies");
        auto stationaryBinding=std::make_shared<ParticleGeometryBinding>(*movingBinding);
        stationaryBinding->mesh=binding->mesh; movingExport.geometryBinding=stationaryBinding;
        const auto noMotion=sampleSolidBodyFixture(movingExport,.5,error);
        check(error.empty() && noMotion.count==1 && noMotion.bodies[0].z==.28f && noMotion.bodies[0].vz==0,
            "disabling authored shell motion removes the replay impulse");
        movingExport.geometryBinding=movingBinding;
        movingExport.geometryAnchors[0]={.9f,.9f,.9f,.9f};
        const auto movingMiss=sampleSolidBodyFixture(movingExport,.5,error);
        check(error.empty() && movingMiss.count==1 && movingMiss.bodies[0].z==.28f && movingMiss.bodies[0].vz==0,
            "a moving-shell historical near miss preserves the body trajectory");
        // The same rising shell under a fixed unit-cube box proxy. With
        // bodyRadius .015*sqrt(3) the box half extent is .015, the sphere
        // radius above, so the independent plane oracle is unchanged.
        auto unitCube=lowerMeshGenerator(4256,4256,1,OperationCode::cube,{1,1,1,0},contract,error);
        check(unitCube.has_value(),"moving box replay fixture lowers a unit cube body");
        if (unitCube) {
            auto boxBinding=std::make_shared<ParticleGeometryBinding>(*movingBinding);
            boxBinding->body=std::make_shared<const ValueDescriptor>(*unitCube);
            auto boxReplay=movingReplay; boxReplay.resetTime=0; boxReplay.geometryBinding=boxBinding;
            boxReplay.bodyShape=2; boxReplay.bodyRadius=.015f*std::sqrt(3.0f);
            boxReplay.geometryAnchors[0]={.4f,.45f,.4f,.45f};
            SolidVector boxHalf{};
            check(solidBodyBoxHalfExtents(boxReplay,boxHalf) && std::abs(boxHalf[0]-.015f)<1.0e-6f
                && boxHalf[0]==boxHalf[1] && boxHalf[1]==boxHalf[2],
                "the unit cube box proxy matches the sphere oracle radius");
            error.clear(); const auto boxDirect=sampleSolidBodyFixture(boxReplay,.5,error);
            check(error.empty() && boxDirect.count==1 && std::abs(boxDirect.bodies[0].z-.3125f)<.0001f
                && std::abs(boxDirect.bodies[0].vz-.125f)<.0001f
                && std::all_of(boxDirect.bodies[0].angular.begin(),boxDirect.bodies[0].angular.end(),
                    [](float v) { return std::abs(v)<1.0e-3f; }),
                "project-history moving face lifts a fixed box with the independent constant-speed plane response");
            auto boxPreview=boxReplay,boxExport=boxReplay;
            for (int frame=0;frame<=30;++frame) sampleSolidBodyFixture(boxPreview,double(frame)/60,error);
            sampleSolidBodyFixture(boxPreview,1.75,error); sampleSolidBodyFixture(boxPreview,.125,error);
            check(error.empty() && sameSolidBodies(boxDirect,sampleSolidBodyFixture(boxPreview,.5,error))
                && sameSolidBodies(boxDirect,sampleSolidBodyFixture(boxExport,.5,error))
                && sameSolidBodies(boxDirect,sampleSolidBodyFixture(boxReplay,2.5,error)),
                "moving box sequential, backward, direct, phase loop and same-time preview/export replay agree");
            boxReplay.resetTime=.25f;
            const auto boxReset=sampleSolidBodyFixture(boxReplay,.75,error);
            sampleSolidBodyFixture(boxReplay,1.25,error); sampleSolidBodyFixture(boxReplay,.3,error);
            check(error.empty() && boxReset.count==1
                && sameSolidBodies(boxReset,sampleSolidBodyFixture(boxReplay,.75,error))
                && sampleSolidBodyFixture(boxReplay,.1,error).count==0,
                "moving box reset reconstructs on authored project time and suppresses pre-reset bodies");
        }
    }
    error.clear(); const auto animated=particleGeometryFixture(true,error);
    check(animated.has_value(),"moving multiface fixture uses the existing history evaluator");
    if (animated) {
        auto movingReplay=*animated; movingReplay.simulationSpace=1;
        movingReplay.geometryAnchors[0]={.5f,.5f,.5f,.5f}; movingReplay.geometryDepths[0]={.5f,.5f};
        movingReplay.historyProjectSeconds=.5; failed=direct; error.clear();
        const auto rejected=replayParticleBodies(movingReplay,.5,1,nullptr,&error,&failed);
        check(!error.empty() && failed.count==0 && rejected[0][2]==0,
            "uncertified moving multiface contact clears the entire historical solid and Points frame");
    }
}

// Two canonical notes joined by one signed link, admitted by FrameProducer at
// beat zero. Shared by the coupled-spring checks and the fixed-step oracle.
auto canonicalLinkFrame(bool reverse, int numerator)
{
    auto score = std::make_shared<arbitmod::Score>();
    score->scoreRevision = 1;
    arbitmod::Note a; a.id = -101; a.midiNote = 60; a.velocity = 100;
    a.freqHz = 261.625565f; a.startBeat = 0; a.lengthBeats = 4;
    arbitmod::Note b = a; b.id = 202; b.midiNote = 72; b.freqHz *= 2;
    score->notes = reverse ? std::vector<arbitmod::Note>{b, a} : std::vector<arbitmod::Note>{a, b};
    score->links.push_back({-301, -101, 202, numerator, 2, 0});
    canonicalblockc::FrameKey key;
    key.projectGeneration = key.sourceGeneration = key.helperGeneration = key.backendGeneration = 1;
    key.deviceGeneration = key.scoreGeneration = key.beatMapGeneration = key.fpsGeneration = 1;
    key.loopGeneration = key.seekGeneration = 1; key.fps = 30;
    canonicalblockc::FrameProducer producer;
    return producer.evaluate(key, score, 0);
}

void checkCoupledBodyReplay()
{
    videorender::ParticleParams p;
    p.motionMode = 2; p.count = 2; p.lifetime = 4; p.force = 0;
    p.geometryCount = 2; p.collisionMode = 2; p.bodyRadius = 0.06f;
    p.geometryAnchors[0] = {0.48f, 0.5f, 0.48f, 0.5f};
    p.geometryAnchors[1] = {0.52f, 0.54f, 0.52f, 0.54f};
    const auto circle = videorender::replayParticleBodies(p, 0.5, 1, nullptr);
    check(std::hypot(circle[1][0] - circle[0][0], circle[1][1] - circle[0][1]) >= 0.1199f,
          "coupled circle bodies resolve pair overlap");
    check(circle == videorender::replayParticleBodies(p, 4.5, 1, nullptr),
          "coupled repeat returns the exact earlier state at a loop");
    p.bodyShape = 1;
    const auto box = videorender::replayParticleBodies(p, 0.5, 1, nullptr);
    check(box != circle && (std::abs(box[1][0] - box[0][0]) >= 0.1199f
                        || std::abs(box[1][1] - box[0][1]) >= 0.1199f),
          "axis-aligned box contacts use their own separation shape");
    p.resetTime = 0.25f;
    check(videorender::replayParticleBodies(p, 0.75, 1, nullptr) == box
          && videorender::replayParticleBodies(p, 0.2, 1, nullptr)[0][2] == 0,
          "reset time shifts deterministic replay and suppresses earlier bodies");
    p.resetTime = 0; p.resetMode = 1; p.impulseX = 0.4f;
    check(videorender::replayParticleBodies(p, 4, 1, nullptr)
          == videorender::replayParticleBodies(p, 10000, 1, nullptr),
          "hold-after-lifetime is bounded and independent of a distant seek");
    p.collisionMode = 0;
    const auto light = videorender::replayParticleBodies(p, 0.5, 1, nullptr);
    p.bodyMass = 2;
    const auto heavy = videorender::replayParticleBodies(p, 0.5, 1, nullptr);
    check(light[0][0] > heavy[0][0], "mass changes the response to an authored impulse");
    p.impulseX = 0; p.attraction = 3; p.geometryAnchors[0][2] = 0.8f;
    const auto attracted = videorender::replayParticleBodies(p, 0.5, 1, nullptr);
    check(attracted[0][0] > p.geometryAnchors[0][0], "typed point attractors excite coupled bodies");

    const auto score = canonicalLinkFrame(false, 3), reordered = canonicalLinkFrame(true, 3), ratio = canonicalLinkFrame(false, 7);
    check(score && reordered && ratio, "canonical signed note/link fixture admits");
    if (!score || !reordered || !ratio) return;
    p = {}; p.motionMode = 2; p.count = 2; p.lifetime = 4; p.force = 0;
    p.linkSpring = 3; p.linkRestScale = 0.5f;
    const auto originalNotes = score->noteTextureValues;
    const auto originalLinks = score->linkTextureValues;
    const auto spring = videorender::replayParticleBodies(p, 0.5, 1, score.get());
    check(spring == videorender::replayParticleBodies(p, 0.5, 1, reordered.get())
          && originalNotes == score->noteTextureValues && originalLinks == score->linkTextureValues,
          "coupled endpoints use canonical IDs independent of packing and do not mutate Block C");
    check(spring != videorender::replayParticleBodies(p, 0.5, 1, ratio.get()),
          "canonical link ratio weights dynamic spring force");
    p.linkConstraint = 1;
    const auto strut = videorender::replayParticleBodies(p, 0.5, 1, score.get());
    check(std::abs(std::hypot(strut[1][0] - strut[0][0], strut[1][1] - strut[0][1]) - 0.08f) < 1.0e-5f,
          "canonical harmonic strut constrains moving endpoint distance");
    p.linkSpring = 0;
    check(strut != videorender::replayParticleBodies(p, 0.5, 1, score.get()),
          "zero link strength disconnects the physical constraint");
}

void checkHistoricalBodyReplay()
{
    auto score = std::make_shared<arbitmod::Score>();
    score->scoreRevision = 1;
    arbitmod::Note a;
    a.id = -101; a.midiNote = 60; a.freqHz = 261.625565f; a.velocity = 127;
    a.startBeat = 0; a.lengthBeats = 8;
    score->notes.push_back(a);
    auto history = std::make_shared<videorender::ParticleHistorySource>();
    history->score = score;
    history->key.projectGeneration = history->key.sourceGeneration = history->key.helperGeneration = 1;
    history->key.backendGeneration = history->key.deviceGeneration = history->key.scoreGeneration = 1;
    history->key.beatMapGeneration = history->key.fpsGeneration = history->key.loopGeneration = history->key.seekGeneration = 1;
    videorender::ParticleParams p;
    p.motionMode = 2; p.historicalReplay = true; p.history = history; p.count = 64;
    p.lifetime = 2; p.force = 0; p.impulseX = 0.2f;
    const auto render = [&](double seconds, double origin = 0)
    {
        p.historyProjectSeconds = origin + seconds;
        return videorender::replayParticleBodies(p, seconds, 1, nullptr);
    };
    const auto golden = render(0.25);
    check(std::abs(golden[0][0] - 0.47f) < 1.0e-5f && golden[0][2] == 1 && golden[1][2] == 0,
          "historical note impulse has the expected constant-velocity trajectory");
    for (int frame = 0; frame <= 15; ++frame) render(double(frame) / 60);
    check(render(0.25) == golden, "historical sequential preview equals direct rational-time seek");
    render(10000); render(0.9); render(0.05);
    check(render(0.25) == golden, "historical state survives arbitrary seek ordering");
    p.resetTime = 0.125f;
    check(render(0.375) == golden && render(0.1)[0][2] == 0,
          "historical reset origin shifts physical time without a prior frame dependency");
    p.resetTime = 0; p.lifetime = 0.5f;
    check(render(0.75) == golden, "repeat restarts bodies using the following cycle's score");
    p.resetMode = 1;
    check(render(10000) == render(0.5), "hold stops input sampling and physical time after lifetime");
    p.resetMode = 0; p.lifetime = 2; p.impulseX = 0;
    auto notes = std::make_shared<arbitmod::Score>(*score);
    auto b = a; b.id = 202; b.midiNote = 72; b.freqHz *= 2;
    b.startBeat = 0.2f; b.lengthBeats = 0.01f;
    notes->notes.push_back(b);
    notes->links.push_back({-301, -101, 202, 3, 2, 0});
    history->score = notes;
    p.linkSpring = 5; p.linkRestScale = 0.5f;
    const auto afterRelease = render(0.5);
    check(afterRelease[0][0] > 0.42f && afterRelease[1][2] == 0,
          "a note shorter than one simulation step changes its linked body before note-off");
    const auto before = render(0.09);
    check(std::abs(before[0][0] - 0.42f) < 1.0e-6f && before[1][2] == 0,
          "upcoming notes do not create historical bodies or link forces early");
    check(render(0.5) == afterRelease, "canonical endpoint history replays identically after seeking backwards");
    auto changed = std::make_shared<arbitmod::Score>(*notes);
    changed->links.clear(); history->score = changed;
    check(render(0.5) != afterRelease, "editing canonical links invalidates their reconstructed trajectory");
    history->score = notes;
    auto shifted = std::make_shared<arbitmod::Score>(*notes);
    for (auto& note : shifted->notes) note.startBeat += 8;
    history->score = shifted;
    const auto range = render(0.5, 4);
    p.resetTime = 4;
    check(render(4.5) == range,
          "a clip and export range use project-time history rather than range-relative score time");
    p.resetTime = 0;

    auto tempoScore = std::make_shared<arbitmod::Score>(*score);
    tempoScore->notes[0].startBeat = 1;
    history->score = tempoScore;
    history->timeline.set({{0, 60, false}, {4, 180, true}}, {});
    const double startsAt = history->timeline.beatToSeconds(1);
    check(render(startsAt - 0.01)[0][2] == 0 && render(startsAt + 0.01)[0][2] == 1,
          "historical note events follow the canonical tempo ramp rather than constant BPM");
    history->timeline.reset(120, 4);

    // Two onsets and a varying envelope must both influence the final body.
    history->score = std::make_shared<const arbitmod::Score>();
    std::vector<arbitblockb::FeatureFrame> frames(120);
    for (std::size_t index = 0; index < frames.size(); ++index)
    {
        auto& f = frames[index]; f.frameIndex = static_cast<std::int64_t>(index);
        f.sampleIndex = static_cast<std::int64_t>(index + 1) * arbitblockb::kHop;
        f.rms = index < 30 ? 0.8f : 0.1f;
        if (index >= 10)
        {
            const auto onset = index >= 40 ? 40 : 10;
            f.onset = 1; f.onsetAge = float(index - onset) * float(arbitblockb::kHop) / 48000;
        }
    }
    history->audioFrames = &frames; history->audioSampleRate = 48000;
    p.geometryCount = 1; p.geometryAnchors[0] = {0.5f,0.2f,0.5f,0.2f};
    p.onsetGain = 0.2f; p.rmsGain = 0.1f; p.linkSpring = 0;
    const auto both = render(1);
    auto once = frames;
    for (std::size_t index = 40; index < once.size(); ++index)
        once[index].onsetAge = float(index - 10) * float(arbitblockb::kHop) / 48000;
    history->audioFrames = &once;
    const auto one = render(1);
    check(both[0][1] > one[0][1], "all historical audio onsets contribute separate impulses");
    history->audioFrames = &frames;
    check(render(1) == both && render(0.25) != both,
          "baked audio history is stable across direct range and sequential frame requests");
    const float resetOnset = float(11 * arbitblockb::kHop) / 48000;
    p.resetTime = resetOnset;
    const auto exactReset = render(double(resetOnset) + 0.1);
    p.onsetGain = 0;
    check(exactReset[0][1] > render(double(resetOnset) + 0.1)[0][1],
          "an onset exactly on the reset boundary excites newly reset bodies once");
    p.onsetGain = 0.2f; p.resetTime = 0;
    auto dense = frames;
    dense.resize(600);
    for (std::size_t i = 0; i < dense.size(); ++i)
    { dense[i].sampleIndex = std::int64_t(i + 1) * arbitblockb::kHop; dense[i].onset = 1; dense[i].onsetAge = 0; }
    history->audioFrames = &dense;
    p.lifetime = 10; p.historyProjectSeconds = 9;
    std::string error;
    check(!videorender::particleHistoryReady(p, error, 9) && error.find("512") != std::string::npos,
          "over-cap event histories are diagnosed rather than silently truncating replay");
    history->audioFrames = nullptr;
    check(!videorender::particleHistoryReady(p, error) && error.find("baked project audio") != std::string::npos,
          "missing history diagnoses the required project bake instead of substituting live audio");
}

struct ConnectedWaterSetup
{
    videowire::geometry::PortContract contract;
    std::optional<videowire::geometry::ValueDescriptor> rest, water;
    videowire::geometry::ValueDescriptor emitter;
    std::shared_ptr<videorender::ParticleGeometryBinding> binding;
    std::shared_ptr<videorender::ParticleHistorySource> history;
    videorender::ParticleParams p;
};

// One exact rest grid, Water Surface Simulation mesh and sphere emitter for the
// coupled-water checks. Callers set their own clock and automation fields.
std::optional<ConnectedWaterSetup> makeConnectedWaterSetup(std::string& error)
{
    using namespace videorender;
    using namespace videowire::geometry;
    ConnectedWaterSetup setup;
    setup.contract.carrier=CarrierKind::geometry3D;
    setup.contract.maxVertices=256; setup.contract.maxIndices=384;
    setup.rest=lowerGrid(18000,18000,1,9,9,.25f,setup.contract,error);
    setup.water=setup.rest ? lowerWaterSurfaceSimulation(*setup.rest,18001,18002,0,0,0,0,setup.contract,error,100) : std::nullopt;
    if (!setup.water) return std::nullopt;
    setup.history=std::make_shared<ParticleHistorySource>(waterreplayfixture::source());
    setup.binding=std::make_shared<ParticleGeometryBinding>(); setup.binding->scale=.25f;
    setup.binding->mesh=std::make_shared<const ValueDescriptor>(*setup.water);
    setup.emitter.carrier=CarrierKind::points3D; setup.emitter.stableId=18100;
    setup.emitter.sourceStableId=18100; setup.emitter.sourceRevision=1;
    setup.emitter.data=PointsData{};
    std::get<PointsData>(setup.emitter.data).points.push_back({{-.4f,-.2f,.12f},18101,1,{0,0,0,1}});
    setup.binding->emitters=std::make_shared<const ValueDescriptor>(setup.emitter);
    setup.p.motionMode=2; setup.p.simulationSpace=1; setup.p.historicalReplay=true;
    setup.p.history=setup.history; setup.p.geometryBinding=setup.binding; setup.p.count=1; setup.p.lifetime=1;
    setup.p.force=0; setup.p.impulseZ=-4; setup.p.bodyRadius=.015f; setup.p.colliderThickness=0;
    setup.p.restitution=1; setup.p.bodyMass=1; setup.p.resetMode=1;
    return setup;
}

void checkConnectedWaterBodies()
{
    using namespace videorender;
    using namespace videowire::geometry;
    std::string error;
    auto setup=makeConnectedWaterSetup(error);
    check(setup.has_value(),"connected water fixture authors its positive-mass exact rest grid");
    if (!setup) return;
    const auto& contract=setup->contract;
    const auto& rest=setup->rest;
    const auto& water=setup->water;
    auto& history=setup->history;
    auto& binding=setup->binding;
    const auto& emitter=setup->emitter;
    ParticleParams p=setup->p;
    constexpr double at=1.0/120;
    p.historyProjectSeconds=at;
    const auto frame=waterreplayfixture::frame(*history,at);
    ParticleSolidState state;
    const auto poses=replayParticleBodies(p,at,1,frame.get(),&error,&state);
    check(error.empty() && state.count==1 && state.colliders.meshIdentity==water->stableId,
        "production particle replay connects exact positive-mass water and sphere state");
    if (!error.empty() || state.count!=1) return;
    check(state.bodies[0].vz>0 && state.bodies[0].geometryIdentity==18101,
        "water contact reverses the emitted sphere velocity and preserves its authored identity");
    const auto& mesh=std::get<GeometryData>(rest->data);
    std::vector<double> masses(mesh.positions.size(),0);
    for (std::size_t triangle=0;triangle<mesh.indices.size();triangle+=3) {
        const auto a=mesh.positions[mesh.indices[triangle]],b=mesh.positions[mesh.indices[triangle+1]],c=mesh.positions[mesh.indices[triangle+2]];
        const auto area=.5*std::abs((double(b.x)-a.x)*(double(c.y)-a.y)-(double(b.y)-a.y)*(double(c.x)-a.x));
        for (int corner=0;corner<3;++corner)
            masses[mesh.indices[triangle+corner]]+=100*area*binding->scale*binding->scale/3;
    }
    std::vector<double> waterVelocity;
    WaterCoupledFrame completedWater;
    ParticleBodyState observedPoses;
    ParticleSolidState observedSolids;
    check(replayWaterCoupledBodies(p,at,1,frame.get(),observedPoses,observedSolids,nullptr,error,&waterVelocity,&completedWater)
        && observedPoses==poses && waterVelocity.size()==masses.size(),
        "coupled request publishes derived water velocity only after complete body and mesh replay");
    double waterMomentum=0;
    for (std::size_t vertex=0;vertex<waterVelocity.size();++vertex)
        waterMomentum+=masses[vertex]*waterVelocity[vertex]*binding->scale;
    auto evaluation=waterSurfaceFrameEvaluation(history.get(),at,at,frame,&completedWater);
    MaterializedInstancePlan visibleReplay;
    const auto* visible=replayGeometryMesh(*water,visibleReplay,error,&evaluation);
    check(visible && error.empty() && visible->positions.size()==state.colliders.meshVertexCount
        && std::equal(visible->positions.begin(),visible->positions.end(),state.colliders.meshVertices.begin(),
            [&](const auto& vertex,const auto& collider) {
                return collider==SolidVector{binding->centerX+binding->scale*vertex.x,
                    binding->centerY+binding->scale*vertex.y,binding->centerZ+binding->scale*vertex.z};
            }),"Geometry Core and particle sampling consume the same immutable completed water heights");
    auto wrongOperation=completedWater.operation; ++wrongOperation.stableId;
    std::vector<double> retainedHeights{42};
    check(!completedWater.sample(wrongOperation,completedWater.restMesh,completedWater.grid,at,retainedHeights,error)
        && retainedHeights==std::vector<double>{42},"another water operation cannot borrow a coupled frame");
    check(!completedWater.sample(completedWater.operation,completedWater.restMesh,completedWater.grid,2*at,retainedHeights,error)
        && retainedHeights==std::vector<double>{42},"historical water queries cannot substitute the current coupled frame");
    const double bodyMomentum=double(state.bodies[0].vz)-(-4);
    check(std::abs(waterMomentum+bodyMomentum)<1e-4,
        "body actual clamped delta momentum has equal opposite simulation-space water momentum");
    auto following=p; following.historyProjectSeconds=2*at;
    const auto nextFrame=waterreplayfixture::frame(*history,2*at);
    ParticleSolidState movedWater;
    check(replayWaterCoupledBodies(following,2*at,1,nextFrame.get(),observedPoses,movedWater,nullptr,error,
        nullptr,&completedWater),"following wave request completes its shared body and water frame");
    evaluation=waterSurfaceFrameEvaluation(history.get(),2*at,2*at,nextFrame,&completedWater);
    visible=replayGeometryMesh(*water,visibleReplay,error,&evaluation);
    check(visible && error.empty() && visible->positions.size()==movedWater.colliders.meshVertexCount
        && std::equal(visible->positions.begin(),visible->positions.end(),movedWater.colliders.meshVertices.begin(),
            [&](const auto& vertex,const auto& collider) {
                return collider==SolidVector{binding->centerX+binding->scale*vertex.x,
                    binding->centerY+binding->scale*vertex.y,binding->centerZ+binding->scale*vertex.z};
            }),"nonzero body-driven water is identical in geometry evaluation and the published collider");
    check(error.empty() && std::any_of(movedWater.colliders.meshVertices.begin(),
        movedWater.colliders.meshVertices.begin()+movedWater.colliders.meshVertexCount,
        [](const auto& position) { return position[2]<.5f; }),
        "body feedback displaces the exact drawn water mesh during the following wave step");
    auto exported=p;
    ParticleSolidState other;
    const auto exportFrame=waterreplayfixture::frame(*history,at,120,true);
    check(poses==replayParticleBodies(exported,at,1,exportFrame.get(),&error,&other)
        && error.empty() && sameSolidBodies(state,other)
        && state.colliders.meshVertices==other.colliders.meshVertices,
        "independent preview and export requests publish identical coupled sphere and water geometry");
    p.historyProjectSeconds=0; const auto zeroFrame=waterreplayfixture::frame(*history,0);
    replayParticleBodies(p,0,1,zeroFrame.get(),&error,&other);
    p.historyProjectSeconds=at;
    check(poses==replayParticleBodies(p,at,1,frame.get(),&error,&other)
        && error.empty() && state.colliders.meshVertices==other.colliders.meshVertices,
        "water direct seek after a reset reconstructs the same request-local coupled state");
    auto movedHistory=std::make_shared<ParticleHistorySource>(*history); ++movedHistory->key.seekGeneration;
    p.history=movedHistory;
    const auto seekFrame=waterreplayfixture::frame(*movedHistory,at);
    check(poses==replayParticleBodies(p,at,1,seekFrame.get(),&error,&other)
        && error.empty() && state.colliders.meshVertices==other.colliders.meshVertices,
        "fresh seek generation uses authored project-zero conditions without retained frame-order state");
    p.history=history;
    ParticleBodyState unchanged=poses; ParticleSolidState retained=state;
    auto invalid=p; invalid.historyProjectSeconds=.5; invalid.resetMode=0; invalid.lifetime=.1f;
    const auto cycleFrame=waterreplayfixture::frame(*history,.5);
    check(!replayWaterCoupledBodies(invalid,.5,1,cycleFrame.get(),unchanged,retained,nullptr,error)
        && !error.empty() && unchanged==poses && sameSolidBodies(retained,state),
        "cycled body clock refuses before replacing either published water or body result");
    invalid=p; invalid.count=64;
    check(!replayWaterCoupledBodies(invalid,at,1,frame.get(),unchanged,retained,nullptr,error),
        "coupled water preserves the existing combined 64-object scene bound");
    invalid=p; invalid.history.reset();
    check(!replayWaterCoupledBodies(invalid,at,1,frame.get(),unchanged,retained,nullptr,error),
        "missing history cannot publish a silently uncoupled water collider");
    check(!replayWaterCoupledBodies(p,at,1,cycleFrame.get(),unchanged,retained,nullptr,error),
        "coupled replay refuses a canonical frame from another project time");
    auto wrongBinding=std::make_shared<ParticleGeometryBinding>(*binding);
    auto wrongMesh=*water; std::swap(std::get<GeometryData>(wrongMesh.data).vertexIds[0],std::get<GeometryData>(wrongMesh.data).vertexIds[1]);
    wrongBinding->mesh=std::make_shared<const ValueDescriptor>(wrongMesh); invalid=p; invalid.geometryBinding=wrongBinding;
    check(!replayWaterCoupledBodies(invalid,at,1,frame.get(),unchanged,retained,nullptr,error),
        "water binding refuses a terminal mesh whose vertex identities differ from its exact operation replay");
    const auto zeroWater=lowerWaterSurfaceSimulation(*rest,18001,18002,0,0,0,0,contract,error);
    wrongBinding=std::make_shared<ParticleGeometryBinding>(*binding);
    wrongBinding->mesh=std::make_shared<const ValueDescriptor>(*zeroWater); invalid=p; invalid.geometryBinding=wrongBinding;
    replayParticleBodies(invalid,at,1,frame.get(),&error,&other);
    check(error.empty() && std::all_of(other.colliders.meshVertices.begin(),
        other.colliders.meshVertices.begin()+other.colliders.meshVertexCount,[](const auto& position) { return position[2]==.5f; }),
        "zero-mass water uses the ordinary replay without body feedback");
    // A music-excited rising surface must transfer velocity to a resting body.
    auto excitedHistory=std::make_shared<ParticleHistorySource>(*history);
    auto excitedScore=std::make_shared<arbitmod::Score>(*history->score);
    auto note=waterreplayfixture::note(7,0,50); note.lengthBeats=1;
    excitedScore->notes.push_back(note); excitedHistory->score=excitedScore;
    const auto excited=lowerWaterSurfaceSimulation(*rest,18001,18002,0,0,1,0,contract,error,100);
    auto risingBinding=std::make_shared<ParticleGeometryBinding>(*binding);
    risingBinding->mesh=std::make_shared<const ValueDescriptor>(*excited);
    auto restingEmitter=emitter;
    std::get<PointsData>(restingEmitter.data).points[0].position={
        float(-1+2*50.0/127),float(-1+2*waterreplay::identityY(7)),.0604f};
    risingBinding->emitters=std::make_shared<const ValueDescriptor>(restingEmitter);
    auto resting=p; resting.geometryBinding=risingBinding; resting.history=excitedHistory; resting.impulseZ=0;
    const auto excitedFrame=waterreplayfixture::frame(*excitedHistory,at);
    replayParticleBodies(resting,at,1,excitedFrame.get(),&error,&other);
    check(error.empty() && other.count==1 && other.bodies[0].vz>0,
        "the actual moving water endpoint interval transfers velocity to a resting sphere");
    invalid=p; invalid.bodyShape=2;
    check(!replayWaterCoupledBodies(invalid,at,1,frame.get(),unchanged,retained,nullptr,error),
        "water coupling explicitly refuses OBB bodies");
    invalid=p; invalid.historyProjectSeconds=2*at;
    check(!replayWaterCoupledBodies(invalid,at,1,nextFrame.get(),unchanged,retained,nullptr,error),
        "late clip origins cannot masquerade as project-zero water replay");
    error.clear();
    const auto savedBefore=admitValue(*water,contract,error);
    const auto savedAfter=admitValue(*binding->mesh,contract,error);
    check(savedBefore && savedAfter && encodeRuntimeValue(*savedBefore)==encodeRuntimeValue(*savedAfter),
        "derived coupled water motion does not mutate the authored saved geometry plan");
}

void checkWaterCoupledResetGuardAndReplayIdentity()
{
    using namespace videorender;
    std::string error;
    auto setup=makeConnectedWaterSetup(error);
    check(setup.has_value(),"water reset fixture authors its positive-mass exact rest grid");
    if (!setup) return;
    auto& history=setup->history;
    ParticleParams p=setup->p;
    p.historyClipId=1; p.historyNodeId=7;
    constexpr double stepSeconds=1.0/120, endSeconds=3*stepSeconds;

    // The resetTime lane is nonzero only at the sample one step into the
    // window. The requested end reads zero, so the plan-level zero-reset gate
    // passes and only the sampled history can refuse the mid-history reset.
    const auto historyWithResetAt=[&](bool midHistoryReset)
    {
        auto owned=std::make_shared<ParticleHistorySource>(*history);
        owned->parameterAt=[midHistoryReset](const std::string& destination,double seconds,
            double& value,std::string&)
        {
            constexpr double step=1.0/120;
            if (midHistoryReset && destination.find("/resetTime")!=std::string::npos
                && seconds>=step-1.0e-9 && seconds<2*step-1.0e-9) value=.25;
            return true;
        };
        return owned;
    };
    // One public replay request at a project time, with that time's own frame.
    const auto replayAt=[&](const std::shared_ptr<ParticleHistorySource>& source,double seconds,
        std::string& diagnostic,ParticleSolidState& solids)
    {
        auto input=p; input.history=source; input.historyProjectSeconds=seconds;
        const auto frame=waterreplayfixture::frame(*source,seconds);
        return replayParticleBodies(input,seconds,1,frame.get(),&diagnostic,&solids);
    };

    std::string diagnostic;
    ParticleSolidState refusedSolids;
    replayAt(historyWithResetAt(true),endSeconds,diagnostic,refusedSolids);
    check(diagnostic=="Water body coupling does not support mid-history particle resets" && refusedSolids.count==0,
        "water coupling refuses a sampled reset inside the replay window with the exact diagnostic and publishes no bodies");

    const auto steady=historyWithResetAt(false);
    ParticleSolidState single;
    const auto singleBodies=replayAt(steady,endSeconds,diagnostic,single);
    check(diagnostic.empty() && single.count==1,
        "water-coupled body replay without a mid-history reset completes at the window end");
    // replayParticleBodies clears its state on every call, so compare its
    // wrapper result with the direct coupled solver at the same window end.
    ParticleParams directInput=p; directInput.history=steady; directInput.historyProjectSeconds=endSeconds;
    const auto directFrame=waterreplayfixture::frame(*steady,endSeconds);
    ParticleBodyState direct{};
    ParticleSolidState directSolids;
    std::string directError;
    const bool directOk=replayWaterCoupledBodies(directInput,endSeconds,1,directFrame.get(),direct,directSolids,nullptr,directError);
    check(directOk && directError.empty() && direct==singleBodies && sameSolidBodies(directSolids,single)
        && directSolids.colliders.meshVertices==single.colliders.meshVertices,
        "replayParticleBodies wrapper matches the direct water-coupled solver at the window end");
}

void checkReactiveParticleGeometryHistory()
{
    using namespace videowire::geometry;
    using namespace videorender;
    std::string error;
    PortContract meshContract; meshContract.carrier=CarrierKind::geometry3D;
    meshContract.maxVertices=256; meshContract.maxIndices=384;
    PortContract pointsContract; pointsContract.carrier=CarrierKind::points3D; pointsContract.maxPoints=64;
    const auto base=lowerGrid(10000,10000,1,2,2,1,meshContract,error);
    check(base.has_value(),error.c_str()); if (!base) return;
    const auto basePoints=lowerPointsFromVertices(*base,10001,pointsContract,error);
    check(basePoints.has_value(),error.c_str()); if (!basePoints) return;
    PortContract vector; vector.carrier=CarrierKind::field; vector.fieldDomain=Domain::vertex;
    vector.fieldValueType=ValueType::vector; vector.fieldInterpolation=Interpolation::constant; vector.maxFieldElements=64;
    auto scalar=vector; scalar.fieldValueType=ValueType::floatValue;
    const auto position=lowerElementField(*base,10002,OperationCode::positionField,vector,error);
    FieldOperationSettings settings; settings.domain=Domain::vertex; settings.valueType=ValueType::floatValue;
    settings.values={2,.25,0,.125,10099,0,0,0};
    const auto sampled=position ? lowerConstructedField(10003,OperationCode::fieldScoreSample,settings,
        &*position,nullptr,nullptr,scalar,error) : std::nullopt;
    settings.valueType=ValueType::vector; settings.mode=1; settings.values={1,0,0,0,0,0,0,0};
    const auto offset=sampled ? lowerConstructedField(10004,OperationCode::fieldComposeVector,settings,
        &*sampled,nullptr,nullptr,vector,error) : std::nullopt;
    const auto scoreMesh=offset ? lowerVectorDisplacement(*base,*offset,10005,1,meshContract,error) : std::nullopt;
    const auto scorePoints=scoreMesh ? lowerPointsFromVertices(*scoreMesh,10006,pointsContract,error) : std::nullopt;
    check(scorePoints.has_value(),error.c_str()); if (!scorePoints) return;
    auto score=std::make_shared<arbitmod::Score>(); score->scoreRevision=1;
    arbitmod::Note note; note.id=-501; note.startBeat=0; note.lengthBeats=8;
    note.freqHz=score->rootFreq; note.trackId=0; score->notes.push_back(note);
    auto history=std::make_shared<ParticleHistorySource>(); history->score=score; history->timeline.reset(120,4);
    history->key.projectGeneration=history->key.sourceGeneration=history->key.helperGeneration=1;
    history->key.backendGeneration=history->key.deviceGeneration=history->key.scoreGeneration=1;
    history->key.beatMapGeneration=history->key.fpsGeneration=history->key.loopGeneration=history->key.seekGeneration=1;
    auto binding=std::make_shared<ParticleGeometryBinding>(); binding->scale=.25f;
    binding->emitters=std::make_shared<const ValueDescriptor>(*scorePoints);
    binding->attractors=binding->emitters; binding->points=binding->emitters;
    binding->mesh=std::make_shared<const ValueDescriptor>(*scoreMesh);
    ParticleParams p; p.motionMode=2; p.historicalReplay=true; p.history=history; p.geometryBinding=binding;
    p.geometryRevision=1; p.count=4; p.geometryCount=4; p.lifetime=1; p.gravity=0; p.force=0;
    ParticleGeometryFrame frame;
    check(sampleParticleGeometry(p,0,frame,error),error.c_str()); const auto first=p.geometryAnchors;
    check(sampleParticleGeometry(p,kParticleReplayTicksPerSecond/2,frame,error)
        && p.geometryAnchors!=first && frame.pointCount==4 && frame.count>frame.pointCount,
        "canonical Score changes emitter, attractor, mesh and point contacts at exact historical ticks");
    auto original=std::get<PointsData>(basePoints->data).points;
    std::sort(original.begin(),original.end(),[](const auto& a,const auto& b) { return a.stableId<b.stableId; });
    for (std::size_t i=0;i<original.size();++i) {
        const auto& point=original[i].position;
        // At 0.5 seconds the sole root note maps to x=one beat*0.125, y=z=0.
        const double distance=std::sqrt(std::pow(double(point.x)-.125,2)+double(point.y)*point.y+double(point.z)*point.z);
        const double expected=.5+.25*(double(point.x)+distance*.25);
        check(std::abs(p.geometryAnchors[i][0]-expected)<1.0e-6,
            "Score emitter displacement agrees with an independent nearest-note distance oracle");
        check(p.geometryIdentities[i]==original[i].stableId,"reactive points retain exact source identities");
    }
    const auto scoreAnchors=p.geometryAnchors;
    auto preview=p,exported=p;
    preview.historyProjectSeconds=exported.historyProjectSeconds=.5;
    const auto golden=replayParticleBodies(preview,.5,1,nullptr,&error);
    check(error.empty() && golden==replayParticleBodies(exported,.5,1,nullptr,&error),
        "independent preview and export replays share exact historical Score geometry");
    preview.historyProjectSeconds=.9; replayParticleBodies(preview,.9,1,nullptr,&error);
    preview.historyProjectSeconds=.1; replayParticleBodies(preview,.1,1,nullptr,&error);
    preview.historyProjectSeconds=.5;
    check(golden==replayParticleBodies(preview,.5,1,nullptr,&error),"reactive Score seek order leaves the replay unchanged");
    for (int at=0;at<=60;++at) { preview.historyProjectSeconds=double(at)/120; replayParticleBodies(preview,double(at)/120,1,nullptr,&error); }
    check(golden==replayParticleBodies(preview,.5,1,nullptr,&error),"sequential 120 Hz Score replay equals direct seek");
    preview.historyProjectSeconds=1.5;
    const auto looped=replayParticleBodies(preview,1.5,1,nullptr,&error);
    preview.resetTime=1;
    check(looped==replayParticleBodies(preview,1.5,1,nullptr,&error),"reactive repeat and explicit reset share project-time input geometry");
    preview.resetTime=0;
    check(looped==replayParticleBodies(preview,.5,1,nullptr,&error),"range-relative replay retains project-time field inputs");
    auto invalidHistory=std::make_shared<ParticleHistorySource>(*history); invalidHistory->key.sourceGeneration=0;
    auto rejected=p; rejected.history=invalidHistory;
    check(!sampleParticleGeometry(rejected,kParticleReplayTicksPerSecond/2,frame,error),"Score geometry refuses unowned canonical history metadata");
    rejected=p; rejected.history.reset();
    check(!sampleParticleGeometry(rejected,0,frame,error),"Score geometry refuses absent historical owner");
    auto editedScore=std::make_shared<arbitmod::Score>(*score); editedScore->notes[0].freqHz*=2;
    invalidHistory=std::make_shared<ParticleHistorySource>(*history); invalidHistory->score=editedScore;
    ++invalidHistory->key.scoreGeneration; rejected=p; rejected.history=invalidHistory;
    check(sampleParticleGeometry(rejected,kParticleReplayTicksPerSecond/2,frame,error)
        && rejected.geometryAnchors!=scoreAnchors,"owned Score changes affect replay without a current-frame aggregate");

    auto legacyScore=*base;
    scorefield::Binding legacy;
    legacy.fieldStableId=10020; legacy.sourceGeometryStableId=base->stableId;
    legacy.consumerGeometryStableId=10021; legacy.scoreSourceStableId=10099;
    legacy.gain=.25f; legacy.direction={1,0,0};
    legacy.elementIds=std::get<GeometryData>(base->data).vertexIds;
    for (const auto& point:std::get<GeometryData>(base->data).positions) legacy.positions.push_back({point.x,point.y,point.z});
    legacyScore.scoreFields.push_back(legacy);
    legacyScore.operations.push_back({10020,OperationCode::evaluateField,10020,10020,{}});
    legacyScore.operations.push_back({10021,OperationCode::transformTRS,base->stableId,10021,{}});
    legacyScore.stableId=10021; legacyScore.dispatchCount=legacyScore.operations.size();
    check(admitValue(legacyScore,meshContract,error).has_value(),error.c_str());
    auto legacyBinding=std::make_shared<ParticleGeometryBinding>(); legacyBinding->mesh=std::make_shared<const ValueDescriptor>(legacyScore);
    rejected=p; rejected.geometryBinding=legacyBinding;
    check(sampleParticleGeometry(rejected,0,frame,error),error.c_str()); const auto legacyStart=frame.segments;
    check(sampleParticleGeometry(rejected,kParticleReplayTicksPerSecond/2,frame,error) && frame.segments!=legacyStart,
        "legacy vertex Score Fields use their shared retained consumer during particle replay");
    auto shuffled=*scorePoints;
    std::reverse(std::get<PointsData>(shuffled.data).points.begin(),std::get<PointsData>(shuffled.data).points.end());
    auto invalidBinding=std::make_shared<ParticleGeometryBinding>(*binding);
    invalidBinding->emitters=std::make_shared<const ValueDescriptor>(shuffled);
    rejected=p; rejected.geometryBinding=invalidBinding;
    check(!sampleParticleGeometry(rejected,0,frame,error),"reactive replay refuses a changed admitted point identity order");

    std::vector<arbitblockb::FeatureFrame> master;
    auto sources=std::make_shared<videohelper::AnalysisSources>();
    for (int source=0;source<3;++source) {
        videohelper::AnalysisSourceStream stream; stream.trackId=source==1 ? 13:29; stream.group=source==2;
        stream.sampleRate=double(arbitblockb::kHop)*128;
        for (int i=0;i<512;++i) {
            arbitblockb::FeatureFrame sample; sample.sampleIndex=(i+1)*arbitblockb::kHop;
            sample.bands.fill(i<32 ? (source==0 ? .1f:source==1 ? .2f:.3f)
                : (source==0 ? .4f:source==1 ? .6f:.8f));
            stream.frames.push_back(sample);
        }
        if (source==0) master=stream.frames; else sources->push_back(std::move(stream));
    }
    history->audioFrames=&master; history->audioSampleRate=double(arbitblockb::kHop)*128; history->analysisSources=sources;
    for (const auto source:{spectrum::Source::master,spectrum::Source::track,spectrum::Source::group}) {
        spectrum::Binding field; field.source=source; field.sourceTrackId=source==spectrum::Source::master ? -1:source==spectrum::Source::track ? 13:29;
        field.firstBand=field.lastBand=0; field.direction={1,0,0}; field.deformer.mode=spectrum::Deformation::vector;
        const auto audioMesh=lowerAudioDeformer(*base,10010,field,meshContract,error);
        auto pointVector=vector; pointVector.fieldDomain=Domain::point;
        settings.domain=Domain::point; settings.valueType=ValueType::vector; settings.mode=0; settings.values={};
        const auto audioField=lowerAudioSpectrumField(*basePoints,10011,field,settings,pointVector,error);
        const auto audioPoints=audioField ? lowerVectorDisplacement(*basePoints,*audioField,10012,1,pointsContract,error) : std::nullopt;
        check(audioMesh.has_value() && audioPoints.has_value(),error.c_str()); if (!audioMesh || !audioPoints) continue;
        auto audioBinding=std::make_shared<ParticleGeometryBinding>(*binding);
        audioBinding->mesh=std::make_shared<const ValueDescriptor>(*audioMesh);
        audioBinding->emitters=std::make_shared<const ValueDescriptor>(*audioPoints);
        audioBinding->attractors=audioBinding->emitters; audioBinding->points=audioBinding->emitters;
        auto audio=p; audio.geometryBinding=audioBinding;
        check(sampleParticleGeometry(audio,0,frame,error),error.c_str()); const auto start=audio.geometryAnchors;
        check(sampleParticleGeometry(audio,kParticleReplayTicksPerSecond/2,frame,error)
            && audio.geometryAnchors!=start,"selected baked Spectrum changes all particle input geometries at replay time");
        const float amount=source==spectrum::Source::master ? .4f:source==spectrum::Source::track ? .6f:.8f;
        for (std::size_t i=0;i<original.size();++i)
            check(std::abs(audio.geometryAnchors[i][0]-(.5f+.25f*(original[i].position.x+amount)))<1.0e-6f,
                "selected Spectrum displacement has an independent source-band oracle");
        const auto audioGolden=audio.geometryAnchors; const auto segmentGolden=frame.segments;
        sampleParticleGeometry(audio,kParticleReplayTicksPerSecond*2,frame,error);
        check(sampleParticleGeometry(audio,kParticleReplayTicksPerSecond/2,frame,error)
            && audioGolden==audio.geometryAnchors && segmentGolden==frame.segments,
            "Spectrum field and deformer sampling is independent of seek and preview/export order");
        auto followerPoints=*audioPoints;
        followerPoints.spectrumFields[0].attackSeconds=.125f; followerPoints.spectrumFields[0].releaseSeconds=.25f;
        auto followerBinding=std::make_shared<ParticleGeometryBinding>(*audioBinding);
        followerBinding->emitters=std::make_shared<const ValueDescriptor>(followerPoints);
        auto follower=audio; follower.geometryBinding=followerBinding;
        check(sampleParticleGeometry(follower,kParticleReplayTicksPerSecond/2,frame,error),error.c_str());
        const auto followed=follower.geometryAnchors;
        sampleParticleGeometry(follower,kParticleReplayTicksPerSecond*2,frame,error);
        check(sampleParticleGeometry(follower,kParticleReplayTicksPerSecond/2,frame,error)
            && followed==follower.geometryAnchors && followed!=audioGolden,
            "Spectrum followers reconstruct bounded project history without saved live samples");
        auto audioPreview=audio,audioExport=audio;
        audioPreview.historyProjectSeconds=audioExport.historyProjectSeconds=.5;
        error.clear();
        const auto bodies=replayParticleBodies(audioPreview,.5,1,nullptr,&error);
        check(error.empty() && bodies==replayParticleBodies(audioExport,.5,1,nullptr,&error),
            "equal-time Spectrum preview and export replays share full project history");
        auto legacySpectrum=*base; auto terminal=field;
        terminal.fieldStableId=10030; terminal.sourceGeometryStableId=base->stableId; terminal.consumerGeometryStableId=10031;
        terminal.coordinates.assign(4,0); terminal.history.enabled=true; terminal.history.depth=2;
        terminal.history.spacingSeconds=.25f; terminal.history.coordinates={0,1,0,1};
        legacySpectrum.spectrumFields.push_back(terminal);
        legacySpectrum.operations.push_back({10030,OperationCode::evaluateField,10030,10030,{}});
        legacySpectrum.operations.push_back({10031,OperationCode::transformTRS,base->stableId,10031,{}});
        legacySpectrum.stableId=10031; legacySpectrum.dispatchCount=legacySpectrum.operations.size();
        check(admitValue(legacySpectrum,meshContract,error).has_value(),error.c_str());
        auto terminalBinding=std::make_shared<ParticleGeometryBinding>(); terminalBinding->mesh=std::make_shared<const ValueDescriptor>(legacySpectrum);
        auto terminalParameters=audio; terminalParameters.geometryBinding=terminalBinding;
        check(sampleParticleGeometry(terminalParameters,0,frame,error),error.c_str()); const auto terminalStart=frame.segments;
        check(sampleParticleGeometry(terminalParameters,kParticleReplayTicksPerSecond/2,frame,error) && frame.segments!=terminalStart,
            "baked legacy Spectral History rows use their shared consumer at each particle replay tick");
        auto missing=std::make_shared<ParticleHistorySource>(*history);
        if (source==spectrum::Source::master) missing->audioFrames=nullptr; else missing->analysisSources.reset();
        audio.history=missing;
        check(!sampleParticleGeometry(audio,0,frame,error),"missing selected full-project bake refuses reactive particle geometry");
        if (source==spectrum::Source::track) {
            auto wrong=std::make_shared<videohelper::AnalysisSources>(*sources); (*wrong)[0].group=true;
            missing->analysisSources=wrong;
            check(!sampleParticleGeometry(audio,0,frame,error),"selected source tap identity cannot switch track to group");
            wrong=std::make_shared<videohelper::AnalysisSources>(*sources); wrong->push_back(wrong->front()); missing->analysisSources=wrong;
            check(!sampleParticleGeometry(audio,0,frame,error),"duplicate selected project source identities refuse");
            wrong=std::make_shared<videohelper::AnalysisSources>(*sources); (*wrong)[0].frames[0].bands[0]=std::numeric_limits<float>::quiet_NaN(); missing->analysisSources=wrong;
            check(!sampleParticleGeometry(audio,0,frame,error),"nonfinite source bands refuse before normalization can hide missing history");
            wrong=std::make_shared<videohelper::AnalysisSources>(*sources); (*wrong)[0].frames[20].sampleIndex+=arbitblockb::kHop; missing->analysisSources=wrong;
            follower.history=missing;
            check(!sampleParticleGeometry(follower,kParticleReplayTicksPerSecond/2,frame,error),
                "missing older follower input refuses even when the current source frame is valid");
            wrong=std::make_shared<videohelper::AnalysisSources>(*sources); (*wrong)[0].frames[0].sampleIndex+=arbitblockb::kHop; missing->analysisSources=wrong;
            check(!sampleParticleGeometry(audio,0,frame,error),"range-scoped source samples cannot replace project-zero history");
        }
        auto body=*audioMesh; body.spectrumFields[0].history.live=true; body.spectrumFields[0].history.enabled=true;
        check(!validateParticleGeometry(body,CarrierKind::geometry3D,error),"historical particles refuse live spectral history");
        body=*audioMesh; body.operations.resize(33);
        check(!validateParticleGeometry(body,CarrierKind::geometry3D,error),"reactive geometry preserves the 32-operation cap");
        const auto admitted=admitValue(*audioMesh,meshContract,error);
        check(admitted.has_value(),error.c_str());
        if (admitted) {
            const auto encoded=encodeLoweredPlanText(lowerRuntimePlan(meshContract,*admitted));
            ParticleParams bodyParameters;
            const std::string bodyPayload="<NodeParams motionMode=\"2\" historicalReplay=\"1\" simulationSpace=\"1\" count=\"4\" bodyGeometryPlan=\""+encoded+"\"/>";
            check(!videowire::admitParticleParameters(bodyPayload,bodyParameters,error) && error.find("spectrum")!=std::string::npos,
                "direct body admission refuses unscheduled audio deformation");
        }
    }
}

void checkAnimatedGeometryBodies()
{
    std::string error;
    auto fixture=particleGeometryFixture(true,error),stationary=particleGeometryFixture(false,error);
    check(fixture.has_value() && stationary.has_value(),error.c_str());
    if (!fixture || !stationary) return;
    auto p=*fixture;
    const auto render=[&](double time) { p.historyProjectSeconds=time; return videorender::replayParticleBodies(p,time,1,nullptr,&error); };
    const auto moved=render(0.5);
    check(error.empty() && moved[0][2]==1,"animated projected mesh barriers retain a live body");
    stationary->historyProjectSeconds=0.5;
    const auto fixed=videorender::replayParticleBodies(*stationary,0.5,1,nullptr,&error);
    auto free=p; free.geometryBinding.reset();
    check(moved!=fixed && moved!=videorender::replayParticleBodies(free,0.5,1,nullptr),
          "timeline-driven mesh barriers alter contact motion independently of static barriers");
    render(3.7); render(0.1);
    check(render(0.5)==moved,"animated collider replay is independent of seek order");
    for (int frame=0;frame<=15;++frame) render(double(frame)/30);
    check(render(0.5)==moved,"sequential collider preview matches a direct seek");
    const auto looped=render(1.5);
    p.resetTime=1;
    check(render(1.5)==looped && looped!=moved,
          "repeat and explicit reset share a new cycle using current project geometry");
    p.resetTime=0; p.historyProjectSeconds=1.5;
    check(videorender::replayParticleBodies(p,0.5,1,nullptr,&error)==looped,
          "range-relative clip time preserves project-time collider history");
    p.resetMode=1;
    check(render(100)==render(1),"hold freezes geometry sampling as well as body integration");
    p.resetMode=0;
    render(1000.5);
    check(error.find("coordinate limit")!=std::string::npos,
          "unbounded timeline projection stops with a diagnostic");
    error.clear();
    videowire::geometry::MaterializedInstancePlan excessive;
    excessive.points[1].points.resize(65);
    check(!videowire::geometry::particleGeometryIntermediatesFit(excessive,error),
          "particle geometry rejects over-cap intermediate point sources");
    error.clear();
    auto binding=std::make_shared<videorender::ParticleGeometryBinding>(*p.geometryBinding);
    videowire::geometry::PortContract pointsContract; pointsContract.carrier=videowire::geometry::CarrierKind::points3D; pointsContract.maxPoints=64;
    const auto points=videowire::geometry::lowerPointsFromVertices(*binding->mesh,9810,pointsContract,error);
    check(points.has_value(),error.c_str());
    if (points)
    {
        binding->emitters=std::make_shared<const videowire::geometry::ValueDescriptor>(*points);
        binding->attractors=binding->emitters; binding->points=binding->emitters; binding->mesh.reset();
        p.geometryBinding=binding; p.count=4;
        videorender::ParticleGeometryFrame a,b;
        check(videorender::sampleParticleGeometry(p,0,a,error),error.c_str());
        const auto start=p.geometryAnchors;
        check(videorender::sampleParticleGeometry(p,videorender::kParticleReplayTicksPerSecond/2,b,error)
            && p.geometryCount==4 && b.count==4 && start!=p.geometryAnchors,
            "animated point emission, attraction and disc contacts share one geometry evaluation");
    }
}

struct FakeLayer
{
    struct Clock { double time = 0.0, timeSec = 0.0, timeDelta = 0.0, beat = 0.0; int frame = 0; bool playing = false; };
    struct Effect { bool enabled = false; int type = -1; float params[9] {}; };
    Clock shaderClock;
    std::map<std::string, double> genParams;
    unsigned texture = 0;
    int texWidth = 0, texHeight = 0;
    std::string nativeTextureBackend;
    std::uintptr_t nativeTextureView = 0;
    arbitgpu::NativeTextureViewDescriptor nativeTextureDescriptor;
    std::shared_ptr<const void> nativeTextureOwner;
    std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> canonicalBlockCFrame;
    int shaderTransitionFromClipId = 0;
    bool shaderSource = false;
    bool scoreSource = false;
    bool isAdjustment = false;
    bool flatShaderBridge = false;
    videowire::ImmutableShaderOperationPlan shaderOperationPlan;
    std::map<int, std::map<std::string, double>> shaderOperationParameters;
    bool particleSource = false;
    bool importedParticleOverlay = false;
    bool particleStateReset = false;
    bool particleTriggerConnected = false;
    videorender::ParticleParams particleParameters;
    std::shared_ptr<const videorender::ParticleHistorySource> particleHistory;
    double particleProjectSeconds = 0.0;
    bool audioPresent = false;
    struct Audio { float rms = 0.0f, onset = 0.0f, onsetAge = 0.0f; } audioFeatures;
    int particleTriggerCount = 0;
    float particleTriggerStrength = 0.0f;
    std::uint64_t visualPlanStructuralRevision = 0;
    bool visualPlanTelemetryHold = false;
    bool graphTemporalActive = false;
    int graphTemporalNodeId = 0;
    visualtemporaloperation::Payload graphTemporalPayload;
    int particleNodeId = 0;
    int drawShapeNodeId = 0;
    bool drawShape = false;
    bool inspectionDrawShapeOutput = false;
    bool drawShapeEllipse = false;
    bool drawShapeHasSecondary = false;
    bool drawShapeSecondaryEllipse = false;
    int drawShapeOperation = 0;
    float drawShapeCx = 0.5f, drawShapeCy = 0.5f, drawShapeW = 1.0f, drawShapeH = 1.0f;
    float drawShape2Cx = 0.5f, drawShape2Cy = 0.5f, drawShape2W = 0.0f, drawShape2H = 0.0f;
    float drawShapeR = 1.0f, drawShapeG = 1.0f, drawShapeB = 1.0f, drawShapeA = 1.0f;
    bool pathMatte = false;
    bool pathMatteEllipse = false, pathMatteHasSecondary = false;
    bool pathMatteSecondaryEllipse = false, pathMatteInvert = false;
    int pathMatteOperation = 0;
    float pathMatteCx = 0.5f, pathMatteCy = 0.5f, pathMatteW = 1.0f, pathMatteH = 1.0f;
    float pathMatte2Cx = 0.5f, pathMatte2Cy = 0.5f, pathMatte2W = 0.0f, pathMatte2H = 0.0f;
    float scale = 2.0f, translateX = 1.0f, translateY = 1.0f, rotationDeg = 4.0f;
    float cropLeft = 0.1f, cropRight = 0.1f, cropTop = 0.1f, cropBottom = 0.1f;
    const Effect* effects = reinterpret_cast<const Effect*> (1);
    int effectCount = 2;
    Effect graphFeedbackEffect;
    Effect graphKeyCleanupEffect;
    Effect graphLumaKeyEffect;
    Effect graphCommonEffect;
    bool graphKeyCleanupActive = false;
    colortransform::Description graphColorTransform;
    bool graphColorTransformActive = false;
    bool graphMotionBlurActive = false;
    unsigned graphMotionTexture = 0;
    visualtemporalsampling::Payload graphMotionBlur;
    float graphKeyCleanupChoke = 0.0f, graphKeyCleanupFeather = 0.0f;
    float graphKeyCleanupEdgeR = 1.0f, graphKeyCleanupEdgeG = 1.0f;
    float graphKeyCleanupEdgeB = 1.0f, graphKeyCleanupEdgeAmount = 0.0f;
    bool graphKeyCleanupMatteView = false;
    bool feedbackHistoryReset = false;
    bool feedbackHistoryHold = false;
    unsigned lutTexture = 4;
    int lutSize = 17;
    int maskType = 2;
    bool maskInvert = true;
    unsigned matteTexture = 0;
    unsigned matteTextureB = 0;
    int matteWidth = 0, matteHeight = 0;
    int matteWidthB = 0, matteHeightB = 0;
    bool matteApply = false, matteInvert = false;
    int matteCombineMode = -1;
    float matteBlack = 0.0f, matteWhite = 1.0f;
    float matteErodeDilate = 0.0f, matteFeather = 0.0f, matteChoke = 0.0f;
    unsigned depthTexture = 0;
    int depthWidth = 0, depthHeight = 0;
    bool depthFog = false;
    int depthEffect = 0;
    float fogNear = 0.0f, fogFar = 1.0f, fogDensity = 1.0f;
    float fogRed = 1.0f, fogGreen = 1.0f, fogBlue = 1.0f, fogAlpha = 1.0f;
    float depthParam0 = 0.0f, depthParam1 = 0.0f, depthParam2 = 0.0f;
    float depthColorRed = 1.0f, depthColorGreen = 1.0f, depthColorBlue = 1.0f;
    float opacity = 0.4f;
    int blendMode = 3;
};

struct FakeProductRenderer
{
    int colorAovExecutions = 0;
    int motionBlurPreparations = 0;
    int temporalFeedbackPreparations = 0;
    int bridgePreparations = 0;
    bool rejectColorAov = false;
    bool rejectTemporalFeedback = false;
    bool rejectFlatShaderBridge = false;
    renderpassoutput::Description publishedColorAov;
    std::optional<videowire::TemporalFeedbackPass> preparedTemporalFeedback;
    std::optional<videowire::TemporalSamplingExecution> preparedMotionBlur;

    int outputWidth() const { return 1920; }
    int outputHeight() const { return 1080; }

    bool prepareTemporalFeedbackPass (const videowire::TemporalFeedbackPass& pass,
                                      std::string& error)
    {
        ++temporalFeedbackPreparations;
        if (rejectTemporalFeedback)
        {
            error = "native temporal feedback backend is unsupported";
            return false;
        }
        preparedTemporalFeedback = pass;
        return true;
    }

    bool replaceColorAovPass (const renderpassoutput::Description& description,
                              std::string& error)
    {
        ++colorAovExecutions;
        if (rejectColorAov)
        {
            error = "native Color AOV backend is unsupported";
            return false;
        }
        publishedColorAov = description;
        return true;
    }

    bool prepareMotionBlurPass (const videowire::TemporalSamplingExecution& execution,
                                FakeLayer& layer, std::string& error)
    {
        ++motionBlurPreparations;
        const auto& payload = execution.pass.payload;
        if (! visualtemporalsampling::validate(payload, &error)
            || payload.mode != visualtemporalsampling::Mode::motionBlur
            || execution.transition.action == visualtemporalsampling::TransitionAction::reject)
            return false;
        preparedMotionBlur = execution;
        layer.graphMotionBlurActive = true;
        layer.graphMotionTexture = 9;
        layer.graphMotionBlur = payload;
        error.clear();
        return true;
    }

    bool replaceSceneAovPass (const sceneaov::Payload&, std::string&)
    {
        return true;
    }

    bool replaceAovInspectionPass (const sceneaov::Payload&,
                                   const aovinspection::Payload&,
                                   std::string&)
    {
        return true;
    }

    bool prepareFlatShaderBridge (FakeLayer&, std::string&)
    {
        ++bridgePreparations;
        return ! rejectFlatShaderBridge;
    }
};

videowire::CompiledVisualLayerPlan colorAovPlan()
{
    videowire::CompiledVisualLayerPlan plan;
    plan.clipId = 91;
    plan.producerValidated = true;
    plan.nodeKinds = { std::string (coloraov::kOperationKind) };
    plan.nodeIds = { 811 };
    plan.ports = {
        { 811, 1, 1, "out", "frame", "image", "rgba16f", "linearSRGB" }
    };
    coloraov::Payload payload;
    payload.extent = { 640, 360 };
    plan.operations = {
        { 811, std::string (coloraov::kOperationKind),
          std::string (coloraov::kBackendCapability), coloraov::serialize (payload) }
    };
    return plan;
}

videowire::CompiledVisualLayerPlan motionAovPlan()
{
    videowire::CompiledVisualLayerPlan plan;
    plan.clipId = 92;
    plan.structuralRevision = 4;
    plan.producerValidated = true;
    plan.nodeKinds = { std::string (opticalflowoperation::kOperationKind) };
    plan.nodeIds = { 812 };
    plan.ports = {
        { 812, 4, 2, "out", "frame", "motionVectors", "rg16f", "data" }
    };
    opticalflowoperation::Payload payload;
    payload.extent = { 640, 360 };
    plan.operations = {
        { 812, std::string (opticalflowoperation::kOperationKind),
          std::string (opticalflowoperation::kBackendCapability),
          opticalflowoperation::serialize (payload) }
    };
    return plan;
}

videowire::CompiledVisualLayerPlan importedAnimationPlan()
{
    videowire::CompiledVisualLayerPlan plan;
    plan.clipId = 7;
    plan.structuralRevision = 9;
    plan.producerValidated = true;
    plan.nodeKinds = {
        std::string(visualanimationoperation::kSourceNodeKind),
        std::string(visualanimationoperation::kDeformationNodeKind)
    };
    plan.nodeIds = { 70, 71 };
    const std::array<const char*, 4> types {
        "mesh", "skeleton", "morphTargets", "animationClip"
    };
    for (int port = 0; port < 4; ++port)
    {
        plan.ports.push_back({ 70, port, 1, "out", "control", types[(size_t) port],
                               "unspecified", "unspecified" });
        plan.ports.push_back({ 71, port, 1, "in", "control", types[(size_t) port],
                               "unspecified", "unspecified" });
        plan.edges.push_back({ 70, port, 71, port });
    }
    plan.ports.push_back({ 70, 4, 1, "out", "control", "scene3D",
                           "unspecified", "unspecified" });
    plan.ports.push_back({ 71, 4, 1, "out", "control", "mesh",
                           "unspecified", "unspecified" });

    visualanimationimport::Request request;
    request.sourceStableId = 71;
    request.deformationStableId = 72;
    request.schedule = { 71, 72 };
    request.asset.id = "model-asset-1";
    request.asset.version = 3;
    request.asset.contentSha256 = std::string(64, 'a');
    request.asset.sourceMediaType = "model/gltf-binary";
    request.asset.sourceByteSize = 4096;
    request.sceneIndex = 0;
    request.clipName = "Walk";
    plan.operations = {
        { 70, std::string(visualanimationoperation::kSourceNodeKind),
          std::string(visualanimationoperation::kSourceBackendCapability), "" },
        { 71, std::string(visualanimationoperation::kDeformationNodeKind),
          std::string(visualanimationoperation::kDeformationBackendCapability),
          visualanimationoperation::encode(request) }
    };
    return plan;
}

videowire::CompiledVisualLayerPlan directPlan()
{
    videowire::CompiledVisualLayerPlan plan;
    plan.clipId = 7;
    plan.producerValidated = true;
    plan.nodeKinds = { "video.source", "video.out" };
    plan.nodeIds = { 11, 12 };
    plan.ports = {
        { 11, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 12, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    plan.edges = { { 11, 0, 12, 0 } };
    plan.operations = {
        { 11, "video.source", "source-decode", "" },
        { 12, "video.out", "native-gpu", "" }
    };
    return plan;
}

videowire::CompiledVisualLayerPlan mediumBudgetPlan (int clipId, int firstNodeId)
{
    videowire::CompiledVisualLayerPlan plan;
    plan.clipId = clipId;
    plan.structuralRevision = 17;
    plan.producerValidated = true;
    plan.nodeKinds = { "video.source", "video.transform", "video.effects",
                       "video.mask.shape", "video.blend", "video.out" };
    for (int index = 0; index < 6; ++index)
    {
        const int nodeId = firstNodeId + index;
        plan.nodeIds.push_back(nodeId);
        plan.operations.push_back({ nodeId, plan.nodeKinds[(size_t) index],
            index == 0 ? "source-decode" : "native-gpu", "" });
        if (index > 0)
            plan.ports.push_back({ nodeId, 0, 1, "in", "frame", "image", "rgba8", "sRGB" });
        if (index + 1 < 6)
        {
            plan.ports.push_back({ nodeId, 1, 1, "out", "frame", "image", "rgba8", "sRGB" });
            plan.edges.push_back({ nodeId, 1, nodeId + 1, 0 });
        }
    }
    videowire::VisualPlanResourceUsage usage;
    std::string error;
    if (! videowire::accountVisualPlanResources(plan, 64, 64, usage, error))
        return {};
    plan.descriptorCount = usage.descriptors;
    plan.operationCount = usage.operations;
    plan.sceneRecordCount = usage.sceneRecords;
    plan.frameOutputCount = usage.frameOutputs;
    plan.peakLiveFrameCount = usage.peakLiveFrames;
    plan.allocatedFrameSlotCount = usage.frameSlots;
    return plan;
}

videowire::CompiledVisualLayerPlan keyCleanupPlan()
{
    auto plan = directPlan();
    plan.nodeKinds = { "video.source", "visual.key.cleanup", "video.out" };
    plan.nodeIds = { 11, 13, 12 };
    plan.ports = {
        { 11, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 13, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 13, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 12, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    plan.edges = { { 11, 0, 13, 0 }, { 13, 1, 12, 0 } };
    plan.operations = {
        { 11, "video.source", "source-decode", "" },
        { 13, "visual.key.cleanup", "native-gpu",
          "<KeyCleanup schemaVersion=\"1\" keyR=\"0.1\" keyG=\"0.9\" keyB=\"0.2\" tolerance=\"0.25\" softness=\"0.15\" despill=\"0.75\" choke=\"0\" feather=\"0\" edgeRed=\"1\" edgeGreen=\"1\" edgeBlue=\"1\" edgeAmount=\"0\" view=\"0\"/>" },
        { 12, "video.out", "native-gpu", "" }
    };
    return plan;
}

videowire::CompiledVisualLayerPlan mattePlan()
{
    auto plan = directPlan();
    plan.nodeKinds = { "video.source", "visual.matte.asset", "visual.matte.refine",
                       "visual.matte.apply", "video.out" };
    plan.nodeIds = { 11, 41, 42, 43, 12 };
    plan.edges = { { 11, 0, 43, 0 }, { 41, 0, 42, 0 },
                   { 42, 1, 43, 1 }, { 43, 2, 12, 0 } };
    plan.operations = {
        { 11, "video.source", "source-decode", "" },
        { 41, "visual.matte.asset", "source-decode",
          "<MatteAssetBinding matteAssetId=\"matte-1\" state=\"available\" cacheKey=\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\" contentReceipt=\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\" framePrefix=\"subject-\" frameExtension=\".rgba\" firstFrame=\"42\" frameDigits=\"3\" backend=\"rgba-cpu-decode-native-gpu-upload\"/>" },
        { 42, "visual.matte.refine", "native-gpu",
          "<NodeParams invert=\"1\" black=\"-2\" white=\"3\" erodeDilate=\"20\" feather=\"20\" choke=\"-2\"/>" },
        { 43, "visual.matte.apply", "native-gpu", "" },
        { 12, "video.out", "native-gpu", "" }
    };
    return plan;
}

videowire::CompiledVisualLayerPlan combinedMattePlan()
{
    auto plan = mattePlan();
    plan.nodeKinds = { "video.source", "visual.matte.asset", "visual.matte.asset",
                       "visual.matte.combine", "visual.matte.refine", "visual.matte.apply", "video.out" };
    plan.nodeIds = { 11, 41, 44, 45, 42, 43, 12 };
    plan.edges = { {11,0,43,0}, {41,0,45,0}, {44,0,45,1}, {45,2,42,0},
                   {42,1,43,1}, {43,2,12,0} };
    auto bindingB = plan.operations[1];
    bindingB.nodeId = 44;
    bindingB.payloadXml = "<MatteAssetBinding matteAssetId=\"matte-2\" matteAssetVersion=\"v2\" state=\"available\" cacheKey=\"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\" contentReceipt=\"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\" framePrefix=\"other-\" frameExtension=\".rgba\" firstFrame=\"7\" frameDigits=\"3\" fps=\"24\" frames=\"8\" backend=\"rgba-cpu-decode-native-gpu-upload\"/>";
    plan.operations = { plan.operations[0], plan.operations[1], bindingB,
        {45, "visual.matte.combine", "native-gpu", "<NodeParams mode=\"3\"/>"},
        plan.operations[2], plan.operations[3], plan.operations[4] };
    return plan;
}

videowire::CompiledVisualLayerPlan reusedMattePlan()
{
    auto plan = combinedMattePlan();
    plan.nodeKinds.erase(plan.nodeKinds.begin() + 2);
    plan.nodeIds.erase(plan.nodeIds.begin() + 2);
    plan.operations.erase(plan.operations.begin() + 2);
    plan.edges[2].fromNodeId = 41;
    plan.ports = {
        { 11, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 41, 0, 1, "out", "frame", "mask", "r8", "linearSRGB" },
        { 45, 0, 1, "in", "frame", "mask", "r8", "linearSRGB" },
        { 45, 1, 1, "in", "frame", "mask", "r8", "linearSRGB" },
        { 45, 2, 1, "out", "frame", "mask", "r8", "linearSRGB" },
        { 42, 0, 1, "in", "frame", "mask", "r8", "linearSRGB" },
        { 42, 1, 1, "out", "frame", "mask", "r8", "linearSRGB" },
        { 43, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 43, 1, 1, "in", "frame", "mask", "r8", "linearSRGB" },
        { 43, 2, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 12, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    return plan;
}

videowire::CompiledVisualLayerPlan depthPlan()
{
    auto plan = directPlan();
    plan.nodeKinds = { "video.source", "visual.depth.asset", "visual.depth.fog", "video.out" };
    plan.nodeIds = { 11, 61, 62, 12 };
    plan.ports = {
        { 11, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 61, 0, 0, "out", "none", "unspecified", "unspecified", "unspecified" },
        { 61, 1, 1, "out", "frame", "depth", "r16", "linearSRGB" },
        { 62, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 62, 1, 1, "in", "frame", "depth", "r16", "linearSRGB" },
        { 62, 2, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 12, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    plan.edges = { { 11, 0, 62, 0 }, { 61, 1, 62, 1 }, { 62, 2, 12, 0 } };
    plan.operations = {
        { 11, "video.source", "source-decode", "" },
        { 61, "visual.depth.asset", "source-decode",
          "<DepthAssetBinding depthAssetId=\"depth-1\" depthAssetVersion=\"depth-sequence-v1\" state=\"available\" cacheKey=\"depth-job-1-abc/published\" contentReceipt=\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\" analysisReceipt=\"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\" framePrefix=\"depth_\" frameExtension=\".png\" firstFrame=\"1\" frameDigits=\"6\" width=\"2\" height=\"2\" fps=\"24\" frames=\"3\" format=\"r16-unorm\"/>" },
        { 62, "visual.depth.fog", "native-gpu",
          "<NodeParams near=\"-1\" far=\"2\" density=\"99\" red=\"0.2\" green=\"0.3\" blue=\"0.4\" alpha=\"2\"/>" },
        { 12, "video.out", "native-gpu", "" }
    };
    return plan;
}
}

// Solid 3D bodies are born only from notes or emitters. A notes-free project
// with no emitter must refuse the empty frame with a message, while a project
// with one note keeps its ordinary replay.
void checkNotesRequiredForSolidBodies()
{
    struct Layer {
        unsigned texture=0; float opacity=1; bool particleSource=true;
        std::string nativeTextureBackend;
        std::uintptr_t nativeTextureView=0;
        arbitgpu::NativeTextureViewDescriptor nativeTextureDescriptor;
        std::shared_ptr<const void> nativeTextureOwner;
        int texWidth=0,texHeight=0;
    };
    const videohelper::geometry::PlanOwnerIdentity owner{1,2,3,4,5,videohelper::geometry::PlanUse::preview};
    std::string error;
    auto p=solidBodyFixture();
    p.geometryCount=0; p.count=64;
    check(sampleSolidBodyFixture(p,1.24,error).count==0 && error.empty(),
          "a notes-free Solid 3D replay yields zero bodies without a replay error");
    check(videorender::particleSolidBodiesNeedNotes(p),
          "a notes-free Solid 3D replay with no emitter is classified as note-driven");
    Layer empty;
    check(!videorender::renderParticleSolids(p,owner,7,96,96,empty,error)
          && error.find("need notes on the timeline")!=std::string::npos
          && !empty.nativeTextureOwner && empty.opacity==1,
          "an empty notes-free Solid 3D frame is refused with a message instead of a silent blank");
    auto emitted=solidBodyFixture();
    auto emitters=std::make_shared<videorender::ParticleGeometryBinding>();
    emitters->emitters=std::make_shared<const videowire::geometry::ValueDescriptor>();
    emitted.geometryCount=0; emitted.geometryBinding=emitters;
    check(!videorender::particleSolidBodiesNeedNotes(emitted),
          "an emitter-driven Solid 3D replay is not refused for lacking notes");
    check(!videorender::particleSolidBodiesNeedNotes(manySolidBodyFixture(1)),
          "a project with a note is not refused for lacking notes");
}

// Particle score impulse: zero gain leaves a note-born body at rest, a positive
// gain kicks it once per onset along the unit impulse direction, the initial
// impulse integrates at fixed 120 Hz steps, and seeks reproduce the same bodies.
void checkParticleScoreImpulse()
{
    using namespace videorender;
    std::string error;
    auto score=std::make_shared<arbitmod::Score>(); score->scoreRevision=1;
    arbitmod::Note note; note.id=7001; note.midiNote=60; note.freqHz=261.6256f;
    note.velocity=100; note.startBeat=2; note.lengthBeats=4;
    score->notes.push_back(note);
    auto history=std::make_shared<ParticleHistorySource>();
    history->score=score; history->timeline.reset(120,4);
    auto& key=history->key;
    key.projectGeneration=key.sourceGeneration=key.helperGeneration=key.backendGeneration=key.deviceGeneration=1;
    key.scoreGeneration=key.beatMapGeneration=key.fpsGeneration=key.loopGeneration=key.seekGeneration=1;
    ParticleParams base;
    base.motionMode=2; base.simulationSpace=1; base.historicalReplay=true;
    base.count=1; base.geometryCount=0; base.lifetime=2; base.force=0; base.bodyRadius=0.02f;
    base.history=history;
    // Beat 2 at 120 BPM is 1.0 s. Sampling at 1.5 s covers exactly one interval after it.
    const double onsetSeconds=history->timeline.beatToSeconds(2), sampleSeconds=1.5;
    const double velocity=note.velocity/127.0;
    const auto sample=[&](ParticleParams p,double seconds)
    {
        p.historyProjectSeconds=seconds;
        ParticleSolidState out;
        replayParticleBodies(p,seconds,1,nullptr,&error,&out);
        return out;
    };

    // Gain 0 is the baseline: no force, no impulse, so the body starts at rest.
    const auto silent=sample(base,sampleSeconds);
    check(silent.count==1 && silent.bodies[0].vx==0 && silent.bodies[0].vy==0 && silent.bodies[0].vz==0,
          "zero score impulse gain leaves a note-born body at rest");

    // With no impulse controls the kick is +Y, and its size is gain times normalized velocity.
    const float kickGain=2.0f;
    const double expected=kickGain*velocity;
    auto kicked=base; kicked.scoreImpulseGain=kickGain;
    const auto up=sample(kicked,sampleSeconds);
    check(up.count==1 && std::abs(up.bodies[0].vx)<1e-6 && std::abs(up.bodies[0].vy-expected)<1e-5
          && up.bodies[0].vz==0,
          "a score onset with zero impulse direction kicks along +Y by gain times normalized velocity");

    // A slanted impulse direction (0.6, 0.8) sets the direction of the same kick.
    auto slantBase=base; slantBase.impulseX=0.6f; slantBase.impulseY=0.8f;
    auto slantKick=slantBase; slantKick.scoreImpulseGain=kickGain;
    const auto plain=sample(slantBase,sampleSeconds), slanted=sample(slantKick,sampleSeconds);
    const double dvx=slanted.bodies[0].vx-plain.bodies[0].vx, dvy=slanted.bodies[0].vy-plain.bodies[0].vy;
    check(plain.count==1 && slanted.count==1 && std::abs(dvx-0.6*expected)<1e-5
          && std::abs(dvy-0.8*expected)<1e-5 && slanted.bodies[0].vz==0,
          "a score kick follows the unit impulse X and Y direction");

    // Seeks to other times and back return the same bodies at the same time.
    const auto direct=sample(slantKick,sampleSeconds);
    sample(slantKick,0.9); sample(slantKick,2.2); sample(slantKick,1.2);
    check(sameSolidBodies(direct,sample(slantKick,sampleSeconds)),
          "a seek and replay of the same score onset reproduces identical bodies");

    // The initial impulse is velocity impulseY*velocity/mass. With no forces,
    // each 120 Hz step adds vy*dt, so y advances by vy over the half second.
    auto spawnBase=base; spawnBase.impulseY=0.5f;
    const auto spawned=sample(spawnBase,sampleSeconds);
    const double spawnVy=0.5*velocity;
    check(spawned.count==1 && std::abs(spawned.bodies[0].vy-spawnVy)<1e-5
          && std::abs(spawned.bodies[0].y-(0.12+spawnVy*(sampleSeconds-onsetSeconds)))<1e-5,
          "initial impulse velocity integrates at fixed steps from the spawn time");
    // Births and kicks share one track rule. A note on another track adds no
    // body and no kick. A note on the spawn track still kicks both live bodies.
    const auto sameMotion=[](const ParticleSolidState& a,const ParticleSolidState& b)
    {
        if (a.count!=b.count) return false;
        for (int i=0;i<a.count;++i)
            if (std::abs(a.bodies[i].x-b.bodies[i].x)>1e-5 || std::abs(a.bodies[i].y-b.bodies[i].y)>1e-5
                || std::abs(a.bodies[i].vx-b.bodies[i].vx)>1e-5 || std::abs(a.bodies[i].vy-b.bodies[i].vy)>1e-5)
                return false;
        return true;
    };
    arbitmod::Note spawnNote=note; spawnNote.lengthBeats=6;
    arbitmod::Note otherNote=note; otherNote.id=7002; otherNote.trackId=3; otherNote.startBeat=6; otherNote.lengthBeats=2;
    auto onlySpawn=std::make_shared<arbitmod::Score>(); onlySpawn->scoreRevision=1; onlySpawn->notes={spawnNote};
    auto withOther=std::make_shared<arbitmod::Score>(*onlySpawn); withOther->notes={spawnNote,otherNote};
    auto onlySpawnHistory=std::make_shared<ParticleHistorySource>(*history); onlySpawnHistory->score=onlySpawn;
    auto withOtherHistory=std::make_shared<ParticleHistorySource>(*history); withOtherHistory->score=withOther;
    auto filtered=base; filtered.lifetime=5; filtered.scoreImpulseGain=kickGain;
    auto alone=filtered; alone.history=onlySpawnHistory;
    filtered.history=withOtherHistory;
    const auto aloneKick=sample(alone,3.5), filteredKick=sample(filtered,3.5);
    check(aloneKick.count==1 && sameMotion(aloneKick,filteredKick),
          "a score note on a track that births ignore adds neither a body nor a kick");
    auto otherOnSpawn=std::make_shared<arbitmod::Score>(*withOther); otherOnSpawn->notes[1].trackId=0;
    auto spawnTrackHistory=std::make_shared<ParticleHistorySource>(*history); spawnTrackHistory->score=otherOnSpawn;
    // Two births are allowed. Body one is born at beat 2 and takes both kicks;
    // body two is born at beat 6 and takes that kick at its own birth. Sum: 3 kicks.
    auto spawnTrackKick=filtered; spawnTrackKick.history=spawnTrackHistory; spawnTrackKick.count=2;
    const auto kickedTwice=sample(spawnTrackKick,3.5);
    check(kickedTwice.count==2 && std::abs(kickedTwice.bodies[0].vy+kickedTwice.bodies[1].vy-3*expected)<1e-4,
          "a score note on the spawn track is born and kicks every live body at its onset");

    // A note exactly at reset kicks the body it births. A window one tick earlier,
    // with the note one tick in, ends at the same absolute tick and must match.
    // resetTime 1 s is exact in float, and the one-tick shift is on the clip time.
    const double oneTick=1.0/28224000.0;
    ParticleParams atReset=base; atReset.resetMode=1; atReset.resetTime=1.0f; atReset.lifetime=2;
    atReset.historyProjectSeconds=1.5; atReset.scoreImpulseGain=kickGain;
    ParticleSolidState resetBodies, earlierBodies, unkickedBodies;
    replayParticleBodies(atReset,1.5,1,nullptr,&error,&resetBodies);
    replayParticleBodies(atReset,1.5+oneTick,1,nullptr,&error,&earlierBodies);
    auto unkicked=atReset; unkicked.scoreImpulseGain=0;
    replayParticleBodies(unkicked,1.5,1,nullptr,&error,&unkickedBodies);
    check(resetBodies.count==1 && std::abs(resetBodies.bodies[0].vy-expected)<1e-5
          && unkickedBodies.count==1 && std::abs(unkickedBodies.bodies[0].vy)<1e-6,
          "a score note exactly at reset kicks the body it births");
    check(earlierBodies.count==1 && sameMotion(resetBodies,earlierBodies),
          "a score note at reset matches a window that started one tick earlier");

    // Planar replay ignores impulseZ, so a nonzero Z does not change the kick,
    // and the kick follows the unit XY direction. Admission rejects planar
    // impulseZ, so this exercises the replay helper directly.
    auto planarBase=base; planarBase.simulationSpace=0; planarBase.impulseX=0.6f; planarBase.impulseY=0.8f;
    const auto planarPlain=sample(planarBase,sampleSeconds);
    auto planarKickBase=planarBase; planarKickBase.scoreImpulseGain=kickGain;
    const auto planarKick=sample(planarKickBase,sampleSeconds);
    auto planarZ=planarKickBase; planarZ.impulseZ=0.5f;
    const auto planarIgnoresZ=sample(planarZ,sampleSeconds);
    const double planarDvx=planarKick.bodies[0].vx-planarPlain.bodies[0].vx;
    const double planarDvy=planarKick.bodies[0].vy-planarPlain.bodies[0].vy;
    check(planarPlain.count==1 && planarKick.count==1 && std::abs(planarDvx-0.6*expected)<1e-5
          && std::abs(planarDvy-0.8*expected)<1e-5 && planarKick.bodies[0].vz==0,
          "a planar score kick follows the unit XY direction with no Z component");
    check(sameSolidBodies(planarIgnoresZ,planarKick),
          "planar replay ignores a nonzero impulseZ in its score kick");

    // The length is a hypot, not a squared sum. A 1e-25 X impulse squares to
    // zero in float, so the old sum fell back to +Y. The kick must go along +X.
    auto tinyDirection=planarKickBase; tinyDirection.impulseX=1e-25f; tinyDirection.impulseY=0;
    const auto tinyKick=sample(tinyDirection,sampleSeconds);
    check(tinyKick.count==1 && std::abs(tinyKick.bodies[0].vx-expected)<1e-5 && std::abs(tinyKick.bodies[0].vy)<1e-5,
          "a tiny impulse direction still kicks along its own unit direction");

    // Continuous 120 Hz playback to a frame just after the kick at 1 s, an
    // interior boundary of the window, gives the same bodies as a direct seek.
    ParticleSolidState continuous;
    for (int step=0; step<=121; ++step) continuous=sample(kicked,step/120.0);
    const auto seekAfterKick=sample(kicked,121/120.0);
    check(continuous.count==1 && sameSolidBodies(continuous,seekAfterKick)
          && std::abs(seekAfterKick.bodies[0].vy-expected)<1e-5,
          "continuous playback past an interior score kick matches a direct seek");

    // A window whose onsets alone cross the shared cap names score note impulses
    // as the cause. Without kicks the same window fits, so the cap is unchanged.
    auto crowd=std::make_shared<arbitmod::Score>(); crowd->scoreRevision=1;
    for (int i=0;i<520;++i)
    {
        arbitmod::Note crowded=note; crowded.id=9000+i; crowded.startBeat=0.01f+0.03f*float(i); crowded.lengthBeats=0.01f;
        crowd->notes.push_back(crowded);
    }
    auto crowdHistory=std::make_shared<ParticleHistorySource>(*history); crowdHistory->score=crowd;
    ParticleParams packed=base; packed.history=crowdHistory; packed.lifetime=10; packed.geometryCount=1;
    packed.historyProjectSeconds=9.0;
    auto packedKick=packed; packedKick.scoreImpulseGain=kickGain;
    const auto capWindow=particleReplayWindow(packedKick,9.0);
    const auto noop=[](std::int64_t,double) {};
    check(visitParticleHistoryEvents(packed,capWindow,noop,0.0) && !visitParticleHistoryEvents(packedKick,capWindow,noop,kickGain),
          "520 score onsets exceed the shared 512-event cap only with a positive impulse gain");
    const auto kickCap=particleEventCapDiagnostic(packedKick,capWindow,"generic cap");
    check(kickCap.find("Score note impulse")!=std::string::npos && kickCap.find("512")!=std::string::npos,
          "the cap refusal names score note impulses as its cause");
    check(particleEventCapDiagnostic(packed,capWindow,"generic cap")=="generic cap",
          "the cap refusal without impulses keeps its generic diagnostic");
    check(error.empty(), "score impulse replays complete without diagnostics");
}

// Fixed-step oracle for the coupled spring on the canonical link frame. Planar
// bodies start at rest with no impulse, attraction or gravity, so the link spring
// is the only force. Unit ratio weight comes from linkRatioInfluence zero. The
// replay takes 60 steps of 1/120 s, and the oracle repeats that integration.
void checkParticleLinkSpringOracle()
{
    const auto frame=canonicalLinkFrame(false,3);
    check(frame!=nullptr,"canonical signed link fixture admits for the spring oracle");
    if (!frame) return;
    videorender::ParticleParams p;
    p.motionMode=2; p.count=2; p.lifetime=4; p.force=0;
    p.linkSpring=3; p.linkRestScale=0.5f; p.linkRatioInfluence=0;
    const auto start=videorender::replayParticleBodies(p,0,1,frame.get());
    const auto spring=videorender::replayParticleBodies(p,0.5,1,frame.get());
    double x[2][2]={{start[0][0],start[0][1]},{start[1][0],start[1][1]}};
    double v[2][2]={{0,0},{0,0}};
    const double rest=0.5*std::hypot(x[1][0]-x[0][0],x[1][1]-x[0][1]);
    const double dt=1.0/120;
    for (int step=0;step<60;++step)
    {
        const double dx=x[1][0]-x[0][0], dy=x[1][1]-x[0][1], d=std::hypot(dx,dy);
        const double force=std::clamp(3.0*(d-rest),-100.0,100.0)/d;
        const double a[2][2]={{dx*force,dy*force},{-dx*force,-dy*force}};
        for (int body=0;body<2;++body)
            for (int axis=0;axis<2;++axis) v[body][axis]+=a[body][axis]*dt;
        for (int body=0;body<2;++body)
            for (int axis=0;axis<2;++axis) x[body][axis]+=v[body][axis]*dt;
    }
    const double moved=std::hypot(spring[0][0]-start[0][0],spring[0][1]-start[0][1]);
    check(moved>1e-3 && std::abs(spring[0][0]-x[0][0])<1e-5 && std::abs(spring[0][1]-x[0][1])<1e-5
          && std::abs(spring[1][0]-x[1][0])<1e-5 && std::abs(spring[1][1]-x[1][1])<1e-5,
          "canonical link spring matches a fixed-step oracle of its force law over 60 steps");
}

int main(int argc, char** argv)
{
    if (argc == 2 && std::string(argv[1]) == "--rotating-plane-response-prefix")
    { checkRotatingPlaneResponsePrerequisite(); return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE; }
    if (argc == 2 && std::string(argv[1]) == "--rotating-plane-geometric-prefix")
    { checkRotatingPlaneGeometricPrerequisite(); return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE; }
    if (argc == 2 && std::string(argv[1]) == "--rotating-plane-sweep-caller")
    { checkRotatingPlaneSweepCaller(); return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE; }
    if (argc == 2 && std::string(argv[1]) == "--imported-material-frame-dag")
    { checkMaterialFrameResolverBounds(); checkImportedMaterialFrameDag(); return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE; }
    checkMaterialFrameResolverBounds();
    checkImportedMaterialFrameDag();
    { std::string bodyError; const bool passed=verifyParticleBodyGeometryAdmissionOracles(bodyError); check(passed,bodyError.c_str()); }
    { std::string bodyError; const bool passed=verifyParticleBodyGeometrySampleOracles(bodyError); check(passed,bodyError.c_str()); }
    checkCoupledBodyReplay();
    checkParticleScoreImpulse();
    checkParticleLinkSpringOracle();
    checkHistoricalBodyReplay();
    checkNotesRequiredForSolidBodies();
    { std::string pointError; const bool passed=verifyParticlePointsOracles(pointError); check(passed,pointError.c_str()); }
    checkParticlePointsCompiledProvider();
    checkVisualParameterHistory();
    checkTrackGroupSimulationHistory();
    checkTrackGroupBandSimulationHistory();
    checkRotatingExactScalarPrerequisite();
    checkRotatingPlaneGeometricPrerequisite();
    checkRotatingPlaneResponsePrerequisite();
    checkSolidBodyReplay();
    checkSolidBodyInstancingBound();
    checkSolidBodyMotionPriors();
    checkSolidBodyMotionProduction();
    checkDynamicBodyProductionJoin();
    checkGraphAuthoredTimelineBodyProtocol();
    checkTriangleSurfaceContacts();
    checkBoxTriangleSurfaceContacts();
    checkRotatingPlaneSweepCaller();
    checkConnectedWaterBodies();
    checkWaterCoupledResetGuardAndReplayIdentity();
    checkAnimatedGeometryBodies();
    checkReactiveParticleGeometryHistory();
    std::string error;
    error.reserve(256);
    videowire::CompiledVisualLayerPlan multipleGeometry;
    multipleGeometry.structuralRevision = 1;
    multipleGeometry.operations.push_back({1, "geometry.core.runtime", "native-gpu", ""});
    multipleGeometry.operations.push_back({2, "geometry.core.runtime", "native-gpu", ""});
    FakeLayer rejectedGeometryLayer;
    const videowire::VisualPlanEvaluationContext geometryContext {
        11, false, visualtemporalsampling::EvaluationMode::preview, 12, 13
    };
    const auto beforeMultiGeometry = allocations.load(std::memory_order_relaxed);
    check(!videowire::admitGeometryCoreOperations(
              multipleGeometry, videohelper::geometry::PlanUse::preview,
              &geometryContext, 64, 64, rejectedGeometryLayer, error)
              && allocations.load(std::memory_order_relaxed) == beforeMultiGeometry
              && error == "multiple Geometry Core image outputs require an explicit compositor",
          "multiple Geometry Core outputs reject before decode or native allocation");
    {
        videowire::CompiledVisualLayerPlan harmonic;
        harmonic.clipId = 17;
        harmonic.structuralRevision = 2;
        harmonic.nodeKinds = {visualharmonicgeometry::kOperationKind};
        harmonic.nodeIds = {71};
        harmonic.ports = {{71, 1, 1, "out", "frame", "image", "rgba8", "sRGB"}};
        visualharmonicgeometry::Mapping mapping;
        mapping.stableId = 72;
        harmonic.operations = {{71, visualharmonicgeometry::kOperationKind, "native-gpu",
                                 visualharmonicgeometry::encode(mapping)}};
        videowire::VisualLayerExecution execution;
        check(videowire::compileVisualLayerExecution(harmonic, execution, error),
              "harmonic ribbon operation must be admitted before native execution");
        FakeLayer emptyScore;
        check(videowire::admitGeometryCoreOperations(harmonic,
                  videohelper::geometry::PlanUse::preview, &geometryContext, 64, 64, emptyScore, error)
                  && emptyScore.opacity == 0 && !emptyScore.nativeTextureOwner,
              "an empty score produces a transparent layer without GPU allocation");
        harmonic.operations[0].payloadXml += "trailing";
        check(!videowire::compileVisualLayerExecution(harmonic, execution, error),
              "harmonic ribbon malformed payload must fail admission");
        harmonic.operations[0].payloadXml = visualharmonicgeometry::encode(mapping);
        harmonic.ports[0].dataType = "depth";
        check(!videowire::compileVisualLayerExecution(harmonic, execution, error),
              "harmonic ribbon output cannot be relabeled as Depth");
    }
    {
        using namespace videowire::geometry;
        PortContract contract;
        contract.carrier=CarrierKind::geometry3D;
        contract.maxVertices=4; contract.maxIndices=6;
        auto mesh=lowerGrid(901,901,1,2,2,1,contract,error);
        check(mesh.has_value(),"spectrum ordered-plan fixture lowers");
        if (mesh)
        {
            spectrum::Binding binding;
            binding.fieldStableId=902; binding.sourceGeometryStableId=901;
            binding.consumerGeometryStableId=903;
            binding.coordinates={0.0f,0.25f,0.75f,1.0f};
            mesh->spectrumFields.push_back(binding);
            mesh->operations.push_back({902,OperationCode::evaluateField,902,902,{}});
            mesh->operations.push_back({903,OperationCode::transform,901,903,{}});
            mesh->stableId=903; mesh->dispatchCount=3;
            auto admitted=admitValue(*mesh,contract,error);
            check(admitted.has_value(),"spectrum ordered-plan value admits");
            if (admitted)
            {
                videowire::CompiledVisualLayerPlan plan;
                plan.clipId=18; plan.structuralRevision=3;
                plan.nodeKinds={"geometry.core.runtime"}; plan.nodeIds={71};
                plan.ports={{71,1,1,"out","frame","image","rgba8","sRGB"}};
                plan.operations={{71,"geometry.core.runtime","native-gpu",
                    encodeLoweredPlanText(lowerRuntimePlan(contract,*admitted))}};
                videowire::VisualLayerExecution execution;
                check(videowire::compileVisualLayerExecution(plan,execution,error),
                      "spectrum geometry is admitted by the actual ordered production compiler");
                check(videowire::isGeometryCoreRenderPlan({plan},18),
                      "geometry source bypasses decoded video in preview and export");
                plan.ports[0].dataType="depth";
                check(!videowire::compileVisualLayerExecution(plan,execution,error),
                      "geometry image output cannot claim a different AOV");
                plan.ports[0].dataType="image";
                plan.operations[0].payloadXml+="00";
                check(!videowire::compileVisualLayerExecution(plan,execution,error),
                      "malformed spectrum transport fails before native allocation");
            }
        }
    }
    {
        using namespace videowire::geometry;
        PortContract meshContract;
        meshContract.carrier=CarrierKind::geometry3D;
        meshContract.maxVertices=4; meshContract.maxIndices=6;
        PortContract contract;
        contract.carrier=CarrierKind::instances3D; contract.maxInstances=64;
        auto mesh=lowerGrid(911,911,1,2,2,1,meshContract,error);
        spectrum::Binding binding;
        binding.firstBand=4; binding.lastBand=15;
        auto instances=mesh ? lowerSpectrumInstancer(*mesh,912,binding,{1,3,3,4},contract,error) : std::nullopt;
        auto admitted=instances ? admitValue(*instances,contract,error) : std::nullopt;
        check(admitted.has_value(),"Spectrum Instancer production-plan fixture admits");
        if (admitted)
        {
            videowire::CompiledVisualLayerPlan plan;
            plan.clipId=19; plan.structuralRevision=3;
            plan.nodeKinds={"geometry.core.runtime"}; plan.nodeIds={72};
            plan.ports={{72,1,1,"out","frame","image","rgba8","sRGB"}};
            plan.operations={{72,"geometry.core.runtime","native-gpu",
                encodeLoweredPlanText(lowerRuntimePlan(contract,*admitted))}};
            videowire::VisualLayerExecution execution;
            check(videowire::compileVisualLayerExecution(plan,execution,error),
                  "Spectrum Instancer enters the production ordered native render schedule");
            check(videowire::isGeometryCoreRenderPlan({plan},19),
                  "Spectrum Instancer preview and export bypass decoded-media and dummy-frame paths");
        }
    }
    {
        using namespace videowire::geometry;
        PortContract contract;
        contract.carrier=CarrierKind::geometry3D; contract.maxVertices=4; contract.maxIndices=6;
        auto mesh=lowerGrid(961,961,1,2,2,1,contract,error);
        check(mesh.has_value(),"Score Field production fixture lowers");
        if (mesh) {
            scorefield::Binding binding;
            binding.fieldStableId=962; binding.sourceGeometryStableId=961;
            binding.consumerGeometryStableId=963; binding.scoreSourceStableId=964;
            binding.elementIds=std::get<GeometryData>(mesh->data).vertexIds;
            for (const auto& p : std::get<GeometryData>(mesh->data).positions)
                binding.positions.push_back({p.x,p.y,p.z});
            mesh->scoreFields.push_back(binding);
            mesh->operations.push_back({962,OperationCode::evaluateField,962,962,{}});
            mesh->operations.push_back({963,OperationCode::transformTRS,961,963,{}});
            mesh->stableId=963; mesh->dispatchCount=3;
            const auto admitted=admitValue(*mesh,contract,error);
            check(admitted.has_value(),"Score Field typed transport admits");
            if (admitted) {
                videowire::CompiledVisualLayerPlan plan;
                plan.clipId=20; plan.structuralRevision=4;
                plan.nodeKinds={"geometry.core.runtime"}; plan.nodeIds={73};
                plan.ports={{73,1,1,"out","frame","image","rgba8","sRGB"}};
                plan.operations={{73,"geometry.core.runtime","native-gpu",
                    encodeLoweredPlanText(lowerRuntimePlan(contract,*admitted))}};
                videowire::VisualLayerExecution execution;
                check(videowire::compileVisualLayerExecution(plan,execution,error),
                      "Score Field enters the production native render schedule");
                check(videowire::isGeometryCoreRenderPlan({plan},20),
                      "Score Field preview/export requires no dummy video source");
            }
        }
    }
    videowire::VisualEventScheduleBinding eventSchedule;
    eventSchedule.clipId = 7;
    eventSchedule.nodeId = 70;
    eventSchedule.portId = 0;
    eventSchedule.sessionRevision = 3;
    eventSchedule.coverageVersion = 1; eventSchedule.coverageComplete = true;
    eventSchedule.planGeneration = 9; eventSchedule.transportEpoch = 4; eventSchedule.streamId = 1;
    eventSchedule.routeId = 1; eventSchedule.sourceNodeId = 100; eventSchedule.sourcePortId = 0;
    eventSchedule.sampleRate = 64; eventSchedule.bpm = 120;
    eventSchedule.coverageStartSample = 0; eventSchedule.coverageEndSample = 128;
    eventSchedule.coverageStartBeat = 0; eventSchedule.coverageEndBeat = 4;
    eventSchedule.triggers = {
        { 0, 1.0, 0.25f, 1, 32 }, { 32, 1.5, 0.75f, 2, 48 }, { 64, 1.5, 0.5f, 3, 48 }
    };
    const std::vector<videowire::VisualEventScheduleBinding> eventSchedules { eventSchedule };
    videowire::VisualEventTriggerCursor viewportCursor (false);
    check (viewportCursor.consume(eventSchedules, 7, 70, 0, 3, 1.25).count == 0,
           "viewport cursor attaches without replaying historical triggers");
    const auto viewportAdvance = viewportCursor.consume(eventSchedules, 7, 70, 0, 3, 1.5);
    check (viewportAdvance.count == 2 && viewportAdvance.strongest == 0.75f
           && viewportCursor.consume(eventSchedules, 7, 70, 0, 3, 1.5).count == 0,
           "viewport cursor consumes equal-beat triggers once in deterministic order");
    check (viewportCursor.consume(eventSchedules, 7, 70, 0, 3, 1.0).count == 0,
           "viewport backward seek resets without bursting retained history");
    check (viewportCursor.consume(eventSchedules, 7, 70, 0, 4, 2.0).count == 0,
           "viewport revision replacement resets independently");
    videowire::VisualEventTriggerCursor exportCursor (true);
    check (exportCursor.consume(eventSchedules, 7, 70, 0, 3, 1.0).count == 1
           && exportCursor.consume(eventSchedules, 7, 70, 0, 3, 1.5).count == 2,
           "export cursor deterministically replays schedules from timeline start");
    check (exportCursor.consume(eventSchedules, 7, 70, 1, 3, 2.0).count == 0
           && exportCursor.consume(eventSchedules, 8, 70, 0, 3, 2.0).count == 0,
           "cursor never retargets a stable sink by presentation identity");
    check (exportCursor.consume(eventSchedules, 7, 70, 0, 3, 1.0).count == 1,
           "export backward seek reconstructs its own trigger cursor");
    const auto colorAov = colorAovPlan();
    const std::vector<videowire::CompiledVisualLayerPlan> colorAovPlans { colorAov };
    FakeLayer viewportColorLayer;
    FakeLayer exportColorLayer;
    FakeProductRenderer viewportRenderer;
    FakeProductRenderer exportRenderer;
    check (videowire::executeVisualLayerPlanForRenderer (
               viewportRenderer, colorAovPlans, colorAov.clipId, viewportColorLayer, error, videohelper::geometry::PlanUse::preview, nullptr)
           && videowire::executeVisualLayerPlanForRenderer (
               exportRenderer, colorAovPlans, colorAov.clipId, exportColorLayer, error, videohelper::geometry::PlanUse::preview, nullptr)
           && viewportRenderer.colorAovExecutions == 1
           && viewportRenderer.bridgePreparations == 1
           && exportRenderer.colorAovExecutions == 1
           && exportRenderer.bridgePreparations == 1
           && viewportRenderer.publishedColorAov.extent == renderpassoutput::Extent { 640, 360 }
           && exportRenderer.publishedColorAov.extent == viewportRenderer.publishedColorAov.extent
           && viewportRenderer.publishedColorAov.attachments.size() == 1
           && exportRenderer.publishedColorAov.attachments.size() == 1
           && viewportRenderer.publishedColorAov.attachments.front().output
                == renderpassoutput::Output::Color
           && exportRenderer.publishedColorAov.attachments.front().output
                == viewportRenderer.publishedColorAov.attachments.front().output
           && exportRenderer.publishedColorAov.attachments.front().format
                == viewportRenderer.publishedColorAov.attachments.front().format
           && exportRenderer.publishedColorAov.attachments.front().colorSpace
                == viewportRenderer.publishedColorAov.attachments.front().colorSpace,
           "viewport and export execute and publish the same typed Color AOV description");
    FakeLayer unsupportedColorLayer;
    FakeProductRenderer unsupportedRenderer;
    unsupportedRenderer.rejectColorAov = true;
    error.clear();
    check (! videowire::executeVisualLayerPlanForRenderer (
               unsupportedRenderer, colorAovPlans, colorAov.clipId, unsupportedColorLayer, error, videohelper::geometry::PlanUse::preview, nullptr)
           && unsupportedRenderer.colorAovExecutions == 1
           && unsupportedRenderer.bridgePreparations == 0
           && error == "native Color AOV backend is unsupported",
           "product execution fails closed when native Color AOV submission is unsupported");

    const auto motionAov = motionAovPlan();
    const std::vector<videowire::CompiledVisualLayerPlan> motionAovPlans { motionAov };
    for (const auto use : { videohelper::geometry::PlanUse::preview,
                            videohelper::geometry::PlanUse::exportRender })
    {
        FakeLayer unboundMotionLayer;
        unboundMotionLayer.texture = 73;
        FakeProductRenderer unboundMotionRenderer;
        error.clear();
        check (! videowire::executeVisualLayerPlanForRenderer (
                   unboundMotionRenderer, motionAovPlans, motionAov.clipId,
                   unboundMotionLayer, error, use, nullptr)
               && error == "Render Passes Motion has no two-frame optical-flow source"
               && unboundMotionRenderer.colorAovExecutions == 0
               && unboundMotionRenderer.motionBlurPreparations == 0
               && unboundMotionRenderer.bridgePreparations == 0
               && unboundMotionLayer.texture == 73
               && ! unboundMotionLayer.graphMotionBlurActive,
               "preview and export reject extent-only Motion before compositor publication");
    }
    auto retainedColor = colorAov;
    retainedColor.clipId = motionAov.clipId;
    retainedColor.structuralRevision = motionAov.structuralRevision - 1;
    auto motionAdmission = std::make_unique<videowire::VisualPlanExecutionState>();
    check (motionAdmission->admitPlans ({ retainedColor }, &error, 640, 360),
           "a prior Color plan admits before the unsupported Motion replacement");
    const auto retainedReceipt = motionAdmission->resourceReceipt();
    error.clear();
    check (! motionAdmission->admitPlans (motionAovPlans, &error, 640, 360)
           && error == "Render Passes Motion has no two-frame optical-flow source"
           && motionAdmission->compiled (retainedColor.clipId,
                                         retainedColor.structuralRevision) != nullptr
           && motionAdmission->compiled (motionAov.clipId,
                                         motionAov.structuralRevision) == nullptr
           && motionAdmission->resourceReceipt() == retainedReceipt,
           "unbound optical-flow admission retains the previous plan and resource receipt");
    auto direct = directPlan();
    FakeLayer layer;
    videowire::VisualLayerExecution execution;

    const auto importedAnimation = importedAnimationPlan();
    check (videowire::compileVisualLayerExecution(importedAnimation, execution, error)
           && execution.importedAnimation.has_value()
           && execution.importedAnimation->asset.id == "model-asset-1"
           && execution.importedAnimation->asset.sourceByteSize == 4096,
           "exact imported-animation schedule lowers its path-free asset identity");

    auto missingImportedSource = importedAnimation;
    missingImportedSource.nodeKinds.erase(missingImportedSource.nodeKinds.begin());
    missingImportedSource.nodeIds.erase(missingImportedSource.nodeIds.begin());
    missingImportedSource.operations.erase(missingImportedSource.operations.begin());
    error.clear();
    check (! videowire::compileVisualLayerExecution(
                missingImportedSource, execution, error)
           && error == "imported animation requires one exact source-to-deformation typed schedule",
           "imported-animation lowering rejects a deformation without dereferencing a missing source");

    auto missingImportedDeformation = importedAnimation;
    missingImportedDeformation.nodeKinds.pop_back();
    missingImportedDeformation.nodeIds.pop_back();
    missingImportedDeformation.operations.pop_back();
    error.clear();
    check (! videowire::compileVisualLayerExecution(
                missingImportedDeformation, execution, error)
           && error == "imported animation requires one exact source-to-deformation typed schedule",
           "imported-animation lowering rejects a source without dereferencing a missing deformation");

    FakeProductRenderer missingAnimationRenderer;
    FakeLayer missingAnimationLayer;
    error.clear();
    check (! videowire::executeVisualLayerPlanForRenderer(
                missingAnimationRenderer, { importedAnimation }, 7,
                missingAnimationLayer, error, videohelper::geometry::PlanUse::preview, nullptr)
           && missingAnimationRenderer.bridgePreparations == 0
           && error == "imported animation plan requires a retained native compositor frame binding",
           "admitted imported animation fails closed until the product retains and binds its native frame");

    const auto keyCleanup = keyCleanupPlan();
    FakeLayer keyLayer;
    check (videowire::executeVisualLayerPlan({ keyCleanup }, 7, keyLayer, error, videohelper::geometry::PlanUse::preview)
           && keyLayer.effects == &keyLayer.graphKeyCleanupEffect
           && keyLayer.effectCount == 1 && keyLayer.graphKeyCleanupEffect.enabled
           && keyLayer.graphKeyCleanupEffect.type == 17
           && keyLayer.graphKeyCleanupEffect.params[0] == 0.1f
           && keyLayer.graphKeyCleanupEffect.params[1] == 0.9f
           && keyLayer.graphKeyCleanupEffect.params[2] == 0.2f
           && keyLayer.graphKeyCleanupEffect.params[3] == 0.25f
           && keyLayer.graphKeyCleanupEffect.params[4] == 0.15f
           && keyLayer.graphKeyCleanupEffect.params[5] == 0.75f,
           "typed key cleanup lowers to the existing bounded ChromaKey GPU pass");
    FakeLayer repeatedKeyLayer;
    check (videowire::executeVisualLayerPlan({ keyCleanup }, 7, repeatedKeyLayer, error, videohelper::geometry::PlanUse::preview)
           && repeatedKeyLayer.graphKeyCleanupEffect.type == keyLayer.graphKeyCleanupEffect.type
           && std::equal(std::begin(repeatedKeyLayer.graphKeyCleanupEffect.params),
                         std::end(repeatedKeyLayer.graphKeyCleanupEffect.params),
                         std::begin(keyLayer.graphKeyCleanupEffect.params)),
           "typed key cleanup lowering is deterministic");
    auto fullKeyCleanup = keyCleanup;
    fullKeyCleanup.operations[1].payloadXml.replace(
        fullKeyCleanup.operations[1].payloadXml.find("feather=\"0\""),
        std::string("feather=\"0\"").size(), "feather=\"1\"");
    check (videowire::executeVisualLayerPlan({ fullKeyCleanup }, 7, keyLayer, error, videohelper::geometry::PlanUse::preview)
           && keyLayer.graphKeyCleanupActive
           && keyLayer.graphKeyCleanupFeather == 1.0f,
           "authored key cleanup reaches the shared bounded native-GPU pass");
    auto malformedKeyCleanup = keyCleanup;
    malformedKeyCleanup.operations[1].payloadXml.replace(
        malformedKeyCleanup.operations[1].payloadXml.find("despill=\"0.75\""),
        std::string("despill=\"0.75\"").size(), "despill=\"nan\"");
    check (! videowire::executeVisualLayerPlan({ malformedKeyCleanup }, 7, keyLayer, error, videohelper::geometry::PlanUse::preview)
           && error == "visual.key.cleanup has a malformed or out-of-bounds immutable payload",
           "typed key cleanup rejects malformed non-finite payloads");
    auto mistypedKeyCleanup = keyCleanup;
    mistypedKeyCleanup.ports[2].dataType = "depth";
    check (! videowire::executeVisualLayerPlan({ mistypedKeyCleanup }, 7, keyLayer, error, videohelper::geometry::PlanUse::preview)
           && error == "visual.key.cleanup requires exact Frame<Image> input/output descriptors",
           "typed key cleanup rejects inexact Frame output descriptors");
    auto misclassifiedKeyCleanup = keyCleanup;
    misclassifiedKeyCleanup.operations[1].backendCapability = "control-eval";
    check (! videowire::executeVisualLayerPlan({ misclassifiedKeyCleanup }, 7, keyLayer, error, videohelper::geometry::PlanUse::preview),
           "typed key cleanup rejects a false backend capability classification");

    auto sharpen = keyCleanupPlan();
    sharpen.nodeKinds[1] = "visual.effect.sharpen";
    sharpen.operations[1].kind = "visual.effect.sharpen";
    sharpen.operations[1].payloadXml =
        "<CommonEffect schemaVersion=\"1\" effect=\"sharpen\" amount=\"2.5\"/>";
    FakeLayer sharpenLayer;
    check (videowire::executeVisualLayerPlan({ sharpen }, 7, sharpenLayer, error, videohelper::geometry::PlanUse::preview)
           && sharpenLayer.effects == &sharpenLayer.graphCommonEffect
           && sharpenLayer.effectCount == 1 && sharpenLayer.graphCommonEffect.enabled
           && sharpenLayer.graphCommonEffect.type == 7
           && sharpenLayer.graphCommonEffect.params[0] == 2.5f,
           "graph-native sharpen lowers to the shared renderer effect pass");
    auto unsupportedSharpen = sharpen;
    unsupportedSharpen.operations[1].payloadXml =
        "<CommonEffect schemaVersion=\"1\" effect=\"sharpen\" amount=\"2.5\" mix=\"0.5\"/>";
    check (! videowire::executeVisualLayerPlan({ unsupportedSharpen }, 7, sharpenLayer, error, videohelper::geometry::PlanUse::preview),
           "graph-native sharpen fails closed on unsupported controls");

    auto saturation = sharpen;
    saturation.nodeKinds[1] = "visual.color.saturation";
    saturation.operations[1].kind = "visual.color.saturation";
    saturation.operations[1].payloadXml =
        "<CommonEffect schemaVersion=\"1\" effect=\"saturation\" amount=\"1.75\"/>";
    FakeLayer saturationLayer;
    check (videowire::executeVisualLayerPlan({ saturation }, 7, saturationLayer, error, videohelper::geometry::PlanUse::preview)
           && saturationLayer.effects == &saturationLayer.graphCommonEffect
           && saturationLayer.effectCount == 1 && saturationLayer.graphCommonEffect.enabled
           && saturationLayer.graphCommonEffect.type == 2
           && saturationLayer.graphCommonEffect.params[0] == 1.75f,
           "graph-native saturation lowers to the shared renderer color rack");

    auto exposure = sharpen;
    exposure.nodeKinds[1] = "visual.color.exposure";
    exposure.operations[1].kind = "visual.color.exposure";
    exposure.operations[1].payloadXml =
        "<CommonEffect schemaVersion=\"1\" effect=\"exposure\" stops=\"-1.5\"/>";
    FakeLayer exposureLayer;
    check (videowire::executeVisualLayerPlan({ exposure }, 7, exposureLayer, error, videohelper::geometry::PlanUse::preview)
           && exposureLayer.effects == &exposureLayer.graphCommonEffect
           && exposureLayer.effectCount == 1 && exposureLayer.graphCommonEffect.enabled
           && exposureLayer.graphCommonEffect.type == 4
           && exposureLayer.graphCommonEffect.params[0] == -1.5f,
           "graph-native exposure lowers to the shared renderer color rack");

    struct CommonEffectCase
    {
        const char* graphKind;
        const char* payload;
        int rendererType;
        std::array<float, 2> parameters;
        std::size_t parameterCount;
    };
    const std::array<CommonEffectCase, 6> additionalCommonEffects {{
        { "visual.color.brightness",
          "<CommonEffect schemaVersion=\"1\" effect=\"brightness\" amount=\"-0.25\"/>",
          0, { -0.25f, 0.0f }, 1 },
        { "visual.color.contrast",
          "<CommonEffect schemaVersion=\"1\" effect=\"contrast\" amount=\"1.5\"/>",
          1, { 1.5f, 0.0f }, 1 },
        { "visual.color.hue",
          "<CommonEffect schemaVersion=\"1\" effect=\"hue\" shift=\"45\"/>",
          3, { 45.0f, 0.0f }, 1 },
        { "visual.color.gamma",
          "<CommonEffect schemaVersion=\"1\" effect=\"gamma\" value=\"2.2\"/>",
          5, { 2.2f, 0.0f }, 1 },
        { "visual.effect.blur",
          "<CommonEffect schemaVersion=\"1\" effect=\"blur\" radius=\"8.5\"/>",
          6, { 8.5f, 0.0f }, 1 },
        { "visual.effect.vignette",
          "<CommonEffect schemaVersion=\"1\" effect=\"vignette\" amount=\"0.7\" softness=\"0.25\"/>",
          8, { 0.7f, 0.25f }, 2 }
    }};
    for (const auto& test : additionalCommonEffects)
    {
        auto plan = sharpen;
        plan.nodeKinds[1] = test.graphKind;
        plan.operations[1].kind = test.graphKind;
        plan.operations[1].payloadXml = test.payload;
        FakeLayer layer;
        bool parametersMatch = true;
        const bool executed = videowire::executeVisualLayerPlan({ plan }, 7, layer, error, videohelper::geometry::PlanUse::preview);
        for (std::size_t index = 0; index < test.parameterCount; ++index)
            parametersMatch = parametersMatch
                && layer.graphCommonEffect.params[index] == test.parameters[index];
        const auto message = std::string("graph-native ") + test.graphKind
            + " lowers to the shared renderer effect rack";
        check (executed && layer.effects == &layer.graphCommonEffect
               && layer.effectCount == 1 && layer.graphCommonEffect.enabled
               && layer.graphCommonEffect.type == test.rendererType && parametersMatch,
               message.c_str());
    }

    auto incompleteVignette = sharpen;
    incompleteVignette.nodeKinds[1] = "visual.effect.vignette";
    incompleteVignette.operations[1].kind = "visual.effect.vignette";
    incompleteVignette.operations[1].payloadXml =
        "<CommonEffect schemaVersion=\"1\" effect=\"vignette\" amount=\"0.7\"/>";
    FakeLayer incompleteVignetteLayer;
    check (!videowire::executeVisualLayerPlan(
               { incompleteVignette }, 7, incompleteVignetteLayer, error, videohelper::geometry::PlanUse::preview),
           "graph-native vignette rejects an incomplete immutable payload");

    auto convertedGamma = sharpen;
    convertedGamma.nodeKinds[1] = "visual.color.gamma";
    convertedGamma.operations[1].kind = "visual.color.gamma";
    convertedGamma.operations[1].payloadXml =
        "<CommonEffect schemaVersion=\"1\" effect=\"gamma\" value=\"2.2\"/>";
    convertedGamma.ports[2].colorSpace = "linearSRGB";
    FakeLayer convertedGammaLayer;
    check (! videowire::executeVisualLayerPlan(
               { convertedGamma }, 7, convertedGammaLayer, error, videohelper::geometry::PlanUse::preview)
           && error == "visual.color.gamma requires exact Frame<Image> input/output descriptors",
           "graph-native color effects reject implicit typed-edge color conversion");

    auto mismatchedColorPayload = exposure;
    mismatchedColorPayload.operations[1].payloadXml =
        "<CommonEffect schemaVersion=\"1\" effect=\"saturation\" amount=\"1.5\"/>";
    check (!videowire::executeVisualLayerPlan(
               { mismatchedColorPayload }, 7, exposureLayer, error, videohelper::geometry::PlanUse::preview),
           "graph-native color node rejects a payload for another admitted kind");

    videowire::ExecutableCommonEffectPayload unavailablePayload;
    videowire::CommonEffectPayloadFailure unavailableFailure =
        videowire::CommonEffectPayloadFailure::none;
    check (!videowire::parseExecutableCommonEffectPayload(
               exposure.operations[1].payloadXml, unavailablePayload, &unavailableFailure,
               commoneffect::BackendSupport { true, true, false })
           && unavailableFailure
               == videowire::CommonEffectPayloadFailure::productionBackendUnavailable,
           "graph-native exposure reports truthful backend unavailability");

    auto particles = direct;
    particles.nodeKinds = { "visual.particles", "video.out" };
    particles.nodeIds = { 51, 12 };
    particles.edges = { { 51, 1, 12, 0 } };
    particles.operations = {
        { 51, "visual.particles", "native-gpu", "<NodeParams seed=\"70000\" count=\"9000\" lifetime=\"20\" size=\"0\" speed=\"9\" red=\"-1\" green=\"2\" blue=\"0.4\" alpha=\"2\"/>" },
        { 12, "video.out", "native-gpu", "" }
    };
    particles.ports = {
        { 51, 0, 1, "in", "event", "unspecified", "unspecified", "unspecified" },
        { 51, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 12, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    check (videowire::validateCompiledVisualLayerPlans({}, { particles }, false, error),
           "snapshot admission accepts the producer's descriptorless particle trigger port");
    auto forgedDescriptorlessPort = particles;
    forgedDescriptorlessPort.ports[0].nodeId = 12;
    check (! videowire::validateCompiledVisualLayerPlans({}, { forgedDescriptorlessPort }, false, error)
           && error == "visual layer plan has invalid typed port descriptors",
           "descriptorless event admission remains scoped to the exact particle trigger port");
    FakeLayer particleLayer;
    particleLayer.shaderClock.time = 1.25;
    particleLayer.shaderClock.frame = 30;
    particleLayer.shaderClock.timeDelta = 1.0 / 24.0;
    particleLayer.shaderClock.playing = false;
    check (videowire::executeVisualLayerPlan({ particles }, 7, particleLayer, error,videohelper::geometry::PlanUse::preview,
                                             nullptr, nullptr, nullptr, 2.5)
           && particleLayer.particleSource && ! particleLayer.shaderSource
           && particleLayer.genParams.empty()
           && particleLayer.particleParameters.seed == 65535
           && particleLayer.particleParameters.count == 4096
           && particleLayer.particleParameters.lifetime == 10.0f
           && particleLayer.particleParameters.size == 1.0f
           && particleLayer.particleParameters.force == 4.0f
           && particleLayer.particleParameters.red == 0.0f
           && particleLayer.particleParameters.green == 1.0f
           && particleLayer.shaderClock.time == 1.25
           && particleLayer.shaderClock.frame == 30
           && particleLayer.shaderClock.timeDelta == 1.0 / 24.0
           && ! particleLayer.shaderClock.playing,
           "curated particle source clamps payload and preserves the canonical ShaderClock");
    auto particleState = std::make_unique<videowire::VisualPlanExecutionState>();
    particleState->admitPlans({ particles });
    const std::vector<videowire::CompiledVisualLayerPlan> admittedParticles { particles };
    const auto particleMapSize = particleLayer.genParams.size();
    const auto particleAllocations = allocations.load(std::memory_order_relaxed);
    check (videowire::executeVisualLayerPlan(admittedParticles, 7, particleLayer, error,videohelper::geometry::PlanUse::preview,
                                             nullptr, nullptr, particleState.get(), 3.0)
           && particleLayer.genParams.size() == particleMapSize
           && allocations.load(std::memory_order_relaxed) == particleAllocations,
           "typed particle render carries admitted values without map allocation or insertion");
    auto reactiveParticles = particles;
    reactiveParticles.operations[0].payloadXml =
        "<NodeParams motionMode=\"1\" spawnTrack=\"9\" seed=\"71\" count=\"1024\" "
        "gravity=\"0.4\" drag=\"2\" attraction=\"3\" linkSpring=\"2.5\" linkRatioInfluence=\"0.75\" "
        "collisionMode=\"1\" bodyRadius=\"0.025\" restitution=\"0.85\" "
        "rmsGain=\"4\" onsetGain=\"5\" resetTime=\"2.5\"/>";
    FakeLayer reactiveLayer;
    check(videowire::isTypedParticlePlan({ reactiveParticles }, 7),
          "typed particles select a source without probing dummy media");
    check(videowire::executeVisualLayerPlan({ reactiveParticles }, 7, reactiveLayer, error,
              videohelper::geometry::PlanUse::preview),
          "timeline reactive particles execute through the production plan");
    reactiveLayer.audioPresent = true;
    reactiveLayer.audioFeatures = { 0.25f, 0.75f, 0.5f };
    const auto nativeParameters = videorender::particleParamsForLayer(reactiveLayer);
    check(nativeParameters.motionMode == 1 && nativeParameters.spawnTrack == 9
          && nativeParameters.seed == 71 && nativeParameters.count == 1024
          && nativeParameters.drag == 2.0f && nativeParameters.attraction == 3.0f
          && nativeParameters.linkSpring == 2.5f && nativeParameters.linkRatioInfluence == 0.75f
          && nativeParameters.collisionMode == 1 && nativeParameters.bodyRadius == 0.025f
          && nativeParameters.restitution == 0.85f
          && nativeParameters.rmsGain == 4.0f && nativeParameters.onsetGain == 5.0f
          && nativeParameters.resetTime == 2.5f && nativeParameters.rms == 0.25f
          && nativeParameters.onset == 0.75f && nativeParameters.onsetAge == 0.5f,
          "native particles consume graph settings and canonical Block B without generator map keys");
    FakeLayer exportParticles;
    check(videowire::executeVisualLayerPlan({ reactiveParticles }, 7, exportParticles, error,
              videohelper::geometry::PlanUse::exportRender)
          && exportParticles.particleParameters.linkSpring == nativeParameters.linkSpring
          && exportParticles.particleParameters.collisionMode == nativeParameters.collisionMode
          && exportParticles.particleParameters.bodyRadius == nativeParameters.bodyRadius
          && exportParticles.particleParameters.restitution == nativeParameters.restitution
          && exportParticles.particleParameters.linkRatioInfluence == nativeParameters.linkRatioInfluence,
          "offline export admits the same harmonic-link spring controls as preview");
    {
        using namespace videowire::geometry;
        PortContract contract; contract.carrier=CarrierKind::geometry3D;
        contract.maxVertices=256; contract.maxIndices=384;
        const auto grid=lowerGrid(18200,18200,1,9,9,.25f,contract,error);
        const auto water=grid ? lowerWaterSurfaceSimulation(*grid,18201,18202,0,0,0,0,contract,error,100):std::nullopt;
        const auto admitted=water ? admitValue(*water,contract,error):std::nullopt;
        check(admitted.has_value(),"shared caller coupling refusal fixture admits its exact GCRV15 mesh");
        if (admitted) {
            auto plan=particles;
            plan.operations[0].payloadXml="<NodeParams motionMode=\"2\" historicalReplay=\"1\" simulationSpace=\"1\" "
                "count=\"1\" lifetime=\"1\" resetMode=\"1\" colliderGeometryPlan=\""
                +encodeLoweredPlanText(lowerRuntimePlan(contract,*admitted))+"\"/>";
            FakeLayer rejected;
            rejected.particleHistory=std::make_shared<videorender::ParticleHistorySource>(waterreplayfixture::source());
            rejected.particleProjectSeconds=rejected.shaderClock.timeSec=1.0/120;
            rejected.canonicalBlockCFrame=waterreplayfixture::frame(*rejected.particleHistory,1.0/120);
            auto oldBodies=std::make_shared<const videorender::ParticleBodyState>();
            auto oldSolids=std::make_shared<const videorender::ParticleSolidState>();
            rejected.particleParameters.preparedBodies=oldBodies;
            rejected.particleParameters.preparedSolids=oldSolids;
            FakeProductRenderer renderer;
            auto refused=plan;
            std::vector<videowire::VisualEventScheduleBinding> triggers(1);
            triggers[0].clipId=7; triggers[0].nodeId=51; triggers[0].portId=0;
            check(!videowire::executeVisualLayerPlanForRenderer(renderer,{refused},7,rejected,error,
                videohelper::geometry::PlanUse::preview,nullptr,nullptr,nullptr,nullptr,1.0/120,
                &triggers,nullptr,&geometryContext)
                && error.find("Event-triggered")!=std::string::npos
                && rejected.particleParameters.preparedBodies==oldBodies
                && rejected.particleParameters.preparedSolids==oldSolids,
                "shared preview caller refuses Event input before publishing either coupled output");
            refused=plan;
            const auto marker=refused.operations[0].payloadXml.find("resetMode=\"1\"");
            refused.operations[0].payloadXml.replace(marker,13,"resetMode=\"0\"");
            check(!videowire::executeVisualLayerPlanForRenderer(renderer,{refused},7,rejected,error,
                videohelper::geometry::PlanUse::exportRender,nullptr,nullptr,nullptr,nullptr,1.0/120,
                nullptr,nullptr,&geometryContext)
                && error.find("project-zero")!=std::string::npos
                && rejected.particleParameters.preparedBodies==oldBodies
                && rejected.particleParameters.preparedSolids==oldSolids,
                "shared export caller refuses cyclic coupling before publishing either coupled output");
        }
    }
    auto coupledParticles = reactiveParticles;
    coupledParticles.operations[0].payloadXml =
        "<NodeParams motionMode=\"2\" count=\"64\" collisionMode=\"2\" bodyShape=\"1\" "
        "bodyMass=\"2\" friction=\"0.5\" linkConstraint=\"1\" linkSpring=\"3\" "
        "linkRestScale=\"0.75\" impulseX=\"-0.2\" impulseY=\"0.4\" resetMode=\"1\" historicalReplay=\"1\"/>";
    FakeLayer coupledPreview, coupledExport;
    auto coupledHistory=std::make_shared<videorender::ParticleHistorySource>();
    coupledHistory->score=std::make_shared<const arbitmod::Score>();
    coupledPreview.particleHistory=coupledExport.particleHistory=coupledHistory;
    FakeProductRenderer coupledPreviewRenderer,coupledExportRenderer;
    check(videowire::executeVisualLayerPlanForRenderer(coupledPreviewRenderer,{coupledParticles},7,coupledPreview,error,
              videohelper::geometry::PlanUse::preview,nullptr)
          && videowire::executeVisualLayerPlanForRenderer(coupledExportRenderer,{coupledParticles},7,coupledExport,error,
              videohelper::geometry::PlanUse::exportRender,nullptr),
          "coupled bodies admit through preview and export");
    const auto& cp = coupledPreview.particleParameters;
    const auto& ce = coupledExport.particleParameters;
    check(cp.motionMode == 2 && cp.count == 64 && cp.collisionMode == 2 && cp.bodyShape == 1
          && cp.bodyMass == 2 && cp.friction == 0.5f && cp.linkConstraint == 1
          && cp.linkRestScale == 0.75f && cp.impulseX == -0.2f && cp.impulseY == 0.4f
          && cp.resetMode == 1 && cp.historicalReplay && ce.historicalReplay
          && ce.motionMode == cp.motionMode && ce.bodyMass == cp.bodyMass
          && ce.linkConstraint == cp.linkConstraint && ce.resetMode == cp.resetMode,
          "physical authoring parameters retain exact values across native plan paths");
    for (const auto* xml : { "<NodeParams motionMode=\"2\" count=\"65\"/>",
                            "<NodeParams motionMode=\"2\" count=\"1.5\"/>",
                            "<NodeParams motionMode=\"2\" count=\"64\" bodyMass=\"0\"/>",
                            "<NodeParams motionMode=\"2\" count=\"64\" bodyShape=\"2\"/>",
                            "<NodeParams motionMode=\"2\" count=\"64\" friction=\"nan\"/>",
                            "<NodeParams motionMode=\"2\" count=\"64\" historicalReplay=\"0.5\"/>",
                            "<NodeParams motionMode=\"1\" historicalReplay=\"1\"/>",
                            "<NodeParams motionMode=\"1\" collisionMode=\"2\"/>",
                            "<NodeParams motionMode=\"1\" linkConstraint=\"1\"/>" })
    {
        videorender::ParticleParams rejected;
        check(!videowire::admitParticleParameters(xml, rejected, error),
              "unsupported coupled-body capacities and motion combinations fail closed");
    }
    {
        using namespace videowire::geometry;
        PortContract meshContract; meshContract.carrier=CarrierKind::geometry3D;
        meshContract.maxVertices=4; meshContract.maxIndices=6;
        auto mesh=lowerGrid(9700,9700,1,2,2,1,meshContract,error);
        PortContract pointsContract; pointsContract.carrier=CarrierKind::points3D; pointsContract.maxPoints=64;
        auto points=mesh ? lowerPointsFromVertices(*mesh,9701,pointsContract,error) : std::nullopt;
        auto admitted=points ? admitValue(*points,pointsContract,error) : std::nullopt;
        check(admitted.has_value(),"particle geometry fixture admits its exact point provenance");
        if (admitted) {
            const auto encoded=encodeLoweredPlanText(lowerRuntimePlan(pointsContract,*admitted));
            auto geometryParticles=particles;
            geometryParticles.operations[0].payloadXml="<NodeParams motionMode=\"1\" geometryScale=\"0.5\" emitterGeometryPlan=\""+encoded
                +"\" attractorGeometryPlan=\""+encoded+"\"/>";
            FakeLayer previewGeometry,exportGeometry;
            check(videowire::executeVisualLayerPlan({geometryParticles},7,previewGeometry,error,videohelper::geometry::PlanUse::preview)
                && videowire::executeVisualLayerPlan({geometryParticles},7,exportGeometry,error,videohelper::geometry::PlanUse::exportRender)
                && previewGeometry.particleParameters.geometryCount==4
                && previewGeometry.particleParameters.geometryAnchors==exportGeometry.particleParameters.geometryAnchors
                && previewGeometry.particleParameters.geometryAnchors[0][0]==0.25f,
                "typed geometry emitters and attractors use identical bounded projected positions for preview and export");
            geometryParticles.operations[0].payloadXml="<NodeParams motionMode=\"2\" count=\"64\" collisionMode=\"2\" emitterGeometryPlan=\""+encoded
                +"\" attractorGeometryPlan=\""+encoded+"\"/>";
            check(videowire::executeVisualLayerPlan({geometryParticles},7,previewGeometry,error,videohelper::geometry::PlanUse::preview)
                && videowire::executeVisualLayerPlan({geometryParticles},7,exportGeometry,error,videohelper::geometry::PlanUse::exportRender)
                && previewGeometry.particleParameters.motionMode==2
                && previewGeometry.particleParameters.geometryAnchors==exportGeometry.particleParameters.geometryAnchors,
                "coupled bodies reuse the typed Geometry Core emitter and attractor plans");
            const auto colliderMesh=admitValue(*mesh,meshContract,error);
            check(colliderMesh.has_value(),error.c_str());
            if (colliderMesh)
            {
                const auto collider=encodeLoweredPlanText(lowerRuntimePlan(meshContract,*colliderMesh));
                geometryParticles.operations[0].payloadXml="<NodeParams motionMode=\"2\" historicalReplay=\"1\" count=\"4\" "
                    "speed=\"0\" emitterGeometryPlan=\""+encoded+"\" colliderGeometryPlan=\""+collider+"\"/>";
                auto history=std::make_shared<videorender::ParticleHistorySource>(); history->score=std::make_shared<const arbitmod::Score>();
                previewGeometry.particleHistory=history; exportGeometry.particleHistory=history;
                previewGeometry.particleProjectSeconds=exportGeometry.particleProjectSeconds=0.5;
                previewGeometry.shaderClock.timeSec=exportGeometry.shaderClock.timeSec=0.5;
                FakeProductRenderer previewBackend,exportBackend;
                check(videowire::executeVisualLayerPlanForRenderer(previewBackend,{geometryParticles},7,previewGeometry,error,
                          videohelper::geometry::PlanUse::preview,nullptr)
                    && videowire::executeVisualLayerPlanForRenderer(exportBackend,{geometryParticles},7,exportGeometry,error,
                          videohelper::geometry::PlanUse::exportRender,nullptr)
                    && previewGeometry.particleParameters.preparedBodies && exportGeometry.particleParameters.preparedBodies
                    && *previewGeometry.particleParameters.preparedBodies==*exportGeometry.particleParameters.preparedBodies,
                    "preview and export prepare identical collision states before entering native compositor passes");
            }
            auto removed=previewGeometry.particleParameters;
            check(videowire::admitParticleParameters("<NodeParams motionMode=\"1\"/>",removed,error)
                && removed.geometryCount==0 && !removed.geometryBinding && !removed.preparedBodies,
                "removing geometry inputs clears previously admitted particle anchors");
            for (const auto& xml : {
                "<NodeParams motionMode=\"0\" emitterGeometryPlan=\""+encoded+"\"/>",
                "<NodeParams motionMode=\"1\" attractorGeometryPlan=\""+encoded+"\"/>",
                "<NodeParams motionMode=\"1\" geometryProjection=\"3\" emitterGeometryPlan=\""+encoded+"\"/>",
                std::string("<NodeParams motionMode=\"1\" emitterGeometryPlan=\"invalid\"/>")}) {
                videorender::ParticleParams rejected;
                check(!videowire::admitParticleParameters(xml,rejected,error),
                    "malformed particle geometry, missing emitter and unsupported projection fail closed");
            }
        }
    }
    for (const auto* payload : { "<NodeParams motionMode=\"3\"/>",
                                "<NodeParams drag=\"nan\"/>",
                                "<NodeParams attraction=\"11\"/>",
                                "<NodeParams collisionMode=\"1.5\"/>",
                                "<NodeParams bodyRadius=\"0\"/>",
                                "<NodeParams restitution=\"nan\"/>",
                                "<NodeParams linkSpring=\"11\"/>",
                                "<NodeParams linkRatioInfluence=\"nan\"/>",
                                "<NodeParams linkRatioInfluence=\"1.5\"/>",
                                "<NodeParams spawnTrack=\"1.5\"/>",
                                "<NodeParams resetTime=\"-1\"/>" })
    {
        auto invalidMotion = reactiveParticles;
        invalidMotion.operations[0].payloadXml = payload;
        videowire::VisualLayerExecution rejected;
        check(!videowire::compileVisualLayerExecution(invalidMotion, rejected, error)
              && error.find("particle motion controls") != std::string::npos,
              "invalid particle motion parameters fail with a bounded diagnostic");
    }
    for (const auto* payload : { "<NodeParams linkSpring=\"1\"/>",
                                "<NodeParams motionMode=\"1\" linkSpring=\"1\" count=\"4097\"/>" })
    {
        auto invalidSpring = reactiveParticles;
        invalidSpring.operations[0].payloadXml = payload;
        videowire::VisualLayerExecution rejected;
        check(!videowire::compileVisualLayerExecution(invalidSpring, rejected, error)
              && error.find("128") != std::string::npos
              && error.find("256") != std::string::npos,
              "unsupported harmonic spring motion and particle capacity report score bounds");
    }
    auto particleSchedule = eventSchedule;
    {
        auto invalidBodies = reactiveParticles;
        invalidBodies.operations[0].payloadXml = "<NodeParams collisionMode=\"1\"/>";
        videowire::VisualLayerExecution rejected;
        check(!videowire::compileVisualLayerExecution(invalidBodies, rejected, error)
            && error.find("imported mesh topology") != std::string::npos,
            "body collision mode requires absolute timeline replay and states its collider limit");
    }
    particleSchedule.nodeId = 51;
    const std::vector<videowire::VisualEventScheduleBinding> particleSchedules { particleSchedule };
    videowire::VisualEventTriggerCursor particleCursor (true);
    particleLayer.shaderClock.beat = 1.5;
    check (videowire::executeVisualLayerPlan(admittedParticles, 7, particleLayer, error,videohelper::geometry::PlanUse::preview,
                                             nullptr, nullptr, particleState.get(), 3.0,
                                             &particleSchedules, &particleCursor)
           && particleLayer.particleTriggerConnected
           && particleLayer.particleTriggerCount == 3
           && particleLayer.particleTriggerStrength == 0.75f,
           "typed particle execution consumes the exact Event sink at the canonical beat");
    particleLayer.shaderClock.beat = 1.5;
    check (videowire::executeVisualLayerPlan(admittedParticles, 7, particleLayer, error,videohelper::geometry::PlanUse::preview,
                                             nullptr, nullptr, particleState.get(), 3.0,
                                             &particleSchedules, &particleCursor)
           && particleLayer.particleTriggerConnected && particleLayer.particleTriggerCount == 0,
           "typed particle execution does not retrigger a held frame");
    particleSchedule.nodeId = 52;
    const std::vector<videowire::VisualEventScheduleBinding> staleSchedules { particleSchedule };
    check (videowire::executeVisualLayerPlan(admittedParticles, 7, particleLayer, error,videohelper::geometry::PlanUse::preview,
                                             nullptr, nullptr, particleState.get(), 3.0,
                                             &staleSchedules, &particleCursor)
           && ! particleLayer.particleTriggerConnected,
           "typed particle execution never retargets a stale visual NodeId");
    auto malformedParticles = particles;
    malformedParticles.edges[0].fromPort = 0;
    check (! videowire::executeVisualLayerPlan({ malformedParticles }, 7, particleLayer, error, videohelper::geometry::PlanUse::preview)
           && error == "visual.particles has unsupported production topology",
           "particle source preserves exact image port and fails closed on malformed topology");
    particles.operations[0].backendCapability = "control-eval";
    check (! videowire::executeVisualLayerPlan({ particles }, 7, particleLayer, error, videohelper::geometry::PlanUse::preview),
           "particle image source rejects non-native backend classification");

    auto particleComposite = direct;
    // Deliberately scramble document and operation order. The compiled schedule
    // must derive execution and Blend routing from stable edges and port IDs.
    particleComposite.nodeKinds = { "video.out", "video.blend", "video.mask.shape",
        "visual.draw.shape", "video.layer.source", "visual.particles",
        "video.effects", "visual.shape.rectangle", "video.transform" };
    particleComposite.nodeIds = { 12, 24, 23, 32, 26, 51, 22, 31, 21 };
    particleComposite.operations = {
        { 24, "video.blend", "native-gpu", "" },
        { 51, "visual.particles", "native-gpu",
          "<NodeParams seed=\"42\" count=\"1024\" lifetime=\"3\" size=\"6\" speed=\"2\" red=\"0.1\" green=\"0.2\" blue=\"0.3\" alpha=\"0.8\"/>" },
        { 12, "video.out", "native-gpu", "" },
        { 31, "visual.shape.rectangle", "control-eval",
          "<NodeParams centerX=\"0.2\" centerY=\"0.3\" width=\"0.4\" height=\"0.5\"/>" },
        { 22, "video.effects", "native-gpu", "" },
        { 26, "video.layer.source", "source-decode", "" },
        { 21, "video.transform", "native-gpu", "" },
        { 32, "visual.draw.shape", "native-gpu",
          "<NodeParams red=\"0.6\" green=\"0.7\" blue=\"0.8\" alpha=\"0.9\"/>" },
        { 23, "video.mask.shape", "native-gpu", "" }
    };
    particleComposite.edges = {
        { 24, 1, 12, 0 }, { 32, 1, 24, 2 }, { 51, 1, 21, 0 },
        { 26, 0, 24, 3 }, { 23, 1, 24, 0 }, { 21, 1, 22, 0 },
        { 31, 0, 32, 0 }, { 22, 1, 23, 0 }
    };
    particleComposite.ports = {
        { 51, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 21, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 21, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 22, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 22, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 23, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 23, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 31, 0, 1, "out", "control", "shape", "unspecified", "unspecified" },
        { 32, 0, 1, "in", "control", "shape", "unspecified", "unspecified" },
        { 32, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 26, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 24, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 24, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 24, 2, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 24, 3, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 12, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    check (videowire::validateCompiledVisualLayerPlans({}, { particleComposite }, false, error),
           "particle composite satisfies the complete typed snapshot contract");
    check (videowire::compileVisualLayerExecution(particleComposite, execution, error)
           && execution.particles && execution.particleNodeId == 51
           && execution.transform && execution.effects && execution.mask
           && execution.drawShape && execution.drawShapeNodeId == 32
           && execution.compositeInputCount == 3
           && execution.compositeSourceNodeIds[0] == 51
           && execution.compositeDestinationPorts[0] == 0
           && execution.compositeSourceNodeIds[1] == 31
           && execution.compositeDestinationPorts[1] == 2
           && execution.compositeSourceNodeIds[2] == 26
           && execution.compositeDestinationPorts[2] == 3
           && execution.dagSchedule.operations.size() == particleComposite.operations.size(),
           "particle root lowers with mask, shape and layer inputs in deterministic Blend-port order");
    FakeLayer viewportParticleComposite, exportParticleComposite;
    check (videowire::executeVisualLayerPlan({ particleComposite }, 7,
                                             viewportParticleComposite, error, videohelper::geometry::PlanUse::preview)
           && videowire::executeVisualLayerPlan({ particleComposite }, 7,
                                                exportParticleComposite, error, videohelper::geometry::PlanUse::preview)
           && viewportParticleComposite.particleSource
           && viewportParticleComposite.drawShape
           && viewportParticleComposite.maskType == 2
           && viewportParticleComposite.particleNodeId == exportParticleComposite.particleNodeId
           && viewportParticleComposite.drawShapeNodeId == exportParticleComposite.drawShapeNodeId
           && viewportParticleComposite.particleTriggerConnected
                == exportParticleComposite.particleTriggerConnected,
           "viewport/export shared execution preserves the generalized particle composite topology");
    auto wrongParticlePort = particleComposite;
    const auto particleEdge = std::find_if(wrongParticlePort.edges.begin(), wrongParticlePort.edges.end(),
        [](const auto& edge) { return edge.fromNodeId == 51; });
    particleEdge->fromPort = 0;
    check (! videowire::compileVisualLayerExecution(wrongParticlePort, execution, error)
           && error == "bounded visual DAG has an incompatible typed edge",
           "particle composite fails closed when its exact image output port is not routed");

    check (videowire::executeVisualLayerPlan ({ direct }, 7, layer, error, videohelper::geometry::PlanUse::preview),
           "ordered direct chain executes");
    check (layer.scale == 1.0f && layer.translateX == 0.0f
           && layer.effects == nullptr && layer.effectCount == 0
           && layer.lutTexture == 0 && layer.maskType == 0
           && layer.opacity == 0.4f && layer.blendMode == 3,
           "absent production operations are neutralized through LayerDesc");

    auto matte = mattePlan();
    FakeLayer matteLayer;
    matteLayer.matteTexture = 91;
    matteLayer.matteWidth = 640;
    matteLayer.matteHeight = 360;
    check (videowire::executeVisualLayerPlan({ matte }, 7, matteLayer, error, videohelper::geometry::PlanUse::preview)
           && matteLayer.matteApply && matteLayer.matteInvert
           && matteLayer.matteBlack == 0.0f && matteLayer.matteWhite == 1.0f
           && matteLayer.matteErodeDilate == 4.0f && matteLayer.matteFeather == 4.0f
           && matteLayer.matteChoke == -1.0f,
           "valid matte topology propagates bounded refinement parameters");
    auto combined = combinedMattePlan();
    matteLayer.matteTextureB = 92; matteLayer.matteWidthB = 640; matteLayer.matteHeightB = 360;
    check(videowire::executeVisualLayerPlan({combined}, 7, matteLayer, error, videohelper::geometry::PlanUse::preview)
           && matteLayer.matteCombineMode == 3,
           "two immutable matte receipts lower xor through retained GPU resources");
    auto reorderedCombined = combined;
    std::swap(reorderedCombined.nodeKinds[1], reorderedCombined.nodeKinds[2]);
    std::swap(reorderedCombined.nodeIds[1], reorderedCombined.nodeIds[2]);
    std::swap(reorderedCombined.operations[1], reorderedCombined.operations[2]);
    videowire::RenderSegment primaryMatte;
    primaryMatte.matteAssetId = "matte-1";
    videowire::RenderSegment secondaryMatte;
    check(videowire::visualPlanSecondaryMatte({reorderedCombined}, 7, primaryMatte, secondaryMatte)
          && secondaryMatte.matteAssetId == "matte-2",
          "secondary matte identity follows the exact Combine port across legal source order");
    auto reusedMatte = reusedMattePlan();
    FakeLayer reusedMatteLayer = matteLayer;
    reusedMatteLayer.matteTexture = reusedMatteLayer.matteTextureB = 91;
    check(videowire::validateCompiledVisualLayerPlans({}, {reusedMatte}, false, error)
          && videowire::visualPlanReusesPrimaryMatte({reusedMatte}, 7)
          && videowire::executeVisualLayerPlan({reusedMatte}, 7, reusedMatteLayer, error, videohelper::geometry::PlanUse::preview)
          && reusedMatteLayer.matteCombineMode == 3,
          "one exact typed matte output fans out to both ports and reuses one retained texture");
    videowire::VisualPlanResourceUsage reusedMatteUsage;
    check(videowire::accountVisualPlanResources(reusedMatte, 64, 64, reusedMatteUsage, error)
          && reusedMatteUsage.frameOutputs == 5
          && reusedMatteUsage.peakLiveFrames == 3
          && reusedMatteUsage.frameSlots == 4,
          "fan-out liveness counts one shared output through its last consumer and reuses exact slots");
    auto badReusedMatte = reusedMatte;
    badReusedMatte.edges[2].toPort = 0;
    check(!videowire::compileVisualLayerExecution(badReusedMatte, execution, error)
          && error == "typed matte graph has unsupported production topology",
          "matte fan-out still requires exact distinct typed destination ports");
    const std::array<std::array<float, 4>, 4> oracles {{
        {{0.7f, 0.4f, 0.7f, 0.0f}}, {{0.7f, 0.4f, 0.4f, 1.0f}},
        {{0.7f, 0.4f, 0.3f, 2.0f}}, {{0.7f, 0.4f, 0.3f, 3.0f}} }};
    for (const auto& oracle : oracles)
    {
        const int mode = (int)oracle[3];
        const float actual = mode == 0 ? std::max(oracle[0], oracle[1])
                           : mode == 1 ? std::min(oracle[0], oracle[1])
                           : mode == 2 ? std::max(oracle[0] - oracle[1], 0.0f)
                                       : std::abs(oracle[0] - oracle[1]);
        check(std::abs(actual - oracle[2]) < 1.0e-6f, "tiny independent matte-combine pixel oracle");
    }
    FakeLayer missingSecond = matteLayer; missingSecond.matteTextureB = 0;
    check(!videowire::executeVisualLayerPlan({combined}, 7, missingSecond, error, videohelper::geometry::PlanUse::preview)
          && error == "typed matte combine GPU texture is unavailable",
          "combine rejects a missing retained second GPU texture");
    auto malformedMatte = matte;
    malformedMatte.edges[2].toPort = 0;
    check (! videowire::executeVisualLayerPlan({ malformedMatte }, 7, matteLayer, error, videohelper::geometry::PlanUse::preview)
           && error == "typed matte graph has unsupported production topology",
           "malformed matte topology fails closed");
    auto missingMatte = matte;
    missingMatte.operations[1].payloadXml =
        "<MatteAssetBinding state=\"available\" backend=\"rgba-cpu-decode-native-gpu-upload\"/>";
    check (! videowire::executeVisualLayerPlan({ missingMatte }, 7, matteLayer, error, videohelper::geometry::PlanUse::preview)
           && error == "typed matte asset binding is missing, stale, or unsupported",
           "missing matte binding fails closed");
    auto staleMatte = matte;
    staleMatte.operations[1].payloadXml =
        "<MatteAssetBinding matteAssetId=\"matte-1\" state=\"stale\" backend=\"rgba-cpu-decode-native-gpu-upload\"/>";
    check (! videowire::executeVisualLayerPlan({ staleMatte }, 7, matteLayer, error, videohelper::geometry::PlanUse::preview)
           && error == "typed matte asset binding is missing, stale, or unsupported",
           "stale matte binding fails closed");
    FakeLayer texturelessMatte;
    check (! videowire::executeVisualLayerPlan({ matte }, 7, texturelessMatte, error, videohelper::geometry::PlanUse::preview)
           && error == "typed matte GPU texture is unavailable",
           "unavailable matte texture fails closed");
    auto depth = depthPlan();
    videowire::ExecutableDepthPayload viewportPayload, exportPayload;
    std::string viewportDepthError, exportDepthError;
    check (videowire::visualPlanDepthBinding({ depth }, 7, viewportPayload, viewportDepthError)
           && videowire::visualPlanDepthBinding({ depth }, 7, exportPayload, exportDepthError)
           && viewportPayload.cacheKey == exportPayload.cacheKey
           && viewportPayload.contentReceipt == exportPayload.contentReceipt
           && viewportPayload.width == exportPayload.width
           && viewportPayload.height == exportPayload.height,
           "viewport/export shared depth preparation reads identical immutable descriptors");
    FakeLayer depthLayer;
    depthLayer.depthTexture = 9; depthLayer.depthWidth = 2; depthLayer.depthHeight = 2;
    check (videowire::executeVisualLayerPlan({ depth }, 7, depthLayer, error, videohelper::geometry::PlanUse::preview)
           && depthLayer.depthFog && depthLayer.fogNear == 0.0f && depthLayer.fogFar == 1.0f
           && depthLayer.fogDensity == 32.0f && depthLayer.fogAlpha == 1.0f,
           "depth fog exact topology executes with bounded parameters");
    const std::array<std::pair<const char*, const char*>, 3> depthKinds {{
        { "visual.depth.blur", "<NodeParams radius=\"99\" focus=\"0.25\" falloff=\"0\"/>" },
        { "visual.depth.displace", "<NodeParams amountX=\"1\" amountY=\"-1\" center=\"2\"/>" },
        { "visual.depth.relight", "<NodeParams intensity=\"3\" ambient=\"-1\" red=\"3\" green=\"0.5\" blue=\"1\"/>" } }};
    for (size_t i = 0; i < depthKinds.size(); ++i) {
        auto primitive = depthPlan();
        primitive.nodeKinds[2] = depthKinds[i].first;
        primitive.operations[2].kind = depthKinds[i].first;
        primitive.operations[2].payloadXml = depthKinds[i].second;
        FakeLayer primitiveLayer;
        primitiveLayer.depthTexture = 10 + static_cast<unsigned>(i);
        primitiveLayer.depthWidth = primitiveLayer.depthHeight = 2;
        check(videowire::executeVisualLayerPlan({ primitive }, 7, primitiveLayer, error, videohelper::geometry::PlanUse::preview)
              && primitiveLayer.depthEffect == static_cast<int>(i) + 2 && !primitiveLayer.depthFog,
              "depth primitive exact topology lowers to its stable native wire mode");
    }
    videowire::VisualInspectionTarget depthTarget { 7, 0, 61, 0 };
    check (! videowire::validateVisualInspectionTarget({ depth }, depthTarget, error)
           && error == "inspection target is not an image output port",
           "metadata-only depth node cannot be inspected as a materialized output");

    for (const char* extra : { "path", "file", "uri", "decoder", "upload", "futureField" })
    {
        auto injected = depth;
        injected.operations[1].payloadXml.insert(injected.operations[1].payloadXml.size() - 2,
                                                  std::string(" ") + extra + "=\"x\"");
        check (! videowire::executeVisualLayerPlan({ injected }, 7, depthLayer, error, videohelper::geometry::PlanUse::preview),
               "depth executor rejects every extra payload attribute");
    }
    for (const char* malformed : {
             "<Other depthAssetId=\"depth-1\" depthAssetVersion=\"v1\" state=\"available\" contract=\"parked-metadata\" pendingExecutionSeam=\"depth-consuming-operation\"/>",
             "<DepthAssetBinding depthAssetId=\"depth-1\" depthAssetId=\"duplicate\" depthAssetVersion=\"v1\" state=\"available\" contract=\"parked-metadata\" pendingExecutionSeam=\"depth-consuming-operation\"/>",
             "<DepthAssetBinding depthAssetId=\"depth-1\" depthAssetVersion=\"v1\" state=\"available\" contract=\"parked-metadata\" pendingExecutionSeam=\"depth-consuming-operation\"><child/></DepthAssetBinding>",
             "<DepthAssetBinding depthAssetId=\"depth-1\" depthAssetVersion=\"v1\" state=\"available\" contract=\"parked-metadata\" pendingExecutionSeam=\"depth-consuming-operation\">text</DepthAssetBinding>" })
    {
        auto rejectedPayload = depth;
        rejectedPayload.operations[1].payloadXml = malformed;
        check (! videowire::executeVisualLayerPlan({ rejectedPayload }, 7, depthLayer, error, videohelper::geometry::PlanUse::preview),
               "depth executor rejects malformed exact-schema payload");
    }
    depth.edges[1].fromPort = 0;
    check (! videowire::executeVisualLayerPlan({ depth }, 7, depthLayer, error, videohelper::geometry::PlanUse::preview)
           && error == "typed depth asset binding is missing, stale, or unsupported",
           "depth topology rejects any connection into composition");
    videowire::VisualInspectionTarget inspection { 7, 0, 11, 0 };
    check (videowire::validateVisualInspectionTarget({ direct }, inspection, error),
           "exact executable image output is inspectable");
    inspection.structuralRevision = 1;
    check (! videowire::validateVisualInspectionTarget({ direct }, inspection, error)
           && error == "inspection target revision is stale",
           "stale inspection revisions fail closed");
    inspection = { 7, 0, 12, 0 };
    check (! videowire::validateVisualInspectionTarget({ direct }, inspection, error)
           && error == "inspection target is not an image output port",
           "input ports cannot be selected as intermediate outputs");

    auto sourceThenEffects = direct;
    sourceThenEffects.nodeKinds = { "video.source", "video.transform", "video.effects", "video.out" };
    sourceThenEffects.nodeIds = { 11, 14, 13, 12 };
    sourceThenEffects.operations = {
        { 11, "video.source", "source-decode", "" },
        { 14, "video.transform", "native-gpu", "" },
        { 13, "video.effects", "native-gpu", "" },
        { 12, "video.out", "native-gpu", "" }
    };
    sourceThenEffects.edges = { { 11, 0, 14, 0 }, { 14, 1, 13, 0 }, { 13, 1, 12, 0 } };
    sourceThenEffects.ports.push_back(
        { 14, 1, 1, "out", "frame", "image", "rgba8", "sRGB" });
    inspection = { 7, 0, 11, 0 };
    check (videowire::validateVisualInspectionTarget({ sourceThenEffects }, inspection, error),
           "decoded source output is the admitted non-terminal inspection slice");
    FakeLayer sourceLayer;
    sourceLayer.texture = 77;
    sourceLayer.texWidth = 1920;
    sourceLayer.texHeight = 1080;
    videowire::VisualInspectionResource sourceResource;
    check (videowire::executeVisualLayerPlan({ sourceThenEffects }, 7, sourceLayer, error,videohelper::geometry::PlanUse::preview,
                                             &inspection, &sourceResource)
           && sourceResource.handle == 77 && sourceResource.width == 1920
           && sourceResource.height == 1080 && sourceResource.nodeId == 11,
           "executor retains the genuine decoded GPU texture without readback");
    inspection = { 7, 0, 14, 1 };
    check (! videowire::validateVisualInspectionTarget({ sourceThenEffects }, inspection, error)
           && error == "inspection target has no retainable GPU output resource",
           "transform followed by another operation is not misreported at the renderer boundary");
    auto sourceThenTransform = sourceThenEffects;
    sourceThenTransform.nodeKinds.erase(sourceThenTransform.nodeKinds.begin() + 2);
    sourceThenTransform.nodeIds.erase(sourceThenTransform.nodeIds.begin() + 2);
    sourceThenTransform.operations.erase(sourceThenTransform.operations.begin() + 2);
    sourceThenTransform.edges = { { 11, 0, 14, 0 }, { 14, 1, 12, 0 } };
    check (videowire::validateVisualInspectionTarget({ sourceThenTransform }, inspection, error)
           && videowire::classifyVisualInspectionTarget(sourceThenTransform, inspection)
                == videowire::VisualInspectionSlice::transformedLayer,
           "exact terminal transform output uses the genuine renderer resource seam");

    auto reversed = direct;
    std::reverse (reversed.operations.begin(), reversed.operations.end());
    check (! videowire::executeVisualLayerPlan ({ reversed }, 7, layer, error, videohelper::geometry::PlanUse::preview),
           "ordered non-DAG operation storage must agree with exact edges");

    auto branch = direct;
    branch.nodeKinds.insert (branch.nodeKinds.begin() + 1, "video.effects");
    branch.nodeIds.insert (branch.nodeIds.begin() + 1, 13);
    branch.operations.insert (branch.operations.begin() + 1,
                              { 13, "video.effects", "native-gpu", "" });
    branch.edges = { { 11, 0, 13, 0 }, { 11, 0, 12, 0 } };
    check (! videowire::compileVisualLayerExecution (branch, execution, error),
           "branching image execution fails closed instead of flattening");

    auto threeSource = direct;
    threeSource.nodeKinds = { "video.out", "video.layer.source", "video.blend",
                              "video.source", "video.text" };
    threeSource.nodeIds = { 12, 26, 24, 11, 25 };
    // Deliberately store nodes, operations and edges out of execution order.
    // Stable identities and exact destination ports, not vector position, own
    // lowering.
    threeSource.operations = {
        { 12, "video.out", "native-gpu", "" },
        { 26, "video.layer.source", "source-decode", "" },
        { 24, "video.blend", "native-gpu", "" },
        { 25, "video.text", "native-gpu", "" },
        { 11, "video.source", "source-decode", "" }
    };
    threeSource.ports = {
        { 11, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 25, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 26, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 24, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 24, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 24, 2, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 24, 3, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 12, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    threeSource.edges = {
        { 24, 1, 12, 0 }, { 26, 0, 24, 3 },
        { 11, 0, 24, 0 }, { 25, 0, 24, 2 }
    };
    check (videowire::validateCompiledVisualLayerPlans({}, { threeSource }, false, error),
           "three-source composite satisfies the complete typed snapshot contract");
    auto sceneComposite = threeSource;
    sceneComposite.nodeKinds = { "visual.3d.imported-animation", "visual.3d.render", "video.blend", "video.out" };
    sceneComposite.nodeIds = { 70, 71, 24, 12 };
    sceneComposite.operations = {
        { 70, "visual.3d.imported-animation", "source-decode", "" },
        { 71, "visual.3d.render", "native-gpu", "" },
        { 24, "video.blend", "native-gpu", "" }, { 12, "video.out", "native-gpu", "" }
    };
    sceneComposite.ports = {
        { 70, 4, 1, "out", "control", "scene3D", "unspecified", "unspecified" },
        { 71, 0, 1, "in", "control", "scene3D", "unspecified", "unspecified" },
        { 71, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 24, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 24, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 24, 2, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 24, 3, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 12, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    sceneComposite.edges = { { 70, 4, 71, 0 }, { 71, 1, 24, 0 }, { 24, 1, 12, 0 } };
    check (videowire::validateCompiledVisualLayerPlans({}, { sceneComposite }, false, error),
           "bounded image admission stops at a native scene control boundary");
    auto wrongSceneImage = sceneComposite;
    wrongSceneImage.ports[2].pixelFormat = "rgba16f";
    check (! videowire::validateCompiledVisualLayerPlans({}, { wrongSceneImage }, false, error)
               && error == "bounded visual DAG requires RGBA8 sRGB Frame<Image> descriptors",
           "native scene compositor still rejects a non-RGBA8 image boundary");
    check (videowire::compileVisualLayerExecution(threeSource, execution, error)
           && execution.compositeInputCount == 3
           && execution.compositeSourceNodeIds[0] == 11
           && execution.compositeDestinationPorts[0] == 0
           && execution.compositeSourceNodeIds[1] == 25
           && execution.compositeDestinationPorts[1] == 2
           && execution.compositeSourceNodeIds[2] == 26
           && execution.compositeDestinationPorts[2] == 3
           && execution.dagSchedule.operations.size() == 5
           && threeSource.operations[execution.dagSchedule.operations[0].operationIndex].nodeId == 11
           && threeSource.operations[execution.dagSchedule.operations[1].operationIndex].nodeId == 25
           && threeSource.operations[execution.dagSchedule.operations[2].operationIndex].nodeId == 26
           && execution.dagSchedule.operations[3].inputs.size() == 3
           && execution.dagSchedule.operations[3].inputs[0].toPort == 0
           && execution.dagSchedule.operations[3].inputs[0].fromNodeId == 11
           && execution.dagSchedule.operations[3].inputs[1].toPort == 2
           && execution.dagSchedule.operations[3].inputs[1].fromNodeId == 25
           && execution.dagSchedule.operations[3].inputs[2].toPort == 3
           && execution.dagSchedule.operations[3].inputs[2].fromNodeId == 26,
           "three-source composite lowers exact predecessors in deterministic Blend-port order");

    const auto withWholeGraphImageDescriptor = [&] (const char* format,
                                                     const char* colorSpace)
    {
        auto candidate = threeSource;
        for (auto& port : candidate.ports)
            if (port.carrier == "frame" && port.dataType == "image")
            {
                port.pixelFormat = format;
                port.colorSpace = colorSpace;
            }
        return candidate;
    };
    for (const auto& unsupported : {
             std::pair<const char*, const char*> { "rgba16f", "sRGB" },
             std::pair<const char*, const char*> { "rgba8", "linearSRGB" },
             std::pair<const char*, const char*> { "futureFormat", "sRGB" },
             std::pair<const char*, const char*> { "rgba8", "displayP3" },
             std::pair<const char*, const char*> { "rgba8", "futureColorSpace" } })
    {
        const auto candidate = withWholeGraphImageDescriptor(unsupported.first,
                                                              unsupported.second);
        error.clear();
        check (! videowire::validateCompiledVisualLayerPlans({}, { candidate }, false, error)
               && error == "bounded visual DAG requires RGBA8 sRGB Frame<Image> descriptors",
               "snapshot admission rejects a consistently unsupported compositor image descriptor");
        error.clear();
        check (! videowire::compileVisualLayerExecution(candidate, execution, error)
               && error == "bounded visual DAG requires RGBA8 sRGB Frame<Image> descriptors",
               "execution lowering rejects a consistently unsupported compositor image descriptor");
        auto rejectedState = std::make_unique<videowire::VisualPlanExecutionState>();
        error.clear();
        check (! rejectedState->admitPlans({ candidate }, &error)
               && rejectedState->compiled(candidate.clipId,
                                           candidate.structuralRevision) == nullptr
               && error == "bounded visual DAG requires RGBA8 sRGB Frame<Image> descriptors",
               "admission installation rejects a consistently unsupported compositor image descriptor");
    }
    struct ImageDescriptorBoundary
    {
        size_t portIndex;
        const char* name;
    };
    const std::array<ImageDescriptorBoundary, 8> imageDescriptorBoundaries {{
        { 0, "primary output" },
        { 1, "text secondary output" },
        { 2, "layer secondary output" },
        { 3, "Blend primary input" },
        { 4, "Blend output" },
        { 5, "Blend text input" },
        { 6, "Blend layer input" },
        { 7, "Output input" }
    }};
    for (size_t boundaryIndex = 0; boundaryIndex < imageDescriptorBoundaries.size();
         ++boundaryIndex)
    {
        auto candidate = threeSource;
        auto& port = candidate.ports[imageDescriptorBoundaries[boundaryIndex].portIndex];
        const auto boundaryEdge = std::find_if(candidate.edges.begin(), candidate.edges.end(),
            [&](const auto& edge)
            {
                return (edge.fromNodeId == port.nodeId && edge.fromPort == port.port)
                    || (edge.toNodeId == port.nodeId && edge.toPort == port.port);
            });
        for (auto& candidatePort : candidate.ports)
            if ((candidatePort.nodeId == boundaryEdge->fromNodeId
                 && candidatePort.port == boundaryEdge->fromPort)
                || (candidatePort.nodeId == boundaryEdge->toNodeId
                    && candidatePort.port == boundaryEdge->toPort))
                candidatePort.dataType = "futureImage";
        const std::string boundary = imageDescriptorBoundaries[boundaryIndex].name;
        error.clear();
        check (! videowire::validateCompiledVisualLayerPlans({}, { candidate }, false, error)
               && error == "bounded visual DAG requires RGBA8 sRGB Frame<Image> descriptors",
               ("snapshot validation rejects futureImage at the native " + boundary).c_str());
        error.clear();
        check (! videowire::compileVisualLayerExecution(candidate, execution, error)
               && error == "bounded visual DAG requires RGBA8 sRGB Frame<Image> descriptors",
               ("direct lowering rejects futureImage at the native " + boundary).c_str());
        auto rejectedState = std::make_unique<videowire::VisualPlanExecutionState>();
        error.clear();
        check (! rejectedState->admitPlans({ candidate }, &error)
               && rejectedState->compiled(candidate.clipId,
                                           candidate.structuralRevision) == nullptr
               && error == "bounded visual DAG requires RGBA8 sRGB Frame<Image> descriptors",
               ("admission installation rejects futureImage at the native " + boundary).c_str());
    }
    for (int malformedField = 0; malformedField < 3; ++malformedField)
    {
        auto candidate = threeSource;
        auto& port = candidate.ports.front();
        if (malformedField == 0) port.channels = 2;
        if (malformedField == 1) port.carrier = "futureFrame";
        if (malformedField == 2) port.dataType = "futureImage";
        error.clear();
        check (! videowire::validateCompiledVisualLayerPlans({}, { candidate }, false, error)
               && error == "bounded visual DAG requires RGBA8 sRGB Frame<Image> descriptors",
               "snapshot validation requires the exact native image descriptor tuple");
        error.clear();
        check (! videowire::compileVisualLayerExecution(candidate, execution, error)
               && error == "bounded visual DAG requires RGBA8 sRGB Frame<Image> descriptors",
               "direct lowering requires the exact native image descriptor tuple");
        auto rejectedState = std::make_unique<videowire::VisualPlanExecutionState>();
        error.clear();
        check (! rejectedState->admitPlans({ candidate }, &error)
               && rejectedState->compiled(candidate.clipId,
                                           candidate.structuralRevision) == nullptr
               && error == "bounded visual DAG requires RGBA8 sRGB Frame<Image> descriptors",
               "admission installation requires the exact native image descriptor tuple");
    }
    FakeLayer threeSourceLayer;
    check (videowire::executeVisualLayerPlan({ threeSource }, 7, threeSourceLayer, error, videohelper::geometry::PlanUse::preview)
           && threeSourceLayer.scale == 1.0f && threeSourceLayer.effects == nullptr
           && threeSourceLayer.maskType == 0,
           "three-source composite executes through the existing neutral primary descriptor");
    auto permutedThreeSource = threeSource;
    std::rotate(permutedThreeSource.nodeKinds.begin(), permutedThreeSource.nodeKinds.begin() + 2,
                permutedThreeSource.nodeKinds.end());
    std::rotate(permutedThreeSource.nodeIds.begin(), permutedThreeSource.nodeIds.begin() + 2,
                permutedThreeSource.nodeIds.end());
    std::reverse(permutedThreeSource.operations.begin(), permutedThreeSource.operations.end());
    std::reverse(permutedThreeSource.edges.begin(), permutedThreeSource.edges.end());
    videowire::CompiledVisualDagSchedule permutedSchedule;
    check (videowire::compileBoundedVisualDagSchedule(permutedThreeSource, permutedSchedule, error)
           && permutedSchedule.operations.size() == execution.dagSchedule.operations.size()
           && permutedThreeSource.operations[permutedSchedule.operations[0].operationIndex].nodeId == 11
           && permutedThreeSource.operations[permutedSchedule.operations[1].operationIndex].nodeId == 25
           && permutedThreeSource.operations[permutedSchedule.operations[2].operationIndex].nodeId == 26,
           "bounded compositor scheduling is stable across document permutations");

    videowire::VisualPlanResourceUsage threeSourceUsage;
    check (videowire::accountVisualPlanResources(threeSource, 64, 64, threeSourceUsage, error),
           "three-source composite resource usage is independently measured");
    auto accountedThreeSource = threeSource;
    accountedThreeSource.descriptorCount = threeSourceUsage.descriptors;
    accountedThreeSource.operationCount = threeSourceUsage.operations;
    accountedThreeSource.sceneRecordCount = threeSourceUsage.sceneRecords;
    accountedThreeSource.frameOutputCount = threeSourceUsage.frameOutputs;
    accountedThreeSource.peakLiveFrameCount = threeSourceUsage.peakLiveFrames;
    accountedThreeSource.allocatedFrameSlotCount = threeSourceUsage.frameSlots;
    auto threeSourceState = std::make_unique<videowire::VisualPlanExecutionState>();
    auto threeSourceLimits = videowire::VisualBackendResourceLimits::forCanvas(64, 64);
    check (threeSourceState->admitPlans({ accountedThreeSource }, &error, 64, 64, &threeSourceLimits)
           && threeSourceState->compiled(7, accountedThreeSource.structuralRevision) != nullptr,
           "three-source lowering preserves producer resource fields and admission");
    auto constrainedThreeSourceLimits = threeSourceLimits;
    constrainedThreeSourceLimits.descriptors = 4;
    auto rejectedThreeSourceState = std::make_unique<videowire::VisualPlanExecutionState>();
    check (! rejectedThreeSourceState->admitPlans({ accountedThreeSource }, &error, 64, 64,
                                                   &constrainedThreeSourceLimits)
           && error == "visual graph descriptor capacity exceeded: 5 > 4",
           "three-source lowering cannot bypass the descriptor budget");

    const auto mediumA = mediumBudgetPlan(101, 1000);
    const auto mediumB = mediumBudgetPlan(102, 2000);
    const std::vector<videowire::CompiledVisualLayerPlan> mediumPlans { mediumB, mediumA };
    auto mediumLimits = videowire::VisualBackendResourceLimits::forCanvas(64, 64);
    mediumLimits.descriptors = 12;
    mediumLimits.operations = 12;
    mediumLimits.sceneRecords = 0;
    mediumLimits.liveFrames = 4;
    mediumLimits.frameSlots = 4;
    mediumLimits.allocatedFrameBytes = 64ull * 64ull * 4ull * 4ull;
    mediumLimits.backendProfile = "opengl-4.6";
    mediumLimits.maximumImageDimensionSource = "OpenGL.GL_MAX_TEXTURE_SIZE";
    mediumLimits.allocatedFrameBytesSource = "test-exact-backend-budget";
    auto previewBudgetState = std::make_unique<videowire::VisualPlanExecutionState>();
    auto exportBudgetState = std::make_unique<videowire::VisualPlanExecutionState>();
    FakeLayer previewMediumA, previewMediumB, exportMediumA, exportMediumB;
    error.clear();
    check (mediumA.descriptorCount == 6 && mediumA.operationCount == 6
           && mediumA.frameOutputCount == 5 && mediumA.peakLiveFrameCount == 2
           && mediumA.allocatedFrameSlotCount == 2
           && previewBudgetState->admitPlans(mediumPlans, &error, 64, 64, &mediumLimits)
           && exportBudgetState->admitPlans(mediumPlans, &error, 64, 64, &mediumLimits)
           && videowire::executeVisualLayerPlan(mediumPlans, 101, previewMediumA, error,videohelper::geometry::PlanUse::preview,
                                                nullptr, nullptr, previewBudgetState.get(), 1.0)
           && videowire::executeVisualLayerPlan(mediumPlans, 102, previewMediumB, error,videohelper::geometry::PlanUse::preview,
                                                nullptr, nullptr, previewBudgetState.get(), 1.0)
           && videowire::executeVisualLayerPlan(mediumPlans, 101, exportMediumA, error,videohelper::geometry::PlanUse::preview,
                                                nullptr, nullptr, exportBudgetState.get(), 1.0)
           && videowire::executeVisualLayerPlan(mediumPlans, 102, exportMediumB, error,videohelper::geometry::PlanUse::preview,
                                                nullptr, nullptr, exportBudgetState.get(), 1.0),
           "preview and export execute two medium graphs at the exact aggregate budget");
    const videowire::VisualPlanResourceReceipt expectedMediumReceipt {
        2, 12, 12, 0, 10, 4, 4, 64ull * 64ull * 4ull * 4ull
    };
    check (previewBudgetState->resourceReceipt() == expectedMediumReceipt
           && exportBudgetState->resourceReceipt() == expectedMediumReceipt,
           "preview and export publish the same exact liveness, reuse, byte, and work receipt");
    auto sceneA = importedAnimationPlan();
    auto sceneB = sceneA;
    sceneB.clipId = 8;
    sceneB.structuralRevision = 10;
    videowire::VisualPlanResourceUsage sceneUsage;
    check(videowire::accountVisualPlanResources(sceneA, 64, 64, sceneUsage, error)
          && sceneUsage.sceneRecords > 0,
          "scene plan resource usage includes bounded scene records");
    for (auto* scenePlan : { &sceneA, &sceneB })
    {
        scenePlan->descriptorCount = sceneUsage.descriptors;
        scenePlan->operationCount = sceneUsage.operations;
        scenePlan->sceneRecordCount = sceneUsage.sceneRecords;
        scenePlan->frameOutputCount = sceneUsage.frameOutputs;
        scenePlan->peakLiveFrameCount = sceneUsage.peakLiveFrames;
        scenePlan->allocatedFrameSlotCount = sceneUsage.frameSlots;
    }
    auto sceneLimits = videowire::VisualBackendResourceLimits::forCanvas(64, 64);
    sceneLimits.descriptors = sceneUsage.descriptors * 2;
    sceneLimits.operations = sceneUsage.operations * 2;
    sceneLimits.sceneRecords = sceneUsage.sceneRecords * 2;
    auto sceneBudgetState = std::make_unique<videowire::VisualPlanExecutionState>();
    error.clear();
    check(sceneBudgetState->admitPlans({ sceneA, sceneB }, &error, 64, 64, &sceneLimits)
          && sceneBudgetState->resourceReceipt().planCount == 2
          && sceneBudgetState->resourceReceipt().descriptors == sceneUsage.descriptors * 2
          && sceneBudgetState->resourceReceipt().operations == sceneUsage.operations * 2
          && sceneBudgetState->resourceReceipt().sceneRecords == sceneUsage.sceneRecords * 2,
          "descriptor, operation, and scene-record receipts report aggregate owner work");
    const auto expectedBudgetReceipt = previewBudgetState->budgetReceipt();
    const auto previewBudgetTelemetry = previewBudgetState->telemetry().snapshot();
    const auto exportBudgetTelemetry = exportBudgetState->telemetry().snapshot();
    const auto previewBudgetJson = videowire::visualTelemetryJson(previewBudgetTelemetry);
    check (expectedBudgetReceipt == exportBudgetState->budgetReceipt()
           && previewBudgetTelemetry.budgetObserved && exportBudgetTelemetry.budgetObserved
           && previewBudgetTelemetry.budgetReceipt == expectedBudgetReceipt
           && exportBudgetTelemetry.budgetReceipt == expectedBudgetReceipt
           && previewBudgetJson["budget"]["backendProfile"] == "opengl-4.6"
           && previewBudgetJson["budget"]["usage"]["allocatedFrameBytes"] == 65536
           && previewBudgetJson["budget"]["provenance"]["allocatedFrameBytes"]
                == "test-exact-backend-budget",
           "preview and export telemetry carry the same immutable budget and usage receipt");
    auto staleMediumA = mediumA;
    staleMediumA.structuralRevision = mediumA.structuralRevision - 1;
    staleMediumA.descriptorCount = 100;
    staleMediumA.operationCount = 100;
    auto duplicateBudgetState = std::make_unique<videowire::VisualPlanExecutionState>();
    error.clear();
    check (duplicateBudgetState->admitPlans({ mediumB, staleMediumA, mediumA }, &error,
                                            64, 64, &mediumLimits)
           && duplicateBudgetState->resourceReceipt() == expectedMediumReceipt
           && duplicateBudgetState->compiled(101, mediumA.structuralRevision) != nullptr,
           "aggregate accounting normalizes stale clip revisions before admission and receipts");
    const std::vector<videowire::CompiledVisualLayerPlan> duplicateOrder {
        staleMediumA, mediumB, mediumA
    };
    const auto* selectedDuplicate = videowire::findVisualLayerPlan(duplicateOrder, 101);
    check (selectedDuplicate != nullptr
           && selectedDuplicate->structuralRevision == mediumA.structuralRevision,
           "execution selects the same newest clip revision normalized by aggregate admission");
    FakeLayer duplicateExecutionLayer;
    error.clear();
    check(videowire::executeVisualLayerPlan(
              duplicateOrder, 101, duplicateExecutionLayer, error,videohelper::geometry::PlanUse::preview,
              nullptr, nullptr, duplicateBudgetState.get(), 1.0)
          && duplicateBudgetState->compiled(101, mediumA.structuralRevision) != nullptr,
          "stale-first newest-second duplicate execution uses the admitted newest revision end to end");
    const auto stableReceipt = previewBudgetState->resourceReceipt();
    auto insufficientWork = mediumLimits;
    insufficientWork.operations = 11;
    error.clear();
    check (! previewBudgetState->admitPlans(mediumPlans, &error, 64, 64, &insufficientWork)
           && error == "visual graph aggregate operation capacity exceeded: 12 > 11"
           && previewBudgetState->resourceReceipt() == stableReceipt
           && previewBudgetState->compiled(101, mediumA.structuralRevision) != nullptr
           && previewBudgetState->compiled(102, mediumB.structuralRevision) != nullptr,
           "aggregate work rejection preserves both last-good graphs and their receipt");
    check (previewBudgetState->budgetReceipt() == expectedBudgetReceipt
           && previewBudgetState->telemetry().snapshot().budgetReceipt == expectedBudgetReceipt,
           "rejected preview admission preserves last-good budget telemetry");
    uint64_t overflowTotal = std::numeric_limits<uint64_t>::max() - 3;
    check(! videowire::checkedAccumulateVisualPlanBytes(overflowTotal, 2, 2)
          && overflowTotal == std::numeric_limits<uint64_t>::max() - 3,
          "split history and transient byte overflow fails without partial accumulation");
    auto insufficientLiveness = mediumLimits;
    insufficientLiveness.liveFrames = 3;
    error.clear();
    check (! previewBudgetState->admitPlans(mediumPlans, &error, 64, 64,
                                            &insufficientLiveness)
           && error == "visual graph live frame capacity exceeded: 4 > 3"
           && previewBudgetState->resourceReceipt() == stableReceipt,
           "aggregate liveness sums simultaneously admitted graph peaks");
    auto insufficientBytes = mediumLimits;
    --insufficientBytes.allocatedFrameBytes;
    error.clear();
    check (! exportBudgetState->admitPlans(mediumPlans, &error, 64, 64, &insufficientBytes)
           && error == "visual graph allocated frame byte capacity exceeded: 65536 > 65535"
           && exportBudgetState->resourceReceipt() == expectedMediumReceipt,
           "aggregate backend byte rejection preserves the export receipt");
    check (exportBudgetState->budgetReceipt() == expectedBudgetReceipt
           && exportBudgetState->telemetry().snapshot().budgetReceipt == expectedBudgetReceipt,
           "one-over export admission preserves last-good budget telemetry");
    auto swappedThreeSource = threeSource;
    swappedThreeSource.edges[1].toPort = 2;
    swappedThreeSource.edges[3].toPort = 3;
    check (! videowire::compileVisualLayerExecution(swappedThreeSource, execution, error)
           && error == "bounded visual DAG requires exact primary/text/layer Blend bindings",
           "text and layer Blend ports are not interchangeable");
    auto incompleteThreeSource = threeSource;
    incompleteThreeSource.ports.erase(incompleteThreeSource.ports.begin() + 6);
    check (! videowire::compileVisualLayerExecution(incompleteThreeSource, execution, error)
           && error == "bounded visual DAG has an incompatible typed edge",
           "three-source execution rejects a missing typed Blend port");
    auto incompatibleThreeSource = threeSource;
    incompatibleThreeSource.ports[6].colorSpace = "linearSRGB";
    check (! videowire::compileVisualLayerExecution(incompatibleThreeSource, execution, error)
           && error == "bounded visual DAG requires RGBA8 sRGB Frame<Image> descriptors",
           "three-source execution rejects implicit color conversion");
    auto fallbackThreeSource = threeSource;
    fallbackThreeSource.operations[1].backendCapability = "native-gpu";
    check (! videowire::compileVisualLayerExecution(fallbackThreeSource, execution, error)
           && error == "bounded visual DAG has an incompatible native backend capability",
           "three-source execution rejects a layer source without decode capability");
    auto duplicateInputThreeSource = threeSource;
    duplicateInputThreeSource.edges.push_back({ 25, 0, 24, 3 });
    check (! videowire::compileVisualLayerExecution(duplicateInputThreeSource, execution, error)
           && error == "bounded visual DAG has a duplicate edge or multiply-bound input port",
           "three-source execution rejects duplicate Blend input bindings");
    auto cyclicThreeSource = threeSource;
    cyclicThreeSource.ports.push_back({ 12, 1, 1, "out", "frame", "image", "rgba8", "sRGB" });
    cyclicThreeSource.ports.push_back({ 11, 1, 1, "in", "frame", "image", "rgba8", "sRGB" });
    cyclicThreeSource.edges.push_back({ 12, 1, 11, 1 });
    check (! videowire::compileVisualLayerExecution(cyclicThreeSource, execution, error)
           && error == "bounded visual DAG contains a cycle",
           "bounded compositor rejects cycles before execution lowering");

    auto composite = direct;
    composite.nodeKinds = { "video.source", "video.transform", "video.effects",
        "video.mask.shape", "video.text", "video.layer.source", "video.blend", "video.out" };
    composite.nodeIds = { 11, 21, 22, 23, 25, 26, 24, 12 };
    composite.operations.clear();
    for (size_t i = 0; i < composite.nodeIds.size(); ++i)
        composite.operations.push_back({ composite.nodeIds[i], composite.nodeKinds[i],
            i == 0 || i == 5 ? "source-decode" : "native-gpu", "" });
    composite.edges = {
        { 11, 0, 21, 0 }, { 21, 1, 22, 0 }, { 22, 1, 23, 0 },
        { 23, 1, 24, 0 }, { 25, 0, 24, 2 }, { 26, 0, 24, 3 }, { 24, 1, 12, 0 }
    };
    composite.ports = {
        { 11, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 21, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 21, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 22, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 22, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 23, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 23, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 25, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 26, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 24, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 24, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 24, 2, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 24, 3, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 12, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    for (const size_t portIndex : { size_t { 1 }, size_t { 2 } })
    {
        auto candidate = composite;
        const auto& port = candidate.ports[portIndex];
        const auto boundaryEdge = std::find_if(candidate.edges.begin(), candidate.edges.end(),
            [&](const auto& edge)
            {
                return (edge.fromNodeId == port.nodeId && edge.fromPort == port.port)
                    || (edge.toNodeId == port.nodeId && edge.toPort == port.port);
            });
        for (auto& candidatePort : candidate.ports)
            if ((candidatePort.nodeId == boundaryEdge->fromNodeId
                 && candidatePort.port == boundaryEdge->fromPort)
                || (candidatePort.nodeId == boundaryEdge->toNodeId
                    && candidatePort.port == boundaryEdge->toPort))
                candidatePort.dataType = "futureImage";
        error.clear();
        check (! videowire::validateCompiledVisualLayerPlans({}, { candidate }, false, error)
               && error == "bounded visual DAG requires RGBA8 sRGB Frame<Image> descriptors",
               "snapshot validation rejects an inexact intermediate pass image descriptor");
        error.clear();
        check (! videowire::compileVisualLayerExecution(candidate, execution, error)
               && error == "bounded visual DAG requires RGBA8 sRGB Frame<Image> descriptors",
               "direct lowering rejects an inexact intermediate pass image descriptor");
        auto rejectedState = std::make_unique<videowire::VisualPlanExecutionState>();
        error.clear();
        check (! rejectedState->admitPlans({ candidate }, &error)
               && rejectedState->compiled(candidate.clipId,
                                           candidate.structuralRevision) == nullptr
               && error == "bounded visual DAG requires RGBA8 sRGB Frame<Image> descriptors",
               "admission installation rejects an inexact intermediate pass image descriptor");
    }
    FakeLayer compositeLayer;
    check (videowire::executeVisualLayerPlan({ composite }, 7, compositeLayer, error, videohelper::geometry::PlanUse::preview)
           && compositeLayer.scale == 2.0f && compositeLayer.effects != nullptr
           && compositeLayer.maskType == 2,
           "established typed multi-source composite preserves production rendering");

    auto nativeSecondarySources = direct;
    nativeSecondarySources.nodeKinds = { "video.source", "video.layer.source",
                                         "video.text", "video.blend", "video.out" };
    nativeSecondarySources.nodeIds = { 11, 41, 42, 24, 12 };
    nativeSecondarySources.operations = {
        { 11, "video.source", "source-decode", "" },
        { 41, "video.layer.source", "source-decode", "" },
        { 42, "video.text", "native-gpu", "" },
        { 24, "video.blend", "native-gpu", "" },
        { 12, "video.out", "native-gpu", "" }
    };
    // Deliberately reverse source document order at the Blend inputs. Exact
    // destination ports, not document order, own compositor ordering.
    nativeSecondarySources.edges = { { 11, 0, 24, 0 }, { 41, 0, 24, 3 },
                             { 42, 0, 24, 2 }, { 24, 1, 12, 0 } };
    nativeSecondarySources.ports = {
        { 11, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 41, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 42, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 24, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 24, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 24, 2, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 24, 3, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 12, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    check (videowire::validateCompiledVisualLayerPlans({}, { nativeSecondarySources }, false, error),
           "native secondary-source DAG satisfies the complete typed snapshot contract");
    check (videowire::compileVisualLayerExecution(nativeSecondarySources, execution, error)
           && execution.compositeInputCount == 3
           && execution.compositeSourceNodeIds[0] == 11
           && execution.compositeSourceNodeIds[1] == 42
           && execution.compositeSourceNodeIds[2] == 41
           && execution.compositeDestinationPorts[0] == 0
           && execution.compositeDestinationPorts[1] == 2
           && execution.compositeDestinationPorts[2] == 3
           && ! execution.transform && ! execution.effects && ! execution.mask,
           "native text and layer sources execute in deterministic exact port order");
    auto differentSecondaryIdentity = nativeSecondarySources;
    differentSecondaryIdentity.nodeKinds[2] = "video.layer.source";
    differentSecondaryIdentity.operations[2].kind = "video.layer.source";
    differentSecondaryIdentity.operations[2].backendCapability = "source-decode";
    check (! videowire::compileVisualLayerExecution(differentSecondaryIdentity, execution, error)
           && error == "visual composite secondary branch is not an executable native source",
           "native compositor rejects a different source identity at its text-overlay input");

    auto commonEffectComposite = nativeSecondarySources;
    // Keep storage and operation order deliberately unrelated to execution
    // order. Stable node/port bindings own both the primary effect branch and
    // native secondary-source ordering.
    commonEffectComposite.structuralRevision = 8;
    commonEffectComposite.nodeKinds = { "video.out", "video.layer.source", "video.blend",
        "visual.color.saturation", "video.text", "video.source" };
    commonEffectComposite.nodeIds = { 12, 41, 24, 50, 42, 11 };
    commonEffectComposite.operations = {
        { 24, "video.blend", "native-gpu", "" },
        { 50, "visual.color.saturation", "native-gpu",
          "<CommonEffect schemaVersion=\"1\" effect=\"saturation\" amount=\"1.75\"/>" },
        { 41, "video.layer.source", "source-decode", "" },
        { 12, "video.out", "native-gpu", "" },
        { 11, "video.source", "source-decode", "" },
        { 42, "video.text", "native-gpu", "" }
    };
    commonEffectComposite.edges = { { 50, 1, 24, 0 }, { 41, 0, 24, 3 },
        { 24, 1, 12, 0 }, { 11, 0, 50, 0 }, { 42, 0, 24, 2 } };
    commonEffectComposite.ports = {
        { 11, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 50, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 50, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 41, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 42, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 24, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 24, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 24, 2, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 24, 3, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 12, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    check (videowire::validateCompiledVisualLayerPlans(
               {}, { commonEffectComposite }, false, error)
           && videowire::compileVisualLayerExecution(commonEffectComposite, execution, error)
           && execution.commonEffect && ! execution.effects
           && execution.commonEffectType == 2
           && execution.commonEffectParameterCount == 1
           && execution.commonEffectValues[0] == 1.75f
           && execution.compositeInputCount == 3
           && execution.compositeSourceNodeIds[0] == 11
           && execution.compositeSourceNodeIds[1] == 42
           && execution.compositeSourceNodeIds[2] == 41,
           "ordinary multi-source composite admits one exact immutable common-effect stage");
    FakeLayer viewportCommonEffect, exportCommonEffect;
    FakeProductRenderer viewportCommonRenderer, exportCommonRenderer;
    check (videowire::executeVisualLayerPlanForRenderer(
               viewportCommonRenderer, { commonEffectComposite }, 7,
               viewportCommonEffect, error, videohelper::geometry::PlanUse::preview, nullptr)
           && videowire::executeVisualLayerPlanForRenderer(
               exportCommonRenderer, { commonEffectComposite }, 7,
               exportCommonEffect, error, videohelper::geometry::PlanUse::preview, nullptr)
           && viewportCommonRenderer.bridgePreparations == 1
           && exportCommonRenderer.bridgePreparations == 1
           && viewportCommonEffect.effects == &viewportCommonEffect.graphCommonEffect
           && exportCommonEffect.effects == &exportCommonEffect.graphCommonEffect
           && viewportCommonEffect.graphCommonEffect.type
                == exportCommonEffect.graphCommonEffect.type
           && viewportCommonEffect.graphCommonEffect.params[0]
                == exportCommonEffect.graphCommonEffect.params[0],
           "viewport and export share one common-effect composite executor seam");

    auto malformedCompositeEffect = commonEffectComposite;
    malformedCompositeEffect.operations[1].payloadXml =
        "<CommonEffect schemaVersion=\"1\" effect=\"saturation\" amount=\"nan\"/>";
    check (! videowire::compileVisualLayerExecution(
               malformedCompositeEffect, execution, error)
           && error == "visual.color.saturation has a malformed, unsupported, or out-of-bounds immutable payload",
           "ordinary composite rejects malformed common-effect payloads without fallback");
    auto convertedCompositeEffect = commonEffectComposite;
    convertedCompositeEffect.ports[2].colorSpace = "linearSRGB";
    check (! videowire::compileVisualLayerExecution(
               convertedCompositeEffect, execution, error)
           && error == "bounded visual DAG requires RGBA8 sRGB Frame<Image> descriptors",
           "ordinary composite rejects implicit color conversion at typed effect edges");
    auto fallbackCompositeEffect = commonEffectComposite;
    fallbackCompositeEffect.operations[1].backendCapability = "cpu-fallback";
    check (! videowire::compileVisualLayerExecution(
               fallbackCompositeEffect, execution, error)
           && error == "visual composite operation has incompatible native backend capability",
           "ordinary composite rejects CPU effect fallback capability");

    auto unsupportedFanIn = nativeSecondarySources;
    unsupportedFanIn.edges[1].toPort = 2;
    check (! videowire::compileVisualLayerExecution(unsupportedFanIn, execution, error)
           && error == "bounded visual DAG has a duplicate edge or multiply-bound input port",
           "duplicate Blend-port fan-in fails closed with a focused diagnostic");

    auto repeatedTransform = nativeSecondarySources;
    repeatedTransform.nodeKinds = { "video.source", "video.transform", "video.transform",
                                    "video.layer.source", "video.blend", "video.out" };
    repeatedTransform.nodeIds = { 11, 31, 32, 41, 24, 12 };
    repeatedTransform.operations = {
        { 11, "video.source", "source-decode", "" },
        { 31, "video.transform", "native-gpu", "" },
        { 32, "video.transform", "native-gpu", "" },
        { 41, "video.layer.source", "source-decode", "" },
        { 24, "video.blend", "native-gpu", "" },
        { 12, "video.out", "native-gpu", "" }
    };
    repeatedTransform.edges = { { 11, 0, 31, 0 }, { 31, 1, 32, 0 },
                                { 32, 1, 24, 0 }, { 41, 0, 24, 3 },
                                { 24, 1, 12, 0 } };
    repeatedTransform.ports.clear();
    check (! videowire::compileVisualLayerExecution(repeatedTransform, execution, error)
           && error == "visual composite primary branch cannot be represented exactly by LayerDesc",
           "repeated unary passes are rejected instead of collapsing into one descriptor field");

    auto pathMatte = direct;
    pathMatte.nodeKinds = { "video.source", "visual.shape.rectangle", "visual.shape.ellipse",
                            "visual.shape.union", "visual.shape.invert", "visual.matte.path",
                            "video.layer.source", "video.blend", "video.out" };
    pathMatte.nodeIds = { 11, 31, 32, 33, 34, 35, 41, 24, 12 };
    pathMatte.operations = {
        { 11, "video.source", "source-decode", "" },
        { 31, "visual.shape.rectangle", "control-eval",
          "<NodeParams centerX=\"0.25\" centerY=\"0.75\" width=\"0.5\" height=\"0.2\"/>" },
        { 32, "visual.shape.ellipse", "control-eval",
          "<NodeParams centerX=\"0.75\" centerY=\"0.25\" width=\"0.3\" height=\"0.4\"/>" },
        { 33, "visual.shape.union", "control-eval", "" },
        { 34, "visual.shape.invert", "control-eval", "" },
        { 35, "visual.matte.path", "native-gpu", "" },
        { 41, "video.layer.source", "source-decode", "" },
        { 24, "video.blend", "native-gpu", "" },
        { 12, "video.out", "native-gpu", "" }
    };
    pathMatte.edges = { { 11, 0, 35, 0 }, { 31, 0, 33, 0 }, { 32, 0, 33, 1 },
                        { 33, 2, 34, 0 }, { 34, 1, 35, 1 }, { 35, 2, 24, 0 },
                        { 41, 0, 24, 3 },
                        { 24, 1, 12, 0 } };
    pathMatte.ports = {
        { 11, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 31, 0, 1, "out", "control", "shape", "unspecified", "unspecified" },
        { 32, 0, 1, "out", "control", "shape", "unspecified", "unspecified" },
        { 33, 0, 1, "in", "control", "shape", "unspecified", "unspecified" },
        { 33, 1, 1, "in", "control", "shape", "unspecified", "unspecified" },
        { 33, 2, 1, "out", "control", "shape", "unspecified", "unspecified" },
        { 34, 0, 1, "in", "control", "shape", "unspecified", "unspecified" },
        { 34, 1, 1, "out", "control", "shape", "unspecified", "unspecified" },
        { 35, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 35, 1, 1, "in", "control", "shape", "unspecified", "unspecified" },
        { 35, 2, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 41, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 24, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 24, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 24, 3, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 12, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    for (const auto& mode : { std::pair<const char*, int>{ "visual.shape.union", 1 },
                              { "visual.shape.intersection", 2 },
                              { "visual.shape.subtract", 3 } })
    {
        pathMatte.nodeKinds[3] = mode.first;
        pathMatte.operations[3].kind = mode.first;
        const bool pathCompiled = videowire::compileVisualLayerExecution(pathMatte, execution, error);
        check (pathCompiled
               && execution.pathMatte && execution.pathMatteHasSecondary
               && execution.pathMatteInvert && execution.pathMatteOperation == mode.second
               && execution.pathMatteEllipse == false
               && execution.pathMatteSecondaryEllipse,
               "bounded path matte lowers add, intersect, subtract and invert exactly");
    }
    FakeLayer pathLayer;
    const bool pathExecuted = videowire::executeVisualLayerPlan({ pathMatte }, 7, pathLayer, error, videohelper::geometry::PlanUse::preview);
    check (pathExecuted
           && pathLayer.pathMatte && pathLayer.pathMatteOperation == 3
           && pathLayer.pathMatteInvert && pathLayer.maskType == 2,
           "path matte execution preserves the existing clip shape mask");
    auto inventedPathPayload = pathMatte;
    inventedPathPayload.operations[5].payloadXml = "<NodeParams fallback=\"1\"/>";
    const bool inventedValid = videowire::validateCompiledVisualLayerPlans(
        {}, { inventedPathPayload }, false, error);
    check (! inventedValid
           && error == "visual path-matte operation payload must be empty",
           "snapshot admission fails closed on invented path-matte payload");
    auto unboundedPath = pathMatte;
    unboundedPath.operations[1].payloadXml =
        "<NodeParams centerX=\"0.25\" centerY=\"0.75\" width=\"4.01\" height=\"0.2\"/>";
    const bool unboundedCompiled = videowire::compileVisualLayerExecution(
        unboundedPath, execution, error);
    check (! unboundedCompiled
           && error == "visual Path Matte primitive payload is malformed or out of bounds",
           "path-matte execution rejects out-of-bounds primitive payloads");
    auto nestedPath = pathMatte;
    nestedPath.nodeKinds[4] = "visual.shape.union";
    nestedPath.operations[4].kind = "visual.shape.union";
    check (! videowire::compileVisualLayerExecution(nestedPath, execution, error),
           "path-matte execution rejects nested Boolean trees outside the bound");

    auto drawShape = direct;
    drawShape.nodeKinds = { "video.source", "visual.shape.rectangle",
                            "visual.draw.shape", "video.blend", "video.out" };
    drawShape.nodeIds = { 11, 31, 32, 24, 12 };
    drawShape.operations = {
        { 11, "video.source", "source-decode", "" },
        { 31, "visual.shape.rectangle", "control-eval",
          "<NodeParams centerX=\"0.25\" centerY=\"0.75\" width=\"0.5\" height=\"0.2\"/>" },
        { 32, "visual.draw.shape", "native-gpu",
          "<NodeParams red=\"0.1\" green=\"0.2\" blue=\"0.3\" alpha=\"0.4\"/>" },
        { 24, "video.blend", "native-gpu", "" },
        { 12, "video.out", "native-gpu", "" }
    };
    drawShape.edges = { { 11, 0, 24, 0 }, { 31, 0, 32, 0 },
                        { 32, 1, 24, 2 }, { 24, 1, 12, 0 } };
    drawShape.ports = {
        { 11, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 31, 0, 1, "out", "control", "shape", "unspecified", "unspecified" },
        { 32, 0, 1, "in", "control", "shape", "unspecified", "unspecified" },
        { 32, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 24, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 24, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 24, 2, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 12, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    FakeLayer shapeLayer;
    check (videowire::executeVisualLayerPlan({ drawShape }, 7, shapeLayer, error, videohelper::geometry::PlanUse::preview)
           && shapeLayer.drawShape && shapeLayer.drawShapeCx == 0.25f
           && shapeLayer.drawShapeCy == 0.75f && shapeLayer.drawShapeW == 0.5f
           && shapeLayer.drawShapeH == 0.2f && shapeLayer.drawShapeR == 0.1f
           && shapeLayer.drawShapeG == 0.2f && shapeLayer.drawShapeB == 0.3f
           && shapeLayer.drawShapeA == 0.4f,
           "rectangle Draw Shape lowers canonical payload into the shared renderer input");
    inspection = { 7, 0, 32, 1 };
    check (videowire::validateVisualInspectionTarget({ drawShape }, inspection, error)
           && videowire::classifyVisualInspectionTarget(drawShape, inspection)
                == videowire::VisualInspectionSlice::retainedDrawShape,
           "Draw Shape output admits its exact native render-pass boundary");
    FakeLayer inspectedShapeLayer;
    check (videowire::executeVisualLayerPlan({ drawShape }, 7, inspectedShapeLayer, error,videohelper::geometry::PlanUse::preview,
                                             &inspection, nullptr)
           && inspectedShapeLayer.inspectionDrawShapeOutput,
           "executor marks only the exact selected Draw Shape output for retention");
    inspection.outputPort = 0;
    check (! videowire::validateVisualInspectionTarget({ drawShape }, inspection, error),
           "Draw Shape control input cannot alias the retained image output");
    inspection = { 7, 1, 32, 1 };
    FakeLayer staleShapeLayer;
    check (videowire::executeVisualLayerPlan({ drawShape }, 7, staleShapeLayer, error,videohelper::geometry::PlanUse::preview,
                                             &inspection, nullptr)
           && ! staleShapeLayer.inspectionDrawShapeOutput,
           "stale Draw Shape revision never arms backend retention");
    auto drawEllipse = drawShape;
    drawEllipse.nodeKinds[1] = "visual.shape.ellipse";
    drawEllipse.operations[1].kind = "visual.shape.ellipse";
    FakeLayer ellipseLayer;
    check (videowire::executeVisualLayerPlan({ drawEllipse }, 7, ellipseLayer, error, videohelper::geometry::PlanUse::preview)
           && ellipseLayer.drawShape && ellipseLayer.drawShapeEllipse,
           "ellipse Draw Shape lowers to the native shape renderer");

    auto booleanShape = drawShape;
    booleanShape.nodeKinds = { "video.source", "visual.shape.rectangle",
        "visual.shape.ellipse", "visual.shape.subtract", "visual.draw.shape",
        "video.blend", "video.out" };
    booleanShape.nodeIds = { 11, 31, 33, 34, 32, 24, 12 };
    booleanShape.operations = {
        { 11, "video.source", "source-decode", "" },
        { 31, "visual.shape.rectangle", "control-eval",
          "<NodeParams centerX=\"0.4\" centerY=\"0.5\" width=\"0.8\" height=\"0.6\"/>" },
        { 33, "visual.shape.ellipse", "control-eval",
          "<NodeParams centerX=\"0.6\" centerY=\"0.5\" width=\"0.3\" height=\"0.2\"/>" },
        { 34, "visual.shape.subtract", "control-eval", "" },
        { 32, "visual.draw.shape", "native-gpu",
          "<NodeParams red=\"0.1\" green=\"0.2\" blue=\"0.3\" alpha=\"0.4\"/>" },
        { 24, "video.blend", "native-gpu", "" },
        { 12, "video.out", "native-gpu", "" }
    };
    booleanShape.ports = {
        { 11, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 31, 0, 1, "out", "control", "shape", "unspecified", "unspecified" },
        { 33, 0, 1, "out", "control", "shape", "unspecified", "unspecified" },
        { 34, 0, 1, "in", "control", "shape", "unspecified", "unspecified" },
        { 34, 1, 1, "in", "control", "shape", "unspecified", "unspecified" },
        { 34, 2, 1, "out", "control", "shape", "unspecified", "unspecified" },
        { 32, 0, 1, "in", "control", "shape", "unspecified", "unspecified" },
        { 32, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 24, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 24, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 24, 2, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 12, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    booleanShape.edges = { { 11, 0, 24, 0 }, { 31, 0, 34, 0 },
        { 33, 0, 34, 1 }, { 34, 2, 32, 0 }, { 32, 1, 24, 2 }, { 24, 1, 12, 0 } };
    FakeLayer booleanLayer, exportBooleanLayer;
    check (videowire::validateCompiledVisualLayerPlans({}, { booleanShape }, false, error)
           && videowire::compileVisualLayerExecution(booleanShape, execution, error)
           && execution.drawShapePrimaryNodeId == 31
           && execution.drawShapeSecondaryNodeId == 33
           && videowire::executeVisualLayerPlan({ booleanShape }, 7, booleanLayer, error, videohelper::geometry::PlanUse::preview)
           && videowire::executeVisualLayerPlan({ booleanShape }, 7, exportBooleanLayer, error, videohelper::geometry::PlanUse::preview)
           && booleanLayer.drawShape && booleanLayer.drawShapeHasSecondary
           && ! booleanLayer.drawShapeEllipse && booleanLayer.drawShapeSecondaryEllipse
           && booleanLayer.drawShapeOperation == 3
           && booleanLayer.drawShapeCx == 0.4f && booleanLayer.drawShapeW == 0.8f
           && booleanLayer.drawShape2Cx == 0.6f && booleanLayer.drawShape2W == 0.3f
           && booleanLayer.drawShapeOperation == exportBooleanLayer.drawShapeOperation
           && booleanLayer.drawShapeCx == exportBooleanLayer.drawShapeCx
           && booleanLayer.drawShape2Cx == exportBooleanLayer.drawShape2Cx,
           "viewport/export lower the same two distinct immutable primitives into the native Boolean draw pass");

    struct BooleanAliasCase { const char* kind; int operation; const char* message; };
    for (const auto& aliasCase : {
             BooleanAliasCase { "visual.shape.union", 1,
                 "Shape Union admits one primitive identity aliased to both operands" },
             BooleanAliasCase { "visual.shape.intersection", 2,
                 "Shape Intersection admits one primitive identity aliased to both operands" },
             BooleanAliasCase { "visual.shape.subtract", 3,
                 "Shape Subtract admits one primitive identity aliased to both operands" } })
    {
        auto aliasedBoolean = booleanShape;
        aliasedBoolean.nodeKinds.erase(aliasedBoolean.nodeKinds.begin() + 2);
        aliasedBoolean.nodeIds.erase(aliasedBoolean.nodeIds.begin() + 2);
        aliasedBoolean.operations.erase(aliasedBoolean.operations.begin() + 2);
        aliasedBoolean.nodeKinds[2] = aliasCase.kind;
        aliasedBoolean.operations[2].kind = aliasCase.kind;
        aliasedBoolean.ports.erase(std::remove_if(aliasedBoolean.ports.begin(),
            aliasedBoolean.ports.end(), [](const auto& port) { return port.nodeId == 33; }),
            aliasedBoolean.ports.end());
        aliasedBoolean.edges[2].fromNodeId = 31;

        FakeLayer aliasedLayer;
        check (videowire::validateCompiledVisualLayerPlans(
                   {}, { aliasedBoolean }, false, error)
               && videowire::compileVisualLayerExecution(aliasedBoolean, execution, error)
               && execution.drawShapePrimaryNodeId == 31
               && execution.drawShapeSecondaryNodeId == 31
               && videowire::executeVisualLayerPlan(
                   { aliasedBoolean }, 7, aliasedLayer, error, videohelper::geometry::PlanUse::preview)
               && aliasedLayer.drawShape && aliasedLayer.drawShapeHasSecondary
               && aliasedLayer.drawShapeOperation == aliasCase.operation
               && ! aliasedLayer.drawShapeEllipse
               && ! aliasedLayer.drawShapeSecondaryEllipse
               && aliasedLayer.drawShapeCx == aliasedLayer.drawShape2Cx
               && aliasedLayer.drawShapeCy == aliasedLayer.drawShape2Cy
               && aliasedLayer.drawShapeW == aliasedLayer.drawShape2W
               && aliasedLayer.drawShapeH == aliasedLayer.drawShape2H,
               aliasCase.message);
    }

    auto missingPrimitiveBinding = booleanShape;
    missingPrimitiveBinding.ports.erase(std::remove_if(missingPrimitiveBinding.ports.begin(),
        missingPrimitiveBinding.ports.end(), [](const auto& port)
        { return port.nodeId == 33 && port.port == 0; }), missingPrimitiveBinding.ports.end());
    check (! videowire::compileVisualLayerExecution(missingPrimitiveBinding, execution, error)
           && error == "bounded visual DAG has an incompatible typed edge",
           "Boolean Draw Shape rejects a primitive without its exact typed Shape output");

    auto mistypedDrawBinding = booleanShape;
    auto drawInputPort = std::find_if(mistypedDrawBinding.ports.begin(), mistypedDrawBinding.ports.end(),
        [](const auto& port) { return port.nodeId == 32 && port.port == 0; });
    drawInputPort->carrier = "frame";
    check (! videowire::compileVisualLayerExecution(mistypedDrawBinding, execution, error)
           && error == "bounded visual DAG has an incompatible typed edge",
           "Boolean Draw Shape rejects an image-carrier alias for its typed Shape input");

    auto mismatchedDrawOutput = booleanShape;
    auto drawOutputPort = std::find_if(mismatchedDrawOutput.ports.begin(), mismatchedDrawOutput.ports.end(),
        [](const auto& port) { return port.nodeId == 32 && port.port == 1; });
    drawOutputPort->colorSpace = "linearSRGB";
    check (! videowire::compileVisualLayerExecution(mismatchedDrawOutput, execution, error)
           && error == "bounded visual DAG requires RGBA8 sRGB Frame<Image> descriptors",
           "Boolean Draw Shape rejects implicit output color conversion");

    auto outOfBoundsPrimitive = booleanShape;
    outOfBoundsPrimitive.operations[1].payloadXml =
        "<NodeParams centerX=\"0.4\" centerY=\"0.5\" width=\"4.01\" height=\"0.6\"/>";
    check (! videowire::compileVisualLayerExecution(outOfBoundsPrimitive, execution, error)
           && error == "visual Draw Shape primitive payload is malformed or out of bounds",
           "Boolean Draw Shape rejects primitive dimensions outside the graph parameter domain");

    auto malformedPrimitive = booleanShape;
    malformedPrimitive.operations[2].payloadXml =
        "<NodeParams centerX=\"0.6junk\" centerY=\"0.5\" width=\"0.3\" height=\"0.2\"/>";
    check (! videowire::compileVisualLayerExecution(malformedPrimitive, execution, error)
           && error == "visual Draw Shape primitive payload is malformed or out of bounds",
           "Boolean Draw Shape rejects partially parsed primitive values");

    auto nonFinitePrimitive = booleanShape;
    nonFinitePrimitive.operations[2].payloadXml =
        "<NodeParams centerX=\"nan\" centerY=\"0.5\" width=\"0.3\" height=\"0.2\"/>";
    check (! videowire::compileVisualLayerExecution(nonFinitePrimitive, execution, error)
           && error == "visual Draw Shape primitive payload is malformed or out of bounds",
           "Boolean Draw Shape rejects non-finite primitive values");

    auto outOfBoundsColor = booleanShape;
    outOfBoundsColor.operations[4].payloadXml =
        "<NodeParams red=\"0.1\" green=\"0.2\" blue=\"0.3\" alpha=\"1.01\"/>";
    check (! videowire::compileVisualLayerExecution(outOfBoundsColor, execution, error)
           && error == "visual Draw Shape color payload is malformed or out of bounds",
           "Boolean Draw Shape rejects color components outside the graph parameter domain");

    auto inventedBooleanPayload = booleanShape;
    inventedBooleanPayload.operations[3].payloadXml = "<NodeParams invented=\"1\"/>";
    check (! videowire::validateCompiledVisualLayerPlans(
               {}, { inventedBooleanPayload }, false, error)
           && error == "visual Shape Boolean operation payload must be empty",
           "snapshot admission rejects invented Shape Boolean payload keys");
    check (! videowire::compileVisualLayerExecution(inventedBooleanPayload, execution, error)
           && error == "visual Shape Boolean operation payload must be empty",
           "executor rejects invented Shape Boolean payload keys");
    check (! videowire::compileVisualLayerExecutionOrdered(
               inventedBooleanPayload, execution, error)
           && error == "visual Shape Boolean operation payload must be empty",
           "ordered executor rejects invented Shape Boolean payload keys before lowering");

    auto malformedBoolean = booleanShape;
    malformedBoolean.edges.erase(malformedBoolean.edges.begin() + 2);
    check (! videowire::compileVisualLayerExecution(malformedBoolean, execution, error)
           && error == "bounded visual DAG contains a disconnected operation",
           "Boolean Shape admission rejects a missing exact operand binding");

    drawShape.operations[2].backendCapability = "cpu-fallback";
    check (! videowire::compileVisualLayerExecution(drawShape, execution, error),
           "Draw Shape rejects CPU fallback capability");

    auto cpu = direct;
    cpu.operations[1].backendCapability = "cpu-fallback";
    check (! videowire::compileVisualLayerExecution (cpu, execution, error)
           && error == "visual layer plan requires unsupported execution capability",
           "CPU image fallback is rejected");

    auto feedback = direct;
    feedback.structuralRevision = 4;
    feedback.nodeKinds.insert(feedback.nodeKinds.begin() + 1, "visual.feedback");
    feedback.nodeIds.insert(feedback.nodeIds.begin() + 1, 30);
    feedback.operations.insert(feedback.operations.begin() + 1,
        { 30, "visual.feedback", "native-gpu",
          visualtemporaloperation::serialize([]
          {
              visualtemporaloperation::Payload payload;
              payload.mode = visualtemporaloperation::Mode::feedback;
              payload.historyLength = 2;
              payload.decay = 0.95f;
              payload.zoom = 1.05f;
              payload.swirl = -0.05f;
              return payload;
          }()) });
    feedback.ports.push_back({ 30, 0, 1, "in", "frame", "image", "rgba8", "sRGB" });
    feedback.ports.push_back({ 30, 1, 1, "out", "frame", "image", "rgba8", "sRGB" });
    feedback.edges = { { 11, 0, 30, 0 }, { 30, 1, 12, 0 } };
    auto feedbackState = std::make_unique<videowire::VisualPlanExecutionState>();
    check (feedbackState->admitPlans({ feedback }, &error, 640, 360),
           "graph feedback admits one exact bounded native history resource");
    const auto* admittedFeedback = feedbackState->compiled(7, feedback.structuralRevision);
    check (admittedFeedback != nullptr && admittedFeedback->temporalFeedbackPass.has_value()
           && admittedFeedback->temporalFeedbackPass->clipId == 7
           && admittedFeedback->temporalFeedbackPass->nodeId == 30
           && admittedFeedback->temporalFeedbackPass->extent.width == 640
           && admittedFeedback->temporalFeedbackPass->extent.height == 360
           && admittedFeedback->temporalFeedbackPass->format
                == videotemporal::PixelFormat::rgba16f
           && admittedFeedback->temporalFeedbackPass->historyLength == 2
           && admittedFeedback->temporalFeedbackPass->footprint.retainedImages == 2
           && admittedFeedback->temporalFeedbackPass->footprint.historyBytes
                == 640ull * 360ull * 8ull * 2ull,
           "feedback admission transports exact owner, extent, format and footprint");
    check (feedbackState->resourceReceipt().peakLiveFrames == 2
           && feedbackState->resourceReceipt().frameSlots == 2,
           "feedback history contributes to both peak-live and allocated-slot receipts");
    FakeLayer feedbackLayer;
    check (videowire::executeVisualLayerPlan({ feedback }, 7, feedbackLayer, error,videohelper::geometry::PlanUse::preview,
                                             nullptr, nullptr, feedbackState.get(), 1.0)
           && feedbackLayer.effects == &feedbackLayer.graphFeedbackEffect
           && feedbackLayer.effectCount == 1 && feedbackLayer.graphFeedbackEffect.type == 25
           && feedbackLayer.graphTemporalActive && feedbackLayer.graphTemporalNodeId == 30
           && feedbackLayer.graphTemporalPayload.mode == visualtemporaloperation::Mode::feedback
           && feedbackLayer.graphFeedbackEffect.params[0] == 0.95f
           && feedbackLayer.graphFeedbackEffect.params[1] == 1.05f
           && feedbackLayer.graphFeedbackEffect.params[2] == -0.05f
           && feedbackLayer.feedbackHistoryReset && ! feedbackLayer.feedbackHistoryHold,
           "graph feedback lowers validated immutable payload into production FeedbackTrail");
    FakeLayer heldFeedback;
    check (videowire::executeVisualLayerPlan({ feedback }, 7, heldFeedback, error,videohelper::geometry::PlanUse::preview,
                                             nullptr, nullptr, feedbackState.get(), 1.0)
           && heldFeedback.feedbackHistoryHold && ! heldFeedback.feedbackHistoryReset,
           "same-time feedback presentation holds production history");
    FakeLayer resetFeedback;
    check (videowire::executeVisualLayerPlan({ feedback }, 7, resetFeedback, error,videohelper::geometry::PlanUse::preview,
                                             nullptr, nullptr, feedbackState.get(), 0.5)
           && resetFeedback.feedbackHistoryReset && ! resetFeedback.feedbackHistoryHold,
           "backward feedback evaluation resets per-owner production history");
    auto newerWithoutFeedback = feedback;
    ++newerWithoutFeedback.structuralRevision;
    newerWithoutFeedback.operations[1].kind = "visual.shape";
    check(! videowire::visualPlansUseTemporalFeedback(
              { feedback, newerWithoutFeedback }),
          "stale temporal feedback revisions cannot enable export pre-roll");
    auto staleWithoutFeedback = feedback;
    staleWithoutFeedback.operations[1].kind = "visual.shape";
    auto newerFeedback = feedback;
    ++newerFeedback.structuralRevision;
    check(videowire::visualPlansUseTemporalFeedback(
              { staleWithoutFeedback, newerFeedback }),
          "the selected revision alone controls export feedback pre-roll");
    auto equalRevisionWithoutFeedback = feedback;
    equalRevisionWithoutFeedback.operations[1].kind = "visual.shape";
    const std::vector<videowire::CompiledVisualLayerPlan> equalRevisionPlans {
        equalRevisionWithoutFeedback, feedback
    };
    check(! videowire::visualPlansUseTemporalFeedback(equalRevisionPlans)
          && videowire::findVisualLayerPlan(equalRevisionPlans, 7)
                 == &equalRevisionPlans.front(),
          "equal clip revisions consistently preserve the first plan");

    const auto temporalPlan = [&feedback](const char* kind,
                                          const visualtemporaloperation::Payload& payload)
    {
        auto plan = feedback;
        plan.structuralRevision += static_cast<std::uint64_t>(payload.mode) + 10u;
        plan.nodeKinds[1] = kind;
        plan.operations[1].kind = kind;
        plan.operations[1].payloadXml = visualtemporaloperation::serialize(payload);
        return plan;
    };
    const auto checkTemporalMode = [&](const char* kind,
                                       const visualtemporaloperation::Payload& payload)
    {
        auto plan = temporalPlan(kind, payload);
        auto temporalState = std::make_unique<videowire::VisualPlanExecutionState>();
        FakeLayer temporalLayer;
        error.clear();
        if (! temporalState->admitPlans({ plan }, &error, 320, 180))
        {
            std::fprintf(stderr, "temporal admit %s: %s\n", kind, error.c_str());
            return false;
        }
        if (! videowire::executeVisualLayerPlan({ plan }, 7, temporalLayer, error,videohelper::geometry::PlanUse::preview,
                                                nullptr, nullptr, temporalState.get(), 1.0))
        {
            std::fprintf(stderr, "temporal execute %s: %s\n", kind, error.c_str());
            return false;
        }
        return temporalLayer.graphTemporalActive
            && temporalLayer.graphTemporalNodeId == 30
            && temporalLayer.graphTemporalPayload == payload
            && temporalLayer.effects == nullptr
            && temporalState->compiled(7, plan.structuralRevision) != nullptr
            && temporalState->compiled(7, plan.structuralRevision)->temporalFeedbackPass->payload
                 == payload;
    };
    visualtemporaloperation::Payload frameDelayPayload;
    frameDelayPayload.mode = visualtemporaloperation::Mode::frameDelay;
    frameDelayPayload.historyLength = 3;
    frameDelayPayload.decay = 0.0f;
    frameDelayPayload.zoom = 1.0f;
    check(checkTemporalMode("visual.frame-delay", frameDelayPayload),
          "frame delay transports one exact immutable bounded GPU payload");
    visualtemporaloperation::Payload echoPayload;
    echoPayload.mode = visualtemporaloperation::Mode::echo;
    echoPayload.historyLength = 4;
    echoPayload.mix = 0.6f;
    echoPayload.decay = 0.0f;
    echoPayload.zoom = 1.0f;
    check(checkTemporalMode("visual.echo", echoPayload),
          "echo transports one exact immutable bounded GPU payload");
    visualtemporaloperation::Payload stutterPayload;
    stutterPayload.mode = visualtemporaloperation::Mode::stutter;
    stutterPayload.historyLength = 1;
    stutterPayload.holdFrames = 6;
    stutterPayload.decay = 0.0f;
    stutterPayload.zoom = 1.0f;
    check(checkTemporalMode("visual.stutter", stutterPayload),
          "stutter transports one exact immutable bounded GPU payload");
    visualtemporaloperation::Payload exposurePayload;
    exposurePayload.mode = visualtemporaloperation::Mode::longExposure;
    exposurePayload.historyLength = 8;
    exposurePayload.decay = 0.0f;
    exposurePayload.zoom = 1.0f;
    check(checkTemporalMode("visual.long-exposure", exposurePayload),
          "long exposure transports one exact immutable bounded GPU payload");

    auto staleTemporalPlan = temporalPlan("visual.echo", echoPayload);
    auto staleTemporalState = std::make_unique<videowire::VisualPlanExecutionState>();
    check(staleTemporalState->admitPlans({ staleTemporalPlan }, &error, 320, 180),
          "temporal stale-state fixture admits current revision");
    ++staleTemporalPlan.structuralRevision;
    FakeLayer staleTemporalLayer;
    error.clear();
    check(! videowire::executeVisualLayerPlan({ staleTemporalPlan }, 7, staleTemporalLayer, error,videohelper::geometry::PlanUse::preview,
                                              nullptr, nullptr, staleTemporalState.get(), 2.0)
          && error == "render layer has no pre-admitted visual execution",
          "temporal execution rejects stale structural state before native rendering");

    FakeProductRenderer viewportFeedbackRenderer;
    FakeProductRenderer exportFeedbackRenderer;
    FakeLayer viewportFeedbackLayer;
    FakeLayer exportFeedbackLayer;
    auto viewportFeedbackState = std::make_unique<videowire::VisualPlanExecutionState>();
    auto exportFeedbackState = std::make_unique<videowire::VisualPlanExecutionState>();
    check (viewportFeedbackState->admitPlans({ feedback }, &error, 640, 360)
           && exportFeedbackState->admitPlans({ feedback }, &error, 640, 360)
           && videowire::executeVisualLayerPlanForRenderer(
                viewportFeedbackRenderer, { feedback }, 7, viewportFeedbackLayer, error,videohelper::geometry::PlanUse::preview, nullptr,
                nullptr, nullptr, viewportFeedbackState.get(), 1.0)
           && videowire::executeVisualLayerPlanForRenderer(
                exportFeedbackRenderer, { feedback }, 7, exportFeedbackLayer, error,videohelper::geometry::PlanUse::preview, nullptr,
                nullptr, nullptr, exportFeedbackState.get(), 1.0)
           && viewportFeedbackRenderer.temporalFeedbackPreparations == 1
           && exportFeedbackRenderer.temporalFeedbackPreparations == 1
           && viewportFeedbackRenderer.bridgePreparations == 1
           && exportFeedbackRenderer.bridgePreparations == 1
           && viewportFeedbackRenderer.preparedTemporalFeedback.has_value()
           && exportFeedbackRenderer.preparedTemporalFeedback.has_value()
           && viewportFeedbackRenderer.preparedTemporalFeedback->footprint.historyBytes
                == exportFeedbackRenderer.preparedTemporalFeedback->footprint.historyBytes,
           "viewport and export invoke the same pre-admitted native feedback pass");

    FakeProductRenderer missingFeedbackRenderer;
    auto resetHistory=std::make_shared<videorender::ParticleHistorySource>();
    resetHistory->parameterAt=[](const auto&,double seconds,double& value,std::string&)
    { value=seconds>=1.01 && seconds<1.02 ? 1:0; return true; };
    viewportFeedbackLayer.particleHistory=resetHistory;
    viewportFeedbackLayer.particleProjectSeconds=1.0+1.0/30;
    viewportFeedbackLayer.shaderClock.timeDelta=1.0/30;
    check(videowire::executeVisualLayerPlanForRenderer(viewportFeedbackRenderer,{feedback},7,
        viewportFeedbackLayer,error,videohelper::geometry::PlanUse::preview,nullptr,nullptr,nullptr,
        viewportFeedbackState.get(),1.0+1.0/30) && viewportFeedbackLayer.feedbackHistoryReset,
        "temporal reset detects a fixed-step pulse shorter than the display frame");
    check(videowire::executeVisualLayerPlanForRenderer(viewportFeedbackRenderer,{feedback},7,
        viewportFeedbackLayer,error,videohelper::geometry::PlanUse::preview,nullptr,nullptr,nullptr,
        viewportFeedbackState.get(),1.0+1.0/30) && !viewportFeedbackLayer.feedbackHistoryReset,
        "repeating a held temporal frame does not consume its reset twice");
    FakeLayer missingFeedbackLayer;
    error.clear();
    check (! videowire::executeVisualLayerPlanForRenderer(
                missingFeedbackRenderer, { feedback }, 7, missingFeedbackLayer, error, videohelper::geometry::PlanUse::preview, nullptr)
           && missingFeedbackRenderer.temporalFeedbackPreparations == 0
           && missingFeedbackRenderer.bridgePreparations == 0
           && error == "visual temporal operation requires a pre-admitted temporal GPU resource",
           "product execution cannot allocate graph history outside snapshot admission");

    FakeProductRenderer rejectedFeedbackRenderer;
    rejectedFeedbackRenderer.rejectTemporalFeedback = true;
    FakeLayer rejectedFeedbackLayer;
    error.clear();
    check (! videowire::executeVisualLayerPlanForRenderer(
                rejectedFeedbackRenderer, { feedback }, 7, rejectedFeedbackLayer, error,videohelper::geometry::PlanUse::preview, nullptr,
                nullptr, nullptr, viewportFeedbackState.get(), 2.0)
           && rejectedFeedbackRenderer.temporalFeedbackPreparations == 1
           && rejectedFeedbackRenderer.bridgePreparations == 0
           && error == "native temporal feedback backend is unsupported",
           "product execution fails closed when the native feedback backend rejects the pass");

    auto invalidFeedbackPayload = feedback;
    invalidFeedbackPayload.operations[1].payloadXml =
        "<NodeParams decay=\"2\" zoom=\"1.0\" swirl=\"0\"/>";
    check (! videowire::compileVisualLayerExecution(invalidFeedbackPayload, execution, error)
           && error == "visual.feedback payload is malformed or out of bounds",
           "feedback payloads outside the producer domain fail closed instead of clamping");

    auto secondFeedback = feedback;
    secondFeedback.clipId = 8;
    secondFeedback.structuralRevision = 5;
    auto constrainedFeedbackLimits = videowire::VisualBackendResourceLimits::forCanvas(64, 64);
    constrainedFeedbackLimits.frameSlots = 3;
    constrainedFeedbackLimits.liveFrames = 4;
    auto constrainedFeedbackState = std::make_unique<videowire::VisualPlanExecutionState>();
    error.clear();
    check (! constrainedFeedbackState->admitPlans(
                { feedback, secondFeedback }, &error, 64, 64, &constrainedFeedbackLimits)
           && error == "visual graph allocated frame slot capacity exceeded: 4 > 3",
           "retained histories consume the aggregate native frame-slot budget");

    auto viewportState = std::make_unique<videowire::VisualPlanExecutionState>();
    auto exportState = std::make_unique<videowire::VisualPlanExecutionState>();
    auto ownerPlan = directPlan(); ownerPlan.structuralRevision = 4;
    viewportState->admitPlans({ ownerPlan });
    const auto publishTime = [&](videowire::VisualPlanExecutionState& state,
                                 uint64_t revision, double time)
    {
        videowire::TemporalSamplingCommitTransaction transaction;
        visualtemporalsampling::LifecycleState lifecycle;
        std::string transactionError;
        return state.temporalLifecycleSnapshot(7, revision, lifecycle)
            && transaction.add(state, 7, revision, time, std::move(lifecycle), transactionError)
            && transaction.commit(transactionError);
    };
    check (publishTime(*viewportState, 4, 1.0) && viewportState->owner(7)->evaluationSequence == 1,
           "temporal owner starts an explicit evaluation sequence");
    check (publishTime(*viewportState, 4, 1.0) && viewportState->owner(7)->evaluationSequence == 1,
           "same-time presentation does not advance temporal state");
    check (publishTime(*viewportState, 4, 2.0) && viewportState->owner(7)->evaluationSequence == 2,
           "forward evaluation advances exactly once");
    check (publishTime(*viewportState, 4, 0.5) && viewportState->owner(7)->evaluationSequence == 1,
           "backward seek commits a reset with the published frame");
    ownerPlan.structuralRevision = 5;
    viewportState->admitPlans({ ownerPlan });
    exportState->admitPlans({ ownerPlan });
    check (publishTime(*viewportState, 5, 0.5) && viewportState->owner(7)->evaluationSequence == 1,
           "structural revision replacement resets temporal ownership");
    check (publishTime(*exportState, 5, 0.5)
           && publishTime(*viewportState, 5, 1.0)
           && exportState->owner(7)->evaluationSequence == 1,
           "viewport and export owners share semantics without sharing state");
    viewportState->reset(7);
    check (viewportState->owner(7).has_value()
           && viewportState->owner(7)->structuralRevision == 5
           && ! viewportState->owner(7)->hasTime,
           "explicit owner reset publishes an empty state for the admitted revision");

    auto motionBlur = directPlan();
    motionBlur.structuralRevision = 40;
    motionBlur.nodeKinds.insert(motionBlur.nodeKinds.begin() + 1, "visual.motion-blur");
    motionBlur.nodeIds.insert(motionBlur.nodeIds.begin() + 1, 40);
    visualtemporalsampling::Payload motionBlurPayload;
    motionBlurPayload.mode = visualtemporalsampling::Mode::motionBlur;
    motionBlurPayload.sampleCount = 8;
    motionBlurPayload.historyWeight = 0.0f;
    motionBlurPayload.jitterSpread = 0.0f;
    motionBlurPayload.shutterAngleDegrees = 180.0f;
    motionBlurPayload.motionScale = 1.0f;
    motionBlur.operations.insert(motionBlur.operations.begin() + 1,
        { 40, "visual.motion-blur", "temporal-sampling",
          visualtemporalsampling::serialize(motionBlurPayload) });
    motionBlur.ports.push_back({ 40, 0, 1, "in", "frame", "image", "rgba8", "sRGB" });
    motionBlur.ports.push_back({ 40, 1, 2, "in", "frame", "motionVectors", "rg16f", "unspecified" });
    motionBlur.ports.push_back({ 40, 2, 1, "out", "frame", "image", "rgba8", "sRGB" });
    auto motionState = std::make_unique<videowire::VisualPlanExecutionState>();
    check(motionState->admitPlans({ motionBlur }, &error, 320, 180)
          && motionState->compiled(7, 40)->temporalSamplingPass->footprint.transientBytes > 0
          && motionState->resourceReceipt().peakLiveFrames
               == visualtemporalsampling::requirements(motionBlurPayload.mode).retainedHistoryImages
                + visualtemporalsampling::requirements(motionBlurPayload.mode).transientImages
          && motionState->resourceReceipt().frameSlots
               == visualtemporalsampling::requirements(motionBlurPayload.mode).retainedHistoryImages
                + visualtemporalsampling::requirements(motionBlurPayload.mode).transientImages
          && motionState->resourceReceipt().allocatedFrameBytes
               == motionState->compiled(7, 40)->temporalSamplingPass->footprint.historyBytes
                + motionState->compiled(7, 40)->temporalSamplingPass->footprint.transientBytes
          && motionState->telemetry().snapshot().resourcesObserved,
          "motion blur history and transient images publish exact aggregate receipts and telemetry");

    const auto ownerBeforeDiscard = motionState->owner(7);
    videowire::TemporalSamplingCommitTransaction discardedCandidate;
    visualtemporalsampling::LifecycleState discardedLifecycle;
    check(motionState->temporalLifecycleSnapshot(7, 40, discardedLifecycle)
          && discardedCandidate.add(*motionState, 7, 40, 3.0,
                                    std::move(discardedLifecycle), error),
          "a temporal frame candidate is derived from the immutable owner snapshot");
    discardedCandidate.discard();
    const auto ownerAfterDiscard = motionState->owner(7);
    check(ownerBeforeDiscard.has_value() && ownerAfterDiscard.has_value()
          && ownerBeforeDiscard->structuralRevision == ownerAfterDiscard->structuralRevision
          && ownerBeforeDiscard->lastTimeSec == ownerAfterDiscard->lastTimeSec
          && ownerBeforeDiscard->evaluationSequence == ownerAfterDiscard->evaluationSequence
          && ownerBeforeDiscard->hasTime == ownerAfterDiscard->hasTime,
          "discarding preparation leaves durable temporal owner bytes unchanged");

    videowire::TemporalSamplingCommitTransaction resetRaceCandidate;
    visualtemporalsampling::LifecycleState resetRaceLifecycle;
    check(motionState->temporalLifecycleSnapshot(7, 40, resetRaceLifecycle)
          && resetRaceCandidate.add(*motionState, 7, 40, 4.0,
                                    std::move(resetRaceLifecycle), error),
          "reset-race candidate prepares before reset publication");
    std::mutex resetGateMutex;
    std::condition_variable resetGate;
    bool releaseReset = false;
    bool resetFinished = false;
    std::thread resetThread([&]
    {
        std::unique_lock<std::mutex> lock(resetGateMutex);
        resetGate.wait(lock, [&] { return releaseReset; });
        lock.unlock();
        motionState->reset(7);
        lock.lock();
        resetFinished = true;
        lock.unlock();
        resetGate.notify_one();
    });
    {
        std::lock_guard<std::mutex> lock(resetGateMutex);
        releaseReset = true;
    }
    resetGate.notify_one();
    {
        std::unique_lock<std::mutex> lock(resetGateMutex);
        resetGate.wait(lock, [&] { return resetFinished; });
    }
    resetThread.join();
    check(! resetRaceCandidate.commit(error)
          && error == "temporal sampling snapshot changed before frame publication"
          && motionState->owner(7)->structuralRevision == 40
          && ! motionState->owner(7)->hasTime,
          "concurrent reset invalidates the whole candidate with zero candidate mutation");

    auto secondMotionBlur = motionBlur;
    secondMotionBlur.clipId = 8;
    auto atomicState = std::make_unique<videowire::VisualPlanExecutionState>();
    check(atomicState->admitPlans({ motionBlur, secondMotionBlur }, &error, 320, 180),
          "two-owner temporal fixture admits production plans");
    videowire::TemporalSamplingCommitTransaction atomicFrame;
    visualtemporalsampling::LifecycleState firstAtomicLifecycle, secondAtomicLifecycle;
    check(atomicState->temporalLifecycleSnapshot(7, 40, firstAtomicLifecycle)
          && atomicState->temporalLifecycleSnapshot(8, 40, secondAtomicLifecycle)
          && atomicFrame.add(*atomicState, 7, 40, 1.0, std::move(firstAtomicLifecycle), error)
          && atomicFrame.add(*atomicState, 8, 40, 1.0, std::move(secondAtomicLifecycle), error),
          "multi-candidate frame prepares as one transaction");
    atomicState->reset(8);
    check(! atomicFrame.commit(error)
          && ! atomicState->owner(7)->hasTime && ! atomicState->owner(8)->hasTime,
          "one stale candidate rejects the complete frame without partial publication");

    bool promotedPublication = false;
    bool discardedPublication = false;
    videowire::TemporalSamplingCommitTransaction rejectedPublication;
    rejectedPublication.addPublication(
        [](std::string& value)
        { value = "injected presentation rejection"; return false; },
        [&] { promotedPublication = true; },
        [&] { discardedPublication = true; });
    check(! rejectedPublication.commit(error) && ! promotedPublication
          && discardedPublication,
          "failed production publication validation promotes nothing and discards its candidate");
    bool teardownDiscarded = false;
    {
        videowire::TemporalSamplingCommitTransaction abandonedPublication;
        abandonedPublication.addPublication(
            [](std::string&) { return true; }, [] {},
            [&] { teardownDiscarded = true; });
    }
    check(teardownDiscarded,
          "abandoned production publication releases its candidate during teardown");
    videowire::injectTemporalPublicationFailure(
        videowire::TemporalPublicationFailurePoint::exportEncoder);
    check(! videowire::consumeTemporalPublicationFailure(
              videowire::TemporalPublicationFailurePoint::viewportPresentation)
          && videowire::consumeTemporalPublicationFailure(
              videowire::TemporalPublicationFailurePoint::exportEncoder),
          "production publication failure injection is deterministic and stage-specific");

    videowire::TemporalSamplingCommitTransaction destroyedOwnerCandidate;
    {
        auto doomedState = std::make_unique<videowire::VisualPlanExecutionState>();
        check(doomedState->admitPlans({ motionBlur }, &error, 320, 180),
              "owner-destruction fixture admits a production plan");
        visualtemporalsampling::LifecycleState doomedLifecycle;
        check(doomedState->temporalLifecycleSnapshot(7, 40, doomedLifecycle)
              && destroyedOwnerCandidate.add(*doomedState, 7, 40, 1.0,
                                             std::move(doomedLifecycle), error),
              "owner-destruction fixture prepares a weak-safe candidate");
    }
    check(! destroyedOwnerCandidate.commit(error)
          && error == "temporal sampling owner was destroyed before frame publication",
          "destroyed owners cannot be dereferenced or published");

    auto pipelineState = std::make_unique<videowire::VisualPlanExecutionState>();
    check(pipelineState->admitPlans({ motionBlur }, &error, 320, 180),
          "pipeline fixture admits a production plan");
    videowire::TemporalSamplingCommitTransaction predecessor;
    visualtemporalsampling::LifecycleState predecessorLifecycle;
    check(pipelineState->temporalLifecycleSnapshot(7, 40, predecessorLifecycle)
          && predecessor.add(*pipelineState, 7, 40, 1.0,
                             std::move(predecessorLifecycle), error),
          "pipeline predecessor prepares");
    videowire::TemporalSamplingCommitTransaction descendant;
    descendant.seedFrom(predecessor);
    visualtemporalsampling::LifecycleState descendantLifecycle;
    check(pipelineState->temporalLifecycleSnapshot(7, 40, descendantLifecycle)
          && descendant.add(*pipelineState, 7, 40, 2.0,
                            std::move(descendantLifecycle), error),
          "pipeline descendant chains from the uncommitted immutable predecessor");
    check(! descendant.commit(error)
          && error == "temporal sampling predecessor was not committed"
          && ! pipelineState->owner(7)->hasTime,
          "out-of-order descendant publication fails without poisoning the owner");
    check(predecessor.commit(error) && pipelineState->owner(7)->evaluationSequence == 1,
          "a rejected descendant does not poison its predecessor");

    videowire::TemporalSamplingExecution firstSampling;
    check(motionState->prepareTemporalSamplingExecution(7, 40, 1.0, 1, false,
              visualtemporalsampling::EvaluationMode::preview, firstSampling, error)
          && firstSampling.pass.clipId == 7 && firstSampling.pass.nodeId == 40
          && firstSampling.pass.structuralRevision == 40
          && firstSampling.pass.payload == motionBlurPayload
          && firstSampling.point.ownerId == 7 && firstSampling.point.structuralRevision == 40
          && firstSampling.point.width == 320 && firstSampling.point.height == 180
          && firstSampling.point.helperGeneration == 1 && firstSampling.point.timeSec == 1.0
          && firstSampling.point.mode == visualtemporalsampling::EvaluationMode::preview
          && firstSampling.transition.action
               == visualtemporalsampling::TransitionAction::resetAndAdvance
          && firstSampling.transition.resetCause
               == visualtemporalsampling::ResetCause::firstEvaluation,
          "motion blur publishes one complete immutable compositor request");
    const auto publishedSampling = firstSampling;
    visualtemporalsampling::Transition samplingTransition;
    check(motionState->evaluateTemporalSampling(7, 40, 1.0, 1, false,
              visualtemporalsampling::EvaluationMode::preview, samplingTransition, error)
          && samplingTransition.action == visualtemporalsampling::TransitionAction::hold
          && motionState->evaluateTemporalSampling(7, 40, 2.0, 1, false,
              visualtemporalsampling::EvaluationMode::preview, samplingTransition, error)
          && samplingTransition.action == visualtemporalsampling::TransitionAction::advance
          && motionState->evaluateTemporalSampling(7, 40, 0.5, 1, false,
              visualtemporalsampling::EvaluationMode::preview, samplingTransition, error)
          && samplingTransition.action == visualtemporalsampling::TransitionAction::resetAndAdvance,
          "motion blur snapshot lifecycle resets, holds, and advances deterministically");
    check(publishedSampling.point.timeSec == 1.0
          && publishedSampling.transition.action
               == visualtemporalsampling::TransitionAction::resetAndAdvance,
          "later lifecycle evaluations cannot rewrite a published compositor request");
    check(motionState->admitPlans({ motionBlur }, &error, 320, 180)
          && motionState->evaluateTemporalSampling(7, 40, 0.5, 1, false,
              visualtemporalsampling::EvaluationMode::preview, samplingTransition, error)
          && samplingTransition.action == visualtemporalsampling::TransitionAction::hold,
          "readmitting the same immutable plan preserves its temporal sampling lifecycle");
    auto revisedMotionBlur = motionBlur;
    revisedMotionBlur.structuralRevision = 41;
    videowire::TemporalSamplingExecution revisedSampling;
    check(motionState->admitPlans({ revisedMotionBlur }, &error, 320, 180)
          && motionState->prepareTemporalSamplingExecution(7, 41, 0.5, 1, false,
              visualtemporalsampling::EvaluationMode::preview, revisedSampling, error)
          && revisedSampling.pass.structuralRevision == 41
          && revisedSampling.transition.action
               == visualtemporalsampling::TransitionAction::resetAndAdvance
          && revisedSampling.transition.resetCause
               == visualtemporalsampling::ResetCause::structuralRevision,
          "a replacement snapshot reports its structural-revision reset through the shared owner");
    videowire::TemporalSamplingExecution rejectedSampling = publishedSampling;
    rejectedSampling.point.ownerId = 99;
    check(! motionState->prepareTemporalSamplingExecution(7, 41, 0.5, 1, false,
              visualtemporalsampling::EvaluationMode::exportFrame, rejectedSampling, error)
          && error == "temporal sampling lifecycle rejected the evaluation point"
          && rejectedSampling.point.ownerId == 99,
          "mixed preview/export evaluation fails closed without publishing partial state");
    auto exportMotionState = std::make_unique<videowire::VisualPlanExecutionState>();
    videowire::TemporalSamplingExecution exportSampling;
    check(exportMotionState->admitPlans({ motionBlur }, &error, 320, 180)
          && exportMotionState->prepareTemporalSamplingExecution(7, 40, 1.0, 1, false,
              visualtemporalsampling::EvaluationMode::exportFrame, exportSampling, error)
          && exportSampling.pass.payload == publishedSampling.pass.payload
          && exportSampling.transition.action
               == visualtemporalsampling::TransitionAction::resetAndAdvance,
          "preview and export consume the same admitted request through isolated lifecycle owners");
    FakeLayer unconnectedMotionLayer;
    check(! videowire::executeVisualLayerPlan({ revisedMotionBlur }, 7, unconnectedMotionLayer, error,videohelper::geometry::PlanUse::preview,
                                               nullptr, nullptr, motionState.get(), 1.0)
          && error == "visual.motion-blur sampling is admitted but not connected to native viewport/export composition",
          "motion blur sampling fails closed before claiming compositor execution");
    FakeLayer connectedMotionLayer;
    FakeProductRenderer connectedMotionRenderer;
    const videowire::VisualPlanEvaluationContext previewMotionContext {
        7, true, visualtemporalsampling::EvaluationMode::preview
    };
    videowire::TemporalSamplingCommitTransaction previewMotionTransaction;
    check(videowire::executeVisualLayerPlanForRenderer(
              connectedMotionRenderer, { revisedMotionBlur }, 7,
              connectedMotionLayer, error, videohelper::geometry::PlanUse::preview,
              &previewMotionTransaction, nullptr, nullptr, motionState.get(), 1.0,
              nullptr, nullptr, &previewMotionContext)
          && connectedMotionRenderer.motionBlurPreparations == 1
          && connectedMotionRenderer.preparedMotionBlur.has_value()
          && connectedMotionRenderer.preparedMotionBlur->point.helperGeneration == 7
          && connectedMotionRenderer.preparedMotionBlur->point.paused
          && connectedMotionRenderer.preparedMotionBlur->point.mode
               == visualtemporalsampling::EvaluationMode::preview
          && connectedMotionRenderer.preparedMotionBlur->transition.action
               == visualtemporalsampling::TransitionAction::resetAndAdvance
          && connectedMotionLayer.graphMotionBlurActive
          && connectedMotionLayer.graphMotionTexture == 9
          && connectedMotionLayer.graphMotionBlur == motionBlurPayload,
          "motion blur prepares a publication-owned lifecycle candidate");
    check(previewMotionTransaction.commit(error),
          "preview publication commits its prepared temporal candidate");
    FakeLayer heldMotionLayer;
    videowire::TemporalSamplingCommitTransaction heldMotionTransaction;
    check(videowire::executeVisualLayerPlanForRenderer(
              connectedMotionRenderer, { revisedMotionBlur }, 7,
              heldMotionLayer, error, videohelper::geometry::PlanUse::preview,
              &heldMotionTransaction, nullptr, nullptr, motionState.get(), 1.0,
              nullptr, nullptr, &previewMotionContext)
          && connectedMotionRenderer.preparedMotionBlur->transition.action
               == visualtemporalsampling::TransitionAction::hold,
          "paused viewport motion blur holds its production temporal lifecycle");
    heldMotionTransaction.discard();
    const videowire::VisualPlanEvaluationContext playingPreviewMotionContext {
        7, false, visualtemporalsampling::EvaluationMode::preview
    };
    connectedMotionRenderer.rejectFlatShaderBridge = true;
    videowire::TemporalSamplingCommitTransaction rejectedMotionTransaction;
    check(! videowire::executeVisualLayerPlanForRenderer(
              connectedMotionRenderer, { revisedMotionBlur }, 7,
              heldMotionLayer, error, videohelper::geometry::PlanUse::preview,
              &rejectedMotionTransaction, nullptr, nullptr, motionState.get(), 2.0,
              nullptr, nullptr, &playingPreviewMotionContext),
          "rejected downstream frame preparation does not publish a temporal transition");
    check(rejectedMotionTransaction.empty(),
          "rejected preparation leaves no temporal candidate to commit");
    connectedMotionRenderer.rejectFlatShaderBridge = false;
    videowire::TemporalSamplingCommitTransaction retryMotionTransaction;
    check(videowire::executeVisualLayerPlanForRenderer(
              connectedMotionRenderer, { revisedMotionBlur }, 7,
              heldMotionLayer, error, videohelper::geometry::PlanUse::preview,
              &retryMotionTransaction, nullptr, nullptr, motionState.get(), 2.0,
              nullptr, nullptr, &playingPreviewMotionContext)
          && connectedMotionRenderer.preparedMotionBlur->transition.action
               == visualtemporalsampling::TransitionAction::advance,
          "retrying a rejected frame advances from the last committed temporal point");
    check(retryMotionTransaction.commit(error),
          "successful retry commits only at simulated publication");
    const videowire::VisualPlanEvaluationContext exportMotionContext {
        1, false, visualtemporalsampling::EvaluationMode::exportFrame
    };
    FakeLayer isolatedExportMotionLayer;
    FakeProductRenderer isolatedExportMotionRenderer;
    videowire::TemporalSamplingCommitTransaction exportMotionTransaction;
    check(videowire::executeVisualLayerPlanForRenderer(
              isolatedExportMotionRenderer, { motionBlur }, 7,
              isolatedExportMotionLayer, error, videohelper::geometry::PlanUse::exportRender,
              &exportMotionTransaction, nullptr, nullptr, exportMotionState.get(), 2.0,
              nullptr, nullptr, &exportMotionContext)
          && isolatedExportMotionRenderer.preparedMotionBlur.has_value()
          && isolatedExportMotionRenderer.preparedMotionBlur->point.mode
               == visualtemporalsampling::EvaluationMode::exportFrame
          && isolatedExportMotionRenderer.preparedMotionBlur->transition.action
               == visualtemporalsampling::TransitionAction::advance,
          "export motion blur prepares through its isolated production lifecycle");
    exportMotionTransaction.discard();
    auto tinyMotionLimits = videowire::VisualBackendResourceLimits::forCanvas(320, 180);
    tinyMotionLimits.allocatedFrameBytes = 1;
    auto rejectedMotionState = std::make_unique<videowire::VisualPlanExecutionState>();
    check(! rejectedMotionState->admitPlans({ motionBlur }, &error, 320, 180, &tinyMotionLimits)
          && error == "temporal sampling transient byte capacity exceeded",
          "motion blur snapshot rejects an insufficient transient image budget");

    auto history = direct;
    history.nodeKinds.insert(history.nodeKinds.begin() + 1, "control.history");
    history.nodeIds.insert(history.nodeIds.begin() + 1, 31);
    history.operations.insert(history.operations.begin() + 1,
                              { 31, "control.history", "control-eval", "" });
    check (! videowire::compileVisualLayerExecution(history, execution, error)
           && error == "control.history temporal evaluation is not yet connected to native viewport/export image execution",
           "temporal operation retains a dedicated unsupported production diagnostic");

    auto full = direct;
    full.nodeKinds = { "video.source", "video.transform", "video.effects",
                       "video.mask.shape", "video.blend", "video.out" };
    full.nodeIds = { 11, 21, 22, 23, 24, 12 };
    full.operations.clear();
    full.edges.clear();
    full.ports.clear();
    for (size_t i = 0; i < full.nodeIds.size(); ++i)
    {
        full.operations.push_back ({ full.nodeIds[i], full.nodeKinds[i],
            i == 0 ? "source-decode" : "native-gpu", "" });
        if (i + 1 < full.nodeIds.size())
            full.edges.push_back ({ full.nodeIds[i], 0, full.nodeIds[i + 1], 0 });
    }
    FakeLayer preserved;
    check (videowire::executeVisualLayerPlan ({ full }, 7, preserved, error, videohelper::geometry::PlanUse::preview)
           && preserved.scale == 2.0f && preserved.effects != nullptr
           && preserved.maskType == 2 && preserved.opacity == 0.4f,
           "supported full production chain preserves existing rendering inputs");

    FakeLayer legacy;
    auto legacyPlan = direct;
    legacyPlan.nodeKinds = {
        "video.legacy.source", "video.legacy.retime", "video.legacy.transform",
        "video.legacy.effects", "video.out"
    };
    legacyPlan.nodeIds.clear();
    legacyPlan.ports.clear();
    legacyPlan.edges.clear();
    legacyPlan.operations.clear();
    check (videowire::executeVisualLayerPlan ({ legacyPlan }, 7, legacy, error, videohelper::geometry::PlanUse::preview)
           && legacy.scale == 2.0f && legacy.effects != nullptr && legacy.maskType == 2,
           "fixed legacy plan preserves the complete established renderer chain");

    FakeLayer noPlan;
    check (videowire::executeVisualLayerPlan ({}, 7, noPlan, error, videohelper::geometry::PlanUse::preview)
           && noPlan.scale == 2.0f && noPlan.effects != nullptr,
           "snapshots without compiled plans retain legacy production rendering");

    // Dependency-light production-path telemetry coverage.
    auto measuredPlan = directPlan();
    measuredPlan.structuralRevision = 9;
    auto measuredState = std::make_unique<videowire::VisualPlanExecutionState>();
    videowire::VisualTelemetryPlanAdmission measuredAdmission;
    measuredAdmission.clipId = 7; measuredAdmission.structuralRevision = 9;
    measuredAdmission.nodeCount = 2; measuredAdmission.stableNodeIds[0] = 12;
    measuredAdmission.stableNodeIds[1] = 11; measuredAdmission.intermediateImageCount = 1;
    measuredState->telemetry().admitPlans({ measuredAdmission });
    measuredState->admitPlans({ measuredPlan });
    FakeLayer measuredLayerA, measuredLayerB;
    check (videowire::executeVisualLayerPlan({ measuredPlan }, 7, measuredLayerA, error,videohelper::geometry::PlanUse::preview,
                                             nullptr, nullptr, measuredState.get(), 1.0)
           && videowire::executeVisualLayerPlan({ measuredPlan }, 7, measuredLayerB, error,videohelper::geometry::PlanUse::preview,
                                                nullptr, nullptr, measuredState.get(), 2.0),
           "telemetry fixture executes two monotonic evaluations without sleeps");
    auto measured = measuredState->telemetry().snapshot();
    check (measured.graphEvaluations == 2 && measured.graphMeasuredTotalNs > 0
           && measured.planCacheMisses == 1 && measured.planCacheHits == 2
           && measured.planInstalls == 1,
           "graph accounting and plan lowering/install cache hit/miss are measured");
    check (measured.layers.size() == 1 && measured.nodes.size() == 2
           && measured.layers[0].clipId == 7 && measured.layers[0].structuralRevision == 9
           && measured.nodes[0].clipId == 7 && measured.nodes[0].structuralRevision == 9
           && measured.nodes[0].stableNodeId == 11 && measured.nodes[1].stableNodeId == 12,
           "telemetry preserves exact clip/revision/stable-node identities in deterministic order");
    FakeLayer seekLayer;
    check (videowire::executeVisualLayerPlan({ measuredPlan }, 7, seekLayer, error,videohelper::geometry::PlanUse::preview,
                                             nullptr, nullptr, measuredState.get(), 0.5),
           "backward seek re-evaluates the current plan");
    measured = measuredState->telemetry().snapshot();
    check (measured.graphEvaluations == 1 && measured.planCacheMisses == 0
           && measured.planCacheHits == 1 && measured.layers[0].evaluations == 1,
           "backward seek resets per-revision costs and compile cache accounting");
    auto revisedPlan = measuredPlan;
    revisedPlan.structuralRevision = 10;
    measuredAdmission.structuralRevision = 10;
    measuredState->telemetry().admitPlans({ measuredAdmission });
    measuredState->admitPlans({ revisedPlan });
    measured = measuredState->telemetry().snapshot();
    check (measured.graphEvaluations == 0 && measured.layers.size() == 1
           && measured.layers[0].structuralRevision == 10 && measured.layers[0].evaluations == 0
           && measured.nodes.size() == 2 && measured.nodes[0].evaluations == 0,
           "admitting a new revision immediately replaces stale identities with zeroed slots");
    FakeLayer revisedLayer;
    check (videowire::executeVisualLayerPlan({ revisedPlan }, 7, revisedLayer, error,videohelper::geometry::PlanUse::preview,
                                             nullptr, nullptr, measuredState.get(), 3.0),
           "new structural revision installs independently");
    measured = measuredState->telemetry().snapshot();
    check (measured.layers.size() == 1 && measured.layers[0].structuralRevision == 10
           && measured.nodes.size() == 2 && measured.nodes[0].structuralRevision == 10,
           "revision change removes stale identities");

    videowire::VisualPlanTelemetry bounded;
    videowire::VisualTelemetryPlanAdmission exact;
    exact.clipId = 2; exact.structuralRevision = 1; exact.nodeCount = 8;
    exact.intermediateImageCount = 7;
    for (int i = 0; i < 8; ++i) exact.stableNodeIds[(size_t) i] = 8 - i;
    auto duplicate = exact; duplicate.structuralRevision = 0;
    auto stale = exact; stale.clipId = 1; stale.structuralRevision = 1;
    check (bounded.admitPlans({ exact, duplicate, stale }), "exact 8-node/7-image plans admit");
    bounded.recordEvaluation(2, 1, 100, false);
    check(bounded.recordNodeEvaluation(2, 1, 8, 40)
          && bounded.recordNodeEvaluation(2, 1, 8, 60)
          && ! bounded.recordNodeEvaluation(2, 1, 999, 10),
          "node costs accept only measured samples for admitted stable identities");
    bounded.recordEvaluation(99, 1, 100, false);
    const auto boundedSnapshot = bounded.snapshot();
    check (boundedSnapshot.layers.size() == 2 && boundedSnapshot.nodes.size() == 16
           && boundedSnapshot.layers[0].clipId == 1 && boundedSnapshot.layers[1].clipId == 2
           && boundedSnapshot.nodes[8].stableNodeId == 1,
           "duplicate identities normalize and fixed-capacity snapshot ordering is deterministic");

    auto expandedTelemetry = exact; expandedTelemetry.clipId = 3;
    expandedTelemetry.nodeCount = 20;
    expandedTelemetry.intermediateImageCount = 19;
    for (size_t index = 0; index < expandedTelemetry.nodeCount; ++index)
        expandedTelemetry.stableNodeIds[index] = static_cast<int>(index + 1);
    std::string admissionError;
    check (bounded.admitPlans({ exact, expandedTelemetry }, &admissionError)
           && bounded.snapshot().rejectedPlans == 0
           && ! bounded.snapshot().executableNodeSubsetTruncated,
           "telemetry preserves graphs above the former sixteen-operation ceiling without truncation");

    auto largerComposite = composite;
    largerComposite.nodeKinds = { "video.source", "video.transform", "video.effects",
        "video.mask.shape", "visual.shape.rectangle", "visual.draw.shape",
        "video.layer.source", "video.blend", "video.out" };
    largerComposite.nodeIds = { 11, 21, 22, 23, 31, 32, 99, 24, 12 };
    largerComposite.operations = {
        { 11, "video.source", "source-decode", "" },
        { 21, "video.transform", "native-gpu", "" },
        { 22, "video.effects", "native-gpu", "" },
        { 23, "video.mask.shape", "native-gpu", "" },
        { 31, "visual.shape.rectangle", "control-eval", "" },
        { 32, "visual.draw.shape", "native-gpu", "" },
        { 99, "video.layer.source", "source-decode", "" },
        { 24, "video.blend", "native-gpu", "" },
        { 12, "video.out", "native-gpu", "" }
    };
    largerComposite.edges = {
        { 11, 0, 21, 0 }, { 21, 1, 22, 0 }, { 22, 1, 23, 0 },
        { 23, 1, 24, 0 }, { 31, 0, 32, 0 }, { 32, 1, 24, 2 },
        { 99, 0, 24, 3 }, { 24, 1, 12, 0 }
    };
    largerComposite.ports = {
        { 11, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 21, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 21, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 22, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 22, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 23, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 23, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 31, 0, 1, "out", "control", "shape", "unspecified", "unspecified" },
        { 32, 0, 1, "in", "control", "shape", "unspecified", "unspecified" },
        { 32, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 99, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 24, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 24, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 24, 2, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 24, 3, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 12, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    FakeLayer largerLayer;
    const auto largerAdmission = videowire::makeVisualTelemetryAdmission(largerComposite);
    check (largerComposite.operations.size() > 8 && ! largerAdmission.nodesTruncated
           && videowire::executeVisualLayerPlan({ largerComposite }, 7, largerLayer, error, videohelper::geometry::PlanUse::preview),
           "larger exact typed graph remains executable without telemetry truncation");

    const auto beforeHold = measuredState->telemetry().snapshot();
    FakeLayer heldMeasured;
    check (videowire::executeVisualLayerPlan({ revisedPlan }, 7, heldMeasured, error,videohelper::geometry::PlanUse::preview,
                                             nullptr, nullptr, measuredState.get(), 3.0),
           "paused hold executes");
    const auto afterHold = measuredState->telemetry().snapshot();
    check (afterHold.graphEvaluations == beforeHold.graphEvaluations
           && afterHold.graphMeasuredTotalNs == beforeHold.graphMeasuredTotalNs
           && afterHold.graphMeasuredMovingNs == beforeHold.graphMeasuredMovingNs
           && afterHold.planCacheHits == beforeHold.planCacheHits
           && afterHold.planCacheMisses == beforeHold.planCacheMisses
           && afterHold.planInstalls == beforeHold.planInstalls
           && afterHold.lastPlanLoweringNs == beforeHold.lastPlanLoweringNs,
           "paused hold freezes the complete telemetry cache and evaluation contract");

    videowire::VisualPlanTelemetry contended;
    contended.admitPlans({ exact });
    {
        auto heldLock = contended.lockForTesting();
        contended.recordEvaluation(2, 1, 10, false);
        contended.recordNodeEvaluation(2, 1, 8, 10);
        contended.resetOwner(2, 1);
        contended.recordPlanLowering(false, 123, true);
    }
    const auto contentionSnapshot = contended.snapshot();
    check (contentionSnapshot.droppedSamples == 3 && contentionSnapshot.planCacheMisses == 1
           && contentionSnapshot.planInstalls == 1 && contentionSnapshot.lastPlanLoweringNs == 123,
           "contention drops evaluation but preserves first post-reset plan-lowering/install sample");

    videowire::VisualPlanTelemetry staleContended;
    staleContended.admitPlans({ exact });
    {
        auto heldLock = staleContended.lockForTesting();
        staleContended.recordPlanLowering(false, 456, true);
        staleContended.resetOwner(2, 1);
    }
    const auto staleContention = staleContended.snapshot();
    check (staleContention.planCacheMisses == 0 && staleContention.planInstalls == 0
           && staleContention.lastPlanLoweringNs == 0,
           "reset generation discards a deferred pre-reset lowering sample");
    {
        auto heldLock = staleContended.lockForTesting();
        staleContended.recordPlanLowering(false, 0, true);
    }
    const auto zeroDuration = staleContended.snapshot();
    check (zeroDuration.planCacheMisses == 1 && zeroDuration.planInstalls == 1
           && zeroDuration.lastPlanLoweringNs == 0,
           "explicit pending presence preserves a legitimate zero-duration lowering sample");

    videowire::VisualPlanTelemetry mixedGeneration;
    mixedGeneration.admitPlans({ exact });
    {
        auto heldLock = mixedGeneration.lockForTesting();
        mixedGeneration.recordPlanLowering(false, 41, true);
        mixedGeneration.resetOwner(2, 1);
        mixedGeneration.recordPlanLowering(true, 0, false);
    }
    const auto mixedSnapshot = mixedGeneration.snapshot();
    check (mixedSnapshot.planCacheMisses == 0 && mixedSnapshot.planInstalls == 0
           && mixedSnapshot.planCacheHits == 1 && mixedSnapshot.lastPlanLoweringNs == 0,
           "one drain rejects pre-reset pending data while preserving post-reset data");

    auto filteredPlan = directPlan();
    filteredPlan.operations.insert(filteredPlan.operations.begin() + 1,
        { 90, "control.constant", "control-eval", "" });
    filteredPlan.operations.insert(filteredPlan.operations.begin() + 2,
        { 91, "visual.depth.asset", "parked-metadata", "" });
    const auto filteredAdmission = videowire::makeVisualTelemetryAdmission(filteredPlan);
    check (filteredAdmission.executableNodeTotal == 2 && filteredAdmission.nodeCount == 2
           && filteredAdmission.stableNodeIds[0] == 11
           && filteredAdmission.stableNodeIds[1] == 12 && ! filteredAdmission.nodesTruncated,
           "telemetry reports only executable source-decode/native-gpu operations");

    FakeLayer invariantLayer;
    const std::vector<videowire::CompiledVisualLayerPlan> invariantPlans { revisedPlan };
    const auto mapSizeBefore = invariantLayer.genParams.size();
    const auto allocationsBefore = allocations.load(std::memory_order_relaxed);
    check (videowire::executeVisualLayerPlan(invariantPlans, 7, invariantLayer, error,videohelper::geometry::PlanUse::preview,
                                             nullptr, nullptr, measuredState.get(), 4.0),
           "pre-admitted execution renders through fixed lookup slots");
    check (allocations.load(std::memory_order_relaxed) == allocationsBefore
           && invariantLayer.genParams.size() == mapSizeBefore,
           "pre-admitted render lookup performs no allocation or map insertion");

    contended.seedForSaturationTesting(std::numeric_limits<uint64_t>::max());
    contended.recordEvaluation(2, 1, std::numeric_limits<uint64_t>::max(), false);
    contended.recordNodeEvaluation(2, 1, 1, 1);
    contended.recordPlanLowering(true, 0, false);
    contended.recordRuntime(1, 1, 1, 1, videowire::VisualTransportMode::zeroCopy, true);
    contended.recordZeroCopyAllocationBytes(1); contended.recordReadbackCopiedBytes(1);
    contended.recordDrop(videowire::VisualDropReason::noBuffer);
    contended.recordDimensions(2, 2, 1, 1);
    const auto saturated = contended.snapshot();
    check (saturated.graphEvaluations == std::numeric_limits<uint64_t>::max()
           && saturated.graphMeasuredTotalNs == std::numeric_limits<uint64_t>::max()
           && saturated.nodes[0].evaluations == std::numeric_limits<uint64_t>::max()
           && saturated.nodes[0].measuredTotalNs == std::numeric_limits<uint64_t>::max()
           && saturated.planCacheHits == std::numeric_limits<uint64_t>::max()
           && saturated.compositor.count == std::numeric_limits<uint64_t>::max()
           && saturated.zeroCopyFrames == std::numeric_limits<uint64_t>::max()
           && saturated.zeroCopyAllocationBytes == std::numeric_limits<uint64_t>::max()
           && saturated.readbackCopiedBytes == std::numeric_limits<uint64_t>::max()
           && saturated.framesDropped == std::numeric_limits<uint64_t>::max()
           && saturated.dimensionMismatchCount == std::numeric_limits<uint64_t>::max(),
           "all telemetry counters and measured durations saturate");
    const auto telemetryJson = videowire::visualTelemetryJson(boundedSnapshot);
    videowire::VisualPlanTelemetry runtime;
    const auto unseenJson = videowire::visualTelemetryJson(runtime.snapshot());
    check(! unseenJson["compositor"]["available"].get<bool>() && unseenJson["compositor"]["movingNs"].is_null()
          && ! unseenJson["transport"]["available"].get<bool>() && unseenJson["transport"]["mode"].is_null()
          && ! unseenJson["backend"]["available"].get<bool>() && unseenJson["backend"]["current"].is_null()
          && ! unseenJson["resources"]["available"].get<bool>() && unseenJson["resources"]["retainedFramesCurrent"].is_null()
          && ! unseenJson["dimensions"]["available"].get<bool>() && unseenJson["dimensions"]["actualWidth"].is_null(),
          "unseen runtime families emit available false and null values, never production-looking zeroes");
    check(runtime.admitSessionBackend(videowire::VisualBackend::openGL)
          && runtime.admitSessionBackend(videowire::VisualBackend::metal)
          && runtime.recordBackendTransition(videowire::VisualBackend::metal)
          && runtime.recordBackendTransition(videowire::VisualBackend::metal),
          "bounded backend admission happens once and only explicit transitions are events");
    check(runtime.recordDimensions(1920, 1080, 1920, 1080)
          && ! runtime.recordDimensions(0, 1080, 1920, 1080)
          && ! runtime.recordDimensions(std::numeric_limits<int>::max(), 1080, 1920, 1080),
          "dimensions reject non-positive and oversized values before arithmetic");
    check(! runtime.recordRuntime(1, 1, 1, 1, videowire::VisualTransportMode::none, true)
          && runtime.recordRuntime(100, 40, 0, 0, videowire::VisualTransportMode::zeroCopy, true)
          && runtime.recordZeroCopyAllocationBytes(1920ull*1080ull*4ull)
          && runtime.recordReadbackCopiedBytes(123),
          "transport rejects none, zero durations count, and byte semantics use separate typed APIs");
    check(! runtime.snapshot().particles.observed && ! runtime.snapshot().generators.observed,
          "generic runtime transport does not fabricate generator or particle execution");
    check(runtime.recordExecutionObservation(videowire::VisualExecutionKind::generator, 17)
          && runtime.recordExecutionObservation(videowire::VisualExecutionKind::particle, 23),
          "typed production execution hooks independently record generator and particle work");
    runtime.recordResources(3, 7, 4096); runtime.recordResources(0, 0, 0);
    runtime.recordDrop(videowire::VisualDropReason::noBuffer);
    runtime.recordDimensions(1000, 1000, 900, 900);
    runtime.recordDimensions(3840, 2160, 1920, 1080);
    const auto runtimeSnapshot = runtime.snapshot();
    const auto runtimeJson = videowire::visualTelemetryJson(runtimeSnapshot);
    check(runtimeSnapshot.compositor.count == 1 && runtimeSnapshot.presentation.totalNs == 40
          && runtimeSnapshot.particles.count == 1 && runtimeSnapshot.particles.totalNs == 23
          && runtimeSnapshot.generators.count == 1 && runtimeSnapshot.generators.totalNs == 17
          && runtimeSnapshot.zeroCopyFrames == 1 && runtimeSnapshot.zeroCopyAllocationBytes == 1920ull*1080ull*4ull
          && runtimeSnapshot.readbackCopiedBytes == 123 && runtimeSnapshot.framesRendered == 1
          && runtimeSnapshot.framesPresented == 1 && runtimeSnapshot.framesDropped == 1
          && runtimeSnapshot.droppedNoBuffer == 1 && runtimeSnapshot.fallbackCount == 1,
          "runtime schema records typed helper samples without claiming production owners");
    check(runtimeSnapshot.retainedFramesCurrent == 0 && runtimeSnapshot.retainedFramesPeak == 3
          && runtimeSnapshot.intermediateImagesCurrent == 0 && runtimeSnapshot.intermediateImagesPeak == 7
          && runtimeSnapshot.retainedBytesCurrent == 0 && runtimeSnapshot.retainedBytesPeak == 4096
          && runtimeSnapshot.requestedWidth == 3840 && runtimeSnapshot.actualWidth == 1920
          && runtimeSnapshot.dimensionMismatchCount == 2 && runtimeSnapshot.halfResolutionMismatchCount == 1,
          "resource snapshots preserve peaks/releases and dimensions count all mismatches plus exact halves");
    check(runtimeJson["transport"]["mode"] == "zero-copy" && runtimeJson["drops"]["reasons"]["noBuffer"] == 1
          && runtimeJson["resources"]["retainedFramesPeak"] == 3
          && runtimeJson["dimensions"]["halfResolutionMismatchCount"] == 1,
          "observed runtime JSON projects typed transport, drop, resource and dimension fields");
    runtime.resetOwner(0, 0);
    const auto graphReset = runtime.snapshot();
    check(! graphReset.transportObserved && ! graphReset.resourcesObserved && ! graphReset.dimensionsObserved
          && graphReset.retainedFramesCurrent == 0 && graphReset.retainedFramesPeak == 3
          && ! graphReset.particles.observed && ! graphReset.generators.observed
          && graphReset.backendObserved && graphReset.fallbackCount == 1,
          "graph/seek reset clears runtime/current values while preserving resource peaks and session backend");
    runtime.recordExecutionObservation(videowire::VisualExecutionKind::generator, 31);
    check(runtime.snapshot().generators.observed && runtime.snapshot().generators.count == 1
          && runtime.snapshot().generators.totalNs == 31,
          "the first real execution after a requested reset belongs to the new generation");
    runtime.recordBudgetReceipt(expectedBudgetReceipt);
    runtime.resetSession();
    const auto sessionReset = runtime.snapshot();
    const auto sessionResetJson = videowire::visualTelemetryJson(sessionReset);
    check(! sessionReset.backendObserved && sessionReset.fallbackCount == 0
          && ! sessionReset.budgetObserved
          && ! sessionResetJson["budget"]["available"].get<bool>()
          && sessionResetJson["budget"]["backendProfile"].is_null(),
          "session reset clears backend history and the prior budget receipt");
    videowire::VisualPlanExecutionState failedNewSession;
    failedNewSession.setTelemetryOwner(runtime);
    auto failedSessionLimits = mediumLimits;
    failedSessionLimits.operations = 0;
    error.clear();
    check(! failedNewSession.admitPlans(mediumPlans, &error, 64, 64, &failedSessionLimits)
          && ! runtime.snapshot().budgetObserved
          && videowire::visualTelemetryJson(runtime.snapshot())["budget"]["backendProfile"].is_null(),
          "failed admission in a new session cannot restore the prior session budget");
    videowire::VisualPlanTelemetry runtimeContended;
    {
        auto heldLock = runtimeContended.lockForTesting();
        runtimeContended.recordRuntime(1, 1, 1, 1, videowire::VisualTransportMode::readback, true);
        runtimeContended.recordReadbackCopiedBytes(1); runtimeContended.recordZeroCopyAllocationBytes(1);
        runtimeContended.recordDrop(videowire::VisualDropReason::renderFailure);
        runtimeContended.recordResources(1, 1, 1); runtimeContended.recordDimensions(1, 1, 1, 1);
        runtimeContended.admitSessionBackend(videowire::VisualBackend::openGL);
        runtimeContended.recordFailedLowering("contended");
        runtimeContended.recordExecutionObservation(videowire::VisualExecutionKind::particle, 1);
    }
    check(runtimeContended.snapshot().recordingContentionDrops == 9,
          "every runtime recording API is nonblocking and explicitly counts contention drops");
    check (telemetryJson.is_object() && telemetryJson["layers"].is_array()
           && telemetryJson["nodes"].is_array() && telemetryJson["nodes"].size() == 16
           && telemetryJson["nodes"][0]["clipId"] == 1
           && telemetryJson["nodes"][0]["structuralRevision"] == 1
           && telemetryJson["nodes"][0]["stableNodeId"] == 1
           && ! telemetryJson["nodes"][0]["available"].get<bool>()
           && telemetryJson["nodes"][0]["evaluations"].is_null()
           && telemetryJson["nodes"][0]["measuredTotalNs"].is_null()
           && telemetryJson["nodes"][0]["measuredMovingNs"].is_null()
           && telemetryJson["nodes"][15]["available"].get<bool>()
           && telemetryJson["nodes"][15]["evaluations"] == 2
           && telemetryJson["nodes"][15]["measuredTotalNs"] == 100
           && telemetryJson["nodes"][15]["measuredMovingNs"] == 44.0
           && telemetryJson.contains("planCacheHits")
           && telemetryJson.contains("lastPlanLoweringWithinBudget"),
           "viewport telemetry JSON has stable bounded arrays and exact identity/cache fields");


    // Plan 3 Event coverage authority. Particle physics connection is a later packet.
    {
        auto covered = eventSchedule;
        videowire::VisualEventTriggerCursor cursor (false);
        check (cursor.consume({covered},7,70,0,3,1.0).complete,
               "event-coverage viewport attaches only inside a covered interval");
        auto incomplete = covered; incomplete.coverageComplete = false; incomplete.lossFlags = 1;
        check (!cursor.consume({incomplete},7,70,0,3,1.5).complete,
               "event-coverage source loss refuses advancement");
        const auto recovered = cursor.consume({covered},7,70,0,3,1.5);
        check (recovered.complete && recovered.count == 2,
               "event-coverage refusal preserves prior position for later exact replay");
        check (!cursor.consume({covered},7,70,0,3,4.0).complete,
               "event-coverage terminal endpoint is exclusive until the next block is proved");
        check (cursor.consume({covered},7,70,0,3,1.25).complete,
               "event-coverage backward seek attaches to its covered point");
        auto second = covered; second.routeId = 2; second.sourceNodeId = 101;
        check (cursor.consume({covered,second},7,70,0,3,1.25).count == 0,
               "event-coverage route-set replacement cannot inherit an old cursor");
        const auto fanIn = cursor.consume({covered,second},7,70,0,3,1.5);
        check (fanIn.complete && fanIn.count == 4,
               "event-coverage two same-sink source routes preserve all producer triggers");
        auto tooNarrow = second; tooNarrow.coverageEndBeat = 1.75;
        check (!cursor.consume({covered,tooNarrow},7,70,0,3,2.0).complete
               && cursor.consume({covered,second},7,70,0,3,2.0).complete,
               "event-coverage one producer interval cannot advance the aggregate cursor past another producer gap");
        second.transportEpoch = 5;
        check (!cursor.consume({covered,second},7,70,0,3,2.0).complete,
               "event-coverage mixed transport lifetimes cannot form one coherent range");
        auto unknown = covered; unknown.coverageVersion = 0;
        check (!cursor.consume({unknown},7,70,0,3,2.0).complete
               && !cursor.consume({},7,70,0,3,2.0).complete,
               "event-coverage legacy and absent schedules are unknown rather than historical silence");
        covered.coverageStartBeat = 1; covered.coverageStartSample = 32;
        videowire::VisualEventTriggerCursor exportRange (true);
        check (!exportRange.consume({covered},7,70,0,3,1.5).complete,
               "event-coverage export refuses a retained window that excludes requested start");
        const auto explicitStart = exportRange.consume({covered},7,70,0,3,1.5,1.0);
        check (explicitStart.complete && explicitStart.count == 3,
               "event-coverage explicit export start includes the boundary onset exactly once");
        covered.sourceNodeId = 102; covered.planGeneration = 10; covered.streamId = 2;
        check (cursor.consume({covered},7,70,0,3,1.5).complete
               && cursor.consume({covered},7,70,0,3,1.5).count == 0,
               "event-coverage source plan and offline identity changes establish a fresh viewport cursor");
    }


    // Plan 3 whole-frame preview clock coordination, prepared without native execution.
    {
        videotime::BeatTimeline timeline; timeline.reset(120,4);
        auto row = eventSchedule; row.triggers.clear();
        row.coverageEndBeat = 0.5; row.coverageEndSample = 16;
        videowire::VisualEventTriggerCursor cursor(false);
        const auto active = [] (int clip,double) { return clip == 7; };
        const auto selected = videowire::selectVisualEventPreviewFrame({row},cursor,0.333,30,timeline,active,0);
        const auto clock = videorender::makeShaderClock(selected.seconds,0,1,timeline,30,true,7);
        check(selected.ready && selected.seconds == 7.0/30 && selected.seconds <= 0.333
              && clock.beat == timeline.secondsToBeat(selected.seconds)
              && cursor.consume({row},7,70,0,3,clock.beat).complete,
              "event-preview extrapolated request selects one covered project frame for scene shader cursor and receipt");
        row.coverageEndBeat = 1.0; row.coverageEndSample = 32;
        const auto exactEnd = videowire::selectVisualEventPreviewFrame({row},cursor,0.5,30,timeline,active,selected.seconds);
        check(exactEnd.ready && exactEnd.seconds == 14.0/30
              && timeline.secondsToBeat(exactEnd.seconds) < row.coverageEndBeat,
              "event-preview exact exclusive watermark selects the prior grid frame without widening coverage");
        const auto rounded = videowire::selectVisualEventPreviewFrame({row},cursor,0.48,30,timeline,active,selected.seconds);
        check(rounded.ready && rounded.seconds == 14.0/30 && rounded.seconds <= 0.48,
              "event-preview frame rounding cannot place the Event scene ahead of the requested time");
        auto second = row; second.routeId = 2; second.sourceNodeId = 101;
        second.coverageEndBeat = 0.4; second.coverageEndSample = 13;
        videowire::VisualEventTriggerCursor fanIn(false);
        const auto common = videowire::selectVisualEventPreviewFrame({row,second},fanIn,0.5,30,timeline,active,0);
        check(common.ready && common.seconds == 5.0/30,
              "event-preview same-sink producers select one mutually covered timestamp");
        check(!videowire::selectVisualEventPreviewFrame({row,second},fanIn,0.5,30,timeline,active,0.2).ready,
              "event-preview stale watermark cannot induce an advancing transport's backward cursor reset");
        auto narrow = row; narrow.coverageStartBeat = 0.49; narrow.coverageStartSample = 15;
        narrow.coverageEndBeat = 0.5; narrow.coverageEndSample = 16;
        check(!videowire::selectVisualEventPreviewFrame({narrow},fanIn,0.5,30,timeline,active,0).ready,
              "event-preview window with no complete project frame defers instead of closing or fabricating silence");
        auto pending = row; pending.coverageComplete = false; pending.lossFlags = 4;
        check(!videowire::selectVisualEventPreviewFrame({pending},fanIn,0.5,30,timeline,active,0).ready
              && videowire::selectVisualEventPreviewFrame({row},fanIn,0.5,30,timeline,active,0).ready,
              "event-preview missing output defers and a later coherent snapshot resumes admission");
        const auto inactive = [] (int,double) { return false; };
        check(videowire::selectVisualEventPreviewFrame({pending},fanIn,0.48,30,timeline,inactive,0).seconds == 0.5,
              "event-preview inactive Event routes preserve ordinary legacy nearest-grid clock behavior");
        videowire::VisualEventTriggerCursor gapCursor(false);
        check(gapCursor.consume({row},7,70,0,3,0.2).complete,
              "event-preview gap fixture establishes the prior cursor");
        auto gap = row; gap.coverageStartBeat = 0.3; gap.coverageStartSample = 10;
        check(!videowire::selectVisualEventPreviewFrame({gap},gapCursor,0.5,30,timeline,active,0.1).ready
              && gapCursor.available({row},7,70,0,3,0.5),
              "event-preview clock selection cannot erase a missing historical interval or advance on refusal");
        auto reset = row; reset.transportEpoch = 5;
        check(videowire::selectVisualEventPreviewFrame({reset},gapCursor,0.1,30,timeline,active,1.0).ready,
              "event-preview explicit transport reseek can select a fresh covered frame");
        videowire::VisualEventTriggerCursor preflightCursor(false);
        check(preflightCursor.consume({eventSchedule},7,70,0,3,1.0).complete,
              "event-preview preflight fixture establishes one consumed point");
        check(videowire::selectVisualEventPreviewFrame({eventSchedule},preflightCursor,0.8,30,timeline,active,0.5).ready
              && preflightCursor.consume({eventSchedule},7,70,0,3,1.6).count == 2,
              "event-preview read-only preflight cannot consume or suppress actual producer triggers");
        auto laterClip = second; laterClip.clipId = 8; laterClip.nodeId = 80; laterClip.routeId = 3;
        laterClip.coverageEndBeat = 0.3; laterClip.coverageEndSample = 10;
        const auto changingClips = [] (int clip,double seconds)
        { return seconds >= 0.2 ? clip == 7 : clip == 8; };
        const auto dependency = videowire::selectVisualEventPreviewFrame({second,laterClip},fanIn,0.5,30,timeline,changingClips,0);
        check(dependency.ready && dependency.seconds == 4.0/30,
              "event-preview earlier scene selection includes newly active source clips before admission");
        second.coverageComplete = false;
        check(!videowire::selectVisualEventPreviewFrame({second,laterClip},fanIn,0.5,30,timeline,changingClips,0).ready,
              "event-preview cannot erase requested-scene loss by selecting an earlier unrelated clip");
        videowire::VisualEventTriggerCursor exporting(true);
        check(!exporting.consume({row},7,70,0,3,1.0).complete,
              "event-preview coordination does not weaken export's missing historical interval refusal");
        auto malformedPlan = admittedParticles;
        malformedPlan[0].edges[0].fromPort = -1;
        check(!videowire::executeVisualLayerPlan(malformedPlan,7,particleLayer,error,videohelper::geometry::PlanUse::preview)
              && !error.empty(),
              "event-preview malformed production plan remains a terminal execution error");
        check(!videowire::selectVisualEventPreviewFrame({row},fanIn,0.5,0,timeline,active,0).ready,
              "event-preview invalid grid refuses rather than constructing a different clock");
    }

    std::printf ("visual plan executor: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
