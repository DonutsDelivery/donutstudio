#pragma once

#include "fixture_glass_caustics.h"
#include "fixture_glass_grating_incident.h"
#include "fixture_glass_grating_cell_transport.h"
#include "fixture_glass_grating_cell_coverage.h"
#include "fixture_glass_camera_cell_coverage.h"
#include "fixture_native_camera_pixel_signs.h"

namespace arbitgpu
{
enum class OpenGlQueryArithmeticRole : std::uint32_t
{ CausticPath,GratingPoint,GratingCamera,CellBoundary,CellCoverage,SelectedCameraPrimary,SelectedCameraTransport,GratingDirect,GlassCameraGratingTrace,Count };
struct OpenGlQueryArithmeticProgramIdentity
{
    struct Stage {unsigned type=0;std::uint64_t fingerprint=0;std::string source;};
    std::uint32_t contractVersion=1,glslVersion=430,contextMajor=0,contextMinor=0,contextProfile=0;
    std::uintptr_t context=0;
    std::uint64_t contextGeneration=0,ownerGeneration=0,rendererGeneration=0;
    unsigned linkedProgram=0,stageCount=0;
    std::array<Stage,2> stages{};
};
struct OpenGlQueryArithmeticEvidence
{
    OpenGlQueryArithmeticRole role=OpenGlQueryArithmeticRole::CausticPath;
    std::shared_ptr<const OpenGlQueryArithmeticProgramIdentity> program;
    glasscaustics::SubmissionIdentity submission;
    std::array<std::shared_ptr<const OpenGlQueryArithmeticEvidence>,2> parents{};
};
// Producer-owned linked source and completed submission for sceneClosestHit
// and its existing renamed glass copy. This does not qualify the legacy shadow
// query or all shader arithmetic, and grants no interval/camera/Color authority.
bool readOpenGlQueryArithmeticEvidenceForFixture(const NativeFixtureSceneFrame& frame,OpenGlQueryArithmeticRole role,
    std::shared_ptr<const OpenGlQueryArithmeticEvidence>& output,std::string& error);
bool validateOpenGlQueryArithmeticEvidenceForFixture(const NativeFixtureSceneFrame& frame,
    const std::shared_ptr<const OpenGlQueryArithmeticEvidence>& evidence,const glasscaustics::SubmissionIdentity& expected,std::string& error);

struct OpenGlGlassCausticDiagnostics
{
    std::uint64_t admittedFrameBytes=0,nativeStorageBytes=0,queryWork=0,nativeTemporaryBytes=0;
    glasscaustics::DepositionSchedule deposition;
    double baseMilliseconds=0,rasterMilliseconds=0,traceMilliseconds=0,sortMilliseconds=0;
    double reduceMilliseconds=0,readbackMilliseconds=0,resolveMilliseconds=0,frameMilliseconds=0;
};
// Backend-private native route used by the scene-linear Color dispatcher.
NativeFixtureSceneSubmission renderOpenGlGlassCausticScene(
    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
    const std::shared_ptr<const NativeFixtureSceneResources>& resources,
    HarmonicMIDI::grid::SceneLightId source,
    HarmonicMIDI::grid::SceneObjectId glassObject, HarmonicMIDI::grid::SceneMaterialId glassMaterial,
    const glasscaustics::AdmittedBranchPolicy& policy,
    glasscaustics::SchedulingDescription scheduling,
    std::uint32_t width, std::uint32_t height, NativeFixtureSceneRuntimeInputs runtime,
    const glasscaustics::MultipleScatterPolicy* multipleScatterPolicy=nullptr);
bool readOpenGlGlassCausticReceiptForFixture(const NativeFixtureSceneFrame& frame,
    std::shared_ptr<const glasscaustics::NativeReadback>& receipt,
    glasscaustics::EnergyLedger& energy, std::string& error,
    OpenGlGlassCausticDiagnostics* diagnostics=nullptr);

struct OpenGlGratingIncidentDiagnostics
{
    std::uint64_t admittedBytes=0,nativeBytes=0,baseQueryWork=0,extraQueryWork=0,orderWork=0;
};
// Produces actual-hit/order adjunct receipts only. No combined Color authority.
bool produceOpenGlGlassGratingIncidents(
    const std::shared_ptr<const NativeFixtureSceneFrame>& nativeFrame,
    const std::shared_ptr<const glasscaustics::GratingIncidentBinding>& binding,
    glasscaustics::NativeGratingReadback& output,std::string& error,
    OpenGlGratingIncidentDiagnostics* diagnostics=nullptr);

// Private eight-band center/corner ray traces only. No footprint or Color grant.
bool produceOpenGlGlassCameraGratingTrace(
    const std::shared_ptr<const NativeFixtureSceneFrame>& nativeFrame,
    const std::shared_ptr<const glasscaustics::GratingIncidentBinding>& binding,
    const std::vector<glasscaustics::NativeGlassCameraGratingTraceInput>& inputs,
    double evaluationTime,glasscaustics::NativeGlassCameraGratingTraceReadback& output,std::string& error);
// Native fixture inspection only; these handles cannot supply a trace receipt.
bool readOpenGlGlassCameraGratingTraceStorageForFixture(const NativeFixtureSceneFrame& frame,
    std::array<unsigned,5>& buffers,std::uintptr_t& fence,std::string& error);
// One selected directional light at native queried endpoints. No base/Color grant.
bool produceOpenGlDirectGratingContribution(
    const std::shared_ptr<const NativeFixtureSceneFrame>& nativeFrame,
    const std::shared_ptr<const glasscaustics::GratingIncidentBinding>& binding,
    const std::vector<glasscaustics::NativeDirectGratingEndpointInput>& inputs,
    double evaluationTime,glasscaustics::NativeDirectGratingContributionReadback& output,std::string& error);

// All frozen directional D8 owners at the same ordered native endpoints.
// Private irradiance sums only; no camera/base/radiance/Color grant.
bool produceOpenGlDirectGratingIrradianceCohort(
    const std::shared_ptr<const NativeFixtureSceneFrame>& nativeFrame,
    const std::shared_ptr<const glasscaustics::DirectGratingIrradianceCohort>& cohort,double evaluationTime,
    std::vector<glasscaustics::NativeDirectGratingContributionReadback>& sourceReadbacks,
    glasscaustics::DirectGratingIrradianceCohortReadback& output,std::string& error);
// Actual admission/registration followed by allocation refusal before shared
// preparation or dispatch. Invalid or over-cap requests remain unadmitted.
bool refuseOpenGlDirectGratingCohortPreparationForFixture(
    const std::shared_ptr<const NativeFixtureSceneFrame>& nativeFrame,
    const std::shared_ptr<const glasscaustics::DirectGratingIrradianceCohort>& cohort,double evaluationTime,
    std::vector<glasscaustics::NativeDirectGratingContributionReadback>& sourceReadbacks,
    glasscaustics::DirectGratingIrradianceCohortReadback& output,std::string& error);
bool readOpenGlDirectGratingCohortStorageForFixture(const NativeFixtureSceneFrame& frame,std::size_t source,
    std::array<unsigned,4>& buffers,unsigned& program,std::uintptr_t& fence,std::string& error,
    std::array<std::uint64_t,2>* reservation=nullptr);
// Registered same-context terminal order ledger. Keeps irradiance per area;
// this private result supplies no finite footprint, radiance or public Color.
bool produceOpenGlDirectGratingTerminalOrders(const std::shared_ptr<const NativeFixtureSceneFrame>& frame,
    const std::shared_ptr<const glasscaustics::DirectGratingIrradianceCohort>& cohort,
    const glasscaustics::SubmissionIdentity&,double,
    const std::vector<glasscaustics::NativeDirectGratingContributionReadback>&,
    glasscaustics::DirectGratingTerminalOrderLedger&,std::string&);
bool readOpenGlDirectGratingTerminalStateForFixture(const NativeFixtureSceneFrame&,
    std::shared_ptr<const glasscaustics::NativeDirectGratingTerminalResult>&,
    // Direct bytes/work, terminal attempted work/deposition, retained owner count.
    std::array<std::uint64_t,5>& counters,std::string&);
// Native outgoing order hits and independent private pixel-center raster rows.
enum class OpenGlTerminalReceiverStopForFixture { None,Reserved,Fence };
bool produceOpenGlDirectGratingTerminalReceivers(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const std::shared_ptr<const glasscaustics::NativeDirectGratingTerminalResult>&,
    const glasscaustics::SubmissionIdentity&,double,glasscaustics::DirectGratingTerminalReceiverReadback&,
    std::string&,OpenGlTerminalReceiverStopForFixture=OpenGlTerminalReceiverStopForFixture::None);
bool readOpenGlDirectGratingTerminalReceiverStateForFixture(const NativeFixtureSceneFrame&,
    std::shared_ptr<const glasscaustics::NativeDirectGratingTerminalReceiverResult>&,
    std::array<unsigned,2>&,std::uintptr_t&,std::array<std::uint64_t,3>&,std::string&);
bool inspectOpenGlDirectGratingTerminalReceiverForFixture(const NativeFixtureSceneFrame&,
    const glasscaustics::DirectGratingTerminalReceiverReadback&,std::string&);
// Counter fault seam does not expose or alter the frozen attempt registry.
// Direct bytes/work, terminal work/deposition, attempt-owner bytes, native
// work/deposition, camera trace work, camera/cell/coverage work, camera bytes,
// query-owner bytes, independently registered attempt count.
bool readOpenGlNativeAccountingForFixture(const NativeFixtureSceneFrame&,std::array<std::uint64_t,14>&,std::string&,
    glasscaustics::DirectGratingTerminalNativeBudget* reconstructedBudget=nullptr);
bool exchangeOpenGlNativeAccountingForFixture(const NativeFixtureSceneFrame&,std::array<std::uint64_t,14>&,std::string&);
enum class OpenGlCameraCoverageStopForFixture { None,Reserved,Fence };
bool produceOpenGlCameraCellCoverageForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const std::shared_ptr<const glasscaustics::NativeDirectGratingTerminalReceiverResult>&,
    const glasscaustics::SubmissionIdentity&,double,glasscaustics::CameraCellCoverageReadback&,std::string&,
    OpenGlCameraCoverageStopForFixture=OpenGlCameraCoverageStopForFixture::None);
bool inspectOpenGlCameraCellCoverageForFixture(const NativeFixtureSceneFrame&,
    const glasscaustics::CameraCellCoverageReadback&,std::string&);
bool readOpenGlCameraCellCoverageStorageForFixture(const NativeFixtureSceneFrame&,
    std::array<unsigned,2>&,std::uintptr_t&,std::array<std::uint64_t,8>&,std::string&);
enum class OpenGlGratingAttemptStopForFixture { None,PointFence,CellFence };
// Real dispatch and completed fence precede the requested refusal seam.
bool produceOpenGlGlassGratingAttemptForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const std::shared_ptr<const glasscaustics::GratingIncidentBinding>&,
    const std::shared_ptr<const glasscaustics::GratingEntryCellBinding>&,
    glasscaustics::NativeReadback&,glasscaustics::EnergyLedger&,glasscaustics::NativeGratingCellTransportReadback&,
    OpenGlGratingAttemptStopForFixture,std::string&);

// Completed private child/deposition receipts. The caller must keep combined
// Color and ordinary dispatch guarded until the paired consumers are admitted.
bool produceOpenGlGlassGratingTerminalReadback(
    const std::shared_ptr<const NativeFixtureSceneFrame>& nativeFrame,
    const std::shared_ptr<const glasscaustics::GratingIncidentBinding>& binding,
    glasscaustics::NativeReadback& output,glasscaustics::EnergyLedger& energy,std::string& error,
    OpenGlGratingIncidentDiagnostics* diagnostics=nullptr);

struct OpenGlGratingCameraDiagnostics
{
    std::uint64_t queryWork=0,nativeBytes=0;
    bool geometryDrawComplete=false,receiverSpectrumResolveComplete=false;
    bool gratingSpectrumComplete=false;
};
struct OpenGlGratingCameraSpectrum
{
    std::shared_ptr<const glasscaustics::NativeGratingCameraReadback> camera;
    std::uint32_t width=0,height=0;
    std::vector<glasscaustics::DepositionKey> keys;
    std::vector<float> spectralRadiance; // Native diagnostic only, top-first.
    std::vector<glasscaustics::Texel> resolvedColor; // Native RGBA16F echo, top-first.
    bool complete=false;
};
bool readOpenGlGratingCameraSpectrumForFixture(const NativeFixtureSceneFrame& frame,
    std::shared_ptr<const OpenGlGratingCameraSpectrum>& output,std::string& error);
// Optional native center/corner boundary samples, never finite coverage.
bool produceOpenGlGlassGratingCellTransportReadback(
    const std::shared_ptr<const NativeFixtureSceneFrame>& nativeFrame,
    const std::shared_ptr<const glasscaustics::GratingEntryCellBinding>& sourceCells,
    glasscaustics::NativeReadback& pointOutput,glasscaustics::EnergyLedger& energy,
    glasscaustics::NativeGratingCellTransportReadback& output,std::string& error);
// Private angular/visibility primitives and separate eight-band diagnostics.
// Finite receiving-cell coverage and complete base Color are still required.
bool produceOpenGlGlassGratingCameraReadback(
    const std::shared_ptr<const NativeFixtureSceneFrame>& nativeFrame,
    const std::shared_ptr<const glasscaustics::GratingIncidentBinding>& binding,
    glasscaustics::NativeReadback& output,glasscaustics::EnergyLedger& energy,std::string& error,
    OpenGlGratingCameraDiagnostics* diagnostics=nullptr);

// Private real mixed owners. Staging is fenced geometry/query ownership only.
NativeFixtureScenePreparation prepareOpenGlGlassGratingResources(
    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
    const std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>& mixedRoot,
    const smoothglass::DistinctGratingAdmission& selection,
    const std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>& compiledLeaf);
bool stageOpenGlGlassGratingFrame(
    const std::shared_ptr<const NativeFixtureSceneResources>& resources,HarmonicMIDI::grid::SceneLightId source,
    const glasscaustics::AdmittedBranchPolicy& policy,std::uint32_t width,std::uint32_t height,
    NativeFixtureSceneRuntimeInputs runtime,double evaluationTime,
    std::shared_ptr<const NativeFixtureSceneFrame>& frameOut,
    std::shared_ptr<const glasscaustics::GratingIncidentBinding>& bindingOut,std::string& error,
    glasscaustics::SchedulingDescription scheduling={});

struct OpenGlGratingCellCoverageReadback
{
    std::shared_ptr<const glasscaustics::GratingCellCoverageCandidate> candidate;
    std::shared_ptr<const glasscaustics::GratingCellCoveragePayload> payload;
};
// Optional whole-leaf native evidence. The caller keeps the strong candidate.
bool produceOpenGlGlassGratingCellCoverageReadback(
    const std::shared_ptr<const NativeFixtureSceneFrame>& nativeFrame,
    const std::shared_ptr<const glasscaustics::GratingEntryCellBinding>& sourceCells,
    glasscaustics::NativeReadback& pointOutput,glasscaustics::EnergyLedger& energy,
    OpenGlGratingCellCoverageReadback& output,std::string& error,std::size_t maximumLeaves=4);

namespace glasscaustics::opengl
{
static_assert(kTableVersion==2&&kScheduleVersion==1&&kTriangleStride==8
    &&kPathReceiptStride==16&&kDepositReceiptStride==32);
inline constexpr ConsumerContract kConsumer {NativeFixtureMaterialBackend::OpenGl,
    kTableVersion,3,kBranchPolicyVersion,true,true,true,kScheduleVersion,true,true,true,kDepositionVersion,kRoughTransportVersion,kPrivateMultipleScatterVersion};
// All packed integer words are checked before casts. A malformed native buffer
// must fail atomically rather than enter the shared validator with rounded IDs.
inline bool causticWord(float value,std::uint32_t maximum,std::uint32_t& result)
{
    if(!std::isfinite(value)||value<0||value>float(maximum)||std::floor(value)!=value)return false;
    result=std::uint32_t(value);return true;
}
inline bool causticId(float low,float high,std::uint32_t& result)
{
    std::uint32_t a=0,b=0;if(!causticWord(low,65535,a)||!causticWord(high,65535,b))return false;
    result=a|(b<<16);return true;
}
inline bool causticHit(const glasscaustics::Texel* words,opaquevisibility::ClosestHit& hit,
    bool retainIndices=true)
{
    const auto& h=words[0];
    if(h[0]!=0&&h[0]!=1)return false;
    if(h[2]!=0&&h[2]!=1)return false;
    hit.status=int(h[0]);hit.distance=h[1];hit.entering=h[2]!=0;
    hit.position={words[1][0],words[1][1],words[1][2]};
    hit.outwardNormal={words[2][0],words[2][1],words[2][2]};
    return words[2][3]==0&&causticId(h[3],words[1][3],hit.work)
        &&causticId(words[3][0],words[3][1],hit.objectId)&&causticId(words[3][2],words[3][3],hit.materialId)
        &&(!retainIndices||(causticId(words[4][0],words[4][1],hit.instanceId)
            &&causticId(words[4][2],words[4][3],hit.triangleId)));
}
inline bool causticDecodePath(const glasscaustics::Texel* p,const glasscaustics::Frame& frame,
    std::size_t work,glasscaustics::NativePathReceipt& output)
{
    glasscaustics::NativePathReceipt r;
    for(unsigned t=0;t<glasscaustics::kPathReceiptStride;++t)
        for(float value:p[t])if(!std::isfinite(value))return false;
    // Inspect the explicit marker before interpreting overlapping texel14.
    std::uint32_t version=0;
    if(!causticWord(p[15][3],kPrivateMultipleScatterVersion,version)
        ||version!=(frame.multipleScatterPolicy?kPrivateMultipleScatterVersion:0u))return false;
    std::uint32_t low=0,high=0;
    if(!causticId(p[0][0],p[0][1],r.work)||!causticWord(p[0][2],glasscaustics::kMaximumBranchVisits,r.depositCount)
        ||!causticWord(p[0][3],glasscaustics::kMaximumBranchVisits,r.visits)
        ||p[1][0]!=1||(p[1][1]!=0&&p[1][1]!=1)
        ||!causticId(p[2][0],p[2][1],low)||!causticId(p[2][2],p[2][3],high)
        ||r.work!=work||work>=frame.work.size()||frame.work[work].entry>=frame.entries.size()
        ||p[1][2]!=0||p[1][3]!=0||p[4][1]!=0||p[4][2]!=0||p[4][3]!=0)return false;
    r.complete=true;r.sourceBlocked=p[1][1]==1;r.queryWork=low|(std::uint64_t(high)<<32);
    r.launchedPower=p[3][0];r.depositedPower=p[3][1];r.absorbedPower=p[3][2];r.escapedPower=p[3][3];r.unresolvedPower=p[4][0];
    if(!causticHit(p+5,r.sourceHit))return false;
    // Blocked work never queries the entry. Its unused sentinel is retained.
    if(!r.sourceBlocked&&!causticHit(p+10,r.entryHit,version==0))return false;
    if(!decodePathExtensions(p[14],p[15],PathExtensionLayout::OpenGl,r))return false;
    output=r;return true;
}
inline bool causticDecodeDeposit(const glasscaustics::Texel* p,std::uint32_t owningMultipleScatterVersion,
    glasscaustics::NativeDeposit& output)
{
    glasscaustics::NativeDeposit r;std::uint32_t low=0,high=0;
    for(unsigned t=0;t<glasscaustics::kDepositReceiptStride;++t)
        for(float value:p[t])if(!std::isfinite(value))return false;
    if(p[4][2]!=0||p[4][3]!=0||p[30]!=glasscaustics::Texel{}||p[31]!=glasscaustics::Texel{})return false;
    std::array<glasscaustics::Texel,8> ancestry;
    std::copy_n(p+22,ancestry.size(),ancestry.begin());
    if(!glasscaustics::decodeMacroAncestry(ancestry,owningMultipleScatterVersion,r.ancestry))return false;
    if(!causticId(p[0][0],p[0][1],r.work)||!causticWord(p[0][2],glasscaustics::kMaximumInterfaces,r.interfaces)
        ||(p[0][3]!=0&&p[0][3]!=1)||!causticId(p[1][0],p[1][1],low)||!causticId(p[1][2],p[1][3],high)
        ||(p[3][0]!=0&&p[3][0]!=1)||!causticWord(p[3][1],kMaximumNativeFixtureExtent,r.footprint.x)
        ||!causticWord(p[3][2],kMaximumNativeFixtureExtent,r.footprint.y))return false;
    r.reflectionHistory=low|(std::uint64_t(high)<<32);r.cameraVisible=p[0][3]==1;
    r.fraction=p[2][0];r.incidentPower=p[2][1];r.outgoingPower=p[2][2];r.absorbedPower=p[2][3];
    r.footprint.projected=p[3][0]==1;r.footprint.linearDepth=p[3][3];r.footprint.worldArea=p[4][0];r.raster.linearDepth=p[4][1];
    if(!causticId(p[5][0],p[5][1],r.raster.object)||!causticId(p[5][2],p[5][3],r.raster.material)
        ||!causticId(p[6][0],p[6][1],r.raster.instance)||!causticId(p[6][2],p[6][3],r.raster.triangle)
        ||!causticHit(p+7,r.receiver)||!causticHit(p+12,r.cameraHit)||!causticHit(p+17,r.pixelHit))return false;
    output=r;return true;
}
inline std::string querySource()
{
    auto source=opaquevisibility::glslShader(true);
    // Instrument the released query's existing work units, without changing
    // its intersection, traversal, tie breaking or unsupported status rules.
    smoothglass::replace(source,"++work;","++work; ++causticLastWork;");
    return "uint causticLastWork=0u;\n"+source;
}
// Native camera trace helpers. The typed record writer is assembled separately
// from the pinned shared trace contract; these functions publish no receipt.
inline std::string cameraGlassGratingMathSource()
{
    return R"GLSL(
bool cameraTraceFinite3(vec3 value) {return all(not(isnan(value)))&&all(not(isinf(value)));}
float cameraTraceBias(vec3 p) {return max(1e-5,2e-6*max(max(abs(p.x),abs(p.y)),abs(p.z)));}
bool cameraTraceInterior(SceneClosestHit hit,vec3 a,vec3 b,vec3 c,vec3 normal,out vec3 barycentric) {
    precise vec3 e1=b-a,e2=c-a,delta=hit.position-a;
    precise float q00=dot(e1,e1),q01=dot(e1,e2),q11=dot(e2,e2),det=q00*q11-q01*q01;
    barycentric=vec3(0);
    if(!causticFinite(det)||det<=0.0)return false;
    precise float b1=(q11*dot(delta,e1)-q01*dot(delta,e2))/det;
    precise float b2=(q00*dot(delta,e2)-q01*dot(delta,e1))/det;
    barycentric=vec3(1.0-b1-b2,b1,b2);
    return cameraTraceFinite3(barycentric)&&all(greaterThan(barycentric,vec3(1e-5)))
        &&all(lessThan(barycentric,vec3(1.0-1e-5)))
        &&length(a+e1*b1+e2*b2-hit.position)<=8.0*cameraTraceBias(hit.position)
        &&length(hit.outwardNormal-normal)<=2e-4;
}
bool cameraTraceInterface(vec3 direction,bool inside,SceneClosestHit hit,float ior,
    out vec3 normal,out vec3 transmitted,out vec3 reflected,out float fresnel,out bool tir) {
    normal=inside?-hit.outwardNormal:hit.outwardNormal;
    transmitted=reflected=vec3(0);fresnel=0.0;tir=false;
    if(inside==hit.entering||!causticFinite(ior)||ior<1.0||dot(direction,normal)>=0.0)return false;
    fresnel=causticFresnel(direction,normal,inside?ior:1.0,inside?1.0:ior,transmitted,reflected,tir);
    return causticFinite(fresnel)&&fresnel>=0.0&&fresnel<=1.0&&cameraTraceFinite3(reflected)
        &&abs(dot(reflected,reflected)-1.0)<=1e-4
        &&cameraTraceFinite3(transmitted)&&(tir||abs(dot(transmitted,transmitted)-1.0)<=1e-4);
}
struct CameraTraceNativeBranch {
    vec3 origin,direction;float fraction;uint parent,token,interfaces,history;bool inside;uvec4 ancestry[2];
};
struct CameraTraceNativeVisit {
    CameraTraceNativeBranch path;SceneClosestHit hit;uint work,kind;
    vec3 normal,transmitted,reflected,barycentric;float fresnel,transmittedFraction,reflectedFraction,ior;bool tir;
};
// The contract-specific writer consumes this native local after its query and
// interface calculation. No record can supply these values to the traversal.
void cameraTraceWriteVisit(int invocation,int ordinal,CameraTraceNativeVisit visit);
uint cameraTraceId(vec2 words) {return uint(words.x)|(uint(words.y)<<16u);}
bool cameraTraceTuple(SceneClosestHit hit,vec4 owners,vec4 indices) {
    return hit.objectId==cameraTraceId(owners.xy)&&hit.materialId==cameraTraceId(owners.zw)
        &&hit.instanceId==uint(indices.x)&&hit.triangleId==uint(indices.y);
}
bool cameraTraceOutside(vec3 origin,uint glassObject,uint glassMaterial) {
    vec3 lower=vec3(3.402823e38),upper=vec3(-3.402823e38);bool found=false;
    for(int triangle=0;triangle<uSceneTriangleCount;++triangle) {
        int at=triangle*8;vec4 owners=sceneTriangles[at+4];
        if(cameraTraceId(owners.xy)!=glassObject||cameraTraceId(owners.zw)!=glassMaterial)continue;
        for(int vertex=0;vertex<3;++vertex) {
            vec3 point=sceneTriangles[at+vertex].xyz;
            if(!cameraTraceFinite3(point))return false;
            lower=min(lower,point);upper=max(upper,point);found=true;
        }
    }
    double margin=4.0*double(cameraTraceBias(origin));
    // An outward closed shell need not be convex. Outside its complete native
    // vertex AABB is sufficient; all other initial media are unsupported.
    if(found)for(int axis=0;axis<3;++axis)
        if(double(lower[axis])-double(origin[axis])>margin||double(origin[axis])-double(upper[axis])>margin)return true;
    return false;
}
void cameraTraceBranches(int invocation,vec3 origin,vec3 direction,float minimum,float maximum,float ior,
    uint glassObject,uint glassMaterial,out int count,out uint work,out uint reasons,out float unresolved) {
    CameraTraceNativeBranch stack[CAUSTIC_MAX_PENDING];int pending=1;
    CameraTraceNativeBranch root;root.origin=origin;root.direction=direction;root.fraction=1.0;
    root.parent=0xffffffffu;root.token=0u;root.interfaces=0u;root.history=0u;root.inside=false;
    root.ancestry[0]=root.ancestry[1]=uvec4(0);stack[0]=root;
    count=0;work=0u;reasons=0u;unresolved=0.0;
    if(!cameraTraceOutside(origin,glassObject,glassMaterial)) {reasons=2u;unresolved=1.0;return;}
    while(pending>0) {
        if(count>=CAUSTIC_MAX_VISITS) {reasons=16u;break;}
        CameraTraceNativeBranch branch=stack[--pending];
        causticLastWork=0u;
        SceneClosestHit hit=sceneClosestHit(branch.origin,branch.direction,minimum,maximum,uOpaqueControl);
        work+=causticLastWork;
        CameraTraceNativeVisit visit;visit.path=branch;visit.hit=hit;visit.work=causticLastWork;visit.kind=4u;
        visit.normal=visit.transmitted=visit.reflected=visit.barycentric=vec3(0);
        visit.fresnel=visit.transmittedFraction=visit.reflectedFraction=0.0;visit.ior=ior;visit.tir=false;
        bool valid=(hit.status==0||hit.status==1)&&causticLastWork>0u;
        if(!valid)reasons=1u;
        else if(hit.status==0) {
            valid=!branch.inside;visit.kind=valid?3u:4u;if(!valid)reasons=2u;
        } else {
            int sceneTriangle=-1;
            for(int triangle=0;triangle<uSceneTriangleCount;++triangle) {
                int at=triangle*8;
                if(cameraTraceTuple(hit,sceneTriangles[at+4],sceneTriangles[at+5])) {sceneTriangle=at;break;}
            }
            valid=sceneTriangle>=0;
            if(valid)valid=length(hit.outwardNormal-sceneTriangles[sceneTriangle+3].xyz)<=2e-4;
            if(valid&&hit.objectId==glassObject) {
                if(branch.interfaces>=uint(CAUSTIC_MAX_INTERFACES)) {valid=false;reasons=4u;}
                else {
                    valid=hit.materialId==glassMaterial&&cameraTraceInterface(branch.direction,branch.inside,hit,ior,
                        visit.normal,visit.transmitted,visit.reflected,visit.fresnel,visit.tir);
                    if(!valid)reasons=64u;
                    else {
                        visit.kind=0u;
                        precise float reflectedFraction=branch.fraction*visit.fresnel;
                        precise float transmittedFraction=branch.fraction*(1.0-visit.fresnel);
                        visit.reflectedFraction=reflectedFraction;visit.transmittedFraction=transmittedFraction;
                        if(!causticFinite(reflectedFraction)||!causticFinite(transmittedFraction)) {valid=false;reasons=64u;}
                        else if((visit.fresnel>0.0&&reflectedFraction<=0.0)
                            ||(!visit.tir&&visit.fresnel<1.0&&transmittedFraction<=0.0)) {valid=false;reasons=32u;}
                    }
                }
            } else if(valid) {
                valid=!branch.inside;
                int target=-1;
                for(int triangle=0;triangle<uGratingCount;++triangle) {
                    int at=triangle*8;if(cameraTraceTuple(hit,witness[at+4],witness[at+5])) {target=at;break;}
                }
                visit.kind=target<0?2u:1u;
                if(valid&&target>=0)valid=hit.entering&&cameraTraceInterior(hit,witness[target].xyz,
                    witness[target+1].xyz,witness[target+2].xyz,witness[target+7].xyz,visit.barycentric);
                // A hit on another primitive of the selected grating is never
                // an ordinary terminal: its seam/ownership remains unresolved.
                if(target<0&&hit.objectId==cameraTraceId(witness[4].xy))valid=false;
            }
            if(!valid&&reasons==0u)reasons=2u;
        }
        if(!valid) {
            visit.kind=4u;visit.normal=visit.transmitted=visit.reflected=visit.barycentric=vec3(0);
            visit.fresnel=visit.transmittedFraction=visit.reflectedFraction=0.0;visit.tir=false;
        }
        cameraTraceWriteVisit(invocation,count,visit);uint parent=uint(count++);
        if(reasons!=0u) {unresolved+=branch.fraction;break;}
        if(visit.kind!=0u)continue;
        int children=(visit.reflectedFraction>0.0?1:0)+(!visit.tir&&visit.transmittedFraction>0.0?1:0);
        if(pending+children>CAUSTIC_MAX_PENDING) {reasons=8u;unresolved+=visit.reflectedFraction+visit.transmittedFraction;break;}
        float offset=cameraTraceBias(hit.position);
        CameraTraceNativeBranch child=branch;child.parent=parent;child.interfaces=branch.interfaces+1u;
        int ancestryRow=int(branch.interfaces/16u),ancestryLane=int((branch.interfaces%16u)/4u);
        uint ancestryShift=(branch.interfaces%4u)*8u;
        // Push reflection first so transmission is visited first. Only exact
        // zero factors are omitted; every positive optical branch is retained.
        if(visit.reflectedFraction>0.0) {
            child.origin=hit.position+visit.normal*offset;child.direction=visit.reflected;
            child.fraction=visit.reflectedFraction;child.token=1u;child.history=branch.history|(1u<<branch.interfaces);
            child.inside=branch.inside;child.ancestry[ancestryRow][ancestryLane]=branch.ancestry[ancestryRow][ancestryLane]|(1u<<ancestryShift);
            stack[pending++]=child;
        }
        if(!visit.tir&&visit.transmittedFraction>0.0) {
            child.origin=hit.position-visit.normal*offset;child.direction=visit.transmitted;
            child.fraction=visit.transmittedFraction;child.token=65u;child.history=branch.history;
            child.inside=!branch.inside;child.ancestry[ancestryRow][ancestryLane]=branch.ancestry[ancestryRow][ancestryLane]|(65u<<ancestryShift);
            stack[pending++]=child;
        }
    }
    for(int branch=0;branch<pending;++branch)unresolved+=stack[branch].fraction;
}
)GLSL";
}
// Fixed shared typed lanes; no CPU optics, visibility or terminal replay.
inline std::string cameraGlassGratingComputeSource()
{
    return std::string("#version 430 core\n")+querySource()+nativeMathSource()+R"GLSL(
layout(local_size_x=32) in;
struct CameraTraceInput {uvec4 pixel,control;vec4 originMinimum,directionMaximum;};
layout(std430,binding=0) readonly buffer Inputs {CameraTraceInput inputs[];};
layout(std430,binding=1) writeonly buffer Traces {uvec4 result[];};
layout(std430,binding=2) readonly buffer SceneTriangles {vec4 sceneTriangles[];};
layout(std430,binding=3) readonly buffer Grating {vec4 witness[];};
uniform int uSampleCount,uSceneTriangleCount,uGratingCount;
uniform vec4 uGlass;
uniform vec4 uCamera,uRotation,uProjection,uIor[2],uWavelength[2];
uniform vec2 uExtent;
)GLSL"+cameraGlassGratingMathSource()+R"GLSL(
void cameraTraceWriteVisit(int invocation,int ordinal,CameraTraceNativeVisit visit) {
    int base=invocation*(10+CAUSTIC_MAX_VISITS*16)+10+ordinal*16;
    CameraTraceNativeBranch path=visit.path;SceneClosestHit hit=visit.hit;
    result[base]=uvec4(uint(ordinal),path.parent,path.token,path.interfaces);
    result[base+1]=uvec4(uint(path.inside),path.history,visit.kind,uint(visit.tir));
    result[base+2]=floatBitsToUint(vec4(path.origin,1e-5));
    result[base+3]=floatBitsToUint(vec4(path.direction,path.fraction));
    result[base+4]=uvec4(uint(hit.status),hit.objectId,hit.materialId,hit.instanceId);
    result[base+5]=uvec4(hit.triangleId,visit.work,uint(hit.entering),0);
    result[base+6]=floatBitsToUint(vec4(hit.position,hit.distance));
    result[base+7]=floatBitsToUint(vec4(hit.outwardNormal,0));
    result[base+8]=floatBitsToUint(vec4(visit.normal,visit.fresnel));
    result[base+9]=floatBitsToUint(vec4(visit.transmitted,visit.transmittedFraction));
    result[base+10]=floatBitsToUint(vec4(visit.reflected,visit.reflectedFraction));
    result[base+11]=floatBitsToUint(vec4(visit.barycentric,visit.ior));
    result[base+12]=path.ancestry[0];result[base+13]=path.ancestry[1];
    result[base+14]=visit.kind==0u?floatBitsToUint(vec4(path.inside?visit.ior:1.0,path.inside?1.0:visit.ior,0,0)):uvec4(0);
}
void main() {
    uint invocation=gl_GlobalInvocationID.x;if(invocation>=uint(uSampleCount)*8u)return;
    int sampleIndex=int(invocation/8u),band=int(invocation%8u),base=int(invocation)*(10+CAUSTIC_MAX_VISITS*16);
    for(int lane=0;lane<10+CAUSTIC_MAX_VISITS*16;++lane)result[base+lane]=uvec4(0);
    CameraTraceInput cameraInput=inputs[sampleIndex];
    float dx=cameraInput.pixel.z==0u?0.5:(cameraInput.pixel.z==2u||cameraInput.pixel.z==3u?1.0:0.0);
    float dy=cameraInput.pixel.z==0u?0.5:(cameraInput.pixel.z==3u||cameraInput.pixel.z==4u?1.0:0.0);
    vec3 ray=causticCameraRay(vec2(cameraInput.pixel.xy)+vec2(dx,dy),uRotation,uProjection,uExtent);
    float ior=uIor[band/4][band%4];
    result[base]=cameraInput.pixel;result[base+1]=cameraInput.control;
    result[base+2]=floatBitsToUint(cameraInput.originMinimum);result[base+3]=floatBitsToUint(cameraInput.directionMaximum);
    result[base+4]=floatBitsToUint(vec4(uCamera.xyz,1e-5));result[base+5]=floatBitsToUint(vec4(ray,uProjection.w));
    int count;uint work,reasons;float unresolved;
    cameraTraceBranches(int(invocation),uCamera.xyz,ray,1e-5,uProjection.w,ior,cameraTraceId(uGlass.xy),cameraTraceId(uGlass.zw),count,work,reasons,unresolved);
    result[base+6]=uvec4(1u,uint(sampleIndex),uint(band),uint(count));
    result[base+7]=uvec4(reasons==0u?1u:0u,reasons,work,0u);
    result[base+8]=uvec4(floatBitsToUint(ior),floatBitsToUint(uWavelength[band/4][band%4]),floatBitsToUint(unresolved),1u);
}
)GLSL";
}
struct CameraGlassGratingPacked
{
    NativeGlassCameraGratingTracePackedHeader header{};
    std::array<NativeGlassCameraGratingTracePackedVisit,kMaximumBranchVisits> visits{};
};
static_assert(sizeof(CameraGlassGratingPacked)==kGlassCameraTraceHeaderBytes+kMaximumBranchVisits*kGlassCameraTraceVisitBytes);
inline std::string directGratingComputeSource()
{
    return std::string("#version 430 core\n")+querySource()+R"GLSL(
layout(local_size_x=32) in;
struct EndpointInput {uvec4 primitive,control;vec4 originMinimum,directionMaximum;};
layout(std430,binding=0) readonly buffer Endpoints {EndpointInput endpoints[];};
layout(std430,binding=1) writeonly buffer Contributions {uvec4 result[];};
layout(std430,binding=2) readonly buffer Source {uvec4 sourceIds;vec4 sourceD[2];};
layout(std430,binding=3) readonly buffer Grating {vec4 witness[];};
uniform int uEndpointCount,uGratingCount;
uniform vec4 uSource,uLi[2];
bool directFinite(float value) {return !isnan(value)&&!isinf(value);}
bool directFinite3(vec3 value) {return all(not(isnan(value)))&&all(not(isinf(value)));}
uint directId(vec2 words) {return uint(words.x)|(uint(words.y)<<16u);}
uvec4 directPrimitive(SceneClosestHit hit) {return uvec4(hit.objectId,hit.materialId,hit.instanceId,hit.triangleId);}
void directHit(int base,SceneClosestHit hit,uint work) {
    result[base]=uvec4(uint(hit.status),floatBitsToUint(hit.distance),uint(hit.entering),work);
    result[base+1]=floatBitsToUint(vec4(hit.position,0));
    result[base+2]=floatBitsToUint(vec4(hit.outwardNormal,0));
    result[base+3]=directPrimitive(hit);result[base+4]=uvec4(0);
}
void main() {
    uint index=gl_GlobalInvocationID.x;if(index>=uint(uEndpointCount))return;
    int base=int(index)*27;
    for(int lane=0;lane<27;++lane)result[base+lane]=uvec4(0);
    EndpointInput endpointInput=endpoints[index];
    result[base]=endpointInput.primitive;result[base+1]=endpointInput.control;
    result[base+2]=floatBitsToUint(endpointInput.originMinimum);result[base+3]=floatBitsToUint(endpointInput.directionMaximum);
    result[base+4]=sourceIds;result[base+5]=floatBitsToUint(sourceD[0]);result[base+6]=floatBitsToUint(sourceD[1]);
    result[base+7]=floatBitsToUint(uLi[0]);result[base+8]=floatBitsToUint(uLi[1]);result[base+9]=floatBitsToUint(uSource);
    result[base+25]=uvec4(1,2,0,0);result[base+26]=uvec4(0,uint(uEndpointCount),0,0);
    if(endpointInput.control!=uvec4(1,0,0,0)||sourceIds.xyz!=uvec3(1,1,1))return;
    causticLastWork=0u;
    SceneClosestHit endpoint=sceneClosestHit(endpointInput.originMinimum.xyz,endpointInput.directionMaximum.xyz,
        endpointInput.originMinimum.w,endpointInput.directionMaximum.w,uOpaqueControl);
    uint endpointWork=causticLastWork;directHit(base+10,endpoint,endpointWork);
    if(endpoint.status!=1||directPrimitive(endpoint)!=endpointInput.primitive||endpointWork==0u)return;
    int target=-1;
    for(int triangle=0;triangle<uGratingCount;++triangle) {
        int at=triangle*8;vec4 owners=witness[at+4],indices=witness[at+5];
        if(uvec4(directId(owners.xy),directId(owners.zw),uint(indices.x),uint(indices.y))==endpointInput.primitive) {
            target=at;break;
        }
    }
    if(target<0)return;
    vec3 a=witness[target].xyz,e1=witness[target+1].xyz-a,e2=witness[target+2].xyz-a;
    vec3 delta=endpoint.position-a,normal=witness[target+7].xyz;
    precise float q00=dot(e1,e1),q01=dot(e1,e2),q11=dot(e2,e2),det=q00*q11-q01*q01;
    if(!directFinite(det)||det<=0.0)return;
    precise float b1=(q11*dot(delta,e1)-q01*dot(delta,e2))/det;
    precise float b2=(q00*dot(delta,e2)-q01*dot(delta,e1))/det;
    precise vec3 barycentric=vec3(1.0-b1-b2,b1,b2);
    result[base+20]=floatBitsToUint(vec4(barycentric,0));
    float tolerance=8.0*max(1e-5,2e-6*max(max(abs(endpoint.position.x),abs(endpoint.position.y)),abs(endpoint.position.z)));
    if(!directFinite3(barycentric)||any(lessThanEqual(barycentric,vec3(1e-5)))
        ||any(greaterThanEqual(barycentric,vec3(1.0-1e-5)))
        ||length(a+e1*b1+e2*b2-endpoint.position)>tolerance
        ||length(endpoint.outwardNormal-normal)>2e-4)return;
    vec4 sourceOrigin=vec4(endpoint.position,1e-5);
    result[base+21]=floatBitsToUint(sourceOrigin);result[base+22]=floatBitsToUint(uSource);
    causticLastWork=0u;
    SceneClosestHit source=sceneClosestHit(endpoint.position,uSource.xyz,sourceOrigin.w,uSource.w,uOpaqueControl);
    uint sourceWork=causticLastWork;directHit(base+15,source,sourceWork);
    result[base+25].w=endpointWork+sourceWork;
    if((source.status!=0&&source.status!=1)||sourceWork==0u)return;
    precise float cosine=max(0.0,dot(normal,uSource.xyz));
    bool positive=false;
    for(int bandGroup=0;bandGroup<2;++bandGroup) {
        precise vec4 bands=source.status==0?sourceD[bandGroup]*cosine:vec4(0);
        if(any(isnan(bands))||any(isinf(bands))||any(lessThan(bands,vec4(0))))return;
        result[base+23+bandGroup]=floatBitsToUint(bands);positive=positive||any(greaterThan(bands,vec4(0)));
    }
    result[base+25]=uvec4(1,positive?0u:1u,1,endpointWork+sourceWork);
    result[base+26].x=8u;
}
)GLSL";
}
using DirectGratingPacked=std::array<std::array<std::uint32_t,4>,27>;
static_assert(sizeof(DirectGratingPacked)==kDirectGratingContributionNativeBytes);
inline bool decodeDirectGratingContribution(const DirectGratingPacked& packed,
    std::size_t endpoints,NativeDirectGratingContributionSample& output)
{
    if(packed[25][0]!=kDirectGratingContributionVersion||packed[25][1]>2||packed[25][2]>1
        ||packed[26]!=std::array<std::uint32_t,4>{8,std::uint32_t(endpoints),0,0})return false;
    NativeDirectGratingContributionSample sample;
    std::memcpy(static_cast<void*>(&sample.input),packed.data(),sizeof(sample.input));
    std::memcpy(static_cast<void*>(&sample.source),packed.data()+4,sizeof(sample.source));
    const auto hit=[&](std::size_t at,opaquevisibility::ClosestHit& value) {
        const auto status=packed[at][0];if(status!=0&&status!=1&&status!=UINT32_MAX)return false;
        value.status=status==UINT32_MAX?-1:int(status);std::memcpy(&value.distance,&packed[at][1],sizeof(float));
        if(packed[at][2]>1||packed[at+1][3]||packed[at+2][3]||packed[at+4]!=std::array<std::uint32_t,4>{})return false;
        value.entering=packed[at][2]!=0;value.work=packed[at][3];
        std::memcpy(value.position.data(),packed[at+1].data(),3*sizeof(float));
        std::memcpy(value.outwardNormal.data(),packed[at+2].data(),3*sizeof(float));
        value.objectId=packed[at+3][0];value.materialId=packed[at+3][1];value.instanceId=packed[at+3][2];value.triangleId=packed[at+3][3];
        return true;
    };
    if(!hit(10,sample.endpoint)||!hit(15,sample.sourceHit)||packed[20][3])return false;
    std::memcpy(sample.barycentric.data(),packed[20].data(),3*sizeof(float));
    std::memcpy(sample.sourceOriginMinimum.data(),packed[21].data(),sizeof(Texel));
    std::memcpy(sample.sourceDirectionMaximum.data(),packed[22].data(),sizeof(Texel));
    std::memcpy(sample.surfaceIrradiance.data(),packed[23].data(),8*sizeof(float));
    sample.queryWork=packed[25][3];sample.complete=packed[25][2]!=0;
    sample.outcome=static_cast<NativeDirectGratingContributionSample::Outcome>(packed[25][1]);
    output=sample;return true;
}
inline std::string computeSource()
{
    return std::string("#version 430 core\n")+querySource()+nativeMathSource()+R"GLSL(
layout(local_size_x=32) in;
layout(std430,binding=0) buffer Paths {vec4 paths[];};
layout(std430,binding=1) buffer Deposits {vec4 deposits[];};
layout(std430,binding=3) readonly buffer Work {vec4 workTable[];};
layout(std430,binding=4) readonly buffer Triangles {vec4 triangles[];};
uniform sampler2D uRasterOwners,uRasterTriangle;
uniform int uWorkCount,uTriangleCount,uVisits,uInterfaces,uPending;
uniform vec4 uSource,uCamera,uRotation,uProjection,uIor[2],uLi[2];
uniform vec2 uExtent,uPolicy;
uint queryWork=0u;
uint id(vec2 words) {return uint(words.x)|(uint(words.y)<<16u);}
vec2 words(uint value) {return vec2(value&65535u,value>>16u);}
vec4 owners(SceneClosestHit hit) {return vec4(words(hit.objectId),words(hit.materialId));}
vec4 indices(SceneClosestHit hit) {return vec4(words(hit.instanceId),words(hit.triangleId));}
float rayBias(vec3 p) {return max(1e-5,2e-6*max(max(abs(p.x),abs(p.y)),abs(p.z)));}
SceneClosestHit query(vec3 origin,vec3 direction,float minimum,float maximum) {
    causticLastWork=0u;
    SceneClosestHit hit=sceneClosestHit(origin,direction,minimum,maximum,uOpaqueControl);
    queryWork+=causticLastWork;return hit;
}
void pathHit(int offset,SceneClosestHit hit,uint work) {
    paths[offset]=vec4(float(hit.status),hit.distance,float(hit.entering),float(work&65535u));
    paths[offset+1]=vec4(hit.position,float(work>>16u));paths[offset+2]=vec4(hit.outwardNormal,0.0);
    paths[offset+3]=owners(hit);paths[offset+4]=indices(hit);
}
void depositHit(int offset,SceneClosestHit hit,uint work) {
    deposits[offset]=vec4(float(hit.status),hit.distance,float(hit.entering),float(work&65535u));
    deposits[offset+1]=vec4(hit.position,float(work>>16u));deposits[offset+2]=vec4(hit.outwardNormal,0.0);
    deposits[offset+3]=owners(hit);deposits[offset+4]=indices(hit);
}
int findTriangle(SceneClosestHit hit) {
    for(int t=0;t<uTriangleCount;++t) {
        vec4 o=triangles[t*8+4],i=triangles[t*8+5];
        if(id(o.xy)==hit.objectId&&id(o.zw)==hit.materialId&&uint(i.x)==hit.instanceId&&uint(i.y)==hit.triangleId) return t;
    }return -1;
}
bool sameTriangle(SceneClosestHit a,SceneClosestHit b) {
    return a.objectId==b.objectId&&a.materialId==b.materialId&&a.instanceId==b.instanceId&&a.triangleId==b.triangleId;
}
struct Branch {vec3 origin,direction;float fraction;uvec2 history;int interfaces;bool inside;};
void main() {
    uint invocation=gl_GlobalInvocationID.x;if(invocation>=uint(uWorkCount)) return;
    int w=int(invocation),base=w*CAUSTIC_PATH_RECEIPT_STRIDE;
    for(int i=0;i<CAUSTIC_PATH_RECEIPT_STRIDE;++i)paths[base+i]=vec4(0.0);
    // Only this invocation writes its fixed receipt segment.
    int depositBase=w*uVisits*CAUSTIC_DEPOSIT_RECEIPT_STRIDE;
    for(int i=0;i<uVisits*CAUSTIC_DEPOSIT_RECEIPT_STRIDE;++i)deposits[depositBase+i]=vec4(0.0);
    vec4 entry=workTable[w*4],normalPower=workTable[w*4+1],entryOwners=workTable[w*4+2],entryIndices=workTable[w*4+3];
    int band=int(entryIndices.w);float potential=causticEntryPower(uLi[band/4][band%4],normalPower.xyz,uSource.xyz,entry.w);
    paths[base]=vec4(words(invocation),0.0,0.0);
    bool complete=true;int count=0,visits=0,pending=0;float outgoing=0.0,absorbed=0.0,escaped=0.0,tail=0.0;
    SceneClosestHit source=query(entry.xyz,uSource.xyz,1e-5,uSource.w);uint sourceWork=causticLastWork;
    pathHit(base+5,source,sourceWork);
    bool blocked=source.status==1;complete=source.status>=0;
    SceneClosestHit entryHit=sceneEmptyHit(-1,0.0);uint entryWork=0u;
    Branch stack[CAUSTIC_MAX_PENDING];
    if(complete&&!blocked) {
        float offset=rayBias(entry.xyz);
        entryHit=query(entry.xyz+uSource.xyz*(2.0*offset),-uSource.xyz,0.0,uSource.w);entryWork=causticLastWork;
        complete=entryHit.status==1&&entryHit.entering&&entryHit.objectId==id(entryOwners.xy)
            &&entryHit.materialId==id(entryOwners.zw)&&entryHit.instanceId==uint(entryIndices.x)&&entryHit.triangleId==uint(entryIndices.y);
        if(complete)stack[pending++]=Branch(entry.xyz+uSource.xyz*(2.0*offset),-uSource.xyz,1.0,uvec2(0u),0,false);
    }
    pathHit(base+10,entryHit,entryWork);
    while(complete&&pending>0&&visits<uVisits) {
        Branch branch=stack[--pending];++visits;
        SceneClosestHit hit=query(branch.origin,branch.direction,branch.interfaces==0?0.0:1e-5,uSource.w);uint hitWork=causticLastWork;
        if(hit.status<0) {complete=false;break;}
        if(hit.status==0) {
            if(branch.inside){complete=false;break;}escaped+=potential*branch.fraction;continue;
        }
        int triangle=findTriangle(hit);if(triangle<0) {complete=false;break;}
        vec4 shape=triangles[triangle*8+5];
        if(shape.z==0.0) {
            if(branch.inside||!hit.entering||shape.w!=1.0||branch.interfaces==0) {complete=false;break;}
            float incident=potential*branch.fraction;
            vec2 power=causticReceiverPower(incident,triangles[triangle*8+6+band/4][band%4]);
            outgoing+=power.x;absorbed+=power.y;
            vec4 footprint;int projected=causticProjectReceiver(hit.position,hit.outwardNormal,uCamera.xyz,uRotation,uProjection,uExtent,footprint);
            if(projected<0) {complete=false;break;}
            SceneClosestHit cameraHit=sceneEmptyHit(0,0.0),pixelHit=sceneEmptyHit(0,0.0);
            uint cameraWork=0u,pixelWork=0u;bool visible=false;vec4 rasterOwners=vec4(0.0),raster=vec4(0.0);
            if(projected>0) {
                vec3 delta=hit.position-uCamera.xyz;float distance=length(delta);
                cameraHit=query(uCamera.xyz,delta,0.0,distance+8.0*rayBias(hit.position));cameraWork=causticLastWork;
                pixelHit=query(uCamera.xyz,causticCameraRay(footprint.xy+vec2(0.5),uRotation,uProjection,uExtent),0.0,uProjection.w);pixelWork=causticLastWork;
                if(cameraHit.status!=1||pixelHit.status<0) {complete=false;break;}
                visible=sameTriangle(hit,cameraHit)&&abs(cameraHit.distance-distance)<=8.0*rayBias(hit.position)
                    &&length(cameraHit.position-hit.position)<=8.0*rayBias(hit.position)&&pixelHit.status==1&&sameTriangle(hit,pixelHit);
                ivec2 pixel=ivec2(footprint.x,uExtent.y-1.0-footprint.y);
                rasterOwners=texelFetch(uRasterOwners,pixel,0);raster=texelFetch(uRasterTriangle,pixel,0);
                if(visible) {
                    vec3 local=causticRotate(vec4(-uRotation.xyz,uRotation.w),pixelHit.position-uCamera.xyz);
                    if(id(rasterOwners.xy)!=hit.objectId||id(rasterOwners.zw)!=hit.materialId
                        ||uint(raster.x)!=hit.instanceId||uint(raster.y)!=hit.triangleId
                        ||abs(raster.z+local.z)>2e-4*max(abs(local.z),1e-20)) {complete=false;break;}
                    float radiance=causticPixelRadiance(power.x,footprint.w);
                    if(!causticFinite(radiance)||radiance<0.0||radiance>65504.0) {complete=false;break;}
                }
            }
            int d=depositBase+count*CAUSTIC_DEPOSIT_RECEIPT_STRIDE;
            deposits[d]=vec4(words(invocation),float(branch.interfaces),float(visible));
            deposits[d+1]=vec4(words(branch.history.x),words(branch.history.y));
            deposits[d+2]=vec4(branch.fraction,incident,power);
            deposits[d+3]=vec4(float(projected>0),footprint.xy,footprint.z);
            deposits[d+4]=vec4(footprint.w,raster.z,0.0,0.0);
            deposits[d+5]=rasterOwners;deposits[d+6]=vec4(words(uint(raster.x)),words(uint(raster.y)));
            depositHit(d+7,hit,hitWork);depositHit(d+12,cameraHit,cameraWork);depositHit(d+17,pixelHit,pixelWork);
            ++count;continue;
        }
        if(branch.inside==hit.entering) {complete=false;break;}
        if(branch.interfaces>=uInterfaces) {tail+=potential*branch.fraction;continue;}
        vec3 normal=branch.inside?-hit.outwardNormal:hit.outwardNormal;
        vec3 transmitted,reflected;bool tir;float ior=uIor[band/4][band%4];
        float reflection=causticFresnel(branch.direction,normal,branch.inside?ior:1.0,branch.inside?1.0:ior,transmitted,reflected,tir);
        float reflectedFraction=branch.fraction*reflection,transmittedFraction=branch.fraction-reflectedFraction;
        if(!causticFinite(reflection)||(reflection>0.0&&reflectedFraction==0.0)||(reflection<1.0&&transmittedFraction==0.0)) {
            tail+=potential*branch.fraction;continue;
        }
        for(int child=0;child<2;++child) {
            bool reflect=child==0;float fraction=reflect?reflectedFraction:transmittedFraction;
            if(fraction==0.0||(tir&&!reflect))continue;
            if(fraction<uPolicy.y||pending>=uPending) {tail+=potential*fraction;continue;}
            uvec2 history=branch.history;
            if(reflect) {if(branch.interfaces<32)history.x|=1u<<uint(branch.interfaces);else history.y|=1u<<uint(branch.interfaces-32);}
            stack[pending++]=Branch(hit.position+normal*(reflect?rayBias(hit.position):-rayBias(hit.position)),
                reflect?reflected:transmitted,fraction,history,branch.interfaces+1,reflect?branch.inside:!branch.inside);
        }
    }
    for(int i=0;i<pending;++i)tail+=potential*stack[i].fraction;
    float launched=blocked?0.0:potential;
    complete=complete&&tail<=launched*uPolicy.x*(1.0+2e-4);
    paths[base].zw=vec2(float(count),float(visits));
    paths[base+1]=vec4(float(complete),float(blocked),0.0,0.0);
    paths[base+2]=vec4(words(queryWork),0.0,0.0);
    paths[base+3]=vec4(launched,outgoing,absorbed,escaped);paths[base+4]=vec4(tail,0.0,0.0,0.0);
}
)GLSL";
}
// Keep the released smooth shader bytes. Add the shared VNDF kernel and
// stratum receipts only for a resource prepared with nonzero roughness.
inline std::string roughComputeSource()
{
    auto source=computeSource();
    smoothglass::replace(source,nativeMathSource(),nativeRoughMathSource());
    smoothglass::replace(source,R"GLSL(uniform int uWorkCount,uTriangleCount,uVisits,uInterfaces,uPending;
)GLSL",R"GLSL(uniform int uWorkCount,uTriangleCount,uVisits,uInterfaces,uPending,uInterfaceSamples;
uniform float uRoughness;
)GLSL");
    smoothglass::replace(source,R"GLSL(    int band=int(entryIndices.w);float potential=causticEntryPower(uLi[band/4][band%4],normalPower.xyz,uSource.xyz,entry.w);
)GLSL",R"GLSL(    int band=int(entryIndices.w);uint interfaceSample=(invocation/8u)%uint(uInterfaceSamples);
    float potential=causticEntryPower(uLi[band/4][band%4],normalPower.xyz,uSource.xyz,entry.w);
    if(uRoughness>0.0)potential/=float(uInterfaceSamples);
)GLSL");
    smoothglass::replace(source,R"GLSL(    bool complete=true;int count=0,visits=0,pending=0;float outgoing=0.0,absorbed=0.0,escaped=0.0,tail=0.0;
)GLSL",R"GLSL(    bool complete=true;int count=0,visits=0,pending=0;float outgoing=0.0,absorbed=0.0,escaped=0.0,tail=0.0,masked=0.0;
)GLSL");
    smoothglass::replace(source,R"GLSL(        float reflection=causticFresnel(branch.direction,normal,branch.inside?ior:1.0,branch.inside?1.0:ior,transmitted,reflected,tir);
        float reflectedFraction=branch.fraction*reflection,transmittedFraction=branch.fraction-reflectedFraction;
        if(!causticFinite(reflection)||(reflection>0.0&&reflectedFraction==0.0)||(reflection<1.0&&transmittedFraction==0.0)) {
            tail+=potential*branch.fraction;continue;
)GLSL",R"GLSL(        float reflectedFraction,transmittedFraction;
        if(uRoughness==0.0) {
            float reflection=causticFresnel(branch.direction,normal,branch.inside?ior:1.0,branch.inside?1.0:ior,transmitted,reflected,tir);
            reflectedFraction=branch.fraction*reflection;transmittedFraction=branch.fraction-reflectedFraction;
            if(!causticFinite(reflection)||(reflection>0.0&&reflectedFraction==0.0)||(reflection<1.0&&transmittedFraction==0.0)) {
                tail+=potential*branch.fraction;continue;
            }
        } else {
            vec2 sampleUv=causticRoughSample(interfaceSample,uint(branch.interfaces));
            vec3 power;bool interfaceComplete;
            causticRoughInterface(branch.direction,normal,branch.inside?ior:1.0,branch.inside?1.0:ior,
                uRoughness,sampleUv.x,sampleUv.y,reflected,transmitted,power,interfaceComplete);
            if(!interfaceComplete||power.x<0.0||power.y<0.0||power.z<0.0
                ||abs(power.x+power.y+power.z-1.0)>2e-5) {complete=false;break;}
            masked+=potential*branch.fraction*power.z;
            reflectedFraction=branch.fraction*power.x;transmittedFraction=branch.fraction*power.y;tir=power.y==0.0;
            if((power.x>0.0&&reflectedFraction==0.0)||(power.y>0.0&&transmittedFraction==0.0)) {
                tail+=potential*branch.fraction*(power.x+power.y);continue;
            }
)GLSL");
    smoothglass::replace(source,R"GLSL(    complete=complete&&tail<=launched*uPolicy.x*(1.0+2e-4);
)GLSL",R"GLSL(    complete=complete&&masked+tail<=launched*uPolicy.x*(1.0+2e-4);
)GLSL");
    smoothglass::replace(source,R"GLSL(    paths[base+3]=vec4(launched,outgoing,absorbed,escaped);paths[base+4]=vec4(tail,0.0,0.0,0.0);
)GLSL",R"GLSL(    paths[base+3]=vec4(launched,outgoing,absorbed,escaped);paths[base+4]=vec4(tail,0.0,0.0,0.0);
    if(uRoughness>0.0)paths[base+15]=vec4(masked,float(CAUSTIC_ROUGH_VERSION),float(interfaceSample),0.0);
)GLSL");
    return source;
}
// V2 emits every physical Smith-height exit through the same macro traversal.
// The released smooth and single-interface generators retain their bytes.
// Reserve the complete fixed16 pending array plus popped/pushed branch
// copies, not just the ancestry delta for the admitted pending limit. A branch
// storage model occupies96 bytes at16-byte alignment; reserve112 per
// branch for conservative native padding. Both exit/order argument
// copies consume320 bytes each. Reserve2048 for retained hit/entry values and
// nested query/walk scalar and vector locals without relying on lifetime reuse.
struct alignas(16) MultipleScatterBranchStorageShape
{
    alignas(16) std::array<float,3> origin,direction;
    float fraction;
    alignas(8) std::array<std::uint32_t,2> history;
    std::uint32_t interfaces,inside;
    alignas(16) std::array<std::uint32_t,4> ancestryLow,ancestryHigh;
    std::uint32_t samplingHistory;
};
static_assert(sizeof(MultipleScatterBranchStorageShape)==96);
inline constexpr std::uint64_t kMultipleScatterBranchBytes=112;
static_assert(sizeof(MultipleScatterBranchStorageShape)<=kMultipleScatterBranchBytes);
inline constexpr std::uint64_t kMultipleScatterNativeTemporaryBytesPerWork=
    (kMaximumPendingBranches+2)*kMultipleScatterBranchBytes+2*16*(16+4)+3*16+8+2048;
static_assert(kMultipleScatterPendingExtensionBytes>=48);
static_assert(smoothglass::kMaximumMicrofacetExits==16&&kMaximumPendingBranches==16);
inline std::string multipleScatterComputeSource()
{
    auto source=computeSource();
    smoothglass::replace(source,nativeMathSource(),nativeMultipleScatterMathSource());
    smoothglass::replace(source,"uniform int uWorkCount,uTriangleCount,uVisits,uInterfaces,uPending;",
        "uniform int uWorkCount,uTriangleCount,uVisits,uInterfaces,uPending,uInterfaceSamples,uMicroEvents,uMicroExits;\nuniform float uRoughness,uMicroResidual;");
    smoothglass::replace(source,"struct Branch {vec3 origin,direction;float fraction;uvec2 history;int interfaces;bool inside;};",
        "struct Branch {vec3 origin,direction;float fraction;uvec2 history;int interfaces;bool inside;uvec4 ancestryLow,ancestryHigh;uint samplingHistory;};");
    smoothglass::replace(source,"int band=int(entryIndices.w);float potential=causticEntryPower(uLi[band/4][band%4],normalPower.xyz,uSource.xyz,entry.w);",
        "int band=int(entryIndices.w);uint interfaceSample=(invocation/8u)%uint(uInterfaceSamples);\n"
        "    uint entryKey=causticMsEntryKey(id(entryOwners.xy),uint(entryIndices.x),uint(entryIndices.y),uint(entryIndices.z));\n"
        "    float potential=causticEntryPower(uLi[band/4][band%4],normalPower.xyz,uSource.xyz,entry.w)/float(uInterfaceSamples);");
    smoothglass::replace(source,"float outgoing=0.0,absorbed=0.0,escaped=0.0,tail=0.0;",
        "float outgoing=0.0,absorbed=0.0,escaped=0.0,tail=0.0,microResidualPower=0.0;uint microEvents=0u,microDraws=0u;");
    smoothglass::replace(source,"1.0,uvec2(0u),0,false);",
        "1.0,uvec2(0u),0,false,uvec4(0u),uvec4(0u),2166136261u);");
    smoothglass::replace(source,"            ++count;continue;",R"GLSL(            for(int token=0;token<32;++token) {
                uint ancestryWord=token<16?branch.ancestryLow[token/4]:branch.ancestryHigh[(token-16)/4];
                deposits[d+22+token/4][token%4]=float((ancestryWord>>uint(8*(token%4)))&255u);
            }
            ++count;continue;)GLSL");
    const auto begin=source.find("        vec3 transmitted,reflected;bool tir;float ior=");
    const auto end=source.find("\n    }\n    for(int i=0;i<pending;",begin);
    if(begin==std::string::npos||end==std::string::npos)return {};
    source.replace(begin,end-begin,R"GLSL(        float ior=uIor[band/4][band%4];
        vec4 exits[16];uint orders[16];uint exitCount,events,draws;float residual;
        uint seed=causticMsSeed(entryKey,interfaceSample,uint(branch.interfaces),branch.samplingHistory);
        int status=causticMsWalk(branch.direction,normal,branch.inside?ior:1.0,branch.inside?1.0:ior,
            uRoughness,seed,uint(uMicroEvents),uint(uMicroExits),uMicroResidual,
            exits,orders,exitCount,events,draws,residual);
        microEvents+=events;microDraws+=draws;
        if(status!=1||exitCount>uint(uMicroExits)||events>uint(uMicroEvents)||draws!=4u*events
            ||!causticFinite(residual)||residual<0.0||residual>uMicroResidual) {complete=false;break;}
        float sum=residual;
        for(uint child=0u;child<exitCount;++child)sum+=abs(exits[child].w);
        if(!causticFinite(sum)||abs(sum-1.0)>2e-5) {complete=false;break;}
        float unresolved=potential*branch.fraction*residual;
        microResidualPower+=unresolved;tail+=unresolved;
        for(uint child=0u;child<exitCount;++child) {
            vec4 emitted=exits[child];bool transmitted=emitted.w<0.0;
            float power=abs(emitted.w),fraction=branch.fraction*power;
            if(!causticFinite3(emitted.xyz)||!causticFinite(power)||power<=0.0||fraction<=0.0
                ||abs(dot(emitted.xyz,emitted.xyz)-1.0)>1e-4||orders[child]>uint(uMicroEvents)
                ||(ior==1.0?(orders[child]!=0u||!transmitted):orders[child]==0u)
                ||(transmitted?dot(emitted.xyz,normal)>=0.0:dot(emitted.xyz,normal)<=0.0)) {complete=false;break;}
            // Cutoffs and pending overflow account for each omitted exit once.
            if(fraction<uPolicy.y||pending>=uPending) {tail+=potential*fraction;continue;}
            uvec2 history=branch.history;
            if(!transmitted)history.x|=1u<<uint(branch.interfaces);
            uvec4 ancestryLow=branch.ancestryLow,ancestryHigh=branch.ancestryHigh;
            uint token=orders[child]|(transmitted?64u:0u),word=uint(branch.interfaces)/4u;
            if(word<4u)ancestryLow[word]|=token<<uint(8*(branch.interfaces%4));
            else ancestryHigh[word-4u]|=token<<uint(8*(branch.interfaces%4));
            stack[pending++]=Branch(hit.position+normal*(transmitted?-rayBias(hit.position):rayBias(hit.position)),
                emitted.xyz,fraction,history,branch.interfaces+1,transmitted?!branch.inside:branch.inside,
                ancestryLow,ancestryHigh,causticMsChildHistory(branch.samplingHistory,orders[child],transmitted));
        })GLSL");
    smoothglass::replace(source,"    paths[base+3]=vec4(launched,outgoing,absorbed,escaped);paths[base+4]=vec4(tail,0.0,0.0,0.0);",
        "    paths[base+3]=vec4(launched,outgoing,absorbed,escaped);paths[base+4]=vec4(tail,0.0,0.0,0.0);\n"
        "    paths[base+14]=vec4(float(microEvents),float(microDraws),microResidualPower,entryIndices.z);\n"
        "    paths[base+15]=vec4(0.0,float(CAUSTIC_ROUGH_VERSION),float(interfaceSample),float(GLASS_MS_VERSION));");
    return source;
}
inline bool decodeGratingIncident(const Texel* p,NativeGratingIncident& output)
{
    NativeGratingIncident result;
    for(unsigned lane=0;lane<kGratingIncidentStride;++lane)
        for(float value:p[lane])if(!std::isfinite(value))return false;
    for(unsigned lane:{16u,18u,19u,20u,21u})if(p[lane][3]!=0)return false;
    if(p[17][2]!=0||p[17][3]!=0||p[22][1]!=1)return false;
    std::uint32_t low=0,high=0,entering=0;
    if(!causticWord(p[0][0],kMaximumEntrySamples*8-1,result.work)||!causticWord(p[0][1],7,result.band)
        ||!causticWord(p[0][2],kMaximumInterfaces,result.interfaces)
        ||!causticWord(p[0][3],kGratingOrdersPerIncident,result.orderCount)
        ||!causticId(p[1][0],p[1][1],low)||!causticId(p[1][2],p[1][3],high)
        ||!causticId(p[13][0],p[13][1],result.hit.objectId)||!causticId(p[13][2],p[13][3],result.hit.materialId)
        // Glass schedules admit one instance per authored scene object.
        ||!causticWord(p[14][0],HarmonicMIDI::grid::Visual3DScene::kMaxObjects-1,result.hit.instanceId)
        ||!causticWord(p[14][1],kMaximumNativeDiffractionTriangles,result.hit.triangleId)
        ||!causticId(p[14][2],p[14][3],result.hit.work)||!causticWord(p[15][3],1,entering)
        ||!causticWord(p[22][0],kGratingIncidentVersion,result.version)||result.version!=kGratingIncidentVersion)return false;
    result.reflectionHistory=low|(std::uint64_t(high)<<32);
    for(unsigned index=0;index<32;++index) {
        std::uint32_t token=0;if(!causticWord(p[2+index/4][index%4],127,token))return false;
        result.ancestry[index]=std::uint8_t(token);
    }
    const auto point=[&](unsigned lane){return Point{p[lane][0],p[lane][1],p[lane][2]};};
    result.origin=point(10);result.fraction=p[10][3];result.direction=point(11);result.power=p[11][3];
    result.hit.status=1;result.hit.position=point(12);result.hit.distance=p[12][3];
    result.hit.outwardNormal=point(15);result.hit.entering=entering!=0;result.barycentric=point(16);
    result.uv={p[17][0],p[17][1]};result.tangent=point(18);result.bitangent=point(19);result.normal=point(20);
    result.dHdu={p[22][2],p[22][3],p[23][0]};result.dHdv={p[23][1],p[23][2],p[23][3]};
    result.substrateReflectedPower=p[21][0];result.absorbedPower=p[21][1];result.unresolvedPower=p[21][2];result.complete=true;
    output=result;return true;
}
inline bool decodeGratingOrder(const Texel* p,std::uint32_t incident,NativeGratingOrder& output)
{
    for(unsigned lane=0;lane<kGratingOrderStride;++lane)
        for(float value:p[lane])if(!std::isfinite(value))return false;
    std::uint32_t locator=0,signedWord=0;
    if(!causticWord(p[0][0],kMaximumEntrySamples*8*kMaximumBranchVisits-1,locator)||locator!=incident
        ||!causticWord(p[0][1],2,signedWord)||p[0][2]!=0||p[0][3]!=0||p[2]!=Texel{}||p[3]!=Texel{})return false;
    NativeGratingOrder result;result.order=std::int32_t(signedWord)-1;
    result.direction={p[1][0],p[1][1],p[1][2]};result.power=p[1][3];output=result;return true;
}
// The caller supplies exact function bodies from the existing native material
// shader. No order efficiencies or substrate constants come from a CPU oracle.
inline std::string gratingIncidentComputeSource(const std::string& materialMath,const std::string& orderEquation)
{
    if(materialMath.empty()||orderEquation.empty())return {};
    auto source=computeSource();
    smoothglass::replace(source,"layout(std430,binding=1) buffer Deposits {vec4 deposits[];};",
        "layout(std430,binding=1) buffer Incidents {vec4 incidents[];};\nlayout(std430,binding=2) buffer Orders {vec4 orders[];};\n"
        "layout(std430,binding=5) readonly buffer Witness {vec4 witness[];};\nuniform int uWitnessCount;\n"
        "uniform vec4 uDiffractionGeometry,uDiffractionMicrostructure,uDiffractionControl,uDiffractionCoating,uDiffractionRoughness,uDiffractionSpectral[8];\n");
    const auto helpers=source.find("uint queryWork=0u;");
    if(helpers==std::string::npos)return {};
    source.insert(helpers,"#define CAUSTIC_GRATING_VERSION "+std::to_string(kGratingIncidentVersion)+"\n"+materialMath+"\nbool gratingFinite3(vec3 p) {return causticFinite(p.x)&&causticFinite(p.y)&&causticFinite(p.z);}\n");
    const auto hitBegin=source.find("void depositHit(");const auto hitEnd=source.find("int findTriangle(",hitBegin);
    if(hitBegin==std::string::npos||hitEnd==std::string::npos)return {};
    source.erase(hitBegin,hitEnd-hitBegin);
    smoothglass::replace(source,"    int depositBase=w*uVisits*CAUSTIC_DEPOSIT_RECEIPT_STRIDE;\n","");
    smoothglass::replace(source,"    for(int i=0;i<uVisits*CAUSTIC_DEPOSIT_RECEIPT_STRIDE;++i)deposits[depositBase+i]=vec4(0.0);",
        "    for(int i=0;i<uVisits*24;++i)incidents[w*uVisits*24+i]=vec4(0.0);\n"
        "    for(int i=0;i<uVisits*3*4;++i)orders[w*uVisits*3*4+i]=vec4(0.0);");
    smoothglass::replace(source,"float outgoing=0.0,absorbed=0.0,escaped=0.0,tail=0.0;",
        "float outgoing=0.0,absorbed=0.0,escaped=0.0,tail=0.0;uint extraQueryWork=0u,orderWork=0u;");
    smoothglass::replace(source,"struct Branch {vec3 origin,direction;float fraction;uvec2 history;int interfaces;bool inside;};",
        "struct Branch {vec3 origin,direction;float fraction;uvec2 history;int interfaces;bool inside;vec3 du,dv;};\n"+nativeGratingDifferentialMathSource());
    smoothglass::replace(source,"    Branch stack[CAUSTIC_MAX_PENDING];",
        "    Branch stack[CAUSTIC_MAX_PENDING];vec3 entryE1=vec3(0.0),entryE2=vec3(0.0);");
    smoothglass::replace(source,"        if(complete)stack[pending++]=Branch(entry.xyz+uSource.xyz*(2.0*offset),-uSource.xyz,1.0,uvec2(0u),0,false);",R"GLSL(        if(complete) {
            int sourceTriangle=findTriangle(entryHit);float dbu,dbv;
            if(sourceTriangle<0) {complete=false;}
            else {
                int t=sourceTriangle*8;entryE1=triangles[t+1].xyz-triangles[t].xyz;entryE2=triangles[t+2].xyz-triangles[t].xyz;
                complete=triangles[t+5].z==1.0&&gratingStrictHit(triangles[t].xyz,triangles[t+1].xyz,triangles[t+2].xyz,entryHit.position,entryHit.outwardNormal)
                    &&gratingBiasDerivative(entry.xyz,entryE1,dbu)&&gratingBiasDerivative(entry.xyz,entryE2,dbv);
                if(complete)stack[pending++]=Branch(entry.xyz+uSource.xyz*(2.0*offset),-uSource.xyz,1.0,uvec2(0u),0,false,
                    entryE1+2.0*uSource.xyz*dbu,entryE2+2.0*uSource.xyz*dbv);
            }
        })GLSL");
    smoothglass::replace(source,"        vec4 shape=triangles[triangle*8+5];",R"GLSL(        vec4 shape=triangles[triangle*8+5];vec3 hitDu,hitDv;
        int plane=triangle*8;
        if(!gratingStrictHit(triangles[plane].xyz,triangles[plane+1].xyz,triangles[plane+2].xyz,hit.position,hit.outwardNormal)
            ||!gratingPlaneDerivative(branch.du,branch.direction,hit.outwardNormal,hitDu)
            ||!gratingPlaneDerivative(branch.dv,branch.direction,hit.outwardNormal,hitDv)
            ||!gratingArea(entryE1,entryE2,entry.w,hitDu,hitDv)) {complete=false;break;})GLSL");
    smoothglass::replace(source,"            stack[pending++]=Branch(hit.position+normal*(reflect?rayBias(hit.position):-rayBias(hit.position)),\n                reflect?reflected:transmitted,fraction,history,branch.interfaces+1,reflect?branch.inside:!branch.inside);",R"GLSL(            float dbu,dbv;
            if(!gratingBiasDerivative(hit.position,hitDu,dbu)||!gratingBiasDerivative(hit.position,hitDv,dbv)) {complete=false;break;}
            float side=reflect?1.0:-1.0;
            stack[pending++]=Branch(hit.position+normal*(reflect?rayBias(hit.position):-rayBias(hit.position)),
                reflect?reflected:transmitted,fraction,history,branch.interfaces+1,reflect?branch.inside:!branch.inside,
                hitDu+side*normal*dbu,hitDv+side*normal*dbv);)GLSL");
    const auto receiverBegin=source.find("        if(shape.z==0.0) {");
    const auto receiverEnd=source.find("        if(branch.inside==hit.entering)",receiverBegin);
    if(receiverBegin==std::string::npos||receiverEnd==std::string::npos)return {};
    source.replace(receiverBegin,receiverEnd-receiverBegin,R"GLSL(        if(shape.z==0.0) {
            if(branch.inside||!hit.entering||branch.interfaces==0) {complete=false;break;}
            int target=-1;
            for(int t=0;t<uWitnessCount;++t) {
                vec4 o=witness[t*8+4],i=witness[t*8+5];
                if(id(o.xy)==hit.objectId&&id(o.zw)==hit.materialId&&uint(i.x)==hit.instanceId&&uint(i.y)==hit.triangleId) {target=t;break;}
            }
            // Other staging receivers terminate their actual macro branch.
            // Their ordinary power is not relabeled as a grating event.
            if(target<0)continue;
            int t=target*8;
            vec3 a=witness[t].xyz,e1=witness[t+1].xyz-a,e2=witness[t+2].xyz-a,delta=hit.position-a;
            float q00=dot(e1,e1),q01=dot(e1,e2),q11=dot(e2,e2),det=q00*q11-q01*q01;
            if(!causticFinite(det)||det<=0.0) {complete=false;break;}
            float b1=(q11*dot(delta,e1)-q01*dot(delta,e2))/det,b2=(q00*dot(delta,e2)-q01*dot(delta,e1))/det;
            vec3 barycentric=vec3(1.0-b1-b2,b1,b2);
            vec2 uv=vec2(dot(barycentric,vec3(witness[t].w,witness[t+1].w,witness[t+2].w)),dot(barycentric,witness[t+3].xyz));
            vec3 tangent=witness[t+6].xyz,normal=witness[t+7].xyz,bitangent=cross(normal,tangent);
            vec3 incident=-vec3(dot(branch.direction,tangent),dot(branch.direction,bitangent),dot(branch.direction,normal));
            float tolerance=8.0*rayBias(hit.position);
            if(!gratingFinite3(barycentric)||any(lessThanEqual(barycentric,vec3(1e-5)))||any(greaterThanEqual(barycentric,vec3(1.0-1e-5)))
                ||length(a+e1*b1+e2*b2-hit.position)>tolerance||length(normal-hit.outwardNormal)>2e-4
                ||incident.z<=1e-6||!gratingArea(entryE1,entryE2,entry.w,hitDu,hitDv)) {complete=false;break;}
            float biasDu,biasDv;
            if(!gratingBiasDerivative(hit.position,hitDu,biasDu)||!gratingBiasDerivative(hit.position,hitDv,biasDv)) {complete=false;break;}
            float incidentPower=potential*branch.fraction;
            float wavelength=uDiffractionSpectral[band].x;
            float substrate=spectralReflectance(wavelength,int(uDiffractionControl.w+0.5),uDiffractionMicrostructure.y,uDiffractionMicrostructure.z);
            if(!causticFinite(substrate)||substrate<0.0||substrate>1.0) {complete=false;break;}
            int locator=w*uVisits+count,d=locator*24,orderCount=0;float resolved=0.0,efficiencySum=0.0;
            orderWork+=3u;
            for(int primary=-1;primary<=1;++primary) {
                /*GRATING_NATIVE_ORDER_EQUATION*/
                if(!causticFinite(efficiency)||efficiency<0.0) {complete=false;break;}
                efficiencySum+=efficiency;float power=incidentPower*substrate*efficiency;
                if(power==0.0)continue;
                vec3 direction=tangent*centre.x+bitangent*centre.y+normal*z;
                if(!gratingFinite3(direction)||!causticFinite(power)||power<0.0||abs(dot(direction,direction)-1.0)>1e-4
                    ||visits>=uVisits) {complete=false;break;}
                ++visits;
                SceneClosestHit childHit=query(hit.position+normal*rayBias(hit.position),direction,1e-5,uSource.w);
                extraQueryWork+=causticLastWork;
                if(childHit.status<0) {complete=false;break;}
                int orderBase=(locator*3+orderCount)*4;
                orders[orderBase]=vec4(float(locator),float(primary+1),0.0,0.0);
                orders[orderBase+1]=vec4(direction,power);++orderCount;resolved+=power;
            }
            if(!complete||!causticFinite(efficiencySum)||efficiencySum>1.0002) {complete=false;break;}
            incidents[d]=vec4(float(w),float(band),float(branch.interfaces),float(orderCount));
            incidents[d+1]=vec4(words(branch.history.x),words(branch.history.y));
            for(int depth=0;depth<branch.interfaces;++depth)
                incidents[d+2+depth/4][depth%4]=((branch.history.x>>uint(depth))&1u)!=0u?1.0:65.0;
            incidents[d+10]=vec4(branch.origin,branch.fraction);incidents[d+11]=vec4(branch.direction,incidentPower);
            incidents[d+12]=vec4(hit.position,hit.distance);incidents[d+13]=owners(hit);
            incidents[d+14]=vec4(float(hit.instanceId),float(hit.triangleId),words(hitWork));
            incidents[d+15]=vec4(hit.outwardNormal,float(hit.entering));incidents[d+16]=vec4(barycentric,0.0);
            incidents[d+17]=vec4(uv,0.0,0.0);incidents[d+18]=vec4(tangent,0.0);incidents[d+19]=vec4(bitangent,0.0);
            incidents[d+20]=vec4(normal,0.0);incidents[d+21]=vec4(incidentPower*substrate,incidentPower*(1.0-substrate),max(0.0,incidentPower*substrate-resolved),0.0);
            incidents[d+22]=vec4(float(CAUSTIC_GRATING_VERSION),1.0,hitDu.xy);incidents[d+23]=vec4(hitDu.z,hitDv);++count;continue;
        }
)GLSL");
    smoothglass::replace(source,"    paths[base+1]=vec4(float(complete),float(blocked),0.0,0.0);",
        "    paths[base+1]=vec4(float(complete),float(blocked),words(extraQueryWork));");
    smoothglass::replace(source,"paths[base+4]=vec4(tail,0.0,0.0,0.0);",
        "paths[base+4]=vec4(tail,float(orderWork),0.0,0.0);");
    smoothglass::replace(source,"/*GRATING_NATIVE_ORDER_EQUATION*/",orderEquation);
    return source;
}

// Explicit dimensional source route. Legacy generators retain their bytes.
// std430 input48 + echo96 + one pre-blocking potential per scheduled work item.
inline std::string directionalIrradianceComputeSource(std::string source)
{
    const auto declarations=source.find("uint queryWork=0u;");
    const auto potential=source.find("uLi[band/4][band%4]");
    const auto receipt=source.find("    paths[base]=vec4(words(invocation),0.0,0.0);");
    if(declarations==std::string::npos||potential==std::string::npos||receipt==std::string::npos
        ||source.find("uLi[band/4][band%4]",potential+1)!=std::string::npos)return {};
    source.replace(receipt,0,R"GLSL(    directionalPotentials[w]=potential;
    if(invocation==0u) {
        directionalEchoIdentifiers=directionalIdentifiers;
        directionalEchoCoefficients[0]=directionalCoefficients[0];
        directionalEchoCoefficients[1]=directionalCoefficients[1];
        directionalEchoRawRadiance[0]=uLi[0];directionalEchoRawRadiance[1]=uLi[1];
        directionalEchoDirectionTrace=uSource;
    }
)GLSL");
    source.replace(potential,std::string("uLi[band/4][band%4]").size(),"directionalCoefficients[band/4][band%4]");
    // Guard before any dimensional power is computed. Invalid inputs leave
    // incomplete path receipts and cannot pass the whole-batch validation.
    const auto entry=source.find("    int band=int(entryIndices.w);");
    if(entry==std::string::npos)return {};
    source.insert(entry,R"GLSL(    if(directionalIdentifiers.x!=1u||directionalIdentifiers.y!=1u
        ||directionalIdentifiers.z!=1u||directionalIdentifiers.w==0u)return;
    for(int b=0;b<8;++b) {
        float coefficient=directionalCoefficients[b/4][b%4];
        if(!causticFinite(coefficient)||coefficient<0.0||coefficient>16.0)return;
    }
)GLSL");
    source.insert(declarations,R"GLSL(layout(std430,binding=7) buffer DirectionalSource {
    uvec4 directionalIdentifiers;
    vec4 directionalCoefficients[2];
    uvec4 directionalEchoIdentifiers;
    vec4 directionalEchoCoefficients[2];
    vec4 directionalEchoRawRadiance[2];
    vec4 directionalEchoDirectionTrace;
    float directionalPotentials[];
};
)GLSL");
    return source;
}

inline bool decodeGratingWork(const Texel* p,NativeGratingReadback::WorkReceipt& output)
{
    NativeGratingReadback::WorkReceipt r;std::uint32_t queries=0;
    for(unsigned t=0;t<kGratingWorkReceiptStride;++t)for(float value:p[t])if(!std::isfinite(value))return false;
    if(!causticWord(p[0][0],kMaximumBranchVisits,r.incidentCount)
        ||!causticWord(p[0][1],kMaximumBranchVisits*3,r.orderWork)||r.orderWork!=r.incidentCount*3
        ||!causticId(p[0][2],p[0][3],queries)||p[1][0]<0||p[1][1]<0||p[1][2]!=1||p[1][3]!=0)return false;
    r.queryWork=queries;r.absorbedPower=p[1][0];r.unresolvedPower=p[1][1];r.complete=true;output=r;return true;
}
inline bool decodeGratingChild(const Texel* p,NativeGratingReadback::ChildReceipt& output)
{
    NativeGratingReadback::ChildReceipt r;std::uint32_t order=0,disposition=0,queries=0;
    for(unsigned t=0;t<kGratingChildReceiptStride;++t)for(float value:p[t])if(!std::isfinite(value))return false;
    if(!causticWord(p[0][0],kMaximumEntrySamples*8*kMaximumBranchVisits-1,r.incident)
        ||!causticWord(p[0][1],2,order)||p[0][2]!=1||!causticWord(p[0][3],2,disposition)
        ||p[1][0]<0||p[1][1]<0||!causticId(p[1][2],p[1][3],queries)||p[7]!=Texel{}||!causticHit(p+2,r.hit))return false;
    r.order=int(order)-1;r.disposition=NativeGratingReadback::Disposition(disposition);r.queryWork=queries;
    r.escapedPower=p[1][0];r.unresolvedPower=p[1][1];r.complete=true;output=r;return true;
}
inline bool decodeGratingDeposit(const Texel* p,NativeDeposit& output)
{
    std::array<Texel,kDepositReceiptStride> legacy;std::copy_n(p,legacy.size(),legacy.begin());
    std::fill(legacy.begin()+22,legacy.end(),Texel{});NativeDeposit r;
    if(!causticDecodeDeposit(legacy.data(),0,r)||!decodeGratingDepositExtension(p[30],p[31],r))return false;
    std::array<Texel,8> ancestry;std::copy_n(p+22,ancestry.size(),ancestry.begin());
    if(r.gratingVersion) {if(!decodeGratingDepositAncestry(ancestry,r.ancestry))return false;}
    else if(!decodeMacroAncestry(ancestry,0,r.ancestry))return false;
    output=r;return true;
}
inline std::string gratingTerminalComputeSource(const std::string& materialMath,const std::string& orderEquation)
{
    auto source=gratingIncidentComputeSource(materialMath,orderEquation);if(source.empty())return {};
    const auto original=computeSource();
    const auto receiverBegin=original.find("        if(shape.z==0.0) {");
    const auto receiverEnd=original.find("        if(branch.inside==hit.entering)",receiverBegin);
    const auto hitBegin=original.find("void depositHit(");const auto hitEnd=original.find("int findTriangle(",hitBegin);
    if(receiverBegin==std::string::npos||receiverEnd==std::string::npos||hitBegin==std::string::npos||hitEnd==std::string::npos)return {};
    auto receiver=original.substr(receiverBegin,receiverEnd-receiverBegin);
    smoothglass::replace(receiver,"        if(shape.z==0.0) {",R"GLSL(bool gratingReceiver(int w,int band,float potential,Branch branch,SceneClosestHit hit,uint hitWork,int triangle,
    bool coupled,int locator,int signedOrder,inout int depositCount,inout float outgoing,inout float absorbed,out uint cameraQueries) {
    cameraQueries=0u;vec4 shape=triangles[triangle*8+5];
    if(depositCount>=uVisits)return false;
)GLSL");
    smoothglass::replace(receiver,"complete=false;break;","return false;");
    smoothglass::replace(receiver,"int d=depositBase+count*CAUSTIC_DEPOSIT_RECEIPT_STRIDE;","int d=(w*uVisits+depositCount)*CAUSTIC_DEPOSIT_RECEIPT_STRIDE;");
    smoothglass::replace(receiver,"words(invocation)","words(uint(w))");
    smoothglass::replace(receiver,"++count;continue;",R"GLSL(if(coupled) {
                for(int depth=0;depth<branch.interfaces;++depth)
                    deposits[d+22+depth/4][depth%4]=((branch.history.x>>uint(depth))&1u)!=0u?1.0:65.0;
                deposits[d+30]=vec4(float(CAUSTIC_GRATING_VERSION),float(locator),float(signedOrder+1),0.0);
            }
            cameraQueries=cameraWork+pixelWork;++depositCount;return true;)GLSL");
    const auto main=source.find("void main() {");if(main==std::string::npos)return {};
    source.insert(main,original.substr(hitBegin,hitEnd-hitBegin)+R"GLSL(
void gratingChildHit(int offset,SceneClosestHit hit,uint work) {
    orders[offset]=vec4(float(hit.status),hit.distance,float(hit.entering),float(work&65535u));
    orders[offset+1]=vec4(hit.position,float(work>>16u));orders[offset+2]=vec4(hit.outwardNormal,0.0);
    orders[offset+3]=owners(hit);orders[offset+4]=indices(hit);
}
)GLSL"+receiver);
    smoothglass::replace(source,"uniform int uWitnessCount;","layout(std430,binding=6) buffer Deposits {vec4 deposits[];};\nuniform int uWitnessCount,uGratingWorkBase,uGratingChildBase;");
    smoothglass::replace(source,"uint extraQueryWork=0u,orderWork=0u;","uint extraQueryWork=0u,orderWork=0u;int depositCount=0;float gratingAbsorbed=0.0,gratingTail=0.0;");
    smoothglass::replace(source,"    // Only this invocation writes its fixed receipt segment.",R"GLSL(    // Only this invocation writes its fixed receipt segment.
    for(int i=0;i<uVisits*CAUSTIC_DEPOSIT_RECEIPT_STRIDE;++i)deposits[w*uVisits*CAUSTIC_DEPOSIT_RECEIPT_STRIDE+i]=vec4(0.0);
    for(int i=0;i<uVisits*3*8;++i)orders[uGratingChildBase+w*uVisits*3*8+i]=vec4(0.0);
    incidents[uGratingWorkBase+w*2]=vec4(0.0);incidents[uGratingWorkBase+w*2+1]=vec4(0.0);
)GLSL");
    smoothglass::replace(source,"            if(target<0)continue;",R"GLSL(            if(target<0) {
                uint cameraQueries;
                if(!gratingReceiver(w,band,potential,branch,hit,hitWork,triangle,false,0,0,depositCount,outgoing,absorbed,cameraQueries)) {complete=false;break;}
                continue;
            })GLSL");
    smoothglass::replace(source,"                if(power==0.0)continue;",R"GLSL(                if(power==0.0) {
                    if(incidentPower>0.0&&substrate>0.0&&efficiency>0.0) {complete=false;break;}
                    continue;
                })GLSL");
    const auto orderBegin=source.find("                ++visits;\n                SceneClosestHit childHit=");
    const auto orderEnd=source.find("                int orderBase=(locator*3+orderCount)*4;",orderBegin);
    if(orderBegin==std::string::npos||orderEnd==std::string::npos)return {};
    source.replace(orderBegin,orderEnd-orderBegin,R"GLSL(                ++visits;
                SceneClosestHit childHit=query(hit.position+normal*rayBias(hit.position),direction,1e-5,uSource.w);
                uint childWork=causticLastWork;extraQueryWork+=childWork;
                if(childHit.status<0) {complete=false;break;}
                int childBase=uGratingChildBase+(locator*3+orderCount)*8;uint cameraQueries=0u;
                if(childHit.status==0) {
                    escaped+=power;orders[childBase]=vec4(float(locator),float(primary+1),1.0,0.0);
                    orders[childBase+1]=vec4(power,0.0,words(childWork));
                } else {
                    int childTriangle=findTriangle(childHit);
                    if(childTriangle<0||childHit.objectId==hit.objectId||triangles[childTriangle*8+5].z!=0.0) {complete=false;break;}
                    float fraction=power/potential;
                    if(!causticFinite(fraction)||fraction<=0.0) {complete=false;break;}
                    Branch child=branch;child.origin=hit.position+normal*rayBias(hit.position);child.direction=direction;child.fraction=fraction;child.inside=false;
                    if(!gratingReceiver(w,band,potential,child,childHit,childWork,childTriangle,true,locator,primary,
                        depositCount,outgoing,absorbed,cameraQueries)) {complete=false;break;}
                    extraQueryWork+=cameraQueries;
                    orders[childBase]=vec4(float(locator),float(primary+1),1.0,1.0);
                    orders[childBase+1]=vec4(0.0,0.0,words(childWork+cameraQueries));
                }
                gratingChildHit(childBase+2,childHit,childWork);
)GLSL");
    smoothglass::replace(source,"            incidents[d+20]=vec4(normal,0.0);incidents[d+21]=",R"GLSL(            float substrateAbsorption=incidentPower*(1.0-substrate),profileTail=max(0.0,incidentPower*substrate-resolved);
            absorbed+=substrateAbsorption;tail+=profileTail;gratingAbsorbed+=substrateAbsorption;gratingTail+=profileTail;
            incidents[d+20]=vec4(normal,0.0);incidents[d+21]=)GLSL");
    smoothglass::replace(source,"    paths[base].zw=vec2(float(count),float(visits));", "    paths[base].zw=vec2(float(depositCount),float(visits));");
    smoothglass::replace(source,"    paths[base+1]=vec4(float(complete),float(blocked),words(extraQueryWork));",R"GLSL(    incidents[uGratingWorkBase+w*2]=vec4(float(count),float(orderWork),words(extraQueryWork));
    incidents[uGratingWorkBase+w*2+1]=vec4(gratingAbsorbed,gratingTail,float(complete),0.0);
    paths[base+1]=vec4(float(complete),float(blocked),words(extraQueryWork));)GLSL");
    return source;
}

inline constexpr const char* depositionKeySource=R"GLSL(#version 430 core
layout(local_size_x=32) in;
layout(std430,binding=0) readonly buffer Paths {vec4 paths[];};
layout(std430,binding=1) readonly buffer Deposits {vec4 deposits[];};
layout(std430,binding=5) buffer Keys {uvec4 keys[];};
uniform int uWorkCount,uVisits,uKeyCount;
uniform vec2 uExtent;
void main() {
    uint at=gl_GlobalInvocationID.x;if(at>=uint(uKeyCount))return;
    uvec4 key=uvec4(0xffffffffu);
    uint work=at/uint(uVisits),slot=at%uint(uVisits);
    if(work<uint(uWorkCount)&&slot<uint(paths[work*16u].z)) {
        uint deposit=at*32u;
        vec4 target=deposits[deposit+3u];
        if(deposits[deposit].w==1.0&&target.x==1.0&&target.y>=0.0&&target.z>=0.0
            &&target.y<uExtent.x&&target.z<uExtent.y)
            key=uvec4(uint(target.z)*uint(uExtent.x)+uint(target.y),work,slot,at);
    }
    keys[at]=key;
}
)GLSL";
inline constexpr const char* depositionSortSource=R"GLSL(#version 430 core
layout(local_size_x=32) in;
layout(std430,binding=5) buffer Keys {uvec4 keys[];};
uniform int uKeyCount,uSortSpan,uSortDistance;
bool lessKey(uvec4 a,uvec4 b) {
    if(a.x!=b.x)return a.x<b.x;
    if(a.y!=b.y)return a.y<b.y;
    if(a.z!=b.z)return a.z<b.z;
    return a.w<b.w;
}
void main() {
    uint pair=gl_GlobalInvocationID.x;if(pair>=uint(uKeyCount)/2u)return;
    uint distance=uint(uSortDistance);
    uint first=(pair/distance)*(2u*distance)+pair%distance,second=first+distance;
    uvec4 a=keys[first],b=keys[second];
    bool ascending=(first&uint(uSortSpan))==0u;
    if(ascending?lessKey(b,a):lessKey(a,b)) {keys[first]=b;keys[second]=a;}
}
)GLSL";
inline std::string depositionReduceSource()
{
    return std::string("#version 430 core\n")+nativeMathSource()+R"GLSL(
layout(local_size_x=32) in;
layout(std430,binding=1) readonly buffer Deposits {vec4 deposits[];};
layout(std430,binding=2) buffer SpectrumLow {uint bandsLow[];};
layout(std430,binding=6) buffer SpectrumHigh {uint bandsHigh[];};
layout(std430,binding=3) readonly buffer Work {vec4 workTable[];};
layout(std430,binding=5) readonly buffer Keys {uvec4 keys[];};
uniform int uKeyCount;
void main() {
    uint at=gl_GlobalInvocationID.x;if(at>=uint(uKeyCount))return;
    uint pixel=keys[at].x;
    if(pixel==0xffffffffu||(at>0u&&keys[at-1u].x==pixel))return;
    float sum[8];for(int b=0;b<8;++b)sum[b]=0.0;
    // One segment leader writes each pixel. Lexicographic keys preserve the
    // native work/deposit order without any unordered floating-point atomics.
    for(uint next=at;next<uint(uKeyCount)&&keys[next].x==pixel;++next) {
        uvec4 key=keys[next];uint deposit=key.w*uint(CAUSTIC_DEPOSIT_RECEIPT_STRIDE);
        int band=int(workTable[key.y*4u+3u].w);
        sum[band]+=causticPixelRadiance(deposits[deposit+2u].z,deposits[deposit+4u].x);
    }
    for(int b=0;b<4;++b) {bandsLow[pixel*4u+uint(b)]=floatBitsToUint(sum[b]);
        bandsHigh[pixel*4u+uint(b)]=floatBitsToUint(sum[b+4]);}
}
)GLSL";
}
inline std::string rasterVertexSource()
{
    return std::string("#version 430 core\n")+nativeMathSource()+R"GLSL(
layout(std430,binding=4) readonly buffer Triangles {vec4 triangles[];};
uniform int uTriangle;
uniform vec4 uCamera,uRotation,uProjection;
out float linearDepth;
void main() {
    vec3 world=triangles[uTriangle*8+gl_VertexID].xyz;
    vec3 local=causticRotate(vec4(-uRotation.xyz,uRotation.w),world-uCamera.xyz);
    float depth=-local.z,n=uProjection.z,f=uProjection.w;linearDepth=depth;
    gl_Position=vec4(local.x/(uProjection.x*uProjection.y),local.y/uProjection.x,
        ((f+n)/(f-n))*depth-2.0*f*n/(f-n),depth);
}
)GLSL";
}
inline constexpr const char* rasterFragmentSource=R"GLSL(#version 430 core
layout(std430,binding=4) readonly buffer Triangles {vec4 triangles[];};
uniform int uTriangle;
in float linearDepth;
layout(location=0) out vec4 owners;
layout(location=1) out vec4 identityDepth;
void main() {
    owners=triangles[uTriangle*8+4];identityDepth=vec4(triangles[uTriangle*8+5].xy,linearDepth,1.0);
}
)GLSL";

inline constexpr unsigned kGratingCameraPrimitiveStride=12;
struct alignas(16) DirectTerminalReceiverPacked
{
    std::array<std::uint32_t,4> key{};
    std::array<std::int32_t,4> order{};
    std::array<Texel,10> data{};
};
static_assert(sizeof(DirectTerminalReceiverPacked)==12*sizeof(Texel));
inline bool decodeDirectTerminalReceiver(const DirectTerminalReceiverPacked& packed,
    const NativeDirectGratingTerminalReceiverInput& input,const DirectGratingTerminalOrder& emitted,
    DirectGratingTerminalReceiverRow& output)
{
    if(packed.key!=input.key||packed.order!=input.order)return false;
    for(const auto& t:packed.data)for(float v:t)if(!std::isfinite(v))return false;
    const auto* p=packed.data.data();DirectGratingTerminalReceiverRow row;row.input=input;row.emitted=emitted;
    std::uint32_t disposition=0,sample=0,projected=0;
    if(!causticHit(p,row.hit)||!exactReceiptInteger(p[5][0],UINT32_MAX,row.pixel.x)
        ||!exactReceiptInteger(p[5][1],UINT32_MAX,row.pixel.y)||!causticWord(p[5][3],3,disposition)
        ||!causticWord(p[8][0],1,projected)||!causticWord(p[8][1],2,sample)
        ||p[8][2]!=0||p[8][3]!=0||p[9]!=Texel{})return false;
    row.pixel.linearDepth=p[5][2];row.pixel.projected=projected!=0;
    row.disposition=DirectGratingTerminalReceiverRow::Disposition(disposition);
    row.rasterSample=DirectGratingTerminalReceiverRow::RasterSample(sample);row.queryWork=row.hit.work;
    if(sample==2) {
        if(p[7][3]!=1||!causticId(p[6][0],p[6][1],row.raster.object)
            ||!causticId(p[6][2],p[6][3],row.raster.material)
            ||!exactReceiptInteger(p[7][0],UINT32_MAX,row.raster.instance)
            ||!exactReceiptInteger(p[7][1],UINT32_MAX,row.raster.triangle))return false;
        row.raster.linearDepth=p[7][2];
    } else if(p[6]!=Texel{}||p[7]!=Texel{})return false;
    output=row;return true;
}
inline std::string directTerminalReceiverComputeSource()
{
    return std::string("#version 430 core\n")+querySource()+nativeMathSource()+R"GLSL(
layout(local_size_x=32) in;
struct Input {uvec4 key;ivec4 order;vec4 originMinimum;vec4 directionMaximum;};
struct Row {uvec4 key;ivec4 order;vec4 data[10];};
layout(std430,binding=0) readonly buffer Inputs {Input inputs[];};
layout(std430,binding=1) buffer Results {Row rows[];};
layout(std430,binding=2) readonly buffer Triangles {vec4 triangles[];};
uniform sampler2D uRasterOwners,uRasterTriangle;
uniform int uRowCount,uTriangleCount;
uniform vec4 uCamera,uRotation,uProjection;
uniform vec2 uExtent;
vec2 words(uint n) {return vec2(n&65535u,n>>16u);}
uint id(vec2 v) {return uint(v.x)|(uint(v.y)<<16u);}
void main() {
    uint index=gl_GlobalInvocationID.x;if(index>=uint(uRowCount))return;
    Input rayInput=inputs[index];rows[index].key=rayInput.key;rows[index].order=rayInput.order;
    for(int i=0;i<10;++i)rows[index].data[i]=vec4(0);
    causticLastWork=0u;
    SceneClosestHit hit=sceneClosestHit(rayInput.originMinimum.xyz,rayInput.directionMaximum.xyz,
        rayInput.originMinimum.w,rayInput.directionMaximum.w,uOpaqueControl);
    rows[index].data[0]=vec4(float(hit.status),hit.distance,float(hit.entering),float(causticLastWork&65535u));
    rows[index].data[1]=vec4(hit.position,float(causticLastWork>>16u));
    rows[index].data[2]=vec4(hit.outwardNormal,0);
    rows[index].data[3]=vec4(words(hit.objectId),words(hit.materialId));
    rows[index].data[4]=vec4(words(hit.instanceId),words(hit.triangleId));
    if(hit.status!=1)return;
    rows[index].data[5].w=1.0;
    int triangle=-1;
    for(int t=0;t<uTriangleCount;++t) {
        vec4 owner=triangles[t*8+4],shape=triangles[t*8+5];
        if(id(owner.xy)==hit.objectId&&id(owner.zw)==hit.materialId
            &&uint(shape.x)==hit.instanceId&&uint(shape.y)==hit.triangleId) {triangle=t;break;}
    }
    if(triangle<0) {rows[index].data[0].x=-1.0;return;}
    vec4 shape=triangles[triangle*8+5];
    if(shape.z!=0.0||shape.w!=1.0||!hit.entering)return;
    rows[index].data[5].w=2.0;
    vec3 local=causticRotate(vec4(-uRotation.xyz,uRotation.w),hit.position-uCamera.xyz);
    float depth=-local.z;rows[index].data[5].z=depth;
    if(!causticFinite(depth)||depth<uProjection.z||depth>uProjection.w)return;
    vec2 projected=vec2((local.x/(depth*uProjection.x*uProjection.y)+1.0)*.5*uExtent.x,
        (1.0-local.y/(depth*uProjection.x))*.5*uExtent.y);
    if(any(isnan(projected))||any(isinf(projected))) {rows[index].data[0].x=-1.0;return;}
    if(any(lessThan(projected,vec2(0)))||any(greaterThanEqual(projected,uExtent)))return;
    uvec2 pixel=uvec2(projected);rows[index].data[5].xy=vec2(pixel);
    vec3 centreRay=causticCameraRay(vec2(pixel)+vec2(.5),uRotation,uProjection,uExtent);
    if(dot(hit.outwardNormal,centreRay)>=-1e-6)return;
    rows[index].data[5].w=3.0;rows[index].data[8]=vec4(1,1,0,0);
    ivec2 rasterPixel=ivec2(pixel.x,uint(uExtent.y)-1u-pixel.y);
    vec4 owner=texelFetch(uRasterOwners,rasterPixel,0),raster=texelFetch(uRasterTriangle,rasterPixel,0);
    // A background sample stays a truthful background. A different foreground
    // owner is retained without asserting visibility of the outgoing hit.
    if(raster.w==0.0&&owner==vec4(0))return;
    rows[index].data[6]=owner;rows[index].data[7]=raster;rows[index].data[8].y=2.0;
}
)GLSL";
}

inline std::string gratingCameraComputeSource()
{
    return std::string("#version 430 core\n")+querySource()+nativeMathSource()+R"GLSL(
layout(local_size_x=32) in;
layout(std430,binding=0) buffer CameraReceipts {vec4 receipts[];};
layout(std430,binding=1) readonly buffer Incidents {vec4 incidents[];};
layout(std430,binding=2) readonly buffer Orders {vec4 orders[];};
uniform sampler2D uRasterOwners,uRasterTriangle;
uniform int uOrderCount,uEntrySubdivisions;
uniform vec4 uCamera,uRotation,uProjection;
uniform vec2 uExtent;
uint id(vec2 v) {return uint(v.x)|(uint(v.y)<<16u);}
vec2 words(uint v) {return vec2(v&65535u,v>>16u);}
double angle(dvec3 a,dvec3 b,dvec3 c) {
    return 2.0*double(atan(float(abs(dot(a,cross(b,c)))),float(1.0+dot(a,b)+dot(b,c)+dot(c,a))));
}
double solidAngle(uvec2 pixel) {
    double halfY=double(uProjection.x),halfX=halfY*double(uProjection.y);
    double l=halfX*(2.0*double(pixel.x)/double(uExtent.x)-1.0);
    double r=halfX*(2.0*double(pixel.x+1u)/double(uExtent.x)-1.0);
    double t=halfY*(1.0-2.0*double(pixel.y)/double(uExtent.y));
    double b=halfY*(1.0-2.0*double(pixel.y+1u)/double(uExtent.y));
    dvec3 a=normalize(dvec3(l,b,1)),bb=normalize(dvec3(r,b,1));
    dvec3 c=normalize(dvec3(r,t,1)),d=normalize(dvec3(l,t,1));
    return angle(a,bb,c)+angle(a,c,d);
}
void main() {
    uint at=gl_GlobalInvocationID.x;if(at>=uint(uOrderCount))return;
    int dst=int(at)*12;
    for(int i=0;i<12;++i)receipts[dst+i]=vec4(0);
    int o=int(at)*4;float power=orders[o+1].w;if(power<=0.0)return;
    int locator=int(orders[o].x),event=locator*24;
    vec3 outgoing=orders[o+1].xyz,normal=incidents[event+20].xyz;
    receipts[dst]=vec4(float(locator),incidents[event].y,orders[o].y,3.0);
    receipts[dst+1]=vec4(-1,-1,0,0);receipts[dst+3].x=-1.0;
    vec3 ray=-outgoing,local=causticRotate(vec4(-uRotation.xyz,uRotation.w),ray);
    float backward=-local.z,halfY=uProjection.x,halfX=halfY*uProjection.y;
    if(backward<=0.0||local.x < -halfX*backward||local.x>=halfX*backward
        ||local.y>halfY*backward||local.y<=-halfY*backward) {
        receipts[dst].w=2.0;receipts[dst+1].w=1.0;return;
    }
    uvec2 pixel=uvec2((local.x/(backward*halfX)+1.0)*0.5*uExtent.x,
        (1.0-local.y/(backward*halfY))*0.5*uExtent.y);
    receipts[dst+1].xy=vec2(pixel);
    causticLastWork=0u;
    SceneClosestHit hit=sceneClosestHit(uCamera.xyz,ray,1e-5,uProjection.w,uOpaqueControl);
    receipts[dst+2]=vec4(words(causticLastWork),0,0);
    receipts[dst+3]=vec4(float(hit.status),hit.distance,float(hit.entering),float(causticLastWork&65535u));
    receipts[dst+4]=vec4(hit.position,float(causticLastWork>>16u));receipts[dst+5]=vec4(hit.outwardNormal,0);
    receipts[dst+6]=vec4(words(hit.objectId),words(hit.materialId));
    receipts[dst+7]=vec4(words(hit.instanceId),words(hit.triangleId));
    ivec2 rasterPixel=ivec2(pixel.x,uint(uExtent.y)-1u-pixel.y);
    vec4 owner=texelFetch(uRasterOwners,rasterPixel,0),identity=texelFetch(uRasterTriangle,rasterPixel,0);
    receipts[dst+8]=owner;receipts[dst+9]=vec4(identity.xy,identity.z,identity.w);
    if(hit.status!=1||!hit.entering||identity.w!=1.0||owner!=receipts[dst+6]
        ||identity.x!=float(hit.instanceId)||identity.y!=float(hit.triangleId))return;
    float depth=-causticRotate(vec4(-uRotation.xyz,uRotation.w),hit.position-uCamera.xyz).z;
    vec3 centreRay=causticCameraRay(vec2(pixel)+vec2(0.5),uRotation,uProjection,uExtent);
    float centreDenominator=dot(centreRay,hit.outwardNormal);
    float centreDistance=dot(hit.position-uCamera.xyz,hit.outwardNormal)/centreDenominator;
    float centreDepth=-causticRotate(vec4(-uRotation.xyz,uRotation.w),centreRay*centreDistance).z;
    if(depth<uProjection.z||depth>uProjection.w||centreDenominator>=-1e-6||!causticFinite(centreDepth)
        ||centreDepth<uProjection.z||centreDepth>uProjection.w||abs(centreDepth-identity.z)>max(1e-5,abs(centreDepth)*2e-4))return;
    bool visible=owner==incidents[event+13]&&identity.x==incidents[event+14].x&&identity.y==incidents[event+14].y;
    float denominator=dot(ray,normal),target=dot(incidents[event+12].xyz-uCamera.xyz,normal)/denominator;
    float hitBias=max(1e-5,2e-6*max(max(abs(hit.position.x),abs(hit.position.y)),abs(hit.position.z)));
    if(denominator>=-1e-6||!causticFinite(target)||target<=0.0||hit.distance>target+8.0*hitBias)return;
    double radiance=0.0;
    if(visible) {
        vec3 du=vec3(incidents[event+22].zw,incidents[event+23].x),dv=incidents[event+23].yzw;
        // The admitted barycentric cells have entryArea/|e1 cross e2|
        // equal to 1/(2*n*n). This is the same receiving Jacobian, without
        // a second triangle search or an additional visibility query.
        double n=double(uEntrySubdivisions);
        double area=length(cross(dvec3(du),dvec3(dv)))/(2.0*n*n);
        double cosine=dot(dvec3(outgoing),dvec3(normal)),omega=solidAngle(pixel);
        if(area<=0.0||cosine<=0.0||omega<=0.0)return;
        // Native order power already includes substrate and profile fractions.
        radiance=double(power)/area/cosine/omega;
        if(isnan(radiance)||isinf(radiance)||radiance<0.0)return;
    }
    receipts[dst].w=visible?0.0:1.0;receipts[dst+1].zw=vec2(float(radiance),1.0);
}
)GLSL";
}
inline constexpr const char* gratingCameraGeometryFragmentSource=R"GLSL(#version 430 core
layout(std430,binding=4) readonly buffer Triangles {vec4 triangles[];};
uniform int uTriangle;
in float linearDepth;
layout(location=1) out float depth;
layout(location=2) out vec4 normal;
layout(location=5) out uint materialId;
layout(location=6) out uint objectId;
uint id(vec2 v) {return uint(v.x)|(uint(v.y)<<16u);}
void main() {
    depth=linearDepth;normal=vec4(triangles[uTriangle*8+3].xyz,1);
    objectId=id(triangles[uTriangle*8+4].xy);materialId=id(triangles[uTriangle*8+4].zw);
}
)GLSL";
inline std::string gratingReceiverResolveSource()
{
    return std::string("#version 430 core\n")+nativeMathSource()+R"GLSL(
layout(std430,binding=2) readonly buffer SpectrumLow {uint low[];};
layout(std430,binding=6) readonly buffer SpectrumHigh {uint high[];};
uniform vec2 uExtent;
uniform vec4 uCie[8];
layout(location=0) out vec4 color;
void main() {
    uvec2 pixel=uvec2(gl_FragCoord.xy);
    uint first=((uint(uExtent.y)-1u-pixel.y)*uint(uExtent.x)+pixel.x)*4u;
    vec3 xyz=vec3(0);
    for(int band=0;band<8;++band)xyz+=causticBandXyz(uintBitsToFloat(band<4?low[first+uint(band)]:high[first+uint(band-4)]),uCie[band]);
    color=vec4(causticXyzToSceneLinear(xyz),1);
}
)GLSL";
}
inline constexpr const char* gratingCameraKeySource=R"GLSL(#version 430 core
layout(local_size_x=32) in;
layout(std430,binding=0) readonly buffer CameraReceipts {vec4 receipts[];};
layout(std430,binding=5) buffer Keys {uvec4 keys[];};
uniform int uOrderCount,uKeyCount;
uniform vec2 uExtent;
void main() {
    uint at=gl_GlobalInvocationID.x;if(at>=uint(uKeyCount))return;
    uvec4 key=uvec4(0xffffffffu);
    if(at<uint(uOrderCount)) {
        uint r=at*12u;vec4 header=receipts[r],target=receipts[r+1u];
        if(header.w==0.0&&target.w==1.0&&target.z>0.0)
            key=uvec4(uint(target.y)*uint(uExtent.x)+uint(target.x),uint(header.x),uint(header.z),at);
    }
    keys[at]=key;
}
)GLSL";
inline constexpr const char* gratingCameraReduceSource=R"GLSL(#version 430 core
layout(local_size_x=32) in;
layout(std430,binding=0) readonly buffer CameraReceipts {vec4 receipts[];};
layout(std430,binding=2) buffer SpectrumLow {uint low[];};
layout(std430,binding=6) buffer SpectrumHigh {uint high[];};
layout(std430,binding=5) readonly buffer Keys {uvec4 keys[];};
uniform int uKeyCount;
void main() {
    uint at=gl_GlobalInvocationID.x;if(at>=uint(uKeyCount))return;
    uint pixel=keys[at].x;
    if(pixel==0xffffffffu||(at>0u&&keys[at-1u].x==pixel))return;
    float sum[8];for(int band=0;band<8;++band)sum[band]=0.0;
    // Reuse the released sorted segment-leader reduction order. Each pixel
    // has one writer and each band follows native incident/signed-order keys.
    for(uint next=at;next<uint(uKeyCount)&&keys[next].x==pixel;++next) {
        uint r=keys[next].w*12u;
        sum[int(receipts[r].y)]+=receipts[r+1u].z;
    }
    for(int band=0;band<4;++band) {
        low[pixel*4u+uint(band)]=floatBitsToUint(sum[band]);
        high[pixel*4u+uint(band)]=floatBitsToUint(sum[band+4]);
    }
}
)GLSL";
inline std::string gratingCellTransportComputeSource()
{
    return std::string("#version 430 core\n")+querySource()+nativeMathSource()+R"GLSL(
layout(local_size_x=32) in;
layout(std430,binding=0) readonly buffer Incidents {vec4 incidents[];};
layout(std430,binding=1) readonly buffer SourceCells {vec4 cells[];};
layout(std430,binding=2) buffer CellTraces {vec4 traces[];};
layout(std430,binding=3) readonly buffer Work {vec4 workTable[];};
layout(std430,binding=4) readonly buffer Triangles {vec4 triangles[];};
layout(std430,binding=5) readonly buffer Locators {uint locators[];};
uniform int uRecordCount,uMaximumInterfaces,uRecordStride,uEntryCount,uTriangleCount;
uniform vec4 uSource,uIor[2];
uint id(vec2 v) {return uint(v.x)|(uint(v.y)<<16u);}
vec2 words(uint v) {return vec2(v&65535u,v>>16u);}
float rayBias(vec3 p) {return max(1e-5,2e-6*max(max(abs(p.x),abs(p.y)),abs(p.z)));}
void stepHit(int at,SceneClosestHit hit,uint work) {
    traces[at+2]=vec4(float(hit.status),hit.distance,float(hit.entering),float(work&65535u));
    traces[at+3]=vec4(hit.position,float(work>>16u));traces[at+4]=vec4(hit.outwardNormal,0);
    traces[at+5]=vec4(words(hit.objectId),words(hit.materialId));
    traces[at+6]=vec4(words(hit.instanceId),words(hit.triangleId));
}
bool sameOwner(int at,SceneClosestHit hit) {
    return traces[at+5]==vec4(words(hit.objectId),words(hit.materialId))
        &&traces[at+6]==vec4(words(hit.instanceId),words(hit.triangleId));
}
bool glassHit(SceneClosestHit hit) {
    for(int t=0;t<uTriangleCount;++t) {
        vec4 owners=triangles[t*8+4],indices=triangles[t*8+5];
        if(owners==vec4(words(hit.objectId),words(hit.materialId))
            &&indices.x==float(hit.instanceId)&&indices.y==float(hit.triangleId))return indices.z==1.0;
    }return false;
}
void main() {
    uint invocation=gl_GlobalInvocationID.x;if(invocation>=uint(uRecordCount))return;
    int base=int(invocation)*uRecordStride,event=int(locators[invocation])*24;
    for(int i=0;i<uRecordStride;++i)traces[base+i]=vec4(0);
    int work=int(incidents[event].x),band=int(incidents[event].y),interfaces=int(incidents[event].z);
    vec4 entry=workTable[work*4],entryOwners=workTable[work*4+2],entryIndices=workTable[work*4+3];
    int entryIndex=int(entryIndices.z),cell=entryIndex*7;
    if(entryIndex<0||entryIndex>=uEntryCount||interfaces<1||interfaces>uMaximumInterfaces
        ||cells[cell+6].x!=1.0||cells[cell+4]!=entryOwners||cells[cell+5].xy!=entryIndices.xy
        ||cells[cell+5].w!=float(entryIndex)||cells[cell+3].w!=entry.w)return;
    traces[base]=vec4(1,float(invocation),float(work),float(entryIndex));
    traces[base+1]=vec4(float(band),float(interfaces),2,0);
    uint totalWork=0u;bool supported=true;
    for(int path=0;path<4;++path) {
        vec3 start=path==0?entry.xyz:cells[cell+path-1].xyz;
        vec3 origin=start+uSource.xyz*(2.0*rayBias(start)),direction=-uSource.xyz;
        bool inside=false,pathSupported=true;
        for(int depth=0;depth<=interfaces;++depth) {
            int at=base+4+(path*(uMaximumInterfaces+1)+depth)*7;
            int token=depth<interfaces?int(incidents[event+2+depth/4][depth%4]):0;
            causticLastWork=0u;
            SceneClosestHit hit=sceneClosestHit(origin,direction,depth==0?0.0:1e-5,uSource.w,uOpaqueControl);
            uint workUnits=causticLastWork;totalWork+=workUnits;
            traces[at]=vec4(origin,float(depth));traces[at+1]=vec4(direction,float(token));stepHit(at,hit,workUnits);
            traces[base+3][path]=float(depth+1);
            if(hit.status<0) {pathSupported=false;break;}
            int center=base+4+depth*7;
            bool ownerMatches=path==0||sameOwner(center,hit);
            if(depth==interfaces) {
                bool endpoint=hit.status==1&&hit.entering&&!inside&&ownerMatches
                    &&vec4(words(hit.objectId),words(hit.materialId))==incidents[event+13]
                    &&float(hit.instanceId)==incidents[event+14].x&&float(hit.triangleId)==incidents[event+14].y;
                if(path==0)endpoint=endpoint&&length(hit.position-incidents[event+12].xyz)<=8.0*rayBias(hit.position)
                    &&length(origin-incidents[event+10].xyz)<=8.0*rayBias(origin)
                    &&length(direction-incidents[event+11].xyz)<=2e-4;
                pathSupported=pathSupported&&endpoint;break;
            }
            bool matches=hit.status==1&&glassHit(hit)&&ownerMatches&&inside!=hit.entering&&(token==1||token==65);
            if(depth==0)matches=matches&&hit.objectId==id(entryOwners.xy)&&hit.materialId==id(entryOwners.zw)
                &&hit.instanceId==uint(entryIndices.x)&&hit.triangleId==uint(entryIndices.y);
            if(!matches) {pathSupported=false;break;}
            vec3 normal=inside?-hit.outwardNormal:hit.outwardNormal,transmitted,reflected;bool tir;
            float ior=uIor[band/4][band%4];
            float reflection=causticFresnel(direction,normal,inside?ior:1.0,inside?1.0:ior,transmitted,reflected,tir);
            bool reflect=token==1;
            if(!causticFinite(reflection)||(reflect?reflection<=0.0:tir||reflection>=1.0)) {pathSupported=false;break;}
            float signedBias=(inside?-1.0:1.0)*(reflect?1.0:-1.0)*rayBias(hit.position);
            traces[at+4].w=signedBias;
            // Match the original producer's oriented-normal operation order.
            origin=hit.position+normal*(reflect?rayBias(hit.position):-rayBias(hit.position));
            direction=reflect?reflected:transmitted;if(!reflect)inside=!inside;
        }
        supported=supported&&pathSupported;
    }
    traces[base+1].zw=vec2(supported?1.0:2.0,1.0);
    traces[base+2]=vec4(words(totalWork),0,0);
}
)GLSL";
}
inline bool decodeGratingCellCoverage(const Texel* rows,std::size_t rowCount,
    const GratingCellCoverageCandidate& candidate,std::uint32_t leaves,std::uint32_t level,
    GratingCellCoveragePayload& output,std::string& error)
{
    const auto refuse=[&](){error="OpenGL completed native interval proof is malformed";return false;};
    if(!candidate.boundary||!candidate.boundary->binding||!candidate.boundary->binding->sourceCells
        ||!candidate.boundary->binding->sourceCells->incidentBinding||!leaves||leaves>candidate.bounds.maximumLeaves
        ||level>kGratingCellCoverageDepth||leaves!=(1u<<(2*level)))return refuse();
    const auto& binding=*candidate.boundary->binding;
    const auto& frame=binding.sourceCells->incidentBinding->frame;
    const auto sweepStride=16+candidate.bounds.maximumBlockers*2;
    const auto leafStride=3+(binding.bounds.maximumInterfaces+1)*sweepStride;
    if(rowCount!=candidate.boundary->records.size()*leaves*leafStride||(rowCount&&!rows))return refuse();
    for(std::size_t i=0;i<rowCount;++i)for(float value:rows[i])if(!std::isfinite(value))return refuse();
    GratingCellCoveragePayload payload;payload.records.resize(candidate.boundary->records.size());
    bool all=!payload.records.empty();
    for(std::size_t incident=0;incident<payload.records.size();++incident) {
        auto& record=payload.records[incident];record.incident=std::uint32_t(incident);
        record.entry=candidate.boundary->records[incident].entry;record.leaves.reserve(leaves);bool recordClear=true;
        const auto interfaces=binding.pointReadback->incidents[incident].interfaces;
        for(std::uint32_t index=0;index<leaves;++index) {
            const auto* h=rows+(incident*leaves+index)*leafStride;
            if(h[0]!=Texel{float(kGratingCellCoverageVersion),float(incident),float(record.entry),float(index)}
                ||h[1][0]!=level||h[1][1]!=interfaces+1||h[1][3]!=1||(h[1][2]!=1&&h[1][2]!=2)
                ||h[2][2]!=0||h[2][3]!=0)return refuse();
            std::uint32_t lo=0,hi=0;
            if(!exactReceiptInteger(h[2][0],65535,lo)||!exactReceiptInteger(h[2][1],65535,hi))return refuse();
            const auto leafWork=lo|(std::uint64_t(hi)<<16);std::uint64_t work=0;bool leafClear=true;
            GratingCellCoverageLeaf leaf;
            if(!gratingCellDyadicTriangle(level,index,leaf.domain))return refuse();
            leaf.sweeps.reserve(interfaces+1);
            for(std::size_t depth=0;depth<=binding.bounds.maximumInterfaces;++depth) {
                const auto* t=h+3+depth*sweepStride;
                if(depth>interfaces) {for(std::size_t i=0;i<sweepStride;++i){if(t[i]!=Texel{})return refuse();}continue;}
                GratingCellSweepEvidence sweep;std::uint32_t token=0,rule=0,disposition=0,blockers=0;
                if(t[0][0]!=depth||!exactReceiptInteger(t[0][1],65,token)||!exactReceiptInteger(t[0][2],1,rule)||rule!=1
                    ||!exactReceiptInteger(t[0][3],2,disposition)||!disposition||(t[3][3]!=0&&t[3][3]!=1)
                    ||(t[4][3]!=-1&&t[4][3]!=0&&t[4][3]!=1)||t[8][3]!=0||t[12][1]!=1||t[12][3]!=0
                    ||!exactReceiptInteger(t[12][0],std::uint32_t(candidate.bounds.maximumBlockers),blockers)
                    ||t[13]!=Texel{}||t[14]!=Texel{}||t[15]!=Texel{})return refuse();
                sweep.depth=std::uint32_t(depth);sweep.token=token;sweep.rule=rule;
                sweep.disposition=GratingCellCoverageDisposition(disposition);sweep.entering=t[3][3]==1;sweep.biasSign=int(t[4][3]);
                if(!causticId(t[1][0],t[1][1],sweep.object)||!causticId(t[1][2],t[1][3],sweep.material)
                    ||!causticId(t[2][0],t[2][1],sweep.instance)||!causticId(t[2][2],t[2][3],sweep.triangle)
                    ||!exactReceiptInteger(t[6][3],65535,lo)||!exactReceiptInteger(t[7][3],65535,hi))return refuse();
                sweep.queryWork=lo|(std::uint64_t(hi)<<16);work+=sweep.queryWork;
                sweep.direction={t[3][0],t[3][1],t[3][2]};sweep.normal={t[4][0],t[4][1],t[4][2]};
                for(unsigned a=0;a<3;++a) {sweep.originBounds[a]={t[5][a],t[6][a]};sweep.hitBounds[a]={t[7][a],t[8][a]};}
                sweep.roundingExpansion=t[5][3];sweep.distance={t[9][0],t[9][1]};sweep.biasMagnitude={t[9][2],t[9][3]};
                sweep.signedBias={t[10][0],t[10][1]};sweep.barycentricBounds={{{t[10][2],t[10][3]},{t[11][0],t[11][1]},{t[11][2],t[11][3]}}};
                if(disposition==1) {
                    std::uint32_t target=0;
                    if(!exactReceiptInteger(t[12][2],std::uint32_t(frame.triangles.size()),target)||target>=frame.triangles.size()
                        ||blockers!=candidate.bounds.maximumBlockers)return refuse();
                    sweep.blockers.reserve(blockers);
                    for(std::uint32_t b=0;b<blockers;++b) {
                        const auto* p=t+16+b*2;GratingCellBlockerSeparation separation;
                        if(!exactReceiptInteger(p[0][0],std::uint32_t(frame.triangles.size()),separation.triangle)
                            ||separation.triangle>=frame.triangles.size()||separation.triangle==target)return refuse();
                        separation.axis={p[0][1],p[0][2],p[0][3]};
                        separation.sweptProjection={p[1][0],p[1][1]};separation.triangleProjection={p[1][2],p[1][3]};
                        sweep.blockers.push_back(separation);
                    }
                } else {
                    leafClear=false;
                    if(blockers||t[12][2]!=-1)return refuse();
                    for(std::size_t b=16;b<sweepStride;++b)if(t[b]!=Texel{})return refuse();
                }
                leaf.sweeps.push_back(std::move(sweep));
            }
            if(work!=leafWork||(h[1][2]==1)!=leafClear)return refuse();
            payload.queryWork+=work;recordClear=recordClear&&leafClear;record.leaves.push_back(std::move(leaf));
        }
        record.disposition=recordClear?GratingCellCoverageDisposition::InteriorCertified:GratingCellCoverageDisposition::Unsupported;
        all=all&&recordClear;
    }
    payload.disposition=all?GratingCellCoverageDisposition::InteriorCertified:GratingCellCoverageDisposition::Unsupported;
    if(!validateGratingCellCoverageShape(candidate,payload,error))return false;
    output=std::move(payload);error.clear();return true;
}

// Each leaf has a 48-byte header, then fixed 256-byte sweep headers and
// 32-byte all-triangle separation rows. These are private proof records.
inline std::string gratingCellCoverageComputeSource()
{
    return std::string("#version 430 core\n")+querySource()+nativeMathSource()+R"GLSL(
layout(local_size_x=32) in;
layout(std430,binding=0) readonly buffer Incidents {vec4 incidents[];};
layout(std430,binding=1) readonly buffer SourceCells {vec4 cells[];};
layout(std430,binding=2) readonly buffer Boundaries {vec4 boundaries[];};
layout(std430,binding=3) buffer Coverage {vec4 coverage[];};
layout(std430,binding=4) readonly buffer Work {vec4 workTable[];};
layout(std430,binding=5) readonly buffer Triangles {vec4 triangles[];};
layout(std430,binding=6) readonly buffer Locators {uint locators[];};
uniform int uRecordCount,uLeaves,uLevel,uLeafStride,uSweepStride,uBoundaryStride,uMaximumInterfaces,uTriangleCount,uQueryAllowance;
uniform vec4 uSource,uIor[2];
bool arithmetic=true;
vec2 words(uint n) {return vec2(n&65535u,n>>16u);}
bool normalFloat(float x) {return !isnan(x)&&!isinf(x)&&(x==0.0||abs(x)>=uintBitsToFloat(0x00800000u));}
float outward(float x,bool upper) {
    if(!normalFloat(x)) {arithmetic=false;return 0.0;}
    if(x==0.0)return upper?uintBitsToFloat(0x00800000u):-uintBitsToFloat(0x00800000u);
    uint bits=floatBitsToUint(x);bits+=((x>0.0)==upper)?1u:0xffffffffu;
    float y=uintBitsToFloat(bits);if(!normalFloat(y))arithmetic=false;return normalFloat(y)?y:0.0;
}
vec2 addI(vec2 a,vec2 b) {
    if(a==vec2(0))return b;if(b==vec2(0))return a;
    if(a.x==a.y&&b.x==b.y&&a.x==-b.x)return vec2(0);
    precise float lo=a.x+b.x;precise float hi=a.y+b.y;
    return vec2(outward(lo,false),outward(hi,true));
}
vec2 negI(vec2 a) {return -a.yx;}
vec2 subI(vec2 a,vec2 b) {return addI(a,negI(b));}
vec2 mulI(vec2 a,vec2 b) {
    if(a==vec2(0)||b==vec2(0))return vec2(0);
    if(a==vec2(1))return b;if(b==vec2(1))return a;
    if(a==vec2(-1))return negI(b);if(b==vec2(-1))return negI(a);
    precise float p0=a.x*b.x;precise float p1=a.x*b.y;precise float p2=a.y*b.x;precise float p3=a.y*b.y;
    if((p0==0.0&&a.x!=0.0&&b.x!=0.0)||(p1==0.0&&a.x!=0.0&&b.y!=0.0)
        ||(p2==0.0&&a.y!=0.0&&b.x!=0.0)||(p3==0.0&&a.y!=0.0&&b.y!=0.0))arithmetic=false;
    if(!normalFloat(p0)||!normalFloat(p1)||!normalFloat(p2)||!normalFloat(p3))arithmetic=false;
    return vec2(outward(min(min(p0,p1),min(p2,p3)),false),outward(max(max(p0,p1),max(p2,p3)),true));
}
vec2 divI(vec2 a,vec2 b) {
    // GLSL 4.30 section 4.7.1 bounds reciprocal error only for positive
    // normal denominators. Exact interval negation preserves a/b when both
    // operands change sign, including entering-plane negative denominators.
    if(b.x<=0.0&&b.y>=0.0) {arithmetic=false;return vec2(0);}
    if(b.y<0.0) {a=negI(a);b=negI(b);}
    // Six outward neighbors cover 2.5 ULP, also across a binade boundary.
    if(b.x<uintBitsToFloat(0x00800000u)||b.y>uintBitsToFloat(0x7e800000u))
        {arithmetic=false;return vec2(0);}
    precise float lo=1.0/b.y;precise float hi=1.0/b.x;
    for(int step=0;step<6;++step) {lo=outward(lo,false);hi=outward(hi,true);}
    return mulI(a,vec2(lo,hi));
}
struct Box {vec3 lo;vec3 hi;};
Box pointB(vec3 p) {for(int k=0;k<3;++k)if(!normalFloat(p[k]))arithmetic=false;return Box(p,p);}
Box addB(Box a,Box b) {
    Box c;for(int k=0;k<3;++k) {vec2 v=addI(vec2(a.lo[k],a.hi[k]),vec2(b.lo[k],b.hi[k]));c.lo[k]=v.x;c.hi[k]=v.y;}return c;
}
Box negB(Box a) {return Box(-a.hi,-a.lo);}
Box subB(Box a,Box b) {return addB(a,negB(b));}
Box scaleB(Box a,vec2 factor) {
    Box c;for(int k=0;k<3;++k) {vec2 v=mulI(vec2(a.lo[k],a.hi[k]),factor);c.lo[k]=v.x;c.hi[k]=v.y;}return c;
}
Box hullB(Box a,Box b) {return Box(min(a.lo,b.lo),max(a.hi,b.hi));}
vec2 dotB(Box a,Box b) {
    vec2 v=vec2(0);for(int k=0;k<3;++k)v=addI(v,mulI(vec2(a.lo[k],a.hi[k]),vec2(b.lo[k],b.hi[k])));return v;
}
Box crossB(Box a,Box b) {
    Box c;for(int k=0;k<3;++k) {int p=(k+1)%3,q=(k+2)%3;
        vec2 v=subI(mulI(vec2(a.lo[p],a.hi[p]),vec2(b.lo[q],b.hi[q])),mulI(vec2(a.lo[q],a.hi[q]),vec2(b.lo[p],b.hi[p])));
        c.lo[k]=v.x;c.hi[k]=v.y;}return c;
}
vec2 biasI(Box p) {
    float lower=0.0,upper=0.0;
    for(int k=0;k<3;++k) {
        float lo=p.lo[k]<=0.0&&p.hi[k]>=0.0?0.0:min(abs(p.lo[k]),abs(p.hi[k]));
        lower=max(lower,lo);upper=max(upper,max(abs(p.lo[k]),abs(p.hi[k])));
    }
    vec2 v=mulI(vec2(lower,upper),vec2(2e-6));return max(vec2(1e-5),v);
}
vec2 triangleProjection(int triangle,vec3 axis) {
    vec2 p=dotB(pointB(triangles[triangle*8].xyz),pointB(axis));
    for(int c=1;c<3;++c) {vec2 q=dotB(pointB(triangles[triangle*8+c].xyz),pointB(axis));p=vec2(min(p.x,q.x),max(p.y,q.y));}return p;
}
void domain(int level,uint code,out ivec2 a,out ivec2 b,out ivec2 c) {
    a=ivec2(0);b=ivec2(256,0);c=ivec2(0,256);
    for(int depth=0;depth<level;++depth) {
        ivec2 ab=(a+b)/2,bc=(b+c)/2,ca=(c+a)/2;
        uint child=(code>>uint(2*(level-depth-1)))&3u;
        if(child==0u) {b=ab;c=ca;}else if(child==1u) {a=ab;c=bc;}
        else if(child==2u) {a=ca;b=bc;}else {a=ab;b=bc;c=ca;}
    }
}
Box sourceVertex(int cell,ivec2 q) {
    return addB(addB(pointB(cells[cell].xyz),scaleB(subB(pointB(cells[cell+1].xyz),pointB(cells[cell].xyz)),vec2(float(q.x)/256.0))),
        scaleB(subB(pointB(cells[cell+2].xyz),pointB(cells[cell].xyz)),vec2(float(q.y)/256.0)));
}
void main() {
    uint invocation=gl_GlobalInvocationID.x;
    if(invocation>=uint(uRecordCount*uLeaves))return;
    int incident=int(invocation)/uLeaves,leaf=int(invocation)%uLeaves,base=int(invocation)*uLeafStride;
    for(int i=0;i<uLeafStride;++i)coverage[base+i]=vec4(0);
    int event=int(locators[incident])*24,boundary=incident*uBoundaryStride;
    int work=int(incidents[event].x),band=int(incidents[event].y),interfaces=int(incidents[event].z);
    int entry=int(workTable[work*4+3].z),cell=entry*7;
    if(boundaries[boundary]!=vec4(1,float(incident),float(work),float(entry))
        ||boundaries[boundary+1].xy!=vec2(float(band),float(interfaces))||boundaries[boundary+1].w!=1.0
        ||cells[cell+6].x!=1.0||cells[cell+5].w!=float(entry))return;
    coverage[base]=vec4(1,float(incident),float(entry),float(leaf));
    coverage[base+1]=vec4(float(uLevel),float(interfaces+1),2,0);
    ivec2 a,b,c;domain(uLevel,uint(leaf),a,b,c);
    arithmetic=true;
    Box start=hullB(hullB(sourceVertex(cell,a),sourceVertex(cell,b)),sourceVertex(cell,c));
    Box origin=addB(start,scaleB(pointB(uSource.xyz),mulI(biasI(start),vec2(2))));
    bool continuing=arithmetic&&boundaries[boundary+1].z==1.0,recordClear=continuing;
    uint totalWork=0u;
    for(int depth=0;depth<=interfaces;++depth) {
        int center=boundary+4+depth*7,at=base+3+depth*uSweepStride;
        vec4 owners=boundaries[center+5],indices=boundaries[center+6];
        vec3 direction=boundaries[center+1].xyz,normal=boundaries[center+4].xyz;
        int token=int(boundaries[center+1].w);float sign=depth<interfaces?(boundaries[center+4].w>0.0?1.0:-1.0):0.0;
        coverage[at]=vec4(float(depth),float(token),1,2);
        coverage[at+1]=owners;coverage[at+2]=indices;
        coverage[at+3]=vec4(direction,boundaries[center+2].z);coverage[at+4]=vec4(normal,sign);
        coverage[at+12]=vec4(0,1,-1,0);
        if(!continuing) {recordClear=false;continue;}
        arithmetic=true;int target=-1;uint workUnits=0u;
        for(int t=0;t<uTriangleCount;++t) {
            ++workUnits;vec4 ids=triangles[t*8+5];
            if(triangles[t*8+4]==owners&&vec4(words(uint(ids.x)),words(uint(ids.y)))==indices) {target=t;break;}
        }
        bool clear=target>=0&&workUnits+uint(uTriangleCount-1)<=uint(uQueryAllowance);
        Box hit=pointB(vec3(0));vec2 distance=vec2(0),bias=vec2(0),signedBias=vec2(0),bary[3];
        bary[0]=bary[1]=bary[2]=vec2(0);float expansion=0.0;
        if(clear) {
            Box p=pointB(triangles[target*8].xyz),e1=subB(pointB(triangles[target*8+1].xyz),p),e2=subB(pointB(triangles[target*8+2].xyz),p);
            Box n=crossB(e1,e2),ray=pointB(direction);vec2 denominator=dotB(ray,n);
            bool entering=coverage[at+3].w==1.0;
            clear=!(denominator.x<=0.0&&denominator.y>=0.0)
                &&(entering?denominator.y<0.0:denominator.x>0.0);
            // Enclose both the true retained plane and the released triangle
            // arithmetic, including its independent barycentric/distance path.
            vec2 plane=divI(dotB(subB(p,origin),n),denominator);
            Box t=subB(origin,p),crossRay=crossB(ray,e2),q=crossB(t,e1);vec2 determinant=dotB(e1,crossRay);
            vec2 nativeDistance=divI(dotB(e2,q),determinant);
            distance=vec2(min(plane.x,nativeDistance.x),max(plane.y,nativeDistance.y));
            hit=addB(origin,scaleB(ray,distance));
            bary[1]=divI(dotB(t,crossRay),determinant);bary[2]=divI(dotB(ray,q),determinant);
            bary[0]=subI(subI(vec2(1),bary[1]),bary[2]);
            clear=clear&&distance.x>=(depth==0?0.0:1e-5)&&distance.y<=uSource.w;
            for(int k=0;k<3;++k)clear=clear&&bary[k].x>0.0&&bary[k].y<1.0;
            if(depth<interfaces) {
                // Entering state fixes the outward-normal orientation.
                bool inside=!entering;vec3 oriented=inside?-normal:normal;
                vec3 transmitted,reflected;bool tir;float ior=uIor[band/4][band%4];
                float reflection=causticFresnel(direction,oriented,inside?ior:1.0,inside?1.0:ior,transmitted,reflected,tir);
                vec3 next=token==1?reflected:transmitted;
                clear=clear&&triangles[target*8+5].z==1.0&&(token==1||token==65)
                    &&causticFinite(reflection)&&(token==1?reflection>0.0:!tir&&reflection<1.0)
                    &&next==boundaries[center+8].xyz;
                bias=biasI(hit);signedBias=sign>0.0?bias:negI(bias);
            }
            Box swept=hullB(origin,hit);
            precise float magnitude=max(1.0,max(max(max(abs(swept.lo.x),abs(swept.hi.x)),max(abs(swept.lo.y),abs(swept.hi.y))),max(abs(swept.lo.z),abs(swept.hi.z))));
            precise float margin=magnitude*4.76837158203125e-7;
            expansion=outward(margin,true);
            int blocker=0;
            for(int triangle=0;triangle<uTriangleCount;++triangle) {
                if(triangle==target)continue;++workUnits;bool separated=false;
                vec3 axis=vec3(0);vec2 sweptProjection=vec2(0),projection=vec2(0);
                for(int choice=0;choice<4;++choice) {
                    vec3 trial=choice==3?triangles[triangle*8+3].xyz:vec3(choice==0?1.0:0.0,choice==1?1.0:0.0,choice==2?1.0:0.0);
                    if(abs(dot(trial,trial)-1.0)>1e-4)continue;
                    vec2 bundle=dotB(swept,pointB(trial)),surface=triangleProjection(triangle,trial);
                    precise float gap0=surface.x-bundle.y;precise float gap1=bundle.x-surface.y;
                    if(arithmetic&&max(outward(gap0,false),outward(gap1,false))>expansion) {
                        axis=trial;sweptProjection=bundle;projection=surface;separated=true;break;
                    }
                }
                int row=at+16+blocker*2;
                coverage[row]=vec4(float(triangle),axis);coverage[row+1]=vec4(sweptProjection,projection);
                ++blocker;clear=clear&&separated;
            }
            clear=clear&&arithmetic;
            if(clear) {
                coverage[at].w=1.0;
                coverage[at+5]=vec4(origin.lo,expansion);coverage[at+6]=vec4(origin.hi,float(workUnits&65535u));
                coverage[at+7]=vec4(hit.lo,float(workUnits>>16u));coverage[at+8]=vec4(hit.hi,0);
                coverage[at+9]=vec4(distance,bias);coverage[at+10]=vec4(signedBias,bary[0]);coverage[at+11]=vec4(bary[1],bary[2]);
                coverage[at+12]=vec4(float(uTriangleCount-1),1,float(target),0);
                origin=addB(hit,scaleB(pointB(normal),signedBias));continuing=arithmetic;
            }
        }
        if(!clear) {
            for(int k=16;k<uSweepStride;++k)coverage[at+k]=vec4(0);
            coverage[at+6].w=float(workUnits&65535u);coverage[at+7].w=float(workUnits>>16u);
            continuing=false;recordClear=false;
        }
        totalWork+=workUnits;
    }
    coverage[base+1].zw=vec2(recordClear?1.0:2.0,1.0);coverage[base+2]=vec4(words(totalWork),0,0);
}
)GLSL";
}

// Each native pair retains its full constructor and the fixed-corner records.
// Unused corner slots are zero. Wide coefficients are constructed natively.
static_assert(cameraraycoefficients::kRequestWords==57&&cameraraycoefficients::kResultWords==557
    &&camerapixelsigns::kRequestWords==65&&camerapixelsigns::kResultWords==73);
inline constexpr std::uint32_t kCameraDiagonalPairWords=557+13*73;
inline constexpr std::uint32_t kCameraDiagonalHeaderWords=16;
inline constexpr std::uint32_t kCameraDiagonalTriangleWords=20;
using CameraDiagonalTriangle=std::array<std::uint32_t,kCameraDiagonalTriangleWords>;
using CameraDiagonalPixel=std::array<std::uint32_t,8>;
inline std::string cameraDiagonalComputeSource()
{
    return std::string("#version 430 core\nlayout(local_size_x=1) in;\n")
        +camerapixelsigns::languageSource(false)+R"DIAGONAL(
layout(std430,binding=0) readonly buffer Inputs {uint inputs[];};
layout(std430,binding=1) buffer Certificates {uint certificates[];};
uniform uint uPixelCount,uTriangleCount,uGeneration;
const uint pairWords=1506u;
uint tri(uint t,uint w){return inputs[23u+8u*uPixelCount+t*20u+w];}
uint pix(uint p,uint w){return inputs[23u+p*8u+w];}
bool endpoint(uint t,uint a,uint s,uint b){
    if(tri(t,5u+a)!=tri(s,5u+b))return false;
    for(uint k=0u;k<3u;k++)if(tri(t,8u+a*3u+k)!=tri(s,8u+b*3u+k))return false;
    return true;
}
void request(uint p,uint a,uint b,out uint q[57]){
    for(uint k=0u;k<23u;k++)q[k]=inputs[k];q[8]=pix(p,0u);q[9]=pix(p,1u);
    for(uint k=0u;k<17u;k++){q[23u+k]=tri(a,k);q[40u+k]=tri(b,k);}
}
bool uniformSign(uint r[73],uint s){
    return r[0]==1u&&r[4]==s&&r[5]==s&&r[6]==s&&r[7]==s;
}
bool corners(uint q[57],uint record[557],uint form,uint base,out uint r[73]){
    uint a[65];for(int k=0;k<57;k++)a[k]=q[k];
    a[57]=1u;a[58]=CP_REQUEST_TAG;a[59]=0u;a[60]=CP_CHARGE;
    a[61]=form;a[62]=CP_WIRE;a[63]=0u;a[64]=0u;
    cpEvaluate(record,a,r);
    for(uint k=0u;k<73u;k++)certificates[base+557u+(form-4u)*73u+k]=r[k];
    return r[0]==1u;
}
bool reversedForm(uint record[557],uint a,uint b){
    uint x=61u+a*28u,y=61u+b*28u;
    if(record[x]!=record[y])return false;
    for(uint k=0u;k<3u;k++){
        uint sx=record[x+1u+k*9u],sy=record[y+1u+k*9u];
        if(sy!=(sx==0u?0u:3u-sx))return false;
        for(uint l=1u;l<9u;l++)if(record[x+1u+k*9u+l]!=record[y+1u+k*9u+l])return false;
    }
    return true;
}
void main(){
    uint p=gl_GlobalInvocationID.x;if(p>=uPixelCount)return;
    uint first=p*(16u+uTriangleCount*pairWords);
    for(uint k=0u;k<16u+uTriangleCount*pairWords;k++)certificates[first+k]=0u;
    for(uint k=0u;k<8u;k++)certificates[first+8u+k]=pix(p,k);
    uint a=pix(p,6u);if(pix(p,7u)!=uGeneration||a>=uTriangleCount||tri(a,18u)!=1u||tri(a,19u)!=0u)return;
    uint b=uTriangleCount,edgeA=0u,edgeB=0u;bool ambiguous=false;
    for(uint t=0u;t<uTriangleCount;t++){
        if(t==a||tri(t,1u)!=pix(p,2u)||tri(t,2u)!=pix(p,3u)||tri(t,3u)!=pix(p,4u)
            ||tri(t,18u)!=1u||tri(t,19u)!=0u)continue;
        for(uint ea=0u;ea<3u;ea++)for(uint eb=0u;eb<3u;eb++)
            if(endpoint(a,ea,t,(eb+1u)%3u)&&endpoint(a,(ea+1u)%3u,t,eb)){
                if(b!=uTriangleCount)ambiguous=true;
                else{b=t;edgeA=ea;edgeB=eb;}
            }
    }
    if(b==uTriangleCount||ambiguous||tri(a,17u)!=tri(b,17u)||tri(a,4u)==tri(b,4u))return;
    uint q[57],record[557],r[73],s[73];request(p,a,b,q);ccEvaluate(q,record);
    uint base=first+16u+b*pairWords;
    for(uint k=0u;k<557u;k++)certificates[base+k]=record[k];
    if(record[0]!=1u)return;
    uint horizon=0u;
    for(uint form=4u;form<=16u;form++)if(!corners(q,record,form,base,r))return;
    for(uint form=4u;form<=16u;form++){
        for(uint k=0u;k<73u;k++)r[k]=certificates[base+557u+(form-4u)*73u+k];
        if(form==7u){horizon=r[4];if(horizon==0u||!uniformSign(r,horizon))return;}
    }
    if(record[538]!=horizon||record[548]!=horizon)return;
    for(uint form=4u;form<16u;form++){
        for(uint k=0u;k<73u;k++)r[k]=certificates[base+557u+(form-4u)*73u+k];
        if(form==4u+edgeA||form==10u+edgeB)continue;
        if(!uniformSign(r,horizon))return;
    }
    for(uint k=0u;k<73u;k++)r[k]=certificates[base+557u+12u*73u+k];
    if(!uniformSign(r,0u)||!reversedForm(record,4u+edgeA,10u+edgeB))return;
    for(uint k=0u;k<73u;k++){
        r[k]=certificates[base+557u+edgeA*73u+k];
        s[k]=certificates[base+557u+(6u+edgeB)*73u+k];
    }
    uint diagonal=2u;
    if(r[4]==0u&&r[7]==0u&&r[5]!=0u&&r[6]==3u-r[5])diagonal=0u;
    if(r[5]==0u&&r[6]==0u&&r[4]!=0u&&r[7]==3u-r[4])diagonal=1u;
    if(diagonal==2u)return;
    for(uint k=4u;k<8u;k++)if(s[k]!=(r[k]==0u?0u:3u-r[k]))return;
    uint side=diagonal==0u?r[5]:r[4];
    uint primary[557];for(uint k=0u;k<557u;k++)primary[k]=record[k];
    for(uint t=0u;t<uTriangleCount;t++){
        if(t==a||t==b)continue;
        request(p,a,t,q);ccEvaluate(q,record);base=first+16u+t*pairWords;
        for(uint k=0u;k<557u;k++)certificates[base+k]=record[k];
        if(record[0]!=1u)return;
        for(uint k=0u;k<10u;k++)if(record[537u+k]!=primary[537u+k])return;
        for(uint k=0u;k<6u*28u;k++)if(record[61u+4u*28u+k]!=primary[61u+4u*28u+k])return;
        bool excluded=false;uint strict[3];
        for(uint e=0u;e<3u;e++){
            if(!corners(q,record,10u+e,base,r))return;
            strict[e]=uniformSign(r,1u)?1u:uniformSign(r,2u)?2u:0u;
        }
        if(!corners(q,record,13u,base,r))return;
        uint other=uniformSign(r,1u)?1u:uniformSign(r,2u)?2u:0u;
        // Two strict opposite edges exclude a finite hit even on an exactly
        // known horizon crossing. No division or inferred horizon orientation.
        for(uint x=0u;x<3u;x++)for(uint y=x+1u;y<3u;y++)
            if(strict[x]!=0u&&strict[y]==3u-strict[x])excluded=true;
        if(other!=0u){
            if(record[548]==3u-other)excluded=true; // Entire plane is backward.
            for(uint e=0u;e<3u;e++)if(strict[e]==3u-other)excluded=true;
        }
        if(!corners(q,record,16u,base,r))return;
        bool behind=other!=0u&&record[548]==other&&uniformSign(r,other==horizon?2u:1u);
        if(!excluded&&!behind)return;
    }
    // The exact equal-depth seam belongs to the released lexicographic key.
    // Both triangles share object-table and instance; triangle id decides it.
    certificates[first]=1u;certificates[first+1u]=a;certificates[first+2u]=b;
    certificates[first+3u]=edgeA;certificates[first+4u]=edgeB;certificates[first+5u]=diagonal;
    certificates[first+6u]=side==horizon?1u:2u;
    certificates[first+7u]=tri(a,4u)<tri(b,4u)?a:b;
}
)DIAGONAL";
}

} // namespace glasscaustics::opengl
} // namespace arbitgpu
