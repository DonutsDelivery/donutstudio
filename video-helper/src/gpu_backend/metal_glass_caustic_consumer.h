#pragma once
#include "fixture_finite_grating_radiance_admission.h"
#include "fixture_finite_grating_rgb_admission.h"

#include "fixture_glass_caustics.h"
#include "fixture_glass_grating_incident.h"
#include "fixture_glass_grating_cell_coverage.h"
#include "fixture_glass_camera_cell_coverage.h"
#include "fixture_native_camera_pixel_signs.h"
#include <algorithm>
#include <cstddef>
#include <cstring>

namespace arbitgpu
{
inline constexpr glasscaustics::ConsumerContract metalGlassCausticConsumerContract() noexcept {
    return {NativeFixtureMaterialBackend::Metal,glasscaustics::kTableVersion,3,
        glasscaustics::kBranchPolicyVersion,true,true,true,glasscaustics::kScheduleVersion,true,true,true,glasscaustics::kDepositionVersion,glasscaustics::kRoughTransportVersion,glasscaustics::kPrivateMultipleScatterVersion};
}
// Private backend seam. The visual request dispatcher enables it separately
// after both native consumers have platform receipts.
bool prepareMetalGlassCaustic(const std::shared_ptr<const NativeFixtureSceneResources>& owner,
    HarmonicMIDI::grid::SceneLightId source,const glasscaustics::AdmittedBranchPolicy& policy,
    glasscaustics::Prepared& prepared,std::string& error);
NativeFixtureSceneSubmission renderMetalGlassCausticFrame(
    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
    const std::shared_ptr<const NativeFixtureSceneResources>& resources,
    std::uint32_t width, std::uint32_t height, NativeFixtureSceneRuntimeInputs runtime,
    const glasscaustics::Prepared& prepared,
    glasscaustics::SchedulingDescription scheduling = {},
    const glasscaustics::MultipleScatterPolicy* multipleScatterPolicy = nullptr);
bool readMetalGlassCausticReceipt(const std::shared_ptr<const NativeFixtureSceneFrame>& owner,
    glasscaustics::Frame& schedule, glasscaustics::NativeReadback& receipt,
    glasscaustics::EnergyLedger& energy, std::string& error);

struct MetalGlassCausticMetrics {
    std::uint64_t admittedBytes=0, retainedNativeBytes=0, peakNativeBytes=0;
    double traceMilliseconds=0, depositionMilliseconds=0, readbackMilliseconds=0, resolveMilliseconds=0, frameMilliseconds=0;
};
bool readMetalGlassCausticMetrics(const std::shared_ptr<const NativeFixtureSceneFrame>& owner,
    MetalGlassCausticMetrics& metrics,std::string& error);

// Private immutable mixed resources and completed query/raster staging only.
// Color and public depth handles remain absent until the paired activation gate.
NativeFixtureScenePreparation prepareMetalGlassGratingResources(
    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
    const std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>& mixedRoot,
    const smoothglass::DistinctGratingAdmission& admission,
    const std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>& originalCompiledLeaf);
bool stageMetalGlassGratingFrame(const std::shared_ptr<const NativeFixtureSceneResources>& resources,
    HarmonicMIDI::grid::SceneLightId source,const glasscaustics::AdmittedBranchPolicy& policy,
    std::uint32_t width,std::uint32_t height,NativeFixtureSceneRuntimeInputs runtime,double capturedTime,
    std::shared_ptr<const NativeFixtureSceneFrame>& frameOut,
    std::shared_ptr<const glasscaustics::GratingIncidentBinding>& bindingOut,std::string& error,
    glasscaustics::SchedulingDescription scheduling = {});

// Private incident prerequisite. Never publishes combined Color.
bool produceMetalGlassGratingIncidents(const std::shared_ptr<const NativeFixtureSceneFrame>& owner,
    const std::shared_ptr<const glasscaustics::GratingIncidentBinding>& binding,
    glasscaustics::NativeGratingReadback& output,std::string& error);

// One selected light's private endpoint contribution. No Color or base grant.
bool produceMetalDirectGratingIrradiance(const std::shared_ptr<const NativeFixtureSceneFrame>& owner,
    const std::shared_ptr<const glasscaustics::GratingIncidentBinding>& binding,
    const std::vector<glasscaustics::NativeDirectGratingEndpointInput>& endpoints,
    glasscaustics::NativeDirectGratingContributionReadback& output,std::string& error);
bool validateMetalDirectGratingIrradiance(const std::shared_ptr<const NativeFixtureSceneFrame>& owner,
    const glasscaustics::SubmissionIdentity& expected,double evaluationTime,
    const glasscaustics::NativeDirectGratingContributionReadback& output,std::string& error);
// One frozen D8 roster shares one actual staging frame and query owner.
bool stageMetalDirectGratingIrradianceCohort(const std::shared_ptr<const NativeFixtureSceneResources>& resources,
    const std::shared_ptr<const glasscaustics::DirectGratingIrradianceCohort>& cohort,
    NativeFixtureSceneRuntimeInputs runtime,std::shared_ptr<const NativeFixtureSceneFrame>& frame,std::string& error);
bool produceMetalDirectGratingIrradianceCohort(const std::shared_ptr<const NativeFixtureSceneFrame>& owner,
    const std::shared_ptr<const glasscaustics::DirectGratingIrradianceCohort>& cohort,
    glasscaustics::DirectGratingIrradianceCohortReadback& output,std::string& error);
bool validateMetalDirectGratingIrradianceCohort(const std::shared_ptr<const NativeFixtureSceneFrame>& owner,
    const glasscaustics::SubmissionIdentity& expected,double time,
    const std::vector<glasscaustics::NativeDirectGratingContributionReadback>& readbacks,
    glasscaustics::DirectGratingIrradianceCohortReadback& output,std::string& error);
// Private per-area terminal scattering under the actual frame/query lease.
// No camera radiance, finite receiving footprint or public attachment grant.
bool produceMetalDirectGratingTerminalOrders(const std::shared_ptr<const NativeFixtureSceneFrame>& owner,
    const std::shared_ptr<const glasscaustics::DirectGratingIrradianceCohort>& cohort,
    const glasscaustics::SubmissionIdentity& expected,double time,
    const std::vector<glasscaustics::NativeDirectGratingContributionReadback>& sources,
    glasscaustics::DirectGratingTerminalOrderLedger& output,std::string& error);
// Separate exact-order query/raster attribution. No finite-cell or Color grant.
bool produceMetalDirectTerminalReceivers(const std::shared_ptr<const NativeFixtureSceneFrame>& owner,
    const std::shared_ptr<const glasscaustics::NativeDirectGratingTerminalResult>& terminal,
    const glasscaustics::SubmissionIdentity& expected,double time,
    glasscaustics::DirectGratingTerminalReceiverReadback& output,std::string& error);
bool inspectMetalDirectTerminalReceivers(const std::shared_ptr<const NativeFixtureSceneFrame>& owner,
    const glasscaustics::DirectGratingTerminalReceiverReadback& output,std::string& error);
// Private sampled camera branches, independent of incident power and Color.
bool produceMetalGlassCameraGratingTrace(const std::shared_ptr<const NativeFixtureSceneFrame>& owner,
    const std::shared_ptr<const glasscaustics::GratingIncidentBinding>& binding,
    const std::vector<glasscaustics::NativeGlassCameraGratingTraceInput>& inputs,
    glasscaustics::NativeGlassCameraGratingTraceReadback& output,std::string& error);
bool validateMetalGlassCameraGratingTrace(const std::shared_ptr<const NativeFixtureSceneFrame>& owner,
    const glasscaustics::SubmissionIdentity& expected,double evaluationTime,
    const glasscaustics::NativeGlassCameraGratingTraceReadback& output,std::string& error);

// Private completed child/deposition proof. Does not resolve or publish Color.
bool produceMetalGlassGratingTerminalReadback(const std::shared_ptr<const NativeFixtureSceneFrame>& owner,
    const std::shared_ptr<const glasscaustics::GratingIncidentBinding>& binding,
    glasscaustics::NativeReadback& output,glasscaustics::EnergyLedger& energy,std::string& error);

// Private native camera proof. Receipt v1 retains angular math and visibility,
// while finite receiving-cell coverage and complete base Color remain refused.
bool produceMetalGlassGratingCameraReadback(const std::shared_ptr<const NativeFixtureSceneFrame>& owner,
    const std::shared_ptr<const glasscaustics::GratingIncidentBinding>& binding,
    glasscaustics::NativeReadback& output,glasscaustics::EnergyLedger& energy,std::string& error);

bool inspectMetalGlassGratingCameraPrimitiveForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>& owner,
    const std::shared_ptr<const glasscaustics::NativeGratingCameraReadback>& expected,std::string& error);

// Read-only inspection of the production incumbent-storage calculation.
bool inspectMetalGlassGratingCellIncumbentBytesForFixture(
    const std::shared_ptr<const NativeFixtureSceneFrame>& owner,
    const std::shared_ptr<const glasscaustics::GratingEntryCellBinding>& sourceCells,
    const glasscaustics::NativeReadback& point,const glasscaustics::NativeGratingCellTransportReadback& cells,
    std::uint64_t& bytes,std::string& error);

// Separate completed center/corner boundary queries. Original point receipts
// remain intact and neither successful corners nor prefixes grant Color.
bool produceMetalGlassGratingCellTransportReadback(const std::shared_ptr<const NativeFixtureSceneFrame>& owner,
    const std::shared_ptr<const glasscaustics::GratingIncidentBinding>& binding,
    const std::shared_ptr<const glasscaustics::GratingEntryCellBinding>& sourceCells,
    glasscaustics::NativeReadback& pointOutput,glasscaustics::EnergyLedger& energy,
    glasscaustics::NativeGratingCellTransportReadback& output,std::string& error);

// Later native swept-domain evidence. Caller owns candidate and payload;
// frame evidence has weak lineage and grants no receiving footprint or Color.
bool produceMetalGlassGratingCellCoverageReadback(
    const std::shared_ptr<const NativeFixtureSceneFrame>& owner,
    const std::shared_ptr<const glasscaustics::NativeGratingCellTransportReadback>& boundary,
    std::size_t maximumLeaves,
    std::shared_ptr<const glasscaustics::GratingCellCoverageCandidate>& candidate,
    std::shared_ptr<const glasscaustics::GratingCellCoveragePayload>& payload,std::string& error);

// Private native whole-pixel diagonal proof. Guarded Color stays unchanged.
bool produceMetalCameraCellCoverageForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const std::shared_ptr<const glasscaustics::NativeDirectGratingTerminalReceiverResult>&,
    const glasscaustics::SubmissionIdentity&,double,glasscaustics::CameraCellCoverageReadback&,std::string&,int stop=-1,
    std::shared_ptr<const glasscaustics::NativeCameraCellCoverageResult>* escaped=nullptr);
bool inspectMetalCameraCellCoverageForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const glasscaustics::CameraCellCoverageReadback&,std::string&);
// Costs: PT, constructor requests, sign requests, native visits/word moves/
// scalar operations, host parser operations/word moves, ID scans, query-source
// association visits, copied bytes and transferred bytes. Storage: total bytes,
// adjunct bytes, geometric work, retained native bytes and submission attempts.
bool readMetalCameraCellCoverageStorageForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    std::array<std::uintptr_t,4>&,std::array<std::uint64_t,12>&,std::array<std::uint64_t,5>&,std::string&);
bool inspectMetalCameraCellCoverageResourceRefusalsForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const glasscaustics::CameraCellCoverageReadback&,std::string&);

// Separate geometric Euclidean-range evidence. It requires an already retained
// native camera trace on the same frame. Coverage alone remains usable without it.
class MetalCameraGeometricRangeResult;
bool produceMetalCameraGeometricRangeForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const std::shared_ptr<const glasscaustics::NativeCameraCellCoverageResult>&,
    std::shared_ptr<const MetalCameraGeometricRangeResult>&,std::string&,int stop=-1,
    std::shared_ptr<const MetalCameraGeometricRangeResult>* escaped=nullptr);
bool inspectMetalCameraGeometricRangeForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const std::shared_ptr<const MetalCameraGeometricRangeResult>&,std::string&);
// Separate units: evaluations, native visits/moves/scalars, host visits/moves/
// scalars, copied/transferred bytes, provenance words and compared bytes.
bool readMetalCameraGeometricRangeStorageForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    std::array<std::uintptr_t,4>&,std::array<std::uint64_t,11>&,std::array<std::uint64_t,5>&,std::string&);
bool inspectMetalCameraGeometricRangeRefusalsForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const std::shared_ptr<const MetalCameraGeometricRangeResult>&,std::string&);

// Finite directional grating support is separate from receiver geometry and
// radiometry. Densities retain the sealed CPU IR authority, not GPU evaluation.
class MetalFiniteGratingSourceResult;
bool produceMetalFiniteGratingSourceForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const std::shared_ptr<const glasscaustics::NativeDirectGratingTerminalResult>&,
    std::shared_ptr<const MetalFiniteGratingSourceResult>&,std::string&,int stop=-1,
    std::shared_ptr<const MetalFiniteGratingSourceResult>* escaped=nullptr);
bool inspectMetalFiniteGratingSourceForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const std::shared_ptr<const MetalFiniteGratingSourceResult>&,std::string&);
// Separate units: evaluations, strict face comparisons, native visits/moves/
// scalars, host visits/moves/scalars, parser operations, provenance words,
// registry/triangle ID scans, copied/transferred bytes and compared bytes.
bool readMetalFiniteGratingSourceStorageForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    std::array<std::uintptr_t,4>&,std::array<std::uint64_t,14>&,std::array<std::uint64_t,5>&,std::string&);
inline constexpr std::size_t kMetalFiniteSourceDiagnosticWords=15006;
bool readMetalFiniteGratingSourceWordsForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const std::shared_ptr<const MetalFiniteGratingSourceResult>&,
    std::array<std::uint32_t,kMetalFiniteSourceDiagnosticWords>&,std::string&);
bool inspectMetalFiniteGratingSourceRefusalsForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const std::shared_ptr<const MetalFiniteGratingSourceResult>&,std::string&);

// Powers use W/nm; irradiance and exitance use W/(m2*nm). Radiance retains
// the symbolic exitance/pi relation in W/(m2*sr*nm), without a dyadic pi.
class MetalFiniteGratingReceiverResult;
bool produceMetalFiniteGratingReceiverForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const std::shared_ptr<const MetalFiniteGratingSourceResult>&,
    const std::shared_ptr<const MetalCameraGeometricRangeResult>&,
    std::shared_ptr<const MetalFiniteGratingReceiverResult>&,std::string&,int stop=-1,
    std::shared_ptr<const MetalFiniteGratingReceiverResult>* escaped=nullptr);
bool inspectMetalFiniteGratingReceiverForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const std::shared_ptr<const MetalFiniteGratingReceiverResult>&,std::string&);
inline constexpr std::size_t kMetalFiniteReceiverDiagnosticWords=14240;
bool readMetalFiniteGratingReceiverWordsForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const std::shared_ptr<const MetalFiniteGratingReceiverResult>&,
    std::array<std::uint32_t,kMetalFiniteReceiverDiagnosticWords>&,std::string&);
bool readMetalFiniteGratingReceiverStorageForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    std::array<std::uintptr_t,4>&,std::array<std::uint64_t,14>&,std::array<std::uint64_t,5>&,std::string&);
bool inspectMetalFiniteGratingReceiverRefusalsForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const std::shared_ptr<const MetalFiniteGratingReceiverResult>&,std::string&);

// Private owning opaque SurfacePbr draw. No texture handles or public Color.
class MetalRestrictedCompleteBaseResult;
bool produceMetalRestrictedCompleteBaseForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const std::shared_ptr<const MetalFiniteGratingReceiverResult>&,
    std::shared_ptr<const MetalRestrictedCompleteBaseResult>&,std::string&,int stop=-1,
    std::shared_ptr<const MetalRestrictedCompleteBaseResult>* escaped=nullptr,
    std::uint64_t limit=renderpassoutput::kMaximumTotalBytes);
bool inspectMetalRestrictedCompleteBaseForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const std::shared_ptr<const MetalRestrictedCompleteBaseResult>&,std::string&);
bool inspectMetalRestrictedCompleteBaseRefusalsForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const std::shared_ptr<const MetalRestrictedCompleteBaseResult>&,std::string&);
bool readMetalRestrictedCompleteBaseStorageForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    std::array<std::uint64_t,5>&,std::array<std::uint64_t,12>&,std::string&);

// Private metric finite-cell L=M/pi carrier. Encodings and all native
// certificates are data; this API grants no RGB, texture or public Color.
// stop 0/1 refuse before bands, 10..17 after a band, 2 after escape.
// stop 100+16*band+phase uses the existing arithmetic child refusal phase.
class MetalFiniteGratingRadianceResult;
bool produceMetalFiniteGratingRadianceForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const std::shared_ptr<const MetalFiniteGratingReceiverResult>&,const std::shared_ptr<const MetalRestrictedCompleteBaseResult>&,
    std::shared_ptr<const MetalFiniteGratingRadianceResult>&,std::string&,int stop=-1,
    std::shared_ptr<const MetalFiniteGratingRadianceResult>* escaped=nullptr,std::uint64_t limit=renderpassoutput::kMaximumTotalBytes);
bool inspectMetalFiniteGratingRadianceForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const std::shared_ptr<const MetalFiniteGratingRadianceResult>&,std::string&);
bool readMetalFiniteGratingRadianceForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const std::shared_ptr<const MetalFiniteGratingRadianceResult>&,std::array<std::uint32_t,8>&,
    std::array<std::array<std::uint32_t,250>,8>&,std::string&);
// Storage: factories, attempts, wrapper/control bytes, whole-parent bytes,
// live wrappers, actual arithmetic slots used, wrapper reservation, child reservation.
// Work units are finiteradiance::Unit and exclude separately journaled parents/children.
bool readMetalFiniteGratingRadianceStorageForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    std::array<std::uint64_t,8>&,finiteradiance::Costs&,std::string&);
bool inspectMetalFiniteGratingRadianceRefusalsForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const std::shared_ptr<const MetalFiniteGratingRadianceResult>&,std::string&);


// Private native spectral conversion. No Color attachment or public handle.
class MetalFiniteGratingRgbResult;
bool produceMetalFiniteGratingRgb(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const std::shared_ptr<const MetalFiniteGratingRadianceResult>&,
    std::shared_ptr<const MetalFiniteGratingRgbResult>&,std::string&,
    std::uint64_t limit=renderpassoutput::kMaximumTotalBytes);
bool inspectMetalFiniteGratingRgb(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const std::shared_ptr<const MetalFiniteGratingRgbResult>&,std::string&);
// Fault selectors are private fixture phases, never production inputs.
bool produceMetalFiniteGratingRgbForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const std::shared_ptr<const MetalFiniteGratingRadianceResult>&,
    std::shared_ptr<const MetalFiniteGratingRgbResult>&,std::string&,int stop=-1,
    std::shared_ptr<const MetalFiniteGratingRgbResult>* escaped=nullptr,std::uint64_t limit=renderpassoutput::kMaximumTotalBytes);
bool readMetalFiniteGratingRgbForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const std::shared_ptr<const MetalFiniteGratingRgbResult>&,std::array<float,4>&,std::string&);
// factories, attempts, ledger/live bytes, whole parent, live owners,
// native bytes, reservation, prospective absent-ledger ceiling, dispatches.
bool readMetalFiniteGratingRgbStorageForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    std::array<std::uint64_t,9>&,finitergb::Costs&,std::string&);
bool metalFiniteGratingRgbNativeRefusalProbesForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const std::shared_ptr<const MetalFiniteGratingRadianceResult>&,std::string&);
bool inspectMetalFiniteGratingRgbRefusalsForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const std::shared_ptr<const MetalFiniteGratingRgbResult>&,std::string&);


namespace metalcaustic
{
inline constexpr std::uint32_t kCameraRangeRequestWords=575,kCameraRangeResultWords=893;
inline constexpr std::uint32_t kCameraRangeWireTag=0x43524731u;
using CameraRangeRequest=std::array<std::uint32_t,kCameraRangeRequestWords>;
using CameraRangeResult=std::array<std::uint32_t,kCameraRangeResultWords>;
// The range evaluator has 17 products, 43 sums/comparisons, 13 coefficient
// loads/normalizations, two binary32 decodes, 31 scalar stores and one 51-integer
// canonical parser. The existing constructor's 223/167 arithmetic schedule
// dominates every nested EL call, with 32768 additional wrapper words. These
// source bounds cover failed paths too; they are not device measurements.
inline constexpr std::uint64_t kCameraRangeVisits=cameraraycoefficients::maximumVisits(256);
inline constexpr std::uint64_t kCameraRangeWordMoves=cameraraycoefficients::maximumWordMoves(256)+32768;
inline constexpr std::uint64_t kCameraRangeScalarOperations=cameraraycoefficients::maximumScalarOperations(256)+8*32768;
inline constexpr std::uint64_t kCameraRangeScratchBytes=128*148+4*51*144+4*kCameraRangeResultWords*4+4*kCameraRangeRequestWords*4+4096;

// Host reconstruction uses fixed integer storage and refuses the same bounded
// precision/exponent domain as CCDyadic. No floating-point generated value,
// tolerance, square root or sampled visibility enters the certificate.
struct CameraRangeDyadic {
    std::array<std::uint32_t,8> limb{};int exponent=0;std::uint32_t sign=0;
};
inline bool cameraRangeNormalize(const std::array<std::uint32_t,16>& words,int exponent,std::uint32_t sign,CameraRangeDyadic& r) {
    int low=-1,high=-1;
    for(int k=0;k<16;++k)if(words[k]){if(low<0){low=k;}high=k;}
    r={};if(low<0)return true;
    int trailing=0,leading=0;while(((words[low]>>trailing)&1u)==0u)++trailing;
    for(int k=0;k<32;++k)if(words[high]>>k)leading=k+1;
    const int shift=low*32+trailing,width=high*32+leading-shift;
    exponent+=shift;if(width>256||exponent< -512||exponent>512||sign<1||sign>2)return false;
    r.exponent=exponent;r.sign=sign;
    for(int k=0;k<8;++k){const int at=k+shift/32,bits=shift%32;
        if(at<16){r.limb[k]=words[at]>>bits;if(bits&&at+1<16)r.limb[k]|=words[at+1]<<(32-bits);}}
    return true;
}
inline bool cameraRangeLoad(const std::uint32_t* words,int exponent,CameraRangeDyadic& r) {
    std::array<std::uint32_t,16> value{};for(unsigned k=0;k<8;++k)value[k]=words[1+k];
    return cameraRangeNormalize(value,exponent,words[0],r);
}
inline bool cameraRangeDecode(std::uint32_t word,CameraRangeDyadic& r) {
    const auto e=(word>>23)&255u;if(e==255)return false;
    std::array<std::uint32_t,16> words{};words[0]=(word&8388607u)|(e?8388608u:0u);
    return cameraRangeNormalize(words,e?int(e)-150:-149,(word>>31)?2u:1u,r);
}
inline bool cameraRangeAdd(const CameraRangeDyadic& a,const CameraRangeDyadic& b,CameraRangeDyadic& r,bool subtract=false) {
    const auto bs=subtract&&b.sign?3u-b.sign:b.sign;
    if(!a.sign){r=b;r.sign=bs;return true;}if(!bs){r=a;return true;}
    const int exponent=std::min(a.exponent,b.exponent);
    std::array<std::uint32_t,16> x{},y{},z{};
    const auto shift=[&](const CameraRangeDyadic& value,std::array<std::uint32_t,16>& out) {
        const int bits=value.exponent-exponent;if(bits<0||bits>256)return false;
        for(int k=0;k<8;++k){const int at=k+bits/32;if(at<16){out[at]|=value.limb[k]<<(bits%32);
            if(bits%32&&at+1<16)out[at+1]|=value.limb[k]>>(32-bits%32);}}
        for(int k=8;k<16;++k){if(out[k])return false;}return true;
    };
    if(!shift(a,x)||!shift(b,y))return false;
    auto sign=a.sign;
    if(a.sign==bs){std::uint64_t carry=0;for(int k=0;k<16;++k){const auto v=std::uint64_t(x[k])+y[k]+carry;z[k]=std::uint32_t(v);carry=v>>32;}if(carry)return false;}
    else {int compare=0;for(int k=15;k>=0&&!compare;--k)if(x[k]!=y[k])compare=x[k]>y[k]?1:-1;
        if(!compare){r={};return true;}if(compare<0){std::swap(x,y);sign=bs;}
        std::uint64_t borrow=0;for(int k=0;k<16;++k){const auto sub=std::uint64_t(y[k])+borrow;z[k]=std::uint32_t(std::uint64_t(x[k])-sub);borrow=std::uint64_t(x[k])<sub;}}
    return cameraRangeNormalize(z,exponent,sign,r);
}
inline bool cameraRangeMultiply(const CameraRangeDyadic& a,const CameraRangeDyadic& b,CameraRangeDyadic& r) {
    if(!a.sign||!b.sign){r={};return true;}
    std::array<std::uint32_t,16> words{};
    for(unsigned i=0;i<8;++i){std::uint64_t carry=0;for(unsigned j=0;j<8;++j){
        const auto v=std::uint64_t(a.limb[i])*b.limb[j]+words[i+j]+carry;words[i+j]=std::uint32_t(v);carry=v>>32;}
        words[i+8]=std::uint32_t(carry);}
    return cameraRangeNormalize(words,a.exponent+b.exponent,a.sign==b.sign?1u:2u,r);
}
inline constexpr std::uint32_t kFiniteSourceTag=0x46534731u,kFiniteSourceHeaderWords=96;
inline constexpr std::uint32_t kFiniteSourceBandWords=54,kFiniteSourceOrderWords=32,kFiniteSourceTriangleWords=20;
inline constexpr std::uint32_t kFiniteSourceMaximumTriangles=256,kFiniteSourceMaximumOrders=24;
inline constexpr std::uint64_t kFiniteSourceScratchBytes=128*148+4096;
// One constructor envelope covers the fixed <=33 products, <=6 sums, <=43
// decodes and <=150+10O scalar stores. Each triangle gets another envelope for
// nine decodes, all 18 strict comparisons and its 32-word witness. The bounded
// binary64 decoder adds one limb store to ccDecode's 204-visit schedule. These
// envelopes also cover complete attempted failure paths and wrapper copies.
inline constexpr std::uint32_t finiteSourceRequestWords(std::uint32_t t,std::uint32_t o) {
    return kFiniteSourceHeaderWords+8*kFiniteSourceBandWords+o*kFiniteSourceOrderWords+t*kFiniteSourceTriangleWords;
}
inline constexpr std::uint32_t finiteSourceResultWords(std::uint32_t t,std::uint32_t o) {
    return 8+finiteSourceRequestWords(t,o)+150+10*o+32*t;
}
static_assert(finiteSourceResultWords(kFiniteSourceMaximumTriangles,kFiniteSourceMaximumOrders)==kMetalFiniteSourceDiagnosticWords);
inline bool finiteSourceDecode64(std::uint32_t lo,std::uint32_t hi,CameraRangeDyadic& r) {
    const auto exponent=(hi>>20)&2047u;
    if(exponent==2047u||(hi>>31))return false;
    std::array<std::uint32_t,16> words{};words[0]=lo;words[1]=(hi&1048575u)|(exponent?1048576u:0u);
    return cameraRangeNormalize(words,exponent?int(exponent)-1075:-1074,1,r);
}
inline bool finiteSourceReconstruct(const std::vector<std::uint32_t>& q,const std::vector<std::uint32_t>& r) {
    if(q.size()<kFiniteSourceHeaderWords||q[0]!=kFiniteSourceTag||q[1]!=1||q[2]!=256
        ||!q[3]||q[3]>kFiniteSourceMaximumTriangles||q[4]>kFiniteSourceMaximumOrders||q[5]>=q[3]
        ||q.size()!=finiteSourceRequestWords(q[3],q[4])||r.size()!=finiteSourceResultWords(q[3],q[4])
        ||r[0]!=1||r[1]!=kFiniteSourceTag||r[2]!=1||r[3]!=256||!r[4]
        ||r[4]>(q[3]+1)*kCameraRangeVisits||r[5]!=q[3]||r[6]!=q[4]||r[7])return false;
    for(std::size_t k=0;k<q.size();++k)if(r[8+k]!=q[k])return false;
    const auto same=[&](std::size_t at,const CameraRangeDyadic& d) {
        if(r[at]!=std::uint32_t(d.exponent)||r[at+1]!=d.sign)return false;
        for(unsigned k=0;k<8;++k){if(r[at+2+k]!=d.limb[k])return false;}return true;
    };
    std::array<CameraRangeDyadic,6> box{};CameraRangeDyadic x,y,z,a,b,dx,dy,area;
    if(!cameraRangeDecode(q[36],x)||!cameraRangeDecode(q[37],y)||!cameraRangeDecode(q[38],z)
        ||!cameraRangeDecode(q[39],box[1])||!cameraRangeDecode(q[43],box[3])
        ||!cameraRangeDecode(q[31],a)||!cameraRangeDecode(q[35],b)||a.sign!=1||b.sign!=1
        ||!cameraRangeAdd(b,a,dx,true)||dx.sign!=1
        ||!cameraRangeAdd(box[1],x,dx,true)||dx.sign!=1||!cameraRangeAdd(box[3],y,dy,true)||dy.sign!=1
        ||!cameraRangeMultiply(dx,dy,area)||area.sign!=1||area.exponent<=-512)return false;
    --area.exponent;box[0]=x;box[2]=y;
    if(!cameraRangeAdd(z,a,box[4])||!cameraRangeAdd(z,b,box[5]))return false;
    std::size_t at=8+q.size();if(!same(at,area)){return false;}at+=10;
    for(const auto& d:box){if(!same(at,d)){return false;}at+=10;}
    for(unsigned band=0;band<8;++band){const auto base=kFiniteSourceHeaderWords+band*kFiniteSourceBandWords;
        CameraRangeDyadic density,power;if(!finiteSourceDecode64(q[base+38],q[base+39],density)
            ||!cameraRangeMultiply(area,density,power)||!same(at,power)){return false;}at+=10;}
    const auto orders=kFiniteSourceHeaderWords+8*kFiniteSourceBandWords;
    for(unsigned o=0;o<q[4];++o){const auto base=orders+o*kFiniteSourceOrderWords;
        CameraRangeDyadic density,power;if(!finiteSourceDecode64(q[base+6],q[base+7],density)
            ||!cameraRangeMultiply(area,density,power)||!same(at,power)){return false;}at+=10;}
    const auto triangles=orders+q[4]*kFiniteSourceOrderWords;
    for(unsigned t=0;t<q[3];++t){std::array<CameraRangeDyadic,9> vertices{};
        for(unsigned k=0;k<9;++k)if(!cameraRangeDecode(q[triangles+t*kFiniteSourceTriangleWords+8+k],vertices[k]))return false;
        unsigned chosen=0;std::array<CameraRangeDyadic,3> selected{};
        for(unsigned face=0;face<6;++face){bool strict=true;std::array<CameraRangeDyadic,3> distances{};
            for(unsigned v=0;v<3;++v){const auto& value=vertices[3*v+face/2];
                if(!cameraRangeAdd(face%2?value:box[face],face%2?box[face]:value,distances[v],true))return false;
                strict=strict&&distances[v].sign==1;}
            if(strict&&!chosen){chosen=face+1;selected=distances;}}
        if(!chosen||r[at]!=t||r[at+1]!=chosen){return false;}at+=2;
        for(const auto& d:selected){if(!same(at,d)){return false;}at+=10;}
    }
    return at==r.size();
}
inline std::string finiteGratingSourceComputeSource() {
    return std::string("#include <metal_stdlib>\nusing namespace metal;\n")+camerawideline::languageSource(true)
        +"\n#define CR_VISITS "+std::to_string(kCameraRangeVisits)+"u\n"
        +R"FINITE(
void fsDecode64(uint lo,uint hi,thread CCDyadic& r,thread ELWork& w){
    ccZero(r,w);uint e=(hi>>20u)&2047u;
    if(e==2047u||(hi>>31u)!=0u){w.ok=false;return;}
    r.exponent=e!=0u?int(e)-1075:-1074;r.m.sign=1u;
    r.m.limb[0]=lo;r.m.limb[1]=(hi&1048575u)|(e!=0u?1048576u:0u);ccNormalize(r,256u,w);
}
void fsStore(device uint* r,uint at,thread const CCDyadic& d,thread ELWork& w){
    elStep(w);r[at]=uint(d.exponent);elStep(w);r[at+1u]=d.m.sign;
    for(uint k=0u;k<8u;k++){elStep(w);r[at+2u+k]=d.m.limb[k];}
}
kernel void finiteGratingSource(device const uint* q [[buffer(0)]],device uint* r [[buffer(1)]],
    constant uint4& limits [[buffer(2)]],uint index [[thread_position_in_grid]]) {
    if(index!=0u)return;for(uint k=0u;k<limits.w;k++)r[k]=0u;
    for(uint k=0u;k<limits.z;k++)r[8u+k]=q[k];
    r[1]=0x46534731u;r[2]=1u;r[3]=256u;r[5]=limits.x;r[6]=limits.y;
    ELWork w;w.used=0u;w.limit=(limits.x+1u)*CR_VISITS;w.capacity=35u;w.ok=true;
    bool ok=q[0]==0x46534731u&&q[1]==1u&&q[2]==256u&&q[3]==limits.x&&q[4]==limits.y
        &&limits.x>0u&&limits.x<=256u&&limits.y<=24u&&q[5]<limits.x
        &&limits.z==96u+8u*54u+limits.y*32u+limits.x*20u&&limits.w==8u+limits.z+150u+limits.y*10u+limits.x*32u;
    CCDyadic box[6],x,y,z,a,b,dx,dy,area,density,power,d;
    ccDecode(q[36],true,x,256u,w);ccDecode(q[37],true,y,256u,w);ccDecode(q[38],true,z,256u,w);
    ccDecode(q[39],true,box[1],256u,w);ccDecode(q[43],true,box[3],256u,w);
    ccDecode(q[31],true,a,256u,w);ccDecode(q[35],true,b,256u,w);ccSubtract(b,a,d,256u,w);
    ok=ok&&a.m.sign==1u&&b.m.sign==1u&&d.m.sign==1u;
    ccSubtract(box[1],x,dx,256u,w);ccSubtract(box[3],y,dy,256u,w);ccMultiply(dx,dy,area,256u,w);
    ok=ok&&dx.m.sign==1u&&dy.m.sign==1u&&area.m.sign==1u&&area.exponent> -512;
    area.exponent--;box[0]=x;box[2]=y;ccAdd(z,a,box[4],256u,w);ccAdd(z,b,box[5],256u,w);
    uint at=8u+limits.z;fsStore(r,at,area,w);at+=10u;
    for(uint k=0u;k<6u;k++){fsStore(r,at,box[k],w);at+=10u;}
    for(uint band=0u;band<8u;band++){uint base=96u+54u*band;fsDecode64(q[base+38u],q[base+39u],density,w);
        ccMultiply(area,density,power,256u,w);fsStore(r,at,power,w);at+=10u;}
    uint orders=96u+8u*54u;
    for(uint o=0u;o<limits.y;o++){uint base=orders+32u*o;fsDecode64(q[base+6u],q[base+7u],density,w);
        ccMultiply(area,density,power,256u,w);fsStore(r,at,power,w);at+=10u;}
    uint triangles=orders+limits.y*32u;
    for(uint t=0u;t<limits.x;t++){CCDyadic vertices[9],distances[3],selected[3];uint chosen=0u;
        for(uint k=0u;k<9u;k++)ccDecode(q[triangles+t*20u+8u+k],true,vertices[k],256u,w);
        for(uint face=0u;face<6u;face++){bool strict=true;
            for(uint v=0u;v<3u;v++){if((face&1u)!=0u)ccSubtract(vertices[3u*v+face/2u],box[face],distances[v],256u,w);
                else ccSubtract(box[face],vertices[3u*v+face/2u],distances[v],256u,w);
                strict=strict&&distances[v].m.sign==1u;}
            if(strict&&chosen==0u){chosen=face+1u;for(uint v=0u;v<3u;v++)selected[v]=distances[v];}}
        r[at]=t;r[at+1u]=chosen;at+=2u;if(chosen==0u)ok=false;
        if(chosen!=0u)for(uint v=0u;v<3u;v++){fsStore(r,at,selected[v],w);at+=10u;}
        else at+=30u;
    }
    r[4]=w.used;if(ok&&w.ok&&at==limits.w)r[0]=1u;
}
)FINITE";
}

// Request: authenticated lineage, nine source coordinates, rectangle5,
// uploaded camera13, shadow/minimum/bias, eight complete emitted inputs,
// binary64 Eo8/rho8, signed CPU direction records and every raw triangle.
// Result: request echo, AS/AC/(AS-AC)/t, then seven values per band in order
// PhiOrder/PhiCell/PhiRemaining/PhiOut/PhiAbsorbed/ER/M; all are ten-word
// dyadics. PI1 is the symbolic L=M/pi relation, never a dyadic pi value.
// The final32T words are index/strict separating face/three vertex margins.
inline constexpr std::uint32_t kFiniteReceiverTag=0x46524731u,kFiniteReceiverHeaderWords=320;
inline constexpr std::uint32_t finiteReceiverRequestWords(std::uint32_t t){return 320+20*t;}
inline constexpr std::uint32_t finiteReceiverResultWords(std::uint32_t t){return 8+finiteReceiverRequestWords(t)+600+32*t;}
static_assert(finiteReceiverResultWords(256)==kMetalFiniteReceiverDiagnosticWords);
// Complete call census: metric2 + endpoint2 + corner8 + band48 products;
// initial7 + endpoint6 + corners32 + rectangle12 + intervals16 + rho8 sums;
// source9 + rectangle5 + camera6 + shadow1 + endpoint2 + intervals16 +
// densities16 decodes. Each competitor adds6 products,24 sums,9 decodes;
// nominated pair branches are smaller. Stores visit600+30T words; index/face
// writes add2T moves. Selected zeros visit35*(3T+1) limbs. Thus80/160/64
// fixed calls and600+32T stores conservatively cover the complete kernel,
// including arithmetic refusals, independently of earlier kernel envelopes.
inline constexpr std::uint64_t finiteReceiverVisits(std::uint32_t t) {
    return (80ull+6*t)*cameraraycoefficients::multiplyVisits(256)
        +(160ull+24*t)*cameraraycoefficients::kAddVisits+(64ull+9*t)*205+600+32*t+105*t+35;
}
inline constexpr std::uint64_t finiteReceiverWordMoves(std::uint32_t t) {
    // A product has 64 limb products/accumulates, <=3 capacity checks, two
    // zeros and one normalization. A sum has two shifts, one EL sum and one
    // normalization. 100000 words per CC call covers those bodies, value/ref
    // copies, failure zeros and nested EL calls, plus eight work-counter moves
    // per visit. 32 wire passes cover clear, echo, stores and wrapper copies.
    return (80ull+6*t+160ull+24*t+64ull+9*t)*100000+8*finiteReceiverVisits(t)
        +32ull*(finiteReceiverRequestWords(t)+finiteReceiverResultWords(t))+32768;
}
inline constexpr std::uint64_t finiteReceiverScalars(std::uint32_t t) {
    return 32768+1280ull*(304+39*t)+64*finiteReceiverVisits(t)+8*finiteReceiverWordMoves(t);
}
static_assert(finiteReceiverVisits(256)<=0xffffffffull,"receiver native visit counter bound");
inline constexpr std::uint64_t kFiniteReceiverScratchBytes=192*148+8192;
inline bool finiteReceiverReconstruct(const std::vector<std::uint32_t>& q,const std::vector<std::uint32_t>& r) {
    if(q.size()<320||q[0]!=kFiniteReceiverTag||q[1]!=1||q[2]!=256||!q[3]||q[3]>256
        ||q[4]>=q[3]||q[5]>=q[3]||q[6]>=q[3]||q[5]==q[6]
        ||q.size()!=finiteReceiverRequestWords(q[3])||r.size()!=finiteReceiverResultWords(q[3])
        ||r[0]!=1||r[1]!=kFiniteReceiverTag||r[2]!=1||r[3]!=256||!r[4]||r[4]>finiteReceiverVisits(q[3])
        ||r[5]!=q[3]||r[6]!=8||r[7]!=0x504931u)return false;
    for(std::size_t k=0;k<q.size();++k)if(r[8+k]!=q[k])return false;
    const auto zero=[](std::uint32_t w){return !(w&0x7fffffffu);};
    if(!zero(q[18])||!zero(q[21])||!zero(q[24])||!zero(q[33])||q[34]!=0x3f800000u
        ||!zero(q[35])||!zero(q[36])||q[38]!=0x3f800000u||q[41]!=0x3f800000u||q[42]!=0x3f800000u
        ||q[45]!=q[43]||q[45]!=0x3727c5acu)return false;
    for(unsigned v=0;v<3;++v)for(unsigned a=0;a<2;++a){const auto f=glasscaustics::glassCameraTraceFloat(q[16+3*v+a]);
        if(!std::isfinite(f)||f< -1||f>1)return false;}
    std::array<CameraRangeDyadic,9> p{};std::array<CameraRangeDyadic,5> rect{};
    CameraRangeDyadic dx,dy,twice,area,cell,remaining,tau,h,delta,distance,d,one,cx,cy,cz;
    for(unsigned k=0;k<9;++k)if(!cameraRangeDecode(q[16+k],p[k]))return false;
    for(unsigned k=0;k<5;++k){if(!cameraRangeDecode(q[25+k],rect[k]))return false;}h=rect[4];
    if(!cameraRangeDecode(q[37],tau)||tau.sign!=1||!cameraRangeDecode(q[45],delta)||delta.sign!=1
        ||!cameraRangeDecode(q[30],cx)||!cameraRangeDecode(q[31],cy)||!cameraRangeDecode(q[32],cz)
        ||!cameraRangeDecode(0x3f800000u,one)||!cameraRangeAdd(h,cz,d,true)||!cameraRangeAdd(d,one,d,true)||d.sign
        ||h.sign!=1||!cameraRangeDecode(q[44],d)||!cameraRangeAdd(h,d,d,true)||d.sign!=1
        ||!cameraRangeAdd(p[3],p[0],dx,true)||dx.sign!=1||!cameraRangeAdd(p[7],p[1],dy,true)||dy.sign!=1
        ||q[22]!=q[16]||q[20]!=q[17])return false;
    // Positions are (x0,y0,0),(x1,y0,0),(x0,y1,0).
    if(q[21]!=q[18]||q[24]!=q[18])return false;
    if(!cameraRangeMultiply(dx,dy,twice)||twice.sign!=1||twice.exponent<=-512){return false;}area=twice;--area.exponent;
    if(!cameraRangeMultiply(tau,tau,cell)||cell.exponent>510){return false;}cell.exponent+=2;
    if(!cameraRangeAdd(area,cell,remaining,true)||remaining.sign!=1||!cameraRangeAdd(h,delta,distance,true)||distance.sign!=1)return false;
    const auto face=[&](unsigned f,const std::array<CameraRangeDyadic,3>& v,CameraRangeDyadic& out){
        if(f==0)return cameraRangeAdd(p[0],v[0],out,true);
        if(f==1)return cameraRangeAdd(p[1],v[1],out,true);
        if(f==3)return cameraRangeAdd(delta,v[2],out,true);
        if(f==4)return cameraRangeAdd(v[2],h,out,true);
        CameraRangeDyadic x,y,a,b,s;
        return cameraRangeAdd(v[0],p[0],x,true)&&cameraRangeAdd(v[1],p[1],y,true)
            &&cameraRangeMultiply(dy,x,a)&&cameraRangeMultiply(dx,y,b)&&cameraRangeAdd(a,b,s)&&cameraRangeAdd(s,twice,out,true);
    };
    if(!zero(q[274]))return false;
    std::array<CameraRangeDyadic,3> endpoint{};
    if(!cameraRangeDecode(q[272],endpoint[0])||!cameraRangeDecode(q[273],endpoint[1]))return false;
    for(unsigned f=0;f<3;++f)if(!face(f,endpoint,d)||d.sign==1)return false;
    for(unsigned c=0;c<4;++c){std::array<CameraRangeDyadic,3> v{};v[2]=h;
        if(!cameraRangeAdd(cx,tau,v[0],!(c&1))||!cameraRangeAdd(cy,tau,v[1],!(c&2)))return false;
        for(unsigned f=0;f<3;++f)if(!face(f,v,d)||d.sign!=2)return false;}
    for(unsigned v=0;v<3;++v)for(unsigned f=0;f<4;++f){
        if(!cameraRangeAdd(f&1?rect[f]:p[3*v+f/2],f&1?p[3*v+f/2]:rect[f],d,true)||d.sign!=1)return false;}
    const auto tri=[&](unsigned index,unsigned word){return q[320+20*index+word];};
    if(tri(q[5],1)!=tri(q[6],1)||tri(q[5],2)!=tri(q[6],2)||tri(q[5],3)!=tri(q[6],3)
        ||tri(q[5],18)!=1||tri(q[6],18)!=1||tri(q[5],19)||tri(q[6],19)
        ||q[7]!=(tri(q[5],4)<tri(q[6],4)?q[5]:q[6]))return false;
    const unsigned corner[2][3]={{0,3,1},{0,2,3}};
    for(unsigned k=0;k<2;++k)for(unsigned v=0;v<3;++v){const auto c=corner[k][v];const auto index=q[5+k];
        if(tri(index,8+3*v)!=q[25+(c&1)]||tri(index,9+3*v)!=q[27+((c>>1)&1)]||tri(index,10+3*v)!=q[29])return false;}
    for(unsigned b=0;b<8;++b){const auto at=46+16*b,order=208+8*b;
        if(q[at]||q[at+1]||q[at+2]!=b||q[at+3]!=b||q[at+4]||q[at+5]
            ||q[at+8]!=q[272]||q[at+9]!=q[273]||q[at+10]!=q[45]
            ||!zero(q[at+12])||!zero(q[at+13])||q[at+14]!=0x3f800000u
            ||q[order]||q[order+1]||q[order+2]||!zero(q[order+3])||q[order+4]||!zero(q[order+5])
            ||q[order+6]||q[order+7]!=0x3ff00000u
            ||!cameraRangeDecode(q[at+11],d)||!cameraRangeAdd(distance,d,d,true)||d.sign!=1
            ||!cameraRangeDecode(q[at+15],d)||!cameraRangeAdd(d,distance,d,true)||d.sign!=1)return false;}
    std::size_t at=8+q.size();
    const auto same=[&](const CameraRangeDyadic& x){if(r[at]!=std::uint32_t(x.exponent)||r[at+1]!=x.sign)return false;
        for(unsigned k=0;k<8;++k){if(r[at+2+k]!=x.limb[k])return false;}at+=10;return true;};
    for(const auto& x:{area,cell,remaining,distance}){if(!same(x)){return false;}}
    for(unsigned b=0;b<8;++b){CameraRangeDyadic e,rho,absorb;std::array<CameraRangeDyadic,7> power{};
        if(!finiteSourceDecode64(q[174+2*b],q[175+2*b],e)||!finiteSourceDecode64(q[190+2*b],q[191+2*b],rho)
            ||!cameraRangeAdd(one,rho,absorb,true)||absorb.sign==2||!cameraRangeMultiply(e,area,power[0])
            ||!cameraRangeMultiply(e,cell,power[1])||!cameraRangeMultiply(e,remaining,power[2])
            ||!cameraRangeMultiply(rho,power[1],power[3])||!cameraRangeMultiply(absorb,power[1],power[4])
            ||!cameraRangeMultiply(rho,e,power[6])){return false;}power[5]=e;
        for(const auto& x:power){if(!same(x)){return false;}}}
    for(unsigned k=0;k<q[3];++k){std::array<CameraRangeDyadic,3> selected{};unsigned chosen=0;
        if(k!=q[5]&&k!=q[6]){std::array<std::array<CameraRangeDyadic,3>,3> vertices{};
            for(unsigned v=0;v<3;++v)for(unsigned a=0;a<3;++a)if(!cameraRangeDecode(tri(k,8+3*v+a),vertices[v][a]))return false;
            for(unsigned f=0;f<5;++f){bool strict=true;std::array<CameraRangeDyadic,3> margins{};
                for(unsigned v=0;v<3;++v){if(!face(f,vertices[v],margins[v])){return false;}strict=strict&&margins[v].sign==1;}
                if(strict&&!chosen){chosen=f+1;selected=margins;}}
            if(!chosen)return false;}
        if(r[at++]!=k||r[at++]!=chosen){return false;}for(const auto& x:selected)if(!same(x))return false;}
    return at==r.size();
}

inline std::string finiteGratingReceiverComputeSource() {
    return finiteGratingSourceComputeSource()+"\n#define FR_MUL "+std::to_string(cameraraycoefficients::multiplyVisits(256))
        +"u\n#define FR_ADD "+std::to_string(cameraraycoefficients::kAddVisits)+"u\n"+R"FINITE_RECEIVER(
bool frZero(uint v){return (v&0x7fffffffu)==0u;}
void frFace(uint f,thread const CCDyadic* p,thread const CCDyadic& dx,thread const CCDyadic& dy,
    thread const CCDyadic& twice,thread const CCDyadic& delta,thread const CCDyadic& h,
    thread const CCDyadic* v,thread CCDyadic& out,thread ELWork& w){
    if(f==0u){ccSubtract(p[0],v[0],out,256u,w);return;}
    if(f==1u){ccSubtract(p[1],v[1],out,256u,w);return;}
    if(f==3u){ccSubtract(delta,v[2],out,256u,w);return;}
    if(f==4u){ccSubtract(v[2],h,out,256u,w);return;}
    CCDyadic x,y,a,b,s;ccSubtract(v[0],p[0],x,256u,w);ccSubtract(v[1],p[1],y,256u,w);
    ccMultiply(dy,x,a,256u,w);ccMultiply(dx,y,b,256u,w);ccAdd(a,b,s,256u,w);ccSubtract(s,twice,out,256u,w);
}
kernel void finiteGratingReceiver(device const uint* q [[buffer(0)]],device uint* r [[buffer(1)]],
    constant uint4& limits [[buffer(2)]],uint index [[thread_position_in_grid]]){
    if(index!=0u)return;for(uint k=0u;k<limits.w;k++)r[k]=0u;
    for(uint k=0u;k<limits.z;k++)r[8u+k]=q[k];
    r[1]=0x46524731u;r[2]=1u;r[3]=256u;r[5]=limits.x;r[6]=8u;r[7]=0x504931u;
    ELWork w;w.used=0u;w.limit=(80u+6u*limits.x)*FR_MUL+(160u+24u*limits.x)*FR_ADD
        +(64u+9u*limits.x)*205u+600u+32u*limits.x+105u*limits.x+35u;w.capacity=35u;w.ok=true;
    bool ok=q[0]==0x46524731u&&q[1]==1u&&q[2]==256u&&q[3]==limits.x&&limits.x>0u&&limits.x<=256u
        &&q[4]<limits.x&&q[5]<limits.x&&q[6]<limits.x&&q[5]!=q[6]&&limits.y==8u
        &&limits.z==320u+20u*limits.x&&limits.w==8u+limits.z+600u+32u*limits.x
        &&frZero(q[18])&&frZero(q[21])&&frZero(q[24])&&q[22]==q[16]&&q[20]==q[17]
        &&frZero(q[33])&&q[34]==0x3f800000u&&frZero(q[35])&&frZero(q[36])
        &&q[38]==0x3f800000u&&q[41]==0x3f800000u&&q[42]==0x3f800000u
        &&q[45]==q[43]&&q[45]==0x3727c5acu;
    uint failure=ok?0u:1u;
    CCDyadic p[9],rect[5],dx,dy,twice,area,cell,remaining,tau,h,delta,distance,d,one,cx,cy,cz;
    for(uint k=0u;k<9u;k++)ccDecode(q[16u+k],true,p[k],256u,w);
    for(uint v=0u;v<3u;v++)for(uint a=0u;a<2u;a++){float f=as_type<float>(q[16u+3u*v+a]);ok=ok&&isfinite(f)&&f>=-1.0f&&f<=1.0f;}
    for(uint k=0u;k<5u;k++)ccDecode(q[25u+k],true,rect[k],256u,w);h=rect[4];
    ccDecode(q[37],true,tau,256u,w);ccDecode(q[45],true,delta,256u,w);
    ccDecode(q[30],true,cx,256u,w);ccDecode(q[31],true,cy,256u,w);ccDecode(q[32],true,cz,256u,w);
    ccDecode(0x3f800000u,true,one,256u,w);ccSubtract(h,cz,d,256u,w);ccSubtract(d,one,d,256u,w);
    ok=ok&&tau.m.sign==1u&&delta.m.sign==1u&&h.m.sign==1u&&d.m.sign==0u;
    ccDecode(q[44],true,d,256u,w);ccSubtract(h,d,d,256u,w);if(d.m.sign!=1u)failure|=2048u;ok=ok&&d.m.sign==1u;
    ccSubtract(p[3],p[0],dx,256u,w);ccSubtract(p[7],p[1],dy,256u,w);ccMultiply(dx,dy,twice,256u,w);
    ok=ok&&dx.m.sign==1u&&dy.m.sign==1u&&twice.m.sign==1u&&twice.exponent> -512;
    area=twice;area.exponent--;ccMultiply(tau,tau,cell,256u,w);ok=ok&&cell.exponent<=510;cell.exponent+=2;
    ccSubtract(area,cell,remaining,256u,w);ccSubtract(h,delta,distance,256u,w);
    ok=ok&&remaining.m.sign==1u&&distance.m.sign==1u;
    if(!ok)failure|=2u;
    CCDyadic endpoint[3];ccDecode(q[272],true,endpoint[0],256u,w);ccDecode(q[273],true,endpoint[1],256u,w);ccZero(endpoint[2],w);
    bool endpointOK=frZero(q[274]);
    for(uint f=0u;f<3u;f++){frFace(f,p,dx,dy,twice,delta,h,endpoint,d,w);endpointOK=endpointOK&&d.m.sign!=1u;}
    if(!endpointOK)failure|=16u;ok=ok&&endpointOK;
    bool cellOK=true;
    for(uint c=0u;c<4u;c++){CCDyadic v[3];v[2]=h;
        if((c&1u)!=0u)ccAdd(cx,tau,v[0],256u,w);else ccSubtract(cx,tau,v[0],256u,w);
        if((c&2u)!=0u)ccAdd(cy,tau,v[1],256u,w);else ccSubtract(cy,tau,v[1],256u,w);
        for(uint f=0u;f<3u;f++){frFace(f,p,dx,dy,twice,delta,h,v,d,w);cellOK=cellOK&&d.m.sign==2u;}}
    for(uint v=0u;v<3u;v++)for(uint f=0u;f<4u;f++){
        if((f&1u)!=0u)ccSubtract(rect[f],p[3u*v+f/2u],d,256u,w);
        else ccSubtract(p[3u*v+f/2u],rect[f],d,256u,w);ok=ok&&d.m.sign==1u;}
    if(!cellOK)failure|=4u;ok=ok&&cellOK;
    bool pairOK=true;
    uint left=320u+20u*q[5],right=320u+20u*q[6];
    if(q[5]<limits.x&&q[6]<limits.x){
        pairOK=pairOK&&q[left+1u]==q[right+1u]&&q[left+2u]==q[right+2u]&&q[left+3u]==q[right+3u]
            &&q[left+18u]==1u&&q[right+18u]==1u&&q[left+19u]==0u&&q[right+19u]==0u
            &&q[7]==(q[left+4u]<q[right+4u]?q[5]:q[6]);
        const uint corners[6]={0u,3u,1u,0u,2u,3u};
        for(uint k=0u;k<2u;k++)for(uint v=0u;v<3u;v++){uint c=corners[3u*k+v],base=k==0u?left:right;
            pairOK=pairOK&&q[base+8u+3u*v]==q[25u+(c&1u)]&&q[base+9u+3u*v]==q[27u+((c>>1u)&1u)]&&q[base+10u+3u*v]==q[29];}
    }
    if(!pairOK)failure|=8u;ok=ok&&pairOK;
    for(uint b=0u;b<8u;b++){uint at=46u+16u*b,o=208u+8u*b;
        bool orderOK=q[at]==0u&&q[at+1u]==0u&&q[at+2u]==b&&q[at+3u]==b&&q[at+4u]==0u&&q[at+5u]==0u
            &&q[at+8u]==q[272]&&q[at+9u]==q[273]&&q[at+10u]==q[45]
            &&frZero(q[at+12u])&&frZero(q[at+13u])&&q[at+14u]==0x3f800000u
            &&q[o]==0u&&q[o+1u]==0u&&q[o+2u]==0u&&frZero(q[o+3u])&&q[o+4u]==0u&&frZero(q[o+5u])
            &&q[o+6u]==0u&&q[o+7u]==0x3ff00000u;
        if(!orderOK)failure|=16u;ok=ok&&orderOK;
        ccDecode(q[at+11u],true,d,256u,w);ccSubtract(distance,d,d,256u,w);if(d.m.sign!=1u)failure|=32u;ok=ok&&d.m.sign==1u;
        ccDecode(q[at+15u],true,d,256u,w);ccSubtract(d,distance,d,256u,w);if(d.m.sign!=1u)failure|=64u;ok=ok&&d.m.sign==1u;
    }
    uint at=8u+limits.z;fsStore(r,at,area,w);at+=10u;fsStore(r,at,cell,w);at+=10u;
    fsStore(r,at,remaining,w);at+=10u;fsStore(r,at,distance,w);at+=10u;
    for(uint b=0u;b<8u;b++){CCDyadic e,rho,absorb,power[7];fsDecode64(q[174u+2u*b],q[175u+2u*b],e,w);
        if(!w.ok)failure|=128u;
        fsDecode64(q[190u+2u*b],q[191u+2u*b],rho,w);ccSubtract(one,rho,absorb,256u,w);if(absorb.m.sign==2u)failure|=256u;ok=ok&&absorb.m.sign!=2u;
        ccMultiply(e,area,power[0],256u,w);ccMultiply(e,cell,power[1],256u,w);ccMultiply(e,remaining,power[2],256u,w);
        ccMultiply(rho,power[1],power[3],256u,w);ccMultiply(absorb,power[1],power[4],256u,w);power[5]=e;
        ccMultiply(rho,e,power[6],256u,w);for(uint k=0u;k<7u;k++){fsStore(r,at,power[k],w);at+=10u;}}
    for(uint t=0u;t<limits.x;t++){uint chosen=0u;CCDyadic selected[3];for(uint v=0u;v<3u;v++)ccZero(selected[v],w);
        if(t!=q[5]&&t!=q[6]){CCDyadic vertices[9];for(uint k=0u;k<9u;k++)ccDecode(q[320u+20u*t+8u+k],true,vertices[k],256u,w);
            for(uint f=0u;f<5u;f++){bool strict=true;CCDyadic margins[3];
                for(uint v=0u;v<3u;v++){frFace(f,p,dx,dy,twice,delta,h,vertices+3u*v,margins[v],w);strict=strict&&margins[v].m.sign==1u;}
                if(strict&&chosen==0u){chosen=f+1u;for(uint v=0u;v<3u;v++)selected[v]=margins[v];}}
            if(chosen==0u)failure|=512u;ok=ok&&chosen!=0u;}
        r[at]=t;r[at+1u]=chosen;at+=2u;for(uint v=0u;v<3u;v++){fsStore(r,at,selected[v],w);at+=10u;}}
    if(!w.ok)failure|=1024u;r[4]=w.used;r[7]=failure?failure:0x504931u;
    if(ok&&w.ok&&at==limits.w)r[0]=1u;
}
)FINITE_RECEIVER";
}

inline bool cameraRangeReconstruct(const CameraRangeRequest& request,std::array<CameraRangeDyadic,31>& values,std::uint32_t& axis,
    bool upperRefusal=false) {
    cameraraycoefficients::Result record{};cameraraycoefficients::Request expected{};
    std::copy_n(request.begin(),557,record.begin());std::copy_n(record.begin()+4,57,expected.begin());
    if(!camerawideline::validateRecord(record,expected)||request[559]!=1||request[560]!=256||request[561]!=kCameraRangeWireTag)return false;
    std::array<CameraRangeDyadic,31> v{};
    for(unsigned form=0;form<4;++form){const unsigned f=form<3?form:7;const unsigned base=61+28*f;
        CameraRangeDyadic a,b,c;if(!cameraRangeLoad(record.data()+base+1,int(std::int32_t(record[base])),a)
            ||!cameraRangeLoad(record.data()+base+10,int(std::int32_t(record[base])),b)
            ||!cameraRangeLoad(record.data()+base+19,int(std::int32_t(record[base])),c))return false;
        v[form*4]=c;if(!cameraRangeAdd(a,c,v[form*4+1])||!cameraRangeAdd(b,c,v[form*4+2])
            ||!cameraRangeAdd(v[form*4+1],b,v[form*4+3]))return false;
    }
    if(!cameraRangeLoad(record.data()+538,int(std::int32_t(record[537])),v[16])
        ||!cameraRangeDecode(request[557],v[17])||!cameraRangeDecode(request[558],v[18])
        ||v[17].sign==2||v[18].sign!=1)return false;
    CameraRangeDyadic difference;if(!cameraRangeAdd(v[18],v[17],difference,true)||difference.sign!=1)return false;
    for(unsigned form=0;form<4;++form){CameraRangeDyadic low=v[form*4],high=low;
        const auto sign=low.sign;bool uniform=sign!=0;if(low.sign)low.sign=high.sign=1;
        for(unsigned corner=1;corner<4;++corner){auto x=v[form*4+corner];uniform=uniform&&x.sign==sign;if(x.sign)x.sign=1;
            if(!cameraRangeAdd(x,low,difference,true)){return false;}if(difference.sign==2)low=x;
            if(!cameraRangeAdd(x,high,difference,true)){return false;}if(difference.sign==1)high=x;}
        if(form==3){if(!uniform||!low.sign){return false;}v[19]=low;v[20]=high;}
        else {v[21+form]=high;v[24+form]=uniform?low:CameraRangeDyadic{};}
    }
    CameraRangeDyadic h2,a2,b2,nlo2,nhi2,squares[3],sum,temp,lower;
    if(!cameraRangeMultiply(v[16],v[16],h2)||!cameraRangeMultiply(v[17],v[17],a2)
        ||!cameraRangeMultiply(v[18],v[18],b2)||!cameraRangeMultiply(v[19],v[19],nlo2)
        ||!cameraRangeMultiply(v[20],v[20],nhi2))return false;
    for(unsigned k=0;k<3;++k)if(!cameraRangeMultiply(v[21+k],v[21+k],squares[k]))return false;
    if(!cameraRangeAdd(squares[0],squares[1],temp)||!cameraRangeAdd(temp,squares[2],sum)
        ||!cameraRangeMultiply(h2,sum,temp)||!cameraRangeMultiply(b2,nlo2,lower)
        ||!cameraRangeAdd(lower,temp,v[30],true)||!cameraRangeMultiply(a2,nhi2,lower))return false;
    axis=0;for(unsigned k=0;k<3;++k){if(!cameraRangeMultiply(v[24+k],v[24+k],sum)
        ||!cameraRangeMultiply(h2,sum,temp)||!cameraRangeAdd(temp,lower,v[27+k],true))return false;
        if(!axis&&v[24+k].sign&&v[27+k].sign==1)axis=k+1;}
    if(!axis||(upperRefusal?v[30].sign==1:v[30].sign!=1)){return false;}values=v;return true;
}

inline std::string cameraGeometricRangeComputeSource() {
    return std::string("#include <metal_stdlib>\nusing namespace metal;\n")+camerawideline::languageSource(true)
        +"\n#define CR_VISITS "+std::to_string(kCameraRangeVisits)+"u\n#define CR_SCALARS "+std::to_string(kCameraRangeScalarOperations)+"u\n"
        +R"RANGE(
bool crLoad(thread const uint* words,int exponent,thread CCDyadic& r,thread ELWork& w){
    ccZero(r,w);r.exponent=exponent;r.m.sign=words[0];for(uint k=0u;k<8u;k++)r.m.limb[k]=words[1u+k];
    ccNormalize(r,256u,w);return w.ok;
}
kernel void cameraGeometricRange(device const uint* inputs [[buffer(0)]],device uint* outputs [[buffer(1)]],
    constant uint4& limits [[buffer(2)]],uint pixel [[thread_position_in_grid]]) {
    if(pixel>=limits.x)return;uint request[575],result[893],record[557],q[65];
    for(uint k=0u;k<575u;k++)request[k]=inputs[pixel*575u+k];
    for(uint k=0u;k<893u;k++)result[k]=0u;
    for(uint k=0u;k<575u;k++)result[8u+k]=request[k];
    result[1]=CR_SCALARS;result[4]=0x43524731u;result[5]=1u;result[6]=256u;
    ELWork w;w.used=0u;w.limit=CR_VISITS;w.capacity=35u;w.ok=true;
    for(uint k=0u;k<557u;k++)record[k]=request[k];for(uint k=0u;k<65u;k++)q[k]=k<57u?record[4u+k]:0u;
    CWLine lines[17];CCDyadic v[31],a,b,c,d,h2,a2,b2,nlo2,nhi2,squares[3],sum,temp,lower;
    bool ok=request[559]==1u&&request[560]==256u&&request[561]==0x43524731u&&cwPrepare(record,q,lines,w);
    for(uint form=0u;form<4u&&ok;form++){
        uint f=form<3u?form:7u,base=61u+28u*f;
        ok=crLoad(record+base+1u,int(record[base]),a,w)&&crLoad(record+base+10u,int(record[base]),b,w)
            &&crLoad(record+base+19u,int(record[base]),c,w);
        if(!ok)break;
        v[form*4u]=c;ccAdd(a,c,v[form*4u+1u],256u,w);ccAdd(b,c,v[form*4u+2u],256u,w);
        ccAdd(v[form*4u+1u],b,v[form*4u+3u],256u,w);ok=ok&&w.ok;
    }
    if(ok){ok=crLoad(record+538u,int(record[537]),v[16],w);ccDecode(request[557],true,v[17],256u,w);
        ccDecode(request[558],true,v[18],256u,w);ccSubtract(v[18],v[17],d,256u,w);
        ok=ok&&w.ok&&v[17].m.sign!=2u&&v[18].m.sign==1u&&d.m.sign==1u;}
    for(uint form=0u;form<4u&&ok;form++){
        CCDyadic lo=v[form*4u],hi=lo;uint sign=lo.m.sign;bool uniform=sign!=0u;if(sign!=0u)lo.m.sign=hi.m.sign=1u;
        for(uint corner=1u;corner<4u;corner++){
            CCDyadic x=v[form*4u+corner];uniform=uniform&&x.m.sign==sign;if(x.m.sign!=0u)x.m.sign=1u;
            ccSubtract(x,lo,d,256u,w);if(d.m.sign==2u)lo=x;
            ccSubtract(x,hi,d,256u,w);if(d.m.sign==1u)hi=x;
        }
        if(form==3u){ok=uniform&&lo.m.sign!=0u;v[19]=lo;v[20]=hi;}
        else {v[21u+form]=hi;if(uniform)v[24u+form]=lo;else ccZero(v[24u+form],w);}
        ok=ok&&w.ok;
    }
    uint axis=0u;bool arithmeticComplete=false;
    if(ok){ccMultiply(v[16],v[16],h2,256u,w);ccMultiply(v[17],v[17],a2,256u,w);ccMultiply(v[18],v[18],b2,256u,w);
        ccMultiply(v[19],v[19],nlo2,256u,w);ccMultiply(v[20],v[20],nhi2,256u,w);
        for(uint k=0u;k<3u;k++)ccMultiply(v[21u+k],v[21u+k],squares[k],256u,w);
        ccAdd(squares[0],squares[1],temp,256u,w);ccAdd(temp,squares[2],sum,256u,w);
        ccMultiply(h2,sum,temp,256u,w);ccMultiply(b2,nlo2,lower,256u,w);ccSubtract(lower,temp,v[30],256u,w);
        ok=w.ok&&v[30].m.sign==1u;ccMultiply(a2,nhi2,lower,256u,w);
        for(uint k=0u;k<3u;k++){ccMultiply(v[24u+k],v[24u+k],sum,256u,w);ccMultiply(h2,sum,temp,256u,w);
            ccSubtract(temp,lower,v[27u+k],256u,w);if(axis==0u&&v[24u+k].m.sign!=0u&&v[27u+k].m.sign==1u)axis=k+1u;}
        arithmeticComplete=w.ok&&axis!=0u;ok=ok&&arithmeticComplete;
    }
    // The existing computed fields also identify a completed upper refusal.
    // They remain private diagnostics and never acquire success status.
    if(arithmeticComplete){for(uint k=0u;k<31u;k++){
        uint base=583u+k*10u;elStep(w);result[base]=uint(v[k].exponent);elStep(w);result[base+1u]=v[k].m.sign;
        for(uint limb=0u;limb<8u;limb++){elStep(w);result[base+2u+limb]=v[k].m.limb[limb];}
    }arithmeticComplete=arithmeticComplete&&w.ok;ok=ok&&w.ok;}
    result[2]=w.used;if(ok){result[0]=1u;result[3]=axis;}
    else if(arithmeticComplete&&v[30].m.sign!=1u){result[3]=axis;result[7]=1u;}
    else for(uint k=583u;k<893u;k++)result[k]=0u;
    for(uint k=0u;k<893u;k++)outputs[pixel*893u+k]=result[k];
}
)RANGE";
}

inline constexpr std::uint32_t kCameraDiagonalPairWords=557+13*73;
inline constexpr std::uint32_t kCameraDiagonalHeaderWords=16;
inline constexpr std::uint32_t kCameraDiagonalTriangleWords=20;
using CameraDiagonalTriangle=std::array<std::uint32_t,kCameraDiagonalTriangleWords>;
using CameraDiagonalPixel=std::array<std::uint32_t,8>;
inline std::string cameraDiagonalComputeSource()
{
    return std::string("#include <metal_stdlib>\nusing namespace metal;\n")
        +camerapixelsigns::languageSource(true)+R"DIAGONAL(
struct CameraDiagonal {
    device const uint* inputs; device uint* certificates;
    uint uPixelCount,uTriangleCount,uGeneration;
static constexpr uint pairWords=1506u;
uint tri(uint t,uint w){return inputs[23u+8u*uPixelCount+t*20u+w];}
uint pix(uint p,uint w){return inputs[23u+p*8u+w];}
bool endpoint(uint t,uint a,uint s,uint b){
    if(tri(t,5u+a)!=tri(s,5u+b))return false;
    for(uint k=0u;k<3u;k++)if(tri(t,8u+a*3u+k)!=tri(s,8u+b*3u+k))return false;
    return true;
}
void request(uint p,uint a,uint b,thread uint (&q)[57]){
    for(uint k=0u;k<23u;k++)q[k]=inputs[k];q[8]=pix(p,0u);q[9]=pix(p,1u);
    for(uint k=0u;k<17u;k++){q[23u+k]=tri(a,k);q[40u+k]=tri(b,k);}
}
bool uniformSign(thread const uint (&r)[73],uint s){
    return r[0]==1u&&r[4]==s&&r[5]==s&&r[6]==s&&r[7]==s;
}
bool corners(thread const uint (&q)[57],thread const uint (&record)[557],uint form,uint base,thread uint (&r)[73]){
    uint a[65];for(int k=0;k<57;k++)a[k]=q[k];
    a[57]=1u;a[58]=CP_REQUEST_TAG;a[59]=0u;a[60]=CP_CHARGE;
    a[61]=form;a[62]=CP_WIRE;a[63]=0u;a[64]=0u;
    cpEvaluate(record,a,r);
    for(uint k=0u;k<73u;k++)certificates[base+557u+(form-4u)*73u+k]=r[k];
    return r[0]==1u;
}
bool reversedForm(thread const uint (&record)[557],uint a,uint b){
    uint x=61u+a*28u,y=61u+b*28u;
    if(record[x]!=record[y])return false;
    for(uint k=0u;k<3u;k++){
        uint sx=record[x+1u+k*9u],sy=record[y+1u+k*9u];
        if(sy!=(sx==0u?0u:3u-sx))return false;
        for(uint l=1u;l<9u;l++)if(record[x+1u+k*9u+l]!=record[y+1u+k*9u+l])return false;
    }
    return true;
}
void run(uint p){
    if(p>=uPixelCount)return;
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
};
kernel void cameraDiagonalCoverage(device const uint* inputs [[buffer(0)]],device uint* certificates [[buffer(1)]],
    constant uint4& limits [[buffer(2)]],uint p [[thread_position_in_grid]]) {
    CameraDiagonal work{inputs,certificates,limits.x,limits.y,limits.z};work.run(p);
}
)DIAGONAL";
}

// Scalar-only storage ABI avoids float3 padding and preserves uint identities.
struct Hit {
    std::int32_t status; float distance, position[3], normal[3];
    std::uint32_t object, material, instance, triangle, entering, work;
};
struct Path {
    std::uint32_t work, count, visits, complete, blocked, queries;
    float launched, deposited, absorbed, escaped, unresolved;
    Hit source, entry;
    float padding[21], masked;
    std::uint32_t roughTransportVersion, interfaceSample, reserved;
};
struct Deposit {
    std::uint32_t work, historyLow, historyHigh, interfaces, visible, projected, x, y;
    float fraction, incident, outgoing, absorbed, area, depth, rasterDepth;
    std::uint32_t object, material, instance, triangle;
    Hit receiver, camera, pixel;
};
static_assert(sizeof(Hit)==56 && sizeof(Path)==256 && sizeof(Deposit)==244);
static_assert(offsetof(Path,masked)==15*16);
static_assert(sizeof(Path)<=glasscaustics::kPathReceiptStride*16
    && sizeof(Deposit)<=glasscaustics::kDepositReceiptStride*16);
// V2 consumes the last four padding floats; the native entry IDs stay intact.
struct MultipleScatterPath {
    std::uint32_t work, count, visits, complete, blocked, queries;
    float launched, deposited, absorbed, escaped, unresolved;
    Hit source, entry;
    float padding[17], microEvents, microDraws, microResidual, quadrature, masked;
    std::uint32_t roughTransportVersion, interfaceSample, reserved;
};
struct MultipleScatterDeposit { Deposit value; std::uint8_t ancestry[32]; };
struct CoupledDeposit { Deposit value; std::uint8_t ancestry[32]; std::uint32_t extension[4]; };
static_assert(sizeof(CoupledDeposit)==292 && offsetof(CoupledDeposit,ancestry)==244
    && offsetof(CoupledDeposit,extension)==276 && sizeof(CoupledDeposit)<=glasscaustics::kDepositReceiptStride*16);

static_assert(sizeof(MultipleScatterPath)==256 && offsetof(MultipleScatterPath,source)==44
    && offsetof(MultipleScatterPath,entry)==100 && offsetof(MultipleScatterPath,microEvents)==224
    && offsetof(MultipleScatterPath,masked)==240 && offsetof(MultipleScatterPath,reserved)==252);
static_assert(sizeof(MultipleScatterDeposit)==276 && offsetof(MultipleScatterDeposit,ancestry)==244
    && sizeof(MultipleScatterDeposit)<=glasscaustics::kDepositReceiptStride*16);
// MSL float3 alignment gives CBranch64 and the v2 branch96. The shared
// admission reserves48 extra bytes per pending branch and320 for exit arrays.
inline constexpr std::uint64_t multipleScatterBranchBytes=96, multipleScatterBranchDelta=32,
    multipleScatterExitBytes=320;
static_assert(multipleScatterBranchDelta<=48);
inline bool decodePathReceiptExtensions(const Path& native,bool multipleScatter,
    glasscaustics::NativePathReceipt& output) {
    const auto paddingCount=multipleScatter?17u:21u;
    if(native.reserved!=(multipleScatter?glasscaustics::kPrivateMultipleScatterVersion:0u)
        ||native.roughTransportVersion>glasscaustics::kRoughTransportVersion
        ||native.interfaceSample>=glasscaustics::kRoughInterfaceSamples
        ||native.complete>1u||native.blocked>1u
        ||std::any_of(native.padding,native.padding+paddingCount,[](float value){return value!=0;}))return false;
    glasscaustics::Texel micro{};
    if(multipleScatter)std::copy_n(native.padding+17,4,micro.begin());
    return glasscaustics::decodePathExtensions(micro,{native.masked,float(native.roughTransportVersion),
        float(native.interfaceSample),float(native.reserved)},glasscaustics::PathExtensionLayout::Metal,output);
}
struct alignas(16) Params {
    std::array<float,4> query, source, camera, rotation, projection;
    std::array<float,4> ior[2], radiance[2], cie[8];
    std::array<std::uint32_t,4> extent; // width, height, work count, triangle count
    std::array<std::uint32_t,4> policy; // interfaces, visits, pending, glass owner
    std::array<float,4> tail; // minimum branch, maximum unresolved, roughness, samples
    std::array<std::uint32_t,4> deposition; // padded key count, maximum deposits
    std::array<std::uint32_t,4> allowance; // admitted ceiling; receipts retain executed units
};
inline opaquevisibility::ClosestHit decode(const Hit& input) {
    opaquevisibility::ClosestHit out;
    out.status=input.status; out.distance=input.distance;
    std::copy_n(input.position,3,out.position.begin()); std::copy_n(input.normal,3,out.outwardNormal.begin());
    out.objectId=input.object; out.materialId=input.material; out.instanceId=input.instance;
    out.triangleId=input.triangle; out.entering=input.entering!=0; out.work=input.work; return out;
}
struct alignas(16) DirectGratingHit { Hit value {}; std::array<std::uint32_t,6> padding {}; };
struct alignas(16) DirectGratingResult {
    glasscaustics::NativeDirectGratingEndpointInput input;
    glasscaustics::NativeDirectionalSourceEcho source;
    DirectGratingHit endpoint,shadow;
    glasscaustics::Texel barycentric {},originMinimum {},directionMaximum {};
    std::array<float,8> irradiance {};
    std::array<std::uint32_t,4> control {},work {};
};
struct alignas(16) DirectGratingWitness {
    std::array<std::uint32_t,4> primitive {};
    std::array<glasscaustics::Texel,4> geometry {}; // three points, normal
};
struct alignas(16) DirectGratingParams {
    glasscaustics::Texel query {},source {};
    std::array<glasscaustics::Texel,2> radiance {};
    std::array<std::uint32_t,4> control {};
};
static_assert(sizeof(DirectGratingHit)==80&&sizeof(DirectGratingResult)==glasscaustics::kDirectGratingContributionNativeBytes
    &&sizeof(DirectGratingWitness)==80&&sizeof(DirectGratingParams)==80);
static_assert(offsetof(DirectGratingResult,source)==64&&offsetof(DirectGratingResult,endpoint)==160
    &&offsetof(DirectGratingResult,shadow)==240&&offsetof(DirectGratingResult,barycentric)==320
    &&offsetof(DirectGratingResult,irradiance)==368&&offsetof(DirectGratingResult,control)==400
    &&offsetof(DirectGratingResult,work)==416);
inline glasscaustics::NativeDirectGratingContributionSample decodeDirectGrating(const DirectGratingResult& r) {
    glasscaustics::NativeDirectGratingContributionSample s;
    s.input=r.input;s.source=r.source;s.endpoint=decode(r.endpoint.value);s.sourceHit=decode(r.shadow.value);
    s.barycentric={r.barycentric[0],r.barycentric[1],r.barycentric[2]};
    s.sourceOriginMinimum=r.originMinimum;s.sourceDirectionMaximum=r.directionMaximum;s.surfaceIrradiance=r.irradiance;
    s.queryWork=std::uint64_t(r.work[0])+r.work[1];
    s.outcome=glasscaustics::NativeDirectGratingContributionSample::Outcome(r.control[2]);s.complete=r.control[1]==1;
    return s;
}
inline bool sameDirectGratingSample(const glasscaustics::NativeDirectGratingContributionSample& a,
    const glasscaustics::NativeDirectGratingContributionSample& b) {
    const auto hit=[](const auto& h){return std::tie(h.status,h.distance,h.position,h.outwardNormal,
        h.objectId,h.materialId,h.instanceId,h.triangleId,h.entering,h.work);};
    return std::memcmp(&a.input,&b.input,sizeof(a.input))==0&&std::memcmp(&a.source,&b.source,sizeof(a.source))==0
        &&hit(a.endpoint)==hit(b.endpoint)&&hit(a.sourceHit)==hit(b.sourceHit)
        &&a.barycentric==b.barycentric&&a.sourceOriginMinimum==b.sourceOriginMinimum
        &&a.sourceDirectionMaximum==b.sourceDirectionMaximum&&a.surfaceIrradiance==b.surfaceIrradiance
        &&a.queryWork==b.queryWork&&a.outcome==b.outcome&&a.complete==b.complete;
}
inline std::string mathSource(bool rough=false) {
    auto source=rough?glasscaustics::nativeRoughMathSource():glasscaustics::nativeMathSource();
    const auto replace=[&](const std::string& before,const std::string& after) {
        std::size_t at=0; while((at=source.find(before,at))!=std::string::npos) {source.replace(at,before.size(),after);at+=after.size();}
    };
    replace("out vec3 corner","thread float3& corner");
    replace("out vec4 footprint","thread float4& footprint");
    replace("out vec3 transmitted","thread float3& transmitted");
    replace("out vec3 reflected","thread float3& reflected");
    replace("out bool tir","thread bool& tir");
    replace("out vec3 power","thread float3& power");
    replace("out bool complete","thread bool& complete");
    replace("vec2","float2");replace("vec3","float3");replace("vec4","float4");
    return source;
}
inline std::string querySource() {
    auto source=opaquevisibility::metalShader();
    const auto replace=[&](const std::string& before,const std::string& after) {
        std::size_t at=0; while((at=source.find(before,at))!=std::string::npos) {source.replace(at,before.size(),after);at+=after.size();}
    };
    // Instrument the released counter without changing queries or traversal.
    replace("struct SceneClosestHit {", "struct SceneClosestHit { uint causticWork;");
    replace("hit.status=status;", "hit.causticWork=0u;hit.status=status;");
    replace("++work;", "++work;hit.causticWork=work;");
    return source;
}
inline std::string directGratingIrradianceShader() {
    return std::string("#include <metal_stdlib>\nusing namespace metal;\n")+querySource()+R"METAL(
struct DInput {uint4 primitive,control;float4 originMinimum,directionMaximum;};
struct DSource {uint4 control;float4 coefficients[2];};
struct DEcho {DSource input;float4 rawRadiance[2],directionTrace;};
struct DHit {int status;float distance,position[3],normal[3];uint object,material,instance,triangle,entering,work;uint padding[6];};
struct DResult {DInput input;DEcho source;DHit endpoint,shadow;float4 barycentric,originMinimum,directionMaximum;float4 irradiance[2];uint4 control,work;};
struct DWitness {uint4 primitive;float4 geometry[4];};
struct DParams {float4 query,source,radiance[2];uint4 control;};
static_assert(sizeof(DInput)==64&&sizeof(DEcho)==96&&sizeof(DHit)==80&&sizeof(DResult)==432,"direct result ABI");
static_assert(sizeof(DWitness)==80&&sizeof(DParams)==80,"direct query ABI");
DHit directStore(SceneClosestHit h) {
    DHit r={};r.status=h.status;r.distance=h.distance;
    for(uint a=0u;a<3u;++a){r.position[a]=h.position[a];r.normal[a]=h.outwardNormal[a];}
    r.object=h.objectId;r.material=h.materialId;r.instance=h.instanceId;r.triangle=h.triangleId;
    r.entering=h.entering?1u:0u;r.work=h.causticWork;return r;
}
kernel void directGratingIrradiance(constant DParams& u [[buffer(0)]],
    device const DInput* inputs [[buffer(1)]],constant DSource& source [[buffer(2)]],
    device const DWitness* witnesses [[buffer(3)]],device DResult* results [[buffer(4)]],
    texture2d<float> query [[texture(0)]],uint i [[thread_position_in_grid]]) {
    if(i>=u.control.x)return;
    DResult r={};r.input=inputs[i];r.source.input=source;
    for(uint h=0u;h<2u;++h)r.source.rawRadiance[h]=u.radiance[h];r.source.directionTrace=u.source;
    r.control=uint4(1u,0u,2u,0u);
    if(any(r.input.control!=uint4(1u,0u,0u,0u))||source.control.x!=1u||source.control.y!=1u
        ||source.control.z!=1u||source.control.w==0u){results[i]=r;return;}
    SceneClosestHit endpoint=sceneClosestHit(r.input.originMinimum.xyz,r.input.directionMaximum.xyz,
        r.input.originMinimum.w,r.input.directionMaximum.w,u.query,query);
    r.endpoint=directStore(endpoint);r.work.x=endpoint.causticWork;
    if(endpoint.status!=1||any(r.input.primitive!=uint4(endpoint.objectId,endpoint.materialId,endpoint.instanceId,endpoint.triangleId))) {
        results[i]=r;return;
    }
    int found=-1;for(uint t=0u;t<u.control.y;++t)if(all(witnesses[t].primitive==r.input.primitive)){found=int(t);break;}
    if(found<0){results[i]=r;return;}
    DWitness witness=witnesses[uint(found)];float3 a=witness.geometry[0].xyz;
    float3 e=witness.geometry[1].xyz-a,f=witness.geometry[2].xyz-a,h=endpoint.position-a;
    float ee=dot(e,e),ef=dot(e,f),ff=dot(f,f),he=dot(h,e),hf=dot(h,f),det=ee*ff-ef*ef;
    if(!sceneFinite(det)||det<=0.0f){results[i]=r;return;}
    float b=(ff*he-ef*hf)/det,c=(ee*hf-ef*he)/det;float3 bary=float3(1.0f-b-c,b,c);
    float3 normal=witness.geometry[3].xyz;
    r.barycentric=float4(bary,0.0f);
    if(!sceneFinite3(bary)||any(bary<=float3(1e-5f))||any(bary>=float3(1.0f-1e-5f))
        ||length(endpoint.outwardNormal-normal)>1e-5f){results[i]=r;return;}
    r.originMinimum=float4(endpoint.position,1e-5f);r.directionMaximum=u.source;
    SceneClosestHit shadow=sceneClosestHit(endpoint.position,u.source.xyz,1e-5f,u.source.w,u.query,query);
    r.shadow=directStore(shadow);r.work.y=shadow.causticWork;
    if(shadow.status<0){results[i]=r;return;}
    float cosine=max(0.0f,dot(normal,u.source.xyz));bool positive=false;
    for(uint band=0u;band<8u;++band) {
        float d=source.coefficients[band/4u][band%4u];
        float value=shadow.status==0?d*cosine:0.0f;
        if(!sceneFinite(d)||d<0.0f||d>16.0f||!sceneFinite(value)||value<0.0f
            ||(shadow.status==0&&d>0.0f&&cosine>0.0f&&value==0.0f)){results[i]=r;return;}
        r.irradiance[band/4u][band%4u]=value;positive=positive||value>0.0f;
    }
    r.work.z=8u;r.control.y=1u;r.control.z=positive?0u:1u;results[i]=r;
}
)METAL";
}
inline std::string shader(bool rough=false) {
    return std::string("#include <metal_stdlib>\nusing namespace metal;\n")
        +querySource()+mathSource(rough)+"\n#define CAUSTIC_NATIVE_ROUGH "+(rough?"1\n":"0\n")+R"METAL(
struct CHit {
    int status; float distance, position[3], normal[3];
    uint object, material, instance, triangle, entering, work;
};
struct CPath {
    uint work, count, visits, complete, blocked, queries;
    float launched, deposited, absorbed, escaped, unresolved;
    CHit source, entry;
    float padding[21], masked;
    uint roughTransportVersion, interfaceSample, reserved;
};
struct CDeposit {
    uint work, historyLow, historyHigh, interfaces, visible, projected, x, y;
    float fraction, incident, outgoing, absorbed, area, depth, rasterDepth;
    uint object, material, instance, triangle;
    CHit receiver, camera, pixel;
};
struct CParams {
    float4 query, source, camera, rotation, projection;
    float4 ior[2], radiance[2], cie[8];
    uint4 extent, policy; float4 tail; uint4 deposition, allowance;
};
CHit cstore(SceneClosestHit h) {
    CHit r={};r.status=h.status;r.distance=h.distance;
    for(uint a=0;a<3;++a) {r.position[a]=h.position[a];r.normal[a]=h.outwardNormal[a];}
    r.object=h.objectId;r.material=h.materialId;r.instance=h.instanceId;r.triangle=h.triangleId;
    r.entering=h.entering?1u:0u;r.work=h.causticWork;return r;
}
float cbias(float3 p) {return max(1e-5f,2e-6f*max(max(abs(p.x),abs(p.y)),abs(p.z)));}
uint cid(float2 words) {return uint(words.x)|(uint(words.y)<<16u);}
bool csame(SceneClosestHit a,SceneClosestHit b) {
    return a.objectId==b.objectId&&a.materialId==b.materialId&&a.instanceId==b.instanceId&&a.triangleId==b.triangleId;
}
int ctriangle(SceneClosestHit h,device const float4* triangles,uint count) {
    for(uint t=0;t<count;++t) {
        float4 ids=triangles[t*8u+4u],other=triangles[t*8u+5u];
        if(cid(ids.xy)==h.objectId&&cid(ids.zw)==h.materialId&&uint(other.x)==h.instanceId&&uint(other.y)==h.triangleId) return int(t);
    }return -1;
}
struct CBranch {float3 origin,direction;float fraction;ulong history;uint interfaces;bool inside;};
// Each invocation owns one path, its bounded branch stack and deposit slots.
kernel void causticTrace(constant CParams& u [[buffer(0)]],
    device const float4* work [[buffer(1)]],device const float4* triangles [[buffer(2)]],
    device CPath* paths [[buffer(3)]],device CDeposit* deposits [[buffer(4)]],
    device uint* diagnostic [[buffer(6)]],
    texture2d<float> opaqueTexture [[texture(0)]],texture2d<uint> raster [[texture(1)]],
    texture2d<float> rasterDepth [[texture(2)]],uint threadId [[thread_position_in_grid]]) {
    if(threadId>=u.extent.z)return;
    uint w=threadId;
    {
        CPath receipt={};receipt.work=w;
        bool rough=u.tail.z>0.0f;uint interfaceSamples=rough?16u:1u;
        receipt.roughTransportVersion=rough?1u:0u;receipt.interfaceSample=(w/8u)%interfaceSamples;
        if(rough!=bool(CAUSTIC_NATIVE_ROUGH)||u.tail.w!=float(interfaceSamples)) {atomic_store_explicit((device atomic_uint*)diagnostic,15u,memory_order_relaxed);return;}
        float4 position=work[w*4u],normal=work[w*4u+1u],ids=work[w*4u+2u],index=work[w*4u+3u];
        uint band=uint(index.w);float power=normal.w;
        float computed=causticEntryPower(u.radiance[band/4u][band%4u],normal.xyz,u.source.xyz,position.w);
        if(rough)computed/=float(interfaceSamples);
        if(!causticFinite(computed)||abs(power-computed)>2e-4f*max(power,1e-20f)) {atomic_store_explicit((device atomic_uint*)diagnostic,1u,memory_order_relaxed);return;}
        SceneClosestHit source=sceneClosestHit(position.xyz,u.source.xyz,1e-5f,u.source.w,u.query,opaqueTexture);
        receipt.source=cstore(source);receipt.queries=source.causticWork;
        if(source.status<0) {atomic_store_explicit((device atomic_uint*)diagnostic,2u,memory_order_relaxed);return;}
        if(source.status==1) {receipt.blocked=1u;receipt.complete=1u;paths[w]=receipt;return;}
        receipt.launched=power;
        float3 origin=position.xyz+u.source.xyz*(2.0f*cbias(position.xyz)),incoming=-u.source.xyz;
        SceneClosestHit entry=sceneClosestHit(origin,incoming,0.0f,u.source.w,u.query,opaqueTexture);
        receipt.entry=cstore(entry);receipt.queries+=entry.causticWork;
        if(entry.status!=1||!entry.entering||entry.objectId!=cid(ids.xy)||entry.materialId!=cid(ids.zw)
            ||entry.instanceId!=uint(index.x)||entry.triangleId!=uint(index.y)) {atomic_store_explicit((device atomic_uint*)diagnostic,3u,memory_order_relaxed);return;}
        CBranch pending[CAUSTIC_MAX_PENDING];uint size=1u;
        pending[0]={origin,incoming,1.0f,0ul,0u,false};
        while(size>0u&&receipt.visits<u.policy.y) {
            CBranch branch=pending[--size];++receipt.visits;
            SceneClosestHit hit;
            if(branch.interfaces==0u)hit=entry;
            else {hit=sceneClosestHit(branch.origin,branch.direction,1e-5f,u.source.w,u.query,opaqueTexture);receipt.queries+=hit.causticWork;}
            if(hit.status<0) {atomic_store_explicit((device atomic_uint*)diagnostic,4u,memory_order_relaxed);return;}
            if(hit.status==0) {
                if(branch.inside) {atomic_store_explicit((device atomic_uint*)diagnostic,5u,memory_order_relaxed);return;}
                receipt.escaped+=power*branch.fraction;continue;
            }
            int triangle=ctriangle(hit,triangles,u.extent.w);
            if(triangle<0) {atomic_store_explicit((device atomic_uint*)diagnostic,6u,memory_order_relaxed);return;}
            if(hit.objectId!=u.policy.w) {
                if(branch.inside||!hit.entering||branch.interfaces==0u||receipt.count>=u.policy.y) {atomic_store_explicit((device atomic_uint*)diagnostic,7u,memory_order_relaxed);return;}
                CDeposit d={};d.work=w;d.historyLow=uint(branch.history);d.historyHigh=uint(branch.history>>32u);
                d.interfaces=branch.interfaces;d.fraction=branch.fraction;d.incident=power*branch.fraction;
                float rho=triangles[uint(triangle)*8u+6u+band/4u][band%4u];
                float2 split=causticReceiverPower(d.incident,rho);d.outgoing=split.x;d.absorbed=split.y;
                d.receiver=cstore(hit);receipt.deposited+=d.outgoing;receipt.absorbed+=d.absorbed;
                float4 footprint;int projected=causticProjectReceiver(hit.position,hit.outwardNormal,u.camera.xyz,u.rotation,
                    u.projection,float2(u.extent.xy),footprint);
                if(projected<0) {atomic_store_explicit((device atomic_uint*)diagnostic,8u,memory_order_relaxed);return;}
                d.depth=footprint.z;
                if(projected==1) {
                    d.projected=1u;d.x=uint(footprint.x);d.y=uint(footprint.y);d.area=footprint.w;
                    float3 delta=hit.position-u.camera.xyz;float distance=length(delta);
                    SceneClosestHit camera=sceneClosestHit(u.camera.xyz,normalize(delta),0.0f,distance+8.0f*cbias(hit.position),u.query,opaqueTexture);
                    SceneClosestHit pixel=sceneClosestHit(u.camera.xyz,causticCameraRay(footprint.xy+float2(0.5f),u.rotation,u.projection,float2(u.extent.xy)),
                        0.0f,u.projection.w*1e4f,u.query,opaqueTexture);
                    receipt.queries+=camera.causticWork+pixel.causticWork;d.camera=cstore(camera);d.pixel=cstore(pixel);
                    if(camera.status!=1||pixel.status<0) {atomic_store_explicit((device atomic_uint*)diagnostic,9u,memory_order_relaxed);return;}
                    bool visible=csame(hit,camera)&&abs(camera.distance-distance)<=8.0f*cbias(hit.position)
                        &&length(camera.position-hit.position)<=8.0f*cbias(hit.position)&&pixel.status==1&&csame(hit,pixel);
                    if(visible) {
                        uint4 rasterIds=raster.read(uint2(d.x,d.y));d.object=rasterIds.x;d.material=rasterIds.y;
                        d.instance=rasterIds.z;d.triangle=rasterIds.w;d.rasterDepth=rasterDepth.read(uint2(d.x,d.y)).x;
                        d.visible=1u;float value=causticPixelRadiance(d.outgoing,d.area);
                        if(!causticFinite(value)||value<0.0f||value>65504.0f) {atomic_store_explicit((device atomic_uint*)diagnostic,10u,memory_order_relaxed);return;}
                    }
                }
                deposits[w*u.policy.y+receipt.count++]=d;continue;
            }
            if(branch.inside==hit.entering) {atomic_store_explicit((device atomic_uint*)diagnostic,11u,memory_order_relaxed);return;}
            if(branch.interfaces>=u.policy.x) {receipt.unresolved+=power*branch.fraction;continue;}
            float3 n=branch.inside?-hit.outwardNormal:hit.outwardNormal,transmitted,reflected;bool tir;
            float ior=u.ior[band/4u][band%4u];
            float fresnel=0.0f,r,t;float3 interfacePower=float3(0.0f);
            #if CAUSTIC_NATIVE_ROUGH
            if(rough) {
                float2 sample=causticRoughSample(receipt.interfaceSample,branch.interfaces);bool complete;
                causticRoughInterface(branch.direction,n,branch.inside?ior:1.0f,branch.inside?1.0f:ior,
                    u.tail.z,sample.x,sample.y,reflected,transmitted,interfacePower,complete);
                if(!complete||any(interfacePower<float3(0.0f))||abs(interfacePower.x+interfacePower.y+interfacePower.z-1.0f)>2e-4f) {
                    atomic_store_explicit((device atomic_uint*)diagnostic,16u,memory_order_relaxed);return;
                }
                tir=false;r=branch.fraction*interfacePower.x;t=branch.fraction*interfacePower.y;
                receipt.masked+=power*branch.fraction*interfacePower.z;
            } else
            #endif
            {
                fresnel=causticFresnel(branch.direction,n,branch.inside?ior:1.0f,branch.inside?1.0f:ior,transmitted,reflected,tir);
                r=branch.fraction*fresnel;t=branch.fraction-r;
            }
            if(!causticFinite(r)||!causticFinite(t)||r<0.0f||t<0.0f) {atomic_store_explicit((device atomic_uint*)diagnostic,12u,memory_order_relaxed);return;}
            if(rough?((interfacePower.x>0.0f&&r==0.0f)||(interfacePower.y>0.0f&&t==0.0f))
                :((fresnel>0.0f&&r==0.0f)||(fresnel<1.0f&&t==0.0f))) {
                receipt.unresolved+=power*branch.fraction*(rough?interfacePower.x+interfacePower.y:1.0f);continue;
            }
            for(uint child=0;child<2;++child) {
                if(child==1u&&tir)continue;
                float fraction=child==0u?r:t;if(fraction==0.0f)continue;
                if(fraction<u.tail.x||size>=u.policy.z) {receipt.unresolved+=power*fraction;continue;}
                bool reflection=child==0u;
                pending[size++]={hit.position+n*(reflection?cbias(hit.position):-cbias(hit.position)),
                    reflection?reflected:transmitted,fraction,
                    branch.history|(reflection?(1ul<<branch.interfaces):0ul),branch.interfaces+1u,
                    reflection?branch.inside:!branch.inside};
            }
        }
        for(uint p=0;p<size;++p)receipt.unresolved+=power*pending[p].fraction;
        if(receipt.masked+receipt.unresolved>receipt.launched*u.tail.y*(1.0f+2e-4f)) {atomic_store_explicit((device atomic_uint*)diagnostic,13u,memory_order_relaxed);return;}
        receipt.complete=1u;paths[w]=receipt;
    }
}
// The sentinel suffix compacts visible deposits without changing their order.
kernel void causticKeys(constant CParams& u [[buffer(0)]],device const CPath* paths [[buffer(3)]],
    device const CDeposit* deposits [[buffer(4)]],device uint4* keys [[buffer(7)]],
    uint i [[thread_position_in_grid]]) {
    if(i>=u.deposition.x)return;
    uint4 key=uint4(0xffffffffu);
    if(i<u.deposition.y) {
        uint w=i/u.policy.y,slot=i%u.policy.y;
        if(w<u.extent.z&&paths[w].complete==1u&&slot<paths[w].count) {
            CDeposit d=deposits[i];
            if(d.visible==1u)key=uint4(d.y*u.extent.x+d.x,w,slot,i);
        }
    }
    keys[i]=key;
}
bool causticKeyLess(uint4 a,uint4 b) {
    return a.x<b.x||(a.x==b.x&&(a.y<b.y||(a.y==b.y&&(a.z<b.z||(a.z==b.z&&a.w<b.w)))));
}
// One invocation owns both elements of a disjoint compare/exchange pair.
kernel void causticSort(constant CParams& u [[buffer(0)]],device uint4* keys [[buffer(7)]],
    constant uint2& stage [[buffer(8)]],uint pair [[thread_position_in_grid]]) {
    if(pair>=u.deposition.x/2u)return;
    uint i=(pair/stage.y)*(stage.y*2u)+pair%stage.y,j=i+stage.y;
    uint4 a=keys[i],b=keys[j];bool ascending=(i&stage.x)==0u;
    if(ascending?causticKeyLess(b,a):causticKeyLess(a,b)) {keys[i]=b;keys[j]=a;}
}
// Only a segment leader writes its pixel. Addition follows work and slot order.
kernel void causticReduce(constant CParams& u [[buffer(0)]],device const float4* work [[buffer(1)]],
    device const CDeposit* deposits [[buffer(4)]],device float* accumulation [[buffer(5)]],
    device uint* diagnostic [[buffer(6)]],device const uint4* keys [[buffer(7)]],
    uint i [[thread_position_in_grid]]) {
    if(i>=u.deposition.x)return;
    uint4 key=keys[i];
    if(key.x==0xffffffffu||(i>0u&&keys[i-1u].x==key.x))return;
    float sum[8]={};
    for(uint at=i;at<u.deposition.x&&keys[at].x==key.x;++at) {
        uint4 current=keys[at];CDeposit d=deposits[current.w];uint band=uint(work[current.y*4u+3u].w);
        float value=causticPixelRadiance(d.outgoing,d.area);sum[band]+=value;
        if(!causticFinite(sum[band])||sum[band]<0.0f||sum[band]>65504.0f) {
            atomic_store_explicit((device atomic_uint*)diagnostic,10u,memory_order_relaxed);return;
        }
    }
    for(uint band=0;band<8;++band)accumulation[key.x*8u+band]=sum[band];
}
// Color is resolved only after the completed receipt passes the shared gate.
kernel void causticResolve(constant CParams& u [[buffer(0)]],device const float* accumulation [[buffer(1)]],
    device uint* diagnostic [[buffer(2)]],texture2d<float,access::read> color [[texture(0)]],
    texture2d<float,access::write> resolved [[texture(1)]],uint2 p [[thread_position_in_grid]]) {
    if(p.x>=u.extent.x||p.y>=u.extent.y)return;
    uint at=(p.y*u.extent.x+p.x)*8u;float3 xyz=float3(0.0f);
    for(uint b=0;b<8;++b)xyz+=causticBandXyz(accumulation[at+b],u.cie[b]);
    float4 base=color.read(p);float3 value=base.xyz+causticXyzToSceneLinear(xyz);
    if(!sceneFinite3(value)||any(abs(value)>float3(65504.0f))) {atomic_store_explicit((device atomic_uint*)diagnostic,14u,memory_order_relaxed);return;}
    resolved.write(float4(value,base.w),p);
}
)METAL";
}
// Camera branch witnesses use scalar storage and actual uint query identities.
// They retain sampled paths only and never resolve a receiving measure or Color.
struct alignas(16) GlassCameraTraceInput {
    std::array<std::uint32_t,4> pixel {},control {};
    glasscaustics::Texel originMinimum {},directionMaximum {};
};
using GlassCameraTraceVisit=std::array<std::array<std::uint32_t,4>,16>;
struct alignas(16) GlassCameraTraceResult {
    GlassCameraTraceInput input;
    glasscaustics::Texel originMinimum {},directionMaximum {};
    std::array<std::uint32_t,4> control {},work {};
    std::array<std::uint32_t,4> parameters {};
    std::array<std::uint32_t,4> padding {};
};
struct alignas(16) GlassCameraTraceParams {
    glasscaustics::Texel query {},camera {},rotation {},projection {};
    std::array<glasscaustics::Texel,2> ior {},wavelength {};
    std::array<std::uint32_t,4> extent {},glass {};
};
static_assert(sizeof(GlassCameraTraceInput)==64&&sizeof(GlassCameraTraceVisit)==256
    &&sizeof(GlassCameraTraceResult)==160&&sizeof(GlassCameraTraceParams)==160);
static_assert(sizeof(GlassCameraTraceInput)==sizeof(glasscaustics::NativeGlassCameraGratingTraceInput)
    &&sizeof(GlassCameraTraceResult)==glasscaustics::kGlassCameraTraceHeaderBytes
    &&sizeof(GlassCameraTraceVisit)==glasscaustics::kGlassCameraTraceVisitBytes);
static_assert(offsetof(GlassCameraTraceResult,originMinimum)==64&&offsetof(GlassCameraTraceResult,directionMaximum)==80
    &&offsetof(GlassCameraTraceResult,control)==96&&offsetof(GlassCameraTraceResult,work)==112
    &&offsetof(GlassCameraTraceResult,parameters)==128&&offsetof(GlassCameraTraceResult,padding)==144);
inline std::string glassCameraGratingTraceShader() {
    return std::string("#include <metal_stdlib>\nusing namespace metal;\n")+querySource()+mathSource()+R"METAL(
#pragma clang fp contract(off)
struct CTInput {uint4 pixel,control;float4 originMinimum,directionMaximum;};
struct CTVisit {uint4 lanes[16];};
struct CTResult {CTInput input;float4 originMinimum,directionMaximum;uint4 control,work,parameters,padding;};
struct CTParams {float4 query,camera,rotation,projection,ior[2],wavelength[2];uint4 extent,glass;};
struct CTWitness {uint4 primitive;float4 geometry[4];};
struct CTBranch {float3 origin,direction;float fraction;uint parent,token,interfaces,history;bool inside;uint ancestry[8];};
static_assert(sizeof(CTInput)==64&&sizeof(CTVisit)==256&&sizeof(CTResult)==160&&sizeof(CTParams)==160,"camera trace ABI");
float ctBias(float3 p){return max(1e-5f,2e-6f*max(max(abs(p.x),abs(p.y)),abs(p.z)));}
kernel void glassCameraGratingTrace(constant CTParams& u [[buffer(0)]],device const CTInput* inputs [[buffer(1)]],
    device const CTWitness* witnesses [[buffer(2)]],device CTResult* results [[buffer(3)]],
    device CTVisit* visits [[buffer(4)]],texture2d<float> query [[texture(0)]],uint index [[thread_position_in_grid]]) {
    if(index>=u.extent.z*8u)return;
    uint sample=index/8u,band=index%8u;CTResult result={};result.input=inputs[sample];
    result.control=uint4(1u,sample,band,0u);float ior=u.ior[band/4u][band%4u];
    result.parameters=as_type<uint4>(float4(ior,u.wavelength[band/4u][band%4u],0.0f,0.0f));result.parameters.w=1u;
    uint2 pixel=result.input.pixel.xy;uint location=result.input.pixel.z;
    float2 offset=location==0u?float2(0.5f):location==1u?float2(0.0f):location==2u?float2(1.0f,0.0f)
        :location==3u?float2(1.0f):float2(0.0f,1.0f);
    float3 initial=causticCameraRay(float2(pixel)+offset,u.rotation,u.projection,float2(u.extent.xy));
    result.originMinimum=float4(u.camera.xyz,1e-5f);result.directionMaximum=float4(initial,u.projection.w);
    if(location>4u||any(pixel>=u.extent.xy)||result.input.pixel.w!=1u||u.glass.z!=1u
        ||any(result.input.control!=uint4(u.extent.xy,0u,0u))||!sceneFinite(ior)||ior<1.0f||ior>3.0f) {
        result.work.y=2u;result.parameters.z=as_type<uint>(1.0f);results[index]=result;return;
    }
    CTBranch pending[16];uint size=1u,count=0u;CTBranch root={};
    root.origin=u.camera.xyz;root.direction=initial;root.fraction=1.0f;root.parent=0xffffffffu;pending[0]=root;
    uint reasons=0u;float unresolved=0.0f;
    while(size>0u&&count<64u) {
        CTBranch branch=pending[--size];CTVisit v={};uint id=count++;
        v.lanes[0]=uint4(id,branch.parent,branch.token,branch.interfaces);
        v.lanes[1]=uint4(branch.inside?1u:0u,branch.history,0u,0u);
        v.lanes[2]=as_type<uint4>(float4(branch.origin,1e-5f));
        v.lanes[3]=as_type<uint4>(float4(branch.direction,branch.fraction));
        v.lanes[11]=as_type<uint4>(float4(0.0f,0.0f,0.0f,ior));
        for(uint a=0u;a<8u;++a)v.lanes[12u+a/4u][a%4u]=branch.ancestry[a];
        SceneClosestHit h=sceneClosestHit(branch.origin,branch.direction,1e-5f,u.projection.w,u.query,query);
        result.work.z+=h.causticWork;
        v.lanes[4]=uint4(uint(h.status),h.objectId,h.materialId,h.instanceId);
        v.lanes[5]=uint4(h.triangleId,h.causticWork,h.entering?1u:0u,0u);
        v.lanes[6]=as_type<uint4>(float4(h.position,h.distance));v.lanes[7]=as_type<uint4>(float4(h.outwardNormal,0.0f));
        if(h.status<0){v.lanes[1].z=4u;reasons=1u;}
        else if(h.status==0){v.lanes[1].z=branch.inside?4u:3u;if(branch.inside)reasons=2u;}
        else if(h.objectId!=u.glass.x) {
            v.lanes[1].z=2u;
            if(branch.inside){v.lanes[1].z=4u;reasons=2u;}
            else {
                int found=-1;for(uint t=0u;t<u.extent.w;++t)
                    if(all(witnesses[t].primitive==uint4(h.objectId,h.materialId,h.instanceId,h.triangleId))){found=int(t);break;}
                if(found>=0) {
                    CTWitness w=witnesses[uint(found)];float3 a=w.geometry[0].xyz,e=w.geometry[1].xyz-a,f=w.geometry[2].xyz-a,p=h.position-a;
                    float ee=dot(e,e),ef=dot(e,f),ff=dot(f,f),det=ee*ff-ef*ef;
                    float b=(ff*dot(p,e)-ef*dot(p,f))/det,c=(ee*dot(p,f)-ef*dot(p,e))/det;
                    float3 bary=float3(1.0f-b-c,b,c);v.lanes[11]=as_type<uint4>(float4(bary,ior));
                    bool strict=h.entering&&sceneFinite(det)&&det>0.0f&&sceneFinite3(bary)&&all(bary>float3(1e-5f))
                        &&all(bary<float3(1.0f-1e-5f))&&length(h.outwardNormal-w.geometry[3].xyz)<=1e-5f;
                    v.lanes[1].z=strict?1u:4u;if(!strict)reasons=2u;
                }
            }
        } else if(h.materialId!=u.glass.y||branch.inside==h.entering||branch.interfaces>=32u) {
            v.lanes[1].z=4u;reasons=branch.interfaces>=32u?4u:2u;
        } else {
            float3 n=branch.inside?-h.outwardNormal:h.outwardNormal,transmitted,reflected;bool tir=false;
            float from=branch.inside?ior:1.0f,to=branch.inside?1.0f:ior;
            float fresnel=causticFresnel(branch.direction,n,from,to,transmitted,reflected,tir);
            float reflectedFraction=branch.fraction*fresnel,transmittedFraction=tir?0.0f:branch.fraction*(1.0f-fresnel);
            v.lanes[1].w=tir?1u:0u;v.lanes[8]=as_type<uint4>(float4(n,fresnel));
            v.lanes[9]=as_type<uint4>(float4(transmitted,transmittedFraction));
            v.lanes[10]=as_type<uint4>(float4(reflected,reflectedFraction));v.lanes[14]=as_type<uint4>(float4(from,to,0.0f,0.0f));
            if(!sceneFinite(fresnel)||fresnel<0.0f||fresnel>1.0f||!sceneFinite3(reflected)||!sceneFinite3(transmitted)
                ||!sceneFinite(reflectedFraction)||!sceneFinite(transmittedFraction)) {v.lanes[1].z=4u;reasons=64u;}
            else if((fresnel>0.0f&&reflectedFraction==0.0f)||(!tir&&fresnel<1.0f&&transmittedFraction==0.0f)) {
                v.lanes[1].z=4u;reasons=32u;
            }
            else {
                uint children=(reflectedFraction>0.0f?1u:0u)+(transmittedFraction>0.0f?1u:0u);
                if(size+children>16u)reasons=8u;
                else for(uint child=0u;child<2u;++child) {
                    bool transmission=child==1u;float fraction=transmission?transmittedFraction:reflectedFraction;
                    if(fraction==0.0f)continue;
                    CTBranch next=branch;next.parent=id;next.token=transmission?65u:1u;next.fraction=fraction;
                    next.interfaces=branch.interfaces+1u;next.history=branch.history|(transmission?0u:(1u<<branch.interfaces));
                    next.direction=transmission?transmitted:reflected;
                    next.origin=h.position+n*(transmission?-ctBias(h.position):ctBias(h.position));
                    next.inside=transmission?!branch.inside:branch.inside;
                    uint word=branch.interfaces/4u,shift=(branch.interfaces%4u)*8u;
                    next.ancestry[word]=(next.ancestry[word]&~(255u<<shift))|(next.token<<shift);pending[size++]=next;
                }
            }
        }
        if(v.lanes[1].z!=0u&&v.lanes[1].z!=1u) {
            v.lanes[1].w=0u;v.lanes[8]=v.lanes[9]=v.lanes[10]=v.lanes[14]=uint4(0u);
            v.lanes[11]=as_type<uint4>(float4(0.0f,0.0f,0.0f,ior));
        }
        visits[index*64u+id]=v;
        if(reasons){unresolved=branch.fraction;break;}
    }
    if(size>0u&&count==64u&&!reasons)reasons=16u;
    for(uint at=0u;at<size;++at)unresolved+=pending[at].fraction;
    result.control.w=count;result.work.x=reasons?0u:1u;result.work.y=reasons;
    result.parameters.z=as_type<uint>(unresolved);results[index]=result;
}
)METAL";
}
// Keep the released v0/v1 source generator intact. V2 changes only its own
// ABI, physical interface walk and branch continuation.
inline std::string multipleScatterShader() {
    auto source=shader(true);
    bool valid=true;
    const auto replace=[&](const std::string& before,const std::string& after) {
        const auto at=source.find(before);
        if(at==std::string::npos || source.find(before,at+before.size())!=std::string::npos) {valid=false;return;}
        source.replace(at,before.size(),after);
    };
    auto math=glasscaustics::nativeMultipleScatterMathSource();
    const auto adapt=[&](const std::string& before,const std::string& after) {
        std::size_t at=0;while((at=math.find(before,at))!=std::string::npos) {math.replace(at,before.size(),after);at+=after.size();}
    };
    adapt("out vec4 directions[GLASS_MS_EXITS]","thread float4 (&directions)[GLASS_MS_EXITS]");
    adapt("out uint orders[GLASS_MS_EXITS]","thread uint (&orders)[GLASS_MS_EXITS]");
    adapt("out uint count","thread uint& count");adapt("out uint events","thread uint& events");
    adapt("out uint draws","thread uint& draws");adapt("out float residual","thread float& residual");
    adapt("out vec3 corner","thread float3& corner");adapt("out vec4 footprint","thread float4& footprint");
    adapt("out vec3 transmitted","thread float3& transmitted");adapt("out vec3 reflected","thread float3& reflected");
    adapt("out bool tir","thread bool& tir");adapt("out vec3 power","thread float3& power");
    adapt("out bool complete","thread bool& complete");
    adapt("vec2","float2");adapt("vec3","float3");adapt("vec4","float4");
    replace(mathSource(true),math);
    replace("float padding[21], masked;","float padding[17], microEvents, microDraws, microResidual, quadrature, masked;");
    replace("CHit receiver, camera, pixel;","CHit receiver, camera, pixel; uchar ancestry[32];");
    replace("struct CBranch {float3 origin,direction;float fraction;ulong history;uint interfaces;bool inside;};",
        "struct CBranch {float3 origin,direction;float fraction;ulong history;uint interfaces;bool inside;uint ancestry[8];uint samplingHistory;};\n"
        "static_assert(sizeof(CBranch)==96,\"caustic v2 pending ABI\");\n"
        "static_assert(sizeof(CPath)==256 && sizeof(CDeposit)==276,\"caustic v2 storage ABI\");");
    replace("receipt.roughTransportVersion=rough?1u:0u;receipt.interfaceSample=(w/8u)%interfaceSamples;",
        "receipt.roughTransportVersion=1u;receipt.interfaceSample=(w/8u)%interfaceSamples;receipt.reserved=2u;");
    replace("uint band=uint(index.w);float power=normal.w;",R"METAL(uint band=uint(index.w);float power=normal.w;
        receipt.quadrature=index.z;
        uint entryKey=causticMsEntryKey(cid(ids.xy),uint(index.x),uint(index.y),uint(index.z));
        if(!rough||u.deposition.w!=2u||u.allowance.y==0u||u.allowance.y>32u||u.allowance.z==0u||u.allowance.z>16u) {
            atomic_store_explicit((device atomic_uint*)diagnostic,17u,memory_order_relaxed);return;
        })METAL");
    replace("pending[0]={origin,incoming,1.0f,0ul,0u,false};",R"METAL(pending[0]={};pending[0].origin=origin;pending[0].direction=incoming;pending[0].fraction=1.0f;
        pending[0].samplingHistory=2166136261u;)METAL");
    replace("d.interfaces=branch.interfaces;d.fraction=branch.fraction;d.incident=power*branch.fraction;",R"METAL(d.interfaces=branch.interfaces;d.fraction=branch.fraction;d.incident=power*branch.fraction;
                for(uint a=0u;a<32u;++a)d.ancestry[a]=uchar((branch.ancestry[a/4u]>>(8u*(a%4u)))&255u);)METAL");
    const auto begin=source.find("            float3 n=branch.inside?-hit.outwardNormal:hit.outwardNormal,transmitted,reflected;bool tir;");
    const auto end=source.find("        }\n        for(uint p=0;p<size;++p)",begin);
    if(begin==std::string::npos||end==std::string::npos)valid=false;
    else source.replace(begin,end-begin,R"METAL(            float3 n=branch.inside?-hit.outwardNormal:hit.outwardNormal;
            float ior=u.ior[band/4u][band%4u];
            float4 exits[GLASS_MS_EXITS];uint orders[GLASS_MS_EXITS],count,events,draws;float residual;
            uint seed=causticMsSeed(entryKey,receipt.interfaceSample,branch.interfaces,branch.samplingHistory);
            int status=causticMsWalk(branch.direction,n,branch.inside?ior:1.0f,branch.inside?1.0f:ior,
                u.tail.z,seed,u.allowance.y,u.allowance.z,as_type<float>(u.deposition.z),exits,orders,count,events,draws,residual);
            if(status!=1||draws!=4u*events||!causticFinite(residual)||residual<0.0f) {
                atomic_store_explicit((device atomic_uint*)diagnostic,18u,memory_order_relaxed);return;
            }
            receipt.microEvents+=float(events);receipt.microDraws+=float(draws);
            if(float(receipt.queries)+receipt.microEvents+receipt.microDraws>float(u.allowance.w)) {
                atomic_store_explicit((device atomic_uint*)diagnostic,19u,memory_order_relaxed);return;
            }
            float unfinished=power*branch.fraction*residual;
            receipt.microResidual+=unfinished;receipt.unresolved+=unfinished;
            for(uint exit=0u;exit<count;++exit) {
                bool transmitted=exits[exit].w<0.0f;float fraction=branch.fraction*abs(exits[exit].w);
                if(!causticFinite(fraction)||fraction<0.0f||!sceneFinite3(exits[exit].xyz)) {
                    atomic_store_explicit((device atomic_uint*)diagnostic,20u,memory_order_relaxed);return;
                }
                if(fraction==0.0f)continue;
                if(fraction<u.tail.x||size>=u.policy.z) {receipt.unresolved+=power*fraction;continue;}
                CBranch child=branch;
                child.origin=hit.position+n*(transmitted?-cbias(hit.position):cbias(hit.position));
                child.direction=exits[exit].xyz;child.fraction=fraction;
                child.history=branch.history|(transmitted?0ul:(1ul<<branch.interfaces));
                uint token=orders[exit]|(transmitted?64u:0u),word=branch.interfaces/4u,shift=8u*(branch.interfaces%4u);
                child.ancestry[word]=(child.ancestry[word]&~(255u<<shift))|(token<<shift);
                child.samplingHistory=causticMsChildHistory(branch.samplingHistory,orders[exit],transmitted);
                child.interfaces=branch.interfaces+1u;child.inside=transmitted?!branch.inside:branch.inside;
                pending[size++]=child;
            }
)METAL");
    replace("receipt.complete=1u;paths[w]=receipt;\n    }",R"METAL(if(float(receipt.queries)+receipt.microEvents+receipt.microDraws>float(u.allowance.w)) {
            atomic_store_explicit((device atomic_uint*)diagnostic,19u,memory_order_relaxed);return;
        }
        receipt.complete=1u;paths[w]=receipt;
    })METAL");
    return valid?source:std::string{};
}
inline bool decodeGratingIncident(const glasscaustics::Texel* p,const glasscaustics::Texel* orders,
    std::uint32_t locator,glasscaustics::NativeGratingIncident& output) {
    using namespace glasscaustics;
    NativeGratingIncident value;
    const auto integer=[](float input,std::uint32_t maximum,std::uint32_t& out){return exactReceiptInteger(input,maximum,out);};
    const auto zero=[](const Texel& input){return input==Texel{};};
    for(unsigned t=0;t<kGratingIncidentStride;++t)for(float f:p[t])if(!std::isfinite(f))return false;
    if(!integer(p[0][0],std::uint32_t(kMaximumEntrySamples*8-1),value.work)
        ||!integer(p[0][1],7,value.band)||!integer(p[0][2],32,value.interfaces)
        ||!integer(p[0][3],3,value.orderCount))return false;
    for(unsigned a=0;a<4;++a) {std::uint32_t word;if(!integer(p[1][a],65535,word)){return false;}value.reflectionHistory|=std::uint64_t(word)<<(16*a);}
    for(unsigned a=0;a<32;++a) {std::uint32_t token;if(!integer(p[2+a/4][a%4],127,token)){return false;}value.ancestry[a]=std::uint8_t(token);}
    for(unsigned a=0;a<3;++a) {value.origin[a]=p[10][a];value.direction[a]=p[11][a];value.hit.position[a]=p[12][a];
        value.hit.outwardNormal[a]=p[15][a];value.barycentric[a]=p[16][a];value.tangent[a]=p[18][a];value.bitangent[a]=p[19][a];value.normal[a]=p[20][a];}
    value.fraction=p[10][3];value.power=p[11][3];value.hit.distance=p[12][3];value.hit.status=1;
    const auto split=[&](float low,float high,std::uint32_t& result) {std::uint32_t l,h;
        if(!integer(low,65535,l)||!integer(high,65535,h))return false;
        result=l|(h<<16);return true;};
    std::uint32_t entering=0,work=0,complete=0;
    if(!split(p[13][0],p[13][1],value.hit.objectId)||!split(p[13][2],p[13][3],value.hit.materialId)
        ||!integer(p[14][0],std::numeric_limits<std::uint32_t>::max(),value.hit.instanceId)
        ||!integer(p[14][1],std::numeric_limits<std::uint32_t>::max(),value.hit.triangleId)
        ||!split(p[14][2],p[14][3],work)||!integer(p[15][3],1,entering)
        ||!integer(p[22][0],kGratingIncidentVersion,value.version)||value.version!=kGratingIncidentVersion
        ||!integer(p[22][1],1,complete)||complete!=1
        ||p[16][3]!=0||p[17][2]!=0||p[17][3]!=0||p[18][3]!=0||p[19][3]!=0||p[20][3]!=0||p[21][3]!=0)return false;
    value.hit.work=work;value.hit.entering=entering!=0;value.complete=true;value.uv={p[17][0],p[17][1]};
    value.dHdu={p[22][2],p[22][3],p[23][0]};value.dHdv={p[23][1],p[23][2],p[23][3]};
    value.substrateReflectedPower=p[21][0];value.absorbedPower=p[21][1];value.unresolvedPower=p[21][2];
    for(unsigned o=0;o<3;++o) {
        const auto* q=orders+o*kGratingOrderStride;
        if(o>=value.orderCount) {for(unsigned t=0;t<4;++t){if(!zero(q[t]))return false;}continue;}
        std::uint32_t owner,encoded;
        if(!integer(q[0][0],std::uint32_t(kMaximumEntrySamples*8*kMaximumBranchVisits-1),owner)||owner!=locator
            ||!integer(q[0][1],2,encoded)||q[0][2]!=0||q[0][3]!=0||!zero(q[2])||!zero(q[3]))return false;
        value.orders[o].order=std::int32_t(encoded)-1;
        for(unsigned a=0;a<3;++a)value.orders[o].direction[a]=q[1][a];
        value.orders[o].power=q[1][3];
        for(float f:q[1])if(!std::isfinite(f))return false;
    }
    output=value;return true;
}
// One packed record for every exact emitted order, including zero energy.
struct DirectTerminalReceiverResult {
    glasscaustics::NativeDirectGratingTerminalReceiverInput input;
    Hit hit;std::array<std::uint32_t,2> padding{};
    std::array<std::uint32_t,4> control{};
    std::array<float,4> pixel{};
    std::array<std::uint32_t,4> raster{};
    std::array<float,4> depth{};
};
static_assert(sizeof(DirectTerminalReceiverResult)==192
    &&offsetof(DirectTerminalReceiverResult,control)==128
    &&offsetof(DirectTerminalReceiverResult,depth)==176);
inline std::string directTerminalReceiverShader() {
    return shader(false)+R"METAL(
struct TRInput {uint4 key;int4 order;float4 originMinimum,directionMaximum;};
struct TRResult {TRInput input;CHit hit;uint padding[2];uint4 control;float4 pixel;uint4 raster;float4 depth;};
static_assert(sizeof(TRInput)==64&&sizeof(TRResult)==192,"terminal receiver ABI");
kernel void directTerminalReceivers(constant CParams& u [[buffer(0)]],
    device const TRInput* inputs [[buffer(1)]],device const float4* triangles [[buffer(2)]],
    device TRResult* output [[buffer(3)]],constant uint4& limits [[buffer(4)]],
    texture2d<float> query [[texture(0)]],texture2d<uint> raster [[texture(1)]],
    texture2d<float> rasterDepth [[texture(2)]],uint i [[thread_position_in_grid]]) {
    if(i>=limits.x)return;
    TRResult r={};r.input=inputs[i];
    SceneClosestHit hit=sceneClosestHit(r.input.originMinimum.xyz,r.input.directionMaximum.xyz,
        r.input.originMinimum.w,r.input.directionMaximum.w,u.query,query);
    r.hit=cstore(hit);
    if(hit.status<0){output[i]=r;return;}
    r.control=uint4(1u,0u,0u,hit.causticWork);
    if(hit.status==1) {
        r.control.y=1u;
        int t=ctriangle(hit,triangles,u.extent.w);
        if(t<0){r.control.x=0u;output[i]=r;return;}
        float4 owner=triangles[uint(t)*8u+5u];
        if(owner.z==0.0f&&owner.w==1.0f&&hit.entering) {
            r.control.y=2u;
            float3 local=causticRotate(float4(-u.rotation.xyz,u.rotation.w),hit.position-u.camera.xyz);
            float depth=-local.z;r.pixel.z=depth;
            if(!isfinite(depth)){r.control.x=0u;output[i]=r;return;}
            if(depth>=u.projection.z&&depth<=u.projection.w) {
                float2 p=float2((local.x/(depth*u.projection.x*u.projection.y)+1.0f)*0.5f*float(u.extent.x),
                    (1.0f-local.y/(depth*u.projection.x))*0.5f*float(u.extent.y));
                if(!all(isfinite(p))){r.control.x=0u;output[i]=r;return;}
                if(all(p>=float2(0.0f))&&p.x<float(u.extent.x)&&p.y<float(u.extent.y)) {
                    p=floor(p);r.pixel.xy=p;
                    if(dot(hit.outwardNormal,causticCameraRay(p+float2(0.5f),u.rotation,u.projection,float2(u.extent.xy)))<-1e-6f) {
                        r.control.y=3u;r.raster=raster.read(uint2(p));
                        if(all(r.raster==uint4(0u)))r.control.z=1u;
                        else {r.control.z=2u;r.depth.x=rasterDepth.read(uint2(p)).x;}
                    }
                }
            }
        }
    }
    output[i]=r;
}
)METAL";
}
inline glasscaustics::DirectGratingTerminalReceiverRow decodeDirectTerminalReceiver(
    const DirectTerminalReceiverResult& value,const glasscaustics::DirectGratingTerminalOrder& emitted) {
    glasscaustics::DirectGratingTerminalReceiverRow row;row.input=value.input;row.emitted=emitted;row.hit=decode(value.hit);
    row.disposition=glasscaustics::DirectGratingTerminalReceiverRow::Disposition(value.control[1]);
    row.rasterSample=glasscaustics::DirectGratingTerminalReceiverRow::RasterSample(value.control[2]);
    row.pixel.projected=value.control[1]==3;row.pixel.x=std::uint32_t(value.pixel[0]);row.pixel.y=std::uint32_t(value.pixel[1]);
    row.pixel.linearDepth=value.pixel[2];row.raster={value.raster[0],value.raster[1],value.raster[2],value.raster[3],value.depth[0]};
    row.queryWork=value.control[3];return row;
}

// Fixed scalar storage shared with the private MSL camera pass.
struct CameraOrder {
    std::uint32_t locator,band,encoded,outcome,x,y,active,complete;
    Hit hit;
    std::uint32_t object,material,instance,triangle;
    float depth,radiance;
};
static_assert(sizeof(CameraOrder)==112 && offsetof(CameraOrder,hit)==32);
inline std::string gratingCameraShader() {
    return R"METAL(
struct GCameraOrder {
    uint locator,band,encoded,outcome,x,y,active,complete;
    CHit hit;
    uint object,material,instance,triangle;
    float depth,radiance;
};
float gratingCameraTriangleAngle(float3 a,float3 b,float3 c) {
    return 2.0f*atan2(abs(dot(a,cross(b,c))),1.0f+dot(a,b)+dot(b,c)+dot(c,a));
}
// One actual exact-order nearest-hit query, independent of angular integration.
kernel void gratingCamera(constant CParams& u [[buffer(0)]],device const float4* work [[buffer(1)]],
    device const float4* triangles [[buffer(2)]],device const float4* incidents [[buffer(9)]],
    device const float4* orders [[buffer(10)]],device uint* diagnostic [[buffer(6)]],
    device GCameraOrder* camera [[buffer(17)]],constant uint4& limits [[buffer(18)]],
    texture2d<float> query [[texture(0)]],texture2d<uint> raster [[texture(1)]],
    texture2d<float> rasterDepth [[texture(2)]],uint index [[thread_position_in_grid]]) {
    if(index>=limits.x)return;
    uint locator=index/3u,orderIndex=index%3u,base=locator*24u;
    GCameraOrder r={};r.locator=locator;r.x=r.y=0xffffffffu;r.hit.status=-1;
    if(incidents[base+22u].y!=1.0f||orderIndex>=uint(incidents[base].w)) {camera[index]=r;return;}
    r.active=1u;r.band=uint(incidents[base].y);r.encoded=uint(orders[index*4u].y);
    float3 direction=orders[index*4u+1u].xyz;
    float3 local=causticRotate(float4(-u.rotation.xyz,u.rotation.w),direction);
    float backward=local.z,imageX=-local.x,imageY=-local.y,halfY=u.projection.x,halfX=halfY*u.projection.y;
    bool inside=backward>0.0f&&imageX>=-halfX*backward&&imageX<halfX*backward
        &&imageY<=halfY*backward&&imageY>-halfY*backward;
    if(!inside) {r.outcome=2u;r.complete=1u;camera[index]=r;return;}
    r.x=uint((imageX/(backward*halfX)+1.0f)*0.5f*float(u.extent.x));
    r.y=uint((1.0f-imageY/(backward*halfY))*0.5f*float(u.extent.y));
    float3 normal=incidents[base+20u].xyz,position=incidents[base+12u].xyz;
    float denominator=dot(-direction,normal),distance=dot(position-u.camera.xyz,normal)/denominator;
    if(r.x>=u.extent.x||r.y>=u.extent.y||denominator>=-1e-6f||!causticFinite(distance)||distance<=0.0f) {
        atomic_store_explicit((device atomic_uint*)diagnostic,41u,memory_order_relaxed);return;
    }
    SceneClosestHit hit=sceneClosestHit(u.camera.xyz,-direction,0.0f,distance+8.0f*cbias(position),u.query,query);
    r.hit=cstore(hit);
    if(hit.status!=1||!hit.entering) {atomic_store_explicit((device atomic_uint*)diagnostic,42u,memory_order_relaxed);return;}
    uint4 ids=raster.read(uint2(r.x,r.y));r.object=ids.x;r.material=ids.y;r.instance=ids.z;r.triangle=ids.w;
    r.depth=rasterDepth.read(uint2(r.x,r.y)).x;
    bool visible=hit.objectId==cid(incidents[base+13u].xy)&&hit.materialId==cid(incidents[base+13u].zw)
        &&hit.instanceId==uint(incidents[base+14u].x)&&hit.triangleId==uint(incidents[base+14u].y);
    r.outcome=visible?0u:1u;
    if(visible) {
        uint w=uint(incidents[base].x);float4 owner=work[w*4u+2u],identity=work[w*4u+3u];
        int entry=-1;
        for(uint t=0u;t<u.extent.w;++t) {
            uint at=t*8u;float4 ids=triangles[at+4u],other=triangles[at+5u];
            if(cid(ids.xy)==cid(owner.xy)&&cid(ids.zw)==cid(owner.zw)
                &&uint(other.x)==uint(identity.x)&&uint(other.y)==uint(identity.y)) {entry=int(t);break;}
        }
        if(entry<0) {atomic_store_explicit((device atomic_uint*)diagnostic,43u,memory_order_relaxed);return;}
        uint e=uint(entry)*8u;float3 e1=triangles[e+1u].xyz-triangles[e].xyz,e2=triangles[e+2u].xyz-triangles[e].xyz;
        float3 du=float3(incidents[base+22u].zw,incidents[base+23u].x),dv=incidents[base+23u].yzw;
        float area=work[w*4u].w*length(cross(du,dv))/length(cross(e1,e2));
        float cosine=dot(direction,normal);
        float left=halfX*(2.0f*float(r.x)/float(u.extent.x)-1.0f),right=halfX*(2.0f*float(r.x+1u)/float(u.extent.x)-1.0f);
        float top=halfY*(1.0f-2.0f*float(r.y)/float(u.extent.y)),bottom=halfY*(1.0f-2.0f*float(r.y+1u)/float(u.extent.y));
        float3 a=normalize(float3(left,bottom,1.0f)),b=normalize(float3(right,bottom,1.0f));
        float3 c=normalize(float3(right,top,1.0f)),d=normalize(float3(left,top,1.0f));
        float angle=gratingCameraTriangleAngle(a,b,c)+gratingCameraTriangleAngle(a,c,d);
        // Emitted order power already includes the exact order fraction. Convert
        // W/nm to W/(m^2 sr nm) once. This diagnostic is not finite-cell Color.
        r.radiance=orders[index*4u+1u].w/area/cosine/angle;
        if(!causticFinite(area)||area<=0.0f||!causticFinite(angle)||angle<=0.0f||cosine<=0.0f
            ||!causticFinite(r.radiance)||r.radiance<0.0f||r.radiance>65504.0f) {
            atomic_store_explicit((device atomic_uint*)diagnostic,44u,memory_order_relaxed);return;
        }
    }
    r.complete=1u;camera[index]=r;
}
kernel void gratingCameraKeys(constant CParams& u [[buffer(0)]],device const GCameraOrder* camera [[buffer(17)]],
    device uint4* keys [[buffer(7)]],constant uint4& limits [[buffer(18)]],uint i [[thread_position_in_grid]]) {
    if(i>=u.deposition.x)return;uint4 key=uint4(0xffffffffu);
    if(i<limits.x) {GCameraOrder r=camera[i];if(r.active==1u&&r.complete==1u&&r.outcome==0u)
        key=uint4(r.y*u.extent.x+r.x,r.locator,r.encoded,i);}
    keys[i]=key;
}
kernel void gratingCameraReduce(constant CParams& u [[buffer(0)]],device const GCameraOrder* camera [[buffer(17)]],
    device const uint4* keys [[buffer(7)]],device float* accumulation [[buffer(5)]],device uint* diagnostic [[buffer(6)]],
    uint i [[thread_position_in_grid]]) {
    if(i>=u.deposition.x)return;uint4 key=keys[i];
    if(key.x==0xffffffffu||(i>0u&&keys[i-1u].x==key.x))return;
    float sum[8]={};
    for(uint at=i;at<u.deposition.x&&keys[at].x==key.x;++at) {
        GCameraOrder r=camera[keys[at].w];sum[r.band]+=r.radiance;
        if(!causticFinite(sum[r.band])||sum[r.band]>65504.0f) {
            atomic_store_explicit((device atomic_uint*)diagnostic,45u,memory_order_relaxed);return;
        }
    }
    for(uint b=0u;b<8u;++b)accumulation[key.x*8u+b]=sum[b];
}
// A separate private resolve never samples the unrendered base attachment.
kernel void gratingCameraResolve(constant CParams& u [[buffer(0)]],device const float* accumulation [[buffer(5)]],
    device uint* diagnostic [[buffer(6)]],texture2d<float,access::write> resolved [[texture(3)]],
    uint2 pixel [[thread_position_in_grid]]) {
    if(pixel.x>=u.extent.x||pixel.y>=u.extent.y)return;
    uint at=(pixel.y*u.extent.x+pixel.x)*8u;float3 xyz=float3(0.0f);
    for(uint b=0u;b<8u;++b)xyz+=causticBandXyz(accumulation[at+b],u.cie[b]);
    float3 value=causticXyzToSceneLinear(xyz);
    if(!sceneFinite3(value)||any(abs(value)>float3(65504.0f))) {
        atomic_store_explicit((device atomic_uint*)diagnostic,46u,memory_order_relaxed);return;
    }
    resolved.write(float4(value,1.0f),pixel);
}
)METAL";
}
// Supplemental boundary queries consume the original native point buffer.
// The separate seven-texel step record never changes the point24/order4 ABI.
inline std::string gratingCellTransportShader() {
    return R"METAL(
void gratingCellStoreStep(device float4* output,uint at,uint depth,uint token,float3 origin,float3 direction,
    SceneClosestHit hit,float signedBias) {
    output[at]=float4(origin,float(depth));output[at+1u]=float4(direction,float(token));
    output[at+2u]=float4(float(hit.status),hit.distance,hit.entering?1.0f:0.0f,float(hit.causticWork&65535u));
    output[at+3u]=float4(hit.position,float(hit.causticWork>>16u));output[at+4u]=float4(hit.outwardNormal,signedBias);
    output[at+5u]=float4(float(hit.objectId&65535u),float(hit.objectId>>16u),float(hit.materialId&65535u),float(hit.materialId>>16u));
    output[at+6u]=float4(float(hit.instanceId&65535u),float(hit.instanceId>>16u),float(hit.triangleId&65535u),float(hit.triangleId>>16u));
}
bool gratingCellSameOwner(SceneClosestHit hit,device const float4* output,uint at) {
    return hit.status==1&&hit.objectId==cid(output[at+5u].xy)&&hit.materialId==cid(output[at+5u].zw)
        &&hit.instanceId==cid(output[at+6u].xy)&&hit.triangleId==cid(output[at+6u].zw)
        &&hit.entering==(output[at+2u].z==1.0f);
}
kernel void gratingCellUpload(device const float4* cells [[buffer(19)]],device float4* echo [[buffer(20)]],
    constant uint4& limits [[buffer(22)]],uint i [[thread_position_in_grid]]) {
    if(i<limits.z*7u)echo[i]=cells[i];
}
// Center replay first establishes actual interface ownership. Each exact corner
// then follows the same1/65 branch ancestry using its own nearest hits and bias.
kernel void gratingCellTransport(constant CParams& u [[buffer(0)]],device const float4* work [[buffer(1)]],
    device const float4* incidents [[buffer(9)]],device const float4* cells [[buffer(19)]],
    device float4* output [[buffer(21)]],constant uint4& limits [[buffer(22)]],
    device uint* diagnostic [[buffer(23)]],device const uint* locators [[buffer(24)]],
    texture2d<float> query [[texture(0)]],uint index [[thread_position_in_grid]]) {
    if(index>=limits.y)return;
    uint stepCount=limits.x+1u,stride=4u+4u*stepCount*7u,row=index*stride,point=locators[index]*24u;
    for(uint i=0u;i<stride;++i)output[row+i]=float4(0.0f);
    uint w=uint(incidents[point].x),band=uint(incidents[point].y),interfaces=uint(incidents[point].z);
    if(w>=u.extent.z||band>7u||interfaces>limits.x||incidents[point+22u].y!=1.0f) {
        atomic_store_explicit((device atomic_uint*)diagnostic,51u,memory_order_relaxed);return;
    }
    uint entry=uint(work[w*4u+3u].z),cell=entry*7u;
    if(entry>=limits.z||cells[cell+6u].x!=1.0f||cells[cell+5u].w!=float(entry)
        ||!all(cells[cell+4u]==work[w*4u+2u])||cells[cell+5u].x!=work[w*4u+3u].x
        ||cells[cell+5u].y!=work[w*4u+3u].y) {
        atomic_store_explicit((device atomic_uint*)diagnostic,52u,memory_order_relaxed);return;
    }
    output[row]=float4(float(limits.w),float(index),float(w),float(entry));
    uint counts[4]={};uint queries=0u;bool supported=true,centerComplete=false;
    float ior=u.ior[band/4u][band%4u];
    for(uint path=0u;path<4u;++path) {
        float3 position=path==0u?work[w*4u].xyz:cells[cell+path-1u].xyz;
        float3 origin=position+u.source.xyz*(2.0f*cbias(position)),direction=-u.source.xyz;
        bool inside=false,pathComplete=false;float fraction=1.0f;
        for(uint depth=0u;depth<=interfaces;++depth) {
            uint token=depth==interfaces?0u:uint(incidents[point+2u+depth/4u][depth%4u]);
            SceneClosestHit hit=sceneClosestHit(origin,direction,depth==0u?0.0f:1e-5f,u.source.w,u.query,query);
            if(hit.status<0) {atomic_store_explicit((device atomic_uint*)diagnostic,53u,memory_order_relaxed);return;}
            queries+=hit.causticWork;uint at=row+4u+(path*stepCount+depth)*7u;
            float signedBias=0.0f;bool valid=hit.status==1&&inside!=hit.entering;
            if(path>0u)valid=valid&&centerComplete&&gratingCellSameOwner(hit,output,row+4u+depth*7u);
            if(depth==0u)valid=valid&&hit.objectId==cid(cells[cell+4u].xy)&&hit.materialId==cid(cells[cell+4u].zw)
                &&hit.instanceId==uint(cells[cell+5u].x)&&hit.triangleId==uint(cells[cell+5u].y);
            if(depth==interfaces) {
                valid=valid&&!inside&&hit.objectId==cid(incidents[point+13u].xy)&&hit.materialId==cid(incidents[point+13u].zw)
                    &&hit.instanceId==uint(incidents[point+14u].x)&&hit.triangleId==uint(incidents[point+14u].y);
                if(path==0u)valid=valid&&length(hit.position-incidents[point+12u].xyz)<=8.0f*cbias(hit.position)
                    &&length(direction-incidents[point+11u].xyz)<=2e-4f
                    &&abs(fraction-incidents[point+10u].w)<=2e-4f*max(fraction,1e-20f);
                gratingCellStoreStep(output,at,depth,0u,origin,direction,hit,0.0f);++counts[path];
                pathComplete=valid;break;
            }
            valid=valid&&hit.objectId==u.policy.w&&(token==1u||token==65u);
            float3 normal=inside?-hit.outwardNormal:hit.outwardNormal,reflected,transmitted;bool tir=false;
            float fresnel=0.0f,childFraction=0.0f;
            if(valid) {
                fresnel=causticFresnel(direction,normal,inside?ior:1.0f,inside?1.0f:ior,transmitted,reflected,tir);
                childFraction=fraction*(token==1u?fresnel:1.0f-fresnel);
                valid=causticFinite(fresnel)&&causticFinite(childFraction)&&childFraction>0.0f&&!(token==65u&&tir);
                if(valid)signedBias=(inside?-1.0f:1.0f)*(token==1u?1.0f:-1.0f)*cbias(hit.position);
            }
            gratingCellStoreStep(output,at,depth,token,origin,direction,hit,signedBias);++counts[path];
            if(!valid)break;
            origin=hit.position+hit.outwardNormal*signedBias;direction=token==1u?reflected:transmitted;
            fraction=childFraction;if(token==65u)inside=!inside;
        }
        if(path==0u)centerComplete=pathComplete;
        supported=supported&&pathComplete;
    }
    output[row+1u]=float4(float(band),float(interfaces),supported?1.0f:2.0f,1.0f);
    output[row+2u]=float4(float(queries&65535u),float(queries>>16u),0.0f,0.0f);
    output[row+3u]=float4(float(counts[0]),float(counts[1]),float(counts[2]),float(counts[3]));
}
)METAL";
}
inline std::string gratingCellCoverageShader() {
    return R"METAL(
#pragma clang fp contract(off)
struct GCI {float lo,hi;};
struct GCB {GCI a[3];};
struct GCL {uint version,incident,entry,level,code,disposition,complete,sweeps,padding[4];};
struct GCS {
    uint depth,token,object,material,instance,triangle,rule,entering;
    int biasSign;float direction[3],normal[3],originLower[3],originUpper[3],hitLower[3],hitUpper[3];
    float distance[2],bias[2],signedBias[2],baryLower[3],baryUpper[3],rounding;
    uint queryWork,disposition,target,padding[21];
};
struct GCP {uint triangle;float axis[3],swept[2],surface[2];};
static_assert(sizeof(GCL)==48&&sizeof(GCS)==256&&sizeof(GCP)==32,"cell coverage ABI");
bool gcNumber(float x) {return isfinite(x)&&(x==0.0f||abs(x)>=0x1p-126f);}
float gcRound(float x,bool upper,thread bool& ok) {
    if(!gcNumber(x)) {ok=false;return 0.0f;}
    if(x==0.0f)return upper?0x1p-126f:-0x1p-126f;
    uint word=as_type<uint>(x);word+=(upper==(x>0.0f))?1u:uint(-1);
    float value=as_type<float>(word);
    if(!gcNumber(value)) {ok=false;return 0.0f;}return value;
}
GCI gcExact(float x,thread bool& ok) {if(!gcNumber(x))ok=false;return {x,x};}
GCI gcAdd(GCI a,GCI b,thread bool& ok) {
    if(b.lo==0.0f&&b.hi==0.0f)return a;if(a.lo==0.0f&&a.hi==0.0f)return b;
    if(a.lo==a.hi&&b.lo==b.hi&&a.lo==-b.lo)return {0.0f,0.0f};
    float lo=a.lo+b.lo,hi=a.hi+b.hi;
    if((lo==0.0f&&a.lo!=-b.lo)||(hi==0.0f&&a.hi!=-b.hi))ok=false;
    return {gcRound(lo,false,ok),gcRound(hi,true,ok)};
}
GCI gcNeg(GCI a) {return {-a.hi,-a.lo};}
GCI gcSub(GCI a,GCI b,thread bool& ok) {return gcAdd(a,gcNeg(b),ok);}
GCI gcMul(GCI a,GCI b,thread bool& ok) {
    if((a.lo==0.0f&&a.hi==0.0f)||(b.lo==0.0f&&b.hi==0.0f))return {0.0f,0.0f};
    float av[2]={a.lo,a.hi},bv[2]={b.lo,b.hi},lo=INFINITY,hi=-INFINITY;
    for(uint i=0u;i<2u;++i)for(uint j=0u;j<2u;++j) {
        float v=av[i]*bv[j];if(v==0.0f&&av[i]!=0.0f&&bv[j]!=0.0f)ok=false;
        lo=min(lo,gcRound(v,false,ok));hi=max(hi,gcRound(v,true,ok));
    }return {lo,hi};
}
GCI gcDiv(GCI a,GCI b,thread bool& ok) {
    if(b.lo<=0.0f&&b.hi>=0.0f) {ok=false;return {0.0f,0.0f};}
    if(a.lo==0.0f&&a.hi==0.0f)return {0.0f,0.0f};
    float av[2]={a.lo,a.hi},bv[2]={b.lo,b.hi},lo=INFINITY,hi=-INFINITY;
    for(uint i=0u;i<2u;++i)for(uint j=0u;j<2u;++j) {
        float v=av[i]/bv[j];if(v==0.0f&&av[i]!=0.0f)ok=false;
        lo=min(lo,gcRound(v,false,ok));hi=max(hi,gcRound(v,true,ok));
    }return {lo,hi};
}
GCB gcPoint(float3 p,thread bool& ok) {GCB b;for(uint a=0u;a<3u;++a)b.a[a]=gcExact(p[a],ok);return b;}
GCB gcDifference(GCB a,GCB b,thread bool& ok) {for(uint k=0u;k<3u;++k)a.a[k]=gcSub(a.a[k],b.a[k],ok);return a;}
GCI gcDot(GCB a,GCB b,thread bool& ok) {
    GCI r={0.0f,0.0f};for(uint k=0u;k<3u;++k)r=gcAdd(r,gcMul(a.a[k],b.a[k],ok),ok);return r;
}
GCB gcCross(GCB a,GCB b,thread bool& ok) {
    GCB r;for(uint k=0u;k<3u;++k)r.a[k]=gcSub(gcMul(a.a[(k+1u)%3u],b.a[(k+2u)%3u],ok),
        gcMul(a.a[(k+2u)%3u],b.a[(k+1u)%3u],ok),ok);return r;
}
GCB gcAdvance(GCB p,GCB d,GCI t,thread bool& ok) {
    for(uint k=0u;k<3u;++k)p.a[k]=gcAdd(p.a[k],gcMul(d.a[k],t,ok),ok);return p;
}
GCI gcBias(GCB p,thread bool& ok) {
    float lo=0.0f,hi=0.0f;for(uint k=0u;k<3u;++k) {
        GCI v=p.a[k];float lower=v.lo<=0.0f&&v.hi>=0.0f?0.0f:min(abs(v.lo),abs(v.hi));
        lo=max(lo,lower);hi=max(hi,max(abs(v.lo),abs(v.hi)));
    }
    GCI b=gcMul({lo,hi},gcExact(2e-6f,ok),ok);return {max(1e-5f,b.lo),max(1e-5f,b.hi)};
}
GCI gcProjection(GCB p,float3 axis,thread bool& ok) {return gcDot(p,gcPoint(axis,ok),ok);}
// All interval arithmetic uses the actual uploaded endpoints, independently
// rounded operations and exact center-plane lineage. No corner interpolation
// is used as visibility or full-domain proof.
kernel void gratingCellCoverage(constant CParams& u [[buffer(0)]],device const float4* work [[buffer(1)]],
    device const float4* triangles [[buffer(2)]],device const float4* incidents [[buffer(9)]],
    device const float4* cells [[buffer(19)]],device const float4* boundary [[buffer(21)]],
    constant uint4& limits [[buffer(22)]],device uint* diagnostic [[buffer(23)]],
    device const uint* locators [[buffer(24)]],constant uint2& partition [[buffer(25)]],
    device uint* output [[buffer(26)]],uint index [[thread_position_in_grid]]) {
    if(index>=limits.y*limits.w)return;
    uint incident=index/limits.w,code=index%limits.w,slots=limits.x+1u;
    uint sweepWords=64u+(limits.z-1u)*8u,leafWords=12u+slots*sweepWords,base=index*leafWords;
    for(uint j=0u;j<leafWords;++j)output[base+j]=0u;
    uint row=incident*partition.y,point=locators[incident]*24u;
    uint w=uint(incidents[point].x),band=uint(incidents[point].y),interfaces=uint(incidents[point].z),entry=uint(boundary[row].w);
    if(boundary[row].x!=1.0f||boundary[row].y!=float(incident)||boundary[row].z!=float(w)
        ||boundary[row+1u].w!=1.0f||w>=u.extent.z||band>7u||interfaces>limits.x
        ||entry!=uint(work[w*4u+3u].z)||cells[entry*7u+6u].x!=1.0f) {
        atomic_store_explicit((device atomic_uint*)diagnostic,61u,memory_order_relaxed);return;
    }
    uint2 vertex[3]={uint2(0u),uint2(256u,0u),uint2(0u,256u)};
    for(uint depth=0u;depth<partition.x;++depth) {
        uint2 p0=vertex[0],p1=vertex[1],p2=vertex[2],a=(p0+p1)/2u,b=(p1+p2)/2u,c=(p2+p0)/2u;
        uint choice=(code>>(2u*(partition.x-depth-1u)))&3u;
        if(choice==0u) {vertex[0]=p0;vertex[1]=a;vertex[2]=c;}
        else if(choice==1u) {vertex[0]=a;vertex[1]=p1;vertex[2]=b;}
        else if(choice==2u) {vertex[0]=c;vertex[1]=b;vertex[2]=p2;}
        else {vertex[0]=a;vertex[1]=b;vertex[2]=c;}
    }
    bool arithmetic=true;uint cell=entry*7u;GCB p0=gcPoint(cells[cell].xyz,arithmetic);
    GCB e1=gcDifference(gcPoint(cells[cell+1u].xyz,arithmetic),p0,arithmetic);
    GCB e2=gcDifference(gcPoint(cells[cell+2u].xyz,arithmetic),p0,arithmetic),position;
    for(uint k=0u;k<3u;++k)position.a[k]={INFINITY,-INFINITY};
    for(uint v=0u;v<3u;++v) {
        GCB p=gcAdvance(gcAdvance(p0,e1,gcExact(float(vertex[v].x)/256.0f,arithmetic),arithmetic),
            e2,gcExact(float(vertex[v].y)/256.0f,arithmetic),arithmetic);
        for(uint k=0u;k<3u;++k) {position.a[k].lo=min(position.a[k].lo,p.a[k].lo);position.a[k].hi=max(position.a[k].hi,p.a[k].hi);}
    }
    GCB origin=gcAdvance(position,gcPoint(u.source.xyz,arithmetic),gcMul(gcBias(position,arithmetic),{2.0f,2.0f},arithmetic),arithmetic);
    bool continuing=arithmetic&&boundary[row+1u].z==1.0f,allCertified=continuing,inside=false;
    float ior=u.ior[band/4u][band%4u];
    for(uint depth=0u;depth<=interfaces;++depth) {
        device const float4* center=boundary+row+4u+depth*7u;GCS s={};s.depth=depth;s.token=uint(center[1].w);
        s.object=cid(center[5].xy);s.material=cid(center[5].zw);s.instance=cid(center[6].xy);s.triangle=cid(center[6].zw);
        s.rule=1u;s.entering=uint(center[2].z);s.biasSign=depth<interfaces?(center[4].w>0.0f?1:-1):0;s.disposition=2u;s.target=0xffffffffu;
        float3 direction=center[1].xyz,normal=center[4].xyz;
        for(uint k=0u;k<3u;++k) {s.direction[k]=direction[k];s.normal[k]=normal[k];}
        uint at=base+12u+depth*sweepWords;bool ok=continuing;
        if(ok) {
            for(uint t=0u;t<limits.z;++t)if(all(triangles[t*8u+4u]==center[5])
                &&triangles[t*8u+5u].x==float(s.instance)&&triangles[t*8u+5u].y==float(s.triangle)) {s.target=t;break;}
            ok=s.target!=0xffffffffu&&center[2].x==1.0f&&inside!=(s.entering==1u);
        }
        GCB hit={};GCI distance={0.0f,0.0f},bias={0.0f,0.0f},signedBias={0.0f,0.0f},bary[3]={};
        GCB nextOrigin={};float3 nextDirection=float3(0.0f);
        if(ok) {
            uint t=s.target*8u;GCB p=gcPoint(triangles[t].xyz,ok),a=gcDifference(gcPoint(triangles[t+1u].xyz,ok),p,ok);
            GCB b=gcDifference(gcPoint(triangles[t+2u].xyz,ok),p,ok),plane=gcCross(a,b,ok),ray=gcPoint(direction,ok);
            distance=gcDiv(gcDot(gcDifference(p,origin,ok),plane,ok),gcDot(ray,plane,ok),ok);
            ok=ok&&distance.lo>=(depth?1e-5f:0.0f)&&distance.hi<=u.source.w;
            hit=gcAdvance(origin,ray,distance,ok);GCB v=gcDifference(hit,p,ok);
            GCI aa=gcDot(a,a,ok),ab=gcDot(a,b,ok),bb=gcDot(b,b,ok),av=gcDot(a,v,ok),bv=gcDot(b,v,ok);
            GCI denominator=gcSub(gcMul(aa,bb,ok),gcMul(ab,ab,ok),ok);
            bary[1]=gcDiv(gcSub(gcMul(bb,av,ok),gcMul(ab,bv,ok),ok),denominator,ok);
            bary[2]=gcDiv(gcSub(gcMul(aa,bv,ok),gcMul(ab,av,ok),ok),denominator,ok);
            bary[0]=gcSub(gcSub({1.0f,1.0f},bary[1],ok),bary[2],ok);
            for(uint k=0u;k<3u;++k)ok=ok&&bary[k].lo>0.0f&&bary[k].hi<1.0f;
            if(depth<interfaces&&ok) {
                float3 oriented=inside?-normal:normal,transmitted,reflected;bool tir=false;
                float fresnel=causticFresnel(direction,oriented,inside?ior:1.0f,inside?1.0f:ior,transmitted,reflected,tir);
                bool reflect=s.token==1u;nextDirection=reflect?reflected:transmitted;
                ok=causticFinite(fresnel)&&(reflect?fresnel>0.0f:!tir&&fresnel<1.0f)
                    &&(s.token==1u||s.token==65u)&&all(nextDirection==(boundary+row+4u+(depth+1u)*7u)[1].xyz);
                bias=gcBias(hit,ok);signedBias=s.biasSign>0?bias:gcNeg(bias);
                nextOrigin=gcAdvance(hit,gcPoint(normal,ok),signedBias,ok);
            }
        }
        if(ok) {
            float scale=1.0f;GCB swept;
            for(uint k=0u;k<3u;++k) {
                swept.a[k]={min(origin.a[k].lo,hit.a[k].lo),max(origin.a[k].hi,hit.a[k].hi)};
                scale=max(scale,max(abs(swept.a[k].lo),abs(swept.a[k].hi)));
            }
            s.rounding=gcRound(scale*0x1p-21f,true,ok);uint block=0u;
            for(uint t=0u;t<limits.z&&ok;++t) {
                ++s.queryWork;if(t==s.target)continue;
                uint offset=t*8u;float3 axis=float3(0.0f);GCI sweptProjection={0.0f,0.0f},triangleProjection={0.0f,0.0f};bool separated=false;
                // Face-normal axes refine the mapper's xyz-only AABB test.
                // Each projection encloses the entire swept box and triangle.
                for(uint choice=0u;choice<4u&&!separated&&ok;++choice) {
                    axis=choice<3u?float3(choice==0u?1.0f:0.0f,choice==1u?1.0f:0.0f,choice==2u?1.0f:0.0f):triangles[offset+3u].xyz;
                    if(abs(dot(axis,axis)-1.0f)>1e-4f) {ok=false;break;}
                    sweptProjection=gcProjection(swept,axis,ok);triangleProjection={INFINITY,-INFINITY};
                    for(uint v=0u;v<3u;++v) {
                        GCI projected=gcProjection(gcPoint(triangles[offset+v].xyz,ok),axis,ok);
                        triangleProjection.lo=min(triangleProjection.lo,projected.lo);triangleProjection.hi=max(triangleProjection.hi,projected.hi);
                    }
                    GCI gapA=gcSub(triangleProjection,sweptProjection,ok),gapB=gcSub(sweptProjection,triangleProjection,ok);
                    separated=ok&&max(gapA.lo,gapB.lo)>s.rounding;
                }
                if(!separated) {ok=false;break;}
                GCP separation={};separation.triangle=t;for(uint k=0u;k<3u;++k)separation.axis[k]=axis[k];
                separation.swept[0]=sweptProjection.lo;separation.swept[1]=sweptProjection.hi;
                separation.surface[0]=triangleProjection.lo;separation.surface[1]=triangleProjection.hi;
                *((device GCP*)(output+at+64u+block*8u))=separation;++block;
            }
            if(ok) {
                s.disposition=1u;s.distance[0]=distance.lo;s.distance[1]=distance.hi;
                s.bias[0]=bias.lo;s.bias[1]=bias.hi;s.signedBias[0]=signedBias.lo;s.signedBias[1]=signedBias.hi;
                for(uint k=0u;k<3u;++k) {
                    s.originLower[k]=origin.a[k].lo;s.originUpper[k]=origin.a[k].hi;s.hitLower[k]=hit.a[k].lo;s.hitUpper[k]=hit.a[k].hi;
                    s.baryLower[k]=bary[k].lo;s.baryUpper[k]=bary[k].hi;
                }
            }
        }
        if(!ok) {
            // Native Unsupported includes no partial clear certificate.
            for(uint j=0u;j<(limits.z-1u)*8u;++j)output[at+64u+j]=0u;s.rounding=0.0f;
        }
        *((device GCS*)(output+at))=s;continuing=ok;allCertified=allCertified&&ok;
        if(ok&&depth<interfaces) {origin=nextOrigin;if(s.token==65u)inside=!inside;}
    }
    GCL leaf={};leaf.version=1u;leaf.incident=incident;leaf.entry=entry;leaf.level=partition.x;leaf.code=code;
    leaf.disposition=allCertified?1u:2u;leaf.complete=1u;leaf.sweeps=interfaces+1u;*((device GCL*)(output+base))=leaf;
}
)METAL";
}

// Separate coverage ABI matches the shared 48/256/32-byte reservation.
struct CoverageLeafNative {
    std::uint32_t version,incident,entry,level,code,disposition,complete,sweeps,padding[4];
};
struct CoverageSweepNative {
    std::uint32_t depth,token,object,material,instance,triangle,rule,entering;
    std::int32_t biasSign;float direction[3],normal[3],originLower[3],originUpper[3],hitLower[3],hitUpper[3];
    float distance[2],bias[2],signedBias[2],baryLower[3],baryUpper[3],rounding;
    std::uint32_t queryWork,disposition,target,padding[21];
};
struct CoverageBlockerNative {
    std::uint32_t triangle;float axis[3],swept[2],surface[2];
};
static_assert(sizeof(CoverageLeafNative)==48&&sizeof(CoverageSweepNative)==256&&sizeof(CoverageBlockerNative)==32);

struct alignas(16) GratingParams {
    std::array<std::uint32_t,4> control; // witness count, target object/material, slots per work
    std::array<float,4> diffractionGeometry,diffractionMicrostructure,diffractionRoughness;
    std::array<float,4> diffractionSpectral[8];
};
static_assert(sizeof(GratingParams)==192);
// Native math is extracted from the existing Metal material authority by the
// backend. This generator supplies only the adjunct witness and tracing seam.
inline std::string gratingIncidentShader(const std::string& nativeOrderMath) {
    if(nativeOrderMath.empty())return {};
    auto source=shader(false);bool valid=true;
    const auto replace=[&](const std::string& before,const std::string& after) {
        const auto at=source.find(before);
        if(at==std::string::npos||source.find(before,at+before.size())!=std::string::npos) {valid=false;return;}
        source.replace(at,before.size(),after);
    };
    auto differentialMath=glasscaustics::nativeGratingDifferentialMathSource();
    smoothglass::replace(differentialMath,"vec3","float3");
    smoothglass::replace(differentialMath,"out float db","thread float& db");
    smoothglass::replace(differentialMath,"out float3 hitDerivative","thread float3& hitDerivative");
    replace("struct CBranch {float3 origin,direction;float fraction;ulong history;uint interfaces;bool inside;};",
        "#define CAUSTIC_GRATING_VERSION "+std::to_string(glasscaustics::kGratingIncidentVersion)+"\n"+differentialMath+R"METAL(
struct CBranch {float3 origin,direction;float fraction;ulong history;uint interfaces;bool inside;uint ancestry[8];float3 du,dv;};
static_assert(sizeof(CBranch)==128,"grating pending ABI");
struct GParams {uint4 control;float4 diffractionGeometry,diffractionMicrostructure,diffractionRoughness;float4 diffractionSpectral[8];};
)METAL"+nativeOrderMath+R"METAL(
bool gratingEmit(constant CParams& u,constant GParams& g,thread const CBranch& branch,SceneClosestHit hit,uint w,uint band,float power,float3 entryE1,float3 entryE2,float entryArea,
    device const float4* witnesses,device float4* incidents,device float4* orders,
    thread float4 (&counts)[2],thread CPath& receipt,texture2d<float> queryTexture) {
    int found=-1;
    for(uint t=0u;t<g.control.x;++t) {
        float4 ids=witnesses[t*8u+4u],indices=witnesses[t*8u+5u];
        if(cid(ids.xy)==hit.objectId&&cid(ids.zw)==hit.materialId&&uint(indices.x)==hit.instanceId&&uint(indices.y)==hit.triangleId) {found=int(t);break;}
    }
    if(found<0||branch.inside||!hit.entering||branch.interfaces==0u||uint(counts[0].x)>=g.control.w)return false;
    uint t=uint(found)*8u,slot=w*g.control.w+uint(counts[0].x),base=slot*24u;
    float3 a=witnesses[t].xyz,b=witnesses[t+1u].xyz,c=witnesses[t+2u].xyz;
    float3 e=b-a,f=c-a,h=hit.position-a;
    float ee=dot(e,e),ef=dot(e,f),ff=dot(f,f),he=dot(h,e),hf=dot(h,f),denominator=ee*ff-ef*ef;
    if(!causticFinite(denominator)||denominator<=0.0f)return false;
    float bv=(ff*he-ef*hf)/denominator,cv=(ee*hf-ef*he)/denominator;float3 bary=float3(1.0f-bv-cv,bv,cv);
    float2 uv=float2(dot(bary,float3(witnesses[t].w,witnesses[t+1u].w,witnesses[t+2u].w)),dot(bary,witnesses[t+3u].xyz));
    float3 tangent=witnesses[t+6u].xyz,normal=witnesses[t+7u].xyz,bitangent=cross(normal,tangent);
    if(any(bary<=float3(1e-5f))||any(bary>=float3(1.0f-1e-5f))||!sceneFinite3(bary)||dot(branch.direction,normal)>=-1e-6f
        ||!gratingArea(entryE1,entryE2,entryArea,branch.du,branch.dv))return false;
    float3 incident=float3(-dot(branch.direction,tangent),-dot(branch.direction,bitangent),-dot(branch.direction,normal));
    float biasDu,biasDv;
    if(!gratingBiasDerivative(hit.position,branch.du,biasDu)||!gratingBiasDerivative(hit.position,branch.dv,biasDv))return false;
    float P=power*branch.fraction,reflectance=interfaceReflectance(float2(1.0f,0.0f),g.diffractionMicrostructure.yz);
    if(!causticFinite(P)||P<=0.0f||!causticFinite(reflectance))return false;
    float sum=0.0f;uint count=0u;
    for(int order=-1;order<=1;++order) {
        float3 local;float efficiency=gratingNativeEfficiency(order,g.diffractionSpectral[band].x,incident,g,local);
        if(!causticFinite(efficiency)||efficiency<0.0f){return false;}sum+=efficiency;
        float emitted=P*reflectance*efficiency;
        if(!causticFinite(emitted)||emitted<0.0f||(efficiency>0.0f&&reflectance>0.0f&&emitted==0.0f))return false;
        if(emitted==0.0f)continue;
        if(receipt.visits>=u.policy.y){return false;}++receipt.visits;
        float3 direction=local.x*tangent+local.y*bitangent+local.z*normal;
        SceneClosestHit child=sceneClosestHit(hit.position+normal*cbias(hit.position),direction,1e-5f,u.source.w,u.query,queryTexture);
        if(child.status<0){return false;}receipt.queries+=child.causticWork;
        uint extra=uint(counts[0].z)|(uint(counts[0].w)<<16u);extra+=child.causticWork;
        counts[0].z=float(extra&65535u);counts[0].w=float(extra>>16u);
        uint at=(slot*3u+count)*4u;
        orders[at]=float4(float(slot),float(order+1),0.0f,0.0f);orders[at+1u]=float4(direction,emitted);
        orders[at+2u]=float4(0.0f);orders[at+3u]=float4(0.0f);++count;
    }
    if(!causticFinite(sum)||sum>1.0002f)return false;
    float reflected=P*reflectance,absorbed=P-reflected,tail=reflected*(1.0f-min(sum,1.0f));
    for(uint q=0u;q<24u;++q)incidents[base+q]=float4(0.0f);
    incidents[base]=float4(float(w),float(band),float(branch.interfaces),float(count));
    incidents[base+1u]=float4(float(branch.history&65535ul),float((branch.history>>16u)&65535ul),float((branch.history>>32u)&65535ul),float(branch.history>>48u));
    for(uint q=0u;q<8u;++q) {uint word=branch.ancestry[q];incidents[base+2u+q]=float4(float(word&255u),float((word>>8u)&255u),float((word>>16u)&255u),float(word>>24u));}
    incidents[base+10u]=float4(branch.origin,branch.fraction);incidents[base+11u]=float4(branch.direction,P);
    incidents[base+12u]=float4(hit.position,hit.distance);
    incidents[base+13u]=float4(float(hit.objectId&65535u),float(hit.objectId>>16u),float(hit.materialId&65535u),float(hit.materialId>>16u));
    incidents[base+14u]=float4(float(hit.instanceId),float(hit.triangleId),float(hit.causticWork&65535u),float(hit.causticWork>>16u));
    incidents[base+15u]=float4(hit.outwardNormal,1.0f);incidents[base+16u]=float4(bary,0.0f);incidents[base+17u]=float4(uv,0.0f,0.0f);
    incidents[base+18u]=float4(tangent,0.0f);incidents[base+19u]=float4(bitangent,0.0f);incidents[base+20u]=float4(normal,0.0f);
    incidents[base+21u]=float4(reflected,absorbed,tail,0.0f);incidents[base+22u]=float4(float(CAUSTIC_GRATING_VERSION),1.0f,branch.du.xy);incidents[base+23u]=float4(branch.du.z,branch.dv);
    counts[0].x+=1.0f;counts[0].y+=3.0f;counts[1].xy+=float2(absorbed,tail);receipt.unresolved+=tail;
    return true;
}
)METAL");
    replace("device uint* diagnostic [[buffer(6)]],\n    texture2d<float> opaqueTexture","device uint* diagnostic [[buffer(6)]],device const float4* witnesses [[buffer(8)]],device float4* incidents [[buffer(9)]],device float4* orders [[buffer(10)]],device float4* batchCounts [[buffer(11)]],constant GParams& g [[buffer(12)]],\n    texture2d<float> opaqueTexture");
    replace("CPath receipt={};receipt.work=w;","CPath receipt={};receipt.work=w;float4 counts[2]={float4(0.0f),float4(0.0f)};");
    replace("receipt.blocked=1u;receipt.complete=1u;paths[w]=receipt;return;","receipt.blocked=1u;receipt.complete=1u;paths[w]=receipt;counts[1].z=1.0f;batchCounts[w*2u]=counts[0];batchCounts[w*2u+1u]=counts[1];return;");
    replace("pending[0]={origin,incoming,1.0f,0ul,0u,false};",R"METAL(
        int sourceTriangle=ctriangle(entry,triangles,u.extent.w);float dbu,dbv;
        if(sourceTriangle<0) {atomic_store_explicit((device atomic_uint*)diagnostic,31u,memory_order_relaxed);return;}
        uint sourcePlane=uint(sourceTriangle)*8u;
        float3 entryE1=triangles[sourcePlane+1u].xyz-triangles[sourcePlane].xyz,entryE2=triangles[sourcePlane+2u].xyz-triangles[sourcePlane].xyz;
        if(!gratingStrictHit(triangles[sourcePlane].xyz,triangles[sourcePlane+1u].xyz,triangles[sourcePlane+2u].xyz,entry.position,entry.outwardNormal)
            ||!gratingBiasDerivative(position.xyz,entryE1,dbu)||!gratingBiasDerivative(position.xyz,entryE2,dbv)) {
            atomic_store_explicit((device atomic_uint*)diagnostic,31u,memory_order_relaxed);return;
        }
        pending[0]={};pending[0].origin=origin;pending[0].direction=incoming;pending[0].fraction=1.0f;
        pending[0].du=entryE1+2.0f*u.source.xyz*dbu;pending[0].dv=entryE2+2.0f*u.source.xyz*dbv;
)METAL");
    replace("            if(hit.objectId!=u.policy.w) {",R"METAL(            uint plane=uint(triangle)*8u;float3 hitDu,hitDv;
            if(!gratingStrictHit(triangles[plane].xyz,triangles[plane+1u].xyz,triangles[plane+2u].xyz,hit.position,hit.outwardNormal)
                ||!gratingPlaneDerivative(branch.du,branch.direction,hit.outwardNormal,hitDu)
                ||!gratingPlaneDerivative(branch.dv,branch.direction,hit.outwardNormal,hitDv)
                ||!gratingArea(entryE1,entryE2,position.w,hitDu,hitDv)) {
                atomic_store_explicit((device atomic_uint*)diagnostic,31u,memory_order_relaxed);return;
            }
            if(hit.objectId!=u.policy.w) {)METAL");
    replace("            if(hit.objectId!=u.policy.w) {",R"METAL(            if(hit.objectId==g.control.y&&hit.materialId==g.control.z) {
                CBranch measured=branch;measured.du=hitDu;measured.dv=hitDv;
                if(!gratingEmit(u,g,measured,hit,w,band,power,entryE1,entryE2,position.w,witnesses,incidents,orders,counts,receipt,opaqueTexture)) {
                    atomic_store_explicit((device atomic_uint*)diagnostic,30u,memory_order_relaxed);return;
                }
                continue;
            }
            if(hit.objectId!=u.policy.w) {)METAL");
    const auto begin=source.find("                pending[size++]={hit.position+n*(reflection?cbias(hit.position):-cbias(hit.position)),");
    const auto end=source.find("            }\n        }\n        for(uint p=0;p<size;++p)",begin);
    if(begin==std::string::npos||end==std::string::npos)valid=false;
    else source.replace(begin,end-begin,R"METAL(                float dbu,dbv;
                if(!gratingBiasDerivative(hit.position,hitDu,dbu)||!gratingBiasDerivative(hit.position,hitDv,dbv)) {
                    atomic_store_explicit((device atomic_uint*)diagnostic,31u,memory_order_relaxed);return;
                }
                CBranch next=branch;next.origin=hit.position+n*(reflection?cbias(hit.position):-cbias(hit.position));
                float side=reflection?1.0f:-1.0f;next.du=hitDu+side*n*dbu;next.dv=hitDv+side*n*dbv;
                next.direction=reflection?reflected:transmitted;next.fraction=fraction;
                next.history=branch.history|(reflection?(1ul<<branch.interfaces):0ul);
                uint word=branch.interfaces/4u,shift=8u*(branch.interfaces%4u);
                next.ancestry[word]=(next.ancestry[word]&~(255u<<shift))|((reflection?1u:65u)<<shift);
                next.interfaces=branch.interfaces+1u;next.inside=reflection?branch.inside:!branch.inside;pending[size++]=next;
)METAL");
    replace("receipt.complete=1u;paths[w]=receipt;\n    }","receipt.complete=1u;paths[w]=receipt;counts[1].z=1.0f;batchCounts[w*2u]=counts[0];batchCounts[w*2u+1u]=counts[1];\n    }");
    return valid?source:std::string{};
}

inline bool decodeGratingChild(const glasscaustics::Texel* p,std::uint32_t locator,
    glasscaustics::NativeGratingReadback::ChildReceipt& output) {
    using namespace glasscaustics;
    NativeGratingReadback::ChildReceipt value;
    for(unsigned t=0;t<8;++t)for(float f:p[t])if(!std::isfinite(f))return false;
    const auto split=[](float low,float high,std::uint32_t& out){std::uint32_t l,h;
        if(!exactReceiptInteger(low,65535,l)||!exactReceiptInteger(high,65535,h))return false;
        out=l|(h<<16);return true;};
    std::uint32_t encoded,complete,disposition,low,status,entering,hitWork;
    if(!exactReceiptInteger(p[0][0],kMaximumEntrySamples*8*kMaximumBranchVisits,value.incident)||value.incident!=locator
        ||!exactReceiptInteger(p[0][1],2,encoded)||!exactReceiptInteger(p[0][2],1,complete)||complete!=1
        ||!exactReceiptInteger(p[0][3],2,disposition)||!split(p[1][2],p[1][3],low)
        ||!exactReceiptInteger(p[2][0],1,status)||!exactReceiptInteger(p[2][2],1,entering)
        ||!split(p[2][3],p[3][3],hitWork)||p[4][3]!=0||p[7]!=Texel{}
        ||!split(p[5][0],p[5][1],value.hit.objectId)||!split(p[5][2],p[5][3],value.hit.materialId)
        ||!split(p[6][0],p[6][1],value.hit.instanceId)||!split(p[6][2],p[6][3],value.hit.triangleId))return false;
    value.queryWork=low;value.order=int(encoded)-1;value.complete=true;
    value.disposition=NativeGratingReadback::Disposition(disposition);value.escapedPower=p[1][0];value.unresolvedPower=p[1][1];
    value.hit.status=int(status);value.hit.entering=entering!=0;value.hit.work=hitWork;value.hit.distance=p[2][1];
    value.hit.position={p[3][0],p[3][1],p[3][2]};value.hit.outwardNormal={p[4][0],p[4][1],p[4][2]};
    output=value;return true;
}
inline bool decodeCoupledDeposit(const CoupledDeposit& stored,glasscaustics::NativeDeposit& output) {
    using namespace glasscaustics;NativeDeposit value;const auto& n=stored.value;
    if(n.visible>1||n.projected>1||stored.extension[3]!=0||stored.extension[0]>kGratingIncidentVersion)return false;
    value.work=n.work;value.reflectionHistory=std::uint64_t(n.historyLow)|(std::uint64_t(n.historyHigh)<<32);
    value.interfaces=n.interfaces;value.fraction=n.fraction;value.incidentPower=n.incident;value.outgoingPower=n.outgoing;value.absorbedPower=n.absorbed;
    value.receiver=decode(n.receiver);value.cameraHit=decode(n.camera);value.pixelHit=decode(n.pixel);
    value.footprint.projected=n.projected!=0;value.footprint.x=n.x;value.footprint.y=n.y;
    value.footprint.linearDepth=n.depth;value.footprint.worldArea=n.area;
    value.raster={n.object,n.material,n.instance,n.triangle,n.rasterDepth};value.cameraVisible=n.visible!=0;
    std::array<Texel,8> packed{};for(unsigned a=0;a<32;++a)packed[a/4][a%4]=float(stored.ancestry[a]);
    if(!decodeGratingDepositAncestry(packed,value.ancestry)
        ||!decodeGratingDepositExtension({float(stored.extension[0]),float(stored.extension[1]),float(stored.extension[2]),0},{},value))return false;
    output=value;return true;
}
inline std::string gratingTerminalShader(const std::string& nativeOrderMath) {
    auto source=gratingIncidentShader(nativeOrderMath);if(source.empty())return {};
    bool valid=true;const auto replace=[&](const std::string& before,const std::string& after) {
        const auto at=source.find(before);
        if(at==std::string::npos||source.find(before,at+before.size())!=std::string::npos){valid=false;return;}
        source.replace(at,before.size(),after);
    };
    replace("    CHit receiver, camera, pixel;\n};","    CHit receiver, camera, pixel;uint ancestry[8],extension[4];\n};\nstatic_assert(sizeof(CDeposit)==292,\"coupled deposit ABI\");");
    // Reuse the released receiver camera/deposit body. The physical order
    // changes its incident fraction and adds exact parent/order provenance.
    const auto original=shader(false);
    const auto begin=original.find("                CDeposit d={};"),end=original.find("                deposits[w*u.policy.y+receipt.count++]=d;continue;",begin);
    if(begin==std::string::npos||end==std::string::npos)return {};
    auto receiver=original.substr(begin,end-begin);
    smoothglass::replace(receiver,"return;","return false;");
    const auto fraction=receiver.find("d.fraction=branch.fraction;d.incident=power*branch.fraction;");
    if(fraction==std::string::npos)return {};
    receiver.replace(fraction,std::strlen("d.fraction=branch.fraction;d.incident=power*branch.fraction;"),
        "d.fraction=emitted/power;d.incident=emitted;for(uint a=0u;a<8u;++a)d.ancestry[a]=branch.ancestry[a];d.extension[0]=CAUSTIC_GRATING_VERSION;d.extension[1]=slot;d.extension[2]=uint(order+1);");
    const std::string helper=R"METAL(
bool gratingTerminalChild(constant CParams& u,constant GParams& g,thread const CBranch& branch,SceneClosestHit hit,
    uint w,uint band,float power,float emitted,uint slot,int order,
    device const float4* triangles,device CDeposit* deposits,thread CPath& receipt,device uint* diagnostic,
    texture2d<float> opaqueTexture,texture2d<uint> raster,texture2d<float> rasterDepth) {
    int triangle=ctriangle(hit,triangles,u.extent.w);
    if(triangle<0||hit.objectId==u.policy.w||hit.objectId==g.control.y||!hit.entering||receipt.count>=u.policy.y)return false;
)METAL"+receiver+R"METAL(
    deposits[w*u.policy.y+receipt.count++]=d;return true;
}
)METAL";
    replace("bool gratingEmit(",helper+"bool gratingEmit(");
    replace("thread float4 (&counts)[2],thread CPath& receipt,texture2d<float> queryTexture)",
        "thread float4 (&counts)[2],thread CPath& receipt,texture2d<float> queryTexture,device const float4* triangles,device CDeposit* deposits,device float4* children,device uint* diagnostic,texture2d<uint> raster,texture2d<float> rasterDepth)");
    replace("        if(child.status<0)return false;receipt.queries+=child.causticWork;",R"METAL(
        if(child.status<0)return false;receipt.queries+=child.causticWork;
        uint before=receipt.queries,disposition=0u;float escaped=0.0f;
        if(child.status==0){escaped=emitted;receipt.escaped+=emitted;}
        else {disposition=1u;if(!gratingTerminalChild(u,g,branch,child,w,band,power,emitted,slot,order,triangles,deposits,receipt,diagnostic,queryTexture,raster,rasterDepth))return false;}
        uint childWork=child.causticWork+receipt.queries-before,childBase=(slot*3u+count)*8u;
        children[childBase]=float4(float(slot),float(order+1),1.0f,float(disposition));
        children[childBase+1u]=float4(escaped,0.0f,float(childWork&65535u),float(childWork>>16u));
        children[childBase+2u]=float4(float(child.status),child.distance,float(child.entering),float(child.causticWork&65535u));
        children[childBase+3u]=float4(child.position,float(child.causticWork>>16u));children[childBase+4u]=float4(child.outwardNormal,0.0f);
        children[childBase+5u]=float4(float(child.objectId&65535u),float(child.objectId>>16u),float(child.materialId&65535u),float(child.materialId>>16u));
        children[childBase+6u]=float4(float(child.instanceId&65535u),float(child.instanceId>>16u),float(child.triangleId&65535u),float(child.triangleId>>16u));children[childBase+7u]=float4(0.0f);
)METAL");
    replace("extra+=child.causticWork;","extra+=childWork;");
    replace("receipt.unresolved+=tail;","receipt.unresolved+=tail;receipt.absorbed+=absorbed;");
    replace("constant GParams& g [[buffer(12)]],","constant GParams& g [[buffer(12)]],device float4* children [[buffer(13)]],");
    replace("position.w,witnesses,incidents,orders,counts,receipt,opaqueTexture)",
        "position.w,witnesses,incidents,orders,counts,receipt,opaqueTexture,triangles,deposits,children,diagnostic,raster,rasterDepth)");
    return valid?source:std::string{};
}
// The independently authored directional measure has its own native binding.
// The released raw radiance uniform and all path/deposit storage stay intact.
static_assert(glasscaustics::kDirectionalSourceWorldMetres==1
    && std::uint32_t(diffractionmaterial::DirectionalSpectralMeasure::IrradianceCoefficient)==1
    && sizeof(glasscaustics::NativeDirectionalSourceInput)==48
    && sizeof(glasscaustics::NativeDirectionalSourceEcho)==96
    && offsetof(glasscaustics::NativeDirectionalSourceInput,coefficients)==16
    && offsetof(glasscaustics::NativeDirectionalSourceEcho,rawRadiance)==48
    && offsetof(glasscaustics::NativeDirectionalSourceEcho,directionTrace)==80);
inline std::string directionalIrradianceShader(std::string source) {
    bool valid=!source.empty();
    const auto replace=[&](const std::string& before,const std::string& after) {
        const auto at=source.find(before);
        if(at==std::string::npos||source.find(before,at+before.size())!=std::string::npos){valid=false;return;}
        source.replace(at,before.size(),after);
    };
    replace("struct CBranch {",R"METAL(
struct CDirectionalSource {uint4 control;float4 coefficients[2];};
struct CDirectionalEcho {CDirectionalSource source;float4 rawRadiance[2],directionTrace;};
static_assert(sizeof(CDirectionalSource)==48,"directional uniform ABI");
static_assert(sizeof(CDirectionalEcho)==96,"directional echo ABI");
bool causticDirectionalSource(constant CDirectionalSource& d,constant CParams& u,device CDirectionalEcho* echo) {
    if(d.control.x!=1u||d.control.y!=1u||d.control.z!=1u||d.control.w==0u)return false;
    for(uint h=0u;h<2u;++h)if(!all(isfinite(d.coefficients[h]))||any(d.coefficients[h]<0.0f)||any(d.coefficients[h]>16.0f))return false;
    echo->source.control=d.control;
    for(uint h=0u;h<2u;++h){echo->source.coefficients[h]=d.coefficients[h];echo->rawRadiance[h]=u.radiance[h];}
    echo->directionTrace=u.source;return true;
}
kernel void causticDirectionalUpload(constant CParams& u [[buffer(0)]],
    constant CDirectionalSource& d [[buffer(14)]],device CDirectionalEcho* echo [[buffer(15)]],
    uint id [[thread_position_in_grid]]) {
    if(id==0u)causticDirectionalSource(d,u,echo);
}
struct CBranch {)METAL");
    replace("device CPath* paths [[buffer(3)]],device CDeposit* deposits [[buffer(4)]],\n    device uint* diagnostic [[buffer(6)]],",
        "device CPath* paths [[buffer(3)]],device CDeposit* deposits [[buffer(4)]],\n    device uint* diagnostic [[buffer(6)]],constant CDirectionalSource& sourceD [[buffer(14)]],device CDirectionalEcho* sourceEcho [[buffer(15)]],device float* sourcePotential [[buffer(16)]],");
    replace("    if(threadId>=u.extent.z)return;",R"METAL(    if(threadId>=u.extent.z)return;
    if(threadId==0u&&!causticDirectionalSource(sourceD,u,sourceEcho)) {
        atomic_store_explicit((device atomic_uint*)diagnostic,31u,memory_order_relaxed);return;
    })METAL");
    replace("causticEntryPower(u.radiance[band/4u][band%4u],normal.xyz,u.source.xyz,position.w)",
        "causticEntryPower(sourceD.coefficients[band/4u][band%4u],normal.xyz,u.source.xyz,position.w)");
    replace("abs(power-computed)>2e-4f*max(power,1e-20f)",
        "abs(power-computed)>2e-4f*max(power,computed)");
    replace("        if(!causticFinite(computed)||abs(power-computed)",
        "        sourcePotential[w]=computed;\n        if(!causticFinite(computed)||abs(power-computed)");
    return valid?source:std::string{};
}

} // namespace metalcaustic
} // namespace arbitgpu
