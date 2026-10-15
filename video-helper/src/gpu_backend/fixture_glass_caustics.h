#pragma once

#include "fixture_smooth_glass.h"
#include "fixture_diffraction_transport.h"
#include <tuple>

namespace arbitgpu::glasscaustics
{
using Point = opaquevisibility::Point;
using Texel = opaquevisibility::Texel;
inline constexpr std::uint32_t kTableVersion = 2;
inline constexpr std::uint32_t kScheduleVersion = 1;
inline constexpr std::uint32_t kDepositionVersion = 1;
inline constexpr std::uint32_t kGratingCameraReceiptVersion = 1;
inline constexpr std::size_t kGratingCameraOrderNativeBytes = 8*sizeof(opaquevisibility::Texel);
inline constexpr std::uint32_t kRoughTransportVersion = 1;
inline constexpr std::size_t kRoughInterfaceSamples = smoothglass::kRoughInterfaceSamples;
static_assert(kRoughInterfaceSamples==16, "Caustic rough strata require four radical-inverse bits");
inline constexpr std::uint64_t kMaximumDepositionWork = 512ull*1024ull*1024ull;
inline constexpr std::size_t kWorkStride = 4;
inline constexpr std::uint32_t kDirectionalSourceReceiptVersion=1,kDirectionalSourceWorldMetres=1;
inline constexpr std::size_t kDirectionalSourceInputBytes=48,kDirectionalSourceEchoBytes=96;
inline constexpr std::size_t kPathReceiptStride = 16, kDepositReceiptStride = 32;
inline constexpr std::size_t kMicroPathReceiptTexel = 14, kRoughPathReceiptTexel = 15;
inline constexpr std::size_t kAncestryDepositTexel = 22, kMacroAncestryBytes = 32;
// GPU pending ancestry uses eight packed uint32 words, not32 uint tokens.
// Reserve the32 bytes plus seed hash and padding to native16-byte alignment.
inline constexpr std::size_t kMultipleScatterPendingExtensionBytes = 48;
inline constexpr std::uint64_t kAccumulatorBytesPerPixel = 8*sizeof(float);
inline constexpr std::size_t kMaximumSplatsPerDeposit = 1;
inline constexpr std::size_t kTriangleStride = 8;
inline constexpr std::size_t kEntrySubdivisions = 4;
inline constexpr std::size_t kMaximumEntryTriangles = 128;
inline constexpr std::size_t kMaximumEntrySamples = kMaximumEntryTriangles*kEntrySubdivisions*kEntrySubdivisions;
inline constexpr std::size_t kMaximumInterfaces = 32;
inline constexpr std::uint64_t kMaximumQueryWork = 128ull*1024ull*1024ull;
inline constexpr float kRayMinimum = 1.0e-5f;
inline constexpr std::uint32_t kBranchPolicyVersion = 1;
inline constexpr std::size_t kMaximumBranchVisits = smoothglass::kMaximumRayVisits;
inline constexpr std::size_t kMaximumPendingBranches = 16;
// Match the existing smooth-glass maximum accumulated tail. A caustic caller
// must explicitly admit its own policy; this is not an implicit authored value.
// The ordinary scene-source overload binds this exact policy to its immutable
// owner and frame. The bound covers omitted branch flux, not spatial quadrature or
// floating-point error; native acceptance must measure those separately.
inline constexpr float kMaximumUnresolvedFraction = 1.0e-4f;

// Private prerequisite only: version 2 is deliberately absent from Frame and
// consumerCanRender. Both native consumers still require released version 1.
inline constexpr std::uint32_t kPrivateMultipleScatterVersion = smoothglass::kMultipleScatterVersion;
struct MultipleScatterPolicy
{
    std::uint32_t version = kPrivateMultipleScatterVersion;
    std::uint32_t maximumEvents = 32, maximumExits = 16;
    // At most 64 weighted macro visits reserve half the path tail for this
    // walk, leaving half for macro truncation. The final frame gate still owns
    // the measured sum, including any caller-selected different policy.
    float maximumResidual = kMaximumUnresolvedFraction/(2*kMaximumBranchVisits);
};
inline bool validMultipleScatterPolicy(const MultipleScatterPolicy& policy)
{
    return policy.version==kPrivateMultipleScatterVersion && policy.maximumEvents>0
        && policy.maximumEvents<=smoothglass::kMaximumMicrofacetEvents && policy.maximumExits>0
        && policy.maximumExits<=smoothglass::kMaximumMicrofacetExits && std::isfinite(policy.maximumResidual)
        && policy.maximumResidual>0 && policy.maximumResidual<=kMaximumUnresolvedFraction;
}
// Count event evaluations and random draws as additional work in the existing
// 128 Mi work authority. This does not admit queue expansion or a native frame.
inline bool multipleScatterWorkWithinBounds(std::uint64_t queryWork,std::uint64_t interfaces,
    const MultipleScatterPolicy& policy,std::uint64_t& combinedWork)
{
    if(!validMultipleScatterPolicy(policy) || queryWork>kMaximumQueryWork)return false;
    const std::uint64_t perInterface=5ull*policy.maximumEvents;
    if(interfaces>(kMaximumQueryWork-queryWork)/perInterface)return false;
    combinedWork=queryWork+interfaces*perInterface;return true;
}
// Authored discrete path identity only: no time, transform, roughness or IOR.
// quadrature is the stable barycentric sample ordinal within the owned triangle,
// not the frame's entry offset, which can shift when source-facing faces change.
inline std::uint32_t multipleScatterEntryKey(std::uint32_t object,std::uint32_t instance,
    std::uint32_t triangle,std::uint32_t quadrature)
{
    std::uint32_t hash=2166136261u;
    for(auto value:{object,instance,triangle,quadrature})hash=(hash^value)*16777619u;
    return hash;
}
// branchHistory hashes ancestor event orders/medium choices. It must not use
// queue slots or power-dependent visit numbers. This seed is not owner identity.
inline std::uint32_t multipleScatterChildHistory(std::uint32_t parent,std::uint32_t order,bool transmitted)
{return ((parent^order)*16777619u^std::uint32_t(transmitted))*16777619u;}
inline std::uint32_t multipleScatterSeed(std::uint32_t entryKey,std::uint32_t sample,
    std::uint32_t depth,std::uint32_t branchHistory)
{
    std::uint32_t hash=2166136261u;
    for(auto value:{entryKey,sample,depth,branchHistory})hash=(hash^value)*16777619u;
    return hash;
}
inline float multipleScatterRandom(std::uint32_t seed,std::uint32_t dimension)
{
    auto value=seed+0x9e3779b9u*(dimension+1u);
    value=(value^(value>>16))*0x85ebca6bu;value=(value^(value>>13))*0xc2b2ae35u;value^=value>>16;
    return (float(value>>9)+.5f)/8388608.0f;
}
enum class MultipleScatterStatus : int { Invalid=-1, Unresolved=0, Complete=1 };
struct MicrofacetExit
{
    Point direction {};
    float power = 0;
    bool transmitted = false;
    std::uint32_t order = 0;
};
struct MultipleScatterResult
{
    MultipleScatterStatus status = MultipleScatterStatus::Invalid;
    std::array<MicrofacetExit,16> exits {};
    std::uint32_t count = 0, events = 0, draws = 0;
    float residual = 1;
};

struct BranchPolicyDescription
{
    std::uint32_t version = 0;
    float maximumUnresolvedFraction = 0, minimumBranchFraction = 0;
    std::size_t maximumInterfaces = kMaximumInterfaces, maximumVisits = kMaximumBranchVisits;
    std::size_t maximumPending = kMaximumPendingBranches;
};
class AdmittedBranchPolicy final
{
public:
    static std::optional<AdmittedBranchPolicy> admit(const BranchPolicyDescription& source,std::string& error)
    {
        if(source.version!=kBranchPolicyVersion || !std::isfinite(source.maximumUnresolvedFraction)
            || !std::isfinite(source.minimumBranchFraction) || source.maximumUnresolvedFraction<=0
            || source.maximumUnresolvedFraction>kMaximumUnresolvedFraction || source.minimumBranchFraction<=0
            || source.minimumBranchFraction>source.maximumUnresolvedFraction || !source.maximumInterfaces
            || source.maximumInterfaces>kMaximumInterfaces || !source.maximumVisits || source.maximumVisits>kMaximumBranchVisits
            || !source.maximumPending || source.maximumPending>kMaximumPendingBranches)
        {error="Glass caustic branching requires an explicit versioned finite energy error policy";return {};}
        error.clear();return AdmittedBranchPolicy(source);
    }
    const BranchPolicyDescription& description() const noexcept {return description_;}
    std::string identity() const
    {
        std::string out="glass-caustic-branch-policy-v1:";
        const auto add=[&](std::uint64_t value) {
            static constexpr char hex[]="0123456789abcdef";
            for(unsigned d=16;d>0;--d)out.push_back(hex[(value>>((d-1)*4))&15]);
        };
        const auto floating=[&](float value){std::uint32_t raw=0;std::memcpy(&raw,&value,sizeof(raw));add(raw);};
        add(description_.version);floating(description_.maximumUnresolvedFraction);floating(description_.minimumBranchFraction);
        add(description_.maximumInterfaces);add(description_.maximumVisits);add(description_.maximumPending);return out;
    }
private:
    explicit AdmittedBranchPolicy(BranchPolicyDescription source):description_(source) {}
    const BranchPolicyDescription description_;
};

// This contract is deliberately unconnected to native rendering. A backend
// opts in only after tracing these authored triangles itself, conserving the
// per-band power, and refusing unsupported queries or out-of-policy tails.
struct ConsumerContract
{
    NativeFixtureMaterialBackend backend = NativeFixtureMaterialBackend::Invalid;
    std::uint32_t tableVersion = 0, closestHitPayloadVersion = 0, branchPolicyVersion = 0;
    bool authoredSourceBinding = false, nativeDeposition = false, wholeFrameReadback = false;
    std::uint32_t scheduleVersion = 0;
    bool cameraTriangleVisibility = false, perBandEnergyReadback = false, productionCieResolve = false;
    std::uint32_t depositionVersion = 0;
    std::uint32_t roughTransportVersion = 0;
    std::uint32_t multipleScatterVersion = 0;
};
inline bool consumerCanRender(const ConsumerContract& contract, NativeFixtureMaterialBackend backend,float roughness=0,
    std::uint32_t requestedMultipleScatterVersion=0)
{
    return std::isfinite(roughness) && roughness>=0 && roughness<=1
        && (contract.multipleScatterVersion==0 || contract.multipleScatterVersion==kPrivateMultipleScatterVersion)
        && (requestedMultipleScatterVersion==0 || (requestedMultipleScatterVersion==kPrivateMultipleScatterVersion
            && roughness>0 && contract.multipleScatterVersion==kPrivateMultipleScatterVersion))
        && (roughness==0 || contract.roughTransportVersion==kRoughTransportVersion) && backend!=NativeFixtureMaterialBackend::Invalid && contract.backend==backend
        && contract.tableVersion==kTableVersion && contract.closestHitPayloadVersion==3
        && contract.branchPolicyVersion==kBranchPolicyVersion
        && contract.authoredSourceBinding && contract.nativeDeposition && contract.wholeFrameReadback
        && contract.scheduleVersion==kScheduleVersion && contract.cameraTriangleVisibility
        && contract.perBandEnergyReadback && contract.productionCieResolve && contract.depositionVersion==kDepositionVersion;
}
inline bool validateWholeOutput(const std::vector<float>& rgba, std::uint32_t width,
    std::uint32_t height, bool allPathsComplete, std::string& error)
{
    if (!allPathsComplete || !nativeFixtureDimensionsWithinBounds(width,height)
        || rgba.size()!=std::uint64_t(width)*height*4
        || std::any_of(rgba.begin(),rgba.end(),[](float v) {
            return !std::isfinite(v) || std::abs(v)>surfacematerial::kMaximumLinearColorComponent;
        }))
    { error="Glass caustic whole-frame result is incomplete or unrepresentable";return false; }
    error.clear();return true;
}
struct Triangle
{
    std::array<Point,3> points {};
    Point normal {};
    float area = 0;
    std::uint32_t object = 0, material = 0, instance = 0, triangle = 0;
    bool glass = false;
    std::optional<surfacematerial::SpectralReflectanceDescription> reflectance;
};
struct EntrySample
{
    Point position {}, normal {};
    float area = 0;
    std::uint32_t object = 0, material = 0, instance = 0, triangle = 0;
    std::uint32_t quadrature = 0;
};
struct Prepared
{
    smoothglass::Prepared glass;
    // The existing diffraction source admission is accepted explicitly, with
    // its own exact authored owner, separately from an ordinary scene light.
    std::shared_ptr<const NativeFixtureSurfaceMaterialProgram> sourceOwner;
    HarmonicMIDI::grid::SceneLightId sceneLight {};
    std::shared_ptr<const AdmittedBranchPolicy> branchPolicy;
    std::string identity;
};
// Entry quadrature and nearest-pixel deposition are separate discretizations.
// A 1/2/4 refinement comparison must measure power, centroid and footprint error.
// None of these choices certifies spatial convergence from the branch-tail bound.
struct SchedulingDescription
{
    std::uint32_t version = kScheduleVersion;
    std::size_t entrySubdivisions = kEntrySubdivisions;
};
inline bool validScheduling(const SchedulingDescription& value)
{return value.version==kScheduleVersion && (value.entrySubdivisions==1 || value.entrySubdivisions==2 || value.entrySubdivisions==4);}
struct EntryWork
{
    std::uint32_t entry = 0, band = 0;
    float incidentPower = 0;
    std::uint32_t interfaceSample = 0;
    // Only the explicit directional coefficient and its metres-world contract
    // establish W/nm flux.
    // Unspecified retains the legacy numerical power with no new unit grant.
    diffractionmaterial::DirectionalSpectralMeasure sourceMeasure = diffractionmaterial::DirectionalSpectralMeasure::Unspecified;
    std::uint32_t sourceMeasureVersion = 0;
    float directionalCoefficient = 0;
    HarmonicMIDI::grid::SceneLightId sourceLight {};
};
static_assert(sizeof(EntryWork)==32, "Directional source host work accounting changed");
// The native sort operates on uint4 records, never float owner casts. Invalid
// slots sort after every visible pixel. flatIndex retains the trace slot.
struct DepositionKey
{
    std::uint32_t pixel = UINT32_MAX, work = UINT32_MAX, slot = UINT32_MAX, flatIndex = UINT32_MAX;
};
inline bool operator==(const DepositionKey& a,const DepositionKey& b)
{return std::tie(a.pixel,a.work,a.slot,a.flatIndex)==std::tie(b.pixel,b.work,b.slot,b.flatIndex);}
inline bool operator<(const DepositionKey& a,const DepositionKey& b)
{return std::tie(a.pixel,a.work,a.slot,a.flatIndex)<std::tie(b.pixel,b.work,b.slot,b.flatIndex);}
inline bool depositionSentinel(const DepositionKey& value)
{return value==DepositionKey{};}
static_assert(sizeof(DepositionKey)==16);
struct DepositionSchedule
{
    std::uint64_t pixelCount = 0, maximumDeposits = 0, paddedKeyCount = 0, sortPasses = 0;
    std::uint64_t depositionWork = 0, resolveWork = 0, scratchBytes = 0, largestBufferBytes = 0;
    std::uint32_t sortLevels = 0;
};
inline bool operator==(const DepositionSchedule& a,const DepositionSchedule& b)
{
    return std::tie(a.pixelCount,a.maximumDeposits,a.paddedKeyCount,a.sortPasses,a.depositionWork,
        a.resolveWork,a.scratchBytes,a.largestBufferBytes,a.sortLevels)
        ==std::tie(b.pixelCount,b.maximumDeposits,b.paddedKeyCount,b.sortPasses,b.depositionWork,
        b.resolveWork,b.scratchBytes,b.largestBufferBytes,b.sortLevels);
}
inline bool calculateDepositionSchedule(std::uint32_t width,std::uint32_t height,
    std::uint64_t maximumDeposits,DepositionSchedule& output,std::string& error)
{
    const auto refuse=[&](const char* text){error=text;return false;};
    if(!nativeFixtureDimensionsWithinBounds(width,height)
        || maximumDeposits>std::uint64_t(kMaximumEntrySamples)*8*kMaximumBranchVisits)
        return refuse("Glass caustic deposition extent or slot capacity exceeds admission");
    DepositionSchedule candidate;candidate.pixelCount=std::uint64_t(width)*height;
    candidate.maximumDeposits=maximumDeposits;candidate.paddedKeyCount=1;
    if(candidate.pixelCount>=UINT32_MAX)return refuse("Glass caustic projected pixel key cannot be represented");
    while(candidate.paddedKeyCount<std::max<std::uint64_t>(1,maximumDeposits))
    {candidate.paddedKeyCount*=2;++candidate.sortLevels;}
    candidate.sortPasses=std::uint64_t(candidate.sortLevels)*(candidate.sortLevels+1)/2;
    // Each bitonic pass compares N/2 disjoint pairs. Sorting invalid sentinels
    // compacts visible records into a prefix. One segment leader per pixel
    // reduces its prefix in key order; clearing and resolve cover the image.
    candidate.depositionWork=candidate.pixelCount*8+2*candidate.paddedKeyCount
        +(candidate.paddedKeyCount/2)*candidate.sortPasses+maximumDeposits*8;
    candidate.resolveWork=candidate.pixelCount*8;
    if(candidate.depositionWork>kMaximumDepositionWork || candidate.resolveWork>kMaximumDepositionWork)
        return refuse("Glass caustic deposition or resolve work exceeds its separate budget");
    candidate.scratchBytes=2*candidate.paddedKeyCount*sizeof(DepositionKey);
    candidate.largestBufferBytes=std::max({candidate.paddedKeyCount*sizeof(DepositionKey),
        candidate.pixelCount*kAccumulatorBytesPerPixel,maximumDeposits*kDepositReceiptStride*sizeof(Texel)});
    if(candidate.scratchBytes>renderpassoutput::kMaximumTotalBytes
        || candidate.largestBufferBytes>renderpassoutput::kMaximumTotalBytes)
        return refuse("Glass caustic deposition storage exceeds the whole-frame budget");
    output=candidate;error.clear();return true;
}
struct DeviceDepositionLimits
{
    std::uint64_t maximumBufferBytes = 0, maximumDispatchGroups = 0;
    std::uint32_t threadsPerGroup = 128, spectralBandsPerBuffer = 8;
};
inline bool admitDepositionDevice(const DepositionSchedule& schedule,const DeviceDepositionLimits& limits,std::string& error)
{
    if(!schedule.pixelCount || schedule.pixelCount>=UINT32_MAX
        || schedule.maximumDeposits>std::uint64_t(kMaximumEntrySamples)*8*kMaximumBranchVisits
        || !schedule.paddedKeyCount || schedule.paddedKeyCount>std::uint64_t(kMaximumEntrySamples)*8*kMaximumBranchVisits
        || (schedule.paddedKeyCount&(schedule.paddedKeyCount-1)) || schedule.maximumDeposits>schedule.paddedKeyCount
        || (limits.spectralBandsPerBuffer!=4 && limits.spectralBandsPerBuffer!=8))
    {error="Glass caustic native deposition schedule is invalid";return false;}
    const auto groups=schedule.paddedKeyCount/std::max<std::uint32_t>(1,limits.threadsPerGroup)
        +(schedule.paddedKeyCount%std::max<std::uint32_t>(1,limits.threadsPerGroup)!=0);
    const auto largest=std::max({schedule.paddedKeyCount*sizeof(DepositionKey),
        schedule.maximumDeposits*kDepositReceiptStride*sizeof(Texel),
        schedule.pixelCount*limits.spectralBandsPerBuffer*sizeof(float)});
    if(!limits.maximumBufferBytes || !limits.maximumDispatchGroups || !limits.threadsPerGroup
        || (limits.spectralBandsPerBuffer!=4 && limits.spectralBandsPerBuffer!=8)
        || !schedule.paddedKeyCount || largest>limits.maximumBufferBytes
        || groups>limits.maximumDispatchGroups)
    {error="Glass caustic deposition exceeds actual native buffer or dispatch limits";return false;}
    error.clear();return true;
}
struct ResourceBounds
{
    std::uint64_t queryWork = 0, maximumDeposits = 0, transientBytes = 0;
    DepositionSchedule deposition;
    // Zero for legacy frames. Separate optical work consumes the same query
    // budget; temporary storage reserves every pending ancestry and exit array.
    std::uint64_t combinedWork = 0, microfacetWork = 0, microfacetTemporaryBytes = 0;
};
// An optional adjunct to the released v3 closest-hit payload. No query/upload,
// path or deposit bytes change. Native producers must derive the witness from
// their actual closest hit and these exact frozen authored vertices, never a
// nearby diffraction sample or a second CPU query.
inline constexpr std::uint32_t kGratingIncidentVersion=2;
inline constexpr std::size_t kGratingWitnessStride=8, kGratingIncidentStride=24, kGratingOrderStride=4;
inline constexpr std::size_t kGratingOrdersPerIncident=3;
inline constexpr std::size_t kGratingWorkReceiptStride=2,kGratingChildReceiptStride=8;
// Incident texels0 work/band/interfaces/orderCount;1 history as four16-bit
// words;2..9 exact ancestry bytes;10 origin/fraction;11 propagation/power;
// 12 hit position/distance;13 split object/material;14 instance/triangle/
// split hit.work;15 hit normal/entering;16 barycentric/0;17 UV/0/0;
// 18 tangent/0;19 bitangent/0;20 normal/0;21 reflected/absorbed/unresolved/0;
// 22 version/complete/dHdu.x/dHdu.y;23 dHdu.z/dHdv.xyz. Derivatives
// use the exact entry triangle coordinates X=p0+u*(p1-p0)+v*(p2-p0).
// Order texels0 incidentIndex/order+1/0/0;
// 1 direction/power;2..3 zero. All IDs/history use exact integer witnesses.
// Dense work adjunct texels0 incidentCount/orderWork/queryLo16/queryHi16;
// 1 absorbed/unresolved/complete/0. queryWork is extra child/camera work only.
// Child adjunct texels0 incidentIndex/order+1/complete/disposition;
// 1 escaped/unresolved/queryLo16/queryHi16;2..6 released five-texel Hit;
// 7 zero. Disposition0 escaped,1 receiver,2 omitted/unresolved. An omitted
// child has a zero Hit, status0 and no query; it never relabels a queried hit.
// GL deposit extension texel30 version/incidentIndex/order+1/0,31 zero.
// Metal coupled deposit retains244-byte prefix +32 ancestry +16 extension
// =292 bytes. Existing512-byte per-deposit reservation covers it; legacy244
// and private v2276 layouts stay intact.
// The incidentIndex locates a record. Exact parent work/depth/history/ancestry
// and signed order prove identity; no locator or sampling hash replaces them.
struct GratingTriangleWitness
{
    std::uint32_t object=0,material=0,instance=0,triangle=0;
    std::array<Point,3> positions{};
    std::array<std::array<float,2>,3> uv{};
    Point tangent{},bitangent{},normal{};
};
struct GratingIncidentBounds
{
    std::uint64_t maximumIncidents=0,maximumOrders=0,queryWork=0,orderWork=0,combinedWork=0,bytes=0;
};
struct GratingIncidentBinding;
struct NativeGratingReadback;
struct Frame
{
    opaquevisibility::Frame query;
    std::vector<opaquevisibility::Instance> instances;
    std::vector<Triangle> triangles;
    std::vector<EntrySample> entries;
    std::vector<Texel> triangleTexels;
    std::array<float,4> sourceDirectionTrace {};
    std::array<Texel,2> sourceRadiance {};
    diffractionmaterial::DirectionalSpectralIrradiance sourceDirectionalIrradiance {};
    std::array<float,8> ior {};
    float roughness = 0;
    std::uint32_t interfaceSamples = 1;
    std::optional<MultipleScatterPolicy> multipleScatterPolicy;
    SchedulingDescription scheduling;
    std::vector<EntryWork> work;
    // position/area, normal/sourceCoefficient*cosine*area, split object/material IDs,
    // instance/triangle/entry/band. GPU paths start here, never at CPU deposits.
    std::vector<Texel> workTexels;
    HarmonicMIDI::grid::SceneCameraRecord camera;
    std::uint32_t width = 0, height = 0;
    ResourceBounds bounds;
    std::shared_ptr<const HarmonicMIDI::grid::SceneLightRecord> activeSource;
    std::string sourceIdentity, branchPolicyIdentity, cameraIdentity, identity;
    // Private prerequisite only. Existing consumers must refuse this marker
    // until their actual-hit UV witnesses and physical order builders exist.
    std::shared_ptr<const GratingIncidentBinding> gratingIncidentBinding;
    bool gratingStaging=false;
};
struct GratingIncidentBinding
{
    Prepared prepared;
    Frame frame; // Exact immutable staging schedule; contains no binding marker.
    std::shared_ptr<const NativeFixtureSurfaceMaterialProgram> gratingOwner;
    std::shared_ptr<const diffractionmaterial::AdmittedDiffractionMaterialIR> material;
    double evaluationTime=0;
    std::vector<GratingTriangleWitness> triangles;
    std::vector<Texel> witnessTexels;
    GratingIncidentBounds bounds;
    std::string identity;
};
using diffractiontransportnative::sub;
using diffractiontransportnative::dot;
using diffractiontransportnative::cross;
using diffractiontransportnative::finite;
using diffractiontransportnative::normalize;
using diffractiontransportnative::word;
using diffractiontransportnative::bits;
inline Point advance(Point p, Point direction, float distance)
{ for(unsigned a=0;a<3;++a){p[a]+=direction[a]*distance;}return p; }
inline float bias(Point p)
{ return std::max(kRayMinimum,2.0e-6f*std::max({std::abs(p[0]),std::abs(p[1]),std::abs(p[2])})); }

// A local differential of the actual flat-triangle branch. Direction and
// flat interface normal are constant under these entry coordinates. This is
// not a finite-patch ownership proof or an alternate closest-hit traversal.
inline constexpr std::uint64_t kGratingPendingBranchBytes=128;
// Pending array plus four branch copies and six aligned live differential
// vectors. These are logical private scratch reservations, not a claim about
// a particular native compiler's register allocation or spill policy.
inline constexpr std::uint64_t kGratingBranchLocalBytes=4*kGratingPendingBranchBytes+6*sizeof(Texel);
inline bool smoothHitDifferential(Point originDerivative,Point direction,Point normal,Point& output)
{
    const float denominator=dot(normal,direction);
    // Reuse the private grating grazing refusal bound without changing rays.
    if(!finite(originDerivative)||!finite(direction)||!finite(normal)||!std::isfinite(denominator)||std::abs(denominator)<=1e-6f)return false;
    const float dt=-dot(normal,originDerivative)/denominator;
    const auto result=advance(originDerivative,direction,dt);
    if(!std::isfinite(dt)||!finite(result))return false;
    output=result;return true;
}
inline bool smoothBiasDifferential(Point hit,Point derivative,float& output)
{
    if(!finite(hit)||!finite(derivative))return false;
    const float largest=std::max({std::abs(hit[0]),std::abs(hit[1]),std::abs(hit[2])});
    const float scale=2e-6f*largest;
    if(scale==kRayMinimum||!std::isfinite(scale))return false;
    if(scale<kRayMinimum){output=0;return true;}
    unsigned axis=0,count=0;
    for(unsigned a=0;a<3;++a)if(std::abs(hit[a])==largest){axis=a;++count;}
    if(count!=1)return false;
    const float result=2e-6f*(hit[axis]>0?1.0f:-1.0f)*derivative[axis];
    if(!std::isfinite(result))return false;
    output=result;return true;
}
inline bool smoothChildOriginDifferential(Point hit,Point orientedNormal,bool reflected,Point hitDerivative,Point& output)
{
    float db=0;if(!smoothBiasDifferential(hit,hitDerivative,db))return false;
    const auto result=advance(hitDerivative,orientedNormal,reflected?db:-db);
    if(!finite(result))return false;
    output=result;return true;
}
struct GratingReceivingArea
{
    float jacobian=0,area=0; // Square scene units.
    bool squareMetres=false; // Only the explicit authored D v1 world witness.
};
inline bool gratingReceivingArea(Point e1,Point e2,float entryArea,Point dHdu,Point dHdv,
    GratingReceivingArea& output)
{
    if(!finite(e1)||!finite(e2)||!finite(dHdu)||!finite(dHdv)||!std::isfinite(entryArea)||entryArea<=0)return false;
    const auto source=cross(e1,e2),target=cross(dHdu,dHdv);
    const float sourceArea=std::sqrt(dot(source,source)),targetArea=std::sqrt(dot(target,target));
    if(!std::isfinite(sourceArea)||sourceArea<=0||!std::isfinite(targetArea)||targetArea<=0)return false;
    GratingReceivingArea result{targetArea/sourceArea,0};result.area=entryArea*result.jacobian;
    if(!std::isfinite(result.jacobian)||result.jacobian<=0||!std::isfinite(result.area)||result.area<=0)return false;
    output=result;return true;
}
// Original analytic expressions, shared by the two private native generators.
// No additional query or perturbed ray is used. The caller supplies actual
// query-owned triangle positions, hit normal and signed child bias normal.
inline std::string nativeGratingDifferentialMathSource()
{
    return R"GLSL(
bool gratingDifferentialFinite(vec3 p) {return causticFinite(p.x)&&causticFinite(p.y)&&causticFinite(p.z);}
bool gratingStrictHit(vec3 a,vec3 b,vec3 c,vec3 hit,vec3 normal) {
    vec3 e=b-a,f=c-a,h=hit-a;float ee=dot(e,e),ef=dot(e,f),ff=dot(f,f),det=ee*ff-ef*ef;
    if(!causticFinite(det)||det<=0.0)return false;
    float u=(ff*dot(h,e)-ef*dot(h,f))/det,v=(ee*dot(h,f)-ef*dot(h,e))/det;
    vec3 bary=vec3(1.0-u-v,u,v),plane=cross(e,f);float planeLength=length(plane);
    if(!gratingDifferentialFinite(bary)||min(bary.x,min(bary.y,bary.z))<=1e-5
        ||max(bary.x,max(bary.y,bary.z))>=1.0-1e-5||!causticFinite(planeLength)||planeLength<=0.0)return false;
    return length(plane/planeLength-normal)<=2e-4
        &&length(a+e*u+f*v-hit)<=8.0*max(1e-5,2e-6*max(abs(hit.x),max(abs(hit.y),abs(hit.z))));
}
bool gratingBiasDerivative(vec3 hit,vec3 derivative,out float db) {
    if(!gratingDifferentialFinite(hit)||!gratingDifferentialFinite(derivative))return false;
    vec3 a=abs(hit);float largest=max(a.x,max(a.y,a.z)),scale=2e-6*largest;
    if(!causticFinite(scale)||scale==1e-5)return false;
    if(scale<1e-5){db=0.0;return true;}
    int axis=a.x==largest?0:(a.y==largest?1:2);
    if((a.x==largest?1:0)+(a.y==largest?1:0)+(a.z==largest?1:0)!=1)return false;
    db=2e-6*(hit[axis]>0.0?1.0:-1.0)*derivative[axis];return causticFinite(db);
}
bool gratingPlaneDerivative(vec3 originDerivative,vec3 direction,vec3 normal,out vec3 hitDerivative) {
    float denominator=dot(normal,direction);
    if(!gratingDifferentialFinite(originDerivative)||!causticFinite(denominator)||abs(denominator)<=1e-6)return false;
    float dt=-dot(normal,originDerivative)/denominator;hitDerivative=originDerivative+direction*dt;
    return causticFinite(dt)&&gratingDifferentialFinite(hitDerivative);
}
bool gratingArea(vec3 e1,vec3 e2,float entryArea,vec3 du,vec3 dv) {
    float sourceArea=length(cross(e1,e2)),targetArea=length(cross(du,dv));
    if(!causticFinite(sourceArea)||sourceArea<=0.0||!causticFinite(targetArea)||targetArea<=0.0)return false;
    float jacobian=targetArea/sourceArea,area=entryArea*jacobian;
    return causticFinite(jacobian)&&jacobian>0.0&&causticFinite(area)&&area>0.0;
}
)GLSL";
}

// Flux Fresnel, not radiance-throughput eta scaling. Reflected + transmitted
// power equals incident power at each lossless interface. No band averaging.
struct Interface
{
    Point transmitted {}, reflected {};
    float reflection = 0;
    bool tir = false;
};
inline bool interface(Point incoming, Point normal, float from, float to, Interface& output)
{
    if (!finite(incoming) || !finite(normal) || std::abs(dot(incoming,incoming)-1)>1e-4f
        || std::abs(dot(normal,normal)-1)>1e-4f || dot(incoming,normal)>0
        || !std::isfinite(from) || !std::isfinite(to) || from<1 || to<1 || from>3 || to>3) return false;
    Interface candidate;
    const float cosine=std::clamp(-dot(incoming,normal),0.0f,1.0f),ratio=from/to;
    candidate.reflected=advance(incoming,normal,2*cosine);
    if (!normalize(candidate.reflected)) return false;
    // Preserve exact straight-through geometry when the media match.
    if (from==to) {candidate.transmitted=incoming;output=candidate;return true;}
    const float sineSquared=ratio*ratio*std::max(0.0f,1-cosine*cosine);
    candidate.tir=sineSquared>1;
    if (candidate.tir) {candidate.reflection=1;output=candidate;return true;}
    const float exitCosine=std::sqrt(std::max(0.0f,1-sineSquared));
    candidate.transmitted=advance(incoming,normal,(ratio*cosine-exitCosine)/ratio);
    for(auto& v:candidate.transmitted)v*=ratio;
    if (!normalize(candidate.transmitted)) return false;
    const float sDenominator=from*cosine+to*exitCosine,pDenominator=to*cosine+from*exitCosine;
    if (!(sDenominator>0) || !(pDenominator>0)) return false;
    const float s=(from*cosine-to*exitCosine)/sDenominator,p=(to*cosine-from*exitCosine)/pDenominator;
    candidate.reflection=std::clamp(.5f*(s*s+p*p),0.0f,1.0f);
    if (!std::isfinite(candidate.reflection)) return false;
    output=candidate;return true;
}
// The same sixteen visible-normal strata used by camera rough transport.
// Every continued branch keeps its original stratum; depth permutes the set.
inline std::array<float,2> roughSample(unsigned sample,unsigned depth)
{
    const unsigned stratum=(sample*(2*depth+1)+5*depth)%kRoughInterfaceSamples;
    unsigned digits=stratum;float inverse=0,place=.5f;
    for(unsigned bit=0;bit<4;++bit) {inverse+=float(digits&1)*place;digits>>=1;place*=.5f;}
    const float v=inverse+float(depth)*.6180339887498948482f;
    return {(float(stratum)+.5f)/float(kRoughInterfaceSamples),v-std::floor(v)};
}
struct RoughInterface
{
    Interface optics;
    Point facet {};
    double reflected = 0, transmitted = 0, masked = 0;
};
// Diagnostic float32 geometry companion to the released rough dielectric
// kernel. It never supplies native deposits, scratch or product Color.
inline bool roughInterface(Point incoming,Point normal,float from,float to,float roughness,
    unsigned sample,unsigned depth,RoughInterface& output)
{
    if(!std::isfinite(roughness) || roughness<=0 || roughness>1 || sample>=kRoughInterfaceSamples
        || depth>=kMaximumInterfaces || !finite(incoming) || !finite(normal)
        || std::abs(dot(incoming,incoming)-1)>1e-4f || std::abs(dot(normal,normal)-1)>1e-4f
        || dot(incoming,normal)>=0)return false;
    RoughInterface candidate;
    if(from==to) {
        if(!interface(incoming,normal,from,to,candidate.optics))return false;
        candidate.facet=normal;candidate.transmitted=1;output=candidate;return true;
    }
    const float alpha=roughness*roughness;if(!(alpha>0))return false;
    const auto uv=roughSample(sample,depth);
    auto tangent=cross(normal,std::abs(normal[2])<.999f?Point{0,0,1}:Point{1,0,0});
    if(!normalize(tangent))return false;
    const auto bitangent=cross(normal,tangent);
    Point view=incoming;for(auto& v:view)v=-v;
    Point stretched{alpha*dot(view,tangent),alpha*dot(view,bitangent),dot(view,normal)};
    if(!normalize(stretched))return false;
    Point t1=stretched[2]<.99999f?Point{-stretched[1],stretched[0],0}:Point{1,0,0};
    if(!normalize(t1))return false;
    const auto t2=cross(stretched,t1);
    const float radius=std::sqrt(uv[0]),phi=6.283185307179586f*uv[1];
    const float x=radius*std::cos(phi),diskY=radius*std::sin(phi),mix=.5f*(1+stretched[2]);
    const float y=std::sqrt(std::max(0.0f,1-x*x))*(1-mix)+diskY*mix;
    const float z=std::sqrt(std::max(0.0f,1-x*x-y*y));
    Point projected{};for(unsigned a=0;a<3;++a)projected[a]=t1[a]*x+t2[a]*y+stretched[a]*z;
    Point local{alpha*projected[0],alpha*projected[1],std::max(0.0f,projected[2])};
    if(!normalize(local))return false;
    for(unsigned a=0;a<3;++a)candidate.facet[a]=tangent[a]*local[0]+bitangent[a]*local[1]+normal[a]*local[2];
    if(!finite(candidate.facet) || !interface(incoming,candidate.facet,from,to,candidate.optics))return false;
    const auto g1=[&](float cosine) {
        if(cosine<=0)return 0.0f;
        return 2*cosine/(cosine+std::sqrt(alpha*alpha+(1-alpha*alpha)*cosine*cosine));
    };
    candidate.reflected=double(candidate.optics.reflection)*g1(dot(candidate.optics.reflected,normal));
    candidate.transmitted=candidate.optics.tir?0:(1-double(candidate.optics.reflection))*g1(-dot(candidate.optics.transmitted,normal));
    candidate.masked=std::max(0.0,1-candidate.reflected-candidate.transmitted);
    if(!std::isfinite(candidate.masked) || candidate.masked<0 || candidate.masked>1)return false;
    output=candidate;return true;
}
// Float32 diagnostic companion to the private shader. A lossless Smith height
// walk transports unit flux: Fresnel chooses an event, rather than multiplying
// radiance by eta squared. Conditional continuation emits actual exit rays and
// keeps every unescaped/truncated part in residual. It never supplies Color.
inline MultipleScatterResult multipleScatterInterface(Point incoming,Point normal,float from,float to,
    float roughness,std::uint32_t seed,const MultipleScatterPolicy& policy={})
{
    MultipleScatterResult result;
    if(!validMultipleScatterPolicy(policy) || !finite(incoming) || !finite(normal)
        || std::abs(dot(incoming,incoming)-1)>1e-4f || std::abs(dot(normal,normal)-1)>1e-4f
        || dot(incoming,normal)>=0 || !std::isfinite(from) || !std::isfinite(to)
        || from<1 || from>3 || to<1 || to>3 || !std::isfinite(roughness) || roughness<=0 || roughness>1)return result;
    if(from==to) {
        result.exits[0]={incoming,1,true,0};result.count=1;result.residual=0;
        result.status=MultipleScatterStatus::Complete;return result;
    }
    const float alpha=roughness*roughness;if(!(alpha>0))return result;
    const auto lambda=[&](float cosine) {
        if(cosine==0)return 0.0f;
        const float square=alpha*alpha*std::max(0.0f,1-cosine*cosine);
        const float projected=std::sqrt(cosine*cosine+square);
        return cosine>0?square/(2*cosine*(projected+cosine)):-.5f*(1+projected/(-cosine));
    };
    const auto escape=[&](float q,float cosine){return cosine>0?std::pow(q,lambda(cosine)):0.0f;};
    float q=1,weight=1;Point ray=incoming;bool crossed=false;
    for(std::uint32_t event=0;event<policy.maximumEvents;++event)
    {
        Point side=normal;if(crossed)for(auto& c:side)c=-c;
        const float cosine=dot(ray,side),chance=escape(q,cosine);
        const float u=multipleScatterRandom(seed,event*4);++result.draws;
        if(cosine!=0) {
            const float rate=lambda(cosine);if(rate==0)return result;
            q=std::exp(std::log(q)-std::log(1-u*(1-chance))/rate);
        }
        if(!std::isfinite(q) || q<=0 || q>=1)return result;
        auto tangent=cross(side,std::abs(side[2])<.999f?Point{0,0,1}:Point{1,0,0});
        if(!normalize(tangent))return result;
        const auto bitangent=cross(side,tangent);
        Point stretched{-alpha*dot(ray,tangent),-alpha*dot(ray,bitangent),-dot(ray,side)};
        if(!normalize(stretched))return result;
        const float z=1-multipleScatterRandom(seed,event*4+1)*(1+stretched[2]);
        const float radius=std::sqrt(std::max(0.0f,1-z*z));
        const float phi=6.283185307179586f*multipleScatterRandom(seed,event*4+2);
        result.draws+=2;
        Point half{stretched[0]+radius*std::cos(phi),stretched[1]+radius*std::sin(phi),stretched[2]+z};
        if(!normalize(half))return result;
        Point local{alpha*half[0],alpha*half[1],half[2]};if(!normalize(local))return result;
        Point facet{};for(unsigned a=0;a<3;++a)facet[a]=tangent[a]*local[0]+bitangent[a]*local[1]+side[a]*local[2];
        if(!finite(facet) || dot(ray,facet)>=0)return result;
        ++result.events;
        Interface optics;if(!interface(ray,facet,crossed?to:from,crossed?from:to,optics))return result;
        const float choice=multipleScatterRandom(seed,event*4+3);++result.draws;
        if(optics.tir || choice<optics.reflection)ray=optics.reflected;
        else {ray=optics.transmitted;crossed=!crossed;q=1-q;}
        if(!finite(ray) || !std::isfinite(q) || q<=0 || q>=1)return result;
        side=normal;if(crossed)for(auto& c:side)c=-c;
        const float probability=escape(q,dot(ray,side)),emitted=weight*probability;
        if(emitted>0) {
            if(result.count>=policy.maximumExits) {result.status=MultipleScatterStatus::Unresolved;return result;}
            result.exits[result.count++]={ray,emitted,crossed,result.events};
        }
        weight*=1-probability;result.residual=weight;
        if(weight<=policy.maximumResidual) {result.status=MultipleScatterStatus::Complete;return result;}
    }
    result.status=MultipleScatterStatus::Unresolved;return result;
}
inline bool prepareScene(const smoothglass::Prepared& glass, std::string identity,
    Prepared& output, std::string& error)
{
    using namespace HarmonicMIDI::grid;
    const auto refuse=[&](const char* text){error=text;return false;};
    if (!glass.enabled || glass.transport!=surfacematerial::TransportMode::ClosedVolume
        || !glass.sourceScene || !glass.sourceMaterial
        || !validateVisual3DScene(*glass.sourceScene).valid()
        || !validNativeFixtureMaterialPrograms(*glass.sourceMaterial,*glass.sourceScene,glass.sourceMaterial->backend,
            bool(glass.distinctGrating),glass.distinctGrating?NativeFixtureDiffractionConsumer::SmoothGlassDiscreteOrders
                :NativeFixtureDiffractionConsumer::FiniteLobe))
        return refuse("Glass caustics require owning admitted closed-volume geometry");
    Prepared candidate;
    // Re-admit snapshots rather than trusting a caller-supplied query or mutable alias.
    const auto scene=std::make_shared<const Visual3DScene>(*glass.sourceScene);
    const auto root=snapshotNativeSurfaceProgram(glass.sourceMaterial);
    if (!smoothglass::prepare(scene,root,{},candidate.glass,error,glass.roughConsumer,glass.distinctGrating.get())) return false;
    candidate.identity=std::move(identity);
    const auto& sceneRoot=*candidate.glass.sourceMaterial;
    if (!sceneRoot.sceneRevision || !sceneRoot.structuralRevision || !sceneRoot.evaluationRevision)
        return refuse("Glass caustic scene/material revisions are missing");
    word(candidate.identity,scene->id.value);word(candidate.identity,sceneRoot.sceneRevision);
    std::size_t receivers=0;
    for(std::size_t o=0;o<scene->objectCount;++o)
    {
        const auto& object=scene->objects[o];
        if (!object.vertexCount && !object.indexCount) continue;
        const auto* surface=nativeSurfaceForObject(root.get(),object.id);
        const bool grating=candidate.glass.distinctGrating&&object.id==candidate.glass.distinctGrating->object;
        if (!surface || surface->authoredScene!=scene->id || surface->authoredMaterial!=object.material
            || (!grating&&surface->parameters.identifiers[0]!=object.material.value) || surface->sceneRevision!=sceneRoot.sceneRevision
            || surface->structuralRevision!=sceneRoot.structuralRevision
            || surface->evaluationRevision!=sceneRoot.evaluationRevision
            || surface->materialRevision!=surface->structuralRevision)
            return refuse("Glass caustic exact object/material/revision owner is stale or missing");
        const bool volume=object.id==candidate.glass.object;
        if (volume && object.indexCount/3>kMaximumEntryTriangles)
            return refuse("Glass caustic entry triangles exceed their finite sample budget");
        if (volume && surface->spectralReflectance)
            return refuse("Glass caustic volume spectral absorption is unsupported");
        if (!volume&&!grating)
        {
            if (!surface->spectralReflectance || !surfacematerial::validSpectralReflectance(*surface->spectralReflectance))
                return refuse("Glass caustic diffuse receiver requires exact authored eight-band rho");
            if (surface->parameters.emissionRoughness[0]!=0 || surface->parameters.emissionRoughness[1]!=0
                || surface->parameters.emissionRoughness[2]!=0)
                return refuse("Glass caustic receiver emission is unsupported");
            ++receivers;
        }
        word(candidate.identity,object.id.value,8);word(candidate.identity,object.material.value,8);
        word(candidate.identity,surface->structuralRevision);word(candidate.identity,surface->evaluationRevision);
        word(candidate.identity,surface->materialRevision);word(candidate.identity,surface->programIdentity.size());
        candidate.identity+=surface->programIdentity;
        for(const auto* block:{&surface->parameters.baseColorMetallic,&surface->parameters.emissionRoughness,
            &surface->parameters.normalOpacity,&surface->parameters.transmissionIorClearcoat})for(float v:*block)bits(candidate.identity,v);
        if (!volume&&!grating)for(float rho:surface->spectralReflectance->coefficients)bits(candidate.identity,rho);
        if(grating)candidate.identity+=candidate.glass.distinctGrating->material->structuralDigest();
    }
    if (!receivers&&!candidate.glass.distinctGrating) return refuse("Glass caustics require an ordinary authored spectral diffuse receiver");
    output=std::move(candidate);error.clear();return true;
}
inline bool prepare(const smoothglass::Prepared& glass,
    const std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>& sourceOwner,
    Prepared& output, std::string& error)
{
    if (!sourceOwner || !sourceOwner->objectPrograms.empty() || !sourceOwner->diffractionLightingAdmission
        || sourceOwner->diffractionLightingAdmission->description().version!=3
        || !glass.sourceMaterial
        || !validNativeFixtureDiffractionProgram(*sourceOwner,glass.sourceMaterial->backend,sourceOwner->object,true))
    {error="Glass caustics require an explicit existing admitted v3 spectral source owner";return false;}
    Prepared candidate;
    if(!prepareScene(glass,"glass-caustics-v1:"+sourceOwner->programIdentity,candidate,error))return false;
    candidate.sourceOwner=snapshotNativeSurfaceProgram(sourceOwner);
    output=std::move(candidate);error.clear();return true;
}
inline bool prepare(const smoothglass::Prepared& glass,HarmonicMIDI::grid::SceneLightId source,
    const AdmittedBranchPolicy& policy,Prepared& output,std::string& error)
{
    using namespace HarmonicMIDI::grid;
    Prepared candidate;
    if(!prepareScene(glass,"glass-caustics-scene-source-v1:",candidate,error))return false;
    const auto& scene=*candidate.glass.sourceScene;
    const auto* light=visual3d_detail::findById(scene.lights,scene.lightCount,source);
    if(!source.isValid() || !light || light->spectralSource.version!=1 || !validSceneSpectralSource(*light))
    {error="Glass caustics require an exact selected authored directional spectral light";return false;}
    candidate.sceneLight=source;candidate.branchPolicy=std::make_shared<const AdmittedBranchPolicy>(policy);
    // Use the existing scene codec, including SLS1, for exact owning geometry,
    // transforms, light records and independently authored spectral values.
    const auto bytes=sceneaov::detail::encodeScene(scene);
    if(bytes.empty()) {error="Glass caustic owning scene snapshot is invalid";return false;}
    word(candidate.identity,bytes.size());candidate.identity+=sceneaov::detail::hex(bytes);
    word(candidate.identity,source.value,8);candidate.identity+=policy.identity();
    for(std::size_t o=0;o<scene.objectCount;++o)
    {
        const auto* surface=nativeSurfaceForObject(candidate.glass.sourceMaterial.get(),scene.objects[o].id);
        if(!surface)continue; // Transform-only objects have no required program.
        word(candidate.identity,surface->layoutVersion);word(candidate.identity,std::uint32_t(surface->backend));
        word(candidate.identity,surface->bindingDigest.size());candidate.identity+=surface->bindingDigest;
        for(auto id:surface->parameters.identifiers)word(candidate.identity,id,8);
    }
    output=std::move(candidate);error.clear();return true;
}
struct SourceInputs
{
    Point direction {};
    std::array<float,8> radiance {};
    diffractionmaterial::DirectionalSpectralIrradiance directionalIrradiance {};
    HarmonicMIDI::grid::SceneLightId light {};
    float maximum = 0;
    std::string identity;
};
inline bool sourceInputs(const Prepared& prepared,
    const std::shared_ptr<const HarmonicMIDI::grid::SceneLightRecord>& active,
    SourceInputs& output,std::string& error)
{
    using namespace HarmonicMIDI::grid;
    SourceInputs candidate;
    if(prepared.sceneLight.isValid())
    {
        const auto* scene=prepared.glass.sourceScene.get();
        const auto* owner=scene?visual3d_detail::findById(scene->lights,scene->lightCount,prepared.sceneLight):nullptr;
        if(prepared.sourceOwner || !prepared.branchPolicy || !owner || owner->spectralSource.version!=1
            || !active || active->id!=owner->id || !validSceneSpectralSource(*owner)
            || !sameSpectralSource(active->spectralSource,owner->spectralSource)
            || !validFixtureLightOverride(*active) || !validSceneSpectralSource(*active)
            || !std::isfinite(active->innerConeAngle) || !std::isfinite(active->outerConeAngle))
        {error="Glass caustic active source has no exact immutable scene-light owner";return false;}
        const auto direction=sceneSpectralSourceDirection(*active);
        candidate.direction={direction.x,direction.y,direction.z};
        candidate.radiance=owner->spectralSource.radiance;candidate.maximum=owner->spectralSource.traceMaximumDistance;
        candidate.directionalIrradiance=owner->spectralSource.directionalIrradiance;candidate.light=owner->id;
        sceneaov::detail::Writer writer;
        writer.u32(active->id.value);writer.u8(std::uint8_t(active->kind));
        sceneaov::detail::writeTransform(writer,active->transform);sceneaov::detail::writeVec3(writer,active->color);
        for(float value:{active->intensity,active->range,active->innerConeAngle,active->outerConeAngle})writer.f32(value);
        candidate.identity=prepared.identity;word(candidate.identity,writer.bytes.size());
        candidate.identity+=sceneaov::detail::hex(writer.bytes);
    }
    else
    {
        if(active || prepared.branchPolicy || !prepared.sourceOwner || !prepared.sourceOwner->diffractionLightingAdmission)
        {error="Glass caustic path ownership is missing";return false;}
        const auto& source=prepared.sourceOwner->diffractionLightingAdmission->description();
        candidate.direction=source.paths[0].incident.direction;candidate.radiance=source.paths[0].incident.radiance;
        candidate.maximum=source.traceMaximumDistance;candidate.identity=prepared.sourceOwner->programIdentity;
    }
    output=std::move(candidate);error.clear();return true;
}
inline float entrySourceCoefficient(const SourceInputs& source,unsigned band)
{
    return source.directionalIrradiance.version==1 ? source.directionalIrradiance.coefficients[band] : source.radiance[band];
}
// Preparation admits the dimensional schedule, but grants no native consumer.
// Existing consumers recompute potential from Li and must refuse this marker.
enum class SourceMeasureUse : std::uint8_t { LegacyNative = 0, DirectionalIrradiancePreparation = 1 };
// Runtime instances remain the released object-world authority. Existing light
// overrides/animation may change this selected light's transform, color and
// intensity. Spectral Li, D and trace bounds require a new owning authored snapshot.
inline bool selectActiveSource(const Prepared& prepared,const NativeFixtureSceneRuntimeInputs* runtime,
    std::shared_ptr<const HarmonicMIDI::grid::SceneLightRecord>& output,std::string& error)
{
    using namespace HarmonicMIDI::grid;
    if(!prepared.sceneLight.isValid())
    {
        // A legacy diffraction source names no scene light, so no runtime light record
        // drives it. Modulated Environment and animated records stay with the scene's
        // own lights. Malformed records are still refused.
        if(runtime && !validFixtureRuntimeInputs(*runtime))
        {error="Legacy diffraction runtime light inputs are invalid";return false;}
        output.reset();return true;
    }
    if(!prepared.glass.sourceScene)
    {error="Glass caustic active source scene is missing";return false;}
    const auto& owned=*prepared.glass.sourceScene;
    const auto* owner=visual3d_detail::findById(owned.lights,owned.lightCount,prepared.sceneLight);
    if(!owner) {error="Glass caustic selected light identity is missing";return false;}
    auto active=*owner;
    if(runtime)
    {
        if(!validFixtureRuntimeInputs(*runtime))
        {error="Glass caustic runtime light inputs are invalid";return false;}
        auto evaluated=std::make_shared<Visual3DScene>(owned);
        if(runtime->lightOverride)
        {
            if(runtime->lightOverride->id!=owner->id)
            {error="Glass caustic runtime override selects a foreign light";return false;}
            evaluated->lights[std::size_t(owner-owned.lights.data())]=*runtime->lightOverride;
        }
        else
        {
            // Same record selection as the fixture readers, so modulated overrides reach the caustic
            // whether the frame carries animated records or reads the scene's own lights.
            std::string targetError;
            if(!validNativeImportedLightOverrideTargets(owned,*runtime,targetError))
            {error=targetError;return false;}
            if(!runtime->animatedLights.empty())
            {
                if(runtime->animatedLights.size()!=owned.lightCount)
                {error="Glass caustic animated light set does not retain its owning scene identities";return false;}
                for(std::size_t i=0;i<owned.lightCount;++i)
                    if(runtime->animatedLights[i].id!=owned.lights[i].id
                        || !sameSpectralSource(runtime->animatedLights[i].spectralSource,owned.lights[i].spectralSource))
                    {error="Glass caustic animated light spectral ownership changed";return false;}
            }
            for(std::size_t i=0;i<owned.lightCount;++i)
                evaluated->lights[i]=effectiveNativeLightRecord(owned,*runtime,i);
        }
        if(!validateVisual3DScene(*evaluated).valid())
        {error="Glass caustic evaluated light scene is invalid";return false;}
        active=*visual3d_detail::findById(evaluated->lights,evaluated->lightCount,prepared.sceneLight);
    }
    const auto snapshot=std::make_shared<const SceneLightRecord>(active);
    SourceInputs inputs;if(!sourceInputs(prepared,snapshot,inputs,error))return false;
    output=snapshot;return true;
}
inline void appendTriangleTexels(const Triangle& t, std::vector<Texel>& output)
{
    for(const auto& p:t.points)output.push_back({p[0],p[1],p[2],t.area});
    output.push_back({t.normal[0],t.normal[1],t.normal[2],0});
    output.push_back({float(t.object&65535),float(t.object>>16),float(t.material&65535),float(t.material>>16)});
    output.push_back({float(t.instance),float(t.triangle),t.glass?1.0f:0.0f,t.reflectance?float(t.reflectance->version):0.0f});
    for(unsigned half=0;half<2;++half)
    {
        Texel rho{};
        if(t.reflectance)for(unsigned b=0;b<4;++b)rho[b]=t.reflectance->coefficients[half*4+b];
        output.push_back(rho);
    }
}
inline void appendEntrySamples(const Triangle& triangle,Point toSource,SchedulingDescription scheduling,std::vector<EntrySample>& output)
{
    if(!triangle.glass || dot(triangle.normal,toSource)<=0)return;
    const auto e1=sub(triangle.points[1],triangle.points[0]),e2=sub(triangle.points[2],triangle.points[0]);
    std::uint32_t quadrature=0;
    const auto append=[&](float u,float v) {
        EntrySample entry;entry.position=advance(advance(triangle.points[0],e1,u),e2,v);
        entry.normal=triangle.normal;entry.area=triangle.area/(scheduling.entrySubdivisions*scheduling.entrySubdivisions);
        entry.object=triangle.object;entry.material=triangle.material;entry.instance=triangle.instance;entry.triangle=triangle.triangle;
        entry.quadrature=quadrature++;
        output.push_back(entry);
    };
    for(unsigned u=0;u<scheduling.entrySubdivisions;++u)for(unsigned v=0;v<scheduling.entrySubdivisions-u;++v)
    {
        append((u+1.0f/3)/scheduling.entrySubdivisions,(v+1.0f/3)/scheduling.entrySubdivisions);
        if(u+v+1<scheduling.entrySubdivisions)append((u+2.0f/3)/scheduling.entrySubdivisions,(v+2.0f/3)/scheduling.entrySubdivisions);
    }
}
inline bool selectCamera(const Prepared& prepared,const NativeFixtureSceneRuntimeInputs* runtime,
    HarmonicMIDI::grid::SceneCameraRecord& output,std::string& identity,std::string& error)
{
    using namespace HarmonicMIDI::grid;
    if(!prepared.glass.sourceScene) {error="Glass caustic camera owner is missing";return false;}
    const auto& scene=*prepared.glass.sourceScene;
    const auto* owner=visual3d_detail::findById(scene.cameras,scene.cameraCount,scene.activeCamera);
    if(!owner || (runtime && !validFixtureRuntimeInputs(*runtime)))
    {error="Glass caustic active camera or runtime inputs are invalid";return false;}
    auto camera=runtime && runtime->cameraOverride?*runtime->cameraOverride
        :runtime && runtime->animatedCamera?*runtime->animatedCamera:*owner;
    if(camera.id!=owner->id || !validFixtureCameraOverride(camera))
    {error="Glass caustic camera does not retain its exact active owner";return false;}
    if(runtime) camera=runtimeCameraRecord(camera,runtime->cameraTranslationOffset,
        runtime->cameraRotationDegrees,runtime->cameraVerticalFovDegrees);
    if(!validFixtureCameraOverride(camera)) {error="Glass caustic evaluated camera is invalid";return false;}
    sceneaov::detail::Writer writer;writer.u32(camera.id.value);sceneaov::detail::writeTransform(writer,camera.transform);
    for(float v:{camera.verticalFovRadians,camera.nearPlane,camera.farPlane})writer.f32(v);
    identity=sceneaov::detail::hex(writer.bytes);output=camera;return true;
}
inline void appendWorkTexels(const EntrySample& entry,const EntryWork& work,std::vector<Texel>& output,
    bool multipleScatter=false)
{
    output.push_back({entry.position[0],entry.position[1],entry.position[2],entry.area});
    output.push_back({entry.normal[0],entry.normal[1],entry.normal[2],work.incidentPower});
    output.push_back({float(entry.object&65535),float(entry.object>>16),float(entry.material&65535),float(entry.material>>16)});
    output.push_back({float(entry.instance),float(entry.triangle),float(multipleScatter?entry.quadrature:work.entry),float(work.band)});
}
inline std::string frameIdentity(const Prepared& prepared,const Frame& frame)
{
    if(frame.gratingIncidentBinding)return frame.gratingIncidentBinding->identity;
    std::string identity=prepared.identity;word(identity,kTableVersion);
    if(frame.gratingStaging){if(!prepared.glass.distinctGrating){return {};}word(identity,kGratingIncidentVersion);identity+=prepared.glass.distinctGrating->material->structuralDigest();}
    // Smooth identities and upload bytes retain their released encoding.
    if(frame.roughness!=0) {word(identity,kRoughTransportVersion);bits(identity,frame.roughness);word(identity,frame.interfaceSamples);}
    if(frame.multipleScatterPolicy) {
        const auto& policy=*frame.multipleScatterPolicy;
        word(identity,policy.version);word(identity,policy.maximumEvents);word(identity,policy.maximumExits);bits(identity,policy.maximumResidual);
    }
    for(const auto* value:{&frame.sourceIdentity,&frame.branchPolicyIdentity,&frame.cameraIdentity})
    {word(identity,value->size());identity+=*value;}
    for(const auto& instance:frame.instances) {word(identity,instance.object);for(float v:instance.world)bits(identity,v);}
    for(const auto& t:frame.triangleTexels)for(float v:t)bits(identity,v);
    for(const auto& entry:frame.entries)
    {
        for(float v:entry.position)bits(identity,v);
        for(float v:entry.normal)bits(identity,v);
        bits(identity,entry.area);word(identity,entry.object,8);word(identity,entry.material,8);
        word(identity,entry.instance,8);word(identity,entry.triangle,8);
        if(frame.multipleScatterPolicy)word(identity,entry.quadrature,8);
    }
    word(identity,frame.width);word(identity,frame.height);word(identity,frame.scheduling.version);word(identity,frame.scheduling.entrySubdivisions);
    word(identity,frame.bounds.queryWork);word(identity,frame.bounds.maximumDeposits);word(identity,frame.bounds.transientBytes);
    if(frame.multipleScatterPolicy) {
        word(identity,frame.bounds.combinedWork);word(identity,frame.bounds.microfacetWork);word(identity,frame.bounds.microfacetTemporaryBytes);
    }
    const auto& deposition=frame.bounds.deposition;word(identity,kDepositionVersion);
    for(auto value:{deposition.pixelCount,deposition.maximumDeposits,deposition.paddedKeyCount,deposition.sortPasses,
        deposition.depositionWork,deposition.resolveWork,deposition.scratchBytes,deposition.largestBufferBytes})word(identity,value);
    word(identity,deposition.sortLevels);
    for(const auto& t:frame.workTexels)for(float v:t)bits(identity,v);
    for(float v:frame.sourceDirectionTrace)bits(identity,v);
    for(const auto& r:frame.sourceRadiance)for(float v:r)bits(identity,v);
    if(frame.sourceDirectionalIrradiance.version!=0) {
        word(identity,frame.sourceDirectionalIrradiance.version);
        word(identity,std::uint32_t(frame.sourceDirectionalIrradiance.measure));
        for(float v:frame.sourceDirectionalIrradiance.coefficients)bits(identity,v);
        for(const auto& task:frame.work) {
            word(identity,task.sourceMeasureVersion);word(identity,std::uint32_t(task.sourceMeasure));
            bits(identity,task.directionalCoefficient);word(identity,task.sourceLight.value);
        }
    }
    for(float v:frame.ior)bits(identity,v);
    for(float v:frame.query.control)bits(identity,v);
    const auto end=frame.query.control[1];
    if (!std::isfinite(end) || end<0 || end>frame.query.texels.size() || std::floor(end)!=end)return {};
    for(std::size_t t=0;t<std::size_t(end);++t)for(float v:frame.query.texels[t])bits(identity,v);
    return identity;
}
inline bool calculateResourceBounds(const Prepared& prepared,const Frame& frame,ResourceBounds& output,std::string& error)
{
    const auto refuse=[&](const char* text){error=text;return false;};
    if(frame.multipleScatterPolicy && (!prepared.branchPolicy || frame.roughness<=0
        || !validMultipleScatterPolicy(*frame.multipleScatterPolicy)))
        return refuse("Glass caustic multiple-scatter policy is unsupported");
    if(!std::isfinite(frame.query.control[0]) || frame.query.control[0]<1
        || frame.query.control[0]>=frame.query.texels.size() || std::floor(frame.query.control[0])!=frame.query.control[0]
        || frame.work.size()>kMaximumEntrySamples*8)return refuse("Glass caustic native query topology is invalid");
    const auto& allowance=frame.query.texels[std::size_t(frame.query.control[0])-1];
    for(unsigned i=0;i<2;++i)if(!std::isfinite(allowance[i]) || allowance[i]<0 || allowance[i]>=65536 || std::floor(allowance[i])!=allowance[i])
        return refuse("Glass caustic native query allowance is invalid");
    const auto work=std::uint64_t(allowance[0])+(std::uint64_t(allowance[1])<<16);
    const auto visits=prepared.branchPolicy?prepared.branchPolicy->description().maximumVisits:kMaximumBranchVisits;
    const auto diagnosticQueries=prepared.branchPolicy?2*(kMaximumInterfaces+3)+visits:kMaximumInterfaces+3;
    // Per path, source/entry plus visits and two camera queries per possible
    // deposit. CPU diagnostic traversals are optional and never GPU authorities.
    ResourceBounds candidate;
    const auto queryCount=std::uint64_t(frame.work.size())*std::max<std::size_t>(diagnosticQueries,2+3*visits);
    if(!work || work>opaquevisibility::kMaximumClosestHitWork || queryCount>kMaximumQueryWork/work)
        return refuse("Glass caustic native path/camera query work exceeds budget");
    candidate.queryWork=queryCount*work;candidate.maximumDeposits=std::uint64_t(frame.work.size())*visits;
    if(frame.multipleScatterPolicy) {
        if(!multipleScatterWorkWithinBounds(candidate.queryWork,candidate.maximumDeposits,*frame.multipleScatterPolicy,candidate.combinedWork))
            return refuse("Glass caustic combined query and microfacet work exceeds budget");
        candidate.microfacetWork=candidate.combinedWork-candidate.queryWork;
        const auto pending=prepared.branchPolicy->description().maximumPending;
        // Every live work invocation reserves the packed ancestry/seed branch
        // extension and sixteen vec4 directions plus uint event orders. Native
        // consumers must account for any larger actual aligned struct delta.
        candidate.microfacetTemporaryBytes=std::uint64_t(frame.work.size())
            *(pending*kMultipleScatterPendingExtensionBytes
                +smoothglass::kMaximumMicrofacetExits*(sizeof(Texel)+sizeof(std::uint32_t)));
    }
    if(!calculateDepositionSchedule(frame.width,frame.height,candidate.maximumDeposits,candidate.deposition,error))return false;
    // Native scratch and completed host readback coexist at the atomic gate.
    // Reserve each once; uploaded source/query tables are charged separately.
    const auto add=[&](std::uint64_t count,std::uint64_t stride) {
        if(count>renderpassoutput::kMaximumTotalBytes/stride
            || candidate.transientBytes>renderpassoutput::kMaximumTotalBytes-count*stride)return false;
        candidate.transientBytes+=count*stride;return true;
    };
    if(!add(std::uint64_t(frame.width)*frame.height,2*kAccumulatorBytesPerPixel)
        || !add(frame.workTexels.size(),sizeof(Texel)) || !add(frame.triangleTexels.size(),sizeof(Texel))
        || !add(frame.query.texels.size(),sizeof(Texel)) || !add(frame.work.size(),2*kPathReceiptStride*sizeof(Texel))
        || !add(candidate.maximumDeposits,2*kDepositReceiptStride*sizeof(Texel))
        || !add(candidate.deposition.scratchBytes,1)
        || !add(candidate.maximumDeposits,128) || !add(frame.work.size(),64)
        // Appended host fields occupy storage even in unmarked legacy frames.
        // Reserve two host work copies at their worst push-back capacity, plus
        // the frame/source measure owners, without changing native strides.
        || !add(frame.work.size(),4*(sizeof(EntryWork)-16))
        || !add(4,sizeof(diffractionmaterial::DirectionalSpectralIrradiance))
        // Optional actual input upload, shader echo and per-work potential.
        // GPU float storage plus two host copies at bounded vector capacity.
        || (frame.sourceDirectionalIrradiance.version!=0
            && (!add(3,kDirectionalSourceInputBytes+kDirectionalSourceEchoBytes)
                || !add(frame.work.size(),5*sizeof(float)) || !add(frameIdentity(prepared,frame).size(),2)))
        || (frame.multipleScatterPolicy && !add(candidate.microfacetTemporaryBytes,1))
        // Frame, native readback, expected submission and completed receipt
        // each retain this exact owner. Bounds values have fixed-width identity
        // encoding, so the size is independent of this reservation's value.
        || !add(frameIdentity(prepared,frame).size(),4))
        return refuse("Glass caustic native scratch exceeds the checked whole-frame memory budget");
    output=candidate;error.clear();return true;
}
inline bool admitFrameMemory(const Frame& frame,std::uint64_t otherOwnedBytes,std::uint64_t& output,std::string& error)
{
    if(frame.bounds.transientBytes>renderpassoutput::kMaximumTotalBytes
        || otherOwnedBytes>renderpassoutput::kMaximumTotalBytes-frame.bounds.transientBytes)
    {error="Glass caustic scratch plus retained native frame resources exceeds budget";return false;}
    output=otherOwnedBytes+frame.bounds.transientBytes;error.clear();return true;
}
inline bool validateTables(const Prepared& prepared,const Frame& frame,std::string& error,bool gratingPrerequisite=false,
    SourceMeasureUse sourceUse=SourceMeasureUse::LegacyNative);
inline bool validateQueries(const Prepared& prepared,const Frame& frame,std::string& error);
inline bool buildFrameImpl(const Prepared& prepared, const std::vector<opaquevisibility::Instance>& instances,
    const NativeFixtureSceneRuntimeInputs* runtime,std::uint32_t width,std::uint32_t height,Frame& output,std::string& error,
    SchedulingDescription scheduling={},std::optional<MultipleScatterPolicy> multipleScatterPolicy={},bool gratingPrerequisite=false)
{
    const auto refuse=[&](const char* text){error=text;return false;};
    if (!prepared.glass.sourceScene || !prepared.glass.sourceMaterial || prepared.identity.empty()
        || !nativeFixtureDimensionsWithinBounds(width,height) || !validScheduling(scheduling))return refuse("Glass caustic frame has no admitted ownership or extent");
    const auto& scene=*prepared.glass.sourceScene;
    std::array<bool,64> seen{};
    Frame candidate;candidate.width=width;candidate.height=height;candidate.scheduling=scheduling;candidate.instances=instances;
    if(prepared.glass.distinctGrating&&!gratingPrerequisite)return refuse("Distinct grating requires guarded native staging admission");
    candidate.gratingStaging=bool(prepared.glass.distinctGrating);
    candidate.roughness=prepared.glass.roughness;
    candidate.interfaceSamples=candidate.roughness==0?1:kRoughInterfaceSamples;
    candidate.multipleScatterPolicy=multipleScatterPolicy;
    if(multipleScatterPolicy && (!prepared.branchPolicy || candidate.roughness<=0 || !validMultipleScatterPolicy(*multipleScatterPolicy)))
        return refuse("Glass caustic multiple-scatter policy is unsupported");
    if(!selectCamera(prepared,runtime,candidate.camera,candidate.cameraIdentity,error))return false;
    if (!opaquevisibility::frame(prepared.glass.query,instances,candidate.query,error))return false;
    if(!selectActiveSource(prepared,runtime,candidate.activeSource,error))return false;
    SourceInputs source;if(!sourceInputs(prepared,candidate.activeSource,source,error))return false;
    const Point toSource=source.direction;
    candidate.sourceIdentity=source.identity;
    candidate.sourceDirectionalIrradiance=source.directionalIrradiance;
    candidate.branchPolicyIdentity=prepared.branchPolicy?prepared.branchPolicy->identity():std::string{};
    candidate.sourceDirectionTrace={toSource[0],toSource[1],toSource[2],source.maximum};
    for(unsigned b=0;b<8;++b)candidate.sourceRadiance[b/4][b%4]=source.radiance[b];
    const auto* volume=nativeSurfaceForObject(prepared.glass.sourceMaterial.get(),prepared.glass.object);
    if (!volume)return refuse("Glass caustic frame volume owner is missing");
    for(unsigned b=0;b<8;++b)
    {
        candidate.ior[b]=surfacematerial::wavelengthIor(volume->parameters.transmissionIorClearcoat[1],
            volume->parameters.transmissionIorClearcoat[3],diffractionmaterial::kProductionWavelengthsNanometres[b]*.001f);
        if (!std::isfinite(candidate.ior[b]) || candidate.ior[b]<1 || candidate.ior[b]>3)
            return refuse("Glass caustic eight-band IOR is outside [1,3]");
    }
    for(std::size_t i=0;i<instances.size();++i)
    {
        const auto& placement=instances[i];
        if (placement.object>=scene.objectCount || seen[placement.object])return refuse("Glass caustics refuse duplicated or foreign instances");
        seen[placement.object]=true;
        const auto& object=scene.objects[placement.object];
        const bool glass=object.id==prepared.glass.object;
        const auto* surface=nativeSurfaceForObject(prepared.glass.sourceMaterial.get(),object.id);
        opaquevisibility::Matrix inverse;
        if (!surface || !opaquevisibility::inverse(placement.world,inverse))return refuse("Glass caustic world transform or material owner is invalid");
        // Inverse-transpose outward normals and triangle winding must agree.
        const auto& m=placement.world;
        const double determinant=double(m[0])*(double(m[5])*m[10]-double(m[9])*m[6])
            -double(m[4])*(double(m[1])*m[10]-double(m[9])*m[2])
            +double(m[8])*(double(m[1])*m[6]-double(m[5])*m[2]);
        if (!(determinant>0))return refuse("Glass caustics refuse reflected world winding");
        for(std::size_t t=0;t<object.indexCount/3;++t)
        {
            Triangle triangle;triangle.glass=glass;triangle.object=object.id.value;triangle.material=object.material.value;
            triangle.instance=std::uint32_t(i);triangle.triangle=std::uint32_t(t);
            if (!glass)triangle.reflectance=surface->spectralReflectance;
            for(unsigned v=0;v<3;++v)triangle.points[v]=opaquevisibility::transform(m,opaquevisibility::point(
                scene.vertices[object.firstVertex+scene.indices[object.firstIndex+t*3+v]].position));
            triangle.normal=cross(sub(triangle.points[1],triangle.points[0]),sub(triangle.points[2],triangle.points[0]));
            triangle.area=.5f*std::sqrt(dot(triangle.normal,triangle.normal));
            if (!std::isfinite(triangle.area) || triangle.area<=0 || !normalize(triangle.normal))
                return refuse("Glass caustic transformed triangle is degenerate or nonfinite");
            appendTriangleTexels(triangle,candidate.triangleTexels);candidate.triangles.push_back(triangle);
            appendEntrySamples(triangle,toSource,scheduling,candidate.entries);
        }
    }
    for(std::size_t o=0;o<scene.objectCount;++o)if(scene.objects[o].indexCount && !seen[o])
        return refuse("Glass caustic frame omitted an authored object");
    const auto root=std::size_t(candidate.query.control[0]);
    if (!root || root>=candidate.query.texels.size() || candidate.entries.size()>kMaximumEntrySamples)
        return refuse("Glass caustic query/sample admission is empty or over budget");
    const auto& allowance=candidate.query.texels[root-1];
    const auto work=std::uint64_t(allowance[0])+(std::uint64_t(allowance[1])<<16);
    const auto perEntry=prepared.branchPolicy?2*(kMaximumInterfaces+3)+prepared.branchPolicy->description().maximumVisits
        :kMaximumInterfaces+3;
    const auto queries=std::uint64_t(candidate.entries.size())*8*candidate.interfaceSamples*perEntry;
    if (!work || queries>kMaximumQueryWork/work)return refuse("Glass caustic finite path query work exceeds budget");
    for(std::size_t e=0;e<candidate.entries.size();++e)for(unsigned sample=0;sample<candidate.interfaceSamples;++sample)for(unsigned b=0;b<8;++b)
    {
        const auto& entry=candidate.entries[e];
        const double power=double(entrySourceCoefficient(source,b))*std::max(0.0f,dot(entry.normal,toSource))*entry.area/candidate.interfaceSamples;
        if(!std::isfinite(power) || power>std::numeric_limits<float>::max() || (power>0 && float(power)==0))
            return refuse("Glass caustic entry power is unrepresentable");
        EntryWork task{std::uint32_t(e),b,float(power),sample};
        if(source.directionalIrradiance.version==1) {
            task.sourceMeasure=source.directionalIrradiance.measure;task.sourceMeasureVersion=source.directionalIrradiance.version;
            task.directionalCoefficient=source.directionalIrradiance.coefficients[b];task.sourceLight=source.light;
        }
        candidate.work.push_back(task);appendWorkTexels(entry,task,candidate.workTexels,candidate.multipleScatterPolicy.has_value());
    }
    if(!calculateResourceBounds(prepared,candidate,candidate.bounds,error))return false;
    candidate.identity=frameIdentity(prepared,candidate);
    if (!validateTables(prepared,candidate,error,gratingPrerequisite,SourceMeasureUse::DirectionalIrradiancePreparation))return false;
    output=std::move(candidate);error.clear();return true;
}

inline bool buildScheduledFrame(const Prepared& prepared,const std::vector<opaquevisibility::Instance>& instances,
    const NativeFixtureSceneRuntimeInputs* runtime,std::uint32_t width,std::uint32_t height,
    SchedulingDescription scheduling,Frame& output,std::string& error)
{return buildFrameImpl(prepared,instances,runtime,width,height,output,error,scheduling);}
inline bool buildGratingStagingFrame(const Prepared& prepared,const std::vector<opaquevisibility::Instance>& instances,
    const NativeFixtureSceneRuntimeInputs* runtime,std::uint32_t width,std::uint32_t height,
    SchedulingDescription scheduling,Frame& output,std::string& error)
{return buildFrameImpl(prepared,instances,runtime,width,height,output,error,scheduling,{},true);}
// Explicit private request only. Existing builders and dispatcher never choose
// version 2; native consumers must advertise the exact separate contract.
inline bool buildMultipleScatterFrame(const Prepared& prepared,const std::vector<opaquevisibility::Instance>& instances,
    const NativeFixtureSceneRuntimeInputs* runtime,std::uint32_t width,std::uint32_t height,
    SchedulingDescription scheduling,const MultipleScatterPolicy& policy,Frame& output,std::string& error)
{return buildFrameImpl(prepared,instances,runtime,width,height,output,error,scheduling,policy);}
inline bool buildFrame(const Prepared& prepared,const std::vector<opaquevisibility::Instance>& instances,
    std::uint32_t width,std::uint32_t height,Frame& output,std::string& error)
{return buildFrameImpl(prepared,instances,nullptr,width,height,output,error);}
inline bool buildFrame(const Prepared& prepared,const std::vector<opaquevisibility::Instance>& instances,
    const NativeFixtureSceneRuntimeInputs& runtime,std::uint32_t width,std::uint32_t height,Frame& output,std::string& error)
{return buildFrameImpl(prepared,instances,&runtime,width,height,output,error);}

// Admission/reference receipt only. No receiver contribution, RGB or product
// pixel is computed/cached here. Native consumers must trace independently.
// This bounded first path follows transmission at each interface; reflected
// flux is retained in its separate ledger, not discarded or relabeled caustic.
// Reflected branches use the separate explicitly admitted contract below.
struct PathReceipt
{
    bool sourceBlocked = false, receiverHit = false;
    opaquevisibility::ClosestHit receiver;
    float reflectedFraction = 0, transmittedFraction = 1;
    std::size_t interfaces = 0;
};
inline bool validatePath(const Prepared& prepared,const Frame& frame,const EntrySample& entry,
    unsigned band,PathReceipt& output,std::string& error,bool entryOnly=false)
{
    const auto refuse=[&](const char* text){error=text;return false;};
    if (band>=8 || !prepared.glass.sourceScene || !prepared.glass.sourceMaterial)return refuse("Glass caustic path ownership is missing");
    SourceInputs source;
    if(!sourceInputs(prepared,frame.activeSource,source,error))return false;
    if(!HarmonicMIDI::grid::sameDirectionalIrradiance(frame.sourceDirectionalIrradiance,source.directionalIrradiance))
        return refuse("Glass caustic directional measure changed after admission");
    if(frame.sourceIdentity!=source.identity
        || frame.branchPolicyIdentity!=(prepared.branchPolicy?prepared.branchPolicy->identity():std::string{}))
        return refuse("Glass caustic path source or branch policy identity is stale");
    if(std::memcmp(&source.radiance[band],&frame.sourceRadiance[band/4][band%4],sizeof(float))!=0)
        return refuse("Glass caustic authored Li changed after admission");
    const Point toSource{frame.sourceDirectionTrace[0],frame.sourceDirectionTrace[1],frame.sourceDirectionTrace[2]};
    const float maximum=frame.sourceDirectionTrace[3];
    if (!finite(entry.position) || !finite(entry.normal) || !finite(toSource) || maximum<=2*bias(entry.position)
        || maximum!=source.maximum || toSource!=source.direction)
        return refuse("Glass caustic path source or finite range is stale");
    const auto endpoint=[&](const opaquevisibility::ClosestHit& hit)
    {
        const auto* object=HarmonicMIDI::grid::visual3d_detail::findById(prepared.glass.sourceScene->objects,
            prepared.glass.sourceScene->objectCount,HarmonicMIDI::grid::SceneObjectId{hit.objectId});
        return object && object->material.value==hit.materialId && hit.triangleId<object->indexCount/3
            && hit.instanceId<prepared.glass.sourceScene->objectCount && finite(hit.position) && finite(hit.outwardNormal);
    };
    const auto* volume=nativeSurfaceForObject(prepared.glass.sourceMaterial.get(),prepared.glass.object);
    if (!volume || frame.ior[band]!=surfacematerial::wavelengthIor(volume->parameters.transmissionIorClearcoat[1],
        volume->parameters.transmissionIorClearcoat[3],diffractionmaterial::kProductionWavelengthsNanometres[band]*.001f))
        return refuse("Glass caustic path wavelength IOR changed after admission");
    if(frame.roughness!=0 && !entryOnly)return refuse("Rough caustic paths require sampled branch transport");
    PathReceipt candidate;
    const auto sourceHit=opaquevisibility::closestHitCpu(prepared.glass.query,frame.query,entry.position,toSource,kRayMinimum,maximum);
    if(sourceHit.status<0 || (sourceHit.status==1 && !endpoint(sourceHit)))return refuse("Glass caustic source leg query is incomplete");
    if(sourceHit.status==1) {candidate.sourceBlocked=true;candidate.transmittedFraction=0;output=candidate;error.clear();return true;}
    Point incoming=toSource;for(auto& v:incoming)v=-v;
    // Query the entry too, retaining exact object/material/instance/triangle.
    const float offset=bias(entry.position);
    const auto first=opaquevisibility::closestHitCpu(prepared.glass.query,frame.query,advance(entry.position,toSource,2*offset),
        incoming,0,4*offset);
    if(first.status!=1 || !first.entering || first.objectId!=entry.object || first.materialId!=entry.material
        || first.instanceId!=entry.instance || first.triangleId!=entry.triangle || !endpoint(first))
        return refuse("Glass caustic entry query identity is incomplete or stale");
    if(entryOnly) {output=candidate;error.clear();return true;}
    auto hit=first;bool inside=false;
    for(std::size_t depth=0;depth<kMaximumInterfaces;++depth)
    {
        if (hit.objectId!=prepared.glass.object.value || inside==hit.entering)return refuse("Glass caustic interface order is invalid");
        Point normal=hit.outwardNormal;if(inside)for(auto& v:normal)v=-v;
        Interface optics;
        if(!interface(incoming,normal,inside?frame.ior[band]:1,inside?1:frame.ior[band],optics))
            return refuse("Glass caustic interface physics is nonfinite");
        candidate.reflectedFraction+=candidate.transmittedFraction*optics.reflection;
        candidate.transmittedFraction*=1-optics.reflection;++candidate.interfaces;
        if(optics.tir) {output=candidate;error.clear();return true;}
        inside=!inside;incoming=optics.transmitted;
        const auto origin=advance(hit.position,normal,-bias(hit.position));
        hit=opaquevisibility::closestHitCpu(prepared.glass.query,frame.query,origin,incoming,kRayMinimum,maximum);
        if(hit.status<0 || (hit.status==1 && !endpoint(hit)))return refuse("Glass caustic interface/receiver query is incomplete");
        if(hit.status==0)
        {
            if(inside)return refuse("Glass caustic finite bound did not close the glass path");
            output=candidate;error.clear();return true;
        }
        if(hit.objectId!=prepared.glass.object.value)
        {
            if(inside)return refuse("Glass caustic unsupported geometry overlaps the volume");
            const auto* surface=nativeSurfaceForObject(prepared.glass.sourceMaterial.get(),HarmonicMIDI::grid::SceneObjectId{hit.objectId});
            if(!surface || !surface->spectralReflectance || !surfacematerial::validSpectralReflectance(*surface->spectralReflectance))
                return refuse("Glass caustic receiver hit has no authored rho owner");
            candidate.receiverHit=hit.entering;candidate.receiver=hit;
            output=candidate;error.clear();return true;
        }
    }
    return refuse("Glass caustic path exhausted its interface bound");
}
inline bool validateTables(const Prepared& prepared,const Frame& frame,std::string& error,bool gratingPrerequisite,SourceMeasureUse sourceUse)
{
    if(frame.sourceDirectionalIrradiance.version!=0 && sourceUse!=SourceMeasureUse::DirectionalIrradiancePreparation)
    {error="Directional irradiance native consumers are unavailable";return false;}
    if((frame.gratingIncidentBinding||frame.gratingStaging)&&!gratingPrerequisite)
    {error="Glass to grating native incident consumers are unavailable";return false;}
    if(frame.gratingIncidentBinding) {
        const auto& binding=*frame.gratingIncidentBinding;
        if(frame.identity!=binding.identity||binding.prepared.identity!=prepared.identity)
        {error="Glass to grating bound frame identity changed";return false;}
        auto staging=frame;staging.gratingIncidentBinding.reset();staging.identity=frameIdentity(prepared,staging);
        if(staging.identity!=binding.frame.identity)
        {error="Glass to grating bound staging geometry changed";return false;}
        return validateTables(prepared,staging,error,true,sourceUse);
    }
    if(frame.gratingStaging!=bool(prepared.glass.distinctGrating))
    {error="Glass to grating staging owner marker changed";return false;}
    if (!prepared.glass.sourceScene || !prepared.glass.sourceMaterial || prepared.identity.empty() || frame.entries.empty()
        || frame.entries.size()>kMaximumEntrySamples || frame.identity.empty()
        || frame.identity!=frameIdentity(prepared,frame)
        || frame.triangleTexels.size()!=frame.triangles.size()*kTriangleStride)
    {error="Glass caustic frame table/query identity is incomplete or changed";return false;}
    if(!validScheduling(frame.scheduling) || !nativeFixtureDimensionsWithinBounds(frame.width,frame.height)
        || frame.work.size()!=frame.entries.size()*8*frame.interfaceSamples || frame.workTexels.size()!=frame.work.size()*kWorkStride)
    {error="Glass caustic native schedule extent or work identity changed";return false;}
    std::string cameraIdentity;HarmonicMIDI::grid::SceneCameraRecord camera;
    NativeFixtureSceneRuntimeInputs cameraRuntime;cameraRuntime.cameraOverride=frame.camera;
    if(!selectCamera(prepared,&cameraRuntime,camera,cameraIdentity,error) || cameraIdentity!=frame.cameraIdentity)
    {error="Glass caustic native camera identity changed";return false;}
    SourceInputs source;if(!sourceInputs(prepared,frame.activeSource,source,error))return false;
    if(!HarmonicMIDI::grid::sameDirectionalIrradiance(frame.sourceDirectionalIrradiance,source.directionalIrradiance))
    {error="Glass caustic directional measure changed after admission";return false;}
    if(frame.sourceIdentity!=source.identity
        || frame.branchPolicyIdentity!=(prepared.branchPolicy?prepared.branchPolicy->identity():std::string{}))
    {error="Glass caustic frame source or branch policy identity changed";return false;}
    if(frame.sourceDirectionTrace!=std::array<float,4>{source.direction[0],source.direction[1],source.direction[2],source.maximum})
    {error="Glass caustic native source direction/range changed";return false;}
    const auto* volume=nativeSurfaceForObject(prepared.glass.sourceMaterial.get(),prepared.glass.object);
    if(!volume) {error="Glass caustic native glass owner is missing";return false;}
    if(frame.roughness!=prepared.glass.roughness || frame.roughness!=volume->parameters.emissionRoughness[3]
        || frame.interfaceSamples!=(frame.roughness==0?1:kRoughInterfaceSamples))
    {error="Glass caustic rough interface owner or sample count changed";return false;}
    for(unsigned band=0;band<8;++band)
        if(frame.ior[band]!=surfacematerial::wavelengthIor(volume->parameters.transmissionIorClearcoat[1],
            volume->parameters.transmissionIorClearcoat[3],diffractionmaterial::kProductionWavelengthsNanometres[band]*.001f))
        {error="Glass caustic native wavelength IOR changed";return false;}
    ResourceBounds bounds;if(!calculateResourceBounds(prepared,frame,bounds,error))return false;
    if(bounds.queryWork!=frame.bounds.queryWork || bounds.maximumDeposits!=frame.bounds.maximumDeposits || bounds.transientBytes!=frame.bounds.transientBytes
        || bounds.combinedWork!=frame.bounds.combinedWork || bounds.microfacetWork!=frame.bounds.microfacetWork
        || bounds.microfacetTemporaryBytes!=frame.bounds.microfacetTemporaryBytes
        || !(bounds.deposition==frame.bounds.deposition))
    {error="Glass caustic native resource reservation changed";return false;}
    for(unsigned band=0;band<8;++band)
        if(std::memcmp(&source.radiance[band],&frame.sourceRadiance[band/4][band%4],sizeof(float))!=0)
        {error="Glass caustic authored Li changed after admission";return false;}
    // Reconstruct tables from the retained world authority. This prepares the
    // existing query format; it performs no ray query or product deposition.
    opaquevisibility::Frame exactQuery;
    if(!opaquevisibility::frame(prepared.glass.query,frame.instances,exactQuery,error))return false;
    if(exactQuery.control!=frame.query.control || exactQuery.height!=frame.query.height || exactQuery.texels!=frame.query.texels)
    {error="Glass caustic native closest-hit table no longer matches its retained instances";return false;}
    const auto& scene=*prepared.glass.sourceScene;std::size_t expectedTriangles=0;std::array<bool,64> seen{};
    for(const auto& instance:frame.instances)
    {
        if(instance.object>=scene.objectCount || seen[instance.object])
        {error="Glass caustic native instance ownership changed";return false;}
        seen[instance.object]=true;expectedTriangles+=scene.objects[instance.object].indexCount/3;
    }
    for(std::size_t o=0;o<scene.objectCount;++o)if(scene.objects[o].indexCount && !seen[o])
    {error="Glass caustic native instance omitted an authored object";return false;}
    if(frame.triangles.size()!=expectedTriangles)
    {error="Glass caustic native triangle coverage changed";return false;}
    std::set<std::pair<std::uint32_t,std::uint32_t>> triangleKeys;
    std::vector<Texel> actual;
    for(const auto& triangle:frame.triangles)
    {
        if(triangle.instance>=frame.instances.size() || !triangleKeys.emplace(triangle.instance,triangle.triangle).second)
        {error="Glass caustic native triangle instance identity changed";return false;}
        const auto& instance=frame.instances[triangle.instance];const auto& object=scene.objects[instance.object];
        const auto* surface=nativeSurfaceForObject(prepared.glass.sourceMaterial.get(),object.id);
        if(!surface || triangle.object!=object.id.value || triangle.material!=object.material.value
            || triangle.triangle>=object.indexCount/3 || triangle.glass!=(object.id==prepared.glass.object)
            || bool(triangle.reflectance)!=bool(surface->spectralReflectance)
            || (triangle.reflectance && (triangle.reflectance->version!=surface->spectralReflectance->version
                || triangle.reflectance->coefficients!=surface->spectralReflectance->coefficients)))
        {error="Glass caustic native triangle/rho no longer matches its exact material owner";return false;}
        for(unsigned v=0;v<3;++v)
            if(triangle.points[v]!=opaquevisibility::transform(instance.world,opaquevisibility::point(
                scene.vertices[object.firstVertex+scene.indices[object.firstIndex+triangle.triangle*3+v]].position)))
            {error="Glass caustic native triangle no longer matches its retained world transform";return false;}
        auto normal=cross(sub(triangle.points[1],triangle.points[0]),sub(triangle.points[2],triangle.points[0]));
        const float area=.5f*std::sqrt(dot(normal,normal));
        if(!normalize(normal) || normal!=triangle.normal || area!=triangle.area)
        {error="Glass caustic native geometric area/normal changed";return false;}
        appendTriangleTexels(triangle,actual);
    }
    if(actual!=frame.triangleTexels)
    {error="Glass caustic exact triangle/rho table changed after admission";return false;}
    std::vector<EntrySample> exactEntries;
    for(const auto& triangle:frame.triangles)appendEntrySamples(triangle,source.direction,frame.scheduling,exactEntries);
    if(exactEntries.size()!=frame.entries.size())
    {error="Glass caustic native entry quadrature coverage changed";return false;}
    std::vector<Texel> workTexels;
    for(std::size_t w=0;w<frame.work.size();++w)
    {
        const auto& task=frame.work[w];
        if(task.entry!=w/(8*frame.interfaceSamples) || task.band!=w%8
            || task.interfaceSample!=(w/8)%frame.interfaceSamples)
        {error="Glass caustic entry/band schedule changed";return false;}
        const auto& entry=frame.entries[task.entry];
        const auto& exact=exactEntries[task.entry];
        if(std::tie(entry.position,entry.normal,entry.area,entry.object,entry.material,entry.instance,entry.triangle)
                !=std::tie(exact.position,exact.normal,exact.area,exact.object,exact.material,exact.instance,exact.triangle)
            || (frame.multipleScatterPolicy && entry.quadrature!=exact.quadrature))
        {error="Glass caustic native entry quadrature identity changed";return false;}
        const bool physical=source.directionalIrradiance.version==1;
        if(task.sourceMeasureVersion!=(physical?1u:0u)
            || task.sourceMeasure!=(physical?source.directionalIrradiance.measure:diffractionmaterial::DirectionalSpectralMeasure::Unspecified)
            || task.sourceLight!=(physical?source.light:HarmonicMIDI::grid::SceneLightId{})
            || (physical ? std::memcmp(&task.directionalCoefficient,&source.directionalIrradiance.coefficients[task.band],sizeof(float))!=0
                         : task.directionalCoefficient!=0.0f))
        {error="Glass caustic entry directional coefficient or exact source owner changed";return false;}
        const float power=float(double(entrySourceCoefficient(source,task.band))*std::max(0.0f,dot(entry.normal,source.direction))*entry.area/frame.interfaceSamples);
        if(!std::isfinite(task.incidentPower) || task.incidentPower!=power)
        {error="Glass caustic Li/cosine/area schedule changed";return false;}
        appendWorkTexels(entry,task,workTexels,frame.multipleScatterPolicy.has_value());
    }
    if(workTexels!=frame.workTexels)
    {error="Glass caustic native work table changed";return false;}
    error.clear();return true;
}
// Diagnostic CPU traversal only. Native consumers use validateTables followed
// by their own sceneClosestHit paths and completed native receipt validation.
inline bool validateQueries(const Prepared& prepared,const Frame& frame,std::string& error)
{
    if(frame.multipleScatterPolicy) {error="Private multiple-scatter frames require an exact native consumer";return false;}
    if(!validateTables(prepared,frame,error))return false;
    for(const auto& entry:frame.entries)for(unsigned band=0;band<8;++band)
    {PathReceipt receipt;if(!validatePath(prepared,frame,entry,band,receipt,error))return false;}
    error.clear();return true;
}

// Geometry/flux diagnostic receipts only. No Li*rho, RGB, radiance or product
// pixels are deposited on the CPU. Native consumers must trace and deposit
// independently with this same explicitly admitted policy and whole-frame gate.
struct BranchDeposit
{
    opaquevisibility::ClosestHit receiver;
    Point incidentDirection {};
    double fraction = 0;
    std::uint64_t reflectionHistory = 0;
    std::size_t interfaces = 0, reflections = 0;
    unsigned band = 0;
};
struct BranchInterfaceReceipt
{
    opaquevisibility::ClosestHit surface;
    Point incomingDirection {};
    double incidentFraction = 0, reflectedFraction = 0, transmittedFraction = 0;
    std::uint64_t reflectionHistory = 0;
    std::size_t depth = 0;
    bool inside = false, tir = false;
    Point facetNormal {};
    double maskedFraction = 0;
    std::uint32_t interfaceSample = 0;
};
struct BranchReceipt
{
    bool sourceBlocked = false, withinErrorBound = false;
    double depositedFraction = 0, escapedFraction = 0, unresolvedFraction = 0;
    double reflectedDepositedFraction = 0, maskedFraction = 0;
    std::size_t visits = 0, interfaces = 0, tirEvents = 0;
    std::vector<BranchDeposit> deposits;
    std::vector<BranchInterfaceReceipt> interfaceLedger;
    std::string policyIdentity;
};
inline bool traceBranches(const Prepared& prepared,const Frame& frame,const EntrySample& entry,
    unsigned band,const AdmittedBranchPolicy& admitted,BranchReceipt& output,std::string& error,unsigned interfaceSample=0)
{
    if(frame.multipleScatterPolicy) {error="Private multiple-scatter frames require an exact native consumer";return false;}
    if(frame.roughness!=prepared.glass.roughness || frame.interfaceSamples!=(frame.roughness==0?1:kRoughInterfaceSamples)
        || interfaceSample>=frame.interfaceSamples)
    {error="Glass caustic sampled branch roughness or stratum identity changed";return false;}
    if(prepared.branchPolicy && admitted.identity()!=prepared.branchPolicy->identity())
    {error="Glass caustic requested branch policy differs from its owning frame policy";return false;}
    const auto& policy=admitted.description();
    BranchReceipt candidate;candidate.policyIdentity=admitted.identity();
    struct Path
    {
        Point origin,direction;
        double fraction;
        std::uint64_t history;
        std::size_t interfaces,reflections;
        bool inside;
    };
    std::vector<Path> pending;
    const auto refuse=[&](const char* message) {
        for(const auto& path:pending)candidate.unresolvedFraction+=path.fraction;
        pending.clear();candidate.withinErrorBound=false;output=std::move(candidate);error=message;return false;
    };
    // Reuse source/entry/primary-path identity checks, without replacing or
    // caching that released transmitted-path contract.
    PathReceipt primary;
    if(!validatePath(prepared,frame,entry,band,primary,error,frame.roughness!=0))return false;
    if(primary.sourceBlocked)
    {candidate.sourceBlocked=true;candidate.withinErrorBound=true;output=std::move(candidate);error.clear();return true;}
    const Point towardSource{frame.sourceDirectionTrace[0],frame.sourceDirectionTrace[1],frame.sourceDirectionTrace[2]};
    Point incoming=towardSource;for(auto& v:incoming)v=-v;
    const float offset=bias(entry.position),maximum=frame.sourceDirectionTrace[3];
    pending.push_back({advance(entry.position,towardSource,2*offset),incoming,1,0,0,0,false});
    const auto endpoint=[&](const opaquevisibility::ClosestHit& hit) {
        const auto* object=HarmonicMIDI::grid::visual3d_detail::findById(prepared.glass.sourceScene->objects,
            prepared.glass.sourceScene->objectCount,HarmonicMIDI::grid::SceneObjectId{hit.objectId});
        return object && object->material.value==hit.materialId && hit.triangleId<object->indexCount/3
            && hit.instanceId<prepared.glass.sourceScene->objectCount && finite(hit.position) && finite(hit.outwardNormal);
    };
    while(!pending.empty() && candidate.visits<policy.maximumVisits)
    {
        const auto path=pending.back();pending.pop_back();++candidate.visits;
        const auto hit=opaquevisibility::closestHitCpu(prepared.glass.query,frame.query,path.origin,path.direction,
            path.interfaces? kRayMinimum:0,maximum);
        if(hit.status<0 || (hit.status==1 && !endpoint(hit)))
        {candidate.unresolvedFraction+=path.fraction;return refuse("Glass caustic reflected branch query is incomplete");}
        if(hit.status==0)
        {
            if(path.inside)
            {candidate.unresolvedFraction+=path.fraction;return refuse("Glass caustic reflected branch did not close within its finite distance");}
            candidate.escapedFraction+=path.fraction;continue;
        }
        if(hit.objectId!=prepared.glass.object.value)
        {
            if(path.inside || !hit.entering)
            {candidate.unresolvedFraction+=path.fraction;return refuse("Glass caustic reflected branch hit unsupported interior/back-face coverage");}
            const auto* surface=nativeSurfaceForObject(prepared.glass.sourceMaterial.get(),HarmonicMIDI::grid::SceneObjectId{hit.objectId});
            if(!surface || !surface->spectralReflectance || !surfacematerial::validSpectralReflectance(*surface->spectralReflectance))
            {candidate.unresolvedFraction+=path.fraction;return refuse("Glass caustic reflected receiver has no exact authored rho owner");}
            candidate.depositedFraction+=path.fraction;
            if(path.reflections)candidate.reflectedDepositedFraction+=path.fraction;
            candidate.deposits.push_back({hit,path.direction,path.fraction,path.history,path.interfaces,path.reflections,band});
            continue;
        }
        if(path.inside==hit.entering)
        {candidate.unresolvedFraction+=path.fraction;return refuse("Glass caustic reflected branch interface order is invalid");}
        if(path.interfaces>=policy.maximumInterfaces)
        {candidate.unresolvedFraction+=path.fraction;continue;}
        Point normal=hit.outwardNormal;if(path.inside)for(auto& v:normal)v=-v;
        Interface optics;Point facet=normal;double reflected=0,transmitted=0,masked=0;
        if(frame.roughness==0) {
            if(!interface(path.direction,normal,path.inside?frame.ior[band]:1,path.inside?1:frame.ior[band],optics))
            {candidate.unresolvedFraction+=path.fraction;return refuse("Glass caustic reflected branch interface physics is nonfinite");}
            reflected=path.fraction*double(optics.reflection);transmitted=path.fraction-reflected;
        } else {
            RoughInterface rough;
            if(!roughInterface(path.direction,normal,path.inside?frame.ior[band]:1,path.inside?1:frame.ior[band],
                frame.roughness,interfaceSample,path.interfaces,rough))
            {candidate.unresolvedFraction+=path.fraction;return refuse("Glass caustic rough interface physics is nonfinite");}
            optics=rough.optics;facet=rough.facet;
            reflected=path.fraction*rough.reflected;transmitted=path.fraction*rough.transmitted;
            masked=path.fraction*rough.masked;candidate.maskedFraction+=masked;
        }
        ++candidate.interfaces;if(optics.tir)++candidate.tirEvents;
        candidate.interfaceLedger.push_back({hit,path.direction,path.fraction,reflected,transmitted,
            path.history,path.interfaces,path.inside,optics.tir,facet,masked,interfaceSample});
        // If positive physical branches cannot be represented independently,
        // retain the unsplit flux as unresolved rather than calling it absorbed.
        if(frame.roughness==0 && ((optics.reflection>0 && reflected==0) || (optics.reflection<1 && transmitted==0)))
        {candidate.unresolvedFraction+=path.fraction;continue;}
        const auto append=[&](Point direction,double fraction,bool reflection) {
            if(fraction==0)return;
            if(fraction<policy.minimumBranchFraction || pending.size()>=policy.maximumPending)
            {candidate.unresolvedFraction+=fraction;return;}
            pending.push_back({advance(hit.position,normal,reflection?bias(hit.position):-bias(hit.position)),direction,
                fraction,path.history|(reflection?(std::uint64_t(1)<<path.interfaces):0),path.interfaces+1,
                path.reflections+(reflection?1:0),reflection?path.inside:!path.inside});
        };
        append(optics.reflected,reflected,true);
        if(!optics.tir)append(optics.transmitted,transmitted,false);
    }
    for(const auto& path:pending)candidate.unresolvedFraction+=path.fraction;
    candidate.withinErrorBound=std::isfinite(candidate.depositedFraction) && std::isfinite(candidate.escapedFraction)
        && std::isfinite(candidate.unresolvedFraction) && std::isfinite(candidate.maskedFraction)
        && candidate.maskedFraction>=0 && candidate.unresolvedFraction+candidate.maskedFraction<=policy.maximumUnresolvedFraction;
    const double total=candidate.depositedFraction+candidate.escapedFraction+candidate.unresolvedFraction+candidate.maskedFraction;
    if(!std::isfinite(total) || std::abs(total-1)>1e-12)
    {candidate.withinErrorBound=false;output=std::move(candidate);error="Glass caustic branch energy accounting is incomplete";return false;}
    const bool accepted=candidate.withinErrorBound;output=std::move(candidate);
    if(!accepted) {
        error=frame.roughness!=0 && output.maskedFraction>policy.maximumUnresolvedFraction
            ?"Glass caustic rough interfaces require microfacet multiple scattering beyond the admitted energy error bound"
            :"Glass caustic unresolved branch power exceeds its admitted energy error bound";
        return false;
    }
    error.clear();return true;
}
inline bool validateBranchQueries(const Prepared& prepared,const Frame& frame,const AdmittedBranchPolicy& policy,
    std::string& outputIdentity,std::string& error)
{
    if(prepared.branchPolicy && policy.identity()!=prepared.branchPolicy->identity())
    {error="Glass caustic requested branch policy differs from its owning frame policy";return false;}
    if(!(frame.roughness==0?validateQueries(prepared,frame,error):validateTables(prepared,frame,error)))return false;
    const auto root=std::size_t(frame.query.control[0]);
    if(!root || root>=frame.query.texels.size())
    {error="Glass caustic branch query topology is incomplete";return false;}
    const auto& allowance=frame.query.texels[root-1];
    const auto work=std::uint64_t(allowance[0])+(std::uint64_t(allowance[1])<<16);
    // Includes the compatible primary identity validation plus branching.
    const auto queries=std::uint64_t(frame.work.size())*(2*(kMaximumInterfaces+3)+policy.description().maximumVisits);
    if(!work || queries>kMaximumQueryWork/work)
    {error="Glass caustic branch frame query work exceeds budget";return false;}
    for(const auto& work:frame.work)
    {BranchReceipt receipt;if(!traceBranches(prepared,frame,frame.entries[work.entry],work.band,policy,receipt,error,work.interfaceSample))return false;}
    outputIdentity=frame.identity+policy.identity();error.clear();return true;
}

// Geometry projection uses the same perspective camera and top-first pixel
// convention as native scene output. It computes no spectrum or product pixels.
struct PixelFootprint
{
    bool projected = false;
    std::uint32_t x = 0, y = 0;
    float linearDepth = 0, worldArea = 0;
};
inline Point cameraPosition(const Frame& frame)
{return {frame.camera.transform.translation.x,frame.camera.transform.translation.y,frame.camera.transform.translation.z};}
inline Point cameraRay(const Frame& frame,float x,float y)
{
    const float tangent=std::tan(frame.camera.verticalFovRadians*.5f),aspect=float(frame.width)/frame.height;
    const auto rotated=rotateSceneVector(frame.camera.transform.rotation,
        {(2*x/frame.width-1)*tangent*aspect,(1-2*y/frame.height)*tangent,-1});
    Point ray{rotated.x,rotated.y,rotated.z};normalize(ray);return ray;
}
// One nominated triangle and pixel only. These geometric candidates establish
// no native ownership, visibility, union coverage or radiometric contribution.
struct CameraCellGeometry
{
    using Vertex=std::array<double,3>;
    // A convex quad gains at most one vertex per half-plane. Three triangle
    // edges plus two camera depth planes therefore require at most nine.
    std::array<Vertex,9> vertices{};
    std::size_t vertexCount=0;
    std::uint32_t camera=0,x=0,y=0,object=0,material=0,instance=0,triangle=0;
    double worldArea=0,clippedSolidAngle=0,pixelSolidAngle=0;
};
inline bool prepareCameraCellGeometry(const Frame& frame,const Triangle& triangle,
    std::uint32_t x,std::uint32_t y,CameraCellGeometry& output,std::string& error)
{
    using V=CameraCellGeometry::Vertex;
    const auto refuse=[&](const char* text){error=text;return false;};
    if(!nativeFixtureDimensionsWithinBounds(frame.width,frame.height)||x>=frame.width||y>=frame.height
        ||!frame.camera.id.isValid()||!validFixtureCameraOverride(frame.camera)
        ||!triangle.object||!triangle.material||triangle.glass||!finite(triangle.normal)
        ||!std::isfinite(triangle.area)||triangle.area<=0)
        return refuse("Camera cell requires a finite camera, pixel and nominated receiver triangle");
    const auto subtract=[](V a,V b){for(unsigned i=0;i<3;++i){a[i]-=b[i];}return a;};
    const auto inner=[](V a,V b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];};
    const auto exterior=[](V a,V b){return V{a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};};
    const auto finiteVertex=[](V value){return std::all_of(value.begin(),value.end(),[](double v){return std::isfinite(v);});};
    const auto unit=[&](V& value) {
        const double length=std::sqrt(inner(value,value));
        if(!std::isfinite(length)||length<=0)return false;
        for(auto& v:value)v/=length;
        return finiteVertex(value);
    };
    const auto originPoint=cameraPosition(frame);const V origin{originPoint[0],originPoint[1],originPoint[2]};
    std::array<V,3> points{};
    for(unsigned i=0;i<3;++i) {
        if(!finite(triangle.points[i]))return refuse("Camera cell triangle is nonfinite");
        points[i]=subtract(V{triangle.points[i][0],triangle.points[i][1],triangle.points[i][2]},origin);
    }
    const auto e1=subtract(points[1],points[0]),e2=subtract(points[2],points[0]);
    V normal=exterior(e1,e2);const double twiceArea=std::sqrt(inner(normal,normal));
    const double edgeScale=std::max({inner(e1,e1),inner(e2,e2),inner(subtract(points[2],points[1]),subtract(points[2],points[1]))});
    // Relative sliver rejection; no absolute world-area floor. Float authored
    // normals/areas may differ from binary64 reconstruction by 1e-4 relative.
    const V authoredNormal{triangle.normal[0],triangle.normal[1],triangle.normal[2]};
    if(!std::isfinite(twiceArea)||twiceArea<=1e-12*edgeScale||!unit(normal)
        ||std::abs(inner(authoredNormal,authoredNormal)-1)>1e-4||inner(authoredNormal,normal)<1-1e-4
        ||std::abs(2*double(triangle.area)-twiceArea)>1e-4*twiceArea)
        return refuse("Camera cell triangle area or oriented normal is degenerate or stale");
    const auto forwardValue=rotateSceneVector(frame.camera.transform.rotation,{0,0,-1});
    const V forward{forwardValue.x,forwardValue.y,forwardValue.z};
    const double planeDistance=inner(normal,points[0]);
    std::array<V,9> polygon{};std::array<V,4> rays{};std::size_t count=4;
    // Top-left, bottom-left, bottom-right, top-right gives the front-facing
    // receiver's oriented winding. Reuse the released perspective ray formula.
    for(unsigned i=0;i<4;++i) {
        const auto ray=cameraRay(frame,float(x)+(i>=2),float(y)+(i==1||i==2));
        rays[i]={ray[0],ray[1],ray[2]};if(!unit(rays[i]))return refuse("Camera cell boundary ray is invalid");
        const double denominator=inner(normal,rays[i]);
        // Match the existing footprint's conservative horizon/distance bound.
        // Actual near/far clipping below uses linear camera depth, not ray range.
        if(denominator>=-1e-6||inner(forward,rays[i])<=0)
            return refuse("Camera cell pixel crosses a backface, grazing plane or receiver horizon");
        const double distance=planeDistance/denominator;
        if(!std::isfinite(distance)||distance<=0||distance>double(frame.camera.farPlane)*1e4)
            return refuse("Camera cell plane footprint is behind the camera or unbounded");
        for(unsigned a=0;a<3;++a)polygon[i][a]=rays[i][a]*distance;
        if(!finiteVertex(polygon[i]))return refuse("Camera cell plane intersection is nonfinite");
    }
    const auto solidTriangle=[&](V a,V b,V c,double& angle) {
        if(!unit(a)||!unit(b)||!unit(c))return false;
        const double determinant=std::abs(inner(a,exterior(b,c))),denominator=1+inner(a,b)+inner(b,c)+inner(c,a);
        if(!std::isfinite(determinant)||!std::isfinite(denominator)||(determinant==0&&denominator<=0))return false;
        angle=2*std::atan2(determinant,denominator);return std::isfinite(angle)&&angle>=0;
    };
    CameraCellGeometry candidate;candidate.camera=frame.camera.id.value;candidate.x=x;candidate.y=y;
    candidate.object=triangle.object;candidate.material=triangle.material;candidate.instance=triangle.instance;candidate.triangle=triangle.triangle;
    for(unsigned i=1;i<3;++i) {
        double angle=0;if(!solidTriangle(rays[0],rays[i],rays[i+1],angle))return refuse("Camera cell pixel solid angle is singular");
        candidate.pixelSolidAngle+=angle;
    }
    if(candidate.pixelSolidAngle<=0||candidate.pixelSolidAngle>=2*3.14159265358979323846)
        return refuse("Camera cell pixel solid angle is outside the perspective hemisphere");
    const auto clip=[&](V inward,double offset) {
        if(!count)return true;
        std::array<V,9> clipped{};std::size_t next=0;
        const auto append=[&](V vertex) {
            // Exact boundary duplicates are removed, without inflating any
            // half-plane by a tolerance or dropping a positive sliver.
            if(next&&(vertex==clipped[next-1]||vertex==clipped[0]))return true;
            if(next==clipped.size()||!finiteVertex(vertex))return false;
            clipped[next++]=vertex;return true;
        };
        V previous=polygon[count-1];double previousSide=inner(inward,previous)-offset;
        for(std::size_t i=0;i<count;++i) {
            const auto current=polygon[i];const double side=inner(inward,current)-offset;
            if(!std::isfinite(side)||!std::isfinite(previousSide))return false;
            if((previousSide>=0)!=(side>=0)) {
                V crossing=previousSide==0?previous:side==0?current:V{};
                if(previousSide!=0&&side!=0) {
                    const double fraction=previousSide/(previousSide-side);
                    if(!std::isfinite(fraction)||fraction<0||fraction>1)return false;
                    for(unsigned a=0;a<3;++a)crossing[a]=previous[a]+fraction*(current[a]-previous[a]);
                }
                if(!append(crossing))return false;
            }
            if(side>=0&&!append(current))return false;
            previous=current;previousSide=side;
        }
        polygon=clipped;count=next;return true;
    };
    for(unsigned i=0;i<3;++i) {
        auto inward=exterior(normal,subtract(points[(i+1)%3],points[i]));
        if(!unit(inward)||!clip(inward,inner(inward,points[i])))return refuse("Camera cell triangle clipping exceeds its finite polygon bound");
    }
    V backward=forward;for(auto& v:backward)v=-v;
    if(!clip(forward,frame.camera.nearPlane)||!clip(backward,-double(frame.camera.farPlane)))
        return refuse("Camera cell depth clipping exceeds its finite polygon bound");
    for(std::size_t i=1;i+1<count;++i) {
        const double area=.5*std::abs(inner(normal,exterior(subtract(polygon[i],polygon[0]),subtract(polygon[i+1],polygon[0]))));
        double angle=0;
        if(!std::isfinite(area)||!solidTriangle(polygon[0],polygon[i],polygon[i+1],angle))
            return refuse("Camera cell clipped measures are nonfinite or singular");
        candidate.worldArea+=area;candidate.clippedSolidAngle+=angle;
    }
    if(!std::isfinite(candidate.worldArea)||!std::isfinite(candidate.clippedSolidAngle)
        ||candidate.clippedSolidAngle>candidate.pixelSolidAngle*(1+1e-10))
        return refuse("Camera cell clipped solid angle exceeds its pixel");
    if(candidate.worldArea>0&&candidate.clippedSolidAngle>0) {
        candidate.vertexCount=count;candidate.clippedSolidAngle=std::min(candidate.clippedSolidAngle,candidate.pixelSolidAngle);
        for(std::size_t i=0;i<count;++i)for(unsigned a=0;a<3;++a)candidate.vertices[i][a]=origin[a]+polygon[i][a];
    } else {candidate.worldArea=0;candidate.clippedSolidAngle=0;}
    output=candidate;error.clear();return true;
}
inline const Triangle* hitTriangle(const Frame& frame,const opaquevisibility::ClosestHit& hit)
{
    if(hit.status!=1 || !finite(hit.position) || !finite(hit.outwardNormal)
        || !std::isfinite(hit.distance) || hit.distance<0)return nullptr;
    for(const auto& triangle:frame.triangles)
        if(triangle.object==hit.objectId && triangle.material==hit.materialId
            && triangle.instance==hit.instanceId && triangle.triangle==hit.triangleId)
        {
            const float tolerance=8*bias(hit.position);
            if(std::abs(dot(sub(hit.position,triangle.points[0]),triangle.normal))>tolerance
                || std::abs(dot(hit.outwardNormal,hit.outwardNormal)-1)>1e-4f
                || dot(hit.outwardNormal,triangle.normal)<1-1e-4f)return nullptr;
            for(unsigned e=0;e<3;++e)
                if(dot(cross(sub(triangle.points[(e+1)%3],triangle.points[e]),sub(hit.position,triangle.points[e])),triangle.normal)
                    < -tolerance*std::sqrt(dot(sub(triangle.points[(e+1)%3],triangle.points[e]),sub(triangle.points[(e+1)%3],triangle.points[e]))))return nullptr;
            return &triangle;
        }
    return nullptr;
}
inline const Triangle* receiverTriangle(const Frame& frame,const opaquevisibility::ClosestHit& hit)
{
    const auto* triangle=hitTriangle(frame,hit);
    return triangle && !triangle->glass && triangle->reflectance && hit.entering?triangle:nullptr;
}
inline bool projectReceiver(const Frame& frame,const opaquevisibility::ClosestHit& hit,PixelFootprint& output,std::string& error)
{
    if(!nativeFixtureDimensionsWithinBounds(frame.width,frame.height) || !receiverTriangle(frame,hit))
    {error="Glass caustic deposit has no exact receiver triangle";return false;}
    const auto& rotation=frame.camera.transform.rotation;
    const auto delta=sub(hit.position,cameraPosition(frame));
    const auto local=rotateSceneVector({-rotation.x,-rotation.y,-rotation.z,rotation.w},{delta[0],delta[1],delta[2]});
    PixelFootprint candidate;candidate.linearDepth=-local.z;
    if(candidate.linearDepth<frame.camera.nearPlane || candidate.linearDepth>frame.camera.farPlane)
    {output=candidate;error.clear();return true;}
    const float tangent=std::tan(frame.camera.verticalFovRadians*.5f);
    const float px=(local.x/(candidate.linearDepth*tangent*float(frame.width)/frame.height)+1)*.5f*frame.width;
    const float py=(1-local.y/(candidate.linearDepth*tangent))*.5f*frame.height;
    if(!std::isfinite(px) || !std::isfinite(py)) {error="Glass caustic camera projection is nonfinite";return false;}
    if(px<0 || py<0 || px>=frame.width || py>=frame.height) {output=candidate;error.clear();return true;}
    candidate.x=std::uint32_t(px);candidate.y=std::uint32_t(py);
    std::array<Point,4> corners;
    const auto origin=cameraPosition(frame);
    if(dot(hit.outwardNormal,cameraRay(frame,float(candidate.x)+.5f,float(candidate.y)+.5f))>=-1e-6f)
    {output=candidate;error.clear();return true;}
    for(unsigned c=0;c<4;++c)
    {
        const auto ray=cameraRay(frame,float(candidate.x)+(c==1 || c==2),float(candidate.y)+(c>=2));
        const double denominator=dot(hit.outwardNormal,ray);
        const double distance=dot(hit.outwardNormal,sub(hit.position,origin))/denominator;
        if(!std::isfinite(distance) || denominator>=-1e-6 || distance<=0 || distance>frame.camera.farPlane*1e4)
        {error="Glass caustic pixel footprint crosses the receiver horizon";return false;}
        corners[c]=advance(origin,ray,float(distance));
    }
    const auto area=[&](unsigned a,unsigned b,unsigned c) {
        const auto n=cross(sub(corners[b],corners[a]),sub(corners[c],corners[a]));return .5*std::sqrt(double(dot(n,n)));
    };
    const double worldArea=area(0,1,2)+area(0,2,3);
    if(!std::isfinite(worldArea) || worldArea<=0 || worldArea>std::numeric_limits<float>::max() || float(worldArea)==0)
    {error="Glass caustic pixel footprint is unrepresentable";return false;}
    candidate.worldArea=float(worldArea);candidate.projected=true;output=candidate;error.clear();return true;
}
struct RasterReceiver
{
    std::uint32_t object = 0, material = 0, instance = 0, triangle = 0;
    float linearDepth = 0;
};
struct NativeDeposit
{
    std::uint32_t work = 0;
    std::uint64_t reflectionHistory = 0;
    std::uint32_t interfaces = 0;
    float fraction = 0, incidentPower = 0, outgoingPower = 0, absorbedPower = 0;
    opaquevisibility::ClosestHit receiver, cameraHit, pixelHit;
    PixelFootprint footprint;
    RasterReceiver raster;
    bool cameraVisible = false;
    // Each macro exit stores microfacet order bits0..5, transmission bit6,
    // reserved bit7 zero. Used prefix length is interfaces, never a seed hash.
    std::array<std::uint8_t,32> ancestry {};
    std::uint32_t gratingVersion=0,gratingIncident=UINT32_MAX;
    std::int32_t gratingOrder=0;
};
struct NativePathReceipt
{
    std::uint32_t work = 0, depositCount = 0, visits = 0;
    std::uint64_t queryWork = 0;
    bool complete = false, sourceBlocked = false;
    float launchedPower = 0, depositedPower = 0, absorbedPower = 0, escapedPower = 0, unresolvedPower = 0;
    opaquevisibility::ClosestHit sourceHit, entryHit;
    // Rough consumers use reserved path texel 15: masked, version, sample, zero.
    float maskedPower = 0;
    std::uint32_t roughTransportVersion = 0, interfaceSample = 0;
    std::uint32_t multipleScatterVersion = 0, microEvents = 0, microDraws = 0, quadrature = 0;
    // Weighted physical power already included in unresolvedPower, never a
    // second conservation term or a replacement for maskedPower.
    float microResidualPower = 0;
};
inline bool exactReceiptInteger(float value,std::uint32_t maximum,std::uint32_t& output)
{
    if(!std::isfinite(value) || value<0 || double(value)>maximum || std::floor(value)!=value)return false;
    output=std::uint32_t(value);return true;
}
enum class PathExtensionLayout { OpenGl, Metal };
// V2 texel14={events,draws,weightedResidualPower,quadrature}. Legacy OpenGL
// texel14 holds the entry instance/triangle uint16 words; Metal retains those
// IDs in its typed Hit and has padding here. Texel15 is the rough/version tuple.
// V2 OpenGL entry indices require bindOpenGlMultipleScatterEntry before the
// final readback gate. The producer must verify actual queried IDs before it
// overwrites texel14; neither metadata nor the work table proves that query.
inline bool decodePathExtensions(const Texel& micro,const Texel& rough,PathExtensionLayout layout,NativePathReceipt& output)
{
    if(layout!=PathExtensionLayout::OpenGl && layout!=PathExtensionLayout::Metal)return false;
    NativePathReceipt candidate=output;
    if(!std::isfinite(rough[0]) || rough[0]<0
        || !exactReceiptInteger(rough[1],kRoughTransportVersion,candidate.roughTransportVersion)
        || !exactReceiptInteger(rough[2],kRoughInterfaceSamples-1,candidate.interfaceSample)
        || !exactReceiptInteger(rough[3],kPrivateMultipleScatterVersion,candidate.multipleScatterVersion))return false;
    candidate.maskedPower=rough[0];
    candidate.microEvents=candidate.microDraws=candidate.quadrature=0;candidate.microResidualPower=0;
    if(candidate.multipleScatterVersion==0) {
        if(layout==PathExtensionLayout::OpenGl) {
            std::array<std::uint32_t,4> indices{};
            for(unsigned i=0;i<4;++i)if(!exactReceiptInteger(micro[i],65535,indices[i]))return false;
            candidate.entryHit.instanceId=indices[0]|(indices[1]<<16);
            candidate.entryHit.triangleId=indices[2]|(indices[3]<<16);
        } else if(micro!=Texel{})return false;
    } else if(candidate.multipleScatterVersion==kPrivateMultipleScatterVersion) {
        if(candidate.roughTransportVersion!=kRoughTransportVersion || candidate.maskedPower!=0
            || !exactReceiptInteger(micro[0],kMaximumBranchVisits*smoothglass::kMaximumMicrofacetEvents,candidate.microEvents)
            || !exactReceiptInteger(micro[1],4*kMaximumBranchVisits*smoothglass::kMaximumMicrofacetEvents,candidate.microDraws)
            || !exactReceiptInteger(micro[3],15,candidate.quadrature)
            || !std::isfinite(micro[2]) || micro[2]<0 || candidate.microDraws!=4*candidate.microEvents)return false;
        candidate.microResidualPower=micro[2];
        if(layout==PathExtensionLayout::OpenGl && !candidate.sourceBlocked)
            candidate.entryHit.instanceId=candidate.entryHit.triangleId=UINT32_MAX;
    } else return false;
    output=candidate;return true;
}
inline std::array<Texel,2> pathExtensionTexels(const NativePathReceipt& path,PathExtensionLayout layout)
{
    std::array<Texel,2> output{};
    if(path.multipleScatterVersion==kPrivateMultipleScatterVersion)
        output[0]={float(path.microEvents),float(path.microDraws),path.microResidualPower,float(path.quadrature)};
    else if(layout==PathExtensionLayout::OpenGl)
        output[0]={float(path.entryHit.instanceId&65535),float(path.entryHit.instanceId>>16),
            float(path.entryHit.triangleId&65535),float(path.entryHit.triangleId>>16)};
    output[1]={path.maskedPower,float(path.roughTransportVersion),float(path.interfaceSample),float(path.multipleScatterVersion)};
    return output;
}
// The caller first validates the frozen tables once for its decode batch, and
// the native producer verifies the exact full entry tuple against uploaded
// work before emitting complete=true. Retained status/owners/world geometry
// are checked here before binding only the two overwritten OpenGL indices.
// This is not a CPU query, a replacement hit, or native completion evidence.
inline bool bindOpenGlMultipleScatterEntry(const Frame& frame,NativePathReceipt& path,std::string& error)
{
    const auto refuse=[&](){error="Glass caustic overwritten entry indices lack exact native/world binding";return false;};
    if(!frame.multipleScatterPolicy || !validMultipleScatterPolicy(*frame.multipleScatterPolicy) || frame.identity.empty()
        || path.multipleScatterVersion!=kPrivateMultipleScatterVersion || !path.complete || path.work>=frame.work.size())return refuse();
    const auto& work=frame.work[path.work];
    if(work.entry>=frame.entries.size())return refuse();
    const auto& entry=frame.entries[work.entry];
    if(path.quadrature!=entry.quadrature || path.interfaceSample!=work.interfaceSample)return refuse();
    if(path.sourceBlocked) {if(path.sourceHit.status!=1){return refuse();}error.clear();return true;}
    if(path.entryHit.instanceId!=UINT32_MAX || path.entryHit.triangleId!=UINT32_MAX)return refuse();
    auto hit=path.entryHit;const auto offset=bias(entry.position);
    if(hit.status!=1 || !hit.entering || hit.objectId!=entry.object || hit.materialId!=entry.material
        || !std::isfinite(hit.distance) || hit.distance<0 || hit.distance>4*offset
        || std::abs(hit.distance-2*offset)>8*offset || !finite(hit.position) || !finite(hit.outwardNormal)
        || dot(sub(hit.position,entry.position),sub(hit.position,entry.position))>64*offset*offset)return refuse();
    const Point toSource{frame.sourceDirectionTrace[0],frame.sourceDirectionTrace[1],frame.sourceDirectionTrace[2]};
    if(!finite(toSource) || std::abs(dot(toSource,toSource)-1)>1e-4f)return refuse();
    const auto residual=sub(advance(entry.position,toSource,2*offset-hit.distance),hit.position);
    if(dot(residual,residual)>64*offset*offset)return refuse();
    hit.instanceId=entry.instance;hit.triangleId=entry.triangle;
    const auto* triangle=hitTriangle(frame,hit);
    if(!triangle || !triangle->glass || dot(hit.outwardNormal,toSource)<=0)return refuse();
    path.entryHit=hit;error.clear();return true;
}
inline bool validMacroAncestry(const NativeDeposit& deposit,const MultipleScatterPolicy& policy,bool matchedMedia,
    std::uint32_t& minimumEvents)
{
    minimumEvents=0;bool inside=false;
    if(!validMultipleScatterPolicy(policy) || !deposit.interfaces || deposit.interfaces>kMaximumInterfaces
        || (deposit.reflectionHistory>>deposit.interfaces)!=0)return false;
    for(unsigned i=0;i<deposit.ancestry.size();++i)
    {
        const auto token=deposit.ancestry[i];
        if(i>=deposit.interfaces) {if(token!=0){return false;}continue;}
        const auto order=std::uint32_t(token&63);const bool transmitted=(token&64)!=0;
        if((token&128) || order>policy.maximumEvents || (matchedMedia?(order!=0 || !transmitted):order==0)
            || bool((deposit.reflectionHistory>>i)&1)==transmitted)return false;
        minimumEvents+=order;if(transmitted)inside=!inside;
    }
    return !inside;
}
// Deposit texels22..29 hold exactly32 byte tokens as float integers. The caller
// also checks texels30..31 remain zero. Legacy producers retain a zero suffix.
inline bool decodeMacroAncestry(const std::array<Texel,8>& packed,std::uint32_t version,
    std::array<std::uint8_t,32>& output)
{
    std::array<std::uint8_t,32> candidate{};
    if(version!=0 && version!=kPrivateMultipleScatterVersion)return false;
    for(unsigned i=0;i<candidate.size();++i) {
        std::uint32_t token=0;if(!exactReceiptInteger(packed[i/4][i%4],127,token) || (!version && token))return false;
        candidate[i]=std::uint8_t(token);
    }
    output=candidate;return true;
}
inline std::array<Texel,8> macroAncestryTexels(const std::array<std::uint8_t,32>& ancestry)
{
    std::array<Texel,8> result{};
    for(unsigned i=0;i<ancestry.size();++i)result[i/4][i%4]=float(ancestry[i]);
    return result;
}
// RGBA32F encoders must split uint32 owner/query IDs into exact 16-bit words,
// and uint64 history/work into four words. Neither float casts of full IDs nor
// RGB scratch are admissible. The reserved strides cover these exact records.
struct SubmissionIdentity
{
    std::string frameIdentity;
    std::uint64_t rendererGeneration = 0, submissionSerial = 0, rasterGeneration = 0;
    std::uint64_t queryGeneration = 0, scratchGeneration = 0, completedFence = 0;
};
inline bool operator==(const SubmissionIdentity& a,const SubmissionIdentity& b)
{
    return std::tie(a.frameIdentity,a.rendererGeneration,a.submissionSerial,a.rasterGeneration,
        a.queryGeneration,a.scratchGeneration,a.completedFence)
        ==std::tie(b.frameIdentity,b.rendererGeneration,b.submissionSerial,b.rasterGeneration,
        b.queryGeneration,b.scratchGeneration,b.completedFence);
}
// These are native uint/float buffers, not RGBA32F owner casts. GL std430 and
// Metal uint4/float4 records use identical 16-byte alignment and offsets.
struct alignas(16) NativeDirectionalSourceInput
{
    // Measure version, unit enum, metres-world declaration, exact uint light ID.
    std::array<std::uint32_t,4> identifiers {};
    std::array<float,8> coefficients {};
};
struct alignas(16) NativeDirectionalSourceEcho
{
    NativeDirectionalSourceInput input;
    std::array<float,8> rawRadiance {};
    std::array<float,4> directionTrace {};
};
static_assert(sizeof(NativeDirectionalSourceInput)==kDirectionalSourceInputBytes);
static_assert(sizeof(NativeDirectionalSourceEcho)==kDirectionalSourceEchoBytes);
static_assert(offsetof(NativeDirectionalSourceInput,coefficients)==16);
static_assert(offsetof(NativeDirectionalSourceEcho,rawRadiance)==48
    &&offsetof(NativeDirectionalSourceEcho,directionTrace)==80);
struct NativeDirectionalSourceUpload
{
    NativeFixtureMaterialBackend backend=NativeFixtureMaterialBackend::OpenGl;
    SubmissionIdentity submission;
    std::uint64_t bindingGeneration=0;
    NativeDirectionalSourceInput input;
    bool complete=false;
};
// Only the existing backend-private frame implements this optional interface.
// upload() exposes actual retained input-buffer readback after its fence, not
// canonical host values. There is no shared constructor/factory feature grant.
class NativeDirectionalSourceBinding
{
public:
    virtual ~NativeDirectionalSourceBinding()=default;
    virtual const NativeFixtureSceneFrame& owner() const noexcept=0;
    virtual const NativeDirectionalSourceUpload& upload() const noexcept=0;
protected:
    NativeDirectionalSourceBinding()=default;
};
struct NativeDirectionalSourceReceipt
{
    std::uint32_t version=0;
    SubmissionIdentity submission;
    std::uint64_t bindingGeneration=0;
    // The shader reads D from its real input, Li/source from original uniforms,
    // and writes potential before source-blocked paths zero launched power.
    NativeDirectionalSourceEcho echo;
    std::vector<float> potential;
    bool complete=false,overflow=false;
};
inline NativeDirectionalSourceInput directionalSourceInput(const Prepared& prepared,const Frame& frame)
{
    NativeDirectionalSourceInput input;
    input.identifiers={frame.sourceDirectionalIrradiance.version,
        std::uint32_t(frame.sourceDirectionalIrradiance.measure),kDirectionalSourceWorldMetres,prepared.sceneLight.value};
    input.coefficients=frame.sourceDirectionalIrradiance.coefficients;
    return input;
}
// Derive the complete D8 roster from the immutable scene.
inline bool directionalSourceRoster(const Prepared& prepared,
    std::array<HarmonicMIDI::grid::SceneLightId,HarmonicMIDI::grid::Visual3DScene::kMaxLights>& output,
    std::size_t& count,std::string& error)
{
    using namespace HarmonicMIDI::grid;
    if(!prepared.glass.sourceScene||!validateVisual3DScene(*prepared.glass.sourceScene).valid())
    {error="Directional source cohort has no valid immutable scene roster";return false;}
    std::array<SceneLightId,Visual3DScene::kMaxLights> candidate{};std::size_t size=0;
    const auto& scene=*prepared.glass.sourceScene;
    for(std::size_t i=0;i<scene.lightCount;++i) {
        const auto& light=scene.lights[i];
        if(light.kind==SceneLightKind::Directional&&light.spectralSource.version==1
            &&light.spectralSource.directionalIrradiance.version==1)candidate[size++]=light.id;
    }
    if(!size){error="Directional source cohort has no admitted D8 scene lights";return false;}
    output=candidate;count=size;error.clear();return true;
}
inline bool validSourceSubmission(const SubmissionIdentity& submission,const Frame& frame,
    const GratingIncidentBinding* incidentBinding=nullptr)
{
    const auto& identity=incidentBinding?incidentBinding->identity:frame.identity;
    return !identity.empty() &&submission.frameIdentity==identity
        &&submission.rendererGeneration &&submission.submissionSerial &&submission.rasterGeneration
        &&submission.queryGeneration &&submission.scratchGeneration
        &&submission.completedFence==submission.submissionSerial;
}
inline bool validateDirectionalSourceUpload(const Prepared& prepared,const Frame& frame,
    const NativeDirectionalSourceBinding& binding,const SubmissionIdentity& expected,std::string& error,
    bool gratingPrerequisite=false,const GratingIncidentBinding* incidentBinding=nullptr)
{
    const auto& upload=binding.upload();const auto& owner=binding.owner();
    if(frame.sourceDirectionalIrradiance.version!=1 ||!prepared.glass.sourceMaterial
        ||dynamic_cast<const NativeDirectionalSourceBinding*>(&owner)!=&binding
        ||owner.width()!=frame.width ||owner.height()!=frame.height ||!owner.nativeResourceCacheIdentity()
        ||upload.backend!=prepared.glass.sourceMaterial->backend
        ||(upload.backend!=NativeFixtureMaterialBackend::OpenGl &&upload.backend!=NativeFixtureMaterialBackend::Metal)
        ||(incidentBinding &&(!gratingPrerequisite ||incidentBinding->prepared.identity!=prepared.identity
            ||incidentBinding->frame.identity!=frame.identity))
        ||!upload.complete ||!upload.bindingGeneration ||!validSourceSubmission(expected,frame,incidentBinding)
        ||!(upload.submission==expected))
    {error="Directional source upload has no completed exact native frame owner";return false;}
    if(!validateTables(prepared,frame,error,gratingPrerequisite,SourceMeasureUse::DirectionalIrradiancePreparation))return false;
    const auto exact=directionalSourceInput(prepared,frame);
    if(std::memcmp(&upload.input,&exact,sizeof(exact))!=0)
    {error="Directional source native input bytes differ from their authored owner";return false;}
    error.clear();return true;
}
// Native staging only. The upload owner is independently retained by the native
// caller. This is distinct from preparation and does not validate transport.
inline bool validateDirectionalSourceTables(const Prepared& prepared,const Frame& frame,
    const NativeDirectionalSourceBinding& binding,std::string& error,bool gratingPrerequisite=false)
{
    return validateDirectionalSourceUpload(prepared,frame,binding,binding.upload().submission,error,gratingPrerequisite);
}
struct NativeGratingOrder
{
    std::int32_t order=0; // Only -1,0,+1. Exact order is child provenance.
    Point direction{};
    float power=0;
};
struct NativeGratingIncident
{
    std::uint32_t version=0,work=0,band=0,interfaces=0,orderCount=0;
    std::uint64_t reflectionHistory=0;
    std::array<std::uint8_t,32> ancestry{};
    Point origin{},direction{}; // Actual propagation toward the grating.
    float fraction=0,power=0;
    opaquevisibility::ClosestHit hit;
    Point barycentric{},dHdu{},dHdv{};
    std::array<float,2> uv{};
    Point tangent{},bitangent{},normal{};
    float substrateReflectedPower=0,absorbedPower=0,unresolvedPower=0;
    std::array<NativeGratingOrder,kGratingOrdersPerIncident> orders{};
    bool complete=false;
};
struct NativeGratingReadback
{
    SubmissionIdentity submission;
    std::shared_ptr<const NativeDirectionalSourceReceipt> directionalSource;
    std::vector<NativeGratingIncident> incidents;
    std::uint64_t queryWork=0,orderWork=0;
    bool complete=false,overflow=false;
    struct WorkReceipt {
        std::uint32_t incidentCount=0,orderWork=0;
        std::uint64_t queryWork=0; // Extra order and receiver-camera legs only.
        float absorbedPower=0,unresolvedPower=0;
        bool complete=false;
    };
    enum class Disposition : std::uint32_t { Escaped=0,Receiver=1,Unresolved=2,Invalid=3 };
    struct ChildReceipt {
        std::uint32_t incident=UINT32_MAX;
        std::int32_t order=0;
        Disposition disposition=Disposition::Invalid;
        opaquevisibility::ClosestHit hit;
        std::uint64_t queryWork=0;
        float escapedPower=0,unresolvedPower=0;
        bool complete=false;
    };
    std::vector<WorkReceipt> paths;
    std::vector<ChildReceipt> children;
    bool childrenComplete=false;
};
inline bool validateGratingIncidentReadback(const GratingIncidentBinding&,const NativeGratingReadback&,
    const SubmissionIdentity&,std::string&,const NativeDirectionalSourceBinding* =nullptr);
// Private native angular/visibility prerequisite. A local hit Jacobian does
// not prove finite receiving-cell coverage, so this version cannot grant Color.
struct GratingCameraBounds
{
    std::uint64_t maximumOrders=0,queryWork=0,combinedWork=0,bytes=0;
};
struct NativeGratingCameraReadback
{
    enum class Outcome : std::uint32_t { Visible=0,Occluded=1,OutsideCamera=2,Invalid=3 };
    struct OrderReceipt {
        std::uint32_t incident=UINT32_MAX,band=UINT32_MAX,x=UINT32_MAX,y=UINT32_MAX;
        std::int32_t order=0;
        Outcome outcome=Outcome::Invalid;
        opaquevisibility::ClosestHit cameraHit; // Actual native exact-order ray.
        RasterReceiver raster;
        float spectralRadiance=0; // Private per-order angular box average.
        bool complete=false;
    };
    std::uint32_t version=0;
    SubmissionIdentity submission;
    std::shared_ptr<const NativeGratingReadback> incidents;
    std::vector<OrderReceipt> orders; // Canonical incident, then signed order.
    std::uint64_t queryWork=0;
    bool physicalColor=false,complete=false,overflow=false;
};
static_assert(sizeof(NativeGratingCameraReadback::OrderReceipt)<=kGratingCameraOrderNativeBytes);
struct DepositionReceipt
{
    std::uint32_t version = 0;
    SubmissionIdentity submission;
    std::vector<DepositionKey> keys;
    std::uint64_t visibleCount = 0;
    bool complete = false, overflow = false;
};
struct NativeReadback
{
    std::string frameIdentity;
    std::shared_ptr<const NativeDirectionalSourceReceipt> directionalSource;
    std::vector<NativePathReceipt> paths;
    std::vector<NativeDeposit> deposits;
    // Completed native eight-band radiance, band-major within each top-first
    // pixel. The scratch is private. RGB conversion follows this atomic gate.
    std::vector<float> spectralRadiance;
    DepositionReceipt deposition;
    // No legacy producer writes these. Their explicit marker cannot be ignored
    // by a released receiver-rho validator or published as coupled Color.
    std::uint32_t gratingIncidentVersion = 0;
    std::shared_ptr<const NativeGratingReadback> gratingIncidents;
    std::uint32_t gratingCameraVersion = 0;
    std::shared_ptr<const NativeGratingCameraReadback> gratingCamera;
};
struct EnergyLedger
{
    std::array<double,8> potential {}, launched {}, sourceBlocked {}, deposited {}, absorbed {}, escaped {}, unresolved {}, cameraVisible {}, masked {};
};
inline bool validateGratingTerminalReadback(const GratingIncidentBinding&,const NativeReadback&,
    const SubmissionIdentity&,std::string&,const NativeDirectionalSourceBinding* =nullptr);
inline bool calculateGratingCameraBounds(const GratingIncidentBinding&,GratingCameraBounds&,std::string&);
inline bool validateGratingCameraReadback(const GratingIncidentBinding&,const NativeReadback&,
    const SubmissionIdentity&,std::string&,const NativeDirectionalSourceBinding* =nullptr);
inline bool nearPower(double a,double b,double tolerance=2e-4)
{return std::isfinite(a) && std::isfinite(b) && std::abs(a-b)<=tolerance*std::max({std::abs(a),std::abs(b),1e-20});}
inline bool nearDirectionalSourcePotential(double actual,double expected)
{
    if(!std::isfinite(actual) || !std::isfinite(expected) || actual<0 || expected<0)return false;
    const auto scale=std::max(actual,expected);
    return scale==0 || std::abs(actual-expected)/scale<=2e-4;
}
inline bool validateDirectionalSourceReadback(const Prepared& prepared,const Frame& frame,
    const NativeDirectionalSourceBinding& binding,const SubmissionIdentity& expected,
    const NativeDirectionalSourceReceipt& receipt,std::string& error,bool gratingPrerequisite=false,
    const GratingIncidentBinding* incidentBinding=nullptr)
{
    const auto& uploaded=binding.upload().submission;
    if(!validateDirectionalSourceUpload(prepared,frame,binding,uploaded,error,gratingPrerequisite,incidentBinding))return false;
    if(!validSourceSubmission(expected,frame,incidentBinding)
        ||expected.rendererGeneration!=uploaded.rendererGeneration ||expected.rasterGeneration!=uploaded.rasterGeneration
        ||expected.queryGeneration!=uploaded.queryGeneration ||expected.submissionSerial<uploaded.submissionSerial)
    {error="Directional source transport lost its retained native upload owner";return false;}
    NativeDirectionalSourceEcho exact;exact.input=directionalSourceInput(prepared,frame);
    for(unsigned band=0;band<8;++band)exact.rawRadiance[band]=frame.sourceRadiance[band/4][band%4];
    exact.directionTrace=frame.sourceDirectionTrace;
    if(receipt.version!=kDirectionalSourceReceiptVersion ||!receipt.complete ||receipt.overflow
        ||!(receipt.submission==expected) ||receipt.bindingGeneration!=binding.upload().bindingGeneration
        ||std::memcmp(&receipt.echo,&exact,sizeof(exact))!=0 ||receipt.potential.size()!=frame.work.size()
        ||receipt.potential.capacity()>2*frame.work.size())
    {error="Directional source shader echo or completed binding receipt changed";return false;}
    for(std::size_t work=0;work<frame.work.size();++work)
        if(!std::isfinite(receipt.potential[work]) ||receipt.potential[work]<0
            ||!nearDirectionalSourcePotential(receipt.potential[work],frame.work[work].incidentPower))
        {error="Directional source native per-entry potential differs from authored D flux";return false;}
    error.clear();return true;
}
inline bool sameTriangle(const opaquevisibility::ClosestHit& a,const opaquevisibility::ClosestHit& b)
{return a.objectId==b.objectId && a.materialId==b.materialId && a.instanceId==b.instanceId && a.triangleId==b.triangleId;}
inline bool exactCameraHit(const Frame& frame,const opaquevisibility::ClosestHit& receiver,const opaquevisibility::ClosestHit& hit)
{
    if(!receiverTriangle(frame,hit) || !sameTriangle(receiver,hit))return false;
    const auto delta=sub(receiver.position,cameraPosition(frame));
    const float distance=std::sqrt(dot(delta,delta)),tolerance=8*bias(receiver.position);
    return std::abs(hit.distance-distance)<=tolerance
        && dot(sub(hit.position,receiver.position),sub(hit.position,receiver.position))<=tolerance*tolerance;
}
// The caller retains the expected submission separately from GPU/readback
// storage. Its frame identity includes exact scene/material/light revisions,
// transforms, camera, branch policy and query bytes. Native tokens pin the
// renderer, raster, query, scratch and completed fence of this same submission.
inline bool validateDepositionReadback(const Frame& frame,const SubmissionIdentity& expected,
    const NativeReadback& readback,std::string& error)
{
    const auto refuse=[&](const char* message){error=message;return false;};
    const auto& receipt=readback.deposition;
    if(receipt.version!=kDepositionVersion || !receipt.complete || receipt.overflow
        || expected.frameIdentity!=frame.identity || expected.frameIdentity.empty()
        || !expected.rendererGeneration || !expected.submissionSerial || !expected.rasterGeneration
        || !expected.queryGeneration || !expected.scratchGeneration || !expected.completedFence
        || !(receipt.submission==expected) || readback.frameIdentity!=frame.identity)
        return refuse("Glass caustic deposition submission owner or completed fence changed");
    if(receipt.keys.size()!=frame.bounds.deposition.paddedKeyCount
        || receipt.visibleCount>readback.deposits.size() || readback.paths.size()!=frame.work.size())
        return refuse("Glass caustic deposition compacted key count exceeds admission");
    const auto visits=frame.work.empty()?0:frame.bounds.maximumDeposits/frame.work.size();
    if(!visits || visits>kMaximumBranchVisits)return refuse("Glass caustic deposition trace slot capacity changed");
    std::vector<std::size_t> offsets(frame.work.size()+1);
    std::uint64_t visible=0;
    for(std::size_t w=0;w<frame.work.size();++w)
    {
        const auto count=readback.paths[w].depositCount;
        if(count>visits || offsets[w]>readback.deposits.size() || count>readback.deposits.size()-offsets[w])
            return refuse("Glass caustic deposition path slot count changed");
        offsets[w+1]=offsets[w]+count;
        for(std::size_t slot=0;slot<count;++slot)
        {
            const auto& deposit=readback.deposits[offsets[w]+slot];
            if(deposit.work!=w)return refuse("Glass caustic deposition readback lost canonical trace slot order");
            if(deposit.cameraVisible)++visible;
        }
    }
    if(offsets.back()!=readback.deposits.size() || visible!=receipt.visibleCount)
        return refuse("Glass caustic deposition visible permutation is incomplete");
    for(std::size_t i=0;i<receipt.keys.size();++i)
    {
        const auto& key=receipt.keys[i];
        if(i>=receipt.visibleCount)
        {if(!depositionSentinel(key)){return refuse("Glass caustic deposition invalid suffix is not compacted");}continue;}
        if(key.pixel>=frame.bounds.deposition.pixelCount || key.work>=frame.work.size()
            || key.slot>=readback.paths[key.work].depositCount || key.flatIndex!=std::uint64_t(key.work)*visits+key.slot
            || (i && !(receipt.keys[i-1]<key)))
            return refuse("Glass caustic deposition key is reordered, duplicated or outside its trace slot");
        const auto& deposit=readback.deposits[offsets[key.work]+key.slot];
        if(!deposit.cameraVisible || !deposit.footprint.projected || deposit.footprint.x>=frame.width
            || deposit.footprint.y>=frame.height || key.pixel!=std::uint64_t(deposit.footprint.y)*frame.width+deposit.footprint.x)
            return refuse("Glass caustic deposition key does not retain its visible receiver pixel");
    }
    error.clear();return true;
}
inline bool validateNativeReadback(const Prepared& prepared,const Frame& frame,const NativeReadback& readback,
    EnergyLedger& output,std::string& error,const SubmissionIdentity* expectedSubmission=nullptr,bool gratingPrerequisite=false,
    const NativeDirectionalSourceBinding* sourceBinding=nullptr)
{
    const auto refuse=[&](const char* message){error=message;return false;};
    const bool grating=gratingPrerequisite&&frame.gratingIncidentBinding&&readback.gratingIncidentVersion==kGratingIncidentVersion
        &&readback.gratingIncidents&&expectedSubmission;
    if((readback.gratingIncidentVersion || readback.gratingIncidents || frame.gratingIncidentBinding
        || readback.gratingCameraVersion || readback.gratingCamera)&&!grating)
        return refuse("Glass to grating native incident readback is unavailable");
    if(!prepared.sceneLight.isValid() || !prepared.branchPolicy)
        return refuse("Native glass caustic deposition requires exact scene light and admitted branch policy");
    if(frame.sourceDirectionalIrradiance.version!=0) {
        if(!sourceBinding ||!expectedSubmission ||!readback.directionalSource)
            return refuse("Directional source native binding and shader receipt are required");
        if(!validateDirectionalSourceReadback(prepared,frame,*sourceBinding,*expectedSubmission,*readback.directionalSource,error,grating))return false;
    } else {
        if(sourceBinding ||readback.directionalSource)return refuse("Legacy radiance frame cannot carry a directional source grant");
        if(!validateTables(prepared,frame,error,grating))return false;
    }
    if(grating &&readback.directionalSource.get()!=readback.gratingIncidents->directionalSource.get())
        return refuse("Glass and grating transport must share one native directional source receipt");
    if(grating&&!validateGratingTerminalReadback(*frame.gratingIncidentBinding,readback,*expectedSubmission,error,sourceBinding))return false;
    if((readback.gratingCameraVersion || readback.gratingCamera)
        &&!validateGratingCameraReadback(*frame.gratingIncidentBinding,readback,*expectedSubmission,error,sourceBinding))return false;
    if(readback.frameIdentity!=frame.identity || readback.paths.size()!=frame.work.size()
        || readback.deposits.size()>frame.bounds.maximumDeposits
        || readback.spectralRadiance.size()!=std::uint64_t(frame.width)*frame.height*8)
        return refuse("Glass caustic native readback owner, count or extent changed");
    if((readback.deposition.version!=0 || expectedSubmission) && (!expectedSubmission
        || !validateDepositionReadback(frame,*expectedSubmission,readback,error)))
    {if(!expectedSubmission){error="Glass caustic deposition expected submission owner is missing";}return false;}
    // Sparse diagnostic sums validate the native accumulation. These values are
    // never handed to a renderer or returned as product Color.
    const auto& queryAllowance=frame.query.texels[std::size_t(frame.query.control[0])-1];
    const auto queryMaximum=std::uint64_t(queryAllowance[0])+(std::uint64_t(queryAllowance[1])<<16);
    struct PixelSum {double physical=0;float ordered=0;};
    std::map<std::uint64_t,PixelSum> expectedPixels;
    std::vector<double> outgoing(frame.work.size()),absorbed(frame.work.size()),fraction(frame.work.size());
    std::vector<std::size_t> counts(frame.work.size());
    std::vector<std::uint64_t> observedWork(frame.work.size());
    std::vector<std::uint32_t> minimumMicroEvents(frame.work.size());
    std::set<std::tuple<std::uint32_t,std::uint32_t,std::uint64_t,std::int32_t>> branches;
    std::set<std::tuple<std::uint32_t,std::uint32_t,std::array<std::uint8_t,32>>> microBranches;
    EnergyLedger candidate;
    if(grating)for(std::size_t w=0;w<frame.work.size();++w)absorbed[w]=readback.gratingIncidents->paths[w].absorbedPower;
    for(const auto& deposit:readback.deposits)
    {
        if(!grating&&(deposit.gratingVersion||deposit.gratingIncident!=UINT32_MAX||deposit.gratingOrder))
            return refuse("Glass caustic legacy deposit contains a private grating child");
        if(deposit.receiver.work>queryMaximum || deposit.cameraHit.work>queryMaximum || deposit.pixelHit.work>queryMaximum)
            return refuse("Glass caustic native closest-hit receipt exceeded its query budget");
        if(deposit.work>=frame.work.size() || !std::isfinite(deposit.fraction) || deposit.fraction<=0 || deposit.fraction>1
            || !deposit.interfaces || deposit.interfaces>prepared.branchPolicy->description().maximumInterfaces
            || (deposit.interfaces<64 && (deposit.reflectionHistory>>deposit.interfaces)!=0))
            return refuse("Glass caustic native deposit branch identity or fraction is invalid");
        if(frame.multipleScatterPolicy) {
            std::uint32_t events=0;
            if(!validMacroAncestry(deposit,*frame.multipleScatterPolicy,frame.ior[frame.work[deposit.work].band]==1,events)
                || !microBranches.emplace(deposit.work,deposit.interfaces,deposit.ancestry).second)
                return refuse("Glass caustic native exact microfacet ancestry is invalid or duplicated");
            minimumMicroEvents[deposit.work]=std::max(minimumMicroEvents[deposit.work],events);
        } else {
            if(deposit.gratingVersion) {
                if(!grating||deposit.gratingVersion!=kGratingIncidentVersion||deposit.gratingIncident>=readback.gratingIncidents->incidents.size())
                    return refuse("Glass caustic grating child owner is invalid");
            } else if(deposit.gratingIncident!=UINT32_MAX||deposit.gratingOrder||deposit.ancestry!=std::array<std::uint8_t,32>{})
                return refuse("Glass caustic legacy native branch identity or reserved ancestry changed");
            if(!branches.emplace(deposit.work,deposit.interfaces,deposit.reflectionHistory,
                deposit.gratingVersion?deposit.gratingOrder:INT32_MIN).second)
                return refuse("Glass caustic exact branch and signed order is duplicated");
        }
        for(float power:{deposit.incidentPower,deposit.outgoingPower,deposit.absorbedPower})
            if(!std::isfinite(power) || power<0)return refuse("Glass caustic native deposit power is invalid");
        const auto& work=frame.work[deposit.work];const auto* receiver=receiverTriangle(frame,deposit.receiver);
        if(!receiver)return refuse("Glass caustic native deposit receiver identity is stale");
        const double incident=double(work.incidentPower)*deposit.fraction,rho=receiver->reflectance->coefficients[work.band];
        if(!nearPower(deposit.incidentPower,incident) || !nearPower(deposit.outgoingPower,incident*rho)
            || !nearPower(deposit.absorbedPower,incident*(1-rho)))
            return refuse("Glass caustic native receiver Li/fraction/rho power is nonconserving");
        PixelFootprint footprint;if(!projectReceiver(frame,deposit.receiver,footprint,error))return false;
        if(deposit.footprint.projected!=footprint.projected || (footprint.projected
            && (deposit.footprint.x!=footprint.x || deposit.footprint.y!=footprint.y
                || !nearPower(deposit.footprint.worldArea,footprint.worldArea)
                || !nearPower(deposit.footprint.linearDepth,footprint.linearDepth))))
            return refuse("Glass caustic native projected pixel footprint changed");
        bool visible=false;
        if(footprint.projected)
        {
            if((deposit.cameraHit.status!=0 && deposit.cameraHit.status!=1) || (deposit.pixelHit.status!=0 && deposit.pixelHit.status!=1))
                return refuse("Glass caustic native camera visibility query is incomplete");
            const auto origin=cameraPosition(frame),delta=sub(deposit.receiver.position,origin);
            const float distance=std::sqrt(dot(delta,delta));
            if(deposit.cameraHit.status!=1 || !hitTriangle(frame,deposit.cameraHit) || deposit.cameraHit.distance>distance+8*bias(deposit.receiver.position))
                return refuse("Glass caustic native camera endpoint query is incomplete");
            auto endpointRay=delta;if(!normalize(endpointRay))return refuse("Glass caustic native camera endpoint ray is degenerate");
            const auto endpointResidual=sub(advance(origin,endpointRay,deposit.cameraHit.distance),deposit.cameraHit.position);
            if(dot(endpointResidual,endpointResidual)>std::pow(8*bias(deposit.cameraHit.position),2))
                return refuse("Glass caustic native camera blocker is off the endpoint ray");
            if(sameTriangle(deposit.receiver,deposit.cameraHit) && !exactCameraHit(frame,deposit.receiver,deposit.cameraHit))
                return refuse("Glass caustic native camera endpoint distance or position changed");
            const auto ray=cameraRay(frame,float(footprint.x)+.5f,float(footprint.y)+.5f);
            const auto expected=advance(origin,ray,deposit.pixelHit.distance);
            const auto residual=sub(expected,deposit.pixelHit.position);
            if(deposit.pixelHit.status==1 && (!hitTriangle(frame,deposit.pixelHit)
                || dot(residual,residual)>std::pow(8*bias(expected),2)))
                return refuse("Glass caustic native pixel camera hit is stale");
            visible=exactCameraHit(frame,deposit.receiver,deposit.cameraHit) && deposit.pixelHit.status==1
                && sameTriangle(deposit.receiver,deposit.pixelHit);
            if(visible)
            {
                const auto& r=deposit.raster;
                const auto& q=frame.camera.transform.rotation;
                const auto p=sub(deposit.pixelHit.position,origin);
                const auto local=rotateSceneVector({-q.x,-q.y,-q.z,q.w},{p[0],p[1],p[2]});
                if(r.object!=deposit.receiver.objectId || r.material!=deposit.receiver.materialId
                    || r.instance!=deposit.receiver.instanceId || r.triangle!=deposit.receiver.triangleId
                    || !nearPower(r.linearDepth,-local.z))
                    return refuse("Glass caustic native raster depth or exact triangle IDs changed");
                constexpr double pi=3.14159265358979323846;
                const double radiance=deposit.outgoingPower/(pi*footprint.worldArea);
                if(!std::isfinite(radiance) || radiance>surfacematerial::kMaximumLinearColorComponent)
                    return refuse("Glass caustic native pixel splat is unrepresentable");
                const auto pixel=(std::uint64_t(footprint.y)*frame.width+footprint.x)*8+work.band;
                auto& sum=expectedPixels[pixel];sum.physical+=radiance;
                // No regrouping by band, pairwise tree or fixed-point scaling.
                // The native segment leader uses these same float32 operations.
                const float nativeRadiance=deposit.outgoingPower/(float(pi)*deposit.footprint.worldArea);
                sum.ordered+=nativeRadiance;candidate.cameraVisible[work.band]+=deposit.outgoingPower;
            }
        }
        if(deposit.cameraVisible!=visible)return refuse("Glass caustic native camera-visible deposit mask changed");
        observedWork[deposit.work]+=deposit.receiver.work;
        if(footprint.projected)observedWork[deposit.work]+=std::uint64_t(deposit.cameraHit.work)+deposit.pixelHit.work;
        ++counts[deposit.work];outgoing[deposit.work]+=deposit.outgoingPower;
        absorbed[deposit.work]+=deposit.absorbedPower;fraction[deposit.work]+=deposit.fraction;
    }
    std::uint64_t totalQueryWork=0,totalMicroWork=0;
    const auto queryLimit=grating?frame.gratingIncidentBinding->bounds.queryWork:frame.bounds.queryWork;
    if(grating) {
        for(const auto& incident:readback.gratingIncidents->incidents)observedWork[incident.work]+=incident.hit.work;
        for(const auto& child:readback.gratingIncidents->children)
            if(child.disposition!=NativeGratingReadback::Disposition::Receiver)
                observedWork[readback.gratingIncidents->incidents[child.incident].work]+=child.hit.work;
    }
    for(std::size_t w=0;w<readback.paths.size();++w)
    {
        const auto& path=readback.paths[w];const auto& work=frame.work[w];
        const auto& entry=frame.entries[work.entry];
        const auto& sourceHit=path.sourceHit;
        if(path.sourceHit.work>queryMaximum || path.entryHit.work>queryMaximum)
            return refuse("Glass caustic native source/entry query exceeded its budget");
        if((sourceHit.status!=0 && sourceHit.status!=1) || path.sourceBlocked!=(sourceHit.status==1))
            return refuse("Glass caustic native source visibility receipt is incomplete");
        if(sourceHit.status==1)
        {
            const Point toSource{frame.sourceDirectionTrace[0],frame.sourceDirectionTrace[1],frame.sourceDirectionTrace[2]};
            const auto residual=sub(advance(entry.position,toSource,sourceHit.distance),sourceHit.position);
            if(!hitTriangle(frame,sourceHit) || sourceHit.distance<kRayMinimum || sourceHit.distance>frame.sourceDirectionTrace[3]
                || dot(residual,residual)>std::pow(8*bias(sourceHit.position),2))
                return refuse("Glass caustic native source blocker identity or ray changed");
        }
        else
        {
            const auto& hit=path.entryHit;const auto offset=bias(entry.position);
            if(hit.status!=1 || !hit.entering || !hitTriangle(frame,hit)
                || hit.objectId!=entry.object || hit.materialId!=entry.material || hit.instanceId!=entry.instance || hit.triangleId!=entry.triangle
                || hit.distance<0 || hit.distance>4*offset || std::abs(hit.distance-2*offset)>8*offset || dot(sub(hit.position,entry.position),sub(hit.position,entry.position))>64*offset*offset)
                return refuse("Glass caustic native entry endpoint identity changed");
        }
        const auto measured=observedWork[w]+path.sourceHit.work+(path.sourceBlocked?0:path.entryHit.work);
        if((!path.sourceBlocked && !path.visits) || (path.sourceBlocked && path.visits)
            || path.visits>prepared.branchPolicy->description().maximumVisits || path.visits<counts[w]
            || path.queryWork<measured || !path.queryWork || path.queryWork>(2+3*prepared.branchPolicy->description().maximumVisits)*queryMaximum
            || path.queryWork>queryLimit || totalQueryWork>queryLimit-path.queryWork)
            return refuse("Glass caustic native measured query work exceeds admission");
        totalQueryWork+=path.queryWork;
        if(frame.multipleScatterPolicy) {
            const auto& policy=*frame.multipleScatterPolicy;
            if(path.multipleScatterVersion!=kPrivateMultipleScatterVersion || path.quadrature!=entry.quadrature
                || path.maskedPower!=0 || path.microEvents>std::uint64_t(path.visits)*policy.maximumEvents
                || path.microEvents<minimumMicroEvents[w] || path.microDraws!=4*path.microEvents
                || (!path.sourceBlocked && frame.ior[work.band]!=1 && !path.microEvents)
                || (frame.ior[work.band]==1 && (path.microEvents || path.microDraws || path.microResidualPower!=0))
                || (path.sourceBlocked && (path.microEvents || path.microDraws || path.microResidualPower!=0))
                || !std::isfinite(path.microResidualPower) || path.microResidualPower<0
                || double(path.microResidualPower)>double(path.unresolvedPower)*(1+2e-4)
                || double(path.microResidualPower)>double(path.launchedPower)*path.visits*policy.maximumResidual*(1+2e-4))
                return refuse("Glass caustic native microfacet version, work or weighted residual receipt is invalid");
            const auto optical=std::uint64_t(path.microEvents)+path.microDraws;
            if(optical>frame.bounds.microfacetWork || totalMicroWork>frame.bounds.microfacetWork-optical)
                return refuse("Glass caustic native measured microfacet work exceeds admission");
            totalMicroWork+=optical;
            if(totalQueryWork>frame.bounds.combinedWork || totalMicroWork>frame.bounds.combinedWork-totalQueryWork)
                return refuse("Glass caustic native combined measured work exceeds admission");
        } else if(path.multipleScatterVersion || path.microEvents || path.microDraws || path.quadrature || path.microResidualPower!=0)
            return refuse("Glass caustic legacy reserved microfacet receipt changed");
        if((frame.roughness==0 && (path.maskedPower!=0 || path.roughTransportVersion!=0 || path.interfaceSample!=0))
            || (frame.roughness!=0 && (path.roughTransportVersion!=kRoughTransportVersion || path.interfaceSample!=work.interfaceSample)))
            return refuse("Glass caustic native rough sample receipt is missing or stale");
        if(!path.complete || path.work!=w || path.depositCount!=counts[w]
            || path.depositCount>prepared.branchPolicy->description().maximumVisits
            || !nearPower(path.launchedPower,path.sourceBlocked?0:work.incidentPower)
            || !nearPower(path.depositedPower,outgoing[w]) || !nearPower(path.absorbedPower,absorbed[w]))
            return refuse("Glass caustic native terminal path receipt is incomplete");
        for(float power:{path.launchedPower,path.depositedPower,path.absorbedPower,path.escapedPower,path.unresolvedPower,path.maskedPower})
            if(!std::isfinite(power) || power<0)return refuse("Glass caustic native terminal power is invalid");
        if((path.sourceBlocked && counts[w]) || fraction[w]>1.0002
            || !nearPower(path.launchedPower,double(path.depositedPower)+path.absorbedPower+path.escapedPower+path.unresolvedPower+path.maskedPower)
            || double(path.unresolvedPower)+path.maskedPower>double(path.launchedPower)*prepared.branchPolicy->description().maximumUnresolvedFraction*(1+2e-4))
            return refuse("Glass caustic native per-band energy or unresolved tail exceeds policy");
        if(grating) {
            const auto& coupling=readback.gratingIncidents->paths[w];
            if(coupling.queryWork>path.queryWork||coupling.unresolvedPower>path.unresolvedPower*(1+2e-4)
                ||(path.sourceBlocked&&coupling.incidentCount))return refuse("Glass caustic grating work or residual is not a subset of its original path");
        }
        const auto b=work.band;candidate.potential[b]+=work.incidentPower;candidate.launched[b]+=path.launchedPower;
        if(path.sourceBlocked)candidate.sourceBlocked[b]+=work.incidentPower;
        candidate.deposited[b]+=path.depositedPower;candidate.absorbed[b]+=path.absorbedPower;
        candidate.escaped[b]+=path.escapedPower;candidate.unresolved[b]+=path.unresolvedPower;candidate.masked[b]+=path.maskedPower;
    }
    const auto cameraQueries=readback.gratingCamera?readback.gratingCamera->queryWork:0;
    auto combinedLimit=grating?frame.gratingIncidentBinding->bounds.combinedWork:0;
    if(readback.gratingCamera) {
        GratingCameraBounds cameraBounds;
        if(!calculateGratingCameraBounds(*frame.gratingIncidentBinding,cameraBounds,error))return false;
        combinedLimit=cameraBounds.combinedWork;
    }
    if(grating&&(totalQueryWork>combinedLimit ||cameraQueries>combinedLimit-totalQueryWork
        ||readback.gratingIncidents->orderWork>combinedLimit-totalQueryWork-cameraQueries))
        return refuse("Glass caustic native combined query and diffraction work exceeds admission");
    for(std::size_t p=0;p<readback.spectralRadiance.size();++p)
    {
        const float actual=readback.spectralRadiance[p];const auto found=expectedPixels.find(p);
        const double expected=found==expectedPixels.end()?0:found->second.physical;
        const float ordered=found==expectedPixels.end()?0:found->second.ordered;
        // Native division may differ by a final rounding on GL and Metal. The
        // receipt still proves the unique reduction order and exact permutation.
        if(readback.deposition.version && !nearPower(actual,ordered,4e-6))
            return refuse("Glass caustic native canonical float32 reduction mismatch");
        if(!std::isfinite(actual) || actual<0 || actual>surfacematerial::kMaximumLinearColorComponent || !nearPower(actual,expected,5e-4))
            return refuse("Glass caustic native spectral accumulation overflow or receipt mismatch");
    }
    output=candidate;error.clear();return true;
}
// CPU-only power oracle companion. NativeReadback must come from completed GPU
// work; this diagnostic never manufactures it or populates an accumulator.
inline bool referencePower(const Prepared& prepared,const Frame& frame,EnergyLedger& output,std::string& error)
{
    if(!prepared.branchPolicy) {error="Glass caustic power diagnostic requires owning admitted branch policy";return false;}
    if(!validateTables(prepared,frame,error))return false;
    EnergyLedger candidate;
    for(const auto& work:frame.work)
    {
        BranchReceipt receipt;
        if(!traceBranches(prepared,frame,frame.entries[work.entry],work.band,*prepared.branchPolicy,receipt,error,work.interfaceSample))return false;
        const auto b=work.band;const double power=work.incidentPower;candidate.potential[b]+=power;
        if(receipt.sourceBlocked) {candidate.sourceBlocked[b]+=power;continue;}
        candidate.launched[b]+=power;candidate.escaped[b]+=power*receipt.escapedFraction;
        candidate.unresolved[b]+=power*receipt.unresolvedFraction;candidate.masked[b]+=power*receipt.maskedFraction;
        for(const auto& deposit:receipt.deposits)
        {
            const auto* triangle=receiverTriangle(frame,deposit.receiver);
            if(!triangle) {error="Glass caustic diagnostic receiver identity is stale";return false;}
            const double incident=power*deposit.fraction,rho=triangle->reflectance->coefficients[b];
            candidate.deposited[b]+=incident*rho;candidate.absorbed[b]+=incident*(1-rho);
        }
    }
    output=candidate;error.clear();return true;
}
// Native generators consume these exact production CIE/quad weights. Convert
// XYZ to scene-linear sRGB with the existing diffraction matrix, then perform
// the selected output transform at the existing Color boundary. Do not clamp
// individual bands or apply rho again in this resolve.
inline std::array<Texel,8> resolveWeights()
{
    std::array<Texel,8> result{};double whiteY=0;
    using namespace diffractionmaterial::physicalcheckpoint;
    for(unsigned b=0;b<8;++b)whiteY+=double(kQuadratureWeights[b])*kCie1931[b][1];
    for(unsigned b=0;b<8;++b)result[b]={kCie1931[b][0],kCie1931[b][1],kCie1931[b][2],float(kQuadratureWeights[b]/whiteY)};
    return result;
}
inline std::string nativeContractDefines()
{
    return "\n#define CAUSTIC_TABLE_VERSION "+std::to_string(kTableVersion)
        +"\n#define CAUSTIC_WORK_STRIDE "+std::to_string(kWorkStride)
        +"\n#define CAUSTIC_PATH_RECEIPT_STRIDE "+std::to_string(kPathReceiptStride)
        +"\n#define CAUSTIC_DEPOSIT_RECEIPT_STRIDE "+std::to_string(kDepositReceiptStride)
        +"\n#define CAUSTIC_MAX_PENDING "+std::to_string(kMaximumPendingBranches)
        +"\n#define CAUSTIC_MAX_VISITS "+std::to_string(kMaximumBranchVisits)
        +"\n#define CAUSTIC_MAX_INTERFACES "+std::to_string(kMaximumInterfaces)
        +"\n#define CAUSTIC_MAX_SPLATS "+std::to_string(kMaximumSplatsPerDeposit)+"\n";
}

// GLSL generator input shared by direct GL and the existing Metal translation
// lane. The caller supplies resolveWeights and uses sceneClosestHit directly.
// This source performs no scene query and introduces no second query authority.
inline std::string nativeMathSource()
{
    return nativeContractDefines()+R"GLSL(
bool causticFinite(float value) {return !isnan(value)&&!isinf(value);}
vec3 causticRotate(vec4 q,vec3 p) {return p+2.0*cross(q.xyz,cross(q.xyz,p)+q.w*p);}
vec3 causticCameraRay(vec2 pixel,vec4 rotation,vec4 projection,vec2 extent) {
    vec2 p=vec2(2.0*pixel.x/extent.x-1.0,1.0-2.0*pixel.y/extent.y);
    return normalize(causticRotate(rotation,vec3(p.x*projection.x*projection.y,p.y*projection.x,-1.0)));
}
bool causticPixelCorner(vec2 pixel,vec3 position,vec3 normal,vec3 camera,
    vec4 rotation,vec4 projection,vec2 extent,out vec3 corner) {
    vec3 ray=causticCameraRay(pixel,rotation,projection,extent);
    float denominator=dot(normal,ray),distance=dot(normal,position-camera)/denominator;
    if(!causticFinite(distance)||denominator>=-1e-6||distance<=0.0||distance>projection.w*1e4) return false;
    corner=camera+ray*distance;return true;
}
// 0 is outside/back-facing, -1 refuses a horizon/overflow footprint, 1 projects.
// footprint is top-first x/y, linear depth, exact receiver-plane pixel area.
int causticProjectReceiver(vec3 position,vec3 normal,vec3 camera,vec4 rotation,
    vec4 projection,vec2 extent,out vec4 footprint) {
    vec3 local=causticRotate(vec4(-rotation.xyz,rotation.w),position-camera);
    float depth=-local.z;footprint=vec4(0.0,0.0,depth,0.0);
    if(depth<projection.z||depth>projection.w) return 0;
    vec2 pixel=vec2((local.x/(depth*projection.x*projection.y)+1.0)*0.5*extent.x,
        (1.0-local.y/(depth*projection.x))*0.5*extent.y);
    if(!causticFinite(pixel.x)||!causticFinite(pixel.y)) return -1;
    if(pixel.x<0.0||pixel.y<0.0||pixel.x>=extent.x||pixel.y>=extent.y) return 0;
    pixel=floor(pixel);footprint.xy=pixel;
    if(dot(normal,causticCameraRay(pixel+vec2(0.5),rotation,projection,extent))>=-1e-6) return 0;
    vec3 a,b,c,d;
    if(!causticPixelCorner(pixel,position,normal,camera,rotation,projection,extent,a)
        ||!causticPixelCorner(pixel+vec2(1.0,0.0),position,normal,camera,rotation,projection,extent,b)
        ||!causticPixelCorner(pixel+vec2(1.0),position,normal,camera,rotation,projection,extent,c)
        ||!causticPixelCorner(pixel+vec2(0.0,1.0),position,normal,camera,rotation,projection,extent,d)) return -1;
    float area=0.5*length(cross(b-a,c-a))+0.5*length(cross(c-a,d-a));
    if(!causticFinite(area)||area<=0.0) return -1;
    footprint.w=area;return 1;
}
float causticEntryPower(float Li,vec3 outward,vec3 toSource,float area) {
    return Li*max(0.0,dot(outward,toSource))*area;
}
float causticFresnel(vec3 ray,vec3 normal,float from,float to,
    out vec3 transmitted,out vec3 reflected,out bool tir) {
    float cosine=clamp(-dot(ray,normal),0.0,1.0);
    reflected=normalize(ray+2.0*cosine*normal);tir=false;
    if(from==to) {transmitted=ray;return 0.0;}
    float ratio=from/to,sineSquared=ratio*ratio*max(0.0,1.0-cosine*cosine);
    tir=sineSquared>1.0;
    if(tir) {transmitted=vec3(0.0);return 1.0;}
    float exitCosine=sqrt(max(0.0,1.0-sineSquared));
    transmitted=normalize(ratio*ray+(ratio*cosine-exitCosine)*normal);
    float s=(from*cosine-to*exitCosine)/(from*cosine+to*exitCosine);
    float p=(to*cosine-from*exitCosine)/(to*cosine+from*exitCosine);
    return clamp(0.5*(s*s+p*p),0.0,1.0);
}
vec2 causticReceiverPower(float incident,float rho) {
    return vec2(incident*rho,incident*(1.0-rho));
}
float causticPixelRadiance(float outgoingPower,float receiverPixelArea) {
    return outgoingPower/(3.14159265358979323846*receiverPixelArea);
}
vec3 causticBandXyz(float radiance,vec4 cieWeight) {
    return radiance*cieWeight.w*cieWeight.xyz;
}
vec3 causticXyzToSceneLinear(vec3 xyz) {
    return vec3(3.2406*xyz.x-1.5372*xyz.y-0.4986*xyz.z,
        -0.9689*xyz.x+1.8758*xyz.y+0.0415*xyz.z,
        0.0557*xyz.x-0.2040*xyz.y+1.0570*xyz.z);
}
)GLSL";
}
// Reuse the released rough dielectric kernel verbatim through its existing
// GLSL-to-Metal translation lane. Smooth generators never call this function.
inline std::string nativeRoughMathSource()
{
    auto kernel=smoothglass::transportShader(false);
    const auto begin=kernel.find("vec3 glassRoughNormal(");
    const auto end=kernel.find("vec3 glassEnvironment(",begin);
    if(begin==std::string::npos || end==std::string::npos)return {};
    kernel=kernel.substr(begin,end-begin);
    for(const auto& names:std::array<std::pair<const char*,const char*>,6>{{
        {"glassRoughNormal","causticRoughNormal"},{"glassRoughG1","causticRoughG1"},
        {"glassRoughInterface","causticRoughInterface"},{"glassSceneFinite3","causticFinite3"},
        {"glassSceneFinite","causticFinite"},{"glassFresnel","causticRoughFresnel"}}})
    {
        std::size_t at=0;
        while((at=kernel.find(names.first,at))!=std::string::npos)
        {kernel.replace(at,std::strlen(names.first),names.second);at+=std::strlen(names.second);}
    }
    return nativeMathSource()+"\n#define CAUSTIC_ROUGH_VERSION "+std::to_string(kRoughTransportVersion)
        +"\n#define CAUSTIC_ROUGH_SAMPLES "+std::to_string(kRoughInterfaceSamples)+R"GLSL(
// Separate facet Fresnel from geometric branch power and origin bias.
bool causticFinite3(vec3 value) {return causticFinite(value.x)&&causticFinite(value.y)&&causticFinite(value.z);}
float causticRoughFresnel(vec3 incoming,vec3 normal,float fromIor,float toIor,out vec3 transmitted,out bool tir) {
    vec3 reflected;return causticFresnel(incoming,normal,fromIor,toIor,transmitted,reflected,tir);
}
vec2 causticRoughSample(uint sampleIndex,uint depth) {
    uint stratum=(sampleIndex*(2u*depth+1u)+5u*depth)%uint(CAUSTIC_ROUGH_SAMPLES);
    uint digits=stratum;float inverse=0.0,place=0.5;
    for(int bit=0;bit<4;++bit) {inverse+=float(digits&1u)*place;digits>>=1u;place*=0.5;}
    return vec2((float(stratum)+0.5)/float(CAUSTIC_ROUGH_SAMPLES),fract(inverse+float(depth)*0.6180339887498948482));
}
)GLSL"+kernel;
}
// Private v2 prerequisite: supplies math only, no frame/receipt/Color admission.
// The current v1 generators and reserved receipt texels remain unchanged.
inline std::string nativeMultipleScatterMathSource()
{
    auto kernel=smoothglass::multipleScatterShader();
    for(const auto& names:std::array<std::pair<const char*,const char*>,4>{{
        {"glassMs","causticMs"},{"glassSceneFinite3","causticFinite3"},
        {"glassSceneFinite","causticFinite"},{"glassFresnel","causticRoughFresnel"}}})
    {
        std::size_t at=0;
        while((at=kernel.find(names.first,at))!=std::string::npos)
        {kernel.replace(at,std::strlen(names.first),names.second);at+=std::strlen(names.second);}
    }
    // Reuse the released Fresnel authority and its existing finite helpers.
    return nativeRoughMathSource()+kernel;
}
static_assert(sizeof(NativePathReceipt)<=kPathReceiptStride*sizeof(Texel));
static_assert(sizeof(NativeDeposit)<=kDepositReceiptStride*sizeof(Texel));
} // namespace arbitgpu::glasscaustics

#include "fixture_glass_grating_incident.h"
