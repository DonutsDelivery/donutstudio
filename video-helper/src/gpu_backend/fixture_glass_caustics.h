#pragma once

#include "fixture_smooth_glass.h"
#include "fixture_diffraction_transport.h"
#include "glass_caustic_history.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <memory>
#include <tuple>
#include <unordered_map>

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
// Entries per frame traced by the native query. Glass with more entries is
// thinned by thinEntrySamples, which keeps its total area, so this sets caustic
// sampling density and not which meshes can cast.
inline constexpr std::size_t kMaximumEntrySamples = 128*kEntrySubdivisions*kEntrySubdivisions;
// Smooth macro paths keep a reflection history sized from the policy
// (glass_caustic_history.h), four inline words plus a tail past 128 interfaces.
// Multiple-scatter paths store one ancestry byte per interface: 32 inline in
// kMacroAncestryBytes and the rest in the tail (ResourceBounds). Grating
// incidents keep 32 inline tokens and a history tail sized from the policy.
// A branch policy is only bounded by these caps; policy 32 behaves as before.
// Smooth paths carry one history bit per interface in words sized from the
// policy (glass_caustic_history.h). The bound is the largest count deposit
// texel 0.z stores exactly as a float, not a scene limit; memory is the limit.
inline constexpr std::size_t kMaximumInterfaces = 16777215;
inline constexpr std::size_t kMaximumAncestryInterfaces = kMacroAncestryBytes;
inline constexpr std::size_t kDefaultPolicyInterfaces = 32;
static_assert(kMaximumAncestryInterfaces<=kMaximumInterfaces && kMaximumAncestryInterfaces==32,
    "Ancestry, camera-trace and grating layouts hold 32 inline tokens");
// Worst-case query work of one native frame is, per path, the source and entry
// queries plus three per visit (the visit and two camera queries per possible
// deposit), times the BVH allowance. It grows with the scene and assumes every
// query visits every BVH node and triangle. Work is not capped: this is the
// overflow bound, a quarter of 64 bits so the checked sums of up to four
// charges stay exact. Only a total past it refuses, for every frame kind.
inline constexpr std::uint64_t kMaximumQueryWork = 1ull<<62;
inline constexpr float kRayMinimum = 1.0e-5f;
inline constexpr std::uint32_t kBranchPolicyVersion = 1;
// Visits per path come from the branch policy. The bound is the largest count
// a receipt float stores exactly, not a scene limit; 64 is only the default.
inline constexpr std::size_t kMaximumBranchVisits = 16777215;
inline constexpr std::size_t kDefaultBranchVisits = 64;
// Incident locators and record counts are receipt floats too, so they share
// that exact bound whatever the visit count multiplies out to.
inline constexpr std::uint32_t kMaximumExactReceiptCount = 16777215;
// Pending branches per path come from the policy too. Both native walks keep the
// stack in the path tail, kPendingBranchTexels texels per entry sized from it, so
// the bound is the visit bound (a depth-first walk never holds more than visits+1),
// not a scene limit; 16 is only the default.
inline constexpr std::size_t kMaximumPendingBranches = kMaximumBranchVisits;
inline constexpr std::size_t kDefaultPendingBranches = 16;
inline constexpr std::size_t kPendingBranchTexels = 9;
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
    float maximumResidual = kMaximumUnresolvedFraction/(2*kDefaultBranchVisits);
};
inline bool validMultipleScatterPolicy(const MultipleScatterPolicy& policy)
{
    return policy.version==kPrivateMultipleScatterVersion && policy.maximumEvents>0
        && policy.maximumEvents<=smoothglass::kMaximumMicrofacetEvents && policy.maximumExits>0
        && policy.maximumExits<=smoothglass::kMaximumMicrofacetExits && std::isfinite(policy.maximumResidual)
        && policy.maximumResidual>0 && policy.maximumResidual<=kMaximumUnresolvedFraction;
}
// Count event evaluations and random draws as additional work under the same
// overflow bound. This does not admit queue expansion or a native frame.
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
    std::size_t maximumInterfaces = kDefaultPolicyInterfaces, maximumVisits = kDefaultBranchVisits;
    std::size_t maximumPending = kDefaultPendingBranches;
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
    // The thin camera dielectric beside a selected caustic volume. Its texel
    // owner flag is 2; photons that hit it are absorbed (see Prepared::absorber).
    bool absorber = false;
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
    // The caustic view of the smooth glass admission. When the admission has a
    // volume slot, prepareScene moves that volume into object, material,
    // transport and roughness, so every photon consumer traces the volume.
    smoothglass::Prepared glass;
    // The thin camera dielectric (the water) when glass came from a volume
    // slot, else invalid. A photon branch that hits it is absorbed: it ends
    // with no deposit, is not refused, and is reported as thin-absorbed power,
    // never as unresolved power. The water therefore receives no caustic.
    HarmonicMIDI::grid::SceneObjectId absorber {};
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
    // Each buffer is checked against the device's own buffer limit at its
    // allocation site (admitDepositionDevice, openGlCausticLimitFailure).
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
// Kernel splat deposition, version 1. It is off by default: a frame with
// splatVersion 0 keeps the released single-pixel keys, reduce and validator.
// A visible deposit spreads its outgoing flux over the receiver pixels in a
// fixed (2R+1)^2 window around its own pixel. Pixel p gets
//   L_p = outgoing * K_p / (pi * N),  N = sum over accepted q of K_q * A_q,
// so sum of L_p * pi * A_p is exactly the outgoing flux. K(d) = (1-d^2/r^2)^2
// for d < r, else 0. The deposit's own pixel always has K = 1, so N >= its area.
// Grating children keep single-pixel deposits. When the deposits of a traced
// frame overflow the tile deposit cap or the gather work cap, the frame falls
// back to single-pixel deposits: the consumer re-runs the deposition as the
// frame's single-pixel copy (disableSplat, the identity it had before
// enableSplat), and the CPU reference must overflow the same caps.
inline constexpr std::uint32_t kSplatVersion = 1;
// Entry samples land on the receiver about one cell side h apart, where the
// cell (triangle area / subdivisions^2) is half an h x h square. In the 4K
// lighthouse starter h is 30 to 35 pixels, so the window must reach about that
// far for neighbouring kernels to overlap. An 8 pixel cap left round spots.
inline constexpr std::uint32_t kMaximumSplatRadiusPixels = 32;
inline constexpr std::uint32_t kSplatWindow = 2*kMaximumSplatRadiusPixels+1;
// Key and gather tile. The GL gather still runs 16x16 work groups, so a tile
// is four of them; Metal dispatches per pixel.
inline constexpr std::uint32_t kSplatTile = 32;
inline constexpr std::uint32_t kSplatGatherGroup = 16;
inline constexpr std::uint32_t kSplatTilesPerAxis = 3;
inline constexpr std::uint32_t kSplatTilesPerDeposit = kSplatTilesPerAxis*kSplatTilesPerAxis;
static_assert(kSplatWindow<=(kSplatTilesPerAxis-1)*kSplatTile+1, "A splat window must span at most three tiles on each axis");
static_assert(kSplatTile%kSplatGatherGroup==0, "A splat tile must be whole gather groups");
// World radius: scale * sqrt(entry sample area / pi), clamped to between the
// minimum and kMaximumSplatRadiusPixels times the deposit pixel's scale s. s is
// the smallest singular value of the deposit pixel's receiver-plane Jacobian
// (the mean x and y edges of its footprint quad), so a world disc of radius r
// spans at most r/s pixels on screen in any direction. With r <= 32s the kernel
// reaches about 32 pixels from the deposit, while every pixel centre outside
// the fixed window is at least 32.5 pixels away. At ordinary pixel sizes the
// half pixel absorbs the perspective change across the window. A wide angle at
// low resolution can clip a tilted kernel's tail at the window edge; the
// native passes and the CPU reference clip it the same way, so flux is kept.
// Scale 2.5 makes r about h (r = 2.5 sqrt(h^2 / 2pi) = 0.997h). On the cell
// centroid lattice the summed kernels then vary by about 10%; scale 1.5
// (r = 0.6h) left the weakest point at 13% of the brightest.
inline constexpr double kSplatEntryRadiusScale = 2.5;
inline constexpr double kSplatMinimumRadiusPixels = 1.5;
inline constexpr float kSplatNormalAgreement = .9f;
// Grazing receivers. Window pixel positions on the centre ray carry a float32
// error that grows as 1/cos of the view angle at the receiver, and each pixel
// footprint stretches by the same factor. At cos 0.025, the underside of the
// lighthouse lantern at 4K, the footprint is 3.5 mm by 14 cm, the position
// error is 0.19 pixel, and the kernel touches one pixel row whose centres sit
// near its edge: the normaliser tolerance reached 0.11 of N. So a deposit's
// neighbour weights are scaled by its spread s, 0 at view cosine 0.1 or less,
// 1 at 0.2 or more and linear between. s=0 keeps a single-pixel deposit; the
// own pixel always weighs 1. s is continuous in the deposit's receipt
// position and normal, so the native float32 s differs from the binary64 one
// only by the cosine error, which kSplatSpreadCosineTolerance bounds.
inline constexpr double kSplatSpreadCosineLow = .1, kSplatSpreadCosineHigh = .2;
inline constexpr double kSplatSpreadCosineTolerance = 1e-5;
inline double splatSpread(double cosine)
{
    return std::clamp((cosine-kSplatSpreadCosineLow)/(kSplatSpreadCosineHigh-kSplatSpreadCosineLow),0.0,1.0);
}
// Gather work is the sum over tiles of (tile pixels x deposits keyed to the tile).
// A deposit keys at most nine tiles of 1024 pixels, so admission refuses a frame
// whose keyed deposits x 9216 exceed the gather cap. That bound is exact, so
// the native gather-work overflow is then unreachable. Only the first
// splatKeySlots(frame) slots of a path can hold a deposit, so a faceted lens
// with 24 sunlit triangles x 16 entries x 8 bands keys 3072 paths x 32 of its
// 64 visit slots = 98304 deposits, under the 116508 the cap admits. That also
// keeps the padded key count at 2^20, one OpenGL dispatch of 32768 groups.
// Measured gather work counts only visible deposits and stays far below the
// bound. Per-tile crowding depends on where deposits land; a crowded frame
// falls back to single-pixel deposits. A 32x32 tile collects deposits from a
// 96x96 pixel region, so its cap is four times the 1024 that a 16x16 tile with
// a 17x17 window had.
inline constexpr std::uint64_t kMaximumSplatDepositsPerTile = 4096;
inline constexpr std::uint64_t kMaximumSplatGatherWork = 1024ull*1024ull*1024ull;
// Normaliser and key work. The normaliser bound charges a 65x65 window per keyed
// deposit; native normaliser threads past a path's deposit count exit at once.
// The key bound is the oblivious bitonic sort over the padded keys. This budget
// is separate from kMaximumDepositionWork by design: the splat passes run in
// place of the single-pixel deposition, not on top of it.
inline constexpr std::uint64_t kMaximumSplatPassWork = 1024ull*1024ull*1024ull;
// Test hook. A nonzero value below kMaximumSplatGatherWork lowers the gather cap
// that the native host checks and the CPU reference apply after the trace, so a
// fixture can force the single-pixel fallback. Admission and the shaders keep
// kMaximumSplatGatherWork. Product code never sets it.
inline std::atomic<std::uint64_t>& splatGatherWorkCapOverride()
{
    static std::atomic<std::uint64_t> value{0};return value;
}
inline std::uint64_t splatGatherWorkCap()
{
    const auto value=splatGatherWorkCapOverride().load(std::memory_order_relaxed);
    return value && value<kMaximumSplatGatherWork?value:kMaximumSplatGatherWork;
}
// The CPU validator queries each distinct window pixel once. These budgets are
// enforced at readback, not admission: the distinct-pixel count depends on how
// many deposits are visible and how their windows overlap, and the worst-case
// product (maximumDeposits x 4225) would refuse ordinary 4K frames. A frame over
// either budget refuses as a whole and never publishes partly validated Color.
// The 4K faceted lens has about 1.5k distinct visible deposit spots in clusters
// of overlapping windows, an estimated 1 to 3.4 Mi window pixels, at about 40 to
// 80 measured closest-hit work each: 40 to 270 Mi in all, past the 128 Mi
// kMaximumQueryWork. 4 Mi pixels hold the validator's pixel map to about 600 MB.
// Work per pixel grows with the scene BVH and has no cap of its own.
inline constexpr std::uint64_t kMaximumSplatCpuPixelQueries = 4ull<<20;
// Largest admitted relative normaliser tolerance. Per-pixel footprint area
// tolerances alone reach about 3e-3 at 4K and up to the 1e-2 verification cap,
// so a 1e-3 bound would refuse ordinary frames. Twice that cap keeps the
// per-pixel 2 x relative term at most 4e-2, well below the value it bounds.
inline constexpr double kSplatMaximumNormaliserTolerance = 2e-2;
struct SplatSchedule
{
    std::uint64_t tilesX = 0, tilesY = 0, maximumKeys = 0, paddedKeyCount = 0, sortPasses = 0;
    std::uint64_t normaliserWork = 0, keyWork = 0, scratchBytes = 0, keySlots = 0;
    std::uint32_t sortLevels = 0;
};
inline bool operator==(const SplatSchedule& a,const SplatSchedule& b)
{
    return std::tie(a.tilesX,a.tilesY,a.maximumKeys,a.paddedKeyCount,a.sortPasses,a.normaliserWork,a.keyWork,a.scratchBytes,a.keySlots,a.sortLevels)
        ==std::tie(b.tilesX,b.tilesY,b.maximumKeys,b.paddedKeyCount,b.sortPasses,b.normaliserWork,b.keyWork,b.scratchBytes,b.keySlots,b.sortLevels);
}
// workCount paths each own visits trace slots, and only the first keySlots of
// them are keyed. Key i belongs to path (i/9)/keySlots, slot (i/9)%keySlots.
inline bool calculateSplatSchedule(std::uint32_t width,std::uint32_t height,std::uint64_t workCount,std::uint64_t visits,
    std::uint64_t keySlots,SplatSchedule& output,std::string& error)
{
    const auto refuse=[&](const char* text){error=text;return false;};
    if(!nativeFixtureDimensionsWithinBounds(width,height) || !workCount || !visits || visits>kMaximumBranchVisits
        || !keySlots || keySlots>visits || workCount>std::uint64_t(kMaximumEntrySamples)*8*kMaximumBranchVisits/visits)
        return refuse("Glass caustic splat extent or deposit capacity exceeds admission");
    const auto maximumDeposits=workCount*visits,keyedDeposits=workCount*keySlots;
    SplatSchedule candidate;candidate.keySlots=keySlots;
    candidate.tilesX=(width+kSplatTile-1)/kSplatTile;candidate.tilesY=(height+kSplatTile-1)/kSplatTile;
    if(candidate.tilesX*candidate.tilesY>=UINT32_MAX)return refuse("Glass caustic splat tile key cannot be represented");
    if(keyedDeposits>kMaximumSplatGatherWork/(std::uint64_t(kSplatTilesPerDeposit)*kSplatTile*kSplatTile))
        return refuse("Glass caustic splat worst-case gather work exceeds its budget");
    candidate.maximumKeys=keyedDeposits*kSplatTilesPerDeposit;candidate.paddedKeyCount=1;
    while(candidate.paddedKeyCount<candidate.maximumKeys){candidate.paddedKeyCount*=2;++candidate.sortLevels;}
    candidate.sortPasses=std::uint64_t(candidate.sortLevels)*(candidate.sortLevels+1)/2;
    candidate.normaliserWork=keyedDeposits*kSplatWindow*kSplatWindow;
    // Key fill and compaction, bitonic pairs, then one segment pass per tile.
    candidate.keyWork=2*candidate.paddedKeyCount+(candidate.paddedKeyCount/2)*candidate.sortPasses
        +candidate.tilesX*candidate.tilesY;
    if(candidate.normaliserWork+candidate.keyWork>kMaximumSplatPassWork)
        return refuse("Glass caustic splat normaliser and key work exceeds its budget");
    // Two key arrays, a begin/end pair per tile and one float normaliser per slot.
    candidate.scratchBytes=2*candidate.paddedKeyCount*sizeof(DepositionKey)
        +candidate.tilesX*candidate.tilesY*2*sizeof(std::uint32_t)+maximumDeposits*sizeof(float);
    // The key, tile and normaliser buffers are checked against the device's
    // buffer limit where they are allocated (Metal maxBufferLength, GL splat limits).
    output=candidate;error.clear();return true;
}
// Inclusive window of a deposit pixel, clipped to the image.
struct SplatWindowBounds {std::uint32_t x0=0,y0=0,x1=0,y1=0;};
inline SplatWindowBounds splatWindow(std::uint32_t width,std::uint32_t height,std::uint32_t x,std::uint32_t y)
{
    constexpr auto R=kMaximumSplatRadiusPixels;
    return {x>=R?x-R:0,y>=R?y-R:0,std::min(x+R,width-1),std::min(y+R,height-1)};
}
inline double splatKernel(double distance,double radius)
{
    if(!(radius>0) || !(distance>=0) || !(distance<radius))return 0;
    const double q=distance/radius,w=1-q*q;return w*w;
}
struct ResourceBounds
{
    std::uint64_t queryWork = 0, maximumDeposits = 0, transientBytes = 0;
    DepositionSchedule deposition;
    // Zero for legacy frames. Separate optical work consumes the same query
    // budget; temporary storage reserves every pending ancestry and exit array.
    std::uint64_t combinedWork = 0, microfacetWork = 0, microfacetTemporaryBytes = 0;
    // Smooth reflection history words 4 and up, sized from the branch policy's
    // interface count, and the path-receipt tail texels that hold them: one block
    // per deposit slot and maximumPending+1 branch blocks per path, then the
    // pending branch stack, kPendingBranchTexels per entry.
    std::uint64_t historyExtraWords = 0, historyTexels = 0;
    // Multiple-scatter ancestry tokens for interfaces 32 and up, four per word,
    // stored in the same tail blocks after the history words.
    std::uint64_t ancestryExtraWords = 0;
    // The branch policy's interface count, which sizes grating incident tails.
    std::uint64_t policyInterfaces = 0;
};
// Ancestry tail words for a multiple-scatter policy of this many interfaces.
inline std::uint64_t ancestryTailWords(std::size_t interfaces)
{
    return interfaces>kMacroAncestryBytes?(interfaces-kMacroAncestryBytes+3)/4:0;
}
// Texels of the path tail for this many paths, deposits, extra words and pending
// branches: the history blocks, two words per texel, then each path's pending
// stack. Refuses only a total past 64 bits.
inline bool historyTailTexels(std::uint64_t paths,std::uint64_t deposits,std::uint64_t extraWords,
    std::uint64_t pending,std::uint64_t& texels)
{
    constexpr auto maximum=std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t block=(extraWords+1)/2;
    if(pending>=maximum/kPendingBranchTexels || (paths && pending+1>maximum/paths))return false;
    const std::uint64_t branches=paths*(pending+1),stack=paths*pending*kPendingBranchTexels;
    if(deposits>maximum-branches || (block && deposits+branches>maximum/block))return false;
    const std::uint64_t history=(deposits+branches)*block;
    if(history>maximum-stack)return false;
    texels=history+stack;return true;
}
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
    // Reflection history words 2 and up per incident, sized from the policy
    // (gratingIncidentHistoryTailWords). Zero at 64 interfaces or fewer.
    std::uint64_t historyTailWords=0;
};
// Incident texel 1 holds history words 0 and 1. A policy past 64 interfaces
// adds words 2..historyWordsFor-1 after the incident records, as 16-bit halves
// like texel 1, so records of 64 interfaces or fewer keep their bytes.
constexpr std::size_t gratingIncidentHistoryTailWords(std::size_t interfaces)
{
    return interfaces>64?historyWordsFor(interfaces)-2:0;
}
constexpr std::size_t gratingIncidentHistoryTailTexels(std::size_t interfaces)
{
    return (gratingIncidentHistoryTailWords(interfaces)+1)/2;
}
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
    // Zero keeps single-pixel deposition. enableSplat sets kSplatVersion.
    std::uint32_t splatVersion=0;
    SplatSchedule splat;
};
// Keyed slots per path. Deposits fill a path's slots in order, and a deposit
// ends its branch. Each glass visit stacks at most two children, so V visits
// leave at most (V+1)/2 deposits. Multiple-scatter visits can stack more
// children, so those frames key every slot.
inline std::uint64_t splatKeySlots(const Frame& frame)
{
    if(frame.work.empty())return 0;
    const auto visits=frame.bounds.maximumDeposits/frame.work.size();
    const auto slots=frame.multipleScatterPolicy?visits:(visits+1)/2;
    // Key only the slots the dispatch-pinned gather budget admits: keyed
    // deposits x kSplatTilesPerDeposit x kSplatTile^2 must stay under
    // kMaximumSplatGatherWork, which is exactly what keeps the admitted sort
    // inside one 2^20 key dispatch. The authored lighthouse dome pushed the
    // design's 24 sunlit triangles to 30 - 3840 paths x 32 slots keyed to
    // 122880 against a 116508 cap - so every frame of that scene fell to
    // single-pixel deposits and its caustic coverage check failed. A closed
    // dome-and-water trace physically cannot make the deposits a clamped path
    // would need to reach the refused slots, and a frame that somehow does
    // refuses honestly instead of dispatching an oversized sort. Frames at or
    // under the design envelope keep their full slot count unchanged.
    const auto keyedCap=kMaximumSplatGatherWork
        /(std::uint64_t(kSplatTilesPerDeposit)*kSplatTile*kSplatTile);
    const auto byBudget=keyedCap/frame.work.size();
    return byBudget && byBudget<slots ? byBudget : slots;
}
inline bool calculateSplatSchedule(const Frame& frame,SplatSchedule& output,std::string& error)
{
    const auto workCount=std::uint64_t(frame.work.size());
    if(!workCount || frame.bounds.maximumDeposits%workCount)
    {error="Glass caustic splat extent or deposit capacity exceeds admission";return false;}
    return calculateSplatSchedule(frame.width,frame.height,workCount,frame.bounds.maximumDeposits/workCount,
        splatKeySlots(frame),output,error);
}
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
        || depth>=kMaximumAncestryInterfaces || !finite(incoming) || !finite(normal)
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
    const auto& slot=glass.causticVolume;
    const auto causticTransport=slot.present?surfacematerial::TransportMode::ClosedVolume:glass.transport;
    if (!glass.enabled || causticTransport!=surfacematerial::TransportMode::ClosedVolume
        || !glass.sourceScene || !glass.sourceMaterial
        || !validateVisual3DScene(*glass.sourceScene).valid()
        || !validNativeFixtureMaterialPrograms(*glass.sourceMaterial,*glass.sourceScene,glass.sourceMaterial->backend,
            bool(glass.distinctGrating),glass.distinctGrating?NativeFixtureDiffractionConsumer::SmoothGlassDiscreteOrders
                :NativeFixtureDiffractionConsumer::FiniteLobe))
        return refuse("Glass caustics require owning admitted closed-volume geometry");
    // A camera-only slot was prepared with Glass Caustics off. It owns no
    // caustic selection, so a caustic pass must come from a fresh caustic prepare.
    if (slot.present && slot.cameraOnly)
        return refuse("Glass caustics require a caustic volume selection, not a camera-only volume");
    Prepared candidate;
    // Re-admit snapshots rather than trusting a caller-supplied query or mutable alias.
    const auto scene=std::make_shared<const Visual3DScene>(*glass.sourceScene);
    const auto root=snapshotNativeSurfaceProgram(glass.sourceMaterial);
    // A rough camera dielectric and a rough volume share the one native consumer.
    const auto consumer=glass.roughness!=0?glass.roughConsumer:slot.roughConsumer;
    const smoothglass::CausticVolumeSelection selection{slot.object,slot.material};
    if (!smoothglass::prepare(scene,root,{},candidate.glass,error,consumer,glass.distinctGrating.get(),
            slot.present?&selection:nullptr)) return false;
    if (candidate.glass.smoothReceiverNormals)
        return refuse("Closed glass receivers require flat geometric vertex normals");
    if (slot.present!=candidate.glass.causticVolume.present)
        return refuse("Glass caustic volume slot changed on re-admission");
    if (candidate.glass.causticVolume.present)
    {
        auto& view=candidate.glass;
        const auto volume=view.causticVolume;
        candidate.absorber=view.object;
        view.transport=surfacematerial::TransportMode::ClosedVolume;
        view.object=volume.object;view.material=volume.material;
        view.roughness=volume.roughness;view.roughConsumer=volume.roughConsumer;
    }
    candidate.identity=std::move(identity);
    const auto& sceneRoot=*candidate.glass.sourceMaterial;
    if (!sceneRoot.sceneRevision || !sceneRoot.structuralRevision || !sceneRoot.evaluationRevision)
        return refuse("Glass caustic scene/material revisions are missing");
    word(candidate.identity,scene->id.value);word(candidate.identity,sceneRoot.sceneRevision);
    std::size_t receivers=0;
    // The first refusal is returned. With ARBIT_GLASS_ADMISSION_AUDIT=1 the scan
    // continues so every object reports its first refusal to stderr.
    std::string firstRefusal;
    for(std::size_t o=0;o<scene->objectCount;++o)
    {
        const auto& object=scene->objects[o];
        if (!object.vertexCount && !object.indexCount) continue;
        const auto stop=[&](const char* text) {
            if (firstRefusal.empty()) firstRefusal=text;
            if (!smoothglass::admissionAudit()) return true;
            smoothglass::auditLog("caustic prepareScene object index "+std::to_string(o)+" id "
                +std::to_string(object.id.value)+": "+text);
            return false;
        };
        const auto* surface=nativeSurfaceForObject(root.get(),object.id);
        const bool grating=candidate.glass.distinctGrating&&object.id==candidate.glass.distinctGrating->object;
        if (!surface || surface->authoredScene!=scene->id || surface->authoredMaterial!=object.material
            || (!grating&&surface->parameters.identifiers[0]!=object.material.value) || surface->sceneRevision!=sceneRoot.sceneRevision
            || surface->structuralRevision!=sceneRoot.structuralRevision
            || surface->evaluationRevision!=sceneRoot.evaluationRevision
            || surface->materialRevision!=surface->structuralRevision)
        {
            if (stop("Glass caustic exact object/material/revision owner is stale or missing")) return refuse(firstRefusal.c_str());
            continue;
        }
        const bool volume=object.id==candidate.glass.object;
        const bool absorber=candidate.absorber.isValid()&&object.id==candidate.absorber;
        if (volume && surface->spectralReflectance)
        {
            if (stop("Glass caustic volume spectral absorption is unsupported")) return refuse(firstRefusal.c_str());
            continue;
        }
        if (!volume&&!grating&&!absorber)
        {
            if (!surface->spectralReflectance || !surfacematerial::validSpectralReflectance(*surface->spectralReflectance))
            {
                if (stop("Glass caustic diffuse receiver requires exact authored eight-band rho")) return refuse(firstRefusal.c_str());
                continue;
            }
            if (surface->parameters.emissionRoughness[0]!=0 || surface->parameters.emissionRoughness[1]!=0
                || surface->parameters.emissionRoughness[2]!=0)
            {
                if (stop("Glass caustic receiver emission is unsupported")) return refuse(firstRefusal.c_str());
                continue;
            }
            ++receivers;
        }
        word(candidate.identity,object.id.value,8);word(candidate.identity,object.material.value,8);
        word(candidate.identity,surface->structuralRevision);word(candidate.identity,surface->evaluationRevision);
        word(candidate.identity,surface->materialRevision);word(candidate.identity,surface->programIdentity.size());
        candidate.identity+=surface->programIdentity;
        for(const auto* block:{&surface->parameters.baseColorMetallic,&surface->parameters.emissionRoughness,
            &surface->parameters.normalOpacity,&surface->parameters.transmissionIorClearcoat})for(float v:*block)bits(candidate.identity,v);
        if (!volume&&!grating&&!absorber)for(float rho:surface->spectralReflectance->coefficients)bits(candidate.identity,rho);
        if(grating)candidate.identity+=candidate.glass.distinctGrating->material->structuralDigest();
    }
    if (!firstRefusal.empty()) return refuse(firstRefusal.c_str());
    if (!receivers&&!candidate.glass.distinctGrating) return refuse("Glass caustics require an ordinary authored spectral diffuse receiver");
    // Single-dielectric identities keep their released encoding.
    if (candidate.absorber.isValid()) {candidate.identity+="thin-absorber:";word(candidate.identity,candidate.absorber.value,8);}
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
    output.push_back({float(t.instance),float(t.triangle),t.glass?1.0f:t.absorber?2.0f:0.0f,t.reflectance?float(t.reflectance->version):0.0f});
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
// Keeps an even stride of entries across the glass when a mesh yields more than
// the native query traces. Each kept entry stands for the area of those it
// replaced, so the deposited energy is preserved and the caustic gets noisier.
inline void thinEntrySamples(std::vector<EntrySample>& entries)
{
    if(entries.size()<=kMaximumEntrySamples)return;
    const auto stride=(entries.size()+kMaximumEntrySamples-1)/kMaximumEntrySamples;
    std::size_t kept=0;
    for(std::size_t e=0;e<entries.size();e+=stride)
    {
        entries[kept]=entries[e];entries[kept].area*=float(stride);++kept;
    }
    entries.resize(kept);
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
        runtime->cameraRotationDegrees,runtime->cameraVerticalFovDegrees,runtime->cameraLens);
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
    // Frames of 128 or fewer interfaces keep their identity unchanged.
    if(frame.bounds.historyExtraWords)word(identity,frame.bounds.historyExtraWords);
    if(frame.bounds.ancestryExtraWords)word(identity,frame.bounds.ancestryExtraWords);
    if(frame.multipleScatterPolicy) {
        word(identity,frame.bounds.combinedWork);word(identity,frame.bounds.microfacetWork);word(identity,frame.bounds.microfacetTemporaryBytes);
    }
    const auto& deposition=frame.bounds.deposition;word(identity,kDepositionVersion);
    for(auto value:{deposition.pixelCount,deposition.maximumDeposits,deposition.paddedKeyCount,deposition.sortPasses,
        deposition.depositionWork,deposition.resolveWork,deposition.scratchBytes,deposition.largestBufferBytes})word(identity,value);
    word(identity,deposition.sortLevels);
    if(frame.splatVersion) {
        word(identity,frame.splatVersion);
        for(auto value:{frame.splat.tilesX,frame.splat.tilesY,frame.splat.maximumKeys,frame.splat.paddedKeyCount,frame.splat.sortPasses,
            frame.splat.normaliserWork,frame.splat.keyWork,frame.splat.scratchBytes,frame.splat.keySlots})word(identity,value);
        word(identity,frame.splat.sortLevels);
    }
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
// enableSplat appends the splat version, nine schedule values and the sort
// level count to the identity, each as one 16-digit word.
inline constexpr std::size_t kSplatIdentityBytes = 11*16;
// Identity bytes a frame reserves. It is the splat-enabled length whether or
// not the frame splats, so enableSplat and disableSplat keep the admitted
// reservation valid.
inline std::size_t reservedFrameIdentityBytes(const Prepared& prepared,const Frame& frame)
{
    const auto size=frameIdentity(prepared,frame).size();
    return frame.splatVersion?size:size+kSplatIdentityBytes;
}
inline bool calculateResourceBounds(const Prepared& prepared,const Frame& frame,ResourceBounds& output,std::string& error)
{
    const auto refuse=[&](std::string text){error=text;return false;};
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
    const auto visits=prepared.branchPolicy?prepared.branchPolicy->description().maximumVisits:kDefaultBranchVisits;
    // Per path, source/entry plus visits and two camera queries per possible
    // deposit. CPU diagnostic traversals check their own budget when they run.
    ResourceBounds candidate;
    const auto queryCount=std::uint64_t(frame.work.size())*(2+3*std::uint64_t(visits));
    if(!work || work>opaquevisibility::kMaximumClosestHitWork)
        return refuse("Glass caustic native query allowance "+std::to_string(work)+" is invalid");
    if(queryCount>std::numeric_limits<std::uint64_t>::max()/work)
        return refuse("Glass caustic native path/camera query work does not fit 64 bits: "
            +std::to_string(queryCount)+" queries at "+std::to_string(work)+" work");
    candidate.queryWork=queryCount*work;candidate.maximumDeposits=std::uint64_t(frame.work.size())*visits;
    candidate.policyInterfaces=prepared.branchPolicy?prepared.branchPolicy->description().maximumInterfaces:0;
    candidate.historyExtraWords=historyWordsFor(candidate.policyInterfaces)-kHistoryWords;
    if(frame.multipleScatterPolicy && prepared.branchPolicy)
        candidate.ancestryExtraWords=ancestryTailWords(prepared.branchPolicy->description().maximumInterfaces);
    const std::uint64_t pendingBranches=prepared.branchPolicy?prepared.branchPolicy->description().maximumPending:0;
    if(!historyTailTexels(frame.work.size(),candidate.maximumDeposits,
        candidate.historyExtraWords+candidate.ancestryExtraWords,pendingBranches,candidate.historyTexels))
        return refuse("Glass caustic path tail for "+std::to_string(pendingBranches)+" pending branches does not fit 64 bits");
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
    // The sum only has to stay representable: each native buffer meets the
    // device's own allocation limit where it is created.
    const auto add=[&](std::uint64_t count,std::uint64_t stride) {
        if(count>UINT64_MAX/stride
            || candidate.transientBytes>UINT64_MAX-count*stride)return false;
        candidate.transientBytes+=count*stride;return true;
    };
    if(!add(std::uint64_t(frame.width)*frame.height,2*kAccumulatorBytesPerPixel)
        || !add(frame.workTexels.size(),sizeof(Texel)) || !add(frame.triangleTexels.size(),sizeof(Texel))
        || !add(frame.query.texels.size(),sizeof(Texel)) || !add(frame.work.size(),2*kPathReceiptStride*sizeof(Texel))
        || !add(candidate.maximumDeposits,2*kDepositReceiptStride*sizeof(Texel))
        || !add(candidate.historyTexels,2*sizeof(Texel))
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
                || !add(frame.work.size(),5*sizeof(float)) || !add(reservedFrameIdentityBytes(prepared,frame),2)))
        || (frame.multipleScatterPolicy && !add(candidate.microfacetTemporaryBytes,1))
        // Frame, native readback, expected submission and completed receipt
        // each retain this exact owner. Bounds values have fixed-width identity
        // encoding, so the size is independent of this reservation's value.
        // The splat words are always reserved: admission bounds a single-pixel
        // frame, and enableSplat keeps those bounds while lengthening the identity.
        || !add(reservedFrameIdentityBytes(prepared,frame),4))
        return refuse("Glass caustic native scratch total overflows 64 bits");
    output=candidate;error.clear();return true;
}
inline bool admitFrameMemory(const Frame& frame,std::uint64_t otherOwnedBytes,std::uint64_t& output,std::string& error)
{
    // A frame total only has to stay representable; the device refuses an
    // allocation it cannot hold where that allocation is made.
    if(otherOwnedBytes>UINT64_MAX-frame.bounds.transientBytes)
    {error="Glass caustic scratch plus retained native frame resources overflows 64 bits";return false;}
    output=otherOwnedBytes+frame.bounds.transientBytes;error.clear();return true;
}
inline bool validateTables(const Prepared& prepared,const Frame& frame,std::string& error,bool gratingPrerequisite=false,
    SourceMeasureUse sourceUse=SourceMeasureUse::LegacyNative);
inline bool validateQueries(const Prepared& prepared,const Frame& frame,std::string& error);
inline bool buildFrameImpl(const Prepared& prepared, const std::vector<opaquevisibility::Instance>& instances,
    const NativeFixtureSceneRuntimeInputs* runtime,std::uint32_t width,std::uint32_t height,Frame& output,std::string& error,
    SchedulingDescription scheduling={},std::optional<MultipleScatterPolicy> multipleScatterPolicy={},bool gratingPrerequisite=false)
{
    const auto refuse=[&](std::string text){error=text;return false;};
    if (!prepared.glass.sourceScene || !prepared.glass.sourceMaterial || prepared.identity.empty()
        || !nativeFixtureDimensionsWithinBounds(width,height) || !validScheduling(scheduling))return refuse("Glass caustic frame has no admitted ownership or extent");
    const auto& scene=*prepared.glass.sourceScene;
    std::vector<char> seen(scene.objectCount,0);
    Frame candidate;candidate.width=width;candidate.height=height;candidate.scheduling=scheduling;candidate.instances=instances;
    if(prepared.glass.distinctGrating&&!gratingPrerequisite)return refuse("Distinct grating requires guarded native staging admission");
    candidate.gratingStaging=bool(prepared.glass.distinctGrating);
    candidate.roughness=prepared.glass.roughness;
    candidate.interfaceSamples=candidate.roughness==0?1:kRoughInterfaceSamples;
    candidate.multipleScatterPolicy=multipleScatterPolicy;
    if(multipleScatterPolicy && (!prepared.branchPolicy || candidate.roughness<=0 || !validMultipleScatterPolicy(*multipleScatterPolicy)))
        return refuse("Glass caustic multiple-scatter policy is unsupported");
    // The absorber rule exists in the smooth CPU, GL and Metal tracers. Rough
    // volumes also need the camera rough schedule to follow the volume, not
    // the thin camera dielectric, so they stay refused here.
    // Grating incidents carry the full reflection history in a tail sized from
    // the policy; ancestry tokens past 32 follow from that history.
    if(prepared.absorber.isValid() && (candidate.roughness!=0 || multipleScatterPolicy || prepared.glass.distinctGrating))
        return refuse("Glass caustics beside a thin camera dielectric require a smooth closed volume");
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
        if (!std::isfinite(candidate.ior[b]))
            return refuse("Glass caustic eight-band IOR is not finite");
    }
    for(std::size_t i=0;i<instances.size();++i)
    {
        const auto& placement=instances[i];
        if (placement.object>=scene.objectCount || seen[placement.object])return refuse("Glass caustics refuse duplicated or foreign instances");
        // Triangle instance IDs are i, and the native rasters write the scene
        // object index as the instance ID (backend_sokol.mm caustic raster), so
        // the splat owner lookup needs instance i to be object i.
        if (placement.object!=i)return refuse("Glass caustic instances must list scene objects in order");
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
            triangle.absorber=prepared.absorber.isValid()&&object.id==prepared.absorber;
            triangle.instance=std::uint32_t(i);triangle.triangle=std::uint32_t(t);
            if (!glass&&!triangle.absorber)triangle.reflectance=surface->spectralReflectance;
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
    thinEntrySamples(candidate.entries);
    const auto root=std::size_t(candidate.query.control[0]);
    if (!root || root>=candidate.query.texels.size())
        return refuse("Glass caustic query table is empty");
    const auto& allowance=candidate.query.texels[root-1];
    const auto work=std::uint64_t(allowance[0])+(std::uint64_t(allowance[1])<<16);
    // The native per-path bound calculateResourceBounds reserves. The CPU
    // diagnostic traces are not admitted here; they check their own budget.
    const auto visits=prepared.branchPolicy?prepared.branchPolicy->description().maximumVisits:kDefaultBranchVisits;
    const auto queries=std::uint64_t(candidate.entries.size())*8*candidate.interfaceSamples*(2+3*std::uint64_t(visits));
    if (!work)return refuse("Glass caustic finite path query allowance is invalid");
    if (queries>std::numeric_limits<std::uint64_t>::max()/work)
        return refuse("Glass caustic finite path query work does not fit 64 bits: "
            +std::to_string(queries)+" queries at "+std::to_string(work)+" work");
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
    // The transmitted path ended on the thin camera dielectric and was absorbed.
    bool absorberHit = false;
    opaquevisibility::ClosestHit receiver;
    float reflectedFraction = 0, transmittedFraction = 1;
    std::size_t interfaces = 0;
};
// Diagnostic CPU traces walk and budget the owning policy's interface bound,
// so a policy of 32 keeps the same work as before the cap rose to 48.
inline std::size_t policyInterfaces(const Prepared& prepared)
{
    return prepared.branchPolicy?prepared.branchPolicy->description().maximumInterfaces:kDefaultPolicyInterfaces;
}
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
    for(std::size_t depth=0,limit=policyInterfaces(prepared);depth<limit;++depth)
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
            if(prepared.absorber.isValid() && hit.objectId==prepared.absorber.value)
            {candidate.absorberHit=true;candidate.receiver=hit;output=candidate;error.clear();return true;}
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
        || bounds.historyExtraWords!=frame.bounds.historyExtraWords || bounds.historyTexels!=frame.bounds.historyTexels
        || bounds.ancestryExtraWords!=frame.bounds.ancestryExtraWords || bounds.policyInterfaces!=frame.bounds.policyInterfaces
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
    const auto& scene=*prepared.glass.sourceScene;std::size_t expectedTriangles=0;std::vector<char> seen(scene.objectCount,0);
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
            || triangle.absorber!=(prepared.absorber.isValid()&&object.id==prepared.absorber)
            || bool(triangle.reflectance)!=(!triangle.absorber&&bool(surface->spectralReflectance))
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
    thinEntrySamples(exactEntries);
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
// CPU diagnostic traces share the overflow bound of native frames. queries is
// the trace's worst-case count.
inline bool diagnosticQueryWorkWithinBudget(const Frame& frame,std::uint64_t queries,std::string& error)
{
    const auto root=std::size_t(frame.query.control[0]);
    if(!root || root>=frame.query.texels.size())
    {error="Glass caustic diagnostic query topology is incomplete";return false;}
    const auto& allowance=frame.query.texels[root-1];
    const auto work=std::uint64_t(allowance[0])+(std::uint64_t(allowance[1])<<16);
    if(!work || queries>kMaximumQueryWork/work)
    {error="Glass caustic CPU diagnostic query work exceeds budget";return false;}
    return true;
}
// Diagnostic CPU traversal only. Native consumers use validateTables followed
// by their own sceneClosestHit paths and completed native receipt validation.
inline bool validateQueries(const Prepared& prepared,const Frame& frame,std::string& error)
{
    if(frame.multipleScatterPolicy) {error="Private multiple-scatter frames require an exact native consumer";return false;}
    if(!validateTables(prepared,frame,error))return false;
    if(!diagnosticQueryWorkWithinBudget(frame,std::uint64_t(frame.entries.size())*8*(policyInterfaces(prepared)+3),error))return false;
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
    CausticHistory reflectionHistory {};
    std::size_t interfaces = 0, reflections = 0;
    unsigned band = 0;
};
struct BranchInterfaceReceipt
{
    opaquevisibility::ClosestHit surface;
    Point incomingDirection {};
    double incidentFraction = 0, reflectedFraction = 0, transmittedFraction = 0;
    CausticHistory reflectionHistory {};
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
    // Branches ended on the thin camera dielectric. A separate energy term,
    // never part of the unresolved budget.
    double absorberFraction = 0;
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
        CausticHistory history;
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
            // Water absorbs from either side, so facing is not checked. Reaching
            // it from inside the volume is overlapping geometry and still refuses.
            if(prepared.absorber.isValid() && hit.objectId==prepared.absorber.value)
            {
                if(path.inside)
                {candidate.unresolvedFraction+=path.fraction;return refuse("Glass caustic reflected branch hit unsupported interior/back-face coverage");}
                candidate.absorberFraction+=path.fraction;continue;
            }
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
                fraction,path.history.with(path.interfaces,reflection),path.interfaces+1,
                path.reflections+(reflection?1:0),reflection?path.inside:!path.inside});
        };
        append(optics.reflected,reflected,true);
        if(!optics.tir)append(optics.transmitted,transmitted,false);
    }
    for(const auto& path:pending)candidate.unresolvedFraction+=path.fraction;
    candidate.withinErrorBound=std::isfinite(candidate.depositedFraction) && std::isfinite(candidate.escapedFraction)
        && std::isfinite(candidate.unresolvedFraction) && std::isfinite(candidate.maskedFraction)
        && candidate.maskedFraction>=0 && candidate.unresolvedFraction+candidate.maskedFraction<=policy.maximumUnresolvedFraction;
    const double total=candidate.depositedFraction+candidate.escapedFraction+candidate.unresolvedFraction+candidate.maskedFraction
        +candidate.absorberFraction;
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
    const auto queries=std::uint64_t(frame.work.size())*(2*(policy.description().maximumInterfaces+3)+policy.description().maximumVisits);
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
// The x of the native uProjection uniform: the tangent of half the vertical field of
// view, or minus half the view height for an orthographic camera.
inline float cameraProjectionX(const HarmonicMIDI::grid::SceneCameraRecord& camera)
{return camera.orthographic()?-.5f*camera.orthographicHeight:std::tan(camera.verticalFovRadians*.5f);}
inline Point cameraPosition(const Frame& frame)
{return {frame.camera.transform.translation.x,frame.camera.transform.translation.y,frame.camera.transform.translation.z};}
// An orthographic camera (orthographicHeight above zero) casts parallel rays along its
// view axis, one from each pixel's point on the camera plane. cameraOrigin is that
// point, and the camera position itself for a perspective camera.
inline Point cameraRay(const Frame& frame,float x,float y)
{
    if(frame.camera.orthographic()) {
        const auto forward=rotateSceneVector(frame.camera.transform.rotation,{0,0,-1});
        Point ray{forward.x,forward.y,forward.z};normalize(ray);return ray;
    }
    const float tangent=std::tan(frame.camera.verticalFovRadians*.5f),aspect=float(frame.width)/frame.height;
    const auto rotated=rotateSceneVector(frame.camera.transform.rotation,
        {(2*x/frame.width-1)*tangent*aspect,(1-2*y/frame.height)*tangent,-1});
    Point ray{rotated.x,rotated.y,rotated.z};normalize(ray);return ray;
}
inline Point cameraOrigin(const Frame& frame,float x,float y)
{
    auto origin=cameraPosition(frame);
    if(!frame.camera.orthographic())return origin;
    const float half=.5f*frame.camera.orthographicHeight,aspect=float(frame.width)/frame.height;
    const auto offset=rotateSceneVector(frame.camera.transform.rotation,
        {(2*x/frame.width-1)*half*aspect,(1-2*y/frame.height)*half,0});
    return {origin[0]+offset.x,origin[1]+offset.y,origin[2]+offset.z};
}
// Where the camera's ray to a world point starts: the camera, or the point's
// projection onto an orthographic camera's plane.
inline Point cameraEndpointOrigin(const Frame& frame,const Point& point)
{
    auto origin=cameraPosition(frame);
    if(!frame.camera.orthographic())return origin;
    const auto forward=rotateSceneVector(frame.camera.transform.rotation,{0,0,-1});
    const float along=(point[0]-origin[0])*forward.x+(point[1]-origin[1])*forward.y+(point[2]-origin[2])*forward.z;
    return {point[0]-forward.x*along,point[1]-forward.y*along,point[2]-forward.z*along};
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
    // Steradians for a perspective camera. An orthographic camera stores areas on
    // its camera plane in both, so clipped/pixel stays the coverage fraction.
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
    // An orthographic camera's rays are parallel, so its cell measures are areas on
    // the camera plane instead of solid angles: the pixel's own area, and the
    // clipped polygon projected along the view axis. Their ratio is the same
    // coverage fraction either way.
    const bool orthographic=frame.camera.orthographic();
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
        const float cornerX=float(x)+(i>=2),cornerY=float(y)+(i==1||i==2);
        const auto ray=cameraRay(frame,cornerX,cornerY);
        rays[i]={ray[0],ray[1],ray[2]};if(!unit(rays[i]))return refuse("Camera cell boundary ray is invalid");
        // Each orthographic corner ray starts at its own point on the camera plane.
        const auto startPoint=cameraOrigin(frame,cornerX,cornerY);
        const V start=subtract(V{startPoint[0],startPoint[1],startPoint[2]},origin);
        const double denominator=inner(normal,rays[i]);
        // Match the existing footprint's conservative horizon/distance bound.
        // Actual near/far clipping below uses linear camera depth, not ray range.
        if(denominator>=-1e-6||inner(forward,rays[i])<=0)
            return refuse("Camera cell pixel crosses a backface, grazing plane or receiver horizon");
        const double distance=(planeDistance-inner(normal,start))/denominator;
        if(!std::isfinite(distance)||distance<=0||distance>double(frame.camera.farPlane)*1e4)
            return refuse("Camera cell plane footprint is behind the camera or unbounded");
        for(unsigned a=0;a<3;++a)polygon[i][a]=start[a]+rays[i][a]*distance;
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
    // The orthographic measure: area on the camera plane, along the view axis.
    const auto projectedTriangle=[&](V a,V b,V c,double& area) {
        area=.5*std::abs(inner(forward,exterior(subtract(b,a),subtract(c,a))));
        return std::isfinite(area);
    };
    if(orthographic) {
        const double height=double(frame.camera.orthographicHeight);
        candidate.pixelSolidAngle=(height/frame.height)*(height*double(frame.width)/frame.height/frame.width);
        if(!std::isfinite(candidate.pixelSolidAngle)||candidate.pixelSolidAngle<=0)
            return refuse("Camera cell orthographic pixel area is singular");
    } else {
        for(unsigned i=1;i<3;++i) {
            double angle=0;if(!solidTriangle(rays[0],rays[i],rays[i+1],angle))return refuse("Camera cell pixel solid angle is singular");
            candidate.pixelSolidAngle+=angle;
        }
        if(candidate.pixelSolidAngle<=0||candidate.pixelSolidAngle>=2*3.14159265358979323846)
            return refuse("Camera cell pixel solid angle is outside the perspective hemisphere");
    }
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
        if(!std::isfinite(area)||!(orthographic?projectedTriangle(polygon[0],polygon[i],polygon[i+1],angle)
            :solidTriangle(polygon[0],polygon[i],polygon[i+1],angle)))
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
// The raster owner of a visible deposit's pixel. A pixel centre on an edge
// shared by two triangles of one receiver belongs to both, and rasterization
// and the closest-hit query may each name a different one. Another triangle
// of the same object, material and instance owns the pixel only when it holds
// the pixel-centre hit within the 8-bias tolerance of the camera checks. The
// OpenGL photon kernel applies the same rule in causticRasterOwns.
inline bool rasterOwnsPixelHit(const Frame& frame,const opaquevisibility::ClosestHit& receiver,
    const opaquevisibility::ClosestHit& pixelHit,std::uint32_t rasterTriangle)
{
    if(rasterTriangle==receiver.triangleId)return true;
    if(pixelHit.status!=1 || !finite(pixelHit.position))return false;
    for(const auto& triangle:frame.triangles)
        if(triangle.object==receiver.objectId && triangle.material==receiver.materialId
            && triangle.instance==receiver.instanceId && triangle.triangle==rasterTriangle)
        {
            const float tolerance=8*bias(pixelHit.position);
            if(std::abs(dot(sub(pixelHit.position,triangle.points[0]),triangle.normal))>tolerance)return false;
            for(unsigned e=0;e<3;++e)
            {
                const auto edge=sub(triangle.points[(e+1)%3],triangle.points[e]);
                if(dot(cross(edge,sub(pixelHit.position,triangle.points[e])),triangle.normal)<-tolerance*std::sqrt(dot(edge,edge)))
                    return false;
            }
            return true;
        }
    return false;
}
// GLSL and Metal promise no IEEE rounding, FMA contraction or correctly
// rounded division, so native verification never demands bit-equal float32
// pixel indices or corner areas. The CPU reference is binary64, measured
// from the camera origin so the pixel corners cancel nothing.
// A native projected pixel is admitted when the receiver point lies within
// kNativeFootprintPixelMargin of it. That includes points just outside the
// image (up to the margin beyond column 0 or the last column/row) and points
// just outside the frustum (depth up to the depth margin beyond near or far),
// because float32 can put either one on the inside.
// A native unprojected result is admitted when the receiver point is outside
// the frustum or the image, within the depth margin of near or far, within the
// pixel margin of an image edge, or when its centre ray is not clearly front
// facing (dot >= -1e-6 minus kNativeFootprintFacingMargin, so -1.1e-5). A
// projected claim keeps the shader's -1e-6 facing and corner horizon thresholds.
// The area tolerance is kNativeFootprintAreaBase (a 2e-4 floor) plus
// kNativeFootprintAreaConditioning unit roundoffs (8 x 2^-24) times the
// coordinate scale S times the quad diagonals over the area, the first-order
// area change when every float32 corner moves by that much. Far pixels do not
// reach the cap, because S grows with distance as the footprint does and the
// term levels off near 3e-3. The cap binds on pixels that are small next to S
// (here within about 2 to 2.5 m of the camera at 4K and this FOV) and on
// strongly sheared footprints at the frame edges. Those are refused on purpose,
// since their float32 area cannot be verified at any useful tolerance.
inline constexpr double kNativeFootprintPixelMargin=1.0/64;
inline constexpr double kNativeFootprintDepthMargin=1e-5;
inline constexpr double kNativeFootprintFacingMargin=1e-5;
inline constexpr double kNativeFootprintAreaConditioning=8;
inline constexpr double kNativeFootprintAreaBase=2e-4;
inline constexpr double kNativeFootprintAreaToleranceCap=1e-2;
// claim is the native footprint under verification, or null for the CPU
// projection used by synthetic readbacks. areaTolerance is relative.
// binaryArea, when given, receives the binary64 area of a projected footprint,
// and binaryPixelScale the smallest singular value of its mean x/y edges.
inline bool projectReceiverFootprint(const Frame& frame,const opaquevisibility::ClosestHit& hit,const PixelFootprint* claim,
    PixelFootprint& output,double& areaTolerance,std::string& error,double* binaryArea=nullptr,double* binaryPixelScale=nullptr)
{
    if(!nativeFixtureDimensionsWithinBounds(frame.width,frame.height) || !receiverTriangle(frame,hit))
    {error="Glass caustic deposit has no exact receiver triangle";return false;}
    using D3=std::array<double,3>;
    const auto rotate=[](const D3& axis,double w,const D3& v) {
        const D3 first{axis[1]*v[2]-axis[2]*v[1]+w*v[0],axis[2]*v[0]-axis[0]*v[2]+w*v[1],axis[0]*v[1]-axis[1]*v[0]+w*v[2]};
        return D3{v[0]+2*(axis[1]*first[2]-axis[2]*first[1]),v[1]+2*(axis[2]*first[0]-axis[0]*first[2]),
            v[2]+2*(axis[0]*first[1]-axis[1]*first[0])};
    };
    const auto dot3=[](const D3& a,const D3& b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];};
    const auto& rotation=frame.camera.transform.rotation;
    const D3 q{rotation.x,rotation.y,rotation.z},inverse{-double(rotation.x),-double(rotation.y),-double(rotation.z)};
    const auto origin=cameraPosition(frame);
    const D3 camera{origin[0],origin[1],origin[2]},normal{hit.outwardNormal[0],hit.outwardNormal[1],hit.outwardNormal[2]};
    const D3 delta{double(hit.position[0])-camera[0],double(hit.position[1])-camera[1],double(hit.position[2])-camera[2]};
    const auto local=rotate(inverse,rotation.w,delta);
    const double depth=-local[2],width=frame.width,height=frame.height;
    const double nearPlane=frame.camera.nearPlane,farPlane=frame.camera.farPlane;
    const bool orthographic=frame.camera.orthographic();
    const double tangent=orthographic?0.0:std::tan(double(frame.camera.verticalFovRadians)*.5),aspect=width/height;
    const double half=.5*double(frame.camera.orthographicHeight);
    // Native depth carries float32 roundoff of the camera and receiver coordinates,
    // bounded by 8 unit roundoffs (8 x 2^-24) of the largest absolute coordinate.
    const double coordinateScale=std::max({std::abs(camera[0]),std::abs(camera[1]),std::abs(camera[2]),
        std::abs(double(hit.position[0])),std::abs(double(hit.position[1])),std::abs(double(hit.position[2]))});
    const auto depthMargin=[&](double plane){return std::max(kNativeFootprintDepthMargin*plane,8*std::ldexp(1.0,-24)*coordinateScale);};
    PixelFootprint candidate;candidate.linearDepth=float(depth);
    if(!std::isfinite(depth)) {error="Glass caustic camera projection is nonfinite";return false;}
    const auto unprojected=[&]{output=candidate;areaTolerance=0;error.clear();return true;};
    const double px=orthographic?(local[0]/(half*aspect)+1)*.5*width:(local[0]/(depth*tangent*aspect)+1)*.5*width;
    const double py=orthographic?(1-local[1]/half)*.5*height:(1-local[1]/(depth*tangent))*.5*height;
    const bool inside=depth>=nearPlane && depth<=farPlane && px>=0 && py>=0 && px<width && py<height;
    if(depth>=nearPlane && depth<=farPlane && (!std::isfinite(px) || !std::isfinite(py))) {error="Glass caustic camera projection is nonfinite";return false;}
    const auto ray=[&](double x,double y) {
        auto r=orthographic?rotate(q,rotation.w,D3{0,0,-1})
            :rotate(q,rotation.w,D3{(2*x/width-1)*tangent*aspect,(1-2*y/height)*tangent,-1});
        const double length=std::sqrt(dot3(r,r));for(auto& a:r)a/=length;return r;
    };
    // An orthographic ray starts at its film point on the camera plane, relative to the camera.
    const auto start=[&](double x,double y) {
        return orthographic?rotate(q,rotation.w,D3{(2*x/width-1)*half*aspect,(1-2*y/height)*half,0}):D3{0,0,0};
    };
    double facingLimit=-1e-6;
    if(!claim) {
        if(!inside)return unprojected();
        candidate.x=std::uint32_t(px);candidate.y=std::uint32_t(py);
    } else if(!claim->projected) {
        const double m=kNativeFootprintPixelMargin;
        const bool boundary=std::abs(depth-nearPlane)<=depthMargin(nearPlane)
            || std::abs(depth-farPlane)<=depthMargin(farPlane)
            || px<=m || py<=m || px>=width-m || py>=height-m;
        if(!inside || boundary)return unprojected();
        candidate.x=std::uint32_t(px);candidate.y=std::uint32_t(py);facingLimit-=kNativeFootprintFacingMargin;
    } else {
        const double m=kNativeFootprintPixelMargin;
        if(!std::isfinite(px) || !std::isfinite(py) || depth<nearPlane-depthMargin(nearPlane)
            || depth>farPlane+depthMargin(farPlane) || claim->x>=frame.width || claim->y>=frame.height
            || px<double(claim->x)-m || px>double(claim->x)+1+m || py<double(claim->y)-m || py>double(claim->y)+1+m)
            return unprojected();
        candidate.x=claim->x;candidate.y=claim->y;
    }
    if(dot3(normal,ray(candidate.x+.5,candidate.y+.5))>=facingLimit)return unprojected();
    std::array<D3,4> corners;double scale=std::max({std::abs(camera[0]),std::abs(camera[1]),std::abs(camera[2])});
    for(unsigned c=0;c<4;++c)
    {
        const double cx=double(candidate.x)+(c==1 || c==2),cy=double(candidate.y)+(c>=2);
        const auto r=ray(cx,cy);const auto o=start(cx,cy);
        const D3 reach{delta[0]-o[0],delta[1]-o[1],delta[2]-o[2]};
        const double denominator=dot3(normal,r),distance=dot3(normal,reach)/denominator;
        if(!std::isfinite(distance) || denominator>=-1e-6 || distance<=0 || distance>double(frame.camera.farPlane)*1e4)
        {error="Glass caustic pixel footprint crosses the receiver horizon";return false;}
        for(unsigned a=0;a<3;++a){corners[c][a]=o[a]+r[a]*distance;scale=std::max(scale,std::abs(camera[a]+corners[c][a]));}
        scale=std::max(scale,distance);
    }
    const auto edge=[&](unsigned a,unsigned b){return D3{corners[b][0]-corners[a][0],corners[b][1]-corners[a][1],corners[b][2]-corners[a][2]};};
    const auto area=[&](unsigned a,unsigned b,unsigned c) {
        const auto u=edge(a,b),v=edge(a,c);
        const D3 n{u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]};return .5*std::sqrt(dot3(n,n));
    };
    const double worldArea=area(0,1,2)+area(0,2,3);
    if(!std::isfinite(worldArea) || worldArea<=0 || worldArea>std::numeric_limits<float>::max() || float(worldArea)==0)
    {error="Glass caustic pixel footprint is unrepresentable";return false;}
    const auto diagonal=[&](unsigned a,unsigned b){const auto e=edge(a,b);return std::sqrt(dot3(e,e));};
    const double tolerance=kNativeFootprintAreaBase+kNativeFootprintAreaConditioning*std::ldexp(1.0,-24)*scale
        *(diagonal(0,2)+diagonal(1,3))/worldArea;
    // The cap is deliberate. Refusing beats silently accepting any native area.
    if(claim && !(tolerance<=kNativeFootprintAreaToleranceCap))
    {error="Glass caustic native pixel footprint area is too ill-conditioned to verify";return false;}
    candidate.worldArea=float(worldArea);candidate.projected=true;output=candidate;areaTolerance=tolerance;
    if(binaryArea)*binaryArea=worldArea;
    if(binaryPixelScale) {
        D3 ex,ey;
        for(unsigned a=0;a<3;++a) {
            ex[a]=.5*((corners[1][a]-corners[0][a])+(corners[2][a]-corners[3][a]));
            ey[a]=.5*((corners[3][a]-corners[0][a])+(corners[2][a]-corners[1][a]));
        }
        const double xx=dot3(ex,ex),yy=dot3(ey,ey),xy=dot3(ex,ey),trace=xx+yy;
        const double determinant=std::max(0.0,xx*yy-xy*xy),largest=.5*(trace+std::sqrt(std::max(0.0,trace*trace-4*determinant)));
        *binaryPixelScale=largest>0?std::sqrt(determinant/largest):0;
    }
    error.clear();return true;
}
inline bool projectReceiver(const Frame& frame,const opaquevisibility::ClosestHit& hit,PixelFootprint& output,std::string& error)
{double areaTolerance=0;return projectReceiverFootprint(frame,hit,nullptr,output,areaTolerance,error);}
struct RasterReceiver
{
    std::uint32_t object = 0, material = 0, instance = 0, triangle = 0;
    float linearDepth = 0;
};
struct NativeDeposit
{
    std::uint32_t work = 0;
    CausticHistory reflectionHistory {};
    std::uint32_t interfaces = 0;
    float fraction = 0, incidentPower = 0, outgoingPower = 0, absorbedPower = 0;
    opaquevisibility::ClosestHit receiver, cameraHit, pixelHit;
    PixelFootprint footprint;
    RasterReceiver raster;
    bool cameraVisible = false;
    // Each macro exit stores microfacet order bits0..5, transmission bit6,
    // reserved bit7 zero. Used prefix length is interfaces, never a seed hash.
    std::array<std::uint8_t,32> ancestry {};
    // Tokens for interfaces 32 and up, from the native ancestry tail. Empty when
    // the policy holds 32 or fewer interfaces.
    std::vector<std::uint8_t> ancestryTail;
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
    // Power absorbed by the thin camera dielectric. A separate conservation
    // term, zero unless the prepared caustic has an absorber. OpenGL stores it
    // in path texel 4.y and Metal in the first legacy padding float.
    float thinAbsorbedPower = 0;
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
    const auto stored=deposit.ancestry.size()+deposit.ancestryTail.size();
    if(!validMultipleScatterPolicy(policy) || !deposit.interfaces || deposit.interfaces>stored
        || !deposit.reflectionHistory.clearFrom(deposit.interfaces))return false;
    for(std::size_t i=0;i<stored;++i)
    {
        const auto token=i<deposit.ancestry.size()?deposit.ancestry[i]:deposit.ancestryTail[i-deposit.ancestry.size()];
        if(i>=deposit.interfaces) {if(token!=0){return false;}continue;}
        const auto order=std::uint32_t(token&63);const bool transmitted=(token&64)!=0;
        if((token&128) || order>policy.maximumEvents || (matchedMedia?(order!=0 || !transmitted):order==0)
            || deposit.reflectionHistory.test(i)==transmitted)return false;
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
// and uint64 work into four words; smooth history splits its four uint32 words
// the same way across texels 1 and 30. Neither float casts of full IDs nor
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
    std::vector<HarmonicMIDI::grid::SceneLightId>& output,
    std::size_t& count,std::string& error)
{
    using namespace HarmonicMIDI::grid;
    if(!prepared.glass.sourceScene||!validateVisual3DScene(*prepared.glass.sourceScene).valid())
    {error="Directional source cohort has no valid immutable scene roster";return false;}
    std::vector<SceneLightId> candidate;
    const auto& scene=*prepared.glass.sourceScene;
    for(std::size_t i=0;i<scene.lightCount;++i) {
        const auto& light=scene.lights[i];
        if(light.kind==SceneLightKind::Directional&&light.spectralSource.version==1
            &&light.spectralSource.directionalIrradiance.version==1)candidate.push_back(light.id);
    }
    if(candidate.empty()){error="Directional source cohort has no admitted D8 scene lights";return false;}
    count=candidate.size();output=std::move(candidate);error.clear();return true;
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
    // Inline tokens cover the first 32 interfaces; later ones follow from the
    // history (1 reflect, 65 transmit).
    CausticHistory reflectionHistory{};
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
    // Splat frames only. keys hold (tile, work, slot, flatIndex), visibleCount
    // counts valid tile keys, and splatNormalisers has one native float N per
    // readback deposit (zero unless camera visible). gatherWork is measured.
    std::vector<float> splatNormalisers;
    std::uint64_t splatGatherWork = 0;
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
    // Power absorbed by the thin camera dielectric, apart from receiver absorption.
    std::array<double,8> thinAbsorbed {};
};
inline bool validateGratingTerminalReadback(const GratingIncidentBinding&,const NativeReadback&,
    const SubmissionIdentity&,std::string&,const NativeDirectionalSourceBinding* =nullptr);
inline bool calculateGratingCameraBounds(const GratingIncidentBinding&,GratingCameraBounds&,std::string&);
inline bool validateGratingCameraReadback(const GratingIncidentBinding&,const NativeReadback&,
    const SubmissionIdentity&,std::string&,const NativeDirectionalSourceBinding* =nullptr);
inline bool nearPower(double a,double b,double tolerance=2e-4)
{return std::isfinite(a) && std::isfinite(b) && std::abs(a-b)<=tolerance*std::max({std::abs(a),std::abs(b),1e-20});}
// Checks every band of a native spectral readback against a sparse reference
// keyed by band index, walking the ordered reference once beside the bands.
// check(actual,sum) gets a null sum for a band with no reference entry. Such a
// band holding +0 or -0 is skipped because both validators accept an exact
// zero with no reference, so the outcome equals calling check on every band.
// Keys at or past the readback size are never visited, as with a lookup per
// band.
template<class Map,class Check>
inline bool checkSpectralBands(const std::vector<float>& bands,const Map& reference,const Check& check)
{
    using Sum=typename Map::mapped_type;
    auto next=reference.begin();const auto end=reference.end();
    const float* values=bands.data();const std::size_t count=bands.size();
    std::size_t p=0;
    while(p<count) {
        const std::size_t stop=next==end?count:std::size_t(std::min<std::uint64_t>(next->first,count));
        for(;p<stop;++p)
            if(!(values[p]==0.0f) && !check(values[p],static_cast<const Sum*>(nullptr)))return false;
        if(p==count)break;
        if(!check(values[p],&next->second))return false;
        ++next;++p;
    }
    return true;
}
// The same walk for the splat validator: the refusal needs the flat band index
// of its first mismatch, from which the pixel and band decode, because
// "radiance differs" alone cannot say which pixel the native gather left at
// zero or wrote across.
template<class Map,class Check>
inline bool checkSpectralBandsIndexed(const std::vector<float>& bands,const Map& reference,const Check& check)
{
    using Sum=typename Map::mapped_type;
    auto next=reference.begin();const auto end=reference.end();
    const float* values=bands.data();const std::size_t count=bands.size();
    std::size_t p=0;
    while(p<count) {
        const std::size_t stop=next==end?count:std::size_t(std::min<std::uint64_t>(next->first,count));
        for(;p<stop;++p)
            if(!(values[p]==0.0f) && !check(values[p],static_cast<const Sum*>(nullptr),p))return false;
        if(p==count)break;
        if(!check(values[p],&next->second,p))return false;
        ++next;++p;
    }
    return true;
}
// The single native footprint acceptance predicate. The caller gets the
// reference and areaTolerance from projectReceiverFootprint with native as claim.
inline bool nativeFootprintMatches(const PixelFootprint& native,const PixelFootprint& reference,double areaTolerance)
{
    if(native.projected!=reference.projected)return false;
    return !reference.projected || (native.x==reference.x && native.y==reference.y
        && areaTolerance<=kNativeFootprintAreaToleranceCap && nearPower(native.worldArea,reference.worldArea,areaTolerance)
        && nearPower(native.linearDepth,reference.linearDepth));
}
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
    const auto delta=sub(receiver.position,cameraEndpointOrigin(frame,receiver.position));
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
    const bool splat=frame.splatVersion!=0;
    if(receipt.keys.size()!=(splat?frame.splat.paddedKeyCount:frame.bounds.deposition.paddedKeyCount)
        || receipt.visibleCount>(splat?receipt.keys.size():readback.deposits.size()) || readback.paths.size()!=frame.work.size()
        || (!splat && (!receipt.splatNormalisers.empty() || receipt.splatGatherWork)))
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
    if(offsets.back()!=readback.deposits.size() || (!splat && visible!=receipt.visibleCount))
        return refuse("Glass caustic deposition visible permutation is incomplete");
    // Contract: every splat frame reaches validateSplatReadback. This early
    // return is taken only for splat frames, and validateNativeReadback is the
    // only caller; it refuses a splat frame without an expected submission
    // owner and replaces its single-pixel loop with validateSplatReadback, which
    // checks these tile keys once each deposit's visibility is verified.
    if(splat){error.clear();return true;}
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
// Camera closest hit through the centre of pixel (x,y). Status 0 is a miss.
// Validation uses splatCpuQuery; contract tests may supply an analytic scene.
using SplatPixelQuery=std::function<bool(std::uint32_t,std::uint32_t,opaquevisibility::ClosestHit&,std::string&)>;
inline SplatPixelQuery splatCpuQuery(const Prepared& prepared,const Frame& frame)
{
    return [&prepared,&frame](std::uint32_t x,std::uint32_t y,opaquevisibility::ClosestHit& hit,std::string& error) {
        const auto ray=cameraRay(frame,float(x)+.5f,float(y)+.5f);
        const float maximum=4*frame.camera.farPlane;
        if(!finite(ray) || !std::isfinite(maximum) || maximum<=0) {error="Glass caustic splat camera ray is invalid";return false;}
        hit=opaquevisibility::closestHitCpu(prepared.glass.query,frame.query,cameraOrigin(frame,float(x)+.5f,float(y)+.5f),ray,kRayMinimum,maximum);
        if(hit.status!=0 && hit.status!=1) {error="Glass caustic splat CPU camera query failed";return false;}
        return true;
    };
}
// Camera closest hit through an exact film point (x,y), already offset from
// the pixel corner. The edge-aware splat reference casts these just inside a
// pixel centre to find the other surfaces a native raster may name there.
using SplatOffsetQuery=std::function<bool(float,float,opaquevisibility::ClosestHit&,std::string&)>;
// Offset of those rays from the pixel centre. Rasterizers snap vertices to
// 1/256 pixel, so a centre within that snap of an edge may land on either
// side. In the 4K starter the closest such centre, (1610, 895), sits within
// 1/2048 pixel of the lens silhouette; 1/64 pixel brackets every snap with room.
inline constexpr float kSplatRasterSnapMargin=1.0f/64;
inline SplatOffsetQuery splatCpuOffsetQuery(const Prepared& prepared,const Frame& frame)
{
    return [&prepared,&frame](float x,float y,opaquevisibility::ClosestHit& hit,std::string& error) {
        const auto ray=cameraRay(frame,x,y);
        const float maximum=4*frame.camera.farPlane;
        if(!finite(ray) || !std::isfinite(maximum) || maximum<=0) {error="Glass caustic splat camera ray is invalid";return false;}
        hit=opaquevisibility::closestHitCpu(prepared.glass.query,frame.query,cameraOrigin(frame,x,y),ray,kRayMinimum,maximum);
        if(hit.status!=0 && hit.status!=1) {error="Glass caustic splat CPU camera query failed";return false;}
        return true;
    };
}
// pixelScale is the deposit pixel's smallest Jacobian singular value.
// unclamped, when given, is the entry radius before the pixel clamp.
inline bool splatRadius(const Frame& frame,const NativeDeposit& deposit,double pixelScale,double& radius,
    double* unclamped=nullptr)
{
    if(deposit.work>=frame.work.size() || frame.work[deposit.work].entry>=frame.entries.size())return false;
    const double entryArea=frame.entries[frame.work[deposit.work].entry].area;
    if(!std::isfinite(entryArea) || entryArea<=0 || !std::isfinite(pixelScale) || pixelScale<=0)return false;
    const double entry=kSplatEntryRadiusScale*std::sqrt(entryArea/3.14159265358979323846);
    radius=std::clamp(entry,kSplatMinimumRadiusPixels*pixelScale,double(kMaximumSplatRadiusPixels)*pixelScale);
    if(unclamped)*unclamped=entry;
    return std::isfinite(radius) && radius>0;
}
// Refuses a tile over kMaximumSplatDepositsPerTile or a frame over
// splatGatherWorkCap(), and sets capExceeded for either. A tile key outside
// the image is a defect and leaves it false. keys must be sorted, valid and
// sentinel free.
inline bool admitSplatGather(const Frame& frame,const std::vector<DepositionKey>& keys,std::uint64_t& gatherWork,std::string& error,
    bool* capExceeded=nullptr)
{
    gatherWork=0;if(capExceeded)*capExceeded=false;
    const auto tiles=frame.splat.tilesX*frame.splat.tilesY;
    // Keys are sorted by tile, so the last key bounds them all. Checked first so
    // a crowded tile cannot hide an out-of-image key behind a cap fallback.
    if(!keys.empty() && (!frame.splat.tilesX || keys.back().pixel>=tiles))
    {error="Glass caustic splat tile key is outside the image";return false;}
    for(std::size_t i=0;i<keys.size();) {
        const auto tile=keys[i].pixel;
        if(!frame.splat.tilesX || tile>=tiles) {error="Glass caustic splat tile key is outside the image";return false;}
        std::size_t end=i;while(end<keys.size() && keys[end].pixel==tile)++end;
        const std::uint64_t count=end-i,tx=tile%frame.splat.tilesX,ty=tile/frame.splat.tilesX;
        if(count>kMaximumSplatDepositsPerTile) {
            if(capExceeded)*capExceeded=true;
            error="Glass caustic splat tile exceeds its deposit cap";return false;
        }
        const std::uint64_t pixels=(std::min<std::uint64_t>(frame.width,(tx+1)*kSplatTile)-tx*kSplatTile)
            *(std::min<std::uint64_t>(frame.height,(ty+1)*kSplatTile)-ty*kSplatTile);
        if(count*pixels>splatGatherWorkCap()-gatherWork) {
            if(capExceeded)*capExceeded=true;
            error="Glass caustic splat gather work exceeds its budget";return false;
        }
        gatherWork+=count*pixels;i=end;
    }
    error.clear();return true;
}
struct SplatReference
{
    struct Sum {double physical=0,tolerance=0;};
    // Key (y*width+x)*8+band. Terms are added in gather order, ascending work
    // then slot, which is the canonical readback deposit order.
    std::map<std::uint64_t,Sum> pixels;
    // Binary64 world area of every accepted pixel, for flux checks.
    std::map<std::uint64_t,double> pixelAreas;
    // One entry per readback deposit; zero unless the deposit is camera visible.
    std::vector<double> normalisers,normaliserTolerances,splattedFlux,radii;
    std::vector<DepositionKey> keys; // Sorted valid tile keys, no sentinels.
    std::uint64_t pixelQueries=0,queryWork=0,gatherWork=0;
    // Window pixels where the offset rays found another surface the raster may
    // name (kSplatRasterSnapMargin).
    std::uint64_t edgePixels=0;
    // The tile deposit cap or the gather work cap overflowed. The frame then
    // falls back to single-pixel deposits, and pixels, keys and gatherWork are
    // empty.
    bool capOverflow=false;
};
// CPU reference for one splat frame. Each distinct window pixel gets one
// camera query. A pixel is accepted when its closest hit is a reflective
// non-glass receiver whose binary64 footprint projects back to the same pixel,
// and its normal agrees with the deposit's receiver normal. Its weight is
// K(|hit - receiver|, r); the deposit's own pixel always weighs 1.
// Tolerances bound the native float32 result: the weight interval over a
// position error of 8 biases at each end and a radius error of half the
// centre area tolerance, each pixel's area tolerance, and float32 summation.
inline bool buildSplatReference(const Frame& frame,const std::vector<NativeDeposit>& deposits,const SplatPixelQuery& query,
    SplatReference& output,std::string& error,const SplatOffsetQuery& offsetQuery={})
{
    const auto refuse=[&](const char* text){error=text;return false;};
    constexpr double pi=3.14159265358979323846;
    SplatSchedule schedule;std::string scheduleError;
    if(frame.splatVersion!=kSplatVersion || frame.gratingIncidentBinding || frame.gratingStaging || frame.work.empty() || !query
        || !calculateSplatSchedule(frame,schedule,scheduleError) || !(schedule==frame.splat)
        || deposits.size()>frame.bounds.maximumDeposits)
        return refuse("Glass caustic splat frame version, schedule or grating owner is invalid");
    const auto visits=frame.bounds.maximumDeposits/frame.work.size();
    if(!visits || visits>kMaximumBranchVisits)return refuse("Glass caustic splat trace slot capacity changed");
    SplatReference candidate;
    candidate.normalisers.assign(deposits.size(),0);candidate.normaliserTolerances.assign(deposits.size(),0);
    candidate.splattedFlux.assign(deposits.size(),0);candidate.radii.assign(deposits.size(),0);
    struct Sample {
        bool accepted=false;opaquevisibility::ClosestHit hit;double area=0,areaTolerance=0,pixelScale=0,positionError=0;
        // Set once the pixel has been checked for edge outcomes; null when it has none.
        mutable bool edgeChecked=false;mutable const std::vector<Sample>* edge=nullptr;
    };
    const auto camera=cameraPosition(frame);
    // Acceptance, footprint area and position error of value.hit at pixel (x,y).
    const auto evaluate=[&](std::uint32_t x,std::uint32_t y,Sample& value) {
        if(!(value.hit.status==1 && receiverTriangle(frame,value.hit)))return;
        PixelFootprint footprint;std::string ignored;double area=0,scale=0;
        if(projectReceiverFootprint(frame,value.hit,nullptr,footprint,value.areaTolerance,ignored,&area,&scale)
            && footprint.projected && footprint.x==x && footprint.y==y) {
            // Native and CPU positions on the centre ray each carry a few
            // float32 roundoffs of the coordinate scale plus the distance,
            // stretched along the plane by 1/cos at the receiver.
            double ray[3],length=0,coordinates=0;
            const auto start=cameraEndpointOrigin(frame,value.hit.position);
            for(unsigned a=0;a<3;++a) {
                ray[a]=double(value.hit.position[a])-start[a];length+=ray[a]*ray[a];
                coordinates=std::max({coordinates,std::abs(double(camera[a])),std::abs(double(value.hit.position[a]))});
            }
            length=std::sqrt(length);
            const double cosine=std::abs(ray[0]*value.hit.outwardNormal[0]+ray[1]*value.hit.outwardNormal[1]
                +ray[2]*value.hit.outwardNormal[2])/length;
            value.positionError=32*std::ldexp(1.0,-24)*(coordinates+length/std::max(cosine,1e-6));
            value.accepted=std::isfinite(value.positionError);value.area=area;value.pixelScale=scale;
        }
    };
    const auto charge=[&](const opaquevisibility::ClosestHit& hit) {
        ++candidate.pixelQueries;
        if(hit.work>std::numeric_limits<std::uint64_t>::max()-candidate.queryWork)
        {error="Glass caustic splat CPU query work does not fit 64 bits: "+std::to_string(candidate.queryWork)+" plus "+std::to_string(hit.work);return false;}
        candidate.queryWork+=hit.work;return true;
    };
    std::unordered_map<std::uint64_t,Sample> samples;
    const auto sample=[&](std::uint32_t x,std::uint32_t y,const Sample*& result) {
        const auto key=std::uint64_t(y)*frame.width+x;
        auto found=samples.find(key);
        if(found==samples.end()) {
            if(candidate.pixelQueries>=kMaximumSplatCpuPixelQueries)
            {error="Glass caustic splat CPU pixel queries exceed their budget";return false;}
            Sample value;
            if(!query(x,y,value.hit,error) || !charge(value.hit))return false;
            evaluate(x,y,value);
            found=samples.emplace(key,value).first;
        }
        result=&found->second;return true;
    };
    // Other surfaces the raster may name at a window pixel (kSplatRasterSnapMargin).
    // Each is evaluated like a centre hit, with the extra position error of
    // moving its offset hit back onto the centre ray within its own plane,
    // where the native pass puts it.
    const auto sameSurface=[](const opaquevisibility::ClosestHit& a,const opaquevisibility::ClosestHit& b) {
        return a.status==b.status && (a.status!=1 || sameTriangle(a,b));
    };
    std::vector<std::unique_ptr<std::vector<Sample>>> alternatives;
    const auto alternativesAt=[&](std::uint32_t x,std::uint32_t y,const Sample* centre,const std::vector<Sample>*& result) {
        result=nullptr;
        if(!offsetQuery)return true;
        if(!centre->edgeChecked) {
            const auto centreHit=centre->hit;
            constexpr int dx[4]{1,-1,0,0},dy[4]{0,0,1,-1};
            bool boundary=false;
            for(unsigned n=0;n<4 && !boundary;++n) {
                const auto nx=std::int64_t(x)+dx[n],ny=std::int64_t(y)+dy[n];
                if(nx<0 || ny<0 || nx>=std::int64_t(frame.width) || ny>=std::int64_t(frame.height))continue;
                const Sample* neighbour=nullptr;
                if(!sample(std::uint32_t(nx),std::uint32_t(ny),neighbour))return false;
                boundary=!sameSurface(neighbour->hit,centreHit);
            }
            std::vector<Sample> list;
            if(boundary) {
                const auto ray=cameraRay(frame,float(x)+.5f,float(y)+.5f);
                for(unsigned n=0;n<4;++n) {
                    if(candidate.pixelQueries>=kMaximumSplatCpuPixelQueries)
                    {error="Glass caustic splat CPU pixel queries exceed their budget";return false;}
                    Sample value;
                    if(!offsetQuery(float(x)+.5f+float(dx[n])*kSplatRasterSnapMargin,float(y)+.5f+float(dy[n])*kSplatRasterSnapMargin,
                        value.hit,error) || !charge(value.hit))return false;
                    if(sameSurface(value.hit,centreHit) || std::any_of(list.begin(),list.end(),
                        [&](const Sample& seen){return sameSurface(seen.hit,value.hit);}))continue;
                    evaluate(x,y,value);
                    if(value.accepted) {
                        double facing=0,offset=0,shift=0;
                        for(unsigned a=0;a<3;++a) {
                            facing+=double(ray[a])*value.hit.outwardNormal[a];
                            offset+=(double(value.hit.position[a])-camera[a])*value.hit.outwardNormal[a];
                        }
                        const double t=offset/facing;
                        for(unsigned a=0;a<3;++a) {
                            const double e=double(value.hit.position[a])-(camera[a]+t*double(ray[a]));shift+=e*e;
                        }
                        value.positionError+=std::sqrt(shift);
                        value.accepted=std::isfinite(value.positionError);
                    }
                    list.push_back(value);
                }
            }
            if(!list.empty()) {
                alternatives.push_back(std::make_unique<std::vector<Sample>>(std::move(list)));
                centre->edge=alternatives.back().get();
            }
            centre->edgeChecked=true;
        }
        result=centre->edge;return true;
    };
    struct Term {std::uint64_t pixel=0;double weight=0,area=0,weightTolerance=0;};
    std::vector<Term> terms;terms.reserve(kSplatWindow*kSplatWindow);
    std::vector<std::uint32_t> nextSlot(frame.work.size());
    std::vector<DepositionKey> keys;std::uint32_t previousWork=0;
    for(std::size_t i=0;i<deposits.size();++i) {
        const auto& deposit=deposits[i];
        if(deposit.work>=frame.work.size() || deposit.work<previousWork)
            return refuse("Glass caustic splat deposits are not in canonical work order");
        previousWork=deposit.work;
        const auto slot=nextSlot[deposit.work]++;
        if(slot>=visits)return refuse("Glass caustic splat deposit exceeds its trace slots");
        if(slot>=frame.splat.keySlots)return refuse("Glass caustic splat deposit exceeds its keyed slots");
        if(!deposit.cameraVisible)continue;
        if(!deposit.footprint.projected || deposit.footprint.x>=frame.width || deposit.footprint.y>=frame.height
            || !std::isfinite(deposit.outgoingPower) || deposit.outgoingPower<0)
            return refuse("Glass caustic splat deposit has no visible receiver pixel");
        const auto cx=deposit.footprint.x,cy=deposit.footprint.y;
        const Sample* centre=nullptr;
        if(!sample(cx,cy,centre))return false;
        // The pixel-centre hit may name the coplanar partner across a shared
        // edge, as the native raster may (rasterOwnsPixelHit).
        const bool ownCentre=centre->accepted && (sameTriangle(centre->hit,deposit.receiver)
            || (centre->hit.objectId==deposit.receiver.objectId && centre->hit.materialId==deposit.receiver.materialId
                && centre->hit.instanceId==deposit.receiver.instanceId
                && rasterOwnsPixelHit(frame,centre->hit,centre->hit,deposit.receiver.triangleId)));
        if(!ownCentre)return refuse("Glass caustic splat centre pixel does not see its own receiver");
        if(!(centre->areaTolerance<=kNativeFootprintAreaToleranceCap))
            return refuse("Glass caustic splat pixel area is too ill-conditioned to verify");
        double radius=0,entryRadius=0;
        if(!splatRadius(frame,deposit,centre->pixelScale,radius,&entryRadius))return refuse("Glass caustic splat radius is invalid");
        // A clamped radius follows the native pixel scale, which comes from the
        // same four float32 corners as its area. An entry radius more than that
        // tolerance inside both clamps is clamped on neither side, and the
        // native value is 2.5 sqrt(A / pi) of the same float area: a few float32
        // roundoffs, so 1e-5 leaves room for fast-math division and sqrt.
        const double lowClamp=kSplatMinimumRadiusPixels*centre->pixelScale,highClamp=double(kMaximumSplatRadiusPixels)*centre->pixelScale;
        const bool clamped=!(entryRadius>lowClamp*(1+centre->areaTolerance) && entryRadius<highClamp*(1-centre->areaTolerance));
        const double radiusTolerance=clamped?centre->areaTolerance+1e-6:1e-5;
        // Neighbour weights scale by the deposit's grazing spread. The cosine
        // uses the receipt position and normal that the native pass reads.
        double toReceiver[3],receiverDistance=0,facing=0;
        const auto viewStart=cameraEndpointOrigin(frame,deposit.receiver.position);
        for(unsigned a=0;a<3;++a) {
            toReceiver[a]=double(deposit.receiver.position[a])-viewStart[a];receiverDistance+=toReceiver[a]*toReceiver[a];
            facing+=toReceiver[a]*deposit.receiver.outwardNormal[a];
        }
        receiverDistance=std::sqrt(receiverDistance);
        const double cosine=std::abs(facing)/receiverDistance;
        if(!std::isfinite(cosine))return refuse("Glass caustic splat receiver view cosine is not finite");
        const double spread=splatSpread(cosine),spreadHigh=splatSpread(cosine+kSplatSpreadCosineTolerance),
            spreadLow=splatSpread(cosine-kSplatSpreadCosineTolerance);
        const auto window=splatWindow(frame.width,frame.height,cx,cy);
        terms.clear();double normaliser=0,normaliserTolerance=0,edgeTolerance=0;
        // Weight interval of one surface outcome at pixel (x,y); zero when it is
        // not an agreeing receiver.
        struct Outcome {bool accepted=false;double weight=0,low=0,high=0;};
        const auto outcome=[&](const Sample& h,std::uint32_t x,std::uint32_t y) {
            Outcome o;
            if(!h.accepted || !(dot(h.hit.outwardNormal,deposit.receiver.outwardNormal)>=kSplatNormalAgreement))return o;
            o.accepted=true;o.weight=o.low=o.high=1;
            if(x!=cx || y!=cy) {
                double squared=0;
                for(unsigned a=0;a<3;++a){const double e=double(h.hit.position[a])-deposit.receiver.position[a];squared+=e*e;}
                // The deposit position is the native receipt itself, shared by both sides.
                const double distance=std::sqrt(squared),delta=h.positionError;
                o.weight=splatKernel(distance,radius)*spread;
                o.high=splatKernel(std::max(0.0,distance-delta),radius*(1+radiusTolerance))*spreadHigh;
                o.low=splatKernel(distance+delta,radius*(1-radiusTolerance))*spreadLow;
            }
            return o;
        };
        for(auto y=window.y0;y<=window.y1;++y)for(auto x=window.x0;x<=window.x1;++x) {
            const Sample* s=nullptr;
            if(!sample(x,y,s))return false;
            // The deposit's own pixel keeps the centre ray alone: a single-pixel
            // deposit has nothing else to normalise by.
            const std::vector<Sample>* others=nullptr;
            if((x!=cx || y!=cy) && !alternativesAt(x,y,s,others))return false;
            const auto nominal=outcome(*s,x,y);
            if(!others || others->empty()) {
                if(!nominal.accepted || nominal.high==0)continue;
                if(!(s->areaTolerance<=kNativeFootprintAreaToleranceCap))
                    return refuse("Glass caustic splat pixel area is too ill-conditioned to verify");
                const double weight=nominal.weight,weightTolerance=std::max(nominal.high-weight,weight-nominal.low)+1e-6;
                terms.push_back({std::uint64_t(y)*frame.width+x,weight,s->area,weightTolerance});
                normaliser+=weight*s->area;
                normaliserTolerance+=(weight*s->areaTolerance+weightTolerance*(1+s->areaTolerance))*s->area;
                continue;
            }
            // An edge pixel: the native pass may shade any of these surfaces, or
            // none. The reference keeps the centre ray's value and bounds the rest.
            const double weight=nominal.accepted?nominal.weight:0,mass=weight*(nominal.accepted?s->area:0);
            double highest=nominal.accepted?nominal.high:0,lowest=nominal.accepted?nominal.low:0;
            double upperMass=nominal.accepted?nominal.high*s->area*(1+s->areaTolerance):0;
            double lowerMass=nominal.accepted?nominal.low*s->area*(1-s->areaTolerance):0;
            double area=nominal.accepted?s->area:0;
            const auto include=[&](const Sample& h,const Outcome& o) {
                if(o.high>0 && !(h.areaTolerance<=kNativeFootprintAreaToleranceCap))return false;
                highest=std::max(highest,o.accepted?o.high:0.0);lowest=std::min(lowest,o.accepted?o.low:0.0);
                upperMass=std::max(upperMass,o.accepted?o.high*h.area*(1+h.areaTolerance):0.0);
                lowerMass=std::min(lowerMass,o.accepted?o.low*h.area*(1-h.areaTolerance):0.0);
                if(!(area>0) && o.accepted)area=h.area;
                return true;
            };
            if(!include(*s,nominal))return refuse("Glass caustic splat pixel area is too ill-conditioned to verify");
            for(const auto& other:*others)
                if(!include(other,outcome(other,x,y)))return refuse("Glass caustic splat pixel area is too ill-conditioned to verify");
            if(highest==0)continue;
            const double weightTolerance=std::max(highest-weight,weight-lowest)+1e-6;
            terms.push_back({std::uint64_t(y)*frame.width+x,weight,area,weightTolerance});
            normaliser+=mass;
            edgeTolerance+=std::max(upperMass-mass,mass-lowerMass)+2e-6*area;
        }
        normaliserTolerance+=double(terms.size())*std::ldexp(1.0,-23)*normaliser;
        if(!std::isfinite(normaliser) || !std::isfinite(normaliserTolerance))
            return refuse("Glass caustic splat normaliser is unverifiable: it is not finite");
        if(!(normaliser>0))return refuse("Glass caustic splat normaliser is unverifiable: no window pixel supports it");
        if(!(normaliserTolerance<=kSplatMaximumNormaliserTolerance*normaliser)) {
            char text[160];
            std::snprintf(text,sizeof text,"Glass caustic splat normaliser is unverifiable: tolerance ratio %.3g exceeds %.3g at pixel (%u, %u)",
                normaliserTolerance/normaliser,kSplatMaximumNormaliserTolerance,unsigned(cx),unsigned(cy));
            error=text;return false;
        }
        // Edge outcomes are admitted results, not roundoff, so they are bounded
        // apart: the per-pixel bound below holds while the whole tolerance is at
        // most half of N.
        normaliserTolerance+=edgeTolerance;
        if(!(normaliserTolerance<=.5*normaliser)) {
            char text[160];
            std::snprintf(text,sizeof text,"Glass caustic splat normaliser is unverifiable: edge pixels put %.3g of it in doubt at pixel (%u, %u)",
                normaliserTolerance/normaliser,unsigned(cx),unsigned(cy));
            error=text;return false;
        }
        const auto band=frame.work[deposit.work].band;
        const double scale=double(deposit.outgoingPower)/(pi*normaliser),relative=normaliserTolerance/normaliser;
        double flux=0;
        for(const auto& term:terms) {
            auto& sum=candidate.pixels[term.pixel*8+band];const double value=scale*term.weight;
            sum.physical+=value;
            // |w'/N' - w/N| <= (wT (1 + 2 rel) + 2 rel w) / N for rel <= 1/2.
            sum.tolerance+=scale*(term.weightTolerance*(1+2*relative)+term.weight*(2*relative+4*std::ldexp(1.0,-24)));
            candidate.pixelAreas[term.pixel]=term.area;flux+=value*pi*term.area;
        }
        candidate.normalisers[i]=normaliser;candidate.normaliserTolerances[i]=normaliserTolerance;
        candidate.splattedFlux[i]=flux;candidate.radii[i]=radius;
        const auto flat=std::uint64_t(deposit.work)*visits+slot;
        for(auto ty=window.y0/kSplatTile;ty<=window.y1/kSplatTile;++ty)
            for(auto tx=window.x0/kSplatTile;tx<=window.x1/kSplatTile;++tx)
                keys.push_back({std::uint32_t(ty*frame.splat.tilesX+tx),deposit.work,slot,std::uint32_t(flat)});
    }
    candidate.edgePixels=alternatives.size();
    if(keys.size()>frame.splat.paddedKeyCount)return refuse("Glass caustic splat tile keys exceed admission");
    std::sort(keys.begin(),keys.end());
    bool capExceeded=false;
    if(!admitSplatGather(frame,keys,candidate.gatherWork,error,&capExceeded)) {
        if(!capExceeded)return false;
        // Crowded tiles fall back to single-pixel deposits, like the native consumers.
        candidate.pixels.clear();candidate.keys.clear();candidate.gatherWork=0;candidate.capOverflow=true;
        output=std::move(candidate);error.clear();return true;
    }
    for(const auto& entry:candidate.pixels)
        if(!std::isfinite(entry.second.physical) || entry.second.physical>surfacematerial::kMaximumLinearColorComponent)
            return refuse("Glass caustic splat pixel radiance is unrepresentable");
    candidate.keys=std::move(keys);output=std::move(candidate);error.clear();return true;
}
// The native consumer reads back no per-pixel product: no window camera hits,
// positions or areas. It returns the tile keys, measured gather work, one
// float normaliser per deposit slot and the final eight-band radiance. The CPU
// runs its own closest-hit query through each window pixel centre (splatCpuQuery
// over the admitted query BVH), its own binary64 footprint and kernel, and
// compares only those results. The GL consumer derives its surface from the
// private raster owner and the triangle plane; nothing it computes per pixel is
// trusted here.
// Native splat receipts against the CPU reference: exact compacted tile keys
// and measured gather work, each visible deposit's float normaliser within its
// derived tolerance, and every pixel and band within its summed tolerance plus
// 5e-4 relative. A pixel nobody splats must read exactly zero. There is no
// float32 ordering check: native weights use raster positions, not CPU hits.
inline bool validateSplatReadback(const Frame& frame,const NativeReadback& readback,const SplatPixelQuery& query,std::string& error,
    const SplatOffsetQuery& offsetQuery={})
{
    const auto refuse=[&](const char* text){error=text;return false;};
    const auto& receipt=readback.deposition;
    if(frame.splatVersion!=kSplatVersion || receipt.version!=kDepositionVersion || !receipt.complete || receipt.overflow
        || receipt.keys.size()!=frame.splat.paddedKeyCount || receipt.splatNormalisers.size()!=readback.deposits.size()
        || readback.spectralRadiance.size()!=std::uint64_t(frame.width)*frame.height*8)
        return refuse("Glass caustic splat receipt is incomplete, overflowed or resized");
    SplatReference reference;
    if(!buildSplatReference(frame,readback.deposits,query,reference,error,offsetQuery))return false;
    if(reference.capOverflow)return refuse("Glass caustic splat caps overflow, so the frame must fall back to single-pixel deposits");
    if(receipt.visibleCount!=reference.keys.size() || receipt.splatGatherWork!=reference.gatherWork)
        return refuse("Glass caustic splat tile key count or gather work changed");
    for(std::size_t i=0;i<receipt.keys.size();++i)
        if(i<reference.keys.size()?!(receipt.keys[i]==reference.keys[i]):!depositionSentinel(receipt.keys[i]))
            return refuse("Glass caustic splat tile keys are reordered, missing or not compacted");
    for(std::size_t i=0;i<readback.deposits.size();++i) {
        const float native=receipt.splatNormalisers[i];
        if(!readback.deposits[i].cameraVisible) {
            if(native!=0)return refuse("Glass caustic splat hidden deposit has a normaliser");
            continue;
        }
        if(!std::isfinite(native) || std::abs(double(native)-reference.normalisers[i])>reference.normaliserTolerances[i])
            return refuse("Glass caustic splat native normaliser differs from the CPU reference");
    }
    // One ordered walk of reference.pixels beside the readback, as in the
    // single-pixel check; every pixel band is still compared. The first
    // mismatch carries its arithmetic into the refused error: expected zero
    // means the native gather wrote a pixel the reference claims empty, a
    // magnitude over tolerance means the sums themselves disagree, and the
    // schedule numbers say which keyed set produced them.
    std::size_t bandChecks=0;
    double firstActual=0,firstExpected=0,firstTolerance=0;std::size_t firstFlat=0;bool firstFound=false;
    const auto checkBand=[&](float actual,const SplatReference::Sum* found,std::size_t flat) {
        const double expected=found?found->physical:0;
        const double tolerance=found?found->tolerance+5e-4*expected:0;
        if(!std::isfinite(actual) || actual<0 || actual>surfacematerial::kMaximumLinearColorComponent
            || std::abs(double(actual)-expected)>tolerance)
        {
            if(!firstFound)
            {
                firstFound=true;firstActual=actual;firstExpected=expected;firstTolerance=tolerance;firstFlat=flat;
            }
            return refuse("Glass caustic splat native pixel radiance differs from the CPU reference");
        }
        ++bandChecks;
        return true;
    };
    if(!checkSpectralBandsIndexed(readback.spectralRadiance,reference.pixels,checkBand))
    {
        const auto firstPixel=firstFlat/8u;
        error+=" (check="+std::to_string(bandChecks)
            +" pixel="+std::to_string(firstPixel%frame.width)+","+(firstPixel/frame.width>=frame.height
                ?std::string("?"):std::to_string(firstPixel/frame.width))
            +" band="+std::to_string(firstFlat%8)
            +" actual="+std::to_string(firstActual)
            +" expected="+std::to_string(firstExpected)
            +" tolerance="+std::to_string(firstTolerance)
            +" keys="+std::to_string(receipt.keys.size())
            +" padded="+std::to_string(frame.splat.paddedKeyCount)
            +" keySlots="+std::to_string(frame.splat.keySlots)
            +" visits="+std::to_string(frame.bounds.maximumDeposits/(frame.work.empty()?1:frame.work.size()))
            +" paths="+std::to_string(frame.work.size())+")";
        return false;
    }
    error.clear();return true;
}
// Opt-in for a native consumer that implements the splat passes. Grating
// staging and grating-bound frames keep single-pixel deposits.
inline bool enableSplat(const Prepared& prepared,Frame& frame,std::string& error)
{
    if(frame.splatVersion || frame.gratingIncidentBinding || frame.gratingStaging)
    {error="Glass caustic splat requires an ordinary single-pixel frame";return false;}
    SplatSchedule schedule;
    if(!calculateSplatSchedule(frame,schedule,error))return false;
    auto candidate=frame;candidate.splatVersion=kSplatVersion;candidate.splat=schedule;
    candidate.identity=frameIdentity(prepared,candidate);
    if(candidate.identity.empty()){error="Glass caustic splat frame identity is unavailable";return false;}
    frame=std::move(candidate);error.clear();return true;
}
// The single-pixel frame of a splat frame, with the identity it had before
// enableSplat. Native consumers use it when the splat caps overflow, and grating
// bindings use it for their ordinary prerequisite.
inline bool disableSplat(const Prepared& prepared,Frame& frame,std::string& error)
{
    if(frame.splatVersion!=kSplatVersion || frame.gratingIncidentBinding || frame.gratingStaging)
    {error="Glass caustic splat fallback requires an ordinary splat frame";return false;}
    auto candidate=frame;candidate.splatVersion=0;candidate.splat={};
    candidate.identity=frameIdentity(prepared,candidate);
    if(candidate.identity.empty()){error="Glass caustic single-pixel frame identity is unavailable";return false;}
    frame=std::move(candidate);error.clear();return true;
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
    // Grating paths keep single-pixel deposits; a splat frame needs its owner.
    const bool splat=frame.splatVersion!=0;
    if(splat && (frame.splatVersion!=kSplatVersion || grating || frame.gratingStaging || !expectedSubmission))
        return refuse("Glass caustic splat deposition requires an ordinary owned frame");
    // A receipt owned by the single-pixel frame is the cap overflow fallback. It
    // is valid only when the CPU reference overflows the same caps, and it is
    // then checked as that single-pixel frame's receipt.
    if(splat && readback.frameIdentity!=frame.identity) {
        auto single=frame;
        if(!disableSplat(prepared,single,error))return false;
        if(readback.frameIdentity!=single.identity || !readback.deposition.splatNormalisers.empty() || readback.deposition.splatGatherWork)
            return refuse("Glass caustic splat fallback receipt is not owned by the single-pixel frame");
        EnergyLedger fallback;
        if(!validateNativeReadback(prepared,single,readback,fallback,error,expectedSubmission,gratingPrerequisite,sourceBinding))return false;
        SplatReference reference;
        if(!buildSplatReference(frame,readback.deposits,splatCpuQuery(prepared,frame),reference,error,splatCpuOffsetQuery(prepared,frame)))return false;
        if(!reference.capOverflow)return refuse("Glass caustic splat fell back to single-pixel deposits without a cap overflow");
        output=fallback;error.clear();return true;
    }
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
    struct PixelSum {double physical=0;float ordered=0;double areaTolerance=0;};
    std::map<std::uint64_t,PixelSum> expectedPixels;
    std::vector<double> outgoing(frame.work.size()),absorbed(frame.work.size()),fraction(frame.work.size());
    std::vector<std::size_t> counts(frame.work.size());
    std::vector<std::uint64_t> observedWork(frame.work.size());
    std::vector<std::uint32_t> minimumMicroEvents(frame.work.size());
    std::set<std::tuple<std::uint32_t,std::uint32_t,CausticHistory,std::int32_t>> branches;
    std::set<std::tuple<std::uint32_t,std::uint32_t,std::array<std::uint8_t,32>,std::vector<std::uint8_t>>> microBranches;
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
            || !deposit.reflectionHistory.clearFrom(deposit.interfaces))
            return refuse("Glass caustic native deposit branch identity or fraction is invalid");
        if(frame.multipleScatterPolicy) {
            std::uint32_t events=0;
            if(!validMacroAncestry(deposit,*frame.multipleScatterPolicy,frame.ior[frame.work[deposit.work].band]==1,events)
                || !microBranches.emplace(deposit.work,deposit.interfaces,deposit.ancestry,deposit.ancestryTail).second)
                return refuse("Glass caustic native exact microfacet ancestry is invalid or duplicated");
            minimumMicroEvents[deposit.work]=std::max(minimumMicroEvents[deposit.work],events);
        } else {
            if(deposit.gratingVersion) {
                if(!grating||deposit.gratingVersion!=kGratingIncidentVersion||deposit.gratingIncident>=readback.gratingIncidents->incidents.size())
                    return refuse("Glass caustic grating child owner is invalid");
            } else if(deposit.gratingIncident!=UINT32_MAX||deposit.gratingOrder||deposit.ancestry!=std::array<std::uint8_t,32>{}
                ||std::any_of(deposit.ancestryTail.begin(),deposit.ancestryTail.end(),[](std::uint8_t token){return token!=0;}))
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
        // The native pixel is adopted only within the binary64 boundary margin;
        // every later camera query and splat uses that same native pixel.
        PixelFootprint footprint;double areaTolerance=0;
        if(!projectReceiverFootprint(frame,deposit.receiver,&deposit.footprint,footprint,areaTolerance,error))return false;
        if(!nativeFootprintMatches(deposit.footprint,footprint,areaTolerance))
            return refuse("Glass caustic native projected pixel footprint changed");
        bool visible=false;
        if(footprint.projected)
        {
            if((deposit.cameraHit.status!=0 && deposit.cameraHit.status!=1) || (deposit.pixelHit.status!=0 && deposit.pixelHit.status!=1))
                return refuse("Glass caustic native camera visibility query is incomplete");
            const auto origin=cameraEndpointOrigin(frame,deposit.receiver.position),delta=sub(deposit.receiver.position,origin);
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
            const auto expected=advance(cameraOrigin(frame,float(footprint.x)+.5f,float(footprint.y)+.5f),ray,deposit.pixelHit.distance);
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
                    || r.instance!=deposit.receiver.instanceId
                    || !rasterOwnsPixelHit(frame,deposit.receiver,deposit.pixelHit,r.triangle)
                    || !nearPower(r.linearDepth,-local.z))
                    return refuse("Glass caustic native raster depth or exact triangle IDs changed");
                candidate.cameraVisible[work.band]+=deposit.outgoingPower;
                // Splat frames check their pixels in validateSplatReadback.
                if(!splat)
                {
                    constexpr double pi=3.14159265358979323846;
                    const double radiance=deposit.outgoingPower/(pi*footprint.worldArea);
                    if(!std::isfinite(radiance) || radiance>surfacematerial::kMaximumLinearColorComponent)
                        return refuse("Glass caustic native pixel splat is unrepresentable");
                    const auto pixel=(std::uint64_t(footprint.y)*frame.width+footprint.x)*8+work.band;
                    auto& sum=expectedPixels[pixel];sum.physical+=radiance;sum.areaTolerance=std::max(sum.areaTolerance,areaTolerance);
                    // No regrouping by band, pairwise tree or fixed-point scaling.
                    // The native segment leader uses these same float32 operations.
                    const float nativeRadiance=deposit.outgoingPower/(float(pi)*deposit.footprint.worldArea);
                    sum.ordered+=nativeRadiance;
                }
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
        for(float power:{path.launchedPower,path.depositedPower,path.absorbedPower,path.escapedPower,path.unresolvedPower,path.maskedPower,path.thinAbsorbedPower})
            if(!std::isfinite(power) || power<0)return refuse("Glass caustic native terminal power is invalid");
        if(path.thinAbsorbedPower!=0 && (!prepared.absorber.isValid() || path.sourceBlocked))
            return refuse("Glass caustic native thin-absorbed power has no absorber owner");
        if((path.sourceBlocked && counts[w]) || fraction[w]>1.0002
            || !nearPower(path.launchedPower,double(path.depositedPower)+path.absorbedPower+path.escapedPower+path.unresolvedPower+path.maskedPower
                +path.thinAbsorbedPower)
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
        candidate.thinAbsorbed[b]+=path.thinAbsorbedPower;
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
    // Contract: every splat frame reaches validateSplatReadback here, after the
    // per-deposit visibility, footprint, power and path checks above.
    if(splat) {
        if(!validateSplatReadback(frame,readback,splatCpuQuery(prepared,frame),error,splatCpuOffsetQuery(prepared,frame)))return false;
        output=candidate;error.clear();return true;
    }
    const auto checkBand=[&](float actual,const PixelSum* found) {
        const double expected=found?found->physical:0;
        const float ordered=found?found->ordered:0;
        // A positive sum keeps the largest relative footprint area tolerance of its terms.
        const double physicalTolerance=5e-4+(found?found->areaTolerance:0);
        // Native division may differ by a final rounding on GL and Metal. The
        // receipt still proves the unique reduction order and exact permutation.
        if(readback.deposition.version && !nearPower(actual,ordered,4e-6))
            return refuse("Glass caustic native canonical float32 reduction mismatch");
        if(!std::isfinite(actual) || actual<0 || actual>surfacematerial::kMaximumLinearColorComponent || !nearPower(actual,expected,physicalTolerance))
            return refuse("Glass caustic native spectral accumulation overflow or receipt mismatch");
        return true;
    };
    // Every pixel band is still checked. expectedPixels is ordered by band
    // index, so one cursor walks it beside the readback instead of a tree
    // lookup per band (66 Mi lookups at 4K). An undeposited band that reads
    // +0 or -0 passes checkBand(actual,nullptr), so only nonzero values pay
    // for the full check.
    if(!checkSpectralBands(readback.spectralRadiance,expectedPixels,checkBand))return false;
    output=candidate;error.clear();return true;
}
// CPU-only power oracle companion. NativeReadback must come from completed GPU
// work; this diagnostic never manufactures it or populates an accumulator.
inline bool referencePower(const Prepared& prepared,const Frame& frame,EnergyLedger& output,std::string& error)
{
    if(!prepared.branchPolicy) {error="Glass caustic power diagnostic requires owning admitted branch policy";return false;}
    if(!validateTables(prepared,frame,error))return false;
    if(!diagnosticQueryWorkWithinBudget(frame,std::uint64_t(frame.work.size())
        *(2*(policyInterfaces(prepared)+3)+prepared.branchPolicy->description().maximumVisits),error))return false;
    EnergyLedger candidate;
    for(const auto& work:frame.work)
    {
        BranchReceipt receipt;
        if(!traceBranches(prepared,frame,frame.entries[work.entry],work.band,*prepared.branchPolicy,receipt,error,work.interfaceSample))return false;
        const auto b=work.band;const double power=work.incidentPower;candidate.potential[b]+=power;
        if(receipt.sourceBlocked) {candidate.sourceBlocked[b]+=power;continue;}
        candidate.launched[b]+=power;candidate.escaped[b]+=power*receipt.escapedFraction;
        candidate.unresolved[b]+=power*receipt.unresolvedFraction;candidate.masked[b]+=power*receipt.maskedFraction;
        candidate.thinAbsorbed[b]+=power*receipt.absorberFraction;
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
        +"\n#define CAUSTIC_BRANCH_TEXELS "+std::to_string(kPendingBranchTexels)
        +"\n#define CAUSTIC_MAX_VISITS "+std::to_string(kMaximumBranchVisits)
        +"\n#define CAUSTIC_MAX_ANCESTRY_INTERFACES "+std::to_string(kMaximumAncestryInterfaces)
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
// projection.x below zero is an orthographic camera, minus half its view height. Its
// rays are parallel along the view axis, each from the pixel's point on the camera plane.
vec3 causticCameraRay(vec2 pixel,vec4 rotation,vec4 projection,vec2 extent) {
    if(projection.x<0.0) return normalize(causticRotate(rotation,vec3(0.0,0.0,-1.0)));
    vec2 p=vec2(2.0*pixel.x/extent.x-1.0,1.0-2.0*pixel.y/extent.y);
    return normalize(causticRotate(rotation,vec3(p.x*projection.x*projection.y,p.y*projection.x,-1.0)));
}
vec3 causticCameraOrigin(vec2 pixel,vec3 camera,vec4 rotation,vec4 projection,vec2 extent) {
    if(projection.x>=0.0) return camera;
    vec2 p=vec2(2.0*pixel.x/extent.x-1.0,1.0-2.0*pixel.y/extent.y);
    return camera+causticRotate(rotation,vec3(-p.x*projection.x*projection.y,-p.y*projection.x,0.0));
}
// The start of the camera's ray to a world point: the camera, or the point's
// projection onto an orthographic camera's plane.
vec3 causticEndpointOrigin(vec3 position,vec3 camera,vec4 rotation,vec4 projection) {
    if(projection.x>=0.0) return camera;
    vec3 forward=normalize(causticRotate(rotation,vec3(0.0,0.0,-1.0)));
    return position-forward*dot(position-camera,forward);
}
bool causticPixelCorner(vec2 pixel,vec3 position,vec3 normal,vec3 camera,
    vec4 rotation,vec4 projection,vec2 extent,out vec3 corner) {
    vec3 ray=causticCameraRay(pixel,rotation,projection,extent);
    vec3 origin=causticCameraOrigin(pixel,camera,rotation,projection,extent);
    float denominator=dot(normal,ray),distance=dot(normal,position-origin)/denominator;
    if(!causticFinite(distance)||denominator>=-1e-6||distance<=0.0||distance>projection.w*1e4) return false;
    corner=origin+ray*distance;return true;
}
// 0 is outside/back-facing, -1 refuses a horizon/overflow footprint, 1 projects.
// footprint is top-first x/y, linear depth, exact receiver-plane pixel area.
int causticProjectReceiver(vec3 position,vec3 normal,vec3 camera,vec4 rotation,
    vec4 projection,vec2 extent,out vec4 footprint) {
    vec3 local=causticRotate(vec4(-rotation.xyz,rotation.w),position-camera);
    float depth=-local.z;footprint=vec4(0.0,0.0,depth,0.0);
    if(depth<projection.z||depth>projection.w) return 0;
    float spread=projection.x<0.0?-projection.x:depth*projection.x;
    vec2 pixel=vec2((local.x/(spread*projection.y)+1.0)*0.5*extent.x,
        (1.0-local.y/spread)*0.5*extent.y);
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
// Splat helpers for the GL and Metal splat passes, appended after
// nativeMathSource. No released program includes them.
static_assert(kSplatEntryRadiusScale==2.5 && kSplatMinimumRadiusPixels==1.5,
    "nativeSplatMathSource spells the splat radius constants literally");
static_assert(kSplatSpreadCosineLow==.1 && kSplatSpreadCosineHigh==.2,
    "nativeSplatMathSource spells the splat spread cosines literally");
inline std::string nativeSplatMathSource()
{
    return "\n#define CAUSTIC_SPLAT_VERSION "+std::to_string(kSplatVersion)
        +"\n#define CAUSTIC_SPLAT_RADIUS "+std::to_string(kMaximumSplatRadiusPixels)
        +"\n#define CAUSTIC_SPLAT_TILE "+std::to_string(kSplatTile)
        +"\n#define CAUSTIC_SPLAT_TILES_PER_DEPOSIT "+std::to_string(kSplatTilesPerDeposit)
        +"\n#define CAUSTIC_SPLAT_TILE_CAP "+std::to_string(kMaximumSplatDepositsPerTile)+"\n"+R"GLSL(
float causticSplatKernel(float distance,float radius) {
    if(!(radius>0.0)||!(distance>=0.0)||!(distance<radius)) return 0.0;
    float q=distance/radius,w=1.0-q*q;return w*w;
}
// Smallest singular value of the receiver-plane Jacobian of top-first pixel p,
// from the corners causticProjectReceiver uses. 0 refuses the deposit.
float causticSplatPixelScale(uvec2 pixel,vec3 position,vec3 normal,vec3 camera,vec4 rotation,vec4 projection,vec2 extent) {
    vec2 p=vec2(pixel);vec3 a,b,c,d;
    if(!causticPixelCorner(p,position,normal,camera,rotation,projection,extent,a)
        ||!causticPixelCorner(p+vec2(1.0,0.0),position,normal,camera,rotation,projection,extent,b)
        ||!causticPixelCorner(p+vec2(1.0),position,normal,camera,rotation,projection,extent,c)
        ||!causticPixelCorner(p+vec2(0.0,1.0),position,normal,camera,rotation,projection,extent,d)) return 0.0;
    vec3 ex=0.5*((b-a)+(c-d)),ey=0.5*((d-a)+(c-b));
    float xx=dot(ex,ex),yy=dot(ey,ey),xy=dot(ex,ey),trace=xx+yy;
    float det=max(0.0,xx*yy-xy*xy),largest=0.5*(trace+sqrt(max(0.0,trace*trace-4.0*det)));
    float scale=largest>0.0?sqrt(det/largest):0.0;
    return causticFinite(scale)?scale:0.0;
}
float causticSplatRadiusFromScale(float entryArea,float pixelScale) {
    return clamp(2.5*sqrt(entryArea/3.14159265358979323846),1.5*pixelScale,float(CAUSTIC_SPLAT_RADIUS)*pixelScale);
}
// Neighbour weight scale of a deposit seen at view cosine c: 0 at c<=0.1, 1 at
// c>=0.2, linear between. The CPU reference is splatSpread.
float causticSplatSpread(vec3 position,vec3 normal,vec3 camera) {
    float cosine=abs(dot(normal,normalize(position-camera)));
    return clamp((cosine-0.1)/(0.2-0.1),0.0,1.0);
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
