#pragma once

#include "fixture_glass_grating_entry_cells.h"

namespace arbitgpu::glasscaustics
{
// Completed native boundary samples only. This contract supplies no interior
// coverage, camera visibility, finite footprint, irradiance or Color authority.
inline constexpr std::uint32_t kGratingCellTransportVersion=1;
inline constexpr std::size_t kGratingCellTransportHeaderStride=4,kGratingCellTraceStepStride=7,kGratingCellTracePaths=4;
enum class GratingCellTransportDisposition : std::uint32_t { BoundaryWitness=1,Unsupported=2 };
struct NativeGratingCellTraceStep
{
    Point origin{},direction{};
    std::uint32_t depth=0,token=0;
    opaquevisibility::ClosestHit hit;
    float signedBias=0; // Signed distance along actual outwardNormal.
};
struct NativeGratingCellTransportRecord
{
    std::uint32_t version=0,incident=UINT32_MAX,work=0,entry=0,band=0,interfaces=0;
    GratingCellTransportDisposition disposition=GratingCellTransportDisposition::Unsupported;
    std::array<std::vector<NativeGratingCellTraceStep>,kGratingCellTracePaths> paths;
    std::uint64_t queryWork=0;
    bool complete=false;
};
struct GratingCellTransportBounds
{
    std::uint64_t maximumRecords=0,maximumInterfaces=0,recordStride=0,maximumQueries=0,queryWork=0;
    std::uint64_t hostBytes=0,nativeBytes=0,bytes=0;
};
struct GratingCellTransportBinding
{
    std::shared_ptr<const GratingEntryCellBinding> sourceCells;
    std::shared_ptr<const NativeGratingReadback> pointReadback;
    std::shared_ptr<const NativeFixtureSceneFrame> nativeOwner;
    SubmissionIdentity pointSubmission,submission;
    GratingCellTransportBounds bounds;
    std::string identity;
};
struct NativeGratingCellSourceUpload
{
    std::uint32_t version=0;
    std::shared_ptr<const GratingEntryCellBinding> sourceCells;
    std::shared_ptr<const NativeGratingReadback> pointReadback;
    SubmissionIdentity pointSubmission,submission;
    std::uint64_t bindingGeneration=0;
    std::vector<Texel> cellTexels; // Actual native input readback after upload.
    bool complete=false;
};
// Implemented by the existing backend-owned candidate. The echo must come
// from its dedicated input buffer, never a copy of canonical host vertices.
class NativeGratingCellSourceBinding
{
public:
    virtual ~NativeGratingCellSourceBinding()=default;
    virtual const NativeFixtureSceneFrame& owner() const noexcept=0;
    virtual const NativeGratingCellSourceUpload& cellUpload() const noexcept=0;
protected:
    NativeGratingCellSourceBinding()=default;
};
struct NativeGratingCellTransportReadback
{
    std::uint32_t version=0;
    std::shared_ptr<const GratingCellTransportBinding> binding;
    SubmissionIdentity submission;
    std::uint64_t bindingGeneration=0,queryWork=0;
    std::vector<NativeGratingCellTransportRecord> records;
    bool complete=false,overflow=false;
};
inline bool gratingCellSubmissionContinuity(const SubmissionIdentity& point,const SubmissionIdentity& next)
{
    return point.rendererGeneration&&point.submissionSerial&&point.rasterGeneration&&point.queryGeneration&&point.scratchGeneration
        &&point.completedFence==point.submissionSerial&&next.frameIdentity==point.frameIdentity
        &&next.rendererGeneration==point.rendererGeneration&&next.rasterGeneration==point.rasterGeneration
        &&next.queryGeneration==point.queryGeneration&&next.scratchGeneration&&next.scratchGeneration!=point.scratchGeneration
        &&next.submissionSerial>point.submissionSerial&&next.completedFence==next.submissionSerial;
}
inline std::uint64_t gratingCellQueryAllowance(const Frame& frame)
{
    if(!std::isfinite(frame.query.control[0])||frame.query.control[0]<1
        ||frame.query.control[0]>=frame.query.texels.size()||std::floor(frame.query.control[0])!=frame.query.control[0])return 0;
    const auto& t=frame.query.texels[std::size_t(frame.query.control[0])-1];std::uint32_t lo=0,hi=0;
    if(!exactReceiptInteger(t[0],65535,lo)||!exactReceiptInteger(t[1],65535,hi))return 0;
    return lo|(std::uint64_t(hi)<<16);
}
inline bool calculateGratingCellTransportBounds(const GratingEntryCellBinding& cells,std::size_t incidents,
    GratingCellTransportBounds& output,std::string& error)
{
    const auto refuse=[&](const char* text){error=text;return false;};
    if(!cells.incidentBinding||!cells.incidentBinding->prepared.branchPolicy)return refuse("Cell transport incident parent is missing");
    const auto& parent=*cells.incidentBinding;const auto& frame=parent.frame;
    const auto interfaces=parent.prepared.branchPolicy->description().maximumInterfaces;
    const auto query=gratingCellQueryAllowance(frame);
    if(!interfaces||interfaces>kMaximumInterfaces||incidents>kMaximumEntrySamples*8*kMaximumBranchVisits
        ||incidents>parent.bounds.maximumIncidents
        ||!query||query>opaquevisibility::kMaximumClosestHitWork)return refuse("Cell transport schedule or query allowance is invalid");
    GratingCellTransportBounds bounds;bounds.maximumRecords=incidents;bounds.maximumInterfaces=interfaces;
    bounds.recordStride=kGratingCellTransportHeaderStride+kGratingCellTracePaths*(interfaces+1)*kGratingCellTraceStepStride;
    // A supplementary center replay is charged along with all three corners.
    // No original point/terminal work counter is reused or increased.
    bounds.maximumQueries=incidents*kGratingCellTracePaths*(interfaces+1);bounds.queryWork=bounds.maximumQueries*query;
    if(parent.bounds.combinedWork>kMaximumQueryWork||bounds.queryWork>kMaximumQueryWork-parent.bounds.combinedWork)
        return refuse("Cell transport replays exceed the existing whole-frame query cap");
    const auto recordsBytes=incidents*bounds.recordStride*sizeof(Texel);
    // Actual source echo retains a host copy and a native input/readback copy.
    // The source-cell reservation already charges its first native upload.
    // The uint locator map and its host staging copy are also simultaneous.
    // Identity strings include two private identities during validation and
    // five retained submission strings, with no copies of parent receipts.
    bounds.nativeBytes=recordsBytes+cells.bounds.nativeUploadBytes+incidents*sizeof(std::uint32_t);
    bounds.hostBytes=sizeof(GratingCellTransportBinding)+sizeof(NativeGratingCellTransportReadback)+sizeof(NativeGratingCellSourceUpload)
        +incidents*(sizeof(NativeGratingCellTransportRecord)+kGratingCellTracePaths*(interfaces+1)*sizeof(NativeGratingCellTraceStep))
        +recordsBytes+cells.bounds.nativeUploadBytes+incidents*sizeof(std::uint32_t)
        +2*cells.identity.size()+9*parent.identity.size()+640;
    bounds.bytes=bounds.hostBytes+bounds.nativeBytes;
    const auto maximum=renderpassoutput::kMaximumTotalBytes;
    if(frame.bounds.transientBytes>maximum||parent.bounds.bytes>maximum-frame.bounds.transientBytes
        ||cells.bounds.bytes>maximum-frame.bounds.transientBytes-parent.bounds.bytes
        ||bounds.bytes>maximum-frame.bounds.transientBytes-parent.bounds.bytes-cells.bounds.bytes)
        return refuse("Cell transport storage exceeds the existing whole-frame byte cap");
    output=bounds;error.clear();return true;
}
inline std::string gratingCellTransportIdentity(const GratingCellTransportBinding& binding)
{
    std::string identity;identity.reserve(binding.sourceCells->identity.size()+2*binding.pointSubmission.frameIdentity.size()+304);
    word(identity,kGratingCellTransportVersion);word(identity,binding.sourceCells->identity.size());identity+=binding.sourceCells->identity;
    for(const auto* submission:{&binding.pointSubmission,&binding.submission}) {
        word(identity,submission->frameIdentity.size());identity+=submission->frameIdentity;
        for(auto value:{submission->rendererGeneration,submission->submissionSerial,submission->rasterGeneration,
            submission->queryGeneration,submission->scratchGeneration,submission->completedFence})word(identity,value);
    }
    word(identity,binding.bounds.maximumRecords);word(identity,binding.bounds.maximumInterfaces);word(identity,binding.bounds.recordStride);
    return identity;
}
inline bool validateGratingCellTransportBinding(const GratingCellTransportBinding& binding,std::string& error)
{
    const auto refuse=[&](const char* text){error=text;return false;};
    if(!binding.sourceCells||!binding.sourceCells->incidentBinding||!binding.pointReadback||!binding.nativeOwner
        ||!gratingCellSubmissionContinuity(binding.pointSubmission,binding.submission)
        ||!(binding.pointReadback->submission==binding.pointSubmission))return refuse("Cell transport exact parent or supplemental fence changed");
    const auto& parent=*binding.sourceCells->incidentBinding;const auto& owner=*binding.nativeOwner;
    if(!validateGratingEntryCellBinding(*binding.sourceCells,binding.sourceCells->incidentBinding,error))return false;
    if(owner.width()!=parent.frame.width||owner.height()!=parent.frame.height||!owner.nativeResourceCacheIdentity()
        ||owner.backend()!=(parent.prepared.glass.sourceMaterial->backend==NativeFixtureMaterialBackend::OpenGl?"opengl":"metal"))
        return refuse("Cell transport native candidate owner changed");
    const auto* source=parent.frame.sourceDirectionalIrradiance.version?dynamic_cast<const NativeDirectionalSourceBinding*>(&owner):nullptr;
    if(!validateGratingIncidentReadback(parent,*binding.pointReadback,binding.pointSubmission,error,source))return false;
    GratingCellTransportBounds exact;
    if(!calculateGratingCellTransportBounds(*binding.sourceCells,binding.pointReadback->incidents.size(),exact,error))return false;
    if(std::tie(binding.bounds.maximumRecords,binding.bounds.maximumInterfaces,binding.bounds.recordStride,binding.bounds.maximumQueries,
            binding.bounds.queryWork,binding.bounds.hostBytes,binding.bounds.nativeBytes,binding.bounds.bytes)
        !=std::tie(exact.maximumRecords,exact.maximumInterfaces,exact.recordStride,exact.maximumQueries,
            exact.queryWork,exact.hostBytes,exact.nativeBytes,exact.bytes)
        ||binding.identity!=gratingCellTransportIdentity(binding))return refuse("Cell transport private identity or reservation changed");
    error.clear();return true;
}
inline bool prepareGratingCellTransportBinding(const std::shared_ptr<const GratingEntryCellBinding>& cells,
    const std::shared_ptr<const NativeGratingReadback>& point,const std::shared_ptr<const NativeFixtureSceneFrame>& nativeOwner,
    const SubmissionIdentity& pointExpected,const SubmissionIdentity& supplementalExpected,
    std::shared_ptr<const GratingCellTransportBinding>& output,std::string& error)
{
    if(!cells||!point) {error="Cell transport source or completed point receipt is missing";return false;}
    auto binding=std::make_shared<GratingCellTransportBinding>();binding->sourceCells=cells;binding->pointReadback=point;
    binding->nativeOwner=nativeOwner;binding->pointSubmission=pointExpected;binding->submission=supplementalExpected;
    if(!calculateGratingCellTransportBounds(*cells,point->incidents.size(),binding->bounds,error))return false;
    binding->identity=gratingCellTransportIdentity(*binding);
    if(!validateGratingCellTransportBinding(*binding,error))return false;
    output=std::move(binding);error.clear();return true;
}
// GPU record keyed by the ORIGINAL dense point-readback incident locator.
// Header0 version/locator/work/entry;1 band/interfaces/disposition/complete;
// 2 queryWork in four 16-bit words;3 step counts center/corner0/corner1/corner2.
// Then four fixed path regions of (maximumInterfaces+1)*seven texels:
// step0 origin.xyz/depth;1 direction.xyz/token;2 status/distance/entering/workLo;
// 3 hitPosition.xyz/workHi;4 outwardNormal.xyz/signedBias;
// 5 split object/material;6 split instance/triangle. Endpoint token/bias are0.
// Every inactive slot is zero. Counts include the attempted final failed query.
inline bool decodeGratingCellTransportRecord(const Texel* row,const GratingCellTransportBinding& binding,
    NativeGratingCellTransportRecord& output,std::string& error)
{
    const auto refuse=[&](){error="Native cell transport row is malformed";return false;};
    if(!row||!binding.sourceCells||!binding.pointReadback||!binding.bounds.maximumInterfaces
        ||binding.bounds.maximumInterfaces>kMaximumInterfaces
        ||binding.bounds.recordStride!=kGratingCellTransportHeaderStride+kGratingCellTracePaths*(binding.bounds.maximumInterfaces+1)*kGratingCellTraceStepStride)return refuse();
    for(std::size_t i=0;i<binding.bounds.recordStride;++i)for(float x:row[i])if(!std::isfinite(x))return refuse();
    NativeGratingCellTransportRecord r;std::uint32_t disposition=0,words[4]{};
    if(row[0][0]!=kGratingCellTransportVersion||row[1][3]!=1
        ||!exactReceiptInteger(row[0][1],std::uint32_t(binding.pointReadback->incidents.size()),r.incident)
        ||r.incident>=binding.pointReadback->incidents.size()
        ||!exactReceiptInteger(row[0][2],kMaximumEntrySamples*8,r.work)||!exactReceiptInteger(row[0][3],kMaximumEntrySamples,r.entry)
        ||!exactReceiptInteger(row[1][0],7,r.band)||!exactReceiptInteger(row[1][1],std::uint32_t(binding.bounds.maximumInterfaces),r.interfaces)
        ||!exactReceiptInteger(row[1][2],2,disposition)||!disposition)return refuse();
    r.version=kGratingCellTransportVersion;r.complete=true;r.disposition=GratingCellTransportDisposition(disposition);
    for(unsigned a=0;a<4;++a)if(!exactReceiptInteger(row[2][a],65535,words[a]))return refuse();
    r.queryWork=std::uint64_t(words[0])|(std::uint64_t(words[1])<<16)|(std::uint64_t(words[2])<<32)|(std::uint64_t(words[3])<<48);
    const auto slots=binding.bounds.maximumInterfaces+1;
    for(unsigned path=0;path<4;++path) {
        std::uint32_t count=0;if(!exactReceiptInteger(row[3][path],std::uint32_t(slots),count))return refuse();
        r.paths[path].reserve(count);
        for(std::size_t depth=0;depth<slots;++depth) {
            const auto* t=row+kGratingCellTransportHeaderStride+(path*slots+depth)*kGratingCellTraceStepStride;
            if(depth>=count) {
                for(unsigned i=0;i<kGratingCellTraceStepStride;++i)if(t[i]!=Texel{})return refuse();
                continue;
            }
            NativeGratingCellTraceStep step;std::uint32_t status=0,entering=0,lo=0,hi=0;
            if(!exactReceiptInteger(t[0][3],std::uint32_t(binding.bounds.maximumInterfaces),step.depth)
                ||!exactReceiptInteger(t[1][3],65,step.token)||!exactReceiptInteger(t[2][0],1,status)
                ||!exactReceiptInteger(t[2][2],1,entering)||!exactReceiptInteger(t[2][3],65535,lo)
                ||!exactReceiptInteger(t[3][3],65535,hi))return refuse();
            step.origin={t[0][0],t[0][1],t[0][2]};step.direction={t[1][0],t[1][1],t[1][2]};
            step.hit.status=int(status);step.hit.distance=t[2][1];step.hit.entering=entering!=0;step.hit.work=lo|(hi<<16);
            step.hit.position={t[3][0],t[3][1],t[3][2]};step.hit.outwardNormal={t[4][0],t[4][1],t[4][2]};step.signedBias=t[4][3];
            const auto id=[&](float low,float high,std::uint32_t& result) {
                std::uint32_t a=0,b=0;if(!exactReceiptInteger(low,65535,a)||!exactReceiptInteger(high,65535,b))return false;
                result=a|(b<<16);return true;
            };
            if(!id(t[5][0],t[5][1],step.hit.objectId)||!id(t[5][2],t[5][3],step.hit.materialId)
                ||!id(t[6][0],t[6][1],step.hit.instanceId)||!id(t[6][2],t[6][3],step.hit.triangleId))return refuse();
            r.paths[path].push_back(step);
        }
    }
    output=std::move(r);error.clear();return true;
}
inline bool sameGratingCellHitOwner(const opaquevisibility::ClosestHit& a,const opaquevisibility::ClosestHit& b)
{return std::tie(a.objectId,a.materialId,a.instanceId,a.triangleId)==std::tie(b.objectId,b.materialId,b.instanceId,b.triangleId);}
inline bool validateGratingCellTransportRecord(const GratingCellTransportBinding& binding,
    const NativeGratingCellTransportRecord& record,std::size_t locator,std::string& error)
{
    const auto refuse=[&](const char* text){error=text;return false;};
    if(!binding.pointReadback||!binding.sourceCells||!binding.sourceCells->incidentBinding
        ||locator>=binding.pointReadback->incidents.size())return refuse("Cell transport incident locator is outside its point parent");
    const auto& point=binding.pointReadback->incidents[locator];const auto& parent=*binding.sourceCells->incidentBinding;
    const auto& frame=parent.frame;
    if(point.work>=frame.work.size()||frame.work[point.work].entry>=frame.entries.size()
        ||frame.work[point.work].entry>=binding.sourceCells->cells.size()||point.band>=8)
        return refuse("Cell transport point work or source cell is outside the retained table");
    const auto& work=frame.work[point.work];const auto& entry=frame.entries[work.entry];
    if(record.version!=kGratingCellTransportVersion||!record.complete||record.incident!=locator||record.work!=point.work
        ||record.entry!=work.entry||record.band!=point.band||record.interfaces!=point.interfaces
        ||record.interfaces>binding.bounds.maximumInterfaces||record.paths[0].size()!=point.interfaces+1
        ||(record.disposition!=GratingCellTransportDisposition::BoundaryWitness&&record.disposition!=GratingCellTransportDisposition::Unsupported))
        return refuse("Cell transport point locator, source entry or completion changed");
    const Point source{frame.sourceDirectionTrace[0],frame.sourceDirectionTrace[1],frame.sourceDirectionTrace[2]};
    const auto& cell=binding.sourceCells->cells[work.entry];std::uint64_t queries=0;bool allSupported=true;
    const auto allowance=gratingCellQueryAllowance(frame);
    for(unsigned path=0;path<4;++path) {
        const auto& steps=record.paths[path];bool supported=steps.size()==point.interfaces+1,inside=false,continuing=true;
        if(steps.size()>point.interfaces+1||steps.capacity()>binding.bounds.maximumInterfaces+1)
            return refuse("Cell transport trace exceeds its retained depth or host reservation");
        const Point start=path?cell.vertices[path-1]:entry.position;
        Point expectedOrigin=advance(start,source,2*bias(start)),expectedDirection=advance({},source,-1);
        for(std::size_t depth=0;depth<steps.size();++depth) {
            const auto& step=steps[depth];const auto& hit=step.hit;
            if(!continuing||step.depth!=depth||step.token!=(depth<point.interfaces?point.ancestry[depth]:0u)
                ||!finite(step.origin)||!finite(step.direction)||std::abs(dot(step.direction,step.direction)-1)>1e-4f
                ||!sameGratingPoint(step.origin,expectedOrigin,8*bias(expectedOrigin))||!sameGratingPoint(step.direction,expectedDirection)
                ||!std::isfinite(step.signedBias)||hit.status<0||hit.status>1||!hit.work||hit.work>allowance
                ||!std::isfinite(hit.distance)||hit.distance<0||hit.distance>frame.sourceDirectionTrace[3])
                return refuse("Cell transport ray, token or query work is invalid");
            queries+=hit.work;
            const auto* triangle=hit.status==1?hitTriangle(frame,hit):nullptr;
            if(hit.status==0) {
                if(hit.entering||hit.objectId||hit.materialId||hit.instanceId||hit.triangleId||hit.position!=Point{}
                    ||hit.outwardNormal!=Point{}||step.signedBias!=0)return refuse("Cell transport miss carries fabricated hit data");
                supported=false;continuing=false;
            } else {
                if(!triangle||!sameGratingPoint(advance(step.origin,step.direction,hit.distance),hit.position,8*bias(hit.position)))
                    return refuse("Cell transport nearest-hit geometry is invalid");
                const auto& center=record.paths[0][depth].hit;
                const bool ownerMatches=path?sameGratingCellHitOwner(hit,center):true;
                if(depth==point.interfaces) {
                    if(step.signedBias!=0)return refuse("Cell transport endpoint carries an interface bias");
                    if(inside||!hit.entering||!ownerMatches||!sameGratingCellHitOwner(hit,point.hit))supported=false;
                    if(path==0&&(!sameGratingPoint(step.origin,point.origin)||!sameGratingPoint(step.direction,point.direction)
                        ||!sameGratingPoint(hit.position,point.hit.position,8*bias(point.hit.position))
                        ||!sameGratingPoint(hit.outwardNormal,point.hit.outwardNormal)||hit.entering!=point.hit.entering
                        ||hit.work!=point.hit.work||std::abs(hit.distance-point.hit.distance)>8*bias(point.hit.position)))
                        return refuse("Cell transport center replay differs from the original native point");
                } else {
                    bool matches=triangle->glass&&ownerMatches&&inside!=hit.entering;
                    if(depth==0)matches=matches&&hit.objectId==entry.object&&hit.materialId==entry.material
                        &&hit.instanceId==entry.instance&&hit.triangleId==entry.triangle;
                    Point oriented=hit.outwardNormal;if(inside)oriented=advance({},oriented,-1);
                    Interface optics;const bool reflected=step.token==1;
                    if(matches)matches=interface(step.direction,oriented,inside?frame.ior[point.band]:1,inside?1:frame.ior[point.band],optics)
                        &&(reflected?optics.reflection>0:!optics.tir&&optics.reflection<1);
                    if(matches) {
                        const float signedBias=(inside?-1.0f:1.0f)*(reflected?1.0f:-1.0f)*bias(hit.position);
                        if(step.signedBias!=signedBias)return refuse("Cell transport signed interface bias changed");
                        expectedOrigin=advance(hit.position,hit.outwardNormal,signedBias);
                        expectedDirection=reflected?optics.reflected:optics.transmitted;if(!reflected)inside=!inside;
                    } else {
                        if(step.signedBias!=0)return refuse("Unsupported cell interface carries an applied branch bias");
                        supported=false;continuing=false;
                    }
                }
            }
            if(!continuing&&depth+1!=steps.size())return refuse("Unsupported cell trace continues past its failed boundary query");
        }
        if(path==0&&!supported)return refuse("Cell transport center replay has no original point lineage");
        allSupported=allSupported&&supported;
    }
    if(record.queryWork!=queries||queries>binding.bounds.queryWork
        ||(record.disposition==GratingCellTransportDisposition::BoundaryWitness)!=allSupported)
        return refuse("Cell transport boundary disposition or separate query charge changed");
    error.clear();return true;
}
inline bool validateGratingCellTransportReadback(const GratingCellTransportBinding& binding,
    const NativeGratingCellTransportReadback& readback,const std::shared_ptr<const NativeFixtureSceneFrame>& exactNativeOwner,
    const SubmissionIdentity& supplementalExpected,std::string& error)
{
    const auto refuse=[&](const char* text){error=text;return false;};
    if(!binding.sourceCells||!binding.pointReadback||!exactNativeOwner||binding.nativeOwner!=exactNativeOwner||readback.binding.get()!=&binding
        ||!(supplementalExpected==binding.submission)||!(readback.submission==supplementalExpected)
        ||readback.version!=kGratingCellTransportVersion||!readback.complete||readback.overflow
        ||readback.records.size()!=binding.pointReadback->incidents.size()||readback.records.capacity()>binding.bounds.maximumRecords)return refuse("Cell transport native candidate, fence or dense extent changed");
    if(!validateGratingCellTransportBinding(binding,error))return false;
    const auto* source=dynamic_cast<const NativeGratingCellSourceBinding*>(exactNativeOwner.get());
    if(!source||&source->owner()!=exactNativeOwner.get())return refuse("Cell transport has no exact native input-buffer owner");
    const auto& upload=source->cellUpload();
    if(upload.version!=kGratingCellTransportVersion||!upload.complete||!upload.bindingGeneration
        ||upload.bindingGeneration!=readback.bindingGeneration||upload.sourceCells!=binding.sourceCells
        ||upload.pointReadback!=binding.pointReadback||!(upload.pointSubmission==binding.pointSubmission)
        ||!(upload.submission==binding.submission)||upload.cellTexels.size()!=binding.sourceCells->cellTexels.size()
        ||upload.cellTexels.capacity()>binding.sourceCells->cellTexels.size()
        ||std::memcmp(upload.cellTexels.data(),binding.sourceCells->cellTexels.data(),upload.cellTexels.size()*sizeof(Texel))!=0)
        return refuse("Cell transport source upload or original point completion changed");
    std::uint64_t queryWork=0;
    for(std::size_t i=0;i<readback.records.size();++i) {
        if(!validateGratingCellTransportRecord(binding,readback.records[i],i,error))return false;
        if(readback.records[i].queryWork>binding.bounds.queryWork-queryWork)return refuse("Cell transport native query work overflowed its separate reservation");
        queryWork+=readback.records[i].queryWork;
    }
    if(queryWork!=readback.queryWork)return refuse("Cell transport native total query work changed");
    error.clear();return true;
}
}
