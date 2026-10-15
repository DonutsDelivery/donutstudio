#pragma once

#include "fixture_glass_grating_cell_transport.h"
#include <map>

namespace arbitgpu::glasscaustics
{
// Proof domains only. Subdivision never changes source quadrature or flux.
inline constexpr std::uint32_t kGratingCellCoverageVersion=1,kGratingCellCoverageRule=1;
inline constexpr unsigned kGratingCellCoverageDepth=8;
inline constexpr std::uint32_t kGratingCellCoverageDenominator=1u<<kGratingCellCoverageDepth;
inline constexpr std::size_t kMaximumGratingCellCoverageLeaves=256;
enum class GratingCellCoverageDisposition : std::uint32_t { InteriorCertified=1,Unsupported=2 };
struct GratingCellDyadicTriangle
{
    std::uint32_t level=0,code=0;
    std::array<std::array<std::uint32_t,2>,3> vertices{};
};
inline bool gratingCellDyadicTriangle(std::uint32_t level,std::uint32_t code,
    GratingCellDyadicTriangle& output)
{
    if(level>kGratingCellCoverageDepth||code>=(1u<<(2*level)))return false;
    using Vertex=std::array<std::uint32_t,2>;
    GratingCellDyadicTriangle result;result.level=level;result.code=code;
    result.vertices={{{0,0},{kGratingCellCoverageDenominator,0},{0,kGratingCellCoverageDenominator}}};
    const auto midpoint=[](Vertex a,Vertex b){return Vertex{(a[0]+b[0])/2,(a[1]+b[1])/2};};
    for(unsigned depth=0;depth<level;++depth) {
        const auto p=result.vertices;const auto a=midpoint(p[0],p[1]),b=midpoint(p[1],p[2]),c=midpoint(p[2],p[0]);
        switch((code>>(2*(level-depth-1)))&3u) {
            case 0:result.vertices={p[0],a,c};break;
            case 1:result.vertices={a,p[1],b};break;
            case 2:result.vertices={c,b,p[2]};break;
            case 3:result.vertices={a,b,c};break;
        }
    }
    output=result;return true;
}
inline bool validateGratingCellDyadicPartition(const std::vector<GratingCellDyadicTriangle>& leaves,std::string& error)
{
    if(leaves.empty()||leaves.size()>kMaximumGratingCellCoverageLeaves||leaves.capacity()>kMaximumGratingCellCoverageLeaves)
    {error="Cell coverage dyadic partition extent is invalid";return false;}
    std::uint32_t cursor=0;
    for(const auto& leaf:leaves) {
        GratingCellDyadicTriangle exact;
        if(!gratingCellDyadicTriangle(leaf.level,leaf.code,exact)||leaf.vertices!=exact.vertices)
        {error="Cell coverage leaf has noncanonical integer vertices";return false;}
        const unsigned shift=2*(kGratingCellCoverageDepth-leaf.level);
        const auto begin=leaf.code<<shift,end=(leaf.code+1)<<shift;
        // Canonical prefix intervals are disjoint and tile the triangle iff
        // they cover this integer interval exactly, including adaptive leaves.
        if(begin!=cursor) {error="Cell coverage partition has a gap, overlap or ancestor prefix";return false;}
        cursor=end;
    }
    if(cursor!=(1u<<(2*kGratingCellCoverageDepth))) {error="Cell coverage partition omits part of the source cell";return false;}
    error.clear();return true;
}
struct GratingCoverageInterval {float lower=0,upper=0;};
using GratingCoverageBox=std::array<GratingCoverageInterval,3>;
struct GratingCellBlockerSeparation
{
    std::uint32_t triangle=UINT32_MAX;
    Point axis{};
    GratingCoverageInterval sweptProjection,triangleProjection;
};
struct GratingCellSweepEvidence
{
    std::uint32_t depth=0,token=0,object=0,material=0,instance=0,triangle=0,rule=0;
    bool entering=false;
    std::int32_t biasSign=0;
    Point direction{},normal{};
    GratingCoverageBox originBounds{},hitBounds{};
    GratingCoverageInterval distance,biasMagnitude,signedBias;
    std::array<GratingCoverageInterval,3> barycentricBounds{};
    float roundingExpansion=0;
    std::vector<GratingCellBlockerSeparation> blockers;
    std::uint64_t queryWork=0;
    GratingCellCoverageDisposition disposition=GratingCellCoverageDisposition::Unsupported;
};
struct GratingCellCoverageLeaf
{
    GratingCellDyadicTriangle domain;
    std::vector<GratingCellSweepEvidence> sweeps;
};
struct GratingCellCoverageRecord
{
    std::uint32_t incident=UINT32_MAX,entry=UINT32_MAX;
    std::vector<GratingCellCoverageLeaf> leaves;
    GratingCellCoverageDisposition disposition=GratingCellCoverageDisposition::Unsupported;
};
struct GratingCellCoveragePayload
{
    std::vector<GratingCellCoverageRecord> records;
    std::uint64_t queryWork=0;
    GratingCellCoverageDisposition disposition=GratingCellCoverageDisposition::Unsupported;
};
struct GratingCellCoverageBounds
{
    std::uint64_t maximumLeaves=0,maximumSweeps=0,maximumBlockers=0,queryWork=0,hostBytes=0,nativeBytes=0,bytes=0;
};
struct GratingCellCoverageCandidate
{
    std::shared_ptr<const NativeGratingCellTransportReadback> boundary;
    SubmissionIdentity submission;
    GratingCellCoverageBounds bounds;
    std::string identity;
};
struct NativeGratingCellCoverageEvidence
{
    std::uint32_t version=0,rule=0;
    std::weak_ptr<const GratingCellCoverageCandidate> candidate;
    std::weak_ptr<const NativeGratingCellTransportReadback> boundary;
    SubmissionIdentity boundarySubmission,submission;
    std::shared_ptr<const GratingCellCoveragePayload> payload;
    std::uint64_t bindingGeneration=0,nativeBytes=0;
    bool complete=false,overflow=false;
};
// Only a backend with a reviewed conservative swept-domain query may implement
// this interface. Its immutable payload must be decoded from actual completed
// native proof output. A host-constructed payload or covered bool is no grant.
// Existing GL/Metal boundary producers do not implement this interface.
class NativeGratingCellCoverageSource
{
public:
    virtual ~NativeGratingCellCoverageSource()=default;
    virtual const NativeFixtureSceneFrame& owner() const noexcept=0;
    virtual const NativeGratingCellCoverageEvidence& coverageEvidence() const noexcept=0;
protected:
    NativeGratingCellCoverageSource()=default;
};
inline bool calculateGratingCellCoverageBounds(const GratingCellTransportBinding& boundary,std::size_t maximumLeaves,
    GratingCellCoverageBounds& output,std::string& error)
{
    const auto refuse=[&](const char* text){error=text;return false;};
    if(!boundary.sourceCells||!boundary.sourceCells->incidentBinding||!maximumLeaves||maximumLeaves>kMaximumGratingCellCoverageLeaves)
        return refuse("Cell coverage parent or partition budget is invalid");
    const auto& parent=*boundary.sourceCells->incidentBinding;const auto& frame=parent.frame;
    if(!frame.triangles.size()||frame.triangles.size()>opaquevisibility::kMaximumClosestHitWork
        ||boundary.bounds.maximumRecords>kMaximumEntrySamples*8*kMaximumBranchVisits
        ||boundary.bounds.maximumRecords>parent.bounds.maximumIncidents||!boundary.bounds.maximumInterfaces||boundary.bounds.maximumInterfaces>kMaximumInterfaces)
        return refuse("Cell coverage retained triangle or interface extent is invalid");
    GratingCellCoverageBounds bounds;bounds.maximumLeaves=maximumLeaves;
    const auto leaves=boundary.bounds.maximumRecords*maximumLeaves;
    bounds.maximumSweeps=leaves*(boundary.bounds.maximumInterfaces+1);bounds.maximumBlockers=frame.triangles.size()-1;
    const auto query=gratingCellQueryAllowance(frame);
    if(!query||query<frame.triangles.size())return refuse("Cell coverage has no bounded all-surface sweep work allowance");
    bounds.queryWork=bounds.maximumSweeps*query;
    if(parent.bounds.combinedWork>kMaximumQueryWork||boundary.bounds.queryWork>kMaximumQueryWork-parent.bounds.combinedWork
        ||bounds.queryWork>kMaximumQueryWork-parent.bounds.combinedWork-boundary.bounds.queryWork)
        return refuse("Cell coverage exceeds the original whole-frame query cap");
    // Separate native proof reservation: 48-byte leaf header, 256-byte sweep
    // header and 32-byte separation row. No released transport stride changes.
    // Two native generations cover incumbent and candidate storage at publish.
    const auto rawBytes=leaves*48+bounds.maximumSweeps*(256+bounds.maximumBlockers*32);
    bounds.nativeBytes=2*rawBytes;
    bounds.hostBytes=sizeof(GratingCellCoverageCandidate)+sizeof(NativeGratingCellCoverageEvidence)+sizeof(GratingCellCoveragePayload)
        +boundary.bounds.maximumRecords*sizeof(GratingCellCoverageRecord)+leaves*sizeof(GratingCellCoverageLeaf)
        +bounds.maximumSweeps*(sizeof(GratingCellSweepEvidence)+bounds.maximumBlockers*sizeof(GratingCellBlockerSeparation))
        +rawBytes+maximumLeaves*sizeof(GratingCellDyadicTriangle)+2*boundary.identity.size()+8*parent.identity.size()+640;
    bounds.bytes=bounds.hostBytes+bounds.nativeBytes;
    const auto maximum=renderpassoutput::kMaximumTotalBytes;
    std::uint64_t owned=frame.bounds.transientBytes;
    for(auto bytes:{parent.bounds.bytes,boundary.sourceCells->bounds.bytes,boundary.bounds.bytes,bounds.bytes}) {
        if(owned>maximum||bytes>maximum-owned)return refuse("Cell coverage exceeds the original whole-frame byte cap");
        owned+=bytes;
    }
    output=bounds;error.clear();return true;
}
inline std::string gratingCellCoverageIdentity(const GratingCellCoverageCandidate& candidate)
{
    std::string identity;identity.reserve(candidate.boundary->binding->identity.size()+candidate.submission.frameIdentity.size()+192);
    word(identity,kGratingCellCoverageVersion);word(identity,candidate.boundary->binding->identity.size());identity+=candidate.boundary->binding->identity;
    word(identity,candidate.submission.frameIdentity.size());identity+=candidate.submission.frameIdentity;
    for(auto value:{candidate.submission.rendererGeneration,candidate.submission.submissionSerial,candidate.submission.rasterGeneration,
        candidate.submission.queryGeneration,candidate.submission.scratchGeneration,candidate.submission.completedFence,candidate.bounds.maximumLeaves})word(identity,value);
    return identity;
}
inline bool validateGratingCellCoverageCandidate(const GratingCellCoverageCandidate& candidate,
    const std::shared_ptr<const NativeFixtureSceneFrame>& exactOwner,const SubmissionIdentity& expected,std::string& error)
{
    const auto refuse=[&](const char* text){error=text;return false;};
    if(!candidate.boundary||!candidate.boundary->binding||!exactOwner||candidate.boundary->binding->nativeOwner!=exactOwner
        ||!(candidate.submission==expected)||!gratingCellSubmissionContinuity(candidate.boundary->submission,expected)
        ||expected.scratchGeneration==candidate.boundary->binding->pointSubmission.scratchGeneration)
        return refuse("Cell coverage boundary parent, native owner or newer fence changed");
    if(!validateGratingCellTransportReadback(*candidate.boundary->binding,*candidate.boundary,exactOwner,candidate.boundary->submission,error))return false;
    GratingCellCoverageBounds exact;
    if(!calculateGratingCellCoverageBounds(*candidate.boundary->binding,candidate.bounds.maximumLeaves,exact,error))return false;
    if(std::tie(candidate.bounds.maximumLeaves,candidate.bounds.maximumSweeps,candidate.bounds.maximumBlockers,candidate.bounds.queryWork,
            candidate.bounds.hostBytes,candidate.bounds.nativeBytes,candidate.bounds.bytes)
        !=std::tie(exact.maximumLeaves,exact.maximumSweeps,exact.maximumBlockers,exact.queryWork,exact.hostBytes,exact.nativeBytes,exact.bytes)
        ||candidate.identity!=gratingCellCoverageIdentity(candidate))return refuse("Cell coverage identity or reservation changed");
    error.clear();return true;
}
inline bool prepareGratingCellCoverageCandidate(const std::shared_ptr<const NativeGratingCellTransportReadback>& boundary,
    const std::shared_ptr<const NativeFixtureSceneFrame>& exactOwner,const SubmissionIdentity& boundaryExpected,
    const SubmissionIdentity& coverageExpected,std::size_t maximumLeaves,
    std::shared_ptr<const GratingCellCoverageCandidate>& output,std::string& error)
{
    if(!boundary||!boundary->binding||!(boundary->submission==boundaryExpected))
    {error="Cell coverage exact completed boundary parent is missing";return false;}
    auto candidate=std::make_shared<GratingCellCoverageCandidate>();candidate->boundary=boundary;candidate->submission=coverageExpected;
    if(!calculateGratingCellCoverageBounds(*boundary->binding,maximumLeaves,candidate->bounds,error))return false;
    candidate->identity=gratingCellCoverageIdentity(*candidate);
    if(!validateGratingCellCoverageCandidate(*candidate,exactOwner,coverageExpected,error))return false;
    output=std::move(candidate);error.clear();return true;
}
inline bool validGratingCoverageInterval(GratingCoverageInterval value)
{return std::isfinite(value.lower)&&std::isfinite(value.upper)&&value.lower<=value.upper;}
inline bool validateGratingCellCoverageRecordShape(const GratingCellCoverageCandidate& candidate,
    const GratingCellCoverageRecord& record,std::size_t locator,std::uint64_t& queryOutput,std::string& error)
{
    const auto refuse=[&](const char* text){error=text;return false;};
    if(!candidate.boundary||!candidate.boundary->binding||!candidate.boundary->binding->sourceCells
        ||!candidate.boundary->binding->sourceCells->incidentBinding||!candidate.boundary->binding->pointReadback
        ||locator>=candidate.boundary->binding->pointReadback->incidents.size()||locator>=candidate.boundary->records.size())return refuse("Cell coverage incident parent is missing");
    const auto& binding=*candidate.boundary->binding;const auto& frame=binding.sourceCells->incidentBinding->frame;
    const auto& point=binding.pointReadback->incidents[locator];const auto& boundary=candidate.boundary->records[locator];
    if(record.incident!=locator||record.entry!=boundary.entry||record.leaves.empty()||record.leaves.size()>candidate.bounds.maximumLeaves
        ||record.leaves.capacity()>candidate.bounds.maximumLeaves||boundary.paths[0].size()!=point.interfaces+1)
        return refuse("Cell coverage exact source locator or leaf extent changed");
    std::vector<GratingCellDyadicTriangle> domains;domains.reserve(record.leaves.size());
    for(const auto& leaf:record.leaves)domains.push_back(leaf.domain);
    if(!validateGratingCellDyadicPartition(domains,error))return false;
    std::uint64_t queries=0;bool certified=true;
    for(const auto& leaf:record.leaves) {
        if(leaf.sweeps.size()!=point.interfaces+1||leaf.sweeps.capacity()>binding.bounds.maximumInterfaces+1)
            return refuse("Cell coverage omits an interface or endpoint");
        for(std::size_t depth=0;depth<leaf.sweeps.size();++depth) {
            const auto& sweep=leaf.sweeps[depth];const auto& center=boundary.paths[0][depth];const auto& hit=center.hit;
            const std::int32_t sign=depth<point.interfaces?(center.signedBias>0?1:-1):0;
            if(sweep.depth!=depth||sweep.token!=center.token||sweep.rule!=kGratingCellCoverageRule||sweep.entering!=hit.entering
                ||std::tie(sweep.object,sweep.material,sweep.instance,sweep.triangle)!=std::tie(hit.objectId,hit.materialId,hit.instanceId,hit.triangleId)
                ||sweep.biasSign!=sign||sweep.direction!=center.direction||sweep.normal!=hit.outwardNormal
                ||sweep.queryWork>gratingCellQueryAllowance(frame)||sweep.queryWork>candidate.bounds.queryWork-queries)
                return refuse("Cell coverage interface owner, ancestry, entering, bias rule or work changed");
            queries+=sweep.queryWork;
            if(!validGratingCoverageInterval(sweep.distance)||!validGratingCoverageInterval(sweep.biasMagnitude)
                ||!validGratingCoverageInterval(sweep.signedBias)||!std::isfinite(sweep.roundingExpansion)
                ||sweep.blockers.capacity()>candidate.bounds.maximumBlockers)
                return refuse("Cell coverage evidence has nonfinite bounds or unreserved blocker storage");
            for(unsigned a=0;a<3;++a)if(!validGratingCoverageInterval(sweep.originBounds[a])
                ||!validGratingCoverageInterval(sweep.hitBounds[a])||!validGratingCoverageInterval(sweep.barycentricBounds[a]))
                return refuse("Cell coverage evidence has nonfinite domain bounds");
            if(sweep.disposition==GratingCellCoverageDisposition::Unsupported) {
                if(!sweep.blockers.empty())return refuse("Unsupported cell sweep carries an incomplete clear certificate");
                certified=false;continue;
            }
            if(sweep.disposition!=GratingCellCoverageDisposition::InteriorCertified
                ||boundary.disposition!=GratingCellTransportDisposition::BoundaryWitness||sweep.queryWork<frame.triangles.size()
                ||!std::isfinite(sweep.roundingExpansion)||sweep.roundingExpansion<=0||!validGratingCoverageInterval(sweep.distance)
                ||sweep.distance.lower<(depth?kRayMinimum:0)||sweep.distance.upper>frame.sourceDirectionTrace[3])
                return refuse("Cell coverage conservative sweep rule or finite-distance evidence is absent");
            double lowerSum=0,upperSum=0;
            for(unsigned a=0;a<3;++a) {
                if(!validGratingCoverageInterval(sweep.originBounds[a])||!validGratingCoverageInterval(sweep.hitBounds[a])
                    ||!validGratingCoverageInterval(sweep.barycentricBounds[a])||sweep.barycentricBounds[a].lower<=0
                    ||sweep.barycentricBounds[a].upper>=1)return refuse("Cell coverage full-domain bounds or strict triangle containment are absent");
                lowerSum+=sweep.barycentricBounds[a].lower;upperSum+=sweep.barycentricBounds[a].upper;
            }
            if(lowerSum>1||upperSum<1||!validGratingCoverageInterval(sweep.biasMagnitude)||!validGratingCoverageInterval(sweep.signedBias))
                return refuse("Cell coverage interval or barycentric enclosure is invalid");
            if(depth<point.interfaces) {
                float minimum=0,maximum=0;
                for(const auto interval:sweep.hitBounds) {
                    const float minAbs=interval.lower<=0&&interval.upper>=0?0:std::min(std::abs(interval.lower),std::abs(interval.upper));
                    minimum=std::max(minimum,minAbs);maximum=std::max({maximum,std::abs(interval.lower),std::abs(interval.upper)});
                }
                minimum=std::max(kRayMinimum,2e-6f*minimum);maximum=std::max(kRayMinimum,2e-6f*maximum);
                if(sweep.biasMagnitude.lower<kRayMinimum||sweep.biasMagnitude.lower>minimum||sweep.biasMagnitude.upper<maximum
                    ||sweep.signedBias.lower!=(sign>0?sweep.biasMagnitude.lower:-sweep.biasMagnitude.upper)
                    ||sweep.signedBias.upper!=(sign>0?sweep.biasMagnitude.upper:-sweep.biasMagnitude.lower))
                    return refuse("Cell coverage signed bias does not enclose the full hit domain");
            } else if(sweep.biasMagnitude.lower||sweep.biasMagnitude.upper||sweep.signedBias.lower||sweep.signedBias.upper)
                return refuse("Cell coverage endpoint carries an interface bias");
            const auto expected=std::find_if(frame.triangles.begin(),frame.triangles.end(),[&](const auto& triangle) {
                return std::tie(triangle.object,triangle.material,triangle.instance,triangle.triangle)==std::tie(sweep.object,sweep.material,sweep.instance,sweep.triangle);});
            if(expected==frame.triangles.end()||sweep.blockers.size()!=frame.triangles.size()-1
                ||sweep.blockers.capacity()>candidate.bounds.maximumBlockers)return refuse("Cell coverage omits an admitted all-surface blocker");
            std::size_t index=0;
            for(std::size_t triangle=0;triangle<frame.triangles.size();++triangle) {
                if(triangle==std::size_t(expected-frame.triangles.begin()))continue;
                const auto& blocker=sweep.blockers[index++];
                if(blocker.triangle!=triangle||!finite(blocker.axis)||std::abs(dot(blocker.axis,blocker.axis)-1)>1e-4f
                    ||!validGratingCoverageInterval(blocker.sweptProjection)||!validGratingCoverageInterval(blocker.triangleProjection)
                    ||std::max(double(blocker.triangleProjection.lower)-blocker.sweptProjection.upper,
                        double(blocker.sweptProjection.lower)-blocker.triangleProjection.upper)<=sweep.roundingExpansion)
                    return refuse("Cell coverage native conservative blocker separation is missing or ambiguous");
            }
        }
    }
    if((record.disposition==GratingCellCoverageDisposition::InteriorCertified)!=certified
        ||(record.disposition!=GratingCellCoverageDisposition::InteriorCertified&&record.disposition!=GratingCellCoverageDisposition::Unsupported))
        return refuse("Cell coverage disposition differs from complete leaf evidence");
    queryOutput=queries;error.clear();return true;
}
// Shape validation is not physical admission. Numeric enclosures and blocker
// intervals must be produced by the native conservative query, not host rays,
// sampled corners, center differentials or fabricated bounds. Its follow-on
// kernel must use outward-rounded bundle/plane bounds, exact signed bias and
// every retained all-surface triangle. Any uncertain overlap is Unsupported.
inline bool validateGratingCellCoverageShape(const GratingCellCoverageCandidate& candidate,
    const GratingCellCoveragePayload& payload,std::string& error)
{
    if(!candidate.boundary||payload.records.size()!=candidate.boundary->records.size()
        ||payload.records.capacity()>candidate.boundary->records.size())
    {error="Cell coverage dense record extent changed";return false;}
    std::uint64_t queries=0;bool certified=!payload.records.empty();
    for(std::size_t i=0;i<payload.records.size();++i) {
        std::uint64_t work=0;if(!validateGratingCellCoverageRecordShape(candidate,payload.records[i],i,work,error))return false;
        if(work>candidate.bounds.queryWork-queries) {error="Cell coverage actual work exceeds its admitted reservation";return false;}
        queries+=work;certified=certified&&payload.records[i].disposition==GratingCellCoverageDisposition::InteriorCertified;
    }
    if(payload.queryWork!=queries||(payload.disposition==GratingCellCoverageDisposition::InteriorCertified)!=certified
        ||(payload.disposition!=GratingCellCoverageDisposition::InteriorCertified&&payload.disposition!=GratingCellCoverageDisposition::Unsupported))
    {error="Cell coverage total work or complete partition disposition changed";return false;}
    error.clear();return true;
}
inline bool validateNativeGratingCellCoverage(const std::shared_ptr<const GratingCellCoverageCandidate>& candidate,
    const std::shared_ptr<const GratingCellCoveragePayload>& payload,const std::shared_ptr<const NativeFixtureSceneFrame>& exactOwner,
    const SubmissionIdentity& expected,std::string& error)
{
    if(!candidate||!payload||!validateGratingCellCoverageCandidate(*candidate,exactOwner,expected,error))return false;
    const auto* source=dynamic_cast<const NativeGratingCellCoverageSource*>(exactOwner.get());
    if(!source||&source->owner()!=exactOwner.get())
    {error="Native conservative cell coverage is unavailable; boundary samples remain Unsupported";return false;}
    const auto& evidence=source->coverageEvidence();
    if(evidence.version!=kGratingCellCoverageVersion||evidence.rule!=kGratingCellCoverageRule||!evidence.complete||evidence.overflow
        ||!evidence.bindingGeneration||evidence.candidate.lock()!=candidate||evidence.boundary.lock()!=candidate->boundary||evidence.payload!=payload
        ||!(evidence.boundarySubmission==candidate->boundary->submission)||!(evidence.submission==expected)
        ||(candidate->bounds.maximumSweeps&&!evidence.nativeBytes)||evidence.nativeBytes>candidate->bounds.nativeBytes)
    {error="Native cell coverage evidence, exact lineage, completed fence or allocation changed";return false;}
    if(!validateGratingCellCoverageShape(*candidate,*payload,error))return false;
    std::uint64_t minimumBytes=0;
    for(const auto& record:payload->records)for(const auto& leaf:record.leaves) {
        minimumBytes+=48;
        for(const auto& sweep:leaf.sweeps)minimumBytes+=256+sweep.blockers.size()*32;
    }
    if(evidence.nativeBytes<minimumBytes) {error="Native cell coverage allocation cannot contain its complete evidence";return false;}
    error.clear();return true;
}
// V2 is a structural prerequisite only. Closed rational regions can tile a
// source cell exactly; they do not prove native predicates on their seams.
inline constexpr std::uint32_t kGratingBoundaryCoverageVersionV2=2;
inline constexpr std::uint32_t kGratingBoundaryDenominatorV2=65536;
inline constexpr std::size_t kGratingBoundaryRegionsV2=64,kGratingBoundaryVerticesV2=8;
enum class GratingBoundaryPrerequisiteV2 : std::uint32_t { NativePredicateRequired=1 };
using GratingDomainPointV2=std::array<std::int32_t,2>;
using GratingDomainEdgeV2=std::array<GratingDomainPointV2,2>;
struct GratingBoundaryStepV2
{
    NativeGratingCellTraceStep trace;
    GratingNativeTriangleKeyV2 owner;
    float fromIor=1,toIor=1,reflection=0,transmission=0;
};
struct GratingBoundaryPathV2
{
    std::vector<GratingBoundaryStepV2> steps;
    double sourceFraction=0,reflectedPower=0,absorbedPower=0,unresolvedPower=0;
};
struct GratingBoundaryRegionV2
{
    // Strict convex corners, CCW from the lexicographically smallest vertex.
    // Edges/vertices are included. Incidence normalization splits T-junctions.
    std::vector<GratingDomainPointV2> vertices;
    std::uint64_t twiceArea=0;
    double incidentPower=0,unresolvedSourcePower=0;
    std::vector<GratingBoundaryPathV2> paths;
};
struct GratingBoundaryTransitionV2
{
    std::uint32_t path=0,depth=0;
    GratingMeshEdgeV2 left,right;
    std::vector<GratingMeshEdgeV2> adjacent;
};
struct GratingBoundarySeamV2
{
    GratingDomainEdgeV2 domain;
    std::array<std::uint32_t,2> regions{UINT32_MAX,UINT32_MAX};
    std::uint32_t sourceEdge=UINT32_MAX;
    std::vector<GratingMeshEdgeV2> sourceAdjacent;
    std::vector<GratingBoundaryTransitionV2> transitions;
    double power=0; // Exactly zero only for this nondegenerate 1D segment.
};
struct GratingBoundaryRecordV2
{
    std::uint32_t work=UINT32_MAX,entry=UINT32_MAX,band=UINT32_MAX,denominator=0;
    std::vector<GratingBoundaryRegionV2> regions;
    std::vector<GratingBoundarySeamV2> seams;
};
struct GratingBoundaryPayloadV2
{
    std::uint32_t version=kGratingBoundaryCoverageVersionV2;
    GratingBoundaryPrerequisiteV2 prerequisite=GratingBoundaryPrerequisiteV2::NativePredicateRequired;
    std::vector<GratingBoundaryRecordV2> records;
    std::uint64_t queryWork=0; // Sum of supplied actual regional hit work.
};
struct GratingBoundaryBoundsV2
{
    std::uint64_t regions=0,paths=0,seams=0,transitions=0,adjacent=0,interfaces=0;
    std::uint64_t queryWork=0,hostBytes=0,nativeBytes=0,bytes=0;
};
struct GratingBoundaryCandidateV2
{
    std::uint32_t version=kGratingBoundaryCoverageVersionV2;
    std::shared_ptr<const NativeGratingCellTransportReadback> boundary;
    SubmissionIdentity submission;
    GratingBoundaryBoundsV2 bounds;
    std::string identity;
};
// Call only after all coordinates satisfy gratingDomainPointValidV2.
inline std::int64_t gratingDomainTurnV2(GratingDomainPointV2 a,GratingDomainPointV2 b,GratingDomainPointV2 c)
{return (std::int64_t(b[0])-a[0])*(std::int64_t(c[1])-a[1])-(std::int64_t(b[1])-a[1])*(std::int64_t(c[0])-a[0]);}
inline bool gratingDomainPointValidV2(GratingDomainPointV2 p,std::uint32_t denominator)
{return denominator&&denominator<=kGratingBoundaryDenominatorV2&&p[0]>=0&&p[1]>=0
    &&std::uint64_t(p[0])+std::uint64_t(p[1])<=denominator;}
inline bool gratingDomainPolygonV2(const GratingBoundaryRegionV2& region,std::uint32_t denominator,
    std::uint64_t& area,std::string& error)
{
    if(region.vertices.size()<3||region.vertices.size()>kGratingBoundaryVerticesV2||region.vertices.capacity()>kGratingBoundaryVerticesV2)
    {error="Boundary v2 Unsupported polygon extent or curved transition";return false;}
    for(auto vertex:region.vertices)if(!gratingDomainPointValidV2(vertex,denominator)||vertex<region.vertices[0])
    {error="Boundary v2 exact polygon coordinate or canonical start is invalid";return false;}
    std::int64_t twiceArea=0;
    for(std::size_t i=0;i<region.vertices.size();++i) {
        const auto a=region.vertices[i],b=region.vertices[(i+1)%region.vertices.size()];
        if(a==b) {error="Boundary v2 polygon has a degenerate edge";return false;}
        for(std::size_t j=0;j<region.vertices.size();++j)if(j!=i&&j!=(i+1)%region.vertices.size())
            if(gratingDomainTurnV2(a,b,region.vertices[j])<=0)
            {error="Boundary v2 polygon is not simple strictly convex CCW";return false;}
        // Validated 16-bit coordinates and at most eight corners bound every
        // product and the complete signed sum well below INT64_MAX.
        twiceArea+=std::int64_t(a[0])*b[1]-std::int64_t(a[1])*b[0];
    }
    if(twiceArea<=0||std::uint64_t(twiceArea)>std::uint64_t(denominator)*denominator
        ||region.twiceArea!=std::uint64_t(twiceArea))
    {error="Boundary v2 positive region area is not exact";return false;}
    area=std::uint64_t(twiceArea);return true;
}
inline bool gratingDomainDisjointV2(const GratingBoundaryRegionV2& first,const GratingBoundaryRegionV2& second)
{
    const auto separated=[](const auto& a,const auto& b) {
        for(std::size_t i=0;i<a.vertices.size();++i) {
            bool outside=true;
            for(auto point:b.vertices)outside=outside&&gratingDomainTurnV2(a.vertices[i],a.vertices[(i+1)%a.vertices.size()],point)<=0;
            if(outside)return true;
        }
        return false; };
    return separated(first,second)||separated(second,first);
}
using GratingDomainIncidenceV2=std::map<GratingDomainEdgeV2,std::vector<std::uint32_t>>;
inline bool gratingBoundaryPartitionV2(const GratingBoundaryRecordV2& record,std::size_t maximumRegions,
    GratingDomainIncidenceV2& output,std::string& error)
{
    const auto refuse=[&](const char* text){error=text;return false;};
    if(record.regions.empty()||record.regions.size()>maximumRegions||record.regions.capacity()>maximumRegions
        ||maximumRegions>kGratingBoundaryRegionsV2||!record.denominator||record.denominator>kGratingBoundaryDenominatorV2)
        return refuse("Boundary v2 Unsupported exact region or arithmetic budget");
    const std::uint64_t whole=std::uint64_t(record.denominator)*record.denominator;std::uint64_t total=0;
    std::vector<GratingDomainPointV2> vertices;vertices.reserve(record.regions.size()*kGratingBoundaryVerticesV2);
    for(std::size_t i=0;i<record.regions.size();++i) {
        std::uint64_t area=0;if(!gratingDomainPolygonV2(record.regions[i],record.denominator,area,error))return false;
        if(area>whole-total)return refuse("Boundary v2 region area overlaps or exceeds the source cell");
        total+=area;vertices.insert(vertices.end(),record.regions[i].vertices.begin(),record.regions[i].vertices.end());
        for(std::size_t j=0;j<i;++j)if(!gratingDomainDisjointV2(record.regions[i],record.regions[j]))
            return refuse("Boundary v2 positive region interiors overlap");
    }
    if(total!=whole)return refuse("Boundary v2 exact source area has a gap");
    std::sort(vertices.begin(),vertices.end());vertices.erase(std::unique(vertices.begin(),vertices.end()),vertices.end());
    GratingDomainIncidenceV2 incidence;
    for(std::size_t r=0;r<record.regions.size();++r)for(std::size_t i=0;i<record.regions[r].vertices.size();++i) {
        auto a=record.regions[r].vertices[i],b=record.regions[r].vertices[(i+1)%record.regions[r].vertices.size()];
        if(b<a)std::swap(a,b);
        std::vector<GratingDomainPointV2> edge;edge.reserve(vertices.size());
        for(auto p:vertices)if(a<=p&&p<=b&&gratingDomainTurnV2(a,b,p)==0)edge.push_back(p);
        for(std::size_t j=1;j<edge.size();++j) {
            if(incidence.size()>=3*maximumRegions
                &&incidence.find({edge[j-1],edge[j]})==incidence.end())return refuse("Boundary v2 normalized edge budget exceeded");
            auto& owners=incidence[{edge[j-1],edge[j]}];
            if(owners.size()>=2)return refuse("Boundary v2 normalized edge has too many positive-area sides");
            owners.push_back(std::uint32_t(r));
        }
    }
    // A finite union of these closed convex regions covers the whole closed
    // source triangle. Seam ray ownership remains a separate native question.
    output=std::move(incidence);error.clear();return true;
}
inline bool calculateGratingBoundaryBoundsV2(const GratingCellTransportBinding& boundary,std::size_t regions,std::size_t paths,
    GratingBoundaryBoundsV2& output,std::string& error)
{
    const auto refuse=[&](){error="Boundary v2 Unsupported original work or whole-frame byte cap";return false;};
    if(!boundary.sourceCells||!boundary.sourceCells->incidentBinding||!boundary.sourceCells->incidentBinding->prepared.branchPolicy
        ||!regions||regions>kGratingBoundaryRegionsV2||!paths
        ||paths>boundary.sourceCells->incidentBinding->prepared.branchPolicy->description().maximumVisits)return refuse();
    const auto& parent=*boundary.sourceCells->incidentBinding;const auto& frame=parent.frame;
    const auto maximum=renderpassoutput::kMaximumTotalBytes;
    GratingBoundaryBoundsV2 bounds;bounds.regions=regions;bounds.paths=paths;bounds.interfaces=boundary.bounds.maximumInterfaces;
    // In a convex tiling, only the three source corners can have degree
    // two. Euler plus 2E>=3V-3 and E=V+R-1 gives E<=3R.
    bounds.seams=3*regions;
    bounds.transitions=paths*(bounds.interfaces+1);bounds.adjacent=3*frame.triangles.size();
    if(!bounds.interfaces||bounds.interfaces>kMaximumInterfaces||frame.triangles.empty()||bounds.adjacent>3*opaquevisibility::kMaximumClosestHitWork
        ||frame.work.size()>kMaximumEntrySamples*8||paths>kMaximumBranchVisits)return refuse();
    const auto vertices=regions*kGratingBoundaryVerticesV2;
    // Work includes partition SAT/normalization, retained topology lookup and
    // every adjacency comparison. No existing point/replay counter is raised.
    const auto product=[](std::uint64_t a,std::uint64_t b,std::uint64_t& result) {
        if(b&&a>kMaximumQueryWork/b)return false;
        result=a*b;return true; };
    std::uint64_t levels=0;for(auto n=bounds.adjacent-1;n;n>>=1)++levels;
    // Heap construction/sort use at most (3+2*ceil(log2(N)))*N
    // comparisons; five further scans cover owner/edge lookup and equality.
    std::uint64_t partition=0,topology=0,pathWork=0,rowWork=0;
    if(!product(8*vertices,vertices,partition)||!product(bounds.seams,1+bounds.transitions,topology)
        ||!product(topology,bounds.adjacent,topology)||!product(topology,8+2*levels,topology)
        ||!product(regions*paths,bounds.interfaces+1,pathWork)
        ||!gratingCellQueryAllowance(frame)
        ||!product(pathWork,gratingCellQueryAllowance(frame)+8*frame.triangles.size(),pathWork)||partition>kMaximumQueryWork-topology
        ||pathWork>kMaximumQueryWork-partition-topology)return refuse();
    rowWork=partition+topology+pathWork;
    if(!product(rowWork,frame.work.size(),bounds.queryWork))return refuse();
    if(parent.bounds.combinedWork>kMaximumQueryWork||boundary.bounds.queryWork>kMaximumQueryWork-parent.bounds.combinedWork
        ||bounds.queryWork>kMaximumQueryWork-parent.bounds.combinedWork-boundary.bounds.queryWork)return refuse();
    const auto regionBytes=sizeof(GratingBoundaryRegionV2)+kGratingBoundaryVerticesV2*sizeof(GratingDomainPointV2)
        +paths*(sizeof(GratingBoundaryPathV2)+(bounds.interfaces+1)*sizeof(GratingBoundaryStepV2));
    const auto seamBytes=sizeof(GratingBoundarySeamV2)+bounds.adjacent*sizeof(GratingMeshEdgeV2)
        +bounds.transitions*(sizeof(GratingBoundaryTransitionV2)+bounds.adjacent*sizeof(GratingMeshEdgeV2));
    // All factors are capped before multiplication. Reserve host values plus
    // validation maps/copies; a future native format must fit this allowance.
    if(regionBytes>maximum/regions||seamBytes>maximum/bounds.seams)return refuse();
    const auto row=sizeof(GratingBoundaryRecordV2)+regions*regionBytes+bounds.seams*(seamBytes+128);
    if(row>maximum||frame.work.size()>maximum/(2*row))return refuse();
    if(boundary.identity.size()>maximum/8||parent.identity.size()>maximum/8)return refuse();
    bounds.hostBytes=2*frame.work.size()*row+sizeof(GratingBoundaryCandidateV2)+sizeof(GratingBoundaryPayloadV2)
        +2*boundary.identity.size()+2*parent.identity.size()+1024;
    if(bounds.hostBytes>maximum/3)return refuse();
    bounds.nativeBytes=2*bounds.hostBytes;bounds.bytes=bounds.hostBytes+bounds.nativeBytes;
    std::uint64_t owned=frame.bounds.transientBytes;
    for(auto bytes:{parent.bounds.bytes,boundary.sourceCells->bounds.bytes,boundary.bounds.bytes,bounds.bytes}) {
        if(owned>maximum||bytes>maximum-owned)return refuse();
        owned+=bytes;
    }
    output=bounds;error.clear();return true;
}
inline std::string gratingBoundaryIdentityV2(const GratingBoundaryCandidateV2& candidate)
{
    std::string identity;word(identity,kGratingBoundaryCoverageVersionV2);
    word(identity,candidate.boundary->binding->identity.size());identity+=candidate.boundary->binding->identity;
    for(auto value:{candidate.submission.rendererGeneration,candidate.submission.submissionSerial,candidate.submission.rasterGeneration,
        candidate.submission.queryGeneration,candidate.submission.scratchGeneration,candidate.submission.completedFence,
        candidate.bounds.regions,candidate.bounds.paths})word(identity,value);
    return identity;
}
// Caller copies can share exact native source/point parents while retaining
// independent strings and boundary vectors. Admit that backing by capacity.
inline bool gratingBoundaryParentStorageV2(const NativeGratingCellTransportReadback& boundary,std::string& error)
{
    if(!boundary.binding) {error="Boundary v2 retained binding is missing";return false;}
    const auto& binding=*boundary.binding;const auto limit=binding.bounds.hostBytes;
    if(limit>renderpassoutput::kMaximumTotalBytes) {error="Boundary v2 retained host reservation exceeds the whole-frame cap";return false;}
    std::uint64_t bytes=sizeof(binding)+sizeof(boundary);
    const auto charge=[&](std::size_t count,std::size_t stride) {
        if(!limit||bytes>limit||count>limit/stride||std::uint64_t(count)*stride>limit-bytes)return false;
        bytes+=std::uint64_t(count)*stride;return true; };
    if(!charge(binding.identity.capacity(),1)||!charge(binding.pointSubmission.frameIdentity.capacity(),1)
        ||!charge(binding.submission.frameIdentity.capacity(),1)||!charge(boundary.submission.frameIdentity.capacity(),1)
        ||!charge(boundary.records.capacity(),sizeof(NativeGratingCellTransportRecord)))
    {error="Boundary v2 retained parent metadata exceeds its existing host reservation";return false;}
    for(const auto& record:boundary.records)for(const auto& path:record.paths)
        if(!charge(path.capacity(),sizeof(NativeGratingCellTraceStep)))
        {error="Boundary v2 retained parent paths exceed their existing host reservation";return false;}
    // Subsequent native validation binds the shared source/point parents to
    // the frame's exact immutable upload, rather than caller value copies.
    return true;
}
inline bool validateGratingBoundaryCandidateV2(const GratingBoundaryCandidateV2& candidate,
    const std::shared_ptr<const NativeFixtureSceneFrame>& owner,const SubmissionIdentity& expected,std::string& error)
{
    if(candidate.version!=kGratingBoundaryCoverageVersionV2||!candidate.boundary||!candidate.boundary->binding
        ||candidate.boundary->binding->nativeOwner!=owner||!(candidate.submission==expected)
        ||!gratingCellSubmissionContinuity(candidate.boundary->submission,expected)
        ||expected.scratchGeneration==candidate.boundary->binding->pointSubmission.scratchGeneration
        ||!gratingBoundaryParentStorageV2(*candidate.boundary,error)
        ||!validateGratingCellTransportReadback(*candidate.boundary->binding,*candidate.boundary,owner,candidate.boundary->submission,error))
    {error="Boundary v2 exact source/point/native owner or later completed fence changed";return false;}
    GratingBoundaryBoundsV2 exact;
    if(!calculateGratingBoundaryBoundsV2(*candidate.boundary->binding,candidate.bounds.regions,candidate.bounds.paths,exact,error))return false;
    if(candidate.identity!=gratingBoundaryIdentityV2(candidate)
        ||std::tie(candidate.bounds.regions,candidate.bounds.paths,candidate.bounds.seams,candidate.bounds.transitions,candidate.bounds.adjacent,
            candidate.bounds.interfaces,candidate.bounds.queryWork,candidate.bounds.hostBytes,candidate.bounds.nativeBytes,candidate.bounds.bytes)
            !=std::tie(exact.regions,exact.paths,exact.seams,exact.transitions,exact.adjacent,exact.interfaces,exact.queryWork,exact.hostBytes,exact.nativeBytes,exact.bytes)
        ||candidate.identity.capacity()>candidate.bounds.hostBytes
        ||candidate.submission.frameIdentity.capacity()>candidate.bounds.hostBytes-candidate.identity.capacity())
    {error="Boundary v2 immutable identity or bounded reservation changed";return false;}
    error.clear();return true;
}
inline bool prepareGratingBoundaryCandidateV2(const std::shared_ptr<const NativeGratingCellTransportReadback>& boundary,
    const std::shared_ptr<const NativeFixtureSceneFrame>& owner,const SubmissionIdentity& expected,std::size_t regions,std::size_t paths,
    std::shared_ptr<const GratingBoundaryCandidateV2>& output,std::string& error)
{
    if(!boundary||!boundary->binding) {error="Boundary v2 retained completed parent is missing";return false;}
    GratingBoundaryBoundsV2 bounds;
    if(!validateGratingCellTransportReadback(*boundary->binding,*boundary,owner,boundary->submission,error)
        ||!gratingBoundaryParentStorageV2(*boundary,error)||!calculateGratingBoundaryBoundsV2(*boundary->binding,regions,paths,bounds,error))return false;
    auto candidate=std::make_shared<GratingBoundaryCandidateV2>();candidate->boundary=boundary;candidate->submission=expected;candidate->bounds=bounds;
    candidate->identity=gratingBoundaryIdentityV2(*candidate);
    if(!validateGratingBoundaryCandidateV2(*candidate,owner,expected,error))return false;
    output=std::move(candidate);return true;
}

inline bool gratingBoundaryAdjacentV2(const GratingIncidentBinding& parent,GratingMeshEdgeV2 edge,
    const std::vector<GratingMeshEdgeV2>& adjacent,std::string& error)
{
    if(adjacent.capacity()>3*parent.frame.triangles.size()) {error="Boundary v2 adjacent topology storage is unbounded";return false;}
    std::array<Point,2> endpoints;if(!gratingMeshEdgePointsV2(parent,edge,endpoints,error))return false;
    std::vector<GratingMeshEdgeV2> exact;exact.reserve(3*parent.frame.triangles.size());
    for(const auto& triangle:parent.frame.triangles)for(unsigned e=0;e<3;++e) {
        if(triangle.instance>=parent.frame.instances.size())
        {error="Boundary v2 retained adjacency instance is invalid";return false;}
        GratingMeshEdgeV2 candidate{{std::uint32_t(parent.frame.instances[triangle.instance].object),triangle.instance,triangle.triangle},e};
        if(sameGratingMeshEdgeV2(endpoints,{triangle.points[e],triangle.points[(e+1)%3]}))exact.push_back(candidate);
    }
    std::make_heap(exact.begin(),exact.end());std::sort_heap(exact.begin(),exact.end());
    if(exact!=adjacent) {error="Boundary v2 exact retained edge adjacency is omitted or changed";return false;}
    return true;
}
inline std::uint32_t gratingBoundarySourceEdgeV2(const GratingEntryCell& cell,GratingDomainEdgeV2 edge,std::uint32_t denominator)
{
    std::array<std::array<std::int64_t,3>,2> original{};
    for(unsigned end=0;end<2;++end) {
        const std::array<std::int64_t,3> weights{std::int64_t(denominator)-edge[end][0]-edge[end][1],edge[end][0],edge[end][1]};
        for(unsigned v=0;v<3;++v) {
            // Retained source subdivisions 1/2/4 have exact quarter coordinates.
            const auto u=std::int64_t(cell.barycentrics[v][0]*4),w=std::int64_t(cell.barycentrics[v][1]*4);
            original[end][0]+=weights[v]*(4-u-w);original[end][1]+=weights[v]*u;original[end][2]+=weights[v]*w;
        }
    }
    for(unsigned e=0;e<3;++e)if(original[0][(e+2)%3]==0&&original[1][(e+2)%3]==0)return e;
    return UINT32_MAX;
}
inline bool gratingBoundaryPowerNearV2(double actual,double expected)
{return std::isfinite(actual)&&actual>=0&&std::abs(actual-expected)<=64*std::numeric_limits<double>::epsilon()*std::max(std::abs(expected),1e-30);}
inline bool validateGratingBoundaryRegionPathsV2(const GratingBoundaryCandidateV2& candidate,const GratingBoundaryRecordV2& record,
    const GratingBoundaryRegionV2& region,std::string& error)
{
    const auto refuse=[&](const char* text){error=text;return false;};
    const auto& parent=*candidate.boundary->binding->sourceCells->incidentBinding;const auto& frame=parent.frame;
    if(region.paths.capacity()>candidate.bounds.paths||region.paths.size()>candidate.bounds.paths
        ||!gratingBoundaryPowerNearV2(region.incidentPower,double(frame.work[record.work].incidentPower)*region.twiceArea
            /(std::uint64_t(record.denominator)*record.denominator))||!std::isfinite(region.unresolvedSourcePower)||region.unresolvedSourcePower<0)
        return refuse("Boundary v2 region area-weighted source power or path storage changed");
    const auto& cell=candidate.boundary->binding->sourceCells->cells[record.entry];
    double u=0,v=0;for(auto corner:region.vertices) {u+=corner[0];v+=corner[1];}
    u/=double(record.denominator)*region.vertices.size();v/=double(record.denominator)*region.vertices.size();
    Point sample{};for(unsigned a=0;a<3;++a)sample[a]=float((1-u-v)*cell.vertices[0][a]+u*cell.vertices[1][a]+v*cell.vertices[2][a]);
    const Point source{frame.sourceDirectionTrace[0],frame.sourceDirectionTrace[1],frame.sourceDirectionTrace[2]};
    double accounted=region.unresolvedSourcePower;std::size_t visits=0;
    for(std::size_t p=0;p<region.paths.size();++p) {
        const auto& path=region.paths[p];
        if(path.steps.size()<2||path.steps.size()>candidate.bounds.interfaces+1||path.steps.capacity()>candidate.bounds.interfaces+1)
            return refuse("Boundary v2 path omits its source interface or endpoint or exceeds the interface cap");
        if(path.steps.size()>parent.prepared.branchPolicy->description().maximumVisits-visits)
            return refuse("Boundary v2 regional paths exceed the unchanged branch visit cap");
        visits+=path.steps.size();
        Point origin=advance(sample,source,2*bias(sample)),direction=advance({},source,-1);bool inside=false;double fraction=1;
        // Existing float point-geometry tolerance checks only this supplied
        // representative ray. It never changes exact domain/seam membership.
        for(std::size_t d=0;d<path.steps.size();++d) {
            const auto& value=path.steps[d];const auto& trace=value.trace;const auto& hit=trace.hit;
            GratingRetainedTopologyV2 topology;
            if(!gratingRetainedTopologyV2(parent,value.owner,topology,error))return false;
            const bool endpoint=d+1==path.steps.size();
            if(trace.depth!=d||hit.status!=1||!hit.work||hit.work>gratingCellQueryAllowance(frame)
                ||!finite(hit.position)||!std::isfinite(hit.distance)||hit.distance<kRayMinimum||hit.distance>frame.sourceDirectionTrace[3]
                ||!sameGratingPoint(trace.origin,origin,8*bias(origin))||!sameGratingPoint(trace.direction,direction)
                ||std::abs(dot(trace.direction,trace.direction)-1)>1e-4f||hit.outwardNormal!=topology.triangle->normal
                ||std::tie(hit.objectId,hit.materialId,hit.instanceId,hit.triangleId)
                    !=std::tie(topology.object,topology.material,value.owner.instance,value.owner.triangle)
                ||hit.entering!=(dot(trace.direction,hit.outwardNormal)<0)||hitTriangle(frame,hit)!=topology.triangle
                ||!sameGratingPoint(advance(trace.origin,trace.direction,hit.distance),hit.position,8*bias(hit.position)))
                return refuse("Boundary v2 regional actual owner, normal or native path continuity changed");
            if(endpoint) {
                if(topology.triangle->glass||inside||trace.token||trace.signedBias||value.fromIor!=1||value.toIor!=1
                    ||value.reflection||value.transmission)return refuse("Boundary v2 endpoint material, medium or bias changed");
            } else {
                const auto oriented=inside?advance({},hit.outwardNormal,-1):hit.outwardNormal;
                Interface optics;const bool reflected=trace.token==1;
                if(!topology.triangle->glass||inside==hit.entering
                    ||(d==0&&(hit.objectId!=cell.object||hit.materialId!=cell.material
                        ||hit.instanceId!=cell.instance||hit.triangleId!=cell.triangle))
                    ||(trace.token!=1&&trace.token!=65)
                    ||value.fromIor!=(inside?frame.ior[record.band]:1)||value.toIor!=(inside?1:frame.ior[record.band])
                    ||!interface(trace.direction,oriented,value.fromIor,value.toIor,optics)
                    ||value.reflection!=optics.reflection||value.transmission!=(optics.tir?0:1-optics.reflection)
                    ||(reflected?value.reflection<=0:value.transmission<=0))
                    return refuse("Boundary v2 actual face IOR, Fresnel split or ancestry changed");
                const float signedBias=(inside?-1.0f:1.0f)*(reflected?1.0f:-1.0f)*bias(hit.position);
                if(trace.signedBias!=signedBias)return refuse("Boundary v2 regional signed bias changed");
                fraction*=reflected?double(value.reflection):double(value.transmission);
                origin=advance(hit.position,hit.outwardNormal,signedBias);direction=reflected?optics.reflected:optics.transmitted;
                if(!reflected)inside=!inside;
            }
        }
        for(std::size_t previous=0;previous<p;++previous) {
            const auto& other=region.paths[previous];std::size_t common=0;
            while(common<std::min(path.steps.size(),other.steps.size())) {
                const auto& a=path.steps[common];const auto& b=other.steps[common];
                if(!(a.owner==b.owner)||a.trace.origin!=b.trace.origin||a.trace.direction!=b.trace.direction
                    ||a.trace.hit.position!=b.trace.hit.position||a.trace.hit.outwardNormal!=b.trace.hit.outwardNormal
                    ||a.fromIor!=b.fromIor||a.toIor!=b.toIor||a.reflection!=b.reflection||a.transmission!=b.transmission)
                    return refuse("Boundary v2 common ancestry has inconsistent regional owners");
                if(a.trace.token!=b.trace.token)break;
                ++common;
            }
            if(common==std::min(path.steps.size(),other.steps.size()))return refuse("Boundary v2 path ancestry duplicates or overlaps a completed prefix");
        }
        if(!gratingBoundaryPowerNearV2(path.sourceFraction,fraction)||!std::isfinite(path.reflectedPower)||path.reflectedPower<0
            ||!std::isfinite(path.absorbedPower)||path.absorbedPower<0||!std::isfinite(path.unresolvedPower)||path.unresolvedPower<0
            ||!gratingBoundaryPowerNearV2(path.reflectedPower+path.absorbedPower+path.unresolvedPower,region.incidentPower*fraction))
            return refuse("Boundary v2 regional branch or endpoint power is not conserved");
        accounted+=region.incidentPower*path.sourceFraction;
    }
    if(!gratingBoundaryPowerNearV2(accounted,region.incidentPower))return refuse("Boundary v2 unresolved branches drop or duplicate regional source power");
    return true;
}
inline bool validateGratingBoundaryShapeV2(const GratingBoundaryCandidateV2& candidate,const GratingBoundaryPayloadV2& payload,std::string& error)
{
    const auto refuse=[&](const char* text){error=text;return false;};
    if(!candidate.boundary||!candidate.boundary->binding||!candidate.boundary->binding->sourceCells
        ||!candidate.boundary->binding->sourceCells->incidentBinding||payload.version!=kGratingBoundaryCoverageVersionV2
        ||payload.prerequisite!=GratingBoundaryPrerequisiteV2::NativePredicateRequired)
        return refuse("Boundary v2 exact parent or native predicate prerequisite changed");
    if(!validateGratingBoundaryCandidateV2(candidate,candidate.boundary->binding->nativeOwner,candidate.submission,error))return false;
    const auto& parent=*candidate.boundary->binding->sourceCells->incidentBinding;const auto& frame=parent.frame;
    GratingBoundaryBoundsV2 exact;
    if(!calculateGratingBoundaryBoundsV2(*candidate.boundary->binding,candidate.bounds.regions,candidate.bounds.paths,exact,error)
        ||candidate.bounds.bytes!=exact.bytes||candidate.bounds.queryWork!=exact.queryWork
        ||payload.records.size()!=frame.work.size()||payload.records.capacity()>frame.work.size())return refuse("Boundary v2 source work coverage or capacity changed");
    std::uint64_t actualQueries=0;
    for(std::size_t work=0;work<payload.records.size();++work) {
        const auto& record=payload.records[work];const auto& input=frame.work[work];
        if(record.work!=work||record.entry!=input.entry||record.band!=input.band||record.entry>=candidate.boundary->binding->sourceCells->cells.size()
            ||record.band>=8||record.seams.capacity()>candidate.bounds.seams)return refuse("Boundary v2 original cell/band locator or seam budget changed");
        GratingDomainIncidenceV2 incidence;if(!gratingBoundaryPartitionV2(record,candidate.bounds.regions,incidence,error))return false;
        if(record.seams.size()!=incidence.size())return refuse("Boundary v2 normalized edge incidence omits or duplicates a seam");
        double power=0;for(const auto& region:record.regions) {
            if(!validateGratingBoundaryRegionPathsV2(candidate,record,region,error))return false;
            power+=region.incidentPower;
            for(const auto& path:region.paths)for(const auto& step:path.steps) {
                if(step.trace.hit.work>candidate.bounds.queryWork-actualQueries)return refuse("Boundary v2 actual regional query work exceeds its cap");
                actualQueries+=step.trace.hit.work;
            }
        }
        if(!gratingBoundaryPowerNearV2(power,input.incidentPower))return refuse("Boundary v2 exact area partition loses cell/band power");
        std::size_t index=0;
        const auto& cell=candidate.boundary->binding->sourceCells->cells[record.entry];
        if(cell.instance>=frame.instances.size())return refuse("Boundary v2 source topology instance is missing");
        const GratingNativeTriangleKeyV2 sourceKey{std::uint32_t(frame.instances[cell.instance].object),cell.instance,cell.triangle};
        for(const auto& edge:incidence) {
            const auto& seam=record.seams[index++];const auto& sides=edge.second;
            const std::array<std::uint32_t,2> expectedSides{sides[0],sides.size()==2?sides[1]:UINT32_MAX};
            const auto sourceEdge=gratingBoundarySourceEdgeV2(cell,edge.first,record.denominator);
            if(seam.domain!=edge.first||seam.domain[0]==seam.domain[1]||seam.regions!=expectedSides||seam.power!=0
                ||seam.sourceEdge!=sourceEdge||seam.transitions.capacity()>candidate.bounds.transitions)
                return refuse("Boundary v2 seam is not an exact zero-power 1D incidence");
            if(sourceEdge!=UINT32_MAX) {
                if(!gratingBoundaryAdjacentV2(parent,{sourceKey,sourceEdge},seam.sourceAdjacent,error))return false;
            } else if(!seam.sourceAdjacent.empty()||seam.sourceAdjacent.capacity()>candidate.bounds.adjacent)
                return refuse("Boundary v2 internal source edge invents mesh adjacency");
            std::size_t transition=0;
            if(sides.size()==2) {
                const auto& left=record.regions[sides[0]];const auto& right=record.regions[sides[1]];
                if(left.paths.size()!=right.paths.size())return refuse("Boundary v2 Unsupported differing branch topology across a transition");
                for(std::size_t path=0;path<left.paths.size();++path) {
                    if(left.paths[path].steps.size()!=right.paths[path].steps.size())return refuse("Boundary v2 Unsupported differing regional interface depths");
                    for(std::size_t depth=0;depth<left.paths[path].steps.size();++depth) {
                        const auto& a=left.paths[path].steps[depth];const auto& b=right.paths[path].steps[depth];
                        if(a.trace.token!=b.trace.token)return refuse("Boundary v2 Unsupported branch-policy transition");
                        if(a.owner==b.owner)continue;
                        if(transition>=seam.transitions.size())return refuse("Boundary v2 actual owner transition is omitted");
                        const auto& value=seam.transitions[transition++];std::array<Point,2> first,second;
                        if(value.path!=path||value.depth!=depth||!(value.left.owner==a.owner)||!(value.right.owner==b.owner)
                            ||!gratingMeshEdgePointsV2(parent,value.left,first,error)||!gratingMeshEdgePointsV2(parent,value.right,second,error)
                            ||!sameGratingMeshEdgeV2(first,second)||!gratingBoundaryAdjacentV2(parent,value.left,value.adjacent,error))
                            return refuse("Boundary v2 retained one-sided owner/material edge topology changed");
                    }
                }
            }
            if(transition!=seam.transitions.size())return refuse("Boundary v2 seam invents a downstream owner transition");
        }
    }
    if(payload.queryWork!=actualQueries)return refuse("Boundary v2 actual regional query ledger changed");
    error.clear();return true;
}
// No enum, caller-built topology, interface return or boolean authenticates
// full-segment float edge inclusion, equal native distance or tie semantics.
// A separate reviewed predicate contract/producer is an explicit prerequisite.
inline bool validateNativeGratingBoundaryCoverageV2(const std::shared_ptr<const GratingBoundaryCandidateV2>& candidate,
    const std::shared_ptr<const GratingBoundaryPayloadV2>& payload,const std::shared_ptr<const NativeFixtureSceneFrame>& owner,
    const SubmissionIdentity& expected,std::string& error)
{
    if(!candidate||!payload||!validateGratingBoundaryCandidateV2(*candidate,owner,expected,error)
        ||!validateGratingBoundaryShapeV2(*candidate,*payload,error))return false;
    error="Boundary v2 native full-segment float predicate and seam winner certificate is unavailable; Unsupported";
    return false;
}

}
