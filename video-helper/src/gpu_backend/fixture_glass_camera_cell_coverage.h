#pragma once

#include "fixture_glass_grating_incident.h"

namespace arbitgpu::glasscaustics
{
// Geometry preparation and native finite-pixel proof are distinct. Neither
// grants diffraction integration, radiance, authored base or physical Color.
struct CameraCellCoverageInput
{
    std::uint32_t x=0,y=0,object=0,material=0,instance=0;
    std::uint64_t firstAssociation=0,associationCount=0;
};
struct CameraCellCoverageAssociation
{
    std::uint32_t parentRow=0,pixel=0;
    NativeDirectGratingTerminalReceiverInput order; // Exact emitted occurrence.
};
struct CameraCellCoverageCandidate
{
    enum class Measure : std::uint32_t { Unavailable=0,Prepared=1 };
    std::uint32_t pixel=0,triangle=0; // Index in the exact frozen all-triangle table.
    Measure measure=Measure::Unavailable;
    CameraCellGeometry geometry; // One-triangle candidate only, never proof.
};
struct CameraCellCoveragePreparation
{
    std::vector<CameraCellCoverageInput> pixels;
    std::vector<CameraCellCoverageAssociation> associations;
    std::vector<CameraCellCoverageCandidate> candidates;
};
struct NativeCameraCellCoverageBudget
{
    std::uint64_t retainedBytes=0,combinedWork=0;
    // Sum actual native resources and any extra decoded inspector copies beyond
    // the shared reservation. Reconstruct from registered resources/attempts,
    // including prior live results and full admitted work on every failure.
    std::uint64_t pixelBytes=0,candidateBytes=0,regionBytes=0,vertexBytes=0,fixedBytes=0;
};
struct CameraCellCoverageBounds
{
    std::uint64_t bytes=0,combinedWork=0,operationWork=0,nativeBytes=0;
    std::uint64_t maximumPixels=0,maximumAssociations=0,triangles=0,maximumCandidates=0;
    std::uint64_t depthPairs=0,linesPerPixel=0,regionsPerPixel=0,verticesPerPixel=0;
    std::uint64_t maximumRegions=0,maximumVertices=0,preparationWork=0,partitionWork=0,triangleWork=0;
};
struct CameraCellCoverageRegion
{
    std::uint32_t pixel=0,triangle=0;
    std::uint64_t firstVertex=0,vertexCount=0;
    double projectedArea=0; // Fraction of this pixel in perspective coordinates.
};
struct CameraCellCoverageProof
{
    std::uint64_t generation=0,work=0;
    std::vector<CameraCellCoverageRegion> regions;
    // Top-first normalized pixel coordinates in [0,1]^2. No GL attachment row
    // convention is inherited here; native GL must decode its bottom-first rows.
    std::vector<std::array<double,2>> vertices;
};
class NativeCameraCellCoverageResult;
struct CameraCellCoverageReadback
{
    std::shared_ptr<const NativeCameraCellCoverageResult> nativeResult;
    std::shared_ptr<const NativeDirectGratingTerminalReceiverResult> parent;
    SubmissionIdentity submission;
    double evaluationTime=0;
    std::uint64_t inputGeneration=0;
    CameraCellCoverageBounds bounds;
    CameraCellCoveragePreparation preparation;
    CameraCellCoverageProof proof;
};
inline bool proveCameraCellCoverage(const std::shared_ptr<const NativeDirectGratingTerminalReceiverResult>&,
    const NativeFixtureSceneFrame&,const SubmissionIdentity&,double,CameraCellCoverageReadback&,std::string&);
class NativeCameraCellCoverageLease
{
public:
    virtual ~NativeCameraCellCoverageLease()=default;
    virtual const NativeFixtureSceneFrame& owner() const noexcept=0;
    virtual const Frame& frozenFrame() const noexcept=0;
    virtual std::uint64_t inputGeneration() const noexcept=0;
    virtual std::uint64_t proofGeneration() const noexcept=0;
protected:
    NativeCameraCellCoverageLease()=default;
    NativeCameraCellCoverageLease(const NativeCameraCellCoverageLease&)=delete;
    NativeCameraCellCoverageLease& operator=(const NativeCameraCellCoverageLease&)=delete;
    virtual void commit() noexcept=0; // Publication only; no allocation/failure.
private:
    friend bool proveCameraCellCoverage(const std::shared_ptr<const NativeDirectGratingTerminalReceiverResult>&,
        const NativeFixtureSceneFrame&,const SubmissionIdentity&,double,CameraCellCoverageReadback&,std::string&);
};
class NativeCameraCellCoverageBinding
{
public:
    virtual ~NativeCameraCellCoverageBinding()=default;
    virtual const NativeFixtureSceneFrame& coverageOwner() const noexcept=0;
    // Authenticate the exact published parent, terminal/cohort, scene/camera,
    // query/raster resources, generations/time and completed submission under
    // the real native guard. Pointer equality alone is not publication proof.
    virtual bool beginCameraCellCoverageLease(const std::shared_ptr<const NativeDirectGratingTerminalReceiverResult>&,
        const SubmissionIdentity&,double,std::unique_ptr<NativeCameraCellCoverageLease>&,
        NativeCameraCellCoverageBudget&,std::string& error) const
    {error="Native camera-cell coverage is unsupported";return false;}
    // Reconstruct/recheck all old owners, actual capacities and attempt records.
    // Register and permanently charge full operationWork BEFORE any candidate,
    // dedup, pairwise boundary, partition or native resource allocation/work.
    virtual bool reserveCameraCellCoverageLease(NativeCameraCellCoverageLease&,const CameraCellCoverageBounds&,
        const CameraCellCoverageReadback&,std::string& error) const
    {error="Native camera-cell coverage reservation is unsupported";return false;}
    // Prove a closed union covering each full admitted camera pixel, clipped
    // consistently by owning near/far planes, using the nominated receiver
    // object's/material's/instance's triangles across seams. Establish nearest
    // ownership over every region against ALL frozen triangles, with released
    // tie ordering. Exact or outward-rounded predicates and actual registered
    // native proof output are mandatory. Gaps, unsupported clipping/seams,
    // competing owners/depths or uncertain arithmetic refuse the whole operation.
    // Center/corner sampling, raster attachments and candidate measures cannot
    // certify this. No CPU visibility, status-flag fallback or replacement query.
    virtual bool dispatchCameraCellCoverage(NativeCameraCellCoverageLease&,const CameraCellCoveragePreparation&,
        CameraCellCoverageProof&,std::string& error) const
    {error="Native camera-cell conservative predicate is unsupported";return false;}
    // Required on actual dispatch and again on staged immutable output. Inspect
    // registered buffers/fences/generations, complete conservative partition,
    // union/seam/depth certificates, all-triangle work and retained byte ledger.
    // A shape-valid polygon or nonzero generation is not authenticated proof.
    virtual bool validateCameraCellCoverageProof(const NativeCameraCellCoverageLease&,
        const CameraCellCoverageReadback&,std::string& error) const
    {error="Native camera-cell proof inspection is unsupported";return false;}
    // Independently copy/stage under this lease; reconcile all actual capacities.
    // Uncommitted destruction preserves parent/old coverage and admitted work,
    // releasing only unescaped staged storage. Keep escaped/retained owners charged.
    virtual bool finishCameraCellCoverageLease(NativeCameraCellCoverageLease&,const CameraCellCoverageReadback&,
        std::shared_ptr<const NativeCameraCellCoverageResult>&,std::string& error) const
    {error="Native camera-cell coverage finish is unsupported";return false;}
protected:
    std::shared_ptr<const NativeCameraCellCoverageResult> makeCameraCellCoverageResult(const CameraCellCoverageReadback&) const;
};
class NativeCameraCellCoverageResult final
{
public:
    ~NativeCameraCellCoverageResult()=default;
    const NativeFixtureSceneFrame* owner() const noexcept{return owner_;}
    const CameraCellCoverageReadback& retained() const noexcept{return retained_;}
private:
    NativeCameraCellCoverageResult(const NativeFixtureSceneFrame& owner,const CameraCellCoverageReadback& value)
        :owner_(&owner),retained_(value){retained_.nativeResult.reset();}
    NativeCameraCellCoverageResult(const NativeCameraCellCoverageResult&)=delete;
    NativeCameraCellCoverageResult& operator=(const NativeCameraCellCoverageResult&)=delete;
    const NativeFixtureSceneFrame* owner_;
    CameraCellCoverageReadback retained_;
    friend class NativeCameraCellCoverageBinding;
};
inline std::shared_ptr<const NativeCameraCellCoverageResult>
NativeCameraCellCoverageBinding::makeCameraCellCoverageResult(const CameraCellCoverageReadback& value) const
{return std::shared_ptr<const NativeCameraCellCoverageResult>(new NativeCameraCellCoverageResult(coverageOwner(),value));}
inline bool sameCameraCellCoverageBounds(const CameraCellCoverageBounds& a,const CameraCellCoverageBounds& b)
{
    return std::tie(a.bytes,a.combinedWork,a.operationWork,a.nativeBytes,a.maximumPixels,a.maximumAssociations,a.triangles,
        a.maximumCandidates,a.depthPairs,a.linesPerPixel,a.regionsPerPixel,a.verticesPerPixel,a.maximumRegions,a.maximumVertices,
        a.preparationWork,a.partitionWork,a.triangleWork)
        ==std::tie(b.bytes,b.combinedWork,b.operationWork,b.nativeBytes,b.maximumPixels,b.maximumAssociations,b.triangles,
        b.maximumCandidates,b.depthPairs,b.linesPerPixel,b.regionsPerPixel,b.verticesPerPixel,b.maximumRegions,b.maximumVertices,
        b.preparationWork,b.partitionWork,b.triangleWork);
}
inline bool calculateCameraCellCoverageBounds(const NativeDirectGratingTerminalReceiverResult& parent,const Frame& frame,
    const NativeCameraCellCoverageBudget& native,const CameraCellCoverageReadback& incumbent,
    CameraCellCoverageBounds& output,std::string& error)
{
    const auto refuse=[&](){error="Native camera-cell arrangement exceeds authenticated whole-frame bounds";return false;};
    const auto maximum=renderpassoutput::kMaximumTotalBytes;const auto& rows=parent.retained();
    CameraCellCoverageBounds b;b.bytes=native.retainedBytes;
    b.triangles=frame.triangles.size();
    if(!rows.terminal||!rows.bounds.maximumRows||rows.bounds.maximumRows>UINT32_MAX||rows.rows.size()>rows.bounds.maximumRows
        ||!b.triangles||b.triangles>UINT32_MAX||native.retainedBytes<rows.bounds.bytes||native.retainedBytes>maximum
        ||native.combinedWork<rows.bounds.combinedWork||native.combinedWork>kMaximumQueryWork||!native.fixedBytes)return refuse();
    // Begin has authenticated the immutable parent. Count only disposition
    // words here, without allocation, dedup, geometry or pairwise work. At most
    // this many unique pixels can occur, always <=parent.maximumRows. Charge
    // this bounded linear metadata pass in preparationWork below.
    for(const auto& row:rows.rows)if(row.disposition==DirectGratingTerminalReceiverRow::Disposition::Pixel)++b.maximumPixels;
    b.maximumAssociations=b.maximumPixels;
    const auto multiply=[](std::uint64_t a,std::uint64_t c,std::uint64_t& value) {
        if(c&&a>UINT64_MAX/c){return false;}value=a*c;return true;
    };
    const auto sum=[](std::uint64_t a,std::uint64_t c,std::uint64_t& value) {
        if(a>UINT64_MAX-c){return false;}value=a+c;return true;
    };
    std::uint64_t pairProduct=0,edgeLines=0,product=0,perPixelVertices=0;
    // A triangle clipped by near/far planes has at most five projected edges;
    // add its plane-denominator/horizon line. Two plane ray-depth functions
    // exchange order on at most one line. Degenerate/equal functions require
    // native exact tie proof or Unsupported, never another unbounded partition.
    if(!multiply(b.triangles,b.triangles-1,pairProduct)||!multiply(6,b.triangles,edgeLines))return refuse();
    b.depthPairs=pairProduct/2;
    if(!sum(edgeLines,b.depthPairs,b.linesPerPixel)||!sum(b.linesPerPixel,1,product)
        ||!multiply(b.linesPerPixel,product,pairProduct))return refuse();
    b.regionsPerPixel=1+pairProduct/2;
    // Rectangle plus L full lines has <=(L+4)(L+1) edges. Counting both
    // incident faces bounds total polygon vertices by twice that number,
    // rather than allocating (L+4) vertices independently for every face.
    if(!sum(b.linesPerPixel,4,product)||!sum(b.linesPerPixel,1,perPixelVertices)
        ||!multiply(product,perPixelVertices,perPixelVertices)||!multiply(2,perPixelVertices,b.verticesPerPixel)
        ||!multiply(b.maximumPixels,b.triangles,b.maximumCandidates)
        ||!multiply(b.maximumPixels,b.regionsPerPixel,b.maximumRegions)
        ||!multiply(b.maximumPixels,b.verticesPerPixel,b.maximumVertices))return refuse();
    unsigned lineLog=0;
    for(auto n=b.linesPerPixel;n>1;n=(n+1)/2)++lineLog;
    std::uint64_t dedup=0,geometry=0,pairs=0,arrangement=0,walk=0,regions=0,vertices=0;
    // Work units are bounded geometry/predicate visits, not FP instructions.
    // Reserve sort/dedup, fixed-size clipping, all depth pairs, all line-event
    // construction/sorting, face walking and every region/vertex vs every
    // triangle. A native writer must use this bounded arrangement or refuse;
    // arbitrary recursive subdivision is not admitted by these bounds.
    if(!multiply(b.maximumAssociations,128,dedup)
        ||!multiply(b.maximumCandidates,96,geometry)||!sum(dedup,geometry,b.preparationWork)
        ||!sum(b.preparationWork,rows.rows.size(),b.preparationWork)
        ||!multiply(b.maximumPixels,b.depthPairs,pairs)||!multiply(pairs,16,pairs)
        ||!multiply(b.maximumPixels,pairProduct,arrangement)||!multiply(arrangement,std::uint64_t(lineLog)+8,arrangement)
        ||!multiply(b.maximumVertices,8,walk)||!sum(pairs,arrangement,b.partitionWork)
        ||!sum(b.partitionWork,walk,b.partitionWork)
        ||!multiply(b.maximumRegions,b.triangles,regions)||!multiply(b.maximumVertices,b.triangles,vertices)
        ||!sum(regions,vertices,b.triangleWork)||!sum(b.preparationWork,b.partitionWork,b.operationWork)
        ||!sum(b.operationWork,b.triangleWork,b.operationWork)
        ||b.operationWork>kMaximumQueryWork-native.combinedWork)return refuse();
    b.combinedWork=native.combinedWork+b.operationWork;
    const auto add=[&](std::uint64_t count,std::uint64_t stride=1) {
        if(!stride||count>(maximum-b.bytes)/stride){return false;}b.bytes+=count*stride;return true;
    };
    const auto nativeAdd=[&](std::uint64_t count,std::uint64_t stride) {
        if(!stride){return true;}const auto prior=b.bytes;
        if(!add(count,stride)){return false;}b.nativeBytes+=b.bytes-prior;return true;
    };
    if(!nativeAdd(b.maximumPixels,native.pixelBytes)||!nativeAdd(b.maximumCandidates,native.candidateBytes)
        ||!nativeAdd(b.maximumRegions,native.regionBytes)||!nativeAdd(b.maximumVertices,native.vertexBytes)
        ||!nativeAdd(1,native.fixedBytes))return refuse();
    // Three independently decoded copies, each with <=2x vector capacity:
    // operation/caller, registered native inspection and immutable retained.
    if(!add(6*b.maximumPixels,sizeof(CameraCellCoverageInput))
        ||!add(6*b.maximumAssociations,sizeof(CameraCellCoverageAssociation))
        ||!add(6*b.maximumCandidates,sizeof(CameraCellCoverageCandidate))
        ||!add(6*b.maximumRegions,sizeof(CameraCellCoverageRegion))
        ||!add(6*b.maximumVertices,sizeof(std::array<double,2>))
        ||!add(3,sizeof(CameraCellCoverageReadback))||!add(sizeof(NativeCameraCellCoverageResult)+128)
        ||!add(6,rows.submission.frameIdentity.size()))return refuse();
    const auto old=[&](const CameraCellCoverageReadback& value) {
        return add(value.preparation.pixels.capacity(),sizeof(CameraCellCoverageInput))
            &&add(value.preparation.associations.capacity(),sizeof(CameraCellCoverageAssociation))
            &&add(value.preparation.candidates.capacity(),sizeof(CameraCellCoverageCandidate))
            &&add(value.proof.regions.capacity(),sizeof(CameraCellCoverageRegion))
            &&add(value.proof.vertices.capacity(),sizeof(std::array<double,2>))&&add(value.submission.frameIdentity.capacity());
    };
    if(!old(incumbent))return refuse();
    if(incumbent.parent&&incumbent.parent.get()!=&parent&&!add(incumbent.parent->retained().bounds.bytes))return refuse();
    if(incumbent.nativeResult) {
        const auto& retained=incumbent.nativeResult->retained();
        if(!add(sizeof(NativeCameraCellCoverageResult)+128)||!old(retained))return refuse();
        if(incumbent.nativeResult->owner()!=parent.owner()&&!add(retained.bounds.bytes))return refuse();
        if(retained.parent&&retained.parent.get()!=&parent&&retained.parent!=incumbent.parent
            &&!add(retained.parent->retained().bounds.bytes))return refuse();
    }
    output=b;error.clear();return true;
}
// Call only after full admission. This is sorting and one-triangle geometry,
// not a CPU union, partition, depth sweep or visibility operation.
inline bool prepareCameraCellCoverage(const NativeDirectGratingTerminalReceiverResult& parent,const Frame& frame,
    const CameraCellCoverageBounds& bounds,CameraCellCoveragePreparation& output,std::string& error)
{
    const auto refuse=[&](){error="Camera-cell candidate association is unsupported or exceeds admission";return false;};
    const auto& rows=parent.retained().rows;CameraCellCoveragePreparation candidate;
    if(rows.size()>parent.retained().bounds.maximumRows||frame.triangles.size()!=bounds.triangles
        ||!nativeFixtureDimensionsWithinBounds(frame.width,frame.height)||!validFixtureCameraOverride(frame.camera))return refuse();
    candidate.pixels.reserve(bounds.maximumPixels);candidate.associations.reserve(bounds.maximumAssociations);
    candidate.candidates.reserve(bounds.maximumCandidates);
    for(std::size_t i=0;i<rows.size();++i) {
        const auto& row=rows[i];
        if(row.disposition!=DirectGratingTerminalReceiverRow::Disposition::Pixel)continue;
        if(candidate.associations.size()>=bounds.maximumAssociations||!row.pixel.projected||row.pixel.worldArea!=0
            ||row.pixel.x>=frame.width||row.pixel.y>=frame.height
            ||!receiverTriangle(frame,row.hit))return refuse();
        candidate.associations.push_back({std::uint32_t(i),0,row.input});
    }
    const auto key=[&](const CameraCellCoverageAssociation& a) {
        const auto& pixel=rows[a.parentRow].pixel;
        return ((std::uint64_t(pixel.y)*frame.width+pixel.x)<<32)|a.parentRow;
    };
    // Allocation-free 64-bit radix partition. Each association is inspected
    // at most twice per bit; sorting has an explicit <=128*N visit bound.
    const auto sort=[&](auto&& self,std::size_t first,std::size_t last,unsigned bit)->void {
        if(last-first<2){return;}std::size_t low=first,high=last;
        while(low<high) {
            if(!(key(candidate.associations[low])&(std::uint64_t(1)<<bit)))++low;
            else {--high;std::swap(candidate.associations[low],candidate.associations[high]);}
        }
        if(bit){self(self,first,low,bit-1);self(self,low,last,bit-1);}
    };
    sort(sort,0,candidate.associations.size(),63);
    for(auto& association:candidate.associations) {
        const auto& row=rows[association.parentRow];const auto& h=row.hit;
        if(candidate.pixels.empty()||candidate.pixels.back().x!=row.pixel.x||candidate.pixels.back().y!=row.pixel.y)
            candidate.pixels.push_back({row.pixel.x,row.pixel.y,h.objectId,h.materialId,h.instanceId,
                std::uint64_t(&association-candidate.associations.data()),0});
        auto& pixel=candidate.pixels.back();
        // Conflicting target surfaces in one pixel need a different proof
        // domain. Preserve both parents and refuse rather than drop one row.
        if(std::tie(pixel.object,pixel.material,pixel.instance)!=std::tie(h.objectId,h.materialId,h.instanceId))return refuse();
        association.pixel=std::uint32_t(candidate.pixels.size()-1);++pixel.associationCount;
    }
    for(std::size_t p=0;p<candidate.pixels.size();++p)for(std::size_t t=0;t<frame.triangles.size();++t) {
        const auto& triangle=frame.triangles[t];const auto& pixel=candidate.pixels[p];
        CameraCellCoverageCandidate measure;measure.pixel=std::uint32_t(p);measure.triangle=std::uint32_t(t);
        if(!triangle.glass&&triangle.reflectance&&std::tie(triangle.object,triangle.material,triangle.instance)
            ==std::tie(pixel.object,pixel.material,pixel.instance)) {
            std::string geometryError;
            if(prepareCameraCellGeometry(frame,triangle,pixel.x,pixel.y,measure.geometry,geometryError))
                measure.measure=CameraCellCoverageCandidate::Measure::Prepared;
        }
        candidate.candidates.push_back(measure);
    }
    if(candidate.pixels.size()>bounds.maximumPixels||candidate.pixels.capacity()>2*bounds.maximumPixels
        ||candidate.associations.capacity()>2*bounds.maximumAssociations||candidate.candidates.size()>bounds.maximumCandidates
        ||candidate.candidates.capacity()>2*bounds.maximumCandidates)return refuse();
    output=std::move(candidate);error.clear();return true;
}
inline bool sameCameraCellCoverageReadback(const CameraCellCoverageReadback& a,const CameraCellCoverageReadback& b,
    bool compareNativeResult=true)
{
    if((compareNativeResult&&a.nativeResult!=b.nativeResult)||a.parent!=b.parent||!(a.submission==b.submission)
        ||a.evaluationTime!=b.evaluationTime||a.inputGeneration!=b.inputGeneration||!sameCameraCellCoverageBounds(a.bounds,b.bounds)
        ||a.preparation.pixels.size()!=b.preparation.pixels.size()||a.preparation.associations.size()!=b.preparation.associations.size()
        ||a.preparation.candidates.size()!=b.preparation.candidates.size()||a.proof.generation!=b.proof.generation
        ||a.proof.work!=b.proof.work||a.proof.regions.size()!=b.proof.regions.size()||a.proof.vertices!=b.proof.vertices)return false;
    for(std::size_t i=0;i<a.preparation.pixels.size();++i) {
        const auto& x=a.preparation.pixels[i];const auto& y=b.preparation.pixels[i];
        if(std::tie(x.x,x.y,x.object,x.material,x.instance,x.firstAssociation,x.associationCount)
            !=std::tie(y.x,y.y,y.object,y.material,y.instance,y.firstAssociation,y.associationCount))return false;
    }
    for(std::size_t i=0;i<a.preparation.associations.size();++i) {
        const auto& x=a.preparation.associations[i];const auto& y=b.preparation.associations[i];
        if(x.parentRow!=y.parentRow||x.pixel!=y.pixel||!sameDirectGratingTerminalReceiverInput(x.order,y.order))return false;
    }
    for(std::size_t i=0;i<a.preparation.candidates.size();++i) {
        const auto& x=a.preparation.candidates[i];const auto& y=b.preparation.candidates[i];const auto& g=x.geometry;const auto& h=y.geometry;
        if(x.pixel!=y.pixel||x.triangle!=y.triangle||x.measure!=y.measure||g.vertices!=h.vertices||g.vertexCount!=h.vertexCount
            ||std::tie(g.camera,g.x,g.y,g.object,g.material,g.instance,g.triangle,g.worldArea,g.clippedSolidAngle,g.pixelSolidAngle)
                !=std::tie(h.camera,h.x,h.y,h.object,h.material,h.instance,h.triangle,h.worldArea,h.clippedSolidAngle,h.pixelSolidAngle))return false;
    }
    for(std::size_t i=0;i<a.proof.regions.size();++i) {
        const auto& x=a.proof.regions[i];const auto& y=b.proof.regions[i];
        if(std::tie(x.pixel,x.triangle,x.firstVertex,x.vertexCount,x.projectedArea)
            !=std::tie(y.pixel,y.triangle,y.firstVertex,y.vertexCount,y.projectedArea))return false;
    }
    return true;
}
inline bool cameraCellCoverageCapacities(const CameraCellCoverageReadback& value)
{
    const auto& b=value.bounds;const auto& p=value.preparation;
    return p.pixels.size()<=b.maximumPixels&&p.pixels.capacity()<=2*b.maximumPixels
        &&p.associations.size()<=b.maximumAssociations&&p.associations.capacity()<=2*b.maximumAssociations
        &&p.candidates.size()<=b.maximumCandidates&&p.candidates.capacity()<=2*b.maximumCandidates
        &&value.proof.regions.size()<=b.maximumRegions&&value.proof.regions.capacity()<=2*b.maximumRegions
        &&value.proof.vertices.size()<=b.maximumVertices&&value.proof.vertices.capacity()<=2*b.maximumVertices;
}
// Decoded shape checks only. This function cannot authenticate finite coverage,
// seams or visibility. The mandatory native inspector proves those separately.
inline bool validateCameraCellCoverageShape(const Frame& frame,const CameraCellCoverageReadback& value,std::string& error)
{
    const auto refuse=[&](){error="Native camera-cell decoded partition shape changed";return false;};
    if(!value.proof.generation||!cameraCellCoverageCapacities(value)||value.proof.work>value.bounds.operationWork)return refuse();
    std::size_t nextVertex=0,nextRegion=0;
    for(std::size_t p=0;p<value.preparation.pixels.size();++p) {
        const auto& pixel=value.preparation.pixels[p];double area=0;std::uint64_t regions=0,vertices=0;
        while(nextRegion<value.proof.regions.size()&&value.proof.regions[nextRegion].pixel==p) {
            const auto& region=value.proof.regions[nextRegion++];
            if(region.triangle>=frame.triangles.size()||region.firstVertex!=nextVertex||region.vertexCount<3
                ||region.vertexCount>value.bounds.linesPerPixel+4||region.vertexCount>value.proof.vertices.size()-nextVertex
                ||!std::isfinite(region.projectedArea)||region.projectedArea<=0||region.projectedArea>1)return refuse();
            const auto& triangle=frame.triangles[region.triangle];
            if(triangle.glass||!triangle.reflectance||std::tie(triangle.object,triangle.material,triangle.instance)
                !=std::tie(pixel.object,pixel.material,pixel.instance))return refuse();
            double signedArea=0;
            for(std::size_t v=0;v<region.vertexCount;++v) {
                const auto& x=value.proof.vertices[nextVertex+v];const auto& y=value.proof.vertices[nextVertex+(v+1)%region.vertexCount];
                if(!std::isfinite(x[0])||!std::isfinite(x[1])||x[0]<0||x[0]>1||x[1]<0||x[1]>1)return refuse();
                signedArea+=x[0]*y[1]-y[0]*x[1];
            }
            if(std::abs(.5*std::abs(signedArea)-region.projectedArea)>1e-10)return refuse();
            area+=region.projectedArea;nextVertex+=region.vertexCount;vertices+=region.vertexCount;++regions;
        }
        if(!regions||regions>value.bounds.regionsPerPixel||vertices>value.bounds.verticesPerPixel||std::abs(area-1)>1e-10)return refuse();
    }
    if(nextRegion!=value.proof.regions.size()||nextVertex!=value.proof.vertices.size()
        ||(!value.preparation.pixels.empty()&&!value.proof.work))return refuse();
    error.clear();return true;
}
inline bool proveCameraCellCoverage(const std::shared_ptr<const NativeDirectGratingTerminalReceiverResult>& parent,
    const NativeFixtureSceneFrame& owner,const SubmissionIdentity& expected,double evaluationTime,
    CameraCellCoverageReadback& output,std::string& error)
{
    const auto refuse=[&](const char* message){error=message;return false;};
    const auto* native=dynamic_cast<const NativeCameraCellCoverageBinding*>(&owner);
    if(!native)return refuse("Native camera-cell coverage is unsupported");
    if(!parent||parent->owner()!=&owner||&native->coverageOwner()!=&owner||!parent->retained().terminal
        ||!(parent->retained().submission==expected)||evaluationTime!=parent->retained().evaluationTime)
        return refuse("Native camera-cell coverage requires the exact published receiver parent");
    std::unique_ptr<NativeCameraCellCoverageLease> lease;NativeCameraCellCoverageBudget budget;
    if(!native->beginCameraCellCoverageLease(parent,expected,evaluationTime,lease,budget,error))return false;
    const auto& cohort=parent->retained().terminal->retained().incident.cohort;
    if(!cohort||cohort->sources().empty()||!lease||&lease->owner()!=&owner||!lease->inputGeneration()||!lease->proofGeneration()
        ||&lease->frozenFrame()!=&cohort->sources()[0]->frame||!owner.nativeResourceCacheIdentity())
        return refuse("Native camera-cell coverage lacks its registered frozen lease");
    const auto& frame=lease->frozenFrame();CameraCellCoverageBounds bounds;
    if(frame.width!=owner.width()||frame.height!=owner.height())return refuse("Native camera-cell frozen extent changed");
    if(!calculateCameraCellCoverageBounds(*parent,frame,budget,output,bounds,error)
        ||!native->reserveCameraCellCoverageLease(*lease,bounds,output,error))return false;
    CameraCellCoverageReadback candidate;candidate.parent=parent;candidate.submission=expected;
    candidate.evaluationTime=evaluationTime;candidate.inputGeneration=lease->inputGeneration();candidate.bounds=bounds;
    if(!prepareCameraCellCoverage(*parent,frame,bounds,candidate.preparation,error)
        ||!native->dispatchCameraCellCoverage(*lease,candidate.preparation,candidate.proof,error))return false;
    if(candidate.proof.generation!=lease->proofGeneration()||candidate.submission.frameIdentity.capacity()>2*expected.frameIdentity.size())
        return refuse("Native camera-cell proof generation or capacity changed");
    if(!validateCameraCellCoverageShape(frame,candidate,error)||!native->validateCameraCellCoverageProof(*lease,candidate,error)
        ||!native->finishCameraCellCoverageLease(*lease,candidate,candidate.nativeResult,error))return false;
    if(!candidate.nativeResult||candidate.nativeResult->owner()!=&owner)return refuse("Native camera-cell result has no staged native owner");
    const auto& retained=candidate.nativeResult->retained();
    if(retained.nativeResult||!cameraCellCoverageCapacities(retained)
        ||retained.submission.frameIdentity.capacity()>2*expected.frameIdentity.size()
        ||!sameCameraCellCoverageReadback(candidate,retained,false))return refuse("Native camera-cell retained result differs or exceeds admission");
    if(!native->validateCameraCellCoverageProof(*lease,candidate,error))return false;
    lease->commit();output=std::move(candidate);error.clear();return true;
}
} // namespace arbitgpu::glasscaustics
