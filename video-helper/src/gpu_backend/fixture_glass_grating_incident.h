#pragma once

#include "fixture_glass_caustics.h"
#include "../diffraction_material_binding_admission.h"
#include "../diffraction_order_bsdf.h"

namespace arbitgpu::glasscaustics
{
inline bool sameGratingPoint(Point a,Point b,float tolerance=2e-4f)
{return finite(a)&&finite(b)&&dot(sub(a,b),sub(a,b))<=tolerance*tolerance;}
inline bool gratingFrame(const std::array<Point,3>& p,const std::array<std::array<float,2>,3>& uv,
    Point& tangent,Point& bitangent,Point& normal)
{
    const auto e1=sub(p[1],p[0]),e2=sub(p[2],p[0]);normal=cross(e1,e2);
    const float du1=uv[1][0]-uv[0][0],dv1=uv[1][1]-uv[0][1];
    const float du2=uv[2][0]-uv[0][0],dv2=uv[2][1]-uv[0][1],det=du1*dv2-dv1*du2;
    if(!normalize(normal)||!std::isfinite(det)||det<=1e-12f)return false;
    Point authoredBitangent{};
    for(unsigned a=0;a<3;++a) {tangent[a]=(e1[a]*dv2-e2[a]*dv1)/det;authoredBitangent[a]=(e2[a]*du1-e1[a]*du2)/det;}
    tangent=advance(tangent,normal,-dot(tangent,normal));
    if(!normalize(tangent)||!finite(authoredBitangent))return false;
    bitangent=cross(normal,tangent);
    // Mirrored UVs are refused at this first bound; silently reversing the
    // reciprocal-lattice orientation would change signed diffraction orders.
    return dot(bitangent,authoredBitangent)>0;
}
// Shared production lowering, also used by the renderer's finite-lobe compiler.
// This creates immutable compiler bytes; it grants no native render capability.
inline std::shared_ptr<const NativeFixtureSurfaceMaterialProgram> lowerNativeDiffractionMaterial(
    const diffractionmaterialbinding::AdmittedImportedSceneDiffractionMaterial& admitted,
    HarmonicMIDI::grid::SceneMaterialId authoredMaterial,NativeFixtureMaterialBackend backend)
{
    using NativeProgram=NativeFixtureSurfaceMaterialProgram;
    const auto& lighting=admitted.lighting().description();
    auto native = std::make_shared<NativeProgram>();
    native->backend = backend;
    native->kind = arbitgpu::NativeFixtureMaterialKind::DiffractionReflective;
    native->object = admitted.object();
    native->authoredScene = admitted.scene();
    native->authoredMaterial = authoredMaterial;
    native->sceneRevision = admitted.sceneRevision();
    native->structuralRevision = admitted.structuralRevision();
    native->evaluationRevision = admitted.evaluationRevision();
    native->materialRevision = admitted.materialRevision();
    native->bindingDigest = admitted.material().structuralDigest();
    native->diffractionPathCount = lighting.pathCount;
    native->diffractionLightingAdmission
        = std::make_shared<const diffractionmaterial::AdmittedLightingPlan>(admitted.lighting());
    native->programIdentity = arbitgpu::nativeFixtureDiffractionProgramIdentity(
        native->bindingDigest, *native->diffractionLightingAdmission, 0, 1.0f, {}, false, native.get());
    for (std::size_t pathIndex = 0; pathIndex < lighting.pathCount; ++pathIndex)
    {
        native->diffractionPaths[pathIndex] = diffractionmaterial::makeGpuLightingPath(
            diffractionmaterial::physicalcheckpoint::makeGpuParameters(
                admitted.material(), lighting.paths[pathIndex].incident),
            lighting.paths[pathIndex]);
        if (lighting.version != 1)
            native->diffractionPaths[pathIndex].material.incident = { 0.0f, 0.0f, 1.0f, 0.0f };
        native->diffractionMaximumBounceDepth = std::max(
            native->diffractionMaximumBounceDepth,
            lighting.paths[pathIndex].bounceDepth);
    }
    if (lighting.version == 3)
        native->diffractionMaximumBounceDepth = diffractionmaterial::kMaximumIndirectBounces;
    return native;
}
// Required private lowering seam for discrete glass-incident orders. Ordinary
// surface camera shaders still require their released positive RMS slope.
inline std::shared_ptr<const NativeFixtureSurfaceMaterialProgram> compileDiscreteGrating(
    const HarmonicMIDI::grid::Visual3DScene& scene,
    const diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest& request,
    NativeFixtureMaterialBackend backend,std::string& error)
{
    const auto refuse=[&](const char* message)->std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>{error=message;return {};};
    if(request.graphFrame||request.spatialFoil)return refuse("Discrete grating compiler refuses frame products and spatial foil");
    const auto material=diffractionmaterialbinding::admit(request,error);if(!material)return {};
    const auto& admitted=*material;
    if(!HarmonicMIDI::grid::validateVisual3DScene(scene).valid())return refuse("Discrete grating compiler scene is invalid");
    const auto* object=HarmonicMIDI::grid::visual3d_detail::findById(scene.objects,scene.objectCount,admitted.object());
    if(admitted.scene()!=scene.id||!object||!object->indexCount
        ||object->indexCount%3||object->indexCount/3>kMaximumNativeDiffractionTriangles||admitted.spatialFoil())
        return refuse("Discrete grating compiler requires exact bounded authored scene geometry");
    Point firstNormal{},firstTangent{},firstPosition{};
    for(std::size_t triangle=0;triangle<object->indexCount/3;++triangle) {
        std::array<Point,3> positions;std::array<std::array<float,2>,3> uv;
        for(unsigned corner=0;corner<3;++corner) {
            const auto index=scene.indices[object->firstIndex+triangle*3+corner];
            if(index>=object->vertexCount)return refuse("Discrete grating compiler triangle index is invalid");
            const auto& vertex=scene.vertices[object->firstVertex+index];
            positions[corner]={vertex.position.x,vertex.position.y,vertex.position.z};uv[corner]={vertex.uv.x,vertex.uv.y};
        }
        Point tangent,bitangent,normal;
        if(!gratingFrame(positions,uv,tangent,bitangent,normal))return refuse("Discrete grating compiler requires right-handed authored UVs");
        if(!triangle){firstNormal=normal;firstTangent=tangent;firstPosition=positions[0];}
        else if(!sameGratingPoint(normal,firstNormal)||!sameGratingPoint(tangent,firstTangent)
            ||std::abs(dot(sub(positions[0],firstPosition),firstNormal))>kRayMinimum)
            return refuse("Discrete grating compiler requires one static planar frame");
    }
    auto native=lowerNativeDiffractionMaterial(admitted,object->material,backend);
    smoothglass::DistinctGratingAdmission selection{admitted.object(),
        std::make_shared<const diffractionmaterial::AdmittedDiffractionMaterialIR>(admitted.material())};
    if(!smoothglass::validDistinctGrating(*native,selection))return refuse("Discrete grating compiler IR, lighting or owner is outside the private contract");
    error.clear();return native;
}
inline void appendGratingWitness(const GratingTriangleWitness& t,std::vector<Texel>& out)
{
    for(unsigned v=0;v<3;++v)out.push_back({t.positions[v][0],t.positions[v][1],t.positions[v][2],t.uv[v][0]});
    out.push_back({t.uv[0][1],t.uv[1][1],t.uv[2][1],0});
    out.push_back({float(t.object&65535),float(t.object>>16),float(t.material&65535),float(t.material>>16)});
    out.push_back({float(t.instance),float(t.triangle),float(kGratingIncidentVersion),0});
    out.push_back({t.tangent[0],t.tangent[1],t.tangent[2],0});
    out.push_back({t.normal[0],t.normal[1],t.normal[2],0});
}
inline bool calculateGratingIncidentBounds(const Frame& frame,std::size_t triangles,
    GratingIncidentBounds& output,std::string& error)
{
    if(frame.work.empty()||frame.work.size()>kMaximumEntrySamples*8||!frame.bounds.maximumDeposits
        ||frame.bounds.maximumDeposits>frame.work.size()*kMaximumBranchVisits||frame.bounds.queryWork>kMaximumQueryWork
        ||frame.bounds.deposition.depositionWork>kMaximumDepositionWork
        ||frame.bounds.transientBytes>renderpassoutput::kMaximumTotalBytes||!triangles||triangles>kMaximumNativeDiffractionTriangles
        ||!std::isfinite(frame.query.control[0])||frame.query.control[0]<1
        ||frame.query.control[0]>=frame.query.texels.size()||std::floor(frame.query.control[0])!=frame.query.control[0])
    {error="Glass to grating staging bounds are incomplete";return false;}
    const auto& allowance=frame.query.texels[std::size_t(frame.query.control[0])-1];
    for(unsigned i=0;i<2;++i)if(!std::isfinite(allowance[i])||allowance[i]<0||allowance[i]>=65536||std::floor(allowance[i])!=allowance[i])
    {error="Glass to grating query allowance is invalid";return false;}
    const auto query=std::uint64_t(allowance[0])+(std::uint64_t(allowance[1])<<16);
    GratingIncidentBounds candidate;candidate.maximumIncidents=frame.bounds.maximumDeposits;
    candidate.maximumOrders=candidate.maximumIncidents*kGratingOrdersPerIncident;
    // Each physical order needs one closest-hit leg and two possible receiver
    // camera legs. The native visit/stack/deposit caps still apply globally.
    candidate.queryWork=frame.bounds.queryWork+candidate.maximumOrders*3*query;
    candidate.orderWork=candidate.maximumOrders;candidate.combinedWork=candidate.queryWork+candidate.orderWork;
    candidate.bytes=triangles*(2*kGratingWitnessStride*sizeof(Texel)+sizeof(GratingTriangleWitness))
        +candidate.maximumIncidents*kGratingIncidentStride*sizeof(Texel)
        +candidate.maximumOrders*kGratingOrderStride*sizeof(Texel);
    // Reserve native adjunct storage and decoded host copies simultaneously.
    candidate.bytes+=candidate.maximumIncidents*sizeof(NativeGratingIncident);
    // Reserve the entire aligned private pending array, including ancestry and
    // both origin derivatives. GL uses registers; Metal CBranch is 128 bytes.
    candidate.bytes+=frame.work.size()*(kMaximumPendingBranches*kGratingPendingBranchBytes+kGratingBranchLocalBytes);
    // The frozen staging Frame owns its copied arrays alongside the caller's
    // frame. Prepared retains immutable scene/material owners; its query copy
    // and retained program/identity strings are charged at binding admission.
    candidate.bytes+=sizeof(GratingIncidentBinding)+sizeof(NativeGratingReadback)
        +candidate.maximumOrders*(kGratingChildReceiptStride*sizeof(Texel)+sizeof(NativeGratingReadback::ChildReceipt))
        +frame.work.size()*(kGratingWorkReceiptStride*sizeof(Texel)+sizeof(NativeGratingReadback::WorkReceipt))
        +(frame.query.texels.size()+frame.workTexels.size()+frame.triangleTexels.size())*sizeof(Texel)
        +frame.instances.size()*sizeof(opaquevisibility::Instance)+frame.triangles.size()*sizeof(Triangle)
        +frame.entries.size()*sizeof(EntrySample)+frame.work.size()*sizeof(EntryWork);
    candidate.bytes+=frame.identity.size()+frame.sourceIdentity.size()+frame.branchPolicyIdentity.size()+frame.cameraIdentity.size();
    if(!query||query>opaquevisibility::kMaximumClosestHitWork||candidate.combinedWork>kMaximumQueryWork
        ||candidate.orderWork>kMaximumDepositionWork-frame.bounds.deposition.depositionWork
        ||candidate.bytes>renderpassoutput::kMaximumTotalBytes-frame.bounds.transientBytes)
    {error="Glass to grating queries, orders or storage exceed existing frame bounds";return false;}
    output=candidate;error.clear();return true;
}
// Freeze either a legacy Surface staging target plus an exact grating override,
// or the opt-in distinct-grating staging admission. Neither enables Color.
inline bool prepareGratingIncidentBinding(const Prepared& prepared,const Frame& frame,
    const std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>& gratingOwner,
    const std::shared_ptr<const diffractionmaterial::AdmittedDiffractionMaterialIR>& material,
    double evaluationTime,std::shared_ptr<const GratingIncidentBinding>& output,std::string& error,
    SourceMeasureUse sourceUse=SourceMeasureUse::LegacyNative)
{
    const auto refuse=[&](const char* text){error=text;return false;};
    if(!validateTables(prepared,frame,error,true,sourceUse))return false;
    if(!prepared.sceneLight.isValid()||!prepared.branchPolicy||frame.roughness!=0||frame.multipleScatterPolicy
        ||!gratingOwner||!material||!std::isfinite(evaluationTime)||gratingOwner->object==prepared.glass.object)
        return refuse("Glass to grating requires distinct static smooth owners and directional Li8");
    const auto& root=*prepared.glass.sourceMaterial;const auto& scene=*prepared.glass.sourceScene;
    const auto* glass=nativeSurfaceForObject(&root,prepared.glass.object);
    if(!glass||glass->parameters.transmissionIorClearcoat[3]<=0)
        return refuse("Glass to grating first contract requires authored nonzero Cauchy dispersion");
    const auto* target=HarmonicMIDI::grid::visual3d_detail::findById(scene.objects,scene.objectCount,gratingOwner->object);
    if(!target||gratingOwner->kind!=NativeFixtureMaterialKind::DiffractionReflective
        ||gratingOwner->layoutVersion!=NativeFixtureSurfaceMaterialProgram::kLayoutVersion
        ||gratingOwner->backend!=root.backend||gratingOwner->authoredScene!=scene.id
        ||gratingOwner->authoredMaterial!=target->material||gratingOwner->sceneRevision!=root.sceneRevision
        ||gratingOwner->structuralRevision!=root.structuralRevision||gratingOwner->evaluationRevision!=root.evaluationRevision
        ||gratingOwner->materialRevision!=root.structuralRevision||gratingOwner->bindingDigest.empty()
        ||gratingOwner->programIdentity.empty()||gratingOwner->vertexProgram||gratingOwner->frameEndpoint
        ||gratingOwner->baseColorSource!=NativeFixtureSurfaceMaterialProgram::BaseColorSource::ConstantLinear
        ||gratingOwner->importedBaseColorTexture||!gratingOwner->objectPrograms.empty()
        ||gratingOwner->diffractionPathCount!=1||gratingOwner->diffractionPatternMask
        ||!gratingOwner->diffractionProductDigest.empty()||gratingOwner->diffractionMaskCoverage!=1
        ||gratingOwner->diffractionFoilMaximumEvaluations||gratingOwner->diffractionOccupancyRectangleCount)
        return refuse("Glass to grating exact material override owner is unsupported or stale");
    if(prepared.glass.distinctGrating&&(prepared.glass.distinctGrating->object!=gratingOwner->object
        ||prepared.glass.distinctGrating->material->structuralDigest()!=material->structuralDigest()))
        return refuse("Glass to grating admitted target and incident IR differ");
    const auto& d=material->description();
    if(d.geometry.lattice!=diffractionmaterial::GratingLattice::OneDimensional
        ||d.grooveField.mode!=diffractionmaterial::GrooveFieldMode::Constant
        ||d.microstructure.profile!=diffractionmaterial::GrooveProfile::BinaryRectangular
        ||d.coating.model!=diffractionmaterial::CoatingModel::Uncoated
        ||d.roughness.rmsHeightNanometres!=0||d.roughness.rmsSlope!=0
        ||d.spectrum.firstOrder!=1||d.spectrum.lastOrder!=1
        ||d.spectrum.wavelengthsNanometres!=diffractionmaterial::physicalcheckpoint::kWavelengths)
        return refuse("Glass to grating first contract requires canonical planar binary orders zero and plus/minus one");
    auto candidate=std::make_shared<GratingIncidentBinding>();candidate->prepared=prepared;candidate->frame=frame;
    candidate->gratingOwner=snapshotNativeSurfaceProgram(gratingOwner);candidate->material=material;candidate->evaluationTime=evaluationTime;
    // Material IR must match the existing compiler-owned native parameter block.
    auto gpu=diffractionmaterial::physicalcheckpoint::makeGpuParameters(*material,{});
    auto actual=gratingOwner->diffractionPaths[0].material;
    actual.incident=gpu.incident; // The new incident supplies direction/flux.
    for(unsigned b=0;b<8;++b)actual.spectral[b].y=gpu.spectral[b].y;
    if(std::memcmp(&actual,&gpu,sizeof(gpu))!=0)return refuse("Glass to grating native material bytes differ from admitted IR");
    for(const auto& triangle:frame.triangles)if(triangle.object==target->id.value) {
        GratingTriangleWitness witness;witness.object=triangle.object;witness.material=triangle.material;
        witness.instance=triangle.instance;witness.triangle=triangle.triangle;witness.positions=triangle.points;
        for(unsigned v=0;v<3;++v) {
            const auto index=scene.indices[target->firstIndex+triangle.triangle*3+v];
            const auto& uv=scene.vertices[target->firstVertex+index].uv;witness.uv[v]={uv.x,uv.y};
        }
        if(!gratingFrame(witness.positions,witness.uv,witness.tangent,witness.bitangent,witness.normal))
            return refuse("Glass to grating authored UV frame is degenerate or mirrored");
        if(!candidate->triangles.empty()) {
            const auto& first=candidate->triangles.front();
            if(!sameGratingPoint(first.normal,witness.normal)||!sameGratingPoint(first.tangent,witness.tangent)
                ||std::abs(dot(sub(witness.positions[0],first.positions[0]),first.normal))>kRayMinimum)
                return refuse("Glass to grating target is not one static planar UV frame");
        }
        appendGratingWitness(witness,candidate->witnessTexels);candidate->triangles.push_back(witness);
    }
    if(candidate->triangles.empty()||candidate->triangles.size()>kMaximumNativeDiffractionTriangles)
        return refuse("Glass to grating target triangle coverage exceeds its existing bound");
    if(!calculateGratingIncidentBounds(frame,candidate->triangles.size(),candidate->bounds,error))return false;
    const auto programBytes=sizeof(NativeFixtureSurfaceMaterialProgram)+gratingOwner->bindingDigest.size()+gratingOwner->programIdentity.size()
        +sizeof(diffractionmaterial::AdmittedDiffractionMaterialIR)+material->structuralDigest().size()
        +prepared.glass.query.texels.size()*sizeof(Texel)+prepared.glass.query.meshes.size()*sizeof(opaquevisibility::Mesh)
        +prepared.identity.size()+frame.identity.size()+material->structuralDigest().size()+gratingOwner->bindingDigest.size()
        +gratingOwner->programIdentity.size()+candidate->witnessTexels.size()*16*sizeof(float)+128;
    if(programBytes>renderpassoutput::kMaximumTotalBytes-frame.bounds.transientBytes-candidate->bounds.bytes)
        return refuse("Glass to grating retained material storage exceeds the existing frame bound");
    candidate->bounds.bytes+=programBytes;
    candidate->identity=frame.identity;word(candidate->identity,kGratingIncidentVersion);
    for(const auto* id:{&material->structuralDigest(),&gratingOwner->bindingDigest,&gratingOwner->programIdentity})
    {word(candidate->identity,id->size());candidate->identity+=*id;}
    std::uint64_t timeBits=0;static_assert(sizeof(timeBits)==sizeof(evaluationTime));
    std::memcpy(&timeBits,&evaluationTime,sizeof(timeBits));word(candidate->identity,timeBits);
    for(const auto& texel:candidate->witnessTexels)for(float value:texel)bits(candidate->identity,value);
    output=std::move(candidate);error.clear();return true;
}
// Private selected-light contribution at actual queried endpoints. These
// transient records grant neither all-light illumination nor patch coverage.
inline constexpr std::uint32_t kDirectGratingContributionVersion=1;
// Four input texels, six source echo, two five-texel hits, barycentric,
// two source-ray texels, two band texels and two exact uint control texels.
inline constexpr std::size_t kDirectGratingContributionNativeBytes=27*sizeof(Texel);
struct alignas(16) NativeDirectGratingEndpointInput
{
    std::array<std::uint32_t,4> primitive{}; // object, material, instance, triangle
    std::array<std::uint32_t,4> control{kDirectGratingContributionVersion,0,0,0};
    Texel originMinimum{},directionMaximum{};
};
static_assert(sizeof(NativeDirectGratingEndpointInput)==4*sizeof(Texel));
struct DirectGratingContributionBounds
{
    std::uint64_t maximumEndpoints=0,queryWork=0,bandWork=0,combinedWork=0,bytes=0;
};
struct NativeDirectGratingEndpointUpload
{
    SubmissionIdentity submission;
    std::uint64_t generation=0;
    // Other live adjuncts beyond Frame/GratingIncidentBinding reservations,
    // including retained query program evidence and earlier direct batches.
    std::uint64_t otherOwnedBytes=0,otherWork=0;
    std::vector<NativeDirectGratingEndpointInput> inputs; // Actual buffer readback
    bool complete=false;
};
struct NativeDirectGratingContributionSample
{
    enum class Outcome : std::uint32_t { Positive=0,Zero=1,Unsupported=2 };
    NativeDirectGratingEndpointInput input;
    NativeDirectionalSourceEcho source;
    opaquevisibility::ClosestHit endpoint,sourceHit;
    Point barycentric{};
    Texel sourceOriginMinimum{},sourceDirectionMaximum{};
    std::array<float,8> surfaceIrradiance{}; // W/(m^2 nm), includes cosine once
    std::uint64_t queryWork=0;
    Outcome outcome=Outcome::Unsupported;
    bool complete=false;
};
struct NativeDirectGratingContributionReadback
{
    std::uint32_t version=0;
    SubmissionIdentity submission;
    std::uint64_t endpointGeneration=0,sourceGeneration=0,queryWork=0,bandWork=0;
    std::vector<NativeDirectGratingContributionSample> samples;
    bool complete=false,overflow=false;
};
// Compare decoded fields with the retained native result, including exact float
// bits. Struct padding and host vector capacity are not result fields. This
// comparison alone cannot authenticate a native buffer or completed dispatch.
inline bool sameDirectGratingContributionReadback(const NativeDirectGratingContributionReadback& a,
    const NativeDirectGratingContributionReadback& b)
{
    const auto sameFloat=[](float x,float y) {
        std::uint32_t xBits=0,yBits=0;
        static_assert(sizeof(xBits)==sizeof(x));
        std::memcpy(&xBits,&x,sizeof(xBits));std::memcpy(&yBits,&y,sizeof(yBits));return xBits==yBits;
    };
    const auto sameFloats=[&](const auto& x,const auto& y) {
        for(std::size_t i=0;i<x.size();++i)if(!sameFloat(x[i],y[i]))return false;
        return true;
    };
    const auto sameHit=[&](const opaquevisibility::ClosestHit& x,const opaquevisibility::ClosestHit& y) {
        return std::tie(x.status,x.objectId,x.materialId,x.instanceId,x.triangleId,x.entering,x.work)
            ==std::tie(y.status,y.objectId,y.materialId,y.instanceId,y.triangleId,y.entering,y.work)
            &&sameFloat(x.distance,y.distance)&&sameFloats(x.position,y.position)&&sameFloats(x.outwardNormal,y.outwardNormal);
    };
    if(!(a.submission==b.submission)||a.samples.size()!=b.samples.size()
        ||std::tie(a.version,a.endpointGeneration,a.sourceGeneration,a.queryWork,a.bandWork,a.complete,a.overflow)
            !=std::tie(b.version,b.endpointGeneration,b.sourceGeneration,b.queryWork,b.bandWork,b.complete,b.overflow))return false;
    for(std::size_t i=0;i<a.samples.size();++i) {
        const auto& x=a.samples[i];const auto& y=b.samples[i];
        if(x.input.primitive!=y.input.primitive||x.input.control!=y.input.control
            ||!sameFloats(x.input.originMinimum,y.input.originMinimum)||!sameFloats(x.input.directionMaximum,y.input.directionMaximum)
            ||x.source.input.identifiers!=y.source.input.identifiers||!sameFloats(x.source.input.coefficients,y.source.input.coefficients)
            ||!sameFloats(x.source.rawRadiance,y.source.rawRadiance)||!sameFloats(x.source.directionTrace,y.source.directionTrace)
            ||!sameHit(x.endpoint,y.endpoint)||!sameHit(x.sourceHit,y.sourceHit)||!sameFloats(x.barycentric,y.barycentric)
            ||!sameFloats(x.sourceOriginMinimum,y.sourceOriginMinimum)||!sameFloats(x.sourceDirectionMaximum,y.sourceDirectionMaximum)
            ||!sameFloats(x.surfaceIrradiance,y.surfaceIrradiance)
            ||std::tie(x.queryWork,x.outcome,x.complete)!=std::tie(y.queryWork,y.outcome,y.complete))return false;
    }
    return true;
}
// Implemented only by the existing native frame. The query hook must validate
// the retained production sceneClosestHit program and this completed dispatch,
// including live context, the exact endpoint/source buffers and query textures,
// and actual aggregate work/storage admission. A program identity is insufficient.
// It must also compare every supplied readback field against its actual retained
// decoded native result using sameDirectGratingContributionReadback.
class NativeDirectGratingIrradianceBinding
{
public:
    virtual ~NativeDirectGratingIrradianceBinding()=default;
    virtual const NativeDirectionalSourceBinding& directSourceBinding() const noexcept=0;
    virtual const std::shared_ptr<const GratingIncidentBinding>& directGratingBinding() const noexcept=0;
    virtual const NativeDirectGratingEndpointUpload& directEndpointUpload() const noexcept=0;
    virtual bool validateDirectProductionQuery(const SubmissionIdentity&,
        const NativeDirectGratingContributionReadback&,std::string&) const=0;
protected:
    NativeDirectGratingIrradianceBinding()=default;
};
inline bool calculateDirectGratingContributionBounds(const GratingIncidentBinding& binding,
    std::size_t endpoints,std::uint64_t otherOwnedBytes,std::uint64_t otherWork,
    DirectGratingContributionBounds& output,std::string& error)
{
    GratingIncidentBounds original;
    if(!calculateGratingIncidentBounds(binding.frame,binding.triangles.size(),original,error))return false;
    const auto refuse=[&](){error="Direct grating contribution exceeds original whole-frame work or storage caps";return false;};
    if(!endpoints||endpoints>kMaximumEntrySamples||binding.bounds.combinedWork!=original.combinedWork
        ||binding.bounds.queryWork!=original.queryWork||binding.bounds.orderWork!=original.orderWork
        ||binding.bounds.bytes<original.bytes||binding.bounds.bytes>renderpassoutput::kMaximumTotalBytes-binding.frame.bounds.transientBytes)
        return refuse();
    if(otherWork>kMaximumQueryWork-binding.bounds.combinedWork
        ||otherOwnedBytes>renderpassoutput::kMaximumTotalBytes-binding.frame.bounds.transientBytes-binding.bounds.bytes)
        return refuse();
    const auto& allowance=binding.frame.query.texels[std::size_t(binding.frame.query.control[0])-1];
    const auto perQuery=std::uint64_t(allowance[0])+(std::uint64_t(allowance[1])<<16);
    DirectGratingContributionBounds candidate;
    candidate.maximumEndpoints=endpoints;candidate.queryWork=endpoints*2*perQuery;
    candidate.bandWork=endpoints*8;
    candidate.combinedWork=binding.bounds.combinedWork+otherWork+candidate.queryWork+candidate.bandWork;
    // Native inputs/output and their packed readbacks coexist with retained
    // decoded host copies. Reserve twice the host vector size, as elsewhere.
    candidate.bytes=endpoints*(2*sizeof(NativeDirectGratingEndpointInput)
        +2*kDirectGratingContributionNativeBytes+2*sizeof(NativeDirectGratingContributionSample)
        +2*sizeof(NativeDirectGratingEndpointInput))
        +sizeof(NativeDirectGratingEndpointUpload)+sizeof(NativeDirectGratingContributionReadback)
        +sizeof(NativeDirectGratingIrradianceBinding)
        +4*binding.identity.size();
    if(candidate.combinedWork>kMaximumQueryWork
        ||candidate.bandWork>kMaximumDepositionWork-binding.frame.bounds.deposition.depositionWork-binding.bounds.orderWork
        ||candidate.bytes>renderpassoutput::kMaximumTotalBytes-binding.frame.bounds.transientBytes-binding.bounds.bytes-otherOwnedBytes)
        return refuse();
    output=candidate;error.clear();return true;
}
inline bool validDirectGratingEndpointInput(const GratingIncidentBinding& binding,
    const NativeDirectGratingEndpointInput& input)
{
    const Point origin{input.originMinimum[0],input.originMinimum[1],input.originMinimum[2]};
    const Point direction{input.directionMaximum[0],input.directionMaximum[1],input.directionMaximum[2]};
    return input.control==std::array<std::uint32_t,4>{kDirectGratingContributionVersion,0,0,0}
        &&finite(origin)&&finite(direction)&&std::abs(dot(direction,direction)-1)<=1e-4f
        &&std::isfinite(input.originMinimum[3])&&input.originMinimum[3]>=kRayMinimum
        &&std::isfinite(input.directionMaximum[3])&&input.directionMaximum[3]>input.originMinimum[3]
        &&input.directionMaximum[3]<=binding.frame.sourceDirectionTrace[3]
        &&std::any_of(binding.triangles.begin(),binding.triangles.end(),[&](const auto& t) {
            return input.primitive==std::array<std::uint32_t,4>{t.object,t.material,t.instance,t.triangle};});
}
inline bool validateDirectGratingContributionReadback(const GratingIncidentBinding& binding,
    const NativeDirectGratingIrradianceBinding& native,const SubmissionIdentity& expected,
    double evaluationTime,const NativeDirectGratingContributionReadback& readback,std::string& error)
{
    const auto refuse=[&](const char* message){error=message;return false;};
    const auto& source=native.directSourceBinding();const auto& owner=source.owner();
    const auto& upload=native.directEndpointUpload();const auto& sourceUpload=source.upload();
    if(!binding.gratingOwner||!binding.material||native.directGratingBinding().get()!=&binding
        ||dynamic_cast<const NativeDirectGratingIrradianceBinding*>(&owner)!=&native
        ||!std::isfinite(evaluationTime)||evaluationTime!=binding.evaluationTime
        ||!validSourceSubmission(expected,binding.frame,&binding)
        ||!upload.complete||!upload.generation||upload.inputs.empty()
        ||!validSourceSubmission(upload.submission,binding.frame,&binding))
        return refuse("Direct grating contribution lost its exact endpoint, time or native frame owner");
    if(!validateDirectionalSourceUpload(binding.prepared,binding.frame,source,sourceUpload.submission,error,true,&binding))return false;
    for(const auto* submission:{&sourceUpload.submission,&upload.submission})
        if(submission->rendererGeneration!=expected.rendererGeneration||submission->rasterGeneration!=expected.rasterGeneration
            ||submission->queryGeneration!=expected.queryGeneration||submission->submissionSerial>expected.submissionSerial)
            return refuse("Direct grating contribution has stale source or endpoint upload generations");
    DirectGratingContributionBounds bounds;
    if(!calculateDirectGratingContributionBounds(binding,upload.inputs.size(),upload.otherOwnedBytes,upload.otherWork,bounds,error))return false;
    if(readback.version!=kDirectGratingContributionVersion||!readback.complete||readback.overflow
        ||!(readback.submission==expected)||readback.endpointGeneration!=upload.generation
        ||readback.sourceGeneration!=sourceUpload.bindingGeneration||readback.samples.size()!=upload.inputs.size()
        ||readback.samples.capacity()>2*upload.inputs.size()||upload.inputs.capacity()>2*upload.inputs.size()
        ||upload.submission.frameIdentity.capacity()>2*binding.identity.size()
        ||readback.submission.frameIdentity.capacity()>2*binding.identity.size()
        ||readback.queryWork>bounds.queryWork||readback.bandWork!=bounds.bandWork)
        return refuse("Direct grating contribution native completion, count or work changed");
    if(!native.validateDirectProductionQuery(expected,readback,error))return false;
    NativeDirectionalSourceEcho sourceEcho;sourceEcho.input=directionalSourceInput(binding.prepared,binding.frame);
    sourceEcho.directionTrace=binding.frame.sourceDirectionTrace;
    for(unsigned b=0;b<8;++b)sourceEcho.rawRadiance[b]=binding.frame.sourceRadiance[b/4][b%4];
    const auto& allowance=binding.frame.query.texels[std::size_t(binding.frame.query.control[0])-1];
    const auto perQuery=std::uint64_t(allowance[0])+(std::uint64_t(allowance[1])<<16);
    const Point toSource{sourceEcho.directionTrace[0],sourceEcho.directionTrace[1],sourceEcho.directionTrace[2]};
    std::uint64_t work=0;
    for(std::size_t index=0;index<readback.samples.size();++index) {
        const auto& sample=readback.samples[index];const auto& hit=sample.endpoint;
        if(!sample.complete||!validDirectGratingEndpointInput(binding,sample.input)
            ||std::memcmp(&sample.input,&upload.inputs[index],sizeof(sample.input))!=0
            ||std::memcmp(&sample.source,&sourceEcho,sizeof(sourceEcho))!=0
            ||hit.status!=1||!hit.work||hit.work>perQuery||!std::isfinite(hit.distance)
            ||hit.distance<sample.input.originMinimum[3]||hit.distance>sample.input.directionMaximum[3]
            ||sample.sourceHit.status<0||sample.sourceHit.status>1||!sample.sourceHit.work||sample.sourceHit.work>perQuery
            ||sample.queryWork!=std::uint64_t(hit.work)+sample.sourceHit.work)
            return refuse("Direct grating contribution endpoint or source query is incomplete");
        const auto triangle=std::find_if(binding.triangles.begin(),binding.triangles.end(),[&](const auto& t) {
            return sample.input.primitive==std::array<std::uint32_t,4>{t.object,t.material,t.instance,t.triangle};});
        if(sample.input.primitive!=std::array<std::uint32_t,4>{hit.objectId,hit.materialId,hit.instanceId,hit.triangleId}
            ||!sameGratingPoint(hit.outwardNormal,triangle->normal)||!finite(sample.barycentric)
            ||std::abs(sample.barycentric[0]+sample.barycentric[1]+sample.barycentric[2]-1)>1e-5f)
            return refuse("Direct grating contribution primitive or normal differs from admitted grating");
        Point position{};
        for(unsigned v=0;v<3;++v) {
            if(sample.barycentric[v]<=1e-5f||sample.barycentric[v]>=1-1e-5f)
                return refuse("Direct grating contribution refuses ambiguous primitive seams");
            position=advance(position,triangle->positions[v],sample.barycentric[v]);
        }
        const Point origin{sample.input.originMinimum[0],sample.input.originMinimum[1],sample.input.originMinimum[2]};
        Point direction{sample.input.directionMaximum[0],sample.input.directionMaximum[1],sample.input.directionMaximum[2]};
        if(!normalize(direction))return refuse("Direct grating contribution endpoint direction is invalid");
        if(!sameGratingPoint(position,hit.position,8*bias(hit.position))
            ||!sameGratingPoint(advance(origin,direction,hit.distance),hit.position,8*bias(hit.position))
            ||hit.entering!=(dot(direction,triangle->normal)<0)
            ||sample.sourceOriginMinimum!=Texel{hit.position[0],hit.position[1],hit.position[2],kRayMinimum}
            ||sample.sourceDirectionMaximum!=sourceEcho.directionTrace)
            return refuse("Direct grating contribution actual endpoint-to-source ray differs from its owners");
        const auto& shadow=sample.sourceHit;
        if(!std::isfinite(shadow.distance)||shadow.distance<kRayMinimum||shadow.distance>sourceEcho.directionTrace[3])
            return refuse("Direct grating contribution source query distance is invalid");
        if(shadow.status==0) {
            if(shadow.distance!=sourceEcho.directionTrace[3]||shadow.position!=Point{}||shadow.outwardNormal!=Point{}
                ||shadow.objectId||shadow.materialId||shadow.instanceId||shadow.triangleId||shadow.entering)
                return refuse("Direct grating contribution clear source query has nonempty hit data");
        } else {
            const auto blocker=std::find_if(binding.frame.triangles.begin(),binding.frame.triangles.end(),[&](const auto& t) {
                return std::tie(t.object,t.material,t.instance,t.triangle)==std::tie(shadow.objectId,shadow.materialId,shadow.instanceId,shadow.triangleId);});
            if(blocker==binding.frame.triangles.end()||!sameGratingPoint(shadow.outwardNormal,blocker->normal)
                ||!sameGratingPoint(advance(hit.position,toSource,shadow.distance),shadow.position,8*bias(shadow.position))
                ||shadow.entering!=(dot(toSource,blocker->normal)<0))
                return refuse("Direct grating contribution blocker is outside the production query scene");
            // Check the returned primitive witness, without tracing visibility
            // or deciding which scene triangle the native query should hit.
            const auto e1=sub(blocker->points[1],blocker->points[0]),e2=sub(blocker->points[2],blocker->points[0]);
            const auto relative=sub(shadow.position,blocker->points[0]);
            const float a=dot(e1,e1),b=dot(e1,e2),c=dot(e2,e2),d=dot(relative,e1),e=dot(relative,e2),det=a*c-b*b;
            if(!std::isfinite(det)||det<=0)return refuse("Direct grating contribution blocker primitive is degenerate");
            const float u=(c*d-b*e)/det,v=(a*e-b*d)/det;
            if(!std::isfinite(u)||!std::isfinite(v)||u<-1e-5f||v<-1e-5f||u+v>1+1e-5f
                ||std::abs(dot(relative,blocker->normal))>8*bias(shadow.position))
                return refuse("Direct grating contribution blocker position is outside its returned primitive");
        }
        const float cosine=std::max(0.0f,dot(triangle->normal,toSource));bool positive=false;
        for(unsigned band=0;band<8;++band) {
            const float irradiance=shadow.status==0?sourceEcho.input.coefficients[band]*cosine:0;
            if(!nearDirectionalSourcePotential(sample.surfaceIrradiance[band],irradiance))
                return refuse("Direct grating contribution does not equal visible authored D times incident cosine");
            positive|=irradiance>0;
        }
        using Outcome=NativeDirectGratingContributionSample::Outcome;
        if(sample.outcome!=(positive?Outcome::Positive:Outcome::Zero))
            return refuse("Direct grating contribution confuses supported zero with unsupported evaluation");
        work+=sample.queryWork;
    }
    if(work!=readback.queryWork)return refuse("Direct grating contribution whole-frame query work changed");
    error.clear();return true;
}
inline bool sameDirectGratingEndpointInput(const NativeDirectGratingEndpointInput& a,
    const NativeDirectGratingEndpointInput& b)
{
    return a.primitive==b.primitive&&a.control==b.control
        &&std::memcmp(a.originMinimum.data(),b.originMinimum.data(),sizeof(a.originMinimum))==0
        &&std::memcmp(a.directionMaximum.data(),b.directionMaximum.data(),sizeof(a.directionMaximum))==0;
}
struct DirectGratingIrradianceCohortBounds
{
    std::uint64_t bytes=0,combinedWork=0,queryWork=0,bandWork=0,depositionWork=0;
};
class DirectGratingIrradianceCohort;
struct DirectGratingEndpointIrradianceSum
{
    NativeDirectGratingEndpointInput input;
    opaquevisibility::ClosestHit endpoint;
    Point barycentric{};
    std::array<float,8> surfaceIrradiance{}; // W/(m^2 nm), no further cosine
};
struct DirectGratingIrradianceCohortReadback
{
    std::shared_ptr<const DirectGratingIrradianceCohort> cohort;
    SubmissionIdentity submission;
    diffractionmaterial::DirectionalSpectralMeasure sourceMeasure=diffractionmaterial::DirectionalSpectralMeasure::IrradianceCoefficient;
    std::vector<DirectGratingEndpointIrradianceSum> endpoints;
};
inline bool prepareDirectGratingIrradianceCohort(
    const std::vector<std::shared_ptr<const GratingIncidentBinding>>&,
    const std::vector<NativeDirectGratingEndpointInput>&,std::uint64_t,std::uint64_t,
    std::shared_ptr<const DirectGratingIrradianceCohort>&,std::string&);
// Ephemeral immutable preparation only. No native completion or Color grant.
class DirectGratingIrradianceCohort
{
public:
    const auto& sources() const noexcept{return sources_;}
    const auto& inputs() const noexcept{return inputs_;}
    const auto& identity() const noexcept{return identity_;}
    const auto& bounds() const noexcept{return bounds_;}
private:
    DirectGratingIrradianceCohort()=default;
    std::vector<std::shared_ptr<const GratingIncidentBinding>> sources_;
    std::vector<NativeDirectGratingEndpointInput> inputs_;
    std::string identity_;
    DirectGratingIrradianceCohortBounds bounds_;
    friend bool prepareDirectGratingIrradianceCohort(
        const std::vector<std::shared_ptr<const GratingIncidentBinding>>&,
        const std::vector<NativeDirectGratingEndpointInput>&,std::uint64_t,std::uint64_t,
        std::shared_ptr<const DirectGratingIrradianceCohort>&,std::string&);
};
inline bool directGratingCommonIdentity(const GratingIncidentBinding& binding,std::string& output,std::string& error)
{
    Prepared common;
    if(!prepareScene(binding.prepared.glass,"direct-D8-cohort-v1:",common,error))return false;
    const auto scene=sceneaov::detail::encodeScene(*common.glass.sourceScene);
    if(scene.empty()){error="Directional cohort common scene is invalid";return false;}
    std::string candidate=common.identity;word(candidate,scene.size());candidate+=sceneaov::detail::hex(scene);
    word(candidate,std::uint32_t(common.glass.sourceMaterial->backend));word(candidate,common.glass.sourceMaterial->layoutVersion);
    word(candidate,common.glass.object.value);word(candidate,binding.gratingOwner->object.value);
    word(candidate,binding.gratingOwner->authoredMaterial.value);
    for(auto revision:{binding.gratingOwner->sceneRevision,binding.gratingOwner->structuralRevision,
        binding.gratingOwner->evaluationRevision,binding.gratingOwner->materialRevision})word(candidate,revision);
    for(auto id:binding.gratingOwner->parameters.identifiers)word(candidate,id);
    for(const auto* block:{&binding.gratingOwner->parameters.baseColorMetallic,&binding.gratingOwner->parameters.emissionRoughness,
        &binding.gratingOwner->parameters.normalOpacity,&binding.gratingOwner->parameters.transmissionIorClearcoat})
        for(float value:*block)bits(candidate,value);
    for(const auto* id:{&binding.material->structuralDigest(),&binding.gratingOwner->bindingDigest,
        &binding.gratingOwner->programIdentity,&binding.frame.cameraIdentity,&binding.frame.branchPolicyIdentity})
    {word(candidate,id->size());candidate+=*id;}
    std::uint64_t time=0;std::memcpy(&time,&binding.evaluationTime,sizeof(time));word(candidate,time);
    word(candidate,binding.frame.width);word(candidate,binding.frame.height);word(candidate,binding.frame.query.height);
    word(candidate,binding.frame.scheduling.version);word(candidate,binding.frame.scheduling.entrySubdivisions);
    word(candidate,binding.frame.interfaceSamples);bits(candidate,binding.frame.roughness);word(candidate,binding.frame.gratingStaging);
    for(float value:binding.frame.query.control)bits(candidate,value);
    for(const auto* table:{&binding.frame.query.texels,&binding.frame.triangleTexels,&binding.witnessTexels}) {
        word(candidate,table->size());for(const auto& texel:*table)for(float value:texel)bits(candidate,value);
    }
    for(float value:binding.frame.ior)bits(candidate,value);
    output=std::move(candidate);return true;
}
inline bool prepareDirectGratingIrradianceCohort(
    const std::vector<std::shared_ptr<const GratingIncidentBinding>>& sources,
    const std::vector<NativeDirectGratingEndpointInput>& inputs,std::uint64_t otherOwnedBytes,std::uint64_t otherWork,
    std::shared_ptr<const DirectGratingIrradianceCohort>& output,std::string& error)
{
    const auto refuse=[&](const char* message){error=message;return false;};
    if(sources.empty()||sources.size()>HarmonicMIDI::grid::Visual3DScene::kMaxLights
        ||!sources.front()||inputs.empty()||inputs.size()>kMaximumEntrySamples
        ||otherOwnedBytes>renderpassoutput::kMaximumTotalBytes||otherWork>kMaximumQueryWork)
        return refuse("Directional cohort source or endpoint count is unsupported");
    std::array<HarmonicMIDI::grid::SceneLightId,HarmonicMIDI::grid::Visual3DScene::kMaxLights> roster{};
    std::size_t count=0;if(!directionalSourceRoster(sources.front()->prepared,roster,count,error))return false;
    if(count!=sources.size())return refuse("Directional cohort omits or adds an admitted scene D8 source");
    DirectGratingIrradianceCohortBounds bounds;bounds.bytes=otherOwnedBytes;bounds.combinedWork=otherWork;
    bounds.depositionWork=otherWork; // Conservatively charge unknown adjunct work to both caps.
    for(std::size_t i=0;i<count;++i) {
        if(!sources[i]||sources[i]->prepared.sceneLight!=roster[i])
            return refuse("Directional cohort source roster order or identity changed");
        const auto& binding=*sources[i];DirectGratingContributionBounds batch;
        if(!calculateDirectGratingContributionBounds(binding,inputs.size(),0,0,batch,error))return false;
        // Reserve every retained source owner/table/batch and a validation copy.
        bounds.bytes+=2*(binding.frame.bounds.transientBytes+binding.bounds.bytes)+batch.bytes
            +2*sizeof(NativeDirectionalSourceUpload)+4*sizeof(NativeDirectionalSourceInput)
            +2*sizeof(NativeDirectionalSourceEcho)+4*binding.identity.size();
        bounds.combinedWork+=batch.combinedWork;bounds.queryWork+=batch.queryWork;bounds.bandWork+=batch.bandWork;
        bounds.depositionWork+=binding.frame.bounds.deposition.depositionWork+binding.bounds.orderWork+batch.bandWork;
        if(bounds.bytes>renderpassoutput::kMaximumTotalBytes||bounds.combinedWork>kMaximumQueryWork
            ||bounds.depositionWork>kMaximumDepositionWork)return refuse("Directional cohort exceeds cumulative frame bounds");
    }
    auto candidate=std::shared_ptr<DirectGratingIrradianceCohort>(new DirectGratingIrradianceCohort);
    std::string commonIdentity;
    for(std::size_t i=0;i<count;++i) {
        const auto& binding=*sources[i];std::shared_ptr<const GratingIncidentBinding> exact;Prepared prepared;
        if(!binding.prepared.branchPolicy||binding.frame.gratingIncidentBinding)
            return refuse("Directional cohort requires the original source policy and unmarked staging frame");
        if(!prepare(binding.prepared.glass,binding.prepared.sceneLight,*binding.prepared.branchPolicy,prepared,error))return false;
        auto frame=binding.frame;
        if(frame.activeSource)frame.activeSource=std::make_shared<const HarmonicMIDI::grid::SceneLightRecord>(*frame.activeSource);
        if(!prepareGratingIncidentBinding(prepared,frame,binding.gratingOwner,binding.material,
            binding.evaluationTime,exact,error,SourceMeasureUse::DirectionalIrradiancePreparation))return false;
        if(binding.identity!=exact->identity||binding.frame.sourceDirectionalIrradiance.version!=1
            ||binding.bounds.bytes!=exact->bounds.bytes||binding.bounds.combinedWork!=exact->bounds.combinedWork
            ||binding.bounds.queryWork!=exact->bounds.queryWork||binding.bounds.orderWork!=exact->bounds.orderWork
            ||binding.bounds.maximumIncidents!=exact->bounds.maximumIncidents||binding.bounds.maximumOrders!=exact->bounds.maximumOrders)
            return refuse("Directional cohort source binding is stale or lacks typed D8");
        std::string identity;if(!directGratingCommonIdentity(binding,identity,error))return false;
        if(!i)commonIdentity=std::move(identity);
        else if(identity!=commonIdentity)return refuse("Directional cohort scene, frame, material or time differs");
        for(const auto& input:inputs)if(!validDirectGratingEndpointInput(binding,input))
            return refuse("Directional cohort endpoint is outside a source binding");
        candidate->sources_.push_back(std::move(exact));
    }
    candidate->identity_=std::move(commonIdentity);word(candidate->identity_,count);
    for(const auto& binding:sources){word(candidate->identity_,binding->identity.size());candidate->identity_+=binding->identity;}
    word(candidate->identity_,inputs.size());
    for(const auto& input:inputs) {
        for(auto value:input.primitive)word(candidate->identity_,value);
        for(auto value:input.control)word(candidate->identity_,value);
        for(float value:input.originMinimum)bits(candidate->identity_,value);
        for(float value:input.directionMaximum)bits(candidate->identity_,value);
    }
    bounds.bytes+=sizeof(DirectGratingIrradianceCohort)+2*count*sizeof(sources.front())+2*candidate->identity_.size()
        +2*inputs.size()*(sizeof(NativeDirectGratingEndpointInput)+sizeof(DirectGratingEndpointIrradianceSum))
        +2*sizeof(DirectGratingIrradianceCohortReadback)+2*candidate->identity_.size();
    if(bounds.bytes>renderpassoutput::kMaximumTotalBytes)return refuse("Directional cohort retained sum storage exceeds frame bounds");
    candidate->bounds_=bounds;candidate->inputs_=inputs;
    output=std::move(candidate);error.clear();return true;
}
// A backend must implement this additional inspector. Existing single-source
// heads do not authenticate a cohort. Per-source views share the actual native
// scene/cache; the inspector must authenticate that common frame, immutable
// source buffers and all retained dispatches, live leases/fences and cumulative
// reservations including failed attempts. It must inspect actual resources,
// not accept caller completeness flags or repeated head access as a source set.
struct DirectGratingTerminalNativeBudget
{
    std::uint64_t retainedBytes=0,combinedWork=0,depositionWork=0;
};
struct DirectGratingTerminalOrderBounds;
struct DirectGratingTerminalOrderLedger;
class NativeDirectGratingTerminalResult;
struct DirectGratingTerminalReceiverNativeBudget;
struct DirectGratingTerminalReceiverBounds;
struct DirectGratingTerminalReceiverReadback;
struct NativeDirectGratingTerminalReceiverInput;
class NativeDirectGratingTerminalReceiverResult;
class NativeDirectGratingTerminalReceiverLease;
inline bool evaluateDirectGratingTerminalOrders(const std::shared_ptr<const DirectGratingIrradianceCohort>&,
    const NativeFixtureSceneFrame&,const SubmissionIdentity&,double,
    const std::vector<NativeDirectGratingContributionReadback>&,DirectGratingTerminalOrderLedger&,std::string&);
class NativeDirectGratingTerminalLease
{
public:
    virtual ~NativeDirectGratingTerminalLease()=default;
    virtual const NativeFixtureSceneFrame& owner() const noexcept=0;
protected:
    NativeDirectGratingTerminalLease()=default;
    NativeDirectGratingTerminalLease(const NativeDirectGratingTerminalLease&)=delete;
    NativeDirectGratingTerminalLease& operator=(const NativeDirectGratingTerminalLease&)=delete;
    // Only the shared operation commits after the native owner has validated
    // its registered staged result. No allocation or failure may occur here.
    virtual void commit() noexcept=0;
private:
    friend bool evaluateDirectGratingTerminalOrders(const std::shared_ptr<const DirectGratingIrradianceCohort>&,
        const NativeFixtureSceneFrame&,const SubmissionIdentity&,double,
        const std::vector<NativeDirectGratingContributionReadback>&,DirectGratingTerminalOrderLedger&,std::string&);
};
class NativeDirectGratingIrradianceCohortBinding
{
public:
    virtual ~NativeDirectGratingIrradianceCohortBinding()=default;
    virtual const NativeFixtureSceneFrame& cohortOwner() const noexcept=0;
    virtual const std::shared_ptr<const DirectGratingIrradianceCohort>& directCohort() const noexcept=0;
    virtual std::size_t directSourceCount() const noexcept=0;
    virtual const NativeDirectGratingIrradianceBinding* directSourceBatch(std::size_t) const noexcept=0;
    virtual bool validateDirectCohortProductionQueries(const SubmissionIdentity&,
        const std::vector<NativeDirectGratingContributionReadback>&,std::string&) const=0;
    // Register a unique operation under the actual frame/context/cohort/fence
    // guard. Reconstruct whole-frame host/native ownership and original/failed
    // work, including retained terminal results. Keep the guard until lease
    // destruction. A caller token or an inspector-only total is insufficient.
    virtual bool beginDirectCohortTerminalLease(const SubmissionIdentity&,double,
        const std::vector<NativeDirectGratingContributionReadback>&,
        std::unique_ptr<NativeDirectGratingTerminalLease>&,DirectGratingTerminalNativeBudget&,std::string& error) const
    {error="Directional terminal native lease is unsupported";return false;}
    // Recheck this exact registered lease and the reconstructed combined
    // ledger, reserve full additional bytes and charge worst band/order work
    // before allocation/accumulation. Admitted work survives every failure.
    virtual bool reserveDirectCohortTerminalLease(NativeDirectGratingTerminalLease&,
        const DirectGratingTerminalOrderBounds&,const DirectGratingTerminalOrderLedger&,std::string& error) const
    {error="Directional terminal native reservation is unsupported";return false;}
    // Stage an independently copied immutable result under this frame and
    // lease. Reconcile actual retained/returned capacities with reservation.
    // Do not replace the incumbent frame result until lease.commit(). On every
    // uncommitted exit the lease releases staged storage/bytes, preserves the
    // old frame result, and retains the full admitted attempt work charge.
    virtual bool finishDirectCohortTerminalLease(NativeDirectGratingTerminalLease&,
        const DirectGratingTerminalOrderLedger&,std::shared_ptr<const NativeDirectGratingTerminalResult>&,
        std::string& error) const
    {error="Directional terminal native finish is unsupported";return false;}
    // Require the real registered staged owner, exact context/cohort/fences,
    // immutable payload and actual capacity/byte/work ledger. This callback
    // is mandatory after finish; neither a value nor a lease grants completion.
    virtual bool validateDirectCohortTerminalResult(const NativeDirectGratingTerminalLease&,
        const DirectGratingTerminalOrderLedger&,std::string& error) const
    {error="Directional terminal native result inspection is unsupported";return false;}
    // An additive operation on the exact live scattering result. Authenticate
    // its registered owner, frozen frame/query/raster buffers, time, generations,
    // cohort and fences under the native guard. Reconstruct retained bytes and
    // all admitted work, including failed attempts and still-live old results.
    virtual bool beginDirectTerminalReceiverLease(const std::shared_ptr<const NativeDirectGratingTerminalResult>&,
        const SubmissionIdentity&,double,std::unique_ptr<NativeDirectGratingTerminalReceiverLease>&,
        DirectGratingTerminalReceiverNativeBudget&,std::string& error) const
    {error="Directional terminal native receiver lease is unsupported";return false;}
    // Register and charge the full conservative query allowance before any
    // input/result/resource allocation or native dispatch. Recheck the exact
    // incumbent and actual frame ledger; failures retain every admitted unit.
    virtual bool reserveDirectTerminalReceiverLease(NativeDirectGratingTerminalReceiverLease&,
        const DirectGratingTerminalReceiverBounds&,const DirectGratingTerminalReceiverReadback&,
        std::string& error) const
    {error="Directional terminal native receiver reservation is unsupported";return false;}
    // Query every emitted order, including zero-power orders, through the
    // existing native closest-hit authority. Read the actual raster owner and
    // depth at each projected pixel. No CPU traversal or visibility substitute.
    virtual bool dispatchDirectTerminalReceivers(NativeDirectGratingTerminalReceiverLease&,
        const std::vector<NativeDirectGratingTerminalReceiverInput>&,
        DirectGratingTerminalReceiverReadback&,std::string& error) const
    {error="Directional terminal native receiver dispatch is unsupported";return false;}
    // Stage an independent immutable copy. Do not replace the frame's receiver
    // result until commit. Uncommitted destruction releases staged bytes only,
    // preserves prior receiver/scattering results and keeps failed work charged.
    virtual bool finishDirectTerminalReceiverLease(NativeDirectGratingTerminalReceiverLease&,
        const DirectGratingTerminalReceiverReadback&,
        std::shared_ptr<const NativeDirectGratingTerminalReceiverResult>&,std::string& error) const
    {error="Directional terminal native receiver finish is unsupported";return false;}
    // Inspect actual registered dispatch/resources and immutable rows, exact
    // terminal identity, frozen camera/query/raster, submission/time/generations,
    // input generation and attempt ledger. Neither pointers nor caller status
    // flags authenticate native production. Required before and after staging.
    virtual bool validateDirectTerminalReceiverResult(const NativeDirectGratingTerminalReceiverLease&,
        const DirectGratingTerminalReceiverReadback&,std::string& error) const
    {error="Directional terminal native receiver inspection is unsupported";return false;}
protected:
    NativeDirectGratingIrradianceCohortBinding()=default;
    std::shared_ptr<const NativeDirectGratingTerminalResult> makeDirectCohortTerminalResult(
        const DirectGratingTerminalOrderLedger&) const;
    std::shared_ptr<const NativeDirectGratingTerminalReceiverResult> makeDirectTerminalReceiverResult(
        const DirectGratingTerminalReceiverReadback&) const;
};
inline bool accumulateDirectGratingIrradianceCohort(const std::shared_ptr<const DirectGratingIrradianceCohort>& cohort,
    const NativeFixtureSceneFrame& owner,const SubmissionIdentity& expected,double evaluationTime,
    const std::vector<NativeDirectGratingContributionReadback>& readbacks,
    DirectGratingIrradianceCohortReadback& output,std::string& error)
{
    const auto refuse=[&](const char* message){error=message;return false;};
    const auto* native=dynamic_cast<const NativeDirectGratingIrradianceCohortBinding*>(&owner);
    if(!cohort||!native||&native->cohortOwner()!=&owner||native->directCohort()!=cohort
        ||!owner.nativeResourceCacheIdentity()||native->directSourceCount()!=cohort->sources().size()
        ||readbacks.size()!=cohort->sources().size()||readbacks.capacity()>2*readbacks.size()
        ||expected.frameIdentity!=cohort->identity()||!expected.rendererGeneration||!expected.rasterGeneration
        ||!expected.queryGeneration||!expected.scratchGeneration||!expected.submissionSerial
        ||expected.completedFence!=expected.submissionSerial
        ||expected.frameIdentity.capacity()>2*cohort->identity().size())
        return refuse("Directional cohort has no complete common native frame inspector");
    std::array<const NativeDirectGratingIrradianceBinding*,HarmonicMIDI::grid::Visual3DScene::kMaxLights> seen{};
    for(std::size_t i=0;i<readbacks.size();++i) {
        const auto* batch=native->directSourceBatch(i);const auto& source=*cohort->sources()[i];const auto& receipt=readbacks[i];
        if(!batch||std::find(seen.begin(),seen.begin()+i,batch)!=seen.begin()+i)
            return refuse("Directional cohort repeats or omits a retained source batch");
        seen[i]=batch;
        if(!batch||batch->directGratingBinding()!=cohort->sources()[i]||receipt.samples.size()!=cohort->inputs().size()
            ||batch->directSourceBinding().owner().nativeResourceCacheIdentity()!=owner.nativeResourceCacheIdentity()
            ||owner.width()!=source.frame.width||owner.height()!=source.frame.height
            ||receipt.submission.rendererGeneration!=expected.rendererGeneration
            ||receipt.submission.rasterGeneration!=expected.rasterGeneration||receipt.submission.queryGeneration!=expected.queryGeneration
            ||receipt.submission.submissionSerial>expected.submissionSerial)
            return refuse("Directional cohort source batch is missing, stale or not exact native output");
        if(!validateDirectGratingContributionReadback(source,*batch,receipt.submission,evaluationTime,receipt,error))return false;
        for(std::size_t endpoint=0;endpoint<cohort->inputs().size();++endpoint) {
            const auto& sample=receipt.samples[endpoint];
            if(!sameDirectGratingEndpointInput(sample.input,cohort->inputs()[endpoint]))
                return refuse("Directional cohort ordered endpoint set changed");
            if(i) {
                const auto& first=readbacks[0].samples[endpoint];const auto& a=sample.endpoint;const auto& b=first.endpoint;
                if(std::tie(a.status,a.objectId,a.materialId,a.instanceId,a.triangleId,a.entering,a.work)
                    !=std::tie(b.status,b.objectId,b.materialId,b.instanceId,b.triangleId,b.entering,b.work)
                    ||std::memcmp(&a.distance,&b.distance,sizeof(float))!=0
                    ||std::memcmp(a.position.data(),b.position.data(),sizeof(Point))!=0
                    ||std::memcmp(a.outwardNormal.data(),b.outwardNormal.data(),sizeof(Point))!=0
                    ||std::memcmp(sample.barycentric.data(),first.barycentric.data(),sizeof(Point))!=0)
                    return refuse("Directional cohort endpoint primitive witness differs between sources");
            }
        }
    }
    if(!native->validateDirectCohortProductionQueries(expected,readbacks,error))return false;
    DirectGratingIrradianceCohortReadback candidate;candidate.cohort=cohort;candidate.submission=expected;
    candidate.endpoints.resize(cohort->inputs().size());
    for(std::size_t e=0;e<candidate.endpoints.size();++e) {
        auto& endpoint=candidate.endpoints[e];endpoint.input=cohort->inputs()[e];
        endpoint.endpoint=readbacks[0].samples[e].endpoint;endpoint.barycentric=readbacks[0].samples[e].barycentric;
        for(unsigned band=0;band<8;++band) {
            double sum=0;
            for(const auto& receipt:readbacks) {
                const float value=receipt.samples[e].surfaceIrradiance[band];
                if(!std::isfinite(value)||value<0)return refuse("Directional cohort physical contribution is nonfinite or negative");
                sum+=value;
            }
            if(!std::isfinite(sum)||sum>std::numeric_limits<float>::max())return refuse("Directional cohort irradiance sum overflows");
            endpoint.surfaceIrradiance[band]=float(sum);
            if(sum>0&&endpoint.surfaceIrradiance[band]<=0)return refuse("Directional cohort positive irradiance sum underflows");
        }
    }
    output=std::move(candidate);error.clear();return true;
}
// Private sampled camera branches only. Neither a complete dispatch nor a
// complete finite trace establishes a refracted pixel domain or terminal light.
inline constexpr std::uint32_t kGlassCameraGratingTraceVersion=1;
inline constexpr std::uint32_t kGlassCameraTraceRoot=std::numeric_limits<std::uint32_t>::max();
inline constexpr std::size_t kGlassCameraTraceHeaderLanes=10,kGlassCameraTraceVisitLanes=16;
using GlassCameraTraceWords=std::array<std::uint32_t,4>;
struct alignas(16) NativeGlassCameraGratingTraceInput
{
    // Top-first pixel x/y; location0 center,1 TL,2 TR,3 BR,4 BL; version.
    GlassCameraTraceWords pixel{},control{}; // width,height,0,0
    Texel originMinimum{},directionMaximum{};
};
static_assert(sizeof(NativeGlassCameraGratingTraceInput)==64);
static_assert(offsetof(NativeGlassCameraGratingTraceInput,pixel)==0&&offsetof(NativeGlassCameraGratingTraceInput,control)==16
    &&offsetof(NativeGlassCameraGratingTraceInput,originMinimum)==32&&offsetof(NativeGlassCameraGratingTraceInput,directionMaximum)==48);
enum class GlassCameraTraceDisposition : std::uint32_t { Interface=0,Grating=1,Other=2,Escape=3,Unresolved=4 };
enum GlassCameraTraceReason : std::uint32_t {
    GlassCameraQueryFailed=1,GlassCameraTopologyFailed=2,GlassCameraInterfaceCap=4,
    GlassCameraPendingCap=8,GlassCameraVisitCap=16,GlassCameraFractionUnderflow=32,
    GlassCameraOpticsFailed=64,GlassCameraStorageCap=128
};
struct NativeGlassCameraGratingTraceVisit
{
    std::uint32_t visit=0,parent=kGlassCameraTraceRoot,token=0,interfaces=0,history=0;
    bool inside=false,tir=false;
    GlassCameraTraceDisposition disposition=GlassCameraTraceDisposition::Unresolved;
    Texel originMinimum{},directionFraction{};
    opaquevisibility::ClosestHit hit;
    Point orientedNormal{},transmitted{},reflected{},barycentric{};
    float reflection=0,transmittedFraction=0,reflectedFraction=0,ior=0,fromIor=0,toIor=0;
    std::array<std::uint8_t,32> ancestry{}; // token1 reflection,65 transmission
};
struct NativeGlassCameraGratingTraceBand
{
    NativeGlassCameraGratingTraceInput input; // Actual native input-buffer echo
    Texel nativeOriginMinimum{},nativeDirectionMaximum{}; // Native camera derivation
    std::uint32_t version=0,sample=0,band=0,reasons=0;
    float ior=0,wavelength=0,unresolvedFraction=0;
    std::uint64_t queryWork=0;
    std::vector<NativeGlassCameraGratingTraceVisit> visits;
    // Unknown uses canonical numeric zero, not a zero-fraction witness.
    bool traceComplete=false,unresolvedFractionKnown=false;
};
struct NativeGlassCameraGratingTraceReadback
{
    std::uint32_t version=0;
    SubmissionIdentity submission;
    std::uint64_t inputGeneration=0,queryWork=0;
    std::vector<NativeGlassCameraGratingTraceBand> bands; // input-major, eight bands
    bool dispatchComplete=false,overflow=false;
};
struct NativeGlassCameraGratingTraceUpload
{
    NativeFixtureMaterialBackend backend=NativeFixtureMaterialBackend::OpenGl;
    SubmissionIdentity submission;
    std::uint64_t generation=0,otherOwnedBytes=0,otherWork=0;
    std::vector<NativeGlassCameraGratingTraceInput> inputs;
    bool complete=false;
};
// The GPU writes native uint4 lanes, with float payloads stored by bit cast.
// Header0..3 input;4..5 native ray;6 version/sample/band/count;
// 7 traceComplete/reasons/queryLo32/queryHi32;8 ior/wavelength/unresolved/known;
// lane8.w is a native uint0/1, not float bits. Header9 is zero.
// Visit0 visit/parent/token/interfaces;1 inside/history/disposition/tir;
// 2 origin/min;3 direction/fraction;4 status/object/material/instance;
// 5 triangle/work/entering/0;6 position/distance;7 outwardNormal/0;
// 8 orientedNormal/Fresnel;9 transmitted/transmittedFraction;
// 10 reflected/reflectedFraction;11 barycentric/ior;12..13 packed ancestry;
// 14 fromIor/toIor/0/0;15 zero. Unused fixed visit slots are zero.
using NativeGlassCameraGratingTracePackedHeader=std::array<GlassCameraTraceWords,kGlassCameraTraceHeaderLanes>;
using NativeGlassCameraGratingTracePackedVisit=std::array<GlassCameraTraceWords,kGlassCameraTraceVisitLanes>;
inline constexpr std::size_t kGlassCameraTraceHeaderBytes=sizeof(NativeGlassCameraGratingTracePackedHeader);
inline constexpr std::size_t kGlassCameraTraceVisitBytes=sizeof(NativeGlassCameraGratingTracePackedVisit);
static_assert(kGlassCameraTraceHeaderBytes==160&&kGlassCameraTraceVisitBytes==256);
inline std::uint32_t glassCameraTraceFloatBits(float value)
{std::uint32_t bits=0;static_assert(sizeof(bits)==sizeof(value));std::memcpy(&bits,&value,sizeof(bits));return bits;}
inline float glassCameraTraceFloat(std::uint32_t bits)
{float value=0;std::memcpy(&value,&bits,sizeof(value));return value;}
template<std::size_t N> inline bool sameGlassCameraTraceFloats(const std::array<float,N>& a,const std::array<float,N>& b)
{for(std::size_t i=0;i<N;++i){if(glassCameraTraceFloatBits(a[i])!=glassCameraTraceFloatBits(b[i]))return false;}return true;}
inline bool sameGlassCameraGratingTraceInput(const NativeGlassCameraGratingTraceInput& a,const NativeGlassCameraGratingTraceInput& b)
{return a.pixel==b.pixel&&a.control==b.control&&sameGlassCameraTraceFloats(a.originMinimum,b.originMinimum)
    &&sameGlassCameraTraceFloats(a.directionMaximum,b.directionMaximum);}
inline bool sameGlassCameraGratingTraceReadback(const NativeGlassCameraGratingTraceReadback& a,const NativeGlassCameraGratingTraceReadback& b)
{
    if(!(a.submission==b.submission)||a.bands.size()!=b.bands.size()
        ||std::tie(a.version,a.inputGeneration,a.queryWork,a.dispatchComplete,a.overflow)
            !=std::tie(b.version,b.inputGeneration,b.queryWork,b.dispatchComplete,b.overflow))return false;
    for(std::size_t i=0;i<a.bands.size();++i) {
        const auto& x=a.bands[i];const auto& y=b.bands[i];
        if(!sameGlassCameraGratingTraceInput(x.input,y.input)||!sameGlassCameraTraceFloats(x.nativeOriginMinimum,y.nativeOriginMinimum)
            ||!sameGlassCameraTraceFloats(x.nativeDirectionMaximum,y.nativeDirectionMaximum)
            ||std::tie(x.version,x.sample,x.band,x.reasons,x.queryWork,x.traceComplete,x.unresolvedFractionKnown)
                !=std::tie(y.version,y.sample,y.band,y.reasons,y.queryWork,y.traceComplete,y.unresolvedFractionKnown)
            ||!sameGlassCameraTraceFloats(std::array{x.ior,x.wavelength,x.unresolvedFraction},std::array{y.ior,y.wavelength,y.unresolvedFraction})
            ||x.visits.size()!=y.visits.size())return false;
        for(std::size_t v=0;v<x.visits.size();++v) {
            const auto& p=x.visits[v];const auto& q=y.visits[v];const auto& h=p.hit;const auto& k=q.hit;
            if(std::tie(p.visit,p.parent,p.token,p.interfaces,p.history,p.inside,p.tir,p.disposition,p.ancestry)
                !=std::tie(q.visit,q.parent,q.token,q.interfaces,q.history,q.inside,q.tir,q.disposition,q.ancestry)
                ||!sameGlassCameraTraceFloats(p.originMinimum,q.originMinimum)||!sameGlassCameraTraceFloats(p.directionFraction,q.directionFraction)
                ||std::tie(h.status,h.objectId,h.materialId,h.instanceId,h.triangleId,h.entering,h.work)
                    !=std::tie(k.status,k.objectId,k.materialId,k.instanceId,k.triangleId,k.entering,k.work)
                ||!sameGlassCameraTraceFloats(std::array{h.distance},std::array{k.distance})
                ||!sameGlassCameraTraceFloats(h.position,k.position)||!sameGlassCameraTraceFloats(h.outwardNormal,k.outwardNormal)
                ||!sameGlassCameraTraceFloats(p.orientedNormal,q.orientedNormal)||!sameGlassCameraTraceFloats(p.transmitted,q.transmitted)
                ||!sameGlassCameraTraceFloats(p.reflected,q.reflected)||!sameGlassCameraTraceFloats(p.barycentric,q.barycentric)
                ||!sameGlassCameraTraceFloats(std::array{p.reflection,p.transmittedFraction,p.reflectedFraction,p.ior,p.fromIor,p.toIor},
                    std::array{q.reflection,q.transmittedFraction,q.reflectedFraction,q.ior,q.fromIor,q.toIor}))return false;
        }
    }
    return true;
}
// Conservative initial-air admission only. A point separated from the actual
// selected closed volume's world AABB is outside even for a concave volume.
// In/on the box is unknown and unsupported. This is no visibility query or
// general membership test; the native owner must authenticate these same
// retained scene triangles and camera uniforms against its actual buffers.
inline bool admitGlassCameraGratingTraceCamera(const Frame& frame,std::string& error)
{
    Point minimum{},maximum{};bool found=false;
    for(const auto& triangle:frame.triangles)if(triangle.glass)for(const auto& point:triangle.points) {
        if(!finite(point)){error="Camera trace glass bounds are nonfinite";return false;}
        if(!found){minimum=maximum=point;found=true;}
        else for(unsigned axis=0;axis<3;++axis){minimum[axis]=std::min(minimum[axis],point[axis]);maximum[axis]=std::max(maximum[axis],point[axis]);}
    }
    const auto origin=cameraPosition(frame);
    if(found&&finite(origin))for(unsigned axis=0;axis<3;++axis)
        if((double(minimum[axis])-origin[axis]>4*double(bias(origin)))
            ||(double(origin[axis])-maximum[axis]>4*double(bias(origin)))){error.clear();return true;}
    error="Camera trace initial medium is inside or unknown within the retained glass bounds";return false;
}
inline bool makeGlassCameraGratingTraceInput(const Frame& frame,std::uint32_t x,std::uint32_t y,std::uint32_t location,
    NativeGlassCameraGratingTraceInput& output,std::string& error)
{
    if(!nativeFixtureDimensionsWithinBounds(frame.width,frame.height)||x>=frame.width||y>=frame.height||location>4
        ||!validFixtureCameraOverride(frame.camera)||frame.camera.farPlane<=kRayMinimum)
    {error="Camera trace sample has no finite owning camera or pixel";return false;}
    if(!admitGlassCameraGratingTraceCamera(frame,error))return false;
    const float dx=location==0?.5f:(location==2||location==3?1.0f:0.0f);
    const float dy=location==0?.5f:(location==3||location==4?1.0f:0.0f);
    const auto origin=cameraPosition(frame),ray=cameraRay(frame,float(x)+dx,float(y)+dy);
    if(!finite(origin)||!finite(ray)||std::abs(dot(ray,ray)-1)>1e-4f)
    {error="Camera trace owning projection is nonfinite";return false;}
    NativeGlassCameraGratingTraceInput result;
    result.pixel={x,y,location,kGlassCameraGratingTraceVersion};result.control={frame.width,frame.height,0,0};
    result.originMinimum={origin[0],origin[1],origin[2],kRayMinimum};
    result.directionMaximum={ray[0],ray[1],ray[2],frame.camera.farPlane};
    output=result;error.clear();return true;
}
inline bool decodeGlassCameraGratingTraceBand(const NativeGlassCameraGratingTracePackedHeader& header,
    const NativeGlassCameraGratingTracePackedVisit* packed,std::size_t available,
    NativeGlassCameraGratingTraceBand& output,std::string& error)
{
    const auto zero=GlassCameraTraceWords{};
    if(header[6][3]>kMaximumBranchVisits||header[6][3]>available||(!packed&&header[6][3])
        ||header[7][0]>1||header[8][3]>1||header[9]!=zero)
    {error="Camera trace native header/count is malformed";return false;}
    const auto floats=[](const auto& row){return Texel{glassCameraTraceFloat(row[0]),glassCameraTraceFloat(row[1]),
        glassCameraTraceFloat(row[2]),glassCameraTraceFloat(row[3])};};
    const auto point=[&](const auto& row){const auto values=floats(row);return Point{values[0],values[1],values[2]};};
    NativeGlassCameraGratingTraceBand band;
    band.input.pixel=header[0];band.input.control=header[1];band.input.originMinimum=floats(header[2]);band.input.directionMaximum=floats(header[3]);
    band.nativeOriginMinimum=floats(header[4]);band.nativeDirectionMaximum=floats(header[5]);
    band.version=header[6][0];band.sample=header[6][1];band.band=header[6][2];
    band.traceComplete=header[7][0]!=0;band.reasons=header[7][1];band.queryWork=header[7][2]|(std::uint64_t(header[7][3])<<32);
    band.unresolvedFractionKnown=header[8][3]!=0;
    band.ior=glassCameraTraceFloat(header[8][0]);band.wavelength=glassCameraTraceFloat(header[8][1]);band.unresolvedFraction=glassCameraTraceFloat(header[8][2]);
    band.visits.reserve(header[6][3]);
    for(std::size_t i=0;i<header[6][3];++i) {
        const auto& words=packed[i];
        if(words[1][0]>1||words[1][2]>4||words[1][3]>1||words[5][2]>1||words[5][3]||words[7][3]
            ||words[14][2]||words[14][3]||words[15]!=zero||(words[4][0]>1&&words[4][0]!=kGlassCameraTraceRoot))
        {error="Camera trace native visit flags or padding are malformed";return false;}
        NativeGlassCameraGratingTraceVisit visit;
        visit.visit=words[0][0];visit.parent=words[0][1];visit.token=words[0][2];visit.interfaces=words[0][3];
        visit.inside=words[1][0]!=0;visit.history=words[1][1];visit.disposition=GlassCameraTraceDisposition(words[1][2]);visit.tir=words[1][3]!=0;
        visit.originMinimum=floats(words[2]);visit.directionFraction=floats(words[3]);
        visit.hit.status=words[4][0]==kGlassCameraTraceRoot?-1:int(words[4][0]);visit.hit.objectId=words[4][1];visit.hit.materialId=words[4][2];visit.hit.instanceId=words[4][3];
        visit.hit.triangleId=words[5][0];visit.hit.work=words[5][1];visit.hit.entering=words[5][2]!=0;
        visit.hit.position=point(words[6]);visit.hit.distance=glassCameraTraceFloat(words[6][3]);visit.hit.outwardNormal=point(words[7]);
        visit.orientedNormal=point(words[8]);visit.reflection=glassCameraTraceFloat(words[8][3]);
        visit.transmitted=point(words[9]);visit.transmittedFraction=glassCameraTraceFloat(words[9][3]);
        visit.reflected=point(words[10]);visit.reflectedFraction=glassCameraTraceFloat(words[10][3]);
        visit.barycentric=point(words[11]);visit.ior=glassCameraTraceFloat(words[11][3]);
        for(unsigned token=0;token<32;++token)visit.ancestry[token]=std::uint8_t(words[12+token/16][(token%16)/4]>>((token%4)*8));
        visit.fromIor=glassCameraTraceFloat(words[14][0]);visit.toIor=glassCameraTraceFloat(words[14][1]);band.visits.push_back(visit);
    }
    output=std::move(band);error.clear();return true;
}
// Existing native frame only. The hook authenticates current camera/IOR/query
// uniforms, retained scene-triangle/query resources and outside-camera bounds,
// exact native input/output buffers, completed dispatch/fence and cumulative
// allocation/attempt reservation. It compares supplied to actual
// retained decoded output with sameGlassCameraGratingTraceReadback. Keep its
// native frame/context/resource lease alive through validation and consumption.
class NativeGlassCameraGratingTraceBinding
{
public:
    virtual ~NativeGlassCameraGratingTraceBinding()=default;
    virtual const NativeFixtureSceneFrame& cameraTraceOwner() const noexcept=0;
    virtual const std::shared_ptr<const GratingIncidentBinding>& cameraTraceGratingBinding() const noexcept=0;
    virtual const NativeGlassCameraGratingTraceUpload& cameraTraceInputUpload() const noexcept=0;
    virtual bool validateCameraTraceProductionQuery(const SubmissionIdentity&,
        const NativeGlassCameraGratingTraceReadback&,std::string&) const=0;
protected:
    NativeGlassCameraGratingTraceBinding()=default;
};
struct GlassCameraGratingTraceBounds
{
    std::uint64_t maximumSamples=0,maximumBands=0,maximumVisits=0,queryWork=0,branchWork=0;
    std::uint64_t nativeInputBytes=0,nativeOutputBytes=0,pendingBytes=0,combinedWork=0,bytes=0;
};
inline bool calculateGlassCameraGratingTraceBounds(const GratingIncidentBinding& binding,std::size_t samples,
    std::uint64_t otherOwnedBytes,std::uint64_t otherWork,GlassCameraGratingTraceBounds& output,std::string& error)
{
    const auto refuse=[&](){error="Camera trace exceeds original cumulative work or storage caps";return false;};
    if(!admitGlassCameraGratingTraceCamera(binding.frame,error))return false;
    const auto control=binding.frame.query.control[0];
    if(!std::isfinite(control)||control<1||std::floor(control)!=control||control>binding.frame.query.texels.size())return refuse();
    GratingIncidentBounds original;
    if(!calculateGratingIncidentBounds(binding.frame,binding.triangles.size(),original,error))return false;
    const auto totalBytes=renderpassoutput::kMaximumTotalBytes;
    if(!samples||samples>kMaximumEntrySamples||binding.bounds.combinedWork!=original.combinedWork
        ||binding.bounds.queryWork!=original.queryWork||binding.bounds.orderWork!=original.orderWork||binding.bounds.bytes<original.bytes
        ||binding.frame.bounds.transientBytes>totalBytes||binding.bounds.bytes>totalBytes-binding.frame.bounds.transientBytes
        ||binding.bounds.combinedWork>kMaximumQueryWork||otherWork>kMaximumQueryWork-binding.bounds.combinedWork
        ||otherOwnedBytes>totalBytes-binding.frame.bounds.transientBytes-binding.bounds.bytes)return refuse();
    const auto& allowance=binding.frame.query.texels[std::size_t(binding.frame.query.control[0])-1];
    std::uint32_t low=0,high=0;
    if(!exactReceiptInteger(allowance[0],65535,low)||!exactReceiptInteger(allowance[1],65535,high))return refuse();
    const auto perQuery=std::uint64_t(low)+(std::uint64_t(high)<<16);
    GlassCameraGratingTraceBounds candidate;
    candidate.maximumSamples=samples;candidate.maximumBands=samples*8;candidate.maximumVisits=candidate.maximumBands*kMaximumBranchVisits;
    if(!perQuery||candidate.maximumVisits>kMaximumQueryWork/perQuery)return refuse();
    candidate.queryWork=candidate.maximumVisits*perQuery;candidate.branchWork=candidate.maximumVisits;
    const auto remainingWork=kMaximumQueryWork-binding.bounds.combinedWork-otherWork;
    if(candidate.queryWork>remainingWork||candidate.branchWork>remainingWork-candidate.queryWork)return refuse();
    candidate.combinedWork=binding.bounds.combinedWork+otherWork+candidate.queryWork+candidate.branchWork;
    candidate.nativeInputBytes=samples*sizeof(NativeGlassCameraGratingTraceInput);
    candidate.nativeOutputBytes=candidate.maximumBands*kGlassCameraTraceHeaderBytes+candidate.maximumVisits*kGlassCameraTraceVisitBytes;
    // Fixed16 pending plus popped and child copies, each128 bytes including
    // full ancestry. Native buffers/readbacks and two decoded copies coexist.
    candidate.pendingBytes=candidate.maximumBands*(kMaximumPendingBranches+2)*128;
    candidate.bytes=4*candidate.nativeInputBytes+2*candidate.nativeOutputBytes+candidate.pendingBytes
        +4*candidate.maximumVisits*sizeof(NativeGlassCameraGratingTraceVisit)
        +4*candidate.maximumBands*sizeof(NativeGlassCameraGratingTraceBand)
        +sizeof(NativeGlassCameraGratingTraceUpload)+2*sizeof(NativeGlassCameraGratingTraceReadback)
        +sizeof(NativeGlassCameraGratingTraceBinding)+6*binding.identity.size();
    if(candidate.bytes>totalBytes-binding.frame.bounds.transientBytes-binding.bounds.bytes-otherOwnedBytes)return refuse();
    output=candidate;error.clear();return true;
}
inline bool validateGlassCameraGratingTraceReadback(const GratingIncidentBinding& binding,
    const NativeGlassCameraGratingTraceBinding& native,const SubmissionIdentity& expected,double evaluationTime,
    const NativeGlassCameraGratingTraceReadback& readback,std::string& error)
{
    const auto refuse=[&](const char* message){error=message;return false;};
    const auto& owner=native.cameraTraceOwner();const auto& upload=native.cameraTraceInputUpload();const auto& frame=binding.frame;
    if(!binding.gratingOwner||!binding.material||native.cameraTraceGratingBinding().get()!=&binding
        ||dynamic_cast<const NativeGlassCameraGratingTraceBinding*>(&owner)!=&native||!owner.nativeResourceCacheIdentity()
        ||owner.width()!=frame.width||owner.height()!=frame.height||!binding.prepared.glass.sourceMaterial
        ||upload.backend!=binding.prepared.glass.sourceMaterial->backend||frame.roughness!=0||frame.multipleScatterPolicy
        ||!std::isfinite(evaluationTime)||evaluationTime!=binding.evaluationTime
        ||!validSourceSubmission(expected,frame,&binding)||!upload.complete||!upload.generation
        ||!validSourceSubmission(upload.submission,frame,&binding)||upload.submission.submissionSerial>expected.submissionSerial
        ||upload.submission.rendererGeneration!=expected.rendererGeneration||upload.submission.rasterGeneration!=expected.rasterGeneration
        ||upload.submission.queryGeneration!=expected.queryGeneration)
        return refuse("Camera trace lost its exact smooth frame, input or completion owner");
    if(!validateTables(binding.prepared,frame,error,true,frame.sourceDirectionalIrradiance.version
        ?SourceMeasureUse::DirectionalIrradiancePreparation:SourceMeasureUse::LegacyNative))return false;
    GlassCameraGratingTraceBounds bounds;
    if(!calculateGlassCameraGratingTraceBounds(binding,upload.inputs.size(),upload.otherOwnedBytes,upload.otherWork,bounds,error))return false;
    if(readback.version!=kGlassCameraGratingTraceVersion||!readback.dispatchComplete||readback.overflow
        ||!(readback.submission==expected)||readback.inputGeneration!=upload.generation||readback.bands.size()!=bounds.maximumBands
        ||readback.queryWork>bounds.queryWork||readback.bands.capacity()>2*bounds.maximumBands
        ||upload.inputs.capacity()>2*upload.inputs.size()||upload.submission.frameIdentity.capacity()>2*binding.identity.size()
        ||readback.submission.frameIdentity.capacity()>2*binding.identity.size())return refuse("Camera trace native count, work or completion changed");
    for(const auto& band:readback.bands)if(band.visits.size()>kMaximumBranchVisits||band.visits.capacity()>2*kMaximumBranchVisits)
        return refuse("Camera trace decoded visit storage exceeds original limits");
    if(!native.validateCameraTraceProductionQuery(expected,readback,error))return false;
    const auto& allowance=frame.query.texels[std::size_t(frame.query.control[0])-1];
    const auto perQuery=std::uint64_t(allowance[0])+(std::uint64_t(allowance[1])<<16);
    const auto unit=[](Point p){return finite(p)&&std::abs(dot(p,p)-1)<=1e-4f;};
    const auto splitNear=[](double actual,double expected,float parent) {
        return std::abs(actual-expected)<=8*double(std::numeric_limits<float>::epsilon())*parent
            +double(std::numeric_limits<float>::denorm_min());
    };
    std::uint64_t totalWork=0;
    for(std::size_t index=0;index<readback.bands.size();++index) {
        const auto& band=readback.bands[index];NativeGlassCameraGratingTraceInput canonical;
        const auto& input=upload.inputs[index/8];
        if(!makeGlassCameraGratingTraceInput(frame,input.pixel[0],input.pixel[1],input.pixel[2],canonical,error))return false;
        const Point nativeRay{band.nativeDirectionMaximum[0],band.nativeDirectionMaximum[1],band.nativeDirectionMaximum[2]};
        const Point expectedRay{canonical.directionMaximum[0],canonical.directionMaximum[1],canonical.directionMaximum[2]};
        if(!sameGlassCameraGratingTraceInput(input,canonical)||!sameGlassCameraGratingTraceInput(band.input,input)
            ||!sameGlassCameraTraceFloats(band.nativeOriginMinimum,canonical.originMinimum)
            ||glassCameraTraceFloatBits(band.nativeDirectionMaximum[3])!=glassCameraTraceFloatBits(frame.camera.farPlane)
            ||!unit(nativeRay)||!sameGratingPoint(nativeRay,expectedRay,2e-5f)
            ||band.version!=kGlassCameraGratingTraceVersion||band.sample!=index/8||band.band!=index%8
            ||glassCameraTraceFloatBits(band.ior)!=glassCameraTraceFloatBits(frame.ior[index%8])
            ||glassCameraTraceFloatBits(band.wavelength)!=glassCameraTraceFloatBits(diffractionmaterial::kProductionWavelengthsNanometres[index%8])
            ||(band.reasons&~255u)||band.traceComplete!=(band.reasons==0)
            ||!std::isfinite(band.unresolvedFraction)||band.unresolvedFraction<0||band.unresolvedFraction>1.0002f
            ||(band.traceComplete&&(!band.unresolvedFractionKnown||band.visits.empty()||band.unresolvedFraction!=0)))
            return refuse("Camera trace sample, spectral owner or incomplete prefix changed");
        std::array<std::array<bool,2>,kMaximumBranchVisits> children{};std::uint64_t work=0;bool unresolved=false;
        double unresolvedFraction=band.visits.empty()?1:0;bool unknownFraction=false;
        for(std::size_t v=0;v<band.visits.size();++v) {
            const auto& visit=band.visits[v];const auto& hit=visit.hit;
            const Point origin{visit.originMinimum[0],visit.originMinimum[1],visit.originMinimum[2]};
            const Point direction{visit.directionFraction[0],visit.directionFraction[1],visit.directionFraction[2]};
            const auto fraction=visit.directionFraction[3];
            if(visit.visit!=v||visit.interfaces>kMaximumInterfaces||!finite(origin)||!unit(direction)
                ||visit.originMinimum[3]!=kRayMinimum||!std::isfinite(fraction)||fraction<=0||fraction>1.0002f
                ||glassCameraTraceFloatBits(visit.ior)!=glassCameraTraceFloatBits(band.ior)||hit.work>perQuery
                ||hit.status<-1||hit.status>1||std::uint32_t(visit.disposition)>4||!std::isfinite(hit.distance)
                ||!finite(hit.position)||!finite(hit.outwardNormal)||!finite(visit.orientedNormal)||!finite(visit.transmitted)
                ||!finite(visit.reflected)||!finite(visit.barycentric)||!std::isfinite(visit.reflection)
                ||!std::isfinite(visit.transmittedFraction)||!std::isfinite(visit.reflectedFraction)
                ||!std::isfinite(visit.fromIor)||!std::isfinite(visit.toIor))
                return refuse("Camera trace visit or query work is malformed");
            if(v==0) {
                if(visit.parent!=kGlassCameraTraceRoot||visit.token||visit.interfaces||visit.history||visit.inside
                    ||visit.ancestry!=std::array<std::uint8_t,32>{}||fraction!=1
                    ||!sameGlassCameraTraceFloats(visit.originMinimum,band.nativeOriginMinimum)
                    ||!sameGlassCameraTraceFloats(direction,nativeRay))return refuse("Camera trace root is not the actual native camera ray");
            } else {
                if(visit.parent>=v||(visit.token!=1&&visit.token!=65))return refuse("Camera trace ancestry has no preceding native parent");
                const auto& parent=band.visits[visit.parent];const bool reflected=visit.token==1;const unsigned child=reflected?0:1;
                auto ancestry=parent.ancestry;
                if(parent.disposition!=GlassCameraTraceDisposition::Interface||parent.interfaces>=kMaximumInterfaces||children[visit.parent][child])
                    return refuse("Camera trace branch repeats or crosses a terminal parent");
                ancestry[parent.interfaces]=std::uint8_t(visit.token);children[visit.parent][child]=true;
                const auto expectedFraction=reflected?parent.reflectedFraction:parent.transmittedFraction;
                const auto expectedDirection=reflected?parent.reflected:parent.transmitted;
                const auto expectedOrigin=advance(parent.hit.position,parent.orientedNormal,(reflected?1.0f:-1.0f)*bias(parent.hit.position));
                const auto history=parent.history|(reflected?(1u<<parent.interfaces):0u);
                if(visit.interfaces!=parent.interfaces+1||visit.ancestry!=ancestry||visit.history!=history
                    ||visit.inside!=(reflected?parent.inside:!parent.inside)||expectedFraction<=0
                    ||glassCameraTraceFloatBits(fraction)!=glassCameraTraceFloatBits(expectedFraction)
                    ||!sameGlassCameraTraceFloats(direction,expectedDirection)||!sameGratingPoint(origin,expectedOrigin,bias(expectedOrigin)))
                    return refuse("Camera trace child differs from its actual native interface branch");
            }
            work+=hit.work;
            if(visit.disposition!=GlassCameraTraceDisposition::Interface
                &&(visit.tir||!sameGlassCameraTraceFloats(visit.orientedNormal,Point{})
                    ||!sameGlassCameraTraceFloats(visit.transmitted,Point{})||!sameGlassCameraTraceFloats(visit.reflected,Point{})
                    ||!sameGlassCameraTraceFloats(std::array{visit.reflection,visit.transmittedFraction,visit.reflectedFraction,visit.fromIor,visit.toIor},
                        std::array<float,5>{})))return refuse("Camera trace non-interface visit carries an interface witness");
            if(visit.disposition!=GlassCameraTraceDisposition::Grating&&!sameGlassCameraTraceFloats(visit.barycentric,Point{}))
                return refuse("Camera trace non-grating visit carries grating barycentrics");
            if(visit.disposition==GlassCameraTraceDisposition::Unresolved) {
                if(!band.reasons||(hit.status<0&&!(band.reasons&GlassCameraQueryFailed)))return refuse("Camera trace failure became a completed path");
                unresolvedFraction+=fraction;unresolved=true;continue;
            }
            if(hit.status<0||!hit.work||!std::isfinite(hit.distance)||hit.distance<kRayMinimum||hit.distance>frame.camera.farPlane)
                return refuse("Camera trace has no completed finite native query");
            if(visit.disposition==GlassCameraTraceDisposition::Escape) {
                if(visit.inside||hit.status!=0||hit.distance!=frame.camera.farPlane||hit.position!=Point{}||hit.outwardNormal!=Point{}
                    ||hit.objectId||hit.materialId||hit.instanceId||hit.triangleId||hit.entering)
                    return refuse("Camera trace finite miss has nonempty or interior hit data");
                continue;
            }
            const auto* triangle=hitTriangle(frame,hit);
            if(hit.status!=1||!triangle||!sameGratingPoint(hit.outwardNormal,triangle->normal)
                ||!sameGratingPoint(advance(origin,direction,hit.distance),hit.position,8*bias(hit.position))
                ||hit.entering!=(dot(direction,hit.outwardNormal)<0))return refuse("Camera trace hit differs from its owning primitive or query ray");
            if(visit.disposition==GlassCameraTraceDisposition::Interface) {
                Point normal=hit.outwardNormal;if(visit.inside)for(auto& value:normal)value=-value;
                if(!triangle->glass||visit.inside==hit.entering||visit.interfaces>=kMaximumInterfaces
                    ||!sameGlassCameraTraceFloats(normal,visit.orientedNormal)||dot(direction,normal)>=0
                    ||!std::isfinite(visit.reflection)||visit.reflection<0||visit.reflection>1
                    ||!unit(visit.reflected)||(!visit.tir&&!unit(visit.transmitted))
                    ||dot(visit.reflected,normal)<0||(!visit.tir&&dot(visit.transmitted,normal)>0)
                    ||visit.fromIor!=(visit.inside?band.ior:1)||visit.toIor!=(visit.inside?1:band.ior)
                    ||!std::isfinite(visit.reflectedFraction)||visit.reflectedFraction<0
                    ||!std::isfinite(visit.transmittedFraction)||visit.transmittedFraction<0
                    ||(visit.tir&&(!visit.inside||band.ior<=1||visit.reflection!=1||visit.transmittedFraction!=0))
                    ||(visit.reflection==0&&visit.reflectedFraction!=0)||(visit.reflection==1&&visit.transmittedFraction!=0)
                    ||!splitNear(double(visit.reflectedFraction)+visit.transmittedFraction,fraction,fraction))
                    return refuse("Camera trace smooth native interface witness is malformed");
                if((visit.reflection>0&&visit.reflectedFraction==0)||(!visit.tir&&visit.reflection<1&&visit.transmittedFraction==0)) {
                    if(!(band.reasons&GlassCameraFractionUnderflow))return refuse("Camera trace pruned a positive underflow branch");
                    unresolved=true;
                }
                // Fraction bookkeeping checks the reported native split. It
                // does not calculate Fresnel or generate an optical branch.
                if((visit.reflectedFraction>0&&!splitNear(visit.reflectedFraction,double(fraction)*visit.reflection,fraction))
                    ||(visit.transmittedFraction>0&&!splitNear(visit.transmittedFraction,double(fraction)*(1-visit.reflection),fraction)))
                    return refuse("Camera trace fraction differs from its retained native split");
            } else if(visit.disposition==GlassCameraTraceDisposition::Grating) {
                const auto witness=std::find_if(binding.triangles.begin(),binding.triangles.end(),[&](const auto& t) {
                    return std::tie(t.object,t.material,t.instance,t.triangle)==std::tie(hit.objectId,hit.materialId,hit.instanceId,hit.triangleId);});
                if(visit.inside||!hit.entering||witness==binding.triangles.end()||!finite(visit.barycentric)
                    ||std::abs(visit.barycentric[0]+visit.barycentric[1]+visit.barycentric[2]-1)>1e-5f)
                    return refuse("Camera trace grating terminal has no strict owning primitive");
                Point position{};
                for(unsigned corner=0;corner<3;++corner) {
                    if(visit.barycentric[corner]<=1e-5f||visit.barycentric[corner]>=1-1e-5f)return refuse("Camera trace grating terminal is on an ambiguous seam");
                    position=advance(position,witness->positions[corner],visit.barycentric[corner]);
                }
                if(!sameGratingPoint(position,hit.position,8*bias(hit.position)))return refuse("Camera trace grating barycentric witness changed");
            } else if(visit.disposition!=GlassCameraTraceDisposition::Other||visit.inside||triangle->glass
                ||std::any_of(binding.triangles.begin(),binding.triangles.end(),[&](const auto& t){return t.object==hit.objectId;}))
                return refuse("Camera trace terminal classification changed");
        }
        for(std::size_t v=0;v<band.visits.size();++v)if(band.visits[v].disposition==GlassCameraTraceDisposition::Interface) {
            const auto& visit=band.visits[v];
            if((visit.reflectedFraction>0&&!children[v][0])||(visit.transmittedFraction>0&&!children[v][1]))unresolved=true;
            // A stopped split with neither child represented owns its whole
            // incoming branch once. Otherwise only the missing child frontier
            // is unresolved, never that child's already represented ancestor.
            if(!children[v][0]&&!children[v][1])unresolvedFraction+=visit.directionFraction[3];
            else {
                if(!children[v][0])unresolvedFraction+=visit.reflectedFraction;
                if(!children[v][1])unresolvedFraction+=visit.transmittedFraction;
                if((visit.reflection>0&&visit.reflectedFraction==0)
                    ||(!visit.tir&&visit.reflection<1&&visit.transmittedFraction==0))unknownFraction=true;
            }
        }
        const auto fractionTolerance=kMaximumBranchVisits*double(std::numeric_limits<float>::epsilon())*unresolvedFraction
            +double(std::numeric_limits<float>::denorm_min());
        if(band.unresolvedFractionKnown!=(!unknownFraction)
            ||(!band.unresolvedFractionKnown&&glassCameraTraceFloatBits(band.unresolvedFraction)!=0)
            ||(band.unresolvedFractionKnown&&((unresolvedFraction>0&&band.unresolvedFraction<=0)
                ||(unresolvedFraction==0&&glassCameraTraceFloatBits(band.unresolvedFraction)!=0)
                ||std::abs(double(band.unresolvedFraction)-unresolvedFraction)>fractionTolerance)))
            return refuse("Camera trace unresolved fraction differs from its native branch frontier");
        if(work!=band.queryWork||(band.traceComplete&&unresolved))return refuse("Camera trace omitted positive branches or changed query work");
        totalWork+=work;
    }
    if(totalWork!=readback.queryWork)return refuse("Camera trace whole-frame native query work changed");
    error.clear();return true;
}
// This helper consumes a validated native event. The whole readback gate must
// establish its submission and source receipt before power/area gains units.
// It reconstructs only the entry basis and never replays the branch on CPU.
inline bool gratingReceivingArea(const GratingIncidentBinding& binding,const NativeGratingIncident& event,
    GratingReceivingArea& output)
{
    if(event.version!=kGratingIncidentVersion||!event.complete||event.work>=binding.frame.work.size()
        ||binding.frame.roughness!=0||binding.frame.multipleScatterPolicy
        ||(binding.frame.sourceDirectionalIrradiance.version!=0
            &&(binding.frame.sourceDirectionalIrradiance.version!=1
                ||binding.frame.sourceDirectionalIrradiance.measure!=diffractionmaterial::DirectionalSpectralMeasure::IrradianceCoefficient)))return false;
    const auto& work=binding.frame.work[event.work];
    if(work.entry>=binding.frame.entries.size())return false;
    const auto& entry=binding.frame.entries[work.entry];
    const auto source=std::find_if(binding.frame.triangles.begin(),binding.frame.triangles.end(),[&](const auto& t) {
        return std::tie(t.object,t.material,t.instance,t.triangle)==std::tie(entry.object,entry.material,entry.instance,entry.triangle);});
    const auto target=std::find_if(binding.triangles.begin(),binding.triangles.end(),[&](const auto& t) {
        return std::tie(t.object,t.material,t.instance,t.triangle)==std::tie(event.hit.objectId,event.hit.materialId,event.hit.instanceId,event.hit.triangleId);});
    GratingReceivingArea result;float dbu,dbv;
    if(source==binding.frame.triangles.end()||!source->glass||target==binding.triangles.end()
        ||!gratingReceivingArea(sub(source->points[1],source->points[0]),sub(source->points[2],source->points[0]),entry.area,event.dHdu,event.dHdv,result)
        ||std::abs(dot(event.dHdu,target->normal))>2e-4f*std::sqrt(dot(event.dHdu,event.dHdu))
        ||std::abs(dot(event.dHdv,target->normal))>2e-4f*std::sqrt(dot(event.dHdv,event.dHdv))
        ||!smoothBiasDifferential(event.hit.position,event.dHdu,dbu)||!smoothBiasDifferential(event.hit.position,event.dHdv,dbv))return false;
    for(float coordinate:event.barycentric)if(!std::isfinite(coordinate)||coordinate<=1e-5f||coordinate>=1-1e-5f)return false;
    result.squareMetres=binding.frame.sourceDirectionalIrradiance.version==1;
    output=result;return true;
}
// This verifies a completed producer's typed geometry and energy contract.
// It cannot prove native execution or render Color by itself.
inline bool validateGratingIncident(const GratingIncidentBinding& binding,const NativeGratingIncident& event,
    const SubmissionIdentity& expected,const SubmissionIdentity& received,std::string& error)
{
    const auto refuse=[&](){error="Glass to grating incident witness, lineage or energy is invalid";return false;};
    if(!binding.material||!binding.prepared.branchPolicy||binding.identity.empty()||binding.triangles.empty()
        ||binding.frame.roughness!=0||binding.frame.multipleScatterPolicy
        ||!(expected==received)||expected.frameIdentity!=binding.identity||!expected.rendererGeneration
        ||!expected.submissionSerial||!expected.rasterGeneration||!expected.queryGeneration
        ||!expected.scratchGeneration||expected.completedFence!=expected.submissionSerial
        ||event.version!=kGratingIncidentVersion||!event.complete||event.work>=binding.frame.work.size()
        ||event.band!=binding.frame.work[event.work].band||!event.interfaces||event.interfaces>kMaximumInterfaces
        ||event.orderCount>kGratingOrdersPerIncident||!finite(event.origin)||!finite(event.direction)
        ||std::abs(dot(event.direction,event.direction)-1)>1e-4f||!std::isfinite(event.fraction)
        ||event.fraction<=0||event.fraction>1||!nearPower(event.power,binding.frame.work[event.work].incidentPower*event.fraction)
        ||event.hit.status!=1||!event.hit.entering||!event.hit.work||!std::isfinite(event.hit.distance)
        ||event.hit.distance<kRayMinimum||event.hit.distance>binding.frame.sourceDirectionTrace[3])return refuse();
    const auto& allowance=binding.frame.query.texels[std::size_t(binding.frame.query.control[0])-1];
    if(event.hit.work>std::uint64_t(allowance[0])+(std::uint64_t(allowance[1])<<16))return refuse();
    bool inside=false;
    for(unsigned depth=0;depth<32;++depth) {
        const bool reflected=(event.reflectionHistory>>depth)&1u;
        if(depth<event.interfaces) {
            if(event.ancestry[depth]!=(reflected?1u:65u))return refuse();
            if(!reflected)inside=!inside;
        } else if(event.ancestry[depth]||reflected)return refuse();
    }
    if(inside||(event.reflectionHistory>>32))return refuse();
    const auto found=std::find_if(binding.triangles.begin(),binding.triangles.end(),[&](const auto& t) {
        return std::tie(t.object,t.material,t.instance,t.triangle)==std::tie(event.hit.objectId,event.hit.materialId,event.hit.instanceId,event.hit.triangleId);});
    if(found==binding.triangles.end()||dot(event.direction,found->normal)>=-1e-6f
        ||!sameGratingPoint(event.normal,found->normal)||!sameGratingPoint(event.hit.outwardNormal,found->normal)
        ||!sameGratingPoint(event.tangent,found->tangent)||!sameGratingPoint(event.bitangent,found->bitangent)
        ||!finite(event.barycentric)||std::abs(event.barycentric[0]+event.barycentric[1]+event.barycentric[2]-1)>1e-5f)return refuse();
    Point position{};std::array<float,2> uv{};
    for(unsigned v=0;v<3;++v) {
        if(event.barycentric[v]<=1e-5f||event.barycentric[v]>=1-1e-5f)return refuse();
        position=advance(position,found->positions[v],event.barycentric[v]);
        for(unsigned a=0;a<2;++a)uv[a]+=found->uv[v][a]*event.barycentric[v];
    }
    const float tolerance=8*bias(event.hit.position);
    if(!sameGratingPoint(position,event.hit.position,tolerance)
        ||!sameGratingPoint(advance(event.origin,event.direction,event.hit.distance),event.hit.position,tolerance))return refuse();
    for(unsigned a=0;a<2;++a)if(!std::isfinite(event.uv[a])||std::abs(event.uv[a]-uv[a])>1e-5f)return refuse();
    GratingReceivingArea receiving;
    if(!gratingReceivingArea(binding,event,receiving))return refuse();
    double resolved=0;std::set<std::int32_t> orders;
    for(unsigned i=0;i<event.orderCount;++i) {
        const auto& order=event.orders[i];
        if(order.order<-1||order.order>1||!orders.insert(order.order).second||!finite(order.direction)
            ||std::abs(dot(order.direction,order.direction)-1)>1e-4f||dot(order.direction,found->normal)<=0
            ||!std::isfinite(order.power)||order.power<=0)return refuse();
        // Existing spectral grating equation, checked without constructing any
        // product ray. Incident BSDF convention is -event.direction.
        const auto& geometry=binding.material->description().geometry;
        const float step=order.order*binding.material->description().spectrum.wavelengthsNanometres[event.band]
            /geometry.grooveSpacingNanometres;
        if(std::abs(dot(order.direction,found->tangent)-dot(event.direction,found->tangent)-step*geometry.directionUv[0])>2e-4f
            ||std::abs(dot(order.direction,found->bitangent)-dot(event.direction,found->bitangent)-step*geometry.directionUv[1])>2e-4f)return refuse();
        resolved+=order.power;
    }
    for(unsigned i=event.orderCount;i<3;++i)if(event.orders[i].order||event.orders[i].direction!=Point{}||event.orders[i].power)return refuse();
    if(!std::isfinite(event.substrateReflectedPower)||event.substrateReflectedPower<0||event.substrateReflectedPower>event.power
        ||!std::isfinite(event.absorbedPower)||event.absorbedPower<0||!std::isfinite(event.unresolvedPower)||event.unresolvedPower<0
        ||!nearPower(resolved+event.unresolvedPower,event.substrateReflectedPower)
        ||!nearPower(event.substrateReflectedPower+event.absorbedPower,event.power))return refuse();
    error.clear();return true;
}
inline bool validateGratingIncidentReadback(const GratingIncidentBinding& binding,const NativeGratingReadback& readback,
    const SubmissionIdentity& expected,std::string& error,const NativeDirectionalSourceBinding* sourceBinding)
{
    const auto refuse=[&](){error="Glass to grating incident batch is incomplete or exceeds existing bounds";return false;};
    if(!binding.prepared.branchPolicy||!binding.material||!binding.bounds.maximumIncidents
        ||binding.bounds.queryWork<binding.frame.bounds.queryWork
        ||binding.identity.empty()||!expected.rendererGeneration||!expected.submissionSerial
        ||!expected.rasterGeneration||!expected.queryGeneration||!expected.scratchGeneration
        ||expected.completedFence!=expected.submissionSerial
        ||!readback.complete||readback.overflow||readback.incidents.size()>binding.bounds.maximumIncidents
        ||!(readback.submission==expected)||expected.frameIdentity!=binding.identity
        ||readback.orderWork!=readback.incidents.size()*kGratingOrdersPerIncident
        ||readback.queryWork>binding.bounds.queryWork-binding.frame.bounds.queryWork
        ||readback.queryWork+readback.orderWork+binding.frame.bounds.queryWork>binding.bounds.combinedWork)return refuse();
    if(binding.frame.sourceDirectionalIrradiance.version!=0) {
        if(!sourceBinding ||!readback.directionalSource)
        {error="Directional grating incident batch has no native source proof";return false;}
        if(!validateDirectionalSourceReadback(binding.prepared,binding.frame,*sourceBinding,expected,
            *readback.directionalSource,error,true,&binding))return false;
    } else if(sourceBinding ||readback.directionalSource) {
        error="Legacy grating incident batch cannot carry a directional source grant";return false;
    }
    std::set<std::tuple<std::uint32_t,std::uint32_t,std::uint64_t>> branches;
    std::vector<double> fractions(binding.frame.work.size()),unresolved(binding.frame.work.size());
    std::uint64_t orderQueries=0;
    for(const auto& event:readback.incidents) {
        if(!validateGratingIncident(binding,event,expected,readback.submission,error))return false;
        if(!branches.emplace(event.work,event.interfaces,event.reflectionHistory).second)return refuse();
        fractions[event.work]+=event.fraction;unresolved[event.work]+=event.unresolvedPower;
        orderQueries+=event.orderCount;
    }
    if(readback.childrenComplete) {
        orderQueries=0;
        for(const auto& child:readback.children) {
            if(child.incident>=readback.incidents.size())return refuse();
            if(child.disposition==NativeGratingReadback::Disposition::Receiver
                ||child.disposition==NativeGratingReadback::Disposition::Escaped)++orderQueries;
            else if(child.disposition!=NativeGratingReadback::Disposition::Unresolved)return refuse();
        }
    }
    // Incident-only producers query every emitted order. Completed terminal
    // batches may honestly omit a weak child, retaining its power as unresolved.
    if(readback.queryWork<orderQueries)return refuse();
    for(std::size_t w=0;w<fractions.size();++w)
        if(fractions[w]>1.0002||unresolved[w]>binding.frame.work[w].incidentPower
            *binding.prepared.branchPolicy->description().maximumUnresolvedFraction)return refuse();
    if(!readback.paths.empty()) {
        if(readback.paths.size()!=binding.frame.work.size())return refuse();
        std::vector<unsigned> counts(readback.paths.size());std::vector<double> absorbed(readback.paths.size()),tails(readback.paths.size());
        for(const auto& event:readback.incidents){++counts[event.work];absorbed[event.work]+=event.absorbedPower;tails[event.work]+=event.unresolvedPower;}
        std::uint64_t queries=0,optical=0;
        for(std::size_t w=0;w<counts.size();++w) {
            const auto& path=readback.paths[w];
            if(!path.complete||path.incidentCount!=counts[w]||path.orderWork!=3*counts[w]
                ||!nearPower(path.absorbedPower,absorbed[w])||!std::isfinite(path.unresolvedPower)||path.unresolvedPower<tails[w]*(1-2e-4)
                ||path.queryWork>readback.queryWork||queries>readback.queryWork-path.queryWork)return refuse();
            queries+=path.queryWork;optical+=path.orderWork;
        }
        if(queries!=readback.queryWork||optical!=readback.orderWork)return refuse();
    }
    // This is only the grating subset. The later shared path gate must also
    // include glass and omitted-order tails in its original total1e-4 ledger.
    error.clear();return true;
}
// Optional camera admission is separate so terminal-only stages retain their
// existing query/storage bounds. Native records and bounded host copies coexist.
inline bool calculateGratingCameraBounds(const GratingIncidentBinding& binding,GratingCameraBounds& output,std::string& error)
{
    GratingIncidentBounds exact;
    if(!calculateGratingIncidentBounds(binding.frame,binding.triangles.size(),exact,error))return false;
    if(binding.bounds.maximumOrders!=exact.maximumOrders ||binding.bounds.combinedWork!=exact.combinedWork
        ||binding.bounds.bytes<exact.bytes ||binding.bounds.bytes>renderpassoutput::kMaximumTotalBytes-binding.frame.bounds.transientBytes)
    {error="Grating camera retained incident resource admission changed";return false;}
    const auto& allowance=binding.frame.query.texels[std::size_t(binding.frame.query.control[0])-1];
    const auto query=std::uint64_t(allowance[0])+(std::uint64_t(allowance[1])<<16);
    GratingCameraBounds candidate;candidate.maximumOrders=exact.maximumOrders;
    candidate.queryWork=candidate.maximumOrders*query;
    candidate.bytes=sizeof(NativeGratingCameraReadback)+binding.identity.size()
        +candidate.maximumOrders*(kGratingCameraOrderNativeBytes+2*sizeof(NativeGratingCameraReadback::OrderReceipt));
    if(candidate.queryWork>kMaximumQueryWork-exact.combinedWork
        ||candidate.bytes>renderpassoutput::kMaximumTotalBytes-binding.frame.bounds.transientBytes-binding.bounds.bytes)
    {error="Grating camera optional queries or storage exceed the existing whole-frame budget";return false;}
    candidate.combinedWork=exact.combinedWork+candidate.queryWork;
    output=candidate;error.clear();return true;
}
// Native vectors have binary32 unit-length admission. Normalize only that
// rounding when entering the binary64 optical authority, preserving direction.
inline bool gratingCameraUnitDirection(Point native,std::array<double,3>& output)
{
    if(!finite(native))return false;
    const double squared=double(native[0])*native[0]+double(native[1])*native[1]+double(native[2])*native[2];
    if(!std::isfinite(squared)||std::abs(squared-1)>1e-4)return false;
    std::array<double,3> result{};
    for(unsigned a=0;a<3;++a)result[a]=native[a]/std::sqrt(squared);
    output=result;return true;
}
inline bool gratingCameraOpticalFrame(Point tangent,Point bitangent,Point normal,
    diffractionmaterial::DiffractionLocalFrame& output)
{
    diffractionmaterial::DiffractionLocalFrame result;
    if(!gratingCameraUnitDirection(tangent,result.tangent)
        ||!gratingCameraUnitDirection(bitangent,result.bitangent)
        ||!gratingCameraUnitDirection(normal,result.normal))return false;
    double projection=0,squared=0;
    for(unsigned a=0;a<3;++a)projection+=result.tangent[a]*result.normal[a];
    for(unsigned a=0;a<3;++a) {
        result.tangent[a]-=projection*result.normal[a];squared+=result.tangent[a]*result.tangent[a]; }
    if(!std::isfinite(squared)||squared<=0)return false;
    for(auto& component:result.tangent)component/=std::sqrt(squared);
    result.bitangent={result.normal[1]*result.tangent[2]-result.normal[2]*result.tangent[1],
        result.normal[2]*result.tangent[0]-result.normal[0]*result.tangent[2],
        result.normal[0]*result.tangent[1]-result.normal[1]*result.tangent[0]};
    const auto withinNative=[](Point original,const std::array<double,3>& corrected) {
        double distance=0;
        for(unsigned a=0;a<3;++a)distance+=(corrected[a]-original[a])*(corrected[a]-original[a]);
        return distance<=double(2e-4f)*double(2e-4f); };
    // The existing retained native frame comparison uses this vector-distance
    // bound. Mirrored or materially nonorthogonal axes cannot be repaired here.
    if(!withinNative(tangent,result.tangent)||!withinNative(bitangent,result.bitangent)
        ||!withinNative(normal,result.normal))return false;
    output=result;return true;
}
// This validates only retained native visibility and private angular math.
// It never queries geometry on CPU, writes the radiance accumulator, or grants
// finite receiving-patch coverage from an incident point and its derivatives.
inline bool validateGratingCameraReadback(const GratingIncidentBinding& binding,const NativeReadback& native,
    const SubmissionIdentity& expected,std::string& error,const NativeDirectionalSourceBinding* sourceBinding)
{
    const auto refuse=[&](const char* message){error=message;return false;};
    if(native.gratingCameraVersion!=kGratingCameraReceiptVersion ||!native.gratingCamera
        ||!native.gratingIncidents ||native.gratingIncidentVersion!=kGratingIncidentVersion)
        return refuse("Grating camera receipt marker or retained incident owner is missing");
    const auto& receipt=*native.gratingCamera;
    if(receipt.version!=kGratingCameraReceiptVersion ||!receipt.complete ||receipt.overflow
        ||!(receipt.submission==expected) ||receipt.incidents.get()!=native.gratingIncidents.get()
        ||native.frameIdentity!=binding.identity ||native.directionalSource.get()!=receipt.incidents->directionalSource.get()
        ||!validSourceSubmission(expected,binding.frame,&binding)
        ||receipt.orders.size()>binding.bounds.maximumOrders ||receipt.orders.capacity()>2*binding.bounds.maximumOrders)
        return refuse("Grating camera native owner, generation, count or completed fence changed");
    if(!validateGratingIncidentReadback(binding,*receipt.incidents,expected,error,sourceBinding))return false;
    if(binding.frame.sourceDirectionalIrradiance.version!=1 ||!sourceBinding)
        return refuse("Grating camera physical irradiance requires authored directional D v1 metre-world units");
    GratingCameraBounds cameraBounds;
    if(!calculateGratingCameraBounds(binding,cameraBounds,error))return false;
    const auto& frame=binding.frame;
    const auto& allowance=frame.query.texels[std::size_t(frame.query.control[0])-1];
    const auto queryMaximum=std::uint64_t(allowance[0])+(std::uint64_t(allowance[1])<<16);
    diffractionmaterial::DiffractionLocalFrame axes;
    const auto rotate=[&](Point p) {
        const auto v=rotateSceneVector(frame.camera.transform.rotation,{p[0],p[1],p[2]});
        return Point{v.x,v.y,v.z}; };
    if(!gratingCameraOpticalFrame(rotate({1,0,0}),rotate({0,1,0}),rotate({0,0,1}),axes))
        return refuse("Grating camera native camera axes exceed optical conversion precision");
    const double halfY=std::tan(double(frame.camera.verticalFovRadians)/2),halfX=halfY*frame.width/frame.height;
    std::size_t count=0;std::uint64_t queries=0;
    for(std::size_t i=0;i<receipt.incidents->incidents.size();++i) {
        const auto& event=receipt.incidents->incidents[i];
        GratingReceivingArea receiving;
        if(!gratingReceivingArea(binding,event,receiving) ||!receiving.squareMetres)
            return refuse("Grating camera incident has no exact metric receiving-area witness");
        const double irradiance=double(event.power)/receiving.area;
        diffractionmaterial::DiffractionLocalFrame local;
        std::array<double,3> incident;
        if(!gratingCameraOpticalFrame(event.tangent,event.bitangent,event.normal,local)
            ||!gratingCameraUnitDirection(advance({},event.direction,-1),incident))
            return refuse("Grating camera native incident axes exceed optical conversion precision");
        // Native orders retain signed provenance. Visit them canonically so
        // duplicate, omitted or reordered receipts cannot change float sums.
        for(int order=-1;order<=1;++order) {
            const auto found=std::find_if(event.orders.begin(),event.orders.begin()+event.orderCount,
                [&](const auto& o){return o.order==order;});
            if(found==event.orders.begin()+event.orderCount)continue;
            if(count>=receipt.orders.size())return refuse("Grating camera omitted a native incident order or band");
            const auto& r=receipt.orders[count++];
            if(!r.complete ||r.incident!=i ||r.band!=event.band ||r.order!=order
                ||!std::isfinite(r.spectralRadiance) ||r.spectralRadiance<0)
                return refuse("Grating camera incident, order or band is duplicated, reordered or incomplete");
            double backward=0,imageX=0,imageY=0;
            for(unsigned a=0;a<3;++a) {backward+=found->direction[a]*axes.normal[a];imageX-=found->direction[a]*axes.tangent[a];imageY-=found->direction[a]*axes.bitangent[a];}
            const bool inCamera=backward>0 &&imageX>=-halfX*backward &&imageX<halfX*backward
                &&imageY<=halfY*backward &&imageY>-halfY*backward;
            const auto x=inCamera?std::uint32_t((imageX/(backward*halfX)+1)*.5*frame.width):0;
            const auto y=inCamera?std::uint32_t((1-imageY/(backward*halfY))*.5*frame.height):0;
            if(inCamera &&(r.x!=x ||r.y!=y ||x>=frame.width ||y>=frame.height))
                return refuse("Grating camera angular pixel partition changed");
            diffractionmaterial::DiffractionCameraPixel pixel{axes,frame.camera.verticalFovRadians,frame.width,frame.height,x,y};
            diffractionmaterial::DiffractionCameraPixelIntegral integral;
            if(!diffractionmaterial::integrateDiscreteDiffractionCameraPixel(*binding.material,event.band,local,incident,
                {diffractionmaterial::DiffractionIncidentMeasure::SurfaceSpectralIrradiance,irradiance},pixel,integral,error))return false;
            const auto measured=std::find_if(integral.orders.begin(),integral.orders.begin()+integral.orderCount,
                [&](const auto& o){return o.order.primary==order &&o.order.secondary==0;});
            if(measured==integral.orders.begin()+integral.orderCount ||measured->inPixel!=inCamera
                ||!nearDirectionalSourcePotential(event.substrateReflectedPower,integral.substrateReflectedIrradiance*receiving.area)
                ||!nearDirectionalSourcePotential(event.absorbedPower,integral.absorbedIrradiance*receiving.area)
                ||!nearDirectionalSourcePotential(event.unresolvedPower,integral.unresolvedReflectedIrradiance*receiving.area)
                ||!nearDirectionalSourcePotential(found->power,measured->reflectedIrradiance*receiving.area))
                return refuse("Grating camera discrete physical order or parent power changed");
            for(unsigned a=0;a<3;++a)if(std::abs(found->direction[a]-measured->order.direction[a])>2e-4)
                return refuse("Grating camera discrete order direction changed");
            if(!inCamera) {
                if(r.outcome!=NativeGratingCameraReadback::Outcome::OutsideCamera ||r.x!=UINT32_MAX ||r.y!=UINT32_MAX
                    ||r.cameraHit.status!=-1 ||r.cameraHit.work ||r.cameraHit.distance!=0 ||r.cameraHit.position!=Point{}
                    ||r.cameraHit.outwardNormal!=Point{} ||r.cameraHit.objectId ||r.cameraHit.materialId
                    ||r.cameraHit.instanceId ||r.cameraHit.triangleId ||r.cameraHit.entering ||r.spectralRadiance!=0
                    ||r.raster.object ||r.raster.material ||r.raster.instance ||r.raster.triangle ||r.raster.linearDepth!=0)
                    return refuse("Grating camera outside-support outcome contains visibility or radiance");
                continue;
            }
            const auto& hit=r.cameraHit;const auto* triangle=hitTriangle(frame,hit);
            Point ray=found->direction;for(float& value:ray)value=-value;
            const auto origin=cameraPosition(frame),residual=sub(advance(origin,ray,hit.distance),hit.position);
            const auto& q=frame.camera.transform.rotation;
            const auto p=sub(hit.position,origin);
            const auto v=rotateSceneVector({-q.x,-q.y,-q.z,q.w},{p[0],p[1],p[2]});
            // Nearest-hit ownership is native evidence, separate from the CPU
            // angular integral. A miss cannot be called an occluding surface.
            const float denominator=dot(ray,event.normal);
            const float targetDistance=dot(sub(event.hit.position,origin),event.normal)/denominator;
            if(!triangle ||!hit.work ||hit.work>queryMaximum ||!hit.entering ||!std::isfinite(targetDistance)
                ||denominator>=-1e-6f ||targetDistance<=0 ||hit.distance>targetDistance+8*bias(hit.position)
                ||dot(residual,residual)>std::pow(8*bias(hit.position),2)
                ||-v.z<frame.camera.nearPlane ||-v.z>frame.camera.farPlane
                ||r.raster.object!=hit.objectId ||r.raster.material!=hit.materialId
                ||r.raster.instance!=hit.instanceId ||r.raster.triangle!=hit.triangleId)
                return refuse("Grating camera native nearest-hit ray, raster or exact owner changed");
            // The raster samples the pinhole pixel center, while the exact
            // order query can land elsewhere within this angular pixel. Check
            // the native raster against its retained triangle plane, without
            // manufacturing a visibility query or replacing its native depth.
            const auto pixelRay=cameraRay(frame,float(x)+.5f,float(y)+.5f);
            const float pixelDenominator=dot(pixelRay,triangle->normal);
            if(!std::isfinite(pixelDenominator) ||pixelDenominator>=-1e-6f)
                return refuse("Grating camera raster pixel center is grazing or backfacing");
            const float pixelDistance=dot(sub(triangle->points[0],origin),triangle->normal)/pixelDenominator;
            const auto pixelPosition=advance(origin,pixelRay,pixelDistance);
            const auto delta=sub(pixelPosition,origin);
            const auto pixelLocal=rotateSceneVector({-q.x,-q.y,-q.z,q.w},{delta[0],delta[1],delta[2]});
            if(!std::isfinite(pixelDistance) ||pixelDistance<=0 ||!finite(pixelPosition)
                ||-pixelLocal.z<frame.camera.nearPlane ||-pixelLocal.z>frame.camera.farPlane
                ||!nearPower(r.raster.linearDepth,-pixelLocal.z))
                return refuse("Grating camera native raster pixel-center depth changed");
            const float pixelTolerance=8*bias(pixelPosition);
            for(unsigned edge=0;edge<3;++edge) {
                const auto side=sub(triangle->points[(edge+1)%3],triangle->points[edge]);
                if(dot(cross(side,sub(pixelPosition,triangle->points[edge])),triangle->normal)
                    < -pixelTolerance*std::sqrt(dot(side,side)))
                    return refuse("Grating camera raster pixel center has a different triangle owner");
            }
            const bool visible=sameTriangle(event.hit,hit);
            const auto outcome=visible?NativeGratingCameraReadback::Outcome::Visible:NativeGratingCameraReadback::Outcome::Occluded;
            const double radiance=visible?measured->integratedRadiance/integral.solidAngleSteradians:0;
            if(r.outcome!=outcome ||!nearDirectionalSourcePotential(r.spectralRadiance,radiance)
                ||r.spectralRadiance>surfacematerial::kMaximumLinearColorComponent)
                return refuse("Grating camera native visibility mask or private irradiance-to-radiance conversion changed");
            queries+=hit.work;
        }
    }
    if(count!=receipt.orders.size() ||queries!=receipt.queryWork ||queries>cameraBounds.queryWork)
        return refuse("Grating camera native order or query accounting is incomplete");
    if(receipt.physicalColor)
        return refuse("Grating camera Color requires exact finite receiving-cell coverage, unavailable in receipt v1");
    // The released eight-band gate still accepts receiver-rho deposits only.
    // No private grating radiance enters that sum until finite-cell ownership
    // and a complete native base draw have their own validated authority.
    error.clear();return true;
}
inline bool decodeGratingDepositExtension(const Texel& extension,const Texel& reserved,NativeDeposit& output)
{
    if(reserved!=Texel{}||extension[3]!=0)return false;
    if(extension==Texel{}){output.gratingVersion=0;output.gratingIncident=UINT32_MAX;output.gratingOrder=0;return true;}
    std::uint32_t incident=0,order=0;
    if(extension[0]!=kGratingIncidentVersion||!exactReceiptInteger(extension[1],kMaximumEntrySamples*8*kMaximumBranchVisits,incident)
        ||!exactReceiptInteger(extension[2],2,order))return false;
    output.gratingVersion=kGratingIncidentVersion;output.gratingIncident=incident;output.gratingOrder=int(order)-1;return true;
}
// Coupled smooth children carry exact binary interface ancestry in the
// existing32-byte ancestry reservation, separately from private v2 walk tokens.
inline bool decodeGratingDepositAncestry(const std::array<Texel,8>& packed,std::array<std::uint8_t,32>& output)
{
    std::array<std::uint8_t,32> candidate{};
    for(unsigned i=0;i<32;++i) {
        std::uint32_t value=0;
        if(!exactReceiptInteger(packed[i/4][i%4],65,value)||(value!=0&&value!=1&&value!=65))return false;
        candidate[i]=std::uint8_t(value);
    }
    output=candidate;return true;
}
// Full child proof belongs to the existing path/deposition ledger. This helper
// checks exact native links and contributes no independent rendering or ledger.
inline bool validateGratingTerminalReadback(const GratingIncidentBinding& binding,const NativeReadback& native,
    const SubmissionIdentity& expected,std::string& error,const NativeDirectionalSourceBinding* sourceBinding)
{
    const auto refuse=[&](){error="Glass to grating terminal child proof is incomplete or nonconserving";return false;};
    if(!native.gratingIncidents||native.gratingIncidentVersion!=kGratingIncidentVersion)return refuse();
    const auto& batch=*native.gratingIncidents;
    if(!validateGratingIncidentReadback(binding,batch,expected,error,sourceBinding))return false;
    if(!batch.childrenComplete||batch.paths.size()!=binding.frame.work.size()||batch.children.size()>binding.bounds.maximumOrders)return refuse();
    const auto& allowance=binding.frame.query.texels[std::size_t(binding.frame.query.control[0])-1];
    const auto queryMaximum=std::uint64_t(allowance[0])+(std::uint64_t(allowance[1])<<16);
    std::set<std::pair<std::uint32_t,int>> children;
    std::vector<double> tails(binding.frame.work.size()),escapes(binding.frame.work.size());
    std::vector<std::uint64_t> queries(binding.frame.work.size());
    std::vector<unsigned> minimumVisits(binding.frame.work.size());
    for(const auto& incident:batch.incidents) {
        tails[incident.work]+=incident.unresolvedPower;++minimumVisits[incident.work];
        for(const auto& deposit:native.deposits)
            if(!deposit.gratingVersion&&deposit.work==incident.work&&deposit.interfaces==incident.interfaces
                &&deposit.reflectionHistory==incident.reflectionHistory)return refuse();
    }
    for(const auto& child:batch.children) {
        if(!child.complete||child.incident>=batch.incidents.size()||child.hit.work>queryMaximum
            ||!children.emplace(child.incident,child.order).second)return refuse();
        const auto& parent=batch.incidents[child.incident];
        const auto order=std::find_if(parent.orders.begin(),parent.orders.begin()+parent.orderCount,[&](const auto& o){return o.order==child.order;});
        if(order==parent.orders.begin()+parent.orderCount||!std::isfinite(child.escapedPower)||child.escapedPower<0
            ||!std::isfinite(child.unresolvedPower)||child.unresolvedPower<0)return refuse();
        std::size_t matches=0;const NativeDeposit* deposited=nullptr;
        for(const auto& deposit:native.deposits)if(deposit.gratingVersion&&deposit.gratingIncident==child.incident&&deposit.gratingOrder==child.order)
        {++matches;deposited=&deposit;}
        const auto work=parent.work;
        if(child.disposition==NativeGratingReadback::Disposition::Receiver) {
            if(matches!=1||child.hit.status!=1||!child.hit.work||child.hit.objectId==binding.gratingOwner->object.value
                ||!receiverTriangle(binding.frame,child.hit)||!child.hit.entering
                ||deposited->gratingVersion!=kGratingIncidentVersion||deposited->work!=work||deposited->interfaces!=parent.interfaces
                ||deposited->reflectionHistory!=parent.reflectionHistory||deposited->ancestry!=parent.ancestry
                ||!sameTriangle(deposited->receiver,child.hit)||!sameGratingPoint(deposited->receiver.position,child.hit.position)
                ||deposited->receiver.work!=child.hit.work||!nearPower(deposited->incidentPower,order->power)
                ||!nearPower(deposited->fraction,order->power/binding.frame.work[work].incidentPower)
                ||child.escapedPower||child.unresolvedPower)return refuse();
            const auto extra=std::uint64_t(child.hit.work)+(deposited->footprint.projected?std::uint64_t(deposited->cameraHit.work)+deposited->pixelHit.work:0);
            if(child.queryWork!=extra)return refuse();
            ++minimumVisits[work];
        } else if(child.disposition==NativeGratingReadback::Disposition::Escaped) {
            if(matches||child.hit.status!=0||!child.hit.work||!std::isfinite(child.hit.distance)||child.hit.distance<kRayMinimum
                ||child.queryWork!=child.hit.work
                ||!nearPower(child.hit.distance,binding.frame.sourceDirectionTrace[3])
                ||!nearPower(child.escapedPower,order->power)||child.unresolvedPower)return refuse();
            ++minimumVisits[work];
        } else if(child.disposition==NativeGratingReadback::Disposition::Unresolved) {
            if(matches||child.queryWork||child.hit.work||child.hit.status!=0||child.hit.distance!=0
                ||child.hit.position!=Point{}||child.hit.outwardNormal!=Point{}||child.hit.entering
                ||child.hit.objectId||child.hit.materialId||child.hit.instanceId||child.hit.triangleId||child.escapedPower
                ||!nearPower(child.unresolvedPower,order->power))return refuse();
        } else return refuse();
        if(child.hit.status==1) {
            if(!hitTriangle(binding.frame,child.hit)||!std::isfinite(child.hit.distance)||child.hit.distance<kRayMinimum
                ||child.hit.distance>binding.frame.sourceDirectionTrace[3]
                ||!sameGratingPoint(advance(advance(parent.hit.position,parent.normal,bias(parent.hit.position)),order->direction,child.hit.distance),
                    child.hit.position,8*bias(child.hit.position)))return refuse();
        }
        queries[work]+=child.queryWork;tails[work]+=child.unresolvedPower;escapes[work]+=child.escapedPower;
    }
    for(std::size_t i=0;i<batch.incidents.size();++i)
        for(unsigned o=0;o<batch.incidents[i].orderCount;++o)
            if(!children.count({std::uint32_t(i),batch.incidents[i].orders[o].order}))return refuse();
    for(const auto& deposit:native.deposits) {
        if(deposit.receiver.objectId==binding.gratingOwner->object.value)return refuse();
        if(deposit.gratingVersion&&!children.count({deposit.gratingIncident,deposit.gratingOrder}))return refuse();
    }
    for(std::size_t w=0;w<queries.size();++w) {
        if(w>=native.paths.size()||batch.paths[w].queryWork!=queries[w]||!nearPower(batch.paths[w].unresolvedPower,tails[w])
            ||native.paths[w].visits<minimumVisits[w]||escapes[w]>native.paths[w].escapedPower*(1+2e-4))return refuse();
    }
    error.clear();return true;
}
// Private terminal scattering math. Irradiance fields use W/(m^2 nm).
// No camera fraction, solid angle or area conversion.
struct DirectGratingTerminalBand
{
    std::uint32_t source=0,endpoint=0,band=0,witness=0;
    diffractionmaterial::DiffractionLocalFrame frame;
    std::array<double,3> toSource{};
    std::array<double,2> uv{};
    double wavelengthNanometres=0,incidentIrradiance=0;
    double substrateReflectedPower=0,resolvedReflectedPower=0,unresolvedReflectedPower=0; // Dimensionless shares.
    double substrateReflectedIrradiance=0,resolvedReflectedIrradiance=0,unresolvedReflectedIrradiance=0,absorbedIrradiance=0;
    std::size_t firstOrder=0,orderCount=0;
    bool responseEvaluated=false; // Nonincident supported zero has no orders.
};
struct DirectGratingTerminalOrder
{
    std::uint32_t source=0,endpoint=0,band=0;
    diffractionmaterial::DiffractionOrder order; // Exact signed order/direction/power.
    double orderIrradiance=0;
};
struct DirectGratingTerminalOrderBounds
{
    std::uint64_t bytes=0,combinedWork=0,depositionWork=0,maximumBands=0,maximumOrders=0;
};
struct DirectGratingTerminalOrderLedger
{
    std::shared_ptr<const NativeDirectGratingTerminalResult> nativeResult;
    DirectGratingIrradianceCohortReadback incident;
    // Retain authenticated source directions and exact native endpoint
    // witnesses alongside the frozen IR. This math result grants no new lease.
    std::vector<NativeDirectGratingContributionReadback> sources;
    std::vector<DirectGratingTerminalBand> bands;
    std::vector<DirectGratingTerminalOrder> orders;
    DirectGratingTerminalOrderBounds bounds;
};
// A sealed retained payload, created only through the native owner. The owner
// address is association metadata, not a lifetime grant. Never dereference it
// without that live frame's registered lease/result inspection. The frame and
// caller share this receipt, not the mutable returned ledger containers.
class NativeDirectGratingTerminalResult final
{
public:
    ~NativeDirectGratingTerminalResult()=default;
    const NativeFixtureSceneFrame* owner() const noexcept{return owner_;}
    const DirectGratingTerminalOrderLedger& retained() const noexcept{return retained_;}
private:
    NativeDirectGratingTerminalResult(const NativeDirectGratingTerminalResult&)=delete;
    NativeDirectGratingTerminalResult& operator=(const NativeDirectGratingTerminalResult&)=delete;
    NativeDirectGratingTerminalResult(const NativeFixtureSceneFrame& owner,const DirectGratingTerminalOrderLedger& value)
        :owner_(&owner),retained_(value){retained_.nativeResult.reset();}
    const NativeFixtureSceneFrame* owner_;
    DirectGratingTerminalOrderLedger retained_;
    friend class NativeDirectGratingIrradianceCohortBinding;
};
inline std::shared_ptr<const NativeDirectGratingTerminalResult>
NativeDirectGratingIrradianceCohortBinding::makeDirectCohortTerminalResult(const DirectGratingTerminalOrderLedger& value) const
{return std::shared_ptr<const NativeDirectGratingTerminalResult>(new NativeDirectGratingTerminalResult(cohortOwner(),value));}
inline bool calculateDirectGratingTerminalOrderBounds(const DirectGratingIrradianceCohort& cohort,
    const std::vector<NativeDirectGratingContributionReadback>& sources,
    const DirectGratingTerminalNativeBudget& native,const DirectGratingTerminalOrderLedger& incumbent,
    DirectGratingTerminalOrderBounds& output,std::string& error)
{
    const auto refuse=[&](){error="Directional terminal orders exceed authenticated whole-frame bounds";return false;};
    const auto maximum=renderpassoutput::kMaximumTotalBytes;
    if(cohort.sources().empty()||cohort.sources().size()>16||cohort.inputs().empty()||cohort.inputs().size()>kMaximumEntrySamples
        ||sources.size()!=cohort.sources().size()||sources.capacity()>2*sources.size()
        ||native.retainedBytes<cohort.bounds().bytes||native.retainedBytes>maximum
        ||native.combinedWork<cohort.bounds().combinedWork||native.combinedWork>kMaximumQueryWork
        ||native.depositionWork<cohort.bounds().depositionWork||native.depositionWork>kMaximumDepositionWork)
        return refuse();
    DirectGratingTerminalOrderBounds b;b.bytes=native.retainedBytes;
    b.maximumBands=cohort.sources().size()*cohort.inputs().size()*8;b.maximumOrders=b.maximumBands*kGratingOrdersPerIncident;
    const auto add=[&](std::uint64_t count,std::uint64_t stride=1) {
        if(count>(maximum-b.bytes)/stride){return false;}b.bytes+=count*stride;return true;
    };
    // Reserve validation's frozen-owner copies, all input/decoded/result copies,
    // and worst-case vector capacities before the accumulator allocates its sum.
    if(!add(2,cohort.bounds().bytes)||!add(3,sizeof(DirectGratingTerminalOrderLedger))
        ||!add(sizeof(NativeDirectGratingTerminalResult)+128) // Retained owner/control block.
        ||!add(2,sizeof(diffractionmaterial::DiffractionOrderDistribution))
        ||!add(4*b.maximumBands,sizeof(DirectGratingTerminalBand))||!add(4*b.maximumOrders,sizeof(DirectGratingTerminalOrder))
        ||!add(8*sources.size(),sizeof(NativeDirectGratingContributionReadback))
        ||!add(8*cohort.inputs().size(),sizeof(DirectGratingEndpointIrradianceSum))
        ||!add(8,cohort.identity().size()))return refuse();
    for(std::size_t s=0;s<sources.size();++s) {
        const auto& r=sources[s];
        if(!cohort.sources()[s]||r.samples.size()!=cohort.inputs().size()||r.samples.capacity()>2*r.samples.size()
            ||r.submission.frameIdentity.capacity()>2*cohort.sources()[s]->identity.size()
            ||!add(8*r.samples.size(),sizeof(NativeDirectGratingContributionSample))
            ||!add(8,cohort.sources()[s]->identity.size()))return refuse();
    }
    if(!add(incumbent.sources.capacity(),sizeof(NativeDirectGratingContributionReadback))
        ||!add(incumbent.bands.capacity(),sizeof(DirectGratingTerminalBand))
        ||!add(incumbent.orders.capacity(),sizeof(DirectGratingTerminalOrder))
        ||!add(incumbent.incident.endpoints.capacity(),sizeof(DirectGratingEndpointIrradianceSum))
        ||!add(incumbent.incident.submission.frameIdentity.capacity()))return refuse();
    for(const auto& r:incumbent.sources)
        if(!add(r.samples.capacity(),sizeof(NativeDirectGratingContributionSample))||!add(r.submission.frameIdentity.capacity()))return refuse();
    if(incumbent.incident.cohort&&incumbent.incident.cohort.get()!=&cohort
        &&!add(incumbent.incident.cohort->bounds().bytes))return refuse();
    if(incumbent.nativeResult) {
        const auto& retained=incumbent.nativeResult->retained();
        if(!add(sizeof(NativeDirectGratingTerminalResult)+128)
            ||!add(retained.sources.capacity(),sizeof(NativeDirectGratingContributionReadback))
            ||!add(retained.bands.capacity(),sizeof(DirectGratingTerminalBand))
            ||!add(retained.orders.capacity(),sizeof(DirectGratingTerminalOrder))
            ||!add(retained.incident.endpoints.capacity(),sizeof(DirectGratingEndpointIrradianceSum))
            ||!add(retained.incident.submission.frameIdentity.capacity()))return refuse();
        for(const auto& r:retained.sources)if(!add(r.samples.capacity(),sizeof(NativeDirectGratingContributionSample))
            ||!add(r.submission.frameIdentity.capacity()))return refuse();
        if(retained.incident.cohort&&retained.incident.cohort.get()!=&cohort
            &&retained.incident.cohort!=incumbent.incident.cohort&&!add(retained.incident.cohort->bounds().bytes))return refuse();
    }
    const auto work=b.maximumBands+b.maximumOrders;
    if(work>kMaximumQueryWork-native.combinedWork||b.maximumOrders>kMaximumDepositionWork-native.depositionWork)return refuse();
    b.combinedWork=native.combinedWork+work;b.depositionWork=native.depositionWork+b.maximumOrders;
    output=b;error.clear();return true;
}
inline bool evaluateDirectGratingTerminalOrders(const std::shared_ptr<const DirectGratingIrradianceCohort>& cohort,
    const NativeFixtureSceneFrame& owner,const SubmissionIdentity& expected,double evaluationTime,
    const std::vector<NativeDirectGratingContributionReadback>& sources,
    DirectGratingTerminalOrderLedger& output,std::string& error)
{
    const auto refuse=[&](const char* message){error=message;return false;};
    const auto* native=dynamic_cast<const NativeDirectGratingIrradianceCohortBinding*>(&owner);
    if(!cohort||!native||&native->cohortOwner()!=&owner||native->directCohort()!=cohort)
        return refuse("Directional terminal orders require the exact common native cohort");
    DirectGratingTerminalNativeBudget budget;DirectGratingTerminalOrderBounds bounds;
    std::unique_ptr<NativeDirectGratingTerminalLease> lease;
    if(!native->beginDirectCohortTerminalLease(expected,evaluationTime,sources,lease,budget,error))return false;
    if(!lease||&lease->owner()!=&owner)return refuse("Directional terminal lease belongs to another native owner");
    if(!calculateDirectGratingTerminalOrderBounds(*cohort,sources,budget,output,bounds,error)
        ||!native->reserveDirectCohortTerminalLease(*lease,bounds,output,error))return false;
    DirectGratingTerminalOrderLedger candidate;
    // First physical operation: authenticate the complete roster, actual
    // buffers/results, current frame/time/fences and ordered endpoint witnesses.
    if(!accumulateDirectGratingIrradianceCohort(cohort,owner,expected,evaluationTime,sources,candidate.incident,error))return false;
    candidate.sources=sources;candidate.bounds=bounds;
    candidate.bands.reserve(bounds.maximumBands);candidate.orders.reserve(bounds.maximumOrders);
    const auto product=[](double e,double share,double& result) {
        result=e*share;return std::isfinite(share)&&share>=0&&std::isfinite(result)&&result>=0
            &&(!(e>0&&share>0)||result>0);
    };
    const auto conserved=[](double a,double b) {
        return std::isfinite(a)&&std::isfinite(b)&&a>=0&&b>=0
            &&std::abs(a-b)<=2e-10*std::max(a,b);
    };
    for(std::size_t s=0;s<sources.size();++s)for(std::size_t e=0;e<cohort->inputs().size();++e) {
        const auto& binding=*cohort->sources()[s];const auto& sample=sources[s].samples[e];
        const auto witness=std::find_if(binding.triangles.begin(),binding.triangles.end(),[&](const auto& t) {
            return std::tie(t.object,t.material,t.instance,t.triangle)
                ==std::tie(sample.endpoint.objectId,sample.endpoint.materialId,sample.endpoint.instanceId,sample.endpoint.triangleId);
        });
        if(witness==binding.triangles.end()||!binding.material
            ||binding.material->description().spectrum.wavelengthsNanometres!=diffractionmaterial::kProductionWavelengthsNanometres)
            return refuse("Directional terminal material or endpoint witness is unsupported");
        DirectGratingTerminalBand base;base.source=std::uint32_t(s);base.endpoint=std::uint32_t(e);
        base.witness=std::uint32_t(witness-binding.triangles.begin());
        if(!gratingCameraOpticalFrame(witness->tangent,witness->bitangent,witness->normal,base.frame)
            ||!gratingCameraUnitDirection({sample.source.directionTrace[0],sample.source.directionTrace[1],sample.source.directionTrace[2]},base.toSource))
            return refuse("Directional terminal optical frame or source direction is invalid");
        for(unsigned corner=0;corner<3;++corner)for(unsigned a=0;a<2;++a)
            base.uv[a]+=double(sample.barycentric[corner])*witness->uv[corner][a];
        for(double coordinate:base.uv)if(!std::isfinite(coordinate)||coordinate<0||coordinate>1)
            return refuse("Directional terminal material coordinate is invalid");
        double facing=0;for(unsigned a=0;a<3;++a)facing+=base.toSource[a]*base.frame.normal[a];
        for(unsigned band=0;band<8;++band) {
            auto row=base;row.band=band;row.wavelengthNanometres=diffractionmaterial::kProductionWavelengthsNanometres[band];
            row.incidentIrradiance=sample.surfaceIrradiance[band];row.firstOrder=candidate.orders.size();
            if(!std::isfinite(row.incidentIrradiance)||row.incidentIrradiance<0)return refuse("Directional terminal irradiance is invalid");
            if(facing<=1e-6) {
                // The released evaluator has no response outside its incident
                // hemisphere. Preserve a complete zero band without inventing
                // an order direction or extrapolating that physical authority.
                if(row.incidentIrradiance!=0)return refuse("Directional terminal positive grazing response is unsupported");
                candidate.bands.push_back(row);continue;
            }
            diffractionmaterial::DiffractionOrderDistribution distribution;
            if(!diffractionmaterial::evaluateDiffractionOrders(*binding.material,row.wavelengthNanometres,row.frame,row.toSource,row.uv,distribution,error))return false;
            if(!distribution.orderCount||distribution.orderCount>kGratingOrdersPerIncident
                ||!std::isfinite(distribution.substrateReflectedPower)||distribution.substrateReflectedPower<0||distribution.substrateReflectedPower>1
                ||!conserved(distribution.resolvedReflectedPower+distribution.unresolvedReflectedPower,distribution.substrateReflectedPower))
                return refuse("Directional terminal order distribution violates energy bounds");
            row.responseEvaluated=true;row.orderCount=distribution.orderCount;
            row.substrateReflectedPower=distribution.substrateReflectedPower;row.resolvedReflectedPower=distribution.resolvedReflectedPower;
            row.unresolvedReflectedPower=distribution.unresolvedReflectedPower;
            if(!product(row.incidentIrradiance,row.substrateReflectedPower,row.substrateReflectedIrradiance)
                ||!product(row.incidentIrradiance,row.resolvedReflectedPower,row.resolvedReflectedIrradiance)
                ||!product(row.incidentIrradiance,row.unresolvedReflectedPower,row.unresolvedReflectedIrradiance)
                ||!product(row.incidentIrradiance,1-row.substrateReflectedPower,row.absorbedIrradiance))
                return refuse("Directional terminal energy is nonfinite or underflows");
            double resolved=0;
            for(std::size_t o=0;o<distribution.orderCount;++o) {
                const auto& order=distribution.orders[o];DirectGratingTerminalOrder result{row.source,row.endpoint,band,order,0};
                if(order.primary<-1||order.primary>1||order.secondary||order.angularStandardDeviationRadians!=0
                    ||!product(row.incidentIrradiance,order.reflectedPower,result.orderIrradiance))
                    return refuse("Directional terminal admitted order or energy is invalid");
                resolved+=result.orderIrradiance;candidate.orders.push_back(result);
            }
            if(!conserved(resolved,row.resolvedReflectedIrradiance)
                ||!conserved(resolved+row.unresolvedReflectedIrradiance,row.substrateReflectedIrradiance)
                ||!conserved(row.substrateReflectedIrradiance+row.absorbedIrradiance,row.incidentIrradiance))
                return refuse("Directional terminal per-area energy is not conserved");
            candidate.bands.push_back(row);
        }
    }
    if(candidate.bands.size()!=bounds.maximumBands||candidate.bands.capacity()>2*bounds.maximumBands
        ||candidate.orders.size()>bounds.maximumOrders||candidate.orders.capacity()>2*bounds.maximumOrders
        ||candidate.sources.capacity()>2*sources.size()||candidate.incident.endpoints.capacity()>2*cohort->inputs().size())
        return refuse("Directional terminal candidate capacities exceed admission");
    for(const auto& r:candidate.sources)if(r.samples.capacity()>2*cohort->inputs().size()
        ||r.submission.frameIdentity.capacity()>2*cohort->sources()[&r-candidate.sources.data()]->identity.size())
        return refuse("Directional terminal source copy capacities exceed admission");
    if(!native->finishDirectCohortTerminalLease(*lease,candidate,candidate.nativeResult,error))return false;
    if(!candidate.nativeResult||candidate.nativeResult->owner()!=&owner)
        return refuse("Directional terminal result has no exact retained native owner");
    const auto& retained=candidate.nativeResult->retained();
    if(retained.nativeResult||retained.sources.size()!=sources.size()||retained.sources.capacity()>2*sources.size()
        ||retained.bands.size()!=candidate.bands.size()||retained.bands.capacity()>2*bounds.maximumBands
        ||retained.orders.size()!=candidate.orders.size()||retained.orders.capacity()>2*bounds.maximumOrders
        ||retained.incident.endpoints.size()!=cohort->inputs().size()||retained.incident.endpoints.capacity()>2*cohort->inputs().size()
        ||retained.incident.submission.frameIdentity.capacity()>2*cohort->identity().size())
        return refuse("Directional terminal retained copy exceeds its admitted capacities");
    for(std::size_t s=0;s<retained.sources.size();++s)if(retained.sources[s].samples.size()!=cohort->inputs().size()
        ||retained.sources[s].samples.capacity()>2*cohort->inputs().size()
        ||retained.sources[s].submission.frameIdentity.capacity()>2*cohort->sources()[s]->identity.size())
        return refuse("Directional terminal retained source copy exceeds admission");
    if(!native->validateDirectCohortTerminalResult(*lease,candidate,error))return false;
    lease->commit();output=std::move(candidate);error.clear();return true;
}
// Receiver attribution is separate from scattering and finite camera-cell
// geometry. It grants no visibility, coverage, radiance or Color capability.
inline constexpr std::uint32_t kDirectGratingTerminalReceiverVersion=1;
struct alignas(16) NativeDirectGratingTerminalReceiverInput
{
    std::array<std::uint32_t,4> key{}; // source,endpoint,band,exact emitted index
    std::array<std::int32_t,4> order{}; // signed primary,secondary,version,zero
    Texel originMinimum{},directionMaximum{};
};
static_assert(sizeof(NativeDirectGratingTerminalReceiverInput)==4*sizeof(Texel));
struct DirectGratingTerminalReceiverRow
{
    enum class Disposition : std::uint32_t { Miss=0,NonReceiver=1,Offscreen=2,Pixel=3 };
    enum class RasterSample : std::uint32_t { Absent=0,Background=1,Surface=2 };
    NativeDirectGratingTerminalReceiverInput input;
    DirectGratingTerminalOrder emitted; // Exact retained order, including energy.
    opaquevisibility::ClosestHit hit;
    Disposition disposition=Disposition::Miss;
    PixelFootprint pixel; // worldArea remains zero; only point projection.
    RasterSample rasterSample=RasterSample::Absent;
    RasterReceiver raster; // Actual pixel-center owner/depth, possibly occluding.
    std::uint64_t queryWork=0;
};
struct DirectGratingTerminalReceiverNativeBudget
{
    std::uint64_t retainedBytes=0,combinedWork=0;
    // Actual byte strides summed across every native input/output allocation
    // and additional decoded inspection copies beyond the three shared row
    // copies below. Fixed bytes include native lease/attempt,
    // dispatch scratch and retained owner metadata. The owner reconstructs them.
    std::uint64_t nativeInputBytesPerRow=0,nativeOutputBytesPerRow=0,nativeFixedBytes=0;
};
struct DirectGratingTerminalReceiverBounds
{
    std::uint64_t bytes=0,combinedWork=0,queryWork=0,maximumRows=0,perRayWork=0;
    std::uint64_t nativeInputBytes=0,nativeOutputBytes=0,nativeFixedBytes=0;
};
struct DirectGratingTerminalReceiverReadback
{
    std::shared_ptr<const NativeDirectGratingTerminalReceiverResult> nativeResult;
    std::shared_ptr<const NativeDirectGratingTerminalResult> terminal;
    SubmissionIdentity submission;
    double evaluationTime=0;
    std::uint64_t inputGeneration=0,queryWork=0;
    DirectGratingTerminalReceiverBounds bounds;
    std::vector<DirectGratingTerminalReceiverRow> rows;
};
inline bool bindDirectGratingTerminalReceivers(const std::shared_ptr<const NativeDirectGratingTerminalResult>&,
    const NativeFixtureSceneFrame&,const SubmissionIdentity&,double,
    DirectGratingTerminalReceiverReadback&,std::string&);
class NativeDirectGratingTerminalReceiverLease
{
public:
    virtual ~NativeDirectGratingTerminalReceiverLease()=default;
    virtual const NativeFixtureSceneFrame& owner() const noexcept=0;
    // The exact first source's immutable Frame supplies scene triangles,
    // camera/extent and query texels/control/allowance to both native writers.
    // Begin must authenticate it against the actual common native resources.
    virtual const Frame& frozenFrame() const noexcept=0;
    virtual std::uint64_t inputGeneration() const noexcept=0;
protected:
    NativeDirectGratingTerminalReceiverLease()=default;
    NativeDirectGratingTerminalReceiverLease(const NativeDirectGratingTerminalReceiverLease&)=delete;
    NativeDirectGratingTerminalReceiverLease& operator=(const NativeDirectGratingTerminalReceiverLease&)=delete;
    virtual void commit() noexcept=0; // Publication only; no allocation/failure.
private:
    friend bool bindDirectGratingTerminalReceivers(const std::shared_ptr<const NativeDirectGratingTerminalResult>&,
        const NativeFixtureSceneFrame&,const SubmissionIdentity&,double,
        DirectGratingTerminalReceiverReadback&,std::string&);
};
class NativeDirectGratingTerminalReceiverResult final
{
public:
    ~NativeDirectGratingTerminalReceiverResult()=default;
    const NativeFixtureSceneFrame* owner() const noexcept{return owner_;}
    const DirectGratingTerminalReceiverReadback& retained() const noexcept{return retained_;}
private:
    NativeDirectGratingTerminalReceiverResult(const NativeDirectGratingTerminalReceiverResult&)=delete;
    NativeDirectGratingTerminalReceiverResult& operator=(const NativeDirectGratingTerminalReceiverResult&)=delete;
    NativeDirectGratingTerminalReceiverResult(const NativeFixtureSceneFrame& owner,
        const DirectGratingTerminalReceiverReadback& value):owner_(&owner),retained_(value){retained_.nativeResult.reset();}
    const NativeFixtureSceneFrame* owner_;
    DirectGratingTerminalReceiverReadback retained_;
    friend class NativeDirectGratingIrradianceCohortBinding;
};
inline std::shared_ptr<const NativeDirectGratingTerminalReceiverResult>
NativeDirectGratingIrradianceCohortBinding::makeDirectTerminalReceiverResult(const DirectGratingTerminalReceiverReadback& value) const
{return std::shared_ptr<const NativeDirectGratingTerminalReceiverResult>(new NativeDirectGratingTerminalReceiverResult(cohortOwner(),value));}
inline bool sameDirectGratingTerminalReceiverInput(const NativeDirectGratingTerminalReceiverInput& a,
    const NativeDirectGratingTerminalReceiverInput& b)
{return a.key==b.key&&a.order==b.order&&a.originMinimum==b.originMinimum&&a.directionMaximum==b.directionMaximum;}
inline bool sameDirectGratingTerminalEmittedOrder(const DirectGratingTerminalOrder& a,const DirectGratingTerminalOrder& b)
{
    return std::tie(a.source,a.endpoint,a.band,a.order.primary,a.order.secondary,a.order.direction,a.order.reflectedPower,
        a.order.selectionProbability,a.order.angularStandardDeviationRadians,a.orderIrradiance)
        ==std::tie(b.source,b.endpoint,b.band,b.order.primary,b.order.secondary,b.order.direction,b.order.reflectedPower,
        b.order.selectionProbability,b.order.angularStandardDeviationRadians,b.orderIrradiance);
}
inline bool sameDirectGratingTerminalReceiverReadback(const DirectGratingTerminalReceiverReadback& a,
    const DirectGratingTerminalReceiverReadback& b,bool compareNativeResult=true)
{
    const auto& x=a.bounds;const auto& y=b.bounds;
    if((compareNativeResult&&a.nativeResult!=b.nativeResult)||a.terminal!=b.terminal||!(a.submission==b.submission)
        ||a.evaluationTime!=b.evaluationTime||a.inputGeneration!=b.inputGeneration||a.queryWork!=b.queryWork
        ||a.rows.size()!=b.rows.size()
        ||std::tie(x.bytes,x.combinedWork,x.queryWork,x.maximumRows,x.perRayWork,x.nativeInputBytes,x.nativeOutputBytes,x.nativeFixedBytes)
            !=std::tie(y.bytes,y.combinedWork,y.queryWork,y.maximumRows,y.perRayWork,y.nativeInputBytes,y.nativeOutputBytes,y.nativeFixedBytes))return false;
    for(std::size_t i=0;i<a.rows.size();++i) {
        const auto& p=a.rows[i];const auto& q=b.rows[i];const auto& h=p.hit;const auto& k=q.hit;
        if(!sameDirectGratingTerminalReceiverInput(p.input,q.input)||!sameDirectGratingTerminalEmittedOrder(p.emitted,q.emitted)
            ||std::tie(h.status,h.distance,h.position,h.outwardNormal,h.objectId,h.materialId,h.instanceId,h.triangleId,h.entering,h.work)
                !=std::tie(k.status,k.distance,k.position,k.outwardNormal,k.objectId,k.materialId,k.instanceId,k.triangleId,k.entering,k.work)
            ||std::tie(p.disposition,p.pixel.projected,p.pixel.x,p.pixel.y,p.pixel.linearDepth,p.pixel.worldArea,p.rasterSample,
                p.raster.object,p.raster.material,p.raster.instance,p.raster.triangle,p.raster.linearDepth,p.queryWork)
                !=std::tie(q.disposition,q.pixel.projected,q.pixel.x,q.pixel.y,q.pixel.linearDepth,q.pixel.worldArea,q.rasterSample,
                q.raster.object,q.raster.material,q.raster.instance,q.raster.triangle,q.raster.linearDepth,q.queryWork))return false;
    }
    return true;
}
inline bool directGratingTerminalReceiverAllowance(const Frame& frame,std::uint64_t& output)
{
    const auto& control=frame.query.control;std::uint32_t root=0,end=0,low=0,high=0,stack=0;
    if(control[3]!=2||control[2]!=opaquevisibility::kTextureWidth
        ||!exactReceiptInteger(control[0],UINT32_MAX,root)||!root
        ||!exactReceiptInteger(control[1],UINT32_MAX,end)||root>=end||end>frame.query.texels.size()
        ||end>double(frame.query.height)*opaquevisibility::kTextureWidth)return false;
    const auto& allowance=frame.query.texels[root-1];
    if(!exactReceiptInteger(allowance[0],65535,low)||!exactReceiptInteger(allowance[1],65535,high)
        ||!exactReceiptInteger(allowance[2],opaquevisibility::kClosestHitStackCapacity,stack)||!stack||allowance[3]!=3)return false;
    const auto value=std::uint64_t(low)+(std::uint64_t(high)<<16);
    if(!value||value>opaquevisibility::kMaximumClosestHitWork)return false;
    output=value;return true;
}
inline bool makeDirectGratingTerminalReceiverInput(const DirectGratingTerminalOrderLedger& terminal,
    const Frame& frame,std::size_t index,NativeDirectGratingTerminalReceiverInput& output,std::string& error)
{
    const auto refuse=[&](){error="Directional terminal receiver ray has no exact frozen scene bound";return false;};
    if(index>=terminal.orders.size()||index>UINT32_MAX||frame.triangles.empty())return refuse();
    const auto& emitted=terminal.orders[index];
    if(emitted.source>=terminal.sources.size()||emitted.endpoint>=terminal.incident.endpoints.size()||emitted.band>=8
        ||emitted.order.primary<-1||emitted.order.primary>1||emitted.order.secondary||emitted.order.angularStandardDeviationRadians!=0
        ||!std::isfinite(emitted.orderIrradiance)||emitted.orderIrradiance<0)return refuse();
    Point direction{};for(unsigned a=0;a<3;++a)direction[a]=float(emitted.order.direction[a]);
    const auto& endpoint=terminal.incident.endpoints[emitted.endpoint].endpoint;
    if(!finite(direction)||std::abs(dot(direction,direction)-1)>1e-4f||!hitTriangle(frame,endpoint)
        ||dot(direction,endpoint.outwardNormal)<=0)return refuse();
    const auto origin=advance(endpoint.position,endpoint.outwardNormal,bias(endpoint.position));
    if(!finite(origin))return refuse();
    double maximum=0;
    // Every possible triangle hit lies no farther than its farthest vertex.
    // This outgoing segment is independent of the incoming endpoint segment
    // and camera far plane. Padding uses the released world-position bias.
    for(const auto& triangle:frame.triangles)for(const auto& point:triangle.points) {
        if(!finite(point)){return refuse();}double square=0;
        for(unsigned a=0;a<3;++a){const double delta=double(point[a])-origin[a];square+=delta*delta;}
        maximum=std::max(maximum,std::sqrt(square)+8*double(bias(point)));
    }
    if(!std::isfinite(maximum)||maximum<=kRayMinimum||maximum>=std::numeric_limits<float>::max())return refuse();
    const float limit=std::nextafter(float(maximum),std::numeric_limits<float>::infinity());
    if(!std::isfinite(limit))return refuse();
    NativeDirectGratingTerminalReceiverInput candidate;
    candidate.key={emitted.source,emitted.endpoint,emitted.band,std::uint32_t(index)};
    candidate.order={emitted.order.primary,emitted.order.secondary,kDirectGratingTerminalReceiverVersion,0};
    candidate.originMinimum={origin[0],origin[1],origin[2],kRayMinimum};
    candidate.directionMaximum={direction[0],direction[1],direction[2],limit};
    output=candidate;error.clear();return true;
}
inline bool calculateDirectGratingTerminalReceiverBounds(const NativeDirectGratingTerminalResult& terminal,
    const Frame& frame,const DirectGratingTerminalReceiverNativeBudget& native,
    const DirectGratingTerminalReceiverReadback& incumbent,DirectGratingTerminalReceiverBounds& output,std::string& error)
{
    const auto refuse=[&](){error="Directional terminal receivers exceed authenticated whole-frame bounds";return false;};
    const auto maximum=renderpassoutput::kMaximumTotalBytes;const auto& scattering=terminal.retained();
    DirectGratingTerminalReceiverBounds b;b.bytes=native.retainedBytes;b.maximumRows=scattering.bounds.maximumOrders;
    if(!b.maximumRows||b.maximumRows>UINT32_MAX||scattering.orders.size()>b.maximumRows
        ||!scattering.incident.cohort||native.retainedBytes<scattering.bounds.bytes||native.retainedBytes>maximum
        ||native.combinedWork<scattering.bounds.combinedWork||native.combinedWork>kMaximumQueryWork
        ||native.nativeInputBytesPerRow<sizeof(NativeDirectGratingTerminalReceiverInput)
        ||native.nativeOutputBytesPerRow<8*sizeof(Texel)||!native.nativeFixedBytes
        ||!directGratingTerminalReceiverAllowance(frame,b.perRayWork)
        ||b.maximumRows>(kMaximumQueryWork-native.combinedWork)/b.perRayWork)return refuse();
    b.queryWork=b.maximumRows*b.perRayWork;b.combinedWork=native.combinedWork+b.queryWork;
    const auto add=[&](std::uint64_t count,std::uint64_t stride=1) {
        if(!stride||count>(maximum-b.bytes)/stride){return false;}b.bytes+=count*stride;return true;
    };
    // Native resources plus worst input capacities and three decoded row copies
    // for dispatch inspection, caller and immutable result. Reserve before
    // makeInput, vector growth or dispatch.
    if(!add(b.maximumRows,native.nativeInputBytesPerRow))return refuse();
    b.nativeInputBytes=b.maximumRows*native.nativeInputBytesPerRow;
    if(!add(b.maximumRows,native.nativeOutputBytesPerRow))return refuse();
    b.nativeOutputBytes=b.maximumRows*native.nativeOutputBytesPerRow;b.nativeFixedBytes=native.nativeFixedBytes;
    if(!add(b.nativeFixedBytes)||!add(2*b.maximumRows,sizeof(NativeDirectGratingTerminalReceiverInput))
        ||!add(6*b.maximumRows,sizeof(DirectGratingTerminalReceiverRow))
        ||!add(3,sizeof(DirectGratingTerminalReceiverReadback))
        ||!add(sizeof(NativeDirectGratingTerminalReceiverResult)+128)
        ||!add(4,scattering.incident.submission.frameIdentity.size())
        ||!add(incumbent.rows.capacity(),sizeof(DirectGratingTerminalReceiverRow))
        ||!add(incumbent.submission.frameIdentity.capacity()))return refuse();
    // Native retainedBytes covers its live frame/results. Caller-owned copies
    // and foreign frozen owners also stay live during replacement. Conservatively
    // charge a foreign terminal's entire admitted owner bound, without traversal.
    if(incumbent.terminal&&incumbent.terminal.get()!=&terminal&&!add(incumbent.terminal->retained().bounds.bytes))return refuse();
    if(incumbent.nativeResult) {
        const auto& retained=incumbent.nativeResult->retained();
        if(!add(sizeof(NativeDirectGratingTerminalReceiverResult)+128)
            ||!add(retained.rows.capacity(),sizeof(DirectGratingTerminalReceiverRow))
            ||!add(retained.submission.frameIdentity.capacity()))return refuse();
        if(incumbent.nativeResult->owner()!=terminal.owner()&&!add(retained.bounds.bytes))return refuse();
        if(retained.terminal&&retained.terminal.get()!=&terminal&&retained.terminal!=incumbent.terminal
            &&!add(retained.terminal->retained().bounds.bytes))return refuse();
    }
    output=b;error.clear();return true;
}
inline bool projectDirectGratingTerminalReceiver(const Frame& frame,const opaquevisibility::ClosestHit& hit,
    PixelFootprint& output,std::string& error)
{
    if(!nativeFixtureDimensionsWithinBounds(frame.width,frame.height)||!validFixtureCameraOverride(frame.camera)
        ||!receiverTriangle(frame,hit)){error="Directional terminal receiver has no frozen camera projection";return false;}
    const auto& q=frame.camera.transform.rotation;const auto p=sub(hit.position,cameraPosition(frame));
    const auto local=rotateSceneVector({-q.x,-q.y,-q.z,q.w},{p[0],p[1],p[2]});
    PixelFootprint candidate;candidate.linearDepth=-local.z;
    if(!std::isfinite(candidate.linearDepth)){error="Directional terminal receiver depth is nonfinite";return false;}
    if(candidate.linearDepth>=frame.camera.nearPlane&&candidate.linearDepth<=frame.camera.farPlane) {
        const float tangent=std::tan(frame.camera.verticalFovRadians*.5f);
        const float x=(local.x/(candidate.linearDepth*tangent*float(frame.width)/frame.height)+1)*.5f*frame.width;
        const float y=(1-local.y/(candidate.linearDepth*tangent))*.5f*frame.height;
        if(!std::isfinite(x)||!std::isfinite(y)){error="Directional terminal receiver projection is nonfinite";return false;}
        if(x>=0&&y>=0&&x<frame.width&&y<frame.height) {
            candidate.x=std::uint32_t(x);candidate.y=std::uint32_t(y);
            candidate.projected=dot(hit.outwardNormal,cameraRay(frame,float(candidate.x)+.5f,float(candidate.y)+.5f))<-1e-6f;
        }
    }
    output=candidate;error.clear();return true;
}
inline bool validateDirectGratingTerminalReceiverRows(const DirectGratingTerminalOrderLedger& terminal,const Frame& frame,
    const DirectGratingTerminalReceiverReadback& readback,std::string& error)
{
    const auto refuse=[&](){error="Directional terminal receiver native row, ray or raster attribution changed";return false;};
    using Row=DirectGratingTerminalReceiverRow;std::uint64_t allowance=0,observed=0;
    if(!directGratingTerminalReceiverAllowance(frame,allowance)||readback.bounds.perRayWork!=allowance
        ||readback.bounds.maximumRows!=terminal.bounds.maximumOrders||readback.rows.size()!=terminal.orders.size()
        ||readback.rows.capacity()>2*readback.bounds.maximumRows)return refuse();
    for(std::size_t i=0;i<readback.rows.size();++i) {
        const auto& row=readback.rows[i];const auto& hit=row.hit;NativeDirectGratingTerminalReceiverInput exact;
        if(!makeDirectGratingTerminalReceiverInput(terminal,frame,i,exact,error))return false;
        if(!sameDirectGratingTerminalReceiverInput(row.input,exact)||!sameDirectGratingTerminalEmittedOrder(row.emitted,terminal.orders[i])
            ||(hit.status!=0&&hit.status!=1)||!hit.work||hit.work>allowance||row.queryWork!=hit.work
            ||!std::isfinite(hit.distance)||hit.distance<exact.originMinimum[3]||hit.distance>exact.directionMaximum[3]
            ||!finite(hit.position)||!finite(hit.outwardNormal)||row.pixel.worldArea!=0)return refuse();
        if(row.queryWork>readback.bounds.queryWork-observed){return refuse();}observed+=row.queryWork;
        PixelFootprint pixel;auto disposition=Row::Disposition::Miss;
        if(hit.status==0) {
            if(hit.distance!=exact.directionMaximum[3]||hit.position!=Point{}||hit.outwardNormal!=Point{}
                ||hit.objectId||hit.materialId||hit.instanceId||hit.triangleId||hit.entering)return refuse();
        } else {
            const Point origin{exact.originMinimum[0],exact.originMinimum[1],exact.originMinimum[2]};
            const Point direction{exact.directionMaximum[0],exact.directionMaximum[1],exact.directionMaximum[2]};
            const auto expected=advance(origin,direction,hit.distance);const float tolerance=8*bias(hit.position);
            if(!hitTriangle(frame,hit)||!sameGratingPoint(expected,hit.position,tolerance)
                ||hit.entering!=(dot(direction,hit.outwardNormal)<0))return refuse();
            disposition=Row::Disposition::NonReceiver;
            if(receiverTriangle(frame,hit)) {
                if(!projectDirectGratingTerminalReceiver(frame,hit,pixel,error))return false;
                disposition=pixel.projected?Row::Disposition::Pixel:Row::Disposition::Offscreen;
            }
        }
        if(row.disposition!=disposition||row.pixel.projected!=pixel.projected||row.pixel.x!=pixel.x||row.pixel.y!=pixel.y
            ||!std::isfinite(row.pixel.linearDepth)||!nearPower(row.pixel.linearDepth,pixel.linearDepth))return refuse();
        const auto& raster=row.raster;
        if(!pixel.projected||row.rasterSample==Row::RasterSample::Background) {
            if(row.rasterSample!=(pixel.projected?Row::RasterSample::Background:Row::RasterSample::Absent)
                ||raster.object||raster.material||raster.instance||raster.triangle||raster.linearDepth!=0)return refuse();
            continue;
        }
        if(row.rasterSample!=Row::RasterSample::Surface||!raster.object||!raster.material
            ||!std::isfinite(raster.linearDepth)||raster.linearDepth<frame.camera.nearPlane
            ||raster.linearDepth>frame.camera.farPlane)return refuse();
        const auto triangle=std::find_if(frame.triangles.begin(),frame.triangles.end(),[&](const auto& t) {
            return std::tie(t.object,t.material,t.instance,t.triangle)==std::tie(raster.object,raster.material,raster.instance,raster.triangle);});
        if(triangle==frame.triangles.end())return refuse();
        const auto origin=cameraPosition(frame),ray=cameraRay(frame,float(pixel.x)+.5f,float(pixel.y)+.5f);
        const auto& q=frame.camera.transform.rotation;
        const auto local=rotateSceneVector({-q.x,-q.y,-q.z,q.w},{ray[0],ray[1],ray[2]});
        if(!finite(ray)||!std::isfinite(local.z)||local.z>=0)return refuse();
        opaquevisibility::ClosestHit center;center.status=1;center.distance=raster.linearDepth/-local.z;
        center.position=advance(origin,ray,center.distance);center.outwardNormal=triangle->normal;
        center.objectId=raster.object;center.materialId=raster.material;center.instanceId=raster.instance;center.triangleId=raster.triangle;
        // Pixel-center raster depth follows its own frozen triangle plane.
        // It need not equal the off-center order-hit depth on a slanted triangle.
        if(!hitTriangle(frame,center))return refuse();
    }
    if(readback.queryWork!=observed||observed>readback.bounds.queryWork)return refuse();
    error.clear();return true;
}
inline bool bindDirectGratingTerminalReceivers(const std::shared_ptr<const NativeDirectGratingTerminalResult>& terminal,
    const NativeFixtureSceneFrame& owner,const SubmissionIdentity& expected,double evaluationTime,
    DirectGratingTerminalReceiverReadback& output,std::string& error)
{
    const auto refuse=[&](const char* message){error=message;return false;};
    const auto* native=dynamic_cast<const NativeDirectGratingIrradianceCohortBinding*>(&owner);
    if(!terminal||!native||terminal->owner()!=&owner||&native->cohortOwner()!=&owner
        ||terminal->retained().incident.cohort!=native->directCohort()
        ||!native->directCohort()||native->directCohort()->sources().empty()
        ||!(terminal->retained().incident.submission==expected)||!std::isfinite(evaluationTime))
        return refuse("Directional terminal receivers require the exact live scattering owner");
    std::unique_ptr<NativeDirectGratingTerminalReceiverLease> lease;DirectGratingTerminalReceiverNativeBudget budget;
    if(!native->beginDirectTerminalReceiverLease(terminal,expected,evaluationTime,lease,budget,error))return false;
    const auto& frame=native->directCohort()->sources()[0]->frame;
    if(!lease||&lease->owner()!=&owner||&lease->frozenFrame()!=&frame||!lease->inputGeneration()
        ||frame.width!=owner.width()||frame.height!=owner.height()||!owner.nativeResourceCacheIdentity()
        ||evaluationTime!=native->directCohort()->sources()[0]->evaluationTime)
        return refuse("Directional terminal receiver lease lacks exact frozen native ownership");
    DirectGratingTerminalReceiverBounds bounds;
    if(!calculateDirectGratingTerminalReceiverBounds(*terminal,frame,budget,output,bounds,error)
        ||!native->reserveDirectTerminalReceiverLease(*lease,bounds,output,error))return false;
    DirectGratingTerminalReceiverReadback candidate;candidate.terminal=terminal;candidate.submission=expected;
    candidate.evaluationTime=evaluationTime;candidate.inputGeneration=lease->inputGeneration();candidate.bounds=bounds;
    std::vector<NativeDirectGratingTerminalReceiverInput> inputs;inputs.reserve(bounds.maximumRows);
    for(std::size_t i=0;i<terminal->retained().orders.size();++i) {
        NativeDirectGratingTerminalReceiverInput input;
        if(!makeDirectGratingTerminalReceiverInput(terminal->retained(),frame,i,input,error))return false;
        inputs.push_back(input);
    }
    if(inputs.capacity()>2*bounds.maximumRows)return refuse("Directional terminal receiver input capacity exceeds admission");
    if(!native->dispatchDirectTerminalReceivers(*lease,inputs,candidate,error))return false;
    // Dispatch may fill rows/work only. Frozen keys and budgets stay exact.
    const auto& b=candidate.bounds;
    if(candidate.nativeResult||candidate.terminal!=terminal||!(candidate.submission==expected)
        ||candidate.evaluationTime!=evaluationTime||candidate.inputGeneration!=lease->inputGeneration()
        ||candidate.submission.frameIdentity.capacity()>2*expected.frameIdentity.size()
        ||std::tie(b.bytes,b.combinedWork,b.queryWork,b.maximumRows,b.perRayWork,b.nativeInputBytes,b.nativeOutputBytes,b.nativeFixedBytes)
            !=std::tie(bounds.bytes,bounds.combinedWork,bounds.queryWork,bounds.maximumRows,bounds.perRayWork,
                bounds.nativeInputBytes,bounds.nativeOutputBytes,bounds.nativeFixedBytes))
        return refuse("Directional terminal receiver dispatch changed frozen identity or reservation");
    if(!validateDirectGratingTerminalReceiverRows(terminal->retained(),frame,candidate,error))return false;
    if(!native->validateDirectTerminalReceiverResult(*lease,candidate,error)
        ||!native->finishDirectTerminalReceiverLease(*lease,candidate,candidate.nativeResult,error))return false;
    if(!candidate.nativeResult||candidate.nativeResult->owner()!=&owner)
        return refuse("Directional terminal receiver result has no exact staged native owner");
    const auto& retained=candidate.nativeResult->retained();
    if(retained.nativeResult||retained.rows.capacity()>2*bounds.maximumRows
        ||retained.submission.frameIdentity.capacity()>2*expected.frameIdentity.size()
        ||!sameDirectGratingTerminalReceiverReadback(candidate,retained,false))
        return refuse("Directional terminal receiver retained copy differs or exceeds admission");
    if(!native->validateDirectTerminalReceiverResult(*lease,candidate,error))return false;
    lease->commit();output=std::move(candidate);error.clear();return true;
}
} // namespace arbitgpu::glasscaustics
