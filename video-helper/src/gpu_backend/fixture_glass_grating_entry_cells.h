#pragma once

#include "fixture_glass_grating_incident.h"
#include <limits>

namespace arbitgpu::glasscaustics
{
// Optional source geometry only. No finite branch, visibility or Color grant.
inline constexpr std::uint32_t kGratingEntryCellVersion=1;
inline constexpr std::size_t kGratingEntryCellStride=7;
struct GratingEntryCell
{
    std::uint32_t object=0,material=0,instance=0,triangle=0,quadrature=0,entry=0;
    std::array<Point,3> vertices{};
    std::array<std::array<float,2>,3> barycentrics{};
};
struct GratingEntryCellBounds
{
    std::uint64_t hostBytes=0,nativeUploadBytes=0,bytes=0;
};
struct GratingEntryCellBinding
{
    std::shared_ptr<const GratingIncidentBinding> incidentBinding;
    std::vector<GratingEntryCell> cells;
    std::vector<Texel> cellTexels;
    GratingEntryCellBounds bounds;
    std::string identity;
};
// Seven read-only texels per ENTRY, indexed through EntryWork.entry. Neither
// the existing work upload nor the target grating Witness buffer changes.
// 0..2 world vertex xyz/barycentric u;3 barycentric v0/v1/v2/sample area;
// 4 split object/material;5 instance/triangle/quadrature/entry;
// 6 private version/subdivisions/0/0.
inline std::array<Texel,kGratingEntryCellStride> gratingEntryCellTexels(
    const GratingEntryCell& cell,float area,std::size_t subdivisions)
{
    std::array<Texel,kGratingEntryCellStride> result{};
    for(unsigned v=0;v<3;++v)result[v]={cell.vertices[v][0],cell.vertices[v][1],cell.vertices[v][2],cell.barycentrics[v][0]};
    result[3]={cell.barycentrics[0][1],cell.barycentrics[1][1],cell.barycentrics[2][1],area};
    result[4]={float(cell.object&65535),float(cell.object>>16),float(cell.material&65535),float(cell.material>>16)};
    result[5]={float(cell.instance),float(cell.triangle),float(cell.quadrature),float(cell.entry)};
    result[6]={float(kGratingEntryCellVersion),float(subdivisions),0,0};return result;
}
inline bool reconstructGratingEntryCell(const Triangle& triangle,const EntrySample& sample,
    SchedulingDescription scheduling,std::uint32_t index,GratingEntryCell& output,std::string& error)
{
    const auto refuse=[&](const char* message){error=message;return false;};
    if(!validScheduling(scheduling)||index>=kMaximumEntrySamples||!triangle.glass
        ||std::tie(triangle.object,triangle.material,triangle.instance,triangle.triangle)
            !=std::tie(sample.object,sample.material,sample.instance,sample.triangle)
        ||sample.quadrature>=scheduling.entrySubdivisions*scheduling.entrySubdivisions
        ||triangle.instance>16777216u||triangle.triangle>16777216u
        ||!std::isfinite(triangle.area)||triangle.area<=0||!finite(triangle.normal)
        ||sample.normal!=triangle.normal)
        return refuse("Grating source cell schedule or triangle owner is invalid");
    const auto n=scheduling.entrySubdivisions;
    GratingEntryCell cell;cell.object=sample.object;cell.material=sample.material;
    cell.instance=sample.instance;cell.triangle=sample.triangle;cell.quadrature=sample.quadrature;cell.entry=index;
    std::uint32_t ordinal=0;float centerU=0,centerV=0;bool found=false;
    for(unsigned u=0;u<n;++u)for(unsigned v=0;v<n-u;++v) {
        if(ordinal++==sample.quadrature) {
            cell.barycentrics={{{float(u)/n,float(v)/n},{float(u+1)/n,float(v)/n},{float(u)/n,float(v+1)/n}}};
            centerU=(u+1.0f/3)/n;centerV=(v+1.0f/3)/n;found=true;
        }
        if(u+v+1<n&&ordinal++==sample.quadrature) {
            cell.barycentrics={{{float(u+1)/n,float(v)/n},{float(u+1)/n,float(v+1)/n},{float(u)/n,float(v+1)/n}}};
            centerU=(u+2.0f/3)/n;centerV=(v+2.0f/3)/n;found=true;
        }
    }
    const auto e1=sub(triangle.points[1],triangle.points[0]),e2=sub(triangle.points[2],triangle.points[0]);
    if(!found||!finite(triangle.points[0])||!finite(e1)||!finite(e2))
        return refuse("Grating source cell vertices are nonfinite");
    // Preserve appendEntrySamples' original operation order and centroid.
    const auto canonicalCenter=advance(advance(triangle.points[0],e1,centerU),e2,centerV);
    const float area=triangle.area/(n*n);
    if(!finite(canonicalCenter)||sample.position!=canonicalCenter||!std::isfinite(area)||area<=0||sample.area!=area)
        return refuse("Grating source cell centroid or quadrature area changed");
    double roundoff=0;
    for(unsigned a=0;a<3;++a)roundoff=std::max(roundoff,16*double(std::numeric_limits<float>::epsilon())
        *(std::abs(double(triangle.points[0][a]))+std::abs(double(e1[a]))+std::abs(double(e2[a]))));
    for(unsigned v=0;v<3;++v) {
        cell.vertices[v]=advance(advance(triangle.points[0],e1,cell.barycentrics[v][0]),e2,cell.barycentrics[v][1]);
        if(!finite(cell.vertices[v]))return refuse("Grating source cell vertex is nonfinite");
    }
    // Averaging rounded vertices need not be bit-identical to the separately
    // evaluated sample. This bound covers float affine evaluation, not rays.
    for(unsigned a=0;a<3;++a) {
        const double average=(double(cell.vertices[0][a])+cell.vertices[1][a]+cell.vertices[2][a])/3;
        if(std::abs(average-double(sample.position[a]))>roundoff)
            return refuse("Grating source cell vertex centroid is inconsistent");
    }
    std::array<double,3> a{},b{},normal{};
    double a2=0,b2=0,c2=0;
    for(unsigned d=0;d<3;++d) {
        a[d]=double(cell.vertices[1][d])-cell.vertices[0][d];
        b[d]=double(cell.vertices[2][d])-cell.vertices[0][d];a2+=a[d]*a[d];b2+=b[d]*b[d];
    }
    for(unsigned d=0;d<3;++d) {normal[d]=a[(d+1)%3]*b[(d+2)%3]-a[(d+2)%3]*b[(d+1)%3];c2+=normal[d]*normal[d];}
    const double actualArea=.5*std::sqrt(c2);
    const double tolerance=64*double(std::numeric_limits<float>::epsilon())*area
        +4*roundoff*(std::sqrt(a2)+std::sqrt(b2))+8*roundoff*roundoff;
    if(!std::isfinite(actualArea)||actualArea<=0||!std::isfinite(tolerance)||tolerance>area*.001
        ||std::abs(actualArea-area)>tolerance)
        return refuse("Grating source cell area is degenerate or unrepresentable");
    output=cell;error.clear();return true;
}
inline std::string gratingEntryCellIdentity(const GratingIncidentBinding& parent,const std::vector<Texel>& texels)
{
    std::string identity;identity.reserve(5*16+parent.identity.size()+texels.size()*4*8);
    word(identity,kGratingEntryCellVersion);word(identity,parent.identity.size());identity+=parent.identity;
    word(identity,parent.frame.scheduling.version);word(identity,parent.frame.scheduling.entrySubdivisions);
    word(identity,parent.frame.entries.size());
    for(const auto& texel:texels)for(float value:texel)bits(identity,value);
    return identity;
}
inline bool calculateGratingEntryCellBounds(const GratingIncidentBinding& parent,
    GratingEntryCellBounds& output,std::string& error)
{
    const auto count=parent.frame.entries.size();
    const auto maximum=renderpassoutput::kMaximumTotalBytes;
    if(parent.identity.size()>maximum||!validScheduling(parent.frame.scheduling)||!count||count>kMaximumEntrySamples||parent.identity.empty())
    {error="Grating source cell retained extent is invalid";return false;}
    GratingEntryCellBounds bounds;bounds.nativeUploadBytes=count*kGratingEntryCellStride*sizeof(Texel);
    const auto identityBytes=5*16+parent.identity.size()+count*kGratingEntryCellStride*4*8;
    // Retained cells/upload/string and temporary cell/parent identities. The
    // original immutable parent is retained by reference, never copied here.
    bounds.hostBytes=sizeof(GratingEntryCellBinding)+count*sizeof(GratingEntryCell)+bounds.nativeUploadBytes+2*identityBytes+parent.identity.size();
    bounds.bytes=bounds.hostBytes+bounds.nativeUploadBytes;
    if(parent.frame.bounds.transientBytes>maximum||parent.bounds.bytes>maximum-parent.frame.bounds.transientBytes
        ||bounds.bytes>maximum-parent.frame.bounds.transientBytes-parent.bounds.bytes)
    {error="Grating source cell storage exceeds the existing whole-frame bound";return false;}
    output=bounds;error.clear();return true;
}
// Validate the existing parent tables and the parent's own identity encoding.
// The optional cells never change that encoding or the terminal admission.
inline bool validateGratingEntryCellParent(const GratingIncidentBinding& parent,std::string& error)
{
    const auto refuse=[&](const char* message){error=message;return false;};
    if(parent.frame.gratingIncidentBinding||!parent.gratingOwner||!parent.material||!std::isfinite(parent.evaluationTime)
        ||parent.frame.roughness!=0||parent.frame.multipleScatterPolicy)
        return refuse("Grating source cells require the original smooth incident parent");
    if(!validateTables(parent.prepared,parent.frame,error,true,SourceMeasureUse::DirectionalIrradiancePreparation))return false;
    std::string identity;identity.reserve(parent.identity.size());identity=parent.frame.identity;
    word(identity,kGratingIncidentVersion);
    for(const auto* id:{&parent.material->structuralDigest(),&parent.gratingOwner->bindingDigest,&parent.gratingOwner->programIdentity})
    {word(identity,id->size());identity+=*id;}
    std::uint64_t timeBits=0;std::memcpy(&timeBits,&parent.evaluationTime,sizeof(timeBits));word(identity,timeBits);
    for(const auto& texel:parent.witnessTexels)for(float value:texel)bits(identity,value);
    if(identity!=parent.identity)return refuse("Grating source cell parent identity changed");
    GratingIncidentBounds bounds;
    if(!calculateGratingIncidentBounds(parent.frame,parent.triangles.size(),bounds,error))return false;
    if(parent.bounds.maximumIncidents!=bounds.maximumIncidents||parent.bounds.maximumOrders!=bounds.maximumOrders
        ||parent.bounds.queryWork!=bounds.queryWork||parent.bounds.orderWork!=bounds.orderWork
        ||parent.bounds.combinedWork!=bounds.combinedWork||parent.bounds.bytes<bounds.bytes)
        return refuse("Grating source cell parent work reservation changed");
    const auto& prepared=parent.prepared;const auto& frame=parent.frame;
    const auto& gratingOwner=*parent.gratingOwner;const auto& material=*parent.material;
    const auto retainedBytes=sizeof(NativeFixtureSurfaceMaterialProgram)+gratingOwner.bindingDigest.size()+gratingOwner.programIdentity.size()
        +sizeof(diffractionmaterial::AdmittedDiffractionMaterialIR)+material.structuralDigest().size()
        +prepared.glass.query.texels.size()*sizeof(Texel)+prepared.glass.query.meshes.size()*sizeof(opaquevisibility::Mesh)
        +prepared.identity.size()+frame.identity.size()+material.structuralDigest().size()+gratingOwner.bindingDigest.size()
        +gratingOwner.programIdentity.size()+parent.witnessTexels.size()*16*sizeof(float)+128;
    if(parent.bounds.bytes-bounds.bytes!=retainedBytes)
        return refuse("Grating source cell retained parent byte reservation changed");
    error.clear();return true;
}
template<class Visitor>
inline bool visitGratingEntryCells(const GratingIncidentBinding& parent,
    const Visitor& visit,std::string& error)
{
    std::size_t index=0;
    const Point source{parent.frame.sourceDirectionTrace[0],parent.frame.sourceDirectionTrace[1],parent.frame.sourceDirectionTrace[2]};
    for(const auto& triangle:parent.frame.triangles) {
        if(!triangle.glass||dot(triangle.normal,source)<=0)continue;
        const auto count=parent.frame.scheduling.entrySubdivisions*parent.frame.scheduling.entrySubdivisions;
        for(std::size_t ordinal=0;ordinal<count;++ordinal) {
            if(index>=parent.frame.entries.size()||parent.frame.entries[index].quadrature!=ordinal)
            {error="Grating source cell ordinal coverage changed";return false;}
            GratingEntryCell cell;
            if(!reconstructGratingEntryCell(triangle,parent.frame.entries[index],parent.frame.scheduling,std::uint32_t(index),cell,error))return false;
            if(!visit(cell,parent.frame.entries[index].area))return false;
            ++index;
        }
    }
    if(index!=parent.frame.entries.size()) {error="Grating source cell triangle coverage changed";return false;}
    error.clear();return true;
}
inline bool prepareGratingEntryCellBinding(const std::shared_ptr<const GratingIncidentBinding>& parent,
    std::shared_ptr<const GratingEntryCellBinding>& output,std::string& error)
{
    if(!parent) {error="Grating source cell incident parent is missing";return false;}
    GratingEntryCellBounds bounds;
    if(!calculateGratingEntryCellBounds(*parent,bounds,error)||!validateGratingEntryCellParent(*parent,error))return false;
    auto binding=std::make_shared<GratingEntryCellBinding>();binding->incidentBinding=parent;binding->bounds=bounds;
    binding->cells.reserve(parent->frame.entries.size());binding->cellTexels.reserve(parent->frame.entries.size()*kGratingEntryCellStride);
    if(!visitGratingEntryCells(*parent,[&](const auto& cell,float area) {
        binding->cells.push_back(cell);const auto packed=gratingEntryCellTexels(cell,area,parent->frame.scheduling.entrySubdivisions);
        binding->cellTexels.insert(binding->cellTexels.end(),packed.begin(),packed.end());return true;
    },error))return false;
    binding->identity=gratingEntryCellIdentity(*parent,binding->cellTexels);
    output=std::move(binding);error.clear();return true;
}
inline bool validateGratingEntryCellBinding(const GratingEntryCellBinding& binding,
    const std::shared_ptr<const GratingIncidentBinding>& exactParent,std::string& error)
{
    const auto refuse=[&](const char* message){error=message;return false;};
    if(!exactParent||binding.incidentBinding!=exactParent)
        return refuse("Grating source cell immutable incident parent differs");
    GratingEntryCellBounds bounds;
    if(!calculateGratingEntryCellBounds(*exactParent,bounds,error)||!validateGratingEntryCellParent(*exactParent,error))return false;
    if(binding.bounds.hostBytes!=bounds.hostBytes||binding.bounds.nativeUploadBytes!=bounds.nativeUploadBytes
        ||binding.bounds.bytes!=bounds.bytes||binding.cells.size()!=exactParent->frame.entries.size()
        ||binding.cellTexels.size()!=binding.cells.size()*kGratingEntryCellStride)
        return refuse("Grating source cell extent or byte reservation changed");
    if(!visitGratingEntryCells(*exactParent,[&](const auto& exact,float area) {
        const auto& cell=binding.cells[exact.entry];
        if(std::tie(cell.object,cell.material,cell.instance,cell.triangle,cell.quadrature,cell.entry)
            !=std::tie(exact.object,exact.material,exact.instance,exact.triangle,exact.quadrature,exact.entry)
            ||std::memcmp(cell.vertices.data(),exact.vertices.data(),sizeof(cell.vertices))!=0
            ||std::memcmp(cell.barycentrics.data(),exact.barycentrics.data(),sizeof(cell.barycentrics))!=0)
            return refuse("Grating source cell geometry or owner changed");
        const auto packed=gratingEntryCellTexels(exact,area,exactParent->frame.scheduling.entrySubdivisions);
        if(std::memcmp(packed.data(),binding.cellTexels.data()+exact.entry*kGratingEntryCellStride,sizeof(packed))!=0)
            return refuse("Grating source cell native upload changed");
        return true;
    },error))return false;
    if(binding.identity!=gratingEntryCellIdentity(*exactParent,binding.cellTexels))return refuse("Grating source cell private identity changed");
    error.clear();return true;
}
// Coverage v2 topology references use the closest-hit query's native tie key.
// Object/material IDs remain attached metadata, never tie-order substitutes.
struct GratingNativeTriangleKeyV2
{
    std::uint32_t objectIndex=UINT32_MAX,instance=UINT32_MAX,triangle=UINT32_MAX;
    bool operator==(const GratingNativeTriangleKeyV2& other) const
    {return std::tie(objectIndex,instance,triangle)==std::tie(other.objectIndex,other.instance,other.triangle);}
    bool operator<(const GratingNativeTriangleKeyV2& other) const
    {return std::tie(objectIndex,instance,triangle)<std::tie(other.objectIndex,other.instance,other.triangle);}
};
struct GratingMeshEdgeV2
{
    GratingNativeTriangleKeyV2 owner;
    std::uint32_t edge=UINT32_MAX;
    bool operator==(const GratingMeshEdgeV2& other) const {return owner==other.owner&&edge==other.edge;}
    bool operator<(const GratingMeshEdgeV2& other) const
    {return owner<other.owner||(!(other.owner<owner)&&edge<other.edge);}
};
struct GratingRetainedTopologyV2
{
    std::uint32_t object=0,material=0;
    std::array<std::uint32_t,3> vertices{};
    const Triangle* triangle=nullptr;
};
inline bool gratingRetainedTopologyV2(const GratingIncidentBinding& parent,GratingNativeTriangleKeyV2 key,
    GratingRetainedTopologyV2& output,std::string& error)
{
    const auto* scene=parent.prepared.glass.sourceScene.get();const auto& frame=parent.frame;
    if(!scene||key.objectIndex>=scene->objectCount||key.instance>=frame.instances.size()
        ||frame.instances[key.instance].object!=key.objectIndex)
    {error="Boundary v2 retained native object-array key is invalid";return false;}
    const auto& object=scene->objects[key.objectIndex];
    if(object.firstIndex>scene->indexCount||object.indexCount>scene->indexCount-object.firstIndex
        ||object.firstVertex>scene->vertexCount||object.vertexCount>scene->vertexCount-object.firstVertex
        ||key.triangle>=object.indexCount/3)
    {error="Boundary v2 retained topology triangle or source range is invalid";return false;}
    const auto found=std::find_if(frame.triangles.begin(),frame.triangles.end(),[&](const auto& t) {
        return t.instance==key.instance&&t.triangle==key.triangle; });
    if(found==frame.triangles.end()||found->object!=object.id.value||found->material!=object.material.value)
    {error="Boundary v2 topology lost its exact object or material owner";return false;}
    GratingRetainedTopologyV2 result;result.object=object.id.value;result.material=object.material.value;result.triangle=&*found;
    for(unsigned v=0;v<3;++v) {
        const auto local=scene->indices[object.firstIndex+3*key.triangle+v];
        if(local>=object.vertexCount||object.firstVertex+local>=scene->vertexCount)
        {error="Boundary v2 source vertex index is invalid";return false;}
        result.vertices[v]=std::uint32_t(object.firstVertex+local);
        if(found->points[v]!=opaquevisibility::transform(frame.instances[key.instance].world,
            opaquevisibility::point(scene->vertices[result.vertices[v]].position)))
        {error="Boundary v2 exact retained world topology changed";return false;}
    }
    output=result;error.clear();return true;
}
inline bool gratingMeshEdgePointsV2(const GratingIncidentBinding& parent,GratingMeshEdgeV2 edge,
    std::array<Point,2>& output,std::string& error)
{
    GratingRetainedTopologyV2 topology;
    if(edge.edge>=3) {error="Boundary v2 local topology edge is invalid";return false;}
    if(!gratingRetainedTopologyV2(parent,edge.owner,topology,error))return false;
    output={topology.triangle->points[edge.edge],topology.triangle->points[(edge.edge+1)%3]};return true;
}
inline bool sameGratingMeshEdgeV2(const std::array<Point,2>& first,const std::array<Point,2>& second)
{return (first[0]==second[0]&&first[1]==second[1])||(first[0]==second[1]&&first[1]==second[0]);}

}
