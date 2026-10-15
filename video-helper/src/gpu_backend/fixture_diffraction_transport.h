#pragma once

#include "fixture_opaque_visibility.h"
#include <cstring>
#include <memory>

namespace arbitgpu::diffractiontransportnative
{
using Point = opaquevisibility::Point;
using Texel = opaquevisibility::Texel;
using Matrix = opaquevisibility::Matrix;
inline constexpr std::size_t kTextureWidth = 512;
inline constexpr std::size_t kSampleStride = 8;
inline constexpr std::uint32_t kTableVersion = 1;
inline constexpr float kRayMinimumDistance = 1.0e-5f;

// A backend keeps this unavailable until its owned shader consumes this exact
// table/query ABI and its readback turns every invalid query into submission
// failure. GL and Metal select independently. Shared admission is not rendering.
struct ConsumerContract
{
    NativeFixtureMaterialBackend backend = NativeFixtureMaterialBackend::Invalid;
    std::uint32_t tableVersion = 0;
    std::size_t sampleStride = 0;
    std::uint32_t closestHitPayloadVersion = 0;
    bool exactTriangleTable = false;
    bool wholeOutputRefusalReadback = false;
};
inline bool consumerCanRender(const ConsumerContract& consumer, NativeFixtureMaterialBackend backend)
{
    return backend!=NativeFixtureMaterialBackend::Invalid && consumer.backend==backend
        && consumer.tableVersion==kTableVersion && consumer.sampleStride==kSampleStride
        && consumer.closestHitPayloadVersion==3 && consumer.exactTriangleTable
        && consumer.wholeOutputRefusalReadback;
}
// Native shaders publish nonfinite diagnostic Color on invalid/truncated query.
// The owner must read the complete scene-linear Color attachment before it can
// publish ANY output from that frame, including non-Color AOV requests.
inline bool validateWholeOutput(const std::vector<float>& rgba,std::uint32_t width,
    std::uint32_t height,std::string& error)
{
    if (!nativeFixtureDimensionsWithinBounds(width,height) || rgba.size()!=std::uint64_t(width)*height*4
        || std::any_of(rgba.begin(),rgba.end(),[](float value){return !std::isfinite(value) || std::abs(value)>surfacematerial::kMaximumLinearColorComponent;}))
    {error="Native diffraction whole-output readback is incomplete, nonfinite or unrepresentable";return false;}
    error.clear();return true;
}

// RGBA32F rows, receiver samples first, then grating samples:
// 0 position.xyz/area, 1 outwardNormal.xyz/0,
// 2 object low16/high16/material low16/high16,
// 3 instanceId/triangleId/role (0 receiver, 1 grating)/rho version,
// 4 rho[0..3], 5 rho[4..7], 6 tangent.xyz/uv.x, 7 bitangent.xyz/uv.y.
// All ids are exact float integers. Rho and source Li retain raw authored bits.
// Grating samples carry the UV frame needed to evaluate the same physical BSDF
// from a receiver pixel. No transport contribution or output pixel is cached.
struct Sample
{
    Point position {}, normal {}, tangent {}, bitangent {};
    std::array<float, 2> uv {};
    float area = 0;
    std::uint32_t object = 0, material = 0, instance = 0, triangle = 0;
    bool grating = false;
    std::optional<surfacematerial::SpectralReflectanceDescription> reflectance;
};
struct Prepared
{
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> scene;
    std::shared_ptr<const NativeFixtureSurfaceMaterialProgram> program;
    opaquevisibility::Prepared query;
    std::uint64_t sceneRevision = 0;
    std::string identity;
};
struct Frame
{
    opaquevisibility::Frame query;
    std::vector<Texel> texels;
    std::vector<Sample> samples;
    std::array<float, 4> sampleControl {}; // receiverCount, gratingCount, stride, version
    std::array<float, 4> sourceDirectionTrace {}; // WORLD toward source, finite maximum
    std::array<Texel, 2> sourceRadiance {};
    float rayMinimumDistance = kRayMinimumDistance;
    std::size_t height = 1;
    std::uint64_t sceneRevision = 0;
    std::string identity;
};
inline Point sub(Point a, Point b) { for (int i=0;i<3;++i) a[i]-=b[i]; return a; }
inline float dot(Point a, Point b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
inline Point cross(Point a, Point b)
{ return {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]}; }
inline bool finite(Point a)
{ return std::all_of(a.begin(),a.end(),[](float v){return std::isfinite(v);}); }
inline bool normalize(Point& a)
{
    const float length=std::sqrt(dot(a,a));
    if (!finite(a) || !std::isfinite(length) || length<=0) return false;
    for (auto& v:a) v/=length;
    return finite(a);
}
inline void word(std::string& out, std::uint64_t value, unsigned digits=16)
{
    static constexpr char hex[]="0123456789abcdef";
    for (unsigned d=digits;d>0;--d) out.push_back(hex[(value>>((d-1)*4))&15]);
}
inline void bits(std::string& out, float value)
{ std::uint32_t b=0;std::memcpy(&b,&value,sizeof(b));word(out,b,8); }
inline bool prepareScene(const HarmonicMIDI::grid::Visual3DScene& scene,
    const NativeFixtureSurfaceMaterialProgram& program, std::uint64_t sceneRevision,
    Prepared& output, std::string& error)
{
    using namespace HarmonicMIDI::grid;
    const auto refuse=[&](const char* message){error=message;return false;};
    if (!validateVisual3DScene(scene).valid() || !sceneRevision
        || program.authoredScene!=scene.id || program.sceneRevision!=sceneRevision
        || !program.diffractionLightingAdmission
        || program.diffractionLightingAdmission->description().version!=3
        || !validNativeFixtureMaterialPrograms(program,scene,program.backend,true))
        return refuse("Native diffraction triangle table scene/program ownership is invalid");
    Prepared candidate;
    candidate.scene=std::make_shared<const Visual3DScene>(scene);
    candidate.program=snapshotNativeSurfaceProgram(std::make_shared<const NativeFixtureSurfaceMaterialProgram>(program));
    candidate.sceneRevision=sceneRevision;
    candidate.identity="diffraction-triangle-table-v1:"+program.programIdentity;
    std::size_t receivers=0,receiverTriangles=0,gratingTriangles=0;
    for (std::size_t o=0;o<scene.objectCount;++o)
    {
        const auto& object=scene.objects[o];
        const auto* surface=nativeSurfaceForObject(&program,object.id);
        if (surface==&program)
        {
            if (object.material!=program.authoredMaterial) return refuse("Native diffraction grating material identity is stale");
            gratingTriangles+=object.indexCount/3;
        }
        else if (surface && surface->spectralReflectance)
        {
            if (surface->kind!=NativeFixtureMaterialKind::SurfacePbr || surface->authoredScene!=scene.id
                || surface->authoredMaterial!=object.material || surface->parameters.identifiers[0]!=object.material.value
                || surface->sceneRevision!=sceneRevision || surface->structuralRevision!=program.structuralRevision
                || surface->evaluationRevision!=program.evaluationRevision
                || surface->materialRevision!=surface->structuralRevision
                || !surfacematerial::validSpectralReflectance(*surface->spectralReflectance)
                || object.vertexCount==0)
                return refuse("Native diffraction receiver object/material/revision/rho identity is invalid");
            ++receivers;
            receiverTriangles+=(object.indexCount?object.indexCount:object.vertexCount)/3;
            word(candidate.identity,object.id.value,8);word(candidate.identity,object.material.value,8);
            word(candidate.identity,surface->sceneRevision);word(candidate.identity,surface->structuralRevision);
            word(candidate.identity,surface->evaluationRevision);word(candidate.identity,surface->materialRevision);
            word(candidate.identity,surface->programIdentity.size());candidate.identity+=surface->programIdentity;
            word(candidate.identity,surface->spectralReflectance->version,8);
            for (float rho:surface->spectralReflectance->coefficients) bits(candidate.identity,rho);
        }
    }
    if (!receivers || !receiverTriangles || !gratingTriangles
        || receiverTriangles>kMaximumNativeDiffractionTriangles || gratingTriangles>kMaximumNativeDiffractionTriangles)
        return refuse("Native diffraction actual triangle table exceeds its receiver/grating budget");
    if (!opaquevisibility::prepareScene(scene,&program,{},candidate.query,error,
        opaquevisibility::AdmissionMode::allSurfaces,true)) return false;
    output=std::move(candidate);error.clear();return true;
}

inline bool appendSamples(const Prepared& prepared, const std::vector<opaquevisibility::Instance>& instances,
    bool grating, std::vector<Sample>& output, std::string& error)
{
    const auto& scene=*prepared.scene;
    const auto& root=*prepared.program;
    std::size_t triangleCount=0;
    for (std::size_t instance=0;instance<instances.size();++instance)
    {
        const auto& placement=instances[instance];
        if (placement.object>=scene.objectCount) {error="Native diffraction instance object is invalid";return false;}
        const auto& object=scene.objects[placement.object];
        const auto* surface=nativeSurfaceForObject(&root,object.id);
        if ((grating && surface!=&root) || (!grating && (!surface || surface==&root || !surface->spectralReflectance))) continue;
        const auto count=object.indexCount?object.indexCount:object.vertexCount;
        triangleCount+=count/3;
        if (triangleCount>kMaximumNativeDiffractionTriangles)
        {error="Native diffraction expanded triangle instances exceed table budget";return false;}
        Matrix inverse;
        if (!opaquevisibility::inverse(placement.world,inverse)) {error="Native diffraction transform is invalid";return false;}
        for (std::size_t at=0;at<count;at+=3)
        {
            std::array<Point,3> local,world;
            std::array<std::array<float,2>,3> uv;
            for (unsigned v=0;v<3;++v)
            {
                const auto index=object.indexCount?scene.indices[object.firstIndex+at+v]:at+v;
                if (index>=object.vertexCount) {error="Native diffraction triangle index is invalid";return false;}
                const auto& vertex=scene.vertices[object.firstVertex+index];
                local[v]=opaquevisibility::point(vertex.position);world[v]=opaquevisibility::transform(placement.world,local[v]);
                uv[v]={vertex.uv.x,vertex.uv.y};
            }
            const auto e1=sub(world[1],world[0]),e2=sub(world[2],world[0]);
            const auto areaVector=cross(e1,e2),localNormal=cross(sub(local[1],local[0]),sub(local[2],local[0]));
            const float area=.5f*std::sqrt(dot(areaVector,areaVector));
            Point normal{inverse[0]*localNormal[0]+inverse[1]*localNormal[1]+inverse[2]*localNormal[2],
                inverse[4]*localNormal[0]+inverse[5]*localNormal[1]+inverse[6]*localNormal[2],
                inverse[8]*localNormal[0]+inverse[9]*localNormal[1]+inverse[10]*localNormal[2]};
            if (!std::isfinite(area) || area<=1e-12f || !normalize(normal))
            {error="Native diffraction world triangle area/normal is invalid";return false;}
            Point tangent{},bitangent{};
            if (grating)
            {
                const float du1=uv[1][0]-uv[0][0],dv1=uv[1][1]-uv[0][1];
                const float du2=uv[2][0]-uv[0][0],dv2=uv[2][1]-uv[0][1],det=du1*dv2-dv1*du2;
                if (!std::isfinite(det) || std::abs(det)<=1e-12f)
                {error="Native diffraction triangle UV frame is invalid";return false;}
                for (int a=0;a<3;++a) {tangent[a]=(e1[a]*dv2-e2[a]*dv1)/det;bitangent[a]=(e2[a]*du1-e1[a]*du2)/det;}
                const float projection=dot(tangent,normal);
                for (int a=0;a<3;++a) tangent[a]-=projection*normal[a];
                if (!normalize(tangent) || !finite(bitangent)) {error="Native diffraction UV tangent is invalid";return false;}
                const float sign=dot(cross(normal,tangent),bitangent)<0?-1.0f:1.0f;
                bitangent=cross(normal,tangent);for (auto& a:bitangent) a*=sign;
            }
            const auto append=[&](float u,float v)
            {
                Sample sample;sample.normal=normal;sample.tangent=tangent;sample.bitangent=bitangent;
                sample.area=area/(kNativeDiffractionSubdivisions*kNativeDiffractionSubdivisions);
                sample.object=object.id.value;sample.material=object.material.value;
                sample.instance=std::uint32_t(instance);sample.triangle=std::uint32_t(at/3);sample.grating=grating;
                if (!grating) sample.reflectance=surface->spectralReflectance;
                for (int a=0;a<3;++a) sample.position[a]=world[0][a]+u*e1[a]+v*e2[a];
                for (int a=0;a<2;++a) sample.uv[a]=uv[0][a]+u*(uv[1][a]-uv[0][a])+v*(uv[2][a]-uv[0][a]);
                output.push_back(sample);
            };
            constexpr unsigned n=kNativeDiffractionSubdivisions;
            for (unsigned i=0;i<n;++i) for (unsigned j=0;j<n-i;++j)
            {append((i+1.0f/3)/n,(j+1.0f/3)/n);if (i+j+1<n) append((i+2.0f/3)/n,(j+2.0f/3)/n);}
        }
    }
    return true;
}
// Query admission checks metadata and every finite sample leg. This is geometry
// validation only. Native fragments must run their own queries and refuse the
// whole output on -1/-2, including rays from positions between these samples.
inline bool validateQueries(const Prepared& prepared, const Frame& frame, std::string& error)
{
    const auto refuse=[&](const char* message){error=message;return false;};
    if (!prepared.scene || !prepared.program || !prepared.program->diffractionLightingAdmission)
        return refuse("Native diffraction query table is not prepared");
    if (frame.sceneRevision!=prepared.sceneRevision || frame.sampleControl[3]!=kTableVersion
        || frame.sampleControl[2]!=kSampleStride || !std::isfinite(frame.rayMinimumDistance)
        || frame.rayMinimumDistance<=0 || !std::isfinite(frame.sourceDirectionTrace[3])
        || frame.sourceDirectionTrace[3]<=frame.rayMinimumDistance)
        return refuse("Native diffraction query frame identity/range is invalid");
    const Point light{frame.sourceDirectionTrace[0],frame.sourceDirectionTrace[1],frame.sourceDirectionTrace[2]};
    const float maximum=frame.sourceDirectionTrace[3],minimum=frame.rayMinimumDistance;
    const auto& authored=prepared.program->diffractionLightingAdmission->description();
    if (!finite(light) || std::abs(dot(light,light)-1)>1.0e-4f || maximum!=authored.traceMaximumDistance
        || light!=authored.paths[0].incident.direction || minimum!=kRayMinimumDistance)
        return refuse("Native diffraction query source or finite bounds changed after admission");
    for (unsigned b=0;b<8;++b)
        if (std::memcmp(&frame.sourceRadiance[b/4][b%4],&authored.paths[0].incident.radiance[b],sizeof(float))!=0)
            return refuse("Native diffraction query authored source radiance changed after admission");
    const auto trace=[&](const Sample& origin, Point direction,float limit,opaquevisibility::ClosestHit& hit)
    {
        hit=opaquevisibility::closestHitCpu(prepared.query,frame.query,origin.position,direction,minimum,limit);
        if (hit.status<0 || hit.status>1) return refuse("Native diffraction closest-hit query is invalid or exhausted");
        if (hit.status==1 && (!hit.objectId || !hit.materialId || !finite(hit.outwardNormal)
            || !std::isfinite(hit.distance) || hit.distance<minimum || hit.distance>limit))
            return refuse("Native diffraction closest-hit endpoint is invalid");
        if (hit.status==1)
        {
            const auto* object=HarmonicMIDI::grid::visual3d_detail::findById(prepared.scene->objects,
                prepared.scene->objectCount,HarmonicMIDI::grid::SceneObjectId{hit.objectId});
            if (!object || object->material.value!=hit.materialId || hit.triangleId>=
                (object->indexCount?object->indexCount:object->vertexCount)/3)
                return refuse("Native diffraction query hit object/material is not in the admitted scene");
        }
        return true;
    };
    if (!std::isfinite(frame.sampleControl[0]) || !std::isfinite(frame.sampleControl[1])
        || frame.sampleControl[0]<1 || frame.sampleControl[1]<1
        || frame.sampleControl[0]>kMaximumNativeDiffractionSamples || frame.sampleControl[1]>kMaximumNativeDiffractionSamples
        || std::floor(frame.sampleControl[0])!=frame.sampleControl[0] || std::floor(frame.sampleControl[1])!=frame.sampleControl[1]
        || frame.sampleControl[0]+frame.sampleControl[1]!=frame.samples.size())
        return refuse("Native diffraction query sample counts are invalid");
    const auto receiverCount=std::size_t(frame.sampleControl[0]);
    if (!receiverCount || receiverCount>=frame.samples.size()) return refuse("Native diffraction query sample counts are invalid");
    for (std::size_t r=0;r<receiverCount;++r)
    {
        const auto& receiver=frame.samples[r];opaquevisibility::ClosestHit source;
        if (!trace(receiver,light,maximum,source)) return false;
        for (std::size_t g=receiverCount;g<frame.samples.size();++g)
        {
            const auto& grating=frame.samples[g];auto direction=sub(grating.position,receiver.position);
            const float distance=std::sqrt(dot(direction,direction));
            if (!std::isfinite(distance) || distance<=2*minimum || distance+minimum>maximum)
                return refuse("Native diffraction finite endpoint range exceeds trace bound");
            for (auto& a:direction) a/=distance;
            Point reciprocal=direction;for (auto& a:reciprocal) a=-a;
            opaquevisibility::ClosestHit forward,backward;
            if (!trace(receiver,direction,distance+minimum,forward) || !trace(grating,reciprocal,distance+minimum,backward)) return false;
            if (!forward.status || !backward.status) return refuse("Native diffraction reciprocal query missed its endpoint");
            const auto endpoint=[&](const opaquevisibility::ClosestHit& hit,const Sample& expected)
            {
                if (hit.distance<distance-minimum) return true; // strictly nearer admitted opaque blocker
                return hit.objectId==expected.object && hit.materialId==expected.material && hit.instanceId==expected.instance
                    && hit.triangleId==expected.triangle && std::abs(hit.distance-distance)<=minimum
                    && dot(hit.outwardNormal,expected.normal)>=1.0f-1.0e-5f;
            };
            if (!endpoint(forward,grating) || !endpoint(backward,receiver))
                return refuse("Native diffraction reciprocal endpoint object/material/triangle identity is stale");
        }
    }
    error.clear();return true;
}
inline bool buildFrame(const Prepared& prepared,const std::vector<opaquevisibility::Instance>& instances,
    std::uint32_t width,std::uint32_t height,Frame& output,std::string& error)
{
    if (!prepared.scene || !prepared.program) {error="Native diffraction table is not prepared";return false;}
    std::array<bool,64> covered{};
    for (const auto& instance:instances)
    {
        if (instance.object>=prepared.scene->objectCount) {error="Native diffraction instance object is invalid";return false;}
        covered[instance.object]=true;
    }
    for (std::size_t o=0;o<prepared.scene->objectCount;++o)
        if (prepared.scene->objects[o].vertexCount && !covered[o])
        {error="Native diffraction query omitted a composed scene object";return false;}
    Frame candidate;candidate.sceneRevision=prepared.sceneRevision;
    const auto& lighting=prepared.program->diffractionLightingAdmission->description();
    candidate.sourceDirectionTrace={lighting.paths[0].incident.direction[0],lighting.paths[0].incident.direction[1],
        lighting.paths[0].incident.direction[2],lighting.traceMaximumDistance};
    for (unsigned b=0;b<8;++b) candidate.sourceRadiance[b/4][b%4]=lighting.paths[0].incident.radiance[b];
    if (!opaquevisibility::frame(prepared.query,instances,candidate.query,error)
        || !appendSamples(prepared,instances,false,candidate.samples,error)) return false;
    const auto receivers=candidate.samples.size();
    if (!appendSamples(prepared,instances,true,candidate.samples,error)) return false;
    const auto gratings=candidate.samples.size()-receivers;
    const auto root=std::size_t(candidate.query.control[0]);
    if (!root || root>=candidate.query.texels.size()) {error="Native diffraction query topology is empty";return false;}
    const auto& allowance=candidate.query.texels[root-1];
    const auto work=std::uint64_t(allowance[0])+(std::uint64_t(allowance[1])<<16);
    if (!nativeFixtureDiffractionQueryWorkWithinBudget(width,height,receivers,gratings,work)
        || !nativeFixtureDiffractionWorkWithinBudget(*prepared.program,width,height,std::max(receivers,gratings)))
    {error="Native diffraction sample/query/lobe frame work exceeds budget";return false;}
    candidate.sampleControl={float(receivers),float(gratings),float(kSampleStride),float(kTableVersion)};
    if (!validateQueries(prepared,candidate,error)) return false;
    for (const auto& s:candidate.samples)
    {
        candidate.texels.push_back({s.position[0],s.position[1],s.position[2],s.area});
        candidate.texels.push_back({s.normal[0],s.normal[1],s.normal[2],0});
        candidate.texels.push_back({float(s.object&65535),float(s.object>>16),float(s.material&65535),float(s.material>>16)});
        candidate.texels.push_back({float(s.instance),float(s.triangle),s.grating?1.0f:0.0f,s.reflectance?float(s.reflectance->version):0.0f});
        for (unsigned half=0;half<2;++half)
        {
            Texel rho{};if(s.reflectance)for(unsigned b=0;b<4;++b)rho[b]=s.reflectance->coefficients[half*4+b];
            candidate.texels.push_back(rho);
        }
        candidate.texels.push_back({s.tangent[0],s.tangent[1],s.tangent[2],s.uv[0]});
        candidate.texels.push_back({s.bitangent[0],s.bitangent[1],s.bitangent[2],s.uv[1]});
    }
    candidate.identity=prepared.identity;
    for (const auto& t:candidate.texels) for(float v:t) bits(candidate.identity,v);
    for (float v:candidate.sourceDirectionTrace) bits(candidate.identity,v);
    for (const auto& t:candidate.sourceRadiance) for(float v:t) bits(candidate.identity,v);
    for (float v:candidate.query.control) bits(candidate.identity,v);
    for (std::size_t i=0;i<std::size_t(candidate.query.control[1]);++i)
        for (float v:candidate.query.texels[i]) bits(candidate.identity,v);
    candidate.height=std::max(std::size_t(1),(candidate.texels.size()+kTextureWidth-1)/kTextureWidth);
    candidate.texels.resize(kTextureWidth*candidate.height);
    output=std::move(candidate);error.clear();return true;
}
} // namespace arbitgpu::diffractiontransportnative
