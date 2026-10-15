#pragma once
#include "../../src/gpu_backend/fixture_opaque_visibility.h"
#include "../../src/gpu_backend/fixture_smooth_glass.h"
#include "vertex_modifier_fixture.h"
#include <limits>
#include <functional>

namespace videohelper::tests::sceneclosest
{
using namespace arbitgpu::opaquevisibility;
using Scene = HarmonicMIDI::grid::Visual3DScene;
using D3 = std::array<double, 3>;
struct Ray { Point origin{}, direction{}; float minimum = 0, maximum = 100; };
using Result = std::array<Texel, 4>;
using Execute = std::function<bool(const Frame&, const Ray&, Result&, std::string&)>;
inline D3 sub(D3 a, D3 b) { for (int i=0;i<3;++i) a[i]-=b[i]; return a; }
inline double dot(D3 a,D3 b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
inline D3 cross(D3 a,D3 b) { return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}; }
inline D3 worldPoint(const Matrix& m, const HarmonicMIDI::grid::SceneVec3& p)
{
    return {double(m[0])*p.x+double(m[4])*p.y+double(m[8])*p.z+m[12],
            double(m[1])*p.x+double(m[5])*p.y+double(m[9])*p.z+m[13],
            double(m[2])*p.x+double(m[6])*p.y+double(m[10])*p.z+m[14]};
}
struct Expected {
    bool hit=false,entering=false;
    double distance=0;
    D3 position{},normal{};
    std::uint32_t object=0,material=0,instance=0,triangle=0;
    std::size_t objectIndex=0;
};
// Double-precision plane intersection and edge half-spaces over world triangles.
// No acceleration payload, renderer transform helper, or shader intersection is used.
inline Expected oracle(const Scene& scene,const std::vector<Instance>& instances,const Ray& ray)
{
    Expected best;best.distance=ray.maximum;
    D3 origin{ray.origin[0],ray.origin[1],ray.origin[2]},direction{ray.direction[0],ray.direction[1],ray.direction[2]};
    double len=std::sqrt(dot(direction,direction));for(auto& c:direction)c/=len;
    for(std::size_t i=0;i<instances.size();++i) {
        const auto& instance=instances[i];const auto& object=scene.objects[instance.object];
        const auto& m=instance.world;
        const double determinant=dot(D3{m[0],m[1],m[2]},cross(D3{m[4],m[5],m[6]},D3{m[8],m[9],m[10]}));
        const auto count=object.indexCount?object.indexCount:object.vertexCount;
        for(std::size_t t=0;t<count/3;++t) {
            std::array<D3,3> p;
            for(std::size_t c=0;c<3;++c) {
                auto v=object.indexCount?scene.indices[object.firstIndex+t*3+c]:t*3+c;
                p[c]=worldPoint(m,scene.vertices[object.firstVertex+v].position);
            }
            auto normal=cross(sub(p[1],p[0]),sub(p[2],p[0]));
            const double denominator=dot(normal,direction);
            if(std::abs(denominator)<1e-14)continue;
            double distance=dot(normal,sub(p[0],origin))/denominator;
            if(distance<ray.minimum||distance>best.distance)continue;
            D3 position=origin;for(int a=0;a<3;++a)position[a]+=distance*direction[a];
            bool inside=true;
            for(int e=0;e<3;++e)
                if(dot(normal,cross(sub(p[(e+1)%3],p[e]),sub(position,p[e]))) < -1e-12)inside=false;
            if(!inside)continue;
            if(best.hit&&distance==best.distance&&
                (instance.object>best.objectIndex||(instance.object==best.objectIndex&&
                 (i>best.instance||(i==best.instance&&t>=best.triangle)))))continue;
            const double scale=std::sqrt(dot(normal,normal))*(determinant<0?-1:1);
            for(auto& c:normal)c/=scale;
            best={true,dot(direction,normal)<0,distance,position,normal,object.id.value,
                  object.material.value,static_cast<std::uint32_t>(i),static_cast<std::uint32_t>(t),instance.object};
        }
    }
    return best;
}
inline Matrix identity() { return {1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1}; }
inline std::shared_ptr<Scene> slab(bool indexed=true,bool prism=false)
{
    auto s=std::make_shared<Scene>();
    const std::array<Point,8> points{{{-1,-1,-.5f},{1,-1,-.5f},{1,1,-.5f},{-1,1,-.5f},
                                     {-1,-1,.5f},{1,-1,.5f},{1,1,.5f},{-1,1,.5f}}};
    const std::array<std::uint32_t,36> faces{{0,2,1,0,3,2,4,5,6,4,6,7,0,1,5,0,5,4,
                                             1,2,6,1,6,5,2,3,7,2,7,6,3,0,4,3,4,7}};
    const std::array<Point,6> wedge{{{-1,-1,-.5f},{1,-1,-.5f},{0,1,-.5f},{-1,-1,.5f},{1,-1,.5f},{0,1,.5f}}};
    const std::array<std::uint32_t,24> wedgeFaces{{0,2,1,3,4,5,0,1,4,0,4,3,1,2,5,1,5,4,2,0,3,2,3,5}};
    const std::size_t count=prism?wedgeFaces.size():faces.size();
    for(std::size_t i=0;i<(indexed?(prism?6u:8u):count);++i) {
        const auto index=indexed?i:(prism?wedgeFaces[i]:faces[i]);
        const auto& p=prism?wedge[index]:points[index];
        s->vertices[i].position={p[0],p[1],p[2]};s->vertices[i].color={1,1,1,1};
    }
    if(indexed)for(std::size_t i=0;i<count;++i)s->indices[i]=prism?wedgeFaces[i]:faces[i];
    s->vertexCount=indexed?(prism?6:8):count;s->indexCount=indexed?count:0;
    s->objects[0].id={0xfedcba98u};s->objects[0].material={0x87654321u};
    s->objects[0].vertexCount=s->vertexCount;s->objects[0].indexCount=s->indexCount;s->objectCount=1;
    s->materials[0].id=s->objects[0].material;s->materials[0].alphaMode=HarmonicMIDI::grid::SceneAlphaMode::Blend;
    s->materials[0].opacity=.4f;s->materialCount=1;return s;
}
inline Result cpuOutput(const ClosestHit& hit)
{
    return {{{float(hit.status),hit.distance,hit.entering?1.0f:0.0f,float(hit.triangleId)},
        {hit.position[0],hit.position[1],hit.position[2],float(hit.instanceId)},
        {hit.outwardNormal[0],hit.outwardNormal[1],hit.outwardNormal[2],0},
        {float(hit.objectId&65535u),float(hit.objectId>>16),float(hit.materialId&65535u),float(hit.materialId>>16)}}};
}
inline bool glassExpandedAdmission(std::string& error)
{
    auto scene=slab();
    scene->materials[0].alphaMode=HarmonicMIDI::grid::SceneAlphaMode::Opaque;
    scene->materials[0].opacity=1;
    arbitgpu::NativeFixtureSurfaceMaterialProgram volume;
    volume.object=scene->objects[0].id;
    volume.parameters.baseColorMetallic={1,1,1,0};
    volume.parameters.emissionRoughness={0,0,0,0};
    volume.parameters.normalOpacity={0,0,1,1};
    volume.parameters.transmissionIorClearcoat={1,1.5f,0,.004f};
    volume.parameters.identifiers={scene->objects[0].material.value,1,0,0};
    scene->materials[1].id={9};scene->materials[1].opacity=1;
    scene->materials[1].metallic=0;scene->materials[1].roughness=1;scene->materialCount=2;
    const std::array<Point,3> points{{{-2,-2,-5},{2,-2,-5},{0,2,-5}}};
    for(int c=0;c<3;++c) {
        scene->vertices[8+c].position={points[c][0],points[c][1],points[c][2]};
        scene->vertices[8+c].normal={0,0,1};scene->vertices[8+c].color={1,1,1,1};
    }
    for(std::size_t t=0;t<256;++t)for(std::size_t c=0;c<3;++c)scene->indices[36+t*3+c]=c;
    scene->vertexCount=11;scene->indexCount=36+256*3;scene->objectCount=33;
    for(std::size_t o=1;o<33;++o) {
        auto& object=scene->objects[o];object.id={std::uint32_t(o)};object.material={9};
        object.firstVertex=8;object.vertexCount=3;object.firstIndex=36;object.indexCount=256*3;
    }
    // 12 shell triangles + 31 reused 256-triangle ranges + an overlapping
    // 244-triangle range reaches exactly 8192, despite only 268 source triangles.
    scene->objects[32].indexCount=244*3;
    arbitgpu::smoothglass::Prepared prepared;
    if(!arbitgpu::smoothglass::prepare(*scene,&volume,{},prepared,error))return false;
    scene->objects[32].indexCount=248*3;
    if(arbitgpu::smoothglass::prepare(*scene,&volume,{},prepared,error)||error.find("8192")==std::string::npos)
        {error="expanded reused/overlapping glass scene exceeded admission without refusal";return false;}
    error.clear();return true;
}
inline bool run(const Execute& execute,std::string& error)
{
    std::size_t comparisons=0;
    if(!glassExpandedAdmission(error))return false;
    auto transparent=slab();Prepared opaque;
    if(!prepareScene(*transparent,nullptr,{},opaque,error)||!opaque.meshes.empty())
        {error="transparent geometry entered the legacy opaque query";return false;}
    auto empty=std::make_shared<Scene>();Prepared emptyPrepared;Frame emptyFrame;Result emptyResult{};
    if(!prepareScene(*empty,nullptr,{},emptyPrepared,error,AdmissionMode::allSurfaces)||
       !arbitgpu::opaquevisibility::frame(emptyPrepared,{},emptyFrame,error)||
       !execute(emptyFrame,{{0,0,0},{0,0,1},0,1},emptyResult,error)||emptyResult[0][0]!=0||
       closestHitCpu(emptyPrepared,emptyFrame,{0,0,0},{0,0,1},0,1).status!=0)
        {error="empty all-surfaces query did not return a complete miss";return false;}
    auto check=[&](const Scene& scene,const std::vector<Instance>& instances,const std::vector<Ray>& rays) {
        Prepared prepared;Frame frame;
        if(!prepareScene(scene,nullptr,{},prepared,error,AdmissionMode::allSurfaces)||!arbitgpu::opaquevisibility::frame(prepared,instances,frame,error))return false;
        for(const auto& ray:rays) {
            Result native{};if(!execute(frame,ray,native,error))return false;
            const auto cpu=closestHitCpu(prepared,frame,ray.origin,ray.direction,ray.minimum,ray.maximum);
            const auto expected=oracle(scene,instances,ray);
            for(const auto& actual : std::array<Result,2>{native,cpuOutput(cpu)}) {
                if(actual[0][0]!=(expected.hit?1.0f:0.0f)) {error="closest-hit status differs from brute-force oracle";return false;}
                if(!expected.hit)continue;
                const double tolerance=2e-5*std::max(1.0,std::abs(expected.distance));
                if(std::abs(actual[0][1]-expected.distance)>tolerance||actual[0][2]!=(expected.entering?1.0f:0.0f)||
                   actual[0][3]!=expected.triangle||actual[1][3]!=expected.instance||
                   actual[3]!=Texel{float(expected.object&65535u),float(expected.object>>16),float(expected.material&65535u),float(expected.material>>16)})
                    {error="closest-hit distance, orientation or identity differs from oracle";return false;}
                for(int a=0;a<3;++a)
                    if(std::abs(actual[1][a]-expected.position[a])>tolerance||std::abs(actual[2][a]-expected.normal[a])>2e-5)
                        {error="closest-hit position or outward normal differs from oracle";return false;}
                ++comparisons;
            }
        }
        return true;
    };
    const std::vector<Ray> rays{{{.17f,.11f,3},{0,0,-7},0,100},{{.17f,.11f,0},{0,0,1},0,100},
        {{3,.13f,.1f},{-1,0,0},0,100},{{3,3,3},{0,0,-1},0,100},
        {{.17f,.11f,3},{0,0,-1},0,2.4f},{{.17f,.11f,3},{0,0,-1},2.6f,10},
        {{.17f,.11f,3},{0,0,-1},2.5f,2.5f}};
    for(bool indexed:{true,false})for(bool prism:{false,true}) {
        auto scene=slab(indexed,prism);
        if(!check(*scene,{{identity(),0}},rays))return false;
        auto m=identity();m[0]=-1.4f;m[4]=.2f;m[5]=.7f;m[8]=.13f;m[10]=1.8f;m[12]=.1f;m[14]=-.2f;
        if(!check(*scene,{{m,0}},rays))return false;
    }
    auto scene=slab();scene->objects[1]=scene->objects[0];scene->objects[1].id={7};scene->objects[1].material={9};
    scene->materials[1].id={9};scene->objectCount=2;scene->materialCount=2;
    auto far=identity(),near=identity();far[14]=-2;near[14]=1;
    if(!check(*scene,{{far,0},{near,1},{near,1}},rays)||!check(*scene,{{identity(),1},{identity(),0}},rays))return false;
    // Degenerate triangles must never replace a real hit.
    scene=slab();scene->indices[0]=scene->indices[1]=scene->indices[2]=0;
    if(!check(*scene,{{identity(),0}},rays))return false;
    Prepared prepared;Frame frame;
    if(!prepareScene(*scene,nullptr,{},prepared,error,AdmissionMode::allSurfaces)||!arbitgpu::opaquevisibility::frame(prepared,{{identity(),0}},frame,error))return false;
    auto invalid=[&](Frame bad,Ray ray,int status=-1) {Result actual{};if(!execute(bad,ray,actual,error))return false;
        if(actual[0][0]!=status||closestHitCpu(prepared,bad,ray.origin,ray.direction,ray.minimum,ray.maximum).status!=status) {error="invalid closest-hit input was not refused";return false;}return true;};
    auto zero=rays[0];zero.direction={0,0,0};if(!invalid(frame,zero))return false;
    auto bounds=rays[0];bounds.minimum=10;bounds.maximum=1;if(!invalid(frame,bounds))return false;
    auto nonfinite=rays[0];nonfinite.maximum=std::numeric_limits<float>::infinity();if(!invalid(frame,nonfinite))return false;
    auto negative=rays[0];negative.minimum=-1;if(!invalid(frame,negative))return false;
    auto corrupt=frame;corrupt.texels[std::size_t(frame.control[0])][3]=frame.control[0];if(!invalid(corrupt,rays[0]))return false;
    corrupt=frame;corrupt.texels[std::size_t(frame.control[0])][0]=100;if(!invalid(corrupt,rays[0]))return false;
    corrupt=frame;corrupt.control[1]=float(kTextureWidth*corrupt.height+1);if(!invalid(corrupt,rays[0]))return false;
    corrupt=frame;corrupt.texels[prepared.meshes[0].root][3]=0;if(!invalid(corrupt,rays[0]))return false;
    // All leaves overlap, making a deliberately one-entry stack insufficient.
    auto repeated=slab();for(std::size_t i=0;i<192;++i)repeated->indices[i]=repeated->indices[i%3];
    repeated->objects[0].indexCount=repeated->indexCount=192;
    if(!check(*repeated,{{identity(),0}},rays))return false;
    if(!prepareScene(*repeated,nullptr,{},prepared,error,AdmissionMode::allSurfaces)||
       !arbitgpu::opaquevisibility::frame(prepared,{{identity(),0}},frame,error))return false;
    // A hit in the first overlapping leaf cannot hide later work exhaustion.
    const auto completed=closestHitCpu(prepared,frame,rays[0].origin,rays[0].direction,0,100);
    if(completed.status!=1||completed.work<2) {error="exhaustion fixture lacks a complete candidate";return false;}
    corrupt=frame;auto& allowance=corrupt.texels[std::size_t(frame.control[0])-1];
    const auto shortWork=completed.work-1;allowance[0]=float(shortWork&65535u);allowance[1]=float(shortWork>>16);
    if(!invalid(corrupt,rays[0],-2))return false;
    corrupt=frame;corrupt.texels[std::size_t(frame.control[0])-1][3]=2;
    if(!invalid(corrupt,rays[0]))return false;
    corrupt=frame;corrupt.texels[std::size_t(frame.control[0])-1][0]=0;
    corrupt.texels[std::size_t(frame.control[0])-1][1]=0;
    if(!invalid(corrupt,rays[0]))return false;
    corrupt=frame;corrupt.texels[std::size_t(frame.control[0])-1][2]=1;
    if(!invalid(corrupt,rays[0],-2))return false;
    // Distinct overlapping ranges and exact-range BLAS reuse have identical
    // world hits. Object order wins despite reversed input instance order.
    repeated->indexCount=195;for(int c=0;c<3;++c)repeated->indices[192+c]=repeated->indices[c];
    repeated->objectCount=64;std::vector<Instance> overlap;
    for(std::size_t o=0;o<64;++o) {
        repeated->objects[o]=repeated->objects[0];repeated->objects[o].id={std::uint32_t(100+o)};
        repeated->objects[o].firstIndex=(o%2)*3;
        overlap.push_back({identity(),63-o});
    }
    if(!check(*repeated,overlap,rays))return false;
    if(!prepareScene(*repeated,nullptr,{},prepared,error,AdmissionMode::allSurfaces)||
       !arbitgpu::opaquevisibility::frame(prepared,overlap,frame,error))return false;
    corrupt=frame;corrupt.texels[std::size_t(frame.control[0])-1][2]=1;
    if(!invalid(corrupt,rays[0],-2))return false;
    // Near-first scheduling must prune the farther half after the first hit.
    overlap.clear();
    for(std::size_t o=0;o<64;++o) {auto m=identity();m[14]=-float(63-o)*2;overlap.push_back({m,o});}
    if(!check(*repeated,overlap,rays))return false;
    Prepared nearPrepared;Frame nearFrame;
    if(!prepareScene(*repeated,nullptr,{},nearPrepared,error,AdmissionMode::allSurfaces)||
       !arbitgpu::opaquevisibility::frame(nearPrepared,overlap,nearFrame,error))return false;
    const auto nearHit=closestHitCpu(nearPrepared,nearFrame,rays[0].origin,rays[0].direction,0,100);
    const auto& nearBudget=nearFrame.texels[std::size_t(nearFrame.control[0])-1];
    const auto fullWork=std::uint32_t(nearBudget[0])+(std::uint32_t(nearBudget[1])<<16);
    if(nearHit.status!=1||nearHit.work>=fullWork/4) {error="near-first query failed to prune separated instances";return false;}
    // Restore the simple scene used by the malformed admission checks below.
    scene=slab();
    if(!prepareScene(*scene,nullptr,{},prepared,error,AdmissionMode::allSurfaces)||
       !arbitgpu::opaquevisibility::frame(prepared,{{identity(),0}},frame,error))return false;
    auto singular=identity();singular[0]=0;
    if(arbitgpu::opaquevisibility::frame(prepared,{{singular,0}},corrupt,error)) {error="singular scene query transform admitted";return false;}
    if(arbitgpu::opaquevisibility::frame(prepared,std::vector<Instance>(kMaximumInstances+1,{identity(),0}),corrupt,error))
        {error="over-budget scene query instances admitted";return false;}
    scene->vertices[0].position.x=std::numeric_limits<float>::quiet_NaN();
    if(prepareScene(*scene,nullptr,{},prepared,error,AdmissionMode::allSurfaces)) {error="nonfinite scene query vertex admitted";return false;}
    scene->vertices[0].position.x=-1;
    scene->materials[0].alphaMode=HarmonicMIDI::grid::SceneAlphaMode::Mask;
    if(prepareScene(*scene,nullptr,{},prepared,error,AdmissionMode::allSurfaces)) {error="masked scene query admitted";return false;}
    scene->materials[0].alphaMode=HarmonicMIDI::grid::SceneAlphaMode::Blend;scene->materials[0].baseColorTexture={1};
    if(prepareScene(*scene,nullptr,{},prepared,error,AdmissionMode::allSurfaces)) {error="textured-alpha scene query admitted";return false;}
    scene->materials[0].baseColorTexture={};
    const auto vertex=videohelper::vertexmodifier::admitVertexModifierIr(videohelper::test::audioNormalDisplacement(),{},error);
    if(!vertex)return false;
    const auto compiled=videohelper::materialprogram::compileMaterialProgram(nullptr,&*vertex,
        videohelper::materialprogram::BackendTarget::OpenGl,error);
    if(!compiled)return false;
    arbitgpu::NativeFixtureSurfaceMaterialProgram deform;deform.object=scene->objects[0].id;
    deform.vertexProgram=std::make_shared<const videohelper::materialprogram::MaterialProgramRecord>(*compiled);
    if(prepareScene(*scene,&deform,{},prepared,error,AdmissionMode::allSurfaces))
        {error="unevaluated vertex deformation entered closest-hit query";return false;}
    scene->objects[0].indexCount=4;
    if(prepareScene(*scene,nullptr,{},prepared,error,AdmissionMode::allSurfaces)) {error="malformed scene query range admitted";return false;}
    if(comparisons<20) {error="closest-hit oracle fixture did not exercise enough hits";return false;}
    error.clear();return true;
}
inline std::string outputShader()
{
    return R"shader(
SV4 sceneTestOutput(SceneClosestHit hit,int pixel) {
    if(pixel==0) return SV4(float(hit.status),hit.distance,hit.entering?1.0:0.0,float(hit.triangleId));
    if(pixel==1) return SV4(hit.position,float(hit.instanceId));
    if(pixel==2) return SV4(hit.outwardNormal,0.0);
    return SV4(float(hit.objectId&uint(65535)),float(hit.objectId>>uint(16)),
               float(hit.materialId&uint(65535)),float(hit.materialId>>uint(16)));
}
)shader";
}
} // namespace videohelper::tests::sceneclosest
