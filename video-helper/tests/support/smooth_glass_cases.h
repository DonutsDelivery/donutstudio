#pragma once
#include "smooth_glass_authored_material.h"
#include "scene_closest_hit_cases.h"
#include "fixture_scene.h"
#include "../../src/fixture_scene_renderer.h"
#include "../../src/gpu_backend/fixture_smooth_glass.h"
#include "../../src/gpu_backend/fixture_rough_glass_tiles.h"
#include "../../../shared/HdrImageOutputContract.h"
#include <deque>
#include <cstring>
#include <iostream>
#include <limits>

namespace videohelper::tests::smoothglassfixture
{
using namespace HarmonicMIDI::grid;
using D3 = sceneclosest::D3;
inline D3 unit(D3 v) { const auto length=std::sqrt(sceneclosest::dot(v,v)); for(auto& c:v)c/=length; return v; }
inline D3 plus(D3 a,D3 b,double factor=1) { for(int c=0;c<3;++c)a[c]+=factor*b[c];return a; }
struct Interface { D3 transmission{},reflection{}; double reflectance=0; bool tir=false; };
// Binary64 vector tangential-component Snell construction and s/p admittance.
// This uses neither the shader's refract expression nor its Fresnel routine.
inline Interface interface(D3 ray,D3 normal,double from,double to)
{
    Interface result;
    const double incidentCosine=-sceneclosest::dot(ray,normal);
    result.reflection=unit(plus(ray,normal,2*incidentCosine));
    const auto tangent=plus(ray,normal,incidentCosine);
    const double tangentLength=std::sqrt(sceneclosest::dot(tangent,tangent));
    const double sine=from/to*tangentLength;
    result.tir=sine>1;
    if(result.tir) {result.reflectance=1;return result;}
    const double cosine=std::sqrt(1-sine*sine);
    if(tangentLength>0) result.transmission=plus(unit(tangent),normal,-cosine/sine);
    else result.transmission=plus(D3{},normal,-1);
    result.transmission=unit(result.transmission);
    if(from==to) return result;
    const double sIncident=from*incidentCosine,sExit=to*cosine;
    const double pIncident=incidentCosine/from,pExit=cosine/to;
    result.reflectance=.5*(std::pow((sIncident-sExit)/(sIncident+sExit),2)
                            +std::pow((pIncident-pExit)/(pIncident+pExit),2));
    return result;
}
// Independent binary64 world-triangle intersection shared by the two physical
// references. This never reads prepared query bytes or native rendered pixels.
inline sceneclosest::Expected triangleHit(const Visual3DScene& scene,D3 at,D3 direction)
{
    std::vector<arbitgpu::opaquevisibility::Instance> instances;
    for(std::size_t o=0;o<scene.objectCount;++o) {
        auto world=sceneclosest::identity();
        world[12]=scene.objects[o].transform.translation.x;
        world[13]=scene.objects[o].transform.translation.y;
        world[14]=scene.objects[o].transform.translation.z;
        instances.push_back({world,o});
    }
    const auto hit=[&](D3 at,D3 direction) {
        // The independent triangle oracle takes a float Ray for native-query
        // comparisons. Keep the binary64 transport/intersection here instead.
        sceneclosest::Expected best; best.distance=1e30;
        for(std::size_t o=0;o<scene.objectCount;++o) {
            const auto& object=scene.objects[o];
            for(std::size_t t=0;t<object.indexCount/3;++t) {
                std::array<D3,3> p;
                for(int c=0;c<3;++c)p[c]=sceneclosest::worldPoint(instances[o].world,
                    scene.vertices[object.firstVertex+scene.indices[object.firstIndex+t*3+c]].position);
                auto n=sceneclosest::cross(sceneclosest::sub(p[1],p[0]),sceneclosest::sub(p[2],p[0]));
                const double denom=sceneclosest::dot(n,direction);
                if(std::abs(denom)<1e-14)continue;
                const double distance=sceneclosest::dot(n,sceneclosest::sub(p[0],at))/denom;
                if(distance<1e-9||distance>=best.distance)continue;
                const auto position=plus(at,direction,distance);
                bool inside=true;
                for(int e=0;e<3;++e)if(sceneclosest::dot(n,sceneclosest::cross(sceneclosest::sub(p[(e+1)%3],p[e]),
                    sceneclosest::sub(position,p[e])))<-1e-12)inside=false;
                if(!inside)continue;
                n=unit(n);best={true,denom<0,distance,position,n,object.id.value,object.material.value,0,
                    static_cast<std::uint32_t>(t),o};
            }
        }
        return best;
    };
    return hit(at,direction);
}
struct Oracle { D3 rgb{}; std::array<D3,3> firstExit{},firstExitPoint{}; std::size_t tir=0,reflections=0; bool complete=true; };
inline Oracle trace(const Visual3DScene& scene,D3 origin,D3 ray,double referenceIor,double b)
{
    Oracle result;
    const auto hit=[&](D3 at,D3 direction) {return triangleHit(scene,at,direction);};
    const auto first=hit(origin,ray);
    if(!first.hit||first.object!=scene.objects[0].id.value) {result.complete=false;return result;}
    for(int channel=0;channel<3;++channel) {
        const double wavelength=channel==0?.6563:channel==1?.5461:.4358;
        const double ior=referenceIor+b*(1/(wavelength*wavelength)-1/(.58756*.58756));
        struct Path { D3 origin,ray; double weight; std::size_t depth; bool inside,primary; };
        std::deque<Path> paths {{origin,ray,1,0,!first.entering,true}};
        std::size_t visits=0;
        while(!paths.empty()) {
            const auto path=paths.front();paths.pop_front();
            if(++visits>512) {result.complete=false;return result;}
            const auto found=hit(path.origin,path.ray);
            if(!found.hit) {
                if(path.inside) {result.complete=false;return result;}
                const auto& color=scene.lights[0].color;
                const D3 environment{color.x,color.y,color.z};
                result.rgb[channel]+=path.weight*environment[channel]*scene.lights[0].intensity;
                continue;
            }
            if(found.object!=scene.objects[0].id.value) {
                if(path.inside) {result.complete=false;return result;}
                const auto& material=scene.materials[found.objectIndex];
                // Receivers are black diffuse with authored constant emission.
                const D3 emission{material.emissive.x,material.emissive.y,material.emissive.z};
                result.rgb[channel]+=path.weight*emission[channel];continue;
            }
            if(path.inside==found.entering||path.depth>=64) {result.complete=false;return result;}
            auto normal=found.normal;if(!found.entering)for(auto& c:normal)c=-c;
            const auto optics=interface(path.ray,normal,path.inside?ior:1,path.inside?1:ior);
            if(optics.tir)++result.tir;
            if(path.primary&&path.inside&&!optics.tir) {
                result.firstExit[channel]=optics.transmission;result.firstExitPoint[channel]=found.position;
            }
            // The geometric bias here is much smaller than the native float
            // bias. Comparisons avoid receiver edges and near-contact geometry.
            if(!optics.tir&&path.weight*(1-optics.reflectance)>1e-10)
                paths.push_back({plus(found.position,normal,-1e-8),optics.transmission,
                    path.weight*(1-optics.reflectance),path.depth+1,!path.inside,path.primary});
            if(path.weight*optics.reflectance>1e-10) {
                ++result.reflections;
                paths.push_back({plus(found.position,normal,1e-8),optics.reflection,
                    path.weight*optics.reflectance,path.depth+1,path.inside,false});
            }
        }
    }
    return result;
}
struct RoughInterface
{
    Interface optics;
    double reflected=0,transmitted=0,masked=0;
};
// Independent binary64 GGX quadrature, followed by the tangential Snell and
// s/p admittance oracle above. No product shader or Fresnel function is called.
inline RoughInterface roughInterface(D3 ray,D3 normal,double from,double to,
                                     double roughness,double u,double v)
{
    if(from==to) {
        RoughInterface result;result.optics.transmission=ray;result.transmitted=1;return result;
    }
    const double alpha=roughness*roughness;
    const auto tangent=unit(sceneclosest::cross(normal,std::abs(normal[2])<.999?D3{0,0,1}:D3{1,0,0}));
    const auto bitangent=sceneclosest::cross(normal,tangent);
    const D3 view{-ray[0],-ray[1],-ray[2]};
    const auto stretched=unit(D3{alpha*sceneclosest::dot(view,tangent),alpha*sceneclosest::dot(view,bitangent),sceneclosest::dot(view,normal)});
    const auto axis=stretched[2]<.99999?unit(D3{-stretched[1],stretched[0],0}):D3{1,0,0};
    const auto perpendicular=sceneclosest::cross(stretched,axis);
    const double radius=std::sqrt(u),azimuth=2*std::acos(-1.0)*v;
    const double x=radius*std::cos(azimuth);
    const double blend=(1+stretched[2])/2;
    const double y=(1-blend)*std::sqrt(std::max(0.0,1-x*x))+blend*radius*std::sin(azimuth);
    const auto disk=plus(plus(D3{},axis,x),perpendicular,y);
    const auto projected=plus(disk,stretched,std::sqrt(std::max(0.0,1-x*x-y*y)));
    const auto local=unit(D3{alpha*projected[0],alpha*projected[1],std::max(0.0,projected[2])});
    const auto facet=plus(plus(plus(D3{},tangent,local[0]),bitangent,local[1]),normal,local[2]);
    RoughInterface result;result.optics=interface(ray,facet,from,to);
    const auto smith=[&](double cosine) {
        if(cosine<=0)return 0.0;
        // GGX slope variance expressed as tan(theta), independently of the
        // shader's cosine-polynomial rearrangement.
        const double tangentSquared=(1-cosine*cosine)/(cosine*cosine);
        return 2/(1+std::sqrt(1+alpha*alpha*std::max(0.0,tangentSquared)));
    };
    result.reflected=result.optics.reflectance*smith(sceneclosest::dot(result.optics.reflection,normal));
    result.transmitted=result.optics.tir?0:(1-result.optics.reflectance)*smith(-sceneclosest::dot(result.optics.transmission,normal));
    result.masked=std::max(0.0,1-result.reflected-result.transmitted);
    return result;
}
struct RoughOracle
{
    Oracle optics;
    arbitgpu::smoothglass::RoughEnergyReceipt energy;
    D3 footprintPower{},footprintMean{},footprintVariance{};
    std::size_t closestHitQueries=0;
    std::uint64_t microEvents=0;
    bool multipleMicroEvents=false;
};
struct RoughReferenceLimits
{
    std::size_t visits=64,depth=32,pending=16;
};
// Sixteen fixed samples of the visible-normal integral at each actual triangle
// interface. Dimension rotations prevent repeating the entry normal at exit.
// This is a bounded quadrature of single scattering, not lossless multiple
// scattering. Its masked power stays explicit, separate from the branch tail.
// Only external cameras are included, so terminal radiance eta factors cancel.
inline RoughOracle traceRough(const Visual3DScene& scene,D3 origin,D3 ray,
                              double referenceIor,double b,double roughness,RoughReferenceLimits limits={})
{
    RoughOracle result;
    if(roughness==0) {result.optics=trace(scene,origin,ray,referenceIor,b);return result;}
    if(!std::isfinite(roughness)||roughness<0||roughness>1
        ||limits.visits==0||limits.visits>64||limits.depth==0||limits.depth>32||limits.pending==0||limits.pending>16)
    {result.optics.complete=false;return result;}
    const auto first=triangleHit(scene,origin,ray);
    ++result.closestHitQueries;
    if(!first.hit||first.object!=scene.objects[0].id.value||!first.entering)
    {result.optics.complete=false;return result;}
    constexpr std::size_t samples=16;
    for(int channel=0;channel<3;++channel) {
        const double wavelength=channel==0?.6563:channel==1?.5461:.4358;
        const double ior=referenceIor+b*(1/(wavelength*wavelength)-1/(.58756*.58756));
        if(!std::isfinite(ior)||ior<1||ior>3) {result.optics.complete=false;return result;}
        for(std::size_t sample=0;sample<samples;++sample) {
            struct Path {D3 origin,ray;double weight;std::size_t depth;bool inside,primary;};
            std::deque<Path> paths{{origin,ray,1.0/samples,0,!first.entering,true}};
            std::size_t visits=0;
            while(!paths.empty()) {
                const auto path=paths.back();paths.pop_back();
                if(++visits>limits.visits) {result.optics.complete=false;return result;}
                const auto found=triangleHit(scene,path.origin,path.ray);
                ++result.closestHitQueries;
                if(!found.hit||found.object!=scene.objects[0].id.value) {
                    if(path.inside) {result.optics.complete=false;return result;}
                    result.energy.terminal[channel]+=path.weight;
                    if(!found.hit) {
                        const auto& color=scene.lights[0].color;
                        const D3 environment{color.x,color.y,color.z};
                        result.optics.rgb[channel]+=path.weight*environment[channel]*scene.lights[0].intensity;
                    } else {
                        const auto& material=scene.materials[found.objectIndex];
                        const D3 emission{material.emissive.x,material.emissive.y,material.emissive.z};
                        result.optics.rgb[channel]+=path.weight*emission[channel];
                        if(path.primary) {
                            result.footprintPower[channel]+=path.weight;
                            result.footprintMean[channel]+=path.weight*found.position[0];
                            result.footprintVariance[channel]+=path.weight*found.position[0]*found.position[0];
                        }
                    }
                    continue;
                }
                if(path.inside==found.entering||path.depth>=limits.depth) {result.optics.complete=false;return result;}
                auto normal=found.normal;if(!found.entering)for(auto& c:normal)c=-c;
                const std::size_t stratum=(sample*(2*path.depth+1)+5*path.depth)%samples;
                std::size_t digits=stratum;double inverse=0;
                for(double place=.5;place>=.0625;place*=.5) {inverse+=double(digits%2)*place;digits/=2;}
                const double v=std::fmod(inverse+double(path.depth)*.6180339887498948482,1.0);
                const auto optics=roughInterface(path.ray,normal,path.inside?ior:1,path.inside?1:ior,
                    roughness,(double(stratum)+.5)/samples,v);
                result.energy.masked[channel]+=path.weight*optics.masked;
                if(optics.optics.tir)++result.optics.tir;
                const auto enqueue=[&](D3 direction,double fraction,bool transmitted) {
                    const double power=path.weight*fraction;
                    // The cutoff scales with the sample count, so adding
                    // quadrature samples cannot discard more branch power.
                    if(power<1e-5/samples) {result.energy.tail[channel]+=power;return true;}
                    if(paths.size()>=limits.pending)return false;
                    paths.push_back({plus(found.position,normal,transmitted?-1e-8:1e-8),direction,
                        power,path.depth+1,transmitted?!path.inside:path.inside,path.primary&&transmitted});
                    return true;
                };
                if(!enqueue(optics.optics.transmission,optics.transmitted,true)
                    ||!enqueue(optics.optics.reflection,optics.reflected,false)) {result.optics.complete=false;return result;}
            }
        }
        if(result.footprintPower[channel]>0) {
            result.footprintMean[channel]/=result.footprintPower[channel];
            result.footprintVariance[channel]=std::max(0.0,result.footprintVariance[channel]/result.footprintPower[channel]
                -result.footprintMean[channel]*result.footprintMean[channel]);
        }
    }
    return result;
}
// The caller supplies the independent binary64 Smith-height distribution.
// Authored triangle intersection, medium traversal and radiance remain here.
template<class Walk> inline RoughOracle traceRoughMultiple(const Visual3DScene& scene,D3 origin,D3 ray,
                              double referenceIor,double b,double roughness,std::uint32_t pixelKey,Walk walk,RoughReferenceLimits limits={},bool selectedPrimary=false)
{
    RoughOracle result;
    if(roughness==0) {result.optics=trace(scene,origin,ray,referenceIor,b);return result;}
    if(!std::isfinite(roughness)||roughness<0||roughness>1
        ||limits.visits==0||limits.visits>64||limits.depth==0||limits.depth>32||limits.pending==0||limits.pending>16)
    {result.optics.complete=false;return result;}
    const auto first=triangleHit(scene,origin,ray);
    ++result.closestHitQueries;
    if(!first.hit||first.object!=scene.objects[0].id.value||!first.entering)
    {result.optics.complete=false;return result;}
    constexpr std::size_t samples=16;
    for(int channel=0;channel<3;++channel) {
        const double wavelength=channel==0?.6563:channel==1?.5461:.4358;
        const double ior=referenceIor+b*(1/(wavelength*wavelength)-1/(.58756*.58756));
        if(!std::isfinite(ior)||ior<1||ior>3) {result.optics.complete=false;return result;}
        for(std::size_t sample=0;sample<samples;++sample) {
            struct Path {D3 origin,ray;double weight;std::size_t depth;bool inside,primary;std::uint32_t history;double throughput;};
            std::deque<Path> paths{{origin,ray,1.0/samples,0,!first.entering,true,2166136261u,1.0/samples}};
            std::size_t visits=0;
            while(!paths.empty()) {
                const auto path=paths.back();paths.pop_back();
                if(++visits>limits.visits) {result.optics.complete=false;return result;}
                const auto found=selectedPrimary && path.depth==0?first:triangleHit(scene,path.origin,path.ray);
                if(!selectedPrimary || path.depth!=0)++result.closestHitQueries;
                if(!found.hit||found.object!=scene.objects[0].id.value) {
                    if(path.inside) {result.optics.complete=false;return result;}
                    result.energy.terminal[channel]+=path.weight;
                    if(!found.hit) {
                        const auto& color=scene.lights[0].color;
                        const D3 environment{color.x,color.y,color.z};
                        result.optics.rgb[channel]+=path.throughput*environment[channel]*scene.lights[0].intensity;
                    } else {
                        const auto& material=scene.materials[found.objectIndex];
                        const D3 emission{material.emissive.x,material.emissive.y,material.emissive.z};
                        result.optics.rgb[channel]+=path.throughput*emission[channel];
                        if(path.primary) {
                            result.footprintPower[channel]+=path.weight;
                            result.footprintMean[channel]+=path.weight*found.position[0];
                            result.footprintVariance[channel]+=path.weight*found.position[0]*found.position[0];
                        }
                    }
                    continue;
                }
                if(path.inside==found.entering||path.depth>=limits.depth) {result.optics.complete=false;return result;}
                auto normal=found.normal;if(!found.entering)for(auto& c:normal)c=-c;
                std::uint32_t key=2166136261u;
                for(const auto value:{found.object,found.instance,found.triangle,pixelKey})key=(key^value)*16777619u;
                std::uint32_t seed=2166136261u;
                for(const auto value:{key,std::uint32_t(sample),std::uint32_t(path.depth),path.history})seed=(seed^value)*16777619u;
                const double from=path.inside?ior:1,to=path.inside?1:ior;
                const auto distribution=walk(path.ray,normal,from,to,roughness,seed);
                result.microEvents+=distribution.events;
                result.multipleMicroEvents|=distribution.events>1;
                if(distribution.status<0) {result.optics.complete=false;return result;}
                result.energy.tail[channel]+=path.weight*distribution.residual;
                for(const auto& exit:distribution.exits) {
                    const double power=path.weight*exit.power;
                    if(power<1e-5/samples) {result.energy.tail[channel]+=power;continue;}
                    if(paths.size()>=limits.pending) {result.optics.complete=false;return result;}
                    const auto history=((path.history^exit.order)*16777619u^std::uint32_t(exit.transmitted))*16777619u;
                    const double eta=from/to;
                    // Match the product's geometric bias, independent of the microfacet normal.
                    const double bias=std::max(1e-5,2e-6*std::max({std::abs(found.position[0]),std::abs(found.position[1]),std::abs(found.position[2])}));
                    paths.push_back({plus(found.position,normal,exit.transmitted?-bias:bias),exit.direction,
                        power,path.depth+1,exit.transmitted?!path.inside:path.inside,path.primary&&exit.transmitted,
                        history,path.throughput*exit.power*(exit.transmitted?eta*eta:1)});
                }
            }
        }
        if(result.footprintPower[channel]>0) {
            result.footprintMean[channel]/=result.footprintPower[channel];
            result.footprintVariance[channel]=std::max(0.0,result.footprintVariance[channel]/result.footprintPower[channel]
                -result.footprintMean[channel]*result.footprintMean[channel]);
        }
    }
    std::string ignored;
    result.optics.complete=arbitgpu::smoothglass::completeRoughEnergy(result.energy,ignored);
    return result;
}
inline Visual3DScene scene(float slope=0,float thickness=1,float cameraAngle=0,bool internal=false)
{
    auto authored=fixture3d::makeScene();
    auto slab=sceneclosest::slab();
    authored.vertexCount=8;authored.indexCount=36;authored.objectCount=1;authored.materialCount=1;
    authored.textureCount=authored.textureTexelCount=0;authored.textureTexels.clear();
    authored.objects[0]=slab->objects[0];authored.objects[0].id={101};authored.objects[0].material={6};
    for(std::size_t v=0;v<8;++v) {
        authored.vertices[v]=slab->vertices[v];auto& p=authored.vertices[v].position;
        p.z=p.z*thickness+(p.z>0?slope*p.x:0);
        authored.vertices[v].normal={0,0,1};authored.vertices[v].tangent={1,0,0,1};
    }
    std::copy_n(slab->indices.begin(),36,authored.indices.begin());
    authored.materials[0]={};authored.materials[0].id={6};authored.materials[0].baseColor={1,1,1};
    authored.materials[0].metallic=0;authored.materials[0].roughness=0;authored.materials[0].opacity=1;
    authored.ambientColor={};authored.lightCount=1;
    authored.lights[0]={};authored.lights[0].id={301};authored.lights[0].kind=SceneLightKind::Environment;
    authored.lights[0].color={.125f,.25f,.5f};authored.lights[0].intensity=1;
    auto& camera=authored.cameras[0];camera.transform={};
    const float angle=cameraAngle*3.14159265358979323846f/180;
    camera.transform.rotation={0,std::sin(angle/2),0,std::cos(angle/2)};
    camera.transform.translation=internal?SceneVec3{}:SceneVec3{3*std::sin(angle),0,3*std::cos(angle)};
    camera.verticalFovRadians=.04f;
    return authored;
}
inline void receiver(Visual3DScene& authored,float boundary=-.435f,D3 emission={.8,.4,.1})
{
    const std::size_t o=authored.objectCount,v=authored.vertexCount,i=authored.indexCount;
    authored.objects[o]={};auto& object=authored.objects[o];object.id={static_cast<std::uint32_t>(101+o)};
    object.material={static_cast<std::uint32_t>(6+o)};object.firstVertex=v;object.vertexCount=4;
    object.firstIndex=i;object.indexCount=6;
    const SceneVec3 points[]={{boundary,-6,-3},{6,-6,-3},{6,6,-3},{boundary,6,-3}};
    for(int c=0;c<4;++c) {
        authored.vertices[v+c]={};authored.vertices[v+c].position=points[c];
        authored.vertices[v+c].normal={0,0,1};authored.vertices[v+c].color={1,1,1,1};
        authored.vertices[v+c].tangent={1,0,0,1};
    }
    const std::uint32_t indices[]={0,1,2,0,2,3};std::copy_n(indices,6,authored.indices.begin()+i);
    authored.materials[o]={};auto& material=authored.materials[o];material.id=object.material;
    material.baseColor={};material.roughness=1;material.opacity=1;
    material.emissive={float(emission[0]),float(emission[1]),float(emission[2])};
    ++authored.objectCount;++authored.materialCount;authored.vertexCount+=4;authored.indexCount+=6;
}
inline bool roughPhysicsCases(std::string& error)
{
    auto authored=scene(.25f,1,20);receiver(authored,-6,{1,1,1});
    authored.lights[0].color={};
    const double angle=20*std::acos(-1.0)/180;
    const D3 camera{authored.cameras[0].transform.translation.x,0,authored.cameras[0].transform.translation.z};
    const D3 ray{-std::sin(angle),0,-std::cos(angle)};
    const auto smooth=trace(authored,camera,ray,1.5,.004);
    const auto zero=traceRough(authored,camera,ray,1.5,.004,0);
    if(!smooth.complete||!zero.optics.complete||smooth.rgb!=zero.optics.rgb
        ||smooth.firstExit!=zero.optics.firstExit||smooth.firstExitPoint!=zero.optics.firstExitPoint)
    {error="roughness zero changed the independent smooth glass boundary";return false;}
    const auto narrow=traceRough(authored,camera,ray,1.5,.004,.08);
    const auto broad=traceRough(authored,camera,ray,1.5,.004,.16);
    const auto nondispersive=traceRough(authored,camera,ray,1.5,0,.16);
    if(!narrow.optics.complete||!broad.optics.complete||!nondispersive.optics.complete)
    {error="bounded rough closed-triangle optical reference did not complete";return false;}
    const auto replay=traceRough(authored,camera,ray,1.5,.004,.08);
    if(replay.energy.terminal!=narrow.energy.terminal||replay.energy.masked!=narrow.energy.masked
        ||replay.energy.tail!=narrow.energy.tail||replay.optics.rgb!=narrow.optics.rgb
        ||replay.closestHitQueries!=narrow.closestHitQueries||narrow.closestHitQueries>3073)
    {error="rough deterministic transport or closest-hit work bound changed between identical inputs";return false;}
    for(const RoughReferenceLimits limit : {RoughReferenceLimits{1,32,16},RoughReferenceLimits{64,1,16},RoughReferenceLimits{64,32,1}})
        if(traceRough(authored,camera,ray,1.5,.004,.08,limit).optics.complete)
        {error="rough actual-triangle transport silently completed after exhausting a branch work bound";return false;}
    for(int c=0;c<3;++c) {
        const auto conserved=[&](const RoughOracle& oracle) {
            return std::abs(oracle.energy.terminal[c]+oracle.energy.masked[c]+oracle.energy.tail[c]-1)<1e-12
                &&oracle.energy.masked[c]>=0&&oracle.energy.tail[c]>=0&&oracle.optics.rgb[c]>=0&&oracle.optics.rgb[c]<=1;
        };
        if(!conserved(narrow)||!conserved(broad)||!conserved(nondispersive)
            ||narrow.footprintPower[c]<.5||broad.footprintPower[c]<.5
            ||narrow.footprintVariance[c]<=0||broad.footprintVariance[c]<=narrow.footprintVariance[c]*2)
        {error="rough triangle transport failed bounded energy or broadened receiver footprint";return false;}
    }
    if(nondispersive.footprintMean[0]!=nondispersive.footprintMean[1]
        ||nondispersive.footprintMean[1]!=nondispersive.footprintMean[2]
        ||nondispersive.footprintVariance[0]!=nondispersive.footprintVariance[2]
        ||std::abs(broad.footprintMean[0]-broad.footprintMean[2])<.001)
    {error="rough prism wavelength geometry did not converge at zero dispersion or split with authored dispersion";return false;}
    if(!arbitgpu::smoothglass::completeRoughEnergy(narrow.energy,error))return false;
    auto farther=std::make_unique<Visual3DScene>(authored);
    for(std::size_t v=8;v<12;++v)farther->vertices[v].position.z=-4;
    const auto far=traceRough(*farther,camera,ray,1.5,.004,.16);
    if(!far.optics.complete||far.footprintVariance[1]<=broad.footprintVariance[1])
    {error="rough receiver footprint ignored actual receiver geometry movement";return false;}
    auto blocker=std::make_unique<Visual3DScene>(authored);receiver(*blocker,-6,{});
    for(std::size_t v=12;v<16;++v)blocker->vertices[v].position.z=-2;
    const auto blocked=traceRough(*blocker,camera,ray,1.5,.004,.08);
    if(!blocked.optics.complete||blocked.optics.rgb[1]>=narrow.optics.rgb[1]*.1)
    {error="rough transmission ignored an actual intervening triangle blocker";return false;}
    const auto stronglyMasked=roughInterface(unit(D3{.98,0,-.2}),D3{0,0,1},1,1.5,.9,.9,.25);
    if(stronglyMasked.masked<=.0001
        ||std::abs(stronglyMasked.reflected+stronglyMasked.transmitted+stronglyMasked.masked-1)>1e-12)
    {error="rough single scattering concealed its unresolved masked power";return false;}
    const auto matched=roughInterface(ray,D3{0,0,1},1,1,.9,.9,.25);
    if(matched.optics.transmission!=ray||matched.reflected!=0||matched.transmitted!=1||matched.masked!=0)
    {error="rough index-matched media created spurious scattering or a masking deficit";return false;}
    auto internal=std::make_unique<Visual3DScene>(scene(0,1,0,true));
    if(traceRough(*internal,D3{},D3{0,0,-1},1.5,0,.08).optics.complete)
    {error="rough external-camera reference silently admitted an internal camera";return false;}
    arbitgpu::smoothglass::RoughEnergyReceipt receipt;
    receipt.terminal.fill(1);
    if(!arbitgpu::smoothglass::completeRoughEnergy(receipt,error))return false;
    receipt.terminal[0]=.99;receipt.masked[0]=.01;
    if(arbitgpu::smoothglass::completeRoughEnergy(receipt,error)
        ||error.find("multiple scattering")==std::string::npos)
    {error="rough unresolved-power receipt exceeded the bound without a precise refusal";return false;}
    receipt.masked[0]=0;
    if(arbitgpu::smoothglass::completeRoughEnergy(receipt,error))
    {error="nonconserving rough power receipt admitted";return false;}
    receipt.terminal.fill(1);receipt.tail[2]=std::numeric_limits<double>::quiet_NaN();
    if(arbitgpu::smoothglass::completeRoughEnergy(receipt,error))
    {error="nonfinite rough power receipt admitted";return false;}
    error.clear();return true;
}
// Prepared native API checks catch incomplete GLSL-to-Metal argument rewriting
// before a backend is allowed to adopt the float32 energy receipt protocol.
inline bool roughSchedulerSourceCases(std::string& error)
{
    const auto gl=arbitgpu::smoothglass::transportShader(false,true);
    const auto metal=arbitgpu::smoothglass::transportShader(true,true);
    if(gl.find("out vec3 terminal,out vec3 masked,out vec3 tail")==std::string::npos
        ||gl.find("out int status,out int queryCount GX_ARGS")==std::string::npos
        ||metal.find("thread float3& terminal,thread float3& masked,thread float3& tail")==std::string::npos
        ||metal.find("thread int& status,thread int& queryCount GX_ARGS")==std::string::npos
        ||metal.find("out int ")!=std::string::npos)
    {error="rough scheduler did not generate compatible GL/Metal float32 receipt arguments";return false;}
    // GLSL and MSL reserve `smooth`, so the local must never be spelled that way.
    // GLASS_ROTATE must name a function defined in the same assembled source.
    for(const auto* source:{&gl,&metal})
    {
        const bool isMetal=source==&metal;
        const std::string rotate="#define GLASS_ROTATE ";
        const auto define=source->find(rotate);
        const auto nameEnd=define==std::string::npos?std::string::npos:source->find('\n',define);
        const std::string name=nameEnd==std::string::npos?std::string():source->substr(define+rotate.size(),nameEnd-define-rotate.size());
        const std::string definition=std::string(isMetal?"float3 ":"vec3 ")+name+"(";
        if(source->find("vec3 smooth=")!=std::string::npos||source->find("float3 smooth=")!=std::string::npos
            ||source->find("GLASS_ROTATE(")==std::string::npos||name.empty()||source->find(definition)==std::string::npos)
        {error="glass GLSL/MSL source reuses a reserved identifier or leaves GLASS_ROTATE undefined";return false;}
    }
    const auto start=gl.find("vec3 roughGlassTransport(");
    const auto zero=gl.find("vec3 smoothColour=glassTransport(camera,entry,smoothComplete,smoothStatus GX_PASS);",start);
    const auto query=gl.find("GlassClosestHit first=glassSceneClosestHit",start);
    if(start==std::string::npos||zero==std::string::npos||query==std::string::npos||zero>=query
        ||arbitgpu::smoothglass::kMaximumRoughClosestHitQueries!=3073
        ||arbitgpu::smoothglass::kMaximumRoughColorRayQueries!=52225
        ||std::string(arbitgpu::smoothglass::roughTransportDiagnostic(arbitgpu::smoothglass::RoughTransportStatus::UnresolvedEnergy)).find("multiple scattering")==std::string::npos)
    {error="rough scheduler changed the zero delegation, work receipt or unresolved-power diagnostic";return false;}
    if(gl.find("roughGlassTransportV2(")==std::string::npos
        ||gl.find("glassMsWalk(direction,normal,fromIor,toIor,roughness,seed")==std::string::npos
        ||gl.find("tail[channel]+=power*residual;")==std::string::npos
        ||gl.find("histories[pending]=glassMsChildHistory(history,orders[branch],transmission)")==std::string::npos
        ||metal.find("thread float4 (&directions)[GLASS_MS_EXITS]")==std::string::npos
        ||metal.find("thread uint (&orders)[GLASS_MS_EXITS]")==std::string::npos
        ||metal.find("thread uint& events")==std::string::npos
        ||metal.find("out uint ")!=std::string::npos || metal.find("out vec4 ")!=std::string::npos)
    {error="v2 camera lost its shared walk, weighted residual, ancestry or native argument rewrite";return false;}
    const auto selectedGl=arbitgpu::smoothglass::transportShader(false,true,true);
    const auto selectedMetal=arbitgpu::smoothglass::transportShader(true,true,true);
    const auto selectedBegin=selectedGl.find("vec3 roughGlassTransportV2Selected(");
    if(selectedBegin==std::string::npos
        ||selectedGl.find("queryCount=1;",selectedBegin)==std::string::npos
        ||selectedGl.find("GlassClosestHit hit=first;",selectedBegin)==std::string::npos
        ||selectedGl.find("if(depth!=0)",selectedBegin)==std::string::npos
        ||selectedGl.find("GlassClosestHit first=glassSceneClosestHit",selectedBegin)!=std::string::npos
        ||selectedMetal.find("roughGlassTransportV2Selected(float3 camera,float3 incoming,GlassClosestHit first,")==std::string::npos
        ||selectedMetal.find("out int ")!=std::string::npos ||selectedMetal.find("out vec3 ")!=std::string::npos)
    {error="Selected native camera shader lost its saved primary or native output rewrite";return false;}
    error.clear();return true;
}
inline std::shared_ptr<const videorender::fixture3d::AdmittedSurfaceMaterialBinding> binding(
    const std::shared_ptr<const Visual3DScene>& authored,videohelper::materialprogram::BackendTarget target,
    float ior,float b,std::string& error,float roughness=0,std::uint32_t scatteringVersion=1)
{
    surfacematerialbinding::ImportedSceneMaterialRequest request;
    request.scene=authored->id;request.sceneRevision=request.structuralRevision=request.evaluationRevision=request.programRevision=1;
    request.sceneSnapshot=authored;request.program=authoredProgram(6,ior,b);
    request.roughGlassScatteringVersion=scatteringVersion;
    request.version=surfacematerialbinding::requestVersion(false,false,scatteringVersion);
    for(auto& operation:request.program.operations)
        if(operation.id==static_cast<std::uint32_t>(surfacematerial::OutputSemantic::Roughness)+1)operation.literal[0]=roughness;
    request.binding.targetKind=surfacematerialbinding::BindingTargetKind::ObjectOverride;
    request.binding.object=authored->objects[0].id;request.binding.surfaceMaterialRevision=1;
    const auto admitted=surfacematerial::admit(request.program,error);if(!admitted)return {};
    request.binding.surfaceMaterialDigest=admitted->structuralDigest();
    return videorender::fixture3d::admitSurfaceMaterialBinding(authored,request,target,error);
}
// Preview/Export use the persisted authored override selector. The independent
// walk callback supplies binary64 interface physics, never product shader code.
template<class Walk,class Read> inline bool roughMultipleNativeCases(
    arbitgpu::NativeFixtureSceneBackend& backend,Walk walk,Read read,std::string& error)
{
    using namespace arbitgpu;
    constexpr std::uint32_t extent=5;
    const auto target=backend.info().backend=="metal"?videohelper::materialprogram::BackendTarget::Metal:videohelper::materialprogram::BackendTarget::OpenGl;
    struct Case {float roughness,dispersion;bool blocker,foreground;unsigned geometry=0;};
    const Case rows[]={{.16f,.004f,false,false},{.35f,.004f,false,false},{.35f,0,false,false},
        {.35f,.004f,true,false},{.35f,.004f,false,true},
        {.16f,.004f,false,false,1},{.16f,.004f,false,false,2},{.35f,.004f,false,true,3}};
    bool multipleEvents=false,occlusion=false;
    for(const auto& row:rows) {
        auto authored=scene(.25f,1,20);authored.lights[0].color={};
        if(row.geometry==2) {
            // One connected, nonintersecting U-shaped shell. Camera rays cross
            // two separate arms, exposing layered raster overdraw in one volume.
            const float outline[][2]={{-2,.5f},{2,.5f},{2,0},{-1,0},{-1,-1},{2,-1},{2,-1.5f},{-2,-1.5f}};
            authored.vertexCount=16;authored.indexCount=0;
            for(unsigned side=0;side<2;++side)for(unsigned point=0;point<8;++point) {
                authored.vertices[side*8+point]=authored.vertices[0];
                authored.vertices[side*8+point].position={outline[point][0],side?2.f:-2.f,outline[point][1]};
            }
            const auto triangle=[&](unsigned a,unsigned b,unsigned c) {
                authored.indices[authored.indexCount++]=a;authored.indices[authored.indexCount++]=b;authored.indices[authored.indexCount++]=c;
            };
            const unsigned caps[][3]={{0,1,2},{0,2,3},{0,3,4},{0,4,7},{7,4,5},{7,5,6}};
            for(const auto& cap:caps) {triangle(cap[2],cap[1],cap[0]);triangle(cap[0]+8,cap[1]+8,cap[2]+8);}
            for(unsigned point=0;point<8;++point) {
                const auto next=(point+1)%8;
                triangle(point,next,next+8);triangle(point,next+8,point+8);
            }
            authored.objects[0].vertexCount=authored.vertexCount;authored.objects[0].indexCount=authored.indexCount;
        }
        if(row.geometry==1 || row.geometry==2) {
            std::vector<std::array<std::uint32_t,3>> triangles;
            for(unsigned i=0;i<authored.indexCount;i+=3)triangles.push_back({authored.indices[i],authored.indices[i+1],authored.indices[i+2]});
            const auto depth=[&](const auto& triangle) {
                float z=0;for(const auto vertex:triangle)z+=authored.vertices[vertex].position.z;return z;
            };
            std::stable_sort(triangles.begin(),triangles.end(),[&](const auto& a,const auto& b){return depth(a)<depth(b);});
            for(std::size_t i=0;i<triangles.size();++i)std::copy(triangles[i].begin(),triangles[i].end(),authored.indices.begin()+i*3);
        }
        receiver(authored,-6,{1,1,1});
        if(row.blocker||row.foreground) {
            receiver(authored,-6,{});
            const auto& foreground=authored.objects[authored.objectCount-1];
            for(std::size_t v=foreground.firstVertex;v<foreground.firstVertex+foreground.vertexCount;++v)
                authored.vertices[v].position.z=row.foreground?2:-2;
            if(row.geometry==3) {
                receiver(authored,-6,{});
                const auto& tie=authored.objects[authored.objectCount-1];
                for(std::size_t v=tie.firstVertex;v<tie.firstVertex+tie.vertexCount;++v)authored.vertices[v].position.z=2;
            }
        }
        auto snapshot=std::make_shared<const Visual3DScene>(std::move(authored));
        const auto material=binding(snapshot,target,1.5f,row.dispersion,error,row.roughness,2);
        if(!material||material->roughGlassScatteringVersion()!=2)return false;
        videorender::fixture3d::FixtureSceneRenderer renderer(backend);
        NativeFixtureSceneRuntimeInputs inputs;inputs.linearColor=true;
        videorender::fixture3d::RenderedFrame preview,exported;
        if(!renderer.renderPreview(snapshot,material,{},inputs,{extent,extent},"native-gpu",preview,error)
            ||!renderer.renderExport(snapshot,material,{},inputs,{extent,extent},"native-gpu",exported,error))return false;
        std::array<NativeRawPassPixels,3> receipt,replay;
        NativeRawPassPixels objects,materials,replayObjects,replayMaterials,color,replayColor;
        if(!read(preview.nativeFrame,receipt,objects,materials,error)
            ||!read(exported.nativeFrame,replay,replayObjects,replayMaterials,error)
            ||!preview.nativeFrame->readRawPass(renderpassoutput::Output::Color,color,error)
            ||!exported.nativeFrame->readRawPass(renderpassoutput::Output::Color,replayColor,error))return false;
        if(color.bytes!=replayColor.bytes||objects.bytes!=replayObjects.bytes||materials.bytes!=replayMaterials.bytes)
        {error="v2 authored rough camera Preview/Export owner pixels differ";return false;}
        for(unsigned i=0;i<3;++i)if(receipt[i].bytes!=replay[i].bytes)
        {error="v2 authored rough camera Preview/Export power receipts differ";return false;}
        if(!row.blocker&&!row.foreground) {
            smoothglass::Prepared owner;
            if(!smoothglass::prepare(snapshot,material->nativeProgram(),{},owner,error,
                {material->nativeProgram()->backend,true,true,true,2,true}))return false;
            for(unsigned corrupt=0;corrupt<3;++corrupt) {
                auto bad=receipt;
                const float value=corrupt==0?1.f:corrupt==1?1.5f:float(smoothglass::kMaximumCameraMicroEvents+1);
                std::memcpy(bad[corrupt==0?0:2].bytes.data()+12,&value,4);
                NativeFixtureSceneStats untouched;untouched.roughGlassCoveredPixels=997;
                if(smoothglass::completeRoughFrame(owner,extent,extent,bad[0],bad[1],bad[2],objects,materials,untouched,error)
                    ||untouched.roughGlassCoveredPixels!=997||untouched.roughGlassReceiptComplete)
                {error="v2 native receipt accepted stale version or invalid event work";return false;}
            }
            const auto prepared=backend.prepare(snapshot,material->nativeProgram());
            if(!prepared.prepared) {error=prepared.error;return false;}
            const auto stale=backend.render(std::make_shared<const Visual3DScene>(*snapshot),prepared.resources,extent,extent,inputs);
            if(stale.rendered||stale.frame) {error="v2 camera stale native scene owner published";return false;}
        }
        const double angle=20*std::acos(-1.0)/180,tangent=std::tan(snapshot->cameras[0].verticalFovRadians*.5);
        const D3 camera{snapshot->cameras[0].transform.translation.x,0,snapshot->cameras[0].transform.translation.z};
        std::uint64_t covered=0;
        for(std::uint32_t y=0;y<extent;++y)for(std::uint32_t x=0;x<extent;++x) {
            const auto pixel=y*extent+x;
            const double vx=(2*(double(x)+.5)/extent-1)*tangent,vy=(1-2*(double(y)+.5)/extent)*tangent;
            const auto ray=unit(D3{std::cos(angle)*vx-std::sin(angle),vy,-std::sin(angle)*vx-std::cos(angle)});
            const auto first=triangleHit(*snapshot,camera,ray);
            std::uint32_t object=0,materialId=0;
            std::memcpy(&object,objects.bytes.data()+pixel*4,4);std::memcpy(&materialId,materials.bytes.data()+pixel*4,4);
            if(!first.hit||object!=first.object||materialId!=first.material)
            {error="v2 camera native primary IDs differ from authored triangle visibility";return false;}
            std::array<std::array<float,4>,3> values;
            for(unsigned i=0;i<3;++i)std::memcpy(values[i].data(),receipt[i].bytes.data()+pixel*16,16);
            if(object!=101) {
                occlusion=true;
                for(const auto& value:values)for(const float v:value)if(v!=0)
                {error="v2 camera wrote receipt through actual foreground triangles";return false;}
                continue;
            }
            ++covered;
            const auto key=((2166136261u^x)*16777619u^y)*16777619u;
            const auto expected=traceRoughMultiple(*snapshot,camera,ray,1.5,row.dispersion,row.roughness,key,walk,{},target==videohelper::materialprogram::BackendTarget::OpenGl
                ||target==videohelper::materialprogram::BackendTarget::Metal);
            if(!expected.optics.complete || values[0][3]!=smoothglass::kMultipleScatterCameraComplete
                ||values[1][3]!=expected.closestHitQueries||values[2][3]!=expected.microEvents)
            {error="v2 camera native version/query/event receipt differs from binary64 authored traversal";return false;}
            multipleEvents|=expected.multipleMicroEvents;
            for(unsigned c=0;c<3;++c) {
                const double powers[]={expected.energy.terminal[c],expected.energy.masked[c],expected.energy.tail[c]};
                for(unsigned i=0;i<3;++i)if(!std::isfinite(values[i][c])||std::abs(values[i][c]-powers[i])>.00002)
                {error="v2 camera native power differs from independent weighted interface residual";return false;}
                std::uint16_t half=0;std::memcpy(&half,color.bytes.data()+pixel*8+c*2,2);
                if(!std::isfinite(hdrimage::halfToFloat(half))||std::abs(hdrimage::halfToFloat(half)-expected.optics.rgb[c])>.002)
                {error="v2 camera native radiance differs from binary64 authored triangle exits";return false;}
            }
        }
        if(!preview.stats.roughGlassReceiptComplete||preview.stats.roughGlassCoveredPixels!=covered
            ||preview.stats.roughGlassMaximumUnresolvedPower>.0001||preview.stats.roughGlassMaximumEnergyResidual>.00002)
        {error="v2 authored camera omitted whole-frame bounded energy receipt";return false;}
    }
    if(!multipleEvents||!occlusion) {error="v2 camera fixtures omitted multiple micro-events or real foreground visibility";return false;}
    // Either saved version at exact zero must reuse released smooth bytes.
    auto snapshot=std::make_shared<const Visual3DScene>(scene(.25f,1,20));
    NativeRawPassPixels zero[2];
    for(unsigned version=1;version<=2;++version) {
        const auto material=binding(snapshot,target,1.5f,.004f,error,0,version);if(!material)return false;
        videorender::fixture3d::FixtureSceneRenderer renderer(backend);
        NativeFixtureSceneRuntimeInputs inputs;inputs.linearColor=true;videorender::fixture3d::RenderedFrame frame;
        if(!renderer.renderPreview(snapshot,material,{},inputs,{extent,extent},"native-gpu",frame,error)
            ||frame.stats.roughGlassReceiptComplete||frame.stats.roughGlassReceiptBytes
            ||!frame.nativeFrame->readRawPass(renderpassoutput::Output::Color,zero[version-1],error))return false;
    }
    if(zero[0].bytes!=zero[1].bytes) {error="v2 authored selector changed exact-zero smooth Color";return false;}
    error.clear();return true;
}
// Synthetic scalar receipts exercise the native readback contract. They are
// never reference render pixels and do not claim shader/runtime verification.
inline bool roughTileCases(const arbitgpu::smoothglass::Prepared& prepared,std::string& error)
{
    using namespace arbitgpu;
    using namespace arbitgpu::roughglasstiles;
    const auto check=[&](bool condition,const char* message) { if(!condition)error=message;return condition; };
    FrameKey key;
    key.scene=prepared.sourceScene;key.program=prepared.sourceMaterial;
    NativeFixtureSceneRuntimeInputs inputs;inputs.linearColor=true;
    key.runtime=std::make_shared<const NativeFixtureSceneRuntimeInputs>(inputs);
    key.owner=std::make_shared<const int>(19);
    key.camera=*findFixtureActiveCamera(*key.scene);
    key.sceneRevision=key.program->sceneRevision;key.rendererGeneration=7;
    const auto scheduleFor=[&](std::uint32_t w,std::uint32_t h,std::shared_ptr<const Schedule>& plan) {
        auto candidate=key;candidate.width=w;candidate.height=h;
        return Schedule::admit(prepared,candidate,false,plan,error);
    };
    for(const auto extent : {std::array<std::uint32_t,2>{9708,1},{9709,1},{1920,1080},{1921,1081},
                             {3840,2160},{3841,2161},{1,9709}}) {
        std::shared_ptr<const Schedule> plan,replay;
        if(!scheduleFor(extent[0],extent[1],plan)||!scheduleFor(extent[0],extent[1],replay))return false;
        if(!check(plan->tiles().size()==replay->tiles().size() && plan->tiles().size()<=kMaximumFrameJobs,
                  "rough tile schedule replay or finite job bound failed"))return false;
        std::vector<std::uint8_t> coverage(static_cast<std::size_t>(plan->pixels()));
        std::uint64_t pixels=0,queries=0;
        for(std::size_t i=0;i<plan->tiles().size();++i) {
            const auto& tile=plan->tiles()[i];
            if(!check(sameTile(tile,replay->tiles()[i]) && tile.index==i && tile.pixels()<=9708
                && tile.reservedQueries()<=smoothglass::kMaximumPixelRayQueries
                && tile.receiptBytes()==tile.pixels()*48 && tile.x+tile.width<=extent[0]
                && tile.y+tile.height<=extent[1],"rough tile geometry, replay or per-submission reservation failed"))return false;
            pixels+=tile.pixels();queries+=tile.reservedQueries();
            for(std::uint32_t y=tile.y;y<tile.y+tile.height;++y)
                for(std::uint32_t x=tile.x;x<tile.x+tile.width;++x)
                    if(!check(++coverage[std::size_t(y)*extent[0]+x]==1,"rough tiles overlap"))return false;
        }
        if(!check(pixels==plan->pixels() && queries==plan->reservedQueries()
            && queries<=kMaximumFrameRayQueries && std::all_of(coverage.begin(),coverage.end(),[](auto n){return n==1;}),
            "rough tile coverage has a hole or exceeds total frame work"))return false;
        if(extent[0]==9708 && !check(plan->tiles().size()==1,"9708-pixel tile boundary split"))return false;
        if(extent[0]==9709 && !check(plan->tiles().size()==2,"9709-pixel frame bypassed tile boundary"))return false;
        FrameMemoryAdmission memory;
        if(!admitMemory(*plan,64,4096,memory,error)
            ||!check(memory.requestedBytes>=plan->pixels()*64+plan->maximumTilePixels()*120
                && memory.requestedBytes<renderpassoutput::kMaximumTotalBytes
                && plan->scratchBytes()<=9708*48,"rough tile/full-frame memory charge was not bounded"))return false;
        if(!check(!admitMemory(*plan,renderpassoutput::kMaximumTotalBytes,0,memory,error)
            && memory.failure!=FrameMemoryAdmissionFailure::none,"over-budget full-frame attachments admitted"))return false;
        if(!check(!admitMemory(*plan,64,std::numeric_limits<std::uint64_t>::max(),memory,error),
                  "overflowing retained memory charge admitted"))return false;
    }
    std::shared_ptr<const Schedule> plan;
    if(!scheduleFor(9709,1,plan))return false;
    key=plan->key();
    auto unchangedPlan=plan;
    auto invalidKey=key;invalidKey.width=0;
    if(!check(!Schedule::admit(prepared,invalidKey,false,unchangedPlan,error) && unchangedPlan==plan,
        "failed tile admission changed immutable schedule"))return false;
    invalidKey=key;invalidKey.width=renderpassoutput::kMaximumWidth+1;
    if(!check(!Schedule::admit(prepared,invalidKey,false,unchangedPlan,error),"oversize rough tile frame admitted"))return false;
    if(!check(!Schedule::admit(prepared,key,true,unchangedPlan,error),"rough tiled dynamic instances admitted"))return false;
    const auto sameAggregate=[](const Aggregate& a,const Aggregate& b) {
        return a.completedTiles==b.completedTiles && a.completedPixels==b.completedPixels && a.coveredPixels==b.coveredPixels
            && a.reservedQueries==b.reservedQueries && a.receiptClosestHitQueries==b.receiptClosestHitQueries
            && a.peakReceiptBytes==b.peakReceiptBytes && a.processedReceiptBytes==b.processedReceiptBytes
            && a.source==b.source && a.terminal==b.terminal && a.masked==b.masked && a.tail==b.tail
            && a.maximumEnergyResidual==b.maximumEnergyResidual && a.maximumUnresolvedPower==b.maximumUnresolvedPower;
    };
    struct Storage {
        NativeRawPassPixels terminal,masked,tail,objects,materials,normal,color;
        NativeTileReceipt receipt() const { return {true,&terminal,&masked,&tail,&objects,&materials,&normal,&color}; }
    };
    const auto store=[&](const Tile& tile) {
        const auto image=[&](NativeTexturePixelFormat format,std::size_t stride) {
            return NativeRawPassPixels{format,tile.width,tile.height,std::vector<std::uint8_t>(tile.pixels()*stride)};
        };
        Storage value;
        value.terminal=image(NativeTexturePixelFormat::Rgba32Float,16);value.masked=value.tail=value.terminal;
        value.objects=image(NativeTexturePixelFormat::R32Uint,4);value.materials=value.objects;
        value.normal=image(NativeTexturePixelFormat::Rgba16Float,8);value.color=value.normal;
        const std::array<float,4> t{.99998f,.99998f,.99998f,1},m{.00001f,.00001f,.00001f,3},r{.00001f,.00001f,.00001f,1};
        std::memcpy(value.terminal.bytes.data(),t.data(),16);std::memcpy(value.masked.bytes.data(),m.data(),16);
        std::memcpy(value.tail.bytes.data(),r.data(),16);
        std::memcpy(value.objects.bytes.data(),&prepared.object.value,4);
        std::memcpy(value.materials.bytes.data(),&prepared.material.value,4);
        return value;
    };
    std::array<Storage,2> storage{store(plan->tiles()[0]),store(plan->tiles()[1])};
    Completion forward(plan),reverse(plan);
    NativeFixtureSceneStats stats;stats.drawCount=42;stats.roughGlassCoveredPixels=777;
    Aggregate output;output.completedTiles=99;
    if(!check(!forward.publish(key,stats,output,error) && stats.roughGlassCoveredPixels==777 && output.completedTiles==99,
        "missing rough tiles published or changed final stats"))return false;
    for(std::size_t i=0;i<2;++i) {
        if(!forward.accept(key,plan->tiles()[i],storage[i].receipt(),error)
            ||!reverse.accept(key,plan->tiles()[1-i],storage[1-i].receipt(),error))return false;
        if(i==0 && !check(!forward.publish(key,stats,output,error) && stats.roughGlassCoveredPixels==777
            && output.completedTiles==99,"partial rough frame published or changed final stats"))return false;
    }
    if(!forward.publish(key,stats,output,error)
        ||!check(sameAggregate(forward.aggregate(),reverse.aggregate()) && stats.drawCount==42
            && stats.roughGlassReceiptComplete && stats.roughGlassCoveredPixels==2
            && stats.roughGlassReceiptBytes==plan->scratchBytes() && output.receiptClosestHitQueries==6
            && output.reservedQueries==9709*smoothglass::kMaximumRoughFrameRayQueries
            && output.processedReceiptBytes==9709*48 && output.source==std::array<double,3>{2,2,2} && output.terminal[0]>1.99
            && output.masked[0]>0 && output.tail[0]>0,"rough tile final aggregate, work split or deterministic order failed"))return false;
    for(int failure=0;failure<19;++failure) {
        Completion failed(plan);
        if(!failed.accept(key,plan->tiles()[0],storage[0].receipt(),error))return false;
        const auto before=failed.aggregate();
        auto stale=key;auto tile=plan->tiles()[1];auto bad=storage[1];auto receipt=bad.receipt();
        if(failure==0)tile=plan->tiles()[0];
        if(failure==1)tile.x=0;
        if(failure==2)++tile.width;
        if(failure==3)++stale.camera.transform.translation.x;
        if(failure==4)stale.owner=std::make_shared<const int>(19);
        if(failure==5)stale.runtime=std::make_shared<const NativeFixtureSceneRuntimeInputs>(*key.runtime);
        if(failure==6)++stale.rendererGeneration;
        if(failure==7)++stale.sceneRevision;
        if(failure==8)++stale.width;
        if(failure==9)stale.program=std::make_shared<const NativeFixtureSurfaceMaterialProgram>(*key.program);
        if(failure==10)stale.scene=std::make_shared<const Visual3DScene>(*key.scene);
        if(failure==11)receipt.gpuCompleted=false;
        if(failure==12)receipt.tail=nullptr;
        if(failure==13)bad.objects.width=2;
        if(failure==14) {const float status=-4;std::memcpy(bad.terminal.bytes.data()+12,&status,4);std::memcpy(bad.tail.bytes.data()+12,&status,4);}
        if(failure==15) {const std::uint16_t incomplete=0xbc00;std::memcpy(bad.normal.bytes.data()+6,&incomplete,2);}
        if(failure==16) {const std::uint16_t infinity=0x7c00;std::memcpy(bad.color.bytes.data(),&infinity,2);}
        if(failure==17)bad.normal.height=2;
        if(failure==18)receipt.color=nullptr;
        if(!check(!failed.accept(stale,tile,receipt,error) && failed.invalidated() && sameAggregate(before,failed.aggregate()),
            "failed rough tile accepted, remained live or changed aggregate"))return false;
        const auto priorStats=stats;const auto priorOutput=output;
        if(!check(!failed.publish(key,stats,output,error) && sameAggregate(output,priorOutput)
            && stats.roughGlassCoveredPixels==priorStats.roughGlassCoveredPixels
            && stats.roughGlassReceiptBytes==priorStats.roughGlassReceiptBytes,
            "failed rough tile changed final publication"))return false;
    }
    Completion resized(plan);
    if(!resized.accept(key,plan->tiles()[0],storage[0].receipt(),error))return false;
    invalidKey=key;++invalidKey.height;
    if(!check(!resized.publish(invalidKey,stats,output,error) && resized.invalidated(),
              "resize did not invalidate partial rough frame"))return false;
    auto previewKey=key;
    auto previewInputs=*key.runtime;previewInputs.linearColor=false;
    previewKey.runtime=std::make_shared<const NativeFixtureSceneRuntimeInputs>(previewInputs);
    std::shared_ptr<const Schedule> previewPlan;
    if(!Schedule::admit(prepared,previewKey,false,previewPlan,error))return false;
    Completion preview(previewPlan);
    for(std::size_t i=0;i<2;++i) {
        auto pixels=storage[i];pixels.color.format=NativeTexturePixelFormat::Rgba8Unorm;
        pixels.color.bytes.resize(previewPlan->tiles()[i].pixels()*4);
        if(!preview.accept(previewKey,previewPlan->tiles()[i],pixels.receipt(),error))return false;
    }
    if(!preview.publish(previewKey,stats,output,error))return false;
    error.clear();return true;
}
// Contract-only scalar data. Native optical evidence comes from the backend fixtures.
inline bool roughMultipleReceiptCases(const arbitgpu::smoothglass::Prepared& v1,std::string& error)
{
    using namespace arbitgpu;
    namespace sg=smoothglass;
    namespace tiles=roughglasstiles;
    auto prepared=v1;prepared.roughConsumer.version=sg::kMultipleScatterVersion;prepared.roughConsumer.selectedComputeV2=true;
    const auto reject=[&](bool accepted,const char* message) {if(accepted)error=message;return !accepted;};
    std::uint64_t bytes=0;
    auto unsupported=prepared;unsupported.roughConsumer.selectedComputeV2=false;
    if(!reject(sg::admitFrame(unsupported,1,1,false,error,&bytes),"v2 admitted a fragment-only consumer"))return false;
    if(sg::kMaximumCameraMicroEvents!=98304 || sg::kMaximumCameraPixelMicroWork!=983040
        ||!sg::admitFrame(prepared,136,1,false,error,&bytes) || bytes!=136*48
        ||!reject(sg::admitFrame(prepared,137,1,false,error),"v2 two-pass micro-work tile bound bypassed"))return false;
    tiles::FrameKey key;
    key.scene=prepared.sourceScene;key.program=prepared.sourceMaterial;
    NativeFixtureSceneRuntimeInputs inputs;inputs.linearColor=true;
    key.runtime=std::make_shared<const NativeFixtureSceneRuntimeInputs>(inputs);
    key.owner=std::make_shared<const int>(29);key.camera=*findFixtureActiveCamera(*key.scene);
    key.sceneRevision=key.program->sceneRevision;key.rendererGeneration=11;key.width=137;key.height=1;
    std::shared_ptr<const tiles::Schedule> plan;
    if(!tiles::Schedule::admit(prepared,key,false,plan,error)||plan->tiles().size()!=2
        ||plan->maximumTilePixels()>136)return false;
    const auto storage=[&](const tiles::Tile& tile) {
        std::array<NativeRawPassPixels,7> value;
        for(unsigned i=0;i<7;++i) {
            value[i].width=tile.width;value[i].height=tile.height;
            value[i].format=i<3?NativeTexturePixelFormat::Rgba32Float:i<5?NativeTexturePixelFormat::R32Uint:NativeTexturePixelFormat::Rgba16Float;
            value[i].bytes.resize(tile.pixels()*(i<3?16:i<5?4:8));
        }
        for(std::size_t p=0;p<tile.pixels();++p) {
            const std::array<float,4> t{1,1,1,float(sg::kMultipleScatterCameraComplete)},m{0,0,0,3073},r{0,0,0,98304};
            std::memcpy(value[0].bytes.data()+p*16,t.data(),16);
            std::memcpy(value[1].bytes.data()+p*16,m.data(),16);
            std::memcpy(value[2].bytes.data()+p*16,r.data(),16);
            std::memcpy(value[3].bytes.data()+p*4,&prepared.object.value,4);
            std::memcpy(value[4].bytes.data()+p*4,&prepared.material.value,4);
        }
        return value;
    };
    auto pixels=storage(plan->tiles()[0]);
    const auto scan=[&](const auto& owner) {
        NativeFixtureSceneStats stats;
        return sg::completeRoughFrame(owner,pixels[0].width,1,pixels[0],pixels[1],pixels[2],pixels[3],pixels[4],stats,error);
    };
    if(!scan(prepared)||!reject(scan(v1),"v1 accepted a v2 status/event receipt"))return false;
    const auto put=[&](unsigned image,unsigned slot,float value){std::memcpy(pixels[image].bytes.data()+slot*4,&value,4);};
    put(0,3,1);if(!reject(scan(prepared),"v2 accepted a v1 completion version"))return false;
    put(0,3,33);
    for(const float events:{-1.f,1.5f,98305.f,std::numeric_limits<float>::infinity()}) {
        put(2,3,events);if(!reject(scan(prepared),"v2 invalid event work admitted"))return false;
    }
    put(2,3,98304);put(1,3,2);
    if(!reject(scan(prepared),"v2 micro-events exceeded scene-query ancestry"))return false;
    put(1,3,3073);
    put(0,0,.99999f);put(1,0,.00001f);
    if(!reject(scan(prepared),"v2 masked power hid a recovered walk"))return false;
    put(0,0,1);put(1,0,0);
    std::vector<tiles::NativeTileReceipt::Work> work;
    const auto receipt=[&](const auto& v) {
        work.resize(v[0].bytes.size()/16);
        for(std::size_t p=0;p<work.size();++p) {
            float queries=0,events=0;std::memcpy(&queries,v[1].bytes.data()+p*16+12,4);std::memcpy(&events,v[2].bytes.data()+p*16+12,4);
            work[p]={1,1,std::uint32_t(queries),std::uint32_t(events)};
        }
        tiles::NativeTileReceipt r;r.gpuCompleted=true;r.terminal=&v[0];r.masked=&v[1];r.tail=&v[2];
        r.objectIds=&v[3];r.materialIds=&v[4];r.normal=&v[5];r.color=&v[6];r.nativeWork=&work;return r;
    };
    for(unsigned corruption=0;corruption<7;++corruption) {
        auto corrupted=receipt(pixels);
        if(corruption==0)corrupted.nativeWork=nullptr;
        else if(corruption==1)work[0].starts=2;
        else if(corruption==2)work[0].selections=0;
        else if(corruption==3)--work[0].events;
        else if(corruption==4)--work[0].queries;
        else if(corruption==5)work.pop_back();
        else work.push_back({});
        tiles::Completion invalid(plan);
        if(!reject(invalid.accept(key,plan->tiles()[0],corrupted,error),"v2 accepted malformed actual native invocation work")
            ||!invalid.invalidated()||invalid.aggregate().completedTiles)return false;
    }
    auto nonglass=pixels;
    for(unsigned image=0;image<3;++image)std::memset(nonglass[image].bytes.data(),0,16);
    const std::uint32_t foreground=999;std::memcpy(nonglass[3].bytes.data(),&foreground,4);
    auto nonglassReceipt=receipt(nonglass);work[0]={1,0,1,0};
    tiles::Completion suppressed(plan);
    if(!suppressed.accept(key,plan->tiles()[0],nonglassReceipt,error)
        ||suppressed.aggregate().transportStarts!=plan->tiles()[0].pixels()-1
        ||suppressed.aggregate().selectionQueries!=plan->tiles()[0].pixels())return false;
    for(unsigned corruption=0;corruption<3;++corruption) {
        auto invalidReceipt=receipt(nonglass);work[0]={1,0,1,0};
        if(corruption==0)work[0].starts=1;
        else if(corruption==1)work[0].queries=2;
        else work[0].events=1;
        tiles::Completion invalid(plan);
        if(!reject(invalid.accept(key,plan->tiles()[0],invalidReceipt,error),"v2 accepted expensive work through nonglass foreground")
            ||!invalid.invalidated()||invalid.aggregate().completedTiles)return false;
    }
    tiles::Completion complete(plan);
    if(!complete.accept(key,plan->tiles()[0],receipt(pixels),error))return false;
    const auto before=complete.aggregate();auto second=storage(plan->tiles()[1]);
    if(!reject(complete.accept(key,plan->tiles()[1],receipt(second),error),"v2 whole-frame work overflow published")
        ||!complete.invalidated()||complete.aggregate().microWork!=before.microWork)return false;
    auto stale=key;stale.owner=std::make_shared<const int>(29);tiles::Completion owner(plan);
    if(!reject(owner.accept(stale,plan->tiles()[0],receipt(pixels),error),"v2 stale native owner accepted"))return false;
    // Lower actual event work permits all jobs while retaining deterministic totals.
    for(auto* storage:{&pixels,&second})for(std::size_t p=0;p<(*storage)[0].bytes.size()/16;++p) {
        const float events=2;std::memcpy((*storage)[2].bytes.data()+p*16+12,&events,4);
    }
    tiles::Completion accepted(plan);
    if(!accepted.accept(key,plan->tiles()[1],receipt(second),error)
        ||!accepted.accept(key,plan->tiles()[0],receipt(pixels),error))return false;
    NativeFixtureSceneStats stats;tiles::Aggregate aggregate;
    if(!accepted.publish(key,stats,aggregate,error)||aggregate.microEvents!=274||aggregate.microWork!=2740)return false;
    auto large=key;large.width=3840;large.height=2160;
    if(!reject(tiles::Schedule::admit(prepared,large,false,plan,error),"v2 unbounded full-size tile job schedule admitted"))return false;
    error.clear();return true;
}
inline bool roughNativeReceiptCases(videohelper::materialprogram::BackendTarget target,std::string& error)
{
    using namespace arbitgpu;
    using namespace arbitgpu::smoothglass;
    auto authored=scene();receiver(authored);
    auto snapshot=std::make_shared<const Visual3DScene>(authored);
    const auto admitted=binding(snapshot,target,1.5f,.004f,error);if(!admitted)return false;
    auto native=std::make_shared<NativeFixtureSurfaceMaterialProgram>(*admitted->nativeProgram());
    native->parameters.emissionRoughness[3]=.08f;
    const RoughNativeConsumer consumer{native->backend,true,true,true};
    Prepared prepared;
    const auto reject=[&](bool accepted,const char* message) {if(accepted)error=message;return !accepted;};
    if(!reject(prepare(authored,native.get(),{},prepared,error),"default nonowning caller admitted rough glass")
        ||!reject(prepare(snapshot,native,{},prepared,error),"default owning caller admitted rough glass"))return false;
    for(int missing=0;missing<4;++missing) {
        auto partial=consumer;
        if(missing==0)partial.shader=false;
        if(missing==1)partial.float32Receipt=false;
        if(missing==2)partial.readback=false;
        if(missing==3)partial.backend=native->backend==NativeFixtureMaterialBackend::Metal
            ?NativeFixtureMaterialBackend::OpenGl:NativeFixtureMaterialBackend::Metal;
        if(!reject(prepare(snapshot,native,{},prepared,error,partial),"partial or mismatched rough native consumer admitted"))return false;
    }
    for(const float roughness : {-1.f,1.1f,std::numeric_limits<float>::quiet_NaN()}) {
        auto invalid=std::make_shared<NativeFixtureSurfaceMaterialProgram>(*native);
        invalid->parameters.emissionRoughness[3]=roughness;
        if(!reject(prepare(snapshot,invalid,{},prepared,error,consumer),"native consumer admitted invalid roughness"))return false;
    }
    if(!prepare(snapshot,native,{},prepared,error,consumer))return false;
    if(prepared.roughness!=.08f || prepared.material!=authored.objects[0].material
        ||prepared.sourceScene!=snapshot ||prepared.sourceMaterial!=native)
    {error="rough prepared ownership, material identity or roughness was lost";return false;}
    std::uint64_t bytes=0;
    const auto maximumPixels=kMaximumPixelRayQueries/kMaximumRoughFrameRayQueries;
    if(kMaximumRoughFrameRayQueries!=55298 ||kRoughReceiptBytesPerPixel!=48
        ||!admitFrame(prepared,static_cast<std::uint32_t>(maximumPixels),1,false,error,&bytes)
        ||bytes!=maximumPixels*48
        ||!reject(admitFrame(prepared,static_cast<std::uint32_t>(maximumPixels+1),1,false,error,&bytes),"rough two-pass query limit exceeded")
        ||bytes!=0
        ||!reject(admitFrame(prepared,1920,1080,false,error),"rough 1080p work bound bypassed")
        ||!reject(admitFrame(prepared,2,1,true,error),"rough dynamic instances admitted"))return false;
    const auto pixels=[](NativeTexturePixelFormat format,std::size_t stride) {
        return NativeRawPassPixels{format,2,1,std::vector<std::uint8_t>(2*stride)};
    };
    auto terminal=pixels(NativeTexturePixelFormat::Rgba32Float,16);
    auto masked=terminal,tail=terminal;
    auto object=pixels(NativeTexturePixelFormat::R32Uint,4),material=object;
    const auto put=[](NativeRawPassPixels& value,std::size_t slot,float scalar) {
        std::memcpy(value.bytes.data()+slot*4,&scalar,4);
    };
    const auto putId=[](NativeRawPassPixels& value,std::size_t slot,std::uint32_t id) {
        std::memcpy(value.bytes.data()+slot*4,&id,4);
    };
    putId(object,0,prepared.object.value);putId(material,0,prepared.material.value);
    for(int c=0;c<3;++c) {put(terminal,c,.99998f);put(masked,c,.00001f);put(tail,c,.00001f);}
    put(terminal,3,1);put(masked,3,3073);put(tail,3,1);
    NativeFixtureSceneStats stats;
    const auto complete=[&]() {return completeRoughFrame(prepared,2,1,terminal,masked,tail,object,material,stats,error);};
    if(!complete() ||!stats.roughGlassReceiptComplete ||stats.roughGlassCoveredPixels!=1
        ||stats.roughGlassReceiptBytes!=96 ||stats.roughGlassMaximumEnergyResidual>kRoughEnergyConservationTolerance
        ||!(stats.roughGlassMaximumUnresolvedPower>0) ||stats.roughGlassMaximumUnresolvedPower>kMaximumRoughUnresolvedPower)
    {error="finite float32 rough receipt with nonzero tail did not produce complete stats";return false;}
    const auto unchanged=stats;
    const auto refused=[&](const char* message) {
        if(!reject(complete(),message))return false;
        if(stats.roughGlassReceiptComplete!=unchanged.roughGlassReceiptComplete
            ||stats.roughGlassCoveredPixels!=unchanged.roughGlassCoveredPixels
            ||stats.roughGlassReceiptBytes!=unchanged.roughGlassReceiptBytes
            ||stats.roughGlassMaximumEnergyResidual!=unchanged.roughGlassMaximumEnergyResidual
            ||stats.roughGlassMaximumUnresolvedPower!=unchanged.roughGlassMaximumUnresolvedPower)
        {error="failed rough receipt scan changed committed stats";return false;}
        return true;
    };
    for(int code=-1;code>=-10;--code) {
        put(terminal,3,float(code));put(tail,3,float(code));
        if(!refused("negative rough status accepted") ||error!=roughTransportDiagnostic(static_cast<RoughTransportStatus>(code)))return false;
    }
    for(const float invalid : {0.f,2.f,1.5f,std::numeric_limits<float>::infinity()}) {
        put(terminal,3,invalid);put(tail,3,invalid);
        if(!refused("invalid or smooth receipt certified rough transport"))return false;
    }
    put(terminal,3,1);put(tail,3,2);
    if(!refused("inconsistent rough completion statuses accepted"))return false;
    put(tail,3,1);
    for(const float count : {0.f,1.5f,3074.f,std::numeric_limits<float>::quiet_NaN()}) {
        put(masked,3,count);if(!refused("invalid rough query count accepted"))return false;
    }
    put(masked,3,3073);
    for(auto* plane : {&terminal,&masked,&tail}) for(int c=0;c<3;++c) {
        float original=0;std::memcpy(&original,plane->bytes.data()+c*4,4);
        for(const float value : {-1.f,2.f,std::numeric_limits<float>::quiet_NaN()}) {
            put(*plane,c,value);if(!refused("invalid rough wavelength energy accepted"))return false;
        }
        put(*plane,c,original);
    }
    put(terminal,0,.5f);if(!refused("nonconserving rough receipt accepted"))return false;
    put(terminal,0,.9998f);put(masked,0,.00019f);
    if(!refused("over-bound unresolved power accepted"))return false;
    put(terminal,0,.99998f);put(masked,0,.00001f);
    putId(material,0,prepared.material.value+1);
    if(!refused("rough receipt material mismatch accepted"))return false;
    putId(material,0,prepared.material.value);
    put(terminal,4,1);if(!refused("stale nonglass energy accepted"))return false;
    put(terminal,4,0);
    put(masked,7,1);if(!refused("stale nonglass query coverage accepted"))return false;
    put(masked,7,0);
    put(tail,4,std::numeric_limits<float>::quiet_NaN());
    if(!refused("nonfinite nonglass energy accepted"))return false;
    put(tail,4,0);
    // A later missing visible glass receipt must refuse the whole frame.
    putId(object,1,prepared.object.value);putId(material,1,prepared.material.value);
    if(!refused("missing second visible glass receipt accepted"))return false;
    putId(object,1,0);putId(material,1,0);
    terminal.width=1;if(!refused("mismatched rough receipt dimensions accepted"))return false;terminal.width=2;
    terminal.bytes.pop_back();if(!refused("truncated rough float32 receipt accepted"))return false;terminal.bytes.push_back(0);
    for(auto* plane : {&terminal,&masked,&tail}) {
        plane->format=NativeTexturePixelFormat::Rgba16Float;
        if(!refused("half-float energy attachment accepted"))return false;
        plane->format=NativeTexturePixelFormat::Rgba32Float;
    }
    object.format=NativeTexturePixelFormat::R32Float;
    if(!refused("float object ID attachment accepted"))return false;
    object.format=NativeTexturePixelFormat::R32Uint;
    if(!reject(completeRoughFrame(prepared,1,2,terminal,masked,tail,object,material,stats,error),"uniformly wrong frame dimensions accepted"))return false;
    auto smoothNative=std::make_shared<NativeFixtureSurfaceMaterialProgram>(*native);
    smoothNative->parameters.emissionRoughness[3]=0;
    auto thinNative=std::make_shared<NativeFixtureSurfaceMaterialProgram>(*native);
    thinNative->parameters.identifiers[1]=static_cast<std::uint32_t>(surfacematerial::TransportMode::ThinInterface);
    thinNative->parameters.transmissionIorClearcoat[3]=0;
    Prepared thin;
    if(!reject(prepare(snapshot,thinNative,{},thin,error),"rough water admitted without a native receipt consumer")
        ||!prepare(snapshot,thinNative,{},thin,error,consumer)
        ||thin.transport!=surfacematerial::TransportMode::ThinInterface ||thin.roughness!=.08f
        ||!admitFrame(thin,2,1,false,error,&bytes) ||bytes!=96)return false;
    put(masked,3,float(kMaximumRoughThinClosestHitQueries));
    if(!completeRoughFrame(thin,2,1,terminal,masked,tail,object,material,stats,error))return false;
    const auto waterStats=stats;
    put(masked,3,float(kMaximumRoughThinClosestHitQueries+1));
    if(!reject(completeRoughFrame(thin,2,1,terminal,masked,tail,object,material,stats,error),"rough water exceeded its one-interface query bound")
        ||stats.roughGlassCoveredPixels!=waterStats.roughGlassCoveredPixels)return false;
    put(masked,3,float(kMaximumRoughThinClosestHitQueries));
    put(terminal,3,-4);put(tail,3,-4);
    if(!reject(completeRoughFrame(thin,2,1,terminal,masked,tail,object,material,stats,error),"rough water re-hit status published")
        ||stats.roughGlassCoveredPixels!=waterStats.roughGlassCoveredPixels)return false;
    put(terminal,3,1);put(tail,3,1);
    putId(object,1,thin.object.value);putId(material,1,thin.material.value);
    if(!reject(completeRoughFrame(thin,2,1,terminal,masked,tail,object,material,stats,error),"missing later water receipt published")
        ||stats.roughGlassCoveredPixels!=waterStats.roughGlassCoveredPixels)return false;
    putId(object,1,0);putId(material,1,0);
    if(!roughTileCases(thin,error))return false;
    Prepared smooth;
    if(!prepare(authored,smoothNative.get(),{},smooth,error) ||smooth.roughness!=0
        ||!admitFrame(smooth,1920,1080,false,error,&bytes) ||bytes!=0)return false;
    NativeFixtureSceneStats smoothStats;
    const NativeRawPassPixels absent;
    if(!completeRoughFrame(smooth,2,1,absent,absent,absent,absent,absent,smoothStats,error)
        ||smoothStats.roughGlassReceiptComplete ||smoothStats.roughGlassReceiptBytes!=0)
    {error="smooth path invented rough certification or required rough receipts";return false;}
    if(!roughTileCases(prepared,error)||!roughMultipleReceiptCases(prepared,error))return false;
    error.clear();return true;
}
inline bool run(arbitgpu::NativeFixtureSceneBackend& backend,std::string& error)
{
    if(!roughPhysicsCases(error)||!roughSchedulerSourceCases(error))return false;
    const auto target=backend.info().backend=="metal"?videohelper::materialprogram::BackendTarget::Metal:videohelper::materialprogram::BackendTarget::OpenGl;
    if(!roughNativeReceiptCases(target,error))return false;
    const auto reject=[&](bool accepted,const char* message) {if(accepted)error=message;return !accepted;};
    // Native and double paths agree for parallel faces, nonparallel prism faces,
    // IOR/B, thickness, camera movement, internal TIR, and receiver appearance.
    struct Case {float slope,thickness,angle,ior,b;bool inside,receiver;bool spectralEdge=false;float environmentIntensity=1,objectOffset=0,receiverGain=1;};
    const Case rows[]={{0,1,0,1.5f,0,false,true},{0,1,20,1.5f,0,false,true},
        {.25f,1,20,1.5f,.004f,false,true},{.25f,1,20,1.8f,.004f,false,true},
        {.25f,1.4f,20,1.5f,.004f,false,true},{.25f,1,10,1.5f,.004f,false,true},
        {0,1,55,1.5f,0,true,false},{0,1,0,1,0,false,true},
        {.25f,1,20,1.5f,.004f,false,true,true},{.25f,1,20,1.5f,0,false,true,true},
        {.25f,1,20,1.5f,.004f,false,true,false,.5f,0,1},
        {.25f,1,20,1.5f,.004f,false,true,false,1,.15f,1},
        {.25f,1.4f,20,1.5f,.004f,false,true,false,1,0,.25f}};
    D3 baseline{},dispersive{},splitPixels{};
    std::size_t row=0;
    for(const auto& test:rows) {
        auto authored=scene(test.slope,test.thickness,test.angle,test.inside);
        authored.lights[0].intensity=test.environmentIntensity;
        authored.objects[0].transform.translation.x=test.objectOffset;
        const double radians=test.angle*3.14159265358979323846/180;
        const D3 camera{authored.cameras[0].transform.translation.x,0,authored.cameras[0].transform.translation.z};
        const D3 direction{-std::sin(radians),0,-std::cos(radians)};
        if(test.receiver) {
            float edge=-.435f;
            if(test.spectralEdge) {
                const auto split=trace(authored,camera,direction,test.ior,.004);
                const auto landing=[&](int c) {return split.firstExitPoint[c][0]
                    +(-3-split.firstExitPoint[c][2])/split.firstExit[c][2]*split.firstExit[c][0];};
                edge=float((landing(0)+landing(2))*.5);
            }
            receiver(authored,edge,{.8*test.receiverGain,.4*test.receiverGain,.1*test.receiverGain});
        }
        const auto expected=trace(authored,camera,direction,test.ior,test.b);
        if(!expected.complete||(expected.reflections==0&&test.ior!=1)) {error="independent glass optical oracle did not complete reflected paths";return false;}
        if(test.inside&&expected.tir==0) {error="independent internal slab ray did not exercise TIR";return false;}
        if(test.b==0 && (expected.firstExit[0]!=expected.firstExit[1]||expected.firstExit[1]!=expected.firstExit[2]))
            {error="zero dispersion split independent ray geometry";return false;}
        if(test.slope!=0&&test.b!=0 && std::abs(expected.firstExit[0][0]-expected.firstExit[2][0])<.001)
            {error="authored prism did not separate physical wavelength directions";return false;}
        auto snapshot=std::make_shared<const Visual3DScene>(std::move(authored));
        const auto material=binding(snapshot,target,test.ior,test.b,error);if(!material)return false;
        videorender::fixture3d::FixtureSceneRenderer renderer(backend);
        arbitgpu::NativeFixtureSceneRuntimeInputs inputs;inputs.linearColor=true;
        videorender::fixture3d::RenderedFrame preview,exported;
        if(!renderer.renderPreview(snapshot,material,{},inputs,{9,9},"native-gpu",preview,error)
            ||!renderer.renderExport(snapshot,material,{},inputs,{9,9},"native-gpu",exported,error)
            ||!preview.stats.smoothGlassTransportApplied||preview.stats.smoothGlassQueryBytes==0)
            {error="ordinary glass preview/export failed: "+error;return false;}
        arbitgpu::NativeRawPassPixels pixels,exportPixels;
        if(!preview.nativeFrame->readRawPass(renderpassoutput::Output::Color,pixels,error)
            ||!exported.nativeFrame->readRawPass(renderpassoutput::Output::Color,exportPixels,error)
            ||pixels.format!=arbitgpu::NativeTexturePixelFormat::Rgba16Float||pixels.bytes!=exportPixels.bytes)
            {error="glass same-frame preview/export pixel identity failed: "+error;return false;}
        D3 actual{};
        for(int c=0;c<3;++c) {
            std::uint16_t half=0;std::memcpy(&half,pixels.bytes.data()+(4*9+4)*8+c*2,2);actual[c]=hdrimage::halfToFloat(half);
            // Includes finite interface-tail omission and binary16 storage.
            if(!std::isfinite(actual[c])||std::abs(actual[c]-expected.rgb[c])>.001)
                {error="glass native row "+std::to_string(row)+" channel "+std::to_string(c)+" expected "+std::to_string(expected.rgb[c])+", got "+std::to_string(actual[c]);return false;}
        }
        if(row==0)baseline=actual;
        if(row==2)dispersive=actual;
        if(row==8)splitPixels=actual;
        if(row==9 && std::abs(actual[0]-splitPixels[0])<.1 && std::abs(actual[2]-splitPixels[2])<.1)
            {error="physical dispersion did not separate native receiver-edge pixels";return false;}
        std::cout<<"smooth-glass-pixels backend="<<backend.info().backend<<" row="<<row++<<" rgb="<<actual[0]<<','<<actual[1]<<','<<actual[2]<<'\n';
        if(row==1) {
            auto prepared=backend.prepare(snapshot,material->nativeProgram());if(!prepared.prepared){error=prepared.error;return false;}
            if(!reject(backend.render(snapshot,prepared.resources,3840,2160,inputs).rendered,"over-budget glass 4K frame admitted"))return false;
        }
    }
    if(std::abs(baseline[0]-dispersive[0])<.0005) {error="glass camera/prism controls did not change native radiance";return false;}
    auto authored=scene();receiver(authored);
    auto snapshot=std::make_shared<const Visual3DScene>(authored);
    const auto material=binding(snapshot,target,1.5f,.004f,error);if(!material)return false;
    const auto native=material->nativeProgram();
    const auto refuseScene=[&](Visual3DScene bad,const char* message) {
        auto candidate=std::make_shared<const Visual3DScene>(std::move(bad));
        return reject(backend.prepare(candidate,native).prepared,message);
    };
    auto bad=authored;bad.objects[0].indexCount-=3;
    if(!refuseScene(bad,"open glass shell admitted"))return false;
    bad=authored;std::swap(bad.indices[0],bad.indices[1]);
    if(!refuseScene(bad,"inconsistently wound glass shell admitted"))return false;
    bad=authored;bad.indices[0]=bad.indices[1];
    if(!refuseScene(bad,"degenerate glass face admitted"))return false;
    bad=authored;bad.objects[0].firstVertex=bad.vertexCount+1;
    if(!refuseScene(bad,"out-of-range glass vertex read admitted"))return false;
    bad=authored;bad.vertices[6].position=bad.vertices[0].position;
    if(!refuseScene(bad,"folded/self-contacting glass shell admitted"))return false;
    bad=authored;for(std::size_t i=0;i<36;i+=3)std::swap(bad.indices[i],bad.indices[i+1]);
    if(!refuseScene(bad,"inward glass volume admitted"))return false;
    bad=authored;bad.materials[1].baseColorTexture={1};
    if(!refuseScene(bad,"textured glass receiver admitted"))return false;
    bad=authored;bad.materials[1].baseColor={2,0,0};
    if(!refuseScene(bad,"imported receiver reflectance above one admitted"))return false;
    auto receiverProgram=std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*native);
    receiverProgram->object=authored.objects[1].id;
    receiverProgram->parameters={};
    receiverProgram->parameters.baseColorMetallic={2,0,0,0};
    receiverProgram->parameters.emissionRoughness={0,0,0,1};
    receiverProgram->parameters.normalOpacity={0,0,1,1};
    receiverProgram->parameters.transmissionIorClearcoat={0,1.5f,0,0};
    receiverProgram->parameters.identifiers[0]=authored.objects[1].material.value;
    auto receiverRoot=std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*native);
    receiverRoot->objectPrograms.push_back(receiverProgram);
    if(!reject(backend.prepare(snapshot,receiverRoot).prepared,"authored receiver reflectance above one admitted"))return false;
    auto excessive=scene();excessive.lights[0].color={1,1,1};excessive.lights[0].intensity=100000;
    auto excessiveSnapshot=std::make_shared<const Visual3DScene>(excessive);
    const auto excessiveMaterial=binding(excessiveSnapshot,target,1.5f,.004f,error);if(!excessiveMaterial)return false;
    const auto excessivePrepared=backend.prepare(excessiveSnapshot,excessiveMaterial->nativeProgram());
    if(!excessivePrepared.prepared) {error="finite high-radiance scene did not reach native output validation: "+excessivePrepared.error;return false;}
    arbitgpu::NativeFixtureSceneRuntimeInputs linear;linear.linearColor=true;
    const auto overflow=backend.render(excessiveSnapshot,excessivePrepared.resources,9,9,linear);
    if(overflow.rendered || overflow.frame || overflow.error.find("Color")==std::string::npos
        || overflow.error.find("RGBA16F")==std::string::npos)
    {error="unrepresentable native glass Color was published or lacked its output diagnostic";return false;}
    auto rough=std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*native);rough->parameters.emissionRoughness[3]=.1f;
    arbitgpu::smoothglass::Prepared unclaimedRough;
    std::string roughError;
    if(!reject(arbitgpu::smoothglass::prepare(snapshot,rough,{},unclaimedRough,roughError),"rough glass admitted without native energy receipts")
        ||roughError.find("terminal, masked and tail")==std::string::npos)
    {error="rough glass capability refusal did not explain missing native energy receipts";return false;}
    auto indexBound=std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*native);
    indexBound->parameters.transmissionIorClearcoat[1]=2.986f;
    if(!reject(backend.prepare(snapshot,indexBound).prepared,"visible-interval glass IOR overflow admitted"))return false;
    indexBound->parameters.transmissionIorClearcoat[1]=1.002f;
    if(!reject(backend.prepare(snapshot,indexBound).prepared,"visible-interval glass IOR underflow admitted"))return false;
    auto receiverNormal=authored;receiverNormal.vertices[8].normal={0,1,0};
    if(!refuseScene(receiverNormal,"unevaluated receiver normal interpolation admitted"))return false;
    auto tinted=std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*native);tinted->parameters.baseColorMetallic[0]=.5f;
    if(!reject(backend.prepare(snapshot,tinted).prepared,"ignored volume tint admitted"))return false;
    arbitgpu::smoothglass::Prepared topology;std::vector<videowire::geometry::InstanceAppearance> appearances(1);
    if(!reject(arbitgpu::smoothglass::prepare(authored,native.get(),appearances,topology,error),"glass appearance instances admitted"))return false;
    arbitgpu::NativeRawPassPixels incomplete{arbitgpu::NativeTexturePixelFormat::Rgba16Float,1,1,std::vector<std::uint8_t>(8)};
    std::uint16_t failure=0xbc00;std::memcpy(incomplete.bytes.data()+6,&failure,2);
    if(!reject(arbitgpu::smoothglass::completeFrame(incomplete,error),"incomplete native path accepted"))return false;
    // Each incomplete smooth code names its cause after the unchanged prefix.
    static_assert(arbitgpu::smoothglass::negativeHalfBits(-4)==0xc400u,"-4 binary16 encoding changed");
    static_assert(arbitgpu::smoothglass::negativeHalfBits(-5)==0xc500u,"-5 binary16 encoding changed");
    const std::string prefix="Closed glass ray transport did not produce a complete finite result (";
    const struct { int status; const char* expected; } refusals[]{
        {-1,"(no reason recorded; 1 pixel)"},
        {static_cast<int>(arbitgpu::smoothglass::SmoothTransportStatus::QueryRefused),"(query refused: invalid scene query; 1 pixel)"},
        {static_cast<int>(arbitgpu::smoothglass::SmoothTransportStatus::QueryBudget),"(query refused: work budget; 1 pixel)"},
        {static_cast<int>(arbitgpu::smoothglass::SmoothTransportStatus::FirstHitMismatch),"(first hit is not the glass volume; 1 pixel)"},
        {static_cast<int>(arbitgpu::smoothglass::SmoothTransportStatus::RayVisits),"(bounce budget exhausted; 1 pixel)"},
        // Thin water and rough refusal codes name their cause too.
        {static_cast<int>(arbitgpu::smoothglass::SmoothTransportStatus::ThinReentry),"(water receiver ray re-entered the water surface; 1 pixel)"},
        {static_cast<int>(arbitgpu::smoothglass::SmoothTransportStatus::ThinNonfiniteIncoming),"(nonfinite incoming ray or normal; 1 pixel)"},
        {static_cast<int>(arbitgpu::smoothglass::SmoothTransportStatus::ThinFirstHit),"(first hit is not the water surface; 1 pixel)"},
        {static_cast<int>(arbitgpu::smoothglass::SmoothTransportStatus::ThinReceiver),"(water receiver shading incomplete; 1 pixel)"},
        {static_cast<int>(arbitgpu::smoothglass::SmoothTransportStatus::ThinRadiance),"(nonfinite water radiance; 1 pixel)"},
        {arbitgpu::smoothglass::roughReasonCode(arbitgpu::smoothglass::RoughTransportStatus::InvalidInput),"(rough glass input not admitted; 1 pixel)"},
        {arbitgpu::smoothglass::roughReasonCode(arbitgpu::smoothglass::RoughTransportStatus::InternalCamera),"(rough glass refuses internal camera; 1 pixel)"},
        {arbitgpu::smoothglass::roughReasonCode(arbitgpu::smoothglass::RoughTransportStatus::IncompleteQuery),"(rough glass query incomplete; 1 pixel)"},
        {arbitgpu::smoothglass::roughReasonCode(arbitgpu::smoothglass::RoughTransportStatus::IdentityOrMedium),"(rough glass identity or medium inconsistent; 1 pixel)"},
        {arbitgpu::smoothglass::roughReasonCode(arbitgpu::smoothglass::RoughTransportStatus::InterfaceDepth),"(rough glass interface depth budget exhausted; 1 pixel)"},
        {arbitgpu::smoothglass::roughReasonCode(arbitgpu::smoothglass::RoughTransportStatus::PendingBranches),"(rough glass pending branch budget exhausted; 1 pixel)"},
        {arbitgpu::smoothglass::roughReasonCode(arbitgpu::smoothglass::RoughTransportStatus::RayVisits),"(rough glass ray query budget exhausted; 1 pixel)"},
        {arbitgpu::smoothglass::roughReasonCode(arbitgpu::smoothglass::RoughTransportStatus::UnresolvedEnergy),"(rough glass unresolved microfacet energy; 1 pixel)"},
        {arbitgpu::smoothglass::roughReasonCode(arbitgpu::smoothglass::RoughTransportStatus::Nonfinite),"(rough glass nonfinite scattering or radiance; 1 pixel)"},
        {arbitgpu::smoothglass::roughReasonCode(arbitgpu::smoothglass::RoughTransportStatus::NonconservingEnergy),"(rough glass energy receipt nonconserving; 1 pixel)"}};
    for(const auto& refusal:refusals) {
        arbitgpu::NativeRawPassPixels coded{arbitgpu::NativeTexturePixelFormat::Rgba16Float,1,1,std::vector<std::uint8_t>(8)};
        const std::uint16_t bits=arbitgpu::smoothglass::negativeHalfBits(refusal.status);
        std::memcpy(coded.bytes.data()+6,&bits,2);
        std::string message;
        if(arbitgpu::smoothglass::completeFrame(coded,message)
            ||message.compare(0,prefix.size(),prefix)!=0||message.find(refusal.expected)==std::string::npos)
        {error="smooth refusal "+std::to_string(refusal.status)+" did not name its cause: "+message;return false;}
    }
    std::fill(incomplete.bytes.begin(),incomplete.bytes.end(),0);
    std::uint16_t infinity=0x7c00;std::memcpy(incomplete.bytes.data(),&infinity,2);
    if(!reject(arbitgpu::smoothglass::finiteColorFrame(incomplete,error),"nonfinite owned glass Color accepted"))return false;
    error.clear();return true;
}
} // namespace videohelper::tests::smoothglassfixture
