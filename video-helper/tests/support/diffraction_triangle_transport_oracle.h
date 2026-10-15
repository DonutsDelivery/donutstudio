#pragma once

#include "../../../shared/DiffractionTriangleTransport.h"
#include "../../src/material_program_compiler.h"
#include "../../src/gpu_backend/fixture_opaque_visibility.h"
#include <memory>

namespace triangletransporttests
{
namespace transport = diffractiontransport;
using D3 = std::array<double,3>;
using Scene = HarmonicMIDI::grid::Visual3DScene;
// Separate oracle vector operations and plane/edge intersection. It never reads
// a BVH payload or calls closestHitCpu, product samples, or product evaluate.
inline D3 difference(D3 a,D3 b) {return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
inline double inner(D3 a,D3 b) {return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
inline D3 exterior(D3 a,D3 b) {return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
inline D3 normalized(D3 a) {double length=std::sqrt(inner(a,a));for(auto& c:a)c/=length;return a;}
struct WorldTriangle { std::array<D3,3> p; std::uint32_t object,material; };
inline std::vector<WorldTriangle> worldTriangles(const Scene& scene)
{
    std::vector<WorldTriangle> result;
    for(std::size_t i=0;i<scene.objectCount;++i)
    {
        const auto& o=scene.objects[i];
        // Fixtures carry composed world vertices and identity object transforms.
        const auto count=o.indexCount?o.indexCount:o.vertexCount;
        for(std::size_t t=0;t<count;t+=3)
        {
            WorldTriangle triangle{{},o.id.value,o.material.value};
            for(unsigned corner=0;corner<3;++corner)
            {
                const auto local=o.indexCount?scene.indices[o.firstIndex+t+corner]:t+corner;
                const auto& p=scene.vertices[o.firstVertex+local].position;
                triangle.p[corner]={p.x,p.y,p.z};
            }
            result.push_back(triangle);
        }
    }
    return result;
}
inline transport::Hit intersection(const std::vector<WorldTriangle>& triangles,const transport::Ray& ray)
{
    transport::Hit result;result.status=0;result.sceneRevision=1;result.distance=ray.maximum;
    for(const auto& triangle:triangles)
    {
        const auto normal=exterior(difference(triangle.p[1],triangle.p[0]),difference(triangle.p[2],triangle.p[0]));
        const double denominator=inner(normal,ray.direction);
        if(std::abs(denominator)<1e-14)continue;
        const double distance=inner(normal,difference(triangle.p[0],ray.origin))/denominator;
        if(distance<ray.minimum||distance>result.distance)continue;
        D3 position=ray.origin;for(unsigned a=0;a<3;++a)position[a]+=distance*ray.direction[a];
        bool inside=true;
        for(unsigned edge=0;edge<3;++edge)
            if(inner(normal,exterior(difference(triangle.p[(edge+1)%3],triangle.p[edge]),difference(position,triangle.p[edge])))<-1e-12)inside=false;
        if(inside) {result.status=1;result.identity={triangle.object,triangle.material};result.distance=distance;result.normal=normalized(normal);}
    }
    return result;
}
struct QuadraturePoint {D3 p,normal;double weight;};
inline std::vector<QuadraturePoint> quadrature(const std::vector<WorldTriangle>& triangles,std::uint32_t object)
{
    // Six-point Gauss-Legendre in each Duffy coordinate, independent of the
    // production equal-area subdivision and triangle-centroid rule.
    constexpr std::array<double,6> x{.033765242898423975,.16939530676686775,.38069040695840155,
                                    .61930959304159845,.83060469323313225,.966234757101576025};
    constexpr std::array<double,6> w{.085662246189585175,.1803807865240693,.2339569672863455,
                                    .2339569672863455,.1803807865240693,.085662246189585175};
    std::vector<QuadraturePoint> result;
    for(const auto& triangle:triangles)if(triangle.object==object)
    {
        const auto a=difference(triangle.p[1],triangle.p[0]),b=difference(triangle.p[2],triangle.p[0]);
        const auto c=exterior(a,b);const double determinant=std::sqrt(inner(c,c));
        const auto normal=normalized(c);
        for(unsigned i=0;i<x.size();++i)for(unsigned j=0;j<x.size();++j)
        {
            D3 p=triangle.p[0];for(unsigned k=0;k<3;++k)p[k]+=x[i]*a[k]+(1-x[i])*x[j]*b[k];
            result.push_back({p,normal,determinant*(1-x[i])*w[i]*w[j]});
        }
    }
    return result;
}
inline transport::Result oracle(const Scene& scene,const transport::Input& input)
{
    const auto triangles=worldTriangles(scene);
    const auto receiver=quadrature(triangles,input.receiver.sceneIdentity.object),grating=quadrature(triangles,input.gratingIdentity.object);
    transport::Result result;
    for(const auto& p:receiver)result.receiverArea+=p.weight;
    for(const auto& p:grating)result.gratingArea+=p.weight;
    double exchange=0;
    for(const auto& r:receiver)
    {
        const auto lightHit=intersection(triangles,{r.p,input.source.direction,input.rayMinimumDistance,input.source.traceMaximumDistance});
        const double illumination=lightHit.status?0:std::max(0.0,inner(r.normal,input.source.direction));
        for(unsigned band=0;band<8;++band)
        {
            result.receiverIncidentPower[band]+=r.weight*input.source.radiance[band]*illumination;
            result.receiverOutgoingPower[band]+=r.weight*input.source.radiance[band]*illumination*input.receiver.reflectance->coefficients[band];
        }
        for(const auto& g:grating)
        {
            const auto delta=difference(g.p,r.p);const double squared=inner(delta,delta),distance=std::sqrt(squared);
            const auto direction=normalized(delta),back=normalized(difference(r.p,g.p));
            const auto hitG=intersection(triangles,{r.p,direction,input.rayMinimumDistance,distance+input.rayMinimumDistance});
            const auto hitR=intersection(triangles,{g.p,back,input.rayMinimumDistance,distance+input.rayMinimumDistance});
            if(!hitG.status||!hitR.status||hitG.identity.object!=input.gratingIdentity.object||
               hitR.identity.object!=input.receiver.sceneIdentity.object||std::abs(hitG.distance-distance)>1e-8||std::abs(hitR.distance-distance)>1e-8)continue;
            const double weight=r.weight*g.weight*std::max(0.0,inner(r.normal,direction))*std::max(0.0,inner(g.normal,back))/(3.14159265358979323846*squared);
            exchange+=weight;
            for(unsigned band=0;band<8;++band)result.arrivingPower[band]+=weight*input.source.radiance[band]*illumination*input.receiver.reflectance->coefficients[band];
        }
    }
    result.formFactor=exchange/result.receiverArea;
    for(unsigned band=0;band<8;++band)
    {
        result.meanReceiverIrradiance[band]=result.receiverIncidentPower[band]/result.receiverArea;
        result.meanReceiverRadiance[band]=result.receiverOutgoingPower[band]/(3.14159265358979323846*result.receiverArea);
        result.meanGratingIrradiance[band]=result.arrivingPower[band]/result.gratingArea;
    }
    return result;
}
// Hottel's identical directly opposed finite-rectangle factor, C-11 in Howell's
// catalog: https://www.thermalradiation.net/sectionc/C-11.html . The analytic
// comparison checks the fixture; it never supplies an actual receiver plane.
inline double rectangleFactor(double a,double b,double separation)
{
    const double x=a/separation,y=b/separation;
    return 2/(3.14159265358979323846*x*y)*
        (.5*std::log((1+x*x)*(1+y*y)/(1+x*x+y*y))+
         x*std::sqrt(1+y*y)*std::atan(x/std::sqrt(1+y*y))+
         y*std::sqrt(1+x*x)*std::atan(y/std::sqrt(1+x*x))-x*std::atan(x)-y*std::atan(y));
}
struct Fixture
{
    std::shared_ptr<Scene> scene=std::make_shared<Scene>();
    arbitgpu::NativeFixtureSurfaceMaterialProgram program;
    transport::Input input;
    arbitgpu::opaquevisibility::Prepared prepared;
    arbitgpu::opaquevisibility::Frame frame;
    void addObject(std::array<D3,4> p,std::uint32_t object,std::uint32_t material,bool reverse=false)
    {
        using namespace HarmonicMIDI::grid;
        auto& o=scene->objects[scene->objectCount++];o.id={object};o.material={material};o.firstVertex=scene->vertexCount;o.vertexCount=4;
        o.firstIndex=scene->indexCount;o.indexCount=6;
        auto& m=scene->materials[scene->materialCount++];m.id={material};m.alphaMode=SceneAlphaMode::Opaque;m.opacity=1;
        std::array<unsigned,6> indices=reverse?std::array<unsigned,6>{0,2,1,0,3,2}:std::array<unsigned,6>{0,1,2,0,2,3};
        for(const auto& point:p)
        {
            auto& v=scene->vertices[scene->vertexCount++];v.position={float(point[0]),float(point[1]),float(point[2])};v.color={1,1,1,1};
        }
        for(auto index:indices)scene->indices[scene->indexCount++]=index;
    }
    void extract()
    {
        input.receiverTriangles.clear();input.gratingTriangles.clear();
        for(const auto& t:worldTriangles(*scene))
        {
            transport::Triangle triangle{t.p,normalized(exterior(difference(t.p[1],t.p[0]),difference(t.p[2],t.p[0])))};
            if(t.object==input.receiver.sceneIdentity.object)input.receiverTriangles.push_back(triangle);
            if(t.object==input.gratingIdentity.object)input.gratingTriangles.push_back(triangle);
        }
    }
    bool prepare(std::string& error)
    {
        using namespace arbitgpu::opaquevisibility;
        if(!prepareScene(*scene,&program,{},prepared,error,AdmissionMode::allSurfaces))return false;
        const Matrix identity{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};std::vector<Instance> instances;
        for(std::size_t i=0;i<scene->objectCount;++i)instances.push_back({identity,i});
        return arbitgpu::opaquevisibility::frame(prepared,instances,frame,error);
    }
    transport::Hit query(const transport::Ray& ray) const
    {
        const auto convert=[](D3 p)->arbitgpu::opaquevisibility::Point{return {float(p[0]),float(p[1]),float(p[2])};};
        const auto hit=arbitgpu::opaquevisibility::closestHitCpu(prepared,frame,convert(ray.origin),convert(ray.direction),float(ray.minimum),float(ray.maximum));
        return {hit.status,{hit.objectId,hit.materialId},1,{hit.outwardNormal[0],hit.outwardNormal[1],hit.outwardNormal[2]},hit.distance,true};
    }
};
inline Fixture fixture(bool nonplanar=false,double move=0,int blocker=0)
{
    Fixture f;f.scene->id={1};
    f.addObject({{{-.5+move,-.4,0},{.5+move,-.4,0},{.5+move,.4,nonplanar?.25:0},{-.5+move,.4,0}}},101,201);
    f.addObject({{{-.5,-.4,2},{.5,-.4,2},{.5,.4,2},{-.5,.4,2}}},102,202,true);
    if(blocker==1)f.addObject({{{-1,-1,1},{1,-1,1},{1,1,1},{-1,1,1}}},103,203);
    if(blocker==2)f.addObject({{{1.4,-1,1},{2.6,-1,1},{2.6,1,1},{1.4,1,1}}},103,203);
    f.program.object={101};f.program.kind=arbitgpu::NativeFixtureMaterialKind::SurfacePbr;
    f.program.parameters.normalOpacity={0,0,1,1};f.program.parameters.transmissionIorClearcoat={0,1.5,0,0};f.program.parameters.identifiers[0]=201;
    surfacematerial::SpectralReflectanceDescription rho;rho.coefficients={.05f,.12f,.25f,.48f,.72f,.91f,.37f,.18f};f.program.spectralReflectance=rho;
    const auto& receiverObject=f.scene->objects[0];
    const auto* receiverProgram=arbitgpu::nativeSurfaceForObject(&f.program,receiverObject.id);
    f.input.receiver={{receiverObject.id.value,receiverObject.material.value},
        {receiverProgram->object.value,receiverProgram->parameters.identifiers[0]},
        receiverProgram->parameters.identifiers[0],1,1,receiverProgram->spectralReflectance,
        true,false,false,false,1,0};
    f.input.gratingIdentity={102,202};f.input.source={normalized({2,0,1}),{.2,.7,1.3,.4,2.1,.9,1.7,.3},20};
    f.extract();return f;
}
} // namespace triangletransporttests
