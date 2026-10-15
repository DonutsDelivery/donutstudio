#pragma once

#include <deque>
#include <tuple>

#include "smooth_glass_cases.h"
#include "diffraction_native_transport_fixture.h"
#include "../../src/gpu_backend/fixture_glass_caustics.h"

namespace glasscaustictests
{
using namespace HarmonicMIDI::grid;
using D3=std::array<double,3>;
using Program=arbitgpu::NativeFixtureSurfaceMaterialProgram;
namespace gc=arbitgpu::glasscaustics;
inline D3 minus(D3 a,D3 b) {for(unsigned c=0;c<3;++c)a[c]-=b[c];return a;}
inline D3 plus(D3 a,D3 b,double scale=1) {for(unsigned c=0;c<3;++c)a[c]+=scale*b[c];return a;}
inline double inner(D3 a,D3 b) {return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
inline D3 exterior(D3 a,D3 b) {return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
inline D3 normalized(D3 a) {const double length=std::sqrt(inner(a,a));for(auto& c:a)c/=length;return a;}
inline D3 doublePoint(gc::Point a) {return {a[0],a[1],a[2]};}
struct WorldTriangle
{
    std::array<D3,3> p;
    D3 normal;
    double area;
    std::uint32_t object,material,instance,triangle;
};
inline std::vector<WorldTriangle> worldTriangles(const Visual3DScene& scene)
{
    std::vector<WorldTriangle> result;
    for(std::size_t o=0;o<scene.objectCount;++o)
    {
        const auto& object=scene.objects[o];
        // Fixtures author translations only. This independent world conversion
        // neither consumes the acceleration payload nor calls its transform.
        const auto& translation=object.transform.translation;
        for(std::size_t t=0;t<object.indexCount/3;++t)
        {
            WorldTriangle triangle{};
            triangle.object=object.id.value;triangle.material=object.material.value;
            triangle.instance=std::uint32_t(o);triangle.triangle=std::uint32_t(t);
            for(unsigned v=0;v<3;++v)
            {
                const auto& p=scene.vertices[object.firstVertex+scene.indices[object.firstIndex+3*t+v]].position;
                triangle.p[v]={double(p.x)+translation.x,double(p.y)+translation.y,double(p.z)+translation.z};
            }
            const auto area=exterior(minus(triangle.p[1],triangle.p[0]),minus(triangle.p[2],triangle.p[0]));
            triangle.area=.5*std::sqrt(inner(area,area));triangle.normal=normalized(area);result.push_back(triangle);
        }
    }
    return result;
}
struct Hit
{
    bool found=false,entering=false;
    double distance=0;
    D3 position{},normal{};
    std::uint32_t object=0,material=0,instance=0,triangle=0;
};
// Binary64 plane intersection and oriented edge half-spaces. The product uses
// its float closest-hit BVH; this oracle enumerates the authored triangles.
inline Hit closest(const std::vector<WorldTriangle>& triangles,D3 origin,D3 ray,double minimum,double maximum)
{
    Hit best;best.distance=maximum;
    for(const auto& t:triangles)
    {
        const auto n=exterior(minus(t.p[1],t.p[0]),minus(t.p[2],t.p[0]));
        const double denominator=inner(n,ray);
        if(std::abs(denominator)<1e-14)continue;
        const double distance=inner(n,minus(t.p[0],origin))/denominator;
        if(distance<minimum || distance>best.distance)continue;
        const auto p=plus(origin,ray,distance);bool inside=true;
        for(unsigned e=0;e<3;++e)
            if(inner(n,exterior(minus(t.p[(e+1)%3],t.p[e]),minus(p,t.p[e]))) < -1e-12)inside=false;
        if(!inside)continue;
        if(best.found && distance==best.distance && std::tie(t.instance,t.triangle)>=std::tie(best.instance,best.triangle))continue;
        best={true,denominator<0,distance,p,t.normal,t.object,t.material,t.instance,t.triangle};
    }
    return best;
}
struct Optics {D3 direction{};double reflection=0;bool tir=false;};
// Construct transmission from the tangential component. Fresnel uses optical
// admittances, separately from the product ratio/refract implementation.
inline Optics snell(D3 ray,D3 normal,double from,double to)
{
    Optics result;
    if(from==to) {result.direction=ray;return result;}
    const double cosine=-inner(ray,normal);
    const auto tangent=plus(ray,normal,cosine);
    const double sine=from/to*std::sqrt(inner(tangent,tangent));
    if(sine>1) {result.tir=true;result.reflection=1;return result;}
    const double transmittedCosine=std::sqrt(std::max(0.0,1-sine*sine));
    result.direction=normalized(plus(plus(D3{},tangent,from/to),normal,-transmittedCosine));
    const double s0=from*cosine,s1=to*transmittedCosine,p0=cosine/from,p1=transmittedCosine/to;
    result.reflection=.5*(std::pow((s0-s1)/(s0+s1),2)+std::pow((p0-p1)/(p0+p1),2));
    return result;
}
struct MicrofacetReferenceExit {D3 direction{};double power=0;bool transmitted=false;unsigned order=0;};
struct MicrofacetReference
{
    int status=-1;
    std::vector<MicrofacetReferenceExit> exits;
    unsigned events=0,draws=0;
    double residual=1;
};
// Independent binary64 Smith-height reference, evaluated from the published
// CDF/phase-function equations, not the product float32 or generated source.
// Physical unit flux; transmission does not apply a radiance eta^2 factor.
inline MicrofacetReference microfacetReference(D3 incoming,D3 normal,double from,double to,double roughness,
    std::uint32_t seed,unsigned eventLimit=32,unsigned exitLimit=16,double maximumResidual=1e-4/128)
{
    MicrofacetReference result;
    const auto valid=[](D3 v){return std::all_of(v.begin(),v.end(),[](double x){return std::isfinite(x);});};
    if(!valid(incoming) || !valid(normal) || std::abs(inner(incoming,incoming)-1)>1e-4
        || std::abs(inner(normal,normal)-1)>1e-4 || inner(incoming,normal)>=0
        || !std::isfinite(from) || !std::isfinite(to) || from<1 || from>3 || to<1 || to>3
        || !std::isfinite(roughness) || roughness<=0 || roughness>1 || !eventLimit || eventLimit>32
        || !exitLimit || exitLimit>16 || !std::isfinite(maximumResidual) || maximumResidual<=0 || maximumResidual>1e-4)return result;
    if(from==to) {result.exits.push_back({incoming,1,true,0});result.residual=0;result.status=1;return result;}
    const auto uniform=[&](unsigned dimension) {
        std::uint32_t bits=seed+0x9e3779b9u*(dimension+1u);
        bits=(bits^(bits>>16))*0x85ebca6bu;bits=(bits^(bits>>13))*0xc2b2ae35u;bits^=bits>>16;
        return (double(bits>>9)+.5)/8388608.0;
    };
    const double alpha=roughness*roughness;
    const auto rate=[&](double c) {
        if(c==0)return 0.0;
        const double slope=alpha*alpha*std::max(0.0,1-c*c);
        const double projected=std::hypot(c,std::sqrt(slope));
        return c>0?slope/(2*c*(projected+c)):-.5*(1+projected/(-c));
    };
    const auto survival=[&](double c,double cdf){return c>0?std::pow(cdf,rate(c)):0.0;};
    double cdf=1,remaining=1;D3 travel=incoming;bool otherMedium=false;
    for(unsigned bounce=0;bounce<eventLimit;++bounce)
    {
        const auto outward=plus(D3{},normal,otherMedium?-1:1);
        const double cosine=inner(travel,outward),escaped=survival(cosine,cdf);
        const double draw=uniform(4*bounce);++result.draws;
        if(cosine!=0) {
            const double lambda=rate(cosine);if(lambda==0)return result;
            // Conditional height CDF inverse: q/(1-U*(1-Pescape))^(1/Lambda).
            cdf/=std::pow(1-draw*(1-escaped),1/lambda);
        }
        if(!std::isfinite(cdf) || cdf<=0 || cdf>=1)return result;
        const auto x=normalized(exterior(outward,std::abs(outward[2])<.999?D3{0,0,1}:D3{1,0,0}));
        const auto y=exterior(outward,x);
        const auto view=normalized(D3{-alpha*inner(travel,x),-alpha*inner(travel,y),-inner(travel,outward)});
        const double capCosine=1-uniform(4*bounce+1)*(1+view[2]);
        const double capRadius=std::sqrt(std::max(0.0,1-capCosine*capCosine));
        const double azimuth=2*std::acos(-1.0)*uniform(4*bounce+2);
        result.draws+=2;
        const auto half=normalized(plus(view,D3{capRadius*std::cos(azimuth),capRadius*std::sin(azimuth),capCosine}));
        const auto local=normalized(D3{alpha*half[0],alpha*half[1],half[2]});
        const auto facet=plus(plus(plus(D3{},x,local[0]),y,local[1]),outward,local[2]);
        if(!valid(facet) || inner(travel,facet)>=0)return result;
        ++result.events;
        const auto optics=snell(travel,facet,otherMedium?to:from,otherMedium?from:to);
        const double selection=uniform(4*bounce+3);++result.draws;
        if(optics.tir || selection<optics.reflection)travel=normalized(plus(travel,facet,-2*inner(travel,facet)));
        else {travel=optics.direction;otherMedium=!otherMedium;cdf=1-cdf;}
        if(!valid(travel) || !std::isfinite(cdf) || cdf<=0 || cdf>=1)return result;
        const double probability=survival(inner(travel,normal)*(otherMedium?-1:1),cdf);
        const double exitPower=remaining*probability;
        if(exitPower>0) {
            if(result.exits.size()>=exitLimit) {result.status=0;return result;}
            result.exits.push_back({travel,exitPower,otherMedium,result.events});
        }
        remaining*=1-probability;result.residual=remaining;
        if(remaining<=maximumResidual) {result.status=1;return result;}
    }
    result.status=0;return result;
}
struct Path
{
    bool complete=true,sourceBlocked=false;
    Hit receiver;
    double reflected=0,transmitted=1;
};
inline Path trace(const std::vector<WorldTriangle>& triangles,D3 entry,D3 normal,D3 toSource,
    std::uint32_t glass,double ior,double maximum)
{
    Path result;
    // Float centroid inputs need not lie exactly on the binary64 triangle.
    // Recover its authored interface with the independent triangle query before
    // applying the oracle's smaller interior bias and source visibility bound.
    const auto initial=closest(triangles,plus(entry,toSource,1e-6),plus(D3{},toSource,-1),0,2e-6);
    if(!initial.found || initial.object!=glass || !initial.entering) {result.complete=false;return result;}
    entry=initial.position;normal=initial.normal;
    if(closest(triangles,entry,toSource,1e-8,maximum).found) {result.sourceBlocked=true;return result;}
    auto ray=plus(D3{},toSource,-1);
    auto at=entry;bool inside=false;
    for(unsigned depth=0;depth<64;++depth)
    {
        const auto optics=snell(ray,normal,inside?ior:1,inside?1:ior);
        result.reflected+=result.transmitted*optics.reflection;
        result.transmitted*=1-optics.reflection;
        if(optics.tir)return result;
        ray=optics.direction;inside=!inside;
        const auto hit=closest(triangles,plus(at,normal,-1e-8),ray,1e-9,maximum);
        if(!hit.found) {result.complete=!inside;return result;}
        if(hit.object!=glass) {result.complete=!inside;result.receiver=hit;return result;}
        if(inside==hit.entering) {result.complete=false;return result;}
        at=hit.position;normal=hit.normal;if(inside)normal=plus(D3{},normal,-1);
    }
    result.complete=false;return result;
}
inline std::shared_ptr<const Program> source(gc::Point direction={.19611613f,0,.98058068f},
    std::array<float,8> radiance={.25f,.5f,.75f,1,1.25f,1.5f,1.75f,2})
{
    // Reuse the admitted v3 source owner fixture, replacing its exact authored
    // descriptor and the dependent native receipt rather than grafting Li on RGB.
    auto owner=std::make_shared<Program>(triangletransporttests::nativeFixture().program);
    owner->objectPrograms.clear();
    auto lighting=owner->diffractionLightingAdmission->description();
    lighting.paths[0].incident.direction=direction;lighting.paths[0].incident.radiance=radiance;
    lighting.traceMaximumDistance=20;
    std::string error;
    const auto admitted=diffractionmaterial::AdmittedLightingPlan::admit(lighting,error);
    if(!admitted)return {};
    owner->diffractionLightingAdmission=std::make_shared<const diffractionmaterial::AdmittedLightingPlan>(*admitted);
    auto description=diffractionmaterial::makeAluminiumBinaryGratingPreset();
    description.spectrum.firstOrder=description.spectrum.lastOrder=1;description.roughness.rmsSlope=.01f;
    const auto material=diffractionmaterial::admit(description,error);
    if(!material)return {};
    owner->diffractionPaths[0]=diffractionmaterial::makeGpuLightingPath(
        diffractionmaterial::physicalcheckpoint::makeGpuParameters(*material,lighting.paths[0].incident),lighting.paths[0]);
    owner->diffractionPaths[0].material.incident={0,0,1,0};
    owner->programIdentity=arbitgpu::nativeFixtureDiffractionProgramIdentity(owner->bindingDigest,*admitted,0,1,{},false,owner.get());
    return owner;
}
struct Fixture
{
    std::shared_ptr<Visual3DScene> scene;
    std::shared_ptr<Program> program;
    std::shared_ptr<const Program> light;
    SceneLightId ordinaryLight {};
    std::shared_ptr<const gc::AdmittedBranchPolicy> policy;
    std::optional<arbitgpu::NativeFixtureSceneRuntimeInputs> runtime;
    double referenceIor=1.5,dispersion=.004;
    gc::Prepared prepared;
    gc::Frame frame;
    bool prepare(std::string& error,arbitgpu::smoothglass::RoughNativeConsumer consumer={})
    {
        arbitgpu::smoothglass::Prepared glass;
        const std::shared_ptr<const Visual3DScene> snapshot=scene;
        const std::shared_ptr<const Program> root=program;
        if(!arbitgpu::smoothglass::prepare(snapshot,root,{},glass,error,consumer))return false;
        if(ordinaryLight.isValid())
        {if(!policy || !gc::prepare(glass,ordinaryLight,*policy,prepared,error))return false;}
        else if(!gc::prepare(glass,light,prepared,error))return false;
        std::vector<arbitgpu::opaquevisibility::Instance> instances;
        for(std::size_t o=0;o<scene->objectCount;++o)
        {
            auto m=videohelper::tests::sceneclosest::identity();const auto& p=scene->objects[o].transform.translation;
            m[12]=p.x;m[13]=p.y;m[14]=p.z;instances.push_back({m,o});
        }
        return runtime?gc::buildFrame(prepared,instances,*runtime,32,32,frame,error)
            :gc::buildFrame(prepared,instances,32,32,frame,error);
    }
};
inline Fixture fixture(float dispersion=.004f,int blocker=0,bool diffractionSource=true)
{
    Fixture f;f.scene=videohelper::tests::sceneclosest::slab();
    // This focused fixture is self-contained. The generic backend makeScene
    // implementation also owns GLB decoding, which this triangle oracle needs
    // neither to link nor to execute.
    f.scene->id={1};f.scene->objects[0].id={101};f.scene->objects[0].material={6};
    f.scene->materials[0]={};f.scene->materials[0].id={6};
    f.scene->materials[0].baseColor={1,1,1};f.scene->materials[0].roughness=0;
    f.scene->materials[0].opacity=1;
    for(std::size_t v=0;v<f.scene->vertexCount;++v)
    {
        auto& vertex=f.scene->vertices[v];
        if(vertex.position.z>0)vertex.position.z+=.25f*vertex.position.x;
        vertex.normal={0,0,1};vertex.tangent={1,0,0,1};
    }
    f.scene->cameraCount=1;f.scene->cameras[0].id={401};f.scene->activeCamera={401};
    f.scene->cameras[0].transform.translation={0,0,3};
    f.scene->lightCount=1;f.scene->lights[0].id={301};
    f.scene->lights[0].kind=SceneLightKind::Environment;f.scene->lights[0].color={.125f,.25f,.5f};
    videohelper::tests::smoothglassfixture::receiver(*f.scene,-6,{0,0,0});
    if(diffractionSource)f.light=source();
    f.dispersion=dispersion;
    if(blocker)
    {
        const auto first=f.scene->vertexCount;
        videohelper::tests::smoothglassfixture::receiver(*f.scene,-6,{0,0,0});
        for(std::size_t v=first;v<f.scene->vertexCount;++v)f.scene->vertices[v].position.z=blocker==1?2:-1.5f;
    }
    for(std::size_t o=0;o<f.scene->objectCount;++o)
    {
        auto p=std::make_shared<Program>();
        p->backend=arbitgpu::NativeFixtureMaterialBackend::OpenGl;p->object=f.scene->objects[o].id;
        p->bindingDigest="caustic-fixture-binding-"+std::to_string(o);p->programIdentity="caustic-fixture-program-"+std::to_string(o);
        p->authoredScene=f.scene->id;p->authoredMaterial=f.scene->objects[o].material;
        p->sceneRevision=p->structuralRevision=p->evaluationRevision=p->materialRevision=1;
        p->parameters.identifiers[0]=p->authoredMaterial.value;
        p->parameters.normalOpacity={0,0,1,1};p->parameters.baseColorMetallic={1,1,1,0};
        p->parameters.transmissionIorClearcoat={0,1.5f,0,0};p->parameters.emissionRoughness={0,0,0,1};
        if(o==0)
        {
            p->parameters.identifiers[1]=std::uint32_t(surfacematerial::TransportMode::ClosedVolume);
            p->parameters.transmissionIorClearcoat={1,1.5f,0,dispersion};p->parameters.emissionRoughness[3]=0;f.program=p;
        }
        else
        {
            p->spectralReflectance=surfacematerial::SpectralReflectanceDescription{};
            p->spectralReflectance->coefficients=o==1?std::array<float,8>{.1f,.2f,.3f,.4f,.5f,.6f,.7f,.8f}:std::array<float,8>{};
            f.program->objectPrograms.push_back(p);
        }
    }
    return f;
}
inline Fixture ordinaryFixture()
{
    auto f=fixture(.004f,0,false);
    auto& light=f.scene->lights[0];light.kind=SceneLightKind::Directional;
    light.spectralSource.version=1;light.spectralSource.traceMaximumDistance=20;
    light.spectralSource.radiance={.25f,.5f,.75f,1,1.25f,1.5f,1.75f,2};
    const double angle=std::atan(.2);
    light.transform.rotation={0,float(std::sin(angle/2)),0,float(std::cos(angle/2))};
    f.ordinaryLight=light.id;
    gc::BranchPolicyDescription description;description.version=gc::kBranchPolicyVersion;
    description.maximumUnresolvedFraction=1e-4f;description.minimumBranchFraction=1e-5f;
    std::string error;const auto admitted=gc::AdmittedBranchPolicy::admit(description,error);
    if(admitted)f.policy=std::make_shared<const gc::AdmittedBranchPolicy>(*admitted);
    return f;
}
// A finite rough prism control with no grazing side-entry quadrature. The
// sloped top still refracts eight independently authored wavelengths.
inline Fixture roughFixture(float roughness=.05f)
{
    auto f=ordinaryFixture();
    f.scene->lights[0].transform.rotation={0,0,0,1};
    f.scene->cameras[0].transform.translation.z=-2;
    f.program->parameters.emissionRoughness[3]=roughness;
    f.scene->materials[0].roughness=roughness;
    return f;
}
struct AuthoredSource
{
    D3 direction;
    std::array<double,8> radiance;
    double maximum;
};
inline AuthoredSource authoredSource(const Fixture& fixture)
{
    AuthoredSource result{};
    if(fixture.ordinaryLight.isValid())
    {
        const SceneLightRecord* record=nullptr;
        for(std::size_t i=0;i<fixture.scene->lightCount;++i)
            if(fixture.scene->lights[i].id==fixture.ordinaryLight)record=&fixture.scene->lights[i];
        if(fixture.runtime && fixture.runtime->lightOverride)record=&*fixture.runtime->lightOverride;
        else if(fixture.runtime)for(const auto& light:fixture.runtime->animatedLights)
            if(light.id==fixture.ordinaryLight)record=&light;
        // Independent binary64 quaternion-vector rotation, without calling the
        // product's direction conversion or reading a prepared source receipt.
        const auto& q=record->transform.rotation;
        const double length=std::sqrt(double(q.x)*q.x+double(q.y)*q.y+double(q.z)*q.z+double(q.w)*q.w);
        const D3 vector{q.x/length,q.y/length,q.z/length},axis{0,0,1};
        const auto cross=exterior(vector,axis);
        result.direction=normalized(plus(plus(axis,cross,2*q.w/length),exterior(vector,cross),2));
        for(unsigned b=0;b<8;++b)result.radiance[b]=record->spectralSource.radiance[b];
        result.maximum=record->spectralSource.traceMaximumDistance;
    }
    else
    {
        const auto& light=fixture.light->diffractionLightingAdmission->description();
        result.direction=doublePoint(light.paths[0].incident.direction);
        for(unsigned b=0;b<8;++b)result.radiance[b]=light.paths[0].incident.radiance[b];
        result.maximum=light.traceMaximumDistance;
    }
    return result;
}
struct Result
{
    bool complete=true;
    std::array<double,8> incident{},reflected{},escaped{},receiverPower{},outgoingPower{},meanRadiance{};
    std::array<D3,8> landingMoment{},landingMean{};
};
inline Result oracle(const Fixture& fixture)
{
    Result result;
    const auto triangles=worldTriangles(*fixture.scene);
    const auto light=authoredSource(fixture);
    const auto direction=light.direction;
    double receiverArea=0;
    for(const auto& triangle:triangles)if(triangle.object==fixture.scene->objects[1].id.value)receiverArea+=triangle.area;
    // Independent six-point Gauss-Legendre Duffy quadrature, rather than the
    // contract's uniform subtriangle centroid sampling.
    constexpr std::array<double,6> nodes{.033765242898423975,.16939530676686775,.38069040695840155,
        .61930959304159845,.83060469323313225,.966234757101576025};
    constexpr std::array<double,6> weights{.085662246189585175,.1803807865240693,.2339569672863455,
        .2339569672863455,.1803807865240693,.085662246189585175};
    constexpr std::array<double,8> wavelengths{.380,.420,.460,.500,.540,.580,.640,.700};
    for(const auto& t:triangles)
    {
        if(t.object!=fixture.scene->objects[0].id.value || inner(t.normal,direction)<=0)continue;
        const auto e1=minus(t.p[1],t.p[0]),e2=minus(t.p[2],t.p[0]);
        for(unsigned u=0;u<nodes.size();++u)for(unsigned v=0;v<nodes.size();++v)
        {
            const auto point=plus(plus(t.p[0],e1,nodes[u]),e2,(1-nodes[u])*nodes[v]);
            const double projectedArea=2*t.area*weights[u]*weights[v]*(1-nodes[u])*inner(t.normal,direction);
            for(unsigned band=0;band<8;++band)
            {
                const double ior=fixture.referenceIor+fixture.dispersion*(1/(wavelengths[band]*wavelengths[band])-1/(.58756*.58756));
                const auto path=trace(triangles,point,t.normal,direction,t.object,ior,light.maximum);
                result.complete=result.complete&&path.complete;if(path.sourceBlocked)continue;
                const double input=projectedArea*light.radiance[band];
                result.incident[band]+=input;result.reflected[band]+=input*path.reflected;
                if(path.receiver.found && path.receiver.entering && path.receiver.object==fixture.scene->objects[1].id.value)
                {
                    const auto* receiver=arbitgpu::nativeSurfaceForObject(fixture.program.get(),SceneObjectId{path.receiver.object});
                    result.receiverPower[band]+=input*path.transmitted;
                    result.outgoingPower[band]+=input*path.transmitted*receiver->spectralReflectance->coefficients[band];
                    result.landingMoment[band]=plus(result.landingMoment[band],path.receiver.position,input*path.transmitted);
                }
                else result.escaped[band]+=input*path.transmitted;
            }
        }
    }
    for(unsigned b=0;b<8;++b)
    {
        if(result.receiverPower[b]>0)result.landingMean[b]=plus(D3{},result.landingMoment[b],1/result.receiverPower[b]);
        result.meanRadiance[b]=result.outgoingPower[b]/(3.14159265358979323846*receiverArea);
    }
    return result;
}

inline Fixture reflectionFixture(int blocker=0)
{
    auto f=fixture();
    const auto append=[&](float z,float right,bool black,bool downward) {
        const auto object=f.scene->objectCount;
        videohelper::tests::smoothglassfixture::receiver(*f.scene,-6,{0,0,0});
        auto& record=f.scene->objects[object];
        for(std::size_t v=record.firstVertex;v<record.firstVertex+record.vertexCount;++v)
        {
            auto& vertex=f.scene->vertices[v];vertex.position.z=z;
            if(vertex.position.x>0)vertex.position.x=right;
            if(downward)vertex.normal={0,0,-1};
        }
        if(downward)for(std::size_t t=record.firstIndex;t<record.firstIndex+record.indexCount;t+=3)
            std::swap(f.scene->indices[t+1],f.scene->indices[t+2]);
        auto surface=std::make_shared<Program>(*f.program->objectPrograms[0]);
        surface->object=record.id;surface->authoredMaterial=record.material;surface->parameters.identifiers[0]=record.material.value;
        surface->programIdentity="caustic-reflected-receiver-"+std::to_string(object);
        surface->bindingDigest="caustic-reflected-binding-"+std::to_string(object);
        surface->spectralReflectance->coefficients.fill(black?0:.7f);
        f.program->objectPrograms.push_back(surface);
    };
    // A real bounded offset triangle receiver faces the reflected light. It
    // does not lie on the authored incoming directional-source leg.
    append(2,-1.1f,false,true);
    if(blocker==1)append(2.5f,6,true,false);
    if(blocker==2)append(1.2f,-.4f,true,true);
    return f;
}
struct RoughOptics
{
    Optics snell;
    D3 reflectedDirection {}, facet {};
    double reflected = 0, transmitted = 0, masked = 0;
};
// Binary64 visible-normal reference. It uses authored geometry and optical
// admittances, independently of the product float kernel and scene query.
inline RoughOptics roughOptics(D3 ray,D3 normal,double from,double to,double roughness,
    unsigned sample,std::size_t depth)
{
    RoughOptics out;
    if(from==to) {out.snell.direction=ray;out.transmitted=1;out.facet=normal;return out;}
    const double alpha=roughness*roughness;
    const auto xAxis=normalized(exterior(normal,std::abs(normal[2])<.999?D3{0,0,1}:D3{1,0,0}));
    const auto yAxis=exterior(normal,xAxis),view=plus(D3{},ray,-1);
    const auto visible=normalized(D3{alpha*inner(view,xAxis),alpha*inner(view,yAxis),inner(view,normal)});
    const auto diskX=visible[2]<.99999?normalized(D3{-visible[1],visible[0],0}):D3{1,0,0};
    const auto diskY=exterior(visible,diskX);
    const unsigned stratum=(sample*(2*depth+1)+5*depth)%16;
    double azimuth=0;for(unsigned digit=0;digit<4;++digit)azimuth+=double((stratum>>digit)&1)/double(2u<<digit);
    azimuth+=depth*.6180339887498948482;azimuth-=std::floor(azimuth);
    const double radius=std::sqrt((stratum+.5)/16),angle=6.2831853071795864769*azimuth;
    const double x=radius*std::cos(angle),blend=(1+visible[2])/2;
    const double y=(1-blend)*std::sqrt(std::max(0.0,1-x*x))+blend*radius*std::sin(angle);
    const auto projected=plus(plus(plus(D3{},diskX,x),diskY,y),visible,std::sqrt(std::max(0.0,1-x*x-y*y)));
    const auto facetLocal=normalized(D3{alpha*projected[0],alpha*projected[1],std::max(0.0,projected[2])});
    out.facet=plus(plus(plus(D3{},xAxis,facetLocal[0]),yAxis,facetLocal[1]),normal,facetLocal[2]);
    out.snell=snell(ray,out.facet,from,to);
    out.reflectedDirection=normalized(plus(ray,out.facet,-2*inner(ray,out.facet)));
    const auto visibility=[&](double cosine) {
        if(cosine<=0)return 0.0;
        const double slopeSquared=(1-cosine*cosine)/(cosine*cosine);
        return 2/(1+std::sqrt(1+alpha*alpha*slopeSquared));
    };
    out.reflected=out.snell.reflection*visibility(inner(out.reflectedDirection,normal));
    out.transmitted=out.snell.tir?0:(1-out.snell.reflection)*visibility(-inner(out.snell.direction,normal));
    out.masked=1-out.reflected-out.transmitted;return out;
}
struct BranchArrival
{
    Hit receiver;
    double fraction;
    std::uint64_t history;
    std::size_t interfaces,reflections;
};
struct BranchInterface
{
    Hit surface;
    double incident,reflected,transmitted;
    std::uint64_t history;
    std::size_t depth;
    bool tir;
    D3 facet {};
    double masked=0;
};
struct BranchOracle
{
    bool valid=true,sourceBlocked=false;
    double deposited=0,escaped=0,unresolved=0,reflectedDeposited=0,masked=0;
    std::size_t tirEvents=0;
    std::vector<BranchArrival> arrivals;
    std::vector<BranchInterface> interfaces;
};
// Breadth-first binary64 reference over authored world triangles. Its much
// smaller declared residual and larger work limits independently bound the
// omitted high-order power when comparing native-policy finite prefixes.
inline BranchOracle branchOracle(const Fixture& fixture,const gc::EntrySample& entry,unsigned band,
    double minimumFraction=1e-13,std::size_t maximumVisits=4096,std::size_t maximumInterfaces=64,
    std::optional<D3> quadraturePosition={},unsigned interfaceSample=0)
{
    BranchOracle result;
    const auto triangles=worldTriangles(*fixture.scene);
    const auto light=authoredSource(fixture);
    const auto towardSource=normalized(light.direction);
    constexpr std::array<double,8> wavelengths{.380,.420,.460,.500,.540,.580,.640,.700};
    const double ior=fixture.referenceIor+fixture.dispersion*(1/(wavelengths[band]*wavelengths[band])-1/(.58756*.58756));
    const auto point=quadraturePosition.value_or(doublePoint(entry.position));
    if(closest(triangles,point,towardSource,1e-5,light.maximum).found)
    {result.sourceBlocked=true;return result;}
    struct Photon
    {
        D3 origin,direction;
        double weight;
        std::uint64_t history;
        std::size_t depth,reflections;
        bool inside;
    };
    std::deque<Photon> queue{{plus(point,towardSource,1e-6),plus(D3{},towardSource,-1),1,0,0,0,false}};
    std::size_t visits=0;
    while(!queue.empty() && visits++<maximumVisits)
    {
        const auto photon=queue.front();queue.pop_front();
        const auto hit=closest(triangles,photon.origin,photon.direction,photon.depth?1e-9:0,light.maximum);
        if(!hit.found)
        {
            if(photon.inside) {result.valid=false;result.unresolved+=photon.weight;}else result.escaped+=photon.weight;
            continue;
        }
        if(hit.object!=fixture.scene->objects[0].id.value)
        {
            if(photon.inside || !hit.entering) {result.valid=false;result.unresolved+=photon.weight;continue;}
            result.deposited+=photon.weight;if(photon.reflections)result.reflectedDeposited+=photon.weight;
            result.arrivals.push_back({hit,photon.weight,photon.history,photon.depth,photon.reflections});continue;
        }
        if(photon.inside==hit.entering) {result.valid=false;result.unresolved+=photon.weight;continue;}
        if(photon.depth>=maximumInterfaces || photon.depth>=64) {result.unresolved+=photon.weight;continue;}
        auto normal=hit.normal;if(photon.inside)normal=plus(D3{},normal,-1);
        auto optics=snell(photon.direction,normal,photon.inside?ior:1,photon.inside?1:ior);
        auto reflectedDirection=normalized(plus(photon.direction,normal,-2*inner(photon.direction,normal)));
        double reflectedWeight=optics.reflection,transmittedWeight=1-optics.reflection,maskedWeight=0;
        auto facet=normal;
        if(fixture.program->parameters.emissionRoughness[3]!=0) {
            const auto rough=roughOptics(photon.direction,normal,photon.inside?ior:1,photon.inside?1:ior,
                fixture.program->parameters.emissionRoughness[3],interfaceSample,photon.depth);
            optics=rough.snell;reflectedDirection=rough.reflectedDirection;
            reflectedWeight=rough.reflected;transmittedWeight=rough.transmitted;
            maskedWeight=rough.masked;facet=rough.facet;
            result.masked+=photon.weight*rough.masked;
        }
        if(optics.tir)++result.tirEvents;
        const auto append=[&](D3 direction,double fraction,bool reflected) {
            if(fraction==0)return;
            if(fraction<minimumFraction) {result.unresolved+=fraction;return;}
            queue.push_back({plus(hit.position,normal,reflected?1e-8:-1e-8),direction,fraction,
                photon.history|(reflected?(std::uint64_t(1)<<photon.depth):0),photon.depth+1,
                photon.reflections+(reflected?1:0),reflected?photon.inside:!photon.inside});
        };
        const double reflected=photon.weight*reflectedWeight;
        const double transmitted=fixture.program->parameters.emissionRoughness[3]==0?photon.weight-reflected:photon.weight*transmittedWeight;
        result.interfaces.push_back({hit,photon.weight,reflected,transmitted,photon.history,photon.depth,optics.tir,
            facet,photon.weight*maskedWeight});
        append(reflectedDirection,reflected,true);
        if(!optics.tir)append(optics.direction,transmitted,false);
    }
    for(const auto& photon:queue)result.unresolved+=photon.weight;
    return result;
}

struct PowerOracle
{
    bool complete = true;
    gc::EnergyLedger energy;
    std::array<D3,8> centroid {}, moment {};
};
inline PowerOracle powerOracle(const Fixture& fixture)
{
    PowerOracle result;const auto triangles=worldTriangles(*fixture.scene);const auto light=authoredSource(fixture);
    for(const auto& work:fixture.frame.work)
    {
        const auto& entry=fixture.frame.entries[work.entry];
        const auto triangle=std::find_if(triangles.begin(),triangles.end(),[&](const auto& t) {
            return std::tie(t.object,t.material,t.instance,t.triangle)==std::tie(entry.object,entry.material,entry.instance,entry.triangle);
        });
        if(triangle==triangles.end()) {result.complete=false;return result;}
        const auto n=fixture.frame.scheduling.entrySubdivisions;
        const double incident=light.radiance[work.band]*std::max(0.0,inner(triangle->normal,light.direction))*triangle->area/(n*n*fixture.frame.interfaceSamples);
        const auto path=branchOracle(fixture,entry,work.band,1e-13,4096,64,{},work.interfaceSample);result.complete=result.complete && path.valid;
        auto& energy=result.energy;const auto b=work.band;energy.potential[b]+=incident;
        if(path.sourceBlocked) {energy.sourceBlocked[b]+=incident;continue;}
        energy.launched[b]+=incident;energy.escaped[b]+=incident*path.escaped;energy.unresolved[b]+=incident*path.unresolved;energy.masked[b]+=incident*path.masked;
        for(const auto& arrival:path.arrivals)
        {
            const auto* material=arbitgpu::nativeSurfaceForObject(fixture.program.get(),SceneObjectId{arrival.receiver.object});
            if(!material || !material->spectralReflectance) {result.complete=false;return result;}
            const double power=incident*arrival.fraction,rho=material->spectralReflectance->coefficients[b];
            energy.deposited[b]+=power*rho;energy.absorbed[b]+=power*(1-rho);
            result.moment[b]=plus(result.moment[b],arrival.receiver.position,power*rho);
        }
    }
    for(unsigned b=0;b<8;++b)if(result.energy.deposited[b]>0)
        result.centroid[b]=plus(D3{},result.moment[b],1/result.energy.deposited[b]);
    return result;
}
// Independent six-point Gauss-Legendre Duffy quadrature through the complete
// binary64 branch oracle. It samples authored triangles directly, rather than
// consuming the product entry lattice or its native deposits. The residual is
// bounded by branchOracle's separately declared 1e-13 policy.
inline PowerOracle quadraturePowerOracle(const Fixture& fixture)
{
    PowerOracle result;const auto triangles=worldTriangles(*fixture.scene);const auto source=authoredSource(fixture);
    constexpr std::array<double,6> nodes{.033765242898423975,.16939530676686775,.38069040695840155,
        .61930959304159845,.83060469323313225,.966234757101576025};
    constexpr std::array<double,6> weights{.085662246189585175,.1803807865240693,.2339569672863455,
        .2339569672863455,.1803807865240693,.085662246189585175};
    for(const auto& triangle:triangles)
    {
        const auto cosine=inner(triangle.normal,source.direction);
        if(triangle.object!=fixture.scene->objects[0].id.value || cosine<=0)continue;
        const auto e1=minus(triangle.p[1],triangle.p[0]),e2=minus(triangle.p[2],triangle.p[0]);
        for(unsigned u=0;u<6;++u)for(unsigned v=0;v<6;++v)
        {
            const auto point=plus(plus(triangle.p[0],e1,nodes[u]),e2,(1-nodes[u])*nodes[v]);
            gc::EntrySample sample;
            for(unsigned axis=0;axis<3;++axis)
            {sample.position[axis]=float(point[axis]);sample.normal[axis]=float(triangle.normal[axis]);}
            const double area=2*triangle.area*weights[u]*weights[v]*(1-nodes[u])*cosine;
            for(unsigned interfaceSample=0;interfaceSample<fixture.frame.interfaceSamples;++interfaceSample)for(unsigned band=0;band<8;++band)
            {
                const auto path=branchOracle(fixture,sample,band,1e-13,4096,64,point,interfaceSample);result.complete=result.complete&&path.valid;
                const double incident=area*source.radiance[band]/fixture.frame.interfaceSamples;auto& energy=result.energy;
                energy.potential[band]+=incident;
                if(path.sourceBlocked) {energy.sourceBlocked[band]+=incident;continue;}
                energy.launched[band]+=incident;energy.escaped[band]+=incident*path.escaped;
                energy.unresolved[band]+=incident*path.unresolved;energy.masked[band]+=incident*path.masked;
                for(const auto& arrival:path.arrivals)
                {
                    const auto* material=arbitgpu::nativeSurfaceForObject(fixture.program.get(),SceneObjectId{arrival.receiver.object});
                    if(!material || !material->spectralReflectance) {result.complete=false;return result;}
                    const double power=incident*arrival.fraction,rho=material->spectralReflectance->coefficients[band];
                    energy.deposited[band]+=power*rho;energy.absorbed[band]+=power*(1-rho);
                    result.moment[band]=plus(result.moment[band],arrival.receiver.position,power*rho);
                }
            }
        }
    }
    for(unsigned b=0;b<8;++b)if(result.energy.deposited[b]>0)
        result.centroid[b]=plus(D3{},result.moment[b],1/result.energy.deposited[b]);
    return result;
}
// Explicitly synthetic contract readback. Paths come from the diagnostic
// contract, camera rays from independent binary64 triangle enumeration. This
// never represents GPU evidence and is never used by product rendering.
inline arbitgpu::opaquevisibility::ClosestHit nativeHit(const Hit& hit)
{
    arbitgpu::opaquevisibility::ClosestHit out;out.status=hit.found?1:0;
    out.work=1;out.distance=float(hit.distance);out.position={float(hit.position[0]),float(hit.position[1]),float(hit.position[2])};
    out.outwardNormal={float(hit.normal[0]),float(hit.normal[1]),float(hit.normal[2])};out.entering=hit.entering;
    out.objectId=hit.object;out.materialId=hit.material;out.instanceId=hit.instance;out.triangleId=hit.triangle;return out;
}
inline bool syntheticReadback(const Fixture& fixture,gc::NativeReadback& output,std::string& error)
{
    const auto& frame=fixture.frame;gc::NativeReadback candidate;candidate.frameIdentity=frame.identity;
    candidate.spectralRadiance.resize(std::size_t(frame.width)*frame.height*8);
    const auto triangles=worldTriangles(*fixture.scene);
    const D3 camera{frame.camera.transform.translation.x,frame.camera.transform.translation.y,frame.camera.transform.translation.z};
    if(frame.camera.transform.rotation.x!=0 || frame.camera.transform.rotation.y!=0 || frame.camera.transform.rotation.z!=0)
    {error="Synthetic camera fixture supports translation only";return false;}
    const double tangent=std::tan(double(frame.camera.verticalFovRadians)/2),aspect=double(frame.width)/frame.height;
    for(std::size_t w=0;w<frame.work.size();++w)
    {
        const auto& work=frame.work[w];gc::BranchReceipt branches;
        if(!gc::traceBranches(fixture.prepared,frame,frame.entries[work.entry],work.band,*fixture.policy,branches,error,work.interfaceSample))return false;
        gc::NativePathReceipt receipt;receipt.work=std::uint32_t(w);receipt.complete=true;receipt.sourceBlocked=branches.sourceBlocked;
        const auto& entry=frame.entries[work.entry];
        const D3 toSource{frame.sourceDirectionTrace[0],frame.sourceDirectionTrace[1],frame.sourceDirectionTrace[2]};
        receipt.sourceHit=nativeHit(closest(triangles,doublePoint(entry.position),toSource,1e-5,frame.sourceDirectionTrace[3]));
        if(!branches.sourceBlocked)
        {
            const auto offset=gc::bias(entry.position);
            const auto origin=gc::advance(entry.position,{float(toSource[0]),float(toSource[1]),float(toSource[2])},2*offset);
            receipt.entryHit=nativeHit(closest(triangles,doublePoint(origin),plus(D3{},toSource,-1),0,4*offset));
        }
        receipt.visits=std::uint32_t(branches.visits);
        receipt.queryWork=receipt.sourceHit.work+(branches.sourceBlocked?0:receipt.entryHit.work)+branches.visits;
        receipt.launchedPower=branches.sourceBlocked?0:work.incidentPower;
        receipt.escapedPower=float(work.incidentPower*branches.escapedFraction);
        receipt.unresolvedPower=float(work.incidentPower*branches.unresolvedFraction);
        receipt.maskedPower=float(work.incidentPower*branches.maskedFraction);
        if(frame.roughness!=0) {receipt.roughTransportVersion=gc::kRoughTransportVersion;receipt.interfaceSample=work.interfaceSample;}
        double outgoing=0,absorbed=0;
        for(const auto& branch:branches.deposits)
        {
            gc::NativeDeposit deposit;deposit.work=std::uint32_t(w);deposit.reflectionHistory=branch.reflectionHistory;
            deposit.interfaces=std::uint32_t(branch.interfaces);deposit.fraction=float(branch.fraction);deposit.receiver=branch.receiver;
            const auto* receiver=arbitgpu::nativeSurfaceForObject(fixture.program.get(),SceneObjectId{branch.receiver.objectId});
            const double power=double(work.incidentPower)*deposit.fraction,rho=receiver->spectralReflectance->coefficients[work.band];
            deposit.incidentPower=float(power);deposit.outgoingPower=float(power*rho);deposit.absorbedPower=float(power*(1-rho));
            outgoing+=deposit.outgoingPower;absorbed+=deposit.absorbedPower;
            if(!gc::projectReceiver(frame,deposit.receiver,deposit.footprint,error))return false;
            if(deposit.footprint.projected)
            {
                const auto delta=minus(doublePoint(deposit.receiver.position),camera);const double distance=std::sqrt(inner(delta,delta));
                deposit.cameraHit=nativeHit(closest(triangles,camera,normalized(delta),frame.camera.nearPlane,distance+1e-3));
                const auto x=deposit.footprint.x,y=deposit.footprint.y;
                const auto ray=normalized(D3{(2*(x+.5)/frame.width-1)*tangent*aspect,(1-2*(y+.5)/frame.height)*tangent,-1});
                deposit.pixelHit=nativeHit(closest(triangles,camera,ray,frame.camera.nearPlane,frame.camera.farPlane));
                deposit.cameraVisible=gc::sameTriangle(deposit.receiver,deposit.cameraHit)
                    && deposit.cameraHit.status==1 && deposit.pixelHit.status==1 && gc::sameTriangle(deposit.receiver,deposit.pixelHit);
                if(deposit.cameraVisible)
                {
                    deposit.raster={deposit.receiver.objectId,deposit.receiver.materialId,deposit.receiver.instanceId,
                        deposit.receiver.triangleId,float(camera[2]-deposit.pixelHit.position[2])};
                    const auto pixel=(std::size_t(y)*frame.width+x)*8+work.band;
                    candidate.spectralRadiance[pixel]+=float(deposit.outgoingPower/(3.14159265358979323846*deposit.footprint.worldArea));
                }
            }
            receipt.queryWork+=deposit.receiver.work;
            if(deposit.footprint.projected)receipt.queryWork+=std::uint64_t(deposit.cameraHit.work)+deposit.pixelHit.work;
            candidate.deposits.push_back(deposit);++receipt.depositCount;
        }
        receipt.depositedPower=float(outgoing);receipt.absorbedPower=float(absorbed);candidate.paths.push_back(receipt);
    }
    output=std::move(candidate);error.clear();return true;
}
// Synthetic ordering diagnostic, independent of the product comparator and
// validator. It does not manufacture a native candidate or product image.
inline gc::DepositionReceipt syntheticDeposition(const gc::Frame& frame,const gc::NativeReadback& readback,
    const gc::SubmissionIdentity& owner)
{
    gc::DepositionReceipt result;result.version=gc::kDepositionVersion;result.submission=owner;
    result.complete=true;std::vector<std::uint32_t> slots(frame.work.size());
    const auto visits=frame.bounds.maximumDeposits/frame.work.size();
    for(const auto& deposit:readback.deposits)
    {
        const auto slot=slots[deposit.work]++;
        if(deposit.cameraVisible)result.keys.push_back({deposit.footprint.y*frame.width+deposit.footprint.x,
            deposit.work,slot,std::uint32_t(deposit.work*visits+slot)});
    }
    std::sort(result.keys.begin(),result.keys.end(),[](const auto& a,const auto& b) {
        if(a.pixel!=b.pixel)return a.pixel<b.pixel;
        if(a.work!=b.work)return a.work<b.work;
        return a.slot<b.slot;
    });
    result.visibleCount=result.keys.size();result.keys.resize(frame.bounds.deposition.paddedKeyCount);return result;
}
struct DepositionMoment
{
    std::array<double,8> power{},x{},y{},xx{},yy{};
    std::map<std::uint64_t,float> radiance;
};
inline DepositionMoment depositionMoment(const gc::Frame& frame,const gc::NativeReadback& readback)
{
    struct Record {std::uint32_t pixel,work,slot,band;const gc::NativeDeposit* deposit;};
    std::vector<Record> records;std::vector<std::uint32_t> slots(frame.work.size());
    for(const auto& deposit:readback.deposits)
    {
        const auto slot=slots[deposit.work]++;
        if(deposit.cameraVisible)records.push_back({deposit.footprint.y*frame.width+deposit.footprint.x,
            deposit.work,slot,frame.work[deposit.work].band,&deposit});
    }
    std::sort(records.begin(),records.end(),[](const auto& a,const auto& b) {
        return std::tie(a.pixel,a.work,a.slot)<std::tie(b.pixel,b.work,b.slot);
    });
    DepositionMoment result;
    for(const auto& record:records)
    {
        const auto& deposit=*record.deposit;const auto b=record.band;
        const double power=deposit.outgoingPower,x=deposit.footprint.x+.5,y=deposit.footprint.y+.5;
        result.power[b]+=power;result.x[b]+=power*x;result.y[b]+=power*y;
        result.xx[b]+=power*x*x;result.yy[b]+=power*y*y;
        result.radiance[std::uint64_t(record.pixel)*8+b]+=deposit.outgoingPower/(3.14159265358979323846f*deposit.footprint.worldArea);
    }
    return result;
}
} // namespace glasscaustictests
