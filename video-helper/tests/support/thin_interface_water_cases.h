#pragma once

#include "smooth_glass_cases.h"
#include "../../../shared/GeometryFieldLowering.h"
#include "../../../shared/GeometryTimelineSceneBinding.h"
#include <functional>

namespace videohelper::tests::waterfixture
{
using namespace HarmonicMIDI::grid;
using D3 = sceneclosest::D3;
using smoothglassfixture::unit;
using smoothglassfixture::plus;
struct Hit final { bool found=false; std::size_t object=0; double distance=1e30; D3 position{},normal{},geometric{}; };
// Binary64 world triangle plane/edge oracle with barycentric vertex normals.
// It reads authored triangles, never the prepared BVH or native query source.
inline Hit hit(const Visual3DScene& scene,D3 origin,D3 ray)
{
    Hit result;
    for(std::size_t o=0;o<scene.objectCount;++o) {
        const auto& object=scene.objects[o];
        const D3 translation{object.transform.translation.x,object.transform.translation.y,object.transform.translation.z};
        for(std::size_t t=0;t<object.indexCount/3;++t) {
            std::array<D3,3> p,normals;
            for(int c=0;c<3;++c) {
                const auto& vertex=scene.vertices[object.firstVertex+scene.indices[object.firstIndex+t*3+c]];
                p[c]=plus(D3{vertex.position.x,vertex.position.y,vertex.position.z},translation);
                normals[c]=unit(D3{vertex.normal.x,vertex.normal.y,vertex.normal.z});
            }
            const auto a=sceneclosest::sub(p[1],p[0]),b=sceneclosest::sub(p[2],p[0]);
            const auto n=sceneclosest::cross(a,b);
            const auto denominator=sceneclosest::dot(n,ray);
            if(std::abs(denominator)<1e-14) continue;
            const double distance=sceneclosest::dot(n,sceneclosest::sub(p[0],origin))/denominator;
            if(distance<1e-9 || distance>=result.distance) continue;
            const auto position=plus(origin,ray,distance),delta=sceneclosest::sub(position,p[0]);
            const double aa=sceneclosest::dot(a,a),ab=sceneclosest::dot(a,b),bb=sceneclosest::dot(b,b);
            const double da=sceneclosest::dot(delta,a),db=sceneclosest::dot(delta,b),det=aa*bb-ab*ab;
            const double u=(bb*da-ab*db)/det,v=(aa*db-ab*da)/det;
            if(u<-1e-12 || v<-1e-12 || u+v>1+1e-12) continue;
            result={true,o,distance,position,{},unit(n)};
            result.normal=unit(plus(plus(plus(D3{},normals[0],1-u-v),normals[1],u),normals[2],v));
        }
    }
    return result;
}
inline D3 sky(const SceneLightRecord& light,D3 direction)
{
    if(light.sky.version==0) return {light.color.x*light.intensity,light.color.y*light.intensity,light.color.z*light.intensity};
    constexpr double radians=0.01745329251994329577;
    const auto& sky=light.sky;
    const double az=(sky.sunAzimuthDegrees+sky.orientationDegrees)*radians,el=sky.sunElevationDegrees*radians;
    const D3 sun{std::sin(az)*std::cos(el),std::sin(el),std::cos(az)*std::cos(el)};
    const double edge=std::cos(sky.sunRadiusDegrees*radians);
    const double cap=std::clamp((sceneclosest::dot(direction,sun)-edge)/(1-edge),0.0,1.0);
    const auto end=direction[1]>=0 ? sky.zenith : sky.ground;
    D3 result{};
    const double horizon[]={sky.horizon.x,sky.horizon.y,sky.horizon.z},limit[]={end.x,end.y,end.z};
    const double tint[]={light.color.x,light.color.y,light.color.z},sunColor[]={sky.sunColor.x,sky.sunColor.y,sky.sunColor.z};
    for(int c=0;c<3;++c) result[c]=(horizon[c]+std::abs(direction[1])*(limit[c]-horizon[c])
        +sunColor[c]*sky.sunIntensity*cap*cap*(3-2*cap))*tint[c]*light.intensity;
    return result;
}
struct Oracle final { D3 rgb{},entry{},reflection{},refraction{}; double fresnel=0; bool tir=false,complete=false; };
inline Oracle trace(const Visual3DScene& scene,D3 camera,D3 ray,double ior)
{
    Oracle result;
    const auto first=hit(scene,camera,ray);
    if(!first.found || first.object!=0) return result;
    auto normal=first.normal;
    const bool entering=sceneclosest::dot(normal,ray)<0;
    if(!entering) for(auto& c:normal)c=-c;
    const auto optics=smoothglassfixture::interface(ray,normal,entering?1:ior,entering?ior:1);
    result.entry=first.position;result.reflection=optics.reflection;result.refraction=optics.transmission;
    result.fresnel=optics.reflectance;result.tir=optics.tir;result.complete=true;
    const auto radiance=[&](D3 direction,double side) {
        const auto target=hit(scene,plus(first.position,normal,side*1e-8),direction);
        if(!target.found) return sky(scene.lights[0],direction);
        if(target.object==0) {result.complete=false;return D3{};}
        const auto& emission=scene.materials[target.object].emissive;
        return D3{emission.x,emission.y,emission.z};
    };
    if(optics.reflectance>0) result.rgb=plus(result.rgb,radiance(optics.reflection,1),optics.reflectance);
    if(!optics.tir && optics.reflectance<1) result.rgb=plus(result.rgb,radiance(optics.transmission,-1),1-optics.reflectance);
    return result;
}
struct RoughOracle final
{
    D3 rgb{},entry{};
    arbitgpu::smoothglass::RoughEnergyReceipt energy;
    std::array<D3,16> reflection{},refraction{};
    std::array<double,16> reflectionPower{},refractionPower{};
    std::size_t tir=0,closestHitQueries=0;
    arbitgpu::smoothglass::RoughTransportStatus status=arbitgpu::smoothglass::RoughTransportStatus::InvalidInput;
    bool complete=false;
};
using EnvironmentRadiance = std::function<D3(const SceneLightRecord&,D3)>;
// Binary64 one-interface VNDF quadrature with independent authored-triangle
// intersection and tangential Snell/Fresnel construction. This never invokes
// the closed-volume oracle, product shader or native receipt reader.
inline RoughOracle traceRough(const Visual3DScene& scene,D3 camera,D3 ray,double ior,
                              double roughness,const EnvironmentRadiance& environment={})
{
    using Status=arbitgpu::smoothglass::RoughTransportStatus;
    RoughOracle result;
    if(roughness==0) {
        const auto smooth=trace(scene,camera,ray,ior);
        result.rgb=smooth.rgb;result.entry=smooth.entry;result.complete=smooth.complete;
        result.status=smooth.complete?Status::Smooth:Status::IncompleteQuery;return result;
    }
    if(!std::isfinite(roughness)||roughness<0||roughness>1||!std::isfinite(ior)||ior<1||ior>3
        ||!scene.objectCount||scene.lightCount!=1) return result;
    const auto finite=[](D3 value) {return std::all_of(value.begin(),value.end(),[](double c){return std::isfinite(c);});};
    if(!finite(camera)||!finite(ray)||sceneclosest::dot(ray,ray)<=0) return result;
    ray=unit(ray);
    const auto first=hit(scene,camera,ray);++result.closestHitQueries;
    if(!first.found||first.object!=0) {result.status=Status::IdentityOrMedium;return result;}
    result.entry=first.position;
    const bool entering=sceneclosest::dot(ray,first.geometric)<0;
    const auto geometric=plus(D3{},first.geometric,entering?1:-1);
    const auto normal=plus(D3{},first.normal,entering?1:-1);
    if(!finite(normal)||sceneclosest::dot(normal,geometric)<=0||sceneclosest::dot(ray,normal)>=0)
    {result.status=Status::IdentityOrMedium;return result;}
    const double from=entering?1:ior,to=entering?ior:1,eta=from/to;
    // Match the admitted geometric origin clearance, not native query results.
    const double bias=std::max(1e-5,2e-6*std::max({std::abs(first.position[0]),std::abs(first.position[1]),std::abs(first.position[2])}));
    for(std::size_t sample=0;sample<16;++sample) {
        double inverse=0;
        for(int bit=0;bit<4;++bit) inverse+=double((sample>>bit)&1)*std::ldexp(1.0,-bit-1);
        const auto optics=smoothglassfixture::roughInterface(ray,normal,from,to,roughness,(double(sample)+.5)/16,inverse);
        result.reflection[sample]=optics.optics.reflection;
        result.refraction[sample]=optics.optics.transmission;
        result.tir+=optics.optics.tir;
        for(auto& masked:result.energy.masked) masked+=optics.masked/16;
        for(int branch=0;branch<2;++branch) {
            const bool transmission=branch==0;
            const double power=(transmission?optics.transmitted:optics.reflected)/16;
            const auto direction=transmission?optics.optics.transmission:optics.optics.reflection;
            if(!std::isfinite(power)||power<0||!finite(direction)) {result.status=Status::Nonfinite;return result;}
            if(power<.00001/16) {for(auto& tail:result.energy.tail)tail+=power;continue;}
            if(sceneclosest::dot(direction,geometric)*(transmission?-1:1)<=0) {
                for(auto& masked:result.energy.masked)masked+=power;
                continue;
            }
            const auto target=hit(scene,plus(first.position,geometric,transmission?-bias:bias),direction);
            ++result.closestHitQueries;
            if(target.found&&target.object==0) {result.status=Status::IdentityOrMedium;return result;}
            D3 radiance{};
            if(target.found) {
                const auto* material=visual3d_detail::findById(scene.materials,scene.materialCount,scene.objects[target.object].material);
                if(!material) {result.status=Status::IdentityOrMedium;return result;}
                // Native fixtures use black diffuse receivers with HDR emission.
                // A nonblack diffuse receiver needs a separate lighting oracle.
                if(material->baseColor.x!=0||material->baseColor.y!=0||material->baseColor.z!=0
                    ||scene.lights[0].kind!=SceneLightKind::Environment
                    ||scene.lights[0].sky.version||scene.lights[0].environmentImage.version)
                {result.status=Status::InvalidInput;return result;}
                radiance={material->emissive.x,material->emissive.y,material->emissive.z};
            } else {
                if(scene.lights[0].environmentImage.version&&!environment)
                {result.status=Status::InvalidInput;return result;}
                radiance=environment?environment(scene.lights[0],direction):sky(scene.lights[0],direction);
            }
            if(!finite(radiance)) {result.status=Status::Nonfinite;return result;}
            if(transmission) result.refractionPower[sample]=power;
            else result.reflectionPower[sample]=power;
            result.rgb=plus(result.rgb,radiance,power*(transmission?eta*eta:1));
            for(auto& terminal:result.energy.terminal)terminal+=power;
        }
    }
    for(std::size_t c=0;c<3;++c) {
        const double terminal=result.energy.terminal[c],masked=result.energy.masked[c],tail=result.energy.tail[c];
        if(!std::isfinite(terminal+masked+tail)||!finite(result.rgb)) {result.status=Status::Nonfinite;return result;}
        if(terminal<0||terminal>1||masked<0||masked>1||tail<0||tail>1||std::abs(terminal+masked+tail-1)>.00002)
        {result.status=Status::NonconservingEnergy;return result;}
        if(masked+tail>.0001) {result.status=Status::UnresolvedEnergy;return result;}
    }
    result.status=Status::Complete;result.complete=true;return result;
}
inline Visual3DScene scene(float angle=20)
{
    auto scene=smoothglassfixture::scene(0,1,angle);
    scene.vertexCount=4;scene.indexCount=6;
    scene.objects[0].vertexCount=4;scene.objects[0].indexCount=6;
    const SceneVec3 positions[]={{-2,-2,0},{2,-2,0},{2,2,0},{-2,2,0}};
    for(int c=0;c<4;++c) scene.vertices[c].position=positions[c];
    const std::uint32_t indices[]={0,1,2,0,2,3};std::copy_n(indices,6,scene.indices.begin());
    scene.cameras[0].verticalFovRadians=.004f;
    return scene;
}
inline void receiver(Visual3DScene& scene,double z,double x,D3 emission,double halfWidth=.06)
{
    smoothglassfixture::receiver(scene,-8,emission);
    const auto& object=scene.objects[scene.objectCount-1];
    const SceneVec3 positions[]={{float(x-halfWidth),-2,float(z)},{float(x+halfWidth),-2,float(z)},
        {float(x+halfWidth),2,float(z)},{float(x-halfWidth),2,float(z)}};
    for(int c=0;c<4;++c) {
        auto& vertex=scene.vertices[object.firstVertex+c];vertex.position=positions[c];
        vertex.normal={0,0,z>0?-1.0f:1.0f};
    }
    if(z>0) for(std::size_t i=object.firstIndex;i<object.firstIndex+object.indexCount;i+=3)
        std::swap(scene.indices[i],scene.indices[i+1]);
}
inline std::shared_ptr<const videorender::fixture3d::AdmittedSurfaceMaterialBinding> binding(
    const std::shared_ptr<const Visual3DScene>& scene,videohelper::materialprogram::BackendTarget target,std::string& error,
    float roughness=0,float ior=1.333f)
{
    surfacematerialbinding::ImportedSceneMaterialRequest request;
    request.scene=scene->id;request.sceneRevision=request.structuralRevision=request.evaluationRevision=request.programRevision=1;
    request.sceneSnapshot=scene;request.program=smoothglassfixture::authoredProgram(scene->objects[0].material.value,ior,0);
    for(auto& operation:request.program.operations)
        if(operation.id==request.program.outputs[static_cast<std::size_t>(surfacematerial::OutputSemantic::Roughness)])
            operation.literal[0]=roughness;
    request.program.transport=surfacematerial::TransportMode::ThinInterface;
    request.binding.targetKind=surfacematerialbinding::BindingTargetKind::ObjectOverride;
    request.binding.object=scene->objects[0].id;request.binding.surfaceMaterialRevision=1;
    const auto admitted=surfacematerial::admit(request.program,error);if(!admitted)return {};
    request.binding.surfaceMaterialDigest=admitted->structuralDigest();
    return videorender::fixture3d::admitSurfaceMaterialBinding(scene,request,target,error);
}
inline bool render(arbitgpu::NativeFixtureSceneBackend& backend,Visual3DScene authored,double seconds,
    std::vector<std::uint8_t>& bytes,D3& actual,std::string& error)
{
    const auto& transform=authored.cameras[0].transform;
    const D3 camera{transform.translation.x,transform.translation.y,transform.translation.z};
    const double angle=2*std::atan2(transform.rotation.y,transform.rotation.w);
    const D3 ray{-std::sin(angle),0,-std::cos(angle)};
    const auto expected=trace(authored,camera,ray,double(1.333f));
    if(!expected.complete) {error="Independent water Snell/Fresnel/triangle oracle failed";return false;}
    const auto snapshot=std::make_shared<const Visual3DScene>(std::move(authored));
    const auto target=backend.info().backend=="metal"?videohelper::materialprogram::BackendTarget::Metal:videohelper::materialprogram::BackendTarget::OpenGl;
    const auto material=binding(snapshot,target,error);if(!material)return false;
    videorender::fixture3d::FixtureSceneRenderer renderer(backend);
    arbitgpu::NativeFixtureSceneRuntimeInputs inputs;inputs.linearColor=true;inputs.timeSeconds=float(seconds);
    videorender::fixture3d::RenderedFrame preview,exported;
    if(!renderer.renderPreview(snapshot,material,{},inputs,{9,9},"native-gpu",preview,error)
        ||!renderer.renderExport(snapshot,material,{},inputs,{9,9},"native-gpu",exported,error)
        ||!preview.stats.smoothGlassTransportApplied || preview.stats.smoothGlassQueryBytes==0)
    {error="Ordinary water preview/export transport failed: "+error;return false;}
    arbitgpu::NativeRawPassPixels color,exportColor;
    if(!preview.nativeFrame->readRawPass(renderpassoutput::Output::Color,color,error)
        ||!exported.nativeFrame->readRawPass(renderpassoutput::Output::Color,exportColor,error)
        ||color.format!=arbitgpu::NativeTexturePixelFormat::Rgba16Float ||color.bytes!=exportColor.bytes)
    {error="Water equal-time preview/export lost owned Color identity: "+error;return false;}
    bytes=color.bytes;
    for(int c=0;c<3;++c) {
        std::uint16_t half=0;std::memcpy(&half,bytes.data()+(4*9+4)*8+c*2,2);actual[c]=hdrimage::halfToFloat(half);
        if(!std::isfinite(actual[c]) || std::abs(actual[c]-expected.rgb[c])>.002)
        {error="Water native channel "+std::to_string(c)+" expected "+std::to_string(expected.rgb[c])+", got "+std::to_string(actual[c]);return false;}
    }
    std::cout<<"thin-interface-water backend="<<backend.info().backend<<" time="<<seconds<<" rgb="<<actual[0]<<','<<actual[1]<<','<<actual[2]<<'\n';
    return true;
}
inline bool roughPhysicsCases(std::string& error)
{
    using Status=arbitgpu::smoothglass::RoughTransportStatus;
    const auto reject=[&](const char* message){error=message;return false;};
    const auto rayFor=[](const Visual3DScene& value) {
        const auto& c=value.cameras[0].transform.translation;
        return unit(D3{-c.x,-c.y,-c.z});
    };
    const auto cameraFor=[](const Visual3DScene& value) {
        const auto& c=value.cameras[0].transform.translation;return D3{c.x,c.y,c.z};
    };
    auto authored=scene();
    const auto camera=cameraFor(authored),ray=rayFor(authored);
    const auto smooth=trace(authored,camera,ray,1.333f);
    const auto zero=traceRough(authored,camera,ray,1.333f,0);
    if(!smooth.complete||!zero.complete||zero.status!=Status::Smooth||zero.rgb!=smooth.rgb)
        return reject("Zero rough water changed the released one-interface oracle");
    const auto narrow=traceRough(authored,camera,ray,1.333f,.08);
    const auto broad=traceRough(authored,camera,ray,1.333f,.16);
    const auto replay=traceRough(authored,camera,ray,1.333f,.08);
    if(!narrow.complete||!broad.complete||narrow.status!=Status::Complete
        ||narrow.rgb!=replay.rgb||narrow.energy.terminal!=replay.energy.terminal
        ||narrow.refraction==broad.refraction||narrow.closestHitQueries>33)
        return reject("One-interface rough quadrature, energy or deterministic branch footprint failed");
    for(const auto* result:{&narrow,&broad}) for(std::size_t c=0;c<3;++c)
        if(std::abs(result->energy.terminal[c]+result->energy.masked[c]+result->energy.tail[c]-1)>.00002
            ||result->energy.masked[c]+result->energy.tail[c]>.0001)
            return reject("Independent rough water power closure or unresolved bound failed");
    auto below=scene();below.cameras[0].transform.translation.z*=-1;
    const auto internal=traceRough(below,cameraFor(below),rayFor(below),1.333f,.08);
    if(!internal.complete||internal.tir||internal.rgb==narrow.rgb)
        return reject("Rough water failed the opposite incident medium and radiance eta factor");
    auto tir=scene(60);tir.cameras[0].transform.translation.z*=-1;
    const auto reflected=traceRough(tir,cameraFor(tir),rayFor(tir),1.333f,.08);
    if(!reflected.complete||reflected.tir!=16
        ||std::any_of(reflected.refractionPower.begin(),reflected.refractionPower.end(),[](double p){return p!=0;}))
        return reject("One-interface rough water failed internal TIR");
    const auto matched=traceRough(authored,camera,ray,1,.9);
    if(!matched.complete||matched.energy.terminal[0]!=1||matched.energy.masked[0]!=0||matched.energy.tail[0]!=0)
        return reject("Index-matched rough water created an optical or masking deficit");
    auto grazing=scene(80);
    const auto masked=traceRough(grazing,cameraFor(grazing),rayFor(grazing),1.333f,.9);
    if(masked.complete||masked.status!=Status::UnresolvedEnergy)
        return reject("Strongly masked single-interface water concealed unresolved power");
    auto blocked=authored;
    const double z=-3,x=narrow.entry[0]+(z-narrow.entry[2])/narrow.refraction[0][2]*narrow.refraction[0][0];
    receiver(blocked,z,x,{8,4,2},.04);
    const auto receiverResult=traceRough(blocked,camera,ray,1.333f,.08);
    if(!receiverResult.complete||receiverResult.rgb==narrow.rgb)
        return reject("Rough water ignored an actual off-camera HDR receiver triangle");
    const double blockX=narrow.entry[0]+(-2-narrow.entry[2])/narrow.refraction[0][2]*narrow.refraction[0][0];
    receiver(blocked,-2,blockX,{0,0,0},.2);
    const auto blockerResult=traceRough(blocked,camera,ray,1.333f,.08);
    if(!blockerResult.complete||blockerResult.rgb==receiverResult.rgb)
        return reject("Rough water ignored nearer blocker geometry");
    auto waved=authored;
    for(std::size_t v=0;v<4;++v)waved.vertices[v].normal={.08f,.12f,1};
    const auto waveResult=traceRough(waved,camera,ray,1.333f,.08);
    if(!waveResult.complete||waveResult.refraction==narrow.refraction)
        return reject("Evaluated wave macro normals did not change rough-water scattering");
    for(auto& vertex:waved.vertices)vertex.normal={0,0,-1};
    const auto wrongNormal=traceRough(waved,camera,ray,1.333f,.08);
    if(wrongNormal.complete||wrongNormal.status!=Status::IdentityOrMedium)
        return reject("Inconsistent geometric and evaluated-normal hemispheres admitted");
    auto environment=authored;environment.lights[0].intensity=4;
    const auto brighter=traceRough(environment,camera,ray,1.333f,.08);
    if(!brighter.complete||std::abs(brighter.rgb[0]-narrow.rgb[0]*4)>1e-12)
        return reject("Rough water ignored authored HDR environment radiance");
    environment.lights[0].environmentImage.version=1;
    if(traceRough(environment,camera,ray,1.333f,.08).complete)
        return reject("Image environment without an independent source oracle admitted");
    const auto hdr=traceRough(environment,camera,ray,1.333f,.08,
        [](const SceneLightRecord&,D3 direction){return D3{4+direction[0],2+direction[1],1+direction[2]};});
    if(!hdr.complete||hdr.rgb==narrow.rgb)return reject("Independent HDR image-source callback did not reach water");
    auto recursive=scene(0);
    for(int c=0;c<4;++c){recursive.vertices[4+c]=recursive.vertices[c];recursive.vertices[4+c].position.z=5;recursive.vertices[4+c].normal={0,0,-1};}
    const std::uint32_t indices[]={4,6,5,4,7,6};std::copy_n(indices,6,recursive.indices.begin()+6);
    recursive.vertexCount=recursive.objects[0].vertexCount=8;recursive.indexCount=recursive.objects[0].indexCount=12;
    const auto rehit=traceRough(recursive,cameraFor(recursive),rayFor(recursive),1.333f,.08);
    if(rehit.complete||rehit.status!=Status::IdentityOrMedium)return reject("Rough water re-hit completed as a terminal receiver");
    for(double roughness:{-1.,1.1,std::numeric_limits<double>::quiet_NaN()})
        if(traceRough(authored,camera,ray,1.333f,roughness).complete)return reject("Invalid rough water input admitted");
    const auto gl=arbitgpu::smoothglass::transportShader(false,true),metal=arbitgpu::smoothglass::transportShader(true,true);
    if(gl.find("vec3 roughThinInterfaceTransport(")==std::string::npos
        ||metal.find("float3 roughThinInterfaceTransport(")==std::string::npos
        ||gl.find("vec3 smoothColour=thinInterfaceTransport(camera,entry,evaluatedNormal,smoothComplete,smoothStatus GX_PASS);")==std::string::npos
        ||gl.find("vec3 origin=first.position+geometric*")==std::string::npos
        ||gl.find("target.status==1&&target.objectId==water")==std::string::npos)
        return reject("Shared GL/Metal water entry or exact smooth delegation is missing");
    error.clear();return true;
}
inline bool run(arbitgpu::NativeFixtureSceneBackend& backend,std::string& error)
{
    if(!roughPhysicsCases(error))return false;
    auto authored=scene();
    const auto& camera=authored.cameras[0].transform.translation;
    const D3 origin{camera.x,camera.y,camera.z},ray=unit(plus(D3{},origin,-1));
    const auto optics=trace(authored,origin,ray,1.333f);
    const auto landing=[&](D3 direction,double z){return optics.entry[0]+(z-optics.entry[2])/direction[2]*direction[0];};
    // Farther-first objects test nearest hit. Reflection panels are behind the
    // camera; refraction panels avoid the camera ray's projected footprint.
    receiver(authored,7,landing(optics.reflection,7),{.1,.1,.1});
    receiver(authored,5,landing(optics.reflection,5),{.8,.4,.1});
    receiver(authored,-5,landing(optics.refraction,-5),{.1,.1,.1});
    receiver(authored,-3,landing(optics.refraction,-3),{.2,.5,.8});
    if(std::abs(landing(optics.refraction,-3)-landing(ray,-3))<.1)
    {error="Water refraction target overlaps primary camera footprint";return false;}
    std::vector<std::uint8_t> baseline,bytes;D3 actual{},before{};
    if(!render(backend,authored,0,baseline,before,error))return false;
    auto changed=authored;changed.materials[2].emissive={.2f,.4f,.9f};
    if(!render(backend,changed,0,bytes,actual,error)||bytes==baseline || std::abs(actual[0]-before[0])<.005)
    {error="Off-camera reflected receiver did not causally change water: "+error;return false;}
    changed=authored;changed.materials[4].emissive={.9f,.2f,.1f};
    if(!render(backend,changed,0,bytes,actual,error)||bytes==baseline || std::abs(actual[0]-before[0])<.1)
    {error="Off-camera refracted receiver did not causally change water: "+error;return false;}
    changed=authored;changed.objects[2].transform.translation.x=10;
    if(!render(backend,changed,0,bytes,actual,error)||bytes==baseline)
    {error="Moving nearest reflected receiver did not reveal farther geometry: "+error;return false;}
    changed=authored;changed.objects[4].transform.translation.x=10;
    if(!render(backend,changed,0,bytes,actual,error)||bytes==baseline)
    {error="Moving nearest refracted receiver did not reveal farther geometry: "+error;return false;}
    auto internal=scene(0);
    constexpr double internalAngle=1.04719755119659774615;
    internal.cameras[0].transform.translation={float(3*std::sin(internalAngle)),0,float(-3*std::cos(internalAngle))};
    internal.cameras[0].transform.rotation={0,float(std::sin((3.14159265358979323846-internalAngle)/2)),0,
        float(std::cos((3.14159265358979323846-internalAngle)/2))};
    const auto& internalCamera=internal.cameras[0].transform.translation;
    const auto internalOptics=trace(internal,{internalCamera.x,0,internalCamera.z},
        {-std::sin(internalAngle),0,std::cos(internalAngle)},1.333f);
    if(!internalOptics.complete || !internalOptics.tir || internalOptics.fresnel!=1)
    {error="Independent water interface did not exercise total internal reflection";return false;}
    receiver(internal,-5,internalOptics.entry[0]+(-5-internalOptics.entry[2])/internalOptics.reflection[2]*internalOptics.reflection[0],{.7,.3,.2});
    if(!render(backend,internal,0,bytes,actual,error))return false;
    std::vector<std::uint8_t> skyBefore;
    for(float azimuth:{0.0f,180.0f,90.0f}) {
        auto skyScene=scene(0);auto& light=skyScene.lights[0];light.color={1,1,1};light.sky.version=1;
        light.sky.zenith={.1f,.2f,.4f};light.sky.horizon={.5f,.6f,.7f};light.sky.ground={.025f,.03f,.04f};
        light.sky.sunColor={1,.9f,.7f};light.sky.sunIntensity=2;light.sky.sunRadiusDegrees=15;
        light.sky.sunElevationDegrees=0;light.sky.sunAzimuthDegrees=azimuth;
        if(!render(backend,skyScene,0,bytes,actual,error))return false;
        if(azimuth==0)skyBefore=bytes;
        else if(bytes==skyBefore) {error="Water missed rays ignored independently authored sky sun direction";return false;}
    }
    // Reuse the existing lowered wave and timeline mesh replacement authority.
    using namespace videowire::geometry;
    PortContract contract;contract.carrier=CarrierKind::geometry3D;contract.maxVertices=4096;contract.maxIndices=12288;
    auto grid=lowerGrid(601,601,1,9,9,.35f,contract,error);
    auto wave=grid?lowerDirectionalSurfaceWave(*grid,606,10000,.15,2,30,.5,.3,4,contract,error):std::nullopt;
    auto admitted=wave?admitValue(*wave,contract,error):std::nullopt;if(!admitted)return false;
    auto waveScene=scene(0);waveScene.vertexCount=waveScene.indexCount=waveScene.objectCount=waveScene.materialCount=0;
    SceneObjectRecord object;object.id={101};SceneMaterialRecord material;material.id={6};material.baseColor={1,1,1};material.roughness=0;
    if(!appendGeometryToScene(*admitted,object,material,waveScene,error))return false;
    const TimelineSceneBinding timeline{{101},encodeLoweredPlanText(lowerRuntimePlan(contract,*admitted))};
    std::vector<std::uint8_t> atZero,atHalf;
    // Gradient sky makes moving wave normals causal even when both rays miss.
    waveScene.lights[0].color={1,1,1};waveScene.lights[0].sky.version=1;
    waveScene.lights[0].sky.sunIntensity=0;
    for(double seconds:{0.0,.5,4.5,.25,0.0,.5,-3.5}) {
        auto evaluated=waveScene;
        if(!replaceTimelineSceneMesh(timeline,*admitted,seconds,evaluated,error)
            ||!render(backend,std::move(evaluated),seconds,bytes,actual,error))return false;
        if(seconds==0 && atZero.empty())atZero=bytes;
        else if(seconds==0 && atZero!=bytes) {error="Water wave reset lost exact replay";return false;}
        if(seconds==.5 && atHalf.empty())atHalf=bytes;
        else if((seconds==.5 || seconds==4.5 || seconds==-3.5) && atHalf!=bytes)
        {error="Water wave seek/loop lost equal-time optical replay";return false;}
    }
    if(atHalf==atZero) {error="Timeline wave normals did not change water radiance";return false;}
    auto snapshot=std::make_shared<const Visual3DScene>(authored);
    const auto target=backend.info().backend=="metal"?videohelper::materialprogram::BackendTarget::Metal:videohelper::materialprogram::BackendTarget::OpenGl;
    const auto water=binding(snapshot,target,error);if(!water)return false;
    const auto refuse=[&](Visual3DScene value,const std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram>& native,const char* message) {
        if(backend.prepare(std::make_shared<const Visual3DScene>(std::move(value)),native).prepared) {error=message;return false;}return true;
    };
    auto bad=authored;bad.materials[0].normalTexture={1};if(!refuse(bad,water->nativeProgram(),"Textured water admitted"))return false;
    bad=authored;bad.materials[0].alphaMode=SceneAlphaMode::Mask;if(!refuse(bad,water->nativeProgram(),"Masked water admitted"))return false;
    bad=authored;bad.vertices[0].normal={};if(!refuse(bad,water->nativeProgram(),"Zero evaluated water normal admitted"))return false;
    auto rough=std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*water->nativeProgram());rough->parameters.emissionRoughness[3]=.1f;
    arbitgpu::smoothglass::Prepared unsupportedRough;
    if(arbitgpu::smoothglass::prepare(authored,rough.get(),{},unsupportedRough,error))
    {error="Rough water admitted without a matching receipt consumer";return false;}
    auto dispersion=std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*water->nativeProgram());dispersion->parameters.transmissionIorClearcoat[3]=.001f;
    if(!refuse(authored,dispersion,"Dispersive thin water admitted"))return false;
    auto deformed=std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*water->nativeProgram());
    const auto vertex=videohelper::vertexmodifier::admitVertexModifierIr(videohelper::test::audioNormalDisplacement(),{},error);
    if(!vertex)return false;
    const auto compiled=videohelper::materialprogram::compileMaterialProgram(nullptr,&*vertex,target,error);
    if(!compiled)return false;
    deformed->vertexProgram=std::make_shared<const videohelper::materialprogram::MaterialProgramRecord>(*compiled);
    if(!refuse(authored,deformed,"Unevaluated vertex-deformed water admitted"))return false;
    arbitgpu::smoothglass::Prepared prepared;
    if(!arbitgpu::smoothglass::prepare(authored,water->nativeProgram().get(),{},prepared,error))return false;
    if(arbitgpu::smoothglass::admitFrame(prepared,16384,16384,false,error)
        ||arbitgpu::smoothglass::admitFrame(prepared,9,9,true,error))
    {error="Unbounded or instanced water frame admitted";return false;}
    std::vector<videowire::geometry::InstanceAppearance> appearances(1);
    if(arbitgpu::smoothglass::prepare(authored,water->nativeProgram().get(),appearances,prepared,error))
    {error="Instance appearance water admitted";return false;}
    auto recursive=scene(0);
    for(int c=0;c<4;++c) {recursive.vertices[4+c]=recursive.vertices[c];recursive.vertices[4+c].position.z=5;recursive.vertices[4+c].normal={0,0,-1};}
    const std::uint32_t recursiveIndices[]={4,6,5,4,7,6};std::copy_n(recursiveIndices,6,recursive.indices.begin()+6);
    recursive.vertexCount=recursive.objects[0].vertexCount=8;recursive.indexCount=recursive.objects[0].indexCount=12;
    const auto recursiveSnapshot=std::make_shared<const Visual3DScene>(recursive);
    const auto recursiveMaterial=binding(recursiveSnapshot,target,error);if(!recursiveMaterial)return false;
    const auto recursivePrepared=backend.prepare(recursiveSnapshot,recursiveMaterial->nativeProgram());
    if(!recursivePrepared.prepared) {error="Recursive water did not reach path admission: "+recursivePrepared.error;return false;}
    arbitgpu::NativeFixtureSceneRuntimeInputs linear;linear.linearColor=true;
    const auto recursiveFrame=backend.render(recursiveSnapshot,recursivePrepared.resources,9,9,linear);
    if(recursiveFrame.rendered || recursiveFrame.frame || recursiveFrame.error.find("transport")==std::string::npos)
    {error="Recursive water ray was published as complete";return false;}
    auto legacy=std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*water->nativeProgram());
    legacy->parameters.identifiers[1]=0;
    if(!arbitgpu::nativeSurfaceRequiresBlend(legacy.get()) || arbitgpu::nativeSurfaceRequiresBlend(water->nativeProgram().get()))
    {error="ThinSurface legacy coverage or water opaque transport changed";return false;}
    if(!arbitgpu::smoothglass::prepare(authored,legacy.get(),{},prepared,error)||prepared.enabled)
    {error="Default ThinSurface unexpectedly opted into transport";return false;}
    error.clear();return true;
}
} // namespace videohelper::tests::waterfixture
