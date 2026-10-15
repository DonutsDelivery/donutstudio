#pragma once

#include "environment_image_cases.h"
#include "thin_interface_water_cases.h"
#include "../../src/gpu_backend/fixture_environment_image.h"

namespace videohelper::tests::environmentsecondaryfixture
{
using namespace HarmonicMIDI::grid;
using D3 = sceneclosest::D3;
using smoothglassfixture::unit;
using smoothglassfixture::plus;
constexpr double pi = 3.14159265358979323846;

// Authored RGBE polynomial texels and binary64 spherical/bilinear sampling.
// The reference never reads decoded pixels, prepared textures or shader source.
inline double texel(int x,int y,int channel,int variant)
{
    const int values[]={16+3*x*x+4*y+32*variant,16+2*x+8*y+16*variant,64+3*x+y+8*variant};
    return values[channel]/16.0;
}
inline std::vector<std::uint8_t> imageBytes(int variant)
{
    auto bytes=environmentImageBytes();
    // Four RLE rows, each a four-byte header and four literal channels.
    const auto header=bytes.size()-4*40;
    for(int y=0;y<4;++y) for(int c=0;c<3;++c) for(int x=0;x<8;++x)
        bytes[header+y*40+4+c*9+1+x]=static_cast<std::uint8_t>(std::lround(texel(x,y,c,variant)*16));
    return bytes;
}
inline D3 radiance(const SceneLightRecord& light,D3 direction,int variant)
{
    direction=unit(direction);
    const double longitude=direction[0]*direction[0]+direction[2]*direction[2]>1e-12
        ? std::atan2(direction[0],direction[2]) : 0;
    double u=(longitude-light.environmentImage.orientationDegrees*pi/180)/(2*pi)+.5;
    u-=std::floor(u);
    const double x=u*8-.5,y=std::acos(std::clamp(direction[1],-1.0,1.0))/pi*4-.5;
    const int ix=static_cast<int>(std::floor(x)),iy=static_cast<int>(std::floor(y));
    const double fx=x-ix,fy=y-iy;
    const int x0=(ix%8+8)%8,x1=(x0+1)%8,y0=std::clamp(iy,0,3),y1=std::clamp(iy+1,0,3);
    const double tint[]={light.color.x,light.color.y,light.color.z};
    const double scale=std::exp2(double(light.environmentImage.exposureStops))*light.intensity;
    D3 rgb{};
    for(int c=0;c<3;++c) rgb[c]=((1-fy)*((1-fx)*texel(x0,y0,c,variant)+fx*texel(x1,y0,c,variant))
        +fy*((1-fx)*texel(x0,y1,c,variant)+fx*texel(x1,y1,c,variant)))*tint[c]*scale;
    return rgb;
}
inline double radical(int index)
{
    double value=0;
    for(int bit=0;bit<6;++bit) {value+=(index&1)*std::ldexp(1.0,-bit-1);index>>=1;}
    return value;
}
// Independent binary64 evaluation of the admitted base GGX quadrature, with
// diffuse receiver roughness one and F0 .04. Receiver radiance is view dependent.
inline D3 receiverRadiance(const Visual3DScene& scene,std::size_t object,D3 normal,D3 view,int variant)
{
    const auto& material=scene.materials[object];
    const D3 base{material.baseColor.x,material.baseColor.y,material.baseColor.z};
    D3 rgb{material.emissive.x,material.emissive.y,material.emissive.z};
    normal=unit(normal);view=unit(view);
    const double nv=std::clamp(sceneclosest::dot(normal,view),0.0,1.0);
    if(nv<=0) return rgb;
    const auto tangent=unit(sceneclosest::cross(normal,std::abs(normal[2])<.999?D3{0,0,1}:D3{1,0,0}));
    const auto bitangent=sceneclosest::cross(normal,tangent);
    const auto stretched=unit(D3{sceneclosest::dot(view,tangent),sceneclosest::dot(view,bitangent),nv});
    const auto t1=stretched[2]<.99999?unit(D3{-stretched[1],stretched[0],0}):D3{1,0,0};
    const auto t2=sceneclosest::cross(stretched,t1);
    for(int i=0;i<64;++i) {
        const double radius=std::sqrt((i+.5)/64),phi=2*pi*radical(i);
        const double dx=radius*std::cos(phi),blend=(1+stretched[2])/2;
        const double dy=(1-blend)*std::sqrt(std::max(0.0,1-dx*dx))+blend*radius*std::sin(phi);
        auto projected=plus(plus(plus(D3{},t1,dx),t2,dy),stretched,std::sqrt(std::max(0.0,1-dx*dx-dy*dy)));
        projected[2]=std::max(.000001,projected[2]);projected=unit(projected);
        const auto half=plus(plus(plus(D3{},tangent,projected[0]),bitangent,projected[1]),normal,projected[2]);
        const double vh=sceneclosest::dot(view,half);
        const auto direction=plus(plus(D3{},half,2*vh),view,-1);
        const double nl=std::clamp(sceneclosest::dot(normal,direction),0.0,1.0);
        if(vh>0&&nl>0) {
            const double fresnel=.04+.96*std::pow(1-vh,5),smith=2*nl/(nl+1);
            rgb=plus(rgb,radiance(scene.lights[0],direction,variant),fresnel*smith/64);
        }
    }
    for(int i=0;i<32;++i) {
        const double u=(i+.5)/32,phi=2*pi*radical(i);
        const auto direction=plus(plus(plus(D3{},tangent,std::cos(phi)*std::sqrt(u)),
            bitangent,std::sin(phi)*std::sqrt(u)),normal,std::sqrt(1-u));
        const double vh=std::clamp(sceneclosest::dot(view,unit(plus(direction,view))),0.0,1.0);
        const double fresnel=.04+.96*std::pow(1-vh,5);
        const auto incident=radiance(scene.lights[0],direction,variant);
        for(int c=0;c<3;++c) rgb[c]+=incident[c]*(1-fresnel)*base[c]/32;
    }
    return rgb;
}
struct Hit {bool found=false,entering=false;std::size_t object=0;double distance=1e30;D3 position{},normal{};};
inline Hit hit(const Visual3DScene& scene,D3 origin,D3 direction,bool water)
{
    Hit result;
    for(std::size_t o=0;o<scene.objectCount;++o) {
        const auto& object=scene.objects[o];
        const D3 offset{object.transform.translation.x,object.transform.translation.y,object.transform.translation.z};
        for(std::size_t t=0;t<object.indexCount/3;++t) {
            std::array<D3,3> p,n;
            for(int c=0;c<3;++c) {
                const auto& vertex=scene.vertices[object.firstVertex+scene.indices[object.firstIndex+t*3+c]];
                p[c]=plus(D3{vertex.position.x,vertex.position.y,vertex.position.z},offset);
                n[c]={vertex.normal.x,vertex.normal.y,vertex.normal.z};
            }
            const auto a=sceneclosest::sub(p[1],p[0]),b=sceneclosest::sub(p[2],p[0]);
            auto normal=sceneclosest::cross(a,b);
            const double denominator=sceneclosest::dot(normal,direction);
            if(std::abs(denominator)<1e-14) continue;
            const double distance=sceneclosest::dot(normal,sceneclosest::sub(p[0],origin))/denominator;
            if(distance<1e-9||distance>=result.distance) continue;
            const auto position=plus(origin,direction,distance),delta=sceneclosest::sub(position,p[0]);
            const double aa=sceneclosest::dot(a,a),ab=sceneclosest::dot(a,b),bb=sceneclosest::dot(b,b);
            const double da=sceneclosest::dot(delta,a),db=sceneclosest::dot(delta,b),det=aa*bb-ab*ab;
            const double u=(bb*da-ab*db)/det,v=(aa*db-ab*da)/det;
            if(u<-1e-12||v<-1e-12||u+v>1+1e-12) continue;
            if(water&&o==0) normal=plus(plus(plus(D3{},n[0],1-u-v),n[1],u),n[2],v);
            result={true,denominator<0,o,distance,position,unit(normal)};
        }
    }
    return result;
}
struct Oracle {D3 rgb{};std::size_t misses=0,receivers=0,blockers=0,reflections=0;bool complete=true;};
inline Oracle trace(const Visual3DScene& scene,bool water,int variant,double ior,double dispersion)
{
    Oracle result;
    const auto& camera=scene.cameras[0].transform;
    const double angle=2*std::atan2(double(camera.rotation.y),double(camera.rotation.w));
    const D3 origin{camera.translation.x,camera.translation.y,camera.translation.z},ray{-std::sin(angle),0,-std::cos(angle)};
    const auto first=hit(scene,origin,ray,water);
    if(!first.found||first.object!=0) {result.complete=false;return result;}
    for(int channel=0;channel<3;++channel) {
        const double wavelength=channel==0?.6563:channel==1?.5461:.4358;
        const double index=ior+dispersion*(1/(wavelength*wavelength)-1/(.58756*.58756));
        struct Path {D3 origin,direction;double weight;bool inside;std::size_t depth;};
        std::deque<Path> paths{{origin,ray,1,!first.entering,0}};
        std::size_t visits=0;
        while(!paths.empty()) {
            const auto path=paths.front();paths.pop_front();
            if(++visits>512) {result.complete=false;return result;}
            const auto target=hit(scene,path.origin,path.direction,water);
            if(!target.found) {
                if(!water&&path.inside) {result.complete=false;return result;}
                ++result.misses;result.rgb[channel]+=path.weight*radiance(scene.lights[0],path.direction,variant)[channel];continue;
            }
            if(target.object!=0) {
                if(!water&&path.inside) {result.complete=false;return result;}
                ++result.receivers;if(target.object==2) ++result.blockers;
                const auto view=plus(D3{},path.direction,-1);
                result.rgb[channel]+=path.weight*receiverRadiance(scene,target.object,target.normal,view,variant)[channel];continue;
            }
            if((water&&path.depth>0)||(!water&&(path.inside==target.entering||path.depth>=64)))
            {result.complete=false;return result;}
            auto normal=target.normal;
            const bool entering=water?sceneclosest::dot(path.direction,normal)<0:target.entering;
            if(!entering) for(auto& c:normal)c=-c;
            const auto optics=smoothglassfixture::interface(path.direction,normal,entering?1:index,entering?index:1);
            if(!optics.tir&&path.weight*(1-optics.reflectance)>1e-10)
                paths.push_back({plus(target.position,normal,-1e-8),optics.transmission,path.weight*(1-optics.reflectance),!path.inside,path.depth+1});
            if(path.weight*optics.reflectance>1e-10) {
                ++result.reflections;
                paths.push_back({plus(target.position,normal,1e-8),optics.reflection,path.weight*optics.reflectance,path.inside,path.depth+1});
            }
        }
    }
    return result;
}
inline bool admissionCases(const std::shared_ptr<const Visual3DScene>& scene,
    const std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram>& material,std::string& error)
{
    using namespace arbitgpu;
    smoothglass::Prepared transport;
    if(!smoothglass::prepare(scene,material,{},transport,error)) return false;
    arbitgpu::environmentimage::Prepared environment;
    const auto reject=[&](bool accepted,const char* failure) {if(accepted)error=failure;return !accepted;};
    if(!reject(arbitgpu::environmentimage::prepare(*scene,material.get(),environment,error),"Default HDR transmission admission changed")
        ||!reject(arbitgpu::environmentimage::prepare(*scene,material.get(),environment,error,arbitgpu::environmentimage::SecondaryRayConsumer::SmoothTransport),
            "HDR secondary transport admitted without owned geometry")) return false;
    auto replacement=std::make_shared<const Visual3DScene>(*scene);
    if(!reject(arbitgpu::environmentimage::prepare(*replacement,material.get(),environment,error,
        arbitgpu::environmentimage::SecondaryRayConsumer::SmoothTransport,&transport),"HDR secondary admitted another scene owner")) return false;
    auto replacementMaterial=std::make_shared<const NativeFixtureSurfaceMaterialProgram>(*material);
    if(!reject(arbitgpu::environmentimage::prepare(*scene,replacementMaterial.get(),environment,error,
        arbitgpu::environmentimage::SecondaryRayConsumer::SmoothTransport,&transport),"HDR secondary admitted another material owner")) return false;
    if(!arbitgpu::environmentimage::prepare(*scene,material.get(),environment,error,
        arbitgpu::environmentimage::SecondaryRayConsumer::SmoothTransport,&transport)||!environment.resident||!environment.gpuBudget)
    {error="Exact owned HDR secondary admission failed: "+error;return false;}
    // Failed admission publishes no resident lease into a fresh output.
    arbitgpu::environmentimage::Prepared refused;
    if(!reject(arbitgpu::environmentimage::prepare(*replacement,material.get(),refused,error,
        arbitgpu::environmentimage::SecondaryRayConsumer::SmoothTransport,&transport),"HDR secondary stale owner admitted")
        ||refused.resident||refused.gpuBudget||!refused.texels.empty())
    {error="Failed HDR secondary admission retained new resources";return false;}
    for(int row=0;row<2;++row) {
        auto bad=*scene;
        if(row==0) ++bad.lights[0].environmentImage.asset.version;
        else bad.lights[0].environmentImage.exposureStops=16;
        const auto owner=std::make_shared<const Visual3DScene>(std::move(bad));
        smoothglass::Prepared admitted;
        if(!smoothglass::prepare(owner,material,{},admitted,error)) return false;
        if(!reject(arbitgpu::environmentimage::prepare(*owner,material.get(),refused,error,
            arbitgpu::environmentimage::SecondaryRayConsumer::SmoothTransport,&admitted),
            "HDR secondary admitted stale image identity or clipped exposure overflow")
            ||refused.resident||refused.gpuBudget||!refused.texels.empty()) return false;
    }
    auto rough=std::make_shared<NativeFixtureSurfaceMaterialProgram>(*material);
    rough->parameters.emissionRoughness[3]=.2f;
    smoothglass::Prepared rejected;
    if(!reject(smoothglass::prepare(scene,std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>(rough),{},rejected,error),
        "HDR secondary owning admission broadened to rough interfaces")) return false;
    error.clear();return true;
}
inline bool run(arbitgpu::NativeFixtureSceneBackend& backend,std::string& error)
{
    visualenvironmentimage::Identity keys[2];
    for(int variant=0;variant<2;++variant) {
        const auto bytes=imageBytes(variant);
        keys[variant]={"native-hdr-secondary-"+std::to_string(variant),1,modelpayload::detail::digestHex(bytes)};
        const visualenvironmentimage::Metadata metadata{8,4,visualenvironmentimage::SourcePrimaries::Rec709D65};
        const auto transfer="native-hdr-secondary-upload-"+std::to_string(variant);
        auto& store=environmentimage::store();std::uint64_t received=0;
        if(!store.begin(transfer,keys[variant],metadata,bytes.size(),error)
            ||!store.append(transfer,0,environmentImageBase64(bytes),received,error)||!store.commit(transfer,error)) return false;
    }
    const auto target=backend.info().backend=="metal"?videohelper::materialprogram::BackendTarget::Metal:videohelper::materialprogram::BackendTarget::OpenGl;
    // Rows independently exercise miss, diffuse receiver, a nearer blocker,
    // its movement, orientation, exposure, tint, image replacement and camera.
    struct Case {bool water,receiver,blocker,moved;float orientation,exposure,angle;int variant;SceneVec3 tint;};
    const Case cases[]={{true,false,false,false,0,0,20,0,{1,1,1}},
        {true,false,false,false,90,0,20,0,{1,1,1}},{true,false,false,false,0,1,20,0,{1,1,1}},
        {true,false,false,false,0,0,20,0,{.5f,.75f,.25f}},{true,false,false,false,0,0,20,1,{1,1,1}},
        {true,false,false,false,0,0,35,0,{1,1,1}},
        {true,true,false,false,0,0,20,0,{1,1,1}},{true,true,true,false,0,0,20,0,{1,1,1}},
        {true,true,true,true,0,0,20,0,{1,1,1}},{true,true,false,false,90,1,20,0,{1,1,1}},
        {false,false,false,false,0,0,20,0,{1,1,1}},{false,false,false,false,90,0,20,0,{1,1,1}},
        {false,false,false,false,0,1,20,0,{1,1,1}},{false,false,false,false,0,0,20,1,{1,1,1}},
        {false,false,false,false,0,0,35,0,{1,1,1}},{false,false,false,false,0,0,20,0,{.5f,.75f,.25f}},
        {false,true,false,false,0,0,20,0,{1,1,1}},{false,true,true,false,0,0,20,0,{1,1,1}},
        {false,true,true,true,0,0,20,0,{1,1,1}},{false,true,false,false,90,1,20,0,{1,1,1}}};
    std::vector<std::vector<std::uint8_t>> allPixels;
    for(std::size_t row=0;row<std::size(cases);++row) {
        const auto& test=cases[row];
        auto authored=test.water?waterfixture::scene(test.angle):smoothglassfixture::scene(.25f,1,test.angle);
        authored.cameras[0].verticalFovRadians=.004f;
        authored.lights[0].color=test.tint;
        authored.lights[0].environmentImage={1,keys[test.variant],test.orientation,test.exposure};
        if(test.receiver) {
            waterfixture::receiver(authored,-3,0,{.04,.08,.12},6);
            authored.materials[1].baseColor={.3f,.2f,.1f};
        }
        if(test.blocker) {
            waterfixture::receiver(authored,-2,0,{.2,.04,.08},6);
            authored.materials[2].baseColor={.1f,.4f,.2f};
            if(test.moved) authored.objects[2].transform.translation.x=30;
        }
        const auto expected=trace(authored,test.water,test.variant,test.water?double(1.333f):double(1.5f),test.water?0:double(.004f));
        if(!expected.complete||expected.misses==0||expected.reflections==0
            ||(test.receiver&&expected.receivers==0)||(test.blocker&&!test.moved&&expected.blockers==0)
            ||(test.moved&&expected.blockers!=0))
        {error="Independent HDR secondary ray fixture did not exercise its declared geometry row "+std::to_string(row);return false;}
        const auto snapshot=std::make_shared<const Visual3DScene>(std::move(authored));
        const auto material=test.water?waterfixture::binding(snapshot,target,error):smoothglassfixture::binding(snapshot,target,1.5f,.004f,error);
        if(!material) return false;
        if((row==0||row==10)&&!admissionCases(snapshot,material->nativeProgram(),error)) return false;
        videorender::fixture3d::FixtureSceneRenderer renderer(backend);
        arbitgpu::NativeFixtureSceneRuntimeInputs inputs;inputs.linearColor=true;inputs.timeSeconds=.375f;
        videorender::fixture3d::RenderedFrame preview,exported;
        if(!renderer.renderPreview(snapshot,material,{},inputs,{9,9},"native-gpu",preview,error)
            ||!renderer.renderExport(snapshot,material,{},inputs,{9,9},"native-gpu",exported,error)
            ||!preview.stats.smoothGlassTransportApplied||preview.stats.smoothGlassQueryBytes==0)
        {error="HDR secondary ordinary preview/export failed: "+error;return false;}
        arbitgpu::NativeRawPassPixels pixels,exportPixels;
        if(!preview.nativeFrame->readRawPass(renderpassoutput::Output::Color,pixels,error)
            ||!exported.nativeFrame->readRawPass(renderpassoutput::Output::Color,exportPixels,error)
            ||pixels.format!=arbitgpu::NativeTexturePixelFormat::Rgba16Float||pixels.bytes.size()!=9*9*8||pixels.bytes!=exportPixels.bytes)
        {error="HDR secondary equal-time preview/export Color mismatch";return false;}
        for(int c=0;c<3;++c) {
            std::uint16_t half=0;std::memcpy(&half,pixels.bytes.data()+(4*9+4)*8+c*2,2);
            const double actual=hdrimage::halfToFloat(half);
            if(!std::isfinite(actual)||std::abs(actual-expected.rgb[c])>std::max(.006,std::abs(expected.rgb[c])*.002))
            {error="HDR secondary independent native pixel row "+std::to_string(row)+" channel "+std::to_string(c)
                +" expected "+std::to_string(expected.rgb[c])+", got "+std::to_string(actual);return false;}
        }
        allPixels.push_back(pixels.bytes);
        std::cout<<"environment-secondary-pixels backend="<<backend.info().backend<<" row="<<row<<'\n';
    }
    for(const auto pair: {std::pair<std::size_t,std::size_t>{0,1},{0,2},{0,3},{0,4},{0,5},{6,7},{7,8},{6,9},
        {10,11},{10,12},{10,13},{10,14},{10,15},{16,17},{17,18},{16,19}})
        if(allPixels[pair.first]==allPixels[pair.second])
        {error="HDR secondary authored control or blocker failed native pixel causality";return false;}
    // Existing analytic and constant-environment oracles run on the same new
    // consumer to guard its version branch. These produce native pixels too.
    return waterfixture::run(backend,error)&&smoothglassfixture::run(backend,error);
}
} // namespace videohelper::tests::environmentsecondaryfixture
