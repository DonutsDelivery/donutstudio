#pragma once

#include "environment_image_cases.h"
#include "smooth_glass_cases.h"
#include "../../src/gpu_backend/fixture_realtime_opengl.h"
#include <optional>

namespace videohelper::tests::environmentmaterialfixture
{
using namespace HarmonicMIDI::grid;
using Rgb = std::array<double,3>;

inline bool upload(visualenvironmentimage::Identity& key,std::string& error)
{
    const auto bytes=environmentImageBytes();
    key={"native-hdr-material",1,modelpayload::detail::digestHex(bytes)};
    const visualenvironmentimage::Metadata metadata{8,4,visualenvironmentimage::SourcePrimaries::Rec709D65};
    auto& store=environmentimage::store();std::uint64_t received=0;
    return store.begin("native-hdr-material-upload",key,metadata,bytes.size(),error)
        && store.append("native-hdr-material-upload",0,environmentImageBase64(bytes),received,error)
        && store.commit("native-hdr-material-upload",error);
}

inline bool center(arbitgpu::NativeFixtureSceneBackend& backend,const std::shared_ptr<const Visual3DScene>& scene,
    const std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram>& material,bool realtime,Rgb& rgb,std::string& error)
{
    constexpr int extent=33;
    std::optional<arbitgpu::realtimeengine::Scope> realtimeScope;
    if(realtime) realtimeScope.emplace();
    const auto prepared=backend.prepare(scene,material);
    if(!prepared.prepared) {error="HDR material row was refused: "+prepared.error;return false;}
    arbitgpu::NativeFixtureSceneRuntimeInputs inputs;inputs.linearColor=true;
    const auto frame=backend.render(scene,prepared.resources,extent,extent,inputs);
    arbitgpu::NativeRawPassPixels pixels;
    if(!frame.rendered||!frame.frame||frame.stats.realtimeProgramApplied!=realtime
        ||!frame.frame->readRawPass(renderpassoutput::Output::Color,pixels,error)
        ||pixels.format!=arbitgpu::NativeTexturePixelFormat::Rgba16Float||pixels.bytes.size()!=extent*extent*8)
    {error="HDR material row drew no float frame: "+frame.error+error;return false;}
    for(int c=0;c<3;++c) {
        std::uint16_t half=0;std::memcpy(&half,pixels.bytes.data()+(16*extent+16)*8+c*2,2);
        rgb[c]=hdrimage::halfToFloat(half);
        if(!std::isfinite(rgb[c])) {error="HDR material row wrote a nonfinite center";return false;}
    }
    return true;
}

// The image is the only light, so one more stop of exposure doubles the pixel.
inline bool followsImage(arbitgpu::NativeFixtureSceneBackend& backend,Visual3DScene authored,
    const std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram>& material,bool realtime,
    const visualenvironmentimage::Identity& key,const char* row,std::string& error)
{
    Rgb stops[2];
    for(int stop=0;stop<2;++stop) {
        authored.lights[0].environmentImage={1,key,0,float(stop)};
        if(!center(backend,std::make_shared<const Visual3DScene>(authored),material,realtime,stops[stop],error))
        {error=std::string(row)+": "+error;return false;}
    }
    for(int c=0;c<3;++c)
        if(stops[0][c]<=0||std::abs(stops[1][c]-2*stops[0][c])>stops[0][c]*.05)
        {error=std::string(row)+" does not follow the HDR image in channel "+std::to_string(c)
            +": "+std::to_string(stops[0][c])+" then "+std::to_string(stops[1][c]);return false;}
    return true;
}

inline Visual3DScene coatedScene()
{
    auto scene=fixture3d::makeScene();scene.vertexCount=4;scene.indexCount=6;
    scene.objects[0].vertexCount=4;scene.objects[0].indexCount=6;scene.objects[0].transform={};
    scene.ambientColor={};scene.lightCount=1;
    scene.materials[0].baseColor={1,1,1};scene.materials[0].baseColorTexture={};
    scene.materials[0].metallic=1;scene.materials[0].roughness=.5f;scene.materials[0].emissive={};
    auto& light=scene.lights[0];light.kind=SceneLightKind::Environment;light.color={1,1,1};light.intensity=1;
    scene.cameras[0].transform={};scene.cameras[0].transform.translation={0,0,5};
    return scene;
}

inline std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram> coatedMaterial(float clearcoat)
{
    auto material=std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>();
    material->backend=arbitgpu::NativeFixtureMaterialBackend::OpenGl;
    material->object=fixture3d::kCubeObjectId;
    material->bindingDigest="hdr-clearcoat-binding-"+std::to_string(clearcoat);
    material->programIdentity="hdr-clearcoat-program-"+std::to_string(clearcoat);
    material->baseColorSource=arbitgpu::NativeFixtureSurfaceMaterialProgram::BaseColorSource::ConstantLinear;
    material->parameters.baseColorMetallic={1,1,1,1};
    material->parameters.emissionRoughness={0,0,0,.5f};
    material->parameters.normalOpacity={0,0,1,1};
    material->parameters.transmissionIorClearcoat={0,1.5f,clearcoat,0};
    material->parameters.identifiers={fixture3d::kCubeMaterialId.value,0,0,0};
    return material;
}

inline bool glassCases(arbitgpu::NativeFixtureSceneBackend& backend,const visualenvironmentimage::Identity& key,std::string& error)
{
    auto authored=smoothglassfixture::scene(.25f,1,20);
    authored.cameras[0].verticalFovRadians=.004f;
    authored.lights[0].color={1,1,1};
    authored.lights[0].environmentImage={1,key,0,0};
    const auto snapshot=std::make_shared<const Visual3DScene>(authored);
    const auto material=smoothglassfixture::binding(snapshot,videohelper::materialprogram::BackendTarget::OpenGl,1.5f,.004f,error);
    if(!material) return false;
    return followsImage(backend,authored,material->nativeProgram(),true,key,"Realtime glass",error);
}

inline bool clearcoatCases(arbitgpu::NativeFixtureSceneBackend& backend,const visualenvironmentimage::Identity& key,std::string& error)
{
    const auto bare=coatedMaterial(0),coated=coatedMaterial(1);
    if(!followsImage(backend,coatedScene(),coated,false,key,"Clearcoat",error)
        ||!followsImage(backend,coatedScene(),coated,true,key,"Realtime clearcoat",error)) return false;
    // The analytic path's constant layer. Sky light leaves through (1 - 0.04 * clearcoat).
    auto authored=coatedScene();authored.lights[0].environmentImage={1,key,0,0};
    const auto snapshot=std::make_shared<const Visual3DScene>(authored);
    Rgb without,with;
    if(!center(backend,snapshot,bare,false,without,error)||!center(backend,snapshot,coated,false,with,error)) return false;
    for(int c=0;c<3;++c)
        if(without[c]<=0||std::abs(with[c]-without[c]*.96)>std::max(.001,without[c]*.002))
        {error="HDR clearcoat differs from the analytic constant layer in channel "+std::to_string(c);return false;}
    return true;
}

inline bool run(arbitgpu::NativeFixtureSceneBackend& backend,std::string& error)
{
    visualenvironmentimage::Identity key;
    return upload(key,error) && glassCases(backend,key,error) && clearcoatCases(backend,key,error);
}
} // namespace videohelper::tests::environmentmaterialfixture
