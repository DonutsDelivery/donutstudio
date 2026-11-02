#pragma once

#include "fixture_scene.h"
#include "../../../shared/HdrImageOutputContract.h"
#include "../../src/gpu_backend/backend.h"
#include "../../src/gpu_backend/fixture_analytic_sky.h"
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

namespace videohelper::tests
{
// Clouds must change what the frame shows. The sky is a flat (.1,.2,.8), so red is
// .1 everywhere until a cloud covers it, and a cloud's ambient white comes from the
// blue channel. The measure is cloudShare: the fraction of a 65 by 65 frame looking
// 50 degrees up whose red exceeds .15. The thresholds come from evaluating the
// recipe offline. There, coverage .35 gives a share of .42 and coverage 1 gives 1.
inline bool analyticSkyCloudCases(arbitgpu::NativeFixtureSceneBackend& backend,std::string& error)
{
    using namespace HarmonicMIDI::grid;
    struct Clouds { float coverage=0, type=0, drift=0, heading=0; };
    struct Frame { std::vector<std::uint8_t> bytes; std::vector<double> red; };
    const auto render=[&](Clouds clouds,bool mirror,Frame& output) {
        auto scene=fixture3d::makeScene();
        scene.vertexCount=4; scene.indexCount=6;
        scene.objects[0].vertexCount=4; scene.objects[0].indexCount=6;
        scene.ambientColor={}; scene.lightCount=1;
        scene.materials[0].baseColor={1,1,1}; scene.materials[0].baseColorTexture={};
        scene.materials[0].metallic=1; scene.materials[0].roughness=.05f; scene.materials[0].emissive={};
        auto& light=scene.lights[0]; light.kind=SceneLightKind::Environment; light.transform={};
        light.color={1,1,1}; light.intensity=1;
        light.sky.version=1;
        light.sky.zenith=light.sky.horizon=light.sky.ground={.1f,.2f,.8f};
        light.sky.sunIntensity=0; light.sky.sunRadiusDegrees=15;
        light.sky.cloudCoverage=clouds.coverage; light.sky.cloudType=clouds.type;
        light.sky.cloudDrift=clouds.drift; light.sky.cloudHeadingDegrees=clouds.heading;
        scene.cameras[0].transform={};
        scene.objects[0].transform={};
        if (mirror)
        {
            // A plane tilted 30 degrees sends the camera's center ray 60 degrees up.
            scene.objects[0].transform.rotation={-.2588190451f,0,0,.9659258263f};
            scene.objects[0].transform.translation={0,-.5f,.1339745962f};
            scene.cameras[0].transform.translation={0,0,5};
        }
        else
        {
            scene.objects[0].transform.translation={0,0,20};
            scene.cameras[0].transform.rotation={.4226182617f,0,0,.906307787f};
        }
        const auto snapshot=std::make_shared<const Visual3DScene>(std::move(scene));
        const auto prepared=backend.prepare(snapshot);
        if (!prepared.prepared) {error=prepared.error;return false;}
        arbitgpu::NativeFixtureSceneRuntimeInputs inputs; inputs.linearColor=true;
        const auto frame=backend.render(snapshot,prepared.resources,65,65,inputs);
        arbitgpu::NativeRawPassPixels pixels;
        if (!frame.rendered || !frame.frame
            || !frame.frame->readRawPass(renderpassoutput::Output::Color,pixels,error)
            || pixels.bytes.size()!=65*65*8)
        {error="Cloud frame missing owned native float pixels: "+frame.error+error;return false;}
        output.bytes=pixels.bytes;
        output.red.resize(65*65);
        for (std::size_t i=0;i<output.red.size();++i)
        {
            std::uint16_t half=0;std::memcpy(&half,pixels.bytes.data()+i*8,2);
            output.red[i]=hdrimage::halfToFloat(half);
        }
        return true;
    };
    const auto cloudShare=[](const Frame& frame) {
        std::size_t count=0;
        for (const auto red:frame.red) if (red>.15) ++count;
        return static_cast<double>(count)/frame.red.size();
    };
    const auto spread=[](const Frame& frame) {
        double mean=0,energy=0;
        for (const auto red:frame.red) mean+=red/frame.red.size();
        for (const auto red:frame.red) energy+=(red-mean)*(red-mean)/frame.red.size();
        return std::sqrt(energy);
    };
    Frame clear,parked,some,drifted,streaky,turned,closed,partial;
    if (!render({},false,clear) || !render({0,1,3,40},false,parked) || !render({.35f},false,some)
        || !render({.35f,0,.5f},false,drifted) || !render({.35f,1},false,streaky)
        || !render({.35f,0,0,90},false,turned) || !render({1},false,closed) || !render({.45f},false,partial))
        return false;
    if (cloudShare(clear)!=0 || spread(clear)>1e-3)
    {error="Clear sky frame is not the flat authored radiance";return false;}
    if (parked.bytes!=clear.bytes)
    {error="Zero cloud coverage changed the clear sky bytes";return false;}
    if (cloudShare(some)<.2 || cloudShare(some)>.7 || spread(some)<.1)
    {error="Cloud coverage .35 did not draw broken clouds, share "+std::to_string(cloudShare(some))
        +" spread "+std::to_string(spread(some));return false;}
    if (cloudShare(closed)<.98)
    {error="Cloud coverage 1 did not close the sky, share "+std::to_string(cloudShare(closed));return false;}
    if (drifted.bytes==some.bytes || streaky.bytes==some.bytes || turned.bytes==some.bytes)
    {error="Cloud drift, type or heading did not change the frame";return false;}
    const auto center=32*65+32;
    const auto uniforms=[] {
        SceneLightRecord light; light.kind=SceneLightKind::Environment; light.color={1,1,1}; light.intensity=1;
        light.sky.version=1; light.sky.zenith=light.sky.horizon=light.sky.ground={.1f,.2f,.8f};
        light.sky.sunIntensity=0; light.sky.sunRadiusDegrees=15; light.sky.cloudCoverage=.45f;
        return arbitgpu::analyticsky::uniforms(light);
    }();
    const auto twin=arbitgpu::analyticsky::radiance({0,.7660444431f,-.6427876097f},uniforms)[0];
    if (std::abs(partial.red[center]-twin)>.02)
    {error="Shader cloud radiance "+std::to_string(partial.red[center])+" differs from the C++ twin "
        +std::to_string(twin);return false;}
    // The BRDF path uses the same field, so a mirror must reflect the closed layer.
    Frame clearMirror,closedMirror;
    if (!render({},true,clearMirror) || !render({1},true,closedMirror)) return false;
    if (clearMirror.red[center]>.12 || closedMirror.red[center]<.4)
    {error="Mirror did not reflect the cloud layer: clear "+std::to_string(clearMirror.red[center])
        +" closed "+std::to_string(closedMirror.red[center]);return false;}
    std::cout << "analytic-sky-clouds backend=" << backend.info().backend
              << " share=" << cloudShare(some) << " spread=" << spread(some)
              << " closedShare=" << cloudShare(closed) << " mirror=" << closedMirror.red[center] << '\n';
    return true;
}

// Authored independent binary64 visible-normal and cosine quadrature values.
// No expected pixel is obtained by calling the production sky/BRDF function.
inline bool analyticSkyCases(arbitgpu::NativeFixtureSceneBackend& backend,std::string& error)
{
    using namespace HarmonicMIDI::grid;
    struct Case { bool background; float angle; float roughness; float azimuth;
                  float orientation; bool white; bool legacy; std::array<double,3> expected;
                  float elevation=0; float pitch=0; bool diffuse=false; bool secondConstant=false; };
    const Case cases[] {
        {true,0,.5f,0,0,false,false,{.5,.6,.7}},
        {true,0,.5f,180,0,false,false,{10.5,9.6,7.7}},
        {true,0,.5f,180,90,false,false,{.5,.6,.7}},
        {true,90,.5f,180,90,false,false,{10.5,9.6,7.7}},
        {true,0,.5f,0,0,false,false,{.217157287525381,.317157287525381,.4878679656440357},0,45},
        {true,0,.5f,180,0,false,false,{10.21715728752538,9.317157287525382,7.487867965644035},45,45},
        {true,0,.5f,0,0,false,false,{.1641242789363899,.1969491347236679,.2333095244168786},0,-45},
        {false,0,.5f,0,0,false,false,{.6884117635115949,.7307153041995945,.7574970296763973},0,0,true},
        {false,0,.5f,0,0,true,false,{.9965854273487498,.9965854273487498,.9965854273487498},0,0,true},
        {false,0,.05f,0,0,false,false,{10.496653180355233,9.596554314978272,7.6968525396719585}},
        {false,0,.5f,0,0,false,false,{1.519632468263313,1.4794720107461907,1.335433407684803}},
        {false,0,1,0,0,false,false,{.1815398071582062,.19697902151760202,.21063548367101992}},
        {false,60,.5f,0,0,false,false,{.5072721606652072,.5673504271959525,.6210139390992714}},
        {false,0,.5f,0,90,false,false,{.33589934166970137,.41411219681194095,.506820219069275}},
        {false,0,.05f,0,0,true,false,{.9999999997997575,.9999999997997575,.9999999997997575}},
        {false,0,.5f,0,0,true,false,{.9158759363193283,.9158759363193283,.9158759363193283}},
        {false,0,1,0,0,true,false,{.30688333050244215,.30688333050244215,.30688333050244215}},
        // Released fill policy: diffuse=.75, specular=.04*.5, exactly once.
        {false,0,.5f,0,0,false,true,{.154,.231,.308}},
        {false,0,.5f,0,0,false,true,{.308,.462,.616},0,0,false,true}
    };
    int row=0;
    for (const auto& test:cases)
    {
        auto scene=fixture3d::makeScene();
        scene.vertexCount=4; scene.indexCount=6;
        scene.objects[0].vertexCount=4; scene.objects[0].indexCount=6;
        scene.objects[0].transform={};
        scene.ambientColor={}; scene.lightCount=1;
        scene.materials[0].baseColor={1,1,1}; scene.materials[0].baseColorTexture={};
        scene.materials[0].metallic=test.legacy || test.diffuse ? 0 : 1;
        scene.materials[0].roughness=test.roughness; scene.materials[0].emissive={};
        auto& light=scene.lights[0]; light.kind=SceneLightKind::Environment; light.transform={};
        light.color={1,1,1}; light.intensity=1;
        if (!test.legacy)
        {
            light.sky.version=1;
            light.sky.zenith={.1f,.2f,.4f}; light.sky.horizon={.5f,.6f,.7f};
            light.sky.ground={.025f,.03f,.04f}; light.sky.sunColor={1,.9f,.7f};
            light.sky.sunIntensity=10; light.sky.sunRadiusDegrees=15;
            light.sky.sunAzimuthDegrees=test.azimuth; light.sky.sunElevationDegrees=test.elevation;
            light.sky.orientationDegrees=test.orientation;
            if (test.white)
            {
                light.sky.zenith=light.sky.horizon=light.sky.ground={1,1,1};
                light.sky.sunIntensity=0;
            }
        }
        else light.color={.2f,.3f,.4f};
        if (test.secondConstant)
        {
            scene.lights[1]=light; scene.lights[1].id.value+=1;
            scene.lightCount=2;
        }
        const double radians=test.angle*3.14159265358979323846/180;
        scene.cameras[0].transform={};
        const double pitch=test.pitch*3.14159265358979323846/180;
        scene.cameras[0].transform.rotation={static_cast<float>(std::cos(radians/2)*std::sin(pitch/2)),
            static_cast<float>(std::sin(radians/2)*std::cos(pitch/2)),
            static_cast<float>(-std::sin(radians/2)*std::sin(pitch/2)),
            static_cast<float>(std::cos(radians/2)*std::cos(pitch/2))};
        if (test.background) scene.objects[0].transform.translation={0,0,20};
        else scene.cameras[0].transform.translation={static_cast<float>(4*std::sin(radians)),0,
            static_cast<float>(1+4*std::cos(radians))};
        const auto snapshot=std::make_shared<const Visual3DScene>(std::move(scene));
        const auto prepared=backend.prepare(snapshot);
        if (!prepared.prepared) {error=prepared.error;return false;}
        arbitgpu::NativeFixtureSceneRuntimeInputs inputs; inputs.linearColor=true;
        if (row==0)
        {
            for (const bool secondSky:{false,true})
            {
                auto ambiguous=*snapshot;
                ambiguous.lightCount=2;
                ambiguous.lights[1]=ambiguous.lights[0]; ambiguous.lights[1].id.value+=1;
                if (!secondSky) ambiguous.lights[1].sky={};
                if (backend.prepare(std::make_shared<const Visual3DScene>(ambiguous)).prepared)
                {error="Mixed or multiple analytic sky environment owners were admitted";return false;}
                auto animated=inputs;
                animated.animatedLights.assign(ambiguous.lights.begin(),ambiguous.lights.begin()+2);
                if (backend.render(snapshot,prepared.resources,65,65,animated).rendered)
                {error="Animated mixed or multiple analytic sky environment owners were admitted";return false;}
            }
        }
        const auto frame=backend.render(snapshot,prepared.resources,65,65,inputs);
        arbitgpu::NativeRawPassPixels pixels,normals,depths;
        if (!frame.rendered || !frame.frame
            || !frame.frame->readRawPass(renderpassoutput::Output::Color,pixels,error)
            || !frame.frame->readRawPass(renderpassoutput::Output::Normal,normals,error)
            || !frame.frame->readRawPass(renderpassoutput::Output::Depth,depths,error)
            || pixels.format!=arbitgpu::NativeTexturePixelFormat::Rgba16Float
            || pixels.bytes.size()!=65*65*8 || normals.bytes.size()!=pixels.bytes.size()
            || depths.bytes.size()!=65*65*4)
        {error="Analytic sky missing owned native float pixels: "+frame.error+error;return false;}
        const auto offset=(32*65+32)*8;
        for (int c=0;c<4;++c)
        {
            std::uint16_t half=0;std::memcpy(&half,normals.bytes.data()+offset+2*c,2);
            const float normalExpected=test.background ? 0 : (c>=2 ? 1 : 0);
            if (hdrimage::halfToFloat(half)!=normalExpected)
            {error="Analytic sky sample is not the authored background/front plane";return false;}
        }
        float depth;std::memcpy(&depth,depths.bytes.data()+(32*65+32)*4,4);
        const auto expectedDepth=test.background ? 1.0
            : (4.0-snapshot->cameras[0].nearPlane)/(snapshot->cameras[0].farPlane-snapshot->cameras[0].nearPlane);
        if (!std::isfinite(depth) || std::abs(depth-expectedDepth)>1e-7)
        {error="Analytic sky sample depth differs from authored camera/plane";return false;}
        std::cout << "analytic-sky-pixels backend=" << backend.info().backend << " row=" << row++ << " rgb=";
        for (int c=0;c<4;++c)
        {
            std::uint16_t half=0;std::memcpy(&half,pixels.bytes.data()+offset+2*c,2);
            const double actual=hdrimage::halfToFloat(half);
            const double expected=c==3 ? 1 : test.expected[c];
            double low=expected,high=expected;
            if (c<3)
            {
                // Bound shader binary32 arithmetic then adjacent binary16 storage.
                const auto roundoff=expected*512*std::numeric_limits<float>::epsilon();
                int exponent=0;std::frexp(expected,&exponent);
                const auto step=std::ldexp(1.0,std::max(exponent-11,-24));
                low=std::floor((expected-roundoff)/step)*step;
                high=std::ceil((expected+roundoff)/step)*step;
            }
            if (!std::isfinite(actual) || actual<low || actual>high)
            {error="Analytic sky independent pixel row "+std::to_string(row-1)+" expected "+std::to_string(expected)
                +" in ["+std::to_string(low)+","+std::to_string(high)+"], got "+std::to_string(actual);return false;}
            if (c<3) std::cout << actual << ',';
        }
        std::cout << '\n';
    }
    return analyticSkyCloudCases(backend,error);
}
} // namespace videohelper::tests
