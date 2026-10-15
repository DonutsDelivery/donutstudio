#pragma once

#include "fixture_scene.h"
#include "../../../shared/HdrImageOutputContract.h"
#include "../../src/gpu_backend/backend.h"
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>

namespace videohelper::tests
{
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
    return true;
}
} // namespace videohelper::tests
