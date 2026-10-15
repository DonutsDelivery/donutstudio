#pragma once

#include "fixture_scene.h"
#include "../../src/environment_image_payload_transport.h"
#include "../../../shared/HdrImageOutputContract.h"
#include <cstring>
#include <iostream>

namespace videohelper::tests
{
inline std::vector<std::uint8_t> environmentImageBytes()
{
    const std::string header="#?RADIANCE\nFORMAT=32-bit_rle_rgbe\nPRIMARIES=0.64 0.33 0.30 0.60 0.15 0.06 0.3127 0.3290\n\n-Y 4 +X 8\n";
    std::vector<std::uint8_t> bytes(header.begin(),header.end());
    for (int y=0;y<4;++y)
    {
        bytes.insert(bytes.end(),{2,2,0,8});
        for (int c=0;c<4;++c)
        {
            bytes.push_back(8);
            for (int x=0;x<8;++x)
                bytes.push_back(static_cast<std::uint8_t>(c==0 ? 16+3*x*x+4*y : c==1 ? 16+2*x+8*y : c==2 ? 64+3*x+y : 132));
        }
    }
    return bytes;
}
inline std::string environmentImageBase64(const std::vector<std::uint8_t>& bytes)
{
    constexpr char alphabet[]="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result;
    for (std::size_t i=0;i<bytes.size();i+=3)
    {
        const auto count=std::min(std::size_t(3),bytes.size()-i);
        const unsigned packed=(unsigned(bytes[i])<<16)|(count>1 ? unsigned(bytes[i+1])<<8 : 0)|(count>2 ? bytes[i+2] : 0);
        result+=alphabet[(packed>>18)&63]; result+=alphabet[(packed>>12)&63];
        result+=count>1 ? alphabet[(packed>>6)&63] : '='; result+=count>2 ? alphabet[packed&63] : '=';
    }
    return result;
}
inline bool environmentImageCases(arbitgpu::NativeFixtureSceneBackend& backend,std::string& error)
{
    using namespace HarmonicMIDI::grid;
    const auto bytes=environmentImageBytes();
    const visualenvironmentimage::Identity key {"native-hdr-oracle",1,modelpayload::detail::digestHex(bytes)};
    const visualenvironmentimage::Metadata metadata {8,4,visualenvironmentimage::SourcePrimaries::Rec709D65};
    const auto encoded=environmentImageBase64(bytes);
    auto& store=environmentimage::store(); std::uint64_t received=0;
    if (!store.begin("native-hdr-upload",key,metadata,bytes.size(),error)
        || !store.append("native-hdr-upload",0,encoded,received,error)
        || !store.commit("native-hdr-upload",error)) return false;
    // Independently authored binary64 values. The polynomial texels above are
    // RGBE /16. Reference sampling uses atan2/acos and pixel-center bilinear
    // interpolation. Reflection uses the GGX normal-incidence distribution,
    // reflected directions, Smith G1 and the fixed 64 midpoint/radical samples.
    // Expected values never call the production decoder, sampler or shader.
    struct Case {bool background;float angle,orientation,exposure,roughness;std::array<double,3> expected;float pitch=0;};
    const Case cases[] {
        {true,0,0,0,.5f,{5.96875,2.1875,4.75}},
        {true,90,0,0,.5f,{1.84375,1.9375,4.375}},
        {true,180,0,0,.5f,{3.71875,2.1875,4.75}}, // +Z is u=.5
        {true,270,0,0,.5f,{7.09375,2.4375,5.125}},
        {true,0,90,0,.5f,{7.09375,2.4375,5.125}},
        {true,0,0,1,.5f,{11.9375,4.375,9.5}},
        {true,0,0,0,.5f,{5.71875,1.6875,4.6875},45},
        {true,0,0,0,.5f,{6.21875,2.6875,4.8125},-45},
        {false,0,0,0,.25f,{3.7294984264035205,2.1841593345485726,4.75025087997509}},
        {false,0,0,0,.5f,{3.4651622657678294,2.005885137659506,4.351393108419195}},
        {false,0,0,0,1,{1.1849052712742287,.6731472858287398,1.4575755807711797}},
        {false,0,90,0,.5f,{1.7633890283443856,1.7789681953622702,4.011017694973341}},
        {false,0,0,1,.5f,{6.930324531535659,4.011770275319012,8.70278621683839}}
    };
    std::size_t row=0;
    for (const auto& test:cases)
    {
        auto scene=fixture3d::makeScene(); scene.vertexCount=4; scene.indexCount=6;
        scene.objects[0].vertexCount=4; scene.objects[0].indexCount=6; scene.objects[0].transform={};
        scene.ambientColor={}; scene.lightCount=1;
        scene.materials[0].baseColor={1,1,1}; scene.materials[0].baseColorTexture={};
        scene.materials[0].metallic=1; scene.materials[0].roughness=test.roughness; scene.materials[0].emissive={};
        auto& light=scene.lights[0]; light.kind=SceneLightKind::Environment; light.color={1,1,1}; light.intensity=1;
        light.environmentImage={1,key,test.orientation,test.exposure};
        scene.cameras[0].transform={};
        const double radians=test.angle*3.14159265358979323846/180;
        const double pitch=test.pitch*3.14159265358979323846/180;
        scene.cameras[0].transform.rotation={static_cast<float>(std::cos(radians/2)*std::sin(pitch/2)),
            static_cast<float>(std::sin(radians/2)*std::cos(pitch/2)),
            static_cast<float>(-std::sin(radians/2)*std::sin(pitch/2)),static_cast<float>(std::cos(radians/2)*std::cos(pitch/2))};
        if (test.background) scene.objects[0].transform.translation={0,1000,20};
        else scene.cameras[0].transform.translation={0,0,5};
        const auto snapshot=std::make_shared<const Visual3DScene>(scene);
        const auto prepared=backend.prepare(snapshot);
        if (!prepared.prepared) {error=prepared.error;return false;}
        arbitgpu::NativeFixtureSceneRuntimeInputs inputs; inputs.linearColor=true;
        const auto frame=backend.render(snapshot,prepared.resources,65,65,inputs);
        arbitgpu::NativeRawPassPixels pixels;
        if (!frame.rendered || !frame.frame || !frame.frame->readRawPass(renderpassoutput::Output::Color,pixels,error)
            || pixels.format!=arbitgpu::NativeTexturePixelFormat::Rgba16Float || pixels.bytes.size()!=65*65*8)
        {error="HDR environment native float output missing: "+frame.error+error;return false;}
        for (std::size_t c=0;c<3;++c)
        {
            std::uint16_t half=0;std::memcpy(&half,pixels.bytes.data()+(32*65+32)*8+2*c,2);
            const double actual=hdrimage::halfToFloat(half),expected=test.expected[c];
            if (!std::isfinite(actual) || std::abs(actual-expected)>std::max(.001,expected*.001))
            {error="HDR environment independent pixel row "+std::to_string(row)+" expected "+std::to_string(expected)+", got "+std::to_string(actual);return false;}
        }
        const auto again=backend.render(snapshot,prepared.resources,65,65,inputs);
        arbitgpu::NativeRawPassPixels repeat;
        if (!again.rendered || !again.frame || !again.frame->readRawPass(renderpassoutput::Output::Color,repeat,error)
            || pixels.bytes!=repeat.bytes)
        {error="HDR environment same-frame native replay changed pixels";return false;}
        if (row==0)
        {
            auto bad=scene; bad.lights[0].environmentImage.asset.version++;
            if (backend.prepare(std::make_shared<const Visual3DScene>(bad)).prepared)
            {error="HDR environment stale exact version admitted";return false;}
            bad=scene; bad.lights[0].environmentImage.asset.contentSha256=std::string(64,'b');
            if (backend.prepare(std::make_shared<const Visual3DScene>(bad)).prepared)
            {error="HDR environment stale digest admitted";return false;}
            bad=scene; bad.lights[0].environmentImage.exposureStops=16;
            if (backend.prepare(std::make_shared<const Visual3DScene>(bad)).prepared)
            {error="HDR environment over-range radiance was clipped";return false;}
            auto animated=inputs; animated.animatedLights={scene.lights[0]};
            animated.animatedLights[0].environmentImage.exposureStops=16;
            if (backend.render(snapshot,prepared.resources,65,65,animated).rendered)
            {error="HDR environment animated exposure overflow admitted";return false;}
        }
        std::cout<<"environment-image-pixels backend="<<backend.info().backend<<" row="<<row++<<'\n';
    }
    environmentimage::Store isolated;
    auto forged=key; forged.contentSha256=std::string(64,'a');
    if (!isolated.begin("digest-negative",forged,metadata,bytes.size(),error)
        || !isolated.append("digest-negative",0,encoded,received,error)
        || isolated.commit("digest-negative",error))
    {error="HDR environment digest mismatch admission did not fail";return false;}
    if (isolated.begin("oversize",key,metadata,visualenvironmentimagepayload::kMaxPayloadBytes+1,error))
    {error="HDR environment oversized source admitted";return false;}
    auto malformed=bytes; malformed.pop_back();
    auto malformedKey=key; malformedKey.contentSha256=modelpayload::detail::digestHex(malformed);
    if (!isolated.begin("malformed",malformedKey,metadata,malformed.size(),error)
        || !isolated.append("malformed",0,environmentImageBase64(malformed),received,error)
        || isolated.commit("malformed",error))
    {error="HDR environment malformed exact bytes admitted";return false;}
    auto wrong=metadata;wrong.sourcePrimaries=visualenvironmentimage::SourcePrimaries::RadianceEqualEnergy;
    if (!isolated.begin("metadata",key,wrong,bytes.size(),error)
        || !isolated.append("metadata",0,encoded,received,error) || isolated.commit("metadata",error))
    {error="HDR environment mismatched declared color primaries admitted";return false;}
    std::vector<environmentimage::ResidentPtr> held;
    for (std::size_t i=0;i<visualenvironmentimagepayload::kMaxResidentPayloads;++i)
    {
        auto residentKey=key;residentKey.id="resident-"+std::to_string(i);
        const auto transfer="resident-upload-"+std::to_string(i);
        if (!isolated.begin(transfer,residentKey,metadata,bytes.size(),error)
            || !isolated.append(transfer,0,encoded,received,error)) return false;
        auto resident=isolated.commit(transfer,error);
        if (!resident) return false;
        held.push_back(std::move(resident));
    }
    auto ninth=key;ninth.id="resident-nine";
    if (!isolated.begin("resident-overflow",ninth,metadata,bytes.size(),error)
        || !isolated.append("resident-overflow",0,encoded,received,error)
        || isolated.commit("resident-overflow",error))
    {error="HDR environment aggregate resident count admitted nine leased owners";return false;}
    error.clear();return true;
}
} // namespace videohelper::tests
