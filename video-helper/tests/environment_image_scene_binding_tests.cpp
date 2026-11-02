// A stored Radiance environment image reaches the helper's environment store
// through the chunked payload transport, and a scene Environment light that names
// it resolves to that exact resident. Exposure and orientation uniforms follow the
// lowered light, and a stale SHA or version is refused with its reason.
//
// The GL/Metal pixel path is covered by environmentImageCases in
// opengl_fixture_scene_backend_tests.cpp. This file stops at prepare and uniforms.

#include "../src/gpu_backend/fixture_environment_image.h"
#include "../../shared/SceneAovOperationContract.h"
#include "support/fixture_scene.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace
{
using HarmonicMIDI::grid::SceneLightKind;
using HarmonicMIDI::grid::Visual3DScene;

// Same synthetic 8x4 RGBE image as support/environment_image_cases.h. That header
// also declares a GL/Metal backend case, so its bytes are repeated here instead of
// including it.
std::vector<std::uint8_t> syntheticRadianceBytes()
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

// Standard base64, as the plugin's juce::Base64 writes chunk data.
std::string base64(const std::vector<std::uint8_t>& bytes)
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

bool check(bool condition, const std::string& message)
{
    if (!condition) std::cerr << "environment image scene binding: " << message << '\n';
    return condition;
}

bool nearly(float actual, float expected)
{
    return std::abs(actual-expected) <= 1.0e-6f;
}
} // namespace

int main()
{
    using namespace videohelper;
    const auto bytes=syntheticRadianceBytes();
    const visualenvironmentimage::Identity key {"scene-binding-hdr",3,
        modelpayload::detail::digestHex(bytes)};
    const visualenvironmentimage::Metadata metadata {8,4,visualenvironmentimage::SourcePrimaries::Rec709D65};

    // 1. Chunked transport into the helper store, using the same begin, append and
    //    commit sequence as the environment_image_* RPC handlers. Chunks are 48 bytes,
    //    a multiple of three, so every non-final chunk encodes without padding.
    const std::string transfer="scene-binding-transfer";
    auto& residents=environmentimage::store();
    std::string error;
    if (!check(residents.begin(transfer,key,metadata,bytes.size(),error),
               "transfer begin refused: "+error)) return 1;
    for (std::size_t offset=0;offset<bytes.size();offset+=48)
    {
        const auto size=std::min<std::size_t>(48,bytes.size()-offset);
        const std::vector<std::uint8_t> chunk(bytes.begin()+static_cast<std::ptrdiff_t>(offset),
            bytes.begin()+static_cast<std::ptrdiff_t>(offset+size));
        std::uint64_t received=0;
        if (!check(residents.append(transfer,offset,base64(chunk),received,error)
                   && received==offset+size, "chunk at "+std::to_string(offset)+" refused: "+error)) return 1;
    }
    const auto committed=residents.commit(transfer,error);
    if (!check(committed && committed->payload, "commit refused: "+error)) return 1;
    if (!check(committed->payload->key().id==key.id && committed->payload->key().version==key.version
               && committed->payload->key().contentSha256==key.contentSha256,
               "committed resident identity differs from the transfer identity")) return 1;

    // 2. Lower one scene: the cube fixture with its key light replaced by an
    //    Environment light that names the stored image. Exposure -2 gives an exact
    //    scale of 0.25 and orientation 90 degrees.
    auto scene=std::make_unique<Visual3DScene>();
    *scene=fixture3d::makeScene();
    auto& light=scene->lights[0];
    light.kind=SceneLightKind::Environment;
    light.color={0.5f,0.75f,1.0f};
    light.intensity=2.0f;
    light.environmentImage.version=1;
    light.environmentImage.asset=key;
    light.environmentImage.orientationDegrees=90.0f;
    light.environmentImage.exposureStops=-2.0f;

    // 3. Scene transport: the canonical encoder and decoder that carry the
    //    environment extension inside the imported-scene render request.
    const auto encoded=sceneaov::detail::encodeScene(*scene);
    if (!check(!encoded.empty(), "scene with environment light failed validation")) return 1;
    auto decoded=std::make_unique<Visual3DScene>();
    if (!check(sceneaov::detail::decodeScene(encoded,*decoded), "scene decode refused the environment extension")) return 1;
    const auto& decodedImage=decoded->lights[0].environmentImage;
    if (!check(decoded->lights[0].kind==SceneLightKind::Environment && decodedImage.version==1
               && decodedImage.asset.id==key.id && decodedImage.asset.version==key.version
               && decodedImage.asset.contentSha256==key.contentSha256
               && nearly(decodedImage.orientationDegrees,90.0f) && nearly(decodedImage.exposureStops,-2.0f),
               "decoded scene lost the environment binding")) return 1;

    // 4. The helper resolves the decoded binding to the stored resident.
    arbitgpu::environmentimage::Prepared prepared;
    std::string prepareError;
    if (!check(arbitgpu::environmentimage::prepare(*decoded,nullptr,prepared,prepareError),
               "bound environment light was refused: "+prepareError)) return 1;
    if (!check(prepared.resident && prepared.resident->payload
               && prepared.resident->payload->key().id==key.id
               && prepared.resident->payload->key().version==key.version
               && prepared.resident->payload->key().contentSha256==key.contentSha256,
               "resolved resident is not the stored environment image")) return 1;
    if (!check(prepared.resident->image.pixels.size()==std::size_t(8)*4,
               "resolved resident pixel count differs from the 8x4 source")) return 1;
    if (!check(prepared.gpuBudget && prepared.texels.size()==prepared.resident->image.pixels.size(),
               "native staging does not cover every resident pixel")) return 1;
    // Every mip level is staged at its extent, and averaging keeps a uniform image uniform.
    if (!check(prepared.mips.size()==2 && prepared.mips[0].size()==std::size_t(4)*2 && prepared.mips[1].size()==std::size_t(2)*1,
               "the 8x4 image does not stage its 4x2 and 2x1 mip levels")) return 1;
    const std::vector<std::array<float,4>> uniformImage(8*4,std::array<float,4>{.25f,.5f,2.f,1.f});
    for (const auto& level:arbitgpu::environmentimage::buildMips(uniformImage,8,4))
        for (const auto& texel:level)
            if (!check(nearly(texel[0],.25f) && nearly(texel[1],.5f) && nearly(texel[2],2.f),
                       "a mip level changed a uniform image's radiance")) return 1;
    if (!check(arbitgpu::environmentimage::mipLevelCount(5,3)==2
                   && arbitgpu::environmentimage::mipBytes(8,4)==(8+2)*16,
               "mip level count or bytes differ from the staged chain")) return 1;

    // 5. Uniforms follow the lowered light. Scale is 2^-2 times intensity 2, so the tint
    //    is color*0.5. The orientation uniform is 90 degrees in radians.
    const auto uniforms=arbitgpu::environmentimage::uniforms(decoded->lights[0]);
    if (!check(nearly(uniforms[0][0],0.25f) && nearly(uniforms[0][1],0.375f)
               && nearly(uniforms[0][2],0.5f) && nearly(uniforms[0][3],2.0f),
               "exposure and tint uniforms differ from color*2^exposure*intensity")) return 1;
    if (!check(nearly(uniforms[1][0],1.5707963267948966f),
               "orientation uniform differs from 90 degrees in radians")) return 1;

    // Raster clips at the target's range, keeps hue, and leaves smaller values alone.
    {
        const float top=std::numeric_limits<float>::max();
        float small[3]{1.0f,2.0f,4.0f};
        arbitgpu::analyticsky::clipRasterRadiance(small,2.0);
        if (!check(small[0]==2.0f && small[1]==4.0f && small[2]==8.0f,"a gain inside the range changed the value")) return 1;
        float gained[3]{1.0e6f,5.0e5f,0.0f};
        arbitgpu::analyticsky::clipRasterRadiance(gained,64.0);
        if (!check(nearly(gained[0],60000.0f) && nearly(gained[1],30000.0f) && gained[2]==0.0f,"a gained value lost its hue")) return 1;
        float infinite[3]{std::numeric_limits<float>::infinity(),top,0.0f};
        arbitgpu::analyticsky::clipRasterRadiance(infinite,top);
        if (!check(std::isfinite(infinite[0]) && std::isfinite(infinite[1]) && infinite[0]<=60000.0f && infinite[2]==0.0f,
                   "an infinite emission times a huge gain did not stay finite")) return 1;
        float light[4]{top,top,1.0f,top};
        arbitgpu::analyticsky::clipRasterLight(light);
        if (!check(std::isfinite(light[0]*light[3]) && light[3]==1.0f,"a light at the largest float stayed infinite")) return 1;
        float plain[4]{1.0f,0.5f,0.25f,1000.0f};
        arbitgpu::analyticsky::clipRasterLight(plain);
        if (!check(plain[0]==1.0f && plain[3]==1000.0f,"an ordinary light changed")) return 1;
        const std::vector<std::array<float,4>> hottest(8*4,std::array<float,4>{top,top,top,1.f});
        for (const auto& level:arbitgpu::environmentimage::buildMips(hottest,8,4))
            for (const auto& texel:level)
                if (!check(std::isfinite(texel[0]) && std::isfinite(texel[1]) && std::isfinite(texel[2]),"a mip of the largest float went infinite")) return 1;
    }

    // 6. A stale or foreign binding is refused, and the reason is reported.
    const auto refused=[&](auto mutate,const char* reason)
    {
        auto candidate=std::make_unique<Visual3DScene>(*decoded);
        mutate(candidate->lights[0].environmentImage);
        arbitgpu::environmentimage::Prepared rejected;
        std::string rejectError;
        const bool failed=!arbitgpu::environmentimage::prepare(*candidate,nullptr,rejected,rejectError);
        return failed && !rejected.resident && rejectError==reason;
    };
    const char* notResident="HDR environment exact payload is not resident";
    if (!check(refused([](auto& image){ image.asset.contentSha256=std::string(64,'b'); },notResident),
               "wrong content SHA was not refused with its reason")) return 1;
    if (!check(refused([](auto& image){ image.asset.version=2; },notResident),
               "stale version 2 was not refused with its reason")) return 1;
    if (!check(refused([](auto& image){ image.asset.version=4; },notResident),
               "unstored version 4 was not refused with its reason")) return 1;
    if (!check(refused([](auto& image){ image.asset.id="scene-binding-other"; },notResident),
               "foreign asset id was not refused with its reason")) return 1;

    // 7. Every material lobe binds the image. The pixel cases that follow the image per
    //    lobe live in support/environment_material_cases.h.
    using Program=arbitgpu::NativeFixtureSurfaceMaterialProgram;
    const auto admits=[&](const Visual3DScene& scene,auto edit,const char* lobe)
    {
        Program program;
        program.parameters.normalOpacity={0,0,1,1};
        program.parameters.transmissionIorClearcoat={0,1.5f,0,0};
        edit(program);
        arbitgpu::environmentimage::Prepared admitted;
        std::string admitError;
        return check(arbitgpu::environmentimage::prepare(scene,&program,admitted,admitError)
                     && admitted.resident && admitted.gpuBudget,
                     std::string(lobe)+" was refused: "+admitError);
    };
    const auto transport=[](surfacematerial::TransportMode mode)
    { return static_cast<std::uint32_t>(mode); };
    if (!admits(*decoded,[&](Program& p){ p.parameters.transmissionIorClearcoat[0]=1;
            p.parameters.identifiers[1]=transport(surfacematerial::TransportMode::ClosedVolume); },"closed volume glass")) return 1;
    if (!admits(*decoded,[&](Program& p){ p.parameters.transmissionIorClearcoat[0]=1;
            p.parameters.identifiers[1]=transport(surfacematerial::TransportMode::ThinInterface); },"thin interface glass")) return 1;
    if (!admits(*decoded,[&](Program& p){ p.parameters.transmissionIorClearcoat[0]=1;
            p.parameters.identifiers[1]=transport(surfacematerial::TransportMode::ThinSurface); },"thin surface transmission")) return 1;
    if (!admits(*decoded,[](Program& p){ p.parameters.transmissionIorClearcoat[2]=1; },"clearcoat")) return 1;
    if (!admits(*decoded,[](Program& p){ p.kind=arbitgpu::NativeFixtureMaterialKind::DiffractionReflective; },"diffraction")) return 1;
    if (!admits(*decoded,[](Program& p){ p.parameters.normalOpacity[3]=.5f; },"blended surface")) return 1;
    if (!admits(*decoded,[](Program& p){
            auto glass=std::make_shared<Program>();
            glass->parameters.transmissionIorClearcoat={1,1.5f,0,0};
            p.objectPrograms.push_back(glass); },"glass object program")) return 1;
    auto blended=std::make_unique<Visual3DScene>(*decoded);
    blended->materials[0].alphaMode=HarmonicMIDI::grid::SceneAlphaMode::Blend;
    blended->materials[0].opacity=.5f;
    if (!admits(*blended,[](Program&){},"blended scene material")) return 1;

    residents.reset();
    std::cout << "environment-image-scene-binding ok\n";
    return 0;
}
