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

    // 5. Uniforms follow the lowered light. Scale is 2^-2 times intensity 2, so the tint
    //    is color*0.5. The orientation uniform is 90 degrees in radians.
    const auto uniforms=arbitgpu::environmentimage::uniforms(decoded->lights[0]);
    if (!check(nearly(uniforms[0][0],0.25f) && nearly(uniforms[0][1],0.375f)
               && nearly(uniforms[0][2],0.5f) && nearly(uniforms[0][3],2.0f),
               "exposure and tint uniforms differ from color*2^exposure*intensity")) return 1;
    if (!check(nearly(uniforms[1][0],1.5707963267948966f),
               "orientation uniform differs from 90 degrees in radians")) return 1;

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

    residents.reset();
    std::cout << "environment-image-scene-binding ok\n";
    return 0;
}
