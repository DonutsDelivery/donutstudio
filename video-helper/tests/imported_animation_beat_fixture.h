#pragma once

#include "../src/imported_geometry_execution.h"
#include "../src/imported_animated_scene_payload_execution.h"
#include <nlohmann/json.hpp>
#include <cstring>

namespace importedanimationbeatfixture
{
// Independent skinned triangle. One joint translates X from 0 to 0.8 in two
// seconds. All three vertices have unit weight on that joint.
inline std::vector<std::uint8_t> bytes()
{
    nlohmann::json root = {
        {"asset", {{"version", "2.0"}}}, {"scene", 0},
        {"scenes", nlohmann::json::array({{{"nodes", {0, 1, 2}}}})},
        {"nodes", nlohmann::json::array({{{"mesh", 0}, {"skin", 0}},
            {{"name", "Translate joint"}}, {{"camera", 0}, {"translation", {0, 0, 4}}}})},
        {"cameras", nlohmann::json::array({{{"type", "perspective"},
            {"perspective", {{"yfov", 0.8}, {"znear", 0.1}, {"zfar", 100}}}}})},
        {"skins", nlohmann::json::array({{{"joints", {1}}}})},
        {"materials", nlohmann::json::array({{{"emissiveFactor", {1, 0.2, 0.1}},
            {"pbrMetallicRoughness", {{"baseColorFactor", {1, 1, 1, 1}}, {"metallicFactor", 0}}}}})},
        {"buffers", nlohmann::json::array()}, {"bufferViews", nlohmann::json::array()},
        {"accessors", nlohmann::json::array()}
    };
    std::vector<std::uint8_t> bin;
    const auto append = [](auto& target, std::uint32_t value) {
        for (unsigned shift = 0; shift < 32; shift += 8)
            target.push_back(static_cast<std::uint8_t>(value >> shift));
    };
    const auto floats = [&](std::initializer_list<float> values, const char* type, std::size_t count) {
        const auto offset = bin.size();
        for (auto value : values) {
            std::uint32_t bits; std::memcpy(&bits, &value, sizeof(bits)); append(bin, bits);
        }
        const auto view = root["bufferViews"].size();
        root["bufferViews"].push_back({{"buffer", 0}, {"byteOffset", offset}, {"byteLength", bin.size() - offset}});
        const auto accessor = root["accessors"].size();
        root["accessors"].push_back({{"bufferView", view}, {"componentType", 5126}, {"count", count}, {"type", type}});
        return accessor;
    };
    const auto positions = floats({-0.5f,-0.5f,0, 0.5f,-0.5f,0, 0,0.5f,0}, "VEC3", 3);
    const auto normals = floats({0,0,1, 0,0,1, 0,0,1}, "VEC3", 3);
    const auto jointsView = root["bufferViews"].size();
    root["bufferViews"].push_back({{"buffer",0}, {"byteOffset",bin.size()}, {"byteLength",12}});
    bin.insert(bin.end(),12,0);
    const auto joints = root["accessors"].size();
    root["accessors"].push_back({{"bufferView",jointsView}, {"componentType",5121}, {"count",3}, {"type","VEC4"}});
    const auto weights = floats({1,0,0,0, 1,0,0,0, 1,0,0,0}, "VEC4", 3);
    const auto times = floats({0,2}, "SCALAR", 2);
    const auto translations = floats({0,0,0, 0.8f,0,0}, "VEC3", 2);
    root["buffers"].push_back({{"byteLength", bin.size()}});
    root["meshes"] = nlohmann::json::array({{{"primitives", nlohmann::json::array({{
        {"attributes", {{"POSITION",positions}, {"NORMAL",normals}, {"JOINTS_0",joints}, {"WEIGHTS_0",weights}}},
        {"material",0}}})}}});
    root["animations"] = nlohmann::json::array({{{"name","Joint translation"},
        {"samplers", nlohmann::json::array({{{"input",times}, {"output",translations}, {"interpolation","LINEAR"}}})},
        {"channels", nlohmann::json::array({{{"sampler",0}, {"target",{{"node",1}, {"path","translation"}}}}})}}});
    auto json = root.dump(); while (json.size()%4) json += ' ';
    std::vector<std::uint8_t> result;
    for (auto value : {0x46546c67u,2u,std::uint32_t(28+json.size()+bin.size()),std::uint32_t(json.size()),0x4e4f534au})
        append(result,value);
    result.insert(result.end(),json.begin(),json.end());
    append(result,std::uint32_t(bin.size())); append(result,0x004e4942u);
    result.insert(result.end(),bin.begin(),bin.end()); return result;
}

inline videotime::BeatTimeline timeline()
{
    videotime::BeatTimeline result;
    result.set({{0,60,true},{2,120,true}},{{0,4,4}});
    return result;
}

inline bool publish(videohelper::modelpayload::Store& store, visualanimationimport::ExactContentAssetKey& key)
{
    const auto data=bytes();
    key={"beat-joint-triangle",1,videohelper::modelpayload::detail::digestHex(data),"model/gltf-binary",data.size()};
    constexpr char alphabet[]="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string encoded;
    for (std::size_t offset=0;offset<data.size();offset+=3) {
        const auto remaining=data.size()-offset;
        const auto value=std::uint32_t(data[offset])<<16 | (remaining>1 ? std::uint32_t(data[offset+1])<<8 : 0)
            | (remaining>2 ? data[offset+2] : 0);
        encoded+=alphabet[(value>>18)&63]; encoded+=alphabet[(value>>12)&63];
        encoded+=remaining>1 ? alphabet[(value>>6)&63] : '='; encoded+=remaining>2 ? alphabet[value&63] : '=';
    }
    return store.begin(key.id,key) && store.appendBase64(key.id,0,encoded) && store.commit(key.id);
}

inline visualanimationimport::Request request(const visualanimationimport::ExactContentAssetKey& key)
{
    visualanimationimport::Request result;
    result.sourceStableId=71; result.deformationStableId=72; result.schedule={71,72};
    result.asset=key; result.sceneIndex=0; result.meshStableId=1; result.animationClipStableId=1;
    result.clipName="Joint translation";
    result.playback.timeSource=visualanimation::TimeSource::BeatSync;
    result.playback.timelineBeat=0.5; result.playback.clipStartBeat=1; result.playback.beatsPerLoop=2;
    result.playback.trimStartSeconds=0.25; result.playback.trimEndSeconds=1.25;
    result.playback.playback=visualanimation::Playback::Loop;
    return result;
}

// This is independent of BeatTimeline and resolvePlaybackControl. At the
// authored tempo step, beat=s before 2s and beat=2+2*(s-2) afterwards.
inline double expectedSample(double seconds, bool loop=true)
{
    const auto beat=seconds<=2 ? seconds : 2+2*(seconds-2);
    const auto position=(beat+0.5-1)*0.5;
    if (!loop) return std::clamp(0.25+position,0.25,1.25);
    auto local=std::fmod(position,1.0); if (local<0) local+=1;
    return 0.25+local;
}

template<class ReadPixels>
bool verify(arbitgpu::NativeDeformationBackend& backend, ReadPixels readPixels, std::string& error)
{
    using namespace videohelper;
    modelpayload::Store store;
    visualanimationimport::ExactContentAssetKey key;
    if (!publish(store,key)) { error="Beat fixture exact publication failed"; return false; }
    const auto authored=request(key);
    const auto beatTimeline=timeline();
    const auto admitted=geometry::prepareAnimatedGeometry(store.resolvePreview(key),authored,73,error);
    if (!admitted) return false;
    const auto base=*admitted->descriptor().operations.front().retainedMesh;
    geometry::ImportedGeometryExecution preview(&store),exported(&store),partition(&store);
    modelpayload::ImportedAnimatedScenePayloadExecution nativePreview(store,backend),nativeExport(store,backend);
    for (const bool loop : {true,false})
        for (const auto frame : std::array<visualdeformation::RationalFrameTime,11> {{
            {0,24,1},{12,24,1},{24,24,1},{36,24,1},{48,24,1},{54,24,1},
            {60,24,1},{72,24,1},{12,24,1},{0,24,1},{12,24000,1001}}}) {
            const double seconds=static_cast<double>(frame.frame)*frame.rateDenominator/frame.rateNumerator;
            auto operation=authored;
            operation.playback.playback=loop ? visualanimation::Playback::Loop : visualanimation::Playback::Clamp;
            auto a=base,b=base,c=base;
            a.importedAnimation=b.importedAnimation=c.importedAnimation=visualanimationoperation::encode(operation);
            if (!preview.evaluate(a,frame,9,false,error,&beatTimeline)
                || !exported.evaluate(b,frame,9,true,error,&beatTimeline)
                || !partition.evaluate(c,{frame.frame-3,frame.rateNumerator,frame.rateDenominator},9,false,error,&beatTimeline)
                || !partition.evaluate(c,frame,9,false,error,&beatTimeline)) return false;
            for (std::size_t vertex=0;vertex<3;++vertex) {
                const auto expectedX=base.geometry.positions[vertex].x+0.4*expectedSample(seconds,loop);
                if (std::abs(a.geometry.positions[vertex].x-expectedX)>0.00001
                    || std::abs(b.geometry.positions[vertex].x-expectedX)>0.00001
                    || std::abs(c.geometry.positions[vertex].x-expectedX)>0.00001
                    || a.geometry.positions[vertex].y!=base.geometry.positions[vertex].y
                    || a.geometry.vertexIds!=base.geometry.vertexIds || a.geometry.indices!=base.geometry.indices) {
                    error="Native beat joint position disagrees with the independent trimmed tempo-step oracle"; return false;
                }
            }
            modelpayload::ImportedAnimatedSceneRequest native;
            native.operation=operation; native.frame=frame; native.structuralRevision=9; native.width=96; native.height=96;
            if (!importedanimation::applyImportedAnimationPlaybackClock(native.operation.playback,frame,&beatTimeline,seconds,error)) return false;
            modelpayload::ImportedAnimatedSceneReceipt p,e;
            if (!nativePreview.executePreview(native,p,error) || !nativeExport.executeExport(native,e,error)) return false;
            if (!p.evaluation.deformation || !e.evaluation.deformation
                || std::abs(p.evaluation.deformation->sampleTimeSeconds()-expectedSample(seconds,loop))>0.000001
                || p.evaluation.deformation->sampleTimeSeconds()!=e.evaluation.deformation->sampleTimeSeconds()
                || readPixels(p.frame.nativeFrame).empty() || readPixels(p.frame.nativeFrame)!=readPixels(e.frame.nativeFrame)) {
                error="Native beat preview/export sample time or pixels disagree"; return false;
            }
        }
    auto invalid=base;
    if (preview.evaluate(invalid,{24,24,1},9,false,error)) {
        error="Beat animation accepted a missing project tempo map"; return false;
    }
    auto bad=authored; bad.playback.trimEndSeconds=2.5;
    invalid.importedAnimation=visualanimationoperation::encode(bad);
    if (preview.evaluate(invalid,{24,24,1},9,false,error,&beatTimeline)) {
        error="Beat animation accepted a trim beyond the admitted clip"; return false;
    }
    error.clear(); return true;
}
} // namespace importedanimationbeatfixture
