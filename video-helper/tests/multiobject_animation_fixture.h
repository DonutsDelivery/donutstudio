#pragma once

#include "../src/imported_animated_scene_payload_execution.h"
#include "../../shared/VisualStarterModelAssets.h"
#include <nlohmann/json.hpp>
#include <cstring>

namespace multiobjectanimationfixture {
inline std::vector<std::uint8_t> bytes(bool rigidOnly = false, bool repeated = false, unsigned sceneOnly = 0) {
    const auto& original = visualstartermodel::kAnimatedTriangleGlb;
    const auto read = [](const std::uint8_t* p) {
        return std::uint32_t(p[0]) | std::uint32_t(p[1]) << 8
            | std::uint32_t(p[2]) << 16 | std::uint32_t(p[3]) << 24;
    };
    const auto jsonSize = read(original.data() + 12);
    auto root = nlohmann::json::parse(original.begin() + 20, original.begin() + 20 + jsonSize);
    const auto binStart = 28u + jsonSize;
    std::vector<std::uint8_t> bin(original.begin() + binStart,
        original.begin() + binStart + read(original.data() + binStart - 8));
    const auto append = [](auto& target, std::uint32_t value) {
        for (unsigned shift = 0; shift < 32; shift += 8)
            target.push_back(static_cast<std::uint8_t>(value >> shift));
    };
    const auto outputOffset = bin.size();
    const auto inputAccessor = root["animations"][0]["samplers"][0]["input"].get<std::size_t>();
    const auto keyCount = root["accessors"][inputAccessor]["count"].get<std::size_t>();
    for (std::size_t key = 0; key < keyCount; ++key)
        for (const float value : {0.35f * float(key) / float(keyCount - 1), 0.0f, 0.0f}) {
            std::uint32_t bits; std::memcpy(&bits, &value, sizeof(bits)); append(bin, bits);
        }
    root["buffers"][0]["byteLength"] = bin.size();
    const auto view = root["bufferViews"].size();
    root["bufferViews"].push_back({{"buffer", 0}, {"byteOffset", outputOffset}, {"byteLength", keyCount * 12}});
    const auto accessor = root["accessors"].size();
    root["accessors"].push_back({{"bufferView", view}, {"componentType", 5126}, {"count", keyCount}, {"type", "VEC3"}});
    root["materials"].push_back(root["materials"][0]);
    root["materials"][1]["pbrMetallicRoughness"]["baseColorFactor"] = {0.1, 0.8, 0.2, 1.0};
    root["meshes"].push_back(root["meshes"][0]);
    root["meshes"][1]["primitives"][0]["material"] = 1;
    root["meshes"][1]["primitives"].push_back(root["meshes"][1]["primitives"][0]);
    auto left = root["nodes"][0];
    left["translation"] = {-0.45, 0.0, 0.0}; left["scale"] = {0.45, 0.45, 0.45};
    const auto leftIndex = root["nodes"].size(); root["nodes"].push_back(left);
    left["mesh"] = 1; left["translation"] = {0.45, 0.0, 0.0};
    const auto rightIndex = root["nodes"].size(); root["nodes"].push_back(left);
    if (repeated) {
        root["nodes"][rightIndex]["mesh"] = 0;
        root["nodes"][leftIndex]["name"] = "Repeated Object";
        root["nodes"][rightIndex]["name"] = "Repeated Object";
        root["meshes"].erase(root["meshes"].begin() + 1);
    }
    root["nodes"][0] = {{"children", {leftIndex, rightIndex}}};
    auto& clip = root["animations"][0];
    clip["channels"][0]["target"]["node"] = leftIndex;
    auto rightChannel = clip["channels"][0]; rightChannel["target"]["node"] = rightIndex;
    clip["channels"].push_back(rightChannel);
    const auto sampler = clip["samplers"].size();
    clip["samplers"].push_back({{"input", clip["samplers"][0]["input"]}, {"output", accessor}, {"interpolation", "LINEAR"}});
    clip["channels"].push_back({{"sampler", sampler}, {"target", {{"node", 0}, {"path", "translation"}}}});
    if (rigidOnly) {
        for (auto& mesh : root["meshes"]) {
            mesh.erase("weights");
            for (auto& primitive : mesh["primitives"]) primitive.erase("targets");
        }
        auto parentSampler = clip["samplers"].back();
        auto parentChannel = clip["channels"].back(); parentChannel["sampler"] = 0;
        clip["samplers"] = nlohmann::json::array({parentSampler});
        clip["channels"] = nlohmann::json::array({parentChannel});
    }
    if (sceneOnly!=0) {
        const auto target=sceneOnly==1 ? 1u : 2u;
        const auto parent=root["nodes"].size();
        root["nodes"].push_back({{"name",sceneOnly==1 ? "Camera Rig" : "Light Rig"},{"children",{target}}});
        for (auto& node : root["scenes"][0]["nodes"]) if (node==target) node=parent;
        clip["channels"][0]["target"]["node"]=parent;
        if (sceneOnly==2) {
            root["extensions"]["KHR_lights_punctual"]["lights"][0]["type"]="point";
            root["extensions"]["KHR_lights_punctual"]["lights"][0]["range"]=10.0;
            root["nodes"][target]["translation"]={0.0,0.0,2.0};
        }
    }
    auto json = root.dump(); while (json.size() % 4) json += ' ';
    while (bin.size() % 4) bin.push_back(0);
    std::vector<std::uint8_t> result;
    for (const auto value : {0x46546c67u, 2u, std::uint32_t(28 + json.size() + bin.size()),
                            std::uint32_t(json.size()), 0x4e4f534au}) append(result, value);
    result.insert(result.end(), json.begin(), json.end());
    append(result, static_cast<std::uint32_t>(bin.size())); append(result, 0x004e4942u);
    result.insert(result.end(), bin.begin(), bin.end()); return result;
}

inline bool publish(videohelper::modelpayload::Store& store,
                    visualanimationimport::ExactContentAssetKey& key, bool rigidOnly = false, bool repeated = false,
                    unsigned sceneOnly = 0) {
    const auto data = bytes(rigidOnly, repeated, sceneOnly);
    key = {"multiobject-animation", 1, videohelper::modelpayload::detail::digestHex(data),
           "model/gltf-binary", data.size()};
    constexpr char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string encoded;
    for (std::size_t offset = 0; offset < data.size(); offset += 3) {
        const auto remaining = data.size() - offset;
        const auto value = std::uint32_t(data[offset]) << 16
            | (remaining > 1 ? std::uint32_t(data[offset + 1]) << 8 : 0)
            | (remaining > 2 ? data[offset + 2] : 0);
        encoded += alphabet[(value >> 18) & 63]; encoded += alphabet[(value >> 12) & 63];
        encoded += remaining > 1 ? alphabet[(value >> 6) & 63] : '=';
        encoded += remaining > 2 ? alphabet[value & 63] : '=';
    }
    return store.begin(key.id, key) && store.appendBase64(key.id, 0, encoded) && store.commit(key.id);
}

template<class ReadPixels>
bool verifySceneOnly(arbitgpu::NativeDeformationBackend& backend, ReadPixels readPixels,
                     unsigned target, std::string& error) {
    videohelper::modelpayload::Store store;
    visualanimationimport::ExactContentAssetKey key;
    if (!publish(store,key,true,false,target)) { error="scene-only fixture publication failed"; return false; }
    videohelper::modelpayload::ImportedAnimatedSceneCompatibility choices;
    if (!videohelper::modelpayload::ImportedAnimatedScenePayloadExecution::inspectCompatibility(
            store.resolvePreview(key),0,choices,error)) return false;
    if (choices.clips.size()!=1 || choices.clips[0].compatibleMeshStableIds
        !=std::vector<visualanimationimport::StableId>{1,2}) {
        error="camera/light-only clip must be selectable with static meshes"; return false;
    }
    videohelper::modelpayload::ImportedAnimatedSceneRequest request;
    request.operation.sourceStableId=1; request.operation.deformationStableId=2;
    request.operation.schedule={1,2}; request.operation.asset=key; request.operation.sceneIndex=0;
    request.operation.animationClipStableId=1; request.operation.clipName=std::string(visualstartermodel::kClipName);
    request.structuralRevision=8; request.width=96; request.height=96; request.frame={0,24000,1001};
    videohelper::modelpayload::ImportedAnimatedScenePayloadExecution preview(store,backend),exported(store,backend);
    videohelper::modelpayload::ImportedAnimatedSceneReceipt start,moved,reset,output;
    if (!preview.executePreview(request,start,error)) return false;
    request.frame={12,24000,1001}; request.operation.playback.timelineSeconds=12.0*1001.0/24000.0;
    if (!preview.executePreview(request,moved,error) || !exported.executeExport(request,output,error)) return false;
    if (moved.evaluation.deformation || !moved.evaluation.sceneAnimation || !moved.source->draws.empty()
        || moved.source->deformation || moved.frame.stats.dispatchCount!=0 || moved.frame.stats.drawCount!=3
        || start.source!=moved.source || !moved.valid()) {
        error="camera/light-only animation must render without a synthetic mesh deformation binding"; return false;
    }
    const auto first=readPixels(start.frame.nativeFrame),next=readPixels(moved.frame.nativeFrame);
    if (first.empty() || first==next || next!=readPixels(output.frame.nativeFrame)) {
        error="camera/light-only preview must move and match rational-time export"; return false;
    }
    if (target==1) request.runtimeInputs.cameraOverride=start.runtimeInputs.animatedCamera;
    else request.runtimeInputs.lightOverride=start.runtimeInputs.animatedLights.front();
    if (!preview.executePreview(request,output,error) || readPixels(output.frame.nativeFrame)!=first) {
        error="explicit graph camera/light selection must override imported animation"; return false;
    }
    request.runtimeInputs={}; request.frame={0,24000,1001}; request.operation.playback.timelineSeconds=0;
    if (!preview.executePreview(request,reset,error) || readPixels(reset.frame.nativeFrame)!=first) {
        error="camera/light-only animation must rewind without prior-frame state"; return false;
    }
    request.operation.meshStableId=2;
    if (!exported.executeExport(request,output,error) || output.frame.stats.drawCount!=2) {
        error="camera/light-only selected mesh must remain renderable"; return false;
    }
    return true;
}

template<class ReadPixels>
bool verify(arbitgpu::NativeDeformationBackend& backend, ReadPixels readPixels, std::string& error,
            bool rigidOnly = false, bool repeated = false) {
    videohelper::modelpayload::Store store;
    visualanimationimport::ExactContentAssetKey key;
    if (!publish(store, key, rigidOnly, repeated)) { error = "multi-object fixture publication failed"; return false; }
    videohelper::modelpayload::ImportedAnimatedSceneCompatibility choices;
    if (!videohelper::modelpayload::ImportedAnimatedScenePayloadExecution::inspectCompatibility(
            store.resolvePreview(key), 0, choices, error)) return false;
    if (choices.clips.size() != 1 || choices.clips[0].compatibleMeshStableIds
        != (repeated ? std::vector<visualanimationimport::StableId>{1}
                     : std::vector<visualanimationimport::StableId>{1, 2})) {
        error = "both meshes must expose their shared-parent animation clip"; return false;
    }
    videohelper::modelpayload::ImportedAnimatedSceneRequest request;
    request.operation.sourceStableId = 1; request.operation.deformationStableId = 2;
    request.operation.schedule = {1, 2}; request.operation.asset = key; request.operation.sceneIndex = 0;
    request.operation.animationClipStableId = 1; request.operation.clipName = std::string(visualstartermodel::kClipName);
    request.structuralRevision = 7; request.width = 96; request.height = 96; request.frame = {0, 24, 1};
    videohelper::modelpayload::ImportedAnimatedScenePayloadExecution preview(store, backend), exported(store, backend);
    videohelper::modelpayload::ImportedAnimatedSceneReceipt start, moved, reset, output;
    if (!preview.executePreview(request, start, error)) return false;
    request.frame = {12, 24, 1}; request.operation.playback.timelineSeconds = 0.5;
    if (!preview.executePreview(request, moved, error) || !exported.executeExport(request, output, error)) return false;
    request.frame = {0, 24, 1}; request.operation.playback.timelineSeconds = 0;
    if (!preview.executePreview(request, reset, error)) return false;
    if (start.source != moved.source || start.source != reset.source || start.source->draws.size() != 2
        || start.source->scene->objectCount != (repeated ? 2u : 3u) || start.source->scene->materialCount != 2
        || moved.frame.stats.dispatchCount != 2 || moved.frame.stats.drawCount != (repeated ? 2u : 3u)) {
        error = "multi-object rendering must retain two compute ranges and three material draws"; return false;
    }
    const auto first = readPixels(start.frame.nativeFrame), next = readPixels(moved.frame.nativeFrame);
    if (first.empty() || next.empty() || first == next || first != readPixels(reset.frame.nativeFrame)
        || next != readPixels(output.frame.nativeFrame)) {
        error = "multi-object pixels must move, rewind exactly and match preview/export"; return false;
    }
    if (repeated) {
        if (start.source->draws[0]->mesh != start.source->draws[1]->mesh
            || start.source->draws[0]->deformation != start.source->draws[1]->deformation
            || start.source->draws[0]->object == start.source->draws[1]->object) {
            error = "repeated nodes must share immutable mesh data and retain distinct object IDs"; return false;
        }
        const auto baselineEvaluation = moved.evaluation.deformation;
        request.frame = {12, 24, 1}; request.operation.playback.timelineSeconds = 0.5;
        request.operation.pose.nodeStableId = 5; request.operation.pose.meshStableId = 1;
        request.operation.pose.boneEnabled = true; request.operation.pose.boneStableId = 5;
        request.operation.pose.translation[2] = 0.2;
        request.runtimeInputs.objectTranslationOffset[0] = 0.15f;
        if (!rigidOnly) {
            request.operation.pose.morphEnabled = true;
            request.operation.pose.morphTargetStableId = 1;
            request.operation.pose.morphTargetIndex = 0; request.operation.pose.morphWeight = 0.8;
        }
        if (!preview.executePreview(request, moved, error) || !exported.executeExport(request, output, error)) return false;
        if (readPixels(moved.frame.nativeFrame) == next
            || readPixels(moved.frame.nativeFrame) != readPixels(output.frame.nativeFrame)
            || moved.runtimeInputs.objectNodeStableId != 5) {
            error = "whole-scene exact object pose must change preview and export identically"; return false;
        }
        arbitgpu::NativeDeformationFrameData untouched, posed, before;
        if (!arbitgpu::prepareNativeDeformationFrame(*moved.source->draws[0], *moved.evaluation.deformation, untouched, error)
            || !arbitgpu::prepareNativeDeformationFrame(*moved.source->draws[1], *moved.evaluation.deformation, posed, error)
            || !arbitgpu::prepareNativeDeformationFrame(*output.source->draws[0], *baselineEvaluation, before, error)) return false;
        if (untouched.jointPalette != before.jointPalette || untouched.morphWeights != before.morphWeights
            || (!rigidOnly && (std::abs(posed.morphWeights[0] - 0.8f) > 0.0001f
                || untouched.morphWeights[0] == posed.morphWeights[0]))
            || arbitgpu::runtimeTargetsObject(moved.runtimeInputs, moved.source->scene->objects[0])
            || !arbitgpu::runtimeTargetsObject(moved.runtimeInputs, moved.source->scene->objects[1])) {
            error = "selected pose controls must not affect the other node sharing its mesh"; return false;
        }
        request.operation.pose = {}; request.runtimeInputs.objectTranslationOffset = {};
        request.frame = {0, 24, 1}; request.operation.playback.timelineSeconds = 0;
        if (!preview.executePreview(request, reset, error) || readPixels(reset.frame.nativeFrame) != first) {
            error = "repeated object pose must rewind without leaking its prior frame"; return false;
        }
        return true;
    }
    for (visualanimationimport::StableId mesh : {1u, 2u}) {
        request.operation.meshStableId = mesh;
        request.frame = {12, 24, 1}; request.operation.playback.timelineSeconds = 0.5;
        if (!preview.executePreview(request, moved, error)
            || !exported.executeExport(request, output, error)) return false;
        if (moved.source->scene->objectCount != mesh || moved.frame.stats.dispatchCount != 1
            || moved.frame.stats.drawCount != mesh
            || readPixels(moved.frame.nativeFrame) != readPixels(output.frame.nativeFrame)) {
            error = "selected mesh animation must retain its primitive draws and export pixels"; return false;
        }
    }
    return true;
}
} // namespace multiobjectanimationfixture
