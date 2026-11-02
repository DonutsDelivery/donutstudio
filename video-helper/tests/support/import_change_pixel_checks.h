#pragma once

#include "../../src/imported_scene_payload_execution.h"
#include "../../src/imported_scene_visual_plan_execution.h"
#include "../../src/surface_material_admission.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>
#include <set>
#include <string>
#include <vector>

// A changed source file, local transform or material must change the Color bytes
// of a real native frame on the static import route. Each case first repeats its
// request, which must give the same bytes, so a difference means the change did it.
namespace importchangepixels
{
using namespace videohelper::modelpayload;
using Bytes = std::vector<std::uint8_t>;

inline void appendU32(Bytes& bytes, std::uint32_t value)
{
    for (int shift = 0; shift < 32; shift += 8) bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

inline Bytes triangleBin()
{
    Bytes bin;
    const auto appendFloat = [&](float value) {
        std::uint32_t encoded = 0;
        std::memcpy(&encoded, &value, sizeof(encoded));
        appendU32(bin, encoded);
    };
    for (const auto value : {-0.5f, -0.5f, 0.0f, 0.5f, -0.5f, 0.0f, 0.0f, 0.5f, 0.0f}) appendFloat(value);
    for (int vertex = 0; vertex < 3; ++vertex)
        for (const auto value : {0.0f, 0.0f, 1.0f}) appendFloat(value);
    for (const auto value : {0.0f, 0.0f, 1.0f, 0.0f, 0.5f, 1.0f}) appendFloat(value);
    for (const std::uint8_t value : {0, 0, 1, 0, 2, 0, 0, 0}) bin.push_back(value);
    return bin;
}

inline nlohmann::json triangleRoot()
{
    using array = nlohmann::json;
    return {
        {"asset", {{"version", "2.0"}}},
        {"scene", 0},
        {"scenes", array::array({{{"nodes", array::array({0, 1, 2})}}})},
        {"nodes", array::array({
            {{"name", "Triangle"}, {"mesh", 0}},
            {{"name", "Camera"}, {"camera", 0}, {"translation", array::array({0.0, 0.0, 2.0})}},
            {{"name", "Key"}, {"extensions", {{"KHR_lights_punctual", {{"light", 0}}}}}}})},
        {"meshes", array::array({{{"primitives", array::array({
            {{"attributes", {{"POSITION", 0}, {"NORMAL", 1}, {"TEXCOORD_0", 2}}}, {"indices", 3}, {"material", 0}}})}}})},
        {"materials", array::array({{{"pbrMetallicRoughness", {
            {"baseColorFactor", array::array({0.2, 0.4, 0.8, 1.0})}, {"metallicFactor", 0.1}, {"roughnessFactor", 0.7}}}}})},
        {"cameras", array::array({{{"type", "perspective"}, {"perspective", {{"yfov", 1.0}, {"znear", 0.1}, {"zfar", 20.0}}}}})},
        {"extensionsUsed", array::array({"KHR_lights_punctual"})},
        {"extensionsRequired", array::array({"KHR_lights_punctual"})},
        {"extensions", {{"KHR_lights_punctual", {{"lights", array::array({{{"type", "directional"}, {"intensity", 2.0}}})}}}}},
        {"buffers", array::array({{{"byteLength", 104}}})},
        {"bufferViews", array::array({
            {{"buffer", 0}, {"byteOffset", 0}, {"byteLength", 36}, {"target", 34962}},
            {{"buffer", 0}, {"byteOffset", 36}, {"byteLength", 36}, {"target", 34962}},
            {{"buffer", 0}, {"byteOffset", 72}, {"byteLength", 24}, {"target", 34962}},
            {{"buffer", 0}, {"byteOffset", 96}, {"byteLength", 6}, {"target", 34963}}})},
        {"accessors", array::array({
            {{"bufferView", 0}, {"componentType", 5126}, {"count", 3}, {"type", "VEC3"},
             {"min", array::array({-0.5, -0.5, 0.0})}, {"max", array::array({0.5, 0.5, 0.0})}},
            {{"bufferView", 1}, {"componentType", 5126}, {"count", 3}, {"type", "VEC3"}},
            {{"bufferView", 2}, {"componentType", 5126}, {"count", 3}, {"type", "VEC2"}},
            {{"bufferView", 3}, {"componentType", 5123}, {"count", 3}, {"type", "SCALAR"}}})}
    };
}

inline Bytes makeGlb(const nlohmann::json& root)
{
    auto text = root.dump();
    while ((text.size() & 3u) != 0) text.push_back(' ');
    const auto bin = triangleBin();
    Bytes bytes;
    appendU32(bytes, 0x46546c67u);
    appendU32(bytes, 2u);
    appendU32(bytes, static_cast<std::uint32_t>(12u + 8u + text.size() + 8u + bin.size()));
    appendU32(bytes, static_cast<std::uint32_t>(text.size()));
    appendU32(bytes, 0x4e4f534au);
    bytes.insert(bytes.end(), text.begin(), text.end());
    appendU32(bytes, static_cast<std::uint32_t>(bin.size()));
    appendU32(bytes, 0x004e4942u);
    bytes.insert(bytes.end(), bin.begin(), bin.end());
    return bytes;
}

inline std::string encodeBase64(const Bytes& bytes)
{
    static constexpr char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string encoded;
    for (std::size_t offset = 0; offset < bytes.size(); offset += 3u)
    {
        const auto remaining = bytes.size() - offset;
        const auto value = static_cast<std::uint32_t>(bytes[offset]) << 16u
            | (remaining > 1u ? static_cast<std::uint32_t>(bytes[offset + 1u]) << 8u : 0u)
            | (remaining > 2u ? static_cast<std::uint32_t>(bytes[offset + 2u]) : 0u);
        encoded.push_back(alphabet[(value >> 18u) & 63u]);
        encoded.push_back(alphabet[(value >> 12u) & 63u]);
        encoded.push_back(remaining > 1u ? alphabet[(value >> 6u) & 63u] : '=');
        encoded.push_back(remaining > 2u ? alphabet[value & 63u] : '=');
    }
    return encoded;
}

inline surfacematerial::ProgramDescription materialProgram(const HarmonicMIDI::grid::Visual3DScene& scene,
    const std::array<float, 4>& baseColor)
{
    surfacematerial::ProgramDescription program;
    const auto appendConstant = [&](surfacematerial::ValueType type, std::array<float, 4> literal) {
        surfacematerial::Operation operation;
        operation.id = static_cast<surfacematerial::ValueId>(program.operations.size() + 1u);
        operation.kind = type == surfacematerial::ValueType::UInt
            ? surfacematerial::OperationKind::UIntConstant : surfacematerial::OperationKind::FloatConstant;
        operation.resultType = type;
        operation.literal = literal;
        program.operations.push_back(operation);
        return operation.id;
    };
    for (const auto output : surfacematerial::kSurfaceOutputs)
    {
        const auto literal = output == surfacematerial::OutputSemantic::BaseColor ? baseColor
            : output == surfacematerial::OutputSemantic::Normal ? std::array<float, 4>{0.0f, 0.0f, 1.0f, 0.0f}
            : output == surfacematerial::OutputSemantic::Opacity ? std::array<float, 4>{1.0f, 0.0f, 0.0f, 0.0f}
            : output == surfacematerial::OutputSemantic::Ior ? std::array<float, 4>{1.5f, 0.0f, 0.0f, 0.0f}
            : std::array<float, 4>{};
        program.outputs[static_cast<std::size_t>(output)] = appendConstant(surfacematerial::outputType(output), literal);
    }
    program.operations.back().unsignedLiteral = scene.objects[0].material.value;
    return program;
}

inline bool run(std::string& error)
{
    const auto reject = [&](const std::string& message) { error = message; return false; };
    Store store;
    const auto publish = [&](const char* id, const Bytes& bytes) {
        const visualanimationimport::ExactContentAssetKey key {id, 1u, detail::digestHex(bytes), "model/gltf-binary", bytes.size()};
        const auto transfer = std::string(id) + "-transfer";
        return store.begin(transfer.c_str(), key) && store.appendBase64(transfer.c_str(), 0u, encodeBase64(bytes))
            && store.commit(transfer.c_str()) ? std::optional(key) : std::nullopt;
    };
    const auto identity = [](const ImportedSceneRequest& selected) {
        visualimportedscenerender::Request operation;
        operation.sourceStableId = 1; operation.renderStableId = 2;
        operation.asset = selected.asset; operation.sceneIndex = selected.sceneIndex;
        videowire::CompiledVisualLayerPlan plan;
        plan.operations.push_back({1, std::string(visualimportedscenerender::kRenderNodeKind),
            std::string(visualimportedscenerender::kRenderBackendCapability), visualimportedscenerender::encode(operation)});
        return videohelper::importedscene::exactImportedScenePayloadIdentity(plan);
    };
    auto movedRoot = triangleRoot();
    movedRoot["nodes"][0]["translation"] = {0.5, 0.0, 0.0};
    const auto sourceKey = publish("import-change-source", makeGlb(triangleRoot()));
    const auto movedKey = publish("import-change-moved", makeGlb(movedRoot));
    if (!sourceKey || !movedKey) return reject("The import change fixtures did not publish");

    ImportedSceneRequest base;
    base.asset = *sourceKey; base.sceneIndex = 0; base.sourceStableId = 83; base.renderStableId = 84;
    base.structuralRevision = base.evaluationRevision = base.projectGeneration = base.helperGeneration = 1;
    base.dimensions = {64, 36}; base.frame = {0, 24, 1};
    base.staticPayloadIdentity = identity(base);

    ImportedScenePayloadExecution execution(store, arbitgpu::nativeFixtureSceneBackend());
    ImportedSceneExecutionReceipt receipt;
    // Export never returns pending, so each call gives a finished frame.
    const auto color = [&](const ImportedSceneRequest& request, Bytes& bytes) {
        arbitgpu::NativeRawPassPixels pixels;
        if (!execution.executeExport(request, receipt, error) || !receipt.frame.rendered.nativeFrame
            || !receipt.frame.rendered.nativeFrame->readRawPass(renderpassoutput::Output::Color, pixels, error)) return false;
        bytes = std::move(pixels.bytes);
        return true;
    };
    // The frame must hold a drawn image, or two equal blank frames would pass the control.
    const auto drawn = [](const Bytes& bytes) {
        std::set<std::uint32_t> values;
        for (std::size_t offset = 0; offset + 4 <= bytes.size(); offset += 4)
        {
            std::uint32_t value = 0;
            std::memcpy(&value, bytes.data() + offset, sizeof(value));
            values.insert(value);
        }
        return values.size() > 1;
    };
    const auto differing = [](const Bytes& left, const Bytes& right) {
        std::size_t count = 0;
        for (std::size_t index = 0; index < std::min(left.size(), right.size()); ++index) count += left[index] != right[index];
        return count;
    };
    const auto baseline = [&](const ImportedSceneRequest& request, const char* name, Bytes& bytes) {
        Bytes repeat;
        if (!color(request, bytes) || !color(request, repeat)) return reject(std::string(name) + " did not draw: " + error);
        if (!drawn(bytes)) return reject(std::string(name) + " drew a blank frame");
        if (bytes != repeat) return reject(std::string(name) + " repeat drew different bytes");
        return true;
    };
    const auto changed = [&](const ImportedSceneRequest& request, const char* name, const Bytes& before) {
        Bytes after;
        if (!color(request, after)) return reject(std::string(name) + " did not draw: " + error);
        const auto count = differing(before, after);
        std::cout << "import change " << name << ": " << count << " of " << before.size() << " Color bytes differ\n";
        return count > 0 || reject(std::string(name) + " left the Color bytes unchanged");
    };

    auto source = base; source.clipId = 201;
    Bytes before;
    if (!baseline(source, "source file", before)) return false;
    auto moved = source;
    moved.asset = *movedKey; moved.staticPayloadIdentity = identity(moved);
    if (!changed(moved, "source file", before)) return false;

    auto transform = base; transform.clipId = 202;
    if (!baseline(transform, "local transform", before)) return false;
    transform.objectLocalTransform.version = 1;
    transform.objectLocalTransform.nodeStableId = 1;
    transform.objectLocalTransform.meshStableId = 1;
    transform.objectLocalTransform.translation[0] = 0.5;
    if (!changed(transform, "local transform", before)) return false;

    auto material = base; material.clipId = 203;
    if (!color(material, before)) return reject("The material fixture did not draw: " + error);
    const auto scene = receipt.admission.scene;
    const auto request = [&](const std::array<float, 4>& baseColor, std::uint64_t revision) {
        const auto program = materialProgram(*scene, baseColor);
        const auto admitted = surfacematerial::admit(program, error);
        surfacematerialbinding::ImportedSceneMaterialRequest value;
        value.scene = scene->id;
        value.sceneRevision = value.evaluationRevision = 1;
        value.structuralRevision = value.programRevision = revision;
        value.program = program;
        value.binding.targetKind = surfacematerialbinding::BindingTargetKind::ObjectOverride;
        value.binding.object = scene->objects[0].id;
        value.binding.surfaceMaterialDigest = admitted ? admitted->structuralDigest() : std::string(64, '0');
        value.binding.surfaceMaterialRevision = revision;
        return value;
    };
    material.material = request({0.8f, 0.15f, 0.05f, 0.0f}, 1);
    if (!baseline(material, "material", before)) return false;
    if (receipt.usedLastGoodMaterial) return reject("The material fixture fell back: " + receipt.materialDiagnostic);
    material.material = request({0.05f, 0.15f, 0.8f, 0.0f}, 2);
    if (!changed(material, "material", before)) return false;
    if (receipt.usedLastGoodMaterial) return reject("The changed material fell back: " + receipt.materialDiagnostic);
    return true;
}
} // namespace importchangepixels
