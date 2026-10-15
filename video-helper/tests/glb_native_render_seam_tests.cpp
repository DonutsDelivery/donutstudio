#include "../src/imported_scene_payload_execution.h"
#include "../src/imported_scene_visual_plan_execution.h"
#include "../src/imported_animated_scene_payload_execution.h"
#include "../../shared/DiffractionMaterialPresets.h"
#include "../src/surface_material_admission.h"
#include "../../shared/VisualStarterModelAssets.h"
#include "reactive_surface_fixture.h"
#include "mixed_surface_diffraction_fixture.h"
#include "support/smooth_glass_authored_material.h"
#include "support/scene_closest_hit_cases.h"
#include "support/vertex_modifier_fixture.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#if !defined(_WIN32)
#include <pthread.h>
#endif

namespace videohelper::modelpayload
{
// Reads composition leaf admission state that execute() does not route yet.
struct ImportedAnimatedSceneCompositionProbe final
{
    using Execution = ImportedAnimatedScenePayloadExecution;
    using Input = Execution::CompositionLeafInput;
    static bool admit(Execution& execution, const std::vector<Input>& inputs, std::string& error)
    { return execution.admitCompositionLeaves(inputs, error); }
    static std::size_t leaves(const Execution& execution) noexcept { return execution.compositionLeaves_.size(); }
    static bool animated(const Execution& execution, std::size_t leaf) noexcept
    { return execution.compositionLeaves_[leaf].animated; }
    static std::size_t bindings(const Execution& execution, std::size_t leaf) noexcept
    { return execution.compositionLeaves_[leaf].bindings.size(); }
    static bool singleActive(const Execution& execution) noexcept
    { return execution.scene_ || execution.baseScene_ || execution.admittedIdentity_.payload || !execution.bindings_.empty(); }
    static const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene(const Execution& execution) noexcept
    { return execution.scene_; }
    static bool ledger(const Execution& execution) noexcept { return static_cast<bool>(execution.sceneAnimationBudget_); }
    static std::size_t singleBindings(const Execution& execution) noexcept { return execution.bindings_.size(); }
    // The single leaf's binding copy reservation; zero when absent.
    static std::size_t singleCopyBytes(const Execution& execution) noexcept
    { return execution.bindingCopyCharge_ ? execution.bindingCopyCharge_->bytes : 0; }
    static std::size_t ledgerBytes(const Execution& execution) noexcept
    { return execution.sceneAnimationBudget_ ? execution.sceneAnimationBudget_->logicalBytes() : 0; }
    // Sum of every leaf's reservations; zero when a fixed or closure
    // reservation is missing. Only a leaf with bindings has a binding copy.
    static std::size_t leafChargeBytes(const Execution& execution) noexcept
    {
        std::size_t bytes = 0;
        for (const auto& leaf : execution.compositionLeaves_)
        {
            if (!leaf.charges[0] || !leaf.charges[1] || leaf.bindings.empty() == static_cast<bool>(leaf.charges[2]))
                return 0;
            for (const auto& charge : leaf.charges)
                if (charge) bytes += charge->bytes;
        }
        return bytes;
    }
    // One leaf reservation: 0 fixed, 1 closure, 2 binding copy; zero when absent.
    static std::size_t chargeBytes(const Execution& execution, std::size_t leaf, std::size_t index) noexcept
    {
        const auto& charge = execution.compositionLeaves_[leaf].charges[index];
        return charge ? charge->bytes : 0;
    }
    static std::size_t leafRecordBytes() noexcept { return sizeof(Execution::CompositionLeaf); }
    static std::shared_ptr<NativeSceneAnimationCompositionBudget>& animationLedger(Execution& execution) noexcept
    { return execution.sceneAnimationBudget_; }
    static std::shared_ptr<NativeCompositionRetentionBudget>& compositionLedger(Execution& execution) noexcept
    { return execution.compositionBudget_; }
};
} // namespace videohelper::modelpayload

namespace
{
int failures = 0;
int checks = 0;

void check (bool value, const char* message)
{
    ++checks;
    if (! value)
    {
        ++failures;
        std::fprintf (stderr, "FAIL: %s\n", message);
    }
}

void appendU16 (std::vector<std::uint8_t>& bytes, std::uint16_t value)
{
    bytes.push_back (static_cast<std::uint8_t> (value));
    bytes.push_back (static_cast<std::uint8_t> (value >> 8u));
}

void appendU32 (std::vector<std::uint8_t>& bytes, std::uint32_t value)
{
    bytes.push_back (static_cast<std::uint8_t> (value));
    bytes.push_back (static_cast<std::uint8_t> (value >> 8u));
    bytes.push_back (static_cast<std::uint8_t> (value >> 16u));
    bytes.push_back (static_cast<std::uint8_t> (value >> 24u));
}

std::uint32_t readU32 (const std::uint8_t* bytes)
{
    return static_cast<std::uint32_t> (bytes[0])
        | static_cast<std::uint32_t> (bytes[1]) << 8u
        | static_cast<std::uint32_t> (bytes[2]) << 16u
        | static_cast<std::uint32_t> (bytes[3]) << 24u;
}

void appendFloat (std::vector<std::uint8_t>& bytes, float value)
{
    std::uint32_t encoded = 0;
    std::memcpy (&encoded, &value, sizeof (encoded));
    appendU32 (bytes, encoded);
}

std::vector<std::uint8_t> makeGlb (nlohmann::json root,
                                   std::vector<std::uint8_t> bin)
{
    auto text = root.dump();
    while ((text.size() & 3u) != 0)
        text.push_back (' ');
    while ((bin.size() & 3u) != 0)
        bin.push_back (0);

    std::vector<std::uint8_t> bytes;
    const auto total = 12u + 8u + static_cast<unsigned> (text.size())
                     + 8u + static_cast<unsigned> (bin.size());
    appendU32 (bytes, 0x46546c67u);
    appendU32 (bytes, 2u);
    appendU32 (bytes, total);
    appendU32 (bytes, static_cast<std::uint32_t> (text.size()));
    appendU32 (bytes, 0x4e4f534au);
    bytes.insert (bytes.end(), text.begin(), text.end());
    appendU32 (bytes, static_cast<std::uint32_t> (bin.size()));
    appendU32 (bytes, 0x004e4942u);
    bytes.insert (bytes.end(), bin.begin(), bin.end());
    return bytes;
}

template <std::size_t Size>
std::vector<std::uint8_t> withoutStarterLightBinding (
    const std::array<std::uint8_t, Size>& bytes)
{
    const auto jsonSize = readU32 (bytes.data() + 12u);
    auto root = nlohmann::json::parse (
        bytes.begin() + 20u, bytes.begin() + 20u + jsonSize);
    root["nodes"][1].erase ("extensions");
    const auto binHeader = 20u + jsonSize;
    const auto binSize = readU32 (bytes.data() + binHeader);
    std::vector<std::uint8_t> bin (
        bytes.begin() + binHeader + 8u,
        bytes.begin() + binHeader + 8u + binSize);
    return makeGlb (std::move (root), std::move (bin));
}

std::vector<std::uint8_t> nestedAnimatedStarter()
{
    const auto& bytes = visualstartermodel::kAnimatedTriangleGlb;
    const auto jsonSize = readU32(bytes.data() + 12u);
    auto root = nlohmann::json::parse(
        bytes.begin() + 20u, bytes.begin() + 20u + jsonSize);
    auto& nodes = root["nodes"];
    const auto meshNodeIndex = nodes.size();
    auto mesh = nodes[0];
    mesh["translation"] = {0.1, 0.0, 0.0};
    mesh["scale"] = {0.75, 0.5, 1.0};
    nodes.push_back(std::move(mesh));
    const auto parentIndex = nodes.size();
    nodes.push_back({{"children", {meshNodeIndex}},
                     {"translation", {0.4, 0.2, 0.0}}, {"scale", {2.0, 2.0, 2.0}}});
    nodes[0] = {{"children", {parentIndex}}, {"translation", {0.2, -0.1, 0.0}},
                {"rotation", {0.0, 0.0, std::sqrt(0.5), std::sqrt(0.5)}},
                {"scale", {0.5, 0.5, 0.5}}};
    for (auto& animation : root["animations"])
        for (auto& channel : animation["channels"])
            if (channel["target"]["node"] == 0)
                channel["target"]["node"] = meshNodeIndex;
    const auto binHeader = 20u + jsonSize;
    const auto binSize = readU32(bytes.data() + binHeader);
    return makeGlb(std::move(root), {bytes.begin() + binHeader + 8u,
                                   bytes.begin() + binHeader + 8u + binSize});
}

std::vector<std::uint8_t> starterWithTransformTrack(bool meshTarget, bool keepMorph = true)
{
    const auto& bytes = visualstartermodel::kAnimatedTriangleGlb;
    const auto jsonSize = readU32(bytes.data() + 12u);
    auto root = nlohmann::json::parse(bytes.begin() + 20u, bytes.begin() + 20u + jsonSize);
    const auto binHeader = 20u + jsonSize;
    const auto binSize = readU32(bytes.data() + binHeader);
    std::vector<std::uint8_t> bin(bytes.begin() + binHeader + 8u, bytes.begin() + binHeader + 8u + binSize);
    const auto input = root["animations"][0]["samplers"][0]["input"].get<std::size_t>();
    const auto count = root["accessors"][input]["count"].get<std::size_t>();
    const auto offset = bin.size();
    for (std::size_t key = 0; key < count; ++key)
        for (const auto value : {static_cast<float>(key) * .25f, 0.0f, 0.0f}) appendFloat(bin, value);
    const auto view = root["bufferViews"].size();
    root["bufferViews"].push_back({{"buffer",0}, {"byteOffset",offset}, {"byteLength",count * 12u}});
    const auto output = root["accessors"].size();
    root["accessors"].push_back({{"bufferView",view}, {"componentType",5126}, {"count",count}, {"type","VEC3"}});
    auto& animation = root["animations"][0];
    const auto sampler = animation["samplers"].size();
    animation["samplers"].push_back({{"input",input}, {"output",output}, {"interpolation","LINEAR"}});
    animation["channels"].push_back({{"sampler",sampler}, {"target",{{"node",meshTarget ? 0 : 1}, {"path","translation"}}}});
    if (!keepMorph) animation["channels"].erase(animation["channels"].begin());
    if (meshTarget)
    {
        root["nodes"][0]["children"] = {2};
        root["scenes"][0]["nodes"] = {0,1};
    }
    root["buffers"][0]["byteLength"] = bin.size();
    return makeGlb(std::move(root), std::move(bin));
}

std::string encodeBase64 (const std::vector<std::uint8_t>& bytes)
{
    static constexpr char alphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string encoded;
    encoded.reserve ((bytes.size() + 2u) / 3u * 4u);
    for (std::size_t offset = 0; offset < bytes.size(); offset += 3u)
    {
        const auto remaining = bytes.size() - offset;
        const auto value = static_cast<std::uint32_t> (bytes[offset]) << 16u
            | (remaining > 1u ? static_cast<std::uint32_t> (bytes[offset + 1u]) << 8u : 0u)
            | (remaining > 2u ? static_cast<std::uint32_t> (bytes[offset + 2u]) : 0u);
        encoded.push_back (alphabet[(value >> 18u) & 63u]);
        encoded.push_back (alphabet[(value >> 12u) & 63u]);
        encoded.push_back (remaining > 1u ? alphabet[(value >> 6u) & 63u] : '=');
        encoded.push_back (remaining > 2u ? alphabet[value & 63u] : '=');
    }
    return encoded;
}

visualanimationimport::ExactContentAssetKey exactKey (
    const char* id, const std::vector<std::uint8_t>& bytes)
{
    return {id, 1u, videohelper::modelpayload::detail::digestHex (bytes),
            "model/gltf-binary", bytes.size()};
}

bool publish (videohelper::modelpayload::Store& store,
              const char* transferId,
              const visualanimationimport::ExactContentAssetKey& key,
              const std::vector<std::uint8_t>& bytes)
{
    return store.begin (transferId, key)
        && store.appendBase64 (transferId, 0u, encodeBase64 (bytes))
        && store.commit (transferId);
}

std::vector<std::uint8_t> triangleBin()
{
    std::vector<std::uint8_t> bin;
    for (const auto value : {-0.5f, -0.5f, 0.0f,
                              0.5f, -0.5f, 0.0f,
                              0.0f,  0.5f, 0.0f})
        appendFloat (bin, value);
    for (int vertex = 0; vertex < 3; ++vertex)
        for (const auto value : {0.0f, 0.0f, 1.0f})
            appendFloat (bin, value);
    for (const auto value : {0.0f, 0.0f, 1.0f, 0.0f, 0.5f, 1.0f})
        appendFloat (bin, value);
    appendU16 (bin, 0);
    appendU16 (bin, 1);
    appendU16 (bin, 2);
    return bin;
}

const std::vector<std::uint8_t>& opaquePngImageBytes()
{
    static const std::vector<std::uint8_t> bytes {
        137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82,
        0, 0, 0, 1, 0, 0, 0, 1, 8, 6, 0, 0, 0, 31, 21, 196, 137,
        0, 0, 0, 13, 73, 68, 65, 84, 120, 156, 99, 248, 239, 160, 240,
        31, 0, 6, 0, 2, 95, 52, 205, 64, 192, 0, 0, 0, 0, 73, 69,
        78, 68, 174, 66, 96, 130
    };
    return bytes;
}

std::vector<std::uint8_t> texturedTriangleBin()
{
    auto bin = triangleBin();
    bin.resize (104, 0);
    const auto& image = opaquePngImageBytes();
    bin.insert (bin.end(), image.begin(), image.end());
    return bin;
}

nlohmann::json triangleRoot()
{
    return {
        {"asset", {{"version", "2.0"}, {"generator", "native seam fixture"}}},
        {"scene", 0},
        {"scenes", nlohmann::json::array ({
            {{"name", "Primary"}, {"nodes", nlohmann::json::array ({0, 1, 2})}},
            {{"name", "Alternate"}, {"nodes", nlohmann::json::array ({0, 1, 2})}}
        })},
        {"nodes", nlohmann::json::array ({
            {{"name", "Triangle"}, {"mesh", 0}},
            {{"name", "Camera"}, {"camera", 0},
             {"translation", nlohmann::json::array ({0.0, 0.0, 2.0})}},
            {{"name", "Key"},
             {"extensions", {{"KHR_lights_punctual", {{"light", 0}}}}}}
        })},
        {"meshes", nlohmann::json::array ({
            {{"primitives", nlohmann::json::array ({
                {{"attributes", {{"POSITION", 0}, {"NORMAL", 1}, {"TEXCOORD_0", 2}}},
                 {"indices", 3}, {"material", 0}}
            })}}
        })},
        {"materials", nlohmann::json::array ({
            {{"pbrMetallicRoughness", {
                {"baseColorFactor", nlohmann::json::array ({0.2, 0.4, 0.8, 1.0})},
                {"metallicFactor", 0.1}, {"roughnessFactor", 0.7}}}}
        })},
        {"cameras", nlohmann::json::array ({
            {{"type", "perspective"}, {"perspective", {
                {"yfov", 1.0}, {"znear", 0.1}, {"zfar", 20.0}}}}
        })},
        {"extensionsUsed", nlohmann::json::array ({"KHR_lights_punctual"})},
        {"extensionsRequired", nlohmann::json::array ({"KHR_lights_punctual"})},
        {"extensions", {{"KHR_lights_punctual", {
            {"lights", nlohmann::json::array ({
                {{"type", "directional"}, {"intensity", 2.0}}
            })}}}}},
        {"buffers", nlohmann::json::array ({
            {{"byteLength", 102}}
        })},
        {"bufferViews", nlohmann::json::array ({
            {{"buffer", 0}, {"byteOffset", 0}, {"byteLength", 36}, {"target", 34962}},
            {{"buffer", 0}, {"byteOffset", 36}, {"byteLength", 36}, {"target", 34962}},
            {{"buffer", 0}, {"byteOffset", 72}, {"byteLength", 24}, {"target", 34962}},
            {{"buffer", 0}, {"byteOffset", 96}, {"byteLength", 6}, {"target", 34963}}
        })},
        {"accessors", nlohmann::json::array ({
            {{"bufferView", 0}, {"componentType", 5126}, {"count", 3}, {"type", "VEC3"}},
            {{"bufferView", 1}, {"componentType", 5126}, {"count", 3}, {"type", "VEC3"}},
            {{"bufferView", 2}, {"componentType", 5126}, {"count", 3}, {"type", "VEC2"}},
            {{"bufferView", 3}, {"componentType", 5123}, {"count", 3}, {"type", "SCALAR"}}
        })}
    };
}

std::vector<std::uint8_t> importedObjectMotionFixture(bool parentTarget, bool moving)
{
    auto root = triangleRoot();
    auto bin = triangleBin();
    bin.resize(104, 0);
    for (const auto time : {0.0f, 1.0f}) appendFloat(bin, time);
    const std::array<float, 3> first = parentTarget
        ? std::array<float, 3>{0.1f, 0.2f, 0.0f}
        : std::array<float, 3>{0.25f, -0.125f, 0.0f};
    for (int key = 0; key < 2; ++key)
        for (std::size_t axis = 0; axis < 3; ++axis)
            appendFloat(bin, first[axis] + (moving && axis == 0 ? 0.5f * key : 0.0f));
    root["buffers"][0]["byteLength"] = bin.size();
    root["bufferViews"].push_back({{"buffer",0}, {"byteOffset",104}, {"byteLength",8}});
    root["bufferViews"].push_back({{"buffer",0}, {"byteOffset",112}, {"byteLength",24}});
    root["accessors"].push_back({{"bufferView",4}, {"componentType",5126}, {"count",2}, {"type","SCALAR"}});
    root["accessors"].push_back({{"bufferView",5}, {"componentType",5126}, {"count",2}, {"type","VEC3"}});
    root["nodes"][0]["translation"] = {0.25, -0.125, 0.0};
    root["nodes"][0]["scale"] = {0.5, 0.75, 1.0};
    root["nodes"].push_back({{"name","Animated Parent"}, {"children",{0}},
        {"translation",{0.1,0.2,0.0}}, {"rotation",{0.0,0.0,std::sqrt(0.5),std::sqrt(0.5)}},
        {"scale",{2.0,2.0,1.0}}});
    for (auto& scene : root["scenes"]) scene["nodes"] = {3,1,2};
    root["nodes"][1]["translation"] = {0.0,0.0,4.0};
    root["cameras"][0]["perspective"]["yfov"] = 1.5707963267948966;
    root["animations"] = nlohmann::json::array({{
        {"name","Imported Object Motion"},
        {"samplers",nlohmann::json::array({{{"input",4}, {"output",5}, {"interpolation","LINEAR"}}})},
        {"channels",nlohmann::json::array({{{"sampler",0},
            {"target",{{"node",parentTarget ? 3 : 0}, {"path","translation"}}}}})}
    }});
    return makeGlb(std::move(root), std::move(bin));
}

const std::array<float,16>* importedWorld(
    const arbitgpu::NativeSceneMotionSample::ImportedNodeWorldMatrices& worlds, std::uint64_t id)
{
    for (const auto& entry : worlds)
        if (entry.nodeStableId == id) return &entry.world;
    return nullptr;
}

std::array<double,2> projectedImportedTriangleVertex(const std::array<float,16>& world,
    const std::array<double,3>& vertex, std::uint32_t width, std::uint32_t height)
{
    std::array<double,3> position {};
    for (std::size_t axis = 0; axis < 3; ++axis)
        position[axis] = world[axis] * vertex[0] + world[axis + 4] * vertex[1]
            + world[axis + 8] * vertex[2] + world[axis + 12];
    // The fixture camera is at z=4, with identity rotation and a 90 degree FOV.
    const double distance = 4.0 - position[2];
    const double aspect = static_cast<double>(width) / height;
    return {0.5 * width * position[0] / (distance * aspect),
            0.5 * height * position[1] / distance};
}

std::array<double,2> importedTrianglePixelOracle(bool parentTarget, bool moving, double seconds,
    const std::array<double,3>& vertex, std::uint32_t height)
{
    // Hand-composed parent T(0.1,0.2) Rz(90) S(2,2) and child
    // T(0.25,-0.125) S(0.5,0.75). No production matrix helper is used.
    const double x = 0.35 - 1.5 * vertex[1] + (parentTarget && moving ? 0.5 * seconds : 0.0);
    const double y = 0.7 + vertex[0] + (!parentTarget && moving ? seconds : 0.0);
    const double scale = 0.5 * height / (4.0 - vertex[2]);
    return {x * scale, y * scale};
}

nlohmann::json texturedTriangleRoot()
{
    auto root = triangleRoot();
    root["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]
        = {{"index", 0}};
    root["textures"] = nlohmann::json::array ({
        {{"source", 0}, {"sampler", 0}}
    });
    root["images"] = nlohmann::json::array ({
        {{"bufferView", 4}, {"mimeType", "image/png"}}
    });
    root["samplers"] = nlohmann::json::array ({
        {{"magFilter", 9728}, {"minFilter", 9728},
         {"wrapS", 10497}, {"wrapT", 10497}}
    });
    root["buffers"][0]["byteLength"] = 174;
    root["bufferViews"].push_back (
        {{"buffer", 0}, {"byteOffset", 104}, {"byteLength", 70}});
    return root;
}

class FakeFrame final : public arbitgpu::NativeFixtureSceneFrame
{
public:
    FakeFrame (std::string backend, std::uint32_t width, std::uint32_t height,
               std::uintptr_t handle) noexcept
        : backend_ (std::move (backend)), width_ (width), height_ (height), handle_ (handle)
    {
    }

    const std::string& backend() const noexcept override { return backend_; }
    std::uint32_t width() const noexcept override { return width_; }
    std::uint32_t height() const noexcept override { return height_; }
    std::uintptr_t colorImageHandle() const noexcept override { return handle_; }
    std::uintptr_t colorTextureViewHandle() const noexcept override { return handle_ + 1000; }
    std::uintptr_t depthImageHandle() const noexcept override { return handle_ + 2000; }
    std::uintptr_t depthTextureViewHandle() const noexcept override { return handle_ + 3000; }
    arbitgpu::NativeTextureViewDescriptor colorTextureDescriptor() const noexcept override
    {
        return { backend_, arbitgpu::NativeTextureViewKind::Texture2D,
            arbitgpu::NativeTexturePixelFormat::Rgba8Unorm, colorImageHandle(), colorTextureViewHandle(),
            width_, height_, 1, true, 77, 1 };
    }

private:
    std::string backend_;
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    std::uintptr_t handle_ = 0;
};

class FakeResources final : public arbitgpu::NativeFixtureSceneResources
{
public:
    explicit FakeResources (std::string backend) : backend_ (std::move (backend)) {}
    const std::string& backend() const noexcept override { return backend_; }
private:
    std::string backend_;
};

class FakeBackend final : public arbitgpu::NativeFixtureSceneBackend
{
public:
    arbitgpu::BackendInfo info() const override { return backendInfo; }
    bool supportsRetainedNoteMotion() const noexcept override { return retainedNoteMotion; }
    bool supportsAuthoredSpectralReceiverTransport() const noexcept override { return spectralReceiverTransport; }

    arbitgpu::NativeFixtureScenePreparation prepare (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
        const std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram>& material,
        arbitgpu::NativeFixturePrepareUse = arbitgpu::NativeFixturePrepareUse::Blocking) override
    {
        ++preparations;
        preparedProgram = material;
        preparedWithSurfaceMaterial = material != nullptr;
        preparedMaterialKind = material != nullptr
            ? material->kind : arbitgpu::NativeFixtureMaterialKind::SurfacePbr;
        arbitgpu::NativeFixtureScenePreparation result;
        result.prepared = true;
        result.resources = std::make_shared<FakeResources> (backendInfo.backend);
        result.stats.staticUploadCount = static_cast<std::uint32_t> (scene->objectCount);
        result.stats.textureUploadCount = static_cast<std::uint32_t> (scene->objectCount);
        result.stats.vertexBytes = scene->vertexCount * sizeof (scene->vertices[0]);
        result.stats.indexBytes = scene->indexCount * sizeof (scene->indices[0]);
        result.stats.textureBytes = scene->textureTexelCount > 0
            ? scene->textureTexelCount * sizeof (scene->textureTexels[0]) : 4;
        return result;
    }

    arbitgpu::NativeFixtureSceneSubmission render (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
        const std::shared_ptr<const arbitgpu::NativeFixtureSceneResources>&,
        std::uint32_t width,
        std::uint32_t height,
        arbitgpu::NativeFixtureSceneRuntimeInputs runtimeInputs) override
    {
        ++submissions;
        submittedScene = scene.get();
        lastRuntimeInputs = runtimeInputs;
        arbitgpu::NativeFixtureSceneSubmission result;
        if (rejectSubmission)
        {
            result.error = "fake draw failed";
            return result;
        }
        result.rendered = true;
        result.frame = std::make_shared<FakeFrame> (
            backendInfo.backend, width, height,
            static_cast<std::uintptr_t> (submissions));
        result.stats.drawCount = static_cast<std::uint32_t> (scene->objectCount);
        result.stats.materialBytes = sizeof (scene->materials[0]) * scene->objectCount;
        return result;
    }

    arbitgpu::BackendInfo backendInfo {true, false, "test-native", "fake", {}};
    bool rejectSubmission = false;
    bool retainedNoteMotion = false;
    bool preparedWithSurfaceMaterial = false;
    std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram> preparedProgram;
    arbitgpu::NativeFixtureMaterialKind preparedMaterialKind
        = arbitgpu::NativeFixtureMaterialKind::SurfacePbr;
    bool spectralReceiverTransport = false;
    int submissions = 0;
    int preparations = 0;
    const HarmonicMIDI::grid::Visual3DScene* submittedScene = nullptr;
    arbitgpu::NativeFixtureSceneRuntimeInputs lastRuntimeInputs;
};

class FakeDeformationResources final : public arbitgpu::NativeDeformationResources
{
public:
    const std::string& backend() const noexcept override { return backend_; }
private:
    std::string backend_ = "opengl";
};

class FakeDeformationBackend final : public arbitgpu::NativeDeformationBackend
{
public:
    bool supportsRetainedDeformationMotion() const noexcept override { return retainedDeformationMotion; }
    bool supportsRetainedImportedObjectMotion() const noexcept override { return retainedImportedObjectMotion; }
    bool supportsRetainedNoteMotion() const noexcept override { return retainedNoteMotion; }
    arbitgpu::BackendInfo info() const override
    {
        return {true, true, "opengl", "strict-test-device", {}};
    }

    arbitgpu::NativeDeformationPreparation prepare (
        const std::shared_ptr<const arbitgpu::NativeDeformationScene>&,
        const std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram>&) override
    {
        ++preparations;
        arbitgpu::NativeDeformationPreparation result;
        result.prepared = true;
        result.resources = std::make_shared<const FakeDeformationResources>();
        return result;
    }

    arbitgpu::NativeDeformationSubmission render (
        const std::shared_ptr<const arbitgpu::NativeDeformationScene>& source,
        const std::shared_ptr<const visualdeformation::AnimationDeformationSnapshot>&,
        const std::shared_ptr<const arbitgpu::NativeDeformationResources>&,
        std::uint32_t width,
        std::uint32_t height,
        arbitgpu::NativeDeformationRuntimeInputs inputs) override
    {
        ++submissions;
        lastRuntimeInputs = std::move(inputs);
        arbitgpu::NativeDeformationSubmission result;
        if (rejectSubmission) { result.error = "fake deformation draw failed"; return result; }
        result.rendered = true;
        result.frame = std::make_shared<FakeFrame> (
            "opengl", width, height, static_cast<std::uintptr_t> (submissions + 100));
        result.stats.drawCount = 1;
        result.stats.meshId = source->mesh.value;
        return result;
    }

    arbitgpu::NativeDeformationRuntimeInputs lastRuntimeInputs;
    bool rejectSubmission = false;
    bool retainedDeformationMotion = false, retainedNoteMotion = false;
    bool retainedImportedObjectMotion = false;
    int preparations = 0;
    int submissions = 0;
};

void importedObjectMatrixHistory(videohelper::modelpayload::Store& payloads)
{
    using Worlds = arbitgpu::NativeSceneMotionSample::ImportedNodeWorldMatrices;
    constexpr std::array<std::array<double,3>,3> triangle {{
        {{-0.5,-0.5,0.0}}, {{0.5,-0.5,0.0}}, {{0.0,0.5,0.0}}
    }};
    for (const auto mode : {0,1,2})
    {
        const bool parentTarget = mode == 1;
        const bool moving = mode != 2;
        const auto bytes = importedObjectMotionFixture(parentTarget, moving);
        const auto key = exactKey(parentTarget ? "motion-matrix-parent" : moving
            ? "motion-matrix-mesh" : "motion-matrix-static", bytes);
        check(publish(payloads, key.id.c_str(), key, bytes),
            "imported matrix fixture publishes its exact independent triangle source");
        FakeDeformationBackend backend;
        backend.retainedDeformationMotion = true;
        backend.retainedImportedObjectMotion = true;
        videohelper::modelpayload::ImportedAnimatedScenePayloadExecution execution(payloads, backend);
        videohelper::modelpayload::ImportedAnimatedSceneRequest request;
        request.operation.sourceStableId = 801;
        request.operation.deformationStableId = 802;
        request.operation.schedule = {801,802};
        request.operation.asset = key;
        request.operation.sceneIndex = 0;
        request.operation.meshStableId = request.operation.animationClipStableId = 1;
        request.operation.clipName = "Imported Object Motion";
        request.frame = {0,24,1}; request.structuralRevision = 1;
        request.runtimeInputs.imageOutput = renderpassoutput::Output::Motion;
        request.renderClipId = 83; request.projectGeneration = request.helperGeneration = 1;
        request.width = 96; request.height = 64;
        videohelper::modelpayload::ImportedAnimatedSceneReceipt receipt;
        std::string error;
        const bool first = execution.executePreview(request, receipt, error);
        check(first && !receipt.runtimeInputs.previousMotion,
            "first imported matrix sample has no predecessor");
        if (!first) { std::fprintf(stderr, "matrix fixture rejected: %s\n", error.c_str()); continue; }
        arbitgpu::NativeDeformationFrameData frame;
        Worlds firstWorlds;
        check(arbitgpu::prepareNativeDeformationFrame(*receipt.source, *receipt.evaluation.deformation,
                frame, error, &firstWorlds), "canonical draw optionally exposes its unmodified sampled worlds");
        arbitgpu::NativeDeformationFrameData ordinaryFrame;
        check(arbitgpu::prepareNativeDeformationFrame(*receipt.source, *receipt.evaluation.deformation,
                ordinaryFrame, error) && ordinaryFrame.jointPalette == frame.jointPalette
                && ordinaryFrame.morphWeights == frame.morphWeights,
            "optional sampled worlds leave ordinary palette and morph outputs byte-identical");
        auto foreign = *receipt.source;
        ++foreign.clip.value;
        auto rejectedWorlds = firstWorlds;
        check(!arbitgpu::prepareNativeDeformationFrame(foreign, *receipt.evaluation.deformation,
                frame, error, &rejectedWorlds) && rejectedWorlds.empty(),
            "foreign clip cannot expose partially sampled node worlds");
        check(!arbitgpu::sceneMotionSnapshotFits(receipt.evaluation.deformation,
                arbitgpu::kNativeSceneMotionMaximumSnapshotBytes)
                && !arbitgpu::sceneMotionSnapshotFits({}, arbitgpu::kNativeSceneMotionMaximumSnapshotBytes + 1),
            "node matrix bytes share the existing bounded snapshot budget");
        request.frame.frame = 1;
        request.operation.playback.timelineSeconds = 1.0 / 24.0;
        const bool second = execution.executePreview(request, receipt, error);
        const auto predecessor = second && receipt.runtimeInputs.previousMotion
            ? receipt.runtimeInputs.previousMotion->importedNodeWorldMatrices : nullptr;
        check(predecessor && predecessor->size() == 2 && (*predecessor)[0].nodeStableId == 1
                && (*predecessor)[1].nodeStableId == 4,
            "existing history retains only drawn-node and ancestor matrices in stable identity order");
        if (!predecessor) continue;
        auto legacy = *receipt.runtimeInputs.previousMotion;
        legacy.importedNodeWorldMatrices.reset();
        check(arbitgpu::validSceneMotionImportedNodeWorldMatrices(legacy)
                && !arbitgpu::validSceneMotionImportedNodeWorldMatrices(legacy, true),
            "legacy no-table samples remain optional only when imported object Motion is not required");
        auto structuralInputs = receipt.runtimeInputs;
        auto structuralWorlds = std::make_shared<Worlds>(*predecessor);
        structuralInputs.previousMotion->importedNodeWorldMatrices = structuralWorlds;
        check(arbitgpu::validFixtureRuntimeInputs(structuralInputs),
            "complete retained imported worlds pass shared structural validation");
        for (int invalid = 0; invalid < 5; ++invalid)
        {
            *structuralWorlds = *predecessor;
            if (invalid == 0) (*structuralWorlds)[0].nodeStableId = 0;
            if (invalid == 1) (*structuralWorlds)[1].nodeStableId = (*structuralWorlds)[0].nodeStableId;
            if (invalid == 2) std::swap((*structuralWorlds)[0], (*structuralWorlds)[1]);
            if (invalid == 3) (*structuralWorlds)[0].world[7] = std::numeric_limits<float>::quiet_NaN();
            if (invalid == 4)
            {
                structuralWorlds->reserve((arbitgpu::kNativeSceneMotionMaximumSnapshotBytes - sizeof(Worlds))
                    / sizeof(Worlds::value_type));
                check(arbitgpu::sceneMotionSnapshotFits({}, sizeof(Worlds)
                        + structuralWorlds->capacity() * sizeof(Worlds::value_type)),
                    "table alone fits while its exact deformation shares the retained-byte cap");
            }
            check(!arbitgpu::validFixtureRuntimeInputs(structuralInputs),
                "zero, duplicate, unsorted, nonfinite and over-cap imported worlds refuse shared admission");
        }
        const auto* previousMesh = importedWorld(*predecessor, 1);
        const auto* canonicalFirst = importedWorld(firstWorlds, 1);
        check(previousMesh && canonicalFirst && *previousMesh == *canonicalFirst,
            "retained world is the exact earlier canonical output without inverse reconstruction");
        const auto* previousParent = importedWorld(*predecessor, 4);
        const auto* canonicalParent = importedWorld(firstWorlds, 4);
        check(previousParent && canonicalParent && *previousParent == *canonicalParent
                && std::abs((*previousParent)[12] - 0.1f) < 0.000001f
                && std::abs((*previousParent)[13] - 0.2f) < 0.000001f,
            "retained ancestor uses the exact canonical world and authored parent translation");
        Worlds secondWorlds;
        const bool current = arbitgpu::prepareNativeDeformationFrame(*receipt.source,
            *receipt.evaluation.deformation, frame, error, &secondWorlds);
        const auto* currentMesh = importedWorld(secondWorlds, 1);
        check(current && previousMesh && currentMesh, "adjacent matrix sample preserves exact drawn-node identity");
        if (current && previousMesh && currentMesh)
            for (const auto& vertex : triangle)
            {
                const auto prior = projectedImportedTriangleVertex(*previousMesh, vertex, request.width, request.height);
                const auto now = projectedImportedTriangleVertex(*currentMesh, vertex, request.width, request.height);
                const auto expectedPrior = importedTrianglePixelOracle(parentTarget, moving, 0.0, vertex, request.height);
                const auto expectedNow = importedTrianglePixelOracle(parentTarget, moving, 1.0 / 24.0, vertex, request.height);
                for (std::size_t axis = 0; axis < 2; ++axis)
                    check(std::abs(prior[axis] - expectedPrior[axis]) < 0.00002
                            && std::abs(now[axis] - expectedNow[axis]) < 0.00002
                            && std::abs((now[axis] - prior[axis]) - (expectedNow[axis] - expectedPrior[axis])) < 0.00002,
                        "moving mesh, animated parent and static triangle match independent pixel-displacement equations");
            }
        request.frame.frame = 2; backend.rejectSubmission = true;
        request.operation.playback.timelineSeconds = 2.0 / 24.0;
        check(!execution.executePreview(request, receipt, error) && !receipt.valid(),
            "failed imported matrix draw cannot publish a receipt");
        const auto failedPrior = backend.lastRuntimeInputs.previousMotion
            ? backend.lastRuntimeInputs.previousMotion->importedNodeWorldMatrices : nullptr;
        backend.rejectSubmission = false;
        check(failedPrior && execution.executePreview(request, receipt, error) && receipt.runtimeInputs.previousMotion
                && receipt.runtimeInputs.previousMotion->importedNodeWorldMatrices == failedPrior,
            "retry retains exactly the last successful matrix owner");
        check(execution.executeExport(request, receipt, error) && !receipt.runtimeInputs.previousMotion,
            "Export matrices start independently from Preview");
        request.frame.frame = 3;
        request.operation.playback.timelineSeconds = 3.0 / 24.0;
        check(execution.executeExport(request, receipt, error) && receipt.runtimeInputs.previousMotion
                && receipt.runtimeInputs.previousMotion->importedNodeWorldMatrices,
            "Export retains its own adjacent imported matrices");
        request.frame.frame = 2;
        request.operation.playback.timelineSeconds = 2.0 / 24.0;
        check(execution.executePreview(request, receipt, error) && receipt.runtimeInputs.previousMotion
                && receipt.runtimeInputs.previousMotion->importedNodeWorldMatrices == failedPrior,
            "same-frame Preview redraw preserves its predecessor across interleaved Export");
        request.frame.frame = 3;
        request.operation.playback.timelineSeconds = 3.0 / 24.0;
        check(execution.executePreview(request, receipt, error) && receipt.runtimeInputs.previousMotion
                && receipt.runtimeInputs.previousMotion->importedNodeWorldMatrices,
            "Export does not replace Preview matrix history");
        for (int reset = 0; reset < 12; ++reset)
        {
            ++request.frame.frame;
            if (reset == 0) request.frame.frame += 10;
            if (reset == 1) request.frame.frame = 0;
            if (reset == 2) request.frame.rateNumerator = 30;
            if (reset == 3) request.frame.rateDenominator = 2;
            if (reset == 4) request.width = 80;
            if (reset == 5) request.height = 60;
            if (reset == 6) ++request.projectGeneration;
            if (reset == 7) ++request.helperGeneration;
            if (reset == 8) ++request.renderClipId;
            if (reset == 9) ++request.structuralRevision;
            if (reset == 10) ++request.operation.sourceStableId;
            if (reset == 11) ++request.operation.deformationStableId;
            request.operation.schedule = {request.operation.sourceStableId, request.operation.deformationStableId};
            request.operation.playback.timelineSeconds = static_cast<double>(request.frame.frame)
                * request.frame.rateDenominator / request.frame.rateNumerator;
            check(execution.executePreview(request, receipt, error) && !receipt.runtimeInputs.previousMotion,
                "seek, loop, rate, extent and exact owner changes reset imported matrix predecessors");
        }
        ++request.frame.frame;
        request.operation.sourceStableId = 801; request.operation.deformationStableId = 802;
        request.operation.schedule = {801,802};
        check(execution.executePreview(request, receipt, error) && !receipt.runtimeInputs.previousMotion,
            "returning to an interleaved source cannot borrow its earlier matrix history");
        backend.retainedImportedObjectMotion = false;
        const auto beforeRefusal = backend.submissions;
        check(!execution.executePreview(request, receipt, error) && !receipt.valid()
                && backend.submissions == beforeRefusal && error.find("object matrices") != std::string::npos,
            "object Motion keeps a separate default-false native consumer capability");
        backend.retainedImportedObjectMotion = true;
        for (int invalid = 0; invalid < 11; ++invalid)
        {
            execution.reset();
            request.frame = {0,24,1}; request.operation.playback.timelineSeconds = 0;
            const bool prime = execution.executePreview(request, receipt, error);
            request.frame.frame = 1; request.operation.playback.timelineSeconds = 1.0 / 24.0;
            const bool adjacent = prime && execution.executePreview(request, receipt, error);
            const auto retained = adjacent && receipt.runtimeInputs.previousMotion
                ? receipt.runtimeInputs.previousMotion->importedNodeWorldMatrices : nullptr;
            check(retained && retained->size() == 2,
                "malformed-world fixture starts from a complete successful predecessor");
            if (!retained || retained->size() != 2) continue;
            // The producer allocated a mutable vector before publishing it as
            // const. Deliberately corrupt that allocation to exercise admission.
            auto worlds = std::const_pointer_cast<Worlds>(retained);
            if (invalid == 0) worlds->clear();
            if (invalid == 1) worlds->pop_back();
            if (invalid == 2) worlds->push_back({5, worlds->front().world});
            if (invalid == 3) (*worlds)[0].nodeStableId = 0;
            if (invalid == 4) std::swap((*worlds)[0], (*worlds)[1]);
            if (invalid == 5) (*worlds)[1].nodeStableId = (*worlds)[0].nodeStableId;
            if (invalid == 6) (*worlds)[0].world[12] = std::numeric_limits<float>::infinity();
            if (invalid == 7) worlds->reserve(arbitgpu::kNativeSceneMotionMaximumSnapshotBytes
                / sizeof(Worlds::value_type) + 1);
            if (invalid == 8) (*worlds)[0].world[12] += 1;
            if (invalid == 9) (*worlds)[1].world[12] += 1;
            if (invalid == 10) (*worlds)[1].nodeStableId = 3;
            const auto beforeInvalid = backend.submissions;
            check(!execution.executePreview(request, receipt, error) && !receipt.valid()
                    && backend.submissions == beforeInvalid && error.find("node world") != std::string::npos,
                "missing, forged, stale, malformed and over-cap full tables refuse before native submission");
            check(execution.executePreview(request, receipt, error) && !receipt.runtimeInputs.previousMotion,
                "refused predecessor retires its lane so retry has neutral Motion");
        }
    }
}

// Two mesh nodes, each with its own mesh and translation track in one clip.
// Object A is node 1 and object B is node 4 (nodeIndex + 1 stable identity).
std::vector<std::uint8_t> twoAnimatedObjectsFixture()
{
    auto root = triangleRoot();
    auto bin = triangleBin();
    bin.resize(104, 0);
    for (const auto time : {0.0f, 1.0f}) appendFloat(bin, time);
    const std::array<std::array<float, 3>, 2> first {{
        {{0.25f, -0.125f, 0.0f}}, {{-0.4f, 0.3f, 0.0f}}}};
    const std::array<std::array<float, 3>, 2> delta {{
        {{0.5f, 0.0f, 0.0f}}, {{0.0f, 0.25f, 0.0f}}}};
    for (std::size_t object = 0; object < 2; ++object)
        for (int key = 0; key < 2; ++key)
            for (std::size_t axis = 0; axis < 3; ++axis)
                appendFloat(bin, first[object][axis] + static_cast<float>(key) * delta[object][axis]);
    auto mesh = root["meshes"][0];
    root["meshes"].push_back(mesh);
    root["nodes"].push_back({{"name", "Second Triangle"}, {"mesh", 1}});
    for (auto& scene : root["scenes"]) scene["nodes"] = {0, 1, 2, 3};
    root["bufferViews"].push_back({{"buffer",0}, {"byteOffset",104}, {"byteLength",8}});
    root["bufferViews"].push_back({{"buffer",0}, {"byteOffset",112}, {"byteLength",24}});
    root["bufferViews"].push_back({{"buffer",0}, {"byteOffset",136}, {"byteLength",24}});
    root["accessors"].push_back({{"bufferView",4}, {"componentType",5126}, {"count",2}, {"type","SCALAR"}});
    root["accessors"].push_back({{"bufferView",5}, {"componentType",5126}, {"count",2}, {"type","VEC3"}});
    root["accessors"].push_back({{"bufferView",6}, {"componentType",5126}, {"count",2}, {"type","VEC3"}});
    root["buffers"][0]["byteLength"] = bin.size();
    root["animations"] = nlohmann::json::array({{
        {"name","Two Object Motion"},
        {"samplers",nlohmann::json::array({
            {{"input",4}, {"output",5}, {"interpolation","LINEAR"}},
            {{"input",4}, {"output",6}, {"interpolation","LINEAR"}}})},
        {"channels",nlohmann::json::array({
            {{"sampler",0}, {"target",{{"node",0}, {"path","translation"}}}},
            {{"sampler",1}, {"target",{{"node",3}, {"path","translation"}}}}})}
    }});
    return makeGlb(std::move(root), std::move(bin));
}

bool nearTranslation(const std::array<float, 16>* world, float x, float y, float z)
{
    // Column-major translation lives in elements 12, 13 and 14.
    return world != nullptr
        && std::abs((*world)[12] - x) < 0.00001f
        && std::abs((*world)[13] - y) < 0.00001f
        && std::abs((*world)[14] - z) < 0.00001f;
}

void twoAnimatedObjectsOneComposedScene(videohelper::modelpayload::Store& payloads)
{
    using Worlds = arbitgpu::NativeSceneMotionSample::ImportedNodeWorldMatrices;
    const auto bytes = twoAnimatedObjectsFixture();
    const auto key = exactKey("two-animated-objects", bytes);
    check(publish(payloads, key.id.c_str(), key, bytes),
        "two-object animated fixture publishes its exact source");
    const float step = 1.0f / 24.0f;
    // Selection 0 composes both objects. Selection 2 targets only object B.
    const std::array<unsigned, 2> selections {{0u, 2u}};
    for (const auto selection : selections)
    {
        const bool wholeScene = selection == 0u;
        FakeDeformationBackend backend;
        backend.retainedDeformationMotion = true;
        backend.retainedImportedObjectMotion = true;
        videohelper::modelpayload::ImportedAnimatedScenePayloadExecution execution(payloads, backend);
        videohelper::modelpayload::ImportedAnimatedSceneRequest request;
        request.operation.sourceStableId = 811;
        request.operation.deformationStableId = 812;
        request.operation.schedule = {811,812};
        request.operation.asset = key;
        request.operation.sceneIndex = 0;
        request.operation.meshStableId = selection;
        request.operation.animationClipStableId = 1;
        request.operation.clipName = "Two Object Motion";
        request.frame = {0,24,1}; request.structuralRevision = 1;
        request.runtimeInputs.imageOutput = renderpassoutput::Output::Motion;
        request.renderClipId = 84; request.projectGeneration = request.helperGeneration = 1;
        request.width = 96; request.height = 64;
        videohelper::modelpayload::ImportedAnimatedSceneReceipt receipt;
        std::string error;
        const bool first = execution.executePreview(request, receipt, error);
        check(first, "per-object selection admits its first sample");
        if (!first)
        {
            std::fprintf(stderr, "two-object fixture rejected: %s\n", error.c_str());
            continue;
        }
        check(!receipt.runtimeInputs.previousMotion,
            "per-object first sample has no predecessor");
        request.frame.frame = 1;
        request.operation.playback.timelineSeconds = 1.0 / 24.0;
        const bool second = execution.executePreview(request, receipt, error);
        const auto predecessor = second && receipt.runtimeInputs.previousMotion
            ? receipt.runtimeInputs.previousMotion->importedNodeWorldMatrices : nullptr;
        check(predecessor != nullptr, "two-object second sample retains a predecessor");
        if (!predecessor)
        {
            std::fprintf(stderr, "two-object predecessor missing: %s\n", error.c_str());
            continue;
        }
        if (wholeScene)
            check(predecessor->size() == 2 && (*predecessor)[0].nodeStableId == 1
                    && (*predecessor)[1].nodeStableId == 4
                    && nearTranslation(importedWorld(*predecessor, 1), 0.25f, -0.125f, 0.0f)
                    && nearTranslation(importedWorld(*predecessor, 4), -0.4f, 0.3f, 0.0f),
                "composed scene retains both objects by identity with their own frame-zero worlds");
        else
            check(predecessor->size() == 1 && (*predecessor)[0].nodeStableId == 4
                    && nearTranslation(importedWorld(*predecessor, 4), -0.4f, 0.3f, 0.0f),
                "selected object retains only its own identity and frame-zero world");
        arbitgpu::NativeDeformationFrameData frame;
        Worlds current;
        check(arbitgpu::prepareNativeDeformationFrame(*receipt.source, *receipt.evaluation.deformation,
                frame, error, &current)
                && nearTranslation(importedWorld(current, 4), -0.4f, 0.3f + 0.25f * step, 0.0f)
                && (!wholeScene || nearTranslation(importedWorld(current, 1), 0.25f + 0.5f * step, -0.125f, 0.0f)),
            "each object samples its own track at frame one");
    }
    // Mesh selection 3 names mesh index 2, which this two-mesh source does not contain.
    FakeDeformationBackend refusedBackend;
    refusedBackend.retainedDeformationMotion = true;
    refusedBackend.retainedImportedObjectMotion = true;
    videohelper::modelpayload::ImportedAnimatedScenePayloadExecution refused(payloads, refusedBackend);
    videohelper::modelpayload::ImportedAnimatedSceneRequest missing;
    missing.operation.sourceStableId = 813;
    missing.operation.deformationStableId = 814;
    missing.operation.schedule = {813,814};
    missing.operation.asset = key;
    missing.operation.sceneIndex = 0;
    missing.operation.meshStableId = 3u;
    missing.operation.animationClipStableId = 1;
    missing.operation.clipName = "Two Object Motion";
    missing.frame = {0,24,1}; missing.structuralRevision = 1;
    missing.runtimeInputs.imageOutput = renderpassoutput::Output::Motion;
    missing.renderClipId = 85; missing.projectGeneration = missing.helperGeneration = 1;
    missing.width = 96; missing.height = 64;
    videohelper::modelpayload::ImportedAnimatedSceneReceipt missingReceipt;
    std::string missingError;
    const auto beforeMissing = refusedBackend.submissions;
    check(!refused.executePreview(missing, missingReceipt, missingError) && !missingReceipt.valid()
        && refusedBackend.submissions == beforeMissing && !missingError.empty(),
        "an object selection with no source mesh is refused before native submission");
}

// A refused multi-leaf admission keeps the admitted single leaf and refunds
// every per-leaf reservation; a successful one replaces the single leaf, and a
// later single admission replaces the leaves and refunds their reservations.
void compositionLeafAdmissionIsExclusiveAndCharged()
{
    using Probe = videohelper::modelpayload::ImportedAnimatedSceneCompositionProbe;
    using Budget = videohelper::modelpayload::NativeSceneAnimationCompositionBudget;
    const std::vector<std::uint8_t> starter {
        visualstartermodel::kAnimatedTriangleGlb.begin(), visualstartermodel::kAnimatedTriangleGlb.end() };
    const auto twoObjects = twoAnimatedObjectsFixture();
    const auto starterKey = exactKey("composition-leaf-starter", starter);
    const auto twoObjectsKey = exactKey("composition-leaf-two-objects", twoObjects);
    const auto staticKey = exactKey("composition-leaf-static", starter);
    videohelper::modelpayload::Store store;
    check(publish(store, "composition-leaf-starter", starterKey, starter)
            && publish(store, "composition-leaf-two-objects", twoObjectsKey, twoObjects)
            && publish(store, "composition-leaf-static", staticKey, starter),
        "composition leaf fixtures publish their exact sources");
    FakeDeformationBackend backend;
    videohelper::modelpayload::ImportedAnimatedScenePayloadExecution execution(store, backend);
    videohelper::modelpayload::ImportedAnimatedSceneRequest request;
    request.operation.sourceStableId = 901;
    request.operation.deformationStableId = 902;
    request.operation.schedule = {901, 902};
    request.operation.asset = starterKey;
    request.operation.sceneIndex = 0u;
    request.operation.meshStableId = 1u;
    request.operation.animationClipStableId = 1u;
    request.operation.clipName = std::string(visualstartermodel::kClipName);
    request.frame = {0, 24, 1};
    request.structuralRevision = 1;
    request.renderClipId = 86; request.projectGeneration = request.helperGeneration = 1;
    request.width = 96; request.height = 64;
    videohelper::modelpayload::ImportedAnimatedSceneReceipt receipt;
    std::string error;
    check(execution.executePreview(request, receipt, error) && receipt.valid() && backend.submissions == 1,
        "the single leaf admits and renders before any composition admission");
    const auto single = Probe::scene(execution);
    const auto singleCopy = Probe::singleCopyBytes(execution);
    check(single && Probe::singleActive(execution) && Probe::leaves(execution) == 0 && Probe::ledger(execution)
            && Probe::singleBindings(execution) > 0
            && singleCopy >= Probe::singleBindings(execution) * sizeof(videohelper::gltf::GlbDeformationRenderBinding)
            && Probe::ledgerBytes(execution) == sizeof(Budget) + singleCopy,
        "one admitted leaf holds the single-leaf state and only its binding copy reservation");

    Probe::Input animated;
    animated.payload = store.resolvePreview(starterKey);
    animated.operation = request.operation;
    Probe::Input objects;
    objects.payload = store.resolvePreview(twoObjectsKey);
    objects.operation = request.operation;
    objects.operation.asset = twoObjectsKey;
    objects.operation.meshStableId = 0u;
    objects.operation.clipName = "Two Object Motion";
    Probe::Input still;
    still.payload = store.resolvePreview(staticKey);
    still.operation = request.operation;
    still.operation.asset = staticKey;
    still.operation.animationClipStableId = 0u;
    still.operation.clipName.clear();
    still.animated = false;
    check(animated.payload && objects.payload && still.payload, "composition leaf payloads resolve");

    // Refusals: a missing payload after an admitted leaf, a static leaf that
    // names a clip, and a list of static leaves only.
    auto missing = objects;
    missing.payload.reset();
    auto clipped = still;
    clipped.operation.clipName = "Two Object Motion";
    auto stillOnly = still;
    const std::vector<std::pair<std::vector<Probe::Input>, const char*>> refused {
        {{animated, missing}, "imported composition leaf exact payload is not resident"},
        {{animated, clipped}, "imported composition static leaf names an animation clip"},
        {{still, stillOnly}, "imported animated composition requires at least one animated leaf"},
        {{animated}, "imported composition requires two to eight leaves"}};
    for (const auto& [inputs, message] : refused)
    {
        check(!Probe::admit(execution, inputs, error) && error == message, message);
        check(Probe::scene(execution) == single && Probe::singleActive(execution) && Probe::leaves(execution) == 0,
            "a refused composition admission keeps the admitted single leaf unchanged");
        check(Probe::ledgerBytes(execution) == sizeof(Budget) + singleCopy
                && Probe::singleCopyBytes(execution) == singleCopy,
            "a refused composition admission refunds every leaf reservation and keeps the single leaf's");
    }
    request.frame.frame = 1;
    request.operation.playback.timelineSeconds = 1.0 / 24.0;
    check(execution.executePreview(request, receipt, error) && receipt.valid() && backend.submissions == 2
            && Probe::scene(execution) == single,
        "execute() after refused composition admissions reuses the same single leaf");

    check(Probe::admit(execution, {animated, objects, still}, error) && error.empty(),
        "an animated, a two-object animated and a static leaf admit together");
    const auto charged = Probe::leafChargeBytes(execution);
    check(Probe::leaves(execution) == 3 && !Probe::singleActive(execution) && !Probe::scene(execution)
            && Probe::animated(execution, 0) && Probe::animated(execution, 1) && !Probe::animated(execution, 2)
            && Probe::bindings(execution, 0) > 0 && Probe::bindings(execution, 2) == 0,
        "successful composition admission replaces the single leaf with three leaves in order");
    check(charged >= 3 * sizeof(HarmonicMIDI::grid::Visual3DScene) + starter.size() + twoObjects.size() + starter.size()
            && Probe::singleCopyBytes(execution) == 0 && Probe::ledgerBytes(execution) == sizeof(Budget) + charged,
        "every leaf holds its reservations, the single leaf's copy is refunded, and the ledger carries exactly their sum");

    request.frame.frame = 2;
    request.operation.playback.timelineSeconds = 2.0 / 24.0;
    check(execution.executePreview(request, receipt, error) && receipt.valid() && backend.submissions == 3,
        "a later single-leaf execution admits again");
    check(Probe::leaves(execution) == 0 && Probe::singleActive(execution)
            && Probe::singleCopyBytes(execution) == singleCopy
            && Probe::ledgerBytes(execution) == sizeof(Budget) + singleCopy,
        "single-leaf admission drops the composition leaves and refunds their reservations");
    execution.reset();
    check(!Probe::singleActive(execution) && Probe::leaves(execution) == 0
            && Probe::singleCopyBytes(execution) == 0 && Probe::ledgerBytes(execution) == sizeof(Budget),
        "reset clears both admission states and refunds the single leaf's binding copy");
}

// The kept render binding copy takes its own reservation. A leaf that fits its
// fixed storage and decoded closure but not its binding copy refuses, and the
// refusal refunds every reservation the leaf took.
void compositionLeafBindingCopyIsCharged()
{
    using Probe = videohelper::modelpayload::ImportedAnimatedSceneCompositionProbe;
    using Budget = videohelper::modelpayload::NativeSceneAnimationCompositionBudget;
    const std::vector<std::uint8_t> starter {
        visualstartermodel::kAnimatedTriangleGlb.begin(), visualstartermodel::kAnimatedTriangleGlb.end() };
    const auto animatedKey = exactKey("binding-copy-animated", starter);
    const auto staticKey = exactKey("binding-copy-static", starter);
    videohelper::modelpayload::Store store;
    check(publish(store, "binding-copy-animated", animatedKey, starter)
            && publish(store, "binding-copy-static", staticKey, starter),
        "binding copy fixtures publish their exact sources");
    Probe::Input animated;
    animated.payload = store.resolvePreview(animatedKey);
    animated.operation.sourceStableId = 921;
    animated.operation.deformationStableId = 922;
    animated.operation.schedule = {921, 922};
    animated.operation.asset = animatedKey;
    animated.operation.sceneIndex = 0u;
    animated.operation.meshStableId = 1u;
    animated.operation.animationClipStableId = 1u;
    animated.operation.clipName = std::string(visualstartermodel::kClipName);
    Probe::Input still;
    still.payload = store.resolvePreview(staticKey);
    still.operation.sourceStableId = 923;
    still.operation.asset = staticKey;
    still.operation.sceneIndex = 0u;
    still.animated = false;
    FakeDeformationBackend backend;
    videohelper::modelpayload::ImportedAnimatedScenePayloadExecution measured(store, backend);
    std::string error;
    check(Probe::admit(measured, {animated, still}, error) && error.empty(),
        "an animated and a static leaf admit with room to spare");
    if (Probe::leaves(measured) != 2) return;
    const auto fixed = Probe::chargeBytes(measured, 0, 0);
    const auto closure = Probe::chargeBytes(measured, 0, 1);
    const auto copy = Probe::chargeBytes(measured, 0, 2);
    check(fixed > 0 && closure > 0 && Probe::bindings(measured, 0) > 0
            && copy >= Probe::bindings(measured, 0) * sizeof(videohelper::gltf::GlbDeformationRenderBinding)
            && Probe::chargeBytes(measured, 1, 2) == 0,
        "an animated leaf reserves its binding copy and a static leaf reserves none");

    videohelper::modelpayload::ImportedAnimatedScenePayloadExecution nearCap(store, backend);
    auto& ledger = Probe::animationLedger(nearCap);
    ledger = std::make_shared<Budget>();
    const auto room = fixed + closure + copy - 1;
    const auto filler = Budget::reserve(ledger, Budget::kMaximumBytes - ledger->logicalBytes() - room, error);
    const auto before = ledger->logicalBytes();
    check(filler && !Probe::admit(nearCap, {animated, still}, error)
            && error == "imported composition leaf render binding copy exceeds logical CPU capacity"
            && Probe::leaves(nearCap) == 0 && ledger->logicalBytes() == before,
        "a near-cap composition refuses at the binding copy and refunds the leaf");
}

// One leaf's kept binding copy reserves its bytes too. With one byte less than
// the copy free, the frame refuses before submission and leaves the ledger as
// it found it; with exactly the copy free, the same frame renders.
void singleLeafBindingCopyIsCharged()
{
    using namespace videohelper::modelpayload;
    using Probe = ImportedAnimatedSceneCompositionProbe;
    using Budget = NativeSceneAnimationCompositionBudget;
    const std::vector<std::uint8_t> starter {
        visualstartermodel::kAnimatedTriangleGlb.begin(), visualstartermodel::kAnimatedTriangleGlb.end() };
    const auto key = exactKey("single-binding-copy", starter);
    Store store;
    check(publish(store, "single-binding-copy", key, starter), "single binding copy fixture publishes its exact source");
    ImportedAnimatedSceneRequest request;
    request.operation.sourceStableId = 951;
    request.operation.deformationStableId = 952;
    request.operation.schedule = {951, 952};
    request.operation.asset = key;
    request.operation.sceneIndex = 0u;
    request.operation.meshStableId = 1u;
    request.operation.animationClipStableId = 1u;
    request.operation.clipName = std::string(visualstartermodel::kClipName);
    request.frame = {0, 24, 1};
    request.structuralRevision = 1;
    request.renderClipId = 87; request.projectGeneration = request.helperGeneration = 1;
    request.width = 96; request.height = 64;
    std::string error;
    FakeDeformationBackend measuredBackend;
    ImportedAnimatedScenePayloadExecution measured(store, measuredBackend);
    ImportedAnimatedSceneReceipt measuredReceipt;
    check(measured.executePreview(request, measuredReceipt, error) && measuredReceipt.valid(),
        "the single leaf renders with room to spare");
    const auto copy = Probe::singleCopyBytes(measured);
    check(copy > 0 && Probe::ledgerBytes(measured) == sizeof(Budget) + copy,
        "the single leaf's binding copy is its only scene animation reservation");
    if (copy == 0) return;

    for (const bool fits : {false, true})
    {
        FakeDeformationBackend backend;
        ImportedAnimatedScenePayloadExecution execution(store, backend);
        auto& ledger = Probe::animationLedger(execution);
        ledger = std::make_shared<Budget>();
        const auto room = fits ? copy : copy - 1;
        const auto filler = Budget::reserve(ledger, Budget::kMaximumBytes - ledger->logicalBytes() - room, error);
        const auto before = ledger->logicalBytes();
        ImportedAnimatedSceneReceipt receipt;
        const bool rendered = filler && execution.executePreview(request, receipt, error);
        if (fits)
            check(rendered && receipt.valid() && backend.submissions == 1
                    && Probe::singleCopyBytes(execution) == copy && ledger->logicalBytes() == Budget::kMaximumBytes,
                "a single leaf whose binding copy exactly fits renders");
        else
            check(filler && !rendered && !receipt.valid() && backend.submissions == 0
                    && error == "imported animated scene render binding copy exceeds logical CPU capacity"
                    && !Probe::singleActive(execution) && Probe::singleCopyBytes(execution) == 0
                    && ledger->logicalBytes() == before,
                "a single leaf one byte short of its binding copy refuses before submission and refunds");
    }
}

// The starter's procedural twin with every material, texture, light and camera
// identity moved out of the imported ranges, so it joins any imported leaf.
std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> twoLeafProceduralScene()
{
    FakeBackend sceneBackend;
    videohelper::gltf::GlbNativeRenderSeam seam(sceneBackend);
    videohelper::gltf::GlbNativeSceneAdmission admission;
    std::string error;
    const auto glb = makeGlb(triangleRoot(), triangleBin());
    if (!seam.admit(glb, 0u, admission, error) || !admission.scene) return {};
    auto scene = std::make_shared<HarmonicMIDI::grid::Visual3DScene>(*admission.scene);
    const auto shift = [](auto& id) { if (id.isValid()) id.value += 1000u; };
    for (std::size_t i = 0; i < scene->textureCount; ++i) shift(scene->textures[i].id);
    for (std::size_t i = 0; i < scene->materialCount; ++i)
    {
        auto& material = scene->materials[i];
        shift(material.id); shift(material.baseColorTexture); shift(material.metallicRoughnessTexture);
        shift(material.normalTexture); shift(material.occlusionTexture); shift(material.emissiveTexture);
    }
    for (std::size_t i = 0; i < scene->objectCount; ++i) shift(scene->objects[i].material);
    for (std::size_t i = 0; i < scene->lightCount; ++i) shift(scene->lights[i].id);
    for (std::size_t i = 0; i < scene->cameraCount; ++i) shift(scene->cameras[i].id);
    shift(scene->activeCamera);
    return scene;
}

// Two animated leaves render through one owner. Each leaf's consumer and rest
// scene keep raw IDs; the composed draws and selectors use the leaf scheme,
// where leaf 1 node 1 is 2049 and its object is 2048 * 65536 + 1.
void twoLeafCompositionRendersEveryLeaf()
{
    using namespace videohelper::modelpayload;
    constexpr std::uint32_t leafOneObject = 2048u * 65536u + 1u;
    const std::vector<std::uint8_t> starter {
        visualstartermodel::kAnimatedTriangleGlb.begin(), visualstartermodel::kAnimatedTriangleGlb.end() };
    const auto firstKey = exactKey("two-leaf-first", starter);
    const auto secondKey = exactKey("two-leaf-second", starter);
    Store store;
    check(publish(store, "two-leaf-first", firstKey, starter) && publish(store, "two-leaf-second", secondKey, starter),
        "two-leaf fixtures publish their exact sources");
    const auto procedural = twoLeafProceduralScene();
    check(procedural != nullptr, "the two-leaf procedural scene admits");
    if (!procedural) return;
    visualanimationimport::Request first;
    first.sourceStableId = 931;
    first.deformationStableId = 932;
    first.schedule = {931, 932};
    first.asset = firstKey;
    first.sceneIndex = 0u;
    first.meshStableId = 1u;
    first.animationClipStableId = 1u;
    first.clipName = std::string(visualstartermodel::kClipName);
    auto second = first;
    second.sourceStableId = 933;
    second.deformationStableId = 934;
    second.schedule = {933, 934};
    second.asset = secondKey;
    ImportedAnimatedSceneRequest request;
    request.operation = first;
    request.compositionLeaves.resize(2);
    request.compositionLeaves[0].operation = first;
    request.compositionLeaves[1].operation = second;
    request.proceduralScene = procedural;
    request.structuralRevision = 1; request.evaluationRevision = 1;
    request.frame = {0, 24, 1};
    request.renderClipId = 88; request.projectGeneration = request.helperGeneration = 1;
    request.width = 96; request.height = 64;
    FakeDeformationBackend backend;
    ImportedAnimatedScenePayloadExecution execution(store, backend);
    ImportedAnimatedSceneReceipt receipt;
    std::string error;
    const bool rendered = execution.executePreview(request, receipt, error);
    if (!rendered) std::fprintf(stderr, "two-leaf composition refused: %s\n", error.c_str());
    check(rendered && receipt.valid() && backend.submissions == 1 && receipt.runtimeInputs.compositionFrame
            && receipt.source && receipt.source->compositionOwner,
        "two animated leaves render one composed native frame");
    if (!rendered || !receipt.source || !receipt.source->compositionOwner || !receipt.runtimeInputs.compositionFrame)
        return;
    const auto owner = receipt.source->compositionOwner;
    const auto& frame = *receipt.runtimeInputs.compositionFrame;
    std::size_t leafZero = owner->memberCount(), leafOne = owner->memberCount();
    bool proceduralRigid = false, inRange = true;
    for (std::size_t i = 0; i < owner->memberCount(); ++i)
    {
        const auto object = owner->memberObject(i).value;
        if (object == 1u) leafZero = i;
        else if (object == leafOneObject) leafOne = i;
        else if (object >= 0x80000000u) proceduralRigid = owner->memberRigid(i);
        else inRange = false;
    }
    check(owner->leafCount() == 2 && !owner->staticLeaf(0) && !owner->staticLeaf(1) && inRange && proceduralRigid
            && leafZero < owner->memberCount() && leafOne < owner->memberCount(),
        "each leaf draws its object in its own composed range beside the rigid procedural object");
    if (leafZero >= owner->memberCount() || leafOne >= owner->memberCount()) return;
    check(owner->memberLeaf(leafZero) == 0 && owner->memberLeaf(leafOne) == 1
            && !owner->memberRigid(leafZero) && !owner->memberRigid(leafOne)
            && owner->originalMember(leafZero)->sourceStableId == 931
            && owner->originalMember(leafOne)->sourceStableId == 933
            && owner->originalMember(leafOne)->object.value == 1u
            && &frame.memberSnapshot(leafZero) != &frame.memberSnapshot(leafOne),
        "each draw keeps its leaf, its raw leaf object and its own leaf sample");
    // The two ID spaces never mix: the composed spelling names leaf 1, the raw
    // spelling names leaf 0 only.
    check(owner->targetsObject(2049, {leafOneObject}) && !owner->targetsObject(1, {leafOneObject})
            && owner->targetsObject(1, {1u}) && !owner->targetsObject(2049, {1u})
            && owner->importedNode(2049) && !owner->importedNode(2048),
        "composed selectors reach leaf 1 only through the leaf ID scheme");

    request.frame.frame = 1;
    request.compositionLeaves[1].operation.playback.timelineSeconds = 0.5;
    request.compositionLeaves[0].operation.playback.timelineSeconds = 1.0 / 24.0;
    check(execution.executePreview(request, receipt, error) && receipt.valid() && backend.submissions == 2
            && receipt.source && receipt.source->compositionOwner == owner,
        "a later frame with new leaf playback times reuses the admitted leaves and owner");

    // A static leaf 1 selects its node in the leaf scheme. The runtime keeps
    // that spelling; the raw spelling would name leaf 0 and is refused.
    visualanimationimport::Request still;
    still.sourceStableId = 933;
    still.asset = secondKey;
    still.sceneIndex = 0u;
    auto selected = request;
    selected.compositionLeaves[1].operation = still;
    selected.compositionLeaves[1].animated = false;
    selected.compositionLeaves[1].objectNodeStableId = 2049u;
    const bool selectedRendered = execution.executePreview(selected, receipt, error);
    if (!selectedRendered) std::fprintf(stderr, "two-leaf static selection refused: %s\n", error.c_str());
    check(selectedRendered && receipt.valid() && backend.submissions == 3
            && receipt.runtimeInputs.objectNodeStableId == 2049u
            && backend.lastRuntimeInputs.objectNodeStableId == 2049u
            && receipt.source && receipt.source->compositionOwner
            && receipt.source->compositionOwner->staticLeaf(1),
        "a static leaf selection reaches the native draw in the leaf ID scheme");
    selected.compositionLeaves[1].objectNodeStableId = 1u;
    check(!execution.executePreview(selected, receipt, error) && !receipt.valid() && backend.submissions == 3
            && error == "Scene Compose refused: imported leaf 1 selects a node outside its own leaf",
        "a raw leaf 1 selector is refused before native submission");
    // Execution admits one selected object across all leaves: leaf 0 posing
    // its node 1 beside leaf 1 selecting 2049 is refused before submission.
    auto twoSelectors = selected;
    twoSelectors.compositionLeaves[1].objectNodeStableId = 2049u;
    twoSelectors.compositionLeaves[0].operation.pose.nodeStableId = 1u;
    twoSelectors.compositionLeaves[0].operation.pose.meshStableId = 1u;
    check(!execution.executePreview(twoSelectors, receipt, error) && !receipt.valid() && backend.submissions == 3
            && error == "Scene Compose refused: more than one imported leaf selects an object",
        "two selecting leaves are refused before native submission");
}

// One Render Material on two animated leaves. The plugin lowers leaf 1's copy
// with its object at 2048 * 65536 + 1 and its Material ID literal moved by
// 2048, and computes the digest after that move. Each copy joins the composed
// Surface collection as its own Render override.
void twoLeafCompositionBindsEveryLeafMaterial()
{
    using namespace videohelper::modelpayload;
    namespace composition = HarmonicMIDI::grid::scenecomposition;
    constexpr std::uint32_t leafOneObject = 2048u * 65536u + 1u;
    const std::vector<std::uint8_t> starter {
        visualstartermodel::kAnimatedTriangleGlb.begin(), visualstartermodel::kAnimatedTriangleGlb.end() };
    const auto firstKey = exactKey("leaf-material-first", starter);
    const auto secondKey = exactKey("leaf-material-second", starter);
    Store store;
    check(publish(store, "leaf-material-first", firstKey, starter) && publish(store, "leaf-material-second", secondKey, starter),
        "leaf material fixtures publish their exact sources");
    const auto procedural = twoLeafProceduralScene();
    check(procedural != nullptr, "the leaf material procedural scene admits");
    if (!procedural) return;

    // The starter's raw material on its object 1, read from its own admission.
    std::uint32_t rawMaterial = 0;
    {
        FakeBackend sceneBackend;
        videohelper::gltf::GlbNativeRenderSeam seam(sceneBackend);
        videohelper::gltf::GlbNativeSceneAdmission admission;
        std::string admissionError;
        const bool admitted = seam.admit(starter, 0u, admission, admissionError) && admission.scene;
        if (admitted)
            for (std::size_t i = 0; i < admission.scene->objectCount; ++i)
                if (admission.scene->objects[i].id.value == 1u) rawMaterial = admission.scene->objects[i].material.value;
        if (!admitted) std::fprintf(stderr, "leaf material starter admission: %s\n", admissionError.c_str());
        check(admitted && rawMaterial != 0u && rawMaterial < composition::kImportedLeafIdStride,
            "the starter object 1 has a material inside the leaf range");
        if (!admitted || rawMaterial == 0u) return;
    }
    const auto leafMaterial = [&](std::size_t leaf)
    {
        surfacematerial::ProgramDescription program;
        for (const auto output : surfacematerial::kSurfaceOutputs)
        {
            surfacematerial::Operation operation;
            operation.id = static_cast<surfacematerial::ValueId>(program.operations.size() + 1u);
            operation.resultType = surfacematerial::outputType(output);
            operation.kind = operation.resultType == surfacematerial::ValueType::UInt
                ? surfacematerial::OperationKind::UIntConstant : surfacematerial::OperationKind::FloatConstant;
            if (output == surfacematerial::OutputSemantic::BaseColor) operation.literal = {0.8f, 0.15f, 0.05f, 0.0f};
            else if (output == surfacematerial::OutputSemantic::Normal) operation.literal = {0.0f, 0.0f, 1.0f, 0.0f};
            else if (output == surfacematerial::OutputSemantic::Opacity) operation.literal[0] = 1.0f;
            else if (output == surfacematerial::OutputSemantic::Ior) operation.literal[0] = 1.5f;
            else if (output == surfacematerial::OutputSemantic::MaterialId) operation.unsignedLiteral = rawMaterial;
            program.outputs[static_cast<std::size_t>(output)] = operation.id;
            program.operations.push_back(operation);
        }
        // The plugin's lowering order: move the literal, then admit.
        const bool moved = geometrysurfacematerial::remapImportedLeafMaterialId(program, leaf, 2);
        std::string admissionError;
        const auto admitted = surfacematerial::admit(program, admissionError);
        if (!admitted) std::fprintf(stderr, "leaf material program admission: %s\n", admissionError.c_str());
        check(moved && admitted.has_value(), "the leaf material program admits after its literal moves");
        surfacematerialbinding::ImportedSceneMaterialRequest material;
        material.scene.value = 1u;
        material.sceneRevision = material.structuralRevision = material.evaluationRevision = material.programRevision = 1;
        material.program = program;
        material.binding.targetKind = surfacematerialbinding::BindingTargetKind::ObjectOverride;
        material.binding.object = *composition::composedImportedLeafObjectId(leaf, 2, HarmonicMIDI::grid::SceneObjectId {1u});
        material.binding.surfaceMaterialDigest = admitted ? admitted->structuralDigest() : std::string(64, '0');
        material.binding.surfaceMaterialRevision = 1;
        return material;
    };

    visualanimationimport::Request first;
    first.sourceStableId = 941;
    first.deformationStableId = 942;
    first.schedule = {941, 942};
    first.asset = firstKey;
    first.sceneIndex = 0u;
    first.meshStableId = 1u;
    first.animationClipStableId = 1u;
    first.clipName = std::string(visualstartermodel::kClipName);
    auto second = first;
    second.sourceStableId = 943;
    second.deformationStableId = 944;
    second.schedule = {943, 944};
    second.asset = secondKey;
    ImportedAnimatedSceneRequest request;
    request.operation = first;
    request.compositionLeaves.resize(2);
    request.compositionLeaves[0].operation = first;
    request.compositionLeaves[1].operation = second;
    request.material = leafMaterial(0);
    request.compositionLeaves[1].material = leafMaterial(1);
    request.proceduralScene = procedural;
    request.structuralRevision = 1; request.evaluationRevision = 1;
    request.frame = {0, 24, 1};
    request.renderClipId = 89; request.projectGeneration = request.helperGeneration = 1;
    request.width = 96; request.height = 64;
    check(request.material->binding.object.value == 1u
            && request.compositionLeaves[1].material->binding.object.value == leafOneObject
            && geometrysurfacematerial::materialId(*request.material) == rawMaterial
            && geometrysurfacematerial::materialId(*request.compositionLeaves[1].material) == 2048u + rawMaterial,
        "the two leaf materials name their own composed object and material");

    FakeDeformationBackend backend;
    ImportedAnimatedScenePayloadExecution execution(store, backend);
    ImportedAnimatedSceneReceipt receipt;
    std::string error;
    const bool rendered = execution.executePreview(request, receipt, error);
    if (!rendered) std::fprintf(stderr, "two-leaf material composition refused: %s\n", error.c_str());
    check(rendered && receipt.valid() && backend.submissions == 1 && !receipt.usedLastGoodMaterial
            && receipt.materialDiagnostic.empty(),
        "both leaves' Render Materials admit as one composed Surface collection and draw");

    // Leaf 1 alone also binds, beside no leaf 0 material.
    auto onlyLeafOne = request;
    onlyLeafOne.material.reset();
    onlyLeafOne.renderClipId = 90;
    {
        FakeDeformationBackend leafBackend;
        ImportedAnimatedScenePayloadExecution leafExecution(store, leafBackend);
        const bool leafRendered = leafExecution.executePreview(onlyLeafOne, receipt, error);
        if (!leafRendered) std::fprintf(stderr, "leaf 1 material composition refused: %s\n", error.c_str());
        check(leafRendered && receipt.valid() && leafBackend.submissions == 1,
            "a Render Material on leaf 1 alone joins the composed Surface collection");
    }

    const auto refuses = [&](const ImportedAnimatedSceneRequest& candidate, const std::string& fragment, const char* message)
    {
        FakeDeformationBackend refusedBackend;
        ImportedAnimatedScenePayloadExecution refusedExecution(store, refusedBackend);
        ImportedAnimatedSceneReceipt refusedReceipt;
        std::string refusal;
        const bool refused = !refusedExecution.executePreview(candidate, refusedReceipt, refusal)
            && !refusedReceipt.valid() && refusedBackend.submissions == 0
            && refusal.find(fragment) != std::string::npos;
        if (!refused) std::fprintf(stderr, "%s: %s\n", message, refusal.c_str());
        check(refused, message);
    };
    // Moving the literal after admission would leave the digest of the
    // unmoved program on the moved one; the helper's recomputed digest
    // refuses that pairing.
    auto patched = request;
    patched.compositionLeaves[1].material->binding.surfaceMaterialDigest
        = leafMaterial(0).binding.surfaceMaterialDigest;
    refuses(patched, "digest", "a leaf 1 literal moved after admission fails the exact digest check");
    auto foreignObject = request;
    foreignObject.compositionLeaves[1].material->binding.object.value = 1u;
    refuses(foreignObject, "Scene Compose refused: imported leaf 1 Render Material names an object outside its own leaf",
        "a leaf 1 Render Material on a leaf 0 object refuses");
    auto unmovedLiteral = request;
    unmovedLiteral.compositionLeaves[1].material = leafMaterial(0);
    unmovedLiteral.compositionLeaves[1].material->binding.object.value = leafOneObject;
    refuses(unmovedLiteral, "Scene Compose refused: imported leaf 1 Render Material ID does not match its object's material",
        "a leaf 1 Render Material with the leaf 0 literal refuses");
    auto leafZeroCopy = request;
    leafZeroCopy.compositionLeaves[0].material = leafMaterial(0);
    refuses(leafZeroCopy, "imported leaf 0 material travels as the request material",
        "a leaf 0 material on the leaf list refuses");
    // A vertex processor on any leaf's Render Material stays refused with
    // several leaves; the plugin refuses it at lowering with the same text.
    auto vertexLeaf = request;
    vertexLeaf.compositionLeaves[1].material->vertexModifier = videohelper::test::audioNormalDisplacement();
    refuses(vertexLeaf, "a material vertex processor with multiple imported leaves is not rendered yet",
        "a leaf 1 Render Material vertex processor refuses before native submission");
}

// A composed single-leaf frame copies the admitted composition leaves into its
// incumbent, so it needs more free composition ledger while leaves are held.
void compositionIncumbentChargesLeaves()
{
    using namespace videohelper::modelpayload;
    using Probe = ImportedAnimatedSceneCompositionProbe;
    using Retention = NativeCompositionRetentionBudget;
    const std::vector<std::uint8_t> starter {
        visualstartermodel::kAnimatedTriangleGlb.begin(), visualstartermodel::kAnimatedTriangleGlb.end() };
    const auto firstKey = exactKey("incumbent-first", starter);
    const auto secondKey = exactKey("incumbent-second", starter);
    Store store;
    check(publish(store, "incumbent-first", firstKey, starter) && publish(store, "incumbent-second", secondKey, starter),
        "incumbent fixtures publish their exact sources");
    const auto procedural = twoLeafProceduralScene();
    if (!procedural) return;
    ImportedAnimatedSceneRequest single;
    single.operation.sourceStableId = 941;
    single.operation.deformationStableId = 942;
    single.operation.schedule = {941, 942};
    single.operation.asset = firstKey;
    single.operation.sceneIndex = 0u;
    single.operation.meshStableId = 1u;
    single.operation.animationClipStableId = 1u;
    single.operation.clipName = std::string(visualstartermodel::kClipName);
    single.proceduralScene = procedural;
    single.structuralRevision = 1; single.evaluationRevision = 1;
    single.frame = {0, 24, 1};
    single.renderClipId = 89; single.projectGeneration = single.helperGeneration = 1;
    single.width = 96; single.height = 64;
    Probe::Input first;
    first.payload = store.resolvePreview(firstKey);
    first.operation = single.operation;
    auto second = first;
    second.payload = store.resolvePreview(secondKey);
    second.operation.asset = secondKey;
    second.operation.sourceStableId = 943;
    std::string error;
    // Smallest free composition ledger with which the frame renders.
    const auto threshold = [&](bool heldLeaves) {
        std::size_t low = 0, high = Retention::kMaximumBytes - sizeof(Retention);
        while (low < high)
        {
            const auto room = low + (high - low) / 2;
            FakeDeformationBackend backend;
            ImportedAnimatedScenePayloadExecution execution(store, backend);
            if (heldLeaves && !Probe::admit(execution, {first, second}, error)) return std::size_t {0};
            auto& ledger = Probe::compositionLedger(execution);
            ledger = std::make_shared<Retention>();
            const auto filler = Retention::reserve(ledger, Retention::kMaximumBytes - ledger->logicalBytes() - room, error);
            ImportedAnimatedSceneReceipt receipt;
            if (filler && execution.executePreview(single, receipt, error)) high = room;
            else low = room + 1;
        }
        return low;
    };
    const auto bare = threshold(false);
    const auto held = threshold(true);
    check(bare > 0 && bare < Retention::kMaximumBytes - sizeof(Retention)
            && held >= bare + 2 * Probe::leafRecordBytes(),
        "a composed frame reserves its copy of every held composition leaf before it admits");
}

#if !defined(_WIN32)
void animatedPayloadAdmissionFitsViewportThreadStack()
{
    const std::vector<std::uint8_t> bytes {
        visualstartermodel::kAnimatedTriangleGlb.begin(),
        visualstartermodel::kAnimatedTriangleGlb.end() };
    const auto key = exactKey ("stack-animated-starter", bytes);
    videohelper::modelpayload::Store store;
    check (publish (store, "stack-animated-transfer", key, bytes),
           "small-stack animated fixture publishes exact bytes");
    FakeDeformationBackend backend;
    videohelper::modelpayload::ImportedAnimatedScenePayloadExecution execution (store, backend);
    videohelper::modelpayload::ImportedAnimatedSceneRequest request;
    request.operation.sourceStableId = 1;
    request.operation.deformationStableId = 2;
    request.operation.schedule = {1, 2};
    request.operation.asset = key;
    request.operation.sceneIndex = 0u;
    request.operation.meshStableId = 1u;
    request.operation.animationClipStableId = 1u;
    request.operation.clipName = std::string (visualstartermodel::kClipName);
    request.frame = {0, 24, 1};
    request.structuralRevision = 1;
    request.width = 96;
    request.height = 64;

    struct AdmissionCase
    {
        videohelper::modelpayload::ImportedAnimatedScenePayloadExecution* execution;
        const videohelper::modelpayload::ImportedAnimatedSceneRequest* request;
        bool admitted = false;
        std::string error;
    } test {&execution, &request, false, {}};
    pthread_attr_t attributes {};
    if (pthread_attr_init (&attributes) != 0)
    {
        check (false, "small-stack animated payload thread attributes initialize");
        return;
    }
    const bool stackSet = pthread_attr_setstacksize (&attributes, 544u * 1024u) == 0;
    check (stackSet, "animated payload viewport-sized POSIX stack is available");
    pthread_t thread {};
    const int created = stackSet ? pthread_create (&thread, &attributes, [] (void* value) -> void*
    {
        auto* test = static_cast<AdmissionCase*> (value);
        videohelper::modelpayload::ImportedAnimatedSceneReceipt receipt;
        test->admitted = test->execution->executePreview (*test->request, receipt, test->error)
            && receipt.valid() && receipt.source->scene->objectCount == 1;
        return nullptr;
    }, &test) : -1;
    pthread_attr_destroy (&attributes);
    check (created == 0, "animated payload viewport-sized POSIX admission thread starts");
    if (created != 0) return;
    if (pthread_join (thread, nullptr) != 0)
    {
        check (false, "animated payload viewport-sized POSIX admission thread joins");
        return;
    }
    check (test.admitted && backend.submissions == 1,
           test.error.empty() ? "animated payload admits on the viewport-sized stack"
                              : test.error.c_str());
}
#endif
} // namespace

std::optional<visualimportedscenerender::Request> smoothGlassDiscreteGratingFixture(std::string& error)
{
    using namespace HarmonicMIDI::grid;
    auto fixture = mixedSurfaceDiffractionFixture(error);
    if (!fixture) return {};
    auto scene = std::make_shared<Visual3DScene>(*fixture->sceneSnapshot);
    const auto slab = videohelper::tests::sceneclosest::slab();
    std::copy_n(slab->vertices.begin(), 8, scene->vertices.begin());
    for (std::size_t vertex = 0; vertex < 8; ++vertex) scene->vertices[vertex].normal = {0,0,1};
    std::copy_n(slab->indices.begin(), 36, scene->indices.begin());
    std::copy_n(fixture->sceneSnapshot->vertices.begin() + 4, 4, scene->vertices.begin() + 8);
    std::copy_n(fixture->sceneSnapshot->indices.begin() + 6, 6, scene->indices.begin() + 36);
    scene->vertexCount = 12; scene->indexCount = 42;
    scene->objects[0].firstVertex = scene->objects[0].firstIndex = 0;
    scene->objects[0].vertexCount = 8; scene->objects[0].indexCount = 36;
    scene->objects[0].transform.translation = {};
    scene->objects[1].firstVertex = 8; scene->objects[1].firstIndex = 36;
    scene->objects[1].transform.translation = {0,0,-6};
    scene->lights[0].spectralSource.version = 1;
    scene->lights[0].spectralSource.wavelengthsNanometres = diffractionmaterial::kProductionWavelengthsNanometres;
    scene->lights[0].spectralSource.radiance.fill(1);
    scene->lights[0].spectralSource.traceMaximumDistance = 20;
    fixture->sceneSnapshot = scene;
    auto& surface = fixture->surfacePrograms.front().program.material;
    surface.sceneSnapshot = scene; surface.version = surfacematerialbinding::kWireVersion;
    surface.program = videohelper::tests::smoothglassfixture::authoredProgram(scene->objects[0].material.value);
    const auto glass = surfacematerial::admit(surface.program, error);
    if (!glass) return {};
    surface.binding.surfaceMaterialDigest = glass->structuralDigest();
    auto& grating = *fixture->diffractionMaterial;
    grating.version = diffractionmaterialbinding::kSceneReceiverWireVersion;
    grating.material.roughness.rmsHeightNanometres = grating.material.roughness.rmsSlope = 0;
    grating.material.microstructure.grooveDepthNanometres = .9f;
    grating.material.spectrum.firstOrder = grating.material.spectrum.lastOrder = 1;
    const auto physical = diffractionmaterial::admit(grating.material, error);
    if (!physical) return {};
    grating.structuralDigest = physical->structuralDigest();
    grating.lighting = {};
    grating.lighting.version = 3; grating.lighting.pathCount = 1; grating.lighting.traceMaximumDistance = 20;
    grating.lighting.sceneReceiverTransport = diffractionmaterial::SceneReceiverTransport::AuthoredSpectralLambertian;
    grating.lighting.paths[0].kind = diffractionmaterial::LightingPathKind::Direct;
    grating.lighting.paths[0].incident.direction = {0,0,-1};
    grating.lighting.paths[0].incident.radiance.fill(1);
    fixture->glassCaustic = visualimportedscenerender::GlassCausticSelection {
        1, scene->lights[0].id, scene->objects[0].id, scene->objects[0].material, {1,1.e-4f,1.e-6f,32,64,16}};
    if (!visualimportedscenerender::valid(*fixture)) return {};
    return fixture;
}

void smoothGlassDiscreteGratingAdmissionChecks()
{
    using namespace videorender::fixture3d;
    using Consumer = arbitgpu::NativeFixtureDiffractionConsumer;
    using Target = videohelper::materialprogram::BackendTarget;
    std::string error;
    const auto fixture = smoothGlassDiscreteGratingFixture(error);
    check(bool(fixture), "ordinary mixed caustic fixture owns real closed glass and authored discrete IR");
    if (!fixture) return;
    const auto& request = *fixture->diffractionMaterial;
    check(!admitDiffractionMaterialBinding(fixture->sceneSnapshot, request, Target::OpenGl, error, true),
        "default mixed admission preserves zero-slope finite-lobe refusal");
    for (const auto target : {Target::OpenGl, Target::Metal})
    {
        const auto binding = admitDiffractionMaterialBinding(fixture->sceneSnapshot, request, target, error, true,
            Consumer::SmoothGlassDiscreteOrders);
        check(binding && binding->consumer() == Consumer::SmoothGlassDiscreteOrders
            && binding->authoredRequest() && binding->materialIR()
            && visualimportedscenerender::detail::sameDiffractionMaterial(*binding->authoredRequest(), request)
            && binding->materialIR()->structuralDigest() == request.structuralDigest
            && binding->nativeProgram()->objectPrograms.empty()
            && binding->nativeProgram()->parameters.identifiers == std::array<std::uint32_t,4>{},
            "explicit discrete compiler retains exact original request, IR and released zero Surface blocks");
    }
    check(!admitDiffractionMaterialBinding(fixture->sceneSnapshot, request, Target::OpenGl, error, false,
        Consumer::SmoothGlassDiscreteOrders), "private discrete admission requires the composed target route");
    FakeBackend backend; backend.backendInfo.backend = "opengl"; backend.spectralReceiverTransport = true;
    videohelper::gltf::GlbNativeRenderSeam seam(backend);
    videohelper::gltf::GlbNativeSceneAdmission admission; admission.scene = fixture->sceneSnapshot;
    check(seam.publishSurfacePrograms(admission, fixture->surfacePrograms, {}, error, &request, &*fixture->glassCaustic)
        && seam.publishSurfacePrograms(admission, fixture->surfacePrograms, {}, error, &request, &*fixture->glassCaustic)
        && backend.preparations == 0, "selected smooth glass publishes and reuses strict discrete material without native work");
    check(!seam.publishSurfacePrograms(admission, fixture->surfacePrograms, {}, error, &request)
        && error.find("immutable") != std::string::npos,
        "same revision cannot replace discrete admission with a different selection identity");
    arbitgpu::NativeFixtureSceneRuntimeInputs inputs; inputs.linearColor = true; inputs.glassCaustic = fixture->glassCaustic;
    videohelper::gltf::GlbNativeFrameReceipt output;
    check(!seam.renderPreview(admission, {32,32}, inputs, output, error)
        && error.find("without pass composition, diffraction") != std::string::npos
        && !output.rendered.nativeFrame && backend.preparations == 0 && backend.submissions == 0,
        "ordinary strict mixed admission reaches the unchanged combined Color guard without pixels");
    inputs.diffractionParameters = {{"grooveDepthNanometres", .8f}};
    check(!seam.renderExport(admission, {32,32}, inputs, output, error)
        && error.find("static smooth glass") != std::string::npos && backend.submissions == 0,
        "discrete binding refuses runtime material modulation before re-lowering");
    for (int fault = 0; fault < 6; ++fault)
    {
        auto invalid = *fixture;
        if (fault == 0) invalid.glassCaustic->glassObject = request.object;
        if (fault == 1) invalid.glassCaustic->sourceLight = {999};
        if (fault == 2) invalid.diffractionMaterial->graphFrame = surfacematerialbinding::TextureSlotBinding::GraphFrameEndpoint{9,0};
        if (fault == 3 || fault == 4)
        {
            auto& material = invalid.surfacePrograms.front().program.material;
            for (auto& operation : material.program.operations)
                if (operation.id == material.program.outputs[fault == 3 ? 2 : 10]) operation.literal[0] = fault == 3 ? .05f : 0;
            const auto physical = surfacematerial::admit(material.program, error);
            if (physical) material.binding.surfaceMaterialDigest = physical->structuralDigest();
        }
        if (fault == 5) invalid.surfacePrograms.front().program.material.sceneSnapshot
            = std::make_shared<const HarmonicMIDI::grid::Visual3DScene>(*fixture->sceneSnapshot);
        FakeBackend refusedBackend; refusedBackend.backendInfo.backend = "opengl";
        videohelper::gltf::GlbNativeRenderSeam refused(refusedBackend);
        check(!refused.publishSurfacePrograms(admission, invalid.surfacePrograms, {}, error,
            &*invalid.diffractionMaterial, &*invalid.glassCaustic) && refusedBackend.preparations == 0,
            "discrete admission refuses overlapping, foreign, dynamic, rough or nondispersive owners");
    }
}

void mixedSurfaceDiffractionAdmissionChecks()
{
    using namespace HarmonicMIDI::grid;
    using namespace videorender::fixture3d;
    std::string error;
    const auto fixture = mixedSurfaceDiffractionFixture(error);
    check(fixture.has_value(), "independent two-patch mixed fixture has exact scene identities");
    if (!fixture) return;
    const auto wire = visualimportedscenerender::encode(*fixture);
    visualimportedscenerender::Request decoded;
    const bool reopened = wire.rfind("visual.imported-scene.render.v18\n", 0) == 0
        && visualimportedscenerender::decode(wire, decoded)
        && visualimportedscenerender::sameRequest(*fixture, decoded);
    check(reopened, "V18 roundtrip retains one exact nonzero grating target and disjoint Surface programs");
    if (!reopened) return;
    const auto& coefficients = fixture->surfacePrograms.front().program.material.program.spectralReflectance->coefficients;
    check(decoded.surfacePrograms.front().program.material.sceneSnapshot == decoded.sceneSnapshot
        && std::memcmp(coefficients.data(), decoded.surfacePrograms.front().program.material.program.spectralReflectance->coefficients.data(), sizeof(coefficients)) == 0,
        "mixed reopen retains the scene owner and all eight reflectance binary32 bit patterns");
    auto legacy = wire;
    legacy.replace(0, std::string("visual.imported-scene.render.v18").size(), "visual.imported-scene.render.v13");
    check(!visualimportedscenerender::decode(legacy, decoded),
        "legacy Surface-only V13 refuses a mixed terminal instead of silently acquiring V18 behavior");
    for (int fault = 0; fault < 9; ++fault)
    {
        auto wrong = *fixture;
        auto& surface = wrong.surfacePrograms.front().program.material;
        if (fault == 0) surface.binding.object = {991};
        if (fault == 1) wrong.diffractionMaterial->object = {992};
        if (fault == 2) --surface.evaluationRevision;
        if (fault == 3) --wrong.diffractionMaterial->sceneRevision;
        if (fault == 4) surface.binding.surfaceMaterialDigest[0] = surface.binding.surfaceMaterialDigest[0] == '0' ? '1' : '0';
        if (fault == 5) wrong.diffractionMaterial->structuralDigest[0] = wrong.diffractionMaterial->structuralDigest[0] == '0' ? '1' : '0';
        if (fault == 6) surface.sceneSnapshot = std::make_shared<const Visual3DScene>(*wrong.sceneSnapshot);
        if (fault == 7) surface.binding.object = {198};
        if (fault == 8) for (auto& operation : surface.program.operations)
            if (operation.id == surface.program.outputs[static_cast<std::size_t>(surfacematerial::OutputSemantic::MaterialId)])
                operation.unsignedLiteral = 124;
        if (fault == 4 || fault == 5) {
            FakeBackend rejectedBackend; rejectedBackend.backendInfo.backend = "opengl";
            videohelper::gltf::GlbNativeRenderSeam rejectedSeam(rejectedBackend);
            videohelper::gltf::GlbNativeSceneAdmission admission; admission.scene = wrong.sceneSnapshot;
            check(!rejectedSeam.publishSurfacePrograms(admission, wrong.surfacePrograms, {}, error, &*wrong.diffractionMaterial)
                && rejectedBackend.preparations == 0, "mixed program digest mismatch refuses at native admission");
        } else check(visualimportedscenerender::encode(wrong).empty(),
            "mixed target overlap, wrong IDs, stale revisions and foreign owners refuse transport");
    }
    auto unsorted = *fixture;
    auto expanded = std::make_shared<Visual3DScene>(*fixture->sceneSnapshot);
    expanded->objects[2] = expanded->objects[0]; expanded->objects[2].id = {300}; expanded->objectCount = 3;
    unsorted.sceneSnapshot = expanded;
    unsorted.surfacePrograms.front().program.material.sceneSnapshot = expanded;
    auto second = unsorted.surfacePrograms.front(); second.program.material.binding.object = {300};
    unsorted.surfacePrograms.push_back(second);
    check(visualimportedscenerender::valid(unsorted), "sorted disjoint exact Surface object programs admit");
    std::swap(unsorted.surfacePrograms[0], unsorted.surfacePrograms[1]);
    check(!visualimportedscenerender::valid(unsorted), "unsorted mixed Surface object programs refuse");

    FakeBackend backend; backend.backendInfo.backend = "opengl";
    videohelper::gltf::GlbNativeRenderSeam seam(backend);
    videohelper::gltf::GlbNativeSceneAdmission admission; admission.scene = fixture->sceneSnapshot;
    videohelper::gltf::GlbNativeFrameReceipt preview, exported;
    check(seam.publishSurfacePrograms(admission, fixture->surfacePrograms, {}, error, &*fixture->diffractionMaterial)
        && seam.renderPreview(admission, {64,64}, preview, error)
        && seam.renderExport(admission, {64,64}, exported, error)
        && backend.preparations == 1 && backend.submittedScene == fixture->sceneSnapshot.get()
        && backend.preparedProgram && backend.preparedProgram->object.value == 991
        && backend.preparedProgram->kind == arbitgpu::NativeFixtureMaterialKind::DiffractionReflective
        && backend.preparedProgram->objectPrograms.size() == 1
        && backend.preparedProgram->objectPrograms[0]->object.value == 197
        && backend.preparedProgram->objectPrograms[0]->kind == arbitgpu::NativeFixtureMaterialKind::SurfacePbr,
        "one production seam prepares both exact object programs against one scene and reuses them for export");
    if (backend.preparedProgram && backend.preparedProgram->objectPrograms.size() == 1) {
        const auto& program = *backend.preparedProgram;
        check(program.objectPrograms[0]->spectralReflectance
            && std::memcmp(coefficients.data(), program.objectPrograms[0]->spectralReflectance->coefficients.data(), sizeof(coefficients)) == 0
            && sizeof(program.objectPrograms[0]->parameters) == 80
            && sizeof(program.diffractionPaths[0]) == 432,
            "mixed native admission preserves companion bits and both released buffer layouts");
        auto wrong = program; wrong.object = {197};
        check(!arbitgpu::validNativeFixtureMaterialPrograms(wrong, *fixture->sceneSnapshot, program.backend),
            "native preparation independently refuses a grating/Surface overlap");
        wrong = program; wrong.programIdentity[0] ^= 1;
        check(!arbitgpu::validNativeFixtureMaterialPrograms(wrong, *fixture->sceneSnapshot, program.backend),
            "native preparation independently refuses a mismatched grating program receipt");
        arbitgpu::opaquevisibility::Prepared query;
        check(arbitgpu::opaquevisibility::prepareScene(*fixture->sceneSnapshot, &program, {}, query, error)
            && query.objectMesh[0] >= 0 && query.objectMesh[1] >= 0,
            "both constant mixed objects enter the existing bounded opaque blocker query");
        auto bad = program; bad.programIdentity[0] ^= 1;
        check(!arbitgpu::opaquevisibility::prepareScene(*fixture->sceneSnapshot, &bad, {}, query, error),
            "opaque blocker classification refuses forged diffraction identity");
        check(!arbitgpu::opaquevisibility::prepareScene(*fixture->sceneSnapshot, &program, {}, query, error,
            arbitgpu::opaquevisibility::AdmissionMode::allSurfaces),
            "mixed publication does not widen the all-surfaces or glass query subset");
    }
    auto stale = fixture->surfacePrograms;
    for (auto& entry : stale) {
        auto& material = entry.program.material;
        material.sceneRevision = material.structuralRevision = material.programRevision = 2;
        material.evaluationRevision = 2; material.binding.surfaceMaterialRevision = 2;
    }
    const auto before = backend.submissions;
    check(!seam.publishSurfacePrograms(admission, stale, {}, error, &*fixture->diffractionMaterial)
        && backend.submissions == before, "stale mixed revisions refuse before any native submission");
    const auto target = videohelper::materialprogram::BackendTarget::OpenGl;
    check(!admitDiffractionMaterialBinding(fixture->sceneSnapshot, *fixture->diffractionMaterial, target, error),
        "legacy standalone admission retains its object-zero refusal");
    auto frame = *fixture; mixedSurfaceFrame(frame);
    check(visualimportedscenerender::valid(frame), "mixed Surface-only Frame input retains its exact endpoint");
    frame.diffractionMaterial->version = diffractionmaterialbinding::kGraphFrameWireVersion;
    frame.diffractionMaterial->graphFrame = surfacematerialbinding::TextureSlotBinding::GraphFrameEndpoint {91,0};
    check(visualimportedscenerender::valid(frame), "equal dual Frame endpoints share the one admitted timeline owner");
    frame.diffractionMaterial->graphFrame->node = 92;
    check(!visualimportedscenerender::valid(frame), "different mixed Frame endpoints refuse instead of sharing a foreign texture");
    frame.diffractionMaterial->graphFrame->node = 91;
    const auto framedGrating = admitDiffractionMaterialBinding(frame.sceneSnapshot, *frame.diffractionMaterial, target, error, true);
    arbitgpu::opaquevisibility::Prepared query;
    check(framedGrating && arbitgpu::opaquevisibility::prepareScene(*frame.sceneSnapshot, framedGrating->nativeProgram().get(), {}, query, error)
        && query.objectMesh[1] < 0, "Frame grating coverage is never inferred to be an opaque blocker");
    const auto framedSurface = admitSurfaceMaterialCollection(frame.sceneSnapshot, frame.surfacePrograms, {}, target, error);
    check(framedSurface != nullptr, "Surface Frame companion retains native admission in a mixed scene");
    if (framedGrating && framedSurface) {
        auto mixed = *framedGrating->nativeProgram();
        mixed.objectPrograms.push_back(framedSurface->nativeProgram());
        check(arbitgpu::validNativeFixtureMaterialPrograms(mixed, *frame.sceneSnapshot, mixed.backend),
            "native mixed table admits equal Frame endpoint owners");
        mixed.frameEndpoint->node = 92;
        check(!arbitgpu::validNativeFixtureMaterialPrograms(mixed, *frame.sceneSnapshot, mixed.backend),
            "native mixed table independently refuses mismatched Frame endpoint owners");
    }
}

void staticImportedObjectModulationTargetsExactSourceNode()
{
    std::string error;
    auto root = triangleRoot();
    root["nodes"].push_back({{"name", "Repeated Triangle"}, {"mesh", 0}, {"translation", {0.75, 0.0, 0.0}}});
    root["scenes"][0]["nodes"].push_back(3);
    const auto bytes = makeGlb(std::move(root), triangleBin());
    const auto key = exactKey("static-repeated-objects", bytes);
    videohelper::modelpayload::Store store;
    check(publish(store, "static-repeated-transfer", key, bytes), "static exact repeated-object bytes publish");
    FakeBackend backend;
    backend.backendInfo.backend = "opengl";
    videohelper::modelpayload::ImportedScenePayloadExecution execution(store, backend);
    videohelper::modelpayload::ImportedSceneRequest request;
    request.asset = key; request.sceneIndex = 0; request.dimensions = {64, 64};
    request.clipId = 901; request.frame = {12, 24, 1};
    request.structuralRevision = 9; request.evaluationRevision = 9;
    request.objectNodeStableId = 4; request.objectMeshStableId = 1;
    request.runtimeInputs.objectTranslationOffset[0] = 0.25f;
    const auto serializedIdentity = [&](const videohelper::modelpayload::ImportedSceneRequest& selected) {
        visualimportedscenerender::Request operation;
        operation.sourceStableId = 1; operation.renderStableId = 2;
        operation.asset = selected.asset; operation.sceneIndex = selected.sceneIndex;
        operation.structuralRevision = selected.structuralRevision;
        operation.evaluationRevision = selected.evaluationRevision;
        operation.objectNodeStableId = selected.objectNodeStableId;
        operation.objectMeshStableId = selected.objectMeshStableId;
        videowire::CompiledVisualLayerPlan plan;
        plan.operations.push_back({1, std::string(visualimportedscenerender::kRenderNodeKind),
            std::string(visualimportedscenerender::kRenderBackendCapability), visualimportedscenerender::encode(operation)});
        return videohelper::importedscene::exactImportedScenePayloadIdentity(plan);
    };
    request.staticPayloadIdentity = serializedIdentity(request);
    check(!request.staticPayloadIdentity.empty(), "static selection uses its genuine canonical V24 production identity");
    const auto publishPreparation = [&](const videohelper::modelpayload::ImportedSceneRequest& selected) {
        // Match the pre-compilation publication in the real preparation path.
        execution.publishStaticPayload(selected.clipId, selected.projectGeneration,
            selected.helperGeneration, selected.staticPayloadIdentity);
    };
    publishPreparation(request);
    videohelper::modelpayload::ImportedSceneExecutionReceipt preview, exported;
    check(execution.executePreview(request, preview, error), "static selected node reaches the real payload execution seam");
    if (!preview.valid()) return;
    const auto scene = preview.admission.scene;
    const auto targeted = backend.lastRuntimeInputs;
    check(scene->objectCount == 2 && targeted.objectNodeStableId == 4
        && !arbitgpu::runtimeTargetsObject(targeted, scene->objects[0])
        && arbitgpu::runtimeTargetsObject(targeted, scene->objects[1]),
        "same-mesh static siblings retain independent exact source-node modulation targets");
    check(execution.executeExport(request, exported, error) && exported.admission.scene == scene
        && backend.lastRuntimeInputs.objectNodeStableId == targeted.objectNodeStableId
        && backend.lastRuntimeInputs.objectTranslationOffset == targeted.objectTranslationOffset
        && backend.preparations == 1,
        "static Preview and Export share one prepared owner and the same selected frame overlay");
    const auto priorFrame = preview.frame.rendered.nativeFrame;
    const auto priorExport = exported.frame.rendered.nativeFrame;
    const auto submissions = backend.submissions;
    const auto refuse = [&](videohelper::modelpayload::ImportedSceneRequest bad, const char* message) {
        const auto node = bad.objectNodeStableId, mesh = bad.objectMeshStableId;
        publishPreparation(bad);
        check(!execution.executePreview(bad, preview, error) && backend.submissions == submissions
            && backend.preparations == 1 && preview.frame.rendered.nativeFrame == priorFrame
            && preview.admission.scene == scene && bad.objectNodeStableId == node && bad.objectMeshStableId == mesh,
            message);
        publishPreparation(bad);
        check(!execution.executeExport(bad, exported, error) && backend.submissions == submissions
            && backend.preparations == 1 && exported.frame.rendered.nativeFrame == priorExport
            && exported.admission.scene == scene, "Export refusal preserves the existing selected-object receipt and owner");
    };
    auto bad = request; bad.objectNodeStableId = 99; bad.staticPayloadIdentity = serializedIdentity(bad);
    refuse(bad, "nonexistent static source node refuses without replacing request, receipt or prepared owner");
    bad = request; bad.objectMeshStableId = 2; bad.staticPayloadIdentity = serializedIdentity(bad);
    refuse(bad, "node-mesh mismatch refuses without rendering or changing the caller's receipt");
    bad = request; bad.sceneIndex = 1; bad.staticPayloadIdentity = serializedIdentity(bad);
    refuse(bad, "a genuine node outside the selected source scene refuses");
    bad = request; ++bad.asset.version;
    refuse(bad, "a selected node cannot reuse differently versioned exact source bytes");
    bad = request; bad.objectMeshStableId = 0;
    refuse(bad, "a naked source node without its mesh fails closed");
    bad = request; bad.objectNodeStableId = 0;
    refuse(bad, "mesh-only half-pair preserves the authorized selected receipt and owner");
    bad = request; bad.objectNodeStableId = 0; bad.objectMeshStableId = 4097;
    refuse(bad, "mesh-only bound failure preserves the authorized selected receipt and owner");
    publishPreparation(request);
    check(execution.executePreview(request, preview, error) && backend.preparations == 1,
        "failed serialized selection replacements preserve the reusable valid static owner");
    publishPreparation(request);
    check(execution.executeExport(request, exported, error) && backend.preparations == 1,
        "Export retries the original serialized pair without another preparation");
    const auto genuineRetirement = [&](const videohelper::modelpayload::ImportedSceneRequest& replacement,
                                       const char* message) {
        const auto preparations = backend.preparations, submissions = backend.submissions;
        publishPreparation(replacement);
        check(!execution.executePreview(replacement, preview, error) && !preview.valid()
            && backend.preparations == preparations && backend.submissions == submissions, message);
        publishPreparation(replacement);
        check(!execution.executeExport(replacement, exported, error) && !exported.valid()
            && backend.preparations == preparations && backend.submissions == submissions,
            "Export clears a receipt whose genuine publication authority retired its owner");
        publishPreparation(request);
        check(execution.executePreview(request, preview, error) && backend.preparations == preparations + 1,
            "retry after genuine retirement admits and prepares a fresh source owner");
        publishPreparation(request);
        check(execution.executeExport(request, exported, error) && backend.preparations == preparations + 1
            && exported.admission.scene == preview.admission.scene,
            "the restored genuine source shares its fresh preparation across Preview and Export");
    };
    bad = request; ++bad.asset.version; bad.objectNodeStableId = 99;
    bad.staticPayloadIdentity = serializedIdentity(bad);
    genuineRetirement(bad, "genuinely published asset-version replacement retires the prior authorized owner and receipt");
    bad = request; bad.asset.contentSha256 = std::string(64, 'c'); bad.objectNodeStableId = 99;
    bad.staticPayloadIdentity = serializedIdentity(bad);
    genuineRetirement(bad, "genuinely published changed content cannot use selected-target retirement deferral");
    bad = request; ++bad.structuralRevision; ++bad.evaluationRevision; bad.objectNodeStableId = 99;
    bad.staticPayloadIdentity = serializedIdentity(bad);
    genuineRetirement(bad, "changed graph revision retains genuine immediate publication retirement");
    bad = request; bad.objectNodeStableId = 99; bad.staticPayloadIdentity = serializedIdentity(bad) + "\n";
    genuineRetirement(bad, "noncanonical serialized publication cannot defer retirement through a partial identity match");
    bad = request; ++bad.projectGeneration; bad.objectNodeStableId = 99;
    bad.staticPayloadIdentity = serializedIdentity(bad);
    genuineRetirement(bad, "a changed project generation retires the static owner before selected-target refusal");
    bad = request; ++bad.helperGeneration; bad.objectNodeStableId = 99;
    bad.staticPayloadIdentity = serializedIdentity(bad);
    genuineRetirement(bad, "a changed helper generation retires the static owner before selected-target refusal");
    backend.backendInfo.device += "-replacement";
    bad = request; bad.objectNodeStableId = 99; bad.staticPayloadIdentity = serializedIdentity(bad);
    genuineRetirement(bad, "a changed native device retires the static owner before selected-target refusal");
    request.objectNodeStableId = request.objectMeshStableId = 0;
    request.runtimeInputs.objectNodeStableId = 4;
    const auto beforeNaked = backend.submissions;
    check(!execution.executePreview(request, exported, error) && backend.submissions == beforeNaked,
        "an arbitrary runtime node cannot create selected static source authority");
    request.runtimeInputs.objectNodeStableId = 0; request.runtimeInputs.objectTranslationOffset = {};
    request.frame = {0, 24, 1};
    request.staticPayloadIdentity = serializedIdentity(request);
    publishPreparation(request);
    check(execution.executePreview(request, preview, error) && backend.lastRuntimeInputs.objectNodeStableId == 0
        && backend.lastRuntimeInputs.objectTranslationOffset == std::array<float, 3>{}
        && arbitgpu::runtimeTargetsObject(backend.lastRuntimeInputs, preview.admission.scene->objects[0])
        && arbitgpu::runtimeTargetsObject(backend.lastRuntimeInputs, preview.admission.scene->objects[1]),
        "selection reset restores legacy whole-scene targeting without frame accumulation");
    videohelper::gltf::GlbNativeRenderSeam seam(backend);
    auto admitted = preview.admission;
    check(!seam.admit(bytes, 1, admitted, error, 4, 1) && admitted.scene == preview.admission.scene,
        "failed exact selection validation leaves the admission destination unchanged");
}

void importedPrototypeSelectionBorrowsExactScene()
{
    using namespace videohelper::gltf;
    auto root = triangleRoot();
    const auto primitive = root["meshes"][0]["primitives"][0];
    root["meshes"][0]["primitives"].push_back(primitive);
    const auto mesh = root["meshes"][0];
    root["meshes"].push_back(mesh);
    root["nodes"].push_back({{"mesh", 0}});
    const auto bytes = makeGlb(std::move(root), triangleBin());
    FakeBackend backend;
    GlbNativeRenderSeam seam(backend);
    auto budget = std::make_shared<videohelper::modelpayload::NativeSceneAnimationCompositionBudget>();
    GlbNativeSceneAdmission admission;
    std::string error;
    check(seam.admit(bytes, 0, admission, error, 1, 1, budget)
        && error.empty() && admission.document && admission.scene->objectCount == 2,
        "real selected GLB admission retains the exact document and two drawn primitives");
    if (!admission.document || !admission.scene || admission.scene->objectCount != 2) return;
    const auto& document = *admission.document;
    const auto& scene = *admission.scene;
    const auto* first = GlbNativeRenderSeam::selectedImportedPrototype(document, scene, 1, 1);
    check(first == &scene.objects[0] && first->indexCount != 0
        && (first->id.value - 1u) / HarmonicMIDI::grid::visualimportedsceneidentity::kObjectPrimitiveStride + 1u == 1,
        "prototype selection borrows the first drawn primitive's address in the exact admitted scene");
    auto changed = std::make_unique<HarmonicMIDI::grid::Visual3DScene>(scene);
    check(GlbNativeRenderSeam::selectedImportedPrototype(document, *changed, 1, 1)
            == &changed->objects[0],
        "an independent supplied scene returns its own address rather than an admission identity witness");
    changed->objects[0].indexCount = 0;
    check(GlbNativeRenderSeam::selectedImportedPrototype(document, *changed, 1, 1)
            == &changed->objects[1],
        "prototype selection skips an undrawn primitive and returns the first remaining drawn primitive");
    changed->objects[1].indexCount = 0;
    check(!GlbNativeRenderSeam::selectedImportedPrototype(document, *changed, 1, 1)
        && !GlbNativeRenderSeam::selectedImportedPrototype(document, scene, 4, 1),
        "undrawn primitives and a genuine node outside the selected scene have no prototype");
    check(!GlbNativeRenderSeam::selectedImportedPrototype(document, scene, 1, 2)
        && !GlbNativeRenderSeam::selectedImportedPrototype(document, scene, 2, 1)
        && !GlbNativeRenderSeam::selectedImportedPrototype(document, scene, 0, 1)
        && !GlbNativeRenderSeam::selectedImportedPrototype(document, scene, 1, 0)
        && !GlbNativeRenderSeam::selectedImportedPrototype(document, scene, 0, 0)
        && !GlbNativeRenderSeam::selectedImportedPrototype(document, scene,
            static_cast<std::uint32_t>(document.nodes.size() + 1u), 1)
        && !GlbNativeRenderSeam::selectedImportedPrototype(document, scene, 1,
            static_cast<std::uint32_t>(document.meshes.size() + 1u)),
        "direct selection refuses wrong mesh, meshless node, zero IDs and out-of-range IDs");
    changed->objectCount = changed->objects.size() + 1u;
    check(!GlbNativeRenderSeam::selectedImportedPrototype(document, *changed, 1, 1),
        "public selection rejects malformed object count before constructing a scene iterator");
    changed->objectCount = 0;
    check(!GlbNativeRenderSeam::selectedImportedPrototype(document, *changed, 1, 1),
        "an empty supplied scene has no drawn prototype");

    const auto retainedScene = admission.scene;
    const auto retainedDocument = admission.document;
    const auto retainedCharge = admission.documentRetention;
    const auto logicalBytes = budget->logicalBytes();
    const auto logicalWork = budget->logicalWork();
    error = "selection diagnostic sentinel";
    check(GlbNativeRenderSeam::selectedImportedPrototype(document, scene, 1, 1) == first
        && error == "selection diagnostic sentinel"
        && budget->logicalBytes() == logicalBytes && budget->logicalWork() == logicalWork
        && backend.preparations == 0 && backend.submissions == 0,
        "borrowed selection does not mutate diagnostics, retained charges or native callback counts");
    const auto refuse = [&](std::optional<std::size_t> selectedScene,
                            std::uint32_t node, std::uint32_t selectedMesh) {
        error = "prior diagnostic";
        return !seam.admit(bytes, selectedScene, admission, error, node, selectedMesh)
            && error == "GLB object selection rejected: node and mesh do not identify a drawn object in the exact source scene"
            && admission.scene == retainedScene && admission.document == retainedDocument
            && admission.documentRetention == retainedCharge
            && backend.preparations == 0 && backend.submissions == 0;
    };
    check(refuse(0, 1, 2) && refuse(0, 4, 1) && refuse(0, 99, 1) && refuse(0, 1, 99)
        && refuse(0, 1, 0) && refuse(0, 0, 1) && refuse(std::nullopt, 1, 1),
        "actual admission preserves selection refusal precedence, exact diagnostic and destination owners");
    GlbNativeSceneAdmission ordinary;
    check(seam.admit(bytes, 0, ordinary, error) && error.empty() && ordinary.scene->objectCount == 2,
        "paired-zero ordinary admission retains its original successful whole-scene route");
}

void staticLocalDocumentAndRefusalBoundaries()
{
    using namespace videohelper::modelpayload;
    using namespace videohelper::gltf;
    auto root = triangleRoot();
    root["nodes"] = nlohmann::json::array({
        {{"children", {1,3,4}}, {"rotation", {0.0,0.0,std::sqrt(.5),std::sqrt(.5)}}},
        {{"mesh", 0}}, {{"mesh", 0}},
        {{"camera", 0}, {"translation", {0.0,0.0,2.0}}},
        {{"extensions", {{"KHR_lights_punctual", {{"light", 0}}}}}}
    });
    root["scenes"][0]["nodes"] = {0,2};
    root["scenes"][1]["nodes"] = {0,2};
    const auto bytes = makeGlb(root, triangleBin());
    FakeBackend backend;
    GlbNativeRenderSeam seam(backend);
    auto budget = std::make_shared<NativeSceneAnimationCompositionBudget>();
    const auto baseline = budget->logicalBytes();
    GlbNativeSceneAdmission admission;
    std::string error;
    check(seam.admit(bytes, 0, admission, error, 0, 0, budget) && admission.document
        && admission.document->nodes.size() == 5 && admission.retainedDocumentBytes > sizeof(*admission.document),
        "local static admission retains and charges the first decoder's complete document");
    const auto* camera = GlbNativeRenderSeam::selectedSourceCamera(admission, 4);
    check(camera && camera->id.value == 4 && !GlbNativeRenderSeam::selectedSourceCamera(admission, 2)
        && backend.preparations == 0 && backend.submissions == 0,
        "selected source camera borrows exact admitted node provenance without native work");
    auto escapedDocument = admission.document;
    admission = {};
    check(budget->logicalBytes() > baseline && escapedDocument->nodes[1].parent == std::optional<std::size_t>{0},
        "an escaped first-document alias retains its original logical charge");
    escapedDocument.reset();
    check(budget->logicalBytes() == baseline, "the document charge refunds only after the last decoded alias");
    {
        GlbNativeSceneAdmission optional;
        check(seam.admit(bytes,0,optional,error,0,0,budget,true) && optional.document
            && optional.sourceScene==optional.scene,
            "compatible ordinary admission retains its first document and source scene under the existing cap");
        auto escapedScene=optional.scene;
        optional={};
        check(escapedScene && budget->logicalBytes()>baseline,
            "an escaped ordinary scene alias keeps the first-document source closure charged");
        escapedScene.reset();
        check(budget->logicalBytes()==baseline,
            "the first-document source closure refunds after its last scene or document alias");
    }
    {
        auto exhausted=NativeSceneAnimationCompositionBudget::reserve(budget,
            NativeSceneAnimationCompositionBudget::kMaximumBytes-budget->logicalBytes(),error);
        GlbNativeSceneAdmission ordinary;
        check(exhausted && seam.admit(bytes,0,ordinary,error,0,0,budget,true)
            && ordinary.valid() && !ordinary.document && !ordinary.sourceScene,
            "optional first-document retention preserves ordinary admission when its existing logical cap is full");
        GlbNativeFrameReceipt original;
        original.nativeBackend="inactive public receipt";
        GlbNativeDrawReceipt inactiveDraw;inactiveDraw.object.value=77;
        original.draws.push_back(inactiveDraw);
        GlbNativeFrameReceipt copy=original;
        GlbNativeFrameReceipt moved=std::move(copy);
        check(moved.draws.size()==1 && moved.draws[0].object.value==77 && moved.nativeBackend==original.nativeBackend
            && !moved.localFrame() && original.draws.size()==1,
            "inactive public receipt copy and move values remain independent of local caller admission");
    }

    Store store;
    const auto key = exactKey("static-local-guard", bytes);
    check(publish(store, "static-local-transfer", key, bytes), "local static fixture publishes exact source bytes");
    ImportedScenePayloadExecution execution(store, backend);
    ImportedSceneRequest request;
    request.asset = key; request.sceneIndex = 0; request.clipId = 71;
    request.sourceStableId = 81; request.renderStableId = 82;
    request.structuralRevision = request.evaluationRevision = request.projectGeneration = request.helperGeneration = 1;
    request.dimensions = {32,32}; request.frame = {0,24,1}; request.sourceCameraNodeId = 4;
    ImportedSceneExecutionReceipt receipt;
    check(execution.executePreview(request, receipt, error), "neutral static fixture establishes the genuine incumbent");
    const auto incumbentScene = receipt.admission.scene;
    check(receipt.admission.document && receipt.admission.sourceScene==incumbentScene,
        "released static admission retains its first document for later local activation");
    request.objectLocalTransform.version = 1; request.objectLocalTransform.nodeStableId = 1;
    request.objectLocalTransform.translation[0] = 1;
    for (const bool exported : {false,true})
    {
        const auto submissionsBefore = backend.submissions;
        const auto staged = exported ? execution.executeExport(request, receipt, error)
                                     : execution.executePreview(request, receipt, error);
        const auto worlds = backend.lastRuntimeInputs.objectWorlds;
        check(staged && backend.submissions == submissionsBefore + 1
            && receipt.admission.scene == incumbentScene
            && worlds && arbitgpu::nativeImportedObjectWorldsCover(*receipt.admission.scene, *worlds),
            "a genuinely evaluated static candidate renders once with object worlds covering the drawn objects");
    }
    // Later refusals are measured against the candidate render that now succeeds.
    const auto renderedNative = receipt.frame.rendered.nativeFrame;
    const auto renderedPreparations = backend.preparations, renderedSubmissions = backend.submissions;
    request.objectLocalTransform.scale = {2,2,2};
    check(!execution.executePreview(request, receipt, error) && error.find("not representable") != std::string::npos
        && receipt.admission.scene == incumbentScene && receipt.frame.rendered.nativeFrame == renderedNative,
        "inherited source-camera scale refuses without coercion or candidate publication");
    request.objectLocalTransform.scale = {1.00005,1,1};
    check(!execution.executePreview(request, receipt, error) && error.find("not representable") != std::string::npos
        && receipt.admission.scene == incumbentScene && receipt.frame.rendered.nativeFrame == renderedNative
        && backend.preparations == renderedPreparations && backend.submissions == renderedSubmissions,
        "small authored camera scale refuses rather than being coerced to unit scale");
    request.objectLocalTransform.scale = {1,1,1};
    // Use separate exact assets so final camera/light component limits are
    // reached by the real decoder, local evaluator and final-record issuance.
    const auto componentRefusal = [&](nlohmann::json source, const char* id, const char* diagnostic) {
        const auto sourceBytes=makeGlb(source, triangleBin());const auto sourceKey=exactKey(id,sourceBytes);
        const std::string transferId = std::string(id) + "-transfer";
        check(publish(store,transferId.c_str(),sourceKey,sourceBytes), "bounded record fixture publishes exact source");
        auto candidate=request;candidate.asset=sourceKey;candidate.objectLocalTransform.translation={0,0,10000};
        check(!execution.executePreview(candidate,receipt,error) && error.find(diagnostic)!=std::string::npos
            && receipt.admission.scene==incumbentScene && receipt.frame.rendered.nativeFrame==renderedNative
            && backend.preparations==renderedPreparations && backend.submissions==renderedSubmissions,
            "transformed camera/light component overflow refuses before record issuance and preserves incumbent");
    };
    auto cameraBound=root;cameraBound["nodes"][3]["translation"]={0,0,999999};
    componentRefusal(cameraBound,"static-local-camera-bound","final camera record");
    auto lightBound=root;lightBound["nodes"][4]["translation"]={0,0,999999};
    componentRefusal(lightBound,"static-local-light-bound","final light record");
    auto inheritedScale=root;inheritedScale["nodes"][0]["scale"]={1.00005,1,1};
    const auto inheritedBytes=makeGlb(inheritedScale,triangleBin());const auto inheritedKey=exactKey("static-local-inherited-scale",inheritedBytes);
    check(publish(store,"static-local-inherited-scale-transfer",inheritedKey,inheritedBytes), "inherited scale fixture publishes exact bytes");
    auto inherited=request;inherited.asset=inheritedKey;
    check(!execution.executePreview(inherited,receipt,error) && error.find("not representable")!=std::string::npos
        && receipt.admission.scene==incumbentScene && receipt.frame.rendered.nativeFrame==renderedNative
        && backend.preparations==renderedPreparations && backend.submissions==renderedSubmissions,
        "tight source rigid interpretation refuses an intentional small inherited camera scale");
    request.objectLocalTransform.meshStableId = 1;
    check(!execution.executePreview(request, receipt, error) && error.find("original node") != std::string::npos,
        "a meshless local parent cannot borrow a mesh-backed same-node selector");
    request.objectLocalTransform = {};
    check(execution.executePreview(request, receipt, error) && receipt.admission.scene == incumbentScene
        && backend.preparations == renderedPreparations,
        "guarded local refusals preserve the released static source preparation");
}

// Checked object worlds name distinct drawn objects. A single leaf lists each of
// its drawn objects exactly once. A Scene Compose may list a subset, and a drawn
// object that the candidate does not list keeps its authored chain.
void staticLocalObjectWorldHelpers()
{
    using arbitgpu::NativeImportedObjectWorlds;
    const std::array<float, 16> unitWorld {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    auto scene = std::make_shared<HarmonicMIDI::grid::Visual3DScene>();
    scene->objectCount = 2;
    scene->objects[0].id.value = 10;
    scene->objects[0].indexCount = 3;
    scene->objects[1].id.value = 11;
    scene->objects[1].indexCount = 0;
    NativeImportedObjectWorlds drawn(1);
    drawn[0].object.value = 10;
    drawn[0].world = unitWorld;
    drawn[0].world[12] = 1.0f;
    check(arbitgpu::nativeImportedObjectWorldsCover(*scene, drawn),
        "one checked world covers the single drawn object");
    const NativeImportedObjectWorlds none;
    check(!arbitgpu::nativeImportedObjectWorldsCover(*scene, none),
        "an empty candidate does not cover a scene with a drawn object");
    NativeImportedObjectWorlds undrawn(1);
    undrawn[0].object.value = 11;
    undrawn[0].world = unitWorld;
    check(!arbitgpu::nativeImportedObjectWorldsCover(*scene, undrawn),
        "an undrawn object is never listed");
    NativeImportedObjectWorlds unknown(1);
    unknown[0].object.value = 99;
    unknown[0].world = unitWorld;
    check(!arbitgpu::nativeImportedObjectWorldsCover(*scene, unknown),
        "a world for an object outside the scene does not cover it");
    auto twoDrawn = std::make_shared<HarmonicMIDI::grid::Visual3DScene>();
    twoDrawn->objectCount = 2;
    twoDrawn->objects[0].id.value = 10;
    twoDrawn->objects[0].indexCount = 3;
    twoDrawn->objects[1].id.value = 11;
    twoDrawn->objects[1].indexCount = 3;
    check(arbitgpu::nativeImportedObjectWorldsCover(*twoDrawn, drawn),
        "a subset of the drawn objects covers the scene");
    check(!arbitgpu::nativeImportedObjectWorldsCover(*twoDrawn, none),
        "an empty candidate does not cover two drawn objects");
    check(arbitgpu::nativeImportedObjectWorldsExactCover(*scene, drawn),
        "one checked world is the exact cover of the single drawn object");
    check(!arbitgpu::nativeImportedObjectWorldsExactCover(*twoDrawn, drawn),
        "a single leaf that lists one of two drawn objects is partial");
    check(!arbitgpu::nativeImportedObjectWorldsExactCover(*twoDrawn, none),
        "an empty list is never an exact cover of drawn objects");
    NativeImportedObjectWorlds bothDrawn(2);
    bothDrawn[0].object.value = 10;
    bothDrawn[0].world = unitWorld;
    bothDrawn[1].object.value = 11;
    bothDrawn[1].world = unitWorld;
    check(arbitgpu::nativeImportedObjectWorldsExactCover(*twoDrawn, bothDrawn),
        "one checked world per drawn object is the exact cover");
    const NativeImportedObjectWorlds repeatedDrawn(2, drawn[0]);
    check(!arbitgpu::nativeImportedObjectWorldsExactCover(*twoDrawn, repeatedDrawn),
        "a repeated entry does not stand in for the second drawn object");

    std::array<float, 16> authored = unitWorld;
    authored[12] = 5.0f;
    check(arbitgpu::nativeObjectWorldMatrix(nullptr, scene->objects[0], authored)[12] == 5.0f,
        "without object worlds every object keeps its authored chain");
    check(arbitgpu::nativeObjectWorldMatrix(&drawn, scene->objects[0], authored)[12] == 1.0f,
        "a listed object takes its checked world");
    check(arbitgpu::nativeObjectWorldMatrix(&drawn, scene->objects[1], authored)[12] == 5.0f,
        "an unlisted object keeps its authored chain");
    check(arbitgpu::nativeObjectWorldMatrix(&drawn, twoDrawn->objects[1], authored)[12] == 5.0f,
        "a drawn object without an entry keeps its authored chain under a subset");

    check(arbitgpu::validNativeImportedObjectWorlds(drawn),
        "one affine finite world with a distinct id is admitted");
    check(!arbitgpu::validNativeImportedObjectWorlds(none),
        "an empty world list is refused");
    const NativeImportedObjectWorlds duplicated(2, drawn[0]);
    check(!arbitgpu::validNativeImportedObjectWorlds(duplicated),
        "a repeated object id is refused");
    NativeImportedObjectWorlds projective = drawn;
    projective[0].world[15] = 2.0f;
    check(!arbitgpu::validNativeImportedObjectWorlds(projective),
        "a world that is not affine is refused");
    NativeImportedObjectWorlds nonfinite = drawn;
    nonfinite[0].world[12] = std::numeric_limits<float>::quiet_NaN();
    check(!arbitgpu::validNativeImportedObjectWorlds(nonfinite),
        "a nonfinite world is refused");

    const std::shared_ptr<const NativeImportedObjectWorlds> sourceWorlds
        = std::make_shared<NativeImportedObjectWorlds>(drawn);
    const std::shared_ptr<const NativeImportedObjectWorlds> copiedWorlds
        = std::make_shared<NativeImportedObjectWorlds>(drawn);
    NativeImportedObjectWorlds movedDrawn = drawn;
    movedDrawn[0].world[12] = 2.0f;
    const std::shared_ptr<const NativeImportedObjectWorlds> movedWorlds
        = std::make_shared<NativeImportedObjectWorlds>(movedDrawn);
    check(arbitgpu::sameNativeImportedObjectWorlds(sourceWorlds, sourceWorlds)
        && arbitgpu::sameNativeImportedObjectWorlds(sourceWorlds, copiedWorlds)
        && !arbitgpu::sameNativeImportedObjectWorlds(sourceWorlds, movedWorlds)
        && !arbitgpu::sameNativeImportedObjectWorlds(sourceWorlds, nullptr),
        "object world identity follows the checked world values, not the allocation");
}

// The Triangle mesh on the root node takes a local translation. Its only drawn
// object then has that translation as its checked world. Preview and export
// both capture it, a changed translation changes the world identity, and a
// neutral request sends none.
void staticLocalSingleObjectWorldContract()
{
    using namespace videohelper::modelpayload;
    const auto bytes = makeGlb(triangleRoot(), triangleBin());
    FakeBackend backend;
    Store store;
    const auto key = exactKey("static-local-single-object-world", bytes);
    check(publish(store, "static-local-single-object-world-transfer", key, bytes),
        "single-object world fixture publishes exact source bytes");
    ImportedScenePayloadExecution execution(store, backend);
    ImportedSceneRequest request;
    request.asset = key; request.sceneIndex = 0; request.clipId = 72;
    request.sourceStableId = 83; request.renderStableId = 84;
    request.structuralRevision = request.evaluationRevision = request.projectGeneration = request.helperGeneration = 1;
    request.dimensions = {32,32}; request.frame = {0,24,1};
    ImportedSceneExecutionReceipt receipt;
    std::string error;
    const auto neutralSubmissions = backend.submissions;
    check(execution.executePreview(request, receipt, error)
        && backend.submissions == neutralSubmissions + 1 && !backend.lastRuntimeInputs.objectWorlds,
        "a neutral static request renders once and sends no object worlds");
    request.objectLocalTransform.version = 1;
    request.objectLocalTransform.nodeStableId = 1;
    request.objectLocalTransform.meshStableId = 1;
    request.objectLocalTransform.translation[0] = 1;
    check(execution.executePreview(request, receipt, error)
        && backend.lastRuntimeInputs.objectWorlds && backend.lastRuntimeInputs.objectWorlds->size() == 1
        && arbitgpu::nativeImportedObjectWorldsCover(*receipt.admission.scene, *backend.lastRuntimeInputs.objectWorlds)
        && nearTranslation(&(*backend.lastRuntimeInputs.objectWorlds)[0].world, 1.0f, 0.0f, 0.0f),
        "the Triangle mesh preview carries its checked local translation");
    const auto previewWorlds = backend.lastRuntimeInputs.objectWorlds;
    request.objectLocalTransform.translation[0] = 2;
    check(execution.executeExport(request, receipt, error)
        && backend.lastRuntimeInputs.objectWorlds && backend.lastRuntimeInputs.objectWorlds->size() == 1
        && nearTranslation(&(*backend.lastRuntimeInputs.objectWorlds)[0].world, 2.0f, 0.0f, 0.0f)
        && !arbitgpu::sameNativeImportedObjectWorlds(previewWorlds, backend.lastRuntimeInputs.objectWorlds),
        "an export carries the changed translation under a different world identity");
    request.objectLocalTransform = {};
    const auto restoredSubmissions = backend.submissions;
    check(execution.executePreview(request, receipt, error)
        && backend.submissions == restoredSubmissions + 1 && !backend.lastRuntimeInputs.objectWorlds,
        "a neutral request after a local one renders once and sends no object worlds");
}

// Motion steps for checked static-local worlds. A moved world steps back to its
// prior world by a similarity. An unchanged or first-frame world keeps the frame
// step, and a draw the runtime does not target has no frame step.
void staticLocalObjectMotionWorldHelpers()
{
    using arbitgpu::NativeImportedObjectWorlds;
    using arbitgpu::NativeSceneSimilarity;
    const auto closeTo = [](float value, float expected) { return std::fabs(value - expected) < 1.0e-5f; };
    const float root = std::sqrt(0.5f);
    const std::array<float, 16> unitWorld {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    std::array<float, 16> movedWorld = unitWorld;
    movedWorld[12] = 1.0f;
    const std::array<float, 16> quarterX {1,0,0,0, 0,0,1,0, 0,-1,0,0, 0,0,0,1};
    const std::array<float, 16> doubled {2,0,0,0, 0,2,0,0, 0,0,2,0, 0,0,0,1};
    const std::array<float, 16> stretched {2,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    const std::array<float, 16> mirrored {-1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    const std::array<float, 16> flattened {0,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};

    // The step takes a current world back to its prior world. It is prior times the inverse of current.
    NativeSceneSimilarity step;
    check(arbitgpu::nativeObjectWorldStep(unitWorld, movedWorld, step)
        && closeTo(step.translation.x, -1.0f) && closeTo(step.translation.y, 0.0f)
        && closeTo(step.translation.z, 0.0f) && closeTo(step.rotation.w, 1.0f)
        && closeTo(step.rotation.x, 0.0f) && closeTo(step.scale, 1.0f),
        "a moved world steps back by its translation");
    check(arbitgpu::nativeObjectWorldStep(quarterX, unitWorld, step)
        && closeTo(step.rotation.x, root) && closeTo(step.rotation.y, 0.0f)
        && closeTo(step.rotation.z, 0.0f) && closeTo(step.rotation.w, root)
        && closeTo(step.scale, 1.0f),
        "a quarter turn about x extracts its half angle quaternion");
    check(arbitgpu::nativeObjectWorldStep(doubled, unitWorld, step)
        && closeTo(step.scale, 2.0f) && closeTo(step.rotation.w, 1.0f),
        "a uniform scale is a similarity");
    check(!arbitgpu::nativeObjectWorldStep(stretched, unitWorld, step),
        "a non-uniform scale has no similarity step");
    check(!arbitgpu::nativeObjectWorldStep(mirrored, unitWorld, step),
        "a reflection has no rotation step");
    check(!arbitgpu::nativeObjectWorldStep(unitWorld, flattened, step),
        "a singular current world has no step");

    NativeImportedObjectWorlds priorList(1);
    priorList[0].object.value = 10;
    priorList[0].world = unitWorld;
    NativeImportedObjectWorlds movedList(1);
    movedList[0].object.value = 10;
    movedList[0].world = movedWorld;
    const auto object = priorList[0].object;
    const HarmonicMIDI::grid::SceneTransform3D identityRuntime {};
    const arbitgpu::NativeSceneMotionUniforms identityFrame {};
    arbitgpu::NativeSceneMotionUniforms frame;
    frame.rotation = { 0.0f, 0.0f, root, root };
    frame.translationScale = { 0.25f, 0.5f, 0.75f, 1.5f };

    const auto kept = arbitgpu::sceneObjectWorldMotionUniforms(frame, identityRuntime, true,
        &priorList, &priorList, object);
    check(kept.rotation == frame.rotation && kept.translationScale == frame.translationScale,
        "an unchanged world keeps the frame step bit for bit");
    const auto firstUniforms = arbitgpu::sceneObjectWorldMotionUniforms(frame, identityRuntime, true,
        nullptr, &movedList, object);
    check(firstUniforms.rotation == frame.rotation
        && firstUniforms.translationScale == frame.translationScale,
        "a first frame with no prior world keeps the frame step");
    const auto untargeted = arbitgpu::sceneObjectWorldMotionUniforms(frame, identityRuntime, false,
        &priorList, &priorList, object);
    check(untargeted.rotation == identityFrame.rotation
        && untargeted.translationScale == identityFrame.translationScale,
        "a draw the runtime does not target has no frame step");
    const auto movedUniforms = arbitgpu::sceneObjectWorldMotionUniforms(identityFrame, identityRuntime,
        true, &priorList, &movedList, object);
    check(closeTo(movedUniforms.translationScale[0], -1.0f)
        && closeTo(movedUniforms.translationScale[1], 0.0f)
        && closeTo(movedUniforms.translationScale[2], 0.0f)
        && closeTo(movedUniforms.translationScale[3], 1.0f) && closeTo(movedUniforms.rotation[3], 1.0f),
        "a moved world carries its own step into the motion uniforms");

    // A runtime turn about z maps the x step onto the y axis.
    HarmonicMIDI::grid::SceneTransform3D turned {};
    turned.rotation = HarmonicMIDI::grid::SceneQuaternion { 0.0f, 0.0f, root, root };
    NativeImportedObjectWorlds quarterList(1);
    quarterList[0].object.value = 10;
    quarterList[0].world = quarterX;
    const auto conjugated = arbitgpu::sceneObjectWorldMotionUniforms(identityFrame, turned, true,
        &quarterList, &priorList, object);
    check(closeTo(conjugated.rotation[0], 0.0f) && closeTo(conjugated.rotation[1], root)
        && closeTo(conjugated.rotation[2], 0.0f) && closeTo(conjugated.rotation[3], root),
        "a runtime turn about z conjugates the object step into the draw frame");

    // The Motion gate admits a changed world only by a similarity.
    auto scene = std::make_shared<HarmonicMIDI::grid::Visual3DScene>();
    scene->objectCount = 1;
    scene->objects[0].id.value = 10;
    scene->objects[0].indexCount = 3;
    arbitgpu::NativeFixtureSceneRuntimeInputs motionCurrent;
    motionCurrent.imageOutput = renderpassoutput::Output::Motion;
    motionCurrent.objectWorlds = std::make_shared<NativeImportedObjectWorlds>(movedList);
    std::string error;
    check(arbitgpu::nativeObjectWorldMotionSupported(*scene, motionCurrent, error),
        "a first Motion frame needs no prior world");
    arbitgpu::NativeImportedSceneRuntimeInputs priorInputs;
    priorInputs.objectWorlds = std::make_shared<NativeImportedObjectWorlds>(priorList);
    auto motionMoved = motionCurrent;
    motionMoved.previousMotion = arbitgpu::sceneMotionSample(priorInputs);
    check(arbitgpu::nativeObjectWorldMotionSupported(*scene, motionMoved, error),
        "a translated world is a similarity and passes the Motion gate");
    NativeImportedObjectWorlds stretchedList(1);
    stretchedList[0].object.value = 10;
    stretchedList[0].world = stretched;
    priorInputs.objectWorlds = std::make_shared<NativeImportedObjectWorlds>(stretchedList);
    auto motionSheared = motionCurrent;
    motionSheared.previousMotion = arbitgpu::sceneMotionSample(priorInputs);
    check(!arbitgpu::nativeObjectWorldMotionSupported(*scene, motionSheared, error)
        && error.find("rotation, uniform scale") != std::string::npos,
        "a changed world without a similarity is refused by name");
    // A scene prior transform on the same object builds on the authored chain the worlds replace.
    scene->previousObjectTransformCount = scene->objectCount;
    scene->previousObjectTransforms[0].translation.x = 5.0f;
    check(!arbitgpu::nativeObjectWorldMotionSupported(*scene, motionMoved, error)
        && error.find("scene object prior transforms") != std::string::npos,
        "an object with a scene prior transform is refused");

    // The history keeps the prior world with the predecessor. Only the drawn object set has to match.
    arbitgpu::NativeSceneMotionHistory history;
    arbitgpu::NativeImportedSceneRuntimeInputs firstFrame;
    firstFrame.objectWorlds = std::make_shared<NativeImportedObjectWorlds>(priorList);
    check(history.commit(0, 24, 1, 32, 32, firstFrame), "a checked first frame commits");
    arbitgpu::NativeImportedSceneRuntimeInputs secondFrame;
    secondFrame.objectWorlds = std::make_shared<NativeImportedObjectWorlds>(movedList);
    const auto predecessorSample = history.predecessor(1, 24, 1, 32, 32, {}, secondFrame);
    check(predecessorSample && predecessorSample->objectWorlds
        && nearTranslation(&(*predecessorSample->objectWorlds)[0].world, 0.0f, 0.0f, 0.0f),
        "a moved world keeps its prior world in the predecessor");
    NativeImportedObjectWorlds otherList(1);
    otherList[0].object.value = 11;
    otherList[0].world = unitWorld;
    arbitgpu::NativeImportedSceneRuntimeInputs otherFrame;
    otherFrame.objectWorlds = std::make_shared<NativeImportedObjectWorlds>(otherList);
    check(!history.predecessor(1, 24, 1, 32, 32, {}, otherFrame).has_value(),
        "a changed object set has no predecessor");
    arbitgpu::NativeSceneMotionHistory authored;
    check(authored.commit(0, 24, 1, 32, 32, arbitgpu::NativeImportedSceneRuntimeInputs {}),
        "an authored first frame commits");
    check(!authored.predecessor(1, 24, 1, 32, 32, {}, secondFrame).has_value(),
        "a checked frame never pairs with an authored predecessor");
}

// The checked worlds of the first Motion frame become the predecessor of the next
// frame. A moved object therefore steps from its prior world. A first Motion frame
// has no predecessor.
void staticLocalMotionCarriesPriorWorld()
{
    using namespace videohelper::modelpayload;
    const auto bytes = makeGlb(triangleRoot(), triangleBin());
    FakeBackend backend;
    Store store;
    const auto key = exactKey("static-local-motion-prior-world", bytes);
    check(publish(store, "static-local-motion-prior-world-transfer", key, bytes),
        "motion world fixture publishes exact source bytes");
    ImportedScenePayloadExecution execution(store, backend);
    ImportedSceneRequest request;
    request.asset = key; request.sceneIndex = 0; request.clipId = 73;
    request.sourceStableId = 85; request.renderStableId = 86;
    request.structuralRevision = request.evaluationRevision = request.projectGeneration = request.helperGeneration = 1;
    request.dimensions = {32,32}; request.frame = {0,24,1};
    request.runtimeInputs.imageOutput = renderpassoutput::Output::Motion;
    request.objectLocalTransform.version = 1;
    request.objectLocalTransform.nodeStableId = 1;
    request.objectLocalTransform.meshStableId = 1;
    request.objectLocalTransform.translation[0] = 1;
    ImportedSceneExecutionReceipt receipt;
    std::string error;
    check(execution.executePreview(request, receipt, error)
        && backend.lastRuntimeInputs.objectWorlds && !backend.lastRuntimeInputs.previousMotion,
        "the first Motion frame carries its checked world and no predecessor");
    const auto firstWorlds = backend.lastRuntimeInputs.objectWorlds;
    request.frame.frame = 1;
    request.objectLocalTransform.translation[0] = 2;
    check(execution.executePreview(request, receipt, error)
        && backend.lastRuntimeInputs.previousMotion && backend.lastRuntimeInputs.previousMotion->objectWorlds
        && arbitgpu::sameNativeImportedObjectWorlds(backend.lastRuntimeInputs.previousMotion->objectWorlds, firstWorlds)
        && nearTranslation(&(*backend.lastRuntimeInputs.previousMotion->objectWorlds)[0].world, 1.0f, 0.0f, 0.0f)
        && nearTranslation(&(*backend.lastRuntimeInputs.objectWorlds)[0].world, 2.0f, 0.0f, 0.0f),
        "a moved Motion frame keeps the prior frame's world as its predecessor");
}

// The rotation extraction takes Shepperd branches 2 to 4. A targeted changed world
// applies the frame after its own step, and an untargeted draw drops the frame.
void staticLocalMotionFrameOrderAndBranches()
{
    using arbitgpu::NativeImportedObjectWorlds;
    const auto closeTo = [](float value, float expected) { return std::fabs(value - expected) < 1.0e-5f; };
    const auto quaternionIs = [&closeTo](const std::array<float, 9>& r, float x, float y, float z, float w) {
        const auto q = arbitgpu::sceneQuaternionFromRotation(r);
        return closeTo(q.x, x) && closeTo(q.y, y) && closeTo(q.z, z) && closeTo(q.w, w);
    };
    check(quaternionIs({1,0,0, 0,-1,0, 0,0,-1}, 1.0f, 0.0f, 0.0f, 0.0f),
        "a half turn about x takes Shepperd branch 2");
    check(quaternionIs({-1,0,0, 0,1,0, 0,0,-1}, 0.0f, 1.0f, 0.0f, 0.0f),
        "a half turn about y takes Shepperd branch 3");
    check(quaternionIs({-1,0,0, 0,-1,0, 0,0,1}, 0.0f, 0.0f, 1.0f, 0.0f),
        "a half turn about z takes Shepperd branch 4");
    check(quaternionIs({0,0,1, 1,0,0, 0,1,0}, 0.5f, 0.5f, 0.5f, 0.5f),
        "a cyclic axis permutation has zero trace and takes Shepperd branch 4");

    const std::array<float, 16> unitWorld {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    std::array<float, 16> movedWorld = unitWorld;
    movedWorld[12] = 1.0f;
    NativeImportedObjectWorlds priorList(1);
    priorList[0].object.value = 10;
    priorList[0].world = unitWorld;
    NativeImportedObjectWorlds movedList(1);
    movedList[0].object.value = 10;
    movedList[0].world = movedWorld;
    const auto object = priorList[0].object;
    const HarmonicMIDI::grid::SceneTransform3D identityRuntime {};
    const float root = std::sqrt(0.5f);
    arbitgpu::NativeSceneMotionUniforms frame;
    frame.rotation = { 0.0f, 0.0f, root, root };
    frame.translationScale = { 0.0f, 0.0f, 2.0f, 1.0f };

    // The own step translates by -1 on x. The frame turns that onto negative y and adds 2 on z.
    // The reversed order would keep x at -1 and y at 0.
    const auto targeted = arbitgpu::sceneObjectWorldMotionUniforms(frame, identityRuntime, true,
        &priorList, &movedList, object);
    check(closeTo(targeted.rotation[0], 0.0f) && closeTo(targeted.rotation[1], 0.0f)
        && closeTo(targeted.rotation[2], root) && closeTo(targeted.rotation[3], root)
        && closeTo(targeted.translationScale[0], 0.0f) && closeTo(targeted.translationScale[1], -1.0f)
        && closeTo(targeted.translationScale[2], 2.0f) && closeTo(targeted.translationScale[3], 1.0f),
        "a targeted changed world applies the frame after its own step");
    // An untargeted draw drops the frame, so only the own step remains.
    const auto untargeted = arbitgpu::sceneObjectWorldMotionUniforms(frame, identityRuntime, false,
        &priorList, &movedList, object);
    check(closeTo(untargeted.rotation[0], 0.0f) && closeTo(untargeted.rotation[1], 0.0f)
        && closeTo(untargeted.rotation[2], 0.0f) && closeTo(untargeted.rotation[3], 1.0f)
        && closeTo(untargeted.translationScale[0], -1.0f) && closeTo(untargeted.translationScale[1], 0.0f)
        && closeTo(untargeted.translationScale[2], 0.0f) && closeTo(untargeted.translationScale[3], 1.0f),
        "an untargeted changed world keeps only its own step");
}

// A repeated frame reuses its own predecessor only while the drawn object set still
// matches. A changed set has no predecessor, so its Motion frame has zero motion.
void staticLocalRepeatedFramePredecessor()
{
    using arbitgpu::NativeImportedObjectWorlds;
    const std::array<float, 16> unitWorld {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    std::array<float, 16> movedWorld = unitWorld;
    movedWorld[12] = 1.0f;
    NativeImportedObjectWorlds firstList(1);
    firstList[0].object.value = 10;
    firstList[0].world = unitWorld;
    NativeImportedObjectWorlds secondList(1);
    secondList[0].object.value = 10;
    secondList[0].world = movedWorld;
    NativeImportedObjectWorlds otherList(1);
    otherList[0].object.value = 11;
    otherList[0].world = movedWorld;

    arbitgpu::NativeSceneMotionHistory history;
    arbitgpu::NativeImportedSceneRuntimeInputs frameZero;
    frameZero.objectWorlds = std::make_shared<NativeImportedObjectWorlds>(firstList);
    check(history.commit(0, 24, 1, 32, 32, frameZero), "a checked frame zero commits");
    arbitgpu::NativeImportedSceneRuntimeInputs frameOne;
    frameOne.objectWorlds = std::make_shared<NativeImportedObjectWorlds>(secondList);
    check(history.commit(1, 24, 1, 32, 32, frameOne), "a checked frame one commits over frame zero");

    // The same drawn set returns frame zero, the predecessor of frame one.
    arbitgpu::NativeImportedSceneRuntimeInputs repeatedSame;
    repeatedSame.objectWorlds = std::make_shared<NativeImportedObjectWorlds>(secondList);
    const auto reused = history.predecessor(1, 24, 1, 32, 32, {}, repeatedSame);
    check(reused && reused->objectWorlds
        && nearTranslation(&(*reused->objectWorlds)[0].world, 0.0f, 0.0f, 0.0f),
        "a repeated frame with the same drawn set reuses frame zero");

    // Repeating frame one after the drawn set changed finds no predecessor.
    arbitgpu::NativeImportedSceneRuntimeInputs repeatedOther;
    repeatedOther.objectWorlds = std::make_shared<NativeImportedObjectWorlds>(otherList);
    const auto changed = history.predecessor(1, 24, 1, 32, 32, {}, repeatedOther);
    check(!changed.has_value(), "a repeated frame with a changed drawn set has no predecessor");
    auto scene = std::make_shared<HarmonicMIDI::grid::Visual3DScene>();
    scene->objectCount = 1;
    scene->objects[0].id.value = 11;
    scene->objects[0].indexCount = 3;
    const HarmonicMIDI::grid::SceneCameraRecord camera {};
    arbitgpu::NativeFixtureSceneRuntimeInputs changedMotion;
    changedMotion.imageOutput = renderpassoutput::Output::Motion;
    changedMotion.objectWorlds = repeatedOther.objectWorlds;
    changedMotion.previousMotion = changed;
    std::string error;
    check(arbitgpu::nativeObjectWorldMotionSupported(*scene, changedMotion, error),
        "a repeated frame without a predecessor is not refused by the Motion gate");
    // Without a predecessor projection w is zero, and the shader falls back to the current clip.
    check(arbitgpu::sceneMotionUniforms(changedMotion, camera, 32, 32).projection[3] == 0.0f,
        "a Motion frame without a predecessor has zero motion");

    arbitgpu::NativeFixtureSceneRuntimeInputs reusedMotion;
    reusedMotion.imageOutput = renderpassoutput::Output::Motion;
    reusedMotion.objectWorlds = repeatedSame.objectWorlds;
    reusedMotion.previousMotion = reused;
    check(arbitgpu::sceneMotionUniforms(reusedMotion, camera, 32, 32).projection[3] == 1.0f,
        "a repeated frame with its predecessor reads the previous clip");
}

// A modulated PBR value inside Scene Compose makes a new procedural snapshot
// on each changed quantum. The owner must recompose from the admitted import:
// no second decode, no new document and no growth of the lifetime ledger.
void composedImportRecomposesWithoutReadmission()
{
    using namespace HarmonicMIDI::grid;
    std::string error;
    const auto bytes = makeGlb(triangleRoot(), triangleBin());
    const auto key = exactKey("composed-pbr-sweep", bytes);
    videohelper::modelpayload::Store store;
    check(publish(store, "composed-pbr-sweep-transfer", key, bytes), "composed sweep exact bytes publish");
    FakeBackend backend;
    backend.backendInfo.backend = "opengl";
    videohelper::modelpayload::ImportedScenePayloadExecution execution(store, backend);
    const auto procedural = [](float red) {
        auto scene = std::make_shared<Visual3DScene>();
        scene->id.value = 7;
        scene->activeCamera.value = 50;
        scene->vertexCount = scene->indexCount = 3;
        scene->vertices[0] = {{-1, -1, 0}, {0, 0, 1}, {}};
        scene->vertices[1] = {{1, -1, 0}, {0, 0, 1}, {}};
        scene->vertices[2] = {{0, 1, 0}, {0, 0, 1}, {}};
        for (std::uint32_t corner = 0; corner < 3; ++corner) scene->indices[corner] = corner;
        scene->materialCount = 1;
        scene->materials[0].id.value = 9000;
        scene->materials[0].baseColor = {red, 0.25f, 0.125f};
        scene->objectCount = 1;
        scene->objects[0].id.value = 10;
        scene->objects[0].material = scene->materials[0].id;
        scene->objects[0].vertexCount = scene->objects[0].indexCount = 3;
        scene->lightCount = 1;
        scene->lights[0].id.value = 40;
        scene->lights[0].kind = SceneLightKind::Directional;
        scene->cameraCount = 1;
        scene->cameras[0].id = scene->activeCamera;
        scene->cameras[0].transform.translation.z = 4;
        return std::shared_ptr<const Visual3DScene>(std::move(scene));
    };
    visualimportedscenerender::Request leaf;
    leaf.sourceStableId = 71; leaf.renderStableId = 72;
    leaf.asset = key; leaf.sceneIndex = 0;
    leaf.structuralRevision = 9; leaf.evaluationRevision = 9;
    videohelper::modelpayload::ImportedSceneRequest request;
    request.asset = key; request.sceneIndex = 0; request.dimensions = {64, 64};
    request.clipId = 902; request.frame = {12, 24, 1};
    request.structuralRevision = 9; request.evaluationRevision = 9;
    request.staticPayloadIdentity = "composed-pbr-sweep-identity";
    request.importedScene = std::make_shared<const visualimportedscenerender::Request>(leaf);
    request.sceneSnapshot = procedural(0.25f);
    videohelper::modelpayload::ImportedSceneExecutionReceipt preview;
    check(execution.executePreview(request, preview, error) && preview.valid() && preview.admission.document,
        "Scene Compose admits the exact import once and retains its document");
    if (!preview.valid() || !preview.admission.document) return;
    const auto document = preview.admission.document;
    const auto ledger = execution.staticLedgerWork();
    check(ledger > 0, "the first admission draws on the lifetime ledger, so the sweep check is meaningful");
    // 512 changed quanta. Each old re-admission charged at least two scene
    // records of lifetime work, so the ledger grew on every step.
    constexpr int kSweep = 512;
    bool recomposed = true, oneAdmission = true, flatLedger = true;
    auto previous = preview.admission.scene;
    for (int step = 1; step <= kSweep; ++step)
    {
        const auto red = 0.25f + static_cast<float>(step) / 1024.0f;
        request.sceneSnapshot = procedural(red);
        if (!execution.executePreview(request, preview, error) || !preview.valid()) { recomposed = false; break; }
        recomposed = recomposed && preview.admission.scene != previous
            && preview.admission.scene->materials[0].id.value == 9000
            && preview.admission.scene->materials[0].baseColor.x == red;
        oneAdmission = oneAdmission && preview.admission.document == document;
        flatLedger = flatLedger && execution.staticLedgerWork() == ledger;
        previous = preview.admission.scene;
    }
    check(recomposed, "each changed procedural snapshot recomposes and carries its PBR value");
    check(oneAdmission, "the sweep keeps the one admitted document, with no second decode");
    check(flatLedger, "the sweep never draws on the lifetime ledger");
    const auto submissions = backend.submissions;
    check(execution.executePreview(request, preview, error) && preview.admission.scene == previous
        && backend.submissions == submissions + 1 && execution.staticLedgerWork() == ledger,
        "an unchanged snapshot reuses the composed scene");
}

// Node 0 is the only drawn object of triangleRoot. By default it has no rest transform,
// so its rest matrix is identity. Its rotation channel turns about +Z from 0 degrees at
// time 0 to 90 degrees at the float key 1/24 s. Frame 1 samples the double 1/24 s, which
// lands about 5e-8 rad short of that key, and the 1e-5 tolerance absorbs the difference.
// With restTurn, node 0 also starts at a 90 degree rest rotation, which the rotation
// channel replaces at every sampled time.
std::vector<std::uint8_t> importedObjectRotationFixture(bool restTurn = false)
{
    auto root = triangleRoot();
    if (restTurn) root["nodes"][0]["rotation"] = {0.0, 0.0, std::sqrt(0.5), std::sqrt(0.5)};
    auto bin = triangleBin();
    bin.resize(104, 0);
    appendFloat(bin, 0.0f);
    appendFloat(bin, 1.0f / 24.0f);
    const float quarter = std::sqrt(0.5f);
    for (const auto value : {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, quarter, quarter})
        appendFloat(bin, value);
    root["buffers"][0]["byteLength"] = bin.size();
    root["bufferViews"].push_back({{"buffer",0}, {"byteOffset",104}, {"byteLength",8}});
    root["bufferViews"].push_back({{"buffer",0}, {"byteOffset",112}, {"byteLength",32}});
    root["accessors"].push_back({{"bufferView",4}, {"componentType",5126}, {"count",2}, {"type","SCALAR"}});
    root["accessors"].push_back({{"bufferView",5}, {"componentType",5126}, {"count",2}, {"type","VEC4"}});
    root["animations"] = nlohmann::json::array({{
        {"name","Imported Object Rotation"},
        {"samplers",nlohmann::json::array({{{"input",4}, {"output",5}, {"interpolation","LINEAR"}}})},
        {"channels",nlohmann::json::array({{{"sampler",0},
            {"target",{{"node",0}, {"path","rotation"}}}}})}
    }});
    return makeGlb(std::move(root), std::move(bin));
}

// A column-major affine world with identity rotation and scale, moved to (x, y, z).
bool nearIdentityWithTranslation(const std::array<float, 16>& world, float x, float y, float z)
{
    constexpr std::array<float, 12> identity {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    for (std::size_t i = 0; i < identity.size(); ++i)
        if (std::abs(world[i] - identity[i]) >= 0.00001f) return false;
    return nearTranslation(&world, x, y, z) && std::abs(world[15] - 1.0f) < 0.00001f;
}

// The correction C = W T(a) W^-1 equals T(R a) for the moved node's turning world W,
// so each frame's drawn world carries the turned offset. The rest matrix is identity,
// which makes the drawn world exactly C.
void importedObjectLocalRigidAnimation(videohelper::modelpayload::Store& payloads)
{
    const auto bytes = importedObjectRotationFixture();
    const auto key = exactKey("local-rigid-animation", bytes);
    check(publish(payloads, key.id.c_str(), key, bytes),
        "rigid animated local fixture publishes its exact source");
    FakeDeformationBackend backend;
    videohelper::modelpayload::ImportedAnimatedScenePayloadExecution execution(payloads, backend);
    videohelper::modelpayload::ImportedAnimatedSceneRequest request;
    request.operation.sourceStableId = 1201;
    request.operation.deformationStableId = 1202;
    request.operation.schedule = {1201, 1202};
    request.operation.asset = key;
    request.operation.sceneIndex = 0;
    request.operation.meshStableId = request.operation.animationClipStableId = 1;
    request.operation.clipName = "Imported Object Rotation";
    request.frame = {0, 24, 1}; request.structuralRevision = 1;
    request.renderClipId = 84; request.projectGeneration = request.helperGeneration = 1;
    request.width = 96; request.height = 64;
    videohelper::modelpayload::ImportedAnimatedSceneReceipt receipt;
    std::string error;
    request.operation.playback.timelineSeconds = 0.0;
    check(execution.executePreview(request, receipt, error) && backend.submissions == 1
        && !backend.lastRuntimeInputs.objectWorlds,
        "a neutral object-local transform keeps the authored chain without object worlds");

    request.objectLocalTransform.version = 1;
    request.objectLocalTransform.nodeStableId = 1;
    request.objectLocalTransform.meshStableId = 1;
    request.objectLocalTransform.translation[0] = 1;
    check(execution.executePreview(request, receipt, error) && backend.submissions == 2
        && backend.lastRuntimeInputs.objectWorlds && backend.lastRuntimeInputs.objectWorlds->size() == 1
        && nearIdentityWithTranslation((*backend.lastRuntimeInputs.objectWorlds)[0].world, 1.0f, 0.0f, 0.0f),
        "frame 0 places the local offset at the unturned rest pose");
    request.frame.frame = 1;
    request.operation.playback.timelineSeconds = 1.0 / 24.0;
    check(execution.executePreview(request, receipt, error) && backend.submissions == 3
        && backend.lastRuntimeInputs.objectWorlds && backend.lastRuntimeInputs.objectWorlds->size() == 1
        && nearIdentityWithTranslation((*backend.lastRuntimeInputs.objectWorlds)[0].world, 0.0f, 1.0f, 0.0f),
        "frame 1 turns the local offset with the moved node's ninety-degree world");
    request.objectLocalTransform = {};
    check(execution.executePreview(request, receipt, error) && backend.submissions == 4
        && !backend.lastRuntimeInputs.objectWorlds,
        "returning to the neutral transform drops the object worlds on the same frame");

    request.objectLocalTransform.version = 1;
    request.objectLocalTransform.nodeStableId = 1;
    request.objectLocalTransform.meshStableId = 1;
    request.objectLocalTransform.translation[0] = 1;
    visualimportedscenerender::ImportedParentIntentV1 parentIntent;
    const auto refuse = [&](const char* message, auto&& edit)
    {
        auto candidate = request;
        edit(candidate);
        videohelper::modelpayload::ImportedAnimatedSceneReceipt rejected;
        std::string rejectedError;
        const auto before = backend.submissions;
        const bool accepted = execution.executePreview(candidate, rejected, rejectedError);
        check(!accepted && rejectedError == message && backend.submissions == before, message);
    };
    refuse("Imported object local transform is malformed",
        [](auto& candidate) { candidate.objectLocalTransform.scale[0] = 0.0; });
    refuse("Imported object local transforms with imported parenting are not rendered yet",
        [&](auto& candidate) { candidate.parentIntent = &parentIntent; });
    refuse("Imported object local transforms with procedural composition are not rendered yet",
        [](auto& candidate) { candidate.proceduralScene = std::make_shared<HarmonicMIDI::grid::Visual3DScene>(); });
    refuse("Imported object local transforms with a selected source camera are not rendered yet",
        [](auto& candidate) { candidate.sourceCameraNodeId = 1u; });
    refuse("Imported object local transforms with note instancing are not rendered yet",
        [](auto& candidate) { candidate.runtimeInputs.noteInstanceMapping = visualnoteinstancing::Mapping {}; });
    refuse("Imported object local transforms with glass caustics are not rendered yet",
        [](auto& candidate) { candidate.runtimeInputs.glassCaustic = visualimportedscenerender::GlassCausticSelection {}; });
    refuse("Imported object local transforms with diffraction gratings are not rendered yet",
        [](auto& candidate) { candidate.runtimeInputs.diffractionSourceProgram = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(); });
    refuse("Imported object local transforms cannot reuse a composition frame or caller object worlds",
        [](auto& candidate) { candidate.runtimeInputs.objectWorlds = std::make_shared<arbitgpu::NativeImportedObjectWorlds>(); });
    refuse("Imported object local transforms with Motion are not rendered yet",
        [](auto& candidate) { candidate.runtimeInputs.imageOutput = renderpassoutput::Output::Motion; });
    refuse("Imported object local transform does not name the selected mesh",
        [](auto& candidate) { candidate.objectLocalTransform.meshStableId = 2; });
    refuse("Imported object local transform does not name a node of the selected leaf",
        [](auto& candidate) { candidate.objectLocalTransform.nodeStableId = 4; candidate.objectLocalTransform.meshStableId = 0; });
    refuse("Imported object local transform names no drawn object of the selected leaf",
        [](auto& candidate) { candidate.objectLocalTransform.nodeStableId = 2; candidate.objectLocalTransform.meshStableId = 0; });
    check(execution.executePreview(request, receipt, error) && backend.submissions == 5
        && backend.lastRuntimeInputs.objectWorlds && backend.lastRuntimeInputs.objectWorlds->size() == 1
        && nearIdentityWithTranslation((*backend.lastRuntimeInputs.objectWorlds)[0].world, 0.0f, 1.0f, 0.0f),
        "the executor renders the local frame again after every refusal");
}

// Every element of a column-major world is within the 1e-5 tolerance of the expected one.
bool nearWorld(const std::array<float, 16>& world, const std::array<float, 16>& expected)
{
    for (std::size_t i = 0; i < expected.size(); ++i)
        if (std::abs(world[i] - expected[i]) >= 0.00001f) return false;
    return true;
}

// The moved node has a 90 degree rest rotation, so its authored chain is M = Rz(90).
// At frame 1 the animated world is W = Rz(90), so C = W T(1, 0, 0) W^-1 = T(0, 1, 0).
// The recorded world is C * M, with translation (0, 1, 0). M * C would move it to (-1, 0, 0).
void importedObjectLocalAuthoredChainOrder(videohelper::modelpayload::Store& payloads)
{
    const auto bytes = importedObjectRotationFixture(true);
    const auto key = exactKey("local-rigid-authored-chain", bytes);
    check(publish(payloads, key.id.c_str(), key, bytes),
        "rest-turned rigid fixture publishes its exact source");
    FakeDeformationBackend backend;
    videohelper::modelpayload::ImportedAnimatedScenePayloadExecution execution(payloads, backend);
    videohelper::modelpayload::ImportedAnimatedSceneRequest request;
    request.operation.sourceStableId = 1301;
    request.operation.deformationStableId = 1302;
    request.operation.schedule = {1301, 1302};
    request.operation.asset = key;
    request.operation.sceneIndex = 0;
    request.operation.meshStableId = request.operation.animationClipStableId = 1;
    request.operation.clipName = "Imported Object Rotation";
    request.operation.playback.timelineSeconds = 1.0 / 24.0;
    request.frame = {1, 24, 1}; request.structuralRevision = 1;
    request.renderClipId = 85; request.projectGeneration = request.helperGeneration = 1;
    request.width = 96; request.height = 64;
    request.objectLocalTransform.version = 1;
    request.objectLocalTransform.nodeStableId = 1;
    request.objectLocalTransform.meshStableId = 1;
    request.objectLocalTransform.translation[0] = 1;
    videohelper::modelpayload::ImportedAnimatedSceneReceipt receipt;
    std::string error;
    constexpr std::array<float, 16> expected {0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1};
    check(execution.executePreview(request, receipt, error) && backend.submissions == 1
        && backend.lastRuntimeInputs.objectWorlds && backend.lastRuntimeInputs.objectWorlds->size() == 1
        && nearWorld((*backend.lastRuntimeInputs.objectWorlds)[0].world, expected),
        "the recorded world is C times the authored rest chain, not the reverse order");
}

int main()
{
    staticImportedObjectModulationTargetsExactSourceNode();
    importedPrototypeSelectionBorrowsExactScene();
    staticLocalDocumentAndRefusalBoundaries();
    staticLocalObjectWorldHelpers();
    staticLocalSingleObjectWorldContract();
    staticLocalObjectMotionWorldHelpers();
    staticLocalMotionCarriesPriorWorld();
    staticLocalMotionFrameOrderAndBranches();
    staticLocalRepeatedFramePredecessor();
    composedImportRecomposesWithoutReadmission();
#if !defined(_WIN32)
    animatedPayloadAdmissionFitsViewportThreadStack();
#endif
    using namespace videohelper::gltf;
    using videorender::fixture3d::RenderDimensions;
    using videorender::fixture3d::RenderUse;

    FakeBackend backend;
    GlbNativeRenderSeam seam (backend);
    GlbNativeSceneAdmission admission;
    std::string error;
    const auto glb = makeGlb (triangleRoot(), triangleBin());

    check (seam.admit (glb, 1, admission, error) && error.empty(),
           "the seam admits a bounded static GLB from caller-owned bytes");
    check (admission.valid() && admission.scene->id.value == 2
           && admission.sourceBytes == glb.size()
           && admission.decodedVertexCount == 3
           && admission.decodedIndexCount == 3
           && admission.scene->objectCount == 1
           && admission.scene->cameraCount == 1
           && admission.scene->lightCount == 1,
           "admission selects one scene and owns the translated native snapshot");

    GlbNativeSceneAdmission animatedAdmission;
    const auto animatedAdmitted = seam.admit (
        visualstartermodel::kAnimatedTriangleGlb.data(),
        visualstartermodel::kAnimatedTriangleGlb.size(),
        std::nullopt, animatedAdmission, error);
    if (! animatedAdmitted)
        std::fprintf (stderr, "animated native seam admission: %s\n", error.c_str());
    check (animatedAdmitted && error.empty() && animatedAdmission.valid()
           && animatedAdmission.metadata.animations == 1
           && animatedAdmission.scene->objectCount == 1,
           "the native seam admits the bounded base scene from an animated GLB");

    GlbNativeFrameReceipt preview;
    GlbNativeFrameReceipt exported;
    check (seam.renderPreview (admission, RenderDimensions {96, 64}, preview, error),
           "preview submits the admitted GLB scene to the native renderer");
    check (seam.renderExport (admission, RenderDimensions {96, 64}, exported, error),
           "export submits through the same native renderer seam");
    check (backend.preparations == 1 && backend.submissions == 2
           && backend.submittedScene == admission.scene.get()
           && preview.rendered.sceneId == exported.rendered.sceneId
           && preview.rendered.use == RenderUse::Preview
           && exported.rendered.use == RenderUse::Export,
           "preview and export consume the exact same immutable scene snapshot");
    check (std::string (preview.supportedSubset)
                == "static-retained-rich-scene-v4"
           && ! preview.downstreamHandoffComplete
           && ! exported.downstreamHandoffComplete
           && preview.rendered.nativeFrame->backend() == "test-native",
           "receipts name the retained rich-scene subset without claiming compositor or encoder handoff");
    check (preview.draws.size() == 1
           && preview.draws[0].object == admission.scene->objects[0].id
           && preview.draws[0].parent == admission.scene->objects[0].parent
           && preview.draws[0].material == admission.scene->objects[0].material
           && preview.materials.size() == 1
           && preview.materials[0].material == admission.scene->materials[0].id
           && preview.materials[0].baseColor.z == admission.scene->materials[0].baseColor.z
           && preview.textures.empty()
           && preview.attributes.tangentWithBitangentSign
           && preview.cameras.size() == admission.scene->cameraCount
           && preview.lights.size() == admission.scene->lightCount
           && preview.activeCamera == admission.scene->activeCamera
           && preview.nativeBackend == "test-native",
           "the frame receipt retains draw hierarchy, object identity, and complete material identity");
    check (preview.rendered.stats.staticUploadCount == 1
           && exported.rendered.stats.staticUploadCount == 0
           && exported.rendered.stats.reusedStaticResources,
           "preview and export share one persistent static GPU resource preparation");

    // Two definitions, with the first definition instanced on two nodes. Their
    // authored intensities and RGB colors are independent expected controls.
    {
        auto root = triangleRoot();
        root["extensionsUsed"] = {"KHR_lights_punctual"};
        root["extensions"]["KHR_lights_punctual"]["lights"] = {
            {{"name", "Red key"}, {"type", "directional"}, {"color", {1, 0, 0}}, {"intensity", 2}},
            {{"name", "Blue fill"}, {"type", "point"}, {"color", {0, 0, 1}}, {"intensity", 5}, {"range", 10}}};
        // Remove any fixture light bindings before adding exact named instances.
        for (auto& node : root["nodes"]) node.erase("extensions");
        const auto first = root["nodes"].size();
        root["nodes"].push_back({{"extensions", {{"KHR_lights_punctual", {{"light", 0}}}}}});
        root["nodes"].push_back({{"extensions", {{"KHR_lights_punctual", {{"light", 1}}}}}, {"translation", {1, 0, 0}}});
        root["nodes"].push_back({{"extensions", {{"KHR_lights_punctual", {{"light", 0}}}}}, {"translation", {-1, 0, 0}}});
        for (auto& scene : root["scenes"])
            for (std::size_t index = first; index < root["nodes"].size(); ++index) scene["nodes"].push_back(index);
        const auto bytes = makeGlb(std::move(root), triangleBin());
        const auto key = exactKey("independent-punctual-lights", bytes);
        videohelper::modelpayload::Store store;
        check(publish(store, "independent-lights-transfer", key, bytes), "two-light exact GLB enters the immutable payload store");
        FakeBackend backend;
        backend.backendInfo.backend = "opengl";
        videohelper::modelpayload::ImportedScenePayloadExecution execution(store, backend);
        videohelper::modelpayload::ImportedSceneRequest request;
        request.asset = key; request.sceneIndex = 1;
        request.dimensions = {96, 64}; request.frame = {0, 24, 1};
        videohelper::modelpayload::ImportedSceneExecutionReceipt preview, exported;
        check(execution.executePreview(request, preview, error) && backend.lastRuntimeInputs.animatedLights.empty(),
              "default imported lighting uses unchanged scene records");
        request.importedLights.selectedDefinition = 1;
        request.importedLights.intensityMultipliers = {{1, 3}};
        const bool redOk = execution.executePreview(request, preview, error);
        auto redLights = backend.lastRuntimeInputs.animatedLights;
        check(redOk && redLights.size() == 3 && redLights[0].intensity == 6
              && redLights[1].intensity == 5 && redLights[2].intensity == 6
              && redLights[0].color.x == 1 && redLights[1].color.z == 1,
              "one selected definition changes all its instances and preserves the other definition");
        const auto* immutableScene = preview.admission.scene.get();
        request.importedLights.selectedDefinition = 2;
        request.importedLights.intensityMultipliers[2] = 0.25f;
        const bool blueOk = execution.executePreview(request, preview, error)
            && execution.executeExport(request, exported, error);
        const auto& lights = backend.lastRuntimeInputs.animatedLights;
        check(blueOk && lights.size() == 3 && lights[0].intensity == 6
              && lights[1].intensity == 1.25f && lights[2].intensity == 6
              && preview.admission.scene.get() == immutableScene
              && exported.admission.scene.get() == immutableScene && backend.preparations == 1
              && immutableScene->lights[0].intensity == 2 && immutableScene->lights[1].intensity == 5,
              "preview and export resolve both saved definitions while retaining original GPU scene ownership");
        const auto submissions = backend.submissions;
        request.importedLights.selectedDefinition = 3;
        check(!execution.executeExport(request, exported, error) && backend.submissions == submissions
              && error.find("missing") != std::string::npos, "missing selected definition cannot reach a native submission");
        request.importedLights.selectedDefinition = 2;
        request.importedLights.intensityMultipliers[2] = std::numeric_limits<float>::infinity();
        check(!execution.executePreview(request, preview, error) && backend.submissions == submissions,
              "nonfinite light multiplier fails before native submission");
        request.importedLights.intensityMultipliers[2] = 0.25f;
        auto different = request;
        ++different.asset.version;
        check(!execution.executePreview(different, preview, error) && backend.submissions == submissions,
              "an override cannot reuse different-version resident model bytes");
        request.light = HarmonicMIDI::grid::SceneLightRecord{};
        request.light->id.value = 999;
        check(!execution.executePreview(request, preview, error) && error.find("replaces imported") != std::string::npos,
              "explicit Render 3D replacement lighting refuses a dead imported-light override");
        request.light.reset();
        request.importedLights = {};
        check(execution.executePreview(request, preview, error) && backend.lastRuntimeInputs.animatedLights.empty(),
              "clearing overrides restores the two definitions' original light content");
    }

    // Product route: exact bytes are first transferred into the helper-owned
    // immutable store, then preview/export resolve and consume that same owner.
    FakeBackend productBackend;
    productBackend.backendInfo.backend = "opengl";
    videohelper::modelpayload::Store payloads;
    videohelper::modelpayload::ImportedScenePayloadExecution execution (
        payloads, productBackend);
    const auto productKey = exactKey ("product-scene", glb);
    auto identityPayload = visualmodelassetpayload::ImmutablePayload::create (
        productKey, std::vector<std::uint8_t> (glb));
    videohelper::modelpayload::ImportedAnimatedSceneAdmissionIdentity firstIdentity {
        identityPayload, std::nullopt, 1u, 1u, 0u, "Morph Pulse", {} };
    auto otherClipIdentity = firstIdentity;
    otherClipIdentity.animationClipStableId = 2u;
    auto otherCrossfadeIdentity = firstIdentity;
    otherCrossfadeIdentity.crossfadeClipStableId = 2u;
    auto otherNameIdentity = firstIdentity;
    otherNameIdentity.crossfadeClipName = "Changed action";
    auto otherMeshIdentity = firstIdentity;
    otherMeshIdentity.meshStableId = 2u;
    auto otherSceneIdentity = firstIdentity;
    otherSceneIdentity.sceneIndex = 1u;
    std::weak_ptr<const visualmodelassetpayload::ImmutablePayload> retainedIdentityPayload
        = identityPayload;
    identityPayload.reset();
    check (firstIdentity != otherClipIdentity
           && firstIdentity != otherCrossfadeIdentity && firstIdentity != otherNameIdentity
           && firstIdentity != otherMeshIdentity
           && firstIdentity != otherSceneIdentity
           && ! retainedIdentityPayload.expired(),
           "animated scene admission identity retains exact bytes and includes scene, mesh, and clip");
    check (publish (payloads, "product-transfer", productKey, glb),
           "the product fixture publishes exact processor-owned bytes");

    FakeDeformationBackend animatedBackend;
    videohelper::modelpayload::ImportedAnimatedScenePayloadExecution animatedExecution (
        payloads, animatedBackend);
    const std::vector<std::uint8_t> starterBytes {
        visualstartermodel::kAnimatedTriangleGlb.begin(),
        visualstartermodel::kAnimatedTriangleGlb.end() };
    const auto starterKey = exactKey ("animated-starter", starterBytes);
    check (publish (payloads, "animated-starter-transfer", starterKey, starterBytes),
           "the animated fixture publishes exact required-extension bytes");
    videohelper::modelpayload::ImportedAnimatedSceneRequest animatedRequest;
    animatedRequest.operation.sourceStableId = 1;
    animatedRequest.operation.deformationStableId = 2;
    animatedRequest.operation.schedule = {1, 2};
    animatedRequest.operation.asset = starterKey;
    animatedRequest.operation.sceneIndex = 0u;
    animatedRequest.operation.meshStableId = 1u;
    animatedRequest.operation.animationClipStableId = 1u;
    animatedRequest.operation.clipName = std::string (visualstartermodel::kClipName);
    animatedRequest.frame = {0, 24, 1};
    animatedRequest.structuralRevision = 1;
    animatedRequest.width = 96;
    animatedRequest.height = 64;
    videohelper::modelpayload::ImportedAnimatedSceneReceipt animatedReceipt;
    const auto animatedExecuted
        = animatedExecution.executePreview (animatedRequest, animatedReceipt, error);
    if (! animatedExecuted)
        std::fprintf (stderr, "animated starter execution rejected: %s\n", error.c_str());
    check (animatedExecuted && animatedReceipt.valid() && animatedBackend.submissions == 1,
           "runtime animation execution admits the preflight-supported punctual-light extension");

    {
        FakeDeformationBackend motionBackend;
        videohelper::modelpayload::ImportedAnimatedScenePayloadExecution motionExecution(payloads, motionBackend);
        auto request = animatedRequest;
        request.renderClipId = 71; request.projectGeneration = 1; request.helperGeneration = 1;
        videohelper::modelpayload::ImportedAnimatedSceneReceipt receipt;
        check(motionExecution.executePreview(request, receipt, error)
            && !motionBackend.lastRuntimeInputs.previousMotion,
            "first animation sample has no retained predecessor");
        const auto first = receipt.evaluation.deformation;
        const auto firstGain = receipt.runtimeInputs.morphWeight;
        request.frame.frame = 1; request.operation.playback.weight = .25;
        request.operation.playback.timelineSeconds = 8;
        request.operation.playback.speed = .5;
        request.operation.playback.offsetSeconds = .25;
        request.operation.playback.playback = visualanimation::Playback::Loop;
        check(motionExecution.executePreview(request, receipt, error)
            && motionBackend.lastRuntimeInputs.previousMotion
            && motionBackend.lastRuntimeInputs.previousMotion->deformation == first
            && motionBackend.lastRuntimeInputs.previousMotion->morphWeight == firstGain
            && receipt.evaluation.deformation != first,
            "adjacent animation keeps immutable prior snapshot and gain after controls change");
        check(motionBackend.lastRuntimeInputs.previousMotion
                && motionBackend.lastRuntimeInputs.previousMotion->importedNodeWorldMatrices
                && importedWorld(*motionBackend.lastRuntimeInputs.previousMotion->importedNodeWorldMatrices, 1),
            "static imported draw uses the admitted base world beside retained morph animation");
        const auto decodeOptions = videohelper::nativeImportedAnimationDecodeOptions(starterBytes.size(), 0u);
        const auto immutableBase = videohelper::gltf::decodeAnimatedGlbBaseScene(starterBytes, decodeOptions.admission, error);
        const auto* staticWorld = motionBackend.lastRuntimeInputs.previousMotion
                && motionBackend.lastRuntimeInputs.previousMotion->importedNodeWorldMatrices
            ? importedWorld(*motionBackend.lastRuntimeInputs.previousMotion->importedNodeWorldMatrices, 1) : nullptr;
        check(immutableBase && staticWorld && *staticWorld == immutableBase->nodes[0].worldTransform,
            "unsampled static node world matches the immutable admitted base matrix exactly");
        check(receipt.evaluation.deformation
            && receipt.evaluation.deformation->time() == request.frame
            && receipt.evaluation.deformation->requestedTimeSeconds() != 1.0 / 24.0,
            "speed/offset/loop sampling preserves exact presentation-frame identity for history");
        const auto second = receipt.evaluation.deformation;
        request.frame.frame = 2; motionBackend.rejectSubmission = true;
        check(!motionExecution.executePreview(request, receipt, error), "failed native animation draw does not commit history");
        motionBackend.rejectSubmission = false;
        request.operation.playback.weight = .5;
        check(motionExecution.executePreview(request, receipt, error)
            && motionBackend.lastRuntimeInputs.previousMotion
            && motionBackend.lastRuntimeInputs.previousMotion->deformation == second,
            "retry after failed draw retains last successful predecessor without reevaluation");
        const auto third = receipt.evaluation.deformation;
        check(motionExecution.executeExport(request, receipt, error)
            && !motionBackend.lastRuntimeInputs.previousMotion, "Export first sample does not borrow Preview history");
        request.frame.frame = 3;
        check(motionExecution.executePreview(request, receipt, error)
            && motionBackend.lastRuntimeInputs.previousMotion
            && motionBackend.lastRuntimeInputs.previousMotion->deformation == third,
            "interleaved Export does not replace Preview immutable predecessor");
        request.frame.frame = 20;
        check(motionExecution.executePreview(request, receipt, error)
            && !motionBackend.lastRuntimeInputs.previousMotion, "nonadjacent animation seek resets predecessor");
        request.frame.frame = 0;
        check(motionExecution.executePreview(request, receipt, error)
            && !motionBackend.lastRuntimeInputs.previousMotion, "animation timeline loop resets predecessor");
        for (int reset = 0; reset < 8; ++reset)
        {
            ++request.frame.frame;
            if (reset == 0) ++request.renderClipId;
            if (reset == 1) ++request.projectGeneration;
            if (reset == 2) ++request.helperGeneration;
            if (reset == 3) ++request.structuralRevision;
            if (reset == 4) request.frame.rateNumerator = 30;
            if (reset == 5) request.width = 80;
            if (reset == 6) request.operation.sourceStableId += 10;
            if (reset == 7) request.operation.deformationStableId += 10;
            request.operation.schedule = {request.operation.sourceStableId, request.operation.deformationStableId};
            check(motionExecution.executePreview(request, receipt, error)
                && !motionBackend.lastRuntimeInputs.previousMotion,
                "render-clip/generation/structure/rate/extent change resets animated history");
        }
        if (receipt.evaluation.deformation)
        {
            auto excessive = std::make_shared<visualdeformation::AnimationDeformationSnapshot>(*receipt.evaluation.deformation);
            auto& joints = const_cast<std::vector<visualdeformation::JointTransformEvaluation>&>(excessive->jointTransforms());
            joints.reserve(arbitgpu::kNativeSceneMotionMaximumSnapshotBytes / sizeof(visualdeformation::JointTransformEvaluation) + 1);
            check(!arbitgpu::sceneMotionSnapshotFits(excessive),
                "oversized snapshot capacity cannot become retained Motion history");
        }
        const auto beforeRefusal = motionBackend.submissions;
        request.runtimeInputs.imageOutput = renderpassoutput::Output::Motion;
        check(!motionExecution.executePreview(request, receipt, error)
            && !receipt.valid() && motionBackend.submissions == beforeRefusal,
            "native backend label without explicit retained-pose capability cannot admit Motion");
        motionBackend.retainedDeformationMotion = true;
        request.frame.frame = 0;
        check(motionExecution.executePreview(request, receipt, error)
            && !motionBackend.lastRuntimeInputs.previousMotion,
            "explicit retained-pose contract admits first neutral Motion frame");
        const auto motionFirst = receipt.evaluation.deformation;
        request.frame.frame = 1;
        check(motionExecution.executePreview(request, receipt, error)
            && motionBackend.lastRuntimeInputs.previousMotion
            && motionBackend.lastRuntimeInputs.previousMotion->deformation == motionFirst,
            "admitted Motion retains exact source-bound preceding deformed pose");
        request.runtimeInputs.noteInstanceMapping = visualnoteinstancing::Mapping {};
        request.runtimeInputs.canonicalBlockCFrame = scoreFieldFrame();
        const auto beforeNotes = motionBackend.submissions;
        check(!motionExecution.executePreview(request, receipt, error) && !receipt.valid()
            && motionBackend.submissions == beforeNotes,
            "combined pose and note Motion requires both explicit consumers");
        motionBackend.retainedNoteMotion = true;
        check(motionExecution.executePreview(request, receipt, error)
            && !motionBackend.lastRuntimeInputs.previousMotion,
            "supported pose-plus-note Motion starts neutral after refused consumer change");
        const auto noteFirst = receipt.evaluation.deformation;
        request.frame.frame = 2;
        request.runtimeInputs.canonicalBlockCFrame = scoreFieldFrame(true);
        check(motionExecution.executePreview(request, receipt, error)
            && motionBackend.lastRuntimeInputs.previousMotion
            && motionBackend.lastRuntimeInputs.previousMotion->deformation == noteFirst
            && motionBackend.lastRuntimeInputs.previousMotion->noteTransforms->find(-102),
            "combined Motion retains prior pose and canonical signed note identities across reorder");
        const auto noteSecond = receipt.evaluation.deformation;
        request.frame.frame = 3; motionBackend.rejectSubmission = true;
        check(!motionExecution.executePreview(request, receipt, error) && !receipt.valid(),
            "failed admitted Motion draw cannot publish preceding receipt");
        motionBackend.rejectSubmission = false;
        check(motionExecution.executePreview(request, receipt, error)
            && motionBackend.lastRuntimeInputs.previousMotion
            && motionBackend.lastRuntimeInputs.previousMotion->deformation == noteSecond,
            "failed admitted Motion draw leaves successful predecessor unchanged for retry");
        ++request.frame.frame; ++request.helperGeneration;
        check(motionExecution.executePreview(request, receipt, error)
            && !motionBackend.lastRuntimeInputs.previousMotion,
            "supported Motion resets on helper generation replacement");
        for (const bool meshTarget : {false, true})
        {
            const auto tracked = starterWithTransformTrack(meshTarget);
            const auto key = exactKey(meshTarget ? "motion-object-track" : "motion-light-track", tracked);
            check(publish(payloads, meshTarget ? "motion-object-transfer" : "motion-light-transfer", key, tracked),
                "Motion classification fixture publishes exact transformed source bytes");
            auto trackedRequest = animatedRequest;
            trackedRequest.operation.asset = key;
            trackedRequest.runtimeInputs.imageOutput = renderpassoutput::Output::Motion;
            const auto beforeTrack = motionBackend.submissions;
            const bool rendered = motionExecution.executePreview(trackedRequest, receipt, error);
            check(meshTarget ? (!rendered && !receipt.valid() && motionBackend.submissions == beforeTrack
                    && error.find("object matrices") != std::string::npos) : rendered,
                "Motion refuses imported object matrices while preserving unrelated light animation");
        }
        const auto lightOnly = starterWithTransformTrack(false, false);
        const auto lightOnlyKey = exactKey("motion-light-only", lightOnly);
        check(publish(payloads, "motion-light-only-transfer", lightOnlyKey, lightOnly),
            "camera/light-only Motion fixture publishes exact source");
        auto lightOnlyRequest = animatedRequest;
        lightOnlyRequest.operation.asset = lightOnlyKey; lightOnlyRequest.operation.meshStableId = 0;
        lightOnlyRequest.runtimeInputs.imageOutput = renderpassoutput::Output::Motion;
        const auto beforeLightOnly = motionBackend.submissions;
        check(!motionExecution.executePreview(lightOnlyRequest, receipt, error) && !receipt.valid()
            && motionBackend.submissions == beforeLightOnly,
            "skin/morph consumer does not broadly enable camera/light-only animation Motion");
    }
    importedObjectMatrixHistory(payloads);
    importedObjectLocalRigidAnimation(payloads);
    importedObjectLocalAuthoredChainOrder(payloads);
    twoAnimatedObjectsOneComposedScene(payloads);
    compositionLeafAdmissionIsExclusiveAndCharged();
    compositionLeafBindingCopyIsCharged();
    singleLeafBindingCopyIsCharged();
    twoLeafCompositionRendersEveryLeaf();
    twoLeafCompositionBindsEveryLeafMaterial();
    compositionIncumbentChargesLeaves();
    {
        FakeBackend noteBackend;
        noteBackend.backendInfo.backend = "opengl";
        videohelper::modelpayload::ImportedScenePayloadExecution noteExecution(payloads, noteBackend);
        videohelper::modelpayload::ImportedSceneRequest request;
        request.asset = productKey; request.sceneIndex = 0u;
        request.dimensions = {96,64}; request.frame = {0,60,1}; request.structuralRevision = 1;
        request.clipId = 72; request.projectGeneration = request.helperGeneration = 1;
        request.runtimeInputs.imageOutput = renderpassoutput::Output::Motion;
        request.runtimeInputs.noteInstanceMapping = visualnoteinstancing::Mapping {};
        request.runtimeInputs.canonicalBlockCFrame = scoreFieldFrame();
        videohelper::modelpayload::ImportedSceneExecutionReceipt receipt;
        check(!noteExecution.executePreview(request, receipt, error) && !receipt.valid()
            && noteBackend.submissions == 0,
            "static imported note Motion requires explicit consumer even for opengl-labelled fixture");
        noteBackend.retainedNoteMotion = true;
        check(noteExecution.executePreview(request, receipt, error) && !noteBackend.lastRuntimeInputs.previousMotion,
            "static imported note Motion admits first neutral frame through explicit fixture contract");
        request.frame.frame = 1; request.runtimeInputs.canonicalBlockCFrame = scoreFieldFrame(true);
        check(noteExecution.executePreview(request, receipt, error) && noteBackend.lastRuntimeInputs.previousMotion
            && noteBackend.lastRuntimeInputs.previousMotion->noteTransforms->find(-1)
            && !noteBackend.lastRuntimeInputs.previousMotion->noteTransforms->find(12345),
            "static note Motion preserves canonical reorder and missing identity neutrality contract");
        check(noteExecution.executeExport(request, receipt, error) && !noteBackend.lastRuntimeInputs.previousMotion,
            "admitted note Motion keeps Preview and Export histories separate");
        request.frame.frame = 2;
        check(noteExecution.executePreview(request, receipt, error) && noteBackend.lastRuntimeInputs.previousMotion,
            "Export execution does not replace admitted Preview note predecessor");
        const auto priorNotes = noteBackend.lastRuntimeInputs.previousMotion->noteTransforms;
        request.frame.frame = 3; noteBackend.rejectSubmission = true;
        check(!noteExecution.executePreview(request, receipt, error) && !receipt.valid(),
            "failed note Motion draw clears caller output");
        noteBackend.rejectSubmission = false;
        check(noteExecution.executePreview(request, receipt, error) && noteBackend.lastRuntimeInputs.previousMotion
            && noteBackend.lastRuntimeInputs.previousMotion->noteTransforms != priorNotes,
            "failed note Motion submission retains last successful sample for adjacent retry");
        ++request.frame.frame; request.runtimeInputs.noteInstanceMapping->xScale = 2;
        check(noteExecution.executePreview(request, receipt, error) && !noteBackend.lastRuntimeInputs.previousMotion,
            "admitted note Motion resets for mapping changes");
        ++request.frame.frame; request.frame.rateNumerator = 30;
        check(noteExecution.executePreview(request, receipt, error) && !noteBackend.lastRuntimeInputs.previousMotion,
            "admitted note Motion resets for rational-rate changes");
        request.frame.frame = 100;
        check(noteExecution.executePreview(request, receipt, error) && !noteBackend.lastRuntimeInputs.previousMotion,
            "admitted note Motion resets for nonadjacent presentation frame");
        request.frame.rateDenominator = 0;
        const auto beforeInvalid = noteBackend.submissions;
        check(!noteExecution.executePreview(request, receipt, error) && !receipt.valid()
            && noteBackend.submissions == beforeInvalid,
            "note Motion refuses missing rational frame rate before native draw");
        request.frame = {0,60,1}; request.asset = {}; request.sceneIndex.reset();
        videohelper::gltf::GlbNativeRenderSeam composedSeam(noteBackend);
        videohelper::gltf::GlbNativeSceneAdmission composedAdmission;
        check(composedSeam.admit(glb, 0u, composedAdmission, error), "composed Motion refusal fixture admits static scene");
        request.sceneSnapshot = composedAdmission.scene; request.evaluationRevision = 1;
        check(!noteExecution.executePreview(request, receipt, error) && !receipt.valid()
            && noteBackend.submissions == beforeInvalid,
            "composed prototype note Motion remains refused without admitted ancestry authority");
    }
    {
        arbitgpu::NativeSceneMotionHistory history;
        arbitgpu::NativeImportedSceneRuntimeInputs inputs;
        inputs.canonicalBlockCFrame = scoreFieldFrame();
        inputs.noteInstanceMapping = visualnoteinstancing::Mapping {};
        check(history.commit(0, 60, 1, 640, 360, inputs), "bounded note predecessor commits after successful draw");
        const auto retained = history.current->noteTransforms;
        const auto originalBatch = arbitgpu::prepareNativeNoteInstances(inputs);
        inputs.canonicalBlockCFrame = scoreFieldFrame(true, false, true, 32);
        const auto currentBatch = arbitgpu::prepareNativeNoteInstances(inputs);
        const auto previous = history.predecessor(1, 60, 1, 640, 360, {}, inputs);
        check(previous && previous->noteTransforms == retained && retained->count == originalBatch.count,
            "reordered score rows retain the immutable note transform owner");
        for (std::size_t row = 0; row < currentBatch.count; ++row)
        {
            const auto transform = retained->find(currentBatch.identities[row]);
            bool exact = false;
            for (std::size_t prior = 0; prior < originalBatch.count; ++prior)
                if (originalBatch.identities[prior] == currentBatch.identities[row])
                    exact = transform && *transform == originalBatch.transforms[prior];
            check(exact, "previous note lookup uses canonical signed stable identity, never row order");
        }
        check(!retained->find(0) && !retained->find(12345), "missing/new note identity has neutral Motion contract");
        inputs.previousMotion = *previous;
        check(arbitgpu::validFixtureRuntimeInputs(inputs), "retained native note predecessor passes shared validation");
        inputs.previousMotion->morphWeight = std::numeric_limits<float>::quiet_NaN();
        check(!arbitgpu::validFixtureRuntimeInputs(inputs), "nonfinite prior morph gain is refused before native draw");
        inputs.previousMotion = *previous;
        auto invalidNotes = std::make_shared<arbitgpu::NativeSceneNoteMotionTransforms>(*retained);
        inputs.previousMotion->noteTransforms = invalidNotes;
        invalidNotes->count = invalidNotes->identities.size() + 1;
        check(!arbitgpu::validFixtureRuntimeInputs(inputs), "oversized prior note table is refused before indexing");
        *invalidNotes = *retained;
        invalidNotes->identities[0] = 0;
        check(!arbitgpu::validFixtureRuntimeInputs(inputs), "zero prior note identity is refused");
        *invalidNotes = *retained;
        invalidNotes->count = 2; invalidNotes->identities[1] = invalidNotes->identities[0];
        check(!arbitgpu::validFixtureRuntimeInputs(inputs), "duplicate prior note identity is refused");
        *invalidNotes = *retained;
        invalidNotes->transforms[0][0] = std::numeric_limits<float>::infinity();
        check(!arbitgpu::validFixtureRuntimeInputs(inputs), "nonfinite prior note translation is refused");
        *invalidNotes = *retained; invalidNotes->transforms[0][3] = 0;
        check(!arbitgpu::validFixtureRuntimeInputs(inputs), "nonpositive prior note scale is refused");
        *invalidNotes = *retained; invalidNotes->transforms[0][3] = 10001;
        check(!arbitgpu::validFixtureRuntimeInputs(inputs), "out-of-mapping-bound prior note scale is refused");
        inputs.previousMotion.reset();
        inputs.noteInstanceMapping->xScale = 2;
        check(!history.predecessor(1, 60, 1, 640, 360, {}, inputs), "changed note mapping resets predecessor");
        inputs.noteInstanceMapping->xScale = 1;
        auto changedEpoch = history;
        ++changedEpoch.noteOwner->seekGeneration;
        check(!changedEpoch.predecessor(1, 60, 1, 640, 360, {}, inputs), "canonical seek epoch prevents predecessor reuse");
        changedEpoch = history; ++changedEpoch.noteOwner->loopGeneration;
        check(!changedEpoch.predecessor(1, 60, 1, 640, 360, {}, inputs), "canonical loop epoch prevents predecessor reuse");
        arbitgpu::NativeSceneMotionOwner owner {{1,2,3,4,5,6,7,8}};
        check(!history.predecessor(1, 60, 1, 640, 360, owner, inputs), "source/deformation/clip owner identity cannot borrow note history");
        check(!history.predecessor(std::numeric_limits<std::int64_t>::max(), 60, 1, 640, 360, {}, inputs),
            "extreme rational frame does not overflow adjacency");
        check(!history.commit(1, 0, 1, 640, 360, inputs) && history.frame == 0,
            "invalid candidate frame leaves retained success unchanged");
        if (animatedReceipt.evaluation.deformation)
            check(!history.commit(1, 60, 1, 640, 360, inputs, animatedReceipt.evaluation.deformation)
                && history.frame == 0, "snapshot presentation identity must match exact rational draw frame");
    }

    {
        FakeDeformationBackend controlledDeformation;
        FakeBackend controlledScene;
        controlledScene.backendInfo.backend = "opengl";
        videohelper::modelpayload::ImportedScenePayloadExecution controlledExecution(
            payloads, controlledScene, controlledDeformation);
        videohelper::modelpayload::ImportedSceneRequest request;
        request.asset = starterKey; request.sceneIndex = 0;
        request.dimensions = {96, 64}; request.frame = {12, 24, 1};
        request.structuralRevision = 1; request.evaluationRevision = 1;
        request.deformation = animatedRequest.operation;
        request.deformation->playback.timelineSeconds = 0.5;
        request.importedLights.selectedDefinition = 1;
        request.importedLights.intensityMultipliers = {{1, 3}};
        videohelper::modelpayload::ImportedSceneExecutionReceipt preview, exported;
        const bool ok = controlledExecution.executePreview(request, preview, error)
            && controlledExecution.executeExport(request, exported, error);
        check(ok && preview.deformationFrame && exported.deformationFrame
              && preview.deformationFrame->runtimeInputs.animatedLights.size() == 1
              && exported.deformationFrame->runtimeInputs.animatedLights.size() == 1
              && preview.deformationFrame->runtimeInputs.animatedLights[0].intensity == 6147
              && exported.deformationFrame->runtimeInputs.animatedLights[0].intensity == 6147
              && preview.deformationFrame->source == exported.deformationFrame->source
              && preview.deformationFrame->source->scene->lights[0].intensity == 2049,
              "authored starter intensity 2049 is multiplied after animation identically for preview/export without rewriting its snapshot");
        request.importedLights.selectedDefinition = 2;
        check(!controlledExecution.executeExport(request, exported, error)
              && error.find("missing") != std::string::npos,
              "animated scene rejects a missing exact imported-light definition");
    }

    {
        FakeDeformationBackend selectedDeformation;
        FakeBackend selectedScene;
        selectedScene.backendInfo.backend = "opengl";
        videohelper::modelpayload::ImportedScenePayloadExecution selectedExecution(
            payloads, selectedScene, selectedDeformation);
        videohelper::modelpayload::ImportedSceneRequest request;
        request.asset = starterKey; request.sceneIndex = 0; request.clipId = 902;
        request.dimensions = {96, 64}; request.frame = {12, 24, 1};
        request.structuralRevision = request.evaluationRevision = 1;
        request.deformation = animatedRequest.operation;
        request.deformation->playback.timelineSeconds = 0.5;
        request.objectNodeStableId = 1; request.objectMeshStableId = 1;
        request.deformation->pose.nodeStableId = 1; request.deformation->pose.meshStableId = 1;
        request.deformation->pose.morphEnabled = true;
        request.deformation->pose.morphTargetStableId = 1; request.deformation->pose.morphWeight = 0.8;
        const auto serialized = [&](const videohelper::modelpayload::ImportedSceneRequest& selected) {
            visualimportedscenerender::Request operation;
            operation.sourceStableId = animatedRequest.operation.sourceStableId;
            operation.renderStableId = 3;
            operation.asset = selected.asset; operation.sceneIndex = selected.sceneIndex;
            operation.structuralRevision = selected.structuralRevision;
            operation.evaluationRevision = selected.evaluationRevision;
            operation.objectNodeStableId = selected.objectNodeStableId;
            operation.objectMeshStableId = selected.objectMeshStableId;
            operation.deformation = selected.deformation;
            return visualimportedscenerender::encode(operation);
        };
        request.staticPayloadIdentity = serialized(request);
        check(!request.staticPayloadIdentity.empty(), "selected morph fixture has a canonical matching outer/deformation pair");
        const auto publishPreparation = [&](const videohelper::modelpayload::ImportedSceneRequest& selected) {
            selectedExecution.publishStaticPayload(selected.clipId, selected.projectGeneration,
                selected.helperGeneration, selected.staticPayloadIdentity);
        };
        videohelper::modelpayload::ImportedSceneExecutionReceipt preview, exported;
        const auto restore = [&] {
            publishPreparation(request);
            check(selectedExecution.executePreview(request, preview, error) && preview.valid() && preview.deformationFrame
                && preview.deformationFrame->runtimeInputs.objectNodeStableId == 1,
                "matching selected morph pair retains ordinary valid Preview deformation transport");
            publishPreparation(request);
            check(selectedExecution.executeExport(request, exported, error) && exported.valid() && exported.deformationFrame
                && exported.deformationFrame->runtimeInputs.objectNodeStableId == 1,
                "matching selected morph pair retains ordinary valid Export deformation transport");
        };
        restore();
        const auto preparations = selectedDeformation.preparations, submissions = selectedDeformation.submissions;
        auto unresident = request;
        ++unresident.asset.version; unresident.deformation->asset = unresident.asset;
        unresident.staticPayloadIdentity = serialized(unresident);
        check(!unresident.staticPayloadIdentity.empty() && unresident.staticPayloadIdentity != request.staticPayloadIdentity,
            "unresident animated version publishes its genuinely changed coherent canonical identity");
        publishPreparation(unresident);
        check(!selectedExecution.executePreview(unresident, preview, error) && !preview.valid() && !preview.deformationFrame
            && error == "imported animated scene exact payload is not resident"
            && selectedDeformation.preparations == preparations && selectedDeformation.submissions == submissions,
            "selected animated unresident-source Preview refusal clears its caller as the legacy route does");
        publishPreparation(unresident);
        check(!selectedExecution.executeExport(unresident, exported, error) && !exported.valid() && !exported.deformationFrame
            && error == "imported animated scene exact payload is not resident"
            && selectedDeformation.preparations == preparations && selectedDeformation.submissions == submissions,
            "selected animated unresident-source Export refusal clears its caller as the legacy route does");
        restore();
        selectedDeformation.rejectSubmission = true;
        check(!selectedExecution.executePreview(request, preview, error) && !preview.valid() && !preview.deformationFrame
            && error.find("fake deformation draw failed") != std::string::npos,
            "refused selected animated native Preview submission cannot retain the preceding caller receipt");
        check(!selectedExecution.executeExport(request, exported, error) && !exported.valid() && !exported.deformationFrame
            && error.find("fake deformation draw failed") != std::string::npos,
            "refused selected animated native Export submission cannot retain the preceding caller receipt");
        selectedDeformation.rejectSubmission = false;
        restore();
        auto invalidStatic = request;
        invalidStatic.deformation.reset(); invalidStatic.objectNodeStableId = 99;
        invalidStatic.staticPayloadIdentity = serialized(invalidStatic);
        const auto staticPreparations = selectedScene.preparations, staticSubmissions = selectedScene.submissions;
        publishPreparation(invalidStatic);
        check(!selectedExecution.executePreview(invalidStatic, preview, error) && !preview.valid() && !preview.deformationFrame
            && error.find("GLB object selection rejected") != std::string::npos
            && selectedScene.preparations == staticPreparations && selectedScene.submissions == staticSubmissions,
            "animated-to-static invalid selection clears the old animated Preview receipt before refusal");
        publishPreparation(invalidStatic);
        check(!selectedExecution.executeExport(invalidStatic, exported, error) && !exported.valid() && !exported.deformationFrame
            && error.find("GLB object selection rejected") != std::string::npos
            && selectedScene.preparations == staticPreparations && selectedScene.submissions == staticSubmissions,
            "animated-to-static invalid selection clears the old animated Export receipt before refusal");
    }

    const auto noLightBytes
        = withoutStarterLightBinding (visualstartermodel::kAnimatedTriangleGlb);
    const auto noLightKey = exactKey ("animated-no-light", noLightBytes);
    check (publish (payloads, "animated-no-light-transfer", noLightKey, noLightBytes),
           "the downstream-rejection fixture publishes exact bytes");
    auto rejectedAnimatedRequest = animatedRequest;
    rejectedAnimatedRequest.operation.asset = noLightKey;
    const auto ambientAnimatedExecuted = animatedExecution.executePreview (
        rejectedAnimatedRequest, animatedReceipt, error);
    if (!ambientAnimatedExecuted)
        std::fprintf (stderr, "ambient animated scene rejected: %s\n", error.c_str());
    check (ambientAnimatedExecuted && error.empty() && animatedBackend.submissions == 2,
           "animated scenes preserve native ambient-only lighting without a single-light restriction");
    check (animatedExecution.executePreview (animatedRequest, animatedReceipt, error)
           && animatedReceipt.valid() && animatedBackend.submissions == 3,
           "switching scene resources preserves exact animation request identity");

    for (const bool indexed : {false, true})
        for (const int deformationMode : {0, 1, 2})
    {
        if (!indexed && deformationMode != 0) continue;
        // An authored YZ triangle has no NORMAL and rotates around Z.
        // Its generated +X normal must survive admission and the skin palette.
        auto root = triangleRoot();
        auto bin = triangleBin();
        const std::array<float, 9> positions {{0, 0, 0, 0, 1, 0, 0, 0, 1}};
        std::memcpy(bin.data(), positions.data(), sizeof(positions));
        root["meshes"][0]["primitives"][0]["attributes"].erase("NORMAL");
        root["meshes"][0]["primitives"][0].erase("indices");
        bin.resize(104, 0);
        const auto appendAccessor = [&](std::size_t offset, std::size_t bytes, int component,
                                        const char* type, std::size_t count, bool vertex)
        {
            auto view = nlohmann::json{{"buffer", 0}, {"byteOffset", offset}, {"byteLength", bytes}};
            if (vertex) view["target"] = 34962;
            const auto viewIndex = root["bufferViews"].size();
            root["bufferViews"].push_back(std::move(view));
            const auto index = root["accessors"].size();
            root["accessors"].push_back({{"bufferView", viewIndex}, {"componentType", component},
                                        {"count", count}, {"type", type}});
            return index;
        };
        auto offset = bin.size();
        bin.insert(bin.end(), 12, 0);
        const auto joints = appendAccessor(offset, 12, 5121, "VEC4", 3, true);
        offset = bin.size();
        for (int vertex = 0; vertex < 3; ++vertex)
            for (const auto value : {1.0f, 0.0f, 0.0f, 0.0f}) appendFloat(bin, value);
        const auto weights = appendAccessor(offset, 48, 5126, "VEC4", 3, true);
        offset = bin.size();
        appendFloat(bin, 0.0f); appendFloat(bin, 1.0f);
        const auto times = appendAccessor(offset, 8, 5126, "SCALAR", 2, false);
        offset = bin.size();
        for (const auto value : {0.0f, 0.0f, 0.0f, 1.0f,
                                 0.0f, 0.0f, std::sqrt(0.5f), std::sqrt(0.5f)})
            appendFloat(bin, value);
        const auto rotations = appendAccessor(offset, 32, 5126, "VEC4", 2, false);
        if (indexed)
        {
            offset = bin.size();
            for (const auto index : {0u, 1u, 2u, 0u, 1u, 2u}) appendU16(bin, static_cast<std::uint16_t>(index));
            const auto indices = appendAccessor(offset, 12, 5123, "SCALAR", 6, false);
            root["bufferViews"].back()["target"] = 34963;
            root["meshes"][0]["primitives"][0]["indices"] = indices;
        }
        root["buffers"][0]["byteLength"] = bin.size();
        auto& attributes = root["meshes"][0]["primitives"][0]["attributes"];
        attributes["JOINTS_0"] = joints;
        attributes["WEIGHTS_0"] = weights;
        root["nodes"][0]["skin"] = 0;
        root["nodes"].push_back({{"name", "Turn joint"}});
        for (auto& selectedScene : root["scenes"]) selectedScene["nodes"].push_back(3);
        root["skins"] = nlohmann::json::array({{{"joints", {3}}}});
        root["animations"] = nlohmann::json::array({{
            {"name", "Flat turn"},
            {"samplers", nlohmann::json::array({{{"input", times}, {"output", rotations},
                                                {"interpolation", "LINEAR"}}})},
            {"channels", nlohmann::json::array({{{"sampler", 0},
                {"target", {{"node", 3}, {"path", "rotation"}}}}})}
        }});
        if (deformationMode != 0)
        {
            // Remove skinning so this fixture reaches retained transform-only
            // synthesis rather than the ordinary deformation mesh decoder.
            root.erase("skins");
            root["nodes"][0].erase("skin");
            attributes.erase("JOINTS_0");
            attributes.erase("WEIGHTS_0");
            root["animations"][0]["channels"][0]["target"]["node"] = deformationMode == 1 ? 0 : 3;
            if (deformationMode == 2)
            {
                root["nodes"][3]["children"] = {0};
                for (auto& selectedScene : root["scenes"])
                    selectedScene["nodes"].erase(selectedScene["nodes"].begin());
            }
        }
        const auto bytes = makeGlb(root, bin);
        const auto fixtureId = deformationMode == 1 ? "indexed-flat-normal-rigid-node"
            : deformationMode == 2 ? "indexed-flat-normal-rigid-ancestor"
            : indexed ? "indexed-flat-normal-skinned-triangle" : "flat-normal-skinned-triangle";
        const auto key = exactKey(fixtureId, bytes);
        check(publish(payloads, fixtureId, key, bytes),
              "missing-normal skinned and transform-only triangles publish their exact authored bytes");
        videohelper::modelpayload::ImportedAnimatedSceneCompatibility compatibility;
        check(videohelper::modelpayload::ImportedAnimatedScenePayloadExecution::inspectCompatibility(
                  payloads.resolvePreview(key), 0u, compatibility, error)
              && compatibility.clips.size() == 1 && compatibility.clips[0].name == "Flat turn"
              && compatibility.clips[0].compatibleMeshStableIds == std::vector<visualanimationimport::StableId>{1},
              "missing-normal skin and transform-only animation enter the production import compatibility route");
        FakeDeformationBackend flatBackend;
        videohelper::modelpayload::ImportedAnimatedScenePayloadExecution flatExecution(payloads, flatBackend);
        auto request = animatedRequest;
        request.operation.asset = key;
        request.operation.clipName = "Flat turn";
        request.operation.playback.timelineSeconds = 1.0;
        request.frame = {24, 24, 1};
        videohelper::modelpayload::ImportedAnimatedSceneReceipt preview, exported;
        const auto previewOk = flatExecution.executePreview(request, preview, error);
        const auto exportOk = flatExecution.executeExport(request, exported, error);
        check(previewOk && exportOk && preview.source == exported.source,
              "preview and export share the same generated-normal deformation source");
        if (previewOk && exportOk)
        {
            const auto& vertices = preview.source->scene->vertices;
            arbitgpu::NativeDeformationFrameData data, exportData;
            check(vertices[0].normal.x == 1.0f && vertices[1].normal.x == 1.0f
                  && vertices[2].normal.x == 1.0f && vertices[2].normal.z == 0.0f
                  && arbitgpu::prepareNativeDeformationFrame(*preview.source,
                      *preview.evaluation.deformation, data, error)
                  && arbitgpu::prepareNativeDeformationFrame(*exported.source,
                      *exported.evaluation.deformation, exportData, error)
                  && data.vertexCount == (indexed ? 6u : 3u)
                  && data.jointCount == (deformationMode == 2 ? 2u : 1u)
                  && preview.source->scene->vertexCount == data.vertexCount
                  && preview.source->scene->objects[0].vertexCount == data.vertexCount
                  && (!indexed || (vertices[3].normal.x == 1.0f
                      && vertices[5].uv.y == vertices[2].uv.y))
                  && std::abs(data.jointPalette[0]) < 0.0001f
                  && std::abs(data.jointPalette[1] - 1.0f) < 0.0001f
                  && data.jointPalette == exportData.jointPalette,
                  "generated +X normals and the +Y-rotating palette reach native upload contracts unchanged");
        }
        if (deformationMode == 0)
        {
            // Node 3 is the only joint of the skinned triangle at node 0. Node 0 lies
            // outside the subtree of node 3, so the offset would not reach that skin.
            // The executor refuses before any draw is submitted.
            auto candidate = request;
            candidate.objectLocalTransform.version = 1;
            candidate.objectLocalTransform.nodeStableId = 4;
            candidate.objectLocalTransform.meshStableId = 0;
            candidate.objectLocalTransform.translation[0] = 1;
            videohelper::modelpayload::ImportedAnimatedSceneReceipt rejected;
            std::string rejectedError;
            const auto before = flatBackend.submissions;
            check(!flatExecution.executePreview(candidate, rejected, rejectedError)
                && rejectedError == "Imported object local transforms with skin joints in the moved subtree are not rendered yet"
                && flatBackend.submissions == before,
                "a skin that lists a joint of the moved subtree refuses the object-local transform");
        }
    }

    {
        const auto nestedBytes = nestedAnimatedStarter();
        const auto nestedKey = exactKey("nested-animated-starter", nestedBytes);
        check(publish(payloads, "nested-animated-transfer", nestedKey, nestedBytes),
              "nested animated GLB publishes exact source bytes");
        videohelper::modelpayload::ImportedAnimatedSceneCompatibility compatibility;
        check(videohelper::modelpayload::ImportedAnimatedScenePayloadExecution::inspectCompatibility(
                  payloads.resolvePreview(nestedKey), 0u, compatibility, error)
              && compatibility.clips.size() == 1
              && compatibility.clips[0].compatibleMeshStableIds
                  == std::vector<visualanimationimport::StableId> {1},
              "animated mesh under two transform-only parents is authoring-compatible");

        FakeDeformationBackend nestedBackend;
        videohelper::modelpayload::ImportedAnimatedScenePayloadExecution nestedExecution(
            payloads, nestedBackend);
        auto request = animatedRequest;
        request.operation.asset = nestedKey;
        request.operation.playback.timelineSeconds = 0.5;
        request.frame = {12, 24, 1};
        videohelper::modelpayload::ImportedAnimatedSceneReceipt preview, exported;
        const bool previewOk = nestedExecution.executePreview(request, preview, error);
        if (!previewOk) std::fprintf(stderr, "nested animated preview: %s\n", error.c_str());
        const bool exportOk = nestedExecution.executeExport(request, exported, error);
        if (!exportOk) std::fprintf(stderr, "nested animated export: %s\n", error.c_str());
        check(previewOk && exportOk && preview.valid() && exported.valid()
              && preview.source == exported.source && nestedBackend.preparations == 1,
              "nested preview and export retain one immutable native draw owner");
        if (previewOk && exportOk)
        {
            const auto& scene = *preview.source->scene;
            const auto& draw = scene.objects[0];
            check(scene.objectCount == 1 && draw.id.value == 196609u
                  && preview.source->mesh.value == 1u && !draw.parent.isValid()
                  && std::abs(draw.transform.translation.x - 0.1f) < 0.0001f
                  && std::abs(draw.transform.translation.y - 0.2f) < 0.0001f
                  && std::abs(draw.transform.rotation.z - std::sqrt(0.5f)) < 0.0001f
                  && std::abs(draw.transform.rotation.w - std::sqrt(0.5f)) < 0.0001f
                  && std::abs(draw.transform.scale.x - 0.75f) < 0.0001f
                  && std::abs(draw.transform.scale.y - 0.5f) < 0.0001f,
                  "the admitted GPU draw carries the composed parent world TRS and exact object identity");
            check(preview.payload->bytes() == nestedBytes
                  && preview.evaluation.deformation->sampleTimeSeconds()
                      == exported.evaluation.deformation->sampleTimeSeconds()
                  && preview.evaluation.deformation->morphWeights().size() == 1
                  && exported.evaluation.deformation->morphWeights().size() == 1,
                  "hierarchy adaptation retains exact asset bytes and preview/export animation time");
            auto referenceRequest = request;
            referenceRequest.operation.asset = starterKey;
            videohelper::modelpayload::ImportedAnimatedSceneReceipt reference;
            const bool referenceOk = animatedExecution.executePreview(referenceRequest, reference, error);
            check(referenceOk, "the root-level animation supplies a matching reference sample");
            if (referenceOk && preview.evaluation.deformation->morphWeights().size() == 1
                && exported.evaluation.deformation->morphWeights().size() == 1
                && reference.evaluation.deformation->morphWeights().size() == 1)
                check(preview.evaluation.deformation->morphWeights()[0].values()
                          == reference.evaluation.deformation->morphWeights()[0].values()
                      && preview.evaluation.deformation->morphWeights()[0].values()
                          == exported.evaluation.deformation->morphWeights()[0].values(),
                      "parent adaptation preserves the original morph animation samples");
        }
    }

    {
        auto controlled = animatedRequest;
        controlled.operation.playback.timelineSeconds = 0.5;
        controlled.operation.playback.weight = 0.25;
        controlled.frame = {12, 24, 1};
        controlled.operation.pose.meshStableId = 1;
        controlled.operation.pose.morphEnabled = true;
        controlled.operation.pose.morphTargetStableId = 1;
        controlled.operation.pose.morphWeight = 0.8;
        videohelper::modelpayload::ImportedAnimatedSceneReceipt preview, exported;
        const bool previewOk = animatedExecution.executePreview(controlled, preview, error);
        const bool exportOk = animatedExecution.executeExport(controlled, exported, error);
        check(previewOk && exportOk, "selected morph override reaches exact GLB preview and export");
        if (previewOk && exportOk)
        {
            arbitgpu::NativeDeformationFrameData previewData, exportData;
            check(arbitgpu::prepareNativeDeformationFrame(*preview.source,
                      *preview.evaluation.deformation, previewData, error)
                  && arbitgpu::prepareNativeDeformationFrame(*exported.source,
                      *exported.evaluation.deformation, exportData, error)
                  && std::abs(previewData.morphWeights[0] - 0.8f) < 0.0001f
                  && previewData.morphWeights == exportData.morphWeights
                  && preview.runtimeInputs.morphWeight == 1.0f
                  && exported.runtimeInputs.morphWeight == 1.0f
                  && preview.evaluation.deformation->time() == controlled.frame
                  && exported.evaluation.deformation->time() == controlled.frame,
                  "preview and export use the selected weight at the same presentation frame");
        }
        auto invalid = controlled;
        invalid.operation.pose.morphTargetIndex = 1;
        check(!animatedExecution.executePreview(invalid, preview, error)
              && error.find("selected morph target") != std::string::npos,
              "a missing morph target is rejected even when model bytes are already admitted");
        invalid = controlled;
        invalid.operation.pose.morphTargetStableId = 2;
        check(!animatedExecution.executeExport(invalid, exported, error)
              && error.find("catalog identity") != std::string::npos,
              "a mismatched named morph identity is rejected by export");
        invalid = controlled;
        invalid.operation.pose.boneEnabled = true;
        invalid.operation.pose.boneStableId = 1;
        check(!animatedExecution.executePreview(invalid, preview, error)
              && error.find("selected bone") != std::string::npos,
              "selected bone controls reject an unskinned imported mesh");
        check(animatedExecution.executePreview(controlled, preview, error),
              "invalid pose selection preserves the admitted model for a valid edit");
    }

    videohelper::modelpayload::ImportedSceneRequest productRequest;
    productRequest.asset = productKey;
    productRequest.sceneIndex = 1u;
    productRequest.dimensions = {96, 64};
    productRequest.runtimeInputs.objectRotationDegrees[1] = 30.0f;
    productRequest.runtimeInputs.objectScale = 1.25f;
    HarmonicMIDI::grid::SceneCameraRecord cameraOverride;
    cameraOverride.id.value = 701;
    cameraOverride.transform.translation.z = 4.0f;
    cameraOverride.verticalFovRadians = 0.75f;
    cameraOverride.nearPlane = 0.25f;
    cameraOverride.farPlane = 500.0f;
    productRequest.camera = cameraOverride;
    HarmonicMIDI::grid::SceneLightRecord lightOverride;
    lightOverride.id.value = 702;
    lightOverride.color = { 0.25f, 0.5f, 0.75f };
    lightOverride.intensity = 8.0f;
    productRequest.light = lightOverride;
    videohelper::modelpayload::ImportedSceneExecutionReceipt productPreview;
    videohelper::modelpayload::ImportedSceneExecutionReceipt productExport;
    check (execution.executePreview (productRequest, productPreview, error)
           && execution.executeExport (productRequest, productExport, error),
           "product preview and export execute resident exact GLB bytes");
    check (productPreview.valid() && productExport.valid()
           && productPreview.payload == productExport.payload
           && productPreview.admission.scene == productExport.admission.scene
           && productPreview.frame.rendered.use == RenderUse::Preview
           && productExport.frame.rendered.use == RenderUse::Export
           && productBackend.preparations == 1 && productBackend.submissions == 2
           && productBackend.lastRuntimeInputs.objectRotationDegrees[1] == 30.0f
           && productBackend.lastRuntimeInputs.objectScale == 1.25f
           && productBackend.lastRuntimeInputs.cameraOverride
           && productBackend.lastRuntimeInputs.cameraOverride->id.value == 701
           && productBackend.lastRuntimeInputs.lightOverride
           && productBackend.lastRuntimeInputs.lightOverride->id.value == 702
           && productBackend.lastRuntimeInputs.lightOverride->intensity == 8.0f,
           "product preview/export share admission, immutable ownership, and render bindings");
    const auto productScene = productExport.admission.scene;

    const auto makeComposedProductScene = [&] (std::uint32_t identityOffset, float childX)
    {
        auto value = *admission.scene;
        value.id.value = 1000u + identityOffset;
        value.vertexCount = 6;
        value.indexCount = 6;
        value.objectCount = 2;
        value.materialCount = 2;
        value.textureCount = 1;
        value.textureTexelCount = 2;
        value.textureTexels.resize (value.textureTexelCount);
        value.textures[0].id.value = 1100u + identityOffset;
        value.textures[0].width = 2;
        value.textures[0].height = 1;
        value.textures[0].firstTexel = 0;
        value.textureTexels[0] = { 255, 32, 16, 255 };
        value.textureTexels[1] = { 16, 64, 255, 255 };
        for (std::size_t vertex = 0; vertex < 3; ++vertex)
            value.vertices[3 + vertex] = value.vertices[vertex];
        for (std::size_t index = 0; index < 3; ++index)
            value.indices[3 + index] = value.indices[index];
        value.materials[0].id.value = 1200u + identityOffset;
        value.materials[0].baseColorTexture = value.textures[0].id;
        value.materials[1] = value.materials[0];
        value.materials[1].id.value = 1300u + identityOffset;
        value.materials[1].baseColor = { 0.1f, 0.8f, 0.2f };
        value.objects[0].id.value = 1400u + identityOffset;
        value.objects[0].material = value.materials[0].id;
        value.objects[0].firstVertex = 0;
        value.objects[0].vertexCount = 3;
        value.objects[0].firstIndex = 0;
        value.objects[0].indexCount = 3;
        value.objects[1] = value.objects[0];
        value.objects[1].id.value = 1500u + identityOffset;
        value.objects[1].parent = value.objects[0].id;
        value.objects[1].material = value.materials[1].id;
        value.objects[1].firstVertex = 3;
        value.objects[1].firstIndex = 3;
        value.objects[1].transform.translation.x = childX;
        return std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (std::move (value));
    };
    auto clipA = productRequest;
    clipA.asset = {};
    clipA.sceneIndex.reset();
    clipA.camera.reset();
    clipA.light.reset();
    // A composed Scene3D request mirrors the plugin's Render-3D lowering
    // contract: nonzero structural revision with evaluation >= structural
    // (VideoVisualGraph lowers plan.structuralRevision for every composed
    // scene). Zero revisions describe the import route only.
    clipA.structuralRevision = 1;
    clipA.evaluationRevision = 1;
    clipA.sceneSnapshot = makeComposedProductScene (1, -1.5f);
    clipA.projectGeneration = 77;
    clipA.helperGeneration = 9;
    clipA.clipId = 101;
    clipA.staticPayloadIdentity = "composed-clip-a-v1";
    auto clipB = clipA;
    clipB.sceneSnapshot = makeComposedProductScene (20, 1.5f);
    clipB.clipId = 102;
    clipB.staticPayloadIdentity = "composed-clip-b-v1";
    {
        auto full = std::make_shared<HarmonicMIDI::grid::Visual3DScene>(*clipA.sceneSnapshot);
        full->textures[0].width = full->textures[0].height = 1024;
        full->textureTexelCount = 1024u * 1024u;
        full->textureTexels.resize(full->textureTexelCount, {17, 29, 43, 255});
        visualimportedscenerender::Request request;
        request.sourceStableId = full->id.value;
        request.renderStableId = request.sourceStableId + 1;
        request.structuralRevision = request.evaluationRevision = 1;
        request.sceneSnapshot = std::move(full);
        const auto wire = visualimportedscenerender::encode(request);
        visualimportedscenerender::Request restored;
        check(wire.size() > 8u * 1024u * 1024u
              && wire.size() <= visualimportedscenerender::kMaximumV8EncodedBytes
              && visualimportedscenerender::decode(wire, restored)
              && restored.sceneSnapshot
              && restored.sceneSnapshot->textureTexels.size() == 1024u * 1024u
              && restored.sceneSnapshot->textureTexels.back().red == 17,
              "composed scene wire carries a full admitted texture within an explicit bound");
        request.imageOutput = renderpassoutput::Output::Depth;
        const auto aovWire = visualimportedscenerender::encode(request);
        check(!aovWire.empty() && visualimportedscenerender::decode(aovWire, restored)
              && restored.imageOutput == renderpassoutput::Output::Depth,
              "the AOV request envelope retains the full bounded scene snapshot");
    }
    FakeBackend cacheBackend;
    cacheBackend.backendInfo.backend = "opengl";
    videohelper::modelpayload::Store cacheStore;
    videohelper::modelpayload::ImportedScenePayloadExecution cacheExecution (
        cacheStore, cacheBackend);
    videohelper::modelpayload::ImportedSceneExecutionReceipt clipAFrame1;
    videohelper::modelpayload::ImportedSceneExecutionReceipt clipBFrame1;
    videohelper::modelpayload::ImportedSceneExecutionReceipt clipAFrame2;
    videohelper::modelpayload::ImportedSceneExecutionReceipt clipBFrame2;
    const bool composedFrame1 = cacheExecution.executePreview (clipA, clipAFrame1, error)
        && cacheExecution.executePreview (clipB, clipBFrame1, error);
    clipA.runtimeInputs.objectTranslationOffset[0] = 0.25f;
    clipB.runtimeInputs.objectRotationDegrees[1] = 15.0f;
    const bool composedFrame2 = cacheExecution.executeExport (clipA, clipAFrame2, error)
        && cacheExecution.executeExport (clipB, clipBFrame2, error);

    if (!composedFrame1 || !composedFrame2)
        std::fprintf (stderr, "composed cache execution diagnostic: %s preparations=%d submissions=%d\n",
                      error.c_str(), cacheBackend.preparations, cacheBackend.submissions);
    check (composedFrame1 && composedFrame2
           && cacheBackend.preparations == 2
           && cacheBackend.submissions == 4
           && clipAFrame1.admission.scene == clipAFrame2.admission.scene
           && clipBFrame1.admission.scene == clipBFrame2.admission.scene
           && clipAFrame1.admission.scene != clipBFrame1.admission.scene
           && clipAFrame2.frame.rendered.stats.staticUploadCount == 0
           && clipBFrame2.frame.rendered.stats.staticUploadCount == 0
           && clipAFrame2.frame.rendered.stats.textureUploadCount == 0
           && clipBFrame2.frame.rendered.stats.textureUploadCount == 0
           && clipAFrame2.frame.rendered.stats.materialProgramUploadCount == 0
           && clipBFrame2.frame.rendered.stats.materialProgramUploadCount == 0,
           "preview and export isolate frame receipts while sharing each clip's immutable static owner");

    {
        FakeBackend causticBackend;
        causticBackend.backendInfo.backend = "opengl";
        videohelper::modelpayload::ImportedScenePayloadExecution causticExecution (
            cacheStore, causticBackend);
        videohelper::modelpayload::ImportedSceneExecutionReceipt previous;
        check(causticExecution.executePreview(clipA, previous, error) && previous.valid(),
              "caustic refusal fixture retains a previous ordinary frame");
        const auto submissions = causticBackend.submissions;
        auto caustic = clipA;
        const auto scene = geometryCompositionFixture(error);
        check(scene != nullptr, "caustic payload fixture retains generated geometry");
        if (scene)
        {
            scene->lights[0].spectralSource.version = 1;
            scene->lights[0].spectralSource.wavelengthsNanometres = diffractionmaterial::kProductionWavelengthsNanometres;
            scene->lights[0].spectralSource.radiance.fill(1);
            scene->lights[0].spectralSource.traceMaximumDistance = 75;
            caustic.sceneSnapshot = scene;
            caustic.structuralRevision = caustic.evaluationRevision = 1;
            caustic.runtimeInputs.linearColor = true;
            caustic.frame = {30, 24, 1};
            geometrysurfacematerial::ObjectProgram glass;
            glass.program.sourceMaterial = 9001;
            auto& material = glass.program.material;
            material.scene = scene->id;
            material.sceneSnapshot = scene;
            material.sceneRevision = material.structuralRevision = material.programRevision = 1;
            material.evaluationRevision = 1;
            material.binding.targetKind = surfacematerialbinding::BindingTargetKind::ObjectOverride;
            material.binding.object = scene->objects[0].id;
            material.binding.surfaceMaterialRevision = 1;
            material.program = videohelper::tests::smoothglassfixture::authoredProgram(scene->objects[0].material.value);
            const auto admitted = surfacematerial::admit(material.program, error);
            check(admitted.has_value(), "caustic payload fixture owns an exact closed-volume program");
            if (admitted)
            {
                material.binding.surfaceMaterialDigest = admitted->structuralDigest();
                caustic.surfacePrograms = {glass};
                caustic.glassCaustic = visualimportedscenerender::GlassCausticSelection {
                    1, scene->lights[0].id, scene->objects[0].id, scene->objects[0].material,
                    {1, 1.0e-4f, 1.0e-6f, 32, 64, 16}};
                visualimportedscenerender::Request wire;
                wire.sourceStableId = scene->id.value;
                wire.renderStableId = 99;
                wire.sceneSnapshot = scene;
                wire.structuralRevision = wire.evaluationRevision = 1;
                wire.surfacePrograms = caustic.surfacePrograms;
                wire.glassCaustic = caustic.glassCaustic;
                check(visualimportedscenerender::valid(wire),
                      "payload caustic uses the same admitted retained scene, source and material contract");
                for (int mutation = 0; mutation < 7; ++mutation)
                {
                    auto invalid = caustic;
                    if (mutation == 0) invalid.glassCaustic->sourceLight.value = 999;
                    if (mutation == 1) invalid.glassCaustic->glassObject.value = 999;
                    if (mutation == 2) invalid.glassCaustic->glassMaterial.value = 999;
                    if (mutation == 3) invalid.glassCaustic->policy.maximumPending = 0;
                    if (mutation == 4) invalid.surfacePrograms[0].program.material.sceneRevision = 2;
                    if (mutation == 5) invalid.surfacePrograms[0].program.material.program.transport
                        = surfacematerial::TransportMode::ThinSurface;
                    if (mutation == 6)
                    {
                        auto wrongScene = std::make_shared<HarmonicMIDI::grid::Visual3DScene>(*scene);
                        wrongScene->lights[0].spectralSource = {};
                        invalid.sceneSnapshot = wrongScene;
                        invalid.surfacePrograms[0].program.material.sceneSnapshot = wrongScene;
                    }
                    auto refused = previous;
                    check(!causticExecution.executePreview(invalid, refused, error)
                          && !refused.valid() && causticBackend.submissions == submissions,
                          "direct malformed caustic refuses without last-good pixels even under reused serialized identity");
                }
                const auto discrete = smoothGlassDiscreteGratingFixture(error);
                if (discrete)
                {
                    FakeBackend discreteBackend; discreteBackend.backendInfo.backend = "opengl";
                    discreteBackend.spectralReceiverTransport = true;
                    videohelper::modelpayload::ImportedScenePayloadExecution discreteExecution(cacheStore, discreteBackend);
                    auto mixed = caustic;
                    mixed.sceneSnapshot = discrete->sceneSnapshot;
                    mixed.structuralRevision = discrete->structuralRevision;
                    mixed.evaluationRevision = discrete->evaluationRevision;
                    mixed.surfacePrograms = discrete->surfacePrograms;
                    mixed.diffractionMaterial = discrete->diffractionMaterial;
                    mixed.glassCaustic = discrete->glassCaustic;
                    mixed.runtimeInputs = {}; mixed.runtimeInputs.linearColor = true;
                    for (const auto use : {videohelper::modelpayload::ImportedSceneUse::Preview,
                                           videohelper::modelpayload::ImportedSceneUse::Export})
                    {
                        auto refused = previous;
                        const auto rendered = use == videohelper::modelpayload::ImportedSceneUse::Preview
                            ? discreteExecution.executePreview(mixed, refused, error)
                            : discreteExecution.executeExport(mixed, refused, error);
                        check(!rendered && !refused.valid()
                            && error.find("without pass composition, diffraction") != std::string::npos
                            && discreteBackend.preparations == 0 && discreteBackend.submissions == 0,
                            "ordinary imported caller threads exact discrete selection and still refuses combined Color atomically");
                    }
                }
                for (const auto use : {videohelper::modelpayload::ImportedSceneUse::Preview,
                                       videohelper::modelpayload::ImportedSceneUse::Export})
                {
                    auto refused = previous;
                    const auto rendered = use == videohelper::modelpayload::ImportedSceneUse::Preview
                        ? causticExecution.executePreview(caustic, refused, error)
                        : causticExecution.executeExport(caustic, refused, error);
                    check(!rendered && !refused.valid() && causticBackend.submissions == submissions
                          && error.find("caustic") != std::string::npos,
                          "unsupported native caustic refuses before ordinary backend draw in preview and export");
                }
                auto plainImport = productRequest;
                plainImport.glassCaustic = caustic.glassCaustic;
                auto refused = previous;
                check(!causticExecution.executePreview(plainImport, refused, error)
                      && !refused.valid() && causticBackend.submissions == submissions
                      && error.find("exact retained Scene3D snapshot owner") != std::string::npos,
                      "plain imported GLB eligibility does not expand to selected retained-scene caustics");
                auto ordinary = clipA;
                ordinary.runtimeInputs.glassCaustic = caustic.glassCaustic;
                check(causticExecution.executeExport(ordinary, refused, error)
                      && refused.valid() && !causticBackend.lastRuntimeInputs.glassCaustic,
                      "only the validated request selection can enter native runtime inputs");
            }
        }
    }

    auto malformedReplacement = clipA;
    malformedReplacement.staticPayloadIdentity = "composed-clip-a-malformed";
    auto malformedScene = std::make_shared<HarmonicMIDI::grid::Visual3DScene> (
        *clipA.sceneSnapshot);
    malformedScene->objects[1].id = malformedScene->objects[0].id;
    malformedReplacement.sceneSnapshot = malformedScene;
    const auto submissionsBeforeMalformed = cacheBackend.submissions;
    auto malformedReceipt = clipAFrame2;
    const bool malformedExecuted = cacheExecution.executePreview (
        malformedReplacement, malformedReceipt, error);
    // The retired owner renders no stale pixels (publishStaticPayload removed
    // the owner before any attempt); the caller's earlier receipt is unchanged.
    check (! malformedExecuted
           && cacheBackend.submissions == submissionsBeforeMalformed
           && ! malformedReceipt.valid(),
           "a malformed same-generation replacement retires the old owner without rendering stale pixels");
    auto recoveredReplacement = clipA;
    recoveredReplacement.staticPayloadIdentity = "composed-clip-a-v2";
    recoveredReplacement.sceneSnapshot = makeComposedProductScene (40, -2.0f);
    check (cacheExecution.executePreview (recoveredReplacement, clipAFrame2, error)
           && cacheBackend.preparations == 3,
           "a later valid same-generation replacement creates one new owner and recovers");

    auto helperReplacementA = recoveredReplacement;
    auto helperReplacementB = clipB;
    helperReplacementA.helperGeneration = 10;
    helperReplacementB.helperGeneration = 10;
    check (cacheExecution.executePreview (helperReplacementA, clipAFrame2, error)
           && cacheExecution.executePreview (helperReplacementB, clipBFrame2, error)
           && cacheBackend.preparations == 5,
           "helper generation replacement retires only each exact clip owner");
    cacheBackend.backendInfo.device = "replacement-device";
    check (cacheExecution.executePreview (helperReplacementA, clipAFrame2, error)
           && cacheBackend.preparations == 6,
           "device identity replacement prunes production GPU owners before reuse");
    helperReplacementA.projectGeneration = 78;
    check (cacheExecution.executePreview (helperReplacementA, clipAFrame2, error)
           && cacheBackend.preparations == 7,
           "project generation replacement prunes production GPU owners before reuse");

    FakeBackend evictionBackend;
    videohelper::modelpayload::ImportedScenePayloadExecution evictionExecution (
        cacheStore, evictionBackend);
    videohelper::modelpayload::ImportedSceneExecutionReceipt leasedOldest;
    bool filledBoundedCache = true;
    for (int clip = 0; clip < 17; ++clip)
    {
        auto boundedRequest = clipA;
        boundedRequest.clipId = 100 + clip;
        boundedRequest.staticPayloadIdentity = "bounded-composed-" + std::to_string (clip);
        boundedRequest.sceneSnapshot = makeComposedProductScene (
            static_cast<std::uint32_t> (100 + clip * 10), static_cast<float> (clip));
        videohelper::modelpayload::ImportedSceneExecutionReceipt current;
        filledBoundedCache = filledBoundedCache
            && evictionExecution.executePreview (boundedRequest, current, error);
        if (clip == 0)
            leasedOldest = current;
    }
    auto oldestRequest = clipA;
    oldestRequest.clipId = 100;
    oldestRequest.staticPayloadIdentity = "bounded-composed-0";
    oldestRequest.sceneSnapshot = makeComposedProductScene (100, 0.0f);
    videohelper::modelpayload::ImportedSceneExecutionReceipt reloadedOldest;
    check (filledBoundedCache && evictionBackend.preparations == 17
           && leasedOldest.valid()
           && evictionExecution.executePreview (oldestRequest, reloadedOldest, error)
           && evictionBackend.preparations == 18
           && leasedOldest.valid(),
           "the 16-owner cache evicts the deterministic oldest owner while existing receipts retain their lease");

    auto invalidLightRequest = productRequest;
    invalidLightRequest.light->id = {};
    check (! execution.executePreview (invalidLightRequest, productPreview, error)
           && error == "imported scene graph light is invalid"
           && productBackend.submissions == 2,
           "invalid light identity is rejected before native submission");

    auto wrongKey = productKey;
    wrongKey.id = "other-scene";
    productRequest.asset = wrongKey;
    check (! execution.executePreview (productRequest, productPreview, error)
           && error == "imported scene exact payload is not resident"
           && productBackend.submissions == 2,
           "catalog-like metadata cannot select another resident payload");

    wrongKey = productKey;
    ++wrongKey.version;
    productRequest.asset = wrongKey;
    check (! execution.executePreview (productRequest, productPreview, error)
           && error == "imported scene exact payload is not resident"
           && productBackend.submissions == 2,
           "a different catalog version cannot substitute resident bytes");

    wrongKey = productKey;
    wrongKey.contentSha256[0] = wrongKey.contentSha256[0] == '0' ? '1' : '0';
    productRequest.asset = wrongKey;
    check (! execution.executePreview (productRequest, productPreview, error)
           && error == "imported scene exact payload is not resident"
           && productBackend.submissions == 2,
           "a different digest cannot substitute resident bytes");

    productRequest.asset = productKey;
    productRequest.sceneIndex = 64u;
    check (! execution.executePreview (productRequest, productPreview, error)
           && error == "imported scene index exceeds the product bound"
           && productBackend.preparations == 1,
           "scene bounds are rejected before product parsing or GPU preparation");

    auto externalRoot = triangleRoot();
    externalRoot["buffers"][0]["uri"] = "outside.bin";
    const auto externalGlb = makeGlb (externalRoot, triangleBin());
    const auto externalKey = exactKey ("external-uri-scene", externalGlb);
    check (publish (payloads, "external-transfer", externalKey, externalGlb),
           "an exact external-URI fixture reaches decoder admission");
    productRequest.asset = externalKey;
    productRequest.sceneIndex.reset();
    check (! execution.executeExport (productRequest, productExport, error)
           && error.find ("requires an unextended embedded buffer") != std::string::npos
           && productBackend.preparations == 1 && productBackend.submissions == 2,
           "product execution rejects external URIs without opening a helper path");

    productRequest.asset = productKey;
    productRequest.sceneIndex = 1u;
    surfacematerial::ProgramDescription materialProgram;
    const auto appendConstant = [&] (surfacematerial::ValueType type,
                                     std::array<float, 4> literal = {})
    {
        surfacematerial::Operation operation;
        operation.id = static_cast<surfacematerial::ValueId> (
            materialProgram.operations.size() + 1u);
        operation.kind = type == surfacematerial::ValueType::UInt
            ? surfacematerial::OperationKind::UIntConstant
            : surfacematerial::OperationKind::FloatConstant;
        operation.resultType = type;
        operation.literal = literal;
        materialProgram.operations.push_back (operation);
        return operation.id;
    };
    for (const auto output : surfacematerial::kSurfaceOutputs)
    {
        const auto type = surfacematerial::outputType (output);
        const auto literal = output == surfacematerial::OutputSemantic::BaseColor
            ? std::array<float, 4> { 0.8f, 0.15f, 0.05f, 0.0f }
            : output == surfacematerial::OutputSemantic::Normal
                ? std::array<float, 4> { 0.0f, 0.0f, 1.0f, 0.0f }
                : output == surfacematerial::OutputSemantic::Opacity
                    ? std::array<float, 4> { 1.0f, 0.0f, 0.0f, 0.0f }
                    : output == surfacematerial::OutputSemantic::Ior
                        ? std::array<float, 4> { 1.5f, 0.0f, 0.0f, 0.0f }
                        : std::array<float, 4> {};
        materialProgram.outputs[static_cast<std::size_t> (output)]
            = appendConstant (type, literal);
    }
    materialProgram.operations[static_cast<std::size_t> (
        surfacematerial::OutputSemantic::Opacity)].literal[0] = 1.0f;
    materialProgram.operations[static_cast<std::size_t> (
        surfacematerial::OutputSemantic::Normal)].literal[2] = 1.0f;
    materialProgram.operations[static_cast<std::size_t> (
        surfacematerial::OutputSemantic::Ior)].literal[0] = 1.5f;
    materialProgram.operations.back().unsignedLiteral
        = productScene->objects[0].material.value;
    const auto admittedMaterial = surfacematerial::admit (materialProgram, error);
    check (admittedMaterial.has_value(),
           "the product material fixture is an admitted typed surface program");

    surfacematerialbinding::ImportedSceneMaterialRequest materialRequest;
    materialRequest.scene = productScene->id;
    materialRequest.sceneRevision = 1;
    materialRequest.structuralRevision = 1;
    materialRequest.evaluationRevision = 1;
    materialRequest.programRevision = 1;
    materialRequest.program = materialProgram;
    materialRequest.binding.targetKind
        = surfacematerialbinding::BindingTargetKind::ObjectOverride;
    materialRequest.binding.object = productScene->objects[0].id;
    materialRequest.binding.surfaceMaterialDigest = admittedMaterial
        ? admittedMaterial->structuralDigest() : std::string (64, '0');
    materialRequest.binding.surfaceMaterialRevision = 1;
    productRequest.material = materialRequest;
    {
        FakeBackend vertexBackend; vertexBackend.backendInfo.backend = "opengl";
        videohelper::modelpayload::ImportedScenePayloadExecution vertexExecution(payloads, vertexBackend);
        auto request = productRequest; request.frame = {0,60,1}; request.clipId = 73;
        request.material->vertexModifier = videohelper::test::audioNormalDisplacement();
        request.runtimeInputs.imageOutput = renderpassoutput::Output::Motion;
        videohelper::modelpayload::ImportedSceneExecutionReceipt receipt;
        check(!vertexExecution.executePreview(request, receipt, error) && !receipt.valid()
            && vertexBackend.submissions == 0 && error.find("vertex processor") != std::string::npos,
            "imported Motion refuses material vertex displacement without previous processor inputs");
        request.runtimeInputs.imageOutput = renderpassoutput::Output::Color;
        check(vertexExecution.executePreview(request, receipt, error), "material vertex Color route remains available");
        request.material.reset(); request.frame.frame = 1;
        request.runtimeInputs.imageOutput = renderpassoutput::Output::Motion;
        check(vertexExecution.executePreview(request, receipt, error) && !vertexBackend.lastRuntimeInputs.previousMotion,
            "Color with unsupported vertex geometry cannot become next Motion predecessor");
    }
    {
        FakeDeformationBackend vertexBackend; vertexBackend.retainedDeformationMotion = true;
        videohelper::modelpayload::ImportedAnimatedScenePayloadExecution vertexExecution(payloads, vertexBackend);
        auto request = animatedRequest;
        request.material = materialRequest; request.material->scene.value = 1;
        request.material->binding.object.value = 1;
        request.material->vertexModifier = videohelper::test::audioNormalDisplacement();
        request.runtimeInputs.imageOutput = renderpassoutput::Output::Motion;
        videohelper::modelpayload::ImportedAnimatedSceneReceipt receipt;
        check(!vertexExecution.executePreview(request, receipt, error) && !receipt.valid()
            && vertexBackend.submissions == 0 && error.find("reads Normal over deformed geometry") != std::string::npos,
            "retained skin/morph Motion refuses a vertex program that needs the prior normal");
        // A position and time program runs again over the prior skinned or
        // morphed pose, so it keeps the retained history.
        request.material->vertexModifier = videohelper::test::timeAxisDisplacement();
        const bool firstExecuted = vertexExecution.executePreview(request, receipt, error);
        if (!firstExecuted) std::fprintf(stderr, "animated vertex Motion: %s\n", error.c_str());
        check(firstExecuted && receipt.valid() && !vertexBackend.lastRuntimeInputs.previousMotion,
            "retained skin/morph Motion admits a position and time vertex program");
        const auto firstPose = receipt.evaluation.deformation;
        request.frame.frame = 1;
        check(vertexExecution.executePreview(request, receipt, error)
            && vertexBackend.lastRuntimeInputs.previousMotion
            && vertexBackend.lastRuntimeInputs.previousMotion->deformation == firstPose
            && vertexBackend.lastRuntimeInputs.previousMotion->timeSeconds == 0.0f
            && vertexBackend.lastRuntimeInputs.timeSeconds == static_cast<float>(1.0 / 24.0),
            "the vertex program's next Motion frame receives the prior pose and prior time");
    }
    {
        FakeDeformationBackend nestedBackend;
        videohelper::modelpayload::ImportedAnimatedScenePayloadExecution nestedExecution(
            payloads, nestedBackend);
        auto request = animatedRequest;
        request.operation.asset = exactKey("nested-animated-starter", nestedAnimatedStarter());
        request.material = materialRequest;
        request.material->scene.value = 1u;
        request.material->binding.object.value = 196609u;
        videohelper::modelpayload::ImportedAnimatedSceneReceipt preview, exported;
        const bool previewOk = nestedExecution.executePreview(request, preview, error);
        if (!previewOk) std::fprintf(stderr, "nested animated material preview: %s\n", error.c_str());
        const bool exportOk = nestedExecution.executeExport(request, exported, error);
        if (!exportOk) std::fprintf(stderr, "nested animated material export: %s\n", error.c_str());
        check(previewOk && exportOk && preview.valid() && exported.valid()
              && preview.source == exported.source && nestedBackend.preparations == 1,
              "the nested mesh accepts its exact surface-material binding in preview and export");
    }
    videohelper::modelpayload::ImportedSceneExecutionReceipt materialPreview;
    videohelper::modelpayload::ImportedSceneExecutionReceipt materialExport;
    const bool materialPreviewOk
        = execution.executePreview (productRequest, materialPreview, error);
    if (! materialPreviewOk)
        std::fprintf (stderr, "material product preview diagnostic: %s\n", error.c_str());
    const bool materialExportOk
        = materialPreviewOk && execution.executeExport (productRequest, materialExport, error);
    if (materialPreviewOk && ! materialExportOk)
        std::fprintf (stderr, "material product export diagnostic: %s\n", error.c_str());
    check (materialPreviewOk && materialExportOk,
           "product preview and export publish the exact surface material before rendering");
    check (productBackend.preparations == 2 && productBackend.submissions == 4
           && productBackend.preparedWithSurfaceMaterial
           && materialPreview.frame.rendered.stats.materialProgramUploadCount == 1
           && materialExport.frame.rendered.stats.materialProgramUploadCount == 0
           && materialExport.frame.rendered.stats.reusedMaterialProgram,
           "the imported scene product shares one admitted native material across preview and export");

    {
        auto frameRequest = productRequest;
        frameRequest.material->version = surfacematerialbinding::kGraphFrameWireVersion;
        frameRequest.material->program.textureSlotCount = 1;
        surfacematerialbinding::TextureSlotBinding texture;
        texture.source = surfacematerialbinding::TextureSourceKind::GraphFrame;
        texture.graphFrame = surfacematerialbinding::TextureSlotBinding::GraphFrameEndpoint { 91, 0 };
        frameRequest.material->binding.textures = { texture };
        const auto submissionsBefore = productBackend.submissions;
        videohelper::modelpayload::ImportedSceneExecutionReceipt rejected;
        check(!execution.executePreview(frameRequest, rejected, error)
                  && !rejected.valid() && productBackend.submissions == submissionsBefore,
              "a Frame material cannot submit without its owned dependency");
        frameRequest.runtimeInputs.materialFrameTexture = std::make_shared<FakeFrame>("opengl", 64, 64, 400);
        check(!execution.executePreview(frameRequest, rejected, error)
                  && !error.empty()
                  && !rejected.valid() && productBackend.submissions == submissionsBefore,
              "a malformed constant program with an unused Frame slot cannot reuse a last-good preview material");
        check(!execution.executeExport(frameRequest, rejected, error)
                  && !rejected.valid() && productBackend.submissions == submissionsBefore,
              "a malformed Frame material also rejects export");
        frameRequest.material = productRequest.material;
        check(!execution.executePreview(frameRequest, rejected, error)
                  && !rejected.valid() && productBackend.submissions == submissionsBefore,
              "unrequested Frame owner cannot be injected into an imported-texture material");
    }

    productRequest.material->evaluationRevision = 2;
    productRequest.material->program.operations[static_cast<std::size_t> (
        surfacematerial::OutputSemantic::BaseColor)].literal[0] = 0.9f;
    const auto evaluatedMaterial
        = surfacematerial::admit (productRequest.material->program, error);
    if (evaluatedMaterial)
        productRequest.material->binding.surfaceMaterialDigest
            = evaluatedMaterial->structuralDigest();
    const bool evaluatedPreviewOk
        = execution.executePreview (productRequest, materialPreview, error);
    const bool evaluatedExportOk
        = evaluatedPreviewOk && execution.executeExport (productRequest, materialExport, error);
    check (evaluatedMaterial.has_value() && evaluatedPreviewOk && evaluatedExportOk
           && productBackend.preparations == 3 && productBackend.submissions == 6
           && materialPreview.frame.rendered.stats.materialProgramUploadCount == 1
           && materialExport.frame.rendered.stats.reusedMaterialProgram,
           "a newer evaluation republishes one immutable material program for preview and export");

    const auto rejectedRevision = materialRequest.structuralRevision + 1;
    productRequest.material->structuralRevision = rejectedRevision;
    productRequest.material->evaluationRevision = 3;
    productRequest.material->programRevision = rejectedRevision;
    productRequest.material->binding.surfaceMaterialRevision = rejectedRevision;
    productRequest.material->binding.surfaceMaterialDigest[0]
        = productRequest.material->binding.surfaceMaterialDigest[0] == '0' ? '1' : '0';
    check (execution.executePreview (productRequest, materialPreview, error)
           && materialPreview.usedLastGoodMaterial
           && ! materialPreview.materialDiagnostic.empty()
           && productBackend.preparations == 3 && productBackend.submissions == 7,
           "a rejected material revision keeps the last good product preview visible");
    check (! execution.executeExport (productRequest, materialExport, error)
           && productBackend.submissions == 7,
           "a rejected current material revision cannot reach product export");

    productRequest.material.reset();
    check (execution.executePreview (productRequest, materialPreview, error)
           && ! materialPreview.usedLastGoodMaterial
           && materialPreview.materialDiagnostic.empty()
           && productBackend.preparations == 4 && productBackend.submissions == 8
           && ! productBackend.preparedWithSurfaceMaterial,
           "removing the material product restores the imported GLB material without stale reuse");

    diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest
        diffractionRequest;
    diffractionRequest.scene = materialPreview.admission.scene->id;
    diffractionRequest.sceneRevision = 1;
    diffractionRequest.structuralRevision = 4;
    diffractionRequest.evaluationRevision = 4;
    diffractionRequest.materialRevision = 4;
    diffractionRequest.object = materialPreview.admission.scene->objects[0].id;
    diffractionRequest.material
        = diffractionmaterial::makeAluminiumBinaryGratingPreset();
    // The binding contract admits only the version-defined canonical lighting
    // plan; legacy producers migrate through migrateLegacyV6.
    diffractionRequest.lighting = diffractionmaterial::makeReferenceLighting();
    if (const auto admittedDiffraction = diffractionmaterial::admit (
            diffractionRequest.material, error))
        diffractionRequest.structuralDigest
            = admittedDiffraction->structuralDigest();
    productRequest.structuralRevision = 4;
    productRequest.diffractionMaterial = diffractionRequest;
    videohelper::modelpayload::ImportedSceneExecutionReceipt diffractionPreview;
    videohelper::modelpayload::ImportedSceneExecutionReceipt diffractionExport;
    auto unrelatedDiffractionScene = productRequest;
    ++unrelatedDiffractionScene.diffractionMaterial->sceneRevision;
    check (! execution.executePreview (
               unrelatedDiffractionScene, diffractionPreview, error)
           && error == "imported diffraction material does not match the exact scene identity"
           && productBackend.preparations == 4 && productBackend.submissions == 8,
           "product execution rejects diffraction identity from another scene revision");
    auto unrelatedDiffractionPlan = productRequest;
    --unrelatedDiffractionPlan.structuralRevision;
    check (! execution.executePreview (
               unrelatedDiffractionPlan, diffractionPreview, error)
           && error == "imported diffraction material does not match the exact plan revision"
           && productBackend.preparations == 4 && productBackend.submissions == 8,
           "product execution rejects diffraction identity from another compiled plan");
    const bool diffractionPreviewOk
        = execution.executePreview (productRequest, diffractionPreview, error);
    if (! diffractionPreviewOk)
        std::fprintf (stderr, "diffraction product preview diagnostic: %s\n",
                      error.c_str());
    const bool diffractionExportOk = diffractionPreviewOk
        && execution.executeExport (productRequest, diffractionExport, error);
    check (diffractionPreviewOk && diffractionExportOk
           && productBackend.preparations == 5 && productBackend.submissions == 10
           && productBackend.preparedMaterialKind
                == arbitgpu::NativeFixtureMaterialKind::DiffractionReflective
           && diffractionPreview.frame.rendered.stats.materialProgramUploadCount == 1
           && diffractionExport.frame.rendered.stats.reusedMaterialProgram,
           "product preview and export share one exact native diffraction program");

    productRequest.diffractionMaterial->structuralRevision = 5;
    productRequest.diffractionMaterial->materialRevision = 5;
    productRequest.diffractionMaterial->evaluationRevision = 5;
    productRequest.structuralRevision = 5;
    productRequest.diffractionMaterial->structuralDigest[0]
        = productRequest.diffractionMaterial->structuralDigest[0] == '0' ? '1' : '0';
    check (execution.executePreview (productRequest, diffractionPreview, error)
           && diffractionPreview.usedLastGoodMaterial
           && ! diffractionPreview.materialDiagnostic.empty()
           && productBackend.preparations == 5 && productBackend.submissions == 11,
           "a rejected diffraction revision keeps the last-good native preview visible");
    check (! execution.executeExport (productRequest, diffractionExport, error)
           && productBackend.submissions == 11,
           "a rejected current diffraction revision cannot reach product export");

    productRequest.diffractionMaterial.reset();
    check (execution.executePreview (productRequest, diffractionPreview, error)
           && productBackend.preparations == 6 && productBackend.submissions == 12
           && ! productBackend.preparedWithSurfaceMaterial,
           "removing diffraction restores the imported material without stale reuse");

    visualanimationimport::Request deformationRequest;
    deformationRequest.sourceStableId = 1;
    deformationRequest.deformationStableId = 2;
    deformationRequest.asset = productKey;
    deformationRequest.sceneIndex = 1u;
    deformationRequest.clipName = "Idle";
    deformationRequest.schedule = { 1, 2 };
    productRequest.deformation = deformationRequest;
    productRequest.frame = { 0, 24, 1 };
    productRequest.structuralRevision = 9;
    check (! execution.executePreview (productRequest, productPreview, error)
           && error == "native imported scene deformation execution is unavailable"
           && productBackend.submissions == 12,
           "the product route rejects deformation when no native deformation backend is owned");
    productRequest.deformation.reset();

    check (payloads.erase (productKey)
           && productPreview.payload->bytes() == glb,
           "successful receipts safely retain immutable bytes after store eviction");
    FakeBackend texturedBackend;
    GlbNativeRenderSeam texturedSeam (texturedBackend);
    GlbNativeSceneAdmission texturedAdmission;
    const auto texturedGlb = makeGlb (texturedTriangleRoot(), texturedTriangleBin());
    check (texturedSeam.admit (
               texturedGlb, std::nullopt, texturedAdmission, error)
           && error.empty() && texturedAdmission.valid(),
           "the seam decodes an embedded base-color image without path authority");
    check (texturedAdmission.valid()
           && texturedAdmission.scene->textureCount == 1
           && texturedAdmission.scene->textureTexelCount == 1
           && texturedAdmission.scene->textures[0].width == 1
           && texturedAdmission.scene->textures[0].height == 1
           && texturedAdmission.scene->materials[0].baseColorTexture
                == texturedAdmission.scene->textures[0].id
           && texturedAdmission.scene->textureTexels[0].red == 255
           && texturedAdmission.scene->textureTexels[0].green == 64
           && texturedAdmission.scene->textureTexels[0].blue == 32
           && texturedAdmission.scene->textureTexels[0].alpha == 255,
           "embedded sRGB texels remain bounded and value-owned by the admitted scene");
    GlbNativeFrameReceipt texturedPreview;
    GlbNativeFrameReceipt texturedExport;
    check (texturedSeam.renderPreview (
               texturedAdmission, RenderDimensions {96, 64}, texturedPreview, error)
           && texturedSeam.renderExport (
               texturedAdmission, RenderDimensions {96, 64}, texturedExport, error)
           && texturedBackend.preparations == 1
           && texturedBackend.submissions == 2
           && texturedPreview.rendered.stats.textureBytes
                == sizeof (HarmonicMIDI::grid::SceneTexelRgba8)
           && texturedExport.rendered.stats.reusedStaticResources,
           "preview and export execute the exact embedded texture through one native preparation");
    check (texturedPreview.materials.size() == 1
           && texturedPreview.materials[0].baseColorTexture
                == texturedAdmission.scene->textures[0].id
           && texturedPreview.textures.size() == 1
           && texturedPreview.textures[0].texture
                == texturedAdmission.scene->textures[0].id
           && texturedPreview.textures[0].texelCount == 1
           && ! texturedPreview.downstreamHandoffComplete,
           "the receipt hands off texture and sampler identity without claiming downstream consumption");
    const auto retainedScene = admission.scene;
    auto externalImageRoot = triangleRoot();
    externalImageRoot["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]
        = {{"index", 0}};
    externalImageRoot["textures"] = nlohmann::json::array ({
        {{"source", 0}}
    });
    externalImageRoot["images"] = nlohmann::json::array ({
        {{"uri", "file:///tmp/never-opened.png"}}
    });
    check (! seam.admit (makeGlb (externalImageRoot, triangleBin()), std::nullopt,
                         admission, error)
           && error == "GLB decode rejected: static GLB images must use an embedded bufferView without extensions or URIs"
           && admission.scene == retainedScene && backend.submissions == 2,
           "external image URIs fail before any path opening and preserve the last admission");

    auto animatedRoot = triangleRoot();
    animatedRoot["animations"] = nlohmann::json::array ({nlohmann::json::object()});
    check (seam.admit (makeGlb (animatedRoot, triangleBin()), std::nullopt,
                        admission, error)
           && error.empty() && admission.metadata.animations == 1
           && backend.submissions == 2,
           "base-scene admission ignores bounded animation metadata until a graph requests it");

    GlbNativeSceneAdmission empty;
    check (! seam.renderPreview (empty, RenderDimensions {32, 32}, preview, error)
           && error == "native GLB render rejected: scene admission is empty"
           && backend.submissions == 2,
           "an empty admission cannot reach the native backend");

    const auto retainedPreviewHandle = preview.rendered.nativeFrame->colorImageHandle();
    backend.rejectSubmission = true;
    check (! seam.renderPreview (admission, RenderDimensions {96, 64}, preview, error)
           && error == "native GLB render rejected: fake draw failed"
           && preview.rendered.nativeFrame->colorImageHandle() == retainedPreviewHandle,
           "a native draw failure preserves the caller's last successful frame");
    backend.rejectSubmission = false;

    {
        arbitgpu::NativeFixtureSceneRuntimeInputs causticInputs;
        causticInputs.glassCaustic.emplace();
        auto rejectedCaustic = preview;
        check(!seam.renderPreview(empty, RenderDimensions {32,32}, causticInputs, rejectedCaustic, error)
              && !rejectedCaustic.rendered.nativeFrame && backend.submissions == 2,
              "caustic preview clears an earlier frame even when scene admission fails");
        rejectedCaustic = preview;
        check(!seam.renderExport(admission, RenderDimensions {96,64}, causticInputs, rejectedCaustic, error)
              && !rejectedCaustic.rendered.nativeFrame && backend.submissions == 2,
              "caustic export cannot reuse ordinary Color after a malformed selection");
    }

    std::vector<std::uint8_t> oversized (GlbNativeRenderSeam::kMaxSourceBytes + 1u);
    check (! seam.admit (oversized, std::nullopt, admission, error)
           && error == "GLB decode rejected: source exceeds the 16 MiB native-render bound",
           "the public seam enforces its source bound before parsing");

    GlbNativeRenderSeam stubSeam (arbitgpu::nativeFixtureSceneBackend());
    GlbNativeFrameReceipt stubFrame;
    check (! stubSeam.renderExport (admission, RenderDimensions {32, 32}, stubFrame, error)
           && error == "native GLB render rejected: native GPU backend not compiled in"
           && stubFrame.rendered.nativeFrame == nullptr,
           "stub builds report native rendering as unavailable without CPU fallback");

    {
        const auto authored=retainedSurfaceFixture(error);
        check(authored.has_value(),"retained Surface product fixture is canonical");
        if (authored) {
            FakeBackend surfaceBackend; surfaceBackend.backendInfo.backend="opengl";
            videohelper::modelpayload::Store surfaceStore;
            videohelper::modelpayload::ImportedScenePayloadExecution surfaceExecution(surfaceStore,surfaceBackend);
            videohelper::modelpayload::ImportedSceneRequest request;
            request.sceneSnapshot=authored->sceneSnapshot; request.surfacePrograms=authored->surfacePrograms;
            request.structuralRevision=request.evaluationRevision=request.projectGeneration=request.helperGeneration=1;
            request.clipId=200; request.dimensions={64,64}; request.frame={0,30,1};
            request.staticPayloadIdentity=visualimportedscenerender::encode(*authored);
            request.runtimeInputs.canonicalBlockCFrame=scoreFieldFrame();
            videohelper::modelpayload::ImportedSceneExecutionReceipt receipt;
            check(surfaceExecution.executePreview(request,receipt,error)
                && surfaceExecution.executePreview(request,receipt,error) && surfaceBackend.preparations==1,
                "retained Surface product publication reuses identical sampled native resources");
            check(canonicalblockc::valid(request.runtimeInputs.canonicalBlockCFrame)
                && receipt.canonicalBlockCFrame==request.runtimeInputs.canonicalBlockCFrame
                && !surfaceBackend.lastRuntimeInputs.canonicalBlockCFrame
                && !surfaceBackend.lastRuntimeInputs.noteInstanceMapping
                && arbitgpu::validFixtureRuntimeInputs(surfaceBackend.lastRuntimeInputs),
                "material evaluation retains its score owner without authorizing native note instancing");
            request.frame.frame=90;
            check(surfaceExecution.executePreview(request,receipt,error)
                && surfaceExecution.executeExport(request,receipt,error) && surfaceBackend.preparations==2
                && receipt.canonicalBlockCFrame==request.runtimeInputs.canonicalBlockCFrame
                && arbitgpu::validFixtureRuntimeInputs(surfaceBackend.lastRuntimeInputs),
                "timeline Material Fields update the draw program once and export shares that exact result");
            request.frame.frame=0;
            check(surfaceExecution.executePreview(request,receipt,error) && surfaceBackend.preparations==3,
                "backward seeks re-evaluate retained material fields without historical state");
            check(!arbitgpu::validFixtureRuntimeInputs(request.runtimeInputs),
                "a material score lease alone remains invalid at the native note-instance boundary");
            request.runtimeInputs.noteInstanceMapping.emplace();
            check(surfaceExecution.executePreview(request,receipt,error)
                && surfaceExecution.executeExport(request,receipt,error)
                && surfaceBackend.lastRuntimeInputs.canonicalBlockCFrame==request.runtimeInputs.canonicalBlockCFrame
                && surfaceBackend.lastRuntimeInputs.noteInstanceMapping.has_value()
                && arbitgpu::validFixtureRuntimeInputs(surfaceBackend.lastRuntimeInputs)
                && arbitgpu::prepareNativeNoteInstances(surfaceBackend.lastRuntimeInputs).admitted,
                "an explicit note-instance mapping preserves the same score owner through preview and export");
            request.runtimeInputs.noteInstanceMapping.reset();
            const auto submissions=surfaceBackend.submissions;
            request.surfacePrograms.front().program.fields.front().sample.gain=0.8;
            check(!surfaceExecution.executePreview(request,receipt,error) && surfaceBackend.submissions==submissions,
                "changed authored Surface fields cannot reuse an existing immutable owner or draw last-good fallback");
        }
    }

    mixedSurfaceDiffractionAdmissionChecks();
    smoothGlassDiscreteGratingAdmissionChecks();

    std::printf ("glb-native-render-seam: %d/%d checks passed; submissions=%d\n",
                 checks - failures, checks, backend.submissions);
    return failures == 0 ? 0 : 1;
}
