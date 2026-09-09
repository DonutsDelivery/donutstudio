#include "../src/imported_scene_payload_execution.h"
#include "../src/imported_animated_scene_payload_execution.h"
#include "../../shared/DiffractionMaterialPresets.h"
#include "../src/surface_material_admission.h"
#include "../../shared/VisualStarterModelAssets.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

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

    arbitgpu::NativeFixtureScenePreparation prepare (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
        const std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram>& material) override
    {
        ++preparations;
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
    bool preparedWithSurfaceMaterial = false;
    arbitgpu::NativeFixtureMaterialKind preparedMaterialKind
        = arbitgpu::NativeFixtureMaterialKind::SurfacePbr;
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
        arbitgpu::NativeDeformationRuntimeInputs) override
    {
        ++submissions;
        arbitgpu::NativeDeformationSubmission result;
        result.rendered = true;
        result.frame = std::make_shared<FakeFrame> (
            "opengl", width, height, static_cast<std::uintptr_t> (submissions + 100));
        result.stats.drawCount = 1;
        result.stats.meshId = source->mesh.value;
        return result;
    }

    int preparations = 0;
    int submissions = 0;
};
} // namespace

int main()
{
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
        identityPayload, std::nullopt, 1u, 1u };
    auto otherClipIdentity = firstIdentity;
    otherClipIdentity.animationClipStableId = 2u;
    auto otherMeshIdentity = firstIdentity;
    otherMeshIdentity.meshStableId = 2u;
    auto otherSceneIdentity = firstIdentity;
    otherSceneIdentity.sceneIndex = 1u;
    std::weak_ptr<const visualmodelassetpayload::ImmutablePayload> retainedIdentityPayload
        = identityPayload;
    identityPayload.reset();
    check (firstIdentity != otherClipIdentity
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

    const auto noLightBytes
        = withoutStarterLightBinding (visualstartermodel::kAnimatedTriangleGlb);
    const auto noLightKey = exactKey ("animated-no-light", noLightBytes);
    check (publish (payloads, "animated-no-light-transfer", noLightKey, noLightBytes),
           "the downstream-rejection fixture publishes exact bytes");
    auto rejectedAnimatedRequest = animatedRequest;
    rejectedAnimatedRequest.operation.asset = noLightKey;
    const auto rejectedAnimatedExecuted = animatedExecution.executePreview (
        rejectedAnimatedRequest, animatedReceipt, error);
    if (rejectedAnimatedExecuted
        || error != "imported animated scene is outside the single-draw native subset")
        std::fprintf (stderr, "downstream candidate result: executed=%d error=%s\n",
                      rejectedAnimatedExecuted ? 1 : 0, error.c_str());
    check (! rejectedAnimatedExecuted
           && error == "imported animated scene is outside the single-draw native subset"
           && animatedBackend.submissions == 1,
           "a candidate rejected after animation decode does not render");
    check (animatedExecution.executePreview (animatedRequest, animatedReceipt, error)
           && animatedReceipt.valid() && animatedBackend.submissions == 2,
           "downstream candidate rejection preserves the previous admitted consumer and cache identity");

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
    clipA.sceneSnapshot = makeComposedProductScene (1, -1.5f);
    clipA.projectGeneration = 77;
    clipA.helperGeneration = 9;
    clipA.clipId = 101;
    clipA.staticPayloadIdentity = "composed-clip-a-v1";
    auto clipB = clipA;
    clipB.sceneSnapshot = makeComposedProductScene (20, 1.5f);
    clipB.clipId = 102;
    clipB.staticPayloadIdentity = "composed-clip-b-v1";
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

    auto malformedReplacement = clipA;
    malformedReplacement.staticPayloadIdentity = "composed-clip-a-malformed";
    auto malformedScene = std::make_shared<HarmonicMIDI::grid::Visual3DScene> (
        *clipA.sceneSnapshot);
    malformedScene->objects[1].id = malformedScene->objects[0].id;
    malformedReplacement.sceneSnapshot = malformedScene;
    const auto submissionsBeforeMalformed = cacheBackend.submissions;
    check (! cacheExecution.executePreview (malformedReplacement, clipAFrame2, error)
           && cacheBackend.submissions == submissionsBeforeMalformed
           && ! clipAFrame2.valid(),
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
    diffractionRequest.lighting.pathCount = 1;
    diffractionRequest.lighting.paths[0].kind
        = diffractionmaterial::LightingPathKind::Direct;
    diffractionRequest.lighting.paths[0].incident.radiance.fill (0.01f);
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

    std::printf ("glb-native-render-seam: %d/%d checks passed; submissions=%d\n",
                 checks - failures, checks, backend.submissions);
    return failures == 0 ? 0 : 1;
}
