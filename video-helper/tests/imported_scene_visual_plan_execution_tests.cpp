#include "imported_scene_visual_plan_execution.h"
#include "../../shared/DiffractionMaterialPresets.h"
#include "../../shared/DiffractionProductPlans.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace
{
int failures = 0;

void check(bool condition, const std::string& message)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

class FakeFrame final : public arbitgpu::NativeFixtureSceneFrame
{
public:
    std::string api = "opengl";
    std::uint32_t frameWidth = 640;
    std::uint32_t frameHeight = 360;
    std::uintptr_t image = 33;
    std::uintptr_t view = 44;
    std::uintptr_t depthImage = 55;
    std::uintptr_t depthView = 66;
    arbitgpu::NativeTexturePixelFormat depthFormat = arbitgpu::NativeTexturePixelFormat::R32Float;
    std::uint32_t descriptorWidth = 640;
    std::uint32_t descriptorSampleCount = 1;

    const std::string& backend() const noexcept override { return api; }
    std::uint32_t width() const noexcept override { return frameWidth; }
    std::uint32_t height() const noexcept override { return frameHeight; }
    std::uintptr_t colorImageHandle() const noexcept override { return image; }
    std::uintptr_t colorTextureViewHandle() const noexcept override { return view; }
    std::uintptr_t depthImageHandle() const noexcept override { return depthImage; }
    std::uintptr_t depthTextureViewHandle() const noexcept override { return depthView; }
    arbitgpu::NativeTextureViewDescriptor colorTextureDescriptor() const noexcept override
    {
        return { api, arbitgpu::NativeTextureViewKind::Texture2D,
                 api == "metal" ? arbitgpu::NativeTexturePixelFormat::Bgra8Unorm
                                  : arbitgpu::NativeTexturePixelFormat::Rgba8Unorm,
                 image, view, descriptorWidth, frameHeight, descriptorSampleCount, true, 77, 1 };
    }
    arbitgpu::NativeTextureViewDescriptor depthTextureDescriptor() const noexcept override
    {
        return { api, arbitgpu::NativeTextureViewKind::Texture2D,
                 depthFormat, depthImage, depthView, descriptorWidth, frameHeight,
                 descriptorSampleCount, true, 77, 1 };
    }
};

struct FakeReceipt
{
    bool complete = true;
    std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> frame;
    int owner = 0;
    std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> canonicalBlockCFrame;
};

struct FakeExecution
{
    using Receipt = FakeReceipt;

    int previewCalls = 0;
    int exportCalls = 0;
    int staticGpuUploads = 0;
    const HarmonicMIDI::grid::Visual3DScene* admittedScene = nullptr;
    videohelper::modelpayload::ImportedSceneRequest lastRequest;
    std::shared_ptr<FakeFrame> nextFrame = std::make_shared<FakeFrame>();
    bool succeed = true;

    static bool validReceipt(const Receipt& receipt) noexcept
    {
        return receipt.complete && receipt.frame != nullptr;
    }

    static const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& nativeFrame(
        const Receipt& receipt) noexcept
    {
        return receipt.frame;
    }

    void publishStaticPayload(int, std::uint64_t, std::uint64_t,
                              const std::string&) noexcept {}

    bool executePreview(const videohelper::modelpayload::ImportedSceneRequest& request,
                        Receipt& receipt, std::string& error)
    {
        ++previewCalls;
        lastRequest = request;
        if (request.sceneSnapshot != nullptr
            && request.sceneSnapshot.get() != admittedScene)
        {
            admittedScene = request.sceneSnapshot.get();
            ++staticGpuUploads;
        }
        if (!succeed)
        {
            error = "preview rejected";
            return false;
        }
        receipt = { true, nextFrame, 1, nullptr };
        return true;
    }

    bool executeExport(const videohelper::modelpayload::ImportedSceneRequest& request,
                       Receipt& receipt, std::string& error)
    {
        ++exportCalls;
        lastRequest = request;
        if (request.sceneSnapshot != nullptr
            && request.sceneSnapshot.get() != admittedScene)
        {
            admittedScene = request.sceneSnapshot.get();
            ++staticGpuUploads;
        }
        if (!succeed)
        {
            error = "export rejected";
            return false;
        }
        receipt = { true, nextFrame, 2, nullptr };
        return true;
    }
};

struct FakeLayer
{
    std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> canonicalBlockCFrame;
    unsigned texture = 0;
    std::string nativeTextureBackend;
    std::uintptr_t nativeTextureView = 0;
    arbitgpu::NativeTextureViewDescriptor nativeTextureDescriptor;
    unsigned depthTexture = 0;
    std::string nativeDepthTextureBackend;
    std::uintptr_t nativeDepthTextureView = 0;
    arbitgpu::NativeTextureViewDescriptor nativeDepthTextureDescriptor;
    int depthWidth = 0;
    int depthHeight = 0;
    int texWidth = 0;
    int texHeight = 0;
};

std::uint32_t floatBits(float value)
{
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

int surfaceChannels(surfacematerial::ValueType type)
{
    switch (type)
    {
        case surfacematerial::ValueType::Scalar:
        case surfacematerial::ValueType::UInt: return 1;
        case surfacematerial::ValueType::Vec2: return 2;
        case surfacematerial::ValueType::Vec3: return 3;
        case surfacematerial::ValueType::Vec4: return 4;
        case surfacematerial::ValueType::Invalid: break;
    }
    return 0;
}

std::string surfaceDataType(surfacematerial::ValueType type)
{
    return type == surfacematerial::ValueType::UInt
        ? "integer" : std::string(surfacematerial::token(type));
}

surfacematerialbinding::ImportedSceneMaterialRequest materialRequest()
{
    surfacematerialbinding::ImportedSceneMaterialRequest material;
    material.scene.value = 17;
    material.sceneRevision = 31;
    material.structuralRevision = 32;
    material.evaluationRevision = 33;
    material.programRevision = 32;
    material.program.textureSlotCount = 1;
    for (const auto output : surfacematerial::kSurfaceOutputs)
    {
        surfacematerial::Operation operation;
        operation.id = static_cast<surfacematerial::ValueId>(
            material.program.operations.size() + 1u);
        operation.kind = output == surfacematerial::OutputSemantic::MaterialId
            ? surfacematerial::OperationKind::UIntConstant
            : surfacematerial::OperationKind::FloatConstant;
        operation.resultType = surfacematerial::outputType(output);
        if (output == surfacematerial::OutputSemantic::BaseColor)
            operation.literal = { 0.25f, -0.0f, 0.75f, 0.0f };
        else if (output == surfacematerial::OutputSemantic::Normal)
            operation.literal = { 0.0f, 0.0f, 1.0f, 0.0f };
        else if (output == surfacematerial::OutputSemantic::Opacity)
            operation.literal[0] = 1.0f;
        else if (output == surfacematerial::OutputSemantic::Ior)
            operation.literal[0] = 1.5f;
        else if (output == surfacematerial::OutputSemantic::MaterialId)
            operation.unsignedLiteral = 23;
        material.program.outputs[static_cast<std::size_t>(output)] = operation.id;
        material.program.operations.push_back(operation);
    }
    material.binding.targetKind
        = surfacematerialbinding::BindingTargetKind::ObjectOverride;
    material.binding.object.value = 23;
    material.binding.surfaceMaterialDigest = std::string(64, 'b');
    material.binding.surfaceMaterialRevision = 32;
    surfacematerialbinding::TextureSlotBinding texture;
    texture.slot = 0;
    texture.source = surfacematerialbinding::TextureSourceKind::ImportedBaseColor;
    for (std::size_t index = 0; index < texture.videoResource.size(); ++index)
        texture.videoResource[index] = static_cast<std::uint8_t>(index + 1u);
    material.binding.textures.push_back(texture);
    return material;
}

diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest
diffractionMaterialRequest()
{
    diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest material;
    material.scene.value = 1;
    material.sceneRevision = 3;
    material.structuralRevision = 9;
    material.evaluationRevision = 33;
    material.materialRevision = 9;
    material.object.value = 23;
    material.material = diffractionmaterial::makeAluminiumBinaryGratingPreset();
    material.lighting = diffractionmaterial::makeReferenceLighting();
    std::string error;
    const auto admitted = diffractionmaterial::admit(material.material, error);
    if (admitted) material.structuralDigest = admitted->structuralDigest();
    return material;
}

diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest
spatialDiffractionMaterialRequest()
{
    auto material = diffractionMaterialRequest();
    diffractivefoil::Description foil;
    foil.physicalBsdf = material.material;
    foil.workBudget.maximumEvaluations = diffractivefoil::kMaximumEvaluations;
    foil.grooveField = {{
        { { 1.0f, 0.0f }, foil.physicalBsdf.geometry.grooveSpacingNanometres, 0.0f },
        { { 0.0f, 1.0f }, diffractionmaterial::kMaximumGrooveSpacingNanometres, 1.0f },
        { { 0.6f, 0.8f }, 800.0f, 0.25f },
        { { 0.6f, -0.8f }, 2000.0f, 0.75f }
    }};
    material.spatialFoil = foil;
    std::string foilError;
    const auto admittedFoil = diffractivefoil::admit(foil, foilError);
    if (admittedFoil)
        material.structuralDigest = admittedFoil->structuralDigest();
    return material;
}

videowire::VertexModifierIr vertexModifierRequest()
{
    using Operation = videowire::VertexModifierOperation;
    videowire::VertexModifierIr vertex;
    const auto append = [&vertex](videowire::VertexModifierStableId id,
                                  Operation operation,
                                  std::initializer_list<videowire::VertexModifierStableId> inputs,
                                  std::initializer_list<double> parameters)
    {
        const auto schema = videowire::vertexModifierOperationSchema(operation);
        videowire::VertexModifierRecord record;
        record.stableId = id;
        record.operation = operation;
        record.resultType = schema.resultType;
        record.inputCount = schema.inputCount;
        record.parameterCount = schema.parameterCount;
        std::copy(inputs.begin(), inputs.end(), record.inputs.begin());
        std::copy(parameters.begin(), parameters.end(), record.parameters.begin());
        vertex.records.push_back(record);
    };
    append(101, Operation::importedPosition, {}, {});
    append(102, Operation::vec3Constant, {}, { 0.0, 0.25, -0.0 });
    append(103, Operation::boundedDisplacementOutput, { 101, 102 }, { 0.5 });
    vertex.rootId = 103;
    return vertex;
}

std::string replaceLine(std::string value, std::size_t lineIndex,
                        const std::string& replacement)
{
    std::size_t start = 0;
    for (std::size_t index = 0; index < lineIndex; ++index)
    {
        start = value.find('\n', start);
        if (start == std::string::npos) return {};
        ++start;
    }
    const auto end = value.find('\n', start);
    if (end == std::string::npos) return {};
    value.replace(start, end - start, replacement);
    return value;
}

enum class CrossVersionAdmissionStage
{
    baseSyntax,
    flagSchema,
    cameraSyntax,
    lightSyntax,
    deformationSyntax,
    materialSyntax,
    diffractionSyntax,
    trailingBytes,
    destinationAuthority
};

struct DirectionalRelabelFixture
{
    std::string_view identity;
    std::string_view sourceHeader;
    std::string_view destinationHeader;
    std::string sourceBytes;
    std::string relabeledBytes;
    CrossVersionAdmissionStage expectedStage;
};

std::string relabelHeaderOnly(const std::string& sourceBytes,
                              std::string_view destinationHeader)
{
    return replaceLine(sourceBytes, 0, std::string(destinationHeader));
}

CrossVersionAdmissionStage probeCrossVersionAdmission(
    const std::string& encoded, std::string_view destinationHeader)
{
    std::istringstream input(encoded);
    std::string header;
    visualimportedscenerender::Request request;
    if (!std::getline(input, header)
        || header != destinationHeader
        || !visualimportedscenerender::detail::decodeBase(input, request))
        return CrossVersionAdmissionStage::baseSyntax;

    const bool v8 = destinationHeader
        == visualimportedscenerender::kCompiledOperationHeaderV8;
    const bool v7 = destinationHeader
        == visualimportedscenerender::kCompiledOperationHeaderV7;
    if (v8)
    {
        int deformation = -1, material = -1, vertex = -1;
        int diffraction = -1, camera = -1, light = -1;
        if (!(input >> deformation >> material >> vertex >> diffraction >> camera >> light)
            || (deformation != 0 && deformation != 1)
            || (material != 0 && material != 1)
            || (vertex != 0 && vertex != 1)
            || (diffraction != 0 && diffraction != 1)
            || (camera != 0 && camera != 1) || light != 1
            || (vertex != 0 && material == 0)
            || (material != 0 && diffraction != 0))
            return CrossVersionAdmissionStage::flagSchema;
        if (camera != 0)
        {
            HarmonicMIDI::grid::SceneCameraRecord value;
            if (!visualimportedscenerender::detail::decodeCamera(input, value))
                return CrossVersionAdmissionStage::cameraSyntax;
            request.camera = value;
        }
        HarmonicMIDI::grid::SceneLightRecord value;
        if (!visualimportedscenerender::detail::decodeLight(input, value))
            return CrossVersionAdmissionStage::lightSyntax;
        request.light = value;
        if (deformation != 0)
            return CrossVersionAdmissionStage::deformationSyntax;
        if (material != 0)
        {
            surfacematerialbinding::ImportedSceneMaterialRequest value;
            if (!visualimportedscenerender::detail::decodeMaterial(input, value, vertex != 0))
                return CrossVersionAdmissionStage::materialSyntax;
            request.material = value;
        }
        if (diffraction != 0)
        {
            diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest value;
            if (!visualimportedscenerender::detail::decodeDiffractionMaterial(input, value))
                return CrossVersionAdmissionStage::diffractionSyntax;
            request.diffractionMaterial = value;
        }
    }
    else
    {
        int deformation = -1, diffraction = -1, camera = -1;
        if (!(input >> deformation >> diffraction >> camera)
            || (deformation != 0 && deformation != 1) || diffraction != 1
            || (camera != 0 && camera != 1))
            return CrossVersionAdmissionStage::flagSchema;
        if (camera != 0)
        {
            HarmonicMIDI::grid::SceneCameraRecord value;
            if (!visualimportedscenerender::detail::decodeCamera(input, value))
                return CrossVersionAdmissionStage::cameraSyntax;
            request.camera = value;
        }
        if (deformation != 0)
            return CrossVersionAdmissionStage::deformationSyntax;
        diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest value;
        const bool decoded = v7
            ? visualimportedscenerender::detail::decodeDiffractionMaterial(input, value)
            : visualimportedscenerender::detail::decodeLegacyV6DiffractionMaterial(input, value);
        if (!decoded) return CrossVersionAdmissionStage::diffractionSyntax;
        request.diffractionMaterial = value;
    }
    input >> std::ws;
    if (!input.eof()) return CrossVersionAdmissionStage::trailingBytes;
    return CrossVersionAdmissionStage::destinationAuthority;
}

visualanimationimport::Request deformationRequest()
{
    visualanimationimport::Request request;
    request.sourceStableId = 71;
    request.deformationStableId = 73;
    request.asset.id = "model-asset-1";
    request.asset.version = 3;
    request.asset.contentSha256 = std::string(64, 'a');
    request.asset.sourceMediaType = "model/gltf-binary";
    request.asset.sourceByteSize = 4096;
    request.sceneIndex = 0;
    request.clipName = "Idle";
    request.playback.timelineSeconds = 1.0;
    request.playback.speed = 1.5;
    request.playback.trimStartSeconds = 0.25;
    request.playback.trimEndSeconds = 0.75;
    request.playback.weight = 0.75;
    request.schedule = { request.sourceStableId, request.deformationStableId };
    return request;
}

visualimportedscenerender::Request renderRequest(
    bool withMaterial = false, bool withDeformation = false,
    bool withDiffractionMaterial = false)
{
    visualimportedscenerender::Request request;
    request.sourceStableId = 71;
    request.renderStableId = 72;
    request.asset.id = "model-asset-1";
    request.asset.version = 3;
    request.asset.contentSha256 = std::string(64, 'a');
    request.asset.sourceMediaType = "model/gltf-binary";
    request.asset.sourceByteSize = 4096;
    request.sceneIndex = 0;
    if (withMaterial) request.material = materialRequest();
    if (withDiffractionMaterial)
        request.diffractionMaterial = diffractionMaterialRequest();
    if (withDeformation) request.deformation = deformationRequest();
    return request;
}

std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> composedScene()
{
    using namespace HarmonicMIDI::grid;
    Visual3DScene scene;
    scene.id.value = 7;
    scene.activeCamera.value = 50;
    scene.ambientColor = { 0.1f, 0.2f, 0.3f };
    scene.vertexCount = 6;
    scene.indexCount = 6;
    scene.objectCount = 2;
    scene.materialCount = 2;
    scene.lightCount = 1;
    scene.cameraCount = 1;
    for (std::size_t object = 0; object < 2; ++object)
    {
        const auto first = static_cast<std::uint32_t>(object * 3);
        scene.vertices[first] = { { -1.0f, -1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f }, {} };
        scene.vertices[first + 1] = { { 1.0f, -1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f }, {} };
        scene.vertices[first + 2] = { { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f }, {} };
        scene.indices[first] = 0;
        scene.indices[first + 1] = 1;
        scene.indices[first + 2] = 2;
        scene.materials[object].id.value = static_cast<std::uint32_t>(20 + object);
        scene.objects[object].id.value = static_cast<std::uint32_t>(10 + object);
        scene.objects[object].material = scene.materials[object].id;
        scene.objects[object].firstVertex = first;
        scene.objects[object].vertexCount = 3;
        scene.objects[object].firstIndex = first;
        scene.objects[object].indexCount = 3;
    }
    scene.objects[1].transform.translation.x = 2.0f;
    scene.lights[0].id.value = 40;
    scene.lights[0].kind = SceneLightKind::Directional;
    scene.cameras[0].id = scene.activeCamera;
    scene.cameras[0].transform.translation.z = 4.0f;
    return std::make_shared<const Visual3DScene>(std::move(scene));
}

videowire::CompiledVisualLayerPlan composedScenePlan()
{
    videowire::CompiledVisualLayerPlan plan;
    plan.clipId = 8;
    plan.structuralRevision = 9;
    plan.producerValidated = true;
    plan.nodeKinds = {
        "visual.3d.transform", "visual.3d.material.pbr", "visual.3d.scene.primitive",
        "visual.3d.transform", "visual.3d.material.pbr", "visual.3d.scene.primitive",
        "visual.3d.scene.compose", "visual.3d.transform", "visual.3d.camera.perspective",
        "visual.3d.transform", "visual.3d.light.directional", "visual.3d.render"
    };
    for (int id = 0; id < static_cast<int>(plan.nodeKinds.size()); ++id)
    {
        plan.nodeIds.push_back(id);
        plan.operations.push_back({ id, plan.nodeKinds[static_cast<std::size_t>(id)],
            id == 11 ? "native-gpu" : "control-eval", "" });
    }
    const auto port = [&](int node, int id, const char* direction, const char* carrier,
                          const char* type, const char* format = "unspecified",
                          const char* color = "unspecified")
    {
        plan.ports.push_back({ node, id, 1, direction, carrier, type, format, color });
    };
    port(0, 0, "out", "control", "transform3D");
    port(1, 0, "out", "control", "material");
    port(2, 0, "in", "control", "transform3D");
    port(2, 1, "in", "control", "material");
    port(2, 2, "in", "control", "light");
    port(2, 3, "in", "control", "camera");
    port(2, 4, "out", "control", "scene3D");
    port(3, 0, "out", "control", "transform3D");
    port(4, 0, "out", "control", "material");
    port(5, 0, "in", "control", "transform3D");
    port(5, 1, "in", "control", "material");
    port(5, 2, "in", "control", "light");
    port(5, 3, "in", "control", "camera");
    port(5, 4, "out", "control", "scene3D");
    port(6, 0, "in", "control", "scene3D");
    port(6, 1, "in", "control", "scene3D");
    port(6, 2, "in", "control", "scene3D");
    port(6, 3, "in", "control", "scene3D");
    port(6, 4, "out", "control", "scene3D");
    port(7, 0, "out", "control", "transform3D");
    port(8, 0, "in", "control", "transform3D");
    port(8, 1, "out", "control", "camera");
    port(9, 0, "out", "control", "transform3D");
    port(10, 0, "in", "control", "transform3D");
    port(10, 1, "out", "control", "light");
    port(11, 0, "in", "control", "scene3D");
    port(11, 1, "out", "frame", "image", "rgba8", "sRGB");
    port(11, 2, "in", "control", "material");
    port(11, 3, "in", "control", "mesh");
    port(11, 4, "out", "frame", "depth", "r32f");
    port(11, 5, "in", "control", "camera");
    port(11, 6, "in", "control", "light");
    plan.edges = {
        { 0, 0, 2, 0 }, { 1, 0, 2, 1 }, { 3, 0, 5, 0 }, { 4, 0, 5, 1 },
        { 2, 4, 6, 0 }, { 5, 4, 6, 1 }, { 6, 4, 11, 0 },
        { 7, 0, 8, 0 }, { 8, 1, 2, 3 }, { 8, 1, 5, 3 }, { 8, 1, 11, 5 },
        { 9, 0, 10, 0 }, { 10, 1, 2, 2 }, { 10, 1, 5, 2 }, { 10, 1, 11, 6 }
    };
    visualimportedscenerender::Request request;
    request.sourceStableId = 7;
    request.renderStableId = 12;
    request.sceneSnapshot = composedScene();
    request.structuralRevision = 9;
    request.evaluationRevision = 12;
    plan.operations.back().payloadXml = visualimportedscenerender::encode(request);
    return plan;
}

videowire::CompiledVisualLayerPlan importedScenePlan(
    bool withMaterial = false, bool withDeformation = false,
    bool withDiffractionMaterial = false)
{
    videowire::CompiledVisualLayerPlan plan;
    plan.clipId = 7;
    plan.structuralRevision = 9;
    plan.producerValidated = true;
    plan.nodeKinds = {
        std::string(visualimportedscenerender::kSourceNodeKind),
        std::string(visualimportedscenerender::kRenderNodeKind)
    };
    plan.nodeIds = { 70, 71 };
    const std::array<const char*, 5> sourceTypes {
        "mesh", "skeleton", "morphTargets", "animationClip", "scene3D"
    };
    for (int port = 0; port < 5; ++port)
        plan.ports.push_back({ 70, port, 1, "out", "control",
                              sourceTypes[static_cast<std::size_t>(port)],
                              "unspecified", "unspecified" });
    plan.ports.push_back({ 71, 0, 1, "in", "control", "scene3D",
                           "unspecified", "unspecified" });
    plan.ports.push_back({ 71, 1, 1, "out", "frame", "image",
                           "rgba8", "sRGB" });
    plan.edges = { { 70, 4, 71, 0 } };

    const auto request = renderRequest(
        withMaterial, withDeformation, withDiffractionMaterial);
    plan.operations.push_back({
        70, std::string(visualimportedscenerender::kSourceNodeKind),
        std::string(visualimportedscenerender::kSourceBackendCapability), "" });
    if (withDeformation)
    {
        plan.nodeKinds.insert(plan.nodeKinds.end() - 1,
                              std::string(visualanimationoperation::kDeformationNodeKind));
        plan.nodeIds.insert(plan.nodeIds.end() - 1, 72);
        const std::array<const char*, 5> deformationTypes {
            "mesh", "skeleton", "morphTargets", "animationClip", "mesh"
        };
        for (int port = 0; port < 5; ++port)
            plan.ports.push_back({
                72, port, 1, port == 4 ? "out" : "in", "control",
                deformationTypes[static_cast<std::size_t>(port)],
                "unspecified", "unspecified" });
        plan.ports.push_back({ 71, 2, 1, "in", "control", "material",
                               "unspecified", "unspecified" });
        plan.ports.push_back({ 71, 3, 1, "in", "control", "mesh",
                               "unspecified", "unspecified" });
        plan.ports.push_back({ 71, 4, 1, "out", "frame", "depth",
                               "r32f", "unspecified" });
        for (int port = 0; port < 4; ++port)
            plan.edges.push_back({ 70, port, 72, port });
        plan.edges.push_back({ 72, 4, 71, 3 });
        plan.operations.push_back({
            72, std::string(visualanimationoperation::kDeformationNodeKind),
            std::string(visualanimationoperation::kDeformationBackendCapability),
            visualanimationoperation::encode(*request.deformation) });
    }
    if (withMaterial)
    {
        const int firstMaterialValueNode = withDeformation ? 73 : 72;
        const int materialTerminalNode = firstMaterialValueNode + 10;
        const auto& program = request.material->program;
        for (std::size_t index = 0; index < program.operations.size(); ++index)
        {
            const auto& value = program.operations[index];
            const auto nodeId = firstMaterialValueNode + static_cast<int>(index);
            const auto kind = value.resultType == surfacematerial::ValueType::UInt
                ? "visual.surface.constant.uint"
                : std::string("visual.surface.constant.")
                    + std::string(surfacematerial::token(value.resultType));
            plan.nodeKinds.insert(plan.nodeKinds.end() - 1, kind);
            plan.nodeIds.insert(plan.nodeIds.end() - 1, nodeId);
            plan.ports.push_back({
                nodeId, 0, surfaceChannels(value.resultType),
                "out", "control", surfaceDataType(value.resultType),
                "unspecified", "unspecified" });
            plan.operations.push_back({ nodeId, kind, "control-eval", "" });
        }
        plan.nodeKinds.insert(plan.nodeKinds.end() - 1, "visual.surface.material");
        plan.nodeIds.insert(plan.nodeIds.end() - 1, materialTerminalNode);
        for (const auto output : surfacematerial::kSurfaceOutputs)
        {
            const auto port = static_cast<int>(output);
            const auto type = surfacematerial::outputType(output);
            const auto valueNode = firstMaterialValueNode + port;
            plan.ports.push_back({
                materialTerminalNode, port,
                surfaceChannels(type), "in", "control", surfaceDataType(type),
                "unspecified", "unspecified" });
            plan.edges.push_back({ valueNode, 0, materialTerminalNode, port });
        }
        plan.ports.push_back({ materialTerminalNode, 10, 1, "out", "control", "material",
                               "unspecified", "unspecified" });
        if (!withDeformation)
            plan.ports.push_back({ 71, 2, 1, "in", "control", "material",
                                   "unspecified", "unspecified" });
        plan.edges.push_back({ materialTerminalNode, 10, 71, 2 });
        plan.operations.push_back({
            materialTerminalNode, "visual.surface.material", "control-eval", "" });
    }
    if (withDiffractionMaterial)
    {
        const int materialTerminalNode = withDeformation ? 73 : 72;
        plan.nodeKinds.insert(
            plan.nodeKinds.end() - 1, "visual.material.diffraction-grating");
        plan.nodeIds.insert(plan.nodeIds.end() - 1, materialTerminalNode);
        plan.ports.push_back({ materialTerminalNode, 0, 1, "out", "control",
                               "material", "unspecified", "unspecified" });
        if (! withDeformation)
            plan.ports.push_back({ 71, 2, 1, "in", "control", "material",
                                   "unspecified", "unspecified" });
        plan.edges.push_back({ materialTerminalNode, 0, 71, 2 });
        plan.operations.push_back({
            materialTerminalNode, "visual.material.diffraction-grating",
            "control-eval", "" });
    }
    plan.operations.push_back({
        71, std::string(visualimportedscenerender::kRenderNodeKind),
        std::string(visualimportedscenerender::kRenderBackendCapability),
        visualimportedscenerender::encode(request) });
    return plan;
}
} // namespace

int main()
{
    using videohelper::importedscene::NativeImportedSceneRenderUse;
    using videohelper::importedscene::VisualImportedScenePreparation;
    using videohelper::importedscene::prepareVisualImportedSceneLayer;
    using videohelper::importedscene::prepareVisualImportedSceneLayerAtTime;

    const auto legacyRequest = renderRequest();
    const auto legacyEncoded = visualimportedscenerender::encode(legacyRequest);
    const auto expectedLegacy = std::string(
        "visual.imported-scene.render.v1\n"
        "71 72\n"
        "\"model-asset-1\"\n"
        "3\n"
        "\"") + std::string(64, 'a')
        + "\"\n\"model/gltf-binary\"\n4096\n1 0\n";
    visualimportedscenerender::Request decodedLegacy;
    check(legacyEncoded == expectedLegacy,
          "a request without material preserves the v1 bytes exactly");
    check(visualimportedscenerender::decode(legacyEncoded, decodedLegacy)
              && visualimportedscenerender::sameRequest(
                  legacyRequest, decodedLegacy)
              && !decodedLegacy.material,
          "the legacy v1 request still decodes without material");

    auto boundSceneRequest = renderRequest();
    HarmonicMIDI::grid::SceneCameraRecord boundCamera;
    boundCamera.id.value = 701;
    boundCamera.transform.translation.z = 4.0f;
    boundCamera.verticalFovRadians = 0.75f;
    boundCamera.nearPlane = 0.25f;
    boundCamera.farPlane = 500.0f;
    boundSceneRequest.camera = boundCamera;
    HarmonicMIDI::grid::SceneLightRecord boundLight;
    boundLight.id.value = 702;
    boundLight.kind = HarmonicMIDI::grid::SceneLightKind::Point;
    boundLight.transform.translation = { 1.0f, 2.0f, 3.0f };
    boundLight.color = { 0.25f, 0.5f, 0.75f };
    boundLight.intensity = 8.0f;
    boundLight.range = 16.0f;
    boundSceneRequest.light = boundLight;
    const auto boundSceneEncoded = visualimportedscenerender::encode(boundSceneRequest);
    visualimportedscenerender::Request decodedBoundScene;
    check(boundSceneEncoded.rfind(
              std::string(visualimportedscenerender::kCompiledOperationHeaderV8) + "\n",
              0) == 0
              && visualimportedscenerender::decode(boundSceneEncoded, decodedBoundScene)
              && visualimportedscenerender::sameRequest(boundSceneRequest, decodedBoundScene)
              && visualimportedscenerender::encode(decodedBoundScene) == boundSceneEncoded,
          "the immutable camera and light request has one canonical exact round trip");
    auto invalidBoundScene = boundSceneRequest;
    invalidBoundScene.light->id = {};
    check(visualimportedscenerender::encode(invalidBoundScene).empty(),
          "light transport rejects a missing stable identity");
    auto malformedBoundScene = boundSceneEncoded;
    const auto lightIdentity = malformedBoundScene.find("\n702 ");
    check(lightIdentity != std::string::npos,
          "v8 encodes the exact light identity");
    if (lightIdentity != std::string::npos)
        malformedBoundScene.replace(lightIdentity + 1, 3, "0");
    check(lightIdentity != std::string::npos
              && !visualimportedscenerender::decode(
                  malformedBoundScene, decodedBoundScene),
          "v8 rejects a malformed light identity deterministically");

    visualimportedscenerender::Request composedRequest;
    composedRequest.sourceStableId = 7;
    composedRequest.renderStableId = 12;
    composedRequest.sceneSnapshot = composedScene();
    composedRequest.structuralRevision = 9;
    composedRequest.evaluationRevision = 12;
    const auto composedEncoded = visualimportedscenerender::encode(composedRequest);
    visualimportedscenerender::Request decodedComposed;
    check(composedEncoded.rfind(
              std::string(visualimportedscenerender::kCompiledOperationHeaderV8)
                  + "\nscene ", 0) == 0
              && visualimportedscenerender::decode(composedEncoded, decodedComposed)
              && visualimportedscenerender::sameRequest(composedRequest, decodedComposed)
              && decodedComposed.sceneSnapshot
              && decodedComposed.sceneSnapshot->objectCount == 2
              && decodedComposed.sceneSnapshot->objects[1].firstVertex == 3
              && decodedComposed.sceneSnapshot->objects[1].firstIndex == 3
              && decodedComposed.sceneSnapshot->activeCamera.value == 50
              && decodedComposed.sceneSnapshot->lights[0].id.value == 40
              && visualimportedscenerender::encode(decodedComposed) == composedEncoded,
          "v8 preserves the exact composed object camera and light topology");
    auto invalidComposed = composedRequest;
    auto invalidScene = std::make_shared<HarmonicMIDI::grid::Visual3DScene>(
        *composedRequest.sceneSnapshot);
    invalidScene->objects[1].id = invalidScene->objects[0].id;
    invalidComposed.sceneSnapshot = std::move(invalidScene);
    check(visualimportedscenerender::encode(invalidComposed).empty(),
          "v8 rejects duplicate composed object identity before publication");

    const auto exactMaterialRequest = renderRequest(true);
    const auto materialEncoded
        = visualimportedscenerender::encode(exactMaterialRequest);
    visualimportedscenerender::Request decodedMaterial;
    check(materialEncoded.rfind(
              std::string(visualimportedscenerender::kCompiledOperationHeaderV2)
                  + "\n",
              0) == 0
              && visualimportedscenerender::decode(
                  materialEncoded, decodedMaterial)
              && visualimportedscenerender::sameRequest(
                  exactMaterialRequest, decodedMaterial)
              && visualimportedscenerender::encode(decodedMaterial)
                  == materialEncoded,
          "the v2 material request has one canonical exact round trip");
    check(decodedMaterial.material
              && decodedMaterial.material->sceneSnapshot == nullptr
              && floatBits(decodedMaterial.material->program.operations.front()
                               .literal[1])
                  == floatBits(-0.0f)
              && decodedMaterial.material->binding.textures.size() == 1
              && decodedMaterial.material->binding.textures.front()
                     .videoResource[15] == 16,
          "v2 preserves float bits and value fields without a scene owner");

    const auto exactDeformationRequest = renderRequest(false, true);
    const auto deformationEncoded
        = visualimportedscenerender::encode(exactDeformationRequest);
    visualimportedscenerender::Request decodedDeformation;
    check(deformationEncoded.rfind(
              std::string(visualimportedscenerender::kCompiledOperationHeaderV3)
                  + "\n",
              0) == 0
              && visualimportedscenerender::decode(
                  deformationEncoded, decodedDeformation)
              && visualimportedscenerender::sameRequest(
                  exactDeformationRequest, decodedDeformation)
              && visualimportedscenerender::encode(decodedDeformation)
                  == deformationEncoded,
          "the v3 deformation request has one canonical exact round trip");
    check(decodedDeformation.deformation
              && decodedDeformation.deformation->deformationStableId == 73
              && decodedDeformation.deformation->clipName == "Idle",
          "v3 retains the exact deformation identity and playback request");

    auto exactVertexRequest = renderRequest(true, true);
    exactVertexRequest.material->vertexModifier = vertexModifierRequest();
    const auto vertexEncoded = visualimportedscenerender::encode(exactVertexRequest);
    visualimportedscenerender::Request decodedVertex;
    check(vertexEncoded.rfind(
              std::string(visualimportedscenerender::kCompiledOperationHeaderV4) + "\n",
              0) == 0
              && visualimportedscenerender::decode(vertexEncoded, decodedVertex)
              && visualimportedscenerender::sameRequest(exactVertexRequest, decodedVertex)
              && visualimportedscenerender::encode(decodedVertex) == vertexEncoded,
          "the v4 material and vertex request has one canonical exact round trip");
    check(decodedVertex.material && decodedVertex.material->vertexModifier
              && decodedVertex.material->vertexModifier->rootId == 103
              && floatBits(static_cast<float>(
                     decodedVertex.material->vertexModifier->records[1].parameters[2]))
                   == floatBits(-0.0f),
          "v4 preserves the graph-native vertex stage and signed parameter bits");

    auto unsupportedVertex = exactVertexRequest;
    unsupportedVertex.material->vertexModifier->records[1].parameters[3] = 1.0;
    check(visualimportedscenerender::encode(unsupportedVertex).empty(),
          "vertex transport rejects data outside the typed parameter shape");
    unsupportedVertex = exactVertexRequest;
    unsupportedVertex.material->vertexModifier->records.push_back(
        unsupportedVertex.material->vertexModifier->records.front());
    check(visualimportedscenerender::encode(unsupportedVertex).empty(),
          "vertex transport rejects duplicate or unreachable operations");

    auto foreignOwner = exactMaterialRequest;
    foreignOwner.material->sceneSnapshot
        = std::make_shared<const HarmonicMIDI::grid::Visual3DScene>();
    visualimportedscenerender::Request decodedForeignOwner;
    check(visualimportedscenerender::encode(foreignOwner) == materialEncoded
              && visualimportedscenerender::decode(
                  visualimportedscenerender::encode(foreignOwner),
                  decodedForeignOwner)
              && decodedForeignOwner.material
              && decodedForeignOwner.material->sceneSnapshot == nullptr,
          "material transport excludes rather than serializes a scene snapshot owner");
    auto tooManyOperations = exactMaterialRequest;
    tooManyOperations.material->program.operations.resize(
        surfacematerial::kMaximumOperations + 1u);
    check(visualimportedscenerender::encode(tooManyOperations).empty(),
          "material transport rejects an oversized operation count");
    auto tooManyTextures = exactMaterialRequest;
    tooManyTextures.material->binding.textures.resize(
        surfacematerialbinding::kMaximumTextureBindings + 1u);
    check(visualimportedscenerender::encode(tooManyTextures).empty(),
          "material transport rejects an oversized texture-binding count");
    visualimportedscenerender::Request rejectedMaterial;
    check(!visualimportedscenerender::decode(
              replaceLine(materialEncoded, 9, "1 1 129"), rejectedMaterial),
          "v2 rejects an operation count above the hard bound before allocation");
    check(!visualimportedscenerender::decode(
              replaceLine(materialEncoded, 0,
                          std::string(visualimportedscenerender::kCompiledOperationHeader)),
              rejectedMaterial),
          "material fields cannot be smuggled through the v1 header");

    std::string error;
    FakeExecution execution;
    FakeLayer previewLayer;
    std::vector<FakeReceipt> previewOwners;

    check(prepareVisualImportedSceneLayer(
              {}, 7, 640, 360, NativeImportedSceneRenderUse::Preview,
              &execution, previewLayer, previewOwners, error)
              == VisualImportedScenePreparation::notApplicable,
          "a clip without an imported scene render stays on its ordinary route");

    auto plan = importedScenePlan();
    std::vector<videowire::CompiledVisualLayerPlan> plans { plan };
    check(prepareVisualImportedSceneLayer(
              plans, 7, 640, 360, NativeImportedSceneRenderUse::Preview,
              &execution, previewLayer, previewOwners, error)
              == VisualImportedScenePreparation::rendered,
          "the exact imported scene plan renders for preview");
    check(execution.previewCalls == 1 && execution.exportCalls == 0,
          "preview uses the preview execution owner");
    check(previewOwners.size() == 1 && previewOwners.front().owner == 1,
          "preview retains its complete frame receipt");
    check(previewOwners.front().frame->depthImageHandle() == 55
              && previewOwners.front().frame->depthTextureViewHandle() == 66,
          "preview retains the native Render 3D depth publication");
    check(previewLayer.texture == 44 && previewLayer.nativeTextureView == 44
              && previewLayer.nativeTextureBackend == "opengl"
              && previewLayer.texWidth == 640 && previewLayer.texHeight == 360,
          "the exact native OpenGL texture view reaches the ordinary layer");
    check(previewLayer.depthTexture == 66
              && previewLayer.nativeDepthTextureBackend == "opengl"
              && previewLayer.nativeDepthTextureView == 66
              && previewLayer.depthWidth == 640 && previewLayer.depthHeight == 360,
          "the exact native OpenGL R32F depth view reaches the ordinary layer");
    check(execution.lastRequest.asset.id == "model-asset-1"
              && execution.lastRequest.asset.version == 3
              && execution.lastRequest.sceneIndex == 0,
          "the compiled exact asset identity reaches payload execution");

    auto composedPlan = composedScenePlan();
    std::vector<videowire::CompiledVisualLayerPlan> composedPlans { composedPlan };
    FakeLayer composedLayer;
    std::vector<FakeReceipt> composedOwners;
    videohelper::importedscene::VisualImportedScenePlanCache composedPlanCache;
    check(prepareVisualImportedSceneLayer(
              composedPlans, 8, 640, 360, NativeImportedSceneRenderUse::Export,
              &execution, &composedPlanCache, composedLayer, composedOwners, error)
              == VisualImportedScenePreparation::rendered,
          "the production-shaped composed Scene3D V7 plan reaches export execution");
    check(execution.exportCalls == 1 && execution.lastRequest.sceneSnapshot
              && execution.lastRequest.sceneSnapshot->objectCount == 2
              && execution.lastRequest.sceneSnapshot->objects[1].firstVertex == 3
              && execution.lastRequest.sceneSnapshot->cameraCount == 1
              && execution.lastRequest.sceneSnapshot->lightCount == 1
              && execution.lastRequest.asset.id.empty(),
          "export receives the exact composed topology without an imported asset surrogate");
    const auto* firstComposedOwner = execution.lastRequest.sceneSnapshot.get();
    std::map<std::string, double> secondFrameParameters {
        { "visual12/objectX", 3.5 },
        { "visual12/objectRotationY", 45.0 },
        { "visual12/cameraZ", -2.0 },
        { "visual12/emissionGain", 4.0 }
    };
    check(prepareVisualImportedSceneLayerAtTime(
              composedPlans, 8, 640, 360, 1.0 / 24.0, 24.0,
              NativeImportedSceneRenderUse::Export, &execution, &composedPlanCache,
              composedLayer, composedOwners, error, &secondFrameParameters)
              == VisualImportedScenePreparation::rendered,
          "a second composed-scene frame evaluates through the cached production plan");
    check(execution.lastRequest.sceneSnapshot.get() == firstComposedOwner
              && execution.staticGpuUploads == 1
              && execution.lastRequest.runtimeInputs.objectTranslationOffset[0] == 3.5f
              && execution.lastRequest.runtimeInputs.objectRotationDegrees[1] == 45.0f
              && execution.lastRequest.runtimeInputs.cameraTranslationOffset[2] == -2.0f
              && execution.lastRequest.runtimeInputs.emissionGain == 4.0f,
          "two frame evaluations reuse static GPU ownership while transforms and lights update");
    auto disconnectedComposedPlan = composedPlan;
    disconnectedComposedPlan.edges.erase(disconnectedComposedPlan.edges.begin());
    std::vector<videowire::CompiledVisualLayerPlan> disconnectedComposedPlans {
        disconnectedComposedPlan
    };
    check(prepareVisualImportedSceneLayer(
              disconnectedComposedPlans, 8, 640, 360,
              NativeImportedSceneRenderUse::Preview, &execution,
              &composedPlanCache, composedLayer, composedOwners, error)
              == VisualImportedScenePreparation::rejected,
          "the cache revalidates and rejects changed V7 topology at the same revision");
    auto mistypedComposedPlan = composedPlan;
    const auto cameraPrimitiveEdge = std::find_if(
        mistypedComposedPlan.edges.begin(), mistypedComposedPlan.edges.end(),
        [](const auto& edge)
        { return edge.fromNodeId == 8 && edge.toNodeId == 2; });
    check(cameraPrimitiveEdge != mistypedComposedPlan.edges.end(),
          "the production-shaped fixture contains the primitive camera edge");
    if (cameraPrimitiveEdge != mistypedComposedPlan.edges.end())
        cameraPrimitiveEdge->toPort = 2;
    std::vector<videowire::CompiledVisualLayerPlan> mistypedComposedPlans {
        mistypedComposedPlan
    };
    check(prepareVisualImportedSceneLayer(
              mistypedComposedPlans, 8, 640, 360,
              NativeImportedSceneRenderUse::Preview, &execution,
              composedLayer, composedOwners, error)
              == VisualImportedScenePreparation::rejected,
          "helper execution rejects a reachable but mistyped Scene3D edge");
    auto detachedRenderStatePlan = composedPlan;
    detachedRenderStatePlan.edges.erase(
        std::remove_if(detachedRenderStatePlan.edges.begin(),
                       detachedRenderStatePlan.edges.end(), [](const auto& edge)
                       { return edge.fromNodeId == 8 && edge.toNodeId == 11; }),
        detachedRenderStatePlan.edges.end());
    std::vector<videowire::CompiledVisualLayerPlan> detachedRenderStatePlans {
        detachedRenderStatePlan
    };
    check(prepareVisualImportedSceneLayer(
              detachedRenderStatePlans, 8, 640, 360,
              NativeImportedSceneRenderUse::Preview, &execution,
              composedLayer, composedOwners, error)
              == VisualImportedScenePreparation::rejected,
          "helper execution requires explicit composed-scene camera topology");

    std::map<std::string, double> staticRuntimeParameters;
    videohelper::importedscene::seedImportedSceneRuntimeParameters(
        plans, 7, staticRuntimeParameters);
    check(staticRuntimeParameters["visual72/objectRotationY"] == 0.0
              && staticRuntimeParameters["visual72/objectScale"] == 1.0,
          "static imported scenes seed Render 3D transform targets without deformation");

    auto deformationPlan = importedScenePlan(false, true);
    std::vector<videowire::CompiledVisualLayerPlan> deformationPlans {
        deformationPlan
    };
    FakeLayer deformationLayer;
    std::vector<FakeReceipt> deformationOwners;
    std::map<std::string, double> runtimeParameters;
    videohelper::importedscene::seedImportedSceneRuntimeParameters(
        deformationPlans, 7, runtimeParameters);
    check(runtimeParameters["visual73/timelineSeconds"] == 1.0
              && runtimeParameters["visual73/speed"] == 1.5
              && runtimeParameters["visual73/trimStartSeconds"] == 0.25
              && runtimeParameters["visual73/trimEndSeconds"] == 0.75
              && runtimeParameters["visual73/weight"] == 0.75
              && runtimeParameters["visual72/objectX"] == 0.0
              && runtimeParameters["visual72/objectRotationY"] == 0.0
              && runtimeParameters["visual72/objectScale"] == 1.0
              && runtimeParameters["visual72/cameraZ"] == 0.0
              && runtimeParameters["visual72/emissionGain"] == 1.0,
          "the immutable deformation values seed stable per-node runtime targets");
    runtimeParameters["visual73/timelineSeconds"] = 4.0;
    runtimeParameters["visual73/speed"] = 2.0;
    runtimeParameters["visual73/trimStartSeconds"] = 0.5;
    runtimeParameters["visual73/trimEndSeconds"] = 1.5;
    runtimeParameters["visual73/weight"] = 0.25;
    runtimeParameters["visual72/objectX"] = 0.5;
    runtimeParameters["visual72/objectRotationY"] = 45.0;
    runtimeParameters["visual72/objectScale"] = 1.5;
    runtimeParameters["visual72/cameraZ"] = -0.25;
    runtimeParameters["visual72/emissionGain"] = 2.0;
    check(prepareVisualImportedSceneLayerAtTime(
              deformationPlans, 7, 640, 360, 2.5, 24.0,
              NativeImportedSceneRenderUse::Preview,
              &execution, deformationLayer, deformationOwners, error,
              &runtimeParameters)
              == VisualImportedScenePreparation::rendered,
          "the exact Source -> Deformation -> Render 3D plan reaches preview");
    check(execution.lastRequest.deformation
              && execution.lastRequest.deformation->deformationStableId == 73
              && execution.lastRequest.frame.frame == 60
              && execution.lastRequest.frame.rateNumerator == 24
              && execution.lastRequest.frame.rateDenominator == 1
              && execution.lastRequest.structuralRevision == 9
              && execution.lastRequest.deformation->playback.timelineSeconds == 4.0
              && execution.lastRequest.deformation->playback.speed == 2.0
              && execution.lastRequest.deformation->playback.trimStartSeconds == 0.5
              && execution.lastRequest.deformation->playback.trimEndSeconds == 1.5
              && execution.lastRequest.deformation->playback.weight == 0.25
              && execution.lastRequest.runtimeInputs.objectTranslationOffset[0] == 0.5f
              && execution.lastRequest.runtimeInputs.objectRotationDegrees[1] == 45.0f
              && execution.lastRequest.runtimeInputs.objectScale == 1.5f
              && execution.lastRequest.runtimeInputs.cameraTranslationOffset[2] == -0.25f
              && execution.lastRequest.runtimeInputs.emissionGain == 2.0f,
          "preview receives the immutable deformation payload and exact frame time");

    std::vector<videowire::CompiledVisualLayerPlan> materialDeformationPlans {
        importedScenePlan(true, true)
    };
    check(prepareVisualImportedSceneLayerAtTime(
              materialDeformationPlans, 7, 640, 360, 2.5, 24.0,
              NativeImportedSceneRenderUse::Preview,
              &execution, deformationLayer, deformationOwners, error)
              == VisualImportedScenePreparation::rendered,
          "the ordinary graph admits deformation and Surface Material together");
    check(execution.lastRequest.deformation
              && execution.lastRequest.material
              && execution.lastRequest.material->evaluationRevision == 33,
          "the shared request carries both immutable deformation and material payloads");

    auto tamperedDeformationPlan = deformationPlan;
    tamperedDeformationPlan.edges.pop_back();
    std::vector<videowire::CompiledVisualLayerPlan> tamperedDeformationPlans {
        tamperedDeformationPlan
    };
    check(prepareVisualImportedSceneLayerAtTime(
              tamperedDeformationPlans, 7, 640, 360, 2.5, 24.0,
              NativeImportedSceneRenderUse::Preview,
              &execution, deformationLayer, deformationOwners, error)
              == VisualImportedScenePreparation::rejected,
          "the deformation route rejects a missing exact graph edge");

    for (const auto hostile : { arbitgpu::NativeTexturePixelFormat::Rgba8Unorm,
                                arbitgpu::NativeTexturePixelFormat::Invalid })
    {
        execution.nextFrame = std::make_shared<FakeFrame>();
        execution.nextFrame->depthFormat = hostile;
        FakeLayer hostileLayer;
        std::vector<FakeReceipt> hostileOwners;
        check(prepareVisualImportedSceneLayer(
                  plans, 7, 640, 360, NativeImportedSceneRenderUse::Preview,
                  &execution, hostileLayer, hostileOwners, error)
                  == VisualImportedScenePreparation::rejected
                  && hostileOwners.empty(),
              "wrong native depth formats reject before ownership publication");
    }
    execution.nextFrame = std::make_shared<FakeFrame>();
    execution.nextFrame->descriptorWidth = 639;
    FakeLayer hostileDimensions;
    std::vector<FakeReceipt> hostileDimensionOwners;
    check(prepareVisualImportedSceneLayer(
              plans, 7, 640, 360, NativeImportedSceneRenderUse::Preview,
              &execution, hostileDimensions, hostileDimensionOwners, error)
              == VisualImportedScenePreparation::rejected,
          "mismatched native descriptor dimensions reject before composition");
    execution.nextFrame = std::make_shared<FakeFrame>();
    execution.nextFrame->descriptorSampleCount = 4;
    FakeLayer hostileSamples;
    std::vector<FakeReceipt> hostileSampleOwners;
    check(prepareVisualImportedSceneLayer(
              plans, 7, 640, 360, NativeImportedSceneRenderUse::Preview,
              &execution, hostileSamples, hostileSampleOwners, error)
              == VisualImportedScenePreparation::rejected,
          "multisampled native descriptors reject before composition");

    execution.nextFrame = std::make_shared<FakeFrame>();
    execution.nextFrame->api = "metal";
    execution.nextFrame->view = 0x12345678u;
    FakeLayer exportLayer;
    std::vector<FakeReceipt> exportOwners;
    check(prepareVisualImportedSceneLayer(
              plans, 7, 640, 360, NativeImportedSceneRenderUse::Export,
              &execution, exportLayer, exportOwners, error)
              == VisualImportedScenePreparation::rendered,
          "the same exact plan renders for export");
    check(execution.exportCalls == 3 && previewOwners.size() == 1
              && exportOwners.size() == 1 && exportOwners.front().owner == 2,
          "preview and export retain independent owners");
    check(exportLayer.texture == 0 && exportLayer.nativeTextureBackend == "metal"
              && exportLayer.nativeTextureView == 0x12345678u,
          "Metal preserves native texture view identity without narrowing");
    check(exportLayer.depthTexture == 0
              && exportLayer.nativeDepthTextureBackend == "metal"
              && exportLayer.nativeDepthTextureView == 66
              && exportLayer.depthWidth == 640 && exportLayer.depthHeight == 360,
          "Metal preserves the native R32F depth view without readback or narrowing");

    execution.nextFrame = std::make_shared<FakeFrame>();
    FakeLayer materialLayer;
    std::vector<FakeReceipt> materialOwners;
    std::vector<videowire::CompiledVisualLayerPlan> materialPlans {
        importedScenePlan(true)
    };
    check(prepareVisualImportedSceneLayer(
              materialPlans, 7, 640, 360,
              NativeImportedSceneRenderUse::Preview,
              &execution, materialLayer, materialOwners, error)
              == VisualImportedScenePreparation::rendered
              && execution.lastRequest.material
              && execution.lastRequest.material->sceneSnapshot == nullptr
              && visualimportedscenerender::sameMaterialRequest(
                  *exactMaterialRequest.material,
                  *execution.lastRequest.material),
          "the decoded material value reaches payload execution without a scene owner");

    auto unequalCrossedBlazedRequest = renderRequest(false, false, true);
    auto& unequalCrossedBlazed
        = unequalCrossedBlazedRequest.diffractionMaterial->material;
    unequalCrossedBlazed = diffractionmaterial::makeBlazedGratingPreset();
    unequalCrossedBlazed.geometry.lattice
        = diffractionmaterial::GratingLattice::CrossedTwoDimensional;
    unequalCrossedBlazed.geometry.secondaryDirectionUv = { 0.0f, 1.0f };
    unequalCrossedBlazed.geometry.secondaryGrooveSpacingNanometres
        = unequalCrossedBlazed.geometry.grooveSpacingNanometres + 1.0f;
    check(visualimportedscenerender::encode(unequalCrossedBlazedRequest).empty(),
          "the imported-scene wire validator rejects unequal crossed blazed periods");

    auto currentDiffractionRequest = renderRequest(false, false, true);
    currentDiffractionRequest.diffractionMaterial = spatialDiffractionMaterialRequest();
    const auto canonicalLighting = diffractionmaterial::makeReferenceLighting();
    std::string currentFoilAdmissionError;
    const auto admittedCurrentFoil = diffractivefoil::admit(
        *currentDiffractionRequest.diffractionMaterial->spatialFoil,
        currentFoilAdmissionError);
    std::string currentLightingAdmissionError;
    const auto admittedCurrentLighting = diffractionmaterial::AdmittedLightingPlan::admit(
        currentDiffractionRequest.diffractionMaterial->lighting,
        currentLightingAdmissionError);
    std::string currentBindingAdmissionError;
    const auto admittedCurrentBinding = diffractionmaterialbinding::admit(
        *currentDiffractionRequest.diffractionMaterial,
        currentBindingAdmissionError);
    check(admittedCurrentFoil.has_value(),
          "production admission accepts the current V7 spatial foil before encoding: "
              + currentFoilAdmissionError);
    check(admittedCurrentLighting.has_value()
              && admittedCurrentLighting->hasIndirectBounce()
              && diffractionmaterial::sameLightingDescription(
                  admittedCurrentLighting->description(), canonicalLighting),
          "production admission accepts the canonical three-path V7 diffraction lighting before encoding: "
              + currentLightingAdmissionError);
    check(admittedCurrentBinding.has_value()
              && admittedCurrentBinding->spatialFoil().has_value()
              && diffractionmaterial::sameLightingDescription(
                  admittedCurrentBinding->lighting().description(), canonicalLighting),
          "binding admission owns the current V7 spatial foil before encoding: "
              + currentBindingAdmissionError);
    auto noncanonicalLightingRequest = currentDiffractionRequest;
    noncanonicalLightingRequest.diffractionMaterial->lighting.paths[0]
        .incident.radiance[0] = 0.64f;
    std::string noncanonicalLightingError;
    check(visualimportedscenerender::encode(noncanonicalLightingRequest).empty()
              && !diffractionmaterialbinding::admit(
                  *noncanonicalLightingRequest.diffractionMaterial,
                  noncanonicalLightingError)
              && noncanonicalLightingError.find("version-defined canonical plan")
                  != std::string::npos,
          "V7 rejects valid but noncanonical caller lighting instead of hiding it behind version-defined restoration");
    auto malformedTrailingLightingRequest = currentDiffractionRequest;
    malformedTrailingLightingRequest.diffractionMaterial->lighting.paths.back().kind
        = diffractionmaterial::LightingPathKind::Direct;
    std::string malformedTrailingLightingError;
    check(visualimportedscenerender::encode(malformedTrailingLightingRequest).empty()
              && !diffractionmaterialbinding::admit(
                  *malformedTrailingLightingRequest.diffractionMaterial,
                  malformedTrailingLightingError)
              && malformedTrailingLightingError.find("noncanonical trailing paths")
                  != std::string::npos,
          "V7 rejects noncanonical unused lighting paths before wire encoding or binding admission");
    const auto currentDiffraction
        = visualimportedscenerender::encode(currentDiffractionRequest);
    visualimportedscenerender::Request decodedCurrentDiffraction;
    const bool decodedCurrent = currentDiffraction.rfind(
              std::string(visualimportedscenerender::kCompiledOperationHeaderV7) + "\n",
              0) == 0
              && visualimportedscenerender::decode(
                  currentDiffraction, decodedCurrentDiffraction);
    check(decodedCurrent, "the current diffraction producer decodes canonical V7 bytes");
    if (decodedCurrent)
    {
        const auto& expected = *currentDiffractionRequest.diffractionMaterial;
        const auto& actual = *decodedCurrentDiffraction.diffractionMaterial;
        check(currentDiffractionRequest.sourceStableId
                  == decodedCurrentDiffraction.sourceStableId
                  && currentDiffractionRequest.renderStableId
                      == decodedCurrentDiffraction.renderStableId
                  && currentDiffractionRequest.asset.id
                      == decodedCurrentDiffraction.asset.id
                  && currentDiffractionRequest.asset.version
                      == decodedCurrentDiffraction.asset.version
                  && currentDiffractionRequest.asset.contentSha256
                      == decodedCurrentDiffraction.asset.contentSha256
                  && currentDiffractionRequest.asset.sourceMediaType
                      == decodedCurrentDiffraction.asset.sourceMediaType
                  && currentDiffractionRequest.asset.sourceByteSize
                      == decodedCurrentDiffraction.asset.sourceByteSize
                  && currentDiffractionRequest.sceneIndex
                      == decodedCurrentDiffraction.sceneIndex
                  && expected.version == actual.version
                  && expected.scene == actual.scene
                  && expected.sceneRevision == actual.sceneRevision
                  && expected.structuralRevision == actual.structuralRevision
                  && expected.evaluationRevision == actual.evaluationRevision
                  && expected.materialRevision == actual.materialRevision
                  && expected.object == actual.object
                  && expected.structuralDigest == actual.structuralDigest
                  && visualimportedscenerender::detail::sameDiffractionDescription(
                      expected.material, actual.material)
                  && diffractionmaterial::sameLightingDescription(
                      expected.lighting, actual.lighting)
                  && diffractionmaterial::sameLightingDescription(
                      actual.lighting, canonicalLighting),
              "V7 preserves imported-scene identities, every diffraction material field, and every bit of the implicit canonical lighting plan");

        const auto& expectedFoil = *expected.spatialFoil;
        const auto& actualFoil = *actual.spatialFoil;
        bool sameFoilField = expectedFoil.version == actualFoil.version
            && expectedFoil.workBudget.maximumEvaluations
                == actualFoil.workBudget.maximumEvaluations
            && visualimportedscenerender::detail::sameDiffractionDescription(
                expectedFoil.physicalBsdf, actualFoil.physicalBsdf);
        for (std::size_t index = 0; index < expectedFoil.grooveField.size(); ++index)
        {
            const auto& expectedSample = expectedFoil.grooveField[index];
            const auto& actualSample = actualFoil.grooveField[index];
            sameFoilField = sameFoilField
                && visualimportedscenerender::detail::sameFloatArray(
                    expectedSample.reciprocalDirectionUv,
                    actualSample.reciprocalDirectionUv)
                && visualimportedscenerender::detail::floatBits(
                    expectedSample.grooveSpacingNanometres)
                    == visualimportedscenerender::detail::floatBits(
                        actualSample.grooveSpacingNanometres)
                && visualimportedscenerender::detail::floatBits(
                    expectedSample.diffractionCoverage)
                    == visualimportedscenerender::detail::floatBits(
                        actualSample.diffractionCoverage);
        }
        check(sameFoilField,
              "V7 preserves the foil version, immutable ceiling, physical BSDF, and all four ordered field samples");

        constexpr std::uint32_t graphAuthoredWidth = 4096;
        constexpr std::uint32_t graphAuthoredHeight = 4096;
        constexpr std::uint32_t graphAuthoredPathCount = 4;
        diffractivefoil::EvaluationSchedule schedule;
        check(diffractivefoil::makeEvaluationSchedule(
                  actualFoil.workBudget.maximumEvaluations,
                  graphAuthoredWidth, graphAuthoredHeight,
                  graphAuthoredPathCount, schedule)
                  && schedule.width == graphAuthoredWidth
                  && schedule.height == graphAuthoredHeight
                  && schedule.pathCount == graphAuthoredPathCount
                  && schedule.maximumEvaluations
                      == diffractivefoil::kMaximumEvaluations
                  && schedule.requiredEvaluations
                      == diffractivefoil::kMaximumEvaluations,
              "the decoded graph-authored ceiling admits the exact width, height, and path boundary schedule");
        check(visualimportedscenerender::sameRequest(
                  currentDiffractionRequest, decodedCurrentDiffraction)
                  && visualimportedscenerender::encode(decodedCurrentDiffraction)
                      == currentDiffraction,
              "the full V7 diffraction-plus-foil value round-trips to identical canonical bytes");
    }


    auto legacyV6Request = renderRequest(false, false, true);
    legacyV6Request.diffractionMaterial->version
        = visualimportedscenerender::detail::kLegacyV6BindingWireVersion;
    legacyV6Request.diffractionMaterial->material.version
        = visualimportedscenerender::detail::kLegacyV6MaterialWireVersion;
    legacyV6Request.diffractionMaterial->structuralDigest = std::string(64, 'd');
    auto noncanonicalLegacyLightingRequest = legacyV6Request;
    noncanonicalLegacyLightingRequest.diffractionMaterial->lighting.paths[1]
        .incident.radiance[7] = 0.15f;
    check(visualimportedscenerender::encodeLegacyV6(
              noncanonicalLegacyLightingRequest).empty(),
          "V6 rejects noncanonical caller lighting instead of silently restoring over it");
    const auto legacyV6 = visualimportedscenerender::encodeLegacyV6(legacyV6Request);
    visualimportedscenerender::Request migratedV6;
    check(!legacyV6.empty()
              && !visualimportedscenerender::decode(legacyV6, migratedV6)
              && visualimportedscenerender::decode(
                  legacyV6, migratedV6,
                  diffractionmaterialbinding::migrateLegacyV6)
              && migratedV6.diffractionMaterial
              && migratedV6.diffractionMaterial->version
                  == diffractionmaterialbinding::kWireVersion
              && migratedV6.diffractionMaterial->material.version
                  == diffractionmaterial::kWireVersion
              && migratedV6.diffractionMaterial->material.grooveField.mode
                  == diffractionmaterial::GrooveFieldMode::Constant
              && migratedV6.diffractionMaterial->material.grooveField.originUv
                  == std::array<float, 2> {}
              && migratedV6.diffractionMaterial->structuralDigest
                  == diffractionMaterialRequest().structuralDigest
              && diffractionmaterial::sameLightingDescription(
                  migratedV6.diffractionMaterial->lighting, canonicalLighting),
          "V6 fieldless diffraction decodes only through owning admission and migrates to canonical V7 identity");
    const auto legacyFixture = [](std::uint32_t sourceId)
    {
        auto request = renderRequest(false, false, true);
        request.sourceStableId = sourceId;
        request.diffractionMaterial->version
            = visualimportedscenerender::detail::kLegacyV6BindingWireVersion;
        request.diffractionMaterial->material.version
            = visualimportedscenerender::detail::kLegacyV6MaterialWireVersion;
        request.diffractionMaterial->structuralDigest = std::string(64, 'd');
        return visualimportedscenerender::encodeLegacyV6(request);
    };
    const auto currentFixture = [](std::uint32_t sourceId)
    {
        auto request = renderRequest(false, false, true);
        request.sourceStableId = sourceId;
        return visualimportedscenerender::encode(request);
    };
    const auto compositionFixture = [&boundLight](std::uint32_t sourceId)
    {
        auto request = renderRequest(true);
        request.sourceStableId = sourceId;
        request.light = boundLight;
        return visualimportedscenerender::encode(request);
    };

    // V6 and V7 carry three flags: deformation, mandatory diffraction, camera.
    // V8 carries six. A canonical V6 body always presents 0 1 0 1 to V8,
    // which violates V8's material/diffraction exclusion. A canonical V7 body
    // presents binding version 3 as V8's fourth flag. Those two directions
    // therefore cannot pass V8 flag admission. The other four collisions enter
    // the destination diffraction grammar before its version-specific fields
    // reject the shifted source tokens.
    std::array<DirectionalRelabelFixture, 6> crossVersionFixtures {{
        { "V6-to-V7", visualimportedscenerender::kCompiledOperationHeaderV6,
          visualimportedscenerender::kCompiledOperationHeaderV7, legacyFixture(601), {},
          CrossVersionAdmissionStage::diffractionSyntax },
        { "V7-to-V6", visualimportedscenerender::kCompiledOperationHeaderV7,
          visualimportedscenerender::kCompiledOperationHeaderV6, currentFixture(701), {},
          CrossVersionAdmissionStage::diffractionSyntax },
        { "V6-to-V8", visualimportedscenerender::kCompiledOperationHeaderV6,
          visualimportedscenerender::kCompiledOperationHeaderV8, legacyFixture(608), {},
          CrossVersionAdmissionStage::flagSchema },
        { "V8-to-V6", visualimportedscenerender::kCompiledOperationHeaderV8,
          visualimportedscenerender::kCompiledOperationHeaderV6, compositionFixture(806), {},
          CrossVersionAdmissionStage::diffractionSyntax },
        { "V7-to-V8", visualimportedscenerender::kCompiledOperationHeaderV7,
          visualimportedscenerender::kCompiledOperationHeaderV8, currentFixture(708), {},
          CrossVersionAdmissionStage::flagSchema },
        { "V8-to-V7", visualimportedscenerender::kCompiledOperationHeaderV8,
          visualimportedscenerender::kCompiledOperationHeaderV7, compositionFixture(807), {},
          CrossVersionAdmissionStage::diffractionSyntax }
    }};

    std::set<std::string> sourceBodies;
    for (auto& fixture : crossVersionFixtures)
    {
        fixture.relabeledBytes = relabelHeaderOnly(
            fixture.sourceBytes, fixture.destinationHeader);
        visualimportedscenerender::Request canonicalSource;
        const bool sourceDecoded = fixture.sourceHeader
                == visualimportedscenerender::kCompiledOperationHeaderV6
            ? visualimportedscenerender::decode(
                fixture.sourceBytes, canonicalSource,
                diffractionmaterialbinding::migrateLegacyV6)
            : visualimportedscenerender::decode(fixture.sourceBytes, canonicalSource);
        const auto sourceReencoded = fixture.sourceHeader
                == visualimportedscenerender::kCompiledOperationHeaderV6
            ? visualimportedscenerender::encodeLegacyV6([&]
              {
                  auto source = canonicalSource;
                  source.diffractionMaterial->version
                      = visualimportedscenerender::detail::kLegacyV6BindingWireVersion;
                  source.diffractionMaterial->material.version
                      = visualimportedscenerender::detail::kLegacyV6MaterialWireVersion;
                  source.diffractionMaterial->material.grooveField = {};
                  source.diffractionMaterial->structuralDigest = std::string(64, 'd');
                  return source;
              }())
            : visualimportedscenerender::encode(canonicalSource);
        check(sourceDecoded && !fixture.sourceBytes.empty()
                  && sourceReencoded == fixture.sourceBytes,
              std::string(fixture.identity) + " starts from source-canonical bytes");
        check(relabelHeaderOnly(fixture.relabeledBytes, fixture.sourceHeader)
                  == fixture.sourceBytes,
              std::string(fixture.identity) + " changes only the version header");
        check(probeCrossVersionAdmission(
                  fixture.relabeledBytes, fixture.destinationHeader)
                  == fixture.expectedStage,
              std::string(fixture.identity)
                  + " reaches its exact wrong-schema admission boundary");
        const bool rejected = fixture.destinationHeader
                == visualimportedscenerender::kCompiledOperationHeaderV6
            ? !visualimportedscenerender::decode(
                fixture.relabeledBytes, migratedV6,
                diffractionmaterialbinding::migrateLegacyV6)
            : !visualimportedscenerender::decode(fixture.relabeledBytes, migratedV6);
        check(rejected,
              std::string(fixture.identity)
                  + " is rejected by destination schema authority");
        sourceBodies.insert(fixture.sourceBytes.substr(
            fixture.sourceBytes.find('\n') + 1));
    }
    check(sourceBodies.size() == crossVersionFixtures.size(),
          "all six directional fixtures have byte-distinct source-shaped bodies");
    const auto malformedLegacyV6 = replaceLine(
        legacyV6, 9,
        "2 1 3 9 33 9 23 \"" + std::string(64, 'd') + "\"");
    check(!visualimportedscenerender::decode(
              malformedLegacyV6, migratedV6,
              diffractionmaterialbinding::migrateLegacyV6),
          "the frozen V6 decoder rejects a non-V1 legacy binding schema");
    auto legacyV6Plan = importedScenePlan(false, false, true);
    legacyV6Plan.operations.back().payloadXml = legacyV6;
    std::vector<videowire::CompiledVisualLayerPlan> legacyV6Plans {
        std::move(legacyV6Plan)
    };
    check(prepareVisualImportedSceneLayer(
              legacyV6Plans, 7, 640, 360,
              NativeImportedSceneRenderUse::Preview,
              &execution, materialLayer, materialOwners, error)
              == VisualImportedScenePreparation::rendered
              && execution.lastRequest.diffractionMaterial
              && execution.lastRequest.diffractionMaterial->structuralDigest
                  == diffractionMaterialRequest().structuralDigest
              && diffractionmaterial::sameLightingDescription(
                  execution.lastRequest.diffractionMaterial->lighting,
                  canonicalLighting),
          "the imported-scene owner admits the migrated V6 material before execution: "
              + error);

    std::vector<videowire::CompiledVisualLayerPlan> diffractionPlans {
        importedScenePlan(false, false, true)
    };
    const auto diffractionPreparation = prepareVisualImportedSceneLayer(
              diffractionPlans, 7, 640, 360,
              NativeImportedSceneRenderUse::Preview,
              &execution, materialLayer, materialOwners, error);
    const auto expectedDiffractionDigest = diffractionMaterialRequest().structuralDigest;
    const auto actualDiffractionDigest = execution.lastRequest.diffractionMaterial
        ? execution.lastRequest.diffractionMaterial->structuralDigest : std::string {};
    check(diffractionPreparation == VisualImportedScenePreparation::rendered
              && execution.lastRequest.diffractionMaterial
              && actualDiffractionDigest == expectedDiffractionDigest
              && diffractionmaterial::sameLightingDescription(
                  execution.lastRequest.diffractionMaterial->lighting,
                  canonicalLighting),
          "a valid diffraction binding reaches the shared native preview execution owner: preparation="
              + std::to_string(static_cast<int>(diffractionPreparation))
              + ", expectedDigest=" + expectedDiffractionDigest
              + ", actualDigest=" + actualDiffractionDigest
              + ", error=" + error);
    const int previewCallsAfterDiffraction = execution.previewCalls;

    auto tamperedDiffractionPlan = importedScenePlan(false, false, true);
    auto tamperedDiffractionRequest = renderRequest(false, false, true);
    tamperedDiffractionRequest.diffractionMaterial->structuralDigest
        = std::string(64, 'c');
    tamperedDiffractionPlan.operations.back().payloadXml
        = visualimportedscenerender::encode(tamperedDiffractionRequest);
    std::vector<videowire::CompiledVisualLayerPlan> tamperedDiffractionPlans {
        std::move(tamperedDiffractionPlan)
    };
    check(prepareVisualImportedSceneLayer(
              tamperedDiffractionPlans, 7, 640, 360,
              NativeImportedSceneRenderUse::Preview,
              &execution, materialLayer, materialOwners, error)
              == VisualImportedScenePreparation::rejected
              && error.find("structural digest does not match") != std::string::npos
              && execution.previewCalls == previewCallsAfterDiffraction,
          "helper admission recomputes diffraction identity before execution");

    auto detachedDiffractionPlan = importedScenePlan(false, false, true);
    detachedDiffractionPlan.edges.pop_back();
    std::vector<videowire::CompiledVisualLayerPlan> detachedDiffractionPlans {
        std::move(detachedDiffractionPlan)
    };
    check(prepareVisualImportedSceneLayer(
              detachedDiffractionPlans, 7, 640, 360,
              NativeImportedSceneRenderUse::Preview,
              &execution, materialLayer, materialOwners, error)
              == VisualImportedScenePreparation::rejected
              && error.find("exact source-to-Render 3D typed schedule")
                  != std::string::npos,
          "a disconnected diffraction material terminal fails topology admission");

    auto detachedMaterialPlan = importedScenePlan(true);
    detachedMaterialPlan.edges.erase(detachedMaterialPlan.edges.begin() + 1);
    std::vector<videowire::CompiledVisualLayerPlan> detachedMaterialPlans {
        std::move(detachedMaterialPlan)
    };
    FakeLayer detachedMaterialLayer;
    std::vector<FakeReceipt> detachedMaterialOwners;
    check(prepareVisualImportedSceneLayer(
              detachedMaterialPlans, 7, 640, 360,
              NativeImportedSceneRenderUse::Preview,
              &execution, detachedMaterialLayer, detachedMaterialOwners, error)
              == VisualImportedScenePreparation::rejected,
          "a disconnected ordinary material terminal input fails closed");

    auto terminalOnlyMaterialPlan = importedScenePlan(true);
    const auto materialValueNode = [](int nodeId)
        { return nodeId >= 72 && nodeId <= 81; };
    terminalOnlyMaterialPlan.nodeKinds.erase(
        terminalOnlyMaterialPlan.nodeKinds.begin() + 1,
        terminalOnlyMaterialPlan.nodeKinds.begin() + 11);
    terminalOnlyMaterialPlan.nodeIds.erase(
        terminalOnlyMaterialPlan.nodeIds.begin() + 1,
        terminalOnlyMaterialPlan.nodeIds.begin() + 11);
    terminalOnlyMaterialPlan.operations.erase(
        terminalOnlyMaterialPlan.operations.begin() + 1,
        terminalOnlyMaterialPlan.operations.begin() + 11);
    terminalOnlyMaterialPlan.ports.erase(
        std::remove_if(terminalOnlyMaterialPlan.ports.begin(),
                       terminalOnlyMaterialPlan.ports.end(),
                       [&](const auto& port) { return materialValueNode(port.nodeId); }),
        terminalOnlyMaterialPlan.ports.end());
    terminalOnlyMaterialPlan.edges.erase(
        std::remove_if(terminalOnlyMaterialPlan.edges.begin(),
                       terminalOnlyMaterialPlan.edges.end(),
                       [&](const auto& edge) { return materialValueNode(edge.fromNodeId); }),
        terminalOnlyMaterialPlan.edges.end());
    std::vector<videowire::CompiledVisualLayerPlan> terminalOnlyMaterialPlans {
        std::move(terminalOnlyMaterialPlan)
    };
    check(prepareVisualImportedSceneLayer(
              terminalOnlyMaterialPlans, 7, 640, 360,
              NativeImportedSceneRenderUse::Preview,
              &execution, detachedMaterialLayer, detachedMaterialOwners, error)
              == VisualImportedScenePreparation::rejected,
          "a terminal-only schedule cannot claim a transported material program");

    auto mismatchedMaterialPlan = importedScenePlan(true);
    mismatchedMaterialPlan.nodeKinds[1] = "visual.surface.input.position";
    mismatchedMaterialPlan.operations[1].kind = "visual.surface.input.position";
    std::vector<videowire::CompiledVisualLayerPlan> mismatchedMaterialPlans {
        std::move(mismatchedMaterialPlan)
    };
    check(prepareVisualImportedSceneLayer(
              mismatchedMaterialPlans, 7, 640, 360,
              NativeImportedSceneRenderUse::Preview,
              &execution, detachedMaterialLayer, detachedMaterialOwners, error)
              == VisualImportedScenePreparation::rejected,
          "a transported material program cannot disagree with the ordinary DAG");

    FakeLayer rejectedLayer;
    std::vector<FakeReceipt> rejectedOwners;
    check(prepareVisualImportedSceneLayer(
              plans, 7, 640, 360, NativeImportedSceneRenderUse::Preview,
              static_cast<FakeExecution*>(nullptr), rejectedLayer, rejectedOwners, error)
              == VisualImportedScenePreparation::rejected,
          "missing processor payload execution fails closed");

    execution.succeed = false;
    check(prepareVisualImportedSceneLayer(
              plans, 7, 640, 360, NativeImportedSceneRenderUse::Preview,
              &execution, rejectedLayer, rejectedOwners, error)
              == VisualImportedScenePreparation::rejected
              && error == "preview rejected",
          "missing exact resident payload execution fails closed");
    execution.succeed = true;

    check(prepareVisualImportedSceneLayer(
              plans, 7, 0, 360, NativeImportedSceneRenderUse::Preview,
              &execution, rejectedLayer, rejectedOwners, error)
              == VisualImportedScenePreparation::rejected,
          "out-of-bounds dimensions fail closed before execution");

    execution.nextFrame = std::make_shared<FakeFrame>();
    execution.nextFrame->api = "vulkan";
    check(prepareVisualImportedSceneLayer(
              plans, 7, 640, 360, NativeImportedSceneRenderUse::Preview,
              &execution, rejectedLayer, rejectedOwners, error)
              == VisualImportedScenePreparation::rejected,
          "an unsupported native frame backend fails closed");

    execution.nextFrame = std::make_shared<FakeFrame>();
    execution.nextFrame->view = 0;
    check(prepareVisualImportedSceneLayer(
              plans, 7, 640, 360, NativeImportedSceneRenderUse::Preview,
              &execution, rejectedLayer, rejectedOwners, error)
              == VisualImportedScenePreparation::rejected,
          "a missing native texture view fails closed");

    if (std::numeric_limits<std::uintptr_t>::max()
        > std::numeric_limits<unsigned>::max())
    {
        execution.nextFrame = std::make_shared<FakeFrame>();
        execution.nextFrame->view = static_cast<std::uintptr_t>(
            std::numeric_limits<unsigned>::max()) + 1u;
        check(prepareVisualImportedSceneLayer(
                  plans, 7, 640, 360, NativeImportedSceneRenderUse::Preview,
                  &execution, rejectedLayer, rejectedOwners, error)
                  == VisualImportedScenePreparation::rejected,
              "an oversized OpenGL texture view is never narrowed");
    }

    auto malformed = importedScenePlan();
    malformed.operations.back().payloadXml += "x";
    videowire::VisualLayerExecution compiled;
    check(!videowire::compileVisualLayerExecution(malformed, compiled, error),
          "a noncanonical imported scene operation fails closed");

    auto sourceOnly = importedScenePlan();
    sourceOnly.nodeKinds.pop_back();
    sourceOnly.nodeIds.pop_back();
    sourceOnly.operations.pop_back();
    sourceOnly.ports.resize(5);
    sourceOnly.edges.clear();
    check(!videowire::compileVisualLayerExecution(sourceOnly, compiled, error),
          "source-only imported scene execution fails closed");

    auto deformationOnly = sourceOnly;
    deformationOnly.nodeKinds.front() =
        std::string(visualanimationoperation::kDeformationNodeKind);
    deformationOnly.operations.front().kind =
        std::string(visualanimationoperation::kDeformationNodeKind);
    deformationOnly.operations.front().backendCapability =
        std::string(visualanimationoperation::kDeformationBackendCapability);
    check(!videowire::compileVisualLayerExecution(deformationOnly, compiled, error),
          "deformation-only imported scene execution fails closed");

    auto wrongEdge = importedScenePlan();
    wrongEdge.edges.front().fromPort = 3;
    check(!videowire::compileVisualLayerExecution(wrongEdge, compiled, error),
          "a non-Scene source edge fails closed");

    if (failures != 0)
    {
        std::cerr << failures << " imported scene visual plan checks failed\n";
        return 1;
    }
    std::cout << "imported scene visual plan execution: all checks passed\n";
    return 0;
}
