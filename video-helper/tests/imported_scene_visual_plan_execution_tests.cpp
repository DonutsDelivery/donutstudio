#include "imported_scene_visual_plan_execution.h"
#include "support/vertex_modifier_fixture.h"
#include "support/render_pass_program_cases.h"
#include "support/raw_pass_export_checks.h"
#include "support/material_field_fixture.h"
#include "support/imported_starter_topology.h"
#include "../../shared/DiffractionMaterialPresets.h"
#include "../../shared/DiffractionProductPlans.h"
#include "geometry_scene_composition_fixture.h"
#include "support/generated_surface_scene.h"
#include "reactive_frame_fixture.h"
#include "../src/renderer.h"
#include "reactive_surface_fixture.h"

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
    std::uintptr_t contextIdentity = 77;
    bool rawPasses = false;
    bool linearColor = false;

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
                 linearColor ? arbitgpu::NativeTexturePixelFormat::Rgba16Float
                 : api == "metal" ? arbitgpu::NativeTexturePixelFormat::Bgra8Unorm
                                  : arbitgpu::NativeTexturePixelFormat::Rgba8Unorm,
                 image, view, descriptorWidth, frameHeight, descriptorSampleCount, true, contextIdentity, 1,
                 linearColor ? colortransform::ColorSpace::LinearSRGB : colortransform::ColorSpace::Unspecified,
                 linearColor ? colortransform::TransferFunction::Linear : colortransform::TransferFunction::Unspecified };
    }
    arbitgpu::NativeTextureViewDescriptor depthTextureDescriptor() const noexcept override
    {
        return { api, arbitgpu::NativeTextureViewKind::Texture2D,
                 depthFormat, depthImage, depthView, descriptorWidth, frameHeight,
                 descriptorSampleCount, true, 77, 1 };
    }
    arbitgpu::NativeTextureViewDescriptor passTextureDescriptor(renderpassoutput::Output output) const noexcept override
    {
        if (!rawPasses || output == renderpassoutput::Output::Color || output == renderpassoutput::Output::Depth)
            return NativeFixtureSceneFrame::passTextureDescriptor(output);
        const auto id = static_cast<unsigned>(output);
        const auto format = output == renderpassoutput::Output::Mask ? arbitgpu::NativeTexturePixelFormat::R8Unorm
            : output == renderpassoutput::Output::Motion ? arbitgpu::NativeTexturePixelFormat::Rg16Float
            : output == renderpassoutput::Output::MaterialId || output == renderpassoutput::Output::ObjectId
                ? arbitgpu::NativeTexturePixelFormat::R32Uint : arbitgpu::NativeTexturePixelFormat::Rgba16Float;
        return { api, arbitgpu::NativeTextureViewKind::Texture2D, format, 100 + id, 200 + id,
            descriptorWidth, frameHeight, descriptorSampleCount, true, 77, 1 };
    }
};

struct FakeReceipt
{
    bool complete = true;
    std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> frame;
    int owner = 0;
    std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> canonicalBlockCFrame;
    std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> materialFrameTexture;
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
        receipt = { true, nextFrame, 1, request.runtimeInputs.canonicalBlockCFrame,
                    request.runtimeInputs.materialFrameTexture };
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
        receipt = { true, nextFrame, 2, request.runtimeInputs.canonicalBlockCFrame,
                    request.runtimeInputs.materialFrameTexture };
        return true;
    }
};

struct FakeLayer
{
    std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> rawExportFrame;
    std::uint32_t rawExportMask = 0, rawExportRenderId = 0;
    int rawExportClipId = -1;
    renderpassoutput::Output rawExportImageOutput = renderpassoutput::Output::Color;
    struct Clock { double time = 0.0; int frame = 0; } shaderClock;
    bool audioPresent = false;
    struct AudioFeatures { std::vector<float> bands; } audioFeatures;
    std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> canonicalBlockCFrame;
    int shaderTransitionFromClipId = 0;
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
void testAuthoredImportedStarterTopologies()
{
    using videohelper::tests::appendImportedStarterOutput;
    using videohelper::tests::useRegisteredImportedStarterPorts;
    using videohelper::importedscene::NativeImportedSceneRenderUse;
    using videohelper::importedscene::VisualImportedScenePreparation;
    using videohelper::importedscene::prepareVisualImportedSceneLayer;
    std::string error;
    videowire::VisualLayerExecution compiled;
    const auto operation = [](auto& plan, int node) -> auto& {
        return *std::find_if(plan.operations.begin(), plan.operations.end(),
            [&](const auto& item) { return item.nodeId == node; });
    };
    const auto addBefore = [](auto& plan, int before, int node, const char* kind,
                              const char* backend, const std::string& payload = "") {
        const auto found = std::find(plan.nodeIds.begin(), plan.nodeIds.end(), before);
        const auto index = found - plan.nodeIds.begin();
        plan.nodeIds.insert(found, node);
        plan.nodeKinds.insert(plan.nodeKinds.begin() + index, kind);
        plan.operations.insert(plan.operations.begin() + index, {node, kind, backend, payload});
    };
    const auto accepted = [&](const auto& plan) {
        error.clear();
        return videowire::validateCompiledVisualLayerPlans({}, {plan}, true, error)
            && videowire::compileVisualLayerExecution(plan, compiled, error);
    };
    const auto reject = [&](const auto& plan, const char* reason) {
        error.clear();
        check(!videowire::compileVisualLayerExecution(plan, compiled, error), reason);
    };

    // BuiltInVisualModules::depthReactiveFogSceneStarter keeps the complete
    // animated material graph and binds both fog inputs to its one Render 3D.
    auto fog = importedScenePlan(true, true);
    useRegisteredImportedStarterPorts(fog, 71);
    addBefore(fog, -1, 90, "visual.depth.fog", "native-gpu",
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<NodeParams near=\"0\" far=\"1\" density=\"1.35\" "
        "red=\"0.72\" green=\"0.82\" blue=\"0.95\" alpha=\"1\"/>");
    fog.ports.insert(fog.ports.end(), {
        {90, 0, 1, "in", "frame", "image", "rgba8", "sRGB"},
        {90, 1, 1, "in", "frame", "depth", "r32f", "unspecified"},
        {90, 2, 1, "out", "frame", "image", "rgba8", "sRGB"}
    });
    fog.edges.insert(fog.edges.end(), {{71, 1, 90, 0}, {71, 4, 90, 1}});
    check(accepted(fog) && compiled.depthFog && compiled.depthEffect == 1,
        "authored fog admits the module image before its outer output is connected: " + error);
    appendImportedStarterOutput(fog, 90, 2);
    check(accepted(fog) && compiled.importedSceneRender && compiled.importedSceneRender->deformation
        && compiled.importedSceneRender->material && compiled.depthFog
        && std::abs(compiled.fogDensity - 1.35f) < 0.0001f,
        "authored Depth-Reactive Fog retains animation, material and outer Video Out: " + error);
    for (const auto use : {NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export})
    {
        FakeExecution native;
        videorender::LayerDesc layer;
        std::vector<FakeReceipt> owners;
        const auto prepared = prepareVisualImportedSceneLayer({fog}, 7, 640, 360, use,
            &native, layer, owners, error);
        check(prepared == VisualImportedScenePreparation::rendered
            && videowire::executeVisualLayerPlan({fog}, 7, layer, error,
                use == NativeImportedSceneRenderUse::Preview ? videohelper::geometry::PlanUse::preview
                    : videohelper::geometry::PlanUse::exportRender)
            && layer.depthFog && layer.depthEffect == 1 && layer.depthTexture == 66
            && layer.nativeDepthTextureView == 66 && !owners.empty()
            && std::abs(layer.fogBlue - 0.95f) < 0.0001f,
            "preview and export compose fog using their retained imported depth: " + error);
    }
    auto invalid = fog;
    invalid.edges.erase(invalid.edges.end() - 2);
    reject(invalid, "fog rejects an unbound depth input");
    invalid = fog;
    invalid.edges[invalid.edges.size() - 2].fromNodeId = 70;
    reject(invalid, "fog rejects depth from a different source owner");
    invalid = fog;
    invalid.edges.push_back({71, 1, 6, 0});
    reject(invalid, "fog rejects an extra direct Render-to-output edge");
    invalid = fog;
    invalid.ports.push_back({90, 3, 1, "out", "frame", "image", "rgba8", "sRGB"});
    reject(invalid, "fog rejects extra ports");
    invalid = fog;
    operation(invalid, 90).kind = "visual.depth.blur";
    *std::find(invalid.nodeKinds.begin(), invalid.nodeKinds.end(), "visual.depth.fog") = "visual.depth.blur";
    reject(invalid, "fog lowering cannot admit another depth effect kind");
    invalid = fog;
    operation(invalid, 90).payloadXml = "<NodeParams density=\"nan\"/>";
    reject(invalid, "fog rejects nonfinite authored parameters");

    // PolyphonicImportedModelArray has no material terminal. Its exact output
    // must count alongside the imported source, score source and instancer.
    auto notes = importedScenePlan();
    useRegisteredImportedStarterPorts(notes, 71);
    visualnoteinstancing::Mapping mapping;
    mapping.appearanceSource = visualnoteinstancing::AppearanceSource::StableNoteIdentity;
    addBefore(notes, 71, 90, "visual.score.note-collection", "control-eval");
    addBefore(notes, 71, 91, "visual.3d.note-instanced-mesh", "control-eval", visualnoteinstancing::encode(mapping));
    notes.ports.insert(notes.ports.end(), {
        {90, 0, 1, "out", "control", "noteCollection", "unspecified", "unspecified"},
        {91, 0, 1, "in", "control", "mesh", "unspecified", "unspecified"},
        {91, 1, 1, "in", "control", "noteCollection", "unspecified", "unspecified"},
        {91, 2, 1, "out", "control", "mesh", "unspecified", "unspecified"}
    });
    notes.edges.insert(notes.edges.end(), {{70, 0, 91, 0}, {90, 0, 91, 1}, {91, 2, 71, 3}});
    appendImportedStarterOutput(notes, 71, 1);
    check(accepted(notes) && compiled.noteInstanceMapping && compiled.importedSceneRender
        && !compiled.importedSceneRender->material,
        "authored polyphonic imported array admits its exact outer output without a material: " + error);
    invalid = notes;
    invalid.edges.erase(invalid.edges.end() - 3);
    reject(invalid, "imported note array rejects an unbound score producer");
    invalid = notes;
    invalid.edges[2] = invalid.edges.front();
    reject(invalid, "duplicate scene edges cannot hide an unbound note source");
    invalid = notes;
    operation(invalid, 90).kind = "visual.score.future-collection";
    *std::find(invalid.nodeKinds.begin(), invalid.nodeKinds.end(), "visual.score.note-collection")
        = "visual.score.future-collection";
    reject(invalid, "imported instancer rejects a wrong-kind note source before dereferencing it");
    invalid = notes;
    addBefore(invalid, 6, 92, "visual.surface.constant.scalar", "control-eval");
    invalid.ports.push_back({92, 0, 1, "out", "control", "scalar", "unspecified", "unspecified"});
    reject(invalid, "no-material array rejects an extra unbound operation");

    // The current Note-Driven Particle and Light Rig uses all seven registered
    // particle ports; optional geometry ports are present but unconnected.
    auto rig = importedScenePlan(true, true);
    useRegisteredImportedStarterPorts(rig, 71);
    auto rigRequest = renderRequest(true, true);
    HarmonicMIDI::grid::SceneLightRecord light;
    light.id.value = 86;
    light.kind = HarmonicMIDI::grid::SceneLightKind::Directional;
    rigRequest.light = light;
    rigRequest.noteLight = visualimportedscenerender::NoteLight {0, 2, 45};
    operation(rig, 71).payloadXml = visualimportedscenerender::encode(rigRequest);
    addBefore(rig, 71, 84, "visual.3d.transform", "control-eval");
    addBefore(rig, 71, 85, "visual.3d.light.directional", "control-eval");
    rig.ports.insert(rig.ports.end(), {
        {84, 0, 1, "out", "control", "transform3D", "unspecified", "unspecified"},
        {85, 0, 1, "in", "control", "transform3D", "unspecified", "unspecified"},
        {85, 1, 1, "out", "control", "light", "unspecified", "unspecified"}
    });
    rig.edges.insert(rig.edges.end(), {{84, 0, 85, 0}, {85, 1, 71, 6}});
    addBefore(rig, -1, 90, "visual.particles", "native-gpu",
        "<NodeParams motionMode=\"1\" count=\"256\" lifetime=\"3\" size=\"7\" speed=\"0.4\" "
        "drag=\"0.25\" attraction=\"0.3\" alpha=\"0.75\"/>");
    addBefore(rig, -1, 91, "video.blend", "native-gpu", "<NodeParams opacity=\"1\" mode=\"0\"/>");
    rig.ports.insert(rig.ports.end(), {
        {90, 0, 1, "in", "event", "unspecified", "unspecified", "unspecified"},
        {90, 1, 1, "out", "frame", "image", "rgba8", "sRGB"},
        {90, 2, 1, "in", "control", "points3D", "unspecified", "unspecified"},
        {90, 3, 1, "in", "control", "points3D", "unspecified", "unspecified"},
        {90, 4, 1, "in", "control", "geometry3D", "unspecified", "unspecified"},
        {90, 5, 1, "in", "control", "points3D", "unspecified", "unspecified"},
        {90, 6, 1, "in", "control", "geometry3D", "unspecified", "unspecified"},
        {91, 0, 1, "in", "frame", "image", "rgba8", "sRGB"},
        {91, 1, 1, "out", "frame", "image", "rgba8", "sRGB"},
        {91, 2, 1, "in", "frame", "image", "rgba8", "sRGB"},
        {91, 3, 1, "in", "frame", "image", "rgba8", "sRGB"}
    });
    rig.edges.insert(rig.edges.end(), {{71, 1, 91, 0}, {90, 1, 91, 2}});
    appendImportedStarterOutput(rig, 91, 1);
    check(accepted(rig) && compiled.importedParticleOverlay && compiled.particles
        && compiled.importedSceneRender->material && compiled.importedSceneRender->deformation
        && compiled.importedSceneRender->noteLight && compiled.particleParameters.count == 256,
        "authored seven-port particle rig retains animation, surface material and note light: " + error);
    for (int port : {2, 3, 4, 5, 6})
    {
        invalid = rig;
        auto& binding = *std::find_if(invalid.ports.begin(), invalid.ports.end(), [&](const auto& item)
            { return item.nodeId == 90 && item.port == port; });
        binding.dataType = binding.dataType == "points3D" ? "geometry3D" : "points3D";
        reject(invalid, "particle rig rejects a changed optional geometry kind");
    }
    invalid = rig;
    invalid.ports.push_back({90, 7, 1, "in", "control", "geometry3D", "unspecified", "unspecified"});
    reject(invalid, "particle rig rejects extra ports beyond the registered seven");
    invalid = rig;
    invalid.edges[invalid.edges.size() - 2].toPort = 3;
    reject(invalid, "particle rig rejects the layer input in place of the overlay input");
    invalid = rig;
    invalid.edges.push_back({70, 0, 90, 4});
    reject(invalid, "particle overlay cannot consume an unlowered mesh-barrier source");

    // HolographicTradingCard uses a spatial foil, separate camera/light
    // transforms, an unconnected imagery port and the looping vertex field.
    auto card = importedScenePlan(false, false, true);
    operation(card, 72).kind = "visual.material.diffractive-foil";
    *std::find(card.nodeKinds.begin(), card.nodeKinds.end(), "visual.material.diffraction-grating")
        = "visual.material.diffractive-foil";
    useRegisteredImportedStarterPorts(card, 71);
    auto cardRequest = renderRequest(false, false, true);
    cardRequest.diffractionMaterial = spatialDiffractionMaterialRequest();
    HarmonicMIDI::grid::SceneCameraRecord camera;
    camera.id.value = 75;
    camera.transform.translation.z = 0.16f;
    camera.verticalFovRadians = 0.65f;
    camera.nearPlane = 0.01f;
    camera.farPlane = 10;
    cardRequest.camera = camera;
    light.id.value = 77;
    cardRequest.light = light;
    addBefore(card, 72, 73, "visual.3d.transform", "control-eval");
    addBefore(card, 72, 74, "visual.3d.camera.perspective", "control-eval");
    addBefore(card, 72, 75, "visual.3d.transform", "control-eval");
    addBefore(card, 72, 76, "visual.3d.light.directional", "control-eval");
    card.ports.insert(card.ports.end(), {
        {73, 0, 1, "out", "control", "transform3D", "unspecified", "unspecified"},
        {74, 0, 1, "in", "control", "transform3D", "unspecified", "unspecified"},
        {74, 1, 1, "out", "control", "camera", "unspecified", "unspecified"},
        {75, 0, 1, "out", "control", "transform3D", "unspecified", "unspecified"},
        {76, 0, 1, "in", "control", "transform3D", "unspecified", "unspecified"},
        {76, 1, 1, "out", "control", "light", "unspecified", "unspecified"},
        {72, 1, 1, "in", "frame", "image", "rgba8", "sRGB"},
        {72, 2, 1, "in", "control", "field", "unspecified", "unspecified"}
    });
    card.edges.insert(card.edges.end(), {{73, 0, 74, 0}, {74, 1, 71, 5}, {75, 0, 76, 0}, {76, 1, 71, 6}});
    cardRequest.materialField = videohelper::tests::materialFieldFixture(72);
    cardRequest.materialField->target = 13;
    cardRequest.materialField->gain = 30;
    const auto field = materialfield::admit(*cardRequest.materialField, error);
    check(field.has_value(), "card's looping foil field is canonical: " + error);
    if (!field) return;
    const auto topology = materialfield::topology(*cardRequest.materialField, field->descriptor());
    for (const auto& [id, kind] : topology.nodes)
    {
        addBefore(card, 72, id, kind.c_str(), "control-eval");
        const int output = topology.outputPorts.at(id);
        for (int port = 0; port <= output; ++port)
            card.ports.push_back({id, port, 1, port == output ? "out" : "in", "control", "field", "unspecified", "unspecified"});
    }
    for (const auto& [from, output, to, input] : topology.edges) card.edges.push_back({from, output, to, input});
    operation(card, 71).payloadXml = visualimportedscenerender::encode(cardRequest);
    appendImportedStarterOutput(card, 71, 1);
    check(accepted(card) && compiled.importedSceneRender && compiled.importedSceneRender->diffractionMaterial
        && compiled.importedSceneRender->diffractionMaterial->spatialFoil && compiled.importedSceneRender->materialField,
        "authored holographic card reaches strict execution with its exact foil field and outer output: " + error);
    invalid = card;
    operation(invalid, 72).kind = "visual.material.diffraction-grating";
    *std::find(invalid.nodeKinds.begin(), invalid.nodeKinds.end(), "visual.material.diffractive-foil")
        = "visual.material.diffraction-grating";
    reject(invalid, "a grating terminal cannot own the spatial foil request");
    auto analyticFoil = card;
    auto analyticRequest = cardRequest;
    analyticRequest.diffractionMaterial = diffractionMaterialRequest();
    analyticRequest.materialField->target = 4;
    operation(analyticFoil, 71).payloadXml = visualimportedscenerender::encode(analyticRequest);
    check(accepted(analyticFoil), "foil's authored analytic mode retains the physical grating route: " + error);
    auto grating = analyticFoil;
    operation(grating, 72).kind = "visual.material.diffraction-grating";
    *std::find(grating.nodeKinds.begin(), grating.nodeKinds.end(), "visual.material.diffractive-foil")
        = "visual.material.diffraction-grating";
    check(accepted(grating), "physical grating snapshot admits only its exact material topology: " + error);
    invalid = card;
    operation(invalid, 72).kind = "visual.material.diffractive-foil.future";
    *std::find(invalid.nodeKinds.begin(), invalid.nodeKinds.end(), "visual.material.diffractive-foil")
        = "visual.material.diffractive-foil.future";
    check(!accepted(invalid), "unknown foil kinds fail snapshot admission");
    invalid = card;
    invalid.edges.erase(std::find_if(invalid.edges.begin(), invalid.edges.end(),
        [](const auto& edge) { return edge.fromNodeId == 72 && edge.toNodeId == 71; }));
    reject(invalid, "foil must own the render material input");
    invalid = card;
    addBefore(invalid, 71, 92, "visual.material.diffractive-foil", "control-eval");
    invalid.ports.push_back({92, 0, 1, "out", "control", "material", "unspecified", "unspecified"});
    reject(invalid, "card rejects an extra unbound foil terminal");
    invalid = card;
    invalid.edges.erase(std::find_if(invalid.edges.begin(), invalid.edges.end(),
        [](const auto& edge) { return edge.toNodeId == 72 && edge.toPort == 2; }));
    reject(invalid, "foil field request cannot survive an unbound field input");
}
videowire::CompiledVisualLayerPlan importedMaterialFramePlan(bool diffraction = false)
{
    auto plan = importedScenePlan(!diffraction, false, diffraction);
    visualimportedscenerender::Request request;
    visualimportedscenerender::decode(plan.operations.back().payloadXml, request);
    if (diffraction)
    {
        request.diffractionMaterial->version = diffractionmaterialbinding::kGraphFrameWireVersion;
        request.diffractionMaterial->graphFrame = surfacematerialbinding::TextureSlotBinding::GraphFrameEndpoint {91, 0};
    }
    else
    {
        request.material->version = surfacematerialbinding::kGraphFrameWireVersion;
        auto& texture = request.material->binding.textures.front();
        texture.source = surfacematerialbinding::TextureSourceKind::GraphFrame;
        texture.videoResource = {};
        texture.graphFrame = surfacematerialbinding::TextureSlotBinding::GraphFrameEndpoint { 91, 0 };
    }
    plan.operations.back().payloadXml = visualimportedscenerender::encode(request);
    // Deliberately serialize the prerequisite after Render 3D. The common
    // preparation seam, not incidental document order, must run it first.
    plan.nodeKinds.push_back("video.source");
    plan.nodeIds.push_back(91);
    plan.operations.push_back({ 91, "video.source", "source-decode", "" });
    plan.ports.push_back({ 91, 0, 1, "out", "frame", "image", "rgba8", "sRGB" });
    if (!diffraction)
        plan.ports.push_back({ 82, 11, 3, "in", "control", "vec3", "unspecified", "unspecified" });
    plan.ports.push_back({ diffraction ? 72 : 82, diffraction ? 1 : 12, 1, "in", "frame", "image", "rgba8", "sRGB" });
    plan.edges.push_back({ 91, 0, diffraction ? 72 : 82, diffraction ? 1 : 12 });
    return plan;
}

void testMaterialFramePrerequisite(bool diffraction = false, bool generated = false)
{
    using namespace videohelper::importedscene;
    auto plan = importedMaterialFramePlan(diffraction);
    if (generated)
    {
        std::string generatedError;
        const auto lighting = geometryCompositionFixture(generatedError);
        const auto scene = lighting ? videohelper::tests::generatedSurfaceScene(*lighting, generatedError) : std::nullopt;
        check(scene.has_value(), "generated Frame schedule retains a real Geometry Core mesh");
        if (!scene) return;
        visualimportedscenerender::Request request;
        visualimportedscenerender::decode(plan.operations[plan.operations.size() - 2].payloadXml, request);
        request.sourceStableId = scene->id.value;
        request.renderStableId = 12;
        request.asset = {};
        request.sceneIndex.reset();
        request.structuralRevision = request.evaluationRevision = plan.structuralRevision;
        request.sceneSnapshot = std::make_shared<const HarmonicMIDI::grid::Visual3DScene>(*scene);
        auto& material = *request.material;
        material.scene = scene->id;
        material.sceneSnapshot = request.sceneSnapshot;
        material.sceneRevision = material.structuralRevision = material.evaluationRevision
            = material.programRevision = material.binding.surfaceMaterialRevision = plan.structuralRevision;
        material.binding.object = scene->objects[0].id;
        const int source = static_cast<int>(request.sourceStableId - 1);
        plan.nodeIds = {source, 91, 11};
        plan.nodeKinds = {videowire::geometry::kRetainedSceneOperation, "video.source", "visual.3d.render"};
        plan.operations = {{source, plan.nodeKinds[0], "control-eval", ""},
            {91, "video.source", "source-decode", ""},
            {11, plan.nodeKinds[2], "native-gpu", visualimportedscenerender::encode(request)}};
        plan.edges = {{source, 0, 11, 0}, {91, 0, 11, 12}};
        plan.ports = {{source, 0, 1, "out", "control", "scene3D", "unspecified", "unspecified"},
            {11, 0, 1, "in", "control", "scene3D", "unspecified", "unspecified"},
            {11, 1, 1, "out", "frame", "image", "rgba8", "sRGB"},
            {91, 0, 1, "out", "frame", "image", "rgba8", "sRGB"},
            {11, 12, 1, "in", "frame", "image", "rgba8", "sRGB"}};
        check(!plan.operations[2].payloadXml.empty(), "generated Frame material serializes the existing Surface binding");
    }
    if (diffraction)
    {
        visualimportedscenerender::Request request;
        check(visualimportedscenerender::decode(plan.operations[plan.operations.size() - 2].payloadXml, request),
              "Frame diffraction v5 decodes with the legacy canonical lighting plan");
        if (request.diffractionMaterial)
            for (int fault = 0; fault < 4; ++fault)
            {
                auto invalid = *request.diffractionMaterial;
                if (fault == 0) invalid.version = diffractionmaterialbinding::kWireVersion;
                if (fault == 1) invalid.graphFrame.reset();
                if (fault == 2) invalid.graphFrame->node = -1;
                if (fault == 3) invalid.graphFrame->port = 1;
                std::string admissionError;
                check(!visualimportedscenerender::detail::validDiffractionMaterial(invalid)
                          && !diffractionmaterialbinding::admit(invalid, admissionError),
                      "diffraction Frame identity and version must agree at wire and native admission");
            }
    }
    videowire::VisualLayerExecution compiled;
    std::string error;
    const auto frameSource = std::find_if(plan.operations.begin(), plan.operations.end(),
        [](const auto& operation) { return operation.nodeId == 91; });
    check(frameSource != plan.operations.end(), "Frame prerequisite fixture contains its exact source");
    if (frameSource == plan.operations.end()) return;
    const auto frameSourceIndex = static_cast<std::size_t>(frameSource - plan.operations.begin());
    check(videowire::compileVisualLayerExecution(plan, compiled, error)
              && compiled.materialFrameSource && compiled.materialFrameSource->nodeId == 91
              && compiled.materialFrameSource->operationIndex == frameSourceIndex
              && compiled.materialFrameSource->nodeIndex < plan.nodeIds.size()
              && plan.nodeIds[compiled.materialFrameSource->nodeIndex] == 91,
          "Frame dependency identifies its exact authored source for Render 3D preparation: " + error);
    for (int fault = 0; fault < 6; ++fault)
    {
        auto malformed = plan;
        if (fault == 0) malformed.edges.back().fromNodeId = 70;
        if (fault == 1) malformed.edges.push_back(malformed.edges.back());
        if (fault == 2) malformed.operations[frameSourceIndex].backendCapability = "native-gpu";
        if (fault == 3)
            for (auto& port : malformed.ports)
                if (port.nodeId == 91) port.colorSpace = "linear";
        if (fault == 4) malformed.ports.pop_back();
        if (fault == 5) malformed.operations[frameSourceIndex].payloadXml = "unadmitted source override";
        check(!videowire::compileVisualLayerExecution(malformed, compiled, error),
              "Frame prerequisite rejects topology, descriptor, or source-authority mismatch " + std::to_string(fault));
    }

    FakeExecution execution;
    FakeLayer layer;
    std::vector<FakeReceipt> owners;
    VisualImportedScenePlanCache cache;
    const auto prepare = [&](NativeImportedSceneRenderUse use, const MaterialFrameResolver* resolver)
    {
        return prepareVisualImportedSceneLayerAtTime({plan}, 7, 640, 360, 1.25, 24.0,
            use, &execution, &cache, layer, owners, error, nullptr, { 12, 34 }, resolver);
    };
    check(prepare(NativeImportedSceneRenderUse::Preview, nullptr) == VisualImportedScenePreparation::rejected
              && execution.previewCalls == 0 && owners.empty(),
          "missing native Frame resolver rejects before scene submission or imported-texture fallback");

    for (const auto* backend : { "opengl", "metal" })
        for (const auto use : { NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export })
        {
            execution = {};
            execution.nextFrame->api = backend;
            auto input = std::make_shared<FakeFrame>();
            input->api = backend;
            input->frameWidth = input->descriptorWidth = 160;
            input->frameHeight = 90;
            const std::weak_ptr<FakeFrame> retained = input;
            int evaluations = 0;
            MaterialFrameResolver resolver = [&](const MaterialFrameEvaluation& evaluation,
                                                 MaterialFrameReceipt& receipt, std::string&)
            {
                ++evaluations;
                check(execution.previewCalls == 0 && execution.exportCalls == 0
                          && evaluation.endpoint.node == 91 && evaluation.endpoint.port == 0
                          && evaluation.frame.frame == 30 && evaluation.frame.rateNumerator == 24
                          && evaluation.frame.rateDenominator == 1 && evaluation.clipId == 7
                          && evaluation.structuralRevision == plan.structuralRevision
                          && evaluation.evaluationRevision == (generated ? plan.structuralRevision : 0)
                          && evaluation.projectGeneration == 12 && evaluation.helperGeneration == 34
                          && evaluation.use == use,
                      "Frame is evaluated at the exact publication/time before preview or export draw");
                receipt = { evaluation, input };
                return true;
            };
            check(prepare(use, &resolver) == VisualImportedScenePreparation::rendered
                      && evaluations == 1 && owners.size() == 1
                      && execution.lastRequest.runtimeInputs.materialFrameTexture == input
                      && owners.front().materialFrameTexture == input,
                  std::string(backend) + " shared prerequisite seam carries the owner into submission and receipt: " + error);
            input.reset();
            execution.lastRequest = {};
            check(!retained.expired(), "output receipt keeps the Frame dependency alive after evaluator/request release");
            owners.clear();
            check(retained.expired(), "releasing the output receipt releases its final Frame lease");
        }

    execution = {};
    auto input = std::make_shared<FakeFrame>();
    for (int fault = 0; fault < 12; ++fault)
    {
        MaterialFrameResolver resolver = [&](const MaterialFrameEvaluation& evaluation,
                                             MaterialFrameReceipt& receipt, std::string&)
        {
            receipt = { evaluation, input };
            if (fault == 0) ++receipt.evaluation.endpoint.node;
            if (fault == 1) ++receipt.evaluation.endpoint.port;
            if (fault == 2) ++receipt.evaluation.frame.frame;
            if (fault == 3) ++receipt.evaluation.frame.rateNumerator;
            if (fault == 4) ++receipt.evaluation.clipId;
            if (fault == 5) ++receipt.evaluation.structuralRevision;
            if (fault == 6) ++receipt.evaluation.evaluationRevision;
            if (fault == 7) ++receipt.evaluation.projectGeneration;
            if (fault == 8) ++receipt.evaluation.helperGeneration;
            if (fault == 9) receipt.evaluation.use = NativeImportedSceneRenderUse::Export;
            if (fault == 10) receipt.nativeFrame.reset();
            if (fault == 11) input->descriptorSampleCount = 2;
            return true;
        };
        check(prepare(NativeImportedSceneRenderUse::Preview, &resolver) == VisualImportedScenePreparation::rejected
                  && execution.previewCalls == 0 && owners.empty(),
              "stale/unowned Frame prerequisite cannot submit or retain a scene frame " + std::to_string(fault));
    }
    input->descriptorSampleCount = 1;
    MaterialFrameResolver resolver = [&](const MaterialFrameEvaluation& evaluation,
                                         MaterialFrameReceipt& receipt, std::string&)
    { receipt = { evaluation, input }; return true; };
    input->contextIdentity = 999;
    check(prepare(NativeImportedSceneRenderUse::Preview, &resolver) == VisualImportedScenePreparation::rejected
              && owners.empty(), "different native contexts cannot publish a material Frame scene result");
    execution = {};
    input->contextIdentity = 77;
    input->api = "metal";
    check(prepare(NativeImportedSceneRenderUse::Preview, &resolver) == VisualImportedScenePreparation::rejected
              && owners.empty(), "different native backends cannot publish a material Frame scene result");

    for (const auto use : { NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export })
    {
        execution = {};
        owners.clear();
        std::vector<std::int64_t> sourceFrames;
        MaterialFrameResolver seekResolver = [&](const MaterialFrameEvaluation& evaluation,
                                                MaterialFrameReceipt& receipt, std::string&)
        {
            sourceFrames.push_back(evaluation.frame.frame);
            auto sought = std::make_shared<FakeFrame>();
            sought->image = sought->view = static_cast<std::uintptr_t>(1000 + evaluation.frame.frame);
            receipt = { evaluation, std::move(sought) };
            return true;
        };
        // Forward playback, backwards seek, then a loop back to the same time.
        for (const double seconds : { 0.5, 1.5, 0.25, 0.5 })
        {
            owners.clear();
            check(prepareVisualImportedSceneLayerAtTime({plan}, 7, 640, 360, seconds, 24.0,
                      use, &execution, &cache, layer, owners, error, nullptr, {12, 34}, &seekResolver)
                      == VisualImportedScenePreparation::rendered
                      && execution.lastRequest.runtimeInputs.materialFrameTexture->colorImageHandle()
                          == static_cast<std::uintptr_t>(1000 + std::llround(seconds * 24)),
                  "preview/export seek and loop resolve the exact source frame before each scene draw");
        }
        check(sourceFrames == std::vector<std::int64_t> {12, 36, 6, 12},
              "preview and offline export have identical stateless Frame evaluation order");
        const auto draws = execution.previewCalls + execution.exportCalls;
        owners.clear();
        MaterialFrameResolver missing = [](const MaterialFrameEvaluation&, MaterialFrameReceipt&, std::string& error)
        { error = "video source missing"; return false; };
        check(prepare(use, &missing) == VisualImportedScenePreparation::rejected
                  && execution.previewCalls + execution.exportCalls == draws && owners.empty(),
              "a missing decoder frame after a successful seek cannot publish the previous video texture");
    }

    plan = importedScenePlan(!diffraction, false, diffraction);
    execution = {};
    MaterialFrameResolver forbidden = [](const MaterialFrameEvaluation&, MaterialFrameReceipt&, std::string&)
    { check(false, "imported texture fallback must not invoke a Frame resolver"); return false; };
    check(prepare(NativeImportedSceneRenderUse::Preview, &forbidden) == VisualImportedScenePreparation::rendered
              && !execution.lastRequest.runtimeInputs.materialFrameTexture,
          "legacy imported slot-zero texture remains unchanged without a Frame edge");
}

videowire::CompiledVisualLayerPlan importedVertexPlan()
{
    auto plan = importedScenePlan(true, true);
    auto render = std::move(plan.operations.back());
    plan.operations.pop_back();
    plan.nodeIds.pop_back();
    plan.nodeKinds.pop_back();
    auto request = renderRequest(true, true);
    request.material->vertexModifier = videohelper::test::audioNormalDisplacement();
    const auto& vertex = *request.material->vertexModifier;
    constexpr int material = 83;
    plan.ports.push_back({material, 11, 3, "in", "control", "vec3", "unspecified", "unspecified"});
    for (const auto& record : vertex.records)
    {
        const auto id = static_cast<int>(record.stableId - 1u);
        const auto kind = videowire::vertexModifierGraphKindToken(record.operation);
        plan.operations.push_back({id, kind, "control-eval", ""});
        plan.nodeIds.push_back(id);
        plan.nodeKinds.push_back(kind);
        const auto schema = videowire::vertexModifierOperationSchema(record.operation);
        const auto addPort = [&](int port, const char* direction, videowire::VertexModifierValueType type)
        {
            const auto scalar = type == videowire::VertexModifierValueType::scalar;
            plan.ports.push_back({id, port, scalar ? 1 : 3, direction, "control",
                                 scalar ? "scalar" : "vec3", "unspecified", "unspecified"});
        };
        for (std::uint8_t input = 0; input < schema.inputCount; ++input)
        {
            addPort(input, "in", schema.inputTypes[input]);
            const auto& source = vertex.records[record.inputs[input] - 1u];
            plan.edges.push_back({static_cast<int>(source.stableId - 1u), source.inputCount, id, input});
        }
        addPort(schema.inputCount, "out", schema.resultType);
    }
    plan.edges.push_back({6, 2, material, 11});
    render.payloadXml = visualimportedscenerender::encode(request);
    plan.nodeIds.push_back(render.nodeId);
    plan.nodeKinds.push_back(render.kind);
    plan.operations.push_back(std::move(render));
    return plan;
}
} // namespace

void testDiffractionRuntimeParameters()
{
    using namespace videohelper::importedscene;
    const auto plan = importedScenePlan(false, false, true);
    const std::vector<videowire::CompiledVisualLayerPlan> plans {plan};
    std::map<std::string, double> parameters;
    seedImportedSceneRuntimeParameters(plans, 7, parameters);
    check(parameters.count("visual73/grooveSpacingNanometres") == 1
              && parameters.count("visual73/rmsSlope") == 1
              && parameters.count("visual73/emissionGain") == 0,
          "the exact diffraction terminal seeds physical scalar targets, not a false emission channel");
    const auto originalSpacing = parameters.at("visual73/grooveSpacingNanometres");
    visualimportedscenerender::Request authored;
    check(visualimportedscenerender::decode(plan.operations.back().payloadXml, authored),
          "runtime parameter tests retain the canonical authored material request");
    if (authored.diffractionMaterial)
    {
        using namespace diffractionmaterialbinding;
        const auto original = *authored.diffractionMaterial;
        auto evaluated = original;
        std::string diagnostic;
        check(applyRuntimeParameters(original, {{"grooveDepthNanometres", 90}, {"rmsSlope", 0.2},
                {"grooveSpacingNanometres", 1300}, {"directionAngleDegrees", 45}}, evaluated, diagnostic)
                  && diffractionmaterial::admit(evaluated.material, diagnostic).has_value()
                  && evaluated.material.geometry.grooveSpacingNanometres == 1300
                  && evaluated.scene == original.scene && evaluated.materialRevision == original.materialRevision,
              "physical scalar modulation preserves material identity and re-admits valid values");
        for (const auto& invalid : std::vector<RuntimeParameters> {{{"profile", 1}}, {{"lattice", 1}},
                {{"coating", 1}}, {{"grooveSpacingNanometres", 0}}, {{"maximumOrder", 0}},
                {{"coatingThicknessNanometres", 200}}, {{"not-a-parameter", 1}}})
            check(!applyRuntimeParameters(original, invalid, evaluated, diagnostic),
                  "runtime controls reject unadmitted modes, inactive coating and unsupported destinations");
        auto blazed = original;
        blazed.material.microstructure.profile = diffractionmaterial::GrooveProfile::BlazedSawtooth;
        check(applyRuntimeParameters(blazed, {{"grooveSpacingNanometres", 1400}, {"grooveDepthNanometres", 100}},
                    evaluated, diagnostic) && diffractionmaterial::admit(evaluated.material, diagnostic).has_value(),
              "blazed scalar modulation recomputes its constrained blaze angle");
        blazed.material.geometry.lattice = diffractionmaterial::GratingLattice::CrossedTwoDimensional;
        blazed.material.geometry.secondaryDirectionUv = {0, 1};
        blazed.material.geometry.secondaryGrooveSpacingNanometres = 1000;
        check(applyRuntimeParameters(blazed, {{"grooveSpacingNanometres", 1400}}, evaluated, diagnostic)
                  && !diffractionmaterial::admit(evaluated.material, diagnostic),
              "unequal crossed blazed periods cannot pass physical admission");
        check(applyRuntimeParameters(blazed, {{"grooveSpacingNanometres", 1400},
                {"secondaryGrooveSpacingNanometres", 1400}}, evaluated, diagnostic)
                  && diffractionmaterial::admit(evaluated.material, diagnostic).has_value(),
              "compatible crossed blazed periods remain editable together");
        auto coated = original;
        coated.material.coating = {diffractionmaterial::CoatingModel::IncoherentDielectric, 1000, {1.5f, 0.001f}};
        check(applyRuntimeParameters(coated, {{"coatingThicknessNanometres", 200},
                {"coatingRefractiveIndex", 1.7}, {"coatingExtinctionCoefficient", 0.01}}, evaluated, diagnostic)
                  && diffractionmaterial::admit(evaluated.material, diagnostic).has_value(),
              "the active coating's optical parameters remain physical runtime values");
    }
    FakeExecution execution;
    FakeLayer layer;
    std::vector<FakeReceipt> owners;
    std::string error;
    for (const auto use : {NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export})
        for (const double seconds : {0.0, 1.0, 0.5, 0.0})
        {
            parameters["visual73/grooveSpacingNanometres"] = originalSpacing + 100 * seconds;
            check(prepareVisualImportedSceneLayerAtTime(plans, 7, 640, 360, seconds, 24,
                    use, &execution, layer, owners, error, &parameters)
                      == VisualImportedScenePreparation::rendered,
                  "timeline/score/audio parameter values reach diffraction at forward, seek and loop samples");
            const auto& request = execution.lastRequest;
            check(request.frame.frame == static_cast<std::int64_t>(seconds * 24)
                      && request.diffractionMaterial->material.geometry.grooveSpacingNanometres == originalSpacing,
                  "modulation keeps the published material immutable and carries the rational frame identity");
            if (seconds == 0)
                check(request.runtimeInputs.diffractionParameters.empty(), "returning to the authored value clears frame-local modulation");
            else
                check(request.runtimeInputs.diffractionParameters.at("grooveSpacingNanometres") == originalSpacing + 100 * seconds,
                      "the frame carries only the exact changed material scalar");
        }
    for (const auto& key : {"visual73/not-a-parameter", "visual73/rmsSlope"})
    {
        auto invalid = parameters;
        invalid[key] = std::numeric_limits<double>::quiet_NaN();
        check(prepareVisualImportedSceneLayerAtTime(plans, 7, 640, 360, 0, 24,
                NativeImportedSceneRenderUse::Preview, &execution, layer, owners, error, &invalid)
                  == VisualImportedScenePreparation::rejected,
              "nonfinite and unknown modulation cannot reuse a previous diffraction frame");
    }
}

void testMaterialFieldBinding()
{
    using namespace videohelper::importedscene;
    auto plan = importedScenePlan(false, false, true);
    visualimportedscenerender::Request request;
    check(visualimportedscenerender::decode(plan.operations.back().payloadXml, request), "field fixture has an authored request");
    request.materialField = videohelper::tests::materialFieldFixture(72);
    std::string error;
    const auto field = materialfield::admit(*request.materialField, error);
    check(field.has_value(), "canonical Geometry Core field transport admits without a new field identity: " + error);
    if (!field) return;
    const auto topology = materialfield::topology(*request.materialField, field->descriptor());
    check(std::any_of(topology.nodes.begin(), topology.nodes.end(), [](const auto& node)
            { return node.second == "visual.geometry.field.float"; }),
        "canonical material field topology carries the exact field-float kind");
    for (const auto& [id, kind] : topology.nodes)
    {
        plan.nodeIds.insert(plan.nodeIds.end()-1, id);
        plan.nodeKinds.insert(plan.nodeKinds.end()-1, kind);
        plan.operations.insert(plan.operations.end()-1, {id, kind, "control-eval", ""});
        const auto output = topology.outputPorts.at(id);
        for (int port = 0; port <= output; ++port)
            plan.ports.push_back({id, port, 1, port == output ? "out" : "in", "control", "field", "unspecified", "unspecified"});
    }
    for (const auto& [from, output, to, input] : topology.edges) plan.edges.push_back({from, output, to, input});
    plan.ports.push_back({72, 1, 1, "in", "frame", "image", "rgba8", "sRGB"});
    plan.ports.push_back({72, 2, 1, "in", "control", "field", "unspecified", "unspecified"});
    plan.operations.back().payloadXml = visualimportedscenerender::encode(request);
    visualimportedscenerender::Request restored;
    check(!plan.operations.back().payloadXml.empty()
              && visualimportedscenerender::decode(plan.operations.back().payloadXml, restored)
              && visualimportedscenerender::sameRequest(request, restored), "v12 preserves the field adapter and existing request exactly");
    FakeExecution execution; FakeLayer layer; std::vector<FakeReceipt> owners;
    const auto base = request.diffractionMaterial->material.roughness.rmsHeightNanometres;
    for (auto use : {NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export})
        for (double seconds : {0.0, 1.0, 0.5, 4.0, 0.0})
        {
            std::map<std::string,double> parameters{{"visual73/rmsHeightNanometres", base + 5}};
            check(prepareVisualImportedSceneLayerAtTime({plan}, 7, 640, 360, seconds, 24,
                    use, &execution, layer, owners, error, &parameters) == VisualImportedScenePreparation::rendered,
                  "field scheduler admits forward, backward and loop samples: " + error);
            const auto& evaluated = execution.lastRequest.runtimeInputs.diffractionParameters;
            check(evaluated.count("rmsHeightNanometres")
                && std::abs(evaluated.at("rmsHeightNanometres") - (base + 5 + 10*std::sin(seconds*1.5707963267948966))) < 1e-6,
                "rational-time field evaluation adds after Patch Bay and repeats on seek/export");
        }
    for (int mutation = 0; mutation < 3; ++mutation)
    {
        auto invalid = plan;
        if (mutation == 0) invalid.edges.back().toPort = 1;
        if (mutation == 1) invalid.operations[2].kind = "visual.geometry.field-timeline.point";
        if (mutation == 2) invalid.ports.back().dataType = "vector";
        check(prepareVisualImportedSceneLayerAtTime({invalid}, 7, 640, 360, 0, 24,
                NativeImportedSceneRenderUse::Preview, &execution, layer, owners, error)
                  == VisualImportedScenePreparation::rejected, "material field port, node kind and endpoint tampering fail admission");
    }
    for (int mutation = 0; mutation < 4; ++mutation)
    {
        auto invalid = *request.materialField;
        if (mutation == 0) invalid.index = materialfield::kMaximumElements;
        if (mutation == 1) invalid.gain = std::numeric_limits<double>::quiet_NaN();
        if (mutation == 2) { invalid.reduction = materialfield::Reduction::index; invalid.index = 4; }
        if (mutation == 3) invalid.materialNode = 9000;
        check(!materialfield::admit(invalid, error), "out-of-range samples, nonfinite gain and overlapping identities reject");
    }
    for (auto reduction : {materialfield::Reduction::index, materialfield::Reduction::mean,
                           materialfield::Reduction::minimum, materialfield::Reduction::maximum})
    {
        auto binding = *request.materialField; binding.reduction = reduction;
        videowire::geometry::RuntimeFieldEvaluation clock; clock.timelineSeconds = 1;
        diffractionmaterialbinding::RuntimeParameters parameters;
        check(materialfield::evaluate(binding, clock, *request.diffractionMaterial, parameters, error)
                  && std::abs(parameters.at("rmsHeightNanometres") - (base+10)) < 1e-6,
              "each bounded reduction uses the canonical evaluated Field samples");
    }
    const auto score = videohelper::tests::materialFieldFixture(72, true);
    check(!materialfield::admit(videohelper::tests::materialFieldFixture(72, false, 2049), error),
          "Material Field rejects excess cardinality before runtime evaluation");
    videowire::geometry::RuntimeFieldEvaluation evaluation;
    diffractionmaterialbinding::RuntimeParameters parameters;
    check(!materialfield::evaluate(score, evaluation, *request.diffractionMaterial, parameters, error),
          "Score Sample Field cannot fall back to a constant without the canonical score evaluator");
    evaluation.scoreAt = [&](const auto& operation, const auto& positions, auto& output, auto& diagnostic)
    {
        check(operation.field.values[4] == 9003 && positions.elements.size() == 4,
              "existing Score Sample identity and sample positions reach RuntimeFieldEvaluation");
        output.elements = {{{1,0,0,0}},{{2,0,0,0}},{{3,0,0,0}},{{4,0,0,0}}};
        diagnostic.clear(); return true;
    };
    for (auto reduction : {materialfield::Reduction::index, materialfield::Reduction::mean,
                           materialfield::Reduction::minimum, materialfield::Reduction::maximum})
    {
        auto binding = score; binding.reduction = reduction; binding.index = 2;
        parameters.clear();
        const auto expected = reduction == materialfield::Reduction::index ? 3.0
            : reduction == materialfield::Reduction::mean ? 2.5
            : reduction == materialfield::Reduction::minimum ? 1.0 : 4.0;
        check(materialfield::evaluate(binding, evaluation, *request.diffractionMaterial, parameters, error)
                  && parameters.at("rmsHeightNanometres") == base + 10*expected,
              "nonuniform canonical score samples use defined index, mean, minimum and maximum semantics");
    }
}

void testReactiveMaterialFramePrerequisite()
{
    using namespace videohelper::importedscene;
    std::string error;
    const auto plan = reactiveFramePlan(error);
    videowire::VisualLayerExecution execution;
    check(videowire::compileVisualLayerExecution(plan,execution,error) && execution.geometryFramePlan,
        "reactive Geometry Image admits a separately scheduled canonical Frame prerequisite");
    if (!execution.geometryFramePlan) return;
    struct Renderer
    {
        std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> leaseShaderFrame(const videorender::LayerDesc&,std::string&) { return {}; }
        bool prepareFlatShaderBridge(const videorender::LayerDesc&,std::string&) { return true; }
        int outputWidth() const { return 640; }
        int outputHeight() const { return 360; }
    } renderer;
    videorender::LayerDesc layer;
    videowire::VisualPlanEvaluationContext context;
    context.projectGeneration = 12; context.helperGeneration = 13; context.deviceGeneration = 14;
    context.materialFrameClock = [](double seconds,auto& frame,std::string& failure) {
        return videohelper::importedanimation::importedAnimationFrameIdentity(seconds,29.97,frame,failure);
    };
    auto image = std::make_shared<FakeFrame>();
    int calls = 0;
    context.materialFrameResolver = [&](const MaterialFrameEvaluation& evaluation,MaterialFrameReceipt& receipt,std::string&) {
        ++calls; receipt = {evaluation,image}; return true;
    };
    for (auto use : {videohelper::geometry::PlanUse::preview,videohelper::geometry::PlanUse::exportRender})
        for (double seconds : {0.0,1.0,0.0})
        {
            const auto before = calls;
            check(videowire::prepareGeometryMaterialFrame(renderer,plan,layer,use,seconds,nullptr,context,error)
                && calls == before + 1,"reactive Frame dependency refreshes on playback, repeat seek and preview/export evaluation");
            visualdeformation::RationalFrameTime expected;
            context.materialFrameClock(seconds,expected,error);
            check(context.geometryMaterialFrame && context.geometryMaterialFrame->evaluation.frame == expected
                && context.geometryMaterialFrame->evaluation.clipId == plan.clipId
                && context.geometryMaterialFrame->evaluation.endpoint.node == 91
                && context.geometryMaterialFrame->evaluation.use == (use == videohelper::geometry::PlanUse::preview
                    ? NativeImportedSceneRenderUse::Preview : NativeImportedSceneRenderUse::Export),
                "reactive Frame uses the same rational timeline and owned receipt contract as imported Scene3D");
        }
    for (int fault = 0; fault < 7; ++fault)
    {
        context.materialFrameResolver = [&](const auto& evaluation,auto& receipt,std::string&) {
            receipt = {evaluation,image};
            if (fault == 0) ++receipt.evaluation.frame.frame;
            if (fault == 1) ++receipt.evaluation.endpoint.node;
            if (fault == 2) ++receipt.evaluation.structuralRevision;
            if (fault == 3) ++receipt.evaluation.projectGeneration;
            if (fault == 4) ++receipt.evaluation.helperGeneration;
            if (fault == 5) receipt.evaluation.use = NativeImportedSceneRenderUse::Export;
            if (fault == 6) receipt.nativeFrame.reset();
            return true;
        };
        check(!videowire::prepareGeometryMaterialFrame(renderer,plan,layer,videohelper::geometry::PlanUse::preview,
            0,nullptr,context,error),"reactive Frame rejects stale timeline, endpoint, revision, reopen, helper and mode receipts");
    }
    context.materialFrameResolver = {};
    check(!videowire::prepareGeometryMaterialFrame(renderer,plan,layer,videohelper::geometry::PlanUse::preview,
        0,nullptr,context,error),"missing reactive Frame decoder fails closed despite a previous live receipt");
    for (int fault = 0; fault < 4; ++fault)
    {
        auto bad = plan;
        if (fault == 0) bad.edges.clear();
        if (fault == 1) bad.edges.push_back(bad.edges.front());
        if (fault == 2) bad.edges[0].fromNodeId = 999;
        if (fault == 3) bad.operations[1].payloadXml = "foreign-decoder";
        check(!videowire::compileVisualLayerExecution(bad,execution,error),
            "reactive Frame schedule cannot drop, duplicate or substitute the authored producer");
    }
}

void testMaterialFrameDagAndTime()
{
    using namespace videowire;
    using namespace videohelper::importedscene;
    std::string error;
    auto plan = reactiveFramePlan(error);
    plan.nodeIds.push_back(92); plan.nodeKinds.push_back("video.layer.source");
    plan.operations.push_back({92,"video.layer.source","source-decode",materialframesource::encode(111)});
    plan.ports.push_back({92,0,1,"out","frame","image","rgba8","sRGB"});
    plan.ports.push_back({71,13,1,"in","frame","image","rgba8","sRGB"});
    plan.edges.push_back({92,0,71,13});
    const std::vector<materialframe::Endpoint> endpoints {{91,0},{92,0}};
    auto schedule = materialframe::admit(plan,71,endpoints,[](auto&,auto&) { return false; },error);
    check(schedule && schedule->operations.size() == 2,"independent decoded sources admit a bounded Frame DAG");
    if (!schedule) return;
    int clip = 0;
    check(materialframe::sourceClip(plan,{92,0},clip,error) && clip == 111,"Layer Source preserves the canonical project clip ID");
    struct Renderer {
        std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> leaseShaderFrame(const videorender::LayerDesc&,std::string&) { return {}; }
        std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> leaseTemporalFrame(
            const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& current,
            const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& previous,
            const visualtemporaloperation::Payload&,float,std::string&) { return previous ? previous : current; }
    } renderer;
    VisualLayerExecution execution; execution.geometryFrameSchedule = schedule;
    videorender::LayerDesc layer;
    VisualPlanEvaluationContext context;
    context.materialFrameClock = [](double seconds,auto& frame,std::string&) {
        frame = {static_cast<std::int64_t>(std::llround(seconds*30)),30,1}; return true;
    };
    std::vector<std::pair<int,std::int64_t>> requested;
    context.materialFrameResolver = [&](const auto& request,auto& receipt,std::string&) {
        requested.emplace_back(request.endpoint.node,request.frame.frame);
        auto image = std::make_shared<FakeFrame>(); image->image = 100 + request.endpoint.node;
        receipt = {request,std::move(image)}; return true;
    };
    for (auto use : {videohelper::geometry::PlanUse::preview,videohelper::geometry::PlanUse::exportRender})
        for (double seconds : {1.0,0.0,1.0}) {
            requested.clear();
            check(prepareGeometryMaterialFrame(renderer,plan,layer,use,seconds,nullptr,context,error,&execution)
                && requested.size() == 2 && context.geometryMaterialFrames.size() == 2
                && context.geometryMaterialFrames.at({91,0}).nativeFrame != context.geometryMaterialFrames.at({92,0}).nativeFrame,
                "independent Frame leases survive reverse seeks and preview/export without source aliasing");
        }
    auto temporalPlan = plan;
    temporalPlan.nodeIds.push_back(93); temporalPlan.nodeKinds.push_back("visual.frame-delay");
    visualtemporaloperation::Payload payload; payload.mode = visualtemporaloperation::Mode::frameDelay;
    payload.historyLength = 2; payload.decay = 0; payload.zoom = 1;
    temporalPlan.operations.push_back({93,"visual.frame-delay","native-gpu",visualtemporaloperation::serialize(payload)});
    temporalPlan.ports.push_back({93,0,1,"in","frame","image","rgba8","sRGB"});
    temporalPlan.ports.push_back({93,1,1,"out","frame","image","rgba8","sRGB"});
    temporalPlan.edges[0] = {93,1,71,12}; temporalPlan.edges.push_back({91,0,93,0});
    schedule = materialframe::admit(temporalPlan,71,{{93,1},{92,0}},[](auto&,auto&) { return false; },error);
    check(schedule.has_value(),"Frame Delay participates in a multi-source dependency DAG");
    if (schedule) {
        execution.geometryFrameSchedule = schedule; requested.clear();
        check(prepareGeometryMaterialFrame(renderer,temporalPlan,layer,videohelper::geometry::PlanUse::preview,
            1.0,nullptr,context,error,&execution)
            && requested == std::vector<std::pair<int,std::int64_t>>{{91,28},{92,30}},
            "Frame Delay requests its source at the exact prior rational sample, independently of the other endpoint");
        auto resetPlan = temporalPlan;
        auto resetPayload = payload; resetPayload.reset = 1;
        resetPlan.operations.back().payloadXml = visualtemporaloperation::serialize(resetPayload);
        check(!materialframe::admit(resetPlan,71,{{93,1},{92,0}},[](auto&,auto&) { return false; },error),
            "Surface DAG cannot silently ignore an authored shared temporal reset");
        auto resetHistory = std::make_shared<videorender::ParticleHistorySource>();
        resetHistory->parameterAt = [](const auto& destination,double at,double& value,std::string&) {
            if (destination.find("/visual93/reset") != std::string::npos && at > .97 && at < .99) value = 1;
            return true;
        };
        layer.particleHistory = resetHistory;
        check(!prepareGeometryMaterialFrame(renderer,temporalPlan,layer,videohelper::geometry::PlanUse::preview,
            1.0,nullptr,context,error,&execution) && context.geometryMaterialFrames.empty(),
            "Surface DAG rejects a transient shared reset edge even when the current reset value is neutral");
        layer.particleHistory.reset();
        auto cyclic = temporalPlan; cyclic.edges.back() = {93,1,93,0};
        check(!materialframe::admit(cyclic,71,{{93,1},{92,0}},[](auto&,auto&) { return false; },error),
            "temporal Frame cycles cannot bypass dependency admission");
        payload.mode = visualtemporaloperation::Mode::feedback; payload.decay = 0.92f; payload.zoom = 0.99f;
        temporalPlan.operations.back().kind = temporalPlan.nodeKinds.back() = "visual.feedback";
        temporalPlan.operations.back().payloadXml = visualtemporaloperation::serialize(payload);
        execution.geometryFrameSchedule = materialframe::admit(temporalPlan,71,{{93,1},{92,0}},[](auto&,auto&) { return false; },error);
        requested.clear();
        check(!prepareGeometryMaterialFrame(renderer,temporalPlan,layer,videohelper::geometry::PlanUse::exportRender,
            20.0,nullptr,context,error,&execution) && context.geometryMaterialFrames.empty(),
            "long feedback replay rejects its budget before publishing partial material images");
    }
    colortransform::AdmissionFailure failure;
    for (auto transfer : {colortransform::TransferFunction::Rec709,colortransform::TransferFunction::Gamma22})
        check(decodedframecolor::toMaterialSrgb({colortransform::ColorSpace::Rec709,transfer},{32,32},failure).has_value(),
            "declared SDR decoder transfer uses the shared native colour transform admission");
    for (auto transfer : {colortransform::TransferFunction::PQ,colortransform::TransferFunction::HLG,
                          colortransform::TransferFunction::Unspecified})
        check(!decodedframecolor::toMaterialSrgb({colortransform::ColorSpace::Rec2020,transfer},{32,32},failure),
            "HDR and undeclared decoder transfers cannot silently enter the SDR texture path");
}

int main()
{
    rawexportchecks::verify(check);
    check(videowire::isAdmittedVertexModifierKind("visual.vertex.imported-position")
            && !videowire::isAdmittedVertexModifierKind("visual.vertex.future-position"),
        "snapshot kind admission covers the exact native vertex IR without accepting unknown extensions");
    videowire::SurfaceGraphOperationContract surfaceInputContract;
    check(videowire::isAdmittedSurfaceProgramKind("visual.surface.input.tex-coord-0")
            && videowire::surfaceGraphOperationContract(
                "visual.surface.input.tex-coord-0", surfaceInputContract)
            && surfaceInputContract.kind == surfacematerial::OperationKind::Input
            && surfaceInputContract.semantic == surfacematerial::InputSemantic::TexCoord0
            && !videowire::isAdmittedSurfaceProgramKind("visual.surface.input.future-coordinate"),
        "producer surface input slugs map to the immutable native material IR and unknown inputs reject");
    testMaterialFramePrerequisite();
    testReactiveMaterialFramePrerequisite();
    testMaterialFrameDagAndTime();
    testMaterialFramePrerequisite(true);
    testMaterialFramePrerequisite(false, true);
    testDiffractionRuntimeParameters();
    testMaterialFieldBinding();
    testAuthoredImportedStarterTopologies();
    using videohelper::importedscene::NativeImportedSceneRenderUse;
    using videohelper::importedscene::VisualImportedScenePreparation;
    using videohelper::importedscene::prepareVisualImportedSceneLayer;
    using videohelper::importedscene::prepareVisualImportedSceneLayerAtTime;

    const auto legacyRequest = renderRequest();
    {
        auto producerPlan = importedScenePlan();
        producerPlan.ports.insert(producerPlan.ports.end(), {
            { 71, 2, 1, "in", "control", "material", "unspecified", "unspecified" },
            { 71, 3, 1, "in", "control", "mesh", "unspecified", "unspecified" },
            { 71, 4, 1, "out", "frame", "depth", "r32f", "unspecified" },
            { 71, 5, 1, "in", "control", "camera", "unspecified", "unspecified" },
            { 71, 6, 1, "in", "control", "light", "unspecified", "unspecified" },
            { 71, 7, 1, "out", "frame", "normal", "rgba16f", "unspecified" },
            { 71, 8, 1, "out", "frame", "emission", "rgba16f", "linearSRGB" },
            { 71, 9, 1, "out", "frame", "mask", "r8", "unspecified" },
            { 71, 10, 1, "out", "frame", "materialId", "r32uint", "unspecified" },
            { 71, 11, 1, "out", "frame", "objectId", "r32uint", "unspecified" },
            { 71, 12, 2, "out", "frame", "motionVectors", "rg16f", "unspecified" }
        });
        videowire::VisualLayerExecution compiled;
        std::string error;
        check(videowire::compileVisualLayerExecution(producerPlan, compiled, error),
            "producer-compatible 13-port Render 3D contract is admitted: " + error);

        auto malformed = producerPlan;
        auto outputPlan = producerPlan;
        outputPlan.operations.push_back({ 72, "video.out", "native-gpu", "<NodeParams/>", "" });
        outputPlan.nodeIds.push_back(72);
        outputPlan.nodeKinds.push_back("video.out");
        outputPlan.ports.push_back({72,0,1,"in","frame","image","rgba8","sRGB"});
        outputPlan.edges.push_back({71,1,72,0});
        check(videowire::compileVisualLayerExecution(outputPlan, compiled, error),
            "producer Render 3D may retain its exact outer VideoOut sink: " + error);
        malformed = outputPlan;
        malformed.edges.back().fromPort = 4;
        check(!videowire::compileVisualLayerExecution(malformed, compiled, error),
            "outer VideoOut rejects a non-image Render 3D source port");
        malformed = outputPlan;
        malformed.edges.push_back({71,1,72,0});
        check(!videowire::compileVisualLayerExecution(malformed, compiled, error),
            "outer VideoOut rejects duplicate fan-in");
        malformed = outputPlan;
        malformed.operations.back().payloadXml = "<NodeParams future=\"1\"/>";
        check(!videowire::compileVisualLayerExecution(malformed, compiled, error),
            "outer VideoOut rejects unknown parameter payloads");

        malformed = producerPlan;
        malformed.ports.back().channels = 1;
        check(!videowire::compileVisualLayerExecution(malformed, compiled, error),
            "producer Render 3D motion output rejects changed channel count");
        malformed = producerPlan;
        malformed.ports.back().pixelFormat = "rgba16f";
        check(!videowire::compileVisualLayerExecution(malformed, compiled, error),
            "producer Render 3D motion output rejects changed data format");
        malformed = producerPlan;
        malformed.ports.back().port = 13;
        check(!videowire::compileVisualLayerExecution(malformed, compiled, error),
            "producer Render 3D motion output rejects changed port identity");

        auto legacyRawPassPlan = producerPlan;
        legacyRawPassPlan.ports.pop_back();
        check(videowire::compileVisualLayerExecution(legacyRawPassPlan, compiled, error),
            "producer-compatible legacy 12-port Render 3D raw-pass contract is admitted: " + error);
        malformed = legacyRawPassPlan;
        malformed.ports.back().colorSpace = "linearSRGB";
        check(!videowire::compileVisualLayerExecution(malformed, compiled, error),
            "legacy Render 3D raw object-ID output rejects changed color space");
    }
    {
        arbitgpu::NativeSceneMotionHistory history;
        arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
        check(!history.predecessor(0, 60, 1, 640, 360), "first motion sample is reset");
        history.commit(0, 60, 1, 640, 360, inputs);
        inputs.objectTranslationOffset[0] = 0.25f;
        check(history.predecessor(1, 60, 1, 640, 360).has_value(), "adjacent motion sample retains the predecessor");
        history.commit(1, 60, 1, 640, 360, inputs);
        check(history.predecessor(1, 60, 1, 640, 360)->objectTranslationOffset[0] == 0,
            "same-frame export keeps the original predecessor");
        check(!history.predecessor(10, 60, 1, 640, 360)
            && !history.predecessor(0, 60, 1, 640, 360)
            && !history.predecessor(2, 30, 1, 640, 360)
            && !history.predecessor(2, 60, 1, 1280, 720),
            "seek, loop, rate and extent changes reset motion");
        history.commit(10, 60, 1, 640, 360, inputs);
        check(!history.predecessor(10, 60, 1, 640, 360), "repeated seek frame remains reset");
    }
    {
        auto plan = importedScenePlan();
        plan.ports.push_back({ 71, 2, 1, "in", "control", "material", "unspecified", "unspecified" });
        plan.ports.push_back({ 71, 3, 1, "in", "control", "mesh", "unspecified", "unspecified" });
        plan.ports.push_back({ 71, 4, 1, "out", "frame", "depth", "r32f", "unspecified" });
        plan.ports.push_back({ 71, 5, 1, "in", "control", "camera", "unspecified", "unspecified" });
        plan.ports.push_back({ 71, 6, 1, "in", "control", "light", "unspecified", "unspecified" });
        const std::array<const char*, 7> types { "image", "depth", "normal", "emission", "mask", "materialId", "objectId" };
        const std::array<const char*, 7> formats { "rgba8", "r32f", "rgba16f", "rgba16f", "r8", "r32uint", "r32uint" };
        for (std::size_t i = 0; i < 7; ++i)
        {
            const auto space = i == 0 ? "sRGB" : i == 3 ? "linearSRGB" : "unspecified";
            const auto renderPort = renderpasscomposite::renderPort(renderpasscomposite::kInputs[i]);
            if (i >= 2) plan.ports.push_back({ 71, renderPort, 1, "out", "frame", types[i], formats[i], space });
            plan.ports.push_back({ 90, static_cast<int>(i), 1, "in", "frame", types[i], formats[i], space });
            plan.edges.push_back({ 71, renderPort, 90, static_cast<int>(i) });
        }
        plan.ports.push_back({ 90, 7, 1, "out", "frame", "image", "rgba8", "sRGB" });
        plan.nodeIds.push_back(90); plan.nodeKinds.push_back(renderpasscomposite::kNodeKind);
        renderpasscomposite::Parameters parameters;
        parameters.fogColor = { 0.2f, 0.4f, 0.6f };
        plan.operations.push_back({ 90, renderpasscomposite::kNodeKind, "native-gpu", renderpasscomposite::encode(parameters) });
        videowire::VisualLayerExecution compiled;
        std::string error;
        check(videowire::compileVisualLayerExecution(plan, compiled, error) && compiled.passComposite.has_value(),
            "raw Render 3D fan-out compiles with one immutable composite consumer: " + error);
        FakeExecution native;
        native.nextFrame->rawPasses = true;
        FakeLayer layer;
        std::vector<FakeReceipt> owners;
        for (const auto use : { NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export })
            check(prepareVisualImportedSceneLayer({ plan }, 7, 640, 360, use, &native, layer, owners, error)
                == VisualImportedScenePreparation::rendered && native.lastRequest.runtimeInputs.passComposite
                && renderpasscomposite::encode(*native.lastRequest.runtimeInputs.passComposite) == renderpasscomposite::encode(parameters),
                "preview and export forward identical raw-pass composite parameters: " + error);
        native.nextFrame->rawPasses = false;
        check(prepareVisualImportedSceneLayer({ plan }, 7, 640, 360, NativeImportedSceneRenderUse::Preview,
            &native, layer, owners, error) == VisualImportedScenePreparation::rejected,
            "a flattened color frame cannot satisfy raw AOV publication");
        auto corrupted = plan;
        for (auto& port : corrupted.ports)
            if (port.nodeId == 71 && port.port == 10) port.pixelFormat = "rgba8";
        check(!videowire::compileVisualLayerExecution(corrupted, compiled, error),
            "integer ID attachments reject palette-image formats");
        corrupted = plan;
        corrupted.edges.back().fromNodeId = 70;
        check(!videowire::compileVisualLayerExecution(corrupted, compiled, error),
            "composite inputs must share the exact scene render identity");
        auto motionPlan = plan;
        motionPlan.ports.push_back({71, 12, 2, "out", "frame", "motionVectors", "rg16f", "unspecified"});
        motionPlan.ports.push_back({90, 8, 2, "in", "frame", "motionVectors", "rg16f", "unspecified"});
        motionPlan.edges.push_back({71, 12, 90, 8});
        parameters.mode = renderpasscomposite::Mode::MotionView;
        motionPlan.operations.back().payloadXml = renderpasscomposite::encode(parameters);
        check(videowire::compileVisualLayerExecution(motionPlan, compiled, error)
            && compiled.passComposite->mode == renderpasscomposite::Mode::MotionView,
            "motion inspection admits exact RG16F fan-out from the same render: " + error);
        native.nextFrame->rawPasses = true;
        videohelper::importedscene::VisualImportedScenePlanCache motionCache;
        for (const auto use : { NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export })
            check(prepareVisualImportedSceneLayerAtTime({motionPlan}, 7, 640, 360, 0.5, 60, use,
                &native, &motionCache, layer, owners, error) == VisualImportedScenePreparation::rendered
                && native.lastRequest.frame.frame == 30,
                "static motion preview/export receive the same rational project sample: " + error);
        {
            auto branched = motionPlan;
            const auto program = renderpassfixture::hdrBranches();
            auto oversized = program;
            oversized.count = renderpasscomposite::kMaximumPasses + 1;
            check(!renderpasscomposite::valid(oversized), "pass programs enforce their fixed GPU register capacity");
            for (int id = 90; id <= 93; ++id)
            {
                if (id != 90)
                {
                    branched.nodeIds.push_back(id);
                    branched.nodeKinds.push_back(renderpasscomposite::kNodeKind);
                    for (const auto& port : motionPlan.ports)
                        if (port.nodeId == 90)
                        { auto copy = port; copy.nodeId = id; branched.ports.push_back(copy); }
                    branched.operations.push_back({id, renderpasscomposite::kNodeKind, "native-gpu", {}});
                }
                branched.ports.push_back({id, 9, 1, "out", "frame", "image", "rgba32f", "linearSRGB"});
                branched.ports.push_back({id, 10, 1, "in", "frame", "image", "rgba32f", "linearSRGB"});
                branched.ports.push_back({id, 11, 1, "in", "frame", "image", "rgba32f", "linearSRGB"});
                const auto& step = program.steps[static_cast<std::size_t>(id - 90)];
                for (auto& operation : branched.operations)
                    if (operation.nodeId == id) operation.payloadXml = renderpasscomposite::encode(step.parameters);
                if (id == 91)
                    for (const auto& edge : motionPlan.edges)
                        if (edge.toNodeId == 90)
                        { auto copy = edge; copy.toNodeId = id; branched.edges.push_back(copy); }
                if (id >= 92)
                {
                    branched.edges.push_back({90 + step.inputA, 9, id, 10});
                    branched.edges.push_back({90 + step.inputB, 9, id, 11});
                }
            }
            branched.nodeIds.push_back(94); branched.nodeKinds.push_back("video.out");
            branched.operations.push_back({94, "video.out", "native-gpu", {}});
            branched.ports.push_back({94, 0, 1, "in", "frame", "image", "rgba8", "sRGB"});
            branched.edges.push_back({93, 7, 94, 0});
            check(videowire::compileVisualLayerExecution(branched, compiled, error)
                && compiled.passProgram && compiled.passProgram->count == 4,
                "raw AOV consumers share deterministic linear HDR branch execution: " + error);
            const auto expected = renderpasscomposite::gpuProgram(program);
            native.nextFrame->rawPasses = true;
            for (const auto use : {NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export})
            {
                check(prepareVisualImportedSceneLayer({branched}, 7, 640, 360, use,
                    &native, layer, owners, error) == VisualImportedScenePreparation::rendered
                    && native.lastRequest.runtimeInputs.passProgram.has_value(),
                    "preview and export carry the same retained branch program: " + error);
                if (native.lastRequest.runtimeInputs.passProgram)
                {
                    const auto actual = renderpasscomposite::gpuProgram(*native.lastRequest.runtimeInputs.passProgram);
                    check(actual.control == expected.control && actual.modes == expected.modes
                        && actual.colors == expected.colors && actual.references == expected.references
                        && actual.transforms == expected.transforms,
                        "branch wiring, HDR parameters and output transform survive production transport");
                }
            }
            auto reordered = branched;
            std::reverse(reordered.operations.end() - 5, reordered.operations.end() - 1);
            check(videowire::compileVisualLayerExecution(reordered, compiled, error)
                && compiled.passProgram && renderpasscomposite::gpuProgram(*compiled.passProgram).references == expected.references,
                "branch scheduling is independent of serialized pass order: " + error);
            auto invalid = branched;
            invalid.edges.back().fromPort = 9;
            check(!videowire::compileVisualLayerExecution(invalid, compiled, error)
                && error.find("HDR output encoding") != std::string::npos,
                "raw linear HDR is rejected as an SDR file output with an explicit reason");
            invalid = branched;
            for (auto& port : invalid.ports)
                if (port.nodeId == 92 && port.port == 10) port.colorSpace = "sRGB";
            check(!videowire::compileVisualLayerExecution(invalid, compiled, error),
                "linear branch inputs cannot silently reinterpret sRGB values");
            invalid = branched;
            for (auto& edge : invalid.edges)
                if (edge.toNodeId == 92 && edge.toPort == 10) edge.fromNodeId = 93;
            check(!videowire::compileVisualLayerExecution(invalid, compiled, error),
                "feedback cycles are rejected before the native pass program executes");
        }
        motionPlan.ports.back().pixelFormat = "rgba8";
        check(!videowire::compileVisualLayerExecution(motionPlan, compiled, error),
            "motion never admits display-color pixels as raw vectors");
    }
    for (const auto output : { renderpassoutput::Output::Color, renderpassoutput::Output::Depth,
                              renderpassoutput::Output::Normal, renderpassoutput::Output::Motion,
                              renderpassoutput::Output::Emission, renderpassoutput::Output::Mask,
                              renderpassoutput::Output::MaterialId, renderpassoutput::Output::ObjectId })
    {
        auto request = legacyRequest;
        request.imageOutput = output;
        request.rawExportMask = 1u << static_cast<unsigned>(output);
        const auto encoded = visualimportedscenerender::encode(request);
        visualimportedscenerender::Request decoded;
        check(!encoded.empty() && visualimportedscenerender::decode(encoded, decoded)
                  && visualimportedscenerender::sameRequest(request, decoded),
              "every supported image output retains the exact imported request through transport");
        auto outputPlan = importedScenePlan();
        for (auto& operation : outputPlan.operations)
            if (operation.kind == visualimportedscenerender::kRenderNodeKind)
                operation.payloadXml = encoded;
        FakeExecution outputExecution;
        FakeLayer outputLayer;
        std::vector<FakeReceipt> outputOwners;
        std::string outputError;
        for (const auto use : { NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export })
        {
            check(prepareVisualImportedSceneLayer(
                      { outputPlan }, 7, 640, 360, use, &outputExecution,
                      outputLayer, outputOwners, outputError) == VisualImportedScenePreparation::rendered
                      && outputExecution.lastRequest.runtimeInputs.imageOutput == output,
                  "preview and export pass the same output to the native draw and image compositor");
            const auto expectedMask = use == NativeImportedSceneRenderUse::Export ? request.rawExportMask : 0u;
            check(outputExecution.lastRequest.runtimeInputs.rawExportMask == expectedMask
                && outputLayer.rawExportMask == expectedMask && outputLayer.rawExportClipId == 7
                && outputLayer.rawExportRenderId == request.renderStableId
                && !outputOwners.empty() && outputLayer.rawExportFrame == outputOwners.back().frame,
                "only export receives persisted pass selections and retains the exact frame and clip identities");
        }
        for (const auto header : { visualimportedscenerender::kCompiledOperationHeaderV9,
                                  visualimportedscenerender::kCompiledOperationHeaderV11 })
            check(!visualimportedscenerender::decode(std::string(header) + "\n2\n" + encoded, decoded),
                "out-of-order or repeated raw selection envelopes cannot recurse");
        request.rawExportMask = 256;
        check(visualimportedscenerender::encode(request).empty(), "unknown raw output selection bits fail closed");
        if (output != renderpassoutput::Output::Color)
        {
            const auto nested = std::string(visualimportedscenerender::kCompiledOperationHeaderV9)
                + "\n2\n" + encoded;
            check(!visualimportedscenerender::decode(nested, decoded), "nested output envelopes are rejected");
        }
    }
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
    {
        auto request = renderRequest();
        request.light = boundLight;
        request.light->kind = HarmonicMIDI::grid::SceneLightKind::Directional;
        request.light->id.value = 74;
        request.noteLight = visualimportedscenerender::NoteLight { 9, 2, 60 };
        const auto payload = visualimportedscenerender::encode(request);
        visualimportedscenerender::Request decoded;
        check(!payload.empty() && visualimportedscenerender::decode(payload, decoded)
            && visualimportedscenerender::sameRequest(request, decoded),
            "note-light controls have an exact V10 round trip");
        const auto nested = std::string(visualimportedscenerender::kCompiledOperationHeaderV10)
            + "\n9 1073741824 1114636288\n" + payload;
        check(!visualimportedscenerender::decode(nested, decoded),
            "note-light envelopes cannot recursively nest");
        auto invalid = request;
        invalid.noteLight->track = 65536;
        check(visualimportedscenerender::encode(invalid).empty(), "note-light track capacity is explicit");
        invalid = request; invalid.sceneSnapshot = composedScene();
        check(visualimportedscenerender::encode(invalid).empty(),
            "unsupported composed-scene note light is rejected before execution");

        auto plan = importedScenePlan();
        plan.nodeKinds.insert(plan.nodeKinds.end() - 1,
            { "visual.3d.transform", "visual.3d.light.directional" });
        plan.nodeIds.insert(plan.nodeIds.end() - 1, { 72, 73 });
        plan.operations.insert(plan.operations.end() - 1, {
            { 72, "visual.3d.transform", "control-eval", "" },
            { 73, "visual.3d.light.directional", "control-eval", "" } });
        plan.operations.back().payloadXml = payload;
        plan.ports.insert(plan.ports.end(), {
            { 72, 0, 1, "out", "control", "transform3D", "unspecified", "unspecified" },
            { 73, 0, 1, "in", "control", "transform3D", "unspecified", "unspecified" },
            { 73, 1, 1, "out", "control", "light", "unspecified", "unspecified" },
            { 71, 2, 1, "in", "control", "material", "unspecified", "unspecified" },
            { 71, 3, 1, "in", "control", "mesh", "unspecified", "unspecified" },
            { 71, 4, 1, "out", "frame", "depth", "r32f", "unspecified" },
            { 71, 5, 1, "in", "control", "camera", "unspecified", "unspecified" },
            { 71, 6, 1, "in", "control", "light", "unspecified", "unspecified" } });
        plan.edges.insert(plan.edges.end(), { { 72, 0, 73, 0 }, { 73, 1, 71, 6 } });
        auto score = std::make_shared<arbitmod::Score>();
        score->scoreRevision = 1;
        arbitmod::Note note;
        note.id = -101; note.midiNote = 84; note.velocity = 63.5f;
        note.freqHz = 1046.5f; note.startBeat = 0; note.lengthBeats = 4; note.trackId = 9;
        score->notes.push_back(note);
        canonicalblockc::FrameKey key;
        key.projectGeneration = key.sourceGeneration = key.helperGeneration = 1;
        key.backendGeneration = key.deviceGeneration = key.scoreGeneration = 1;
        key.beatMapGeneration = key.fpsGeneration = key.loopGeneration = key.seekGeneration = 1;
        key.fps = 60;
        canonicalblockc::FrameProducer producer;
        const auto notes = producer.evaluate(key, score, 0.0f);
        FakeExecution lightExecution;
        FakeLayer lightLayer;
        std::vector<FakeReceipt> lightOwners;
        std::string lightError;
        for (const auto use : { NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export })
        {
            lightLayer.canonicalBlockCFrame = notes;
            check(prepareVisualImportedSceneLayer({ plan }, 7, 640, 360, use,
                    &lightExecution, lightLayer, lightOwners, lightError) == VisualImportedScenePreparation::rendered,
                "note light reaches preview and export: " + lightError);
            check(lightExecution.lastRequest.light
                && std::abs(lightExecution.lastRequest.light->intensity - 8.0f) < 0.001f
                && lightExecution.lastRequest.light->transform.rotation.y > 0.3f,
                "sounding note velocity and pitch change native light intensity and direction");
        }
        lightLayer.canonicalBlockCFrame.reset();
        check(prepareVisualImportedSceneLayer({ plan }, 7, 640, 360,
                NativeImportedSceneRenderUse::Preview, &lightExecution, lightLayer, lightOwners, lightError)
                == VisualImportedScenePreparation::rejected
                && lightError.find("canonical score frame") != std::string::npos,
            "missing note-light score input fails before native submission");
        auto otherTrack = *request.light;
        check(videohelper::applyNoteReactiveLight(otherTrack, { 8, 2, 60 }, notes)
                && otherTrack.intensity == 0, "unmatched note track produces no direct light");
        auto replay = *request.light;
        check(videohelper::applyNoteReactiveLight(replay, *request.noteLight, notes)
                && lightExecution.lastRequest.light
                && visualimportedscenerender::detail::sameLight(replay, *lightExecution.lastRequest.light),
            "note-light replay depends only on the canonical frame and authored controls");

        auto rig = plan;
        rig.nodeIds.insert(rig.nodeIds.end(), { 90, 91, 92 });
        rig.nodeKinds.insert(rig.nodeKinds.end(), { "visual.particles", "video.blend", "video.out" });
        rig.operations.insert(rig.operations.end(), {
            { 90, "visual.particles", "native-gpu", "<NodeParams motionMode=\"1\" spawnTrack=\"9\" alpha=\"0.4\"/>" },
            { 91, "video.blend", "native-gpu", "<NodeParams opacity=\"1\" mode=\"0\"/>" },
            { 92, "video.out", "native-gpu", "" } });
        rig.ports.insert(rig.ports.end(), {
            { 90, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
            { 91, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
            { 91, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
            { 91, 2, 1, "in", "frame", "image", "rgba8", "sRGB" },
            { 91, 3, 1, "in", "frame", "image", "rgba8", "sRGB" },
            { 92, 0, 1, "in", "frame", "image", "rgba8", "sRGB" } });
        rig.edges.insert(rig.edges.end(), { { 71, 1, 91, 0 }, { 90, 1, 91, 2 }, { 91, 1, 92, 0 } });
        videowire::VisualLayerExecution rigExecution;
        check(videowire::compileVisualLayerExecution(rig, rigExecution, lightError)
            && rigExecution.importedParticleOverlay && rigExecution.particles
            && rigExecution.particleNodeId == 90 && rigExecution.importedSceneRender->noteLight
            && rigExecution.particleParameters.spawnTrack == 9,
            "note light and particles admit one native overlay schedule: " + lightError);
        check(videowire::isImportedParticleOverlayPlan({ rig }, 7),
            "viewport and export select scene preparation for the particle rig");
        rig.ports.insert(rig.ports.end(), {
            { 90, 0, 1, "in", "event", "unspecified", "unspecified", "unspecified" },
            { 90, 2, 1, "in", "control", "points3D", "unspecified", "unspecified" },
            { 90, 3, 1, "in", "control", "points3D", "unspecified", "unspecified" } });
        check(videowire::compileVisualLayerExecution(rig, rigExecution, lightError)
            && rigExecution.importedParticleOverlay && rigExecution.particleParameters.geometryCount == 0,
            "new unconnected geometry ports preserve the existing note particle and light rig");
        auto invalidGeometryPort = rig;
        invalidGeometryPort.ports.back().dataType = "geometry3D";
        check(!videowire::compileVisualLayerExecution(invalidGeometryPort, rigExecution, lightError),
            "particle overlay rejects a forged optional geometry descriptor");
        for (const auto use : { NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export })
        {
            lightLayer.canonicalBlockCFrame = notes;
            lightLayer.shaderClock = { 1.25, 30 };
            check(prepareVisualImportedSceneLayer({ rig }, 7, 640, 360, use,
                    &lightExecution, lightLayer, lightOwners, lightError) == VisualImportedScenePreparation::rendered,
                "particle rig prepares the same native scene in preview and export: " + lightError);
            check(lightLayer.texture == 44 && lightLayer.nativeTextureView == 44
                    && lightLayer.canonicalBlockCFrame == notes
                    && lightLayer.shaderClock.frame == 30 && lightLayer.shaderClock.time == 1.25,
                "scene preparation retains particle clock, canonical note identity and native background");
        }
        for (const auto wrongPort : { 0, 3 })
        {
            auto invalidRig = rig;
            invalidRig.edges[invalidRig.edges.size() - 2].toPort = wrongPort;
            check(!videowire::compileVisualLayerExecution(invalidRig, rigExecution, lightError),
                "particle rig rejects an aliased or swapped compositor input");
        }
        auto invalidRig = rig;
        invalidRig.operations[invalidRig.operations.size() - 3].payloadXml = "<NodeParams motionMode=\"0\"/>";
        check(!videowire::compileVisualLayerExecution(invalidRig, rigExecution, lightError)
                && lightError.find("deterministic timeline motion mode") != std::string::npos,
            "particle rig rejects stateful legacy particles rather than claiming repeatable export");
    }
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
    for (const auto& extentPlan : { importedScenePlan(), composedScenePlan() })
    {
        const std::vector<videowire::CompiledVisualLayerPlan> extentPlans { extentPlan };
        const auto originalIdentity = videohelper::importedscene::exactImportedScenePayloadIdentity(extentPlan);
        videohelper::importedscene::VisualImportedScenePlanCache extentCache;
        FakeExecution extentExecution;
        for (const auto& dimensions : { std::pair<int, int>{3840, 2160}, {320, 180} })
        {
            extentExecution.nextFrame->frameWidth = static_cast<std::uint32_t>(dimensions.first);
            extentExecution.nextFrame->descriptorWidth = extentExecution.nextFrame->frameWidth;
            extentExecution.nextFrame->frameHeight = static_cast<std::uint32_t>(dimensions.second);
            FakeLayer layer;
            std::vector<FakeReceipt> owners;
            error.clear();
            check(prepareVisualImportedSceneLayerAtTime(extentPlans, extentPlan.clipId,
                      dimensions.first, dimensions.second, 0.5, 30.0,
                      NativeImportedSceneRenderUse::Export, &extentExecution, &extentCache,
                      layer, owners, error) == VisualImportedScenePreparation::rendered
                      && extentExecution.lastRequest.dimensions.width == static_cast<std::uint32_t>(dimensions.first)
                      && extentExecution.lastRequest.dimensions.height == static_cast<std::uint32_t>(dimensions.second)
                      && layer.texWidth == dimensions.first && layer.texHeight == dimensions.second
                      && videohelper::importedscene::exactImportedScenePayloadIdentity(extentPlans.front())
                          == originalIdentity,
                  "project and diagnostic extents reuse exact scene payload identity with newly sized native frames: " + error);
        }
        // Matching byte counts alone would not distinguish these swapped extents.
        extentExecution.nextFrame->frameWidth = 180;
        extentExecution.nextFrame->descriptorWidth = 180;
        extentExecution.nextFrame->frameHeight = 320;
        FakeLayer wrongExtentLayer;
        std::vector<FakeReceipt> wrongExtentOwners;
        error.clear();
        check(prepareVisualImportedSceneLayerAtTime(extentPlans, extentPlan.clipId,
                  320, 180, 0.5, 30.0, NativeImportedSceneRenderUse::Export,
                  &extentExecution, &extentCache, wrongExtentLayer, wrongExtentOwners, error)
                  == VisualImportedScenePreparation::rejected
                  && error == "imported scene native frame dimensions do not match the declared render extent",
              "diagnostic rendering cannot relabel an equal-byte-count native frame of the wrong shape");
    }
    for (const auto& hdrPlan : { importedScenePlan(), composedScenePlan() })
        for (const auto* api : { "opengl", "metal" })
        {
            FakeExecution hdrExecution;
            hdrExecution.nextFrame->api = api;
            hdrExecution.nextFrame->linearColor = true;
            const std::vector<videowire::CompiledVisualLayerPlan> hdrPlans {hdrPlan};
            for (const auto use : {NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export})
            {
                // Fresh cache models reopen; the persisted plan stays identical,
                // while the selected export profile chooses native precision.
                videohelper::importedscene::VisualImportedScenePlanCache reopened;
                FakeLayer hdrLayer;
                std::vector<FakeReceipt> owners;
                check(prepareVisualImportedSceneLayerAtTime(hdrPlans, hdrPlan.clipId, 640, 360, 0, 24,
                    use, &hdrExecution, &reopened, hdrLayer, owners, error, nullptr, {}, nullptr, true)
                        == VisualImportedScenePreparation::rendered
                    && hdrExecution.lastRequest.runtimeInputs.linearColor
                    && arbitgpu::isLinearSceneColor(hdrLayer.nativeTextureDescriptor),
                    "imported/generated reopened plans retain explicit linear native preview/export color: " + error);
                hdrExecution.nextFrame->linearColor = false;
                check(prepareVisualImportedSceneLayerAtTime(hdrPlans, hdrPlan.clipId, 640, 360, 0, 24,
                    use, &hdrExecution, &reopened, hdrLayer, owners, error, nullptr, {}, nullptr, true)
                        == VisualImportedScenePreparation::rejected,
                    "HDR scene preparation rejects an SDR attachment from the native executor");
                hdrExecution.nextFrame->linearColor = true;
            }
        }
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

    {
        auto snapshot=geometryCompositionFixture(error);
        check(snapshot!=nullptr,"procedural mesh and instance composition produces valid scene records");
        auto retainedPlan=composedScenePlan();
        retainedPlan.nodeIds={6,11};
        retainedPlan.nodeKinds={videowire::geometry::kRetainedSceneOperation,"visual.3d.render"};
        visualimportedscenerender::Request request;
        request.sourceStableId=7; request.renderStableId=12;
        request.structuralRevision=9; request.evaluationRevision=12;
        request.sceneSnapshot=snapshot;
        retainedPlan.operations={{6,retainedPlan.nodeKinds[0],"control-eval",""},
                                 {11,retainedPlan.nodeKinds[1],"native-gpu",visualimportedscenerender::encode(request)}};
        retainedPlan.edges={{6,0,11,0}};
        retainedPlan.ports={{6,0,1,"out","control","scene3D","unspecified","unspecified"},
                            {11,0,1,"in","control","scene3D","unspecified","unspecified"},
                            {11,1,1,"out","frame","image","rgba8","sRGB"}};
        videowire::VisualLayerExecution retainedCompiled;
        check(videowire::compileVisualLayerExecution(retainedPlan, retainedCompiled, error),
              "canonical retained scene without a Frame prerequisite compiles: " + error);
        FakeExecution retainedExecution;
        videohelper::importedscene::VisualImportedScenePlanCache cache;
        FakeLayer layer;
        std::vector<FakeReceipt> owners;
        for (const auto use : {NativeImportedSceneRenderUse::Preview,NativeImportedSceneRenderUse::Export}) {
            const std::vector<videowire::CompiledVisualLayerPlan> plans{retainedPlan};
            check(prepareVisualImportedSceneLayer(plans,8,640,360,use,&retainedExecution,&cache,
                      layer,owners,error)==VisualImportedScenePreparation::rendered,
                  "retained Geometry Core scene reaches production preview and export without media");
            check(retainedExecution.lastRequest.sceneSnapshot
                      && retainedExecution.lastRequest.sceneSnapshot->objectCount==6
                      && retainedExecution.lastRequest.sceneSnapshot->vertexCount==66
                      && retainedExecution.lastRequest.asset.id.empty(),
                  "production retains independent object materials and shared instance geometry");
        }
        const auto surfaces=retainedSurfaceFixture(error);
        check(surfaces.has_value(),"retained Surface execution fixture is canonical");
        if (surfaces) {
            request.surfacePrograms=surfaces->surfacePrograms;
            for (auto& object : request.surfacePrograms) {
                auto& material=object.program.material;
                material.sceneSnapshot=snapshot;
                material.sceneRevision=material.structuralRevision=material.programRevision=9;
                material.binding.surfaceMaterialRevision=9; material.evaluationRevision=12;
            }
            auto surfacePlan=retainedPlan;
            surfacePlan.operations[1].payloadXml=visualimportedscenerender::encode(request);
            for (const auto use : {NativeImportedSceneRenderUse::Preview,NativeImportedSceneRenderUse::Export})
                check(prepareVisualImportedSceneLayer({surfacePlan},8,640,360,use,&retainedExecution,&cache,
                    layer,owners,error)==VisualImportedScenePreparation::rendered
                    && retainedExecution.lastRequest.surfacePrograms.size()==5
                    && retainedExecution.lastRequest.surfacePrograms.front().program.material.sceneSnapshot
                        ==retainedExecution.lastRequest.sceneSnapshot,
                    "production retained-scene preparation preserves exact Surface collection ownership in preview/export: "+error);
        }
        auto compositePlan = retainedPlan;
        renderpasscomposite::Parameters compositeParameters;
        compositeParameters.mode = renderpasscomposite::Mode::ObjectMatte;
        compositeParameters.identity = 0xf1234567u;
        compositePlan.nodeIds.push_back(90);
        compositePlan.nodeKinds.push_back(renderpasscomposite::kNodeKind);
        compositePlan.operations.push_back({90, renderpasscomposite::kNodeKind, "native-gpu",
            renderpasscomposite::encode(compositeParameters)});
        const std::array<const char*, 7> types {"image", "depth", "normal", "emission", "mask", "materialId", "objectId"};
        const std::array<const char*, 7> formats {"rgba8", "r32f", "rgba16f", "rgba16f", "r8", "r32uint", "r32uint"};
        for (std::size_t i = 0; i < 7; ++i)
        {
            const auto space = i == 0 ? "sRGB" : i == 3 ? "linearSRGB" : "unspecified";
            const auto port = renderpasscomposite::renderPort(renderpasscomposite::kInputs[i]);
            if (i != 0) compositePlan.ports.push_back({11, port, 1, "out", "frame", types[i], formats[i], space});
            compositePlan.ports.push_back({90, static_cast<int>(i), 1, "in", "frame", types[i], formats[i], space});
            compositePlan.edges.push_back({11, port, 90, static_cast<int>(i)});
        }
        compositePlan.ports.push_back({90, 7, 1, "out", "frame", "image", "rgba8", "sRGB"});
        compositePlan.nodeIds.push_back(91); compositePlan.nodeKinds.push_back("video.out");
        compositePlan.operations.push_back({91, "video.out", "native-gpu", ""});
        compositePlan.ports.push_back({91, 0, 1, "in", "frame", "image", "rgba8", "sRGB"});
        compositePlan.edges.push_back({90, 7, 91, 0});
        retainedExecution.nextFrame->rawPasses = true;
        for (const auto use : {NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export})
            check(prepareVisualImportedSceneLayer({compositePlan}, 8, 640, 360, use, &retainedExecution,
                &cache, layer, owners, error) == VisualImportedScenePreparation::rendered
                && retainedExecution.lastRequest.runtimeInputs.passComposite
                && retainedExecution.lastRequest.runtimeInputs.passComposite->identity == 0xf1234567u,
                "retained generated scenes preserve full-width ID matte fan-out and Video Output in preview/export: " + error);
        auto broken=retainedPlan;
        broken.edges[0].fromNodeId=5;
        check(prepareVisualImportedSceneLayer({broken},8,640,360,NativeImportedSceneRenderUse::Preview,
                  &retainedExecution,&cache,layer,owners,error)==VisualImportedScenePreparation::rejected,
              "retained scene rejects forged source identity");
        broken=retainedPlan;
        broken.operations[0].payloadXml="unexpected";
        check(prepareVisualImportedSceneLayer({broken},8,640,360,NativeImportedSceneRenderUse::Preview,
                  &retainedExecution,&cache,layer,owners,error)==VisualImportedScenePreparation::rejected,
              "retained scene rejects additional producer payload");
    }
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
    runtimeParameters["visual73/selectedMorphWeight"] = 0.7;
    runtimeParameters["visual73/boneTranslateX"] = 2.5;
    runtimeParameters["visual73/boneRotateY"] = 30.0;
    runtimeParameters["visual73/boneScaleZ"] = 1.5;
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
              && execution.lastRequest.deformation->playback.timelineSeconds == 6.5
              && execution.lastRequest.deformation->playback.speed == 2.0
              && execution.lastRequest.deformation->playback.trimStartSeconds == 0.5
              && execution.lastRequest.deformation->playback.trimEndSeconds == 1.5
              && execution.lastRequest.deformation->playback.weight == 0.25
              && execution.lastRequest.deformation->pose.morphWeight == 0.7
              && execution.lastRequest.deformation->pose.translation[0] == 2.5
              && execution.lastRequest.deformation->pose.rotationDegrees[1] == 30.0
              && execution.lastRequest.deformation->pose.scale[2] == 1.5
              && execution.lastRequest.runtimeInputs.objectTranslationOffset[0] == 0.5f
              && execution.lastRequest.runtimeInputs.objectRotationDegrees[1] == 45.0f
              && execution.lastRequest.runtimeInputs.objectScale == 1.5f
              && execution.lastRequest.runtimeInputs.cameraTranslationOffset[2] == -0.25f
              && execution.lastRequest.runtimeInputs.emissionGain == 2.0f,
          "preview receives the immutable deformation payload and exact frame time");

    for (const auto use : {NativeImportedSceneRenderUse::Preview,
                           NativeImportedSceneRenderUse::Export})
    {
        check(prepareVisualImportedSceneLayerAtTime(
                  deformationPlans, 7, 640, 360, 1.25, 24.0, use,
                  &execution, deformationLayer, deformationOwners, error,
                  &runtimeParameters) == VisualImportedScenePreparation::rendered
                  && execution.lastRequest.deformation
                  && execution.lastRequest.deformation->playback.timelineSeconds == 5.25,
              "preview and export seek to the rational frame plus authored offset without accumulation");
    }

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
    const auto exportCallsBefore = execution.exportCalls;
    const auto previewCallsBefore = execution.previewCalls;
    const auto previewFrameBefore = previewOwners.empty() ? nullptr : previewOwners.front().frame;
    check(prepareVisualImportedSceneLayer(
              plans, 7, 640, 360, NativeImportedSceneRenderUse::Export,
              &execution, exportLayer, exportOwners, error)
              == VisualImportedScenePreparation::rendered,
          "the same exact plan renders for export");
    check(execution.exportCalls == exportCallsBefore + 1
              && execution.previewCalls == previewCallsBefore
              && previewOwners.size() == 1 && previewOwners.front().owner == 1
              && previewOwners.front().frame == previewFrameBefore
              && exportOwners.size() == 1 && exportOwners.front().owner == 2
              && exportOwners.front().frame != previewFrameBefore,
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
    {
        const auto vertexPlan = importedVertexPlan();
        FakeLayer vertexLayer;
        vertexLayer.audioPresent = true;
        vertexLayer.audioFeatures.bands.assign(64, 0.0f);
        vertexLayer.audioFeatures.bands[6] = 0.8f;
        std::vector<FakeReceipt> vertexOwners;
        videohelper::importedscene::VisualImportedScenePlanCache vertexCache;
        const auto previewResult = prepareVisualImportedSceneLayerAtTime(
            {vertexPlan}, 7, 640, 360, 0.5, 60.0, NativeImportedSceneRenderUse::Preview,
            &execution, &vertexCache, vertexLayer, vertexOwners, error);
        check(previewResult == VisualImportedScenePreparation::rendered
              && execution.lastRequest.material && execution.lastRequest.material->vertexModifier
              && execution.lastRequest.runtimeInputs.vertexSpectrum[6] == 0.8f
              && execution.lastRequest.runtimeInputs.timeSeconds == 0.5f,
              "authored vertex topology and the current audio frame reach native preview: " + error);
        const auto previewRequest = execution.lastRequest;
        check(prepareVisualImportedSceneLayerAtTime(
            {vertexPlan}, 7, 640, 360, 0.5, 60.0, NativeImportedSceneRenderUse::Export,
            &execution, &vertexCache, vertexLayer, vertexOwners, error)
                == VisualImportedScenePreparation::rendered
              && execution.lastRequest.runtimeInputs.vertexSpectrum == previewRequest.runtimeInputs.vertexSpectrum
              && visualimportedscenerender::sameMaterialRequest(*execution.lastRequest.material, *previewRequest.material),
              "export retains the same vertex graph, frame time and canonical audio samples");
        vertexLayer.audioFeatures.bands.resize(63);
        check(prepareVisualImportedSceneLayerAtTime(
            {vertexPlan}, 7, 640, 360, 0.5, 60.0, NativeImportedSceneRenderUse::Preview,
            &execution, &vertexCache, vertexLayer, vertexOwners, error)
                == VisualImportedScenePreparation::rejected
              && error.find("64-band") != std::string::npos,
              "a truncated spectrum frame cannot silently change the band's meaning");
        vertexLayer.audioPresent = false;
        check(prepareVisualImportedSceneLayerAtTime(
            {vertexPlan}, 7, 640, 360, 0.5, 60.0, NativeImportedSceneRenderUse::Preview,
            &execution, &vertexCache, vertexLayer, vertexOwners, error)
                == VisualImportedScenePreparation::rendered
              && std::all_of(execution.lastRequest.runtimeInputs.vertexSpectrum.begin(),
                             execution.lastRequest.runtimeInputs.vertexSpectrum.end(),
                             [](float value) { return value == 0.0f; }),
              "missing audio keeps the imported vertex graph silent");
        auto motionVertexPlan = vertexPlan;
        for (auto& operation : motionVertexPlan.operations)
            if (operation.kind == visualimportedscenerender::kRenderNodeKind)
            {
                visualimportedscenerender::Request motionRequest;
                check(visualimportedscenerender::decode(operation.payloadXml, motionRequest),
                    "vertex motion rejection fixture decodes");
                motionRequest.imageOutput = renderpassoutput::Output::Motion;
                operation.payloadXml = visualimportedscenerender::encode(motionRequest);
            }
        for (const auto use : {NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export})
            check(prepareVisualImportedSceneLayerAtTime(
                {motionVertexPlan}, 7, 640, 360, 0.5, 60.0, use,
                &execution, &vertexCache, vertexLayer, vertexOwners, error)
                    == VisualImportedScenePreparation::rejected
                && error.find("Motion output is unavailable for graph vertex deformation") != std::string::npos,
                "Motion selection rejects vertex deformation until previous deformed positions exist");
        auto mismatched = vertexPlan;
        mismatched.edges.back().fromPort = 0;
        videowire::VisualLayerExecution rejected;
        check(!videowire::compileVisualLayerExecution(mismatched, rejected, error),
              "vertex topology rejects a material edge from the wrong typed output");
        mismatched = vertexPlan;
        for (auto& operation : mismatched.operations)
            if (operation.nodeId == 2) operation.kind = "visual.vertex.control-parameter";
        check(!videowire::compileVisualLayerExecution(mismatched, rejected, error),
              "vertex topology rejects a node kind that disagrees with its immutable IR");
    }
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

    {
        visualnoteinstancing::Mapping appearance;
        appearance.appearanceSource = visualnoteinstancing::AppearanceSource::StableNoteIdentity;
        appearance.appearanceLow = { 1.0f, 0.0f, 0.0f, 0.25f };
        appearance.appearanceHigh = { 0.0f, 0.0f, 1.0f, 2.0f };
        auto notesPlan = importedScenePlan();
        notesPlan.nodeKinds.insert(notesPlan.nodeKinds.end() - 1,
            { "visual.score.note-collection", "visual.3d.note-instanced-mesh" });
        notesPlan.nodeIds.insert(notesPlan.nodeIds.end() - 1, { 90, 91 });
        notesPlan.operations.insert(notesPlan.operations.end() - 1, {
            { 90, "visual.score.note-collection", "control-eval", "" },
            { 91, "visual.3d.note-instanced-mesh", "control-eval", visualnoteinstancing::encode(appearance) }
        });
        notesPlan.ports.insert(notesPlan.ports.end(), {
            { 90, 0, 1, "out", "control", "noteCollection", "unspecified", "unspecified" },
            { 91, 0, 1, "in", "control", "mesh", "unspecified", "unspecified" },
            { 91, 1, 1, "in", "control", "noteCollection", "unspecified", "unspecified" },
            { 91, 2, 1, "out", "control", "mesh", "unspecified", "unspecified" },
            { 71, 2, 1, "in", "control", "material", "unspecified", "unspecified" },
            { 71, 3, 1, "in", "control", "mesh", "unspecified", "unspecified" },
            { 71, 4, 1, "out", "frame", "depth", "r32f", "unspecified" },
            { 71, 5, 1, "in", "control", "camera", "unspecified", "unspecified" },
            { 71, 6, 1, "in", "control", "light", "unspecified", "unspecified" }
        });
        notesPlan.edges.insert(notesPlan.edges.end(), {
            { 70, 0, 91, 0 }, { 90, 0, 91, 1 }, { 91, 2, 71, 3 }
        });
        videowire::VisualLayerExecution notesCompiled;
        check(videowire::compileVisualLayerExecution(notesPlan, notesCompiled, error),
              "authored note appearance and current Render 3D ports compile: " + error);
        check(notesCompiled.noteInstanceMapping.has_value()
                  && visualnoteinstancing::encode(*notesCompiled.noteInstanceMapping)
                      == visualnoteinstancing::encode(appearance),
              "helper compilation preserves the exact note appearance mapping");
        auto score = std::make_shared<arbitmod::Score>();
        score->rootFreq = 440.0f;
        score->notationVersion = score->scoreRevision = 1;
        canonicalblockc::FrameKey frameKey;
        frameKey.projectGeneration = frameKey.sourceGeneration = frameKey.helperGeneration = 1;
        frameKey.backendGeneration = frameKey.deviceGeneration = frameKey.scoreGeneration = 1;
        frameKey.beatMapGeneration = frameKey.fpsGeneration = frameKey.loopGeneration = frameKey.seekGeneration = 1;
        frameKey.fps = 60.0;
        canonicalblockc::FrameProducer frameProducer;
        const auto noteFrame = frameProducer.evaluate(frameKey, score, 0.0f);
        check(canonicalblockc::valid(noteFrame) && noteFrame->noteRows() == 0,
              "an empty score produces a valid zero-row canonical frame");
        FakeExecution notesExecution;
        FakeLayer notesLayer;
        notesLayer.canonicalBlockCFrame = noteFrame;
        std::vector<FakeReceipt> notesOwners;
        for (const auto use : { NativeImportedSceneRenderUse::Preview, NativeImportedSceneRenderUse::Export })
        {
            check(prepareVisualImportedSceneLayer({ notesPlan }, 7, 640, 360, use,
                      &notesExecution, notesLayer, notesOwners, error)
                      == VisualImportedScenePreparation::rendered,
                  "note appearance reaches both preview and export: " + error);
            check(notesExecution.lastRequest.runtimeInputs.noteInstanceMapping.has_value()
                      && visualnoteinstancing::encode(*notesExecution.lastRequest.runtimeInputs.noteInstanceMapping)
                          == visualnoteinstancing::encode(appearance)
                      && notesExecution.lastRequest.runtimeInputs.canonicalBlockCFrame == noteFrame
                      && notesLayer.canonicalBlockCFrame == noteFrame,
                  "preview and export retain the same mapping and canonical frame owner");
            const auto batch = arbitgpu::prepareNativeNoteInstances(notesExecution.lastRequest.runtimeInputs);
            check(batch.admitted && batch.count == 0,
                  "the empty canonical frame renders zero note instances");
        }
        FakeLayer missingFrameLayer;
        check(prepareVisualImportedSceneLayer({ notesPlan }, 7, 640, 360,
                  NativeImportedSceneRenderUse::Preview, &notesExecution,
                  missingFrameLayer, notesOwners, error)
                  == VisualImportedScenePreparation::rejected
                  && error == "note-instanced imported scene requires one canonical Block C frame",
              "a note-instanced scene still rejects a missing canonical frame");
        notesPlan.operations[2].payloadXml += "0\n";
        check(!videowire::compileVisualLayerExecution(notesPlan, notesCompiled, error),
              "malformed note appearance is rejected before native submission");
    }

    if (failures != 0)
    {
        std::cerr << failures << " imported scene visual plan checks failed\n";
        return 1;
    }
    std::cout << "imported scene visual plan execution: all checks passed\n";
    return 0;
}
