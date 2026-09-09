#include "fixture_scene.h"
#include "../../src/gltf_glb.h"
#include "../../src/glb_scene_adapter.h"
#include "../../src/sha256.h"
#include "../../../shared/HolographicTradingCardAsset.h"
#include "../../../shared/HolographicTradingCardCompiledFixture.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace videohelper::fixture3d
{

using namespace HarmonicMIDI::grid;

Visual3DScene makeScene() noexcept
{
    Visual3DScene scene;
    scene.id = kSceneId;
    scene.activeCamera = kCameraId;
    scene.ambientColor = { 0.08f, 0.09f, 0.12f };

    const std::array<SceneVertex, 24> vertices {{
        {{-1,-1, 1}, { 0, 0, 1}, {}, {0,0}}, {{ 1,-1, 1}, { 0, 0, 1}, {}, {1,0}},
        {{ 1, 1, 1}, { 0, 0, 1}, {}, {1,1}}, {{-1, 1, 1}, { 0, 0, 1}, {}, {0,1}},
        {{ 1,-1,-1}, { 0, 0,-1}, {}, {0,0}}, {{-1,-1,-1}, { 0, 0,-1}, {}, {1,0}},
        {{-1, 1,-1}, { 0, 0,-1}, {}, {1,1}}, {{ 1, 1,-1}, { 0, 0,-1}, {}, {0,1}},
        {{-1,-1,-1}, {-1, 0, 0}, {}, {0,0}}, {{-1,-1, 1}, {-1, 0, 0}, {}, {1,0}},
        {{-1, 1, 1}, {-1, 0, 0}, {}, {1,1}}, {{-1, 1,-1}, {-1, 0, 0}, {}, {0,1}},
        {{ 1,-1, 1}, { 1, 0, 0}, {}, {0,0}}, {{ 1,-1,-1}, { 1, 0, 0}, {}, {1,0}},
        {{ 1, 1,-1}, { 1, 0, 0}, {}, {1,1}}, {{ 1, 1, 1}, { 1, 0, 0}, {}, {0,1}},
        {{-1, 1, 1}, { 0, 1, 0}, {}, {0,0}}, {{ 1, 1, 1}, { 0, 1, 0}, {}, {1,0}},
        {{ 1, 1,-1}, { 0, 1, 0}, {}, {1,1}}, {{-1, 1,-1}, { 0, 1, 0}, {}, {0,1}},
        {{-1,-1,-1}, { 0,-1, 0}, {}, {0,0}}, {{ 1,-1,-1}, { 0,-1, 0}, {}, {1,0}},
        {{ 1,-1, 1}, { 0,-1, 0}, {}, {1,1}}, {{-1,-1, 1}, { 0,-1, 0}, {}, {0,1}}
    }};
    const std::array<std::uint32_t, 36> indices {{
         0, 1, 2,  0, 2, 3,  4, 5, 6,  4, 6, 7,
         8, 9,10,  8,10,11, 12,13,14, 12,14,15,
        16,17,18, 16,18,19, 20,21,22, 20,22,23
    }};
    for (std::size_t index = 0; index < vertices.size(); ++index)
        scene.vertices[index] = vertices[index];
    for (std::size_t index = 0; index < indices.size(); ++index)
        scene.indices[index] = indices[index];
    scene.vertexCount = vertices.size();
    scene.indexCount = indices.size();

    scene.textureTexels[0] = { 255, 236, 176, 255 };
    scene.textureTexels[1] = { 48, 92, 220, 255 };
    scene.textureTexels[2] = { 28, 168, 118, 255 };
    scene.textureTexels[3] = { 224, 54, 96, 255 };
    scene.textureTexelCount = 4;
    scene.textures[0].id = kCubeTextureId;
    scene.textures[0].width = 2;
    scene.textures[0].height = 2;
    scene.textureCount = 1;

    scene.materials[0].id = kCubeMaterialId;
    scene.materials[0].baseColor = { 0.72f, 0.82f, 1.0f };
    scene.materials[0].baseColorTexture = kCubeTextureId;
    scene.materials[0].metallic = 0.15f;
    scene.materials[0].roughness = 0.4f;
    scene.materialCount = 1;

    scene.objects[0].id = kCubeObjectId;
    scene.objects[0].material = kCubeMaterialId;
    scene.objects[0].transform.rotation = { -0.16773126f, 0.25488700f,
                                            0.04494346f, 0.95125124f };
    scene.objects[0].firstVertex = 0;
    scene.objects[0].vertexCount = static_cast<std::uint32_t> (vertices.size());
    scene.objects[0].firstIndex = 0;
    scene.objects[0].indexCount = static_cast<std::uint32_t> (indices.size());
    scene.objectCount = 1;

    scene.lights[0].id = kKeyLightId;
    scene.lights[0].kind = SceneLightKind::Directional;
    scene.lights[0].transform.rotation = { -0.29045950f, -0.24684011f,
                                           -0.07782839f, 0.92121983f };
    scene.lights[0].color = { 1.0f, 0.94f, 0.84f };
    scene.lights[0].intensity = 1.25f;
    scene.lightCount = 1;

    scene.cameras[0].id = kCameraId;
    scene.cameras[0].transform.translation = { 0.0f, 0.0f, 5.0f };
    scene.cameras[0].verticalFovRadians = 0.7853981634f;
    scene.cameras[0].nearPlane = 0.1f;
    scene.cameras[0].farPlane = 100.0f;
    scene.cameraCount = 1;

    return scene;
}

std::optional<HolographicTradingCardScene>
loadHolographicTradingCardScene(std::string& error)
{
    gltf::GlbAdmissionOptions options;
    options.supportedRequiredExtensions = { "KHR_lights_punctual" };
    const auto document = gltf::decodeStaticGlb(
        holographictradingcard::kGlb.data(), holographictradingcard::kGlb.size(),
        options, error);
    if (!document)
        return std::nullopt;
    const auto adapted = gltf::adaptStaticGlbToVisual3DScene(*document, error);
    if (!adapted)
        return std::nullopt;
    if (document->scenes.size() != 1 || document->nodes.size() != 3
        || document->materials.size() != 1 || document->cameras.size() != 1
        || document->lights.size() != 1 || document->meshes.size() != 1)
    {
        error = "holographic trading card GLB lost its exact named scene records";
        return std::nullopt;
    }
    HolographicTradingCardScene result;
    result.scene = std::make_shared<const Visual3DScene>(*adapted);
    result.contentSha256 = videohelper::sha256Bytes(
        holographictradingcard::kGlb.data(), holographictradingcard::kGlb.size());
    if (result.contentSha256 != holographictradingcard::kContentSha256)
    {
        error = "holographic trading card embedded-byte digest mismatch";
        return std::nullopt;
    }
    result.assetId = result.contentSha256 == holographictradingcard::kContentSha256
        ? std::string(holographictradingcard::kAssetId) : std::string {};
    result.sceneName = document->scenes[0].name;
    result.objectName = document->nodes[0].name;
    result.materialName = document->materials[0].name;
    result.cameraName = document->cameras[0].name;
    result.lightName = document->lights[0].name;
    std::string admissionError;
    result.encodedOperation = std::string(
        holographiccardcompiledfixture::kEncodedV7Operation);
    if (!visualimportedscenerender::decodeCanonical(
            result.encodedOperation, result.operation, admissionError))
    {
        error = admissionError.empty() ? "card compiled fixture failed to decode" : admissionError;
        return std::nullopt;
    }
    const auto& operation = result.operation;
    const auto* binding = operation.diffractionMaterial
        ? &*operation.diffractionMaterial : nullptr;
    if (operation.asset.id != holographiccardcompiledfixture::kAssetId
        || operation.asset.version != holographiccardcompiledfixture::kAssetVersion
        || operation.asset.contentSha256 != holographiccardcompiledfixture::kAssetSha256
        || operation.sourceStableId != holographiccardcompiledfixture::kSourceStableId
        || operation.renderStableId != holographiccardcompiledfixture::kRenderStableId
        || binding == nullptr
        || binding->scene.value != holographiccardcompiledfixture::kSceneId
        || binding->object.value != holographiccardcompiledfixture::kObjectId
        || binding->sceneRevision != holographiccardcompiledfixture::kSceneRevision
        || binding->structuralRevision != holographiccardcompiledfixture::kStructuralRevision
        || binding->evaluationRevision != holographiccardcompiledfixture::kEvaluationRevision
        || binding->materialRevision != holographiccardcompiledfixture::kMaterialRevision
        || !binding->spatialFoil
        || binding->spatialFoil->physicalBsdf.roughness.rmsSlope != 0.08f
        || binding->spatialFoil->workBudget.maximumEvaluations
            != holographiccardcompiledfixture::kMaximumEvaluations
        || operation.asset.id != result.assetId
        || operation.asset.contentSha256 != result.contentSha256
        || binding->scene != result.scene->id
        || binding->object != result.scene->objects[0].id)
    {
        error = "card compiled fixture identity or budget mismatch";
        return std::nullopt;
    }
    return result;
}

} // namespace videohelper::fixture3d
