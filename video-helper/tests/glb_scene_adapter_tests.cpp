#include "../src/glb_scene_adapter.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace
{
using namespace videohelper::gltf;
using namespace HarmonicMIDI::grid;

int failures = 0;

void check(bool value, const char* message)
{
    if (!value)
    {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", message);
    }
}

bool near(float left, float right)
{
    return std::abs(left - right) < 0.0001f;
}

std::array<float, 16> translation(float x, float y, float z)
{
    return {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        x, y, z, 1.0f
    };
}

std::array<float, 16> transform(float angleRadians, float xScale, float yScale,
                                float zScale)
{
    const auto cosine = std::cos(angleRadians);
    const auto sine = std::sin(angleRadians);
    return {
        cosine * xScale, sine * xScale, 0.0f, 0.0f,
        -sine * yScale, cosine * yScale, 0.0f, 0.0f,
        0.0f, 0.0f, zScale, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
}

GlbStaticMeshDocument makeTriangleAsset()
{
    GlbStaticMeshDocument asset;
    asset.selectedScene = 0;
    asset.scenes.resize(1);
    asset.scenes[0].rootNodes = {0, 1, 2};
    asset.nodes.resize(3);

    asset.nodes[0].mesh = 0;
    asset.nodes[0].localTransform = translation(1.0f, 2.0f, 3.0f);
    asset.nodes[0].worldTransform = asset.nodes[0].localTransform;
    asset.nodes[1].camera = 0;
    asset.nodes[1].localTransform = translation(0.0f, 0.0f, 5.0f);
    asset.nodes[1].worldTransform = asset.nodes[1].localTransform;
    asset.nodes[2].light = 0;
    asset.nodes[2].localTransform = translation(4.0f, 5.0f, 6.0f);
    asset.nodes[2].worldTransform = asset.nodes[2].localTransform;

    GlbPrimitiveRecord primitive;
    primitive.positionAccessor = 0;
    primitive.normalAccessor = 1;
    primitive.texCoord0Accessor = 2;
    primitive.indexAccessor = 3;
    primitive.material = 0;
    primitive.positions = {
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f
    };
    primitive.normals = {
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f
    };
    primitive.texCoords0 = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f};
    primitive.indices = {0, 1, 2};
    asset.meshes.resize(1);
    asset.meshes[0].primitives.push_back(std::move(primitive));

    asset.materials.resize(1);
    asset.materials[0].baseColorFactor = {0.25f, 0.5f, 0.75f, 0.2f};
    asset.materials[0].metallicFactor = 0.3f;
    asset.materials[0].roughnessFactor = 0.6f;
    asset.materials[0].emissiveFactor = {0.1f, 0.2f, 0.3f};

    asset.cameras.resize(1);
    asset.cameras[0].type = GlbCameraType::Perspective;
    asset.cameras[0].verticalFovRadians = 1.0f;
    asset.cameras[0].nearPlane = 0.1f;
    asset.cameras[0].farPlane = 100.0f;

    asset.lights.resize(1);
    asset.lights[0].type = GlbLightType::Point;
    asset.lights[0].color = {1.0f, 0.5f, 0.25f};
    asset.lights[0].intensity = 2.0f;
    asset.lights[0].range = 12.0f;
    return asset;
}

GlbStaticMeshDocument makeAggregateTranslationOverflowDocument()
{
    auto asset = makeTriangleAsset();
    asset.nodes.resize(5);
    asset.nodes[3].children = {4};
    asset.nodes[3].localTransform = translation(600000.0f, 0.0f, 0.0f);
    asset.nodes[3].worldTransform = asset.nodes[3].localTransform;
    asset.nodes[4].parent = 3;
    asset.nodes[4].mesh = 0;
    asset.nodes[4].localTransform = translation(600000.0f, 0.0f, 0.0f);
    asset.nodes[4].worldTransform = translation(1200000.0f, 0.0f, 0.0f);
    asset.scenes[0].rootNodes = {3, 1, 2};
    return asset;
}

bool rejectedExactly(GlbStaticMeshDocument asset, const char* diagnostic)
{
    std::string error;
    const auto scene = adaptStaticGlbToVisual3DScene(asset, error);
    if (!scene && error == diagnostic)
        return true;
    std::fprintf(stderr, "FAIL: expected diagnostic '%s', got '%s'\n",
                 diagnostic, error.c_str());
    return false;
}
} // namespace

int main()
{
    auto asset = makeTriangleAsset();
    std::string error;
    const auto scene = adaptStaticGlbToVisual3DScene(asset, error);
    check(scene.has_value() && error.empty(),
          "an admitted in-memory static triangle translates without file access");
    check(scene && validateVisual3DScene(*scene).valid(),
          "the translated triangle passes the landed Visual3DScene validator");
    check(scene && scene->id == Scene3DId {1}
          && scene->objects[0].id == SceneObjectId {1}
          && scene->objects[0].material == SceneMaterialId {1}
          && scene->activeCamera == SceneCameraId {2}
          && scene->lights[0].id == SceneLightId {3},
          "scene, object, material, camera, and light identities derive from source indices");
    check(scene && scene->vertexCount == 3 && scene->indexCount == 3
          && scene->objectCount == 1 && scene->materialCount == 1
          && scene->cameraCount == 1 && scene->lightCount == 1,
          "the complete bounded triangle record set is owned by the scene value");
    check(scene && near(scene->vertices[1].position.x, 1.0f)
          && near(scene->vertices[2].uv.y, 1.0f)
          && scene->indices[2] == 2
          && near(scene->objects[0].transform.translation.x, 1.0f)
          && near(scene->objects[0].transform.translation.y, 2.0f)
          && near(scene->objects[0].transform.translation.z, 3.0f),
          "geometry, UVs, indices, and the root world transform are copied exactly");
    check(scene && near(scene->materials[0].baseColor.z, 0.75f)
          && near(scene->materials[0].opacity, 1.0f)
          && near(scene->materials[0].metallic, 0.3f)
          && near(scene->materials[0].roughness, 0.6f)
          && near(scene->materials[0].emissive.y, 0.2f)
          && scene->lights[0].kind == SceneLightKind::Point
          && near(scene->lights[0].range, 12.0f)
          && near(scene->cameras[0].farPlane, 100.0f),
          "supported material, light, and perspective-camera values are retained");

    asset.meshes[0].primitives[0].positions[0] = 99.0f;
    asset.meshes[0].primitives[0].indices[0] = 2;
    asset.materials[0].baseColorFactor[2] = 0.0f;
    check(scene && near(scene->vertices[0].position.x, 0.0f)
          && scene->indices[0] == 0
          && near(scene->materials[0].baseColor.z, 0.75f),
          "the output owns immutable copies rather than source-vector references");

    {
        auto indexed = makeTriangleAsset();
        indexed.selectedScene = 1;
        indexed.scenes.resize(2);
        indexed.scenes[1] = indexed.scenes[0];
        indexed.scenes[0].rootNodes.clear();
        indexed.meshes[0].primitives.push_back(indexed.meshes[0].primitives[0]);
        indexed.materials.push_back(indexed.materials[0]);
        indexed.meshes[0].primitives[1].material = 1;
        indexed.nodes.resize(4);
        indexed.nodes[3].mesh = 0;
        indexed.scenes[1].rootNodes.push_back(3);

        std::string indexedError;
        const auto indexedScene = adaptStaticGlbToVisual3DScene(indexed, indexedError);
        check(indexedScene && indexedError.empty()
              && indexedScene->id == Scene3DId {2}
              && indexedScene->objectCount == 4
              && indexedScene->objects[0].id == SceneObjectId {1}
              && indexedScene->objects[1].id == SceneObjectId {2}
              && indexedScene->objects[2].id == SceneObjectId {196609}
              && indexedScene->objects[3].id == SceneObjectId {196610}
              && indexedScene->materials[1].id == SceneMaterialId {2}
              && indexedScene->objects[1].material == SceneMaterialId {2}
              && indexedScene->objects[3].material == SceneMaterialId {2},
              "scene, object, and material identities retain their source indices");
    }
    {
        auto nested = makeTriangleAsset();
        nested.nodes.resize(4);
        nested.nodes[3].children = {0};
        nested.nodes[3].localTransform = translation(10.0f, 0.0f, 0.0f);
        nested.nodes[3].worldTransform = nested.nodes[3].localTransform;
        nested.nodes[0].parent = 3;
        nested.nodes[0].worldTransform = translation(11.0f, 2.0f, 3.0f);
        nested.scenes[0].rootNodes = {3, 1, 2};

        std::string nestedError;
        const auto nestedScene = adaptStaticGlbToVisual3DScene(nested, nestedError);
        check(nestedScene && nestedError.empty()
              && nestedScene->objectCount == 3
              && nestedScene->objects[0].id == SceneObjectId {65536}
              && nestedScene->objects[0].parent == SceneObjectId {262144}
              && nestedScene->objects[1].id == SceneObjectId {262144}
              && nestedScene->objects[2].id == SceneObjectId {1}
              && nestedScene->objects[2].parent == SceneObjectId {65536}
              && near(nestedScene->objects[1].transform.translation.x, 10.0f),
              "transform-only parents and mesh-node transforms remain distinct hierarchy records");
    }
    {
        auto nested = makeTriangleAsset();
        nested.nodes.resize(4);
        nested.nodes[0].children = {3};
        nested.nodes[3].parent = 0;
        nested.nodes[3].mesh = 0;
        nested.nodes[3].localTransform = translation(2.0f, 0.0f, 0.0f);
        nested.nodes[3].worldTransform = translation(3.0f, 2.0f, 3.0f);

        std::string nestedError;
        const auto nestedScene = adaptStaticGlbToVisual3DScene(nested, nestedError);
        check(nestedScene && nestedError.empty()
              && nestedScene->objectCount == 4
              && nestedScene->objects[1].id == SceneObjectId {262144}
              && nestedScene->objects[1].parent == SceneObjectId {65536}
              && nestedScene->objects[3].id == SceneObjectId {196609}
              && nestedScene->objects[3].parent == SceneObjectId {262144}
              && near(nestedScene->objects[1].transform.translation.x, 2.0f),
              "mesh-node hierarchy retains stable parent identity and a local child transform");
    }
    {
        auto nested = makeTriangleAsset();
        nested.nodes.resize(5);
        nested.nodes[3].children = {4};
        nested.nodes[3].localTransform = transform(0.5235987756f, 0.75f, 1.25f, 1.0f);
        nested.nodes[3].worldTransform = nested.nodes[3].localTransform;
        nested.nodes[4].parent = 3;
        nested.nodes[4].mesh = 0;
        nested.nodes[4].localTransform = translation(1.0f, 0.0f, 0.0f);
        nested.nodes[4].worldTransform = nested.nodes[3].localTransform;
        nested.nodes[4].worldTransform[12] = 0.649519026f;
        nested.nodes[4].worldTransform[13] = 0.375f;
        nested.scenes[0].rootNodes = {3, 1, 2};

        std::string nestedError;
        const auto nestedScene = adaptStaticGlbToVisual3DScene(nested, nestedError);
        check(nestedScene && nestedError.empty() && nestedScene->objectCount == 3
              && nestedScene->objects[0].vertexCount == 0
              && nestedScene->objects[1].parent == nestedScene->objects[0].id
              && nestedScene->objects[2].parent == nestedScene->objects[1].id,
              "a non-uniform transform-only parent remains valid when its composed child transform is TRS");
        if (nestedScene)
        {
            auto shearedHierarchy = *nestedScene;
            shearedHierarchy.objects[1].transform.rotation
                = {0.0f, 0.0f, 0.258819045f, 0.965925826f};
            const auto validation = validateVisual3DScene(shearedHierarchy);
            check(validation.code == Visual3DSceneValidationCode::InvalidObjectHierarchyTransform
                  && validation.recordIndex == 1,
                  "a rotated child under a non-uniform ancestor is rejected at the sheared record");
        }
    }
    {
        auto nested = makeTriangleAsset();
        nested.nodes.resize(5);
        nested.nodes[3].children = {4};
        nested.nodes[3].localTransform = transform(0.0f, 0.75f, 1.25f, 1.0f);
        nested.nodes[3].worldTransform = nested.nodes[3].localTransform;
        nested.nodes[4].parent = 3;
        nested.nodes[4].mesh = 0;
        nested.nodes[4].localTransform = transform(0.5235987756f, 1.0f, 1.0f, 1.0f);
        nested.nodes[4].worldTransform = {
            0.649519026f, 0.625f, 0.0f, 0.0f,
            -0.375f, 1.082531755f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f
        };
        nested.scenes[0].rootNodes = {3, 1, 2};

        check(rejectedExactly(std::move(nested),
                              "selected GLB node transform cannot be represented as Visual3DScene TRS"),
              "a rotated child under a non-uniform ancestor remains rejected when its world transform has shear");
    }
    {
        check(rejectedExactly(
                  makeAggregateTranslationOverflowDocument(),
                  "translated GLB scene failed Visual3DScene validation: InvalidObjectHierarchyTransform at record 1"),
              "admitted local GLB translations fail closed when their hierarchy exceeds the aggregate translation bound");
    }
    {
        auto spot = makeTriangleAsset();
        spot.lights[0].type = GlbLightType::Spot;
        spot.lights[0].innerConeAngle = 0.2f;
        spot.lights[0].outerConeAngle = 0.6f;
        std::string spotError;
        const auto spotScene = adaptStaticGlbToVisual3DScene(spot, spotError);
        check(spotScene && spotError.empty()
              && spotScene->lights[0].kind == SceneLightKind::Spot
              && near(spotScene->lights[0].innerConeAngle, 0.2f)
              && near(spotScene->lights[0].outerConeAngle, 0.6f),
              "spot-light identity and cone angles survive scene adaptation");
    }

    {
        auto malformed = makeTriangleAsset();
        malformed.meshes[0].primitives[0].positions.pop_back();
        check(rejectedExactly(std::move(malformed),
                              "GLB POSITION data has invalid cardinality"),
              "malformed POSITION cardinality is rejected");
    }
    {
        auto malformed = makeTriangleAsset();
        malformed.meshes[0].primitives[0].normals.pop_back();
        check(rejectedExactly(std::move(malformed),
                              "GLB NORMAL cardinality does not match POSITION"),
              "NORMAL cardinality must match POSITION");
    }
    {
        auto malformed = makeTriangleAsset();
        malformed.meshes[0].primitives[0].texCoords0.pop_back();
        check(rejectedExactly(std::move(malformed),
                              "GLB TEXCOORD_0 cardinality does not match POSITION"),
              "TEXCOORD_0 cardinality must match POSITION");
    }
    {
        auto attributed = makeTriangleAsset();
        auto& primitive = attributed.meshes[0].primitives[0];
        primitive.tangentAccessor = 4;
        primitive.tangents = {
            1.0f, 0.0f, 0.0f, 1.0f,
            1.0f, 0.0f, 0.0f, 1.0f,
            1.0f, 0.0f, 0.0f, 1.0f
        };
        primitive.color0Accessor = 5;
        primitive.colors0 = {
            1.0f, 0.0f, 0.0f, 1.0f,
            0.0f, 1.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 1.0f, 0.5f
        };
        std::string attributeError;
        const auto attributedScene = adaptStaticGlbToVisual3DScene(attributed, attributeError);
        check(attributedScene && attributeError.empty()
                  && near(attributedScene->vertices[0].tangent[0], 1.0f)
                  && near(attributedScene->vertices[1].color[1], 1.0f)
                  && near(attributedScene->vertices[2].color[3], 0.5f),
              "tangents and vertex colors adapt into the immutable vertex contract");
    }
    {
        auto malformed = makeTriangleAsset();
        malformed.meshes[0].primitives[0].indices = {0, 1};
        check(rejectedExactly(std::move(malformed),
                              "GLB primitive topology is not a non-empty triangle list"),
              "non-triangle-list cardinality is rejected");
    }
    {
        auto malformed = makeTriangleAsset();
        malformed.meshes[0].primitives[0].material = 1;
        check(rejectedExactly(std::move(malformed),
                              "GLB primitive material reference is missing or out of range"),
              "out-of-range primitive material references are rejected");
    }
    {
        auto malformed = makeTriangleAsset();
        malformed.meshes[0].primitives[0].material.reset();
        check(rejectedExactly(std::move(malformed),
                              "GLB primitive material reference is missing or out of range"),
              "implicit GLB materials are outside the bounded adapter subset");
    }
    {
        auto malformed = makeTriangleAsset();
        malformed.nodes[0].worldTransform[12] = 9.0f;
        check(rejectedExactly(std::move(malformed),
                              "selected GLB node world transform disagrees with its hierarchy"),
              "inconsistent local and world transforms are rejected");
    }
    {
        auto malformed = makeTriangleAsset();
        malformed.nodes[0].localTransform[4] = 0.25f;
        malformed.nodes[0].worldTransform = malformed.nodes[0].localTransform;
        check(rejectedExactly(std::move(malformed),
                              "selected GLB node transform cannot be represented as Visual3DScene TRS"),
              "sheared transforms outside the target TRS contract are rejected");
    }
    {
        auto malformed = makeTriangleAsset();
        malformed.nodes[0].localTransform[0] = std::numeric_limits<float>::quiet_NaN();
        malformed.nodes[0].worldTransform = malformed.nodes[0].localTransform;
        check(rejectedExactly(std::move(malformed),
                              "selected GLB node has a non-finite or non-affine transform"),
              "non-finite transforms are rejected before scene publication");
    }
    {
        auto textured = makeTriangleAsset();
        textured.images.emplace_back();
        textured.images[0].width = 1;
        textured.images[0].height = 1;
        textured.images[0].decodedRgba8 = {10, 20, 30, 40};
        textured.samplers.emplace_back();
        textured.samplers[0].magFilter = 9728;
        textured.samplers[0].minFilter = 9728;
        textured.textures.emplace_back();
        textured.textures[0].source = 0;
        textured.textures[0].sampler = 0;
        textured.materials[0].baseColorTexture = GlbTextureInfo {0, 0};
        std::string texturedError;
        const auto texturedScene = adaptStaticGlbToVisual3DScene(textured, texturedError);
        check(texturedScene && texturedError.empty()
                  && texturedScene->textureCount == 1
                  && texturedScene->textureTexelCount == 1
                  && texturedScene->materials[0].baseColorTexture == SceneTextureId {1}
                  && texturedScene->textureTexels[0].red == 10
                  && texturedScene->textureTexels[0].green == 20
                  && texturedScene->textureTexels[0].blue == 30
                  && texturedScene->textureTexels[0].alpha == 40,
              "decoded base-color texels adapt into the bounded scene texture subset");
    }
    {
        auto malformed = makeTriangleAsset();
        malformed.images.emplace_back();
        malformed.images[0].width = 2;
        malformed.images[0].height = 1;
        malformed.images[0].decodedRgba8 = {137, 80, 78, 71};
        malformed.samplers.emplace_back();
        malformed.samplers[0].magFilter = 9728;
        malformed.samplers[0].minFilter = 9728;
        malformed.textures.emplace_back();
        malformed.textures[0].source = 0;
        malformed.textures[0].sampler = 0;
        malformed.materials[0].baseColorTexture = GlbTextureInfo {0, 0};
        check(rejectedExactly(std::move(malformed),
                              "GLB decoded image texels are malformed or exceed Visual3DScene dimensions"),
              "malformed decoded image texels are rejected before scene publication");
    }
    {
        auto malformed = makeTriangleAsset();
        malformed.materials.resize(Visual3DScene::kMaxMaterials + 1);
        check(rejectedExactly(std::move(malformed),
                              "GLB material count exceeds Visual3DScene capacity"),
              "material capacity overflow is rejected before copying");
    }
    {
        auto malformed = makeTriangleAsset();
        const auto primitive = malformed.meshes[0].primitives[0];
        malformed.meshes[0].primitives.resize(Visual3DScene::kMaxObjects + 1, primitive);
        check(rejectedExactly(std::move(malformed),
                              "GLB primitive count exceeds Visual3DScene object capacity"),
              "primitive instance overflow is rejected at the object capacity");
    }
    {
        auto malformed = makeTriangleAsset();
        auto& primitive = malformed.meshes[0].primitives[0];
        primitive.positions.resize((Visual3DScene::kMaxVertices + 1) * 3, 0.0f);
        primitive.normals.resize((Visual3DScene::kMaxVertices + 1) * 3, 0.0f);
        primitive.texCoords0.resize((Visual3DScene::kMaxVertices + 1) * 2, 0.0f);
        check(rejectedExactly(std::move(malformed),
                              "GLB vertex data exceeds Visual3DScene capacity"),
              "vertex capacity overflow is rejected before fixed-array writes");
    }
    {
        auto malformed = makeTriangleAsset();
        malformed.meshes[0].primitives[0].indices.resize(
            Visual3DScene::kMaxIndices + 3, 0);
        check(rejectedExactly(std::move(malformed),
                              "GLB index data exceeds Visual3DScene capacity"),
              "index capacity overflow is rejected before fixed-array writes");
    }
    {
        auto malformed = makeTriangleAsset();
        malformed.meshes[0].primitives[0].normals[0] = 0.0f;
        malformed.meshes[0].primitives[0].normals[1] = 0.0f;
        malformed.meshes[0].primitives[0].normals[2] = 0.0f;
        check(rejectedExactly(
                  std::move(malformed),
                  "translated GLB scene failed Visual3DScene validation: InvalidVertexValue at record 0"),
              "scene validation failures include the exact code and record index");
    }

    std::fprintf(stderr, failures ? "%d GLB scene adapter checks failed\n"
                                  : "GLB scene adapter checks passed\n",
                 failures);
    return failures == 0 ? 0 : 1;
}
