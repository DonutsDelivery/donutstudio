#include "../src/glb_scene_adapter.h"
#include "imported_geometry_fixture.h"
#include "../../shared/VideoScreenAsset.h"
#include "../../shared/ReactiveCharacterAsset.h"
#include "../../shared/SceneAovOperationContract.h"

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
    {
        using namespace videowire::geometry;
        auto repeated=makeTriangleAsset();
        repeated.nodes.push_back(repeated.nodes[0]);
        repeated.nodes[3].localTransform=translation(-2,0,1);
        repeated.nodes[3].worldTransform=repeated.nodes[3].localTransform;
        repeated.scenes[0].rootNodes.push_back(3);
        std::string diagnostic;
        const auto joined=adaptGlbObjectsToGeometryCore(repeated,0,0,81,80,1,diagnostic);
        check(joined.has_value(),"Scene Objects extracts repeated mesh nodes as joined Geometry3D");
        if (joined) {
            const auto& value=joined->descriptor(); const auto& mesh=std::get<GeometryData>(value.data);
            check(mesh.positions.size()==6 && near(mesh.positions[0].x,1) && near(mesh.positions[3].x,-2)
                && mesh.vertexIds[0]==65537 && mesh.vertexIds[3]==262145
                && mesh.indices==std::vector<std::uint32_t>{0,1,2,3,4,5},
                "joined geometry bakes per-node world transforms without merging vertex identities");
            check(value.attributes.size()==6 && value.attributes[4].elements[0].components[0]==1
                && value.attributes[4].elements[1].components[0]==196609
                && value.attributes[5].elements[3].components[0]==4,
                "joined geometry retains imported draw IDs and node IDs as fields");
            const auto restored=decodeRuntimeValue(encodeRuntimeValue(*joined),
                withAttributeContract(importedGeometryContract(),value),{}, {},diagnostic);
            check(restored && equalAttributes(restored->descriptor().attributes,value.attributes),
                "joined imported attributes survive immutable transport");
        }
        const auto selected=adaptGlbObjectsToGeometryCore(repeated,0,4,82,80,1,diagnostic);
        check(selected && std::get<GeometryData>(selected->descriptor().data).positions.size()==3,
            "Pose Object geometry selects one exact repeated node");
        check(!adaptGlbObjectsToGeometryCore(repeated,0,2,83,80,1,diagnostic)
            && diagnostic.find("not drawn")!=std::string::npos,"non-mesh node geometry selection fails closed");
    }
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
    {
        auto asset = makeTriangleAsset();
        auto& primitive = asset.meshes[0].primitives[0];
        primitive.normalAccessor.reset();
        primitive.indexAccessor.reset();
        primitive.generatedFlatNormals = true;
        primitive.positions = {0, 0, 0, 0, 1, 0, 0, 0, 1};
        primitive.normals = {1, 0, 0, 1, 0, 0, 1, 0, 0};
        std::string error;
        const auto scene = adaptStaticGlbToVisual3DScene(asset, error);
        check(scene && error.empty() && scene->vertexCount == 3 && scene->indexCount == 3
              && scene->vertices[0].normal.x == 1 && scene->vertices[2].normal.x == 1
              && scene->vertices[2].uv.y == 1 && scene->indices[2] == 2,
              "generated non-indexed flat normals reach native vertices without changing UV or topology");
        auto malformed = asset;
        malformed.meshes[0].primitives[0].indices = {0, 2, 1};
        check(rejectedExactly(std::move(malformed),
                              "generated flat GLB normals require sequential triangle indices"),
              "generated-normal provenance cannot hide changed triangle corner order");
        malformed = asset;
        malformed.meshes[0].primitives[0].normalAccessor = 1;
        check(rejectedExactly(std::move(malformed),
                              "generated flat GLB normals require non-indexed triangle corners"),
              "generated normals cannot also claim an authored NORMAL accessor");
        malformed = asset;
        malformed.meshes[0].primitives[0].indexAccessor = 3;
        check(rejectedExactly(std::move(malformed),
                              "flat GLB normals for indexed TRIANGLES require unsupported corner expansion"),
              "native adaptation explicitly rejects indexed missing-normal geometry");
    }
    {
        auto asset = makeTriangleAsset();
        asset.nodes[1].camera.reset();
        asset.cameras.clear();
        std::string error;
        const auto framed = adaptStaticGlbToVisual3DScene(asset, error);
        check(framed && framed->cameraCount == 1
            && near(framed->cameras[0].transform.translation.x, 1.5f)
            && near(framed->cameras[0].transform.translation.y, 2.5f)
            && framed->cameras[0].transform.translation.z > 3.0f
            && framed->activeCamera == framed->cameras[0].id,
            "mesh-only imports receive a deterministic camera around selected world-space geometry");
        GlbAdmissionOptions options;
        options.admitAnimations = true;
        options.admitSkins = true;
        options.sceneIndex = 0;
        const auto character = decodeAnimatedGlbBaseScene(
            std::vector<std::uint8_t>(reactivecharacter::kGlb.begin(), reactivecharacter::kGlb.end()),
            options, error);
        const auto scene = character ? adaptAnimatedGlbMeshToVisual3DScene(*character, error, 0, true)
                                     : std::nullopt;
        check(scene && scene->cameraCount == 1 && scene->vertexCount > 3,
            "bundled skinned character without an embedded camera reaches native scene admission");
    }
    {
        std::string error;
        const auto asset = decodeStaticGlb(videoscreen::kGlb.data(), videoscreen::kGlb.size(), {}, error);
        check(asset.has_value(), "screen starter GLB decodes through the production static asset path");
        const auto screen = asset ? adaptStaticGlbToVisual3DScene(*asset, error) : std::nullopt;
        check(screen.has_value(), "screen starter UV mesh and embedded camera admit as a native scene");
        if (asset && screen)
        {
            const auto& mesh = asset->meshes.front().primitives.front();
            check(mesh.positions.size() == 12 && mesh.indices.size() == 6
                      && near(mesh.positions[3] - mesh.positions[0], 1.6f)
                      && near(mesh.positions[7] - mesh.positions[1], 0.9f),
                  "screen starter keeps exact 16:9 geometry without imported texture bytes");
            check(mesh.texCoords0 == std::vector<float> {0, 1, 1, 1, 1, 0, 0, 0},
                  "screen UVs address the first decoded video row at the top on GL and Metal");
            check(screen->textureCount == 0 && screen->objectCount == 1,
                  "screen has one authored material target and no silent imported texture fallback");
        }
    }
    {
        using namespace videowire::geometry;
        auto selected=importedGeometryFixture();
        std::string diagnostic;
        auto extracted=adaptGlbMeshToGeometryCore(selected,1,71,70,3,diagnostic);
        check(extracted.has_value(),"selected nonzero GLB mesh extracts into admitted Geometry3D");
        if (extracted) {
            const auto& value=extracted->descriptor();
            const auto& mesh=std::get<GeometryData>(value.data);
            check(mesh.positions.size()==3 && mesh.vertexIds==std::vector<StableId>{131073,131074,131075}
                && mesh.indices==std::vector<std::uint32_t>{0,1,2},"imported topology and stable mesh vertex IDs are retained");
            check(value.attributes.size()==4 && value.attributes[1].elements[2].components[1]==1
                && value.attributes[3].elements[0].components[0]==1,"UV, color, normal and face material attributes survive extraction");
            const auto contract=withAttributeContract(importedGeometryContract(),value);
            const auto bytes=encodeRuntimeValue(*extracted);
            const auto restored=decodeRuntimeValue(bytes,contract,{}, {},diagnostic);
            check(restored && equal(std::get<GeometryData>(restored->descriptor().data),mesh)
                && equalAttributes(restored->descriptor().attributes,value.attributes),"imported source replays through the immutable transport");
            auto tampered=value; std::get<GeometryData>(tampered.data).positions[0].x+=1;
            check(!admitValue(tampered,contract,diagnostic),"retained source rejects a forged terminal mesh");
            auto aliased=value;
            auto mutableSource=std::make_shared<RetainedMeshData>(*value.operations.front().retainedMesh);
            aliased.operations.front().retainedMesh=mutableSource;
            auto owned=admitValue(aliased,contract,diagnostic);
            mutableSource->geometry.positions[0].x+=10;
            check(owned && equal(owned->descriptor().operations.front().retainedMesh->geometry,mesh),
                "admission copies producer aliases before publishing immutable imported geometry");
            Transform transform; transform.translation={1,2,3}; transform.scale={2,1,1};
            auto moved=lowerTransformGeometry(value,72,transform,contract,diagnostic);
            auto material=moved ? lowerAssignMaterial(*moved,73,19,contract,diagnostic) : std::nullopt;
            check(material && admitValue(*material,contract,diagnostic),"imported source supports retained transforms and material assignment");
            PortContract pointsContract; pointsContract.carrier=CarrierKind::points3D; pointsContract.maxPoints=4096;
            auto points=lowerPointsFromVertices(value,74,pointsContract,diagnostic);
            PortContract instancesContract; instancesContract.carrier=CarrierKind::instances3D; instancesContract.maxInstances=4096;
            auto instances=points ? lowerInstanceOnPoints(*points,value.stableId,75,instancesContract,diagnostic) : std::nullopt;
            check(instances && admitValue(*instances,instancesContract,diagnostic),"imported vertices can instance their retained mesh without duplicate sources");
        }
        check(!adaptGlbMeshToGeometryCore(selected,0,71,70,3,diagnostic),"mesh selection outside the selected scene is rejected");
        selected.metadata.animations=1;
        check(!adaptGlbMeshToGeometryCore(selected,1,71,70,3,diagnostic)
            && diagnostic.find("animated")!=std::string::npos,"animated extraction has an explicit diagnostic");
        selected=importedGeometryFixture(); selected.meshes[1].primitives[0].indices[2]=99;
        check(!adaptGlbMeshToGeometryCore(selected,1,71,70,3,diagnostic),"out-of-range imported indices are rejected");
    }
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
        const auto animatedScene = adaptAnimatedGlbMeshToVisual3DScene(nested, nestedError, 0);
        check(animatedScene && nestedError.empty() && animatedScene->objectCount == 1
              && animatedScene->objects[0].id == SceneObjectId {1}
              && !animatedScene->objects[0].parent.isValid()
              && near(animatedScene->objects[0].transform.translation.x, 11.0f)
              && near(animatedScene->objects[0].transform.translation.y, 2.0f)
              && near(animatedScene->objects[0].transform.translation.z, 3.0f),
              "the animated draw retains its mesh identity and composed parent translation");
        check(nested.nodes[0].parent == std::optional<std::size_t> {3}
              && near(nested.nodes[0].localTransform[12], 1.0f),
              "single-draw adaptation leaves the imported source hierarchy intact");
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
        check(!adaptAnimatedGlbMeshToVisual3DScene(nested, nestedError, 0)
              && nestedError == "imported animated scene requires one renderable mesh primitive",
              "single-draw adaptation rejects a second renderable object");
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
        const auto animatedScene = adaptAnimatedGlbMeshToVisual3DScene(nested, nestedError, 0);
        check(animatedScene && nestedError.empty()
              && animatedScene->objects[0].id == SceneObjectId {262145}
              && near(animatedScene->objects[0].transform.translation.x, 0.649519026f)
              && near(animatedScene->objects[0].transform.translation.y, 0.375f)
              && near(animatedScene->objects[0].transform.rotation.z, 0.258819045f)
              && near(animatedScene->objects[0].transform.rotation.w, 0.965925826f)
              && near(animatedScene->objects[0].transform.scale.x, 0.75f)
              && near(animatedScene->objects[0].transform.scale.y, 1.25f),
              "animated adaptation composes parent rotation and non-uniform scale");
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
        static_assert(sizeof(Visual3DScene) < 512u * 1024u,
                      "texture capacity must not become an inline multi-MiB scene array");
        auto textured = makeTriangleAsset();
        textured.images.emplace_back();
        auto& image = textured.images[0];
        image.width = 1024;
        image.height = 1024;
        image.decodedRgba8.resize(1024u * 1024u * 4u);
        for (std::size_t texel = 0; texel < 1024u * 1024u; ++texel)
        {
            image.decodedRgba8[texel * 4] = static_cast<std::uint8_t>(texel % 251);
            image.decodedRgba8[texel * 4 + 1] = 20;
            image.decodedRgba8[texel * 4 + 2] = 30;
            image.decodedRgba8[texel * 4 + 3] = 255;
        }
        textured.samplers.emplace_back();
        textured.textures.emplace_back();
        textured.textures[0].source = 0;
        textured.textures[0].sampler = 0;
        textured.materials[0].baseColorTexture = GlbTextureInfo {0, 0};
        std::string textureError;
        auto full = adaptStaticGlbToVisual3DScene(textured, textureError);
        check(full && textureError.empty() && full->textureTexelCount == 1024u * 1024u
              && full->textureTexels.size() == full->textureTexelCount
              && full->textures[0].width == 1024 && full->textures[0].height == 1024
              && full->textures[0].minFilter == 9987 && full->textures[0].wrapS == 10497,
              "a full 4 MiB texture adapts without changing pixels, dimensions or sampler");
        if (full)
        {
            textured.images.clear();
            check(full->textureTexels.front().green == 20
                  && full->textureTexels.back().red == (1024u * 1024u - 1u) % 251,
                  "the adapted scene owns all texels after decoded images are released");
            const auto encoded = sceneaov::detail::encodeScene(*full);
            Visual3DScene decoded;
            check(!encoded.empty() && sceneaov::detail::decodeScene(encoded, decoded)
                  && sceneaov::detail::encodeScene(decoded) == encoded,
                  "full-capacity scene wire round trips all texture bytes and sampler fields");
            if (!decoded.textureTexels.empty()) decoded.textureTexels.front().green = 99;
            check(!decoded.textureTexels.empty() && full->textureTexels.front().green == 20,
                  "scene copies have independent texture ownership");

            sceneaov::Payload payload;
            payload.output = renderpassoutput::Output::Depth;
            payload.extent = {64, 64};
            payload.scene = std::make_shared<const Visual3DScene>(std::move(*full));
            const auto xml = sceneaov::serialize(payload);
            sceneaov::Payload parsed;
            check(!xml.empty() && sceneaov::parse(xml, parsed)
                  && parsed.scene->textureTexels.size() == 1024u * 1024u,
                  "SceneAov XML admits a bounded full-resolution scene snapshot");

            auto malformed = *payload.scene;
            malformed.textureTexels.pop_back();
            check(!validateVisual3DScene(malformed).valid(),
                  "texture counts cannot read beyond allocated scene storage");
            malformed = *payload.scene;
            ++malformed.textureTexelCount;
            check(!validateVisual3DScene(malformed).valid(),
                  "scene texel count cannot exceed the fixed aggregate cap");
            malformed = *payload.scene;
            malformed.textureCount = 2;
            malformed.textures[1] = malformed.textures[0];
            malformed.textures[1].id.value = 2;
            check(validateVisual3DScene(malformed).code
                      == Visual3DSceneValidationCode::TextureTexelCapacityExceeded,
                  "overlapping texture records cannot multiply the admitted upload budget");
            malformed = *payload.scene;
            malformed.textures[0].width = std::numeric_limits<std::uint32_t>::max();
            check(!validateVisual3DScene(malformed).valid(),
                  "huge dimensions fail before texture pointer arithmetic");

            const auto writeCount = [](std::vector<std::uint8_t>& bytes,
                                       std::size_t index, std::uint32_t count)
            {
                for (unsigned byte = 0; byte < 4; ++byte)
                    bytes[24 + index * 4 + byte] = static_cast<std::uint8_t>(count >> (byte * 8));
            };
            for (std::size_t countIndex = 0; countIndex < 8; ++countIndex)
            {
                std::vector<std::uint8_t> header(encoded.begin(), encoded.begin() + 56);
                writeCount(header, countIndex, static_cast<std::uint32_t>(
                    sceneaov::detail::kSceneCountLimits[countIndex] + 1));
                Visual3DScene rejected;
                check(!sceneaov::detail::decodeScene(header, rejected)
                      && rejected.textureTexels.empty(),
                      "every forged scene count is rejected before texture allocation");
            }
            auto truncated = encoded;
            truncated.pop_back();
            Visual3DScene rejected;
            check(!sceneaov::detail::decodeScene(truncated, rejected)
                  && rejected.textureTexels.empty(),
                  "truncated texture payload is rejected before allocating declared texels");
            truncated = encoded;
            truncated.push_back(0);
            check(!sceneaov::detail::decodeScene(truncated, rejected)
                  && rejected.textureTexels.empty(),
                  "trailing scene bytes cannot hide a mismatched array payload");
            std::vector<std::uint8_t> hexOutput;
            check(!sceneaov::detail::unhex(std::string(
                      sceneaov::detail::kMaximumSceneBytes * 2 + 2, '0'), hexOutput)
                  && hexOutput.empty(),
                  "oversized hex input is rejected before the decoded wire allocation");
        }
        // A second texture record needs its own upload budget even with the same image.
        textured.images.emplace_back();
        textured.images[0].width = 1024;
        textured.images[0].height = 1024;
        textured.images[0].decodedRgba8.resize(1024u * 1024u * 4u);
        textured.textures.push_back(textured.textures[0]);
        textured.materials[0].normalTexture = GlbTextureInfo {1, 0};
        check(rejectedExactly(std::move(textured), "GLB decoded image texels exceed Visual3DScene capacity"),
              "adapter texture capacity is aggregate across material texture records");
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
