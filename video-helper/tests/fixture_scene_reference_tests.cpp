#include "support/fixture_scene.h"
#include "support/fixture_scene_reference_renderer.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <type_traits>

using namespace HarmonicMIDI::grid;
using videohelper::fixture3d::reference::RenderCode;

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

std::uint64_t hashBytes (const std::vector<std::uint8_t>& bytes)
{
    std::uint64_t hash = 1469598103934665603ull;
    for (const auto byte : bytes)
    {
        hash ^= byte;
        hash *= 1099511628211ull;
    }
    return hash;
}

std::uint64_t hashWords (const std::vector<std::uint32_t>& values)
{
    std::uint64_t hash = 1469598103934665603ull;
    for (const auto value : values)
    {
        for (unsigned shift = 0; shift < 32; shift += 8)
        {
            hash ^= static_cast<std::uint8_t> (value >> shift);
            hash *= 1099511628211ull;
        }
    }
    return hash;
}

bool same (SceneVec3 left, SceneVec3 right)
{
    return left.x == right.x && left.y == right.y && left.z == right.z;
}
} // namespace

int main()
{
    static_assert (! std::is_same_v<SceneObjectId, SceneMaterialId>);
    static_assert (! std::is_same_v<SceneMaterialId, SceneLightId>);
    static_assert (! std::is_same_v<SceneMaterialId, SceneTextureId>);
    static_assert (! std::is_same_v<SceneLightId, SceneCameraId>);

    check (Visual3DScene::kSchemaVersion == 1,
           "scene schema version is stable");
    check (Visual3DScene::kMaxVertices == 4096
           && Visual3DScene::kMaxIndices == 12288
           && Visual3DScene::kMaxObjects == 64
           && Visual3DScene::kMaxMaterials == 32
           && Visual3DScene::kMaxTextures == 16
           && Visual3DScene::kMaxTextureTexels == 16384
           && Visual3DScene::kMaxLights == 16
           && Visual3DScene::kMaxCameras == 8,
           "scene record capacities are fixed");
    check (Visual3DSceneRenderContract::kBackendRequirement
               == Visual3DSceneBackendRequirement::NativeGpu
           && ! Visual3DSceneRenderContract::kAllowsCpuImageFallback,
           "production scene rendering requires an admitted native GPU backend");

    const SceneTransform3D identity;
    check (same (identity.translation, {})
           && same (identity.scale, { 1.0f, 1.0f, 1.0f })
           && identity.rotation.x == 0.0f
           && identity.rotation.y == 0.0f
           && identity.rotation.z == 0.0f
           && identity.rotation.w == 1.0f,
           "the default transform is identity");

    const auto fixture = videohelper::fixture3d::makeScene();
    check (validateVisual3DScene (fixture).valid(),
           "the authored cube fixture passes scene admission");
    check (videohelper::fixture3d::kFixtureVersion == 1
           && fixture.id == videohelper::fixture3d::kSceneId
           && fixture.activeCamera == videohelper::fixture3d::kCameraId,
           "fixture version and stable identities match the contract");
    check (fixture.vertexCount == 24 && fixture.indexCount == 36
           && fixture.objectCount == 1 && fixture.materialCount == 1
           && fixture.textureCount == 1 && fixture.textureTexelCount == 4
           && fixture.lightCount == 1 && fixture.cameraCount == 1,
           "fixture record counts are exact");
    check (fixture.objects[0].id == videohelper::fixture3d::kCubeObjectId
           && fixture.objects[0].material == videohelper::fixture3d::kCubeMaterialId
           && fixture.materials[0].id == videohelper::fixture3d::kCubeMaterialId
           && fixture.materials[0].baseColorTexture
                  == videohelper::fixture3d::kCubeTextureId
           && fixture.textures[0].id == videohelper::fixture3d::kCubeTextureId
           && fixture.lights[0].id == videohelper::fixture3d::kKeyLightId
           && fixture.cameras[0].id == videohelper::fixture3d::kCameraId,
           "fixture object references retain stable typed identities");
    check (same (fixture.vertices[0].position, { -1.0f, -1.0f, 1.0f })
           && same (fixture.vertices[23].position, { -1.0f, -1.0f, 1.0f })
           && fixture.indices[0] == 0 && fixture.indices[35] == 23,
           "fixture vertex and triangle order are authored data");

    auto malformed = fixture;
    malformed.indices[35] = 24;
    check (validateVisual3DScene (malformed).code
               == Visual3DSceneValidationCode::IndexOutsideObjectVertexRange,
           "object-local indices cannot escape the vertex range");

    malformed = fixture;
    malformed.materials[0].roughness = std::numeric_limits<float>::quiet_NaN();
    check (validateVisual3DScene (malformed).code
               == Visual3DSceneValidationCode::InvalidMaterialValue,
           "non-finite material values fail admission");

    malformed = fixture;
    malformed.textures[0].width = 3;
    check (validateVisual3DScene (malformed).code
               == Visual3DSceneValidationCode::InvalidTextureRange,
           "texture ranges cannot escape the immutable texel snapshot");

    malformed = fixture;
    malformed.materials[0].baseColorTexture = SceneTextureId { 999 };
    check (validateVisual3DScene (malformed).code
               == Visual3DSceneValidationCode::MissingBaseColorTexture,
           "materials cannot reference absent textures");

    malformed = fixture;
    malformed.objects[1] = malformed.objects[0];
    malformed.objects[1].id = SceneObjectId { 102 };
    malformed.objects[1].parent = malformed.objects[0].id;
    malformed.objects[0].parent = malformed.objects[1].id;
    malformed.objectCount = 2;
    const auto cycle = validateVisual3DScene (malformed);
    check (cycle.code == Visual3DSceneValidationCode::CyclicObjectHierarchy
           && cycle.recordIndex == 0,
           "multi-object parent cycles fail admission deterministically");

    const auto first = videohelper::fixture3d::reference::render (fixture, 96, 96);
    const auto repeated = videohelper::fixture3d::reference::render (fixture, 96, 96);
    check (first.rendered() && first.width == 96 && first.height == 96,
           "the test-only fixture oracle renders the admitted scene");
    check (first.rgba.size() == 96u * 96u * 4u
           && first.depth24.size() == 96u * 96u
           && first.normalRgb8.size() == 96u * 96u * 3u
           && first.objectIds.size() == 96u * 96u
           && first.materialIds.size() == 96u * 96u,
           "the fixture oracle returns every bounded fixture output");
    check (first.rgba == repeated.rgba && first.depth24 == repeated.depth24
           && first.normalRgb8 == repeated.normalRgb8
           && first.objectIds == repeated.objectIds
           && first.materialIds == repeated.materialIds
           && first.rasterizedTriangles == repeated.rasterizedTriangles
           && first.coveredPixels == repeated.coveredPixels,
           "identical fixture inputs repeat exactly");
    check (first.rasterizedTriangles == 12 && first.coveredPixels == 3486,
           "the rendered fixture has exact triangle and pixel coverage");
    check (first.rgba[0] == 7 && first.rgba[1] == 10
           && first.rgba[2] == 18 && first.rgba[3] == 255,
           "uncovered pixels retain the fixed oracle background");

    std::size_t identifiedPixels = 0;
    bool identityOutputsValid = true;
    for (std::size_t pixel = 0; pixel < first.objectIds.size(); ++pixel)
    {
        if (first.objectIds[pixel] == 0)
        {
            identityOutputsValid = identityOutputsValid
                                   && first.materialIds[pixel] == 0;
            continue;
        }
        ++identifiedPixels;
        identityOutputsValid = identityOutputsValid
            && first.objectIds[pixel] == videohelper::fixture3d::kCubeObjectId.value
            && first.materialIds[pixel]
                   == videohelper::fixture3d::kCubeMaterialId.value;
    }
    check (identityOutputsValid && identifiedPixels == first.coveredPixels,
           "identity outputs cover exactly the depth-tested fixture pixels");

    malformed = fixture;
    malformed.materials[0].baseColorTexture = {};
    const auto untextured = videohelper::fixture3d::reference::render (malformed, 96, 96);
    check (untextured.rendered() && untextured.rgba != first.rgba,
           "the authored texture contributes to the reference pixels");

    malformed = fixture;
    malformed.lightCount = 0;
    const auto unlit = videohelper::fixture3d::reference::render (malformed, 96, 96);
    check (unlit.rendered() && unlit.rgba != first.rgba,
           "the authored directional light contributes to the reference pixels");

    malformed = fixture;
    malformed.objects[0].transform.translation = { 0.2f, -0.1f, 0.0f };
    malformed.objects[0].transform.scale = { 0.7f, 1.15f, 1.0f };
    const auto transformed = videohelper::fixture3d::reference::render (malformed, 96, 96);
    check (transformed.rendered() && transformed.rgba != first.rgba
           && transformed.depth24 != first.depth24
           && transformed.normalRgb8 != first.normalRgb8,
           "object translation and non-uniform scale alter all geometric outputs");

    malformed = fixture;
    malformed.cameras[0].transform.translation.x = 0.5f;
    const auto moved = videohelper::fixture3d::reference::render (malformed, 96, 96);
    check (moved.rendered() && moved.rgba != first.rgba,
           "camera changes alter the reference pixels");

    malformed = fixture;
    malformed.cameras[0].nearPlane = malformed.cameras[0].farPlane;
    const auto rejected = videohelper::fixture3d::reference::render (malformed, 96, 96);
    check (rejected.code == RenderCode::InvalidScene
           && rejected.sceneValidation.code
                  == Visual3DSceneValidationCode::InvalidCameraProjection
           && rejected.rgba.empty() && rejected.depth24.empty()
           && rejected.normalRgb8.empty() && rejected.objectIds.empty()
           && rejected.materialIds.empty(),
           "the fixture oracle renders no partial result for an invalid scene");

    malformed = fixture;
    malformed.materials[0].opacity = 0.5f;
    const auto transparent = videohelper::fixture3d::reference::render (malformed, 96, 96);
    check (transparent.code == RenderCode::UnsupportedTransparentMaterial
           && transparent.rgba.empty(),
           "the bounded oracle rejects unsupported transparency");

    const auto invalidSize = videohelper::fixture3d::reference::render (fixture, 1025, 1);
    check (invalidSize.code == RenderCode::InvalidDimensions
           && invalidSize.rgba.empty(),
           "the fixture oracle enforces its pixel bound before allocation");

    const auto rgbaHash = hashBytes (first.rgba);
    const auto depthHash = hashWords (first.depth24);
    const auto normalHash = hashBytes (first.normalRgb8);
    const auto objectHash = hashWords (first.objectIds);
    const auto materialHash = hashWords (first.materialIds);
    check (rgbaHash == 0x8af3999e2dba40cfull
           && depthHash == 0x8902b6401525f97dull
           && normalHash == 0x089cce690a4eb187ull
           && objectHash == 0x90e67611e21504b3ull
           && materialHash == 0xe56795398767d433ull,
           "fixture outputs match the versioned golden output");
    std::printf ("fixture-scene-reference: %d/%d checks passed; "
                 "triangles=%u covered=%u rgba=%016llx depth=%016llx "
                 "normal=%016llx object=%016llx material=%016llx\n",
                 checks - failures, checks, first.rasterizedTriangles,
                 first.coveredPixels,
                 static_cast<unsigned long long> (rgbaHash),
                 static_cast<unsigned long long> (depthHash),
                 static_cast<unsigned long long> (normalHash),
                 static_cast<unsigned long long> (objectHash),
                 static_cast<unsigned long long> (materialHash));
    return failures == 0 ? 0 : 1;
}
