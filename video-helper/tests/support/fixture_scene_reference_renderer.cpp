#include "fixture_scene_reference_renderer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace videohelper::fixture3d::reference
{
namespace
{
using namespace HarmonicMIDI::grid;

constexpr std::uint32_t kMaxFixtureExtent = 1024;
constexpr std::uint32_t kFarDepth24 = 0x00ffffffu;
constexpr std::int64_t kSubpixelScale = 256;
constexpr std::int64_t kSubpixelCenter = kSubpixelScale / 2;
constexpr std::array<std::uint8_t, 4> kBackground {{ 7, 10, 18, 255 }};

struct ProjectedVertex
{
    std::int64_t x = 0;
    std::int64_t y = 0;
    std::uint32_t depth = kFarDepth24;
    SceneVec2 uv {};
    SceneVec3 lighting {};
    SceneVec3 worldNormal {};
    bool visible = false;
};

SceneVec3 add (SceneVec3 a, SceneVec3 b) noexcept
{
    return { a.x + b.x, a.y + b.y, a.z + b.z };
}

SceneVec3 subtract (SceneVec3 a, SceneVec3 b) noexcept
{
    return { a.x - b.x, a.y - b.y, a.z - b.z };
}

SceneVec3 multiply (SceneVec3 a, float value) noexcept
{
    return { a.x * value, a.y * value, a.z * value };
}

SceneVec3 multiply (SceneVec3 a, SceneVec3 b) noexcept
{
    return { a.x * b.x, a.y * b.y, a.z * b.z };
}

float dot (SceneVec3 a, SceneVec3 b) noexcept
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

SceneVec3 normalize (SceneVec3 value) noexcept
{
    const auto lengthSquared = dot (value, value);
    if (lengthSquared <= 0.0f)
        return {};
    return multiply (value, 1.0f / std::sqrt (lengthSquared));
}

SceneVec3 rotate (const SceneQuaternion& q, SceneVec3 value) noexcept
{
    const SceneVec3 axis { q.x, q.y, q.z };
    const SceneVec3 twiceCross {
        2.0f * (axis.y * value.z - axis.z * value.y),
        2.0f * (axis.z * value.x - axis.x * value.z),
        2.0f * (axis.x * value.y - axis.y * value.x)
    };
    const SceneVec3 axisCross {
        axis.y * twiceCross.z - axis.z * twiceCross.y,
        axis.z * twiceCross.x - axis.x * twiceCross.z,
        axis.x * twiceCross.y - axis.y * twiceCross.x
    };
    return add (value, add (multiply (twiceCross, q.w), axisCross));
}

SceneVec3 inverseRotate (const SceneQuaternion& q, SceneVec3 value) noexcept
{
    return rotate ({ -q.x, -q.y, -q.z, q.w }, value);
}

SceneVec3 transformPoint (const SceneTransform3D& transform, SceneVec3 value) noexcept
{
    return add (rotate (transform.rotation, multiply (value, transform.scale)),
                transform.translation);
}

SceneVec3 transformNormal (const SceneTransform3D& transform, SceneVec3 value) noexcept
{
    value = { value.x / transform.scale.x,
              value.y / transform.scale.y,
              value.z / transform.scale.z };
    return normalize (rotate (transform.rotation, value));
}

const SceneObjectRecord* findObject (const Visual3DScene& scene, SceneObjectId id) noexcept
{
    return visual3d_detail::findById (scene.objects, scene.objectCount, id);
}

const SceneMaterialRecord* findMaterial (const Visual3DScene& scene,
                                         SceneMaterialId id) noexcept
{
    return visual3d_detail::findById (scene.materials, scene.materialCount, id);
}

const SceneTextureRecord* findTexture (const Visual3DScene& scene,
                                       SceneTextureId id) noexcept
{
    return visual3d_detail::findById (scene.textures, scene.textureCount, id);
}

SceneVec3 objectPointToWorld (const Visual3DScene& scene,
                              const SceneObjectRecord& object,
                              SceneVec3 value) noexcept
{
    auto* current = &object;
    for (std::size_t depth = 0; current != nullptr && depth < scene.objectCount; ++depth)
    {
        value = transformPoint (current->transform, value);
        current = findObject (scene, current->parent);
    }
    return value;
}

SceneVec3 objectNormalToWorld (const Visual3DScene& scene,
                               const SceneObjectRecord& object,
                               SceneVec3 value) noexcept
{
    auto* current = &object;
    for (std::size_t depth = 0; current != nullptr && depth < scene.objectCount; ++depth)
    {
        value = transformNormal (current->transform, value);
        current = findObject (scene, current->parent);
    }
    return normalize (value);
}

std::uint8_t quantizeUnit (float value) noexcept
{
    return static_cast<std::uint8_t> (
        std::lround (std::clamp (value, 0.0f, 1.0f) * 255.0f));
}

SceneVec3 vertexLighting (const Visual3DScene& scene,
                          SceneVec3 worldPosition,
                          SceneVec3 worldNormal) noexcept
{
    auto lighting = scene.ambientColor;
    for (std::size_t index = 0; index < scene.lightCount; ++index)
    {
        const auto& light = scene.lights[index];
        SceneVec3 toLight {};
        float attenuation = 1.0f;
        if (light.kind == SceneLightKind::Directional)
        {
            const auto rayDirection = normalize (
                rotate (light.transform.rotation, { 0.0f, 0.0f, -1.0f }));
            toLight = multiply (rayDirection, -1.0f);
        }
        else
        {
            const auto offset = subtract (light.transform.translation, worldPosition);
            const auto distanceSquared = dot (offset, offset);
            toLight = normalize (offset);
            attenuation = 1.0f / (1.0f + distanceSquared);
            if (light.range > 0.0f)
            {
                const auto distance = std::sqrt (distanceSquared);
                attenuation *= std::clamp (1.0f - distance / light.range, 0.0f, 1.0f);
            }
        }

        const auto lambert = std::max (0.0f, dot (worldNormal, toLight));
        lighting = add (lighting,
                        multiply (light.color, light.intensity * attenuation * lambert));
    }
    return lighting;
}

SceneVec3 sampleBaseColor (const Visual3DScene& scene,
                           const SceneMaterialRecord& material,
                           SceneVec2 uv) noexcept
{
    if (! material.baseColorTexture.isValid())
        return { 1.0f, 1.0f, 1.0f };

    const auto* texture = findTexture (scene, material.baseColorTexture);
    const auto wrappedU = uv.x - std::floor (uv.x);
    const auto wrappedV = uv.y - std::floor (uv.y);
    const auto x = std::min (static_cast<std::uint32_t> (wrappedU * texture->width),
                             texture->width - 1);
    const auto y = std::min (static_cast<std::uint32_t> (wrappedV * texture->height),
                             texture->height - 1);
    const auto& texel = scene.textureTexels[texture->firstTexel + y * texture->width + x];
    constexpr auto inverseByte = 1.0f / 255.0f;
    return { texel.red * inverseByte,
             texel.green * inverseByte,
             texel.blue * inverseByte };
}

ProjectedVertex projectVertex (SceneVec3 cameraPosition,
                               SceneVec2 uv,
                               SceneVec3 lighting,
                               SceneVec3 worldNormal,
                               const SceneCameraRecord& camera,
                               std::uint32_t width,
                               std::uint32_t height) noexcept
{
    ProjectedVertex result;
    const auto distance = -cameraPosition.z;
    if (distance < camera.nearPlane || distance > camera.farPlane)
        return result;

    const auto tangent = std::tan (camera.verticalFovRadians * 0.5f);
    const auto aspect = static_cast<float> (width) / static_cast<float> (height);
    const auto ndcX = cameraPosition.x / (distance * tangent * aspect);
    const auto ndcY = cameraPosition.y / (distance * tangent);
    if (! std::isfinite (ndcX) || ! std::isfinite (ndcY))
        return result;

    const auto screenX = (ndcX * 0.5f + 0.5f) * static_cast<float> (width - 1);
    const auto screenY = (0.5f - ndcY * 0.5f) * static_cast<float> (height - 1);
    const auto normalizedDepth = (distance - camera.nearPlane)
                               / (camera.farPlane - camera.nearPlane);
    result.x = static_cast<std::int64_t> (
        std::llround (screenX * static_cast<float> (kSubpixelScale)));
    result.y = static_cast<std::int64_t> (
        std::llround (screenY * static_cast<float> (kSubpixelScale)));
    result.depth = static_cast<std::uint32_t> (
        std::llround (std::clamp (normalizedDepth, 0.0f, 1.0f)
                      * static_cast<float> (kFarDepth24 - 1)));
    result.uv = uv;
    result.lighting = lighting;
    result.worldNormal = worldNormal;
    result.visible = true;
    return result;
}

std::int64_t edge (const ProjectedVertex& a, const ProjectedVertex& b,
                   std::int64_t x, std::int64_t y) noexcept
{
    return (b.x - a.x) * (y - a.y) - (b.y - a.y) * (x - a.x);
}

int pixelFloor (std::int64_t value) noexcept
{
    if (value >= 0)
        return static_cast<int> (value / kSubpixelScale);
    return static_cast<int> (- ((-value + kSubpixelScale - 1) / kSubpixelScale));
}

float interpolate (float first, float second, float third,
                   std::int64_t weight0, std::int64_t weight1,
                   std::int64_t weight2, std::int64_t area) noexcept
{
    return (static_cast<float> (weight0) * first
            + static_cast<float> (weight1) * second
            + static_cast<float> (weight2) * third)
           / static_cast<float> (area);
}

void rasterizeTriangle (const ProjectedVertex& first,
                        const ProjectedVertex& second,
                        const ProjectedVertex& third,
                        const Visual3DScene& scene,
                        const SceneObjectRecord& object,
                        const SceneMaterialRecord& material,
                        RenderResult& result) noexcept
{
    auto area = edge (first, second, third.x, third.y);
    if (area == 0)
        return;
    const auto sign = area < 0 ? -1 : 1;
    area *= sign;

    const auto minimumX = std::min ({ first.x, second.x, third.x });
    const auto maximumX = std::max ({ first.x, second.x, third.x });
    const auto minimumY = std::min ({ first.y, second.y, third.y });
    const auto maximumY = std::max ({ first.y, second.y, third.y });
    const auto minX = std::max (0, pixelFloor (minimumX) - 1);
    const auto maxX = std::min (static_cast<int> (result.width) - 1,
                                pixelFloor (maximumX) + 1);
    const auto minY = std::max (0, pixelFloor (minimumY) - 1);
    const auto maxY = std::min (static_cast<int> (result.height) - 1,
                                pixelFloor (maximumY) + 1);
    if (minX > maxX || minY > maxY)
        return;

    ++result.rasterizedTriangles;
    for (auto y = minY; y <= maxY; ++y)
    {
        for (auto x = minX; x <= maxX; ++x)
        {
            const auto sampleX = static_cast<std::int64_t> (x) * kSubpixelScale
                               + kSubpixelCenter;
            const auto sampleY = static_cast<std::int64_t> (y) * kSubpixelScale
                               + kSubpixelCenter;
            const auto weight0 = edge (second, third, sampleX, sampleY) * sign;
            const auto weight1 = edge (third, first, sampleX, sampleY) * sign;
            const auto weight2 = edge (first, second, sampleX, sampleY) * sign;
            if (weight0 < 0 || weight1 < 0 || weight2 < 0)
                continue;

            const auto depthNumerator = weight0 * first.depth
                                      + weight1 * second.depth
                                      + weight2 * third.depth;
            const auto depth = static_cast<std::uint32_t> (depthNumerator / area);
            const auto pixelIndex = static_cast<std::size_t> (y) * result.width
                                  + static_cast<std::size_t> (x);
            if (depth >= result.depth24[pixelIndex])
                continue;

            if (result.depth24[pixelIndex] == kFarDepth24)
                ++result.coveredPixels;
            result.depth24[pixelIndex] = depth;

            const SceneVec2 uv {
                interpolate (first.uv.x, second.uv.x, third.uv.x,
                             weight0, weight1, weight2, area),
                interpolate (first.uv.y, second.uv.y, third.uv.y,
                             weight0, weight1, weight2, area)
            };
            const SceneVec3 lighting {
                interpolate (first.lighting.x, second.lighting.x, third.lighting.x,
                             weight0, weight1, weight2, area),
                interpolate (first.lighting.y, second.lighting.y, third.lighting.y,
                             weight0, weight1, weight2, area),
                interpolate (first.lighting.z, second.lighting.z, third.lighting.z,
                             weight0, weight1, weight2, area)
            };
            const auto textureColor = sampleBaseColor (scene, material, uv);
            const auto linearColor = add (
                multiply (multiply (material.baseColor, textureColor), lighting),
                material.emissive);
            const auto byteIndex = pixelIndex * 4;
            result.rgba[byteIndex] = quantizeUnit (linearColor.x);
            result.rgba[byteIndex + 1] = quantizeUnit (linearColor.y);
            result.rgba[byteIndex + 2] = quantizeUnit (linearColor.z);
            result.rgba[byteIndex + 3] = 255;

            const auto normal = normalize ({
                interpolate (first.worldNormal.x, second.worldNormal.x,
                             third.worldNormal.x, weight0, weight1, weight2, area),
                interpolate (first.worldNormal.y, second.worldNormal.y,
                             third.worldNormal.y, weight0, weight1, weight2, area),
                interpolate (first.worldNormal.z, second.worldNormal.z,
                             third.worldNormal.z, weight0, weight1, weight2, area)
            });
            const auto normalIndex = pixelIndex * 3;
            result.normalRgb8[normalIndex] = quantizeUnit (normal.x * 0.5f + 0.5f);
            result.normalRgb8[normalIndex + 1] = quantizeUnit (normal.y * 0.5f + 0.5f);
            result.normalRgb8[normalIndex + 2] = quantizeUnit (normal.z * 0.5f + 0.5f);
            result.objectIds[pixelIndex] = object.id.value;
            result.materialIds[pixelIndex] = material.id.value;
        }
    }
}
} // namespace

RenderResult render (const Visual3DScene& scene,
                     std::uint32_t width,
                     std::uint32_t height)
{
    RenderResult result;
    if (width == 0 || height == 0
        || width > kMaxFixtureExtent || height > kMaxFixtureExtent)
    {
        result.code = RenderCode::InvalidDimensions;
        return result;
    }

    result.sceneValidation = validateVisual3DScene (scene);
    if (! result.sceneValidation.valid())
    {
        result.code = RenderCode::InvalidScene;
        return result;
    }

    for (std::size_t index = 0; index < scene.materialCount; ++index)
    {
        if (scene.materials[index].opacity != 1.0f)
        {
            result.code = RenderCode::UnsupportedTransparentMaterial;
            return result;
        }
    }
    for (std::size_t index = 0; index < scene.textureTexelCount; ++index)
    {
        if (scene.textureTexels[index].alpha != 255)
        {
            result.code = RenderCode::UnsupportedTransparentMaterial;
            return result;
        }
    }

    const auto* camera = visual3d_detail::findById (
        scene.cameras, scene.cameraCount, scene.activeCamera);
    result.width = width;
    result.height = height;
    const auto pixelCount = static_cast<std::size_t> (width) * height;
    result.rgba.resize (pixelCount * 4);
    result.depth24.assign (pixelCount, kFarDepth24);
    result.normalRgb8.assign (pixelCount * 3, 0);
    result.objectIds.assign (pixelCount, 0);
    result.materialIds.assign (pixelCount, 0);
    for (std::size_t pixel = 0; pixel < pixelCount; ++pixel)
        for (std::size_t channel = 0; channel < kBackground.size(); ++channel)
            result.rgba[pixel * 4 + channel] = kBackground[channel];

    std::array<ProjectedVertex, Visual3DScene::kMaxVertices> projected {};
    for (std::size_t objectIndex = 0; objectIndex < scene.objectCount; ++objectIndex)
    {
        const auto& object = scene.objects[objectIndex];
        const auto* material = findMaterial (scene, object.material);
        for (std::size_t local = 0; local < object.vertexCount; ++local)
        {
            const auto& vertex = scene.vertices[object.firstVertex + local];
            const auto worldPosition = objectPointToWorld (scene, object, vertex.position);
            const auto worldNormal = objectNormalToWorld (scene, object, vertex.normal);
            const auto lighting = vertexLighting (scene, worldPosition, worldNormal);
            auto cameraPosition = subtract (worldPosition, camera->transform.translation);
            cameraPosition = inverseRotate (camera->transform.rotation, cameraPosition);
            projected[local] = projectVertex (cameraPosition, vertex.uv, lighting,
                                              worldNormal, *camera, width, height);
        }

        for (std::size_t item = object.firstIndex;
             item < object.firstIndex + object.indexCount; item += 3)
        {
            const auto& first = projected[scene.indices[item]];
            const auto& second = projected[scene.indices[item + 1]];
            const auto& third = projected[scene.indices[item + 2]];
            if (first.visible && second.visible && third.visible)
                rasterizeTriangle (first, second, third, scene, object,
                                   *material, result);
        }
    }

    result.code = RenderCode::Rendered;
    return result;
}

} // namespace videohelper::fixture3d::reference
