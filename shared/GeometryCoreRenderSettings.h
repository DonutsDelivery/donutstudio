#pragma once

// The Render 3D settings of Geometry wired straight into Render 3D. The direct
// route lowers no retained scene, so its lowered plan (GeometryCoreTransport.h,
// version 4) carries them beside the Surface binding and slot media. The helper
// hands the Geometry Core scene it evaluates each frame to the chosen engine.
// A plan whose settings are Raster and Image/Color with no Render 3D Camera
// or Light carries none, so its bytes stay those of a plan written before
// these settings existed. Version 2 adds the lighting rig after the version 1
// fields, written only when Render 3D has a Camera or Light wired.

#include "VisualImportedSceneRenderOperationContract.h"
#include "RenderPassOutputContract.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace videowire::geometry
{
struct RenderSettings final
{
    visualimportedscenerender::RenderEngine engine = visualimportedscenerender::RenderEngine::Raster;
    visualimportedscenerender::RenderMode mode = visualimportedscenerender::RenderMode::Accurate;
    renderpassoutput::Output imageOutput = renderpassoutput::Output::Color;
    std::uint16_t exportSamples = visualimportedscenerender::kDefaultPathTraceExportSamples;
    std::uint16_t previewSamples = visualimportedscenerender::kDefaultPathTracePreviewSamples;
    visualimportedscenerender::PathTraceDenoise denoise {};
    std::uint16_t maxBounces = visualimportedscenerender::kDefaultPathTraceMaxBounces;
    std::uint8_t volume = visualimportedscenerender::kDefaultPathTraceVolume;
    visualimportedscenerender::PathTraceLightPaths lightPaths {};
    visualimportedscenerender::ViewTransform viewTransform = visualimportedscenerender::ViewTransform::Standard;
    std::uint16_t motionBlurShutter = 0;
    visualimportedscenerender::MotionBlurPosition motionBlurPosition = visualimportedscenerender::MotionBlurPosition::Centre;
    // Render 3D's Camera and Light inputs, lowered over defaultLighting as the
    // retained scene route lowers them over its scene. The ambient color, fog,
    // lights, cameras and active camera replace the helper's. When Render 3D.Scene
    // is a Scene Compose of the Geometry Scene and static scene objects (a floor),
    // the rig is that composition lowered without the Geometry branch, and its
    // objects, materials, vertices and indices join every frame's scene
    // (appendStage), so each engine lights and shadows them with the geometry.
    // Null when neither input is wired, and the helper draws defaultLighting.
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> lighting;
};

// The key light, camera and ambient color a Geometry scene draws with when
// Render 3D has no Camera or Light wired. The scene must hold one light and
// one camera of capacity.
// Camera-facing key light: toLight = (-0.299, 0.399, 0.867), up and toward the viewer. Every
// camera-facing starter surface then gets a positive NdotL whatever its tilt, so displaced
// slopes shade instead of staying at the flat ambient-plus-emission floor. The previous
// direction raked from the upper right and left most front faces unlit.
inline void defaultLighting(HarmonicMIDI::grid::Visual3DScene& scene)
{
    using namespace HarmonicMIDI::grid;
    scene.activeCamera = {1};
    scene.ambientColor = {0.12f, 0.12f, 0.12f};
    scene.lights[0] = {};
    scene.lights[0].id = {1};
    scene.lights[0].kind = SceneLightKind::Directional;
    scene.lights[0].transform.rotation = {-0.2064f, -0.1548f, 0.0f, 0.9661f};
    scene.lights[0].intensity = 1.5f;
    scene.lightCount = 1;
    scene.cameras[0] = {};
    scene.cameras[0].id = {1};
    scene.cameras[0].transform.translation = {0.0f, 0.0f, 5.0f};
    scene.cameras[0].verticalFovRadians = 0.7853981634f;
    scene.cameras[0].nearPlane = 0.1f;
    scene.cameras[0].farPlane = 1000.0f;
    scene.cameraCount = 1;
}

// Copies a lighting rig's ambient color, fog, lights and cameras into a scene.
inline bool applyLighting(const HarmonicMIDI::grid::Visual3DScene& lighting,
                          HarmonicMIDI::grid::Visual3DScene& scene, std::string& error)
{
    using namespace HarmonicMIDI::grid;
    Visual3DSceneGrowth growth;
    growth.lights = lighting.lightCount > scene.lightCount ? lighting.lightCount - scene.lightCount : 0;
    growth.cameras = lighting.cameraCount > scene.cameraCount ? lighting.cameraCount - scene.cameraCount : 0;
    if (!growVisual3DScene(scene, growth, error)) return false;
    for (std::size_t i = 0; i < scene.lightCount; ++i) scene.lights[i] = {};
    for (std::size_t i = 0; i < scene.cameraCount; ++i) scene.cameras[i] = {};
    for (std::size_t i = 0; i < lighting.lightCount; ++i) scene.lights[i] = lighting.lights[i];
    for (std::size_t i = 0; i < lighting.cameraCount; ++i) scene.cameras[i] = lighting.cameras[i];
    scene.lightCount = lighting.lightCount;
    scene.cameraCount = lighting.cameraCount;
    scene.activeCamera = lighting.activeCamera;
    scene.ambientColor = lighting.ambientColor;
    scene.worldVolume = lighting.worldVolume;
    return true;
}

// Appends a rig's static objects (the stage) after the scene's own. Object and
// material identities move past the scene's largest, so they never collide with
// instance identities; parents, materials and a camera focus follow them.
// A rig without objects leaves the scene unchanged.
inline bool appendStage(const HarmonicMIDI::grid::Visual3DScene& lighting,
                        HarmonicMIDI::grid::Visual3DScene& scene, std::string& error)
{
    using namespace HarmonicMIDI::grid;
    if (lighting.objectCount == 0) return true;
    if (lighting.textureCount != 0 || lighting.instancedObject.isValid())
    { error = "Render 3D stage objects carry no textures or instanced draws"; return false; }
    std::uint32_t objectBase = 0, materialBase = 0;
    for (std::size_t i = 0; i < scene.objectCount; ++i) objectBase = std::max(objectBase, scene.objects[i].id.value);
    for (std::size_t i = 0; i < scene.materialCount; ++i) materialBase = std::max(materialBase, scene.materials[i].id.value);
    std::uint32_t stageObjects = 0, stageMaterials = 0;
    for (std::size_t i = 0; i < lighting.objectCount; ++i) stageObjects = std::max(stageObjects, lighting.objects[i].id.value);
    for (std::size_t i = 0; i < lighting.materialCount; ++i) stageMaterials = std::max(stageMaterials, lighting.materials[i].id.value);
    constexpr auto top = std::numeric_limits<std::uint32_t>::max();
    if (stageObjects > top - objectBase || stageMaterials > top - materialBase
        || lighting.vertexCount > top - scene.vertexCount || lighting.indexCount > top - scene.indexCount)
    { error = "Render 3D stage objects exhaust native scene identities"; return false; }
    Visual3DSceneGrowth growth;
    growth.vertices = lighting.vertexCount;
    growth.indices = lighting.indexCount;
    growth.objects = lighting.objectCount;
    growth.materials = lighting.materialCount;
    if (!growVisual3DScene(scene, growth, error)) return false;
    const auto firstObject = scene.objectCount;
    const auto firstVertex = static_cast<std::uint32_t>(scene.vertexCount);
    const auto firstIndex = static_cast<std::uint32_t>(scene.indexCount);
    for (std::size_t i = 0; i < lighting.vertexCount; ++i) scene.vertices[scene.vertexCount++] = lighting.vertices[i];
    for (std::size_t i = 0; i < lighting.indexCount; ++i) scene.indices[scene.indexCount++] = lighting.indices[i];
    for (std::size_t i = 0; i < lighting.materialCount; ++i)
    {
        auto material = lighting.materials[i];
        material.id = {material.id.value + materialBase};
        scene.materials[scene.materialCount++] = material;
    }
    for (std::size_t i = 0; i < lighting.objectCount; ++i)
    {
        auto object = lighting.objects[i];
        object.id = {object.id.value + objectBase};
        if (object.parent.isValid()) object.parent = {object.parent.value + objectBase};
        if (object.material.isValid()) object.material = {object.material.value + materialBase};
        object.firstVertex += firstVertex;
        object.firstIndex += firstIndex;
        scene.objects[scene.objectCount++] = object;
    }
    // Motion keeps either no per-object prior or one per object; a static stage repeats its own.
    if (scene.previousObjectTransformCount != 0)
    {
        if (scene.previousObjectTransformCount != firstObject)
        { error = "Render 3D stage objects need one prior transform per scene object"; return false; }
        for (std::size_t i = firstObject; i < scene.objectCount; ++i)
            scene.previousObjectTransforms[scene.previousObjectTransformCount++] = scene.objects[i].transform;
    }
    for (std::size_t i = 0; i < scene.cameraCount; ++i)
        if (scene.cameras[i].focusObject != 0)
            scene.cameras[i].focusObject += static_cast<std::uint32_t>(firstObject);
    if (!validateVisual3DScene(scene).valid())
    { error = "Render 3D stage objects produced an invalid native scene"; return false; }
    return true;
}

// Raster and Image/Color without a Camera or Light draw with the Geometry Core
// raster backend as before; every other choice takes the engine route.
inline bool rasterColor(const RenderSettings& settings) noexcept
{
    return settings.engine == visualimportedscenerender::RenderEngine::Raster
        && settings.imageOutput == renderpassoutput::Output::Color;
}
inline bool engineRoute(const RenderSettings& settings) noexcept
{
    return !rasterColor(settings) || settings.lighting != nullptr;
}

namespace rendersettings_detail
{
inline constexpr char kMagic[4] = {'G', 'C', 'R', 'S'};
inline constexpr std::uint8_t kVersion = 1;
inline constexpr std::uint8_t kLightingVersion = 2;

inline void put16(std::string& out, std::uint16_t value)
{
    out.push_back(static_cast<char>(value & 0xffu));
    out.push_back(static_cast<char>(value >> 8));
}
inline void putFloat(std::string& out, float value)
{
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof bits);
    put16(out, static_cast<std::uint16_t>(bits & 0xffffu));
    put16(out, static_cast<std::uint16_t>(bits >> 16));
}
struct Reader final
{
    const std::string& text;
    std::size_t at = 0;
    bool byte(std::uint8_t& value)
    {
        if (at >= text.size()) return false;
        value = static_cast<std::uint8_t>(text[at++]);
        return true;
    }
    bool u16(std::uint16_t& value)
    {
        std::uint8_t low = 0, high = 0;
        if (!byte(low) || !byte(high)) return false;
        value = static_cast<std::uint16_t>(low | (high << 8));
        return true;
    }
    bool f32(float& value)
    {
        std::uint16_t low = 0, high = 0;
        if (!u16(low) || !u16(high)) return false;
        const std::uint32_t bits = static_cast<std::uint32_t>(low) | (static_cast<std::uint32_t>(high) << 16);
        std::memcpy(&value, &bits, sizeof value);
        return true;
    }
};
} // namespace rendersettings_detail

// Empty for Raster and Image/Color without lighting, which the lowered plan
// then leaves out. Empty too for a lighting rig the scene codec refuses.
inline std::string encodeRenderSettings(const RenderSettings& settings)
{
    using namespace rendersettings_detail;
    if (!engineRoute(settings)) return {};
    std::vector<std::uint8_t> lighting;
    if (settings.lighting)
    {
        lighting = sceneaov::detail::encodeScene(*settings.lighting);
        if (lighting.empty() || lighting.size() > std::numeric_limits<std::uint32_t>::max()) return {};
    }
    std::string out(kMagic, sizeof kMagic);
    out.push_back(static_cast<char>(settings.lighting ? kLightingVersion : kVersion));
    out.push_back(static_cast<char>(settings.engine));
    out.push_back(static_cast<char>(settings.mode));
    out.push_back(static_cast<char>(settings.imageOutput));
    put16(out, settings.exportSamples);
    put16(out, settings.previewSamples);
    out.push_back(static_cast<char>(settings.denoise.denoiser));
    put16(out, settings.maxBounces);
    out.push_back(static_cast<char>(settings.volume));
    put16(out, settings.lightPaths.diffuseBounces);
    put16(out, settings.lightPaths.glossyBounces);
    put16(out, settings.lightPaths.transmissionBounces);
    put16(out, settings.lightPaths.transparentBounces);
    putFloat(out, settings.lightPaths.directClamp);
    putFloat(out, settings.lightPaths.indirectClamp);
    putFloat(out, settings.lightPaths.filterGlossy);
    out.push_back(static_cast<char>(settings.lightPaths.reflectiveCaustics ? 1 : 0));
    out.push_back(static_cast<char>(settings.lightPaths.refractiveCaustics ? 1 : 0));
    out.push_back(static_cast<char>(settings.viewTransform));
    put16(out, settings.motionBlurShutter);
    out.push_back(static_cast<char>(settings.motionBlurPosition));
    if (settings.lighting)
    {
        const auto size = static_cast<std::uint32_t>(lighting.size());
        put16(out, static_cast<std::uint16_t>(size & 0xffffu));
        put16(out, static_cast<std::uint16_t>(size >> 16));
        out.append(reinterpret_cast<const char*>(lighting.data()), lighting.size());
    }
    return out;
}

// Empty text decodes to Raster and Image/Color. Anything else must be one
// canonical encoding of an engine route.
inline std::optional<RenderSettings> decodeRenderSettings(const std::string& text, std::string& error)
{
    using namespace rendersettings_detail;
    RenderSettings settings;
    if (text.empty()) return settings;
    Reader r {text};
    std::uint8_t magic[4] {}, version = 0, engine = 0, mode = 0, output = 0, denoiser = 0, volume = 0,
        reflective = 0, refractive = 0, view = 0, position = 0;
    for (auto& byte : magic)
        if (!r.byte(byte)) { error = "Geometry Core render settings are truncated"; return std::nullopt; }
    if (std::memcmp(magic, kMagic, sizeof kMagic) != 0 || !r.byte(version)
        || (version != kVersion && version != kLightingVersion))
    { error = "Geometry Core render settings have an unknown version"; return std::nullopt; }
    auto& paths = settings.lightPaths;
    if (!r.byte(engine) || !r.byte(mode) || !r.byte(output) || !r.u16(settings.exportSamples)
        || !r.u16(settings.previewSamples) || !r.byte(denoiser) || !r.u16(settings.maxBounces) || !r.byte(volume)
        || !r.u16(paths.diffuseBounces) || !r.u16(paths.glossyBounces) || !r.u16(paths.transmissionBounces)
        || !r.u16(paths.transparentBounces) || !r.f32(paths.directClamp) || !r.f32(paths.indirectClamp)
        || !r.f32(paths.filterGlossy) || !r.byte(reflective) || !r.byte(refractive) || !r.byte(view)
        || !r.u16(settings.motionBlurShutter) || !r.byte(position))
    { error = "Geometry Core render settings have the wrong length"; return std::nullopt; }
    if (version == kLightingVersion)
    {
        std::uint16_t low = 0, high = 0;
        if (!r.u16(low) || !r.u16(high))
        { error = "Geometry Core render settings have the wrong length"; return std::nullopt; }
        const std::size_t size = static_cast<std::size_t>(low) | (static_cast<std::size_t>(high) << 16);
        if (size == 0 || size != text.size() - r.at)
        { error = "Geometry Core render settings have the wrong length"; return std::nullopt; }
        const std::vector<std::uint8_t> bytes(text.begin() + static_cast<std::ptrdiff_t>(r.at), text.end());
        auto lighting = std::make_shared<HarmonicMIDI::grid::Visual3DScene>();
        // A rig holds lights, cameras and untextured stage objects, and re-encodes to the same bytes.
        if (!sceneaov::detail::decodeScene(bytes, *lighting)
            || !HarmonicMIDI::grid::validateVisual3DScene(*lighting).valid()
            || (lighting->objectCount == 0 && (lighting->vertexCount != 0 || lighting->indexCount != 0
                                               || lighting->materialCount != 0))
            || lighting->textureCount != 0 || lighting->instancedObject.isValid() || lighting->cameraCount == 0
            || sceneaov::detail::encodeScene(*lighting) != bytes)
        { error = "Geometry Core render settings carry an invalid Render 3D lighting rig"; return std::nullopt; }
        settings.lighting = std::move(lighting);
        r.at = text.size();
    }
    if (r.at != text.size())
    { error = "Geometry Core render settings have the wrong length"; return std::nullopt; }
    using visualimportedscenerender::RenderEngine;
    if ((engine != static_cast<std::uint8_t>(RenderEngine::Raster) && engine != static_cast<std::uint8_t>(RenderEngine::PathTrace)
            && engine != static_cast<std::uint8_t>(RenderEngine::Realtime))
        || mode > 1 || output >= static_cast<std::uint8_t>(renderpassoutput::Output::Count)
        || denoiser > static_cast<std::uint8_t>(visualimportedscenerender::PathTraceDenoiser::Oidn)
        || reflective > 1 || refractive > 1 || view > 1
        || settings.motionBlurShutter > visualimportedscenerender::kMaximumMotionBlurShutter
        || position > static_cast<std::uint8_t>(visualimportedscenerender::MotionBlurPosition::End)
        || (settings.motionBlurShutter == 0 && position != static_cast<std::uint8_t>(visualimportedscenerender::MotionBlurPosition::Centre))
        || !visualimportedscenerender::valid(paths))
    { error = "Geometry Core render settings hold an unknown value"; return std::nullopt; }
    settings.engine = static_cast<RenderEngine>(engine);
    settings.mode = static_cast<visualimportedscenerender::RenderMode>(mode);
    settings.imageOutput = static_cast<renderpassoutput::Output>(output);
    settings.denoise = static_cast<visualimportedscenerender::PathTraceDenoiser>(denoiser);
    settings.volume = volume;
    paths.reflectiveCaustics = reflective != 0;
    paths.refractiveCaustics = refractive != 0;
    settings.viewTransform = static_cast<visualimportedscenerender::ViewTransform>(view);
    settings.motionBlurPosition = static_cast<visualimportedscenerender::MotionBlurPosition>(position);
    if (!engineRoute(settings)) { error = "Geometry Core render settings for Raster Image/Color are left out"; return std::nullopt; }
    error.clear();
    return settings;
}
} // namespace videowire::geometry
