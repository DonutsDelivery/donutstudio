#pragma once

#include "../../../shared/Visual3DScene.h"

#include <algorithm>
#include <cstddef>
#include <string>

// Sizes of the Metal fixture shaders' fixed arrays and uniform blocks. OpenGL Raster
// binds these from the scene. Metal has not been converted yet: its shaders still
// index these arrays by scene record, so this guard stops a scene that would write
// past them and names what did not fit. Path Trace takes the scene whole on Metal.
// Vertices and indices are not among them: they reach the shaders through vertex and
// index buffers sized from the scene, so no vertex or index count is refused.
namespace metalbinding
{
inline constexpr std::size_t kObjects = 64;
inline constexpr std::size_t kMaterials = 32;
inline constexpr std::size_t kTextures = 16;
inline constexpr std::size_t kTextureTexels = 1024 * 1024;
inline constexpr std::size_t kTextureDimension = 8192;
inline constexpr std::size_t kLights = 16;
inline constexpr std::size_t kCameras = 8;
inline constexpr std::size_t kObjectInstances = 128;
// The closest-hit tables the Metal census charges: every object under every note instance.
inline constexpr std::size_t kQueryInstances = kObjects * 128;
inline constexpr std::size_t kQueryTexels = 1024 * 2048;

inline bool sceneIsBound (const HarmonicMIDI::grid::Visual3DScene& scene, std::string& error)
{
    const auto check = [&error] (const char* name, std::size_t count, std::size_t binding)
    {
        if (count <= binding)
            return true;
        error = "Metal fixture shader array holds " + std::to_string (binding) + " " + name
            + " and this scene has " + std::to_string (count)
            + ". Path Trace renders this scene; the Metal shader conversion is its own slice.";
        return false;
    };
    if (! check ("objects", scene.objectCount, kObjects)
        || ! check ("materials", scene.materialCount, kMaterials)
        || ! check ("textures", scene.textureCount, kTextures)
        || ! check ("texture texels", scene.textureTexelCount, kTextureTexels)
        || ! check ("lights", scene.lightCount, kLights)
        || ! check ("cameras", scene.cameraCount, kCameras)
        || ! check ("object instances", scene.objectInstanceCount, kObjectInstances))
        return false;
    for (std::size_t index = 0; index < scene.textureCount; ++index)
        if (! check ("texels per texture axis", std::max<std::size_t> (scene.textures[index].width, scene.textures[index].height), kTextureDimension))
            return false;
    return true;
}
}
