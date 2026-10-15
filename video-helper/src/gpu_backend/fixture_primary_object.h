#pragma once

#include "../../../shared/Visual3DScene.h"

#include <cstddef>

namespace arbitgpu::fixtureprimary
{
// Non-diffraction fixture renders take their material and initial world
// transform from the first object that draws geometry. A GLB import emits
// transform-only ancestors before the meshes they parent. Those ancestors
// have no vertices or indices and reference material 0, which never
// resolves, so objects[0] is often not drawable.
inline const HarmonicMIDI::grid::SceneObjectRecord* firstDrawableObject (
    const HarmonicMIDI::grid::Visual3DScene& scene) noexcept
{
    for (std::size_t index = 0;
         index < scene.objectCount && index < scene.objects.size(); ++index)
        if (scene.objects[index].indexCount > 0)
            return &scene.objects[index];
    return nullptr;
}

// Null when there is no object or when its material id does not resolve.
inline const HarmonicMIDI::grid::SceneMaterialRecord* materialOf (
    const HarmonicMIDI::grid::Visual3DScene& scene,
    const HarmonicMIDI::grid::SceneObjectRecord* object) noexcept
{
    if (object == nullptr)
        return nullptr;
    return HarmonicMIDI::grid::visual3d_detail::findById (
        scene.materials, scene.materialCount, object->material);
}
} // namespace arbitgpu::fixtureprimary
