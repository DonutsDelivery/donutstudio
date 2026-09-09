#pragma once

#include "gltf_glb.h"
#include "Visual3DScene.h"

#include <optional>
#include <string>

namespace videohelper::gltf
{
// Copies the selected scene from an already admitted, in-memory static GLB
// document into the fixed-capacity renderer value contract. The returned scene
// owns every translated byte and keeps no reference to the source document.
std::optional<HarmonicMIDI::grid::Visual3DScene> adaptStaticGlbToVisual3DScene(
    const GlbStaticMeshDocument& asset,
    std::string& error,
    std::optional<std::size_t> meshIndex = std::nullopt);
} // namespace videohelper::gltf
