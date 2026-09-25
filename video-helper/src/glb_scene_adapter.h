#pragma once

#include "gltf_glb.h"
#include "../../shared/Visual3DScene.h"
#include "../../shared/VisualAnimationPlaybackControl.h"

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

// Compose each draw's static node hierarchy into world TRS while preserving
// object/material identities. Callers opt into bounded multi-draw composition.
std::optional<HarmonicMIDI::grid::Visual3DScene> adaptAnimatedGlbMeshToVisual3DScene(
    const GlbStaticMeshDocument& asset,
    std::string& error,
    std::optional<std::size_t> meshIndex = std::nullopt,
    bool multipleDraws = false);

bool sampleGlbCameraLights(const GlbStaticMeshDocument& asset,
    const visualanimation::Sample& sample, visualanimation::CombinationMode mode, float weight,
    const HarmonicMIDI::grid::Visual3DScene& scene,
    std::vector<HarmonicMIDI::grid::SceneCameraRecord>& cameras,
    std::vector<HarmonicMIDI::grid::SceneLightRecord>& lights, std::string& error);
} // namespace videohelper::gltf
