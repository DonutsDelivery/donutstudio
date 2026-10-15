#pragma once

#include "gltf_glb.h"
#include "../../shared/ImportedLightControl.h"
#include "../../shared/Visual3DScene.h"
#include <algorithm>
#include <string>
#include <vector>
#include <utility>

namespace videohelper::importedscene
{
using LightDefinitions = std::map<std::uint32_t, std::uint32_t>;

inline LightDefinitions importedLightDefinitions(const gltf::GlbStaticMeshDocument& asset,
                                                 const HarmonicMIDI::grid::Visual3DScene& scene)
{
    LightDefinitions definitions;
    for (std::size_t i = 0; i < scene.lightCount; ++i)
    {
        const auto instance = scene.lights[i].id.value;
        if (instance == 0 || instance > asset.nodes.size()) continue;
        const auto definition = asset.nodes[instance - 1u].light;
        if (definition && *definition < asset.lights.size())
            definitions.emplace(instance, static_cast<std::uint32_t>(*definition + 1u));
    }
    return definitions;
}

// Resolve after scene animation. GPU preparation continues to own the original
// immutable snapshot; both native backends consume these frame light records.
inline bool evaluateImportedLights(const importedlightcontrol::Controls& controls,
    const LightDefinitions& definitions, const HarmonicMIDI::grid::Visual3DScene& scene,
    std::vector<HarmonicMIDI::grid::SceneLightRecord>& frameLights, std::string& error)
{
    if (!importedlightcontrol::valid(controls))
    { error = "Imported light intensity controls are malformed"; return false; }
    if (controls.empty()) return true;
    if (controls.selectedDefinition != 0 && std::none_of(definitions.begin(), definitions.end(),
        [&](const auto& entry) { return entry.second == controls.selectedDefinition; }))
    { error = "Selected imported light definition is missing from the exact scene"; return false; }
    if (controls.intensityMultipliers.empty()) return true;
    auto candidate = frameLights;
    if (candidate.empty()) candidate.assign(scene.lights.begin(), scene.lights.begin() + scene.lightCount);
    if (candidate.size() != scene.lightCount)
    { error = "Imported light frame does not match its retained scene"; return false; }
    for (std::size_t i = 0; i < candidate.size(); ++i)
    {
        auto& light = candidate[i];
        if (light.id != scene.lights[i].id)
        { error = "Imported light frame identity does not match its retained scene"; return false; }
        const auto definition = definitions.find(light.id.value);
        if (definition == definitions.end()) continue;
        const auto multiplier = controls.intensityMultipliers.find(definition->second);
        if (multiplier == controls.intensityMultipliers.end()) continue;
        const double intensity = static_cast<double>(light.intensity) * multiplier->second;
        if (!std::isfinite(intensity) || intensity < 0 || intensity > 1000000.0)
        { error = "Imported light intensity exceeds the native scene bound"; return false; }
        light.intensity = static_cast<float>(intensity);
    }
    frameLights = std::move(candidate);
    return true;
}
} // namespace videohelper::importedscene
