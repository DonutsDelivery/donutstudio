#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <map>

namespace importedlightcontrol
{
inline constexpr const char* kParameter = "selectedLightIntensity";
// Light definition IDs span the 32-bit range; a scene has as many lights as it imports.
inline constexpr std::uint32_t kMaximumDefinitions = std::numeric_limits<std::uint32_t>::max();
// The knob and slider cover 0 to 64. A stored or typed multiplier only has to stay a finite float.
inline constexpr float kKnobMaximumMultiplier = 64.0f;
inline constexpr float kMaximumMultiplier = std::numeric_limits<float>::max();

// Definition IDs are one-based KHR_lights_punctual.lights[] indices. The
// enclosing imported-scene request supplies the exact asset id/version/SHA.
struct Controls
{
    std::uint32_t selectedDefinition = 0;
    std::map<std::uint32_t, float> intensityMultipliers;

    bool empty() const noexcept
    { return selectedDefinition == 0 && intensityMultipliers.empty(); }
    float selectedMultiplier() const noexcept
    {
        const auto found = intensityMultipliers.find(selectedDefinition);
        return found == intensityMultipliers.end() ? 1.0f : found->second;
    }
};

inline bool valid(const Controls& controls) noexcept
{
    if (controls.selectedDefinition > kMaximumDefinitions
        || controls.intensityMultipliers.size() > kMaximumDefinitions) return false;
    for (const auto& [definition, multiplier] : controls.intensityMultipliers)
        if (definition == 0 || definition > kMaximumDefinitions
            || !std::isfinite(multiplier) || multiplier < 0
            || multiplier > kMaximumMultiplier || std::signbit(multiplier)) return false;
    return true;
}
inline bool same(const Controls& left, const Controls& right) noexcept
{
    return left.selectedDefinition == right.selectedDefinition
        && left.intensityMultipliers == right.intensityMultipliers;
}
} // namespace importedlightcontrol
