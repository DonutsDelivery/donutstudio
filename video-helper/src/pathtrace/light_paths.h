#pragma once

// The Render node's light paths (shared/PathTraceLightPaths.h) applied to the
// integrator. A kind limit at its top step is the integrator's unlimited
// value; no value refuses a frame.

#include "integrator.h"
#include "../../../shared/PathTraceLightPaths.h"

namespace videohelper::pathtrace
{
inline std::uint32_t kindBounceLimit (std::uint16_t value) noexcept
{
    return value >= visualimportedscenerender::kMaximumPathTraceKindBounces ? kUnlimitedBounces : value;
}

inline void applyLightPaths (const visualimportedscenerender::PathTraceLightPaths& paths,
                             IntegratorSettings& settings) noexcept
{
    settings.maxDiffuseBounces = kindBounceLimit (paths.diffuseBounces);
    settings.maxGlossyBounces = kindBounceLimit (paths.glossyBounces);
    settings.maxTransmissionBounces = kindBounceLimit (paths.transmissionBounces);
    settings.maxTransparentHits = paths.transparentBounces;
    const auto amount = [] (float value) { return visualimportedscenerender::validLightPathAmount (value) ? value : 0.0f; };
    settings.directClamp = amount (paths.directClamp);
    settings.indirectClamp = amount (paths.indirectClamp);
    settings.filterGlossy = amount (paths.filterGlossy);
    settings.reflectiveCaustics = paths.reflectiveCaustics;
    settings.refractiveCaustics = paths.refractiveCaustics;
}
} // namespace videohelper::pathtrace
