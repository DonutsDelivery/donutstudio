#pragma once

// Native diffraction transport validation of a surface's material uv.
// Both the OpenGL/Metal compiler and the geometry admission tests include
// this, so the renderer's refusal and the unit test share one predicate.

#include "../../shared/Visual3DScene.h"

#include <cmath>
#include <cstdint>

namespace videorender::fixture3d
{

inline bool hasValidUvDerivatives (const HarmonicMIDI::grid::Visual3DScene& scene,
                                   const HarmonicMIDI::grid::SceneObjectRecord& object) noexcept
{
    if (object.indexCount == 0 || object.indexCount % 3u != 0u)
        return false;
    for (std::uint32_t index = 0; index < object.indexCount; index += 3u)
    {
        const auto first = scene.indices[object.firstIndex + index];
        const auto second = scene.indices[object.firstIndex + index + 1u];
        const auto third = scene.indices[object.firstIndex + index + 2u];
        if (first >= object.vertexCount || second >= object.vertexCount
            || third >= object.vertexCount)
            return false;
        const auto& a = scene.vertices[object.firstVertex + first].uv;
        const auto& b = scene.vertices[object.firstVertex + second].uv;
        const auto& c = scene.vertices[object.firstVertex + third].uv;
        const auto determinant = (b.x - a.x) * (c.y - a.y)
                               - (b.y - a.y) * (c.x - a.x);
        if (! std::isfinite (determinant) || std::abs (determinant) <= 1.0e-8f)
            return false;
    }
    return true;
}

inline bool hasUnitUvDomain (const HarmonicMIDI::grid::Visual3DScene& scene,
                             const HarmonicMIDI::grid::SceneObjectRecord& object) noexcept
{
    for (std::uint32_t index = 0; index < object.vertexCount; ++index)
    {
        const auto& uv = scene.vertices[object.firstVertex + index].uv;
        if (! std::isfinite (uv.x) || ! std::isfinite (uv.y)
            || uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f)
            return false;
    }
    return true;
}

} // namespace videorender::fixture3d
