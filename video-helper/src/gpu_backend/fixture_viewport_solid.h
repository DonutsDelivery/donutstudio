#pragma once

// Preview-only solid viewport shading shared by the OpenGL and Metal fixture
// renderers. X-ray draws every object blended, back to front by object origin,
// with both face sides and no depth writes.

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>
#include <vector>

namespace fixtureviewportsolid
{
    /** Alpha of every solid X-ray fragment. */
    inline constexpr float kXrayAlpha = 0.4f;

    /** Solid shading is on for this preview frame. `solid` is
        NativeImportedSceneRuntimeInputs::viewportSolid. */
    inline bool solidApplies (bool previewUse, const std::array<float, 4>& solid) noexcept
    {
        return previewUse && solid[0] > 0.5f;
    }

    /** X-ray blending is on. `effects` is viewportSolidEffects: cavity, X-ray. */
    inline bool xrayApplies (bool previewUse, const std::array<float, 4>& solid,
                             const std::array<float, 4>& effects) noexcept
    {
        return solidApplies (previewUse, solid) && effects[1] > 0.5f;
    }

    /** One object for the draw order: whether it blends and its squared
        distance from the camera. */
    struct DrawCandidate
    {
        bool blended = false;
        float distanceSquared = 0.0f;
    };

    /** Opaque objects first in scene order, then blended ones farthest first.
        Equal distances keep scene order. X-ray blends every object. */
    inline std::vector<std::size_t> drawOrder (const std::vector<DrawCandidate>& candidates, bool xray)
    {
        std::vector<std::size_t> order;
        std::vector<std::pair<float, std::size_t>> blended;
        order.reserve (candidates.size());
        for (std::size_t index = 0; index < candidates.size(); ++index)
        {
            if (! xray && ! candidates[index].blended)
                order.push_back (index);
            else
                blended.emplace_back (candidates[index].distanceSquared, index);
        }
        std::stable_sort (blended.begin(), blended.end(),
                          [] (const auto& left, const auto& right) { return left.first > right.first; });
        for (const auto& entry : blended)
            order.push_back (entry.second);
        return order;
    }
}
