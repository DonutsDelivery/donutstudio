#pragma once

#include "particle_parameters.h"

#include <cstddef>
#include <vector>

namespace videorender
{
// Planar timeline particles read their emitter anchors and coupled bodies
// from one float4 list sized to them: anchors first, then bodies. OpenGL
// uploads it as a texture kParticleTimelineDataWidth texels wide, wrapping
// entry i to (i % width, i / width). Metal binds it as a storage buffer.
inline constexpr int kParticleTimelineDataWidth = 1024;

struct ParticleTimelineData
{
    std::vector<float> values;   // RGBA float4 entries, padded to whole rows
    int anchors = 0, bodies = 0;
    int rows = 1;
};

// Bodies are read only in Coupled motion. An empty list still holds one
// zero row, so the texture and buffer always exist.
inline ParticleTimelineData packParticleTimelineData(const ParticleParams& params,
    const ParticleBodyState* bodies)
{
    ParticleTimelineData data;
    data.anchors = std::max(params.geometryCount, 0);
    data.bodies = bodies ? static_cast<int>(bodies->size()) : 0;
    const auto entries = static_cast<std::size_t>(data.anchors) + static_cast<std::size_t>(data.bodies);
    const auto width = static_cast<std::size_t>(kParticleTimelineDataWidth);
    data.rows = static_cast<int>(std::max<std::size_t>(1, (entries + width - 1) / width));
    data.values.assign(static_cast<std::size_t>(data.rows) * width * 4, 0.0f);
    std::size_t at = 0;
    const auto put = [&](const std::array<float, 4>& value)
    {
        std::copy(value.begin(), value.end(), data.values.begin() + static_cast<std::ptrdiff_t>(at * 4));
        ++at;
    };
    for (int i = 0; i < data.anchors; ++i) put(params.geometryAnchors[static_cast<std::size_t>(i)]);
    for (int i = 0; i < data.bodies; ++i) put((*bodies)[static_cast<std::size_t>(i)]);
    return data;
}
}
