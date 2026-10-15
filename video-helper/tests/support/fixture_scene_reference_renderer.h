#pragma once

#include "Visual3DScene.h"

#include <cstdint>
#include <vector>

namespace videohelper::fixture3d::reference
{

enum class RenderCode : std::uint8_t
{
    Rendered = 0,
    InvalidDimensions,
    InvalidScene,
    UnsupportedTransparentMaterial
};

struct RenderResult
{
    RenderCode code = RenderCode::InvalidScene;
    HarmonicMIDI::grid::Visual3DSceneValidation sceneValidation {};
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t rasterizedTriangles = 0;
    std::uint32_t coveredPixels = 0;
    std::vector<std::uint8_t> rgba;
    std::vector<std::uint32_t> depth24;
    // World-space normals use RGB8 signed-normal encoding. IDs are zero on the
    // background and retain the stable scene record values on covered pixels.
    std::vector<std::uint8_t> normalRgb8;
    std::vector<std::uint32_t> objectIds;
    std::vector<std::uint32_t> materialIds;

    bool rendered() const noexcept
    {
        return code == RenderCode::Rendered;
    }
};

// Test-only deterministic pixel oracle. Production preview and export must use
// an admitted GPU backend. This function is not a CPU fallback.
RenderResult render (
    const HarmonicMIDI::grid::Visual3DScene& scene,
    std::uint32_t width,
    std::uint32_t height);

} // namespace videohelper::fixture3d::reference
