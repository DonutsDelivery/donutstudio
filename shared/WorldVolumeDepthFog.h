#pragma once

// Raster and Realtime march a scene's fog with shadowed light shafts
// (rastermedium::worldFogForScene in RasterMedium.h). This maps the same fog
// onto a Depth Fog pass instead: the fog reaches 95%
// at three mean free paths (3 / density scene units) from the camera, measured
// over the camera's near-to-far span, and takes the medium's scatter colour.
// Depth Fog has no density field, so Noise fog draws at the field's mean
// density, half its peak (kNoiseFogMeanDensity).

#include "RenderPassCompositeContract.h"
#include "Visual3DScene.h"

#include <algorithm>
#include <cmath>
#include <optional>

namespace renderpasscomposite
{
inline constexpr double kNoiseFogMeanDensity = 0.5;

inline std::optional<Parameters> depthFogForWorldVolume (const HarmonicMIDI::grid::SceneVolume& volume,
                                                         float nearPlane, float farPlane) noexcept
{
    if (! HarmonicMIDI::grid::sceneVolumeActive (volume) || ! std::isfinite (nearPlane)
        || ! std::isfinite (farPlane) || ! (farPlane > nearPlane))
        return std::nullopt;
    const double span = static_cast<double> (farPlane) - static_cast<double> (nearPlane);
    const double density = static_cast<double> (volume.density)
        * (volume.field == HarmonicMIDI::grid::SceneVolumeField::Noise ? kNoiseFogMeanDensity : 1.0);
    const double reach = 3.0 / (density * span);
    Parameters fog;
    fog.mode = Mode::DepthFog;
    fog.nearDepth = 0.0f;
    if (reach >= 1.0)
    {
        // Thin fog never reaches 95% inside the far plane: it holds its true
        // opacity at the far plane instead.
        fog.farDepth = 1.0f;
        fog.amount = static_cast<float> (1.0 - std::exp (-density * span));
    }
    else
    {
        fog.farDepth = std::max (static_cast<float> (reach), 1.0e-6f);
        fog.amount = 0.95f;
    }
    const auto unit = [] (float value) { return std::isfinite (value) ? std::min (1.0f, std::max (0.0f, value)) : 0.0f; };
    fog.fogColor = { unit (volume.color.x), unit (volume.color.y), unit (volume.color.z) };
    fog.fogColorSpace = ColorInput::Linear;
    if (! valid (fog)) return std::nullopt;
    return fog;
}

// The fog a Raster or Realtime frame draws: the render's own fog (the Render
// node's or the imported scene's) over the scene's world volume, measured over
// the active camera's span.
inline std::optional<Parameters> depthFogForScene (const HarmonicMIDI::grid::Visual3DScene& scene,
                                                   const HarmonicMIDI::grid::SceneVolume* renderFog) noexcept
{
    const auto* camera = HarmonicMIDI::grid::visual3d_detail::findById (scene.cameras, scene.cameraCount,
                                                                       scene.activeCamera);
    if (camera == nullptr) return std::nullopt;
    return depthFogForWorldVolume (renderFog != nullptr ? *renderFog : scene.worldVolume,
                                   camera->nearPlane, camera->farPlane);
}

// Whether a material carries a medium: glass or water media and Volume
// Materials. OpenGL Raster and Realtime draw them (RasterMedium.h); Metal
// Raster reports them.
inline bool sceneHasMaterialMedia (const HarmonicMIDI::grid::Visual3DScene& scene) noexcept
{
    const auto count = std::min<std::size_t> (scene.materialCount, scene.materials.size());
    for (std::size_t index = 0; index < count; ++index)
        if (HarmonicMIDI::grid::sceneVolumeActive (scene.materials[index].volume))
            return true;
    return false;
}
} // namespace renderpasscomposite
