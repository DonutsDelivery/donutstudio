#pragma once

#include "Visual3DScene.h"
#include "VisualImportedSceneRenderOperationContract.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace videohelper::fixture3d
{

inline constexpr HarmonicMIDI::grid::Scene3DId kSceneId { 1 };
inline constexpr HarmonicMIDI::grid::SceneObjectId kCubeObjectId { 101 };
inline constexpr HarmonicMIDI::grid::SceneMaterialId kCubeMaterialId { 201 };
inline constexpr HarmonicMIDI::grid::SceneTextureId kCubeTextureId { 251 };
inline constexpr HarmonicMIDI::grid::SceneLightId kKeyLightId { 301 };
inline constexpr HarmonicMIDI::grid::SceneCameraId kCameraId { 401 };
inline constexpr std::uint32_t kFixtureVersion = 1;

// Small authored cube retained for generic backend checks.
HarmonicMIDI::grid::Visual3DScene makeScene() noexcept;

struct HolographicTradingCardScene final
{
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> scene;
    std::string assetId;
    std::string contentSha256;
    std::string sceneName;
    std::string objectName;
    std::string materialName;
    std::string cameraName;
    std::string lightName;
    std::string encodedOperation;
    visualimportedscenerender::Request operation;
};

// Decodes the exact embedded GLB with the production decoder, then copies it
// through the production imported-scene adapter. No fallback geometry exists.
std::optional<HolographicTradingCardScene>
loadHolographicTradingCardScene(std::string& error);

} // namespace videohelper::fixture3d
