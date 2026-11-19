#pragma once

#include "DiffractionLightingPlan.h"
#include "DiffractiveFoilIR.h"
#include "Visual3DScene.h"
#include "SurfaceMaterialBindingContract.h"

#include <cstdint>
#include <optional>
#include <string>

namespace diffractionmaterialbinding
{
inline constexpr std::uint32_t kWireVersion = 3;
inline constexpr std::uint32_t kEnvironmentWireVersion = 4;
inline constexpr std::uint32_t kGraphFrameWireVersion = 5;
inline constexpr std::uint32_t kSceneReceiverWireVersion = 6;

// Exact value-owned product emitted by one graph-authored diffraction terminal
// for one imported-scene object. This transport grants no renderer capability,
// backend object, shader source, or CPU fallback.
struct ImportedSceneDiffractionMaterialRequest final
{
    std::uint32_t version = kWireVersion;
    HarmonicMIDI::grid::Scene3DId scene {};
    std::uint64_t sceneRevision = 0;
    std::uint64_t structuralRevision = 0;
    std::uint64_t evaluationRevision = 0;
    std::uint64_t materialRevision = 0;
    HarmonicMIDI::grid::SceneObjectId object {};
    diffractionmaterial::Description material;
    diffractionmaterial::LightingDescription lighting;
    std::string structuralDigest;
    std::optional<diffractivefoil::Description> spatialFoil;
    // Same authored endpoint as surface materials. The scheduler supplies its
    // owned native Frame per evaluation; no texture handle enters this request.
    std::optional<surfacematerialbinding::TextureSlotBinding::GraphFrameEndpoint> graphFrame;
};
} // namespace diffractionmaterialbinding
