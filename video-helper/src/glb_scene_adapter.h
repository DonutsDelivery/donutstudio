#pragma once

#include "gltf_glb.h"
#include "../../shared/Visual3DScene.h"
#include "../../shared/VisualAnimationPlaybackControl.h"

#include <memory>
#include <optional>
#include <string>

namespace videohelper::gltf
{
#if defined(ARBIT_TEST_IMPORTED_PARENT_STAGING_FAILURES)
// Capacity pairs bracket reserve. After reserve the old range has retired;
// its overlap with the new allocation follows the selected vector semantics.
struct GlbTextureGrowthProbe final
{
    std::size_t oldCapacity = 0, newCapacity = 0, maxOverlapTexels = 0, growths = 0;
    bool overflow = false;
    void* context = nullptr;
    void (*observer)(const GlbTextureGrowthProbe&, void*) = nullptr;
    static GlbTextureGrowthProbe& state() noexcept { static thread_local GlbTextureGrowthProbe value; return value; }
};
struct GlbTextureGrowthProbeScope final
{
    GlbTextureGrowthProbe prior;
    GlbTextureGrowthProbeScope() noexcept : prior(GlbTextureGrowthProbe::state()) { GlbTextureGrowthProbe::state() = {}; }
    ~GlbTextureGrowthProbeScope() { GlbTextureGrowthProbe::state() = prior; }
    GlbTextureGrowthProbeScope(const GlbTextureGrowthProbeScope&) = delete;
    GlbTextureGrowthProbeScope& operator=(const GlbTextureGrowthProbeScope&) = delete;
};
#endif

// Copies the selected scene from an already admitted, in-memory static GLB
// document into the fixed-capacity renderer value contract. The returned scene
// owns every translated byte and keeps no reference to the source document.
std::optional<HarmonicMIDI::grid::Visual3DScene> adaptStaticGlbToVisual3DScene(
    const GlbStaticMeshDocument& asset,
    std::string& error,
    std::optional<std::size_t> meshIndex = std::nullopt);

// Join one admitted import with authored procedural records. Imported targets
// keep their original IDs; procedural IDs use the explicit structural mapping.
std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> composeImportedVisual3DScene(
    const HarmonicMIDI::grid::Visual3DScene& procedural,
    const HarmonicMIDI::grid::Visual3DScene& imported, std::string& error);

// Compose each draw's static node hierarchy into world TRS while preserving
// object/material identities. Callers opt into bounded multi-draw composition.
std::optional<HarmonicMIDI::grid::Visual3DScene> adaptAnimatedGlbMeshToVisual3DScene(
    const GlbStaticMeshDocument& asset,
    std::string& error,
    std::optional<std::size_t> meshIndex = std::nullopt,
    bool multipleDraws = false);

// Keeps the fixed-capacity scene in heap storage through animated admission.
std::unique_ptr<HarmonicMIDI::grid::Visual3DScene> adaptAnimatedGlbMeshToVisual3DSceneHeap(
    const GlbStaticMeshDocument& asset,
    std::string& error,
    std::optional<std::size_t> meshIndex = std::nullopt,
    bool multipleDraws = false);

bool sampleGlbCameraLights(const GlbStaticMeshDocument& asset,
    const visualanimation::Sample& sample, visualanimation::CombinationMode mode, float weight,
    const HarmonicMIDI::grid::Visual3DScene& scene,
    std::vector<HarmonicMIDI::grid::SceneCameraRecord>& cameras,
    std::vector<HarmonicMIDI::grid::SceneLightRecord>& lights, std::string& error);
} // namespace videohelper::gltf
