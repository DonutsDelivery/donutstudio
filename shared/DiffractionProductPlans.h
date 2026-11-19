#pragma once

#include "DiffractionLightingPlan.h"
#include "DiffractionMaterialPresets.h"
#include "../video-helper/src/diffraction_material_admission.h"

#include <array>
#include <cstring>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <string_view>

namespace diffractionmaterial
{
enum class PatternMask : std::uint8_t
{
    FullSurface = 1,
    CardFrameAndEmblem = 2,
    SpatialCells = 3
};

struct MaterialOccupancyMask final
{
    static constexpr std::size_t kMaximumRectangles = 5;
    std::uint8_t rectangleCount = 1;
    // Axis-aligned rectangles in normalized material UV: minU, minV, maxU, maxV.
    std::array<std::array<float, 4>, kMaximumRectangles> rectangles {{
        { 0.0f, 0.0f, 1.0f, 1.0f }
    }};
};

struct TradingCardFoilModule
{
    Description diffractionBsdf;
    SurfaceCoating clearcoat;
    GrooveField embossedGrooves;
    PatternMask patternMask = PatternMask::CardFrameAndEmblem;
    float maskCoverage = 0.72f;
    MaterialOccupancyMask occupancyMask;
};

enum class ReferenceSceneKind : std::uint8_t
{
    CdRadialGrooves = 1,
    LinearGrating = 2,
    BlazedGrating = 3,
    SpatialFoil = 4
};

struct ImmutableDiffractionScenePlan
{
    std::uint32_t version = 1;
    ReferenceSceneKind kind = ReferenceSceneKind::LinearGrating;
    TradingCardFoilModule material;
    LightingDescription lighting;
};

inline TradingCardFoilModule makeTradingCardFoilModule() noexcept
{
    TradingCardFoilModule result;
    result.diffractionBsdf = makeRealtimeCrossedTwoDimensionalGratingPreset();
    result.clearcoat = {
        CoatingModel::IncoherentDielectric, 1800.0f, { 1.52f, 0.0005f }
    };
    result.diffractionBsdf.coating = result.clearcoat;
    result.embossedGrooves.mode = GrooveFieldMode::Linear;
    result.embossedGrooves.originUv = { 0.5f, 0.5f };
    result.embossedGrooves.axisUv = { 0.70710677f, 0.70710677f };
    result.embossedGrooves.grooveSpacingDeltaNanometresPerUnit = 240.0f;
    result.embossedGrooves.secondarySpacingDeltaNanometresPerUnit = -160.0f;
    result.embossedGrooves.orientationDegreesPerUnit = 24.0f;
    result.diffractionBsdf.grooveField = result.embossedGrooves;
    result.occupancyMask.rectangleCount = 5;
    result.occupancyMask.rectangles = {{
        { 0.0f, 0.0f, 1.0f, 0.08f }, { 0.0f, 0.92f, 1.0f, 1.0f },
        { 0.0f, 0.08f, 0.08f, 0.92f }, { 0.92f, 0.08f, 1.0f, 0.92f },
        { 0.38f, 0.38f, 0.62f, 0.62f }
    }};
    return result;
}

inline LightingDescription makeReferenceLighting() noexcept
{
    LightingDescription result;
    result.version = 2;
    result.pathCount = 3;
    result.paths[0].kind = LightingPathKind::Direct;
    result.paths[0].incident.direction = { 0.0f, 0.0f, 1.0f };
    result.paths[0].incident.radiance.fill(0.65f);
    result.paths[1].kind = LightingPathKind::Environment;
    result.paths[1].incident.direction = { 0.0f, 0.0f, 1.0f };
    result.paths[1].incident.radiance = {
        0.12f, 0.14f, 0.18f, 0.22f, 0.24f, 0.22f, 0.18f, 0.14f
    };
    result.paths[2].kind = LightingPathKind::Indirect;
    result.paths[2].bounceDepth = 1;
    result.paths[2].incident.direction = { 0.0f, 0.0f, 1.0f };
    result.paths[2].incident.radiance = {
        0.04f, 0.05f, 0.06f, 0.07f, 0.07f, 0.06f, 0.05f, 0.04f
    };
    result.bouncePlaneHeight = -1.0f;
    result.bounceMaximumDistance = 20.0f;
    return result;
}

inline ImmutableDiffractionScenePlan makeDiffractionReferenceScene(
    ReferenceSceneKind kind) noexcept
{
    ImmutableDiffractionScenePlan result;
    result.kind = kind;
    result.material = makeTradingCardFoilModule();
    result.lighting = makeReferenceLighting();
    switch (kind)
    {
        case ReferenceSceneKind::CdRadialGrooves:
            result.material.patternMask = PatternMask::FullSurface;
            result.material.maskCoverage = 1.0f;
            result.material.occupancyMask = {};
            result.material.diffractionBsdf = makeSinusoidalGratingPreset();
            result.material.diffractionBsdf.geometry.grooveSpacingNanometres = 1600.0f;
            result.material.diffractionBsdf.grooveField.mode = GrooveFieldMode::Radial;
            result.material.diffractionBsdf.grooveField.originUv = { 0.5f, 0.5f };
            result.material.diffractionBsdf.grooveField.orientationDegreesPerUnit = 360.0f;
            break;
        case ReferenceSceneKind::LinearGrating:
            result.material.patternMask = PatternMask::FullSurface;
            result.material.maskCoverage = 1.0f;
            result.material.occupancyMask = {};
            result.material.diffractionBsdf = makeAluminiumBinaryGratingPreset();
            break;
        case ReferenceSceneKind::BlazedGrating:
            result.material.patternMask = PatternMask::FullSurface;
            result.material.maskCoverage = 1.0f;
            result.material.occupancyMask = {};
            result.material.diffractionBsdf = makeBlazedGratingPreset();
            break;
        case ReferenceSceneKind::SpatialFoil:
            break;
    }
    result.material.clearcoat = result.material.diffractionBsdf.coating;
    result.material.embossedGrooves = result.material.diffractionBsdf.grooveField;
    return result;
}

struct DiffractionProductRenderRequest
{
    std::shared_ptr<const ImmutableDiffractionScenePlan> plan;
    std::string productDigest;
};

inline std::string exactDiffractionProductDigest (
    const ImmutableDiffractionScenePlan& plan)
{
    std::string error;
    const auto admittedMaterial = diffractionmaterial::admit (
        plan.material.diffractionBsdf, error);
    const auto admittedLighting = AdmittedLightingPlan::admit (plan.lighting, error);
    if (! admittedMaterial || ! admittedLighting)
        return {};

    std::string result = "diffraction-product-v1:";
    const auto appendHex = [&result] (std::uint64_t value, std::size_t digits)
    {
        static constexpr char table[] = "0123456789abcdef";
        for (std::size_t shift = digits; shift > 0; --shift)
            result.push_back (table[(value >> ((shift - 1u) * 4u)) & 0xfu]);
    };
    const auto appendFloat = [&appendHex] (float value)
    {
        std::uint32_t bits = 0;
        static_assert (sizeof (bits) == sizeof (value), "float digest width changed");
        static_assert (std::numeric_limits<float>::is_iec559,
                       "diffraction products require IEEE-754 float identity");
        std::memcpy (&bits, &value, sizeof (bits));
        appendHex (bits, 8);
    };
    const auto& materialDigest = admittedMaterial->structuralDigest();
    appendHex (plan.version, 8);
    appendHex (static_cast<std::uint8_t> (plan.kind), 2);
    appendHex (static_cast<std::uint8_t> (plan.material.patternMask), 2);
    appendFloat (plan.material.maskCoverage);
    appendHex (plan.material.occupancyMask.rectangleCount, 2);
    for (const auto& rectangle : plan.material.occupancyMask.rectangles)
        for (const auto value : rectangle)
            appendFloat (value);
    appendHex (static_cast<std::uint8_t> (plan.material.clearcoat.model), 2);
    appendFloat (plan.material.clearcoat.thicknessNanometres);
    appendFloat (plan.material.clearcoat.opticalConstants.refractiveIndex);
    appendFloat (plan.material.clearcoat.opticalConstants.extinctionCoefficient);
    appendHex (static_cast<std::uint8_t> (plan.material.embossedGrooves.mode), 2);
    for (const auto value : plan.material.embossedGrooves.originUv)
        appendFloat (value);
    for (const auto value : plan.material.embossedGrooves.axisUv)
        appendFloat (value);
    appendFloat (plan.material.embossedGrooves.grooveSpacingDeltaNanometresPerUnit);
    appendFloat (plan.material.embossedGrooves.secondarySpacingDeltaNanometresPerUnit);
    appendFloat (plan.material.embossedGrooves.orientationDegreesPerUnit);
    appendHex (materialDigest.size(), 16);
    result.append (materialDigest);
    const auto& lighting = admittedLighting->description();
    appendHex (lighting.version, 8);
    if (lighting.version == 2)
    {
        appendFloat (lighting.bouncePlaneHeight);
        appendFloat (lighting.bounceMaximumDistance);
    }
    appendHex (lighting.pathCount, 2);
    for (std::size_t pathIndex = 0; pathIndex < lighting.pathCount; ++pathIndex)
    {
        const auto& path = lighting.paths[pathIndex];
        appendHex (static_cast<std::uint8_t> (path.kind), 2);
        appendHex (path.bounceDepth, 2);
        for (const auto component : path.incident.direction)
            appendFloat (component);
        for (const auto radiance : path.incident.radiance)
            appendFloat (radiance);
    }
    return result;
}

inline DiffractionProductRenderRequest makePreviewRequest(
    const ImmutableDiffractionScenePlan& plan)
{
    auto owned = std::make_shared<const ImmutableDiffractionScenePlan> (plan);
    return { owned, exactDiffractionProductDigest (*owned) };
}
inline DiffractionProductRenderRequest makeExportRequest(
    const ImmutableDiffractionScenePlan& plan)
{
    auto owned = std::make_shared<const ImmutableDiffractionScenePlan> (plan);
    return { owned, exactDiffractionProductDigest (*owned) };
}

static_assert(static_cast<unsigned>(ReferenceSceneKind::SpatialFoil) == 4,
              "Reference products are exact and do not include legacy Holo Foil");
} // namespace diffractionmaterial
