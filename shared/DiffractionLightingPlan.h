#pragma once

#include "DiffractionMaterialIR.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <utility>

namespace diffractionmaterial
{
inline constexpr std::size_t kMaximumLightingPaths = 8;
inline constexpr std::uint8_t kMaximumIndirectBounces = 1;

enum class LightingPathKind : std::uint8_t
{
    Invalid = 0,
    Direct = 1,
    Environment = 2,
    Indirect = 3
};

struct SpectralLightingPath
{
    LightingPathKind kind = LightingPathKind::Invalid;
    std::uint8_t bounceDepth = 0;
    SpectralIncidentLight incident;
};

enum class SceneReceiverTransport : std::uint8_t
{
    None = 0,
    AuthoredSpectralLambertian = 1
};

struct LightingDescription
{
    std::uint32_t version = 1;
    std::uint8_t pathCount = 0;
    std::array<SpectralLightingPath, kMaximumLightingPaths> paths {};
    // Version 2 traces reciprocal orders into an analytic spectral environment
    // and a horizontal Lambertian receiver. No mesh-to-mesh visibility is implied.
    float bouncePlaneHeight = 0.0f;
    float bounceMaximumDistance = 0.0f;
    // Version 3 uses one world-space directional source (toward the light).
    // Receiver rho is the exact eight-band Surface spectral reflectance, never
    // reconstructed from RGB or texture values. Native transport is separate.
    SceneReceiverTransport sceneReceiverTransport = SceneReceiverTransport::None;
    float traceMaximumDistance = 0.0f;
};

inline bool sameLightingDescription(const LightingDescription& left,
                                    const LightingDescription& right) noexcept
{
    const auto sameFloat = [](float a, float b) noexcept
    {
        std::uint32_t aBits = 0;
        std::uint32_t bBits = 0;
        static_assert(sizeof(aBits) == sizeof(a), "float comparison width changed");
        std::memcpy(&aBits, &a, sizeof(aBits));
        std::memcpy(&bBits, &b, sizeof(bBits));
        return aBits == bBits;
    };
    if (left.version != right.version || left.pathCount != right.pathCount
        || !sameFloat(left.bouncePlaneHeight, right.bouncePlaneHeight)
        || !sameFloat(left.bounceMaximumDistance, right.bounceMaximumDistance)
        || left.sceneReceiverTransport != right.sceneReceiverTransport
        || !sameFloat(left.traceMaximumDistance, right.traceMaximumDistance))
        return false;
    for (std::size_t pathIndex = 0; pathIndex < left.paths.size(); ++pathIndex)
    {
        const auto& a = left.paths[pathIndex];
        const auto& b = right.paths[pathIndex];
        if (a.kind != b.kind || a.bounceDepth != b.bounceDepth)
            return false;
        for (std::size_t component = 0; component < a.incident.direction.size(); ++component)
            if (!sameFloat(a.incident.direction[component], b.incident.direction[component]))
                return false;
        for (std::size_t sample = 0; sample < a.incident.radiance.size(); ++sample)
            if (!sameFloat(a.incident.radiance[sample], b.incident.radiance[sample]))
                return false;
    }
    return true;
}

class AdmittedLightingPlan final
{
public:
    static std::optional<AdmittedLightingPlan> admit(const LightingDescription& source,
                                                      std::string& error)
    {
        error.clear();
        if ((source.version != 1 && source.version != 2 && source.version != 3) || source.pathCount == 0
            || source.pathCount > kMaximumLightingPaths)
        {
            error = "physical diffraction lighting path count or version is unsupported";
            return std::nullopt;
        }
        if ((source.version != 3
             && (source.sceneReceiverTransport != SceneReceiverTransport::None
                 || source.traceMaximumDistance != 0.0f))
            || (source.version == 3
                && (source.pathCount != 1
                    || source.paths[0].kind != LightingPathKind::Direct
                    || source.sceneReceiverTransport != SceneReceiverTransport::AuthoredSpectralLambertian
                    || source.bouncePlaneHeight != 0.0f || source.bounceMaximumDistance != 0.0f
                    || !std::isfinite(source.traceMaximumDistance)
                    || source.traceMaximumDistance < 0.001f
                    || source.traceMaximumDistance > 10000.0f)))
        {
            error = "physical diffraction scene receiver transport requires one authored spectral source and a finite trace bound";
            return std::nullopt;
        }
        if ((source.version == 1
             && (source.bouncePlaneHeight != 0.0f || source.bounceMaximumDistance != 0.0f))
            || (source.version == 2
                && (source.pathCount != 3
                    || source.paths[0].kind != LightingPathKind::Direct
                    || source.paths[1].kind != LightingPathKind::Environment
                    || source.paths[2].kind != LightingPathKind::Indirect
                    || source.paths[0].incident.direction != std::array<float, 3> { 0.0f, 0.0f, 1.0f }
                    || source.paths[1].incident.direction != source.paths[2].incident.direction
                    || !std::isfinite(source.bouncePlaneHeight)
                    || std::abs(source.bouncePlaneHeight) > 10000.0f
                    || !std::isfinite(source.bounceMaximumDistance)
                    || source.bounceMaximumDistance < 0.001f
                    || source.bounceMaximumDistance > 10000.0f)))
        {
            error = "physical diffraction environment requires three paths and a bounded diffuse plane";
            return std::nullopt;
        }
        if (source.version == 2)
            for (std::size_t i = 0; i < kMaximumSpectralSamples; ++i)
                if (source.paths[2].incident.radiance[i] > source.paths[1].incident.radiance[i])
                {
                    error = "physical diffraction diffuse-plane reflectance exceeds one";
                    return std::nullopt;
                }
        bool hasIndirect = false;
        for (std::size_t i = 0; i < source.pathCount; ++i)
        {
            const auto& path = source.paths[i];
            const bool depthMatches =
                (path.kind == LightingPathKind::Direct
                 || path.kind == LightingPathKind::Environment) ? path.bounceDepth == 0
                : path.kind == LightingPathKind::Indirect
                    && path.bounceDepth == kMaximumIndirectBounces;
            if (!depthMatches)
            {
                error = "physical diffraction lighting path kind or bounce depth is unsupported";
                return std::nullopt;
            }
            const float lengthSquared = path.incident.direction[0] * path.incident.direction[0]
                + path.incident.direction[1] * path.incident.direction[1]
                + path.incident.direction[2] * path.incident.direction[2];
            if (!std::isfinite(lengthSquared) || std::abs(lengthSquared - 1.0f) > 1.0e-4f
                || (source.version == 1 && path.incident.direction[2] <= 1.0e-6f))
            {
                error = "physical diffraction lighting direction is invalid";
                return std::nullopt;
            }
            for (float radiance : path.incident.radiance)
                if (!std::isfinite(radiance) || radiance < 0.0f)
                {
                    error = "physical diffraction lighting spectrum is invalid";
                    return std::nullopt;
                }
            hasIndirect = hasIndirect || path.kind == LightingPathKind::Indirect;
        }
        for (std::size_t i = source.pathCount; i < source.paths.size(); ++i)
            if (source.paths[i].kind != LightingPathKind::Invalid
                || (source.version == 3
                    && (source.paths[i].bounceDepth != 0
                        || source.paths[i].incident.direction != SpectralIncidentLight {}.direction
                        || source.paths[i].incident.radiance != SpectralIncidentLight {}.radiance)))
            {
                error = "physical diffraction lighting descriptor has noncanonical trailing paths";
                return std::nullopt;
            }
        return AdmittedLightingPlan(source, hasIndirect
            || source.sceneReceiverTransport == SceneReceiverTransport::AuthoredSpectralLambertian);
    }

    const LightingDescription& description() const noexcept { return description_; }
    bool hasIndirectBounce() const noexcept { return hasIndirectBounce_; }

private:
    AdmittedLightingPlan(LightingDescription source, bool hasIndirect)
        : description_(std::move(source)), hasIndirectBounce_(hasIndirect) {}
    const LightingDescription description_;
    const bool hasIndirectBounce_;
};

struct TransportEnergyReceipt
{
    std::uint8_t pathCount = 0;
    std::uint8_t maximumBounceDepth = 0;
    std::array<double, 4> energyLedger {};
    std::array<double, 4> orderLedger {};
    std::array<double, 4> directionMoment {};
};
} // namespace diffractionmaterial
