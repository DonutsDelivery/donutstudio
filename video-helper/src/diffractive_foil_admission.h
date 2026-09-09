#pragma once

#include "diffraction_material_admission.h"
#include "../../shared/DiffractiveFoilIR.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>

namespace diffractivefoil
{
class AdmittedDiffractiveFoilIR;

std::optional<AdmittedDiffractiveFoilIR> admit(
    const Description& source, std::string& error,
    const diffractionmaterial::AdmissionLimits& physicalLimits = {});

class AdmittedDiffractiveFoilIR final
{
public:
    AdmittedDiffractiveFoilIR(const AdmittedDiffractiveFoilIR&) = default;
    AdmittedDiffractiveFoilIR(AdmittedDiffractiveFoilIR&&) = default;
    AdmittedDiffractiveFoilIR& operator=(const AdmittedDiffractiveFoilIR&) = delete;
    AdmittedDiffractiveFoilIR& operator=(AdmittedDiffractiveFoilIR&&) = delete;

    const Description& description() const noexcept { return description_; }
    const diffractionmaterial::AdmittedDiffractionMaterialIR& physicalBsdf() const noexcept
    {
        return physicalBsdf_;
    }
    const std::string& structuralDigest() const noexcept { return digest_; }
    std::size_t physicalOrderEvaluationsPerPoint() const noexcept
    {
        return physicalOrderEvaluationsPerPoint_;
    }
    std::size_t maximumPhysicalOrderEvaluations() const noexcept
    {
        return physicalOrderEvaluationsPerPoint_
            * static_cast<std::size_t>(description_.workBudget.maximumEvaluations);
    }

    static constexpr bool authorizesCpuProductionRendering = false;
    static constexpr bool authorizesNativeGpuExecution = true;

private:
    friend std::optional<AdmittedDiffractiveFoilIR> admit(
        const Description&, std::string&,
        const diffractionmaterial::AdmissionLimits&);

    AdmittedDiffractiveFoilIR(
        Description description,
        diffractionmaterial::AdmittedDiffractionMaterialIR physicalBsdf,
        std::string digest,
        std::size_t physicalOrderEvaluationsPerPoint)
        : description_(std::move(description)),
          physicalBsdf_(std::move(physicalBsdf)),
          digest_(std::move(digest)),
          physicalOrderEvaluationsPerPoint_(physicalOrderEvaluationsPerPoint)
    {
    }

    const Description description_;
    const diffractionmaterial::AdmittedDiffractionMaterialIR physicalBsdf_;
    const std::string digest_;
    const std::size_t physicalOrderEvaluationsPerPoint_;
};

struct LocalEvaluation final
{
    diffractionmaterial::AdmittedDiffractionMaterialIR physicalBsdf;
    float diffractionCoverage = 0.0f;
    std::size_t physicalOrderEvaluations = 0;
};

namespace detail
{
inline bool finite(float value) noexcept
{
    return std::isfinite(value);
}

inline bool validUnitDirection(const std::array<float, 2>& direction) noexcept
{
    const auto lengthSquared = direction[0] * direction[0] + direction[1] * direction[1];
    return finite(lengthSquared) && std::abs(lengthSquared - 1.0f) <= 1.0e-4f;
}

inline void canonicalize(float& value) noexcept
{
    if (value == 0.0f)
        value = 0.0f;
}

inline std::string structuralDigest(const Description& description,
                                    const std::string& physicalDigest)
{
    static constexpr char domain[] = "DonutStudio/DiffractiveFoilIR/Structure/v1";
    videohelper::Sha256 hash;
    hash.update(domain, sizeof(domain) - 1);
    hash.update(physicalDigest.data(), physicalDigest.size());
    diffractionmaterial::detail::hashU32(hash, description.version);
    for (const auto& sample : description.grooveField)
    {
        diffractionmaterial::detail::hashFloats(hash, sample.reciprocalDirectionUv);
        diffractionmaterial::detail::hashFloat(hash, sample.grooveSpacingNanometres);
        diffractionmaterial::detail::hashFloat(hash, sample.diffractionCoverage);
    }
    diffractionmaterial::detail::hashU32(hash, description.workBudget.maximumEvaluations);
    return hash.finishHex();
}
} // namespace detail

inline std::optional<AdmittedDiffractiveFoilIR> admit(
    const Description& source, std::string& error,
    const diffractionmaterial::AdmissionLimits& physicalLimits)
{
    error.clear();
    if (source.version != kWireVersion)
    {
        error = "diffractive foil wire version is unsupported";
        return std::nullopt;
    }
    if (source.workBudget.maximumEvaluations == 0
        || source.workBudget.maximumEvaluations > kMaximumEvaluations)
    {
        error = "diffractive foil evaluation budget is out of bounds";
        return std::nullopt;
    }

    Description candidate = source;
    const auto reference = candidate.grooveField.front().reciprocalDirectionUv;
    if (!detail::validUnitDirection(reference))
    {
        error = "diffractive foil groove field direction is invalid";
        return std::nullopt;
    }
    for (auto& sample : candidate.grooveField)
    {
        if (!detail::validUnitDirection(sample.reciprocalDirectionUv)
            || !detail::finite(sample.grooveSpacingNanometres)
            || sample.grooveSpacingNanometres
                < diffractionmaterial::kMinimumGrooveSpacingNanometres
            || sample.grooveSpacingNanometres
                > diffractionmaterial::kMaximumGrooveSpacingNanometres
            || !detail::finite(sample.diffractionCoverage)
            || sample.diffractionCoverage < 0.0f
            || sample.diffractionCoverage > 1.0f)
        {
            error = "diffractive foil groove field sample is out of bounds";
            return std::nullopt;
        }
        const auto alignment = reference[0] * sample.reciprocalDirectionUv[0]
            + reference[1] * sample.reciprocalDirectionUv[1];
        if (!(alignment >= 0.0f))
        {
            error = "diffractive foil groove field exceeds the bounded orientation span";
            return std::nullopt;
        }
        for (auto& component : sample.reciprocalDirectionUv)
            detail::canonicalize(component);
        detail::canonicalize(sample.grooveSpacingNanometres);
        detail::canonicalize(sample.diffractionCoverage);
    }
    for (std::size_t first = 0; first < candidate.grooveField.size(); ++first)
        for (std::size_t second = first + 1;
             second < candidate.grooveField.size(); ++second)
        {
            const auto dot = candidate.grooveField[first].reciprocalDirectionUv[0]
                    * candidate.grooveField[second].reciprocalDirectionUv[0]
                + candidate.grooveField[first].reciprocalDirectionUv[1]
                    * candidate.grooveField[second].reciprocalDirectionUv[1];
            if (dot < -0.95f)
            {
                error = "diffractive foil groove field can interpolate to a degenerate direction";
                return std::nullopt;
            }
        }

    if (candidate.physicalBsdf.geometry.directionUv != reference
        || candidate.physicalBsdf.geometry.grooveSpacingNanometres
            != candidate.grooveField.front().grooveSpacingNanometres)
    {
        error = "diffractive foil base BSDF must match groove field corner zero";
        return std::nullopt;
    }

    auto physical = diffractionmaterial::admit(
        candidate.physicalBsdf, error, physicalLimits);
    if (!physical)
    {
        error = "diffractive foil physical BSDF rejected: " + error;
        return std::nullopt;
    }
    const auto& spectrum = physical->description().spectrum;
    const auto orderCount = static_cast<std::size_t>(spectrum.lastOrder)
        - static_cast<std::size_t>(spectrum.firstOrder) + 1u;
    const auto workPerPoint = static_cast<std::size_t>(spectrum.wavelengthCount)
        * (1u + 2u * orderCount);
    const auto digest = detail::structuralDigest(candidate, physical->structuralDigest());
    return AdmittedDiffractiveFoilIR(
        std::move(candidate), std::move(*physical), digest, workPerPoint);
}

inline std::optional<LocalEvaluation> evaluateLocalPhysicalBsdf(
    const AdmittedDiffractiveFoilIR& foil,
    std::array<float, 2> normalizedMaterialUv,
    std::uint32_t workItemIndex,
    std::string& error)
{
    error.clear();
    if (!detail::finite(normalizedMaterialUv[0])
        || !detail::finite(normalizedMaterialUv[1])
        || normalizedMaterialUv[0] < 0.0f || normalizedMaterialUv[0] > 1.0f
        || normalizedMaterialUv[1] < 0.0f || normalizedMaterialUv[1] > 1.0f)
    {
        error = "diffractive foil material UV is outside the normalized domain";
        return std::nullopt;
    }
    if (workItemIndex >= foil.description().workBudget.maximumEvaluations)
    {
        error = "diffractive foil deterministic work budget is exhausted";
        return std::nullopt;
    }

    const auto u = normalizedMaterialUv[0];
    const auto v = normalizedMaterialUv[1];
    const std::array<float, 4> weights {
        (1.0f - u) * (1.0f - v), u * (1.0f - v),
        (1.0f - u) * v, u * v
    };
    std::array<float, 2> direction {};
    float spacing = 0.0f;
    float coverage = 0.0f;
    for (std::size_t index = 0; index < weights.size(); ++index)
    {
        const auto& sample = foil.description().grooveField[index];
        direction[0] += weights[index] * sample.reciprocalDirectionUv[0];
        direction[1] += weights[index] * sample.reciprocalDirectionUv[1];
        spacing += weights[index] * sample.grooveSpacingNanometres;
        coverage += weights[index] * sample.diffractionCoverage;
    }
    const auto length = std::sqrt(direction[0] * direction[0] + direction[1] * direction[1]);
    if (!(length > 1.0e-6f) || !detail::finite(length))
    {
        error = "diffractive foil interpolated groove direction is degenerate";
        return std::nullopt;
    }
    direction[0] /= length;
    direction[1] /= length;

    auto local = foil.physicalBsdf().description();
    local.geometry.directionUv = direction;
    local.geometry.grooveSpacingNanometres = spacing;
    auto physical = diffractionmaterial::admit(local, error);
    if (!physical)
    {
        error = "diffractive foil local physical BSDF rejected: " + error;
        return std::nullopt;
    }
    return LocalEvaluation {
        std::move(*physical), std::clamp(coverage, 0.0f, 1.0f),
        foil.physicalOrderEvaluationsPerPoint()
    };
}

inline bool requireProductRoute(ProductUse use, Backend backend, std::string& error)
{
    if (use == ProductUse::CpuReferenceValidation && backend == Backend::CpuReference)
    {
        error.clear();
        return true;
    }
    if (use == ProductUse::Preview || use == ProductUse::Export)
    {
        if (backend == Backend::OpenGl || backend == Backend::Metal)
        {
            error.clear();
            return true;
        }
        error = "diffractive foil production route requires a native spatial groove-field executor";
        return false;
    }
    error = "diffractive foil product route is invalid";
    return false;
}

static_assert(!AdmittedDiffractiveFoilIR::authorizesCpuProductionRendering);
static_assert(AdmittedDiffractiveFoilIR::authorizesNativeGpuExecution);
} // namespace diffractivefoil
