#pragma once

#include "diffraction_material_admission.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace diffractionmaterial
{
inline constexpr std::size_t kMaximumDiffractionBsdfOrders
    = (2u * kMaximumDiffractionOrder + 1u)
        * (2u * kMaximumDiffractionOrder + 1u);

struct DiffractionLocalFrame
{
    std::array<double, 3> tangent { 1.0, 0.0, 0.0 };
    std::array<double, 3> bitangent { 0.0, 1.0, 0.0 };
    std::array<double, 3> normal { 0.0, 0.0, 1.0 };
};

struct DiffractionOrder
{
    std::int8_t primary = 0;
    std::int8_t secondary = 0;
    std::array<double, 3> direction {};
    // Canonical support is exact: reflectedPower and selectionProbability are
    // either both zero or both positive. The sampler assigns normalization
    // rounding, bounded by its validation tolerance, to the last positive bin.
    double reflectedPower = 0.0;
    double selectionProbability = 0.0;
    double angularStandardDeviationRadians = 0.0;
};

struct DiffractionOrderDistribution
{
    std::array<DiffractionOrder, kMaximumDiffractionBsdfOrders> orders {};
    std::size_t orderCount = 0;
    double substrateReflectedPower = 0.0;
    double resolvedReflectedPower = 0.0;
    double unresolvedReflectedPower = 0.0;
};

struct DiffractionOrderSample
{
    DiffractionOrder order;
    std::size_t orderIndex = 0;
    // Probability mass for this discrete order. This is not a solid-angle PDF.
    // A future angular-lobe sampler must report its directional density separately.
    double pdf = 0.0;
    double throughput = 0.0;
};

bool evaluateDiffractionOrders(const AdmittedDiffractionMaterialIR& material,
                               double wavelengthNanometres,
                               const DiffractionLocalFrame& frame,
                               const std::array<double, 3>& incidentDirection,
                               const std::array<double, 2>& materialUv,
                               DiffractionOrderDistribution& result,
                               std::string& error) noexcept;

bool sampleDiffractionOrder(const DiffractionOrderDistribution& distribution,
                            double unitSample,
                            DiffractionOrderSample& result,
                            std::string& error) noexcept;
} // namespace diffractionmaterial
