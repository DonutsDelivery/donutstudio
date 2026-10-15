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

// Perspective pinhole angular box filter. Axes are camera right, up, and
// backward, so cross(right, up) == backward. Rows are top first. Adjacent
// pixels partition the angular domain using [x,x+1) and [y,y+1).
struct DiffractionCameraPixel
{
    DiffractionLocalFrame axes;
    double verticalFovRadians = 0.0;
    std::uint32_t width = 0, height = 0, x = 0, y = 0;
};
struct DiffractionCameraPixelOrder
{
    DiffractionOrder order; // Original exact direction, power and discrete mass
    double outgoingCosine = 0.0;
    bool inPixel = false;
    double reflectedIrradiance = 0.0; // Whole order, W / (m^2 nm)
    double integratedRadiance = 0.0; // Pixel integral of Lo dOmega, W / (m^2 nm)
};
struct DiffractionCameraPixelIntegral
{
    std::array<DiffractionCameraPixelOrder, 3> orders {};
    std::size_t orderCount = 0;
    double solidAngleSteradians = 0.0;
    double meanSpectralRadiance = 0.0; // W / (m^2 sr nm), angular box average
    double pixelReflectedIrradiance = 0.0; // Sum only this field across pixels
    // Whole-distribution metadata, counted once, never once per pixel.
    double incidentIrradiance = 0.0;
    double substrateReflectedIrradiance = 0.0;
    double resolvedReflectedIrradiance = 0.0;
    double unresolvedReflectedIrradiance = 0.0;
    double absorbedIrradiance = 0.0; // Reflective authority has no transmission
};
// Uniform planar illumination only. For each exact order, Lo is the angular
// measure E_surface * reflectedPower / outgoingCosine * deltaOmega. Integrate
// its support in the actual camera pixel domain, then divide by that domain's
// solid angle. No lobe width or discrete sampling probability enters the filter.
// Original derivation from projected-flux radiometry and delta reflection:
// https://www.pbr-book.org/4ed/Reflection_Models/BSDF_Representation
// https://www.pbr-book.org/4ed/Cameras_and_Film/Film_and_Imaging
// This CPU authority produces no pixels or native execution grant. The consumer
// must establish incident-area/Jacobian and exact source/camera/visibility owners.
bool integrateDiscreteDiffractionCameraPixel(const AdmittedDiffractionMaterialIR& material,
    std::size_t wavelengthBand, const DiffractionLocalFrame& gratingFrame,
    const std::array<double, 3>& incidentDirection, const DiffractionCameraIncident& incident,
    const DiffractionCameraPixel& pixel, DiffractionCameraPixelIntegral& result,
    std::string& error) noexcept;

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
