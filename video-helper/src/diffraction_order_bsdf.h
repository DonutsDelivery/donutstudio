#pragma once

#include "diffraction_material_admission.h"
#include "diffraction_order_core.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace diffractionmaterial
{
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
    const std::array<float, 3>& incidentDirection, const DiffractionCameraIncident& incident,
    const DiffractionCameraPixel& pixel, DiffractionCameraPixelIntegral& result,
    std::string& error) noexcept;

bool evaluateDiffractionOrders(const AdmittedDiffractionMaterialIR& material,
                               float wavelengthNanometres,
                               const DiffractionLocalFrame& frame,
                               const std::array<float, 3>& incidentDirection,
                               const std::array<float, 2>& materialUv,
                               DiffractionOrderDistribution& result,
                               std::string& error,
                               // Index of the medium the grating sits in, at this wavelength.
                               float mediumIndex = 1.0f,
                               // A spatial foil's grating at this point, in place of the
                               // material's own: reciprocal direction in uv and period.
                               const LocalGrating* local = nullptr) noexcept;

bool sampleDiffractionOrder(const DiffractionOrderDistribution& distribution,
                            float unitSample,
                            DiffractionOrderSample& result,
                            std::string& error) noexcept;
} // namespace diffractionmaterial
