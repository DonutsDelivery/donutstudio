// Grating-equation check for the CPU diffraction order BSDF.
//
// Convention (matches evaluateDiffractionOrders): local tangent = +x, bitangent
// = +y (grooves run along y), normal = +z. The incident direction points toward
// the light, L = (sin ti, 0, cos ti). Angles are measured from the normal in the
// x-z plane, signed by the tangent (x) component of the outgoing direction
// V = (sin to, 0, cos to). The code sets V_x = -L_x + m*lambda/d, so the
// predicted angle is sin(to) = m*lambda/d - sin(ti). Order 0 is the mirror
// direction to = -ti.

#include "diffraction_order_bsdf.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

namespace
{
using namespace diffractionmaterial;

constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double gratingSpacingNanometres = 1000.0;
constexpr double angleToleranceDegrees = 0.1;
// Orders whose |sin(to)| lies this close to 1 sit on the grazing boundary and
// are not asserted in either direction.
constexpr double grazingMargin = 1.0e-9;
int checks = 0;
int failures = 0;

void check(bool condition, const std::string& message)
{
    ++checks;
    if (!condition)
    {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

Description makeGrating()
{
    Description description;
    description.geometry.directionUv = { 1.0f, 0.0f };
    description.geometry.grooveSpacingNanometres = static_cast<float>(gratingSpacingNanometres);
    description.geometry.lattice = GratingLattice::OneDimensional;
    description.microstructure.profile = GrooveProfile::BinaryRectangular;
    description.microstructure.grooveDepthNanometres = 250.0f;
    description.microstructure.dutyCycle = 0.5f;
    description.substrate.refractiveIndex = 1.5f;
    description.coating.model = CoatingModel::Uncoated;
    description.spectrum.integrationModel = SpectralIntegrationModel::Cie1931Xyz;
    description.spectrum.wavelengthCount = static_cast<std::uint8_t>(kMaximumSpectralSamples);
    description.spectrum.wavelengthsNanometres = kProductionWavelengthsNanometres;
    description.spectrum.firstOrder = 1;
    description.spectrum.lastOrder = 2;
    return description;
}

void runCase(double incidentDegrees, const std::string& label)
{
    std::string error;
    const auto material = admit(makeGrating(), error);
    check(material.has_value(), label + " admits: " + error);
    if (!material)
        return;

    const DiffractionLocalFrame frame {};
    const double incidentRadians = incidentDegrees * pi / 180.0;
    const double sinIncident = std::sin(incidentRadians);
    const double cosIncident = std::cos(incidentRadians);
    const std::array<double, 3> incident { sinIncident, 0.0, cosIncident };

    std::cout << "case " << label << " theta_i_deg " << incidentDegrees
              << " d_nm " << gratingSpacingNanometres << '\n';
    std::cout << "lambda_nm m predicted_deg measured_deg error_deg\n";
    std::cout << std::fixed << std::setprecision(4);
    for (const float wavelengthNanometres : kProductionWavelengthsNanometres)
    {
        const double wavelength = wavelengthNanometres;
        DiffractionOrderDistribution distribution;
        check(evaluateDiffractionOrders(*material, wavelength, frame, incident,
                                        { 0.5, 0.5 }, distribution, error),
              label + " evaluates: " + error);
        if (distribution.orderCount == 0)
            continue;

        for (int primary = -1; primary <= 1; ++primary)
        {
            const double sinOut = primary * wavelength / gratingSpacingNanometres - sinIncident;
            if (std::abs(std::abs(sinOut) - 1.0) < grazingMargin)
                continue;

            const DiffractionOrder* found = nullptr;
            std::size_t matches = 0;
            for (std::size_t index = 0; index < distribution.orderCount; ++index)
            {
                const auto& order = distribution.orders[index];
                if (order.primary == primary && order.secondary == 0)
                {
                    found = &order;
                    ++matches;
                }
            }
            if (std::abs(sinOut) > 1.0)
            {
                check(matches == 0, label + " evanescent order carries no energy");
                continue;
            }
            check(matches == 1, label + " propagating order is present once");
            if (!found)
                continue;

            const auto& order = *found;
            const double predicted = std::asin(sinOut) * 180.0 / pi;
            const double measured = std::atan2(order.direction[0], order.direction[2]) * 180.0 / pi;
            const double errorDegrees = measured - predicted;
            check(std::abs(errorDegrees) <= angleToleranceDegrees,
                  label + " angle within 0.1 degrees");
            check(std::abs(gratingSpacingNanometres * (order.direction[0] - sinIncident)
                           - primary * wavelength) <= 1.0e-6,
                  label + " grating equation holds in nanometres");
            if (primary == 0)
                check(std::abs(order.direction[0] + sinIncident) <= 1.0e-9
                          && std::abs(order.direction[1]) <= 1.0e-12
                          && std::abs(order.direction[2] - cosIncident) <= 1.0e-9,
                      label + " order zero is the mirror direction");
            std::cout << wavelength << ' ' << primary << ' ' << predicted << ' '
                      << measured << ' ' << errorDegrees << '\n';
        }
    }
}
} // namespace

int main()
{
    runCase(30.0, "oblique");
    runCase(0.0, "normal");
    std::cout << "checks " << checks << " failures " << failures << '\n';
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
