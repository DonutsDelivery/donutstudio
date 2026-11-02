#include "diffraction_order_bsdf.h"

#include <cmath>

namespace diffractionmaterial
{
namespace
{
using ordercore::validFrame;

// This filter divides by a pixel's solid angle, far below float's reach at 4096 rows.
constexpr double pi = 3.141592653589793238462643383279502884;
using Wide = std::array<double, 3>;
Wide wide(const std::array<float, 3>& v) noexcept { return { v[0], v[1], v[2] }; }
double dot(const Wide& a, const Wide& b) noexcept { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Wide cross(const Wide& a, const Wide& b) noexcept
{
    return { a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] };
}

const char* problemText(DiffractionOrderProblem problem) noexcept
{
    switch (problem)
    {
        case DiffractionOrderProblem::Wavelength: return "diffraction order BSDF wavelength is invalid";
        case DiffractionOrderProblem::MediumIndex: return "diffraction order BSDF medium index is invalid";
        case DiffractionOrderProblem::LocalFrame: return "diffraction order BSDF local frame is invalid";
        case DiffractionOrderProblem::MaterialCoordinate: return "diffraction order BSDF material coordinate is invalid";
        case DiffractionOrderProblem::Lattice: return "diffraction order BSDF lattice is unsupported";
        case DiffractionOrderProblem::Profile: return "diffraction order BSDF profile is unsupported";
        case DiffractionOrderProblem::SinusoidalPhase: return "diffraction order BSDF sinusoidal phase exceeds the bounded model";
        case DiffractionOrderProblem::GrazingIncident: return "diffraction order BSDF incident direction is grazing or backfacing";
        case DiffractionOrderProblem::GrooveField: return "diffraction order BSDF groove field is unsupported";
        case DiffractionOrderProblem::LocalPeriod: return "diffraction order BSDF local grating period is invalid";
        case DiffractionOrderProblem::Reflectance: return "diffraction order BSDF reflectance is invalid";
        case DiffractionOrderProblem::OrderPower: return "diffraction order BSDF produced invalid order power";
        case DiffractionOrderProblem::NoResolvedEnergy: return "diffraction order BSDF has no resolved reflected energy";
        case DiffractionOrderProblem::NoSampleableEnergy: return "diffraction order BSDF has no sampleable reflected energy";
        case DiffractionOrderProblem::InvalidDistribution: return "diffraction order BSDF produced an invalid distribution";
        case DiffractionOrderProblem::SampleRange: return "diffraction order BSDF sample must be in [0,1)";
        case DiffractionOrderProblem::SampleDistribution: return "diffraction order BSDF distribution is invalid";
        case DiffractionOrderProblem::SampleSupport: return "diffraction order BSDF power and probability have different support";
        case DiffractionOrderProblem::SampleInconsistent: return "diffraction order BSDF power and probability are inconsistent";
        case DiffractionOrderProblem::SampleNotNormalized: return "diffraction order BSDF probabilities are not normalized";
        case DiffractionOrderProblem::SampleResolvedPower: return "diffraction order BSDF resolved power is inconsistent";
        case DiffractionOrderProblem::SampleZeroProbability: return "diffraction order BSDF selected a zero-probability order";
        case DiffractionOrderProblem::SampleThroughput: return "diffraction order BSDF sample throughput is invalid";
        case DiffractionOrderProblem::None: break;
    }
    return "";
}

bool finish(DiffractionOrderProblem problem, std::string& error) noexcept
{
    error = problemText(problem);
    return problem == DiffractionOrderProblem::None;
}
} // namespace

bool evaluateDiffractionOrders(const AdmittedDiffractionMaterialIR& material,
                               float wavelength,
                               const DiffractionLocalFrame& frame,
                               const std::array<float, 3>& incidentDirection,
                               const std::array<float, 2>& materialUv,
                               DiffractionOrderDistribution& result,
                               std::string& error,
                               float mediumIndex,
                               const LocalGrating* local) noexcept
{
    return finish(evaluateDiffractionOrdersCore(material.description(), wavelength, frame, incidentDirection,
                                                materialUv, result, mediumIndex, local), error);
}

bool integrateDiscreteDiffractionCameraPixel(const AdmittedDiffractionMaterialIR& material,
    std::size_t wavelengthBand, const DiffractionLocalFrame& gratingFrame,
    const std::array<float, 3>& incidentDirection, const DiffractionCameraIncident& incident,
    const DiffractionCameraPixel& pixel, DiffractionCameraPixelIntegral& result,
    std::string& error) noexcept
{
    const auto fail = [&](const char* message) { error = message; return false; };
    const auto& description = material.description();
    if (incident.measure != DiffractionIncidentMeasure::SurfaceSpectralIrradiance
        || !std::isfinite(incident.value) || incident.value < 0)
        return fail("discrete camera filter requires explicit finite surface spectral irradiance");
    if (description.geometry.lattice != GratingLattice::OneDimensional
        || description.grooveField.mode != GrooveFieldMode::Constant
        || description.microstructure.profile != GrooveProfile::BinaryRectangular
        || description.coating.model != CoatingModel::Uncoated
        || description.roughness.rmsHeightNanometres != 0 || description.roughness.rmsSlope != 0
        || description.spectrum.firstOrder != 1 || description.spectrum.lastOrder != 1
        || description.spectrum.wavelengthCount != kMaximumSpectralSamples
        || description.spectrum.wavelengthsNanometres != kProductionWavelengthsNanometres
        || wavelengthBand >= kMaximumSpectralSamples)
        return fail("discrete camera filter requires exact canonical reflective orders 0 and plus/minus 1");
    if (!validFrame(pixel.axes) || !pixel.width || !pixel.height
        || pixel.width > 4096 || pixel.height > 4096
        || pixel.x >= pixel.width || pixel.y >= pixel.height
        || !std::isfinite(pixel.verticalFovRadians)
        || !(pixel.verticalFovRadians > 0 && pixel.verticalFovRadians < pi))
        return fail("discrete camera filter perspective pixel domain is invalid");
    DiffractionOrderDistribution distribution;
    if (!evaluateDiffractionOrders(material, description.spectrum.wavelengthsNanometres[wavelengthBand],
            gratingFrame, incidentDirection, {0.0f,0.0f}, distribution, error)) return false;
    if (distribution.orderCount > 3)
        return fail("discrete camera filter order count is invalid");
    const double halfY = std::tan(pixel.verticalFovRadians / 2);
    const double halfX = halfY * double(pixel.width) / pixel.height;
    const double left = halfX * (2 * double(pixel.x) / pixel.width - 1);
    const double right = halfX * (2 * double(pixel.x + 1) / pixel.width - 1);
    const double top = halfY * (1 - 2 * double(pixel.y) / pixel.height);
    const double bottom = halfY * (1 - 2 * double(pixel.y + 1) / pixel.height);
    const auto direction = [](double x, double y) {
        const double length = std::sqrt(1 + x*x + y*y);
        return std::array<double,3>{x/length, y/length, 1/length};
    };
    // Spherical triangle area, van Oosterom and Strackee, TBME 30(2), 1983.
    // DOI: 10.1109/TBME.1983.325207. Original implementation of the formula.
    const auto triangleAngle = [](const auto& a, const auto& b, const auto& c) {
        return 2 * std::atan2(std::abs(dot(a, cross(b,c))),
            1 + dot(a,b) + dot(b,c) + dot(c,a));
    };
    const auto a = direction(left,bottom), b = direction(right,bottom);
    const auto c = direction(right,top), d = direction(left,top);
    DiffractionCameraPixelIntegral candidate;
    candidate.solidAngleSteradians = triangleAngle(a,b,c) + triangleAngle(a,c,d);
    if (!std::isfinite(candidate.solidAngleSteradians) || !(candidate.solidAngleSteradians > 0)
        || candidate.solidAngleSteradians >= 2*pi)
        return fail("discrete camera filter solid angle is unrepresentable");
    candidate.orderCount = distribution.orderCount;
    candidate.incidentIrradiance = incident.value;
    candidate.substrateReflectedIrradiance = incident.value * distribution.substrateReflectedPower;
    candidate.resolvedReflectedIrradiance = incident.value * distribution.resolvedReflectedPower;
    candidate.unresolvedReflectedIrradiance = incident.value * distribution.unresolvedReflectedPower;
    candidate.absorbedIrradiance = incident.value * (1 - distribution.substrateReflectedPower);
    double integratedRadiance = 0;
    for (std::size_t index = 0; index < distribution.orderCount; ++index)
    {
        auto& output = candidate.orders[index];
        output.order = distribution.orders[index];
        output.outgoingCosine = dot(wide(output.order.direction), wide(gratingFrame.normal));
        if (!std::isfinite(output.outgoingCosine) || !(output.outgoingCosine > 0))
            return fail("discrete camera filter outgoing direction is singular");
        output.reflectedIrradiance = incident.value * output.order.reflectedPower;
        const double backward = dot(wide(output.order.direction), wide(pixel.axes.normal));
        if (backward > 0)
        {
            // Camera rays travel from camera to scene, opposite the emitted order.
            const double imageX = -dot(wide(output.order.direction), wide(pixel.axes.tangent));
            const double imageY = -dot(wide(output.order.direction), wide(pixel.axes.bitangent));
            output.inPixel = imageX >= left * backward && imageX < right * backward
                && imageY <= top * backward && imageY > bottom * backward;
        }
        if (output.inPixel)
        {
            output.integratedRadiance = output.reflectedIrradiance / output.outgoingCosine;
            integratedRadiance += output.integratedRadiance;
            candidate.pixelReflectedIrradiance += output.reflectedIrradiance;
        }
        if (!std::isfinite(output.reflectedIrradiance) || !std::isfinite(output.integratedRadiance))
            return fail("discrete camera filter order measure is unrepresentable");
    }
    candidate.meanSpectralRadiance = integratedRadiance / candidate.solidAngleSteradians;
    for (const double value : {candidate.incidentIrradiance, candidate.substrateReflectedIrradiance,
            candidate.resolvedReflectedIrradiance, candidate.unresolvedReflectedIrradiance,
            candidate.absorbedIrradiance, candidate.pixelReflectedIrradiance, candidate.meanSpectralRadiance})
        if (!std::isfinite(value) || value < 0)
            return fail("discrete camera filter integrated measure is unrepresentable");
    result = candidate;
    error.clear();
    return true;
}

bool sampleDiffractionOrder(const DiffractionOrderDistribution& distribution,
                            float unitSample,
                            DiffractionOrderSample& result,
                            std::string& error) noexcept
{
    return finish(sampleDiffractionOrderCore(distribution, unitSample, result), error);
}
} // namespace diffractionmaterial
