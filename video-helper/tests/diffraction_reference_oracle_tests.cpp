#include "../src/diffraction_material_admission.h"
#include "support/diffraction_reference_oracle.h"
#include "support/diffraction_wavelength_validation_scenes.h"
#include "../../shared/DiffractionMaterialPresets.h"
#include "../../shared/DiffractionProductPlans.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>

namespace
{
using namespace diffractionmaterial;
using namespace diffractionmaterial::reference;

int failures = 0;
int checks = 0;

void check(bool value, const char* message)
{
    ++checks;
    if (!value)
    {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", message);
    }
}

bool near(double left, double right, double tolerance = 1.0e-10)
{
    return std::abs(left - right) <= tolerance;
}

bool finiteNonnegative(LinearSrgb value)
{
    return std::isfinite(value.red) && std::isfinite(value.green)
        && std::isfinite(value.blue) && value.red >= 0.0
        && value.green >= 0.0 && value.blue >= 0.0;
}

bool unit(Direction direction)
{
    return near(direction.x * direction.x + direction.y * direction.y
                    + direction.z * direction.z,
                1.0, 1.0e-9);
}

void checkEnergyLedger(const EvaluationResult& result, const char* message)
{
    bool valid = near(result.incidentEnergy,
                      result.substrateReflectedEnergy + result.absorbedEnergy)
        && near(result.substrateReflectedEnergy,
                result.resolvedReflectedEnergy + result.unresolvedReflectedEnergy)
        && near(result.resolvedReflectedEnergy,
                result.zeroOrderEnergy + result.higherOrderEnergy)
        && result.substrateReflectedEnergy <= result.incidentEnergy
        && result.unresolvedReflectedEnergy >= 0.0;
    for (std::size_t index = 0; index < result.spectralSampleCount; ++index)
    {
        const auto& sample = result.spectralSamples[index];
        valid = valid
            && near(sample.incidentEnergy,
                    sample.substrateReflectedEnergy + sample.absorbedEnergy)
            && near(sample.substrateReflectedEnergy,
                    sample.resolvedReflectedEnergy
                        + sample.unresolvedReflectedEnergy)
            && near(sample.resolvedReflectedEnergy,
                    sample.zeroOrderEnergy + sample.higherOrderEnergy);
    }
    check(valid, message);
}

void checkPreset(const validation::WavelengthValidationScene& scene)
{
    const auto& description = scene.description;
    std::string error;
    const auto admitted = admit(description, error);
    EvaluationFailure failure = EvaluationFailure::None;
    const auto result = admitted
        ? evaluate(*admitted, scene.input, failure) : std::nullopt;
    const auto* resultValue = result ? &*result : nullptr;
    check(admitted && resultValue != nullptr && failure == EvaluationFailure::None
              && resultValue->eventCount > resultValue->spectralSampleCount
              && finiteNonnegative(resultValue->outputLinearSrgb),
          scene.name.data());
    if (resultValue == nullptr)
        return;

    checkEnergyLedger(*resultValue,
                      "each wavelength and the integrated result conserve energy");
    bool physicalEvents = true;
    for (std::size_t index = 0; index < resultValue->eventCount; ++index)
    {
        const auto& event = resultValue->events[index];
        physicalEvents = physicalEvents && unit(event.outgoingDirection)
            && event.outgoingDirection.z > 0.0
            && std::isfinite(event.energy) && event.energy >= 0.0;
        if (event.order != 0 || event.secondaryOrder != 0)
        {
            const auto solved = description.geometry.lattice
                    == GratingLattice::CrossedTwoDimensional
                ? solveCrossedOrder(
                    { 0.0, 0.0, 1.0 },
                    { description.geometry.directionUv[0],
                      description.geometry.directionUv[1] },
                    description.geometry.grooveSpacingNanometres, event.order,
                    { description.geometry.secondaryDirectionUv[0],
                      description.geometry.secondaryDirectionUv[1] },
                    description.geometry.secondaryGrooveSpacingNanometres,
                    event.secondaryOrder, event.wavelengthNanometres)
                : solveOrder(
                    { 0.0, 0.0, 1.0 },
                    { description.geometry.directionUv[0],
                      description.geometry.directionUv[1] },
                    description.geometry.grooveSpacingNanometres,
                    event.wavelengthNanometres, event.order);
            physicalEvents = physicalEvents
                && solved.disposition == OrderDisposition::Propagating
                && near(event.outgoingDirection.x, solved.outgoingDirection.x)
                && near(event.outgoingDirection.y, solved.outgoingDirection.y)
                && near(event.outgoingDirection.z, solved.outgoingDirection.z);
        }
    }
    check(physicalEvents,
          "emitted events are normalized propagating grating-order directions");

    const auto high = evaluateHighSampleReference(*admitted, scene.input, failure);
    const auto* highValue = high ? &*high : nullptr;
    check(highValue != nullptr && failure == EvaluationFailure::None
              && highValue->spectralSampleCount == kHighSampleReferenceCount
              && validation::kFiniteSceneReferenceStepNanometres == 5.0f
              && finiteNonnegative(highValue->outputLinearSrgb)
              && std::abs(resultValue->outputLinearSrgb.red
                          - highValue->outputLinearSrgb.red)
                   <= validation::kFiniteSceneMaximumZeroClampedLinearSrgbChannelError
              && std::abs(resultValue->outputLinearSrgb.green
                          - highValue->outputLinearSrgb.green)
                   <= validation::kFiniteSceneMaximumZeroClampedLinearSrgbChannelError
              && std::abs(resultValue->outputLinearSrgb.blue
                          - highValue->outputLinearSrgb.blue)
                   <= validation::kFiniteSceneMaximumZeroClampedLinearSrgbChannelError,
          "the finite scene meets its channel error bound against the 5 nm oracle");
    check(std::abs(resultValue->outputLinearSrgb.red - scene.expectedZeroClampedLinearSrgb[0])
                  <= validation::kFiniteSceneMaximumZeroClampedLinearSrgbChannelError
              && std::abs(resultValue->outputLinearSrgb.green - scene.expectedZeroClampedLinearSrgb[1])
                   <= validation::kFiniteSceneMaximumZeroClampedLinearSrgbChannelError
              && std::abs(resultValue->outputLinearSrgb.blue - scene.expectedZeroClampedLinearSrgb[2])
                   <= validation::kFiniteSceneMaximumZeroClampedLinearSrgbChannelError,
          "the finite scene meets its generated zero-clamped linear-sRGB reference");
}
} // namespace

int main()
{
    using namespace diffractionmaterial;
    using namespace diffractionmaterial::reference;

    const auto firstOrder = solveOrder(
        { 0.0, 0.0, 1.0 }, { 1.0, 0.0 }, 1000.0, 500.0, 1);
    const auto negativeOrder = solveOrder(
        { 0.0, 0.0, 1.0 }, { 1.0, 0.0 }, 1000.0, 500.0, -1);
    check(firstOrder.disposition == OrderDisposition::Propagating
              && near(firstOrder.outgoingDirection.x, 0.5)
              && near(firstOrder.outgoingDirection.z, std::sqrt(0.75))
              && negativeOrder.disposition == OrderDisposition::Propagating
              && near(negativeOrder.outgoingDirection.x, -0.5),
          "signed orders follow the reflective vector grating equation");
    check(solveOrder({ 0.0, 0.0, 1.0 }, { 1.0, 0.0 }, 1000.0, 500.0, 2)
              .disposition == OrderDisposition::Grazing
              && solveOrder(
                     { 0.0, 0.0, 1.0 }, { 1.0, 0.0 }, 1000.0, 500.0, 3)
                     .disposition == OrderDisposition::Evanescent,
          "grazing and evanescent orders are classified before event emission");
    check(solveOrder(
              { 0.0, 0.0, 1.0 }, { 2.0, 0.0 }, 1000.0, 500.0, 1)
              .disposition == OrderDisposition::Invalid,
          "degenerate grating directions fail closed");
    const auto crossedOrder = solveCrossedOrder(
        { 0.0, 0.0, 1.0 }, { 1.0, 0.0 }, 1000.0, 1,
        { 0.0, 1.0 }, 1000.0, 1, 500.0);
    check(crossedOrder.disposition == OrderDisposition::Propagating
              && near(crossedOrder.outgoingDirection.x, 0.5)
              && near(crossedOrder.outgoingDirection.y, 0.5)
              && near(crossedOrder.outgoingDirection.z, std::sqrt(0.5)),
          "crossed orders apply both orthogonal reciprocal-lattice vectors");
    check(solveCrossedOrder(
              { 0.0, 0.0, 1.0 }, { 1.0, 0.0 }, 1000.0, 1,
              { 1.0, 0.0 }, 1000.0, 1, 500.0)
              .disposition == OrderDisposition::Invalid,
          "non-orthogonal crossed directions fail closed");

    const auto binaryZero = profileOrderEfficiency(
        GrooveProfile::BinaryRectangular, 500.0, 125.0, 0.5, 1.0, 0);
    const auto binaryFirst = profileOrderEfficiency(
        GrooveProfile::BinaryRectangular, 500.0, 125.0, 0.5, 1.0, 1);
    check(near(binaryZero, 0.0, 1.0e-12)
              && near(binaryFirst, 4.0 / (3.14159265358979323846
                                         * 3.14159265358979323846),
                      1.0e-12),
          "binary profile Fourier coefficients match the analytic fixture");

    constexpr double pi = 3.14159265358979323846;
    const auto unitArgumentDepth = 500.0 / (2.0 * pi);
    const auto sinusoidalZero = profileOrderEfficiency(
        GrooveProfile::Sinusoidal, 500.0, unitArgumentDepth, 0.5, 1.0, 0);
    const auto sinusoidalFirst = profileOrderEfficiency(
        GrooveProfile::Sinusoidal, 500.0, unitArgumentDepth, 0.5, 1.0, 1);
    const auto sinusoidalSecond = profileOrderEfficiency(
        GrooveProfile::Sinusoidal, 500.0, unitArgumentDepth, 0.5, 1.0, 2);
    check(near(sinusoidalZero, 0.58552749951366402438, 1.0e-12)
              && near(sinusoidalFirst, 0.19364451801445908452, 1.0e-12)
              && near(sinusoidalSecond, 0.01320281084949548076, 1.0e-12),
          "sinusoidal order efficiencies match independent J_m(1) fixtures");

    const auto blazeSelected = profileOrderEfficiency(
        GrooveProfile::BlazedSawtooth, 500.0, 250.0, 1.0, 1.0, 1);
    const auto blazeRejected = profileOrderEfficiency(
        GrooveProfile::BlazedSawtooth, 500.0, 250.0, 1.0, 1.0, -1);
    check(near(blazeSelected, 1.0, 1.0e-12)
              && near(blazeRejected, 0.0, 1.0e-12),
          "the sawtooth phase ramp directs the analytic blaze order asymmetrically");

    double binaryParseval = 0.0;
    double sinusoidalParseval = 0.0;
    double blazedParseval = 0.0;
    for (int order = -128; order <= 128; ++order)
    {
        binaryParseval += profileOrderEfficiency(
            GrooveProfile::BinaryRectangular, 510.0, 137.0, 0.37, 0.91, order);
        sinusoidalParseval += profileOrderEfficiency(
            GrooveProfile::Sinusoidal, 510.0, 137.0, 0.5, 0.91, order);
        blazedParseval += profileOrderEfficiency(
            GrooveProfile::BlazedSawtooth, 510.0, 137.0, 0.73, 0.91, order);
    }
    check(near(binaryParseval, 1.0, 0.004)
              && near(sinusoidalParseval, 1.0, 1.0e-10)
              && near(blazedParseval, 1.0, 0.004),
          "binary, sinusoidal, and blazed scalar orders obey Parseval energy bounds");

    std::string convergenceError;
    const auto& orderEightFixture = *std::find_if(
        validation::wavelengthValidationScenes().begin(),
        validation::wavelengthValidationScenes().end(),
        [] (const auto& scene) { return scene.name == "sinusoidal-order-eight"; });
    auto orderOneDescription = orderEightFixture.description;
    orderOneDescription.spectrum.lastOrder = 1;
    auto orderTwoDescription = orderOneDescription;
    orderTwoDescription.spectrum.lastOrder = 2;
    auto orderFourDescription = orderOneDescription;
    orderFourDescription.spectrum.lastOrder = 4;
    auto orderEightDescription = orderOneDescription;
    orderEightDescription.spectrum.lastOrder = 8;
    const auto orderOne = admit(orderOneDescription, convergenceError);
    const auto orderTwo = admit(orderTwoDescription, convergenceError);
    const auto orderFour = admit(orderFourDescription, convergenceError);
    const auto orderEight = admit(orderEightDescription, convergenceError);
    EvaluationFailure convergenceFailure = EvaluationFailure::None;
    const auto orderOneResult = orderOne
        ? evaluate(*orderOne, orderEightFixture.input, convergenceFailure) : std::nullopt;
    const auto orderTwoResult = orderTwo
        ? evaluate(*orderTwo, orderEightFixture.input, convergenceFailure) : std::nullopt;
    const auto orderFourResult = orderFour
        ? evaluate(*orderFour, orderEightFixture.input, convergenceFailure) : std::nullopt;
    const auto orderEightResult = orderEight
        ? evaluate(*orderEight, orderEightFixture.input, convergenceFailure) : std::nullopt;
    const auto* orderOneValue = orderOneResult ? &*orderOneResult : nullptr;
    const auto* orderTwoValue = orderTwoResult ? &*orderTwoResult : nullptr;
    const auto* orderFourValue = orderFourResult ? &*orderFourResult : nullptr;
    const auto* orderEightValue = orderEightResult ? &*orderEightResult : nullptr;
    check(orderOneValue != nullptr && orderTwoValue != nullptr
              && orderFourValue != nullptr && orderEightValue != nullptr
              && orderTwoValue->resolvedReflectedEnergy
                   > orderOneValue->resolvedReflectedEnergy
              && orderFourValue->resolvedReflectedEnergy
                   > orderTwoValue->resolvedReflectedEnergy
              && orderEightValue->resolvedReflectedEnergy
                   >= orderFourValue->resolvedReflectedEnergy
              && orderEightValue->unresolvedReflectedEnergy
                   <= orderFourValue->unresolvedReflectedEnergy
              && orderFourValue->unresolvedReflectedEnergy
                   < orderTwoValue->unresolvedReflectedEnergy,
          "increasing the bounded sinusoidal order range converges monotonically");

    const auto canonicalSpectrum = admit(orderEightFixture.description, convergenceError);
    const auto canonicalResult = canonicalSpectrum
        ? evaluate(*canonicalSpectrum, orderEightFixture.input, convergenceFailure)
        : std::nullopt;
    const auto highResult = canonicalSpectrum
        ? evaluateHighSampleReference(
            *canonicalSpectrum, orderEightFixture.input, convergenceFailure)
        : std::nullopt;
    check(canonicalResult && highResult
              && std::abs(canonicalResult->outputLinearSrgb.red
                          - highResult->outputLinearSrgb.red)
                   <= validation::kFiniteSceneMaximumZeroClampedLinearSrgbChannelError
              && std::abs(canonicalResult->outputLinearSrgb.green
                          - highResult->outputLinearSrgb.green)
                   <= validation::kFiniteSceneMaximumZeroClampedLinearSrgbChannelError
              && std::abs(canonicalResult->outputLinearSrgb.blue
                          - highResult->outputLinearSrgb.blue)
                   <= validation::kFiniteSceneMaximumZeroClampedLinearSrgbChannelError,
          "sinusoidal-order-eight meets its finite-scene oracle tolerance");

    const auto specularEfficiency = profileOrderEfficiency(
        GrooveProfile::BinaryRectangular, 500.0, 160.0, 0.5, 0.8, 1);
    const auto explicitSpecularEfficiency = reflectiveProfileOrderEfficiency(
        GrooveProfile::BinaryRectangular, 500.0, 160.0, 0.5, 0.8, 0.8, 1);
    const auto diffractedEfficiency = reflectiveProfileOrderEfficiency(
        GrooveProfile::BinaryRectangular, 500.0, 160.0, 0.5, 0.8, 0.55, 1);
    check(near(specularEfficiency, explicitSpecularEfficiency)
              && !near(specularEfficiency, diffractedEfficiency),
          "reflective relief phase depends on both incident and outgoing angle");

    const auto smoothFraction = coherentRoughnessFraction(500.0, 0.0, 1.0);
    const auto roughFraction = coherentRoughnessFraction(500.0, 20.0, 1.0);
    check(near(smoothFraction, 1.0)
              && roughFraction > 0.0 && roughFraction < smoothFraction,
          "RMS height attenuates coherent orders without creating energy");

    const OpticalConstants aluminium { 0.17f, 3.10f };
    const SurfaceCoating uncoated { CoatingModel::Uncoated, 0.0f, {} };
    const SurfaceCoating coated {
        CoatingModel::IncoherentDielectric, 1000.0f, { 1.50f, 0.001f }
    };
    const auto uncoatedReflectance = incoherentCoatedSubstrateReflectance(
        500.0, aluminium, uncoated);
    const auto coatedReflectance = incoherentCoatedSubstrateReflectance(
        500.0, aluminium, coated);
    check(uncoatedReflectance >= 0.0 && uncoatedReflectance <= 1.0
              && coatedReflectance >= 0.0 && coatedReflectance <= 1.0
              && !near(uncoatedReflectance, coatedReflectance),
          "the incoherent coating model stays bounded and changes reflectance");

    for (const auto& scene : validation::wavelengthValidationScenes())
        checkPreset(scene);

    std::string error;
    auto smoothDescription = makeAluminiumBinaryGratingPreset();
    smoothDescription.roughness = {};
    auto roughDescription = smoothDescription;
    roughDescription.roughness = { 20.0f, 0.08f };
    const auto smooth = admit(smoothDescription, error);
    const auto rough = admit(roughDescription, error);
    EvaluationFailure failure = EvaluationFailure::None;
    const auto smoothResult = evaluate(*smooth, {}, failure);
    const auto roughResult = evaluate(*rough, {}, failure);
    check(smoothResult && roughResult
              && roughResult->resolvedReflectedEnergy
                   < smoothResult->resolvedReflectedEnergy
              && roughResult->unresolvedReflectedEnergy
                   > smoothResult->unresolvedReflectedEnergy
              && near(roughResult->events[0].angularStandardDeviationRadians,
                      0.16, 1.0e-7),
          "roughness moves coherent energy to the unresolved ledger and records width");

    double integratedLobe = 0.0;
    constexpr double lobeStep = 0.004;
    for (double y = -0.8; y <= 0.8; y += lobeStep)
        for (double x = -0.8; x <= 0.8; x += lobeStep)
            integratedLobe += gaussianOrderLobeDensity(
                { 0.0, 0.0, 1.0 }, { x, y, 1.0 }, 0.16)
                * lobeStep * lobeStep;
    check(near(integratedLobe, 1.0, 2.0e-4),
          "the RMS-slope Gaussian order lobe is normalized in the tangent plane");

    auto directionalDescription = makeAluminiumBinaryGratingPreset();
    const auto directionalAdmitted = admit(directionalDescription, error);
    const auto firstOrderDirection = solveOrder(
        { 0.0, 0.0, 1.0 }, { 1.0, 0.0 },
        directionalDescription.geometry.grooveSpacingNanometres,
        540.0, 1);
    const auto peak = evaluateDirectionalBsdf(
        *directionalAdmitted, {}, firstOrderDirection.outgoingDirection, failure);
    const auto offPeak = evaluateDirectionalBsdf(
        *directionalAdmitted, {}, { 0.0, 0.75, 0.6614378278 }, failure);
    check(peak && offPeak
              && peak->radianceXyz.y > offPeak->radianceXyz.y * 10.0,
          "the directional oracle resolves a bright order peak and off-peak falloff");

    auto broadDescription = directionalDescription;
    broadDescription.roughness.rmsSlope = 0.16f;
    const auto broadAdmitted = admit(broadDescription, error);
    const auto broadPeak = evaluateDirectionalBsdf(
        *broadAdmitted, {}, firstOrderDirection.outgoingDirection, failure);
    check(broadPeak && peak
              && broadPeak->radianceXyz.y < peak->radianceXyz.y,
          "increasing RMS slope broadens the lobe and lowers its normalized peak");

    auto rotatedDescription = directionalDescription;
    rotatedDescription.geometry.directionUv = { 0.0f, 1.0f };
    const auto rotatedAdmitted = admit(rotatedDescription, error);
    const auto rotatedOrderDirection = solveOrder(
        { 0.0, 0.0, 1.0 }, { 0.0, 1.0 },
        rotatedDescription.geometry.grooveSpacingNanometres,
        540.0, 1);
    const auto rotatedPeak = evaluateDirectionalBsdf(
        *rotatedAdmitted, {}, rotatedOrderDirection.outgoingDirection, failure);
    const auto rotatedAtOriginal = evaluateDirectionalBsdf(
        *rotatedAdmitted, {}, firstOrderDirection.outgoingDirection, failure);
    check(rotatedPeak && rotatedAtOriginal
              && rotatedPeak->radianceXyz.y > rotatedAtOriginal->radianceXyz.y * 10.0,
          "rotating the reciprocal-lattice direction rotates the rendered order lobe");

    auto spatialDescription = makeCrossedTwoDimensionalGratingPreset();
    spatialDescription.grooveField.mode = GrooveFieldMode::Linear;
    spatialDescription.grooveField.originUv = { 0.5f, 0.5f };
    spatialDescription.grooveField.axisUv = { 1.0f, 0.0f };
    spatialDescription.grooveField.grooveSpacingDeltaNanometresPerUnit = 400.0f;
    spatialDescription.grooveField.secondarySpacingDeltaNanometresPerUnit = -200.0f;
    spatialDescription.grooveField.orientationDegreesPerUnit = 60.0f;
    spatialDescription.spectrum.lastOrder = 1;
    const auto spatialAdmitted = admit(spatialDescription, error);
    const auto leftGeometry = resolveLocalGrooveGeometry(
        spatialAdmitted->description(), { 0.0, 0.5 }).geometry;
    const auto rightGeometry = resolveLocalGrooveGeometry(
        spatialAdmitted->description(), { 1.0, 0.5 }).geometry;
    constexpr double thirtyDegrees = 0.52359877559829887308;
    check(near(leftGeometry.grooveSpacingNanometres, 800.0, 1.0e-6)
              && near(rightGeometry.grooveSpacingNanometres, 1200.0, 1.0e-6)
              && near(leftGeometry.directionUv[0], std::cos(-thirtyDegrees), 1.0e-6)
              && near(leftGeometry.directionUv[1], std::sin(-thirtyDegrees), 1.0e-6)
              && near(rightGeometry.directionUv[0], std::cos(thirtyDegrees), 1.0e-6)
              && near(rightGeometry.directionUv[1], std::sin(thirtyDegrees), 1.0e-6)
              && near(leftGeometry.secondaryGrooveSpacingNanometres, 1500.0, 1.0e-6)
              && near(rightGeometry.secondaryGrooveSpacingNanometres, 1300.0, 1.0e-6)
              && near(leftGeometry.directionUv[0] * leftGeometry.secondaryDirectionUv[0]
                          + leftGeometry.directionUv[1] * leftGeometry.secondaryDirectionUv[1],
                      0.0, 1.0e-6)
              && near(rightGeometry.directionUv[0] * rightGeometry.secondaryDirectionUv[0]
                          + rightGeometry.directionUv[1] * rightGeometry.secondaryDirectionUv[1],
                      0.0, 1.0e-6),
          "the CPU oracle evaluates local crossed periods and axial orientation at each material UV");

    const auto admitted = admit(makeAluminiumBinaryGratingPreset(), error);
    EvaluationInput invalid;
    invalid.incidentDirection = { 0.0, 0.0, 0.0 };
    check(!evaluate(*admitted, invalid, failure)
              && failure == EvaluationFailure::InvalidIncidentDirection,
          "invalid incident directions fail closed");
    invalid = {};
    invalid.incidentSpectrum[3] = std::numeric_limits<double>::quiet_NaN();
    check(!evaluate(*admitted, invalid, failure)
              && failure == EvaluationFailure::InvalidSpectrum,
          "nonfinite incident spectra fail closed");
    invalid = {};
    invalid.incidentSpectrum[4] = 17.0;
    check(!evaluate(*admitted, invalid, failure)
              && failure == EvaluationFailure::InvalidSpectrum,
          "incident radiance above the hard bound fails closed");

    const auto spatialFoil = makeDiffractionReferenceScene(
        ReferenceSceneKind::SpatialFoil);
    const auto preview = makePreviewRequest(spatialFoil);
    const auto exportRequest = makeExportRequest(spatialFoil);
    check(preview.plan != nullptr && exportRequest.plan != nullptr
              && preview.plan != exportRequest.plan
              && preview.productDigest == exportRequest.productDigest
              && preview.plan->kind == spatialFoil.kind
              && exportRequest.plan->material.maskCoverage
                  == spatialFoil.material.maskCoverage,
          "preview and export own isolated copies of the same exact diffraction scene plan");
    check(spatialFoil.material.patternMask == PatternMask::CardFrameAndEmblem
              && spatialFoil.material.diffractionBsdf.coating.model
                  == CoatingModel::IncoherentDielectric
              && spatialFoil.material.diffractionBsdf.grooveField.mode
                  == GrooveFieldMode::Linear,
          "Trading Card Foil nests coating, embossed spatial grooves, and a pattern mask");
    const auto lighting = AdmittedLightingPlan::admit(spatialFoil.lighting, error);
    check(lighting && lighting->description().pathCount == 3
              && lighting->hasIndirectBounce(),
          "reference lighting admits bounded direct, environment, and one-bounce paths");
    double transportIncident = 0.0;
    double transportReflected = 0.0;
    double transportAbsorbed = 0.0;
    bool transportOracleOk = lighting.has_value();
    if (lighting)
    {
        const auto foilMaterial = admit(makeAluminiumBinaryGratingPreset(), error);
        for (std::size_t pathIndex = 0;
             foilMaterial && pathIndex < lighting->description().pathCount; ++pathIndex)
        {
            EvaluationInput pathInput;
            const auto& path = lighting->description().paths[pathIndex];
            pathInput.incidentDirection = {
                path.incident.direction[0], path.incident.direction[1],
                path.incident.direction[2] };
            for (std::size_t wavelength = 0; wavelength < pathInput.incidentSpectrum.size();
                 ++wavelength)
                pathInput.incidentSpectrum[wavelength] = path.incident.radiance[wavelength];
            const auto pathResult = evaluate(*foilMaterial, pathInput, failure);
            const auto highSample = evaluateHighSampleReference(
                *foilMaterial, pathInput, failure);
            transportOracleOk = transportOracleOk && pathResult && highSample
                && near(pathResult->resolvedReflectedEnergy,
                        highSample->resolvedReflectedEnergy, 2.5e-3)
                && near(pathResult->incidentEnergy,
                        highSample->incidentEnergy, 1.0e-9);
            if (pathResult)
            {
                transportIncident += pathResult->incidentEnergy;
                transportReflected += pathResult->substrateReflectedEnergy;
                transportAbsorbed += pathResult->absorbedEnergy;
            }
        }
    }
    check(transportOracleOk
              && near(transportIncident, transportReflected + transportAbsorbed, 1.0e-6),
          "independent high-sample oracle covers direct, environment, and indirect energy");

    auto rejectedLighting = spatialFoil.lighting;
    rejectedLighting.pathCount = static_cast<std::uint8_t>(kMaximumLightingPaths + 1);
    check(!AdmittedLightingPlan::admit(rejectedLighting, error),
          "lighting over the backend path bound rejects before executor allocation");
    rejectedLighting = spatialFoil.lighting;
    rejectedLighting.paths[0].kind = static_cast<LightingPathKind>(255);
    check(!AdmittedLightingPlan::admit(rejectedLighting, error),
          "unknown lighting path kinds reject before executor allocation");
    rejectedLighting = spatialFoil.lighting;
    rejectedLighting.paths[2].bounceDepth = kMaximumIndirectBounces + 1;
    check(!AdmittedLightingPlan::admit(rejectedLighting, error),
          "excessive lighting bounce depth rejects before executor allocation");
    rejectedLighting = spatialFoil.lighting;
    rejectedLighting.paths[1].incident.direction[0]
        = std::numeric_limits<float>::quiet_NaN();
    check(!AdmittedLightingPlan::admit(rejectedLighting, error),
          "malformed lighting directions reject before executor allocation");

    std::printf("diffraction reference oracle: %d/%d checks passed\n",
                checks - failures, checks);
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
