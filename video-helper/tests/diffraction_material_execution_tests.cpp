#include "diffraction_material_execution.h"
#include "DiffractionMaterialPresets.h"
#include "gpu_backend/backend.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

namespace
{
int failures = 0;
int checks = 0;

void check(bool condition, const char* message)
{
    ++checks;
    if (!condition)
    {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}
} // namespace

int main()
{
    using namespace diffractionmaterial;
    using diffractionmaterial::physicalcheckpoint::validateGpuReadback;

    std::string error;
    auto description = makeAluminiumBinaryGratingPreset();
    description.roughness = { 0.0f, 0.01f };
    const auto admitted = admit(description, error);
    check(admitted && error.empty(),
          "the finite-lobe binary fixture reaches strict execution admission");
    if (!admitted)
        return EXIT_FAILURE;

    SpectralIncidentLight light;
    check(physicalcheckpoint::validate(*admitted, light, "test", error)
              && error.empty(),
          "the canonical visible spectrum and direct light pass the strict checkpoint");
    const auto parameters = physicalcheckpoint::makeGpuParameters(*admitted, light);
    check(parameters.geometry.x == description.geometry.directionUv[0]
              && parameters.geometry.y == description.geometry.directionUv[1]
              && parameters.geometry.z == description.geometry.grooveSpacingNanometres
              && parameters.geometry.w
                   == description.microstructure.grooveDepthNanometres
              && parameters.secondaryGeometry.x == 0.0f
              && parameters.secondaryGeometry.y == 0.0f
              && parameters.secondaryGeometry.z == 0.0f
              && parameters.secondaryGeometry.w
                   == static_cast<float>(GratingLattice::OneDimensional)
              && parameters.microstructure.x == description.microstructure.dutyCycle
              && parameters.microstructure.y == description.substrate.refractiveIndex
              && parameters.microstructure.z
                   == description.substrate.extinctionCoefficient
              && parameters.microstructure.w
                   == static_cast<float>(description.spectrum.firstOrder)
              && parameters.control.x
                   == static_cast<float>(description.spectrum.lastOrder)
              && parameters.control.y
                   == static_cast<float>(description.microstructure.profile)
              && parameters.control.w
                   == static_cast<float>(CoatingModel::Uncoated)
              && parameters.coating.x == 0.0f
              && parameters.roughness.x == 0.0f,
          "the shared GPU wire block carries physical geometry and optical constants");
    check(arbitgpu::validNativeFixtureDiffractionParameters(parameters),
          "the backend preflight accepts the canonical packed diffraction block");

    const HarmonicMIDI::grid::SceneObjectId backendObject { 17 };
    arbitgpu::NativeFixtureSurfaceMaterialProgram backendProgram;
    backendProgram.backend = arbitgpu::NativeFixtureMaterialBackend::OpenGl;
    backendProgram.kind = arbitgpu::NativeFixtureMaterialKind::DiffractionReflective;
    backendProgram.object = backendObject;
    backendProgram.bindingDigest = "hostile-direct-backend-binding";
    backendProgram.programIdentity = "hostile-direct-backend-program";
    SpectralLightingPath backendLightingPath;
    backendLightingPath.kind = LightingPathKind::Direct;
    backendLightingPath.incident = light;
    LightingDescription backendLighting;
    backendLighting.pathCount = 1;
    backendLighting.paths[0] = backendLightingPath;
    const auto admittedBackendLighting = AdmittedLightingPlan::admit(
        backendLighting, error);
    backendProgram.diffractionPathCount = 1;
    if (admittedBackendLighting)
    {
        backendProgram.diffractionLightingAdmission
            = std::make_shared<const AdmittedLightingPlan>(*admittedBackendLighting);
        backendProgram.programIdentity = arbitgpu::nativeFixtureDiffractionProgramIdentity (
            backendProgram.bindingDigest, *admittedBackendLighting);
    }
    backendProgram.diffractionPaths[0] = makeGpuLightingPath(
        parameters, backendLightingPath);
    check(arbitgpu::validNativeFixtureDiffractionProgram(
              backendProgram, arbitgpu::NativeFixtureMaterialBackend::OpenGl,
              backendObject),
          "the direct backend boundary accepts one exact canonical diffraction envelope");
    const auto rejectsProgramMutation = [&] (const auto& mutate)
    {
        auto candidate = backendProgram;
        mutate(candidate);
        return !arbitgpu::validNativeFixtureDiffractionProgram(
            candidate, arbitgpu::NativeFixtureMaterialBackend::OpenGl,
            backendObject);
    };
    check(rejectsProgramMutation([] (auto& value) { value.layoutVersion += 1; })
              && rejectsProgramMutation([] (auto& value) {
                     value.backend = arbitgpu::NativeFixtureMaterialBackend::Metal;
                 })
              && rejectsProgramMutation([] (auto& value) {
                     value.kind = arbitgpu::NativeFixtureMaterialKind::SurfacePbr;
                 })
              && rejectsProgramMutation([] (auto& value) { value.object.value += 1; })
              && rejectsProgramMutation([] (auto& value) { value.bindingDigest.clear(); })
              && rejectsProgramMutation([] (auto& value) { value.programIdentity.clear(); })
              && rejectsProgramMutation([] (auto& value) {
                     value.baseColorSource = decltype(value.baseColorSource)::TimeLinearMix;
                 })
              && rejectsProgramMutation([] (auto& value) {
                     value.timeMixEndColor[2] = 1.0f;
                 })
              && rejectsProgramMutation([] (auto& value) {
                     value.parameters.baseColorMetallic[0] = 1.0f;
                 })
              && rejectsProgramMutation([] (auto& value) {
                     value.parameters.emissionRoughness[3] = 1.0f;
                 })
              && rejectsProgramMutation([] (auto& value) {
                     value.parameters.normalOpacity[2] = 1.0f;
                 })
              && rejectsProgramMutation([] (auto& value) {
                     value.parameters.transmissionIorClearcoat[1] = 1.0f;
                 })
              && rejectsProgramMutation([] (auto& value) {
                     value.parameters.identifiers[3] = 1u;
                 })
              && rejectsProgramMutation([] (auto& value) {
                     value.importedBaseColorTexture.emplace();
                 })
              && rejectsProgramMutation([] (auto& value) {
                     value.diffractionPaths[0].material.incident.x += 0.01f;
                 })
              && rejectsProgramMutation([] (auto& value) {
                     value.diffractionPaths[0].material.spectral[3].y += 0.01f;
                 }),
          "the backend rejects mutated envelope, direction, and spectral fields");
    auto zeroSlopeParameters = parameters;
    zeroSlopeParameters.roughness.y = 0.0f;
    check(!arbitgpu::validNativeFixtureDiffractionParameters(zeroSlopeParameters),
          "surface backend preflight rejects a zero-width angular lobe");

    const auto mutateGpuField = [] (
        PhysicalDiffractionGpuParameters& candidate, std::size_t target, float replacement)
    {
        std::size_t current = 0;
        const auto mutateLane = [&] (GpuFloat4& lane)
        {
            for (auto* field : std::array<float*, 4> {
                     &lane.x, &lane.y, &lane.z, &lane.w })
            {
                if (current == target)
                    *field = replacement;
                ++current;
            }
        };
        mutateLane(candidate.geometry);
        mutateLane(candidate.secondaryGeometry);
        mutateLane(candidate.microstructure);
        mutateLane(candidate.control);
        mutateLane(candidate.incident);
        mutateLane(candidate.coating);
        mutateLane(candidate.roughness);
        mutateLane(candidate.grooveField);
        mutateLane(candidate.grooveVariation);
        for (auto& lane : candidate.spectral) mutateLane(lane);
        for (auto& lane : candidate.spectralZ) mutateLane(lane);
        return current;
    };
    constexpr std::size_t gpuFieldCount
        = (9u + 2u * kMaximumSpectralSamples) * 4u;
    bool everyNonfiniteFieldRejected = true;
    for (std::size_t field = 0; field < gpuFieldCount; ++field)
    {
        auto malformedParameters = parameters;
        const auto visited = mutateGpuField(
            malformedParameters, field, std::numeric_limits<float>::quiet_NaN());
        everyNonfiniteFieldRejected &= visited == gpuFieldCount
            && !arbitgpu::validNativeFixtureDiffractionParameters(malformedParameters);
    }
    check(everyNonfiniteFieldRejected,
          "backend preflight rejects a nonfinite value in every packed GPU field");

    auto malformedParameters = parameters;
    malformedParameters.secondaryGeometry.w = 4.0f;
    check(!arbitgpu::validNativeFixtureDiffractionParameters(malformedParameters),
          "backend preflight rejects unknown lattice tokens");
    malformedParameters = parameters;
    malformedParameters.control.y = 4.0f;
    check(!arbitgpu::validNativeFixtureDiffractionParameters(malformedParameters),
          "backend preflight rejects unknown profile tokens");
    malformedParameters = parameters;
    malformedParameters.control.w = 3.0f;
    check(!arbitgpu::validNativeFixtureDiffractionParameters(malformedParameters),
          "backend preflight rejects unknown coating tokens");
    malformedParameters = parameters;
    malformedParameters.grooveField.x = 4.0f;
    check(!arbitgpu::validNativeFixtureDiffractionParameters(malformedParameters),
          "backend preflight rejects unknown groove-field tokens");
    malformedParameters = parameters;
    malformedParameters.control.z = 1.0f;
    check(!arbitgpu::validNativeFixtureDiffractionParameters(malformedParameters),
          "backend preflight rejects noncanonical reserved lanes");
    malformedParameters = parameters;
    malformedParameters.spectral[2].x += 1.0f;
    check(!arbitgpu::validNativeFixtureDiffractionParameters(malformedParameters),
          "backend preflight rejects a noncanonical spectral table");
    malformedParameters = parameters;
    malformedParameters.geometry.x = 0.5f;
    check(!arbitgpu::validNativeFixtureDiffractionParameters(malformedParameters),
          "backend preflight rejects non-normalized primary directions");
    malformedParameters = parameters;
    malformedParameters.incident.z = 0.5f;
    check(!arbitgpu::validNativeFixtureDiffractionParameters(malformedParameters),
          "backend preflight rejects non-normalized incident directions");
    malformedParameters = parameters;
    malformedParameters.geometry.z = kMinimumGrooveSpacingNanometres - 1.0f;
    check(!arbitgpu::validNativeFixtureDiffractionParameters(malformedParameters),
          "backend preflight rejects out-of-range base periods");
    malformedParameters = parameters;
    malformedParameters.microstructure.w = 1.5f;
    check(!arbitgpu::validNativeFixtureDiffractionParameters(malformedParameters),
          "backend preflight rejects fractional diffraction order ranges");
    malformedParameters = parameters;
    malformedParameters.grooveField = {
        static_cast<float>(GrooveFieldMode::Linear), 0.5f, 0.5f, 0.0f
    };
    malformedParameters.grooveVariation = { 1.0f, 0.0f, -2000.0f, 0.0f };
    check(!arbitgpu::validNativeFixtureDiffractionParameters(malformedParameters),
          "backend preflight rejects spatial gradients that cross the physical period bound");

    auto orderOneCrossed = parameters;
    orderOneCrossed.secondaryGeometry = { 0.0f, 1.0f, parameters.geometry.z,
        static_cast<float>(GratingLattice::CrossedTwoDimensional) };
    orderOneCrossed.microstructure.w = 1.0f;
    orderOneCrossed.control.x = 1.0f;
    check(arbitgpu::validNativeFixtureDiffractionParameters(orderOneCrossed)
              && arbitgpu::nativeFixtureDiffractionLobeEvaluations(
                     [&] {
                         auto value = backendProgram;
                         value.diffractionPaths[0].material = orderOneCrossed;
                         return value;
                     }(), 1, 1) == kMaximumSpectralSamples * 9u,
          "crossed order-one budgeting preserves zero, four corner, and four axial lobes");

    diffractivefoil::EvaluationSchedule schedule;
    auto exactSpatialBudget = backendProgram;
    exactSpatialBudget.diffractionFoilMaximumEvaluations
        = 4u * exactSpatialBudget.diffractionPathCount;
    check(arbitgpu::nativeFixtureSpatialFoilEvaluationSchedule(
              exactSpatialBudget, 2, 2, schedule)
              && schedule.requiredEvaluations
                  == exactSpatialBudget.diffractionFoilMaximumEvaluations
              && diffractivefoil::evaluationIndex(schedule, 0, 0, 0) == 0u
              && diffractivefoil::evaluationIndex(
                     schedule, 1, 1, schedule.pathCount - 1u)
                  == schedule.requiredEvaluations - 1u
              && arbitgpu::nativeFixtureDiffractionWorkWithinBudget(
                  exactSpatialBudget, 2, 2),
          "CPU oracle admits and indexes work exactly at the evaluation ceiling");
    ++exactSpatialBudget.diffractionFoilMaximumEvaluations;
    check(arbitgpu::nativeFixtureSpatialFoilEvaluationSchedule(
              exactSpatialBudget, 2, 2, schedule)
              && schedule.requiredEvaluations < schedule.maximumEvaluations,
          "CPU oracle admits work below the evaluation ceiling");
    exactSpatialBudget.diffractionFoilMaximumEvaluations
        = schedule.requiredEvaluations - 1u;
    check(!arbitgpu::nativeFixtureSpatialFoilEvaluationSchedule(
              exactSpatialBudget, 2, 2, schedule)
              && !arbitgpu::nativeFixtureDiffractionWorkWithinBudget(
                  exactSpatialBudget, 2, 2),
          "CPU oracle rejects work one evaluation above the ceiling");
    check(!diffractivefoil::makeEvaluationSchedule(0, 2, 2, 1, schedule)
              && !diffractivefoil::makeEvaluationSchedule(
                  diffractivefoil::kMaximumEvaluations, 0, 2, 1, schedule)
              && !diffractivefoil::makeEvaluationSchedule(
                  diffractivefoil::kMaximumEvaluations,
                  std::numeric_limits<std::uint32_t>::max(), 2, 2, schedule),
          "CPU oracle rejects zero, malformed, and overflowing schedules");

    auto crossedDescription = makeCrossedTwoDimensionalGratingPreset();
    const auto crossed = admit(crossedDescription, error);
    check(crossed
              && physicalcheckpoint::validate(*crossed, light, "test", error)
              && error.empty(),
          "the strict native checkpoint admits the bounded crossed two-dimensional lattice");
    if (crossed)
    {
        const auto crossedParameters =
            physicalcheckpoint::makeGpuParameters(*crossed, light);
        check(crossedParameters.secondaryGeometry.x
                      == crossedDescription.geometry.secondaryDirectionUv[0]
                  && crossedParameters.secondaryGeometry.y
                      == crossedDescription.geometry.secondaryDirectionUv[1]
                  && crossedParameters.secondaryGeometry.z
                      == crossedDescription.geometry.secondaryGrooveSpacingNanometres
                  && crossedParameters.secondaryGeometry.w
                      == static_cast<float>(GratingLattice::CrossedTwoDimensional)
                  && physicalcheckpoint::expectedSignedOrderEvaluations(*crossed)
                      == kMaximumSpectralSamples * 80u,
              "the GPU wire block and order ledger carry the crossed lattice exactly");
        auto malformedCrossedBlazed = crossedParameters;
        malformedCrossedBlazed.control.y
            = static_cast<float>(GrooveProfile::BlazedSawtooth);
        malformedCrossedBlazed.secondaryGeometry.z
            = malformedCrossedBlazed.geometry.z + 1.0f;
        check(!arbitgpu::validNativeFixtureDiffractionParameters(
                  malformedCrossedBlazed),
              "packed backend preflight rejects unequal crossed blazed periods");
    }

    auto sinusoidalDescription = makeSinusoidalGratingPreset();
    sinusoidalDescription.roughness = {};
    const auto sinusoidal = admit(sinusoidalDescription, error);
    check(sinusoidal
              && physicalcheckpoint::validate(*sinusoidal, light, "test", error)
              && error.empty(),
          "the strict native checkpoint admits the bounded sinusoidal profile");
    if (sinusoidal)
    {
        const auto sinusoidalParameters =
            physicalcheckpoint::makeGpuParameters(*sinusoidal, light);
        check(sinusoidalParameters.control.y
                  == static_cast<float>(GrooveProfile::Sinusoidal),
              "the GPU wire block identifies the sinusoidal Fourier model");
    }

    auto blazedDescription = makeBlazedGratingPreset();
    blazedDescription.coating = { CoatingModel::Uncoated, 0.0f, {} };
    blazedDescription.roughness = {};
    const auto blazed = admit(blazedDescription, error);
    check(blazed
              && physicalcheckpoint::validate(*blazed, light, "test", error)
              && error.empty(),
          "the strict native checkpoint admits the bounded blazed-sawtooth profile");
    if (blazed)
    {
        const auto blazedParameters =
            physicalcheckpoint::makeGpuParameters(*blazed, light);
        check(blazedParameters.control.y
                  == static_cast<float>(GrooveProfile::BlazedSawtooth),
              "the GPU wire block identifies the blazed Fourier model");
        auto spatialBlazedParameters = blazedParameters;
        spatialBlazedParameters.roughness.y = 0.01f;
        spatialBlazedParameters.grooveField = {
            static_cast<float>(GrooveFieldMode::Linear), 0.5f, 0.5f, 10.0f
        };
        spatialBlazedParameters.grooveVariation = { 1.0f, 0.0f, 1.0f, 0.0f };
        check(!arbitgpu::validNativeFixtureDiffractionParameters(
                  spatialBlazedParameters),
              "packed blazed profiles reject period variation at the backend boundary");
    }

    auto overboundSinusoidalDescription = sinusoidalDescription;
    overboundSinusoidalDescription.microstructure.grooveDepthNanometres = 250.0f;
    const auto overboundSinusoidal = admit(overboundSinusoidalDescription, error);
    check(overboundSinusoidal
              && !physicalcheckpoint::validate(
                  *overboundSinusoidal, light, "test", error)
              && error
                   == "physical diffraction test sinusoidal profile exceeds the "
                      "bounded phase argument",
          "sinusoidal execution rejects phase arguments outside its analytic bound");

    auto extendedDescription = description;
    extendedDescription.coating = {
        CoatingModel::IncoherentDielectric, 1000.0f, { 1.5f, 0.001f }
    };
    const auto coated = admit(extendedDescription, error);
    check(coated && physicalcheckpoint::validate(*coated, light, "test", error),
          "the native checkpoint admits incoherent dielectric coatings");
    if (coated)
    {
        const auto coatedParameters =
            physicalcheckpoint::makeGpuParameters(*coated, light);
        check(coatedParameters.control.w
                  == static_cast<float>(CoatingModel::IncoherentDielectric)
                  && coatedParameters.coating.x == 1000.0f
                  && coatedParameters.coating.y == 1.5f
                  && coatedParameters.coating.z == 0.001f,
              "the GPU wire block carries bounded coating optical constants");
    }

    extendedDescription = description;
    extendedDescription.roughness.rmsHeightNanometres = 1.0f;
    const auto roughHeight = admit(extendedDescription, error);
    check(roughHeight
              && physicalcheckpoint::validate(*roughHeight, light, "test", error),
          "the native checkpoint admits coherent RMS-height attenuation");
    if (roughHeight)
    {
        const auto roughParameters =
            physicalcheckpoint::makeGpuParameters(*roughHeight, light);
        check(roughParameters.roughness.x == 1.0f,
              "the GPU wire block carries RMS height in nanometres");
    }
    extendedDescription = description;
    extendedDescription.roughness.rmsSlope = 0.01f;
    const auto roughSlope = admit(extendedDescription, error);
    check(roughSlope
              && physicalcheckpoint::validate(*roughSlope, light, "test", error),
          "the native checkpoint admits bounded RMS-slope broadening");
    if (roughSlope)
    {
        const auto roughParameters =
            physicalcheckpoint::makeGpuParameters(*roughSlope, light);
        check(roughParameters.roughness.y == 0.01f,
              "the GPU wire block carries the RMS tangent-plane slope");
    }

    extendedDescription = description;
    extendedDescription.spectrum.wavelengthsNanometres[1] = 410.0f;
    const auto alternateWavelengths = admit(extendedDescription, error);
    check(!alternateWavelengths
              && error
                   == "diffraction material production tier requires the canonical wavelength schedule",
          "shared admission rejects alternate wavelength schedules before backend creation");

    auto invalidLight = light;
    invalidLight.direction = { 0.0f, 0.0f, 0.5f };
    check(!physicalcheckpoint::validate(*admitted, invalidLight, "test", error)
              && error == "physical diffraction test incident direction is invalid",
          "non-unit incident directions fail before command submission");
    invalidLight = light;
    invalidLight.direction[0] = std::numeric_limits<float>::infinity();
    check(!physicalcheckpoint::validate(*admitted, invalidLight, "test", error)
              && error == "physical diffraction test incident direction is invalid",
          "nonfinite incident directions fail before command submission");
    invalidLight = light;
    invalidLight.radiance[3] = std::numeric_limits<float>::quiet_NaN();
    check(!physicalcheckpoint::validate(*admitted, invalidLight, "test", error)
              && error == "physical diffraction test incident spectrum is invalid",
          "nonfinite incident spectra fail before command submission");
    invalidLight = light;
    invalidLight.radiance[3] = physicalcheckpoint::kMaximumSpectralRadiance + 1.0f;
    check(!physicalcheckpoint::validate(*admitted, invalidLight, "test", error)
              && error == "physical diffraction test incident spectrum is invalid",
          "incident spectra above the hard bound fail before command submission");

    const std::array<float, 4> color { 0.2f, 0.1f, 0.4f, 1.0f };
    const std::array<float, 4> energy { 1.0f, 0.8f, 0.3f, 0.2f };
    const std::array<float, 4> orders { 0.3f, 0.2f, 80.0f, 48.0f };
    const std::array<float, 4> moment { 0.1f, -0.1f, 0.2f, 0.5f };
    check(validateGpuReadback(color, energy, orders, moment, 128),
          "a bounded energy-conserving native readback is valid");

    auto sentinelColor = color;
    sentinelColor[3] = -1.0f;
    check(!validateGpuReadback(sentinelColor, energy, orders, moment, 128),
          "the shader rejection sentinel fails closed");
    auto nonfiniteColor = color;
    nonfiniteColor[0] = std::numeric_limits<float>::infinity();
    check(!validateGpuReadback(nonfiniteColor, energy, orders, moment, 128),
          "a nonfinite native color fails closed");
    auto nonfiniteEnergy = energy;
    nonfiniteEnergy[2] = std::numeric_limits<float>::quiet_NaN();
    check(!validateGpuReadback(color, nonfiniteEnergy, orders, moment, 128),
          "a nonfinite native ledger fails closed");
    auto negativeEnergy = energy;
    negativeEnergy[0] = -0.1f;
    check(!validateGpuReadback(color, negativeEnergy, orders, moment, 128),
          "negative native energy fails closed");
    auto unbalancedEnergy = energy;
    unbalancedEnergy[3] = 0.4f;
    check(!validateGpuReadback(color, unbalancedEnergy, orders, moment, 128),
          "an unbalanced energy ledger fails closed");
    auto missingOrders = orders;
    missingOrders[3] = 47.0f;
    check(!validateGpuReadback(color, energy, missingOrders, moment, 128),
          "missing signed-order evaluations fail closed");
    auto fractionalOrders = orders;
    fractionalOrders[2] = 80.25f;
    fractionalOrders[3] = 47.75f;
    check(!validateGpuReadback(color, energy, fractionalOrders, moment, 128),
          "fractional order counters fail closed");
    auto impossibleMoment = moment;
    impossibleMoment[0] = 0.6f;
    check(!validateGpuReadback(color, energy, orders, impossibleMoment, 128),
          "a direction moment larger than resolved energy fails closed");
    check(!validateGpuReadback(color, energy, orders, moment, 0),
          "an empty order domain fails closed");
    check(!validateGpuReadback(
              color, energy, orders, moment,
              kMaximumSpectralSamples
                      * ((2u * kMaximumDiffractionOrder + 1u)
                         * (2u * kMaximumDiffractionOrder + 1u) - 1u)
                  + 1u),
          "an order domain above the hard spectral bound fails closed");

    std::cout << "diffraction material execution: "
              << checks - failures << '/' << checks << " checks passed\n";
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
