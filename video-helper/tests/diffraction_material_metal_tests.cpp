#include "diffraction_material_metal.h"
#include "../../shared/DiffractionMaterialPresets.h"
#include "support/diffraction_reference_oracle.h"
#include "support/diffraction_wavelength_validation_scenes.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace
{
bool near(float actual, double expected, double tolerance = 3.0e-4)
{
    return std::abs(static_cast<double>(actual) - expected) <= tolerance;
}

std::array<double, 3> expectedDirectionMoment(
    const diffractionmaterial::reference::EvaluationResult& result)
{
    std::array<double, 3> moment {};
    for (std::size_t index = 0; index < result.eventCount; ++index)
    {
        const auto& event = result.events[index];
        const auto broadeningMoment = std::exp(
            -0.5 * event.angularStandardDeviationRadians
                * event.angularStandardDeviationRadians);
        moment[0] += broadeningMoment * event.energy * event.outgoingDirection.x;
        moment[1] += broadeningMoment * event.energy * event.outgoingDirection.y;
        moment[2] += broadeningMoment * event.energy * event.outgoingDirection.z;
    }
    return moment;
}

bool compareCase(diffractionmaterial::MetalPhysicalDiffractionExecutor& executor,
                 const diffractionmaterial::AdmittedDiffractionMaterialIR& material,
                 const diffractionmaterial::SpectralIncidentLight& light,
                 std::uint64_t expectedGeneration,
                 std::string& error,
                 diffractionmaterial::MetalPhysicalDiffractionFrame* captured = nullptr,
                 const diffractionmaterial::reference::EvaluationInput* suppliedInput = nullptr)
{
    using namespace diffractionmaterial;
    using namespace diffractionmaterial::reference;

    MetalPhysicalDiffractionFrame frame;
    const std::array<float, 2> materialUv {
        suppliedInput ? static_cast<float>(suppliedInput->materialUv[0]) : 0.5f,
        suppliedInput ? static_cast<float>(suppliedInput->materialUv[1]) : 0.5f
    };
    if (!executor.executeAtMaterialUv(material, light, materialUv, frame, error))
        return false;

    EvaluationInput input = suppliedInput ? *suppliedInput : EvaluationInput {};
    const auto directionLength = std::sqrt (
        static_cast<double> (light.direction[0]) * light.direction[0]
        + static_cast<double> (light.direction[1]) * light.direction[1]
        + static_cast<double> (light.direction[2]) * light.direction[2]);
    input.incidentDirection = {
        light.direction[0] / directionLength,
        light.direction[1] / directionLength,
        light.direction[2] / directionLength
    };
    for (std::size_t index = 0; index < light.radiance.size(); ++index)
        input.incidentSpectrum[index] = light.radiance[index];
    EvaluationFailure failure = EvaluationFailure::None;
    const auto oracle = evaluate(material, input, failure);
    if (!oracle)
    {
        error = "independent diffraction oracle rejected a Metal fixture (failure="
              + std::to_string (static_cast<int> (failure)) + ")";
        return false;
    }
    const auto moment = expectedDirectionMoment(*oracle);
    const auto nonzero = [](const auto& handles)
    {
        return std::all_of(handles.begin(), handles.end(),
                           [](std::uintptr_t handle) { return handle != 0; });
    };

    const bool matched = frame.generation == expectedGeneration
        && nonzero(frame.imageHandles)
        && nonzero(frame.textureViewHandles)
        && nonzero(frame.metalTextureHandles)
        && near(frame.linearSrgb[0], oracle->outputLinearSrgb.red)
        && near(frame.linearSrgb[1], oracle->outputLinearSrgb.green)
        && near(frame.linearSrgb[2], oracle->outputLinearSrgb.blue)
        && near(frame.linearSrgb[3], 1.0)
        && near(frame.energyLedger[0], oracle->incidentEnergy)
        && near(frame.energyLedger[1], oracle->substrateReflectedEnergy)
        && near(frame.energyLedger[2], oracle->zeroOrderEnergy)
        && near(frame.energyLedger[3], oracle->higherOrderEnergy)
        && near(frame.orderLedger[0], oracle->unresolvedReflectedEnergy)
        && near(frame.orderLedger[1], oracle->absorbedEnergy)
        && near(frame.orderLedger[2], static_cast<double>(
            oracle->eventCount - oracle->spectralSampleCount), 0.01)
        && near(frame.orderLedger[3], static_cast<double>(
            oracle->evanescentOrderCount + oracle->grazingOrderCount), 0.01)
        && near(frame.directionMoment[0], moment[0])
        && near(frame.directionMoment[1], moment[1])
        && near(frame.directionMoment[2], moment[2])
        && near(frame.directionMoment[3], oracle->resolvedReflectedEnergy);
    if (!matched)
    {
        error = "Metal diffraction pixels disagree with the independent oracle";
        std::cerr << "generation=" << frame.generation
                  << " expectedGeneration=" << expectedGeneration
                  << " color=" << frame.linearSrgb[0] << ','
                  << frame.linearSrgb[1] << ',' << frame.linearSrgb[2]
                  << " expectedColor=" << oracle->outputLinearSrgb.red << ','
                  << oracle->outputLinearSrgb.green << ','
                  << oracle->outputLinearSrgb.blue
                  << " energy=" << frame.energyLedger[0] << ','
                  << frame.energyLedger[1] << ',' << frame.energyLedger[2] << ','
                  << frame.energyLedger[3]
                  << " expectedEnergy=" << oracle->incidentEnergy << ','
                  << oracle->substrateReflectedEnergy << ','
                  << oracle->zeroOrderEnergy << ','
                  << oracle->higherOrderEnergy
                  << " orders=" << frame.orderLedger[2] << '/'
                  << frame.orderLedger[3]
                  << " expectedOrders="
                  << oracle->eventCount - oracle->spectralSampleCount << '/'
                  << oracle->evanescentOrderCount + oracle->grazingOrderCount
                  << " direction=" << frame.directionMoment[0] << ','
                  << frame.directionMoment[1] << ',' << frame.directionMoment[2] << ','
                  << frame.directionMoment[3]
                  << " expectedDirection=" << moment[0] << ',' << moment[1] << ','
                  << moment[2] << ',' << oracle->resolvedReflectedEnergy
                  << '\n';
    }
    else if (captured != nullptr)
    {
        *captured = frame;
    }
    return matched;
}

template <typename Handles>
bool disjoint(const Handles& first, const Handles& second)
{
    return std::none_of(first.begin(), first.end(), [&second](std::uintptr_t value)
    {
        return std::find(second.begin(), second.end(), value) != second.end();
    });
}
} // namespace

int main(int argc, char** argv)
{
    const bool strict = argc == 2 && std::string(argv[1]) == "--strict";
#if !defined(__APPLE__)
    std::cerr << "This acceptance target requires native macOS Metal\n";
    return strict ? EXIT_FAILURE : 77;
#else
    using namespace diffractionmaterial;

    auto description = makeAluminiumBinaryGratingPreset();
    description.roughness = {};
    std::string admissionError;
    auto alternateDescription = description;
    alternateDescription.spectrum.wavelengthsNanometres[1] = 410.0f;
    const auto alternate = admit(alternateDescription, admissionError);
    const bool alternateAdmissionRejected = !alternate
        && admissionError
            == "diffraction material production tier requires the canonical wavelength schedule";

    const auto capabilities = queryMetalPhysicalDiffractionCapabilities();
    if (!capabilities.available)
    {
        std::cerr << "BLOCKED: physical Metal diffraction unavailable: "
                  << capabilities.error << '\n';
        return strict ? EXIT_FAILURE : 77;
    }
    bool pass = alternateAdmissionRejected
        && capabilities.backend == "metal"
        && !capabilities.device.empty()
        && capabilities.spectralSamples == kMaximumSpectralSamples
        && capabilities.maximumOrder == kMaximumDiffractionOrder
        && capabilities.canonicalWavelengthLayout
        && capabilities.binaryRectangularProfile
        && capabilities.sinusoidalProfile
        && capabilities.blazedSawtoothProfile
        && capabilities.crossedTwoDimensionalLattice
        && capabilities.incoherentCoating
        && capabilities.rmsHeightAttenuation
        && capabilities.roughnessBroadening;

    std::string error;
    pass = pass && !requireNativeExecution(
        NativeExecutionUse::Preview, NativeBackend::Metal, false, false, error)
        && error == "physical diffraction native Metal backend is unavailable";

    MetalPhysicalDiffractionExecutor executor;
    if (!executor.initialize(error))
    {
        std::cerr << "physical Metal diffraction initialization failed: "
                  << error << '\n';
        return EXIT_FAILURE;
    }
    pass = pass && executor.ready()
        && requireNativeExecution(
            NativeExecutionUse::Preview, NativeBackend::Metal,
            false, executor.ready(), error)
        && error.empty();

    MetalPhysicalDiffractionExecutor isolatedExecutor;
    pass = pass && isolatedExecutor.initialize(error) && isolatedExecutor.ready();

    const auto admitted = admit(description, admissionError);
    if (!admitted)
    {
        std::cerr << "physical diffraction fixture admission failed: "
                  << admissionError << '\n';
        return EXIT_FAILURE;
    }

    SpectralIncidentLight whiteNormal;
    MetalPhysicalDiffractionFrame firstWhite;
    pass = pass && compareCase(
        executor, *admitted, whiteNormal, 1, error, &firstWhite);

    SpectralIncidentLight obliqueSpectrum;
    obliqueSpectrum.direction = {
        0.3f, -0.2f, std::sqrt(1.0f - 0.3f * 0.3f - 0.2f * 0.2f)
    };
    obliqueSpectrum.radiance = {
        0.15f, 0.35f, 0.8f, 1.3f, 1.8f, 1.1f, 0.55f, 0.2f
    };
    MetalPhysicalDiffractionFrame firstOblique;
    pass = pass && compareCase(
        executor, *admitted, obliqueSpectrum, 2, error, &firstOblique);
    pass = pass
        && firstOblique.imageHandles == firstWhite.imageHandles
        && firstOblique.textureViewHandles == firstWhite.textureViewHandles
        && firstOblique.metalTextureHandles == firstWhite.metalTextureHandles;

    auto sinusoidalDescription = makeSinusoidalGratingPreset();
    sinusoidalDescription.roughness = {};
    const auto sinusoidal = admit(sinusoidalDescription, admissionError);
    MetalPhysicalDiffractionFrame sinusoidalWhite;
    pass = pass && sinusoidal && compareCase(
        executor, *sinusoidal, whiteNormal, 3, error, &sinusoidalWhite)
        && sinusoidalWhite.imageHandles == firstWhite.imageHandles
        && sinusoidalWhite.textureViewHandles == firstWhite.textureViewHandles
        && sinusoidalWhite.metalTextureHandles == firstWhite.metalTextureHandles;

    auto blazedDescription = makeBlazedGratingPreset();
    blazedDescription.coating = { CoatingModel::Uncoated, 0.0f, {} };
    blazedDescription.roughness = {};
    const auto blazed = admit(blazedDescription, admissionError);
    MetalPhysicalDiffractionFrame blazedWhite;
    pass = pass && blazed && compareCase(
        executor, *blazed, whiteNormal, 4, error, &blazedWhite)
        && blazedWhite.imageHandles == firstWhite.imageHandles
        && blazedWhite.textureViewHandles == firstWhite.textureViewHandles
        && blazedWhite.metalTextureHandles == firstWhite.metalTextureHandles;

    const auto crossed = admit(makeCrossedTwoDimensionalGratingPreset(), admissionError);
    MetalPhysicalDiffractionFrame crossedWhite;
    pass = pass && crossed && compareCase(
        executor, *crossed, whiteNormal, 5, error, &crossedWhite)
        && crossedWhite.imageHandles == firstWhite.imageHandles
        && crossedWhite.textureViewHandles == firstWhite.textureViewHandles
        && crossedWhite.metalTextureHandles == firstWhite.metalTextureHandles;

    MetalPhysicalDiffractionFrame isolatedWhite;
    pass = pass && compareCase(
        isolatedExecutor, *admitted, whiteNormal, 1, error, &isolatedWhite);
    pass = pass
        && disjoint(firstWhite.imageHandles, isolatedWhite.imageHandles)
        && disjoint(firstWhite.textureViewHandles, isolatedWhite.textureViewHandles)
        && disjoint(firstWhite.metalTextureHandles, isolatedWhite.metalTextureHandles);

    auto coatedDescription = makeBlazedGratingPreset();
    coatedDescription.roughness = {};
    const auto coatedBlazed = admit(coatedDescription, admissionError);
    MetalPhysicalDiffractionFrame rejectedFrame;
    MetalPhysicalDiffractionFrame coatedWhite;
    pass = pass && coatedBlazed && compareCase(
        executor, *coatedBlazed, whiteNormal, 6, error, &coatedWhite)
        && coatedWhite.imageHandles == firstWhite.imageHandles
        && coatedWhite.textureViewHandles == firstWhite.textureViewHandles
        && coatedWhite.metalTextureHandles == firstWhite.metalTextureHandles;

    auto roughDescription = description;
    roughDescription.roughness.rmsHeightNanometres = 8.0f;
    const auto roughHeight = admit(roughDescription, admissionError);
    MetalPhysicalDiffractionFrame roughWhite;
    pass = pass && roughHeight && compareCase(
        executor, *roughHeight, whiteNormal, 7, error, &roughWhite)
        && roughWhite.imageHandles == firstWhite.imageHandles
        && roughWhite.textureViewHandles == firstWhite.textureViewHandles
        && roughWhite.metalTextureHandles == firstWhite.metalTextureHandles;

    roughDescription.roughness.rmsSlope = 0.08f;
    const auto roughSlope = admit(roughDescription, admissionError);
    MetalPhysicalDiffractionFrame roughSlopeWhite;
    pass = pass && roughSlope && compareCase(
        executor, *roughSlope, whiteNormal, 8, error, &roughSlopeWhite)
        && roughSlopeWhite.imageHandles == firstWhite.imageHandles
        && roughSlopeWhite.textureViewHandles == firstWhite.textureViewHandles
        && roughSlopeWhite.metalTextureHandles == firstWhite.metalTextureHandles;

    auto invalidDirection = whiteNormal;
    invalidDirection.direction = { 0.0f, 0.0f, 0.5f };
    pass = pass
        && !executor.execute(*admitted, invalidDirection, rejectedFrame, error)
        && error == "physical diffraction Metal incident direction is invalid";

    auto invalidSpectrum = whiteNormal;
    invalidSpectrum.radiance[3] = -1.0f;
    pass = pass
        && !executor.execute(*admitted, invalidSpectrum, rejectedFrame, error)
        && error == "physical diffraction Metal incident spectrum is invalid";

    MetalPhysicalDiffractionFrame firstAfterRejections;
    pass = pass && compareCase(
        executor, *admitted, whiteNormal, 9, error, &firstAfterRejections)
        && firstAfterRejections.imageHandles == firstWhite.imageHandles
        && firstAfterRejections.textureViewHandles == firstWhite.textureViewHandles
        && firstAfterRejections.metalTextureHandles == firstWhite.metalTextureHandles;

    std::uint64_t spatialGeneration = 10;
    for (const auto& sample : reference::spatialParitySamples())
    {
        const auto spatialMaterial = admit(sample.description, admissionError);
        const bool sampleMatches = spatialMaterial
            && compareCase(executor, *spatialMaterial, whiteNormal,
                           spatialGeneration, error, nullptr, &sample.input);
        if (!sampleMatches)
            std::cerr << "Metal spatial parity failed: " << sample.name << '\n';
        pass = pass && sampleMatches;
        ++spatialGeneration;
    }

    for (const auto& scene : validation::wavelengthValidationScenes())
    {
        const auto material = admit(scene.description, admissionError);
        SpectralIncidentLight sceneLight;
        sceneLight.direction = {
            static_cast<float>(scene.input.incidentDirection.x),
            static_cast<float>(scene.input.incidentDirection.y),
            static_cast<float>(scene.input.incidentDirection.z)
        };
        for (std::size_t index = 0; index < sceneLight.radiance.size(); ++index)
            sceneLight.radiance[index] = static_cast<float>(scene.input.incidentSpectrum[index]);
        MetalPhysicalDiffractionFrame validationFrame;
        const std::array<float, 2> materialUv {
            static_cast<float>(scene.input.materialUv[0]),
            static_cast<float>(scene.input.materialUv[1])
        };
        const bool sceneExecuted = material
            && executor.executeAtMaterialUv(
                *material, sceneLight, materialUv, validationFrame, error);
        const bool sceneMatches = sceneExecuted
            && near(validationFrame.linearSrgb[0], scene.expectedZeroClampedLinearSrgb[0],
                    validation::kFiniteSceneMaximumZeroClampedLinearSrgbChannelError)
            && near(validationFrame.linearSrgb[1], scene.expectedZeroClampedLinearSrgb[1],
                    validation::kFiniteSceneMaximumZeroClampedLinearSrgbChannelError)
            && near(validationFrame.linearSrgb[2], scene.expectedZeroClampedLinearSrgb[2],
                    validation::kFiniteSceneMaximumZeroClampedLinearSrgbChannelError);
        if (!sceneMatches)
            std::cerr << "Metal wavelength validation failed: " << scene.name
                      << " admitted=" << static_cast<bool>(material)
                      << " admissionError=" << admissionError
                      << " executed=" << sceneExecuted
                      << " color=" << validationFrame.linearSrgb[0] << ','
                      << validationFrame.linearSrgb[1] << ','
                      << validationFrame.linearSrgb[2]
                      << " expected=" << scene.expectedZeroClampedLinearSrgb[0] << ','
                      << scene.expectedZeroClampedLinearSrgb[1] << ','
                      << scene.expectedZeroClampedLinearSrgb[2]
                      << " executionError=" << error << '\n';
        pass = pass && sceneMatches;
    }

    executor.shutdown();
    pass = pass && !executor.ready();
    pass = pass
        && !executor.execute(*admitted, whiteNormal, rejectedFrame, error)
        && error == "physical diffraction native Metal backend is unavailable";

    MetalPhysicalDiffractionFrame isolatedAfterShutdown;
    pass = pass && compareCase(
        isolatedExecutor, *admitted, obliqueSpectrum, 2, error,
        &isolatedAfterShutdown)
        && isolatedAfterShutdown.imageHandles == isolatedWhite.imageHandles
        && isolatedAfterShutdown.textureViewHandles == isolatedWhite.textureViewHandles
        && isolatedAfterShutdown.metalTextureHandles == isolatedWhite.metalTextureHandles;
    isolatedExecutor.shutdown();
    pass = pass && !isolatedExecutor.ready();

    if (!pass)
    {
        std::cerr << "physical diffraction native Metal FAIL: " << error << '\n';
        return EXIT_FAILURE;
    }
    std::cout << "physical diffraction native Metal PASS on " << capabilities.device
              << ": finite wavelength scenes and seven spectral pixel fixtures "
                 "matched their references; unsupported inputs rejected; "
                 "executor resources stayed isolated\n";
    return EXIT_SUCCESS;
#endif
}
