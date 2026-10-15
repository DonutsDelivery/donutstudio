#include "../src/diffraction_material_admission.h"
#include "../../shared/DiffractionMaterialPresets.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>

using namespace diffractionmaterial;

static_assert(std::is_trivially_copyable_v<GratingGeometry>);
static_assert(std::is_trivially_copyable_v<GratingMicrostructure>);
static_assert(std::is_trivially_copyable_v<OpticalConstants>);
static_assert(std::is_trivially_copyable_v<SurfaceCoating>);
static_assert(std::is_trivially_copyable_v<SurfaceRoughness>);
static_assert(std::is_trivially_copyable_v<SpectralSampling>);
static_assert(std::is_trivially_copyable_v<Description>);
static_assert(!kAuthorizesNativeGpuExecution);
static_assert(!kAllowsCpuProductionRenderingFallback);

namespace
{
int failures = 0;
int checks = 0;

void check(bool value, const char* message)
{
    ++checks;
    if (!value)
    {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

void reject(const Description& source,
            std::string_view text,
            const char* message,
            const AdmissionLimits& limits = {})
{
    std::string error;
    const auto admitted = admit(source, error, limits);
    check(!admitted && error.find(text) != std::string::npos, message);
}
} // namespace

int main()
{
    check(token(GratingLattice::OneDimensional) == "one-dimensional"
              && token(GratingLattice::CrossedTwoDimensional)
                   == "crossed-two-dimensional"
              && token(WavelengthQualityTier::BoundedProductionV1)
                   == "bounded-production-v1"
              && token(GrooveProfile::BinaryRectangular) == "binary-rectangular"
              && token(GrooveProfile::Sinusoidal) == "sinusoidal"
              && token(GrooveProfile::BlazedSawtooth) == "blazed-sawtooth"
              && token(CoatingModel::Uncoated) == "uncoated"
              && token(CoatingModel::IncoherentDielectric)
                   == "incoherent-dielectric"
              && token(SpectralIntegrationModel::Cie1931Xyz) == "cie-1931-xyz",
          "the physical diffraction vocabulary is stable");
    check(token(static_cast<GratingLattice>(255)).empty()
              && token(static_cast<GrooveProfile>(255)).empty()
              && token(static_cast<CoatingModel>(255)).empty()
              && token(static_cast<SpectralIntegrationModel>(255)).empty(),
          "unknown contract values have no executable token");

    std::string error;
    const auto binary = admit(makeAluminiumBinaryGratingPreset(), error);
    const auto sinusoidal = admit(makeSinusoidalGratingPreset(), error);
    const auto blazed = admit(makeBlazedGratingPreset(), error);
    const auto crossed = admit(makeCrossedTwoDimensionalGratingPreset(), error);
    check(binary && sinusoidal && blazed && crossed && error.empty(),
          "all bounded physical grating presets are admitted");
    if (!binary || !sinusoidal || !blazed || !crossed)
        return EXIT_FAILURE;

    check(binary->structuralDigest().size() == 64
              && sinusoidal->structuralDigest().size() == 64
              && blazed->structuralDigest().size() == 64
              && crossed->structuralDigest().size() == 64
              && binary->structuralDigest() != sinusoidal->structuralDigest()
              && binary->structuralDigest() != blazed->structuralDigest()
              && binary->structuralDigest() != crossed->structuralDigest(),
          "profile, coating, roughness, and lattice data participate in structural identity");

    auto source = makeAluminiumBinaryGratingPreset();
    const auto digest = binary->structuralDigest();
    source.microstructure.grooveDepthNanometres = 140.0f;
    source.substrate.extinctionCoefficient = 2.0f;
    check(binary->description().microstructure.grooveDepthNanometres == 125.0f
              && binary->description().substrate.extinctionCoefficient == 3.10f
              && binary->structuralDigest() == digest,
          "admission owns immutable physical material data");
    const auto changed = admit(source, error);
    check(changed && changed->structuralDigest() != digest,
          "physical material changes alter structural identity");

    auto negativeZero = makeAluminiumBinaryGratingPreset();
    negativeZero.microstructure.blazeAngleDegrees = -0.0f;
    negativeZero.coating.thicknessNanometres = -0.0f;
    negativeZero.roughness.rmsHeightNanometres = -0.0f;
    const auto canonical = admit(negativeZero, error);
    check(canonical && canonical->structuralDigest() == digest
              && !std::signbit(canonical->description().coating.thicknessNanometres)
              && !std::signbit(canonical->description().roughness.rmsHeightNanometres),
          "negative zero has one canonical physical identity");

    AdmissionLimits lowerSamples;
    lowerSamples.spectralSamples = kMaximumSpectralSamples - 1;
    reject(makeAluminiumBinaryGratingPreset(), "bounded production tier",
           "callers cannot request an unsupported seven-sample production tier",
           lowerSamples);
    for (std::size_t sampleCount = 4; sampleCount < kMaximumSpectralSamples;
         ++sampleCount)
    {
        auto unsupported = makeAluminiumBinaryGratingPreset();
        unsupported.spectrum.wavelengthCount = static_cast<std::uint8_t>(sampleCount);
        for (std::size_t index = sampleCount;
             index < unsupported.spectrum.wavelengthsNanometres.size(); ++index)
        {
            unsupported.spectrum.wavelengthsNanometres[index] = 0.0f;
        }
        reject(unsupported, "exactly eight spectral samples",
               "unsupported four-through-seven-sample requests fail at shared admission");
    }
    AdmissionLimits lowerOrders;
    lowerOrders.diffractionOrder = 3;
    reject(makeAluminiumBinaryGratingPreset(), "order range",
           "callers may lower the diffraction order bound", lowerOrders);
    AdmissionLimits invalidLimits;
    invalidLimits.diffractionOrder = kMaximumDiffractionOrder + 1;
    reject(makeAluminiumBinaryGratingPreset(), "bounded production tier",
           "callers cannot raise hard admission bounds", invalidLimits);

    auto malformed = makeAluminiumBinaryGratingPreset();
    ++malformed.version;
    reject(malformed, "wire version", "unknown wire versions fail closed");
    malformed = makeAluminiumBinaryGratingPreset();
    malformed.geometry.directionUv = { 2.0f, 0.0f };
    reject(malformed, "unit length", "non-unit grating directions fail closed");
    malformed = makeAluminiumBinaryGratingPreset();
    malformed.geometry.grooveSpacingNanometres
        = std::numeric_limits<float>::quiet_NaN();
    reject(malformed, "groove period", "nonfinite groove periods fail closed");
    malformed = makeAluminiumBinaryGratingPreset();
    malformed.geometry.secondaryDirectionUv = { 0.0f, 1.0f };
    malformed.geometry.secondaryGrooveSpacingNanometres = 1400.0f;
    reject(malformed, "carries secondary lattice data",
           "one-dimensional gratings reject hidden secondary lattice data");
    malformed = makeCrossedTwoDimensionalGratingPreset();
    malformed.geometry.secondaryDirectionUv = { 1.0f, 0.0f };
    reject(malformed, "orthonormal secondary lattice",
           "crossed gratings reject non-orthogonal lattice directions");
    malformed = makeCrossedTwoDimensionalGratingPreset();
    malformed.geometry.secondaryGrooveSpacingNanometres = 0.0f;
    reject(malformed, "orthonormal secondary lattice",
           "crossed gratings reject invalid secondary periods");
    malformed = makeAluminiumBinaryGratingPreset();
    malformed.microstructure.profile = static_cast<GrooveProfile>(255);
    reject(malformed, "groove profile", "unknown groove profiles fail closed");
    malformed = makeSinusoidalGratingPreset();
    malformed.microstructure.dutyCycle = 0.4f;
    reject(malformed, "canonical duty cycle",
           "sinusoidal profiles reject invented duty-cycle semantics");
    malformed = makeBlazedGratingPreset();
    malformed.microstructure.blazeAngleDegrees = 30.0f;
    reject(malformed, "blaze angle",
           "blazed profiles require angle, depth, duty, and period agreement");
    malformed = makeAluminiumBinaryGratingPreset();
    malformed.microstructure.grooveDepthNanometres
        = std::numeric_limits<float>::infinity();
    reject(malformed, "physical groove microstructure",
           "nonfinite groove microstructure fails closed");
    malformed = makeAluminiumBinaryGratingPreset();
    malformed.substrate.extinctionCoefficient
        = std::numeric_limits<float>::quiet_NaN();
    reject(malformed, "substrate optical constants",
           "nonfinite substrate optical constants fail closed");
    malformed = makeAluminiumBinaryGratingPreset();
    malformed.coating.model = CoatingModel::Uncoated;
    malformed.coating.thicknessNanometres = 1.0f;
    reject(malformed, "uncoated surface carries coating data",
           "uncoated materials reject hidden coating data");
    malformed = makeBlazedGratingPreset();
    malformed.coating.opticalConstants.refractiveIndex
        = std::numeric_limits<float>::infinity();
    reject(malformed, "physical coating",
           "nonfinite coating data fails closed");
    malformed = makeAluminiumBinaryGratingPreset();
    malformed.roughness.rmsSlope = std::numeric_limits<float>::quiet_NaN();
    reject(malformed, "physical roughness", "nonfinite roughness fails closed");
    malformed = makeAluminiumBinaryGratingPreset();
    malformed.spectrum.integrationModel
        = static_cast<SpectralIntegrationModel>(255);
    reject(malformed, "spectral integration",
           "unknown spectral integration fails closed");
    malformed = makeAluminiumBinaryGratingPreset();
    malformed.spectrum.wavelengthsNanometres[2]
        = malformed.spectrum.wavelengthsNanometres[1];
    reject(malformed, "canonical wavelength schedule",
           "noncanonical wavelength schedules fail shared admission");
    malformed = makeAluminiumBinaryGratingPreset();
    malformed.spectrum.wavelengthCount = 7;
    reject(malformed, "exactly eight spectral samples",
           "malformed declared spectral counts fail closed");
    malformed = makeAluminiumBinaryGratingPreset();
    malformed.spectrum.firstOrder = 2;
    reject(malformed, "order range",
           "production order schedules start at the first nonzero order");
    malformed = makeAluminiumBinaryGratingPreset();
    malformed.spectrum.lastOrder = kMaximumDiffractionOrder + 1;
    reject(malformed, "order range", "orders above the hard bound fail closed");

    auto spatial = makeCrossedTwoDimensionalGratingPreset();
    spatial.grooveField.mode = GrooveFieldMode::Linear;
    spatial.grooveField.originUv = { 0.5f, 0.5f };
    spatial.grooveField.axisUv = { 1.0f, 0.0f };
    spatial.grooveField.grooveSpacingDeltaNanometresPerUnit = 200.0f;
    spatial.grooveField.secondarySpacingDeltaNanometresPerUnit = -100.0f;
    spatial.grooveField.orientationDegreesPerUnit = 30.0f;
    const auto spatialAdmitted = admit(spatial, error);
    check(spatialAdmitted && error.empty()
              && spatialAdmitted->structuralDigest() != crossed->structuralDigest(),
          "bounded linear groove fields are admitted and own their digest identity");
    auto radial = spatial;
    radial.grooveField.mode = GrooveFieldMode::Radial;
    radial.grooveField.axisUv = {};
    radial.grooveField.originUv = { 0.5f, 0.5f };
    const auto radialAdmitted = admit(radial, error);
    check(radialAdmitted && error.empty()
              && radialAdmitted->structuralDigest()
                  != spatialAdmitted->structuralDigest(),
          "bounded radial groove fields are admitted with distinct identity");
    malformed = spatial;
    malformed.grooveField.axisUv = { 2.0f, 0.0f };
    reject(malformed, "unit length", "non-unit linear field axes fail closed");
    malformed = spatial;
    malformed.grooveField.grooveSpacingDeltaNanometresPerUnit = -3000.0f;
    reject(malformed, "out-of-bounds local period",
           "fields whose local period crosses the physical bound fail closed");
    malformed = spatial;
    malformed.grooveField.mode = static_cast<GrooveFieldMode>(255);
    reject(malformed, "mode is unsupported", "unknown field modes fail closed");
    malformed = makeAluminiumBinaryGratingPreset();
    malformed.grooveField.orientationDegreesPerUnit = 1.0f;
    reject(malformed, "carries analytic field data",
           "constant fields reject hidden noncanonical variation data");
    auto spatialBlazed = makeBlazedGratingPreset();
    spatialBlazed.grooveField.mode = GrooveFieldMode::Linear;
    spatialBlazed.grooveField.originUv = { 0.5f, 0.5f };
    spatialBlazed.grooveField.axisUv = { 1.0f, 0.0f };
    spatialBlazed.grooveField.orientationDegreesPerUnit = 20.0f;
    check(admit(spatialBlazed, error).has_value() && error.empty(),
          "blazed profiles admit spatial orientation with fixed physical geometry");
    spatialBlazed.grooveField.grooveSpacingDeltaNanometresPerUnit = 1.0f;
    reject(spatialBlazed, "spatially constant period",
           "blazed profiles reject period variation that contradicts fixed angle and depth");
    auto crossedBlazed = makeBlazedGratingPreset();
    crossedBlazed.geometry.lattice = GratingLattice::CrossedTwoDimensional;
    crossedBlazed.geometry.secondaryDirectionUv = { 0.0f, 1.0f };
    crossedBlazed.geometry.secondaryGrooveSpacingNanometres
        = crossedBlazed.geometry.grooveSpacingNanometres + 1.0f;
    reject(crossedBlazed, "equal primary and secondary periods",
           "crossed blazed profiles reject unequal periods under the shared depth and blaze model");

    check(!requireNativeExecution(
              NativeExecutionUse::Invalid, NativeBackend::OpenGl,
              true, false, error)
              && error == "physical diffraction execution use is invalid",
          "only preview and export may request physical diffraction execution");
    check(!requireNativeExecution(
              NativeExecutionUse::Preview, NativeBackend::OpenGl,
              false, false, error)
              && error == "physical diffraction native OpenGL backend is unavailable",
          "preview fails closed without its native OpenGL executor");
    check(requireNativeExecution(
              NativeExecutionUse::Preview, NativeBackend::OpenGl,
              true, false, error) && error.empty(),
          "preview accepts an initialized native OpenGL executor");
    check(requireNativeExecution(
              NativeExecutionUse::Export, NativeBackend::OpenGl,
              true, false, error) && error.empty(),
          "export uses the same strict native OpenGL gate");
    check(!requireNativeExecution(
              NativeExecutionUse::Export, NativeBackend::Metal,
              true, false, error)
              && error == "physical diffraction native Metal backend is unavailable",
          "export cannot substitute OpenGL readiness for Metal readiness");
    check(requireNativeExecution(
              NativeExecutionUse::Preview, NativeBackend::Metal,
              false, true, error) && error.empty()
              && requireNativeExecution(
                  NativeExecutionUse::Export, NativeBackend::Metal,
                  false, true, error) && error.empty(),
          "preview and export accept an initialized native Metal executor");
    check(!requireNativeExecution(
              NativeExecutionUse::Preview, NativeBackend::Invalid,
              true, true, error)
              && error == "physical diffraction native backend is invalid",
          "invalid native backends fail closed");

    std::cout << "diffraction material admission: "
              << checks - failures << '/' << checks << " checks passed\n";
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
