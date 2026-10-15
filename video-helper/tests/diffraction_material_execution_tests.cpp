#include "diffraction_material_execution.h"
#include "DiffractionMaterialPresets.h"
#include "gpu_backend/backend.h"
#include "DiffractionProductPlans.h"

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
    check(admittedBackendLighting && arbitgpu::nativeFixtureDiffractionProgramIdentity("x",*admittedBackendLighting)
        == "diffraction-receipt-v1:0000000000000001x00000001080100084180000001010000000000000000003f8000003f80000043be00003f80000043d200003f80000043e600003f80000043fa00003f800000440700003f800000441100003f800000442000003f800000442f00003f800000", "released v1 receipt bytes stay unchanged");
    const auto legacyEnvironment=AdmittedLightingPlan::admit(makeReferenceLighting(),error);
    check(legacyEnvironment && arbitgpu::nativeFixtureDiffractionProgramIdentity("x",*legacyEnvironment)
        == "diffraction-receipt-v1:0000000000000001x00000002bf80000041a00000080100084180000003010000000000000000003f8000003f26666643be00003f26666643d200003f26666643e600003f26666643fa00003f266666440700003f266666441100003f266666442000003f266666442f00003f266666020000000000000000003f8000003e75c28f43be00003df5c28f43d200003e0f5c2943e600003e3851ec43fa00003e6147ae440700003e75c28f441100003e6147ae442000003e3851ec442f00003e0f5c29030100000000000000003f8000003d8f5c2943be00003d23d70a43d200003d4ccccd43e600003d75c28f43fa00003d8f5c29440700003d8f5c29441100003d75c28f442000003d4ccccd442f00003d23d70a", "released v2 analytic environment receipt bytes stay unchanged");
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
    auto authoredLighting = backendLighting;
    authoredLighting.version = 3;
    authoredLighting.sceneReceiverTransport = SceneReceiverTransport::AuthoredSpectralLambertian;
    authoredLighting.traceMaximumDistance = 100.0f;
    const auto admittedAuthored = AdmittedLightingPlan::admit(authoredLighting, error);
    check(admittedAuthored.has_value(), "authored scene lighting is a valid value-owned contract");
    if (admittedAuthored)
    {
        auto directAuthored = backendProgram;
        directAuthored.diffractionLightingAdmission = std::make_shared<const AdmittedLightingPlan>(*admittedAuthored);
        directAuthored.diffractionMaximumBounceDepth = kMaximumIndirectBounces;
        directAuthored.programIdentity = arbitgpu::nativeFixtureDiffractionProgramIdentity(
            directAuthored.bindingDigest, *admittedAuthored);
        check(!arbitgpu::validNativeFixtureDiffractionProgram(directAuthored,
                  arbitgpu::NativeFixtureMaterialBackend::OpenGl, backendObject),
              "direct OpenGL callers cannot render authored scene transport before its native consumer exists");
        directAuthored.authoredScene={3};directAuthored.authoredMaterial={19};
        directAuthored.sceneRevision=directAuthored.structuralRevision=directAuthored.evaluationRevision=directAuthored.materialRevision=1;
        directAuthored.programIdentity=arbitgpu::nativeFixtureDiffractionProgramIdentity(
            directAuthored.bindingDigest,*admittedAuthored,0,1,{},false,&directAuthored);
        check(arbitgpu::validNativeFixtureDiffractionProgram(directAuthored,
                  arbitgpu::NativeFixtureMaterialBackend::OpenGl,backendObject,true),
              "exact v3 receipt admits shared query preparation only under explicit consumer selection");
        const auto original=directAuthored.programIdentity;
        const auto changedReceipt=[&](auto mutate)
        {
            auto lighting=authoredLighting;mutate(lighting);
            const auto admitted=AdmittedLightingPlan::admit(lighting,error);
            return admitted && arbitgpu::nativeFixtureDiffractionProgramIdentity(
                directAuthored.bindingDigest,*admitted,0,1,{},false,&directAuthored)!=original;
        };
        check(changedReceipt([](auto& value){value.paths[0].incident.radiance[2]=std::nextafter(1.0f,2.0f);}),
              "v3 receipt changes on one raw authored radiance bit");
        check(changedReceipt([](auto& value){value.paths[0].incident.direction[0]=-0.0f;}),
              "v3 receipt preserves signed zero direction bits");
        check(changedReceipt([](auto& value){value.traceMaximumDistance=std::nextafter(100.0f,101.0f);}),
              "v3 receipt changes on one finite trace-range bit");
        check(changedReceipt([](auto& value){value.bouncePlaneHeight=-0.0f;}),
              "v3 receipt retains canonical zero plane raw bits");
        check(changedReceipt([](auto& value){value.paths[7].incident.radiance[1]=std::nextafter(1.0f,2.0f);})==false,
              "v3 admission refuses noncanonical trailing spectra");
        check(changedReceipt([](auto& value){value.paths[7].incident.direction[0]=-0.0f;}),
              "v3 receipt retains all canonical trailing direction bits");
        for(unsigned field=0;field<7;++field)
        {
            auto changed=directAuthored;
            if(field==0)++changed.authoredScene.value;
            if(field==1)++changed.object.value;
            if(field==2)++changed.authoredMaterial.value;
            if(field==3)++changed.sceneRevision;
            if(field==4)++changed.structuralRevision;
            if(field==5)++changed.evaluationRevision;
            if(field==6)++changed.materialRevision;
            check(arbitgpu::nativeFixtureDiffractionProgramIdentity(changed.bindingDigest,*admittedAuthored,0,1,{},false,&changed)!=original,
                  "v3 receipt retains each exact ownership identity and revision");
        }
        const auto sourceOwner=std::make_shared<const arbitgpu::NativeFixtureSurfaceMaterialProgram>(directAuthored);
        const auto prepared=arbitgpu::snapshotNativeSurfaceProgram(sourceOwner);
        auto evaluated=std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(directAuthored);
        evaluated->diffractionPaths[0].material.roughness.y=.02f;
        evaluated->bindingDigest="evaluated-native-v3-rms-slope";
        evaluated->programIdentity=arbitgpu::nativeFixtureDiffractionProgramIdentity(
            evaluated->bindingDigest,*admittedAuthored,0,1,{},false,evaluated.get());
        arbitgpu::NativeImportedSceneRuntimeInputs runtime;
        runtime.diffractionParameters={{"rmsSlope",.02f}};
        runtime.diffractionSourceProgram=sourceOwner;runtime.diffractionEvaluatedProgram=evaluated;
        std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram> resolved;
        check(!arbitgpu::resolveDiffractionRuntimeProgram(prepared,sourceOwner,runtime,resolved,error),
              "default runtime resolution keeps evaluated v3 rendering guarded");
        check(arbitgpu::resolveDiffractionRuntimeProgram(prepared,sourceOwner,runtime,resolved,error,true)
              && resolved==evaluated && resolved->diffractionPaths[0].material.roughness.y==.02f,
              "an explicit native table consumer resolves valid evaluated v3 scalar edits");
        runtime.diffractionSourceProgram=std::make_shared<const arbitgpu::NativeFixtureSurfaceMaterialProgram>(*sourceOwner);
        check(!arbitgpu::resolveDiffractionRuntimeProgram(prepared,sourceOwner,runtime,resolved,error,true),
              "v3 consumer opt-in cannot substitute a copied authored owner");
        runtime.diffractionSourceProgram=sourceOwner;
        auto changedLayout=std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*evaluated);
        changedLayout->diffractionPaths[0].material.control.x=2;
        runtime.diffractionEvaluatedProgram=changedLayout;
        check(!arbitgpu::resolveDiffractionRuntimeProgram(prepared,sourceOwner,runtime,resolved,error,true),
              "v3 consumer opt-in preserves the authored diffraction order schedule");
        runtime.diffractionEvaluatedProgram=evaluated;
        const auto legacyOwner=std::make_shared<const arbitgpu::NativeFixtureSurfaceMaterialProgram>(backendProgram);
        runtime.diffractionSourceProgram=legacyOwner;runtime.diffractionEvaluatedProgram=legacyOwner;
        check(arbitgpu::resolveDiffractionRuntimeProgram(arbitgpu::snapshotNativeSurfaceProgram(legacyOwner),
                  legacyOwner,runtime,resolved,error) && resolved==legacyOwner,
              "released v1 runtime resolution retains its default consumer behavior");
        directAuthored.backend = arbitgpu::NativeFixtureMaterialBackend::Metal;
        check(!arbitgpu::validNativeFixtureDiffractionProgram(directAuthored,
                  arbitgpu::NativeFixtureMaterialBackend::Metal, backendObject),
              "direct Metal callers cannot render authored scene transport before its native consumer exists");
    }
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
