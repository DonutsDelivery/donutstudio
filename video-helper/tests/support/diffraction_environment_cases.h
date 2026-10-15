#pragma once

#include "../../src/fixture_scene_renderer.h"
#include "../../../shared/VisualImportedSceneRenderOperationContract.h"
#include "diffraction_frame_cases.h"

#include <iostream>
#include <vector>

namespace diffractionenvironmenttests
{
template <typename Backend, typename ReadPixels>
bool run(Backend& backend,
         const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
         visualimportedscenerender::Request request,
         videohelper::materialprogram::BackendTarget target,
         ReadPixels readPixels)
{
    bool ok = videohelper::tests::diffractionFrameCases(backend, scene, request, target, readPixels);
    std::string error;
    const auto starterRequest = request;
    const auto cardFoil = request.diffractionMaterial->spatialFoil;
    auto& binding = *request.diffractionMaterial;
    binding.spatialFoil.reset();
    binding.material = diffractionmaterial::makeAluminiumBinaryGratingPreset();
    binding.material.spectrum.lastOrder = 1;
    const auto material = diffractionmaterial::admit(binding.material, error);
    if (!material) return false;
    binding.structuralDigest = material->structuralDigest();
    binding.version = diffractionmaterialbinding::kEnvironmentWireVersion;
    binding.lighting = diffractionmaterial::makeReferenceLighting();
    binding.lighting.version = 2;
    binding.lighting.bouncePlaneHeight = -0.06f;
    binding.lighting.bounceMaximumDistance = 100.0f;
    binding.lighting.paths[0].incident.radiance.fill(0.0f);
    binding.lighting.paths[1].incident.direction = { 0.0f, 0.8f, 0.6f };
    binding.lighting.paths[2].incident.direction = binding.lighting.paths[1].incident.direction;
    binding.lighting.paths[1].incident.radiance.fill(0.8f);
    binding.lighting.paths[2].incident.radiance.fill(0.4f);
    // The card's period00 Field belongs to its spatial foil. These plain-BSDF
    // variants must detach that input along with the foil it addresses.
    if (request.materialField && visualimportedscenerender::valid(request))
    {
        std::cerr << "FAIL: a plain diffraction material accepted an inactive foil Field target\n";
        ok = false;
    }
    request.materialField.reset();
    const auto render = [&](const visualimportedscenerender::Request& source, bool exporting)
    {
        visualimportedscenerender::Request decoded;
        if (!visualimportedscenerender::decodeCanonical(
                visualimportedscenerender::encode(source), decoded, error))
        {
            ok = false;
            std::cerr << "Environment wire rejected: " << error << '\n';
            return std::vector<std::uint8_t> {};
        }
        const auto admitted = videorender::fixture3d::admitDiffractionMaterialBinding(
            scene, *decoded.diffractionMaterial, target, error);
        videorender::fixture3d::FixtureSceneRenderer renderer(backend);
        videorender::fixture3d::RenderedFrame frame;
        arbitgpu::NativeImportedSceneRuntimeInputs runtime;
        runtime.cameraOverride = decoded.camera;
        runtime.lightOverride = decoded.light;
        const auto rendered = admitted && (exporting
            ? renderer.renderExport(scene, admitted, runtime, { 96, 96 },
                videorender::fixture3d::kNativeGpuCapability, frame, error)
            : renderer.renderPreview(scene, admitted, runtime, { 96, 96 },
                videorender::fixture3d::kNativeGpuCapability, frame, error));
        if (!rendered)
        {
            ok = false;
            std::cerr << "Environment native draw rejected: " << error << '\n';
            return std::vector<std::uint8_t> {};
        }
        return readPixels(frame.nativeFrame);
    };
    if (!starterRequest.camera || !starterRequest.light)
    {
        std::cerr << "FAIL: card fixture lacks the graph-authored camera or key light\n";
        ok = false;
    }
    else
    {
        const auto starter = render(starterRequest, false);
        auto movedCamera = starterRequest;
        movedCamera.camera->transform.translation.x += 0.02f;
        auto darkKey = starterRequest;
        darkKey.light->intensity = 0.0f;
        auto turnedKey = starterRequest;
        turnedKey.light->transform.rotation = { 0.0f, 0.70710677f, 0.0f, 0.70710677f };
        const auto cameraPixels = render(movedCamera, false);
        const auto keyPixels = render(darkKey, false);
        const auto turnedPixels = render(turnedKey, false);
        const auto admittedStarter = videorender::fixture3d::admitDiffractionMaterialBinding(
            scene, *starterRequest.diffractionMaterial, target, error);
        const bool starterWorks = !starter.empty() && starter == render(starterRequest, true)
            && starter != cameraPixels && cameraPixels == render(movedCamera, true)
            && starter != keyPixels && keyPixels == render(darkKey, true)
            && starter != turnedPixels && turnedPixels == render(turnedKey, true)
            && admittedStarter
            && arbitgpu::nativeFixtureDiffractionWorkWithinBudget(
                *admittedStarter->nativeProgram(), 1920, 1080)
            && !arbitgpu::nativeFixtureDiffractionWorkWithinBudget(
                *admittedStarter->nativeProgram(), 1921, 1080);
        if (!starterWorks)
            std::cerr << "FAIL: card camera, key light, 1080p budget or preview/export parity\n";
        ok = starterWorks && ok;
    }
    const auto environment = render(request, false);
    const auto exported = render(request, true);
    auto darkRequest = request;
    darkRequest.diffractionMaterial->lighting.paths[1].incident.radiance.fill(0.0f);
    darkRequest.diffractionMaterial->lighting.paths[2].incident.radiance.fill(0.0f);
    const auto dark = render(darkRequest, false);
    auto movedEnvironment = request;
    movedEnvironment.diffractionMaterial->lighting.paths[1].incident.direction = { 0.0f, -0.8f, -0.6f };
    movedEnvironment.diffractionMaterial->lighting.paths[2].incident.direction = { 0.0f, -0.8f, -0.6f };
    const auto moved = render(movedEnvironment, false);
    auto withoutBounce = request;
    withoutBounce.diffractionMaterial->lighting.paths[2].incident.radiance.fill(0.0f);
    const auto noBounce = render(withoutBounce, false);
    auto raisedPlane = request;
    raisedPlane.diffractionMaterial->lighting.bouncePlaneHeight = 1.0f;
    const auto missedPlane = render(raisedPlane, false);
    raisedPlane.diffractionMaterial->lighting.paths[2].incident.radiance.fill(0.0f);
    const auto missedWithoutBounce = render(raisedPlane, false);
    const bool visible = environment.size() == 96u * 96u * 4u
        && environment == exported && environment != dark && environment != moved
        && environment != noBounce && missedPlane == missedWithoutBounce;
    if (!visible)
        std::cerr << "FAIL: native spectral environment, diffuse hit/miss, direction, and preview/export pixels\n";
    auto linear = diffractionmaterial::makeAluminiumBinaryGratingPreset();
    linear.grooveField.mode = diffractionmaterial::GrooveFieldMode::Linear;
    linear.grooveField.originUv = { 0.5f, 0.5f };
    linear.grooveField.axisUv = { 1.0f, 0.0f };
    linear.grooveField.orientationDegreesPerUnit = 90.0f;
    linear.grooveField.grooveSpacingDeltaNanometresPerUnit = 300.0f;
    auto radial = linear;
    radial.grooveField.mode = diffractionmaterial::GrooveFieldMode::Radial;
    radial.grooveField.axisUv = {};
    radial.grooveField.grooveSpacingDeltaNanometresPerUnit = 0.0f;
    radial.grooveField.orientationDegreesPerUnit = 180.0f;
    auto substrate = diffractionmaterial::makeAluminiumBinaryGratingPreset();
    substrate.substrate = { 1.5f, 0.0f };
    auto coating = diffractionmaterial::makeAluminiumBinaryGratingPreset();
    coating.coating = { diffractionmaterial::CoatingModel::IncoherentDielectric,
        400.0f, { 1.5f, 0.1f } };
    auto roughness = diffractionmaterial::makeAluminiumBinaryGratingPreset();
    roughness.roughness = { 20.0f, 0.2f };
    for (auto variant : { diffractionmaterial::makeSinusoidalGratingPreset(),
                          diffractionmaterial::makeBlazedGratingPreset(),
                          diffractionmaterial::makeRealtimeCrossedTwoDimensionalGratingPreset(),
                          linear, radial, substrate, coating, roughness })
    {
        variant.spectrum.lastOrder = 1;
        const auto admittedVariant = diffractionmaterial::admit(variant, error);
        if (!admittedVariant) { ok = false; continue; }
        auto variantRequest = request;
        variantRequest.diffractionMaterial->material = variant;
        variantRequest.diffractionMaterial->structuralDigest = admittedVariant->structuralDigest();
        const auto pixels = render(variantRequest, false);
        if (pixels.empty() || pixels == environment || pixels != render(variantRequest, true))
        {
            ok = false;
            std::cerr << "FAIL: physical profile, lattice, coating, or groove field did not reach environment pixels\n";
        }
    }
    if (cardFoil)
    {
        const auto foilRequest = [&](int variant)
        {
            auto result = request;
            auto& materialBinding = *result.diffractionMaterial;
            materialBinding.spatialFoil = *cardFoil;
            auto& physical = materialBinding.spatialFoil->physicalBsdf;
            physical.spectrum.lastOrder = 1;
            if (variant == 1) physical.microstructure.profile = diffractionmaterial::GrooveProfile::Sinusoidal;
            if (variant == 2) physical.substrate = substrate.substrate;
            if (variant == 3) physical.coating = coating.coating;
            if (variant == 4) physical.roughness = roughness.roughness;
            const auto admitted = diffractivefoil::admit(*materialBinding.spatialFoil, error);
            materialBinding.material = physical;
            materialBinding.structuralDigest = admitted ? admitted->structuralDigest() : std::string {};
            return result;
        };
        const auto cornerPixels = render(foilRequest(0), false);
        for (int variant = 1; variant <= 4; ++variant)
        {
            const auto changed = foilRequest(variant);
            const auto pixels = render(changed, false);
            if (pixels.empty() || pixels == cornerPixels || pixels != render(changed, true))
            {
                ok = false;
                std::cerr << "FAIL: authored foil optical property " << variant
                          << " did not change native preview/export pixels\n";
            }
        }
    }
    auto invalid = request;
    invalid.diffractionMaterial->lighting.bounceMaximumDistance = 0.0f;
    const auto rejected = videorender::fixture3d::admitDiffractionMaterialBinding(
        scene, *invalid.diffractionMaterial, target, error);
    if (rejected) std::cerr << "FAIL: unbounded diffraction bounce was admitted\n";
    return ok && visible && !rejected;
}
} // namespace diffractionenvironmenttests
