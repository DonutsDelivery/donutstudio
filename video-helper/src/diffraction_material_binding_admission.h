#pragma once

#include "../../shared/DiffractionMaterialBindingContract.h"
#include "diffraction_material_admission.h"
#include "../../shared/DiffractionProductPlans.h"
#include "diffractive_foil_admission.h"

#include <optional>
#include <string>
#include <utility>

namespace diffractionmaterialbinding
{
class AdmittedImportedSceneDiffractionMaterial final
{
public:
    AdmittedImportedSceneDiffractionMaterial(
        const AdmittedImportedSceneDiffractionMaterial&) = default;
    AdmittedImportedSceneDiffractionMaterial(
        AdmittedImportedSceneDiffractionMaterial&&) = default;
    AdmittedImportedSceneDiffractionMaterial& operator=(
        const AdmittedImportedSceneDiffractionMaterial&) = delete;
    AdmittedImportedSceneDiffractionMaterial& operator=(
        AdmittedImportedSceneDiffractionMaterial&&) = delete;

    HarmonicMIDI::grid::Scene3DId scene() const noexcept { return scene_; }
    HarmonicMIDI::grid::SceneObjectId object() const noexcept { return object_; }
    std::uint64_t sceneRevision() const noexcept { return sceneRevision_; }
    std::uint64_t structuralRevision() const noexcept { return structuralRevision_; }
    std::uint64_t evaluationRevision() const noexcept { return evaluationRevision_; }
    std::uint64_t materialRevision() const noexcept { return materialRevision_; }
    const diffractionmaterial::AdmittedDiffractionMaterialIR& material() const noexcept
    {
        return material_;
    }
    const diffractionmaterial::AdmittedLightingPlan& lighting() const noexcept
    {
        return lighting_;
    }
    const std::optional<diffractivefoil::AdmittedDiffractiveFoilIR>& spatialFoil() const noexcept
    {
        return spatialFoil_;
    }

    static constexpr bool authorizesNativeGpuExecution = false;
    static constexpr bool allowsCpuProductionRenderingFallback = false;

private:
    friend std::optional<AdmittedImportedSceneDiffractionMaterial> admit(
        const ImportedSceneDiffractionMaterialRequest&, std::string&);

    AdmittedImportedSceneDiffractionMaterial(
        const ImportedSceneDiffractionMaterialRequest& request,
        diffractionmaterial::AdmittedDiffractionMaterialIR material,
        diffractionmaterial::AdmittedLightingPlan lighting,
        std::optional<diffractivefoil::AdmittedDiffractiveFoilIR> spatialFoil)
        : scene_(request.scene),
          object_(request.object),
          sceneRevision_(request.sceneRevision),
          structuralRevision_(request.structuralRevision),
          evaluationRevision_(request.evaluationRevision),
          materialRevision_(request.materialRevision),
          material_(std::move(material)), lighting_(std::move(lighting)),
          spatialFoil_(std::move(spatialFoil))
    {
    }

    const HarmonicMIDI::grid::Scene3DId scene_;
    const HarmonicMIDI::grid::SceneObjectId object_;
    const std::uint64_t sceneRevision_;
    const std::uint64_t structuralRevision_;
    const std::uint64_t evaluationRevision_;
    const std::uint64_t materialRevision_;
    const diffractionmaterial::AdmittedDiffractionMaterialIR material_;
    const diffractionmaterial::AdmittedLightingPlan lighting_;
    const std::optional<diffractivefoil::AdmittedDiffractiveFoilIR> spatialFoil_;
};

inline bool migrateLegacyV6(
    ImportedSceneDiffractionMaterialRequest& request)
{
    std::string error;
    const auto admitted = diffractionmaterial::admit(request.material, error);
    if (!admitted) return false;
    request.lighting = diffractionmaterial::makeReferenceLighting();
    request.structuralDigest = admitted->structuralDigest();
    return true;
}

inline std::optional<AdmittedImportedSceneDiffractionMaterial> admit(
    const ImportedSceneDiffractionMaterialRequest& request,
    std::string& error)
{
    if (request.version != kWireVersion || ! request.scene.isValid()
        || ! request.object.isValid() || request.sceneRevision == 0
        || request.structuralRevision == 0 || request.evaluationRevision == 0
        || request.materialRevision == 0
        || request.materialRevision != request.structuralRevision)
    {
        error = "imported diffraction material binding identity is invalid";
        return std::nullopt;
    }

    std::string materialError;
    auto material = diffractionmaterial::admit(request.material, materialError);
    if (! material)
    {
        error = "imported diffraction material admission rejected: " + materialError;
        return std::nullopt;
    }
    std::string expectedDigest = material->structuralDigest();
    std::optional<diffractivefoil::AdmittedDiffractiveFoilIR> spatialFoil;
    if (request.spatialFoil)
    {
        auto admittedFoil = diffractivefoil::admit(*request.spatialFoil, materialError);
        if (!admittedFoil)
        {
            error = "imported diffraction foil admission rejected: " + materialError;
            return std::nullopt;
        }
        expectedDigest = admittedFoil->structuralDigest();
        spatialFoil.emplace(std::move(*admittedFoil));
    }
    if (request.structuralDigest != expectedDigest)
    {
        error = "imported diffraction material structural digest does not match its admitted payload";
        return std::nullopt;
    }

    std::string lightingError;
    auto lighting = diffractionmaterial::AdmittedLightingPlan::admit(
        request.lighting, lightingError);
    if (!lighting)
    {
        error = "imported diffraction lighting admission rejected: " + lightingError;
        return std::nullopt;
    }
    if (!diffractionmaterial::sameLightingDescription(
            request.lighting, diffractionmaterial::makeReferenceLighting()))
    {
        error = "imported diffraction lighting does not match the version-defined canonical plan";
        return std::nullopt;
    }

    error.clear();
    return AdmittedImportedSceneDiffractionMaterial(
        request, std::move(*material), std::move(*lighting), std::move(spatialFoil));
}
} // namespace diffractionmaterialbinding
