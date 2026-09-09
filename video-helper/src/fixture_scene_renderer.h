#pragma once

#include "gpu_backend/backend.h"
#include "diffraction_material_binding_admission.h"
#include "material_program_compiler.h"
#include "../../shared/SurfaceMaterialBindingContract.h"
#include "../../shared/DiffractionProductPlans.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace videorender::fixture3d
{

inline constexpr std::string_view kNativeGpuCapability = "native-gpu";

struct RenderDimensions
{
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

enum class RenderUse : std::uint8_t
{
    Preview = 0,
    Export = 1
};

struct RenderedFrame
{
    RenderUse use = RenderUse::Preview;
    HarmonicMIDI::grid::Scene3DId sceneId {};
    RenderDimensions dimensions {};
    std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> nativeFrame;
    arbitgpu::NativeFixtureSceneStats stats {};
};

class AdmittedSurfaceMaterialBinding final
{
public:
    HarmonicMIDI::grid::Scene3DId scene() const noexcept { return scene_; }
    HarmonicMIDI::grid::SceneObjectId object() const noexcept { return object_; }
    std::uint64_t sceneRevision() const noexcept { return sceneRevision_; }
    std::uint64_t structuralRevision() const noexcept { return structuralRevision_; }
    std::uint64_t evaluationRevision() const noexcept { return evaluationRevision_; }
    std::uint64_t programRevision() const noexcept { return programRevision_; }
    const std::string& bindingDigest() const noexcept { return bindingDigest_; }
    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& sceneSnapshot() const noexcept
    {
        return sceneSnapshot_;
    }
    const std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram>&
    nativeProgram() const noexcept
    {
        return nativeProgram_;
    }

private:
    friend std::shared_ptr<const AdmittedSurfaceMaterialBinding>
    admitSurfaceMaterialBinding (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>&,
        const surfacematerialbinding::ImportedSceneMaterialRequest&,
        videohelper::materialprogram::BackendTarget,
        std::string&);
    friend class FixtureSceneRenderer;

    AdmittedSurfaceMaterialBinding (
        HarmonicMIDI::grid::Scene3DId scene,
        HarmonicMIDI::grid::SceneObjectId object,
        std::uint64_t sceneRevision,
        std::uint64_t structuralRevision,
        std::uint64_t evaluationRevision,
        std::uint64_t programRevision,
        std::string bindingDigest,
        std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> sceneSnapshot,
        std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram> nativeProgram)
        : scene_ (scene), object_ (object), sceneRevision_ (sceneRevision),
          structuralRevision_ (structuralRevision),
          evaluationRevision_ (evaluationRevision),
          programRevision_ (programRevision), bindingDigest_ (std::move (bindingDigest)),
          sceneSnapshot_ (std::move (sceneSnapshot)),
          nativeProgram_ (std::move (nativeProgram))
    {
    }

    const HarmonicMIDI::grid::Scene3DId scene_ {};
    const HarmonicMIDI::grid::SceneObjectId object_ {};
    const std::uint64_t sceneRevision_ = 0;
    const std::uint64_t structuralRevision_ = 0;
    const std::uint64_t evaluationRevision_ = 0;
    const std::uint64_t programRevision_ = 0;
    const std::string bindingDigest_;
    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> sceneSnapshot_;
    const std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram> nativeProgram_;
};

std::shared_ptr<const AdmittedSurfaceMaterialBinding>
admitSurfaceMaterialBinding (
    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
    const surfacematerialbinding::ImportedSceneMaterialRequest& request,
    videohelper::materialprogram::BackendTarget target,
    std::string& error);

class AdmittedDiffractionMaterialBinding final
{
public:
    HarmonicMIDI::grid::Scene3DId scene() const noexcept { return scene_; }
    HarmonicMIDI::grid::SceneObjectId object() const noexcept { return object_; }
    std::uint64_t sceneRevision() const noexcept { return sceneRevision_; }
    std::uint64_t structuralRevision() const noexcept { return structuralRevision_; }
    std::uint64_t evaluationRevision() const noexcept { return evaluationRevision_; }
    std::uint64_t materialRevision() const noexcept { return materialRevision_; }
    const std::string& bindingDigest() const noexcept { return bindingDigest_; }
    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& sceneSnapshot() const noexcept
    {
        return sceneSnapshot_;
    }
    const std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram>&
    nativeProgram() const noexcept
    {
        return nativeProgram_;
    }
    const std::shared_ptr<const diffractionmaterial::ImmutableDiffractionScenePlan>&
    productPlan() const noexcept { return productPlan_; }
    const std::string& productDigest() const noexcept { return productDigest_; }

private:
    friend std::shared_ptr<const AdmittedDiffractionMaterialBinding>
    admitDiffractionMaterialBinding (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>&,
        const diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest&,
        videohelper::materialprogram::BackendTarget,
        std::string&);
    friend std::shared_ptr<const AdmittedDiffractionMaterialBinding>
    admitDiffractionProductPlan (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>&,
        const diffractionmaterial::DiffractionProductRenderRequest&,
        std::uint64_t, std::uint64_t, std::uint64_t,
        videohelper::materialprogram::BackendTarget,
        std::string&);
    friend class FixtureSceneRenderer;

    AdmittedDiffractionMaterialBinding (
        const diffractionmaterialbinding::AdmittedImportedSceneDiffractionMaterial& admitted,
        std::string bindingDigest,
        std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> sceneSnapshot,
        std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram> nativeProgram,
        std::shared_ptr<const diffractionmaterial::ImmutableDiffractionScenePlan> productPlan = {},
        std::string productDigest = {})
        : scene_ (admitted.scene()), object_ (admitted.object()),
          sceneRevision_ (admitted.sceneRevision()),
          structuralRevision_ (admitted.structuralRevision()),
          evaluationRevision_ (admitted.evaluationRevision()),
          materialRevision_ (admitted.materialRevision()),
          bindingDigest_ (std::move (bindingDigest)),
          sceneSnapshot_ (std::move (sceneSnapshot)),
          nativeProgram_ (std::move (nativeProgram)), productPlan_ (std::move (productPlan)),
          productDigest_ (std::move (productDigest))
    {
    }

    AdmittedDiffractionMaterialBinding (
        const AdmittedDiffractionMaterialBinding& admitted,
        std::shared_ptr<const diffractionmaterial::ImmutableDiffractionScenePlan> productPlan,
        std::string productDigest)
        : scene_ (admitted.scene_), object_ (admitted.object_),
          sceneRevision_ (admitted.sceneRevision_),
          structuralRevision_ (admitted.structuralRevision_),
          evaluationRevision_ (admitted.evaluationRevision_),
          materialRevision_ (admitted.materialRevision_),
          bindingDigest_ (admitted.bindingDigest_), sceneSnapshot_ (admitted.sceneSnapshot_),
          nativeProgram_ (admitted.nativeProgram_), productPlan_ (std::move (productPlan)),
          productDigest_ (std::move (productDigest))
    {
    }

    AdmittedDiffractionMaterialBinding (
        const AdmittedDiffractionMaterialBinding& admitted,
        std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram> nativeProgram)
        : scene_ (admitted.scene_), object_ (admitted.object_),
          sceneRevision_ (admitted.sceneRevision_),
          structuralRevision_ (admitted.structuralRevision_),
          evaluationRevision_ (admitted.evaluationRevision_),
          materialRevision_ (admitted.materialRevision_),
          bindingDigest_ (admitted.bindingDigest_), sceneSnapshot_ (admitted.sceneSnapshot_),
          nativeProgram_ (std::move (nativeProgram)), productPlan_ (admitted.productPlan_),
          productDigest_ (admitted.productDigest_)
    {
    }

    const HarmonicMIDI::grid::Scene3DId scene_ {};
    const HarmonicMIDI::grid::SceneObjectId object_ {};
    const std::uint64_t sceneRevision_ = 0;
    const std::uint64_t structuralRevision_ = 0;
    const std::uint64_t evaluationRevision_ = 0;
    const std::uint64_t materialRevision_ = 0;
    const std::string bindingDigest_;
    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> sceneSnapshot_;
    const std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram> nativeProgram_;
    const std::shared_ptr<const diffractionmaterial::ImmutableDiffractionScenePlan> productPlan_;
    const std::string productDigest_;
};

std::shared_ptr<const AdmittedDiffractionMaterialBinding>
admitDiffractionMaterialBinding (
    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
    const diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest& request,
    videohelper::materialprogram::BackendTarget target,
    std::string& error);

std::shared_ptr<const AdmittedDiffractionMaterialBinding>
admitDiffractionProductPlan (
    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
    const diffractionmaterial::DiffractionProductRenderRequest& request,
    std::uint64_t sceneRevision,
    std::uint64_t structuralRevision,
    std::uint64_t evaluationRevision,
    videohelper::materialprogram::BackendTarget target,
    std::string& error);

class FixtureSceneRenderer
{
public:
    static constexpr std::uint32_t kMaxExtent = 4096;
    static constexpr std::uint64_t kMaxPixels = 4096ull * 4096ull;

    explicit FixtureSceneRenderer (arbitgpu::NativeFixtureSceneBackend& backend) noexcept;

    bool renderPreview (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& snapshot,
        RenderDimensions dimensions,
        std::string_view backendCapability,
        RenderedFrame& output,
        std::string& error);

    bool renderPreview (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& snapshot,
        const std::shared_ptr<const AdmittedSurfaceMaterialBinding>& material,
        RenderDimensions dimensions,
        std::string_view backendCapability,
        RenderedFrame& output,
        std::string& error);

    bool renderPreview (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& snapshot,
        const std::shared_ptr<const AdmittedSurfaceMaterialBinding>& material,
        surfacematerial::MaterialEvaluationInputs runtimeInputs,
        arbitgpu::NativeFixtureSceneRuntimeInputs sceneInputs,
        RenderDimensions dimensions,
        std::string_view backendCapability,
        RenderedFrame& output,
        std::string& error);

    bool renderPreview (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& snapshot,
        const std::shared_ptr<const AdmittedSurfaceMaterialBinding>& material,
        surfacematerial::MaterialEvaluationInputs runtimeInputs,
        RenderDimensions dimensions,
        std::string_view backendCapability,
        RenderedFrame& output,
        std::string& error);

    bool renderPreview (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& snapshot,
        const std::shared_ptr<const AdmittedDiffractionMaterialBinding>& material,
        arbitgpu::NativeFixtureSceneRuntimeInputs sceneInputs,
        RenderDimensions dimensions,
        std::string_view backendCapability,
        RenderedFrame& output,
        std::string& error);

    bool renderExport (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& snapshot,
        RenderDimensions dimensions,
        std::string_view backendCapability,
        RenderedFrame& output,
        std::string& error);

    bool renderExport (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& snapshot,
        const std::shared_ptr<const AdmittedSurfaceMaterialBinding>& material,
        RenderDimensions dimensions,
        std::string_view backendCapability,
        RenderedFrame& output,
        std::string& error);

    bool renderExport (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& snapshot,
        const std::shared_ptr<const AdmittedSurfaceMaterialBinding>& material,
        surfacematerial::MaterialEvaluationInputs runtimeInputs,
        arbitgpu::NativeFixtureSceneRuntimeInputs sceneInputs,
        RenderDimensions dimensions,
        std::string_view backendCapability,
        RenderedFrame& output,
        std::string& error);

    bool renderExport (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& snapshot,
        const std::shared_ptr<const AdmittedSurfaceMaterialBinding>& material,
        surfacematerial::MaterialEvaluationInputs runtimeInputs,
        RenderDimensions dimensions,
        std::string_view backendCapability,
        RenderedFrame& output,
        std::string& error);

    bool renderExport (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& snapshot,
        const std::shared_ptr<const AdmittedDiffractionMaterialBinding>& material,
        arbitgpu::NativeFixtureSceneRuntimeInputs sceneInputs,
        RenderDimensions dimensions,
        std::string_view backendCapability,
        RenderedFrame& output,
        std::string& error);

private:
    struct CachedResources
    {
        std::weak_ptr<const HarmonicMIDI::grid::Visual3DScene> snapshot;
        std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram> materialProgram;
        std::shared_ptr<const arbitgpu::NativeFixtureSceneResources> native;
        arbitgpu::NativeFixtureSceneStats footprint {};
    };

    bool render (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& snapshot,
        RenderDimensions dimensions,
        std::string_view backendCapability,
        RenderUse use,
        const std::shared_ptr<const AdmittedSurfaceMaterialBinding>& material,
        const std::shared_ptr<const AdmittedDiffractionMaterialBinding>& diffractionMaterial,
        surfacematerial::MaterialEvaluationInputs runtimeInputs,
        arbitgpu::NativeFixtureSceneRuntimeInputs sceneInputs,
        RenderedFrame& output,
        std::string& error);

    arbitgpu::NativeFixtureSceneBackend& backend_;
    CachedResources cache_;
};

} // namespace videorender::fixture3d
