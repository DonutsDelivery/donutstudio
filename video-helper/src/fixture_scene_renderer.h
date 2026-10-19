#pragma once

#include "gpu_backend/backend.h"
#include "diffraction_material_binding_admission.h"
#include "material_program_compiler.h"
#include "../../shared/SurfaceMaterialBindingContract.h"
#include "../../shared/SurfaceMaterialProgramBinding.h"
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
    // Set only for a Preview render whose backend has not finished. No frame is present.
    bool pending = false;
    // Non-fatal Preview warning: why the glass caustic rendered reduced or not at all,
    // for example "caustic off: exceeds GL limits (...)". Empty when it rendered as planned.
    // Export never sets it; an export that cannot fit the caustic refuses instead.
    std::string causticDegradation;
};

// Number of Preview renders on the calling thread that came back pending from the
// fixture GPU backend. Diagnostics only: a caller compares two reads around a frame.
std::uint64_t previewPendingResultsOnThisThread() noexcept;

namespace detail
{
inline constexpr bool frameTextureFootprint (
    std::uint64_t permanentBytes,
    const arbitgpu::NativeFixtureSceneStats& submission,
    bool glassCaustic,
    std::uint64_t& totalBytes) noexcept
{
    return renderpassoutput::detail::checkedAdd (permanentBytes,
        glassCaustic ? submission.textureBytes : submission.roughGlassReceiptBytes, totalBytes);
}
} // namespace detail

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
    std::uint32_t roughGlassScatteringVersion() const noexcept { return nativeProgram_->roughGlassScatteringVersion; }
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
    friend std::shared_ptr<const AdmittedSurfaceMaterialBinding> admitSurfaceMaterialCollection (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>&,
        const geometrysurfacematerial::ObjectPrograms&, const videowire::geometry::RuntimeFieldEvaluation&,
        videohelper::materialprogram::BackendTarget, std::string&);

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

std::shared_ptr<const AdmittedSurfaceMaterialBinding> admitSurfaceMaterialCollection (
    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
    const geometrysurfacematerial::ObjectPrograms& programs,
    const videowire::geometry::RuntimeFieldEvaluation& evaluation,
    videohelper::materialprogram::BackendTarget target, std::string& error);

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
    arbitgpu::NativeFixtureDiffractionConsumer consumer() const noexcept { return consumer_; }
    const std::shared_ptr<const diffractionmaterial::AdmittedDiffractionMaterialIR>&
    materialIR() const noexcept { return materialIR_; }
    const std::optional<diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest>&
    authoredRequest() const noexcept { return authoredRequest_; }

private:
    friend std::shared_ptr<const AdmittedDiffractionMaterialBinding>
    admitDiffractionMaterialBinding (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>&,
        const diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest&,
        videohelper::materialprogram::BackendTarget,
        std::string&, bool, arbitgpu::NativeFixtureDiffractionConsumer);
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
        std::string productDigest = {},
        std::optional<diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest> authoredRequest = {},
        arbitgpu::NativeFixtureDiffractionConsumer consumer = arbitgpu::NativeFixtureDiffractionConsumer::FiniteLobe)
        : scene_ (admitted.scene()), object_ (admitted.object()),
          sceneRevision_ (admitted.sceneRevision()),
          structuralRevision_ (admitted.structuralRevision()),
          evaluationRevision_ (admitted.evaluationRevision()),
          materialRevision_ (admitted.materialRevision()),
          bindingDigest_ (std::move (bindingDigest)),
          sceneSnapshot_ (std::move (sceneSnapshot)),
          nativeProgram_ (std::move (nativeProgram)), productPlan_ (std::move (productPlan)),
          productDigest_ (std::move (productDigest)), authoredRequest_ (std::move(authoredRequest)), consumer_ (consumer),
          materialIR_ (consumer == arbitgpu::NativeFixtureDiffractionConsumer::SmoothGlassDiscreteOrders
              ? std::make_shared<const diffractionmaterial::AdmittedDiffractionMaterialIR>(admitted.material()) : nullptr)
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
          productDigest_ (std::move (productDigest)), authoredRequest_ (admitted.authoredRequest_),
          consumer_ (admitted.consumer_), materialIR_ (admitted.materialIR_)
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
          productDigest_ (admitted.productDigest_), authoredRequest_ (admitted.authoredRequest_),
          consumer_ (admitted.consumer_), materialIR_ (admitted.materialIR_)
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
    const std::optional<diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest> authoredRequest_;
    const arbitgpu::NativeFixtureDiffractionConsumer consumer_;
    const std::shared_ptr<const diffractionmaterial::AdmittedDiffractionMaterialIR> materialIR_;
};

// The discrete consumer admits the strict compiler-owned grating prerequisite.
// The mixed caller must also admit its selected smooth glass; Color stays guarded.
std::shared_ptr<const AdmittedDiffractionMaterialBinding>
admitDiffractionMaterialBinding (
    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
    const diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest& request,
    videohelper::materialprogram::BackendTarget target,
    std::string& error, bool exactComposedTarget = false,
    arbitgpu::NativeFixtureDiffractionConsumer consumer = arbitgpu::NativeFixtureDiffractionConsumer::FiniteLobe);

std::shared_ptr<const AdmittedDiffractionMaterialBinding>
admitDiffractionProductPlan (
    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
    const diffractionmaterial::DiffractionProductRenderRequest& request,
    std::uint64_t sceneRevision,
    std::uint64_t structuralRevision,
    std::uint64_t evaluationRevision,
    videohelper::materialprogram::BackendTarget target,
    std::string& error);

// Render engine the 3D Render node selected (visualimportedscenerender::RenderEngine).
// Raster is the GPU fixture draw. PathTrace is the CPU path tracer, chosen by the
// user per node; it is not Visual3DSceneRenderContract's CPU image fallback,
// which stays refused.
enum class SceneEngineKind : std::uint8_t
{
    Raster = 0,
    PathTrace = 1
};

struct SceneEngineSelection
{
    SceneEngineKind engine = SceneEngineKind::Raster;
    std::uint32_t exportSamples = 64;
};

// One admitted draw handed to a non-raster engine. Everything here has passed
// the same scene, material and backend admission as a raster draw.
struct SceneEngineDraw
{
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> snapshot;
    std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram> material;
    const arbitgpu::NativeFixtureSceneRuntimeInputs* inputs = nullptr;
    RenderDimensions dimensions {};
    RenderUse use = RenderUse::Preview;
    std::uint32_t exportSamples = 64;
    std::string backend;  // physical fixture backend name ("opengl", "metal")
};

class SceneRenderEngine
{
public:
    virtual ~SceneRenderEngine() = default;
    // Fills output with a compositor-ready native frame, or fails with error.
    virtual bool render (const SceneEngineDraw& draw, RenderedFrame& output, std::string& error) = 0;
    // Thread safe. Stops work in flight; the next preview starts over.
    virtual void cancel() noexcept = 0;
};

// The path tracer lives in the helper executable only. It registers its
// factory at static initialisation, so the fixture renderer's other link
// targets (tests, tools) carry no tracer code. Without a factory a PathTrace
// draw fails with a clear error.
using SceneRenderEngineFactory = std::shared_ptr<SceneRenderEngine> (*)();
void setPathTraceSceneEngineFactory (SceneRenderEngineFactory factory) noexcept;

class FixtureSceneRenderer
{
public:
    static constexpr std::uint32_t kMaxExtent = 4096;
    static constexpr std::uint64_t kMaxPixels = 4096ull * 4096ull;

    explicit FixtureSceneRenderer (arbitgpu::NativeFixtureSceneBackend& backend) noexcept;

    // Applies to every following render call. Raster releases the path
    // tracer's caches; a PathTrace selection keeps them across draws.
    void selectEngine (SceneEngineSelection selection);
    SceneEngineSelection selectedEngine() const noexcept { return engineSelection_; }

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

    bool renderMixed (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& snapshot,
        const std::shared_ptr<const AdmittedSurfaceMaterialBinding>& surfaces,
        const std::shared_ptr<const AdmittedDiffractionMaterialBinding>& diffraction,
        arbitgpu::NativeFixtureSceneRuntimeInputs sceneInputs,
        RenderDimensions dimensions, RenderUse use,
        std::string_view backendCapability, RenderedFrame& output, std::string& error);

private:
    struct CachedResources
    {
        std::weak_ptr<const HarmonicMIDI::grid::Visual3DScene> snapshot;
        std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram> materialProgram;
        std::shared_ptr<const arbitgpu::NativeFixtureSceneResources> native;
        arbitgpu::NativeFixtureSceneStats footprint {};
        std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram> mixedSurfaces, mixedDiffraction;
        // Selected closed volume prepared beside a thin camera dielectric, or invalid.
        // cameraVolumeOnly separates a camera-only selection (Glass Caustics off)
        // from a caustic selection of the same volume.
        HarmonicMIDI::grid::SceneObjectId causticVolume {};
        bool cameraVolumeOnly = false;
        // Set by a Preview structural hit. The resources then hold refreshed water
        // vertices that only the Preview resident query path reads, so any other
        // frame (export, caustics, diffraction) takes a full prepare.
        bool residentRefreshed = false;
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

    bool renderWithEngine (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& snapshot,
        const std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram>& material,
        const arbitgpu::NativeFixtureSceneRuntimeInputs& sceneInputs,
        RenderDimensions dimensions, RenderUse use, const std::string& backend,
        RenderedFrame& output, std::string& error);

    arbitgpu::NativeFixtureSceneBackend& backend_;
    CachedResources cache_;
    SceneEngineSelection engineSelection_ {};
    std::shared_ptr<SceneRenderEngine> pathTraceEngine_;
};

} // namespace videorender::fixture3d
