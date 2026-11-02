#pragma once

#include "gpu_backend/backend.h"
#include "gpu_backend/fixture_rough_v1_fallback.h"
#include "diffraction_material_binding_admission.h"
#include "material_program_compiler.h"
#include "../../shared/SurfaceMaterialBindingContract.h"
#include "../../shared/SurfaceMaterialProgramBinding.h"
#include "../../shared/DiffractionProductPlans.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
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

// True when the last Preview fixture render on this thread left a glass caustic
// submission on the GPU. Read after the layer renders: while it holds, the caustic's
// spectrum is adopted by a later frame's poll, so the caller's idle gate must not
// settle. It goes false once the poll adopts, drops or forgets that submission.
bool previewCausticPendingOnThisThread() noexcept;

// The readiness seam beside that pending flag: true once when the submitted
// caustic program sets still in flight on this thread's context all read
// complete. The viewport polls it each tick while the pending flag holds and
// wakes for the one adopting frame, instead of waiting out the wake-frame cap
// on a driver that links slower than the cap. Needs a current context; false
// without one, with nothing in flight, or after the set's one true, and false
// on backends with no asynchronous link to poll.
bool takePendingCausticProgramCompletionOnThisThread() noexcept;

// True when the last fixture render on this thread was a Path Trace Preview that
// still wants samples. While it holds, the caller's idle gate must keep producing
// frames, because each frame resolves one more pass and the image converges on
// screen. It goes false once the tracer reaches its preview sample target, or the
// next render is not a progressing Path Trace Preview.
bool pathTracePreviewProgressingOnThisThread() noexcept;

// The preview sample target of the last fixture render on this thread, zero
// when it was not a Path Trace Preview.
unsigned pathTracePreviewTargetOnThisThread() noexcept;

// The sample count the last fixture render reported, zeroed with the target.
// The viewport's hold ends on a target or on a stall (the count stops rising),
// never on an assumed samples-per-frame ratio.
unsigned pathTracePreviewSamplesOnThisThread() noexcept;

// The denoiser that ran on the last fixture render, preview or export: "none",
// "native-filter", and later "optix-ai" or "oidn". This is what ran, never what
// was asked for.
const char* pathTraceDenoiserRanOnThisThread() noexcept;

// The viewport's transport state, set once per loop tick. The display repaints
// faster than the project frame rate, so a playing transport draws the same time
// more than once, and only this tells those draws from a stopped viewport.
void setPathTraceTransportPlaying (bool playing) noexcept;

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
// which stays refused. Realtime is the Eevee-class GPU draw: no ray walks and no
// caustics.
enum class SceneEngineKind : std::uint8_t
{
    Raster = 0,
    PathTrace = 1,
    Realtime = 2
};

// Quality mode the 3D Render node selected (visualimportedscenerender::RenderMode),
// shared by the viewport and by output. Accurate Preview draws take the export
// budgets (gpu_backend/fixture_preview_budget.h) and the Path Trace export sample
// streams; Fast Preview draws keep the Preview budgets. Export does not read it
// until tier K S4.
enum class SceneEngineMode : std::uint8_t
{
    Accurate = 0,
    Fast = 1
};

struct SceneEngineSelection
{
    SceneEngineKind engine = SceneEngineKind::Raster;
    std::uint32_t exportSamples = 64;
    std::uint32_t previewSamples = 4096;
    // Last, so positional initialisers of the fields above keep compiling.
    SceneEngineMode mode = SceneEngineMode::Accurate;
    bool denoise = false;
    std::uint32_t maxBounces = 128;
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
    std::uint32_t previewSamples = 4096;
    std::string backend;  // physical fixture backend name ("opengl", "metal")
    // Last, so positional initialisers of the fields above keep compiling.
    SceneEngineMode mode = SceneEngineMode::Accurate;
    // The object of an admitted diffraction material and that material, with
    // runtime parameters applied. The engine scatters the object's surfaces by
    // the material's diffraction orders. Null when the draw has none.
    HarmonicMIDI::grid::SceneObjectId diffractionObject {};
    std::shared_ptr<const diffractionmaterial::AdmittedDiffractionMaterialIR> diffractionMaterial;
    // The lighting plan authored with that material. Path Trace lights the scene
    // from a version 3 plan's source; the other plans' paths stand in for the
    // sky and bounce it already traces.
    std::optional<diffractionmaterial::LightingDescription> diffractionLighting;
    // A spatial foil's corners, each reciprocal direction u, v, period in nm and coverage.
    std::optional<std::array<std::array<float, 4>, 4>> diffractionField;
    // Denoise the resolved frame before upload, on both preview and export.
    bool denoise = false;
    // Bounces after the camera hit.
    std::uint32_t maxBounces = 128;
    // A Preview draw while the transport plays stays on the fixed playback budget,
    // even when the time did not move. Export draws never set it.
    bool transportPlaying = false;
};

// What the tracer did for one image. A Preview covers its whole accumulation, so
// the counters describe the image on screen. An Export covers its fixed samples.
struct PathTraceFrameStats
{
    RenderUse use = RenderUse::Preview;
    SceneEngineMode mode = SceneEngineMode::Accurate;
    std::uint64_t samplesAdded = 0;
    std::uint32_t samplesPerPixel = 0;
    std::uint64_t nonFiniteSamples = 0;
    std::uint64_t bounceLimitPaths = 0;
    std::uint64_t transparentLimitPaths = 0;
    std::uint64_t limitEndedPaths = 0;
    std::uint64_t lightPaths = 0;
    std::uint64_t causticSplats = 0;
    // The film this image was traced at.
    std::uint32_t filmWidth = 0;
    std::uint32_t filmHeight = 0;
    // The image blends earlier playback frames (pathtrace/engine.h). Export never does.
    bool temporalDenoise = false;
    // What the tracer is doing, for the viewport header: "converging", "settled",
    // "playback-fixed" or "export".
    const char* renderActivity = "converging";
    // The samples per pixel that activity aims for: the viewport target, the
    // playback budget or the export count.
    std::uint32_t sampleTarget = 0;
    // The denoiser that ran on this image.
    const char* denoiser = "none";
    // The device that traced this image and the setting that chose it.
    const char* device = "CPU";
    const char* deviceRequested = "Auto";
    // Why Auto is not on a GPU device, for the Render panel. Null when it is, or
    // when the device was asked for by name.
    std::shared_ptr<const std::string> deviceReason;
};

class SceneRenderEngine
{
public:
    virtual ~SceneRenderEngine() = default;
    // Fills output with a compositor-ready native frame, or fails with error.
    virtual bool render (const SceneEngineDraw& draw, RenderedFrame& output, std::string& error) = 0;
    // Thread safe. Stops work in flight; the next preview starts over.
    virtual void cancel() noexcept = 0;
    // True while a Preview render still wants samples: the engine accumulates
    // progressively and the viewport must not settle on its first noisy frame.
    // One-frame-per-call engines keep the default, which reports nothing.
    virtual bool previewProgressing() const noexcept { return false; }
    // The sample target of the live accumulation. Engines that do not accumulate
    // report zero.
    virtual unsigned previewTarget() const noexcept { return 0; }
    // Samples accumulated so far; engines that do not accumulate report zero.
    virtual unsigned previewSampleCount() const noexcept { return 0; }
    // Counters of the image the engine last traced. Engines without them, and an
    // engine that has not traced yet, return false.
    virtual bool pathTraceStats (PathTraceFrameStats&) const noexcept { return false; }
    // The name of the denoiser that ran on the last frame, "none" when none did.
    virtual const char* denoiserRan() const noexcept { return "none"; }
};

// Counters of the Path Trace renders behind the latest frame of one use, or
// nothing when that frame had none. Export sums every Path Trace layer since
// clearPathTraceExportStats, so the composite probe clears before it renders.
// Preview holds the last Path Trace render until the next Preview render finishes:
// a trace replaces it, and a render that reports nothing retires it. The two never
// mix, so a viewport frame cannot change what a probe reports.
void clearPathTraceExportStats() noexcept;
std::optional<PathTraceFrameStats> lastPathTraceStats (RenderUse use);
// A frame with a stored V1 closed volume that V1 couldn't finish ran on V2 in the last two seconds.
void noteRoughV1RanAsV2() noexcept;
bool roughV1RanAsV2Recently() noexcept;

// What a Realtime draw did, kept under the same Preview and Export rules.
struct RealtimeFrameStats
{
    RenderUse use = RenderUse::Preview;
    SceneEngineMode mode = SceneEngineMode::Accurate;
    // False when the backend has no Realtime program yet and drew its raster one,
    // so the reply never calls those pixels Realtime.
    bool realtimeProgram = false;
    bool causticsNeedPathTrace = false;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};
std::optional<RealtimeFrameStats> lastRealtimeStats (RenderUse use);

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

    // The prepared static GPU resources behind a preview render. An animated
    // imported scene takes a new snapshot every frame, so the preview owner that
    // holds this renderer is rebuilt every frame too; carrying the cache across
    // that rebuild is what lets the resident water refit plan — and the resident
    // caustic schedule it creates — survive longer than one frame.
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
        bool realtime = false;
        bool cameraVolumeOnly = false;
        // Set by a Preview structural hit. The resources then hold refreshed water
        // vertices that only the Preview resident query path reads, so any other
        // frame (export, caustics, diffraction) takes a full prepare.
        bool residentRefreshed = false;
    };

    CachedResources releaseCache () noexcept
    {
        CachedResources out = std::move (cache_);
        cache_ = {};
        return out;
    }
    void adoptCache (CachedResources resources) noexcept { cache_ = std::move (resources); }

    // The path tracer rides the same owner rebuild as the cache. It holds the
    // preview's accumulated samples, so a fresh engine per frame would restart
    // the preview at one noisy sample for ever; a changed scene still compares
    // by content inside the engine and re-ingests on a real difference.
    std::shared_ptr<SceneRenderEngine> releasePathTraceEngine () noexcept
    {
        std::shared_ptr<SceneRenderEngine> out = std::move (pathTraceEngine_);
        pathTraceEngine_.reset();
        return out;
    }
    void adoptPathTraceEngine (std::shared_ptr<SceneRenderEngine> engine) noexcept
    {
        pathTraceEngine_ = std::move (engine);
    }

private:
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
        HarmonicMIDI::grid::SceneObjectId diffractionObject,
        std::shared_ptr<const diffractionmaterial::AdmittedDiffractionMaterialIR> diffractionMaterial,
        std::optional<diffractionmaterial::LightingDescription> diffractionLighting,
        std::optional<std::array<std::array<float, 4>, 4>> diffractionField,
        RenderedFrame& output, std::string& error);

    bool renderAttempt (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& snapshot,
        RenderDimensions dimensions, std::string_view backendCapability,
        RenderUse use,
        const std::shared_ptr<const AdmittedSurfaceMaterialBinding>& material,
        const std::shared_ptr<const AdmittedDiffractionMaterialBinding>& diffractionMaterial,
        surfacematerial::MaterialEvaluationInputs runtimeInputs,
        arbitgpu::NativeFixtureSceneRuntimeInputs sceneInputs,
        RenderedFrame& output, std::string& error);

    arbitgpu::NativeFixtureSceneBackend& backend_;
    CachedResources cache_;
    // A second try for a frame V1 couldn't finish.
    CachedResources roughV2Cache_;
    arbitgpu::smoothglass::RoughV1Fallback roughV1Fallback_;
    std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram> roughV2Source_, roughV2Program_;
    bool roughV2Attempt_ = false, roughV2Ran_ = false;
    SceneEngineSelection engineSelection_ {};
    std::shared_ptr<SceneRenderEngine> pathTraceEngine_;
};

} // namespace videorender::fixture3d
