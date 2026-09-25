// gpu_backend/backend.h -- language-neutral native GPU backend seam.
//
// The existing FrameRenderer remains on OpenGL while the visual-engine P6
// migration lands in slices.  This seam makes native-backend capability and
// verification available without leaking Objective-C types into the helper.
#pragma once

#include "../canonical_block_c_frame.h"

#include "../../../shared/RenderPassOutputContract.h"
#include "../../../shared/AovInspectionOperationContract.h"
#include "../../../shared/SceneAovOperationContract.h"
#include "../../../shared/Render3DImageOutput.h"
#include "../../../shared/ColorTransformContract.h"
#include "../optical_flow_contract.h"
#include "../diffraction_material_execution.h"
#include "../../../shared/DiffractionMaterialGpuLayout.h"
#include "../../../shared/DiffractiveFoilIR.h"
#include "../../../shared/DiffractionLightingPlan.h"
#include "../../../shared/DiffractionRuntimeParameters.h"
#include "../../../shared/SdfIr.h"
#include "../../../shared/SurfaceMaterialIR.h"
#include "../../../shared/SurfaceMaterialBindingContract.h"
#include "../../../shared/Visual3DScene.h"
#include "../../../shared/GeometryCore.h"
#include "../../../shared/VisualNoteInstancingContract.h"
#include "../../../shared/VisualAnimationDeformationEvaluation.h"
#include "frame_memory_budget.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace arbitgpu { class NativeSdfCompiledProgram; }
namespace videohelper::materialprogram { class MaterialProgramRecord; }
namespace videohelper::geometry { class AdmittedPlanValue; }
namespace videohelper::sdf
{
inline std::shared_ptr<const arbitgpu::NativeSdfCompiledProgram> compileNativeSdfProgram (
    const videowire::SdfIr&, std::string&);
}

namespace arbitgpu
{

struct BackendInfo
{
    bool available = false;
    bool compute = false;
    std::string backend;
    std::string device;
    std::string error;
};

struct BackendSelfTest : BackendInfo
{
    bool computePassed = false;
    bool renderPassed = false;
    uint32_t computeChecksum = 0;
    uint32_t renderChecksum = 0;
};

// A render-pass output lifecycle handle is an opaque backend admission token.
// It is not a texture, buffer, or proof that a render or readback occurred.
struct RenderPassOutputLifecycleHandle
{
    uint64_t value = 0;

    explicit operator bool() const noexcept { return value != 0; }
};

inline bool operator== (RenderPassOutputLifecycleHandle left,
                        RenderPassOutputLifecycleHandle right) noexcept
{
    return left.value == right.value;
}

struct RenderPassOutputCapabilities
{
    bool available = false;
    std::array<bool, renderpassoutput::kMaximumAttachments> supportedOutputs {};
    renderpassoutput::AdmissionLimits limits;
    std::string error;

    bool supports (renderpassoutput::Output output) const noexcept
    {
        const auto index = static_cast<size_t> (output);
        return index < supportedOutputs.size() && supportedOutputs[index];
    }
};

// Backend-local handles for one allocated attachment. These handles are
// immutable publication metadata. They do not prove that a render pass wrote
// the attachment or that readback is available.
struct RenderPassOutputResource
{
    renderpassoutput::Output output = renderpassoutput::Output::Count;
    std::uintptr_t image = 0;
    std::uintptr_t attachmentView = 0;
    std::uintptr_t textureView = 0;
};

struct RenderPassOutputAdmission
{
    RenderPassOutputLifecycleHandle lifecycle;
    std::vector<RenderPassOutputResource> resources;
    std::string error;
    FrameMemoryAdmission frameMemory;
};

inline bool admitRenderPassOutputFrameMemory (
    const renderpassoutput::AdmittedOutputs& outputs,
    std::uint64_t byteBudget,
    FrameMemoryAdmission& admission,
    std::string& error)
{
    FrameMemoryAdmissionRequest request;
    request.byteBudget = byteBudget;
    request.classCount = outputs.attachments().size();
    if (request.classCount == 0 || request.classCount > request.classes.size())
    {
        admission = {};
        admission.failure = FrameMemoryAdmissionFailure::invalidRequest;
        admission.byteBudget = byteBudget;
        error = "native frame-memory render-pass attachment count is invalid";
        return false;
    }
    for (std::size_t index = 0; index < request.classCount; ++index)
    {
        const auto& attachment = outputs.attachments()[index];
        request.classes[index] = {
            attachment.extent.width,
            attachment.extent.height,
            renderpassoutput::bytesPerPixel (attachment.format),
            1
        };
    }
    return admitFrameMemory (request, admission, error);
}

struct RenderPassColorAovClear final
{
    std::array<float, 4> linearRgba { 0.0f, 0.0f, 0.0f, 0.0f };
};

struct RenderPassColorAovExecution final
{
    bool submitted = false;
    std::uint64_t submission = 0;
    std::string error;
};

// Motion v1 publishes pixel displacement in image coordinates. The first
// executable checkpoint writes one constant field; the product route uses the
// exact zero vector to represent an initialized/no-prior-sample motion AOV.
struct RenderPassMotionAovClear final
{
    std::array<float, 2> pixelDisplacement { 0.0f, 0.0f };
};

struct RenderPassMotionAovExecution final
{
    bool submitted = false;
    std::uint64_t submission = 0;
    std::string error;
};

struct RenderPassAovInspectionExecution final
{
    bool submitted = false;
    std::uint64_t submission = 0;
    std::string error;
};

struct RenderPassSceneAovExecution final
{
    bool submitted = false;
    std::uint64_t submission = 0;
    std::string error;
};

class RenderPassOutputBackend
{
public:
    virtual ~RenderPassOutputBackend() = default;

    virtual RenderPassOutputCapabilities renderPassOutputCapabilities() const = 0;
    virtual RenderPassOutputAdmission admitRenderPassOutputs (
        const renderpassoutput::AdmittedOutputs& outputs) = 0;
    // Executes one bounded native pass against the Color attachment owned by
    // lifecycle. The backend must reject stale lifecycles, missing Color AOVs,
    // and unsupported semantics. It may not emulate the pass on the CPU.
    virtual RenderPassColorAovExecution executeColorAovClear (
        RenderPassOutputLifecycleHandle lifecycle,
        const RenderPassColorAovClear& clear) = 0;
    // Executes one bounded native write against the exact admitted Motion AOV.
    // This is a GPU attachment pass, never a CPU-produced substitute.
    virtual RenderPassMotionAovExecution executeMotionAovClear (
        RenderPassOutputLifecycleHandle lifecycle,
        const RenderPassMotionAovClear& clear) = 0;
    // Maps a scene-produced source attachment into the exact admitted Color
    // attachment through a native GPU pass; no CPU pixel fallback exists.
    // Backends without the operation fail closed rather than fabricating readiness.
    virtual RenderPassAovInspectionExecution executeAovInspection (
        RenderPassOutputLifecycleHandle,
        const aovinspection::Payload&)
    {
        RenderPassAovInspectionExecution result;
        result.error = "native GPU AOV inspection execution is not compiled in";
        return result;
    }
    // Rasterizes the immutable admitted scene into the exact AOV attachment.
    // The backend may not substitute a CPU-produced image.
    virtual RenderPassSceneAovExecution executeSceneAov (
        RenderPassOutputLifecycleHandle lifecycle,
        const sceneaov::Payload& payload) = 0;
    virtual void releaseRenderPassOutputs (RenderPassOutputLifecycleHandle lifecycle) noexcept = 0;
};

// Immutable native resources bound to one exact admitted optical-flow frame.
// The opaque handles are backend-local GPU objects; production code must never
// substitute CPU pixels or a mutable cache entry at this boundary.
struct NativeOpticalFlowInputResource final
{
    videoopticalflow::ResourceIdentity identity {};
    std::uint64_t helperGeneration = 0;
    std::uint64_t structuralRevision = 0;
    videoopticalflow::FrameDescription descriptor;
    std::uintptr_t imageHandle = 0;
    std::uintptr_t textureViewHandle = 0;
    bool immutable = false;
};

struct NativeOpticalFlowOutputLifecycleHandle final
{
    std::uint64_t value = 0;
    explicit operator bool() const noexcept { return value != 0; }
};

struct NativeOpticalFlowSubmission final
{
    bool completed = false;
    std::uint64_t submission = 0;
    NativeOpticalFlowOutputLifecycleHandle lifecycle;
    // The backend reports the exact resource receipt it created. The execution
    // owner admits this receipt verbatim rather than fabricating format,
    // revision, or backend identity after submission.
    videoopticalflow::ResultDescription result;
    std::uintptr_t imageHandle = 0;
    std::uintptr_t textureViewHandle = 0;
    std::string error;
};

class NativeOpticalFlowExecutionBackend
{
public:
    virtual ~NativeOpticalFlowExecutionBackend() = default;
    virtual videoopticalflow::BackendCapabilities opticalFlowCapabilities() const = 0;
    virtual NativeOpticalFlowSubmission executeOpticalFlow (
        const videoopticalflow::AdmittedRequest& request,
        const NativeOpticalFlowInputResource& first,
        const NativeOpticalFlowInputResource& second,
        bool resetTemporalState) = 0;
    virtual void releaseOpticalFlowOutput (
        NativeOpticalFlowOutputLifecycleHandle lifecycle) noexcept = 0;
};

// Process-owned strict native implementation. Unsupported builds report an
// unavailable capability and reject every execution; there is no CPU fallback.
NativeOpticalFlowExecutionBackend& nativeOpticalFlowExecutionBackend();
void invalidateNativeOpticalFlowExecutionContext (std::uintptr_t contextIdentity) noexcept;

// Cheap device/capability query.  Does not initialize sokol_gfx or submit GPU
// work, so it is safe to expose in the regular version/capability RPCs.
BackendInfo queryNativeBackend();

// Native compute + offscreen-render validation. The P6 backend stays alive for
// the helper process because renderer-facing Metal resources now share it.
BackendSelfTest runNativeBackendSelfTest();

enum class NativeSdfQuality : std::uint8_t
{
    low = 0,
    medium = 1,
    high = 2,
    ultra = 3,
    count = 4
};

// Persistent SDF outputPass selections on the Image port. These values are
// distinct from renderpassoutput::Output and do not select raw AOV attachments.
enum class NativeSdfOutput : std::uint8_t
{
    color = 0,
    depth = 1,
    normal = 2,
    materialId = 3,
    curvature = 4,
    ambientOcclusion = 5,
    softShadow = 6,
    edgeDistance = 7,
    count = 8
};

struct NativeSdfExecutionCapabilities
{
    static constexpr std::size_t kOperationSlots =
        static_cast<std::size_t> (videowire::SdfOperation::domainWarp) + 1;

    bool available = false;
    std::string backend;
    std::string device;
    std::array<bool, kOperationSlots> supportedOperations {};
    std::array<bool, static_cast<std::size_t> (NativeSdfOutput::count)> supportedOutputs {};
    std::size_t maxOperations = 0;
    std::size_t maxDepth = 0;
    std::uint32_t maxExtent = 0;
    std::uint64_t maxPixels = 0;
    std::uint32_t maxSteps = 0;
    double minEpsilon = 0.0;
    double maxEpsilon = 0.0;
    double maxDistance = 0.0;
    std::string error;

    bool supports (videowire::SdfOperation operation) const noexcept
    {
        const auto index = static_cast<std::size_t> (operation);
        return index < supportedOperations.size() && supportedOperations[index];
    }

    bool supports (NativeSdfOutput output) const noexcept
    {
        const auto index = static_cast<std::size_t> (output);
        return index < supportedOutputs.size() && supportedOutputs[index];
    }
};

inline constexpr std::size_t kNativeSdfMaximumRecords = 64;
inline constexpr std::size_t kNativeSdfMaximumDepth = 32;
inline constexpr std::size_t kNativeSdfMaximumEvaluationSteps = 256;

struct NativeSdfCompiledRecord final
{
    // Retain all 64 bits for the primitive-contributor visualization.
    videowire::SdfStableId stableId = 0;
    std::uint32_t operation = 0;
    std::uint32_t input0 = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t input1 = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t parameterCount = 0;
    std::array<float, videowire::kSdfMaximumParametersPerRecord> parameters {};
};

class NativeSdfCompiledProgram final
{
public:
    std::uint32_t rootIndex() const noexcept { return rootIndex_; }
    std::uint32_t maximumDepth() const noexcept { return maximumDepth_; }
    std::uint32_t evaluationSteps() const noexcept { return evaluationSteps_; }
    const std::vector<NativeSdfCompiledRecord>& records() const noexcept { return records_; }
    const std::string& structuralDigest() const noexcept { return structuralDigest_; }

private:
    friend std::shared_ptr<const NativeSdfCompiledProgram>
        videohelper::sdf::compileNativeSdfProgram (const videowire::SdfIr&, std::string&);
    NativeSdfCompiledProgram() = default;
    std::uint32_t rootIndex_ = 0;
    std::uint32_t maximumDepth_ = 0;
    std::uint32_t evaluationSteps_ = 0;
    std::vector<NativeSdfCompiledRecord> records_;
    std::string structuralDigest_;
};

constexpr const char* nativeSdfOutputToken (NativeSdfOutput output) noexcept
{
    switch (output)
    {
        case NativeSdfOutput::color: return "color";
        case NativeSdfOutput::depth: return "depth";
        case NativeSdfOutput::normal: return "normal";
        case NativeSdfOutput::materialId: return "materialId";
        case NativeSdfOutput::curvature: return "curvature";
        case NativeSdfOutput::ambientOcclusion: return "ambientOcclusion";
        case NativeSdfOutput::softShadow: return "softShadow";
        case NativeSdfOutput::edgeDistance: return "edgeDistance";
        case NativeSdfOutput::count: break;
    }
    return "invalid";
}

struct NativeSdfDrawRequest final
{
    videowire::SdfIr geometry;
    std::shared_ptr<const NativeSdfCompiledProgram> compiledProgram;
    // Immutable renderer-owned cache footprint included in the frame receipt.
    std::uint64_t geometryCacheBytes = 0;
    std::uint32_t geometryCacheRecords = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t maximumSteps = 0;
    double epsilon = 0.0;
    double maximumDistance = 0.0;
    NativeSdfQuality adaptiveQuality = NativeSdfQuality::medium;
    NativeSdfQuality normalQuality = NativeSdfQuality::medium;
    NativeSdfQuality shadowQuality = NativeSdfQuality::medium;
    NativeSdfOutput output = NativeSdfOutput::color;
};

struct NativeSdfResourceReceipt final
{
    std::uint64_t compiledRecordBytes = 0;
    std::uint32_t compiledRecordCount = 0;
    std::uint64_t geometryCacheBytes = 0;
    std::uint64_t backendProgramBytes = 0;
    std::uint64_t uniformBytes = 0;
    std::uint64_t attachmentBytes = 0;
    std::uint64_t totalBytes = 0;
};

// The frame owns every native resource created for one submitted SDF draw.
// It contains no CPU pixels and may only be consumed by the matching backend.
struct NativeTextureViewDescriptor;
class NativeSdfSceneFrame
{
public:
    virtual ~NativeSdfSceneFrame() = default;
    virtual const std::string& backend() const noexcept = 0;
    virtual std::uint32_t width() const noexcept = 0;
    virtual std::uint32_t height() const noexcept = 0;
    virtual std::uintptr_t colorImageHandle() const noexcept = 0;
    virtual std::uintptr_t colorTextureViewHandle() const noexcept = 0;
    virtual NativeTextureViewDescriptor colorTextureDescriptor() const noexcept = 0;
    virtual const FrameMemoryAdmission& frameMemoryAdmission() const noexcept = 0;
    virtual const NativeSdfResourceReceipt& sdfResourceReceipt() const noexcept = 0;
    // Explicit diagnostic readback preserves finite float map values for native
    // acceptance. Normal rendering and composition remain GPU-owned.
    virtual bool readColorFloatPixels (std::vector<float>& output) const
    {
        output.clear();
        return false;
    }
    virtual bool readColorPixels (std::vector<std::uint8_t>& output) const
    {
        output.clear();
        return false;
    }
};

struct NativeSdfSceneSubmission final
{
    bool rendered = false;
    std::shared_ptr<const NativeSdfSceneFrame> frame;
    std::string error;
};

class NativeSdfExecutionBackend
{
public:
    virtual ~NativeSdfExecutionBackend() = default;
    virtual NativeSdfExecutionCapabilities capabilities() const = 0;
    virtual NativeSdfSceneSubmission render (const NativeSdfDrawRequest& request) = 0;
};

// This query is deliberately separate from queryNativeBackend(). A working
// Metal device does not imply that admitted SDF IR has a production renderer.
NativeSdfExecutionCapabilities queryNativeSdfExecution();

// Process-owned backend shared by preview and export. Unavailable builds reject
// every request without allocating a frame or evaluating the SDF on the CPU.
NativeSdfExecutionBackend& nativeSdfExecutionBackend();
// Call while the OpenGL context is current and before GLFW destroys it.
void invalidateNativeSdfExecutionContext (std::uintptr_t contextIdentity) noexcept;

// Process-lifetime backend owner shared by preview and export admission.
// Availability means the backend can allocate and own the exact admitted
// attachments. It does not claim that a pass wrote them or that readback exists.
RenderPassOutputBackend& nativeRenderPassOutputBackend();

struct NativeFixtureSceneStats
{
    std::uint32_t drawCount = 0;
    std::uint32_t ordinaryDrawCount = 0;
    std::uint32_t instancedDrawCount = 0;
    std::uint32_t instanceBufferUploadCount = 0;
    std::uint32_t submittedInstanceCount = 0;
    std::uint32_t noteInstanceDrawCount = 0;
    std::uint32_t noteInstanceTransformUploadCount = 0;
    std::uint32_t submittedNoteInstanceCount = 0;
    std::uint32_t staticUploadCount = 0;
    std::uint32_t textureUploadCount = 0;
    bool reusedStaticResources = false;
    std::uint64_t vertexBytes = 0;
    std::uint64_t indexBytes = 0;
    std::uint64_t materialBytes = 0;
    std::uint64_t textureBytes = 0;
    std::uint32_t materialProgramUploadCount = 0;
    bool reusedMaterialProgram = false;
    // Deterministic spatial-foil work receipt. One evaluation is one output
    // pixel slot reserved for one admitted lighting path.
    std::uint64_t diffractionEvaluationBudget = 0;
    std::uint64_t diffractionEvaluationCount = 0;
    // Successful native shader-program builds performed by this preparation.
    std::uint32_t shaderProgramBuildCount = 0;
};

// The compiler target is part of the immutable native-program receipt. Backends
// must reject receipts compiled for another API rather than reinterpret them.
enum class NativeFixtureMaterialBackend : std::uint8_t
{
    Invalid = 0,
    OpenGl = 1,
    Metal = 2
};

enum class NativeFixtureMaterialKind : std::uint8_t
{
    SurfacePbr = 0,
    DiffractionReflective = 1,
    DiffractionReflective1D = DiffractionReflective
};

// Immutable fixed-layout native material input. This is deliberately not a
// general graph or shader-source interface: admission publishes only one of
// the bounded built-in material programs consumed by the fixture pipeline.
struct NativeFixtureSurfaceMaterialProgram final
{
    enum class BaseColorSource : std::uint8_t
    {
        ConstantLinear = 0,
        ImportedSrgbTexture = 1,
        TimeLinearMix = 2,
        GraphFrameSrgbTexture = 3
    };

    struct ImportedSrgbTexture final
    {
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        std::size_t texelCount = 0;
        std::vector<HarmonicMIDI::grid::SceneTexelRgba8> texels;

        bool valid() const noexcept
        {
            using HarmonicMIDI::grid::Visual3DScene;
            return width > 0 && height > 0
                && width <= Visual3DScene::kMaxTextureDimension
                && height <= Visual3DScene::kMaxTextureDimension
                && static_cast<std::uint64_t>(width) * height == texelCount
                && texelCount <= Visual3DScene::kMaxTextureTexels
                && texels.size() == texelCount;
        }
    };

    static constexpr std::uint32_t kLayoutVersion = 13;

    std::uint32_t layoutVersion = kLayoutVersion;
    NativeFixtureMaterialBackend backend = NativeFixtureMaterialBackend::Invalid;
    NativeFixtureMaterialKind kind = NativeFixtureMaterialKind::SurfacePbr;
    HarmonicMIDI::grid::SceneObjectId object {};
    std::string bindingDigest;
    std::string programIdentity;
    std::shared_ptr<const videohelper::materialprogram::MaterialProgramRecord> vertexProgram;
    BaseColorSource baseColorSource = BaseColorSource::ConstantLinear;
    std::optional<ImportedSrgbTexture> importedBaseColorTexture;
    std::optional<surfacematerialbinding::TextureSlotBinding::GraphFrameEndpoint> frameEndpoint;
    // End color for the admitted mix(constant, constant, clamp(time, 0, 1))
    // checkpoint. The built-in GPU program performs the mix per draw.
    std::array<float, 3> timeMixEndColor {};
    surfacematerial::SurfacePbrParameterBlock parameters {};
    std::uint8_t diffractionPathCount = 0;
    std::uint8_t diffractionMaximumBounceDepth = 0;
    std::array<diffractionmaterial::PhysicalDiffractionGpuLightingPath,
               diffractionmaterial::kMaximumLightingPaths> diffractionPaths {};
    std::shared_ptr<const diffractionmaterial::AdmittedLightingPlan>
        diffractionLightingAdmission;
    // Product-only spatial coverage. Zero means the ordinary full-surface
    // diffraction material. Values 1..3 match diffractionmaterial::PatternMask.
    std::uint8_t diffractionPatternMask = 0;
    float diffractionMaskCoverage = 1.0f;
    std::string diffractionProductDigest;
    std::array<diffractionmaterial::GpuFloat4, 4> diffractionFoilField {};
    std::uint32_t diffractionFoilMaximumEvaluations = 0;
    std::array<diffractionmaterial::GpuFloat4, 5> diffractionOccupancyRectangles {};
    std::uint8_t diffractionOccupancyRectangleCount = 0;
    // Additional exact object bindings using this same built-in Surface shader.
    // Never nested; a draw selects by stable SceneObjectId, not array position.
    std::vector<std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>> objectPrograms;
};

inline bool validNativeFixtureSurfaceParameters (
    const surfacematerial::SurfacePbrParameterBlock& parameters) noexcept
{
    const auto finite = [] (const auto& values)
    {
        return std::all_of (values.begin(), values.end(), [] (const auto value)
        {
            return std::isfinite (value);
        });
    };
    return finite (parameters.baseColorMetallic)
        && finite (parameters.emissionRoughness)
        && finite (parameters.normalOpacity)
        && finite (parameters.transmissionIorClearcoat)
        && parameters.transmissionIorClearcoat[0] >= 0.0f && parameters.transmissionIorClearcoat[0] <= 1.0f
        && parameters.transmissionIorClearcoat[1] >= surfacematerial::kMinimumMaterialIor
        && parameters.transmissionIorClearcoat[1] <= surfacematerial::kMaximumMaterialIor
        && parameters.transmissionIorClearcoat[2] >= 0.0f && parameters.transmissionIorClearcoat[2] <= 1.0f;
}

inline bool nativeSurfaceRequiresBlend(const NativeFixtureSurfaceMaterialProgram* program) noexcept
{
    return program != nullptr && program->kind == NativeFixtureMaterialKind::SurfacePbr
        && (program->parameters.normalOpacity[3] < 1.0f
            || program->parameters.transmissionIorClearcoat[0] > 0.0f
            || program->baseColorSource == NativeFixtureSurfaceMaterialProgram::BaseColorSource::ImportedSrgbTexture
            || program->baseColorSource == NativeFixtureSurfaceMaterialProgram::BaseColorSource::GraphFrameSrgbTexture);
}

inline const NativeFixtureSurfaceMaterialProgram* nativeSurfaceForObject(
    const NativeFixtureSurfaceMaterialProgram* program,
    HarmonicMIDI::grid::SceneObjectId object) noexcept
{
    if (program == nullptr) return nullptr;
    if (program->object == object) return program;
    for (const auto& candidate : program->objectPrograms)
        if (candidate && candidate->object == object) return candidate.get();
    return nullptr;
}

inline bool nativeSurfaceUsesTime(const NativeFixtureSurfaceMaterialProgram& root) noexcept
{
    const auto uses = [](const NativeFixtureSurfaceMaterialProgram& program) {
        return program.vertexProgram || program.baseColorSource == NativeFixtureSurfaceMaterialProgram::BaseColorSource::TimeLinearMix;
    };
    return uses(root) || std::any_of(root.objectPrograms.begin(), root.objectPrograms.end(),
        [&](const auto& program) { return program && uses(*program); });
}

inline bool nativeSurfaceUsesFrame(const NativeFixtureSurfaceMaterialProgram& root) noexcept
{
    const auto uses = [](const NativeFixtureSurfaceMaterialProgram& program) {
        return program.baseColorSource == NativeFixtureSurfaceMaterialProgram::BaseColorSource::GraphFrameSrgbTexture;
    };
    return uses(root) || std::any_of(root.objectPrograms.begin(), root.objectPrograms.end(),
        [&](const auto& program) { return program && uses(*program); });
}

inline std::shared_ptr<const NativeFixtureSurfaceMaterialProgram> snapshotNativeSurfaceProgram(
    const std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>& source)
{
    if (!source) return {};
    auto copy = std::make_shared<NativeFixtureSurfaceMaterialProgram>(*source);
    for (auto& program : copy->objectPrograms)
        if (program) program = std::make_shared<const NativeFixtureSurfaceMaterialProgram>(*program);
    return copy;
}

inline bool validNativeSurfaceObjectPrograms(const NativeFixtureSurfaceMaterialProgram& root,
    const HarmonicMIDI::grid::Visual3DScene& scene, NativeFixtureMaterialBackend backend)
{
    using namespace HarmonicMIDI::grid;
    if (root.objectPrograms.size() >= Visual3DScene::kMaxObjects) return false;
    std::set<std::uint32_t> objects;
    std::set<std::string> programs;
    const auto valid = [&](const NativeFixtureSurfaceMaterialProgram& program) {
        const auto* object = visual3d_detail::findById(scene.objects, scene.objectCount, program.object);
        programs.insert(program.programIdentity);
        return object && object->indexCount > 0 && objects.insert(object->id.value).second && program.backend == backend
            && programs.size() <= surfacematerialbinding::kMaximumPrograms
            && program.layoutVersion == NativeFixtureSurfaceMaterialProgram::kLayoutVersion
            && program.kind == NativeFixtureMaterialKind::SurfacePbr
            && !program.programIdentity.empty() && !program.bindingDigest.empty()
            && program.parameters.identifiers[0] == object->material.value
            && validNativeFixtureSurfaceParameters(program.parameters)
            && static_cast<unsigned>(program.baseColorSource)
                <= static_cast<unsigned>(NativeFixtureSurfaceMaterialProgram::BaseColorSource::GraphFrameSrgbTexture)
            && (root.objectPrograms.empty() || !program.vertexProgram);
    };
    if (!valid(root)) return false;
    for (const auto& program : root.objectPrograms)
        if (!program || !program->objectPrograms.empty() || !valid(*program)) return false;
    return true;
}

// Canonical, versioned receipt for the exact admitted diffraction inputs. The
// fixed-width hexadecimal fields preserve integer and IEEE-754 float bits, so
// no locale, decimal formatting, or caller revision can alias two plans.
inline std::string nativeFixtureDiffractionProgramIdentity (
    const std::string& bindingDigest,
    const diffractionmaterial::AdmittedLightingPlan& admitted,
    std::uint8_t patternMask = 0,
    float maskCoverage = 1.0f,
    std::string_view productDigest = {}, bool graphFrame = false)
{
    std::string result = "diffraction-receipt-v1:";
    const auto appendHex = [&result] (std::uint64_t value, std::size_t digits)
    {
        static constexpr char table[] = "0123456789abcdef";
        for (std::size_t shift = digits; shift > 0; --shift)
            result.push_back(table[(value >> ((shift - 1u) * 4u)) & 0xfu]);
    };
    const auto appendFloat = [&appendHex] (float value)
    {
        std::uint32_t bits = 0;
        static_assert(sizeof(bits) == sizeof(value), "float receipt width changed");
        static_assert(std::numeric_limits<float>::is_iec559,
                      "diffraction receipts require IEEE-754 float identity");
        std::memcpy(&bits, &value, sizeof(bits));
        appendHex(bits, 8);
    };

    appendHex(bindingDigest.size(), 16);
    result.append(bindingDigest);
    if (!productDigest.empty())
    {
        result.append(":product:");
        appendHex(patternMask, 2);
        appendFloat(maskCoverage);
        appendHex(productDigest.size(), 16);
        result.append(productDigest);
    }
    const auto& lighting = admitted.description();
    appendHex(lighting.version, 8);
    if (lighting.version == 2)
    {
        appendFloat(lighting.bouncePlaneHeight);
        appendFloat(lighting.bounceMaximumDistance);
    }
    appendHex(diffractionmaterial::kMaximumLightingPaths, 2);
    appendHex(diffractionmaterial::kMaximumIndirectBounces, 2);
    appendHex(diffractionmaterial::kMaximumSpectralSamples, 4);
    appendFloat(diffractionmaterial::physicalcheckpoint::kMaximumSpectralRadiance);
    appendHex(lighting.pathCount, 2);
    for (std::size_t pathIndex = 0; pathIndex < lighting.pathCount; ++pathIndex)
    {
        const auto& path = lighting.paths[pathIndex];
        appendHex(static_cast<std::uint8_t>(path.kind), 2);
        appendHex(path.bounceDepth, 2);
        for (const auto component : path.incident.direction)
            appendFloat(component);
        float intensity = 0.0f;
        for (const auto radiance : path.incident.radiance)
            intensity = std::max(intensity, radiance);
        appendFloat(intensity);
        for (std::size_t spectralIndex = 0;
             spectralIndex < diffractionmaterial::kMaximumSpectralSamples;
             ++spectralIndex)
        {
            appendFloat(diffractionmaterial::physicalcheckpoint::kWavelengths[spectralIndex]);
            appendFloat(path.incident.radiance[spectralIndex]);
        }
    }
    if (graphFrame) result.append(":graph-frame-srgb-v1");
    return result;
}

// Bound fixture targets and diffraction work before allocating GPU resources
// or submitting a draw. The curated crossed tier uses orders -1...1 at 1080p;
// higher user-authored order counts remain available at smaller frame sizes.
inline constexpr std::uint32_t kMaximumNativeFixtureExtent = 4096;
inline constexpr std::uint64_t kMaximumNativeFixturePixels =
    static_cast<std::uint64_t> (kMaximumNativeFixtureExtent)
        * kMaximumNativeFixtureExtent;
inline constexpr std::uint64_t kMaximumNativeFixtureDiffraction1DLobeEvaluations =
    2'250'000'000ull;
inline constexpr std::uint64_t kMaximumNativeFixtureDiffraction2DLobeEvaluations =
    525'000'000ull;
inline constexpr float kMinimumNativeFixtureDiffractionRmsSlope = 1.0e-4f;

// Material programs cross a trust boundary before backend-owned resources are
// allocated. Validate the packed block itself rather than relying on the IR
// admission that originally produced it.
inline bool validNativeFixtureDiffractionParameters (
    const diffractionmaterial::PhysicalDiffractionGpuParameters& value) noexcept
{
    using namespace diffractionmaterial;
    const auto finiteLane = [] (const GpuFloat4& lane)
    {
        return std::isfinite (lane.x) && std::isfinite (lane.y)
            && std::isfinite (lane.z) && std::isfinite (lane.w);
    };
    if (! finiteLane (value.geometry) || ! finiteLane (value.secondaryGeometry)
        || ! finiteLane (value.microstructure) || ! finiteLane (value.control)
        || ! finiteLane (value.incident) || ! finiteLane (value.coating)
        || ! finiteLane (value.roughness) || ! finiteLane (value.grooveField)
        || ! finiteLane (value.grooveVariation)
        || ! std::all_of (value.spectral.begin(), value.spectral.end(), finiteLane)
        || ! std::all_of (value.spectralZ.begin(), value.spectralZ.end(), finiteLane))
        return false;

    const auto bounded = [] (float candidate, float minimum, float maximum)
    {
        return candidate >= minimum && candidate <= maximum;
    };
    const auto unit2 = [] (float x, float y)
    {
        return std::abs (x * x + y * y - 1.0f) <= 1.0e-4f;
    };
    const auto integerIn = [] (float candidate, float minimum, float maximum)
    {
        return candidate >= minimum && candidate <= maximum
            && std::trunc (candidate) == candidate;
    };

    const auto lattice1D = static_cast<float> (GratingLattice::OneDimensional);
    const auto lattice2D = static_cast<float> (GratingLattice::CrossedTwoDimensional);
    const auto lattice = value.secondaryGeometry.w;
    if (! unit2 (value.geometry.x, value.geometry.y)
        || ! bounded (value.geometry.z, kMinimumGrooveSpacingNanometres,
                      kMaximumGrooveSpacingNanometres)
        || ! bounded (value.geometry.w, kMinimumGrooveDepthNanometres,
                      kMaximumGrooveDepthNanometres)
        || (lattice != lattice1D && lattice != lattice2D))
        return false;
    if (lattice == lattice1D)
    {
        if (value.secondaryGeometry.x != 0.0f
            || value.secondaryGeometry.y != 0.0f
            || value.secondaryGeometry.z != 0.0f)
            return false;
    }
    else if (! unit2 (value.secondaryGeometry.x, value.secondaryGeometry.y)
             || std::abs (value.geometry.x * value.secondaryGeometry.x
                           + value.geometry.y * value.secondaryGeometry.y) > 1.0e-4f
             || ! bounded (value.secondaryGeometry.z,
                            kMinimumGrooveSpacingNanometres,
                            kMaximumGrooveSpacingNanometres))
        return false;

    const auto profile = value.control.y;
    const auto binary = static_cast<float> (GrooveProfile::BinaryRectangular);
    const auto sinusoidal = static_cast<float> (GrooveProfile::Sinusoidal);
    const auto blazed = static_cast<float> (GrooveProfile::BlazedSawtooth);
    if (! bounded (value.microstructure.x, 0.01f, 1.0f)
        || ! bounded (value.microstructure.y, kMinimumSubstrateRefractiveIndex,
                      kMaximumSubstrateRefractiveIndex)
        || ! bounded (value.microstructure.z, 0.0f, kMaximumExtinctionCoefficient)
        || ! integerIn (value.microstructure.w, 1.0f,
                        static_cast<float> (kMaximumDiffractionOrder))
        || ! integerIn (value.control.x, value.microstructure.w,
                        static_cast<float> (kMaximumDiffractionOrder))
        || (profile != binary && profile != sinusoidal && profile != blazed)
        || (profile == sinusoidal
            && (std::abs (value.microstructure.x - 0.5f) > 1.0e-6f
                || 6.28318530717958647692f * value.geometry.w / 380.0f
                    > diffractionmaterial::physicalcheckpoint::kMaximumSinusoidalPhaseArgument))
        || (profile == blazed && lattice == lattice2D
            && value.geometry.z != value.secondaryGeometry.z)
        || value.control.z != 0.0f)
        return false;

    const auto coating = value.control.w;
    const auto uncoated = static_cast<float> (CoatingModel::Uncoated);
    const auto dielectric = static_cast<float> (CoatingModel::IncoherentDielectric);
    if (coating != uncoated && coating != dielectric)
        return false;
    if (coating == uncoated)
    {
        if (value.coating.x != 0.0f || value.coating.y != 0.0f
            || value.coating.z != 0.0f)
            return false;
    }
    else if (! bounded (value.coating.x, 0.1f,
                        kMaximumCoatingThicknessNanometres)
             || ! bounded (value.coating.y, kMinimumSubstrateRefractiveIndex,
                            kMaximumSubstrateRefractiveIndex)
             || ! bounded (value.coating.z, 0.0f, kMaximumExtinctionCoefficient))
        return false;
    if (value.coating.w != 0.0f
        || ! bounded (value.roughness.x, 0.0f, kMaximumRmsHeightNanometres)
        || ! bounded (value.roughness.y, kMinimumNativeFixtureDiffractionRmsSlope,
                      kMaximumRmsSlope)
        || value.roughness.z != 0.0f || value.roughness.w != 0.0f)
        return false;

    const auto directionLength = value.incident.x * value.incident.x
        + value.incident.y * value.incident.y + value.incident.z * value.incident.z;
    if (std::abs (directionLength - 1.0f) > 1.0e-4f
        || value.incident.z <= 1.0e-6f || value.incident.w != 0.0f)
        return false;

    const auto constant = static_cast<float> (GrooveFieldMode::Constant);
    const auto linear = static_cast<float> (GrooveFieldMode::Linear);
    const auto radial = static_cast<float> (GrooveFieldMode::Radial);
    const auto fieldMode = value.grooveField.x;
    if (fieldMode != constant && fieldMode != linear && fieldMode != radial)
        return false;
    if (profile == blazed
        && (value.grooveVariation.z != 0.0f || value.grooveVariation.w != 0.0f))
        return false;
    if (fieldMode == constant)
    {
        if (value.grooveField.y != 0.0f || value.grooveField.z != 0.0f
            || value.grooveField.w != 0.0f || value.grooveVariation.x != 0.0f
            || value.grooveVariation.y != 0.0f || value.grooveVariation.z != 0.0f
            || value.grooveVariation.w != 0.0f)
            return false;
    }
    else
    {
        if (! bounded (value.grooveField.y, 0.0f, 1.0f)
            || ! bounded (value.grooveField.z, 0.0f, 1.0f)
            || ! bounded (value.grooveField.w,
                          -kMaximumFieldOrientationDegreesPerUnit,
                          kMaximumFieldOrientationDegreesPerUnit)
            || ! bounded (value.grooveVariation.z,
                          -kMaximumGrooveSpacingNanometres,
                          kMaximumGrooveSpacingNanometres)
            || ! bounded (value.grooveVariation.w,
                          -kMaximumGrooveSpacingNanometres,
                          kMaximumGrooveSpacingNanometres)
            || (lattice == lattice1D && value.grooveVariation.w != 0.0f)
            || (fieldMode == linear
                && ! unit2 (value.grooveVariation.x, value.grooveVariation.y))
            || (fieldMode == radial
                && (value.grooveVariation.x != 0.0f
                    || value.grooveVariation.y != 0.0f)))
            return false;

        float minimumCoordinate = 0.0f;
        float maximumCoordinate = 0.0f;
        bool first = true;
        for (const auto corner : std::array<std::array<float, 2>, 4> {{
                 {{ 0.0f, 0.0f }}, {{ 1.0f, 0.0f }},
                 {{ 0.0f, 1.0f }}, {{ 1.0f, 1.0f }} }})
        {
            const auto x = corner[0] - value.grooveField.y;
            const auto y = corner[1] - value.grooveField.z;
            const auto coordinate = fieldMode == linear
                ? x * value.grooveVariation.x + y * value.grooveVariation.y
                : std::sqrt (x * x + y * y);
            if (first || coordinate < minimumCoordinate) minimumCoordinate = coordinate;
            if (first || coordinate > maximumCoordinate) maximumCoordinate = coordinate;
            first = false;
        }
        const auto localPeriodBounded = [bounded, minimumCoordinate, maximumCoordinate]
            (float base, float delta)
        {
            return bounded (base + delta * minimumCoordinate,
                            kMinimumGrooveSpacingNanometres,
                            kMaximumGrooveSpacingNanometres)
                && bounded (base + delta * maximumCoordinate,
                            kMinimumGrooveSpacingNanometres,
                            kMaximumGrooveSpacingNanometres);
        };
        if (! localPeriodBounded (value.geometry.z, value.grooveVariation.z)
            || (lattice == lattice2D
                && ! localPeriodBounded (value.secondaryGeometry.z,
                                         value.grooveVariation.w)))
            return false;
    }

    for (std::size_t index = 0; index < kMaximumSpectralSamples; ++index)
    {
        const auto& spectral = value.spectral[index];
        const auto& spectralZ = value.spectralZ[index];
        if (spectral.x != physicalcheckpoint::kWavelengths[index]
            || ! bounded (spectral.y, 0.0f,
                          physicalcheckpoint::kMaximumSpectralRadiance)
            || spectral.z != physicalcheckpoint::kCie1931[index][0]
            || spectral.w != physicalcheckpoint::kCie1931[index][1]
            || spectralZ.x != physicalcheckpoint::kCie1931[index][2]
            || spectralZ.y != physicalcheckpoint::kQuadratureWeights[index]
            || spectralZ.z != 0.0f || spectralZ.w != 0.0f)
            return false;
    }
    return true;
}

// Diffraction programs reuse the fixed surface-material envelope, so every
// field outside the diffraction block is a layout sentinel. Reject a direct
// backend caller that smuggles surface parameters, texture state, or a
// time-varying source into that envelope before any GPU resource is allocated.
inline bool validNativeFixtureDiffractionProgram (
    const NativeFixtureSurfaceMaterialProgram& program,
    NativeFixtureMaterialBackend expectedBackend,
    HarmonicMIDI::grid::SceneObjectId expectedObject) noexcept
{
    using Program = NativeFixtureSurfaceMaterialProgram;
    const auto allZero = [] (const auto& values)
    {
        return std::all_of (values.begin(), values.end(), [] (const auto value)
        {
            return value == 0;
        });
    };
    const auto& surface = program.parameters;
    const bool pathsValid = program.diffractionPathCount > 0
        && program.diffractionPathCount <= diffractionmaterial::kMaximumLightingPaths
        && program.diffractionLightingAdmission != nullptr
        && program.diffractionLightingAdmission->description().pathCount
            == program.diffractionPathCount
        && program.diffractionMaximumBounceDepth
            == (program.diffractionLightingAdmission->hasIndirectBounce()
                    ? diffractionmaterial::kMaximumIndirectBounces : 0)
        && program.diffractionMaximumBounceDepth
            <= diffractionmaterial::kMaximumIndirectBounces
        && std::all_of(program.diffractionPaths.begin(),
                       program.diffractionPaths.begin() + program.diffractionPathCount,
                       [&program] (const auto& path)
                       {
                           const auto& value = path.incidentDirectionAndIntensity;
                           const auto length = value.x * value.x + value.y * value.y
                               + value.z * value.z;
                           const auto kind = path.kindBounceAndReserved[0];
                           const auto depth = path.kindBounceAndReserved[1];
                           const bool kindDepthValid =
                               (kind == static_cast<std::uint32_t>(
                                    diffractionmaterial::LightingPathKind::Direct)
                                || kind == static_cast<std::uint32_t>(
                                    diffractionmaterial::LightingPathKind::Environment))
                                   ? depth == 0
                                   : kind == static_cast<std::uint32_t>(
                                        diffractionmaterial::LightingPathKind::Indirect)
                                     && depth == diffractionmaterial::kMaximumIndirectBounces;
                           return validNativeFixtureDiffractionParameters(path.material)
                               && std::isfinite(length)
                               && std::abs(length - 1.0f) <= 1.0e-4f
                               && (program.diffractionLightingAdmission->description().version == 2
                                   || value.z > 1.0e-6f) && std::isfinite(value.w)
                               && value.w >= 0.0f && value.w <= 16.0f
                               && kindDepthValid
                               && path.kindBounceAndReserved[2]
                                    == program.diffractionPatternMask
                               && [&]
                               {
                                   std::uint32_t expectedCoverage = 0;
                                   if (program.diffractionPatternMask != 0)
                                       std::memcpy(&expectedCoverage,
                                           &program.diffractionMaskCoverage,
                                           sizeof(expectedCoverage));
                                   return path.kindBounceAndReserved[3] == expectedCoverage;
                               }();
                       });
    bool receiptValid = pathsValid;
    for (std::size_t index = 0; receiptValid && index < program.diffractionPathCount;
         ++index)
    {
        const auto& path = program.diffractionPaths[index];
        const auto& admittedPath
            = program.diffractionLightingAdmission->description().paths[index];
        const auto& direction = admittedPath.incident.direction;
        float intensity = 0.0f;
        for (const auto radiance : admittedPath.incident.radiance)
            intensity = std::max(intensity, radiance);
        receiptValid = admittedPath.kind
                == static_cast<diffractionmaterial::LightingPathKind>(
                    path.kindBounceAndReserved[0])
            && admittedPath.bounceDepth == path.kindBounceAndReserved[1]
            && direction[0] == path.incidentDirectionAndIntensity.x
            && direction[1] == path.incidentDirectionAndIntensity.y
            && direction[2] == path.incidentDirectionAndIntensity.z
            && intensity == path.incidentDirectionAndIntensity.w
            && (program.diffractionLightingAdmission->description().version == 2
                ? (path.material.incident.x == 0.0f && path.material.incident.y == 0.0f
                   && path.material.incident.z == 1.0f)
                : (direction[0] == path.material.incident.x
                   && direction[1] == path.material.incident.y
                   && direction[2] == path.material.incident.z));
        for (std::size_t spectralIndex = 0;
             receiptValid
                 && spectralIndex < diffractionmaterial::kMaximumSpectralSamples;
             ++spectralIndex)
        {
            const auto radiance = admittedPath.incident.radiance[spectralIndex];
            const auto expectedRadiance = intensity > 0.0f
                ? radiance / intensity : radiance;
            receiptValid = path.material.spectral[spectralIndex].y
                == expectedRadiance;
        }
    }
    const bool maskValid = std::isfinite(program.diffractionMaskCoverage)
        && program.diffractionMaskCoverage >= 0.0f
        && program.diffractionMaskCoverage <= 1.0f
        && ((program.diffractionPatternMask == 0
                && program.diffractionMaskCoverage == 1.0f
                && program.diffractionProductDigest.empty())
            || (program.diffractionPatternMask >= 1
                && program.diffractionPatternMask <= 3
                && !program.diffractionProductDigest.empty()));
    const bool identityValid = pathsValid && maskValid
        && program.programIdentity == nativeFixtureDiffractionProgramIdentity(
               program.bindingDigest, *program.diffractionLightingAdmission,
               program.diffractionPatternMask, program.diffractionMaskCoverage,
               program.diffractionProductDigest,
               program.baseColorSource == Program::BaseColorSource::GraphFrameSrgbTexture);
    return expectedBackend != NativeFixtureMaterialBackend::Invalid
        && expectedObject.isValid()
        && program.layoutVersion == Program::kLayoutVersion
        && program.backend == expectedBackend
        && program.kind == NativeFixtureMaterialKind::DiffractionReflective
        && program.object == expectedObject
        && ! program.bindingDigest.empty()
        && identityValid
        && (program.baseColorSource == Program::BaseColorSource::ConstantLinear
            || program.baseColorSource == Program::BaseColorSource::GraphFrameSrgbTexture)
        && ! program.importedBaseColorTexture.has_value()
        && allZero (program.timeMixEndColor)
        && allZero (surface.baseColorMetallic)
        && allZero (surface.emissionRoughness)
        && allZero (surface.normalOpacity)
        && allZero (surface.transmissionIorClearcoat)
        && allZero (surface.identifiers)
        && pathsValid && receiptValid;
}

inline bool nativeFixtureDimensionsWithinBounds (std::uint32_t width,
                                                 std::uint32_t height) noexcept
{
    return width > 0 && height > 0
        && width <= kMaximumNativeFixtureExtent
        && height <= kMaximumNativeFixtureExtent
        && static_cast<std::uint64_t> (width) * height <= kMaximumNativeFixturePixels;
}

inline std::uint64_t nativeFixtureDiffractionLobeEvaluations (
    const NativeFixtureSurfaceMaterialProgram& material,
    std::uint32_t width,
    std::uint32_t height) noexcept
{
    if (material.kind != NativeFixtureMaterialKind::DiffractionReflective)
        return 0;

    if (material.diffractionPathCount == 0
        || material.diffractionPathCount > diffractionmaterial::kMaximumLightingPaths)
        return std::numeric_limits<std::uint64_t>::max();
    const auto& diffraction = material.diffractionPaths[0].material;
    const auto firstOrderValue = diffraction.microstructure.w;
    const auto lastOrderValue = diffraction.control.x;
    const auto latticeValue = diffraction.secondaryGeometry.w;
    const auto oneDimensional = static_cast<float> (
        diffractionmaterial::GratingLattice::OneDimensional);
    const auto crossedTwoDimensional = static_cast<float> (
        diffractionmaterial::GratingLattice::CrossedTwoDimensional);
    if (!std::isfinite (firstOrderValue)
        || !std::isfinite (lastOrderValue)
        || !std::isfinite (latticeValue)
        || firstOrderValue < 1.0f
        || lastOrderValue < firstOrderValue
        || lastOrderValue > static_cast<float> (
            diffractionmaterial::kMaximumDiffractionOrder)
        || std::trunc (firstOrderValue) != firstOrderValue
        || std::trunc (lastOrderValue) != lastOrderValue
        || (latticeValue != oneDimensional
            && latticeValue != crossedTwoDimensional))
        return std::numeric_limits<std::uint64_t>::max();

    const auto firstOrder = static_cast<std::uint64_t> (firstOrderValue);
    const auto lastOrder = static_cast<std::uint64_t> (lastOrderValue);
    constexpr auto wavelengthCount = static_cast<std::uint64_t> (
        diffractionmaterial::kMaximumSpectralSamples);
    if (wavelengthCount == 0)
        return std::numeric_limits<std::uint64_t>::max();

    const auto signedOrderCount = 1ull + 2ull * (lastOrder - firstOrder + 1ull);
    const auto lobeCount = latticeValue == crossedTwoDimensional
        ? signedOrderCount * signedOrderCount : signedOrderCount;
    const auto pixels = static_cast<std::uint64_t> (width) * height;
    const auto pathCount = material.diffractionLightingAdmission
            && material.diffractionLightingAdmission->description().version == 2
        ? 11ull : static_cast<std::uint64_t>(material.diffractionPathCount);
    if (pixels == 0 || wavelengthCount > std::numeric_limits<std::uint64_t>::max() / lobeCount
        || pathCount > std::numeric_limits<std::uint64_t>::max()
            / (wavelengthCount * lobeCount)
        || pixels > std::numeric_limits<std::uint64_t>::max()
            / (pathCount * wavelengthCount * lobeCount))
        return std::numeric_limits<std::uint64_t>::max();
    return pixels * pathCount * wavelengthCount * lobeCount;
}

inline bool nativeFixtureDiffractionWorkWithinBudget (
    const NativeFixtureSurfaceMaterialProgram& material,
    std::uint32_t width,
    std::uint32_t height) noexcept
{
    const auto estimate = nativeFixtureDiffractionLobeEvaluations (
        material, width, height);
    const auto budget = material.diffractionPaths[0].material.secondaryGeometry.w
        == static_cast<float> (
            diffractionmaterial::GratingLattice::CrossedTwoDimensional)
        ? kMaximumNativeFixtureDiffraction2DLobeEvaluations
        : kMaximumNativeFixtureDiffraction1DLobeEvaluations;
    if (estimate > budget)
        return false;
    if (material.diffractionFoilMaximumEvaluations == 0)
        return true;
    diffractivefoil::EvaluationSchedule schedule;
    return diffractivefoil::makeEvaluationSchedule(
        material.diffractionFoilMaximumEvaluations, width, height,
        material.diffractionPathCount, schedule);
}

inline bool nativeFixtureSpatialFoilEvaluationSchedule (
    const NativeFixtureSurfaceMaterialProgram& material,
    std::uint32_t width, std::uint32_t height,
    diffractivefoil::EvaluationSchedule& schedule) noexcept
{
    schedule = {};
    if (material.kind != NativeFixtureMaterialKind::DiffractionReflective
        || material.diffractionFoilMaximumEvaluations == 0
        || material.diffractionPathCount == 0
        || material.diffractionPathCount > diffractionmaterial::kMaximumLightingPaths)
        return false;
    return diffractivefoil::makeEvaluationSchedule(
        material.diffractionFoilMaximumEvaluations, width, height,
        material.diffractionPathCount, schedule);
}

// Immutable backend-owned vertex, index, texture, sampler, shader, and pipeline
// resources for one exact scene snapshot. The matching backend may retain these
// across preview/export frames; releasing the last shared owner deletes them.
class NativeFixtureSceneResources
{
public:
    virtual ~NativeFixtureSceneResources() = default;
    virtual const std::string& backend() const noexcept = 0;
};

struct NativeFixtureScenePreparation
{
    bool prepared = false;
    std::shared_ptr<const NativeFixtureSceneResources> resources;
    NativeFixtureSceneStats stats {};
    std::string error;
};

// Complete immutable contract for a texture view borrowed across the native
// renderer/compositor boundary. Handles alone are never sufficient: backends
// verify this descriptor against the live object immediately before use.
enum class NativeTextureViewKind : std::uint8_t { Invalid = 0, Texture2D };
enum class NativeTexturePixelFormat : std::uint8_t
{
    Invalid = 0, Rgba8Unorm, Bgra8Unorm, R32Float, Rgba16Float, R8Unorm, R32Uint, Rg16Float, Rgba32Float
};
// Raster row at texture coordinate V=0. Decoded images and compositor targets
// are top-first; native GL scene attachments retain GL's bottom-first rows.
enum class NativeTextureRowOrder : std::uint8_t { TopFirst = 0, BottomFirst };
struct NativeTextureViewDescriptor final
{
    std::string backend;
    NativeTextureViewKind viewKind = NativeTextureViewKind::Invalid;
    NativeTexturePixelFormat format = NativeTexturePixelFormat::Invalid;
    std::uintptr_t imageHandle = 0;
    std::uintptr_t textureViewHandle = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t sampleCount = 0;
    bool sampledBinding = false;
    std::uintptr_t deviceOrContextIdentity = 0;
    // Identifies the backend resource lifetime: a fixture's static preparation
    // or an SDF output allocation. Fixture generations remain stable across
    // repeated renders from one preparation.
    std::uint64_t rendererGeneration = 0;
    colortransform::ColorSpace colorSpace = colortransform::ColorSpace::Unspecified;
    colortransform::TransferFunction transfer = colortransform::TransferFunction::Unspecified;
    NativeTextureRowOrder rowOrder = NativeTextureRowOrder::TopFirst;

    bool complete() const noexcept
    {
        return !backend.empty() && viewKind == NativeTextureViewKind::Texture2D
            && format != NativeTexturePixelFormat::Invalid && imageHandle != 0
            && textureViewHandle != 0 && width != 0 && height != 0
            && sampleCount == 1 && sampledBinding && deviceOrContextIdentity != 0
            && rendererGeneration != 0
            && (rowOrder == NativeTextureRowOrder::TopFirst
                || rowOrder == NativeTextureRowOrder::BottomFirst);
    }
};

inline bool isLinearSceneColor(const NativeTextureViewDescriptor& descriptor) noexcept
{
    return descriptor.complete() && descriptor.format == NativeTexturePixelFormat::Rgba16Float
        && descriptor.colorSpace == colortransform::ColorSpace::LinearSRGB
        && descriptor.transfer == colortransform::TransferFunction::Linear;
}

// Owned tight rows, top row first, native scalar byte order. No display transform.
// BGRA is normalized to RGBA by readRawPass; float16 and uint32 bits are preserved.
struct NativeRawPassPixels final
{
    NativeTexturePixelFormat format = NativeTexturePixelFormat::Invalid;
    std::uint32_t width = 0, height = 0;
    std::vector<std::uint8_t> bytes;
};
inline unsigned rawPassChannels(NativeTexturePixelFormat format) noexcept
{
    switch (format)
    {
        case NativeTexturePixelFormat::Rgba8Unorm:
        case NativeTexturePixelFormat::Bgra8Unorm:
        case NativeTexturePixelFormat::Rgba16Float:
        case NativeTexturePixelFormat::Rgba32Float: return 4;
        case NativeTexturePixelFormat::Rg16Float: return 2;
        case NativeTexturePixelFormat::R32Float:
        case NativeTexturePixelFormat::R32Uint:
        case NativeTexturePixelFormat::R8Unorm: return 1;
        default: return 0;
    }
}
inline unsigned rawPassScalarBytes(NativeTexturePixelFormat format) noexcept
{
    switch (format)
    {
        case NativeTexturePixelFormat::Rgba8Unorm:
        case NativeTexturePixelFormat::Bgra8Unorm:
        case NativeTexturePixelFormat::R8Unorm: return 1;
        case NativeTexturePixelFormat::Rgba16Float:
        case NativeTexturePixelFormat::Rg16Float: return 2;
        case NativeTexturePixelFormat::R32Float:
        case NativeTexturePixelFormat::R32Uint:
        case NativeTexturePixelFormat::Rgba32Float: return 4;
        default: return 0;
    }
}
inline bool rawPassFormatMatches(renderpassoutput::Output output, NativeTexturePixelFormat format) noexcept
{
    using O = renderpassoutput::Output;
    using F = NativeTexturePixelFormat;
    switch (output)
    {
        case O::Color: return format == F::Rgba8Unorm || format == F::Bgra8Unorm || format == F::Rgba16Float;
        case O::Depth: return format == F::R32Float;
        case O::Normal: case O::Emission: return format == F::Rgba16Float;
        case O::Motion: return format == F::Rg16Float;
        case O::Mask: return format == F::R8Unorm;
        case O::MaterialId: case O::ObjectId: return format == F::R32Uint;
        default: return false;
    }
}

// The frame owns every backend resource created for one submitted fixture draw.
// Handles are backend-local and may only be consumed by the matching native
// compositor. Releasing the last shared owner deletes the resources.
class NativeFixtureSceneFrame
{
public:
    virtual ~NativeFixtureSceneFrame() = default;
    virtual const std::string& backend() const noexcept = 0;
    virtual std::uint32_t width() const noexcept = 0;
    virtual std::uint32_t height() const noexcept = 0;
    virtual std::uintptr_t colorImageHandle() const noexcept = 0;
    virtual std::uintptr_t colorTextureViewHandle() const noexcept = 0;
    virtual std::uintptr_t depthImageHandle() const noexcept { return 0; }
    virtual std::uintptr_t depthTextureViewHandle() const noexcept { return 0; }
    virtual std::uintptr_t nativeResourceCacheIdentity() const noexcept { return 0; }
    virtual NativeTextureViewDescriptor colorTextureDescriptor() const noexcept { return {}; }
    virtual NativeTextureViewDescriptor depthTextureDescriptor() const noexcept { return {}; }
    virtual NativeTextureViewDescriptor passTextureDescriptor(renderpassoutput::Output output) const noexcept
    {
        return output == renderpassoutput::Output::Color ? colorTextureDescriptor()
            : output == renderpassoutput::Output::Depth ? depthTextureDescriptor()
            : NativeTextureViewDescriptor {};
    }
    virtual bool readRawPass(renderpassoutput::Output, NativeRawPassPixels& pixels, std::string& error) const
    {
        pixels = {};
        error = "raw pass readback is unavailable for this native frame";
        return false;
    }
};

// Material Frame inputs borrow the same immutable native-frame owner as the
// compositor. A handle without this owner, or an attachment with different
// metadata, cannot serve as an authored Frame<Image> dependency.
inline bool validMaterialFrameTexture(
    const std::shared_ptr<const NativeFixtureSceneFrame>& frame) noexcept
{
    if (!frame) return false;
    const auto descriptor = frame->colorTextureDescriptor();
    return descriptor.complete() && nativeFixtureDimensionsWithinBounds(descriptor.width, descriptor.height)
        && (descriptor.backend == "opengl" || descriptor.backend == "metal")
        && descriptor.backend == frame->backend()
        && (descriptor.format == NativeTexturePixelFormat::Rgba8Unorm
            || (descriptor.backend == "metal"
                && descriptor.format == NativeTexturePixelFormat::Bgra8Unorm))
        && descriptor.imageHandle == frame->colorImageHandle()
        && descriptor.textureViewHandle == frame->colorTextureViewHandle()
        && descriptor.width == frame->width() && descriptor.height == frame->height();
}

// An owned texture copy with a proved colour interpretation. Native handles and
// lifetime stay with the original immutable owner; the declaration is not a new
// image identity and cannot turn a borrowed handle into an owned Frame.
class DeclaredSrgbMaterialFrame final : public NativeFixtureSceneFrame
{
public:
    explicit DeclaredSrgbMaterialFrame(std::shared_ptr<const NativeFixtureSceneFrame> owner) : owner_(std::move(owner)) {}
    const std::string& backend() const noexcept override { return owner_->backend(); }
    std::uint32_t width() const noexcept override { return owner_->width(); }
    std::uint32_t height() const noexcept override { return owner_->height(); }
    std::uintptr_t colorImageHandle() const noexcept override { return owner_->colorImageHandle(); }
    std::uintptr_t colorTextureViewHandle() const noexcept override { return owner_->colorTextureViewHandle(); }
    NativeTextureViewDescriptor colorTextureDescriptor() const noexcept override {
        auto descriptor = owner_->colorTextureDescriptor();
        descriptor.colorSpace = colortransform::ColorSpace::SRGB;
        descriptor.transfer = colortransform::TransferFunction::SRGB; return descriptor;
    }
private:
    std::shared_ptr<const NativeFixtureSceneFrame> owner_;
};
inline std::shared_ptr<const NativeFixtureSceneFrame> declaredSrgbMaterialFrame(
    std::shared_ptr<const NativeFixtureSceneFrame> owner)
{ return validMaterialFrameTexture(owner) ? std::make_shared<const DeclaredSrgbMaterialFrame>(std::move(owner)) : nullptr; }
inline bool materialFrameIsSrgb(const NativeTextureViewDescriptor& descriptor)
{
    return descriptor.complete() && descriptor.colorSpace == colortransform::ColorSpace::SRGB
        && descriptor.transfer == colortransform::TransferFunction::SRGB;
}
inline bool materialFrameIsSrgb(const std::shared_ptr<const NativeFixtureSceneFrame>& frame)
{
    return validMaterialFrameTexture(frame) && materialFrameIsSrgb(frame->colorTextureDescriptor());
}

inline bool materialFrameIsBottomFirst(const std::shared_ptr<const NativeFixtureSceneFrame>& frame)
{
    return frame && frame->colorTextureDescriptor().rowOrder == NativeTextureRowOrder::BottomFirst;
}

struct NativeFixtureSceneSubmission
{
    bool rendered = false;
    std::shared_ptr<const NativeFixtureSceneFrame> frame;
    NativeFixtureSceneStats stats {};
    std::string error;
};

// Per-draw values are copied into the submission. They never participate in
// resource or pipeline identity, so changing time cannot trigger compilation.
// One immutable per-frame contract feeds static and deformed imported scenes.
// The deformation backend consumes morphWeight; the fixture draw consumes the
// remaining fields, including the exact admitted Block C frame and mapping.
struct NativeSceneMotionSample final
{
    std::array<float, 3> objectTranslationOffset {};
    std::array<float, 3> objectRotationDegrees {};
    float objectScale = 1.0f;
    std::array<float, 3> cameraTranslationOffset {};
    std::optional<HarmonicMIDI::grid::SceneCameraRecord> cameraOverride;
};

struct NativeImportedSceneRuntimeInputs final
{
    // Explicit HDR export working representation. Ordinary preview stays SDR.
    bool linearColor = false;
    float timeSeconds = 0.0f;
    // Per-evaluation lease, never a static scene/material/cache identity. The
    // native draw must validate the device/context before sampling this view.
    std::shared_ptr<const NativeFixtureSceneFrame> materialFrameTexture;
    std::map<surfacematerialbinding::TextureSlotBinding::GraphFrameEndpoint,
        std::shared_ptr<const NativeFixtureSceneFrame>> materialFrameTextures;
    // Evaluated by FixtureSceneRenderer against its immutable authored binding.
    // Only uniforms change; the original mesh, textures and pipelines stay owned.
    diffractionmaterialbinding::RuntimeParameters diffractionParameters;
    std::shared_ptr<const NativeFixtureSurfaceMaterialProgram> diffractionSourceProgram;
    std::shared_ptr<const NativeFixtureSurfaceMaterialProgram> diffractionEvaluatedProgram;
    std::array<float, 64> vertexSpectrum {};
    float morphWeight = 1.0f;
    std::uint64_t objectNodeStableId = 0;
    std::array<float, 3> objectTranslationOffset {};
    std::array<float, 3> objectRotationDegrees {};
    float objectScale = 1.0f;
    std::array<float, 3> cameraTranslationOffset {};
    std::optional<HarmonicMIDI::grid::SceneCameraRecord> cameraOverride;
    std::optional<HarmonicMIDI::grid::SceneLightRecord> lightOverride;
    // Sampled imported records are subordinate to explicit graph overrides.
    std::optional<HarmonicMIDI::grid::SceneCameraRecord> animatedCamera;
    std::vector<HarmonicMIDI::grid::SceneLightRecord> animatedLights;
    float emissionGain = 1.0f;
    std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> canonicalBlockCFrame;
    std::optional<visualnoteinstancing::Mapping> noteInstanceMapping;
    renderpassoutput::Output imageOutput = renderpassoutput::Output::Color;
    std::uint32_t rawExportMask = 0;
    std::optional<renderpasscomposite::Parameters> passComposite;
    std::optional<renderpasscomposite::Program> passProgram;
    // Previous adjacent timeline sample. Absence produces exactly zero motion.
    // Raw RG16F stores current minus previous pixel position, positive Y upwards.
    std::optional<NativeSceneMotionSample> previousMotion;
};

inline renderpasscomposite::Program scenePassProgram(const NativeImportedSceneRuntimeInputs& inputs)
{
    if (inputs.passProgram) return *inputs.passProgram;
    renderpasscomposite::Program result;
    if (inputs.passComposite) { result.count = 1; result.steps[0].parameters = *inputs.passComposite; }
    return result;
}
inline bool sceneUsesMotionPass(const NativeImportedSceneRuntimeInputs& inputs)
{
    return inputs.imageOutput == renderpassoutput::Output::Motion
        || render3dimage::exports(inputs.rawExportMask, renderpassoutput::Output::Motion)
        || renderpasscomposite::usesMotion(scenePassProgram(inputs));
}

inline NativeSceneMotionSample sceneMotionSample(const NativeImportedSceneRuntimeInputs& inputs)
{
    return { inputs.objectTranslationOffset, inputs.objectRotationDegrees,
        inputs.objectScale, inputs.cameraTranslationOffset, inputs.cameraOverride };
}

// Owned by the existing per-clip scene owner. A failed render never commits a
// sample. First frame, seek, loop, rate/extent change and owner reset yield zero.
struct NativeSceneMotionHistory final
{
    std::optional<NativeSceneMotionSample> current, previous;
    std::int64_t frame = 0;
    std::uint32_t numerator = 0, denominator = 0, width = 0, height = 0;
    std::optional<NativeSceneMotionSample> predecessor(std::int64_t next,
        std::uint32_t num, std::uint32_t den, std::uint32_t w, std::uint32_t h) const
    {
        if (!current || num != numerator || den != denominator || w != width || h != height)
            return {};
        if (next == frame) return previous;
        return next > frame && next - frame == 1 ? current : std::nullopt;
    }
    void commit(std::int64_t next, std::uint32_t num, std::uint32_t den,
        std::uint32_t w, std::uint32_t h, const NativeImportedSceneRuntimeInputs& inputs)
    {
        previous = predecessor(next, num, den, w, h);
        current = sceneMotionSample(inputs);
        frame = next; numerator = num; denominator = den; width = w; height = h;
    }
};

using NativeFixtureSceneRuntimeInputs = NativeImportedSceneRuntimeInputs;
using NativeDeformationRuntimeInputs = NativeImportedSceneRuntimeInputs;

inline bool runtimeTargetsObject(const NativeImportedSceneRuntimeInputs& inputs,
                                const HarmonicMIDI::grid::SceneObjectRecord& object) noexcept
{
    return inputs.objectNodeStableId == 0
        || (object.id.value - 1u) / 65536u + 1u == inputs.objectNodeStableId;
}

inline bool resolveDiffractionRuntimeProgram(
    const std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>& authored,
    const std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>& authoredSource,
    const NativeImportedSceneRuntimeInputs& inputs,
    std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>& resolved, std::string& error)
{
    resolved = authored;
    if (!inputs.diffractionEvaluatedProgram && !inputs.diffractionSourceProgram)
    {
        if (inputs.diffractionParameters.empty()) return true;
        error = "Diffraction scalar parameters require frame-local material admission";
        return false;
    }
    const auto& value = inputs.diffractionEvaluatedProgram;
    // Preparation snapshots caller-owned values. The retained source owner
    // authorizes this evaluation; only the snapshot defines its allowed layout.
    if (!authored || !authoredSource || inputs.diffractionSourceProgram != authoredSource || !value
        || authored->kind != NativeFixtureMaterialKind::DiffractionReflective
        || !authored->diffractionLightingAdmission
        || !validNativeFixtureDiffractionProgram(*value, authored->backend, authored->object)
        || value->baseColorSource != authored->baseColorSource
        || value->diffractionPathCount != authored->diffractionPathCount
        || value->diffractionFoilMaximumEvaluations != authored->diffractionFoilMaximumEvaluations
        || value->diffractionProductDigest != authored->diffractionProductDigest
        || value->diffractionPatternMask != authored->diffractionPatternMask
        || value->diffractionOccupancyRectangleCount != authored->diffractionOccupancyRectangleCount
        || value->diffractionLightingAdmission->description().version
            != authored->diffractionLightingAdmission->description().version)
    { error = "Diffraction evaluation requires its exact authored program and unchanged draw layout"; return false; }
    for (std::size_t i = 0; i < value->diffractionPathCount; ++i)
    {
        const auto& a = authored->diffractionPaths[i].material;
        const auto& b = value->diffractionPaths[i].material;
        if (a.secondaryGeometry.w != b.secondaryGeometry.w || a.control.x != b.control.x
            || a.control.y != b.control.y || a.control.w != b.control.w
            || a.microstructure.w != b.microstructure.w || a.grooveField.x != b.grooveField.x)
        { error = "Diffraction runtime profile, lattice, coating, field and spectral schedule must remain authored"; return false; }
    }
    resolved = value;
    return true;
}

inline std::shared_ptr<const NativeFixtureSceneFrame> materialFrameForProgram(
    const NativeFixtureSurfaceMaterialProgram* material, const NativeImportedSceneRuntimeInputs& inputs)
{
    if (!material || material->baseColorSource != NativeFixtureSurfaceMaterialProgram::BaseColorSource::GraphFrameSrgbTexture)
        return {};
    if (inputs.materialFrameTextures.empty()) return inputs.materialFrameTexture;
    if (!material->frameEndpoint) return {};
    const auto found = inputs.materialFrameTextures.find(*material->frameEndpoint);
    return found == inputs.materialFrameTextures.end() ? nullptr : found->second;
}

inline bool validateMaterialFrameForDraw(
    const std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>& material,
    const NativeImportedSceneRuntimeInputs& inputs, const char* backend,
    std::uintptr_t deviceOrContext, std::string& error)
{
    const auto usesFrame = [](const auto& program) { return program && program->baseColorSource
        == NativeFixtureSurfaceMaterialProgram::BaseColorSource::GraphFrameSrgbTexture; };
    const bool expected = material && nativeSurfaceUsesFrame(*material);
    if ((inputs.materialFrameTextures.empty() && expected != static_cast<bool>(inputs.materialFrameTexture))
        || (!inputs.materialFrameTextures.empty() && inputs.materialFrameTexture)
        || inputs.materialFrameTextures.size() > surfacematerialbinding::kMaximumPrograms)
    { error = "Native material Frame draw requires its exact owned texture"; return false; }
    const auto validate = [&](const auto& frame) {
        if (!validMaterialFrameTexture(frame))
        { error = "Native material Frame draw has no complete owned texture"; return false; }
        const auto descriptor = frame->colorTextureDescriptor();
        if (descriptor.backend != backend || descriptor.deviceOrContextIdentity != deviceOrContext)
        { error = "Native material Frame texture belongs to another device or context"; return false; }
        if (inputs.linearColor && (descriptor.colorSpace != colortransform::ColorSpace::SRGB
            || descriptor.transfer != colortransform::TransferFunction::SRGB))
        { error = "Linear Scene3D requires an explicitly sRGB material Frame; video transfer is not inferred"; return false; }
        return true;
    };
    if (inputs.materialFrameTexture && !validate(inputs.materialFrameTexture)) return false;
    for (const auto& entry : inputs.materialFrameTextures)
        if (entry.first.node <= 0 || entry.first.port < 0)
        { error = "Native material Frame map has an invalid graph endpoint"; return false; }
        else if (!validate(entry.second)) return false;
    if (material) {
        if (usesFrame(material) && !materialFrameForProgram(material.get(),inputs))
        { error = "Native material Frame endpoint has no scheduled image"; return false; }
        for (const auto& program : material->objectPrograms)
            if (usesFrame(program) && !materialFrameForProgram(program.get(),inputs))
            { error = "Native object material Frame endpoint has no scheduled image"; return false; }
    }
    return true;
}

struct NativeNoteInstanceBatch final
{
    std::array<std::array<float, 4>, visualnoteinstancing::kMaximumInstances> transforms {};
    std::array<float, visualnoteinstancing::kMaximumInstances> appearanceValues {};
    std::array<float, 4> appearanceLow { 1.0f, 1.0f, 1.0f, 0.0f };
    std::array<float, 4> appearanceHigh { 1.0f, 1.0f, 1.0f, 0.0f };
    float meshScale = 1.0f;
    std::array<std::int64_t, visualnoteinstancing::kMaximumInstances> identities {};
    std::array<std::uint16_t, visualnoteinstancing::kMaximumInstances> canonicalRows {};
    std::size_t count = 0;
    bool admitted = false;
};

inline NativeNoteInstanceBatch prepareNativeNoteInstances(
    const NativeFixtureSceneRuntimeInputs& inputs) noexcept
{
    NativeNoteInstanceBatch batch;
    if (!inputs.canonicalBlockCFrame || !inputs.noteInstanceMapping)
    {
        batch.transforms[0] = { 0.0f, 0.0f, 0.0f, 1.0f };
        batch.count = 1;
        return batch;
    }
    const auto& frame = *inputs.canonicalBlockCFrame;
    const auto& mapping = *inputs.noteInstanceMapping;
    batch.admitted = true;
    batch.appearanceLow = mapping.appearanceLow;
    batch.appearanceHigh = mapping.appearanceHigh;
    batch.meshScale = mapping.meshScale;
    const auto axisValue = [&frame](visualnoteinstancing::MappingAxis axis,
                                    std::size_t row) noexcept
    {
        const auto texel = [&frame, row](std::size_t column, std::size_t channel)
        {
            return frame.noteTextureValues[(row * arbitblockc::kTexelsPerNote + column) * 4u
                                           + channel];
        };
        switch (axis)
        {
            case visualnoteinstancing::MappingAxis::Onset: return -texel(0, 2);
            case visualnoteinstancing::MappingAxis::LogFrequency:
                return std::log2(std::max(texel(1, 0), 0.000001f)
                                 / frame.rootFrequencyHz);
            case visualnoteinstancing::MappingAxis::Velocity: return texel(0, 1);
            case visualnoteinstancing::MappingAxis::Duration:
                return texel(0, 2) + texel(0, 3);
            case visualnoteinstancing::MappingAxis::Track: return texel(1, 2);
        }
        return 0.0f;
    };
    const auto rowLimit = std::min<std::size_t>(
        static_cast<std::size_t>(frame.noteRows()), visualnoteinstancing::kMaximumInstances);
    for (std::size_t row = 0; row < rowLimit; ++row)
    {
        const auto identity = frame.noteIdentities[row];
        if (identity == 0)
            continue;
        if (frame.notationRows[row].muted && !mapping.includeMuted)
            continue;
        const auto index = batch.count++;
        batch.transforms[index] = {
            axisValue(mapping.x, row) * mapping.xScale,
            axisValue(mapping.y, row) * mapping.yScale,
            axisValue(mapping.z, row) * mapping.zScale,
            mapping.meshScale
        };
        batch.identities[index] = identity;
        batch.canonicalRows[index] = static_cast<std::uint16_t>(row);
        batch.appearanceValues[index] = visualnoteinstancing::appearanceValue(
            mapping, identity, axisValue(visualnoteinstancing::MappingAxis::Velocity, row));
    }
    return batch;
}

inline auto noteInstanceShaderTransforms(const NativeNoteInstanceBatch& batch) noexcept
{
    auto packed = batch.transforms;
    // Scale is shared by the mapping. Its uniform frees w for the note's
    // appearance value without adding another 128-element shader array.
    for (std::size_t index = 0; index < batch.count; ++index)
        packed[index][3] = batch.appearanceValues[index];
    return packed;
}

inline HarmonicMIDI::grid::SceneQuaternion multiplySceneQuaternion(
    const HarmonicMIDI::grid::SceneQuaternion& left,
    const HarmonicMIDI::grid::SceneQuaternion& right) noexcept
{
    return {
        left.w * right.x + left.x * right.w + left.y * right.z - left.z * right.y,
        left.w * right.y - left.x * right.z + left.y * right.w + left.z * right.x,
        left.w * right.z + left.x * right.y - left.y * right.x + left.z * right.w,
        left.w * right.w - left.x * right.x - left.y * right.y - left.z * right.z
    };
}

inline HarmonicMIDI::grid::SceneQuaternion runtimeEulerQuaternion(
    const std::array<float, 3>& degrees) noexcept
{
    constexpr float radiansPerDegree = 0.01745329251994329577f;
    const auto halfX = degrees[0] * radiansPerDegree * 0.5f;
    const auto halfY = degrees[1] * radiansPerDegree * 0.5f;
    const auto halfZ = degrees[2] * radiansPerDegree * 0.5f;
    const HarmonicMIDI::grid::SceneQuaternion x { std::sin(halfX), 0.0f, 0.0f,
                                                   std::cos(halfX) };
    const HarmonicMIDI::grid::SceneQuaternion y { 0.0f, std::sin(halfY), 0.0f,
                                                   std::cos(halfY) };
    const HarmonicMIDI::grid::SceneQuaternion z { 0.0f, 0.0f, std::sin(halfZ),
                                                   std::cos(halfZ) };
    return multiplySceneQuaternion(z, multiplySceneQuaternion(y, x));
}

inline HarmonicMIDI::grid::SceneVec3 rotateSceneVector(
    const HarmonicMIDI::grid::SceneQuaternion& rotation,
    const HarmonicMIDI::grid::SceneVec3& value) noexcept
{
    const HarmonicMIDI::grid::SceneVec3 quaternion { rotation.x, rotation.y, rotation.z };
    const HarmonicMIDI::grid::SceneVec3 first {
        quaternion.y * value.z - quaternion.z * value.y,
        quaternion.z * value.x - quaternion.x * value.z,
        quaternion.x * value.y - quaternion.y * value.x
    };
    const HarmonicMIDI::grid::SceneVec3 second {
        quaternion.y * first.z - quaternion.z * first.y,
        quaternion.z * first.x - quaternion.x * first.z,
        quaternion.x * first.y - quaternion.y * first.x
    };
    return {
        value.x + 2.0f * (rotation.w * first.x + second.x),
        value.y + 2.0f * (rotation.w * first.y + second.y),
        value.z + 2.0f * (rotation.w * first.z + second.z)
    };
}

inline HarmonicMIDI::grid::SceneTransform3D composeSceneTransform(
    const HarmonicMIDI::grid::SceneTransform3D& parent,
    const HarmonicMIDI::grid::SceneTransform3D& child) noexcept
{
    const HarmonicMIDI::grid::SceneVec3 scaledTranslation {
        child.translation.x * parent.scale.x,
        child.translation.y * parent.scale.y,
        child.translation.z * parent.scale.z
    };
    const auto rotatedTranslation = rotateSceneVector(parent.rotation, scaledTranslation);
    HarmonicMIDI::grid::SceneTransform3D result;
    result.translation = {
        parent.translation.x + rotatedTranslation.x,
        parent.translation.y + rotatedTranslation.y,
        parent.translation.z + rotatedTranslation.z
    };
    result.rotation = multiplySceneQuaternion(parent.rotation, child.rotation);
    result.scale = {
        parent.scale.x * child.scale.x,
        parent.scale.y * child.scale.y,
        parent.scale.z * child.scale.z
    };
    return result;
}

inline const HarmonicMIDI::grid::SceneObjectRecord* findFixtureObject(
    const HarmonicMIDI::grid::Visual3DScene& scene,
    HarmonicMIDI::grid::SceneObjectId id) noexcept
{
    return HarmonicMIDI::grid::visual3d_detail::findById(
        scene.objects, scene.objectCount, id);
}

inline const HarmonicMIDI::grid::SceneMaterialRecord* findFixtureMaterial(
    const HarmonicMIDI::grid::Visual3DScene& scene,
    HarmonicMIDI::grid::SceneMaterialId id) noexcept
{
    return HarmonicMIDI::grid::visual3d_detail::findById(
        scene.materials, scene.materialCount, id);
}

inline const HarmonicMIDI::grid::SceneTextureRecord* findFixtureTexture(
    const HarmonicMIDI::grid::Visual3DScene& scene,
    HarmonicMIDI::grid::SceneTextureId id) noexcept
{
    return HarmonicMIDI::grid::visual3d_detail::findById(
        scene.textures, scene.textureCount, id);
}

inline const HarmonicMIDI::grid::SceneCameraRecord* findFixtureActiveCamera(
    const HarmonicMIDI::grid::Visual3DScene& scene) noexcept
{
    return HarmonicMIDI::grid::visual3d_detail::findById(
        scene.cameras, scene.cameraCount, scene.activeCamera);
}

inline HarmonicMIDI::grid::SceneTransform3D fixtureWorldTransform(
    const HarmonicMIDI::grid::Visual3DScene& scene,
    const HarmonicMIDI::grid::SceneObjectRecord& object) noexcept
{
    auto result = object.transform;
    auto parent = object.parent;
    for (std::size_t depth = 0; parent.isValid() && depth < scene.objectCount; ++depth)
    {
        const auto* record = findFixtureObject(scene, parent);
        if (record == nullptr)
            break;
        result = composeSceneTransform(record->transform, result);
        parent = record->parent;
    }
    return result;
}

inline HarmonicMIDI::grid::SceneTransform3D applyRuntimeObjectTransform(
    HarmonicMIDI::grid::SceneTransform3D transform,
    const std::array<float, 3>& rotationDegrees, float uniformScale) noexcept
{
    transform.rotation = multiplySceneQuaternion(
        runtimeEulerQuaternion(rotationDegrees), transform.rotation);
    transform.scale.x *= uniformScale;
    transform.scale.y *= uniformScale;
    transform.scale.z *= uniformScale;
    return transform;
}

inline bool validFixtureCameraOverride(
    const std::optional<HarmonicMIDI::grid::SceneCameraRecord>& camera) noexcept
{
    return !camera.has_value()
        || (HarmonicMIDI::grid::visual3d_detail::validTransform(camera->transform)
            && camera->transform.scale.x == 1.0f
            && camera->transform.scale.y == 1.0f
            && camera->transform.scale.z == 1.0f
            && std::isfinite(camera->verticalFovRadians)
            && camera->verticalFovRadians > 0.0f
            && camera->verticalFovRadians < 3.1415926535f
            && std::isfinite(camera->nearPlane) && camera->nearPlane > 0.0f
            && std::isfinite(camera->farPlane) && camera->farPlane > camera->nearPlane);
}

struct NativeSceneMotionUniforms final
{
    std::array<float, 4> rotation { 0, 0, 0, 1 }, translationScale { 0, 0, 0, 1 };
    std::array<float, 4> cameraRotation {}, cameraTranslation {}, projection {};
    std::array<float, 4> viewport {};
};

inline NativeSceneMotionUniforms sceneMotionUniforms(const NativeFixtureSceneRuntimeInputs& inputs,
    const HarmonicMIDI::grid::SceneCameraRecord& camera, std::uint32_t width, std::uint32_t height) noexcept
{
    NativeSceneMotionUniforms result;
    const auto previous = inputs.previousMotion.value_or(sceneMotionSample(inputs));
    const auto currentRotation = runtimeEulerQuaternion(inputs.objectRotationDegrees);
    const auto previousRotation = runtimeEulerQuaternion(previous.objectRotationDegrees);
    const auto delta = multiplySceneQuaternion(previousRotation,
        { -currentRotation.x, -currentRotation.y, -currentRotation.z, currentRotation.w });
    const auto scale = previous.objectScale / inputs.objectScale;
    const auto translated = rotateSceneVector(delta, { inputs.objectTranslationOffset[0],
        inputs.objectTranslationOffset[1], inputs.objectTranslationOffset[2] });
    result.rotation = { delta.x, delta.y, delta.z, delta.w };
    result.translationScale = { previous.objectTranslationOffset[0] - translated.x * scale,
        previous.objectTranslationOffset[1] - translated.y * scale,
        previous.objectTranslationOffset[2] - translated.z * scale, scale };
    const auto& priorCamera = previous.cameraOverride ? *previous.cameraOverride : camera;
    const auto& q = priorCamera.transform.rotation;
    const auto& p = priorCamera.transform.translation;
    result.cameraRotation = { q.x, q.y, q.z, q.w };
    result.cameraTranslation = { p.x + previous.cameraTranslationOffset[0],
        p.y + previous.cameraTranslationOffset[1], p.z + previous.cameraTranslationOffset[2], 0 };
    result.projection = { std::tan(priorCamera.verticalFovRadians * 0.5f), static_cast<float>(width) / height,
        priorCamera.nearPlane, inputs.previousMotion ? 1.0f : 0.0f };
    result.viewport = { static_cast<float>(width), static_cast<float>(height), 0, 0 };
    return result;
}

inline bool validFixtureLightOverride(
    const std::optional<HarmonicMIDI::grid::SceneLightRecord>& light) noexcept
{
    using namespace HarmonicMIDI::grid;
    return !light.has_value()
        || (light->id.isValid()
            && (light->kind == SceneLightKind::Directional
                || light->kind == SceneLightKind::Point
                || light->kind == SceneLightKind::Environment)
            && visual3d_detail::validTransform(light->transform)
            && visual3d_detail::finite(light->color)
            && visual3d_detail::nonNegative(light->color)
            && visual3d_detail::finite(light->intensity) && light->intensity >= 0.0f
            && visual3d_detail::finite(light->range) && light->range >= 0.0f);
}

inline bool validFixtureRuntimeInputs(
    const NativeFixtureSceneRuntimeInputs& inputs) noexcept
{
    const auto valid = [] (const std::array<float, 3>& values, float maximum)
    {
        for (const auto value : values)
            if (!std::isfinite(value) || std::abs(value) > maximum)
                return false;
        return true;
    };
    return inputs.objectNodeStableId <= 65536
        && inputs.rawExportMask <= render3dimage::kRawExportMask
        && render3dimage::supported(inputs.imageOutput)
        && (!inputs.linearColor || inputs.imageOutput == renderpassoutput::Output::Color
            || inputs.imageOutput == renderpassoutput::Output::Emission)
        && std::all_of(inputs.vertexSpectrum.begin(), inputs.vertexSpectrum.end(), [](float value)
            { return std::isfinite(value) && value >= 0.0f && value <= 1.0f; })
        && (!inputs.previousMotion || (valid(inputs.previousMotion->objectTranslationOffset, 1000000.0f)
            && valid(inputs.previousMotion->objectRotationDegrees, 360.0f)
            && std::isfinite(inputs.previousMotion->objectScale)
            && inputs.previousMotion->objectScale >= 0.01f && inputs.previousMotion->objectScale <= 100.0f
            && valid(inputs.previousMotion->cameraTranslationOffset, 1000000.0f)
            && validFixtureCameraOverride(inputs.previousMotion->cameraOverride)))
        && (!inputs.passComposite || (renderpasscomposite::valid(*inputs.passComposite)
            && inputs.imageOutput == renderpassoutput::Output::Color))
        && (!inputs.passProgram || (inputs.passComposite && renderpasscomposite::valid(*inputs.passProgram)
            && renderpasscomposite::sameParameters(inputs.passProgram->steps[inputs.passProgram->output].parameters,
                                                  *inputs.passComposite)))
        && (inputs.passProgram || !inputs.passComposite || !renderpasscomposite::branches(inputs.passComposite->mode))
        && valid(inputs.objectTranslationOffset, 1000000.0f)
        && std::isfinite(inputs.morphWeight)
        && inputs.morphWeight >= 0.0f && inputs.morphWeight <= 1.0f
        && valid(inputs.objectRotationDegrees, 360.0f)
        && std::isfinite(inputs.objectScale)
        && inputs.objectScale >= 0.01f && inputs.objectScale <= 100.0f
        && valid(inputs.cameraTranslationOffset, 1000000.0f)
        && validFixtureCameraOverride(inputs.animatedCamera)
        && inputs.animatedLights.size() <= HarmonicMIDI::grid::Visual3DScene::kMaxLights
        && std::all_of(inputs.animatedLights.begin(), inputs.animatedLights.end(),
            [](auto light) {
                if (light.kind == HarmonicMIDI::grid::SceneLightKind::Spot) {
                    if (!std::isfinite(light.innerConeAngle) || !std::isfinite(light.outerConeAngle)
                        || light.innerConeAngle < 0 || light.outerConeAngle <= light.innerConeAngle
                        || light.outerConeAngle > 1.5707964f) return false;
                    light.kind = HarmonicMIDI::grid::SceneLightKind::Point;
                }
                return validFixtureLightOverride(light);
            })
        && validFixtureCameraOverride(inputs.cameraOverride)
        && validFixtureLightOverride(inputs.lightOverride)
        && std::isfinite(inputs.emissionGain)
        && inputs.emissionGain >= 0.0f && inputs.emissionGain <= 64.0f
        && (static_cast<bool>(inputs.canonicalBlockCFrame)
                == inputs.noteInstanceMapping.has_value())
        && (!inputs.canonicalBlockCFrame
            || (canonicalblockc::valid(inputs.canonicalBlockCFrame)
                && visualnoteinstancing::valid(*inputs.noteInstanceMapping)));
}

struct GeometryCoreExecutionCapabilities final
{
    bool immutableSourceBuffers=false;
    bool stableElementIds=false;
    bool typedFieldEvaluation=false;
    bool gpuInstancingWithoutMeshExpansion=false;
    std::uint32_t supportedCarriers=0;
    std::size_t maxVertices=0,maxIndices=0,maxPoints=0,maxCurvePoints=0,maxSplines=0,maxInstances=0,maxFieldElements=0,maxAttributes=0,maxOperations=0,maxDispatches=0,maxBufferBytes=0;
};

inline bool nativeFixtureSceneTopologySupported(
    const HarmonicMIDI::grid::Visual3DScene& scene) noexcept
{
    using namespace HarmonicMIDI::grid;
    if (!validateVisual3DScene(scene).valid() || scene.objectCount == 0
        || scene.materialCount == 0 || scene.cameraCount == 0
        || !scene.activeCamera.isValid())
        return false;

    std::size_t directLightCount = 0;
    std::size_t environmentLightCount = 0;
    for (std::size_t index = 0; index < scene.lightCount; ++index)
    {
        const auto& light = scene.lights[index];
        if (light.kind == SceneLightKind::Environment)
            ++environmentLightCount;
        else
            ++directLightCount;
        if (light.kind == SceneLightKind::Point && light.range <= 0.0f)
            return false;
    }
    if (directLightCount > 1 || environmentLightCount > 1)
        return false;

    return true;
}

class NativeFixtureSceneBackend
{
public:
    virtual ~NativeFixtureSceneBackend() = default;
    virtual BackendInfo info() const = 0;
    virtual GeometryCoreExecutionCapabilities geometryCoreCapabilities() const { return {}; }
    // Release cached programs while their context is current, before destroying
    // it. Live scene-resource leases remain valid until their owners release them.
    virtual void releaseCachedProgramsForCurrentContext() noexcept {}
    virtual NativeFixtureScenePreparation prepare (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
        const std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>& materialProgram = {}) = 0;
    virtual NativeFixtureScenePreparation prepareGeometryInstances (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
        const std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>& materialProgram,
        const std::shared_ptr<const videohelper::geometry::AdmittedPlanValue>&
            geometryAdmission,
        bool diagnosticInstanceIdentityColors = false,
        const std::vector<videowire::geometry::AttributeData>* frameAttributes = nullptr)
    {
        (void) geometryAdmission;
        (void) diagnosticInstanceIdentityColors;
        (void) frameAttributes;
        return prepare (scene, materialProgram);
    }
    virtual NativeFixtureSceneSubmission render (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
        const std::shared_ptr<const NativeFixtureSceneResources>& resources,
        std::uint32_t width,
        std::uint32_t height,
        NativeFixtureSceneRuntimeInputs runtimeInputs = {}) = 0;
};

// Returns the process-owned strict native backend. Stub and unavailable builds
// return a backend that rejects every submission without allocating resources.
NativeFixtureSceneBackend& nativeFixtureSceneBackend();

// Phase-8's first production deformation slice is deliberately bounded to one
// mesh draw.  Static vertices, indices, joint influences, inverse bind matrices,
// and morph deltas are uploaded once; an immutable evaluated snapshot supplies
// only per-frame pose and weights.  The native backend must execute deformation
// and the draw on the GPU.  There is no CPU image or CPU-deformed-vertex path.
inline constexpr std::size_t kNativeDeformationMaxVertices = 4096;
inline constexpr std::size_t kNativeDeformationMaxJoints = 64;
inline constexpr std::size_t kNativeDeformationMaxMorphTargets = 8;

struct NativeDeformationJointBaseTransform
{
    visualdeformation::SkinId skin;
    visualdeformation::JointId joint;
    std::array<float, 3> translation { 0.0f, 0.0f, 0.0f };
    std::array<float, 4> rotation { 0.0f, 0.0f, 0.0f, 1.0f };
    std::array<float, 3> scale { 1.0f, 1.0f, 1.0f };
    std::optional<std::array<float, 16>> matrix = std::nullopt;
};

// One immutable owner joins exact graph, clip, scene-object, mesh, skin and
// morph identity.  Preview and export share this owner and therefore the same
// backend cache entry.
struct NativeDeformationScene
{
    std::uint64_t sourceStableId = 0;
    std::uint64_t deformationStableId = 0;
    std::uint64_t structuralRevision = 0;
    visualanimation::ClipId clip;
    visualdeformation::MeshId mesh;
    HarmonicMIDI::grid::SceneObjectId object;
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> scene;
    std::shared_ptr<const visualdeformation::DeformationAsset> deformation;
    std::vector<NativeDeformationJointBaseTransform> jointBaseTransforms;
    // glTF mesh/node default weights in exact mesh target order. Empty means
    // the schema-defined all-zero base state.
    std::vector<float> morphBaseWeights;
    // Opt-in Geometry3D extraction reads the compute result, never a CPU
    // deformation substitute. Ordinary imported rendering does not read back.
    bool retainDeformedGeometry = false;
    // A scene batch shares one immutable scene and one GPU vertex buffer.
    // Each member addresses a contiguous mesh range in that scene.
    std::vector<std::shared_ptr<const NativeDeformationScene>> draws;
    bool batchMember = false;
    // glTF node identity selects the instance, while mesh/skin data remain shared.
    std::uint64_t animationNodeStableId = 0;
    std::optional<visualdeformation::SkinId> skin;
};

struct NativeDeformationFrameData
{
    std::array<float, kNativeDeformationMaxJoints * 16> jointPalette {};
    std::array<float, kNativeDeformationMaxMorphTargets> morphWeights {};
    std::uint32_t vertexCount = 0;
    std::uint32_t jointCount = 0;
    std::uint32_t morphTargetCount = 0;
};

bool prepareNativeDeformationFrame (
    const NativeDeformationScene& source,
    const visualdeformation::AnimationDeformationSnapshot& snapshot,
    NativeDeformationFrameData& output,
    std::string& error);

struct NativeDeformationStats
{
    std::uint32_t drawCount = 0;
    std::uint32_t dispatchCount = 0;
    std::uint32_t staticUploadCount = 0;
    bool reusedStaticResources = false;
    std::uint64_t staticVertexBytes = 0;
    std::uint64_t staticDeformationBytes = 0;
    std::uint64_t dynamicUniformBytes = 0;
    std::uint64_t sourceStableId = 0;
    std::uint64_t deformationStableId = 0;
    std::uint64_t clipId = 0;
    std::uint64_t meshId = 0;
    std::uint64_t skinId = 0;
    std::uint64_t revision = 0;
    visualdeformation::RationalFrameTime time {};
};

class NativeDeformationResources
{
public:
    virtual ~NativeDeformationResources() = default;
    virtual const std::string& backend() const noexcept = 0;
};

struct NativeDeformationPreparation
{
    bool prepared = false;
    std::shared_ptr<const NativeDeformationResources> resources;
    NativeDeformationStats stats {};
    std::string error;
};

struct NativeDeformationSubmission
{
    bool rendered = false;
    std::shared_ptr<const NativeFixtureSceneFrame> frame;
    NativeDeformationStats stats {};
    std::string error;
    // Eight floats per vertex: position, normal, UV, in selected mesh space.
    std::vector<float> deformedVertices {};
};

class NativeDeformationBackend
{
public:
    virtual ~NativeDeformationBackend() = default;
    virtual BackendInfo info() const = 0;
    virtual NativeDeformationPreparation prepare (
        const std::shared_ptr<const NativeDeformationScene>& source,
        const std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>& materialProgram = {}) = 0;
    virtual NativeDeformationSubmission render (
        const std::shared_ptr<const NativeDeformationScene>& source,
        const std::shared_ptr<const visualdeformation::AnimationDeformationSnapshot>& snapshot,
        const std::shared_ptr<const NativeDeformationResources>& resources,
        std::uint32_t width,
        std::uint32_t height,
        NativeDeformationRuntimeInputs runtimeInputs = {}) = 0;
};

// Every backend must fail closed until its physical GPU deformation path is
// admitted. Unsupported paths may not substitute CPU-deformed vertices/pixels.
NativeDeformationBackend& nativeDeformationBackend();

} // namespace arbitgpu
