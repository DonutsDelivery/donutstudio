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
#include "../optical_flow_contract.h"
#include "../diffraction_material_execution.h"
#include "../../../shared/DiffractionMaterialGpuLayout.h"
#include "../../../shared/DiffractiveFoilIR.h"
#include "../../../shared/DiffractionLightingPlan.h"
#include "../../../shared/SdfIr.h"
#include "../../../shared/SurfaceMaterialIR.h"
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
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace arbitgpu { class NativeSdfCompiledProgram; }
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
class NativeSdfSceneFrame
{
public:
    virtual ~NativeSdfSceneFrame() = default;
    virtual const std::string& backend() const noexcept = 0;
    virtual std::uint32_t width() const noexcept = 0;
    virtual std::uint32_t height() const noexcept = 0;
    virtual std::uintptr_t colorImageHandle() const noexcept = 0;
    virtual std::uintptr_t colorTextureViewHandle() const noexcept = 0;
    virtual const FrameMemoryAdmission& frameMemoryAdmission() const noexcept = 0;
    virtual const NativeSdfResourceReceipt& sdfResourceReceipt() const noexcept = 0;
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
        TimeLinearMix = 2
    };

    struct ImportedSrgbTexture final
    {
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        std::size_t texelCount = 0;
        std::array<HarmonicMIDI::grid::SceneTexelRgba8,
                   HarmonicMIDI::grid::Visual3DScene::kMaxTextureTexels> texels {};
    };

    static constexpr std::uint32_t kLayoutVersion = 10;

    std::uint32_t layoutVersion = kLayoutVersion;
    NativeFixtureMaterialBackend backend = NativeFixtureMaterialBackend::Invalid;
    NativeFixtureMaterialKind kind = NativeFixtureMaterialKind::SurfacePbr;
    HarmonicMIDI::grid::SceneObjectId object {};
    std::string bindingDigest;
    std::string programIdentity;
    BaseColorSource baseColorSource = BaseColorSource::ConstantLinear;
    std::optional<ImportedSrgbTexture> importedBaseColorTexture;
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
        && finite (parameters.transmissionIorClearcoat);
}

// Canonical, versioned receipt for the exact admitted diffraction inputs. The
// fixed-width hexadecimal fields preserve integer and IEEE-754 float bits, so
// no locale, decimal formatting, or caller revision can alias two plans.
inline std::string nativeFixtureDiffractionProgramIdentity (
    const std::string& bindingDigest,
    const diffractionmaterial::AdmittedLightingPlan& admitted,
    std::uint8_t patternMask = 0,
    float maskCoverage = 1.0f,
    std::string_view productDigest = {})
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
                               && value.z > 1.0e-6f && std::isfinite(value.w)
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
            && direction[0] == path.material.incident.x
            && direction[1] == path.material.incident.y
            && direction[2] == path.material.incident.z;
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
               program.diffractionProductDigest);
    return expectedBackend != NativeFixtureMaterialBackend::Invalid
        && expectedObject.isValid()
        && program.layoutVersion == Program::kLayoutVersion
        && program.backend == expectedBackend
        && program.kind == NativeFixtureMaterialKind::DiffractionReflective
        && program.object == expectedObject
        && ! program.bindingDigest.empty()
        && identityValid
        && program.baseColorSource == Program::BaseColorSource::ConstantLinear
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
    const auto pathCount = static_cast<std::uint64_t>(material.diffractionPathCount);
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
    Invalid = 0, Rgba8Unorm, Bgra8Unorm, R32Float
};
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
    // Identifies the prepared static-resource lifetime on every backend. It is
    // stable across repeated renders from one preparation and changes when the
    // scene is prepared again.
    std::uint64_t rendererGeneration = 0;

    bool complete() const noexcept
    {
        return !backend.empty() && viewKind == NativeTextureViewKind::Texture2D
            && format != NativeTexturePixelFormat::Invalid && imageHandle != 0
            && textureViewHandle != 0 && width != 0 && height != 0
            && sampleCount == 1 && sampledBinding && deviceOrContextIdentity != 0
            && rendererGeneration != 0;
    }
};

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
};

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
struct NativeImportedSceneRuntimeInputs final
{
    float timeSeconds = 0.0f;
    float morphWeight = 1.0f;
    std::array<float, 3> objectTranslationOffset {};
    std::array<float, 3> objectRotationDegrees {};
    float objectScale = 1.0f;
    std::array<float, 3> cameraTranslationOffset {};
    std::optional<HarmonicMIDI::grid::SceneCameraRecord> cameraOverride;
    std::optional<HarmonicMIDI::grid::SceneLightRecord> lightOverride;
    float emissionGain = 1.0f;
    std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> canonicalBlockCFrame;
    std::optional<visualnoteinstancing::Mapping> noteInstanceMapping;
};

using NativeFixtureSceneRuntimeInputs = NativeImportedSceneRuntimeInputs;
using NativeDeformationRuntimeInputs = NativeImportedSceneRuntimeInputs;

struct NativeNoteInstanceBatch final
{
    std::array<std::array<float, 4>, visualnoteinstancing::kMaximumInstances> transforms {};
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
    }
    return batch;
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
    return valid(inputs.objectTranslationOffset, 1000000.0f)
        && std::isfinite(inputs.morphWeight)
        && inputs.morphWeight >= 0.0f && inputs.morphWeight <= 1.0f
        && valid(inputs.objectRotationDegrees, 360.0f)
        && std::isfinite(inputs.objectScale)
        && inputs.objectScale >= 0.01f && inputs.objectScale <= 100.0f
        && valid(inputs.cameraTranslationOffset, 1000000.0f)
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
    virtual NativeFixtureScenePreparation prepare (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
        const std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>& materialProgram = {}) = 0;
    virtual NativeFixtureScenePreparation prepareGeometryInstances (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
        const std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>& materialProgram,
        const std::shared_ptr<const videohelper::geometry::AdmittedPlanValue>&
            geometryAdmission,
        bool diagnosticInstanceIdentityColors = false)
    {
        (void) geometryAdmission;
        (void) diagnosticInstanceIdentityColors;
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
