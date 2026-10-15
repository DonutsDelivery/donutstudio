// gpu_backend/backend.h -- language-neutral native GPU backend seam.
//
// The existing FrameRenderer remains on OpenGL while the visual-engine P6
// migration lands in slices.  This seam makes native-backend capability and
// verification available without leaking Objective-C types into the helper.
#pragma once

#include "../canonical_block_c_frame.h"

#include "../../../shared/RenderPassOutputContract.h"
#include "../../../shared/MotionViewOperationContract.h"
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
#include "../../../shared/Visual3DSceneComposition.h"
#include "../../../shared/VisualImportedSceneRenderOperationContract.h"
#include "../../../shared/VisualTemporalSamplingContract.h"
#include "../../../shared/GeometryCore.h"
#include "../../../shared/VisualNoteInstancingContract.h"
#include "../../../shared/VisualAnimationDeformationEvaluation.h"
#include "../../../shared/VisualModelAssetPayload.h"
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

namespace arbitgpu { class NativeSdfCompiledProgram; class NativeDeformationCompositionOwner; class NativeDeformationCompositionFrame; }
namespace videohelper::gltf { struct GlbStaticMeshDocument; struct GlbAnimationDocument; }
namespace videohelper::modelpayload { class ImportedAnimatedScenePayloadExecution; class NativeSceneAnimationCompositionFrame; }
namespace videoopticalflow { class PublishedNativeResult; class NativeSourceFrame; }
namespace videohelper::materialprogram { class MaterialProgramRecord; }
namespace videohelper::geometry { class AdmittedPlanValue; }
namespace videohelper::sdf
{
inline std::shared_ptr<const arbitgpu::NativeSdfCompiledProgram> compileNativeSdfProgram (
    const videowire::SdfIr&, std::string&, bool textTablesAvailable = false);
}

namespace arbitgpu
{
class NativeFixtureSceneFrame;
inline bool validNativeCompositionRetention(const std::shared_ptr<const NativeDeformationCompositionFrame>&,
    const std::shared_ptr<const visualdeformation::AnimationDeformationSnapshot>&) noexcept;
inline bool sameNativeCompositionOwner(const std::shared_ptr<const NativeDeformationCompositionFrame>&,
    const std::shared_ptr<const NativeDeformationCompositionFrame>&) noexcept;
inline bool nativeCompositionTargetsObject(const std::shared_ptr<const NativeDeformationCompositionFrame>&,
    std::uint64_t, HarmonicMIDI::grid::SceneObjectId) noexcept;

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
    std::uint32_t sourceClipId = 0;
    videoopticalflow::ContentIdentity sourceContent {};
    std::uint64_t sourceGeneration = 0;
    videoopticalflow::BackendIdentity backendImplementation {};
    std::uint32_t backendImplementationRevision = 0;
    std::uint64_t byteCount = 0;
    std::uintptr_t imageHandle = 0;
    std::uintptr_t textureViewHandle = 0;
    std::uintptr_t nativeContextIdentity = 0;
    bool immutable = false;
};

inline bool exactOpticalFlowV2InputReceipt (
    const videoopticalflow::AdmittedRequest& request,
    const NativeOpticalFlowInputResource& resource,
    const videoopticalflow::SourceFrameV2& expected) noexcept
{
    const auto& frame = resource.descriptor;
    const auto& selected = expected.frame;
    return resource.immutable
        && videoopticalflow::detail::anyNonZero(resource.identity)
        && resource.nativeContextIdentity != 0
        && resource.sourceClipId == expected.clipId
        && resource.sourceContent == expected.sourceContent
        && resource.sourceGeneration == expected.sourceGeneration
        && frame.identity.stream == selected.identity.stream
        && frame.identity.content == selected.identity.content
        && frame.identity.frameIndex == selected.identity.frameIndex
        && frame.timestamp.ticks == selected.timestamp.ticks
        && frame.timestamp.timescale == selected.timestamp.timescale
        && frame.extent == selected.extent
        && frame.format == selected.format
        && frame.colorSpace == selected.colorSpace
        && resource.backendImplementation == request.backendImplementation()
        && resource.backendImplementationRevision
               == request.backendImplementationRevision()
        && resource.byteCount == request.footprint().referencedInputBytes / 2;
}

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
    std::shared_ptr<const NativeFixtureSceneFrame> renderMotionView (
        const std::shared_ptr<const videoopticalflow::PublishedNativeResult>& field,
        std::string& error)
    {
        return renderMotionView (field, motionviewoperation::Payload {}, error);
    }
    virtual std::shared_ptr<const NativeFixtureSceneFrame> renderMotionView (
        const std::shared_ptr<const videoopticalflow::PublishedNativeResult>&,
        const motionviewoperation::Payload&, std::string& error)
    {
        error = "native Motion View is unavailable on this backend";
        return {};
    }
    virtual std::shared_ptr<const NativeFixtureSceneFrame> renderMotionBlur (
        const std::shared_ptr<const videoopticalflow::NativeSourceFrame>&,
        const std::shared_ptr<const videoopticalflow::PublishedNativeResult>&,
        const visualtemporalsampling::Payload&, std::string& error)
    {
        error = "native optical Motion Blur is unavailable on this backend";
        return {};
    }
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

// Persistent SDF outputPass selections on the Image port. The separate Depth
// port always exposes R32F linear travel / maximumDistance, with misses at 1,
// independently of this visualization choice.
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
        static_cast<std::size_t> (videowire::SdfOperation::textExtrusion) + 1;

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
    // Set only when every output attachment reuses one primary hit/travel.
    bool reusesPrimaryTrace = false;

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

// GPU tables use four 32-bit lanes per entry. Indices address the global
// contour/segment arrays; identity stays in the immutable CPU receipt.
struct NativeSdfTextResourceHeader final
{
    std::uint32_t firstContour = 0, contourCount = 0;
    std::uint32_t firstSegment = 0, segmentCount = 0;
};
struct NativeSdfTextContourHeader final
{
    std::uint32_t firstSegment = 0, segmentCount = 0;
    std::int32_t winding = 0;
    std::uint32_t glyphIndex = 0;
};
struct NativeSdfTextSegment final
{
    float x0 = 0, y0 = 0, x1 = 0, y1 = 0;
};
static_assert (sizeof (NativeSdfTextResourceHeader) == 16);
static_assert (sizeof (NativeSdfTextContourHeader) == 16);
static_assert (sizeof (NativeSdfTextSegment) == 16);

// Derived from packed float outlines, never serialized or hashed. Preorder
// nodes have an absolute one-past-subtree escape. Internal indexCount is zero;
// leaves address original segments through textSegmentIndices.
struct NativeSdfTextIndexNode final
{
    float minX = 0, minY = 0, maxX = 0, maxY = 0;
    std::uint32_t firstIndex = 0, indexCount = 0, escape = 0, reserved = 0;
};
static_assert (sizeof (NativeSdfTextIndexNode) == 32);
inline constexpr std::uint32_t kNativeSdfTextIndexLeafSegments = 4;

// Encloses the packed program's zero sublevel set in real arithmetic. It does
// not certify float raymarch misses or adaptive epsilon hits.
struct NativeSdfSceneBounds final
{
    std::array<double, 3> minimum {}, maximum {};
};

class NativeSdfCompiledProgram final
{
public:
    std::uint32_t rootIndex() const noexcept { return rootIndex_; }
    std::uint32_t maximumDepth() const noexcept { return maximumDepth_; }
    std::uint32_t evaluationSteps() const noexcept { return evaluationSteps_; }
    const std::vector<NativeSdfCompiledRecord>& records() const noexcept { return records_; }
    const std::string& structuralDigest() const noexcept { return structuralDigest_; }
    std::uint32_t schemaVersion() const noexcept { return schemaVersion_; }
    const std::vector<std::uint32_t>& textResourceIndices() const noexcept { return textResourceIndices_; }
    const std::vector<NativeSdfTextResourceHeader>& textResources() const noexcept { return textResources_; }
    const std::vector<NativeSdfTextContourHeader>& textContours() const noexcept { return textContours_; }
    const std::vector<NativeSdfTextSegment>& textSegments() const noexcept { return textSegments_; }
    const std::vector<NativeSdfTextIndexNode>& textIndexNodes() const noexcept { return textIndexNodes_; }
    const std::vector<std::uint32_t>& textSegmentIndices() const noexcept { return textSegmentIndices_; }
    const std::vector<std::uint32_t>& textIndexRoots() const noexcept { return textIndexRoots_; }
    std::uint64_t textIndexNodeVisitsPerQuery() const noexcept { return textIndexNodeVisitsPerQuery_; }
    const std::optional<NativeSdfSceneBounds>& sceneBounds() const noexcept { return sceneBounds_; }
    const std::vector<videowire::SdfTextResourceId>& textResourceStableIds() const noexcept { return textResourceStableIds_; }
    const std::vector<std::string>& textResourceIdentities() const noexcept { return textResourceIdentities_; }
    std::uint64_t segmentVisitsPerQuery() const noexcept { return segmentVisitsPerQuery_; }
    std::uint64_t textTableBytes() const noexcept
    {
        return textResourceIndices_.size() * sizeof (std::uint32_t)
            + textResources_.size() * sizeof (NativeSdfTextResourceHeader)
            + textContours_.size() * sizeof (NativeSdfTextContourHeader)
            + textSegments_.size() * sizeof (NativeSdfTextSegment)
            + textIndexNodes_.size() * sizeof (NativeSdfTextIndexNode)
            + textSegmentIndices_.size() * sizeof (std::uint32_t)
            + textIndexRoots_.size() * sizeof (std::uint32_t);
    }
    std::uint64_t storageBytes() const noexcept
    {
        if (textResources_.empty()) return records_.size() * sizeof (NativeSdfCompiledRecord);
        std::uint64_t bytes = records_.capacity() * sizeof (NativeSdfCompiledRecord)
            + textResourceIndices_.capacity() * sizeof (std::uint32_t)
            + textResources_.capacity() * sizeof (NativeSdfTextResourceHeader)
            + textContours_.capacity() * sizeof (NativeSdfTextContourHeader)
            + textSegments_.capacity() * sizeof (NativeSdfTextSegment)
            + textIndexNodes_.capacity() * sizeof (NativeSdfTextIndexNode)
            + textSegmentIndices_.capacity() * sizeof (std::uint32_t)
            + textIndexRoots_.capacity() * sizeof (std::uint32_t)
            + textResourceStableIds_.capacity() * sizeof (videowire::SdfTextResourceId)
            + textResourceIdentities_.capacity() * sizeof (std::string);
        for (const auto& identity : textResourceIdentities_) bytes += identity.capacity() + 1;
        return bytes;
    }

private:
    friend std::shared_ptr<const NativeSdfCompiledProgram>
        videohelper::sdf::compileNativeSdfProgram (const videowire::SdfIr&, std::string&, bool);
    NativeSdfCompiledProgram() = default;
    std::uint32_t rootIndex_ = 0;
    std::uint32_t maximumDepth_ = 0;
    std::uint32_t evaluationSteps_ = 0;
    std::vector<NativeSdfCompiledRecord> records_;
    std::string structuralDigest_;
    std::uint32_t schemaVersion_ = videowire::kSdfIrSchemaVersion;
    std::vector<std::uint32_t> textResourceIndices_;
    std::vector<NativeSdfTextResourceHeader> textResources_;
    std::vector<NativeSdfTextContourHeader> textContours_;
    std::vector<NativeSdfTextSegment> textSegments_;
    std::vector<videowire::SdfTextResourceId> textResourceStableIds_;
    std::vector<std::string> textResourceIdentities_;
    std::uint64_t segmentVisitsPerQuery_ = 0;
    std::vector<NativeSdfTextIndexNode> textIndexNodes_;
    std::vector<std::uint32_t> textSegmentIndices_;
    std::vector<std::uint32_t> textIndexRoots_;
    std::uint64_t textIndexNodeVisitsPerQuery_ = 0;
    std::optional<NativeSdfSceneBounds> sceneBounds_;
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

struct NativeSdfWorkReceipt final
{
    std::uint64_t distanceQueryCount = 0;
    std::uint64_t recordVisits = 0;
    std::uint64_t segmentVisits = 0;
    std::uint64_t textIndexNodeVisits = 0;
};

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
    bool normalInspection = false;
    NativeSdfWorkReceipt textWork;
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
    std::uint64_t transientUploadBytes = 0;
    std::uint64_t peakBytes = 0;
    // Compiled CPU tables, plus actual backend allocation bytes (including
    // padding) reported by each consumer. Do not count cache tables twice.
    std::uint64_t compiledTextTableBytes = 0;
    std::uint64_t nativeTextTableBytes = 0;
    std::uint64_t distanceQueryCount = 0;
    std::uint64_t recordVisits = 0;
    std::uint64_t segmentVisits = 0;
    std::uint64_t textIndexNodeVisits = 0;
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
    virtual std::uintptr_t depthTextureViewHandle() const noexcept = 0;
    virtual NativeTextureViewDescriptor depthTextureDescriptor() const noexcept = 0;
    virtual std::uintptr_t normalTextureViewHandle() const noexcept = 0;
    virtual NativeTextureViewDescriptor normalTextureDescriptor() const noexcept = 0;
    virtual const FrameMemoryAdmission& frameMemoryAdmission() const noexcept = 0;
    virtual const NativeSdfResourceReceipt& sdfResourceReceipt() const noexcept = 0;
    // Explicit diagnostic readback preserves finite float map values for native
    // acceptance. Normal rendering and composition remain GPU-owned.
    virtual bool readColorFloatPixels (std::vector<float>& output) const
    {
        output.clear();
        return false;
    }
    virtual bool readDepthFloatPixels (std::vector<float>& output) const
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
    // One draw of a scene's instanced object submits every instance transform.
    std::uint32_t objectInstanceDrawCount = 0;
    std::uint32_t submittedObjectInstanceCount = 0;
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
    // Opaque visibility is unavailable for unevaluated vertex shader geometry.
    // The existing color render remains available and reports that boundary.
    bool opaqueVisibilityApplied = false;
    bool opaqueVisibilityUnsupportedVertexDeformation = false;
    std::uint64_t opaqueVisibilityBytes = 0;
    bool smoothGlassTransportApplied = false;
    std::uint64_t smoothGlassQueryBytes = 0;
    // Successful native shader-program builds performed by this preparation.
    std::uint32_t shaderProgramBuildCount = 0;
    // Private float32 rough-energy receipt, certified only after a complete scan.
    bool roughGlassReceiptComplete = false;
    std::uint64_t roughGlassCoveredPixels = 0;
    std::uint64_t roughGlassReceiptBytes = 0;
    double roughGlassMaximumEnergyResidual = 0;
    double roughGlassMaximumUnresolvedPower = 0;
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
    std::uint32_t roughGlassScatteringVersion = 1;
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
    // Renderer-owned admitted companion, separate from the released GPU block.
    std::optional<surfacematerial::SpectralReflectanceDescription> spectralReflectance;
    // Exact authored ownership for scene-receiver transport, outside GPU blocks.
    HarmonicMIDI::grid::Scene3DId authoredScene {};
    HarmonicMIDI::grid::SceneMaterialId authoredMaterial {};
    std::uint64_t sceneRevision = 0;
    std::uint64_t structuralRevision = 0;
    std::uint64_t evaluationRevision = 0;
    std::uint64_t materialRevision = 0;
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
        && parameters.transmissionIorClearcoat[2] >= 0.0f && parameters.transmissionIorClearcoat[2] <= 1.0f
        && parameters.transmissionIorClearcoat[3] >= 0.0f
        && parameters.transmissionIorClearcoat[3] <= surfacematerial::kMaximumCauchyB
        && parameters.identifiers[1] <= static_cast<std::uint32_t>(surfacematerial::TransportMode::ThinInterface)
        && (parameters.identifiers[1] == 1 || parameters.transmissionIorClearcoat[3] == 0.0f);
}

inline bool nativeSurfaceRequiresBlend(const NativeFixtureSurfaceMaterialProgram* program) noexcept
{
    return program != nullptr && program->kind == NativeFixtureMaterialKind::SurfacePbr
        && program->parameters.identifiers[1] == static_cast<std::uint32_t>(surfacematerial::TransportMode::ThinSurface)
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
            && (program.roughGlassScatteringVersion==1 || (program.roughGlassScatteringVersion==2
                && program.parameters.identifiers[1]==static_cast<std::uint32_t>(surfacematerial::TransportMode::ClosedVolume)))
            && (!program.spectralReflectance || surfacematerial::validSpectralReflectance(*program.spectralReflectance))
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
    std::string_view productDigest = {}, bool graphFrame = false,
    const NativeFixtureSurfaceMaterialProgram* ownership = nullptr)
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
    if (lighting.version == 3)
    {
        result.append(":scene-receiver-v3:");
        appendHex(static_cast<std::uint8_t>(lighting.sceneReceiverTransport), 2);
        appendFloat(lighting.traceMaximumDistance);
        appendFloat(lighting.bouncePlaneHeight);
        appendFloat(lighting.bounceMaximumDistance);
        appendHex(ownership ? ownership->authoredScene.value : 0, 8);
        appendHex(ownership ? ownership->object.value : 0, 8);
        appendHex(ownership ? ownership->authoredMaterial.value : 0, 8);
        appendHex(ownership ? ownership->sceneRevision : 0, 16);
        appendHex(ownership ? ownership->structuralRevision : 0, 16);
        appendHex(ownership ? ownership->evaluationRevision : 0, 16);
        appendHex(ownership ? ownership->materialRevision : 0, 16);
    }
    appendHex(diffractionmaterial::kMaximumLightingPaths, 2);
    appendHex(diffractionmaterial::kMaximumIndirectBounces, 2);
    appendHex(diffractionmaterial::kMaximumSpectralSamples, 4);
    appendFloat(diffractionmaterial::physicalcheckpoint::kMaximumSpectralRadiance);
    appendHex(lighting.pathCount, 2);
    for (std::size_t pathIndex = 0; pathIndex < (lighting.version == 3 ? lighting.paths.size() : lighting.pathCount); ++pathIndex)
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
// Discrete grating orders have no finite-lobe camera shader. Only the guarded
// smooth-glass transport prerequisite may request their exact zero-slope IR.
enum class NativeFixtureDiffractionConsumer : std::uint8_t
{ FiniteLobe=0, SmoothGlassDiscreteOrders=1 };

// Material programs cross a trust boundary before backend-owned resources are
// allocated. Validate the packed block itself rather than relying on the IR
// admission that originally produced it.
inline bool validNativeFixtureDiffractionParameters (
    const diffractionmaterial::PhysicalDiffractionGpuParameters& value,
    NativeFixtureDiffractionConsumer consumer=NativeFixtureDiffractionConsumer::FiniteLobe) noexcept
{
    using namespace diffractionmaterial;
    const bool discrete=consumer==NativeFixtureDiffractionConsumer::SmoothGlassDiscreteOrders;
    if(!discrete&&consumer!=NativeFixtureDiffractionConsumer::FiniteLobe)return false;
    if(discrete&&(value.secondaryGeometry.w!=float(GratingLattice::OneDimensional)
        ||value.control.y!=float(GrooveProfile::BinaryRectangular)||value.control.w!=float(CoatingModel::Uncoated)
        ||value.microstructure.w!=1||value.control.x!=1||value.roughness.x!=0||value.roughness.y!=0
        ||value.grooveField.x!=float(GrooveFieldMode::Constant)))return false;
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
        || ! bounded (value.roughness.y, discrete?0.0f:kMinimumNativeFixtureDiffractionRmsSlope,
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
    HarmonicMIDI::grid::SceneObjectId expectedObject,
    bool sceneReceiverConsumer = false,
    NativeFixtureDiffractionConsumer consumer=NativeFixtureDiffractionConsumer::FiniteLobe) noexcept
{
    using Program = NativeFixtureSurfaceMaterialProgram;
    if(consumer==NativeFixtureDiffractionConsumer::SmoothGlassDiscreteOrders
        &&(!sceneReceiverConsumer||!program.diffractionLightingAdmission
            ||program.diffractionLightingAdmission->description().version!=3||program.diffractionPathCount!=1
            ||program.diffractionPatternMask||program.diffractionFoilMaximumEvaluations||program.diffractionOccupancyRectangleCount
            ||program.vertexProgram||program.frameEndpoint||!program.diffractionProductDigest.empty()))return false;
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
        // Each backend opts in only after consuming the shared triangle table.
        && (program.diffractionLightingAdmission->description().version == 1
            || program.diffractionLightingAdmission->description().version == 2
            || (sceneReceiverConsumer && program.diffractionLightingAdmission->description().version == 3
                && program.authoredScene.isValid() && program.authoredMaterial.isValid()
                && program.sceneRevision != 0 && program.structuralRevision != 0
                && program.evaluationRevision != 0 && program.materialRevision == program.structuralRevision))
        && program.diffractionLightingAdmission->description().pathCount
            == program.diffractionPathCount
        && program.diffractionMaximumBounceDepth
            == (program.diffractionLightingAdmission->hasIndirectBounce()
                    ? diffractionmaterial::kMaximumIndirectBounces : 0)
        && program.diffractionMaximumBounceDepth
            <= diffractionmaterial::kMaximumIndirectBounces
        && std::all_of(program.diffractionPaths.begin(),
                       program.diffractionPaths.begin() + program.diffractionPathCount,
                       [&program,consumer] (const auto& path)
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
                           return validNativeFixtureDiffractionParameters(path.material,consumer)
                               && std::isfinite(length)
                               && std::abs(length - 1.0f) <= 1.0e-4f
                               && (program.diffractionLightingAdmission->description().version != 1
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
            && (program.diffractionLightingAdmission->description().version != 1
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
               program.baseColorSource == Program::BaseColorSource::GraphFrameSrgbTexture, &program);
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

// Mixed admission keeps the grating as the root program and the sorted,
// disjoint Surface programs as its flat object table. Single-program callers
// retain the released object-zero diffraction route.
inline bool validNativeFixtureMaterialPrograms(const NativeFixtureSurfaceMaterialProgram& root,
    const HarmonicMIDI::grid::Visual3DScene& scene, NativeFixtureMaterialBackend backend,
    bool sceneReceiverConsumer = false,
    NativeFixtureDiffractionConsumer consumer=NativeFixtureDiffractionConsumer::FiniteLobe)
{
    using namespace HarmonicMIDI::grid;
    if (root.kind == NativeFixtureMaterialKind::SurfacePbr)
        return consumer==NativeFixtureDiffractionConsumer::FiniteLobe&&validNativeSurfaceObjectPrograms(root, scene, backend);
    const auto* object = visual3d_detail::findById(scene.objects, scene.objectCount, root.object);
    if (!object || object->indexCount == 0 || root.vertexProgram || root.spectralReflectance
        || !validNativeFixtureDiffractionProgram(root, backend, object->id, sceneReceiverConsumer,consumer)) return false;
    if (root.objectPrograms.empty()) return root.object == scene.objects[0].id;
    if (root.objectPrograms.size() >= Visual3DScene::kMaxObjects) return false;
    std::uint32_t previous = 0;
    std::optional<surfacematerialbinding::TextureSlotBinding::GraphFrameEndpoint> frame;
    if (root.baseColorSource == NativeFixtureSurfaceMaterialProgram::BaseColorSource::GraphFrameSrgbTexture) {
        if (!root.frameEndpoint) return false;
        frame = root.frameEndpoint;
    }
    for (const auto& surface : root.objectPrograms)
    {
        if (!surface || surface->object.value <= previous || surface->object == root.object
            || !surface->objectPrograms.empty()) return false;
        previous = surface->object.value;
        if (surface->baseColorSource == NativeFixtureSurfaceMaterialProgram::BaseColorSource::GraphFrameSrgbTexture) {
            if (!surface->frameEndpoint || (frame && !(*frame == *surface->frameEndpoint))) return false;
            frame = surface->frameEndpoint;
        }
    }
    auto surfaces = *root.objectPrograms.front();
    surfaces.objectPrograms.assign(root.objectPrograms.begin() + 1, root.objectPrograms.end());
    return validNativeSurfaceObjectPrograms(surfaces, scene, backend);
}

inline bool nativeFixtureDimensionsWithinBounds (std::uint32_t width,
                                                 std::uint32_t height) noexcept
{
    return width > 0 && height > 0
        && width <= kMaximumNativeFixtureExtent
        && height <= kMaximumNativeFixtureExtent
        && static_cast<std::uint64_t> (width) * height <= kMaximumNativeFixturePixels;
}

inline constexpr std::size_t kMaximumNativeDiffractionTriangles = 8;
inline constexpr unsigned kNativeDiffractionSubdivisions = 2;
inline constexpr std::size_t kMaximumNativeDiffractionSamples =
    kMaximumNativeDiffractionTriangles * kNativeDiffractionSubdivisions * kNativeDiffractionSubdivisions;
inline constexpr std::uint64_t kMaximumNativeDiffractionQueries = 268'435'456;
inline constexpr std::uint64_t kMaximumNativeDiffractionQueryWork = 8'589'934'592;
// At the maximum 32 samples, 1080p reserves 201,141,280 queries. A 4K
// frame reserves 804,558,880 and refuses; smaller exact tables can admit 4K.
inline bool nativeFixtureDiffractionQueryWorkWithinBudget(std::uint32_t width,
    std::uint32_t height, std::size_t receiverSamples, std::size_t gratingSamples,
    std::uint64_t closestHitWork) noexcept
{
    if (!nativeFixtureDimensionsWithinBounds(width, height) || receiverSamples == 0 || gratingSamples == 0
        || receiverSamples > kMaximumNativeDiffractionSamples || gratingSamples > kMaximumNativeDiffractionSamples
        || closestHitWork == 0) return false;
    const auto pairs = std::uint64_t(receiverSamples) * gratingSamples;
    const auto queries = std::uint64_t(width) * height *
        (1 + 3 * std::max(receiverSamples, gratingSamples)) + receiverSamples + 2 * pairs;
    return queries <= kMaximumNativeDiffractionQueries
        && closestHitWork <= kMaximumNativeDiffractionQueryWork / queries;
}

inline std::uint64_t nativeFixtureDiffractionLobeEvaluations (
    const NativeFixtureSurfaceMaterialProgram& material,
    std::uint32_t width,
    std::uint32_t height,
    std::size_t sceneSamples = kMaximumNativeDiffractionSamples) noexcept
{
    if (material.kind != NativeFixtureMaterialKind::DiffractionReflective)
        return 0;

    if (material.diffractionPathCount == 0
        || material.diffractionPathCount > diffractionmaterial::kMaximumLightingPaths
        || sceneSamples == 0 || sceneSamples > kMaximumNativeDiffractionSamples)
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
        ? 11ull : material.diffractionLightingAdmission
            && material.diffractionLightingAdmission->description().version == 3
            ? 1ull + sceneSamples : static_cast<std::uint64_t>(material.diffractionPathCount);
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
    std::uint32_t height,
    std::size_t sceneSamples = kMaximumNativeDiffractionSamples) noexcept
{
    if (sceneSamples == 0 || sceneSamples > kMaximumNativeDiffractionSamples) return false;
    if (material.diffractionLightingAdmission && material.diffractionLightingAdmission->description().version == 3
        && !nativeFixtureDiffractionQueryWorkWithinBudget(width,height,sceneSamples,sceneSamples,1)) return false;
    const auto estimate = nativeFixtureDiffractionLobeEvaluations (
        material, width, height, sceneSamples);
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

// Preview callers may receive pending while a backend still compiles. Blocking
// callers never do. Backends that cannot defer ignore the use argument.
enum class NativeFixturePrepareUse : std::uint8_t { Blocking = 0, Preview };

struct NativeFixtureScenePreparation
{
    bool prepared = false;
    std::shared_ptr<const NativeFixtureSceneResources> resources;
    NativeFixtureSceneStats stats {};
    std::string error;
    // Set only for a Preview prepare that has not finished. It is not an error.
    // No resources are returned and no backend cache changes.
    bool pending = false;
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
    // Only an owned decoded PQ conversion may publish this immutable material proof.
    virtual NativeTextureViewDescriptor decodedLinearMaterialDescriptor() const noexcept { return {}; }
    virtual NativeTextureViewDescriptor depthTextureDescriptor() const noexcept { return {}; }
    virtual NativeTextureViewDescriptor passTextureDescriptor(renderpassoutput::Output output) const noexcept
    {
        return output == renderpassoutput::Output::Color ? colorTextureDescriptor()
            : output == renderpassoutput::Output::Depth ? depthTextureDescriptor()
            : NativeTextureViewDescriptor {};
    }
    virtual bool readColorFloatPixels (std::vector<float>& output) const
    {
        output.clear();
        return false;
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
inline bool materialFrameIsLinear(const NativeTextureViewDescriptor& descriptor) noexcept
{
    return descriptor.complete() && descriptor.format == NativeTexturePixelFormat::Rgba16Float
        && descriptor.colorSpace == colortransform::ColorSpace::LinearSRGB
        && descriptor.transfer == colortransform::TransferFunction::Linear
        && descriptor.rowOrder == NativeTextureRowOrder::TopFirst;
}

inline bool matchesDecodedLinearMaterialDescriptor(const NativeTextureViewDescriptor& descriptor,
    const NativeTextureViewDescriptor& proof) noexcept
{
    return materialFrameIsLinear(proof) && descriptor.backend == proof.backend
        && descriptor.viewKind == proof.viewKind && descriptor.format == proof.format
        && descriptor.imageHandle == proof.imageHandle && descriptor.textureViewHandle == proof.textureViewHandle
        && descriptor.width == proof.width && descriptor.height == proof.height
        && descriptor.sampleCount == proof.sampleCount && descriptor.sampledBinding == proof.sampledBinding
        && descriptor.deviceOrContextIdentity == proof.deviceOrContextIdentity
        && descriptor.rendererGeneration == proof.rendererGeneration
        && descriptor.colorSpace == proof.colorSpace && descriptor.transfer == proof.transfer
        && descriptor.rowOrder == proof.rowOrder;
}

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
                && descriptor.format == NativeTexturePixelFormat::Bgra8Unorm)
            || (materialFrameIsLinear(descriptor)
                && matchesDecodedLinearMaterialDescriptor(descriptor,frame->decodedLinearMaterialDescriptor())))
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
// Compact immutable predecessor data, keyed by canonical note identity. A
// missing identity is a new instance and must produce neutral per-note Motion.
struct NativeSceneNoteMotionTransforms final
{
    std::array<std::int64_t, visualnoteinstancing::kMaximumInstances> identities {};
    std::array<std::array<float, 4>, visualnoteinstancing::kMaximumInstances> transforms {};
    std::size_t count = 0;
    std::optional<std::array<float, 4>> find(std::int64_t identity) const noexcept
    {
        if (identity == 0 || count > identities.size()) return {};
        for (std::size_t i = 0; i < count; ++i)
            if (identities[i] == identity) return transforms[i];
        return {};
    }
};

struct NativeSceneMotionOwner final
{
    // Source, deformation, animation clip, structural revision, project/helper
    // generations, render clip and selected node. Composition alone appends its
    // parameter epoch; legacy evaluated controls remain absent.
    std::array<std::uint64_t, 8> identity {};
    std::uint64_t evaluationRevision = 0;
    bool operator==(const NativeSceneMotionOwner& other) const noexcept
    { return identity == other.identity && evaluationRevision == other.evaluationRevision; }
};

// One checked static-local world: the authored chain with the local correction
// and no runtime overlay. Each consumer applies its runtime transform on the
// left, exactly as it does for an authored world.
struct NativeImportedObjectWorld final
{
    HarmonicMIDI::grid::SceneObjectId object {};
    std::array<float, 16> world {};
};
// Each entry names one distinct drawn object of the admitted scene. A drawn
// object without an entry keeps its authored world.
using NativeImportedObjectWorlds = std::vector<NativeImportedObjectWorld>;

// The checked world of one object, or null when the list does not name it.
inline const NativeImportedObjectWorld* nativeObjectWorldEntry(const NativeImportedObjectWorlds* worlds,
    HarmonicMIDI::grid::SceneObjectId object) noexcept
{
    if (worlds)
        for (const auto& entry : *worlds)
            if (entry.object == object) return &entry;
    return nullptr;
}

// Matches only when both lists name the same objects in the same order, with the same worlds.
inline bool sameNativeImportedObjectWorlds(
    const std::shared_ptr<const NativeImportedObjectWorlds>& a,
    const std::shared_ptr<const NativeImportedObjectWorlds>& b) noexcept
{
    if (a == b) return true;
    if (!a || !b || a->size() != b->size()) return false;
    for (std::size_t i = 0; i < a->size(); ++i)
        if ((*a)[i].object != (*b)[i].object || (*a)[i].world != (*b)[i].world) return false;
    return true;
}

// Structural identity of a Motion predecessor. The predecessor keeps the worlds it
// was rendered with, so only the drawn object set has to match, in any order.
// An authored chain (null) matches only another authored chain.
// Each object id in a must have an entry in b, and the sizes must match. Worlds are not compared.
inline bool sameNativeImportedObjectWorldSet(
    const std::shared_ptr<const NativeImportedObjectWorlds>& a,
    const std::shared_ptr<const NativeImportedObjectWorlds>& b) noexcept
{
    if (a == b) return true;
    if (!a || !b || a->size() != b->size()) return false;
    for (const auto& entry : *a)
        if (!nativeObjectWorldEntry(b.get(), entry.object)) return false;
    return true;
}

// Admits only affine finite worlds with distinct valid ids. Scene membership is
// checked where the admitted scene is known, by nativeImportedObjectWorldsCover.
inline bool validNativeImportedObjectWorlds(const NativeImportedObjectWorlds& worlds) noexcept
{
    if (worlds.empty() || worlds.size() > HarmonicMIDI::grid::Visual3DScene::kMaxObjects) return false;
    for (std::size_t i = 0; i < worlds.size(); ++i)
    {
        const auto& entry = worlds[i];
        if (!entry.object.isValid()) return false;
        for (std::size_t j = 0; j < i; ++j)
            if (worlds[j].object == entry.object) return false;
        if (!std::all_of(entry.world.begin(), entry.world.end(), [](float value) { return std::isfinite(value); })
            || entry.world[3] != 0 || entry.world[7] != 0 || entry.world[11] != 0 || entry.world[15] != 1)
            return false;
    }
    return true;
}

// A non-empty subset of the drawn objects of the scene. Every entry must name a
// drawn object, and distinct ids come from validFixtureRuntimeInputs, which both
// raster backends run first. A drawn object without an entry keeps its authored
// world, so a Scene Compose lists only the objects under its moved leaf.
inline bool nativeImportedObjectWorldsCover(const HarmonicMIDI::grid::Visual3DScene& scene,
    const NativeImportedObjectWorlds& worlds) noexcept
{
    if (worlds.empty()) return false;
    for (const auto& entry : worlds)
    {
        const auto* object = HarmonicMIDI::grid::visual3d_detail::findById(
            scene.objects, scene.objectCount, entry.object);
        if (!object || !object->indexCount) return false;
    }
    return true;
}

// Every drawn object has exactly one entry, and every entry names a drawn object.
// A single leaf lists all of its drawn objects this way. A Scene Compose keeps the
// subset cover above, because it lists only the objects under its moved node.
inline bool nativeImportedObjectWorldsExactCover(const HarmonicMIDI::grid::Visual3DScene& scene,
    const NativeImportedObjectWorlds& worlds) noexcept
{
    if (!nativeImportedObjectWorldsCover(scene, worlds)) return false;
    std::size_t drawn = 0;
    for (std::size_t i = 0; i < scene.objectCount; ++i)
    {
        const auto& object = scene.objects[i];
        if (!object.indexCount) continue;
        ++drawn;
        if (!nativeObjectWorldEntry(&worlds, object.id)) return false;
    }
    return drawn == worlds.size();
}

// The world one raster or visibility pass uses for an object: the checked
// static-local world when the candidate lists it, else the authored chain.
// Coverage is verified once per render. An object the candidate does not list
// keeps the authored chain.
inline std::array<float, 16> nativeObjectWorldMatrix(const NativeImportedObjectWorlds* worlds,
    const HarmonicMIDI::grid::SceneObjectRecord& object, const std::array<float, 16>& authored) noexcept
{
    if (worlds)
        for (const auto& entry : *worlds)
            if (entry.object == object.id) return entry.world;
    return authored;
}

struct NativeSceneMotionSample final
{
    struct ImportedNodeWorldMatrix final
    {
        // One-based GLB node identity, including ancestors without mesh draws.
        std::uint64_t nodeStableId = 0;
        std::array<float, 16> world {};
    };
    using ImportedNodeWorldMatrices = std::vector<ImportedNodeWorldMatrix>;

    std::array<float, 3> objectTranslationOffset {};
    std::array<float, 3> objectRotationDegrees {};
    float objectScale = 1.0f;
    std::array<float, 3> cameraTranslationOffset {};
    std::optional<HarmonicMIDI::grid::SceneCameraRecord> cameraOverride;
    std::shared_ptr<const visualdeformation::AnimationDeformationSnapshot> deformation;
    float morphWeight = 1.0f;
    std::shared_ptr<const NativeSceneNoteMotionTransforms> noteTransforms;
    NativeSceneMotionOwner owner;
    // Canonical sampled worlds before inverse object/bind multiplication.
    // Outer graph and note transforms remain in their existing sample fields.
    // Rigid synthetic skins already apply these worlds through jointPalette;
    // consumers must not apply them a second time to deformed vertices.
    std::shared_ptr<const ImportedNodeWorldMatrices> importedNodeWorldMatrices {};
    std::shared_ptr<const NativeDeformationCompositionFrame> compositionFrame {};
    std::shared_ptr<const videohelper::modelpayload::NativeSceneAnimationCompositionFrame> localSceneFrame {};
    // The prior overlay selector belongs to the prior sample.
    std::uint64_t objectNodeStableId = 0;
    // Prior inputs of a graph vertex program, which is a pure function of the
    // vertex attributes, time and spectrum. The motion pass runs it again here.
    float timeSeconds = 0.0f;
    std::array<float, 64> vertexSpectrum {};
    // Prior positions of a topology-stable deformed mesh, one per scene vertex.
    // Timeline Geometry3D keeps vertex order, so index i matches vertex i.
    std::shared_ptr<const std::vector<HarmonicMIDI::grid::SceneVec3>> scenePositions {};
    // Prior runtime camera adjustments, evaluated with the prior camera record.
    std::array<float, 3> cameraRotationDegrees {};
    float cameraVerticalFovDegrees = 0.0f;
    // A predecessor is reused only with the same checked static-local worlds.
    std::shared_ptr<const NativeImportedObjectWorlds> objectWorlds {};
};

inline bool validNativeCompositionWorldTable(
    const std::shared_ptr<const NativeDeformationCompositionFrame>&,
    const std::shared_ptr<const NativeSceneMotionSample::ImportedNodeWorldMatrices>&) noexcept;

// One modulated directional or point light. Flags name the fields that differ
// from the authored record; unflagged fields keep the sampled value. Color and
// intensity are absolute. The position offset is additive to the translation.
// The rotation offset is composed as offset * authored, then normalized.
struct NativeImportedLightOverride final
{
    std::size_t index = 0;
    bool hasColor = false;
    HarmonicMIDI::grid::SceneVec3 color {};
    bool hasIntensity = false;
    float intensity = 1.0f;
    // Point lights only.
    bool hasPositionOffset = false;
    HarmonicMIDI::grid::SceneVec3 positionOffset {};
    // Directional lights only.
    bool hasRotationOffset = false;
    HarmonicMIDI::grid::SceneQuaternion rotationOffset {};

    bool any() const noexcept
    {
        return hasColor || hasIntensity || hasPositionOffset || hasRotationOffset;
    }
};

// Kind-agnostic checks that need no scene: at most kMaxLights entries, strictly
// increasing indexes, one flagged field each, and finite values. The target
// light is checked where the backend reads it, because the list may be empty
// and the scene's light count is not visible here.
inline bool validNativeImportedLightOverrideList(
    const std::vector<NativeImportedLightOverride>& overrides) noexcept
{
    using namespace HarmonicMIDI::grid;
    if (overrides.size() > Visual3DScene::kMaxLights) return false;
    bool first = true;
    std::size_t previous = 0;
    for (const auto& o : overrides) {
        if (!first && o.index <= previous) return false;
        first = false;
        previous = o.index;
        if (!o.any()) return false;
        if (o.hasColor && !(std::isfinite(o.color.x) && std::isfinite(o.color.y) && std::isfinite(o.color.z)
                && o.color.x >= 0.0f && o.color.y >= 0.0f && o.color.z >= 0.0f)) return false;
        if (o.hasIntensity && !(std::isfinite(o.intensity) && o.intensity >= 0.0f)) return false;
        if (o.hasPositionOffset && !(std::isfinite(o.positionOffset.x) && std::isfinite(o.positionOffset.y)
                && std::isfinite(o.positionOffset.z))) return false;
        if (o.hasRotationOffset) {
            const auto& q = o.rotationOffset;
            const double norm = double(q.x)*q.x + double(q.y)*q.y + double(q.z)*q.z + double(q.w)*q.w;
            if (!(std::isfinite(q.x) && std::isfinite(q.y) && std::isfinite(q.z) && std::isfinite(q.w))
                || !(std::isfinite(norm) && norm > 0.0)) return false;
        }
    }
    return true;
}

// The target must be a directional or point light, and each flagged field must
// apply to that kind. Position offsets are point-only; rotation offsets are directional-only.
inline bool validNativeImportedLightOverrideTarget(const NativeImportedLightOverride& o,
    const HarmonicMIDI::grid::SceneLightRecord& target) noexcept
{
    using namespace HarmonicMIDI::grid;
    const bool directional = target.kind == SceneLightKind::Directional;
    const bool point = target.kind == SceneLightKind::Point;
    if (!directional && !point) return false;
    if (o.hasPositionOffset && !point) return false;
    if (o.hasRotationOffset && !directional) return false;
    return true;
}

// Checks the list against a known light list, such as a non-empty animatedLights.
inline bool validNativeImportedLightOverrides(
    const std::vector<NativeImportedLightOverride>& overrides,
    const std::vector<HarmonicMIDI::grid::SceneLightRecord>& lights) noexcept
{
    if (!validNativeImportedLightOverrideList(overrides)) return false;
    return std::all_of(overrides.begin(), overrides.end(), [&](const NativeImportedLightOverride& o) {
        return o.index < lights.size() && validNativeImportedLightOverrideTarget(o, lights[o.index]);
    });
}

// Applies one override to a sampled record. Callers apply overrides after the
// sampler has produced the lights, because sampling overwrites the output.
inline void applyNativeImportedLightOverride(
    const NativeImportedLightOverride& override,
    HarmonicMIDI::grid::SceneLightRecord& light) noexcept
{
    if (override.hasColor) light.color = override.color;
    if (override.hasIntensity) light.intensity = override.intensity;
    if (override.hasPositionOffset) {
        light.transform.translation.x += override.positionOffset.x;
        light.transform.translation.y += override.positionOffset.y;
        light.transform.translation.z += override.positionOffset.z;
    }
    if (override.hasRotationOffset) {
        const auto& a = override.rotationOffset;
        const auto& b = light.transform.rotation;
        const double x = double(a.w)*b.x + double(a.x)*b.w + double(a.y)*b.z - double(a.z)*b.y;
        const double y = double(a.w)*b.y - double(a.x)*b.z + double(a.y)*b.w + double(a.z)*b.x;
        const double z = double(a.w)*b.z + double(a.x)*b.y - double(a.y)*b.x + double(a.z)*b.w;
        const double w = double(a.w)*b.w - double(a.x)*b.x - double(a.y)*b.y - double(a.z)*b.z;
        const double norm = std::sqrt(x*x + y*y + z*z + w*w);
        if (std::isfinite(norm) && norm > 0.0) {
            light.transform.rotation = { static_cast<float>(x/norm), static_cast<float>(y/norm),
                static_cast<float>(z/norm), static_cast<float>(w/norm) };
        }
    }
}

// Field-by-field identity for cache and reuse checks. Unflagged fields are
// compared too, so a changed or removed flag never reuses a cached frame.
inline bool sameNativeImportedLightOverride(
    const NativeImportedLightOverride& a, const NativeImportedLightOverride& b) noexcept
{
    const auto vec=[](const HarmonicMIDI::grid::SceneVec3& x, const HarmonicMIDI::grid::SceneVec3& y) {
        return x.x==y.x && x.y==y.y && x.z==y.z;
    };
    const auto quat=[](const HarmonicMIDI::grid::SceneQuaternion& x, const HarmonicMIDI::grid::SceneQuaternion& y) {
        return x.x==y.x && x.y==y.y && x.z==y.z && x.w==y.w;
    };
    return a.index==b.index
        && a.hasColor==b.hasColor && vec(a.color,b.color)
        && a.hasIntensity==b.hasIntensity && a.intensity==b.intensity
        && a.hasPositionOffset==b.hasPositionOffset && vec(a.positionOffset,b.positionOffset)
        && a.hasRotationOffset==b.hasRotationOffset && quat(a.rotationOffset,b.rotationOffset);
}

inline bool sameNativeImportedLightOverrides(
    const std::vector<NativeImportedLightOverride>& a, const std::vector<NativeImportedLightOverride>& b) noexcept
{
    return a.size()==b.size()
        && std::equal(a.begin(),a.end(),b.begin(),sameNativeImportedLightOverride);
}

struct NativeImportedSceneRuntimeInputs final
{
    // Explicit HDR export working representation. Ordinary preview stays SDR.
    bool linearColor = false;
    // Exact authored selection copied for this draw, never a native cache key.
    std::optional<visualimportedscenerender::GlassCausticSelection> glassCaustic;
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
    // Runtime camera rotation in degrees, composed like the object rotation.
    std::array<float, 3> cameraRotationDegrees {};
    // Zero keeps the authored vertical field of view; otherwise degrees in [1, 179].
    float cameraVerticalFovDegrees = 0.0f;
    std::optional<HarmonicMIDI::grid::SceneCameraRecord> cameraOverride;
    std::optional<HarmonicMIDI::grid::SceneLightRecord> lightOverride;
    // Sampled imported records are subordinate to explicit graph overrides.
    std::optional<HarmonicMIDI::grid::SceneCameraRecord> animatedCamera;
    std::vector<HarmonicMIDI::grid::SceneLightRecord> animatedLights;
    // Modulated fields of animatedLights, applied after sampling. Sorted by index.
    std::vector<NativeImportedLightOverride> lightOverrides;
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
    std::shared_ptr<const NativeDeformationCompositionFrame> compositionFrame {};
    std::shared_ptr<const videohelper::modelpayload::NativeSceneAnimationCompositionFrame> localSceneFrame {};
    // Checked static-local worlds of the admitted scene, or null for authored
    // worlds. Only the single-leaf static candidate supplies them.
    std::shared_ptr<const NativeImportedObjectWorlds> objectWorlds {};
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

inline std::shared_ptr<const NativeSceneNoteMotionTransforms> sceneNoteMotionTransforms(
    const NativeImportedSceneRuntimeInputs& inputs);

// This cap bounds retained snapshots, not the asset or native geometry. Each
// existing owner retains at most current+previous per Preview/Export history.
inline constexpr std::size_t kNativeSceneMotionMaximumSnapshotBytes = 4 * 1024 * 1024;
inline bool sceneMotionSnapshotFits(
    const std::shared_ptr<const visualdeformation::AnimationDeformationSnapshot>& snapshot,
    std::size_t additionalBytes = 0) noexcept
{
    if (additionalBytes > kNativeSceneMotionMaximumSnapshotBytes) return false;
    if (!snapshot) return true;
    if (sizeof(*snapshot) > kNativeSceneMotionMaximumSnapshotBytes - additionalBytes) return false;
    std::size_t bytes = sizeof(*snapshot) + additionalBytes;
    const auto charge = [&](std::size_t count, std::size_t stride) {
        if (count > (kNativeSceneMotionMaximumSnapshotBytes - bytes) / stride) return false;
        bytes += count * stride; return true;
    };
    if (!charge(snapshot->jointTransforms().capacity(), sizeof(visualdeformation::JointTransformEvaluation))
        || !charge(snapshot->morphWeights().capacity(), sizeof(visualdeformation::MorphWeightEvaluation))) return false;
    for (const auto& morph : snapshot->morphWeights())
        if (!charge(1, sizeof(visualdeformation::MorphWeightSample))
            || !charge(morph.values().capacity(), sizeof(float))
            || !charge(morph.targets().capacity(), sizeof(visualdeformation::MorphTargetId))) return false;
    return true;
}

// A missing table is compatible with legacy samples. Imported owners separately
// require the full source-specific table when object animation needs it.
inline bool validSceneMotionImportedNodeWorldMatrices(
    const NativeSceneMotionSample& sample, bool required = false) noexcept
{
    if (!validNativeCompositionRetention(sample.compositionFrame,sample.deformation)) return false;
    const auto& table = sample.importedNodeWorldMatrices;
    if (sample.compositionFrame && !validNativeCompositionWorldTable(sample.compositionFrame,table)) return false;
    if (!table) return !required && sceneMotionSnapshotFits(sample.deformation);
    if (table->capacity() > (kNativeSceneMotionMaximumSnapshotBytes - sizeof(*table))
            / sizeof(NativeSceneMotionSample::ImportedNodeWorldMatrix)
        || !sceneMotionSnapshotFits(sample.deformation, sizeof(*table)
            + table->capacity() * sizeof(NativeSceneMotionSample::ImportedNodeWorldMatrix))) return false;
    std::uint64_t previous = 0;
    for (const auto& entry : *table)
    {
        if (entry.nodeStableId <= previous
            || !std::all_of(entry.world.begin(), entry.world.end(),
                [](float value) { return std::isfinite(value); })) return false;
        previous = entry.nodeStableId;
    }
    return true;
}

inline NativeSceneMotionSample sceneMotionSample(const NativeImportedSceneRuntimeInputs& inputs,
    std::shared_ptr<const visualdeformation::AnimationDeformationSnapshot> deformation = {},
    NativeSceneMotionOwner owner = {})
{
    return { inputs.objectTranslationOffset, inputs.objectRotationDegrees,
        inputs.objectScale, inputs.cameraTranslationOffset,
        inputs.cameraOverride ? inputs.cameraOverride : inputs.animatedCamera,
        std::move(deformation), inputs.morphWeight, sceneNoteMotionTransforms(inputs), owner,
        {}, inputs.compositionFrame, inputs.localSceneFrame, inputs.objectNodeStableId,
        inputs.timeSeconds, inputs.vertexSpectrum, {}, inputs.cameraRotationDegrees,
        inputs.cameraVerticalFovDegrees, inputs.objectWorlds };
}

inline bool validSceneMotionPositions(
    const std::shared_ptr<const std::vector<HarmonicMIDI::grid::SceneVec3>>& positions) noexcept
{
    return !positions || (positions->size() <= HarmonicMIDI::grid::Visual3DScene::kMaxVertices
        && std::all_of(positions->begin(), positions->end(), [](const HarmonicMIDI::grid::SceneVec3& value)
            { return HarmonicMIDI::grid::visual3d_detail::finite(value); }));
}

inline std::optional<canonicalblockc::FrameKey> sceneMotionNoteOwner(
    const NativeImportedSceneRuntimeInputs& inputs)
{
    if (!inputs.noteInstanceMapping || !inputs.canonicalBlockCFrame) return {};
    auto key = inputs.canonicalBlockCFrame->key;
    // Frame position is handled by exact rational adjacency. Only the source
    // and timeline epochs participate in predecessor ownership.
    key.frame = 0; key.beat = 0; key.fps = 0;
    return key;
}

// Owned by the existing scene owner. A failed render never commits a sample.
// Identity, mapping, first frame, seek/loop, rate and extent changes yield zero.
// Checked static-local worlds may change between frames; a changed object set yields zero.
struct NativeSceneMotionHistory final
{
    std::optional<NativeSceneMotionSample> current, previous;
    std::int64_t frame = 0;
    std::uint32_t numerator = 0, denominator = 0, width = 0, height = 0;
    NativeSceneMotionOwner owner;
    std::string mapping;
    std::optional<canonicalblockc::FrameKey> noteOwner;
    std::optional<NativeSceneMotionSample> predecessor(std::int64_t next,
        std::uint32_t num, std::uint32_t den, std::uint32_t w, std::uint32_t h,
        NativeSceneMotionOwner expectedOwner = {},
        const NativeImportedSceneRuntimeInputs& inputs = {}) const
    {
        if (!current || num == 0 || den == 0 || num != numerator || den != denominator
            || w != width || h != height || !(owner == expectedOwner)
            || mapping != (inputs.noteInstanceMapping ? visualnoteinstancing::encode(*inputs.noteInstanceMapping) : "")
            || !(noteOwner == sceneMotionNoteOwner(inputs))
            || !sameNativeCompositionOwner(current->compositionFrame,inputs.compositionFrame)
            || !sameNativeImportedObjectWorldSet(current->objectWorlds,inputs.objectWorlds)) return {};
        // A repeated frame returns its own predecessor, which keeps its worlds and must match too.
        if (next == frame) return previous && sameNativeImportedObjectWorldSet(previous->objectWorlds,inputs.objectWorlds)
            ? previous : std::nullopt;
        return frame < std::numeric_limits<std::int64_t>::max() && next == frame + 1
            ? current : std::nullopt;
    }
    bool commit(std::int64_t next, std::uint32_t num, std::uint32_t den,
        std::uint32_t w, std::uint32_t h, const NativeImportedSceneRuntimeInputs& inputs,
        std::shared_ptr<const visualdeformation::AnimationDeformationSnapshot> deformation = {},
        NativeSceneMotionOwner nextOwner = {},
        std::shared_ptr<const std::vector<HarmonicMIDI::grid::SceneVec3>> scenePositions = {})
    {
        if (num == 0 || den == 0 || w == 0 || h == 0 || !sceneMotionSnapshotFits(deformation)
            || !validSceneMotionPositions(scenePositions)
            || !validNativeCompositionRetention(inputs.compositionFrame,deformation)
            || (deformation && !(deformation->time() == visualdeformation::RationalFrameTime {next, num, den}))) return false;
        auto sample = sceneMotionSample(inputs, std::move(deformation), nextOwner);
        sample.scenePositions = std::move(scenePositions);
        auto predecessorSample = predecessor(next, num, den, w, h, nextOwner, inputs);
        previous = std::move(predecessorSample);
        current = std::move(sample);
        frame = next; numerator = num; denominator = den; width = w; height = h;
        owner = nextOwner;
        mapping = inputs.noteInstanceMapping ? visualnoteinstancing::encode(*inputs.noteInstanceMapping) : "";
        noteOwner = sceneMotionNoteOwner(inputs);
        return true;
    }
};

using NativeFixtureSceneRuntimeInputs = NativeImportedSceneRuntimeInputs;
using NativeDeformationRuntimeInputs = NativeImportedSceneRuntimeInputs;

// Solid bodies appended to a composed scene take ids above this base. Their
// node bits would otherwise alias an authored node's range, so an object-targeted
// transform node never reaches them.
inline constexpr std::uint32_t kAppendedSolidObjectIdBase = 0xFFFF0000u;

inline bool runtimeTargetsObject(const NativeImportedSceneRuntimeInputs& inputs,
                                const HarmonicMIDI::grid::SceneObjectRecord& object) noexcept
{
    if (inputs.objectNodeStableId != 0 && object.id.value > kAppendedSolidObjectIdBase) return false;
    if (inputs.compositionFrame)
        return nativeCompositionTargetsObject(inputs.compositionFrame,inputs.objectNodeStableId,object.id);
    return inputs.objectNodeStableId == 0
        || (object.id.value - 1u) / 65536u + 1u == inputs.objectNodeStableId;
}

inline bool resolveDiffractionRuntimeProgram(
    const std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>& authored,
    const std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>& authoredSource,
    const NativeImportedSceneRuntimeInputs& inputs,
    std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>& resolved, std::string& error,
    bool sceneReceiverConsumer = false)
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
        || !validNativeFixtureDiffractionProgram(*value, authored->backend, authored->object, sceneReceiverConsumer)
        || value->objectPrograms != authoredSource->objectPrograms
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
    if (authored->objectPrograms.empty()) resolved = value;
    else {
        auto mixed = std::make_shared<NativeFixtureSurfaceMaterialProgram>(*value);
        mixed->objectPrograms = authored->objectPrograms;
        resolved = std::move(mixed);
    }
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
        if (materialFrameIsLinear(descriptor) && !inputs.linearColor)
        { error = "Scene-linear material Frame requires the linear Scene3D path"; return false; }
        if (inputs.linearColor && !materialFrameIsSrgb(descriptor) && !materialFrameIsLinear(descriptor))
        { error = "Linear Scene3D requires a declared sRGB or scene-linear material Frame; video transfer is not inferred"; return false; }
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
    const bool velocityScale = mapping.schemaVersion == visualnoteinstancing::kVelocityScaleMappingSchemaVersion;
    if (velocityScale && !visualnoteinstancing::valid(mapping)) return {};
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
            case visualnoteinstancing::MappingAxis::PitchBendSemitones: return texel(3, 3);
            case visualnoteinstancing::MappingAxis::NoteAgeBeats: return texel(0, 2);
            case visualnoteinstancing::MappingAxis::RemainingDurationBeats: return texel(0, 3);
            case visualnoteinstancing::MappingAxis::RootFlag: return texel(1, 3);
            case visualnoteinstancing::MappingAxis::MidiNote: return texel(0, 0);
            case visualnoteinstancing::MappingAxis::ResolvedFrequencyHz: return texel(1, 0);
            case visualnoteinstancing::MappingAxis::Prime2Exponent: return texel(2, 0);
            case visualnoteinstancing::MappingAxis::Prime3Exponent: return texel(2, 1);
            case visualnoteinstancing::MappingAxis::Prime5Exponent: return texel(2, 2);
            case visualnoteinstancing::MappingAxis::Prime7Exponent: return texel(2, 3);
            case visualnoteinstancing::MappingAxis::Prime11Exponent: return texel(3, 0);
            case visualnoteinstancing::MappingAxis::Prime13Exponent: return texel(3, 1);
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
        float meshScale = mapping.meshScale;
        if (velocityScale)
        {
            const auto velocity = axisValue(visualnoteinstancing::MappingAxis::Velocity, row);
            if (!std::isfinite(velocity)) return {};
            const auto v = static_cast<double>(std::clamp(velocity, 0.0f, 1.0f));
            // Weighted endpoints avoid cancellation for a descending response.
            meshScale = static_cast<float>(static_cast<double>(mapping.meshScale)
                * ((1.0 - v) * mapping.velocityScaleLow + v * mapping.velocityScaleHigh));
            if (!std::isfinite(meshScale) || meshScale <= 0.0f || meshScale > 10000.0f) return {};
        }
        const std::array<float, 4> transform {
            axisValue(mapping.x, row) * mapping.xScale,
            axisValue(mapping.y, row) * mapping.yScale,
            axisValue(mapping.z, row) * mapping.zScale,
            meshScale
        };
        if (mapping.schemaVersion == visualnoteinstancing::kDirectAxesMappingSchemaVersion || velocityScale)
            for (std::size_t coordinate = 0; coordinate < 3; ++coordinate)
                if (!std::isfinite(transform[coordinate]) || std::abs(transform[coordinate]) > 1000000.0f)
                    return {};
        const auto index = batch.count++;
        batch.transforms[index] = transform;
        batch.identities[index] = identity;
        batch.canonicalRows[index] = static_cast<std::uint16_t>(row);
        batch.appearanceValues[index] = visualnoteinstancing::appearanceValue(
            mapping, identity, axisValue(visualnoteinstancing::MappingAxis::Velocity, row));
    }
    return batch;
}

inline std::shared_ptr<const NativeSceneNoteMotionTransforms> sceneNoteMotionTransforms(
    const NativeImportedSceneRuntimeInputs& inputs)
{
    if (!inputs.noteInstanceMapping || !inputs.canonicalBlockCFrame) return {};
    const auto batch = prepareNativeNoteInstances(inputs);
    auto table = std::make_shared<NativeSceneNoteMotionTransforms>();
    table->count = batch.count;
    table->identities = batch.identities;
    table->transforms = batch.transforms;
    return table;
}

inline auto noteInstanceShaderTransforms(const NativeNoteInstanceBatch& batch) noexcept
{
    auto packed = batch.transforms;
    // Native consumers admit v1-v3, whose scale is shared by the mapping.
    // Its uniform frees w for appearance without another 128-element array.
    for (std::size_t index = 0; index < batch.count; ++index)
        packed[index][3] = batch.appearanceValues[index];
    return packed;
}

// World matrix of one scene object instance: the object's column-major world
// matrix times the instance's scale, rotation and translation. Raster draws and
// opaque visibility both use this, so shadows match the drawn instance.
inline std::array<float, 16> nativeObjectInstanceMatrix(
    const std::array<float, 16>& objectWorld,
    const HarmonicMIDI::grid::SceneTransform3D& instance) noexcept
{
    const auto x = instance.rotation.x, y = instance.rotation.y;
    const auto z = instance.rotation.z, w = instance.rotation.w;
    const auto sx = instance.scale.x, sy = instance.scale.y, sz = instance.scale.z;
    const std::array<float, 16> local {
        (1.0f - 2.0f * (y * y + z * z)) * sx, (2.0f * (x * y + w * z)) * sx,
        (2.0f * (x * z - w * y)) * sx, 0.0f,
        (2.0f * (x * y - w * z)) * sy, (1.0f - 2.0f * (x * x + z * z)) * sy,
        (2.0f * (y * z + w * x)) * sy, 0.0f,
        (2.0f * (x * z + w * y)) * sz, (2.0f * (y * z - w * x)) * sz,
        (1.0f - 2.0f * (x * x + y * y)) * sz, 0.0f,
        instance.translation.x, instance.translation.y, instance.translation.z, 1.0f
    };
    std::array<float, 16> result {};
    for (std::size_t column = 0; column < 4; ++column)
        for (std::size_t row = 0; row < 4; ++row)
            for (std::size_t inner = 0; inner < 4; ++inner)
                result[column * 4 + row] += objectWorld[inner * 4 + row] * local[column * 4 + inner];
    return result;
}

// Prior world matrix of one instance for Motion, packed into the three
// per-instance vectors that mode 2 draws leave unused (identity color, color
// and emission). Columns 0 to 2 and the translation fill them in order:
// (c0.xyz, c1.x), (c1.yz, c2.xy), (c2.z, t.xyz). The shader rebuilds the same
// column-major matrix that the current instance uses, so equal transforms give
// the same projected position. objectWorld is the current world of the
// instanced object, runtime transform included. The Motion delta then maps the
// current runtime transform to the prior one, as for every other object. An
// instance without a prior transform repeats its current one.
inline std::array<std::array<float, 4>, 3> nativeObjectInstancePreviousColumns(
    const std::array<float, 16>& objectWorld,
    const HarmonicMIDI::grid::Visual3DScene& scene, std::size_t index) noexcept
{
    const bool prior = scene.previousObjectInstanceCount != 0
        && scene.previousObjectInstanceCount == scene.objectInstanceCount
        && index < scene.previousObjectInstanceCount;
    const auto m = nativeObjectInstanceMatrix(objectWorld,
        prior ? scene.previousObjectInstances[index] : scene.objectInstances[index]);
    return {{ { m[0], m[1], m[2], m[4] }, { m[5], m[6], m[8], m[9] },
        { m[10], m[12], m[13], m[14] } }};
}

// One native renderer path draws a scene's instanced object. It has no
// material programs, Geometry Core instances or per-note expansion, so those
// requests are refused with this reason.
inline bool nativeObjectInstancesSupported(
    const HarmonicMIDI::grid::Visual3DScene& scene,
    const NativeFixtureSurfaceMaterialProgram* materialProgram,
    bool geometryInstances, std::string& error)
{
    if (!scene.instancedObject.isValid()) return true;
    if (materialProgram != nullptr || geometryInstances)
    {
        error = "Instanced scene objects require the plain scene material without a graph material program or Geometry instances";
        return false;
    }
    return true;
}

inline bool nativeObjectInstanceRuntimeSupported(
    const HarmonicMIDI::grid::Visual3DScene& scene,
    const NativeFixtureSceneRuntimeInputs& inputs, std::string& error)
{
    if (!scene.instancedObject.isValid()) return true;
    // A predecessor frame needs each instance's prior transform. Without one
    // the instances would report zero Motion while the scene moved them.
    if (sceneUsesMotionPass(inputs) && inputs.previousMotion
        && scene.previousObjectInstanceCount != scene.objectInstanceCount)
    {
        error = "Motion output for instanced scene objects requires each instance's prior transform";
        return false;
    }
    if (inputs.noteInstanceMapping)
    {
        error = "Instanced scene objects do not combine with per-note instancing";
        return false;
    }
    return true;
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

// Light record selection shared by every fixture backend reader and the glass
// caustic path. Sampled or caller records win over the scene's own records.
// The Environment light override replaces the whole list, so overrides address
// no read light in that mode. The validator returns the reason in error.
inline std::size_t effectiveNativeLightCount(const HarmonicMIDI::grid::Visual3DScene& scene,
    const NativeImportedSceneRuntimeInputs& runtime) noexcept
{
    return !runtime.animatedLights.empty() ? runtime.animatedLights.size() : scene.lightCount;
}

// The record a reader picks at index, before overrides. Callers ensure index < count.
inline HarmonicMIDI::grid::SceneLightRecord baseNativeLightRecord(
    const HarmonicMIDI::grid::Visual3DScene& scene,
    const NativeImportedSceneRuntimeInputs& runtime, std::size_t index) noexcept
{
    return !runtime.animatedLights.empty() ? runtime.animatedLights[index] : scene.lights[index];
}

// The record the backend reads at index with its overrides applied. An Environment
// light override replaces the record entirely, which matches the reader's count of one.
inline HarmonicMIDI::grid::SceneLightRecord effectiveNativeLightRecord(
    const HarmonicMIDI::grid::Visual3DScene& scene,
    const NativeImportedSceneRuntimeInputs& runtime, std::size_t index) noexcept
{
    if (runtime.lightOverride) return *runtime.lightOverride;
    auto light = baseNativeLightRecord(scene, runtime, index);
    for (const auto& modulated : runtime.lightOverrides)
        if (modulated.index == index) applyNativeImportedLightOverride(modulated, light);
    return light;
}

// Each override must name a light the reader reads. An index past the effective
// count is refused with an error and never ignored. With an Environment light
// override the read list is that single light, so the overrides are not read
// and are not checked here.
inline bool validNativeImportedLightOverrideTargets(const HarmonicMIDI::grid::Visual3DScene& scene,
    const NativeImportedSceneRuntimeInputs& runtime, std::string& error)
{
    if (runtime.lightOverride) return true;
    const auto count = effectiveNativeLightCount(scene, runtime);
    for (const auto& modulated : runtime.lightOverrides)
    {
        if (modulated.index >= count)
        {
            error = "imported light override names no light record";
            return false;
        }
        if (!validNativeImportedLightOverrideTarget(modulated, baseNativeLightRecord(scene, runtime, modulated.index)))
        {
            error = "imported light override does not apply to its light kind";
            return false;
        }
    }
    return true;
}

// Zero keeps the authored field of view. Other values match the camera lowering range.
inline bool validRuntimeCameraFov(float degrees) noexcept
{
    return std::isfinite(degrees)
        && (degrees == 0.0f || (degrees >= 1.0f && degrees <= 179.0f));
}

// Runtime camera adjustments on an authored camera record. Translation is added
// in world space. Rotation is composed on the left, as applyRuntimeObjectTransform
// does for objects, and a zero rotation leaves the authored quaternion bit-exact.
// A zero field of view keeps the authored vertical field of view.
inline HarmonicMIDI::grid::SceneCameraRecord runtimeCameraRecord(
    HarmonicMIDI::grid::SceneCameraRecord camera,
    const std::array<float, 3>& translationOffset,
    const std::array<float, 3>& rotationDegrees,
    float verticalFovDegrees) noexcept
{
    camera.transform.translation.x += translationOffset[0];
    camera.transform.translation.y += translationOffset[1];
    camera.transform.translation.z += translationOffset[2];
    camera.transform.rotation = multiplySceneQuaternion(
        runtimeEulerQuaternion(rotationDegrees), camera.transform.rotation);
    if (verticalFovDegrees > 0.0f)
        camera.verticalFovRadians = verticalFovDegrees * 0.01745329251994329577f;
    return camera;
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
    // Equal object transforms give exact identity Motion, with no quaternion round-trip residue.
    if (previous.objectRotationDegrees == inputs.objectRotationDegrees
        && previous.objectTranslationOffset == inputs.objectTranslationOffset
        && previous.objectScale == inputs.objectScale)
    {
        result.rotation = { 0.0f, 0.0f, 0.0f, 1.0f };
        result.translationScale = { 0.0f, 0.0f, 0.0f, 1.0f };
    }
    const auto& authoredPriorCamera = previous.cameraOverride ? *previous.cameraOverride : camera;
    const auto priorCamera = runtimeCameraRecord(authoredPriorCamera, previous.cameraTranslationOffset,
        previous.cameraRotationDegrees, previous.cameraVerticalFovDegrees);
    const auto& q = priorCamera.transform.rotation;
    const auto& p = priorCamera.transform.translation;
    result.cameraRotation = { q.x, q.y, q.z, q.w };
    result.cameraTranslation = { p.x, p.y, p.z, 0 };
    result.projection = { std::tan(priorCamera.verticalFovRadians * 0.5f), static_cast<float>(width) / height,
        priorCamera.nearPlane, inputs.previousMotion ? 1.0f : 0.0f };
    result.viewport = { static_cast<float>(width), static_cast<float>(height), 0, 0 };
    return result;
}

// x -> rotation(x) * scale + translation, the form of the Motion delta.
struct NativeSceneSimilarity final
{
    HarmonicMIDI::grid::SceneQuaternion rotation { 0, 0, 0, 1 };
    float scale = 1.0f;
    HarmonicMIDI::grid::SceneVec3 translation {};
};

inline HarmonicMIDI::grid::SceneQuaternion normalizedSceneQuaternion(
    HarmonicMIDI::grid::SceneQuaternion value) noexcept
{
    const auto length = std::sqrt(value.x * value.x + value.y * value.y
        + value.z * value.z + value.w * value.w);
    if (!(length > 0.0f)) return { 0, 0, 0, 1 };
    return { value.x / length, value.y / length, value.z / length, value.w / length };
}

// Callers pass transforms with uniform scale only.
inline NativeSceneSimilarity sceneSimilarity(const HarmonicMIDI::grid::SceneTransform3D& transform) noexcept
{
    return { normalizedSceneQuaternion(transform.rotation), transform.scale.x, transform.translation };
}

// Applies right first, then left.
inline NativeSceneSimilarity composeSceneSimilarity(
    const NativeSceneSimilarity& left, const NativeSceneSimilarity& right) noexcept
{
    const auto moved = rotateSceneVector(left.rotation, right.translation);
    return { multiplySceneQuaternion(left.rotation, right.rotation), left.scale * right.scale,
        { moved.x * left.scale + left.translation.x, moved.y * left.scale + left.translation.y,
          moved.z * left.scale + left.translation.z } };
}

inline NativeSceneSimilarity inverseSceneSimilarity(const NativeSceneSimilarity& value) noexcept
{
    const HarmonicMIDI::grid::SceneQuaternion inverse { -value.rotation.x, -value.rotation.y,
        -value.rotation.z, value.rotation.w };
    const auto moved = rotateSceneVector(inverse, value.translation);
    const auto scale = 1.0f / value.scale;
    return { inverse, scale, { -moved.x * scale, -moved.y * scale, -moved.z * scale } };
}

inline bool sameSceneTransform(const HarmonicMIDI::grid::SceneTransform3D& a,
    const HarmonicMIDI::grid::SceneTransform3D& b) noexcept
{
    return a.translation.x == b.translation.x && a.translation.y == b.translation.y
        && a.translation.z == b.translation.z && a.rotation.x == b.rotation.x
        && a.rotation.y == b.rotation.y && a.rotation.z == b.rotation.z
        && a.rotation.w == b.rotation.w && a.scale.x == b.scale.x
        && a.scale.y == b.scale.y && a.scale.z == b.scale.z;
}

// True when object index carries a prior transform that differs from its current one.
inline bool sceneObjectHasOwnPrior(const HarmonicMIDI::grid::Visual3DScene& scene, std::size_t index) noexcept
{
    return scene.previousObjectTransformCount == scene.objectCount && index < scene.objectCount
        && !sameSceneTransform(scene.previousObjectTransforms[index], scene.objects[index].transform);
}

// Retained prior positions cover a prefix of the scene vertices. A shorter
// prefix leaves the vertices past it to objects that carry their own prior
// transform and keep their current local positions, so no object may straddle
// the boundary.
inline bool sceneMotionPositionsMatch(const HarmonicMIDI::grid::Visual3DScene& scene, std::size_t count) noexcept
{
    if (count == scene.vertexCount) return true;
    if (count > scene.vertexCount || scene.previousObjectTransformCount != scene.objectCount) return false;
    for (std::size_t index = 0; index < scene.objectCount; ++index)
    {
        const auto& object = scene.objects[index];
        if (object.firstVertex < count && object.firstVertex + object.vertexCount > count) return false;
    }
    return true;
}

// An object's own prior reaches the shader through the Motion rotation, uniform
// scale and translation. That holds only for an individually drawn object whose
// current and prior transforms and every ancestor have uniform scale.
inline bool nativeObjectPriorTransformsSupported(const HarmonicMIDI::grid::Visual3DScene& scene,
    const NativeFixtureSceneRuntimeInputs& inputs, bool sharedInstances, std::string& error)
{
    if (scene.previousObjectTransformCount == 0 || !sceneUsesMotionPass(inputs)) return true;
    const auto uniform = [](const HarmonicMIDI::grid::SceneTransform3D& transform) {
        return transform.scale.x > 0.0f && transform.scale.x == transform.scale.y
            && transform.scale.x == transform.scale.z;
    };
    for (std::size_t index = 0; index < scene.objectCount; ++index)
    {
        if (!sceneObjectHasOwnPrior(scene, index)) continue;
        const auto& object = scene.objects[index];
        bool valid = !sharedInstances && object.id != scene.instancedObject
            && uniform(object.transform) && uniform(scene.previousObjectTransforms[index]);
        auto parent = object.parent;
        for (std::size_t depth = 0; valid && parent.isValid() && depth < scene.objectCount; ++depth)
        {
            const auto* record = findFixtureObject(scene, parent);
            valid = record != nullptr && uniform(record->transform);
            parent = record != nullptr ? record->parent : HarmonicMIDI::grid::SceneObjectId {};
        }
        if (!valid)
        {
            error = "Motion output for scene objects with their own prior transform requires individual draws with uniform scale";
            return false;
        }
    }
    return true;
}

// Conjugates an object-level step by the runtime transform its draw applies, then
// composes it after the frame step. Callers pass a step in the object's world frame.
inline NativeSceneMotionUniforms withSceneObjectOwnStep(NativeSceneMotionUniforms motion,
    const HarmonicMIDI::grid::SceneTransform3D& runtime, const NativeSceneSimilarity& step) noexcept
{
    const auto applied = sceneSimilarity(runtime);
    const auto own = composeSceneSimilarity(composeSceneSimilarity(applied, step),
        inverseSceneSimilarity(applied));
    const NativeSceneSimilarity frame { { motion.rotation[0], motion.rotation[1], motion.rotation[2],
        motion.rotation[3] }, motion.translationScale[3], { motion.translationScale[0],
        motion.translationScale[1], motion.translationScale[2] } };
    const auto total = composeSceneSimilarity(frame, own);
    const auto rotation = normalizedSceneQuaternion(total.rotation);
    motion.rotation = { rotation.x, rotation.y, rotation.z, rotation.w };
    motion.translationScale = { total.translation.x, total.translation.y, total.translation.z, total.scale };
    return motion;
}

// Motion uniforms for one draw. The frame uniforms map the current runtime
// world to the prior one. A draw the runtime transform does not target gets no
// object-level runtime step, so its frame object step is identity. The camera
// step is kept. An object with its own prior transform then takes its own step,
// prior world times current world inverse, conjugated by the runtime transform
// the draw applies. A body with no prior repeats its current transform.
// `targeted` must be runtimeTargetsObject for the same draw that colour uses.
inline NativeSceneMotionUniforms sceneObjectMotionUniforms(NativeSceneMotionUniforms motion,
    const HarmonicMIDI::grid::Visual3DScene& scene, std::size_t index,
    const HarmonicMIDI::grid::SceneTransform3D& runtime, bool targeted) noexcept
{
    if (!targeted)
    {
        motion.rotation = { 0, 0, 0, 1 };
        motion.translationScale = { 0, 0, 0, 1 };
    }
    if (!sceneObjectHasOwnPrior(scene, index)) return motion;
    const auto& object = scene.objects[index];
    NativeSceneSimilarity parentWorld;
    auto parent = object.parent;
    for (std::size_t depth = 0; parent.isValid() && depth < scene.objectCount; ++depth)
    {
        const auto* record = findFixtureObject(scene, parent);
        if (record == nullptr) break;
        parentWorld = composeSceneSimilarity(sceneSimilarity(record->transform), parentWorld);
        parent = record->parent;
    }
    const auto step = composeSceneSimilarity(
        composeSceneSimilarity(parentWorld, sceneSimilarity(scene.previousObjectTransforms[index])),
        composeSceneSimilarity(inverseSceneSimilarity(sceneSimilarity(object.transform)),
            inverseSceneSimilarity(parentWorld)));
    return withSceneObjectOwnStep(motion, runtime, step);
}

// Affine product of two checked worlds, column-major with an affine last row.
inline std::array<float, 16> nativeWorldProduct(const std::array<float, 16>& left,
    const std::array<float, 16>& right) noexcept
{
    std::array<float, 16> result {};
    for (std::size_t column = 0; column < 4; ++column)
        for (std::size_t row = 0; row < 4; ++row)
            for (std::size_t inner = 0; inner < 4; ++inner)
                result[column * 4 + row] += left[inner * 4 + row] * right[column * 4 + inner];
    return result;
}

// Inverse of a checked world. Empty when its linear part is singular.
inline std::optional<std::array<float, 16>> nativeWorldInverse(const std::array<float, 16>& world) noexcept
{
    const auto a = [&world](std::size_t row, std::size_t column) { return world[column * 4 + row]; };
    // adjugate[row][column] is the cofactor of the entry at (column, row).
    const float adjugate[3][3] = {
        { a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1), a(0, 2) * a(2, 1) - a(0, 1) * a(2, 2),
          a(0, 1) * a(1, 2) - a(0, 2) * a(1, 1) },
        { a(1, 2) * a(2, 0) - a(1, 0) * a(2, 2), a(0, 0) * a(2, 2) - a(0, 2) * a(2, 0),
          a(0, 2) * a(1, 0) - a(0, 0) * a(1, 2) },
        { a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0), a(0, 1) * a(2, 0) - a(0, 0) * a(2, 1),
          a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0) } };
    const float determinant = a(0, 0) * adjugate[0][0] + a(0, 1) * adjugate[1][0]
        + a(0, 2) * adjugate[2][0];
    if (!std::isfinite(determinant) || determinant == 0.0f) return std::nullopt;
    std::array<float, 16> result {};
    for (std::size_t row = 0; row < 3; ++row)
        for (std::size_t column = 0; column < 3; ++column)
            result[column * 4 + row] = adjugate[row][column] / determinant;
    for (std::size_t row = 0; row < 3; ++row)
        result[12 + row] = -(result[row] * world[12] + result[4 + row] * world[13]
            + result[8 + row] * world[14]);
    result[15] = 1.0f;
    return result;
}

// Unit quaternion of a proper rotation, row-major, in the R(q) convention of
// fixtureTransformMatrix. Shepperd's method divides by the largest term.
inline HarmonicMIDI::grid::SceneQuaternion sceneQuaternionFromRotation(const std::array<float, 9>& r) noexcept
{
    using Quaternion = HarmonicMIDI::grid::SceneQuaternion;
    const float trace = r[0] + r[4] + r[8];
    if (trace > 0.0f)
    {
        const float s = std::sqrt(trace + 1.0f) * 2.0f;
        return Quaternion { (r[7] - r[5]) / s, (r[2] - r[6]) / s, (r[3] - r[1]) / s, 0.25f * s };
    }
    if (r[0] > r[4] && r[0] > r[8])
    {
        const float s = std::sqrt(1.0f + r[0] - r[4] - r[8]) * 2.0f;
        return Quaternion { 0.25f * s, (r[1] + r[3]) / s, (r[2] + r[6]) / s, (r[7] - r[5]) / s };
    }
    if (r[4] > r[8])
    {
        const float s = std::sqrt(1.0f + r[4] - r[0] - r[8]) * 2.0f;
        return Quaternion { (r[1] + r[3]) / s, 0.25f * s, (r[5] + r[7]) / s, (r[2] - r[6]) / s };
    }
    const float s = std::sqrt(1.0f + r[8] - r[0] - r[4]) * 2.0f;
    return Quaternion { (r[2] + r[6]) / s, (r[5] + r[7]) / s, 0.25f * s, (r[3] - r[1]) / s };
}

// Relative tolerance, against the squared scale, for a step to count as a similarity.
// At 1e-4 that allows about 0.006 degrees off orthogonal and about 0.005% in column length.
inline constexpr float kNativeObjectWorldStepTolerance = 1.0e-4f;

// The Motion step that carries one checked world of an object to its prior world:
// prior times the inverse of current. It is a similarity only when the affine step
// has one uniform scale and an orthonormal rotation. Shear and non-uniform scale
// have no similarity form, so they fail.
inline bool nativeObjectWorldStep(const std::array<float, 16>& prior,
    const std::array<float, 16>& current, NativeSceneSimilarity& step) noexcept
{
    const auto inverse = nativeWorldInverse(current);
    if (!inverse) return false;
    const auto product = nativeWorldProduct(prior, *inverse);
    const auto dot = [&product](std::size_t left, std::size_t right) {
        return product[left * 4] * product[right * 4] + product[left * 4 + 1] * product[right * 4 + 1]
            + product[left * 4 + 2] * product[right * 4 + 2];
    };
    const float squared = dot(0, 0);
    if (!(squared > 0.0f) || !std::isfinite(squared)) return false;
    const float tolerance = kNativeObjectWorldStepTolerance * squared;
    if (std::abs(dot(1, 1) - squared) > tolerance || std::abs(dot(2, 2) - squared) > tolerance
        || std::abs(dot(0, 1)) > tolerance || std::abs(dot(0, 2)) > tolerance
        || std::abs(dot(1, 2)) > tolerance) return false;
    // A reflection has a negative determinant and no rotation quaternion.
    const float determinant = product[0] * (product[5] * product[10] - product[9] * product[6])
        - product[4] * (product[1] * product[10] - product[9] * product[2])
        + product[8] * (product[1] * product[6] - product[5] * product[2]);
    if (!(determinant > 0.0f)) return false;
    const float scale = std::sqrt(squared);
    std::array<float, 9> rotation {};
    for (std::size_t row = 0; row < 3; ++row)
        for (std::size_t column = 0; column < 3; ++column)
            rotation[row * 3 + column] = product[column * 4 + row] / scale;
    step = { normalizedSceneQuaternion(sceneQuaternionFromRotation(rotation)), scale,
        { product[12], product[13], product[14] } };
    return true;
}

// Motion uniforms for a draw whose checked world the prior frame also names. The
// object takes its own step only when the world changed. An unchanged world keeps
// the frame step exactly. A first frame has no prior world, so it keeps the frame
// step too, and the frame flag makes that motion zero.
inline NativeSceneMotionUniforms sceneObjectWorldMotionUniforms(NativeSceneMotionUniforms motion,
    const HarmonicMIDI::grid::SceneTransform3D& runtime, bool targeted,
    const NativeImportedObjectWorlds* prior, const NativeImportedObjectWorlds* current,
    HarmonicMIDI::grid::SceneObjectId object) noexcept
{
    if (!targeted)
    {
        motion.rotation = { 0, 0, 0, 1 };
        motion.translationScale = { 0, 0, 0, 1 };
    }
    const auto* before = nativeObjectWorldEntry(prior, object);
    const auto* after = nativeObjectWorldEntry(current, object);
    if (before == nullptr || after == nullptr || before->world == after->world) return motion;
    NativeSceneSimilarity step;
    // nativeObjectWorldMotionSupported refuses every changed world without a similarity.
    if (!nativeObjectWorldStep(before->world, after->world, step)) return motion;
    return withSceneObjectOwnStep(motion, runtime, step);
}

// Motion output under checked static-local worlds. A changed world must differ from
// its prior world by a similarity, the prior frame must name the same objects, and
// no object may also carry a scene prior transform, because the worlds replace the
// authored chain that such a transform builds on.
inline bool nativeObjectWorldMotionSupported(const HarmonicMIDI::grid::Visual3DScene& scene,
    const NativeFixtureSceneRuntimeInputs& inputs, std::string& error)
{
    if (!inputs.objectWorlds || !sceneUsesMotionPass(inputs)) return true;
    for (std::size_t index = 0; index < scene.objectCount; ++index)
        if (sceneObjectHasOwnPrior(scene, index))
        {
            error = "Motion output for static local object worlds does not yet combine with scene object prior transforms";
            return false;
        }
    // A first frame has no predecessor, so its motion is zero and needs no step.
    if (!inputs.previousMotion) return true;
    for (const auto& entry : *inputs.objectWorlds)
    {
        const auto* before = nativeObjectWorldEntry(inputs.previousMotion->objectWorlds.get(), entry.object);
        if (before == nullptr)
        {
            error = "Motion output prior static local object worlds do not match the drawn scene objects";
            return false;
        }
        NativeSceneSimilarity step;
        if (before->world != entry.world && !nativeObjectWorldStep(before->world, entry.world, step))
        {
            error = "Motion output for static local object worlds requires each changed world to differ by a rotation, uniform scale and translation";
            return false;
        }
    }
    return true;
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
            && visual3d_detail::finite(light->range) && light->range >= 0.0f
            && validSceneAnalyticSky(*light) && validSceneEnvironmentImage(*light));
}

inline bool validFixtureRuntimeInputs(
    const NativeFixtureSceneRuntimeInputs& inputs) noexcept
{
    // V4 requires an independent native scale/appearance transport before activation.
    if (inputs.noteInstanceMapping && inputs.noteInstanceMapping->schemaVersion
            == visualnoteinstancing::kVelocityScaleMappingSchemaVersion) return false;

    // P4 must install the authenticated typed-world consumers before activation.
    if (inputs.localSceneFrame || (inputs.previousMotion && inputs.previousMotion->localSceneFrame)) return false;

    const auto validPreviousNotes = [](const auto& table)
    {
        if (!table) return true;
        if (table->count > table->identities.size()) return false;
        for (std::size_t i = 0; i < table->count; ++i)
        {
            if (table->identities[i] == 0) return false;
            for (std::size_t j = 0; j < i; ++j)
                if (table->identities[i] == table->identities[j]) return false;
            for (const auto value : table->transforms[i])
                if (!std::isfinite(value)) return false;
            if (table->transforms[i][3] <= 0.0f || table->transforms[i][3] > 10000.0f) return false;
        }
        return true;
    };
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
            && valid(inputs.previousMotion->cameraRotationDegrees, 360.0f)
            && validRuntimeCameraFov(inputs.previousMotion->cameraVerticalFovDegrees)
            && validFixtureCameraOverride(inputs.previousMotion->cameraOverride)
            && std::isfinite(inputs.previousMotion->morphWeight)
            && inputs.previousMotion->morphWeight >= 0.0f && inputs.previousMotion->morphWeight <= 1.0f
            && validSceneMotionImportedNodeWorldMatrices(*inputs.previousMotion)
            && validPreviousNotes(inputs.previousMotion->noteTransforms)
            && std::isfinite(inputs.previousMotion->timeSeconds)
            && std::all_of(inputs.previousMotion->vertexSpectrum.begin(),
                inputs.previousMotion->vertexSpectrum.end(), [](float value)
                { return std::isfinite(value) && value >= 0.0f && value <= 1.0f; })
            && validSceneMotionPositions(inputs.previousMotion->scenePositions)))
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
        && valid(inputs.cameraRotationDegrees, 360.0f)
        && validRuntimeCameraFov(inputs.cameraVerticalFovDegrees)
        && validFixtureCameraOverride(inputs.animatedCamera)
        && inputs.animatedLights.size() <= HarmonicMIDI::grid::Visual3DScene::kMaxLights
        && HarmonicMIDI::grid::validSceneAnalyticSkySet(inputs.animatedLights.data(),inputs.animatedLights.size())
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
        && validNativeImportedLightOverrideList(inputs.lightOverrides)
        && (inputs.animatedLights.empty()
            || validNativeImportedLightOverrides(inputs.lightOverrides, inputs.animatedLights))
        && validFixtureCameraOverride(inputs.cameraOverride)
        && validFixtureLightOverride(inputs.lightOverride)
        && (!inputs.objectWorlds || validNativeImportedObjectWorlds(*inputs.objectWorlds))
        && std::isfinite(inputs.emissionGain)
        && inputs.emissionGain >= 0.0f && inputs.emissionGain <= 64.0f
        && (static_cast<bool>(inputs.canonicalBlockCFrame)
                == inputs.noteInstanceMapping.has_value())
        && (!inputs.canonicalBlockCFrame
            || (canonicalblockc::valid(inputs.canonicalBlockCFrame)
                && visualnoteinstancing::valid(*inputs.noteInstanceMapping)
                && (inputs.noteInstanceMapping->schemaVersion
                        != visualnoteinstancing::kDirectAxesMappingSchemaVersion
                    || prepareNativeNoteInstances(inputs).admitted)));
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
    // Enable only with the exact stable-note predecessor consumer.
    virtual bool supportsRetainedNoteMotion() const noexcept { return false; }
    // Release cached programs while their context is current, before destroying
    // it. Live scene-resource leases remain valid until their owners release them.
    virtual void releaseCachedProgramsForCurrentContext() noexcept {}
    virtual NativeFixtureScenePreparation prepare (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
        const std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>& materialProgram = {},
        NativeFixturePrepareUse use = NativeFixturePrepareUse::Blocking) = 0;
    virtual NativeFixtureScenePreparation prepareGeometryInstances (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
        const std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>& materialProgram,
        const std::shared_ptr<const videohelper::geometry::AdmittedPlanValue>&
            geometryAdmission,
        bool diagnosticInstanceIdentityColors = false,
        const std::vector<videowire::geometry::AttributeData>* frameAttributes = nullptr,
        NativeFixturePrepareUse use = NativeFixturePrepareUse::Blocking)
    {
        (void) geometryAdmission;
        (void) diagnosticInstanceIdentityColors;
        (void) frameAttributes;
        return prepare (scene, materialProgram, use);
    }
    virtual NativeFixtureSceneSubmission render (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
        const std::shared_ptr<const NativeFixtureSceneResources>& resources,
        std::uint32_t width,
        std::uint32_t height,
        NativeFixtureSceneRuntimeInputs runtimeInputs = {}) = 0;
    // Each backend enables this only with the exact table/query ABI and
    // whole-output invalid-query readback. Shared admission alone is insufficient.
    virtual bool supportsAuthoredSpectralReceiverTransport() const noexcept { return false; }
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
    std::shared_ptr<const NativeDeformationCompositionOwner> compositionOwner {};
    std::uint64_t evaluationRevision = 0;
    // CPU logical retention only; this never authenticates a draw or GPU resource.
    std::shared_ptr<const void> logicalCompositionRetention {};
};

#include "native_deformation_composition_types.h"

struct NativeDeformationFrameData
{
    std::array<float, kNativeDeformationMaxJoints * 16> jointPalette {};
    std::array<float, kNativeDeformationMaxMorphTargets> morphWeights {};
    std::uint32_t vertexCount = 0;
    std::uint32_t jointCount = 0;
    std::uint32_t morphTargetCount = 0;
    NativeDeformationComputePostBlock compositionPost {};
};

// Optional sampled worlds use the admitted joint IDs, which imported sources
// map to one-based GLB node IDs. Publish only after the whole frame validates.
bool prepareNativeDeformationFrame (
    const NativeDeformationScene& source,
    const visualdeformation::AnimationDeformationSnapshot& snapshot,
    NativeDeformationFrameData& output,
    std::string& error,
    NativeSceneMotionSample::ImportedNodeWorldMatrices* sampledWorlds = nullptr);

bool prepareNativeDeformationFrame (
    const NativeDeformationScene& source,
    const visualdeformation::AnimationDeformationSnapshot& snapshot,
    const std::shared_ptr<const NativeDeformationCompositionFrame>& composition,
    NativeDeformationFrameData& output, std::string& error,
    NativeSceneMotionSample::ImportedNodeWorldMatrices* sampledWorlds = nullptr);

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
    // Backend names and compute availability do not imply prior-pose support.
    virtual bool supportsRetainedDeformationMotion() const noexcept { return false; }
    virtual bool supportsRetainedImportedObjectMotion() const noexcept { return false; }
    virtual bool supportsRetainedNoteMotion() const noexcept { return false; }
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
