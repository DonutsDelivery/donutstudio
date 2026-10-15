#pragma once

#include "optical_flow_native_executor.h"
#include "optical_flow_source_resolver.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <utility>

namespace videoopticalflow
{
struct SourcePairLease final
{
    std::array<std::shared_ptr<const NativeSourceFrame>, 2> frames;
    std::optional<ValidatedSourcePairV2> selection;
    std::optional<AdmittedRequest> request;
};

// The uploader must run on the owner of the native GPU context, perform the
// admitted color conversion, and hash the resulting linear RGBA16F pixels.
// The lease adapter checks its immutable receipt before exposing either frame.
class NativeSourceUploader
{
public:
    virtual ~NativeSourceUploader() = default;
    virtual std::shared_ptr<const NativeSourceFrame> upload (
        const DecodedSourceReceipt& decoded,
        const colortransform::AdmittedTransform& transform,
        const SourceFrameV2& expected, std::string& error) = 0;
    virtual std::shared_ptr<const NativeSourceFrame> uploadPreview (
        const DecodedSourceReceipt& decoded,
        const colortransform::AdmittedTransform& transform,
        const SourceFrameV2& expected, bool pixelsValidatedOnWorker,
        std::string& error)
    { (void) pixelsValidatedOnWorker; return upload (decoded, transform, expected, error); }
};

inline constexpr const char* kPreviewUploadPending = "Optical Flow native source digest is pending";

// Construct on the viewport/export worker after its native context is current.
// An unavailable backend returns no uploader, leaving the graph fail-closed.
std::unique_ptr<NativeSourceUploader> makeNativeSourceUploader (
    std::uint64_t helperGeneration, std::uint64_t structuralRevision);
void invalidateNativeSourceUploads (std::uintptr_t contextIdentity) noexcept;

namespace detail
{
inline bool decodeSha256 (const std::string& hex, ContentIdentity& result) noexcept
{
    if (hex.size() != 64) return false;
    const auto digit = [] (char value) noexcept -> int {
        if (value >= '0' && value <= '9') return value - '0';
        if (value >= 'a' && value <= 'f') return value - 'a' + 10;
        return -1;
    };
    for (std::size_t index = 0; index < result.size(); ++index)
    {
        const int high = digit (hex[index * 2]);
        const int low = digit (hex[index * 2 + 1]);
        if (high < 0 || low < 0) return false;
        result[index] = static_cast<std::uint8_t> ((high << 4) | low);
    }
    return true;
}

inline bool expectedSource (
    const DecodedSourceReceipt& decoded, SourceFrameV2& expected,
    colortransform::Description& color, std::string& error,
    bool hashPixels = true)
{
    const auto& frame = decoded.frame;
    if (decoded.clipId == 0 || decoded.sourceGeneration == 0
        || frame.width <= 0 || frame.height <= 0
        || frame.width > static_cast<int> (kMaximumWidth)
        || frame.height > static_cast<int> (kMaximumHeight)
        || frame.strideBytes != frame.width * 4
        || frame.rgba.size() != static_cast<std::size_t> (frame.strideBytes)
                                * static_cast<std::size_t> (frame.height)
        || (hashPixels && videohelper::sha256Bytes (frame.rgba.data(), frame.rgba.size())
               != decoded.pixelSha256)
        || frame.timeBase.num <= 0 || frame.timeBase.den <= 0
        || frame.ptsTicks == AV_NOPTS_VALUE || frame.ptsTicks < 0
        || frame.ptsTicks > std::numeric_limits<std::int64_t>::max() / frame.timeBase.num
        || ! decodeSha256 (decoded.sourceSha256, expected.sourceContent))
    {
        error = "Optical Flow decoded source receipt is incomplete or changed";
        return false;
    }
    expected.clipId = decoded.clipId;
    expected.sourceGeneration = decoded.sourceGeneration;
    expected.frame.identity.frameIndex = decoded.presentationOrdinal;
    expected.frame.timestamp = {
        frame.ptsTicks * frame.timeBase.num,
        static_cast<std::uint32_t> (frame.timeBase.den) };
    expected.frame.extent = {
        static_cast<std::uint32_t> (frame.width),
        static_cast<std::uint32_t> (frame.height) };
    expected.frame.format = renderpassoutput::PixelFormat::RGBA16Float;
    expected.frame.colorSpace = renderpassoutput::ColorSpace::LinearSRGB;

    const auto streamHash = videohelper::sha256Text (
        "DonutStudio/OpticalFlowStream/" + std::to_string (decoded.clipId)
        + "/" + decoded.sourceSha256 + "/"
        + std::to_string (decoded.sourceGeneration));
    ContentIdentity streamDigest {};
    if (! decodeSha256 (streamHash, streamDigest))
    {
        error = "Optical Flow stream identity could not be formed";
        return false;
    }
    std::copy_n (streamDigest.begin(), expected.frame.identity.stream.size(),
                 expected.frame.identity.stream.begin());

    const colortransform::Extent extent { expected.frame.extent.width,
                                          expected.frame.extent.height };
    color.input = { extent, colortransform::PixelFormat::RGBA8,
                    frame.color.primaries, frame.color.transfer,
                    colortransform::AlphaMode::Straight };
    color.output = { extent, colortransform::PixelFormat::RGBA16F,
                     colortransform::ColorSpace::LinearSRGB,
                     colortransform::TransferFunction::Linear,
                     colortransform::AlphaMode::Straight };
    color.outputIntent = colortransform::OutputIntent::SceneLinear;
    return true;
}

inline bool exactLease (const NativeSourceFrame& lease,
                        const SourceFrameV2& expected,
                        std::uint64_t helperGeneration,
                        std::uint64_t structuralRevision,
                        const BackendCapabilities& backend) noexcept
{
    const auto& frame = lease.frame();
    return lease.sourceClipId() == expected.clipId
        && lease.sourceContent() == expected.sourceContent
        && lease.sourceGeneration() == expected.sourceGeneration
        && frame.identity.stream == expected.frame.identity.stream
        && detail::anyNonZero (frame.identity.content)
        && frame.identity.frameIndex == expected.frame.identity.frameIndex
        && frame.timestamp.ticks == expected.frame.timestamp.ticks
        && frame.timestamp.timescale == expected.frame.timestamp.timescale
        && frame.extent == expected.frame.extent
        && frame.format == expected.frame.format
        && frame.colorSpace == expected.frame.colorSpace
        && lease.helperGeneration() == helperGeneration
        && lease.structuralRevision() == structuralRevision
        && lease.backendImplementation() == backend.implementation
        && lease.backendImplementationRevision() == backend.implementationRevision
        && lease.byteCount() == static_cast<std::uint64_t> (frame.extent.width)
                               * frame.extent.height * 8u
        && detail::anyNonZero (lease.resourceIdentity())
        && lease.imageHandle() != 0
        && lease.nativeContextIdentity() != 0
        && lease.textureViewHandle() == lease.imageHandle();
}
} // namespace detail

// The decoded receipt can cross from a media worker to the GPU context owner.
// It contains owned pixels and no native handles. Recheck its generation and
// source IDs before any upload, even when the caller selected the saved pair.
inline std::string leaseDecodedSourcePair (
    const videowire::CompiledVisualLayerPlan& plan,
    const opticalflowoperation::SourcePair& orderedPair,
    const DecodedPairReceipt& decoded, std::uint64_t snapshotGeneration,
    std::uint64_t helperGeneration, const BackendCapabilities& backend,
    NativeSourceUploader& uploader, std::shared_ptr<const SourcePairLease>& out,
    bool pixelsValidatedOnWorker = false, bool previewAsync = false)
{
    out.reset();
    if (helperGeneration == 0 || plan.structuralRevision == 0
        || snapshotGeneration == 0 || decoded.snapshotGeneration != snapshotGeneration)
        return "Optical Flow native source owner is invalid";
    const auto flow = std::find_if (plan.operations.begin(), plan.operations.end(),
        [] (const auto& value)
        { return value.kind == opticalflowoperation::kPairOperationKind; });
    opticalflowoperation::SourcePair savedPair;
    if (flow == plan.operations.end()
        || ! opticalflowoperation::parseSourcePair (flow->payloadXml, savedPair)
        || savedPair.images != orderedPair.images)
        return "Optical Flow ordered pair changed after decoding";
    auto candidate = std::make_shared<SourcePairLease>();
    std::string error;
    bool hasPendingDigest = false;
    SourcePairDescriptionV2 selection;
    selection.graphClipId = static_cast<std::uint32_t> (plan.clipId);
    selection.structuralRevision = plan.structuralRevision;
    for (std::size_t index = 0; index < 2; ++index)
    {
        const auto& source = decoded.sources[index];
        const auto operation = std::find_if (plan.operations.begin(), plan.operations.end(),
            [&] (const auto& value)
            { return value.nodeId == static_cast<int> (orderedPair.images[index].nodeId); });
        std::uint32_t savedClipId = 0;
        if (operation == plan.operations.end()
            || (operation->kind != "video.source"
                && operation->kind != "video.layer.source")
            || ! opticalflowoperation::parseSourceClip (operation->payloadXml, savedClipId)
            || source.clipId != savedClipId
            || (index == 1 && source.clipId == decoded.sources[0].clipId))
            return "Optical Flow decoded receipt changed its ordered source pair";
        SourceFrameV2 expected;
        colortransform::Description color;
        if (! detail::expectedSource (source, expected, color, error,
                                     ! pixelsValidatedOnWorker)) return error;
        colortransform::AdmissionFailure colorFailure {};
        auto transform = colortransform::admit (
            color, colortransform::BackendCapability::NativeGpu, colorFailure);
        if (! transform)
            return "Optical Flow decoded color conversion is unavailable: "
                + std::string (colortransform::token (colorFailure));
        auto lease = previewAsync
            ? uploader.uploadPreview (source, *transform, expected,
                                      pixelsValidatedOnWorker, error)
            : uploader.upload (source, *transform, expected, error);
        if (! lease && previewAsync && error == kPreviewUploadPending)
        {
            hasPendingDigest = true;
            continue;
        }
        if (! lease)
            return error.empty() ? "Optical Flow native source upload failed" : error;
        if (! detail::exactLease (*lease, expected, helperGeneration,
                                 plan.structuralRevision, backend))
            return "Optical Flow native source lease receipt does not match decoded owner";
        expected.frame.identity.content = lease->frame().identity.content;
        (index == 0 ? selection.first : selection.second) = expected;
        candidate->frames[index] = std::move (lease);
    }
    if (hasPendingDigest) return kPreviewUploadPending;
    if (candidate->frames[0]->resourceIdentity()
            == candidate->frames[1]->resourceIdentity()
        || candidate->frames[0]->imageHandle()
            == candidate->frames[1]->imageHandle())
        return "Optical Flow native source pair aliases one GPU resource";
    selection.output = canonicalOutput (selection.first.frame.extent);
    SourcePairFailureV2 pairFailure {};
    auto validated = validateSourcePairV2 (selection, pairFailure);
    if (! validated)
        return "Optical Flow source pair validation failed: "
            + std::string (token (pairFailure));
    candidate->selection.emplace (std::move (*validated));
    AdmissionFailure admissionFailure {};
    auto admitted = admit (*candidate->selection, backend, admissionFailure);
    if (! admitted)
        return "Optical Flow native source admission failed: "
            + std::string (token (admissionFailure));
    candidate->request.emplace (std::move (*admitted));
    out = std::move (candidate);
    return {};
}

inline std::string resolveAndLeaseSourcePair (
    const videowire::CompiledVisualLayerPlan& plan,
    const opticalflowoperation::SourcePair& orderedPair,
    const std::vector<videowire::OpticalFlowSourceBinding>& catalog,
    const std::vector<videowire::RenderSegment>& segments,
    std::uint64_t snapshotGeneration, double displaySec, int maxW, int maxH,
    std::uint64_t helperGeneration, const BackendCapabilities& backend,
    NativeSourceUploader& uploader, std::shared_ptr<const SourcePairLease>& out)
{
    out.reset();
    std::shared_ptr<const DecodedPairReceipt> decoded;
    const auto error = resolveDecodedPair (plan, orderedPair, catalog, segments,
                                           snapshotGeneration, displaySec, maxW, maxH,
                                           decoded);
    if (! error.empty()) return error;
    return leaseDecodedSourcePair (plan, orderedPair, *decoded, snapshotGeneration,
                                   helperGeneration, backend, uploader, out);
}
} // namespace videoopticalflow
