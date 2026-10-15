#include "media.h"
#include "optical_flow_source_resolver.h"
#include "optical_flow_source_lease.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <utility>

namespace
{
class Lease final : public videoopticalflow::NativeSourceFrame
{
public:
    Lease (videoopticalflow::SourceFrameV2 source,
           videoopticalflow::BackendCapabilities capabilities,
           int* released)
        : source_ (std::move (source)), capabilities_ (capabilities), released_ (released)
    {
        source_.frame.identity.content[0] = 0x40;
        identity_[0] = static_cast<std::uint8_t> (source_.clipId);
    }
    ~Lease() override { ++*released_; }
    const videoopticalflow::FrameDescription& frame() const noexcept override { return source_.frame; }
    const videoopticalflow::ResourceIdentity& resourceIdentity() const noexcept override { return identity_; }
    std::uint64_t helperGeneration() const noexcept override { return 5; }
    std::uint64_t structuralRevision() const noexcept override { return 1; }
    const videoopticalflow::BackendIdentity& backendImplementation() const noexcept override
    { return capabilities_.implementation; }
    std::uint32_t backendImplementationRevision() const noexcept override
    { return capabilities_.implementationRevision; }
    std::uint64_t byteCount() const noexcept override
    { return static_cast<std::uint64_t> (source_.frame.extent.width) * source_.frame.extent.height * 8u; }
    std::uintptr_t imageHandle() const noexcept override { return source_.clipId; }
    std::uintptr_t textureViewHandle() const noexcept override { return imageHandle(); }
    std::uintptr_t nativeContextIdentity() const noexcept override { return 77; }
    std::uint32_t sourceClipId() const noexcept override { return source_.clipId; }
    const videoopticalflow::ContentIdentity& sourceContent() const noexcept override
    { return source_.sourceContent; }
    std::uint64_t sourceGeneration() const noexcept override { return source_.sourceGeneration; }
    videoopticalflow::SourceFrameV2 source_;
private:
    videoopticalflow::BackendCapabilities capabilities_;
    videoopticalflow::ResourceIdentity identity_ {};
    int* released_ = nullptr;
};

class Uploader : public videoopticalflow::NativeSourceUploader
{
public:
    enum class Fault { None, WrongOwner, Extent, Color, Pts, MissingSecond };
    videoopticalflow::BackendCapabilities capabilities;
    Fault fault = Fault::None;
    int released = 0;
    int uploads = 0;
    std::shared_ptr<const videoopticalflow::NativeSourceFrame> upload (
        const videoopticalflow::DecodedSourceReceipt&,
        const colortransform::AdmittedTransform& transform,
        const videoopticalflow::SourceFrameV2& expected,
        std::string& error) override
    {
        ++uploads;
        if (fault == Fault::MissingSecond && uploads == 2)
        { error = "missing second GPU source"; return {}; }
        if (transform.description().output.format != colortransform::PixelFormat::RGBA16F
            || transform.description().output.transfer != colortransform::TransferFunction::Linear)
        { error = "wrong color transform"; return {}; }
        auto result = std::make_shared<Lease> (expected, capabilities, &released);
        if (fault == Fault::WrongOwner) ++result->source_.clipId;
        if (fault == Fault::Extent) ++result->source_.frame.extent.width;
        if (fault == Fault::Color)
            result->source_.frame.colorSpace = renderpassoutput::ColorSpace::Data;
        if (fault == Fault::Pts) ++result->source_.frame.timestamp.ticks;
        return result;
    }
};

class PendingUploader final : public Uploader
{
public:
    bool ready[2] { false, false };
    int previewRequests = 0;
    std::shared_ptr<const videoopticalflow::NativeSourceFrame> uploadPreview (
        const videoopticalflow::DecodedSourceReceipt& decoded,
        const colortransform::AdmittedTransform& transform,
        const videoopticalflow::SourceFrameV2& expected,
        bool, std::string& error) override
    {
        ++previewRequests;
        if (! ready[decoded.clipId - 1])
        { error = videoopticalflow::kPreviewUploadPending; return {}; }
        return upload (decoded, transform, expected, error);
    }
};

bool check (bool passed, const char* message)
{
    if (! passed) std::fprintf (stderr, "FAIL: %s\n", message);
    return passed;
}

bool decode (MediaContext& media, double requestSec, DecodedFrame& frame)
{
    const auto error = media.getFrame (requestSec, 16, 16, frame);
    if (error.empty()) return true;
    std::fprintf (stderr, "FAIL: decode at %.3f: %s\n", requestSec, error.c_str());
    return false;
}
}

int main (int argc, char** argv)
{
    if (argc != 2 && argc != 3)
    {
        std::fprintf (stderr, "usage: media_decoded_frame_pts_tests FIXTURE_DIR [OPTICAL_IMAGE_DIR]\n");
        return 2;
    }

    const std::string fixtures = argv[1];
    MediaContext timed;
    const auto error = timed.open (fixtures + "/frames.mp4", false);
    if (! check (error.empty(), "open timestamped MP4")) return 1;

    DecodedFrame frame;
    bool ok = check (frame.ptsTicks == AV_NOPTS_VALUE
                     && frame.timeBase.num == 0 && frame.timeBase.den == 1,
                     "an empty decoded frame has no timestamp");
    DecodedHdrFrame hdr;
    hdr.rgba = { 1.0f };
    const auto hdrError = timed.getHdrFrame (0.052, 16, 16, hdr);
    ok &= check (hdrError.find ("requires declared Rec.2020") != std::string::npos
                 && hdr.rgba.empty() && hdr.width == 0 && hdr.ptsTicks == AV_NOPTS_VALUE,
                 "SDR input cannot produce a float HDR frame or retain stale output");

    // Two lossless 16x16 YUV444P10 frames at 25 fps, with PTS 0 and 512.
    // Y/U/V is 512/512/512 then 512/400/600; x264 VUI declares BT.2020 NCL,
    // limited range and PQ. The complete clip lasts only 0.080 seconds.
    MediaContext hdrMedia;
    if (! check (hdrMedia.open (fixtures + "/hdr-pq-rec2020-16x16.mp4", false).empty(),
                 "open tagged Rec.2020 PQ fixture")) return 1;
    auto positiveHdrError = hdrMedia.getHdrFrame (0.052, 16, 16, hdr);
    if (! check (positiveHdrError.empty(), "decode tagged Rec.2020 PQ to owned float RGBA"))
    {
        std::fprintf (stderr, "HDR decode: %s\n", positiveHdrError.c_str());
        return 1;
    }
    if (! check (hdr.width == 16 && hdr.height == 16 && hdr.strideFloats == 64
                 && hdr.rgba.size() == 16u * 16u * 4u
                 && hdr.ptsTicks == 512 && hdr.timeBase.num == 1 && hdr.timeBase.den == 12800
                 && std::abs (hdr.ptsSec - 0.04) < 1.0e-9,
                 "float HDR frame owns full dimensions and selected-frame timing")) return 1;
    ok &= check (hdr.color.primaries == colortransform::ColorSpace::Rec2020
                 && hdr.color.transfer == colortransform::TransferFunction::PQ
                 && hdr.matrix == AVCOL_SPC_BT2020_NCL && hdr.range == AVCOL_RANGE_MPEG,
                 "float HDR frame carries the source color declaration");
    ok &= check (std::abs (hdr.rgba[0] - 0.654f) < 0.02f
                 && std::abs (hdr.rgba[1] - 0.474f) < 0.02f
                 && std::abs (hdr.rgba[2] - 0.275f) < 0.02f
                 && hdr.rgba[3] == 1.0f,
                 "BT.2020 YUV becomes encoded float RGB in RGBA channel order");
    const auto hdrPixels = hdr.rgba;
    hdr.rgba[0] = 0.0f;
    positiveHdrError = hdrMedia.getHdrFrame (0.060, 16, 16, hdr);
    ok &= check (positiveHdrError.empty() && hdr.ptsTicks == 512 && hdr.rgba == hdrPixels,
                 "repeated HDR request copies stable pixels from the selected frame");

    const DecodedHdrFrame retainedHdr = hdr;
    const auto firstHdrError = hdrMedia.getHdrFrame (0.010, 16, 16, hdr);
    if (! check (firstHdrError.empty() && hdr.rgba.size() == hdrPixels.size()
                 && hdr.ptsTicks == 0 && hdr.timeBase.num == 1 && hdr.timeBase.den == 12800
                 && hdr.ptsSec == 0.0 && hdr.rgba != hdrPixels,
                 "backward HDR selection reaches the distinct first frame at exact PTS zero")) return 1;
    // Limited-range BT.2020 neutral Y=512 gives R=G=B=(512-64)/(940-64).
    ok &= check (std::abs (hdr.rgba[0] - 448.0f / 876.0f) < 0.02f
                 && std::abs (hdr.rgba[1] - 448.0f / 876.0f) < 0.02f
                 && std::abs (hdr.rgba[2] - 448.0f / 876.0f) < 0.02f && hdr.rgba[3] == 1.0f,
                 "first HDR frame contains the independent neutral YUV pixel value");
    const auto firstHdrPixels = hdr.rgba;
    const DecodedHdrFrame retainedFirstHdr = hdr;
    const auto nextHdrError = hdrMedia.getHdrFrame (0.052, 16, 16, hdr);
    ok &= check (nextHdrError.empty() && hdr.ptsTicks == 512 && hdr.rgba == hdrPixels
                 && hdr.timeBase.num == 1 && hdr.timeBase.den == 12800
                 && std::abs (hdr.ptsSec - 0.04) < 1.0e-9
                 && retainedFirstHdr.rgba == firstHdrPixels && retainedFirstHdr.ptsTicks == 0
                 && retainedFirstHdr.timeBase.num == 1 && retainedFirstHdr.timeBase.den == 12800
                 && retainedFirstHdr.ptsSec == 0.0
                 && retainedHdr.rgba == hdrPixels && retainedHdr.ptsTicks == 512
                 && retainedHdr.timeBase.num == 1 && retainedHdr.timeBase.den == 12800
                 && std::abs (retainedHdr.ptsSec - 0.04) < 1.0e-9,
                 "owned HDR pixels and exact PTS survive backward and forward decoder advancement");
    const auto tailHdrError = hdrMedia.getHdrFrame (0.120, 16, 16, hdr);
    ok &= check (tailHdrError.empty() && hdr.ptsTicks == 512 && hdr.rgba == hdrPixels,
                 "HDR request beyond the two-frame clip retains its final selected frame");
    DecodedHdrFrame previewHdr, exportHdr;
    ok &= check(hdrMedia.getHdrFrame(0.052,16,16,previewHdr).empty()
        && hdrMedia.getHdrFrame(0.052,16,16,exportHdr).empty()
        && previewHdr.ptsTicks == exportHdr.ptsTicks && previewHdr.timeBase.num == exportHdr.timeBase.num
        && previewHdr.timeBase.den == exportHdr.timeBase.den && previewHdr.rgba == exportHdr.rgba,
        "shared preview/export HDR selection returns identical exact-PTS source pixels");

    MediaContext untaggedHdr;
    if (! check (untaggedHdr.open (fixtures + "/hdr-pq-no-primaries-16x16.mp4", false).empty(),
                 "open PQ fixture without primaries")) return 1;
    const auto missingTagError = untaggedHdr.getHdrFrame (0.052, 16, 16, hdr);
    ok &= check (missingTagError.find ("requires declared Rec.2020") != std::string::npos
                 && hdr.rgba.empty() && hdr.ptsTicks == AV_NOPTS_VALUE,
                 "PQ frame without Rec.2020 primaries cannot borrow prior float pixels");

    if (! decode (timed, 0.052, frame)) return 1;
    ok &= check (frame.ptsTicks == 512
                 && frame.timeBase.num == 1 && frame.timeBase.den == 12800
                 && std::abs (frame.ptsSec - 0.04) < 1.0e-9,
                 "selected frame carries FFmpeg ticks and stream time base, not request time");
    const auto firstPixels = frame.rgba;

    uint64_t ordinal = UINT64_MAX;
    auto ordinalError = timed.getFrameWithPresentationOrdinal (0.052, 16, 16, frame, ordinal);
    ok &= check (ordinalError.empty() && ordinal == 1 && frame.ptsTicks == 512,
                 "presentation ordinal comes from decoder output order");

    if (! decode (timed, 0.060, frame)) return 1;
    ok &= check (frame.ptsTicks == 512 && frame.rgba == firstPixels,
                 "repeat request retains the selected frame timestamp and pixels");

    if (! decode (timed, 0.100, frame)) return 1;
    ok &= check (frame.ptsTicks == 1024 && std::abs (frame.ptsSec - 0.08) < 1.0e-9,
                 "forward decode reports the next selected frame timestamp");
    ordinalError = timed.getFrameWithPresentationOrdinal (0.100, 16, 16, frame, ordinal);
    ok &= check (ordinalError.empty() && ordinal == 2,
                 "forward selection keeps the independently counted presentation ordinal");

    if (! decode (timed, 0.010, frame)) return 1;
    ok &= check (frame.ptsTicks == 0 && std::abs (frame.ptsSec) < 1.0e-9,
                 "backward seek reports the selected frame timestamp");
    ordinalError = timed.getFrameWithPresentationOrdinal (0.010, 16, 16, frame, ordinal);
    ok &= check (ordinalError.empty() && ordinal == 0,
                 "backward seek independently resolves the first presentation ordinal");

    opticalflowoperation::SourcePair pair;
    pair.images = { opticalflowoperation::SourceEndpoint { 10, 0 },
                    opticalflowoperation::SourceEndpoint { 20, 0 } };
    videowire::CompiledVisualLayerPlan plan;
    plan.clipId = 1;
    plan.structuralRevision = 1;
    plan.producerValidated = true;
    plan.operations = {
        { 10, "video.source", "source-decode", opticalflowoperation::serializeSourceClip (1) },
        { 20, "video.layer.source", "source-decode", opticalflowoperation::serializeSourceClip (2) },
        { 30, "visual.optical-flow", "native-gpu", opticalflowoperation::serializeSourcePair (pair) }
    };
    std::vector<videowire::RenderSegment> segments (2);
    std::vector<videowire::OpticalFlowSourceBinding> catalog (2);
    for (int index = 0; index < 2; ++index)
    {
        segments[index].clipId = index + 1;
        segments[index].sourceKind = videowire::SourceKind::Media;
        segments[index].sourcePath = fixtures + "/frames.mp4";
        segments[index].outSec = 0.12;
        catalog[index].clipId = index + 1;
        catalog[index].sourceKind = videowire::SourceKind::Media;
        catalog[index].sourcePath = segments[index].sourcePath;
        catalog[index].sourceGeneration = 1;
        catalog[index].snapshotGeneration = 2;
    }
    std::shared_ptr<const videoopticalflow::DecodedPairReceipt> receipt;
    auto pairError = videoopticalflow::resolveDecodedPair (
        plan, pair, catalog, segments, 3, 0.052, 16, 16, receipt);
    ok &= check (pairError.find ("stale") != std::string::npos,
                 "stale snapshot generation rejects the pair before decoding");
    segments[0].sourcePath = catalog[0].sourcePath = fixtures + "/red-vfr-b.mp4";
    segments[0].outSec = 0.28;
    segments[1].sourcePath = fixtures + "/moved.mp4";
    pairError = videoopticalflow::resolveDecodedPair (
        plan, pair, catalog, segments, 2, 0.052, 16, 16, receipt);
    ok &= check (pairError.find ("stale") != std::string::npos,
                 "changed timeline media path rejects the saved source binding");
    segments[0].sourcePath = catalog[0].sourcePath = fixtures + "/frames.mp4";
    segments[0].outSec = 0.12;
    segments[1].sourcePath = catalog[1].sourcePath;
    pairError = videoopticalflow::resolveDecodedPair (
        plan, pair, catalog, segments, 2, 0.052, 16, 16, receipt);
    ok &= check (pairError.find ("unsupported color") != std::string::npos,
                 "undeclared decoded transfer cannot acquire an optical source receipt");
    ok &= check (! receipt, "failed resolution publishes no partial pair");

    const char* names[] { "red-vfr-b.mp4", "blue%vfr-b.mp4" };
    const char* sourceHashes[] {
        "f88d018e6918be4dcf0536b480e1818547093f3994f53b67e78ff748d376c0f2",
        "802a14b7400e8b23ae9ea6970349770206a7a12b88623b6166ba38ad8c9cab6b"
    };
    const char* pixelHashes[] {
        "00bf614ef7cc9d64d61a320ecf8e45d2d951c462f3bfca8e965892729894ba46",
        "4dc8da78ba127d616a4ca4fd566f9df7101040868821fb655b00a5a5b93b74e8"
    };
    for (int index = 0; index < 2; ++index)
    {
        segments[index].sourcePath = fixtures + "/" + names[index];
        segments[index].outSec = 0.28;
        catalog[index].sourcePath = segments[index].sourcePath;
    }
    for (const auto selection : { std::pair<double, int64_t> { 0.10, 512 },
                                   std::pair<double, int64_t> { 0.255, 3072 } })
    {
        receipt.reset();
        pairError = videoopticalflow::resolveDecodedPair (
            plan, pair, catalog, segments, 2, selection.first, 16, 16, receipt);
        ok &= check (pairError.empty() && receipt && receipt->snapshotGeneration == 2,
                     "distinct declared-color media sources resolve a complete pair");
        if (! receipt) continue;
        for (int index = 0; index < 2; ++index)
        {
            const auto& source = receipt->sources[index];
            ok &= check (source.clipId == static_cast<uint32_t> (index + 1)
                         && source.sourceGeneration == 1
                         && std::abs (source.sourceTimeSec - selection.first) < 1.0e-9
                         && source.sourceSha256 == sourceHashes[index],
                         "source identity, generation, time and content hash are exact");
            ok &= check (source.frame.width == 16 && source.frame.height == 16
                         && source.frame.strideBytes == 64
                         && source.frame.timeBase.num == 1 && source.frame.timeBase.den == 12800
                         && source.frame.ptsTicks == selection.second
                         && source.presentationOrdinal == (selection.second == 512 ? 1 : 4),
                         "VFR B-frame and delayed tail keep exact PTS and presentation ordinal");
            ok &= check (source.frame.color.primaries == colortransform::ColorSpace::Rec709
                         && source.frame.color.transfer == colortransform::TransferFunction::Rec709
                         && source.pixelSha256 == pixelHashes[index],
                         "declared SDR pixels match the expected decoded hash");
        }
    }

    Uploader uploader;
    uploader.capabilities.kind = videoopticalflow::BackendKind::NativeGpu;
    uploader.capabilities.supportsOpticalFlow = true;
    uploader.capabilities.implementation[0] = 1;
    uploader.capabilities.implementationRevision = 1;
    std::shared_ptr<const videoopticalflow::SourcePairLease> nativePair;
    auto leaseError = videoopticalflow::resolveAndLeaseSourcePair (
        plan, pair, catalog, segments, 2, 0.255, 16, 16, 5,
        uploader.capabilities, uploader, nativePair);
    ok &= check (leaseError.empty() && nativePair && nativePair->request
                 && nativePair->selection
                 && nativePair->selection->description().first.frame.timestamp.ticks == 3072
                 && nativePair->selection->description().second.frame.identity.frameIndex == 4,
                 "exact VFR pair receives two ordered native lease receipts");
    auto retained = nativePair ? nativePair->frames[0] : nullptr;
    nativePair.reset();
    ok &= check (uploader.released == 1 && retained,
                 "pair releases its second source while a retained first lease survives");
    retained.reset();
    ok &= check (uploader.released == 2, "retained source releases with its final owner");

    receipt.reset();
    pairError = videoopticalflow::resolveDecodedPair (
        plan, pair, catalog, segments, 2, 0.255, 16, 16, receipt);
    if (! check (pairError.empty() && receipt != nullptr,
                 "preview fixture keeps the worker-validated ordered pair")) return 1;
    PendingUploader pending;
    pending.capabilities = uploader.capabilities;
    leaseError = videoopticalflow::leaseDecodedSourcePair (
        plan, pair, *receipt, 2, 5, pending.capabilities, pending, nativePair,
        true, true);
    ok &= check (leaseError == videoopticalflow::kPreviewUploadPending
                 && ! nativePair && pending.previewRequests == 2,
                 "preview starts both exact source digests without publishing a partial pair");
    pending.ready[0] = true;
    leaseError = videoopticalflow::leaseDecodedSourcePair (
        plan, pair, *receipt, 2, 5, pending.capabilities, pending, nativePair,
        true, true);
    ok &= check (leaseError == videoopticalflow::kPreviewUploadPending
                 && ! nativePair && pending.uploads == 1,
                 "one completed digest cannot publish the pair");
    pending.ready[1] = true;
    leaseError = videoopticalflow::leaseDecodedSourcePair (
        plan, pair, *receipt, 2, 5, pending.capabilities, pending, nativePair,
        true, true);
    ok &= check (leaseError.empty() && nativePair && nativePair->request
                 && nativePair->selection
                 && nativePair->selection->description().first.clipId == 1
                 && nativePair->selection->description().second.clipId == 2,
                 "both completed digests retain the saved A/B source order");
    nativePair.reset();
    ok &= check (pending.released == 3,
                 "pending first lease and the completed pair release with their last owner");
    auto staleReceipt = *receipt;
    ++staleReceipt.snapshotGeneration;
    const int beforeStale = pending.previewRequests;
    leaseError = videoopticalflow::leaseDecodedSourcePair (
        plan, pair, staleReceipt, 2, 5, pending.capabilities, pending, nativePair,
        true, true);
    ok &= check (! leaseError.empty() && ! nativePair
                 && pending.previewRequests == beforeStale,
                 "stale preview owner never reaches native upload or publication");

    for (const auto fault : { Uploader::Fault::WrongOwner, Uploader::Fault::Extent,
                              Uploader::Fault::Color, Uploader::Fault::Pts,
                              Uploader::Fault::MissingSecond })
    {
        uploader.fault = fault;
        uploader.uploads = 0;
        const int before = uploader.released;
        leaseError = videoopticalflow::resolveAndLeaseSourcePair (
            plan, pair, catalog, segments, 2, 0.10, 16, 16, 5,
            uploader.capabilities, uploader, nativePair);
        ok &= check (! leaseError.empty() && ! nativePair
                     && uploader.released == before + uploader.uploads
                                                - (fault == Uploader::Fault::MissingSecond ? 1 : 0),
                     "wrong owner, extent, color, PTS and partial pairs fail atomically");
    }
    uploader.fault = Uploader::Fault::None;
    uploader.uploads = 0;
    catalog[1].sourceGeneration = 0;
    leaseError = videoopticalflow::resolveAndLeaseSourcePair (
        plan, pair, catalog, segments, 2, 0.10, 16, 16, 5,
        uploader.capabilities, uploader, nativePair);
    ok &= check (! leaseError.empty() && ! nativePair && uploader.uploads == 0,
                 "stale source generation rejects before GPU upload");
    catalog[1].sourceGeneration = 1;

    // The actual-editor driver generates these two independently authored
    // lossless RGB sources. Exercise its paused seek through the production
    // pair decoder, including color admission and exact presentation timing.
    if (argc == 3)
    {
        const char* images[] { "first", "second" };
        for (int index = 0; index < 2; ++index)
        {
            segments[index].sourcePath = std::string (argv[2]) + "/" + images[index] + ".mkv";
            segments[index].outSec = 2.0;
            catalog[index].sourcePath = segments[index].sourcePath;
        }
        receipt.reset();
        pairError = videoopticalflow::resolveDecodedPair (
            plan, pair, catalog, segments, 2, 0.5, 960, 540, receipt);
        if (! check (pairError.empty() && receipt != nullptr,
                     "actual-editor RGB fixtures resolve the paused two-image pair"))
        {
            std::fprintf (stderr, "Optical fixture decode: %s\n", pairError.c_str());
            return 1;
        }
        for (int index = 0; index < 2; ++index)
        {
            MediaContext still;
            DecodedFrame authored;
            const auto path = std::string (argv[2]) + "/" + images[index] + ".png";
            if (! check (still.open (path, false).empty()
                         && still.getFrame (0.0, 960, 540, authored).empty(),
                         "open the driver's independently authored source PNG")) return 1;
            const auto& source = receipt->sources[index];
            ok &= check (source.clipId == static_cast<std::uint32_t> (index + 1)
                         && source.sourceGeneration == 1 && source.sourceTimeSec == 0.5
                         && source.presentationOrdinal == 15 && source.frame.ptsTicks == 500
                         && source.frame.timeBase.num == 1 && source.frame.timeBase.den == 1000
                         && source.frame.color.primaries == colortransform::ColorSpace::SRGB
                         && source.frame.color.transfer == colortransform::TransferFunction::SRGB
                         && source.frame.rgba == authored.rgba,
                         "paused optical source retains its owner, sRGB pixels and exact 0.500s PTS");
        }
        ok &= check (receipt->sources[0].sourceSha256 != receipt->sources[1].sourceSha256
                     && receipt->sources[0].pixelSha256 != receipt->sources[1].pixelSha256,
                     "actual-editor inputs are two distinct owned source images");
    }

    frame.setPresentationTimestamp (AV_NOPTS_VALUE, AVRational { 1, 1200000 });
    ok &= check (frame.ptsTicks == AV_NOPTS_VALUE
                 && frame.timeBase.num == 1 && frame.timeBase.den == 1200000,
                 "missing FFmpeg presentation timestamp remains absent after reusing output");
    frame.setPresentationTimestamp (42, AVRational { 0, 1 });
    ok &= check (frame.ptsTicks == AV_NOPTS_VALUE
                 && frame.timeBase.num == 0 && frame.timeBase.den == 1,
                 "invalid stream time base cannot make ticks appear valid");

    if (ok) std::puts ("PASS: decoded frame presentation timestamps");
    return ok ? 0 : 1;
}
