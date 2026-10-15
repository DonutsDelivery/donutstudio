#include "../src/optical_flow_native_source_upload.h"

#include <array>
#include <cstdio>

namespace
{
int failures = 0;
void check (bool okay, const char* message)
{
    if (! okay) { ++failures; std::fprintf (stderr, "FAIL: %s\n", message); }
}

class TestResource final : public videoopticalflow::UploadedSourceResource
{
public:
    TestResource (std::uintptr_t handle, int& releases)
        : handle_ (handle), releases_ (releases) {}
    ~TestResource() override { ++releases_; }
    std::uintptr_t imageHandle() const noexcept override { return handle_; }
    std::uintptr_t nativeContextIdentity() const noexcept override { return 77; }
private:
    std::uintptr_t handle_ = 0;
    int& releases_;
};
}

int main()
{
    using namespace videoopticalflow;
    DecodedSourceReceipt decoded;
    decoded.clipId = 17;
    decoded.sourceGeneration = 9;
    decoded.sourceSha256 = std::string (64, 'a');
    decoded.presentationOrdinal = 3;
    decoded.frame.width = 2;
    decoded.frame.height = 1;
    decoded.frame.strideBytes = 8;
    decoded.frame.rgba = { 0, 0, 0, 255, 255, 128, 64, 255 };
    decoded.frame.timeBase = { 1, 24000 };
    decoded.frame.ptsTicks = 1001;
    decoded.frame.color.primaries = colortransform::ColorSpace::SRGB;
    decoded.frame.color.transfer = colortransform::TransferFunction::SRGB;
    decoded.pixelSha256 = videohelper::sha256Bytes (
        decoded.frame.rgba.data(), decoded.frame.rgba.size());
    SourceFrameV2 expected;
    colortransform::Description color;
    std::string error;
    if (! detail::expectedSource (decoded, expected, color, error))
    {
        std::fprintf (stderr, "invalid optical-flow source fixture: %s\n", error.c_str());
        return 2;
    }
    colortransform::AdmissionFailure colorFailure {};
    const auto transform = colortransform::admit (
        color, colortransform::BackendCapability::NativeGpu, colorFailure);
    if (! transform) return 2;
    check (validateUploadSource (decoded, *transform, expected, error),
           "native upload accepts the exact decoded frame and color admission");
    auto wrong = expected;
    ++wrong.clipId;
    check (! validateUploadSource (decoded, *transform, wrong, error),
           "native upload rejects a wrong saved clip owner");
    wrong = expected;
    ++wrong.frame.timestamp.ticks;
    check (! validateUploadSource (decoded, *transform, wrong, error),
           "native upload rejects a stale presentation timestamp");
    wrong = expected;
    ++wrong.frame.extent.width;
    check (! validateUploadSource (decoded, *transform, wrong, error),
           "native upload rejects a mismatched extent");
    wrong = expected;
    wrong.frame.format = renderpassoutput::PixelFormat::RG16Float;
    check (! validateUploadSource (decoded, *transform, wrong, error),
           "native upload rejects a non-RGBA16F source claim");
    auto changed = decoded;
    changed.frame.rgba[0] = 1;
    check (! validateUploadSource (changed, *transform, expected, error),
           "native upload rejects changed decoder-owned pixels");
    color.input.colorSpace = colortransform::ColorSpace::DisplayP3;
    const auto wrongTransform = colortransform::admit (
        color, colortransform::BackendCapability::NativeGpu, colorFailure);
    check (wrongTransform.has_value()
           && ! validateUploadSource (decoded, *wrongTransform, expected, error),
           "native upload rejects a different admitted color conversion");

    const std::array<std::uint8_t, 16> linearRgba16f {
        0, 0x3c, 0, 0, 0, 0, 0, 0x3c,
        0, 0, 0, 0x3c, 0, 0, 0, 0x3c
    };
    ContentIdentity exactDigest {}, changedDigest {};
    check (detail::decodeSha256 (videohelper::sha256Bytes (
               linearRgba16f.data(), linearRgba16f.size()), exactDigest),
           "completed readback digests every exact linear RGBA16F byte");
    auto changedLinear = linearRgba16f;
    changedLinear[0] = 1;
    check (detail::decodeSha256 (videohelper::sha256Bytes (
               changedLinear.data(), changedLinear.size()), changedDigest)
           && changedDigest != exactDigest,
           "a changed half-float source byte changes the V2 content identity");

    BackendCapabilities backend;
    backend.kind = BackendKind::NativeGpu;
    backend.supportsOpticalFlow = true;
    backend.implementation[0] = 1;
    backend.implementationRevision = 2;
    ContentIdentity firstPixels {}, secondPixels {};
    firstPixels[0] = 1; secondPixels[0] = 2;
    int releases = 0;
    auto resource = std::make_shared<TestResource> (101, releases);
    auto first = std::make_shared<const UploadedSourceFrame> (
        expected, firstPixels, backend, 11, 12, resource);
    auto second = std::make_shared<const UploadedSourceFrame> (
        expected, secondPixels, backend, 11, 12,
        std::make_shared<TestResource> (102, releases));
    check (first->frame().identity.content == firstPixels
           && second->frame().identity.content == secondPixels
           && first->resourceIdentity() != second->resourceIdentity()
           && first->sourceClipId() == 17 && first->sourceGeneration() == 9
           && first->frame().timestamp.ticks == 1001
           && first->imageHandle() == 101 && second->imageHandle() == 102
           && first->nativeContextIdentity() == 77
           && first->byteCount() == 16,
           "distinct leases keep exact owner, PTS, linear content and resource receipts");
    first.reset();
    second.reset();
    check (releases == 1, "each frame releases its own native resource owner");
    resource.reset();
    check (releases == 2, "an externally held native owner survives frame release");
    std::printf ("optical-flow-native-source-upload %s failures=%d\n",
                 failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
