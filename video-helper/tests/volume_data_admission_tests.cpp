#include "../src/volume_data_admission.h"
#include "../src/volume_native_renderer.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
using videohelper::volume::AdmittedSparseBrick;
using videohelper::volume::AdmittedVolume;
using videohelper::volume::VolumeAdmissionLimits;
using videohelper::volume::VolumeRendererCapabilities;

int failures = 0;

void check (bool condition, const char* message)
{
    if (condition) return;
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

VolumeRendererCapabilities supportedRenderer()
{
    VolumeRendererCapabilities capabilities;
    capabilities.nativeGpuAvailable = true;
    capabilities.volumeRaymarch = true;
    capabilities.denseVolumeUpload = true;
    capabilities.sparseBrickUpload = true;
    capabilities.maxTexture3DDimension = 256;
    capabilities.maxBrickEdge = 16;
    capabilities.maxSparseBrickCount = 64;
    capabilities.maxVolumeBytes = 1024 * 1024;
    return capabilities;
}

class FakeFrame final : public videohelper::volume::NativeVolumeFrame
{
public:
    const std::string& backend() const noexcept override { return backend_; }
    std::uint32_t width() const noexcept override { return 32; }
    std::uint32_t height() const noexcept override { return 24; }
    std::uintptr_t colorImageHandle() const noexcept override { return 1; }
    std::uintptr_t colorTextureViewHandle() const noexcept override { return 2; }
private:
    std::string backend_ = "fake-gpu";
};

class FakeBackend final : public videohelper::volume::NativeVolumeExecutionBackend
{
public:
    videohelper::volume::NativeVolumeExecutionCapabilities capabilities() const override
    {
        return { supportedRenderer(), "fake-gpu", 128, 128u * 128u };
    }
    videohelper::volume::NativeVolumeSubmission render (
        const videohelper::volume::NativeVolumeDrawRequest& request) override
    {
        ++draws;
        receivedIdentity = request.volume == nullptr ? "" : request.volume->cacheIdentity();
        return { true, std::make_shared<FakeFrame>(), {} };
    }
    int draws = 0;
    std::string receivedIdentity;
};

videowire::DenseVolumeDescriptor tinyDense()
{
    videowire::DenseVolumeDescriptor descriptor;
    descriptor.dimensions = { 2, 2, 2 };
    descriptor.bounds = { { -1.0f, -2.0f, -3.0f }, { 1.0f, 2.0f, 3.0f } };
    descriptor.transform.localToWorld[12] = 4.0f;
    descriptor.voxels = { 0, 1, 2, 3, 4, 5, 6, 255 };
    return descriptor;
}

videowire::SparseVolumeDescriptor tinySparse()
{
    videowire::SparseVolumeDescriptor descriptor;
    descriptor.dimensions = { 3, 2, 2 };
    descriptor.brickEdge = 2;
    descriptor.bricks = {
        { 1, 0, 0, { 9, 10, 11, 12 } },
        { 0, 0, 0, { 1, 2, 3, 4, 5, 6, 7, 8 } }
    };
    return descriptor;
}

bool denseRejected (const videowire::DenseVolumeDescriptor& descriptor,
                    VolumeAdmissionLimits limits,
                    VolumeRendererCapabilities capabilities,
                    std::string* diagnostic = nullptr)
{
    std::string error;
    const auto admitted = videohelper::volume::admitDenseVolume (
        descriptor, limits, capabilities, error);
    if (diagnostic != nullptr) *diagnostic = error;
    return ! admitted.has_value() && ! error.empty();
}

bool sparseRejected (const videowire::SparseVolumeDescriptor& descriptor,
                     VolumeAdmissionLimits limits,
                     VolumeRendererCapabilities capabilities,
                     std::string* diagnostic = nullptr)
{
    std::string error;
    const auto admitted = videohelper::volume::admitSparseVolume (
        descriptor, limits, capabilities, error);
    if (diagnostic != nullptr) *diagnostic = error;
    return ! admitted.has_value() && ! error.empty();
}
} // namespace

int main()
{
    static_assert (! videowire::VolumeRenderContract::kAllowsCpuProductionFallback,
                   "volume rendering must not gain a CPU production fallback");
    static_assert (videowire::VolumeRenderContract::kRequirement
                       == videowire::VolumeRendererRequirement::nativeGpuRaymarch,
                   "the volume contract must require native GPU raymarch support");
    static_assert (std::is_same<decltype (std::declval<const AdmittedVolume&>().bytes()),
                                const std::vector<std::uint8_t>&>::value,
                   "admitted volume bytes must be exposed read-only");
    static_assert (std::is_same<decltype (std::declval<const AdmittedVolume&>().bricks()),
                                const std::vector<AdmittedSparseBrick>&>::value,
                   "admitted sparse brick descriptors must be exposed read-only");

    const auto capabilities = supportedRenderer();
    std::string error;

    auto denseSource = tinyDense();
    const auto dense = videohelper::volume::admitDenseVolume (
        denseSource, {}, capabilities, error);
    check (dense.has_value() && error.empty(), "a tiny dense volume is admitted");
    check (dense && dense->storage() == videowire::VolumeStorage::dense
                 && dense->voxelCount() == 8 && dense->bytes() == denseSource.voxels,
           "dense admission preserves exact voxel bytes and dimensions");
    check (dense && dense->cacheIdentity()
                       == "a75695a722dc63f80eded9993f5a66cd7e8bdc31b8e30692c5a7dd257995d71f",
           "the dense fixture has a stable cross-process cache identity");

    if (dense)
    {
        auto admitted = std::make_shared<const AdmittedVolume> (*dense);
        FakeBackend backend;
        videohelper::volume::NativeVolumeRenderer renderer (backend);
        videohelper::volume::NativeVolumeRenderedFrame preview;
        videohelper::volume::NativeVolumeRenderedFrame exportFrame;
        check (renderer.renderPreview (
                   admitted, 32, 24, videohelper::volume::kNativeVolumeGpuCapability,
                   preview, error)
               && renderer.renderExport (
                   admitted, 32, 24, videohelper::volume::kNativeVolumeGpuCapability,
                   exportFrame, error)
               && backend.draws == 2 && backend.receivedIdentity == dense->cacheIdentity()
               && preview.use == videohelper::volume::NativeVolumeRenderUse::Preview
               && exportFrame.use == videohelper::volume::NativeVolumeRenderUse::Export,
               "preview and export dispatch the same admitted volume to a capable GPU backend");

        class MissingDenseBackend final : public videohelper::volume::NativeVolumeExecutionBackend
        {
        public:
            videohelper::volume::NativeVolumeExecutionCapabilities capabilities() const override
            {
                auto result = supportedRenderer();
                result.denseVolumeUpload = false;
                return { result, "fake-gpu", 128, 128u * 128u };
            }
            videohelper::volume::NativeVolumeSubmission render (
                const videohelper::volume::NativeVolumeDrawRequest&) override
            {
                ++draws;
                return {};
            }
            int draws = 0;
        } missingDense;
        videohelper::volume::NativeVolumeRenderer rejectingRenderer (missingDense);
        check (! rejectingRenderer.renderPreview (
                   admitted, 32, 24, videohelper::volume::kNativeVolumeGpuCapability,
                   preview, error)
               && missingDense.draws == 0
               && error == "renderer lacks dense volume upload capability",
               "missing native upload capability fails before backend execution");
    }

    const auto ownedDenseBytes = dense ? dense->bytes() : std::vector<std::uint8_t> {};
    denseSource.voxels.assign (8, 99);
    check (dense && dense->bytes() == ownedDenseBytes,
           "dense admission owns immutable bytes independent of its source");

    auto equivalentDense = tinyDense();
    equivalentDense.bounds.minimum.x = -1.0f;
    equivalentDense.transform.localToWorld[1] = -0.0f;
    const auto equivalentDenseAdmission = videohelper::volume::admitDenseVolume (
        equivalentDense, {}, capabilities, error);
    check (dense && equivalentDenseAdmission
           && dense->cacheIdentity() == equivalentDenseAdmission->cacheIdentity(),
           "equivalent finite placement has a stable cache identity");

    auto sparseSource = tinySparse();
    const auto sparse = videohelper::volume::admitSparseVolume (
        sparseSource, {}, capabilities, error);
    check (sparse.has_value() && error.empty(), "a tiny compact sparse volume is admitted");
    check (sparse && sparse->voxelCount() == 12 && sparse->brickEdge() == 2
                  && sparse->bricks().size() == 2 && sparse->bytes().size() == 12,
           "sparse admission reports logical dimensions and compact resident bytes");
    check (sparse && sparse->bricks()[0].brickX == 0 && sparse->bricks()[0].byteOffset == 0
                  && sparse->bricks()[0].byteCount == 8
                  && sparse->bricks()[1].brickX == 1 && sparse->bricks()[1].width == 1
                  && sparse->bricks()[1].byteOffset == 8
                  && sparse->bricks()[1].byteCount == 4,
           "sparse bricks are canonicalized with compact edge extents and byte ranges");
    check (sparse && sparse->bytes() == std::vector<std::uint8_t> ({ 1, 2, 3, 4, 5, 6,
                                                                    7, 8, 9, 10, 11, 12 }),
           "sparse admission owns exact brick bytes in canonical coordinate order");

    check (sparse && sparse->cacheIdentity()
                        == "2c3dfff12e23573f6163758797dfb8d48251594ab27a32b2a543b12715f7158c",
           "the sparse fixture has a stable cross-process cache identity");

    auto reorderedSparse = tinySparse();
    std::reverse (reorderedSparse.bricks.begin(), reorderedSparse.bricks.end());
    const auto reorderedSparseAdmission = videohelper::volume::admitSparseVolume (
        reorderedSparse, {}, capabilities, error);
    check (sparse && reorderedSparseAdmission
           && sparse->cacheIdentity() == reorderedSparseAdmission->cacheIdentity(),
           "sparse cache identity does not depend on descriptor order");
    const auto ownedSparseBytes = sparse ? sparse->bytes() : std::vector<std::uint8_t> {};
    sparseSource.bricks[0].voxels.assign (4, 42);
    check (sparse && sparse->bytes() == ownedSparseBytes,
           "sparse admission owns immutable bytes independent of brick sources");

    auto overflow = tinyDense();
    overflow.dimensions = { std::numeric_limits<std::uint32_t>::max(),
                            std::numeric_limits<std::uint32_t>::max(),
                            std::numeric_limits<std::uint32_t>::max() };
    overflow.voxels.clear();
    VolumeAdmissionLimits permissiveLimits;
    permissiveLimits.maxDimension = std::numeric_limits<std::uint32_t>::max();
    permissiveLimits.maxVoxelCount = std::numeric_limits<std::size_t>::max();
    permissiveLimits.maxPayloadBytes = std::numeric_limits<std::size_t>::max();
    auto permissiveCapabilities = capabilities;
    permissiveCapabilities.maxTexture3DDimension = std::numeric_limits<std::uint32_t>::max();
    permissiveCapabilities.maxVolumeBytes = std::numeric_limits<std::size_t>::max();
    std::string diagnostic;
    check (denseRejected (overflow, permissiveLimits, permissiveCapabilities, &diagnostic)
               && diagnostic == "volume voxel count overflow",
           "dimension multiplication overflow is rejected before payload access");

    auto malformedBrick = tinySparse();
    malformedBrick.bricks[0].voxels.push_back (13);
    check (sparseRejected (malformedBrick, {}, capabilities, &diagnostic)
               && diagnostic == "sparse volume brick payload size does not match its compact edge extent",
           "malformed compact edge-brick payloads are rejected");

    auto duplicateBrick = tinySparse();
    duplicateBrick.bricks[1].brickX = 1;
    duplicateBrick.bricks[1].voxels.resize (4);
    check (sparseRejected (duplicateBrick, {}, capabilities),
           "duplicate sparse brick coordinates are rejected");

    auto nonFiniteTransform = tinyDense();
    nonFiniteTransform.transform.localToWorld[6]
        = std::numeric_limits<float>::infinity();
    check (denseRejected (nonFiniteTransform, {}, capabilities, &diagnostic)
               && diagnostic == "volume transform contains a non-finite value",
           "non-finite transforms are rejected");

    auto unsupported = capabilities;
    unsupported.volumeRaymarch = false;
    check (denseRejected (tinyDense(), {}, unsupported, &diagnostic)
               && diagnostic == "renderer lacks native GPU volume raymarch capability",
           "a backend without native GPU raymarch support is rejected explicitly");

    auto memoryBounded = tinyDense();
    auto smallMemoryCapabilities = capabilities;
    smallMemoryCapabilities.maxVolumeBytes = 7;
    check (denseRejected (memoryBounded, {}, smallMemoryCapabilities),
           "renderer memory budgets are enforced before immutable byte allocation");

    auto brickBounded = tinySparse();
    auto singleBrickCapabilities = capabilities;
    singleBrickCapabilities.maxSparseBrickCount = 1;
    check (sparseRejected (brickBounded, {}, singleBrickCapabilities),
           "renderer sparse brick budgets are enforced before allocation");

    if (dense) std::cout << "dense cache identity: " << dense->cacheIdentity() << '\n';
    if (sparse) std::cout << "sparse cache identity: " << sparse->cacheIdentity() << '\n';
    if (failures != 0) return EXIT_FAILURE;
    std::cout << "volume data admission checks passed\n";
    return EXIT_SUCCESS;
}
