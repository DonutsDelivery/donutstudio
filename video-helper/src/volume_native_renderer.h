#pragma once

#include "volume_data_admission.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace videohelper::volume
{
inline constexpr std::string_view kNativeVolumeGpuCapability = "native-gpu";

struct NativeVolumeExecutionCapabilities final
{
    VolumeRendererCapabilities volume;
    std::string backend;
    std::uint32_t maxRenderExtent = 0;
    std::uint64_t maxRenderPixels = 0;
};

struct NativeVolumeDrawRequest final
{
    std::shared_ptr<const AdmittedVolume> volume;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

// One submitted frame owns the backend-native 3D upload, raymarch target, and
// sampled color view. No CPU pixels or filesystem authority cross this seam.
class NativeVolumeFrame
{
public:
    virtual ~NativeVolumeFrame() = default;
    virtual const std::string& backend() const noexcept = 0;
    virtual std::uint32_t width() const noexcept = 0;
    virtual std::uint32_t height() const noexcept = 0;
    virtual std::uintptr_t colorImageHandle() const noexcept = 0;
    virtual std::uintptr_t colorTextureViewHandle() const noexcept = 0;
};

struct NativeVolumeSubmission final
{
    bool rendered = false;
    std::shared_ptr<const NativeVolumeFrame> frame;
    std::string error;
};

class NativeVolumeExecutionBackend
{
public:
    virtual ~NativeVolumeExecutionBackend() = default;
    virtual NativeVolumeExecutionCapabilities capabilities() const = 0;
    virtual NativeVolumeSubmission render (const NativeVolumeDrawRequest& request) = 0;
};

// Process-owned backend shared by product preview and export. Builds without a
// physically implemented native volume path return a fail-closed backend.
NativeVolumeExecutionBackend& nativeVolumeExecutionBackend();

enum class NativeVolumeRenderUse : std::uint8_t
{
    Preview = 0,
    Export = 1
};

struct NativeVolumeRenderedFrame final
{
    NativeVolumeRenderUse use = NativeVolumeRenderUse::Preview;
    std::string volumeIdentity;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::shared_ptr<const NativeVolumeFrame> nativeFrame;
};

// Preview and export both enter this exact renderer. It performs bounded target
// admission and dispatches only to a backend that explicitly owns native volume
// upload and raymarch execution. There is deliberately no CPU implementation.
class NativeVolumeRenderer final
{
public:
    explicit NativeVolumeRenderer (
        NativeVolumeExecutionBackend& backend = nativeVolumeExecutionBackend()) noexcept
        : backend_ (backend) {}

    bool renderPreview (const std::shared_ptr<const AdmittedVolume>& volume,
                        std::uint32_t width, std::uint32_t height,
                        std::string_view backendCapability,
                        NativeVolumeRenderedFrame& output, std::string& error)
    {
        return render (volume, width, height, backendCapability,
                       NativeVolumeRenderUse::Preview, output, error);
    }

    bool renderExport (const std::shared_ptr<const AdmittedVolume>& volume,
                       std::uint32_t width, std::uint32_t height,
                       std::string_view backendCapability,
                       NativeVolumeRenderedFrame& output, std::string& error)
    {
        return render (volume, width, height, backendCapability,
                       NativeVolumeRenderUse::Export, output, error);
    }

private:
    bool render (const std::shared_ptr<const AdmittedVolume>& volume,
                 std::uint32_t width, std::uint32_t height,
                 std::string_view backendCapability, NativeVolumeRenderUse use,
                 NativeVolumeRenderedFrame& output, std::string& error)
    {
        if (volume == nullptr)
        {
            error = "native GPU volume rendering requires an immutable admitted volume";
            return false;
        }
        if (backendCapability != kNativeVolumeGpuCapability)
        {
            error = "native volume rendering requires backendCapability native-gpu";
            return false;
        }
        const auto capabilities = backend_.capabilities();
        std::uint64_t pixels = 0;
        if (! capabilities.volume.nativeGpuAvailable || ! capabilities.volume.volumeRaymarch
            || capabilities.backend.empty() || capabilities.maxRenderExtent == 0
            || capabilities.maxRenderPixels == 0 || width == 0 || height == 0
            || width > capabilities.maxRenderExtent || height > capabilities.maxRenderExtent
            || (height != 0 && width > UINT64_MAX / height)
            || (pixels = static_cast<std::uint64_t> (width) * height) > capabilities.maxRenderPixels)
        {
            error = "native GPU volume render target or backend capability is unavailable";
            return false;
        }
        std::string capabilityError;
        if (! detail::validateCapabilities (capabilities.volume, volume->storage(), capabilityError)
            || volume->bytes().size() > capabilities.volume.maxVolumeBytes)
        {
            error = capabilityError.empty()
                ? "admitted volume exceeds the executing backend byte budget"
                : std::move (capabilityError);
            return false;
        }

        NativeVolumeDrawRequest request { volume, width, height };
        auto submission = backend_.render (request);
        if (! submission.rendered || submission.frame == nullptr)
        {
            error = submission.error.empty()
                ? "native GPU volume backend did not return a rendered frame"
                : std::move (submission.error);
            return false;
        }
        if (submission.frame->backend() != capabilities.backend
            || submission.frame->width() != width || submission.frame->height() != height
            || submission.frame->colorImageHandle() == 0
            || submission.frame->colorTextureViewHandle() == 0)
        {
            error = "native GPU volume backend returned an incompatible frame";
            return false;
        }

        output = { use, volume->cacheIdentity(), width, height, std::move (submission.frame) };
        error.clear();
        return true;
    }

    NativeVolumeExecutionBackend& backend_;
};

// The product route fails closed until a platform backend explicitly replaces
// this owner. It never evaluates or uploads the volume on the CPU.
inline NativeVolumeExecutionBackend& unavailableNativeVolumeExecutionBackend()
{
    class UnavailableBackend final : public NativeVolumeExecutionBackend
    {
    public:
        NativeVolumeExecutionCapabilities capabilities() const override { return {}; }
        NativeVolumeSubmission render (const NativeVolumeDrawRequest&) override
        {
            return { false, {}, "native GPU volume execution backend is unavailable" };
        }
    };
    static UnavailableBackend backend;
    return backend;
}
} // namespace videohelper::volume
