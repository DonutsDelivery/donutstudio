#pragma once

#include "volume_native_renderer.h"
#include "../../shared/VisualVolumeOperationContract.h"
#include <list>
#include <vector>

namespace videohelper::volume
{
// Owned by one FrameRenderer. shutdown() clears this while that renderer's
// native context is still alive. No frame or GLFW context enters static storage.
class VisualVolumeRenderCache final
{
public:
    static constexpr std::size_t maximumEntries = 8;
    static constexpr std::uint64_t maximumBytes = 128u * 1024u * 1024u;

    ~VisualVolumeRenderCache() { clear(); }
    VisualVolumeRenderCache() = default;
    VisualVolumeRenderCache(const VisualVolumeRenderCache&) = delete;
    VisualVolumeRenderCache& operator=(const VisualVolumeRenderCache&) = delete;

    void clear() noexcept
    {
        // Retire even a frame borrowed by a LayerDesc after its LRU entry was
        // evicted. This runs while the owning context is valid; late C++ leases
        // see empty resources and cannot switch back to a destroyed GL window.
        for (const auto& retained : submittedFrames_)
            if (const auto frame = retained.lock()) frame->releaseNativeResources();
        submittedFrames_.clear();
        entries_.clear();
        if (contextLifetime_) contextLifetime_->alive.store(false, std::memory_order_release);
        contextLifetime_.reset();
        bytes_ = 0;
        backend_ = nullptr;
        contextIdentity_ = 0;
    }
    std::size_t size() const noexcept { return entries_.size(); }
    std::uint64_t retainedBytes() const noexcept { return bytes_; }

    bool render(const std::string& payload, std::uint32_t width, std::uint32_t height,
                NativeVolumeRenderUse use, NativeVolumeExecutionBackend& backend,
                NativeVolumeRenderedFrame& output, std::string& error)
    {
        visualvolume::Operation operation;
        if (!visualvolume::decode(payload, operation))
        { error = "Volume cache requires the exact canonical density payload"; return false; }
        const auto capabilities = backend.capabilities();
        const auto contextIdentity = backend.contextIdentity();
        if (contextIdentity == 0 || !capabilities.volume.nativeGpuAvailable)
        { error = "Volume cache requires a live native render context"; return false; }
        if (backend_ != nullptr && (backend_ != &backend || contextIdentity_ != contextIdentity))
        { error = "Volume cache cannot change native backend within a render context"; return false; }
        for (auto entry = entries_.begin(); entry != entries_.end(); ++entry)
            if (entry->payload == payload && entry->frame.width == width && entry->frame.height == height)
            {
                output = entry->frame;
                output.use = use;
                entries_.splice(entries_.begin(), entries_, entry);
                error.clear();
                return true;
            }
        if (width == 0 || height == 0 || width > capabilities.maxRenderExtent
            || height > capabilities.maxRenderExtent
            || static_cast<std::uint64_t>(width) * height > capabilities.maxRenderPixels)
        { error = "Volume render extent exceeds the native context limits"; return false; }
        const auto voxels = static_cast<std::uint64_t>(operation.resolution)
            * operation.resolution * operation.resolution;
        const auto bytes = static_cast<std::uint64_t>(width) * height * 4u + voxels * 2u;
        if (bytes > maximumBytes)
        { error = "Volume render extent exceeds the context cache budget"; return false; }
        // Evict before allocating so cache ownership stays bounded even on an
        // edit or canvas resize. In-flight LayerDesc owners survive composition.
        while (!entries_.empty() && (entries_.size() >= maximumEntries || bytes_ > maximumBytes - bytes))
        {
            bytes_ -= entries_.back().bytes;
            entries_.pop_back();
        }
        VolumeAdmissionLimits limits;
        limits.maxDimension = visualvolume::maximumResolution;
        limits.maxVoxelCount = visualvolume::maximumVoxels;
        limits.maxPayloadBytes = visualvolume::maximumVoxels;
        videowire::DenseVolumeDescriptor dense;
        if (!visualvolume::makeDensity(operation, dense))
        { error = "Volume density generation rejected the authored controls"; return false; }
        auto admitted = admitDenseVolume(dense, limits, capabilities.volume, error);
        if (!admitted) return false;
        auto volume = std::make_shared<const AdmittedVolume>(std::move(*admitted));
        if (!contextLifetime_) contextLifetime_ = std::make_shared<NativeVolumeContextLifetime>();
        NativeVolumeRenderer renderer(backend);
        NativeVolumeRenderedFrame frame;
        const bool rendered = use == NativeVolumeRenderUse::Export
            ? renderer.renderExport(volume, width, height, kNativeVolumeGpuCapability, frame, error, true, contextLifetime_)
            : renderer.renderPreview(volume, width, height, kNativeVolumeGpuCapability, frame, error, true, contextLifetime_);
        if (!rendered) return false;
        submittedFrames_.erase(std::remove_if(submittedFrames_.begin(), submittedFrames_.end(),
            [](const auto& frame) { return frame.expired(); }), submittedFrames_.end());
        submittedFrames_.push_back(frame.nativeFrame);
        backend_ = &backend;
        contextIdentity_ = contextIdentity;
        entries_.push_front({payload, std::move(volume), frame, bytes});
        bytes_ += bytes;
        output = std::move(frame);
        return true;
    }

private:
    struct Entry final
    {
        std::string payload;
        std::shared_ptr<const AdmittedVolume> volume;
        NativeVolumeRenderedFrame frame;
        std::uint64_t bytes = 0;
    };
    std::vector<std::weak_ptr<const NativeVolumeFrame>> submittedFrames_;
    std::shared_ptr<NativeVolumeContextLifetime> contextLifetime_;
    NativeVolumeExecutionBackend* backend_ = nullptr;
    std::uintptr_t contextIdentity_ = 0;
    std::list<Entry> entries_;
    std::uint64_t bytes_ = 0;
};
} // namespace videohelper::volume
