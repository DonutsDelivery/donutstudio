#pragma once

#include "optical_flow_source_lease.h"

#include <atomic>
#include <memory>

namespace videoopticalflow
{
class UploadedSourceResource
{
public:
    virtual ~UploadedSourceResource() = default;
    virtual std::uintptr_t imageHandle() const noexcept = 0;
    virtual std::uintptr_t nativeContextIdentity() const noexcept = 0;
};

class UploadedSourceFrame final : public NativeSourceFrame
{
public:
    UploadedSourceFrame (SourceFrameV2 source, ContentIdentity linearPixels,
                         BackendCapabilities backend, std::uint64_t helperGeneration,
                         std::uint64_t structuralRevision,
                         std::shared_ptr<const UploadedSourceResource> resource)
        : source_ (std::move (source)), backend_ (backend),
          helperGeneration_ (helperGeneration), structuralRevision_ (structuralRevision),
          resource_ (std::move (resource))
    {
        source_.frame.identity.content = linearPixels;
        static std::atomic<std::uint64_t> nextResource { 1 };
        const auto serial = nextResource.fetch_add (1, std::memory_order_relaxed);
        for (std::size_t index = 0; index < sizeof (serial); ++index)
            resourceIdentity_[index] = static_cast<std::uint8_t> (serial >> (index * 8));
        std::copy_n (linearPixels.begin(), resourceIdentity_.size() - sizeof (serial),
                     resourceIdentity_.begin() + sizeof (serial));
    }

    const FrameDescription& frame() const noexcept override { return source_.frame; }
    const ResourceIdentity& resourceIdentity() const noexcept override { return resourceIdentity_; }
    std::uint64_t helperGeneration() const noexcept override { return helperGeneration_; }
    std::uint64_t structuralRevision() const noexcept override { return structuralRevision_; }
    const BackendIdentity& backendImplementation() const noexcept override
    { return backend_.implementation; }
    std::uint32_t backendImplementationRevision() const noexcept override
    { return backend_.implementationRevision; }
    std::uint64_t byteCount() const noexcept override
    { return static_cast<std::uint64_t> (source_.frame.extent.width)
             * source_.frame.extent.height * 8u; }
    std::uintptr_t imageHandle() const noexcept override
    { return resource_ ? resource_->imageHandle() : 0; }
    std::uintptr_t textureViewHandle() const noexcept override { return imageHandle(); }
    std::uintptr_t nativeContextIdentity() const noexcept override
    { return resource_ ? resource_->nativeContextIdentity() : 0; }
    std::uint32_t sourceClipId() const noexcept override { return source_.clipId; }
    const ContentIdentity& sourceContent() const noexcept override
    { return source_.sourceContent; }
    std::uint64_t sourceGeneration() const noexcept override
    { return source_.sourceGeneration; }

private:
    SourceFrameV2 source_;
    BackendCapabilities backend_;
    std::uint64_t helperGeneration_ = 0;
    std::uint64_t structuralRevision_ = 0;
    std::shared_ptr<const UploadedSourceResource> resource_;
    ResourceIdentity resourceIdentity_ {};
};

inline bool validateUploadSource (
    const DecodedSourceReceipt& decoded,
    const colortransform::AdmittedTransform& transform,
    const SourceFrameV2& expected, std::string& error,
    bool pixelsValidatedOnWorker = false)
{
    SourceFrameV2 actual;
    colortransform::Description description;
    if (! detail::expectedSource (decoded, actual, description, error,
                                 ! pixelsValidatedOnWorker)) return false;
    colortransform::AdmissionFailure failure {};
    const auto admitted = colortransform::admit (
        description, colortransform::BackendCapability::NativeGpu, failure);
    if (! admitted || admitted->cacheIdentity() != transform.cacheIdentity()
        || actual.clipId != expected.clipId
        || actual.sourceContent != expected.sourceContent
        || actual.sourceGeneration != expected.sourceGeneration
        || actual.frame.identity.stream != expected.frame.identity.stream
        || actual.frame.identity.frameIndex != expected.frame.identity.frameIndex
        || actual.frame.timestamp.ticks != expected.frame.timestamp.ticks
        || actual.frame.timestamp.timescale != expected.frame.timestamp.timescale
        || actual.frame.extent != expected.frame.extent
        || actual.frame.format != expected.frame.format
        || actual.frame.colorSpace != expected.frame.colorSpace
        || detail::anyNonZero (expected.frame.identity.content))
    {
        error = "Optical Flow source upload has a stale frame or color owner";
        return false;
    }
    return true;
}
} // namespace videoopticalflow
