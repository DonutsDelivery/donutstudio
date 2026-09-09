#pragma once

#include "../../shared/OpticalFlowOperationContract.h"
#include "gpu_backend/backend.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace videoopticalflow
{
// Immutable backend-owned input image. The frame identity and resource receipt
// are captured together so an executor cannot retarget an admitted request to a
// mutable texture name or to an image from another helper/backend generation.
class NativeSourceFrame
{
public:
    virtual ~NativeSourceFrame() = default;
    virtual const FrameDescription& frame() const noexcept = 0;
    virtual const ResourceIdentity& resourceIdentity() const noexcept = 0;
    virtual std::uint64_t helperGeneration() const noexcept = 0;
    virtual std::uint64_t structuralRevision() const noexcept = 0;
    virtual const BackendIdentity& backendImplementation() const noexcept = 0;
    virtual std::uint32_t backendImplementationRevision() const noexcept = 0;
    virtual std::uint64_t byteCount() const noexcept = 0;
    virtual std::uintptr_t imageHandle() const noexcept = 0;
    virtual std::uintptr_t textureViewHandle() const noexcept = 0;
};

struct LoweredOperation final
{
    std::string_view kind;
    std::string_view backendCapability;
    std::string_view payloadXml;
};

enum class NativeExecutionFailure : std::uint8_t
{
    None = 0,
    MalformedLoweredOperation,
    LoweredExtentMismatch,
    BackendCapabilityMismatch,
    InvalidSourceResource,
    SourceResourceMismatch,
    DuplicateSourceResource,
    OwnerModeMismatch,
    EvaluationRejected,
    MissingHeldPublication,
    SubmissionRejected,
    InvalidSubmittedResource
};

std::string_view token (NativeExecutionFailure failure) noexcept;

// Immutable publication of one admitted backend-owned motion-vector image.
// Destruction of the final shared owner releases the matching backend lifecycle;
// callers may retain a publication across a later successful replacement.
class PublishedNativeResult final
{
public:
    PublishedNativeResult (const PublishedNativeResult&) = delete;
    PublishedNativeResult& operator= (const PublishedNativeResult&) = delete;
    ~PublishedNativeResult();

    const AdmittedResult& receipt() const noexcept { return receipt_; }
    std::uint64_t submissionSerial() const noexcept { return submissionSerial_; }
    std::uintptr_t imageHandle() const noexcept { return imageHandle_; }
    std::uintptr_t textureViewHandle() const noexcept { return textureViewHandle_; }
    std::uint64_t structuralRevision() const noexcept { return structuralRevision_; }
    std::uint64_t helperGeneration() const noexcept { return helperGeneration_; }

private:
    friend class NativeExecutor;

    PublishedNativeResult (
        AdmittedResult receipt,
        std::uint64_t submissionSerial,
        std::uintptr_t imageHandle,
        std::uintptr_t textureViewHandle,
        std::uint64_t structuralRevision,
        std::uint64_t helperGeneration,
        arbitgpu::NativeOpticalFlowOutputLifecycleHandle lifecycle,
        arbitgpu::NativeOpticalFlowExecutionBackend& backend) noexcept;

    const AdmittedResult receipt_;
    const std::uint64_t submissionSerial_ = 0;
    const std::uintptr_t imageHandle_ = 0;
    const std::uintptr_t textureViewHandle_ = 0;
    const std::uint64_t structuralRevision_ = 0;
    const std::uint64_t helperGeneration_ = 0;
    const arbitgpu::NativeOpticalFlowOutputLifecycleHandle lifecycle_;
    arbitgpu::NativeOpticalFlowExecutionBackend* const backend_ = nullptr;
};

// Each viewport and each offline export run owns a different executor with a
// fixed evaluation mode. A successful dispatch replaces the publication only
// after native submission and exact result admission. Ordinary rejection keeps
// last-good; a defined seek, revision, loop, or helper-generation reset drops
// stale state before dispatch.
class NativeExecutor final
{
public:
    explicit NativeExecutor (
        EvaluationMode ownerMode,
        arbitgpu::NativeOpticalFlowExecutionBackend& backend
            = arbitgpu::nativeOpticalFlowExecutionBackend()) noexcept;
    NativeExecutor (const NativeExecutor&) = delete;
    NativeExecutor& operator= (const NativeExecutor&) = delete;

    // Synchronous checkpoint seam: completion and result admission happen in
    // this call. Product routes must invoke it on an owned helper worker, never
    // a UI, presentation, or render-thread callback.
    bool execute (
        const LoweredOperation& operation,
        const AdmittedRequest& request,
        const std::shared_ptr<const NativeSourceFrame>& first,
        const std::shared_ptr<const NativeSourceFrame>& second,
        const EvaluationContext& context,
        std::string& error,
        NativeExecutionFailure* failure = nullptr);

    const std::shared_ptr<const PublishedNativeResult>& publication() const noexcept
    {
        return publication_;
    }

    void reset() noexcept;

private:
    bool fail (NativeExecutionFailure reason, std::string message,
               std::string& error, NativeExecutionFailure* failure) const;

    arbitgpu::NativeOpticalFlowExecutionBackend& backend_;
    const EvaluationMode ownerMode_;
    EvaluationState evaluationState_;
    std::shared_ptr<const PublishedNativeResult> publication_;
};
} // namespace videoopticalflow
