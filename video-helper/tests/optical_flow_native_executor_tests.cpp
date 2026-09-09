#include "../src/optical_flow_native_executor.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace
{
int checks = 0;
int failures = 0;

void check (bool condition, const char* message)
{
    ++checks;
    if (! condition)
    {
        ++failures;
        std::fprintf (stderr, "FAIL: %s\n", message);
    }
}

videoopticalflow::BackendCapabilities nativeCapabilities()
{
    videoopticalflow::BackendCapabilities result;
    result.kind = videoopticalflow::BackendKind::NativeGpu;
    result.supportsOpticalFlow = true;
    result.implementation[0] = 0x47;
    result.implementation[1] = 0x50;
    result.implementationRevision = 3;
    result.limits.width = 4096;
    result.limits.height = 4096;
    result.limits.pixels = 4096ull * 4096ull;
    result.limits.residentBytes = 512ull * 1024ull * 1024ull;
    result.temporaryBytesPerPixel = 4;
    result.fixedTemporaryBytes = 256;
    return result;
}

videoopticalflow::Description pairDescription (std::uint64_t firstIndex)
{
    videoopticalflow::Description result;
    result.first.identity.stream[0] = 0x51;
    result.second.identity.stream = result.first.identity.stream;
    result.first.identity.content[0] = static_cast<std::uint8_t> (firstIndex + 1);
    result.second.identity.content[0] = static_cast<std::uint8_t> (firstIndex + 2);
    result.first.identity.frameIndex = firstIndex;
    result.second.identity.frameIndex = firstIndex + 1;
    result.first.timestamp = { static_cast<std::int64_t> (firstIndex * 1001), 24000 };
    result.second.timestamp = {
        static_cast<std::int64_t> ((firstIndex + 1) * 1001), 24000 };
    result.first.extent = { 8, 4 };
    result.second.extent = result.first.extent;
    result.output = videoopticalflow::canonicalOutput (result.first.extent);
    return result;
}

videoopticalflow::AdmittedRequest admittedPair (
    std::uint64_t firstIndex,
    const videoopticalflow::BackendCapabilities& capabilities)
{
    videoopticalflow::AdmissionFailure failure =
        videoopticalflow::AdmissionFailure::None;
    auto result = videoopticalflow::admit (
        pairDescription (firstIndex), capabilities, failure);
    if (! result)
    {
        std::fprintf (stderr, "test setup admission failed: %s\n",
                      std::string (videoopticalflow::token (failure)).c_str());
        std::exit (2);
    }
    return std::move (*result);
}

class FakeSourceFrame final : public videoopticalflow::NativeSourceFrame
{
public:
    const videoopticalflow::FrameDescription& frame() const noexcept override
    {
        return frameDescription;
    }
    const videoopticalflow::ResourceIdentity& resourceIdentity() const noexcept override
    {
        return identity;
    }
    std::uint64_t helperGeneration() const noexcept override { return generation; }
    std::uint64_t structuralRevision() const noexcept override { return revision; }
    const videoopticalflow::BackendIdentity& backendImplementation() const noexcept override
    {
        return implementation;
    }
    std::uint32_t backendImplementationRevision() const noexcept override
    {
        return implementationRevision;
    }
    std::uint64_t byteCount() const noexcept override { return bytes; }
    std::uintptr_t imageHandle() const noexcept override { return image; }
    std::uintptr_t textureViewHandle() const noexcept override { return textureView; }

    videoopticalflow::FrameDescription frameDescription;
    videoopticalflow::ResourceIdentity identity {};
    std::uint64_t generation = 0;
    std::uint64_t revision = 0;
    videoopticalflow::BackendIdentity implementation {};
    std::uint32_t implementationRevision = 0;
    std::uint64_t bytes = 0;
    std::uintptr_t image = 0;
    std::uintptr_t textureView = 0;
};

std::shared_ptr<FakeSourceFrame> sourceFrame (
    const videoopticalflow::FrameDescription& frame,
    std::uint8_t identityByte,
    std::uintptr_t image,
    std::uint64_t revision,
    std::uint64_t generation,
    const videoopticalflow::BackendCapabilities& capabilities)
{
    auto result = std::make_shared<FakeSourceFrame>();
    result->frameDescription = frame;
    result->identity[0] = identityByte;
    result->generation = generation;
    result->revision = revision;
    result->implementation = capabilities.implementation;
    result->implementationRevision = capabilities.implementationRevision;
    result->bytes = static_cast<std::uint64_t> (frame.extent.width)
                  * static_cast<std::uint64_t> (frame.extent.height)
                  * renderpassoutput::bytesPerPixel (frame.format);
    result->image = image;
    result->textureView = image + 1000;
    return result;
}

class FakeNativeBackend final : public arbitgpu::NativeOpticalFlowExecutionBackend
{
public:
    videoopticalflow::BackendCapabilities opticalFlowCapabilities() const override
    {
        return reportedCapabilities;
    }

    arbitgpu::NativeOpticalFlowSubmission executeOpticalFlow (
        const videoopticalflow::AdmittedRequest& request,
        const arbitgpu::NativeOpticalFlowInputResource& first,
        const arbitgpu::NativeOpticalFlowInputResource& second,
        bool resetTemporalState) override
    {
        ++submissions;
        firstImages.push_back (first.imageHandle);
        secondImages.push_back (second.imageHandle);
        resetFlags.push_back (resetTemporalState);
        if (rejectSubmission)
        {
            arbitgpu::NativeOpticalFlowSubmission rejected;
            rejected.error = "test native GPU queue rejected submission";
            return rejected;
        }

        arbitgpu::NativeOpticalFlowSubmission submission;
        submission.completed = true;
        submission.submission = zeroSubmissionSerial ? 0 : nextSerial++;
        submission.lifecycle.value = nextLifecycle++;
        submission.result.requestCacheKey = request.cacheKey();
        submission.result.backendImplementation = request.backendImplementation();
        submission.result.backendImplementationRevision =
            request.backendImplementationRevision();
        submission.result.motionVectors.identity[0] =
            static_cast<std::uint8_t> (0x80 + submissions);
        submission.result.motionVectors.helperGeneration = first.helperGeneration;
        submission.result.motionVectors.descriptor = request.output();
        submission.result.motionVectors.byteCount = request.footprint().outputBytes;
        if (malformedResult)
            --submission.result.motionVectors.byteCount;
        submission.imageHandle = nextImage++;
        submission.textureViewHandle = submission.imageHandle + 2000;
        return submission;
    }

    void releaseOpticalFlowOutput (
        arbitgpu::NativeOpticalFlowOutputLifecycleHandle lifecycle) noexcept override
    {
        if (lifecycle && releasedLifecycles.insert (lifecycle.value).second)
            ++releases;
    }

    videoopticalflow::BackendCapabilities reportedCapabilities = nativeCapabilities();
    bool rejectSubmission = false;
    bool malformedResult = false;
    bool zeroSubmissionSerial = false;
    int submissions = 0;
    int releases = 0;
    std::uint64_t nextSerial = 41;
    std::uint64_t nextLifecycle = 1;
    std::uintptr_t nextImage = 9001;
    std::vector<std::uintptr_t> firstImages;
    std::vector<std::uintptr_t> secondImages;
    std::vector<bool> resetFlags;
    std::set<std::uint64_t> releasedLifecycles;
};

videoopticalflow::LoweredOperation operationFor (
    renderpassoutput::Extent extent,
    std::string& storage)
{
    opticalflowoperation::Payload payload;
    payload.extent = extent;
    storage = opticalflowoperation::serialize (payload);
    return { opticalflowoperation::kOperationKind,
             opticalflowoperation::kBackendCapability,
             storage };
}
} // namespace

int main()
{
    using namespace videoopticalflow;

    BackendCapabilities defaultLimits;
    defaultLimits.kind = BackendKind::NativeGpu;
    defaultLimits.supportsOpticalFlow = true;
    defaultLimits.implementation[0] = 1;
    defaultLimits.implementationRevision = 1;
    AdmissionFailure defaultFailure = AdmissionFailure::None;
    check (admit (pairDescription (1), defaultLimits, defaultFailure).has_value(),
           "default backend limits admit the bounded RGBA16F fixture");

    FakeNativeBackend backend;
    NativeExecutor preview (EvaluationMode::Preview, backend);
    NativeExecutor exportRun (EvaluationMode::OfflineSequential, backend);
    auto firstRequest = admittedPair (10, backend.reportedCapabilities);
    constexpr std::uint64_t structuralRevision = 17;
    constexpr std::uint64_t helperGeneration = 5;
    auto firstSource = sourceFrame (
        firstRequest.first(), 0x11, 101, structuralRevision, helperGeneration,
        backend.reportedCapabilities);
    auto secondSource = sourceFrame (
        firstRequest.second(), 0x12, 202, structuralRevision, helperGeneration,
        backend.reportedCapabilities);
    EvaluationContext previewContext;
    previewContext.structuralRevision = structuralRevision;
    previewContext.helperGeneration = helperGeneration;
    previewContext.mode = EvaluationMode::Preview;
    std::string operationStorage;
    auto operation = operationFor (firstRequest.output().extent, operationStorage);
    std::string error;
    NativeExecutionFailure failure = NativeExecutionFailure::SubmissionRejected;

    check (preview.execute (operation, firstRequest, firstSource, secondSource,
                            previewContext, error, &failure),
           "exact lowered operation submits two admitted immutable native resources");
    auto firstPublication = preview.publication();
    check (failure == NativeExecutionFailure::None && error.empty()
           && backend.submissions == 1
           && backend.resetFlags == std::vector<bool> { true }
           && backend.firstImages == std::vector<std::uintptr_t> { 101 }
           && backend.secondImages == std::vector<std::uintptr_t> { 202 },
           "dispatch preserves exact source order and first-evaluation reset");
    check (firstPublication != nullptr
           && firstPublication->receipt().requestCacheKey() == firstRequest.cacheKey()
           && firstPublication->receipt().motionVectors().descriptor.extent
                  == firstRequest.output().extent
           && firstPublication->receipt().motionVectors().descriptor.format
                  == firstRequest.output().format
           && firstPublication->receipt().motionVectors().descriptor.direction
                  == firstRequest.output().direction
           && firstPublication->structuralRevision() == structuralRevision
           && firstPublication->helperGeneration() == helperGeneration,
           "publication preserves exact request, resource descriptor, and revisions");

    EvaluationContext offlineContext = previewContext;
    offlineContext.mode = EvaluationMode::OfflineSequential;
    check (! preview.execute (operation, firstRequest, firstSource, secondSource,
                              offlineContext, error, &failure)
           && failure == NativeExecutionFailure::OwnerModeMismatch
           && backend.submissions == 1 && preview.publication() == firstPublication,
           "a viewport owner cannot be reused as an export owner");
    check (exportRun.execute (operation, firstRequest, firstSource, secondSource,
                              offlineContext, error, &failure)
           && backend.submissions == 2
           && exportRun.publication() != nullptr
           && exportRun.publication()->imageHandle()
                  != firstPublication->imageHandle()
           && exportRun.publication()->receipt().motionVectors().identity
                  != firstPublication->receipt().motionVectors().identity,
           "viewport and export owners receive isolated native output resources");
    exportRun.reset();
    check (backend.releases == 1,
           "reset releases the export owner's resource without touching viewport output");

    check (preview.execute (operation, firstRequest, firstSource, secondSource,
                            previewContext, error)
           && backend.submissions == 2
           && preview.publication() == firstPublication,
           "identical evaluation holds last-good output without resubmission");

    auto nextRequest = admittedPair (11, backend.reportedCapabilities);
    auto nextFirst = sourceFrame (
        nextRequest.first(), 0x12, 202, structuralRevision, helperGeneration,
        backend.reportedCapabilities);
    auto nextSecond = sourceFrame (
        nextRequest.second(), 0x13, 303, structuralRevision, helperGeneration,
        backend.reportedCapabilities);
    backend.rejectSubmission = true;
    const auto rejected = preview.execute (
        operation, nextRequest, nextFirst, nextSecond,
        previewContext, error, &failure);
    check (! rejected
           && failure == NativeExecutionFailure::SubmissionRejected
           && error == "test native GPU queue rejected submission"
           && backend.submissions == 3 && preview.publication() == firstPublication,
           "ordinary GPU rejection preserves exact last-good publication and state");
    backend.rejectSubmission = false;

    backend.malformedResult = true;
    check (! preview.execute (operation, nextRequest, nextFirst, nextSecond,
                              previewContext, error, &failure)
           && failure == NativeExecutionFailure::InvalidSubmittedResource
           && backend.submissions == 4 && backend.releases == 2
           && preview.publication() == firstPublication,
           "malformed backend result is released and cannot replace last-good state");
    backend.malformedResult = false;

    check (preview.execute (operation, nextRequest, nextFirst, nextSecond,
                            previewContext, error, &failure)
           && backend.submissions == 5 && ! backend.resetFlags.back()
           && preview.publication() != firstPublication,
           "next adjacent pair publishes only after successful native submission");
    auto secondPublication = preview.publication();
    check (backend.releases == 2,
           "an in-flight consumer retains an older GPU resource after replacement");

    auto malformedOperation = operation;
    malformedOperation.backendCapability = "native-gpu ";
    check (! preview.execute (malformedOperation, nextRequest, nextFirst, nextSecond,
                              previewContext, error, &failure)
           && failure == NativeExecutionFailure::MalformedLoweredOperation
           && backend.submissions == 5 && preview.publication() == secondPublication,
           "inexact backend capability fails before GPU submission");
    std::string nonCanonical =
        "<NodeParams width=\"08\" height=\"4\" color=\"0\" motion=\"1\"/>";
    malformedOperation = operation;
    malformedOperation.payloadXml = nonCanonical;
    check (! preview.execute (malformedOperation, nextRequest, nextFirst, nextSecond,
                              previewContext, error, &failure)
           && failure == NativeExecutionFailure::MalformedLoweredOperation
           && backend.submissions == 5,
           "noncanonical lowered payload cannot acquire execution authority");

    std::string otherExtentStorage;
    auto wrongExtent = operationFor ({ 9, 4 }, otherExtentStorage);
    check (! preview.execute (wrongExtent, nextRequest, nextFirst, nextSecond,
                              previewContext, error, &failure)
           && failure == NativeExecutionFailure::LoweredExtentMismatch
           && backend.submissions == 5,
           "lowered and admitted extents must match exactly");

    auto missingHandle = sourceFrame (
        nextRequest.first(), 0x21, 404, structuralRevision, helperGeneration,
        backend.reportedCapabilities);
    missingHandle->image = 0;
    check (! preview.execute (operation, nextRequest, missingHandle, nextSecond,
                              previewContext, error, &failure)
           && failure == NativeExecutionFailure::InvalidSourceResource
           && backend.submissions == 5,
           "incomplete native source resource is rejected before submission");

    auto wrongFormat = sourceFrame (
        nextRequest.first(), 0x22, 505, structuralRevision, helperGeneration,
        backend.reportedCapabilities);
    wrongFormat->frameDescription.format = renderpassoutput::PixelFormat::R32Float;
    check (! preview.execute (operation, nextRequest, wrongFormat, nextSecond,
                              previewContext, error, &failure)
           && failure == NativeExecutionFailure::SourceResourceMismatch
           && backend.submissions == 5,
           "source descriptor mismatch is rejected before submission");

    auto staleGeneration = sourceFrame (
        nextRequest.first(), 0x23, 606, structuralRevision, helperGeneration - 1,
        backend.reportedCapabilities);
    check (! preview.execute (operation, nextRequest, staleGeneration, nextSecond,
                              previewContext, error, &failure)
           && failure == NativeExecutionFailure::SourceResourceMismatch
           && backend.submissions == 5,
           "stale helper-generation resource is rejected before submission");

    auto duplicateSecond = sourceFrame (
        nextRequest.second(), 0x12, 202, structuralRevision, helperGeneration,
        backend.reportedCapabilities);
    check (! preview.execute (operation, nextRequest, nextFirst, duplicateSecond,
                              previewContext, error, &failure)
           && failure == NativeExecutionFailure::DuplicateSourceResource
           && backend.submissions == 5,
           "two frame identities cannot alias one native input resource");

    auto changedCapabilities = backend.reportedCapabilities;
    changedCapabilities.kind = BackendKind::CpuReference;
    changedCapabilities.supportsOpticalFlow = false;
    backend.reportedCapabilities = changedCapabilities;
    check (! preview.execute (operation, nextRequest, nextFirst, nextSecond,
                              previewContext, error, &failure)
           && failure == NativeExecutionFailure::BackendCapabilityMismatch
           && backend.submissions == 5,
           "CPU or unavailable backend fails closed without production fallback");
    backend.reportedCapabilities = nativeCapabilities();

    backend.rejectSubmission = true;
    EvaluationContext revisedContext = previewContext;
    ++revisedContext.structuralRevision;
    auto revisedFirst = sourceFrame (
        nextRequest.first(), 0x31, 707, revisedContext.structuralRevision,
        helperGeneration, backend.reportedCapabilities);
    auto revisedSecond = sourceFrame (
        nextRequest.second(), 0x32, 808, revisedContext.structuralRevision,
        helperGeneration, backend.reportedCapabilities);
    check (! preview.execute (operation, nextRequest, revisedFirst, revisedSecond,
                              revisedContext, error, &failure)
           && failure == NativeExecutionFailure::SubmissionRejected
           && preview.publication() == nullptr && backend.submissions == 6,
           "revision reset drops stale owner state before a failed dispatch");
    backend.rejectSubmission = false;

    const auto releasesBeforeFinalReset = backend.releases;
    firstPublication.reset();
    secondPublication.reset();
    preview.reset();
    check (backend.releases > releasesBeforeFinalReset,
           "all retained native outputs release at final shared-owner destruction");

    std::printf ("optical-flow-native-executor %s checks=%d failures=%d\n",
                 failures == 0 ? "PASS" : "FAIL", checks, failures);
    return failures == 0 ? 0 : 1;
}
