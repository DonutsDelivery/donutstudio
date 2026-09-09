#include "../src/sdf_native_renderer.h"
#include "../src/sdf_native_program.h"
#include "../src/sdf_visual_plan_execution.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

static_assert (! std::is_default_constructible_v<arbitgpu::NativeSdfCompiledProgram>,
               "compiled SDF programs must remain factory-only");

namespace
{
int failures = 0;
int checks = 0;

void check (bool condition, const char* message)
{
    ++checks;
    if (! condition)
    {
        ++failures;
        std::fprintf (stderr, "FAIL: %s\n", message);
    }
}

std::shared_ptr<const videohelper::sdf::AdmittedSdfIr> sphereGeometry (double radius = 1.0)
{
    videowire::SdfIr source;
    source.rootId = 71;
    videowire::SdfRecord sphere;
    sphere.stableId = source.rootId;
    sphere.operation = videowire::SdfOperation::sphere;
    sphere.parameterCount = 1;
    sphere.parameters[0] = radius;
    source.records.push_back (sphere);

    std::string error;
    auto admitted = videohelper::sdf::admitSdfIr (source, {}, error);
    if (! admitted)
    {
        std::fprintf (stderr, "test setup failed: %s\n", error.c_str());
        std::exit (2);
    }
    return std::make_shared<const videohelper::sdf::AdmittedSdfIr> (
        std::move (*admitted));
}

arbitgpu::NativeSdfExecutionCapabilities sphereCapabilities()
{
    arbitgpu::NativeSdfExecutionCapabilities result;
    result.available = true;
    result.backend = "test-native";
    result.device = "fake GPU";
    result.supportedOperations[static_cast<std::size_t> (
        videowire::SdfOperation::sphere)] = true;
    result.supportedOutputs[static_cast<std::size_t> (
        arbitgpu::NativeSdfOutput::color)] = true;
    result.maxOperations = 1;
    result.maxDepth = 1;
    result.maxExtent = 1024;
    result.maxPixels = 1024ull * 1024ull;
    result.maxSteps = 512;
    result.minEpsilon = 0.000001;
    result.maxEpsilon = 0.1;
    result.maxDistance = 1000.0;
    return result;
}

// This double carries opaque handles only. These checks cover dispatch and
// ownership, not GPU pixels. The Metal backend test covers native execution.
class FakeFrame final : public arbitgpu::NativeSdfSceneFrame
{
public:
    FakeFrame (std::uint32_t width, std::uint32_t height,
               std::uintptr_t handle, int& deletions) noexcept
        : width_ (width), height_ (height), handle_ (handle), deletions_ (deletions)
    {
    }

    ~FakeFrame() override { ++deletions_; }
    const std::string& backend() const noexcept override { return backend_; }
    std::uint32_t width() const noexcept override { return width_; }
    std::uint32_t height() const noexcept override { return height_; }
    std::uintptr_t colorImageHandle() const noexcept override { return handle_; }
    std::uintptr_t colorTextureViewHandle() const noexcept override { return handle_ + 1000; }
    const arbitgpu::FrameMemoryAdmission& frameMemoryAdmission() const noexcept override
    {
        return frameMemory_;
    }
    const arbitgpu::NativeSdfResourceReceipt& sdfResourceReceipt() const noexcept override
    {
        return receipt_;
    }

private:
    arbitgpu::FrameMemoryAdmission frameMemory_ {};
    arbitgpu::NativeSdfResourceReceipt receipt_ {};
    std::string backend_ = "test-native";
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    std::uintptr_t handle_ = 0;
    int& deletions_;
};

class FakeBackend final : public arbitgpu::NativeSdfExecutionBackend
{
public:
    arbitgpu::NativeSdfExecutionCapabilities capabilities() const override
    {
        return reportedCapabilities;
    }

    arbitgpu::NativeSdfSceneSubmission render (
        const arbitgpu::NativeSdfDrawRequest& request) override
    {
        ++submissions;
        lastRequest = request;
        compiledPrograms.push_back (request.compiledProgram);
        if (submissions == 1)
            firstCompiledProgram = request.compiledProgram;
        arbitgpu::NativeSdfSceneSubmission result;
        if (rejectSubmission)
        {
            result.error = "fake native SDF draw rejected";
            return result;
        }
        result.rendered = true;
        result.frame = std::make_shared<FakeFrame> (
            request.width, request.height,
            static_cast<std::uintptr_t> (submissions), deletions);
        return result;
    }

    arbitgpu::NativeSdfExecutionCapabilities reportedCapabilities = sphereCapabilities();
    arbitgpu::NativeSdfDrawRequest lastRequest;
    std::shared_ptr<const arbitgpu::NativeSdfCompiledProgram> firstCompiledProgram;
    std::vector<std::shared_ptr<const arbitgpu::NativeSdfCompiledProgram>> compiledPrograms;
    bool rejectSubmission = false;
    int submissions = 0;
    int deletions = 0;
};

struct TestLayer
{
    unsigned texture = 0;
    int texWidth = 0;
    int texHeight = 0;
    std::string nativeTextureBackend;
    std::uintptr_t nativeTextureView = 0;
};
} // namespace

int main()
{
    using namespace videohelper::sdf;

    const auto geometry = sphereGeometry();
    const auto equivalentGeometry = sphereGeometry();
    videowire::SdfIr authoritativeSource;
    authoritativeSource.schemaVersion = geometry->schemaVersion();
    authoritativeSource.rootId = geometry->rootId();
    authoritativeSource.records = geometry->records();
    std::string validationError;
    const auto compiled = compileNativeSdfProgram (authoritativeSource, validationError);
    auto mismatchedSource = authoritativeSource;
    mismatchedSource.records[0].parameters[0] = 2.0;
    check (compiled != nullptr
           && ! validateNativeSdfProgram (*compiled, mismatchedSource, validationError),
           "factory-only compiled payloads reject a different valid source payload exactly");

    FakeBackend backend;
    NativeSdfRenderer renderer (backend);
    NativeSdfRenderedFrame preview;
    NativeSdfRenderedFrame exported;
    NativeSdfRenderControls controls;
    controls.maximumSteps = 192;
    controls.epsilon = 0.0005;
    controls.maximumDistance = 250.0;
    controls.adaptiveQuality = arbitgpu::NativeSdfQuality::high;
    controls.normalQuality = arbitgpu::NativeSdfQuality::ultra;
    controls.shadowQuality = arbitgpu::NativeSdfQuality::low;
    std::string error;

    check (renderer.renderPreview (geometry, { 320, 180 }, controls,
                                   kNativeGpuCapability, preview, error),
           "preview submits admitted SDF geometry to the native backend");
    check (renderer.renderExport (equivalentGeometry, { 320, 180 }, controls,
                                  kNativeGpuCapability, exported, error),
           "export submits through the same native SDF renderer");
    check (backend.submissions == 2
           && backend.lastRequest.geometry.rootId == geometry->rootId()
           && backend.lastRequest.geometry.records.size() == 1
           && backend.lastRequest.geometry.records[0].stableId == geometry->rootId()
           && backend.lastRequest.geometry.records[0].operation
               == videowire::SdfOperation::sphere
           && backend.lastRequest.geometry.records[0].parameters[0] == 1.0
           && backend.lastRequest.compiledProgram != nullptr
           && backend.lastRequest.compiledProgram->records().size() == 1
           && backend.lastRequest.compiledProgram->rootIndex() == 0
           && backend.firstCompiledProgram == backend.lastRequest.compiledProgram
           && backend.lastRequest.maximumSteps == controls.maximumSteps
           && backend.lastRequest.epsilon == controls.epsilon
           && backend.lastRequest.maximumDistance == controls.maximumDistance
           && backend.lastRequest.adaptiveQuality == controls.adaptiveQuality
           && backend.lastRequest.normalQuality == controls.normalQuality
           && backend.lastRequest.shadowQuality == controls.shadowQuality,
           "preview and export reuse one immutable compiled program owner");
    check (preview.use == NativeSdfRenderUse::Preview
           && exported.use == NativeSdfRenderUse::Export
           && preview.structuralDigest == geometry->structuralDigest()
           && exported.structuralDigest == geometry->structuralDigest()
           && preview.nativeFrame != nullptr && exported.nativeFrame != nullptr,
           "preview and export differ only in use metadata");

    const NativeSdfCacheIdentity cacheIdentity { 7, 1, 1, 1 };
    const auto oldestGeometry = sphereGeometry (10.0);
    NativeSdfRenderedFrame cached;
    check (renderer.renderPreview (oldestGeometry, { 32, 32 }, controls,
                                   kNativeGpuCapability, cached, error, cacheIdentity),
           "bounded native geometry cache admits its first scoped record");
    const auto oldestProgram = backend.lastRequest.compiledProgram;
    std::shared_ptr<const arbitgpu::NativeSdfCompiledProgram> newestProgram;
    for (int index = 1; index <= 32; ++index)
    {
        check (renderer.renderPreview (sphereGeometry (10.0 + index), { 32, 32 }, controls,
                                       kNativeGpuCapability, cached, error, cacheIdentity),
               "bounded native geometry cache accepts an admitted revision");
        newestProgram = backend.lastRequest.compiledProgram;
    }
    check (renderer.compiledGeometryCount() == 32,
           "native geometry cache evicts deterministically at its fixed capacity");
    check (renderer.renderPreview (oldestGeometry, { 32, 32 }, controls,
                                   kNativeGpuCapability, cached, error, cacheIdentity)
            && backend.lastRequest.compiledProgram != oldestProgram,
           "native geometry cache evicts the least recently used compiled program");
    check (renderer.renderPreview (sphereGeometry (42.0), { 32, 32 }, controls,
                                   kNativeGpuCapability, cached, error, cacheIdentity)
            && backend.lastRequest.compiledProgram == newestProgram,
           "native geometry cache reuses the most recently retained compiled program");
    const NativeSdfCacheIdentity revisedIdentity { 7, 2, 1, 1 };
    check (renderer.renderPreview (geometry, { 32, 32 }, controls,
                                   kNativeGpuCapability, cached, error, cacheIdentity),
           "native geometry cache admits the first plan revision");
    const auto revisionOneProgram = backend.lastRequest.compiledProgram;
    check (renderer.renderPreview (geometry, { 32, 32 }, controls,
                                   kNativeGpuCapability, cached, error, revisedIdentity)
            && backend.lastRequest.compiledProgram != revisionOneProgram,
           "plan revision invalidates the previous scoped compiled geometry");
    const auto planTwoProgram = backend.lastRequest.compiledProgram;
    const NativeSdfCacheIdentity helperRevisedIdentity { 7, 2, 2, 1 };
    check (renderer.renderPreview (geometry, { 32, 32 }, controls,
                                   kNativeGpuCapability, cached, error, helperRevisedIdentity)
            && backend.lastRequest.compiledProgram != planTwoProgram,
           "helper generation invalidates the previous scoped compiled geometry");
    const auto helperProgram = backend.lastRequest.compiledProgram;
    const NativeSdfCacheIdentity projectRevisedIdentity { 8, 2, 2, 1 };
    check (renderer.renderPreview (geometry, { 32, 32 }, controls,
                                   kNativeGpuCapability, cached, error, projectRevisedIdentity)
            && backend.lastRequest.compiledProgram != helperProgram,
           "project generation keeps compiled geometry from unrelated snapshots separate");
    renderer.invalidateCompiledGeometry();
    check (renderer.compiledGeometryCount() == 0,
           "native geometry cache has an explicit lifecycle invalidation path");

    const int submissionsBeforeFailures = backend.submissions;
    const auto previewHandle = preview.nativeFrame->colorImageHandle();
    backend.rejectSubmission = true;
    check (! renderer.renderPreview (geometry, { 320, 180 }, controls,
                                     kNativeGpuCapability, preview, error)
           && error == "fake native SDF draw rejected"
           && preview.nativeFrame->colorImageHandle() == previewHandle,
           "a failed native draw preserves the previous backend-owned frame");
    backend.rejectSubmission = false;

    backend.reportedCapabilities.supportedOperations.fill (false);
    check (! renderer.renderExport (geometry, { 320, 180 }, controls,
                                    kNativeGpuCapability, exported, error)
           && error == "native GPU SDF backend does not support operation sphere"
           && backend.submissions == submissionsBeforeFailures + 1,
           "unsupported geometry fails before backend execution");

    backend.reportedCapabilities = sphereCapabilities();
    check (! renderer.renderPreview (nullptr, { 320, 180 }, controls,
                                     kNativeGpuCapability, preview, error)
           && error == "native GPU SDF rendering requires immutable admitted geometry"
           && backend.submissions == submissionsBeforeFailures + 1,
           "missing immutable geometry fails before backend execution");

    check (! renderer.renderPreview (geometry, { 320, 180 }, controls,
                                     "native-gpu ", preview, error)
           && error == "native SDF rendering requires backendCapability native-gpu"
           && backend.submissions == submissionsBeforeFailures + 1,
           "backendCapability must remain exactly native-gpu before execution");

    NativeSdfRenderer stubRenderer (arbitgpu::nativeSdfExecutionBackend());
    NativeSdfRenderedFrame stubOutput;
    check (! stubRenderer.renderPreview (
               geometry, { 320, 180 }, controls,
               kNativeGpuCapability, stubOutput, error)
           && error == "native GPU SDF execution is not compiled in"
           && stubOutput.nativeFrame == nullptr,
           "the compiled stub has no CPU SDF rendering fallback");

    const auto stubSubmission = arbitgpu::nativeSdfExecutionBackend().render ({});
    check (! stubSubmission.rendered && stubSubmission.frame == nullptr
           && stubSubmission.error == "native GPU SDF execution is not compiled in",
           "the stub backend rejects direct execution without allocating a frame");

    const int deletionsBeforeDirectFrameRelease = backend.deletions;
    preview = {};
    exported = {};
    check (backend.deletions == deletionsBeforeDirectFrameRelease + 2,
           "each successful native SDF frame releases its owner exactly once");

    videowire::SdfRaymarchOperation wireOperation;
    wireOperation.terminalStableId = 81;
    wireOperation.maximumSteps = controls.maximumSteps;
    wireOperation.epsilon = controls.epsilon;
    wireOperation.maximumDistance = controls.maximumDistance;
    wireOperation.adaptiveQuality = static_cast<std::uint8_t> (controls.adaptiveQuality);
    wireOperation.normalQuality = static_cast<std::uint8_t> (controls.normalQuality);
    wireOperation.shadowQuality = static_cast<std::uint8_t> (controls.shadowQuality);
    wireOperation.geometry.rootId = 71;
    videowire::SdfRecord wireSphere;
    wireSphere.stableId = 71;
    wireSphere.operation = videowire::SdfOperation::sphere;
    wireSphere.parameterCount = 1;
    wireSphere.parameters[0] = 1.0;
    wireOperation.geometry.records.push_back (wireSphere);

    videowire::CompiledVisualLayerPlan plan;
    plan.clipId = 501;
    plan.structuralRevision = 2;
    plan.producerValidated = true;
    plan.nodeIds = { 70, 80, 90 };
    plan.nodeKinds = { "visual.sdf.sphere", "visual.sdf.raymarch", "video.out" };
    plan.operations = {
        { 70, "visual.sdf.sphere", "control-eval", {} },
        { 80, "visual.sdf.raymarch", "native-gpu",
          videowire::encodeSdfRaymarchOperation (wireOperation) },
        { 90, "video.out", "native-gpu", {} }
    };
    plan.ports = {
        { 70, 0, 1, "out", "control", "sdf", "unspecified", "unspecified" },
        { 80, 0, 1, "in", "control", "sdf", "unspecified", "unspecified" },
        { 80, 1, 1, "out", "frame", "image", "rgba8", "srgb" },
        { 90, 0, 1, "in", "frame", "image", "rgba8", "srgb" }
    };
    plan.edges = { { 70, 0, 80, 0 }, { 80, 1, 90, 0 } };
    std::vector<NativeSdfRenderedFrame> visualOwners;
    TestLayer visualLayer;
    const int submissionsBeforeVisualPlan = backend.submissions;
    check (prepareVisualSdfLayer ({ plan }, plan.clipId, 640, 360,
                NativeSdfRenderUse::Preview, renderer, visualLayer, visualOwners, error)
               == VisualSdfPreparation::rendered
           && backend.submissions == submissionsBeforeVisualPlan + 1
           && visualLayer.texture == 0
           && visualLayer.nativeTextureBackend == "test-native"
           && visualLayer.nativeTextureView != 0
           && visualLayer.texWidth == 640 && visualLayer.texHeight == 360
           && visualOwners.size() == 1,
           "the exact typed SDF plan publishes its backend-owned native view");

    auto staleMalformedPlan = plan;
    staleMalformedPlan.structuralRevision = plan.structuralRevision - 1;
    staleMalformedPlan.edges.push_back ({ 70, 0, 80, 0 });
    const int submissionsBeforeDuplicatePlan = backend.submissions;
    check (prepareVisualSdfLayer ({ staleMalformedPlan, plan }, plan.clipId, 640, 360,
                NativeSdfRenderUse::Preview, renderer, visualLayer, visualOwners, error)
               == VisualSdfPreparation::rendered
           && backend.submissions == submissionsBeforeDuplicatePlan + 1,
           "SDF execution ignores a stale duplicate in favor of the newest clip revision");

    auto malformedPlan = plan;
    malformedPlan.edges.push_back ({ 70, 0, 80, 0 });
    check (prepareVisualSdfLayer ({ malformedPlan }, malformedPlan.clipId, 640, 360,
                NativeSdfRenderUse::Export, renderer, visualLayer, visualOwners, error)
               == VisualSdfPreparation::rejected
           && backend.submissions == submissionsBeforeVisualPlan + 2,
           "duplicate SDF bindings fail before native export execution");
    const int deletionsBeforeVisualFrameRelease = backend.deletions;
    visualOwners.clear();
    check (backend.deletions == deletionsBeforeVisualFrameRelease + 2,
           "the product SDF frame remains owned through compositing and releases once");

    std::printf ("sdf-native-renderer-dispatch: %d/%d checks passed; submissions=%d deletions=%d\n",
                 checks - failures, checks, backend.submissions, backend.deletions);
    return failures == 0 ? 0 : 1;
}
