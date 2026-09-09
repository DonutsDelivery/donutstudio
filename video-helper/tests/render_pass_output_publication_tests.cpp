#include "../src/render_pass_output_publication.h"
#include "support/fixture_scene.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
[[noreturn]] void fail (const char* message)
{
    std::cerr << "FAIL: " << message << '\n';
    std::exit (1);
}

void require (bool condition, const char* message)
{
    if (! condition)
        fail (message);
}

renderpassoutput::AttachmentDescription attachment (
    renderpassoutput::Output output, renderpassoutput::Extent extent)
{
    const auto exact = renderpassoutput::requirements (output);
    return { output, exact.format, exact.colorSpace, extent };
}

renderpassoutput::Description descriptionFor (
    renderpassoutput::Extent extent,
    std::initializer_list<renderpassoutput::Output> outputs)
{
    renderpassoutput::Description description;
    description.extent = extent;
    for (const auto output : outputs)
        description.attachments.push_back (attachment (output, extent));
    return description;
}

// This test backend supplies opaque nonzero resource and submission identities.
// It verifies helper transactions, not physical GPU pixels.
class ResourceBackend final : public arbitgpu::RenderPassOutputBackend
{
public:
    ResourceBackend()
    {
        capabilities.available = true;
        capabilities.supportedOutputs.fill (true);
    }

    arbitgpu::RenderPassOutputCapabilities renderPassOutputCapabilities() const override
    {
        ++capabilityQueries;
        return capabilities;
    }

    arbitgpu::RenderPassOutputAdmission admitRenderPassOutputs (
        const renderpassoutput::AdmittedOutputs& outputs) override
    {
        ++admissionCalls;
        admittedByteCounts.push_back (outputs.totalByteCount());
        arbitgpu::FrameMemoryAdmission frameMemory;
        std::string frameMemoryError;
        if (! arbitgpu::admitRenderPassOutputFrameMemory (
                outputs, backendAllocationBudget, frameMemory, frameMemoryError))
        {
            lastFrameMemory = frameMemory;
            arbitgpu::RenderPassOutputAdmission rejected;
            rejected.error = std::move (frameMemoryError);
            rejected.frameMemory = frameMemory;
            return rejected;
        }
        lastFrameMemory = frameMemory;
        if (rejectAdmission)
            return { {}, {}, rejectionDiagnostic, frameMemory };
        const arbitgpu::RenderPassOutputLifecycleHandle lifecycle { nextLifecycle++ };
        std::vector<arbitgpu::RenderPassOutputResource> resources;
        for (const auto& attachment : outputs.attachments())
        {
            const auto base = static_cast<std::uintptr_t> (nextResource);
            resources.push_back ({ attachment.output, base, base + 1, base + 2 });
            nextResource += 3;
        }
        if (omitLastResource && ! resources.empty())
            resources.pop_back();
        if (zeroTextureView && ! resources.empty())
            resources.back().textureView = 0;
        if (contradictoryAdmission)
            return { lifecycle, std::move (resources), "test contradiction", frameMemory };
        admittedLifecycles.push_back (lifecycle);
        arbitgpu::RenderPassOutputAdmission result;
        result.lifecycle = lifecycle;
        result.resources = std::move (resources);
        result.frameMemory = frameMemory;
        return result;
    }

    arbitgpu::RenderPassColorAovExecution executeColorAovClear (
        arbitgpu::RenderPassOutputLifecycleHandle lifecycle,
        const arbitgpu::RenderPassColorAovClear& clear) override
    {
        ++executionCalls;
        executedLifecycles.push_back (lifecycle);
        executedClears.push_back (clear);
        if (rejectExecution)
            return { false, 0, executionDiagnostic };
        if (contradictoryExecution)
            return { true, 0, {} };
        return { true, nextSubmission++, {} };
    }

    arbitgpu::RenderPassMotionAovExecution executeMotionAovClear (
        arbitgpu::RenderPassOutputLifecycleHandle lifecycle,
        const arbitgpu::RenderPassMotionAovClear& clear) override
    {
        ++motionExecutionCalls;
        executedMotionLifecycles.push_back (lifecycle);
        executedMotionClears.push_back (clear);
        if (rejectExecution)
            return { false, 0, executionDiagnostic };
        if (contradictoryExecution)
            return { true, 0, {} };
        return { true, nextSubmission++, {} };
    }

    arbitgpu::RenderPassSceneAovExecution executeSceneAov (
        arbitgpu::RenderPassOutputLifecycleHandle lifecycle,
        const sceneaov::Payload& payload) override
    {
        ++sceneExecutionCalls;
        executedSceneLifecycles.push_back (lifecycle);
        executedScenePayloads.push_back (payload);
        if (rejectExecution)
            return { false, 0, executionDiagnostic };
        if (contradictoryExecution)
            return { true, 0, {} };
        return { true, nextSubmission++, {} };
    }

    arbitgpu::RenderPassAovInspectionExecution executeAovInspection (
        arbitgpu::RenderPassOutputLifecycleHandle lifecycle,
        const aovinspection::Payload& payload) override
    {
        ++inspectionExecutionCalls;
        executedInspectionLifecycles.push_back (lifecycle);
        executedInspectionPayloads.push_back (payload);
        if (rejectExecution || rejectInspectionExecution)
            return { false, 0, executionDiagnostic };
        if (contradictoryExecution)
            return { true, 0, {} };
        return { true, nextSubmission++, {} };
    }

    void releaseRenderPassOutputs (
        arbitgpu::RenderPassOutputLifecycleHandle lifecycle) noexcept override
    {
        releasedLifecycles.push_back (lifecycle);
    }

    arbitgpu::RenderPassOutputCapabilities capabilities;
    mutable int capabilityQueries = 0;
    int admissionCalls = 0;
    bool rejectAdmission = false;
    bool contradictoryAdmission = false;
    bool omitLastResource = false;
    bool zeroTextureView = false;
    bool rejectExecution = false;
    bool rejectInspectionExecution = false;
    bool contradictoryExecution = false;
    std::string rejectionDiagnostic;
    std::string executionDiagnostic;
    uint64_t backendAllocationBudget = renderpassoutput::kMaximumTotalBytes;
    arbitgpu::FrameMemoryAdmission lastFrameMemory;
    uint64_t nextLifecycle = 1001;
    uint64_t nextResource = 5001;
    uint64_t nextSubmission = 7001;
    int executionCalls = 0;
    int motionExecutionCalls = 0;
    int sceneExecutionCalls = 0;
    int inspectionExecutionCalls = 0;
    std::vector<uint64_t> admittedByteCounts;
    std::vector<arbitgpu::RenderPassOutputLifecycleHandle> admittedLifecycles;
    std::vector<arbitgpu::RenderPassOutputLifecycleHandle> releasedLifecycles;
    std::vector<arbitgpu::RenderPassOutputLifecycleHandle> executedLifecycles;
    std::vector<arbitgpu::RenderPassColorAovClear> executedClears;
    std::vector<arbitgpu::RenderPassOutputLifecycleHandle> executedMotionLifecycles;
    std::vector<arbitgpu::RenderPassMotionAovClear> executedMotionClears;
    std::vector<arbitgpu::RenderPassOutputLifecycleHandle> executedSceneLifecycles;
    std::vector<sceneaov::Payload> executedScenePayloads;
    std::vector<arbitgpu::RenderPassOutputLifecycleHandle> executedInspectionLifecycles;
    std::vector<aovinspection::Payload> executedInspectionPayloads;
};
} // namespace

int main()
{
    static_assert (! std::is_copy_constructible_v<videowire::RenderPassOutputPublication>);
    static_assert (! std::is_move_constructible_v<videowire::RenderPassOutputPublication>);
    static_assert (std::is_const_v<std::remove_reference_t<decltype (
        std::declval<const videowire::RenderPassOutputPublication&>().outputs())>>);
    static_assert (std::is_const_v<std::remove_reference_t<decltype (
        std::declval<const videowire::RenderPassOutputPublication&>().resources())>>);

    ResourceBackend backend;
    std::string diagnostic;
    videowire::RenderPassOutputPublicationPtr publication;

    renderpassoutput::Description empty;
    empty.extent = { 32, 16 };
    require (videowire::makeRenderPassOutputPublication (
                 empty, publication, diagnostic, arbitgpu::nativeRenderPassOutputBackend())
             && publication != nullptr && publication->outputs().attachments().empty()
             && publication->resources().empty()
             && ! publication->hasBackendAdmission() && diagnostic.empty(),
             "an empty optional-output set should not require a native backend");

    const auto initial = publication;
    auto malformed = descriptionFor ({ 32, 16 }, { renderpassoutput::Output::Color });
    malformed.attachments[0].format = renderpassoutput::PixelFormat::R8Unorm;
    require (! videowire::makeRenderPassOutputPublication (
                 malformed, publication, diagnostic, backend)
             && diagnostic == "render-pass output contract rejected: formatMismatch"
             && publication == initial && backend.capabilityQueries == 0
             && backend.admissionCalls == 0,
             "malformed descriptors must reject before backend capability admission");

    const auto colorDepth = descriptionFor (
        { 64, 32 }, { renderpassoutput::Output::Depth, renderpassoutput::Output::Color });
    require (! videowire::makeRenderPassOutputPublication (
                 colorDepth, publication, diagnostic, arbitgpu::nativeRenderPassOutputBackend())
             && diagnostic == "native GPU render-pass outputs are not compiled in"
             && publication == initial,
             "the stub backend must fail closed for nonempty output requests");

    backend.capabilities.limits.totalBytes = 64ull * 32ull * 12ull - 1ull;
    require (! videowire::makeRenderPassOutputPublication (
                 colorDepth, publication, diagnostic, backend)
             && diagnostic == "render-pass output backend budget rejected: byteLimitExceeded"
             && publication == initial && backend.admissionCalls == 0,
             "backend byte limits must reject before lifecycle admission");

    ResourceBackend allocationBudgetBackend;
    allocationBudgetBackend.backendAllocationBudget = 64ull * 32ull * 12ull - 1ull;
    require (! videowire::makeRenderPassOutputPublication (
                 colorDepth, publication, diagnostic, allocationBudgetBackend)
             && diagnostic == "native frame-memory budget exceeded: requested 24576 bytes for 2 slots, budget 24575 bytes"
             && publication == initial && allocationBudgetBackend.admissionCalls == 1
             && allocationBudgetBackend.admittedLifecycles.empty()
             && allocationBudgetBackend.lastFrameMemory.failure
                    == arbitgpu::FrameMemoryAdmissionFailure::byteBudgetExceeded
             && allocationBudgetBackend.lastFrameMemory.requestedBytes == 24576
             && allocationBudgetBackend.lastFrameMemory.allocatedSlots == 2,
             "backend-owned byte admission must reject before allocation and retain last-good output");

    backend.capabilities.limits = {};
    backend.capabilities.limits.attachments = 1;
    require (! videowire::makeRenderPassOutputPublication (
                 colorDepth, publication, diagnostic, backend)
             && diagnostic == "render-pass output backend budget rejected: attachmentCountExceeded"
             && publication == initial && backend.admissionCalls == 0,
             "backend attachment limits must reject before lifecycle admission");

    backend.capabilities.limits = {};
    backend.capabilities.supportedOutputs[
        static_cast<size_t> (renderpassoutput::Output::Depth)] = false;
    require (! videowire::makeRenderPassOutputPublication (
                 colorDepth, publication, diagnostic, backend)
             && diagnostic == "native GPU backend does not support render-pass output depth"
             && publication == initial && backend.admissionCalls == 0,
             "unsupported optional outputs must fail closed");
    backend.capabilities.supportedOutputs.fill (true);

    const auto allOutputs = descriptionFor (
        { 64, 32 }, { renderpassoutput::Output::ObjectId,
                      renderpassoutput::Output::MaterialId,
                      renderpassoutput::Output::Mask,
                      renderpassoutput::Output::Emission,
                      renderpassoutput::Output::Motion,
                      renderpassoutput::Output::Normal,
                      renderpassoutput::Output::Depth,
                      renderpassoutput::Output::Color });
    videowire::RenderPassOutputPublicationPtr preview;
    videowire::RenderPassOutputPublicationPtr exportRun;
    require (videowire::makeRenderPassOutputPublication (
                 allOutputs, preview, diagnostic, backend)
             && videowire::makeRenderPassOutputPublication (
                 allOutputs, exportRun, diagnostic, backend)
             && preview->hasBackendAdmission() && exportRun->hasBackendAdmission()
             && preview->outputs().attachments().size() == renderpassoutput::kMaximumAttachments
             && preview->outputs().attachments()[0].output == renderpassoutput::Output::Color
             && preview->outputs().attachments()[7].output == renderpassoutput::Output::ObjectId
             && preview->outputs().totalByteCount() == 64ull * 32ull * 41ull
             && preview->resources().size() == renderpassoutput::kMaximumAttachments
             && preview->resource (renderpassoutput::Output::Mask) != nullptr
             && preview->resource (renderpassoutput::Output::Mask)->image != 0
             && preview->frameMemory().failure
                    == arbitgpu::FrameMemoryAdmissionFailure::none
             && preview->frameMemory().requestedBytes == 64ull * 32ull * 41ull
             && preview->frameMemory().allocatedSlots
                    == renderpassoutput::kMaximumAttachments
             && backend.admissionCalls == 2 && backend.releasedLifecycles.empty(),
             "preview and export must publish exact immutable resource and byte admission");

    auto leasedPreview = preview;
    const auto maskOnly = descriptionFor (
        { 64, 32 }, { renderpassoutput::Output::Mask });
    require (videowire::makeRenderPassOutputPublication (
                 maskOnly, preview, diagnostic, backend)
             && backend.admissionCalls == 3 && backend.releasedLifecycles.empty(),
             "publication replacement should install a new backend lifecycle");
    preview.reset();
    require (backend.releasedLifecycles.size() == 1
             && backend.releasedLifecycles[0]
                    == arbitgpu::RenderPassOutputLifecycleHandle { 1003 },
             "the newest lifecycle should release when its final publication is dropped");
    require (leasedPreview->outputs().find (renderpassoutput::Output::Color) != nullptr,
             "a leased old publication must remain immutable after replacement");
    leasedPreview.reset();
    require (backend.releasedLifecycles.size() == 2
             && backend.releasedLifecycles[1]
                    == arbitgpu::RenderPassOutputLifecycleHandle { 1001 },
             "an old lifecycle should release only after its final reader drops it");

    auto retainedExport = exportRun;
    backend.rejectAdmission = true;
    backend.rejectionDiagnostic = "contract-only backend rejected admission";
    require (! videowire::makeRenderPassOutputPublication (
                 maskOnly, exportRun, diagnostic, backend)
             && diagnostic == backend.rejectionDiagnostic && exportRun == retainedExport
             && backend.releasedLifecycles.size() == 2,
             "failed backend admission must retain the last publication");
    backend.rejectAdmission = false;

    backend.contradictoryAdmission = true;
    videowire::RenderPassOutputPublicationPtr contradiction;
    require (! videowire::makeRenderPassOutputPublication (
                 maskOnly, contradiction, diagnostic, backend)
             && contradiction == nullptr
             && diagnostic == "native GPU backend returned contradictory render-pass output admission"
             && backend.releasedLifecycles.size() == 3
             && backend.releasedLifecycles.back()
                    == arbitgpu::RenderPassOutputLifecycleHandle { 1004 },
             "a contradictory backend result must release its lifecycle immediately");
    backend.contradictoryAdmission = false;

    backend.omitLastResource = true;
    require (! videowire::makeRenderPassOutputPublication (
                 maskOnly, contradiction, diagnostic, backend)
             && contradiction == nullptr
             && diagnostic == "native GPU backend returned incomplete render-pass output resources"
             && backend.releasedLifecycles.size() == 4
             && backend.releasedLifecycles.back()
                    == arbitgpu::RenderPassOutputLifecycleHandle { 1005 },
             "an incomplete resource set must fail closed and release its lifecycle");
    backend.omitLastResource = false;

    backend.zeroTextureView = true;
    require (! videowire::makeRenderPassOutputPublication (
                 maskOnly, contradiction, diagnostic, backend)
             && contradiction == nullptr
             && diagnostic == "native GPU backend returned incomplete render-pass output resources"
             && backend.releasedLifecycles.size() == 5
             && backend.releasedLifecycles.back()
                    == arbitgpu::RenderPassOutputLifecycleHandle { 1006 },
             "zero native handles must fail closed and release their lifecycle");
    backend.zeroTextureView = false;

    exportRun.reset();
    require (backend.releasedLifecycles.size() == 5,
             "another shared reader must keep the export lifecycle alive");
    retainedExport.reset();
    require (backend.releasedLifecycles.size() == 6
             && backend.releasedLifecycles.back()
                    == arbitgpu::RenderPassOutputLifecycleHandle { 1002 },
             "the export lifecycle should release exactly once at final ownership drop");

    const auto stubCapabilities = arbitgpu::nativeRenderPassOutputBackend()
                                      .renderPassOutputCapabilities();
    renderpassoutput::AdmissionFailure failure = renderpassoutput::AdmissionFailure::None;
    auto exact = renderpassoutput::admit (maskOnly, failure);
    require (exact.has_value() && ! stubCapabilities.available
             && ! arbitgpu::nativeRenderPassOutputBackend()
                     .admitRenderPassOutputs (*exact).lifecycle,
             "stub capability and direct admission must both remain unavailable");

    const auto colorOnly = descriptionFor (
        { 48, 24 }, { renderpassoutput::Output::Color });
    const arbitgpu::RenderPassColorAovClear clear {
        { 0.125f, 0.25f, 1.5f, 0.75f } };
    ResourceBackend executionBackend;
    videowire::RenderPassOutputPublicationPtr viewportPass;
    videowire::RenderPassOutputPublicationPtr exportPass;

    auto nonColor = descriptionFor (
        { 48, 24 }, { renderpassoutput::Output::Depth });
    require (! videowire::makeColorAovPassPublication (
                 nonColor, clear, viewportPass, diagnostic, executionBackend)
             && diagnostic == "native Color AOV pass requires exactly one Color attachment"
             && executionBackend.capabilityQueries == 0
             && executionBackend.admissionCalls == 0
             && executionBackend.executionCalls == 0,
             "Color AOV execution must reject other pass semantics before allocation");

    auto nonFinite = clear;
    nonFinite.linearRgba[2] = std::nanf ("");
    require (! videowire::makeColorAovPassPublication (
                 colorOnly, nonFinite, viewportPass, diagnostic, executionBackend)
             && diagnostic == "native Color AOV clear values must be finite"
             && executionBackend.admissionCalls == 0
             && executionBackend.executionCalls == 0,
             "Color AOV execution must reject a non-finite immutable payload");

    require (videowire::makeColorAovPassPublication (
                 colorOnly, clear, viewportPass, diagnostic, executionBackend)
             && videowire::makeColorAovPassPublication (
                 colorOnly, clear, exportPass, diagnostic, executionBackend)
             && viewportPass->hasBackendAdmission()
             && viewportPass->resource (renderpassoutput::Output::Color) != nullptr
             && viewportPass->colorAovExecution().submitted
             && viewportPass->colorAovExecution().submission == 7001
             && viewportPass->colorAovClear().linearRgba == clear.linearRgba
             && exportPass->colorAovExecution().submission == 7002
             && executionBackend.admissionCalls == 2
             && executionBackend.executionCalls == 2
             && executionBackend.executedLifecycles[0]
                    == executionBackend.admittedLifecycles[0]
             && executionBackend.executedClears[1].linearRgba == clear.linearRgba,
             "viewport and export must submit the same immutable native Color AOV pass");

    auto retainedViewportPass = viewportPass;
    executionBackend.rejectExecution = true;
    executionBackend.executionDiagnostic = "test native Color AOV submission rejected";
    require (! videowire::makeColorAovPassPublication (
                 colorOnly, clear, viewportPass, diagnostic, executionBackend)
             && diagnostic == executionBackend.executionDiagnostic
             && viewportPass == retainedViewportPass
             && executionBackend.admissionCalls == 3
             && executionBackend.executionCalls == 3
             && executionBackend.releasedLifecycles.size() == 1
             && executionBackend.releasedLifecycles.back()
                    == executionBackend.admittedLifecycles.back(),
             "failed native Color AOV submission must release its candidate and retain last-good");
    executionBackend.rejectExecution = false;

    executionBackend.contradictoryExecution = true;
    videowire::RenderPassOutputPublicationPtr contradictionPass;
    require (! videowire::makeColorAovPassPublication (
                 colorOnly, clear, contradictionPass, diagnostic, executionBackend)
             && diagnostic == "native GPU backend returned contradictory Color AOV execution"
             && contradictionPass == nullptr
             && executionBackend.releasedLifecycles.size() == 2,
             "a contradictory native Color AOV receipt must fail closed and release resources");
    executionBackend.contradictoryExecution = false;

    const auto motionOnly = descriptionFor (
        { 48, 24 }, { renderpassoutput::Output::Motion });
    const arbitgpu::RenderPassMotionAovClear motionClear { { 0.0f, 0.0f } };
    ResourceBackend motionBackend;
    videowire::RenderPassOutputPublicationPtr viewportMotion;
    videowire::RenderPassOutputPublicationPtr exportMotion;
    require (! videowire::makeMotionAovPassPublication (
                 colorOnly, motionClear, viewportMotion, diagnostic, motionBackend)
             && diagnostic == "native Motion AOV pass requires exactly one Motion attachment"
             && motionBackend.admissionCalls == 0
             && motionBackend.motionExecutionCalls == 0,
             "Motion AOV execution must reject other pass semantics before allocation");

    auto nonFiniteMotion = motionClear;
    nonFiniteMotion.pixelDisplacement[1] = std::nanf ("");
    require (! videowire::makeMotionAovPassPublication (
                 motionOnly, nonFiniteMotion, viewportMotion, diagnostic, motionBackend)
             && diagnostic == "native Motion AOV clear values must be finite"
             && motionBackend.admissionCalls == 0
             && motionBackend.motionExecutionCalls == 0,
             "Motion AOV execution must reject a non-finite immutable payload");

    require (videowire::makeMotionAovPassPublication (
                 motionOnly, motionClear, viewportMotion, diagnostic, motionBackend)
             && videowire::makeMotionAovPassPublication (
                 motionOnly, motionClear, exportMotion, diagnostic, motionBackend)
             && viewportMotion->hasBackendAdmission()
             && viewportMotion->resource (renderpassoutput::Output::Motion) != nullptr
             && viewportMotion->motionAovExecution().submitted
             && viewportMotion->motionAovExecution().submission == 7001
             && viewportMotion->motionAovClear().pixelDisplacement
                  == motionClear.pixelDisplacement
             && exportMotion->motionAovExecution().submission == 7002
             && motionBackend.admissionCalls == 2
             && motionBackend.motionExecutionCalls == 2
             && motionBackend.executedMotionLifecycles[0]
                  == motionBackend.admittedLifecycles[0]
             && motionBackend.executedMotionClears[1].pixelDisplacement
                  == motionClear.pixelDisplacement,
             "viewport and export must submit the same immutable native Motion AOV pass");

    auto retainedMotion = viewportMotion;
    motionBackend.rejectExecution = true;
    motionBackend.executionDiagnostic = "test native Motion AOV submission rejected";
    require (! videowire::makeMotionAovPassPublication (
                 motionOnly, motionClear, viewportMotion, diagnostic, motionBackend)
             && diagnostic == motionBackend.executionDiagnostic
             && viewportMotion == retainedMotion
             && motionBackend.admissionCalls == 3
             && motionBackend.motionExecutionCalls == 3
             && motionBackend.releasedLifecycles.size() == 1,
             "failed native Motion AOV submission must release its candidate and retain last-good");
    motionBackend.rejectExecution = false;

    const auto fixture = videohelper::fixture3d::makeScene();
    sceneaov::Payload depthScene;
    depthScene.version = sceneaov::kWireVersion;
    depthScene.output = renderpassoutput::Output::Depth;
    depthScene.extent = { 64, 32 };
    depthScene.scene = std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (fixture);
    aovinspection::Payload depthInspection;
    depthInspection.schemaVersion = aovinspection::kSchemaVersion;
    depthInspection.source = aovinspection::Source::Depth;
    depthInspection.extent = depthScene.extent;
    depthInspection.depthNear = 0.1f;
    depthInspection.depthFar = 1.0f;
    ResourceBackend inspectionBackend;
    videowire::RenderPassOutputPublicationPtr viewportInspection;
    videowire::RenderPassOutputPublicationPtr exportInspection;
    require (videowire::makeAovInspectionPassPublication (
                 depthScene, depthInspection, viewportInspection, diagnostic, inspectionBackend)
             && videowire::makeAovInspectionPassPublication (
                 depthScene, depthInspection, exportInspection, diagnostic, inspectionBackend)
             && viewportInspection->sceneAovExecution().submitted
             && viewportInspection->aovInspectionExecution().submitted
             && viewportInspection->sceneAovExecution().submission == 7001
             && viewportInspection->aovInspectionExecution().submission == 7002
             && exportInspection->sceneAovExecution().submission == 7003
             && exportInspection->aovInspectionExecution().submission == 7004
             && viewportInspection->resource (renderpassoutput::Output::Depth) != nullptr
             && viewportInspection->resource (renderpassoutput::Output::Color) != nullptr
             && inspectionBackend.admissionCalls == 2
             && inspectionBackend.sceneExecutionCalls == 2
             && inspectionBackend.inspectionExecutionCalls == 2
             && inspectionBackend.executedSceneLifecycles[0]
                  == inspectionBackend.executedInspectionLifecycles[0]
             && inspectionBackend.executedSceneLifecycles[1]
                  == inspectionBackend.executedInspectionLifecycles[1],
             "viewport and export must render and inspect the same native scene AOV lifecycle");

    auto retainedInspection = viewportInspection;
    inspectionBackend.rejectInspectionExecution = true;
    inspectionBackend.executionDiagnostic = "test native AOV inspection rejected";
    require (! videowire::makeAovInspectionPassPublication (
                 depthScene, depthInspection, viewportInspection, diagnostic, inspectionBackend)
             && diagnostic == inspectionBackend.executionDiagnostic
             && viewportInspection == retainedInspection
             && inspectionBackend.admissionCalls == 3
             && inspectionBackend.sceneExecutionCalls == 3
             && inspectionBackend.inspectionExecutionCalls == 3
             && inspectionBackend.releasedLifecycles.size() == 1,
             "failed inspection mapping must release its candidate and retain last-good pixels");
    inspectionBackend.rejectInspectionExecution = false;

    const auto stubExecution = arbitgpu::nativeRenderPassOutputBackend()
                                   .executeColorAovClear ({ 999 }, clear);
    require (! stubExecution.submitted && stubExecution.submission == 0
             && stubExecution.error == "native GPU Color AOV execution is not compiled in",
             "the stub backend must reject direct Color AOV execution without fallback");
    const auto stubMotionExecution = arbitgpu::nativeRenderPassOutputBackend()
                                         .executeMotionAovClear ({ 999 }, motionClear);
    require (! stubMotionExecution.submitted && stubMotionExecution.submission == 0
             && stubMotionExecution.error
                  == "native GPU Motion AOV execution is not compiled in",
             "the stub backend must reject direct Motion AOV execution without fallback");

    viewportInspection.reset();
    retainedInspection.reset();
    exportInspection.reset();
    require (inspectionBackend.releasedLifecycles.size() == 3,
             "viewport and export AOV inspection lifecycles must release exactly once");

    viewportMotion.reset();
    retainedMotion.reset();
    exportMotion.reset();
    require (motionBackend.releasedLifecycles.size() == 3,
             "viewport and export Motion AOV lifecycles must release exactly once");

    viewportPass.reset();
    require (executionBackend.releasedLifecycles.size() == 2,
             "a shared Color AOV publication must retain its native lifecycle");
    retainedViewportPass.reset();
    exportPass.reset();
    require (executionBackend.releasedLifecycles.size() == 4,
             "viewport and export Color AOV lifecycles must release exactly once");

    std::cout << "render pass output publication: PASS\n";
    return 0;
}
