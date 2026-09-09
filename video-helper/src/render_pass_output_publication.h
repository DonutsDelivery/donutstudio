#pragma once

#include "gpu_backend/backend.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace videowire
{
class RenderPassOutputPublication;
using RenderPassOutputPublicationPtr = std::shared_ptr<const RenderPassOutputPublication>;

bool makeRenderPassOutputPublication (
    const renderpassoutput::Description& description,
    RenderPassOutputPublicationPtr& publication,
    std::string& diagnostic,
    arbitgpu::RenderPassOutputBackend& backend = arbitgpu::nativeRenderPassOutputBackend());

bool makeColorAovPassPublication (
    const renderpassoutput::Description& description,
    const arbitgpu::RenderPassColorAovClear& clear,
    RenderPassOutputPublicationPtr& publication,
    std::string& diagnostic,
    arbitgpu::RenderPassOutputBackend& backend = arbitgpu::nativeRenderPassOutputBackend());

bool makeMotionAovPassPublication (
    const renderpassoutput::Description& description,
    const arbitgpu::RenderPassMotionAovClear& clear,
    RenderPassOutputPublicationPtr& publication,
    std::string& diagnostic,
    arbitgpu::RenderPassOutputBackend& backend = arbitgpu::nativeRenderPassOutputBackend());

bool makeSceneAovPassPublication (
    const sceneaov::Payload& payload,
    RenderPassOutputPublicationPtr& publication,
    std::string& diagnostic,
    arbitgpu::RenderPassOutputBackend& backend = arbitgpu::nativeRenderPassOutputBackend());

bool makeAovInspectionPassPublication (
    const sceneaov::Payload& scenePayload,
    const aovinspection::Payload& payload,
    RenderPassOutputPublicationPtr& publication,
    std::string& diagnostic,
    arbitgpu::RenderPassOutputBackend& backend = arbitgpu::nativeRenderPassOutputBackend());

// Immutable helper admission shared by preview and export. A nonempty
// publication carries an exact allocated resource for every admitted output.
// The handles remain backend-local and do not claim rendering or readback.
class RenderPassOutputPublication final
{
public:
    RenderPassOutputPublication (const RenderPassOutputPublication&) = delete;
    RenderPassOutputPublication (RenderPassOutputPublication&&) = delete;
    RenderPassOutputPublication& operator= (const RenderPassOutputPublication&) = delete;
    RenderPassOutputPublication& operator= (RenderPassOutputPublication&&) = delete;

    ~RenderPassOutputPublication()
    {
        if (backend_ != nullptr && lifecycle_)
            backend_->releaseRenderPassOutputs (lifecycle_);
    }

    const renderpassoutput::AdmittedOutputs& outputs() const noexcept { return outputs_; }
    const std::vector<arbitgpu::RenderPassOutputResource>& resources() const noexcept
    {
        return resources_;
    }
    const arbitgpu::RenderPassOutputResource* resource (
        renderpassoutput::Output output) const noexcept
    {
        for (const auto& resource : resources_)
            if (resource.output == output)
                return &resource;
        return nullptr;
    }
    bool hasBackendAdmission() const noexcept { return static_cast<bool> (lifecycle_); }
    const arbitgpu::FrameMemoryAdmission& frameMemory() const noexcept
    {
        return frameMemory_;
    }
    const arbitgpu::RenderPassColorAovExecution& colorAovExecution() const noexcept
    {
        return colorAovExecution_;
    }
    const arbitgpu::RenderPassColorAovClear& colorAovClear() const noexcept
    {
        return colorAovClear_;
    }
    const arbitgpu::RenderPassMotionAovExecution& motionAovExecution() const noexcept
    {
        return motionAovExecution_;
    }
    const arbitgpu::RenderPassMotionAovClear& motionAovClear() const noexcept
    {
        return motionAovClear_;
    }
    const arbitgpu::RenderPassSceneAovExecution& sceneAovExecution() const noexcept
    {
        return sceneAovExecution_;
    }
    const sceneaov::Payload& sceneAovPayload() const noexcept
    {
        return sceneAovPayload_;
    }
    const arbitgpu::RenderPassAovInspectionExecution& aovInspectionExecution() const noexcept
    {
        return aovInspectionExecution_;
    }
    const aovinspection::Payload& aovInspectionPayload() const noexcept
    {
        return aovInspectionPayload_;
    }

private:
    friend bool makeRenderPassOutputPublication (
        const renderpassoutput::Description&,
        RenderPassOutputPublicationPtr&,
        std::string&,
        arbitgpu::RenderPassOutputBackend&);
    friend bool makeColorAovPassPublication (
        const renderpassoutput::Description&,
        const arbitgpu::RenderPassColorAovClear&,
        RenderPassOutputPublicationPtr&,
        std::string&,
        arbitgpu::RenderPassOutputBackend&);
    friend bool makeMotionAovPassPublication (
        const renderpassoutput::Description&,
        const arbitgpu::RenderPassMotionAovClear&,
        RenderPassOutputPublicationPtr&,
        std::string&,
        arbitgpu::RenderPassOutputBackend&);
    friend bool makeSceneAovPassPublication (
        const sceneaov::Payload&,
        RenderPassOutputPublicationPtr&,
        std::string&,
        arbitgpu::RenderPassOutputBackend&);
    friend bool makeAovInspectionPassPublication (
        const sceneaov::Payload&,
        const aovinspection::Payload&,
        RenderPassOutputPublicationPtr&,
        std::string&,
        arbitgpu::RenderPassOutputBackend&);

    RenderPassOutputPublication (renderpassoutput::AdmittedOutputs outputs,
                                 std::vector<arbitgpu::RenderPassOutputResource> resources,
                                 arbitgpu::RenderPassOutputBackend* backend,
                                 arbitgpu::RenderPassOutputLifecycleHandle lifecycle,
                                 arbitgpu::FrameMemoryAdmission frameMemory = {})
        : outputs_ (std::move (outputs)), resources_ (std::move (resources)),
          backend_ (backend), lifecycle_ (lifecycle), frameMemory_ (frameMemory)
    {
    }

    const renderpassoutput::AdmittedOutputs outputs_;
    const std::vector<arbitgpu::RenderPassOutputResource> resources_;
    arbitgpu::RenderPassOutputBackend* const backend_;
    const arbitgpu::RenderPassOutputLifecycleHandle lifecycle_;
    const arbitgpu::FrameMemoryAdmission frameMemory_;
    arbitgpu::RenderPassColorAovClear colorAovClear_;
    arbitgpu::RenderPassColorAovExecution colorAovExecution_;
    arbitgpu::RenderPassMotionAovClear motionAovClear_;
    arbitgpu::RenderPassMotionAovExecution motionAovExecution_;
    sceneaov::Payload sceneAovPayload_;
    arbitgpu::RenderPassSceneAovExecution sceneAovExecution_;
    aovinspection::Payload aovInspectionPayload_;
    arbitgpu::RenderPassAovInspectionExecution aovInspectionExecution_;
};

inline bool makeRenderPassOutputPublication (
    const renderpassoutput::Description& description,
    RenderPassOutputPublicationPtr& publication,
    std::string& diagnostic,
    arbitgpu::RenderPassOutputBackend& backend)
{
    diagnostic.clear();
    renderpassoutput::AdmissionFailure failure = renderpassoutput::AdmissionFailure::None;
    auto exact = renderpassoutput::admit (description, failure);
    if (! exact.has_value())
    {
        diagnostic = "render-pass output contract rejected: ";
        diagnostic += renderpassoutput::token (failure);
        return false;
    }

    // A pass may request no optional outputs. It needs no backend admission and
    // remains valid even when the native backend is unavailable.
    if (exact->attachments().empty())
    {
        publication = RenderPassOutputPublicationPtr (
            new RenderPassOutputPublication (std::move (*exact), {}, nullptr, {}));
        return true;
    }

    const auto capabilities = backend.renderPassOutputCapabilities();
    if (! capabilities.available)
    {
        diagnostic = capabilities.error.empty()
            ? "native GPU render-pass outputs are unavailable" : capabilities.error;
        return false;
    }

    auto bounded = renderpassoutput::admit (description, failure, capabilities.limits);
    if (! bounded.has_value())
    {
        diagnostic = "render-pass output backend budget rejected: ";
        diagnostic += renderpassoutput::token (failure);
        return false;
    }

    for (const auto& attachment : bounded->attachments())
        if (! capabilities.supports (attachment.output))
        {
            diagnostic = "native GPU backend does not support render-pass output ";
            diagnostic += renderpassoutput::token (attachment.output);
            return false;
        }

    auto admission = backend.admitRenderPassOutputs (*bounded);
    if (! admission.lifecycle)
    {
        diagnostic = admission.error.empty()
            ? "native GPU backend rejected render-pass outputs" : admission.error;
        return false;
    }
    bool exactResources = admission.resources.size() == bounded->attachments().size();
    for (std::size_t index = 0; exactResources && index < admission.resources.size(); ++index)
    {
        const auto& resource = admission.resources[index];
        exactResources = resource.output == bounded->attachments()[index].output
            && resource.image != 0 && resource.attachmentView != 0
            && resource.textureView != 0;
    }
    if (! admission.error.empty() || ! exactResources)
    {
        backend.releaseRenderPassOutputs (admission.lifecycle);
        diagnostic = ! admission.error.empty()
            ? "native GPU backend returned contradictory render-pass output admission"
            : "native GPU backend returned incomplete render-pass output resources";
        return false;
    }

    publication = RenderPassOutputPublicationPtr (
        new RenderPassOutputPublication (std::move (*bounded), std::move (admission.resources),
                                         &backend, admission.lifecycle, admission.frameMemory));
    return true;
}

// Transactional Color AOV checkpoint shared by viewport and export renderer
// owners. Admission, native clear submission, and publication either all
// succeed or the candidate lifecycle is released and the prior publication is
// retained. This does not claim scene shading or any other AOV semantics.
inline bool makeColorAovPassPublication (
    const renderpassoutput::Description& description,
    const arbitgpu::RenderPassColorAovClear& clear,
    RenderPassOutputPublicationPtr& publication,
    std::string& diagnostic,
    arbitgpu::RenderPassOutputBackend& backend)
{
    diagnostic.clear();
    if (description.attachments.size() != 1
        || description.attachments[0].output != renderpassoutput::Output::Color)
    {
        diagnostic = "native Color AOV pass requires exactly one Color attachment";
        return false;
    }
    if (! std::all_of (clear.linearRgba.begin(), clear.linearRgba.end(),
                       [] (float value) { return std::isfinite (value); }))
    {
        diagnostic = "native Color AOV clear values must be finite";
        return false;
    }

    RenderPassOutputPublicationPtr candidate;
    if (! makeRenderPassOutputPublication (description, candidate, diagnostic, backend))
        return false;

    auto execution = backend.executeColorAovClear (candidate->lifecycle_, clear);
    if (! execution.submitted || execution.submission == 0 || ! execution.error.empty())
    {
        diagnostic = execution.error.empty()
            ? "native GPU backend returned contradictory Color AOV execution"
            : execution.error;
        return false;
    }

    auto mutableCandidate = std::const_pointer_cast<RenderPassOutputPublication> (candidate);
    mutableCandidate->colorAovClear_ = clear;
    mutableCandidate->colorAovExecution_ = std::move (execution);
    publication = std::move (candidate);
    return true;
}

// Transactional Motion AOV checkpoint. The candidate attachment is published
// only after a real native clear/store submission. The product's fixed zero
// displacement field is an initialized/no-prior-sample signal; this checkpoint
// deliberately does not claim scene-derived optical flow.
inline bool makeMotionAovPassPublication (
    const renderpassoutput::Description& description,
    const arbitgpu::RenderPassMotionAovClear& clear,
    RenderPassOutputPublicationPtr& publication,
    std::string& diagnostic,
    arbitgpu::RenderPassOutputBackend& backend)
{
    diagnostic.clear();
    if (description.attachments.size() != 1
        || description.attachments[0].output != renderpassoutput::Output::Motion)
    {
        diagnostic = "native Motion AOV pass requires exactly one Motion attachment";
        return false;
    }
    if (! std::all_of (clear.pixelDisplacement.begin(), clear.pixelDisplacement.end(),
                       [] (float value) { return std::isfinite (value); }))
    {
        diagnostic = "native Motion AOV clear values must be finite";
        return false;
    }

    RenderPassOutputPublicationPtr candidate;
    if (! makeRenderPassOutputPublication (description, candidate, diagnostic, backend))
        return false;

    auto execution = backend.executeMotionAovClear (candidate->lifecycle_, clear);
    if (! execution.submitted || execution.submission == 0 || ! execution.error.empty())
    {
        diagnostic = execution.error.empty()
            ? "native GPU backend returned contradictory Motion AOV execution"
            : execution.error;
        return false;
    }

    auto mutableCandidate = std::const_pointer_cast<RenderPassOutputPublication> (candidate);
    mutableCandidate->motionAovClear_ = clear;
    mutableCandidate->motionAovExecution_ = std::move (execution);
    publication = std::move (candidate);
    return true;
}

inline bool makeSceneAovPassPublication (
    const sceneaov::Payload& payload,
    RenderPassOutputPublicationPtr& publication,
    std::string& diagnostic,
    arbitgpu::RenderPassOutputBackend& backend)
{
    diagnostic.clear();
    if (! sceneaov::valid (payload))
    {
        diagnostic = "native scene AOV payload is malformed";
        return false;
    }

    renderpassoutput::Description description;
    description.extent = payload.extent;
    const auto requirements = renderpassoutput::requirements (payload.output);
    description.attachments.push_back (
        { payload.output, requirements.format, requirements.colorSpace, payload.extent });

    RenderPassOutputPublicationPtr candidate;
    if (! makeRenderPassOutputPublication (description, candidate, diagnostic, backend))
        return false;

    auto execution = backend.executeSceneAov (candidate->lifecycle_, payload);
    if (! execution.submitted || execution.submission == 0 || ! execution.error.empty())
    {
        diagnostic = execution.error.empty()
            ? "native GPU backend returned contradictory scene AOV execution"
            : execution.error;
        return false;
    }

    auto mutableCandidate = std::const_pointer_cast<RenderPassOutputPublication> (candidate);
    mutableCandidate->sceneAovPayload_ = payload;
    mutableCandidate->sceneAovExecution_ = std::move (execution);
    publication = std::move (candidate);
    return true;
}

// Allocates one processor-owned source AOV and one displayable Color target,
// rasterizes the exact immutable scene into the source, then publishes only
// after the backend submits the native mapping pass. Both passes use the same
// lifecycle. No CPU pixels, external texture aliases, constant substitutes, or
// synthetic resource receipts enter here.
inline bool makeAovInspectionPassPublication (
    const sceneaov::Payload& scenePayload,
    const aovinspection::Payload& payload,
    RenderPassOutputPublicationPtr& publication,
    std::string& diagnostic,
    arbitgpu::RenderPassOutputBackend& backend)
{
    diagnostic.clear();
    const auto sceneSerialized = sceneaov::serialize (scenePayload);
    sceneaov::Payload canonicalScene;
    const auto serialized = aovinspection::serialize (payload);
    aovinspection::Payload canonical;
    if (sceneSerialized.empty() || ! sceneaov::parse (sceneSerialized, canonicalScene)
        || sceneaov::serialize (canonicalScene) != sceneSerialized
        || serialized.empty() || ! aovinspection::parse (serialized, canonical)
        || aovinspection::serialize (canonical) != serialized)
    {
        diagnostic = "native AOV inspection payload is malformed";
        return false;
    }
    if (canonicalScene.output != aovinspection::output (canonical.source)
        || canonicalScene.extent != canonical.extent)
    {
        diagnostic = "native AOV inspection scene and display payloads disagree";
        return false;
    }

    RenderPassOutputPublicationPtr candidate;
    if (! makeRenderPassOutputPublication (
            aovinspection::description (canonical), candidate, diagnostic, backend))
        return false;

    if (candidate->resource (renderpassoutput::Output::Color) == nullptr
        || candidate->resource (aovinspection::output (canonical.source)) == nullptr)
    {
        diagnostic = "native AOV inspection requires exact Color and source resources";
        return false;
    }

    auto sceneExecution = backend.executeSceneAov (candidate->lifecycle_, canonicalScene);
    if (! sceneExecution.submitted || sceneExecution.submission == 0
        || ! sceneExecution.error.empty())
    {
        diagnostic = sceneExecution.error.empty()
            ? "native GPU backend returned contradictory scene AOV execution"
            : sceneExecution.error;
        return false;
    }

    auto execution = backend.executeAovInspection (candidate->lifecycle_, canonical);
    if (! execution.submitted || execution.submission == 0 || ! execution.error.empty())
    {
        diagnostic = execution.error.empty()
            ? "native GPU backend returned contradictory AOV inspection execution"
            : execution.error;
        return false;
    }

    auto mutableCandidate = std::const_pointer_cast<RenderPassOutputPublication> (candidate);
    mutableCandidate->sceneAovPayload_ = std::move (canonicalScene);
    mutableCandidate->sceneAovExecution_ = std::move (sceneExecution);
    mutableCandidate->aovInspectionPayload_ = canonical;
    mutableCandidate->aovInspectionExecution_ = std::move (execution);
    publication = std::move (candidate);
    return true;
}
} // namespace videowire
