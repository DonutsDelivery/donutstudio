#include "sdf_native_renderer.h"
#include "sdf_native_program.h"

#include <algorithm>
#include <utility>

namespace videohelper::sdf
{
NativeSdfRenderer::NativeSdfRenderer (
    arbitgpu::NativeSdfExecutionBackend& backend) noexcept
    : backend_ (backend)
{
}

bool NativeSdfRenderer::renderPreview (
    const std::shared_ptr<const AdmittedSdfIr>& geometry,
    NativeSdfRenderDimensions dimensions,
    NativeSdfRenderControls controls,
    std::string_view backendCapability,
    NativeSdfRenderedFrame& output,
    std::string& error, NativeSdfCacheIdentity identity)
{
    return render (geometry, dimensions, controls, backendCapability,
                   NativeSdfRenderUse::Preview, output, error, identity);
}

bool NativeSdfRenderer::renderExport (
    const std::shared_ptr<const AdmittedSdfIr>& geometry,
    NativeSdfRenderDimensions dimensions,
    NativeSdfRenderControls controls,
    std::string_view backendCapability,
    NativeSdfRenderedFrame& output,
    std::string& error, NativeSdfCacheIdentity identity)
{
    return render (geometry, dimensions, controls, backendCapability,
                   NativeSdfRenderUse::Export, output, error, identity);
}

bool NativeSdfRenderer::render (
    const std::shared_ptr<const AdmittedSdfIr>& geometry,
    NativeSdfRenderDimensions dimensions,
    NativeSdfRenderControls controls,
    std::string_view backendCapability,
    NativeSdfRenderUse use,
    NativeSdfRenderedFrame& output,
    std::string& error, NativeSdfCacheIdentity identity)
{
    if (geometry == nullptr)
    {
        error = "native GPU SDF rendering requires immutable admitted geometry";
        return false;
    }
    if (backendCapability != kNativeGpuCapability)
    {
        error = "native SDF rendering requires backendCapability native-gpu";
        return false;
    }

    arbitgpu::NativeSdfDrawRequest request;
    request.geometry.schemaVersion = geometry->schemaVersion();
    request.geometry.rootId = geometry->rootId();
    request.geometry.records = geometry->records();
    {
        std::lock_guard<std::mutex> lock (compiledGeometryMutex_);
        const auto cacheKey = std::to_string (identity.projectGeneration) + ":"
            + std::to_string (identity.clipId) + ":" + std::to_string (identity.planRevision)
            + ":" + std::to_string (identity.helperGeneration) + ":" + geometry->structuralDigest();
        for (auto entry = compiledGeometry_.begin(); entry != compiledGeometry_.end();)
            if (entry->second.identity.projectGeneration == identity.projectGeneration
                && entry->second.identity.clipId == identity.clipId
                && (entry->second.identity.planRevision != identity.planRevision
                    || entry->second.identity.helperGeneration != identity.helperGeneration))
                entry = compiledGeometry_.erase (entry);
            else ++entry;
        auto& cached = compiledGeometry_[cacheKey];
        if (cached.program == nullptr)
        {
            cached.program = compileNativeSdfProgram (request.geometry, error);
            if (cached.program == nullptr)
            {
                compiledGeometry_.erase (cacheKey);
                return false;
            }
            cached.geometry = geometry;
            cached.identity = identity;
        }
        cached.lastUse = ++cacheUse_;
        request.compiledProgram = cached.program;
        if (compiledGeometry_.size() > kMaximumCompiledGeometryEntries)
        {
            const auto oldest = std::min_element (compiledGeometry_.begin(), compiledGeometry_.end(),
                [] (const auto& left, const auto& right)
                { return left.second.lastUse < right.second.lastUse; });
            compiledGeometry_.erase (oldest);
        }
        std::uint64_t cacheBytes = 0;
        std::uint64_t cacheRecords = 0;
        for (const auto& entry : compiledGeometry_)
        {
            cacheRecords += entry.second.program->records().size();
            cacheBytes += entry.second.program->records().size()
                * sizeof (arbitgpu::NativeSdfCompiledRecord);
        }
        request.geometryCacheBytes = cacheBytes;
        request.geometryCacheRecords = static_cast<std::uint32_t> (cacheRecords);
    }

    const auto capabilities = backend_.capabilities();
    auto admitted = admitNativeSdfRender (
        *geometry, dimensions, controls, capabilities, error);
    if (! admitted)
        return false;

    request.width = dimensions.width;
    request.height = dimensions.height;
    request.maximumSteps = controls.maximumSteps;
    request.epsilon = controls.epsilon;
    request.maximumDistance = controls.maximumDistance;
    request.adaptiveQuality = controls.adaptiveQuality;
    request.normalQuality = controls.normalQuality;
    request.shadowQuality = controls.shadowQuality;
    request.output = controls.output;

    auto submission = backend_.render (request);
    if (! submission.rendered || submission.frame == nullptr)
    {
        error = submission.error.empty()
            ? "native GPU SDF backend did not return a rendered frame"
            : std::move (submission.error);
        return false;
    }
    if (submission.frame->backend() != admitted->backend()
        || submission.frame->width() != dimensions.width
        || submission.frame->height() != dimensions.height
        || submission.frame->colorImageHandle() == 0
        || submission.frame->colorTextureViewHandle() == 0)
    {
        error = "native GPU SDF backend returned an incompatible frame";
        return false;
    }

    NativeSdfRenderedFrame rendered;
    rendered.use = use;
    rendered.structuralDigest = admitted->geometry().structuralDigest();
    rendered.dimensions = dimensions;
    rendered.controls = controls;
    rendered.nativeFrame = std::move (submission.frame);
    output = std::move (rendered);
    error.clear();
    return true;
}

void NativeSdfRenderer::invalidateCompiledGeometry() noexcept
{
    std::lock_guard<std::mutex> lock (compiledGeometryMutex_);
    compiledGeometry_.clear();
    cacheUse_ = 0;
}

std::size_t NativeSdfRenderer::compiledGeometryCount() const noexcept
{
    std::lock_guard<std::mutex> lock (compiledGeometryMutex_);
    return compiledGeometry_.size();
}

NativeSdfRenderer& nativeSdfRenderer()
{
    static NativeSdfRenderer renderer (arbitgpu::nativeSdfExecutionBackend());
    return renderer;
}
} // namespace videohelper::sdf
