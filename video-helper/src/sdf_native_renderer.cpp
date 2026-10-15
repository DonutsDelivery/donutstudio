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
    output = {};

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
    for (const auto& outline : geometry->textOutlines())
        request.geometry.textOutlines.push_back (outline.outline());
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
            cached.program = compileNativeSdfProgram (request.geometry, error, true);
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
            cacheBytes += entry.second.program->storageBytes();
            // The cache also retains exact double-precision source outlines.
            cacheBytes += entry.second.geometry->textOutlines().capacity()
                * sizeof (videowire::AdmittedSdfTextOutline);
            for (const auto& admittedOutline : entry.second.geometry->textOutlines())
            {
                const auto& outline = admittedOutline.outline();
                cacheBytes += outline.glyphs.capacity() * sizeof (videowire::SdfTextGlyph)
                    + outline.contours.capacity() * sizeof (videowire::SdfTextContour)
                    + outline.segments.capacity() * sizeof (videowire::SdfTextSegment)
                    + outline.sourceText.capacity() + outline.sourceTextSha256.capacity()
                    + outline.embeddedFontToken.capacity() + outline.embeddedFontSha256.capacity() + 4
                    + admittedOutline.identitySha256().capacity() + 1;
            }
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
    request.normalInspection = controls.normalInspection;

    if (capabilities.reusesPrimaryTrace)
    {
        if (! admitNativeSdfStagedTextWork (request, request.textWork, error)) return false;
    }
    else if (! admitNativeSdfTextWork (request, request.textWork, error)) return false;

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
    const auto descriptor = submission.frame->colorTextureDescriptor();
    if (!arbitgpu::isLinearSceneColor(descriptor)
        || descriptor.backend != submission.frame->backend()
        || descriptor.imageHandle != submission.frame->colorImageHandle()
        || descriptor.textureViewHandle != submission.frame->colorTextureViewHandle()
        || descriptor.width != dimensions.width || descriptor.height != dimensions.height)
    {
        error = "native GPU SDF backend returned an incompatible texture descriptor";
        return false;
    }
    const auto depth = submission.frame->depthTextureDescriptor();
    if (! depth.complete() || depth.format != arbitgpu::NativeTexturePixelFormat::R32Float
        || depth.backend != descriptor.backend
        || depth.textureViewHandle != submission.frame->depthTextureViewHandle()
        || depth.width != dimensions.width || depth.height != dimensions.height
        || depth.deviceOrContextIdentity != descriptor.deviceOrContextIdentity
        || depth.rendererGeneration != descriptor.rendererGeneration
        || depth.rowOrder != descriptor.rowOrder)
    {
        error = "native GPU SDF backend returned an incompatible raw Depth attachment";
        return false;
    }
    const auto normal = submission.frame->normalTextureDescriptor();
    if (! normal.complete() || normal.format != arbitgpu::NativeTexturePixelFormat::Rgba16Float
        || normal.backend != descriptor.backend
        || normal.textureViewHandle != submission.frame->normalTextureViewHandle()
        || normal.width != dimensions.width || normal.height != dimensions.height
        || normal.deviceOrContextIdentity != descriptor.deviceOrContextIdentity
        || normal.rendererGeneration != descriptor.rendererGeneration
        || normal.rowOrder != descriptor.rowOrder)
    {
        error = "native GPU SDF backend returned an incompatible raw Normal attachment";
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
