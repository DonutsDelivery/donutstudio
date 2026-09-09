#include "native_animation_deformation_renderer.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace videorender::animation3d
{
NativeAnimationDeformationRenderer::NativeAnimationDeformationRenderer (
    arbitgpu::NativeDeformationBackend& backend) noexcept
    : backend_ (backend)
{
}

bool NativeAnimationDeformationRenderer::renderPreview (
    const std::shared_ptr<const arbitgpu::NativeDeformationScene>& source,
    const std::shared_ptr<const visualdeformation::AnimationDeformationSnapshot>& deformation,
    std::uint32_t width, std::uint32_t height,
    std::string_view backendCapability,
    RenderedDeformationFrame& output,
    std::string& error)
{
    return render (RenderUse::Preview, source, deformation, {}, {}, width, height,
                   backendCapability, output, error);
}

bool NativeAnimationDeformationRenderer::renderPreview (
    const std::shared_ptr<const arbitgpu::NativeDeformationScene>& source,
    const std::shared_ptr<const visualdeformation::AnimationDeformationSnapshot>& deformation,
    const std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram>& materialProgram,
    arbitgpu::NativeDeformationRuntimeInputs runtimeInputs,
    std::uint32_t width, std::uint32_t height,
    std::string_view backendCapability,
    RenderedDeformationFrame& output,
    std::string& error)
{
    return render (RenderUse::Preview, source, deformation, materialProgram,
                   runtimeInputs, width, height, backendCapability, output, error);
}

bool NativeAnimationDeformationRenderer::renderExport (
    const std::shared_ptr<const arbitgpu::NativeDeformationScene>& source,
    const std::shared_ptr<const visualdeformation::AnimationDeformationSnapshot>& deformation,
    std::uint32_t width, std::uint32_t height,
    std::string_view backendCapability,
    RenderedDeformationFrame& output,
    std::string& error)
{
    return render (RenderUse::Export, source, deformation, {}, {}, width, height,
                   backendCapability, output, error);
}

bool NativeAnimationDeformationRenderer::renderExport (
    const std::shared_ptr<const arbitgpu::NativeDeformationScene>& source,
    const std::shared_ptr<const visualdeformation::AnimationDeformationSnapshot>& deformation,
    const std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram>& materialProgram,
    arbitgpu::NativeDeformationRuntimeInputs runtimeInputs,
    std::uint32_t width, std::uint32_t height,
    std::string_view backendCapability,
    RenderedDeformationFrame& output,
    std::string& error)
{
    return render (RenderUse::Export, source, deformation, materialProgram,
                   runtimeInputs, width, height, backendCapability, output, error);
}

bool NativeAnimationDeformationRenderer::render (
    RenderUse use,
    const std::shared_ptr<const arbitgpu::NativeDeformationScene>& source,
    const std::shared_ptr<const visualdeformation::AnimationDeformationSnapshot>& deformation,
    const std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram>& materialProgram,
    arbitgpu::NativeDeformationRuntimeInputs runtimeInputs,
    std::uint32_t width, std::uint32_t height,
    std::string_view backendCapability,
    RenderedDeformationFrame& output,
    std::string& error)
{
    error.clear();
    if (! source || ! deformation)
    {
        error = "native animation deformation requires immutable source and frame snapshots";
        return false;
    }
    if (backendCapability != kNativeGpuCapability
        || HarmonicMIDI::grid::Visual3DSceneRenderContract::kAllowsCpuImageFallback)
    {
        error = "native animation deformation requires backendCapability native-gpu without CPU fallback";
        return false;
    }
    if (width == 0 || height == 0 || width > kMaxExtent || height > kMaxExtent
        || static_cast<std::uint64_t> (width) * height > kMaxPixels)
    {
        error = "native animation deformation dimensions exceed the bounded extent";
        return false;
    }

    arbitgpu::NativeDeformationFrameData validated;
    if (! arbitgpu::prepareNativeDeformationFrame (*source, *deformation,
                                                    validated, error))
        return false;

    const auto info = backend_.info();
    if (! info.available || ! info.compute || info.backend.empty())
    {
        error = info.error.empty()
            ? "native animation deformation GPU backend is unavailable" : info.error;
        return false;
    }
    if (! std::isfinite (runtimeInputs.morphWeight)
        || runtimeInputs.morphWeight < 0.0f || runtimeInputs.morphWeight > 1.0f)
    {
        error = "native deformation morph weight must be finite and in [0,1]";
        return false;
    }
    if (! arbitgpu::validFixtureRuntimeInputs(runtimeInputs))
    {
        error = "native deformation scene modulation is non-finite or out of bounds";
        return false;
    }
    if (materialProgram != nullptr)
    {
        const bool backendMatches
            = (materialProgram->backend == arbitgpu::NativeFixtureMaterialBackend::OpenGl
               && info.backend == "opengl")
            || (materialProgram->backend == arbitgpu::NativeFixtureMaterialBackend::Metal
                && info.backend == "metal");
        if (! backendMatches)
        {
            error = "native deformation surface material does not match the physical backend";
            return false;
        }
        if (! std::isfinite (runtimeInputs.timeSeconds)
            || std::abs (runtimeInputs.timeSeconds)
                > surfacematerial::kMaximumEvaluationMagnitude)
        {
            error = "native deformation surface material time is non-finite or out of bounds";
            return false;
        }
    }

    const auto exactCached = cachedSource_.lock();
    const bool cacheHit = exactCached == source && cachedResources_ != nullptr
                       && cachedMaterialProgram_ == materialProgram;
    if (! cacheHit)
    {
        auto prepared = backend_.prepare (source, materialProgram);
        if (! prepared.prepared || ! prepared.resources)
        {
            error = prepared.error.empty()
                ? "native animation deformation backend did not prepare static resources"
                : std::move (prepared.error);
            return false;
        }
        if (prepared.resources->backend() != info.backend)
        {
            error = "native animation deformation backend returned incompatible static resources";
            return false;
        }
        cachedSource_ = source;
        cachedMaterialProgram_ = materialProgram;
        cachedResources_ = std::move (prepared.resources);
        cachedFootprint_ = prepared.stats;
    }

    auto submitted = backend_.render (source, deformation, cachedResources_, width, height,
                                      runtimeInputs);
    if (! submitted.rendered || ! submitted.frame)
    {
        error = submitted.error.empty()
            ? "native animation deformation backend did not render a native frame"
            : std::move (submitted.error);
        return false;
    }
    if (submitted.frame->backend() != info.backend
        || submitted.frame->width() != width || submitted.frame->height() != height
        || submitted.frame->colorImageHandle() == 0
        || submitted.frame->colorTextureViewHandle() == 0
        || submitted.frame->depthImageHandle() == 0
        || submitted.frame->depthTextureViewHandle() == 0)
    {
        error = "native animation deformation backend returned an incompatible frame";
        return false;
    }

    RenderedDeformationFrame result;
    result.use = use;
    result.source = source;
    result.deformation = deformation;
    result.nativeFrame = std::move (submitted.frame);
    result.stats = submitted.stats;
    result.stats.staticVertexBytes = cachedFootprint_.staticVertexBytes;
    result.stats.staticDeformationBytes = cachedFootprint_.staticDeformationBytes;
    result.stats.staticUploadCount = cacheHit ? 0u : 1u;
    result.stats.reusedStaticResources = cacheHit;
    output = std::move (result);
    return true;
}
} // namespace videorender::animation3d
