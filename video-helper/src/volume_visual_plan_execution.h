#pragma once

#include "render_snapshot.h"
#include "volume_visual_cache.h"

namespace videohelper::volume
{
enum class VisualVolumePreparation { notPresent, rendered, rejected };

inline bool hasVisualVolume(const std::vector<videowire::CompiledVisualLayerPlan>& plans, int clipId)
{
    const auto* plan = videowire::findVisualLayerPlan(plans, clipId);
    return plan != nullptr && visualvolume::isVolumePlan(*plan);
}

template <typename Layer>
inline VisualVolumePreparation prepareVisualVolumeLayer(
    const std::vector<videowire::CompiledVisualLayerPlan>& plans, int clipId,
    int width, int height, NativeVolumeRenderUse use,
    VisualVolumeRenderCache& cache, Layer& layer, std::string& error,
    NativeVolumeExecutionBackend& backend = nativeVolumeExecutionBackend())
{
    const auto* plan = videowire::findVisualLayerPlan(plans, clipId);
    if (plan == nullptr || !visualvolume::isVolumePlan(*plan)) return VisualVolumePreparation::notPresent;
    visualvolume::Operation operation;
    if (width <= 0 || height <= 0 || !visualvolume::validatePlan(*plan, operation, error))
    {
        if (error.empty()) error = "Volume rendering requires a positive output extent";
        return VisualVolumePreparation::rejected;
    }
    NativeVolumeRenderedFrame frame;
    if (!cache.render(visualvolume::encode(operation), static_cast<std::uint32_t>(width),
            static_cast<std::uint32_t>(height), use, backend, frame, error))
        return VisualVolumePreparation::rejected;
    // LayerDesc owns this lease through every compositor use, including when a
    // Layer Source references the image or another layer causes cache eviction.
    layer.nativeTextureOwner = frame.nativeFrame;
    layer.nativeTextureDescriptor = frame.nativeFrame->colorTextureDescriptor();
    layer.nativeTextureBackend = frame.nativeFrame->backend();
    layer.nativeTextureView = frame.nativeFrame->colorTextureViewHandle();
    layer.texture = frame.nativeFrame->backend() == "opengl"
        ? static_cast<unsigned>(frame.nativeFrame->colorImageHandle()) : 0;
    layer.texWidth = width;
    layer.texHeight = height;
    layer.shaderSource = layer.particleSource = layer.scoreSource = layer.isAdjustment = false;
    layer.visualPlanStructuralRevision = plan->structuralRevision;
    error.clear();
    return VisualVolumePreparation::rendered;
}
} // namespace videohelper::volume
