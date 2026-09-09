#pragma once

#include "imported_animated_scene_payload_execution.h"
#include "visual_plan_executor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace videohelper::importedanimation
{
enum class VisualImportedAnimationPreparation
{
    notPresent,
    rendered,
    rejected
};

enum class NativeImportedAnimationRenderUse : std::uint8_t
{
    Preview = 0,
    Export = 1
};

inline bool importedAnimationFrameIdentity(double sourceTimeSec,
                                           double frameRate,
                                           visualdeformation::RationalFrameTime& frame,
                                           std::string& error)
{
    constexpr std::uint64_t scale = 1000000u;
    if (! std::isfinite(sourceTimeSec) || ! std::isfinite(frameRate)
        || frameRate <= 0.0 || frameRate > 1000.0)
    {
        error = "imported animation compositor frame clock is invalid";
        return false;
    }
    const auto scaledRate = frameRate * static_cast<double>(scale);
    if (scaledRate < 1.0
        || scaledRate > static_cast<double>(std::numeric_limits<std::uint32_t>::max()))
    {
        error = "imported animation compositor frame clock exceeds its rational bound";
        return false;
    }
    const auto numerator64 = static_cast<std::uint64_t>(std::llround(scaledRate));
    const auto divisor = std::gcd(numerator64, scale);
    const auto exactFrame = sourceTimeSec * frameRate;
    if (! std::isfinite(exactFrame)
        || exactFrame < static_cast<double>(std::numeric_limits<std::int64_t>::min())
        || exactFrame > static_cast<double>(std::numeric_limits<std::int64_t>::max()))
    {
        error = "imported animation compositor frame index exceeds its signed bound";
        return false;
    }
    frame.frame = static_cast<std::int64_t>(std::llround(exactFrame));
    frame.rateNumerator = static_cast<std::uint32_t>(numerator64 / divisor);
    frame.rateDenominator = static_cast<std::uint32_t>(scale / divisor);
    error.clear();
    return true;
}

template <typename Execution, typename Layer>
inline VisualImportedAnimationPreparation prepareVisualImportedAnimationLayer(
    const std::vector<videowire::CompiledVisualLayerPlan>& plans,
    int clipId,
    int width,
    int height,
    double sourceTimeSec,
    double frameRate,
    NativeImportedAnimationRenderUse use,
    Execution* execution,
    Layer& layer,
    std::vector<typename Execution::Receipt>& frameOwners,
    std::string& error)
{
    const auto* plan = videowire::findVisualLayerPlan(plans, clipId);
    if (plan == nullptr) return VisualImportedAnimationPreparation::notPresent;

    const auto importedOperation = std::find_if(plan->operations.begin(), plan->operations.end(),
        [](const auto& operation)
        {
            return operation.kind == visualanimationoperation::kSourceNodeKind
                || operation.kind == visualanimationoperation::kDeformationNodeKind;
        });
    if (importedOperation == plan->operations.end())
        return VisualImportedAnimationPreparation::notPresent;

    const auto reject = [&](std::string message)
    {
        error = std::move(message);
        return VisualImportedAnimationPreparation::rejected;
    };

    videowire::VisualLayerExecution lowered;
    if (! videowire::compileVisualLayerExecution(*plan, lowered, error)
        || ! lowered.importedAnimation.has_value())
    {
        if (error.empty())
            error = "imported animation compositor operation failed strict graph lowering";
        return VisualImportedAnimationPreparation::rejected;
    }
    if (execution == nullptr)
        return reject("imported animation compositor product context is unavailable");
    if (width <= 0 || height <= 0
        || static_cast<std::uint64_t>(width) > std::numeric_limits<std::uint32_t>::max()
        || static_cast<std::uint64_t>(height) > std::numeric_limits<std::uint32_t>::max())
        return reject("imported animation compositor dimensions exceed the product bound");

    visualdeformation::RationalFrameTime frame;
    if (! importedAnimationFrameIdentity(sourceTimeSec, frameRate, frame, error))
        return VisualImportedAnimationPreparation::rejected;

    videohelper::modelpayload::ImportedAnimatedSceneRequest request;
    request.operation = *lowered.importedAnimation;
    request.frame = frame;
    request.structuralRevision = plan->structuralRevision;
    request.width = static_cast<std::uint32_t>(width);
    request.height = static_cast<std::uint32_t>(height);

    typename Execution::Receipt receipt;
    const bool executed = use == NativeImportedAnimationRenderUse::Preview
        ? execution->executePreview(request, receipt, error)
        : execution->executeExport(request, receipt, error);
    if (! executed)
    {
        if (error.empty())
            error = "imported animation compositor native execution was rejected";
        return VisualImportedAnimationPreparation::rejected;
    }
    if (! execution->validReceipt(receipt))
        return reject("imported animation compositor receipt is incomplete");

    const auto& nativeFrame = execution->nativeFrame(receipt);
    if (! nativeFrame)
        return reject("imported animation compositor native frame is missing");
    const auto backend = nativeFrame->backend();
    const auto nativeView = nativeFrame->colorTextureViewHandle();
    if (backend != "opengl" && backend != "metal")
        return reject("imported animation compositor native frame backend is unsupported");
    if (nativeFrame->width() != request.width || nativeFrame->height() != request.height)
        return reject("imported animation compositor native frame dimensions are incompatible");
    if (nativeView == 0)
        return reject("imported animation compositor native texture view is missing");

    unsigned openGlTexture = 0;
    if (backend == "opengl")
    {
        if (nativeView > std::numeric_limits<unsigned>::max())
            return reject("imported animation compositor OpenGL texture view exceeds compositor width");
        openGlTexture = static_cast<unsigned>(nativeView);
    }

    // Retain the complete receipt before publishing its borrowed native view into
    // the ordinary layer. The frame owner vector outlives renderer consumption.
    frameOwners.push_back(std::move(receipt));
    layer.texture = openGlTexture;
    layer.nativeTextureBackend = backend;
    layer.nativeTextureView = nativeView;
    layer.texWidth = width;
    layer.texHeight = height;
    error.clear();
    return VisualImportedAnimationPreparation::rendered;
}
} // namespace videohelper::importedanimation
