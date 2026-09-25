#pragma once

#include "gpu_backend/backend.h"
#include "../../shared/VisualAnimationDeformationEvaluation.h"

#include <functional>

namespace videohelper::importedscene
{
enum class NativeImportedSceneRenderUse : std::uint8_t
{
    Preview = 0,
    Export = 1
};

// The flattened Frame endpoint and timeline sample identify an evaluation.
// Receipts reject stale results after seeks, republishing and reopening.
struct MaterialFrameEvaluation final
{
    surfacematerialbinding::TextureSlotBinding::GraphFrameEndpoint endpoint;
    visualdeformation::RationalFrameTime frame;
    int clipId = 0;
    std::uint64_t structuralRevision = 0, evaluationRevision = 0;
    std::uint64_t projectGeneration = 0, helperGeneration = 0;
    NativeImportedSceneRenderUse use = NativeImportedSceneRenderUse::Preview;

    bool operator==(const MaterialFrameEvaluation& other) const noexcept
    {
        return endpoint.node == other.endpoint.node && endpoint.port == other.endpoint.port
            && frame == other.frame && clipId == other.clipId
            && structuralRevision == other.structuralRevision
            && evaluationRevision == other.evaluationRevision
            && projectGeneration == other.projectGeneration
            && helperGeneration == other.helperGeneration && use == other.use;
    }
};

struct MaterialFrameReceipt final
{
    MaterialFrameEvaluation evaluation;
    std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> nativeFrame;
};

using MaterialFrameResolver = std::function<bool(
    const MaterialFrameEvaluation&, MaterialFrameReceipt&, std::string&)>;

inline bool resolveMaterialFrame(const MaterialFrameEvaluation& evaluation,
    const MaterialFrameResolver& resolver, MaterialFrameReceipt& receipt, std::string& error)
{
    receipt = {};
    if (!resolver || !resolver(evaluation, receipt, error))
    {
        if (error.empty()) error = "Graph Frame evaluation requires an available owned source";
        return false;
    }
    if (!(receipt.evaluation == evaluation))
    { error = "Graph Frame texture receipt belongs to another endpoint or evaluation"; return false; }
    if (!arbitgpu::validMaterialFrameTexture(receipt.nativeFrame))
    { error = "Graph Frame texture receipt has no complete owned native image"; return false; }
    return true;
}
} // namespace videohelper::importedscene
